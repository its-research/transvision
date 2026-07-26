# Resilient V2X 数据、检测器与训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 DAIR-V2X-C 原始 cooperative records 生成不可变时序 manifest 与 overlays，构建四路共享 encoder、薄检测器、TransFusion head、frozen teacher/student 训练路径、八个等形状变体和 capacity-matched concat，并通过 synthetic one-batch loss/predict。

**Architecture:** 数据阶段先解析 metadata 和 causal selection，再只加载被选中的 raw slices；模型阶段以共享 LiDAR/Camera encoder 生成统一 `256x288x288` BEV，调用核心计划交付的 repair/routing API，最后进入 TransFusion。训练期 wrapper 持有 frozen teacher，部署配置只构建 student。

**Tech Stack:** Python 3.10、PyTorch 2.0.1、torchvision 0.15.2、MMEngine 0.10.7、MMCV 2.1.0、MMDetection 3.2.0、MMDetection3D 1.3.0、zstandard 0.22.0、pytest 7.4.4。

## Global Constraints

- 开始前必须完整通过核心计划 Completion Gate。
- manifest、transport overlay、fault overlay 是三类独立 immutable artifact；动态 schedule 禁止写回 base manifest。
- base split 必须验证 `data/split_datas/cooperative-split-data.json` 的 SHA-256 为 `d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c`。
- student forward 只能接收 `student_inputs`；oracle teacher 数据只能位于独立 `teacher_inputs`，两者不得共享可穿透字典。
- 所有 Resilient config 必须显式列出 constructor fields，golden snapshot 用 resolved dict 检查。
- 当前 Mac 没有 DAIR 数据和 CUDA 时，只运行 synthetic/CPU/config tests；CUDA/data 验收必须记录 `not executed`。
- 每个任务只提交列出的文件，所有命令从 `/Users/libin/transvision` 执行。

---

### Task 1: 定义时序 manifest schema、canonical hash 和 release inventory

**Files:**

- Create: `transvision/dataset/resilient_v2x_manifest.py`
- Create: `tests/resilient_v2x/test_manifest.py`
- Modify: `transvision/dataset/__init__.py`

**Interfaces:**

```python
class ManifestError(RuntimeError):
    """Temporal manifest structure, containment, or hash validation failed."""

MANIFEST_SCHEMA_VERSION = 1
OFFICIAL_COOPERATIVE_SPLIT_SHA256 = (
    "d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
)

@dataclass(frozen=True)
class RawSliceRecord:
    agent: Literal["ego", "rsu"]
    modality: Literal["lidar", "camera"]
    n_s: int
    tau_s_ms: int
    capture_timestamp_us: int
    frame_id: str
    packet_id: str
    relative_path: str
    world_from_agent: tuple[tuple[float, ...], ...]
    agent_from_sensor: tuple[tuple[float, ...], ...]
    calibration_relative_path: str
    calibration_sha256: str
    camera_intrinsic: tuple[tuple[float, ...], ...] | None
    payload_valid: bool
    pose_valid: bool
    calibration_valid: bool

@dataclass(frozen=True)
class GroundTruthBoxRecord:
    class_name: Literal["Car"]
    x: float
    y: float
    z_bottom: float
    length: float
    width: float
    height: float
    yaw: float
    source_annotation_index: int

@dataclass(frozen=True)
class TemporalSampleRecord:
    sample_id: str
    sequence_id: str
    split: Literal["train", "val", "test"]
    n_t: int
    tau_t_ms: int
    source_slices: tuple[RawSliceRecord, ...]
    annotation_path: str
    annotation_sha256: str
    ground_truth: tuple[GroundTruthBoxRecord, ...]

@dataclass(frozen=True)
class ReleaseInventoryEntry:
    relative_path: str
    size: int
    sha256: str

@dataclass(frozen=True)
class PreparedArtifactRecord:
    source_relative_path: str
    prepared_relative_path: str
    point_count: int
    size: int
    sha256: str
    dtype: Literal["<f4"]
    fields: tuple[
        Literal["x"],
        Literal["y"],
        Literal["z"],
        Literal["intensity"],
    ]

@dataclass(frozen=True)
class TemporalManifest:
    schema_version: int
    protocol_scope: Literal["controlled", "fixture"]
    delta_t_ms: int
    history_limit: int
    interval_min_ms: int
    interval_max_ms: int
    max_capture_skew_ms: int
    split_sha256: str
    dataset_release_sha256: str
    release_inventory: tuple[ReleaseInventoryEntry, ...]
    prepared_artifacts: tuple[PreparedArtifactRecord, ...]
    history_eligible_train_count: int
    sequence_splits: tuple[Mapping[str, object], ...]
    excluded_samples: tuple[Mapping[str, object], ...]
    samples: tuple[TemporalSampleRecord, ...]
    content_sha256: str

def canonical_json_bytes(value: object) -> bytes
def content_sha256(value_without_hash: object) -> str
def build_release_inventory(
    root: Path,
    relative_paths: Iterable[str],
) -> tuple[ReleaseInventoryEntry, ...]
def release_inventory_sha256(
    inventory: Sequence[ReleaseInventoryEntry | Mapping[str, object]],
) -> str
def load_temporal_manifest(
    path: Path,
    expected_split_hash: str,
    allow_fixture: bool = False,
) -> TemporalManifest
```

- [ ] **Step 1: Write hash and schema tests**

```python
import hashlib
import json
from pathlib import Path

import pytest

from transvision.dataset.resilient_v2x_manifest import (
    canonical_json_bytes,
    content_sha256,
    release_inventory_sha256,
)


def test_canonical_json_is_sorted_compact_utf8_and_rejects_nan() -> None:
    value = {"中文": "值", "b": 2, "a": 1}
    assert canonical_json_bytes(value) == (
        '{"a":1,"b":2,"中文":"值"}'.encode("utf-8")
    )
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json_bytes({"value": float("nan")})


def test_content_hash_excludes_only_top_level_content_sha256() -> None:
    first = {"schema_version": 1, "samples": [{"id": "a"}]}
    second = {**first, "content_sha256": "ignored"}
    assert content_sha256(first) == content_sha256(second)
    assert content_sha256(first) == hashlib.sha256(
        canonical_json_bytes(first)
    ).hexdigest()


def test_inventory_hash_depends_on_path_size_and_file_hash() -> None:
    inventory = [
        {"relative_path": "a.bin", "size": 3, "sha256": "0" * 64},
        {"relative_path": "b.jpg", "size": 4, "sha256": "1" * 64},
    ]
    assert release_inventory_sha256(inventory) == hashlib.sha256(
        canonical_json_bytes(inventory)
    ).hexdigest()
```

Add schema tests for duplicate `(agent,modality,n_s)` or packet ID within one
`TemporalSampleRecord`, non-contiguous source positions inside one sample, wrong
`tau=n*100`, path traversal, non-finite
matrices, inventory order/hash mismatch, a wrong `history_eligible_train_count`,
annotation hash mismatch, malformed/non-finite ground-truth boxes, duplicate or
negative `source_annotation_index`, and manifest self-hash mismatch.
Every slice calibration path/hash must occur in the release inventory and its
serialized `agent_from_sensor`/optional camera intrinsic must exactly match those
bytes. LiDAR requires `camera_intrinsic=None`; Camera requires a finite,
nonsingular 3x3 intrinsic with positive focal lengths. Mutating any intrinsic,
extrinsic, calibration path/hash or target-ego LiDAR calibration changes the
manifest hash and fails strict load.
`TemporalManifest` parsing must reject missing or extra top-level/nested fields and
must prove that `history_eligible_train_count` equals the number of emitted train
samples with all four `h=0..history_limit` positions contained in their sequence.
Ground truth is already normalized to the target tick's current ego LiDAR frame:
right-handed X-forward/Y-left/Z-up, bottom-center z, `(length,width,height)`, radians,
and only `Car`; the schema rejects non-positive dimensions.
The same physical source packet may legitimately recur as history in multiple
target samples. Such cross-target reuse must keep identical sequence/source tick,
agent/modality, packet/frame IDs, relative path, timestamps, transforms, validity,
raw inventory bytes/hash and prepared-artifact mapping. A legal reuse fixture
passes; changing any reused provenance field under the same packet ID fails.
By default, loading requires `protocol_scope="controlled"` and both the expected
hash and manifest hash equal `OFFICIAL_COOPERATIVE_SPLIT_SHA256`; fixture scope is
accepted only with explicit `allow_fixture=True` in tests and can never enter a
controlled plan.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_manifest.py -q`

Expected: FAIL because `resilient_v2x_manifest.py` is absent.

- [ ] **Step 3: Implement strict dataclass parsing and hashing**

Use `json.loads`, explicit key-set checks, `Path.resolve().is_relative_to(root.resolve())`, float finiteness checks, and sorted inventory records. Do not silently coerce strings to integers or ignore unknown keys.

The manifest loader must check:

```python
if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
    raise ManifestError("unsupported manifest schema version")
if payload["split_sha256"] != expected_split_hash:
    raise ManifestError("split hash mismatch")
if payload["protocol_scope"] == "controlled" and (
    expected_split_hash != OFFICIAL_COOPERATIVE_SPLIT_SHA256
    or payload["split_sha256"] != OFFICIAL_COOPERATIVE_SPLIT_SHA256
):
    raise ManifestError("controlled split must use the official cooperative hash")
if payload["protocol_scope"] == "fixture" and not allow_fixture:
    raise ManifestError("fixture manifest is not controlled-eligible")
if release_inventory_sha256(payload["release_inventory"]) != payload["dataset_release_sha256"]:
    raise ManifestError("dataset release inventory hash mismatch")
if content_sha256(payload) != payload["content_sha256"]:
    raise ManifestError("temporal manifest content hash mismatch")
```

Parsing constructs every nested dataclass above and verifies the prepared-artifact
`size/sha256` records without folding them into `dataset_release_sha256`; that fingerprint
covers only the raw release inventory. A plain import of
`transvision.dataset.resilient_v2x_manifest` must not import
`transvision.dataset.transforms`, `bev_pool_ext`, or `voxel_layer`; refactor
`transvision/dataset/__init__.py` to explicit lazy exports before running this CPU test.

Calibration directions are canonical: `world_from_agent` maps agent coordinates
to world, `agent_from_sensor` maps raw sensor coordinates to that agent, and
`camera_intrinsic` maps camera coordinates to original image pixels. The
calibration file path/hash, extrinsic and intrinsic are immutable per-slice
provenance and cross-target packet reuse must preserve them exactly.

- [ ] **Step 4: Run manifest tests**

Run: `python -m pytest tests/resilient_v2x/test_manifest.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/dataset/resilient_v2x_manifest.py transvision/dataset/__init__.py tests/resilient_v2x/test_manifest.py
git commit -m "feat: add resilient v2x temporal manifest schema"
```

---

### Task 2: 构建 DAIR cooperative 时序 manifest CLI

**Files:**

- Create: `tools/resilient_v2x/prepare_data.py`
- Create: `transvision/dataset/resilient_v2x_pcd.py`
- Create: `tests/resilient_v2x/fixtures/dair-mini/cooperative/data_info.json`
- Create: `tests/resilient_v2x/fixtures/dair-mini/split.json`
- Create: `tests/resilient_v2x/test_prepare_data.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PreparedPointCloud:
    destination: Path
    point_count: int
    size: int
    sha256: str
    dtype: Literal["<f4"]
    fields: tuple[
        Literal["x"],
        Literal["y"],
        Literal["z"],
        Literal["intensity"],
    ]

def build_protocol_sequences(
    paired_records: Sequence[Mapping[str, object]],
    delta_t_ms: int,
    interval_min_ms: int,
    interval_max_ms: int,
    max_capture_skew_ms: int,
) -> tuple[list[TemporalSampleRecord], list[dict[str, object]]]

def prepare_manifest(
    data_root: Path,
    split_path: Path,
    output_path: Path,
    expected_split_sha256: str,
    protocol_scope: Literal["controlled", "fixture"],
    delta_t_ms: int,
    history_limit: int,
    interval_min_ms: int,
    interval_max_ms: int,
    max_capture_skew_ms: int,
) -> TemporalManifest

def convert_pcd_to_bin(source: Path, destination: Path) -> PreparedPointCloud

def main(argv: Sequence[str] | None = None) -> int
```

- [ ] **Step 1: Add a minimal four-tick fixture and failing CLI test**

The fixture must contain four paired records with all four sensor paths, per-sensor capture timestamps, ego/RSU poses, camera/LiDAR extrinsics, labels, and an official sample ID. The test invokes:

```python
exit_code = main([
    "--data-root", str(fixture_root),
    "--split-file", str(fixture_root / "split.json"),
    "--expected-split-sha256", sha256_file(fixture_root / "split.json"),
    "--protocol-scope", "fixture",
    "--output", str(output_path),
    "--delta-t-ms", "100",
    "--history-limit", "3",
    "--interval-min-ms", "50",
    "--interval-max-ms", "150",
    "--max-capture-skew-ms", "50",
])
assert exit_code == 0
manifest = json.loads(output_path.read_text())
assert manifest["protocol_scope"] == "fixture"
assert [sample["n_t"] for sample in manifest["samples"]] == [0, 1, 2, 3]
assert manifest["samples"][3]["tau_t_ms"] == 300
assert len(manifest["samples"][3]["source_slices"]) == 2 * 2 * 4
assert manifest["samples"][3]["ground_truth"] == [{
    "class_name": "Car",
    "x": 10.0,
    "y": 0.0,
    "z_bottom": 0.0,
    "length": 4.0,
    "width": 2.0,
    "height": 2.0,
    "yaw": 0.0,
    "source_annotation_index": 0,
}]
assert manifest["sequence_splits"] == []
assert manifest["history_eligible_train_count"] == 1
```

Add boundary fixtures:

- exactly 50 ms skew passes, 51 ms fails;
- adjacent interval 49 or 151 ms creates a new sequence and history does not cross it;
- for each of the four `(ego|rsu, lidar|camera)` raw capture streams, timestamps
  must be strictly increasing in official paired-record order: duplicate, zero,
  or negative deltas fail before any manifest or staging file is created. A
  positive 49 ms or 151 ms delta remains a valid sequence boundary rather than a
  timestamp protocol error; parameterized fixtures cover all four streams
  independently;
- missing calibration fails;
- split counts and IDs must match the split file;
- default/controlled CLI rejects any split hash other than
  `OFFICIAL_COOPERATIVE_SPLIT_SHA256`; fixture scope must be explicit, is stamped
  into the manifest, and is rejected by every controlled data/experiment gate;
- a train target with no supported branch is listed in `excluded_samples`, not emitted.
- a world-frame label transformed by a non-identity target-ego pose becomes the
  exact expected ego-LiDAR bottom-center box; a real
  `LiDARInstance3DBoxes` constructed from that normalized tuple has the same
  `bottom_center`, dimensions and yaw;
- annotation frame ambiguity, non-rigid transforms, invalid dimensions, non-finite
  corners, or a label whose transformed corners are not a rectangular cuboid fail;
  non-`Car` objects are ignored while their source annotation file remains hashed.
- ASCII、binary 和 binary-compressed PCD fixtures 转换为相同 little-endian float32 `[x,y,z,intensity]` layout；
- 转换结果按点原始顺序写入，重复运行 byte-identical；
- manifest 同时记录 raw `.pcd` 的 release inventory hash 和 prepared `.bin` 的 size/SHA-256。

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_prepare_data.py -q`

Expected: FAIL because the CLI module is absent.

- [ ] **Step 3: Implement ordered sequence construction**

The parser must use official paired-record order, never nearest-neighbor re-pair
sensors. Before deriving sequence boundaries, it validates strict monotonicity
independently for ego LiDAR, ego camera, RSU LiDAR and RSU camera raw capture
timestamps. Any current timestamp `<=` the prior timestamp in the same stream is
a protocol error and aborts before manifest staging; only positive intervals
outside `[interval_min_ms, interval_max_ms]` create a new sequence. For each
validated sequence:

Raw timestamps remain integer microseconds throughout validation. Convert CLI
millisecond bounds once with exact integer multiplication: per-tick skew is
`max(capture_timestamp_us)-min(capture_timestamp_us) <= 50_000`; a positive
same-stream delta is a sequence boundary iff it is outside
`[50_000,150_000]`. Never use float division or compare microseconds directly to
millisecond values. Boundary fixtures cover skew `50_000/50_001` and positive
intervals `49_999/50_000/150_000/150_001` microseconds.

```python
for n_t, record in enumerate(sequence_records):
    tau_t_ms = n_t * delta_t_ms
    history = sequence_records[max(0, n_t - history_limit):n_t + 1]
```

Every raw referenced file, official metadata file, split file, label, pose, and
calibration enters the release inventory. Write the complete canonical manifest
to a sibling temp, fsync it, publish with an atomic no-replace primitive, fsync
the directory, then remove the temp. An existing byte/hash-identical manifest is
idempotently reused; a conflict is never overwritten. Crash tests at temp
write/fsync/publish/directory-fsync/cleanup prove the final path is always absent
or complete.

`--protocol-scope` defaults to `controlled`. In that mode both the supplied
expected hash and actual split bytes must equal
`OFFICIAL_COOPERATIVE_SPLIT_SHA256`; a caller cannot redefine the official value.
`fixture` mode accepts only test-owned mini data, sets
`protocol_scope="fixture"` in the canonical payload and is never
controlled/paper eligible.

For every target tick, parse its official cooperative annotation in the explicitly
declared source frame and transform all eight box corners through the recorded rigid
chain into the current ego LiDAR frame. Recover bottom center, length/width/height
and yaw from the transformed cuboid, wrap yaw canonically to `[-pi, pi)`, retain the
original annotation array index, and store the normalized `GroundTruthBoxRecord`
tuple plus source annotation SHA-256 inline in `TemporalSampleRecord`. Do not call
the legacy `V2XDataset` camera-coordinate label conversion at evaluation time.
The manifest content hash therefore freezes both the raw annotation provenance and
the exact evaluator-ready ground truth.

`convert_pcd_to_bin()` uses pinned `pypcd4==1.4.3`, requires fields `x/y/z/intensity`, converts to little-endian contiguous float32, rejects non-finite values, writes atomically, and refuses to overwrite a different existing artifact. Raw PCDs remain in `dataset_release_sha256`; the prepared artifact inventory and hashes enter the temporal manifest.

- [ ] **Step 4: Run prepare-data tests and CLI help**

Run:

```bash
python -m pytest tests/resilient_v2x/test_prepare_data.py -q
python tools/resilient_v2x/prepare_data.py --help
```

Expected: tests pass; help lists the nine data/hash/timing arguments plus
`--protocol-scope` and exits 0.

- [ ] **Step 5: Commit**

```bash
git add tools/resilient_v2x/prepare_data.py transvision/dataset/resilient_v2x_pcd.py tests/resilient_v2x/fixtures/dair-mini tests/resilient_v2x/test_prepare_data.py
git commit -m "feat: prepare dair temporal manifests"
```

---

### Task 3: 实现确定性 transport/fault/augmentation overlays

**Files:**

- Create: `transvision/dataset/resilient_v2x_schedule.py`
- Create: `tests/resilient_v2x/test_schedule.py`
- Modify: `transvision/dataset/__init__.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class TransportPlan:
    temporal_manifest_sha256: str
    split: Literal["train", "val", "test"]
    samples: tuple[TemporalSampleRecord, ...]
    mode: Literal["train_random", "fixed_evaluation"]
    protocol_seed: int | None
    epochs: tuple[int, ...]
    delay_values_ms: tuple[int, ...]
    fixed_delay_ms: int | None

@dataclass(frozen=True)
class FaultPlan:
    temporal_manifest_sha256: str
    split: Literal["train", "val", "test"]
    samples: tuple[TemporalSampleRecord, ...]
    mode: Literal["train_random", "global_target", "continuous"]
    protocol_seed: int | None
    epochs: tuple[int, ...]
    condition: Literal["Full", "L-Fail", "C-Fail"] | None
    p_lidar: float | None
    p_camera: float | None
    agents: tuple[Literal["ego", "rsu"], ...]
    modality: Literal["lidar", "camera"] | None
    duration: int | None

@dataclass(frozen=True)
class ArrivalRelativeFaultPlan:
    temporal_manifest_sha256: str
    transport_overlay_sha256: str
    split: Literal["val", "test"]
    samples: tuple[TemporalSampleRecord, ...]
    scope: Literal["ego", "rsu"]
    modality: Literal["lidar", "camera"]
    fixed_delay_ms: Literal[0, 300]

@dataclass(frozen=True)
class OverlayDigest:
    path: Path
    record_count: int
    uncompressed_size: int
    uncompressed_sha256: str
    compressed_size: int
    compressed_sha256: str

@dataclass(frozen=True)
class TransportOverlayRecord:
    epoch: int | None
    sample_id: str
    packet_id: str
    n_s: int
    delay_ms: int
    arrival_tau_ms: int

@dataclass(frozen=True)
class FaultOverlayRecord:
    epoch: int | None
    sample_id: str
    agent: Literal["ego", "rsu"]
    modality: Literal["lidar", "camera"]
    n_s: int
    masked: bool
    pre_mask_selected_n_s: int | None
    fallback_selected_n_s: int | None

def stable_uint64(domain: str, key: Mapping[str, object]) -> int
def bernoulli_from_hash(domain: str, key: Mapping[str, object], probability: float) -> bool
def delay_from_hash(domain: str, key: Mapping[str, object], values_ms: Sequence[int]) -> int
def write_transport_overlay(plan: TransportPlan, output: Path) -> OverlayDigest
def write_fault_overlay(plan: FaultPlan, output: Path) -> OverlayDigest
def write_arrival_relative_fault_overlay(
    plan: ArrivalRelativeFaultPlan,
    temporal_manifest: TemporalManifest,
    transport_records: Sequence[Mapping[str, object]],
    output: Path,
) -> OverlayDigest
def read_overlay(
    path: Path,
    expected_uncompressed_sha256: str,
) -> tuple[Mapping[str, object], ...]
def augmentation_seed(seed: int, epoch: int, sample_id: str) -> int
```

- [ ] **Step 1: Write fixed-vector and order-independence tests**

```python
from transvision.dataset.resilient_v2x_schedule import (
    bernoulli_from_hash,
    delay_from_hash,
    stable_uint64,
)


KEY = {"seed": 0, "epoch": 1, "sample_id": "000123", "n_s": 8}


def test_stable_hash_fixed_vector() -> None:
    assert stable_uint64("transport-delay-v1", KEY) == 46208971577506763


def test_json_key_order_does_not_change_hash() -> None:
    reversed_key = dict(reversed(list(KEY.items())))
    assert stable_uint64("transport-delay-v1", KEY) == stable_uint64(
        "transport-delay-v1", reversed_key
    )


def test_delay_is_one_of_four_protocol_values() -> None:
    assert delay_from_hash(
        "transport-delay-choice-v1", KEY, (0, 100, 200, 300)
    ) in {0, 100, 200, 300}
```

Generate the fixed vector once from the specified algorithm, place the computed literal in the test, and verify it independently with:

```python
payload = b"transport-delay-v1\x00" + canonical_json_bytes(KEY)
expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
assert expected == stable_uint64("transport-delay-v1", KEY)
```

Add tests for worker/order independence, same source tick sharing RSU
LiDAR/Camera delay, eval condition being seed-independent, train target-only
faults, zstd round-trip, sorted JSONL keys, and uncompressed digest mismatch.
Transport coverage is exact: for each sample/epoch it contains every and only the
packet IDs referenced by RSU LiDAR/Camera source slices, once per referenced
packet; there is no ego transport record. The two RSU modalities at one `n_s`
receive the same delay derived from that source tick. Injecting an ego packet
record, omitting/duplicating an RSU packet, or assigning a nonzero ego arrival
delay fails schema/coverage validation. A selection test proves ego branches
never consult the arrival cutoff while RSU branches do.

Add constructor-invariant tests for all plan dataclasses. A `train_random` transport plan
requires a seed, non-empty epochs, `delay_values_ms==(0,100,200,300)`, and no fixed delay;
a fixed-evaluation plan requires no seed/epochs and exactly one fixed delay. Training faults
require `p_lidar/p_camera` and target `n_t` only; global-target and continuous plans forbid
probabilities and encode their condition/scope/duration explicitly. Arrival-relative plans
are val/test-only and bind the exact transport overlay digest. `OverlayDigest` rejects
negative sizes/counts and non-64-character lowercase hashes.

Add exact fault fixtures that verify:

- global-target L/C-Fail at 0/100/200/300 ms produces the approved horizon table；
- continuous LiDAR E+R durations 1/2/3 mask exactly that many target-ending ticks；
- duration 4 exhausts all four history positions；
- arrival-relative source-fail first selects without a fault, masks that selected source exactly once, then permits one fallback selection；
- R-only@300 ms masks the actually arrived RSU source rather than the unarrived target packet；
- arrival-relative fallback is not recursively masked.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_schedule.py -q`

Expected: FAIL because schedule APIs are absent.

- [ ] **Step 3: Implement canonical hash and fixed zstd encoding**

Use `u=(x+0.5)/2**64` and `u < probability`; reject probability outside `[0,1]`. Use a zstd compressor with explicit `level=19`, `threads=0`, `write_checksum=True`, `write_content_size=True`, and canonical newline-terminated JSON records. Digest the uncompressed bytes, not the `.zst` container.

All three overlay writers use the same immutable publication primitive: encode
to a unique sibling staging file, flush/fsync, verify compressed and
uncompressed hashes by reread, atomically publish without replacement, fsync the
parent directory, and remove staging. A byte/hash-identical existing artifact is
an idempotent success; any conflict fails. Crash-injection tests cover every
write/file-fsync/no-replace-publish/directory-fsync/cleanup boundary and prove a
truncated final `.zst` can never be observed. Retry may remove only its own
unpublished staging file.

Transport is an RSU-network overlay only. Its exact key set is reconstructed from
the temporal manifest's RSU LiDAR/Camera source slices; ego packet IDs are
forbidden. Transport key is `(seed,epoch,sample_id,n_s)` and the resulting one
delay is reused by both RSU modalities at that source tick. Fault key is
`(seed,epoch,sample_id,agent,modality,n_t)`. Do not include worker/rank/read order.

Serialized transport records are sorted by
`(epoch_is_none,epoch,sample_id,packet_id)` and serialized fault records by
`(epoch_is_none,epoch,sample_id,agent,modality,n_s)`; fixed evaluation records store
`epoch=null`. Record key sets must match the dataclasses exactly. The uncompressed stream
ends in one newline; `record_count`, both sizes, and both hashes in `OverlayDigest` are
computed from the bytes actually written.

`write_arrival_relative_fault_overlay()` is evaluation-only. It resolves the no-fault
arrived set from the immutable transport overlay, records the selected source tick for each
scoped branch, writes exactly that one mask, and stores both the pre-mask selection and final
fallback diagnostics in the overlay record.

- [ ] **Step 4: Run schedule tests**

Run: `python -m pytest tests/resilient_v2x/test_schedule.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/dataset/resilient_v2x_schedule.py transvision/dataset/__init__.py tests/resilient_v2x/test_schedule.py
git commit -m "feat: add deterministic v2x protocol overlays"
```

---

### Task 4: 实现 metadata-first dataset 和只加载有效 slice 的 transforms

**Files:**

- Create: `transvision/dataset/resilient_v2x_dataset.py`
- Create: `transvision/dataset/transforms/resilient_v2x.py`
- Create: `transvision/dataset/samplers/__init__.py`
- Create: `transvision/dataset/samplers/resilient_v2x_sampler.py`
- Create: `transvision/models/data_preprocessors/resilient_v2x_preprocessor.py`
- Modify: `transvision/models/data_preprocessors/__init__.py`
- Modify: `transvision/dataset/transforms/__init__.py`
- Modify: `transvision/dataset/__init__.py`
- Create: `tests/resilient_v2x/test_dataset_pipeline.py`

**Interfaces:**

```python
BRANCH_ORDER = ("lidar_ego", "lidar_rsu", "camera_ego", "camera_rsu")
HISTORY_HORIZONS = (0, 1, 2, 3)

@dataclass(frozen=True)
class ResolvedHistoryEntry:
    modality: Literal["lidar", "camera"]
    agent: Literal["ego", "rsu"]
    horizon: int
    flat_index: int
    record: RawSliceRecord
    load_relative_path: str
    load_sha256: str
    load_size: int
    point_count: int | None
    source_from_sensor: tuple[tuple[float, ...], ...]
    source_to_target: tuple[tuple[float, ...], ...]
    camera_intrinsic: tuple[tuple[float, ...], ...] | None

@dataclass(frozen=True)
class ResolvedSampleHistory:
    sample_id: str
    epoch: int | None
    selections: tuple[BranchSelection, BranchSelection, BranchSelection, BranchSelection]
    student_entries: tuple[ResolvedHistoryEntry, ...]
    student_availability: tuple[
        tuple[tuple[bool, bool, bool, bool], tuple[bool, bool, bool, bool]],
        tuple[tuple[bool, bool, bool, bool], tuple[bool, bool, bool, bool]],
    ]
    student_selected_flat_index: tuple[int | None, int | None, int | None, int | None]
    teacher_entries: tuple[ResolvedHistoryEntry, ...] | None
    teacher_availability: tuple[
        tuple[tuple[bool, bool, bool, bool], tuple[bool, bool, bool, bool]],
        tuple[tuple[bool, bool, bool, bool], tuple[bool, bool, bool, bool]],
    ] | None
    teacher_selected_flat_index: tuple[int | None, int | None, int | None, int | None] | None

class ModalityHistoryBatch(TypedDict):
    payloads: Sequence[object]
    valid_flat_index: Tensor
    availability: Tensor
    source_from_sensor: Tensor
    source_to_target: Tensor
    camera_intrinsics: Tensor | None

class ResilientModelInputs(TypedDict):
    branch_order: tuple[str, str, str, str]
    history: dict[Literal["lidar", "camera"], ModalityHistoryBatch]
    selected_flat_index: Tensor
    selections: tuple[tuple[BranchSelection, ...], ...]
    augmentation_matrices: Mapping[str, Tensor]
    diagnostics: tuple[Mapping[str, object], ...]

class ResilientBatchInputs(TypedDict):
    student_inputs: ResilientModelInputs
    teacher_inputs: ResilientModelInputs

class ResilientPreprocessedBatch(TypedDict):
    inputs: ResilientModelInputs | ResilientBatchInputs
    data_samples: Sequence[Det3DDataSample]

@DATASETS.register_module()
class ResilientV2XDataset(Det3DDataset):
    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        expected_split_sha256: str,
        split: Literal["train", "val", "test"],
        transport_overlay_path: str,
        transport_overlay_sha256: str,
        fault_overlay_path: str,
        fault_overlay_sha256: str,
        protocol: Mapping[str, object],
        teacher_mode: bool,
        allow_fixture_manifest: bool = False,
        **kwargs: object,
    ) -> None

@TRANSFORMS.register_module()
class ResolveResilientV2XHistory(BaseTransform):
    def __init__(self, delta_t_ms: int, history_limit: int) -> None
    def transform(self, results: dict[str, object]) -> dict[str, object]

@TRANSFORMS.register_module()
class LoadSelectedResilientV2XInputs(BaseTransform):
    def __init__(
        self,
        data_root: str,
        lidar_loader: Mapping[str, object],
        camera_loader: Mapping[str, object],
    ) -> None
    def transform(self, results: dict[str, object]) -> dict[str, object]

@TRANSFORMS.register_module()
class SharedResilientV2XAugmentation(BaseTransform):
    def __init__(
        self,
        rotation_range: tuple[float, float],
        scale_range: tuple[float, float],
        translation_std: tuple[float, float, float],
        flip_y_probability: float,
        resized_image_size: tuple[int, int],
        cropped_image_size: tuple[int, int],
    ) -> None
    def transform(self, results: dict[str, object]) -> dict[str, object]

@TRANSFORMS.register_module()
class PackResilientV2XInputs(BaseTransform):
    def transform(self, results: dict[str, object]) -> dict[str, object]

@DATA_SAMPLERS.register_module()
class ResilientV2XSampler(Sampler[tuple[int, int]]):
    def __init__(
        self,
        dataset: Sized,
        seed: int,
        shuffle: bool,
        global_batch_size: int,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None
    def set_epoch(self, epoch: int) -> None
    def __iter__(self) -> Iterator[tuple[int, int]]
    def __len__(self) -> int

def build_global_epoch_order(
    dataset_size: int,
    seed: int,
    epoch: int,
    global_batch_size: int,
    shuffle: bool,
) -> tuple[int, ...]

def conjugate_source_to_target(
    source_to_target: Tensor,
    bev_augmentation: Tensor,
) -> Tensor
def augment_camera_geometry(
    camera_to_source: Tensor,
    camera_intrinsic: Tensor,
    bev_augmentation: Tensor,
    image_augmentation: Tensor,
) -> tuple[Tensor, Tensor]
def read_verified_regular_file(
    data_root: Path,
    relative_path: str,
    expected_size: int,
    expected_sha256: str,
) -> bytes

@MODELS.register_module()
class ResilientV2XDataPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        bgr_to_rgb: bool,
        image_size: tuple[int, int],
        non_blocking: bool,
        view_mode: Literal["student", "paired"],
    ) -> None
    def forward(
        self,
        data: Mapping[str, object],
        training: bool,
    ) -> ResilientPreprocessedBatch
```

For each modality, `availability` is `[B,2,4]` in agent order `(ego,rsu)` and horizon
order `(0,1,2,3)`, where `horizon=n_t-n_s`. `valid_flat_index` is one-dimensional and
strictly increasing; it indexes the corresponding
flattened `[B,2,4]` grid with
`((batch_index * 2 + agent_index) * 4 + horizon)`. `selected_flat_index` is `[B,4]`
in `BRANCH_ORDER`, stores that modality-local flat index, and uses `-1` only for an
unsupported branch. `source_to_target` is `[B,2,4,4,4]`; unavailable positions contain
identity only as an internal neutral value and remain false in `availability`.

- [ ] **Step 1: Write a spy-loader test proving only causal history is read**

```python
def test_loader_opens_all_and_only_causal_student_history(
    monkeypatch,
    mini_manifest,
) -> None:
    opened: list[str] = []

    def spy_loader(path: str) -> object:
        opened.append(path)
        if "forbidden" in path:
            raise AssertionError("forbidden payload was read")
        return object()

    pipeline = build_test_pipeline(raw_loader=spy_loader, delay_ms=200)
    packed = pipeline(mini_manifest.sample("target-3"))
    batch = preprocess_for_test([packed], view_mode="student")
    student = batch["inputs"]
    assert opened == mini_manifest.expected_student_paths(
        sample_id="target-3",
        delay_ms=200,
    )
    assert len(opened) == 12
    assert all("forbidden" not in path for path in opened)
    assert student["branch_order"] == (
        "lidar_ego", "lidar_rsu", "camera_ego", "camera_rsu"
    )
    assert tuple(
        selection.selected_n_s for selection in student["selections"][0]
    ) == (3, 1, 3, 1)
    expected_availability = torch.tensor(
        [[True, True, True, True], [False, False, True, True]],
        dtype=torch.bool,
    )
    assert torch.equal(student["history"]["lidar"]["availability"][0], expected_availability)
    assert torch.equal(student["history"]["camera"]["availability"][0], expected_availability)
    assert student["history"]["lidar"]["valid_flat_index"].tolist() == [
        0, 1, 2, 3, 6, 7
    ]
    assert student["history"]["camera"]["valid_flat_index"].tolist() == [
        0, 1, 2, 3, 6, 7
    ]
    assert student["selected_flat_index"][0].tolist() == [0, 6, 0, 6]
```

Add tests that:

- controlled dataset construction verifies every raw release-inventory entry and
  every prepared-artifact entry before exposing an index; mutating a source PCD,
  prepared BIN, size, point count, or either hash fails before a loader is called;
- every LiDAR `ResolvedHistoryEntry` maps its raw `.pcd` provenance record to the
  unique manifest-declared prepared `.bin`, carries that artifact's size/hash/point
  count, and the spy loader sees only the prepared path; Camera entries retain their
  raw image path and release-inventory size/hash with `point_count=None`;
- a missing/duplicate prepared mapping, a prepared path outside `data_root`, or a
  selected LiDAR path ending in `.pcd` fails closed;
- a non-unit/non-commuting target ego pose, source agent pose, LiDAR extrinsic,
  camera extrinsic and camera intrinsic fixture maps one raw LiDAR point and one
  camera pixel/depth ray to exact golden target-ego BEV cells. Mutation of any
  calibration field changes the result/hash and fails before an encoder call;
- a replacement-race spy swaps a path between validation and decode; the loader
  consumes only bytes returned by `read_verified_regular_file()` from one stable
  regular-file descriptor, so unverified replacement bytes never reach either
  encoder;
- changing any causally usable non-selected history slice changes the encoded PTF context,
  proving the loader did not load only the four selected sources;
- teacher inputs contain a separately loaded clean `(ego,rsu) x h=0..3` history for both
  modalities (16 distinct payload objects), select `[0,4,0,4]` at `n_t`, and are stored
  only under `teacher_inputs`;
- student dict has no reference to the teacher dict or teacher tensors;
- newest invalid candidate falls back before load;
- same augmentation matrices and target boxes are used for teacher/student;
- a non-commuting rotation-plus-translation fixture verifies the exact geometry
  identities below for ego/RSU LiDAR, ego/RSU camera LSS and target boxes.
  Leaving `source_to_target`, camera extrinsics or intrinsics stale moves a known
  landmark away from the expected augmented BEV cell and fails; student and
  teacher matrices/landmarks are byte-equal;
- image resize from `1080x1920` is `396x704` then bottom crop top offset 140;
- image shorter than 256 after resize fails closed;
- BGR→RGB and ImageNet normalization order is exact;
- batch collation preserves variable selected-slice counts and fixed branch order.
- sampler globally shuffles then truncates to `4*floor(N_train/4)` before rank sharding, never pads/repeats；
- sampler yields `(epoch,index)` so persistent workers receive the exact overlay epoch；
- 1/2/4 ranks see disjoint shards whose union equals the truncated global order；
- sampler `__len__` equals its emitted shard length and rank/world size are derived from
  `get_dist_info()` when the config omits them; explicit values are test-only overrides；
- preprocessor recursively moves nested student/teacher tensors but preserves both contracts and diagnostics；
- preprocessor produces the exact `ResilientModelInputs` keys and flat-index formulas above；
- each modality's payload order matches its sorted `valid_flat_index` one-for-one, and the
  paired outer mapping matches `ResilientBatchInputs` exactly；
- `view_mode="student"` returns one direct `ResilientModelInputs` for a bare detector and
  rejects any loaded teacher view, while `view_mode="paired"` requires two non-aliased views
  and returns `ResilientBatchInputs` for the Task 8 wrapper；
- image normalization/BGR→RGB occurs exactly once in the preprocessor；
- preprocessor neither voxelizes points nor applies the legacy infrastructure-intensity
  `*255` mutation, because `SharedLidarBEVEncoder` owns voxelization.
- the train dataset index contains exactly the manifest's full-history eligible train
  targets, so `len(dataset)==manifest.history_eligible_train_count`; earlier ticks remain
  addressable only as history sources and never become scheduled train targets.

Use this exact sampler fixture:

```python
def test_sampler_truncates_globally_without_padding() -> None:
    rank_zero = ResilientV2XSampler(
        dataset=range(11),
        seed=7,
        shuffle=True,
        global_batch_size=4,
        rank=0,
        world_size=2,
    )
    rank_one = ResilientV2XSampler(
        dataset=range(11),
        seed=7,
        shuffle=True,
        global_batch_size=4,
        rank=1,
        world_size=2,
    )
    rank_zero.set_epoch(3)
    rank_one.set_epoch(3)
    first = list(rank_zero)
    second = list(rank_one)
    assert all(epoch == 3 for epoch, _ in first + second)
    indices = [index for _, index in first + second]
    assert len(indices) == 8
    assert len(set(indices)) == 8
    assert set(index for _, index in first).isdisjoint(
        index for _, index in second
    )
    expected = build_global_epoch_order(11, 7, 3, 4, True)
    reconstructed = tuple(
        item[1]
        for rank_pair in zip(first, second)
        for item in rank_pair
    )
    assert reconstructed == expected
    assert len(rank_zero) == len(first) == 4
    assert len(rank_one) == len(second) == 4


def test_sampler_derives_runtime_rank_and_world_size(monkeypatch) -> None:
    monkeypatch.setattr(
        "transvision.dataset.samplers.resilient_v2x_sampler.get_dist_info",
        lambda: (1, 4),
    )
    sampler = ResilientV2XSampler(
        dataset=range(11),
        seed=7,
        shuffle=True,
        global_batch_size=4,
    )
    assert sampler.rank == 1
    assert sampler.world_size == 4
    assert len(sampler) == 2
```

- [ ] **Step 2: Run dataset tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_dataset_pipeline.py -q`

Expected: FAIL because dataset/transforms are absent.

- [ ] **Step 3: Implement two-pass resolve/load behavior**

`ResolveResilientV2XHistory` may inspect paths and metadata but must not open image/point
files. For each modality it first builds the full `[2,4]` history grid, applies arrival,
fault, metadata, and `h<=history_limit` checks to every position, and then calls the core
`select_causal_source` for each branch. It emits a `ResolvedSampleHistory`: all causally
usable `student_entries`, the four selections, exact availability, rejected diagnostics,
and selected flat indices. Entries are sorted by modality `(lidar,camera)`, agent
`(ego,rsu)`, then horizon `(0,1,2,3)`. The PTF context consumes all usable entries, not just
the newest four selections. During controlled dataset preflight, build a one-to-one map
from each raw LiDAR `source_relative_path` to its `PreparedArtifactRecord`, path-contain
both raw and prepared names, and re-hash both inventories from disk. A resolved LiDAR
entry uses the verified prepared `.bin` as `load_relative_path`; its `record.relative_path`
remains the immutable raw `.pcd` provenance. A Camera entry uses its verified raw image
as `load_relative_path`. This map is immutable for the dataset lifetime and no transform
may infer a prepared filename by suffix replacement.

`LoadSelectedResilientV2XInputs` must:

```python
student_inputs = {
    "history": {"lidar": [], "camera": []},
    "availability": resolved.student_availability,
    "selected_flat_index": resolved.student_selected_flat_index,
    "selections": resolved.selections,
}
for entry in resolved.student_entries:
    loader = self.lidar_loader if entry.modality == "lidar" else self.camera_loader
    verified_bytes = read_verified_regular_file(
        self.data_root,
        entry.load_relative_path,
        entry.load_size,
        entry.load_sha256,
    )
    student_inputs["history"][entry.modality].append(
        (entry.flat_index, loader.decode_bytes(verified_bytes), entry)
    )
results["student_inputs"] = student_inputs
```

No unarrived, faulted, metadata-invalid, or out-of-horizon entry may reach either loader.
`read_verified_regular_file()` resolves containment, rejects symlinks and
non-regular files, opens once with no-follow semantics, verifies pre/post `fstat`
identity/size plus SHA-256 over the exact bytes read, and returns those same bytes
to the decoder; the decoder never reopens a path. LiDAR additionally checks decoded float count equals
`point_count * 4`, uses only little-endian float32 prepared BIN, and never passes raw PCD
bytes to MMDetection3D's `LoadPointsFromFile`.
When `teacher_mode=True`, resolve the same full grid again while ignoring dynamic
transport/fault overlays, load every clean metadata-valid `h=0..3` entry into a separate
dict with new payload objects, and select the synchronized `h=0` sources. Never alias
mutable containers or tensors between views. Augmentation uses one generator seeded by
`augmentation_seed(seed,epoch,sample_id)` and applies the same 4x4 BEV matrix, image
resize/crop matrix, and target boxes to every student/teacher history entry.

Matrix directions are frozen. Let `T = target_from_source`, `A` be the homogeneous
BEV augmentation applied in both source and target local coordinates,
`E = source_from_camera`, and `H` map original image pixels to resized/cropped
pixels. LiDAR/source coordinates become `p'_s=A p_s`, target coordinates and box
corners become `p'_t=A p_t`, and
`conjugate_source_to_target(T,A)` returns
`T' = A @ T @ inverse(A)`. Raw poses remain immutable provenance; their derived
augmented forms are `world_from_agent' = world_from_agent @ inverse(A)`.
For camera LSS, `augment_camera_geometry()` returns `E' = A @ E` and homogeneous
`K' = H @ K`; with augmented image coordinate `u'=H u`, the exact invariant is
`T' @ E' @ backproject(K',u') == A @ T @ E @ backproject(K,u)`.
Target-box center/corners use `A`; uniform scale updates dimensions, and yaw is
recomputed from the transformed heading vector after rotation/optional Y flip.
Shear, non-uniform scale, or a non-invertible/non-finite `A` is rejected. The
augmentation is sampled once and every derived LiDAR, camera, pose, transform and
box matrix is computed once, then reused byte-for-byte by student and teacher.

Before augmentation, let the target ego LiDAR calibration be
`W_t = target_world_from_ego @ ego_from_lidar`. For a source LiDAR slice,
`source_to_target = inverse(W_t) @
(source_world_from_agent @ agent_from_lidar)`, with
`source_from_sensor=I`. For a camera slice, resolve the same-agent/same-tick
paired LiDAR calibration and set
`source_from_sensor = inverse(agent_from_lidar) @ agent_from_camera`,
`source_to_target = inverse(W_t) @
(source_world_from_agent @ agent_from_lidar)`, and carry the exact 3x3 intrinsic.
Thus `source_to_target @ source_from_sensor` maps raw camera coordinates to the
target ego LiDAR frame. Missing/ambiguous paired LiDAR calibration, a mismatched
calibration hash, non-rigid extrinsic or non-finite/singular intrinsic fails
before `ResolvedHistoryEntry` construction. The batched LSS path consumes
`source_from_sensor`, `camera_intrinsics`, image augmentation and
`source_to_target`; no generic unproven matrix dict or unit-extrinsic fallback is
allowed.

`ResilientV2XDataset` explicitly filters by its required `split` constructor field. For the
train split it indexes exactly those targets with every `h=0..history_limit` position in the
same sequence and cross-checks its length against
`TemporalManifest.history_eligible_train_count`; retained earlier records remain available
to history resolution but are not dataset indices.
`protocol.controlled_training=True` additionally requires a controlled-scope
manifest carrying `OFFICIAL_COOPERATIVE_SPLIT_SHA256`; fixture-scope manifests
are accepted only when a synthetic/non-controlled config explicitly sets
`allow_fixture_manifest=True` and propagate `paper_eligible=False`. Construction
rejects `controlled_training=True` together with that flag.

`ResilientV2XDataset.__getitem__()` requires `(epoch,index)` during controlled training,
merges only that epoch's overlays, and rejects a bare index when
`protocol.controlled_training=True`. Generic smoke/evaluation access may use a bare index
only when no epoch-indexed overlay is configured.
`build_global_epoch_order()` seeds one `torch.Generator` with
`stable_uint64("sampler-shuffle-v1",{"seed":seed,"epoch":epoch})`, creates `randperm(N)`
when `shuffle=True`, circularly shifts it by `(epoch*(N % global_batch_size)) % N`, and
truncates to `global_batch_size*floor(N/global_batch_size)`. Rank `r` receives
`global_order[r::world_size]`. The sampler rejects a global batch not divisible by world
size, derives both runtime values together via `get_dist_info()` when omitted, never pads,
and returns `(truncated_count // world_size)` from `__len__`.

`ResilientV2XDataPreprocessor` replaces `Det3DDataDAIRPreprocessor` for all new configs.
It validates and collates both views into `ResilientModelInputs`, calculates batched flat
indices with the declared formula, recursively moves tensors, stacks/normalizes the already
resized/cropped camera tensors, and leaves points as variable-length float tensors for the
shared LiDAR encoder. It leaves metadata dictionaries immutable and never voxelizes.
`view_mode="student"` returns
`{"inputs": ResilientModelInputs(...), "data_samples": ...}` for bare teacher, concat, and
deployment detectors. `view_mode="paired"` returns
`{"inputs": ResilientBatchInputs(...), "data_samples": ...}` for the teacher/student wrapper.
Dataset `teacher_mode` and preprocessor `view_mode` must agree, and config tests reject the
two mismatched combinations before a Runner is built.

- [ ] **Step 4: Run dataset and manifest tests**

Run: `python -m pytest tests/resilient_v2x/test_manifest.py tests/resilient_v2x/test_schedule.py tests/resilient_v2x/test_dataset_pipeline.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/dataset/resilient_v2x_dataset.py transvision/dataset/transforms/resilient_v2x.py transvision/dataset/samplers transvision/models/data_preprocessors/resilient_v2x_preprocessor.py transvision/models/data_preprocessors/__init__.py transvision/dataset/transforms/__init__.py transvision/dataset/__init__.py tests/resilient_v2x/test_dataset_pipeline.py
git commit -m "feat: add causal resilient v2x data pipeline"
```

---

### Task 5: 实现共享 LiDAR 和 Camera BEV encoders

**Files:**

- Create: `transvision/models/resilient_v2x/encoders.py`
- Create: `tests/resilient_v2x/test_encoders.py`
- Modify: `transvision/models/resilient_v2x/__init__.py`

**Interfaces:**

```python
class SharedLidarBEVEncoder(nn.Module):
    def __init__(
        self,
        voxel_layer: Mapping[str, object],
        voxel_encoder: Mapping[str, object],
        middle_encoder: Mapping[str, object],
        backbone: Mapping[str, object],
        neck: Mapping[str, object],
        projection_channels: int,
        expected_neck_shape: tuple[int, int, int],
        expected_output_shape: tuple[int, int, int],
    ) -> None

    def forward(self, valid_point_slices: Sequence[Tensor]) -> Tensor

class SharedCameraBEVEncoder(nn.Module):
    def __init__(
        self,
        backbone: Mapping[str, object],
        neck: Mapping[str, object],
        view_transform: Mapping[str, object],
        projection_channels: int,
        chunk_size: int,
        expected_neck_shape: tuple[int, int, int],
        expected_output_shape: tuple[int, int, int],
    ) -> None

    def forward(
        self,
        valid_images: Tensor,
        camera_matrices: Mapping[str, Tensor],
    ) -> Tensor

def scatter_encoded_history(
    encoded_valid: Tensor,
    valid_index: Tensor,
    output_shape: tuple[int, int, int, int, int, int],
) -> Tensor
```

- [ ] **Step 1: Write registry-built fake-component tests**

Register lightweight fake voxel, backbone, neck, and LSS modules inside the test. Assert:

```python
assert lidar_output.shape == (valid_count, 256, 8, 8)
assert camera_output.shape == (valid_count, 256, 8, 8)
assert fake_lidar_encoder.call_count == 1
assert fake_camera_backbone.seen_batch_sizes == [2, 2, 1]
```

for `valid_count=5`, `chunk_size=2`, with explicit test-only expected shapes. Add tests that:

- LiDAR test neck output must be a sequence whose index 0 is `[N,384,8,8]`;
- camera test neck index 0 must be `[N,256,4,11]`;
- unexpected levels/shapes fail with explicit messages;
- `scatter_encoded_history` fills invalid positions with exact zeros;
- changing invalid raw payload cannot cause an encoder call or output change;
- ego and RSU/time slices share the same module parameter object IDs.
- resolved production config fixes LiDAR neck `[384,288,288]`、Camera neck `[256,32,88]` 和两路输出 `[256,288,288]`，由 Task 7 golden test 检查，不在 CPU unit test 中分配八路完整 BEV。

- [ ] **Step 2: Run encoder tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_encoders.py -q`

Expected: FAIL because `encoders.py` is absent.

- [ ] **Step 3: Implement explicit shape assertions and shared batching**

LiDAR projection is:

```python
nn.Sequential(
    nn.Conv2d(384, 256, kernel_size=1, bias=False),
    nn.GroupNorm(32, 256),
    nn.SiLU(),
)
```

Camera projection is:

```python
nn.Sequential(
    nn.Conv2d(64, 256, kernel_size=1, bias=False),
    nn.GroupNorm(32, 256),
    nn.SiLU(),
)
```

Use the existing `transvision.dataset.transforms.depth_lss.LSSTransform`, not `DepthLSSTransform`. Do not call `.cuda()` or hard-code autocast device. The detector controls AMP.

- [ ] **Step 4: Run encoder tests**

Run: `python -m pytest tests/resilient_v2x/test_encoders.py -q`

Expected: all fake-component tests pass on CPU.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/resilient_v2x/encoders.py transvision/models/resilient_v2x/__init__.py tests/resilient_v2x/test_encoders.py
git commit -m "feat: add shared resilient v2x bev encoders"
```

---

### Task 6: 用薄编排器替换现有单文件 detector 原型

**Files:**

- Modify: `transvision/models/detectors/resilient_v2x.py`
- Create: `transvision/models/resilient_v2x/transfusion_bridge.py`
- Modify: `transvision/models/detectors/__init__.py`
- Create: `tests/resilient_v2x/test_detector.py`

**Interfaces:**

`ResilientFeatureBatch` is the validated 256/3/783-dimensional contract frozen by the
core plan; this task imports it rather than redefining it.

```python
@dataclass
class TransFusionForward:
    raw_outputs: object
    dense_logits: Tensor

@dataclass
class ResilientTrainingForward:
    features: ResilientFeatureBatch
    valid_index: Tensor
    head: TransFusionForward | None
    detection_losses: Mapping[str, Tensor]

@MODELS.register_module()
class ResilientV2XNet(Base3DDetector):
    def __init__(
        self,
        data_preprocessor: Mapping[str, object],
        lidar_encoder: Mapping[str, object],
        camera_encoder: Mapping[str, object],
        lidar_ptf: Mapping[str, object],
        camera_ptf: Mapping[str, object],
        branch_repair: Mapping[str, object],
        lidar_aggregator: Mapping[str, object],
        camera_aggregator: Mapping[str, object],
        router: Mapping[str, object],
        bbox_head: Mapping[str, object],
        protocol: Mapping[str, object],
        variant: Mapping[str, object],
        init_cfg: Mapping[str, object] | None,
    ) -> None

    def extract_resilient_features(
        self,
        student_inputs: Mapping[str, object],
        batch_data_samples: Sequence[Det3DDataSample],
    ) -> ResilientFeatureBatch

    def loss(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: SampleList,
    ) -> dict[str, Tensor]

    def forward_train_artifacts(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: SampleList,
        compute_detection_loss: bool,
    ) -> ResilientTrainingForward

    def forward_head_artifacts(
        self,
        features: ResilientFeatureBatch,
        batch_data_samples: SampleList,
        compute_detection_loss: bool,
    ) -> ResilientTrainingForward

    def predict(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: SampleList,
    ) -> SampleList

    def _forward(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: SampleList | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]

def forward_transfusion_once(
    head: TransFusionHead,
    fused_feature: Tensor,
    batch_data_samples: Sequence[Det3DDataSample],
) -> TransFusionForward

def transfusion_loss_from_forward(
    head: TransFusionHead,
    forward: TransFusionForward,
    batch_data_samples: Sequence[Det3DDataSample],
) -> dict[str, Tensor]
```

Both forward-artifact dataclasses and bridge functions live in
`transfusion_bridge.py`; `resilient_v2x.py` imports them before defining
`ResilientV2XNet`. The interface intentionally avoids a Python 3.10 forward
reference to a later class definition.

- [ ] **Step 1: Write thin-orchestration and all-invalid tests**

Use registry-built spies to assert order:

```python
assert calls == [
    "lidar_encode_valid",
    "camera_encode_valid",
    "geometry_align",
    "lidar_ptf_context",
    "camera_ptf_context",
    "repair_four_branches",
    "aggregate_lidar",
    "aggregate_camera",
    "route",
    "bbox_head",
]
```

Add tests that:

- all-invalid predict returns one empty `InstanceData` without calling bbox head;
- mixed batch passes only valid rows to bbox head and scatters predictions back;
- generic non-controlled loss returns differentiable zero for invalid rows;
- one typed `SampleInferenceDiagnostics` is attached to each
  `Det3DDataSample` under the fixed `resilient_diagnostics` field, with matching
  sample ID, four ordered branch records and the current DER
  descriptor/support/weights;
- constructor rejects missing protocol keys, variant keys, or implicit component config fields;
- deployed state dict contains no teacher-prefixed parameters.
- a spy TransFusion head receives exactly one `forward()` call per loss step；
- `dense_logits` is cloned before `loss_by_feat()` and remains bitwise unchanged after detection loss；
- distillation consumes that preserved dense heatmap, not decoded proposal scores and not a second head forward.
- `ResilientFeatureBatch` validates matching batch dimensions, fused `[B,256,Y,X]`, support `[B]`, weights `[B,3]`, descriptor `[B,783]`, and one diagnostics record per sample.
- the thin orchestrator passes the four `RepairedBranch.normalized_age` values
  unchanged into `DynamicExpertRouter.branch_normalized_age`; horizons
  `(0,1,2,3)` appear exactly as `(0,1/3,2/3,1)` in descriptor slice `[778:782]`,
  and an unsupported branch stays zero;
- `forward_train_artifacts(...,compute_detection_loss=True)` returns fused features, valid indices, preserved dense logits, and losses from one head forward；
- `forward_train_artifacts()` is exactly
  `extract_resilient_features()` followed by `forward_head_artifacts()`; Task 9 may place
  the distributed validity gate between those two calls without duplicating feature/head
  work；
- `compute_detection_loss=False` still performs one head forward for teacher dense logits but does not call `loss_by_feat`.
- tensor mode calls `_forward()` and returns `(fused,valid_index,dense_logits)`; an
  all-invalid batch returns a correctly shaped empty dense-logit tensor without calling the
  head, so `Base3DDetector` is concrete and `Runner.model(...,mode="tensor")` is testable.

- [ ] **Step 2: Run detector tests and observe the current prototype failure**

Run: `python -m pytest tests/resilient_v2x/test_detector.py -q`

Expected: FAIL; the existing detector does not implement the declared contract.

- [ ] **Step 3: Replace rather than patch the prototype**

Remove the nested prototype `PerceptionTrajectoryField` and `DynamicExpertRouting` classes. Import all algorithms from `transvision.models.resilient_v2x`. Do not mutate config dicts with `.pop()`; deep-copy component configs before building.
The detector does not own an age formula: it forwards
`CausalBranchRepair.normalized_age` directly to the router and asserts its shape,
finiteness and `[0,1]` range. Raw integer horizons never enter DER.

`forward_transfusion_once()` calls the head once, immediately clones
`raw_outputs[0][0]["dense_heatmap"]`, and returns both. This is required because the current
`TransFusionHead.loss_by_feat()` reaches `clip_sigmoid()` with an in-place `sigmoid_()`.
`transfusion_loss_from_forward()` may mutate only `raw_outputs`; the cloned logits are the
student/teacher distillation contract.

`loss()` is a thin projection of
`forward_train_artifacts(...,compute_detection_loss=True).detection_losses`. The
teacher/student wrapper calls the artifact method directly, so it receives fused BEV and
dense logits without triggering a second detector/head forward.

`_forward()` calls `forward_train_artifacts(...,compute_detection_loss=False)` and projects
only tensor values; it must not decode predictions or synthesize `Det3DDataSample` objects.

The batch subset logic must use stable indices:

```python
valid_index = feature_batch.overall_support.nonzero(as_tuple=False).flatten()
if valid_index.numel() == 0:
    return self._empty_predictions(batch_data_samples, feature_batch.diagnostics)
valid_feature = feature_batch.fused.index_select(0, valid_index)
valid_samples = [batch_data_samples[index] for index in valid_index.tolist()]
valid_metas = [sample.metainfo for sample in valid_samples]
valid_predictions = self.bbox_head.predict([valid_feature], valid_metas)
return self._scatter_predictions(
    batch_data_samples,
    valid_index,
    valid_predictions,
    feature_batch.diagnostics,
)
```

Both empty and non-empty predictions retain the same typed diagnostics object.
Attachment detaches tensors to CPU but does not serialize or downsample them;
the evaluation evidence layer is the sole durable writer.

- [ ] **Step 4: Run detector and core tests**

Run:

```bash
python -m pytest tests/resilient_v2x/test_detector.py tests/resilient_v2x/test_geometry.py tests/resilient_v2x/test_ptf.py tests/resilient_v2x/test_routing.py -q
python -m compileall -q transvision/models/detectors/resilient_v2x.py
```

Expected: all tests pass and compileall is silent.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/detectors/resilient_v2x.py transvision/models/resilient_v2x/transfusion_bridge.py transvision/models/detectors/__init__.py tests/resilient_v2x/test_detector.py
git commit -m "feat: replace resilient v2x detector prototype"
```

---

### Task 7: 固定 TransFusion head 和完整 resolved model base config

**Files:**

- Create: `configs/resilient_v2x/_base_/model.py`
- Create: `configs/resilient_v2x/_base_/dataset.py`
- Create: `configs/resilient_v2x/_base_/schedule.py`
- Create: `configs/resilient_v2x/_base_/runtime.py`
- Create: `tests/resilient_v2x/golden/resolved_model.json`
- Create: `tests/resilient_v2x/test_configs.py`

**Interfaces:**

```python
SEMANTIC_CONFIG_FORBIDDEN_KEYS = (
    "teacher_checkpoint",
    "teacher_checkpoint_sha256",
    "initialization_state",
    "work_dir",
    "attempt_id",
)

def normalize_for_json(config: Mapping[str, object]) -> dict[str, object]
def semantic_training_config_sha256(config: Config) -> str
```

- [ ] **Step 1: Write config parsing and golden tests**

```python
from mmengine.config import Config


def test_resolved_base_model_matches_golden() -> None:
    config = Config.fromfile("configs/resilient_v2x/_base_/model.py")
    resolved = normalize_for_json(config.model)
    golden = json.loads(
        Path("tests/resilient_v2x/golden/resolved_model.json").read_text()
    )
    assert resolved == golden


def test_head_contract_is_explicit() -> None:
    head = Config.fromfile(
        "configs/resilient_v2x/_base_/model.py"
    ).model["bbox_head"]
    assert head["type"] == "TransFusionHead"
    assert head["in_channels"] == 256
    assert head["hidden_channel"] == 128
    assert head["num_proposals"] == 300
    assert head["num_classes"] == 1
    assert head["auxiliary"] is True
    assert head["nms_kernel_size"] == 3
    assert head["test_cfg"]["nms_type"] is None
    assert head["bbox_coder"]["score_threshold"] == 0.0
```

Add assertions for every LiDAR, Camera, PTF, DER, TransFusion, range, loss, assigner, and decoder field fixed in sections 5, 6, 8, and 9 of the design.

- [ ] **Step 2: Run tests and confirm missing config**

Run: `python -m pytest tests/resilient_v2x/test_configs.py -q`

Expected: FAIL because the config files are absent.

- [ ] **Step 3: Implement complete base configs**

`model.py` must explicitly declare:

- PointPillars voxel/PFN/scatter/SECOND/SECONDFPN and 256-channel projection;
- ResNet-50 local pretrained artifact path, stage outputs, frozen/norm behavior;
- GeneralizedLSSFPN, `LSSTransform`, depth bins, image/feature sizes, bounds;
- both independent PTF configs and variant switches;
- both modality aggregators, router, and complete TransFusionHead;
- `protocol=dict(delta_t_ms=100,history_limit=3,alpha=0.9,train_delay_ms=(0,100,200,300),p_lidar=0.3,p_camera=0.3,detection_range=(0,-46.08,-3,92.16,46.08,1),controlled_training=True)`.

`dataset.py` uses `ResilientV2XSampler`, `ResilientV2XDataPreprocessor`, distinct train/val/test manifest fields, and never aliases test to val. `schedule.py` contains the explicit AdamW/AMP/accumulation/update scheduler contract. `runtime.py` disables WandB/network loggers and records local JSON/log artifacts only.

The base bare-detector data preprocessor explicitly uses `view_mode="student"`. Final
teacher/student-wrapper configs must override it to `"paired"` together with
`dataset.teacher_mode=True`; bare teacher, concat, and deployment configs keep `"student"`
with `teacher_mode=False`.
Resolved semantic configs contain only stable teacher role/training-plan references;
every `SEMANTIC_CONFIG_FORBIDDEN_KEYS` entry is absent recursively. Their canonical
hash can therefore be computed before any training realization. Runtime checkpoint,
initialization and attempt overlays are built only by Task 8/experiment orchestration
and do not change `semantic_training_config_sha256()`.

The golden JSON is generated only after human review of the resolved dict, committed with the config, and never regenerated automatically during tests.

- [ ] **Step 4: Run config tests**

Run: `python -m pytest tests/resilient_v2x/test_configs.py -q`

Expected: all resolved-dict assertions pass.

- [ ] **Step 5: Commit**

```bash
git add configs/resilient_v2x/_base_ tests/resilient_v2x/golden/resolved_model.json tests/resilient_v2x/test_configs.py
git commit -m "feat: define resilient v2x model configuration"
```

---

### Task 8: 实现 frozen teacher/student wrapper 和 checkpoint hash gate

**Files:**

- Create: `transvision/models/detectors/resilient_v2x_distiller.py`
- Create: `transvision/models/resilient_v2x/initialization.py`
- Modify: `transvision/models/detectors/__init__.py`
- Create: `tests/resilient_v2x/test_teacher_student.py`

**Interfaces:**

```python
class CheckpointIntegrityError(RuntimeError):
    """A checkpoint or initialization artifact failed strict verification."""

@dataclass(frozen=True)
class InitializationArtifact:
    schema_version: Literal[1]
    initialization_id: str
    recipe_id: str
    seed: Literal[0, 1, 2]
    semantic_config_sha256: str
    source_teacher_checkpoint_sha256: str | None
    state_relative_path: str
    state_file_sha256: str
    canonical_state_sha256: str
    tensor_manifest: tuple[Mapping[str, object], ...]
    content_sha256: str

def canonical_state_dict_sha256(
    state_dict: Mapping[str, Tensor],
) -> str
def build_initialization_artifact(
    semantic_model_config: Mapping[str, object],
    seed: Literal[0, 1, 2],
    output_dir: Path,
    accepted_teacher_checkpoint: Path | None,
    accepted_teacher_checkpoint_sha256: str | None,
) -> InitializationArtifact
def validate_initialization_artifact(
    manifest_path: Path,
) -> InitializationArtifact
def materialize_training_runtime_config(
    semantic_training_config: Config,
    initialization: InitializationArtifact,
    accepted_teacher_checkpoint: Path | None,
    accepted_teacher_checkpoint_sha256: str | None,
    attempt_dir: Path,
) -> Config

@MODELS.register_module()
class ResilientV2XTeacherStudent(BaseModel):
    def __init__(
        self,
        student: Mapping[str, object],
        teacher: Mapping[str, object],
        teacher_checkpoint: str,
        teacher_checkpoint_sha256: str,
        distillation: Mapping[str, float],
        initialize_student_from_teacher: bool,
        data_preprocessor: Mapping[str, object],
        init_cfg: Mapping[str, object] | None,
    ) -> None

    def forward(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample] | None = None,
        mode: Literal["loss", "predict", "tensor"] = "tensor",
    ) -> dict[str, Tensor] | SampleList | tuple[Tensor, Tensor, Tensor]

    def loss(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample],
    ) -> dict[str, Tensor]

    def predict(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample],
    ) -> SampleList

    def _forward(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]

    def state_dict(
        self,
        destination: MutableMapping[str, Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> OrderedDict[str, Tensor]
    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
    ) -> _IncompatibleKeys
    def student_trainable_parameters(self) -> Iterator[nn.Parameter]
    def train(self, mode: bool = True) -> "ResilientV2XTeacherStudent"
```

- [ ] **Step 1: Write strict-init, hash, frozen-grad, and no-distillation tests**

Tests must:

1. save a tiny teacher checkpoint;
2. compute its SHA-256;
3. build wrapper and assert every student tensor initially equals teacher tensor;
4. assert teacher is `eval`, no teacher parameter requires grad, and no teacher gradient exists after backward;
5. modify checkpoint bytes and expect `CheckpointIntegrityError`;
6. assert `variant=no_distillation` still strict-initializes student but returns only detection losses;
7. assert student deployment checkpoint keys have no `teacher.` prefix;
8. assert teacher and student augmentation matrices are element-wise equal before forward.
9. assert the optimizer parameter ID set is exactly the student trainable parameter ID set and contains no teacher ID.
10. call `wrapper.train()` and assert student is training while teacher remains in eval mode.
11. pass the preprocessor output through `train_step`, `val_step`, and `test_step` and assert
    the wrapper's `forward(...,mode=...)` dispatches without a `TypeError`;
12. assert `predict` and `tensor` mode read only `inputs["student_inputs"]`, never run the
    teacher, and return exactly the student's `SampleList` and tensor tuple;
13. assert an unknown mode raises `RuntimeError`, loss mode requires `data_samples`, and
    distillation-enabled loss requires a non-aliased `teacher_inputs`;
14. assert no-distillation loss does not call the teacher at all.
15. reject construction unless the configured data preprocessor uses `view_mode="paired"`.
16. build random-init artifacts twice for a teacher/concat baseline and obtain
    identical sorted tensor manifests/state hashes; changing seed, config, tensor
    dtype/shape/name/bytes changes the canonical state hash.
17. build a student initialization artifact from an accepted teacher, strict-load
    that stored student state in the child, and re-hash it before the first
    optimizer step; a missing/extra tensor or mutated state file fails.
18. prove semantic training config contains teacher role/training-plan reference
    but no checkpoint/attempt path or realized checkpoint hash. Changing accepted
    teacher checkpoint bytes changes initialization/realization/materialized
    training IDs and runtime command hash, but not `training_plan_id` or its
    semantic `resolved_config_sha256`.
19. verify the fixed `initialization-manifest-v1` content-hash vector:
    `content_sha256` is excluded from its own canonical payload, while mutation of
    any other field changes it; the enclosing evidence artifact manifest supplies
    the detached hash of the serialized file.
20. `initialization_id` is the `model-initialization-v1:` domain-separated digest
    of recipe, seed, semantic config and optional accepted-teacher hash before any
    output path exists. Inject crashes at state write, manifest write, commit
    marker, file/tree fsync and atomic publish; retry reuses one byte-identical
    complete bundle, removes only its own uncommitted staging tree and rejects a
    conflicting final bundle.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_teacher_student.py -q`

Expected: FAIL because the wrapper is absent.

- [ ] **Step 3: Implement hash-first strict loading**

Load order:

```python
actual_sha256 = sha256_file(Path(teacher_checkpoint))
if actual_sha256 != teacher_checkpoint_sha256:
    raise CheckpointIntegrityError("teacher checkpoint sha256 mismatch")
checkpoint = torch.load(teacher_checkpoint, map_location="cpu")
teacher.load_state_dict(checkpoint["state_dict"], strict=True)
student.load_state_dict(checkpoint["state_dict"], strict=True)
freeze_teacher(teacher)
```

The checked-in/resolved **semantic training config** stops before this constructor:
it contains the teacher role and teacher training-plan dependency but no realized
checkpoint path/hash, initialization state path, work directory, or attempt ID.
Its canonical hash is `TrainingPlan.resolved_config_sha256`. Only after an
accepted teacher (when required) and an initialization artifact exist may
`materialize_training_runtime_config()` inject their verified paths/hashes plus
the attempt directory. This runtime config is recorded in the command/training
manifest but is not re-hashed as the training plan.

For role `none`, `build_initialization_artifact()` seeds model construction and
canonicalizes state tensors by sorted key with explicit name/dtype/shape/
little-endian bytes. Before writing, it computes `initialization_id` as
`canonical_id("model-initialization-v1", {recipe_id,seed,
semantic_config_sha256,source_teacher_checkpoint_sha256})`. It writes state,
strict manifest and a bundle commit marker inside a unique sibling staging
directory under `output_dir`, fsyncs every file/tree level, then atomically
publishes without replacement to `<output_dir>/<initialization_id>/` and fsyncs
the parent. Retry re-verifies and reuses an exact complete bundle; it removes only
same-ID staging without a commit marker and never overwrites a conflict. The final
path is never partially populated, even though this transaction precedes
`AttemptStore` creation. For teacher-derived student roles, it first verifies
the accepted teacher checkpoint, applies the explicit strict key mapping, and
persists the resulting student initial state the same way. The child strict-loads
that state and recomputes `canonical_state_sha256` before optimizer creation;
this value is `TrainingRealization.model_initialization_sha256`.
`InitializationArtifact.content_sha256` is
`sha256(b"initialization-manifest-v1\\0" + canonical_json(payload_without_content_sha256))`;
it is not claimed to equal the serialized file hash. The later artifact manifest
hashes the full serialized initialization manifest and state file, eliminating
self-reference while preserving both semantic and byte identity.

Teacher forward must be inside `with torch.no_grad():` and consume only `teacher_inputs`.
The wrapper overrides `state_dict()`/`load_state_dict()` to expose only the student's deployment
state; the teacher is reloaded from its separately hashed checkpoint on build/resume. The
student-only optimizer constructor in Task 9 consumes
`student_trainable_parameters()` and asserts exact parameter-ID equality. Resume re-runs the
same hash and strict-load checks.

External-local baseline initialization parsers must emit the same canonical
tensor inventory into this transaction and return the shared
`InitializationArtifact`; adapters may not publish a separate initialization
bundle format or bypass its ID/commit marker.

Override `train(mode)` to call `self.student.train(mode)`, force
`self.teacher.eval()`, set the wrapper's own `training` flag, and return `self`; never call
the inherited recursive implementation that would put the teacher back into training mode.

`BaseModel.train_step/val_step/test_step` passes the preprocessor result as keyword
arguments named `inputs` and `data_samples`; therefore the wrapper implements the exact
dispatcher:

```python
def forward(self, inputs, data_samples=None, mode="tensor"):
    if mode == "loss":
        if data_samples is None:
            raise RuntimeError("loss mode requires data_samples")
        return self.loss(inputs, data_samples)
    if mode == "predict":
        if data_samples is None:
            raise RuntimeError("predict mode requires data_samples")
        return self.predict(inputs, data_samples)
    if mode == "tensor":
        return self._forward(inputs, data_samples)
    raise RuntimeError(f"unsupported forward mode: {mode}")
```

`predict()` delegates to `self.student.predict(inputs["student_inputs"],data_samples)`;
`_forward()` delegates to the student's tensor path with the same student-only input.
Neither method may read `teacher_inputs`. The training wrapper is therefore usable for
validation during student training, while deployment configs still construct only the
student detector.

The loss path is exactly:

```python
student_forward = self.student.forward_train_artifacts(
    inputs["student_inputs"],
    data_samples,
    compute_detection_loss=True,
)
with torch.no_grad():
    teacher_forward = self.teacher.forward_train_artifacts(
        inputs["teacher_inputs"],
        data_samples,
        compute_detection_loss=False,
    )
valid_index = student_forward.valid_index
distilled = distillation_losses(
    teacher_feature=teacher_forward.features.fused.index_select(0, valid_index),
    student_feature=student_forward.features.fused.index_select(0, valid_index),
    teacher_logits=teacher_forward.head.dense_logits.index_select(0, valid_index),
    student_logits=student_forward.head.dense_logits,
    temperature=self.temperature,
    lambda_feature=self.lambda_feature,
    lambda_logit=self.lambda_logit,
    valid_sample_mask=torch.ones(
        valid_index.numel(),
        dtype=torch.bool,
        device=valid_index.device,
    ),
)
```

Before dereferencing `head`, assert the local artifact contract is internally consistent
and both head artifacts are present. Task 9 inserts the distributed controlled/generic
validity gate before this dereference. Merge `student_forward.detection_losses` with the
two named distillation losses in a fixed key order; do not call either detector a second
time. When distillation is disabled, return the ordered detection-loss schema immediately
after the student forward and never build or call a teacher forward.

- [ ] **Step 4: Run teacher/student tests**

Run: `python -m pytest tests/resilient_v2x/test_teacher_student.py tests/resilient_v2x/test_distillation.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/detectors/resilient_v2x_distiller.py transvision/models/resilient_v2x/initialization.py transvision/models/detectors/__init__.py tests/resilient_v2x/test_teacher_student.py
git commit -m "feat: add frozen teacher student training"
```

---

### Task 9: 实现 fail-fast AMP wrapper、update-step scheduler 和 DDP valid-count

**Files:**

- Create: `transvision/models/hooks/resilient_v2x.py`
- Modify: `transvision/models/hooks/__init__.py`
- Create: `transvision/models/opt/resilient_v2x.py`
- Modify: `transvision/models/opt/__init__.py`
- Create: `transvision/models/resilient_v2x/training.py`
- Modify: `transvision/models/resilient_v2x/__init__.py`
- Create: `transvision/models/wrappers/__init__.py`
- Create: `transvision/models/wrappers/resilient_v2x.py`
- Modify: `transvision/models/detectors/resilient_v2x.py`
- Modify: `transvision/models/detectors/resilient_v2x_distiller.py`
- Modify: `configs/resilient_v2x/_base_/schedule.py`
- Modify: `configs/resilient_v2x/_base_/runtime.py`
- Create: `tests/resilient_v2x/test_training_protocol.py`

**Interfaces:**

```python
DETECTION_LOG_KEYS = (
    "loss_heatmap",
    "layer_-1_loss_cls",
    "layer_-1_loss_bbox",
    "matched_ious",
)
DISTILLATION_LOG_KEYS = (
    "loss_distill_feature",
    "loss_distill_bernoulli",
)
SELECTION_AP_KEY = "selection/BEV_AP_0.7"
SELECTION_LOSS_KEY = "selection/detection_loss"
SELECTION_METADATA = {
    "selection/split": "val",
    "selection/condition": "Full",
    "selection/latency_ms": 0,
}

@dataclass(frozen=True)
class BatchValidityDecision:
    local_valid_count: int
    local_batch_size: int
    global_valid_count: int
    global_batch_size: int
    controlled: bool
    should_compute_local_loss: bool
    should_update_optimizer: bool

class ValidityAwareLosses(OrderedDict[str, Tensor]):
    decision: BatchValidityDecision

@MODEL_WRAPPERS.register_module()
class ResilientDistributedDataParallel(MMDistributedDataParallel):
    def train_step(
        self,
        data: Mapping[str, object],
        optim_wrapper: OptimWrapper,
    ) -> Mapping[str, Tensor]

@OPTIM_WRAPPERS.register_module()
class FailFastAmpOptimWrapper(AmpOptimWrapper):
    last_update_performed: bool
    optimizer_update_count: int

    def update_params(
        self,
        loss: Tensor,
        step_kwargs: Mapping[str, object] | None = None,
        zero_kwargs: Mapping[str, object] | None = None,
    ) -> None

@HOOKS.register_module()
class UpdateStepParamSchedulerHook(Hook):
    def __init__(
        self,
        epochs: int,
        global_batch_size: int,
        base_lr: float,
        warmup_updates: int,
        warmup_start_ratio: float,
        min_lr: float,
    ) -> None
    def before_train(self, runner: Runner) -> None
    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Mapping[str, object],
        outputs: Mapping[str, Tensor],
    ) -> None

@HOOKS.register_module()
class ResilientCheckpointSelectorHook(Hook):
    def after_val_epoch(
        self,
        runner: Runner,
        metrics: Mapping[str, object],
    ) -> None

@PARAM_SCHEDULERS.register_module()
class WarmupCosineUpdateScheduler(_ParamScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        warmup_updates: int,
        warmup_start_ratio: float,
        total_updates: int,
        min_lr: float,
    ) -> None
    def lr_at_update(self, update: int) -> float

@LOOPS.register_module()
class ResilientSelectionValLoop(ValLoop):
    def run_iter(self, idx: int, data_batch: Mapping[str, object]) -> None
    def run(self) -> Mapping[str, object]

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class StudentOnlyOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def __call__(self, model: nn.Module) -> OptimWrapper

class ValidityAwareLossModel(Protocol):
    def loss_with_validity(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample],
    ) -> tuple[OrderedDict[str, Tensor], BatchValidityDecision]

class SelectionLossDetector(Protocol):
    def forward_selection_detection_loss(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample],
    ) -> tuple[Tensor, int]

def global_valid_batch_guard(
    local_valid_count: int,
    local_batch_size: int,
    controlled: bool,
) -> BatchValidityDecision

def fixed_zero_loss_schema(
    student_parameters: Iterable[nn.Parameter],
    distillation_enabled: bool,
) -> OrderedDict[str, Tensor]

def normalize_loss_schema(
    losses: Mapping[str, Tensor],
    distillation_enabled: bool,
) -> OrderedDict[str, Tensor]

def selection_detection_loss(
    model: nn.Module,
    inputs: Mapping[str, object],
    data_samples: Sequence[Det3DDataSample],
) -> tuple[Tensor, int]

def resilient_train_step(
    forward_model: nn.Module,
    state_model: BaseModel,
    data: Mapping[str, object],
    optim_wrapper: OptimWrapper,
) -> Mapping[str, Tensor]

def accumulation_steps(world_size: int) -> int
def updates_per_epoch(history_eligible_train_count: int, global_batch_size: int) -> int
```

- [ ] **Step 1: Write exact schedule and DDP tests**

```python
def test_accumulation_preserves_global_batch_four() -> None:
    assert accumulation_steps(1) == 4
    assert accumulation_steps(2) == 2
    assert accumulation_steps(4) == 1
    with pytest.raises(ValueError, match="world size"):
        accumulation_steps(3)


def test_update_counts_drop_partial_global_batch() -> None:
    assert updates_per_epoch(13, 4) == 3
    assert updates_per_epoch(12, 4) == 3


def test_lr_boundaries() -> None:
    scheduler = build_scheduler(total_updates=1000)
    assert scheduler.lr_at_update(1) == pytest.approx(2e-7)
    assert scheduler.lr_at_update(500) == pytest.approx(2e-4)
    assert scheduler.lr_at_update(1000) == pytest.approx(2e-6)
```

Add real two-rank CPU/Gloo tests, launched through `torch.multiprocessing.spawn` with a
temporary file init method and a bounded join, for both policy modes:

- controlled mode with rank 0 reporting `(valid=0,batch=1)` and rank 1 reporting
  `(valid=1,batch=1)` raises `controlled_batch_contains_invalid_sample` on both ranks after
  the same single all-reduce; neither rank enters the head, teacher, backward, optimizer,
  scaler, or scheduler；
- generic mode for that same split batch lets the valid rank compute the real loss and gives
  the zero rank a differentiable zero connected to every student parameter；
- the generic valid and zero ranks return exactly the same ordered keys:
  `DETECTION_LOG_KEYS + DISTILLATION_LOG_KEYS` when distillation is enabled and
  `DETECTION_LOG_KEYS` otherwise；
- both ranks complete backward/DDP collectives without hanging; before reduction only the
  valid rank contributes a non-zero loss, and after reduction both ranks observe the same
  synchronized gradients；
- the real two-rank Runner path wraps the module in
  `ResilientDistributedDataParallel`; its outer `train_step()` calls
  `forward_model=self(...,mode="loss")` so PyTorch DDP forward/reducer hooks run,
  while preprocessing and loss parsing use `self.module` exactly once. A spy that
  would fail if standard `MMDistributedDataParallel.train_step()` or a direct
  `self.module(...)` forward is used remains untouched；
- a generic batch with global valid count zero returns the same fixed ordered schema but sets
  `should_update_optimizer=False`; no backward, optimizer, scaler, or scheduler event occurs；
- the validity guard always all-reduces one two-element integer tensor
  `[local_valid_count,local_batch_size]`, so all ranks execute one identical collective.

Use local fakes for the remaining event-order and protocol tests:

- `fixed_zero_loss_schema()` connects every emitted loss tensor to every student parameter,
  while `normalize_loss_schema()` rejects missing/extra semantic keys and always emits the
  canonical order regardless of input mapping order；
- the detector and teacher/student wrapper call `extract_resilient_features()`, then the
  distributed guard, then `forward_head_artifacts()`; a local-zero generic rank never calls
  the head or teacher；
- runtime/static conformance fixtures prove Resilient, teacher/student and concat
  implementations expose the exact `ValidityAwareLossModel` and
  `SelectionLossDetector` parameter contract (`Mapping[str, object]`,
  `Sequence[Det3DDataSample]`) without narrowing it to a TypedDict/list subtype;
- non-finite loss or gradient terminates rather than skipping an update；
- an accumulation-boundary event trace is exactly
  `scaled_backward → unscale → finite_check → clip_35 → optimizer_step → scaler_update → zero_grad`；
- a non-boundary microbatch performs only `scaled_backward`, and a `skip_update=True` batch
  performs none of those events and does not advance the accumulation counter；
- the scheduler reads `N_train=history_eligible_train_count` from the loaded
  temporal manifest, independently recounts emitted train rows containing all
  four history positions, derives `updates_per_epoch=floor(N_train/4)` and
  `total_updates=50*floor(N_train/4)`, and rejects a sampler length or configured
  global batch size inconsistent with that manifest; it never substitutes the
  raw split count 4,813 for `N_train`；
- controlled preflight rejects `50*floor(N_train/4) <= 500` instead of changing
  the 500-update warmup；
- scheduler initialization also rejects per-device batch size other than one or an
  accumulation count different from `accumulation_steps(world_size)`；
- immediately before the first optimizer update every parameter group has
  `lr_at_update(1)==2e-7`; immediately after completed update `u`, the optimizer holds the LR
  for update `u+1`；
- resuming with 500 completed optimizer updates restores the counter and injects
  `lr_at_update(501)` without replaying or skipping a scheduler step；
- the scheduler advances only after `last_update_performed=True`, and standard MMEngine
  `ParamSchedulerHook` is absent from every Resilient runtime config；
- `ResilientSelectionValLoop` emits the exact keys `selection/BEV_AP_0.7` and
  `selection/detection_loss` plus exact `val/Full/0 ms` metadata, with detection loss
  globally aggregated by valid-sample count；
- the selection-loss path unwraps DDP and the teacher/student wrapper, runs only the student
  detector's `forward_selection_detection_loss()` under `no_grad`, and never calls teacher
  or distillation；
- a selection epoch with global valid count zero fails closed before division or checkpoint
  comparison；
- degraded-condition metrics, missing metadata, alternate metric spellings, NaN/Inf, and
  test-split metrics are rejected by the checkpoint selector；
- the selector compares `(BEV_AP_0.7,-detection_loss,-epoch)` lexicographically and writes
  checkpoint path, epoch, both values, and checkpoint SHA-256 atomically；
- the student-only optimizer constructor rejects any parameter set that differs from
  `model.student_trainable_parameters()`.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_training_protocol.py -q`

Expected: FAIL because scheduler/hook APIs are absent.

- [ ] **Step 3: Implement update-indexed learning rate**

For update `u` in one-based indexing:

```python
if u <= warmup_updates:
    progress = (u - 1) / max(warmup_updates - 1, 1)
    lr = base_lr * (
        warmup_start_ratio + progress * (1.0 - warmup_start_ratio)
    )
else:
    progress = (u - warmup_updates) / (total_updates - warmup_updates)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = min_lr + (base_lr - min_lr) * cosine
```

Both `ResilientV2XNet` and `ResilientV2XTeacherStudent` implement
`loss_with_validity()` and make their public `loss()` a projection of the returned ordered
loss mapping. They first call `extract_resilient_features()`, derive the local valid count
from `overall_support`, and call `global_valid_batch_guard()` before any detection or
distillation head work. Only a permitted rank then calls `forward_head_artifacts()`;
Task 8's convenience `forward_train_artifacts()` is not used by this guarded path.
The `controlled` argument comes only from the required
`protocol.controlled_training` constructor field; controlled configs set true and generic
unit/smoke configs must explicitly set false.

The guard all-reduces exactly one device-local int64 tensor containing
`[local_valid_count,local_batch_size]`. In controlled mode,
`global_valid_count != global_batch_size` raises
`controlled_batch_contains_invalid_sample` on every rank, even if the global valid count is
positive. In generic mode, local-zero/global-positive returns
`fixed_zero_loss_schema()` and still participates in DDP backward, while global-zero returns
the same schema with `should_update_optimizer=False`. No rank raises merely because the
generic global count is zero.

`normalize_loss_schema()` accepts only the exact TransFusion key set in
`DETECTION_LOG_KEYS`, reorders it canonically, and appends the two exact distillation keys
only when enabled.
`fixed_zero_loss_schema()` first builds
`connected_zero=sum(parameter.sum()*0 for parameter in student_parameters)` and derives
every ordered value from it. Thus valid and zero ranks enter MMEngine loss parsing and DDP
with identical key order and autograd participation.
`loss()` returns a `ValidityAwareLosses` object: numerically it is the same fixed
ordered tensor mapping required by BaseModel, but it carries the immutable
`BatchValidityDecision` for the custom outer train step. The decision is never
inserted as a fake numeric loss key.

For the wrapper, the permitted path performs the student head once, then—only when
distillation is enabled—extracts the clean teacher features under `torch.no_grad()` and
calls the teacher head once. It indexes both feature/logit artifacts with the student's
stable valid indices. A zero local rank runs neither head and never dereferences a missing
head artifact.

`FailFastAmpOptimWrapper.update_params()` owns the entire AMP update boundary. It rejects a
non-finite loss before backward. It pops the reserved boolean `skip_update` from
`step_kwargs` rather than forwarding it to `optimizer.step()`. When true, it sets
`last_update_performed=False`, performs no backward/update event, and does not advance the
inner accumulation counter. Otherwise it runs scaled backward; on an accumulation boundary
it unscales, checks every student gradient for finiteness, clips global norm to 35, performs
the optimizer and scaler updates, zeroes gradients, increments
`optimizer_update_count`, and records `last_update_performed=True`. A GradScaler overflow is
a runtime error, never a silently skipped update.

`resilient_train_step()` applies `state_model.data_preprocessor` exactly once,
enters `optim_wrapper.optim_context(forward_model)`, calls
`forward_model(**processed, mode="loss")`, requires `ValidityAwareLosses`, parses/
logs the fixed schema through `state_model`, and passes
`step_kwargs={"skip_update": not decision.should_update_optimizer}` to the wrapper. Both the
detector and teacher/student wrapper override `train_step()` only to delegate to this
function with `(forward_model=self,state_model=self)` for a single process;
`val_step()` and `test_step()` retain the `BaseModel.forward` dispatcher.

For distributed training, the runtime config imports and selects
`ResilientDistributedDataParallel` as `model_wrapper_cfg`. Its `train_step()`
delegates with `(forward_model=self,state_model=self.module)`, so the public
three-mode model dispatcher remains unchanged while DDP's forward preparation and
gradient reducer are active. It rejects a module that does not return
`ValidityAwareLosses`. Standard `MMDistributedDataParallel.train_step()` is never
reachable in a Resilient controlled job.

`UpdateStepParamSchedulerHook.before_train()` reads
`TemporalManifest.history_eligible_train_count` from the actual train dataset, checks that
the sampler's global truncated length equals
`4*floor(history_eligible_train_count/4)`, derives
`U=floor(history_eligible_train_count/global_batch_size)` and
`total_updates=epochs*U`, and requires `total_updates>warmup_updates`. The base schedule
config leaves `total_updates` unset; this hook also verifies per-device batch size one and
the runtime-world-size accumulation value from `accumulation_steps()`, then constructs or
injects the scheduler from the manifest-derived value. It registers the scheduler with the
Runner so its state is checkpointed, and `FailFastAmpOptimWrapper.state_dict()` persists
`optimizer_update_count` alongside optimizer/scaler state.

The scheduler is one-based: before a fresh run the hook sets every optimizer group to
`lr_at_update(1)`. If a checkpoint says `u` optimizer updates are complete, restore and
cross-check the optimizer/wrapper/scheduler counters, then set `lr_at_update(u+1)`. After a
successful update `u`, `after_train_iter()` prepares `lr_at_update(u+1)` (unless `u` is the
final update) and clears `last_update_performed`; it does nothing otherwise. This makes
resume at update 500 prepare update 501 exactly. All Resilient configs remove the standard
per-iteration `ParamSchedulerHook`.

`ResilientSelectionValLoop` preprocesses each validation batch once, performs the ordinary
student prediction/evaluator path, and under `torch.no_grad()` also calls
`selection_detection_loss()`. That helper unwraps DDP and, when present, the
teacher/student wrapper. For a wrapper it passes `inputs["student_inputs"]`; for a bare
detector it passes the direct `ResilientModelInputs`. It then calls the resulting detector's
`forward_selection_detection_loss()`. `ResilientV2XNet` implements that method as only
`extract_resilient_features()` plus
`forward_head_artifacts(...,compute_detection_loss=True)`; Task 10's independent detector
implements the same protocol with its own extractor. The method defines detection loss as
the sum of the three `loss_*` detection values (never `matched_ious`), multiplies the local
mean by the local valid count, and returns that numerator plus the count. The loop
all-reduces both, divides only after aggregation, and publishes:

```python
{
    "selection/BEV_AP_0.7": bev_ap_07,
    "selection/detection_loss": global_loss_sum / global_valid_count,
    "selection/split": "val",
    "selection/condition": "Full",
    "selection/latency_ms": 0,
}
```

`ResilientCheckpointSelectorHook` requires exactly those selection keys and metadata from
the resolved selection dataloader, rejects non-finite values, and never falls back to
degraded or test metrics. It selects greater AP, then lower detection loss, then lower epoch
number, and atomically writes `selected_checkpoint.json` with the selected checkpoint's
SHA-256; ordinary MMEngine single-metric best-checkpoint logic is disabled.

`StudentOnlyOptimWrapperConstructor` unwraps DDP if present, takes the iterator exposed by
the teacher/student wrapper, verifies uniqueness and `requires_grad=True`, and constructs
AdamW from exactly those tensors. A post-build assertion compares optimizer and student
parameter IDs as sets.

- [ ] **Step 4: Run training protocol tests**

Run: `python -m pytest tests/resilient_v2x/test_training_protocol.py -q`

Expected: all local fakes and the bounded two-rank CPU/Gloo tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/hooks/resilient_v2x.py transvision/models/hooks/__init__.py transvision/models/opt/resilient_v2x.py transvision/models/opt/__init__.py transvision/models/resilient_v2x/training.py transvision/models/resilient_v2x/__init__.py transvision/models/wrappers/__init__.py transvision/models/wrappers/resilient_v2x.py transvision/models/detectors/resilient_v2x.py transvision/models/detectors/resilient_v2x_distiller.py configs/resilient_v2x/_base_/schedule.py configs/resilient_v2x/_base_/runtime.py tests/resilient_v2x/test_training_protocol.py
git commit -m "feat: enforce resilient v2x training updates"
```

---

### Task 10: 增加 teacher、八个 student variants 和 capacity-matched concat

**Files:**

- Create: `transvision/models/resilient_v2x/concat_baseline.py`
- Modify: `transvision/models/resilient_v2x/__init__.py`
- Create: `transvision/models/detectors/resilient_v2x_concat.py`
- Modify: `transvision/models/detectors/__init__.py`
- Create: `configs/resilient_v2x/teacher.py`
- Create: `configs/resilient_v2x/student.py`
- Create: `configs/resilient_v2x/concat_capacity_matched.py`
- Create: `configs/resilient_v2x/capacity_match.json`
- Create: `configs/resilient_v2x/ablations/no_ptf.py`
- Create: `configs/resilient_v2x/ablations/linear_ptf.py`
- Create: `configs/resilient_v2x/ablations/static_experts.py`
- Create: `configs/resilient_v2x/ablations/uniform_gate.py`
- Create: `configs/resilient_v2x/ablations/no_reliability.py`
- Create: `configs/resilient_v2x/ablations/no_delay_metadata.py`
- Create: `configs/resilient_v2x/ablations/no_distillation.py`
- Create: `tools/resilient_v2x/match_capacity.py`
- Modify: `tests/resilient_v2x/test_configs.py`
- Create: `tests/resilient_v2x/test_capacity_match.py`
- Create: `tests/resilient_v2x/test_concat_detector.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CapacityMatch:
    width: int
    full_parameter_count: int
    candidate_parameter_count: int
    absolute_difference: int
    relative_error: float
    max_relative_error: float
    candidate_counts: tuple[tuple[int, int], ...]

@dataclass
class ConcatFeatureBatch:
    branch_features: Tensor
    branch_support: Tensor
    fused: Tensor
    overall_support: Tensor
    diagnostics: tuple[SampleInferenceDiagnostics, ...]

@dataclass
class ConcatTrainingForward:
    features: ConcatFeatureBatch
    valid_index: Tensor
    head: TransFusionForward | None
    detection_losses: Mapping[str, Tensor]

@MODELS.register_module()
class CapacityMatchedConcat(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int) -> None
    def forward(self, branch_features: Tensor, branch_support: Tensor) -> Tensor

@MODELS.register_module()
class CapacityMatchedConcatNet(Base3DDetector):
    def __init__(
        self,
        data_preprocessor: Mapping[str, object],
        lidar_encoder: Mapping[str, object],
        camera_encoder: Mapping[str, object],
        fusion_adapter: Mapping[str, object],
        bbox_head: Mapping[str, object],
        protocol: Mapping[str, object],
        init_cfg: Mapping[str, object] | None,
    ) -> None

    def extract_concat_features(
        self,
        student_inputs: Mapping[str, object],
        batch_data_samples: SampleList,
    ) -> ConcatFeatureBatch

    def forward_head_artifacts(
        self,
        features: ConcatFeatureBatch,
        batch_data_samples: SampleList,
        compute_detection_loss: bool,
    ) -> ConcatTrainingForward

    def loss_with_validity(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: Sequence[Det3DDataSample],
    ) -> tuple[OrderedDict[str, Tensor], BatchValidityDecision]

    def loss(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: SampleList,
    ) -> dict[str, Tensor]

    def predict(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: SampleList,
    ) -> SampleList

    def _forward(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: SampleList | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]

    def forward_selection_detection_loss(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: Sequence[Det3DDataSample],
    ) -> tuple[Tensor, int]

    def train_step(
        self,
        data: Mapping[str, object],
        optim_wrapper: OptimWrapper,
    ) -> Mapping[str, Tensor]

def trainable_parameter_count(module: nn.Module) -> int
def select_capacity_width(
    full_parameter_count: int,
    count_for_width: Callable[[int], int],
    candidates: Sequence[int],
    max_relative_error: float,
) -> CapacityMatch
```

- [ ] **Step 1: Write topology, tie-break, and variant-shape tests**

```python
def test_capacity_search_uses_absolute_relative_error_and_smaller_tie() -> None:
    counts = {32: 900, 64: 1100, 96: 1100}
    match = select_capacity_width(
        full_parameter_count=1000,
        count_for_width=counts.__getitem__,
        candidates=(32, 64, 96),
        max_relative_error=0.11,
    )
    assert match.width == 32
    assert match.relative_error == pytest.approx(0.1)


def test_all_student_variants_have_identical_state_shapes() -> None:
    configs = load_student_variant_configs()
    shapes = [state_shapes(MODELS.build(config.model.student)) for config in configs]
    assert all(shape == shapes[0] for shape in shapes[1:])
```

Add tests that candidate widths are exactly `range(32,2049,32)`, error over 1% fails,
`CapacityMatch` JSON contains exactly the declared fields and matches the hash pinned by the
concat config, concat input is 1024 channels, and variant switches match the design table.

Add independent-detector tests that:

- `MODELS.build()` constructs `CapacityMatchedConcatNet` directly from
  `concat_capacity_matched.py`; the resolved config has no teacher/student wrapper and no
  teacher checkpoint；
- its constructor, module tree, named parameters, and state dict contain no PTF,
  `branch_repair`, modality aggregator, router/DER, reliability descriptor, or teacher；
- spies for PTF, repair, aggregation, and routing remain uncalled during loss, predict, and
  tensor mode；
- it uses the same causal `selected_flat_index` and geometry alignment contract as the full
  detector, gathers only the latest supported feature for each branch, and represents an
  unsupported branch as support=false plus an exact zero feature；
- every prediction attaches a typed diagnostics record built from the same four
  `BranchSelection` values: source/support/observed/propagated fields are real,
  D/Q/gamma are null with explicit no-PTF semantics, and DER
  descriptor/support/weights are null with
  `not_applicable_reason="capacity_concat_has_no_der"`；
- `branch_features` is `[B,4,256,Y,X]`, `branch_support` is `[B,4]`, the adapter receives
  exactly 1024 concatenated channels, and the head receives `[B_valid,256,Y,X]`；
- loss, predict, and tensor mode work for mixed/all-invalid synthetic batches, and controlled
  training uses Task 9's same distributed validity gate and fail-fast optimizer path；
- a fake-component `Runner.from_cfg()` build succeeds for the independent concat config；
- the capacity tool counts every trainable parameter in the deployed full student and in
  the independent concat detector, including shared encoders and detection head, while
  excluding frozen/non-deployed teacher parameters.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_capacity_match.py tests/resilient_v2x/test_concat_detector.py tests/resilient_v2x/test_configs.py -q`

Expected: FAIL because configs and matcher are absent.

- [ ] **Step 3: Implement fixed concat topology and complete configs**

`CapacityMatchedConcatNet` is an independent detector class, not a
`ResilientV2XNet` variant and not a `ResilientV2XTeacherStudent` wrapper. Its resolved config
contains only the data preprocessor, shared LiDAR/Camera encoders, concat adapter,
TransFusion head, protocol, and `init_cfg`. It must not construct dormant PTF, repair,
aggregator, router, descriptor, distillation, or teacher modules.

The detector consumes the same metadata-first history contract. It uses the same causal
selections and geometry alignment, gathers the newest supported selected feature for
`(lidar_ego,lidar_rsu,camera_ego,camera_rsu)`, fixes temporal displacement `D=0` and
confidence `Q=1`, and zero-fills only unsupported branches. It stacks those features as
`[B,4,256,Y,X]`; `CapacityMatchedConcat.forward()` validates the boolean `[B,4]` support,
masks unsupported features again, flattens in canonical branch order to 1024 channels, and
applies this topology:

```python
self.adapter = nn.Sequential(
    nn.Conv2d(1024, width, kernel_size=1, bias=False),
    nn.GroupNorm(32, width),
    nn.SiLU(),
    DepthwiseSeparableResidualBlock(width),
    DepthwiseSeparableResidualBlock(width),
    nn.Conv2d(width, 256, kernel_size=1, bias=False),
    nn.GroupNorm(32, 256),
    nn.SiLU(),
)
```

Gather uses `safe_index=selected_flat_index.clamp_min(0)` followed by the exact support mask;
`-1` is never allowed to index the final history element. The adapter constructor rejects
anything other than `input_channels=1024`, `output_channels=256`, and a positive
`hidden_channels` divisible by 32. After the adapter, it multiplies by
`branch_support.any(dim=1)` so an all-unsupported row remains exactly zero even after learned
normalization affine terms.

The detector reuses `forward_transfusion_once()` and `transfusion_loss_from_forward()` but
does not import or instantiate `ResilientV2XNet`. Its `loss_with_validity()` uses Task 9's
guard before the head, its `train_step()` delegates to `resilient_train_step()`, and its
predict/all-invalid/tensor behavior matches the thin detector contract.
It constructs
`SampleInferenceDiagnostics(schema_version=1,method="capacity_concat",...)`
from the exact selected inputs and attaches it to every empty or non-empty
prediction. Algorithm-specific PTF/DER fields remain schema-valid explicit nulls,
never synthesized tensors.

`CapacityMatch` validates positive counts, strictly increasing unique candidate widths,
non-negative differences, finite error, the chosen candidate's presence in
`candidate_counts`, and:

```python
absolute_difference == abs(candidate_parameter_count - full_parameter_count)
math.isclose(
    relative_error,
    absolute_difference / full_parameter_count,
    rel_tol=0.0,
    abs_tol=1e-15,
)
```

`match_capacity.py` builds the deployed full `student` detector without its teacher wrapper,
then builds the entire independent concat detector for every
`width in range(32,2049,32)`. It calls `trainable_parameter_count()` on those complete
models, chooses `(absolute_difference,width)` lexicographically, records all width/count
pairs in `CapacityMatch`, and fails if the best `relative_error > 0.01`. The reviewed width
and both parameter totals are copied into `concat_capacity_matched.py` and a hashed capacity
match artifact. A separate pre-training `match_capacity.py --verify` gate rebuilds and
verifies both counts, exits, and records its digest; the actual concat training process then
constructs only `CapacityMatchedConcatNet`, never a dormant full detector or teacher.

All student variants instantiate identical modules. Only these behavior fields change:

| config | field/value |
|---|---|
| full | nonlinear, dynamic, reliability on, delay metadata on, distillation on |
| no PTF | `variant.ptf_mode="disabled"` |
| linear PTF | `variant.ptf_mode="linear"` |
| static experts | `variant.routing_mode="static"` |
| uniform gate | `variant.routing_mode="uniform"` |
| no reliability | `variant.use_reliability=False` |
| no delay metadata | `variant.use_delay_metadata=False` |
| no distillation | `distillation.enabled=False`, initialization role remains `initialization_only` |

Teacher config uses synchronized/full/0 ms inputs and no teacher wrapper. Student configs use
50 epochs, AdamW fields, accumulation derived from world size, AMP, clip 35, validation
every epoch, and Full@0 ms BEV AP@0.7 checkpoint rule.
`concat_capacity_matched.py` builds bare `CapacityMatchedConcatNet`, sets
`teacher_checkpoint_role="none"`, and uses that same complete controlled training and
selection protocol without any teacher dependency.

- [ ] **Step 4: Resolve and test every config**

Run:

```bash
python -m pytest tests/resilient_v2x/test_configs.py tests/resilient_v2x/test_capacity_match.py tests/resilient_v2x/test_concat_detector.py -q
for cfg in configs/resilient_v2x/teacher.py configs/resilient_v2x/student.py configs/resilient_v2x/concat_capacity_matched.py configs/resilient_v2x/ablations/*.py; do python -c 'import sys; from mmengine.config import Config; Config.fromfile(sys.argv[1])' "$cfg"; done
```

Expected: tests pass and every config resolves with exit code 0.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/resilient_v2x/concat_baseline.py transvision/models/resilient_v2x/__init__.py transvision/models/detectors/resilient_v2x_concat.py transvision/models/detectors/__init__.py configs/resilient_v2x tools/resilient_v2x/match_capacity.py tests/resilient_v2x/test_configs.py tests/resilient_v2x/test_capacity_match.py tests/resilient_v2x/test_concat_detector.py
git commit -m "feat: add resilient v2x training variants"
```

---

### Task 11: Synthetic Runner build、one-batch loss/predict 和运行入口

**Files:**

- Create: `configs/resilient_v2x/_base_/synthetic.py`
- Create: `tests/resilient_v2x/fixtures/synthetic_manifest.json`
- Create: `tests/resilient_v2x/test_integration.py`
- Create: `scripts/train_resilient_v2x_teacher.sh`
- Create: `scripts/train_resilient_v2x_student.sh`
- Create: `scripts/evaluate_resilient_v2x.sh`
- Create: `docs/resilient_v2x/training.md`

**Interfaces:**

No new Python production API is introduced. The three shell entrypoints expose
only their documented fail-closed argument contracts; synthetic helpers remain
test-only and are not exported.

- [ ] **Step 1: Write Runner build and one-batch tests**

Tests must use fake registered encoders/head, real core PTF/routing/distillation, and the synthetic manifest:

```python
@pytest.mark.parametrize(
    "config_path",
    [
        "configs/resilient_v2x/teacher.py",
        "configs/resilient_v2x/student.py",
        "configs/resilient_v2x/concat_capacity_matched.py",
        "configs/resilient_v2x/ablations/no_ptf.py",
        "configs/resilient_v2x/ablations/linear_ptf.py",
        "configs/resilient_v2x/ablations/static_experts.py",
        "configs/resilient_v2x/ablations/uniform_gate.py",
        "configs/resilient_v2x/ablations/no_reliability.py",
        "configs/resilient_v2x/ablations/no_delay_metadata.py",
        "configs/resilient_v2x/ablations/no_distillation.py",
    ],
)
def test_runner_builds_every_config(config_path: str) -> None:
    config = synthetic_override(Config.fromfile(config_path))
    assert config.train_dataloader.dataset.allow_fixture_manifest is True
    assert config.model.protocol.controlled_training is False
    runner = Runner.from_cfg(config)
    assert runner.model is not None


def test_synthetic_batch_loss_and_predict_are_finite() -> None:
    runner = build_synthetic_runner("configs/resilient_v2x/student.py")
    batch = next(iter(runner.train_dataloader))
    losses = runner.model.train_step(batch, runner.optim_wrapper)
    assert all(torch.isfinite(value).all() for value in tensor_values(losses))
    predictions = runner.model.test_step(batch)
    assert len(predictions) == batch_size(batch)
```

Also invoke `scripts/evaluate_resilient_v2x.sh` with argument-parser fakes:
`--diagnostic --split val` may delegate once to `tools/test.py`, while `test`,
missing `--diagnostic`, any controlled/evidence flag, or an evidence-root output
path must fail before the child command is built.

- [ ] **Step 2: Run integration tests and fix only genuine integration gaps**

Run: `python -m pytest tests/resilient_v2x/test_integration.py -q`

Expected before integration wiring: FAIL at the first unresolved registry/pipeline contract, not at missing real data.

- [ ] **Step 3: Add shell entrypoints with fail-closed arguments**

Each script uses `set -euo pipefail`, resolves repository root from the script
location, and requires explicit manifest/overlay/checkpoint paths. Training scripts
delegate to `tools/train.py`. `evaluate_resilient_v2x.sh` is deliberately a
val-only, non-resumable local diagnostic: it requires the literal
`--diagnostic --split val`, writes outside the controlled evidence root, labels
its output `controlled=false/paper_eligible=false`, and only then may delegate to
`tools/test.py`. It rejects `split=test`, controlled/test-claim arguments and any
attempt to write into an evidence store. All controlled val/test execution is
owned later by the evaluation plan's `run_matrix.py`/`predict_attempt.py`. No
hard-coded workstation paths or checkpoint defaults are permitted.

- [ ] **Step 4: Run complete phase regression**

Run:

```bash
python -m pytest tests/resilient_v2x/test_manifest.py tests/resilient_v2x/test_prepare_data.py tests/resilient_v2x/test_schedule.py tests/resilient_v2x/test_dataset_pipeline.py tests/resilient_v2x/test_encoders.py tests/resilient_v2x/test_detector.py tests/resilient_v2x/test_teacher_student.py tests/resilient_v2x/test_training_protocol.py tests/resilient_v2x/test_configs.py tests/resilient_v2x/test_capacity_match.py tests/resilient_v2x/test_concat_detector.py tests/resilient_v2x/test_integration.py -q
bash -n scripts/train_resilient_v2x_teacher.sh scripts/train_resilient_v2x_student.sh scripts/evaluate_resilient_v2x.sh
python -m compileall -q transvision/dataset transvision/models tools/resilient_v2x
git diff --check
```

Expected: all runnable CPU/synthetic checks pass; shell, compileall, and diff checks are silent.

- [ ] **Step 5: Verify tracked-file and placeholder gates**

Run:

```bash
git ls-files configs/resilient_v2x transvision/models/resilient_v2x transvision/models/detectors/resilient_v2x.py transvision/models/detectors/resilient_v2x_distiller.py transvision/models/detectors/resilient_v2x_concat.py transvision/dataset/resilient_v2x_dataset.py transvision/dataset/resilient_v2x_manifest.py transvision/dataset/resilient_v2x_schedule.py tools/resilient_v2x scripts/train_resilient_v2x_teacher.sh scripts/train_resilient_v2x_student.sh scripts/evaluate_resilient_v2x.sh tests/resilient_v2x
rg -n 'TODO|TBD|NotImplemented|pass$' configs/resilient_v2x transvision/models/resilient_v2x transvision/models/detectors/resilient_v2x.py transvision/models/detectors/resilient_v2x_distiller.py transvision/models/detectors/resilient_v2x_concat.py transvision/dataset/resilient_v2x_*.py tools/resilient_v2x tests/resilient_v2x
```

Expected: every expected file is listed by Git; placeholder scan returns no matches.

- [ ] **Step 6: Commit**

```bash
git add configs/resilient_v2x/_base_/synthetic.py tests/resilient_v2x/fixtures/synthetic_manifest.json tests/resilient_v2x/test_integration.py scripts/train_resilient_v2x_teacher.sh scripts/train_resilient_v2x_student.sh scripts/evaluate_resilient_v2x.sh docs/resilient_v2x/training.md
git commit -m "test: add resilient v2x synthetic integration"
```

## Completion Gate

This plan is complete only when:

- temporal manifest and both overlays are canonical, immutable, hashed, and tested;
- every manifest sample carries annotation-hash-bound, evaluator-ready `Car` ground
  truth normalized once into the current ego LiDAR bottom-center convention;
- spies prove unarrived/faulted/invalid raw slices are never loaded or encoded;
- four branches share encoder weights and produce the fixed BEV shape;
- the old detector prototype is replaced by a thin orchestrator;
- teacher is frozen/hash-verified, student is strict-initialized, and deployment state has no teacher weights;
- controlled invalid batches fail on every rank, generic zero ranks preserve the fixed loss
  schema, the custom outer DDP train step keeps reducer hooks active, and
  manifest-derived update counts/LR resume tests pass；
- checkpoint selection uses only globally aggregated Full@0 ms val AP and detection loss；
- teacher, full student, seven non-full variants, and the independent concat detector all
  resolve and build; concat contains no PTF, repair, modality aggregator, router, or teacher;
- synthetic one-batch loss and predict pass;
- CUDA/data-only acceptance remains explicitly unexecuted until the target environment and dataset exist.
