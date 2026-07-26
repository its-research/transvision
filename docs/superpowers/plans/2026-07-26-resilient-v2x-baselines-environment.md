# Resilient V2X 基线适配与可复现环境 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 锁定 Linux/CUDA 主环境，修复可重复安装边界，为 FFNet、CoFormerNet、BEVFusion、V2X-ViT、CoBEVT 和 LRCP 提供统一、可追溯、fail-closed 的 baseline adapter，并完成全仓回归。

**Architecture:** 环境文件和 capture/build checks 定义主实验 runtime；adapter 只负责 capabilities、资产校验、确定性 argv、prediction parser 和 provenance，不改变统一 evaluator、split 或 evidence contract。内置方法直接走 MMEngine Runner，外部方法通过 declarative manifest 接入。

**Tech Stack:** Python 3.10、CUDA 11.8、PyTorch 2.0.1、torchvision 0.15.2、MMEngine 0.10.7、MMCV 2.1.0、MMDetection 3.2.0、MMDetection3D 1.3.0、fvcore 0.1.5.post20221221、zstandard 0.22.0、Docker、conda-lock。

## Global Constraints

- **跨计划执行顺序：** 先执行本计划 Task 1–2，得到可安装、可捕获的测试环境；再执行核心、数据训练、评估证据三份计划；最后返回执行本计划 Task 3–9。
- 主证据环境仅支持 Linux x86_64；macOS 环境 ID 必须不同，且不能生成 GPU/论文数值通过声明。
- baseline 不得使用旧 detection-model wrappers：FFNet wrapper 会删除输出目录，FFNet/CoFormer wrapper 会把空预测伪造成零框。
- capability 由 adapter 明确声明，不能从旧 config 中的 `use_camera=True` 推断。
- 外部代码、checkpoint、license、commit、命令或 parser 任一缺失时生成 blocked plan，不生成 metrics。
- public/upstream checkpoint 默认是 cross-protocol；只有 shared-protocol 本地重训证据可以标 controlled。
- 不下载或提交 DAIR-V2X-C、外部源码或 checkpoints。
- 本计划 Task 1 和 Task 2 的 bootstrap/static 命令必须先执行
  `source /Users/libin/transvision/.venv/resilient-v2x-dev/bin/activate`；随后用
  `python -c 'import sys; assert sys.executable == "/Users/libin/transvision/.venv/resilient-v2x-dev/bin/python"; assert sys.version_info[:2] == (3, 10)'`
  锁定解释器。核心、数据训练、评估以及本计划 Task 3–9 必须在 Task 2
  生成的 Linux lock runtime 或该 Docker image 内执行；禁止回退到当前机器的
  Python 3.13/3.9。
- 所有命令从 `/Users/libin/transvision` 执行。

---

### Task 1: 修复 package metadata/import 边界并建立精确 constraints

**Files:**

- Modify: `setup.py`
- Modify: `transvision/__init__.py`
- Modify: `transvision/models/__init__.py`
- Create: `transvision/register.py`
- Modify: `tools/train.py`
- Modify: `tools/test.py`
- Modify: `tools/browse_dataset.py`
- Create: `environments/resilient_v2x/constraints.txt`
- Create: `environments/resilient_v2x/environment.yml`
- Create: `environments/resilient_v2x/dev-requirements.txt`
- Create: `tests/resilient_v2x/test_packaging.py`

**Interfaces:**

```python
def register_resilient_v2x_modules() -> None
```

- [ ] **Step 0: Bootstrap the only permitted development interpreter**

Run before writing or executing the first test. This checkout already has a Miniconda
entrypoint, so bootstrap Python 3.10 from it instead of assuming `python3.10` exists:

```bash
/Users/libin/miniconda3/bin/conda create --yes --prefix /Users/libin/transvision/.venv/resilient-v2x-dev --override-channels --channel conda-forge python=3.10.14 pip=23.3.2
/Users/libin/transvision/.venv/resilient-v2x-dev/bin/python -m pip install --upgrade pip==23.3.2
/Users/libin/transvision/.venv/resilient-v2x-dev/bin/python -m pip install setuptools==68.2.2 wheel==0.41.3 packaging==23.2 numpy==1.24.4 pytest==7.4.4 jsonschema==4.23.0 PyYAML==6.0.2 zstandard==0.22.0 pypcd4==1.4.3 conda-lock==2.5.7
source /Users/libin/transvision/.venv/resilient-v2x-dev/bin/activate
python -c 'import sys; assert sys.executable == "/Users/libin/transvision/.venv/resilient-v2x-dev/bin/python"; assert sys.version_info[:2] == (3, 10)'
```

Expected: Python reports `3.10.x`, the final assertion is silent, and `.venv/` remains
ignored by Git. If environment creation or dependency installation is not approved, stop
this task as `blocked`; do not run the tests with another interpreter. This development
environment may run package and pure manifest/protocol/statistics tests, but it has a
distinct environment ID and cannot pass Torch/MMEngine/CUDA tests or produce paper
evidence.

- [ ] **Step 1: Write metadata and version-file immutability tests**

```python
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_setup_metadata_does_not_require_torch_import(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=ROOT,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "transvision"


def test_setup_metadata_does_not_rewrite_tracked_version_file() -> None:
    version_file = ROOT / "transvision/version.py"
    before = hashlib.sha256(version_file.read_bytes()).hexdigest()
    subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = hashlib.sha256(version_file.read_bytes()).hexdigest()
    assert after == before


def test_core_import_does_not_import_custom_ops() -> None:
    code = (
        "import sys; "
        "import transvision; "
        "assert 'transvision.models.bev_pool' not in sys.modules; "
        "assert 'transvision.models.voxel.voxel_layer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
```

Add tests that parse constraints and assert exact versions for every approved runtime package plus `pypcd4==1.4.3`, `pytest==7.4.4`, `jsonschema==4.23.0`, `PyYAML==6.0.2`, and `conda-lock==2.5.7`.

- [ ] **Step 2: Run tests and observe current setup failure**

Run: `python -m pytest tests/resilient_v2x/test_packaging.py -q`

Expected before fix: FAIL because `setup.py` imports torch at module import time and metadata execution rewrites `transvision/version.py`.

- [ ] **Step 3: Refactor setup to lazy-build extensions**

`setup.py` must:

- avoid `import torch` and `torch.utils.cpp_extension` until the extension build command runs;
- derive package version from the existing `transvision/version.py` without writing it;
- keep CUDA extension definitions functionally equivalent;
- use `FORCE_CUDA` only during build, not metadata;
- include `python_requires=">=3.10,<3.11"`;
- read runtime dependencies from constraints-compatible metadata without stripping exact versions.

The top level may import only stdlib and setuptools. A custom `LazyBuildExtension` can import PyTorch inside `run()`.

- [ ] **Step 4: Make registry loading explicit and keep core imports extension-free**

`transvision/__init__.py` exports only version metadata and `register_all_modules`. `transvision/models/__init__.py` must not import detection models or CUDA-backed modules at import time. Preserve legacy callers through an explicit lazy function:

```python
def get_supported_models() -> dict[str, type]:
    from transvision.models.detection_models import (
        CoFormer,
        EarlyFusion,
        FeatureFlow,
        FeatureFusion,
        InfOnly,
        LateFusion,
        SingleSide,
        VehOnly,
    )
    return {
        "single_side": SingleSide,
        "late_fusion": LateFusion,
        "early_fusion": EarlyFusion,
        "veh_only": VehOnly,
        "inf_only": InfOnly,
        "feature_fusion": FeatureFusion,
        "feature_flow": FeatureFlow,
        "coformer": CoFormer,
    }


def __getattr__(name: str) -> object:
    if name == "SUPPROTED_MODELS":
        return get_supported_models()
    raise AttributeError(name)
```

`register_all_modules()` imports the existing package-level dataset transforms,
preprocessors, detectors, heads, necks, hooks, and metrics exactly once. It must not name
future Resilient files or adapters that have not been created yet; later files register
through those package `__init__.py` exports. `tools/train.py`, `tools/test.py`, and
`tools/browse_dataset.py` call it before registry build. This keeps a plain `import
transvision` extension-free while preserving Runner registration after the package
registries are explicitly requested.

- [ ] **Step 5: Add exact environment inputs**

`constraints.txt` must include:

```text
torch==2.0.1
torchvision==0.15.2
numpy==1.24.4
mmengine==0.10.7
mmcv==2.1.0
mmdet==3.2.0
mmdet3d==1.3.0
fvcore==0.1.5.post20221221
zstandard==0.22.0
pypcd4==1.4.3
pytest==7.4.4
jsonschema==4.23.0
PyYAML==6.0.2
conda-lock==2.5.7
```

`pypcd4` is the pinned Python 3 PCD reader used only by deterministic DAIR preparation; its wheel/source hash must be present in the generated lock. `environment.yml` pins Python 3.10 and CUDA 11.8 channels/packages. It must not contain floating `latest`, `>=`, wildcard versions, or remote pretrained URLs.

`dev-requirements.txt` also pins the bootstrap-only packages exactly:
`pip==23.3.2`, `setuptools==68.2.2`, `wheel==0.41.3`,
`packaging==23.2`, `numpy==1.24.4`, `pytest==7.4.4`,
`jsonschema==4.23.0`, `PyYAML==6.0.2`, `zstandard==0.22.0`,
`pypcd4==1.4.3`, and `conda-lock==2.5.7`. After the
files exist, reinstall them with
`python -m pip install -r environments/resilient_v2x/dev-requirements.txt`
and re-run the interpreter assertion before continuing.

- [ ] **Step 6: Run packaging tests**

Run:

```bash
python -m pytest tests/resilient_v2x/test_packaging.py -q
python setup.py --name
python setup.py --version
git diff --check
```

Expected: tests pass; metadata prints `transvision` and `0.1.0`; version file remains unchanged; a plain package import does not load custom ops. The core plan separately tests the later-created core contracts.

- [ ] **Step 7: Commit**

```bash
git add setup.py transvision/__init__.py transvision/models/__init__.py transvision/register.py tools/train.py tools/test.py tools/browse_dataset.py environments/resilient_v2x/constraints.txt environments/resilient_v2x/environment.yml environments/resilient_v2x/dev-requirements.txt tests/resilient_v2x/test_packaging.py
git commit -m "build: define resilient v2x package constraints"
```

---

### Task 2: 生成 Linux lock、Docker、environment capture 和 custom-op check

**Files:**

- Create: `environments/resilient_v2x/environment-linux-64.lock.yml`
- Create: `environments/resilient_v2x/bootstrap-environment.yml`
- Create: `environments/resilient_v2x/bootstrap-linux-64.explicit.txt`
- Create: `environments/resilient_v2x/Dockerfile`
- Create: `environments/resilient_v2x/environment-manifest.schema.json`
- Create: `tools/resilient_v2x/capture_environment.py`
- Create: `tools/resilient_v2x/check_environment.py`
- Create: `scripts/build_resilient_v2x_ops.sh`
- Create: `tests/resilient_v2x/test_environment.py`

**Interfaces:**

```python
class EnvironmentMismatch(RuntimeError):
    """Captured runtime does not satisfy the requested environment contract."""

@dataclass(frozen=True)
class HardwareFingerprint:
    platform_system: str
    platform_machine: str
    cpu_model: str
    logical_cpu_count: int
    total_ram_bytes: int
    gpu_devices: tuple[Mapping[str, object], ...]
    cuda_driver: str
    cuda_runtime: str
    cudnn_version: str
    world_size: Literal[1, 2, 4]
    fingerprint_sha256: str

@dataclass(frozen=True)
class EnvironmentManifest:
    schema_version: Literal[1]
    classification: Literal["development", "controlled"]
    platform: Mapping[str, object]
    python: Mapping[str, object]
    packages: Mapping[str, str]
    cuda: Mapping[str, object]
    custom_ops: Mapping[str, str]
    determinism: Mapping[str, object]
    runtime_lock_sha256: str
    bootstrap_lock_sha256: str
    container_image_digest: str | None
    hardware: HardwareFingerprint
    actual_hardware_fingerprint_sha256: str
    manifest_sha256: str

@dataclass(frozen=True)
class EnvironmentValidation:
    accepted: bool
    classification: Literal["development", "controlled"]
    environment_manifest_sha256: str
    actual_hardware_fingerprint_sha256: str
    verified_fields: tuple[str, ...]
    blocking_mismatches: tuple[str, ...]
    not_executed_checks: tuple[str, ...]

def capture_environment(
    world_size: Literal[1, 2, 4],
) -> dict[str, object]
def canonical_hardware_fingerprint(
    actual: Mapping[str, object],
    world_size: Literal[1, 2, 4],
) -> HardwareFingerprint
def hardware_fingerprint_sha256(
    fingerprint_without_hash: Mapping[str, object],
) -> str
def build_environment_manifest(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    world_size: Literal[1, 2, 4],
    classification: Literal["development", "controlled"],
) -> EnvironmentManifest
def validate_environment(
    actual: EnvironmentManifest,
    expected: Mapping[str, object],
    require_cuda: bool,
) -> EnvironmentValidation
def write_environment_manifest(
    manifest: EnvironmentManifest,
    output: Path,
) -> EnvironmentManifest
def main(argv: Sequence[str] | None = None) -> int
```

- [ ] **Step 1: Write capture and mismatch tests**

```python
def test_capture_has_required_hardware_and_determinism_fields(monkeypatch) -> None:
    stub_runtime(monkeypatch)
    captured = capture_environment(world_size=1)
    assert captured["schema_version"] == 1
    assert captured["platform"]["system"] == "Linux"
    assert captured["python"]["version"] == "3.10.14"
    assert captured["cuda"]["runtime"] == "11.8"
    assert captured["gpu"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert captured["determinism"]["cudnn_benchmark"] is False
    assert captured["determinism"]["tf32"] is False


def test_wrong_mmdet3d_version_fails() -> None:
    actual = expected_environment_capture()
    actual["packages"]["mmdet3d"] = "1.2.0"
    manifest = build_environment_manifest(
        actual,
        expected_environment_fixture(),
        world_size=1,
        classification="controlled",
    )
    with pytest.raises(EnvironmentMismatch, match="mmdet3d"):
        validate_environment(manifest, expected_environment_fixture(), require_cuda=True)
```

Add tests for macOS development classification, missing CUDA, wrong world-size hardware fingerprint, dirty deterministic flags, lock entries missing hashes, exact constraints/lock agreement, both pinned Docker `FROM` digests, bootstrap-lock hashes, final non-root user, and absence of curl/wget/unlocked solves in the Dockerfile.
Add one fixed canonical-hardware-hash vector and mutation tests for CPU model/count,
RAM bytes, GPU order/name/UUID/compute capability/memory, CUDA driver/runtime,
cuDNN version, platform machine,
and world size. `actual_hardware_fingerprint_sha256` must equal the nested
`HardwareFingerprint.fingerprint_sha256`; the environment manifest hash covers that
value, all package/lock/custom-op/determinism fields, and excludes only
`manifest_sha256` itself.
Add a CLI test that `--output <absolute-path>` writes canonical JSON plus a detached
SHA-256 and refuses an output outside the explicitly supplied evidence root.
`--world-size {1,2,4}` is mandatory, enters the hardware fingerprint, and must
equal the later runner launcher world size; omission, inference from visible GPU
count, or a `4 -> 1` mismatch fails before attempt creation.
Inject a crash after the JSON and after its detached hash. Retry recomputes the
canonical capture: an existing byte-identical JSON may receive its missing
sidecar, and an exact JSON+sidecar pair is idempotently reused; any byte/hash
conflict or orphan sidecar fails without overwrite.

- [ ] **Step 2: Run tests and confirm missing tools**

Run: `python -m pytest tests/resilient_v2x/test_environment.py -q`

Expected: FAIL because environment tools are absent.

- [ ] **Step 3: Generate and audit the lock**

Run in a network-enabled, approved environment:

```bash
conda-lock lock --file environments/resilient_v2x/environment.yml --platform linux-64 --lockfile environments/resilient_v2x/environment-linux-64.lock.yml
```

Generate a separate, minimal bootstrap lock containing only exact
`python=3.10.14`, `pip=23.3.2`, and `conda-lock=2.5.7` plus their dependencies:

```bash
conda-lock lock --file environments/resilient_v2x/bootstrap-environment.yml --platform linux-64 --kind explicit --filename-template environments/resilient_v2x/bootstrap-{platform}.explicit.txt
```

Then verify every fetched conda/pip artifact in the unified runtime lock and every
artifact in the explicit bootstrap lock has a cryptographic hash and all approved
versions are exact. The bootstrap spec may contain no project/runtime dependency.
Do not hand-edit resolved hashes. Do not render the runtime lock as an explicit
lock, because its pip dependencies must be installed by `conda-lock`, not silently
dropped by micromamba.

On the Linux implementation host, create the command runtime only from that lock:

```bash
conda-lock install --prefix /Users/libin/transvision/.venv/resilient-v2x-runtime environments/resilient_v2x/environment-linux-64.lock.yml
source /Users/libin/transvision/.venv/resilient-v2x-runtime/bin/activate
python -c 'import sys, torch, mmengine, mmcv, mmdet, mmdet3d; assert sys.version_info[:2] == (3, 10); assert torch.__version__.split("+")[0] == "2.0.1"; assert mmengine.__version__ == "0.10.7"; assert mmcv.__version__ == "2.1.0"; assert mmdet.__version__ == "3.2.0"; assert mmdet3d.__version__ == "1.3.0"'
```

Every later plan's `python` command means this activated interpreter (or the identical
interpreter inside `resilient-v2x:test`). A new shell must reactivate and rerun the
assertion. The Mac bootstrap environment is not an acceptable substitute for runtime
tests; unavailable runtime tests are recorded `not_executed`.

- [ ] **Step 4: Implement Docker and checks**

Dockerfile requirements:

- base exactly `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:bd746eb3b9953805ebe644847a227e218b5da775f47007c69930569a75c9ad7d` for `linux/amd64`;
- obtain `/bin/micromamba` only from a pinned first stage
  `mambaorg/micromamba:2.8.1@sha256:b04d00cd5e7ab14f97217c24bc89f035db33a8d339bfb9857698d9390bc66cf8`,
  copy that binary into the CUDA stage, create `/opt/bootstrap` only from
  `bootstrap-linux-64.explicit.txt`, then run the locked
  `/opt/bootstrap/bin/conda-lock install --prefix /opt/resilient-v2x
  environment-linux-64.lock.yml`; do not curl Miniconda, run an unpinned
  package-manager bootstrap, drop pip-locked packages, or solve project
  dependencies during `docker build`;
- non-root runtime user;
- install from the lock/constraints;
- copy source after dependency layer;
- build both custom extensions;
- run import/version/custom-op smoke;
- no DAIR data, checkpoint, credential, SSH key, or external baseline source in image.

`build_resilient_v2x_ops.sh` uses `set -euo pipefail`, `FORCE_CUDA=1`, builds in an isolated temporary directory, imports `voxel_layer` and `bev_pool_ext`, and exits non-zero on any mismatch.

The capture CLI builds one `EnvironmentManifest`, writes it plus its detached hash
with recoverable create-if-absent semantics, reloads it through strict schema
parsing, and returns the
verified object. Runner materialization never accepts a caller-supplied hardware
fingerprint string: it calls `validate_environment()`, then copies only
`EnvironmentValidation.environment_manifest_sha256` and
`actual_hardware_fingerprint_sha256` into training/evaluation realizations,
claims, aggregation records, and profiling protocols. Every consumer recomputes
both hashes from the accepted manifest before use.

- [ ] **Step 5: Run environment unit/static checks**

Run:

```bash
python -m pytest tests/resilient_v2x/test_environment.py -q
python tools/resilient_v2x/check_environment.py --mode development
bash -n scripts/build_resilient_v2x_ops.sh
docker build --check -f environments/resilient_v2x/Dockerfile .
git diff --check
```

Expected locally: unit/static checks pass; development mode reports missing CUDA/data as `not_executed`, not `passed`. If Docker is unavailable, record the Docker command as not executed.

- [ ] **Step 6: Build and execute the real target image**

Run on an approved Linux host with Docker, NVIDIA Container Toolkit, and a CUDA GPU:

```bash
docker build --platform linux/amd64 -f environments/resilient_v2x/Dockerfile -t resilient-v2x:test .
docker run --rm --platform linux/amd64 --gpus all resilient-v2x:test python tools/resilient_v2x/check_environment.py --mode controlled --require-cuda --require-custom-ops
```

Expected: the image build compiles both custom extensions; the container validates the
exact dependency versions, CUDA runtime, deterministic flags, and imports both custom ops.
This step may be recorded `not_executed` on the development Mac, but the overall
reproduction Completion Gate cannot report the target environment as accepted until both
commands succeed.

- [ ] **Step 7: Commit**

```bash
git add environments/resilient_v2x/environment-linux-64.lock.yml environments/resilient_v2x/bootstrap-environment.yml environments/resilient_v2x/bootstrap-linux-64.explicit.txt environments/resilient_v2x/Dockerfile environments/resilient_v2x/environment-manifest.schema.json tools/resilient_v2x/capture_environment.py tools/resilient_v2x/check_environment.py scripts/build_resilient_v2x_ops.sh tests/resilient_v2x/test_environment.py
git commit -m "build: lock resilient v2x experiment environment"
```

---

### Task 3: 定义统一 baseline adapter、capabilities 和 prediction schema

**Files:**

- Create: `transvision/experiments/resilient_v2x/adapters/__init__.py`
- Create: `transvision/experiments/resilient_v2x/adapters/base.py`
- Create: `transvision/experiments/resilient_v2x/schemas/baseline_template.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/baseline_manifest.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/prediction.schema.json`
- Create: `tests/resilient_v2x/test_adapter_contract.py`

**Interfaces:**

```python
from transvision.experiments.resilient_v2x.protocol import (
    CapabilityDeclaration,
    CompatibilityReport,
    TrainingProtocolDeclaration,
    TrainingProtocolVerification,
)

AdapterCapabilities = CapabilityDeclaration

@dataclass(frozen=True)
class AdapterContext:
    method: str
    mode: Literal["local_training", "upstream"]
    split: Literal["train", "val", "test"]
    seed: Literal[0, 1, 2]
    repo_root: Path
    source_snapshot: Path
    data_root: Path
    temporal_manifest: Path
    dataset_release_sha256: str
    split_manifest_sha256: str
    temporal_manifest_sha256: str
    transport_overlay: Path
    transport_overlay_sha256: str
    fault_overlay: Path
    fault_overlay_sha256: str
    evidence_root: Path
    shared_evaluator_environment_manifest: Path
    shared_evaluator_environment_manifest_sha256: str
    shared_evaluator_hardware_fingerprint_sha256: str
    adapter_execution_environment_manifest: Path
    adapter_execution_environment_manifest_sha256: str
    adapter_execution_hardware_fingerprint_sha256: str
    output_dir: Path
    resolved_manifest_path: Path | None
    resolved_manifest_sha256: str | None
    protocol_sha256: str
    orchestrator_code_commit: str
    source_commit: str
    source_tree: str
    source_snapshot_sha256: str
    checkpoint_path: Path | None
    checkpoint_sha256: str | None

@dataclass(frozen=True)
class AssetValidation:
    ready: bool
    missing_fields: tuple[str, ...]
    missing_assets: tuple[str, ...]
    invalid_assets: tuple[str, ...]
    verified_artifacts: tuple[tuple[str, str], ...]
    capability_report: CompatibilityReport
    validation_sha256: str

class BaselineAdapter(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities

    @property
    @abstractmethod
    def training_protocol(self) -> TrainingProtocolDeclaration

    def validate_assets(self, context: AdapterContext) -> AssetValidation

    def prepare_command(self, context: AdapterContext) -> tuple[str, ...]

    def train_command(self, context: AdapterContext) -> tuple[str, ...]

    def predict_command(self, context: AdapterContext) -> tuple[str, ...]

    @abstractmethod
    def build_config_overrides(self, context: AdapterContext) -> dict[str, object]

    @abstractmethod
    def parse_predictions(self, output_path: Path) -> tuple[SamplePrediction, ...]

@dataclass(frozen=True)
class ExternalPredictionSubset:
    attempt_id: str
    split: Literal["val", "test"]
    expected_sample_ids: tuple[str, ...]
    sample_manifest_path: Path
    per_sample_output_dir: Path

@dataclass(frozen=True)
class BaselineSampleRecord:
    prediction: SamplePrediction
    diagnostics: SampleInferenceDiagnostics

def baseline_inference_diagnostics(
    method: str,
    sample_id: str,
    selections: tuple[
        BranchSelection,
        BranchSelection,
        BranchSelection,
        BranchSelection,
    ],
    supports_der: bool,
) -> SampleInferenceDiagnostics

class ControlledExternalPredictionAdapter(Protocol):
    def predict_subset_command(
        self,
        context: AdapterContext,
        subset: ExternalPredictionSubset,
    ) -> tuple[str, ...]
    def parse_prediction_file(
        self,
        output_path: Path,
        expected_sample_id: str,
        selections: tuple[
            BranchSelection,
            BranchSelection,
            BranchSelection,
            BranchSelection,
        ],
    ) -> BaselineSampleRecord

def verify_training_protocol_evidence(
    declaration: TrainingProtocolDeclaration,
    accepted_training_manifest: Path,
) -> TrainingProtocolVerification

def validate_prediction_contract(
    predictions: Sequence[SamplePrediction],
    expected_sample_ids: Sequence[str],
) -> tuple[SamplePrediction, ...]
```

`AssetValidation.validation_sha256` is exactly
`sha256(b"asset-validation-v1\0" +
canonical_json(payload_without_validation_sha256))`. Verified artifacts are
sorted by canonical artifact key and duplicates are rejected. Add a checked-in
fixed vector plus missing/invalid artifact and capability-report mutation tests;
the digest field is excluded only from its own calculation.

The base class supplies concrete shared implementations of asset validation and
prepare/train/predict command construction from `AdapterContext`,
`build_config_overrides()`, and the immutable protocol command templates.
Only capabilities, training protocol, config overrides, and prediction parsing
remain abstract. Built-in adapters are therefore instantiable without duplicating
the command/evidence path; the manifest-backed external adapter overrides only
subset command/file parsing where its contract differs. A test instantiates every
class in `ADAPTERS` and asserts `__abstractmethods__ == frozenset()`.
Context construction verifies and propagates every dataset/split/temporal/
transport/fault/environment/hardware/code hash into prepare/train/predict argv,
command manifests and model provenance. Adapter overrides cannot replace these
shared values, and mutation tests prove each changed hash changes the plan/command
identity or blocks before launch.
Built-ins set adapter-execution and shared-evaluator environments to the same
accepted manifest. An external adapter obtains its independent execution
environment only from the strict resolved manifest, while the runner supplies the
shared evaluator environment. Both typed path/hash/hardware triples enter command
and provenance records; training realization uses the former and evaluation
realization uses the latter. Missing identity or falsely flattening two different
environments blocks controlled classification.

- [ ] **Step 1: Write capability and output-contract tests**

```python
def test_lidar_only_adapter_is_incompatible_with_camera_fault_table() -> None:
    report = determine_compatibility(
        candidate=lidar_only_capabilities(),
        required=multimodal_c_fail_requirement(),
        training=shared_lidar_training_declaration(),
        training_evidence=verified_shared_lidar_training(),
    )
    assert report.compatible is False
    assert "camera" in report.reasons[0]


def test_empty_prediction_stays_empty() -> None:
    predictions = validate_prediction_contract(
        [SamplePrediction(sample_id="a", objects=())],
        expected_sample_ids=("a",),
    )
    assert predictions[0].objects == ()


def test_training_protocol_deviation_is_not_controlled() -> None:
    declaration = shared_protocol_declaration(
        epochs=50,
        global_batch_size=8,
        seeds=(0, 1, 2),
    )
    report = determine_compatibility(
        candidate=multimodal_capabilities(),
        required=shared_multimodal_requirement(),
        training=declaration,
        training_evidence=verified_shared_multimodal_training(),
    )
    assert report.compatible is False
    assert report.classification == "cross_protocol"
    assert "global_batch_size" in report.blocking_deviations
```

Add tests for duplicate/missing sample IDs, NaN scores, wrong 7D box order, non-bottom z convention, unknown class, score/box length mismatch, destructive paths, shell metacharacters, and asset hash/commit/license validation.
For a controlled external backend, also require the explicit subset/per-sample
protocol above; a whole-split-only command can be cross-protocol but is never
controlled or reserved-test eligible.
`baseline_inference_diagnostics()` preserves the real immutable source
tick/time/horizon/support/reason/observed/propagated selection fields. Methods
without a modality emit `METHOD_NOT_APPLICABLE`; methods without PTF/DER use null
D/Q/gamma/descriptor/support/weights plus explicit non-applicable reasons.
Supported branch reliability is 1 and unsupported reliability is 0; no learned
quantity is fabricated.
Also prove `baseline_template.schema.json` accepts only the checked-in unresolved
template shape, while `baseline_manifest.schema.json` accepts only a fully
materialized executable manifest and references the canonical
`training_protocol.schema.json`; placeholder strings, nulls or missing training
fields never validate as a resolved manifest.
Also assert that a controlled declaration has exactly `epochs=50`,
`global_batch_size=4`, seeds `(0,1,2)`, delays `(0,100,200,300)`,
`p_L=p_C=0.3`, fault scope
`per-agent-modality-global-target-tick-v1`, and checkpoint selector
`full-0ms-bev-ap07-loss-epoch-v1`. Every method must publish a non-empty,
versioned optimizer recipe ID plus SHA-256; capacity-matched concat must use the exact
Resilient AdamW recipe, while other baselines may retain their public recipe only when it
is hashed and any deviation is reported. For Resilient and capacity concat, also require
the exact augmentation recipe, global shuffle/drop/shard sampler,
`(world_size, accumulation)` mapping `((1,4),(2,2),(4,1))`,
optimizer-update-indexed warmup/cosine scheduler, AMP, clip 35 and fail-on-nonfinite
behavior. Other baselines lock those fields to their own public versioned recipe and
prove realized global batch 4, but need not copy Resilient optimizer internals. Every
method requires a resolved config digest and protocol digest.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_adapter_contract.py -q`

Expected: FAIL because adapter APIs are absent.

- [ ] **Step 3: Implement fail-closed base behavior**

Commands are immutable argv tuples, never shell strings. Adapter context must provide explicit repository root, evidence attempt directory, manifest, overlays, checkpoint, environment, and seed. Path validation rejects repository root, evidence root parent, globs, and any output path outside the current attempt.

The training declaration and verification types come only from
`transvision.experiments.resilient_v2x.protocol`, implemented earlier by the
evaluation/evidence plan; adapters must not fork or shadow that schema.

Prediction parser output uses ego LiDAR `[x,y,z_bottom,length,width,height,yaw]`, radians, `Car`, and original prediction index.
`determine_compatibility()` evaluates input capabilities and
`TrainingProtocolDeclaration` together. Missing training fields or any mismatch in the
shared split, 50 epochs, global batch 4, seeds, delay/fault sampling, checkpoint selector,
evaluator, or evidence contract is `cross_protocol`, never controlled. The compatibility
report preserves the declared optimizer/augmentation/sampler/scheduler recipe and
separates `reported_deviations` from `blocking_deviations`. For a non-Resilient
baseline, a verified difference from the Resilient optimizer recipe is reported but is
not blocking; a difference from that method's declared public recipe is blocking.
`deviations` is not a user-controlled waiver: the validator recomputes deviations from
the resolved config/commands, requires the declaration to match that computed tuple, and
fails closed on an omitted or falsely empty deviation list.
For a controlled label, `verify_training_protocol_evidence()` must read an accepted local
training manifest and compare its resolved config, realized epochs/global batches,
sampler order/dropped-sample records, optimizer updates, augmentation/overlay digests,
checkpoint-selection inputs, precision/nonfinite policy, and artifact hashes. Arbitrary
external argv or a self-declared manifest is never sufficient evidence. Upstream
checkpoints remain cross-protocol.

- [ ] **Step 4: Run adapter contract tests**

Run: `python -m pytest tests/resilient_v2x/test_adapter_contract.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/adapters transvision/experiments/resilient_v2x/schemas/baseline_template.schema.json transvision/experiments/resilient_v2x/schemas/baseline_manifest.schema.json transvision/experiments/resilient_v2x/schemas/prediction.schema.json tests/resilient_v2x/test_adapter_contract.py
git commit -m "feat: define resilient v2x baseline adapters"
```

---

### Task 4: 适配 FFNet 为 LiDAR-only shared-protocol reference

**Files:**

- Create: `transvision/experiments/resilient_v2x/adapters/ffnet.py`
- Create: `transvision/models/detectors/resilient_v2x_baselines.py`
- Modify: `transvision/models/detectors/__init__.py`
- Create: `configs/resilient_v2x/baselines/ffnet_lidar.py`
- Create: `tests/resilient_v2x/test_ffnet_adapter.py`

**Interfaces:**

```python
class FFNetAdapter(BaselineAdapter):
    method = "ffnet"

    @property
    def capabilities(self) -> AdapterCapabilities
    @property
    def training_protocol(self) -> TrainingProtocolDeclaration
    def build_config_overrides(self, context: AdapterContext) -> dict[str, object]
    def parse_predictions(self, output_path: Path) -> tuple[SamplePrediction, ...]

@dataclass(frozen=True)
class BaselineLidarHistory:
    sample_ids: tuple[str, ...]
    ego: tuple[tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None], ...]
    rsu: tuple[tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None], ...]
    availability: Tensor
    source_to_target: Tensor
    selected_flat_index: Tensor

def unpack_lidar_history(inputs: ResilientModelInputs) -> BaselineLidarHistory

@MODELS.register_module()
class ResilientFeatureFlowNet(FeatureFlowNet):
    def loss_with_validity(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample],
    ) -> tuple[ValidityAwareLosses, BatchValidityDecision]
    def forward_selection_detection_loss(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample],
    ) -> tuple[Tensor, int]
```

The two validity-aware methods deliberately match the shared
`ValidityAwareLossModel`/`SelectionLossDetector` protocol signatures exactly:
`Mapping[str, object]` plus `Sequence[Det3DDataSample]`. Each implementation
validates and narrows that mapping to `ResilientModelInputs` internally before
calling `unpack_lidar_history()`. A static protocol-conformance test covers
FFNet, CoFormer, BEVFusion, Resilient and concat detectors.

- [ ] **Step 1: Write direct-Runner and empty-output tests**

```python
def test_ffnet_uses_runner_not_legacy_wrapper(adapter: FFNetAdapter) -> None:
    command = adapter.predict_command(ffnet_context())
    assert command[:2] == (
        sys.executable,
        "tools/resilient_v2x/predict_attempt.py",
    )
    assert "mmdet3d_lidar_feature_flow.py" not in " ".join(command)
    assert "tools/test.py" not in command


def test_ffnet_empty_prediction_is_not_zero_box(tmp_path: Path) -> None:
    write_ffnet_output(tmp_path, sample_id="a", boxes=[], scores=[])
    parsed = FFNetAdapter().parse_predictions(tmp_path)
    assert parsed[0].objects == ()
```

Add tests that capabilities are exactly two agents, LiDAR-only, four history positions, 0/100/200/300 ms; C-Fail is incompatible; split/range/evaluator are shared; the complete training declaration records the versioned FFNet optimizer recipe plus every shared field; and no command references or deletes evidence root.
Use the four-tick nested-input fixture to prove `unpack_lidar_history()` reconstructs
only the already-loaded `[B,ego/rsu,h=0..3]` LiDAR grid from
`valid_flat_index/payloads`; it must preserve availability, selected indices and
source-to-target transforms. Monkeypatch `open`, `numpy.fromfile`, `Path.read_bytes`
and legacy metadata-path access to raise during model loss/predict. Mutating an
unavailable or unselected payload cannot change output, while mutating a selected
payload does. Assert controlled loss returns `ValidityAwareLosses`, the custom
outer DDP wrapper is selected, and `forward_selection_detection_loss()` supplies
the exact loss/count used by the shared checkpoint selector.
Both empty and non-empty predictions must attach
`baseline_inference_diagnostics(method="ffnet",...)`; LiDAR selection fields are
real, Camera branches are `METHOD_NOT_APPLICABLE`, and PTF/DER arrays are explicit
nulls accepted by the common evidence schema.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_ffnet_adapter.py -q`

Expected: FAIL because FFNet adapter/config are absent.

- [ ] **Step 3: Implement without legacy wrapper**

Use `FeatureFlowNet` through `tools/train.py`. Every controlled prediction must
enter the evaluation plan's `tools/resilient_v2x/predict_attempt.py`, which
installs `IndependentInferenceRunner`, `ResumableTestLoop`,
`ResumableSampleSampler`, canonical per-sample spooling and the fixed evaluator.
Adapt the new temporal dataset to the LiDAR-only protocol and convert each
`Det3DDataSample` to the canonical `SamplePrediction`; never call `tools/test.py`
for a matrix node. Never instantiate
`transvision.models.detection_models.mmdet3d_lidar_feature_flow.FeatureFlow`,
whose current path deletes output and inserts zero boxes.

`ResilientFeatureFlowNet` is the registered input bridge used by the new config;
the legacy class remains available for old configs. It reconstructs the causal
LiDAR history only from `ResilientModelInputs`, applies the supplied rigid
source-to-target transforms, and calls FFNet's voxel/flow/fusion/head modules
without entering any branch that reads a path from metainfo. It supports all four
horizons, consumes the overlay-selected ego/RSU entries, and fails on a selected
index whose payload/support is missing. Its native detection losses are wrapped in
`ValidityAwareLosses` with the shared global validity decision; local-zero generic
ranks emit the same declared native loss keys as valid ranks. Its selection method
runs the identical feature/head path under `no_grad`, returns the sum of real
detection losses and valid count, and never uses a proxy or precomputed metric.
Its predict path attaches the typed baseline diagnostics before returning each
`Det3DDataSample`, so the shared prediction/diagnostics spool needs no FFNet-only
exception.

Checkpoint selection is LiDAR-Full@0 ms val BEV AP@0.7, then detection loss, then earliest epoch.

- [ ] **Step 4: Run FFNet adapter tests and config build**

Run:

```bash
python -m pytest tests/resilient_v2x/test_ffnet_adapter.py -q
python -c 'from mmengine.config import Config; Config.fromfile("configs/resilient_v2x/baselines/ffnet_lidar.py")'
```

Expected: tests pass and config resolves.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/adapters/ffnet.py transvision/models/detectors/resilient_v2x_baselines.py transvision/models/detectors/__init__.py configs/resilient_v2x/baselines/ffnet_lidar.py tests/resilient_v2x/test_ffnet_adapter.py
git commit -m "feat: adapt ffnet to resilient v2x protocol"
```

---

### Task 5: 适配 CoFormerNet 为 LiDAR-only shared-protocol reference

**Files:**

- Create: `transvision/experiments/resilient_v2x/adapters/coformer.py`
- Modify: `transvision/models/detectors/resilient_v2x_baselines.py`
- Modify: `transvision/models/detectors/__init__.py`
- Create: `configs/resilient_v2x/baselines/coformer_lidar.py`
- Create: `tests/resilient_v2x/test_coformer_adapter.py`

**Interfaces:**

```python
class CoFormerAdapter(BaselineAdapter):
    method = "coformer"

    @property
    def capabilities(self) -> AdapterCapabilities
    @property
    def training_protocol(self) -> TrainingProtocolDeclaration
    def build_config_overrides(self, context: AdapterContext) -> dict[str, object]
    def parse_predictions(self, output_path: Path) -> tuple[SamplePrediction, ...]

@MODELS.register_module()
class ResilientCoFormerNet(CoFormerNet):
    def loss_with_validity(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample],
    ) -> tuple[ValidityAwareLosses, BatchValidityDecision]
    def forward_selection_detection_loss(
        self,
        inputs: Mapping[str, object],
        data_samples: Sequence[Det3DDataSample],
    ) -> tuple[Tensor, int]
```

- [ ] **Step 1: Write capability, input, and empty-output tests**

Tests must prove:

- adapter declares LiDAR only even though an old config contains `use_camera=True`;
- exactly ego/RSU LiDAR selected sources enter the Runner batch;
- Camera payload mutation cannot change CoFormer adapter output;
- empty predictions remain empty;
- latency and global-target LiDAR-Fail follow the shared horizon table;
- old `CoFormer` detection-model wrapper is not imported.
- the same nested-input/raw-read spies used for FFNet prove the registered bridge
  consumes only overlay-selected ego/RSU LiDAR payloads and transforms from all
  four possible horizons, with no metainfo path or internal file access;
- loss returns method-native `ValidityAwareLosses`, two-rank training selects the
  custom DDP outer wrapper, and the real detection loss/count reaches the shared
  checkpoint selector;
- predictions attach schema-valid `coformer` baseline diagnostics with real
  LiDAR selection metadata, Camera `METHOD_NOT_APPLICABLE`, and null PTF/DER
  fields;
- the complete training declaration records the versioned CoFormer optimizer recipe and
  all shared fields; a deliberate epoch/batch/fault/checkpoint mismatch becomes
  cross-protocol.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_coformer_adapter.py -q`

Expected: FAIL because CoFormer adapter/config are absent.

- [ ] **Step 3: Implement direct CoFormerNet adapter**

Build `CoFormerNet` from its detector registry path, use shared dataset/overlays/evaluator/evidence, and make the configuration explicitly LiDAR-only. Do not claim multimodal compatibility until a real four-raw-input implementation exists.
Training uses `tools/train.py`; every controlled val/test prediction uses the same
custom resumable prediction entrypoint and loop as FFNet, never `tools/test.py` or
the old wrapper.
The config builds `ResilientCoFormerNet`, whose bridge overrides the legacy input,
loss, predict and selection-loss paths to consume `BaselineLidarHistory` from the
data pipeline. It reuses CoFormer voxel/backbone/fusion/head modules but never
reconstructs or opens a raw path. It validates selected availability and rigid
alignment, wraps native loss keys with the same global validity decision, and
preserves the old `CoFormerNet` behavior for legacy configs.
Every returned sample receives the typed baseline diagnostics before entering the
shared resumable spool.

- [ ] **Step 4: Run tests and config build**

Run:

```bash
python -m pytest tests/resilient_v2x/test_coformer_adapter.py -q
python -c 'from mmengine.config import Config; Config.fromfile("configs/resilient_v2x/baselines/coformer_lidar.py")'
```

Expected: tests pass and config resolves.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/adapters/coformer.py transvision/models/detectors/resilient_v2x_baselines.py transvision/models/detectors/__init__.py configs/resilient_v2x/baselines/coformer_lidar.py tests/resilient_v2x/test_coformer_adapter.py
git commit -m "feat: adapt coformer to resilient v2x protocol"
```

---

### Task 6: 增加 DAIR cooperative BEVFusion config 与 capability verification

**Files:**

- Modify: `transvision/models/detectors/bevfusion.py`
- Modify: `transvision/models/detectors/__init__.py`
- Create: `transvision/experiments/resilient_v2x/adapters/bevfusion.py`
- Create: `configs/resilient_v2x/baselines/bevfusion_dair_cooperative.py`
- Create: `tests/resilient_v2x/test_bevfusion_adapter.py`

**Interfaces:**

The following interfaces are added to `BEVFusion`:

```python
class BEVFusionAdapter(BaselineAdapter):
    method = "bevfusion"

    @property
    def capabilities(self) -> AdapterCapabilities
    @property
    def training_protocol(self) -> TrainingProtocolDeclaration
    def build_config_overrides(self, context: AdapterContext) -> dict[str, object]
    def parse_predictions(self, output_path: Path) -> tuple[SamplePrediction, ...]

def loss_with_validity(
    self,
    inputs: Mapping[str, object],
    data_samples: Sequence[Det3DDataSample],
) -> tuple[ValidityAwareLosses, BatchValidityDecision]
def forward_selection_detection_loss(
    self,
    inputs: Mapping[str, object],
    data_samples: Sequence[Det3DDataSample],
) -> tuple[Tensor, int]
```

- [ ] **Step 1: Write registry, config, and four-input capability tests**

```python
def test_bevfusion_is_exported_from_detector_registry() -> None:
    from transvision.models.detectors import BEVFusion
    assert BEVFusion.__name__ == "BEVFusion"


def test_dair_config_has_no_nuscenes_or_missing_custom_import() -> None:
    config = Config.fromfile(
        "configs/resilient_v2x/baselines/bevfusion_dair_cooperative.py"
    )
    serialized = repr(config.to_dict())
    assert "NuScenes" not in serialized
    assert "projects.BEVFusion" not in serialized
    assert config.model["bbox_head"]["num_classes"] == 1
```

Use spies to verify both agents' LiDAR and Camera raw inputs are consumed. For each of the
four branches independently, perturb only that branch and assert the fused tensor changes;
then backpropagate a scalar fused output and assert a nonzero finite gradient reaches the
branch's encoded feature. This prevents an implementation that encodes and then discards
one branch. Also run a regression fixture proving the existing single-agent BEVFusion
path retains its prior output shape and loss/predict API. Do not satisfy the test by
relabeling inputs or duplicating ego tensors.
Also assert its `TrainingProtocolDeclaration` contains all shared fields, a hashed
versioned BEVFusion optimizer recipe, and no undeclared deviations before the adapter is
eligible for controlled rows.
Assert controlled loss uses `ValidityAwareLosses` through the custom DDP outer
wrapper and selection loss is computed by the identical real four-input
feature/head path for the shared AP/loss/epoch checkpoint selector.
Assert each prediction attaches `bevfusion` baseline diagnostics with all four
real source selections and explicit null PTF/DER fields.

- [ ] **Step 2: Run tests and capture current gaps**

Run: `python -m pytest tests/resilient_v2x/test_bevfusion_adapter.py -q`

Expected initially: FAIL because BEVFusion is not exported and current config is NuScenes-only with a missing custom import.

- [ ] **Step 3: Implement DAIR cooperative bridge**

Modify the registered `BEVFusion` detector itself to expose a DAIR cooperative four-branch
input path while preserving its existing single-agent compatibility. Reuse shared
LiDAR/Camera encoders, causal source selection, geometry alignment, and detection range.
BEVFusion may provide its own fusion layer, but must accept four real raw branches and the
shared overlays. `_forward()` may remain unused only if `loss()` and `predict()` are
complete and tested. This task is not complete while the four-input capability test is
red; an interim blocked adapter may be committed only on a separate diagnostic branch,
not as completion of this plan.
The controlled path implements the shared validity-aware loss and
`SelectionLossDetector` contracts with BEVFusion's native loss keys; it does not
substitute Resilient TransFusion losses or a proxy selection score.
Its controlled val/test path also uses
`tools/resilient_v2x/predict_attempt.py` and the shared resumable evidence loop;
`tools/test.py` is not a matrix prediction path.
Before returning predictions, BEVFusion attaches the typed baseline diagnostics
so evidence persistence is identical to Resilient and concat runs.

- [ ] **Step 4: Run BEVFusion tests and config build**

Run:

```bash
python -m pytest tests/resilient_v2x/test_bevfusion_adapter.py -q
python -c 'from mmengine.config import Config; Config.fromfile("configs/resilient_v2x/baselines/bevfusion_dair_cooperative.py")'
```

Expected: registry/config and genuine four-input capability tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/detectors/bevfusion.py transvision/models/detectors/__init__.py transvision/experiments/resilient_v2x/adapters/bevfusion.py configs/resilient_v2x/baselines/bevfusion_dair_cooperative.py tests/resilient_v2x/test_bevfusion_adapter.py
git commit -m "feat: adapt bevfusion for dair cooperative data"
```

---

### Task 7: 增加 V2X-ViT、CoBEVT、LRCP declarative external adapters

**Files:**

- Create: `transvision/experiments/resilient_v2x/adapters/external.py`
- Create: `configs/resilient_v2x/baselines/external/v2x_vit.json`
- Create: `configs/resilient_v2x/baselines/external/cobevt.json`
- Create: `configs/resilient_v2x/baselines/external/lrcp.json`
- Create: `tests/resilient_v2x/test_external_adapters.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ExternalManifestBinding:
    method: Literal["v2x_vit", "cobevt", "lrcp"]
    absolute_path: Path
    sha256: str

class ExternalBaselineAdapter(BaselineAdapter):
    @property
    def capabilities(self) -> AdapterCapabilities
    @property
    def training_protocol(self) -> TrainingProtocolDeclaration
    @classmethod
    def from_template(cls, path: Path) -> "ExternalBaselineAdapter"
    @classmethod
    def from_resolved_manifest(
        cls,
        binding: ExternalManifestBinding,
    ) -> "ExternalBaselineAdapter"
    def validate_assets(self, context: AdapterContext) -> AssetValidation
    def build_config_overrides(self, context: AdapterContext) -> dict[str, object]
    def build_initialization_artifact(
        self,
        context: AdapterContext,
        output_dir: Path,
    ) -> InitializationArtifact
    def parse_training_evidence(
        self,
        attempt_dir: Path,
    ) -> TrainingProtocolAssessment
    def parse_predictions(self, output_path: Path) -> tuple[SamplePrediction, ...]
    def predict_subset_command(
        self,
        context: AdapterContext,
        subset: ExternalPredictionSubset,
    ) -> tuple[str, ...]
    def parse_prediction_file(
        self,
        output_path: Path,
        expected_sample_id: str,
        selections: tuple[BranchSelection, ...],
    ) -> BaselineSampleRecord
```

**Checked-in unresolved template shape:**

```json
{
  "schema_version": 1,
  "kind": "external-baseline-template",
  "method": "v2x_vit",
  "required_materialization_fields": [
    "repository.url",
    "repository.path",
    "repository.commit",
    "repository.license",
    "environment_manifest",
    "commands.prepare",
    "commands.predict",
    "commands.train_when_local",
    "initialization_contract",
    "training_evidence_parser",
    "output_parser",
    "prediction_contract",
    "checkpoint_contract",
    "declared_capabilities",
    "training_protocol"
  ]
}
```

`baseline_template.schema.json` allows only this minimal unresolved shape,
requires the exact canonical ordered field list above, and contains no executable paths,
commands, hashes, capabilities or training values. These checked-in templates
always materialize as blocked by themselves.

A user supplies a separate run-local resolved manifest containing concrete
repository URL/path/40-character commit/license, environment-manifest path and
SHA-256, mode-valid argv arrays, allow-listed output parser, declared capabilities,
an explicit per-sample prediction contract, and a discriminated
upstream/local-training checkpoint contract. The resolved
schema's `oneOf` requires non-empty prepare/predict argv for both modes; local
training additionally requires a non-empty train argv and forbids an input
checkpoint, while upstream requires URI/SHA-256 and forbids a train argv.
`baseline_manifest.schema.json` validates that resolved object with no null,
sentinel or placeholder values and directly references the one canonical
`training_protocol.schema.json` from the evaluation/evidence plan; it may not copy
a relaxed local schema. Integer, numeric, tuple/array and hash fields therefore
retain their canonical types. Only a resolved manifest that passes this strict
schema can proceed to asset validation.

For `mode=local_training`, `initialization_contract` additionally requires a
deterministic argv template, allow-listed state-inventory parser, seed/config/
source provenance, `{initialization_manifest}` token, and an attempt-local output
state. The parser canonicalizes parameter name/dtype/shape/bytes into the shared
`InitializationArtifact`; the train command must load that exact state and emit a
pre-optimizer re-hash. `training_evidence_parser` is likewise allow-listed under
the repository adapter package and must produce the shared epoch/update/sampler/
augmentation/precision/nonfinite/checkpoint-selector evidence. Upstream mode
forbids both training and initialization commands.

- [ ] **Step 1: Write blocked-template and resolved-manifest tests**

```python
@pytest.mark.parametrize("method", ["v2x_vit", "cobevt", "lrcp"])
def test_checked_in_external_template_is_blocked(method: str) -> None:
    adapter = ExternalBaselineAdapter.from_template(template_path(method))
    validation = adapter.validate_assets(empty_context())
    assert validation.ready is False
    assert set(validation.missing_fields) >= {
        "repository.path",
        "repository.commit",
        "repository.license",
        "checkpoint_contract.mode",
        "initialization_contract",
        "training_evidence_parser",
        "training_protocol.optimizer_recipe_sha256",
        "training_protocol.augmentation_recipe_sha256",
        "training_protocol.sampler_recipe_sha256",
        "training_protocol.update_scheduler_sha256",
        "training_protocol.resolved_config_sha256",
        "training_protocol.protocol_sha256",
    }
```

Add a fully resolved temporary manifest fixture and assert commit/hash/license/environment/output parser validation, argv execution without shell, prediction contract, and cross-protocol classification for public checkpoints. Test the checkpoint discriminated union: `mode=upstream` requires URI and pre-run SHA-256 and is cross-protocol; `mode=local_training` forbids an input checkpoint requirement, declares only an attempt-relative output, and becomes controlled only after accepted shared-training evidence resolves the checkpoint hash. In a temporary Git repository also assert that:

- a clean exact HEAD records both the 40-character commit and `HEAD^{tree}`;
- any tracked modification is rejected;
- any untracked file, including an executable or source file, is rejected;
- dirty or uninitialized submodules are rejected;
- asset validation is repeated immediately before every prepare/train/predict command.
- ignored Python, shared-library, config, shell, or `.pth` files are rejected;
- commands execute from an attempt-local, hash-recorded Git-object snapshot rather than
  the user-provided working directory;
- source file mutation during prepare/train/predict is detected before artifact acceptance.
- controlled local training additionally requires exact argv tokens for the
  immutable remaining-sample manifest and attempt-local per-sample output
  directory; output filenames are canonical sample IDs, each payload is written
  atomically with a detached SHA-256, and the parser must return that same ID.
  Whole-split output or a parser that cannot validate one file independently
  remains cross-protocol and reserved-test-ineligible.
- external local training dispatches the validated snapshot-bound adapter argv,
  not `tools/train.py`, but still uses the shared training `AttemptStore`,
  `TrainingRealization`, initialization artifact, protocol assessment,
  checkpoint-selection replay, training manifest and success pointer. A backend
  spy proves no adapter can bypass any evidence phase;
- the external initialization parser produces a canonical state inventory/hash,
  the child re-emits the same pre-optimizer hash, and seed/state mutation changes
  materialized identity. A missing initialization/training-evidence parser is
  cross-protocol and never controlled;
- upstream mode executes no initialization/train argv. Retry/force and accepted
  checkpoint propagation match built-ins exactly;
- distinct adapter-execution and shared-evaluator environment manifests/hardware
  hashes are both verified and recorded; swapping, omitting, or flattening either
  identity fails controlled materialization.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_external_adapters.py -q`

Expected: FAIL because external adapter/templates are absent.

- [ ] **Step 3: Implement declarative materialization**

The adapter does not clone, install, or download automatically. It validates a user-provided local repository at the exact commit, validates its independent environment manifest, and emits deterministic blocked/ready plan state. Upstream checkpoint hashes are checked before prediction; locally retrained checkpoint hashes are produced only by the immutable successful training attempt. Validation executes read-only Git commands equivalent to:

```text
git status --porcelain=v1 --untracked-files=all
git status --porcelain=v1 --ignored=matching --untracked-files=all
git rev-parse HEAD
git rev-parse HEAD^{tree}
git submodule status --recursive
```

The first command must be empty; the ignored scan must contain no importable/executable
source or config artifact; every submodule status must be clean and pinned. Materialize
the exact Git objects (including pinned submodules) into
an attempt-local unique `source-snapshot.staging-<nonce>/`, compute a canonical
digest over every file/path/mode, write the canonical snapshot commit marker
inside that staging tree, fsync the marker, inventory, files and directories, and
atomically rename the complete marked tree to `<attempt>/source-snapshot/`. On
resume, only staging owned by that same
attempt and lacking a commit marker may be removed/rebuilt; a published snapshot
is reused only after exact inventory/hash verification and is never modified.
Crash tests cover copy, submodule materialization, inventory, fsync, rename and
commit-marker boundaries and prove a half snapshot never occupies the final path.
Controlled snapshots reject every symlink, FIFO, device, socket and any regular
file whose link count/inode inventory proves a hardlink outside the snapshot
root. Command executable, cwd, `PYTHONPATH` entries and every repo-local argv path
are resolved and must remain regular files/directories beneath the published
snapshot; no import or execution path may follow outside it. Fixtures with a Git-
tracked symlink to mutable external code, a cross-root hardlink and an outside
interpreter/script path all block before command launch.
Then set
`PYTHONDONTWRITEBYTECODE=1`, and execute only from that snapshot with build/output
directories elsewhere in the same attempt. Record commit, tree object, submodule commits,
and snapshot digest in model provenance. Verify the snapshot before and after every
command; any source mutation fails the attempt.

Commands must be JSON arrays of strings. Prepare/predict arrays are always
non-empty; train is non-empty only for `mode=local_training` and is forbidden for
`mode=upstream`. Empty required arrays block materialization, and unresolved
templates never execute. Output parser is an allow-listed import path under
`transvision.experiments.resilient_v2x.adapters`, not arbitrary code from the manifest.

For a controlled external run, `predict_subset_command()` replaces only two exact
argv tokens—`{sample_manifest}` and `{per_sample_output_dir}`—with
attempt-contained paths; no shell expansion occurs. The shared matrix runner
writes the current manifest-ordered remaining IDs, scans any detached-hash-valid
per-sample outputs left by an interruption, combines each parsed prediction with
the matching frozen four-branch selections through
`baseline_inference_diagnostics()`, and promotes the resulting
`BaselineSampleRecord` through `PredictAttempt.record_rank_sample()`. It rewrites
the remaining subset and only
then relaunches. After a successful child exit it repeats that promotion before
evaluation. Thus a durable completed sample is never recomputed; conflicting,
extra, missing-ID or overwritten files terminally fail the attempt. A resolved
external manifest lacking this subset/output contract can produce a declared
cross-protocol reference only and can never acquire a reserved-test claim.
All upstream/cross-protocol evaluation is strictly `split="val"`; requesting
`split="test"` is blocked before source snapshot, checkpoint, ground truth,
dataloader, predictor or evaluator access.
Prediction and schema-valid not-applicable PTF/DER diagnostics therefore always
enter the evidence store as one unit.

- [ ] **Step 4: Run external adapter tests**

Run: `python -m pytest tests/resilient_v2x/test_external_adapters.py -q`

Expected: all templates are valid-but-blocked; resolved fixture is accepted and remains cross-protocol without local shared training.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/adapters/external.py configs/resilient_v2x/baselines/external tests/resilient_v2x/test_external_adapters.py
git commit -m "feat: add external v2x baseline manifests"
```

---

### Task 8: 将 baseline jobs 接入统一 matrix、evaluator 和 evidence

**Files:**

- Modify: `transvision/experiments/resilient_v2x/matrix.py`
- Modify: `transvision/experiments/resilient_v2x/inference.py`
- Modify: `transvision/experiments/resilient_v2x/aggregate.py`
- Modify: `transvision/experiments/resilient_v2x/export.py`
- Modify: `tools/resilient_v2x/run_matrix.py`
- Modify: `tests/resilient_v2x/test_run_matrix.py`
- Modify: `tests/resilient_v2x/test_matrix.py`
- Modify: `tests/resilient_v2x/test_aggregate.py`
- Modify: `tests/resilient_v2x/test_export.py`
- Create: `tests/resilient_v2x/test_baseline_integration.py`

**Interfaces:**

```python
class ExternalLocalTrainingBackend(TrainingBackend):
    name: Literal["external_local"] = "external_local"

    def build_launch(
        self,
        node: PlanNode,
        plan: TrainingPlan,
        context: ExecutionContext,
    ) -> TrainingLaunch

def register_baseline_training_backends() -> None

def aggregate_collection(
    evidence_root: Path,
    matrix: Literal["baselines", "reproduction"],
    split: Literal["val", "test"],
) -> AggregateCollectionIndex
```

- [ ] **Step 1: Write exact baseline count and compatibility tests**

```python
def test_multimodal_and_lidar_only_counts_per_split() -> None:
    assert controlled_evaluation_count(
        capabilities=multimodal_capabilities(),
        seeds=(0, 1, 2),
    ) == 36
    assert controlled_evaluation_count(
        capabilities=lidar_only_capabilities(),
        seeds=(0, 1, 2),
    ) == 24


def test_val_and_test_double_evaluation_without_retraining() -> None:
    nodes = build_controlled_baseline_fixture(splits=("val", "test"))
    assert count_training_nodes(nodes) == 3
    assert count_evaluation_nodes(nodes) == 72


@pytest.mark.parametrize("method", ["ffnet", "coformer"])
def test_lidar_baseline_exact_dag(method: str) -> None:
    nodes = build_resolved_baseline_fixture(method, splits=("val", "test"))
    assert count_training_nodes(nodes) == 3
    assert count_evaluation_nodes(nodes, split="val") == 24
    assert count_evaluation_nodes(nodes, split="test") == 24
    assert count_evaluation_nodes(nodes) == 48


def test_bevfusion_exact_dag() -> None:
    nodes = build_resolved_baseline_fixture("bevfusion", splits=("val", "test"))
    assert count_training_nodes(nodes) == 3
    assert count_evaluation_nodes(nodes, split="val") == 36
    assert count_evaluation_nodes(nodes, split="test") == 36
    assert count_evaluation_nodes(nodes) == 72


@pytest.mark.parametrize("method", ["v2x_vit", "cobevt", "lrcp"])
def test_unresolved_external_has_one_blocked_manifest_and_no_metrics(method: str) -> None:
    nodes = build_checked_in_external_fixture(method)
    blocked = [node for node in nodes if node.kind == "adapter_preflight"]
    assert len(blocked) == 1
    assert blocked[0].state == "blocked"
    assert blocked[0].payload["blocked_job_manifest"] == (
        f"blocked/{method}/blocked_job_manifest.json"
    )
    assert count_training_nodes(nodes) == 0
    assert count_evaluation_nodes(nodes) == 0
```

Add tests that:

- FFNet/CoFormer never enter multimodal controlled rows;
- BEVFusion enters only when four-input capability proof passes;
- public checkpoint plans are cross-protocol;
- public/upstream and every other cross-protocol plan is val-only. A parameterized
  `split=test` test asserts zero checkpoint/GT/dataloader/predictor/evaluator calls
  and no test artifact, metric or reference node;
- blocked external assets produce no metric/evaluator nodes;
- local training creates one checkpoint per seed reused across conditions and val/test;
- built-in baseline training still uses the shared built-in MMEngine backend;
  a resolved local external manifest registers exactly one `external_local`
  `TrainingBackend`, launches only snapshot-bound allow-validated argv/parser,
  and produces the same typed initialization, protocol assessment, training
  manifest, accepted checkpoint and realization ledger. Duplicate registration,
  an unregistered backend, or an upstream-checkpoint training launch fails before
  attempt creation;
- all methods use `resilient-v2x-ap-v1`;
- adapter cannot override shared split, class, range, fault, latency, or evidence root.
- a locally trained job is controlled only when the complete
  `TrainingProtocolDeclaration` passes; missing fields or any shared-protocol deviation
  produce a cross-protocol node and a machine-readable deviation report;
- a declaration that claims 50 epochs but whose accepted training manifest records 49,
  a different sampler/update count, or a mismatched resolved-config digest is rejected
  before any controlled evaluation node is created;
- external source commit/tree/submodule identities propagate into every command manifest
  and model provenance record.
- a controlled external prediction interrupted after two valid per-sample files
  resumes the same attempt, promotes those two before launch, supplies only the
  remaining manifest IDs to the child and never recomputes them; a whole-split
  external command is excluded from controlled/test nodes;
- `--matrix baselines` enumerates all six adapter methods, preserving a blocked
  preflight node for each unresolved external method;
- repeatable `--baseline-manifest method=/abs/file.json` resolves exactly that
  external method through its strict file/path hash, while an omitted method
  remains blocked. Relative paths, duplicates, unknown/mismatched methods,
  post-parse mutation, and implicit directory discovery are rejected before a
  ready node or command is emitted;
- `--matrix reproduction` is the union of the core `all` DAG and `baselines`
  DAG, with no dropped/duplicated template binding; `--matrix all` retains its
  already-frozen core-only `39/198/72` meaning.
- local controlled FFNet/CoFormer aggregate exactly three seeds over 24 rows per
  split and BEVFusion over 36; PDR is paired within one method/seed only, and
  candidate-reference differences require identical controlled
  protocol/evaluator/split/hardware hashes;
- LiDAR-only tables expose only display rows LiDAR-Full/LiDAR-Fail, backed by
  canonical internal `Full`/`L-Fail` keys, and never synthesize
  C-Fail cells; multimodal and LiDAR-only coverage manifests remain distinct;
- upstream checkpoints are written only to a labeled val cross-protocol reference
  table and are never used in controlled pairwise differences, rankings, test
  claims or test prediction. Unresolved adapters appear only as hashed blocked-job
  records;
- val `reproduction` aggregation includes separate main, ablation, sensitivity,
  duration, arrival-relative, concat, and baseline controlled tables. Reserved
  test `reproduction` includes only the frozen core main+concat families and
  test-eligible baseline families; val-only collections receive explicit
  `not_in_reserved_test_scope` markers and are never read. Val includes the
  cross-protocol/blocked indexes; test includes only blocked and explicit
  `val_only_reference_not_in_reserved_test_scope` metadata, never upstream test
  metrics. Every expected result slot is covered exactly once and numeric
  placeholders are forbidden;
- evidence export consumes and hashes that aggregate index, carries its controlled
  tables, cross-protocol reference table, and blocked manifests into the candidate
  bundle, and refuses an incomplete baseline family before emitting a paper patch.

- [ ] **Step 2: Run integration tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_baseline_integration.py tests/resilient_v2x/test_matrix.py -q`

Expected: FAIL until matrix dispatch knows the adapter registry.

- [ ] **Step 3: Implement adapter registry dispatch**

Use a static mapping:

```python
ADAPTERS: dict[str, type[BaselineAdapter]] = {
    "ffnet": FFNetAdapter,
    "coformer": CoFormerAdapter,
    "bevfusion": BEVFusionAdapter,
    "v2x_vit": ExternalBaselineAdapter,
    "cobevt": ExternalBaselineAdapter,
    "lrcp": ExternalBaselineAdapter,
}
```

Extend `PlanNode.kind` with `adapter_preflight`. Matrix materialization validates
assets and compatibility before train/evaluate nodes. Each adapter always gets one
preflight node. A blocked preflight writes one canonical, hashed
`blocked_job_manifest.json` containing method, template bindings, missing assets,
compatibility reasons and source/config hashes; it has no downstream metric node.
The shared evaluator and evidence store run after adapter prediction parsing;
adapters cannot return pre-computed AP for controlled rows.
Built-in adapters use the shared MMEngine resumable loop. A locally trained
external adapter is controlled/test-eligible only through
`ControlledExternalPredictionAdapter`; its subset command and per-sample parser
feed the same `PredictAttempt` spool/promotion/evaluation path. No adapter-specific
resume or bulk overwrite path exists.

This task is the only phase that imports adapter implementations into the generic
runner. It registers the `external_local` `TrainingBackend` against the stable
evaluation Task-5 hook, builds `TrainingLaunch` from the immutable resolved
manifest and adapter-execution environment, and parses the standardized training
evidence through the same `execute_training_attempt()` finalization path. It also
extends runner dispatch for `adapter_preflight`. Therefore core evaluation Task 5
is independently implementable before baseline Tasks 3–8 and no cross-plan
execution cycle exists.

The runner adds two matrix names:

- `baselines`: exactly the six methods in `ADAPTERS`, with the per-method counts
  above when locally resolved and controlled;
- `reproduction`: deterministic union of core `all` plus `baselines`, including
  blocked external preflights.

The command manifest records which matrix expansion supplied every node. An
upstream public checkpoint produces only explicitly declared val cross-protocol
reference nodes and never contributes to controlled counts or any test DAG.

`aggregate_collection()` resolves only accepted attempt pointers through the
shared evidence APIs. It groups controlled baseline records by capability class
and exact template semantics, enforces seed set `(0,1,2)`, delegates AP/PDR/method
difference arithmetic to the core aggregator, and writes a create-exclusive
coverage index. It writes upstream references and blocked preflights into separate
non-numeric artifacts. `build_candidate_registry_patch()` accepts this verified
index rather than rediscovering baseline files, so export cannot silently omit an
adapter or relabel a reference as controlled.

- [ ] **Step 4: Run baseline integration tests**

Run: `python -m pytest tests/resilient_v2x/test_adapter_contract.py tests/resilient_v2x/test_ffnet_adapter.py tests/resilient_v2x/test_coformer_adapter.py tests/resilient_v2x/test_bevfusion_adapter.py tests/resilient_v2x/test_external_adapters.py tests/resilient_v2x/test_baseline_integration.py tests/resilient_v2x/test_matrix.py tests/resilient_v2x/test_aggregate.py tests/resilient_v2x/test_export.py -q`

Expected: all adapter, capability, count, and matrix tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/matrix.py transvision/experiments/resilient_v2x/inference.py transvision/experiments/resilient_v2x/aggregate.py transvision/experiments/resilient_v2x/export.py tools/resilient_v2x/run_matrix.py tests/resilient_v2x/test_run_matrix.py tests/resilient_v2x/test_matrix.py tests/resilient_v2x/test_aggregate.py tests/resilient_v2x/test_export.py tests/resilient_v2x/test_baseline_integration.py
git commit -m "feat: orchestrate resilient v2x baselines"
```

---

### Task 9: 最终运行文档、全仓回归和 CUDA/data 验收清单

**Files:**

- Create: `docs/resilient_v2x/README.md`
- Create: `docs/resilient_v2x/baselines.md`
- Create: `docs/resilient_v2x/environment.md`
- Create: `docs/resilient_v2x/acceptance.md`
- Modify: `README.md`
- Create: `tests/resilient_v2x/test_documented_commands.py`
- Create: `tests/resilient_v2x/test_cuda_acceptance.py`
- Create: `tests/resilient_v2x/golden/planned_files.txt`

**Interfaces:**

No new production Python API is introduced. Documentation, golden inventory and
target-platform acceptance tests consume the interfaces frozen above.

- [ ] **Step 1: Add a documentation command parser test**

Extract fenced `bash` commands marked `testable` and assert all referenced repo-local
tools/configs exist, every `run_matrix.py` example supplies absolute
data/evidence/environment/manifest placeholders, and no example invokes legacy wrappers or
`./cache`.

The parsed command set must be non-empty and contain, by normalized command ID, exactly
the required workflow classes: environment bootstrap, final Docker build/check, data
prepare, core dry-run, diagnostic smoke, `reproduction` val, `reproduction` test,
aggregate, paired profile, qualitative render, and evidence export. Assert their
document order is val → val aggregate/profile/qualitative → test → test aggregate →
export. Invoke each Python CLI's real argument parser with fixture paths and
`--dry-run`/mock execution; checking only that a filename exists is insufficient.
`planned_files.txt` is a human-reviewed sorted list of every
implementation/schema/config/script/test/document promised by the four plans. The test
compares it exactly with the in-scope `git ls-files` set, allowing only an explicit
legacy-file allowlist.

- [ ] **Step 2: Run the documentation test and confirm missing workflow docs**

Run: `python -m pytest tests/resilient_v2x/test_documented_commands.py -q`

Expected: FAIL because the final README, baseline, environment, and acceptance command blocks do not yet exist.

- [ ] **Step 3: Document the complete workflow**

The docs must cover:

- environment creation from lock and Docker;
- dataset preparation without downloading data;
- teacher → student → val matrix → val aggregate/profile/qualitative acceptance
  → freeze → reserved test → test aggregate → export;
- exact 36 Resilient teacher/student training jobs plus 3 capacity-matched concat jobs;
- 162 unique Resilient val evaluations, 36 concat val evaluations, and after freeze the
  corresponding 36 Resilient main plus 36 concat reserved-test evaluations;
- FFNet/CoFormer LiDAR-only reference status;
- BEVFusion four-input capability proof;
- V2X-ViT/CoBEVT/LRCP resolved-manifest procedure;
- controlled versus cross-protocol labels;
- immutable evidence root outside Git;
- current Mac checks versus Linux/CUDA/data checks.

`test_cuda_acceptance.py` must define markers/tests named
`test_cuda_custom_ops_and_one_batch` and `test_real_data_manifest_one_batch`. They skip
with an explicit `not_executed` reason unless target assets are provided; when enabled,
they assert both custom-op imports, one finite train loss, one finite prediction batch,
and one evaluator result through the evidence runner. Documentation may filter these
exact names, not an undefined generic marker.

- [ ] **Step 4: Run all CPU/static tests**

Run:

```bash
python -m pytest tests/resilient_v2x -q
python -m compileall -q transvision tools/resilient_v2x
bash -n scripts/build_resilient_v2x_ops.sh scripts/train_resilient_v2x_teacher.sh scripts/train_resilient_v2x_student.sh scripts/evaluate_resilient_v2x.sh
git diff --check
```

Expected: all locally runnable tests pass; compileall, shell syntax, and diff checks are silent.

- [ ] **Step 5: Run source and provenance gates**

Run:

```bash
rg -n 'TODO|TBD|NotImplemented|pass$' transvision/experiments/resilient_v2x transvision/models/resilient_v2x transvision/evaluation/metrics/resilient_v2x_metric.py tools/resilient_v2x configs/resilient_v2x tests/resilient_v2x environments/resilient_v2x docs/resilient_v2x
rg -n 'shutil\.rmtree|zero.*box|last_checkpoint|eval_vic|shell=True|projects\.BEVFusion|NuScenes' transvision/experiments/resilient_v2x tools/resilient_v2x configs/resilient_v2x
git ls-files configs/resilient_v2x transvision/experiments/resilient_v2x transvision/models/resilient_v2x transvision/evaluation/metrics/resilient_v2x_metric.py tools/resilient_v2x scripts environments/resilient_v2x tests/resilient_v2x docs/resilient_v2x
```

Expected: first two scans return no forbidden production matches; every planned artifact is tracked.

- [ ] **Step 6: Execute target-platform smoke in order when assets exist**

Run on Linux/CUDA with DAIR and local weights:

```bash
docker build --platform linux/amd64 -f environments/resilient_v2x/Dockerfile -t resilient-v2x:final .
docker run --rm --platform linux/amd64 --gpus all resilient-v2x:final python tools/resilient_v2x/check_environment.py --mode controlled --require-cuda --require-custom-ops
bash scripts/build_resilient_v2x_ops.sh
python tools/resilient_v2x/capture_environment.py --world-size 1 --evidence-root /abs/evidence --output /abs/evidence/environment-ws1.json
python tools/resilient_v2x/check_environment.py --mode controlled --require-cuda --require-custom-ops --environment-manifest /abs/evidence/environment-ws1.json
python -m pytest tests/resilient_v2x/test_cuda_acceptance.py -k 'test_cuda_custom_ops_and_one_batch or test_real_data_manifest_one_batch' -q
CUDA_VISIBLE_DEVICES=0 python tools/resilient_v2x/run_matrix.py --matrix smoke --split val --seed 0 --smoke-max-epochs 2 --data-root /abs/data/DAIR-V2X-C --temporal-manifest /abs/data/temporal_manifest.json --evidence-root /abs/evidence --environment-manifest /abs/evidence/environment-ws1.json --launcher none
```

Expected: the final source state builds in the pinned image and imports both custom ops;
the evidence-gated diagnostic smoke records exactly two
teacher epochs, two student epochs, and exactly the three diagnostic val evaluations
Full@0, L-Fail@0, and Full@300 under a non-controlled smoke protocol. No smoke artifact is
accepted for profiling, aggregation, test access, or paper export. Missing assets remain
`not executed`.

- [ ] **Step 7: Execute full controlled acceptance only after the smoke passes**

Run only when the full 50-epoch matrix is intended and all three-seed assets are
available:

```bash
python tools/resilient_v2x/capture_environment.py --world-size 4 --evidence-root /abs/evidence --output /abs/evidence/environment-ws4.json
python tools/resilient_v2x/check_environment.py --mode controlled --require-cuda --require-custom-ops --environment-manifest /abs/evidence/environment-ws4.json
python tools/resilient_v2x/run_matrix.py --matrix reproduction --split val --baseline-manifest v2x_vit=/abs/manifests/v2x_vit.json --baseline-manifest cobevt=/abs/manifests/cobevt.json --baseline-manifest lrcp=/abs/manifests/lrcp.json --data-root /abs/data/DAIR-V2X-C --temporal-manifest /abs/data/temporal_manifest.json --evidence-root /abs/evidence --environment-manifest /abs/evidence/environment-ws4.json --launcher pytorch --nproc-per-node 4
python tools/resilient_v2x/aggregate_results.py --evidence-root /abs/evidence --matrix reproduction --split val
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 2 python tools/resilient_v2x/profile.py --evidence-root /abs/evidence --environment-manifest /abs/evidence/environment-ws1.json --full-result-slot main.full.0ms.seed0 --concat-result-slot baseline.concat.full.0ms.seed0 --temporal-manifest /abs/data/temporal_manifest.json --dataset-root /abs/data/DAIR-V2X-C --profile-corpus /abs/evidence/profile/profile_corpus.json
python tools/resilient_v2x/render_qualitative.py --evidence-root /abs/evidence --split val --sample-count 12
python tools/resilient_v2x/run_matrix.py --matrix reproduction --split test --baseline-manifest v2x_vit=/abs/manifests/v2x_vit.json --baseline-manifest cobevt=/abs/manifests/cobevt.json --baseline-manifest lrcp=/abs/manifests/lrcp.json --data-root /abs/data/DAIR-V2X-C --temporal-manifest /abs/data/temporal_manifest.json --evidence-root /abs/evidence --environment-manifest /abs/evidence/environment-ws4.json --launcher pytorch --nproc-per-node 4
python tools/resilient_v2x/aggregate_results.py --evidence-root /abs/evidence --matrix reproduction --split test
python tools/resilient_v2x/export_evidence.py --matrix reproduction --evidence-root /abs/evidence --paper-repository /abs/path/ResilientV2X --output-dir /abs/candidate-evidence
```

Expected: the complete val matrix, three-seed aggregation, mandatory FP32-headline/
AMP-supplemental paired profile and qualitative artifacts all finish before
architecture/config/checkpoint freeze and any test access. Each reserved-test method
then acquires its entire 36- or 24-run claim family before the first prediction;
aggregation sees exactly seeds 0/1/2; the accepted seed-0 Full and concat runs share the
one audited 32-sample profiling corpus; export consumes complete accepted families. A
blocked external baseline remains a blocked node, never a fabricated metric. This step
is `not_executed` until those expensive assets and runs exist.

- [ ] **Step 8: Commit final docs**

```bash
git add docs/resilient_v2x README.md tests/resilient_v2x/test_documented_commands.py tests/resilient_v2x/test_cuda_acceptance.py tests/resilient_v2x/golden/planned_files.txt
git commit -m "docs: complete resilient v2x reproduction guide"
```

- [ ] **Step 9: Final repository verification**

Run:

```bash
git status --short
git diff --cached --check
git show --check --oneline HEAD
```

Expected: no unplanned worktree changes; checks produce no errors.

## Completion Gate

This plan is complete only when:

- environment lock has artifact hashes and Docker/build/capture checks exist;
- the target Docker image has actually built and its CUDA/custom-op check has succeeded
  before target-environment acceptance is claimed;
- setup metadata no longer imports torch or rewrites tracked files;
- six baselines have explicit adapters/manifests and capability reports;
- `baselines` and `reproduction` matrices enumerate every one of those six methods;
  FFNet/CoFormer resolve to `3 train + 24 val + 24 test`, BEVFusion to
  `3 + 36 + 36`, and each unresolved external method produces a hashed blocked
  preflight manifest with no metric node;
- FFNet/CoFormer remain LiDAR-only, BEVFusion is genuinely four-input compatible, and external assets fail closed;
- baseline jobs reuse checkpoints, use the shared evaluator/evidence chain, and never invoke destructive/zero-box legacy wrappers;
- schema-valid `reproduction` val/test aggregate indexes have exact three-seed
  controlled baseline coverage, separate cross-protocol references, and every
  unresolved blocked manifest; their hashes are accepted evidence;
- the Full-versus-concat FP32+AMP profile has one verified success pointer, and
  `export_evidence.py --matrix reproduction` binds that profile plus the exact val
  and test reproduction-index hashes into its export manifest;
- the documented-workflow parser test passes the frozen command classes and order,
  including data root, world-size-specific environment manifests, optional external
  manifest bindings, split-aware aggregation, profiling, and matrix-selected export;
- controlled labels require a complete shared training-protocol declaration, and external
  repositories are clean with commit/tree/submodule provenance;
- all locally runnable tests pass and unavailable CUDA/data/external assets are reported as `not executed` or `blocked`, never as reproduced results.
