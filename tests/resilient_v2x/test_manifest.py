from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_EXPORTS = (
    "ManifestError",
    "MANIFEST_SCHEMA_VERSION",
    "OFFICIAL_COOPERATIVE_SPLIT_SHA256",
    "RawSliceRecord",
    "GroundTruthBoxRecord",
    "TemporalSampleRecord",
    "ReleaseInventoryEntry",
    "PreparedArtifactRecord",
    "TemporalManifest",
    "canonical_json_bytes",
    "content_sha256",
    "build_release_inventory",
    "release_inventory_sha256",
    "load_temporal_manifest",
)
PACKAGE_EXPORTS = (
    *MODULE_EXPORTS,
    "SUPPROTED_DATASETS",
    "BEVLoadMultiViewImageFromFiles",
)
BRANCH_ORDER = (
    ("ego", "lidar"),
    ("rsu", "lidar"),
    ("ego", "camera"),
    ("rsu", "camera"),
)
FIXTURE_SPLIT_SHA256 = "a" * 64
PRIMARY_CALIBRATION_SHA256 = "b" * 64
IDENTITY_4X4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
CAMERA_INTRINSIC = (
    (1000.0, 0.0, 960.0),
    (0.0, 1000.0, 540.0),
    (0.0, 0.0, 1.0),
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _inventory_sha256(entries: list[dict[str, object]]) -> str:
    return _sha256_bytes(_canonical_bytes(entries))


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    payload["dataset_release_sha256"] = _inventory_sha256(
        payload["release_inventory"]  # type: ignore[arg-type]
    )
    content_payload = copy.deepcopy(payload)
    content_payload.pop("content_sha256", None)
    payload["content_sha256"] = _sha256_bytes(_canonical_bytes(content_payload))
    return payload


def _entry(
    relative_path: str, *, sha256: str | None = None, size: int = 1
) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "size": size,
        "sha256": sha256 or _sha256_bytes(relative_path.encode("utf-8")),
    }


def _slice(
    *,
    sequence_id: str,
    n_s: int,
    agent: str,
    modality: str,
    capture_timestamp_us: int,
) -> dict[str, object]:
    frame_id = f"{agent}-{modality}-{n_s}"
    relative_path = f"raw/{frame_id}.{'pcd' if modality == 'lidar' else 'jpg'}"
    return {
        "agent": agent,
        "modality": modality,
        "n_s": n_s,
        "tau_s_ms": n_s * 100,
        "capture_timestamp_us": capture_timestamp_us,
        "frame_id": frame_id,
        "packet_id": f"{sequence_id}:{n_s}:{agent}:{modality}",
        "relative_path": relative_path,
        "world_from_agent": [list(row) for row in IDENTITY_4X4],
        "agent_from_sensor": [list(row) for row in IDENTITY_4X4],
        "calibration_relative_path": "calib/primary.json",
        "calibration_sha256": PRIMARY_CALIBRATION_SHA256,
        "camera_intrinsic": (
            None if modality == "lidar" else [list(row) for row in CAMERA_INTRINSIC]
        ),
        "payload_valid": True,
        "pose_valid": True,
        "calibration_valid": True,
    }


def _source_tick(
    sequence_id: str, n_s: int, base_timestamp_us: int
) -> list[dict[str, object]]:
    return [
        _slice(
            sequence_id=sequence_id,
            n_s=n_s,
            agent=agent,
            modality=modality,
            capture_timestamp_us=base_timestamp_us + branch_index * 10_000,
        )
        for branch_index, (agent, modality) in enumerate(BRANCH_ORDER)
    ]


def _valid_payload() -> dict[str, object]:
    sequence_id = "seq-0"
    ticks = (
        _source_tick(sequence_id, 0, 1_000_000),
        _source_tick(sequence_id, 1, 1_100_000),
    )
    samples = [
        {
            "sample_id": "sample-0",
            "sequence_id": sequence_id,
            "split": "train",
            "n_t": 0,
            "tau_t_ms": 0,
            "source_slices": copy.deepcopy(ticks[0]),
            "annotation_path": "labels/sample-0.json",
            "annotation_sha256": _sha256_bytes(b"label-0"),
            "ground_truth": [
                {
                    "class_name": "Car",
                    "x": 10.0,
                    "y": 0.0,
                    "z_bottom": 0.0,
                    "length": 4.0,
                    "width": 2.0,
                    "height": 2.0,
                    "yaw": 0.0,
                    "source_annotation_index": 0,
                }
            ],
        },
        {
            "sample_id": "sample-1",
            "sequence_id": sequence_id,
            "split": "train",
            "n_t": 1,
            "tau_t_ms": 100,
            "source_slices": copy.deepcopy([*ticks[0], *ticks[1]]),
            "annotation_path": "labels/sample-1.json",
            "annotation_sha256": _sha256_bytes(b"label-1"),
            "ground_truth": [],
        },
    ]
    inventory_by_path = {
        "calib/primary.json": _entry(
            "calib/primary.json",
            sha256=PRIMARY_CALIBRATION_SHA256,
        ),
        "calib/secondary-camera-intrinsic.json": _entry(
            "calib/secondary-camera-intrinsic.json"
        ),
        "metadata/cooperative-data-info.json": _entry(
            "metadata/cooperative-data-info.json"
        ),
        "poses/secondary-world-pose.json": _entry("poses/secondary-world-pose.json"),
        "labels/sample-0.json": _entry(
            "labels/sample-0.json",
            sha256=samples[0]["annotation_sha256"],  # type: ignore[arg-type]
        ),
        "labels/sample-1.json": _entry(
            "labels/sample-1.json",
            sha256=samples[1]["annotation_sha256"],  # type: ignore[arg-type]
        ),
    }
    prepared_by_source: dict[str, dict[str, object]] = {}
    for tick in ticks:
        for source_slice in tick:
            relative_path = source_slice["relative_path"]
            inventory_by_path[relative_path] = _entry(relative_path)  # type: ignore[index]
            if source_slice["modality"] == "lidar":
                prepared_by_source[relative_path] = {
                    "source_relative_path": relative_path,
                    "prepared_relative_path": (
                        f"prepared/{Path(relative_path).with_suffix('.bin').as_posix()}"
                    ),
                    "point_count": 1,
                    "size": 16,
                    "sha256": _sha256_bytes(
                        f"prepared:{relative_path}".encode("utf-8")  # type: ignore[union-attr]
                    ),
                    "dtype": "<f4",
                    "fields": ["x", "y", "z", "intensity"],
                }
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_scope": "fixture",
        "delta_t_ms": 100,
        "history_limit": 1,
        "interval_min_ms": 50,
        "interval_max_ms": 150,
        "max_capture_skew_ms": 50,
        "split_sha256": FIXTURE_SPLIT_SHA256,
        "dataset_release_sha256": "",
        "release_inventory": [
            inventory_by_path[path] for path in sorted(inventory_by_path)
        ],
        "prepared_artifacts": [
            prepared_by_source[path] for path in sorted(prepared_by_source)
        ],
        "history_eligible_train_count": 1,
        "sequence_splits": [],
        "excluded_samples": [],
        "samples": samples,
        "content_sha256": "",
    }
    return _rehash(payload)


def _valid_split_payload() -> dict[str, object]:
    payload = _valid_payload()
    second_sample = payload["samples"][1]  # type: ignore[index]
    second_sample["sequence_id"] = "seq-1"
    second_sample["n_t"] = 0
    second_sample["tau_t_ms"] = 0
    second_tick = second_sample["source_slices"][-4:]
    second_sample["source_slices"] = second_tick
    for source_slice in second_tick:
        source_slice["n_s"] = 0
        source_slice["tau_s_ms"] = 0
        source_slice["capture_timestamp_us"] += 100_000
        source_slice["packet_id"] = (
            f"seq-1:0:{source_slice['agent']}:{source_slice['modality']}"
        )
    payload["history_eligible_train_count"] = 0
    payload["sequence_splits"] = [
        {
            "source_sequence_id": "official-batch-0",
            "previous_sample_id": "sample-0",
            "current_sample_id": "sample-1",
            "new_sequence_id": "seq-1",
            "triggers": [
                {
                    "agent": "ego",
                    "modality": "lidar",
                    "previous_capture_timestamp_us": 1_000_000,
                    "current_capture_timestamp_us": 1_200_000,
                    "interval_us": 200_000,
                }
            ],
        }
    ]
    return _rehash(payload)


def _write_payload(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    canonical: bool = True,
) -> Path:
    path = tmp_path / "manifest.json"
    data = (
        _canonical_bytes(payload) if canonical else json.dumps(payload).encode("utf-8")
    )
    path.write_bytes(data)
    return path


def _module():
    return importlib.import_module("transvision.dataset.resilient_v2x_manifest")


def test_manifest_import_is_stdlib_only_and_package_exports_are_lazy() -> None:
    code = r"""
import sys

import transvision.dataset
import transvision.dataset.resilient_v2x_manifest
from transvision.dataset import RawSliceRecord

forbidden = (
    "scipy",
    "numpy",
    "torch",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmdet3d",
    "transvision.dataset.transforms",
    "transvision.dataset.dair_v2x_for_detection",
    "transvision.models",
)
loaded = [
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
]
assert loaded == [], loaded
assert RawSliceRecord.__module__ == "transvision.dataset.resilient_v2x_manifest"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_module_and_package_public_exports_are_exact() -> None:
    module = _module()
    package = importlib.import_module("transvision.dataset")

    assert module.__all__ == MODULE_EXPORTS
    assert package.__all__ == PACKAGE_EXPORTS
    assert set(PACKAGE_EXPORTS).issubset(dir(package))
    for name in MODULE_EXPORTS:
        assert getattr(package, name) is getattr(module, name)
        assert getattr(package, name) is getattr(package, name)


def test_lazy_legacy_exports_are_cached_and_keep_exact_dataset_keys() -> None:
    code = r"""
import sys
import types

dair = types.ModuleType("transvision.dataset.dair_v2x_for_detection")
class DAIRV2XV: pass
class DAIRV2XI: pass
class VICSyncDataset: pass
class VICAsyncDataset: pass
class DAIRV2XVSPD: pass
class DAIRV2XISPD: pass
class VICSyncDatasetSPD: pass
class VICAsyncDatasetSPD: pass
for value in (
    DAIRV2XV, DAIRV2XI, VICSyncDataset, VICAsyncDataset,
    DAIRV2XVSPD, DAIRV2XISPD, VICSyncDatasetSPD, VICAsyncDatasetSPD,
):
    setattr(dair, value.__name__, value)
sys.modules[dair.__name__] = dair

transforms = types.ModuleType("transvision.dataset.transforms")
transforms.__path__ = []
sys.modules[transforms.__name__] = transforms
v2x_dataset = types.ModuleType("transvision.dataset.v2x_dataset")
sys.modules[v2x_dataset.__name__] = v2x_dataset
loading = types.ModuleType("transvision.dataset.transforms.loading")
class BEVLoadMultiViewImageFromFiles: pass
loading.BEVLoadMultiViewImageFromFiles = BEVLoadMultiViewImageFromFiles
sys.modules[loading.__name__] = loading

import transvision.dataset as dataset
from transvision.dataset import transforms as imported_transforms
from transvision.dataset import v2x_dataset as imported_v2x_dataset

assert imported_transforms is transforms
assert imported_v2x_dataset is v2x_dataset

first = dataset.SUPPROTED_DATASETS
second = dataset.SUPPROTED_DATASETS
assert first is second
assert tuple(first) == (
    "dair-v2x-v", "dair-v2x-i", "vic-sync", "vic-async",
    "dair-v2x-v-spd", "dair-v2x-i-spd", "vic-sync-spd", "vic-async-spd",
)
first["sentinel"] = object()
assert dataset.SUPPROTED_DATASETS["sentinel"] is first["sentinel"]
assert dataset.BEVLoadMultiViewImageFromFiles is BEVLoadMultiViewImageFromFiles
assert dataset.BEVLoadMultiViewImageFromFiles is dataset.BEVLoadMultiViewImageFromFiles
try:
    dataset.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError("unknown package attribute was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_canonical_json_is_sorted_compact_utf8_and_rejects_nonfinite() -> None:
    module = _module()
    value = {"中文": "值", "b": 2, "a": 1}

    assert module.canonical_json_bytes(value) == (
        '{"a":1,"b":2,"中文":"值"}'.encode("utf-8")
    )
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            module.canonical_json_bytes({"value": invalid})


def test_content_hash_removes_only_top_level_hash_without_mutation() -> None:
    module = _module()
    value = {
        "schema_version": 1,
        "nested": {"content_sha256": "preserved"},
        "content_sha256": "ignored",
    }
    original = copy.deepcopy(value)
    expected = _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": 1,
                "nested": {"content_sha256": "preserved"},
            }
        )
    )

    assert module.content_sha256(value) == expected
    assert value == original
    with pytest.raises(TypeError):
        module.content_sha256(["not", "a", "mapping"])


def test_release_inventory_hash_requires_exact_sorted_entries() -> None:
    module = _module()
    entries = [
        {"relative_path": "a.bin", "size": 3, "sha256": "0" * 64},
        {"relative_path": "b.jpg", "size": 4, "sha256": "1" * 64},
    ]
    expected = _sha256_bytes(_canonical_bytes(entries))

    assert module.release_inventory_sha256(entries) == expected
    with pytest.raises(module.ManifestError, match="sorted"):
        module.release_inventory_sha256(list(reversed(entries)))
    with pytest.raises(module.ManifestError, match="duplicate"):
        module.release_inventory_sha256([entries[0], entries[0]])
    with pytest.raises(module.ManifestError, match="fields"):
        module.release_inventory_sha256([{**entries[0], "extra": True}])


def test_checked_in_official_split_matches_public_constant() -> None:
    module = _module()
    split_path = ROOT / "data/split_datas/cooperative-split-data.json"

    assert _sha256_bytes(split_path.read_bytes()) == (
        module.OFFICIAL_COOPERATIVE_SPLIT_SHA256
    )


def test_valid_fixture_manifest_loads_as_deeply_immutable_dataclasses(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    path = _write_payload(tmp_path, payload)

    manifest = module.load_temporal_manifest(
        path,
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )

    assert isinstance(manifest, module.TemporalManifest)
    assert manifest.content_sha256 == payload["content_sha256"]
    assert isinstance(manifest.samples, tuple)
    assert isinstance(manifest.samples[0].source_slices, tuple)
    assert isinstance(manifest.samples[0].ground_truth, tuple)
    assert isinstance(manifest.samples[0].source_slices[0].world_from_agent, tuple)
    assert isinstance(manifest.release_inventory, tuple)
    assert manifest.excluded_samples == ()
    with pytest.raises(FrozenInstanceError):
        manifest.history_limit = 3


@pytest.mark.parametrize(
    "raw_bytes",
    (
        b'{"schema_version":1,"schema_version":1}',
        b"\xef\xbb\xbf{}",
        b"{}\n",
        b'{ "schema_version": 1 }',
    ),
)
def test_loader_rejects_duplicate_or_noncanonical_json(
    tmp_path: Path,
    raw_bytes: bytes,
) -> None:
    module = _module()
    path = tmp_path / "manifest.json"
    path.write_bytes(raw_bytes)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_loader_rejects_missing_and_extra_nested_fields(tmp_path: Path) -> None:
    module = _module()
    for mutation in ("missing", "extra"):
        payload = _valid_payload()
        first_slice = payload["samples"][0]["source_slices"][0]  # type: ignore[index]
        if mutation == "missing":
            first_slice.pop("frame_id")
        else:
            first_slice["unexpected"] = True
        _rehash(payload)
        path = _write_payload(tmp_path, payload)

        with pytest.raises(module.ManifestError, match="fields"):
            module.load_temporal_manifest(
                path,
                expected_split_hash=FIXTURE_SPLIT_SHA256,
                allow_fixture=True,
            )


def test_fixture_and_controlled_scope_gates_are_fail_closed(tmp_path: Path) -> None:
    module = _module()
    fixture = _valid_payload()
    fixture_path = _write_payload(tmp_path, fixture)
    with pytest.raises(module.ManifestError, match="fixture"):
        module.load_temporal_manifest(
            fixture_path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
        )

    controlled = _valid_payload()
    controlled["protocol_scope"] = "controlled"
    controlled["history_limit"] = 3
    controlled["max_capture_skew_ms"] = 200
    controlled["split_sha256"] = module.OFFICIAL_COOPERATIVE_SPLIT_SHA256
    controlled["history_eligible_train_count"] = 0
    _rehash(controlled)
    controlled_path = _write_payload(tmp_path, controlled)
    loaded = module.load_temporal_manifest(
        controlled_path,
        expected_split_hash=module.OFFICIAL_COOPERATIVE_SPLIT_SHA256,
    )
    assert loaded.protocol_scope == "controlled"

    controlled_test = copy.deepcopy(controlled)
    for sample in controlled_test["samples"]:
        sample["split"] = "test"
    _rehash(controlled_test)
    controlled_test_path = _write_payload(tmp_path, controlled_test)
    with pytest.raises(module.ManifestError, match="train and val"):
        module.load_temporal_manifest(
            controlled_test_path,
            expected_split_hash=module.OFFICIAL_COOPERATIVE_SPLIT_SHA256,
        )

    controlled["delta_t_ms"] = 101
    _rehash(controlled)
    invalid_path = _write_payload(tmp_path, controlled)
    with pytest.raises(module.ManifestError, match="controlled"):
        module.load_temporal_manifest(
            invalid_path,
            expected_split_hash=module.OFFICIAL_COOPERATIVE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_loader_accepts_aggregate_inventory_entries_not_linked_by_slices(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    path = _write_payload(tmp_path, payload)

    manifest = module.load_temporal_manifest(
        path,
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )

    inventory_paths = {entry.relative_path for entry in manifest.release_inventory}
    assert "calib/secondary-camera-intrinsic.json" in inventory_paths
    assert "poses/secondary-world-pose.json" in inventory_paths
    assert "metadata/cooperative-data-info.json" in inventory_paths


def test_loader_rejects_semantically_equivalent_noncanonical_full_payload(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    payload["samples"][0]["sample_id"] = "样本-0"  # type: ignore[index]
    _rehash(payload)
    canonical = _canonical_bytes(payload)
    variants = (
        canonical + b"\n",
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8"),
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )
    assert all(variant != canonical for variant in variants)

    for index, raw in enumerate(variants):
        path = tmp_path / f"noncanonical-{index}.json"
        path.write_bytes(raw)
        with pytest.raises(module.ManifestError, match="canonical"):
            module.load_temporal_manifest(
                path,
                expected_split_hash=FIXTURE_SPLIT_SHA256,
                allow_fixture=True,
            )


@pytest.mark.parametrize("token", ("NaN", "Infinity", "-Infinity", "1e400"))
def test_loader_rejects_nonfinite_json_tokens_in_complete_payload(
    tmp_path: Path,
    token: str,
) -> None:
    module = _module()
    raw = _canonical_bytes(_valid_payload())
    raw = raw.replace(b'"x":10.0', f'"x":{token}'.encode("ascii"), 1)
    path = tmp_path / "nonfinite.json"
    path.write_bytes(raw)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_loader_rejects_nested_duplicate_key_and_invalid_utf8(
    tmp_path: Path,
) -> None:
    module = _module()
    raw = _canonical_bytes(_valid_payload())
    marker = b'"frame_id":"ego-lidar-0"'
    duplicated = raw.replace(marker, marker + b"," + marker, 1)
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_bytes(duplicated)
    with pytest.raises(module.ManifestError, match="duplicate"):
        module.load_temporal_manifest(
            duplicate_path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )

    invalid_utf8_path = tmp_path / "invalid-utf8.json"
    invalid_utf8_path.write_bytes(b'{"value":"\xff"}')
    with pytest.raises(module.ManifestError, match="UTF-8"):
        module.load_temporal_manifest(
            invalid_utf8_path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_direct_slice_construction_copies_nested_matrices() -> None:
    module = _module()
    source = _valid_payload()["samples"][0]["source_slices"][0]  # type: ignore[index]
    world_matrix = copy.deepcopy(source["world_from_agent"])
    sensor_matrix = copy.deepcopy(source["agent_from_sensor"])
    direct = module.RawSliceRecord(
        **{
            **source,
            "world_from_agent": world_matrix,
            "agent_from_sensor": sensor_matrix,
        }
    )

    world_matrix[0][0] = 99.0
    sensor_matrix[1][1] = 99.0
    assert direct.world_from_agent == IDENTITY_4X4
    assert direct.agent_from_sensor == IDENTITY_4X4


def test_sequence_split_mappings_are_deeply_copied_and_frozen(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_split_payload()
    loaded = module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )
    original_boundary = copy.deepcopy(payload["sequence_splits"][0])  # type: ignore[index]
    direct = module.TemporalManifest(
        schema_version=loaded.schema_version,
        protocol_scope=loaded.protocol_scope,
        delta_t_ms=loaded.delta_t_ms,
        history_limit=loaded.history_limit,
        interval_min_ms=loaded.interval_min_ms,
        interval_max_ms=loaded.interval_max_ms,
        max_capture_skew_ms=loaded.max_capture_skew_ms,
        split_sha256=loaded.split_sha256,
        dataset_release_sha256=loaded.dataset_release_sha256,
        release_inventory=loaded.release_inventory,
        prepared_artifacts=loaded.prepared_artifacts,
        history_eligible_train_count=loaded.history_eligible_train_count,
        sequence_splits=(original_boundary,),
        excluded_samples=(),
        samples=loaded.samples,
        content_sha256=loaded.content_sha256,
    )

    original_boundary["current_sample_id"] = "mutated"
    original_boundary["triggers"][0]["interval_us"] = 999_999
    assert direct.sequence_splits[0]["current_sample_id"] == "sample-1"
    assert direct.sequence_splits[0]["triggers"][0]["interval_us"] == 200_000
    assert isinstance(direct.sequence_splits[0], MappingProxyType)
    assert isinstance(direct.sequence_splits[0]["triggers"], tuple)
    assert isinstance(
        direct.sequence_splits[0]["triggers"][0],
        MappingProxyType,
    )
    with pytest.raises(TypeError):
        direct.sequence_splits[0]["current_sample_id"] = "forbidden"


@pytest.mark.parametrize(
    "relative_path",
    (
        "",
        "/absolute.bin",
        "a//b.bin",
        "a/./b.bin",
        "a/../b.bin",
        "a\\b.bin",
        "C:/windows.bin",
        "nul\x00byte.bin",
    ),
)
def test_build_release_inventory_rejects_noncanonical_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    module = _module()
    with pytest.raises(module.ManifestError):
        module.build_release_inventory(tmp_path, [relative_path])


def test_build_release_inventory_hashes_regular_files_in_sorted_order(
    tmp_path: Path,
) -> None:
    module = _module()
    (tmp_path / "b.bin").write_bytes(b"bbbb")
    (tmp_path / "a.bin").write_bytes(b"aaa")

    inventory = module.build_release_inventory(
        tmp_path,
        (path for path in ("b.bin", "a.bin")),
    )

    assert tuple(entry.relative_path for entry in inventory) == ("a.bin", "b.bin")
    assert inventory[0].size == 3
    assert inventory[0].sha256 == _sha256_bytes(b"aaa")
    assert inventory[1].size == 4
    assert inventory[1].sha256 == _sha256_bytes(b"bbbb")


def test_build_release_inventory_rejects_symlinks_and_nonregular_files(
    tmp_path: Path,
) -> None:
    module = _module()
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    (tmp_path / "final-link.bin").symlink_to(target)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "data.bin").write_bytes(b"nested")
    (tmp_path / "nested-link").symlink_to(nested, target_is_directory=True)
    (tmp_path / "directory").mkdir()

    for relative_path in (
        "final-link.bin",
        "nested-link/data.bin",
        "directory",
    ):
        with pytest.raises(module.ManifestError):
            module.build_release_inventory(tmp_path, [relative_path])

    root_link = tmp_path.parent / f"{tmp_path.name}-link"
    root_link.symlink_to(tmp_path, target_is_directory=True)
    try:
        with pytest.raises(module.ManifestError, match="root"):
            module.build_release_inventory(root_link, ["target.bin"])
    finally:
        root_link.unlink()

    real_parent = tmp_path / "real-parent"
    nested_root = real_parent / "root"
    nested_root.mkdir(parents=True)
    (nested_root / "data.bin").write_bytes(b"data")
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(module.ManifestError, match="root"):
        module.build_release_inventory(parent_link / "root", ["data.bin"])


def test_build_release_inventory_rejects_file_changed_while_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    data_path = tmp_path / "data.bin"
    data_path.write_bytes(b"stable")
    real_fstat = os.fstat
    regular_calls = 0

    def drifting_fstat(descriptor: int):
        nonlocal regular_calls
        result = real_fstat(descriptor)
        if not os.path.isfile(data_path):
            return result
        regular_calls += 1
        if regular_calls == 1:
            return result
        return types.SimpleNamespace(
            st_mode=result.st_mode,
            st_dev=result.st_dev,
            st_ino=result.st_ino,
            st_size=result.st_size,
            st_mtime_ns=result.st_mtime_ns + 1,
            st_ctime_ns=result.st_ctime_ns,
        )

    monkeypatch.setattr(module.os, "fstat", drifting_fstat)
    with pytest.raises(module.ManifestError, match="changed"):
        module.build_release_inventory(tmp_path, ["data.bin"])


def test_build_release_inventory_rejects_root_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = tmp_path / "root"
    root.mkdir()
    (root / "data.bin").write_bytes(b"trusted")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "data.bin").write_bytes(b"substituted")
    displaced = tmp_path / "displaced-root"
    real_lstat = Path.lstat
    real_open = os.open
    replaced = False

    def replace_root() -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        root.rename(displaced)
        replacement.rename(root)

    def racing_lstat(path: Path):
        if path == root / "data.bin":
            replace_root()
        return real_lstat(path)

    def racing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) == "data.bin":
            replace_root()
        if dir_fd is None:
            return real_open(path, flags, mode)  # type: ignore[arg-type]
        return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "lstat", racing_lstat)
    monkeypatch.setattr(module.os, "open", racing_open)

    with pytest.raises(module.ManifestError, match="changed"):
        module.build_release_inventory(root, ["data.bin"])
    assert replaced


def test_build_release_inventory_rejects_inner_directory_symlink_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = tmp_path / "root"
    inner = root / "inner"
    inner.mkdir(parents=True)
    (inner / "data.bin").write_bytes(b"trusted")
    substitute = root / "substitute"
    substitute.mkdir()
    (substitute / "data.bin").write_bytes(b"substituted")
    displaced = root / "displaced-inner"
    real_lstat = Path.lstat
    real_open = os.open
    replaced = False

    def replace_inner() -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        inner.rename(displaced)
        inner.symlink_to(substitute, target_is_directory=True)

    def racing_lstat(path: Path):
        if path == inner / "data.bin":
            replace_inner()
        return real_lstat(path)

    def racing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) == "data.bin":
            replace_inner()
        if dir_fd is None:
            return real_open(path, flags, mode)  # type: ignore[arg-type]
        return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "lstat", racing_lstat)
    monkeypatch.setattr(module.os, "open", racing_open)

    with pytest.raises(module.ManifestError, match="changed"):
        module.build_release_inventory(root, ["inner/data.bin"])
    assert replaced


@pytest.mark.parametrize("interval_us", (50_000, 150_000))
def test_temporal_interval_inclusive_boundaries_pass(
    tmp_path: Path,
    interval_us: int,
) -> None:
    module = _module()
    payload = _valid_payload()
    current_tick = payload["samples"][1]["source_slices"][-4:]  # type: ignore[index]
    previous_tick = payload["samples"][0]["source_slices"]  # type: ignore[index]
    for previous, current in zip(previous_tick, current_tick):
        current["capture_timestamp_us"] = previous["capture_timestamp_us"] + interval_us
    _rehash(payload)

    loaded = module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )
    assert loaded.samples[1].n_t == 1


@pytest.mark.parametrize("interval_us", (49_999, 150_001))
def test_temporal_interval_outside_bounds_fails_inside_one_sequence(
    tmp_path: Path,
    interval_us: int,
) -> None:
    module = _module()
    payload = _valid_payload()
    current_tick = payload["samples"][1]["source_slices"][-4:]  # type: ignore[index]
    previous_tick = payload["samples"][0]["source_slices"]  # type: ignore[index]
    for previous, current in zip(previous_tick, current_tick):
        current["capture_timestamp_us"] = previous["capture_timestamp_us"] + interval_us
    _rehash(payload)

    with pytest.raises(module.ManifestError, match="interval"):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


@pytest.mark.parametrize(
    ("skew_us", "passes"),
    ((50_000, True), (50_001, False)),
)
def test_capture_skew_exact_boundary(
    tmp_path: Path,
    skew_us: int,
    passes: bool,
) -> None:
    module = _module()
    payload = _valid_payload()
    for sample in payload["samples"]:  # type: ignore[union-attr]
        for tick_start in range(0, len(sample["source_slices"]), 4):
            tick = sample["source_slices"][tick_start : tick_start + 4]
            tick[-1]["capture_timestamp_us"] = tick[0]["capture_timestamp_us"] + skew_us
    _rehash(payload)
    path = _write_payload(tmp_path, payload)

    if passes:
        assert module.load_temporal_manifest(
            path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        ).samples
    else:
        with pytest.raises(module.ManifestError, match="skew"):
            module.load_temporal_manifest(
                path,
                expected_split_hash=FIXTURE_SPLIT_SHA256,
                allow_fixture=True,
            )


def test_cross_target_packet_and_position_identity_are_bidirectional(
    tmp_path: Path,
) -> None:
    module = _module()
    for mutation in ("provenance", "packet_id"):
        payload = _valid_payload()
        reused = payload["samples"][1]["source_slices"][0]  # type: ignore[index]
        if mutation == "provenance":
            reused["frame_id"] = "changed-frame"
        else:
            reused["packet_id"] = "different-packet"
        _rehash(payload)
        with pytest.raises(module.ManifestError, match="provenance|packet"):
            module.load_temporal_manifest(
                _write_payload(tmp_path, payload),
                expected_split_hash=FIXTURE_SPLIT_SHA256,
                allow_fixture=True,
            )


def test_source_history_grid_order_and_split_are_strict(tmp_path: Path) -> None:
    module = _module()
    mutations = ("missing", "swapped", "wrong_split")
    for mutation in mutations:
        payload = _valid_payload()
        if mutation == "missing":
            payload["samples"][1]["source_slices"].pop(0)  # type: ignore[index]
        elif mutation == "swapped":
            slices = payload["samples"][1]["source_slices"]  # type: ignore[index]
            slices[0], slices[1] = slices[1], slices[0]
        else:
            payload["samples"][1]["split"] = "val"  # type: ignore[index]
        _rehash(payload)
        with pytest.raises(module.ManifestError, match="history|grid|split"):
            module.load_temporal_manifest(
                _write_payload(tmp_path, payload),
                expected_split_hash=FIXTURE_SPLIT_SHA256,
                allow_fixture=True,
            )


def test_valid_sequence_split_loads_and_fake_trigger_provenance_fails(
    tmp_path: Path,
) -> None:
    module = _module()
    valid = _valid_split_payload()
    loaded = module.load_temporal_manifest(
        _write_payload(tmp_path, valid),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )
    assert loaded.sequence_splits[0]["new_sequence_id"] == "seq-1"

    invalid = _valid_split_payload()
    trigger = invalid["sequence_splits"][0]["triggers"][0]  # type: ignore[index]
    trigger["previous_capture_timestamp_us"] += 1
    trigger["current_capture_timestamp_us"] += 1
    _rehash(invalid)
    with pytest.raises(module.ManifestError, match="provenance"):
        module.load_temporal_manifest(
            _write_payload(tmp_path, invalid),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_aggregate_inventory_hash_propagation_and_self_hash_limit(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    secondary = next(
        entry
        for entry in payload["release_inventory"]
        if entry["relative_path"] == "poses/secondary-world-pose.json"
    )
    secondary["sha256"] = "c" * 64
    path = _write_payload(tmp_path, payload)
    with pytest.raises(module.ManifestError, match="inventory"):
        module.load_temporal_manifest(
            path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )

    payload["dataset_release_sha256"] = _inventory_sha256(
        payload["release_inventory"]  # type: ignore[arg-type]
    )
    path = _write_payload(tmp_path, payload)
    with pytest.raises(module.ManifestError, match="content"):
        module.load_temporal_manifest(
            path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )

    _rehash(payload)
    loaded = module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )
    assert loaded.dataset_release_sha256 == payload["dataset_release_sha256"]


def test_canonical_json_rejects_nonstring_mapping_keys() -> None:
    module = _module()
    with pytest.raises(TypeError):
        module.canonical_json_bytes({1: "integer-key"})
    with pytest.raises(TypeError):
        module.content_sha256({"nested": {1: "integer-key"}})


def test_loader_wraps_lone_unicode_surrogate_as_manifest_error(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "surrogate.json"
    path.write_bytes(b'{"value":"\\ud800"}')

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_windows_drive_spelling_is_not_a_posix_relative_path() -> None:
    module = _module()
    with pytest.raises(module.ManifestError, match="POSIX"):
        module.ReleaseInventoryEntry(
            relative_path="C:/dataset/file.bin",
            size=1,
            sha256="0" * 64,
        )


def test_direct_manifest_construction_validates_content_self_hash(
    tmp_path: Path,
) -> None:
    module = _module()
    loaded = module.load_temporal_manifest(
        _write_payload(tmp_path, _valid_payload()),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )
    values = {field: getattr(loaded, field) for field in loaded.__dataclass_fields__}
    values["content_sha256"] = "0" * 64

    with pytest.raises(module.ManifestError, match="content"):
        module.TemporalManifest(**values)


def test_sequence_split_previous_and_current_samples_must_be_adjacent(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_split_payload()
    middle = copy.deepcopy(payload["samples"][0])  # type: ignore[index]
    middle["sample_id"] = "sample-middle"
    middle["sequence_id"] = "seq-middle"
    for source_slice in middle["source_slices"]:
        source_slice["packet_id"] = (
            f"seq-middle:0:{source_slice['agent']}:{source_slice['modality']}"
        )
    payload["samples"].insert(1, middle)  # type: ignore[union-attr]
    _rehash(payload)

    with pytest.raises(module.ManifestError, match="adjacent"):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_integer_json_numbers_remain_compatible_with_content_self_hash(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    ground_truth = payload["samples"][0]["ground_truth"][0]  # type: ignore[index]
    ground_truth["x"] = 10
    source = payload["samples"][0]["source_slices"][0]  # type: ignore[index]
    source["world_from_agent"][0][0] = 1
    _rehash(payload)

    manifest = module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )
    assert manifest.samples[0].ground_truth[0].x == 10


@pytest.mark.parametrize(
    "mutation",
    (
        "top_int_bool",
        "slice_int_bool",
        "slice_bool_int",
        "gt_number_bool",
        "inventory_int_bool",
        "prepared_int_bool",
        "trigger_int_bool",
    ),
)
def test_exact_integer_boolean_and_numeric_types(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    payload = (
        _valid_split_payload() if mutation == "trigger_int_bool" else _valid_payload()
    )
    if mutation == "top_int_bool":
        payload["history_limit"] = True
    elif mutation == "slice_int_bool":
        payload["samples"][0]["source_slices"][0]["n_s"] = False  # type: ignore[index]
    elif mutation == "slice_bool_int":
        payload["samples"][0]["source_slices"][0]["payload_valid"] = 1  # type: ignore[index]
    elif mutation == "gt_number_bool":
        payload["samples"][0]["ground_truth"][0]["x"] = True  # type: ignore[index]
    elif mutation == "inventory_int_bool":
        payload["release_inventory"][0]["size"] = True  # type: ignore[index]
    elif mutation == "prepared_int_bool":
        payload["prepared_artifacts"][0]["point_count"] = True  # type: ignore[index]
    else:
        payload["sequence_splits"][0]["triggers"][0]["interval_us"] = True  # type: ignore[index]
    _rehash(payload)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "reflection",
        "scale",
        "shear",
        "singular",
        "ragged",
        "matrix_bool",
        "last_row",
        "negative_focal",
        "singular_intrinsic",
        "lidar_intrinsic",
        "camera_missing_intrinsic",
    ),
)
def test_transform_and_intrinsic_rejection_matrix(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    payload = _valid_payload()
    lidar = payload["samples"][0]["source_slices"][0]  # type: ignore[index]
    camera = payload["samples"][0]["source_slices"][2]  # type: ignore[index]
    if mutation == "reflection":
        lidar["world_from_agent"][0][0] = -1.0
    elif mutation == "scale":
        lidar["world_from_agent"][0][0] = 2.0
    elif mutation == "shear":
        lidar["world_from_agent"][0][1] = 0.1
    elif mutation == "singular":
        lidar["world_from_agent"][0][0] = 0.0
    elif mutation == "ragged":
        lidar["world_from_agent"][0].pop()
    elif mutation == "matrix_bool":
        lidar["world_from_agent"][0][0] = True
    elif mutation == "last_row":
        lidar["world_from_agent"][3][0] = 1.1e-5
    elif mutation == "negative_focal":
        camera["camera_intrinsic"][0][0] = 0.0
    elif mutation == "singular_intrinsic":
        camera["camera_intrinsic"][2] = [0.0, 0.0, 0.0]
    elif mutation == "lidar_intrinsic":
        lidar["camera_intrinsic"] = [list(row) for row in CAMERA_INTRINSIC]
    else:
        camera["camera_intrinsic"] = None
    _rehash(payload)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_rigid_matrix_tolerance_and_proper_rotation_pass(tmp_path: Path) -> None:
    module = _module()
    payload = _valid_payload()
    angle = 0.25
    rotation = [
        [math.cos(angle), -math.sin(angle), 0.0, 1.0],
        [math.sin(angle), math.cos(angle), 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [1e-5, 0.0, 0.0, 1.0],
    ]
    for sample in payload["samples"]:  # type: ignore[union-attr]
        for source in sample["source_slices"]:
            if source["packet_id"] == "seq-0:0:ego:lidar":
                source["world_from_agent"] = copy.deepcopy(rotation)
            if source["modality"] == "camera":
                source["camera_intrinsic"] = [
                    [1e-7, 0.0, 0.0],
                    [0.0, 1e-7, 0.0],
                    [0.0, 0.0, 1.0],
                ]
    _rehash(payload)

    assert module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    ).samples


def test_camera_agent_from_sensor_accepts_well_conditioned_proper_affine(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    affine = [
        [1.0, 0.1, 0.0, 1.0],
        [0.0, 0.82, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for sample in payload["samples"]:
        for source in sample["source_slices"]:
            if source["modality"] == "camera":
                source["agent_from_sensor"] = copy.deepcopy(affine)
    _rehash(payload)

    loaded = module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )

    assert loaded.samples[0].source_slices[2].agent_from_sensor == tuple(
        tuple(row) for row in affine
    )


@pytest.mark.parametrize(
    ("modality", "field"),
    (
        ("camera", "world_from_agent"),
        ("lidar", "agent_from_sensor"),
    ),
)
def test_affine_allowance_is_limited_to_camera_sensor_extrinsics(
    modality: str,
    field: str,
) -> None:
    module = _module()
    raw_slice = _slice(
        sequence_id="seq-0",
        n_s=0,
        agent="ego",
        modality=modality,
        capture_timestamp_us=1_000_000,
    )
    raw_slice[field] = [
        [1.0, 0.1, 0.0, 1.0],
        [0.0, 0.82, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    with pytest.raises(
        module.ManifestError,
        match=rf"raw slice {field} rotation must be orthonormal",
    ):
        module.RawSliceRecord(**raw_slice)


@pytest.mark.parametrize(
    ("linear", "translation"),
    (
        (
            ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.0),
        ),
        (
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.0),
        ),
        (
            ((1e8, 0.0, 0.0), (0.0, 1e-8, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.0),
        ),
        (
            ((1e39, 0.0, 0.0), (0.0, 1e39, 0.0), (0.0, 0.0, 1e39)),
            (0.0, 0.0, 0.0),
        ),
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (1e39, 0.0, 0.0),
        ),
    ),
)
def test_camera_agent_from_sensor_rejects_unsafe_affine(
    tmp_path: Path,
    linear: tuple[tuple[float, ...], ...],
    translation: tuple[float, float, float],
) -> None:
    module = _module()
    payload = _valid_payload()
    camera = payload["samples"][0]["source_slices"][2]
    camera["agent_from_sensor"] = [
        [*linear[0], translation[0]],
        [*linear[1], translation[1]],
        [*linear[2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    _rehash(payload)

    with pytest.raises(module.ManifestError, match="proper|conditioned|float32"):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_camera_intrinsic_accepts_tiny_exact_nonsingular_diagonal(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    intrinsic = [
        [1e-200, 0.0, 0.0],
        [0.0, 1e-200, 0.0],
        [0.0, 0.0, 1.0],
    ]
    for sample in payload["samples"]:  # type: ignore[union-attr]
        for source in sample["source_slices"]:
            if source["modality"] == "camera":
                source["camera_intrinsic"] = copy.deepcopy(intrinsic)
    _rehash(payload)

    loaded = module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )

    assert loaded.samples[0].source_slices[2].camera_intrinsic == tuple(
        tuple(row) for row in intrinsic
    )


def test_camera_intrinsic_rejects_duplicate_huge_rows_without_nan_escape(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    singular = [
        [1e308, 1e308, 0.0],
        [1e308, 1e308, 0.0],
        [0.0, 0.0, 1.0],
    ]
    for sample in payload["samples"]:  # type: ignore[union-attr]
        for source in sample["source_slices"]:
            if source["modality"] == "camera":
                source["camera_intrinsic"] = copy.deepcopy(singular)
    _rehash(payload)

    with pytest.raises(module.ManifestError, match="nonsingular"):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


@pytest.mark.parametrize(
    ("mutation", "passes"),
    (
        ("yaw_negative_pi", True),
        ("yaw_pi", False),
        ("zero_dimension", False),
        ("wrong_class", False),
        ("negative_index", False),
        ("duplicate_index", False),
        ("descending_index", False),
    ),
)
def test_ground_truth_boundaries_and_source_index_order(
    tmp_path: Path,
    mutation: str,
    passes: bool,
) -> None:
    module = _module()
    payload = _valid_payload()
    boxes = payload["samples"][0]["ground_truth"]  # type: ignore[index]
    if mutation == "yaw_negative_pi":
        boxes[0]["yaw"] = -math.pi
    elif mutation == "yaw_pi":
        boxes[0]["yaw"] = math.pi
    elif mutation == "zero_dimension":
        boxes[0]["length"] = 0.0
    elif mutation == "wrong_class":
        boxes[0]["class_name"] = "Truck"
    elif mutation == "negative_index":
        boxes[0]["source_annotation_index"] = -1
    else:
        second = copy.deepcopy(boxes[0])
        if mutation == "duplicate_index":
            second["source_annotation_index"] = 0
        else:
            boxes[0]["source_annotation_index"] = 1
            second["source_annotation_index"] = 0
        boxes.append(second)
    _rehash(payload)
    path = _write_payload(tmp_path, payload)

    if passes:
        assert module.load_temporal_manifest(
            path,
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        ).samples
    else:
        with pytest.raises(module.ManifestError):
            module.load_temporal_manifest(
                path,
                expected_split_hash=FIXTURE_SPLIT_SHA256,
                allow_fixture=True,
            )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "duplicate_source",
        "duplicate_destination",
        "destination_in_raw",
        "bad_size",
        "bad_dtype",
        "bad_fields",
        "bad_hash",
        "camera_source",
    ),
)
def test_prepared_artifact_exact_coverage_and_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    payload = _valid_payload()
    prepared = payload["prepared_artifacts"]  # type: ignore[assignment]
    if mutation == "missing":
        prepared.pop()
    elif mutation == "extra":
        extra = copy.deepcopy(prepared[-1])
        extra["source_relative_path"] = "raw/unreferenced.pcd"
        extra["prepared_relative_path"] = "prepared/unreferenced.bin"
        prepared.append(extra)
    elif mutation == "duplicate_source":
        duplicate = copy.deepcopy(prepared[0])
        duplicate["prepared_relative_path"] = "prepared/other.bin"
        prepared.insert(1, duplicate)
    elif mutation == "duplicate_destination":
        prepared[1]["prepared_relative_path"] = prepared[0]["prepared_relative_path"]
    elif mutation == "destination_in_raw":
        prepared[0]["prepared_relative_path"] = payload["release_inventory"][0][
            "relative_path"
        ]
    elif mutation == "bad_size":
        prepared[0]["size"] = 15
    elif mutation == "bad_dtype":
        prepared[0]["dtype"] = ">f4"
    elif mutation == "bad_fields":
        prepared[0]["fields"] = ["x", "y", "z", "reflectance"]
    elif mutation == "bad_hash":
        prepared[0]["sha256"] = "A" * 64
    else:
        camera = payload["samples"][0]["source_slices"][2]  # type: ignore[index]
        prepared[0]["source_relative_path"] = camera["relative_path"]
    prepared.sort(key=lambda item: item["source_relative_path"])
    _rehash(payload)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_loader_accepts_reversed_prepared_artifact_order(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    payload["prepared_artifacts"].reverse()  # type: ignore[union-attr]
    _rehash(payload)

    manifest = module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )

    assert tuple(
        item.source_relative_path for item in manifest.prepared_artifacts
    ) == tuple(
        item["source_relative_path"]
        for item in payload["prepared_artifacts"]  # type: ignore[union-attr]
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "excluded",
        "duplicate_sample",
        "sequence_reappears",
        "target_tick_gap",
        "duplicate_packet",
    ),
)
def test_manifest_identity_sequence_and_exclusion_invariants(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    payload = _valid_payload()
    if mutation == "excluded":
        payload["excluded_samples"] = [
            {
                "sample_id": "excluded",
                "sequence_id": "seq-x",
                "split": "train",
                "n_t": 0,
                "reason": "no_supported_branch",
            }
        ]
    elif mutation == "duplicate_sample":
        payload["samples"][1]["sample_id"] = "sample-0"  # type: ignore[index]
    elif mutation == "sequence_reappears":
        middle = copy.deepcopy(payload["samples"][1])  # type: ignore[index]
        middle["sample_id"] = "sample-middle"
        middle["sequence_id"] = "seq-middle"
        middle["n_t"] = 0
        middle["tau_t_ms"] = 0
        middle["source_slices"] = middle["source_slices"][-4:]
        for source in middle["source_slices"]:
            source["n_s"] = 0
            source["tau_s_ms"] = 0
            source["packet_id"] = f"seq-middle:0:{source['agent']}:{source['modality']}"
        payload["samples"].insert(1, middle)  # type: ignore[union-attr]
    elif mutation == "target_tick_gap":
        payload["samples"][1]["n_t"] = 2  # type: ignore[index]
        payload["samples"][1]["tau_t_ms"] = 200  # type: ignore[index]
    else:
        slices = payload["samples"][0]["source_slices"]  # type: ignore[index]
        slices[1]["packet_id"] = slices[0]["packet_id"]
    _rehash(payload)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


@pytest.mark.parametrize(
    "field",
    (
        "agent",
        "modality",
        "n_s",
        "tau_s_ms",
        "packet_id",
        "frame_id",
        "relative_path",
        "capture_timestamp_us",
        "world_from_agent",
        "agent_from_sensor",
        "calibration",
        "camera_intrinsic",
        "payload_valid",
        "pose_valid",
        "calibration_valid",
    ),
)
def test_cross_target_reuse_rejects_every_slice_provenance_mutation(
    tmp_path: Path,
    field: str,
) -> None:
    module = _module()
    payload = _valid_payload()
    reused_index = 2 if field == "camera_intrinsic" else 0
    reused = payload["samples"][1]["source_slices"][reused_index]  # type: ignore[index]
    if field == "agent":
        reused["agent"] = "rsu"
    elif field == "modality":
        reused["modality"] = "camera"
    elif field == "n_s":
        reused["n_s"] = 1
    elif field == "tau_s_ms":
        reused["tau_s_ms"] = 1
    elif field == "packet_id":
        reused["packet_id"] = "different-packet"
    elif field == "frame_id":
        reused["frame_id"] = "other-frame"
    elif field == "relative_path":
        reused["relative_path"] = payload["samples"][1]["source_slices"][4][
            "relative_path"
        ]
    elif field == "capture_timestamp_us":
        reused["capture_timestamp_us"] += 1
    elif field in ("world_from_agent", "agent_from_sensor"):
        reused[field][0][3] = 0.01
    elif field == "calibration":
        aggregate = next(
            entry
            for entry in payload["release_inventory"]
            if entry["relative_path"] == "calib/secondary-camera-intrinsic.json"
        )
        reused["calibration_relative_path"] = aggregate["relative_path"]
        reused["calibration_sha256"] = aggregate["sha256"]
    elif field == "camera_intrinsic":
        reused["camera_intrinsic"][0][2] += 1.0
    else:
        reused[field] = False
    _rehash(payload)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_direct_manifest_rejects_malformed_sequence_split_mapping(
    tmp_path: Path,
) -> None:
    module = _module()
    loaded = module.load_temporal_manifest(
        _write_payload(tmp_path, _valid_split_payload()),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )
    malformed = dict(loaded.sequence_splits[0])
    malformed["extra"] = True
    values = {field: getattr(loaded, field) for field in loaded.__dataclass_fields__}
    values["sequence_splits"] = (malformed,)

    with pytest.raises(module.ManifestError, match="fields"):
        module.TemporalManifest(**values)


def test_direct_nested_sequences_are_defensively_copied() -> None:
    module = _module()
    payload = _valid_payload()
    raw = module.RawSliceRecord(**payload["samples"][0]["source_slices"][0])  # type: ignore[index]
    gt = module.GroundTruthBoxRecord(
        **payload["samples"][0]["ground_truth"][0]  # type: ignore[index]
    )
    source_list = [raw]
    ground_truth_list = [gt]
    sample = module.TemporalSampleRecord(
        sample_id="direct",
        sequence_id="direct-sequence",
        split="train",
        n_t=0,
        tau_t_ms=0,
        source_slices=source_list,
        annotation_path="labels/direct.json",
        annotation_sha256="0" * 64,
        ground_truth=ground_truth_list,
    )
    fields = ["x", "y", "z", "intensity"]
    prepared = module.PreparedArtifactRecord(
        source_relative_path="raw/direct.pcd",
        prepared_relative_path="prepared/direct.bin",
        point_count=1,
        size=16,
        sha256="1" * 64,
        dtype="<f4",
        fields=fields,
    )

    source_list.clear()
    ground_truth_list.clear()
    fields[0] = "mutated"
    assert sample.source_slices == (raw,)
    assert sample.ground_truth == (gt,)
    assert prepared.fields == ("x", "y", "z", "intensity")


@pytest.mark.parametrize(
    "object_kind",
    (
        "top",
        "slice",
        "ground_truth",
        "sample",
        "inventory",
        "prepared",
        "sequence_split",
        "trigger",
        "excluded",
    ),
)
@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_every_json_object_uses_an_exact_key_set(
    tmp_path: Path,
    object_kind: str,
    mutation: str,
) -> None:
    module = _module()
    payload = (
        _valid_split_payload()
        if object_kind in ("sequence_split", "trigger")
        else _valid_payload()
    )
    if object_kind == "top":
        target = payload
    elif object_kind == "slice":
        target = payload["samples"][0]["source_slices"][0]  # type: ignore[index]
    elif object_kind == "ground_truth":
        target = payload["samples"][0]["ground_truth"][0]  # type: ignore[index]
    elif object_kind == "sample":
        target = payload["samples"][0]  # type: ignore[index]
    elif object_kind == "inventory":
        target = payload["release_inventory"][0]  # type: ignore[index]
    elif object_kind == "prepared":
        target = payload["prepared_artifacts"][0]  # type: ignore[index]
    elif object_kind == "sequence_split":
        target = payload["sequence_splits"][0]  # type: ignore[index]
    elif object_kind == "trigger":
        target = payload["sequence_splits"][0]["triggers"][0]  # type: ignore[index]
    else:
        target = {
            "sample_id": "excluded",
            "sequence_id": "seq-x",
            "split": "train",
            "n_t": 0,
            "reason": "no_supported_branch",
        }
        payload["excluded_samples"] = [target]
    if mutation == "extra":
        target["unexpected"] = None
    else:
        target.pop(next(iter(target)))
    _rehash(payload)

    with pytest.raises(module.ManifestError, match="fields"):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_history_count",
        "annotation_path",
        "annotation_hash",
        "calibration_path",
        "calibration_hash",
    ),
)
def test_manifest_counts_and_primary_inventory_cross_links(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    payload = _valid_payload()
    if mutation == "wrong_history_count":
        payload["history_eligible_train_count"] = 0
    elif mutation == "annotation_path":
        payload["samples"][0]["annotation_path"] = "labels/missing.json"  # type: ignore[index]
    elif mutation == "annotation_hash":
        payload["samples"][0]["annotation_sha256"] = "f" * 64  # type: ignore[index]
    elif mutation == "calibration_path":
        payload["samples"][0]["source_slices"][0][  # type: ignore[index]
            "calibration_relative_path"
        ] = "calib/missing.json"
    else:
        payload["samples"][0]["source_slices"][0][  # type: ignore[index]
            "calibration_sha256"
        ] = "f" * 64
    _rehash(payload)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_history_eligibility_is_structural_not_validity_flag_based(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    for sample in payload["samples"]:  # type: ignore[union-attr]
        for source in sample["source_slices"]:
            source["payload_valid"] = False
            source["pose_valid"] = False
            source["calibration_valid"] = False
    _rehash(payload)

    manifest = module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )
    assert manifest.history_eligible_train_count == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "empty",
        "duplicate_branch",
        "wrong_order",
        "nonpositive_timestamp",
        "arithmetic",
        "inside_bounds",
    ),
)
def test_sequence_split_trigger_schema_order_and_arithmetic(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    payload = _valid_split_payload()
    triggers = payload["sequence_splits"][0]["triggers"]  # type: ignore[index]
    second = {
        "agent": "rsu",
        "modality": "lidar",
        "previous_capture_timestamp_us": 1_010_000,
        "current_capture_timestamp_us": 1_210_000,
        "interval_us": 200_000,
    }
    if mutation == "empty":
        triggers.clear()
    elif mutation == "duplicate_branch":
        triggers.append(copy.deepcopy(triggers[0]))
    elif mutation == "wrong_order":
        triggers.insert(0, second)
    elif mutation == "nonpositive_timestamp":
        triggers[0]["previous_capture_timestamp_us"] = 0
    elif mutation == "arithmetic":
        triggers[0]["interval_us"] = 199_999
    else:
        triggers[0]["current_capture_timestamp_us"] = 1_100_000
        triggers[0]["interval_us"] = 100_000
    _rehash(payload)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )


def test_direct_controlled_manifest_requires_official_split_hash(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    loaded = module.load_temporal_manifest(
        _write_payload(tmp_path, payload),
        expected_split_hash=FIXTURE_SPLIT_SHA256,
        allow_fixture=True,
    )
    values = {field: getattr(loaded, field) for field in loaded.__dataclass_fields__}
    values.update(
        protocol_scope="controlled",
        history_limit=3,
        history_eligible_train_count=0,
    )
    plain = {
        field: (
            [
                {
                    nested_field: getattr(item, nested_field)
                    for nested_field in item.__dataclass_fields__
                }
                for item in value
            ]
            if field in ("release_inventory", "prepared_artifacts")
            else value
        )
        for field, value in values.items()
    }
    plain["samples"] = [
        {
            sample_field: (
                [
                    {
                        source_field: getattr(source, source_field)
                        for source_field in source.__dataclass_fields__
                    }
                    for source in sample.source_slices
                ]
                if sample_field == "source_slices"
                else [
                    {
                        gt_field: getattr(box, gt_field)
                        for gt_field in box.__dataclass_fields__
                    }
                    for box in sample.ground_truth
                ]
                if sample_field == "ground_truth"
                else getattr(sample, sample_field)
            )
            for sample_field in sample.__dataclass_fields__
        }
        for sample in loaded.samples
    ]
    plain["sequence_splits"] = []
    plain["excluded_samples"] = []
    values["content_sha256"] = module.content_sha256(plain)

    with pytest.raises(module.ManifestError, match="official"):
        module.TemporalManifest(**values)


def test_unrepresentably_large_finite_number_is_wrapped_as_manifest_error(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _valid_payload()
    payload["samples"][0]["ground_truth"][0]["x"] = 10**400  # type: ignore[index]
    _rehash(payload)

    with pytest.raises(module.ManifestError):
        module.load_temporal_manifest(
            _write_payload(tmp_path, payload),
            expected_split_hash=FIXTURE_SPLIT_SHA256,
            allow_fixture=True,
        )
