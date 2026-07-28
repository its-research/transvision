from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from tools.resilient_v2x.build_overlays import (
    EVALUATION_INDEX_TYPE,
    ProtocolBuildError,
    TRAIN_INDEX_TYPE,
    build_cohort_document,
    build_evaluation_overlays,
    build_training_overlays,
    load_cohort,
    write_cohort,
)
from transvision.dataset.resilient_v2x_manifest import (
    canonical_json_bytes,
    content_sha256,
    load_temporal_manifest,
)
from transvision.dataset.resilient_v2x_schedule import read_overlay


ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_TEST = ROOT / "tests/resilient_v2x/test_schedule.py"


def _schedule_fixture_module():
    specification = importlib.util.spec_from_file_location(
        "resilient_v2x_schedule_fixture",
        SCHEDULE_TEST,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@pytest.fixture(scope="module")
def schedule_fixture():
    return _schedule_fixture_module()


def _write_manifest(tmp_path: Path, schedule_fixture, split: str):
    manifest = schedule_fixture._manifest(split)
    path = tmp_path / f"manifest-{split}.json"
    path.write_bytes(canonical_json_bytes(_plain(manifest)))
    restored = load_temporal_manifest(
        path,
        expected_split_hash=schedule_fixture.SHA_A,
        allow_fixture=True,
    )
    assert restored == manifest
    return path, manifest


def _read_index(path: Path, artifact_type: str) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == canonical_json_bytes(value)
    assert value["artifact_type"] == artifact_type
    assert value["content_sha256"] == content_sha256(value)
    return value


def test_cohort_filters_at_worst_delay_and_records_structured_reasons(
    schedule_fixture,
) -> None:
    manifest = schedule_fixture._manifest("test")
    cohort = build_cohort_document(
        manifest,
        split="test",
        max_delay_ms=300,
        max_continuous_fault_duration=1,
    )

    assert cohort["sample_ids"] == ["test-3"]
    assert cohort["candidate_sample_count"] == 4
    assert cohort["included_sample_count"] == 1
    assert cohort["excluded_sample_count"] == 3
    assert cohort["content_sha256"] == content_sha256(cohort)
    excluded = {item["sample_id"]: item for item in cohort["excluded_samples"]}
    sample_two_reasons = excluded["test-2"]["reasons"]
    assert {
        (reason["agent"], reason["modality"], reason["available_source_count"])
        for reason in sample_two_reasons
    } == {("rsu", "lidar", 0), ("rsu", "camera", 0)}
    assert sample_two_reasons[0]["rejected_source_counts"] == {
        "not_arrived_at_max_delay": 3
    }


def test_cohort_rejects_impossible_joint_delay_and_fault_history_contract(
    schedule_fixture,
) -> None:
    manifest = schedule_fixture._manifest("test")

    with pytest.raises(
        ProtocolBuildError,
        match=(
            "joint delay/fault history requirement exceeds represented history: "
            r"300 / 100 \+ 4 - 1 > 3"
        ),
    ):
        build_cohort_document(
            manifest,
            split="test",
            max_delay_ms=300,
            max_continuous_fault_duration=4,
        )

    duration_cohort = build_cohort_document(
        manifest,
        split="test",
        max_delay_ms=0,
        max_continuous_fault_duration=4,
    )
    assert duration_cohort["max_delay_ms"] == 0
    assert duration_cohort["max_continuous_fault_duration"] == 4


def test_cohort_excludes_invalid_payload_pose_and_calibration_metadata(
    tmp_path: Path,
    schedule_fixture,
) -> None:
    manifest = schedule_fixture._manifest("test")
    payload = _plain(manifest)
    assert isinstance(payload, dict)
    for sample in payload["samples"]:
        for source in sample["source_slices"]:
            if (
                source["agent"] == "ego"
                and source["modality"] == "camera"
                and source["n_s"] == 3
            ):
                source["payload_valid"] = False
                source["pose_valid"] = False
                source["calibration_valid"] = False
    payload["content_sha256"] = content_sha256(payload)
    path = tmp_path / "invalid-metadata-manifest.json"
    path.write_bytes(canonical_json_bytes(payload))
    invalid_manifest = load_temporal_manifest(
        path,
        expected_split_hash=schedule_fixture.SHA_A,
        allow_fixture=True,
    )

    cohort = build_cohort_document(
        invalid_manifest,
        split="test",
        max_delay_ms=0,
        max_continuous_fault_duration=1,
    )

    excluded = {item["sample_id"]: item for item in cohort["excluded_samples"]}
    reason = next(
        item
        for item in excluded["test-3"]["reasons"]
        if (item["agent"], item["modality"]) == ("ego", "camera")
    )
    assert reason["rejected_source_counts"] == {
        "invalid_calibration": 1,
        "invalid_payload": 1,
        "invalid_pose": 1,
    }


def test_cohort_load_recomputes_membership_and_rejects_validly_rehashed_tamper(
    tmp_path: Path,
    schedule_fixture,
) -> None:
    manifest = schedule_fixture._manifest("test")
    cohort_path = tmp_path / "cohort.json"
    expected = write_cohort(manifest, cohort_path, split="test")
    assert load_cohort(cohort_path, manifest) == expected

    tampered = copy.deepcopy(expected)
    tampered["sample_ids"] = ["test-2", "test-3"]
    tampered["included_sample_count"] = 2
    tampered["sample_ids_sha256"] = hashlib.sha256(
        canonical_json_bytes(tampered["sample_ids"])
    ).hexdigest()
    tampered["content_sha256"] = content_sha256(tampered)
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_bytes(canonical_json_bytes(tampered))

    with pytest.raises(ProtocolBuildError, match="content or self-hash"):
        load_cohort(tampered_path, manifest)


def test_cohort_publication_is_idempotent_but_never_overwrites_conflicts(
    tmp_path: Path,
    schedule_fixture,
) -> None:
    manifest = schedule_fixture._manifest("test")
    path = tmp_path / "cohort.json"
    first = write_cohort(manifest, path, split="test", max_delay_ms=300)
    second = write_cohort(manifest, path, split="test", max_delay_ms=300)

    assert first == second
    with pytest.raises(ProtocolBuildError, match="destination conflict"):
        write_cohort(manifest, path, split="test", max_delay_ms=0)


def test_training_random_overlays_are_seeded_and_byte_reproducible(
    tmp_path: Path,
    schedule_fixture,
) -> None:
    manifest = schedule_fixture._manifest("train")
    first = build_training_overlays(
        manifest,
        tmp_path / "first",
        protocol_seed=19,
        epochs=(0, 2),
        p_lidar=0.25,
        p_camera=0.5,
    )
    second = build_training_overlays(
        manifest,
        tmp_path / "second",
        protocol_seed=19,
        epochs=(2, 0),
        p_lidar=0.25,
        p_camera=0.5,
    )

    assert first == second
    assert first["artifact_type"] == TRAIN_INDEX_TYPE
    assert first["sample_ids"] == ["train-3"]
    assert first["overlays"]["transport"]["path"] == "train_transport.jsonl.zst"
    assert first["overlays"]["fault"]["path"] == "train_fault.jsonl.zst"
    assert (
        first["overlays"]["transport"]["compressed_sha256"]
        == (second["overlays"]["transport"]["compressed_sha256"])
    )
    transport_entry = first["overlays"]["transport"]
    records = read_overlay(
        tmp_path / "first" / transport_entry["path"],
        transport_entry["uncompressed_sha256"],
    )
    assert {record["epoch"] for record in records} == {0, 2}
    assert {record["delay_ms"] for record in records}.issubset({0, 100, 200, 300})


def test_full_fixed_matrix_reuses_exact_cohort_and_binds_transport_hashes(
    tmp_path: Path,
    schedule_fixture,
) -> None:
    manifest = schedule_fixture._manifest("test")
    cohort_path = tmp_path / "cohort.json"
    cohort = write_cohort(manifest, cohort_path, split="test")
    index = build_evaluation_overlays(
        manifest,
        cohort_path,
        tmp_path / "matrix",
        duration=1,
    )

    assert index["artifact_type"] == EVALUATION_INDEX_TYPE
    assert index["sample_ids"] == cohort["sample_ids"] == ["test-3"]
    assert len(index["transport_overlays"]) == 4
    assert len(index["fault_overlays"]) == 36
    assert {item["overlay"]["path"] for item in index["transport_overlays"]} == {
        "test_transport_delay_000.jsonl.zst",
        "test_transport_delay_100.jsonl.zst",
        "test_transport_delay_200.jsonl.zst",
        "test_transport_delay_300.jsonl.zst",
    }
    transport_by_delay = {
        item["delay_ms"]: item["overlay"]["uncompressed_sha256"]
        for item in index["transport_overlays"]
    }
    for item in index["fault_overlays"]:
        assert item["transport_overlay_sha256"] == transport_by_delay[item["delay_ms"]]
        overlay = item["overlay"]
        records = read_overlay(
            tmp_path / "matrix" / overlay["path"],
            overlay["uncompressed_sha256"],
        )
        assert {record["sample_id"] for record in records} == {"test-3"}

    selected = next(
        item
        for item in index["fault_overlays"]
        if item["delay_ms"] == 300
        and item["condition"] == "L-Fail"
        and item["agent_scope"] == "E-only"
    )
    main_lidar = next(
        item
        for item in index["fault_overlays"]
        if item["delay_ms"] == 300
        and item["condition"] == "L-Fail"
        and item["agent_scope"] == "E+R"
    )
    assert main_lidar["overlay"]["path"] == ("test_causal_delay_300_l_fail.jsonl.zst")
    assert main_lidar["alias_paths"] == ["test_causal_300_E+R_lidar_d1.jsonl.zst"]
    assert (tmp_path / "matrix" / main_lidar["alias_paths"][0]).read_bytes() == (
        tmp_path / "matrix" / main_lidar["overlay"]["path"]
    ).read_bytes()
    assert selected["overlay"]["path"] == ("test_causal_300_E-only_lidar_d1.jsonl.zst")
    records = read_overlay(
        tmp_path / "matrix" / selected["overlay"]["path"],
        selected["overlay"]["uncompressed_sha256"],
    )
    assert {
        (record["agent"], record["modality"]) for record in records if record["masked"]
    } == {("ego", "lidar")}


def test_continuous_duration_and_agent_scope_are_preserved(
    tmp_path: Path,
    schedule_fixture,
) -> None:
    manifest = schedule_fixture._manifest("test")
    cohort_path = tmp_path / "cohort-duration-2.json"
    cohort = write_cohort(
        manifest,
        cohort_path,
        split="test",
        max_delay_ms=0,
        max_continuous_fault_duration=2,
    )
    assert cohort["sample_ids"] == ["test-1", "test-2", "test-3"]
    index = build_evaluation_overlays(
        manifest,
        cohort_path,
        tmp_path / "duration-2",
        delays=(0,),
        conditions=("C-Fail",),
        agent_scopes=("R-only",),
        duration=2,
    )
    fault = index["fault_overlays"][0]
    assert fault["duration"] == 2
    assert fault["agents"] == ["rsu"]
    records = read_overlay(
        tmp_path / "duration-2" / fault["overlay"]["path"],
        fault["overlay"]["uncompressed_sha256"],
    )
    masked = [record for record in records if record["masked"]]
    assert len(masked) == 6
    assert {(record["agent"], record["modality"]) for record in masked} == {
        ("rsu", "camera")
    }


def test_cli_builds_cohort_evaluation_and_training_artifacts(
    tmp_path: Path,
    schedule_fixture,
) -> None:
    test_manifest_path, _ = _write_manifest(tmp_path, schedule_fixture, "test")
    train_manifest_path, _ = _write_manifest(tmp_path, schedule_fixture, "train")
    cohort_path = tmp_path / "cli-cohort.json"
    evaluation_dir = tmp_path / "cli-evaluation"
    training_dir = tmp_path / "cli-training"
    common_test = [
        str(test_manifest_path),
        "--expected-split-sha256",
        schedule_fixture.SHA_A,
        "--allow-fixture",
    ]

    cohort_result = subprocess.run(
        [
            sys.executable,
            "tools/resilient_v2x/build_overlays.py",
            "cohort",
            *common_test,
            "--split",
            "test",
            "--out",
            str(cohort_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    evaluation_result = subprocess.run(
        [
            sys.executable,
            "tools/resilient_v2x/build_overlays.py",
            "evaluation",
            *common_test,
            "--cohort",
            str(cohort_path),
            "--delays",
            "0",
            "300",
            "--conditions",
            "Full",
            "L-Fail",
            "--agents",
            "E+R",
            "E-only",
            "--duration",
            "1",
            "--out-dir",
            str(evaluation_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    training_result = subprocess.run(
        [
            sys.executable,
            "tools/resilient_v2x/build_overlays.py",
            "train",
            str(train_manifest_path),
            "--expected-split-sha256",
            schedule_fixture.SHA_A,
            "--allow-fixture",
            "--protocol-seed",
            "23",
            "--epochs",
            "0",
            "1",
            "--p-lidar",
            "0.2",
            "--p-camera",
            "0.4",
            "--out-dir",
            str(training_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(cohort_result.stdout)["artifact_type"].endswith("cohort")
    assert json.loads(evaluation_result.stdout)["artifact_type"] == (
        EVALUATION_INDEX_TYPE
    )
    assert json.loads(training_result.stdout)["artifact_type"] == TRAIN_INDEX_TYPE
    evaluation = _read_index(
        evaluation_dir / "evaluation_overlays.json",
        EVALUATION_INDEX_TYPE,
    )
    training = _read_index(
        training_dir / "training_overlays.json",
        TRAIN_INDEX_TYPE,
    )
    assert len(evaluation["fault_overlays"]) == 8
    assert training["epochs"] == [0, 1]
