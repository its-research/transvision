from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Callable

import pytest
import zstandard

from transvision.dataset.resilient_v2x_manifest import (
    MANIFEST_SCHEMA_VERSION,
    GroundTruthBoxRecord,
    PreparedArtifactRecord,
    RawSliceRecord,
    ReleaseInventoryEntry,
    TemporalManifest,
    TemporalSampleRecord,
    canonical_json_bytes,
    content_sha256,
    release_inventory_sha256,
)
from transvision.dataset.resilient_v2x_schedule import (
    ArrivalRelativeFaultPlan,
    CausalFaultPlan,
    FaultOverlayRecord,
    FaultPlan,
    OverlayDigest,
    ScheduleError,
    TRAINING_CONDITION_HASH_DOMAIN,
    TRAINING_CONDITION_MODE,
    TransportOverlayRecord,
    TransportPlan,
    augmentation_seed,
    bernoulli_from_hash,
    delay_from_hash,
    read_overlay,
    stable_uint64,
    training_condition_from_hash,
    write_arrival_relative_fault_overlay,
    write_causal_fault_overlay,
    write_fault_overlay,
    write_transport_overlay,
)


KEY = {"seed": 0, "epoch": 1, "sample_id": "000123", "n_s": 8}
SHA_A = "a" * 64
SHA_B = "b" * 64
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
BRANCHES = (
    ("ego", "lidar"),
    ("rsu", "lidar"),
    ("ego", "camera"),
    ("rsu", "camera"),
)


def _slice(n_s: int, agent: str, modality: str) -> RawSliceRecord:
    suffix = "pcd" if modality == "lidar" else "jpg"
    return RawSliceRecord(
        agent=agent,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        n_s=n_s,
        tau_s_ms=n_s * 100,
        capture_timestamp_us=1_000_000 + n_s * 100_000,
        frame_id=f"{agent}-{modality}-{n_s}",
        packet_id=f"sequence-0:{n_s}:{agent}:{modality}",
        relative_path=f"raw/{agent}-{modality}-{n_s}.{suffix}",
        world_from_agent=IDENTITY_4X4,
        agent_from_sensor=IDENTITY_4X4,
        calibration_relative_path="calib/primary.json",
        calibration_sha256=SHA_B,
        camera_intrinsic=None if modality == "lidar" else CAMERA_INTRINSIC,
        payload_valid=True,
        pose_valid=True,
        calibration_valid=True,
    )


def _samples(split: str = "test") -> tuple[TemporalSampleRecord, ...]:
    slices = {
        (n_s, agent, modality): _slice(n_s, agent, modality)
        for n_s in range(4)
        for agent, modality in BRANCHES
    }
    return tuple(
        TemporalSampleRecord(
            sample_id=f"{split}-{n_t}",
            sequence_id="sequence-0",
            split=split,  # type: ignore[arg-type]
            n_t=n_t,
            tau_t_ms=n_t * 100,
            source_slices=tuple(
                slices[(n_s, agent, modality)]
                for n_s in range(n_t + 1)
                for agent, modality in BRANCHES
            ),
            annotation_path=f"labels/{n_t}.json",
            annotation_sha256=SHA_A,
            ground_truth=(
                GroundTruthBoxRecord(
                    class_name="Car",
                    x=0.0,
                    y=0.0,
                    z_bottom=0.0,
                    length=4.0,
                    width=2.0,
                    height=1.5,
                    yaw=0.0,
                    source_annotation_index=0,
                ),
            ),
        )
        for n_t in range(4)
    )


def _manifest(split: str = "test") -> TemporalManifest:
    samples = _samples(split)
    source_by_path = {
        source.relative_path: source
        for sample in samples
        for source in sample.source_slices
    }
    inventory = [
        ReleaseInventoryEntry(
            relative_path="calib/primary.json",
            size=1,
            sha256=SHA_B,
        ),
        *(
            ReleaseInventoryEntry(
                relative_path=f"labels/{n_t}.json",
                size=1,
                sha256=SHA_A,
            )
            for n_t in range(4)
        ),
        *(
            ReleaseInventoryEntry(
                relative_path=path,
                size=1,
                sha256=hashlib.sha256(path.encode()).hexdigest(),
            )
            for path in source_by_path
        ),
    ]
    inventory.sort(key=lambda item: item.relative_path)
    prepared = tuple(
        PreparedArtifactRecord(
            source_relative_path=source.relative_path,
            prepared_relative_path=f"prepared/{source.frame_id}.bin",
            point_count=0,
            size=0,
            sha256=hashlib.sha256(b"").hexdigest(),
            dtype="<f4",
            fields=("x", "y", "z", "intensity"),
        )
        for source in sorted(
            source_by_path.values(), key=lambda item: item.relative_path
        )
        if source.modality == "lidar"
    )
    values = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_scope": "fixture",
        "delta_t_ms": 100,
        "history_limit": 3,
        "interval_min_ms": 50,
        "interval_max_ms": 150,
        "max_capture_skew_ms": 50,
        "split_sha256": SHA_A,
        "dataset_release_sha256": release_inventory_sha256(inventory),
        "release_inventory": tuple(inventory),
        "prepared_artifacts": prepared,
        "history_eligible_train_count": 1 if split == "train" else 0,
        "sequence_splits": (),
        "excluded_samples": (),
        "samples": samples,
    }
    plain_values = dict(values)
    for key in ("release_inventory", "prepared_artifacts", "samples"):
        plain_values[key] = tuple(asdict(item) for item in values[key])
    values["content_sha256"] = content_sha256(plain_values)
    return TemporalManifest(**values)  # type: ignore[arg-type]


def _transport_plan(
    sample: TemporalSampleRecord,
    *,
    delay_ms: int = 0,
) -> TransportPlan:
    return TransportPlan(
        temporal_manifest_sha256=SHA_A,
        split=sample.split,
        samples=(sample,),
        mode="fixed_evaluation",
        protocol_seed=None,
        epochs=(),
        delay_values_ms=(),
        fixed_delay_ms=delay_ms,
    )


def _global_fault_plan(
    sample: TemporalSampleRecord,
    condition: str,
) -> FaultPlan:
    return FaultPlan(
        temporal_manifest_sha256=SHA_A,
        split=sample.split,
        samples=(sample,),
        mode="global_target",
        protocol_seed=None,
        epochs=(),
        condition=condition,  # type: ignore[arg-type]
        p_lidar=None,
        p_camera=None,
        agents=("ego", "rsu"),
        modality=None,
        duration=None,
    )


def _records(path: Path, digest: OverlayDigest) -> tuple[dict[str, object], ...]:
    return tuple(
        dict(record) for record in read_overlay(path, digest.uncompressed_sha256)
    )


def _select_latest(
    sample: TemporalSampleRecord,
    agent: str,
    modality: str,
    transport_records: tuple[dict[str, object], ...],
    fault_records: tuple[dict[str, object], ...],
) -> int | None:
    arrival_by_packet = {
        record["packet_id"]: record["arrival_tau_ms"] for record in transport_records
    }
    masked = {
        (record["agent"], record["modality"], record["n_s"])
        for record in fault_records
        if record["masked"]
    }
    eligible: list[int] = []
    for source in sample.source_slices:
        if (source.agent, source.modality) != (agent, modality):
            continue
        if (agent, modality, source.n_s) in masked:
            continue
        if agent == "rsu" and arrival_by_packet[source.packet_id] > sample.tau_t_ms:
            continue
        eligible.append(source.n_s)
    return max(eligible, default=None)


def test_stable_hash_fixed_vector() -> None:
    assert stable_uint64("transport-delay-v1", KEY) == 46208971577506763


def test_stable_hash_matches_independent_sha256_construction() -> None:
    payload = b"transport-delay-v1\x00" + canonical_json_bytes(KEY)
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    assert expected == stable_uint64("transport-delay-v1", KEY)


def test_json_key_order_does_not_change_hash() -> None:
    reversed_key = dict(reversed(list(KEY.items())))
    assert stable_uint64("transport-delay-v1", KEY) == stable_uint64(
        "transport-delay-v1", reversed_key
    )


def test_delay_is_one_of_four_protocol_values() -> None:
    assert delay_from_hash("transport-delay-choice-v1", KEY, (0, 100, 200, 300)) in {
        0,
        100,
        200,
        300,
    }


def test_training_condition_hash_maps_uniformly_over_formal_matrix() -> None:
    matrix = tuple(
        (delay_ms, condition)
        for delay_ms in (0, 100, 200, 300)
        for condition in ("Full", "L-Fail", "C-Fail")
    )
    key = {"seed": 17, "epoch": 3, "sample_id": "train-3"}
    expected = matrix[stable_uint64(TRAINING_CONDITION_HASH_DOMAIN, key) % 12]

    assert training_condition_from_hash(17, 3, "train-3") == expected
    assert {
        training_condition_from_hash(17, epoch, "train-3")
        for epoch in range(256)
    } == set(matrix)


@pytest.mark.parametrize(
    "values",
    [
        (-1, 0, "sample"),
        (0, -1, "sample"),
        (0, 0, ""),
        (True, 0, "sample"),
    ],
)
def test_training_condition_hash_rejects_invalid_identity(
    values: tuple[object, object, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        training_condition_from_hash(*values)  # type: ignore[arg-type]


@pytest.mark.parametrize("probability, expected", [(0.0, False), (1.0, True)])
def test_hash_bernoulli_has_exact_closed_probability_boundaries(
    probability: float,
    expected: bool,
) -> None:
    assert bernoulli_from_hash("fault-v1", KEY, probability) is expected


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), True])
def test_hash_bernoulli_rejects_values_outside_probability_domain(
    probability: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        bernoulli_from_hash("fault-v1", KEY, probability)  # type: ignore[arg-type]


def test_augmentation_seed_is_stable_and_epoch_sample_specific() -> None:
    seed = augmentation_seed(7, 3, "sample")
    assert seed == augmentation_seed(7, 3, "sample")
    assert seed != augmentation_seed(7, 4, "sample")
    assert seed != augmentation_seed(7, 3, "other")
    assert 0 <= seed < 2**64


def test_transport_plan_freezes_inputs_and_is_frozen() -> None:
    sample = _samples("train")[-1]
    source_samples = [sample]
    source_epochs = [0, 1]
    plan = TransportPlan(
        temporal_manifest_sha256=SHA_A,
        split="train",
        samples=source_samples,  # type: ignore[arg-type]
        mode="train_random",
        protocol_seed=7,
        epochs=source_epochs,  # type: ignore[arg-type]
        delay_values_ms=[0, 100, 200, 300],  # type: ignore[arg-type]
        fixed_delay_ms=None,
    )
    source_samples.clear()
    source_epochs.clear()
    assert plan.samples == (sample,)
    assert plan.epochs == (0, 1)
    assert plan.delay_values_ms == (0, 100, 200, 300)
    with pytest.raises(FrozenInstanceError):
        plan.protocol_seed = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol_seed": None},
        {"epochs": ()},
        {"epochs": (0, 0)},
        {"delay_values_ms": (0, 100, 300)},
        {"fixed_delay_ms": 0},
        {"split": "val"},
    ],
)
def test_train_transport_plan_rejects_non_protocol_configuration(
    changes: dict[str, object],
) -> None:
    sample = _samples("train")[-1]
    values = {
        "temporal_manifest_sha256": SHA_A,
        "split": "train",
        "samples": (sample,),
        "mode": "train_random",
        "protocol_seed": 7,
        "epochs": (0, 1),
        "delay_values_ms": (0, 100, 200, 300),
        "fixed_delay_ms": None,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        TransportPlan(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol_seed": 0},
        {"epochs": (0,)},
        {"delay_values_ms": (0, 100, 200, 300)},
        {"fixed_delay_ms": None},
        {"fixed_delay_ms": 2000},
        {"split": "train"},
    ],
)
def test_fixed_transport_plan_rejects_non_protocol_configuration(
    changes: dict[str, object],
) -> None:
    sample = _samples("test")[-1]
    values = {
        "temporal_manifest_sha256": SHA_A,
        "split": "test",
        "samples": (sample,),
        "mode": "fixed_evaluation",
        "protocol_seed": None,
        "epochs": (),
        "delay_values_ms": (),
        "fixed_delay_ms": 0,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        TransportPlan(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol_seed": None},
        {"epochs": ()},
        {"p_lidar": None},
        {"p_camera": 1.1},
        {"condition": "Full"},
        {"agents": ("rsu",)},
        {"modality": "lidar"},
        {"duration": 1},
        {"split": "test"},
    ],
)
def test_train_fault_plan_rejects_non_target_random_configuration(
    changes: dict[str, object],
) -> None:
    sample = _samples("train")[-1]
    values = {
        "temporal_manifest_sha256": SHA_A,
        "split": "train",
        "samples": (sample,),
        "mode": "train_random",
        "protocol_seed": 7,
        "epochs": (0,),
        "condition": None,
        "p_lidar": 0.5,
        "p_camera": 0.5,
        "agents": ("ego", "rsu"),
        "modality": None,
        "duration": None,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        FaultPlan(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol_seed": None},
        {"epochs": ()},
        {"condition": "Full"},
        {"p_lidar": 1 / 3},
        {"p_camera": 1 / 3},
        {"agents": ("ego",)},
        {"modality": "camera"},
        {"duration": 1},
        {"split": "test"},
    ],
)
def test_condition_matrix_fault_plan_rejects_independent_controls(
    changes: dict[str, object],
) -> None:
    sample = _samples("train")[-1]
    values = {
        "temporal_manifest_sha256": SHA_A,
        "split": "train",
        "samples": (sample,),
        "mode": TRAINING_CONDITION_MODE,
        "protocol_seed": 7,
        "epochs": (0,),
        "condition": None,
        "p_lidar": None,
        "p_camera": None,
        "agents": ("ego", "rsu"),
        "modality": None,
        "duration": None,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        FaultPlan(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mode, valid, changes",
    [
        ("global_target", _global_fault_plan, {"condition": None}),
        ("global_target", _global_fault_plan, {"p_lidar": 0.0}),
        ("global_target", _global_fault_plan, {"agents": ("ego",)}),
        ("global_target", _global_fault_plan, {"modality": "lidar"}),
        ("global_target", _global_fault_plan, {"duration": 1}),
        ("continuous", None, {"condition": "L-Fail"}),
        ("continuous", None, {"agents": ()}),
        ("continuous", None, {"modality": None}),
        ("continuous", None, {"duration": 0}),
        ("continuous", None, {"duration": 5}),
        ("continuous", None, {"p_camera": 0.0}),
    ],
)
def test_evaluation_fault_plans_encode_scope_and_duration_explicitly(
    mode: str,
    valid: object,
    changes: dict[str, object],
) -> None:
    del valid
    sample = _samples("test")[-1]
    values = {
        "temporal_manifest_sha256": SHA_A,
        "split": "test",
        "samples": (sample,),
        "mode": mode,
        "protocol_seed": None,
        "epochs": (),
        "condition": "Full" if mode == "global_target" else None,
        "p_lidar": None,
        "p_camera": None,
        "agents": ("ego", "rsu"),
        "modality": None if mode == "global_target" else "lidar",
        "duration": None if mode == "global_target" else 1,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        FaultPlan(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"temporal_manifest_sha256": "A" * 64},
        {"transport_overlay_sha256": "short"},
        {"split": "train"},
        {"scope": "both"},
        {"modality": "radar"},
        {"fixed_delay_ms": 100},
    ],
)
def test_arrival_relative_plan_is_evaluation_only_and_digest_bound(
    changes: dict[str, object],
) -> None:
    sample = _samples("test")[-1]
    values = {
        "temporal_manifest_sha256": SHA_A,
        "transport_overlay_sha256": SHA_B,
        "split": "test",
        "samples": (sample,),
        "scope": "rsu",
        "modality": "lidar",
        "fixed_delay_ms": 300,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        ArrivalRelativeFaultPlan(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"record_count": -1},
        {"uncompressed_size": -1},
        {"compressed_size": -1},
        {"uncompressed_sha256": "A" * 64},
        {"compressed_sha256": "0" * 63},
    ],
)
def test_overlay_digest_rejects_negative_numbers_and_noncanonical_hashes(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    values = {
        "path": tmp_path / "overlay.jsonl.zst",
        "record_count": 0,
        "uncompressed_size": 0,
        "uncompressed_sha256": "0" * 64,
        "compressed_size": 0,
        "compressed_sha256": "0" * 64,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        OverlayDigest(**values)  # type: ignore[arg-type]


def test_overlay_record_dataclasses_are_frozen_and_validate_exact_types() -> None:
    transport = TransportOverlayRecord(
        epoch=None,
        sample_id="sample",
        packet_id="packet",
        n_s=0,
        delay_ms=0,
        arrival_tau_ms=0,
    )
    fault = FaultOverlayRecord(
        epoch=None,
        sample_id="sample",
        agent="ego",
        modality="lidar",
        n_s=0,
        masked=True,
        pre_mask_selected_n_s=None,
        fallback_selected_n_s=None,
    )
    with pytest.raises(FrozenInstanceError):
        transport.delay_ms = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        fault.masked = False  # type: ignore[misc]
    with pytest.raises((TypeError, ValueError)):
        replace(transport, n_s=True)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        replace(fault, agent="other")  # type: ignore[arg-type]


def test_transport_overlay_has_exact_rsu_packet_coverage_and_shared_tick_delay(
    tmp_path: Path,
) -> None:
    sample = _samples("train")[-1]
    plan = TransportPlan(
        temporal_manifest_sha256=SHA_A,
        split="train",
        samples=(sample,),
        mode="train_random",
        protocol_seed=17,
        epochs=(2,),
        delay_values_ms=(0, 100, 200, 300),
        fixed_delay_ms=None,
    )
    digest = write_transport_overlay(plan, tmp_path / "transport.jsonl.zst")
    records = _records(digest.path, digest)
    expected_packets = {
        source.packet_id for source in sample.source_slices if source.agent == "rsu"
    }
    assert len(records) == len(expected_packets)
    assert {record["packet_id"] for record in records} == expected_packets
    assert all(record["epoch"] == 2 for record in records)
    assert all(":ego:" not in str(record["packet_id"]) for record in records)
    for n_s in range(4):
        tick = [record for record in records if record["n_s"] == n_s]
        assert len(tick) == 2
        assert {record["delay_ms"] for record in tick} <= {0, 100, 200, 300}
        assert len({record["delay_ms"] for record in tick}) == 1
        assert all(
            record["arrival_tau_ms"] == n_s * 100 + record["delay_ms"]
            for record in tick
        )


def test_transport_writer_rejects_empty_rsu_coverage_before_publication(
    tmp_path: Path,
) -> None:
    sample = _samples("test")[-1]
    ego_only = replace(
        sample,
        source_slices=tuple(
            source for source in sample.source_slices if source.agent == "ego"
        ),
    )
    output = tmp_path / "empty-transport.zst"

    with pytest.raises(ScheduleError, match="RSU"):
        write_transport_overlay(_transport_plan(ego_only), output)

    assert not output.exists()


def test_transport_and_fault_randomness_ignore_sample_and_epoch_iteration_order(
    tmp_path: Path,
) -> None:
    samples = _samples("train")[-2:]
    transport_values = {
        "temporal_manifest_sha256": SHA_A,
        "split": "train",
        "mode": "train_random",
        "protocol_seed": 9,
        "delay_values_ms": (0, 100, 200, 300),
        "fixed_delay_ms": None,
    }
    fault_values = {
        "temporal_manifest_sha256": SHA_A,
        "split": "train",
        "mode": "train_random",
        "protocol_seed": 9,
        "condition": None,
        "p_lidar": 0.4,
        "p_camera": 0.6,
        "agents": ("ego", "rsu"),
        "modality": None,
        "duration": None,
    }
    first_transport = write_transport_overlay(
        TransportPlan(samples=samples, epochs=(3, 1), **transport_values),
        tmp_path / "transport-first.zst",
    )
    second_transport = write_transport_overlay(
        TransportPlan(
            samples=tuple(reversed(samples)),
            epochs=(1, 3),
            **transport_values,
        ),
        tmp_path / "transport-second.zst",
    )
    first_fault = write_fault_overlay(
        FaultPlan(samples=samples, epochs=(3, 1), **fault_values),
        tmp_path / "fault-first.zst",
    )
    second_fault = write_fault_overlay(
        FaultPlan(
            samples=tuple(reversed(samples)),
            epochs=(1, 3),
            **fault_values,
        ),
        tmp_path / "fault-second.zst",
    )
    assert first_transport.uncompressed_sha256 == second_transport.uncompressed_sha256
    assert first_transport.compressed_sha256 == second_transport.compressed_sha256
    assert first_fault.uncompressed_sha256 == second_fault.uncompressed_sha256
    assert first_fault.compressed_sha256 == second_fault.compressed_sha256


def test_training_faults_cover_only_target_tick_for_all_branches(
    tmp_path: Path,
) -> None:
    sample = _samples("train")[-1]
    plan = FaultPlan(
        temporal_manifest_sha256=SHA_A,
        split="train",
        samples=(sample,),
        mode="train_random",
        protocol_seed=11,
        epochs=(4,),
        condition=None,
        p_lidar=1.0,
        p_camera=0.0,
        agents=("ego", "rsu"),
        modality=None,
        duration=None,
    )
    digest = write_fault_overlay(plan, tmp_path / "fault.zst")
    records = _records(digest.path, digest)
    assert len(records) == 4
    assert {record["n_s"] for record in records} == {sample.n_t}
    assert {
        (record["agent"], record["modality"], record["masked"]) for record in records
    } == {
        ("ego", "lidar", True),
        ("rsu", "lidar", True),
        ("ego", "camera", False),
        ("rsu", "camera", False),
    }
    assert all(record["pre_mask_selected_n_s"] is None for record in records)
    assert all(record["fallback_selected_n_s"] is None for record in records)


def test_condition_matrix_training_shares_delay_and_causal_fault_condition(
    tmp_path: Path,
) -> None:
    sample = _samples("train")[-1]
    epochs = tuple(range(256))
    seed = 17
    transport_digest = write_transport_overlay(
        TransportPlan(
            temporal_manifest_sha256=SHA_A,
            split="train",
            samples=(sample,),
            mode=TRAINING_CONDITION_MODE,
            protocol_seed=seed,
            epochs=epochs,
            delay_values_ms=(0, 100, 200, 300),
            fixed_delay_ms=None,
        ),
        tmp_path / "condition-transport.zst",
    )
    fault_digest = write_fault_overlay(
        FaultPlan(
            temporal_manifest_sha256=SHA_A,
            split="train",
            samples=(sample,),
            mode=TRAINING_CONDITION_MODE,
            protocol_seed=seed,
            epochs=epochs,
            condition=None,
            p_lidar=None,
            p_camera=None,
            agents=("ego", "rsu"),
            modality=None,
            duration=None,
        ),
        tmp_path / "condition-fault.zst",
    )
    transports = _records(transport_digest.path, transport_digest)
    faults = _records(fault_digest.path, fault_digest)
    formal_matrix = {
        (delay_ms, condition)
        for delay_ms in (0, 100, 200, 300)
        for condition in ("Full", "L-Fail", "C-Fail")
    }
    observed: set[tuple[int, str]] = set()

    for epoch in epochs:
        delay_ms, condition = training_condition_from_hash(
            seed,
            epoch,
            sample.sample_id,
        )
        observed.add((delay_ms, condition))
        epoch_transport = [
            record for record in transports if record["epoch"] == epoch
        ]
        assert len(epoch_transport) == 8
        assert {record["delay_ms"] for record in epoch_transport} == {delay_ms}

        epoch_faults = [record for record in faults if record["epoch"] == epoch]
        assert len(epoch_faults) == 4
        expected_endpoint = {
            "ego": sample.n_t,
            "rsu": sample.n_t - delay_ms // 100,
        }
        assert all(
            record["n_s"] == expected_endpoint[str(record["agent"])]
            for record in epoch_faults
        )
        expected_masked = {
            "Full": set(),
            "L-Fail": {("ego", "lidar"), ("rsu", "lidar")},
            "C-Fail": {("ego", "camera"), ("rsu", "camera")},
        }[condition]
        assert {
            (record["agent"], record["modality"])
            for record in epoch_faults
            if record["masked"]
        } == expected_masked
        assert all(
            record["pre_mask_selected_n_s"] == record["n_s"]
            for record in epoch_faults
        )

    assert observed == formal_matrix


def test_fixed_evaluation_condition_is_independent_of_sample_iteration_order(
    tmp_path: Path,
) -> None:
    samples = _samples("test")[-2:]
    values = asdict(_global_fault_plan(samples[0], "L-Fail"))
    values["samples"] = samples
    first = write_fault_overlay(
        FaultPlan(**values),  # type: ignore[arg-type]
        tmp_path / "first.zst",
    )
    values["samples"] = tuple(reversed(samples))
    second = write_fault_overlay(
        FaultPlan(**values),  # type: ignore[arg-type]
        tmp_path / "second.zst",
    )
    assert first.uncompressed_sha256 == second.uncompressed_sha256
    assert first.compressed_sha256 == second.compressed_sha256


@pytest.mark.parametrize("delay_ms", [0, 100, 200, 300])
@pytest.mark.parametrize("condition", ["Full", "L-Fail", "C-Fail"])
def test_global_target_faults_produce_approved_horizon_table(
    tmp_path: Path,
    delay_ms: int,
    condition: str,
) -> None:
    sample = _samples("test")[-1]
    transport_digest = write_transport_overlay(
        _transport_plan(sample, delay_ms=delay_ms),
        tmp_path / f"transport-{delay_ms}.zst",
    )
    fault_digest = write_fault_overlay(
        _global_fault_plan(sample, condition),
        tmp_path / f"fault-{condition}.zst",
    )
    transport = _records(transport_digest.path, transport_digest)
    faults = _records(fault_digest.path, fault_digest)
    rsu_without_target_fault = {0: 3, 100: 2, 200: 1, 300: 0}[delay_ms]
    rsu_with_target_fault = {0: 2, 100: 2, 200: 1, 300: 0}[delay_ms]
    expected = {
        ("ego", "lidar"): 2 if condition == "L-Fail" else 3,
        ("rsu", "lidar"): (
            rsu_with_target_fault if condition == "L-Fail" else rsu_without_target_fault
        ),
        ("ego", "camera"): 2 if condition == "C-Fail" else 3,
        ("rsu", "camera"): (
            rsu_with_target_fault if condition == "C-Fail" else rsu_without_target_fault
        ),
    }
    assert {
        branch: _select_latest(sample, *branch, transport, faults)
        for branch in BRANCHES
    } == expected


@pytest.mark.parametrize("duration", [1, 2, 3, 4])
def test_continuous_lidar_ego_and_rsu_masks_exact_target_ending_duration(
    tmp_path: Path,
    duration: int,
) -> None:
    sample = _samples("test")[-1]
    plan = FaultPlan(
        temporal_manifest_sha256=SHA_A,
        split="test",
        samples=(sample,),
        mode="continuous",
        protocol_seed=None,
        epochs=(),
        condition=None,
        p_lidar=None,
        p_camera=None,
        agents=("ego", "rsu"),
        modality="lidar",
        duration=duration,
    )
    digest = write_fault_overlay(plan, tmp_path / f"continuous-{duration}.zst")
    records = _records(digest.path, digest)
    assert len(records) == 2 * duration
    assert {
        (record["agent"], record["modality"], record["n_s"], record["masked"])
        for record in records
    } == {
        (agent, "lidar", n_s, True)
        for agent in ("ego", "rsu")
        for n_s in range(4 - duration, 4)
    }


@pytest.mark.parametrize(
    "scope, delay_ms, expected_pre_mask, expected_fallback",
    [
        ("ego", 300, 3, 2),
        ("rsu", 0, 3, 2),
        ("rsu", 300, 0, None),
    ],
)
def test_arrival_relative_fault_masks_selected_source_once_then_falls_back_once(
    tmp_path: Path,
    scope: str,
    delay_ms: int,
    expected_pre_mask: int,
    expected_fallback: int | None,
) -> None:
    manifest = _manifest("test")
    sample = manifest.samples[-1]
    transport_digest = write_transport_overlay(
        _transport_plan(sample, delay_ms=delay_ms),
        tmp_path / f"transport-{scope}-{delay_ms}.zst",
    )
    transport = read_overlay(
        transport_digest.path,
        transport_digest.uncompressed_sha256,
    )
    plan = ArrivalRelativeFaultPlan(
        temporal_manifest_sha256=manifest.content_sha256,
        transport_overlay_sha256=transport_digest.uncompressed_sha256,
        split="test",
        samples=(sample,),
        scope=scope,  # type: ignore[arg-type]
        modality="lidar",
        fixed_delay_ms=delay_ms,  # type: ignore[arg-type]
    )
    digest = write_arrival_relative_fault_overlay(
        plan,
        manifest,
        transport,
        tmp_path / f"arrival-{scope}-{delay_ms}.zst",
    )
    records = _records(digest.path, digest)
    assert records == (
        {
            "agent": scope,
            "epoch": None,
            "fallback_selected_n_s": expected_fallback,
            "masked": True,
            "modality": "lidar",
            "n_s": expected_pre_mask,
            "pre_mask_selected_n_s": expected_pre_mask,
            "sample_id": sample.sample_id,
        },
    )
    if expected_fallback is not None:
        assert all(record["n_s"] != expected_fallback for record in records)


def test_arrival_selection_uses_maximum_eligible_tick_for_shuffled_sources() -> None:
    import transvision.dataset.resilient_v2x_schedule as schedule

    sample = _samples("test")[-1]
    shuffled = replace(
        sample,
        source_slices=tuple(
            sorted(
                sample.source_slices,
                key=lambda source: source.n_s,
                reverse=True,
            )
        ),
    )
    arrivals = {
        source.packet_id: source.tau_s_ms
        for source in shuffled.source_slices
        if source.agent == "rsu"
    }

    assert (
        schedule._select_latest_source(
            shuffled,
            "rsu",
            "lidar",
            arrivals,
        )
        == 3
    )
    assert (
        schedule._select_latest_source(
            shuffled,
            "rsu",
            "lidar",
            arrivals,
            excluded_n_s=3,
        )
        == 2
    )


@pytest.mark.parametrize("mutation", ["ego", "ego-delay", "omit", "duplicate"])
def test_arrival_relative_writer_rejects_invalid_transport_schema_or_coverage(
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest = _manifest("test")
    sample = manifest.samples[-1]
    transport_digest = write_transport_overlay(
        _transport_plan(sample, delay_ms=0),
        tmp_path / "transport.zst",
    )
    transport = list(_records(transport_digest.path, transport_digest))
    if mutation in ("ego", "ego-delay"):
        source = next(item for item in sample.source_slices if item.agent == "ego")
        transport.append(
            {
                "arrival_tau_ms": source.tau_s_ms
                + (100 if mutation == "ego-delay" else 0),
                "delay_ms": 100 if mutation == "ego-delay" else 0,
                "epoch": None,
                "n_s": source.n_s,
                "packet_id": source.packet_id,
                "sample_id": sample.sample_id,
            }
        )
    elif mutation == "omit":
        transport.pop()
    else:
        transport.append(dict(transport[-1]))
    plan = ArrivalRelativeFaultPlan(
        temporal_manifest_sha256=manifest.content_sha256,
        transport_overlay_sha256=transport_digest.uncompressed_sha256,
        split="test",
        samples=(sample,),
        scope="rsu",
        modality="lidar",
        fixed_delay_ms=0,
    )
    with pytest.raises(ScheduleError):
        write_arrival_relative_fault_overlay(
            plan,
            manifest,
            transport,
            tmp_path / f"bad-{mutation}.zst",
        )


def test_arrival_relative_writer_binds_manifest_and_transport_digests(
    tmp_path: Path,
) -> None:
    manifest = _manifest("test")
    sample = manifest.samples[-1]
    transport_digest = write_transport_overlay(
        _transport_plan(sample),
        tmp_path / "transport.zst",
    )
    transport = read_overlay(
        transport_digest.path,
        transport_digest.uncompressed_sha256,
    )
    base = ArrivalRelativeFaultPlan(
        temporal_manifest_sha256=manifest.content_sha256,
        transport_overlay_sha256=transport_digest.uncompressed_sha256,
        split="test",
        samples=(sample,),
        scope="rsu",
        modality="camera",
        fixed_delay_ms=0,
    )
    with pytest.raises(ScheduleError):
        write_arrival_relative_fault_overlay(
            replace(base, temporal_manifest_sha256=SHA_A),
            manifest,
            transport,
            tmp_path / "bad-manifest.zst",
        )
    with pytest.raises(ScheduleError):
        write_arrival_relative_fault_overlay(
            replace(base, transport_overlay_sha256=SHA_B),
            manifest,
            transport,
            tmp_path / "bad-transport.zst",
        )


@pytest.mark.parametrize("delay_ms", [0, 100, 200, 300])
def test_causal_fault_anchors_ego_and_rsu_to_their_own_endpoints(
    tmp_path: Path,
    delay_ms: int,
) -> None:
    manifest = _manifest("test")
    sample = manifest.samples[-1]
    transport_digest = write_transport_overlay(
        _transport_plan(sample, delay_ms=delay_ms),
        tmp_path / f"transport-causal-{delay_ms}.zst",
    )
    transport = read_overlay(
        transport_digest.path,
        transport_digest.uncompressed_sha256,
    )
    plan = CausalFaultPlan(
        temporal_manifest_sha256=manifest.content_sha256,
        transport_overlay_sha256=transport_digest.uncompressed_sha256,
        split="test",
        samples=(sample,),
        condition="L-Fail",
        agents=("ego", "rsu"),
        duration=1,
        fixed_delay_ms=delay_ms,  # type: ignore[arg-type]
    )
    digest = write_causal_fault_overlay(
        plan,
        manifest,
        transport,
        tmp_path / f"fault-causal-{delay_ms}.zst",
    )
    records = _records(digest.path, digest)
    masked = {
        (record["agent"], record["modality"], record["n_s"])
        for record in records
        if record["masked"]
    }
    assert masked == {
        ("ego", "lidar", 3),
        ("rsu", "lidar", 3 - delay_ms // 100),
    }
    transport_dicts = tuple(map(dict, transport))
    assert _select_latest(
        sample, "ego", "lidar", transport_dicts, records
    ) == 2
    expected_rsu_fallback = 2 - delay_ms // 100
    assert _select_latest(
        sample,
        "rsu",
        "lidar",
        transport_dicts,
        records,
    ) == (expected_rsu_fallback if expected_rsu_fallback >= 0 else None)


@pytest.mark.parametrize(
    "agents, expected_agents",
    [
        (("ego",), {"ego"}),
        (("rsu",), {"rsu"}),
        (("ego", "rsu"), {"ego", "rsu"}),
    ],
)
def test_causal_fault_supports_explicit_agent_scope(
    tmp_path: Path,
    agents: tuple[str, ...],
    expected_agents: set[str],
) -> None:
    manifest = _manifest("test")
    sample = manifest.samples[-1]
    transport_digest = write_transport_overlay(
        _transport_plan(sample, delay_ms=100),
        tmp_path / f"transport-{'-'.join(agents)}.zst",
    )
    transport = read_overlay(
        transport_digest.path,
        transport_digest.uncompressed_sha256,
    )
    digest = write_causal_fault_overlay(
        CausalFaultPlan(
            temporal_manifest_sha256=manifest.content_sha256,
            transport_overlay_sha256=transport_digest.uncompressed_sha256,
            split="test",
            samples=(sample,),
            condition="C-Fail",
            agents=agents,  # type: ignore[arg-type]
            duration=1,
            fixed_delay_ms=100,
        ),
        manifest,
        transport,
        tmp_path / f"fault-{'-'.join(agents)}.zst",
    )
    records = _records(digest.path, digest)
    assert {
        record["agent"] for record in records if record["masked"]
    } == expected_agents
    assert {
        record["modality"] for record in records if record["masked"]
    } == {"camera"}


@pytest.mark.parametrize("duration, expected_fallback", [(3, 0), (4, None)])
def test_causal_fault_duration_uses_bounded_branch_history(
    tmp_path: Path,
    duration: int,
    expected_fallback: int | None,
) -> None:
    manifest = _manifest("test")
    sample = manifest.samples[-1]
    transport_digest = write_transport_overlay(
        _transport_plan(sample),
        tmp_path / f"transport-duration-{duration}.zst",
    )
    transport = read_overlay(
        transport_digest.path,
        transport_digest.uncompressed_sha256,
    )
    digest = write_causal_fault_overlay(
        CausalFaultPlan(
            temporal_manifest_sha256=manifest.content_sha256,
            transport_overlay_sha256=transport_digest.uncompressed_sha256,
            split="test",
            samples=(sample,),
            condition="L-Fail",
            agents=("ego", "rsu"),
            duration=duration,
            fixed_delay_ms=0,
        ),
        manifest,
        transport,
        tmp_path / f"fault-duration-{duration}.zst",
    )
    records = _records(digest.path, digest)
    masked = [record for record in records if record["masked"]]
    assert len(masked) == 2 * duration
    assert {record["fallback_selected_n_s"] for record in masked} == {
        expected_fallback
    }


def test_zstd_round_trip_is_canonical_sorted_jsonl_and_digest_checked(
    tmp_path: Path,
) -> None:
    sample = _samples("test")[-1]
    digest = write_transport_overlay(
        _transport_plan(sample, delay_ms=100),
        tmp_path / "transport.zst",
    )
    compressed = digest.path.read_bytes()
    raw = zstandard.ZstdDecompressor().decompress(compressed)
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert hashlib.sha256(raw).hexdigest() == digest.uncompressed_sha256
    assert hashlib.sha256(compressed).hexdigest() == digest.compressed_sha256
    assert len(raw) == digest.uncompressed_size
    assert len(compressed) == digest.compressed_size
    lines = raw.splitlines()
    assert len(lines) == digest.record_count
    for line in lines:
        decoded = json.loads(line)
        assert line == canonical_json_bytes(decoded)
        assert tuple(decoded) == tuple(sorted(decoded))
        assert set(decoded) == {
            "arrival_tau_ms",
            "delay_ms",
            "epoch",
            "n_s",
            "packet_id",
            "sample_id",
        }
    assert len(read_overlay(digest.path, digest.uncompressed_sha256)) == len(lines)
    with pytest.raises(ScheduleError, match="digest"):
        read_overlay(digest.path, "0" * 64)


def test_overlay_publication_is_idempotent_but_never_replaces_a_conflict(
    tmp_path: Path,
) -> None:
    sample = _samples("test")[-1]
    output = tmp_path / "transport.zst"
    first = write_transport_overlay(_transport_plan(sample), output)
    original = output.read_bytes()
    second = write_transport_overlay(_transport_plan(sample), output)
    assert first == second
    assert output.read_bytes() == original
    with pytest.raises(ScheduleError, match="conflict"):
        write_transport_overlay(_transport_plan(sample, delay_ms=300), output)
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "boundary, final_is_valid",
    [
        ("_write_all", False),
        ("_fsync_file", False),
        ("_publish_no_replace", False),
        ("_fsync_directory", True),
        ("_remove_staging", True),
    ],
)
def test_crash_at_each_publication_boundary_never_exposes_truncated_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    final_is_valid: bool,
) -> None:
    import transvision.dataset.resilient_v2x_schedule as schedule

    sample = _samples("test")[-1]
    output = tmp_path / f"{boundary}.zst"
    original: Callable[..., object] = getattr(schedule, boundary)

    def crash(*args: object, **kwargs: object) -> object:
        if boundary == "_write_all":
            original(args[0], bytes(args[1])[:7])  # type: ignore[arg-type]
        raise OSError(f"injected {boundary} crash")

    monkeypatch.setattr(schedule, boundary, crash)
    with pytest.raises(ScheduleError, match="publish"):
        write_transport_overlay(_transport_plan(sample), output)
    if final_is_valid:
        assert output.exists()
        monkeypatch.setattr(schedule, boundary, original)
        digest = write_transport_overlay(_transport_plan(sample), output)
        assert read_overlay(output, digest.uncompressed_sha256)
    else:
        assert not output.exists()
    if boundary != "_remove_staging":
        assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))


def test_retry_never_removes_an_unowned_staging_file(tmp_path: Path) -> None:
    sample = _samples("test")[-1]
    output = tmp_path / "transport.zst"
    foreign = tmp_path / f".{output.name}.foreign.tmp"
    foreign.write_bytes(b"foreign")
    first = write_transport_overlay(_transport_plan(sample), output)
    second = write_transport_overlay(_transport_plan(sample), output)
    assert first == second
    assert foreign.read_bytes() == b"foreign"


def test_schedule_interfaces_are_lazily_accessible_without_changing_package_all() -> (
    None
):
    import transvision.dataset as dataset

    expected = {
        "ArrivalRelativeFaultPlan",
        "FaultOverlayRecord",
        "FaultPlan",
        "OverlayDigest",
        "ScheduleError",
        "TransportOverlayRecord",
        "TransportPlan",
        "TRAINING_CONDITION_HASH_DOMAIN",
        "TRAINING_CONDITION_MODE",
        "augmentation_seed",
        "bernoulli_from_hash",
        "delay_from_hash",
        "read_overlay",
        "stable_uint64",
        "training_condition_from_hash",
        "write_arrival_relative_fault_overlay",
        "write_fault_overlay",
        "write_transport_overlay",
    }
    assert expected.isdisjoint(dataset.__all__)
    assert all(getattr(dataset, name) is globals()[name] for name in expected)
