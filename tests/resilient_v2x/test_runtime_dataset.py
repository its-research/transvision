from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
import torch

from tools.resilient_v2x.prepare_data import prepare_manifest
from transvision.dataset.resilient_v2x_runtime import (
    EpochIndexSampler,
    ResilientTemporalDataset,
    RuntimeOverlayIndex,
    RuntimeProtocolError,
    collate_resilient_samples,
    resolve_temporal_sample,
)


FIXTURE = Path(__file__).parent / "fixtures/dair-mini"


@pytest.fixture()
def prepared_fixture(tmp_path: Path):
    root = tmp_path / "dair-mini"
    shutil.copytree(FIXTURE, root)
    split_path = root / "split.json"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare_manifest(
        data_root=root,
        split_path=split_path,
        output_path=manifest_path,
        expected_split_sha256=split_hash,
        protocol_scope="fixture",
        delta_t_ms=100,
        history_limit=3,
        interval_min_ms=50,
        interval_max_ms=150,
        max_capture_skew_ms=50,
    )
    return root, manifest_path, manifest, split_hash


def _transport(sample, delay_ms: int):
    return [
        {
            "epoch": None,
            "sample_id": sample.sample_id,
            "packet_id": source.packet_id,
            "n_s": source.n_s,
            "delay_ms": delay_ms,
            "arrival_tau_ms": source.tau_s_ms + delay_ms,
        }
        for source in sample.source_slices
        if source.agent == "rsu"
    ]


def test_runtime_resolution_obeys_arrival_and_fault_fallback(
    prepared_fixture,
) -> None:
    _, _, manifest, _ = prepared_fixture
    sample = manifest.samples[-1]
    fault = {
        "epoch": None,
        "sample_id": sample.sample_id,
        "agent": "ego",
        "modality": "lidar",
        "n_s": sample.n_t,
        "masked": True,
        "pre_mask_selected_n_s": sample.n_t,
        "fallback_selected_n_s": sample.n_t - 1,
    }
    overlays = RuntimeOverlayIndex.build(
        manifest,
        transport_records=_transport(sample, 200),
        fault_records=(fault,),
    )

    resolved = resolve_temporal_sample(
        sample,
        delta_t_ms=manifest.delta_t_ms,
        history_limit=manifest.history_limit,
        overlays=overlays,
        epoch=0,
        global_seed=7,
        zero_latency=False,
    )

    lidar_ego, lidar_rsu, camera_ego, camera_rsu = resolved.branches
    assert lidar_ego.selection.horizon == 1
    assert lidar_ego.selection.endpoint_tick == sample.n_t
    assert lidar_ego.selection.propagated
    assert not lidar_ego.slots[0].available
    assert lidar_rsu.selection.horizon == 2
    assert lidar_rsu.selection.endpoint_tick == sample.n_t - 2
    assert lidar_rsu.selection.observed
    assert not lidar_rsu.slots[0].available
    assert not lidar_rsu.slots[1].available
    assert lidar_rsu.slots[2].available
    assert camera_ego.selection.horizon == 0
    assert camera_rsu.selection.horizon == 2
    assert camera_rsu.selection.endpoint_tick == sample.n_t - 2
    assert camera_rsu.selection.observed
    assert resolved.sample.ground_truth == sample.ground_truth


def test_rsu_transport_age_is_independent_of_sensor_fallback(
    prepared_fixture,
) -> None:
    root, manifest_path, manifest, split_hash = prepared_fixture
    sample = manifest.samples[-1]
    fault = {
        "epoch": None,
        "sample_id": sample.sample_id,
        "agent": "rsu",
        "modality": "lidar",
        "n_s": sample.n_t,
        "masked": True,
        "pre_mask_selected_n_s": sample.n_t,
        "fallback_selected_n_s": sample.n_t - 1,
    }
    transport_records = _transport(sample, 0)
    overlays = RuntimeOverlayIndex.build(
        manifest,
        transport_records=transport_records,
        fault_records=(fault,),
    )
    resolved = resolve_temporal_sample(
        sample,
        delta_t_ms=manifest.delta_t_ms,
        history_limit=manifest.history_limit,
        overlays=overlays,
        epoch=0,
        global_seed=3,
        zero_latency=False,
    )

    lidar_rsu = resolved.branches[1].selection
    camera_rsu = resolved.branches[3].selection
    assert lidar_rsu.endpoint_tick == sample.n_t
    assert lidar_rsu.horizon == 1
    assert lidar_rsu.propagated
    assert camera_rsu.endpoint_tick == sample.n_t
    assert camera_rsu.horizon == 0
    assert camera_rsu.observed

    dataset = ResilientTemporalDataset(
        manifest_path=manifest_path,
        data_root=root,
        split="train",
        expected_split_hash=split_hash,
        allow_fixture=True,
        load_camera=False,
    )
    materialized = dict(dataset[len(dataset) - 1])
    materialized["resolved"] = resolved
    batch = collate_resilient_samples((materialized,))
    assert batch["selections"].rsu_delay_intervals == (0.0,)


def test_early_tick_is_left_empty_without_current_frame_repetition(
    prepared_fixture,
) -> None:
    _, _, manifest, _ = prepared_fixture
    sample = manifest.samples[0]
    overlays = RuntimeOverlayIndex.build(manifest)

    resolved = resolve_temporal_sample(
        sample,
        delta_t_ms=manifest.delta_t_ms,
        history_limit=manifest.history_limit,
        overlays=overlays,
        epoch=0,
        global_seed=0,
        zero_latency=True,
    )

    for branch in resolved.branches:
        assert branch.slots[0].available
        assert all(slot.source is None for slot in branch.slots[1:])
        assert all(not slot.available for slot in branch.slots[1:])


def test_runtime_rejects_cross_modality_rsu_delay_mismatch(
    prepared_fixture,
) -> None:
    _, _, manifest, _ = prepared_fixture
    sample = manifest.samples[0]
    records = _transport(sample, 100)
    records[1] = dict(records[1], delay_ms=200, arrival_tau_ms=200)

    with pytest.raises(RuntimeProtocolError, match="share delay"):
        RuntimeOverlayIndex.build(manifest, transport_records=records)


def test_dataset_loads_only_sparse_available_lidar_and_collates(
    prepared_fixture,
) -> None:
    root, manifest_path, _, split_hash = prepared_fixture
    dataset = ResilientTemporalDataset(
        manifest_path=manifest_path,
        data_root=root,
        split="train",
        expected_split_hash=split_hash,
        allow_fixture=True,
        load_camera=False,
        seed=11,
    )

    first = dataset[0]
    second = dataset[1]
    assert len(first["lidar_points"]) == 2
    assert first["lidar_owner"].tolist() == [[0, 0], [1, 0]]
    assert len(first["camera_images"]) == 0
    assert all(points.dtype == torch.float32 for points in first["lidar_points"])
    expected_box = first["resolved"].sample.ground_truth[0]
    assert first["gt_bboxes_3d"][0].tolist() == pytest.approx(
        [
            expected_box.x,
            expected_box.y,
            expected_box.z_bottom,
            expected_box.length,
            expected_box.width,
            expected_box.height,
            expected_box.yaw,
        ]
    )

    batch = collate_resilient_samples((first, second))
    assert batch["availability"].shape == (2, 2, 2, 4)
    assert batch["source_to_target"].shape == (2, 2, 2, 4, 4, 4)
    assert batch["lidar_owner"].shape == (6, 3)
    assert batch["camera_images"].shape == (0, 3, 1, 1)
    assert batch["selections"].sample_ids == (
        first["resolved"].sample.sample_id,
        second["resolved"].sample.sample_id,
    )


def test_dataset_fails_closed_when_prepared_payload_changes(
    prepared_fixture,
) -> None:
    root, manifest_path, manifest, split_hash = prepared_fixture
    artifact = manifest.prepared_artifacts[0]
    path = root / artifact.prepared_relative_path
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(bytes(raw))
    dataset = ResilientTemporalDataset(
        manifest_path=manifest_path,
        data_root=root,
        split="train",
        expected_split_hash=split_hash,
        allow_fixture=True,
        load_camera=False,
    )

    with pytest.raises(RuntimeProtocolError, match="hash"):
        dataset[0]


def test_epoch_sampler_carries_epoch_into_worker_index() -> None:
    dataset = list(range(5))
    sampler = EpochIndexSampler(
        dataset,
        shuffle=True,
        seed=17,
        rank=1,
        world_size=2,
    )
    sampler.set_epoch(4)
    first = list(sampler)
    second = list(sampler)

    assert first == second
    assert len(first) == 3
    assert all(epoch == 4 for epoch, _ in first)
