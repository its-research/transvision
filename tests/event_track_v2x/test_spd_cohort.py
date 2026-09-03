from dataclasses import replace

import pytest

from transvision.dataset.event_track_v2x_cohort import (
    SPDCohortError,
    build_spd_cohort_manifest,
    verify_spd_cohort_manifest,
)
from transvision.dataset.event_track_v2x_spd import (
    SPDMetadata,
    SPDPair,
    SPDSideFrame,
)


def pair(index: int, *, sequence: str) -> SPDPair:
    frame = f"{index:06d}"
    vehicle = SPDSideFrame(
        side="vehicle",
        frame_id=frame,
        sequence_id=sequence,
        image_timestamp_us=1_000_000 + index * 100_000,
        pointcloud_timestamp_us=1_000_000 + index * 100_000,
        image_relative_path=f"vehicle-side/image/{frame}.jpg",
        pointcloud_relative_path=f"vehicle-side/velodyne/{frame}.pcd",
        label_relative_path=f"vehicle-side/label/{frame}.json",
    )
    infrastructure = replace(
        vehicle,
        side="infrastructure",
        frame_id=f"{index + 100:06d}",
        image_relative_path=f"infrastructure-side/image/{index + 100:06d}.jpg",
        pointcloud_relative_path=f"infrastructure-side/velodyne/{index + 100:06d}.pcd",
        label_relative_path=f"infrastructure-side/label/{index + 100:06d}.json",
    )
    return SPDPair(
        sequence_id=sequence,
        vehicle_sequence_id=sequence,
        infrastructure_sequence_id=sequence,
        vehicle=vehicle,
        infrastructure=infrastructure,
        system_error_offset_xy=(0.0, 0.0),
        labels=(),
    )


def metadata() -> SPDMetadata:
    pairs = tuple(
        [pair(index, sequence="0001") for index in range(10)]
        + [pair(index + 20, sequence="0002") for index in range(10)]
    )
    return SPDMetadata(
        nominal_rate_hz=10,
        sequence_ids=("0001", "0002"),
        vehicle_frame_count=20,
        infrastructure_frame_count=20,
        cooperative_pair_count=20,
        selected_pair_count=20,
        selected_label_count=0,
        pairs=pairs,
    )


def test_frozen_2hz_cohort_samples_each_sequence_without_rewriting_labels() -> None:
    source = metadata()
    split_ids = [item.vehicle.frame_id for item in source.pairs]
    manifest = build_spd_cohort_manifest(
        source,
        split_name="val",
        split_frame_ids=split_ids,
        split_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        target_rate_hz=2,
        phase=1,
    )
    assert manifest["pair_count"] == 4
    assert [item["vehicle_frame_id"] for item in manifest["pairs"]] == [
        "000001",
        "000006",
        "000021",
        "000026",
    ]
    assert {item["label_count"] for item in manifest["pairs"]} == {0}
    assert verify_spd_cohort_manifest(manifest) == manifest["content_sha256"]


def test_native_cohort_keeps_every_split_pair_and_seal_detects_tampering() -> None:
    source = metadata()
    split_ids = [item.vehicle.frame_id for item in source.pairs]
    manifest = build_spd_cohort_manifest(
        source,
        split_name="val",
        split_frame_ids=split_ids,
        split_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        target_rate_hz=10,
    )
    assert manifest["pair_count"] == 20
    manifest["pair_count"] = 19
    with pytest.raises(SPDCohortError, match="SHA-256 mismatch"):
        verify_spd_cohort_manifest(manifest)


def test_cohort_rejects_missing_split_frame_and_invalid_phase() -> None:
    source = metadata()
    with pytest.raises(SPDCohortError, match="absent"):
        build_spd_cohort_manifest(
            source,
            split_name="val",
            split_frame_ids=("999999",),
            split_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            target_rate_hz=2,
        )
    with pytest.raises(SPDCohortError, match="phase"):
        build_spd_cohort_manifest(
            source,
            split_name="val",
            split_frame_ids=(source.pairs[0].vehicle.frame_id,),
            split_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            target_rate_hz=2,
            phase=5,
        )
