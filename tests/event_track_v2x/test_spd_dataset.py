from __future__ import annotations

import json
from pathlib import Path

import pytest

from transvision.dataset.event_track_v2x_spd import (
    SPDDataError,
    load_spd_metadata,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _side_record(side: str, frame: int, timestamp: int) -> dict[str, object]:
    frame_id = f"{frame:06d}"
    label_dir = "lidar" if side == "vehicle" else "virtuallidar"
    return {
        "image_path": f"image/{frame_id}.jpg",
        "pointcloud_path": f"velodyne/{frame_id}.pcd",
        "label_lidar_std_path": f"label/{label_dir}/{frame_id}.json",
        "image_timestamp": str(timestamp + 1_000),
        "pointcloud_timestamp": str(timestamp),
        "frame_id": frame_id,
        "sequence_id": "0087",
    }


def _label(vehicle_frame: int, infrastructure_frame: int, timestamp: int) -> dict[str, object]:
    return {
        "token": "persistent-token",
        "type": "Car",
        "track_id": "000123",
        "3d_dimensions": {"l": 4.2, "w": 1.9, "h": 1.5},
        "3d_location": {"x": 10.0, "y": -2.0, "z": 0.5},
        "rotation": 0.2,
        "from_side": "coop",
        "veh_pointcloud_timestamp": str(timestamp),
        "inf_pointcloud_timestamp": str(timestamp + 10_000),
        "veh_frame_id": f"{vehicle_frame:06d}",
        "inf_frame_id": f"{infrastructure_frame:06d}",
        "veh_track_id": "001234",
        "inf_track_id": "005678",
        "veh_token": "vehicle-token",
        "inf_token": "infrastructure-token",
    }


def _fixture(root: Path) -> Path:
    vehicle = [
        _side_record("vehicle", 15_888 + index, 1_000_000 + index * 100_000)
        for index in range(3)
    ]
    infrastructure = [
        _side_record(
            "infrastructure", 14_996 + index, 1_010_000 + index * 100_000
        )
        for index in range(3)
    ]
    cooperative = []
    for index in range(3):
        vehicle_frame = 15_888 + index
        infrastructure_frame = 14_996 + index
        cooperative.append(
            {
                "vehicle_frame": f"{vehicle_frame:06d}",
                "infrastructure_frame": f"{infrastructure_frame:06d}",
                "vehicle_sequence": "0087",
                "infrastructure_sequence": "0087",
                "system_error_offset": {"delta_x": 0.1, "delta_y": -0.2},
            }
        )
        _write_json(
            root / f"cooperative/label/{vehicle_frame:06d}.json",
            [_label(vehicle_frame, infrastructure_frame, 1_000_000 + index * 100_000)],
        )
    _write_json(root / "vehicle-side/data_info.json", vehicle)
    _write_json(root / "infrastructure-side/data_info.json", infrastructure)
    _write_json(root / "cooperative/data_info.json", cooperative)
    return root


def test_reader_preserves_tracking_and_timestamp_identity(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")

    metadata = load_spd_metadata(root, sequence_ids=["0087"])

    assert metadata.nominal_rate_hz == 10
    assert metadata.sequence_ids == ("0087",)
    assert metadata.selected_pair_count == 3
    assert metadata.selected_label_count == 3
    pair = metadata.pairs[0]
    label = pair.labels[0]
    assert pair.vehicle.frame_id == "015888"
    assert pair.infrastructure.frame_id == "014996"
    assert label.sequence_id == "0087"
    assert label.track_id == "000123"
    assert label.token == "persistent-token"
    assert label.vehicle_track_id == "001234"
    assert label.infrastructure_track_id == "005678"
    assert label.vehicle_pointcloud_timestamp_us == 1_000_000
    assert label.infrastructure_pointcloud_timestamp_us == 1_010_000
    assert label.from_side == "coop"
    summary = metadata.canary_summary()
    assert summary["evidence_scope"] == "metadata-and-ground-truth-canary-only"
    assert summary["model_inference_performed"] is False
    assert summary["model_performance_measured"] is False


def test_reader_rejects_nonmonotonic_10hz_sequence(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")
    path = root / "vehicle-side/data_info.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    values[1]["pointcloud_timestamp"] = "999999"
    _write_json(path, values)

    with pytest.raises(SPDDataError, match="timestamps are not increasing"):
        load_spd_metadata(root)


def test_reader_rejects_pair_join_drift(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")
    path = root / "cooperative/data_info.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    values[0]["vehicle_sequence"] = "0086"
    _write_json(path, values)

    with pytest.raises(SPDDataError, match="joins different sequence IDs"):
        load_spd_metadata(root)


def test_reader_rejects_missing_cooperative_label(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")
    (root / "cooperative/label/015889.json").unlink()

    with pytest.raises(SPDDataError, match="required JSON file is missing"):
        load_spd_metadata(root)


def test_reader_rejects_label_timestamp_drift(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")
    path = root / "cooperative/label/015888.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    values[0]["veh_pointcloud_timestamp"] = "1000001"
    _write_json(path, values)

    with pytest.raises(SPDDataError, match="disagrees with vehicle data_info"):
        load_spd_metadata(root)
