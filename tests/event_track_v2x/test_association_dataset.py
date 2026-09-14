from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transvision.dataset.event_track_v2x_association import (
    FrozenRGBProjectionAppearanceV1,
    SPDAssociationDataError,
    build_spd_association_samples_v1,
    require_spd_train_only_projection_v1,
)
from transvision.models.event_track_v2x.learning_contracts import (
    ASSOCIATION_CLASS_VOCABULARY_V1,
    ASSOCIATION_FEATURE_DIM_V1,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _side_row(side: str, frame: str, timestamp: int) -> dict[str, object]:
    lidar = "lidar" if side == "vehicle" else "virtuallidar"
    return {
        "frame_id": frame,
        "image_path": f"image/{frame}.jpg",
        "image_timestamp": str(timestamp + 1_000),
        "label_camera_std_path": f"label/camera/{frame}.json",
        "label_lidar_std_path": f"label/{lidar}/{frame}.json",
        "pointcloud_path": f"velodyne/{frame}.pcd",
        "pointcloud_timestamp": str(timestamp),
        "sequence_id": "0001",
    }


def _cooperative_label(
    *,
    frame_index: int,
    identity: str,
    vehicle_track_id: str,
    infrastructure_track_id: str,
    x: float,
) -> dict[str, object]:
    vehicle_frame = f"{frame_index + 1:06d}"
    infrastructure_frame = f"{frame_index + 100001:06d}"
    timestamp = 1_000_000 + frame_index * 100_000
    return {
        "token": f"token-{identity}",
        "type": "Car",
        "track_id": identity,
        "3d_dimensions": {"l": 4.2, "w": 1.8, "h": 1.5},
        "3d_location": {"x": x, "y": 2.0, "z": 0.5},
        "rotation": 0.2,
        "from_side": "coop",
        "veh_pointcloud_timestamp": str(timestamp),
        "inf_pointcloud_timestamp": str(timestamp + 10_000),
        "veh_frame_id": vehicle_frame,
        "inf_frame_id": infrastructure_frame,
        "veh_track_id": vehicle_track_id,
        "inf_track_id": infrastructure_track_id,
        "veh_token": (
            "-1" if vehicle_track_id == "-1" else f"vehicle-token-{identity}"
        ),
        "inf_token": (
            "-1"
            if infrastructure_track_id == "-1"
            else f"infrastructure-token-{identity}"
        ),
    }


def _camera_label(track_id: str, offset: int) -> dict[str, object]:
    return {
        "track_id": track_id,
        "2d_box": {
            "xmin": 2 + offset,
            "ymin": 3 + offset,
            "xmax": 18 + offset,
            "ymax": 22 + offset,
        },
    }


def _pointcloud_label(
    track_id: str,
    token: str,
    *,
    category: str,
    x: float,
    y: float,
    yaw: float,
) -> dict[str, object]:
    return {
        "token": token,
        "type": category,
        "track_id": track_id,
        "truncated_state": 0,
        "occluded_state": 0,
        "alpha": 0.0,
        "2d_box": {"xmin": 2.0, "ymin": 3.0, "xmax": 18.0, "ymax": 22.0},
        "3d_dimensions": {"l": 4.4, "w": 1.9, "h": 1.6},
        "3d_location": {"x": x, "y": y, "z": -0.8},
        "rotation": yaw,
    }


def _fixture(root: Path) -> Path:
    vehicle_rows = [
        _side_row("vehicle", f"{index + 1:06d}", 1_000_000 + index * 100_000)
        for index in range(2)
    ]
    infrastructure_rows = [
        _side_row(
            "infrastructure",
            f"{index + 100001:06d}",
            1_010_000 + index * 100_000,
        )
        for index in range(2)
    ]
    cooperative_rows = [
        {
            "vehicle_frame": vehicle_rows[index]["frame_id"],
            "infrastructure_frame": infrastructure_rows[index]["frame_id"],
            "vehicle_sequence": "0001",
            "infrastructure_sequence": "0001",
            "system_error_offset": {"delta_x": 0.25, "delta_y": -0.5},
        }
        for index in range(2)
    ]
    _write_json(root / "vehicle-side/data_info.json", vehicle_rows)
    _write_json(root / "infrastructure-side/data_info.json", infrastructure_rows)
    _write_json(root / "cooperative/data_info.json", cooperative_rows)
    for index, (vehicle, infrastructure) in enumerate(
        zip(vehicle_rows, infrastructure_rows)
    ):
        vehicle_frame = str(vehicle["frame_id"])
        infrastructure_frame = str(infrastructure["frame_id"])
        for side, frame, colour in (
            ("vehicle", vehicle_frame, (180, 50, 20)),
            ("infrastructure", infrastructure_frame, (20, 80, 190)),
        ):
            image_path = root / f"{side}-side/image/{frame}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), colour).save(image_path)
        if index == 0:
            cooperative_labels = [
                _cooperative_label(
                    frame_index=index,
                    identity="1",
                    vehicle_track_id="10",
                    infrastructure_track_id="20",
                    x=90.0,
                ),
                _cooperative_label(
                    frame_index=index,
                    identity="2",
                    vehicle_track_id="11",
                    infrastructure_track_id="-1",
                    x=91.0,
                ),
                _cooperative_label(
                    frame_index=index,
                    identity="3",
                    vehicle_track_id="-1",
                    infrastructure_track_id="21",
                    x=92.0,
                ),
            ]
            vehicle_camera = [_camera_label("10", 0), _camera_label("11", 4)]
            infrastructure_camera = [_camera_label("20", 0), _camera_label("21", 4)]
            vehicle_pointcloud = [
                _pointcloud_label(
                    "10",
                    "vehicle-token-1",
                    category="Car",
                    x=10.0,
                    y=1.0,
                    yaw=0.1,
                ),
                _pointcloud_label(
                    "11",
                    "vehicle-token-2",
                    category="Truck",
                    x=20.0,
                    y=2.0,
                    yaw=0.2,
                ),
            ]
            infrastructure_pointcloud = [
                _pointcloud_label(
                    "20",
                    "infrastructure-token-1",
                    category="Van",
                    x=30.0,
                    y=-2.0,
                    yaw=-0.4,
                ),
                _pointcloud_label(
                    "21",
                    "infrastructure-token-3",
                    category="Bus",
                    x=40.0,
                    y=-3.0,
                    yaw=-0.5,
                ),
            ]
        else:
            cooperative_labels = [
                _cooperative_label(
                    frame_index=index,
                    identity="1",
                    vehicle_track_id="10",
                    infrastructure_track_id="20",
                    x=99.0,
                )
            ]
            vehicle_camera = [_camera_label("10", 0)]
            infrastructure_camera = [_camera_label("20", 0)]
            vehicle_pointcloud = [
                _pointcloud_label(
                    "10",
                    "vehicle-token-1",
                    category="Car",
                    x=11.0,
                    y=1.0,
                    yaw=0.1,
                )
            ]
            infrastructure_pointcloud = [
                _pointcloud_label(
                    "20",
                    "infrastructure-token-1",
                    category="Van",
                    x=32.0,
                    y=-2.0,
                    yaw=-0.4,
                )
            ]
        _write_json(
            root / f"cooperative/label/{vehicle_frame}.json",
            cooperative_labels,
        )
        _write_json(
            root / f"vehicle-side/label/camera/{vehicle_frame}.json",
            vehicle_camera,
        )
        _write_json(
            root / f"infrastructure-side/label/camera/{infrastructure_frame}.json",
            infrastructure_camera,
        )
        _write_json(
            root / f"vehicle-side/label/lidar/{vehicle_frame}.json",
            vehicle_pointcloud,
        )
        _write_json(
            root
            / f"infrastructure-side/label/virtuallidar/{infrastructure_frame}.json",
            infrastructure_pointcloud,
        )
    return root


def test_real_spd_pairs_create_matches_dustbins_and_all_feature_groups(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")

    samples = build_spd_association_samples_v1(
        root,
        official_train_sequence_ids=("0001",),
        selected_sequence_ids=("0001",),
        appearance_provider=FrozenRGBProjectionAppearanceV1(),
    )

    assert len(samples) == 2
    first = samples[0]
    assert first.left_features.shape == (2, ASSOCIATION_FEATURE_DIM_V1)
    assert first.right_features.shape == (2, ASSOCIATION_FEATURE_DIM_V1)
    np.testing.assert_array_equal(first.targets, [[1, 0], [0, 0]])
    assert first.targets.sum(axis=0).tolist() == [1, 0]
    assert first.targets.sum(axis=1).tolist() == [1, 0]
    assert np.linalg.norm(first.left_features[0, 10:138]) == pytest.approx(1.0)
    assert first.left_features[0, 0] == pytest.approx(10.0 / 100.0)
    assert first.right_features[0, 0] == pytest.approx(30.0 / 100.0)
    assert first.left_features[0, 0] != pytest.approx(90.0 / 100.0)
    van_index = ASSOCIATION_CLASS_VOCABULARY_V1.index("Van")
    assert first.right_features[0, 138 + van_index] == 1.0
    assert first.right_features[0, 199] != 0.0  # infrastructure pose x
    assert first.left_features[0, -4] == 1.0  # complete lineage
    second = samples[1]
    assert second.left_features[0, 8] == pytest.approx(10.0 / 30.0)
    assert second.right_features[0, 8] == pytest.approx(20.0 / 30.0)
    assert second.content_sha256 == second.content_sha256
    assert not second.left_features.flags.writeable
    assert not second.targets.flags.writeable


def test_deterministic_frame_cap_is_train_derived_not_synthetic(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")
    provider = FrozenRGBProjectionAppearanceV1()

    samples = build_spd_association_samples_v1(
        root,
        official_train_sequence_ids=("0001",),
        selected_sequence_ids=("0001",),
        appearance_provider=provider,
        max_frame_pairs_per_sequence=1,
    )

    assert len(samples) == 1
    assert samples[0].vehicle_frame_id == "000001"
    assert samples[0].left_identity_ids == (
        "0001:vehicle:10",
        "0001:vehicle:11",
    )


def test_projection_rejects_any_sealed_sequence_metadata(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")
    path = root / "cooperative/data_info.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document[0]["vehicle_sequence"] = "0002"
    document[0]["infrastructure_sequence"] = "0002"
    _write_json(path, document)

    with pytest.raises(SPDAssociationDataError, match="sealed sequence 0002"):
        require_spd_train_only_projection_v1(
            root,
            official_train_sequence_ids=("0001",),
        )


def test_missing_referenced_pointcloud_label_fails_without_cooperative_fallback(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")
    path = root / "vehicle-side/label/lidar/000001.json"
    labels = json.loads(path.read_text(encoding="utf-8"))
    _write_json(path, [label for label in labels if label["track_id"] != "10"])

    with pytest.raises(
        SPDAssociationDataError,
        match="vehicle cooperative track_id 10 is absent from pointcloud labels",
    ):
        build_spd_association_samples_v1(
            root,
            official_train_sequence_ids=("0001",),
            selected_sequence_ids=("0001",),
            appearance_provider=FrozenRGBProjectionAppearanceV1(),
        )


def test_duplicate_pointcloud_track_id_fails(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")
    path = root / "vehicle-side/label/lidar/000001.json"
    labels = json.loads(path.read_text(encoding="utf-8"))
    labels.append(dict(labels[0]))
    _write_json(path, labels)

    with pytest.raises(
        SPDAssociationDataError,
        match="vehicle pointcloud labels contain duplicate track_id 10",
    ):
        build_spd_association_samples_v1(
            root,
            official_train_sequence_ids=("0001",),
            selected_sequence_ids=("0001",),
            appearance_provider=FrozenRGBProjectionAppearanceV1(),
        )


def test_cooperative_token_mismatch_fails(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "V2X-Seq-SPD")
    path = root / "vehicle-side/label/lidar/000001.json"
    labels = json.loads(path.read_text(encoding="utf-8"))
    labels[0]["token"] = "wrong-vehicle-token"
    _write_json(path, labels)

    with pytest.raises(
        SPDAssociationDataError,
        match="vehicle cooperative/pointcloud token mismatch for track_id 10",
    ):
        build_spd_association_samples_v1(
            root,
            official_train_sequence_ids=("0001",),
            selected_sequence_ids=("0001",),
            appearance_provider=FrozenRGBProjectionAppearanceV1(),
        )
