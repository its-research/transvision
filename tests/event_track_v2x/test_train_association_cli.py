from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import torch

from tools.event_track_v2x.train_association import main
from transvision.models.event_track_v2x.development_split import (
    build_development_split_manifest_v1,
)
from transvision.models.event_track_v2x.learning_contracts import (
    ASSOCIATION_CHECKPOINT_NAME_V1,
    ASSOCIATION_FEATURE_SCHEMA_SHA256_V1,
    verify_association_training_artifact,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _projection(root: Path, sequence_ids: tuple[str, ...]) -> None:
    vehicle_rows: list[dict[str, object]] = []
    infrastructure_rows: list[dict[str, object]] = []
    cooperative_rows: list[dict[str, object]] = []
    for index, sequence_id in enumerate(sequence_ids):
        vehicle_frame = f"{index:06d}"
        infrastructure_frame = f"{100000 + index:06d}"
        timestamp = 1_000_000 + index * 1_000_000
        for side, frame, used_timestamp, label_dir in (
            ("vehicle", vehicle_frame, timestamp, "lidar"),
            (
                "infrastructure",
                infrastructure_frame,
                timestamp + 10_000,
                "virtuallidar",
            ),
        ):
            row = {
                "frame_id": frame,
                "image_path": f"image/{frame}.jpg",
                "image_timestamp": str(used_timestamp + 1_000),
                "label_camera_std_path": f"label/camera/{frame}.json",
                "label_lidar_std_path": f"label/{label_dir}/{frame}.json",
                "pointcloud_path": f"velodyne/{frame}.pcd",
                "pointcloud_timestamp": str(used_timestamp),
                "sequence_id": sequence_id,
            }
            if side == "vehicle":
                vehicle_rows.append(row)
            else:
                infrastructure_rows.append(row)
            image_path = root / f"{side}-side/image/{frame}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (40 + index, 80, 120)).save(image_path)
            track_id = str(index + (0 if side == "vehicle" else 1000))
            token = f"{side}-{sequence_id}"
            _write_json(
                root / f"{side}-side/label/camera/{frame}.json",
                [
                    {
                        "track_id": track_id,
                        "2d_box": {"xmin": 1, "ymin": 1, "xmax": 14, "ymax": 14},
                    }
                ],
            )
            _write_json(
                root / f"{side}-side/label/{label_dir}/{frame}.json",
                [
                    {
                        "token": token,
                        "type": "Car" if side == "vehicle" else "Van",
                        "track_id": track_id,
                        "truncated_state": 0,
                        "occluded_state": 0,
                        "alpha": 0.0,
                        "2d_box": {
                            "xmin": 1,
                            "ymin": 1,
                            "xmax": 14,
                            "ymax": 14,
                        },
                        "3d_dimensions": {"l": 4.0, "w": 1.8, "h": 1.5},
                        "3d_location": {
                            "x": float(index + (0 if side == "vehicle" else 10)),
                            "y": 2.0,
                            "z": 0.5,
                        },
                        "rotation": 0.0,
                    }
                ],
            )
        cooperative_rows.append(
            {
                "vehicle_frame": vehicle_frame,
                "infrastructure_frame": infrastructure_frame,
                "vehicle_sequence": sequence_id,
                "infrastructure_sequence": sequence_id,
                "system_error_offset": {"delta_x": 0.1, "delta_y": -0.1},
            }
        )
        _write_json(
            root / f"cooperative/label/{vehicle_frame}.json",
            [
                {
                    "token": f"token-{sequence_id}",
                    "type": "Car",
                    "track_id": f"global-{sequence_id}",
                    "3d_dimensions": {"l": 4.0, "w": 1.8, "h": 1.5},
                    "3d_location": {"x": float(index), "y": 2.0, "z": 0.5},
                    "rotation": 0.0,
                    "from_side": "coop",
                    "veh_pointcloud_timestamp": str(timestamp),
                    "inf_pointcloud_timestamp": str(timestamp + 10_000),
                    "veh_frame_id": vehicle_frame,
                    "inf_frame_id": infrastructure_frame,
                    "veh_track_id": str(index),
                    "inf_track_id": str(1000 + index),
                    "veh_token": f"vehicle-{sequence_id}",
                    "inf_token": f"infrastructure-{sequence_id}",
                }
            ],
        )
    _write_json(root / "vehicle-side/data_info.json", vehicle_rows)
    _write_json(root / "infrastructure-side/data_info.json", infrastructure_rows)
    _write_json(root / "cooperative/data_info.json", cooperative_rows)


def test_cli_trains_real_spd_fold_and_seals_development_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    sequences = tuple(f"{index:04d}" for index in range(46))
    split_path = tmp_path / "split.json"
    _write_json(
        split_path,
        {
            "batch_split": {
                "train": list(sequences),
                "val": [],
                "test": [],
                "test_A": [],
            }
        },
    )
    split_raw = split_path.read_bytes()
    development = build_development_split_manifest_v1(
        sequences,
        split_sha256=hashlib.sha256(split_raw).hexdigest(),
    )
    development_path = tmp_path / "development.json"
    development_path.write_bytes(development.canonical_bytes)
    config_path = tmp_path / "association-canary.json"
    _write_json(
        config_path,
        {"epochs": 1, "frames_per_step": 16, "hidden_dim": 8},
    )
    dataset_root = tmp_path / "V2X-Seq-SPD"
    _projection(dataset_root, sequences)
    output = tmp_path / "artifact"

    assert (
        main(
            [
                "--dataset-root",
                str(dataset_root),
                "--split",
                str(split_path),
                "--development-manifest",
                str(development_path),
                "--fold-id",
                "0",
                "--seed",
                "1337",
                "--output",
                str(output),
                "--config",
                str(config_path),
                "--max-frame-pairs-per-sequence",
                "1",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    manifest = verify_association_training_artifact(output)
    assert printed["gt_supervised_development_only"] is True
    assert printed["ranking_eligible"] is False
    assert printed["formal_evidence"] is False
    assert manifest.train_sequence_count == 36
    assert manifest.validation_sequence_count == 10
    assert manifest.train_frame_count == 36
    assert manifest.validation_frame_count == 10
    checkpoint = torch.load(
        output / ASSOCIATION_CHECKPOINT_NAME_V1,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["feature_schema_sha256"] == ASSOCIATION_FEATURE_SCHEMA_SHA256_V1
    assert checkpoint["cohort_sha256"] == manifest.cohort_sha256
    assert checkpoint["gt_supervised_development_only"] is True
    assert checkpoint["ranking_eligible"] is False
    assert checkpoint["formal_evidence"] is False
