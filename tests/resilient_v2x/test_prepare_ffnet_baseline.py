from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/resilient_v2x/prepare_ffnet_baseline.py"


def _module():
    spec = importlib.util.spec_from_file_location("prepare_ffnet_baseline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(module, *, content_sha256=None):
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "content_sha256": content_sha256 or module.EXPECTED_MANIFEST_CONTENT_SHA256,
        "prepared_artifacts": [
            {
                "source_relative_path": "vehicle-side/velodyne/000001.pcd",
                "prepared_relative_path": (
                    "prepared/resilient_v2x_v2/vehicle-side/velodyne/000001.bin"
                ),
                "size": 16,
            },
            {
                "source_relative_path": "infrastructure-side/velodyne/000002.pcd",
                "prepared_relative_path": (
                    "prepared/resilient_v2x_v2/infrastructure-side/velodyne/000002.bin"
                ),
                "size": 16,
            },
        ],
        "samples": [
            {
                "sample_id": "dairc-v000001-i000002",
                "split": "train",
                "n_t": 3,
                "source_slices": [
                    {
                        "agent": "ego",
                        "modality": "lidar",
                        "n_s": 3,
                        "frame_id": "000001",
                        "relative_path": "vehicle-side/velodyne/000001.pcd",
                        "world_from_agent": identity,
                        "agent_from_sensor": identity,
                    },
                    {
                        "agent": "rsu",
                        "modality": "lidar",
                        "n_s": 3,
                        "frame_id": "000002",
                        "relative_path": "infrastructure-side/velodyne/000002.pcd",
                        "world_from_agent": [
                            [0.0, -1.0, 0.0, 4.0],
                            [1.0, 0.0, 0.0, -3.0],
                            [0.0, 0.0, 1.0, 0.5],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "agent_from_sensor": identity,
                    },
                ],
                "ground_truth": [
                    {
                        "class_name": "Car",
                        "x": 10.0,
                        "y": -2.0,
                        "z_bottom": -1.5,
                        "length": 4.0,
                        "width": 1.8,
                        "height": 1.6,
                        "yaw": 0.25,
                    }
                ],
            }
        ],
    }


def test_build_ffnet_infos_uses_v017_lidar_box_numbers(tmp_path: Path):
    module = _module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(module)), encoding="utf-8")
    output = tmp_path / "infos"

    contract = module.build_ffnet_infos(manifest, output)

    with (output / "dair_infos_train.pkl").open("rb") as stream:
        info = pickle.load(stream)
    sample = info["data_list"][0]
    assert sample["sample_id"] == "dairc-v000001-i000002"
    assert sample["instances"][0]["bbox_3d"] == [
        10.0,
        -2.0,
        -1.5,
        1.8,
        4.0,
        1.6,
        -0.25,
    ]
    assert sample["instances"][0]["bbox_label_3d"] == 2
    assert info["metainfo"]["categories"]["Car"] == 2
    assert sample["lidar_points"]["lidar_path"].endswith("000001.bin")
    assert sample["lidar_points"]["inf_lidar_path"].endswith("000002.bin")
    assert sample["calib"]["lidar_i2v"] == {
        "rotation": [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "translation": [[4.0], [-3.0], [0.5]],
    }
    assert contract["split_counts"] == {"train": 1, "val": 0}
    assert contract["coordinate_convention"].endswith(
        "[x,y,z,width,length,height,-yaw]"
    )
    assert contract["fusion_calibration"].startswith("ego_lidar_from_rsu_lidar")


def test_build_ffnet_infos_rejects_wrong_manifest(tmp_path: Path):
    module = _module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(_manifest(module, content_sha256="0" * 64)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected temporal manifest"):
        module.build_ffnet_infos(manifest, tmp_path / "infos")
