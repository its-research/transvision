#!/usr/bin/env python3
"""Generate FFNet MMDetection3D infos from the sealed temporal manifest."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


EXPECTED_MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
SPLITS = ("train", "val")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    return parser


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _string(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _rigid_matrix(value: object, context: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{context} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{context} must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{context} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{context} rotation must have determinant one")
    return matrix


def _sensor_pose(source_slice: Mapping[str, object]) -> np.ndarray:
    context = (
        f"{source_slice.get('agent')} {source_slice.get('modality')} source slice"
    )
    world_from_agent = _rigid_matrix(
        source_slice.get("world_from_agent"),
        f"{context} world_from_agent",
    )
    agent_from_sensor = _rigid_matrix(
        source_slice.get("agent_from_sensor"),
        f"{context} agent_from_sensor",
    )
    return world_from_agent @ agent_from_sensor


def _target_lidar_slice(
    sample: Mapping[str, object],
    *,
    agent: str,
) -> Mapping[str, object]:
    n_t = sample.get("n_t")
    matches = [
        _mapping(item, "source slice")
        for item in _array(sample.get("source_slices"), "sample source_slices")
        if _mapping(item, "source slice").get("agent") == agent
        and _mapping(item, "source slice").get("modality") == "lidar"
        and _mapping(item, "source slice").get("n_s") == n_t
    ]
    if len(matches) != 1:
        raise ValueError(
            f"sample {_string(sample.get('sample_id'), 'sample_id')} requires "
            f"exactly one current {agent} LiDAR slice, found {len(matches)}"
        )
    return matches[0]


def _instance(box: Mapping[str, object], index: int) -> dict[str, object]:
    if box.get("class_name") != "Car":
        raise ValueError("FFNet vehicle baseline accepts Car ground truth only")
    source_coordinates = [
        box.get("x"),
        box.get("y"),
        box.get("z_bottom"),
        box.get("width"),
        box.get("length"),
        box.get("height"),
        box.get("yaw"),
    ]
    if any(type(value) not in (int, float) for value in source_coordinates):
        raise ValueError("ground-truth boxes must contain seven numeric values")
    coordinates = [float(value) for value in source_coordinates]
    coordinates[-1] = -coordinates[-1]
    return {
        # MMDetection3D v0.17 LiDAR boxes use width, length and clockwise yaw.
        "bbox_3d": coordinates,
        # V2XDataset's fixed source taxonomy is Pedestrian/Cyclist/Car.
        # Its metainfo mapping converts source class 2 into train class 0.
        "bbox_label_3d": 2,
        "bbox_label": 2,
        "attr_label": 2,
        "bbox": [0.0, 0.0, 100.0, 100.0],
        "alpha": -10.0,
        "truncated": 0,
        "occluded": 0,
        "score": 0.0,
        "index": index,
        "group_id": index,
        "difficulty": 0,
        "num_lidar_pts": -1,
    }


def _info(
    sample: Mapping[str, object],
    *,
    sample_idx: int,
    prepared_by_source: Mapping[str, Mapping[str, object]],
    data_root: Path | None,
) -> dict[str, object]:
    sample_id = _string(sample.get("sample_id"), "sample_id")
    ego = _target_lidar_slice(sample, agent="ego")
    rsu = _target_lidar_slice(sample, agent="rsu")

    world_from_ego_lidar = _sensor_pose(ego)
    world_from_rsu_lidar = _sensor_pose(rsu)
    try:
        ego_lidar_from_rsu_lidar = (
            np.linalg.inv(world_from_ego_lidar) @ world_from_rsu_lidar
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(f"sample {sample_id} has singular LiDAR pose") from error
    ego_lidar_from_rsu_lidar = _rigid_matrix(
        ego_lidar_from_rsu_lidar,
        f"sample {sample_id} ego_lidar_from_rsu_lidar",
    )

    def prepared_path(source_slice: Mapping[str, object]) -> str:
        source = _string(source_slice.get("relative_path"), "LiDAR source path")
        try:
            artifact = prepared_by_source[source]
        except KeyError as error:
            raise ValueError(f"prepared LiDAR is missing for {source}") from error
        relative = _string(
            artifact.get("prepared_relative_path"),
            "prepared LiDAR path",
        )
        expected_size = artifact.get("size")
        if type(expected_size) is not int or expected_size <= 0:
            raise ValueError(f"prepared LiDAR size is invalid: {relative}")
        if data_root is not None:
            payload = data_root / relative
            if payload.is_symlink() or not payload.is_file():
                raise FileNotFoundError(f"prepared LiDAR is missing: {payload}")
            if payload.stat().st_size != expected_size:
                raise ValueError(f"prepared LiDAR size mismatch: {payload}")
        return relative

    ground_truth = [
        _mapping(item, "ground truth")
        for item in _array(sample.get("ground_truth"), "sample ground_truth")
    ]
    if not ground_truth:
        raise ValueError(f"FFNet sample has no Car ground truth: {sample_id}")

    return {
        "sample_idx": sample_idx,
        "sample_id": sample_id,
        "veh_sample_idx": _string(ego.get("frame_id"), "ego frame_id"),
        "inf_sample_idx": _string(rsu.get("frame_id"), "rsu frame_id"),
        "lidar_points": {
            "num_pts_feats": 4,
            "lidar_path": prepared_path(ego),
            "inf_lidar_path": prepared_path(rsu),
        },
        "calib": {
            "lidar_i2v": {
                "rotation": ego_lidar_from_rsu_lidar[:3, :3].tolist(),
                "translation": ego_lidar_from_rsu_lidar[:3, 3:4].tolist(),
            }
        },
        "instances": [
            _instance(box, index) for index, box in enumerate(ground_truth)
        ],
        "cam_instances": {},
        "v2x_info": None,
    }


def build_ffnet_infos(
    manifest_path: Path,
    output_dir: Path,
    *,
    data_root: Path | None = None,
) -> dict[str, object]:
    manifest_path = manifest_path.resolve(strict=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse FFNet info directory: {output_dir}")
    if data_root is not None:
        data_root = data_root.resolve(strict=True)

    manifest = _mapping(json.loads(manifest_path.read_bytes()), "manifest")
    content_sha256 = manifest.get("content_sha256")
    if content_sha256 != EXPECTED_MANIFEST_CONTENT_SHA256:
        raise ValueError(
            "FFNet info generation received an unexpected temporal manifest: "
            f"{content_sha256!r}"
        )

    prepared = [
        _mapping(item, "prepared artifact")
        for item in _array(
            manifest.get("prepared_artifacts"),
            "manifest prepared_artifacts",
        )
    ]
    prepared_by_source = {
        _string(item.get("source_relative_path"), "prepared source path"): item
        for item in prepared
    }
    if len(prepared_by_source) != len(prepared):
        raise ValueError("manifest contains duplicate prepared source paths")

    samples = [
        _mapping(item, "sample")
        for item in _array(manifest.get("samples"), "manifest samples")
    ]
    metainfo = {
        "dataset": "dair_v2x_complemented_manifest",
        "info_version": "1.1",
        "classes": ["Pedestrian", "Cyclist", "Car"],
        "categories": {"Pedestrian": 0, "Cyclist": 1, "Car": 2},
        "coordinate_convention": (
            "MMDetection3D v0.17 LiDAR bottom center "
            "[x,y,z,width,length,height,-yaw]"
        ),
        "temporal_manifest_sha256": content_sha256,
    }
    output_dir.mkdir(parents=True)
    split_counts: dict[str, int] = {}
    for split in SPLITS:
        split_samples = [item for item in samples if item.get("split") == split]
        data_list = [
            _info(
                sample,
                sample_idx=index,
                prepared_by_source=prepared_by_source,
                data_root=data_root,
            )
            for index, sample in enumerate(split_samples)
        ]
        destination = output_dir / f"dair_infos_{split}.pkl"
        with destination.open("xb") as stream:
            pickle.dump(
                {"metainfo": metainfo, "data_list": data_list},
                stream,
                protocol=4,
            )
        split_counts[split] = len(data_list)

    return {
        "artifact_type": "ffnet_complemented_manifest_infos",
        "manifest_content_sha256": content_sha256,
        "coordinate_convention": metainfo["coordinate_convention"],
        "fusion_calibration": (
            "ego_lidar_from_rsu_lidar derived from sealed per-sample poses"
        ),
        "split_counts": split_counts,
        "prepared_artifact_count": len(prepared),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_ffnet_infos(
        args.manifest,
        args.output_dir,
        data_root=args.data_root,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
