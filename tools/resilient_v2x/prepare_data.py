from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.dataset.resilient_v2x_manifest import (
    GroundTruthBoxRecord,
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    OFFICIAL_COOPERATIVE_SPLIT_SHA256,
    PreparedArtifactRecord,
    RawSliceRecord,
    ReleaseInventoryEntry,
    TemporalManifest,
    TemporalSampleRecord,
    build_release_inventory,
    canonical_json_bytes,
    content_sha256,
    release_inventory_sha256,
)
from transvision.dataset.resilient_v2x_pcd import (
    _convert_verified_pcd_to_bin,
    _publish_immutable_bytes,
    _read_stable_regular_bytes,
    convert_pcd_to_bin,
)


_METADATA_PATHS = (
    "cooperative/data_info.json",
    "vehicle-side/data_info.json",
    "infrastructure-side/data_info.json",
)
_SPLITS = ("train", "val", "test")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_UNSIGNED_DECIMAL = re.compile(r"^[1-9][0-9]*$")
_FINITE_DECIMAL = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$"
)
_CUBOID_ATOL = 1e-5
_CUBOID_RTOL = 1e-6


def build_protocol_sequences(
    paired_records: Sequence[Mapping[str, object]],
    delta_t_ms: int,
    history_limit: int,
    interval_min_ms: int,
    interval_max_ms: int,
    max_capture_skew_ms: int,
) -> tuple[list[TemporalSampleRecord], list[dict[str, object]]]:
    parameters = {
        "delta_t_ms": delta_t_ms,
        "history_limit": history_limit,
        "interval_min_ms": interval_min_ms,
        "interval_max_ms": interval_max_ms,
        "max_capture_skew_ms": max_capture_skew_ms,
    }
    for name, value in parameters.items():
        minimum = 0 if name in {"history_limit", "max_capture_skew_ms"} else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if interval_min_ms > interval_max_ms:
        raise ValueError("interval_min_ms must not exceed interval_max_ms")

    branch_order = (
        ("ego", "lidar"),
        ("rsu", "lidar"),
        ("ego", "camera"),
        ("rsu", "camera"),
    )
    interval_min_us = interval_min_ms * 1000
    interval_max_us = interval_max_ms * 1000
    max_capture_skew_us = max_capture_skew_ms * 1000
    samples: list[TemporalSampleRecord] = []
    sequence_splits: list[dict[str, object]] = []
    completed_source_pairs: set[tuple[str, str]] = set()
    current_source_pair: tuple[str, str] | None = None
    source_sequence_id = ""
    protocol_sequence_id = ""
    segment_index = 0
    segment_ticks: list[tuple[RawSliceRecord, ...]] = []
    segment_split: object = None
    previous_timestamps: tuple[int, ...] | None = None
    previous_sample_id: str | None = None

    for record_index, record in enumerate(paired_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"paired record {record_index} must be a mapping")
        vehicle_batch_id = _required_string(
            record.get("vehicle_batch_id"),
            f"paired record {record_index} vehicle_batch_id",
        )
        infrastructure_batch_id = _required_string(
            record.get("infrastructure_batch_id"),
            f"paired record {record_index} infrastructure_batch_id",
        )
        source_pair = (vehicle_batch_id, infrastructure_batch_id)
        source_changed = source_pair != current_source_pair
        if source_changed:
            if source_pair in completed_source_pairs:
                raise ValueError(
                    "source batch pair reappeared non-contiguously"
                )
            if current_source_pair is not None:
                completed_source_pairs.add(current_source_pair)
            current_source_pair = source_pair
            source_sequence_id = _stable_identifier(
                "src-",
                {
                    "infrastructure_batch_id": infrastructure_batch_id,
                    "vehicle_batch_id": vehicle_batch_id,
                },
            )
            segment_index = 0
            protocol_sequence_id = _stable_identifier(
                "seq-",
                {
                    "segment_index": segment_index,
                    "source_sequence_id": source_sequence_id,
                },
            )
            segment_ticks = []
            segment_split = None
            previous_timestamps = None

        sample_id = _required_string(
            record.get("sample_id"),
            f"paired record {record_index} sample_id",
        )
        slices_value = record.get("slices")
        if (
            isinstance(slices_value, (str, bytes))
            or not isinstance(slices_value, Sequence)
            or len(slices_value) != 4
        ):
            raise ValueError(
                f"paired record {record_index} must contain four slices"
            )
        slice_mappings: list[Mapping[str, object]] = []
        timestamps: list[int] = []
        for branch_index, (expected_agent, expected_modality) in enumerate(
            branch_order
        ):
            raw_slice = slices_value[branch_index]
            if not isinstance(raw_slice, Mapping):
                raise ValueError("normalized slice must be a mapping")
            if (
                raw_slice.get("agent"),
                raw_slice.get("modality"),
            ) != (expected_agent, expected_modality):
                raise ValueError("normalized slices have the wrong branch order")
            timestamp = raw_slice.get("capture_timestamp_us")
            if type(timestamp) is not int or timestamp <= 0:
                raise ValueError(
                    "capture timestamp must be a positive integer"
                )
            slice_mappings.append(raw_slice)
            timestamps.append(timestamp)

        triggers: list[dict[str, object]] = []
        if previous_timestamps is not None:
            for branch_index, (agent, modality) in enumerate(branch_order):
                previous = previous_timestamps[branch_index]
                current = timestamps[branch_index]
                interval = current - previous
                if interval <= 0:
                    raise ValueError(
                        "capture timestamps must be strictly increasing"
                    )
                if not interval_min_us <= interval <= interval_max_us:
                    triggers.append(
                        {
                            "agent": agent,
                            "modality": modality,
                            "previous_capture_timestamp_us": previous,
                            "current_capture_timestamp_us": current,
                            "interval_us": interval,
                        }
                    )
        if max(timestamps) - min(timestamps) > max_capture_skew_us:
            raise ValueError("capture timestamp skew exceeds protocol limit")

        if triggers:
            if previous_sample_id is None:
                raise ValueError("interval boundary lacks a previous sample")
            segment_index += 1
            protocol_sequence_id = _stable_identifier(
                "seq-",
                {
                    "segment_index": segment_index,
                    "source_sequence_id": source_sequence_id,
                },
            )
            sequence_splits.append(
                {
                    "source_sequence_id": source_sequence_id,
                    "previous_sample_id": previous_sample_id,
                    "current_sample_id": sample_id,
                    "new_sequence_id": protocol_sequence_id,
                    "triggers": triggers,
                }
            )
            segment_ticks = []
            segment_split = None

        split = record.get("split")
        if segment_split is None:
            segment_split = split
        elif split != segment_split:
            raise ValueError("one protocol sequence cannot cross splits")
        n_t = len(segment_ticks)
        tick_slices: list[RawSliceRecord] = []
        for raw_slice in slice_mappings:
            agent = raw_slice["agent"]
            modality = raw_slice["modality"]
            frame_id = _required_string(
                raw_slice.get("frame_id"),
                "normalized slice frame_id",
            )
            packet_id = _stable_identifier(
                "pkt-",
                {
                    "agent": agent,
                    "frame_id": frame_id,
                    "modality": modality,
                    "protocol_sequence_id": protocol_sequence_id,
                },
            )
            tick_slices.append(
                RawSliceRecord(
                    agent=agent,  # type: ignore[arg-type]
                    modality=modality,  # type: ignore[arg-type]
                    n_s=n_t,
                    tau_s_ms=n_t * delta_t_ms,
                    capture_timestamp_us=raw_slice[
                        "capture_timestamp_us"
                    ],  # type: ignore[arg-type]
                    frame_id=frame_id,
                    packet_id=packet_id,
                    relative_path=raw_slice["relative_path"],  # type: ignore[arg-type]
                    world_from_agent=raw_slice[
                        "world_from_agent"
                    ],  # type: ignore[arg-type]
                    agent_from_sensor=raw_slice[
                        "agent_from_sensor"
                    ],  # type: ignore[arg-type]
                    calibration_relative_path=raw_slice[
                        "calibration_relative_path"
                    ],  # type: ignore[arg-type]
                    calibration_sha256=raw_slice[
                        "calibration_sha256"
                    ],  # type: ignore[arg-type]
                    camera_intrinsic=raw_slice.get(
                        "camera_intrinsic"
                    ),  # type: ignore[arg-type]
                    payload_valid=True,
                    pose_valid=True,
                    calibration_valid=True,
                )
            )
        segment_ticks.append(tuple(tick_slices))
        history_start = max(0, n_t - history_limit)
        source_slices = tuple(
            item
            for tick in segment_ticks[history_start:]
            for item in tick
        )
        ground_truth_value = record.get("ground_truth")
        if (
            isinstance(ground_truth_value, (str, bytes))
            or not isinstance(ground_truth_value, Sequence)
        ):
            raise ValueError("ground_truth must be a sequence")
        ground_truth: list[GroundTruthBoxRecord] = []
        for box in ground_truth_value:
            if isinstance(box, GroundTruthBoxRecord):
                ground_truth.append(box)
            elif isinstance(box, Mapping):
                ground_truth.append(GroundTruthBoxRecord(**box))  # type: ignore[arg-type]
            else:
                raise ValueError("ground_truth entries must be mappings")
        samples.append(
            TemporalSampleRecord(
                sample_id=sample_id,
                sequence_id=protocol_sequence_id,
                split=split,  # type: ignore[arg-type]
                n_t=n_t,
                tau_t_ms=n_t * delta_t_ms,
                source_slices=source_slices,
                annotation_path=record["annotation_path"],  # type: ignore[arg-type]
                annotation_sha256=record[
                    "annotation_sha256"
                ],  # type: ignore[arg-type]
                ground_truth=tuple(ground_truth),
            )
        )
        previous_timestamps = tuple(timestamps)
        previous_sample_id = sample_id
    return samples, sequence_splits


def _required_string(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _stable_identifier(prefix: str, payload: Mapping[str, object]) -> str:
    return prefix + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_bytes(raw: bytes, context: str) -> object:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} must be UTF-8 JSON") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {context}: {error}") from error


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _array(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be an array")
    return value


def _split_id(value: object, context: str) -> str:
    if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} is not a canonical sample ID")
    return value


def _parse_split(
    raw: bytes,
    *,
    actual_sha256: str,
    expected_sha256: str,
    protocol_scope: str,
    protocol_values: tuple[int, int, int, int, int],
) -> tuple[dict[str, str], set[str]]:
    if (
        type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("expected split SHA-256 must be lowercase hexadecimal")
    if actual_sha256 != expected_sha256:
        raise ValueError("split SHA-256 does not match expected value")
    if protocol_scope not in {"controlled", "fixture"}:
        raise ValueError("protocol_scope must be controlled or fixture")
    if protocol_scope == "controlled":
        if (
            actual_sha256 != OFFICIAL_COOPERATIVE_SPLIT_SHA256
            or expected_sha256 != OFFICIAL_COOPERATIVE_SPLIT_SHA256
        ):
            raise ValueError("controlled split must use the official SHA-256")
        if protocol_values != (100, 3, 50, 150, 50):
            raise ValueError(
                "controlled protocol values must be 100,3,50,150,50"
            )
    payload = _mapping(_load_json_bytes(raw, "split JSON"), "split JSON")
    cooperative = _mapping(
        payload.get("cooperative_split"),
        "cooperative_split",
    )
    split_lists: dict[str, list[str]] = {}
    split_by_id: dict[str, str] = {}
    for split_name in _SPLITS:
        values = [
            _split_id(item, f"cooperative_split.{split_name}")
            for item in _array(
                cooperative.get(split_name),
                f"cooperative_split.{split_name}",
            )
        ]
        if len(values) != len(set(values)):
            raise ValueError(f"cooperative_split.{split_name} has duplicates")
        split_lists[split_name] = values
        for sample_id in values:
            if sample_id in split_by_id:
                raise ValueError("cooperative split lists overlap")
            split_by_id[sample_id] = split_name
    test_a_value = cooperative.get("test_A", [])
    test_a = [
        _split_id(item, "cooperative_split.test_A")
        for item in _array(test_a_value, "cooperative_split.test_A")
    ]
    if len(test_a) != len(set(test_a)):
        raise ValueError("cooperative_split.test_A has duplicates")
    if not set(test_a).issubset(set(split_lists["test"])):
        raise ValueError("cooperative_split.test_A must be a subset of test")
    if protocol_scope == "controlled":
        counts = tuple(len(split_lists[name]) for name in _SPLITS)
        if counts != (4813, 1783, 2688):
            raise ValueError("controlled cooperative split counts are invalid")
    return split_by_id, set(split_by_id)


def _full_side_path(prefix: str, value: object, context: str) -> str:
    relative = _required_string(value, context)
    if relative.startswith(f"{prefix}/"):
        return relative
    return f"{prefix}/{relative}"


def _side_indexes(
    records_value: object,
    prefix: str,
) -> tuple[
    dict[str, tuple[int, Mapping[str, object]]],
    dict[str, tuple[int, Mapping[str, object]]],
]:
    pointcloud_index: dict[str, tuple[int, Mapping[str, object]]] = {}
    image_index: dict[str, tuple[int, Mapping[str, object]]] = {}
    for index, value in enumerate(
        _array(records_value, f"{prefix} data_info")
    ):
        record = _mapping(value, f"{prefix} data_info[{index}]")
        pointcloud_path = _full_side_path(
            prefix,
            record.get("pointcloud_path"),
            f"{prefix} pointcloud_path",
        )
        image_path = _full_side_path(
            prefix,
            record.get("image_path"),
            f"{prefix} image_path",
        )
        if pointcloud_path in pointcloud_index:
            raise ValueError(f"duplicate {prefix} pointcloud path")
        if image_path in image_index:
            raise ValueError(f"duplicate {prefix} image path")
        pointcloud_index[pointcloud_path] = (index, record)
        image_index[image_path] = (index, record)
    return pointcloud_index, image_index


def _joined_side_record(
    *,
    prefix: str,
    pointcloud_path: str,
    image_path: str,
    pointcloud_index: Mapping[
        str, tuple[int, Mapping[str, object]]
    ],
    image_index: Mapping[str, tuple[int, Mapping[str, object]]],
) -> Mapping[str, object]:
    pointcloud_match = pointcloud_index.get(pointcloud_path)
    image_match = image_index.get(image_path)
    if pointcloud_match is None or image_match is None:
        raise ValueError(f"missing exact {prefix} metadata join")
    if pointcloud_match[0] != image_match[0]:
        raise ValueError(f"conflicting {prefix} metadata joins")
    record = pointcloud_match[1]
    frame_id = _required_string(
        record.get("frame_id"),
        f"{prefix} frame_id",
    )
    if (
        PurePosixPath(pointcloud_path).stem != frame_id
        or PurePosixPath(image_path).stem != frame_id
    ):
        raise ValueError(f"{prefix} path/frame-ID disagreement")
    return record


def _timestamp(value: object, context: str) -> int:
    if type(value) is int:
        if value <= 0:
            raise ValueError(f"{context} must be positive")
        return value
    if type(value) is str and _UNSIGNED_DECIMAL.fullmatch(value):
        return int(value, 10)
    raise ValueError(
        f"{context} must be a positive integer or canonical decimal string"
    )


def _inventory_bytes(
    data_root: Path,
    relative_path: str,
    entry: ReleaseInventoryEntry,
) -> bytes:
    path = data_root.joinpath(*PurePosixPath(relative_path).parts)
    raw = _read_stable_regular_bytes(path)
    if (
        len(raw) != entry.size
        or hashlib.sha256(raw).hexdigest() != entry.sha256
    ):
        raise ValueError(
            f"file differs from frozen raw inventory: {relative_path}"
        )
    return raw


def _json_from_inventory(
    data_root: Path,
    relative_path: str,
    inventory_by_path: Mapping[str, ReleaseInventoryEntry],
) -> object:
    entry = inventory_by_path.get(relative_path)
    if entry is None:
        raise ValueError(f"raw inventory lacks {relative_path}")
    return _load_json_bytes(
        _inventory_bytes(data_root, relative_path, entry),
        relative_path,
    )


def _calibration_paths(
    *,
    vehicle: Mapping[str, object],
    infrastructure: Mapping[str, object],
    vehicle_frame_id: str,
    infrastructure_frame_id: str,
) -> dict[str, str]:
    expected = {
        "ego_lidar_to_novatel": (
            "vehicle-side",
            "calib_lidar_to_novatel_path",
            f"calib/lidar_to_novatel/{vehicle_frame_id}.json",
        ),
        "ego_novatel_to_world": (
            "vehicle-side",
            "calib_novatel_to_world_path",
            f"calib/novatel_to_world/{vehicle_frame_id}.json",
        ),
        "ego_lidar_to_camera": (
            "vehicle-side",
            "calib_lidar_to_camera_path",
            f"calib/lidar_to_camera/{vehicle_frame_id}.json",
        ),
        "ego_camera_intrinsic": (
            "vehicle-side",
            "calib_camera_intrinsic_path",
            f"calib/camera_intrinsic/{vehicle_frame_id}.json",
        ),
        "rsu_lidar_to_world": (
            "infrastructure-side",
            "calib_virtuallidar_to_world_path",
            f"calib/virtuallidar_to_world/{infrastructure_frame_id}.json",
        ),
        "rsu_lidar_to_camera": (
            "infrastructure-side",
            "calib_virtuallidar_to_camera_path",
            f"calib/virtuallidar_to_camera/{infrastructure_frame_id}.json",
        ),
        "rsu_camera_intrinsic": (
            "infrastructure-side",
            "calib_camera_intrinsic_path",
            f"calib/camera_intrinsic/{infrastructure_frame_id}.json",
        ),
    }
    result: dict[str, str] = {}
    for name, (prefix, field, required_relative) in expected.items():
        source = vehicle if prefix == "vehicle-side" else infrastructure
        actual = _required_string(source.get(field), field)
        if actual != required_relative:
            raise ValueError(f"{field} is not the canonical DAIR path")
        result[name] = f"{prefix}/{actual}"
    return result


def _numeric_array(
    value: object,
    *,
    shape: tuple[int, ...],
    context: str,
) -> np.ndarray:
    def reject_bool(item: object) -> None:
        if type(item) is bool:
            raise ValueError(f"{context} must not contain booleans")
        if isinstance(item, (list, tuple)):
            for nested in item:
                reject_bool(nested)

    reject_bool(value)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{context} has invalid shape or non-finite values")
    return array


def _translation(value: object, context: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if any(type(item) is bool for item in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{context} must not contain booleans")
    if array.shape == (3, 1):
        array = array[:, 0]
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{context} must contain three finite values")
    return array


def _transform(
    payload: object,
    *,
    context: str,
    nested_transform: bool = False,
) -> np.ndarray:
    mapping = _mapping(payload, context)
    if nested_transform:
        mapping = _mapping(mapping.get("transform"), f"{context}.transform")
    rotation = _numeric_array(
        mapping.get("rotation"),
        shape=(3, 3),
        context=f"{context}.rotation",
    )
    translation = _translation(
        mapping.get("translation"),
        f"{context}.translation",
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return _public_rigid_matrix(result, context)


def _intrinsic(payload: object, context: str) -> np.ndarray:
    mapping = _mapping(payload, context)
    value = mapping.get("cam_K")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}.cam_K must be numeric") from error
    if any(type(item) is bool for item in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{context}.cam_K must not contain booleans")
    if array.shape == (9,):
        array = array.reshape(3, 3)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{context}.cam_K must contain nine finite values")
    return array


def _plain_matrix(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(item) for item in row)
        for row in value.tolist()
    )


def _public_rigid_matrix(value: np.ndarray, context: str) -> np.ndarray:
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    try:
        probe = RawSliceRecord(
            agent="ego",
            modality="lidar",
            n_s=0,
            tau_s_ms=0,
            capture_timestamp_us=1,
            frame_id="preflight-frame",
            packet_id="preflight-packet",
            relative_path="preflight/payload.pcd",
            world_from_agent=_plain_matrix(value),
            agent_from_sensor=identity,
            calibration_relative_path="preflight/calibration.json",
            calibration_sha256="0" * 64,
            camera_intrinsic=None,
            payload_valid=True,
            pose_valid=True,
            calibration_valid=True,
        )
    except ManifestError as error:
        raise ManifestError(f"{context} is not rigid: {error}") from error
    return np.asarray(probe.world_from_agent, dtype=np.float64)


def _offset_number(value: object, context: str) -> float:
    if type(value) in (int, float):
        parsed = float(value)
    elif type(value) is str and _FINITE_DECIMAL.fullmatch(value):
        parsed = float(value)
    else:
        raise ValueError(f"{context} must be a canonical finite number")
    if not math.isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    return parsed


def _system_error_offset(value: object) -> tuple[float, float]:
    if value == "":
        return 0.0, 0.0
    mapping = _mapping(value, "system_error_offset")
    if set(mapping) != {"delta_x", "delta_y"}:
        raise ValueError(
            "system_error_offset requires exactly delta_x and delta_y"
        )
    return (
        _offset_number(mapping["delta_x"], "system_error_offset.delta_x"),
        _offset_number(mapping["delta_y"], "system_error_offset.delta_y"),
    )


def _clean_float(value: float) -> float:
    return 0.0 if abs(value) <= _CUBOID_ATOL else float(value)


def _dimension(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return result


def _ground_truth(
    payload: object,
    world_from_ego_lidar: np.ndarray,
) -> tuple[GroundTruthBoxRecord, ...]:
    annotations = _array(payload, "world annotation")
    try:
        ego_lidar_from_world = np.linalg.inv(world_from_ego_lidar)
    except np.linalg.LinAlgError as error:
        raise ValueError("ego world transform is singular") from error
    result: list[GroundTruthBoxRecord] = []
    for source_index, value in enumerate(annotations):
        annotation = _mapping(
            value,
            f"world annotation[{source_index}]",
        )
        class_name = annotation.get("type")
        if type(class_name) is not str:
            raise ValueError("annotation type must be a string")
        if class_name != "Car":
            continue
        if "world_8_points" not in annotation:
            raise ValueError("Car annotation requires explicit world_8_points")
        corners_world = _numeric_array(
            annotation["world_8_points"],
            shape=(8, 3),
            context="world_8_points",
        )
        if np.unique(corners_world, axis=0).shape[0] != 8:
            raise ValueError("world_8_points must contain eight unique corners")
        homogeneous = np.column_stack(
            [corners_world, np.ones(8, dtype=np.float64)]
        )
        corners = (ego_lidar_from_world @ homogeneous.T).T[:, :3]
        p0, p1, p2, p3, p4, p5, p6, p7 = corners
        length_vector = p0 - p3
        width_vector = p0 - p1
        height_vector = p4 - p0
        length = float(np.linalg.norm(length_vector))
        width = float(np.linalg.norm(width_vector))
        height = float(np.linalg.norm(height_vector))
        if min(length, width, height) <= _CUBOID_ATOL:
            raise ValueError("annotation cuboid is degenerate")
        for first, second in (
            (length_vector, width_vector),
            (length_vector, height_vector),
            (width_vector, height_vector),
        ):
            if not np.isclose(
                float(np.dot(first, second)),
                0.0,
                atol=_CUBOID_ATOL,
                rtol=_CUBOID_RTOL,
            ):
                raise ValueError("annotation cuboid edges are not orthogonal")
        expected = np.stack(
            (
                p0,
                p0 - width_vector,
                p0 - width_vector - length_vector,
                p0 - length_vector,
                p0 + height_vector,
                p0 - width_vector + height_vector,
                p0 - width_vector - length_vector + height_vector,
                p0 - length_vector + height_vector,
            )
        )
        if not np.allclose(
            corners,
            expected,
            atol=_CUBOID_ATOL,
            rtol=_CUBOID_RTOL,
        ):
            raise ValueError("annotation corners do not form a canonical cuboid")
        dimensions = _mapping(
            annotation.get("3d_dimensions"),
            "annotation 3d_dimensions",
        )
        declared = (
            _dimension(dimensions.get("l"), "declared length"),
            _dimension(dimensions.get("w"), "declared width"),
            _dimension(dimensions.get("h"), "declared height"),
        )
        if not np.allclose(
            np.asarray((length, width, height)),
            np.asarray(declared),
            atol=_CUBOID_ATOL,
            rtol=_CUBOID_RTOL,
        ):
            raise ValueError("annotation dimensions disagree with corners")
        bottom_center = corners[:4].mean(axis=0)
        yaw = math.atan2(length_vector[1], length_vector[0])
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        result.append(
            GroundTruthBoxRecord(
                class_name="Car",
                x=_clean_float(float(bottom_center[0])),
                y=_clean_float(float(bottom_center[1])),
                z_bottom=_clean_float(float(bottom_center[2])),
                length=_clean_float(length),
                width=_clean_float(width),
                height=_clean_float(height),
                yaw=_clean_float(yaw),
                source_annotation_index=source_index,
            )
        )
    return tuple(result)


def _validated_slice(
    *,
    agent: str,
    modality: str,
    capture_timestamp_us: int,
    frame_id: str,
    relative_path: str,
    world_from_agent: np.ndarray,
    agent_from_sensor: np.ndarray,
    calibration_relative_path: str,
    calibration_sha256: str,
    camera_intrinsic: np.ndarray | None,
) -> dict[str, object]:
    probe = RawSliceRecord(
        agent=agent,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        n_s=0,
        tau_s_ms=0,
        capture_timestamp_us=capture_timestamp_us,
        frame_id=frame_id,
        packet_id="preflight-packet",
        relative_path=relative_path,
        world_from_agent=_plain_matrix(world_from_agent),
        agent_from_sensor=_plain_matrix(agent_from_sensor),
        calibration_relative_path=calibration_relative_path,
        calibration_sha256=calibration_sha256,
        camera_intrinsic=(
            None
            if camera_intrinsic is None
            else _plain_matrix(camera_intrinsic)
        ),
        payload_valid=True,
        pose_valid=True,
        calibration_valid=True,
    )
    return {
        "agent": probe.agent,
        "modality": probe.modality,
        "capture_timestamp_us": probe.capture_timestamp_us,
        "frame_id": probe.frame_id,
        "relative_path": probe.relative_path,
        "world_from_agent": probe.world_from_agent,
        "agent_from_sensor": probe.agent_from_sensor,
        "calibration_relative_path": probe.calibration_relative_path,
        "calibration_sha256": probe.calibration_sha256,
        "camera_intrinsic": probe.camera_intrinsic,
    }


def _raw_slice_payload(value: RawSliceRecord) -> dict[str, object]:
    return {
        "agent": value.agent,
        "modality": value.modality,
        "n_s": value.n_s,
        "tau_s_ms": value.tau_s_ms,
        "capture_timestamp_us": value.capture_timestamp_us,
        "frame_id": value.frame_id,
        "packet_id": value.packet_id,
        "relative_path": value.relative_path,
        "world_from_agent": [list(row) for row in value.world_from_agent],
        "agent_from_sensor": [list(row) for row in value.agent_from_sensor],
        "calibration_relative_path": value.calibration_relative_path,
        "calibration_sha256": value.calibration_sha256,
        "camera_intrinsic": (
            None
            if value.camera_intrinsic is None
            else [list(row) for row in value.camera_intrinsic]
        ),
        "payload_valid": value.payload_valid,
        "pose_valid": value.pose_valid,
        "calibration_valid": value.calibration_valid,
    }


def _ground_truth_payload(
    value: GroundTruthBoxRecord,
) -> dict[str, object]:
    return {
        "class_name": value.class_name,
        "x": value.x,
        "y": value.y,
        "z_bottom": value.z_bottom,
        "length": value.length,
        "width": value.width,
        "height": value.height,
        "yaw": value.yaw,
        "source_annotation_index": value.source_annotation_index,
    }


def _sample_payload(value: TemporalSampleRecord) -> dict[str, object]:
    return {
        "sample_id": value.sample_id,
        "sequence_id": value.sequence_id,
        "split": value.split,
        "n_t": value.n_t,
        "tau_t_ms": value.tau_t_ms,
        "source_slices": [
            _raw_slice_payload(item) for item in value.source_slices
        ],
        "annotation_path": value.annotation_path,
        "annotation_sha256": value.annotation_sha256,
        "ground_truth": [
            _ground_truth_payload(item) for item in value.ground_truth
        ],
    }


def _inventory_payload(value: ReleaseInventoryEntry) -> dict[str, object]:
    return {
        "relative_path": value.relative_path,
        "size": value.size,
        "sha256": value.sha256,
    }


def _prepared_payload(value: PreparedArtifactRecord) -> dict[str, object]:
    return {
        "source_relative_path": value.source_relative_path,
        "prepared_relative_path": value.prepared_relative_path,
        "point_count": value.point_count,
        "size": value.size,
        "sha256": value.sha256,
        "dtype": value.dtype,
        "fields": list(value.fields),
    }


def _boundary_payload(value: Mapping[str, object]) -> dict[str, object]:
    triggers = _array(value["triggers"], "sequence split triggers")
    return {
        "source_sequence_id": value["source_sequence_id"],
        "previous_sample_id": value["previous_sample_id"],
        "current_sample_id": value["current_sample_id"],
        "new_sequence_id": value["new_sequence_id"],
        "triggers": [
            {
                "agent": _mapping(trigger, "sequence trigger")["agent"],
                "modality": _mapping(trigger, "sequence trigger")[
                    "modality"
                ],
                "previous_capture_timestamp_us": _mapping(
                    trigger, "sequence trigger"
                )["previous_capture_timestamp_us"],
                "current_capture_timestamp_us": _mapping(
                    trigger, "sequence trigger"
                )["current_capture_timestamp_us"],
                "interval_us": _mapping(trigger, "sequence trigger")[
                    "interval_us"
                ],
            }
            for trigger in triggers
        ],
    }


def _manifest_payload(value: TemporalManifest) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "protocol_scope": value.protocol_scope,
        "delta_t_ms": value.delta_t_ms,
        "history_limit": value.history_limit,
        "interval_min_ms": value.interval_min_ms,
        "interval_max_ms": value.interval_max_ms,
        "max_capture_skew_ms": value.max_capture_skew_ms,
        "split_sha256": value.split_sha256,
        "dataset_release_sha256": value.dataset_release_sha256,
        "release_inventory": [
            _inventory_payload(item) for item in value.release_inventory
        ],
        "prepared_artifacts": [
            _prepared_payload(item) for item in value.prepared_artifacts
        ],
        "history_eligible_train_count": value.history_eligible_train_count,
        "sequence_splits": [
            _boundary_payload(item) for item in value.sequence_splits
        ],
        "excluded_samples": [],
        "samples": [_sample_payload(item) for item in value.samples],
        "content_sha256": value.content_sha256,
    }


def _preflight_records(
    data_root: Path,
    *,
    split_by_id: Mapping[str, str],
    split_ids: set[str],
) -> tuple[
    list[dict[str, object]],
    tuple[ReleaseInventoryEntry, ...],
    tuple[str, ...],
]:
    metadata_inventory = build_release_inventory(data_root, _METADATA_PATHS)
    metadata_by_path = {
        entry.relative_path: entry for entry in metadata_inventory
    }
    cooperative_value = _load_json_bytes(
        _inventory_bytes(
            data_root,
            _METADATA_PATHS[0],
            metadata_by_path[_METADATA_PATHS[0]],
        ),
        _METADATA_PATHS[0],
    )
    vehicle_value = _load_json_bytes(
        _inventory_bytes(
            data_root,
            _METADATA_PATHS[1],
            metadata_by_path[_METADATA_PATHS[1]],
        ),
        _METADATA_PATHS[1],
    )
    infrastructure_value = _load_json_bytes(
        _inventory_bytes(
            data_root,
            _METADATA_PATHS[2],
            metadata_by_path[_METADATA_PATHS[2]],
        ),
        _METADATA_PATHS[2],
    )
    vehicle_pcd_index, vehicle_image_index = _side_indexes(
        vehicle_value,
        "vehicle-side",
    )
    infrastructure_pcd_index, infrastructure_image_index = _side_indexes(
        infrastructure_value,
        "infrastructure-side",
    )

    joined: list[dict[str, object]] = []
    sample_ids: list[str] = []
    raw_paths: set[str] = set(_METADATA_PATHS)
    lidar_paths: set[str] = set()
    for index, value in enumerate(
        _array(cooperative_value, "cooperative data_info")
    ):
        cooperative = _mapping(
            value,
            f"cooperative data_info[{index}]",
        )
        paths = {
            "vehicle_image": _required_string(
                cooperative.get("vehicle_image_path"),
                "vehicle_image_path",
            ),
            "vehicle_pointcloud": _required_string(
                cooperative.get("vehicle_pointcloud_path"),
                "vehicle_pointcloud_path",
            ),
            "infrastructure_image": _required_string(
                cooperative.get("infrastructure_image_path"),
                "infrastructure_image_path",
            ),
            "infrastructure_pointcloud": _required_string(
                cooperative.get("infrastructure_pointcloud_path"),
                "infrastructure_pointcloud_path",
            ),
            "label": _required_string(
                cooperative.get("cooperative_label_path"),
                "cooperative_label_path",
            ),
        }
        required_prefixes = {
            "vehicle_image": "vehicle-side/image/",
            "vehicle_pointcloud": "vehicle-side/velodyne/",
            "infrastructure_image": "infrastructure-side/image/",
            "infrastructure_pointcloud": (
                "infrastructure-side/velodyne/"
            ),
            "label": "cooperative/label_world/",
        }
        for name, prefix in required_prefixes.items():
            if not paths[name].startswith(prefix):
                raise ValueError(f"{name} is not in the canonical DAIR path")
        vehicle = _joined_side_record(
            prefix="vehicle-side",
            pointcloud_path=paths["vehicle_pointcloud"],
            image_path=paths["vehicle_image"],
            pointcloud_index=vehicle_pcd_index,
            image_index=vehicle_image_index,
        )
        infrastructure = _joined_side_record(
            prefix="infrastructure-side",
            pointcloud_path=paths["infrastructure_pointcloud"],
            image_path=paths["infrastructure_image"],
            pointcloud_index=infrastructure_pcd_index,
            image_index=infrastructure_image_index,
        )
        vehicle_frame_id = PurePosixPath(paths["vehicle_image"]).stem
        if (
            PurePosixPath(paths["vehicle_pointcloud"]).stem
            != vehicle_frame_id
            or PurePosixPath(paths["label"]).stem != vehicle_frame_id
            or paths["label"]
            != f"cooperative/label_world/{vehicle_frame_id}.json"
        ):
            raise ValueError("cooperative sample/path frame-ID disagreement")
        infrastructure_frame_id = PurePosixPath(
            paths["infrastructure_pointcloud"]
        ).stem
        if (
            PurePosixPath(paths["infrastructure_image"]).stem
            != infrastructure_frame_id
        ):
            raise ValueError("infrastructure sensor frame-ID disagreement")
        calibration_paths = _calibration_paths(
            vehicle=vehicle,
            infrastructure=infrastructure,
            vehicle_frame_id=vehicle_frame_id,
            infrastructure_frame_id=infrastructure_frame_id,
        )
        sample_ids.append(vehicle_frame_id)
        raw_paths.update(paths.values())
        raw_paths.update(calibration_paths.values())
        lidar_paths.update(
            (
                paths["vehicle_pointcloud"],
                paths["infrastructure_pointcloud"],
            )
        )
        joined.append(
            {
                "cooperative": cooperative,
                "vehicle": vehicle,
                "infrastructure": infrastructure,
                "paths": paths,
                "calibration_paths": calibration_paths,
                "sample_id": vehicle_frame_id,
                "vehicle_frame_id": vehicle_frame_id,
                "infrastructure_frame_id": infrastructure_frame_id,
            }
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("cooperative data_info has duplicate sample IDs")
    if set(sample_ids) != split_ids:
        missing = sorted(split_ids - set(sample_ids))
        unknown = sorted(set(sample_ids) - split_ids)
        raise ValueError(
            f"split/cooperative sample IDs differ; missing={missing}, "
            f"unknown={unknown}"
        )

    release_inventory = build_release_inventory(data_root, raw_paths)
    inventory_by_path = {
        entry.relative_path: entry for entry in release_inventory
    }
    for metadata_entry in metadata_inventory:
        if inventory_by_path.get(metadata_entry.relative_path) != metadata_entry:
            raise ValueError("metadata changed during raw inventory construction")

    normalized: list[dict[str, object]] = []
    identity = np.eye(4, dtype=np.float64)
    for item in joined:
        cooperative = _mapping(item["cooperative"], "cooperative record")
        vehicle = _mapping(item["vehicle"], "vehicle record")
        infrastructure = _mapping(
            item["infrastructure"],
            "infrastructure record",
        )
        paths = _mapping(item["paths"], "record paths")
        calibration_paths = _mapping(
            item["calibration_paths"],
            "calibration paths",
        )
        calibrations = {
            name: _json_from_inventory(
                data_root,
                path,  # type: ignore[arg-type]
                inventory_by_path,
            )
            for name, path in calibration_paths.items()
        }
        ego_lidar_to_novatel = _transform(
            calibrations["ego_lidar_to_novatel"],
            context="ego lidar_to_novatel",
            nested_transform=True,
        )
        ego_novatel_to_world = _transform(
            calibrations["ego_novatel_to_world"],
            context="ego novatel_to_world",
        )
        ego_lidar_to_camera = _transform(
            calibrations["ego_lidar_to_camera"],
            context="ego lidar_to_camera",
        )
        rsu_lidar_to_world = _transform(
            calibrations["rsu_lidar_to_world"],
            context="rsu virtuallidar_to_world",
        )
        rsu_lidar_to_camera = _transform(
            calibrations["rsu_lidar_to_camera"],
            context="rsu virtuallidar_to_camera",
        )
        ego_intrinsic = _intrinsic(
            calibrations["ego_camera_intrinsic"],
            "ego camera_intrinsic",
        )
        rsu_intrinsic = _intrinsic(
            calibrations["rsu_camera_intrinsic"],
            "rsu camera_intrinsic",
        )
        world_from_ego = ego_novatel_to_world @ ego_lidar_to_novatel
        world_from_rsu = rsu_lidar_to_world.copy()
        delta_x, delta_y = _system_error_offset(
            cooperative.get("system_error_offset")
        )
        world_from_rsu[0, 3] += delta_x
        world_from_rsu[1, 3] += delta_y
        try:
            ego_camera_from_agent = np.linalg.inv(ego_lidar_to_camera)
            rsu_camera_from_agent = np.linalg.inv(rsu_lidar_to_camera)
        except np.linalg.LinAlgError as error:
            raise ValueError("camera extrinsic is singular") from error

        ego_lidar_timestamp = _timestamp(
            vehicle.get("pointcloud_timestamp"),
            "vehicle pointcloud_timestamp",
        )
        ego_camera_timestamp = _timestamp(
            vehicle.get("image_timestamp"),
            "vehicle image_timestamp",
        )
        rsu_lidar_timestamp = _timestamp(
            infrastructure.get("pointcloud_timestamp"),
            "infrastructure pointcloud_timestamp",
        )
        rsu_camera_timestamp = _timestamp(
            infrastructure.get("image_timestamp"),
            "infrastructure image_timestamp",
        )
        vehicle_frame_id = _required_string(
            item["vehicle_frame_id"],
            "vehicle frame ID",
        )
        infrastructure_frame_id = _required_string(
            item["infrastructure_frame_id"],
            "infrastructure frame ID",
        )

        def inventory_hash(relative_path: object) -> str:
            path = _required_string(relative_path, "inventory path")
            entry = inventory_by_path.get(path)
            if entry is None:
                raise ValueError(f"inventory lacks {path}")
            return entry.sha256

        slices = [
            _validated_slice(
                agent="ego",
                modality="lidar",
                capture_timestamp_us=ego_lidar_timestamp,
                frame_id=vehicle_frame_id,
                relative_path=_required_string(
                    paths["vehicle_pointcloud"],
                    "vehicle pointcloud path",
                ),
                world_from_agent=world_from_ego,
                agent_from_sensor=identity,
                calibration_relative_path=_required_string(
                    calibration_paths["ego_lidar_to_novatel"],
                    "ego lidar calibration",
                ),
                calibration_sha256=inventory_hash(
                    calibration_paths["ego_lidar_to_novatel"]
                ),
                camera_intrinsic=None,
            ),
            _validated_slice(
                agent="rsu",
                modality="lidar",
                capture_timestamp_us=rsu_lidar_timestamp,
                frame_id=infrastructure_frame_id,
                relative_path=_required_string(
                    paths["infrastructure_pointcloud"],
                    "infrastructure pointcloud path",
                ),
                world_from_agent=world_from_rsu,
                agent_from_sensor=identity,
                calibration_relative_path=_required_string(
                    calibration_paths["rsu_lidar_to_world"],
                    "rsu lidar calibration",
                ),
                calibration_sha256=inventory_hash(
                    calibration_paths["rsu_lidar_to_world"]
                ),
                camera_intrinsic=None,
            ),
            _validated_slice(
                agent="ego",
                modality="camera",
                capture_timestamp_us=ego_camera_timestamp,
                frame_id=vehicle_frame_id,
                relative_path=_required_string(
                    paths["vehicle_image"],
                    "vehicle image path",
                ),
                world_from_agent=world_from_ego,
                agent_from_sensor=ego_camera_from_agent,
                calibration_relative_path=_required_string(
                    calibration_paths["ego_lidar_to_camera"],
                    "ego camera calibration",
                ),
                calibration_sha256=inventory_hash(
                    calibration_paths["ego_lidar_to_camera"]
                ),
                camera_intrinsic=ego_intrinsic,
            ),
            _validated_slice(
                agent="rsu",
                modality="camera",
                capture_timestamp_us=rsu_camera_timestamp,
                frame_id=infrastructure_frame_id,
                relative_path=_required_string(
                    paths["infrastructure_image"],
                    "infrastructure image path",
                ),
                world_from_agent=world_from_rsu,
                agent_from_sensor=rsu_camera_from_agent,
                calibration_relative_path=_required_string(
                    calibration_paths["rsu_lidar_to_camera"],
                    "rsu camera calibration",
                ),
                calibration_sha256=inventory_hash(
                    calibration_paths["rsu_lidar_to_camera"]
                ),
                camera_intrinsic=rsu_intrinsic,
            ),
        ]
        annotation_path = _required_string(
            paths["label"],
            "annotation path",
        )
        annotation_entry = inventory_by_path[annotation_path]
        labels = _ground_truth(
            _json_from_inventory(
                data_root,
                annotation_path,
                inventory_by_path,
            ),
            world_from_ego,
        )
        normalized.append(
            {
                "sample_id": item["sample_id"],
                "split": split_by_id[item["sample_id"]],  # type: ignore[index]
                "vehicle_batch_id": _required_string(
                    vehicle.get("batch_id"),
                    "vehicle batch_id",
                ),
                "infrastructure_batch_id": _required_string(
                    infrastructure.get("batch_id"),
                    "infrastructure batch_id",
                ),
                "slices": slices,
                "annotation_path": annotation_path,
                "annotation_sha256": annotation_entry.sha256,
                "ground_truth": labels,
            }
        )
    return (
        normalized,
        release_inventory,
        tuple(sorted(lidar_paths)),
    )


def _prepare_manifest_impl(
    data_root: Path,
    split_path: Path,
    output_path: Path,
    expected_split_sha256: str,
    protocol_scope: Literal["controlled", "fixture"],
    delta_t_ms: int,
    history_limit: int,
    interval_min_ms: int,
    interval_max_ms: int,
    max_capture_skew_ms: int,
) -> TemporalManifest:
    data_root = Path(data_root)
    split_path = Path(split_path)
    output_path = Path(output_path)
    split_raw = _read_stable_regular_bytes(split_path)
    split_sha256 = hashlib.sha256(split_raw).hexdigest()
    protocol_values = (
        delta_t_ms,
        history_limit,
        interval_min_ms,
        interval_max_ms,
        max_capture_skew_ms,
    )
    split_by_id, split_ids = _parse_split(
        split_raw,
        actual_sha256=split_sha256,
        expected_sha256=expected_split_sha256,
        protocol_scope=protocol_scope,
        protocol_values=protocol_values,
    )
    normalized, release_inventory, lidar_paths = _preflight_records(
        data_root,
        split_by_id=split_by_id,
        split_ids=split_ids,
    )
    samples, sequence_splits = build_protocol_sequences(
        normalized,
        delta_t_ms=delta_t_ms,
        history_limit=history_limit,
        interval_min_ms=interval_min_ms,
        interval_max_ms=interval_max_ms,
        max_capture_skew_ms=max_capture_skew_ms,
    )
    if build_release_inventory(
        data_root,
        (entry.relative_path for entry in release_inventory),
    ) != release_inventory:
        raise ValueError("raw release changed after preflight")

    inventory_by_path = {
        entry.relative_path: entry for entry in release_inventory
    }
    prepared_artifacts: list[PreparedArtifactRecord] = []
    for source_relative_path in lidar_paths:
        source_entry = inventory_by_path[source_relative_path]
        prepared_relative = (
            PurePosixPath("prepared/resilient_v2x")
            / PurePosixPath(source_relative_path).with_suffix(".bin")
        ).as_posix()
        prepared_path = data_root.joinpath(
            *PurePosixPath(prepared_relative).parts
        )
        prepared = _convert_verified_pcd_to_bin(
            data_root.joinpath(
                *PurePosixPath(source_relative_path).parts
            ),
            prepared_path,
            expected_size=source_entry.size,
            expected_sha256=source_entry.sha256,
        )
        prepared_artifacts.append(
            PreparedArtifactRecord(
                source_relative_path=source_relative_path,
                prepared_relative_path=prepared_relative,
                point_count=prepared.point_count,
                size=prepared.size,
                sha256=prepared.sha256,
                dtype=prepared.dtype,
                fields=prepared.fields,
            )
        )

    if build_release_inventory(
        data_root,
        (entry.relative_path for entry in release_inventory),
    ) != release_inventory:
        raise ValueError("raw release changed before manifest publication")
    if _read_stable_regular_bytes(split_path) != split_raw:
        raise ValueError("split file changed before manifest publication")

    history_eligible_train_count = sum(
        sample.split == "train" and sample.n_t >= history_limit
        for sample in samples
    )
    common: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_scope": protocol_scope,
        "delta_t_ms": delta_t_ms,
        "history_limit": history_limit,
        "interval_min_ms": interval_min_ms,
        "interval_max_ms": interval_max_ms,
        "max_capture_skew_ms": max_capture_skew_ms,
        "split_sha256": split_sha256,
        "dataset_release_sha256": release_inventory_sha256(
            release_inventory
        ),
        "release_inventory": release_inventory,
        "prepared_artifacts": tuple(prepared_artifacts),
        "history_eligible_train_count": history_eligible_train_count,
        "sequence_splits": tuple(sequence_splits),
        "excluded_samples": (),
        "samples": tuple(samples),
    }
    hash_payload: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_scope": protocol_scope,
        "delta_t_ms": delta_t_ms,
        "history_limit": history_limit,
        "interval_min_ms": interval_min_ms,
        "interval_max_ms": interval_max_ms,
        "max_capture_skew_ms": max_capture_skew_ms,
        "split_sha256": split_sha256,
        "dataset_release_sha256": common["dataset_release_sha256"],
        "release_inventory": [
            _inventory_payload(item) for item in release_inventory
        ],
        "prepared_artifacts": [
            _prepared_payload(item) for item in prepared_artifacts
        ],
        "history_eligible_train_count": history_eligible_train_count,
        "sequence_splits": [
            _boundary_payload(item) for item in sequence_splits
        ],
        "excluded_samples": [],
        "samples": [_sample_payload(item) for item in samples],
    }
    manifest_hash = content_sha256(hash_payload)
    manifest = TemporalManifest(
        **common,  # type: ignore[arg-type]
        content_sha256=manifest_hash,
    )
    payload = _manifest_payload(manifest)
    if content_sha256(payload) != manifest.content_sha256:
        raise ValueError("canonical manifest content hash is inconsistent")
    raw_manifest = canonical_json_bytes(payload)
    _publish_immutable_bytes(output_path, raw_manifest)
    return manifest


def prepare_manifest(
    data_root: Path,
    split_path: Path,
    output_path: Path,
    expected_split_sha256: str,
    protocol_scope: Literal["controlled", "fixture"],
    delta_t_ms: int,
    history_limit: int,
    interval_min_ms: int,
    interval_max_ms: int,
    max_capture_skew_ms: int,
) -> TemporalManifest:
    try:
        return _prepare_manifest_impl(
            data_root=data_root,
            split_path=split_path,
            output_path=output_path,
            expected_split_sha256=expected_split_sha256,
            protocol_scope=protocol_scope,
            delta_t_ms=delta_t_ms,
            history_limit=history_limit,
            interval_min_ms=interval_min_ms,
            interval_max_ms=interval_max_ms,
            max_capture_skew_ms=max_capture_skew_ms,
        )
    except OSError as error:
        detail = error.strerror or str(error) or type(error).__name__
        one_line_detail = " ".join(detail.splitlines())
        raise ValueError(
            f"filesystem error preparing manifest: {one_line_detail}"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic DAIR cooperative temporal manifest."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument(
        "--protocol-scope",
        choices=("controlled", "fixture"),
        default="controlled",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta-t-ms", type=int, required=True)
    parser.add_argument("--history-limit", type=int, required=True)
    parser.add_argument("--interval-min-ms", type=int, required=True)
    parser.add_argument("--interval-max-ms", type=int, required=True)
    parser.add_argument("--max-capture-skew-ms", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prepare_manifest(
            data_root=args.data_root,
            split_path=args.split_file,
            output_path=args.output,
            expected_split_sha256=args.expected_split_sha256,
            protocol_scope=args.protocol_scope,
            delta_t_ms=args.delta_t_ms,
            history_limit=args.history_limit,
            interval_min_ms=args.interval_min_ms,
            interval_max_ms=args.interval_max_ms,
            max_capture_skew_ms=args.max_capture_skew_ms,
        )
    except (ManifestError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
