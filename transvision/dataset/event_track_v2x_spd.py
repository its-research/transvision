"""Strict V2X-Seq-SPD metadata and cooperative tracking-label reader.

The reader is deliberately independent of MMDetection3D.  It validates the
three ``data_info.json`` tables, joins cooperative pairs to both side tables,
checks nominal 10 Hz timestamp monotonicity, and preserves the persistent
tracking identifiers required by EventTrack-V2X.

Loading labels is a ground-truth/data canary only.  It never runs a detector or
tracker and therefore cannot produce a model-performance claim.
"""

from __future__ import annotations

import json
import math
import re
import stat
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal


__all__ = (
    "SPDDataError",
    "SPDSideFrame",
    "SPDTrackLabel",
    "SPDPair",
    "SPDMetadata",
    "load_spd_metadata",
)


_FRAME_ID = re.compile(r"[0-9]{6}")
_SEQUENCE_ID = re.compile(r"[0-9]{4}")
_FROM_SIDES = ("veh", "inf", "coop")
_NOMINAL_PERIOD_US = 100_000
_MEDIAN_PERIOD_MIN_US = 80_000
_MEDIAN_PERIOD_MAX_US = 120_000


class SPDDataError(ValueError):
    """Raised when SPD metadata or labels violate the dataset contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SPDDataError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise SPDDataError(f"non-finite JSON constant is forbidden: {value}")


def _read_json_regular(root: Path, relative_path: str) -> object:
    relative = _relative_path(relative_path, "JSON path")
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as error:
        raise SPDDataError(f"required JSON file is missing: {relative}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SPDDataError(f"JSON input must be a regular non-symlink file: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise SPDDataError(f"JSON input escapes the dataset root: {relative}") from error
    try:
        return json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SPDDataError(f"invalid UTF-8 JSON: {relative}") from error


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SPDDataError(f"{context} must be an object")
    if not all(type(key) is str for key in value):
        raise SPDDataError(f"{context} keys must be strings")
    return value


def _array(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SPDDataError(f"{context} must be an array")
    return value


def _required_fields(
    value: Mapping[str, object], required: frozenset[str], context: str
) -> None:
    missing = sorted(required - frozenset(value))
    if missing:
        raise SPDDataError(f"{context} is missing fields: {missing}")


def _nonempty_string(value: object, context: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SPDDataError(f"{context} must be a trimmed non-empty string")
    return value


def _identifier(value: object, pattern: re.Pattern[str], context: str) -> str:
    result = _nonempty_string(value, context)
    if pattern.fullmatch(result) is None:
        raise SPDDataError(f"{context} is not canonical: {result!r}")
    return result


def _timestamp(value: object, context: str) -> int:
    if type(value) is int:
        result = value
    elif type(value) is str and value.isascii() and value.isdecimal():
        if len(value) > 1 and value.startswith("0"):
            raise SPDDataError(f"{context} is not a canonical decimal timestamp")
        result = int(value, 10)
    else:
        raise SPDDataError(f"{context} must be a positive integer timestamp")
    if result <= 0:
        raise SPDDataError(f"{context} must be positive")
    return result


def _finite_float(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise SPDDataError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SPDDataError(f"{context} must be finite")
    return result


def _relative_path(value: object, context: str) -> str:
    result = _nonempty_string(value, context)
    if (
        "\\" in result
        or "\x00" in result
        or result.startswith("/")
        or result.endswith("/")
        or "//" in result
    ):
        raise SPDDataError(f"{context} must be a canonical POSIX relative path")
    pure = PurePosixPath(result)
    if pure.is_absolute() or pure.as_posix() != result or any(
        part in ("", ".", "..") for part in pure.parts
    ):
        raise SPDDataError(f"{context} must be a canonical POSIX relative path")
    return result


@dataclass(frozen=True)
class SPDSideFrame:
    side: Literal["vehicle", "infrastructure"]
    frame_id: str
    sequence_id: str
    image_timestamp_us: int
    pointcloud_timestamp_us: int
    image_relative_path: str
    pointcloud_relative_path: str
    label_relative_path: str


@dataclass(frozen=True)
class SPDTrackLabel:
    sequence_id: str
    vehicle_frame_id: str
    infrastructure_frame_id: str
    track_id: str
    token: str
    from_side: Literal["veh", "inf", "coop"]
    vehicle_pointcloud_timestamp_us: int
    infrastructure_pointcloud_timestamp_us: int
    vehicle_track_id: str
    infrastructure_track_id: str
    vehicle_token: str
    infrastructure_token: str
    category: str
    center_xyz: tuple[float, float, float]
    dimensions_lwh: tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class SPDPair:
    sequence_id: str
    vehicle_sequence_id: str
    infrastructure_sequence_id: str
    vehicle: SPDSideFrame
    infrastructure: SPDSideFrame
    system_error_offset_xy: tuple[float, float]
    labels: tuple[SPDTrackLabel, ...]


@dataclass(frozen=True)
class SPDMetadata:
    nominal_rate_hz: int
    sequence_ids: tuple[str, ...]
    vehicle_frame_count: int
    infrastructure_frame_count: int
    cooperative_pair_count: int
    selected_pair_count: int
    selected_label_count: int
    pairs: tuple[SPDPair, ...]

    def canary_summary(self) -> dict[str, object]:
        """Return counts with an explicit non-performance evidence boundary."""

        return {
            "evidence_scope": "metadata-and-ground-truth-canary-only",
            "model_inference_performed": False,
            "model_performance_measured": False,
            "nominal_rate_hz": self.nominal_rate_hz,
            "sequence_ids": list(self.sequence_ids),
            "vehicle_frame_count": self.vehicle_frame_count,
            "infrastructure_frame_count": self.infrastructure_frame_count,
            "cooperative_pair_count": self.cooperative_pair_count,
            "selected_pair_count": self.selected_pair_count,
            "selected_label_count": self.selected_label_count,
        }


_SIDE_REQUIRED = frozenset(
    {
        "image_path",
        "pointcloud_path",
        "image_timestamp",
        "pointcloud_timestamp",
        "frame_id",
        "sequence_id",
    }
)
_PAIR_REQUIRED = frozenset(
    {
        "vehicle_frame",
        "infrastructure_frame",
        "vehicle_sequence",
        "infrastructure_sequence",
        "system_error_offset",
    }
)
_LABEL_REQUIRED = frozenset(
    {
        "token",
        "type",
        "track_id",
        "3d_dimensions",
        "3d_location",
        "rotation",
        "from_side",
        "veh_pointcloud_timestamp",
        "inf_pointcloud_timestamp",
        "veh_frame_id",
        "inf_frame_id",
        "veh_track_id",
        "inf_track_id",
        "veh_token",
        "inf_token",
    }
)


def _side_frames(
    root: Path,
    side: Literal["vehicle", "infrastructure"],
) -> tuple[dict[str, SPDSideFrame], tuple[SPDSideFrame, ...]]:
    relative = f"{side}-side/data_info.json"
    values = _array(_read_json_regular(root, relative), relative)
    result: dict[str, SPDSideFrame] = {}
    ordered: list[SPDSideFrame] = []
    label_key = "label_lidar_std_path"
    for index, raw in enumerate(values):
        context = f"{relative}[{index}]"
        value = _mapping(raw, context)
        _required_fields(value, _SIDE_REQUIRED | {label_key}, context)
        frame_id = _identifier(value["frame_id"], _FRAME_ID, f"{context}.frame_id")
        sequence_id = _identifier(
            value["sequence_id"], _SEQUENCE_ID, f"{context}.sequence_id"
        )
        image_path = _relative_path(value["image_path"], f"{context}.image_path")
        pointcloud_path = _relative_path(
            value["pointcloud_path"], f"{context}.pointcloud_path"
        )
        label_path = _relative_path(value[label_key], f"{context}.{label_key}")
        for name, path in (
            ("image_path", image_path),
            ("pointcloud_path", pointcloud_path),
            (label_key, label_path),
        ):
            if PurePosixPath(path).stem != frame_id:
                raise SPDDataError(f"{context}.{name} disagrees with frame_id")
        frame = SPDSideFrame(
            side=side,
            frame_id=frame_id,
            sequence_id=sequence_id,
            image_timestamp_us=_timestamp(
                value["image_timestamp"], f"{context}.image_timestamp"
            ),
            pointcloud_timestamp_us=_timestamp(
                value["pointcloud_timestamp"], f"{context}.pointcloud_timestamp"
            ),
            image_relative_path=f"{side}-side/{image_path}",
            pointcloud_relative_path=f"{side}-side/{pointcloud_path}",
            label_relative_path=f"{side}-side/{label_path}",
        )
        if frame_id in result:
            raise SPDDataError(f"duplicate {side} frame_id: {frame_id}")
        result[frame_id] = frame
        ordered.append(frame)
    _validate_nominal_10hz(ordered, f"{side}-side data_info")
    return result, tuple(ordered)


def _validate_nominal_10hz(frames: Sequence[SPDSideFrame], context: str) -> None:
    by_sequence: dict[str, list[SPDSideFrame]] = defaultdict(list)
    for frame in frames:
        by_sequence[frame.sequence_id].append(frame)
    for sequence_id, sequence in by_sequence.items():
        frame_numbers = [int(frame.frame_id) for frame in sequence]
        if any(right <= left for left, right in zip(frame_numbers, frame_numbers[1:])):
            raise SPDDataError(
                f"{context} sequence {sequence_id} frame IDs are not increasing"
            )
        timestamps = [frame.pointcloud_timestamp_us for frame in sequence]
        intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
        if any(interval <= 0 for interval in intervals):
            raise SPDDataError(
                f"{context} sequence {sequence_id} timestamps are not increasing"
            )
        if intervals:
            median = float(statistics.median(intervals))
            if not _MEDIAN_PERIOD_MIN_US <= median <= _MEDIAN_PERIOD_MAX_US:
                raise SPDDataError(
                    f"{context} sequence {sequence_id} median period {median}us "
                    f"is inconsistent with nominal 10 Hz ({_NOMINAL_PERIOD_US}us)"
                )


def _track_label(
    raw: object,
    *,
    context: str,
    sequence_id: str,
    vehicle: SPDSideFrame,
    infrastructure: SPDSideFrame,
) -> SPDTrackLabel:
    value = _mapping(raw, context)
    _required_fields(value, _LABEL_REQUIRED, context)
    vehicle_frame_id = _identifier(
        value["veh_frame_id"], _FRAME_ID, f"{context}.veh_frame_id"
    )
    infrastructure_frame_id = _identifier(
        value["inf_frame_id"], _FRAME_ID, f"{context}.inf_frame_id"
    )
    if vehicle_frame_id != vehicle.frame_id:
        raise SPDDataError(f"{context}.veh_frame_id disagrees with cooperative pair")
    if infrastructure_frame_id != infrastructure.frame_id:
        raise SPDDataError(f"{context}.inf_frame_id disagrees with cooperative pair")
    vehicle_timestamp = _timestamp(
        value["veh_pointcloud_timestamp"],
        f"{context}.veh_pointcloud_timestamp",
    )
    infrastructure_timestamp = _timestamp(
        value["inf_pointcloud_timestamp"],
        f"{context}.inf_pointcloud_timestamp",
    )
    if vehicle_timestamp != vehicle.pointcloud_timestamp_us:
        raise SPDDataError(
            f"{context}.veh_pointcloud_timestamp disagrees with vehicle data_info"
        )
    if infrastructure_timestamp != infrastructure.pointcloud_timestamp_us:
        raise SPDDataError(
            f"{context}.inf_pointcloud_timestamp disagrees with infrastructure data_info"
        )
    from_side = _nonempty_string(value["from_side"], f"{context}.from_side")
    if from_side not in _FROM_SIDES:
        raise SPDDataError(f"{context}.from_side is invalid: {from_side!r}")
    dimensions = _mapping(value["3d_dimensions"], f"{context}.3d_dimensions")
    location = _mapping(value["3d_location"], f"{context}.3d_location")
    _required_fields(dimensions, frozenset({"l", "w", "h"}), context)
    _required_fields(location, frozenset({"x", "y", "z"}), context)
    dimensions_lwh = tuple(
        _finite_float(dimensions[name], f"{context}.3d_dimensions.{name}")
        for name in ("l", "w", "h")
    )
    if any(dimension <= 0.0 for dimension in dimensions_lwh):
        raise SPDDataError(f"{context}.3d_dimensions must be positive")
    center_xyz = tuple(
        _finite_float(location[name], f"{context}.3d_location.{name}")
        for name in ("x", "y", "z")
    )
    return SPDTrackLabel(
        sequence_id=sequence_id,
        vehicle_frame_id=vehicle_frame_id,
        infrastructure_frame_id=infrastructure_frame_id,
        track_id=_nonempty_string(value["track_id"], f"{context}.track_id"),
        token=_nonempty_string(value["token"], f"{context}.token"),
        from_side=from_side,  # type: ignore[arg-type]
        vehicle_pointcloud_timestamp_us=vehicle_timestamp,
        infrastructure_pointcloud_timestamp_us=infrastructure_timestamp,
        vehicle_track_id=_nonempty_string(
            value["veh_track_id"], f"{context}.veh_track_id"
        ),
        infrastructure_track_id=_nonempty_string(
            value["inf_track_id"], f"{context}.inf_track_id"
        ),
        vehicle_token=_nonempty_string(value["veh_token"], f"{context}.veh_token"),
        infrastructure_token=_nonempty_string(
            value["inf_token"], f"{context}.inf_token"
        ),
        category=_nonempty_string(value["type"], f"{context}.type"),
        center_xyz=center_xyz,  # type: ignore[arg-type]
        dimensions_lwh=dimensions_lwh,  # type: ignore[arg-type]
        yaw=_finite_float(value["rotation"], f"{context}.rotation"),
    )


def _selected_sequences(value: Sequence[str] | None) -> frozenset[str] | None:
    if value is None:
        return None
    result = tuple(
        _identifier(item, _SEQUENCE_ID, "selected sequence ID") for item in value
    )
    if not result:
        raise SPDDataError("selected sequence IDs must not be empty")
    if len(result) != len(set(result)):
        raise SPDDataError("selected sequence IDs must not contain duplicates")
    return frozenset(result)


def load_spd_metadata(
    dataset_root: str | Path,
    *,
    sequence_ids: Sequence[str] | None = None,
) -> SPDMetadata:
    """Load and validate V2X-Seq-SPD metadata and cooperative labels.

    ``sequence_ids`` narrows label loading for local smoke tests.  All three
    side/pair metadata tables are still validated before selection.
    """

    root = Path(dataset_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise SPDDataError(f"dataset root is not a directory: {root}")
    selected = _selected_sequences(sequence_ids)
    vehicle_by_id, vehicle_frames = _side_frames(root, "vehicle")
    infrastructure_by_id, infrastructure_frames = _side_frames(
        root, "infrastructure"
    )
    pair_values = _array(
        _read_json_regular(root, "cooperative/data_info.json"),
        "cooperative/data_info.json",
    )
    pairs: list[SPDPair] = []
    seen_pairs: set[tuple[str, str]] = set()
    observed_sequences: list[str] = []
    all_sequences: set[str] = set()
    prior_pair_by_sequence: dict[str, tuple[int, int, int, int]] = {}
    label_count = 0
    for index, raw in enumerate(pair_values):
        context = f"cooperative/data_info.json[{index}]"
        value = _mapping(raw, context)
        _required_fields(value, _PAIR_REQUIRED, context)
        vehicle_frame_id = _identifier(
            value["vehicle_frame"], _FRAME_ID, f"{context}.vehicle_frame"
        )
        infrastructure_frame_id = _identifier(
            value["infrastructure_frame"],
            _FRAME_ID,
            f"{context}.infrastructure_frame",
        )
        vehicle_sequence_id = _identifier(
            value["vehicle_sequence"],
            _SEQUENCE_ID,
            f"{context}.vehicle_sequence",
        )
        infrastructure_sequence_id = _identifier(
            value["infrastructure_sequence"],
            _SEQUENCE_ID,
            f"{context}.infrastructure_sequence",
        )
        if vehicle_sequence_id != infrastructure_sequence_id:
            raise SPDDataError(f"{context} joins different sequence IDs")
        sequence_id = vehicle_sequence_id
        all_sequences.add(sequence_id)
        pair_identity = (vehicle_frame_id, infrastructure_frame_id)
        if pair_identity in seen_pairs:
            raise SPDDataError(f"duplicate cooperative pair: {pair_identity}")
        seen_pairs.add(pair_identity)
        vehicle = vehicle_by_id.get(vehicle_frame_id)
        infrastructure = infrastructure_by_id.get(infrastructure_frame_id)
        if vehicle is None or infrastructure is None:
            raise SPDDataError(f"{context} references an unknown side frame")
        if vehicle.sequence_id != vehicle_sequence_id:
            raise SPDDataError(f"{context}.vehicle_sequence disagrees with data_info")
        if infrastructure.sequence_id != infrastructure_sequence_id:
            raise SPDDataError(
                f"{context}.infrastructure_sequence disagrees with data_info"
            )
        current = (
            int(vehicle_frame_id),
            int(infrastructure_frame_id),
            vehicle.pointcloud_timestamp_us,
            infrastructure.pointcloud_timestamp_us,
        )
        prior = prior_pair_by_sequence.get(sequence_id)
        if prior is not None and any(
            right <= left for left, right in zip(prior, current)
        ):
            raise SPDDataError(
                f"cooperative sequence {sequence_id} is not strictly monotonic"
            )
        prior_pair_by_sequence[sequence_id] = current
        if selected is not None and sequence_id not in selected:
            continue
        if sequence_id not in observed_sequences:
            observed_sequences.append(sequence_id)
        offset = _mapping(value["system_error_offset"], f"{context}.offset")
        _required_fields(offset, frozenset({"delta_x", "delta_y"}), context)
        label_relative = f"cooperative/label/{vehicle_frame_id}.json"
        label_values = _array(
            _read_json_regular(root, label_relative), label_relative
        )
        labels = tuple(
            _track_label(
                label,
                context=f"{label_relative}[{label_index}]",
                sequence_id=sequence_id,
                vehicle=vehicle,
                infrastructure=infrastructure,
            )
            for label_index, label in enumerate(label_values)
        )
        track_ids = [label.track_id for label in labels]
        tokens = [label.token for label in labels]
        if len(track_ids) != len(set(track_ids)):
            raise SPDDataError(f"{label_relative} contains duplicate track_id")
        if len(tokens) != len(set(tokens)):
            raise SPDDataError(f"{label_relative} contains duplicate token")
        label_count += len(labels)
        pairs.append(
            SPDPair(
                sequence_id=sequence_id,
                vehicle_sequence_id=vehicle_sequence_id,
                infrastructure_sequence_id=infrastructure_sequence_id,
                vehicle=vehicle,
                infrastructure=infrastructure,
                system_error_offset_xy=(
                    _finite_float(offset["delta_x"], f"{context}.offset.delta_x"),
                    _finite_float(offset["delta_y"], f"{context}.offset.delta_y"),
                ),
                labels=labels,
            )
        )
    if selected is not None:
        unknown = sorted(selected - all_sequences)
        if unknown:
            raise SPDDataError(f"selected sequence IDs are absent: {unknown}")
    return SPDMetadata(
        nominal_rate_hz=10,
        sequence_ids=tuple(observed_sequences),
        vehicle_frame_count=len(vehicle_frames),
        infrastructure_frame_count=len(infrastructure_frames),
        cooperative_pair_count=len(pair_values),
        selected_pair_count=len(pairs),
        selected_label_count=label_count,
        pairs=tuple(pairs),
    )
