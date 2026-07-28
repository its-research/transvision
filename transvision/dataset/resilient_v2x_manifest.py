from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal


__all__ = (
    "ManifestError",
    "MANIFEST_SCHEMA_VERSION",
    "OFFICIAL_COOPERATIVE_SPLIT_SHA256",
    "RawSliceRecord",
    "GroundTruthBoxRecord",
    "TemporalSampleRecord",
    "ReleaseInventoryEntry",
    "PreparedArtifactRecord",
    "TemporalManifest",
    "canonical_json_bytes",
    "content_sha256",
    "build_release_inventory",
    "release_inventory_sha256",
    "load_temporal_manifest",
)


class ManifestError(RuntimeError):
    """Temporal manifest structure, containment, or hash validation failed."""


MANIFEST_SCHEMA_VERSION = 1
OFFICIAL_COOPERATIVE_SPLIT_SHA256 = (
    "d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
)

_MATRIX_ABS_TOL = 1e-5
_MATRIX_REL_TOL = 1e-6
_AFFINE_MAX_CONDITION_NUMBER = 4.0
_BRANCH_ORDER = (
    ("ego", "lidar"),
    ("rsu", "lidar"),
    ("ego", "camera"),
    ("rsu", "camera"),
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_scope",
        "delta_t_ms",
        "history_limit",
        "interval_min_ms",
        "interval_max_ms",
        "max_capture_skew_ms",
        "split_sha256",
        "dataset_release_sha256",
        "release_inventory",
        "prepared_artifacts",
        "history_eligible_train_count",
        "sequence_splits",
        "excluded_samples",
        "samples",
        "content_sha256",
    }
)
_RAW_SLICE_FIELDS = frozenset(
    {
        "agent",
        "modality",
        "n_s",
        "tau_s_ms",
        "capture_timestamp_us",
        "frame_id",
        "packet_id",
        "relative_path",
        "world_from_agent",
        "agent_from_sensor",
        "calibration_relative_path",
        "calibration_sha256",
        "camera_intrinsic",
        "payload_valid",
        "pose_valid",
        "calibration_valid",
    }
)
_GROUND_TRUTH_FIELDS = frozenset(
    {
        "class_name",
        "x",
        "y",
        "z_bottom",
        "length",
        "width",
        "height",
        "yaw",
        "source_annotation_index",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "sequence_id",
        "split",
        "n_t",
        "tau_t_ms",
        "source_slices",
        "annotation_path",
        "annotation_sha256",
        "ground_truth",
    }
)
_INVENTORY_FIELDS = frozenset({"relative_path", "size", "sha256"})
_PREPARED_FIELDS = frozenset(
    {
        "source_relative_path",
        "prepared_relative_path",
        "point_count",
        "size",
        "sha256",
        "dtype",
        "fields",
    }
)
_SEQUENCE_SPLIT_FIELDS = frozenset(
    {
        "source_sequence_id",
        "previous_sample_id",
        "current_sample_id",
        "new_sequence_id",
        "triggers",
    }
)
_SEQUENCE_TRIGGER_FIELDS = frozenset(
    {
        "agent",
        "modality",
        "previous_capture_timestamp_us",
        "current_capture_timestamp_us",
        "interval_us",
    }
)
_EXCLUDED_FIELDS = frozenset({"sample_id", "sequence_id", "split", "n_t", "reason"})


def _expect_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{context} must be an object")
    if not all(type(key) is str for key in value):
        raise ManifestError(f"{context} keys must be strings")
    return value


def _expect_fields(
    value: object,
    expected: frozenset[str],
    context: str,
) -> Mapping[str, object]:
    mapping = _expect_mapping(value, context)
    actual = frozenset(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ManifestError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )
    return mapping


def _exact_int(value: object, context: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ManifestError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ManifestError(f"{context} must be >= {minimum}")
    return value


def _exact_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ManifestError(f"{context} must be a boolean")
    return value


def _finite_float(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise ManifestError(f"{context} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ManifestError(f"{context} must be a finite number") from error
    if not finite:
        raise ManifestError(f"{context} must be a finite number")
    return value  # type: ignore[return-value]


def _nonempty_string(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise ManifestError(f"{context} must be a non-empty string")
    return value


def _literal_string(
    value: object,
    choices: tuple[str, ...],
    context: str,
) -> str:
    result = _nonempty_string(value, context)
    if result not in choices:
        rendered = ", ".join(choices)
        raise ManifestError(f"{context} must be one of: {rendered}")
    return result


def _sequence_tuple(value: object, context: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestError(f"{context} must be a sequence")
    return tuple(value)


def _sha256(value: object, context: str) -> str:
    result = _nonempty_string(value, context)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ManifestError(f"{context} must be 64 lowercase hexadecimal characters")
    return result


def _relative_path(value: object, context: str) -> str:
    result = _nonempty_string(value, context)
    if (
        "\x00" in result
        or "\\" in result
        or result.startswith("/")
        or result.endswith("/")
        or "//" in result
        or (
            len(result) >= 2
            and result[0].isascii()
            and result[0].isalpha()
            and result[1] == ":"
        )
    ):
        raise ManifestError(f"{context} must be a canonical POSIX relative path")
    parts = result.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ManifestError(f"{context} must be a canonical POSIX relative path")
    pure = PurePosixPath(result)
    if pure.is_absolute() or pure.as_posix() != result:
        raise ManifestError(f"{context} must be a canonical POSIX relative path")
    return result


def _matrix(
    value: object,
    rows: int,
    columns: int,
    context: str,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ManifestError(f"{context} must have shape {rows}x{columns}")
    result: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise ManifestError(f"{context} must have shape {rows}x{columns}")
        result.append(
            tuple(
                _finite_float(item, f"{context}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
        )
    return tuple(result)


def _close(first: float, second: float) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=_MATRIX_REL_TOL,
        abs_tol=_MATRIX_ABS_TOL,
    )


def _runtime_float32(value: float, context: str) -> float:
    try:
        rounded = struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ManifestError(
            f"{context} is not representable in runtime float32"
        ) from error
    if not math.isfinite(rounded):
        raise ManifestError(f"{context} is not representable in runtime float32")
    return rounded


def _validate_runtime_inverse(
    matrix: tuple[tuple[float, ...], ...],
    inverse: tuple[tuple[float, ...], ...],
    context: str,
) -> None:
    runtime_matrix = tuple(
        tuple(
            _runtime_float32(item, f"{context}[{row_index}][{column_index}]")
            for column_index, item in enumerate(row)
        )
        for row_index, row in enumerate(matrix)
    )
    runtime_inverse = tuple(
        tuple(
            _runtime_float32(
                item,
                f"{context} inverse[{row_index}][{column_index}]",
            )
            for column_index, item in enumerate(row)
        )
        for row_index, row in enumerate(inverse)
    )
    for left, right in (
        (runtime_matrix, runtime_inverse),
        (runtime_inverse, runtime_matrix),
    ):
        for row_index in range(4):
            for column_index in range(4):
                product = 0.0
                for inner_index in range(4):
                    term = _runtime_float32(
                        left[row_index][inner_index] * right[inner_index][column_index],
                        f"{context} float32 inverse product",
                    )
                    product = _runtime_float32(
                        product + term,
                        f"{context} float32 inverse sum",
                    )
                expected = 1.0 if row_index == column_index else 0.0
                if not _close(product, expected):
                    raise ManifestError(f"{context} float32 inverse is unstable")


def _determinant_3x3(matrix: tuple[tuple[float, ...], ...]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _exact_determinant_3x3(
    matrix: tuple[tuple[float, ...], ...],
) -> Fraction:
    exact = tuple(tuple(Fraction(value) for value in row) for row in matrix)
    return (
        exact[0][0] * (exact[1][1] * exact[2][2] - exact[1][2] * exact[2][1])
        - exact[0][1] * (exact[1][0] * exact[2][2] - exact[1][2] * exact[2][0])
        + exact[0][2] * (exact[1][0] * exact[2][1] - exact[1][1] * exact[2][0])
    )


def _rigid_matrix(value: object, context: str) -> tuple[tuple[float, ...], ...]:
    matrix = _matrix(value, 4, 4, context)
    for actual, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0)):
        if not _close(actual, expected):
            raise ManifestError(f"{context} must have a homogeneous last row")
    rotation = tuple(tuple(row[:3]) for row in matrix[:3])
    for row_index in range(3):
        for other_index in range(3):
            dot = sum(
                rotation[row_index][column] * rotation[other_index][column]
                for column in range(3)
            )
            expected = 1.0 if row_index == other_index else 0.0
            if not _close(dot, expected):
                raise ManifestError(f"{context} rotation must be orthonormal")
    if not _close(_determinant_3x3(rotation), 1.0):
        raise ManifestError(f"{context} rotation must be proper")
    return matrix


def _proper_affine_matrix(
    value: object,
    context: str,
) -> tuple[tuple[float, ...], ...]:
    matrix = _matrix(value, 4, 4, context)
    for row_index, row in enumerate(matrix):
        for column_index, item in enumerate(row):
            _runtime_float32(item, f"{context}[{row_index}][{column_index}]")
    for actual, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0)):
        if not _close(actual, expected):
            raise ManifestError(f"{context} must have a homogeneous last row")
    linear = tuple(tuple(row[:3]) for row in matrix[:3])
    determinant = _determinant_3x3(linear)
    if determinant <= _MATRIX_ABS_TOL:
        raise ManifestError(f"{context} linear component must be proper and invertible")
    a, b, c = linear[0]
    d, e, f = linear[1]
    g, h, i = linear[2]
    inverse = (
        (
            (e * i - f * h) / determinant,
            (c * h - b * i) / determinant,
            (b * f - c * e) / determinant,
        ),
        (
            (f * g - d * i) / determinant,
            (a * i - c * g) / determinant,
            (c * d - a * f) / determinant,
        ),
        (
            (d * h - e * g) / determinant,
            (b * g - a * h) / determinant,
            (a * e - b * d) / determinant,
        ),
    )
    norm = max(sum(abs(item) for item in row) for row in linear)
    inverse_norm = max(sum(abs(item) for item in row) for row in inverse)
    condition_number = norm * inverse_norm
    if (
        not math.isfinite(condition_number)
        or condition_number > _AFFINE_MAX_CONDITION_NUMBER
    ):
        raise ManifestError(f"{context} linear component is ill-conditioned")
    for row_index in range(3):
        for column_index in range(3):
            product = sum(
                linear[row_index][inner] * inverse[inner][column_index]
                for inner in range(3)
            )
            expected = 1.0 if row_index == column_index else 0.0
            if not _close(product, expected):
                raise ManifestError(f"{context} linear inverse is unstable")
    inverse_translation = tuple(
        -sum(
            inverse[row_index][column_index] * matrix[column_index][3]
            for column_index in range(3)
        )
        for row_index in range(3)
    )
    affine_inverse = tuple(
        tuple((*inverse[row_index], inverse_translation[row_index]))
        for row_index in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    _validate_runtime_inverse(matrix, affine_inverse, context)
    return matrix


def _camera_matrix(
    value: object,
    context: str,
) -> tuple[tuple[float, ...], ...]:
    matrix = _matrix(value, 3, 3, context)
    if matrix[0][0] <= 0.0 or matrix[1][1] <= 0.0:
        raise ManifestError(f"{context} focal lengths must be positive")
    if _exact_determinant_3x3(matrix) == 0:
        raise ManifestError(f"{context} must be nonsingular")
    return matrix


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class RawSliceRecord:
    agent: Literal["ego", "rsu"]
    modality: Literal["lidar", "camera"]
    n_s: int
    tau_s_ms: int
    capture_timestamp_us: int
    frame_id: str
    packet_id: str
    relative_path: str
    world_from_agent: tuple[tuple[float, ...], ...]
    agent_from_sensor: tuple[tuple[float, ...], ...]
    calibration_relative_path: str
    calibration_sha256: str
    camera_intrinsic: tuple[tuple[float, ...], ...] | None
    payload_valid: bool
    pose_valid: bool
    calibration_valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent",
            _literal_string(self.agent, ("ego", "rsu"), "raw slice agent"),
        )
        object.__setattr__(
            self,
            "modality",
            _literal_string(
                self.modality,
                ("lidar", "camera"),
                "raw slice modality",
            ),
        )
        object.__setattr__(self, "n_s", _exact_int(self.n_s, "raw slice n_s", 0))
        object.__setattr__(
            self,
            "tau_s_ms",
            _exact_int(self.tau_s_ms, "raw slice tau_s_ms", 0),
        )
        object.__setattr__(
            self,
            "capture_timestamp_us",
            _exact_int(
                self.capture_timestamp_us,
                "raw slice capture_timestamp_us",
                1,
            ),
        )
        object.__setattr__(
            self,
            "frame_id",
            _nonempty_string(self.frame_id, "raw slice frame_id"),
        )
        object.__setattr__(
            self,
            "packet_id",
            _nonempty_string(self.packet_id, "raw slice packet_id"),
        )
        object.__setattr__(
            self,
            "relative_path",
            _relative_path(self.relative_path, "raw slice relative_path"),
        )
        object.__setattr__(
            self,
            "world_from_agent",
            _rigid_matrix(self.world_from_agent, "raw slice world_from_agent"),
        )
        object.__setattr__(
            self,
            "agent_from_sensor",
            (_rigid_matrix if self.modality == "lidar" else _proper_affine_matrix)(
                self.agent_from_sensor, "raw slice agent_from_sensor"
            ),
        )
        object.__setattr__(
            self,
            "calibration_relative_path",
            _relative_path(
                self.calibration_relative_path,
                "raw slice calibration_relative_path",
            ),
        )
        object.__setattr__(
            self,
            "calibration_sha256",
            _sha256(self.calibration_sha256, "raw slice calibration_sha256"),
        )
        if self.modality == "lidar":
            if self.camera_intrinsic is not None:
                raise ManifestError("LiDAR raw slice camera_intrinsic must be null")
        else:
            if self.camera_intrinsic is None:
                raise ManifestError("Camera raw slice requires camera_intrinsic")
            object.__setattr__(
                self,
                "camera_intrinsic",
                _camera_matrix(
                    self.camera_intrinsic,
                    "raw slice camera_intrinsic",
                ),
            )
        object.__setattr__(
            self,
            "payload_valid",
            _exact_bool(self.payload_valid, "raw slice payload_valid"),
        )
        object.__setattr__(
            self,
            "pose_valid",
            _exact_bool(self.pose_valid, "raw slice pose_valid"),
        )
        object.__setattr__(
            self,
            "calibration_valid",
            _exact_bool(
                self.calibration_valid,
                "raw slice calibration_valid",
            ),
        )


@dataclass(frozen=True)
class GroundTruthBoxRecord:
    class_name: Literal["Car"]
    x: float
    y: float
    z_bottom: float
    length: float
    width: float
    height: float
    yaw: float
    source_annotation_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "class_name",
            _literal_string(
                self.class_name,
                ("Car",),
                "ground truth class_name",
            ),
        )
        for name in ("x", "y", "z_bottom", "length", "width", "height", "yaw"):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), f"ground truth {name}"),
            )
        if self.length <= 0.0 or self.width <= 0.0 or self.height <= 0.0:
            raise ManifestError("ground truth dimensions must be positive")
        if not -math.pi <= self.yaw < math.pi:
            raise ManifestError("ground truth yaw must be in [-pi, pi)")
        object.__setattr__(
            self,
            "source_annotation_index",
            _exact_int(
                self.source_annotation_index,
                "ground truth source_annotation_index",
                0,
            ),
        )


@dataclass(frozen=True)
class TemporalSampleRecord:
    sample_id: str
    sequence_id: str
    split: Literal["train", "val", "test"]
    n_t: int
    tau_t_ms: int
    source_slices: tuple[RawSliceRecord, ...]
    annotation_path: str
    annotation_sha256: str
    ground_truth: tuple[GroundTruthBoxRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_id",
            _nonempty_string(self.sample_id, "sample_id"),
        )
        object.__setattr__(
            self,
            "sequence_id",
            _nonempty_string(self.sequence_id, "sequence_id"),
        )
        object.__setattr__(
            self,
            "split",
            _literal_string(
                self.split,
                ("train", "val", "test"),
                "sample split",
            ),
        )
        object.__setattr__(self, "n_t", _exact_int(self.n_t, "sample n_t", 0))
        object.__setattr__(
            self,
            "tau_t_ms",
            _exact_int(self.tau_t_ms, "sample tau_t_ms", 0),
        )
        source_slices = _sequence_tuple(
            self.source_slices,
            "sample source_slices",
        )
        if not all(isinstance(item, RawSliceRecord) for item in source_slices):
            raise ManifestError("sample source_slices must contain RawSliceRecord")
        source_positions = [
            (item.agent, item.modality, item.n_s) for item in source_slices
        ]
        if len(source_positions) != len(set(source_positions)):
            raise ManifestError("sample source positions must be unique")
        packet_ids = [item.packet_id for item in source_slices]
        if len(packet_ids) != len(set(packet_ids)):
            raise ManifestError("sample packet IDs must be unique")
        object.__setattr__(self, "source_slices", source_slices)
        object.__setattr__(
            self,
            "annotation_path",
            _relative_path(self.annotation_path, "sample annotation_path"),
        )
        object.__setattr__(
            self,
            "annotation_sha256",
            _sha256(self.annotation_sha256, "sample annotation_sha256"),
        )
        ground_truth = _sequence_tuple(
            self.ground_truth,
            "sample ground_truth",
        )
        if not all(isinstance(item, GroundTruthBoxRecord) for item in ground_truth):
            raise ManifestError("sample ground_truth must contain GroundTruthBoxRecord")
        indices = [item.source_annotation_index for item in ground_truth]
        if indices != sorted(set(indices)):
            raise ManifestError(
                "ground truth source_annotation_index must be unique and sorted"
            )
        object.__setattr__(self, "ground_truth", ground_truth)


@dataclass(frozen=True)
class ReleaseInventoryEntry:
    relative_path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _relative_path(self.relative_path, "inventory relative_path"),
        )
        object.__setattr__(
            self,
            "size",
            _exact_int(self.size, "inventory size", 0),
        )
        object.__setattr__(
            self,
            "sha256",
            _sha256(self.sha256, "inventory sha256"),
        )


@dataclass(frozen=True)
class PreparedArtifactRecord:
    source_relative_path: str
    prepared_relative_path: str
    point_count: int
    size: int
    sha256: str
    dtype: Literal["<f4"]
    fields: tuple[
        Literal["x"],
        Literal["y"],
        Literal["z"],
        Literal["intensity"],
    ]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_relative_path",
            _relative_path(
                self.source_relative_path,
                "prepared source_relative_path",
            ),
        )
        object.__setattr__(
            self,
            "prepared_relative_path",
            _relative_path(
                self.prepared_relative_path,
                "prepared prepared_relative_path",
            ),
        )
        if self.source_relative_path == self.prepared_relative_path:
            raise ManifestError("prepared source and destination must differ")
        object.__setattr__(
            self,
            "point_count",
            _exact_int(self.point_count, "prepared point_count", 0),
        )
        object.__setattr__(
            self,
            "size",
            _exact_int(self.size, "prepared size", 0),
        )
        if self.size != self.point_count * 16:
            raise ManifestError("prepared size must equal point_count * 16")
        object.__setattr__(
            self,
            "sha256",
            _sha256(self.sha256, "prepared sha256"),
        )
        object.__setattr__(
            self,
            "dtype",
            _literal_string(self.dtype, ("<f4",), "prepared dtype"),
        )
        prepared_fields = _sequence_tuple(self.fields, "prepared fields")
        if any(
            type(field) is not str for field in prepared_fields
        ) or prepared_fields != ("x", "y", "z", "intensity"):
            raise ManifestError("prepared fields must be x, y, z, intensity")
        object.__setattr__(self, "fields", prepared_fields)


@dataclass(frozen=True)
class TemporalManifest:
    schema_version: int
    protocol_scope: Literal["controlled", "fixture"]
    delta_t_ms: int
    history_limit: int
    interval_min_ms: int
    interval_max_ms: int
    max_capture_skew_ms: int
    split_sha256: str
    dataset_release_sha256: str
    release_inventory: tuple[ReleaseInventoryEntry, ...]
    prepared_artifacts: tuple[PreparedArtifactRecord, ...]
    history_eligible_train_count: int
    sequence_splits: tuple[Mapping[str, object], ...]
    excluded_samples: tuple[Mapping[str, object], ...]
    samples: tuple[TemporalSampleRecord, ...]
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _exact_int(self.schema_version, "schema_version"),
        )
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError("unsupported manifest schema version")
        object.__setattr__(
            self,
            "protocol_scope",
            _literal_string(
                self.protocol_scope,
                ("controlled", "fixture"),
                "protocol_scope",
            ),
        )
        object.__setattr__(
            self,
            "delta_t_ms",
            _exact_int(self.delta_t_ms, "delta_t_ms", 1),
        )
        object.__setattr__(
            self,
            "history_limit",
            _exact_int(self.history_limit, "history_limit", 0),
        )
        object.__setattr__(
            self,
            "interval_min_ms",
            _exact_int(self.interval_min_ms, "interval_min_ms", 1),
        )
        object.__setattr__(
            self,
            "interval_max_ms",
            _exact_int(self.interval_max_ms, "interval_max_ms", 1),
        )
        if self.interval_min_ms > self.interval_max_ms:
            raise ManifestError("interval_min_ms must be <= interval_max_ms")
        object.__setattr__(
            self,
            "max_capture_skew_ms",
            _exact_int(
                self.max_capture_skew_ms,
                "max_capture_skew_ms",
                0,
            ),
        )
        object.__setattr__(
            self,
            "split_sha256",
            _sha256(self.split_sha256, "split_sha256"),
        )
        object.__setattr__(
            self,
            "dataset_release_sha256",
            _sha256(
                self.dataset_release_sha256,
                "dataset_release_sha256",
            ),
        )
        release_inventory = _sequence_tuple(
            self.release_inventory,
            "release_inventory",
        )
        if not all(
            isinstance(item, ReleaseInventoryEntry) for item in release_inventory
        ):
            raise ManifestError("release_inventory must contain ReleaseInventoryEntry")
        object.__setattr__(self, "release_inventory", release_inventory)
        prepared_artifacts = _sequence_tuple(
            self.prepared_artifacts,
            "prepared_artifacts",
        )
        if not all(
            isinstance(item, PreparedArtifactRecord) for item in prepared_artifacts
        ):
            raise ManifestError(
                "prepared_artifacts must contain PreparedArtifactRecord"
            )
        object.__setattr__(self, "prepared_artifacts", prepared_artifacts)
        object.__setattr__(
            self,
            "history_eligible_train_count",
            _exact_int(
                self.history_eligible_train_count,
                "history_eligible_train_count",
                0,
            ),
        )
        sequence_splits = tuple(
            _parse_sequence_split(item, f"sequence_splits[{index}]")
            for index, item in enumerate(
                _sequence_tuple(self.sequence_splits, "sequence_splits")
            )
        )
        object.__setattr__(
            self,
            "sequence_splits",
            tuple(_deep_freeze(item) for item in sequence_splits),
        )
        excluded_samples = tuple(
            _parse_excluded(item, f"excluded_samples[{index}]")
            for index, item in enumerate(
                _sequence_tuple(self.excluded_samples, "excluded_samples")
            )
        )
        object.__setattr__(
            self,
            "excluded_samples",
            tuple(_deep_freeze(item) for item in excluded_samples),
        )
        samples = _sequence_tuple(self.samples, "samples")
        if not all(isinstance(item, TemporalSampleRecord) for item in samples):
            raise ManifestError("samples must contain TemporalSampleRecord")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, "content_sha256"),
        )
        _validate_manifest(self)


def _validate_json_domain(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("canonical JSON mapping keys must be strings")
            _validate_json_domain(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_domain(item)
        return
    if value is None or type(value) in (bool, int, float, str):
        return
    raise TypeError("value is outside the canonical JSON domain")


def canonical_json_bytes(value: object) -> bytes:
    _validate_json_domain(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value_without_hash: object) -> str:
    if not isinstance(value_without_hash, Mapping):
        raise TypeError("content_sha256 requires a top-level mapping")
    payload = dict(value_without_hash)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _inventory_plain(
    value: ReleaseInventoryEntry | Mapping[str, object],
    index: int,
) -> dict[str, object]:
    if isinstance(value, ReleaseInventoryEntry):
        return {
            "relative_path": value.relative_path,
            "size": value.size,
            "sha256": value.sha256,
        }
    mapping = _expect_fields(value, _INVENTORY_FIELDS, f"inventory[{index}]")
    entry = ReleaseInventoryEntry(
        relative_path=mapping["relative_path"],  # type: ignore[arg-type]
        size=mapping["size"],  # type: ignore[arg-type]
        sha256=mapping["sha256"],  # type: ignore[arg-type]
    )
    return {
        "relative_path": entry.relative_path,
        "size": entry.size,
        "sha256": entry.sha256,
    }


def release_inventory_sha256(
    inventory: Sequence[ReleaseInventoryEntry | Mapping[str, object]],
) -> str:
    if isinstance(inventory, (str, bytes)) or not isinstance(inventory, Sequence):
        raise ManifestError("release inventory must be a sequence")
    plain = [_inventory_plain(item, index) for index, item in enumerate(inventory)]
    previous: str | None = None
    for entry in plain:
        current = entry["relative_path"]
        assert isinstance(current, str)
        if current == previous:
            raise ManifestError("release inventory contains duplicate paths")
        if previous is not None and current < previous:
            raise ManifestError("release inventory must be strictly path-sorted")
        previous = current
    return hashlib.sha256(canonical_json_bytes(plain)).hexdigest()


def _entry_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


def _file_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _file_open_flags() -> int:
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def _stat_directory_entry(
    parent_descriptor: int,
    name: str,
    context: str,
) -> os.stat_result:
    try:
        metadata = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ManifestError(f"inventory path changed: {context}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ManifestError(f"inventory path contains a symlink: {context}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ManifestError(
            f"inventory directory component is not a directory: {context}"
        )
    return metadata


def _open_directory_entry(
    parent_descriptor: int,
    name: str,
    context: str,
) -> tuple[int, tuple[int, int, int]]:
    entry_before = _stat_directory_entry(
        parent_descriptor,
        name,
        context,
    )
    try:
        descriptor = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ManifestError(f"unable to open inventory directory: {context}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise ManifestError(
                f"inventory directory component is not a directory: {context}"
            )
        identity = _entry_identity(opened)
        if identity != _entry_identity(entry_before):
            raise ManifestError(
                f"inventory directory entry changed before open: {context}"
            )
        entry_after = _stat_directory_entry(
            parent_descriptor,
            name,
            context,
        )
        if identity != _entry_identity(entry_after):
            raise ManifestError(
                f"inventory directory entry changed during open: {context}"
            )
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _verify_directory_chain(
    anchor_descriptor: int,
    anchor_identity: tuple[int, int, int],
    links: Sequence[tuple[int, str, int, tuple[int, int, int], str]],
) -> None:
    anchor = os.fstat(anchor_descriptor)
    if not stat.S_ISDIR(anchor.st_mode) or _entry_identity(anchor) != anchor_identity:
        raise ManifestError("inventory anchor directory changed")
    for (
        parent_descriptor,
        name,
        child_descriptor,
        identity,
        context,
    ) in links:
        opened = os.fstat(child_descriptor)
        if not stat.S_ISDIR(opened.st_mode) or _entry_identity(opened) != identity:
            raise ManifestError(f"inventory directory descriptor changed: {context}")
        try:
            current = _stat_directory_entry(
                parent_descriptor,
                name,
                context,
            )
        except ManifestError as error:
            raise ManifestError(
                f"inventory directory entry changed: {context}"
            ) from error
        if _entry_identity(current) != identity:
            raise ManifestError(f"inventory directory entry changed: {context}")


def _hash_open_regular_file(
    parent_descriptor: int,
    name: str,
    relative_path: str,
    anchor_descriptor: int,
    anchor_identity: tuple[int, int, int],
    directory_links: Sequence[tuple[int, str, int, tuple[int, int, int], str]],
) -> tuple[int, str]:
    try:
        entry_before = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ManifestError(
            f"inventory path does not exist: {relative_path}"
        ) from error
    if stat.S_ISLNK(entry_before.st_mode):
        raise ManifestError(f"inventory path contains a symlink: {relative_path}")
    if not stat.S_ISREG(entry_before.st_mode):
        raise ManifestError(f"inventory path is not a regular file: {relative_path}")
    try:
        descriptor = os.open(
            name,
            _file_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ManifestError(
            f"unable to open inventory file: {relative_path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ManifestError(
                f"inventory path is not a regular file: {relative_path}"
            )
        if _entry_identity(before) != _entry_identity(entry_before):
            raise ManifestError(
                f"inventory file changed before hashing: {relative_path}"
            )
        _verify_directory_chain(
            anchor_descriptor,
            anchor_identity,
            directory_links,
        )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise ManifestError(
                f"inventory file changed while hashing: {relative_path}"
            )
        try:
            final_entry = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ManifestError(
                f"inventory file changed while hashing: {relative_path}"
            ) from error
        if (
            stat.S_ISLNK(final_entry.st_mode)
            or not stat.S_ISREG(final_entry.st_mode)
            or _file_identity(final_entry) != _file_identity(after)
        ):
            raise ManifestError(
                f"inventory file changed while hashing: {relative_path}"
            )
        _verify_directory_chain(
            anchor_descriptor,
            anchor_identity,
            directory_links,
        )
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def build_release_inventory(
    root: Path,
    relative_paths: Iterable[str],
) -> tuple[ReleaseInventoryEntry, ...]:
    root_path = Path(os.path.abspath(os.fspath(root)))
    canonical_paths: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(relative_paths):
        relative = _relative_path(value, f"relative_paths[{index}]")
        if relative in seen:
            raise ManifestError("release inventory contains duplicate paths")
        seen.add(relative)
        canonical_paths.append(relative)

    descriptors: list[int] = []
    root_links: list[tuple[int, str, int, tuple[int, int, int], str]] = []
    try:
        try:
            anchor_descriptor = os.open(
                root_path.anchor,
                _directory_open_flags(),
            )
        except OSError as error:
            raise ManifestError("inventory root anchor does not exist") from error
        descriptors.append(anchor_descriptor)
        anchor_stat = os.fstat(anchor_descriptor)
        if not stat.S_ISDIR(anchor_stat.st_mode):
            raise ManifestError("inventory root anchor is not a directory")
        anchor_identity = _entry_identity(anchor_stat)

        root_descriptor = anchor_descriptor
        root_context_parts: list[str] = []
        for part in root_path.parts[1:]:
            root_context_parts.append(part)
            context = f"root path {'/'.join(root_context_parts)}"
            try:
                child_descriptor, identity = _open_directory_entry(
                    root_descriptor,
                    part,
                    context,
                )
            except ManifestError as error:
                raise ManifestError(
                    "inventory root path changed, contains a symlink, "
                    "or is not a real directory"
                ) from error
            descriptors.append(child_descriptor)
            root_links.append(
                (
                    root_descriptor,
                    part,
                    child_descriptor,
                    identity,
                    context,
                )
            )
            root_descriptor = child_descriptor
        _verify_directory_chain(
            anchor_descriptor,
            anchor_identity,
            root_links,
        )

        result: list[ReleaseInventoryEntry] = []
        for relative in sorted(canonical_paths):
            parts = relative.split("/")
            parent_descriptor = root_descriptor
            relative_descriptors: list[int] = []
            relative_links: list[tuple[int, str, int, tuple[int, int, int], str]] = []
            try:
                context_parts: list[str] = []
                for part in parts[:-1]:
                    context_parts.append(part)
                    context = "/".join(context_parts)
                    child_descriptor, identity = _open_directory_entry(
                        parent_descriptor,
                        part,
                        context,
                    )
                    relative_descriptors.append(child_descriptor)
                    relative_links.append(
                        (
                            parent_descriptor,
                            part,
                            child_descriptor,
                            identity,
                            context,
                        )
                    )
                    parent_descriptor = child_descriptor
                size, digest = _hash_open_regular_file(
                    parent_descriptor,
                    parts[-1],
                    relative,
                    anchor_descriptor,
                    anchor_identity,
                    (*root_links, *relative_links),
                )
                result.append(
                    ReleaseInventoryEntry(
                        relative_path=relative,
                        size=size,
                        sha256=digest,
                    )
                )
            except ManifestError:
                raise
            except OSError as error:
                raise ManifestError(f"inventory path changed: {relative}") from error
            finally:
                for descriptor in reversed(relative_descriptors):
                    os.close(descriptor)
        _verify_directory_chain(
            anchor_descriptor,
            anchor_identity,
            root_links,
        )
        return tuple(result)
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _parse_raw_slice(value: object, context: str) -> RawSliceRecord:
    mapping = _expect_fields(value, _RAW_SLICE_FIELDS, context)
    return RawSliceRecord(**mapping)  # type: ignore[arg-type]


def _parse_ground_truth(value: object, context: str) -> GroundTruthBoxRecord:
    mapping = _expect_fields(value, _GROUND_TRUTH_FIELDS, context)
    return GroundTruthBoxRecord(**mapping)  # type: ignore[arg-type]


def _parse_sample(value: object, context: str) -> TemporalSampleRecord:
    mapping = _expect_fields(value, _SAMPLE_FIELDS, context)
    slices_value = mapping["source_slices"]
    if not isinstance(slices_value, list):
        raise ManifestError(f"{context}.source_slices must be an array")
    ground_truth_value = mapping["ground_truth"]
    if not isinstance(ground_truth_value, list):
        raise ManifestError(f"{context}.ground_truth must be an array")
    return TemporalSampleRecord(
        sample_id=mapping["sample_id"],  # type: ignore[arg-type]
        sequence_id=mapping["sequence_id"],  # type: ignore[arg-type]
        split=mapping["split"],  # type: ignore[arg-type]
        n_t=mapping["n_t"],  # type: ignore[arg-type]
        tau_t_ms=mapping["tau_t_ms"],  # type: ignore[arg-type]
        source_slices=tuple(
            _parse_raw_slice(item, f"{context}.source_slices[{index}]")
            for index, item in enumerate(slices_value)
        ),
        annotation_path=mapping["annotation_path"],  # type: ignore[arg-type]
        annotation_sha256=mapping["annotation_sha256"],  # type: ignore[arg-type]
        ground_truth=tuple(
            _parse_ground_truth(item, f"{context}.ground_truth[{index}]")
            for index, item in enumerate(ground_truth_value)
        ),
    )


def _parse_inventory(value: object, context: str) -> ReleaseInventoryEntry:
    mapping = _expect_fields(value, _INVENTORY_FIELDS, context)
    return ReleaseInventoryEntry(**mapping)  # type: ignore[arg-type]


def _parse_prepared(value: object, context: str) -> PreparedArtifactRecord:
    mapping = _expect_fields(value, _PREPARED_FIELDS, context)
    return PreparedArtifactRecord(**mapping)  # type: ignore[arg-type]


def _parse_sequence_split(value: object, context: str) -> Mapping[str, object]:
    mapping = _expect_fields(value, _SEQUENCE_SPLIT_FIELDS, context)
    triggers_value = mapping["triggers"]
    if not isinstance(triggers_value, (list, tuple)):
        raise ManifestError(f"{context}.triggers must be an array")
    triggers: list[dict[str, object]] = []
    for index, trigger_value in enumerate(triggers_value):
        trigger_context = f"{context}.triggers[{index}]"
        trigger = _expect_fields(
            trigger_value,
            _SEQUENCE_TRIGGER_FIELDS,
            trigger_context,
        )
        agent = _literal_string(
            trigger["agent"],
            ("ego", "rsu"),
            f"{trigger_context}.agent",
        )
        modality = _literal_string(
            trigger["modality"],
            ("lidar", "camera"),
            f"{trigger_context}.modality",
        )
        previous = _exact_int(
            trigger["previous_capture_timestamp_us"],
            f"{trigger_context}.previous_capture_timestamp_us",
            1,
        )
        current = _exact_int(
            trigger["current_capture_timestamp_us"],
            f"{trigger_context}.current_capture_timestamp_us",
            1,
        )
        interval = _exact_int(
            trigger["interval_us"],
            f"{trigger_context}.interval_us",
            1,
        )
        triggers.append(
            {
                "agent": agent,
                "modality": modality,
                "previous_capture_timestamp_us": previous,
                "current_capture_timestamp_us": current,
                "interval_us": interval,
            }
        )
    return {
        "source_sequence_id": _nonempty_string(
            mapping["source_sequence_id"],
            f"{context}.source_sequence_id",
        ),
        "previous_sample_id": _nonempty_string(
            mapping["previous_sample_id"],
            f"{context}.previous_sample_id",
        ),
        "current_sample_id": _nonempty_string(
            mapping["current_sample_id"],
            f"{context}.current_sample_id",
        ),
        "new_sequence_id": _nonempty_string(
            mapping["new_sequence_id"],
            f"{context}.new_sequence_id",
        ),
        "triggers": tuple(triggers),
    }


def _parse_excluded(value: object, context: str) -> Mapping[str, object]:
    mapping = _expect_fields(value, _EXCLUDED_FIELDS, context)
    split = _literal_string(
        mapping["split"],
        ("train", "val", "test"),
        f"{context}.split",
    )
    if mapping["reason"] != "no_supported_branch":
        raise ManifestError(f"{context}.reason is invalid")
    return {
        "sample_id": _nonempty_string(
            mapping["sample_id"],
            f"{context}.sample_id",
        ),
        "sequence_id": _nonempty_string(
            mapping["sequence_id"],
            f"{context}.sequence_id",
        ),
        "split": split,
        "n_t": _exact_int(mapping["n_t"], f"{context}.n_t", 0),
        "reason": "no_supported_branch",
    }


def _validate_manifest(manifest: TemporalManifest) -> None:
    if manifest.protocol_scope == "controlled":
        if manifest.split_sha256 != OFFICIAL_COOPERATIVE_SPLIT_SHA256:
            raise ManifestError(
                "controlled split must use the official cooperative hash"
            )
        if (
            manifest.delta_t_ms,
            manifest.history_limit,
            manifest.interval_min_ms,
            manifest.interval_max_ms,
            manifest.max_capture_skew_ms,
        ) != (100, 3, 50, 150, 200):
            raise ManifestError(
                "controlled manifest must use fixed protocol parameters"
            )
        if any(sample.split == "test" for sample in manifest.samples):
            raise ManifestError(
                "controlled manifest must contain only train and val samples"
            )

    if (
        release_inventory_sha256(manifest.release_inventory)
        != manifest.dataset_release_sha256
    ):
        raise ManifestError("dataset release inventory hash mismatch")

    if manifest.excluded_samples:
        raise ManifestError("excluded_samples must be empty in schema version 1")

    inventory_by_path = {
        entry.relative_path: entry for entry in manifest.release_inventory
    }
    prepared_by_source: dict[str, PreparedArtifactRecord] = {}
    prepared_destinations: set[str] = set()
    for prepared in manifest.prepared_artifacts:
        if prepared.source_relative_path in prepared_by_source:
            raise ManifestError("duplicate prepared source path")
        if prepared.prepared_relative_path in prepared_destinations:
            raise ManifestError("duplicate prepared destination path")
        if prepared.prepared_relative_path in inventory_by_path:
            raise ManifestError("prepared destination must not enter raw inventory")
        prepared_by_source[prepared.source_relative_path] = prepared
        prepared_destinations.add(prepared.prepared_relative_path)

    if not manifest.samples:
        raise ManifestError("manifest must contain at least one sample")

    sample_by_id: dict[str, TemporalSampleRecord] = {}
    sample_position: dict[str, int] = {}
    seen_sequences: set[str] = set()
    current_sequence: str | None = None
    expected_n_t = 0
    sequence_split_value: dict[str, str] = {}
    tick_slices: dict[
        tuple[str, int, str, str],
        RawSliceRecord,
    ] = {}
    packet_provenance: dict[str, tuple[object, ...]] = {}
    packet_by_position: dict[tuple[str, int, str, str], str] = {}
    referenced_lidar_paths: set[str] = set()
    referenced_camera_paths: set[str] = set()

    for position, sample in enumerate(manifest.samples):
        if sample.sample_id in sample_by_id:
            raise ManifestError("sample IDs must be globally unique")
        sample_by_id[sample.sample_id] = sample
        sample_position[sample.sample_id] = position

        if sample.sequence_id != current_sequence:
            if sample.sequence_id in seen_sequences:
                raise ManifestError("sequence blocks must not reappear")
            seen_sequences.add(sample.sequence_id)
            current_sequence = sample.sequence_id
            expected_n_t = 0
            sequence_split_value[sample.sequence_id] = sample.split
        elif sequence_split_value[sample.sequence_id] != sample.split:
            raise ManifestError("one protocol sequence cannot cross splits")
        if sample.n_t != expected_n_t:
            raise ManifestError("sequence target ticks must be contiguous from zero")
        expected_n_t += 1
        if sample.tau_t_ms != sample.n_t * manifest.delta_t_ms:
            raise ManifestError("sample tau_t_ms does not match n_t")

        annotation = inventory_by_path.get(sample.annotation_path)
        if annotation is None or annotation.sha256 != sample.annotation_sha256:
            raise ManifestError("annotation path/hash mismatch")

        expected_slices = [
            (n_s, agent, modality)
            for n_s in range(
                max(0, sample.n_t - manifest.history_limit),
                sample.n_t + 1,
            )
            for agent, modality in _BRANCH_ORDER
        ]
        actual_slices = [
            (source.n_s, source.agent, source.modality)
            for source in sample.source_slices
        ]
        if actual_slices != expected_slices:
            raise ManifestError(
                "sample source slices have the wrong history grid/order"
            )
        packet_ids = [source.packet_id for source in sample.source_slices]
        if len(packet_ids) != len(set(packet_ids)):
            raise ManifestError("packet IDs must be unique within a sample")

        for source in sample.source_slices:
            if source.tau_s_ms != source.n_s * manifest.delta_t_ms:
                raise ManifestError("raw slice tau_s_ms does not match n_s")
            payload = inventory_by_path.get(source.relative_path)
            if payload is None:
                raise ManifestError("raw slice payload is absent from inventory")
            calibration = inventory_by_path.get(source.calibration_relative_path)
            if calibration is None or calibration.sha256 != source.calibration_sha256:
                raise ManifestError("primary calibration path/hash mismatch")
            if source.modality == "lidar":
                referenced_lidar_paths.add(source.relative_path)
                if source.relative_path not in prepared_by_source:
                    raise ManifestError(
                        "referenced LiDAR payload lacks prepared artifact"
                    )
            else:
                referenced_camera_paths.add(source.relative_path)
                if source.relative_path in prepared_by_source:
                    raise ManifestError(
                        "Camera payload must not have a prepared artifact"
                    )

            position_key = (
                sample.sequence_id,
                source.n_s,
                source.agent,
                source.modality,
            )
            existing_packet = packet_by_position.setdefault(
                position_key,
                source.packet_id,
            )
            if existing_packet != source.packet_id:
                raise ManifestError(
                    "one sequence/source position must identify one packet ID"
                )
            existing_slice = tick_slices.setdefault(position_key, source)
            if existing_slice != source:
                raise ManifestError("cross-target source provenance must be identical")
            prepared_mapping = (
                prepared_by_source.get(source.relative_path)
                if source.modality == "lidar"
                else None
            )
            provenance = (
                sample.sequence_id,
                source,
                prepared_mapping,
                payload,
                calibration,
            )
            existing_provenance = packet_provenance.setdefault(
                source.packet_id,
                provenance,
            )
            if existing_provenance != provenance:
                raise ManifestError("cross-target packet provenance must be identical")

    if set(prepared_by_source) != referenced_lidar_paths:
        raise ManifestError(
            "prepared artifacts must exactly cover referenced LiDAR payloads"
        )
    if set(prepared_by_source) & referenced_camera_paths:
        raise ManifestError("Camera payload must not have a prepared artifact")

    for sequence_id in seen_sequences:
        sequence_samples = [
            sample for sample in manifest.samples if sample.sequence_id == sequence_id
        ]
        final_tick = sequence_samples[-1].n_t
        for n_s in range(final_tick + 1):
            timestamps: list[int] = []
            for agent, modality in _BRANCH_ORDER:
                source = tick_slices.get((sequence_id, n_s, agent, modality))
                if source is None:
                    raise ManifestError(
                        "sequence source ticks must contain all four streams"
                    )
                timestamps.append(source.capture_timestamp_us)
            if max(timestamps) - min(timestamps) > (
                manifest.max_capture_skew_ms * 1000
            ):
                raise ManifestError("capture timestamp skew exceeds protocol limit")
            if n_s == 0:
                continue
            for agent, modality in _BRANCH_ORDER:
                previous = tick_slices[
                    (sequence_id, n_s - 1, agent, modality)
                ].capture_timestamp_us
                current = tick_slices[
                    (sequence_id, n_s, agent, modality)
                ].capture_timestamp_us
                interval = current - previous
                if interval <= 0:
                    raise ManifestError(
                        "capture timestamps must be strictly increasing"
                    )
                if not (
                    manifest.interval_min_ms * 1000
                    <= interval
                    <= manifest.interval_max_ms * 1000
                ):
                    raise ManifestError(
                        "same-stream interval lies outside sequence bounds"
                    )

    expected_history_count = sum(
        sample.split == "train" and sample.n_t >= manifest.history_limit
        for sample in manifest.samples
    )
    if manifest.history_eligible_train_count != expected_history_count:
        raise ManifestError("history_eligible_train_count mismatch")

    previous_boundary_position = -1
    seen_boundary_samples: set[str] = set()
    for boundary in manifest.sequence_splits:
        previous_id = boundary["previous_sample_id"]
        current_id = boundary["current_sample_id"]
        new_sequence_id = boundary["new_sequence_id"]
        assert isinstance(previous_id, str)
        assert isinstance(current_id, str)
        assert isinstance(new_sequence_id, str)
        if previous_id not in sample_by_id or current_id not in sample_by_id:
            raise ManifestError("sequence split references unknown sample")
        previous_sample = sample_by_id[previous_id]
        current_sample = sample_by_id[current_id]
        if current_sample.n_t != 0:
            raise ManifestError("sequence split current sample must have n_t zero")
        if current_sample.sequence_id != new_sequence_id:
            raise ManifestError("sequence split new_sequence_id mismatch")
        if previous_sample.sequence_id == current_sample.sequence_id:
            raise ManifestError("sequence split must cross protocol sequences")
        boundary_position = sample_position[current_id]
        if boundary_position != sample_position[previous_id] + 1:
            raise ManifestError(
                "sequence split previous and current samples must be adjacent"
            )
        if boundary_position <= previous_boundary_position:
            raise ManifestError("sequence_splits must follow sample order")
        previous_boundary_position = boundary_position
        if current_id in seen_boundary_samples:
            raise ManifestError("duplicate sequence split")
        seen_boundary_samples.add(current_id)
        triggers = boundary["triggers"]
        assert isinstance(triggers, tuple)
        if not triggers:
            raise ManifestError("sequence split triggers must be non-empty")
        trigger_order: list[tuple[str, str]] = []
        for trigger in triggers:
            assert isinstance(trigger, Mapping)
            branch = (trigger["agent"], trigger["modality"])
            assert isinstance(branch[0], str)
            assert isinstance(branch[1], str)
            trigger_order.append(branch)
            previous_timestamp = trigger["previous_capture_timestamp_us"]
            current_timestamp = trigger["current_capture_timestamp_us"]
            interval = trigger["interval_us"]
            assert isinstance(previous_timestamp, int)
            assert isinstance(current_timestamp, int)
            assert isinstance(interval, int)
            if current_timestamp <= previous_timestamp:
                raise ManifestError(
                    "sequence split timestamps must be strictly increasing"
                )
            if interval != current_timestamp - previous_timestamp:
                raise ManifestError("sequence split interval mismatch")
            if (
                manifest.interval_min_ms * 1000
                <= interval
                <= manifest.interval_max_ms * 1000
            ):
                raise ManifestError(
                    "sequence split trigger interval must be out of bounds"
                )
            previous_slice = next(
                source
                for source in previous_sample.source_slices
                if source.n_s == previous_sample.n_t
                and (source.agent, source.modality) == branch
            )
            current_slice = next(
                source
                for source in current_sample.source_slices
                if source.n_s == current_sample.n_t
                and (source.agent, source.modality) == branch
            )
            if (
                previous_slice.capture_timestamp_us != previous_timestamp
                or current_slice.capture_timestamp_us != current_timestamp
            ):
                raise ManifestError(
                    "sequence split trigger timestamp provenance mismatch"
                )
        if trigger_order != sorted(
            trigger_order,
            key=_BRANCH_ORDER.index,
        ) or len(trigger_order) != len(set(trigger_order)):
            raise ManifestError(
                "sequence split triggers must be unique and branch-ordered"
            )

    manifest_payload = _plain(manifest)
    assert isinstance(manifest_payload, Mapping)
    if content_sha256(manifest_payload) != manifest.content_sha256:
        raise ManifestError("temporal manifest content hash mismatch")


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ManifestError(f"non-finite JSON constant is forbidden: {value}")


def load_temporal_manifest(
    path: Path,
    expected_split_hash: str,
    allow_fixture: bool = False,
) -> TemporalManifest:
    expected_hash = _sha256(expected_split_hash, "expected_split_hash")
    _exact_bool(allow_fixture, "allow_fixture")
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ManifestError("unable to read temporal manifest") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ManifestError("temporal manifest must be strict UTF-8") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except ManifestError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ManifestError("invalid temporal manifest JSON") from error
    payload_mapping = _expect_fields(
        payload,
        _TOP_LEVEL_FIELDS,
        "temporal manifest",
    )
    try:
        if raw != canonical_json_bytes(payload_mapping):
            raise ManifestError("temporal manifest JSON is not canonical")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ManifestError("temporal manifest JSON is not canonical") from error

    payload_split_hash = _sha256(
        payload_mapping["split_sha256"],
        "split_sha256",
    )
    if payload_split_hash != expected_hash:
        raise ManifestError("split hash mismatch")
    scope = payload_mapping["protocol_scope"]
    if scope == "controlled":
        if (
            expected_hash != OFFICIAL_COOPERATIVE_SPLIT_SHA256
            or payload_split_hash != OFFICIAL_COOPERATIVE_SPLIT_SHA256
        ):
            raise ManifestError(
                "controlled split must use the official cooperative hash"
            )
    elif scope == "fixture":
        if not allow_fixture:
            raise ManifestError("fixture manifest is not controlled-eligible")
    else:
        raise ManifestError("protocol_scope must be controlled or fixture")

    release_value = payload_mapping["release_inventory"]
    prepared_value = payload_mapping["prepared_artifacts"]
    sequence_splits_value = payload_mapping["sequence_splits"]
    excluded_value = payload_mapping["excluded_samples"]
    samples_value = payload_mapping["samples"]
    for value, name in (
        (release_value, "release_inventory"),
        (prepared_value, "prepared_artifacts"),
        (sequence_splits_value, "sequence_splits"),
        (excluded_value, "excluded_samples"),
        (samples_value, "samples"),
    ):
        if not isinstance(value, list):
            raise ManifestError(f"{name} must be an array")

    expected_inventory_hash = release_inventory_sha256(release_value)
    actual_inventory_hash = _sha256(
        payload_mapping["dataset_release_sha256"],
        "dataset_release_sha256",
    )
    if expected_inventory_hash != actual_inventory_hash:
        raise ManifestError("dataset release inventory hash mismatch")
    actual_content_hash = _sha256(
        payload_mapping["content_sha256"],
        "content_sha256",
    )
    if content_sha256(payload_mapping) != actual_content_hash:
        raise ManifestError("temporal manifest content hash mismatch")

    return TemporalManifest(
        schema_version=payload_mapping["schema_version"],  # type: ignore[arg-type]
        protocol_scope=scope,  # type: ignore[arg-type]
        delta_t_ms=payload_mapping["delta_t_ms"],  # type: ignore[arg-type]
        history_limit=payload_mapping["history_limit"],  # type: ignore[arg-type]
        interval_min_ms=payload_mapping["interval_min_ms"],  # type: ignore[arg-type]
        interval_max_ms=payload_mapping["interval_max_ms"],  # type: ignore[arg-type]
        max_capture_skew_ms=payload_mapping["max_capture_skew_ms"],  # type: ignore[arg-type]
        split_sha256=payload_split_hash,
        dataset_release_sha256=actual_inventory_hash,
        release_inventory=tuple(
            _parse_inventory(item, f"release_inventory[{index}]")
            for index, item in enumerate(release_value)
        ),
        prepared_artifacts=tuple(
            _parse_prepared(item, f"prepared_artifacts[{index}]")
            for index, item in enumerate(prepared_value)
        ),
        history_eligible_train_count=payload_mapping["history_eligible_train_count"],  # type: ignore[arg-type]
        sequence_splits=tuple(
            _parse_sequence_split(item, f"sequence_splits[{index}]")
            for index, item in enumerate(sequence_splits_value)
        ),
        excluded_samples=tuple(
            _parse_excluded(item, f"excluded_samples[{index}]")
            for index, item in enumerate(excluded_value)
        ),
        samples=tuple(
            _parse_sample(item, f"samples[{index}]")
            for index, item in enumerate(samples_value)
        ),
        content_sha256=actual_content_hash,
    )
