"""Strict V2X-Seq-SPD calibration and coordinate boundary.

The official pinned DAIR-V2X implementation documents three calibration
artifacts for cooperative SPD LiDAR geometry:

* infrastructure virtual LiDAR -> world: top-level ``rotation`` and
  ``translation``;
* vehicle LiDAR -> NovAtel: a top-level ``transform`` object containing
  ``rotation`` and ``translation``;
* vehicle NovAtel -> world: top-level ``rotation`` and ``translation``.

The JSON payloads do not carry self-describing coordinate-frame names.  A
direct calibration document is therefore ambiguous unless its role is bound
to the official dataset-relative path and expected frame id.  This module
requires both and rejects extra fields instead of guessing a schema.

For SPD, the official converter applies ``system_error_offset`` after the
three coordinate transforms, in the target vehicle-LiDAR axes.  That differs
from the older DAIR-V2X-C converter.  The order is explicit and immutable here.

Only standard-library code is used.  This module converts points and emits a
backend payload; it does not read point clouds, implement CenterPoint, or
claim evaluator parity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from .adapter import AdapterContractError, COORDINATE_FRAME


DAIR_V2X_REPOSITORY = "https://github.com/AIR-THU/DAIR-V2X"
DAIR_V2X_COMMIT = "c885c54af0c34bc515fa9ca8b5e8fda76a15462c"
V2XSEQ_REPOSITORY = "https://github.com/AIR-THU/DAIR-V2X-Seq"
V2XSEQ_COMMIT = "8a5b4f82da5ab5c6eeacbf961aafbf2c883f437c"

INFRASTRUCTURE_LIDAR_FRAME = (
    "V2X-Seq SPD infrastructure virtual LiDAR at infrastructure capture time"
)
WORLD_FRAME = "V2X-Seq SPD world"
VEHICLE_NOVATEL_FRAME = "V2X-Seq SPD vehicle NovAtel at vehicle capture time"
VEHICLE_LIDAR_FRAME = COORDINATE_FRAME
POINT_EQUATION = "p_target = R_source_to_target @ p_source + t_source_to_target"

_SINGULAR_TOLERANCE = 1e-12
_OBSERVATION_FIELDS = frozenset(
    {
        "infrastructure_frame_id",
        "infrastructure_lidar_path",
        "infrastructure_sequence_id",
        "infrastructure_timestamp",
        "vehicle_frame_id",
        "vehicle_lidar_path",
        "vehicle_sequence_id",
        "vehicle_timestamp",
    }
)

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


class SpdCalibrationRole(str, Enum):
    """The only three extrinsic directions evidenced by the SPD source."""

    INFRASTRUCTURE_LIDAR_TO_WORLD = "infrastructure_lidar_to_world"
    VEHICLE_LIDAR_TO_NOVATEL = "vehicle_lidar_to_novatel"
    VEHICLE_NOVATEL_TO_WORLD = "vehicle_novatel_to_world"


@dataclass(frozen=True)
class _RoleSpec:
    source_frame: str
    target_frame: str
    path_template: str
    wrapped: bool


_ROLE_SPECS = {
    SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD: _RoleSpec(
        source_frame=INFRASTRUCTURE_LIDAR_FRAME,
        target_frame=WORLD_FRAME,
        path_template=(
            "infrastructure-side/calib/virtuallidar_to_world/{frame_id}.json"
        ),
        wrapped=False,
    ),
    SpdCalibrationRole.VEHICLE_LIDAR_TO_NOVATEL: _RoleSpec(
        source_frame=VEHICLE_LIDAR_FRAME,
        target_frame=VEHICLE_NOVATEL_FRAME,
        path_template="vehicle-side/calib/lidar_to_novatel/{frame_id}.json",
        wrapped=True,
    ),
    SpdCalibrationRole.VEHICLE_NOVATEL_TO_WORLD: _RoleSpec(
        source_frame=VEHICLE_NOVATEL_FRAME,
        target_frame=WORLD_FRAME,
        path_template="vehicle-side/calib/novatel_to_world/{frame_id}.json",
        wrapped=False,
    ),
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterContractError(f"{label} must be a mapping")
    return value


def _strict_mapping(
    value: Any, *, fields: frozenset[str], label: str
) -> Mapping[str, Any]:
    row = _mapping(value, label)
    observed = frozenset(row)
    if observed != fields:
        missing = sorted(fields - observed)
        unknown = sorted(repr(key) for key in observed - fields)
        raise AdapterContractError(
            f"{label} fields do not match the official schema exactly; "
            f"missing={missing}, unknown={unknown}"
        )
    return row


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterContractError(f"{label} must be a non-empty string")
    return value


def _frame_id(value: Any, label: str) -> str:
    result = _string(value, label)
    if not result.isascii() or not result.isdigit():
        raise AdapterContractError(f"{label} must contain only ASCII digits")
    return result


def _timestamp(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterContractError(f"{label} must be an integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterContractError(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise AdapterContractError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise AdapterContractError(f"{label} must be finite")
    return result


def _vector3(value: Any, label: str) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AdapterContractError(
            f"{label} must be a length-3 vector or a 3x1 column vector"
        )
    if all(isinstance(item, (list, tuple)) for item in value):
        if any(len(item) != 1 for item in value):
            raise AdapterContractError(
                f"{label} must be a length-3 vector or a 3x1 column vector"
            )
        values = [item[0] for item in value]
    elif any(isinstance(item, (list, tuple)) for item in value):
        raise AdapterContractError(f"{label} mixes scalar and nested values")
    else:
        values = list(value)
    return tuple(
        _finite(item, f"{label}[{index}]") for index, item in enumerate(values)
    )  # type: ignore[return-value]


def _matrix3(value: Any, label: str) -> Matrix3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AdapterContractError(f"{label} must be a 3x3 matrix")
    rows: list[Vector3] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise AdapterContractError(f"{label} must be a 3x3 matrix")
        rows.append(
            tuple(
                _finite(item, f"{label}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )  # type: ignore[arg-type]
        )
    result = tuple(rows)  # type: ignore[assignment]
    _validate_transform_matrix(result, label)
    return result


def _determinant(matrix: Matrix3) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validate_transform_matrix(matrix: Matrix3, label: str) -> None:
    determinant = _determinant(matrix)
    if not math.isfinite(determinant):
        raise AdapterContractError(f"{label} determinant must be finite")
    if abs(determinant) <= _SINGULAR_TOLERANCE:
        raise AdapterContractError(f"{label} is singular")
    if determinant < 0:
        raise AdapterContractError(f"{label} reverses coordinate handedness")


def _inverse_matrix(matrix: Matrix3) -> Matrix3:
    determinant = _determinant(matrix)
    if not math.isfinite(determinant):
        raise AdapterContractError("transform matrix determinant must be finite")
    if abs(determinant) <= _SINGULAR_TOLERANCE:
        raise AdapterContractError("transform matrix is singular")
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    inverse_determinant = 1.0 / determinant
    return (
        (
            (e * i - f * h) * inverse_determinant,
            (c * h - b * i) * inverse_determinant,
            (b * f - c * e) * inverse_determinant,
        ),
        (
            (f * g - d * i) * inverse_determinant,
            (a * i - c * g) * inverse_determinant,
            (c * d - a * f) * inverse_determinant,
        ),
        (
            (d * h - e * g) * inverse_determinant,
            (b * g - a * h) * inverse_determinant,
            (a * e - b * d) * inverse_determinant,
        ),
    )


def _matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    right_columns = tuple(zip(*right, strict=True))
    return tuple(
        tuple(
            sum(a * b for a, b in zip(row, column, strict=True))
            for column in right_columns
        )
        for row in left
    )  # type: ignore[return-value]


def _matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(
        _finite(
            sum(a * b for a, b in zip(row, vector, strict=True)),
            "transformed coordinate",
        )
        for row in matrix
    )  # type: ignore[return-value]


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(
        _finite(a + b, "transformed coordinate")
        for a, b in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def _negate(vector: Vector3) -> Vector3:
    return tuple(-value for value in vector)  # type: ignore[return-value]


@dataclass(frozen=True)
class CoordinateTransform:
    """A direction-tagged invertible transform with ``R @ p + t`` semantics."""

    source_frame: str
    target_frame: str
    rotation: Matrix3
    translation: Vector3

    def __post_init__(self) -> None:
        source = _string(self.source_frame, "transform source_frame")
        target = _string(self.target_frame, "transform target_frame")
        rotation = _matrix3(self.rotation, "transform rotation")
        translation = _vector3(self.translation, "transform translation")
        object.__setattr__(self, "source_frame", source)
        object.__setattr__(self, "target_frame", target)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def transform_point(self, point: Sequence[int | float]) -> Vector3:
        """Transform one point; labels and boxes are deliberately out of scope."""

        normalized = _vector3(point, "point")
        return _add(_matvec(self.rotation, normalized), self.translation)

    def then(self, following: CoordinateTransform) -> CoordinateTransform:
        """Apply this transform, then ``following``, checking frame direction."""

        if not isinstance(following, CoordinateTransform):
            raise AdapterContractError(
                "following transform must be a CoordinateTransform"
            )
        if self.target_frame != following.source_frame:
            raise AdapterContractError(
                "coordinate direction mismatch: "
                f"{self.source_frame!r}->{self.target_frame!r} cannot be followed by "
                f"{following.source_frame!r}->{following.target_frame!r}"
            )
        return CoordinateTransform(
            source_frame=self.source_frame,
            target_frame=following.target_frame,
            rotation=_matmul(following.rotation, self.rotation),
            translation=_add(
                _matvec(following.rotation, self.translation),
                following.translation,
            ),
        )

    def inverse(self) -> CoordinateTransform:
        """Return the unambiguous reverse direction using a full 3x3 inverse."""

        rotation = _inverse_matrix(self.rotation)
        return CoordinateTransform(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            rotation=rotation,
            translation=_negate(_matvec(rotation, self.translation)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "point_equation": POINT_EQUATION,
            "rotation": [list(row) for row in self.rotation],
            "translation": list(self.translation),
        }


def _canonical_relative_path(value: Any, label: str) -> str:
    raw = _string(value, label)
    if "\\" in raw:
        raise AdapterContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AdapterContractError(f"{label} must be a canonical relative path")
    if str(path) != raw:
        raise AdapterContractError(f"{label} must be a canonical relative path")
    return raw


def _official_lidar_path(value: Any, *, label: str, side: str, frame_id: str) -> str:
    path = Path(_string(value, label))
    if not path.is_absolute() or ".." in path.parts:
        raise AdapterContractError(f"{label} must be a canonical absolute path")
    expected_suffix = (side, "velodyne", f"{frame_id}.pcd")
    if tuple(path.parts[-3:]) != expected_suffix:
        raise AdapterContractError(
            f"{label} must match the official {side}/velodyne frame layout"
        )
    return str(path)


def parse_spd_calibration(
    document: Mapping[str, Any],
    *,
    role: SpdCalibrationRole,
    relative_path: str,
    expected_frame_id: str,
) -> CoordinateTransform:
    """Parse one role- and path-bound official SPD calibration document.

    The two direct schemas are structurally identical, so direction cannot be
    inferred from bytes alone.  Requiring the exact official path makes that
    ambiguity explicit and fail-closed.
    """

    if not isinstance(role, SpdCalibrationRole):
        raise AdapterContractError("calibration role must be a SpdCalibrationRole")
    frame_id = _frame_id(expected_frame_id, "calibration expected_frame_id")
    path = _canonical_relative_path(relative_path, "calibration relative_path")
    spec = _ROLE_SPECS[role]
    expected_path = spec.path_template.format(frame_id=frame_id)
    if path != expected_path:
        raise AdapterContractError(
            f"calibration path does not prove role {role.value!r}; "
            f"expected {expected_path!r}, observed {path!r}"
        )

    if spec.wrapped:
        root = _strict_mapping(
            document,
            fields=frozenset({"transform"}),
            label=f"{role.value} calibration",
        )
        body = _strict_mapping(
            root["transform"],
            fields=frozenset({"rotation", "translation"}),
            label=f"{role.value} calibration transform",
        )
    else:
        body = _strict_mapping(
            document,
            fields=frozenset({"rotation", "translation"}),
            label=f"{role.value} calibration",
        )
    return CoordinateTransform(
        source_frame=spec.source_frame,
        target_frame=spec.target_frame,
        rotation=_matrix3(body["rotation"], f"{role.value} rotation"),
        translation=_vector3(body["translation"], f"{role.value} translation"),
    )


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdapterContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AdapterContractError(f"non-finite JSON constant {value!r} is forbidden")


def loads_spd_calibration_json(
    payload: str,
    *,
    role: SpdCalibrationRole,
    relative_path: str,
    expected_frame_id: str,
) -> CoordinateTransform:
    """Strict JSON-text entry point that also rejects duplicate object keys."""

    if not isinstance(payload, str):
        raise AdapterContractError("calibration JSON payload must be text")
    try:
        document = json.loads(
            payload,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except AdapterContractError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise AdapterContractError("calibration JSON is invalid") from exc
    return parse_spd_calibration(
        _mapping(document, "calibration JSON root"),
        role=role,
        relative_path=relative_path,
        expected_frame_id=expected_frame_id,
    )


def _system_error_offset(value: Any) -> tuple[float, float]:
    row = _strict_mapping(
        value,
        fields=frozenset({"delta_x", "delta_y"}),
        label="SPD system_error_offset",
    )
    return (
        _finite(row["delta_x"], "SPD system_error_offset.delta_x"),
        _finite(row["delta_y"], "SPD system_error_offset.delta_y"),
    )


@dataclass(frozen=True)
class SpdCalibrationBridge:
    """Frozen infrastructure-LiDAR to vehicle-LiDAR SPD geometry."""

    infrastructure_frame_id: str
    vehicle_frame_id: str
    raw_transform: CoordinateTransform
    corrected_transform: CoordinateTransform
    system_error_offset_vehicle_lidar_xy: tuple[float, float]
    artifact_paths: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        infrastructure_frame_id = _frame_id(
            self.infrastructure_frame_id, "bridge infrastructure_frame_id"
        )
        vehicle_frame_id = _frame_id(self.vehicle_frame_id, "bridge vehicle_frame_id")
        if not isinstance(self.raw_transform, CoordinateTransform) or not isinstance(
            self.corrected_transform, CoordinateTransform
        ):
            raise AdapterContractError(
                "bridge transforms must be CoordinateTransform values"
            )
        expected_direction = (INFRASTRUCTURE_LIDAR_FRAME, VEHICLE_LIDAR_FRAME)
        for label, transform in (
            ("raw", self.raw_transform),
            ("corrected", self.corrected_transform),
        ):
            if (transform.source_frame, transform.target_frame) != expected_direction:
                raise AdapterContractError(
                    f"bridge {label} transform direction must be infrastructure to vehicle"
                )
        if self.raw_transform.rotation != self.corrected_transform.rotation:
            raise AdapterContractError("SPD offset may not change calibration rotation")
        offset = self.system_error_offset_vehicle_lidar_xy
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise AdapterContractError(
                "bridge system error offset must contain x and y"
            )
        normalized_offset = (
            _finite(offset[0], "bridge system error offset x"),
            _finite(offset[1], "bridge system error offset y"),
        )
        expected_translation = _add(
            self.raw_transform.translation,
            (normalized_offset[0], normalized_offset[1], 0.0),
        )
        if any(
            abs(observed - expected) > 1e-12
            for observed, expected in zip(
                self.corrected_transform.translation,
                expected_translation,
                strict=True,
            )
        ):
            raise AdapterContractError(
                "SPD offset must be applied after the coordinate chain in vehicle-LiDAR axes"
            )
        expected_paths = {
            role.value: _ROLE_SPECS[role].path_template.format(
                frame_id=(
                    infrastructure_frame_id
                    if role is SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD
                    else vehicle_frame_id
                )
            )
            for role in SpdCalibrationRole
        }
        if not isinstance(self.artifact_paths, tuple) or len(self.artifact_paths) != 3:
            raise AdapterContractError(
                "bridge artifact paths are incomplete or ambiguous"
            )
        observed_paths: dict[str, str] = {}
        for index, entry in enumerate(self.artifact_paths):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise AdapterContractError(
                    f"bridge artifact_paths[{index}] must be a role/path pair"
                )
            role = _string(entry[0], f"bridge artifact_paths[{index}] role")
            path = _canonical_relative_path(
                entry[1], f"bridge artifact_paths[{index}] path"
            )
            if role in observed_paths:
                raise AdapterContractError("bridge artifact path roles must be unique")
            observed_paths[role] = path
        if observed_paths != expected_paths:
            raise AdapterContractError(
                "bridge artifact paths are incomplete or ambiguous"
            )
        normalized_paths = tuple(
            (role.value, observed_paths[role.value]) for role in SpdCalibrationRole
        )
        object.__setattr__(self, "infrastructure_frame_id", infrastructure_frame_id)
        object.__setattr__(self, "vehicle_frame_id", vehicle_frame_id)
        object.__setattr__(
            self, "system_error_offset_vehicle_lidar_xy", normalized_offset
        )
        object.__setattr__(self, "artifact_paths", normalized_paths)

    def transform_point(self, point: Sequence[int | float]) -> Vector3:
        return self.corrected_transform.transform_point(point)

    def inverse_transform_point(self, point: Sequence[int | float]) -> Vector3:
        return self.corrected_transform.inverse().transform_point(point)

    def to_backend_payload(self) -> dict[str, Any]:
        """Emit a label-free, direction-explicit JSON-like backend payload."""

        return {
            "schema_version": 1,
            "transform_kind": (
                "v2xseq_spd_infrastructure_virtual_lidar_to_vehicle_lidar"
            ),
            "infrastructure_frame_id": self.infrastructure_frame_id,
            "vehicle_frame_id": self.vehicle_frame_id,
            "transform": self.corrected_transform.to_payload(),
            "system_error_offset": {
                "application_frame": VEHICLE_LIDAR_FRAME,
                "application_order": "after_coordinate_chain",
                "delta_x": self.system_error_offset_vehicle_lidar_xy[0],
                "delta_y": self.system_error_offset_vehicle_lidar_xy[1],
            },
            "calibration_paths": dict(self.artifact_paths),
        }

    def bind_inference_observation(
        self, observation: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Bind geometry to a strict observation-only cooperative input.

        Exact fields are required so labels, annotations, targets, or other
        future-ground-truth aliases cannot be forwarded to an inference
        backend through this boundary.
        """

        row = _strict_mapping(
            observation,
            fields=_OBSERVATION_FIELDS,
            label="SPD cooperative inference observation",
        )
        infrastructure_frame_id = _frame_id(
            row["infrastructure_frame_id"], "observation infrastructure_frame_id"
        )
        vehicle_frame_id = _frame_id(
            row["vehicle_frame_id"], "observation vehicle_frame_id"
        )
        if infrastructure_frame_id != self.infrastructure_frame_id:
            raise AdapterContractError(
                "observation infrastructure frame does not match calibration bridge"
            )
        if vehicle_frame_id != self.vehicle_frame_id:
            raise AdapterContractError(
                "observation vehicle frame does not match calibration bridge"
            )
        vehicle_lidar_path = _official_lidar_path(
            row["vehicle_lidar_path"],
            label="observation vehicle_lidar_path",
            side="vehicle-side",
            frame_id=vehicle_frame_id,
        )
        infrastructure_lidar_path = _official_lidar_path(
            row["infrastructure_lidar_path"],
            label="observation infrastructure_lidar_path",
            side="infrastructure-side",
            frame_id=infrastructure_frame_id,
        )
        return {
            "vehicle_sequence_id": _frame_id(
                row["vehicle_sequence_id"], "observation vehicle_sequence_id"
            ),
            "infrastructure_sequence_id": _frame_id(
                row["infrastructure_sequence_id"],
                "observation infrastructure_sequence_id",
            ),
            "vehicle_frame_id": vehicle_frame_id,
            "infrastructure_frame_id": infrastructure_frame_id,
            "vehicle_timestamp": _timestamp(
                row["vehicle_timestamp"], "observation vehicle_timestamp"
            ),
            "infrastructure_timestamp": _timestamp(
                row["infrastructure_timestamp"],
                "observation infrastructure_timestamp",
            ),
            "vehicle_lidar_path": vehicle_lidar_path,
            "infrastructure_lidar_path": infrastructure_lidar_path,
            "coordinate_frame": VEHICLE_LIDAR_FRAME,
            "fusion_scope": "vehicle_infrastructure_lidar",
            "infrastructure_to_vehicle_lidar": self.to_backend_payload(),
        }


def build_spd_calibration_bridge(
    *,
    infrastructure_frame_id: str,
    vehicle_frame_id: str,
    infrastructure_lidar_to_world_document: Mapping[str, Any],
    infrastructure_lidar_to_world_path: str,
    vehicle_lidar_to_novatel_document: Mapping[str, Any],
    vehicle_lidar_to_novatel_path: str,
    vehicle_novatel_to_world_document: Mapping[str, Any],
    vehicle_novatel_to_world_path: str,
    system_error_offset: Mapping[str, Any],
) -> SpdCalibrationBridge:
    """Build the official SPD I2V chain with target-frame error correction."""

    infrastructure_frame = _frame_id(infrastructure_frame_id, "infrastructure_frame_id")
    vehicle_frame = _frame_id(vehicle_frame_id, "vehicle_frame_id")
    paths = {
        SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD: (
            infrastructure_lidar_to_world_path
        ),
        SpdCalibrationRole.VEHICLE_LIDAR_TO_NOVATEL: (vehicle_lidar_to_novatel_path),
        SpdCalibrationRole.VEHICLE_NOVATEL_TO_WORLD: (vehicle_novatel_to_world_path),
    }
    infrastructure_to_world = parse_spd_calibration(
        infrastructure_lidar_to_world_document,
        role=SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD,
        relative_path=paths[SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD],
        expected_frame_id=infrastructure_frame,
    )
    vehicle_to_novatel = parse_spd_calibration(
        vehicle_lidar_to_novatel_document,
        role=SpdCalibrationRole.VEHICLE_LIDAR_TO_NOVATEL,
        relative_path=paths[SpdCalibrationRole.VEHICLE_LIDAR_TO_NOVATEL],
        expected_frame_id=vehicle_frame,
    )
    novatel_to_world = parse_spd_calibration(
        vehicle_novatel_to_world_document,
        role=SpdCalibrationRole.VEHICLE_NOVATEL_TO_WORLD,
        relative_path=paths[SpdCalibrationRole.VEHICLE_NOVATEL_TO_WORLD],
        expected_frame_id=vehicle_frame,
    )
    raw_transform = infrastructure_to_world.then(novatel_to_world.inverse()).then(
        vehicle_to_novatel.inverse()
    )
    delta_x, delta_y = _system_error_offset(system_error_offset)
    corrected_transform = CoordinateTransform(
        source_frame=raw_transform.source_frame,
        target_frame=raw_transform.target_frame,
        rotation=raw_transform.rotation,
        translation=_add(raw_transform.translation, (delta_x, delta_y, 0.0)),
    )
    artifact_paths = tuple(
        (role.value, _canonical_relative_path(paths[role], f"{role.value} path"))
        for role in SpdCalibrationRole
    )
    return SpdCalibrationBridge(
        infrastructure_frame_id=infrastructure_frame,
        vehicle_frame_id=vehicle_frame,
        raw_transform=raw_transform,
        corrected_transform=corrected_transform,
        system_error_offset_vehicle_lidar_xy=(delta_x, delta_y),
        artifact_paths=artifact_paths,
    )
