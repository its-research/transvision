"""Strict fixed-detection JSONL contract for V2X-Seq SPD.

This module is deliberately independent of TransVision, MMDetection3D, ClearML,
and dataset libraries.  It validates only already-materialized JSON-like frame
records and provides deterministic JSON Lines serialization and SHA-256
digests.  It never loads a model, reads V2X-Seq data, or evaluates predictions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
CONTRACT_ID = "RTPV2X-FIXED-DETECTIONS-v1"
RECORD_KIND = "fixed_detection_frame"
DATASET_NAME = "V2X-Seq-SPD"
REPOSITORY_URL = "https://github.com/its-research/transvision.git"
MODEL_IDS = frozenset({"coformernet_controlled_adaptation", "resilient_v2x"})
CONFIG_PATH_BY_MODEL = {
    "coformernet_controlled_adaptation": (
        "configs/resilient_v2x/baselines/coformernet.py"
    ),
    "resilient_v2x": "configs/resilient_v2x/dair_resilient_v2x.py",
}
SPLITS = frozenset({"train", "validation", "test"})
MAX_JSONL_BYTES = 512 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024
MAX_DOCUMENT_RECORDS = 20_000
MAX_NESTING_DEPTH = 32
MAX_SIGNED_INT64 = 2**63 - 1
MIN_BOX_DIMENSION_M = 1e-3
MAX_BOX_DIMENSION_M = 100.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "record_kind",
        "scientific_claim_allowed",
        "dataset",
        "frame",
        "coordinate_system",
        "source",
        "policy",
        "detections",
    }
)
DATASET_FIELDS = frozenset(
    {
        "name",
        "release_id",
        "split",
        "dataset_manifest_sha256",
        "split_manifest_sha256",
    }
)
FRAME_FIELDS = frozenset(
    {
        "sequence_id",
        "vehicle_frame_id",
        "infrastructure_frame_id",
        "vehicle_capture_time_ns",
        "infrastructure_capture_time_ns",
        "decision_time_ns",
    }
)
SOURCE_FIELDS = frozenset(
    {
        "repository_url",
        "implementation_revision",
        "implementation_tree_sha256",
        "model_id",
        "config_path",
        "config_sha256",
        "checkpoint_role",
        "checkpoint_sha256",
        "teacher_checkpoint_sha256",
        "inference_adapter_sha256",
        "environment_manifest_sha256",
    }
)
DETECTION_FIELDS = frozenset(
    {
        "detection_id",
        "source_label_id",
        "source_class_name",
        "class_name",
        "box_3d",
        "score",
    }
)

FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "annotation",
        "annotations",
        "evaluation",
        "evaluator",
        "ground_truth",
        "groundtruth",
        "gt",
        "gt_box",
        "gt_boxes",
        "gt_label",
        "gt_labels",
        "label",
        "labels",
        "metric",
        "metrics",
        "target",
        "targets",
        "track",
        "track_id",
        "tracking_id",
        "tracks",
        "truth",
    }
)

_COORDINATE_SYSTEM: dict[str, Any] = {
    "frame": "v2x_seq_spd_vehicle_lidar_at_vehicle_capture_time",
    "axes": {"x": "forward", "y": "left", "z": "up"},
    "length_unit": "metre",
    "angle_unit": "radian",
    "box_3d_order": "x,y,z,l,w,h,yaw",
    "z_reference": "bottom_center",
    "yaw_convention": (
        "right_handed_counterclockwise_about_positive_z_zero_along_positive_x"
    ),
}

_FIXED_POLICY: dict[str, Any] = {
    "class_mapping": {
        "source_label_id": 0,
        "source_class_name": "Car",
        "output_class_name": "vehicle",
    },
    "roi": {
        "rule": "box_bottom_center_inclusive",
        "x_m": [0.0, 80.0],
        "y_m": [-40.0, 40.0],
        "z_m": [-3.0, 1.0],
    },
    "score": {
        "comparison": "greater_than_or_equal",
        "threshold": 0.05,
        "maximum": 1.0,
    },
    "nms": {
        "type": "rotate_nms",
        "iou_threshold": 0.01,
        "nms_across_levels": False,
        "pre_max_detections": 2000,
        "max_detections_per_frame": 100,
    },
}


class FixedDetectionContractError(ValueError):
    """Raised when a fixed-detection artifact is ambiguous or unsafe."""


def expected_coordinate_system() -> dict[str, Any]:
    """Return a detached copy of the frozen coordinate-system record."""

    return _json_clone(_COORDINATE_SYSTEM)


def expected_policy() -> dict[str, Any]:
    """Return a detached copy of the frozen post-processing policy."""

    return _json_clone(_FIXED_POLICY)


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _reject_constant(value: str) -> None:
    raise FixedDetectionContractError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixedDetectionContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_forbidden_fields(
    value: Any, *, label: str = "record", depth: int = 0
) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise FixedDetectionContractError(f"{label} exceeds the maximum nesting depth")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FixedDetectionContractError(
                    f"{label} contains a non-string object key"
                )
            normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in FORBIDDEN_FIELD_NAMES:
                raise FixedDetectionContractError(
                    f"{label} contains forbidden field {key!r}"
                )
            _reject_forbidden_fields(item, label=f"{label}.{key}", depth=depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, label=f"{label}[{index}]", depth=depth + 1)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixedDetectionContractError(f"{label} must be a JSON object")
    return value


def _exact_fields(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    row = _mapping(value, label)
    observed = set(row)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise FixedDetectionContractError(
            f"{label} fields do not match the contract ({', '.join(details)})"
        )
    return row


def _string(value: Any, label: str, *, max_length: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > max_length
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise FixedDetectionContractError(
            f"{label} must be a trimmed non-empty string without control characters"
        )
    return value


def _identifier(value: Any, label: str) -> str:
    result = _string(value, label, max_length=128)
    if not _IDENTIFIER_RE.fullmatch(result):
        raise FixedDetectionContractError(
            f"{label} contains characters outside the identifier contract"
        )
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise FixedDetectionContractError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        raise FixedDetectionContractError(
            f"{label} must be 40 lowercase hexadecimal characters"
        )
    return value


def _nonnegative_int64(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_SIGNED_INT64
    ):
        raise FixedDetectionContractError(
            f"{label} must be a non-negative signed 64-bit integer"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixedDetectionContractError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FixedDetectionContractError(f"{label} exceeds the numeric bound") from exc
    if not math.isfinite(result):
        raise FixedDetectionContractError(f"{label} must be a finite number")
    # JSON distinguishes the spellings -0.0 and 0.0 even though downstream
    # geometry does not. Canonicalize signed zero so equivalent boxes cannot
    # acquire different content hashes.
    return 0.0 if result == 0.0 else result


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise FixedDetectionContractError(
            f"{label} must be a JSON array with shape [{length}]"
        )
    return [
        _finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]


def _repository_url(value: Any, label: str) -> str:
    result = _string(value, label, max_length=2048)
    try:
        parsed = urlsplit(result)
        hostname = parsed.hostname
    except ValueError as exc:
        raise FixedDetectionContractError(f"{label} is not a valid HTTPS URL") from exc
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise FixedDetectionContractError(
            f"{label} must be an HTTPS URL without credentials, query, or fragment"
        )
    return result


def _relative_python_path(value: Any, label: str) -> str:
    result = _string(value, label, max_length=512)
    if "\\" in result:
        raise FixedDetectionContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(result)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or path.suffix != ".py"
    ):
        raise FixedDetectionContractError(
            f"{label} must be a safe relative POSIX .py path"
        )
    return result


def _validate_dataset(value: Any) -> dict[str, Any]:
    row = _exact_fields(value, DATASET_FIELDS, "record.dataset")
    if row["name"] != DATASET_NAME:
        raise FixedDetectionContractError(
            f"record.dataset.name must be {DATASET_NAME!r}"
        )
    split = _string(row["split"], "record.dataset.split", max_length=10)
    if split not in SPLITS:
        raise FixedDetectionContractError(
            "record.dataset.split must be train, validation, or test"
        )
    return {
        "name": DATASET_NAME,
        "release_id": _identifier(row["release_id"], "record.dataset.release_id"),
        "split": split,
        "dataset_manifest_sha256": _sha256(
            row["dataset_manifest_sha256"],
            "record.dataset.dataset_manifest_sha256",
        ),
        "split_manifest_sha256": _sha256(
            row["split_manifest_sha256"],
            "record.dataset.split_manifest_sha256",
        ),
    }


def _validate_frame(value: Any) -> dict[str, Any]:
    row = _exact_fields(value, FRAME_FIELDS, "record.frame")
    vehicle_time = _nonnegative_int64(
        row["vehicle_capture_time_ns"], "record.frame.vehicle_capture_time_ns"
    )
    infrastructure_time = _nonnegative_int64(
        row["infrastructure_capture_time_ns"],
        "record.frame.infrastructure_capture_time_ns",
    )
    decision_time = _nonnegative_int64(
        row["decision_time_ns"], "record.frame.decision_time_ns"
    )
    if decision_time < max(vehicle_time, infrastructure_time):
        raise FixedDetectionContractError(
            "record.frame.decision_time_ns violates the causal time rule"
        )
    return {
        "sequence_id": _identifier(row["sequence_id"], "record.frame.sequence_id"),
        "vehicle_frame_id": _identifier(
            row["vehicle_frame_id"], "record.frame.vehicle_frame_id"
        ),
        "infrastructure_frame_id": _identifier(
            row["infrastructure_frame_id"],
            "record.frame.infrastructure_frame_id",
        ),
        "vehicle_capture_time_ns": vehicle_time,
        "infrastructure_capture_time_ns": infrastructure_time,
        "decision_time_ns": decision_time,
    }


def _validate_coordinate_system(value: Any) -> dict[str, Any]:
    row = _exact_fields(
        value, frozenset(_COORDINATE_SYSTEM), "record.coordinate_system"
    )
    axes = _exact_fields(
        row["axes"],
        frozenset(_COORDINATE_SYSTEM["axes"]),
        "record.coordinate_system.axes",
    )
    normalized = {
        "frame": _string(row["frame"], "record.coordinate_system.frame"),
        "axes": {
            axis: _string(axes[axis], f"record.coordinate_system.axes.{axis}")
            for axis in ("x", "y", "z")
        },
        "length_unit": _string(
            row["length_unit"], "record.coordinate_system.length_unit"
        ),
        "angle_unit": _string(row["angle_unit"], "record.coordinate_system.angle_unit"),
        "box_3d_order": _string(
            row["box_3d_order"], "record.coordinate_system.box_3d_order"
        ),
        "z_reference": _string(
            row["z_reference"], "record.coordinate_system.z_reference"
        ),
        "yaw_convention": _string(
            row["yaw_convention"], "record.coordinate_system.yaw_convention"
        ),
    }
    if normalized != _COORDINATE_SYSTEM:
        raise FixedDetectionContractError(
            "record.coordinate_system differs from the frozen coordinate contract"
        )
    return expected_coordinate_system()


def _validate_source(value: Any) -> dict[str, Any]:
    row = _exact_fields(value, SOURCE_FIELDS, "record.source")
    model_id = _string(row["model_id"], "record.source.model_id", max_length=64)
    if model_id not in MODEL_IDS:
        raise FixedDetectionContractError(
            "record.source.model_id is outside the frozen model set"
        )
    checkpoint_role = _string(
        row["checkpoint_role"], "record.source.checkpoint_role", max_length=32
    )
    if checkpoint_role != "canonical_final":
        raise FixedDetectionContractError(
            "record.source.checkpoint_role must be 'canonical_final'"
        )
    teacher = row["teacher_checkpoint_sha256"]
    if model_id == "coformernet_controlled_adaptation":
        if teacher is not None:
            raise FixedDetectionContractError(
                "CoFormerNet controlled adaptation must use a null teacher checkpoint"
            )
        normalized_teacher = None
    else:
        normalized_teacher = _sha256(teacher, "record.source.teacher_checkpoint_sha256")
    config_path = _relative_python_path(row["config_path"], "record.source.config_path")
    if config_path != CONFIG_PATH_BY_MODEL[model_id]:
        raise FixedDetectionContractError(
            "record.source.config_path does not match record.source.model_id"
        )
    return {
        "repository_url": _exact_string(
            _repository_url(row["repository_url"], "record.source.repository_url"),
            REPOSITORY_URL,
            "record.source.repository_url",
        ),
        "implementation_revision": _revision(
            row["implementation_revision"],
            "record.source.implementation_revision",
        ),
        "implementation_tree_sha256": _sha256(
            row["implementation_tree_sha256"],
            "record.source.implementation_tree_sha256",
        ),
        "model_id": model_id,
        "config_path": config_path,
        "config_sha256": _sha256(row["config_sha256"], "record.source.config_sha256"),
        "checkpoint_role": checkpoint_role,
        "checkpoint_sha256": _sha256(
            row["checkpoint_sha256"], "record.source.checkpoint_sha256"
        ),
        "teacher_checkpoint_sha256": normalized_teacher,
        "inference_adapter_sha256": _sha256(
            row["inference_adapter_sha256"],
            "record.source.inference_adapter_sha256",
        ),
        "environment_manifest_sha256": _sha256(
            row["environment_manifest_sha256"],
            "record.source.environment_manifest_sha256",
        ),
    }


def _exact_string(value: Any, expected: str, label: str) -> str:
    result = _string(value, label, max_length=max(64, len(expected)))
    if result != expected:
        raise FixedDetectionContractError(f"{label} must be {expected!r}")
    return result


def _exact_integer(value: Any, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise FixedDetectionContractError(f"{label} must be integer {expected}")
    return value


def _exact_number(value: Any, expected: float, label: str) -> float:
    result = _finite_number(value, label)
    if result != expected:
        raise FixedDetectionContractError(f"{label} must be {expected}")
    return result


def _exact_interval(value: Any, expected: list[float], label: str) -> list[float]:
    result = _finite_vector(value, 2, label)
    if result != expected:
        raise FixedDetectionContractError(f"{label} differs from the frozen interval")
    return result


def _validate_policy(value: Any) -> dict[str, Any]:
    row = _exact_fields(value, frozenset(_FIXED_POLICY), "record.policy")
    class_mapping = _exact_fields(
        row["class_mapping"],
        frozenset(_FIXED_POLICY["class_mapping"]),
        "record.policy.class_mapping",
    )
    roi = _exact_fields(
        row["roi"], frozenset(_FIXED_POLICY["roi"]), "record.policy.roi"
    )
    score = _exact_fields(
        row["score"], frozenset(_FIXED_POLICY["score"]), "record.policy.score"
    )
    nms = _exact_fields(
        row["nms"], frozenset(_FIXED_POLICY["nms"]), "record.policy.nms"
    )
    normalized = {
        "class_mapping": {
            "source_label_id": _exact_integer(
                class_mapping["source_label_id"],
                0,
                "record.policy.class_mapping.source_label_id",
            ),
            "source_class_name": _exact_string(
                class_mapping["source_class_name"],
                "Car",
                "record.policy.class_mapping.source_class_name",
            ),
            "output_class_name": _exact_string(
                class_mapping["output_class_name"],
                "vehicle",
                "record.policy.class_mapping.output_class_name",
            ),
        },
        "roi": {
            "rule": _exact_string(
                roi["rule"],
                "box_bottom_center_inclusive",
                "record.policy.roi.rule",
            ),
            "x_m": _exact_interval(roi["x_m"], [0.0, 80.0], "record.policy.roi.x_m"),
            "y_m": _exact_interval(roi["y_m"], [-40.0, 40.0], "record.policy.roi.y_m"),
            "z_m": _exact_interval(roi["z_m"], [-3.0, 1.0], "record.policy.roi.z_m"),
        },
        "score": {
            "comparison": _exact_string(
                score["comparison"],
                "greater_than_or_equal",
                "record.policy.score.comparison",
            ),
            "threshold": _exact_number(
                score["threshold"], 0.05, "record.policy.score.threshold"
            ),
            "maximum": _exact_number(
                score["maximum"], 1.0, "record.policy.score.maximum"
            ),
        },
        "nms": {
            "type": _exact_string(nms["type"], "rotate_nms", "record.policy.nms.type"),
            "iou_threshold": _exact_number(
                nms["iou_threshold"],
                0.01,
                "record.policy.nms.iou_threshold",
            ),
            "nms_across_levels": nms["nms_across_levels"],
            "pre_max_detections": _exact_integer(
                nms["pre_max_detections"],
                2000,
                "record.policy.nms.pre_max_detections",
            ),
            "max_detections_per_frame": _exact_integer(
                nms["max_detections_per_frame"],
                100,
                "record.policy.nms.max_detections_per_frame",
            ),
        },
    }
    if nms["nms_across_levels"] is not False:
        raise FixedDetectionContractError(
            "record.policy.nms.nms_across_levels must be false"
        )
    if normalized != _FIXED_POLICY:
        raise FixedDetectionContractError(
            "record.policy differs from the frozen post-processing contract"
        )
    return expected_policy()


def _validate_detection(value: Any, index: int) -> dict[str, Any]:
    label = f"record.detections[{index}]"
    row = _exact_fields(value, DETECTION_FIELDS, label)
    detection_id = _identifier(row["detection_id"], f"{label}.detection_id")
    expected_detection_id = f"det-{index:06d}"
    if detection_id != expected_detection_id:
        raise FixedDetectionContractError(
            f"{label}.detection_id must be frame-local ordinal "
            f"{expected_detection_id!r}"
        )
    source_label_id = _exact_integer(
        row["source_label_id"], 0, f"{label}.source_label_id"
    )
    source_class_name = _exact_string(
        row["source_class_name"], "Car", f"{label}.source_class_name"
    )
    class_name = _exact_string(row["class_name"], "vehicle", f"{label}.class_name")
    box = _finite_vector(row["box_3d"], 7, f"{label}.box_3d")
    x, y, z, length, width, height, yaw = box
    for axis, coordinate, bounds in (
        ("x", x, _FIXED_POLICY["roi"]["x_m"]),
        ("y", y, _FIXED_POLICY["roi"]["y_m"]),
        ("z", z, _FIXED_POLICY["roi"]["z_m"]),
    ):
        if not bounds[0] <= coordinate <= bounds[1]:
            raise FixedDetectionContractError(
                f"{label}.box_3d {axis} bottom-center is outside the frozen ROI"
            )
    if min(length, width, height) < MIN_BOX_DIMENSION_M:
        raise FixedDetectionContractError(
            f"{label}.box_3d dimensions must be strictly positive and at least "
            f"{MIN_BOX_DIMENSION_M} metres"
        )
    if max(length, width, height) > MAX_BOX_DIMENSION_M:
        raise FixedDetectionContractError(
            f"{label}.box_3d dimensions exceed the frozen numeric bound"
        )
    if not -math.pi <= yaw < math.pi:
        raise FixedDetectionContractError(f"{label}.box_3d yaw must be in [-pi, pi)")
    score = _finite_number(row["score"], f"{label}.score")
    threshold = _FIXED_POLICY["score"]["threshold"]
    maximum = _FIXED_POLICY["score"]["maximum"]
    if not threshold <= score <= maximum:
        raise FixedDetectionContractError(
            f"{label}.score must be within the frozen score interval"
        )
    return {
        "detection_id": detection_id,
        "source_label_id": source_label_id,
        "source_class_name": source_class_name,
        "class_name": class_name,
        "box_3d": box,
        "score": score,
    }


def _validate_detections(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise FixedDetectionContractError("record.detections must be a JSON array")
    maximum = _FIXED_POLICY["nms"]["max_detections_per_frame"]
    if len(value) > maximum:
        raise FixedDetectionContractError(
            "record.detections exceeds the frozen per-frame maximum"
        )
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, item in enumerate(value):
        detection = _validate_detection(item, index)
        detection_id = detection["detection_id"]
        if detection_id in identifiers:
            raise FixedDetectionContractError(
                f"duplicate frame-local detection_id: {detection_id}"
            )
        identifiers.add(detection_id)
        normalized.append(detection)
    expected_order = sorted(
        normalized,
        key=lambda item: (
            -item["score"],
            *item["box_3d"],
            item["detection_id"],
        ),
    )
    if normalized != expected_order:
        raise FixedDetectionContractError(
            "record.detections must use descending score then lexicographic box order"
        )
    return normalized


def validate_record(value: Any) -> dict[str, Any]:
    """Validate and normalize one fixed-detection frame record.

    The returned object is detached from the input.  Object fields are strict,
    numeric values are finite, and integer-valued booleans are rejected.
    """

    _reject_forbidden_fields(value)
    row = _exact_fields(value, TOP_LEVEL_FIELDS, "record")
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != SCHEMA_VERSION
    ):
        raise FixedDetectionContractError("record.schema_version must be integer 1")
    if row["contract_id"] != CONTRACT_ID:
        raise FixedDetectionContractError(f"record.contract_id must be {CONTRACT_ID!r}")
    if row["record_kind"] != RECORD_KIND:
        raise FixedDetectionContractError(f"record.record_kind must be {RECORD_KIND!r}")
    if row["scientific_claim_allowed"] is not False:
        raise FixedDetectionContractError(
            "record.scientific_claim_allowed must be false"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "record_kind": RECORD_KIND,
        "scientific_claim_allowed": False,
        "dataset": _validate_dataset(row["dataset"]),
        "frame": _validate_frame(row["frame"]),
        "coordinate_system": _validate_coordinate_system(row["coordinate_system"]),
        "source": _validate_source(row["source"]),
        "policy": _validate_policy(row["policy"]),
        "detections": _validate_detections(row["detections"]),
    }


def _document_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    dataset = record["dataset"]
    frame = record["frame"]
    return (
        dataset["split"],
        frame["sequence_id"],
        frame["decision_time_ns"],
        frame["vehicle_frame_id"],
        frame["infrastructure_frame_id"],
    )


def _document_length(records: Sequence[Mapping[str, Any]]) -> int:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise FixedDetectionContractError(
            "fixed-detection document must be a sequence of frame records"
        )
    try:
        count = len(records)
    except (OverflowError, TypeError, ValueError) as exc:
        raise FixedDetectionContractError(
            "fixed-detection document length is not available"
        ) from exc
    if count == 0:
        raise FixedDetectionContractError(
            "fixed-detection document must contain at least one frame record"
        )
    if count > MAX_DOCUMENT_RECORDS:
        raise FixedDetectionContractError(
            "fixed-detection document exceeds the record-count limit"
        )
    # Every JSONL record consumes at least its final LF byte. This cheap lower
    # bound rejects impossible documents before invoking any record validator.
    if count > MAX_JSONL_BYTES:
        raise FixedDetectionContractError(
            "fixed-detection JSONL document exceeds the byte limit"
        )
    return count


def _validated_record_stream(
    records: Sequence[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield normalized records while enforcing cross-record invariants."""

    expected_count = _document_length(records)
    first: dict[str, Any] | None = None
    frame_identities: set[tuple[str, str, str]] = set()
    last_decision_time: dict[tuple[str, str], int] = {}
    previous_sort_key: tuple[Any, ...] | None = None
    iterator = iter(records)
    for index in range(expected_count):
        try:
            value = next(iterator)
        except StopIteration as exc:
            raise FixedDetectionContractError(
                "fixed-detection document length changed during validation"
            ) from exc
        record = validate_record(value)
        if first is None:
            first = record
        if record["dataset"] != first["dataset"]:
            raise FixedDetectionContractError(
                f"record {index} mixes dataset release or split identities"
            )
        if record["source"] != first["source"]:
            raise FixedDetectionContractError(
                f"record {index} mixes model or source identities"
            )
        if record["coordinate_system"] != first["coordinate_system"]:
            raise FixedDetectionContractError(
                f"record {index} mixes coordinate-system identities"
            )
        if record["policy"] != first["policy"]:
            raise FixedDetectionContractError(
                f"record {index} mixes post-processing policies"
            )
        split = record["dataset"]["split"]
        frame = record["frame"]
        identity = (split, frame["sequence_id"], frame["vehicle_frame_id"])
        if identity in frame_identities:
            raise FixedDetectionContractError(
                f"duplicate frame identity at record {index}: {identity!r}"
            )
        frame_identities.add(identity)
        sequence = (split, frame["sequence_id"])
        decision_time = frame["decision_time_ns"]
        previous = last_decision_time.get(sequence)
        if previous is not None and decision_time <= previous:
            raise FixedDetectionContractError(
                f"decision_time_ns is not strictly increasing at record {index}"
            )
        last_decision_time[sequence] = decision_time
        sort_key = _document_sort_key(record)
        if previous_sort_key is not None and sort_key < previous_sort_key:
            raise FixedDetectionContractError(
                "fixed-detection records are not in canonical document order"
            )
        previous_sort_key = sort_key
        yield record
    try:
        next(iterator)
    except StopIteration:
        return
    raise FixedDetectionContractError(
        "fixed-detection document length changed during validation"
    )


def validate_document(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate one bounded, ordered document and return detached records."""

    return tuple(_validated_record_stream(records))


def _decode_text(data: bytes | str, *, label: str, maximum: int) -> str:
    if isinstance(data, bytes):
        if len(data) > maximum:
            raise FixedDetectionContractError(f"{label} exceeds the byte limit")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FixedDetectionContractError(f"{label} is not UTF-8") from exc
    elif isinstance(data, str):
        try:
            encoded = data.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise FixedDetectionContractError(f"{label} is not UTF-8") from exc
        if len(encoded) > maximum:
            raise FixedDetectionContractError(f"{label} exceeds the byte limit")
        text = data
    else:
        raise FixedDetectionContractError(f"{label} must be bytes or text")
    if text.startswith("\ufeff"):
        raise FixedDetectionContractError(f"{label} must not contain a UTF-8 BOM")
    return text


def _loads_json_object(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except FixedDetectionContractError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise FixedDetectionContractError(f"{label} is not strict JSON") from exc


def loads_jsonl(data: bytes | str) -> tuple[dict[str, Any], ...]:
    """Parse and validate a canonical fixed-detection JSONL document."""

    text = _decode_text(
        data, label="fixed-detection JSONL document", maximum=MAX_JSONL_BYTES
    )
    if "\r" in text:
        raise FixedDetectionContractError(
            "fixed-detection JSONL document must use LF line endings"
        )
    if not text.endswith("\n"):
        raise FixedDetectionContractError(
            "fixed-detection JSONL document must end with LF"
        )
    record_count = text.count("\n")
    if record_count > MAX_DOCUMENT_RECORDS:
        raise FixedDetectionContractError(
            "fixed-detection document exceeds the record-count limit"
        )
    if record_count == 0 or text == "\n":
        raise FixedDetectionContractError(
            "fixed-detection JSONL document must contain at least one record"
        )
    records: list[Mapping[str, Any]] = []
    start = 0
    for index in range(record_count):
        end = text.find("\n", start)
        if end < 0:
            raise FixedDetectionContractError(
                "fixed-detection JSONL document has inconsistent line framing"
            )
        line = text[start:end]
        start = end + 1
        if not line:
            raise FixedDetectionContractError(
                "fixed-detection JSONL document must not contain blank lines"
            )
        if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
            raise FixedDetectionContractError(
                f"fixed-detection JSONL record {index} exceeds the byte limit"
            )
        value = _loads_json_object(line, f"fixed-detection JSONL record {index}")
        if not isinstance(value, Mapping):
            raise FixedDetectionContractError(
                f"fixed-detection JSONL record {index} must be a JSON object"
            )
        records.append(value)
    if start != len(text):
        raise FixedDetectionContractError(
            "fixed-detection JSONL document has inconsistent line framing"
        )
    normalized: list[dict[str, Any]] = []
    canonical = bytearray()
    for record, line in _validated_canonical_lines(records):
        normalized.append(record)
        canonical.extend(line)
    if bytes(canonical) != text.encode("utf-8"):
        raise FixedDetectionContractError(
            "fixed-detection JSONL document is not canonically serialized"
        )
    return tuple(normalized)


def _canonical_float_text(value: float) -> str:
    if not math.isfinite(value):
        raise FixedDetectionContractError(
            "record contains a non-finite canonical float"
        )
    normalized = 0.0 if value == 0.0 else value
    return format(normalized, ".16e")


def _canonical_json_text(value: Any) -> str:
    """Serialize normalized JSON with a fully frozen numeric spelling."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _canonical_float_text(value)
    if isinstance(value, str):
        try:
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise FixedDetectionContractError(
                "record contains a string that cannot be serialized"
            ) from exc
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json_text(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise FixedDetectionContractError(
                "record contains a non-string canonical object key"
            )
        return (
            "{"
            + ",".join(
                _canonical_json_text(key) + ":" + _canonical_json_text(value[key])
                for key in sorted(value)
            )
            + "}"
        )
    raise FixedDetectionContractError(
        f"record contains unsupported canonical type {type(value).__name__}"
    )


def _canonical_line(normalized_record: Mapping[str, Any]) -> bytes:
    text = _canonical_json_text(normalized_record)
    payload = (text + "\n").encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        raise FixedDetectionContractError(
            "fixed-detection JSONL record exceeds the byte limit"
        )
    return payload


def _validated_canonical_lines(
    records: Sequence[Mapping[str, Any]],
) -> Iterator[tuple[dict[str, Any], bytes]]:
    total_bytes = 0
    for record in _validated_record_stream(records):
        line = _canonical_line(record)
        total_bytes += len(line)
        if total_bytes > MAX_JSONL_BYTES:
            raise FixedDetectionContractError(
                "fixed-detection JSONL document exceeds the byte limit"
            )
        yield record, line


def canonical_record_bytes(record: Mapping[str, Any]) -> bytes:
    """Return one validated, canonical JSONL record including its final LF."""

    return _canonical_line(validate_record(record))


def canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Return a validated canonical JSONL document without reordering records."""

    payload = bytearray()
    for _, line in _validated_canonical_lines(records):
        payload.extend(line)
    return bytes(payload)


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""

    if not isinstance(data, bytes):
        raise FixedDetectionContractError("sha256_bytes input must be bytes")
    return hashlib.sha256(data).hexdigest()


def record_sha256(record: Mapping[str, Any]) -> str:
    """Digest one record after validation and canonical serialization."""

    return sha256_bytes(canonical_record_bytes(record))


def document_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Stream a validated document into SHA-256 without materializing JSONL."""

    digest = hashlib.sha256()
    for _, line in _validated_canonical_lines(records):
        digest.update(line)
    return digest.hexdigest()


__all__ = [
    "CONTRACT_ID",
    "CONFIG_PATH_BY_MODEL",
    "DATASET_NAME",
    "FORBIDDEN_FIELD_NAMES",
    "FixedDetectionContractError",
    "MODEL_IDS",
    "RECORD_KIND",
    "REPOSITORY_URL",
    "SCHEMA_VERSION",
    "SPLITS",
    "canonical_jsonl_bytes",
    "canonical_record_bytes",
    "document_sha256",
    "expected_coordinate_system",
    "expected_policy",
    "loads_jsonl",
    "record_sha256",
    "sha256_bytes",
    "validate_document",
    "validate_record",
]
