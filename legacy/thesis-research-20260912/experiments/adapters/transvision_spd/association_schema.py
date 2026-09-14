"""Strict diagnostic association-observation sidecar for V2X-Seq SPD.

The sidecar is deliberately incapable of closing a scientific hypothesis.  It
binds one association-input record to every record in a canonical fixed-
detection document and validates only already-materialized JSON-like values.
It does not read V2X-Seq, load a model, access ClearML, evaluate predictions,
or infer missing observations or reliability values.

Version 1 is diagnostic-only.  Its spatial reliability value is an
uncalibrated ranking signal obtained as the box mean of a hash-bound
reliability map.  Detection score is copied only to make the fixed-detection
join auditable; it is never accepted as reliability.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.adapters.transvision_spd import schema as fixed_schema


SCHEMA_VERSION = 1
CONTRACT_ID = "RTPV2X-ASSOCIATION-OBSERVATIONS-v1"
RECORD_KIND = "diagnostic_association_observation_frame"
DATASET_NAME = fixed_schema.DATASET_NAME
SPLITS = fixed_schema.SPLITS

MAX_JSONL_BYTES = 512 * 1024 * 1024
MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_DOCUMENT_RECORDS = fixed_schema.MAX_DOCUMENT_RECORDS
MAX_NESTING_DEPTH = 32
MAX_AGENT_COUNT = 32
MAX_CANDIDATES_PER_FRAME = 100
MAX_SOURCE_OBSERVATIONS_PER_CANDIDATE = MAX_AGENT_COUNT
MAX_SIGNED_INT64 = 2**63 - 1
MAX_IDENTIFIER_LENGTH = 128

FIXED_DETECTION_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "clearml"
    / "protocols"
    / "fixed-detections-v1.json"
)

BOX_MEAN_AGGREGATION = (
    "arithmetic_mean_of_map_cells_with_centres_inside_or_on_oriented_bev_box"
)
RELIABILITY_CALIBRATION = "uncalibrated"
RELIABILITY_VALUE_ROLE = "ranking_signal_only_not_probability_or_confidence"
DETECTION_SCORE_ROLE = "passthrough_only_never_spatial_reliability"

_SEMANTICS: dict[str, str] = {
    "spatial_reliability_aggregation": BOX_MEAN_AGGREGATION,
    "spatial_reliability_calibration": RELIABILITY_CALIBRATION,
    "spatial_reliability_value_role": RELIABILITY_VALUE_ROLE,
    "detection_score_role": DETECTION_SCORE_ROLE,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "record_kind",
        "scientific_claim_allowed",
        "runs_h1",
        "production_input_allowed",
        "fixed_detection_binding",
        "dataset",
        "frame",
        "source",
        "agent_registry",
        "reliability_map_binding",
        "semantics",
        "candidates",
    }
)
FIXED_BINDING_FIELDS = frozenset(
    {
        "fixed_detection_contract_id",
        "fixed_detection_contract_sha256",
        "fixed_detection_document_sha256",
        "fixed_detection_record_sha256",
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
        "fixed_detection_source_sha256",
        "observation_generator_sha256",
    }
)
AGENT_FIELDS = frozenset({"agent_id", "agent_role", "agent_manifest_sha256"})
RELIABILITY_MAP_BINDING_FIELDS = frozenset(
    {
        "reliability_map_contract_id",
        "reliability_map_contract_sha256",
        "reliability_map_document_sha256",
    }
)
SEMANTICS_FIELDS = frozenset(_SEMANTICS)
CANDIDATE_FIELDS = frozenset(
    {"detection_id", "detection_score_passthrough", "source_observations"}
)
OBSERVATION_FIELDS = frozenset(
    {
        "agent_id",
        "event_time_ns",
        "complete_arrival_time_ns",
        "box_3d",
        "source_observation_record_sha256",
        "reliability_map_record_sha256",
        "spatial_reliability",
    }
)

FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "annotation",
        "annotations",
        "confidence",
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
        "probability",
        "target",
        "targets",
        "track",
        "track_id",
        "tracking_id",
        "tracks",
        "truth",
    }
)


class AssociationObservationContractError(ValueError):
    """Raised when an association-observation artifact is unsafe or ambiguous."""


def expected_semantics() -> dict[str, str]:
    """Return a detached copy of the diagnostic-only value semantics."""

    return dict(_SEMANTICS)


def _reject_constant(value: str) -> None:
    raise AssociationObservationContractError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssociationObservationContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_forbidden_fields(
    value: Any, *, label: str = "record", depth: int = 0
) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise AssociationObservationContractError(
            f"{label} exceeds the maximum nesting depth"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AssociationObservationContractError(
                    f"{label} contains a non-string object key"
                )
            normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in FORBIDDEN_FIELD_NAMES:
                raise AssociationObservationContractError(
                    f"{label} contains forbidden field {key!r}"
                )
            _reject_forbidden_fields(item, label=f"{label}.{key}", depth=depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, label=f"{label}[{index}]", depth=depth + 1)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssociationObservationContractError(f"{label} must be a JSON object")
    return value


def _exact_fields(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    row = _mapping(value, label)
    observed = set(row)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise AssociationObservationContractError(
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
        raise AssociationObservationContractError(
            f"{label} must be a trimmed non-empty string without control characters"
        )
    return value


def _identifier(value: Any, label: str) -> str:
    result = _string(value, label, max_length=MAX_IDENTIFIER_LENGTH)
    if not _IDENTIFIER_RE.fullmatch(result):
        raise AssociationObservationContractError(
            f"{label} contains characters outside the identifier contract"
        )
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise AssociationObservationContractError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _nonnegative_int64(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_SIGNED_INT64
    ):
        raise AssociationObservationContractError(
            f"{label} must be a non-negative signed 64-bit integer"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssociationObservationContractError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise AssociationObservationContractError(
            f"{label} exceeds the numeric bound"
        ) from exc
    if not math.isfinite(result):
        raise AssociationObservationContractError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise AssociationObservationContractError(
            f"{label} must be a JSON array with shape [{length}]"
        )
    return [
        _finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]


def _validate_dataset(value: Any) -> dict[str, Any]:
    row = _exact_fields(value, DATASET_FIELDS, "record.dataset")
    if row["name"] != DATASET_NAME:
        raise AssociationObservationContractError(
            f"record.dataset.name must be {DATASET_NAME!r}"
        )
    split = _string(row["split"], "record.dataset.split", max_length=10)
    if split not in SPLITS:
        raise AssociationObservationContractError(
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
        raise AssociationObservationContractError(
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


def _validate_fixed_binding(value: Any) -> dict[str, str]:
    row = _exact_fields(
        value, FIXED_BINDING_FIELDS, "record.fixed_detection_binding"
    )
    contract_id = _string(
        row["fixed_detection_contract_id"],
        "record.fixed_detection_binding.fixed_detection_contract_id",
    )
    if contract_id != fixed_schema.CONTRACT_ID:
        raise AssociationObservationContractError(
            "record.fixed_detection_binding.fixed_detection_contract_id differs "
            "from the fixed-detection v1 contract"
        )
    return {
        "fixed_detection_contract_id": contract_id,
        "fixed_detection_contract_sha256": _sha256(
            row["fixed_detection_contract_sha256"],
            "record.fixed_detection_binding.fixed_detection_contract_sha256",
        ),
        "fixed_detection_document_sha256": _sha256(
            row["fixed_detection_document_sha256"],
            "record.fixed_detection_binding.fixed_detection_document_sha256",
        ),
        "fixed_detection_record_sha256": _sha256(
            row["fixed_detection_record_sha256"],
            "record.fixed_detection_binding.fixed_detection_record_sha256",
        ),
    }


def _validate_source(value: Any) -> dict[str, str]:
    row = _exact_fields(value, SOURCE_FIELDS, "record.source")
    return {
        "fixed_detection_source_sha256": _sha256(
            row["fixed_detection_source_sha256"],
            "record.source.fixed_detection_source_sha256",
        ),
        "observation_generator_sha256": _sha256(
            row["observation_generator_sha256"],
            "record.source.observation_generator_sha256",
        ),
    }


def _validate_agent_registry(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise AssociationObservationContractError(
            "record.agent_registry must be a JSON array"
        )
    if not value:
        raise AssociationObservationContractError(
            "record.agent_registry must contain at least one stable agent"
        )
    if len(value) > MAX_AGENT_COUNT:
        raise AssociationObservationContractError(
            "record.agent_registry exceeds the agent-count limit"
        )
    result: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for index, item in enumerate(value):
        label = f"record.agent_registry[{index}]"
        row = _exact_fields(item, AGENT_FIELDS, label)
        agent_id = _identifier(row["agent_id"], f"{label}.agent_id")
        if agent_id in identifiers:
            raise AssociationObservationContractError(
                f"duplicate stable agent_id: {agent_id}"
            )
        identifiers.add(agent_id)
        result.append(
            {
                "agent_id": agent_id,
                "agent_role": _identifier(row["agent_role"], f"{label}.agent_role"),
                "agent_manifest_sha256": _sha256(
                    row["agent_manifest_sha256"],
                    f"{label}.agent_manifest_sha256",
                ),
            }
        )
    if result != sorted(result, key=lambda item: item["agent_id"]):
        raise AssociationObservationContractError(
            "record.agent_registry must be ordered by agent_id"
        )
    return result


def _validate_reliability_map_binding(value: Any) -> dict[str, str]:
    row = _exact_fields(
        value, RELIABILITY_MAP_BINDING_FIELDS, "record.reliability_map_binding"
    )
    return {
        "reliability_map_contract_id": _identifier(
            row["reliability_map_contract_id"],
            "record.reliability_map_binding.reliability_map_contract_id",
        ),
        "reliability_map_contract_sha256": _sha256(
            row["reliability_map_contract_sha256"],
            "record.reliability_map_binding.reliability_map_contract_sha256",
        ),
        "reliability_map_document_sha256": _sha256(
            row["reliability_map_document_sha256"],
            "record.reliability_map_binding.reliability_map_document_sha256",
        ),
    }


def _validate_semantics(value: Any) -> dict[str, str]:
    row = _exact_fields(value, SEMANTICS_FIELDS, "record.semantics")
    normalized = {
        key: _string(row[key], f"record.semantics.{key}") for key in _SEMANTICS
    }
    if normalized != _SEMANTICS:
        raise AssociationObservationContractError(
            "record.semantics differs from the diagnostic v1 semantics"
        )
    return expected_semantics()


def _validate_box(value: Any, label: str) -> list[float]:
    box = _finite_vector(value, 7, label)
    x, y, z, length, width, height, yaw = box
    policy = fixed_schema.expected_policy()
    for axis, coordinate, bounds in (
        ("x", x, policy["roi"]["x_m"]),
        ("y", y, policy["roi"]["y_m"]),
        ("z", z, policy["roi"]["z_m"]),
    ):
        if not bounds[0] <= coordinate <= bounds[1]:
            raise AssociationObservationContractError(
                f"{label} {axis} bottom-center is outside the fixed-detection ROI"
            )
    if min(length, width, height) < fixed_schema.MIN_BOX_DIMENSION_M:
        raise AssociationObservationContractError(
            f"{label} dimensions are below the fixed-detection numeric bound"
        )
    if max(length, width, height) > fixed_schema.MAX_BOX_DIMENSION_M:
        raise AssociationObservationContractError(
            f"{label} dimensions exceed the fixed-detection numeric bound"
        )
    if not -math.pi <= yaw < math.pi:
        raise AssociationObservationContractError(f"{label} yaw must be in [-pi, pi)")
    return box


def _validate_observation(
    value: Any,
    *,
    candidate_index: int,
    observation_index: int,
    agent_ids: frozenset[str],
    decision_time_ns: int,
) -> dict[str, Any]:
    label = (
        f"record.candidates[{candidate_index}]."
        f"source_observations[{observation_index}]"
    )
    row = _exact_fields(value, OBSERVATION_FIELDS, label)
    agent_id = _identifier(row["agent_id"], f"{label}.agent_id")
    if agent_id not in agent_ids:
        raise AssociationObservationContractError(
            f"{label}.agent_id is absent from the stable agent registry"
        )
    event_time = _nonnegative_int64(row["event_time_ns"], f"{label}.event_time_ns")
    arrival_time = _nonnegative_int64(
        row["complete_arrival_time_ns"], f"{label}.complete_arrival_time_ns"
    )
    if not event_time <= arrival_time <= decision_time_ns:
        raise AssociationObservationContractError(
            f"{label} violates event_time_ns <= complete_arrival_time_ns "
            "<= decision_time_ns"
        )
    reliability = _finite_number(
        row["spatial_reliability"], f"{label}.spatial_reliability"
    )
    if not 0.0 <= reliability <= 1.0:
        raise AssociationObservationContractError(
            f"{label}.spatial_reliability must be in [0, 1]"
        )
    return {
        "agent_id": agent_id,
        "event_time_ns": event_time,
        "complete_arrival_time_ns": arrival_time,
        "box_3d": _validate_box(row["box_3d"], f"{label}.box_3d"),
        "source_observation_record_sha256": _sha256(
            row["source_observation_record_sha256"],
            f"{label}.source_observation_record_sha256",
        ),
        "reliability_map_record_sha256": _sha256(
            row["reliability_map_record_sha256"],
            f"{label}.reliability_map_record_sha256",
        ),
        "spatial_reliability": reliability,
    }


def _validate_candidates(
    value: Any,
    *,
    agent_ids: frozenset[str],
    decision_time_ns: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise AssociationObservationContractError(
            "record.candidates must be a JSON array"
        )
    if len(value) > MAX_CANDIDATES_PER_FRAME:
        raise AssociationObservationContractError(
            "record.candidates exceeds the per-frame candidate limit"
        )
    result: list[dict[str, Any]] = []
    detection_ids: set[str] = set()
    for candidate_index, item in enumerate(value):
        label = f"record.candidates[{candidate_index}]"
        row = _exact_fields(item, CANDIDATE_FIELDS, label)
        detection_id = _identifier(row["detection_id"], f"{label}.detection_id")
        expected_id = f"det-{candidate_index:06d}"
        if detection_id != expected_id:
            raise AssociationObservationContractError(
                f"{label}.detection_id must be canonical ordinal {expected_id!r}"
            )
        if detection_id in detection_ids:
            raise AssociationObservationContractError(
                f"duplicate fixed-detection reference: {detection_id}"
            )
        detection_ids.add(detection_id)
        score = _finite_number(
            row["detection_score_passthrough"],
            f"{label}.detection_score_passthrough",
        )
        if not 0.0 <= score <= 1.0:
            raise AssociationObservationContractError(
                f"{label}.detection_score_passthrough must be in [0, 1]"
            )
        observations_value = row["source_observations"]
        if not isinstance(observations_value, list) or not observations_value:
            raise AssociationObservationContractError(
                f"{label}.source_observations must be a non-empty JSON array"
            )
        if len(observations_value) > min(
            len(agent_ids), MAX_SOURCE_OBSERVATIONS_PER_CANDIDATE
        ):
            raise AssociationObservationContractError(
                f"{label}.source_observations exceeds the stable agent registry"
            )
        observations: list[dict[str, Any]] = []
        observed_agents: set[str] = set()
        for observation_index, observation_value in enumerate(observations_value):
            observation = _validate_observation(
                observation_value,
                candidate_index=candidate_index,
                observation_index=observation_index,
                agent_ids=agent_ids,
                decision_time_ns=decision_time_ns,
            )
            agent_id = observation["agent_id"]
            if agent_id in observed_agents:
                raise AssociationObservationContractError(
                    f"{label}.source_observations repeats agent_id {agent_id!r}"
                )
            observed_agents.add(agent_id)
            observations.append(observation)
        if observations != sorted(observations, key=lambda row: row["agent_id"]):
            raise AssociationObservationContractError(
                f"{label}.source_observations must be ordered by agent_id"
            )
        result.append(
            {
                "detection_id": detection_id,
                "detection_score_passthrough": score,
                "source_observations": observations,
            }
        )
    return result


def validate_record(value: Any) -> dict[str, Any]:
    """Validate one sidecar record structurally, without asserting its join.

    Call :func:`validate_document` with the complete fixed-detection document
    before using a record.  Record-only validation cannot prove any digest
    binding or fixed-detection coverage.
    """

    _reject_forbidden_fields(value)
    row = _exact_fields(value, TOP_LEVEL_FIELDS, "record")
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise AssociationObservationContractError(
            "record.schema_version must be integer 1"
        )
    if row["contract_id"] != CONTRACT_ID:
        raise AssociationObservationContractError(
            f"record.contract_id must be {CONTRACT_ID!r}"
        )
    if row["record_kind"] != RECORD_KIND:
        raise AssociationObservationContractError(
            f"record.record_kind must be {RECORD_KIND!r}"
        )
    for flag in (
        "scientific_claim_allowed",
        "runs_h1",
        "production_input_allowed",
    ):
        if row[flag] is not False:
            raise AssociationObservationContractError(f"record.{flag} must be false")
    frame = _validate_frame(row["frame"])
    agents = _validate_agent_registry(row["agent_registry"])
    agent_ids = frozenset(agent["agent_id"] for agent in agents)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "record_kind": RECORD_KIND,
        "scientific_claim_allowed": False,
        "runs_h1": False,
        "production_input_allowed": False,
        "fixed_detection_binding": _validate_fixed_binding(
            row["fixed_detection_binding"]
        ),
        "dataset": _validate_dataset(row["dataset"]),
        "frame": frame,
        "source": _validate_source(row["source"]),
        "agent_registry": agents,
        "reliability_map_binding": _validate_reliability_map_binding(
            row["reliability_map_binding"]
        ),
        "semantics": _validate_semantics(row["semantics"]),
        "candidates": _validate_candidates(
            row["candidates"],
            agent_ids=agent_ids,
            decision_time_ns=frame["decision_time_ns"],
        ),
    }


def _document_length(value: Any, *, label: str, maximum: int) -> int:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AssociationObservationContractError(f"{label} must be a sequence")
    try:
        count = len(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise AssociationObservationContractError(
            f"{label} length is not available"
        ) from exc
    if count == 0:
        raise AssociationObservationContractError(
            f"{label} must contain at least one frame record"
        )
    if count > maximum:
        raise AssociationObservationContractError(f"{label} exceeds the record limit")
    return count


def _fixed_protocol_sha256() -> str:
    try:
        payload = FIXED_DETECTION_PROTOCOL_PATH.read_bytes()
    except OSError as exc:
        raise AssociationObservationContractError(
            "cannot read the bound fixed-detection protocol"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _fixed_source_sha256(source: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_text(source).encode("utf-8")).hexdigest()


def _validated_record_stream(
    records: Sequence[Mapping[str, Any]],
    fixed_detection_records: Sequence[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    sidecar_count = _document_length(
        records,
        label="association-observation document",
        maximum=MAX_DOCUMENT_RECORDS,
    )
    fixed_count = _document_length(
        fixed_detection_records,
        label="fixed-detection document",
        maximum=fixed_schema.MAX_DOCUMENT_RECORDS,
    )
    if sidecar_count != fixed_count:
        raise AssociationObservationContractError(
            "association-observation document must contain exactly one record per "
            "fixed-detection frame"
        )
    try:
        fixed_records = fixed_schema.validate_document(fixed_detection_records)
        fixed_document_sha256 = fixed_schema.document_sha256(fixed_records)
    except fixed_schema.FixedDetectionContractError as exc:
        raise AssociationObservationContractError(
            "bound fixed-detection document is invalid"
        ) from exc
    fixed_contract_sha256 = _fixed_protocol_sha256()
    first: dict[str, Any] | None = None
    sidecar_iterator = iter(records)
    for index, fixed_record in enumerate(fixed_records):
        try:
            sidecar_value = next(sidecar_iterator)
        except StopIteration as exc:
            raise AssociationObservationContractError(
                "association-observation document length changed during validation"
            ) from exc
        record = validate_record(sidecar_value)
        if first is None:
            first = record
        else:
            if record["dataset"] != first["dataset"]:
                raise AssociationObservationContractError(
                    f"record {index} mixes dataset identities"
                )
            if record["source"] != first["source"]:
                raise AssociationObservationContractError(
                    f"record {index} mixes source identities"
                )
            if record["agent_registry"] != first["agent_registry"]:
                raise AssociationObservationContractError(
                    f"record {index} changes the stable agent registry"
                )
            if record["reliability_map_binding"] != first["reliability_map_binding"]:
                raise AssociationObservationContractError(
                    f"record {index} mixes reliability-map contracts or documents"
                )
            first_binding = first["fixed_detection_binding"]
            binding = record["fixed_detection_binding"]
            for field in (
                "fixed_detection_contract_id",
                "fixed_detection_contract_sha256",
                "fixed_detection_document_sha256",
            ):
                if binding[field] != first_binding[field]:
                    raise AssociationObservationContractError(
                        f"record {index} mixes fixed-detection contract or document"
                    )

        binding = record["fixed_detection_binding"]
        if binding["fixed_detection_contract_sha256"] != fixed_contract_sha256:
            raise AssociationObservationContractError(
                f"record {index} fixed-detection contract SHA-256 does not match"
            )
        if binding["fixed_detection_document_sha256"] != fixed_document_sha256:
            raise AssociationObservationContractError(
                f"record {index} fixed-detection document SHA-256 does not match"
            )
        if binding["fixed_detection_record_sha256"] != fixed_schema.record_sha256(
            fixed_record
        ):
            raise AssociationObservationContractError(
                f"record {index} fixed-detection record SHA-256 does not match"
            )
        if record["dataset"] != fixed_record["dataset"]:
            raise AssociationObservationContractError(
                f"record {index} dataset does not match its fixed-detection frame"
            )
        if record["frame"] != fixed_record["frame"]:
            raise AssociationObservationContractError(
                f"record {index} frame identity or time does not match"
            )
        if record["source"]["fixed_detection_source_sha256"] != _fixed_source_sha256(
            fixed_record["source"]
        ):
            raise AssociationObservationContractError(
                f"record {index} fixed-detection source SHA-256 does not match"
            )

        candidates = record["candidates"]
        detections = fixed_record["detections"]
        if len(candidates) != len(detections):
            raise AssociationObservationContractError(
                f"record {index} must reference every fixed detection exactly once"
            )
        for candidate_index, (candidate, detection) in enumerate(
            zip(candidates, detections, strict=True)
        ):
            if candidate["detection_id"] != detection["detection_id"]:
                raise AssociationObservationContractError(
                    f"record {index} candidate {candidate_index} references the "
                    "wrong fixed detection"
                )
            if candidate["detection_score_passthrough"] != detection["score"]:
                raise AssociationObservationContractError(
                    f"record {index} candidate {candidate_index} detection score "
                    "is not an exact passthrough"
                )
        yield record
    try:
        next(sidecar_iterator)
    except StopIteration:
        return
    raise AssociationObservationContractError(
        "association-observation document length changed during validation"
    )


def validate_document(
    records: Sequence[Mapping[str, Any]],
    fixed_detection_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate a complete sidecar against a complete fixed-detection document."""

    return tuple(
        record
        for record, _ in _validated_canonical_lines(
            records, fixed_detection_records
        )
    )


def _canonical_float_text(value: float) -> str:
    if not math.isfinite(value):
        raise AssociationObservationContractError(
            "record contains a non-finite canonical float"
        )
    normalized = 0.0 if value == 0.0 else value
    return format(normalized, ".16e")


def _canonical_json_text(value: Any) -> str:
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
            raise AssociationObservationContractError(
                "record contains a string that cannot be serialized"
            ) from exc
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json_text(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise AssociationObservationContractError(
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
    raise AssociationObservationContractError(
        f"record contains unsupported canonical type {type(value).__name__}"
    )


def _canonical_line(normalized_record: Mapping[str, Any]) -> bytes:
    payload = (_canonical_json_text(normalized_record) + "\n").encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        raise AssociationObservationContractError(
            "association-observation JSONL record exceeds the byte limit"
        )
    return payload


def _validated_canonical_lines(
    records: Sequence[Mapping[str, Any]],
    fixed_detection_records: Sequence[Mapping[str, Any]],
) -> Iterator[tuple[dict[str, Any], bytes]]:
    total_bytes = 0
    for record in _validated_record_stream(records, fixed_detection_records):
        line = _canonical_line(record)
        total_bytes += len(line)
        if total_bytes > MAX_JSONL_BYTES:
            raise AssociationObservationContractError(
                "association-observation JSONL document exceeds the byte limit"
            )
        yield record, line


def canonical_record_bytes(record: Mapping[str, Any]) -> bytes:
    """Serialize one structurally valid record; this does not prove its join."""

    return _canonical_line(validate_record(record))


def canonical_jsonl_bytes(
    records: Sequence[Mapping[str, Any]],
    fixed_detection_records: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize a validated, joined sidecar document without reordering it."""

    payload = bytearray()
    for _, line in _validated_canonical_lines(records, fixed_detection_records):
        payload.extend(line)
    return bytes(payload)


def _decode_text(data: bytes | str, *, label: str, maximum: int) -> str:
    if isinstance(data, bytes):
        if len(data) > maximum:
            raise AssociationObservationContractError(f"{label} exceeds the byte limit")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AssociationObservationContractError(f"{label} is not UTF-8") from exc
    elif isinstance(data, str):
        try:
            encoded = data.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise AssociationObservationContractError(f"{label} is not UTF-8") from exc
        if len(encoded) > maximum:
            raise AssociationObservationContractError(f"{label} exceeds the byte limit")
        text = data
    else:
        raise AssociationObservationContractError(f"{label} must be bytes or text")
    if text.startswith("\ufeff"):
        raise AssociationObservationContractError(
            f"{label} must not contain a UTF-8 BOM"
        )
    return text


def _loads_json_object(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except AssociationObservationContractError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise AssociationObservationContractError(f"{label} is not strict JSON") from exc


def loads_jsonl(
    data: bytes | str, *, fixed_detection_data: bytes | str
) -> tuple[dict[str, Any], ...]:
    """Parse canonical JSONL and verify every join against canonical detections."""

    text = _decode_text(
        data,
        label="association-observation JSONL document",
        maximum=MAX_JSONL_BYTES,
    )
    if "\r" in text:
        raise AssociationObservationContractError(
            "association-observation JSONL document must use LF line endings"
        )
    if not text.endswith("\n"):
        raise AssociationObservationContractError(
            "association-observation JSONL document must end with LF"
        )
    record_count = text.count("\n")
    if record_count == 0 or text == "\n":
        raise AssociationObservationContractError(
            "association-observation JSONL document must contain at least one record"
        )
    if record_count > MAX_DOCUMENT_RECORDS:
        raise AssociationObservationContractError(
            "association-observation document exceeds the record limit"
        )
    records: list[Mapping[str, Any]] = []
    start = 0
    for index in range(record_count):
        end = text.find("\n", start)
        if end < 0:
            raise AssociationObservationContractError(
                "association-observation JSONL has inconsistent line framing"
            )
        line = text[start:end]
        start = end + 1
        if not line:
            raise AssociationObservationContractError(
                "association-observation JSONL must not contain blank lines"
            )
        if len(line.encode("utf-8")) + 1 > MAX_RECORD_BYTES:
            raise AssociationObservationContractError(
                f"association-observation JSONL record {index} exceeds the byte limit"
            )
        value = _loads_json_object(line, f"association-observation record {index}")
        if not isinstance(value, Mapping):
            raise AssociationObservationContractError(
                f"association-observation record {index} must be a JSON object"
            )
        records.append(value)
    if start != len(text):
        raise AssociationObservationContractError(
            "association-observation JSONL has inconsistent line framing"
        )
    try:
        fixed_records = fixed_schema.loads_jsonl(fixed_detection_data)
    except fixed_schema.FixedDetectionContractError as exc:
        raise AssociationObservationContractError(
            "fixed-detection JSONL binding is not canonical and valid"
        ) from exc
    normalized: list[dict[str, Any]] = []
    canonical = bytearray()
    for record, line in _validated_canonical_lines(records, fixed_records):
        normalized.append(record)
        canonical.extend(line)
    if bytes(canonical) != text.encode("utf-8"):
        raise AssociationObservationContractError(
            "association-observation JSONL is not canonically serialized"
        )
    return tuple(normalized)


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""

    if not isinstance(data, bytes):
        raise AssociationObservationContractError("sha256_bytes input must be bytes")
    return hashlib.sha256(data).hexdigest()


def record_sha256(record: Mapping[str, Any]) -> str:
    """Digest one structurally valid record; this does not prove its join."""

    return sha256_bytes(canonical_record_bytes(record))


def document_sha256(
    records: Sequence[Mapping[str, Any]],
    fixed_detection_records: Sequence[Mapping[str, Any]],
) -> str:
    """Stream a fully joined sidecar document into SHA-256."""

    digest = hashlib.sha256()
    for _, line in _validated_canonical_lines(records, fixed_detection_records):
        digest.update(line)
    return digest.hexdigest()


__all__ = [
    "BOX_MEAN_AGGREGATION",
    "CONTRACT_ID",
    "DATASET_NAME",
    "DETECTION_SCORE_ROLE",
    "FORBIDDEN_FIELD_NAMES",
    "MAX_AGENT_COUNT",
    "MAX_CANDIDATES_PER_FRAME",
    "MAX_DOCUMENT_RECORDS",
    "MAX_JSONL_BYTES",
    "MAX_NESTING_DEPTH",
    "MAX_RECORD_BYTES",
    "MAX_SOURCE_OBSERVATIONS_PER_CANDIDATE",
    "RECORD_KIND",
    "RELIABILITY_CALIBRATION",
    "RELIABILITY_VALUE_ROLE",
    "SCHEMA_VERSION",
    "AssociationObservationContractError",
    "canonical_jsonl_bytes",
    "canonical_record_bytes",
    "document_sha256",
    "expected_semantics",
    "loads_jsonl",
    "record_sha256",
    "sha256_bytes",
    "validate_document",
    "validate_record",
]
