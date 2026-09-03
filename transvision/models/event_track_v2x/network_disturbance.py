"""Materialize preregistered network disturbances into tracker inputs.

The synthetic trace generator records clock and pose errors, but a recorded
error is not an experimental treatment until it changes the detection input
consumed by a tracker.  This module performs that deterministic realization.
It never reads ground truth and never mutates the detector-locked source
cache.

Clock error is added to the detector event time without clipping.  Therefore
a positive error may create an event time later than packet arrival.  The
realized object is retained as evidence; :func:`require_causal_at_arrival_v1`
rejects it before tracker ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, ClassVar, Iterable, Mapping

import numpy as np

from .contracts import DetectionCacheV1
from .network import (
    NetworkConditionId,
    NetworkTraceEventV1,
    NetworkTraceV1,
    condition_plan_v1,
)
from .wire import canonical_json_bytes


CONDITION_INPUT_REALIZATION_SCHEMA_V1 = (
    "eventtrack-v2x.condition-input-realization.v1"
)
CONDITION_INPUT_MANIFEST_SCHEMA_V1 = (
    "eventtrack-v2x.condition-input-manifest.v1"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")

_REALIZATION_CONTRACT_V1: dict[str, Any] = {
    "clock_rule": (
        "realized_event_time=cache_event_time+"
        "clock_error_at_transmit_seconds"
    ),
    "covariance_rule": "P_realized=J*P_cache*J_transpose",
    "ground_truth_inputs": [],
    "pose_rule": (
        "center_xy=R(yaw_noise)*center_xy+translation_xy;"
        "center_z=center_z+translation_z;"
        "yaw=canonical(cache_yaw+yaw_noise);"
        "velocity_xy=R(yaw_noise)*velocity_xy"
    ),
    "schema_version": CONDITION_INPUT_REALIZATION_SCHEMA_V1,
    "state_order": ["x", "y", "z", "length", "width", "height", "yaw", "vx", "vy"],
    "yaw_interval": "[-pi,pi)",
}
REALIZATION_CONTRACT_SHA256_V1 = hashlib.sha256(
    canonical_json_bytes(_REALIZATION_CONTRACT_V1)
).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a trimmed non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _identifier(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return result


def _strict_mapping(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise ValueError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        missing = sorted(expected - frozenset(value))
        unknown = sorted(frozenset(value) - expected)
        raise ValueError(
            f"{name} fields do not match schema; missing={missing}, unknown={unknown}"
        )
    return value


def _trace_event_sha256(event: NetworkTraceEventV1) -> str:
    return hashlib.sha256(canonical_json_bytes(event.to_dict())).hexdigest()


def _canonical_yaw(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _condition_supports_recorded_disturbance(
    condition_id: NetworkConditionId,
    event: NetworkTraceEventV1,
) -> None:
    clock_values = (
        event.clock_offset_seconds,
        event.clock_drift_ppm,
        event.clock_error_at_transmit_seconds,
    )
    pose_values = (*event.translation_noise_metres, event.yaw_noise_degrees)
    if condition_id not in (NetworkConditionId.C6, NetworkConditionId.C8) and any(
        value != 0.0 for value in clock_values
    ):
        raise ValueError(
            f"{condition_id.value} cannot contain a clock disturbance"
        )
    if condition_id not in (NetworkConditionId.C7, NetworkConditionId.C8) and any(
        value != 0.0 for value in pose_values
    ):
        raise ValueError(
            f"{condition_id.value} cannot contain a pose disturbance"
        )


@dataclass(frozen=True, slots=True)
class ConditionInputEvidenceV1:
    """Canonical provenance for one realized detection input."""

    original_cache_sha256: str
    realized_cache_sha256: str
    network_trace_sha256: str
    trace_event_sha256: str
    condition_id: NetworkConditionId
    trace_seed: int
    message_id: str
    source: str
    packet_sequence: int
    network_transmit_time: float
    arrival_time: float
    original_event_time: float
    realized_event_time: float
    clock_offset_seconds: float
    clock_drift_ppm: float
    clock_error_at_transmit_seconds: float
    translation_noise_metres: tuple[float, float, float]
    yaw_noise_degrees: float
    realization_contract_sha256: str = field(
        init=False,
        default=REALIZATION_CONTRACT_SHA256_V1,
    )
    ground_truth_free: bool = field(init=False, default=True)
    schema_version: str = field(
        init=False,
        default=CONDITION_INPUT_REALIZATION_SCHEMA_V1,
    )
    kind: str = field(init=False, default="condition_input_evidence_v1")

    _SHA_FIELDS: ClassVar[tuple[str, ...]] = (
        "original_cache_sha256",
        "realized_cache_sha256",
        "network_trace_sha256",
        "trace_event_sha256",
        "realization_contract_sha256",
    )

    def __post_init__(self) -> None:
        for name in self._SHA_FIELDS:
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "condition_id", NetworkConditionId(self.condition_id))
        if isinstance(self.trace_seed, bool) or not isinstance(self.trace_seed, int):
            raise TypeError("trace_seed must be an integer")
        if self.trace_seed < 0:
            raise ValueError("trace_seed must be non-negative")
        if isinstance(self.packet_sequence, bool) or not isinstance(
            self.packet_sequence, int
        ):
            raise TypeError("packet_sequence must be an integer")
        if self.packet_sequence < 0:
            raise ValueError("packet_sequence must be non-negative")
        object.__setattr__(self, "message_id", _nonempty(self.message_id, "message_id"))
        object.__setattr__(self, "source", _nonempty(self.source, "source"))
        for name in (
            "network_transmit_time",
            "arrival_time",
            "original_event_time",
            "realized_event_time",
            "clock_offset_seconds",
            "clock_drift_ppm",
            "clock_error_at_transmit_seconds",
            "yaw_noise_degrees",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        translation = tuple(
            _finite(value, "translation_noise_metres")
            for value in self.translation_noise_metres
        )
        if len(translation) != 3:
            raise ValueError(
                "translation_noise_metres must contain exactly three values"
            )
        object.__setattr__(self, "translation_noise_metres", translation)
        if self.arrival_time < self.network_transmit_time:
            raise ValueError("arrival_time cannot precede network_transmit_time")
        expected_event_time = (
            self.original_event_time + self.clock_error_at_transmit_seconds
        )
        if self.realized_event_time != expected_event_time:
            raise ValueError(
                "realized_event_time must apply the recorded clock error exactly"
            )

    @property
    def causal_at_arrival(self) -> bool:
        return self.realized_event_time <= self.arrival_time

    def to_primitive(self) -> dict[str, Any]:
        return {
            "arrival_time": self.arrival_time,
            "causal_at_arrival": self.causal_at_arrival,
            "clock_drift_ppm": self.clock_drift_ppm,
            "clock_error_at_transmit_seconds": self.clock_error_at_transmit_seconds,
            "clock_offset_seconds": self.clock_offset_seconds,
            "condition_id": self.condition_id.value,
            "ground_truth_free": self.ground_truth_free,
            "kind": self.kind,
            "message_id": self.message_id,
            "network_trace_sha256": self.network_trace_sha256,
            "network_transmit_time": self.network_transmit_time,
            "original_cache_sha256": self.original_cache_sha256,
            "original_event_time": self.original_event_time,
            "packet_sequence": self.packet_sequence,
            "realization_contract_sha256": self.realization_contract_sha256,
            "realized_cache_sha256": self.realized_cache_sha256,
            "realized_event_time": self.realized_event_time,
            "schema_version": self.schema_version,
            "source": self.source,
            "trace_event_sha256": self.trace_event_sha256,
            "trace_seed": self.trace_seed,
            "translation_noise_metres": list(self.translation_noise_metres),
            "yaw_noise_degrees": self.yaw_noise_degrees,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class ConditionInputManifestV1:
    """Exact ordered disturbance-input evidence for one C6--C8 run."""

    run_id: str
    condition_id: NetworkConditionId
    network_trace_sha256: str
    condition_plan_sha256: str
    run_config_sha256: str
    detection_cache_sha256: str
    evidence_sha256s: tuple[str, ...]
    realization_contract_sha256: str = field(
        init=False,
        default=REALIZATION_CONTRACT_SHA256_V1,
    )
    schema_version: str = field(
        init=False,
        default=CONDITION_INPUT_MANIFEST_SCHEMA_V1,
    )
    kind: str = field(init=False, default="condition_input_manifest_v1")

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "condition_id",
            "condition_plan_sha256",
            "detection_cache_sha256",
            "evidence_sha256s",
            "kind",
            "network_trace_sha256",
            "realization_contract_sha256",
            "run_config_sha256",
            "run_id",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        condition = NetworkConditionId(self.condition_id)
        if condition not in {
            NetworkConditionId.C6,
            NetworkConditionId.C7,
            NetworkConditionId.C8,
        }:
            raise ValueError("condition input manifest is restricted to C6-C8")
        object.__setattr__(self, "condition_id", condition)
        for name in (
            "network_trace_sha256",
            "condition_plan_sha256",
            "run_config_sha256",
            "detection_cache_sha256",
            "realization_contract_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if (
            self.condition_plan_sha256
            != condition_plan_v1(condition).content_sha256
        ):
            raise ValueError(
                "condition_plan_sha256 does not match the registered condition"
            )
        if self.realization_contract_sha256 != REALIZATION_CONTRACT_SHA256_V1:
            raise ValueError("realization contract is not V1")
        if not isinstance(self.evidence_sha256s, (list, tuple)):
            raise TypeError("evidence_sha256s must be an array")
        evidence = tuple(
            _sha256(item, "evidence_sha256") for item in self.evidence_sha256s
        )
        if not evidence:
            raise ValueError("evidence_sha256s must be non-empty")
        if len(evidence) != len(set(evidence)):
            raise ValueError("evidence_sha256s must be unique")
        object.__setattr__(self, "evidence_sha256s", evidence)

    def to_primitive(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id.value,
            "condition_plan_sha256": self.condition_plan_sha256,
            "detection_cache_sha256": self.detection_cache_sha256,
            "evidence_sha256s": list(self.evidence_sha256s),
            "kind": self.kind,
            "network_trace_sha256": self.network_trace_sha256,
            "realization_contract_sha256": self.realization_contract_sha256,
            "run_config_sha256": self.run_config_sha256,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "ConditionInputManifestV1":
        item = _strict_mapping(value, cls._FIELDS, cls.__name__)
        if (
            item["kind"] != "condition_input_manifest_v1"
            or item["schema_version"] != CONDITION_INPUT_MANIFEST_SCHEMA_V1
            or item["realization_contract_sha256"]
            != REALIZATION_CONTRACT_SHA256_V1
        ):
            raise ValueError("unsupported condition input manifest schema")
        evidence = item["evidence_sha256s"]
        if not isinstance(evidence, list):
            raise ValueError("evidence_sha256s must be an array")
        return cls(
            run_id=item["run_id"],  # type: ignore[arg-type]
            condition_id=item["condition_id"],  # type: ignore[arg-type]
            network_trace_sha256=item["network_trace_sha256"],  # type: ignore[arg-type]
            condition_plan_sha256=item["condition_plan_sha256"],  # type: ignore[arg-type]
            run_config_sha256=item["run_config_sha256"],  # type: ignore[arg-type]
            detection_cache_sha256=item["detection_cache_sha256"],  # type: ignore[arg-type]
            evidence_sha256s=tuple(evidence),  # type: ignore[arg-type]
        )


def build_condition_input_manifest_v1(
    *,
    run_id: str,
    trace: NetworkTraceV1,
    run_config_sha256: str,
    detection_cache_sha256: str,
    evidence: Iterable[ConditionInputEvidenceV1],
) -> ConditionInputManifestV1:
    """Build one manifest after checking every ordered evidence/trace binding."""

    if not isinstance(trace, NetworkTraceV1):
        raise TypeError("trace must be NetworkTraceV1")
    records = tuple(evidence)
    if not records or not all(
        isinstance(item, ConditionInputEvidenceV1) for item in records
    ):
        raise TypeError(
            "evidence must be a non-empty iterable of ConditionInputEvidenceV1"
        )
    trace_sha256 = trace.content_sha256
    for item in records:
        if (
            item.condition_id is not trace.condition_id
            or item.network_trace_sha256 != trace_sha256
        ):
            raise ValueError(
                "condition input evidence does not match the exact run trace"
            )
    return ConditionInputManifestV1(
        run_id=run_id,
        condition_id=trace.condition_id,
        network_trace_sha256=trace_sha256,
        condition_plan_sha256=trace.condition_plan_sha256,
        run_config_sha256=run_config_sha256,
        detection_cache_sha256=detection_cache_sha256,
        evidence_sha256s=tuple(item.content_sha256 for item in records),
    )


def decode_condition_input_manifest_v1(data: bytes) -> ConditionInputManifestV1:
    """Decode canonical JSON while rejecting duplicates and non-finite values."""

    if type(data) is not bytes:
        raise TypeError("condition input manifest data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid ConditionInputManifestV1 JSON") from exc
    manifest = ConditionInputManifestV1.from_mapping(value)
    if manifest.canonical_bytes != data:
        raise ValueError("ConditionInputManifestV1 JSON is not canonical")
    return manifest


def decode_condition_input_manifest_inventory_v1(
    data: bytes,
) -> dict[str, ConditionInputManifestV1]:
    """Decode a canonical content-digest-to-manifest inventory.

    The inventory is the artifact opened by formal registry verifiers.  It is
    intentionally non-empty and rejects duplicate keys, unknown/noncanonical
    JSON encodings, and any key that is not the manifest's actual content hash.
    """

    if type(data) is not bytes:
        raise TypeError("condition input manifest inventory data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid condition input manifest inventory JSON") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError("condition input manifest inventory must be a non-empty object")
    manifests: dict[str, ConditionInputManifestV1] = {}
    for raw_digest, document in value.items():
        digest = _sha256(raw_digest, "condition input manifest inventory key")
        manifest = ConditionInputManifestV1.from_mapping(document)
        if digest != manifest.content_sha256:
            raise ValueError(
                "condition input manifest inventory key does not match content"
            )
        manifests[digest] = manifest
    canonical = canonical_json_bytes(
        {digest: manifest.to_primitive() for digest, manifest in manifests.items()}
    )
    if canonical != data:
        raise ValueError("condition input manifest inventory JSON is not canonical")
    return manifests


@dataclass(frozen=True, slots=True)
class ConditionRealizedDetectionV1:
    """A tracker-ready cache plus its exact disturbance provenance."""

    detection_cache: DetectionCacheV1
    evidence: ConditionInputEvidenceV1
    schema_version: str = field(
        init=False,
        default=CONDITION_INPUT_REALIZATION_SCHEMA_V1,
    )
    kind: str = field(init=False, default="condition_realized_detection_v1")

    def __post_init__(self) -> None:
        if not isinstance(self.detection_cache, DetectionCacheV1):
            raise TypeError("detection_cache must be DetectionCacheV1")
        if not isinstance(self.evidence, ConditionInputEvidenceV1):
            raise TypeError("evidence must be ConditionInputEvidenceV1")
        if self.detection_cache.digest() != self.evidence.realized_cache_sha256:
            raise ValueError("detection_cache does not match realized_cache_sha256")
        if self.detection_cache.agent_id != self.evidence.source:
            raise ValueError("detection cache agent does not match trace source")
        if self.detection_cache.event_time != self.evidence.realized_event_time:
            raise ValueError("detection cache event time does not match evidence")

    @property
    def causal_at_arrival(self) -> bool:
        return self.evidence.causal_at_arrival

    def to_primitive(self) -> dict[str, Any]:
        return {
            "detection_cache": self.detection_cache.to_primitive(),
            "evidence": self.evidence.to_primitive(),
            "evidence_sha256": self.evidence.content_sha256,
            "kind": self.kind,
            "schema_version": self.schema_version,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def _realize_cache(
    cache: DetectionCacheV1,
    event: NetworkTraceEventV1,
) -> DetectionCacheV1:
    clock_error = event.clock_error_at_transmit_seconds
    translation = np.asarray(event.translation_noise_metres, dtype=np.float64)
    yaw_radians = float(np.deg2rad(event.yaw_noise_degrees))
    if clock_error == 0.0 and yaw_radians == 0.0 and np.all(translation == 0.0):
        return cache

    boxes = np.array(cache.boxes_3d, dtype=np.float64, copy=True)
    velocities = np.array(cache.velocities, dtype=np.float64, copy=True)
    covariances = np.array(cache.covariances, dtype=np.float64, copy=True)
    if yaw_radians != 0.0 or np.any(translation != 0.0):
        cosine = float(np.cos(yaw_radians))
        sine = float(np.sin(yaw_radians))
        rotation = np.asarray(
            [[cosine, -sine], [sine, cosine]],
            dtype=np.float64,
        )
        boxes[:, :2] = boxes[:, :2] @ rotation.T + translation[:2]
        boxes[:, 2] += translation[2]
        boxes[:, 6] = _canonical_yaw(boxes[:, 6] + yaw_radians)
        velocities = velocities @ rotation.T
        jacobian = np.eye(9, dtype=np.float64)
        jacobian[np.ix_((0, 1), (0, 1))] = rotation
        jacobian[np.ix_((7, 8), (7, 8))] = rotation
        covariances = np.einsum(
            "ij,njk,lk->nil",
            jacobian,
            covariances,
            jacobian,
        )
        covariances = 0.5 * (
            covariances + np.swapaxes(covariances, 1, 2)
        )

    return DetectionCacheV1(
        sequence_id=cache.sequence_id,
        frame_id=cache.frame_id,
        event_time=cache.event_time + clock_error,
        agent_id=cache.agent_id,
        coordinate_frame=cache.coordinate_frame,
        boxes_3d=boxes,
        scores=cache.scores,
        class_labels=cache.class_labels,
        velocities=velocities,
        covariances=covariances,
        dataset_sha256=cache.dataset_sha256,
        detector_config_sha256=cache.detector_config_sha256,
        checkpoint_sha256=cache.checkpoint_sha256,
    )


def realize_condition_input_v1(
    cache: DetectionCacheV1,
    trace: NetworkTraceV1,
    event: NetworkTraceEventV1,
) -> ConditionRealizedDetectionV1:
    """Apply one delivered trace event to one detector-locked cache frame.

    The exact event must be a member of ``trace``.  Matching only by source or
    sequence is insufficient because it could silently apply a disturbance
    sampled for another message.
    """

    if not isinstance(cache, DetectionCacheV1):
        raise TypeError("cache must be DetectionCacheV1")
    if not isinstance(trace, NetworkTraceV1):
        raise TypeError("trace must be NetworkTraceV1")
    if not isinstance(event, NetworkTraceEventV1):
        raise TypeError("event must be NetworkTraceEventV1")
    if event.dropped or event.arrival_time is None:
        raise ValueError("only a delivered network event can be materialized")
    if event.packet.source != cache.agent_id:
        raise ValueError("trace source must match detection cache agent_id")
    if trace.condition_plan_sha256 != condition_plan_v1(
        trace.condition_id
    ).content_sha256:
        raise ValueError("trace does not use the preregistered condition plan")
    matching = tuple(
        item
        for item in trace.events
        if item.packet.message_id == event.packet.message_id
    )
    if len(matching) != 1 or _trace_event_sha256(matching[0]) != _trace_event_sha256(
        event
    ):
        raise ValueError("trace event is not the exact event sealed by the trace")
    _condition_supports_recorded_disturbance(trace.condition_id, event)

    realized_cache = _realize_cache(cache, event)
    evidence = ConditionInputEvidenceV1(
        original_cache_sha256=cache.digest(),
        realized_cache_sha256=realized_cache.digest(),
        network_trace_sha256=trace.content_sha256,
        trace_event_sha256=_trace_event_sha256(event),
        condition_id=trace.condition_id,
        trace_seed=trace.seed,
        message_id=event.packet.message_id,
        source=event.packet.source,
        packet_sequence=event.packet.sequence,
        network_transmit_time=event.packet.transmitted_at,
        arrival_time=event.arrival_time,
        original_event_time=cache.event_time,
        realized_event_time=realized_cache.event_time,
        clock_offset_seconds=event.clock_offset_seconds,
        clock_drift_ppm=event.clock_drift_ppm,
        clock_error_at_transmit_seconds=event.clock_error_at_transmit_seconds,
        translation_noise_metres=event.translation_noise_metres,
        yaw_noise_degrees=event.yaw_noise_degrees,
    )
    return ConditionRealizedDetectionV1(realized_cache, evidence)


def require_causal_at_arrival_v1(
    value: ConditionRealizedDetectionV1,
) -> DetectionCacheV1:
    """Return a tracker input only when its realized event time is causal."""

    if not isinstance(value, ConditionRealizedDetectionV1):
        raise TypeError("value must be ConditionRealizedDetectionV1")
    if not value.causal_at_arrival:
        raise ValueError(
            "realized event_time is later than packet arrival; future input rejected"
        )
    return value.detection_cache


__all__ = [
    "CONDITION_INPUT_MANIFEST_SCHEMA_V1",
    "CONDITION_INPUT_REALIZATION_SCHEMA_V1",
    "REALIZATION_CONTRACT_SHA256_V1",
    "ConditionInputEvidenceV1",
    "ConditionInputManifestV1",
    "ConditionRealizedDetectionV1",
    "build_condition_input_manifest_v1",
    "decode_condition_input_manifest_v1",
    "decode_condition_input_manifest_inventory_v1",
    "realize_condition_input_v1",
    "require_causal_at_arrival_v1",
]
