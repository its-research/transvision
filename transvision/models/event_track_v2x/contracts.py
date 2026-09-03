"""Strict public experiment contracts for EventTrack-V2X.

The contracts in this module deliberately use a small JSON-compatible surface.
Every decoder rejects missing and unknown fields, and every digest is computed
from the canonical representation.  Detection caches do not contain ground
truth or pre-existing track identifiers by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, ClassVar, Mapping, TypeVar, Union

import numpy as np

from .arrays import immutable_float64
from .schema import Lineage
from .wire import canonical_json_bytes


CONTRACT_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLASS_LABEL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


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


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _sha256(value: object, name: str) -> str:
    value = _nonempty(value, name)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _readonly_array(
    value: object,
    name: str,
    *,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    has_wrong_trailing_shape = bool(trailing_shape) and (
        result.shape[-len(trailing_shape) :] != trailing_shape
    )
    if result.ndim != ndim or has_wrong_trailing_shape:
        raise ValueError(
            f"{name} must have {ndim} dimensions ending in {trailing_shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    # A write-protected array that owns its memory can be made writeable again
    # with ``setflags(write=True)``.  Contract values use an immutable bytes
    # backing so callers cannot mutate a cache or committed prediction after
    # its digest has been recorded.  The bytes round-trip also breaks every
    # alias to caller-owned storage.
    return immutable_float64(result)


def _positive_definite_stack(value: object, count: int) -> np.ndarray:
    result = _readonly_array(
        value, "covariances", ndim=3, trailing_shape=(9, 9)
    )
    if result.shape[0] != count:
        raise ValueError("covariances must have one matrix per detection")
    for covariance in result:
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("each detection covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "each detection covariance must be positive definite"
            ) from exc
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, Any]:
    result = _mapping(value, name)
    actual = frozenset(result)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{name} fields do not match schema; missing={missing}, unknown={unknown}"
        )
    return result


def _schema_header(value: Mapping[str, Any], *, kind: str) -> None:
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != CONTRACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported EventTrack-V2X contract schema version")
    if type(value["kind"]) is not str or value["kind"] != kind:
        raise ValueError(f"expected contract kind {kind!r}")


def _lineage_to_primitive(lineage: Lineage) -> dict[str, Any]:
    return {
        "ancestor_message_ids": list(lineage.ancestor_message_ids),
        "complete": lineage.complete,
        "factor_ids": list(lineage.factor_ids),
    }


def _lineage_from_mapping(value: object) -> Lineage:
    result = _strict_fields(
        value,
        frozenset({"ancestor_message_ids", "complete", "factor_ids"}),
        "lineage",
    )
    if not isinstance(result["complete"], bool):
        raise TypeError("lineage.complete must be bool")
    if not isinstance(result["factor_ids"], list) or not isinstance(
        result["ancestor_message_ids"], list
    ):
        raise TypeError("lineage identifiers must be arrays")
    return Lineage(
        factor_ids=tuple(result["factor_ids"]),
        complete=result["complete"],
        ancestor_message_ids=tuple(result["ancestor_message_ids"]),
    )


@dataclass(frozen=True, slots=True, eq=False)
class DetectionCacheV1:
    """One detector-locked frame with a nine-dimensional box state.

    State order is ``x, y, z, length, width, height, yaw, vx, vy``.  The
    covariance uses the same order.  Keeping frame records self-contained makes
    cache cohort hashing and cross-tracker byte equality straightforward.
    """

    sequence_id: str
    frame_id: str
    event_time: float
    agent_id: str
    coordinate_frame: str
    boxes_3d: np.ndarray
    scores: np.ndarray
    class_labels: tuple[str, ...]
    velocities: np.ndarray
    covariances: np.ndarray
    dataset_sha256: str
    detector_config_sha256: str
    checkpoint_sha256: str
    schema_version: int = field(init=False, default=CONTRACT_SCHEMA_VERSION)
    kind: str = field(init=False, default="detection_cache_v1")

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "agent_id",
            "boxes_3d",
            "checkpoint_sha256",
            "class_labels",
            "coordinate_frame",
            "covariances",
            "dataset_sha256",
            "detector_config_sha256",
            "event_time",
            "frame_id",
            "kind",
            "schema_version",
            "scores",
            "sequence_id",
            "velocities",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence_id", _nonempty(self.sequence_id, "sequence_id"))
        object.__setattr__(self, "frame_id", _nonempty(self.frame_id, "frame_id"))
        object.__setattr__(self, "event_time", _finite(self.event_time, "event_time"))
        object.__setattr__(self, "agent_id", _nonempty(self.agent_id, "agent_id"))
        object.__setattr__(
            self,
            "coordinate_frame",
            _nonempty(self.coordinate_frame, "coordinate_frame"),
        )
        boxes = _readonly_array(
            self.boxes_3d, "boxes_3d", ndim=2, trailing_shape=(7,)
        )
        count = boxes.shape[0]
        if count and np.any(boxes[:, 3:6] <= 0.0):
            raise ValueError("box length, width, and height must be positive")
        if count and np.any((boxes[:, 6] < -np.pi) | (boxes[:, 6] >= np.pi)):
            raise ValueError("box yaw must be canonical in [-pi, pi)")
        scores = _readonly_array(self.scores, "scores", ndim=1)
        if scores.shape != (count,):
            raise ValueError("scores must have one value per detection")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("scores must be in [0, 1]")
        if not isinstance(self.class_labels, tuple):
            raise TypeError("class_labels must be a tuple")
        labels = tuple(
            _nonempty(label, "class label") for label in self.class_labels
        )
        if any(_CLASS_LABEL_RE.fullmatch(label) is None for label in labels):
            raise ValueError("class labels must be canonical identifiers")
        if len(labels) != count:
            raise ValueError("class_labels must have one value per detection")
        velocities = _readonly_array(
            self.velocities, "velocities", ndim=2, trailing_shape=(2,)
        )
        if velocities.shape[0] != count:
            raise ValueError("velocities must have one row per detection")
        covariances = _positive_definite_stack(self.covariances, count)
        object.__setattr__(self, "boxes_3d", boxes)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "class_labels", labels)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "covariances", covariances)
        object.__setattr__(
            self, "dataset_sha256", _sha256(self.dataset_sha256, "dataset_sha256")
        )
        object.__setattr__(
            self,
            "detector_config_sha256",
            _sha256(self.detector_config_sha256, "detector_config_sha256"),
        )
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256(self.checkpoint_sha256, "checkpoint_sha256"),
        )

    @property
    def count(self) -> int:
        return self.boxes_3d.shape[0]

    def state_vectors(self) -> np.ndarray:
        return immutable_float64(
            np.concatenate((self.boxes_3d, self.velocities), axis=1)
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "boxes_3d": self.boxes_3d.tolist(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "class_labels": list(self.class_labels),
            "coordinate_frame": self.coordinate_frame,
            "covariances": self.covariances.tolist(),
            "dataset_sha256": self.dataset_sha256,
            "detector_config_sha256": self.detector_config_sha256,
            "event_time": self.event_time,
            "frame_id": self.frame_id,
            "kind": self.kind,
            "schema_version": self.schema_version,
            "scores": self.scores.tolist(),
            "sequence_id": self.sequence_id,
            "velocities": self.velocities.tolist(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "DetectionCacheV1":
        result = _strict_fields(value, cls._FIELDS, cls.__name__)
        _schema_header(result, kind="detection_cache_v1")
        if not isinstance(result["class_labels"], list):
            raise TypeError("class_labels must be an array")
        return cls(
            sequence_id=result["sequence_id"],
            frame_id=result["frame_id"],
            event_time=result["event_time"],
            agent_id=result["agent_id"],
            coordinate_frame=result["coordinate_frame"],
            boxes_3d=result["boxes_3d"],
            scores=result["scores"],
            class_labels=tuple(result["class_labels"]),
            velocities=result["velocities"],
            covariances=result["covariances"],
            dataset_sha256=result["dataset_sha256"],
            detector_config_sha256=result["detector_config_sha256"],
            checkpoint_sha256=result["checkpoint_sha256"],
        )

    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_primitive())).hexdigest()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DetectionCacheV1) and (
            self.to_primitive() == other.to_primitive()
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class IdentityHypothesisV1:
    identity_id: str
    probability: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identity_id", _nonempty(self.identity_id, "identity_id")
        )
        probability = _finite(self.probability, "identity probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("identity probability must be in [0, 1]")
        object.__setattr__(self, "probability", probability)

    def to_primitive(self) -> dict[str, Any]:
        return {"identity_id": self.identity_id, "probability": self.probability}

    @classmethod
    def from_mapping(cls, value: object) -> "IdentityHypothesisV1":
        result = _strict_fields(
            value, frozenset({"identity_id", "probability"}), cls.__name__
        )
        return cls(result["identity_id"], result["probability"])


@dataclass(frozen=True, slots=True, eq=False)
class TrackingPredictionV1:
    """One tracker output at a decision time."""

    sequence_id: str
    frame_id: str
    track_id: str
    class_label: str
    event_time: float
    arrival_time: float
    decision_time: float
    mean: np.ndarray
    covariance: np.ndarray
    existence_probability: float
    identity_hypotheses: tuple[IdentityHypothesisV1, ...]
    lineage: Lineage
    committed: bool
    other_identity_probability: float = 0.0
    schema_version: int = field(init=False, default=CONTRACT_SCHEMA_VERSION)
    kind: str = field(init=False, default="tracking_prediction_v1")

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "arrival_time",
            "class_label",
            "committed",
            "covariance",
            "decision_time",
            "event_time",
            "existence_probability",
            "frame_id",
            "identity_hypotheses",
            "kind",
            "lineage",
            "mean",
            "other_identity_probability",
            "schema_version",
            "sequence_id",
            "track_id",
        }
    )

    def __post_init__(self) -> None:
        for name in ("sequence_id", "frame_id", "track_id", "class_label"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        event_time = _finite(self.event_time, "event_time")
        arrival_time = _finite(self.arrival_time, "arrival_time")
        decision_time = _finite(self.decision_time, "decision_time")
        if event_time > arrival_time or arrival_time > decision_time:
            raise ValueError(
                "prediction times must satisfy event_time <= arrival_time <= decision_time"
            )
        mean = _readonly_array(self.mean, "mean", ndim=1)
        if mean.shape != (9,):
            raise ValueError("prediction mean must use the nine-dimensional state")
        covariance = _readonly_array(
            self.covariance, "covariance", ndim=2, trailing_shape=(9, 9)
        )
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("prediction covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("prediction covariance must be positive definite") from exc
        existence = _finite(self.existence_probability, "existence_probability")
        if not 0.0 <= existence <= 1.0:
            raise ValueError("existence_probability must be in [0, 1]")
        if not isinstance(self.identity_hypotheses, tuple):
            raise TypeError("identity_hypotheses must be a tuple")
        if not self.identity_hypotheses or any(
            not isinstance(item, IdentityHypothesisV1)
            for item in self.identity_hypotheses
        ):
            raise TypeError("identity_hypotheses must contain IdentityHypothesisV1")
        identities = tuple(
            sorted(
                self.identity_hypotheses,
                key=lambda item: (-item.probability, item.identity_id),
            )
        )
        if len({item.identity_id for item in identities}) != len(identities):
            raise ValueError("identity hypotheses must have unique labels")
        other_identity_probability = _finite(
            self.other_identity_probability, "other_identity_probability"
        )
        if not 0.0 <= other_identity_probability <= 1.0:
            raise ValueError("other_identity_probability must be in [0, 1]")
        probability_total = (
            sum(item.probability for item in identities)
            + other_identity_probability
        )
        if abs(probability_total - 1.0) > 1e-12:
            raise ValueError(
                "identity hypotheses plus other probability must sum to one"
            )
        if not isinstance(self.lineage, Lineage):
            raise TypeError("lineage must be Lineage")
        if not self.lineage.factor_ids:
            raise ValueError("prediction lineage must contain at least one factor")
        if not isinstance(self.committed, bool):
            raise TypeError("committed must be bool")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "arrival_time", arrival_time)
        object.__setattr__(self, "decision_time", decision_time)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "existence_probability", existence)
        object.__setattr__(self, "identity_hypotheses", identities)
        object.__setattr__(
            self, "other_identity_probability", other_identity_probability
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "arrival_time": self.arrival_time,
            "class_label": self.class_label,
            "committed": self.committed,
            "covariance": self.covariance.tolist(),
            "decision_time": self.decision_time,
            "event_time": self.event_time,
            "existence_probability": self.existence_probability,
            "frame_id": self.frame_id,
            "identity_hypotheses": [
                hypothesis.to_primitive() for hypothesis in self.identity_hypotheses
            ],
            "kind": self.kind,
            "lineage": _lineage_to_primitive(self.lineage),
            "mean": self.mean.tolist(),
            "other_identity_probability": self.other_identity_probability,
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
            "track_id": self.track_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "TrackingPredictionV1":
        result = _strict_fields(value, cls._FIELDS, cls.__name__)
        _schema_header(result, kind="tracking_prediction_v1")
        if not isinstance(result["identity_hypotheses"], list):
            raise TypeError("identity_hypotheses must be an array")
        if not isinstance(result["committed"], bool):
            raise TypeError("committed must be bool")
        return cls(
            sequence_id=result["sequence_id"],
            frame_id=result["frame_id"],
            track_id=result["track_id"],
            class_label=result["class_label"],
            event_time=result["event_time"],
            arrival_time=result["arrival_time"],
            decision_time=result["decision_time"],
            mean=result["mean"],
            covariance=result["covariance"],
            existence_probability=result["existence_probability"],
            identity_hypotheses=tuple(
                IdentityHypothesisV1.from_mapping(item)
                for item in result["identity_hypotheses"]
            ),
            lineage=_lineage_from_mapping(result["lineage"]),
            committed=result["committed"],
            other_identity_probability=result["other_identity_probability"],
        )

    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_primitive())).hexdigest()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TrackingPredictionV1) and (
            self.to_primitive() == other.to_primitive()
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class EvaluatorContractV1:
    dataset_name: str
    frequency_hz: float
    split_name: str
    sample_count: int
    roi: tuple[float, float, float, float, float, float]
    class_mapping: tuple[tuple[str, str], ...]
    matching_thresholds: tuple[tuple[str, float], ...]
    evaluator_name: str
    evaluator_version: str
    cohort_sha256: str
    schema_version: int = field(init=False, default=CONTRACT_SCHEMA_VERSION)
    kind: str = field(init=False, default="evaluator_contract_v1")

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "class_mapping",
            "cohort_sha256",
            "dataset_name",
            "evaluator_name",
            "evaluator_version",
            "frequency_hz",
            "kind",
            "matching_thresholds",
            "roi",
            "sample_count",
            "schema_version",
            "split_name",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "dataset_name",
            "split_name",
            "evaluator_name",
            "evaluator_version",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        frequency = _finite(self.frequency_hz, "frequency_hz")
        if frequency <= 0.0:
            raise ValueError("frequency_hz must be positive")
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(
            self, "sample_count", _integer(self.sample_count, "sample_count", minimum=1)
        )
        if not isinstance(self.roi, tuple) or len(self.roi) != 6:
            raise ValueError("roi must be a six-value tuple")
        roi = tuple(_finite(value, "roi value") for value in self.roi)
        if any(roi[index] >= roi[index + 3] for index in range(3)):
            raise ValueError("roi minima must be smaller than maxima")
        object.__setattr__(self, "roi", roi)
        if not isinstance(self.class_mapping, tuple):
            raise TypeError("class_mapping must be a tuple")
        mapping = tuple(
            sorted(
                (
                    _nonempty(source, "source class"),
                    _nonempty(target, "target class"),
                )
                for source, target in self.class_mapping
            )
        )
        if not mapping or len({source for source, _ in mapping}) != len(mapping):
            raise ValueError("class_mapping must be non-empty with unique sources")
        object.__setattr__(self, "class_mapping", mapping)
        if not isinstance(self.matching_thresholds, tuple):
            raise TypeError("matching_thresholds must be a tuple")
        thresholds = tuple(
            sorted(
                (
                    _nonempty(metric, "matching metric"),
                    _finite(threshold, "matching threshold"),
                )
                for metric, threshold in self.matching_thresholds
            )
        )
        if not thresholds or len({metric for metric, _ in thresholds}) != len(
            thresholds
        ):
            raise ValueError(
                "matching_thresholds must be non-empty with unique metrics"
            )
        if any(threshold <= 0.0 for _, threshold in thresholds):
            raise ValueError("matching thresholds must be positive")
        object.__setattr__(self, "matching_thresholds", thresholds)
        object.__setattr__(
            self, "cohort_sha256", _sha256(self.cohort_sha256, "cohort_sha256")
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "class_mapping": [list(item) for item in self.class_mapping],
            "cohort_sha256": self.cohort_sha256,
            "dataset_name": self.dataset_name,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "frequency_hz": self.frequency_hz,
            "kind": self.kind,
            "matching_thresholds": [list(item) for item in self.matching_thresholds],
            "roi": list(self.roi),
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "split_name": self.split_name,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "EvaluatorContractV1":
        result = _strict_fields(value, cls._FIELDS, cls.__name__)
        _schema_header(result, kind="evaluator_contract_v1")
        if not all(
            isinstance(result[name], list)
            for name in ("roi", "class_mapping", "matching_thresholds")
        ):
            raise TypeError("roi, class_mapping, and matching_thresholds must be arrays")
        for name in ("class_mapping", "matching_thresholds"):
            if any(
                not isinstance(item, list) or len(item) != 2
                for item in result[name]
            ):
                raise TypeError(f"{name} entries must be two-value arrays")
        return cls(
            dataset_name=result["dataset_name"],
            frequency_hz=result["frequency_hz"],
            split_name=result["split_name"],
            sample_count=result["sample_count"],
            roi=tuple(result["roi"]),  # type: ignore[arg-type]
            class_mapping=tuple(tuple(item) for item in result["class_mapping"]),
            matching_thresholds=tuple(
                tuple(item) for item in result["matching_thresholds"]
            ),
            evaluator_name=result["evaluator_name"],
            evaluator_version=result["evaluator_version"],
            cohort_sha256=result["cohort_sha256"],
        )

    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_primitive())).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactDigestV1:
    uri: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _nonempty(self.uri, "artifact uri"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "artifact sha256"))
        object.__setattr__(
            self, "byte_size", _integer(self.byte_size, "artifact byte_size")
        )

    def to_primitive(self) -> dict[str, Any]:
        return {"byte_size": self.byte_size, "sha256": self.sha256, "uri": self.uri}

    @classmethod
    def from_mapping(cls, value: object) -> "ArtifactDigestV1":
        result = _strict_fields(
            value, frozenset({"byte_size", "sha256", "uri"}), cls.__name__
        )
        return cls(result["uri"], result["sha256"], result["byte_size"])


@dataclass(frozen=True, slots=True)
class EvidenceBundleV1:
    """Complete digest binding for one publishable experiment run."""

    run_id: str
    source: ArtifactDigestV1
    dataset: ArtifactDigestV1
    detection_cache: ArtifactDigestV1
    network_trace: ArtifactDigestV1
    tracker_config: ArtifactDigestV1
    evaluator_contract: ArtifactDigestV1
    checkpoint: ArtifactDigestV1
    predictions: ArtifactDigestV1
    per_sequence_metrics: ArtifactDigestV1
    logs: ArtifactDigestV1
    environment: ArtifactDigestV1
    schema_version: int = field(init=False, default=CONTRACT_SCHEMA_VERSION)
    kind: str = field(init=False, default="evidence_bundle_v1")

    _ARTIFACT_FIELDS: ClassVar[tuple[str, ...]] = (
        "source",
        "dataset",
        "detection_cache",
        "network_trace",
        "tracker_config",
        "evaluator_contract",
        "checkpoint",
        "predictions",
        "per_sequence_metrics",
        "logs",
        "environment",
    )
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"kind", "run_id", "schema_version", *_ARTIFACT_FIELDS}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        for name in self._ARTIFACT_FIELDS:
            if not isinstance(getattr(self, name), ArtifactDigestV1):
                raise TypeError(f"{name} must be ArtifactDigestV1")

    def to_primitive(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
        }
        result.update(
            {
                name: getattr(self, name).to_primitive()
                for name in self._ARTIFACT_FIELDS
            }
        )
        return result

    @classmethod
    def from_mapping(cls, value: object) -> "EvidenceBundleV1":
        result = _strict_fields(value, cls._FIELDS, cls.__name__)
        _schema_header(result, kind="evidence_bundle_v1")
        return cls(
            run_id=result["run_id"],
            **{
                name: ArtifactDigestV1.from_mapping(result[name])
                for name in cls._ARTIFACT_FIELDS
            },
        )

    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_primitive())).hexdigest()


ContractV1 = Union[
    DetectionCacheV1,
    TrackingPredictionV1,
    EvaluatorContractV1,
    EvidenceBundleV1,
]
_ContractT = TypeVar("_ContractT", bound=ContractV1)


def encode_contract(contract: ContractV1) -> bytes:
    if not isinstance(
        contract,
        (
            DetectionCacheV1,
            TrackingPredictionV1,
            EvaluatorContractV1,
            EvidenceBundleV1,
        ),
    ):
        raise TypeError("unsupported EventTrack-V2X experiment contract")
    return canonical_json_bytes(contract.to_primitive())


def decode_contract(data: bytes, expected_type: type[_ContractT]) -> _ContractT:
    """Decode an exact contract type and reject non-canonical bytes."""

    if not isinstance(data, bytes):
        raise TypeError("contract data must be bytes")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid EventTrack-V2X experiment contract") from exc
    supported: tuple[type[ContractV1], ...] = (
        DetectionCacheV1,
        TrackingPredictionV1,
        EvaluatorContractV1,
        EvidenceBundleV1,
    )
    if expected_type not in supported:
        raise TypeError("expected_type is not a public EventTrack-V2X contract")
    contract = expected_type.from_mapping(value)
    if encode_contract(contract) != data:
        raise ValueError("contract is valid JSON but not canonical")
    return contract


def contract_digest(contract: ContractV1) -> str:
    return hashlib.sha256(encode_contract(contract)).hexdigest()


__all__ = [
    "ArtifactDigestV1",
    "CONTRACT_SCHEMA_VERSION",
    "ContractV1",
    "DetectionCacheV1",
    "EvaluatorContractV1",
    "EvidenceBundleV1",
    "IdentityHypothesisV1",
    "TrackingPredictionV1",
    "contract_digest",
    "decode_contract",
    "encode_contract",
]
