"""Strict protocol objects for EventTrack-V2X.

The wire protocol deliberately distinguishes sender-local timestamps from the
receiver-observed arrival time.  A sender timestamp becomes an event time only
after an explicit clock mapping (see :mod:`.clock`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite matrix")
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def _positive_definite(value: np.ndarray, name: str) -> np.ndarray:
    array = _matrix(value, name)
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return array


@dataclass(frozen=True, slots=True)
class LocalTimestamps:
    """Four timestamps produced by a sender's unsynchronised clock."""

    information_cutoff: float
    state_reference: float
    generated: float
    transmitted: float

    def __post_init__(self) -> None:
        names = (
            "information_cutoff",
            "state_reference",
            "generated",
            "transmitted",
        )
        values = tuple(_finite(getattr(self, name), name) for name in names)
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError(
                "local timestamps must satisfy information_cutoff <= "
                "state_reference <= generated <= transmitted"
            )
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class EventTimes:
    """The six semantic times after mapping to a common receiver clock."""

    information_cutoff: float
    state_reference: float
    generated: float
    transmitted: float
    received: float
    deadline: float

    def __post_init__(self) -> None:
        names = (
            "information_cutoff",
            "state_reference",
            "generated",
            "transmitted",
            "received",
            "deadline",
        )
        for name in names:
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        ordered = (
            self.information_cutoff,
            self.state_reference,
            self.generated,
            self.transmitted,
            self.received,
        )
        if any(left > right for left, right in zip(ordered, ordered[1:])):
            raise ValueError(
                "mapped times must satisfy information_cutoff <= state_reference "
                "<= generated <= transmitted <= received"
            )


@dataclass(frozen=True, slots=True)
class Lineage:
    """A conservative summary of factors already absorbed by a payload."""

    factor_ids: tuple[str, ...]
    complete: bool
    ancestor_message_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.complete, bool):
            raise TypeError("lineage.complete must be bool")
        factors = tuple(sorted({_nonempty(item, "factor_id") for item in self.factor_ids}))
        ancestors = tuple(
            sorted({_nonempty(item, "ancestor_message_id") for item in self.ancestor_message_ids})
        )
        if len(factors) != len(self.factor_ids):
            raise ValueError("lineage factor_ids must be unique")
        if len(ancestors) != len(self.ancestor_message_ids):
            raise ValueError("lineage ancestor_message_ids must be unique")
        object.__setattr__(self, "factor_ids", factors)
        object.__setattr__(self, "ancestor_message_ids", ancestors)

    def proves_independent_from(self, other: "Lineage") -> bool:
        """Return true only when both complete summaries have no common factor."""

        return (
            self.complete
            and other.complete
            and set(self.factor_ids).isdisjoint(other.factor_ids)
        )


@dataclass(frozen=True, slots=True)
class IndependentIncrement:
    """A previously unseen, normalised linear-Gaussian likelihood factor."""

    target_id: str
    measurement: np.ndarray
    measurement_matrix: np.ndarray
    measurement_covariance: np.ndarray
    lineage: Lineage
    log_normalizer: float = 0.0
    kind: str = field(init=False, default="independent_increment")

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _nonempty(self.target_id, "target_id"))
        measurement = _vector(self.measurement, "measurement")
        matrix = _matrix(self.measurement_matrix, "measurement_matrix")
        covariance = _positive_definite(
            self.measurement_covariance, "measurement_covariance"
        )
        if matrix.shape[0] != measurement.size:
            raise ValueError("measurement_matrix rows must equal measurement dimension")
        if covariance.shape != (measurement.size, measurement.size):
            raise ValueError("measurement_covariance has incompatible shape")
        if not self.lineage.complete or not self.lineage.factor_ids:
            raise ValueError(
                "independent increments require a complete, non-empty factor lineage"
            )
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "measurement_matrix", matrix)
        object.__setattr__(self, "measurement_covariance", covariance)
        object.__setattr__(
            self, "log_normalizer", _finite(self.log_normalizer, "log_normalizer")
        )


@dataclass(frozen=True, slots=True)
class CorrelatedTrackBelief:
    """A track posterior whose cross-correlation with the receiver is unknown."""

    track_id: str
    existence_probability: float
    mean: np.ndarray
    covariance: np.ndarray
    identity_probabilities: tuple[tuple[str, float], ...]
    lineage: Lineage
    kind: str = field(init=False, default="correlated_track_belief")

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonempty(self.track_id, "track_id"))
        existence = _finite(self.existence_probability, "existence_probability")
        if not 0.0 <= existence <= 1.0:
            raise ValueError("existence_probability must be in [0, 1]")
        mean = _vector(self.mean, "mean")
        covariance = _positive_definite(self.covariance, "covariance")
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance has incompatible shape")
        identities = tuple(
            sorted(
                (
                    _nonempty(label, "identity label"),
                    _finite(probability, "identity probability"),
                )
                for label, probability in self.identity_probabilities
            )
        )
        if not identities:
            raise ValueError("identity_probabilities must not be empty")
        labels = [label for label, _ in identities]
        if len(set(labels)) != len(labels):
            raise ValueError("identity labels must be unique")
        if any(probability < 0.0 for _, probability in identities):
            raise ValueError("identity probabilities must be non-negative")
        if not np.isclose(sum(probability for _, probability in identities), 1.0):
            raise ValueError("identity probabilities must sum to one")
        object.__setattr__(self, "existence_probability", existence)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "identity_probabilities", identities)


Payload = Union[IndependentIncrement, CorrelatedTrackBelief]


@dataclass(frozen=True, slots=True)
class WireMessage:
    """The immutable, fully serialisable sender-side message."""

    message_id: str
    source: str
    sequence: int
    timestamps: LocalTimestamps
    deadline: float
    coordinate_frame: str
    ttl: float
    payload: Payload

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _nonempty(self.message_id, "message_id"))
        object.__setattr__(self, "source", _nonempty(self.source, "source"))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        object.__setattr__(self, "deadline", _finite(self.deadline, "deadline"))
        object.__setattr__(
            self, "coordinate_frame", _nonempty(self.coordinate_frame, "coordinate_frame")
        )
        ttl = _finite(self.ttl, "ttl")
        if ttl <= 0.0:
            raise ValueError("ttl must be positive")
        object.__setattr__(self, "ttl", ttl)
        if not isinstance(self.payload, (IndependentIncrement, CorrelatedTrackBelief)):
            raise TypeError("payload must be an EventTrack-V2X payload")

    @property
    def lineage(self) -> Lineage:
        return self.payload.lineage


@dataclass(frozen=True, slots=True)
class ReceivedMessage:
    """A complete, integrity-checked wire message at the receiver."""

    message: WireMessage
    received_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "received_at", _finite(self.received_at, "received_at"))


__all__ = [
    "CorrelatedTrackBelief",
    "EventTimes",
    "IndependentIncrement",
    "Lineage",
    "LocalTimestamps",
    "Payload",
    "ReceivedMessage",
    "WireMessage",
]
