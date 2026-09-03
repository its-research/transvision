"""Affine sender-clock mapping and conservative causality gates."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .arrays import immutable_float64
from .schema import EventTimes, ReceivedMessage


@dataclass(frozen=True, slots=True)
class TimeEstimate:
    mean: float
    variance: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("time mean must be finite")
        if not np.isfinite(self.variance) or self.variance < 0.0:
            raise ValueError("time variance must be finite and non-negative")

    @property
    def standard_deviation(self) -> float:
        return float(np.sqrt(self.variance))

    def upper_quantile(self, confidence: float) -> float:
        if not 0.5 < confidence < 1.0:
            raise ValueError("confidence must be in (0.5, 1)")
        return self.mean + NormalDist().inv_cdf(confidence) * self.standard_deviation


@dataclass(frozen=True, slots=True)
class MappedEventTimes:
    information_cutoff: TimeEstimate
    state_reference: TimeEstimate
    generated: TimeEstimate
    transmitted: TimeEstimate
    received: float
    deadline: float

    def at_means(self) -> EventTimes:
        return EventTimes(
            information_cutoff=self.information_cutoff.mean,
            state_reference=self.state_reference.mean,
            generated=self.generated.mean,
            transmitted=self.transmitted.mean,
            received=self.received,
            deadline=self.deadline,
        )


@dataclass(frozen=True, slots=True)
class AffineClockMap:
    """Map ``local = scale * receiver + offset + noise`` to receiver time.

    ``parameter_covariance`` is the covariance of ``(scale, offset)``.  The
    returned variance is first-order plug-in propagation and is named as such;
    it is not a joint clock/source-time posterior.
    """

    scale: float
    offset: float
    parameter_covariance: np.ndarray
    timestamp_variance: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("clock scale must be finite and positive")
        if not np.isfinite(self.offset):
            raise ValueError("clock offset must be finite")
        covariance = np.asarray(self.parameter_covariance, dtype=np.float64)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise ValueError("parameter_covariance must be a finite 2x2 matrix")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("parameter_covariance must be symmetric")
        if np.linalg.eigvalsh(covariance).min() < -1e-12:
            raise ValueError("parameter_covariance must be positive semidefinite")
        timestamp_variance = float(self.timestamp_variance)
        if not np.isfinite(timestamp_variance) or timestamp_variance < 0.0:
            raise ValueError("timestamp_variance must be finite and non-negative")
        covariance = immutable_float64(covariance)
        object.__setattr__(self, "parameter_covariance", covariance)
        object.__setattr__(self, "timestamp_variance", timestamp_variance)

    def map_timestamp(self, local_timestamp: float) -> TimeEstimate:
        local_timestamp = float(local_timestamp)
        if not np.isfinite(local_timestamp):
            raise ValueError("local_timestamp must be finite")
        mean = (local_timestamp - self.offset) / self.scale
        gradient = np.asarray(
            [-mean / self.scale, -1.0 / self.scale], dtype=np.float64
        )
        variance = float(
            gradient @ self.parameter_covariance @ gradient
            + self.timestamp_variance / (self.scale * self.scale)
        )
        return TimeEstimate(mean=mean, variance=max(0.0, variance))

    def map_message(self, received: ReceivedMessage) -> MappedEventTimes:
        local = received.message.timestamps
        return MappedEventTimes(
            information_cutoff=self.map_timestamp(local.information_cutoff),
            state_reference=self.map_timestamp(local.state_reference),
            generated=self.map_timestamp(local.generated),
            transmitted=self.map_timestamp(local.transmitted),
            received=received.received_at,
            deadline=received.message.deadline,
        )


def conservative_no_future_gate(
    estimate: TimeEstimate,
    decision_time: float,
    *,
    confidence: float = 0.999,
) -> bool:
    """Apply the probabilistic no-future gate to a receiver-mapped timestamp.

    This low-level helper deliberately has no caller-supplied override for the
    upper bound.  A certified interval needs its own authenticated protocol
    object; treating an arbitrary number as a certificate would let callers
    bypass the causality check.
    """

    decision_time = float(decision_time)
    if not np.isfinite(decision_time):
        raise ValueError("decision_time must be finite")
    return estimate.upper_quantile(confidence) <= decision_time


__all__ = [
    "AffineClockMap",
    "MappedEventTimes",
    "TimeEstimate",
    "conservative_no_future_gate",
]
