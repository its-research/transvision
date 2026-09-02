"""Fixed-lag chronological replay for independent Gaussian increments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .clock import AffineClockMap, conservative_no_future_gate
from .schema import CorrelatedTrackBelief, IndependentIncrement, ReceivedMessage


MatrixFunction = Callable[[float], np.ndarray]


@dataclass(frozen=True, slots=True)
class LinearGaussianDynamics:
    state_dimension: int
    transition: MatrixFunction
    process_covariance: MatrixFunction

    def __post_init__(self) -> None:
        if (
            isinstance(self.state_dimension, bool)
            or not isinstance(self.state_dimension, int)
            or self.state_dimension <= 0
        ):
            raise ValueError("state_dimension must be a positive integer")
        for delta in (0.0, 0.125, 1.0):
            transition = np.asarray(self.transition(delta), dtype=np.float64)
            process = np.asarray(self.process_covariance(delta), dtype=np.float64)
            expected = (self.state_dimension, self.state_dimension)
            if transition.shape != expected or process.shape != expected:
                raise ValueError("dynamics functions returned an incompatible matrix")
            if not np.all(np.isfinite(transition)) or not np.all(np.isfinite(process)):
                raise ValueError("dynamics matrices must be finite")
            if not np.allclose(process, process.T, rtol=1e-10, atol=1e-12):
                raise ValueError("process covariance must be symmetric")
            if np.linalg.eigvalsh(process).min() < -1e-12:
                raise ValueError("process covariance must be positive semidefinite")


def constant_velocity_dynamics(
    spatial_dimensions: int = 1, *, acceleration_spectral_density: float = 1.0
) -> LinearGaussianDynamics:
    """Continuous white-acceleration model using the actual elapsed seconds."""

    if (
        isinstance(spatial_dimensions, bool)
        or not isinstance(spatial_dimensions, int)
        or spatial_dimensions <= 0
    ):
        raise ValueError("spatial_dimensions must be a positive integer")
    spectral_density = float(acceleration_spectral_density)
    if not np.isfinite(spectral_density) or spectral_density < 0.0:
        raise ValueError("acceleration_spectral_density must be non-negative")
    dimension = 2 * spatial_dimensions

    def transition(delta: float) -> np.ndarray:
        delta = float(delta)
        if not np.isfinite(delta) or delta < 0.0:
            raise ValueError("delta must be finite and non-negative")
        matrix = np.eye(dimension, dtype=np.float64)
        matrix[:spatial_dimensions, spatial_dimensions:] = (
            np.eye(spatial_dimensions) * delta
        )
        return matrix

    def process(delta: float) -> np.ndarray:
        delta = float(delta)
        if not np.isfinite(delta) or delta < 0.0:
            raise ValueError("delta must be finite and non-negative")
        covariance = np.zeros((dimension, dimension), dtype=np.float64)
        identity = np.eye(spatial_dimensions) * spectral_density
        covariance[:spatial_dimensions, :spatial_dimensions] = identity * delta**3 / 3.0
        covariance[:spatial_dimensions, spatial_dimensions:] = identity * delta**2 / 2.0
        covariance[spatial_dimensions:, :spatial_dimensions] = identity * delta**2 / 2.0
        covariance[spatial_dimensions:, spatial_dimensions:] = identity * delta
        return covariance

    return LinearGaussianDynamics(dimension, transition, process)


class OutOfWindowPolicy(str, Enum):
    DROP = "drop"
    REJECT = "reject"


class IngestStatus(str, Enum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"
    FUTURE_ARRIVAL = "future_arrival"
    FUTURE_INFORMATION = "future_information"
    MISSED_DEADLINE = "missed_deadline"
    EXPIRED_TTL = "expired_ttl"
    WRONG_TARGET = "wrong_target"
    OUT_OF_WINDOW_DROPPED = "out_of_window_dropped"
    OUT_OF_WINDOW_REJECTED = "out_of_window_rejected"
    LINEAGE_OVERLAP = "lineage_overlap"


class CorrelatedBeliefRequiresCI(TypeError):
    """Raised when a correlated posterior is offered to the Kalman path."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    status: IngestStatus
    message_id: str

    @property
    def applied(self) -> bool:
        return self.status is IngestStatus.APPLIED


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    time: float
    mean: np.ndarray
    covariance: np.ndarray
    processed_message_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MeasurementFactor:
    event_time: float
    source: str
    generated_local: float
    sequence: int
    message_id: str
    measurement: np.ndarray
    measurement_matrix: np.ndarray
    measurement_covariance: np.ndarray
    factor_ids: tuple[str, ...]

    @property
    def sort_key(self) -> tuple[float, str, float, int, str]:
        return (
            self.event_time,
            self.source,
            self.generated_local,
            self.sequence,
            self.message_id,
        )


def _state(mean: np.ndarray, covariance: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if mean.shape != (dimension,) or not np.all(np.isfinite(mean)):
        raise ValueError("initial mean has incompatible shape")
    if (
        covariance.shape != (dimension, dimension)
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("initial covariance must be finite, symmetric, and compatible")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("initial covariance must be positive definite") from exc
    return mean.copy(), covariance.copy()


class FixedLagReplay:
    """One-track exact linear-Gaussian chronological replay cache.

    Only :class:`IndependentIncrement` payloads are accepted.  A correlated
    track belief must be associated and fused through Covariance Intersection.
    """

    def __init__(
        self,
        *,
        track_id: str,
        initial_time: float,
        initial_mean: np.ndarray,
        initial_covariance: np.ndarray,
        dynamics: LinearGaussianDynamics,
        lag: float,
        absorbed_factor_ids: tuple[str, ...] = (),
        out_of_window_policy: OutOfWindowPolicy = OutOfWindowPolicy.DROP,
    ) -> None:
        initial_time = float(initial_time)
        lag = float(lag)
        if not np.isfinite(initial_time):
            raise ValueError("initial_time must be finite")
        if not np.isfinite(lag) or lag <= 0.0:
            raise ValueError("lag must be finite and positive")
        if not isinstance(track_id, str) or not track_id.strip():
            raise ValueError("track_id must be a non-empty string")
        absorbed = tuple(sorted(absorbed_factor_ids))
        if any(not isinstance(item, str) or not item for item in absorbed):
            raise ValueError("absorbed_factor_ids must be non-empty strings")
        if len(absorbed) != len(set(absorbed)):
            raise ValueError("absorbed_factor_ids must be unique")
        mean, covariance = _state(
            initial_mean, initial_covariance, dynamics.state_dimension
        )
        self._dynamics = dynamics
        self._track_id = track_id
        self._lag = lag
        self._policy = OutOfWindowPolicy(out_of_window_policy)
        self._base_time = initial_time
        self._base_mean = mean
        self._base_covariance = covariance
        self._current_time = initial_time
        self._current_mean = mean.copy()
        self._current_covariance = covariance.copy()
        self._factors: list[_MeasurementFactor] = []
        self._processed_message_ids: set[str] = set()
        # The boundary prior is not evidence-free.  Its absorbed factors must
        # remain in the ledger after fixed-lag pruning to prevent re-ingestion.
        self._used_factor_ids: set[str] = set(absorbed)

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def cache_start_time(self) -> float:
        return self._base_time

    @property
    def factor_count(self) -> int:
        return len(self._factors)

    def _predict(
        self, mean: np.ndarray, covariance: np.ndarray, delta: float
    ) -> tuple[np.ndarray, np.ndarray]:
        transition = np.asarray(self._dynamics.transition(delta), dtype=np.float64)
        process = np.asarray(self._dynamics.process_covariance(delta), dtype=np.float64)
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + process
        return mean, 0.5 * (covariance + covariance.T)

    @staticmethod
    def _update(
        mean: np.ndarray,
        covariance: np.ndarray,
        factor: _MeasurementFactor,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = factor.measurement_matrix
        innovation = factor.measurement - matrix @ mean
        innovation_covariance = (
            matrix @ covariance @ matrix.T + factor.measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance, matrix @ covariance
        ).T
        mean = mean + gain @ innovation
        identity = np.eye(mean.size)
        residual = identity - gain @ matrix
        # Joseph form preserves symmetry/PSD under finite precision.
        covariance = (
            residual @ covariance @ residual.T
            + gain @ factor.measurement_covariance @ gain.T
        )
        return mean, 0.5 * (covariance + covariance.T)

    def _run(
        self, target_time: float, factors: list[_MeasurementFactor] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if target_time < self._base_time - 1e-12:
            raise ValueError("target_time precedes the filtered boundary prior")
        mean = self._base_mean.copy()
        covariance = self._base_covariance.copy()
        cursor = self._base_time
        selected = self._factors if factors is None else factors
        for factor in sorted(selected, key=lambda item: item.sort_key):
            if factor.event_time < self._base_time - 1e-12:
                continue
            if factor.event_time > target_time + 1e-12:
                continue
            delta = max(0.0, factor.event_time - cursor)
            mean, covariance = self._predict(mean, covariance, delta)
            cursor = factor.event_time
            mean, covariance = self._update(mean, covariance, factor)
        mean, covariance = self._predict(mean, covariance, max(0.0, target_time - cursor))
        return mean, covariance

    def _prune_to(self, boundary: float) -> None:
        if boundary <= self._base_time + 1e-12:
            return
        expired = [factor for factor in self._factors if factor.event_time < boundary]
        mean, covariance = self._run(boundary, expired)
        self._base_time = boundary
        self._base_mean = mean
        self._base_covariance = covariance
        self._factors = [
            factor for factor in self._factors if factor.event_time >= boundary
        ]

    def advance_to(self, decision_time: float) -> ReplaySnapshot:
        decision_time = float(decision_time)
        if not np.isfinite(decision_time):
            raise ValueError("decision_time must be finite")
        if decision_time < self._current_time - 1e-12:
            raise ValueError("decision_time cannot move backwards")
        self._prune_to(max(self._base_time, decision_time - self._lag))
        self._current_time = decision_time
        self._current_mean, self._current_covariance = self._run(decision_time)
        return self.snapshot()

    def ingest_increment(
        self,
        received: ReceivedMessage,
        *,
        clock_map: AffineClockMap,
        decision_time: float,
        confidence: float = 0.999,
    ) -> IngestResult:
        """Map the message's own timestamps, insert one factor, and replay.

        ``clock_map`` is receiver state.  The public API intentionally does not
        accept an arbitrary event-time estimate or upper-bound override.
        """

        message = received.message
        if isinstance(message.payload, CorrelatedTrackBelief):
            raise CorrelatedBeliefRequiresCI(
                "correlated track beliefs cannot enter the independent Kalman path"
            )
        if not isinstance(message.payload, IndependentIncrement):  # pragma: no cover
            raise TypeError("unsupported payload")
        if message.message_id in self._processed_message_ids:
            return IngestResult(IngestStatus.DUPLICATE, message.message_id)
        decision_time = float(decision_time)
        if decision_time < self._current_time - 1e-12:
            raise ValueError("decision_time cannot precede tracker current_time")
        self.advance_to(decision_time)
        if received.received_at > decision_time:
            return IngestResult(IngestStatus.FUTURE_ARRIVAL, message.message_id)
        if decision_time > message.deadline:
            self._processed_message_ids.add(message.message_id)
            return IngestResult(IngestStatus.MISSED_DEADLINE, message.message_id)
        mapped = clock_map.map_message(received)
        estimate = mapped.information_cutoff
        if not conservative_no_future_gate(
            estimate,
            decision_time,
            confidence=confidence,
        ):
            return IngestResult(IngestStatus.FUTURE_INFORMATION, message.message_id)
        # TTL is defined from the mapped sender transmit time in this reference
        # protocol.  The mean-time rule is a plug-in engineering policy, not a
        # certified interval claim.
        if decision_time > mapped.transmitted.mean + message.ttl:
            self._processed_message_ids.add(message.message_id)
            return IngestResult(IngestStatus.EXPIRED_TTL, message.message_id)
        if message.payload.target_id != self._track_id:
            self._processed_message_ids.add(message.message_id)
            return IngestResult(IngestStatus.WRONG_TARGET, message.message_id)
        if estimate.mean < self._base_time - 1e-12:
            self._processed_message_ids.add(message.message_id)
            status = (
                IngestStatus.OUT_OF_WINDOW_DROPPED
                if self._policy is OutOfWindowPolicy.DROP
                else IngestStatus.OUT_OF_WINDOW_REJECTED
            )
            return IngestResult(status, message.message_id)
        incoming_factors = set(message.payload.lineage.factor_ids)
        if not incoming_factors.isdisjoint(self._used_factor_ids):
            self._processed_message_ids.add(message.message_id)
            return IngestResult(IngestStatus.LINEAGE_OVERLAP, message.message_id)
        factor = _MeasurementFactor(
            event_time=estimate.mean,
            source=message.source,
            generated_local=message.timestamps.generated,
            sequence=message.sequence,
            message_id=message.message_id,
            measurement=message.payload.measurement,
            measurement_matrix=message.payload.measurement_matrix,
            measurement_covariance=message.payload.measurement_covariance,
            factor_ids=message.payload.lineage.factor_ids,
        )
        if factor.measurement_matrix.shape[1] != self._dynamics.state_dimension:
            raise ValueError("increment measurement matrix has the wrong state dimension")
        self._factors.append(factor)
        self._processed_message_ids.add(message.message_id)
        self._used_factor_ids.update(incoming_factors)
        self._current_mean, self._current_covariance = self._run(self._current_time)
        return IngestResult(IngestStatus.APPLIED, message.message_id)

    def snapshot(self) -> ReplaySnapshot:
        mean = self._current_mean.copy()
        covariance = self._current_covariance.copy()
        mean.setflags(write=False)
        covariance.setflags(write=False)
        return ReplaySnapshot(
            time=self._current_time,
            mean=mean,
            covariance=covariance,
            processed_message_ids=tuple(sorted(self._processed_message_ids)),
        )


__all__ = [
    "CorrelatedBeliefRequiresCI",
    "FixedLagReplay",
    "IngestResult",
    "IngestStatus",
    "LinearGaussianDynamics",
    "OutOfWindowPolicy",
    "ReplaySnapshot",
    "constant_velocity_dynamics",
]
