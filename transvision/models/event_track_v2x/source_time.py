"""Source-time mixture propagation for asynchronous constant-velocity states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist

import numpy as np

from .arrays import immutable_float64


class SourceTimeMode(str, Enum):
    ORACLE = "oracle"
    POINT = "point"
    FIRST_ORDER = "first_order"
    MIXTURE = "mixture"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _state_inputs(
    mean: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if mean.shape != (9,) or not np.all(np.isfinite(mean)):
        raise ValueError("state mean must be a finite nine-vector")
    if covariance.shape != (9, 9) or not np.all(np.isfinite(covariance)):
        raise ValueError("state covariance must be a finite 9x9 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("state covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("state covariance must be positive definite") from exc
    return mean, covariance


@dataclass(frozen=True, slots=True)
class GaussianTimeComponent:
    weight: float
    mean: float
    variance: float

    def __post_init__(self) -> None:
        weight = _finite(self.weight, "weight")
        mean = _finite(self.mean, "time mean")
        variance = _finite(self.variance, "time variance")
        if weight <= 0.0:
            raise ValueError("component weight must be positive")
        if variance < 0.0:
            raise ValueError("time variance must be non-negative")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)


@dataclass(frozen=True, slots=True)
class SourceTimeMixture:
    """Finite Gaussian moment approximation plus an external causal bound.

    Gaussian components have unbounded mathematical support.  Consequently,
    ``causal_upper_bound`` is *not* a support bound inferred from these
    components.  It must come from a separately authenticated protocol fact
    about the realized information cutoff (or from a probabilistic gate whose
    coverage is reported explicitly).  The component-mean check below is only
    a consistency guard for the approximation.
    """

    components: tuple[GaussianTimeComponent, ...]
    causal_upper_bound: float

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple) or not self.components:
            raise ValueError("source-time mixture must contain components")
        if len(self.components) > 9:
            raise ValueError("source-time mixture supports at most nine components")
        if not all(isinstance(item, GaussianTimeComponent) for item in self.components):
            raise TypeError("components must be GaussianTimeComponent values")
        total = sum(item.weight for item in self.components)
        if abs(total - 1.0) > 1e-12:
            raise ValueError("source-time component weights must sum to one")
        causal_upper_bound = _finite(
            self.causal_upper_bound, "causal_upper_bound"
        )
        if any(item.mean > causal_upper_bound for item in self.components):
            raise ValueError(
                "causal_upper_bound must not precede a component mean"
            )
        object.__setattr__(self, "causal_upper_bound", causal_upper_bound)

    @property
    def mean(self) -> float:
        return sum(item.weight * item.mean for item in self.components)

    @property
    def variance(self) -> float:
        centre = self.mean
        return sum(
            item.weight * (item.variance + (item.mean - centre) ** 2)
            for item in self.components
        )

    def log_probability_density(self, value: float) -> float:
        """Evaluate a continuous log density.

        A mixture containing a zero-variance atom has no density with respect
        to Lebesgue measure.  Such Oracle/Point variants must be compared with
        :meth:`log_interval_probability` under a preregistered time bin.
        """

        value = _finite(value, "time value")
        if any(component.variance == 0.0 for component in self.components):
            raise ValueError(
                "log density is undefined for zero-variance time atoms; "
                "use log_interval_probability"
            )
        terms: list[float] = []
        for component in self.components:
            residual = value - component.mean
            terms.append(
                np.log(component.weight)
                - 0.5
                * (
                    np.log(2.0 * np.pi * component.variance)
                    + residual * residual / component.variance
                )
            )
        maximum = max(terms)
        if not np.isfinite(maximum):
            return -np.inf
        return float(maximum + np.log(sum(np.exp(term - maximum) for term in terms)))

    def log_interval_probability(
        self,
        value: float,
        *,
        half_width: float,
    ) -> float:
        """Score probability in a fixed interval around ``value``.

        Unlike a differential density, interval probability handles Point and
        Oracle atoms and is invariant to a consistent seconds-to-milliseconds
        unit conversion.  The paper protocol must freeze ``half_width`` (for
        example 0.5 ms for a 1 ms scoring bin) before confirmatory evaluation.
        """

        value = _finite(value, "time value")
        half_width = _finite(half_width, "half_width")
        if half_width <= 0.0:
            raise ValueError("half_width must be positive")
        lower = value - half_width
        upper = value + half_width
        normal = NormalDist()
        probability = 0.0
        for component in self.components:
            if component.variance == 0.0:
                if lower <= component.mean <= upper:
                    probability += component.weight
                continue
            standard_deviation = float(np.sqrt(component.variance))
            probability += component.weight * (
                normal.cdf((upper - component.mean) / standard_deviation)
                - normal.cdf((lower - component.mean) / standard_deviation)
            )
        if probability <= 0.0:
            return -np.inf
        return float(np.log(min(probability, 1.0)))


@dataclass(frozen=True, slots=True)
class PropagatedStateComponent:
    weight: float
    source_time_mean: float
    source_time_variance: float
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean, covariance = _state_inputs(self.mean, self.covariance)
        mean = immutable_float64(mean)
        covariance = immutable_float64(covariance)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True, slots=True)
class PropagatedSourceTimeState:
    mode: SourceTimeMode
    components: tuple[PropagatedStateComponent, ...]
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean, covariance = _state_inputs(self.mean, self.covariance)
        mean = immutable_float64(mean)
        covariance = immutable_float64(covariance)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


def _transition(delta: float) -> np.ndarray:
    transition = np.eye(9, dtype=np.float64)
    transition[0, 7] = delta
    transition[1, 8] = delta
    return transition


def _propagate_component(
    state_mean: np.ndarray,
    state_covariance: np.ndarray,
    *,
    source_time_mean: float,
    source_time_variance: float,
    decision_time: float,
    process_noise_per_second: float,
    weight: float,
) -> PropagatedStateComponent:
    delta = decision_time - source_time_mean
    if delta < -1e-12:
        raise ValueError("source time cannot be later than the decision time")
    delta = max(delta, 0.0)
    transition = _transition(delta)
    propagated_mean = transition @ state_mean
    # macOS Accelerate may emit spurious matmul floating warnings for finite
    # inputs; the explicit finite postcondition below remains authoritative.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        propagated_covariance = transition @ state_covariance @ transition.T
        propagated_covariance += (
            np.eye(9, dtype=np.float64) * process_noise_per_second * delta
        )
    # For T independent of the state, Y = X + (decision-T)V has the exact
    # second moment Cov(Y) = F(E[decision-T]) P F^T
    # + Var(T) E[V V^T].  The former term is already above; add both the
    # velocity covariance and mean outer product to the x/y position block.
    velocity_mean = state_mean[7:9]
    velocity_second_moment = (
        state_covariance[np.ix_((7, 8), (7, 8))]
        + np.outer(velocity_mean, velocity_mean)
    )
    propagated_covariance[np.ix_((0, 1), (0, 1))] += (
        velocity_second_moment * source_time_variance
    )
    if not np.all(np.isfinite(propagated_covariance)):
        raise FloatingPointError("source-time propagation produced non-finite covariance")
    propagated_covariance = 0.5 * (
        propagated_covariance + propagated_covariance.T
    )
    return PropagatedStateComponent(
        weight=weight,
        source_time_mean=source_time_mean,
        source_time_variance=source_time_variance,
        mean=propagated_mean,
        covariance=propagated_covariance,
    )


def propagate_source_time(
    state_mean: np.ndarray,
    state_covariance: np.ndarray,
    source_time: SourceTimeMixture,
    *,
    decision_time: float,
    mode: SourceTimeMode | str,
    oracle_time: float | None = None,
    process_noise_per_second: float = 0.0,
) -> PropagatedSourceTimeState:
    """Propagate one state under the preregistered timestamp variants."""

    state_mean, state_covariance = _state_inputs(state_mean, state_covariance)
    decision_time = _finite(decision_time, "decision_time")
    process_noise = _finite(process_noise_per_second, "process_noise_per_second")
    if process_noise < 0.0:
        raise ValueError("process_noise_per_second must be non-negative")
    selected_mode = SourceTimeMode(mode)
    if source_time.causal_upper_bound > decision_time:
        raise ValueError(
            "external source-time admissibility bound is later than the decision time"
        )
    if selected_mode is SourceTimeMode.ORACLE:
        if oracle_time is None:
            raise ValueError("oracle mode requires oracle_time")
        oracle = _finite(oracle_time, "oracle_time")
        if oracle > decision_time:
            raise ValueError("oracle_time cannot be later than the decision time")
        descriptions = ((1.0, oracle, 0.0),)
    elif oracle_time is not None:
        raise ValueError("oracle_time is only valid in oracle mode")
    elif selected_mode is SourceTimeMode.POINT:
        descriptions = ((1.0, source_time.mean, 0.0),)
    elif selected_mode is SourceTimeMode.FIRST_ORDER:
        descriptions = ((1.0, source_time.mean, source_time.variance),)
    else:
        descriptions = tuple(
            (item.weight, item.mean, item.variance) for item in source_time.components
        )

    components = tuple(
        _propagate_component(
            state_mean,
            state_covariance,
            source_time_mean=time_mean,
            source_time_variance=time_variance,
            decision_time=decision_time,
            process_noise_per_second=process_noise,
            weight=weight,
        )
        for weight, time_mean, time_variance in descriptions
    )
    mixture_mean = sum(item.weight * item.mean for item in components)
    mixture_covariance = np.zeros((9, 9), dtype=np.float64)
    for item in components:
        residual = item.mean - mixture_mean
        mixture_covariance += item.weight * (
            item.covariance + np.outer(residual, residual)
        )
    mixture_covariance = 0.5 * (mixture_covariance + mixture_covariance.T)
    return PropagatedSourceTimeState(
        mode=selected_mode,
        components=components,
        mean=mixture_mean,
        covariance=mixture_covariance,
    )


__all__ = [
    "GaussianTimeComponent",
    "PropagatedSourceTimeState",
    "PropagatedStateComponent",
    "SourceTimeMixture",
    "SourceTimeMode",
    "propagate_source_time",
]
