"""Gaussian gating and exact one-to-one association primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _spd(covariance: np.ndarray, name: str = "covariance") -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError(f"{name} must be a finite symmetric square matrix")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return covariance


def mahalanobis_squared(innovation: np.ndarray, covariance: np.ndarray) -> float:
    innovation = np.asarray(innovation, dtype=np.float64)
    covariance = _spd(covariance)
    if innovation.shape != (covariance.shape[0],) or not np.all(np.isfinite(innovation)):
        raise ValueError("innovation has incompatible shape")
    return float(innovation @ np.linalg.solve(covariance, innovation))


def gaussian_nll(
    innovation: np.ndarray,
    covariance: np.ndarray,
    *,
    regularization: float = 0.0,
    include_constant: bool = False,
) -> float:
    """Gaussian NLL with one covariance shared by quadratic and log-det terms."""

    covariance = _spd(covariance)
    regularization = float(regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    used = covariance + regularization * np.eye(covariance.shape[0])
    innovation = np.asarray(innovation, dtype=np.float64)
    if innovation.shape != (used.shape[0],) or not np.all(np.isfinite(innovation)):
        raise ValueError("innovation has incompatible shape")
    sign, logdet = np.linalg.slogdet(used)
    if sign <= 0:  # pragma: no cover - guaranteed by SPD validation.
        raise ValueError("regularized covariance must be positive definite")
    value = 0.5 * float(innovation @ np.linalg.solve(used, innovation)) + 0.5 * logdet
    if include_constant:
        value += 0.5 * innovation.size * math.log(2.0 * math.pi)
    return value


def _regularized_gamma_p(shape: float, value: float) -> float:
    """Regularized lower incomplete gamma for the chi-square CDF."""

    if value <= 0.0:
        return 0.0
    epsilon = 1e-14
    tiny = 1e-300
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        cursor = shape
        for _ in range(10000):
            cursor += 1.0
            term *= value / cursor
            total += term
            if abs(term) <= abs(total) * epsilon:
                break
        return total * math.exp(-value + shape * math.log(value) - math.lgamma(shape))

    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for index in range(1, 10000):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            break
    upper = math.exp(-value + shape * math.log(value) - math.lgamma(shape)) * fraction
    return 1.0 - upper


def chi_square_quantile(degrees_of_freedom: int, probability: float) -> float:
    if isinstance(degrees_of_freedom, bool) or not isinstance(degrees_of_freedom, int):
        raise TypeError("degrees_of_freedom must be an integer")
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    lower = 0.0
    upper = float(max(1, degrees_of_freedom))
    while _regularized_gamma_p(0.5 * degrees_of_freedom, 0.5 * upper) < probability:
        upper *= 2.0
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        cdf = _regularized_gamma_p(0.5 * degrees_of_freedom, 0.5 * middle)
        if cdf < probability:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


@dataclass(frozen=True, slots=True)
class GateResult:
    accepted: bool
    distance_squared: float
    threshold: float


def chi_square_gate(
    innovation: np.ndarray,
    covariance: np.ndarray,
    *,
    probability: float = 0.95,
) -> GateResult:
    """Gate against the unregularized model covariance."""

    innovation = np.asarray(innovation, dtype=np.float64)
    distance = mahalanobis_squared(innovation, covariance)
    threshold = chi_square_quantile(innovation.size, probability)
    return GateResult(distance <= threshold, distance, threshold)


@dataclass(frozen=True, slots=True)
class Assignment:
    pairs: tuple[tuple[int, int], ...]
    unmatched_left: tuple[int, ...]
    unmatched_right: tuple[int, ...]
    total_cost: float


def _cost_vector(value: float | np.ndarray, length: int, name: str) -> np.ndarray:
    if np.isscalar(value):
        result = np.full(length, float(value), dtype=np.float64)
    else:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (length,):
            raise ValueError(f"{name} must be scalar or have shape ({length},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite costs")
    return result


def solve_one_to_one(
    pair_costs: np.ndarray,
    *,
    unmatched_left_cost: float | np.ndarray,
    unmatched_right_cost: float | np.ndarray,
) -> Assignment:
    """Exact polynomial assignment with explicit unmatched/dustbin costs."""

    pair_costs = np.asarray(pair_costs, dtype=np.float64)
    if pair_costs.ndim != 2 or np.any(np.isnan(pair_costs)):
        raise ValueError("pair_costs must be a 2-D matrix without NaNs")
    left_count, right_count = pair_costs.shape
    left_unmatched = _cost_vector(unmatched_left_cost, left_count, "unmatched_left_cost")
    right_unmatched = _cost_vector(
        unmatched_right_cost, right_count, "unmatched_right_cost"
    )

    size = left_count + right_count
    if size == 0:
        return Assignment((), (), (), 0.0)
    finite = np.concatenate(
        (
            pair_costs[np.isfinite(pair_costs)],
            left_unmatched,
            right_unmatched,
            np.array([0.0]),
        )
    )
    max_abs = max(1.0, float(np.max(np.abs(finite))))
    forbidden = (2 * size + 1) * max_abs + 1.0
    costs = np.full((size, size), forbidden, dtype=np.float64)
    costs[:left_count, :right_count] = np.where(
        np.isfinite(pair_costs), pair_costs, forbidden
    )
    for left in range(left_count):
        costs[left, right_count + left] = left_unmatched[left]
    for right in range(right_count):
        costs[left_count + right, right] = right_unmatched[right]
    costs[left_count:, right_count:] = 0.0

    # Hungarian shortest augmenting-path algorithm, specialized to a finite
    # square matrix.  This avoids the previous 2**N candidate-state explosion.
    potentials_rows = np.zeros(size + 1, dtype=np.float64)
    potentials_columns = np.zeros(size + 1, dtype=np.float64)
    column_match = np.zeros(size + 1, dtype=np.int64)
    predecessor = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        column_match[0] = row
        minimum = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            matched_row = int(column_match[column])
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    costs[matched_row - 1, candidate_column - 1]
                    - potentials_rows[matched_row]
                    - potentials_columns[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            if not np.isfinite(delta):  # pragma: no cover - dustbins are feasible.
                raise ValueError("assignment has no feasible completion")
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    potentials_rows[column_match[candidate_column]] += delta
                    potentials_columns[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if column_match[column] == 0:
                break
        while True:
            previous = int(predecessor[column])
            column_match[column] = column_match[previous]
            column = previous
            if column == 0:
                break

    row_to_column = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        row_to_column[column_match[column] - 1] = column - 1
    if any(costs[row, column] >= forbidden for row, column in enumerate(row_to_column)):
        raise ValueError("assignment selected a forbidden edge")
    pairs = tuple(
        (left, int(row_to_column[left]))
        for left in range(left_count)
        if row_to_column[left] < right_count
    )
    matched_left = {left for left, _ in pairs}
    matched_right = {right for _, right in pairs}
    unmatched_left = tuple(left for left in range(left_count) if left not in matched_left)
    unmatched_right = tuple(
        right for right in range(right_count) if right not in matched_right
    )
    total = (
        sum(float(pair_costs[left, right]) for left, right in pairs)
        + sum(float(left_unmatched[left]) for left in unmatched_left)
        + sum(float(right_unmatched[right]) for right in unmatched_right)
    )
    return Assignment(pairs, unmatched_left, unmatched_right, total)


@dataclass(frozen=True, slots=True)
class GaussianAssociation:
    assignment: Assignment
    costs: np.ndarray
    distances_squared: np.ndarray
    gate_threshold: float


def associate_gaussians(
    left_means: list[np.ndarray] | tuple[np.ndarray, ...],
    left_covariances: list[np.ndarray] | tuple[np.ndarray, ...],
    right_means: list[np.ndarray] | tuple[np.ndarray, ...],
    right_covariances: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    gate_probability: float = 0.95,
    unmatched_left_cost: float | np.ndarray,
    unmatched_right_cost: float | np.ndarray,
    regularization: float = 0.0,
    innovation_covariances: np.ndarray | None = None,
    assume_independent: bool = False,
) -> GaussianAssociation:
    """Associate Gaussian means using an explicit innovation model.

    For correlated track beliefs callers must provide the full joint
    ``innovation_covariances[left, right]`` including cross, pose, and time
    terms.  ``assume_independent=True`` is an explicit baseline-only shortcut
    that forms ``P_left + P_right``.
    """
    if innovation_covariances is None and not assume_independent:
        raise ValueError(
            "innovation_covariances are required unless the caller explicitly "
            "selects the independent-Gaussian baseline"
        )
    if innovation_covariances is not None and assume_independent:
        raise ValueError("choose either explicit innovation covariance or independence")
    if len(left_means) != len(left_covariances) or len(right_means) != len(
        right_covariances
    ):
        raise ValueError("mean and covariance counts must agree")
    left_count = len(left_means)
    right_count = len(right_means)
    costs = np.full((left_count, right_count), np.inf, dtype=np.float64)
    distances = np.full_like(costs, np.inf)
    dimension: int | None = None
    for left, (left_mean, left_covariance) in enumerate(
        zip(left_means, left_covariances)
    ):
        left_mean = np.asarray(left_mean, dtype=np.float64)
        left_covariance = _spd(left_covariance, "left covariance")
        if left_mean.shape != (left_covariance.shape[0],):
            raise ValueError("left mean and covariance shape mismatch")
        dimension = left_mean.size if dimension is None else dimension
        if left_mean.size != dimension:
            raise ValueError("all Gaussian states must share a dimension")
        for right, (right_mean, right_covariance) in enumerate(
            zip(right_means, right_covariances)
        ):
            right_mean = np.asarray(right_mean, dtype=np.float64)
            right_covariance = _spd(right_covariance, "right covariance")
            if right_mean.shape != left_mean.shape or right_covariance.shape != left_covariance.shape:
                raise ValueError("all Gaussian states must share a dimension")
            innovation = left_mean - right_mean
            if innovation_covariances is None:
                covariance = left_covariance + right_covariance
            else:
                all_covariances = np.asarray(
                    innovation_covariances, dtype=np.float64
                )
                expected_shape = (
                    left_count,
                    right_count,
                    left_mean.size,
                    left_mean.size,
                )
                if all_covariances.shape != expected_shape:
                    raise ValueError(
                        "innovation_covariances has incompatible shape; expected "
                        f"{expected_shape}"
                    )
                covariance = all_covariances[left, right]
            gate = chi_square_gate(
                innovation, covariance, probability=gate_probability
            )
            distances[left, right] = gate.distance_squared
            if gate.accepted:
                costs[left, right] = gaussian_nll(
                    innovation, covariance, regularization=regularization
                )
    threshold = (
        chi_square_quantile(dimension, gate_probability) if dimension is not None else 0.0
    )
    assignment = solve_one_to_one(
        costs,
        unmatched_left_cost=unmatched_left_cost,
        unmatched_right_cost=unmatched_right_cost,
    )
    costs.setflags(write=False)
    distances.setflags(write=False)
    return GaussianAssociation(assignment, costs, distances, threshold)


__all__ = [
    "Assignment",
    "GateResult",
    "GaussianAssociation",
    "associate_gaussians",
    "chi_square_gate",
    "chi_square_quantile",
    "gaussian_nll",
    "mahalanobis_squared",
    "solve_one_to_one",
]
