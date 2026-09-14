"""Reliability-aware association costs and explicit unmatched assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


MAX_ASSOCIATION_DIMENSION = 256
MAX_COST_VALUE = 1e100


class AssociationError(ValueError):
    """Raised when an association problem is malformed or infeasible."""


def _as_float(value: int | float, *, label: str) -> float:
    try:
        return float(value)
    except (OverflowError, ValueError) as exc:
        raise AssociationError(f"{label} exceeds the numeric bound") from exc


@dataclass(frozen=True, slots=True)
class AssociationCostConfig:
    position_weight: float
    bev_iou_weight: float
    embedding_weight: float
    age_weight: float
    reliability_weight: float
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "position_weight",
            "bev_iou_weight",
            "embedding_weight",
            "age_weight",
            "reliability_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise AssociationError(f"{name} must be numeric")
            number = _as_float(value, label=name)
            if not math.isfinite(number) or number < 0.0:
                raise AssociationError(f"{name} must be finite and non-negative")
            if number > MAX_COST_VALUE:
                raise AssociationError(f"{name} exceeds the supported numeric bound")
        if not isinstance(self.epsilon, (int, float)) or isinstance(self.epsilon, bool):
            raise AssociationError("epsilon must be finite and in (0, 1]")
        epsilon = _as_float(self.epsilon, label="epsilon")
        if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
            raise AssociationError("epsilon must be finite and in (0, 1]")


@dataclass(frozen=True, slots=True)
class AssociationResult:
    matches: tuple[tuple[int, int], ...]
    unmatched_queries: tuple[int, ...]
    new_candidates: tuple[int, ...]
    total_cost: float


def _rectangular_matrix(
    values: Sequence[Sequence[object]], *, label: str
) -> tuple[list[list[float]], int, int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise AssociationError(f"{label} must be a rectangular matrix")
    rows = list(values)
    if not rows:
        return [], 0, 0
    if any(
        not isinstance(row, Sequence) or isinstance(row, (str, bytes)) for row in rows
    ):
        raise AssociationError(f"{label} must be a rectangular matrix")
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise AssociationError(f"{label} must be rectangular")
    converted: list[list[float]] = []
    for row_index, row in enumerate(rows):
        converted_row: list[float] = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AssociationError(
                    f"{label}[{row_index}][{column_index}] must be numeric"
                )
            result = _as_float(value, label=f"{label}[{row_index}][{column_index}]")
            if math.isnan(result) or result == -math.inf:
                raise AssociationError(
                    f"{label}[{row_index}][{column_index}] is invalid"
                )
            if math.isfinite(result) and abs(result) > MAX_COST_VALUE:
                raise AssociationError(
                    f"{label}[{row_index}][{column_index}] exceeds the numeric bound"
                )
            converted_row.append(result)
        converted.append(converted_row)
    return converted, len(converted), column_count


def _same_shape(
    values: Sequence[Sequence[object]],
    *,
    label: str,
    rows: int,
    columns: int,
) -> list[list[float]]:
    matrix, observed_rows, observed_columns = _rectangular_matrix(values, label=label)
    if (observed_rows, observed_columns) != (rows, columns):
        raise AssociationError(f"{label} shape does not match the association problem")
    return matrix


def compose_association_cost(
    *,
    mahalanobis_distance: Sequence[Sequence[object]],
    bev_iou: Sequence[Sequence[object]],
    embedding_cosine: Sequence[Sequence[object]],
    normalized_age: Sequence[Sequence[object]],
    reliability: Sequence[Sequence[object]],
    gate: Sequence[Sequence[bool]],
    config: AssociationCostConfig,
) -> list[list[float]]:
    """Compose Equation (6.9)-style costs from independently auditable terms."""

    if not isinstance(config, AssociationCostConfig):
        raise AssociationError("config must be an AssociationCostConfig")
    position, rows, columns = _rectangular_matrix(
        mahalanobis_distance, label="mahalanobis_distance"
    )
    iou = _same_shape(bev_iou, label="bev_iou", rows=rows, columns=columns)
    cosine = _same_shape(
        embedding_cosine, label="embedding_cosine", rows=rows, columns=columns
    )
    age = _same_shape(
        normalized_age, label="normalized_age", rows=rows, columns=columns
    )
    confidence = _same_shape(
        reliability, label="reliability", rows=rows, columns=columns
    )
    if len(gate) != rows or any(len(row) != columns for row in gate):
        raise AssociationError("gate shape does not match the association problem")

    result: list[list[float]] = []
    for row_index in range(rows):
        output_row: list[float] = []
        for column_index in range(columns):
            enabled = gate[row_index][column_index]
            if not isinstance(enabled, bool):
                raise AssociationError("gate values must be boolean")
            if not enabled:
                output_row.append(math.inf)
                continue
            p = position[row_index][column_index]
            overlap = iou[row_index][column_index]
            similarity = cosine[row_index][column_index]
            sample_age = age[row_index][column_index]
            sample_reliability = confidence[row_index][column_index]
            if any(
                not math.isfinite(value)
                for value in (p, overlap, similarity, sample_age, sample_reliability)
            ):
                raise AssociationError("enabled association terms must be finite")
            if p < 0.0:
                raise AssociationError("Mahalanobis distance must be non-negative")
            if not 0.0 <= overlap <= 1.0:
                raise AssociationError("BEV IoU must be in [0, 1]")
            if not -1.0 <= similarity <= 1.0:
                raise AssociationError("embedding cosine must be in [-1, 1]")
            if not 0.0 <= sample_age <= 1.0:
                raise AssociationError("normalized age must be in [0, 1]")
            if not 0.0 <= sample_reliability <= 1.0:
                raise AssociationError("reliability must be in [0, 1]")
            value = (
                config.position_weight * p
                + config.bev_iou_weight * (1.0 - overlap)
                + config.embedding_weight * (1.0 - similarity)
                + config.age_weight * sample_age
                + config.reliability_weight
                * -math.log(max(sample_reliability, config.epsilon))
            )
            if not math.isfinite(value) or value > MAX_COST_VALUE:
                raise AssociationError(
                    "composed association cost exceeds numeric bound"
                )
            output_row.append(float(value))
        result.append(output_row)
    return result


def _finite_nonnegative_vector(
    values: Sequence[object], *, label: str, expected: int
) -> list[float]:
    if len(values) != expected:
        raise AssociationError(f"{label} length does not match the problem")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssociationError(f"{label}[{index}] must be numeric")
        number = _as_float(value, label=f"{label}[{index}]")
        if not math.isfinite(number) or number < 0.0:
            raise AssociationError(f"{label}[{index}] must be finite and non-negative")
        if number > MAX_COST_VALUE:
            raise AssociationError(f"{label}[{index}] exceeds the numeric bound")
        result.append(number)
    return result


def _hungarian_square(cost: list[list[float]]) -> list[int]:
    """Return the minimum-cost column for each row of a finite square matrix."""

    size = len(cost)
    if size == 0:
        return []
    if any(len(row) != size for row in cost):
        raise AssociationError("internal Hungarian matrix must be square")
    potentials_rows = [0.0] * (size + 1)
    potentials_columns = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        matched_row[0] = row
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    cost[current_row - 1][candidate_column - 1]
                    - potentials_rows[current_row]
                    - potentials_columns[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            if not math.isfinite(delta):
                raise AssociationError("association problem has no finite assignment")
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    potentials_rows[matched_row[candidate_column]] += delta
                    potentials_columns[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        if matched_row[column] != 0:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise AssociationError("Hungarian solver returned an incomplete assignment")
    return assignment


def solve_with_unmatched(
    costs: Sequence[Sequence[object]],
    *,
    query_unmatched_costs: Sequence[object],
    candidate_new_costs: Sequence[object],
) -> AssociationResult:
    """Solve one assignment with explicit unmatched-query and new-candidate terms."""

    matrix, query_count, candidate_count = _rectangular_matrix(costs, label="costs")
    if query_count == 0:
        try:
            candidate_count = len(candidate_new_costs)
        except TypeError as exc:
            raise AssociationError("candidate_new_costs must be a sequence") from exc
    if query_count + candidate_count > MAX_ASSOCIATION_DIMENSION:
        raise AssociationError("association dimension exceeds the supported bound")
    query_costs = _finite_nonnegative_vector(
        query_unmatched_costs,
        label="query_unmatched_costs",
        expected=query_count,
    )
    candidate_costs = _finite_nonnegative_vector(
        candidate_new_costs,
        label="candidate_new_costs",
        expected=candidate_count,
    )
    if query_count == 0:
        return AssociationResult(
            matches=(),
            unmatched_queries=(),
            new_candidates=tuple(range(candidate_count)),
            total_cost=float(sum(candidate_costs)),
        )
    if candidate_count == 0:
        return AssociationResult(
            matches=(),
            unmatched_queries=tuple(range(query_count)),
            new_candidates=(),
            total_cost=float(sum(query_costs)),
        )
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            if value != math.inf and (not math.isfinite(value) or value < 0.0):
                raise AssociationError(
                    f"costs[{row_index}][{column_index}] must be non-negative or +inf"
                )

    size = query_count + candidate_count
    finite_values = (
        [value for row in matrix for value in row if math.isfinite(value)]
        + query_costs
        + candidate_costs
    )
    scale = max([1.0, *finite_values])
    forbidden = (size + 1) * scale + sum(query_costs) + sum(candidate_costs) + 1.0
    if not math.isfinite(forbidden):
        raise AssociationError("association sentinel overflowed")
    square = [[forbidden for _ in range(size)] for _ in range(size)]
    for query_index in range(query_count):
        for candidate_index in range(candidate_count):
            value = matrix[query_index][candidate_index]
            square[query_index][candidate_index] = (
                forbidden if value == math.inf else value
            )
        square[query_index][candidate_count + query_index] = query_costs[query_index]
    for candidate_index in range(candidate_count):
        row = query_count + candidate_index
        square[row][candidate_index] = candidate_costs[candidate_index]
        for dummy_column in range(query_count):
            square[row][candidate_count + dummy_column] = 0.0

    assignment = _hungarian_square(square)
    matches: list[tuple[int, int]] = []
    unmatched_queries: list[int] = []
    matched_candidates: set[int] = set()
    total = 0.0
    for query_index in range(query_count):
        column = assignment[query_index]
        if column < candidate_count and math.isfinite(matrix[query_index][column]):
            matches.append((query_index, column))
            matched_candidates.add(column)
            total += matrix[query_index][column]
        else:
            unmatched_queries.append(query_index)
            total += query_costs[query_index]
    new_candidates = [
        candidate_index
        for candidate_index in range(candidate_count)
        if candidate_index not in matched_candidates
    ]
    total += sum(candidate_costs[index] for index in new_candidates)
    if not math.isfinite(total):
        raise AssociationError("association total cost overflowed")
    return AssociationResult(
        matches=tuple(matches),
        unmatched_queries=tuple(unmatched_queries),
        new_candidates=tuple(new_candidates),
        total_cost=float(total),
    )
