"""Budgeted Murty-style K-best partial matching, not a complete MHT tracker.

Each left row owns one private unmatched column. There are no dummy right
rows, so a physical partial matching has exactly one rectangular assignment.
Disjoint first-difference regions cover every unreturned matching, even when
the solve/frontier budget stops enumeration. Weights are absolute model
weights, not renormalized posterior probabilities. The algebraic bounds hold
in exact arithmetic; float64 padding is not an interval certificate.

Kept outside the model package while ongoing experiments bind that package.
No existing replay imports this experimental classical-baseline component.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from transvision.models.event_track_v2x.hypothesis_bank import (
    AssignmentHypothesis, LogAssociationFactors, _sum, assignment_log_weight, logsumexp,
)


@dataclass(frozen=True)
class RankedAssignmentLimits:
    max_solves: int = 10000
    max_frontier: int = 10000
    max_matrix_cells: int = 1000000
    max_returned: int = 4096

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == 'max_solves' else 1):
                raise ValueError('integer ranked-assignment resource limits required')


@dataclass(frozen=True)
class AssignmentRegion:
    prefix: tuple[int, ...]
    excluded: frozenset[tuple[int, int]]
    log_mass_upper: float

    def contains(self, choices):
        # Membership only; the caller separately checks legal factor support.
        return (tuple(choices[:len(self.prefix)]) == self.prefix
                and all((i, j) not in self.excluded for i, j in enumerate(choices)))


@dataclass(frozen=True)
class RankedAssignments:
    hypotheses: tuple[AssignmentHypothesis, ...]
    frontier: tuple[AssignmentRegion, ...]
    factor_sha256: str
    requested_k: int
    termination: str
    assignment_solves: int
    infeasible_solves: int
    peak_frontier: int
    peak_matrix_cells: int
    retained_log_mass: float
    residual_log_mass_upper: float
    omitted_mass_upper: float

    @property
    def requested_k_reached(self):
        return len(self.hypotheses) == self.requested_k

    @property
    def support_exhausted(self):
        return not self.frontier

    @property
    def interval_arithmetic_certified(self):
        return False


def _options(factors, prefix, excluded, row):
    used = {j for j in prefix if j >= 0}
    if (row, -1) not in excluded:
        yield -1, factors.log_left_unmatched[row]
    for j in range(factors.shape[1]):
        if j not in used and factors.allowed[row][j] and (row, j) not in excluded:
            yield j, _sum((factors.log_pair[row][j], -factors.log_right_unmatched[j]))


def _region(factors, prefix=(), excluded=frozenset()):
    # Regions are constructed internally from a legal optimizer solution.
    terms = list(factors.log_right_unmatched)
    for i, j in enumerate(prefix):
        if (i, j) in excluded:
            return None
        terms.append(factors.log_left_unmatched[i] if j < 0 else
                     _sum((factors.log_pair[i][j], -factors.log_right_unmatched[j]), upper=True))
    for i in range(len(prefix), factors.shape[0]):
        options = tuple(v for _, v in _options(factors, prefix, excluded, i))
        if not options:
            return None
        terms.append(logsumexp(options, upper=True))
    return AssignmentRegion(prefix, excluded, _sum(terms, upper=True))


def _optimum(factors, region):
    n, m = factors.shape
    first = len(region.prefix)
    if first == n:
        return AssignmentHypothesis(region.prefix, assignment_log_weight(factors, region.prefix)), 0
    # Real columns plus one PRIVATE unmatched column for each remaining row.
    costs = np.full((n-first, m+n-first), math.inf)
    for local, row in enumerate(range(first, n)):
        options = tuple(_options(factors, region.prefix, region.excluded, row))
        if not options:
            return None, costs.size
        shift = max(value for _, value in options)
        for j, value in options:
            cost = _sum((shift, -value))
            costs[local, m+local if j < 0 else j] = cost
    try:
        rows, columns = linear_sum_assignment(costs)
    except ValueError as error:
        if 'infeasible' not in str(error).lower():
            raise
        return None, costs.size
    if tuple(rows) != tuple(range(n-first)) or not np.all(np.isfinite(costs[rows, columns])):
        raise ValueError('assignment solver did not return a complete finite solution')
    choices = region.prefix + tuple(int(j) if j < m else -1 for j in columns)
    if not region.contains(choices):
        raise ValueError('assignment solver violated region constraints')
    return AssignmentHypothesis(choices, assignment_log_weight(factors, choices)), costs.size


def _children(factors, region, optimum):
    # Every other assignment has a UNIQUE first row differing from optimum.
    for i in range(len(region.prefix), factors.shape[0]):
        child = _region(factors, optimum.choices[:i], region.excluded | {(i, optimum.choices[i])})
        if child is not None:
            yield child


def _fraction_upper(retained, residual):
    if residual == -math.inf:
        return 0.
    if retained == -math.inf:
        return 1.
    delta = retained-residual
    exp = math.exp(-abs(delta))
    value = exp/(1.+exp) if delta >= 0 else 1./(1.+exp)
    # Do not round a positive unresolved mass all the way down to zero.
    return math.nextafter(value, 1.)


def k_best_partial_assignments(factors, k, *, limits=RankedAssignmentLimits()):
    """Return a ranked prefix plus a disjoint cover of all remaining support.

    Same-weight order uses queue insertion and the fixed SciPy solver; no
    global lexicographic tie guarantee is made. A budget stop is explicit and
    never labels a short prefix as the requested Top-K. All-K completion is
    distinct from having enumerated the full support. No exhaustive oracle.
    """
    if type(factors) is not LogAssociationFactors or type(limits) is not RankedAssignmentLimits:
        raise TypeError('immutable positive factors and ranked-assignment limits required')
    if type(k) is not int or not 1 <= k <= limits.max_returned:
        raise ValueError('requested K must fit the positive returned-hypothesis limit')
    n, m = factors.shape
    if n*(m+n)+n+m > limits.max_matrix_cells:
        raise ValueError('ranked-assignment matrix capacity exceeded before allocation')
    queue, serial, found = [], itertools.count(), []
    solves = infeasible = peak_matrix = 0

    def push(region, optimum=None):
        priority = region.log_mass_upper if optimum is None else optimum.log_weight
        heapq.heappush(queue, (-priority, next(serial), region, optimum))

    push(_region(factors))  # Positive unmatched factors make root nonempty.
    peak = 1
    termination = 'support-exhausted'
    while queue and len(found) < k:
        item = heapq.heappop(queue)
        _, _, region, optimum = item
        if optimum is None:
            needs_solve = len(region.prefix) < n
            if needs_solve and solves == limits.max_solves:
                heapq.heappush(queue, item)
                termination = 'solve-budget'
                break
            optimum, matrix_cells = _optimum(factors, region)
            solves += int(needs_solve)
            peak_matrix = max(peak_matrix, matrix_cells)
            if optimum is None:
                infeasible += 1
            else:
                push(region, optimum)
            continue
        children = []
        for child in _children(factors, region, optimum):
            if len(queue)+len(children)+1 > limits.max_frontier:
                break
            children.append(child)
        else:
            found.append(optimum)
            for child in children:
                push(child)
            peak = max(peak, len(queue))
            continue
        # Do not emit optimum before a complete residual partition can fit.
        heapq.heappush(queue, item)
        termination = 'frontier-budget'
        break
    if len(found) == k:
        termination = 'requested-k' if queue else 'support-exhausted'
    frontier = tuple(item[2] for item in sorted(queue))
    retained = logsumexp(h.log_weight for h in found)
    residual = logsumexp((r.log_mass_upper for r in frontier), upper=True)
    return RankedAssignments(tuple(found), frontier, factors.digest(), k, termination,
                             solves, infeasible, peak, peak_matrix, retained, residual,
                             _fraction_upper(retained, residual))
