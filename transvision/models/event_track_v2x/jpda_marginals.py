"""Single-scan JPDA marginals from the SAME positive partial-match factors.

Exact mode uses component-local frontier variable elimination (not Top-K).
LBP is the Williams--Lau matching-graph message update in log coordinates;
convergence is to a BP fixed point, NOT a true-posterior error certificate.
Neither mode preserves a cross-scan identity hypothesis bank.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .hypothesis_bank import LogAssociationFactors, ambiguity_components, logsumexp


@dataclass(frozen=True)
class JPDALimits:
    max_factor_cells: int = 1_000_000
    max_dp_states: int = 250_000  # Sum of stored forward layers, across components.
    max_transitions: int = 5_000_000  # Forward AND reverse marginal passes.
    max_iterations: int = 10_000  # LBP, per component.
    max_message_updates: int = 10_000_000  # Both directions, across components.
    tolerance: float = 1e-10

    def __post_init__(self):
        for name in ('max_factor_cells', 'max_dp_states', 'max_transitions', 'max_iterations', 'max_message_updates'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('positive integer JPDA resource limits required')
        if (isinstance(self.tolerance, bool) or not math.isfinite(self.tolerance)
                or not 0 < self.tolerance <= 1e-6):
            raise ValueError('JPDA tolerance must be in (0, 1e-6]')


@dataclass(frozen=True)
class JPDAMarginals:
    pair: tuple[tuple[float, ...], ...]
    left_unmatched: tuple[float, ...]
    right_unmatched: tuple[float, ...]
    factor_sha256: str
    algorithm: str
    log_partition: float | None
    components: int
    stored_dp_states: int
    peak_layer_states: int
    transitions: int
    iterations: int
    message_updates: int
    log_message_residual: float
    marginal_consistency_error: float

    @property
    def exact_model_marginals(self):
        # Exact elimination algorithm, subject to floating-point arithmetic.
        return self.algorithm == 'jpda-exact-frontier-v1'

    @property
    def posterior_error_bound(self):
        # A message residual is never a posterior approximation/mass bound.
        return None


class _Work:
    def __init__(self, limits):
        self.limits, self.states, self.peak, self.transitions = limits, 0, 0, 0

    def add_states(self, count):
        self.states += count
        self.peak = max(self.peak, count)
        if self.states > self.limits.max_dp_states:
            raise ValueError('JPDA stored-state budget exceeded; no truncated marginals returned')

    def transition(self):
        self.transitions += 1
        if self.transitions > self.limits.max_transitions:
            raise ValueError('JPDA transition budget exceeded; no truncated marginals returned')


def _check(factors, limits):
    if type(factors) is not LogAssociationFactors or type(limits) is not JPDALimits:
        raise TypeError('immutable association factors and JPDA limits required')
    n, m = factors.shape
    if n * m + n + m > limits.max_factor_cells:
        raise ValueError('JPDA factor-cell capacity exceeded before allocating messages')


def _gain(factors, left, right):
    # Divide out the all-unmatched weight. Changing unmatched factors therefore
    # changes posterior weights; they are not discarded by row normalisation.
    gain = np.empty((len(left), len(right)))
    for a, i in enumerate(left):
        for b, j in enumerate(right):
            gain[a, b] = (math.fsum((factors.log_pair[i][j], -factors.log_left_unmatched[i],
                                    -factors.log_right_unmatched[j])) if factors.allowed[i][j]
                          else -math.inf)
    if np.any(np.isnan(gain)) or np.any(gain == math.inf):
        raise ValueError('JPDA factor ratios exceed log arithmetic range')
    return gain


def _exact_component(gain, work):
    n, m = gain.shape
    edges = [tuple(j for j in range(m) if math.isfinite(gain[i, j])) for i in range(n)]
    last = {j: i for i, row in enumerate(edges) for j in row}
    closing = [sum(1 << j for j, end in last.items() if end == i) for i in range(n)]

    def successors(i, mask):
        yield -1, mask & ~closing[i], 0.
        for j in edges[i]:
            if not mask & (1 << j):
                yield j, (mask | (1 << j)) & ~closing[i], float(gain[i, j])

    forward = [{0: 0.}]
    work.add_states(1)
    for i in range(n):
        layer = {}
        for mask, weight in forward[-1].items():
            for _, target, delta in successors(i, mask):
                work.transition()
                if target not in layer:
                    # Enforce the memory cap BEFORE adding a new dictionary slot.
                    if work.states + len(layer) + 1 > work.limits.max_dp_states:
                        raise ValueError('JPDA stored-state budget exceeded; no truncated marginals returned')
                    layer[target] = weight + delta
                else:
                    layer[target] = float(np.logaddexp(layer[target], weight + delta))
        work.add_states(len(layer))
        forward.append(layer)
    log_z = forward[-1][0]
    if not math.isfinite(log_z):
        raise ValueError('JPDA partition exceeds log arithmetic range')
    pair = np.zeros_like(gain)
    missed = np.zeros(n)
    backward, right_logs = {0: 0.}, np.full(m, -math.inf)
    for i in range(n - 1, -1, -1):
        layer, logs = {}, np.full(m + 1, -math.inf)
        expires = tuple(j for j, end in last.items() if end == i)
        for mask, weight in forward[i].items():
            tails = []
            for j, target, delta in successors(i, mask):
                work.transition()
                tail = delta + backward[target]
                tails.append(tail)
                contribution = weight + tail - log_z
                logs[j + 1] = np.logaddexp(logs[j + 1], contribution)
                for column in expires:
                    if j != column and not mask & (1 << column):
                        right_logs[column] = np.logaddexp(right_logs[column], contribution)
            layer[mask] = logsumexp(tails)
        for j, value_log in enumerate(logs):
            value = math.exp(value_log)
            if j == 0:
                missed[i] = value
            else:
                pair[i, j - 1] = value
        backward = layer
    # Compute a column's unused mass when it expires, not as 1 - matched mass.
    # This preserves representable tiny missed probabilities near a hard match.
    unused = np.exp(right_logs)
    error = float(max(np.max(np.abs(pair.sum(axis=1) + missed - 1.), initial=0.),
                      np.max(np.abs(pair.sum(axis=0) + unused - 1.), initial=0.)))
    if error > 1e-8:
        raise ValueError('JPDA elimination marginal conservation failed numerically')
    return pair, missed, unused, log_z


def exact_jpda(factors, *, limits=None):
    """Sum all legal single-scan assignments via sparse frontier elimination.

An expired column is forgotten after its last incident row. Distinct paths
ending at the same frontier mask are summed, never winner-take-all compressed.
Components transpose when this reduces the number of tracked columns.
"""
    limits = limits or JPDALimits()
    _check(factors, limits)
    n, m = factors.shape
    pair, left_miss, right_miss = np.zeros((n, m)), np.ones(n), np.ones(m)
    parts, log_partitions, work = ambiguity_components(factors), [], _Work(limits)
    for part in parts:
        if not part.left or not part.right:
            continue
        gain = _gain(factors, part.left, part.right)
        transpose = gain.shape[1] > gain.shape[0]
        p, left, right, log_z = _exact_component(gain.T if transpose else gain, work)
        if transpose:
            p, left, right = p.T, right, left
        pair[np.ix_(part.left, part.right)] = p
        left_miss[list(part.left)], right_miss[list(part.right)] = left, right
        log_partitions.append(log_z)
    partition = math.fsum((*factors.log_left_unmatched, *factors.log_right_unmatched, *log_partitions))
    if not math.isfinite(partition):
        raise ValueError('JPDA absolute partition exceeds log arithmetic range')
    consistency = max(np.max(np.abs(pair.sum(axis=1) + left_miss - 1.), initial=0.),
                      np.max(np.abs(pair.sum(axis=0) + right_miss - 1.), initial=0.))
    return JPDAMarginals(tuple(map(tuple, pair)), tuple(left_miss), tuple(right_miss),
        factors.digest(), 'jpda-exact-frontier-v1', partition, len(parts), work.states,
        work.peak, work.transitions, 0, 0, 0., float(consistency))


def _excluding_lse(values, axis):
    """log(1 + sum_{k != j} exp(x_k)), without total-minus-term cancellation."""
    values = np.swapaxes(values, axis, -1)
    one = np.zeros((*values.shape[:-1], 1))
    before = np.concatenate((np.full_like(one, -math.inf),
                             np.logaddexp.accumulate(values, axis=-1)[..., :-1]), axis=-1)
    before = np.logaddexp(0., before)
    after = np.concatenate((np.logaddexp.accumulate(values[..., ::-1], axis=-1)[..., ::-1][..., 1:],
                            np.full_like(one, -math.inf)), axis=-1)
    return np.swapaxes(np.logaddexp(before, after), axis, -1)


def _beliefs(gain, nu, mu):
    row_z = np.logaddexp(0., np.logaddexp.reduce(gain + nu, axis=1))
    col_z = np.logaddexp(0., np.logaddexp.reduce(mu, axis=0))
    pair = np.exp(gain + nu - row_z[:, None])
    return pair, np.exp(-row_z), np.exp(-col_z)


def lbp_jpda(factors, *, limits=None):
    """Approximate JPDA matching marginals; no partition or mass guarantee.

Uses log-message change AND row/column consistency as fixed-point stopping
criteria. Iteration exhaustion raises rather than returning unconverged values.
No confidence cutoff, candidate deletion or BP-to-exact error bound is used.
"""
    limits = limits or JPDALimits()
    _check(factors, limits)
    n, m = factors.shape
    pair, left_miss, right_miss = np.zeros((n, m)), np.ones(n), np.ones(m)
    parts = ambiguity_components(factors)
    iterations, messages, residual, consistency = 0, 0, 0., 0.
    for part in parts:
        if not part.left or not part.right:
            continue
        gain = _gain(factors, part.left, part.right)
        allowed = np.isfinite(gain)
        nu = np.zeros_like(gain)
        for iteration in range(1, limits.max_iterations + 1):
            messages += 2 * gain.size
            if messages > limits.max_message_updates:
                raise ValueError('JPDA LBP message-update budget exceeded')
            mu = gain - _excluding_lse(gain + nu, 1)
            updated = -_excluding_lse(mu, 0)
            change = float(np.max(np.abs(updated[allowed] - nu[allowed]), initial=0.))
            nu = updated
            p, left, right = _beliefs(gain, nu, mu)
            error = float(np.max(np.abs(p.sum(axis=0) + right - 1.), initial=0.))
            if change <= limits.tolerance and error <= limits.tolerance:
                break
        else:
            raise ValueError('JPDA LBP did not converge within iteration budget')
        if not all(np.all(np.isfinite(a)) for a in (p, left, right)):
            raise ValueError('JPDA LBP marginal arithmetic is not finite')
        iterations += iteration
        residual, consistency = max(residual, change), max(consistency, error)
        pair[np.ix_(part.left, part.right)] = p
        left_miss[list(part.left)], right_miss[list(part.right)] = left, right
    return JPDAMarginals(tuple(map(tuple, pair)), tuple(left_miss), tuple(right_miss),
        factors.digest(), 'jpda-lbp-williams-lau-v1', None, len(parts), 0, 0, 0,
        iterations, messages, residual, consistency)
