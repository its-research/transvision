"""Recoverable partial-assignment inference with deterministic mass bounds.

This is a local, fixed-support inference primitive, not a complete tracker.  A
frontier of disjoint row-prefix constraints retains the space that has not been
expanded.  New evidence reweights these constraints from immutable factors, so
an assignment never enumerated before can become active later.  No exhaustive
oracle is called by :class:`HypothesisBank`.

Mass certificates concern the supplied factor model, not an unknown calibrated
real-world posterior. Floating-point upper computations include conservative
roundoff padding; they are not interval-arithmetic certificates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

import numpy as np


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_vector(value, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector")
    return tuple(float(item) for item in array)


def _sum(values: Iterable[float], *, upper: bool = False) -> float:
    items = tuple(values)
    try:
        result = math.fsum(items)
        magnitude = math.fsum(abs(value) for value in items)
    except OverflowError as exc:
        raise ValueError("log arithmetic exceeds float64 range") from exc
    if not math.isfinite(result) or not math.isfinite(magnitude):
        raise ValueError("log arithmetic exceeds float64 range")
    if upper:
        result += 32 * np.finfo(float).eps * max(1., magnitude, len(items))
    if not math.isfinite(result):
        raise ValueError("log arithmetic exceeds float64 range")
    return result


def logsumexp(values: Iterable[float], *, upper: bool = False) -> float:
    """Log-sum of positive weights; the empty sum is represented by -inf."""
    items = tuple(float(value) for value in values)
    if not items:
        return -math.inf
    if any(math.isnan(value) or value == math.inf for value in items):
        raise ValueError("logsumexp inputs must be finite or negative infinity")
    maximum = max(items)
    if maximum == -math.inf:
        return maximum
    result = maximum + math.log(math.fsum(math.exp(value - maximum) for value in items))
    if upper:
        result += 32 * np.finfo(float).eps * max(1., abs(maximum), abs(result), len(items))
    if not math.isfinite(result):
        raise ValueError("log arithmetic exceeds float64 range")
    return result


@dataclass(frozen=True, slots=True)
class LogAssociationFactors:
    """Positive matched/unmatched factors; ``allowed`` describes fixed support.

    Inputs are copied into immutable tuples. Finite log factors encode strictly
    positive potentials. Gated-out edges are excluded by ``allowed`` rather
    than by non-finite sentinels. Right unmatched factors are counted once for
    every column unused by the entire legal assignment.
    """

    log_pair: tuple[tuple[float, ...], ...]
    log_left_unmatched: tuple[float, ...]
    log_right_unmatched: tuple[float, ...]
    allowed: tuple[tuple[bool, ...], ...] | None = None

    def __post_init__(self) -> None:
        left = _finite_vector(self.log_left_unmatched, "log_left_unmatched")
        right = _finite_vector(self.log_right_unmatched, "log_right_unmatched")
        pair = np.asarray(self.log_pair, dtype=np.float64)
        if not left and pair.size == 0:
            pair = pair.reshape((0, len(right)))
        if pair.shape != (len(left), len(right)) or not np.all(np.isfinite(pair)):
            raise ValueError("log_pair must have finite shape (left_count, right_count)")
        if self.allowed is None:
            allowed = np.ones(pair.shape, dtype=bool)
        else:
            allowed = np.asarray(self.allowed)
            if not left and allowed.size == 0:
                allowed = allowed.reshape(pair.shape)
            if allowed.size == 0 and allowed.shape == pair.shape:
                allowed = allowed.astype(bool)
            if allowed.shape != pair.shape or allowed.dtype.kind != "b":
                raise ValueError("allowed must be a boolean matrix of the pair shape")
        object.__setattr__(self, "log_left_unmatched", left)
        object.__setattr__(self, "log_right_unmatched", right)
        object.__setattr__(self, "log_pair", tuple(tuple(float(x) for x in row) for row in pair))
        object.__setattr__(self, "allowed", tuple(tuple(bool(x) for x in row) for row in allowed))
        # Reject factors whose ordinary log arithmetic cannot be represented.
        row_relaxation_log_upper(self)

    @classmethod
    def from_positive(cls, pair, left_unmatched, right_unmatched, *, allowed=None):
        values = [np.asarray(item, dtype=float) for item in (pair, left_unmatched, right_unmatched)]
        if any(not np.all(np.isfinite(item)) or np.any(item <= 0) for item in values):
            raise ValueError("all potentials must be finite and strictly positive")
        return cls(*(np.log(item) for item in values), allowed=allowed)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.log_left_unmatched), len(self.log_right_unmatched)

    def add(self, other: "LogAssociationFactors") -> "LogAssociationFactors":
        """Multiply same-support factors (e.g. a factorized evidence likelihood)."""
        if type(other) is not LogAssociationFactors or self.shape != other.shape or self.allowed != other.allowed:
            raise ValueError("evidence factors must preserve the exact original shape and support")
        return LogAssociationFactors(
            tuple(tuple(_sum((a, b)) for a, b in zip(left, right))
                  for left, right in zip(self.log_pair, other.log_pair)),
            tuple(_sum((a, b)) for a, b in zip(self.log_left_unmatched, other.log_left_unmatched)),
            tuple(_sum((a, b)) for a, b in zip(self.log_right_unmatched, other.log_right_unmatched)),
            self.allowed,
        )

    def digest(self) -> str:
        return _digest({"pair": self.log_pair, "left": self.log_left_unmatched,
                        "right": self.log_right_unmatched, "allowed": self.allowed})


def _validate_prefix(factors: LogAssociationFactors, prefix: tuple[int, ...]) -> None:
    n, m = factors.shape
    if not isinstance(prefix, tuple) or len(prefix) > n:
        raise ValueError("prefix must be a tuple with at most one choice per row")
    used = set()
    for row, column in enumerate(prefix):
        if type(column) is not int or column < -1 or column >= m:
            raise ValueError("prefix choices must be -1 (unmatched) or valid column indices")
        if column >= 0:
            if column in used or not factors.allowed[row][column]:
                raise ValueError("prefix violates one-to-one constraints or fixed support")
            used.add(column)


def assignment_log_weight(factors: LogAssociationFactors, choices: tuple[int, ...]) -> float:
    _validate_prefix(factors, choices)
    if len(choices) != factors.shape[0]:
        raise ValueError("assignment must contain exactly one choice per left row")
    used = {column for column in choices if column >= 0}
    return _sum([
        *(factors.log_left_unmatched[row] if column < 0 else factors.log_pair[row][column]
          for row, column in enumerate(choices)),
        *(weight for column, weight in enumerate(factors.log_right_unmatched) if column not in used),
    ])


def row_relaxation_log_upper(factors: LogAssociationFactors, prefix: tuple[int, ...] = ()) -> float:
    """Upper bound subtree Z by relaxing only remaining-column uniqueness.

    Prefix constraints are exact. Remaining rows may independently use any
    still-free column; this adds invalid assignments with positive weights.
    """
    _validate_prefix(factors, prefix)
    if len(prefix) == factors.shape[0]:
        return _sum((assignment_log_weight(factors, prefix),), upper=True)
    used = {column for column in prefix if column >= 0}
    terms = [*factors.log_right_unmatched]
    for row, column in enumerate(prefix):
        terms.append(factors.log_left_unmatched[row] if column < 0 else
                     _sum((factors.log_pair[row][column], -factors.log_right_unmatched[column]), upper=True))
    for row in range(len(prefix), factors.shape[0]):
        terms.append(logsumexp([
            factors.log_left_unmatched[row],
            *(_sum((factors.log_pair[row][column], -factors.log_right_unmatched[column]), upper=True)
              for column in range(factors.shape[1])
              if column not in used and factors.allowed[row][column]),
        ], upper=True))
    return _sum(terms, upper=True)


@dataclass(frozen=True, slots=True)
class AssignmentHypothesis:
    choices: tuple[int, ...]
    log_weight: float

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((row, column) for row, column in enumerate(self.choices) if column >= 0)


def enumerate_assignments(factors: LogAssociationFactors, *, max_hypotheses: int = 100000) -> tuple[AssignmentHypothesis, ...]:
    """Independent exhaustive small-graph oracle; raises instead of truncating."""
    _integer(max_hypotheses, "max_hypotheses", 1)
    result = []

    def visit(prefix):
        if len(prefix) == factors.shape[0]:
            if len(result) == max_hypotheses:
                raise ValueError("exhaustive oracle exceeded max_hypotheses")
            result.append(AssignmentHypothesis(prefix, assignment_log_weight(factors, prefix)))
            return
        row = len(prefix)
        visit(prefix + (-1,))
        used = set(prefix)
        for column in range(factors.shape[1]):
            if column not in used and factors.allowed[row][column]:
                visit(prefix + (column,))

    visit(())
    return tuple(sorted(result, key=lambda item: (-item.log_weight, item.choices)))


@dataclass(frozen=True, slots=True)
class AmbiguityComponent:
    left: tuple[int, ...]
    right: tuple[int, ...]


def ambiguity_components(factors: LogAssociationFactors) -> tuple[AmbiguityComponent, ...]:
    """Connected components of the allowed bipartite graph, including isolates."""
    n, m = factors.shape
    remaining = set(range(n + m))
    components = []
    while remaining:
        seed = min(remaining)
        queue = [seed]
        remaining.remove(seed)
        found = {seed}
        while queue:
            node = queue.pop()
            neighbors = ([n + column for column in range(m) if factors.allowed[node][column]]
                         if node < n else [row for row in range(n) if factors.allowed[row][node - n]])
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    found.add(neighbor)
                    queue.append(neighbor)
        components.append(AmbiguityComponent(tuple(sorted(x for x in found if x < n)),
                                             tuple(sorted(x - n for x in found if x >= n))))
    return tuple(components)


def component_factors(factors: LogAssociationFactors, component: AmbiguityComponent) -> LogAssociationFactors:
    """Extract a component without introducing edges between independent objects."""
    if component not in ambiguity_components(factors):
        raise ValueError("component must be an exact connected component of the factor support")
    return LogAssociationFactors(
        tuple(tuple(factors.log_pair[i][j] for j in component.right) for i in component.left),
        tuple(factors.log_left_unmatched[i] for i in component.left),
        tuple(factors.log_right_unmatched[j] for j in component.right),
        tuple(tuple(factors.allowed[i][j] for j in component.right) for i in component.left),
    )


def posterior_omitted_mass(eta: float, *, log_likelihood_ratio: float) -> float:
    """Exact Bayes update eta_y; finite log(v/u) avoids overflow in v/u."""
    if not math.isfinite(eta) or not 0 <= eta <= 1 or not math.isfinite(log_likelihood_ratio):
        raise ValueError("eta must be in [0,1] and the log likelihood ratio finite")
    if eta in (0., 1.):
        return float(eta)
    log_odds = math.log(eta) - math.log1p(-eta) + log_likelihood_ratio
    if log_odds >= 0:
        return 1. / (1. + math.exp(-log_odds))
    odds = math.exp(log_odds)
    return odds / (1. + odds)


def regret_upper_bound(eta_upper: float, *, loss_range: float = 1., epsilon: float = 0., zeta: float = 0.) -> float:
    """L eta + 2 L epsilon + 2 L zeta; assumptions are the caller's obligation.

    epsilon is TV error against the conditional retained posterior; zeta is TV
    error of the full model against truth. This function does not estimate either.
    """
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (eta_upper, epsilon, zeta)):
        raise ValueError("eta_upper, epsilon and zeta must be finite values in [0,1]")
    if not math.isfinite(loss_range) or loss_range < 0:
        raise ValueError("loss_range must be finite and nonnegative")
    return min(loss_range, loss_range * (eta_upper + 2 * epsilon + 2 * zeta))


@dataclass(frozen=True, slots=True)
class EvidenceUpdate:
    evidence_id: str
    arrival_us: int
    factors: LogAssociationFactors

    def __post_init__(self):
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("evidence_id must be a nonempty string")
        _integer(self.arrival_us, "arrival_us")
        if type(self.factors) is not LogAssociationFactors:
            raise TypeError("evidence factors must be LogAssociationFactors")

    def digest(self) -> str:
        return _digest({"id": self.evidence_id, "arrival_us": self.arrival_us,
                        "factors": self.factors.digest()})


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ActiveBranch:
    branch_id: str
    parent_id: str
    choices: tuple[int, ...]
    log_weight: float
    first_discovered_us: int
    evidence_history: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    decision_us: int
    branch_id: str
    choices: tuple[int, ...]
    previously_enumerated: bool
    prior_active: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class BankSnapshot:
    decision_us: int
    active: tuple[ActiveBranch, ...]
    eta_upper: float
    log_retained_weight: float | None
    log_partition_upper: float
    expansions: int
    total_expansions: int
    frontier_count: int
    frontier_prefix_slots: int
    stored_factor_values: int
    discovered_assignments: int
    discovered_choice_slots: int
    peak_frontier_count: int
    recovery_events: tuple[RecoveryEvent, ...]
    evidence_ids: tuple[str, ...]
    original_factors_sha256: str
    current_factors_sha256: str
    frontier_sha256: str
    resource_limited: bool
    limit_reasons: tuple[str, ...]
    active_limit: int
    max_frontier_nodes: int
    max_discovered_leaves: int
    max_evidence_updates: int
    max_commits: int
    previous_commit: str
    commit: str


class HypothesisBank:
    """Budgeted best-bound search over a recoverable local identity space.

    active_limit is a leaf-storage limit, expansion_budget counts internal
    prefix splits. The frontier is lossless and can grow with search; this is
    bounded *work*, not a constant-memory algorithm. Finite storage limits stop
    expansion without dropping its original frontier constraint. Evidence and
    commit caps require the caller to stop or explicitly close a finite window;
    this primitive never silently forgets evidence or returns an updated claim.
    """

    def __init__(self, factors: LogAssociationFactors, *, active_limit: int = 4, information_us: int = 0,
                 max_frontier_nodes: int = 4096, max_discovered_leaves: int = 4096,
                 max_evidence_updates: int = 128, max_commits: int = 1024):
        if type(factors) is not LogAssociationFactors:
            raise TypeError("factors must be LogAssociationFactors")
        _integer(active_limit, "active_limit", 1)
        _integer(information_us, "information_us")
        for name, value in (("max_frontier_nodes", max_frontier_nodes),
                            ("max_discovered_leaves", max_discovered_leaves),
                            ("max_evidence_updates", max_evidence_updates), ("max_commits", max_commits)):
            _integer(value, name, 1)
        if active_limit > max_discovered_leaves:
            raise ValueError("active_limit cannot exceed max_discovered_leaves")
        self._original = factors
        self._factors = factors
        self._active_limit = active_limit
        self._max_frontier_nodes = max_frontier_nodes
        self._max_discovered_leaves = max_discovered_leaves
        self._max_evidence_updates = max_evidence_updates
        self._max_commits = max_commits
        self._information_us = information_us
        self._frontier = {()}
        self._active: dict[tuple[int, ...], ActiveBranch] = {}
        self._evidence: tuple[EvidenceUpdate, ...] = ()
        self._seen: dict[str, str] = {}
        self._discovered: dict[tuple[int, ...], int] = {}
        self._commits: tuple[BankSnapshot, ...] = ()
        self._total_expansions = 0
        self._peak_frontier = 1
        self._recovery_reference: tuple[tuple[int, ...], ...] = ()
        self._recovery_seen: set[tuple[int, ...]] = set()

    @property
    def original_factors(self) -> LogAssociationFactors:
        return self._original

    @property
    def commits(self) -> tuple[BankSnapshot, ...]:
        return self._commits

    @property
    def frontier_constraints(self) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted(self._frontier))

    def _branch(self, choices, decision_us):
        original = self._original.digest()
        return ActiveBranch(_digest({"original": original, "prefix": choices}),
                            _digest({"original": original, "prefix": choices[:-1]}) if choices else "0" * 64,
                            choices, assignment_log_weight(self._factors, choices),
                            self._discovered.setdefault(choices, decision_us),
                            tuple(update.evidence_id for update in self._evidence))

    def _promote_terminals(self, decision_us):
        terminals = [prefix for prefix in self._frontier if len(prefix) == self._factors.shape[0]]
        for choices in terminals:
            self._discovered.setdefault(choices, decision_us)
        candidates = set(self._active) | set(terminals)
        ranked = sorted(candidates, key=lambda choices: (-assignment_log_weight(self._factors, choices), choices))
        retained = set(ranked[:self._active_limit])
        self._frontier.update(set(self._active) - retained)
        self._frontier.difference_update(retained)
        self._active = {choices: self._branch(choices, decision_us) for choices in ranked[:self._active_limit]}

    def advance(self, *, decision_us: int, expansion_budget: int,
                evidence: EvidenceUpdate | None = None) -> BankSnapshot:
        """Reweight all prefixes, then perform at most expansion_budget splits.

        Repeating an already accepted evidence ID and payload returns its current
        committed state without applying evidence or spending additional budget.
        A new budget-only decision can be made by passing evidence=None.
        """
        _integer(decision_us, "decision_us")
        _integer(expansion_budget, "expansion_budget")
        if decision_us < self._information_us or (self._commits and decision_us < self._commits[-1].decision_us):
            raise ValueError("decision times must be causal and nondecreasing")
        if evidence is not None:
            if type(evidence) is not EvidenceUpdate:
                raise TypeError("evidence must be EvidenceUpdate")
            if evidence.arrival_us > decision_us:
                raise ValueError("future evidence cannot enter the information set")
            if evidence.evidence_id in self._seen:
                if self._seen[evidence.evidence_id] != evidence.digest():
                    raise ValueError("duplicate evidence ID has conflicting content")
                return self._commits[-1]
            if len(self._evidence) >= self._max_evidence_updates:
                raise ValueError("max_evidence_updates reached; caller must close the inference window")
            updated = self._factors.add(evidence.factors)  # Validate before mutation.
        else:
            updated = self._factors
        if len(self._commits) >= self._max_commits:
            raise ValueError("max_commits reached; caller must close the inference window")
        old_active = tuple(sorted(self._active))
        previously_discovered = set(self._discovered)
        if evidence is not None:
            self._factors = updated
            self._evidence += (evidence,)
            self._seen[evidence.evidence_id] = evidence.digest()
            self._recovery_reference = old_active
            self._recovery_seen = set()
        self._promote_terminals(decision_us)
        expansions = 0
        limit_reasons = set()
        while expansions < expansion_budget:
            internal = [prefix for prefix in self._frontier if len(prefix) < self._factors.shape[0]]
            if not internal:
                break
            chosen = None
            for prefix in sorted(internal, key=lambda item: (-row_relaxation_log_upper(self._factors, item), item)):
                row = len(prefix)
                used = set(prefix)
                children = (prefix + (-1,), *(prefix + (column,) for column in range(self._factors.shape[1])
                                              if column not in used and self._factors.allowed[row][column]))
                new_leaves = sum(len(child) == self._factors.shape[0] and child not in self._discovered
                                 for child in children)
                if len(self._frontier) - 1 + len(children) > self._max_frontier_nodes:
                    limit_reasons.add("max_frontier_nodes")
                    continue
                if len(self._discovered) + new_leaves > self._max_discovered_leaves:
                    limit_reasons.add("max_discovered_leaves")
                    continue
                chosen = (prefix, children)
                break
            if chosen is None:
                break
            prefix, children = chosen
            self._frontier.remove(prefix)
            self._frontier.update(children)
            expansions += 1
            self._peak_frontier = max(self._peak_frontier, len(self._frontier))
            self._promote_terminals(decision_us)
        self._total_expansions += expansions
        active = tuple(sorted(self._active.values(), key=lambda item: (-item.log_weight, item.choices)))
        log_retained = logsumexp(item.log_weight for item in active)
        log_residual_upper = logsumexp((row_relaxation_log_upper(self._factors, prefix)
                                      for prefix in self._frontier), upper=True)
        log_partition_upper = logsumexp((log_retained, log_residual_upper), upper=True)
        # If the entire support is explicitly active, there is no omitted mass.
        eta_upper = (0. if not self._frontier else 1. if not active else
                     min(1., max(0., -math.expm1(min(0., log_retained - log_partition_upper)))))
        recoveries = tuple(RecoveryEvent(decision_us, item.branch_id, item.choices,
                                         item.choices in previously_discovered, self._recovery_reference)
                           for item in active if self._recovery_reference
                           and item.choices not in self._recovery_reference and item.choices not in self._recovery_seen)
        self._recovery_seen.update(event.choices for event in recoveries)
        n, m = self._factors.shape
        previous_commit = self._commits[-1].commit if self._commits else "0" * 64
        payload = dict(decision_us=decision_us,
                       active=active, eta_upper=eta_upper,
                       log_retained_weight=None if not active else log_retained,
                       log_partition_upper=log_partition_upper, expansions=expansions,
                       total_expansions=self._total_expansions, frontier_count=len(self._frontier),
                       frontier_prefix_slots=sum(len(prefix) for prefix in self._frontier),
                       stored_factor_values=(2 + len(self._evidence)) * (n * m + n + m),
                       discovered_assignments=len(self._discovered),
                       discovered_choice_slots=sum(len(choices) for choices in self._discovered),
                       peak_frontier_count=self._peak_frontier, recovery_events=recoveries,
                       evidence_ids=tuple(update.evidence_id for update in self._evidence),
                       original_factors_sha256=self._original.digest(), current_factors_sha256=self._factors.digest(),
                       frontier_sha256=_digest(self.frontier_constraints),
                       resource_limited=bool(limit_reasons), limit_reasons=tuple(sorted(limit_reasons)),
                       active_limit=self._active_limit, max_frontier_nodes=self._max_frontier_nodes,
                       max_discovered_leaves=self._max_discovered_leaves,
                       max_evidence_updates=self._max_evidence_updates, max_commits=self._max_commits,
                       previous_commit=previous_commit)
        from dataclasses import asdict
        serializable = {**payload, "active": [asdict(item) for item in active],
                        "recovery_events": [asdict(item) for item in recoveries]}
        snapshot = BankSnapshot(**payload, commit=_digest(serializable))
        self._commits += (snapshot,)
        return snapshot
