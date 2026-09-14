"""Recoverable joint cross-source/temporal identity histories on raw observations.

Each observation chooses a prior observation as parent or starts a new identity.
Parents precede children in the immutable arrival ordering (not necessarily in
sensor time). A cluster cannot contain two observations from one source/frame.
Thus cross-source and temporal choices share one identity exclusion constraint.

This is a finite-window factor MODEL, not an exact physical-world posterior.
Different parent forests can encode the same identity partition; their model
weights are summed, not silently deduplicated. Continuous state is replayed by
the outer tracker, never used here as a moment-matched identity substitute.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import copy
import hashlib
import heapq
import json
import math

from .hypothesis_bank import logsumexp


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class IdentityNode:
    node_id: str
    source_id: int
    information_us: int
    arrival_us: int
    frame_id: str

    def __post_init__(self):
        if (not isinstance(self.node_id, str) or not self.node_id
                or not isinstance(self.frame_id, str) or not self.frame_id
                or type(self.source_id) is not int or self.source_id not in {-1, 0, 1}
                or type(self.information_us) is not int or self.information_us < 0
                or type(self.arrival_us) is not int or self.arrival_us < self.information_us):
            raise ValueError('invalid causal node identity; source -1 denotes a fixed carry anchor')


@dataclass(frozen=True)
class ForestFactors:
    nodes: tuple[IdentityNode, ...]
    # A row always includes (-1, log_birth). Other parents are earlier indices.
    rows: tuple[tuple[tuple[int, float], ...], ...]

    def __post_init__(self):
        nodes = tuple(self.nodes)
        rows = tuple(tuple((parent, float(weight)) for parent, weight in row) for row in self.rows)
        if (len(nodes) != len(rows) or any(type(node) is not IdentityNode for node in nodes)
                or len({node.node_id for node in nodes}) != len(nodes)):
            raise ValueError('unique nodes and one row per node required')
        if any(left.arrival_us > right.arrival_us for left, right in zip(nodes, nodes[1:])):
            raise ValueError('nodes must preserve nondecreasing arrival order')
        for index, row in enumerate(rows):
            parents = [parent for parent, _ in row]
            if (not row or len(parents) != len(set(parents)) or -1 not in parents
                    or any(type(parent) is not int or not -1 <= parent < index for parent in parents)
                    or any(not math.isfinite(weight) or abs(weight) > 1000 for _, weight in row)
                    or (nodes[index].source_id == -1 and parents != [-1])):
                raise ValueError('finite bounded potentials and causal parent support required')
        object.__setattr__(self, 'nodes', nodes)
        object.__setattr__(self, 'rows', tuple(tuple(sorted(row)) for row in rows))

    def roots(self, prefix):
        if not isinstance(prefix, tuple) or len(prefix) > len(self.nodes):
            raise ValueError('invalid prefix length/type')
        roots, slots = [], {}
        for index, parent in enumerate(prefix):
            if type(parent) is not int or parent not in dict(self.rows[index]):
                raise ValueError('parent outside original support')
            root = index if parent == -1 else roots[parent]
            node = self.nodes[index]
            slot = (node.source_id, node.frame_id)
            if node.source_id != -1:
                if slot in slots.setdefault(root, set()):
                    raise ValueError('identity collision: multiple observations from one source/frame')
                slots[root].add(slot)
            roots.append(root)
        return tuple(roots)

    def children(self, prefix):
        self.roots(prefix)
        if len(prefix) == len(self.nodes):
            return ()
        result = []
        for parent, _ in self.rows[len(prefix)]:
            child = prefix + (parent,)
            try:
                self.roots(child)
            except ValueError:
                continue
            result.append(child)
        return tuple(result)

    def log_weight(self, prefix):
        self.roots(prefix)
        return math.fsum(dict(self.rows[index])[parent] for index, parent in enumerate(prefix))

    def log_upper(self, prefix):
        # Relax all remaining cluster-exclusion constraints. Positive factors
        # make every legal completion a term of this product of row sums.
        return math.fsum([self.log_weight(prefix), *(logsumexp(
            (weight for _, weight in row), upper=True) for row in self.rows[len(prefix):])])

    def digest(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ForestSnapshot:
    decision_us: int
    factors_sha256: str
    active: tuple[tuple[int, ...], ...]
    active_log_weights: tuple[float, ...]
    frontier: tuple[tuple[int, ...], ...]
    log_partition_upper: float
    eta_upper: float
    expansions: int
    total_expansions: int
    never_enumerated_recoveries: tuple[tuple[int, ...], ...]
    restored_ancestral_prefixes: tuple[tuple[int, ...], ...]
    resource_limited: bool
    previous_commit: str
    commit: str


class RecoverableForestBank:
    """Lossless disjoint prefix frontier within explicitly bounded storage.

Append/rescore retains all old support. Exhausted budgets stop expansion without
merging or discarding an unresolved prefix. Expiration/conditioning is NOT done
silently by this class; the outer tracker must emit a window-handoff audit.
"""

    def __init__(self, factors, *, active_limit=4, max_frontier=4096,
                 max_nodes=512, max_commits=256, max_discovered=65536):
        if type(factors) is not ForestFactors:
            raise TypeError('ForestFactors required')
        for value in (active_limit, max_frontier, max_nodes, max_commits, max_discovered):
            if type(value) is not int or value < 1:
                raise ValueError('positive integer resource limits required')
        if len(factors.nodes) > max_nodes:
            raise ValueError('node limit exceeded')
        self.factors = factors
        self.active_limit, self.max_frontier = active_limit, max_frontier
        self.max_nodes, self.max_commits = max_nodes, max_commits
        self.max_discovered = max_discovered
        self.frontier, self.active = {()}, set()
        self.discovered, self.commits, self.messages = set(), (), {}
        self.total_expansions = 0

    def fork(self):
        """Copy mutable inference state, share only immutable factors/receipts."""
        result = copy(self)
        result.frontier, result.active = set(self.frontier), set(self.active)
        result.discovered, result.messages = set(self.discovered), dict(self.messages)
        return result

    def _promote(self):
        terminals = {prefix for prefix in self.frontier if len(prefix) == len(self.factors.nodes)}
        self.discovered.update(terminals)
        ranked = sorted(self.active | terminals, key=lambda p: (-self.factors.log_weight(p), p))
        retained = set(ranked[:self.active_limit])
        self.frontier.update(self.active - retained)
        self.frontier.difference_update(retained)
        self.active = retained

    def advance(self, *, decision_us, expansion_budget, factors=None, message_id=None):
        if (type(decision_us) is not int or decision_us < 0 or type(expansion_budget) is not int
                or expansion_budget < 0):
            raise ValueError('nondecreasing causal decisions and nonnegative integer budget required')
        proposed = self.factors if factors is None else factors
        if type(proposed) is not ForestFactors or any(node.arrival_us > decision_us for node in proposed.nodes):
            raise ValueError('future input or invalid factors')
        if message_id is not None:
            if not isinstance(message_id, str) or not message_id:
                raise ValueError('message ID must be a nonempty string')
            fingerprint = proposed.digest()
            if message_id in self.messages:
                previous_fingerprint, previous_snapshot = self.messages[message_id]
                if previous_fingerprint != fingerprint:
                    raise ValueError('conflicting duplicate message')
                return previous_snapshot
        context = self._begin_update(proposed, decision_us)
        expansions, limited = self._refine(expansion_budget)
        return self._finish_update(context, decision_us, expansions, limited, message_id)

    def _begin_update(self, proposed, decision_us):
        """Internal transaction stage; multi-component callers own staging copies.

No immutable commit is created here. A component allocator may refine several
banks between begin and finish without allocating a snapshot per split.
"""
        if (type(decision_us) is not int or decision_us < 0 or type(proposed) is not ForestFactors
                or any(node.arrival_us > decision_us for node in proposed.nodes)):
            raise ValueError('future input or invalid factors/decision')
        if self.commits and decision_us < self.commits[-1].decision_us:
            raise ValueError('nondecreasing causal decisions required')
        if len(self.commits) >= self.max_commits or len(proposed.nodes) > self.max_nodes:
            raise ValueError('window storage exhausted; explicit handoff required before mutation')
        size = len(self.factors.nodes)
        if (proposed.nodes[:size] != self.factors.nodes or len(proposed.nodes) < size
                or any(tuple(p for p, _ in new) != tuple(p for p, _ in old)
                       for new, old in zip(proposed.rows, self.factors.rows))):
            raise ValueError('rescore/append cannot rewrite existing node identity or parent support')
        if len(proposed.nodes) > size and len(self.frontier | self.active) > self.max_frontier:
            raise ValueError('append frontier storage exhausted; explicit handoff required before mutation')
        before = set(self.active)
        discovered = set(self.discovered)
        if len(proposed.nodes) > size:
            self.frontier.update(self.active)
            self.active = set()
            # Old leaves are now internal prefixes. Their discovery history is
            # already in immutable commits; it is not a recovery at the new size.
            self.discovered = set()
        self.factors = proposed
        self._promote()
        return size, before, discovered

    def _refine(self, expansion_budget, *, frontier_cap=None, discovered_cap=None):
        """Refine unpublished state; optional caps enforce shared storage limits."""
        if type(expansion_budget) is not int or expansion_budget < 0:
            raise ValueError('nonnegative integer refinement budget required')
        frontier_cap = self.max_frontier if frontier_cap is None else min(self.max_frontier, frontier_cap)
        discovered_cap = self.max_discovered if discovered_cap is None else min(self.max_discovered, discovered_cap)
        if any(type(x) is not int or x < 0 for x in (frontier_cap, discovered_cap)):
            raise ValueError('nonnegative integer storage caps required')
        proposed = self.factors
        expansions, limited = 0, False
        while expansions < expansion_budget:
            prefixes = [prefix for prefix in self.frontier if len(prefix) < len(proposed.nodes)]
            if not prefixes:
                break
            ranked = sorted(prefixes, key=lambda p: (-proposed.log_upper(p), p))
            chosen = None
            for prefix in ranked:
                children = proposed.children(prefix)
                if len(self.frontier) - 1 + len(children) > frontier_cap:
                    limited = True
                    continue
                new_leaves = {child for child in children if len(child) == len(proposed.nodes)}
                if len(self.discovered | new_leaves) > discovered_cap:
                    limited = True
                    continue
                chosen = (prefix, children)
                break
            if chosen is None:
                break
            prefix, children = chosen
            self.frontier.remove(prefix)
            self.frontier.update(children)
            self._promote()
            expansions += 1
        self.total_expansions += expansions
        return expansions, limited

    def _mass(self):
        proposed = self.factors
        active = tuple(sorted(self.active, key=lambda p: (-proposed.log_weight(p), p)))
        weights = tuple(proposed.log_weight(prefix) for prefix in active)
        retained = logsumexp(weights)
        upper = logsumexp((retained, logsumexp((proposed.log_upper(prefix)
                             for prefix in self.frontier), upper=True)), upper=True)
        eta = (0. if not self.frontier else 1. if not active else
               max(0., min(1., -math.expm1(min(0., retained - upper)))))
        return active, weights, upper, eta

    def _finish_update(self, context, decision_us, expansions, limited, message_id=None):
        size, before, discovered = context
        proposed = self.factors
        active, weights, upper, eta = self._mass()
        # Newly extended histories are not counted as recovery of old histories.
        recoveries = tuple(prefix for prefix in active if before and len(proposed.nodes) == size
                           and prefix not in before and prefix not in discovered)
        restored = tuple(sorted({prefix[:size] for prefix in active if before and len(proposed.nodes) > size
                                 and prefix[:size] not in before and prefix[:size] not in discovered}))
        payload = dict(decision_us=decision_us, factors_sha256=proposed.digest(), active=active,
                       active_log_weights=weights, frontier=tuple(sorted(self.frontier)),
                       log_partition_upper=upper, eta_upper=eta, expansions=expansions,
                       total_expansions=self.total_expansions,
                       never_enumerated_recoveries=recoveries, resource_limited=limited,
                       restored_ancestral_prefixes=restored,
                       previous_commit=self.commits[-1].commit if self.commits else '0' * 64)
        snapshot = ForestSnapshot(**payload, commit=digest(payload))
        self.commits += (snapshot,)
        if message_id is not None:
            self.messages[message_id] = (proposed.digest(), snapshot)
        return snapshot


def validate_snapshot(factors, snapshot):
    """Check factor binding, disjoint exhaustive cover and recomputed mass.

The hash detects accidental edits, not malicious authorship. This check does
not turn floating-point estimates into certified numerical bounds.
"""
    if type(factors) is not ForestFactors or type(snapshot) is not ForestSnapshot:
        raise ValueError('invalid snapshot/factors type')
    payload = asdict(snapshot)
    commit = payload.pop('commit')
    if commit != digest(payload) or factors.digest() != snapshot.factors_sha256:
        raise ValueError('invalid snapshot hash or factor binding')
    if (len(snapshot.active) != len(snapshot.active_log_weights)
            or any(len(p) != len(factors.nodes) for p in snapshot.active)
            or any(node.arrival_us > snapshot.decision_us for node in factors.nodes)):
        raise ValueError('invalid active leaves or future input')
    leaves = snapshot.active + snapshot.frontier
    if not leaves or len(set(leaves)) != len(leaves):
        raise ValueError('cover must have distinct leaves')
    covered, internal = set(leaves), set()
    for prefix in leaves:
        factors.roots(prefix)
        internal.update(prefix[:end] for end in range(len(prefix)))
    if covered & internal:
        raise ValueError('cover must be prefix-free')
    for prefix in internal:
        if not set(factors.children(prefix)) <= covered | internal:
            raise ValueError('cover omitted legal identity histories')
    if any(factors.log_weight(p) != weight for p, weight in
           zip(snapshot.active, snapshot.active_log_weights)):
        raise ValueError('active weight differs from raw factors')
    retained = logsumexp(snapshot.active_log_weights)
    upper = logsumexp((retained, logsumexp((factors.log_upper(prefix)
                         for prefix in snapshot.frontier), upper=True)), upper=True)
    eta = (0. if not snapshot.frontier else 1. if not snapshot.active else
           max(0., min(1., -math.expm1(min(0., retained - upper)))))
    if snapshot.log_partition_upper != upper or snapshot.eta_upper != eta:
        raise ValueError('invalid partition or omitted-mass estimate')


def decode_identity_roots(factors, snapshot, *, expansion_budget=1024, max_frontier=4096):
    """Bounded Bayes action search over ALL legal forests, not just active ones.

Loss is normalized root-identity mismatch across non-anchor observations.
The action-search gap is reported separately from posterior truncation mass.
Regret <= eta + gap concerns the supplied finite-window model, in exact math;
float outputs are estimates, not interval or real-data performance certificates.
"""
    validate_snapshot(factors, snapshot)
    if (type(expansion_budget) is not int or expansion_budget < 0
            or type(max_frontier) is not int or max_frontier < 1):
        raise ValueError('invalid action budget')
    if not snapshot.active:
        return {'status': 'unresolved', 'parents': None, 'roots': None,
                'model_regret_upper_estimate': None, 'action_search_gap': None}
    normalizer = logsumexp(snapshot.active_log_weights)
    marginals = [dict() for _ in factors.nodes]
    for parents, log_weight in zip(snapshot.active, snapshot.active_log_weights):
        if factors.log_weight(parents) != log_weight:
            raise ValueError('active weight differs from raw factors')
        probability = math.exp(log_weight - normalizer)
        for index, root in enumerate(factors.roots(parents)):
            marginals[index][root] = marginals[index].get(root, 0.) + probability
    evaluated = [node.source_id != -1 for node in factors.nodes]
    count = sum(evaluated)

    def score(prefix):
        return math.fsum(marginals[index].get(root, 0.) for index, root in
                         enumerate(factors.roots(prefix)) if evaluated[index])

    def upper(prefix):
        return math.fsum([score(prefix), *(max(marginals[index].values(), default=0.)
                        for index in range(len(prefix), len(factors.nodes)) if evaluated[index])])

    best = min(((-1,) * len(factors.nodes), *snapshot.active), key=lambda p: (-score(p), p))
    best_score = score(best)
    queue = [(-upper(()), ())]
    expansions, resource_limited = 0, False
    while queue and expansions < expansion_budget:
        neg_bound, prefix = heapq.heappop(queue)
        if -neg_bound <= best_score:
            continue
        if len(prefix) == len(factors.nodes):
            value = score(prefix)
            if value > best_score or (value == best_score and prefix < best):
                best, best_score = prefix, value
            continue
        children = factors.children(prefix)
        if len(queue) + len(children) > max_frontier:
            # Put the unexpanded region back before stopping, so the action
            # gap still bounds this region rather than silently losing it.
            heapq.heappush(queue, (neg_bound, prefix))
            resource_limited = True
            break
        expansions += 1
        for child in children:
            heapq.heappush(queue, (-upper(child), child))
    global_upper = max(best_score, -queue[0][0] if queue else best_score)
    gap = max(0., global_upper - best_score) / count if count else 0.
    return {'status': 'decoded', 'parents': best, 'roots': factors.roots(best),
            'conditional_expected_loss': max(0., 1. - best_score / count) if count else 0.,
            'action_search_gap': gap, 'action_expansions': expansions,
            'action_frontier_count': len(queue), 'action_resource_limited': resource_limited,
            'model_omitted_mass_upper_estimate': snapshot.eta_upper,
            'model_regret_upper_estimate': min(1., snapshot.eta_upper + gap) if count else 0.,
            'loss': 'normalized_root_identity_mismatch_nonanchors',
            'true_posterior_or_tracking_metric_bound': False,
            'numerical_certificate': False}
