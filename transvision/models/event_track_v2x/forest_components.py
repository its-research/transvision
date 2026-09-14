"""Exact support components with recoverable joint histories and shared budgets.

For FIXED row potentials, components factorize exactly: Z = product Z_c.
This does not assert that separately running a contextual neural scorer equals
running it globally. A new bridge restarts inference from all original factors,
not from the Cartesian product of previously retained Top-K leaves.
"""
from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
import math

from .hypothesis_bank import logsumexp
from .identity_forest import (
    ForestFactors, ForestSnapshot, RecoverableForestBank, digest, validate_snapshot,
)


@dataclass(frozen=True)
class ForestComponent:
    component_id: str
    indices: tuple[int, ...]
    factors: ForestFactors

    def lift(self, parents):
        if len(parents) != len(self.indices):
            raise ValueError('complete component action required')
        self.factors.roots(parents)
        return tuple(-1 if p < 0 else self.indices[p] for p in parents)


def split_forest(factors):
    """Connected components of ALL candidate edges, including low-score edges.

No confidence threshold or active-set membership is used. Exclusion constraints
cannot couple components: identities never have a parent path between them.
"""
    if type(factors) is not ForestFactors:
        raise TypeError('ForestFactors required')
    owners = list(range(len(factors.nodes)))
    def root(i):
        while owners[i] != i:
            owners[i] = owners[owners[i]]
            i = owners[i]
        return i
    for i, row in enumerate(factors.rows):
        for p, _ in row:
            if p >= 0:
                a, b = root(i), root(p)
                owners[max(a, b)] = min(a, b)
    groups = {}
    for i in range(len(owners)):
        groups.setdefault(root(i), []).append(i)
    result = []
    for group in sorted(groups.values(), key=lambda g: g[0]):
        local = {global_index: index for index, global_index in enumerate(group)}
        rows = tuple(tuple((-1 if p < 0 else local[p], weight) for p, weight in factors.rows[i]) for i in group)
        result.append(ForestComponent(digest(['forest-component', factors.nodes[group[0]].node_id]),
            tuple(group), ForestFactors(tuple(factors.nodes[i] for i in group), rows)))
    return tuple(result)


def join_actions(factors, components, actions):
    if len(components) != len(actions) or tuple(components) != split_forest(factors):
        raise ValueError('actions must cover the exact original components')
    result = [-1] * len(factors.nodes)
    for component, action in zip(components, actions):
        for index, parent in zip(component.indices, component.lift(action)):
            result[index] = parent
    result = tuple(result)
    factors.roots(result)
    return result


@dataclass(frozen=True)
class ComponentPosterior:
    component: ForestComponent
    posterior: ForestSnapshot
    predecessors: tuple[tuple[str, str], ...]
    merge_restart: bool
    restart_reason: str | None
    recovered_predecessor_prefixes: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class ComponentForestSnapshot:
    decision_us: int
    factors_sha256: str
    components: tuple[ComponentPosterior, ...]
    expansion_budget: int
    expansions: int
    # component ID, weighted risk before, splits consumed, eta after
    allocation_trace: tuple[tuple[str, float, int, float], ...]
    log_partition_upper: float
    product_omitted_mass_upper: float
    weighted_truncation_risk_upper: float
    frontier_count: int
    discovered_count: int
    configuration_sha256: str
    previous_commit: str
    commit: str


def _aggregate(components):
    total = sum(n.source_id != -1 for c in components for n in c.component.factors.nodes)
    upper = math.fsum(c.posterior.log_partition_upper for c in components)
    if any(not c.posterior.active for c in components):
        eta = 1.
    elif not any(c.posterior.frontier for c in components):
        eta = 0.
    else:
        ratio = math.fsum(logsumexp(c.posterior.active_log_weights)-c.posterior.log_partition_upper
                          for c in components)
        eta = max(0., min(1., -math.expm1(min(0., ratio))))
    risk = math.fsum(sum(n.source_id != -1 for n in c.component.factors.nodes) * c.posterior.eta_upper
                     for c in components) / total if total else 0.
    return upper, eta, risk


class ComponentForestBank:
    """Per-component recoverable frontiers; exactly one commit per decision.

Merges keep prior immutable component receipts and rebuild the new region's
complete support. They do not preserve its enumeration work, so restarts are
audited and charged. Capacity rejection is atomic and never cuts an edge.
"""
    def __init__(self, *, active_limit=4, max_nodes=4096, max_component_nodes=128,
                 max_components=2048, max_frontier=4096, max_discovered=65536,
                 max_total_frontier=65536, max_total_discovered=262144, max_commits=64):
        options = locals().copy()
        options.pop('self')
        if any(type(value) is not int or value < 1 for value in options.values()):
            raise ValueError('positive integer component resource limits required')
        self.options = options
        self.configuration_sha256 = digest(options)
        self.factors = ForestFactors((), ())
        self.banks, self.commits, self.messages = {}, (), {}

    def fork(self):
        """Share immutable history; advance forks each affected mutable bank."""
        result = copy(self)
        result.options, result.banks, result.messages = dict(self.options), dict(self.banks), dict(self.messages)
        return result

    def advance(self, *, factors, decision_us, expansion_budget, message_id):
        if (type(factors) is not ForestFactors or type(decision_us) is not int or decision_us < 0
                or type(expansion_budget) is not int or expansion_budget < 0
                or not isinstance(message_id, str) or not message_id):
            raise ValueError('validated factors, causal decision, budget and message required')
        request = digest([factors.digest(), decision_us, expansion_budget])
        if self.configuration_sha256 != digest(self.options):
            raise ValueError('component resource configuration changed')
        if message_id in self.messages:
            previous, snapshot = self.messages[message_id]
            if request != previous:
                raise ValueError('conflicting duplicate component update')
            return snapshot
        if (any(n.arrival_us > decision_us for n in factors.nodes)
                or self.commits and decision_us < self.commits[-1].decision_us):
            raise ValueError('future input or nonmonotonic component decision')
        size = len(self.factors.nodes)
        if (len(factors.nodes) < size or factors.nodes[:size] != self.factors.nodes
                or any(tuple(p for p, _ in old) != tuple(p for p, _ in new)
                       for old, new in zip(self.factors.rows, factors.rows))):
            raise ValueError('original identity nodes and parent support cannot be rewritten')
        options = self.options
        components = split_forest(factors)
        if (len(factors.nodes) > options['max_nodes'] or len(components) > options['max_components']
                or any(len(c.indices) > options['max_component_nodes'] for c in components)
                or len(self.commits) >= options['max_commits']):
            raise ValueError('component/node/window capacity exhausted before mutation')
        previous = {} if not self.commits else {c.component.component_id: c for c in self.commits[-1].components}
        owner = {n.node_id: key for key, bank in self.banks.items() for n in bank.factors.nodes}
        staged, contexts, predecessors, reasons = {}, {}, {}, {}
        for c in components:
            keys = tuple(sorted({owner[n.node_id] for n in c.factors.nodes if n.node_id in owner}))
            predecessors[c.component_id] = keys
            append_pressure = (len(keys) == 1
                and len(c.indices) > len(self.banks[keys[0]].factors.nodes)
                and len(self.banks[keys[0]].frontier | self.banks[keys[0]].active) > options['max_frontier'])
            reasons[c.component_id] = ('merge' if len(keys) > 1 else
                                       'append_frontier_pressure' if append_pressure else None)
            if len(keys) == 1 and not append_pressure:
                bank = self.banks[keys[0]].fork()
            else:
                # Coarsening to the root loses enumeration WORK, not support.
                # Original factors and immutable prior receipts remain intact.
                bank = RecoverableForestBank(ForestFactors((), ()), active_limit=options['active_limit'],
                    max_nodes=options['max_component_nodes'], max_frontier=options['max_frontier'],
                    max_discovered=options['max_discovered'], max_commits=options['max_commits'])
            contexts[c.component_id] = bank._begin_update(c.factors, decision_us)
            staged[c.component_id] = bank
        frontier = sum(len(b.frontier) for b in staged.values())
        discovered = sum(len(b.discovered) for b in staged.values())
        if frontier > options['max_total_frontier'] or discovered > options['max_total_discovered']:
            raise ValueError('shared component storage exhausted before mutation')
        counts = {key: sum(n.source_id != -1 for n in b.factors.nodes) for key, b in staged.items()}
        total = sum(counts.values())
        weights = {key: count / total if total else 0. for key, count in counts.items()}
        eta = {key: b._mass()[3] for key, b in staged.items()}
        consumed, limited, blocked, trace = dict.fromkeys(staged, 0), dict.fromkeys(staged, False), set(), []
        remaining = expansion_budget
        while remaining:
            candidates = [key for key, b in staged.items() if key not in blocked and weights[key] > 0
                          and eta[key] > 0 and any(len(p) < len(b.factors.nodes) for p in b.frontier)]
            if not candidates:
                break
            key = min(candidates, key=lambda k: (-weights[k]*eta[k], k))
            b = staged[key]
            before_frontier, before_discovered = len(b.frontier), len(b.discovered)
            risk_before = weights[key]*eta[key]
            steps, stopped = b._refine(1,
                frontier_cap=before_frontier+options['max_total_frontier']-frontier,
                discovered_cap=before_discovered+options['max_total_discovered']-discovered)
            frontier += len(b.frontier)-before_frontier
            discovered += len(b.discovered)-before_discovered
            remaining -= steps
            consumed[key] += steps
            limited[key] |= stopped
            eta[key] = b._mass()[3]
            trace.append((key, risk_before, steps, eta[key]))
            if steps == 0:
                blocked.add(key)
        receipts = []
        for c in components:
            key, bank = c.component_id, staged[c.component_id]
            posterior = bank._finish_update(contexts[key], decision_us, consumed[key], limited[key])
            prior_keys = predecessors[key]
            recovered = set()
            if reasons[key] is not None:
                location = {n.node_id: i for i, n in enumerate(c.factors.nodes)}
                for prior in prior_keys:
                    old = self.banks[prior]
                    indices = tuple(location[n.node_id] for n in old.factors.nodes)
                    inverse = {j: i for i, j in enumerate(indices)}
                    for parents in posterior.active:
                        prefix = tuple(-1 if parents[j] < 0 else inverse[parents[j]] for j in indices)
                        if old.active and prefix not in old.active and prefix not in old.discovered:
                            recovered.add((prior, prefix))
            receipts.append(ComponentPosterior(c, posterior,
                tuple((k, previous[k].posterior.commit) for k in prior_keys),
                len(prior_keys) > 1, reasons[key], tuple(sorted(recovered))))
        receipts = tuple(receipts)
        upper, product_eta, risk = _aggregate(receipts)
        payload = dict(decision_us=decision_us, factors_sha256=factors.digest(), components=receipts,
            expansion_budget=expansion_budget, expansions=expansion_budget-remaining,
            allocation_trace=tuple(trace), log_partition_upper=upper,
            product_omitted_mass_upper=product_eta, weighted_truncation_risk_upper=risk,
            frontier_count=frontier, discovered_count=discovered,
            configuration_sha256=self.configuration_sha256,
            previous_commit=self.commits[-1].commit if self.commits else '0'*64)
        snapshot = ComponentForestSnapshot(**payload, commit=digest(asdict_payload(payload)))
        self.factors, self.banks = factors, staged
        self.commits += (snapshot,)
        self.messages[message_id] = request, snapshot
        return snapshot


def asdict_payload(payload):
    # digest() is intentionally JSON-only; component dataclasses must be expanded.
    return dict(payload, components=[asdict(c) for c in payload['components']])


def validate_component_snapshot(factors, snapshot):
    if type(snapshot) is not ComponentForestSnapshot:
        raise ValueError('ComponentForestSnapshot required')
    payload = asdict(snapshot)
    recorded = payload.pop('commit')
    if (recorded != digest(payload) or snapshot.factors_sha256 != factors.digest()
            or tuple(c.component for c in snapshot.components) != split_forest(factors)):
        raise ValueError('component cover, factor binding or snapshot hash changed')
    for c in snapshot.components:
        validate_snapshot(c.component.factors, c.posterior)
        if c.posterior.decision_us != snapshot.decision_us:
            raise ValueError('component decision mismatch')
    if (snapshot.expansions != sum(c.posterior.expansions for c in snapshot.components)
            or snapshot.expansions != sum(t[2] for t in snapshot.allocation_trace)
            or not 0 <= snapshot.expansions <= snapshot.expansion_budget
            or snapshot.frontier_count != sum(len(c.posterior.frontier) for c in snapshot.components)):
        raise ValueError('component allocation accounting differs')
    if _aggregate(snapshot.components) != (snapshot.log_partition_upper,
            snapshot.product_omitted_mass_upper, snapshot.weighted_truncation_risk_upper):
        raise ValueError('component mass aggregation differs')
