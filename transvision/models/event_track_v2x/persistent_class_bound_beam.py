"""Single-class ranking bounds for irreversible joint beam; NOT mass bounds.

The partition/risk computation is inherited unchanged. Ranking sums all parent
aliases of a root, and bounds unassigned parents by pooling their mass into one
root. Extra work has its own event cap; counters are not latency equivalence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import sys

from .hypothesis_bank import logsumexp
from .persistent_joint_beam import PersistentJointBeamConfig, PersistentJointBeamTracker


@dataclass(frozen=True)
class PersistentClassBoundBeamConfig(PersistentJointBeamConfig):
    max_ranking_operations: int = 50_000_000
    max_ranking_catalog_terms: int = 200_000


class _RankingWork:
    def __init__(self, config):
        self.config = config
        self.counts = Counter()
        self.total = self.catalog_peak = self.root_cache_peak = 0

    def charge(self, kind, amount=1):
        if self.total+amount > self.config.max_ranking_operations:
            raise ValueError('single-class ranking work capacity exhausted; event not committed')
        self.total += amount
        self.counts[kind] += amount


class _ClassEnvelope:
    """Event-local suffix rows only; no full-prefix reconstruction per query."""
    def __init__(self, kernel, start, work):
        if type(start) is not int or not 0 <= start <= kernel.n:
            raise ValueError('causal ranking suffix required')
        self.kernel, self.start, self.work = kernel, start, work
        rows, terms = [], 0
        for i in range(start, kernel.n):
            row = kernel._row(i)
            terms += len(row)
            if terms > work.config.max_ranking_catalog_terms:
                raise ValueError('single-class ranking catalog capacity exhausted; event not committed')
            work.charge('catalog_factor_terms', len(row))
            rows.append(row)
        self.rows = tuple(rows)
        work.catalog_peak = max(work.catalog_peak, terms)

    def _suffix(self, depth, group):
        factors = []
        for row in self.rows[depth-self.start:]:
            known, unknown, birth = {}, [], None
            for parent, weight in row:
                self.work.charge('bound_factor_terms')
                if parent < 0:
                    birth = weight
                elif parent < depth:
                    known.setdefault(group(parent), []).append(weight)
                else:
                    unknown.append(weight)
            best = max((logsumexp(weights) for weights in known.values()), default=-math.inf)
            factors.append(max(birth, logsumexp([best, logsumexp(unknown)])))
        return math.fsum(factors), math.fsum(abs(v) for v in factors)

    def _margin(self, depth, base, absolute):
        # Floating slack, not directed rounding or a formal numeric certificate.
        return 64*sys.float_info.epsilon*(self.kernel.n+1)*max(
            1., abs(base), absolute, self.kernel.suffix_abs[depth])

    def prefix_upper(self, handle):
        depth = self.kernel._prefix(handle).depth
        if not self.start <= depth <= self.kernel.n:
            raise ValueError('ranking prefix outside current suffix')
        self.work.charge('prefix_bound_calls')
        roots = {}
        def root(parent):
            if parent not in roots:
                self.work.charge('root_ancestor_queries')
                roots[parent] = self.kernel._prefix(self.kernel.ancestor(handle, parent+1)).root
            return roots[parent]
        suffix, absolute = self._suffix(depth, root)
        self.work.root_cache_peak = max(self.work.root_cache_peak, len(roots))
        base = self.kernel._weight(handle)
        return math.fsum([base, suffix, self._margin(depth, base, absolute)])

    def predecessor_suffix(self, groups):
        if len(groups) != self.start or any(type(g) is not int or g < 1 for g in groups):
            raise ValueError('complete positive predecessor grouping required')
        # Fixed old support may be rescored but must not cross these groups.
        for i in range(self.start):
            for parent, _ in self.kernel._row(i):
                self.work.charge('old_support_validation_terms')
                if parent >= 0 and groups[parent] != groups[i]:
                    raise ValueError('predecessor grouping cuts an old support edge')
        def group(parent):
            self.work.charge('predecessor_group_queries')
            return groups[parent]
        return self._suffix(self.start, group)


class PersistentClassBoundBeamTracker(PersistentJointBeamTracker):
    CONFIG_TYPE = PersistentClassBoundBeamConfig
    SCHEMA = 'persistent_irreversible_joint_class_bound_beam_v1'
    RANKING_BOUND = 'single_identity_class_completion_v1'

    def _audit_properties(self):
        return dict(super()._audit_properties(), ranking_bound=self.RANKING_BOUND,
            ranking_bound_used_for_partition_mass=False, extra_ranking_work_separately_capped=True)

    def _ranking_audit_extra(self, work):
        return {}

    def _component_inference(self, *args, **kwargs):
        self._ranking_work = _RankingWork(self.config)
        try:
            predictions, audit = super()._component_inference(*args, **kwargs)
            work = self._ranking_work
            audit.update(ranking_operations=work.total, ranking_work_counts=dict(work.counts),
                ranking_operation_cap=self.config.max_ranking_operations,
                ranking_catalog_terms_peak=work.catalog_peak,
                ranking_catalog_terms_cap=self.config.max_ranking_catalog_terms,
                ranking_root_cache_entries_peak=work.root_cache_peak,
                ranking_work_scope='catalog_terms_bound_terms_ancestor_queries_group_validation_and_lookups',
                ranking_work_excludes_inherited_weight_state_SQL_and_binary_lifting_internal_cost=True)
            audit.update(self._ranking_audit_extra(work))
            return predictions, audit
        finally:
            del self._ranking_work

    def _envelope(self, kernel, start):
        return _ClassEnvelope(kernel, start, self._ranking_work)

    def _prefix_upper_function(self, kernel, start, work):
        return self._envelope(kernel, start).prefix_upper

    def _joint_upper_functions(self, kernel, component, start, groups, cursor, work):
        envelope = self._envelope(kernel, start)
        inverse = {global_i: local for local, global_i in enumerate(self.store.members(component)[:start])}
        self._ranking_work.charge('predecessor_membership_terms', len(inverse))
        partition = [None]*start
        for previous, _ in groups:
            for global_i in self.store.members(previous):
                self._ranking_work.charge('predecessor_membership_terms')
                local = inverse.get(global_i)
                if local is None or partition[local] is not None:
                    raise ValueError('overlapping or missing old component membership')
                partition[local] = previous
        suffix, suffix_abs = envelope.predecessor_suffix(partition)
        prior_abs = math.fsum(max(abs(w) for w, _ in options) for _, options in groups)
        margin = envelope._margin(start, prior_abs, suffix_abs)
        def product_upper():
            return math.fsum([cursor.peek(), suffix, margin]) if cursor.queue else -math.inf
        return envelope.prefix_upper, product_upper

    def _advance_component(self, *args, **kwargs):
        before = self._ranking_work.counts.copy()
        beam, complete, pruning, stages = super()._advance_component(*args, **kwargs)
        counts = dict(self._ranking_work.counts-before)
        for row in pruning:
            row.update(ranking_bound=self.RANKING_BOUND,
                ranking_bound_used_for_partition_mass=False, ranking_work_counts=counts)
        return beam, complete, pruning, stages
