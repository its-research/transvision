"""Causal root-reachability relaxation for event-complete irreversible ranking.

For each suffix row, propagate possible roots along its causal parent edges.
The root-specific transition upper bound sums a parent potential only when
that parent can reach the root. This does not multiply transition potentials
along paths, truncate positive edges, or infer a posterior from reachability.
Source/frame assignments and their feasible dual covers remain inherited.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .hypothesis_bank import logsumexp
from .persistent_sparse_slot_bound_beam import (
    PersistentSparseSlotBoundBeamConfig, PersistentSparseSlotBoundBeamTracker,
    _SparseSlotEnvelope,
)


@dataclass(frozen=True)
class PersistentReachableSlotBoundBeamConfig(PersistentSparseSlotBoundBeamConfig):
    max_reachability_terms: int = 200_000


class _ReachableSlotEnvelope(_SparseSlotEnvelope):
    def prefix_upper(self, handle):
        depth = self.kernel._prefix(handle).depth
        if not self.start <= depth <= self.kernel.n:
            raise ValueError('reachable slot-ranking prefix outside current suffix')
        self.work.charge('prefix_bound_calls')
        roots, occupied = {}, {}

        def root(parent):
            if parent not in roots:
                self.work.charge('root_ancestor_queries')
                roots[parent] = self.kernel._prefix(self.kernel.ancestor(handle, parent+1)).root
            return roots[parent]

        # Occupancy depends only on the fixed prefix, not a guessed future.
        for slot in sorted({self.slots[i] for i in range(depth, self.kernel.n)}):
            if slot[0] != -1:
                used = occupied.setdefault(slot, set())
                for i in self.slot_members[slot]:
                    self.work.charge('slot_occupancy_checks')
                    if i < depth:
                        used.add(root(i))

        reachable, grouped, terms = {}, {}, 0
        for index in range(depth, self.kernel.n):
            slot = self.slots[index]
            used = occupied.get(slot, set())
            known, birth = {}, None
            for parent, weight in self.rows[index-self.start]:
                self.work.charge('bound_factor_terms')
                if parent < 0:
                    birth = weight
                    continue
                if slot[0] != -1 and self.slots[parent] == slot:
                    self.work.charge('same_slot_parent_edges_excluded')
                    continue
                candidates = (root(parent),) if parent < depth else reachable[parent]
                for candidate in candidates:
                    self.work.charge('reachability_parent_root_terms')
                    terms += 1
                    if terms > self.work.config.max_reachability_terms:
                        raise ValueError('root reachability capacity exhausted; event not committed')
                    # A future birth already occupies its own source/frame.
                    # An existing root may also occupy other prefix slots.
                    if slot[0] != -1 and (candidate in used or self.slots[candidate] == slot):
                        self.work.charge('unreachable_slot_root_terms_excluded')
                        continue
                    known.setdefault(candidate, []).append(weight)
            if birth is None:
                raise ValueError('finite birth potential required for reachability')
            # Store sorted roots for deterministic traversal. Multiple routes
            # to the SAME parent/root contribute its potential only once.
            reachable[index] = tuple(sorted({index, *known}))
            key = slot if slot[0] != -1 else (-1, index)
            grouped.setdefault(key, []).append((birth,
                {r: logsumexp(weights) for r, weights in known.items()}, -math.inf))
        self.work.reachability_terms_peak = max(
            getattr(self.work, 'reachability_terms_peak', 0), terms)
        bounds, scales = [], []
        for slot, details in sorted(grouped.items()):
            columns = {r for _, known, _ in details for r in known}
            value, scale = self._assignment(details, columns, occupied.get(slot, set()))
            bounds.append(value)
            scales.append(scale)
        self.work.root_cache_peak = max(self.work.root_cache_peak, len(roots))
        base = self.kernel._weight(handle)
        return math.fsum([base, *bounds, self._margin(depth, base, math.fsum(scales))])


class PersistentReachableSlotBoundBeamTracker(PersistentSparseSlotBoundBeamTracker):
    CONFIG_TYPE = PersistentReachableSlotBoundBeamConfig
    SCHEMA = 'persistent_irreversible_joint_reachable_slot_bound_beam_v1'
    RANKING_BOUND = 'causal_root_reachable_source_frame_assignment_dual_v1'

    def _envelope(self, kernel, start):
        return _ReachableSlotEnvelope(kernel, start, self._ranking_work)

    def _audit_properties(self):
        return dict(super()._audit_properties(),
            unknown_parent_rows_keep_all_available_known_roots=False,
            unknown_parent_roots_from_causal_reachability=True,
            positive_parent_support_unchanged=True,
            future_birth_roots_have_source_frame_capacity=True)

    def _ranking_audit_extra(self, work):
        return dict(super()._ranking_audit_extra(work),
            reachability_terms_peak=getattr(work, 'reachability_terms_peak', 0),
            reachability_terms_cap=self.config.max_reachability_terms,
            ranking_work_scope='inherited_sparse_assignment_plus_causal_parent_root_reachability_terms',
            reachability_is_not_full_legal_suffix_enumeration=True)
