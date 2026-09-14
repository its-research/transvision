"""Source/frame assignment relaxation for ONE complete identity class.

Known roots have unit capacity within a source/frame. Unknown future roots use
independent row-private options, an explicit relaxation. Old predecessor GROUPS
are never treated as individual identities: their cursor bound remains the
previous admissible coarse-group relaxation. No partition/mass bound changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .hypothesis_bank import logsumexp
from .persistent_class_bound_beam import (
    PersistentClassBoundBeamConfig, PersistentClassBoundBeamTracker, _ClassEnvelope,
)


@dataclass(frozen=True)
class PersistentSlotBoundBeamConfig(PersistentClassBoundBeamConfig):
    max_assignment_matrix_cells: int = 262_144
    max_assignment_solves: int = 100_000


class _SlotEnvelope(_ClassEnvelope):
    def __init__(self, kernel, start, work):
        super().__init__(kernel, start, work)
        records = tuple(kernel.db.execute('SELECT i,source,frame FROM observations ORDER BY i'))
        if [i for i, _, _ in records] != list(range(kernel.n)):
            raise ValueError('complete local source/frame metadata required')
        work.charge('slot_metadata_rows', len(records))
        work.slot_metadata_peak = max(getattr(work, 'slot_metadata_peak', 0), len(records))
        self.slots = tuple((source, frame) for _, source, frame in records)
        self.slot_members = {}
        for i, slot in enumerate(self.slots):
            self.slot_members.setdefault(slot, []).append(i)

    def _assignment(self, details, roots, occupied):
        """One slot: root columns plus one private unmatched column per row."""
        available = sorted(set(roots)-occupied)
        n = len(details)
        cells = n*(len(available)+n)
        if cells > self.work.config.max_assignment_matrix_cells:
            raise ValueError('slot assignment matrix capacity exhausted; event not committed')
        if self.work.counts['assignment_solves'] >= self.work.config.max_assignment_solves:
            raise ValueError('slot assignment solve capacity exhausted; event not committed')
        self.work.charge('assignment_matrix_cells', cells)
        self.work.charge('assignment_solves')
        self.work.assignment_matrix_peak = max(getattr(self.work, 'assignment_matrix_peak', 0), cells)
        scores = np.full((n, len(available)+n), -np.inf)
        for i, (birth, known, unknown) in enumerate(details):
            for j, root in enumerate(available):
                scores[i, j] = logsumexp([known.get(root, -math.inf), unknown])
            # Birth OR a not-yet-known root, not the sum of these alternatives.
            scores[i, len(available)+i] = max(birth, unknown)
        row, column = linear_sum_assignment(-scores)
        selected = [float(scores[i, j]) for i, j in zip(row, column)]
        if len(selected) != n or not all(math.isfinite(v) for v in selected):
            raise ValueError('finite complete slot assignment required')
        # A primal assignment alone is NOT an upper bound if a solver is only
        # approximate. Use it to improve nonnegative column multipliers, then
        # construct a feasible dual cover regardless of convergence/optimality.
        multipliers = np.zeros(scores.shape[1])
        for _ in range(n):
            self.work.charge('assignment_dual_cells', cells)
            update = np.maximum(multipliers, np.max(
                scores+(multipliers[column]-np.asarray(selected))[:, None], axis=0))
            if np.array_equal(update, multipliers):
                break
            multipliers = update
        self.work.charge('assignment_dual_cells', cells)
        row_cover = np.max(np.nextafter(scores-multipliers[None, :], np.inf), axis=1)
        dual_terms = [*map(float, row_cover), *map(float, multipliers)]
        upper = math.nextafter(math.fsum(dual_terms), math.inf)
        primal = math.fsum(selected)
        if not math.isfinite(upper):
            raise ValueError('finite assignment dual cover required')
        self.work.assignment_dual_gap_max = max(getattr(self.work, 'assignment_dual_gap_max', 0.), upper-primal)
        # Upstream log-sum-exp and prefix sums still use floating slack; this
        # local outward cover is NOT an end-to-end formal numeric certificate.
        return max(upper, primal), math.fsum(abs(v) for v in dual_terms)

    def prefix_upper(self, handle):
        depth = self.kernel._prefix(handle).depth
        if not self.start <= depth <= self.kernel.n:
            raise ValueError('slot-ranking prefix outside current suffix')
        self.work.charge('prefix_bound_calls')
        roots = {}
        def root(parent):
            if parent not in roots:
                self.work.charge('root_ancestor_queries')
                roots[parent] = self.kernel._prefix(self.kernel.ancestor(handle, parent+1)).root
            return roots[parent]
        grouped = {}
        known_roots = set()
        for index in range(depth, self.kernel.n):
            slot = self.slots[index]
            known, unknown, birth = {}, [], None
            for parent, weight in self.rows[index-self.start]:
                self.work.charge('bound_factor_terms')
                if parent < 0:
                    birth = weight
                elif slot[0] != -1 and self.slots[parent] == slot:
                    # Even before root choices are known this edge is illegal.
                    self.work.charge('same_slot_parent_edges_excluded')
                elif parent < depth:
                    r = root(parent)
                    known_roots.add(r)
                    known.setdefault(r, []).append(weight)
                else:
                    unknown.append(weight)
            # Carry anchors (source -1) have no source/frame exclusion.
            key = slot if slot[0] != -1 else (-1, index)
            grouped.setdefault(key, []).append((birth,
                {r: logsumexp(v) for r, v in known.items()}, logsumexp(unknown)))
        bounds = []
        absolute = []
        for slot, details in grouped.items():
            occupied = set()
            if slot[0] != -1:
                for index in self.slot_members[slot]:
                    self.work.charge('slot_occupancy_checks')
                    if index < depth:
                        occupied.add(root(index))
            value, scale = self._assignment(details, known_roots, occupied)
            bounds.append(value)
            absolute.append(scale)
        self.work.root_cache_peak = max(self.work.root_cache_peak, len(roots))
        base = self.kernel._weight(handle)
        return math.fsum([base, *bounds, self._margin(depth, base, math.fsum(absolute))])


class PersistentSlotBoundBeamTracker(PersistentClassBoundBeamTracker):
    CONFIG_TYPE = PersistentSlotBoundBeamConfig
    SCHEMA = 'persistent_irreversible_joint_slot_bound_beam_v1'
    RANKING_BOUND = 'source_frame_assignment_single_class_dual_v1'

    def _envelope(self, kernel, start):
        return _SlotEnvelope(kernel, start, self._ranking_work)

    def _audit_properties(self):
        return dict(super()._audit_properties(), source_frame_exclusion_in_ranking_bound=True,
            unmaterialized_predecessors_use_coarse_group_bound=True,
            assignment_ranking_uses_dual_cover_not_primal_score=True,
            source_frame_assignment_is_not_full_posterior_inference=True)

    def _ranking_audit_extra(self, work):
        return dict(assignment_matrix_cells_peak=getattr(work, 'assignment_matrix_peak', 0),
            assignment_matrix_cells_cap=self.config.max_assignment_matrix_cells,
            assignment_solves=work.counts['assignment_solves'],assignment_solves_cap=self.config.max_assignment_solves,
            slot_metadata_rows_peak=getattr(work, 'slot_metadata_peak', 0),
            assignment_dual_gap_max=getattr(work, 'assignment_dual_gap_max', 0.),
            ranking_work_scope='catalog_bound_terms_ancestor_group_queries_slot_metadata_occupancy_assignment_and_dual_cells_solves',
            assignment_solver_internal_operations_not_counted=True)
