"""Exact sparsity decomposition of the slot assignment relaxation.

Only zero-weight columns and disconnected row/root blocks are separated. This
does NOT discard any feasible assignment, change the row potentials, or add
identity recovery. Unknown-root rows retain every admissible known-root column.
Each nontrivial block uses the original feasible dual cover, never a primal
score masquerading as a maximum bound. Resource caps are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .hypothesis_bank import logsumexp
from .persistent_slot_bound_beam import (
    PersistentSlotBoundBeamConfig, PersistentSlotBoundBeamTracker, _SlotEnvelope,
)


@dataclass(frozen=True)
class PersistentSparseSlotBoundBeamConfig(PersistentSlotBoundBeamConfig):
    pass


class _SparseSlotEnvelope(_SlotEnvelope):
    def _assignment(self, details, roots, occupied):
        available=set(roots)-occupied
        self.work.charge('assignment_sparse_root_inventory',len(roots)+len(occupied))
        owner={};parents=list(range(len(details)))
        def find(i):
            while parents[i]!=i:
                self.work.charge('assignment_sparse_union_steps')
                parents[i]=parents[parents[i]];i=parents[i]
            return i
        for i,(_,known,unknown) in enumerate(details):
            self.work.charge('assignment_sparse_rows')
            self.work.charge('assignment_sparse_known_terms',len(known))
            # Unknown parents may reach a known root indirectly: when U>0,
            # restricting the columns to direct edges would be unsound.
            edges=available if math.isfinite(unknown) else {
                r for r,w in known.items() if r in available and math.isfinite(w)}
            for r in sorted(edges):
                self.work.charge('assignment_sparse_edges')
                if r in owner:
                    a,b=find(i),find(owner[r]);parents[a]=b
                else:
                    owner[r]=i
        groups={};root_groups={}
        for i in range(len(details)):
            groups.setdefault(find(i),[]).append(i)
        for r,i in owner.items():
            root_groups.setdefault(find(i),set()).add(r)
        self.work.sparse_blocks_peak=max(getattr(self.work,'sparse_blocks_peak',0),len(groups))
        bounds=[];scales=[]
        for group,indices in sorted(groups.items()):
            selected=[details[i] for i in indices];columns=root_groups.get(group,set())
            self.work.charge('assignment_sparse_blocks')
            if len(selected)==1:
                birth,known,unknown=selected[0]
                self.work.charge('assignment_single_row_terms',len(columns)+1)
                value=max([birth,unknown,*[
                    logsumexp([known.get(r,-math.inf),unknown]) for r in sorted(columns)]])
                # One row has no shared-column conflict. Its row maximum is
                # a feasible dual cover with all column multipliers zero.
                upper=math.nextafter(value,math.inf);scale=abs(upper)
            else:
                upper,scale=super()._assignment(selected,columns,set())
            bounds.append(upper);scales.append(scale)
        return math.nextafter(math.fsum(bounds),math.inf),math.fsum(scales)


class PersistentSparseSlotBoundBeamTracker(PersistentSlotBoundBeamTracker):
    CONFIG_TYPE=PersistentSparseSlotBoundBeamConfig
    SCHEMA='persistent_irreversible_joint_sparse_slot_bound_beam_v1'
    RANKING_BOUND='source_frame_sparse_assignment_single_class_dual_v1'

    def _envelope(self,kernel,start):
        return _SparseSlotEnvelope(kernel,start,self._ranking_work)

    def _audit_properties(self):
        return dict(super()._audit_properties(),exact_slot_assignment_sparsity_decomposition=True,
            unknown_parent_rows_keep_all_available_known_roots=True)

    def _ranking_audit_extra(self,work):
        return dict(super()._ranking_audit_extra(work),
            sparse_assignment_blocks_peak=getattr(work,'sparse_blocks_peak',0),
            ranking_work_scope='inherited_slot_bound_plus_sparse_root_edge_union_block_and_single_row_terms',
            sparse_decomposition_changes_resource_accounting=True,
            same_physical_compute_as_dense_assignment_claimed=False)
