"""Recoverable component variant with bounded frontier-to-terminal proposals.

Retains the original backend as an ablation. Completion proposes legal explicit
classes; it never erases their frontier regions or replaces an upper bound by
a greedy score. Budgets charge every suffix step, including cached prefixes.
"""
from __future__ import annotations

from dataclasses import dataclass

from .frontier_completion import frontier_completion_operation
from .persistent_component_tracking import PersistentComponentConfig, PersistentComponentTracker
from .learned_component_allocation import AllocationTeacherTracker, LearnedComponentTracker


@dataclass(frozen=True)
class PersistentCompletionConfig(PersistentComponentConfig):
    frontier_completions_per_component: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.frontier_completions_per_component <= 8:
            raise ValueError('one to eight frontier proposals per component required')


class _CompletionProposals:
    CONFIG_TYPE = PersistentCompletionConfig

    def _prepare_allocation_proposals(self, proposals, *, weights, prefix_count, frontier_count):
        super()._prepare_allocation_proposals(proposals,weights=weights,
            prefix_count=prefix_count,frontier_count=frontier_count)
        self._completion_bases={c:set() for c in proposals}
        for c in sorted(proposals):
            if not weights[c]:
                continue
            for _ in range(self.config.frontier_completions_per_component):
                operation=frontier_completion_operation(self.kernels[c],remaining=self.config.state.expansion_budget,
                    prefix_count=prefix_count,frontier_count=frontier_count,config=self.config,excluded=proposals[c])
                if operation is None:
                    break
                proposals[c].append(operation['base'])
                self._completion_bases[c].add(operation['base'])

    def _next_allocation_work(self,kernel,proposals,remaining,prefix_count,frontier_count):
        operation=super()._next_allocation_work(kernel,proposals,remaining,prefix_count,frontier_count)
        # The shared executor rechecks remaining budget/capacity on selection.
        if operation['base'] is not None:
            component=next(c for c,k in self.kernels.items() if k is kernel)
            if operation['base'] in self._completion_bases[component]:
                operation=dict(operation,kind='frontier_completion_proposal')
        return operation

    def _allocation_summary(self):
        return dict(super()._allocation_summary(),frontier_completion_enabled=True,
            frontier_completion_candidates=sum(map(len,self._completion_bases.values())),
            frontier_completion_limit_per_component=self.config.frontier_completions_per_component,
            frontier_completion_preserves_unresolved_support=True)


class CompletionComponentTracker(_CompletionProposals,PersistentComponentTracker):
    SCHEMA = 'persistent_completion_component_identity_v1'


class CompletionTeacherTracker(_CompletionProposals,AllocationTeacherTracker):
    SCHEMA = 'persistent_completion_allocation_teacher_v1'


class CompletionLearnedTracker(_CompletionProposals,LearnedComponentTracker):
    SCHEMA = 'persistent_completion_learned_allocation_v1'
