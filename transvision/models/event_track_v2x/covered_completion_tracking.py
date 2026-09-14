"""Coverage-aware proposal admission, retaining the conservative v1 control.

Changes storage admission only; factors, unresolved support, legal decoding,
suffix costs and upper-bound formulas are unchanged. It does not fix loose
partition bounds or make arbitrarily long suffixes affordable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .completion_component_tracking import (
    PersistentCompletionConfig, CompletionComponentTracker,
    CompletionTeacherTracker, CompletionLearnedTracker,
)
from .covered_proposal_capacity import covered_proposal_operation


@dataclass(frozen=True)
class PersistentCoveredCompletionConfig(PersistentCompletionConfig):
    coverage_admission_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.coverage_admission_version != 1:
            raise ValueError('unsupported coverage admission version')


class _CoveredAdmission:
    CONFIG_TYPE = PersistentCoveredCompletionConfig

    def _prepare_allocation_proposals(self, proposals, *, weights, prefix_count, frontier_count):
        # Replace only the conservative completion mixin's proposal preparation.
        # Existing history proposals stay at the front and are not regenerated.
        self._completion_bases = {component: set() for component in proposals}
        for component in sorted(proposals):
            if not weights[component]:
                continue
            for _ in range(self.config.frontier_completions_per_component):
                operation = covered_proposal_operation(self.kernels[component],
                    remaining=self.config.state.expansion_budget, prefix_count=prefix_count,
                    frontier_count=frontier_count, config=self.config, excluded=proposals[component])
                if operation is None:
                    break
                proposals[component].append(operation['base'])
                self._completion_bases[component].add(operation['base'])

    def _next_allocation_work(self, kernel, proposals, remaining, prefix_count, frontier_count):
        operation = covered_proposal_operation(kernel, proposals=proposals, remaining=remaining,
            prefix_count=prefix_count, frontier_count=frontier_count, config=self.config)
        if operation is None:
            return dict(base=None, kind='recoverable_prefix_refinement', requested_steps=1)
        component = next(c for c, current in self.kernels.items() if current is kernel)
        frontier = operation['base'] in self._completion_bases[component]
        return dict(operation, kind=('covered_frontier_completion_proposal' if frontier
                                     else 'covered_history_suffix_proposal'))

    def _execute_allocation_work(self, kernel, operation, decision_us, prefix_count, frontier_count):
        before = len(kernel.frontier)
        result = super()._execute_allocation_work(kernel, operation, decision_us, prefix_count, frontier_count)
        if (len(kernel.frontier) > kernel.config.state.max_frontier
                or frontier_count+len(kernel.frontier)-before > self.config.max_total_frontier):
            raise ValueError('coverage-admitted operation exceeded frontier capacity')
        return result

    def _allocation_summary(self):
        return dict(super()._allocation_summary(), coverage_aware_proposal_admission=True,
            frontier_reserve_rule='zero_if_all_explicit_classes_covered_else_one')


class CoveredCompletionTracker(_CoveredAdmission, CompletionComponentTracker):
    SCHEMA = 'persistent_covered_completion_component_identity_v1'


class CoveredCompletionTeacher(_CoveredAdmission, CompletionTeacherTracker):
    SCHEMA = 'persistent_covered_completion_allocation_teacher_v1'


class CoveredCompletionLearned(_CoveredAdmission, CompletionLearnedTracker):
    SCHEMA = 'persistent_covered_completion_learned_allocation_v1'
