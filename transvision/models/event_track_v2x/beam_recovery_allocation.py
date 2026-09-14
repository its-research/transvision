"""Learned scheduling and rollback-only teachers on the shared node-beam backend.

Reuse the immutable policy, exact probe rollback/cache and signed MODEL target.
Only extra recovery allocation changes. Features describe each component's next
actually admissible operation, not the old component solver's suffix operation.
The low-level constructors do not certify a train split or fitted checkpoint;
the cache runner and checkpoint loader enforce those separate boundaries.
"""
from __future__ import annotations

from .allocation_policy import priority_features
from .beam_recovery_tracking import BeamRecoveryConfig, BeamRecoveryTracker
from .learned_component_allocation import AllocationTeacherTracker, LearnedComponentTracker


class _BeamRecoveryAllocation:
    CONFIG_TYPE = BeamRecoveryConfig

    def __init__(self, path, *, sequence_id, config=None, **kwargs):
        config = config or self.CONFIG_TYPE()
        if type(config) is not BeamRecoveryConfig or not config.enable_recovery:
            raise ValueError('learned allocation/teacher requires enabled beam recovery')
        super().__init__(path, sequence_id=sequence_id, config=config, **kwargs)

    @classmethod
    def open(cls, path, **kwargs):
        instance = super().open(path, **kwargs)
        if not instance.config.enable_recovery:
            instance.close()
            raise ValueError('learned allocation/teacher requires enabled beam recovery')
        return instance

    def _component_inference(self, *args, **kwargs):
        predictions, audit = super()._component_inference(*args, **kwargs)
        audit.update(self._allocation_summary(), beam_recovery_allocation=True,
            allocation_trace_field='recovery_allocation_trace',
            priority_changes_only_extra_recovery_order=True,
            priority_budget_field='recovery_budget',
            priority_label_weight_scope=self.config.recovery_allocation_scope,
            priority_label_is_native_tracking_improvement=False)
        return predictions, audit

    def _select_recovery(self, candidates, *, summaries, priority_weights, masses, remaining,
                         prefix_count, frontier_count, excluded, completed, decision_us):
        operations = {c: self._recovery_operation(self.kernels[c], remaining, prefix_count,
            frontier_count, excluded[c], completed[c]) for c in candidates}
        state = dict(operations=operations, masses=masses, remaining=remaining,
            prefix_count=prefix_count, frontier_count=frontier_count, decision_us=decision_us,
            scopes={c: tuple(summaries[c]['decision_indices']) for c in candidates},
            contexts={c: dict(fallback=summaries[c]['output_handle']) for c in candidates},
            weights=priority_weights)
        component, selection = self._select_allocation(candidates, **state)
        if component not in operations:
            raise ValueError('priority selected an ineligible recovery component')
        return component, operations[component], selection

    def _priority_candidates(self, candidates, state):
        self.priority_rows += len(candidates)
        if self.priority_rows > self.MAX_PRIORITY_ROWS:
            raise ValueError('priority evaluation capacity exhausted; event not committed')
        records = []
        for component in sorted(candidates):
            operation = state['operations'][component]
            features, decision = priority_features(self.kernels[component], state['masses'][component],
                state['scopes'][component], state['contexts'][component]['fallback'],
                state['weights'][component], operation, state['remaining'], self.config.recovery_budget)
            records.append(dict(component=component, features=features, operation=operation,
                                model_bound_before=decision['risk_bound']))
        return records

    def _execute_allocation_work(self, kernel, operation, decision_us, prefix_count, frontier_count):
        # Teacher probes call the SAME executor as online recovery. A refinement
        # requests one step; a completion requests its entire affordable suffix.
        return self._execute_recovery(kernel, operation, operation['requested_steps'],
                                      prefix_count, frontier_count, decision_us)


class LearnedBeamRecoveryTracker(_BeamRecoveryAllocation, LearnedComponentTracker, BeamRecoveryTracker):
    SCHEMA = 'persistent_learned_beam_recovery_allocation_v1'


class BeamRecoveryTeacherTracker(_BeamRecoveryAllocation, AllocationTeacherTracker, BeamRecoveryTracker):
    SCHEMA = 'persistent_beam_recovery_teacher_train_only_v1'
