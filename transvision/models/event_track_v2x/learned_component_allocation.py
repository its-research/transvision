"""Learned priority and train-only counterfactual teacher on the SAME solver.

Teacher probes are transactionally rolled back and never enter online results.
Targets concern one predeclared operation and current MODEL decision-bound
progress, not real tracking gain, long-horizon VoI, or monotone risk reduction.
"""
from __future__ import annotations

from collections import OrderedDict
import json

from .allocation_policy import FrozenPriorityPolicy, RECIPE, TARGET, decision_bound, priority_features
from .persistent_component_tracking import PersistentComponentTracker


class _PriorityStateMixin:
    MAX_PRIORITY_ROWS = 262144
    MAX_TEACHER_STEPS = 1048576

    def _component_inference(self, *args, **kwargs):
        self.priority_rows = self.teacher_steps = 0
        return super()._component_inference(*args, **kwargs)

    def _priority_candidates(self, candidates, state):
        self.priority_rows += len(candidates)
        if self.priority_rows > self.MAX_PRIORITY_ROWS:
            raise ValueError('priority evaluation capacity exhausted; event not committed')
        records = []
        for c in sorted(candidates):
            k = self.kernels[c]
            operation = self._next_allocation_work(k, state['proposals'][c], state['remaining'],
                state['prefix_count'], state['frontier_count'])
            features, decision = priority_features(k, state['masses'][c], state['scopes'][c],
                state['contexts'][c]['fallback'], state['weights'][c], operation,
                state['remaining'], self.config.state.expansion_budget)
            records.append(dict(component=c, features=features, operation=operation,
                                model_bound_before=decision['risk_bound']))
        return records


class LearnedComponentTracker(_PriorityStateMixin, PersistentComponentTracker):
    SCHEMA = 'persistent_learned_component_allocation_v1'

    def __init__(self, path, *, sequence_id, config=None, allocation_policy):
        if type(allocation_policy) is not FrozenPriorityPolicy:
            raise TypeError('immutable frozen allocation policy required')
        self.allocation_policy = allocation_policy
        super().__init__(path, sequence_id=sequence_id, config=config)
        try:
            self.db.execute('BEGIN IMMEDIATE')
            self._set('allocation_policy_signature', allocation_policy.signature)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK'); self.close()
            raise

    @classmethod
    def open(cls, path, *, expected_prediction_sha256, expected_database_sha256, allocation_policy):
        if type(allocation_policy) is not FrozenPriorityPolicy:
            raise TypeError('immutable frozen allocation policy required')
        instance = super().open(path, expected_prediction_sha256=expected_prediction_sha256,
                                expected_database_sha256=expected_database_sha256)
        instance.allocation_policy = allocation_policy
        try:
            instance._check_policy()
        except BaseException:
            instance.close()
            raise
        return instance

    def _check_policy(self):
        row = self.db.execute("SELECT v FROM meta WHERE k='allocation_policy_signature'").fetchone()
        if (type(self.allocation_policy) is not FrozenPriorityPolicy or row is None
                or json.loads(row[0]) != self.allocation_policy.signature):
            raise ValueError('frozen allocation policy binding changed')

    def step(self, *args, **kwargs):
        self._check_policy()  # Also reject policy changes on duplicate receipts.
        return super().step(*args, **kwargs)

    def _select_allocation(self, candidates, **state):
        records = self._priority_candidates(candidates, state)
        scores = self.allocation_policy.scores([r['features'] for r in records])
        selected = min(range(len(records)), key=lambda i: (-scores[i], records[i]['component']))
        return records[selected]['component'], dict(priority_selection=dict(
            recipe=RECIPE, candidate_count=len(records),
            candidates=[dict(component=r['component'], features=r['features'], priority_score=float(score))
                        for r, score in zip(records, scores)], priority_is_bound=False))

    def _allocation_summary(self):
        return dict(allocation_policy='learned_signed_one_operation_model_bound_progress',
            allocation_policy_signature=self.allocation_policy.signature,
            priority_feature_rows=self.priority_rows, priority_row_cap=self.MAX_PRIORITY_ROWS,
            offline_counterfactual_probes=False, priority_is_bound=False)

    def _audit_properties(self):
        return dict(super()._audit_properties(), kind=self.SCHEMA, learned_allocation_policy=True)


class AllocationTeacherTracker(_PriorityStateMixin, PersistentComponentTracker):
    """Offline helper. Cache runner additionally enforces actual split=train.

Low-level raw-input construction (used by tests) cannot certify a data split.
Teacher trajectories use the unmodified deterministic scheduler. Every eligible
candidate gets a counterfactual label, not only the selected component.
"""
    SCHEMA = 'persistent_allocation_teacher_train_only_v1'
    MAX_PROBE_CACHE_HANDLES = 65536

    def _component_inference(self, *args, **kwargs):
        # Only the selected component changes inside this decision's allocation
        # loop. A probe of another component may be reused if its exact search
        # state AND effective operation/capacity are unchanged. Never reuse
        # across arrival events, revisions, merges or transaction retries.
        self._probe_cache = OrderedDict()
        self._probe_cache_handles = self._probe_cache_peak_handles = 0
        self.probe_cache_hits = self.probe_executions = 0
        try:
            return super()._component_inference(*args, **kwargs)
        finally:
            self._probe_cache.clear()
            self._probe_cache_handles = 0

    def _probe_key(self, record, state):
        c, operation = record['component'], record['operation']
        k = self.kernels[c]
        # _refine clips global allowances to these local limits. A decrease in
        # global capacity matters only if it changes the effective local cap.
        prefix_cap = min(k.config.max_prefix_nodes,
            k.prefix_count+self.config.max_total_prefix_nodes-state['prefix_count'])
        frontier_cap = min(k.config.state.max_frontier,
            len(k.frontier)+self.config.max_total_frontier-state['frontier_count'])
        return (k.n, k.revision, k.prefix_count, frozenset(k.active), frozenset(k.frontier),
            operation['base'], operation['kind'], operation['requested_steps'],
            prefix_cap, frontier_cap, state['decision_us'], state['scopes'][c],
            state['contexts'][c]['fallback'], state['weights'][c], record['model_bound_before'])

    def _probe(self, record, state):
        c, operation = record['component'], record['operation']
        k = self.kernels[c]
        # Keep the original request cap and labels even when cached: this is
        # an implementation optimization, not extra teacher sampling budget.
        self.teacher_steps += operation['requested_steps']
        if self.teacher_steps > self.MAX_TEACHER_STEPS:
            raise ValueError('offline teacher work capacity exhausted; event not committed')
        key = self._probe_key(record, state)
        old = self._probe_cache.pop(c, None)
        if old is not None:
            self._probe_cache_handles -= old[2]
            if old[0] == key:
                self._probe_cache[c] = old
                self._probe_cache_handles += old[2]
                self.probe_cache_hits += 1
                return dict(old[1])
        result = self._probe_uncached(record, state)
        size = len(k.active)+len(k.frontier)
        if size <= self.MAX_PROBE_CACHE_HANDLES:
            while self._probe_cache and self._probe_cache_handles+size > self.MAX_PROBE_CACHE_HANDLES:
                _, discarded = self._probe_cache.popitem(last=False)
                self._probe_cache_handles -= discarded[2]
            self._probe_cache[c] = (key, dict(result), size)
            self._probe_cache_handles += size
            self._probe_cache_peak_handles = max(self._probe_cache_peak_handles, self._probe_cache_handles)
        return result

    def _probe_uncached(self, record, state):
        c, operation = record['component'], record['operation']
        k = self.kernels[c]
        self.probe_executions += 1
        # Search helpers mutate these fields plus prefix/weight/discovery SQL.
        # Restore the shared LRU too: rolled-back rowids may later be reused.
        saved = k.prefix_count, set(k.active), set(k.frontier), k.state_updates
        cache = tuple(self.shared_cache.items())
        self.db.execute('SAVEPOINT allocation_probe')
        try:
            done, limited = self._execute_allocation_work(k, operation, state['decision_us'],
                state['prefix_count'], state['frontier_count'])
            after = decision_bound(k, k._mass(), state['scopes'][c], state['contexts'][c]['fallback'])['risk_bound']
            target = state['weights'][c]*(record['model_bound_before']-after)/max(1, done)
            return dict(target=target, model_bound_after=after, charged_steps=done, resource_limited=limited)
        finally:
            self.db.execute('ROLLBACK TO allocation_probe')
            self.db.execute('RELEASE allocation_probe')
            k.prefix_count, k.active, k.frontier, k.state_updates = saved
            self.shared_cache.clear()
            for key, value in cache:
                self.shared_cache[key] = value

    def _select_allocation(self, candidates, **state):
        records = self._priority_candidates(candidates, state)
        probes = [dict(r, **self._probe(r, state)) for r in records]
        component, _ = super()._select_allocation(candidates, **state)
        return component, dict(allocation_training=dict(feature_recipe=RECIPE, target_recipe=TARGET,
            labels_are_model_not_true_risk=True, future_or_gt_inputs=False,
            behavior='weighted_eta_deterministic', candidates=probes))

    def _allocation_summary(self):
        return dict(super()._allocation_summary(), offline_counterfactual_probes=True,
            priority_feature_rows=self.priority_rows, priority_row_cap=self.MAX_PRIORITY_ROWS,
            teacher_requested_search_steps=self.teacher_steps, teacher_step_cap=self.MAX_TEACHER_STEPS,
            teacher_probe_executions=self.probe_executions, teacher_probe_cache_hits=self.probe_cache_hits,
            teacher_probe_cache_peak_handles=self._probe_cache_peak_handles,
            teacher_probe_cache_handle_cap=self.MAX_PROBE_CACHE_HANDLES,
            teacher_cost_excluded_from_deployment_search_budget=True,
            teacher_latency_is_not_deployment_latency=True)

    def _audit_properties(self):
        return dict(super()._audit_properties(), kind=self.SCHEMA,
                    training_trace_only=True, low_level_input_split_not_certified=True)
