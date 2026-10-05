"""Causal priorities for the separately trained recovery-off ablation.

The feature vector has the existing dimensions, but its decision terms and
teacher targets concern only the irreversibly retained historical support.
Source/configuration bindings prevent transfer of unrestricted checkpoints.
No teacher probe may materialize an unrestricted action or affect a commit.
"""
import math

from .allocation_policy import FEATURES
from .exclusive_completion_tracking import ExclusiveCompletionLearned, ExclusiveCompletionTeacher
from .identity_forest import ForestFactors, digest
from .paper_decision import conditional_action, decode_persistent
from .recovery_off_tracking import RestrictedFactors, _RecoveryOffRegions

SCOPE = 'conditional_on_irreversibly_pruned_support'


def decision_bound(kernel, mass, scope, fallback):
    """Read-only, zero-expansion conditional decision, matching teacher costs."""
    active, weights, retained, _, eta = mass

    def path(handle):
        return tuple(kernel._prefix(kernel.ancestor(handle, i + 1)).choice for i in range(kernel.n))

    fallback_roots = tuple(kernel._prefix(kernel.ancestor(fallback, i + 1)).root
                           for i in range(kernel.n))
    if not kernel.restriction.allows(fallback_roots):
        raise ValueError('priority fallback violates committed historical support')
    if not active or kernel.n > kernel.config.max_decision_nodes:
        # The original nonmaterializing decoder does no action search on these
        # branches; keep its conservative no-active/capacity behavior.
        _, result = decode_persistent(kernel, active, weights, retained, eta, scope,
                                      fallback, materialize=False)
    else:
        factors = RestrictedFactors(ForestFactors(
            tuple(kernel._observation(i).node for i in range(kernel.n)),
            tuple(kernel._row(i) for i in range(kernel.n))), kernel.restriction)
        _, result = conditional_action(factors, tuple(path(h) for h in active), weights,
            scope, budget=0, frontier_limit=kernel.config.state.paper_action_frontier)
        result = dict(result, risk_bound=min(1., eta + result['optimization_gap']),
                      empty_loss_scope=not scope, action_materialization_prefixes=0,
                      action_materialization_steps=0)
    return dict(result, action_space='legal_extensions_of_previous_retained_and_output_classes',
                support_sha256=digest(kernel.restriction.payload()), risk_scope=SCOPE,
                original_model_regret_upper=1., original_model_risk_certified=False)


def priority_features(kernel, mass, scope, fallback, weight, operation, remaining, budget):
    decision = decision_bound(kernel, mass, scope, fallback)
    active, logs, retained, _, eta = mass
    probabilities = [math.exp(w - retained) for w in logs]
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0) / math.log(max(2, len(active)))
    depth = [kernel._prefix(h).depth / kernel.n for h in kernel.frontier] if kernel.n else []
    values = (weight, eta, decision['conditional_risk'] or 0., decision['optimization_gap'] or 0.,
        decision['risk_bound'], entropy, min(50., logs[0] - logs[1]) / 50. if len(logs) > 1 else 1.,
        math.log1p(kernel.n), math.log1p(len(scope)), math.log1p(len(active)), math.log1p(len(depth)),
        min(depth, default=1.), max(depth, default=1.), remaining / max(1, budget),
        float(operation['base'] is not None), math.log1p(operation['requested_steps']),
        math.log1p(kernel.prefix_count), weight * eta)
    if len(values) != len(FEATURES) or not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite recovery-off causal priority features')
    return tuple(values), decision


class _RestrictedPriority:
    def _priority_candidates(self, candidates, state):
        self.priority_rows += len(candidates)
        if self.priority_rows > self.MAX_PRIORITY_ROWS:
            raise ValueError('priority evaluation capacity exhausted; event not committed')
        records = []
        for component in sorted(candidates):
            kernel = self.kernels[component]
            operation = self._next_allocation_work(kernel, state['proposals'][component],
                state['remaining'], state['prefix_count'], state['frontier_count'])
            features, decision = priority_features(kernel, state['masses'][component],
                state['scopes'][component], state['contexts'][component]['fallback'],
                state['weights'][component], operation, state['remaining'], self.config.state.expansion_budget)
            records.append(dict(component=component, features=features, operation=operation,
                model_bound_before=decision['risk_bound'], model_bound_scope=SCOPE,
                historical_support_sha256=decision['support_sha256']))
        return records

    def _allocation_summary(self):
        return dict(super()._allocation_summary(), priority_decision_support_scope=SCOPE,
                    priority_action_materialization=False,
                    unrestricted_priority_checkpoint_transfer=False)


class RecoveryOffTeacher(_RestrictedPriority, _RecoveryOffRegions, ExclusiveCompletionTeacher):
    SCHEMA = 'persistent_exclusive_recovery_off_allocation_teacher_v1'

    def _probe_key(self, record, state):
        return super()._probe_key(record, state) + (record['historical_support_sha256'],)

    def _probe_uncached(self, record, state):
        component, operation = record['component'], record['operation']
        kernel = self.kernels[component]
        self.probe_executions += 1
        saved = kernel.prefix_count, set(kernel.active), set(kernel.frontier), kernel.state_updates
        cache = tuple(self.shared_cache.items())
        self.db.execute('SAVEPOINT allocation_probe')
        try:
            done, limited = self._execute_allocation_work(kernel, operation, state['decision_us'],
                state['prefix_count'], state['frontier_count'])
            after = decision_bound(kernel, kernel._mass(), state['scopes'][component],
                                   state['contexts'][component]['fallback'])['risk_bound']
            target = state['weights'][component] * (record['model_bound_before'] - after) / max(1, done)
            return dict(target=target, model_bound_after=after, charged_steps=done,
                        resource_limited=limited, model_bound_scope=SCOPE)
        finally:
            self.db.execute('ROLLBACK TO allocation_probe')
            self.db.execute('RELEASE allocation_probe')
            kernel.prefix_count, kernel.active, kernel.frontier, kernel.state_updates = saved
            self.shared_cache.clear()
            for key, value in cache:
                self.shared_cache[key] = value


class RecoveryOffLearned(_RestrictedPriority, _RecoveryOffRegions, ExclusiveCompletionLearned):
    SCHEMA = 'persistent_exclusive_recovery_off_learned_allocation_v1'
