import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.allocation_policy import FEATURES, FrozenPriorityPolicy
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.learned_component_allocation import AllocationTeacherTracker, LearnedComponentTracker
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig, PersistentComponentTracker
from test_forest_tracking import observation
from test_persistent_forest import brute
from test_persistent_component_tracking import step, joint_action


def policy(seed=0):
    rng = np.random.default_rng(seed)
    return FrozenPriorityPolicy([rng.normal(size=(8, len(FEATURES))), rng.normal(size=8),
                                 rng.normal(size=(1, 8)), rng.normal(size=1)])


def scene(seed=0):
    rng = np.random.default_rng(seed)
    raw = tuple(observation(str(i), source=i % 2, frame=str(i), state_us=1_000_000+i) for i in range(6))
    support = ((-1,), (-1,), (-1, 0), (-1, 1), (-1, 0, 2), (-1, 1, 3))
    rows = tuple(tuple((p, float(rng.normal())) for p in parents) for parents in support)
    return raw, rows


def config(budget=8, **kwargs):
    return PersistentComponentConfig(state=ForestTrackingConfig(active_limit=2, expansion_budget=budget), **kwargs)


def test_frozen_priority_numpy_forward_is_immutable_and_validated():
    p = policy()
    x = np.zeros((3, len(FEATURES)))
    a, b, c, d = p.weights
    assert np.array_equal(p.scores(x), np.tanh(np.tanh(x @ a.T+b) @ c.T+d)[:, 0])
    with pytest.raises(ValueError):
        a.setflags(write=True)
    with pytest.raises(ValueError):
        p.scores([[math.nan]*len(FEATURES)])
    assert p.signature == policy().signature != policy(1).signature


def test_external_numpy_view_metadata_cannot_change_frozen_policy_behavior():
    p = policy()
    x = np.zeros((2, len(FEATURES)))
    expected, signature = p.scores(x).copy(), p.signature
    exposed = p.weights
    exposed[3].dtype = np.int64
    exposed[0].shape = (exposed[0].size,)
    assert np.array_equal(p.scores(x), expected)
    assert p.signature == signature and p.weights[3].dtype == np.dtype('<f8')


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('budget', [0, 1, 4, 100])
def test_learned_order_preserves_full_model_mass_and_actual_action_regret(tmp_path, seed, budget):
    raw, rows = scene(seed)
    t = LearnedComponentTracker(tmp_path/'learned.db', sequence_id='0003', config=config(budget), allocation_policy=policy(seed))
    result = step(t, raw, rows)
    factors = ForestFactors(tuple(o.node for o in raw), rows)
    probabilities, roots, z = brute(factors)
    chosen = joint_action(t, result)
    def risk(h):
        return math.fsum(p*sum(a != b for a, b in zip(roots[h], roots[g]))/len(raw) for g, p in probabilities.items())
    assert risk(chosen)-min(risk(h) for h in probabilities) <= result.audit['model_regret_upper']+1e-12
    assert result.audit['learned_allocation_policy'] and not result.audit['priority_is_bound']
    assert result.audit['search_steps'] <= budget
    for component in result.audit['components']:
        members = t.store.members(component['component'])
        selected = {tuple(members[t.kernels[component['component']]._prefix(
            t.kernels[component['component']].ancestor(a['handle'], i+1)).root] for i in range(len(members)))
            for a in component['active']}
        omitted = math.fsum(p for h, p in probabilities.items() if tuple(roots[h][i] for i in members) not in selected)
        assert omitted <= component['eta_upper']+1e-12
    t.close()


@pytest.mark.parametrize('seed', range(3))
def test_teacher_probes_restore_database_and_caches_and_keep_behavior_identical(tmp_path, seed):
    class InspectTeacher(AllocationTeacherTracker):
        def _probe(self, record, state):
            before = tuple(self.db.iterdump()), tuple(self.shared_cache.items())
            answer = super()._probe(record, state)
            assert tuple(self.db.iterdump()) == before[0]
            assert tuple(self.shared_cache.items()) == before[1]
            return answer
    normal = PersistentComponentTracker(tmp_path/'normal.db', sequence_id='0003', config=config())
    teacher = InspectTeacher(tmp_path/'teacher.db', sequence_id='0003', config=config())
    raw, rows = scene(seed)
    initial = [step(t, raw, rows) for t in (normal, teacher)]
    new = [observation('bridge', source=0, frame='later', state_us=1_200_000)]
    joined = [step(t, new, [[(-1, -4.), (2, 0.), (3, -.5)]], time=1_300_000, event='merge') for t in (normal, teacher)]
    for left, right in (initial, joined):
        assert left.prediction_json == right.prediction_json
        assert left.audit['factor_rows_sha256'] == right.audit['factor_rows_sha256']
        assert left.audit['model_regret_upper'] == right.audit['model_regret_upper']
        assert left.audit['search_steps'] == right.audit['search_steps']
        for row in right.audit['allocation_trace']:
            options = row['allocation_training']['candidates']
            assert len({r['component'] for r in options}) == len(options)
            for r in options:
                weight = r['features'][FEATURES.index('loss_weight')]
                assert r['target'] == pytest.approx(weight*(r['model_bound_before']-r['model_bound_after'])/max(1, r['charged_steps']))
    assert teacher.teacher_steps > 0
    for t, first in zip((normal, teacher), initial):
        assert step(t, raw, rows).prediction_json == first.prediction_json
        t.close()


def test_reopen_requires_same_policy_and_old_receipts_are_immutable(tmp_path):
    p = policy()
    path = tmp_path/'learned.db'
    t = LearnedComponentTracker(path, sequence_id='0003', config=config(), allocation_policy=p)
    raw, rows = scene()
    first = step(t, raw, rows)
    sha = t.close()
    with pytest.raises(ValueError, match='policy binding'):
        LearnedComponentTracker.open(path, expected_prediction_sha256=first.prediction['commit_sha256'],
                                    expected_database_sha256=sha, allocation_policy=policy(1))
    t = LearnedComponentTracker.open(path, expected_prediction_sha256=first.prediction['commit_sha256'],
                                    expected_database_sha256=sha, allocation_policy=p)
    assert step(t, raw, rows).audit_json == first.audit_json
    result = step(t, time=1_200_000, event='more-work')
    assert result.audit['allocation_policy_signature'] == p.signature
    t.allocation_policy = policy(1)
    with pytest.raises(ValueError, match='policy binding'):
        step(t, raw, rows)
    t.close()


@pytest.mark.parametrize('kind', ['priority', 'teacher'])
def test_capacity_or_probe_exception_cannot_commit_partial_event(tmp_path, kind):
    if kind == 'priority':
        t = LearnedComponentTracker(tmp_path/'failure.db', sequence_id='0003', config=config(), allocation_policy=policy())
        t.MAX_PRIORITY_ROWS = 1
    else:
        t = AllocationTeacherTracker(tmp_path/'failure.db', sequence_id='0003', config=config())
        t.MAX_TEACHER_STEPS = 1
    before = tuple(t.db.iterdump())
    raw, rows = scene()
    with pytest.raises(ValueError, match='capacity'):
        step(t, raw, rows)
    assert tuple(t.db.iterdump()) == before and t.n == 0
    t.close()


def test_exception_inside_counterfactual_restores_state_and_allows_exact_retry(tmp_path, monkeypatch):
    t = AllocationTeacherTracker(tmp_path/'retry.db', sequence_id='0003', config=config())
    raw, rows = scene()
    before = tuple(t.db.iterdump())
    original = t._execute_allocation_work
    def fail_after_work(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('counterfactual failure after database writes')
    monkeypatch.setattr(t, '_execute_allocation_work', fail_after_work)
    with pytest.raises(RuntimeError, match='after database writes'):
        step(t, raw, rows)
    assert tuple(t.db.iterdump()) == before and t.n == 0
    assert not t._probe_cache and t._probe_cache_handles == 0
    monkeypatch.setattr(t, '_execute_allocation_work', original)
    result = step(t, raw, rows)
    normal = PersistentComponentTracker(tmp_path/'reference.db', sequence_id='0003', config=config())
    assert result.prediction_json == step(normal, raw, rows).prediction_json
    normal.close(); t.close()


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('cache_cap', [3, 65536])
def test_cached_teacher_labels_predictions_and_search_equal_uncached_teacher(tmp_path, seed, cache_cap):
    cached = AllocationTeacherTracker(tmp_path/'cached.db', sequence_id='0003', config=config(20))
    reference = AllocationTeacherTracker(tmp_path/'uncached.db', sequence_id='0003', config=config(20))
    cached.MAX_PROBE_CACHE_HANDLES = cache_cap
    reference.MAX_PROBE_CACHE_HANDLES = 0
    raw, rows = scene(seed)
    first = [step(t, raw, rows) for t in (cached, reference)]
    bridge = [observation('bridge', source=0, frame='later', state_us=1_200_000)]
    second = [step(t, bridge, [[(-1,-4.),(2,0.),(3,-.5)]], time=1_300_000, event='merge')
              for t in (cached, reference)]
    for actual, expected in (first, second):
        assert actual.prediction_json == expected.prediction_json
        for field in ('allocation_trace','factor_rows_sha256','model_regret_upper','search_steps',
                      'teacher_requested_search_steps','components'):
            assert actual.audit[field] == expected.audit[field]
        assert actual.audit['teacher_probe_cache_peak_handles'] <= cache_cap
        assert expected.audit['teacher_probe_cache_hits'] == 0
    if cache_cap == 65536:
        assert first[0].audit['teacher_probe_cache_hits'] > 0
        assert first[0].audit['teacher_probe_executions'] < first[1].audit['teacher_probe_executions']
    for t in (cached, reference):
        assert not t._probe_cache and t._probe_cache_handles == 0
        t.close()


def test_probe_cache_invalidates_effective_cap_operation_and_decision_context(tmp_path):
    class InspectTeacher(AllocationTeacherTracker):
        inspected = False
        def _select_allocation(self, candidates, **state):
            if not self.inspected:
                self.inspected = True
                record = self._priority_candidates(candidates, state)[0]
                c = record['component']; k = self.kernels[c]
                expected = self._probe(record, state)
                executions = self.probe_executions
                repeated = self._probe(record, state)
                assert repeated == expected and self.probe_executions == executions
                repeated['target'] = 999  # A caller cannot mutate stored labels.
                assert self._probe(record, state) == expected
                # Unused remaining budget is a feature, not part of this fixed
                # operation's label. Effective caps/operation/context ARE.
                assert self._probe_key(record,dict(state,remaining=state['remaining']+1)) == self._probe_key(record,state)
                variants = [
                    dict(state, prefix_count=k.prefix_count+self.config.max_total_prefix_nodes-k.config.max_prefix_nodes+1),
                    dict(state, frontier_count=len(k.frontier)+self.config.max_total_frontier-k.config.state.max_frontier+1),
                    dict(state, decision_us=state['decision_us']+1),
                    dict(state, scopes=state['scopes'] | {c: ()}),
                    dict(state, weights=state['weights'] | {c: state['weights'][c]/2}),
                ]
                original_key = self._probe_key(record, state)
                for alternate in variants:
                    assert self._probe_key(record, alternate) != original_key
                    before = self.probe_executions
                    result = self._probe(record, alternate)
                    assert self.probe_executions == before+1
                    assert result == self._probe_uncached(record, alternate)
                changed = dict(record, operation=dict(record['operation'], requested_steps=record['operation']['requested_steps']+1))
                assert self._probe_key(changed,state) != original_key
            return super()._select_allocation(candidates, **state)
    t = InspectTeacher(tmp_path/'invalidate.db', sequence_id='0003', config=config(20))
    raw, rows = scene()
    step(t,raw,rows)
    assert t.inspected
    t.close()


@pytest.mark.parametrize('teacher', [False, True])
def test_unarrived_inputs_cannot_generate_training_labels_or_learned_features(tmp_path, teacher):
    cls = AllocationTeacherTracker if teacher else LearnedComponentTracker
    t = cls(tmp_path/'future.db', sequence_id='0003', config=config(),
            **({} if teacher else dict(allocation_policy=policy())))
    before = tuple(t.db.iterdump())
    raw = [observation('not-arrived', arrival_us=2_000_000)]
    with pytest.raises(ValueError, match='future'):
        step(t, raw, [[(-1, 0.)]])
    assert tuple(t.db.iterdump()) == before and not hasattr(t, 'priority_rows')
    t.close()
