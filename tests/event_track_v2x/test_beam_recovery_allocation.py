from dataclasses import replace
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.allocation_policy import FEATURES, FrozenPriorityPolicy
from transvision.models.event_track_v2x.beam_recovery_allocation import (
    BeamRecoveryTeacherTracker, LearnedBeamRecoveryTracker,
)
from transvision.models.event_track_v2x.beam_recovery_tracking import BeamRecoveryConfig, BeamRecoveryTracker
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from test_forest_tracking import observation
from test_learned_component_allocation import policy, scene
from test_persistent_component_tracking import step, joint_action
from test_persistent_forest import brute


def config(budget=8):
    return BeamRecoveryConfig(state=ForestTrackingConfig(active_limit=2), recovery_budget=budget)


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('budget', [0, 1, 4, 100])
def test_learned_recovery_order_preserves_complete_model_bound(tmp_path, seed, budget):
    raw, rows = scene(seed)
    t = LearnedBeamRecoveryTracker(tmp_path/'learned.db', sequence_id='0003', config=config(budget),
                                   allocation_policy=policy(seed))
    result = step(t, raw, rows)
    factors = ForestFactors(tuple(o.node for o in raw), rows)
    probabilities, roots, z = brute(factors)
    chosen = joint_action(t, result)
    def risk(h):
        return math.fsum(p*sum(a != b for a, b in zip(roots[h], roots[g]))/len(raw)
                         for g, p in probabilities.items())
    assert risk(chosen)-min(map(risk, probabilities)) <= result.audit['model_regret_upper']+1e-12
    assert math.log(z) <= result.audit['log_partition_upper']+1e-12
    assert result.audit['learned_allocation_policy'] and not result.audit['priority_is_bound']
    assert result.audit['priority_changes_only_extra_recovery_order']
    assert result.audit['recovery_search_steps'] <= budget
    for component in result.audit['components']:
        k = t.kernels[component['component']]
        local = ForestFactors(tuple(raw[i].node for i in t.store.members(component['component'])),
                              tuple(k._row(i) for i in range(k.n)))
        probs, labels, partition = brute(local)
        def signature(h):
            return tuple(k._prefix(k.ancestor(h, i+1)).root for i in range(k._prefix(h).depth))
        active = {signature(h) for h in k.active}
        frontier = [signature(h) for h in k.frontier]
        assert all(g in active or any(g[:len(p)] == p for p in frontier) for g in labels.values())
        assert sum(p for h, p in probs.items() if labels[h] not in active) <= component['eta_upper']+1e-12
        for item in component['active']:
            mass = partition*sum(p for h, p in probs.items() if labels[h] == signature(item['handle']))
            assert math.exp(item['log_weight']) == pytest.approx(mass)
    t.close()


@pytest.mark.parametrize('seed', range(3))
def test_teacher_is_exact_same_recovery_behavior_and_probe_rolls_back(tmp_path, seed):
    class InspectTeacher(BeamRecoveryTeacherTracker):
        def _probe(self, record, state):
            before = tuple(self.db.iterdump()), tuple(self.shared_cache.items())
            answer = super()._probe(record, state)
            assert tuple(self.db.iterdump()) == before[0]
            assert tuple(self.shared_cache.items()) == before[1]
            return answer
    ordinary = BeamRecoveryTracker(tmp_path/'normal.db', sequence_id='0003', config=config())
    teacher = InspectTeacher(tmp_path/'teacher.db', sequence_id='0003', config=config())
    raw, rows = scene(seed)
    first = [step(t, raw, rows) for t in (ordinary, teacher)]
    bridge = [observation('bridge', source=0, frame='later', state_us=1_200_000)]
    second = [step(t, bridge, [[(-1, -4.), (2, 0.), (3, -.5)]], time=1_300_000, event='merge')
              for t in (ordinary, teacher)]
    for normal, taught in (first, second):
        assert normal.prediction_json == taught.prediction_json
        for key in ('components','factor_rows_sha256','recovery_search_steps','state_updates',
                    'model_regret_upper','candidate_evaluations','beam_expansions'):
            assert normal.audit[key] == taught.audit[key]
        for original, event in zip(normal.audit['recovery_allocation_trace'], taught.audit['recovery_allocation_trace'], strict=True):
            assert {k:v for k,v in event.items() if k != 'allocation_training'} == original
            for record in event['allocation_training']['candidates']:
                x = record['features']
                assert x[FEATURES.index('remaining_budget_fraction')] <= 1
                assert record['target'] == pytest.approx(x[0]*(record['model_bound_before']-
                    record['model_bound_after'])/max(1,record['charged_steps']))
    assert teacher.teacher_steps > 0 and not teacher._probe_cache
    assert step(teacher, raw, rows).prediction_json == first[1].prediction_json
    for t in (ordinary, teacher): t.close()


def test_features_use_recovery_budget_and_selected_operation(tmp_path):
    t = BeamRecoveryTeacherTracker(tmp_path/'teacher.db', sequence_id='0003', config=config(7))
    raw, rows = scene()
    result = step(t, raw, rows)
    first = result.audit['recovery_allocation_trace'][0]['allocation_training']['candidates']
    assert all(r['features'][FEATURES.index('remaining_budget_fraction')] == 1 for r in first)
    assert any(r['operation']['kind'] == 'coverage_admitted_complete_proposal' for r in first)
    assert all(r['charged_steps'] <= r['operation']['requested_steps'] for r in first)
    t.close()


def test_policy_weights_actually_change_extra_search_order_only(tmp_path):
    raw, rows = scene()
    raw += (observation('extra',source=1,frame='extra',state_us=1_000_010),)
    rows += (((-1,-2.),(0,0.)),)
    results = []
    for sign in (1,-1):
        weights = np.zeros((1,len(FEATURES))); weights[0,0] = sign
        p = FrozenPriorityPolicy([weights,np.zeros(1),np.ones((1,1)),np.zeros(1)])
        t = LearnedBeamRecoveryTracker(tmp_path/f'policy-{sign}.db',sequence_id='0003',
            config=config(1),allocation_policy=p)
        results.append(step(t,raw,rows)); t.close()
    assert results[0].audit['recovery_allocation_trace'][0]['component'] != results[1].audit['recovery_allocation_trace'][0]['component']
    assert results[0].audit['factor_rows_sha256'] == results[1].audit['factor_rows_sha256']
    for a,b in zip(results[0].audit['components'],results[1].audit['components'],strict=True):
        for key in ('backbone_active','backbone_output_handle','backbone_retained_log_mass','pruning'):
            assert a[key] == b[key]


def test_reopen_binds_policy_and_preserves_past_output(tmp_path):
    p, path = policy(), tmp_path/'learned.db'
    t = LearnedBeamRecoveryTracker(path, sequence_id='0003', config=config(), allocation_policy=p)
    raw, rows = scene()
    first = step(t, raw, rows); sha = t.close()
    with pytest.raises(ValueError, match='policy binding'):
        LearnedBeamRecoveryTracker.open(path, expected_prediction_sha256=first.prediction['commit_sha256'],
            expected_database_sha256=sha, allocation_policy=policy(1))
    t = LearnedBeamRecoveryTracker.open(path, expected_prediction_sha256=first.prediction['commit_sha256'],
        expected_database_sha256=sha, allocation_policy=p)
    assert step(t, raw, rows).audit_json == first.audit_json
    step(t, time=1_200_000, event='more')
    assert step(t, raw, rows).prediction_json == first.prediction_json
    t.allocation_policy = policy(1)
    with pytest.raises(ValueError, match='policy binding'): step(t, raw, rows)
    t.close()


@pytest.mark.parametrize('kind', ['priority', 'teacher', 'probe'])
def test_caps_or_probe_failure_roll_back_entire_beam_event(tmp_path, monkeypatch, kind):
    cls = LearnedBeamRecoveryTracker if kind == 'priority' else BeamRecoveryTeacherTracker
    t = cls(tmp_path/'failure.db', sequence_id='0003', config=config(),
            **(dict(allocation_policy=policy()) if kind == 'priority' else {}))
    raw, rows = scene(); before = tuple(t.db.iterdump())
    execute = t._execute_recovery
    if kind == 'priority': t.MAX_PRIORITY_ROWS = 1
    elif kind == 'teacher': t.MAX_TEACHER_STEPS = 1
    else:
        def fail(*args):
            execute(*args)
            raise RuntimeError('injected after recovery SQL writes')
        monkeypatch.setattr(t, '_execute_recovery', fail)
    with pytest.raises((ValueError, RuntimeError)): step(t, raw, rows)
    assert tuple(t.db.iterdump()) == before and t.n == 0
    t.MAX_PRIORITY_ROWS, t.MAX_TEACHER_STEPS = 262144, 1048576
    monkeypatch.setattr(t, '_execute_recovery', execute)
    result = step(t, raw, rows)
    assert result.audit['recovery_search_steps'] > 0
    t.close()


@pytest.mark.parametrize('cls', [LearnedBeamRecoveryTracker, BeamRecoveryTeacherTracker])
def test_disabled_recovery_rejects_priority_or_teacher_before_file_creation(tmp_path, cls):
    path = tmp_path/'disabled.db'
    with pytest.raises(ValueError, match='enabled beam recovery'):
        cls(path, sequence_id='0003', config=replace(config(),enable_recovery=False),
            **(dict(allocation_policy=policy()) if cls is LearnedBeamRecoveryTracker else {}))
    assert not path.exists()
