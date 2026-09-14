from dataclasses import replace
import itertools
import json
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig, replay_forest_states
from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig, PersistentComponentTracker
from test_forest_tracking import observation
from test_persistent_forest import brute


def tracker(path, budget=100, **limits):
    return PersistentComponentTracker(path, sequence_id='0003', config=PersistentComponentConfig(
        state=ForestTrackingConfig(active_limit=2, expansion_budget=budget, max_model_regret=.05), **limits))


def step(t, raw=(), rows=(), time=1_100_000, event='first', **kwargs):
    return t.step(raw, rows, frame_id=event, event_id=event, reference_us=time, decision_us=time, **kwargs)


def joint_action(t, result):
    action = [-1]*t.n
    for summary in result.audit['components']:
        component = summary['component']
        members = t.store.members(component)
        local = t.kernels[component].parents(summary['output_handle'])
        for i, parent in zip(members, local):
            action[i] = -1 if parent < 0 else members[parent]
    return tuple(action)


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('budget', [0, 1, 3, 100])
def test_mass_identity_class_sums_and_actual_global_action_regret_against_full_parent_enumeration(tmp_path, seed, budget):
    t = tracker(tmp_path/'components.db', budget)
    rng = np.random.default_rng(seed)
    raw = tuple(observation(str(i), source=i % 2, frame=str(i), state_us=1_000_000+i) for i in range(5))
    support = ((-1,), (-1,), (-1, 0), (-1, 0, 2), (-1, 1))
    rows = tuple(tuple((p, float(rng.normal())) for p in row) for row in support)
    result = step(t, raw, rows)
    factors = ForestFactors(tuple(o.node for o in raw), rows)
    probability, labels, partition = brute(factors)
    action = joint_action(t, result)
    scope = result.audit['decision_indices']
    def risk(a):
        return sum(prob*sum(labels[a][i] != roots[i] for i in scope)/len(scope)
                   for history, prob in probability.items() for roots in (labels[history],))
    regret = risk(action)-min(risk(h) for h in probability)
    assert regret <= result.audit['model_regret_upper']+1e-12
    assert result.audit['expansions'] <= budget
    assert result.audit['search_steps'] == result.audit['expansions']+result.audit['proposal_steps'] <= budget
    assert math.log(partition) <= result.audit['log_partition_upper']+1e-12
    assert len(result.audit['components']) == 2
    for summary in result.audit['components']:
        members = t.store.members(summary['component'])
        retained_roots = []
        for active in summary['active']:
            kernel = t.kernels[summary['component']]
            roots = tuple(kernel._prefix(kernel.ancestor(active['handle'], i+1)).root for i in range(kernel.n))
            global_roots = tuple(members[i] for i in roots)
            mass = sum(p for h, p in probability.items() if tuple(labels[h][i] for i in members) == global_roots)
            local_factors = ForestFactors(tuple(raw[i].node for i in members), tuple(kernel._row(i) for i in range(kernel.n)))
            local_probability, local_roots, local_z = brute(local_factors)
            class_weight = local_z*sum(p for h, p in local_probability.items() if local_roots[h] == roots)
            assert math.exp(active['log_weight']) == pytest.approx(class_weight)
            retained_roots.append((global_roots, mass))
        assert 1-sum(p for _, p in retained_roots) <= summary['eta_upper']+1e-12
        kernel = t.kernels[summary['component']]
        local_factors = ForestFactors(tuple(raw[i].node for i in members), tuple(kernel._row(i) for i in range(kernel.n)))
        local_probability, local_roots, _ = brute(local_factors)
        frontier = [kernel.parents(p['handle']) for p in summary['frontier']]
        active_parents = {kernel.parents(a['handle']) for a in summary['active']}
        for history, roots in local_roots.items():
            representative = tuple(-1 if roots[i] == i else min(p for p, _ in local_factors.rows[i]
                if p >= 0 and roots[p] == roots[i]) for i in range(kernel.n))
            covers = sum(representative[:len(p)] == p for p in frontier)
            assert covers <= 1  # Prefix regions remain disjoint, even with seeded active classes.
            assert representative in active_parents or covers == 1
    expected, _ = replay_forest_states('0003', raw, factors, action, 1_100_000, t.config.state)
    expected = sorted([{k: p[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for p in expected], key=lambda p: p['track_id'])
    assert canonical(result.prediction['predictions']) == canonical(expected)
    t.close()


def test_rescoring_recovers_previously_unexpanded_output_without_rewriting_history(tmp_path):
    config = PersistentComponentConfig(state=ForestTrackingConfig(active_limit=1, expansion_budget=1, max_model_regret=.05))
    t = PersistentComponentTracker(tmp_path/'recover.db', sequence_id='0003', config=config)
    raw = (observation('a'), observation('b', source=1))
    first = step(t, raw, [((-1, 0.),), ((-1, 0.), (0, -8.))])
    assert len(first.prediction['predictions']) == 2
    second = step(t, time=1_200_000, event='evidence', rescored_rows=[(1, ((-1, -15.), (0, 15.)))])
    assert len(second.prediction['predictions']) == 1
    events = second.audit['components'][0]['recovery_events']
    assert len(events) == 1 and events[0]['previously_unexpanded']
    assert not second.audit['global_fallback_used']
    assert step(t, raw, [((-1, 0.),), ((-1, 0.), (0, -8.))]).prediction_json == first.prediction_json
    t.close()


def test_complete_identity_classes_certify_bayes_regret_not_zero_identity_error(tmp_path):
    config = PersistentComponentConfig(state=ForestTrackingConfig(active_limit=10, expansion_budget=100, max_model_regret=0.))
    t = PersistentComponentTracker(tmp_path/'exact-actions.db', sequence_id='0003', config=config)
    raw = (observation('a', source=0, index=0), observation('b', source=0, index=1),
           observation('c', source=1, index=0), observation('d', source=1, index=1))
    row = ((-1, 0.), (0, math.log(3)), (1, math.log(2)))
    result = step(t, raw, [((-1, 0.),), ((-1, 0.),), row, row])
    summary = result.audit['components'][0]
    assert not summary['frontier'] and len(summary['active']) == 7
    assert summary['decision']['conditional_bayes_lower_kind'] == 'complete_supported_action_enumeration'
    assert summary['decision']['conditional_risk'] == pytest.approx(29/92)
    assert result.audit['model_regret_upper'] == 0 and not result.audit['global_fallback_used']
    t.close()


def test_new_arrived_evidence_recovers_identity_with_old_rows_frozen(tmp_path):
    config = PersistentComponentConfig(state=ForestTrackingConfig(active_limit=1, expansion_budget=3, max_model_regret=.05))
    t = PersistentComponentTracker(tmp_path/'future-evidence.db', sequence_id='0003', config=config)
    raw = (observation('a', source=0, frame='initial', index=0),
           observation('b', source=0, frame='initial', index=1),
           observation('c', source=1, frame='initial'))
    rows = (((-1, 0.),), ((-1, 0.),), ((-1, -30.), (0, 0.), (1, -10.)))
    first = step(t, raw, rows)
    assert joint_action(t, first) == (-1, -1, 0)
    # Two new same-source/frame observations cannot share one identity. Their
    # strong links to a and c favour the previously omitted c->b history.
    new = (observation('d', source=0, frame='later', index=0, state_us=1_200_000),
           observation('e', source=0, frame='later', index=1, state_us=1_200_000))
    new_rows = (((-1, -30.), (0, 0.)), ((-1, -30.), (2, 0.)))
    result = step(t, new, new_rows, time=1_200_100, event='evidence')
    factor_sha = result.audit['factor_rows_sha256']
    recovery_events = []
    for iteration in range(12):
        recovery_events.extend(e for c in result.audit['components'] for e in c['recovery_events'])
        if joint_action(t, result)[2] == 1:
            break
        result = step(t, time=1_200_200+100*iteration, event='compute-'+str(iteration))
        assert result.audit['factor_rows_sha256'] == factor_sha
        assert not result.audit['rescored_rows']
    assert joint_action(t, result) == (-1, -1, 1, 0, 2)
    assert any(e['previously_unexpanded'] for e in recovery_events)
    assert step(t, raw, rows).prediction_json == first.prediction_json
    t.close()


@pytest.mark.parametrize('limits', [dict(max_components=1), dict(max_total_prefix_nodes=2), dict(max_member_rows=1)])
def test_shared_capacity_failure_preserves_raw_data_and_output(tmp_path, limits):
    t = tracker(tmp_path/'caps.db', **limits)
    first = step(t, [observation('a')], [((-1, 0.),)])
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='capacity'):
        step(t, [observation('b', source=1, state_us=1_200_000)], [((-1, 0.),)], time=1_300_000, event='bad')
    assert tuple(t.db.iterdump()) == before and t.n == 1
    assert t.meta['prediction_sha256'] == first.prediction['commit_sha256']
    t.close()


def test_bridge_rescore_immutable_outputs_and_closed_reopen(tmp_path):
    t = tracker(tmp_path/'merge.db')
    initial = (observation('a'), observation('b', source=1), observation('c', frame='next', state_us=1_010_000))
    rows = (((-1, 0.),), ((-1, 0.),), ((-1, -10.), (0, 0.)))
    first = step(t, initial, rows)
    raw = observation('bridge', source=1, frame='bridge', state_us=1_200_000)
    second = step(t, [raw], [((-1, -3.), (1, 1.), (2, 1.))], time=1_300_000, event='merge')
    assert len(second.audit['components']) == 1 and second.audit['components'][0]['merge_restart']
    assert step(t, initial, rows).prediction_json == first.prediction_json
    file_sha = t.close()
    t = PersistentComponentTracker.open(tmp_path/'merge.db', expected_prediction_sha256=second.prediction['commit_sha256'], expected_database_sha256=file_sha)
    third = step(t, time=1_400_000, event='rescore', rescored_rows=[(2, ((-1, 10.), (0, -10.)))])
    assert third.audit['rescored_rows'] and len(third.audit['components']) == 1
    assert step(t, initial, rows).prediction_json == first.prediction_json
    assert third.prediction['previous_commit_sha256'] == second.prediction['commit_sha256']
    t.close()


def test_shared_failure_rolls_back_new_components_raw_data_and_output(tmp_path, monkeypatch):
    t = tracker(tmp_path/'rollback.db')
    initial = step(t, [observation('a')], [((-1, 0.),)])
    before = tuple(t.db.iterdump())
    original = t.store.save_kernel
    monkeypatch.setattr(t.store, 'save_kernel', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('save failure')))
    raw = observation('b', source=1, frame='b', state_us=1_200_000)
    with pytest.raises(RuntimeError, match='save failure'):
        step(t, [raw], [((-1, 0.),)], time=1_300_000, event='new')
    assert tuple(t.db.iterdump()) == before and t.n == 1
    monkeypatch.setattr(t.store, 'save_kernel', original)
    assert step(t, [raw], [((-1, 0.),)], time=1_300_000, event='new').prediction['previous_commit_sha256'] == initial.prediction['commit_sha256']
    t.close()


def test_budget_is_shared_and_lru_is_hard_capped(tmp_path):
    t = tracker(tmp_path/'budget.db', budget=3, max_total_cache_entries=3)
    raw = tuple(observation(str(i), source=i % 2, frame=str(i), state_us=1_000_000+i) for i in range(8))
    rows = tuple(((-1, 0.),) if i < 4 else ((-1, -1.), (i-4, 1.)) for i in range(8))
    result = step(t, raw, rows)
    assert len(result.audit['components']) == 4
    assert result.audit['expansions'] <= 3
    assert result.audit['search_steps'] <= 3
    assert len(t.shared_cache) <= 3 and sum(t.shared_cache.counts.values()) == len(t.shared_cache)
    t.close()


def test_long_sequence_merge_and_resume_preserve_ids_without_replaying_old_outputs(tmp_path):
    config = PersistentComponentConfig(state=ForestTrackingConfig(window_us=200_000, active_limit=2,
        expansion_budget=8, max_model_regret=.05), max_total_cache_entries=32)
    t = PersistentComponentTracker(tmp_path/'long.db', sequence_id='0003', config=config)
    all_raw, all_rows, prior, initial_ids = [], [], [], None
    first = None
    for frame in range(120):
        time = 1_000_000+100_000*frame
        current = [observation(f'{frame}-{side}', side*50., source=0, frame=str(frame), index=side,
                               state_us=time) for side in range(2)]
        rows = [((-1, 0.),) if frame == 0 else ((-1, -20.), (prior[side], 0.)) for side in range(2)]
        new_prior = [len(all_raw), len(all_raw)+1]
        if frame == 60:
            current.append(observation('bridge', 100., source=1, frame='bridge', state_us=time, score=.1))
            rows.append(((-1, 0.), (prior[0], -30.), (prior[1], -30.)))
        all_raw.extend(current); all_rows.extend(rows)
        result = step(t, current, rows, time=time+100, event=str(frame))
        ids = {p['track_id'] for p in result.prediction['predictions']}
        if first is None:
            first, initial_ids = result, ids
        assert ids == initial_ids
        assert result.audit['search_steps'] <= 8 and result.audit['shared_cache_entries'] <= 32
        assert len(result.audit['components']) == (2 if frame < 60 else 1)
        if frame in (0, 59, 60, 119):
            factors = ForestFactors(tuple(o.node for o in all_raw), tuple(all_rows))
            expected, _ = replay_forest_states('0003', tuple(all_raw), factors, joint_action(t, result), time+100, t.config.state)
            expected = sorted([{k: p[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for p in expected], key=lambda p: p['track_id'])
            assert canonical(result.prediction['predictions']) == canonical(expected)
        if frame == 60:
            sha = t.close()
            t = PersistentComponentTracker.open(tmp_path/'long.db', expected_database_sha256=sha,
                                                expected_prediction_sha256=result.prediction['commit_sha256'])
        prior = new_prior
    assert t.n == 241
    old_bytes = t.db.execute("SELECT prediction FROM events WHERE event_id='0'").fetchone()[0]
    assert old_bytes == first.prediction_json
    t.close()


def test_empty_and_expired_components_keep_raw_history(tmp_path):
    t = tracker(tmp_path/'expired.db')
    empty = step(t)
    assert not empty.prediction['predictions'] and empty.audit['model_regret_upper'] == 0
    raw = observation('a', state_us=1_200_000)
    first = step(t, [raw], [((-1, 0.),)], time=1_300_000, event='birth')
    expired = step(t, time=5_000_000, event='expired')
    assert first.prediction['predictions'] and not expired.prediction['predictions']
    assert t.n == 1 and len(t.store.live()) == 1
    assert expired.audit['components'][0]['expired_output_only']
    assert expired.audit['search_steps'] == 0
    t.close()
