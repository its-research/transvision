import itertools
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig, replay_forest_states
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.persistent_beam_tracking import (
    PersistentRankedBeamConfig, PersistentRankedBeamTracker, _Work,
)
from transvision.models.event_track_v2x.persistent_joint_beam import (
    PersistentJointBeamConfig, PersistentJointBeamTracker, RankedCartesianCursor,
)
from test_forest_tracking import observation
from test_persistent_forest import brute
from test_persistent_component_tracking import step, joint_action
from test_persistent_beam_tracking import evidence_scene


def tracker(path, width=2, **limits):
    return PersistentJointBeamTracker(path, sequence_id='0003', config=PersistentJointBeamConfig(
        state=ForestTrackingConfig(active_limit=width), **limits))


@pytest.mark.parametrize('seed', range(5))
def test_lazy_cursor_matches_entire_cartesian_product_with_ties(seed):
    rng = np.random.default_rng(seed)
    groups = [(c, [(float(rng.integers(-3, 4)), h) for h in range(c+1)]) for c in range(1, 5)]
    expected = sorted([(math.fsum(w for w, _ in choice), tuple((c, h) for (c, _), (_, h) in zip(groups, choice)))
        for choice in itertools.product(*(options for _, options in groups))], key=lambda item: (-item[0], item[1]))
    work = _Work(PersistentJointBeamConfig())
    cursor = RankedCartesianCursor(groups, state_limit=1000, work=work)
    actual = []
    while cursor.queue:
        assert cursor.peek() == expected[len(actual)][0]
        actual.append(cursor.pop())
    assert actual == expected and cursor.emitted == len(expected)
    assert len(cursor.seen) == len(expected) == work.merge_score_evaluations
    with pytest.raises(StopIteration):
        cursor.pop()


def test_cursor_does_not_eagerly_materialize_large_products():
    groups = [(c, [(0., 1), (-100., 2)]) for c in range(1, 21)]
    work = _Work(PersistentJointBeamConfig())
    cursor = RankedCartesianCursor(groups, state_limit=64, work=work)
    assert len(cursor.seen) == 1
    value, choices = cursor.pop()
    assert value == 0 and len(choices) == 20
    assert len(cursor.seen) == 21 and cursor.peek() == -100.
    assert len(cursor.seen) < 2**20


def merge_scene():
    raw = tuple(observation(str(i), source=i % 2, frame='old'+str(i)) for i in range(4))
    rows = (((-1, 0.),), ((-1, -5.), (0, 0.)), ((-1, 0.),), ((-1, -5.), (2, 0.)))
    new = tuple(observation('new'+str(i), source=0, frame='future', index=i, state_us=1_200_000) for i in range(4))
    new_rows = (((-1, -30.), (0, 0.), (2, -100.)), ((-1, -30.), (1, 0.)),
                ((-1, -30.), (2, 0.)), ((-1, -30.), (3, 0.)))
    return raw, rows, new, new_rows


def test_new_evidence_can_select_prior_combination_below_the_old_product_top_k(tmp_path):
    joint = tracker(tmp_path/'joint.db')
    preselected = PersistentRankedBeamTracker(tmp_path/'preselected.db', sequence_id='0003',
        config=PersistentRankedBeamConfig(state=ForestTrackingConfig(active_limit=2)))
    raw, rows, new, new_rows = merge_scene()
    initial = [step(t, raw, rows) for t in (joint, preselected)]
    results = [step(t, new, new_rows, time=1_300_000, event='merge') for t in (joint, preselected)]
    factors = ForestFactors(tuple(o.node for o in (*raw, *new)), (*rows, *new_rows))
    assert factors.roots(joint_action(joint, results[0]))[:4] == (0, 1, 2, 3)
    assert factors.roots(joint_action(preselected, results[1]))[:4] != (0, 1, 2, 3)
    assert results[0].audit['factor_rows_sha256'] == results[1].audit['factor_rows_sha256']
    summary, = results[0].audit['components']
    assert not results[0].audit['prior_cartesian_preselection']
    assert not summary['merge_pruning_stages'][0]['prior_preselection']
    assert summary['merge_pruning_stages'][0]['prior_combinations_materialized'] > 2
    assert summary['pruning'][0]['exact_arithmetic_top_k_condition_met']
    assert summary['pruning'][0]['candidate_evaluations'] == results[0].audit['candidate_evaluations']
    for t, first in zip((joint, preselected), initial):
        assert step(t, raw, rows).prediction_json == first.prediction_json
        t.close()


@pytest.mark.parametrize('width', [1, 2, 20])
@pytest.mark.parametrize('seed', range(3))
def test_three_way_join_rank_weights_risk_and_state_against_full_parent_oracle(tmp_path, width, seed):
    rng = np.random.default_rng(seed)
    t = tracker(tmp_path/'random.db', width)
    raw = tuple(observation(str(i), source=i % 2, frame='old'+str(i)) for i in range(6))
    supports = [(-1,) if i % 2 == 0 else (-1, i-1) for i in range(6)]
    rows = tuple(tuple((p, float(rng.normal())) for p in parents) for parents in supports)
    first = step(t, raw, rows)
    allowed_projections = []
    for summary in first.audit['components']:
        members = t.store.members(summary['component'])
        kernel = t.kernels[summary['component']]
        roots = {tuple(members[kernel._prefix(kernel.ancestor(a['handle'], i+1)).root] for i in range(kernel.n))
                 for a in summary['active']}
        allowed_projections.append((members, roots))
    new = tuple(observation('new'+str(i), source=0, frame='future', index=i, state_us=1_200_000) for i in range(3))
    supports = ((-1, 0, 2, 4), (-1, 1, 3), (-1, 3, 5))
    new_rows = tuple(tuple((p, float(rng.normal())) for p in parents) for parents in supports)
    updated = tuple((p, w+.25*(p+1)) for p, w in rows[1])
    result = step(t, new, new_rows, time=1_300_000, event='join', rescored_rows=[(1, updated)])
    factors = ForestFactors(tuple(o.node for o in (*raw, *new)), (rows[0], updated, *rows[2:], *new_rows))
    probabilities, roots, z = brute(factors)
    weights = {}
    for h, prob in probabilities.items():
        if all(tuple(roots[h][i] for i in members) in allowed for members, allowed in allowed_projections):
            weights[roots[h]] = weights.get(roots[h], 0.)+prob*z
    expected = sorted(weights.items(), key=lambda item: -item[1])[:width]
    summary, = result.audit['components']
    kernel = t.kernels[summary['component']]
    actual = [(factors.roots(kernel.parents(a['handle'])), math.exp(a['log_weight'])) for a in summary['active']]
    assert [r for r, _ in actual] == [r for r, _ in expected]
    assert [w for _, w in actual] == pytest.approx([w for _, w in expected])
    action = joint_action(t, result)
    def risk(a):
        return sum(prob*sum(roots[a][i] != roots[h][i] for i in range(len(raw)+len(new)))/(len(raw)+len(new))
                   for h, prob in probabilities.items())
    assert risk(action)-min(risk(h) for h in probabilities) <= result.audit['model_regret_upper']+1e-12
    expected_state, _ = replay_forest_states('0003', (*raw, *new), factors, action, 1_300_000, t.config.state)
    expected_state = sorted([{k: p[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for p in expected_state], key=lambda p: p['track_id'])
    assert canonical(result.prediction['predictions']) == canonical(expected_state)
    assert step(t, raw, rows).prediction_json == first.prediction_json
    t.close()


def test_joint_search_is_lazy_when_old_best_combination_is_already_decisive(tmp_path):
    t = tracker(tmp_path/'lazy.db', 2)
    raw, rows, new, new_rows = merge_scene()
    rows = (rows[0], ((-1, -100.), (0, 0.)), rows[2], ((-1, -100.), (2, 0.)))
    step(t, raw, rows)
    # One new bridge, not the later conflicting same-frame disambiguators.
    result = step(t, [new[0]], [((-1, 0.), (0, -1.), (2, -2.))], time=1_300_000, event='bridge')
    detail = result.audit['components'][0]['merge_pruning_stages'][0]
    assert detail['prior_combinations_materialized'] == 1
    assert detail['cartesian_queue_remaining'] > 0
    assert result.audit['components'][0]['pruning'][0]['remaining_log_upper'] < result.audit['components'][0]['pruning'][0]['kth_log_weight']
    t.close()


@pytest.mark.parametrize('limits', [dict(max_cartesian_states=1), dict(max_batch_frontier=1), dict(max_merge_prefix_steps=1)])
def test_joint_capacity_failure_rolls_back_old_weights_maps_and_new_observations(tmp_path, limits):
    t = tracker(tmp_path/'caps.db', **limits)
    # Build the fixture using a harmless wider per-batch frontier if necessary;
    # the one-entry case uses deterministic old components instead.
    if limits.get('max_batch_frontier') == 1:
        raw = (observation('a'), observation('b', source=1))
        rows = (((-1, 0.),), ((-1, 0.),))
        new, new_rows = (observation('bridge', frame='future', state_us=1_200_000),), (((-1, 0.), (0, 0.), (1, 0.)),)
    else:
        raw, rows, new, new_rows = merge_scene()
    first = step(t, raw, rows)
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='capacity'):
        step(t, new, new_rows, time=1_300_000, event='join', rescored_rows=[(0, ((-1, 1.),))])
    assert tuple(t.db.iterdump()) == before and t.n == len(raw)
    assert step(t, raw, rows).prediction_json == first.prediction_json
    t.close()


def test_reopen_preserves_irreversible_previous_event_pruning(tmp_path):
    t = tracker(tmp_path/'resume.db', 1)
    raw, rows, new, new_rows = evidence_scene()
    first = step(t, raw, rows)
    sha = t.close()
    t = PersistentJointBeamTracker.open(tmp_path/'resume.db', expected_database_sha256=sha,
        expected_prediction_sha256=first.prediction['commit_sha256'])
    result = step(t, new, new_rows, time=1_300_000, event='evidence')
    assert joint_action(t, result)[2] == 0 and not result.audit['recovery_enabled']
    assert step(t, raw, rows).prediction_json == first.prediction_json
    sha = t.close()
    with pytest.raises(ValueError, match='schema'):
        PersistentRankedBeamTracker.open(tmp_path/'resume.db', expected_database_sha256=sha,
            expected_prediction_sha256=result.prediction['commit_sha256'])


def test_late_state_time_on_joint_merge_replays_raw_observations_and_preserves_old_output(tmp_path):
    t = tracker(tmp_path/'late.db', 4)
    raw = (observation('a', 1.), observation('b', 10., source=1))
    rows = (((-1, 0.),), ((-1, 0.),))
    first = step(t, raw, rows)
    late = observation('late', 0., source=1, frame='late', state_us=900_000, arrival_us=1_200_000)
    row = ((-1, -20.), (0, 0.), (1, -10.))
    result = step(t, [late], [row], time=1_300_000, event='late')
    factors = ForestFactors(tuple(o.node for o in (*raw, late)), (*rows, row))
    expected, _ = replay_forest_states('0003', (*raw, late), factors, joint_action(t, result), 1_300_000, t.config.state)
    expected = sorted([{k: p[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for p in expected], key=lambda p: p['track_id'])
    assert canonical(result.prediction['predictions']) == canonical(expected)
    assert result.audit['components'][0]['full_model_support_still_in_beam']
    assert step(t, raw, rows).prediction_json == first.prediction_json
    sha = t.close()
    t = PersistentJointBeamTracker.open(tmp_path/'late.db', expected_database_sha256=sha,
        expected_prediction_sha256=result.prediction['commit_sha256'])
    continued = step(t, time=1_400_000, event='continued')
    assert not continued.audit['components'][0]['pruning']
    assert {p['track_id'] for p in continued.prediction['predictions']} == {p['track_id'] for p in first.prediction['predictions']}
    t.close()
