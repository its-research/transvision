import itertools
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig, replay_forest_states
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.persistent_beam_tracking import (
    PersistentBeamConfig, PersistentBeamTracker, PersistentRankedBeamConfig, PersistentRankedBeamTracker, _Work, _top_products,
)
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig, PersistentComponentTracker
from transvision.models.event_track_v2x.persistent_joint_beam import PersistentJointBeamTracker
from test_forest_tracking import observation
from test_persistent_forest import brute
from test_persistent_component_tracking import step, joint_action


def tracker(path, width=4, **limits):
    return PersistentBeamTracker(path, sequence_id='0003', config=PersistentBeamConfig(
        state=ForestTrackingConfig(active_limit=width), **limits))


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('width', [1, 2, 5, 100])
def test_product_join_is_exact_top_k_without_enumerating_cartesian_support(seed, width):
    rng = np.random.default_rng(seed)
    groups = [(c, [(float(rng.integers(-3, 4)), h) for h in range(1, c+2)]) for c in range(1, 5)]
    work = _Work(PersistentBeamConfig())
    actual, complete, _ = _top_products(groups, width, work)
    expected = sorted([(math.fsum(w for w, _ in choice), tuple((c, h) for (c, _), (_, h) in zip(groups, choice)))
        for choice in itertools.product(*(g[1] for g in groups))], key=lambda v: (-v[0], v[1]))[:width]
    assert actual == expected
    assert complete == (math.prod(len(g[1]) for g in groups) <= width)
    assert work.merge_score_evaluations <= 3*width*len(groups)


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('width', [1, 2, 100])
def test_absolute_class_weights_regret_and_states_against_complete_parent_forests(tmp_path, seed, width):
    t = tracker(tmp_path/'beam.db', width)
    rng = np.random.default_rng(seed)
    raw = tuple(observation(str(i), source=i % 2, frame=str(i), state_us=1_000_000+i) for i in range(5))
    support = ((-1,), (-1,), (-1, 0), (-1, 0, 2), (-1, 1))
    rows = tuple(tuple((p, float(rng.normal())) for p in parents) for parents in support)
    result = step(t, raw, rows)
    factors = ForestFactors(tuple(o.node for o in raw), rows)
    probability, labels, partition = brute(factors)
    action, scope = joint_action(t, result), result.audit['decision_indices']
    def risk(a):
        return sum(p*sum(labels[a][i] != labels[h][i] for i in scope)/len(scope) for h, p in probability.items())
    assert risk(action)-min(risk(h) for h in probability) <= result.audit['model_regret_upper']+1e-12
    assert math.log(partition) <= result.audit['log_partition_upper']+1e-12
    for summary in result.audit['components']:
        kernel = t.kernels[summary['component']]
        local = ForestFactors(tuple(raw[i].node for i in t.store.members(summary['component'])), tuple(kernel._row(i) for i in range(kernel.n)))
        probs, roots, z = brute(local)
        selected_roots = []
        for item in summary['active']:
            identity = local.roots(kernel.parents(item['handle']))
            selected_roots.append(identity)
            assert math.exp(item['log_weight']) == pytest.approx(z*sum(p for h, p in probs.items() if roots[h] == identity))
        omitted = sum(p for h, p in probs.items() if roots[h] not in selected_roots)
        assert omitted <= summary['eta_upper']+1e-12
        assert len(summary['active']) <= width
        if width == 100:
            assert summary['full_model_support_still_in_beam'] and omitted == 0
    expected, _ = replay_forest_states('0003', raw, factors, action, 1_100_000, t.config.state)
    expected = sorted([{k: p[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for p in expected], key=lambda p: p['track_id'])
    assert canonical(result.prediction['predictions']) == canonical(expected)
    t.close()


def evidence_scene():
    raw = (observation('a', source=0, frame='initial', index=0),
           observation('b', source=0, frame='initial', index=1), observation('c', source=1, frame='initial'))
    rows = (((-1, 0.),), ((-1, 0.),), ((-1, -30.), (0, 0.), (1, -10.)))
    new = (observation('d', source=0, frame='later', index=0, state_us=1_200_000),
           observation('e', source=0, frame='later', index=1, state_us=1_200_000))
    new_rows = (((-1, -30.), (0, 0.)), ((-1, -30.), (2, 0.)))
    return raw, rows, new, new_rows


def test_old_pruning_is_irreversible_on_new_evidence_rescore_compute_and_reopen(tmp_path):
    t = tracker(tmp_path/'irreversible.db', 1)
    raw, rows, new, new_rows = evidence_scene()
    first = step(t, raw, rows)
    assert joint_action(t, first) == (-1, -1, 0)
    second = step(t, new, new_rows, time=1_200_100, event='evidence')
    assert joint_action(t, second)[2] == 0
    assert not second.audit['recovery_enabled']
    sha = t.close()
    t = PersistentBeamTracker.open(tmp_path/'irreversible.db', expected_database_sha256=sha,
        expected_prediction_sha256=second.prediction['commit_sha256'])
    for i in range(3):
        result = step(t, time=1_300_000+100*i, event='compute'+str(i),
                      rescored_rows=[(2, ((-1, -30.), (0, -100.), (1, 100.)))])
        assert joint_action(t, result)[2] == 0
        assert not result.audit['components'][0]['recovery_events']
        assert result.audit['components'][0]['eta_upper'] > .99
    assert step(t, raw, rows).prediction_json == first.prediction_json
    t.close()


def test_same_evidence_and_raw_factors_recoverable_differs_from_top_one(tmp_path):
    beam = tracker(tmp_path/'beam.db', 1)
    full = PersistentComponentTracker(tmp_path/'recover.db', sequence_id='0003', config=PersistentComponentConfig(
        state=ForestTrackingConfig(active_limit=1, expansion_budget=3, max_model_regret=.05)))
    raw, rows, new, new_rows = evidence_scene()
    first = [step(t, raw, rows) for t in (beam, full)]
    assert first[0].audit['factor_rows_sha256'] == first[1].audit['factor_rows_sha256']
    result = [step(t, new, new_rows, time=1_200_100, event='new') for t in (beam, full)]
    for i in range(12):
        assert result[0].audit['factor_rows_sha256'] == result[1].audit['factor_rows_sha256']
        if joint_action(full, result[1])[2] == 1:
            break
        result = [step(t, time=1_300_000+i, event='compute'+str(i)) for t in (beam, full)]
    assert joint_action(beam, result[0])[2] == 0
    assert joint_action(full, result[1])[2] == 1
    for t in (beam, full):
        t.close()


def test_wider_beam_can_retain_the_later_correct_alternative_without_recovery(tmp_path):
    t = tracker(tmp_path/'top-two.db', 2)
    raw, rows, new, new_rows = evidence_scene()
    first = step(t, raw, rows)
    kernel = t.kernels[first.audit['components'][0]['component']]
    old_classes = {kernel.parents(a['handle'])[2] for a in first.audit['components'][0]['active']}
    assert old_classes == {0, 1}
    result = step(t, new, new_rows, time=1_300_000, event='evidence')
    assert joint_action(t, result) == (-1, -1, 1, 0, 2)
    assert not result.audit['components'][0]['recovery_events']
    t.close()


def test_merge_only_lifts_retained_old_classes_and_never_restarts_raw_support(tmp_path):
    t = tracker(tmp_path/'merge.db', 1)
    raw = (observation('a'), observation('b', source=1, frame='b'),
           observation('c', source=1, frame='c', state_us=1_010_000))
    rows = (((-1, 0.),), ((-1, 0.),), ((-1, -10.), (0, 0.)))
    first = step(t, raw, rows)
    assert joint_action(t, first) == (-1, -1, 0)
    bridge = observation('bridge', source=0, frame='bridge', state_us=1_200_000)
    second = step(t, [bridge], [((-1, -4.), (1, 1.), (2, 1.))], time=1_300_000, event='bridge',
                  rescored_rows=[(2, ((-1, 30.), (0, -30.)))])
    assert len(second.audit['components']) == 1 and joint_action(t, second)[2] == 0
    assert second.audit['components'][0]['merge_retained_beams_only']
    assert not second.audit['components'][0]['merge_restart']
    assert second.audit['merge_prefix_steps'] == 3
    assert step(t, raw, rows).prediction_json == first.prediction_json
    t.close()


def test_unpruned_beam_merge_remains_full_model_and_uses_rescored_class_sums(tmp_path):
    t = tracker(tmp_path/'full-merge.db', 100)
    raw = (observation('a'), observation('b', source=1, frame='b'),
           observation('c', source=1, frame='c', state_us=1_010_000))
    rows = (((-1, 0.),), ((-1, 0.),), ((-1, -1.), (0, 1.)))
    step(t, raw, rows)
    bridge = observation('bridge', source=0, frame='bridge', state_us=1_200_000)
    updated = ((-1, 3.), (0, -2.))
    row = ((-1, 0.), (1, 1.), (2, 2.))
    result = step(t, [bridge], [row], time=1_300_000, event='bridge', rescored_rows=[(2, updated)])
    factors = ForestFactors(tuple(o.node for o in (*raw, bridge)), (*rows[:2], updated, row))
    _, _, z = brute(factors)
    summary, = result.audit['components']
    assert summary['full_model_support_still_in_beam']
    assert math.exp(summary['log_retained']) == pytest.approx(z)
    assert summary['eta_upper'] == 0 and result.audit['model_regret_upper'] == 0
    t.close()


def test_merge_work_limit_rolls_back_catalog_and_predecessor_weights(tmp_path):
    t = tracker(tmp_path/'merge-limit.db', 2, max_merge_prefix_steps=1)
    step(t, [observation('a'), observation('b', source=1)], [((-1, 0.),), ((-1, 0.),)])
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='merge_prefix_steps capacity'):
        step(t, [observation('bridge', frame='bridge', state_us=1_200_000)],
             [((-1, 0.), (0, 0.), (1, 0.))], time=1_300_000, event='merge', rescored_rows=[(0, ((-1, 1.),))])
    assert tuple(t.db.iterdump()) == before and t.n == 2
    t.close()


@pytest.mark.parametrize('tracker_type', [PersistentBeamTracker, PersistentJointBeamTracker])
def test_long_beam_stream_merge_and_closed_reopen_preserve_output_ids(tmp_path, tracker_type):
    t = tracker_type(tmp_path/'long.db', sequence_id='0003', config=tracker_type.CONFIG_TYPE(
        state=ForestTrackingConfig(active_limit=2), max_total_cache_entries=16))
    raw_history, rows_history, prior = [], [], []
    first = initial_ids = None
    for frame in range(120):
        time = 1_000_000+100_000*frame
        current = [observation(f'{frame}-{side}', side*50., source=0, frame=str(frame), index=side, state_us=time) for side in range(2)]
        rows = [((-1, 0.),) if frame == 0 else ((-1, -20.), (prior[side], 0.)) for side in range(2)]
        new_prior = [len(raw_history), len(raw_history)+1]
        if frame == 60:
            current.append(observation('bridge', 100., source=1, frame='bridge', state_us=time, score=.1))
            rows.append(((-1, 0.), (prior[0], -30.), (prior[1], -30.)))
        raw_history.extend(current); rows_history.extend(rows)
        result = step(t, current, rows, time=time+100, event=str(frame))
        ids = {p['track_id'] for p in result.prediction['predictions']}
        if first is None:
            first, initial_ids = result, ids
        assert ids == initial_ids
        assert result.audit['shared_cache_entries'] <= 16
        if frame in (0, 60, 119):
            factors = ForestFactors(tuple(o.node for o in raw_history), tuple(rows_history))
            expected, _ = replay_forest_states('0003', tuple(raw_history), factors, joint_action(t, result), time+100, t.config.state)
            expected = sorted([{k: p[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for p in expected], key=lambda p: p['track_id'])
            assert canonical(result.prediction['predictions']) == canonical(expected)
        if frame == 60:
            sha = t.close()
            t = tracker_type.open(tmp_path/'long.db', expected_database_sha256=sha,
                expected_prediction_sha256=result.prediction['commit_sha256'])
        prior = new_prior
    assert t.db.execute("SELECT prediction FROM events WHERE event_id='0'").fetchone()[0] == first.prediction_json
    t.close()


@pytest.mark.parametrize('limits', [dict(max_beam_expansions=1), dict(max_candidate_evaluations=1), dict(max_total_prefix_nodes=2)])
def test_work_or_storage_failure_is_atomic(tmp_path, limits):
    t = tracker(tmp_path/'rollback.db', 2, **limits)
    first = step(t, [observation('a')], [((-1, 0.),)])
    before = tuple(t.db.iterdump())
    raw = (observation('b', source=1, state_us=1_200_000), observation('c', frame='c', state_us=1_200_000))
    with pytest.raises(ValueError, match='capacity'):
        step(t, raw, [((-1, 0.), (0, 0.)), ((-1, 0.), (1, 0.))], time=1_300_000, event='overload')
    assert tuple(t.db.iterdump()) == before and t.n == 1
    assert step(t, [observation('a')], [((-1, 0.),)]).prediction_json == first.prediction_json
    t.close()


def test_backend_schema_cannot_be_reopened_as_recoverable(tmp_path):
    t = tracker(tmp_path/'beam.db')
    result = step(t)
    sha = t.close()
    with pytest.raises(ValueError, match='schema'):
        PersistentComponentTracker.open(tmp_path/'beam.db', expected_database_sha256=sha,
            expected_prediction_sha256=result.prediction['commit_sha256'])


def test_beam_rejects_ignored_recoverable_configuration():
    with pytest.raises(ValueError, match='does not use'):
        PersistentBeamConfig(state=ForestTrackingConfig(expansion_budget=3))


@pytest.mark.parametrize('width', [1, 2, 4, 20])
@pytest.mark.parametrize('seed', range(3))
def test_batch_ranked_classes_equal_exhaustive_top_k_current_batch(tmp_path, width, seed):
    rng = np.random.default_rng(seed)
    t = PersistentRankedBeamTracker(tmp_path/'ranked.db', sequence_id='0003',
        config=PersistentRankedBeamConfig(state=ForestTrackingConfig(active_limit=width)))
    raw = tuple(observation(str(i), source=i//2, index=i % 2) for i in range(4))
    support = ((-1,), (-1,), (-1, 0, 1), (-1, 0, 1))
    rows = tuple(tuple((p, float(rng.normal())) for p in options) for options in support)
    result = step(t, raw, rows)
    factors = ForestFactors(tuple(o.node for o in raw), rows)
    probability, labels, z = brute(factors)
    masses = {}
    for h, p in probability.items():
        masses[labels[h]] = masses.get(labels[h], 0.)+p*z
    expected = sorted(masses.items(), key=lambda item: -item[1])[:width]
    summary, = result.audit['components']
    kernel = t.kernels[summary['component']]
    actual = [(factors.roots(kernel.parents(a['handle'])), math.exp(a['log_weight'])) for a in summary['active']]
    assert [r for r, _ in actual] == [r for r, _ in expected]
    assert [w for _, w in actual] == pytest.approx([w for _, w in expected])
    assert summary['full_model_support_still_in_beam'] == (width >= len(masses))
    assert summary['pruning'][0]['exact_arithmetic_top_k_condition_met']
    t.close()


def test_joint_batch_ranking_does_not_discard_available_later_row_evidence(tmp_path):
    node = tracker(tmp_path/'node.db', 1)
    batch = PersistentRankedBeamTracker(tmp_path/'batch.db', sequence_id='0003',
        config=PersistentRankedBeamConfig(state=ForestTrackingConfig(active_limit=1)))
    raw = tuple(observation(str(i), source=i//2, index=i % 2) for i in range(4))
    rows = (((-1, 0.),), ((-1, 0.),), ((-1, -30.), (0, math.log(3)), (1, math.log(2))),
            ((-1, -30.), (0, math.log(100)), (1, math.log(.1))))
    a, b = [step(t, raw, rows) for t in (node, batch)]
    assert joint_action(node, a) == (-1, -1, 0, 1)
    assert joint_action(batch, b) == (-1, -1, 1, 0)
    assert a.audit['factor_rows_sha256'] == b.audit['factor_rows_sha256']
    node.close(); batch.close()


def test_batch_ranking_does_not_resurrect_old_event_pruning(tmp_path):
    t = PersistentRankedBeamTracker(tmp_path/'batch.db', sequence_id='0003',
        config=PersistentRankedBeamConfig(state=ForestTrackingConfig(active_limit=1)))
    raw, rows, new, new_rows = evidence_scene()
    first = step(t, raw, rows)
    result = step(t, new, new_rows, time=1_300_000, event='evidence')
    assert joint_action(t, result)[2] == 0 and not result.audit['recovery_enabled']
    assert step(t, raw, rows).prediction_json == first.prediction_json
    t.close()


def test_batch_ranked_prior_merge_selection_is_not_full_product_extension_ranking(tmp_path):
    t = PersistentRankedBeamTracker(tmp_path/'batch-merge.db', sequence_id='0003',
        config=PersistentRankedBeamConfig(state=ForestTrackingConfig(active_limit=2)))
    raw = tuple(observation(str(i), source=i % 2, frame='old'+str(i)) for i in range(4))
    rows = (((-1, 0.),), ((-1, -5.), (0, 0.)), ((-1, 0.),), ((-1, -5.), (2, 0.)))
    first = step(t, raw, rows)
    assert len(first.audit['components']) == 2
    assert all(s['full_model_support_still_in_beam'] for s in first.audit['components'])
    new = tuple(observation('new'+str(i), source=0, frame='future', index=i, state_us=1_200_000) for i in range(4))
    new_rows = (((-1, -30.), (0, 0.), (2, -100.)), ((-1, -30.), (1, 0.)),
                ((-1, -30.), (2, 0.)), ((-1, -30.), (3, 0.)))
    result = step(t, new, new_rows, time=1_300_000, event='merge')
    factors = ForestFactors(tuple(o.node for o in (*raw, *new)), (*rows, *new_rows))
    probabilities, roots, _ = brute(factors)
    best = max(probabilities, key=probabilities.get)
    assert roots[best][:4] == (0, 1, 2, 3)
    assert factors.roots(joint_action(t, result))[:4] != (0, 1, 2, 3)
    summary, = result.audit['components']
    assert summary['merge_retained_beams_only'] and not summary['full_model_support_still_in_beam']
    assert summary['pruning'][0]['selection_domain'] == 'extensions_of_preselected_retained_prior_beam_not_full_history'
    t.close()


def test_batch_frontier_limit_fails_instead_of_silently_returning_non_top_k(tmp_path):
    t = PersistentRankedBeamTracker(tmp_path/'batch-limit.db', sequence_id='0003',
        config=PersistentRankedBeamConfig(max_batch_frontier=1))
    first = step(t, [observation('a')], [((-1, 0.),)])
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='frontier capacity'):
        step(t, [observation('b', source=1, state_us=1_200_000)], [((-1, 0.), (0, 0.))], time=1_300_000, event='b')
    assert tuple(t.db.iterdump()) == before and t.meta['prediction_sha256'] == first.prediction['commit_sha256']
    t.close()
