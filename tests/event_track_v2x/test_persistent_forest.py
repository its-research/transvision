from dataclasses import replace
import hashlib
import itertools
import json
import math
import sqlite3

import numpy as np
import pytest

from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig, replay_forest_states
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig, PersistentForestTracker
from test_forest_tracking import observation


def tracker(path, *, storage=None, **state):
    c = ForestTrackingConfig(**{**dict(active_limit=8, expansion_budget=100, max_model_regret=1.), **state})
    return PersistentForestTracker(path, sequence_id='0003', config=PersistentForestConfig(state=c, **(storage or {})))


def step(t, raw=(), rows=(), *, reference=1_000_000, decision=None, event='first', **kw):
    return t.step(raw, rows, frame_id=event, event_id=event, reference_us=reference,
                  decision_us=decision or reference+100_000, **kw)


def assert_state(t, result, raw, rows):
    factors = ForestFactors(tuple(o.node for o in raw), tuple(rows))
    parents = t.parents(result.audit['output_handle'])
    states, _ = replay_forest_states('0003', tuple(raw), factors, parents,
                                     result.prediction['box_reference_timestamp_us'], t.config.state)
    selected = [{k: s[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for s in states]
    assert canonical(result.prediction['predictions']) == canonical(selected)


def brute(factors):
    weights, roots = {}, {}
    for choices in itertools.product(*factors.rows):
        labels, slots = [], set()
        for i, (p, _) in enumerate(choices):
            r = i if p < 0 else labels[p]
            labels.append(r)
            slot = r, factors.nodes[i].source_id, factors.nodes[i].frame_id
            if slot in slots:
                break
            slots.add(slot)
        else:
            h = tuple(p for p, _ in choices)
            weights[h] = math.exp(math.fsum(w for _, w in choices))
            roots[h] = tuple(labels)
    z = math.fsum(weights.values())
    return {h: w/z for h, w in weights.items()}, roots, z


@pytest.mark.parametrize('budget', [0, 1, 2, 5, 100])
@pytest.mark.parametrize('seed', range(4))
def test_persistent_mass_cover_and_full_legal_action_risk_against_brute(tmp_path, budget, seed):
    t = tracker(tmp_path/'tracker.db', expansion_budget=budget, active_limit=2)
    rng = np.random.default_rng(seed)
    raw = tuple(observation(str(i), i*.2, source=i % 2, index=i//2) for i in range(4))
    rows = tuple(tuple((p, float(rng.normal())) for p in range(-1, i)) for i in range(4))
    result = step(t, raw, rows)
    factors = ForestFactors(tuple(o.node for o in raw), rows)
    probabilities, labels, z = brute(factors)
    active = [t.parents(a['handle']) for a in result.audit['active']]
    frontier = [t.parents(a['handle']) for a in result.audit['frontier']]
    for h in probabilities:
        assert sum(h[:len(p)] == p for p in active+frontier) == 1
    eta = 1-math.fsum(probabilities[a] for a in active)
    assert eta <= result.audit['eta_upper']+1e-12
    assert math.log(z) <= result.audit['log_partition_upper']+1e-12
    action = t.parents(result.audit['output_handle'])
    def risk(a):
        return math.fsum(p*sum(x != y for x, y in zip(labels[a], labels[h]))/4 for h, p in probabilities.items())
    regret = risk(action)-min(risk(a) for a in probabilities)
    assert regret <= result.audit['decision']['risk_bound']+1e-12
    assert_state(t, result, raw, rows)
    t.close()


def test_long_sequence_crosses_many_windows_and_resumes_identical_next_output(tmp_path):
    t = tracker(tmp_path/'long.db', active_limit=2, expansion_budget=4,
                window_us=200_000, storage={'prefix_cache_entries': 16})
    raw, rows, first_id, prior_hash = [], [], None, '0'*64
    first_bytes = None
    for i in range(120):
        time = 1_000_000+i*100_000
        obs = observation(str(i), i*.01, state_us=time)
        row = ((-1, -20.),) if not i else ((-1, -20.), (i-1, 0.))
        raw.append(obs); rows.append(row)
        result = step(t, [obs], [row], reference=time, event=str(i))
        current_id = result.prediction['predictions'][0]['track_id']
        first_id = first_id or current_id
        assert current_id == first_id
        assert result.prediction['previous_commit_sha256'] == prior_hash
        prior_hash = result.prediction['commit_sha256']
        first_bytes = first_bytes or result.prediction_json
        assert result.audit['prefix_cache_entries'] <= 16
        assert result.audit['row_cache_entries'] <= 16
        assert len(result.audit['decision_indices']) <= 3
        if i in (0, 39, 119):
            assert_state(t, result, raw, rows)
        if i == 59:
            sealed = t.close()
            t = PersistentForestTracker.open(tmp_path/'long.db', expected_prediction_sha256=prior_hash,
                                              expected_database_sha256=sealed)
    assert result.prediction['box_reference_timestamp_us']-1_000_000 > 50*t.config.state.window_us
    assert t.db.execute('SELECT prediction FROM events WHERE ordinal=0').fetchone()[0] == first_bytes
    assert t.db.execute('SELECT count(*) FROM observations').fetchone()[0] == 120
    assert t.prefix_count < 2000
    # State cache stores one root update, not growing copies of raw histories.
    assert all('witnesses' not in json.loads(row[0]) for row in t.db.execute('SELECT payload FROM states'))
    t.close()


def test_late_observation_and_rescore_restore_old_identity_without_rewriting_output(tmp_path):
    t = tracker(tmp_path/'late.db', active_limit=1, expansion_budget=10)
    a, b = observation('a'), observation('b', 3., source=1, state_us=1_050_000)
    rows = (((-1, 0.),), ((-1, -10.), (0, 0.)))
    first = step(t, [a, b], rows, reference=1_100_000)
    assert t.parents(first.audit['output_handle']) == (-1, 0)
    late = observation('late', .3, state_us=900_000, arrival_us=1_300_000, frame='late-frame')
    newrow = ((-1, -10.), (0, 0.))
    second = step(t, [late], [newrow], reference=1_350_000, event='late',
                  rescored_rows=((1, ((-1, 10.), (0, -10.))),))
    assert t.parents(second.audit['output_handle']) == (-1, -1, 0)
    assert second.audit['restored_ancestors']
    assert_state(t, second, [a, b, late], [rows[0], ((-1, 10.), (0, -10.)), newrow])
    assert first.prediction_json == t.db.execute('SELECT prediction FROM events WHERE event_id=?', ('first',)).fetchone()[0]
    assert first.prediction['predictions'][0]['track_id'] in {p['track_id'] for p in second.prediction['predictions']}
    t.close()


def test_duplicate_event_is_byte_identical_after_reopen_and_newer_commit(tmp_path):
    t = tracker(tmp_path/'repeat.db')
    obs = observation('a')
    original = step(t, [obs], [((-1, 0.),)])
    newer = step(t, reference=1_200_000, event='newer')
    file_hash = t.close()
    t = PersistentForestTracker.open(tmp_path/'repeat.db', expected_database_sha256=file_hash,
        expected_prediction_sha256=newer.prediction['commit_sha256'])
    assert step(t, [obs], [((-1, 0.),)]) == original
    with pytest.raises(ValueError, match='conflicting duplicate'):
        step(t, [replace(obs, score=.7)], [((-1, 0.),)])
    assert t.meta['events'] == 2
    t.close()


def test_state_failure_rolls_back_sql_prefixes_factors_and_event_then_retry(tmp_path, monkeypatch):
    t = tracker(tmp_path/'atomic.db')
    a = observation('a')
    first = step(t, [a], [((-1, 0.),)])
    before = tuple(t.db.iterdump())
    original = t._ensure_state
    monkeypatch.setattr(t, '_ensure_state', lambda *a: (_ for _ in ()).throw(ValueError('injected replay error')))
    b = observation('b', source=1, state_us=1_200_000)
    with pytest.raises(ValueError, match='injected'):
        step(t, [b], [((-1, -2.), (0, 0.))], reference=1_200_000, event='second')
    assert tuple(t.db.iterdump()) == before
    assert t.meta['prediction_sha256'] == first.prediction['commit_sha256']
    monkeypatch.setattr(t, '_ensure_state', original)
    second = step(t, [b], [((-1, -2.), (0, 0.))], reference=1_200_000, event='second')
    assert_state(t, second, [a, b], [((-1, 0.),), ((-1, -2.), (0, 0.))])
    t.close()


@pytest.mark.parametrize('kind', ['future', 'withheld', 'sequence', 'source_alias', 'changed_support', 'old_time'])
def test_invalid_events_leave_durable_output_intact(tmp_path, kind):
    t = tracker(tmp_path/'invalid.db')
    a = observation('a')
    first = step(t, [a], [((-1, 0.),)])
    b = observation('b', source=1, state_us=1_200_000)
    rescored, reference = (), 1_200_000
    if kind == 'future':
        b = replace(b, node=replace(b.node, arrival_us=1_500_000))
    elif kind == 'withheld':
        b = observation('b', source=1)
    elif kind == 'sequence':
        b = replace(b, sequence_id='other')
    elif kind == 'source_alias':
        b = replace(b, node=replace(b.node, source_id=0, frame_id=a.node.frame_id))
    elif kind == 'changed_support':
        rescored = ((0, ((-1, 0.), (0, 1.))),)
    else:
        reference = 1_000_000
    before = tuple(t.db.iterdump())
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        step(t, [b], [((-1, 0.), (0, 0.))], reference=reference, event='invalid', rescored_rows=rescored)
    assert tuple(t.db.iterdump()) == before
    assert t.meta['prediction_sha256'] == first.prediction['commit_sha256']
    t.close()


def test_disk_tampering_or_wrong_expected_head_rejected(tmp_path):
    path = tmp_path/'sealed.db'
    t = tracker(path)
    first = step(t, [observation('a')], [((-1, 0.),)])
    file_hash = t.close()
    with pytest.raises(ValueError, match='head'):
        PersistentForestTracker.open(path, expected_database_sha256=file_hash, expected_prediction_sha256='f'*64)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE states SET payload=?', (canonical({'fake': True}),))
    with pytest.raises(ValueError, match='file hash'):
        PersistentForestTracker.open(path, expected_database_sha256=file_hash,
            expected_prediction_sha256=first.prediction['commit_sha256'])


def test_prefix_capacity_failure_does_not_delete_the_old_unresolved_cover(tmp_path):
    t = tracker(tmp_path/'capacity.db', storage={'max_prefix_nodes': 2}, expansion_budget=0)
    first = step(t, [observation('a')], [((-1, 0.),)])
    assert first.audit['frontier'][0]['depth'] == 0
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='prefix storage exhausted'):
        step(t, [observation('b', source=1, state_us=1_200_000)], [((-1, 0.), (0, 0.))],
             reference=1_200_000, event='second')
    assert tuple(t.db.iterdump()) == before
    t.close()


def test_empty_output_and_live_configuration_protection(tmp_path):
    t = tracker(tmp_path/'empty.db')
    empty = step(t)
    assert empty.prediction['predictions'] == [] and empty.audit['decision']['risk_bound'] == 0.
    t.config = replace(t.config, max_observations=1)
    with pytest.raises(ValueError, match='configuration changed'):
        step(t, reference=1_200_000, event='changed')
    t.close()


@pytest.mark.parametrize('seed', range(4))
def test_rescored_prefix_weights_and_new_suffix_have_complete_joint_mass(tmp_path, seed):
    rng = np.random.default_rng(seed+100)
    t = tracker(tmp_path/'rescore.db', active_limit=2, expansion_budget=6)
    old = tuple(observation(str(i), i*.1, source=i % 2, state_us=1_000_000+i*10_000) for i in range(3))
    old_rows = tuple(tuple((p, float(rng.normal())) for p in range(-1, i)) for i in range(3))
    first = step(t, old, old_rows, reference=1_020_000)
    current = observation('current', source=1, state_us=1_200_000)
    rows = tuple(tuple((p, float(rng.normal())) for p in range(-1, i)) for i in range(4))
    result = step(t, [current], rows[3:], reference=1_200_000, event='rescore',
                  rescored_rows=tuple(enumerate(rows[:3])))
    factors = ForestFactors(tuple(o.node for o in old)+(current.node,), rows)
    probabilities, _, z = brute(factors)
    retained = []
    for a in result.audit['active']:
        parents = t.parents(a['handle'])
        assert a['log_weight'] == pytest.approx(factors.log_weight(parents), abs=1e-12)
        retained.append(parents)
    assert 1-sum(probabilities[p] for p in retained) <= result.audit['eta_upper']+1e-12
    assert math.log(z) <= result.audit['log_partition_upper']+1e-12
    assert_state(t, result, old+(current,), rows)
    assert first.prediction_json == t.db.execute('SELECT prediction FROM events WHERE ordinal=0').fetchone()[0]
    t.close()


def test_unsupported_finite_window_limits_are_not_silently_ignored():
    with pytest.raises(ValueError, match='not used'):
        PersistentForestConfig(state=ForestTrackingConfig(max_nodes=1))


def test_batch_cap_fails_before_any_persistent_mutation(tmp_path):
    t = tracker(tmp_path/'batch.db', storage={'max_new_observations': 1})
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='invalid persistent'):
        step(t, [observation('a'), observation('b', source=1)], [((-1, 0.),), ((-1, 0.), (0, 0.))])
    assert tuple(t.db.iterdump()) == before
    t.close()


def test_parent_capacity_rejects_instead_of_silently_cutting_support(tmp_path):
    t = tracker(tmp_path/'parents.db', parent_limit=1)
    raw = tuple(observation(str(i), source=i % 2, state_us=1_000_000+i*10_000) for i in range(3))
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='candidate capacity'):
        step(t, raw, [((-1, 0.),), ((-1, 0.),), ((-1, 0.), (0, 0.), (1, 0.))], reference=1_020_000)
    assert tuple(t.db.iterdump()) == before
    t.close()
