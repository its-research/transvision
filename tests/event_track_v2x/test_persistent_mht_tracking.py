"""Independent raw-parent enumeration and transactional V2 MHT checks."""
from dataclasses import replace
import itertools
import json
import math

import numpy as np
import pytest

from tools.event_track_v2x.persistent_mht_tracking import PersistentMHTConfig, PersistentMHTTracker
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream
from test_forest_tracking import observation
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources


def advance(t, obs=(), rows=(), *, reference=1000000, event='first'):
    return t.step(obs, rows, reference_us=reference, decision_us=reference+100000,
                  frame_id=event, event_id=event)


def tracker(tmp_path, width=4, **kwargs):
    return PersistentMHTTracker(tmp_path/'mht.db', sequence_id='0003',
        config=PersistentMHTConfig(state=ForestTrackingConfig(active_limit=width), **kwargs))


def roots(t, handle):
    parents = t.parents(handle)
    values = []
    for i, p in enumerate(parents):
        values.append(i if p < 0 else values[p])
    return tuple(values)


def oracle(beam, old_observations, new_observations, rows, width):
    """Enumerate raw edges, reject same-slot collisions, sum physical aliases.

    No assignment solver, tracker grouping or shared logsum helper is used.
    This tests exact extension of the retained histories, NOT resurrecting
    histories excluded at a previous scan.
    """
    all_obs = old_observations + new_observations
    buckets = {}
    for previous, score in beam.items():
        for edges in itertools.product(*rows):
            assignment = list(previous)
            valid = True
            for i, (p, _) in enumerate(edges, len(previous)):
                identity = i if p < 0 else assignment[p]
                slot = all_obs[i].node.source_id, all_obs[i].node.frame_id
                if any((all_obs[j].node.source_id, all_obs[j].node.frame_id) == slot
                       and assignment[j] == identity for j in range(i)):
                    valid = False
                    break
                assignment.append(identity)
            if valid:
                buckets.setdefault(tuple(assignment), []).append(score + sum(w for _, w in edges))
    result = {}
    for assignment, scores in buckets.items():
        maximum = max(scores)
        result[assignment] = maximum + math.log(sum(math.exp(w-maximum) for w in scores))
    return dict(sorted(result.items(), key=lambda item: -item[1])[:width])


@pytest.mark.parametrize('width', [1, 2, 4, 16])
@pytest.mark.parametrize('seed', range(5))
def test_three_scans_match_independent_raw_history_oracle(tmp_path, width, seed):
    t = tracker(tmp_path, width)
    rng = np.random.default_rng(seed)
    expected, previous = {(): 0.}, []
    try:
        for scan in range(3):
            time = 1000000 + scan*200000
            obs = [observation(f'{scan}-{j}', scan*.2+j*3, index=j,
                               source=scan % 2, state_us=time) for j in range(2)]
            rows = [tuple((p, float(rng.normal())) for p in range(-1, len(previous)+j))
                    for j in range(2)]
            expected = oracle(expected, previous, obs, rows, width)
            result = advance(t, obs, rows, reference=time, event=str(scan))
            actual = {roots(t, h): t._weight(h) for h in t.active}
            assert actual.keys() == expected.keys()
            for key in expected:
                assert actual[key] == pytest.approx(expected[key], abs=1e-11)
            assert roots(t, result.audit['output_handle']) == max(expected, key=expected.get)
            assert len(result.audit['branches']) == len(actual)
            assert not result.audit['restored_ancestors']
            assert result.audit['global_scan_mht']
            assert 'eta_upper' not in result.audit and result.audit['decision']['risk_bound'] is None
            previous.extend(obs)
    finally:
        t.close()


def test_later_evidence_reselects_surviving_old_identity_without_rewriting_output(tmp_path):
    t = tracker(tmp_path, 2)
    first = advance(t, [observation('a'), observation('b', 10, index=1)], [[(-1, 0)], [(-1, 0)]])
    second = advance(t, [observation('c', .1, source=1, state_us=1200000),
                         observation('d', 10.1, source=1, state_us=1200000, index=1)],
                     [[(-1, -20), (0, 0), (1, -.1)], [(-1, -20), (0, -.1), (1, 0)]],
                     reference=1200000, event='second')
    assert roots(t, second.audit['output_handle']) == (0, 1, 0, 1)
    surviving = set(t.active)
    # Under the previously second-best history, b and c are aliases of one root;
    # under the previous MAP history they are DIFFERENT competing roots.
    third = advance(t, [observation('e', .2, state_us=1400000)],
                    [[(-1, -20), (1, 10), (2, 10)]], reference=1400000, event='third')
    chosen = third.audit['output_handle']
    assert roots(t, chosen) == (0, 1, 1, 0, 1)
    assert t.ancestor(chosen, 4) in surviving
    assert t.ancestor(chosen, 4) != second.audit['output_handle']
    # Conditional states differ; no averaged cross-identity state is used.
    assert len({b['state_sha256'] for b in third.audit['branches']}) == 2
    assert json.loads(t.db.execute("SELECT prediction FROM events WHERE event_id='second'").fetchone()[0]) == second.prediction
    assert third.prediction['previous_commit_sha256'] == second.prediction['commit_sha256']
    assert first.prediction['predictions'][0]['score'] == .8
    t.close()


def test_pruned_history_cannot_be_resurrected(tmp_path):
    t = tracker(tmp_path, 1)
    advance(t, [observation('a'), observation('b', 10, index=1)], [[(-1, 0)], [(-1, 0)]])
    second = advance(t, [observation('c', source=1, state_us=1200000)],
                     [[(-1, -20), (0, 0), (1, -.1)]], reference=1200000, event='second')
    old = second.audit['output_handle']
    last = advance(t, [observation('d', state_us=1400000)],
                   [[(-1, -20), (1, 10), (2, 10)]], reference=1400000, event='last')
    assert t.ancestor(last.audit['output_handle'], 3) == old
    assert not last.audit['archived_prefixes_used_for_recovery']
    t.close()


@pytest.mark.parametrize('limit', ['max_assignment_solves', 'max_assignment_frontier', 'max_assignment_matrix_cells',
                                 'max_generated_candidates'])
def test_budget_exhaustion_is_atomic(tmp_path, limit):
    t = tracker(tmp_path, 4, **{limit: 4 if limit == 'max_assignment_matrix_cells' else 1})
    first = advance(t, [observation('a'), observation('a2', index=1)], [[(-1, 0)], [(-1, 0)]])
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='budget|capacity'):
        advance(t, [observation('b', state_us=1200000), observation('c', state_us=1200000, index=1)],
                [[(-1, -2), (0, 0), (1, -.1)], [(-1, -2), (0, -.1), (1, 0)]], reference=1200000, event='failed')
    assert tuple(t.db.iterdump()) == before
    assert t.n == 2 and t.meta['prediction_sha256'] == first.prediction['commit_sha256']
    t.close()


def test_state_failure_rolls_back_then_duplicate_and_reopen_are_identical(tmp_path, monkeypatch):
    t = tracker(tmp_path)
    obs, rows = [observation('a')], [[(-1, 0)]]
    before, original = tuple(t.db.iterdump()), t._ensure_state
    monkeypatch.setattr(t, '_ensure_state', lambda *a: (_ for _ in ()).throw(RuntimeError('state failure')))
    with pytest.raises(RuntimeError, match='state failure'):
        advance(t, obs, rows)
    assert tuple(t.db.iterdump()) == before
    monkeypatch.setattr(t, '_ensure_state', original)
    first = advance(t, obs, rows)
    assert advance(t, obs, rows) == first
    sha = t.close()
    t = PersistentMHTTracker.open(t.path, expected_database_sha256=sha,
                                  expected_prediction_sha256=first.prediction['commit_sha256'])
    assert advance(t, obs, rows) == first
    later = advance(t, reference=1200000, event='empty')
    assert later.prediction['predictions'][0]['track_id'] == first.prediction['predictions'][0]['track_id']
    t.close()


def test_late_observation_current_state_replay_and_causality(tmp_path):
    t = tracker(tmp_path)
    first = advance(t, [observation('a', 1)], [[(-1, 0)]])
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='future'):
        advance(t, [observation('future', state_us=2000000)], [[(-1, 0)]], reference=1200000, event='bad')
    assert tuple(t.db.iterdump()) == before
    late = observation('late', 0, source=1, state_us=900000, arrival_us=1200000)
    second = advance(t, [late], [[(-1, -20), (0, 0)]], reference=1200000, event='late')
    assert second.prediction['predictions'][0]['track_id'] == first.prediction['predictions'][0]['track_id']
    assert 0 < second.prediction['predictions'][0]['mean'][0] < 1
    assert first.prediction['predictions'][0]['mean'][0] == 1
    with pytest.raises(ValueError, match='rescoring'):
        t.step([], [], rescored_rows=[(0, [(-1, 2)])])
    t.close()


def test_interleaved_source_frames_fail_without_partial_insertion(tmp_path):
    t = tracker(tmp_path)
    before = tuple(t.db.iterdump())
    obs = [observation('a'), observation('b', source=1), observation('c', index=1)]
    with pytest.raises(ValueError, match='contiguous'):
        advance(t, obs, [[(-1, 0)]]*3)
    assert tuple(t.db.iterdump()) == before
    t.close()


def test_detection_cache_v2_stream_and_duplicate_receipts_resume(replay_inputs, tmp_path, monkeypatch):
    cache, rows = replay_inputs
    row = rows[0]
    t = tracker(tmp_path)
    stream = PersistentForestCacheStream(cache, t, GeometryForestScorer(), origin_us=row['box_reference_timestamp_us'])
    entry, _ = cache.index[(row['sequence_id'], 'vehicle-side', row['vehicle_frame'])]
    delivery = CacheDelivery(row['sequence_id'], 'vehicle-side', row['vehicle_frame'], 1100000,
                             json.loads(entry)['frame_sha256'])
    first = stream.step([delivery], frame_id='first', event_id='first', reference_us=1000000, decision_us=1100000)
    assert all(p['class_label'] == 'car' for p in first.prediction['predictions'])
    assert first.ingestion_audit['new_observations'] == 2
    sha = t.close()
    t = PersistentMHTTracker.open(t.path, expected_database_sha256=sha,
                                  expected_prediction_sha256=first.prediction['commit_sha256'])
    stream = PersistentForestCacheStream(cache, t, GeometryForestScorer(), origin_us=row['box_reference_timestamp_us'])
    monkeypatch.setattr(cache, 'load_arrived', lambda *a: pytest.fail('duplicate payload read'))
    assert stream.step([delivery], frame_id='first', event_id='first', reference_us=1000000, decision_us=1100000) == first
    later = stream.step([replace(delivery, arrival_us=1300000)], frame_id='later', event_id='later',
                        reference_us=1200000, decision_us=1300000)
    assert later.ingestion_audit['new_observations'] == 0
    t.close()
