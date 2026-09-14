"""Single-history Gaussian baseline: temporal state, anchors and transactions."""
from dataclasses import replace
import json
import math
import sqlite3

import numpy as np
import pytest

from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.jpda_forest_bridge import condition_forest_scan
from transvision.models.event_track_v2x.jpda_marginals import JPDALimits
from transvision.models.event_track_v2x.persistent_probabilistic_tracking import (
    PersistentProbabilisticConfig, PersistentProbabilisticTracker,
)
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.persistent_cache_stream import RowContextScorer
from test_forest_tracking import observation


def advance(t, observations=(), rows=(), *, reference=1_000_000, event='first'):
    return t.step(observations, rows, frame_id=event, reference_us=reference,
                  decision_us=reference+100_000, event_id=event)


@pytest.mark.parametrize('algorithm', ['exact', 'lbp'])
@pytest.mark.parametrize('update', ['jpda-ci', 'jpda-kalman', 'pkf'])
def test_birth_dual_source_and_next_frame_keep_ids_and_audited_semantics(tmp_path, algorithm, update):
    t = PersistentProbabilisticTracker(tmp_path/'track.db', sequence_id='0003',
        config=PersistentProbabilisticConfig(association_algorithm=algorithm, update_rule=update))
    first = advance(t, [observation('a', 0.), observation('b', .2, source=1)],
                    [((-1, -8.),), ((-1, -8.), (0, 0.))])
    saved = first.prediction_json
    assert len(first.prediction['predictions']) == 1
    original_id = first.prediction['predictions'][0]['track_id']
    second = advance(t, [observation('c', .3, state_us=1_200_000)],
                     [((-1, -8.), (0, 0.), (1, 0.))], reference=1_200_000, event='second')
    assert second.prediction['predictions'][0]['track_id'] == original_id
    assert first.prediction_json == saved
    audit = second.audit
    assert audit['kind'] == t.SCHEMA
    assert audit['recovery_enabled'] is False
    assert audit['update_rule'] == update
    assert audit['same_state_time_protocol_as_recoverable'] is False
    assert not any(k in audit for k in ('active', 'frontier', 'eta_upper', 'log_partition_upper', 'restored_ancestors'))
    assert audit['conditional_scans'][0]['log_pair'][0][0] == pytest.approx(math.log(2))
    assert tuple(t.db.execute('SELECT i,root FROM identity_anchors ORDER BY i')) == ((0, 0), (1, 0), (2, 0))
    assert advance(t, reference=5_000_000, event='expired').prediction['predictions'] == []
    assert t.n == 3
    t.close()


def test_birth_score_not_multiplied_by_unmatched_probability(tmp_path):
    t = PersistentProbabilisticTracker(tmp_path/'score.db', sequence_id='0003',
        config=PersistentProbabilisticConfig(association_algorithm='exact'))
    first = advance(t, [observation('a')], [((-1, 0.),)])
    current = [observation('b', -.1, state_us=1_200_000, index=0, score=.31),
               observation('c', .1, state_us=1_200_000, index=1, score=.31)]
    second = advance(t, current, [((-1, -8.), (0, 0.))]*2, reference=1_200_000, event='next')
    original_id = first.prediction['predictions'][0]['track_id']
    newborn = [p for p in second.prediction['predictions'] if p['track_id'] != original_id]
    assert len(newborn) == 1 and newborn[0]['score'] == .31
    assert all(.49 < mass < .51 for mass in second.audit['conditional_scans'][0]['marginals']['right_unmatched'])
    assert len(set(r for _, r in t.db.execute('SELECT i,root FROM identity_anchors WHERE i>=1'))) == 2
    t.close()


def test_symmetric_marginal_bayes_anchor_can_prefer_two_births(tmp_path):
    t = PersistentProbabilisticTracker(tmp_path/'bayes.db', sequence_id='0003',
        config=PersistentProbabilisticConfig(association_algorithm='exact', anchor_decoder='marginal-bayes'))
    advance(t, [observation('a')], [((-1, 0.),)])
    second = advance(t, [observation('b', -.1, state_us=1_200_000, index=0),
                         observation('c', .1, state_us=1_200_000, index=1)],
                     [((-1, -8.), (0, 0.))]*2, reference=1_200_000, event='next')
    marginal = second.audit['conditional_scans'][0]['marginals']
    assert all(marginal['right_unmatched'][j] > marginal['pair'][0][j] for j in range(2))
    assert len(second.prediction['predictions']) == 3
    assert second.audit['anchor_decoder'] == 'marginal-bayes'
    t.close()


def test_streaming_factor_conversion_matches_independent_forest_bridge(tmp_path):
    t = PersistentProbabilisticTracker(tmp_path/'bridge.db', sequence_id='0003',
        config=PersistentProbabilisticConfig(association_algorithm='exact'))
    old = [observation('a'), observation('b', .2, source=1)]
    old_rows = [((-1, -8.),), ((-1, -8.), (0, 0.))]
    advance(t, old, old_rows)
    new = [observation('c', 1., state_us=1_200_000, index=0), observation('d', 2., state_us=1_200_000, index=1)]
    rows = [((-1, 1.), (0, 2.), (1, 3.)), ((-1, 2.), (0, 1.), (1, 4.))]
    bridge = condition_forest_scan(ForestFactors(tuple(o.node for o in old+new), tuple(old_rows+rows)),
                                  (-1, 0), decision_us=1_300_000)
    result = advance(t, new, rows, reference=1_200_000, event='next')
    assert result.audit['conditional_scans'][0]['factors_sha256'] == bridge.factors.digest()
    t.close()


def test_state_failure_rolls_back_raw_factors_identity_states_and_output(tmp_path, monkeypatch):
    t = PersistentProbabilisticTracker(tmp_path/'atomic.db', sequence_id='0003')
    before = tuple(t.db.iterdump())
    original = t._save_state
    def broken(*args):
        original(*args)
        raise RuntimeError('after state write')
    monkeypatch.setattr(t, '_save_state', broken)
    with pytest.raises(RuntimeError, match='after state write'):
        advance(t, [observation('a')], [((-1, 0.),)])
    assert tuple(t.db.iterdump()) == before and t.n == 0
    monkeypatch.setattr(t, '_save_state', original)
    assert advance(t, [observation('a')], [((-1, 0.),)]).prediction['predictions']
    t.close()


def test_source_scan_reordering_is_not_silently_fixed(tmp_path):
    t = PersistentProbabilisticTracker(tmp_path/'order.db', sequence_id='0003')
    obs = [observation('a', index=0), observation('b', source=1), observation('c', index=1)]
    with pytest.raises(ValueError, match='interleaved'):
        advance(t, obs, [((-1, 0.),)]*3)
    assert t.n == 0
    t.close()


def test_future_and_rescore_inputs_fail_without_commit(tmp_path):
    t = PersistentProbabilisticTracker(tmp_path/'future.db', sequence_id='0003')
    with pytest.raises(ValueError, match='future'):
        advance(t, [observation('a', arrival_us=1_200_000)], [((-1, 0.),)])
    with pytest.raises(ValueError, match='rescore'):
        t.step([], [], frame_id='x', reference_us=1_000_000, decision_us=1_100_000, event_id='x',
               rescored_rows=[(0, ((-1, 1.),))])
    assert t.n == 0
    t.close()


def test_late_and_ahead_source_projection_never_reverses_collapsed_prior(tmp_path):
    t = PersistentProbabilisticTracker(tmp_path/'late.db', sequence_id='0003')
    first = advance(t, [observation('a')], [((-1, 0.),)])
    late = observation('late', .2, source=1, state_us=900_000, arrival_us=1_250_000)
    advance(t, [late], [((-1, -8.), (0, 0.))], reference=1_200_000, event='late')
    ahead = observation('ahead', .3, state_us=1_450_000, arrival_us=1_480_000)
    result = advance(t, [ahead], [((-1, -8.), (0, 0.), (1, 0.))], reference=1_400_000, event='ahead')
    assert result.audit['conditional_scans'][0]['negative_raw_projection_count'] == 1
    assert result.prediction['predictions'][0]['track_id'] == first.prediction['predictions'][0]['track_id']
    assert t.db.execute('SELECT reference_us FROM gaussian_tracks WHERE root=0').fetchone()[0] == 1_400_000
    t.close()


def test_normal_reopen_preserves_idempotence_and_rejects_state_tampering(tmp_path):
    path = tmp_path/'reopen.db'
    t = PersistentProbabilisticTracker(path, sequence_id='0003')
    first = advance(t, [observation('a')], [((-1, 0.),)])
    file_hash = t.close()
    t = PersistentProbabilisticTracker.open(path, expected_database_sha256=file_hash,
                                            expected_prediction_sha256=first.prediction['commit_sha256'])
    assert advance(t, [observation('a')], [((-1, 0.),)]) == first
    t.close()
    db = sqlite3.connect(path)
    payload = json.loads(db.execute('SELECT payload FROM gaussian_tracks').fetchone()[0])
    payload['mean'][0] += 1
    db.execute('UPDATE gaussian_tracks SET payload=?', (json.dumps(payload),))
    db.commit(); db.close()
    with pytest.raises(ValueError, match='differ'):
        PersistentProbabilisticTracker.open(path, expected_database_sha256=sha_file(path),
                                           expected_prediction_sha256=first.prediction['commit_sha256'])


def test_small_inference_budget_fails_atomically(tmp_path):
    config = PersistentProbabilisticConfig(association_algorithm='exact', inference=JPDALimits(max_dp_states=1))
    t = PersistentProbabilisticTracker(tmp_path/'limit.db', sequence_id='0003', config=config)
    advance(t, [observation('a')], [((-1, 0.),)])
    before = tuple(t.db.iterdump())
    with pytest.raises(ValueError, match='budget'):
        advance(t, [observation('b', state_us=1_200_000)], [((-1, 0.), (0, 1.))], reference=1_200_000, event='next')
    assert tuple(t.db.iterdump()) == before
    t.close()


@pytest.mark.parametrize('update_rule', ['jpda-ci', 'jpda-kalman', 'pkf'])
def test_sixty_frame_predictions_match_across_mid_sequence_reopen(tmp_path, update_rule):
    outcomes = []
    for split in (False, True):
        t = PersistentProbabilisticTracker(tmp_path/('split.db' if split else 'continuous.db'),
            sequence_id='0003', config=PersistentProbabilisticConfig(update_rule=update_rule))
        scorer = GeometryForestScorer(birth_logit=-8.)
        row_scorer = RowContextScorer(t, scorer)
        predictions = []
        for frame in range(60):
            stamp = 1_000_000+frame*100_000
            observations = [observation(f'{frame}-left', frame*.02, state_us=stamp, index=0),
                            observation(f'{frame}-right', 2.-frame*.02, state_us=stamp, index=1)]
            rows, _ = row_scorer.rows(observations, stamp+100_000)
            commit = t.step(observations, rows, frame_id=str(frame), reference_us=stamp,
                decision_us=stamp+100_000, event_id=str(frame), scorer_binding=row_scorer.binding)
            predictions.append(commit.prediction_json)
            assert commit.audit['state_updates'] == sum(commit.audit['state_work_breakdown'].values())
            roots = [r for r, in t.db.execute('SELECT root FROM identity_anchors WHERE i>=?', (2*frame,))]
            assert len(roots) == len(set(roots)) == 2
            if split and frame == 29:
                sha = t.close()
                t = PersistentProbabilisticTracker.open(t.path, expected_database_sha256=sha,
                    expected_prediction_sha256=commit.prediction['commit_sha256'])
                row_scorer = RowContextScorer(t, scorer)
        assert t.n == 120
        assert t.db.execute('SELECT count(*) FROM events').fetchone()[0] == 60
        t.close()
        outcomes.append(predictions)
    assert outcomes[0] == outcomes[1]
