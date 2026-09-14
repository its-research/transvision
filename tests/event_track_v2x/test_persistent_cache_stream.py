from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from tools.event_track_v2x.build_detection_cache_v2 import build_cache
from transvision.models.event_track_v2x.detection_cache_v2 import DetectionCacheV2, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery, VerifiedForestCache
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer, LearnedForestScorer
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig, append_parent_support
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream, RowContextScorer
from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig, PersistentForestTracker
from test_detection_cache_v2 import sources
from test_forest_tracking import observation


@pytest.fixture
def stream(sources, tmp_path):
    roots, calibration, inputs, output = sources
    sha = build_cache(roots, calibration, sha_file(calibration), inputs, output)
    cache = VerifiedForestCache(output, sha)
    tracker = PersistentForestTracker(tmp_path/'stream.db', sequence_id='0003',
        config=PersistentForestConfig(state=ForestTrackingConfig(window_us=100_000, max_model_regret=1.)))
    result = PersistentForestCacheStream(cache, tracker, GeometryForestScorer(), origin_us=1_000_000)
    yield result
    result.tracker.close()


def delivery(stream, side='vehicle-side', arrival=1_100_000):
    entry = json.loads(stream.cache.index[('0003', side, '000120')][0])
    return CacheDelivery('0003', side, '000120', arrival, entry['frame_sha256'])


def step(stream, deliveries=(), *, reference=1_100_000, event='first'):
    return stream.step(deliveries, frame_id=event, reference_us=reference, decision_us=reference, event_id=event)


def test_v2_stream_survives_window_expiry_and_resumes_first_receipts(stream, monkeypatch):
    d = delivery(stream)
    first = step(stream, [d])
    assert first.ingestion_audit['new_observations'] == 2
    assert all(p['class_label'] == 'car' for p in first.prediction['predictions'])
    newer = step(stream, reference=5_000_000, event='much-later')
    assert newer.prediction['predictions'] == []
    t = stream.tracker
    file_sha = t.close()
    reopened = PersistentForestTracker.open(t.path, expected_database_sha256=file_sha,
        expected_prediction_sha256=newer.prediction['commit_sha256'])
    resumed = PersistentForestCacheStream(stream.cache, reopened, GeometryForestScorer(), origin_us=1_000_000)
    stream.tracker = reopened  # Fixture owns the single live connection.
    monkeypatch.setattr(stream.cache, 'load_arrived', lambda *a: pytest.fail('old frame reloaded'))
    again = step(resumed, [replace(d, arrival_us=5_100_000)], reference=5_100_000, event='again')
    assert again.ingestion_audit['new_observations'] == 0
    assert again.ingestion_audit['duplicate_deliveries'][0]['first_arrival_us'] == d.arrival_us
    assert reopened.n == 2 and reopened.db.execute('SELECT count(*) FROM cache_receipts').fetchone()[0] == 1
    assert step(resumed, [d]) == first


def test_frame_receipts_rollback_with_factors_and_states(stream, monkeypatch):
    before = tuple(stream.tracker.db.iterdump())
    original = stream.tracker._ensure_state
    monkeypatch.setattr(stream.tracker, '_ensure_state', lambda *a: (_ for _ in ()).throw(RuntimeError('state failure')))
    with pytest.raises(RuntimeError, match='state failure'):
        step(stream, [delivery(stream)])
    assert tuple(stream.tracker.db.iterdump()) == before
    monkeypatch.setattr(stream.tracker, '_ensure_state', original)
    result = step(stream, [delivery(stream)])
    assert result.ingestion_audit['new_observations'] == 2
    assert stream.tracker.db.execute('SELECT count(*) FROM cache_receipts').fetchone()[0] == 1


@pytest.mark.parametrize('kind', ['future', 'hash', 'earlier_information', 'receipt_cap'])
def test_invalid_receipt_before_payload_read(stream, monkeypatch, kind):
    d = delivery(stream)
    if kind == 'future':
        d = replace(d, arrival_us=1_200_000)
    elif kind == 'hash':
        d = replace(d, frame_sha256='f'*64)
    elif kind == 'earlier_information':
        d = replace(d, arrival_us=1_099_999)
    else:
        stream = PersistentForestCacheStream(stream.cache, stream.tracker, GeometryForestScorer(),
            origin_us=1_000_000, max_receipts=1)
    monkeypatch.setattr(DetectionCacheV2, 'load', lambda *a: pytest.fail('invalid payload read'))
    deliveries = [d, delivery(stream, 'infrastructure-side')] if kind == 'receipt_cap' else [d]
    with pytest.raises(ValueError):
        step(stream, deliveries)
    assert stream.tracker.n == 0


def test_duplicate_hash_and_receipt_time_cannot_change(stream):
    d = delivery(stream, arrival=1_150_000)
    step(stream, [d], reference=1_150_000)
    with pytest.raises(ValueError, match='first receipt'):
        step(stream, [replace(d, arrival_us=1_120_000)], reference=1_200_000, event='bad')
    with pytest.raises(ValueError, match='withheld'):
        step(stream, [delivery(stream, 'infrastructure-side')], reference=1_200_000, event='withheld')
    with pytest.raises(ValueError, match='protocol differs'):
        PersistentForestCacheStream(stream.cache, stream.tracker, GeometryForestScorer(), origin_us=0)


def test_row_context_geometry_support_matches_original_gating_across_batches(tmp_path):
    t = PersistentForestTracker(tmp_path/'gate.db', sequence_id='0003')
    scorer = RowContextScorer(t, GeometryForestScorer())
    initial = tuple(observation(str(i), float(i), source=i % 2, index=i//2) for i in range(8))
    rows, _ = scorer.rows(initial, 1_100_000)
    expected = append_parent_support(initial, (), t.config.state)
    assert tuple(tuple(p for p, _ in r) for r in rows) == expected
    t.step(initial, rows, frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000)
    next_obs = tuple(observation('next-'+str(i), .2+i, source=i % 2, state_us=1_200_000) for i in range(2))
    added, contexts = scorer.rows(next_obs, 1_300_000)
    assert tuple(tuple(p for p, _ in r) for r in added) == append_parent_support(initial+next_obs, expected, t.config.state)[8:]
    assert all(len(c) <= t.config.state.parent_limit+1 for c in contexts)
    t.close()


def test_untrained_frozen_neural_row_context_executes_without_global_token_growth(tmp_path):
    torch.manual_seed(1337)
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0.).eval()
    model.requires_grad_(False)
    t = PersistentForestTracker(tmp_path/'neural.db', sequence_id='0003', config=PersistentForestConfig(
        state=ForestTrackingConfig(parent_limit=2, active_limit=1, expansion_budget=8)))
    scorer = RowContextScorer(t, LearnedForestScorer(model, max_nodes=3, max_pairs=9))
    for i in range(8):
        time = 1_000_000+100_000*i
        raw = (observation(str(i), i*.1, state_us=time),)
        rows, contexts = scorer.rows(raw, time+100_000)
        assert len(contexts[0]) <= 3
        t.step(raw, rows, frame_id=str(i), event_id=str(i), reference_us=time, decision_us=time+100_000,
               scorer_binding=scorer.binding)
    assert t.n == 8 > scorer.scorer.options['max_nodes']
    t.close()
