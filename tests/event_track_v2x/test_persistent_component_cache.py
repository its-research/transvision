import json
from argparse import Namespace

import pytest

from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from tools.event_track_v2x.run_component_persistent_forest_v2 import run as run_component_val
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig, PersistentComponentTracker
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources


@pytest.mark.parametrize('checkpoint,sha', [(None, 'a'*64), ('missing-checkpoint', None)])
def test_component_cli_requires_checkpoint_path_and_hash_together(checkpoint, sha):
    with pytest.raises(ValueError, match='supplied together'):
        run_component_val(Namespace(checkpoint=checkpoint, checkpoint_sha256=sha))


def test_formal_v2_stream_durable_receipts_component_outputs_and_reopen(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    t = PersistentComponentTracker(tmp_path/'stream.db', sequence_id=rows[0]['sequence_id'])
    stream = PersistentForestCacheStream(cache, t, GeometryForestScorer(), origin_us=rows[0]['box_reference_timestamp_us'])
    deliveries = []
    for side in ('vehicle-side', 'infrastructure-side'):
        entry, _ = cache.index[(rows[0]['sequence_id'], side, rows[0]['vehicle_frame'])]
        deliveries.append(CacheDelivery(rows[0]['sequence_id'], side, rows[0]['vehicle_frame'], 1_100_000, json.loads(entry)['frame_sha256']))
    result = stream.step(deliveries, frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000)
    assert result.tracking_audit['component_allocation_integrated'] and result.ingestion_audit['new_observations'] == 4
    file_sha = t.close()
    t = PersistentComponentTracker.open(tmp_path/'stream.db', expected_database_sha256=file_sha,
        expected_prediction_sha256=result.prediction['commit_sha256'])
    resumed = PersistentForestCacheStream(cache, t, GeometryForestScorer(), origin_us=rows[0]['box_reference_timestamp_us'])
    assert resumed.step(deliveries, frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000) == result
    repeated = resumed.step(deliveries, frame_id='second', event_id='second', reference_us=1_200_000, decision_us=1_300_000)
    assert repeated.ingestion_audit['new_observations'] == 0 and t.n == 4
    t.close()


def test_component_v2_replay_uses_original_predictions_and_sealed_database_heads(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    result = replay_rows(cache, rows, tmp_path/'replay', PersistentComponentConfig())
    assert result['completed_frames'] == 2 and result['persistent_component_allocation']
    assert result['geometry_development_baseline'] and not result['trained_paper_method']
    for sequence, head in result['sequence_heads'].items():
        t = PersistentComponentTracker.open(tmp_path/'replay'/head['database'],
            expected_database_sha256=head['database_sha256'], expected_prediction_sha256=head['prediction_sha256'])
        assert t.sequence_id == sequence
        t.close()


def test_component_cache_failure_rolls_back_receipts_and_graph_together(replay_inputs, tmp_path, monkeypatch):
    cache, rows = replay_inputs
    t = PersistentComponentTracker(tmp_path/'atomic.db', sequence_id=rows[0]['sequence_id'])
    stream = PersistentForestCacheStream(cache, t, GeometryForestScorer(), origin_us=rows[0]['box_reference_timestamp_us'])
    entry, _ = cache.index[(rows[0]['sequence_id'], 'vehicle-side', rows[0]['vehicle_frame'])]
    d = CacheDelivery(rows[0]['sequence_id'], 'vehicle-side', rows[0]['vehicle_frame'], 1_100_000, json.loads(entry)['frame_sha256'])
    before = tuple(t.db.iterdump())
    monkeypatch.setattr(t.store, 'save_kernel', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('state commit failed')))
    with pytest.raises(RuntimeError, match='state commit failed'):
        stream.step([d], frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000)
    assert tuple(t.db.iterdump()) == before and t.n == 0
    t.close()
