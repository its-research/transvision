"""Real V2-format fixtures, immutable cache receipts and official-schedule gate."""
from argparse import Namespace
from dataclasses import replace
import json

import pytest

from tools.event_track_v2x.run_probabilistic_tracking_v2 import run
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream
from transvision.models.event_track_v2x.persistent_probabilistic_tracking import (
    PersistentProbabilisticConfig, PersistentProbabilisticTracker,
)
from transvision.models.event_track_v2x.tracking_evaluation_v2 import validate_predictions
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources


@pytest.mark.parametrize('update_rule', ['jpda-ci', 'jpda-kalman', 'pkf'])
def test_v2_replay_seals_same_prediction_format_and_records_actual_backend(replay_inputs, tmp_path, update_rule):
    cache, rows = replay_inputs
    output = tmp_path/'replay'
    result = replay_rows(cache, rows, output, PersistentProbabilisticConfig(update_rule=update_rule))
    assert result['completed_frames'] == 2
    assert result['probabilistic_single_history_enabled']
    assert result['probabilistic_update_rule'] == update_rule
    assert not result['irreversible_beam_enabled'] and not result['paper_eligible']
    predictions = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
    schedule = [dict(sequence_id=p['sequence_id'], frame_id=p['frame_id'],
                     box_reference_timestamp_us=p['box_reference_timestamp_us']) for p in predictions]
    assert validate_predictions(predictions, schedule)['frames'] == 2
    for sequence, head in result['sequence_heads'].items():
        t = PersistentProbabilisticTracker.open(output/head['database'],
            expected_database_sha256=head['database_sha256'], expected_prediction_sha256=head['prediction_sha256'])
        assert t.sequence_id == sequence
        t.close()


def test_cache_rollback_then_duplicate_delivery_never_reloads_payload(replay_inputs, tmp_path, monkeypatch):
    cache, rows = replay_inputs
    row = rows[0]
    t = PersistentProbabilisticTracker(tmp_path/'cache.db', sequence_id=row['sequence_id'])
    stream = PersistentForestCacheStream(cache, t, GeometryForestScorer(), origin_us=row['box_reference_timestamp_us'])
    entry, _ = cache.index[(row['sequence_id'], 'vehicle-side', row['vehicle_frame'])]
    delivery = CacheDelivery(row['sequence_id'], 'vehicle-side', row['vehicle_frame'], 1_100_000, json.loads(entry)['frame_sha256'])
    before, save = tuple(t.db.iterdump()), t._save_state
    monkeypatch.setattr(t, '_save_state', lambda *args: (_ for _ in ()).throw(RuntimeError('state failed')))
    with pytest.raises(RuntimeError, match='state failed'):
        stream.step([delivery], frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000)
    assert tuple(t.db.iterdump()) == before
    monkeypatch.setattr(t, '_save_state', save)
    first = stream.step([delivery], frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000)
    sha = t.close()
    t = PersistentProbabilisticTracker.open(t.path, expected_database_sha256=sha,
                                           expected_prediction_sha256=first.prediction['commit_sha256'])
    resumed = PersistentForestCacheStream(cache, t, GeometryForestScorer(), origin_us=row['box_reference_timestamp_us'])
    monkeypatch.setattr(cache, 'load_arrived', lambda *a: pytest.fail('duplicate payload was reloaded'))
    assert resumed.step([delivery], frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000) == first
    newer = resumed.step([replace(delivery, arrival_us=1_300_000)], frame_id='later', event_id='later',
                         reference_us=1_200_000, decision_us=1_300_000)
    assert newer.ingestion_audit['new_observations'] == 0
    assert newer.ingestion_audit['duplicate_deliveries'][0]['first_arrival_us'] == delivery.arrival_us
    t.close()


@pytest.mark.parametrize('checkpoint,sha', [(None, 'a'*64), ('missing', None)])
def test_cli_checkpoint_path_hash_pair_required(checkpoint, sha):
    with pytest.raises(ValueError, match='supplied together'):
        run(Namespace(checkpoint=checkpoint, checkpoint_sha256=sha))


def test_cli_rejects_partial_schedule_before_output(replay_inputs, tmp_path):
    _, rows = replay_inputs
    schedule = tmp_path/'schedule.json'
    schedule.write_bytes(canonical(dict(kind='spd_official_validation_prediction_schedule_v1',
        split_sha256='4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3',
        contains_ground_truth=False, contains_system_error_offset=False, frames=rows)))
    with pytest.raises(ValueError, match='full validation coverage'):
        run(Namespace(checkpoint=None, checkpoint_sha256=None, schedule=schedule,
                      schedule_sha256=sha_file(schedule), output=tmp_path/'no-output'))
    assert not (tmp_path/'no-output').exists()
