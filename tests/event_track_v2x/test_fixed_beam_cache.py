from argparse import Namespace
import json

import pytest

from tools.event_track_v2x.run_fixed_beam_v2 import run
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.persistent_beam_tracking import PersistentBeamConfig, PersistentBeamTracker, PersistentRankedBeamTracker
from transvision.models.event_track_v2x.persistent_joint_beam import PersistentJointBeamTracker
from transvision.models.event_track_v2x.persistent_class_bound_beam import PersistentClassBoundBeamTracker
from transvision.models.event_track_v2x.persistent_slot_bound_beam import PersistentSlotBoundBeamTracker
from transvision.models.event_track_v2x.persistent_sparse_slot_bound_beam import PersistentSparseSlotBoundBeamTracker
from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources


@pytest.mark.parametrize('tracker_type', [PersistentBeamTracker, PersistentRankedBeamTracker, PersistentJointBeamTracker, PersistentClassBoundBeamTracker, PersistentSlotBoundBeamTracker, PersistentSparseSlotBoundBeamTracker])
def test_beam_v2_predictions_and_database_heads_keep_the_existing_format(replay_inputs, tmp_path, tracker_type):
    cache, rows = replay_inputs
    result = replay_rows(cache, rows, tmp_path/'beam-run', tracker_type.CONFIG_TYPE())
    assert result['completed_frames'] == 2 and result['irreversible_beam_enabled']
    assert result['single_class_ranking_bound_enabled'] is (tracker_type in (PersistentClassBoundBeamTracker, PersistentSlotBoundBeamTracker, PersistentSparseSlotBoundBeamTracker))
    assert result['source_frame_assignment_bound_enabled'] is (tracker_type in (PersistentSlotBoundBeamTracker, PersistentSparseSlotBoundBeamTracker))
    assert result['sparse_assignment_decomposition_enabled'] is (tracker_type is PersistentSparseSlotBoundBeamTracker)
    assert not result['persistent_component_allocation'] and not result['paper_eligible']
    for sequence, head in result['sequence_heads'].items():
        t = tracker_type.open(tmp_path/'beam-run'/head['database'], expected_database_sha256=head['database_sha256'],
                                      expected_prediction_sha256=head['prediction_sha256'])
        assert t.sequence_id == sequence
        t.close()


def test_beam_cache_atomic_failure_and_repeated_receipt_after_reopen(replay_inputs, tmp_path, monkeypatch):
    cache, rows = replay_inputs
    row = rows[0]
    t = PersistentBeamTracker(tmp_path/'cache.db', sequence_id=row['sequence_id'])
    stream = PersistentForestCacheStream(cache, t, GeometryForestScorer(), origin_us=row['box_reference_timestamp_us'])
    entry, _ = cache.index[(row['sequence_id'], 'vehicle-side', row['vehicle_frame'])]
    delivery = CacheDelivery(row['sequence_id'], 'vehicle-side', row['vehicle_frame'], 1_100_000, json.loads(entry)['frame_sha256'])
    before = tuple(t.db.iterdump())
    save = t.store.save_kernel
    monkeypatch.setattr(t.store, 'save_kernel', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('commit failure')))
    with pytest.raises(RuntimeError, match='commit failure'):
        stream.step([delivery], frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000)
    assert tuple(t.db.iterdump()) == before and t.n == 0
    monkeypatch.setattr(t.store, 'save_kernel', save)
    first = stream.step([delivery], frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000)
    sha = t.close()
    t = PersistentBeamTracker.open(tmp_path/'cache.db', expected_database_sha256=sha,
                                  expected_prediction_sha256=first.prediction['commit_sha256'])
    resumed = PersistentForestCacheStream(cache, t, GeometryForestScorer(), origin_us=row['box_reference_timestamp_us'])
    assert resumed.step([delivery], frame_id='first', event_id='first', reference_us=1_000_000, decision_us=1_100_000) == first
    t.close()


@pytest.mark.parametrize('checkpoint,sha', [(None, 'a'*64), ('missing', None)])
def test_beam_cli_requires_a_checkpoint_identity_pair(checkpoint, sha):
    with pytest.raises(ValueError, match='supplied together'):
        run(Namespace(width=4, checkpoint=checkpoint, checkpoint_sha256=sha))


def test_beam_cli_rejects_partial_schedule_before_any_output(replay_inputs, tmp_path):
    _, rows = replay_inputs
    schedule = tmp_path/'schedule.json'
    schedule.write_bytes(canonical(dict(kind='spd_official_validation_prediction_schedule_v1',
        split_sha256='4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3',
        contains_ground_truth=False, contains_system_error_offset=False, frames=rows)))
    with pytest.raises(ValueError, match='full validation coverage'):
        run(Namespace(width=1, checkpoint=None, checkpoint_sha256=None, schedule=schedule,
                      schedule_sha256=sha_file(schedule), output=tmp_path/'must-not-exist'))
    assert not (tmp_path/'must-not-exist').exists()
