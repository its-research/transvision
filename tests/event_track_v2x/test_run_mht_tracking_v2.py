from argparse import Namespace
from dataclasses import replace
import json

import pytest

from tools.event_track_v2x.run_mht_tracking_v2 import replay_mht_rows, run, mht_sources
from tools.event_track_v2x.persistent_mht_tracking import PersistentMHTConfig, PersistentMHTTracker
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.tracking_evaluation_v2 import validate_predictions
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources


def test_two_frame_v2_replay_and_durable_seals(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    output = tmp_path/'run'
    result = replay_mht_rows(cache, rows, output, PersistentMHTConfig())
    assert result['status'] == 'complete' and result['completed_frames'] == 2
    assert result['global_scan_mht'] and not result['paper_eligible']
    assert not result['parameter_training'] and not result['gt_model_inputs']
    assert 'full_official_validation_schedule_completed' not in result
    predictions = [json.loads(v) for v in (output/'predictions.jsonl').read_bytes().splitlines()]
    schedule = [dict(sequence_id=p['sequence_id'], frame_id=p['frame_id'],
                     box_reference_timestamp_us=p['box_reference_timestamp_us']) for p in predictions]
    assert validate_predictions(predictions, schedule)['frames'] == 2
    for scene, head in result['sequence_heads'].items():
        t = PersistentMHTTracker.open(output/head['database'], expected_database_sha256=head['database_sha256'],
                                      expected_prediction_sha256=head['prediction_sha256'])
        assert t.sequence_id == scene
        t.close()
    for filename, key in [('predictions.jsonl', 'predictions_sha256'), ('tracking.jsonl', 'tracking_sha256')]:
        assert sha_file(output/filename) == result[key]
    plan = json.loads((output/'plan.json').read_bytes())
    assert plan['source_sha256'] == mht_sources()
    assert plan['class_scope'] == ['car'] and plan['geometry_baseline'] is True
    times = [json.loads(v) for v in (output/'frame-timings.jsonl').read_bytes().splitlines()]
    for p,t in zip(predictions,times):
        assert t['box_reference_timestamp_us'] == p['box_reference_timestamp_us']
        assert t['decision_timestamp_us'] == p['decision_timestamp_us']
    with pytest.raises(ValueError, match='new ordinary'):
        replay_mht_rows(cache, rows, output, PersistentMHTConfig())


def test_budget_failure_never_creates_success_receipt(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    output = tmp_path/'failed'
    with pytest.raises(ValueError, match='work cap'):
        replay_mht_rows(cache, rows, output, PersistentMHTConfig(state=ForestTrackingConfig(max_replay_operations=1)))
    failure = json.loads((output/'failure.json').read_bytes())
    assert failure['status'] == 'failed' and failure['partial_outputs_not_final_results']
    assert not (output/'receipt.json').exists()


@pytest.mark.parametrize('bad', ['unknown_field', 'nonmonotonic', 'reference', 'empty'])
def test_invalid_schedule_rejected_before_output(replay_inputs, tmp_path, bad):
    cache, rows = replay_inputs
    rows = [dict(r) for r in rows]
    if bad == 'unknown_field': rows[0]['ground_truth'] = []
    elif bad == 'nonmonotonic': rows = rows[::-1]
    elif bad == 'reference': rows[0]['box_reference_timestamp_us'] += 1
    else: rows = []
    with pytest.raises(ValueError): replay_mht_rows(cache, rows, tmp_path/'absent', PersistentMHTConfig())
    assert not (tmp_path/'absent').exists()


def test_cli_does_not_promote_a_subset_to_official_validation(replay_inputs, tmp_path):
    _, rows = replay_inputs
    p = tmp_path/'schedule.json'
    p.write_bytes(canonical(dict(kind='spd_official_validation_prediction_schedule_v1',
        split_sha256='4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3',
        contains_ground_truth=False, contains_system_error_offset=False, frames=rows)))
    with pytest.raises(ValueError, match='full validation coverage'):
        run(Namespace(checkpoint=None, checkpoint_sha256=None, schedule=p, schedule_sha256=sha_file(p)))


@pytest.mark.parametrize('checkpoint,sha', [(None, 'a'*64), ('missing', None)])
def test_cli_checkpoint_requires_both_path_and_hash(checkpoint, sha):
    with pytest.raises(ValueError, match='supplied together'):
        run(Namespace(checkpoint=checkpoint, checkpoint_sha256=sha))


def test_runtime_binds_assignment_library_and_binary(monkeypatch):
    from tools.event_track_v2x import run_mht_tracking_v2 as runner
    from scipy.optimize import _lsap
    import scipy
    monkeypatch.setattr(runner,'runtime_evidence',lambda device:{'device':device})
    result = runner.mht_runtime_evidence('cpu')
    assert result['scipy'] == scipy.__version__
    assert result['assignment_binary_sha256'] == sha_file(_lsap.__file__)
    assert result['sqlite_version']
