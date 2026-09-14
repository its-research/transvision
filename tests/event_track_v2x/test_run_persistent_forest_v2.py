from dataclasses import replace
import json

import pytest

from tools.event_track_v2x.build_detection_cache_v2 import build_cache
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows, schedule_rows
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig, PersistentForestTracker
from transvision.models.event_track_v2x.tracking_evaluation_v2 import validate_predictions
from test_detection_cache_v2 import sources


@pytest.fixture
def replay_inputs(sources):
    roots, calibration, inputs, output = sources
    sha = build_cache(roots, calibration, sha_file(calibration), inputs, output)
    cache = VerifiedForestCache(output, sha)
    rows = []
    for scene, side, frame in sorted(cache.index):
        if side == 'vehicle-side':
            meta = json.loads(cache.index[(scene, side, frame)][1])
            rows.append(dict(sequence_id=scene, vehicle_frame=frame, infrastructure_frame=frame,
                box_reference_timestamp_us=meta['box_reference_timestamp_us']))
    return cache, rows


def test_real_v2_format_to_durable_sequence_prediction_and_sealed_reopen(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    output = tmp_path/'run'
    config = PersistentForestConfig(state=ForestTrackingConfig(max_model_regret=1.))
    result = replay_rows(cache, rows, output, config)
    assert result['completed_frames'] == 2 and result['status'] == 'complete'
    assert result['geometry_development_baseline'] and not result['trained_paper_method']
    assert not result['paper_eligible'] and not result['gt_model_inputs']
    assert 'full_official_validation_schedule_completed' not in result
    predictions = [json.loads(line) for line in (output/'predictions.jsonl').read_bytes().splitlines()]
    gt = [dict(sequence_id=p['sequence_id'], frame_id=p['frame_id'],
               box_reference_timestamp_us=p['box_reference_timestamp_us'], objects=[]) for p in predictions]
    validate_predictions(predictions, gt)
    for scene, head in result['sequence_heads'].items():
        resumed = PersistentForestTracker.open(output/head['database'],
            expected_database_sha256=head['database_sha256'], expected_prediction_sha256=head['prediction_sha256'])
        assert resumed.sequence_id == scene
        resumed.close()
    assert sha_file(output/'predictions.jsonl') == result['predictions_sha256']
    with pytest.raises(ValueError, match='new output'):
        replay_rows(cache, rows, output, config)


def test_failed_inference_keeps_plan_and_explicitly_partial_receipt(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    output = tmp_path/'failure'
    config = PersistentForestConfig(state=ForestTrackingConfig(max_replay_operations=1))
    with pytest.raises(ValueError, match='work cap'):
        replay_rows(cache, rows, output, config)
    failure = json.loads((output/'failure.json').read_bytes())
    assert failure['status'] == 'failed' and failure['partial_outputs_not_final_results']
    assert failure['completed_frames'] == 0 and not failure['paper_eligible']
    assert failure['plan_sha256'] == sha_file(output/'plan.json')
    assert not (output/'receipt.json').exists()


def test_full_cli_schedule_gate_rejects_small_fixture_before_running(replay_inputs, tmp_path):
    _, rows = replay_inputs
    schedule = tmp_path/'schedule.json'
    schedule.write_bytes(canonical(dict(kind='spd_official_validation_prediction_schedule_v1',
        split_sha256='4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3',
        contains_ground_truth=False, contains_system_error_offset=False, frames=rows)))
    with pytest.raises(ValueError, match='full validation coverage'):
        schedule_rows(schedule, sha_file(schedule))
