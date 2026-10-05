import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.paper_evaluation_policy import VEHICLE_PROTOCOL, NATIVE_VEHICLE_SELECTION
from transvision.models.event_track_v2x.v2v4real_ground_truth import VEHICLE_GT_RECIPE, prepare_vehicle_frame
from transvision.models.event_track_v2x.v2v4real_vehicle_calibration import (
    fit_vehicle_existence, DETECTOR_SEMANTIC_SOURCE_PINS,
)
from tools.event_track_v2x.fit_v2v4real_vehicle_calibration import fit
from tools.event_track_v2x import prepare_v2v4real_ground_truth as gt_tool
from test_v2v4real_ground_truth import pair, obj, make_volume


def fixture():
    metadata = pair()
    metadata['0']['vehicles'] = {1: obj(kind='Truck')}
    metadata['1']['vehicles'] = {}
    objects = prepare_vehicle_frame(metadata, ego_cav='0')['objects']
    frames = [dict(sequence_id='scene', frame_key='000000', objects=objects)]
    manifest = dict(kind='v2v4real_calibration_predictions_v1', protocol_id='v2v4real-nominal-10hz-formal-v1',
        split='train', gt_read=False, official_test_read=False, candidate_protocol='rbf-all-class-top64-v1',
        partition_sha256='p'*64, checkpoint_sha256='a'*64, checkpoint_seed=1337, sequence_ids=['scene'])
    manifest['partition_sha256'] = 'c'*64
    rows = [dict(sequence_id='scene', frame_id='000000', side=side,
        states_ego=[[10., 0., 0., 4., 2., 2., 0., 0., 0.], [12., 0., 0., 4., 2., 2., 0., 0., 0.]],
        raw_scores=[.9, .1], class_indices=[0, 0]) for side in ('vehicle-side', 'infrastructure-side')]
    gt = dict(kind='v2v4real_native_vehicle_gt_projection_v1', recipe=VEHICLE_GT_RECIPE,
        split='train', class_scope=['vehicle'], evaluation_protocol=VEHICLE_PROTOCOL,
        native_label_source=NATIVE_VEHICLE_SELECTION, test_payloads_read=False,
        inference_input=False, GT_read=True)
    partition = dict(kind='v2v4real_formal_nested_training_partition_v1', protocol_id=manifest['protocol_id'],
        official_split='train', official_test_used_for_partition=False, gt_used_for_partition=False,
        calibration_fit=['cal'], detector_fit=['det'], identity_selection=['id'], sequence_to_name_group={'scene': 'cal'})
    semantics = dict(kind='v2v4real_native_vehicle_detector_semantics_v1', native_label_source=NATIVE_VEHICLE_SELECTION,
        numeric_class_index=0, checkpoint_sha256='a'*64, source_sha256=dict(DETECTOR_SEMANTIC_SOURCE_PINS))
    options = dict(prediction_sha256='b'*64, gt_manifest_sha256='d'*64, partition_sha256='c'*64,
                   detector_semantics=semantics, detector_semantics_sha256='e'*64)
    return [manifest, rows, gt, frames, partition], options


def test_vehicle_calibration_matches_native_truck_without_changing_2m_rule():
    args, options = fixture()
    before = copy.deepcopy(args)
    progress = []
    model, receipt = fit_vehicle_existence(*args, **options, progress=progress.append)
    assert model['fit_split'] == 'train'
    assert receipt['positives'] == 2 and receipt['examples'] == 4  # exact 2 m is a negative
    assert receipt['protocol_id'] == VEHICLE_PROTOCOL
    assert receipt['source_prediction_protocol_id'] == 'v2v4real-nominal-10hz-formal-v1'
    assert receipt['evaluation_class_binding']['calibration_evaluation_class'] == 'vehicle'
    assert not receipt['official_test_used'] and not receipt['independent_acceptance_verified']
    assert not receipt['native_nine_state_cache_admitted'] and not receipt['velocity_covariance_fitted']
    assert args == before
    assert progress == [dict(completed_rows=2, total_rows=2)]


@pytest.mark.parametrize('bad', ['prediction_test', 'gt_test', 'old_car_gt', 'old_car_row', 'missing_type',
    'Pedestrian', 'duplicate_side', 'missing_side', 'wrong_role', 'role_overlap', 'wrong_partition',
    'fractional_class', 'nonzero_class', 'wrong_detector', 'unknown_source'])
def test_vehicle_calibration_never_inherits_car_or_test_targets(bad):
    args, options = fixture()
    prediction, rows, gt, frames, partition = args
    if bad == 'prediction_test': prediction['split'] = 'official_test'
    elif bad == 'gt_test': gt['split'] = 'official_test'
    elif bad == 'old_car_gt': gt['class_scope'] = ['Car']
    elif bad == 'old_car_row': frames[0]['objects'][0]['evaluation_class'] = 'car'
    elif bad == 'missing_type': del frames[0]['objects'][0]['raw_class']
    elif bad == 'Pedestrian': frames[0]['objects'][0]['raw_class'] = 'Pedestrian'
    elif bad == 'duplicate_side': rows.append(rows[0])
    elif bad == 'missing_side': rows.pop()
    elif bad == 'wrong_role': prediction['sequence_ids'] = ['test-scene']
    elif bad == 'role_overlap': partition['detector_fit'] = ['cal']
    elif bad == 'wrong_partition': prediction['partition_sha256'] = 'f'*64
    elif bad == 'fractional_class': rows[0]['class_indices'] = [0.1, 0.9]
    elif bad == 'nonzero_class': rows[0]['class_indices'] = [0, 1]
    elif bad == 'wrong_detector': options['detector_semantics']['checkpoint_sha256'] = 'f'*64
    elif bad == 'unknown_source': options['detector_semantics']['source_sha256']['box_utils'] = 'f'*64
    with pytest.raises(ValueError):
        fit_vehicle_existence(*args, **options)


def test_cli_rejects_test_before_gt_frames_are_opened(tmp_path):
    root = tmp_path/'prediction'; root.mkdir()
    (root/'manifest.json').write_bytes(canonical(dict(split='official_test', official_test_read=True)))
    (root/'predictions.jsonl').write_text('unused')
    with pytest.raises(ValueError, match='train-only predictions'):
        fit(root, sha_file(root/'manifest.json'), tmp_path/'missing-GT', 'a'*64,
            tmp_path/'missing-partition', 'a'*64, tmp_path/'missing-semantics', 'a'*64, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_create_once_cli_binds_all_inputs_and_source_bytes(tmp_path):
    source_root = os.environ.get('V2V4REAL_DETECTOR_SEMANTIC_ROOT')
    if not source_root:
        pytest.skip('separate original SHA-pinned detector sources required')
    args, options = fixture()
    prediction, rows, gt, frames, partition = args
    volume_args = make_volume(tmp_path)
    gt_root = tmp_path/'vehicle-gt'
    gt_tool.prepare(*volume_args, gt_root, evaluation_protocol=VEHICLE_PROTOCOL)
    partition_path = tmp_path/'partition.json'; partition_path.write_bytes(canonical(partition))
    prediction['partition_sha256'] = sha_file(partition_path)
    rows = [dict(row, frame_id=f'{i:06d}') for i in range(2) for row in rows]
    raw = b''.join(canonical(row)+b'\n' for row in rows)
    prediction.update(rows=len(rows), rows_sha256=__import__('hashlib').sha256(raw).hexdigest())
    pred_root = tmp_path/'predictions'; pred_root.mkdir()
    (pred_root/'predictions.jsonl').write_bytes(raw)
    (pred_root/'manifest.json').write_bytes(canonical(prediction))
    semantics = options['detector_semantics']
    semantics['source_paths'] = {key: str(Path(source_root)/(key+'.py')) for key in DETECTOR_SEMANTIC_SOURCE_PINS}
    semantics_path = tmp_path/'semantics.json'; semantics_path.write_bytes(canonical(semantics))
    call = (pred_root, sha_file(pred_root/'manifest.json'), gt_root, sha_file(gt_root/'manifest.json'),
            partition_path, sha_file(partition_path), semantics_path, sha_file(semantics_path), tmp_path/'fit')
    result = fit(*call)
    assert result['calibration_sha256'] == sha_file(tmp_path/'fit/calibration.json')
    assert len(result['input_sha256']) == 8
    assert not result['independent_acceptance_verified']
    from tools.event_track_v2x.build_native_paper_cache import recalibrate_native_cache
    from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache, NativeDetectionFrame, write_native_cache
    from transvision.models.event_track_v2x.paper_evaluation_policy import require_vehicle_binding
    from transvision.models.event_track_v2x.detection_cache_v2 import ARRAYS
    from transvision.models.event_track_v2x.paper_calibration import ExistenceCalibration
    from test_paper_pipeline import native_frame
    frame = native_frame(detector_checkpoint_sha256='a'*64)
    raw_root = tmp_path/'raw-cache'
    producer = dict(fit_split='train', candidate_protocol='rbf-all-class-top64-v1',
                    native_label_source=NATIVE_VEHICLE_SELECTION,
                    postprocessing='raw_score_ge_0.05_stable_top64_before_nms', detector_checkpoint_sha256='a'*64)
    raw_sha = write_native_cache(raw_root, [frame], split='official_test', producer=producer, fixture=True)
    cache_args = (raw_root, raw_sha, tmp_path/'fit/calibration.json', result['calibration_sha256'],
                  tmp_path/'fit/receipt.json', sha_file(tmp_path/'fit/receipt.json'), tmp_path/'new-cache')
    # The online cache transform cannot need the calibration's offline GT path.
    gt_root.rename(tmp_path/'sealed-offline-gt')
    receipt = recalibrate_native_cache(*cache_args)
    frozen = NativePaperCache(tmp_path/'new-cache', receipt['cache_sha256'])
    manifest = json.loads(frozen.manifest_json)
    require_vehicle_binding(manifest['producer'])
    loaded = NativeDetectionFrame.load(tmp_path/'new-cache', manifest['frames'][0])
    for field in ARRAYS-{'scores'}:
        assert getattr(loaded, field).tobytes() == getattr(frame, field).tobytes()
    calibration = ExistenceCalibration(**json.loads((tmp_path/'fit/calibration.json').read_bytes()))
    np.testing.assert_array_equal(loaded.scores, calibration.apply(frame.raw_scores))
    assert receipt['gt_payload_opened'] is False and receipt['detector_executed'] is False
    assert receipt['raw_array_bytes_preserved'] and not receipt['new_velocity_covariance_contract_admitted']
    assert not receipt['independent_acceptance_verified']
    # Historical NMS inputs must not be rebranded as the raw-top64 path.
    producer['postprocessing'] = 'raw_score_0.05_rotated_nms_0.15_no_evaluation_roi'
    old_root = tmp_path/'nms-cache'
    old_sha = write_native_cache(old_root, [frame], split='official_test', producer=producer, fixture=True)
    with pytest.raises(ValueError, match='legacy NMS'):
        recalibrate_native_cache(old_root, old_sha, *cache_args[2:-1], tmp_path/'refused-nms')
    for name, value in [('kind', 'v2v4real_existence_calibration_v1'), ('checkpoint_sha256', 'f'*64)]:
        invalid = dict(result, **{name: value})
        invalid_path = tmp_path/(name+'-invalid-receipt.json')
        invalid_path.write_bytes(canonical(invalid))
        with pytest.raises(ValueError):
            recalibrate_native_cache(raw_root, raw_sha, cache_args[2], cache_args[3], invalid_path,
                                     sha_file(invalid_path), tmp_path/('refused-'+name))
    with pytest.raises(ValueError, match='fresh'):
        fit(*call)
