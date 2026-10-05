"""Native source-bound metric entrypoint: fixtures never imply experiment results."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tools.event_track_v2x import evaluate_v2v4real_native_vehicle as wrapper


def rectangular_corners(center=(7., 3., 1.), size=(6., 4., 2.), yaw=0.):
    # Independent analytic corner fixture, not the wrapper/pinned helper output.
    template = ((1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
                (1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1))
    c, s = math.cos(yaw), math.sin(yaw)
    return [[center[0]+u*size[0]/2*c-v*size[1]/2*s,
             center[1]+u*size[0]/2*s+v*size[1]/2*c, center[2]+w*size[2]/2]
            for u, v, w in template]


def world_inputs(tmp_path):
    gt_dir = tmp_path/'vehicle-gt'
    gt_dir.mkdir()
    gt, pred, mapping, poses, egos = [], [], [], [], {}
    for number, (sid, count) in enumerate(wrapper.SEQUENCES.items()):
        scene = 'unordered-scene-'+str(8-number)
        egos[scene] = '0'
        for index in range(count):
            key, stamp, event = f'{1000+index:06d}', 10000+index*100000, 'event-'+str(index)
            gt.append(dict(sequence_id=scene, frame_key=key, frame_ordinal=index, ego_cav='0', objects=[
                dict(track_id=17, evaluation_class='vehicle', raw_class='Truck',
                     corners_ego=rectangular_corners(), selected_cav='0', object_id='17', associated_id=-1)
            ] if index < 2 else []))
            pred.append(dict(sequence_id=scene, frame_id=event, box_reference_timestamp_us=stamp,
                coordinate_frame='world', state_layout=wrapper.WORLD_LAYOUT, predictions=[
                    dict(track_id=scene+':birth-a', class_label='car', mean=[10., 5., 2., 6., 4., 2., 0., 0., 0.],
                         score=.9, covariance=[[0]*9 for _ in range(9)])
                ] if index < 2 else []))
            mapping.append(dict(sequence_id=scene, frame_key=key, frame_ordinal=index, ego_cav='0',
                prediction_frame_id=event, box_reference_timestamp_us=stamp,
                native_sequence_id=sid, native_frame_index=index))
            poses.append(dict(sequence_id=scene, frame_key=key, ego_cav='0', box_reference_timestamp_us=stamp,
                world_to_ego=[[1, 0, 0, -3], [0, 1, 0, -2], [0, 0, 1, -1], [0, 0, 0, 1]]))
    gt_binding = lines(gt_dir/'frames.jsonl', gt)
    gt_manifest = dict(kind=wrapper.GT_KIND, recipe=wrapper.GT_RECIPE, split='official_test',
        evaluation_protocol='v2v4real-official-benchmark-vehicle-v1',
        native_label_source='params.vehicles_except_exact_obj_type_Pedestrian',
        coordinate_frame='current_ego_lidar', time_basis='ordinal-only-no-clock',
        box_representation='eight_ordered_corners_xyz_m', paired_frames=1993,
        ego_agents=egos, frames_sha256=gt_binding['sha256'])
    spec = dict(kind='v2v4real_world_to_native_vehicle_conversion_input_v1', fixture=True,
                source_commit=wrapper.COMMIT, prediction_class_mapping={'car': 'vehicle'},
                predictions=lines(tmp_path/'world.jsonl', pred),
                ground_truth_manifest=save(gt_dir/'manifest.json', gt_manifest),
                frame_mapping=lines(tmp_path/'mapping.jsonl', mapping),
                world_to_ego=lines(tmp_path/'poses.jsonl', poses))
    path = tmp_path/'conversion-input.json'
    save(path, spec)
    return path, spec


@pytest.fixture
def source():
    path = os.environ.get('RBF_DMSTRACK_NATIVE_SOURCE')
    if not path:
        pytest.skip('provide the independently downloaded fixed DMSTrack source')
    path = Path(path)
    wrapper.source_evidence(path)
    return path


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False)+'\n')
    return {'path': str(path), 'sha256': wrapper.sha(path)}


def lines(path, values):
    path.write_text(''.join(json.dumps(v, allow_nan=False)+'\n' for v in values))
    return {'path': str(path), 'sha256': wrapper.sha(path)}


def inputs(tmp_path, offset=0):
    gt, pred = [], []
    for sid, count in wrapper.SEQUENCES.items():
        for frame in range(count):
            box = {'track_id': 1, 'class_label': 'vehicle',
                   'box_hwlxyzry': [4., 4., 4., 0., 0., 10., 0.],
                   'bbox_2d': [0., 0., 100., 100.]}
            target = {'sequence_id': sid, 'frame_index': frame, 'objects': [box] if frame < 2 else []}
            prediction = copy.deepcopy(target)
            for obj in prediction['objects']:
                obj['box_hwlxyzry'][3] += offset
                obj['score'] = 0.9
            gt.append(target)
            pred.append(prediction)
    spec = {'kind': 'v2v4real_native_vehicle_input_v1', 'fixture': True,
            'protocol': dict(wrapper.PROTOCOL), 'source_commit': wrapper.COMMIT,
            'ground_truth': lines(tmp_path/'gt.jsonl', gt), 'predictions': lines(tmp_path/'pred.jsonl', pred)}
    conversion = {'kind': 'v2v4real_native_box_conversion_receipt_v1', 'fixture': True,
                  'target_coordinates': wrapper.PROTOCOL['box_coordinates'], 'evaluation_class': 'vehicle',
                  'complete_test_frames': 1993, 'ground_truth_sha256': spec['ground_truth']['sha256'],
                  'predictions_sha256': spec['predictions']['sha256']}
    spec['conversion_receipt'] = save(tmp_path/'conversion.json', conversion)
    path = tmp_path/'manifest.json'
    save(path, spec)
    return path, spec


def test_fixed_source_and_official_seqmap_identity(source):
    evidence = wrapper.source_evidence(source)
    assert evidence['commit'] == 'd3b9949499c8e68ea33060873bd1cb95b6d4d323'
    assert len(wrapper.SEQUENCES) == 9 and sum(wrapper.LENGTHS) == 1993
    assert evidence['files'][wrapper.SEQMAP] == '840783eed9ab01cdd359015b03ef9b9ba0f867a1ab78882e7dd2d8323f8d1c5f'


def test_fixed_geometry_analytic_yaw_and_explicit_rigid_transform(source):
    import numpy as np
    world, native = wrapper.load_geometry(source)
    for yaw in (0., .3, -.7):
        angle = .4
        c, s = math.cos(angle), math.sin(angle)
        matrix = [[c, -s, 0, 7], [s, c, 0, -3], [0, 0, 1, 2], [0, 0, 0, 1]]
        wrapper.rigid_transform(matrix)
        corners = world([1., 2., 3., 6., 4., 2., yaw, .1, .2], matrix)
        expected_center = (c-2*s+7, s+2*c-3, 5)
        analytic = rectangular_corners(expected_center, yaw=yaw+angle)
        np.testing.assert_allclose(corners[0], analytic, atol=2e-6, rtol=0)
        expected_box = [2, 4, 6, expected_center[0], 5, expected_center[1], yaw+angle]
        np.testing.assert_allclose(native(corners, True)[0], expected_box, atol=2e-6, rtol=0)
        np.testing.assert_allclose(native(np.asarray([analytic]), False)[0], expected_box, atol=1e-12, rtol=0)


def test_geometry_source_identity_checked_before_loading(source, tmp_path):
    import shutil
    for rel in wrapper.GEOMETRY_HASHES:
        target = tmp_path/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/rel, target)
    file = tmp_path/'V2V4Real/opencood/tools/inference.py'
    file.write_bytes(file.read_bytes()+b'\n# no geometry substitution\n')
    with pytest.raises(ValueError, match='geometry source changed'):
        wrapper.load_geometry(tmp_path)


@pytest.mark.parametrize('matrix', [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    [[1, .01, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
    [[True, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
])
def test_ambiguous_or_improper_pose_rejected(matrix):
    with pytest.raises(ValueError):
        wrapper.rigid_transform(matrix)


def test_full_world_conversion_preserves_empty_frames_and_explicit_mapping(source, tmp_path):
    import numpy as np
    path, spec = world_inputs(tmp_path)
    result = wrapper.convert(path, wrapper.sha(path), source, tmp_path/'converted')
    native_manifest = tmp_path/'converted/manifest.json'
    _, data, _, _ = wrapper.validate(native_manifest, wrapper.sha(native_manifest), source)
    assert len(data['predictions']) == 1993
    assert sum(not f['objects'] for f in data['predictions']) == 1975
    np.testing.assert_allclose(data['predictions'][0]['objects'][0]['box_hwlxyzry'], [2, 4, 6, 7, 1, 3, 0], atol=1e-6)
    assert data['ground_truth'][0]['objects'][0]['class_label'] == 'vehicle'
    assert data['predictions'][0]['objects'][0]['track_id'] == data['predictions'][1]['objects'][0]['track_id'] == 0
    identity = json.loads((tmp_path/'converted/identity-map.json').read_text())
    assert identity['0000'][0]['input_track_id'].startswith('unordered-scene-8:')
    assert result['input_sha256'][spec['world_to_ego']['path']] == spec['world_to_ego']['sha256']
    assert result['poses_and_frame_mapping_independently_accepted'] is False
    assert result['world_state_or_corner_conversion_implemented'] is True
    assert result['full_rbf_native_path_accepted'] is False


def test_gt_json_restores_original_native_float32_export_precision(source, tmp_path):
    import numpy as np
    path, spec = world_inputs(tmp_path)
    gm_path = Path(spec['ground_truth_manifest']['path'])
    gm = json.loads(gm_path.read_text())
    frames_path = gm_path.parent/'frames.jsonl'
    frames = wrapper.jsonl(frames_path)
    corners = rectangular_corners(center=(7.1234567, 3.2345678, 1.3456789), yaw=.1234567)
    frames[0]['objects'][0]['corners_ego'] = corners
    gm['frames_sha256'] = lines(frames_path, frames)['sha256']
    spec['ground_truth_manifest'] = save(gm_path, gm)
    save(path, spec)
    wrapper.convert(path, wrapper.sha(path), source, tmp_path/'converted')
    _, native = wrapper.load_geometry(source)
    output = wrapper.jsonl(tmp_path/'converted/ground_truth.jsonl')[0]['objects'][0]['box_hwlxyzry']
    expected = native(np.asarray([corners], dtype=np.float32), False)[0]
    assert np.array_equal(np.asarray(output), expected)
    assert not np.array_equal(expected, native(np.asarray([corners], dtype=np.float64), False)[0])


@pytest.mark.parametrize('fault', ['missing_pose', 'duplicate_mapping', 'pose_ego', 'pose_time',
    'prediction_frame_id', 'native_index', 'swap_native_ordinals', 'mix_scenes', 'bad_class', 'bad_layout', 'missing_empty_prediction',
    'gt_wrong_split', 'gt_pedestrian', 'gt_bad_corner', 'duplicate_track_id', 'gt_ordinal', 'unbound_change'])
def test_world_conversion_binding_and_coverage_fail_closed(source, tmp_path, fault):
    path, spec = world_inputs(tmp_path)
    role = 'world_to_ego' if fault.startswith('pose_') or fault == 'missing_pose' else 'frame_mapping'
    if fault in ('bad_class', 'bad_layout', 'missing_empty_prediction', 'duplicate_track_id'):
        role = 'predictions'
    if fault.startswith('gt_'):
        role = 'ground_truth_manifest'
    file = Path(spec[role]['path'])
    if role == 'ground_truth_manifest':
        gm = json.loads(file.read_text())
        if fault == 'gt_wrong_split':
            gm['split'] = 'train'
        else:
            frames_file = file.parent/'frames.jsonl'
            frames = wrapper.jsonl(frames_file)
            if fault == 'gt_pedestrian': frames[0]['objects'][0]['raw_class'] = 'Pedestrian'
            if fault == 'gt_bad_corner': frames[0]['objects'][0]['corners_ego'][0].pop()
            if fault == 'gt_ordinal': frames[0]['frame_ordinal'] = True
            gm['frames_sha256'] = lines(frames_file, frames)['sha256']
        spec[role] = save(file, gm)
    else:
        rows = wrapper.jsonl(file)
        if fault in ('missing_pose', 'missing_empty_prediction'): rows.pop()
        elif fault == 'duplicate_mapping': rows[-1] = rows[-2]
        elif fault == 'pose_ego': rows[0]['ego_cav'] = '1'
        elif fault == 'pose_time': rows[0]['box_reference_timestamp_us'] += 1
        elif fault == 'prediction_frame_id': rows[0]['prediction_frame_id'] = 'unbound-event'
        elif fault == 'native_index': rows[0]['native_frame_index'] = 1
        elif fault == 'swap_native_ordinals':
            rows[0]['native_frame_index'], rows[1]['native_frame_index'] = rows[1]['native_frame_index'], rows[0]['native_frame_index']
        elif fault == 'mix_scenes': rows[0]['native_sequence_id'], rows[-1]['native_sequence_id'] = rows[-1]['native_sequence_id'], rows[0]['native_sequence_id']
        elif fault == 'bad_class': rows[0]['predictions'][0]['class_label'] = 'bicycle'
        elif fault == 'bad_layout': rows[0]['state_layout'] = 'xyz_hwl_heading'
        elif fault == 'duplicate_track_id': rows[0]['predictions'] *= 2
        elif fault == 'unbound_change': rows[0]['ego_cav'] = '1'
        binding = lines(file, rows)
        if fault != 'unbound_change': spec[role] = binding
    save(path, spec)
    with pytest.raises(ValueError):
        wrapper.convert(path, wrapper.sha(path), source, tmp_path/'not-accepted')
    assert not (tmp_path/'not-accepted/manifest.json').exists()


@pytest.mark.parametrize('channel', ['bicycle', 'pedestrian'])
def test_nonvehicle_channels_cannot_be_relabelled_as_vehicle(source, tmp_path, channel):
    path, spec = world_inputs(tmp_path)
    spec['prediction_class_mapping'][channel] = 'vehicle'
    save(path, spec)
    with pytest.raises(ValueError, match='class mapping'):
        wrapper.convert(path, wrapper.sha(path), source, tmp_path/'not-accepted')
    assert not (tmp_path/'not-accepted').exists()


def test_explicit_nonvehicle_exclusion_leaves_upstream_competition_and_empty_gt_untouched(source, tmp_path):
    path, spec = world_inputs(tmp_path)
    pred_path = Path(spec['predictions']['path'])
    pred = wrapper.jsonl(pred_path)
    bicycle = copy.deepcopy(pred[0]['predictions'][0])
    bicycle.update(track_id='bicycle-competitor', class_label='bicycle')
    pred[0]['predictions'].append(bicycle)
    spec['predictions'] = lines(pred_path, pred)
    spec['prediction_class_mapping']['bicycle'] = None
    gt_path = Path(spec['ground_truth_manifest']['path'])
    gm = json.loads(gt_path.read_text())
    frames = wrapper.jsonl(gt_path.parent/'frames.jsonl')
    frames[0]['objects'] = []  # Prediction survival cannot depend on GT presence.
    gm['frames_sha256'] = lines(gt_path.parent/'frames.jsonl', frames)['sha256']
    spec['ground_truth_manifest'] = save(gt_path, gm)
    save(path, spec)
    result = wrapper.convert(path, wrapper.sha(path), source, tmp_path/'converted')
    assert result['excluded_prediction_objects'] == 1
    assert wrapper.sha(pred_path) == spec['predictions']['sha256']
    assert len(wrapper.jsonl(pred_path)[0]['predictions']) == 2
    assert len(wrapper.jsonl(tmp_path/'converted/predictions.jsonl')[0]['objects']) == 1
    assert wrapper.jsonl(tmp_path/'converted/ground_truth.jsonl')[0]['objects'] == []
    assert result['upstream_competition_modified'] is False
    assert result['prediction_class_mapping_semantics_independently_accepted'] is False


def test_cli_world_to_native_metric_fixture_is_runnable(source, tmp_path):
    path, spec = world_inputs(tmp_path)
    output = tmp_path/'converted-and-evaluated'
    run = subprocess.run([sys.executable, str(Path(wrapper.__file__)), '--source', str(source), '--convert',
        '--evaluate-converted', '--manifest', str(path), '--manifest-sha256', wrapper.sha(path),
        '--output', str(output)], capture_output=True, text=True, env=os.environ.copy())
    assert run.returncode == 0, run.stderr
    report = json.loads((output/'native-evaluation/report.json').read_text())
    assert report['fixture'] is True and report['status'] == 'native_metrics_computed'
    assert report['runtime_jit_disabled'] is True and report['metrics']['AMOTP'] > 0
    assert report['world_state_or_corner_conversion_implemented'] is True
    assert report['full_rbf_native_path_accepted'] is False
    assert report['paper_performance_complete'] is False
    assert report['conversion_receipt_independently_accepted'] is False


def test_source_changed_even_if_git_commit_claim_unchanged(source, tmp_path):
    import shutil
    for rel in wrapper.SOURCE_HASHES:
        p = tmp_path/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/rel, p)
    p = tmp_path/'AB3DMOT/scripts/KITTI/evaluate.py'
    p.write_bytes(p.read_bytes()+b'\n# changed\n')
    with pytest.raises(ValueError, match='source changed'):
        wrapper.source_evidence(tmp_path)


def test_empty_frames_are_retained_and_vehicle_is_backend_alias(source, tmp_path):
    path, _ = inputs(tmp_path)
    _, data, _, _ = wrapper.validate(path, wrapper.sha(path), source)
    assert len(data['predictions']) == 1993
    assert sum(not f['objects'] for f in data['predictions']) == 1975
    wrapper.write_native(data['ground_truth'], tmp_path/'native', False)
    assert len(list((tmp_path/'native').glob('*.txt'))) == 9
    line = (tmp_path/'native/0000.txt').read_text().splitlines()[0].split()
    assert line[2] == 'Car'
    assert list(map(float, line[10:])) == [4, 4, 4, 0, 0, 10, 0]


@pytest.mark.parametrize('fault', ['missing_empty_frame', 'duplicate_frame', 'reorder', 'unknown_sequence',
                                    'bool_frame', 'duplicate_id', 'car', 'van', 'nan', 'negative_dimension',
                                    'world_mean', 'missing_bbox2d', 'bbox_reversed', 'extra_score_gt', 'empty_all'])
def test_invalid_native_inputs_fail_closed(tmp_path, fault):
    path, spec = inputs(tmp_path)
    frames = [json.loads(line) for line in Path(spec['ground_truth']['path']).read_text().splitlines()]
    if fault == 'missing_empty_frame':
        frames.pop()
    elif fault == 'duplicate_frame':
        frames[-1] = frames[-2]
    elif fault == 'reorder':
        frames[0], frames[1] = frames[1], frames[0]
    elif fault == 'unknown_sequence':
        frames[-1]['sequence_id'] = '0009'
    elif fault == 'bool_frame':
        frames[0]['frame_index'] = False
    elif fault == 'duplicate_id':
        frames[0]['objects'] *= 2
    elif fault in ('car', 'van'):
        frames[0]['objects'][0]['class_label'] = fault
    elif fault == 'nan':
        frames[0]['objects'][0]['box_hwlxyzry'][3] = float('nan')
    elif fault == 'negative_dimension':
        frames[0]['objects'][0]['box_hwlxyzry'][0] = -1
    elif fault == 'world_mean':
        frames[0]['objects'][0]['mean'] = [0]*9
    elif fault == 'missing_bbox2d':
        del frames[0]['objects'][0]['bbox_2d']
    elif fault == 'bbox_reversed':
        frames[0]['objects'][0]['bbox_2d'] = [0, 2, 1, 1]
    elif fault == 'extra_score_gt':
        frames[0]['objects'][0]['score'] = .9
    elif fault == 'empty_all':
        for f in frames:
            f['objects'] = []
    bad = tmp_path/'bad.jsonl'
    bad.write_text(''.join(json.dumps(x)+'\n' for x in frames))
    with pytest.raises(ValueError):
        wrapper.read_frames(bad, False)


@pytest.mark.parametrize('key,value', [('split', 'val'), ('backend_split', 'test'), ('evaluation_class', 'car'),
                                      ('iou_threshold', .5), ('nominal_frequency_hz', 2)])
def test_protocol_changes_cannot_run(source, tmp_path, key, value):
    path, spec = inputs(tmp_path)
    spec['protocol'][key] = value
    save(path, spec)
    with pytest.raises(ValueError, match='protocol'):
        wrapper.evaluate(path, wrapper.sha(path), source, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_rehashed_unrelated_conversion_receipt_rejected(source, tmp_path):
    path, spec = inputs(tmp_path)
    conv = Path(spec['conversion_receipt']['path'])
    value = json.loads(conv.read_text())
    value['predictions_sha256'] = '0'*64
    spec['conversion_receipt'] = save(conv, value)
    save(path, spec)
    with pytest.raises(ValueError, match='conversion'):
        wrapper.validate(path, wrapper.sha(path), source)


def test_changed_input_rejected_before_launch(source, tmp_path, monkeypatch):
    path, spec = inputs(tmp_path)
    Path(spec['predictions']['path']).write_text('{}\n')
    monkeypatch.setattr(wrapper.subprocess, 'run', lambda *a, **k: pytest.fail('launched invalid input'))
    with pytest.raises(ValueError, match='checksum'):
        wrapper.evaluate(path, wrapper.sha(path), source, tmp_path/'output')


def test_duplicate_json_keys_fail():
    with pytest.raises(ValueError, match='duplicate JSON'):
        wrapper.strict_json('{"fixture":true,"fixture":false}')


def test_real_source_native_fixture_computes_metrics_without_name_heuristic(source, tmp_path, monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'pytest-mpl'))
    pytest.importorskip('numba')
    pytest.importorskip('matplotlib')
    path, _ = inputs(tmp_path)
    output = tmp_path/'result_name_without_any_native_detector_token'
    report = wrapper.evaluate(path, wrapper.sha(path), source, output)
    assert report['fixture'] is True
    assert report['coverage']['predictions']['frames'] == 1993
    assert report['coverage']['predictions']['empty_frames'] == 1975
    assert report['protocol']['backend_split'] == 'val'
    assert report['metrics']['AMOTA'] > 0
    assert report['metrics']['AMOTP'] == pytest.approx(report['metrics']['AMOTA'])
    assert report['metric_directions']['AMOTP'] == 'higher'
    assert report['paper_performance_complete'] is False
    assert report['full_rbf_native_path_accepted'] is False
    assert (output/'AB3DMOT/results/v2v4real/rbf_vehicle/summary_car_average_eval3D.txt').is_file()
    assert len(json.loads((output/'frame-index.json').read_text())) == 1993
    assert not (output/'AB3DMOT/results/KITTI').exists()
    shifted = tmp_path/'shifted'
    shifted.mkdir()
    shifted_manifest, _ = inputs(shifted, offset=4/3)
    shifted_report = wrapper.evaluate(shifted_manifest, wrapper.sha(shifted_manifest), source, tmp_path/'shifted-output')
    assert shifted_report['metrics']['AMOTP'] == pytest.approx(report['metrics']['AMOTP']/2)
    assert shifted_report['metrics']['AMOTA'] == pytest.approx(report['metrics']['AMOTA'])
    with pytest.raises(ValueError, match='create-once'):
        wrapper.evaluate(path, wrapper.sha(path), source, output)


def test_original_iou_geometry_and_exact_threshold_boundary_in_subprocess(source, tmp_path, monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'pytest-mpl'))
    pytest.importorskip('numba')
    pytest.importorskip('matplotlib')
    # Independently known axis-aligned cube IoU = (4-d)/(4+d).
    code = '''
import json,sys
from pathlib import Path
from tools.event_track_v2x import evaluate_v2v4real_native_vehicle as w
n=w.load_native(sys.argv[1])
from AB3DMOT_libs.box import Box3D
from AB3DMOT_libs.dist_metrics import iou
a=Box3D(x=0.,y=0.,z=10.,h=4.,w=4.,l=4.,ry=0.)
values=[]
for d in [0., 4./3., 2.4-1e-6, 2.4+1e-6, 5.]:
 b=Box3D(x=d,y=0.,z=10.,h=4.,w=4.,l=4.,ry=0.)
 values.append(float(iou(a,b,metric='iou_3d')))
import os,shutil
root=Path(sys.argv[2]).parent
gt=[]
for sid,count in w.SEQUENCES.items():
 for i in range(count):
  box={'track_id':1,'class_label':'vehicle','box_hwlxyzry':[4.,4.,4.,0.,0.,10.,0.], 'bbox_2d':[0.,0.,100.,100.]}
  gt.append({'sequence_id':sid,'frame_index':i,'objects':[box] if i<2 else []})
gt_dir=root/'AB3DMOT/scripts/KITTI/v2v4real_val_label'
w.write_native(gt,gt_dir,False)
shutil.copyfile(Path(sys.argv[1])/w.SEQMAP,root/w.SEQMAP)
os.chdir(root/'AB3DMOT')
gates=[]
for j,d in enumerate([2.4-1e-6,2.4+1e-6]):
 import copy
 pred=copy.deepcopy(gt)
 for frame in pred:
  for box in frame['objects']:
   box['score']=.9;box['box_hwlxyzry'][3]=d
 name='arbitrary_rbf_'+str(j)
 w.write_native(pred,root/'AB3DMOT/results/v2v4real'/name/'data_0',True)
 e=n.trackingEvaluation(name,mail=n.mailpy.Mail(''),cls='car',eval_3diou=True,eval_2diou=False,num_hypo=1,thres=.25,evaluate_v2v4real=True,seq_eval_mode='all',v2v4real_split='val')
 assert e.loadTracker() and e.loadGroundtruth()
 e.compute3rdPartyMetrics()
 gates.append({'tp':e.tp,'fp':e.fp,'fn':e.fn})
Path(sys.argv[2]).write_text(json.dumps({'iou':values,'gates':gates}))
'''
    result = tmp_path/'geometry.json'
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path/'mpl'), PYTHONDONTWRITEBYTECODE='1', NUMBA_DISABLE_JIT='1')
    subprocess.run([sys.executable, '-c', code, str(source), str(result)], check=True, env=env)
    checked = json.loads(result.read_text())
    values = checked['iou']
    assert values[:2] == pytest.approx([1., .5])
    assert values[2] > .25 and values[3] < .25
    assert values[4] == 0
    assert checked['gates'] == [{'tp': 18, 'fp': 0, 'fn': 0}, {'tp': 0, 'fp': 18, 'fn': 18}]


def test_nonzero_native_exit_is_preserved_not_accepted(source, tmp_path, monkeypatch):
    path, _ = inputs(tmp_path)
    monkeypatch.setattr(wrapper.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 3))
    output = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='native evaluator failed'):
        wrapper.evaluate(path, wrapper.sha(path), source, output)
    assert json.loads((output/'failure.json').read_text())['returncode'] == 3
    assert not (output/'report.json').exists()


def test_post_execution_input_change_is_not_accepted(source, tmp_path, monkeypatch):
    path, spec = inputs(tmp_path)
    def changed(*args, **kwargs):
        Path(spec['predictions']['path']).write_text('{}\n')
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(wrapper.subprocess, 'run', changed)
    output = tmp_path/'changed'
    with pytest.raises(ValueError, match='changed during'):
        wrapper.evaluate(path, wrapper.sha(path), source, output)
    assert not (output/'report.json').exists()
