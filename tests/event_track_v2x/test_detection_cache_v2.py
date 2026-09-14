import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.event_track_v2x.build_detection_cache_v2 import build_cache, record, verify, write_json
from transvision.models.event_track_v2x.detection_cache_v2 import (
    BOX_LAYOUT, CLASSES, FEATURE_METHOD, SIDES, DetectionCacheV2, canonical, load_manifest, sha_file,
)


def _frame(n=2, side='vehicle-side'):
    meta = dict(kind='detection_cache_v2', schema_version=2, sequence_id='0003', frame_id='000123',
        side=side, box_reference_timestamp_us=1000000, source_image_timestamp_us=1100000,
        coordinate_system='source_lidar', lidar_to_world_row_rotation=np.eye(3).tolist(),
        lidar_to_world_translation=[1, 2, 3], image_sha256='1' * 64, box_layout=BOX_LAYOUT,
        agent_mask=SIDES[side], dataset_split='val', dataset_sha256='2' * 64,
        detector_config_sha256='3' * 64, detector_checkpoint_sha256='4' * 64,
        feature_checkpoint_sha256='5' * 64, feature_method=FEATURE_METHOD,
        calibration_sha256='6' * 64, calibration_fit_split='train', raw_manifest_sha256='7' * 64,
        raw_arrays_sha256='8' * 64, raw_metadata_sha256='9' * 64, arrays_sha256='a' * 64)
    states = np.tile([1., 2., 3., 4., 2., 1., 0., 1., 0.], (n, 1))
    appearance = np.zeros((n, 128), dtype=np.float32)
    appearance[:, 0] = 1
    return DetectionCacheV2(canonical(meta), states, np.ones(n) * .6, np.ones(n) * .7,
        np.zeros(n, dtype=np.int64), np.tile(np.eye(9), (n, 1, 1)), appearance,
        np.ones(n, dtype=bool))


def _metadata(frame, **changes):
    return replace(frame, metadata_json=canonical(dict(frame.metadata, **changes)))


def test_frame_bits_are_immutable_and_metadata_detached():
    frame = _frame()
    digest = frame.digest()
    for name in ['states', 'raw_scores', 'scores', 'class_indices', 'covariances', 'appearance', 'appearance_valid']:
        value = getattr(frame, name)
        with pytest.raises(ValueError):
            value.setflags(write=True)
        with pytest.raises(ValueError):
            value.flat[0] = 0
    meta = frame.metadata
    meta['agent_mask'] = 3
    meta['lidar_to_world_translation'][0] = 900
    assert frame.digest() == digest
    assert frame.metadata['agent_mask'] == 1


def test_empty_frame_and_missing_appearance_are_explicit():
    assert _frame(0).count == 0
    frame = replace(_frame(), appearance=np.zeros((2, 128)), appearance_valid=np.zeros(2, dtype=bool))
    assert not frame.appearance_valid.any()
    with pytest.raises(ValueError, match='missing appearance'):
        replace(frame, appearance=np.ones((2, 128)))
    with pytest.raises(ValueError, match='L2 normalized'):
        replace(_frame(), appearance=np.zeros((2, 128)))


@pytest.mark.parametrize('field,value', [
    ('agent_mask', 3), ('agent_mask', True), ('side', 'unknown'), ('dataset_split', 'test'),
    ('calibration_fit_split', 'val'), ('source_image_timestamp_us', 1.1),
    ('detector_checkpoint_sha256', 'not-a-sha'), ('object_ids', []), ('future_frame', '123'),
    ('feature_method', 'gt-crop'), ('lidar_to_world_row_rotation', [[1, 0, 0]]),
])
def test_source_leakage_and_provenance_fail_closed(field, value):
    with pytest.raises((ValueError, TypeError)):
        _metadata(_frame(), **{field: value})


@pytest.mark.parametrize('name', ['states', 'raw_scores', 'scores', 'covariances', 'appearance'])
def test_nonfinite_arrays_fail_closed(name):
    array = getattr(_frame(), name).copy()
    array.flat[0] = np.nan
    with pytest.raises(ValueError):
        replace(_frame(), **{name: array})


def test_information_time_uses_both_timestamps():
    frame = _frame()
    assert frame.information_timestamp_us == 1100000
    assert not frame.available_at(1099999)
    assert frame.available_at(1100000)
    assert _metadata(frame, box_reference_timestamp_us=1200000).information_timestamp_us == 1200000


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def _sources(tmp_path, split='val'):
    inputs = tmp_path / 'inputs'
    sequences = ['0003', '0007']
    rows = [dict(sequence_id=seq, frame_id='00012' + str(i), source_image_timestamp_us=1100000 + i,
        box_reference_timestamp_us=1000000 + i, image_sha256='1' * 64) for i, seq in enumerate(sequences)]
    input_manifest = dict(kind=('eventtrack_validation_image_pose_inputs_v1' if split == 'val'
                                else 'eventtrack_train_image_pose_inputs_v1'), gt_payloads_in_package=False,
        test_payloads_read=False, val_payloads_read=split == 'val',
        frames={side: 2 for side in SIDES})
    input_manifest['validation_sequences' if split == 'val' else 'train_sequences'] = sequences
    for side in SIDES:
        index = inputs / side / 'frame-index.json'
        _write(index, rows)
        input_manifest[side] = {'frame_index_sha256': sha_file(index)}
    _write(inputs / 'input-manifest.json', input_manifest)
    calibration = dict(kind='eventtrack_train_calibration_v1', fit_sequences=['0000'],
        evidence={'official_validation_used_for_selection': False, 'test_payloads_read': False},
        sides={side: {name: {'score': {'slope': 1., 'intercept': 0., 'logit_clip': .000001},
                            'covariance': {'matrix': np.eye(9).tolist()}} for name in CLASSES} for side in SIDES})
    calibration_path = tmp_path / 'calibration.json'
    _write(calibration_path, calibration)
    roots = []
    for side in SIDES:
        for shard, row in enumerate(rows):
            root = tmp_path / (side + '-shard-' + str(shard))
            root.mkdir()
            config = root / 'resolved-cache-config.py'
            config.write_text('# fixed config\n')
            # Include an empty frame to test full-cohort coverage with no detections.
            n = 0 if side == 'vehicle-side' and shard == 1 else 2
            frame = _frame(n, side)
            bottom = frame.states.copy()
            bottom[:, 2] -= bottom[:, 5] / 2
            payload = root / 'frames' / row['sequence_id'] / (row['frame_id'] + '.npz')
            payload.parent.mkdir(parents=True)
            np.savez_compressed(payload, boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy=bottom,
                gravity_centers_lidar=frame.states[:, :3], scores=frame.raw_scores,
                class_indices=frame.class_indices, appearance_128=frame.appearance,
                appearance_valid=frame.appearance_valid, image_rois_xyxy=np.zeros((n, 4)))
            meta = dict(row, side=side, coordinate_system='source_lidar',
                lidar_to_world_row_rotation=np.eye(3).tolist(), lidar_to_world_translation=[1, 2, 3])
            metadata = payload.with_suffix('.json')
            _write(metadata, meta)
            launch = dict(kind='eventtrack_raw_cache_launch_v1', side=side, shard_count=2, shard_index=shard,
                sequences=[row['sequence_id']], frames=1, checkpoint_sha256='4' * 64,
                appearance_checkpoint_sha256='5' * 64, appearance_method=FEATURE_METHOD,
                input_manifest_sha256=sha_file(inputs / 'input-manifest.json'),
                resolved_config_sha256=sha_file(config), detector_decode_source='raw_head_before_tracking_summary_v1',
                formal_v2_ready=False, gt_inputs=False, test_payloads_read=False, val_payloads_read=split == 'val',
                optimizer_created=False, score_filter_changed=False, covariance_calibrated=False)
            _write(root / 'launch-receipt.json', launch)
            manifest = dict(launch, kind='eventtrack_raw_detector_cache_v1',
                frames=[dict(arrays=record(payload, root), metadata=record(metadata, root), detections=n)],
                frame_count=1, classes=list(CLASSES), weights_unchanged=True, raw_head_frames_verified=1,
                legacy_placeholder_count=0, detection_count=n, appearance_valid_count=n)
            _write(root / 'raw-cache-manifest.json', manifest)
            roots.append(root)
    return roots, calibration_path, inputs, tmp_path / 'sealed'


@pytest.fixture
def sources(tmp_path):
    return _sources(tmp_path)


def test_train_cache_preserves_split_and_does_not_refit_calibration(tmp_path):
    source = _sources(tmp_path, split='train')
    roots, calibration, inputs, output = source
    before = calibration.read_bytes()
    sha = _build(source)
    manifest, entries = load_manifest(output, sha)
    assert manifest['split'] == 'train'
    assert not manifest['test_payloads_read'] and not manifest['gt_in_cache']
    assert all(DetectionCacheV2.load(output, e).metadata['dataset_split'] == 'train' for e in entries)
    assert calibration.read_bytes() == before
    assert verify(output, sha, inputs)['exact_input_cohort_verified']


@pytest.mark.parametrize('forbidden_read', ['test_payloads_read', 'val_payloads_read'])
def test_train_source_cannot_relabel_a_forbidden_read_as_training(tmp_path, forbidden_read):
    source = _sources(tmp_path, split='train')
    root = source[0][0]
    for name in ['launch-receipt.json', 'raw-cache-manifest.json']:
        path = root / name
        data = json.loads(path.read_bytes())
        data[forbidden_read] = True
        path.write_bytes(canonical(data))
    with pytest.raises(ValueError, match='boundary differs'):
        _build(source)


def _build(sources):
    roots, calibration, inputs, output = sources
    return build_cache(roots, calibration, sha_file(calibration), inputs, output)


def test_build_full_readback_and_source_masks_share_same_cache(sources):
    sha = _build(sources)
    roots, calibration, inputs, output = sources
    report = verify(output, sha, inputs)
    assert report['frames'] == 4 and report['detections'] == 6 and report['sequences'] == 2
    assert report['exact_input_cohort_verified']
    manifest, all_entries = load_manifest(output, sha)
    for mask, side in [(1, 'vehicle-side'), (2, 'infrastructure-side')]:
        same_manifest, entries = load_manifest(output, sha, agent_mask=mask)
        assert same_manifest == manifest and len(entries) == 2
        assert all(DetectionCacheV2.load(output, entry).metadata['side'] == side for entry in entries)
        assert all(entry in all_entries for entry in entries)
    frame = DetectionCacheV2.load(output, all_entries[0])
    assert np.array_equal(frame.raw_scores, np.array([.6, .6]))
    assert np.allclose(frame.scores, frame.raw_scores)
    assert np.array_equal(frame.states[:, 2], [3, 3])
    with pytest.raises(ValueError, match='already exist'):
        _build(sources)


@pytest.mark.parametrize('mutation', ['extra', 'symlink', 'payload', 'manifest_unknown', 'manifest_noncanonical', 'unknown_array'])
def test_sealed_tree_tamper_fails_closed(sources, mutation):
    sha = _build(sources)
    output = sources[-1]
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if mutation == 'extra':
        (output / 'labels.json').write_text('{}')
    elif mutation == 'symlink':
        (output / 'alias').symlink_to('manifest.json')
    elif mutation == 'payload':
        path = output / manifest['frames'][0]['arrays']['path']
        with path.open('ab') as stream:
            stream.write(b'tamper')
    elif mutation == 'unknown_array':
        entry = manifest['frames'][0]
        path = output / entry['arrays']['path']
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['gt_object_ids'] = np.array([1, 2])
        np.savez_compressed(path, **arrays)
        entry['arrays'] = record(path, output)
        manifest_path.write_bytes(canonical(manifest))
        sha = sha_file(manifest_path)
    else:
        if mutation == 'manifest_unknown':
            manifest['gt_labels'] = []
        manifest_path.write_bytes(json.dumps(manifest).encode() if mutation == 'manifest_noncanonical' else canonical(manifest))
        sha = sha_file(manifest_path)
    with pytest.raises(ValueError):
        load_manifest(output, sha)


def test_raw_supervision_and_calibration_identity_rejected(sources):
    roots, calibration, inputs, output = sources
    with pytest.raises(ValueError, match='calibration identity'):
        build_cache(roots, calibration, 'f' * 64, inputs, output)
    manifest_path = roots[0] / 'raw-cache-manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['gt_inputs'] = True
    manifest_path.write_bytes(canonical(manifest))
    with pytest.raises(ValueError, match='boundary differs'):
        _build(sources)
