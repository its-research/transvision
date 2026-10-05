"""Independent complete raw-to-V2 mathematical and source-schedule audit.

The expected values below do not import the producer, raw_state, apply_score,
DetectionCacheV2, or calibration fitting code. This audit uses the fixed 1e-8
cache/state tolerance and never creates labels or a measured arrival history.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rbf_nested_seen_val_v2_common import OUT, admitted_seed, new, register, sha

CLASSES = ('car', 'bicycle', 'pedestrian')
ARRAYS = {'states', 'raw_scores', 'scores', 'class_indices', 'covariances', 'appearance', 'appearance_valid'}
RAW_ARRAYS = {'boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy', 'gravity_centers_lidar', 'scores',
              'class_indices', 'appearance_128', 'appearance_valid', 'image_rois_xyxy'}
ATOL = RTOL = 1e-8


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def record_path(root, record):
    assert set(record) == {'path', 'bytes', 'sha256'}
    relative = Path(record['path'])
    assert not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == record['path']
    path = root / relative
    assert path.is_file() and not path.is_symlink()
    assert path.resolve().is_relative_to(root.resolve())
    assert not any((root / Path(*relative.parts[:n])).is_symlink() for n in range(1, len(relative.parts) + 1))
    assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
    return path


def read_arrays(path, keys):
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == keys
        arrays = {key: payload[key] for key in keys}
    assert all(not array.dtype.hasobject and np.isfinite(array).all() for array in arrays.values())
    return arrays


def frame_digest(metadata_bytes, arrays):
    digest = hashlib.sha256(metadata_bytes)
    for key in sorted(ARRAYS):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(array.dtype.str.encode())
        digest.update(canonical(list(array.shape)))
        digest.update(array.tobytes())
    return digest.hexdigest()


def compare(actual, expected, key, maxima):
    assert actual.shape == expected.shape
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0))
    maxima[key] = max(maxima.get(key, 0), error)
    assert np.allclose(actual, expected, atol=ATOL, rtol=RTOL), key


def expected_arrays(raw, models):
    """Direct float64 gravity/yaw and logistic/covariance equations."""
    bottom = raw['boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy'].astype(np.float64)
    centers = raw['gravity_centers_lidar'].astype(np.float64)
    assert bottom.shape == (900, 9) and centers.shape == (900, 3)
    bottom_center = np.column_stack((bottom[:, 0], bottom[:, 1], bottom[:, 2] + bottom[:, 5] / 2))
    # This is the already frozen raw box convention check, not cache tolerance.
    assert np.allclose(centers, bottom_center, atol=1e-4, rtol=1e-5)
    states = np.column_stack((centers, bottom[:, 3:6],
        np.remainder(bottom[:, 6] + np.pi, 2 * np.pi) - np.pi, bottom[:, 7:9]))
    assert (states[:, 3:6] > 0).all()
    assert raw['scores'].shape == (900,) and ((raw['scores'] >= 0) & (raw['scores'] <= 1)).all()
    classes = raw['class_indices']
    assert classes.shape == (900,) and classes.dtype.kind in 'iu' and ((classes >= 0) & (classes < 3)).all()
    scores = np.empty(900, dtype=np.float64)
    covariance = np.empty((900, 9, 9), dtype=np.float64)
    for index, name in enumerate(CLASSES):
        selected = classes == index
        score = models[name]['score']
        epsilon = score['logit_clip']
        assert 0 < epsilon < .5
        p = np.maximum(epsilon, np.minimum(1 - epsilon, raw['scores'][selected].astype(np.float64)))
        logits = np.maximum(-80, np.minimum(80,
            score['slope'] * (np.log(p) - np.log1p(-p)) + score['intercept']))
        scores[selected] = (1 + np.exp(-logits)) ** -1
        matrix = np.asarray(models[name]['covariance']['matrix'], dtype=np.float64)
        assert matrix.shape == (9, 9) and np.isfinite(matrix).all()
        assert np.allclose(matrix, matrix.T, atol=1e-10)
        np.linalg.cholesky(matrix)
        covariance[selected] = matrix
    assert raw['appearance_128'].shape == (900, 128) and raw['appearance_valid'].shape == (900,)
    assert raw['appearance_valid'].dtype.kind == 'b'
    assert raw['image_rois_xyxy'].shape == (900, 4)
    return dict(states=states, scores=scores, covariances=covariance,
        raw_scores=raw['scores'], class_indices=classes,
        appearance=raw['appearance_128'], appearance_valid=raw['appearance_valid'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(1337, 2027, 3407), required=True)
    args = parser.parse_args()
    gate = admitted_seed(args.seed)
    destination = OUT / f'seed{args.seed}'
    cache = destination / 'cache'
    final = destination / 'independent-acceptance.json'
    if final.exists():
        raise ValueError('independent admission already exists; do not repeat')
    assert json.loads((destination / 'input-gate.json').read_bytes()) == gate
    calibration = json.loads(Path(gate['calibration']).read_bytes())
    inputs = Path(gate['inputs'])
    input_manifest = json.loads((inputs / 'input-manifest.json').read_bytes())
    expected_rows = {}
    for side in ('vehicle-side', 'infrastructure-side'):
        index_path = inputs / side / 'frame-index.json'
        assert sha(index_path) == input_manifest[side]['frame_index_sha256']
        rows = json.loads(index_path.read_bytes())
        assert len(rows) == input_manifest['frames'][side]
        for row in rows:
            identity = side, row['sequence_id'], row['frame_id']
            assert identity not in expected_rows
            expected_rows[identity] = row
    assert len(expected_rows) == 7189
    manifest_path = cache / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == canonical(manifest)
    assert set(manifest) == {'kind', 'schema_version', 'split', 'sequences', 'frames', 'frame_count',
        'detection_count', 'source_manifests', 'calibration_sha256', 'dataset_sha256', 'gt_in_cache', 'test_payloads_read'}
    assert manifest['kind'] == 'detection_cache_v2_manifest' and manifest['schema_version'] == 2
    assert manifest['split'] == 'val' and manifest['sequences'] == input_manifest['validation_sequences']
    assert manifest['gt_in_cache'] is False and manifest['test_payloads_read'] is False
    assert manifest['calibration_sha256'] == gate['calibration_sha256']
    assert manifest['dataset_sha256'] == sha(inputs / 'input-manifest.json')
    assert manifest['frame_count'] == len(manifest['frames']) == 7189
    sealed = {}
    expected_files = {'manifest.json'}
    for entry in manifest['frames']:
        assert set(entry) == {'metadata', 'arrays', 'detections', 'frame_sha256'}
        meta_path = record_path(cache, entry['metadata'])
        array_path = record_path(cache, entry['arrays'])
        metadata_bytes = meta_path.read_bytes()
        meta = json.loads(metadata_bytes)
        assert metadata_bytes == canonical(meta)
        identity = meta['side'], meta['sequence_id'], meta['frame_id']
        assert identity not in sealed and identity in expected_rows
        assert entry['detections'] == 900 and meta['arrays_sha256'] == entry['arrays']['sha256']
        for record in (entry['metadata'], entry['arrays']):
            assert record['path'] not in expected_files
            expected_files.add(record['path'])
        sealed[identity] = (entry, meta, metadata_bytes, array_path)
    assert set(sealed) == set(expected_rows)
    assert {path.relative_to(cache).as_posix() for path in cache.rglob('*') if path.is_file()} == expected_files
    assert not any(path.is_symlink() for path in cache.rglob('*'))
    counts, maxima, raw_identities, raw_manifests = Counter(), {}, set(), []
    information_times = {}
    started, last = time.monotonic(), 0
    for root_string in gate['raw_roots']:
        root = Path(root_string)
        raw_manifest_path = root / 'raw-cache-manifest.json'
        raw_manifest_sha = sha(raw_manifest_path)
        raw_manifests.append(raw_manifest_sha)
        raw_manifest = json.loads(raw_manifest_path.read_bytes())
        side = raw_manifest['side']
        assert raw_manifest['checkpoint_sha256'] == gate['detector_checkpoint_sha256'][side]
        assert raw_manifest['input_manifest_sha256'] == manifest['dataset_sha256']
        assert raw_manifest['TF32_enabled'] is False and raw_manifest['classes'] == list(CLASSES)
        assert raw_manifest['shard_count'] == 4
        assert raw_manifest['sequences'] == manifest['sequences'][raw_manifest['shard_index']::4]
        shard_frames = 0
        for item in raw_manifest['frames']:
            raw_meta_path = record_path(root, item['metadata'])
            raw_arrays_path = record_path(root, item['arrays'])
            raw_meta = json.loads(raw_meta_path.read_bytes())
            identity = side, raw_meta['sequence_id'], raw_meta['frame_id']
            assert identity not in raw_identities and identity in sealed
            raw_identities.add(identity)
            entry, meta, metadata_bytes, array_path = sealed[identity]
            assert item['detections'] == item['raw_query_count'] == 900
            assert all(raw_meta[key] == value for key, value in expected_rows[identity].items())
            expected_meta = dict(raw_meta, kind='detection_cache_v2', schema_version=2,
                box_layout='mmdet3d-legacy-gravity-dimxyz-yaw-vxy',
                agent_mask=1 if side == 'vehicle-side' else 2, dataset_split='val',
                dataset_sha256=manifest['dataset_sha256'], detector_config_sha256=raw_manifest['resolved_config_sha256'],
                detector_checkpoint_sha256=gate['detector_checkpoint_sha256'][side],
                feature_checkpoint_sha256=raw_manifest['appearance_checkpoint_sha256'],
                feature_method='imagenet-r50-c5-roialign3-mean16x128-l2-v1',
                calibration_sha256=gate['calibration_sha256'], calibration_fit_split='train',
                raw_manifest_sha256=raw_manifest_sha, raw_arrays_sha256=item['arrays']['sha256'],
                raw_metadata_sha256=item['metadata']['sha256'], arrays_sha256=entry['arrays']['sha256'])
            assert meta == expected_meta
            raw = read_arrays(raw_arrays_path, RAW_ARRAYS)
            actual = read_arrays(array_path, ARRAYS)
            expected = expected_arrays(raw, calibration['sides'][side])
            for key in ARRAYS:
                compare(actual[key], expected[key], key, maxima)
            for key in ('raw_scores', 'class_indices', 'appearance', 'appearance_valid'):
                assert actual[key].dtype == expected[key].dtype and np.array_equal(actual[key], expected[key])
            for key in ('states', 'scores', 'covariances'):
                assert actual[key].dtype == np.dtype('float64')
            assert frame_digest(metadata_bytes, actual) == entry['frame_sha256']
            counts['frames'] += 1
            counts['detections'] += 900
            counts[side] += 1
            for index, name in enumerate(CLASSES):
                counts[name] += int((actual['class_indices'] == index).sum())
            information_times[identity] = max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us'])
            shard_frames += 1
            now = time.monotonic()
            if now - last > 20 or counts['frames'] == 7189:
                print(json.dumps(dict(stage='independent_full_raw_to_V2_float64_numeric_audit', seed=args.seed,
                    completed_frames=counts['frames'], total_frames=7189,
                    ETA_seconds=(now - started) * (7189 - counts['frames']) / counts['frames'],
                    ETA_scope='remaining independent raw-to-V2 numeric frames only',
                    max_absolute_error=max(maxima.values()), atol=ATOL, rtol=RTOL)), flush=True)
                last = now
        assert shard_frames == raw_manifest['frame_count']
    assert raw_identities == set(expected_rows) and sorted(raw_manifests) == manifest['source_manifests']
    assert counts['detections'] == manifest['detection_count'] == 7189 * 900
    assert all(counts[side] == input_manifest['frames'][side] for side in ('vehicle-side', 'infrastructure-side'))
    schedule_path = Path(gate['schedule'])
    assert sha(schedule_path) == gate['schedule_sha256']
    schedule = json.loads(schedule_path.read_bytes())
    assert schedule['kind'] == 'spd_official_validation_prediction_schedule_v1'
    assert schedule['contains_ground_truth'] is False and schedule['contains_system_error_offset'] is False
    assert len(schedule['frames']) == 3316
    schedule_ids, eligible = set(), Counter()
    previous_times = {}
    for event in schedule['frames']:
        assert set(event) == {'sequence_id', 'vehicle_frame', 'infrastructure_frame', 'box_reference_timestamp_us'}
        sequence = event['sequence_id']
        timestamp = event['box_reference_timestamp_us']
        assert type(timestamp) is int and timestamp >= previous_times.get(sequence, -1)
        previous_times[sequence] = timestamp
        identity = sequence, event['vehicle_frame'], event['infrastructure_frame'], timestamp
        assert identity not in schedule_ids
        schedule_ids.add(identity)
        for side, key in (('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')):
            source_identity = side, sequence, event[key]
            assert source_identity in information_times
            eligible[side + ('_eligible_at_snapshot' if information_times[source_identity] <= timestamp + 100000
                             else '_not_yet_informationally_available_at_snapshot')] += 1
    assert {identity[0] for identity in schedule_ids} == set(manifest['sequences'])
    assert sha(manifest_path) == json.loads((destination / 'builder-component-readback.json').read_bytes())['manifest_sha256']
    receipt = dict(kind='rbf_nested_seen_val_matching_V2_full_independent_numeric_and_source_schedule_admission_v1',
        seed=args.seed, checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_gate_sha256=sha(destination / 'input-gate.json'), source_sha256=sha(__file__),
        cache=str(cache), cache_manifest_sha256=sha(manifest_path), input_manifest_sha256=manifest['dataset_sha256'],
        calibration_sha256=gate['calibration_sha256'], detector_checkpoint_sha256=gate['detector_checkpoint_sha256'],
        raw_independent_acceptances=gate['raw_independent_acceptances'],
        complete_frame_query_counts=dict(counts), max_absolute_errors=maxima, atol=ATOL, rtol=RTOL,
        full_raw_to_V2_numeric_admission=True, all_payloads_independently_rehashed=True,
        raw_scores_classes_appearance_exactly_preserved=True, no_query_class_or_topk_selection=True,
        no_calibration_refit=True, GT_free=True, test_payloads_read=False,
        original_schedule_sha256=gate['schedule_sha256'], original_schedule_events=len(schedule_ids),
        schedule_kind='scheduled_pair_snapshot_at_reference_plus_100ms',
        information_time_census=dict(eligible), measured_network_arrival_history_verified=False,
        full_detector_CNN_numerical_acceptance=False, full_online_RBF_accepted=False,
        whole_pipeline_strict_isolation_accepted=False, paper_performance_complete=False)
    new(final, receipt)
    register(final, receipt['kind'])
    print(json.dumps(dict(seed=args.seed, matching_V2_full_independent_numeric_admission=True,
        frames=counts['frames'], detections=counts['detections'], receipt=str(final),
        full_online_RBF_accepted=False, paper_performance_complete=False)), flush=True)


if __name__ == '__main__':
    main()
