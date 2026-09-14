#!/usr/bin/env python3
"""Seal raw prediction-only frames as DetectionCacheV2, without refitting anything.

The input image/pose manifest defines exact frame coverage. All detections are
retained; selection is downstream. Validation inputs never enter calibration.
Run ``--verify-only --expected-sha256 HASH`` in a separate process for readback.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.detection_cache_v2 import (
    BOX_LAYOUT, CLASSES, FEATURE_METHOD, SIDES, DetectionCacheV2, canonical,
    contained_file, load_manifest, sha_file,
)
from transvision.models.event_track_v2x.prediction_features import apply_score, raw_state

RAW_META = frozenset({'sequence_id', 'frame_id', 'side', 'source_image_timestamp_us',
    'box_reference_timestamp_us', 'lidar_to_world_row_rotation',
    'lidar_to_world_translation', 'coordinate_system', 'image_sha256'})
DECODE_SOURCE = 'raw_head_before_tracking_summary_v1'


def write_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))


def record(path, root):
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
            'sha256': sha_file(path)}


def checked_record(root, value):
    if (not isinstance(value, dict) or set(value) != {'path', 'bytes', 'sha256'}
            or type(value['bytes']) is not int or value['bytes'] <= 0):
        raise ValueError('invalid raw payload record')
    path = contained_file(root, value['path'])
    if path.stat().st_size != value['bytes'] or sha_file(path) != value['sha256']:
        raise ValueError('raw payload hash or byte count differs')
    return path


def input_cohort(root):
    root = Path(root)
    path = contained_file(root, 'input-manifest.json')
    manifest = json.loads(path.read_bytes())
    if (manifest.get('kind') not in {'eventtrack_validation_image_pose_inputs_v1',
                                   'eventtrack_train_image_pose_inputs_v1'}
            or manifest.get('gt_payloads_in_package') is not False
            or manifest.get('test_payloads_read') is not False):
        raise ValueError('input cohort must contain only train/val image and pose data')
    split = 'val' if manifest['kind'] == 'eventtrack_validation_image_pose_inputs_v1' else 'train'
    if manifest.get('val_payloads_read') is not (split == 'val'):
        raise ValueError('input split/read boundary differs')
    sequences = manifest['validation_sequences' if split == 'val' else 'train_sequences']
    if sequences != sorted(set(sequences)) or not sequences:
        raise ValueError('input sequences must be unique and sorted')
    rows = {}
    for side in SIDES:
        index = contained_file(root, side + '/frame-index.json')
        if sha_file(index) != manifest[side]['frame_index_sha256']:
            raise ValueError('input frame index identity differs')
        index_rows = json.loads(index.read_bytes())
        if len(index_rows) != manifest['frames'][side]:
            raise ValueError('input frame index count differs')
        for row in index_rows:
            if set(row) != {'sequence_id', 'frame_id', 'source_image_timestamp_us',
                            'box_reference_timestamp_us', 'image_sha256'}:
                raise ValueError('unknown input frame index fields')
            identity = (side, row['sequence_id'], row['frame_id'])
            if identity in rows or row['sequence_id'] not in sequences:
                raise ValueError('duplicate or out-of-cohort input frame')
            rows[identity] = row
    if {x[1] for x in rows} != set(sequences):
        raise ValueError('input sequence coverage differs')
    return sha_file(path), split, sequences, rows


def validate_raw_arrays(arrays):
    state = raw_state(arrays)
    n = len(state)
    shapes = {'scores': (n,), 'class_indices': (n,), 'appearance_128': (n, 128),
              'appearance_valid': (n,), 'image_rois_xyxy': (n, 4)}
    for key, shape in shapes.items():
        a = arrays[key]
        if a.shape != shape or a.dtype.hasobject or not np.isfinite(a).all():
            raise ValueError('invalid raw array ' + key)
    if (arrays['class_indices'].dtype.kind not in 'iu'
            or np.any((arrays['class_indices'] < 0) | (arrays['class_indices'] > 2))
            or arrays['appearance_valid'].dtype.kind != 'b'
            or np.any((arrays['scores'] < 0) | (arrays['scores'] > 1))):
        raise ValueError('invalid raw class, visibility, or score')
    b = arrays['boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy']
    if np.any((arrays['scores'] == .5) & (b == np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])).all(1)):
        raise ValueError('legacy masked-summary placeholder is forbidden')
    return state


def build_cache(raw_roots, calibration_path, expected_calibration_sha256, inputs, output):
    """Create a fresh V2 root, bound to all raw payloads and exact input cohort."""
    output, calibration_path = Path(output), Path(calibration_path)
    if output.exists() or output.is_symlink():
        raise ValueError('V2 destination must not already exist')
    if calibration_path.is_symlink() or sha_file(calibration_path) != expected_calibration_sha256:
        raise ValueError('frozen calibration identity differs')
    calibration = json.loads(calibration_path.read_bytes())
    if (calibration.get('kind') != 'eventtrack_train_calibration_v1'
            or calibration['evidence']['official_validation_used_for_selection'] is not False
            or calibration['evidence']['test_payloads_read'] is not False):
        raise ValueError('calibration was not frozen using training data only')
    dataset_sha, split, sequences, expected_rows = input_cohort(inputs)
    if split == 'val' and set(sequences) & set(calibration['fit_sequences']):
        raise ValueError('calibration fit sequence overlaps validation')
    if set(calibration['sides']) != set(SIDES):
        raise ValueError('calibration source groups differ')
    for side in SIDES:
        if set(calibration['sides'][side]) != set(CLASSES):
            raise ValueError('calibration coarse classes differ')
    output.mkdir(parents=True, exist_ok=False)
    frames, identities, manifest_shas, shards = [], set(), [], set()
    for raw_root in sorted(map(Path, raw_roots), key=str):
        if raw_root.is_symlink():
            raise ValueError('raw root is a symlink')
        manifest_path = contained_file(raw_root, 'raw-cache-manifest.json')
        raw = json.loads(manifest_path.read_bytes())
        raw_sha = sha_file(manifest_path)
        if raw_sha in manifest_shas:
            raise ValueError('raw manifest reused')
        manifest_shas.append(raw_sha)
        side = raw['side']
        if (raw['kind'] != 'eventtrack_raw_detector_cache_v1' or side not in SIDES
                or raw['input_manifest_sha256'] != dataset_sha or raw['classes'] != list(CLASSES)
                or raw['appearance_method'] != FEATURE_METHOD
                or raw['detector_decode_source'] != DECODE_SOURCE
                or raw['formal_v2_ready'] is not False or raw['gt_inputs'] is not False
                or raw['test_payloads_read'] is not False or raw['val_payloads_read'] is not (split == 'val')
                or raw['optimizer_created'] is not False or raw['score_filter_changed'] is not False
                or raw['weights_unchanged'] is not True or raw['covariance_calibrated'] is not False
                or raw['raw_head_frames_verified'] != raw['frame_count'] or raw['legacy_placeholder_count'] != 0
                or raw['shard_count'] != 2 or raw['shard_index'] not in {0, 1}
                or raw['sequences'] != sequences[raw['shard_index']::2]):
            raise ValueError('raw cache source, fixed-weight, split, or decode boundary differs')
        shard = (side, raw['shard_index'])
        if shard in shards:
            raise ValueError('duplicate raw shard')
        shards.add(shard)
        config = contained_file(raw_root, 'resolved-cache-config.py')
        if sha_file(config) != raw['resolved_config_sha256']:
            raise ValueError('raw detector config differs')
        launch = json.loads(contained_file(raw_root, 'launch-receipt.json').read_bytes())
        if (launch['frames'] != raw['frame_count']
                or any(raw.get(k) != v for k, v in launch.items() if k not in {'kind', 'frames'})):
            raise ValueError('raw launch and completion differ')
        expected_files = {'raw-cache-manifest.json', 'launch-receipt.json', 'resolved-cache-config.py'}
        shard_count = Counter()
        for item in raw['frames']:
            if set(item) != {'arrays', 'metadata', 'detections'} or type(item['detections']) is not int:
                raise ValueError('unknown raw frame inventory fields')
            payloads = {}
            for key in ['arrays', 'metadata']:
                payloads[key] = checked_record(raw_root, item[key])
                if item[key]['path'] in expected_files:
                    raise ValueError('raw payload path reused')
                expected_files.add(item[key]['path'])
            meta = json.loads(payloads['metadata'].read_bytes())
            if set(meta) != RAW_META or meta['side'] != side or meta['sequence_id'] not in raw['sequences']:
                raise ValueError('unknown raw metadata or source/sequence differs')
            identity = (side, meta['sequence_id'], meta['frame_id'])
            row = expected_rows.get(identity)
            if identity in identities or row is None or any(meta[k] != row[k] for k in row):
                raise ValueError('raw frame does not match the exact input cohort')
            identities.add(identity)
            with np.load(payloads['arrays'], allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            state = validate_raw_arrays(arrays)
            n = len(state)
            if n != item['detections']:
                raise ValueError('raw detection count differs')
            scores = np.empty(n, dtype=np.float64)
            covariances = np.empty((n, 9, 9), dtype=np.float64)
            for c, name in enumerate(CLASSES):
                selected = arrays['class_indices'] == c
                model = calibration['sides'][side][name]
                scores[selected] = apply_score(arrays['scores'][selected].astype(np.float64), model['score'])
                covariances[selected] = np.asarray(model['covariance']['matrix'], dtype=np.float64)
            sealed = dict(states=state, raw_scores=arrays['scores'], scores=scores,
                          class_indices=arrays['class_indices'], covariances=covariances,
                          appearance=arrays['appearance_128'], appearance_valid=arrays['appearance_valid'])
            directory = output / 'frames' / side / meta['sequence_id']
            directory.mkdir(parents=True, exist_ok=True)
            array_path = directory / (meta['frame_id'] + '.npz')
            with array_path.open('xb') as stream:
                np.savez_compressed(stream, **sealed)
            metadata = dict(meta, kind='detection_cache_v2', schema_version=2,
                box_layout=BOX_LAYOUT, agent_mask=SIDES[side], dataset_split=split,
                dataset_sha256=dataset_sha, detector_config_sha256=raw['resolved_config_sha256'],
                detector_checkpoint_sha256=raw['checkpoint_sha256'],
                feature_checkpoint_sha256=raw['appearance_checkpoint_sha256'], feature_method=FEATURE_METHOD,
                calibration_sha256=expected_calibration_sha256, calibration_fit_split='train',
                raw_manifest_sha256=raw_sha, raw_arrays_sha256=item['arrays']['sha256'],
                raw_metadata_sha256=item['metadata']['sha256'], arrays_sha256=sha_file(array_path))
            frame = DetectionCacheV2(canonical(metadata), **sealed)
            metadata_path = directory / (meta['frame_id'] + '.json')
            write_json(metadata_path, metadata)
            frames.append(dict(arrays=record(array_path, output), metadata=record(metadata_path, output),
                               detections=n, frame_sha256=frame.digest()))
            shard_count.update(frames=1, detections=n, appearance_valid=int(frame.appearance_valid.sum()))
        actual = {p.relative_to(raw_root).as_posix() for p in raw_root.rglob('*') if p.is_file()}
        if (actual != expected_files or any(p.is_symlink() for p in raw_root.rglob('*'))
                or shard_count['frames'] != raw['frame_count'] or shard_count['detections'] != raw['detection_count']
                or shard_count['appearance_valid'] != raw['appearance_valid_count']):
            raise ValueError('raw full-tree coverage or totals differ')
        print('EVENTTRACK_V2_SHARD ' + json.dumps({'side': side, 'shard': raw['shard_index'], **shard_count}), flush=True)
    if identities != set(expected_rows) or shards != {(s, i) for s in SIDES for i in [0, 1]}:
        raise ValueError('raw shards do not cover the complete input cohort')
    frames.sort(key=lambda entry: entry['metadata']['path'])
    manifest = dict(kind='detection_cache_v2_manifest', schema_version=2, split=split,
        sequences=sequences, frames=frames, frame_count=len(frames),
        detection_count=sum(x['detections'] for x in frames), source_manifests=sorted(manifest_shas),
        calibration_sha256=expected_calibration_sha256, dataset_sha256=dataset_sha,
        gt_in_cache=False, test_payloads_read=False)
    write_json(output / 'manifest.json', manifest)
    return sha_file(output / 'manifest.json')


def verify(root, expected_sha256, inputs=None):
    manifest, entries = load_manifest(root, expected_sha256)
    counts, identities, appearance = Counter(), set(), 0
    for entry in entries:
        frame = DetectionCacheV2.load(root, entry)
        meta = frame.metadata
        counts[meta['side']] += 1
        identities.add((meta['side'], meta['sequence_id'], meta['frame_id']))
        appearance += int(frame.appearance_valid.sum())
    if inputs is not None:
        dataset_sha, split, sequences, rows = input_cohort(inputs)
        if (manifest['dataset_sha256'] != dataset_sha or manifest['split'] != split
                or manifest['sequences'] != sequences or identities != set(rows)):
            raise ValueError('sealed cache differs from exact input cohort')
    return dict(kind='detection_cache_v2_full_readback', manifest_sha256=expected_sha256,
        frames=manifest['frame_count'], detections=manifest['detection_count'], sequences=len(manifest['sequences']),
        source_frame_counts=dict(counts), appearance_valid=appearance, all_payloads_rehashed=True,
        exact_input_cohort_verified=inputs is not None, gt_in_cache=False, test_payloads_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', action='append', type=Path, default=[])
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--calibration-sha256')
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--expected-sha256')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.verify_only:
        if not args.expected_sha256:
            parser.error('--verify-only requires --expected-sha256')
        proof = verify(args.output, args.expected_sha256, args.inputs)
    else:
        if not args.raw_root or not args.calibration or not args.calibration_sha256:
            parser.error('building requires raw roots and hash-pinned calibration')
        sha = build_cache(args.raw_root, args.calibration, args.calibration_sha256, args.inputs, args.output)
        proof = verify(args.output, sha, args.inputs)
    if args.receipt:
        write_json(args.receipt, proof)
    print('EVENTTRACK_DETECTION_CACHE_V2 ' + json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
