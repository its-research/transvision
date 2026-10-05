#!/usr/bin/env python3
"""Fit-only canonical SPD calibration candidate after independent raw acceptance.

No held-out diagnostics, official val or paper acceptance. Retains the recovered
historical numerical recipe, with explicit in-sample detector fit provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from spd_canonical_oof_calibration_supervision import load_fit_supervision, sha, need
from spd_canonical_oof_calibration_examples import frame_examples, fit_groups, sequence_velocity_targets
from spd_canonical_oof_calibration_boundary import validate_input_boundary
from audit_spd_oof_fit_feature_raw_cache_raw_pose_v2 import audit
from transvision.models.event_track_v2x.prediction_features import raw_state, CLASSES
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file

SIDES = ('vehicle-side', 'infrastructure-side')
SCORE = {'logit_clip': 1e-6, 'l2': .0001}
COVARIANCE = {'minimum_class_samples': 32, 'shrinkage': .1,
              'floor_std': [.05]*6 + [.01, .1, .1]}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def admit_fit_sources(a):
    package_path = a.package / 'package-manifest.json'
    package = json.loads(package_path.read_bytes())
    inputs = json.loads((a.fit_inputs / 'input-manifest.json').read_bytes())
    held = json.loads((a.heldout_inputs / 'input-manifest.json').read_bytes())
    validate_input_boundary(package, inputs, held)
    need(sha(a.raw_readback) == a.raw_readback_sha256 and not a.raw_readback.is_symlink(),
         'raw independent acceptance bytes differ')
    proof = json.loads(a.raw_readback.read_bytes())
    freeze = json.loads(a.byte_freeze.read_bytes())
    need(proof.get('kind') == 'spd_canonical_oof_fit_feature_raw_pose_v2_cloud_independent_content_readback'
         and proof.get('status') == 'all_cloud_bytes_frames_arrays_raw_poses_verified'
         and proof.get('source_task_id') == '13babb70d6e34b0ab9dc5c60b166a1ce'
         and proof.get('acceptor_sha256') == 'b6fc8a70cb1848bef600713e41c963af8c2964b0bc89078a0cb234c8d2de61c3'
         and proof.get('fold_id') == package['fold_id']
         and proof.get('seed') == freeze['seed'] and proof.get('training_task_id') == freeze['task_id']
         and proof.get('byte_freeze_sha256') == sha(a.byte_freeze)
         and proof.get('held_out_selection_scoring_eligible') is False
         and freeze.get('byte_freeze_accepted') is True
         and freeze.get('package_manifest_sha256') == sha(package_path),
         'raw source is not independently accepted corresponding fit export')
    from clearml import Task
    task = Task.get_task(task_id=proof['task_id'])
    need(str(task.status) == 'completed', 'fit export no longer completed')
    for name, record in proof['artifacts'].items():
        path = contained_file(a.raw_readback.parent, record['path'])
        need(path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
             and task.artifacts[name].hash == record['sha256']
             and task.artifacts[name].size == record['bytes'], 'fit cloud artifact changed')
    roots, manifests = [], []
    poses = a.raw_poses
    for side in SIDES:
        for shard in (0, 1):
            prefix = side + '-shard-%d' % shard
            root = a.raw_readback.parent / (prefix+'-cache')
            path = root / 'raw-cache-manifest.json'
            need(sha(path) == proof['artifacts'][prefix+'-raw-manifest']['sha256'],
                 'extracted fit cache differs from accepted cloud manifest')
            manifest = json.loads(path.read_bytes())
            need(manifest['side'] == side and manifest['shard_index'] == shard
                 and manifest['fold_id'] == package['fold_id']
                 and manifest['checkpoint_sha256'] == freeze['artifacts'][side+'-final-checkpoint']['sha256']
                 and manifest['resolved_config_sha256'] == freeze['artifacts'][side+'-detector.py']['sha256'],
                 'fit checkpoint/config/shard differs from training freeze')
            audit(root, a.fit_inputs, package_path, poses)
            roots.append(root)
            manifests.append(manifest)
    return package, proof, freeze, roots, manifests


def add_frame_examples(groups, side, arrays, supervision, targets):
    gt, velocity_valid = targets
    examples = frame_examples(raw_state(arrays), arrays['scores'], arrays['class_indices'],
                              gt, supervision['classes'], velocity_valid)
    for name in CLASSES:
        groups[side][name].append(examples['groups'][name])
    return examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('package', 'converted', 'fit-inputs', 'heldout-inputs', 'raw-readback',
                'byte-freeze', 'raw-poses', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--raw-readback-sha256', required=True)
    a = p.parse_args()
    need(not a.output.exists() and not a.output.is_symlink(), 'candidate output already exists')
    package, raw_proof, freeze, roots, manifests = admit_fit_sources(a)
    rows, supervision = load_fit_supervision(a.converted, a.package/'package-manifest.json', a.fit_inputs)
    metadata = {}
    for root, manifest in zip(roots, manifests):
        for item in manifest['frames']:
            meta = json.loads(contained_file(root, item['metadata']['path']).read_bytes())
            key = (meta['side'], meta['sequence_id'], meta['frame_id'])
            need(key in rows and key not in metadata, 'unknown/duplicate fit prediction frame')
            metadata[key] = meta
    need(set(metadata) == set(rows), 'fit prediction and supervision coverage differs')
    targets = sequence_velocity_targets([dict(row, metadata=metadata[key]) for key, row in rows.items()])
    groups = {side: {name: [] for name in CLASSES} for side in SIDES}
    started, done = time.monotonic(), 0
    for root, manifest in zip(roots, manifests):
        for item in manifest['frames']:
            meta = json.loads(contained_file(root, item['metadata']['path']).read_bytes())
            key = (meta['side'], meta['sequence_id'], meta['frame_id'])
            with np.load(contained_file(root, item['arrays']['path']), allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            add_frame_examples(groups, meta['side'], arrays, rows[key], targets[key])
            done += 1
            if done == 1 or done % 500 == 0 or done == len(rows):
                eta = (time.monotonic()-started)*(len(rows)-done)/done
                print('CALIBRATION_COLLECTION_PROGRESS '+json.dumps({'frames': done, 'total': len(rows),
                      'eta_seconds': eta, 'scope': 'fit example collection only', 'overall_eta': 'unknown'}), flush=True)
    # No output is created until complete raw admission and example collection.
    a.output.mkdir(parents=True, exist_ok=False)
    models, example_records = {}, {}
    for side in SIDES:
        print('CALIBRATION_OPTIMIZER_START '+side+' ETA=unknown', flush=True)
        models[side] = fit_groups(groups[side], score_config=SCORE, covariance_config=COVARIANCE)
        for name in CLASSES:
            values = {key: np.concatenate([r[key] for r in groups[side][name]])
                      for key in ('scores', 'targets', 'residuals')}
            path = a.output / (side+'-'+name+'-examples.npz')
            with path.open('xb') as stream:
                np.savez_compressed(stream, **values)
            example_records[side+'/'+name] = {'path': path.name, 'sha256': sha(path),
                                            'bytes': path.stat().st_size}
    result = {'kind': 'eventtrack_train_calibration_v1', 'seed': freeze['seed'],
              'fit_sequences': package['fit_sequence_ids'], 'sides': models,
              'config': {'score': SCORE, 'covariance': COVARIANCE, 'match_distance_m': 2.},
              'candidate_policy': 'raw-score>=0.05/all-class-top64',
              'canonical_oof_binding': {k: package[k] for k in ('fold_id', 'held_out_sequence_ids',
                   'official_split_sha256', 'canonical_fivefold_manifest_sha256')},
              'training_task_id': freeze['task_id'], 'fit_export_task_id': raw_proof['task_id'],
              'byte_freeze_sha256': sha(a.byte_freeze), 'raw_fit_readback_sha256': sha(a.raw_readback),
              'supervision': supervision, 'example_records': example_records,
              'fitter_sha256': sha(Path(__file__)),
              'evidence': {'in_sample_detector_predictions': True, 'held_out_gt_used_for_fitting': False,
                           'official_validation_used_for_selection': False, 'test_payloads_read': False,
                           'diagnostic_scope': 'fit_only_no_held_out_probability_claim',
                           'independent_parameters_recomputed': False, 'paper_eligible': False}}
    write(a.output/'calibration.json', result)
    write(a.output/'candidate-receipt.json', {'kind': 'canonical_oof_calibration_fitted_candidate',
          'calibration_sha256': sha(a.output/'calibration.json'), 'fold_id': package['fold_id'],
          'fit_frames': len(rows), 'independent_parameters_recomputed': False,
          'formal_v2_ready': False, 'paper_eligible': False})
    print('CALIBRATION_CANDIDATE_WRITTEN '+sha(a.output/'calibration.json'), flush=True)


if __name__ == '__main__':
    main()
