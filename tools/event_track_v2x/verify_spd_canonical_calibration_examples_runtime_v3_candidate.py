#!/usr/bin/env python3
"""Rebuild fit calibration labels/residuals from accepted raw caches and GT.

Matching and residual construction are separate from the producer. The admitted
fit supervision loader and past-only velocity reconstruction are shared.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_spd_canonical_oof_calibration_runtime_v3_candidate import admit_fit_sources
from spd_canonical_oof_calibration_supervision import load_fit_supervision, sha, need
from spd_canonical_oof_calibration_examples import sequence_velocity_targets
from verify_spd_canonical_calibration_parameters_candidate import verify as verify_parameters

CLASSES = ('car', 'bicycle', 'pedestrian')
SIDES = ('vehicle-side', 'infrastructure-side')


def frame_oracle(arrays, gt, gt_classes, velocity_valid):
    state = np.array(arrays['boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy'], dtype=float, copy=True)
    state[:, :3] = arrays['gravity_centers_lidar']
    state[:, 6] = (state[:, 6]+np.pi) % (2*np.pi)-np.pi
    scores, classes = arrays['scores'], arrays['class_indices']
    order = np.argsort(-scores, kind='stable')
    selected = order[scores[order] >= .05][:64]
    assigned, used = np.full(len(selected), -1, dtype=np.int64), set()
    for position, raw_index in enumerate(selected):
        candidates = [j for j in range(len(gt)) if gt_classes[j] == classes[raw_index] and j not in used]
        if candidates:
            distances = np.linalg.norm(gt[candidates, :2]-state[raw_index, :2], axis=1)
            local = int(np.argmin(distances))
            if distances[local] < 2.:
                assigned[position] = candidates[local]
                used.add(candidates[local])
    result = {}
    for class_id, name in enumerate(CLASSES):
        positions = np.flatnonzero(classes[selected] == class_id)
        usable = [i for i in positions if assigned[i] >= 0 and velocity_valid[assigned[i]]]
        residuals = state[selected[usable]]-gt[assigned[usable]]
        residuals[:, 6] = (residuals[:, 6]+np.pi) % (2*np.pi)-np.pi
        result[name] = {'scores': scores[selected[positions]].copy(),
                        'targets': (assigned[positions] >= 0).astype(np.uint8),
                        'residuals': residuals}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('package', 'converted', 'fit-inputs', 'heldout-inputs', 'raw-readback',
                'byte-freeze', 'raw-poses', 'calibration', 'receipt'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--raw-readback-sha256', required=True)
    p.add_argument('--calibration-sha256', required=True)
    a = p.parse_args()
    need(not a.receipt.exists(), 'example reconstruction receipt already exists')
    parameters = verify_parameters(a.calibration, a.calibration_sha256)
    c = json.loads(a.calibration.read_bytes())
    package, raw_proof, freeze, roots, manifests = admit_fit_sources(a)
    need(c.get('raw_fit_readback_sha256') == sha(a.raw_readback)
         and c.get('byte_freeze_sha256') == sha(a.byte_freeze)
         and c.get('fit_sequences') == package['fit_sequence_ids']
         and c.get('canonical_oof_binding') == {k: package[k] for k in (
             'fold_id', 'held_out_sequence_ids', 'official_split_sha256', 'canonical_fivefold_manifest_sha256')},
         'calibration source/fold binding differs')
    rows, supervision = load_fit_supervision(a.converted, a.package/'package-manifest.json', a.fit_inputs)
    need(c['supervision'] == supervision, 'calibration supervision source differs')
    metadata = {}
    for root, manifest in zip(roots, manifests):
        for item in manifest['frames']:
            meta = json.loads((root/item['metadata']['path']).read_bytes())
            key = (meta['side'], meta['sequence_id'], meta['frame_id'])
            need(key not in metadata and key in rows, 'duplicate/unknown fit prediction frame')
            metadata[key] = meta
    need(set(metadata) == set(rows), 'fit prediction/supervision coverage differs')
    targets = sequence_velocity_targets([dict(row, metadata=metadata[key]) for key, row in rows.items()])
    groups = {side: {name: [] for name in CLASSES} for side in SIDES}
    done, started = 0, time.monotonic()
    for root, manifest in zip(roots, manifests):
        for item in manifest['frames']:
            meta = json.loads((root/item['metadata']['path']).read_bytes())
            key = (meta['side'], meta['sequence_id'], meta['frame_id'])
            with np.load(root/item['arrays']['path'], allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            gt, valid = targets[key]
            rebuilt = frame_oracle(arrays, gt, rows[key]['classes'], valid)
            for name in CLASSES:
                groups[meta['side']][name].append(rebuilt[name])
            done += 1
            if done == 1 or done % 500 == 0 or done == len(rows):
                print('CALIBRATION_EXAMPLE_READBACK '+json.dumps({'frames': done, 'total': len(rows),
                      'eta_seconds': (time.monotonic()-started)*(len(rows)-done)/done,
                      'scope': 'fit matching reconstruction only', 'overall_eta': 'unknown'}), flush=True)
    comparisons = []
    for side in SIDES:
        for name in CLASSES:
            record = c['example_records'][side+'/'+name]
            with np.load(a.calibration.parent/record['path'], allow_pickle=False) as z:
                for key in ('scores', 'targets', 'residuals'):
                    rebuilt = np.concatenate([row[key] for row in groups[side][name]])
                    need(z[key].shape == rebuilt.shape and np.array_equal(z[key], rebuilt),
                         'saved calibration examples differ from raw/GT reconstruction: '+side+'/'+name+'/'+key)
            comparisons.append({'side': side, 'class': name, 'example_sha256': record['sha256']})
    result = {'kind': 'canonical_calibration_raw_GT_examples_and_parameters_readback',
              'calibration_sha256': sha(a.calibration), 'raw_fit_readback_sha256': sha(a.raw_readback),
              'fold_id': package['fold_id'], 'fit_frames': done, 'groups': comparisons,
              'independent_parameters_recomputed': parameters['independent_parameters_recomputed'],
              'raw_GT_examples_independently_reconstructed': True,
              'shared_helpers': ['admitted fit supervision loader', 'past-only GT velocity reconstruction'],
              'held_out_GT_used_for_fitting': False, 'val_or_test_read': False,
              'formal_v2_ready': False, 'paper_eligible': False}
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    with a.receipt.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('CALIBRATION_EXAMPLES_INDEPENDENTLY_REBUILT '+sha(a.receipt), flush=True)


if __name__ == '__main__':
    main()
