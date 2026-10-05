#!/usr/bin/env python3
"""Check ClearML-bound label-free infos against all canonical held-out raw poses."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
MANIFEST_SHA = 'b0af688bd03a7e386250b50bc8478c6525d9dda3c92661fff3d60df397d5ebb3'
AUDITOR_SHA = '620f541767918e2ce27962cde8338c0812257e19186cb0f5c527233d7f5c188e'
POSE_ADMISSION_SHA = 'cdfe9e0ac37cdf00ef9db979830e962ec60ff6f9e0262e9c4cdae7871fdd0df9'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(q @ q - 1) > 1e-6:
        raise ValueError('source quaternion is not unit/finite')
    w, x, y, z = q
    # Do not renormalize source values or silently repair calibration.
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    a = parser.parse_args()
    if a.receipt.exists():
        raise FileExistsError('audit receipt is create-once')
    source_manifest = a.source / 'input-manifest.json'
    if sha(source_manifest) != MANIFEST_SHA:
        raise ValueError('source manifest differs from registered ClearML package')
    manifest = json.loads(source_manifest.read_bytes())
    auditor = ROOT / 'artifacts/spd-canonical-oof-export-source-recovery-20260930/audit_spd_inference_inputs.py'
    if sha(auditor) != AUDITOR_SHA:
        raise ValueError('recovered whitelist auditor differs')
    spec = importlib.util.spec_from_file_location('historical_label_free_auditor', auditor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    admission = ROOT / 'receipts/spd-canonical-oof-heldout-raw-poses-independent-readback-20260930.json'
    if sha(admission) != POSE_ADMISSION_SHA:
        raise ValueError('raw pose independent admission differs')
    accepted_tables = {r['fold_id']: r['pose_table_sha256']
                       for r in json.loads(admission.read_bytes())['folds']}
    expected = {}
    for fold in range(5):
        path = ROOT / ('artifacts/spd-canonical-oof-heldout-raw-poses-20260930/fold-%d-raw-poses.json' % fold)
        if sha(path) != accepted_tables[fold]:
            raise ValueError('pose table bytes differ from independent admission')
        table = json.loads(path.read_bytes())
        ip = ROOT / ('artifacts/spd-canonical-oof-heldout-image-pose-20260930/fold-%d/input-manifest.json' % fold)
        if table['input_manifest_sha256'] != sha(ip):
            raise ValueError('accepted pose/input identity differs')
        for row in table['frames']:
            key = (row['side'], row['frame_id'])
            if key in expected:
                raise ValueError('held-out frame repeats across folds')
            expected[key] = row
    seen = set()
    source_files = {}
    max_rotation_error = max_translation_error = 0.0
    started = time.monotonic()
    for side in ('vehicle-side', 'infrastructure-side'):
        path = a.source / side / 'image-pose-infos.pkl'
        if path.is_symlink() or sha(path) != manifest[side]['infos_sha256']:
            raise ValueError('source infos bytes differ')
        data = module.Restricted(io.BytesIO(path.read_bytes())).load()
        if set(data) != {'infos', 'metadata'} or data['metadata'] != {'version': 'v1.0-trainval'}:
            raise ValueError('unexpected infos envelope')
        if len(data['infos']) != manifest['frames'][side]:
            raise ValueError('source frame count differs')
        source_files[side] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        for row in data['infos']:
            key = (side, row['token'])
            if key in seen or key not in expected:
                raise ValueError('unexpected/repeated historical frame')
            raw = expected[key]
            module.check_row(row, {'frame_id': raw['frame_id'], 'sequence_id': raw['sequence_id'],
                                  'box_reference_timestamp_us': raw['box_reference_timestamp_us']}, side)
            l2e, e2g = rotation(row['lidar2ego_rotation']), rotation(row['ego2global_rotation'])
            world_r = (e2g @ l2e).T
            world_t = e2g @ np.asarray(row['lidar2ego_translation']) + row['ego2global_translation']
            re = float(np.max(np.abs(world_r - raw['lidar_to_world_row_rotation'])))
            te = float(np.max(np.abs(world_t - raw['lidar_to_world_translation'])))
            if re > 1e-6 or te > 1e-6:
                raise ValueError('historical infos/raw calibration transform differs: ' + str(key))
            max_rotation_error, max_translation_error = max(max_rotation_error, re), max(max_translation_error, te)
            seen.add(key)
        print(json.dumps({'side': side, 'frames_verified': len(seen),
                          'eta_seconds': (time.monotonic()-started)/len(seen)*(len(expected)-len(seen)),
                          'scope': 'infos identity/fields/poses only; not inference'}), flush=True)
    if seen != set(expected) or len(seen) != 16338:
        raise ValueError('canonical OOF subject frame union coverage differs')
    receipt = {'kind': 'spd_historical_label_free_infos_canonical_oof_audit_v2',
               'source_clearml_task_id': '9199a9d7af164056920dd0fe5d0c0247',
               'source_manifest_sha256': MANIFEST_SHA, 'source_infos': source_files,
               'frames_verified': len(seen), 'whitelist_auditor_sha256': AUDITOR_SHA,
               'raw_pose_admission_sha256': POSE_ADMISSION_SHA,
               'raw_pose_tables_individually_hash_verified': True,
               'auditor_sha256': sha(Path(__file__)),
               'max_rotation_abs_error': max_rotation_error,
               'max_translation_abs_error': max_translation_error,
               'source_poses_renormalized': False, 'gt_payloads_read': False,
               'heldout_image_hashes_rechecked': False, 'generated_fold_infos': False,
               'actual_inference_complete': False, 'formal_paper_eligible': False,
               'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    with a.receipt.open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    print('LABEL_FREE_INFOS_AUDITED', sha(a.receipt), flush=True)


if __name__ == '__main__':
    main()
