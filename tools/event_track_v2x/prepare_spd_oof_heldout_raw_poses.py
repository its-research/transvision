#!/usr/bin/env python3
"""Compose source LiDAR poses from admitted GT-free OOF calibration bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def transform(value):
    if set(value) != {'rotation', 'translation'}:
        raise ValueError('unexpected raw transform fields')
    r = np.asarray(value['rotation'], dtype=np.float64)
    t = np.asarray(value['translation'], dtype=np.float64)
    if (r.shape != (3, 3) or t.shape != (3, 1) or not np.isfinite(r).all()
            or not np.isfinite(t).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-6, rtol=0)):
        raise ValueError('raw calibration is not a finite proper rigid transform')
    return r, t[:, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--acceptance', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if sha(a.acceptance) != '81eb06939e27491036bfcc2d6e684953a3828a2c55e4333d23b50a5172691504':
        raise ValueError('held-out independent admission differs')
    accepted = json.loads(a.acceptance.read_bytes())
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    for fold in accepted['folds']:
        fid = fold['fold_id'];root = a.inputs / ('fold-' + str(fid))
        mp = root / 'input-manifest.json'
        if sha(mp) != fold['input_manifest_sha256']:
            raise ValueError('held-out input identity differs')
        manifest = json.loads(mp.read_bytes())
        inventory = {r['path']:r for r in manifest['payload_inventory']}
        frames = [];sources = {}
        def read(side, relative):
            key = side + '/' + relative
            path = root / key
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) != inventory[key]['bytes'] or digest != inventory[key]['sha256']:
                raise ValueError('calibration byte binding differs')
            sources[key] = digest
            return json.loads(raw)
        for side in ('vehicle-side', 'infrastructure-side'):
            rows = json.loads((root / side / 'data_info.json').read_bytes())
            if sha(root / side / 'data_info.json') != manifest[side]['metadata_sha256']:
                raise ValueError('source row projection differs')
            index = json.loads((root / side / 'frame-index.json').read_bytes())
            lookup = {(r['sequence_id'],r['frame_id']):r for r in index}
            for row in rows:
                identity = (row['sequence_id'],row['frame_id'])
                if side == 'vehicle-side':
                    le, lt = transform(read(side,row['calib_lidar_to_novatel_path'])['transform'])
                    ew, et = transform(read(side,row['calib_novatel_to_world_path']))
                    rotation, translation = (ew @ le).T, ew @ lt + et
                else:
                    r, translation = transform(read(side,row['calib_virtuallidar_to_world_path']))
                    rotation = r.T
                transform({'rotation':rotation.T.tolist(),'translation':translation[:,None].tolist()})
                frames.append({**lookup[identity], 'side':side, 'coordinate_system':'source_lidar',
                    'lidar_to_world_row_rotation':rotation.tolist(),
                    'lidar_to_world_translation':translation.tolist()})
        frames.sort(key=lambda r:(r['side'],r['sequence_id'],r['frame_id']))
        report = {'kind':'spd_oof_heldout_raw_calibration_pose_table_v1','fold_id':fid,
            'input_manifest_sha256':fold['input_manifest_sha256'],
            'input_acceptance_sha256':sha(a.acceptance),'frames':frames,
            'source_calibration_sha256':sources,'gt_inputs':False,'val_or_test_read':False,
            'independent_pose_arrival_created':False,'poses_renormalized':False,
            'detector_predictions_available':False,'formal_paper_eligible':False,
            'scope':'raw transform composition only; independent readback pending'}
        out=a.output / ('fold-' + str(fid) + '-raw-poses.json')
        with out.open('x') as f:json.dump(report,f,sort_keys=True,separators=(',', ':'),allow_nan=False);f.write('\n')
        elapsed=time.monotonic()-started
        print(f'OOF raw poses fold={fid} frames={len(frames)} SHA256={sha(out)} remaining ETA={elapsed/(fid+1)*(4-fid):.1f}s',flush=True)


if __name__ == '__main__':
    main()
