#!/usr/bin/env python3
"""Independently compare all held-out camera/calibration bytes to source."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path('/Volumes/Data/test/recover-before-fuse')
    canon_path = base / 'receipts/spd-official-train-canonical-development-fivefold-20260930.json'
    if sha(canon_path) != '1c9d23f291be09e686a2bdcc60ef19ea58a5c2894978cc5f6a6b2a81f43c2bd6':
        raise ValueError('canonical manifest differs')
    canonical = json.loads(canon_path.read_bytes())
    source = base / 'artifacts/spd-official-train-source-20260930/materialized-inputs/inputs'
    sides = ('vehicle-side', 'infrastructure-side')
    original = {s: json.loads((source / s / 'data_info.json').read_bytes()) for s in sides}
    fields = {'sequence_id', 'frame_id', 'image_path', 'image_timestamp', 'pointcloud_timestamp',
        'calib_camera_intrinsic_path', 'calib_lidar_to_camera_path', 'calib_lidar_to_novatel_path',
        'calib_novatel_to_world_path', 'calib_virtuallidar_to_camera_path', 'calib_virtuallidar_to_world_path'}
    all_frames = Counter()
    folds = []
    for fold in canonical['folds']:
        fid = fold['fold_id'];root = args.root / ('fold-' + str(fid))
        mp = root / 'input-manifest.json';manifest = json.loads(mp.read_bytes())
        held = set(fold['held_out_sequence_ids'])
        assert manifest['fold_id'] == fid and manifest['train_sequences'] == sorted(held)
        assert manifest['excluded_fit_sequence_ids'] == fold['fit_sequence_ids']
        assert manifest['gt_payloads_in_package'] is False and manifest['gt_payloads_read'] is False
        assert manifest['test_payloads_read'] is False and manifest['val_payloads_read'] is False
        assert manifest['formal_paper_eligible'] is False
        expected_paths = set();counts = {};expected_files = {'input-manifest.json'}
        for side in sides:
            selected = [r for r in original[side] if r['sequence_id'] in held]
            expected_meta = [{k:r[k] for k in fields & r.keys()} for r in selected]
            actual_meta = json.loads((root / side / 'data_info.json').read_bytes())
            assert actual_meta == expected_meta
            expected_index = []
            for r in selected:
                all_frames[(side,r['frame_id'])] += 1
                for key in fields & r.keys():
                    if key == 'image_path' or key.startswith('calib_'):
                        expected_paths.add(side + '/' + r[key])
                expected_index.append({'sequence_id':r['sequence_id'],'frame_id':r['frame_id'],
                    'source_image_timestamp_us':int(r['image_timestamp']),
                    'box_reference_timestamp_us':int(r['pointcloud_timestamp']),
                    'image_sha256':sha(source / side / r['image_path'])})
            expected_index.sort(key=lambda r:(r['sequence_id'],r['frame_id']))
            ip=root / side / 'frame-index.json'
            assert json.loads(ip.read_bytes()) == expected_index
            assert sha(ip) == manifest[side]['frame_index_sha256']
            assert sha(root / side / 'data_info.json') == manifest[side]['metadata_sha256']
            counts[side] = len(selected)
            expected_files.update({side + '/frame-index.json',side + '/data_info.json'})
        inventory=manifest['payload_inventory']
        assert len(inventory) == len(expected_paths) and {r['path'] for r in inventory} == expected_paths
        for r in inventory:
            dst=root / r['path'];src=source / r['path']
            assert not dst.is_symlink() and dst.stat().st_size == src.stat().st_size == r['bytes']
            assert sha(dst) == sha(src) == r['sha256']
        expected_files.update(expected_paths)
        assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == expected_files
        assert manifest['frames'] == counts
        folds.append({'fold_id':fid,'input_manifest_sha256':sha(mp),'frames':counts,'payload_files':len(inventory)})
        print('HELDOUT_READBACK fold='+str(fid)+' passed; ETA=unknown',flush=True)
    assert all_frames == Counter((s,r['frame_id']) for s in sides for r in original[s])
    report={'kind':'spd_canonical_oof_heldout_image_pose_independent_readback_v1',
        'status':'all_source_bytes_and_exact_once_frame_coverage_verified','folds':folds,
        'total_frames':len(all_frames),'labels_or_val_test_read':False,
        'detector_predictions_available':False,'formal_paper_eligible':False,
        'scope':'GT-free input isolation and byte identity only; transforms and inference pending',
        'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    with args.output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
