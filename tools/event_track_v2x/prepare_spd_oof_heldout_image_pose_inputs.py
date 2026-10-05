#!/usr/bin/env python3
"""Create GT-free canonical OOF held-out camera/calibration views, once."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from prepare_cooptrack_fold_inputs import safe_file
from prepare_spd_official_oof_fivefold_inputs import (
    MATERIALIZED, MANIFEST, MANIFEST_SHA256, SPLIT, SPLIT_SHA256,
)

SIDES = ('vehicle-side', 'infrastructure-side')
CALIB = frozenset({'calib_camera_intrinsic_path', 'calib_lidar_to_camera_path',
    'calib_lidar_to_novatel_path', 'calib_novatel_to_world_path',
    'calib_virtuallidar_to_camera_path', 'calib_virtuallidar_to_world_path'})
META = frozenset({'sequence_id', 'frame_id', 'image_path', 'image_timestamp',
                  'pointcloud_timestamp'}) | CALIB


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(MANIFEST) != MANIFEST_SHA256 or sha(SPLIT) != SPLIT_SHA256:
        raise ValueError('canonical protocol identity differs')
    admission = json.loads((MATERIALIZED / 'readback.json').read_bytes())
    if admission['status'] != 'file_inventory_verified':
        raise ValueError('raw source not admitted')
    canonical = json.loads(MANIFEST.read_bytes())
    source = MATERIALIZED / 'inputs'
    metadata = {}
    metadata_sha = {}
    for side in SIDES:
        path = safe_file(source, side + '/data_info.json')
        metadata[side] = json.loads(path.read_bytes())
        metadata_sha[side] = sha(path)
        rows = metadata[side]
        if {r['sequence_id'] for r in rows} != set(canonical['sequence_ids']):
            raise ValueError('source is not exactly official train')
        if len({r['frame_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate source frame')
    args.output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    receipts = []
    for fold in canonical['folds']:
        fold_id = fold['fold_id']
        selected = set(fold['held_out_sequence_ids'])
        if selected & set(fold['fit_sequence_ids']):
            raise ValueError('fold overlap')
        out = args.output / ('fold-' + str(fold_id))
        out.mkdir()
        inventory = {}
        indexes = {}
        counts = {}
        print(f'OOF held-out fold={fold_id} preparation ETA=unknown', flush=True)
        for side in SIDES:
            (out / side).mkdir()
            rows = [r for r in metadata[side] if r['sequence_id'] in selected]
            if {r['sequence_id'] for r in rows} != selected:
                raise ValueError('held-out sequence missing')
            index = []
            projected = []
            for row in rows:
                for key in ('image_path', *sorted(CALIB & row.keys())):
                    relative = side + '/' + row[key]
                    if relative in inventory:
                        continue
                    src = safe_file(source, relative)
                    if key == 'image_path':
                        if not row[key].startswith('image/') or src.suffix != '.jpg':
                            raise ValueError('unexpected image path')
                    elif not row[key].startswith('calib/') or src.suffix != '.json':
                        raise ValueError('unexpected calibration path')
                    raw = src.read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    dst = out / relative
                    dst.parent.mkdir(exist_ok=True, parents=True)
                    with dst.open('xb') as stream:
                        stream.write(raw)
                    if sha(dst) != digest:
                        raise ValueError('copied payload differs')
                    inventory[relative] = {'path': relative, 'bytes': len(raw), 'sha256': digest}
                projected.append({k: row[k] for k in sorted(META & row.keys())})
                index.append({'sequence_id': row['sequence_id'], 'frame_id': row['frame_id'],
                    'source_image_timestamp_us': int(row['image_timestamp']),
                    'box_reference_timestamp_us': int(row['pointcloud_timestamp']),
                    'image_sha256': inventory[side + '/' + row['image_path']]['sha256']})
            index.sort(key=lambda r: (r['sequence_id'], r['frame_id']))
            write(out / side / 'data_info.json', projected)
            write(out / side / 'frame-index.json', index)
            counts[side] = len(rows)
            indexes[side] = {'frame_index_sha256': sha(out / side / 'frame-index.json'),
                            'metadata_sha256': sha(out / side / 'data_info.json')}
        manifest = {'kind': 'eventtrack_train_image_pose_inputs_v1',
            'cohort': 'canonical-oof-held-out', 'fold_id': fold_id,
            'train_sequences': sorted(selected), 'excluded_fit_sequence_ids': fold['fit_sequence_ids'],
            'canonical_fivefold_manifest_sha256': MANIFEST_SHA256,
            'official_split_sha256': SPLIT_SHA256, 'source_metadata_sha256': metadata_sha,
            'source_admission_sha256': sha(MATERIALIZED / 'readback.json'),
            'frames': counts, **indexes, 'payload_inventory': sorted(inventory.values(), key=lambda r: r['path']),
            'gt_payloads_in_package': False, 'gt_payloads_read': False,
            'test_payloads_read': False, 'val_payloads_read': False,
            'transforms_computed': False, 'raw_calibrations_preserved': True,
            'detector_predictions_available': False, 'formal_paper_eligible': False}
        write(out / 'input-manifest.json', manifest)
        receipts.append({'fold_id': fold_id, 'input_manifest_sha256': sha(out / 'input-manifest.json'),
                         'frames': counts, 'payload_files': len(inventory)})
        elapsed = time.monotonic() - started
        print(f'OOF held-out fold={fold_id} copied; remaining preparation ETA={elapsed/(fold_id+1)*(4-fold_id):.1f}s', flush=True)
    write(args.output / 'preparation-receipt.json', {'kind': 'spd_oof_heldout_gtfree_preparation_v1',
        'folds': receipts, 'scope': 'image/calibration copying only; independent readback pending',
        'created_at_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
