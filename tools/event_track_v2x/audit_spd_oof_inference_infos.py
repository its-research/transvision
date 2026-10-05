#!/usr/bin/env python3
"""Independently rehash held-out views and compare every portable info to its source."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import time

import numpy as np

ROOT = Path('/Volumes/Data/test/recover-before-fuse')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('numpy', 'array'):
            return np.array
        raise ValueError('nonportable or forbidden pickle global: ' + module + '.' + name)


def equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    a = parser.parse_args()
    if a.receipt.exists():
        raise FileExistsError('audit receipt is create-once')
    admission_path = ROOT / 'receipts/spd-canonical-oof-historical-label-free-infos-audit-v2-20260930.json'
    if sha(admission_path) != 'bf964fcc7fbdbf4b0c482269fda0cc130437a588b51df20f80a49b7b3d6269cb':
        raise ValueError('source infos admission changed')
    admission = json.loads(admission_path.read_bytes())
    module_path = ROOT / 'artifacts/spd-canonical-oof-export-source-recovery-20260930/audit_spd_inference_inputs.py'
    if sha(module_path) != '620f541767918e2ce27962cde8338c0812257e19186cb0f5c527233d7f5c188e':
        raise ValueError('original whitelist auditor changed')
    spec = importlib.util.spec_from_file_location('original_label_free_reader', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = {}
    for side in ('vehicle-side', 'infrastructure-side'):
        p = ROOT / 'artifacts/spd-canonical-oof-label-free-infos-source-20260930' / side / 'image-pose-infos.pkl'
        if sha(p) != admission['source_infos'][side]['sha256']:
            raise ValueError('original infos byte identity changed')
        data = module.Restricted(io.BytesIO(p.read_bytes())).load()
        sources[side] = {r['token']: r for r in data['infos']}
    seen, records = set(), []
    started = time.monotonic()
    for fold in range(5):
        root = a.inputs / ('fold-%d' % fold)
        m = json.loads((root / 'input-manifest.json').read_bytes())
        original = ROOT / ('artifacts/spd-canonical-oof-heldout-image-pose-20260930/fold-%d' % fold)
        if sha(original / 'input-manifest.json') != m['source_heldout_input_manifest_sha256']:
            raise ValueError('original held-out input changed')
        orig = json.loads((original / 'input-manifest.json').read_bytes())
        for key in ('fold_id', 'cohort', 'train_sequences', 'excluded_fit_sequence_ids',
                    'canonical_fivefold_manifest_sha256', 'official_split_sha256', 'frames',
                    'gt_payloads_in_package', 'gt_payloads_read', 'test_payloads_read', 'val_payloads_read'):
            if m[key] != orig[key]:
                raise ValueError('inference view changed cohort boundary: ' + key)
        for rec in m['payload_inventory']:
            p = root / rec['path']
            if p.is_symlink() or '..' in Path(rec['path']).parts or Path(rec['path']).is_absolute():
                raise ValueError('unsafe payload record')
            if p.stat().st_size != rec['bytes'] or sha(p) != rec['sha256']:
                raise ValueError('payload hash/size differs')
        expected_files = {r['path'] for r in m['payload_inventory']} | {'input-manifest.json'}
        expected_files |= {s + '/' + n for s in ('vehicle-side', 'infrastructure-side')
                           for n in ('data_info.json', 'frame-index.json')}
        actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        if actual_files != expected_files:
            raise ValueError('view has extra/missing files')
        for side in ('vehicle-side', 'infrastructure-side'):
            index_path = root / side / 'frame-index.json'
            if sha(index_path) != m[side]['frame_index_sha256']:
                raise ValueError('frame index changed')
            if sha(root / side / 'data_info.json') != orig[side]['metadata_sha256']:
                raise ValueError('raw camera/calibration metadata changed')
            index = {r['frame_id']: r for r in json.loads(index_path.read_bytes())}
            info_path = root / side / 'image-pose-infos.pkl'
            if sha(info_path) != m[side]['infos_sha256']:
                raise ValueError('infos hash changed')
            data = Restricted(io.BytesIO(info_path.read_bytes())).load()
            if set(data) != {'infos', 'metadata'} or data['metadata'] != {'version': 'v1.0-trainval'}:
                raise ValueError('unknown pickle envelope')
            if len(data['infos']) != len(index) or len(index) != m['frames'][side]:
                raise ValueError('infos/frame count differs')
            for row in data['infos']:
                key = (side, row['token'])
                if key in seen or row['token'] not in index:
                    raise ValueError('OOF frame missing/repeated')
                camera = next(iter(row['cams'].values()))
                if camera['data_path'] != side + '/image/' + row['token'] + '.jpg':
                    raise ValueError('image path changed unexpectedly')
                if sha(root / camera['data_path']) != index[row['token']]['image_sha256']:
                    raise ValueError('actual image SHA differs from frame index')
                # Normalize just the declared path rewrite for comparison with the source.
                camera['data_path'] = side + '/images/' + row['token'] + '.jpg'
                module.check_row(row, index[row['token']], side)
                if not equal(row, sources[side][row['token']]):
                    raise ValueError('infos changed source fields or array values/dtypes')
                seen.add(key)
        split = json.loads((root / 'train-split.json').read_bytes())
        if split != {'batch_split': {'train': m['train_sequences'], 'val': [], 'test': [], 'test_A': []}}:
            raise ValueError('inference split includes non-held-out cohort')
        records.append({'fold_id': fold, 'input_manifest_sha256': sha(root / 'input-manifest.json'),
                        'frames': m['frames'], 'infos_sha256': {s: m[s]['infos_sha256']
                        for s in ('vehicle-side', 'infrastructure-side')}})
        print(json.dumps({'fold': fold, 'frames_verified': len(seen),
                          'remaining_readback_eta_seconds': (time.monotonic()-started)/(fold+1)*(4-fold)}), flush=True)
    if len(seen) != 16338:
        raise ValueError('not complete canonical OOF subject-frame union')
    receipt = {'kind': 'spd_canonical_oof_inference_infos_independent_readback_v1',
               'auditor_sha256': sha(Path(__file__)), 'folds': records, 'total_frames': len(seen),
               'all_payload_hashes_verified': True, 'all_image_hashes_verified': True,
               'source_info_values_dtypes_preserved': True, 'gt_or_future_fields_present': False,
               'legacy_runtime_readback_passed': False, 'actual_inference_complete': False,
               'formal_paper_eligible': False, 'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    with a.receipt.open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    print('OOF_INFERENCE_INFOS_INDEPENDENTLY_ACCEPTED', sha(a.receipt), flush=True)


if __name__ == '__main__':
    main()
