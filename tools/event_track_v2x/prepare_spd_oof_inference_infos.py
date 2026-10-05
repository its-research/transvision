#!/usr/bin/env python3
"""Create immutable held-out inference views using verified GT-free infos only."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
ADMISSION_SHA = 'bf964fcc7fbdbf4b0c482269fda0cc130437a588b51df20f80a49b7b3d6269cb'
INPUT_ADMISSION_SHA = '81eb06939e27491036bfcc2d6e684953a3828a2c55e4333d23b50a5172691504'
AUDITOR_SHA = '620f541767918e2ce27962cde8338c0812257e19186cb0f5c527233d7f5c188e'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n')


class Portable(pickle.Pickler):
    """Use NumPy's stable public array constructor, not NumPy-2 private globals."""
    def reducer_override(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject or not np.isfinite(obj).all():
                raise ValueError('non-numeric/nonfinite array in label-free infos')
            return np.array, (obj.tolist(), obj.dtype.str)
        return NotImplemented


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    ap = ROOT / 'receipts/spd-canonical-oof-historical-label-free-infos-audit-v2-20260930.json'
    ip = ROOT / 'receipts/spd-canonical-oof-heldout-image-pose-independent-readback-20260930.json'
    if sha(ap) != ADMISSION_SHA or sha(ip) != INPUT_ADMISSION_SHA:
        raise ValueError('independent input/infos admission differs')
    admitted = json.loads(ap.read_bytes())
    source = ROOT / 'artifacts/spd-canonical-oof-label-free-infos-source-20260930'
    auditor = ROOT / 'artifacts/spd-canonical-oof-export-source-recovery-20260930/audit_spd_inference_inputs.py'
    if sha(auditor) != AUDITOR_SHA:
        raise ValueError('whitelist reader source differs')
    spec = importlib.util.spec_from_file_location('label_free_reader', auditor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    infos = {}
    for side in ('vehicle-side', 'infrastructure-side'):
        p = source / side / 'image-pose-infos.pkl'
        if sha(p) != admitted['source_infos'][side]['sha256']:
            raise ValueError('source infos identity differs')
        data = module.Restricted(io.BytesIO(p.read_bytes())).load()
        infos[side] = {r['token']: r for r in data['infos']}
        if len(infos[side]) != len(data['infos']):
            raise ValueError('repeated source token')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    for fold in range(5):
        original = ROOT / ('artifacts/spd-canonical-oof-heldout-image-pose-20260930/fold-%d' % fold)
        manifest = json.loads((original / 'input-manifest.json').read_bytes())
        target = a.output / ('fold-%d' % fold)
        target.mkdir()
        # Read-only reuse of already accepted payload bytes; never mutate linked files.
        for p in sorted(original.rglob('*')):
            if p.is_symlink():
                raise ValueError('source view contains symlink')
            rel = p.relative_to(original)
            if p.is_dir():
                (target / rel).mkdir(exist_ok=True, parents=True)
            elif p.is_file() and rel.as_posix() != 'input-manifest.json':
                os.link(p, target / rel)
        inventory = copy.deepcopy(manifest['payload_inventory'])
        for side in ('vehicle-side', 'infrastructure-side'):
            index = json.loads((target / side / 'frame-index.json').read_bytes())
            selected = []
            for entry in index:
                row = copy.deepcopy(infos[side][entry['frame_id']])
                module.check_row(row, entry, side)
                camera = next(iter(row['cams'].values()))
                camera['data_path'] = side + '/image/' + row['token'] + '.jpg'
                if row['scene_token'] not in manifest['train_sequences']:
                    raise ValueError('fit sequence would enter held-out infos')
                selected.append(row)
            p = target / side / 'image-pose-infos.pkl'
            with p.open('xb') as stream:
                Portable(stream, protocol=2).dump({'infos': selected, 'metadata': {'version': 'v1.0-trainval'}})
            manifest[side]['infos_sha256'] = sha(p)
            inventory.append({'path': side + '/image-pose-infos.pkl',
                              'bytes': p.stat().st_size, 'sha256': sha(p)})
        split = target / 'train-split.json'
        write(split, {'batch_split': {'train': manifest['train_sequences'],
                                     'val': [], 'test': [], 'test_A': []}})
        inventory.append({'path': 'train-split.json', 'bytes': split.stat().st_size, 'sha256': sha(split)})
        manifest.update(source_heldout_input_manifest_sha256=sha(original / 'input-manifest.json'),
                        infos_source_admission_sha256=ADMISSION_SHA,
                        infos_encoding='pickle-protocol2-numpy-public-array-v1',
                        inference_infos_available=True,
                        payload_inventory=sorted(inventory, key=lambda r: r['path']))
        write(target / 'input-manifest.json', manifest)
        records.append({'fold_id': fold, 'input_manifest_sha256': sha(target / 'input-manifest.json'),
                        'frames': manifest['frames'], 'infos_sha256': {
                            s: manifest[s]['infos_sha256'] for s in ('vehicle-side', 'infrastructure-side')}})
        print(json.dumps({'fold': fold, 'remaining_preparation_eta_seconds':
                          (time.monotonic()-started)/(fold+1)*(4-fold),
                          'scope': 'input preparation only; readback pending'}), flush=True)
    write(a.output / 'preparation-receipt.json', {
        'kind': 'spd_canonical_oof_heldout_inference_infos_preparation_v1',
        'source_admission_sha256': ADMISSION_SHA, 'preparer_sha256': sha(Path(__file__)),
        'folds': records, 'gt_payloads_read': False, 'independent_readback_passed': False,
        'legacy_runtime_readback_passed': False, 'actual_inference_complete': False,
        'checked_at_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
