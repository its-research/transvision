"""Byte admission of native source labels for offline motion supervision.

This does not consume merged ego GT, parse labels, construct targets or fit.
The selected (sequence, frame key) set must come from separately admitted train
raw rows/partition. Receipt split is checked before opening any label payload.
"""
import hashlib
import json
import stat
from pathlib import Path


def ordinary(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('symlink input forbidden')
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('ordinary file required')
    return path


def admitted_labels(volume, receipt_sha256, selected):
    volume = Path(volume).absolute()
    data = ordinary(volume / 'receipt.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt_sha256:
        raise ValueError('volume receipt SHA differs')
    receipt = json.loads(data)
    if receipt.get('split') != 'train':
        raise ValueError('native official train required before reading labels')
    selected = list(selected)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('nonempty unique train frame selection required')
    inventory = {}
    for row in receipt['files']:
        name = row['path']
        parts = Path(name).parts
        if Path(name).is_absolute() or '..' in parts or len(parts) != 3:
            raise ValueError('native sequence/CAV/file inventory required')
        if name in inventory:
            raise ValueError('duplicate inventory member')
        inventory[name] = row
    # Resolve every selection before opening the first payload, including both
    # native sources. No union deduplication or ego projection is inherited.
    members = []
    for sequence, key in sorted(selected):
        if (not isinstance(sequence, str) or not isinstance(key, str)
                or not sequence or not key or '/' in sequence or '/' in key
                or sequence in ('.', '..') or key in ('.', '..')):
            raise ValueError('plain native sequence and frame key required')
        for cav in ('0', '1'):
            name = f'{sequence}/{cav}/{key}.yaml'
            if name not in inventory:
                raise ValueError('selected source label missing from train inventory')
            members.append((sequence, key, cav, name, inventory[name]))
    for sequence, key, cav, name, row in members:
        payload = ordinary(volume / 'payload' / name).read_bytes()
        if len(payload) != row['bytes'] or hashlib.sha256(payload).hexdigest() != row['sha256']:
            raise ValueError('native label size/SHA differs')
        yield dict(sequence_id=sequence, frame_key=key, cav_id=cav,
                   label_path=name, label_sha256=row['sha256'], label_bytes=payload)
