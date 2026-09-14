#!/usr/bin/env python3
"""Inspect all native YAML in one hash-pinned volume; no tracking GT conversion."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.event_track_v2x.extract_v2v4real_archive import digest_file, ordinary
from tools.event_track_v2x.prepare_v2v4real_inputs import inventory_native_root, _read
from transvision.models.event_track_v2x.v2v4real_inputs import (
    MAX_YAML_BYTES, load_raw_yaml, pose_projection, read_annotations,
)


def audit_volume(volume, receipt_sha256):
    volume = ordinary(volume, directory=True)
    raw = ordinary(volume / 'receipt.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt_sha256:
        raise ValueError('volume receipt SHA-256 differs')
    receipt = json.loads(raw)
    if receipt.get('kind') != 'v2v4real_verified_volume_v1' or receipt.get('payload_verified') is not True:
        raise ValueError('verified volume receipt required')
    payload = volume / 'payload'
    if inventory_native_root(payload) != receipt['native_inventory']:
        raise ValueError('native inventory differs')
    records = receipt['files']
    if (not isinstance(records, list) or len(records) != len({r['path'] for r in records})
            or any(re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*/[0-9]+/[0-9]+\.(yaml|pcd)', r['path']) is None
                   for r in records)):
        raise ValueError('distinct native relative volume paths required')
    if {r['path'] for r in records} != {str(p.relative_to(payload)) for p in payload.rglob('*') if p.is_file()}:
        raise ValueError('volume file coverage differs')
    classes, keys, associations = Counter(), Counter(), Counter()
    matrix_errors = []
    count = 0
    for row in records:
        path = ordinary(payload / row['path'])
        if payload not in path.parents or path.stat().st_size != row['bytes']:
            raise ValueError('invalid volume path/size')
        if path.suffix == '.yaml':
            data = _read(path, MAX_YAML_BYTES)
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('YAML digest differs')
            metadata = load_raw_yaml(data)
            projection = pose_projection(metadata)
            annotations = read_annotations(metadata)
            classes.update(a.raw_class for a in annotations)
            keys.update(metadata.keys())
            associations.update('minus_one' if a.associated_id == '-1' else 'other' for a in annotations)
            rotation = np.asarray(projection['source_to_world'])[:3, :3]
            matrix_errors.append((float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
                                  float(abs(np.linalg.det(rotation) - 1.))))
            count += 1
        elif digest_file(path)[1] != row['sha256']:
            raise ValueError('point-cloud digest differs')
    expected = receipt['native_inventory']['source_frame_count']
    if count != expected:
        raise ValueError('YAML frame coverage differs')
    sources = [Path(__file__), Path('tools/event_track_v2x/prepare_v2v4real_inputs.py'),
        Path('transvision/models/event_track_v2x/v2v4real_inputs.py'),
        Path('transvision/models/event_track_v2x/v2v4real_numpy_yaml.py')]
    root = Path(__file__).resolve().parents[2]
    return dict(kind='v2v4real_native_volume_audit_v1', volume_receipt_sha256=receipt_sha256,
        archive_sha256=receipt['archive_sha256'], split=receipt['split'], parsed_source_frames=count,
        paired_frames=receipt['native_inventory']['paired_frame_count'],
        raw_annotation_counts=dict(classes), raw_association_value_counts=dict(associations),
        annotation_count_unit='per-source-frame-object; not unique physical objects',
        raw_top_level_key_counts=dict(keys), maximum_rotation_orthogonality_residual=max(x[0] for x in matrix_errors),
        maximum_determinant_residual=max(x[1] for x in matrix_errors),
        source_sha256={str(p.relative_to(root) if p.is_absolute() else p): digest_file(p if p.is_absolute() else root / p)[1] for p in sources},
        all_volume_file_hashes_verified=True, raw_yaml_read=True, point_cloud_format_validated=False,
        class_mapping_applied=False, identity_mapping_verified=False, full_official_split_verified=False,
        tracking_evaluation_performed=False, paper_eligible=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--volume', required=True, type=Path)
    p.add_argument('--receipt-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(argv)
    output = a.output.absolute()
    if a.volume.resolve() == output.resolve() or a.volume.resolve() in output.resolve().parents:
        raise ValueError('audit output must be outside immutable volume')
    ordinary(output.parent, directory=True)
    result = audit_volume(a.volume, a.receipt_sha256)
    with output.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
