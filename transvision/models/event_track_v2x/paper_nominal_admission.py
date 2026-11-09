"""Explicit full-cohort nominal binding without changing sealed cache bytes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .detection_cache_v2 import canonical
from .paper_nominal_clock import validate_formal_variant, nominal_run_binding


def read_pinned(record):
    if set(record) != {'path', 'sha256'}:
        raise ValueError('explicit path and SHA required')
    data = Path(record['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('nominal admission input changed')
    return data


def validate(cache, events, configuration, path, expected_sha256):
    admission = json.loads(read_pinned(dict(path=path, sha256=expected_sha256)))
    if set(admission) != {'formal_config', 'source_clock', 'mapping', 'projection'}:
        raise ValueError('complete nominal admission required')
    formal = validate_formal_variant(json.loads(read_pinned(admission['formal_config'])))
    source_clock = json.loads(read_pinned(admission['source_clock']))
    manifest = json.loads(cache.manifest_json)
    producer = manifest['producer']
    if (manifest['dataset'] != 'v2v4real'
            or producer.get('clock_sha256') != admission['source_clock']['sha256']
            or producer.get('projection_frames_sha256') != admission['projection']['sha256']
            or source_clock.get('protocol_id') != formal['protocol_id']
            or source_clock.get('period_us') != formal['period_us']
            or source_clock.get('measured') is not False):
        raise ValueError('cache clock or projection binding differs')
    original = {}
    shared = {}
    for line in read_pinned(admission['projection']).splitlines():
        row = json.loads(line)
        if row['cav_id'] not in {'0', '1'} or type(row['frame_ordinal']) is not int or row['frame_ordinal'] < 0:
            raise ValueError('invalid original source mapping')
        side = 'vehicle-side' if row['cav_id'] == '0' else 'infrastructure-side'
        key = row['sequence_id'], side, row['frame_key']
        pair = row['sequence_id'], row['frame_key']
        if key in original or pair in shared and shared[pair] != row['frame_ordinal']:
            raise ValueError('duplicate or inconsistent original mapping')
        original[key] = row['frame_ordinal'] * formal['period_us']
        shared[pair] = row['frame_ordinal']
    if set(original) != set(cache.index):
        raise ValueError('original mapping/cache coverage differs')
    for key, (_, meta_bytes) in cache.index.items():
        meta = json.loads(meta_bytes)
        if meta['box_reference_timestamp_us'] != original[key] or meta['source_image_timestamp_us'] != original[key]:
            raise ValueError('cache timestamp differs from original ordinal')
    clock = json.loads(read_pinned(admission['mapping']))
    mapping = {(r['sequence_id'], r['frame_id']): r['frame_index'] for r in clock['mapping']}
    if mapping != shared or len(mapping) != len(clock['mapping']):
        raise ValueError('shared mapping differs from original projection')
    references = {(seq, frame) for seq, side, frame in original if side == 'vehicle-side'}
    event_keys = [(e['sequence_id'], e['frame_id']) for e in events]
    if len(event_keys) != len(references) or set(event_keys) != references:
        raise ValueError('full reference event coverage required')
    wrapper = SimpleNamespace(manifest_json=canonical({'producer': {'clock': clock}}), index=cache.index)
    binding = nominal_run_binding(wrapper, events, configuration)
    if binding is None:
        raise ValueError('nominal binding must not be skipped')
    return dict(binding, admission_sha256=expected_sha256,
                formal_config_sha256=admission['formal_config']['sha256'],
                projection_sha256=admission['projection']['sha256'],
                cache_manifest_sha256=cache.manifest_sha256)
