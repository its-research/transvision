"""Independent full-row post-head trace checker; never imports the detector or GT.

The caller supplies pinned trace and accepted raw-row indexes. This checks every
indexed row, including empty predictions. It does not certify network forward,
causal velocity, covariance, tracking, or benchmark metrics.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .capture_v2v4real_native_heads import CORE_SHA
from .v2v4real_native_numeric_reference import verify_raw_row


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pinned(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise ValueError('input SHA mismatch')
    return Path(path)


def member(root, relative, expected):
    relative = Path(relative)
    root = Path(root).resolve()
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe member path')
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('member escapes root')
    return pinned(path, expected)


def identity(row):
    return row['sequence_id'], row['frame_ordinal'], row['side'], row['frame_id']


def rows(path):
    values = [json.loads(line) for line in Path(path).read_text().splitlines()]
    indexed = {identity(row): row for row in values}
    if not values or len(indexed) != len(values):
        raise ValueError('empty or duplicate frame index')
    return indexed


def verify(*, trace_index, trace_sha, raw_index, raw_sha, trace_root,
           raw_root, checkpoint_sha, source_archive_sha, lidar_range):
    traces = rows(pinned(trace_index, trace_sha))
    raw = rows(pinned(raw_index, raw_sha))
    if traces.keys() != raw.keys():
        raise ValueError('full raw frame coverage required, including empty rows')
    candidates = 0
    started = time.monotonic()
    for completed, key in enumerate(sorted(raw), 1):
        t, r = traces[key], raw[key]
        required = dict(checkpoint_sha256=checkpoint_sha,
                        source_archive_sha256=source_archive_sha,
                        inference_core_sha256=CORE_SHA,
                        raw_rows_sha256=raw_sha, pcd_sha256=r['pcd_sha256'])
        if any(t.get(k) != v for k, v in required.items()) or t.get('TF32_enabled') is not False:
            raise ValueError('trace source/checkpoint/frame binding mismatch')
        trace_path = member(trace_root, t['payload'], t['payload_sha256'])
        raw_path = member(raw_root, r['payload'], r['payload_sha256'])
        with np.load(trace_path, allow_pickle=False) as a, np.load(raw_path, allow_pickle=False) as b:
            common={'psm','regression','anchors','observed_anchor_scores'}
            if set(a.files) not in (common|{'bev'},common|{'bev_shape','bev_cell_indices','bev_cell_values'}):
                raise ValueError('complete unmodified head/BEV trace required')
            result = verify_raw_row(**{k: a[k] for k in a.files}, lidar_range=lidar_range,
                states_legacy7=b['states_source_legacy7'], raw_scores=b['raw_scores'],
                appearance=b['appearance'], appearance_valid=b['appearance_valid'],
                class_indices=b['class_indices'])
        if result['candidates'] != r['candidates']:
            raise ValueError('raw candidate count mismatch')
        candidates += result['candidates']
        elapsed = time.monotonic() - started
        print(json.dumps(dict(stage='independent_post_head_numeric', completed_rows=completed,
                              total_rows=len(raw), candidates_completed=candidates,
                              eta_seconds=elapsed / completed * (len(raw) - completed),
                              eta_scope='this independent post-head check only')), flush=True)
    return dict(kind='v2v4real_independent_post_head_numeric_v1', rows=len(raw),
                candidates=candidates, trace_index_sha256=trace_sha, raw_index_sha256=raw_sha,
                checkpoint_sha256=checkpoint_sha, source_archive_sha256=source_archive_sha,
                inference_core_sha256=CORE_SHA, atol=1e-4, rtol=1e-4,
                independent_post_head_math_verified=True, network_forward_verified=False,
                nine_state_cache_admitted=False, tracking_accepted=False, paper_metric_complete=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('trace-index', 'trace-sha', 'raw-index', 'raw-sha', 'trace-root',
                 'raw-root', 'checkpoint-sha', 'source-archive-sha', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--lidar-range', type=float, nargs=6, required=True)
    args = vars(p.parse_args())
    output = Path(args.pop('output'))
    if output.exists():
        raise ValueError('create-once receipt output required')
    result = verify(**{k.replace('-', '_'): v for k, v in args.items()})
    with output.open('x') as f:
        json.dump(result, f, sort_keys=True)


if __name__ == '__main__':
    main()
