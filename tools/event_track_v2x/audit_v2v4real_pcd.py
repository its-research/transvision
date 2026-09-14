#!/usr/bin/env python3
"""Check every projected point cloud and optionally compare Open3D XYZI exactly.

Reads only the GT-free projection, not native YAML or annotation IDs. The oracle
is the equation in the pinned official pcd_to_np, executed using Open3D 0.19.0.
It is not a detector run, original-paper environment reproduction or benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import platform
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transvision.models.event_track_v2x.v2v4real_inputs import load_prepared_frames, OFFICIAL_SOURCE_COMMIT
from transvision.models.event_track_v2x.v2v4real_pcd import read_native_pcd, RECIPE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(inputs, manifest_sha256, *, open3d_oracle=False):
    source_paths = ('tools/event_track_v2x/audit_v2v4real_pcd.py',
        'tools/event_track_v2x/prepare_v2v4real_inputs.py',
        'transvision/models/event_track_v2x/v2v4real_inputs.py',
        'transvision/models/event_track_v2x/v2v4real_pcd.py')
    sources = {p: sha(ROOT / p) for p in source_paths}
    if open3d_oracle:
        import open3d as o3d
        if o3d.__version__ != '0.19.0':
            raise ValueError('pinned Open3D 0.19.0 oracle required')
    manifest, frames = load_prepared_frames(inputs, expected_manifest_sha256=manifest_sha256)
    records = []
    start = time.monotonic()
    for frame in frames:
        path = inputs / frame['pcd_path']
        cloud = read_native_pcd(path, expected_sha256=frame['pcd_sha256'])
        points = cloud.xyzi
        if open3d_oracle:
            oracle = o3d.io.read_point_cloud(str(path), remove_nan_points=False, remove_infinite_points=False)
            reference = np.hstack((np.asarray(oracle.points), np.asarray(oracle.colors)[:, :1])).astype(np.float32)
            if reference.shape != points.shape or not np.array_equal(reference, points):
                error = float(np.max(np.abs(reference - points))) if reference.shape == points.shape else None
                raise ValueError(f'Open3D XYZI mismatch: {frame["pcd_path"]}, max_abs={error}')
            if sha(path) != frame['pcd_sha256']:
                raise ValueError('point file changed during oracle read')
        records.append(dict(path=frame['pcd_path'], pcd_sha256=cloud.source_sha256, points=len(points),
            xyzi_f32_le_sha256=hashlib.sha256(points.astype('<f4', copy=False).tobytes(order='C')).hexdigest(),
            xyzi_min=points.min(axis=0).tolist(), xyzi_max=points.max(axis=0).tolist()))
        if len(records) % 100 == 0:
            print(json.dumps(dict(kind='native_pcd_progress', frames=len(records), scheduled=len(frames))), flush=True)
    if sha(inputs / 'manifest.json') != manifest_sha256 or any(sha(ROOT / p) != h for p, h in sources.items()):
        raise ValueError('input manifest or decoder source changed')
    return dict(kind='v2v4real_native_pcd_audit_v1', recipe=RECIPE, input_manifest_sha256=manifest_sha256,
        source_sha256=sources, dataset_split=manifest['dataset_split'], source_frames=len(records),
        total_points=sum(r['points'] for r in records), records=records, elapsed_seconds=time.monotonic()-start,
        all_points_finite=True, no_points_dropped=True, point_order_preserved=True, source_yaml_or_GT_read=False,
        open3d_oracle_version='0.19.0' if open3d_oracle else None, all_frames_exact_oracle_parity=open3d_oracle,
        official_reader_source_commit=OFFICIAL_SOURCE_COMMIT, original_paper_runtime_reproduced=False,
        runtime=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
            packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()}),
        detector_executed=False, tracking_evaluation_performed=False, full_official_split_verified=False,
        paper_eligible=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--open3d-oracle', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    output = a.output.absolute()
    if output.exists() or output.is_symlink() or a.inputs.resolve() in output.resolve().parents:
        raise ValueError('fresh output outside immutable inputs required')
    result = audit(a.inputs, a.manifest_sha256, open3d_oracle=a.open3d_oracle)
    with output.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, sort_keys=True))


if __name__ == '__main__':
    main()
