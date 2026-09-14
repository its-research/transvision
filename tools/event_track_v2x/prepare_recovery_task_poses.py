#!/usr/bin/env python3
"""Export ego positions from raw navigation calibration, never a GT/infos PKL.

One pose per sealed vehicle cache frame. The composed raw LiDAR transform must
agree with the existing V2 cache. This does NOT assign independent arrival times.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, contained_file, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.recovery_task_scope import POSE_KIND, VerifiedEgoPoseTable


def transform(value):
    if type(value) is not dict or set(value) != {'rotation','translation'}:
        raise ValueError('exact raw calibration transform required')
    r, t = np.asarray(value['rotation'],dtype=float), np.asarray(value['translation'],dtype=float)
    if (r.shape != (3,3) or t.shape != (3,1) or not np.isfinite(r).all() or not np.isfinite(t).all()
            or not np.allclose(r.T@r,np.eye(3),atol=1e-6,rtol=0)
            or not np.isclose(np.linalg.det(r),1.,atol=1e-6,rtol=0)):
        raise ValueError('finite proper rotation and column translation required')
    return r,t[:,0]


def prepare(cache, raw_vehicle_root, output):
    poses, files = [], {}
    for key, (entry_bytes, meta_bytes) in sorted(cache.index.items()):
        sequence, side, frame = key
        if side != 'vehicle-side':continue
        if len(frame)!=6 or not frame.isascii() or not frame.isdigit():
            raise ValueError('canonical SPD vehicle frame required')
        entry, meta = json.loads(entry_bytes), json.loads(meta_bytes)
        source = {}
        for name in ('novatel_to_world','lidar_to_novatel'):
            path = contained_file(raw_vehicle_root, f'calib/{name}/{frame}.json')
            raw = path.read_bytes(); files[path] = sha_file(path)
            # Ensure the digest belongs to exactly the calibration parsed here.
            if hashlib.sha256(raw).hexdigest()!=files[path]:raise ValueError('calibration changed')
            source[name] = json.loads(raw)
        ew, et = transform(source['novatel_to_world'])
        le, lt = transform(source['lidar_to_novatel']['transform'])
        composed = lt@ew.T+et
        cached = np.asarray(meta['lidar_to_world_translation'])
        # The sealed detector inputs round world translations to float32.
        # Check that exact rounding operation, not a hand-tuned spatial margin.
        translation_matches = (np.array_equal(composed.astype(np.float32).astype(float),cached)
                               or np.allclose(composed,cached,rtol=0,atol=1e-10))
        if (not np.allclose((ew@le).T,meta['lidar_to_world_row_rotation'],rtol=0,atol=1e-6)
                or not translation_matches):
            raise ValueError(f'raw navigation composition disagrees with frozen V2 LiDAR pose: {sequence}/{frame}')
        poses.append(dict(sequence_id=sequence,frame_id=frame,
            state_us=meta['box_reference_timestamp_us'],
            information_us=max(meta['box_reference_timestamp_us'],meta['source_image_timestamp_us']),
            cache_frame_sha256=entry['frame_sha256'],ego_translation_world=et.tolist(),
            **{name+'_sha256':files[Path(raw_vehicle_root)/f'calib/{name}/{frame}.json'] for name in source}))
    if any(sha_file(p)!=h for p,h in files.items()):raise ValueError('raw calibration changed during export')
    output = Path(output)
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('symlink output forbidden')
    data = dict(kind=POSE_KIND,cache_sha256=cache.manifest_sha256,
        split=json.loads(cache.manifest_json)['split'],poses=poses,gt_model_inputs=False)
    with output.open('xb') as stream:stream.write(canonical(data))
    checksum = sha_file(output)
    VerifiedEgoPoseTable(output,checksum,cache)
    result = dict(kind='vehicle_ego_pose_preparation_v1',pose_table_sha256=checksum,
        vehicle_frames=len(poses),raw_calibration_files=len(files),gt_model_inputs=False,
        independent_pose_arrival_created=False,cache_transform_composition_checked=True)
    result['translation_check']='exact_float32_roundtrip_or_float64_atol_1e-10'
    print(json.dumps(result,sort_keys=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','raw-vehicle-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cache-sha256',required=True)
    a=p.parse_args()
    prepare(VerifiedForestCache(a.cache,a.cache_sha256),a.raw_vehicle_root,a.output)
