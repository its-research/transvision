"""Capture genuine missing numeric traces against accepted original raw rows.

Runtime setup must provide the unchanged original detector, point reader and
masks. This driver never exports replacement predictions, fits GT, dispatches a
task, or claims independent numeric acceptance. Full BEV traces are lossless;
an explicit uncompressed trace budget prevents silently truncating a cohort.
"""
import json
from pathlib import Path
import time

import numpy as np

from .capture_v2v4real_native_heads import capture_predict, CORE_SHA
from .run_v2v4real_official_test_raw_features import (
    CHECKPOINTS, SOURCE_ARCHIVE_SHA, IdentityCalibration, asset, shard_records)
from .verify_v2v4real_native_numeric_traces import (
    digest, identity, member, rows)


def capture_rows(records, inputs_root, detector, core, lidar_range, read_pcd,
                 mask_ego, mask_range, *, seed, source_archive, accepted_raw_index,
                 accepted_raw_sha, accepted_raw_root, output, max_trace_bytes,
                 progress=None, trace_layout='full',rank=0,world_size=1):
    if seed not in CHECKPOINTS or not isinstance(max_trace_bytes, int) or max_trace_bytes <= 0:
        raise ValueError('registered seed and explicit positive trace budget required')
    if trace_layout not in ('full','compact-cells-v1'):
        raise ValueError('explicit qualified trace layout required')
    asset(source_archive, SOURCE_ARCHIVE_SHA)
    asset(core.__file__, CORE_SHA)
    if type(detector) is not core.FrozenPointPillar or not str(detector.device).startswith('cuda'):
        raise ValueError('original frozen CUDA detector required')
    if detector.checkpoint_sha256 != CHECKPOINTS[seed]:
        raise ValueError('original registered seed checkpoint required')
    asset(detector.checkpoint, CHECKPOINTS[seed])
    asset(accepted_raw_index, accepted_raw_sha)
    raw = rows(accepted_raw_index)
    source = {}
    for r in records:
        key = (r['sequence_id'], r['frame_ordinal'],
               'vehicle-side' if r['is_ego'] else 'infrastructure-side', r['frame_key'])
        if key in source:
            raise ValueError('duplicate original source frame')
        source[key] = r
    if source.keys() != raw.keys():
        raise ValueError('exact accepted raw frame coverage required')
    full_rows=len(source)
    for key, r in source.items():
        if r['pcd_sha256'] != raw[key]['pcd_sha256']:
            raise ValueError('original PCD identity differs from accepted raw row')
        member(accepted_raw_root, raw[key]['payload'], raw[key]['payload_sha256'])
    if world_size!=1 or rank!=0:
        assigned=shard_records(records,rank,world_size)
        keys={(r['sequence_id'],r['frame_ordinal'],
               'vehicle-side' if r['is_ego'] else 'infrastructure-side',r['frame_key']) for r in assigned}
        source={key:r for key,r in source.items() if key in keys}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'payloads').mkdir()
    started, trace_bytes, index = time.monotonic(), 0, []
    for completed, key in enumerate(sorted(source), 1):
        original, accepted = source[key], raw[key]
        pcd_path = member(inputs_root, original['pcd_path'], original['pcd_sha256'])
        # The original reader checks the PCD SHA itself; neither point filtering
        # nor original inference/postprocessing is replaced by the capture.
        cloud = read_pcd(pcd_path, expected_sha256=original['pcd_sha256'])
        points = mask_range(mask_ego(cloud.xyzi), lidar_range)
        result, trace = capture_predict(detector, core, points, lidar_range=lidar_range,
            covariance_diagonal=np.ones(9), calibration=IdentityCalibration())
        path = member(accepted_raw_root, accepted['payload'], accepted['payload_sha256'])
        with np.load(path, allow_pickle=False) as previous:
            comparisons = dict(states_source_legacy7=result['states'][:, :7],
                raw_scores=result['raw_scores'], class_indices=result['class_indices'],
                appearance=result['appearance'], appearance_valid=result['appearance_valid'])
            for name, actual in comparisons.items():
                if np.asarray(actual).shape != previous[name].shape:
                    raise ValueError('captured raw output shape differs')
                if name in ('class_indices', 'appearance_valid'):
                    np.testing.assert_array_equal(actual, previous[name])
                else:
                    np.testing.assert_allclose(actual, previous[name], atol=1e-4, rtol=1e-4)
        if trace_layout=='compact-cells-v1':
            from .v2v4real_compact_bev_trace import compact_trace
            trace=compact_trace(trace,lidar_range)
        size = sum(a.nbytes for a in trace.values())
        if completed == 1:
            bound=size
            if trace_layout=='compact-cells-v1':
                # Every one of at most 64 selected boxes uses at most four
                # cells; retain all channels as float32 and cell coordinates.
                bound=sum(trace[k].nbytes for k in ('psm','regression','anchors','observed_anchor_scores'))
                bound+=trace['bev_shape'].nbytes+256*(int(trace['bev_shape'][1])*4+2*8)
            capacity=dict(trace_layout=trace_layout,rows=len(source),
                first_row_uncompressed_bytes=size,per_row_upper_bound_bytes=bound,
                projected_cohort_uncompressed_upper_bound_bytes=bound*len(source),
                budget_bytes=max_trace_bytes,within_budget=bound*len(source)<=max_trace_bytes,
                scope='array bytes only; NPZ headers and runtime assets require additional storage')
            with (out/'capacity-preflight.json').open('x') as f:
                json.dump(capacity,f,sort_keys=True,allow_nan=False)
            if not capacity['within_budget']:
                raise ValueError('full cohort trace estimate exceeds explicit budget; no partial acceptance')
        trace_bytes += size
        if trace_bytes > max_trace_bytes:
            raise ValueError('actual trace bytes exceed budget; partial files preserved')
        payload = out / 'payloads' / f'frame-{completed-1:06d}.npz'
        np.savez_compressed(payload, **trace)
        t = dict(zip(('sequence_id', 'frame_ordinal', 'side', 'frame_id'), key))
        t.update(pcd_sha256=original['pcd_sha256'], payload=payload.relative_to(out).as_posix(),
                 payload_sha256=digest(payload), raw_rows_sha256=accepted_raw_sha,
                 checkpoint_sha256=CHECKPOINTS[seed], source_archive_sha256=SOURCE_ARCHIVE_SHA,
                 inference_core_sha256=CORE_SHA, TF32_enabled=False)
        t['trace_layout']=trace_layout
        t['shard']=dict(rank=rank,world_size=world_size,full_source_rows=full_rows,
            rule='sorted_original_frame_pair_ordinal_mod_world_size')
        index.append(t)
        if progress:
            progress(dict(stage='native_numeric_trace_capture', completed_rows=completed,
                total_rows=len(source), uncompressed_trace_bytes=trace_bytes,
                eta_seconds=(time.monotonic()-started)/completed*(len(source)-completed),
                eta_scope='this trace capture only, excluding independent numeric acceptance'))
    for path, expected in ((source_archive, SOURCE_ARCHIVE_SHA), (core.__file__, CORE_SHA),
                           (detector.checkpoint, CHECKPOINTS[seed]), (accepted_raw_index, accepted_raw_sha)):
        asset(path, expected)
    with (out / 'traces.jsonl').open('x') as f:
        for t in index:
            f.write(json.dumps(t, sort_keys=True, allow_nan=False) + '\n')
    return dict(rows=len(index), uncompressed_trace_bytes=trace_bytes,
        trace_index_sha256=digest(out/'traces.jsonl'), accepted_raw_index_sha256=accepted_raw_sha,
        trace_layout=trace_layout,
        shard=dict(rank=rank,world_size=world_size,full_source_rows=full_rows),
        producer_raw_comparison_passed=True, independent_numeric_acceptance=False,
        network_forward_verified=False, nine_state_cache_admitted=False,
        tracking_accepted=False, paper_metric_complete=False)
