#!/usr/bin/env python3
"""Count car-only inference-window load in a sealed TRAIN cache, without GT.

This is input resource sizing, not tracking or accuracy evaluation. Information
timestamps are a zero-transport-delay availability proxy, not measured arrivals.
All source frames are counted, including ones outside a paired output schedule.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from transvision.models.event_track_v2x.detection_cache_v2 import (
    DetectionCacheV2, canonical, load_manifest, sha_file,
)


def window_load(rows, *, window_us, max_nodes):
    """Inclusive [t-window_us,t] load, grouping simultaneous arrivals first."""
    if (type(window_us) is not int or window_us < 1 or type(max_nodes) is not int or max_nodes < 1):
        raise ValueError('positive integer window and node limits required')
    groups = defaultdict(lambda: [0, 0])
    for timestamp, count in rows:
        if type(timestamp) is not int or timestamp < 0 or type(count) is not int or count < 0:
            raise ValueError('nonnegative integer timestamp and count required')
        groups[timestamp][0] += count
        groups[timestamp][1] += 1
    pending, total, loads, exceeded_frames = deque(), 0, [], 0
    for timestamp, (count, frames) in sorted(groups.items()):
        while pending and pending[0][0] < timestamp-window_us:
            total -= pending.popleft()[1]
        pending.append((timestamp, count))
        total += count
        loads.append(total)
        exceeded_frames += frames * int(total > max_nodes)
    return {'information_time_groups': len(loads), 'frames': len(rows),
            'peak_nodes': max(loads, default=0), 'frames_at_over_capacity_decisions': exceeded_frames,
            'load_quantiles_0_50_95_100': np.quantile(loads, [0, .5, .95, 1]).tolist() if loads else []}


def audit(cache, expected_sha256, *, window_us=2_000_000, max_nodes=128,
          minimum_raw_score=.05, maximum_detections=64):
    # Check the split BEFORE the full payload audit; do not inspect test/val
    # predictions while using this train-only resource-design tool.
    path = Path(cache) / 'manifest.json'
    if path.is_symlink() or sha_file(path) != expected_sha256:
        raise ValueError('cache identity differs')
    header = json.loads(path.read_bytes())
    if header.get('split') != 'train':
        raise ValueError('capacity design uses train only')
    if (not math.isfinite(minimum_raw_score) or not 0 <= minimum_raw_score <= 1
            or type(maximum_detections) is not int or maximum_detections < 1):
        raise ValueError('invalid raw-score candidate selection')
    window_load([], window_us=window_us, max_nodes=max_nodes)
    manifest, entries = load_manifest(cache, expected_sha256)
    rows, counts, per_source = defaultdict(list), [], defaultdict(lambda: {'frames': 0, 'selected_car': 0})
    raw_car, selected_car, valid_appearance = 0, 0, 0
    for number, entry in enumerate(entries, 1):
        frame = DetectionCacheV2.load(cache, entry)
        meta = frame.metadata
        raw_car += int((frame.class_indices == 0).sum())
        indices = np.flatnonzero((frame.class_indices == 0) & (frame.raw_scores >= minimum_raw_score))
        indices = sorted(indices.tolist(), key=lambda i: (-float(frame.raw_scores[i]), i))[:maximum_detections]
        count = len(indices)
        selected_car += count
        valid_appearance += int(frame.appearance_valid[np.asarray(indices, dtype=np.int64)].sum())
        rows[meta['sequence_id']].append((frame.information_timestamp_us, count))
        counts.append(count)
        per_source[meta['side']]['frames'] += 1
        per_source[meta['side']]['selected_car'] += count
        if number % 2000 == 0:
            print('FOREST_CAPACITY_PROGRESS '+json.dumps({'frames': number, 'total': len(entries)}), flush=True)
    scenes = {s: window_load(r, window_us=window_us, max_nodes=max_nodes) for s, r in sorted(rows.items())}
    return {'kind': 'recoverable_forest_train_cache_capacity_v1', 'cache_manifest_sha256': expected_sha256,
        'dataset_sha256': manifest['dataset_sha256'], 'calibration_sha256': manifest['calibration_sha256'],
        'class_scope': ['car'], 'split': 'train', 'frames': len(entries), 'sequences': len(scenes),
        'raw_car_detections': raw_car, 'selected_car_detections': selected_car,
        'selected_car_with_valid_appearance': valid_appearance, 'source_counts': dict(per_source),
        'selected_per_frame_quantiles_0_50_95_100': np.quantile(counts, [0, .5, .95, 1]).tolist(),
        'window_us': window_us, 'max_nodes': max_nodes, 'minimum_raw_score': minimum_raw_score,
        'maximum_detections_per_source_frame': maximum_detections,
        'peak_information_time_window_nodes': max(v['peak_nodes'] for v in scenes.values()),
        'frames_at_over_capacity_decisions': sum(v['frames_at_over_capacity_decisions'] for v in scenes.values()),
        'sequence_window_loads': scenes, 'all_payloads_rehashed': True,
        'arrival_policy': 'information_time_zero_transport_delay_proxy',
        'paired_output_schedule_applied': False, 'ambiguity_components_applied': False,
        'identity_tracking_executed': False, 'model_fitted': False, 'gt_payloads_read': False,
        'validation_or_test_payloads_read': False, 'paper_performance_evidence': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--cache-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--window-us', type=int, default=2_000_000)
    p.add_argument('--max-nodes', type=int, default=128)
    p.add_argument('--minimum-raw-score', type=float, default=.05)
    p.add_argument('--maximum-detections', type=int, default=64)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        p.error('output must be a new file')
    report = audit(args.cache, args.cache_sha256, window_us=args.window_us, max_nodes=args.max_nodes,
                   minimum_raw_score=args.minimum_raw_score, maximum_detections=args.maximum_detections)
    report['producer_sha256'] = sha_file(Path(__file__))
    with args.output.open('xb') as stream:
        stream.write(canonical(report))
    print('FOREST_CAPACITY_COMPLETE '+json.dumps({k: v for k, v in report.items()
        if k != 'sequence_window_loads'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
