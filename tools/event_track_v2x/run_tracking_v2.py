#!/usr/bin/env python3
"""Full official-val clean-link tracking with all three frozen association seeds.

Only cache + public schedule + train-fit calibration + checkpoint are accepted.
Ground truth is intentionally not an argument and belongs to a separate process.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transvision.models.event_track_v2x.detection_cache_v2 import DetectionCacheV2, canonical, load_manifest, sha_file
from transvision.models.event_track_v2x.predicted_association_v2 import CALIBRATION_SHA, CHECKPOINTS, load_frozen_model
from transvision.models.event_track_v2x.tracking_v2 import LearnedPairTrackerV2, TrackingConfigV2

SPLIT_SHA = '4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3'


def write_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))


def schedule_rows(path, expected_sha):
    if sha_file(path) != expected_sha:
        raise ValueError('schedule changed')
    m = json.loads(Path(path).read_bytes())
    if (set(m) != {'kind', 'split_sha256', 'contains_ground_truth', 'contains_system_error_offset', 'frames'}
            or m['kind'] != 'spd_official_validation_prediction_schedule_v1' or m['split_sha256'] != SPLIT_SHA
            or m['contains_ground_truth'] is not False or m['contains_system_error_offset'] is not False):
        raise ValueError('not a prediction-only official validation schedule')
    rows = m['frames']
    if len(rows) != 3316 or len({r['sequence_id'] for r in rows}) != 21:
        raise ValueError('full validation coverage required')
    previous = {}
    for r in rows:
        if set(r) != {'sequence_id', 'vehicle_frame', 'infrastructure_frame', 'box_reference_timestamp_us'}:
            raise ValueError('unknown schedule fields; potential GT leakage')
        scene, timestamp = r['sequence_id'], r['box_reference_timestamp_us']
        if type(timestamp) is not int or timestamp <= previous.get(scene, -1):
            raise ValueError('nonmonotonic reference frame schedule')
        previous[scene] = timestamp
    for side in ['vehicle_frame', 'infrastructure_frame']:
        if len({r[side] for r in rows}) != len(rows):
            raise ValueError('schedule contains duplicate source frame')
    if [r['sequence_id'] for r in rows] != sorted(r['sequence_id'] for r in rows):
        raise ValueError('schedule must be sequence-contiguous')
    return rows


def model_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    if sha_file(args.calibration) != CALIBRATION_SHA:
        raise ValueError('calibration must remain the frozen train-fit artifact')
    calibration = json.loads(args.calibration.read_bytes())
    schedule = schedule_rows(args.schedule, args.schedule_sha256)
    manifest, entries = load_manifest(args.cache, args.cache_sha256)
    if (manifest['split'] != 'val' or manifest['frame_count'] != 7189 or len(manifest['sequences']) != 21
            or manifest['calibration_sha256'] != CALIBRATION_SHA
            or set(manifest['sequences']) != {r['sequence_id'] for r in schedule}):
        raise ValueError('not the complete fixed validation cache')
    lookup, origins, sides = {}, {}, Counter()
    for entry in entries:
        # load_manifest already rehashed all payloads; these tiny metadata reads
        # build an index but do not feed future detection values to the model.
        meta = json.loads((args.cache / entry['metadata']['path']).read_bytes())
        key = (meta['side'], meta['frame_id'])
        if key in lookup:
            raise ValueError('ambiguous frame lookup')
        lookup[key] = entry; sides[meta['side']] += 1
        scene = meta['sequence_id']; timestamp = meta['box_reference_timestamp_us']
        origins[scene] = min(origins.get(scene, timestamp), timestamp)
    if sides != {'vehicle-side': 3748, 'infrastructure-side': 3441}:
        raise ValueError('side coverage differs')
    for row in schedule:
        for side, key in [('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')]:
            entry = lookup[(side, row[key])]
            meta = json.loads((args.cache / entry['metadata']['path']).read_bytes())
            if meta['sequence_id'] != row['sequence_id'] or (side == 'vehicle-side' and meta['box_reference_timestamp_us'] != row['box_reference_timestamp_us']):
                raise ValueError('cache/schedule identity mismatch')
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    models = {seed: load_frozen_model(args.checkpoints / f'seed-{seed}-final.pt', seed, args.device) for seed in CHECKPOINTS}
    before = {seed: model_digest(model) for seed, model in models.items()}
    sources = [Path(__file__)] + [ROOT/'transvision/models/event_track_v2x'/name for name in
        ['detection_cache_v2.py', 'predicted_association_v2.py', 'prediction_features.py', 'tracking_v2.py',
         'association.py', 'fusion.py', 'arrays.py']]
    source_hashes = {p.relative_to(ROOT).as_posix(): sha_file(p) for p in sources}
    config = TrackingConfigV2()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = {'kind': 'eventtrack_v2_clean_full_validation_plan_v1', 'split_sha256': SPLIT_SHA,
        'cache_sha256': args.cache_sha256, 'schedule_sha256': args.schedule_sha256,
        'calibration_sha256': CALIBRATION_SHA, 'checkpoints': CHECKPOINTS, 'sources': source_hashes,
        'config': asdict(config), 'torch_version': str(torch.__version__), 'device': args.device,
        'cpu_threads': args.cpu_threads, 'seed_selection': False, 'val_parameter_fitting': False,
        'test_payloads_read': False, 'gt_model_inputs': False, 'official_validation_frames': 3316,
        'condition': 'clean-link-100ms-sensor-availability',
        'wall_clock_inference_latency_in_deadline': False, 'network_C1_C9_executed': False,
        'learned_voi': False, 'out_of_fold_training': False, 'paper_eligible': False,
        'state_time': 'vehicle_lidar_reference', 'commit_time': 'reference_plus_100ms',
        'unpaired_cache_frames': 'sealed_and_verified_but_not_tracking_schedule',
        'association_approximation': 'Top3 joint energy plus all-unmatched; marginal moment CI; omitted mass unknown'}
    write_json(args.output/'frozen-plan.json', frozen)
    receipts, streams = {}, {}
    for seed in models:
        root = args.output / f'seed-{seed}'; root.mkdir()
        streams[seed] = ((root/'predictions.jsonl').open('xb'), (root/'association.jsonl').open('xb'))
        receipts[seed] = {'seed': seed, 'frames': 0, 'predictions': 0, 'source_rejected_late': [0, 0],
                          'selected_detections': [0, 0], 'sequence_commits': {}}
    started = time.monotonic(); trackers = {}; scene = None
    try:
        for number, row in enumerate(schedule, 1):
            if row['sequence_id'] != scene:
                scene = row['sequence_id']
                trackers = {seed: LearnedPairTrackerV2(model, calibration, scene, origins[scene], config) for seed, model in models.items()}
            vehicle = DetectionCacheV2.load(args.cache, lookup[('vehicle-side', row['vehicle_frame'])])
            infra = DetectionCacheV2.load(args.cache, lookup[('infrastructure-side', row['infrastructure_frame'])])
            for seed, tracker in trackers.items():
                result, audit = tracker.step(vehicle, infra)
                pred, assoc = streams[seed]
                pred.write(canonical(result)+b'\n'); assoc.write(canonical(audit)+b'\n')
                receipt = receipts[seed]
                receipt['frames'] += 1; receipt['predictions'] += len(result['predictions'])
                receipt['sequence_commits'][scene] = result['commit_sha256']
                for i in range(2):
                    receipt['source_rejected_late'][i] += int(not result['source_available'][i])
                    receipt['selected_detections'][i] += result['selected_detections'][i]
            if number % 100 == 0:
                print('EVENTTRACK_V2_PROGRESS '+json.dumps({'frames': number, 'total': len(schedule), 'seeds': list(models),
                      'elapsed_seconds': time.monotonic()-started}), flush=True)
    finally:
        for pair in streams.values():
            for stream in pair:
                stream.close()
    for seed, model in models.items():
        if model_digest(model) != before[seed]:
            raise ValueError('frozen model changed during validation')
        root = args.output/f'seed-{seed}'
        receipts[seed].update(predictions_sha256=sha_file(root/'predictions.jsonl'),
            association_sha256=sha_file(root/'association.jsonl'), weights_unchanged=True)
        write_json(root/'receipt.json', receipts[seed])
    summary = {'kind': 'eventtrack_v2_full_tracking_validation_predictions_v1',
        'plan_sha256': sha_file(args.output/'frozen-plan.json'), 'seeds': receipts,
        'elapsed_seconds': time.monotonic()-started, 'all_frames_completed': True,
        'evaluated': False, 'paper_eligible': False}
    write_json(args.output/'summary.json', summary)
    print('EVENTTRACK_V2_COMPLETE '+json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['cache', 'schedule', 'calibration', 'checkpoints', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--cache-sha256', required=True)
    p.add_argument('--schedule-sha256', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--cpu-threads', type=int, default=1)
    args = p.parse_args()
    if args.cpu_threads < 1:
        p.error('cpu threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
