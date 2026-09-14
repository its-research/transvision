#!/usr/bin/env python3
"""Frozen single-factor mechanism controls against sealed full V2 validation.

Only prediction inputs are accepted. No tuning, GT, test data, GPU or external
publication is performed. Original tracker/cache files remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from transvision.models.event_track_v2x.detection_cache_v2 import DetectionCacheV2, canonical, load_manifest, sha_file
from transvision.models.event_track_v2x.predicted_association_v2 import (
    CALIBRATION_SHA, CHECKPOINTS, association_hypotheses, load_frozen_model,
)
from transvision.models.event_track_v2x.prediction_features import choose_candidates
from transvision.models.event_track_v2x.tracking_mechanisms_v2 import MechanismDiagnosticTrackerV2, MODES
from transvision.models.event_track_v2x.tracking_v2 import TrackingConfigV2

_spec = importlib.util.spec_from_file_location('sealed_tracking_runner', ROOT/'tools/event_track_v2x/run_tracking_v2.py')
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
schedule_rows, model_digest = _runner.schedule_rows, _runner.model_digest

CACHE_SHA = '66c58025bd79ea674ff676f1df8d76e7812c8b1d2b62bc54e8de3ced0bc3aca8'
SCHEDULE_SHA = '2c8999ecbe2ab98bedf13ba2da4b22bd6167eca07b368a99db148abf36de982a'
SEALED_SUMMARY_SHA = '681345583204315167a9d3ad36d5d7594bca2bf6f210cda25fe2198cc4e76126'
SEALED_PLAN_SHA = 'dcd4997350645cfdb4629ed57cd7336664187d547c6d61ef875e0886317bc57a'
RUNS = (
    [{'run_id': f'M0-seed-{seed}', 'mode': 'M0', 'seed': seed, 'deterministic_control': False}
     for seed in CHECKPOINTS]
    + [{'run_id': 'M1-all-unmatched', 'mode': 'M1', 'seed': 1337, 'deterministic_control': True}]
    + [{'run_id': f'{mode}-seed-{seed}', 'mode': mode, 'seed': seed, 'deterministic_control': False}
       for mode in ['M2', 'M3'] for seed in CHECKPOINTS]
)



def write_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))


def empty_side_probe(models):
    """Verify the real frozen heads' finite empty-side contract, not a mock."""
    probes = []
    for n, m in [(3, 0), (0, 3), (0, 0)]:
        reference = None
        for seed, model in models.items():
            with torch.inference_mode():
                logits = model(torch.zeros((1, n, 203)), torch.zeros((1, m, 203)))
            h = association_hypotheses(*[x[0].numpy() for x in logits], np.zeros(n, int), np.zeros(m, int))
            expected = [{'pairs': [], 'unmatched_left': list(range(n)), 'unmatched_right': list(range(m)),
                         'weight': 1., 'energy': 0.}]
            if h != expected or any(not torch.isfinite(x).all() for x in logits):
                raise ValueError('single-source frozen-head contract differs')
            encoded = canonical(h)
            if reference is not None and reference != encoded:
                raise ValueError('single-source association depends on seed')
            reference = encoded
            probes.append({'seed': seed, 'left': n, 'right': m, 'finite': True, 'all_unmatched': True})
    return probes


def run(args):
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('experiment output is create-once; inspect an existing run before retry')
    output_parent = args.output.parent
    while not output_parent.exists():
        output_parent = output_parent.parent
    if shutil.disk_usage(output_parent).free < 8 * 1024**3:
        raise ValueError('at least 8 GiB free storage required for ten full prediction/diagnostic streams')
    old = args.validation_root
    cache = old/'detection-cache-v2'
    sealed = old/'tracking-run-v1'
    for path, expected in [(sealed/'summary.json', SEALED_SUMMARY_SHA),
                           (sealed/'frozen-plan.json', SEALED_PLAN_SHA), (args.calibration, CALIBRATION_SHA)]:
        if path.is_symlink() or sha_file(path) != expected:
            raise ValueError('sealed prediction/calibration identity changed')
    sealed_summary = json.loads((sealed/'summary.json').read_bytes())
    sealed_plan = json.loads((sealed/'frozen-plan.json').read_bytes())
    for name, expected in sealed_plan['sources'].items():
        if (ROOT/name).is_symlink() or sha_file(ROOT/name) != expected:
            raise ValueError('original sealed implementation changed: '+name)
    config = TrackingConfigV2()
    if asdict(config) != sealed_plan['config'] or str(torch.__version__) != sealed_plan['torch_version']:
        raise ValueError('backend config or torch runtime differs from sealed run')
    schedule = schedule_rows(old/'schedule.json', SCHEDULE_SHA)
    manifest, entries = load_manifest(cache, CACHE_SHA)
    if (manifest['split'] != 'val' or manifest['frame_count'] != 7189
            or manifest['calibration_sha256'] != CALIBRATION_SHA
            or set(manifest['sequences']) != {r['sequence_id'] for r in schedule}):
        raise ValueError('not the complete frozen official-validation cache')
    lookup, origins, counts = {}, {}, Counter()
    for entry in entries:
        meta = json.loads((cache/entry['metadata']['path']).read_bytes())
        key = (meta['side'], meta['frame_id'])
        if key in lookup:
            raise ValueError('duplicate side/frame')
        lookup[key] = entry; counts[meta['side']] += 1
        scene, timestamp = meta['sequence_id'], meta['box_reference_timestamp_us']
        origins[scene] = min(origins.get(scene, timestamp), timestamp)
    if counts != {'vehicle-side': 3748, 'infrastructure-side': 3441}:
        raise ValueError('source coverage differs')
    for row in schedule:
        for side, key in [('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')]:
            entry = lookup[(side, row[key])]
            meta = json.loads((cache/entry['metadata']['path']).read_bytes())
            if meta['sequence_id'] != row['sequence_id'] or (side == 'vehicle-side' and meta['box_reference_timestamp_us'] != row['box_reference_timestamp_us']):
                raise ValueError('cache and public schedule disagree')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint_files = {seed: args.checkpoints/f'seed-{seed}-final.pt' for seed in CHECKPOINTS}
    if args.checkpoints.is_symlink() or any(path.is_symlink() for path in checkpoint_files.values()):
        raise ValueError('checkpoint symlinks are forbidden')
    models = {seed: load_frozen_model(args.checkpoints/f'seed-{seed}-final.pt', seed, 'cpu') for seed in CHECKPOINTS}
    before = {seed: model_digest(model) for seed, model in models.items()}
    calibration = json.loads(args.calibration.read_bytes())
    probes = empty_side_probe(models)
    source_paths = sorted(set(sealed_plan['sources']) | {
        'tools/event_track_v2x/run_mechanism_diagnostics_v2.py',
        'transvision/models/event_track_v2x/tracking_mechanisms_v2.py',
        'transvision/models/event_track_v2x/__init__.py',
    })
    if any((ROOT/name).is_symlink() for name in source_paths):
        raise ValueError('source symlinks are forbidden')
    sources = {name: sha_file(ROOT/name) for name in source_paths}
    plan = {'kind': 'mechanism_diagnostics_v2_plan', 'runs': RUNS, 'cache_sha256': CACHE_SHA,
        'schedule_sha256': SCHEDULE_SHA, 'split_sha256': sealed_plan['split_sha256'],
        'calibration_sha256': CALIBRATION_SHA, 'checkpoints': CHECKPOINTS,
        'checkpoint_files': {seed: str(path.resolve()) for seed, path in checkpoint_files.items()},
        'calibration_file': str(args.calibration.resolve()),
        'current_source_hashes': sources, 'sealed_plan_sha256': SEALED_PLAN_SHA,
        'sealed_summary_sha256': SEALED_SUMMARY_SHA, 'config': asdict(config),
        'reporting_scope': 'car_only', 'source_availability': 'unchanged sensor availability at reference+100ms',
        'candidate_selection': 'unchanged all-class raw_score>=0.05 top64; car-only metrics',
        'mechanism_modes': MODES,
        'baseline_seed_interpretation': 'M1 forced all-unmatched does not call pair model; seed1337 loaded',
        'diagnostic_encoding': 'canonical JSONL; deterministic gzip filename empty mtime=0 level=6',
        'diagnostic_covariance': 'mean9, upper xyz covariance6, full canonical covariance SHA256 and trace',
        'diagnostic_role': 'prediction-only sidecar; not a future-input or GT channel',
        'empty_side_model_probes': probes, 'device': 'cpu', 'cpu_threads': 1, 'torch_version': str(torch.__version__),
        'official_validation_frames': 3316, 'sequences': 21, 'gt_model_inputs': False, 'test_payloads_read': False,
        'val_parameter_fitting': False, 'seed_selection': False, 'condition': sealed_plan['condition'],
        'exploratory_after_initial_validation': True, 'OOF': False, 'paper_eligible': False,
        'network_C1_C9_executed': False, 'wall_clock_inference_latency_in_deadline': False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'frozen-plan.json', plan)
    plan_sha = sha_file(args.output/'frozen-plan.json')
    receipts, streams = {}, {}
    for spec in RUNS:
        name = spec['run_id']; folder = args.output/name; folder.mkdir()
        raw_diagnostic = (folder/'diagnostics.jsonl.gz').open('xb')
        streams[name] = [(folder/f'{key}.jsonl').open('xb') for key in ['predictions', 'association']]
        streams[name] += [gzip.GzipFile(filename='', mode='wb', fileobj=raw_diagnostic,
                                      mtime=0, compresslevel=6), raw_diagnostic]
        receipts[name] = {**spec, 'frames': 0, 'predictions': 0, 'source_sensor_late': [0, 0],
            'selected_detections': [0, 0], 'sequence_commits': {}, 'diagnostic_sequence_commits': {},
            'events': {}, 'nodes': 0, 'temporal_matches': 0}
    started = time.monotonic(); scene = None; trackers = {}
    try:
        for number, row in enumerate(schedule, 1):
            if row['sequence_id'] != scene:
                scene = row['sequence_id']
                trackers = {r['run_id']: MechanismDiagnosticTrackerV2(models[r['seed']], calibration,
                    scene, origins[scene], config, mode=r['mode'], plan_sha256=plan_sha) for r in RUNS}
            frames = [DetectionCacheV2.load(cache, lookup[(side, row[key])]) for side, key in
                      [('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')]]
            deadline = row['box_reference_timestamp_us'] + config.deadline_us
            sensor = [f.available_at(deadline) for f in frames]
            digests = [f.digest() for f in frames]
            expected_counts = [len(choose_candidates(f.raw_scores, {'minimum_raw_score': .05, 'maximum_per_side': 64}))
                               if allowed else 0 for f, allowed in zip(frames, sensor)]
            for spec in RUNS:
                name = spec['run_id']
                result, association, diagnostic = trackers[name].step(*frames)
                if (result['source_available'] != sensor or result['source_cache_sha256'] != digests
                        or result['selected_detections'] != expected_counts):
                    raise ValueError('source or selection boundary differs')
                for stream, value in zip(streams[name][:3], [result, association, diagnostic]):
                    stream.write(canonical(value)+b'\n')
                receipt = receipts[name]
                receipt['frames'] += 1; receipt['predictions'] += len(result['predictions'])
                receipt['sequence_commits'][scene] = result['commit_sha256']
                receipt['diagnostic_sequence_commits'][scene] = diagnostic['diagnostic_commit_sha256']
                receipt['nodes'] += len(diagnostic['nodes'])
                receipt['temporal_matches'] += len(diagnostic['temporal_assignments'])
                for event in diagnostic['events']:
                    receipt['events'][event['event']] = receipt['events'].get(event['event'], 0) + 1
                for i in (0, 1):
                    receipt['source_sensor_late'][i] += int(not sensor[i])
                    receipt['selected_detections'][i] += expected_counts[i]
            if number % 200 == 0:
                if shutil.disk_usage(args.output).free < 2 * 1024**3:
                    raise OSError('storage guard: less than 2 GiB free; partial outputs retained')
                print('MECHANISM_DIAGNOSTICS_PROGRESS '+json.dumps({'frames_per_run': number, 'total': 3316,
                    'runs': len(RUNS), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    except BaseException as error:
        write_json(args.output/'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                   'message': str(error), 'partial_outputs_retained': True, 'plan_sha256': plan_sha})
        raise
    finally:
        for group in streams.values():
            for stream in group:
                stream.close()
    if any(model_digest(models[seed]) != before[seed] for seed in models):
        raise ValueError('frozen association weights changed')
    if any(sha_file(checkpoint_files[seed]) != expected for seed, expected in CHECKPOINTS.items()):
        raise ValueError('checkpoint file bytes changed during experiment')
    if sha_file(args.calibration) != CALIBRATION_SHA:
        raise ValueError('calibration file bytes changed during experiment')
    if any(sha_file(ROOT/name) != sha for name, sha in sources.items()):
        raise ValueError('source files changed during experiment')
    for spec in RUNS:
        name = spec['run_id']; receipt = receipts[name]; folder = args.output/name
        receipt.update({key+'_sha256': sha_file(folder/f'{key}.jsonl') for key in ['predictions', 'association']})
        receipt['diagnostics_gzip_sha256'] = sha_file(folder/'diagnostics.jsonl.gz')
        receipt['diagnostics_gzip_bytes'] = (folder/'diagnostics.jsonl.gz').stat().st_size
        receipt.update(sequences=len(receipt['sequence_commits']), weights_unchanged=True,
                       sealed_prediction_parity=None, sealed_association_parity=None)
        if receipt['frames'] != 3316 or receipt['sequences'] != 21:
            raise ValueError('full cohort was not completed')
        if spec['mode'] != 'M1':
            previous = sealed_summary['seeds'][str(spec['seed'])]
            if (receipt['association_sha256'] != previous['association_sha256']
                    or sha_file(sealed/f"seed-{spec['seed']}"/'association.jsonl') != previous['association_sha256']):
                raise ValueError('association output differs from sealed experiment: '+name)
            receipt['sealed_association_parity'] = True
            if spec['mode'] == 'M0':
                if (receipt['predictions_sha256'] != previous['predictions_sha256']
                        or receipt['sequence_commits'] != previous['sequence_commits']
                        or sha_file(sealed/f"seed-{spec['seed']}"/'predictions.jsonl') != previous['predictions_sha256']):
                    raise ValueError('M0 prediction output differs from sealed experiment: '+name)
                receipt['sealed_prediction_parity'] = True
        write_json(folder/'receipt.json', receipt)
    summary = {'kind': 'mechanism_diagnostics_v2_complete', 'status': 'completed', 'plan_sha256': plan_sha,
        'reporting_scope': 'car_only', 'runs': receipts, 'all_required_sealed_streams_match': True,
        'weights_unchanged': True, 'elapsed_seconds': time.monotonic()-started, 'paper_eligible': False}
    write_json(args.output/'summary.json', summary)
    print('MECHANISM_DIAGNOSTICS_COMPLETE '+json.dumps({'plan_sha256': plan_sha, 'runs': len(RUNS),
          'frames_per_run': 3316, 'all_required_sealed_streams_match': True,
          'elapsed_seconds': summary['elapsed_seconds']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['validation-root', 'checkpoints', 'calibration', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
