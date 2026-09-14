#!/usr/bin/env python3
"""Strict three-seed birth-score-only supplement to sealed V2 validation.

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
from itertools import zip_longest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from transvision.models.event_track_v2x.detection_cache_v2 import DetectionCacheV2, canonical, load_manifest, sha_file
from transvision.models.event_track_v2x.predicted_association_v2 import (
    CALIBRATION_SHA, CHECKPOINTS, association_hypotheses, load_frozen_model,
)
from transvision.models.event_track_v2x.prediction_features import choose_candidates
from transvision.models.event_track_v2x.tracking_birth_score_v2 import BirthScoreDiagnosticTrackerV2
from transvision.models.event_track_v2x.tracking_v2 import TrackingConfigV2

_spec = importlib.util.spec_from_file_location('sealed_tracking_runner', ROOT/'tools/event_track_v2x/run_tracking_v2.py')
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
schedule_rows, model_digest = _runner.schedule_rows, _runner.model_digest

CACHE_SHA = '66c58025bd79ea674ff676f1df8d76e7812c8b1d2b62bc54e8de3ced0bc3aca8'
SCHEDULE_SHA = '2c8999ecbe2ab98bedf13ba2da4b22bd6167eca07b368a99db148abf36de982a'
SEALED_SUMMARY_SHA = '681345583204315167a9d3ad36d5d7594bca2bf6f210cda25fe2198cc4e76126'
SEALED_PLAN_SHA = 'dcd4997350645cfdb4629ed57cd7336664187d547c6d61ef875e0886317bc57a'
RUNS = [{'run_id': f'M4-seed-{seed}', 'mode': 'M4', 'seed': seed, 'deterministic_control': False}
        for seed in CHECKPOINTS]
MODES = {'M4': 'road residual birth threshold and initial score use original score; temporal node score unchanged'}
FROZEN_MECHANISM_TRACKER_SHA = 'd9063684f54aa3f6bd083a13a05a4c18bdbf23f49855b387930cc89acb1adcf6'
INDEPENDENT_AUDITOR_SHA = '9147acd04d2ffd4e194fb2484f741ebee42e110eee58502df071522b5c2ee258'
FRAME_COUNT, SEQUENCE_COUNT = 3316, 21



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
        raise ValueError('at least 8 GiB free storage required for full prediction/diagnostic streams')
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
    reference = args.reference_mechanism_root
    reference_plan = json.loads((reference/'frozen-plan.json').read_bytes())
    reference_summary = json.loads((reference/'summary.json').read_bytes())
    reference_files = {name: sha_file(reference/name) for name in ['frozen-plan.json', 'summary.json']}
    if (reference.is_symlink() or any((reference/name).is_symlink() for name in reference_files)
            or reference_plan['kind'] != 'mechanism_diagnostics_v2_plan'
            or reference_summary['kind'] != 'mechanism_diagnostics_v2_complete'
            or reference_summary['status'] != 'completed'
            or reference_summary['plan_sha256'] != reference_files['frozen-plan.json']
            or reference_summary['all_required_sealed_streams_match'] is not True
            or reference_summary['weights_unchanged'] is not True
            or reference_plan['cache_sha256'] != CACHE_SHA
            or reference_plan['schedule_sha256'] != SCHEDULE_SHA
            or reference_plan['calibration_sha256'] != CALIBRATION_SHA
            or reference_plan['checkpoints'] != {str(k): v for k, v in CHECKPOINTS.items()}
            or reference_plan['current_source_hashes']['transvision/models/event_track_v2x/tracking_mechanisms_v2.py'] != FROZEN_MECHANISM_TRACKER_SHA
            or sha_file(ROOT/'transvision/models/event_track_v2x/tracking_mechanisms_v2.py') != FROZEN_MECHANISM_TRACKER_SHA):
        raise ValueError('reference mechanism experiment differs from sealed same-input control')
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
        'tools/event_track_v2x/run_birth_score_diagnostic_v2.py',
        'transvision/models/event_track_v2x/tracking_birth_score_v2.py',
        'transvision/models/event_track_v2x/tracking_mechanisms_v2.py',
        'transvision/models/event_track_v2x/__init__.py',
    })
    if any((ROOT/name).is_symlink() for name in source_paths):
        raise ValueError('source symlinks are forbidden')
    sources = {name: sha_file(ROOT/name) for name in source_paths}
    plan = {'kind': 'birth_score_diagnostics_v2_plan', 'runs': RUNS, 'cache_sha256': CACHE_SHA,
        'schedule_sha256': SCHEDULE_SHA, 'split_sha256': sealed_plan['split_sha256'],
        'calibration_sha256': CALIBRATION_SHA, 'checkpoints': CHECKPOINTS,
        'checkpoint_files': {seed: str(path.resolve()) for seed, path in checkpoint_files.items()},
        'calibration_file': str(args.calibration.resolve()),
        'current_source_hashes': sources, 'sealed_plan_sha256': SEALED_PLAN_SHA,
        'sealed_summary_sha256': SEALED_SUMMARY_SHA, 'config': asdict(config),
        'reporting_scope': 'car_only', 'source_availability': 'unchanged sensor availability at reference+100ms',
        'candidate_selection': 'unchanged all-class raw_score>=0.05 top64; car-only metrics',
        'mechanism_modes': MODES,
        'reference_mechanism_root': str(reference.resolve()),
        'reference_mechanism_plan_sha256': reference_files['frozen-plan.json'],
        'reference_mechanism_summary_sha256': reference_files['summary.json'],
        'causal_intervention': 'birth-only score channel; future free-running assignments may differ following births',
        'baseline_seed_interpretation': 'three paired frozen association seeds; compare with same-seed M0 in reference mechanism experiment',
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
                trackers = {r['run_id']: BirthScoreDiagnosticTrackerV2(models[r['seed']], calibration,
                    scene, origins[scene], config, plan_sha256=plan_sha) for r in RUNS}
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
                print('BIRTH_SCORE_DIAGNOSTICS_PROGRESS '+json.dumps({'frames_per_run': number, 'total': 3316,
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
    if any(sha_file(reference/name) != expected for name, expected in reference_files.items()):
        raise ValueError('reference experiment metadata changed during supplement')
    for spec in RUNS:
        name = spec['run_id']; receipt = receipts[name]; folder = args.output/name
        receipt.update({key+'_sha256': sha_file(folder/f'{key}.jsonl') for key in ['predictions', 'association']})
        receipt['diagnostics_gzip_sha256'] = sha_file(folder/'diagnostics.jsonl.gz')
        receipt['diagnostics_gzip_bytes'] = (folder/'diagnostics.jsonl.gz').stat().st_size
        receipt.update(sequences=len(receipt['sequence_commits']), weights_unchanged=True,
                       sealed_prediction_parity=None, sealed_association_parity=None)
        if receipt['frames'] != 3316 or receipt['sequences'] != 21:
            raise ValueError('full cohort was not completed')
        previous = sealed_summary['seeds'][str(spec['seed'])]
        reference_receipt = reference_summary['runs'][f"M0-seed-{spec['seed']}"]
        reference_assoc = reference/f"M0-seed-{spec['seed']}"/'association.jsonl'
        if (receipt['association_sha256'] != previous['association_sha256']
                or sha_file(sealed/f"seed-{spec['seed']}"/'association.jsonl') != previous['association_sha256']
                or receipt['association_sha256'] != reference_receipt['association_sha256']
                or sha_file(reference_assoc) != reference_receipt['association_sha256']):
            raise ValueError('M4 association differs from sealed/reference M0: '+name)
        receipt['sealed_association_parity'] = True
        receipt['reference_M0_association_parity'] = True
        write_json(folder/'receipt.json', receipt)
    summary = {'kind': 'birth_score_diagnostics_v2_complete', 'status': 'completed', 'plan_sha256': plan_sha,
        'reporting_scope': 'car_only', 'runs': receipts, 'all_required_sealed_streams_match': True,
        'weights_unchanged': True, 'elapsed_seconds': time.monotonic()-started, 'paper_eligible': False}
    write_json(args.output/'summary.json', summary)
    print('BIRTH_SCORE_DIAGNOSTICS_COMPLETE '+json.dumps({'plan_sha256': plan_sha, 'runs': len(RUNS),
          'frames_per_run': 3316, 'all_required_sealed_streams_match': True,
          'elapsed_seconds': summary['elapsed_seconds']}), flush=True)


def load_independent_auditor():
    path = ROOT/'tools/event_track_v2x/audit_mechanism_replay_v2.py'
    if path.is_symlink() or sha_file(path) != INDEPENDENT_AUDITOR_SHA:
        raise ValueError('independent mechanism audit helper changed')
    spec = importlib.util.spec_from_file_location('independent_birth_replay_helpers', path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    return audit


def scan_birth_streams(root, spec, plan_sha, audit):
    name = spec['run_id']; folder = root/name
    receipt = audit.read_json(folder/'receipt.json')
    counts = {'frames': 0, 'predictions': 0, 'nodes': 0, 'temporal_matches': 0,
              'events': {}, 'source_sensor_late': [0, 0], 'selected_detections': [0, 0],
              'sequence_commits': {}, 'diagnostic_sequence_commits': {}}
    previous_scene, previous_time, previous_ids = None, None, set()
    seen_frames, source_ids, retired = set(), [set(), set()], set()
    input_digest = hashlib.sha256(); birth_interventions = 0
    with (folder/'predictions.jsonl').open('rb') as p, (folder/'association.jsonl').open('rb') as a, gzip.open(folder/'diagnostics.jsonl.gz', 'rb') as d:
        for pp, aa, dd in zip_longest(p, a, d):
            audit.require(None not in (pp, aa, dd), 'three M4 stream lengths differ')
            pred, assoc, diag = audit.decode(pp, line=True), audit.decode(aa, line=True), audit.decode(dd, line=True)
            sid, fid, timestamp = pred['sequence_id'], pred['frame_id'], pred['box_reference_timestamp_us']
            if sid != previous_scene:
                audit.require(sid not in counts['sequence_commits'] and (previous_scene is None or sid > previous_scene), 'noncontiguous M4 sequence')
                previous_scene, previous_time, previous_ids, retired = sid, None, set(), set()
                counts['sequence_commits'][sid] = counts['diagnostic_sequence_commits'][sid] = '0'*64
            audit.require(type(timestamp) is int and (previous_time is None or timestamp > previous_time), 'nonmonotonic M4 reference')
            previous_time = timestamp
            audit.require((sid, fid) not in seen_frames, 'duplicate M4 frame'); seen_frames.add((sid, fid))
            for item in (assoc, diag):
                audit.require(item['sequence_id'] == sid and item['frame_id'] == fid, 'M4 frame binding differs')
            audit.require(pred['previous_commit_sha256'] == counts['sequence_commits'][sid] and
                          pred['commit_sha256'] == audit.digest({k: v for k, v in pred.items() if k != 'commit_sha256'}), 'invalid M4 prediction commit chain')
            audit.require(diag['previous_diagnostic_commit_sha256'] == counts['diagnostic_sequence_commits'][sid] and
                          diag['diagnostic_commit_sha256'] == audit.digest({k: v for k, v in diag.items() if k != 'diagnostic_commit_sha256'}), 'invalid M4 diagnostic commit chain')
            counts['sequence_commits'][sid] = pred['commit_sha256']
            counts['diagnostic_sequence_commits'][sid] = diag['diagnostic_commit_sha256']
            audit.require(diag['kind'] == 'tracking_birth_score_v2_frame' and diag['mode'] == 'M4' and diag['plan_sha256'] == plan_sha, 'not the frozen M4 diagnostic schema')
            audit.require(diag['prediction_commit_sha256'] == pred['commit_sha256'] and diag['association_frame_sha256'] == hashlib.sha256(aa[:-1]).hexdigest(), 'M4 association/prediction binding invalid')
            audit.require(pred['decision_timestamp_us'] == timestamp+100000, 'M4 deadline changed')
            for key in ('source_cache_sha256', 'source_available', 'selected_detections', 'box_reference_timestamp_us', 'decision_timestamp_us'):
                audit.require(diag[key] == pred[key], 'M4 source or timing binding differs')
            available, selected, info = pred['source_available'], pred['selected_detections'], pred['source_information_timestamp_us']
            audit.require(len(available) == len(selected) == len(info) == len(diag['source_frame_ids']) == 2 and diag['source_frame_ids'][0] == fid, 'M4 source inventory differs')
            for i in (0, 1):
                audit.require(type(available[i]) is bool and type(info[i]) is int and available[i] == (info[i] <= timestamp+100000), 'future M4 source available')
                audit.require(type(selected[i]) is int and 0 <= selected[i] <= 64 and (available[i] or selected[i] == 0), 'invalid M4 selection')
                audit.require(diag['source_frame_ids'][i] not in source_ids[i], 'repeated M4 source frame')
                source_ids[i].add(diag['source_frame_ids'][i])
                counts['source_sensor_late'][i] += not available[i]
                counts['selected_detections'][i] += selected[i]
            # This helper independently expects unchanged mass-weighted generic
            # road-node scores for every non-M3 mode, including the new M4.
            audit.validate_hypotheses(assoc, diag, selected, 'M4')
            for node in diag['nodes']:
                wanted = node['original_score'] if node['kind'] == 'road_residual' else node['score']
                audit.close(node['birth_effective_score'], wanted)
            ids = [item['track_id'] for item in pred['predictions']]
            audit.require(ids == sorted(set(ids)) and all(tid.startswith(sid+':') for tid in ids), 'duplicate M4 identity')
            current_ids = set(ids)
            audit.require(not current_ids & retired, 'retired M4 identity reappeared')
            audit.require(diag['tracks_before'] == len(previous_ids) and diag['tracks_after'] == len(current_ids), 'M4 lifecycle count differs')
            born = {event['track_id'] for event in diag['events'] if event['event'] == 'birth'}
            audit.require(born == current_ids-previous_ids, 'M4 births differ from new IDs')
            output_by_id = {p['track_id']: p for p in pred['predictions']}
            assigned_ids, assigned_nodes = set(), set()
            for match in diag['temporal_assignments']:
                tid, index = match['track_id'], match['node_index']
                audit.require(tid in previous_ids and tid in current_ids and tid not in assigned_ids and index not in assigned_nodes, 'invalid M4 temporal assignment')
                assigned_ids.add(tid); assigned_nodes.add(index)
                audit.close(match['node_score'], diag['nodes'][index]['score'])
                audit.close(match['assigned_cost'], .5*match['mahalanobis_squared']-float(np.log(max(match['node_score'], 1e-12))))
                audit.close(output_by_id[tid]['score'], max(match['prior_score'], match['node_score']))
            for event in diag['events']:
                kind = event['event']; counts['events'][kind] = counts['events'].get(kind, 0)+1
                if kind in ('birth', 'birth_rejected_score'):
                    node = diag['nodes'][event['node_index']]
                    audit.require(event['node_index'] not in assigned_nodes, 'birth attempted on temporally matched node')
                    audit.close(event['node_score'], node['score'])
                    audit.close(event['birth_effective_score'], node['birth_effective_score'])
                    audit.close(event['score'], node['birth_effective_score'])
                    audit.require((node['birth_effective_score'] >= .3) == (kind == 'birth'), 'M4 birth threshold mismatch')
                    if kind == 'birth':
                        audit.close(output_by_id[event['track_id']]['score'], node['birth_effective_score'])
                        birth_interventions += node['birth_effective_score'] != node['score']
            retired |= previous_ids-current_ids; previous_ids = current_ids
            counts['frames'] += 1; counts['predictions'] += len(ids)
            counts['nodes'] += len(diag['nodes']); counts['temporal_matches'] += len(diag['temporal_assignments'])
            input_digest.update(audit.canonical([sid, fid, timestamp, diag['source_frame_ids'], pred['source_cache_sha256'], available, selected])+b'\n')
    counts['sequences'] = len(counts['sequence_commits'])
    audit.require(counts['frames'] == FRAME_COUNT and counts['sequences'] == SEQUENCE_COUNT, 'M4 full cohort incomplete')
    audit.require(all(receipt[k] == v for k, v in counts.items()), 'M4 recomputed receipt counters/tips differ')
    return {**counts, 'input_order_sha256': input_digest.hexdigest(), 'births_with_changed_initial_score': birth_interventions,
            'prediction_and_diagnostic_chains_verified': True, 'birth_threshold_and_initial_score_verified': True,
            'temporal_cost_and_existing_score_update_use_common_node_score': True}


def verify_replay(args):
    audit = load_independent_auditor()
    first, replay, output = args.first, args.replay, args.output
    audit.require(not output.exists() and not output.is_symlink(), 'M4 replay audit output is create-once')
    audit.require(first.resolve() != replay.resolve(), 'M4 first/replay must be distinct')
    audit.require(all(root.resolve() not in output.resolve().parents for root in (first, replay)), 'M4 audit output must be outside sealed roots')
    expected = {'frozen-plan.json', 'summary.json'} | {r['run_id']+'/'+name for r in RUNS for name in ['predictions.jsonl', 'association.jsonl', 'diagnostics.jsonl.gz', 'receipt.json']}
    def inventory(root):
        paths = list(root.rglob('*'))
        audit.require(not root.is_symlink() and not any(p.is_symlink() for p in paths), 'symlink in M4 experiment')
        audit.require({p.relative_to(root).as_posix() for p in paths if p.is_file()} == expected, 'M4 fourteen-file inventory mismatch')
        audit.require({p.relative_to(root).as_posix() for p in paths if p.is_dir()} == {r['run_id'] for r in RUNS}, 'unexpected M4 directory')
        return {name: audit.evidence(root/name) for name in sorted(expected)}
    inventories = {'first': inventory(first), 'replay': inventory(replay)}
    plans, summaries = {}, {}
    for key, root in [('first', first), ('replay', replay)]:
        plan, summary = audit.read_json(root/'frozen-plan.json'), audit.read_json(root/'summary.json')
        audit.require(plan['kind'] == 'birth_score_diagnostics_v2_plan' and plan['runs'] == RUNS, 'invalid M4 plan')
        audit.require(plan['official_validation_frames'] == FRAME_COUNT and plan['sequences'] == SEQUENCE_COUNT, 'M4 planned coverage differs')
        audit.require(plan['cache_sha256'] == CACHE_SHA and plan['schedule_sha256'] == SCHEDULE_SHA, 'M4 cache/schedule differs')
        audit.require(plan['config'] == asdict(TrackingConfigV2()), 'M4 default thresholds changed')
        for field in ['gt_model_inputs', 'test_payloads_read', 'val_parameter_fitting', 'seed_selection', 'paper_eligible']:
            audit.require(plan[field] is False, 'M4 frozen prediction boundary differs')
        audit.require(summary['kind'] == 'birth_score_diagnostics_v2_complete' and summary['status'] == 'completed'
                      and summary['plan_sha256'] == inventories[key]['frozen-plan.json']['sha256']
                      and summary['reporting_scope'] == 'car_only' and summary['paper_eligible'] is False
                      and summary['weights_unchanged'] is True and summary['all_required_sealed_streams_match'] is True
                      and set(summary['runs']) == {r['run_id'] for r in RUNS}, 'M4 incomplete summary')
        reference = Path(plan['reference_mechanism_root'])
        audit.require(audit.file_hash(reference/'frozen-plan.json') == plan['reference_mechanism_plan_sha256']
                      and audit.file_hash(reference/'summary.json') == plan['reference_mechanism_summary_sha256'], 'M4 reference mechanism changed')
        ref_summary = audit.read_json(reference/'summary.json')
        for spec in RUNS:
            name = spec['run_id']; receipt = audit.read_json(root/name/'receipt.json')
            audit.require(receipt == summary['runs'][name] and all(receipt[k] == v for k, v in spec.items())
                          and receipt['sealed_association_parity'] is True and receipt['reference_M0_association_parity'] is True
                          and receipt['sealed_prediction_parity'] is None and receipt['weights_unchanged'] is True, 'M4 receipt differs')
            for base, field in [('predictions.jsonl', 'predictions_sha256'), ('association.jsonl', 'association_sha256'), ('diagnostics.jsonl.gz', 'diagnostics_gzip_sha256')]:
                audit.require(receipt[field] == inventories[key][name+'/'+base]['sha256'], 'M4 stream hash differs')
            audit.require(receipt['diagnostics_gzip_bytes'] == inventories[key][name+'/diagnostics.jsonl.gz']['size'], 'M4 diagnostic size differs')
            ref_name = f"M0-seed-{spec['seed']}"
            audit.require(receipt['association_sha256'] == ref_summary['runs'][ref_name]['association_sha256']
                          == audit.file_hash(reference/ref_name/'association.jsonl'), 'M4 sealed same-seed association parity failed')
        plans[key], summaries[key] = plan, summary
    audit.require(plans['first'] == plans['replay'], 'M4 frozen plans differ')
    audit.require({k: v for k, v in summaries['first'].items() if k != 'elapsed_seconds'} ==
                  {k: v for k, v in summaries['replay'].items() if k != 'elapsed_seconds'}, 'M4 summaries differ beyond elapsed time')
    for path in sorted(expected-{'summary.json'}):
        audit.compare_bytes(first/path, replay/path)
    results = {}; cohort = None
    for spec in RUNS:
        left = scan_birth_streams(first, spec, inventories['first']['frozen-plan.json']['sha256'], audit)
        right = scan_birth_streams(replay, spec, inventories['replay']['frozen-plan.json']['sha256'], audit)
        audit.require(left == right, 'M4 recomputed first/replay evidence differs')
        if cohort is None:
            cohort = left['input_order_sha256']
        audit.require(left['input_order_sha256'] == cohort, 'M4 three seeds used different inputs')
        results[spec['run_id']] = {**left, 'all_three_streams_byte_identical': True}
        print('BIRTH_SCORE_REPLAY_AUDITED '+json.dumps({'run_id': spec['run_id'], 'frames': left['frames']}), flush=True)
    audit.require(inventory(first) == inventories['first'] and inventory(replay) == inventories['replay'], 'M4 evidence changed during audit')
    result = {'kind': 'birth_score_diagnostic_replay_audit_v1', 'status': 'verified', 'runs': results,
              'files_per_root': 14, 'file_manifests': inventories, 'all_nine_streams_byte_identical': True,
              'ground_truth_read': False, 'models_executed_during_audit': False, 'paper_eligible': False,
              'prediction_commits_verified': FRAME_COUNT*6, 'diagnostic_commits_verified': FRAME_COUNT*6,
              'boundary': 'birth-channel and output-chain audit; no independent detector or physical fusion recomputation',
              'controller_source': audit.evidence(__file__),
              'independent_helpers_source': audit.evidence(ROOT/'tools/event_track_v2x/audit_mechanism_replay_v2.py')}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    execute = sub.add_parser('run')
    for key in ['validation-root', 'reference-mechanism-root', 'checkpoints', 'calibration', 'output']:
        execute.add_argument('--'+key, type=Path, required=True)
    replay = sub.add_parser('verify-replay')
    for key in ['first', 'replay', 'output']:
        replay.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    else:
        result = verify_replay(args)
        print('BIRTH_SCORE_REPLAY_AUDIT_COMPLETE '+json.dumps({'status': result['status'], 'runs': len(result['runs'])}), flush=True)


if __name__ == '__main__':
    main()

