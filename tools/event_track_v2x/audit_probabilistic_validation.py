#!/usr/bin/env python3
"""Prediction-only audit of complete fixed LBP/joint-MAP SPD val adapters.

This verifies inputs, output chains, clocks and conditional scan numerics.
It does not compute tracking metrics or turn LBP residuals into risk bounds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools.event_track_v2x import run_probabilistic_tracking_v2 as producer
from tools.event_track_v2x.compare_train_probabilistic_controls import audit_scan
from tools.event_track_v2x.run_tracking_v2 import schedule_rows
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, contained_file, sha_file
from transvision.models.event_track_v2x.persistent_probabilistic_tracking import PersistentProbabilisticConfig


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def audit_streams(rows, predictions, tracking, timings, config):
    """Independent ordered scan check; rows must come from a sealed schedule."""
    counts, heads = Counter(), {}
    factors = hashlib.sha256()
    seconds, frame_seconds = [], []
    max_error = max_residual = 0.
    for row, prediction, wrapper, timing in zip_longest(rows, predictions, tracking, timings):
        if any(v is None for v in (row, prediction, wrapper, timing)):
            raise ValueError('schedule/prediction/audit/timing lengths differ')
        scene, frame, reference = row['sequence_id'], row['vehicle_frame'], row['box_reference_timestamp_us']
        key = scene, frame, reference, reference + 100_000
        actual = lambda v: (v['sequence_id'], v['frame_id'], v['box_reference_timestamp_us'], v['decision_timestamp_us'])
        audit = wrapper['tracking']
        if actual(prediction) != key or actual(timing) != key or (audit['sequence_id'], audit['event_id']) != key[:2]:
            raise ValueError('event identity or causal clock differs from schedule')
        prior = heads.get(scene, dict(frames=0, prediction_sha256='0'*64, audit_sha256='0'*64))
        if (prediction['commit_sha256'] != digest({k:v for k,v in prediction.items() if k != 'commit_sha256'})
                or prediction['previous_commit_sha256'] != prior['prediction_sha256']
                or audit['prediction_sha256'] != prediction['commit_sha256']
                or audit['previous_audit_sha256'] != prior['audit_sha256']):
            raise ValueError('prediction or audit commit chain differs')
        ids = [p['track_id'] for p in prediction['predictions']]
        if len(set(ids)) != len(ids) or any(p['class_label'] != 'car' for p in prediction['predictions']):
            raise ValueError('duplicate output identity or non-car prediction')
        if (audit['association_algorithm'] != 'lbp' or audit['update_rule'] != config['update_rule']
                or audit['anchor_decoder'] != 'joint-map' or audit['recovery_enabled'] is not False
                or audit['same_state_time_protocol_as_recoverable'] is not False
                or audit['full_history_posterior_bound'] is not None
                or audit['unmatched_mass_scales_detection_score'] is not False):
            raise ValueError('executed probabilistic algorithm or uncertainty claim differs')
        work = audit['state_work_breakdown']
        if any(type(v) is not int or v < 0 for v in work.values()) or sum(work.values()) != audit['state_updates']:
            raise ValueError('state work breakdown differs')
        counts.update(frames=1, output_boxes=len(ids), state_updates=audit['state_updates'])
        for scan in audit['conditional_scans']:
            numerical = audit_scan(scan, config)
            counts.update(scans=1, iterations=numerical['iterations'], message_updates=numerical['message_updates'])
            max_error = max(max_error, numerical['max_row_or_column_error'])
            max_residual = max(max_residual, numerical['log_message_residual'])
        a, b = timing['step_seconds'], timing['frame_seconds']
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b)) or not 0 <= a <= b:
            raise ValueError('invalid frame latency')
        seconds.append(a); frame_seconds.append(b)
        factors.update(canonical([scene, frame, audit['factor_rows_sha256']])+b'\n')
        heads[scene] = dict(frames=prior['frames']+1, prediction_sha256=prediction['commit_sha256'],
                           audit_sha256=digest(audit))
    if not counts['frames']:
        raise ValueError('empty replay')
    return dict(counts, sequence_heads=heads, factor_stream_sha256=factors.hexdigest(),
        max_row_or_column_error=max_error, max_log_message_residual=max_residual,
        latency_seconds_p50_p95_p99_max=np.quantile(seconds,[.5,.95,.99,1.]).tolist(),
        frame_latency_seconds_p50_p95_p99_max=np.quantile(frame_seconds,[.5,.95,.99,1.]).tolist())


def audit(directory, receipt_sha256, output):
    directory, output = Path(directory), Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new non-symlink audit output required')
    sealed = {}
    def read(name, expected=None):
        path = contained_file(directory, name)
        h = sha_file(path)
        if expected is not None and h != expected:
            raise ValueError('artifact identity differs: '+name)
        sealed[str(path.absolute())] = h
        return json.loads(path.read_bytes())
    final = read('full-validation-receipt.json', receipt_sha256)
    receipt = read('receipt.json')
    plan = read('plan.json', receipt['plan_sha256'])
    if (any(final.get(k) != v for k,v in receipt.items())
            or final.get('full_official_validation_schedule_completed') is not True
            or receipt['status'] != 'complete' or receipt['cache_split'] != 'val'
            or receipt['completed_frames'] != 3316 or receipt['scheduled_frames'] != 3316
            or len(receipt['sequence_heads']) != 21 or receipt['allocation_teacher'] is not False
            or receipt['learned_identity_enabled'] is not True or receipt['learned_allocation_enabled'] is not False
            or receipt['probabilistic_single_history_enabled'] is not True):
        raise ValueError('complete learned non-teacher official validation receipt required')
    config = plan['configuration']
    if (plan['kind'] != 'probabilistic_single_history_spd_val_adaptation_plan_v1'
            or plan['class_scope'] != ['car'] or plan['checkpoint_seed'] not in (1337,2027,3407)
            or plan['geometry_baseline'] is not False or plan.get('scheduled_frames') != 3316
            or config != asdict(PersistentProbabilisticConfig(update_rule=config['update_rule']))):
        raise ValueError('fixed three-seed learned LBP/joint-MAP car plan required')
    if any(final.get(k) != config[c] for k,c in (
            ('association_algorithm','association_algorithm'),('update_rule','update_rule'),('anchor_decoder','anchor_decoder'))):
        raise ValueError('final algorithm declaration differs')
    if final['checkpoint_sha256'] != plan['checkpoint_sha256']:
        raise ValueError('final checkpoint declaration differs')
    producer.require_unchanged(plan['source_sha256'], plan['input_file_sha256'])
    for p in (Path(__file__), ROOT/'tools/event_track_v2x/compare_train_probabilistic_controls.py'):
        sealed[str(p)] = sha_file(p)
    paths = [Path(p) for p,h in plan['input_file_sha256'].items() if h == plan['schedule_sha256']]
    if len(paths) != 1:
        raise ValueError('one bound prediction schedule required')
    rows = schedule_rows(paths[0], plan['schedule_sha256'])
    for name,key in (('predictions.jsonl','predictions_sha256'),('tracking.jsonl','tracking_sha256'),
                     ('frame-timings.jsonl','frame_timings_sha256')):
        path = contained_file(directory,name)
        if sha_file(path) != receipt[key]:
            raise ValueError('replay stream identity differs: '+name)
        sealed[str(path.absolute())] = receipt[key]
    with (directory/'predictions.jsonl').open('rb') as predictions, (directory/'tracking.jsonl').open('rb') as tracking, \
            (directory/'frame-timings.jsonl').open('rb') as timings:
        measured = audit_streams(rows, map(json.loads,predictions), map(json.loads,tracking), map(json.loads,timings), config)
    if set(measured['sequence_heads']) != set(receipt['sequence_heads']):
        raise ValueError('sequence inventory differs')
    database_bytes = 0
    for scene, head in receipt['sequence_heads'].items():
        if any(head[k] != measured['sequence_heads'][scene][k] for k in ('frames','prediction_sha256')):
            raise ValueError('sequence final commit or coverage differs')
        path = contained_file(directory,head['database'])
        if sha_file(path) != head['database_sha256']:
            raise ValueError('sequence database identity differs')
        sealed[str(path.absolute())] = head['database_sha256']; database_bytes += path.stat().st_size
    for key in ('latency_seconds_p50_p95_p99_max','frame_latency_seconds_p50_p95_p99_max'):
        if measured[key] != receipt[key]:
            raise ValueError('latency summary differs from frame stream')
    if database_bytes != receipt['database_bytes']:
        raise ValueError('database byte summary differs')
    producer.require_unchanged(plan['source_sha256'], plan['input_file_sha256'])
    if any(sha_file(Path(p)) != h for p,h in sealed.items()):
        raise ValueError('audit evidence changed during inspection')
    result = dict(kind='probabilistic_full_spd_val_audit_v1', status='complete', **measured,
        seed=plan['checkpoint_seed'], update_rule=config['update_rule'], configuration=config,
        inference_receipt_sha256=receipt_sha256, source_directory=str(directory.absolute()),
        replay_receipt_sha256=sealed[str((directory/'receipt.json').absolute())],
        database_bytes=database_bytes, process_peak_rss_bytes=receipt['process_peak_rss_bytes'],
        input_sha256=sealed, runtime=plan['runtime'], full_validation_coverage_verified=True,
        ground_truth_read=False, tracking_metrics_computed=False, posterior_risk_bound_verified=False,
        fair_resources_verified=False, paper_eligible=False)
    output.mkdir()
    producer._new(output/'audit.json', result)
    print(json.dumps({k:result[k] for k in ('status','seed','update_rule','frames','scans','factor_stream_sha256')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run,args.receipt_sha256,args.output)
