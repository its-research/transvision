#!/usr/bin/env python3
"""Summarize every captured teacher event, not a selected success subset.

Partial snapshots consume only complete lines within the initial file size.
The prefix is rehashed after parsing to detect rewriting; appending is allowed.
This tool neither checks producer liveness nor certifies tracking quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.allocation_training import _directory
from transvision.models.event_track_v2x.allocation_policy import RECIPE, TARGET
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json


def _captured_lines(path, state):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        limit = path.stat().st_size
        consumed = 0
        while consumed < limit:
            line = stream.readline(limit-consumed)
            if not line:
                raise ValueError('teacher trace shrank during snapshot')
            if not line.endswith(b'\n'):
                break
            consumed += len(line)
            digest.update(line)
            yield json.loads(line)
    state.update(observed_file_bytes=limit, captured_prefix_bytes=consumed,
        incomplete_tail_bytes=limit-consumed, captured_prefix_sha256=digest.hexdigest())
    second = hashlib.sha256()
    with path.open('rb') as stream:
        remaining = consumed
        while remaining:
            block = stream.read(min(1024*1024, remaining))
            if not block:
                raise ValueError('captured teacher prefix disappeared')
            second.update(block)
            remaining -= len(block)
    if second.hexdigest() != digest.hexdigest():
        raise ValueError('captured teacher prefix changed during analysis')


def summarize(replay, output, *, receipt_sha256=None, partial_snapshot=False):
    if type(partial_snapshot) is not bool or partial_snapshot == (receipt_sha256 is not None):
        raise ValueError('choose a sealed receipt or an explicit partial snapshot')
    replay = Path(replay)
    plan_path, trace = contained_file(replay, 'plan.json'), contained_file(replay, 'tracking.jsonl')
    plan_hash = sha_file(plan_path)
    plan = json.loads(plan_path.read_bytes())
    if plan.get('cache_split') != 'train' or plan.get('allocation_teacher') is not True or plan.get('class_scope') != ['car']:
        raise ValueError('explicit train-only car teacher plan required')
    receipt = None
    if not partial_snapshot:
        path = contained_file(replay, 'receipt.json')
        if sha_file(path) != receipt_sha256:
            raise ValueError('teacher receipt identity differs')
        receipt = json.loads(path.read_bytes())
        if (receipt['status'] != 'complete' or receipt['plan_sha256'] != plan_hash
                or receipt['cache_split'] != 'train' or receipt['allocation_teacher'] is not True):
            raise ValueError('completed train teacher receipt required')
    snapshots, events, seen = {}, [], set()
    tolerance = 1e-12
    for record in _captured_lines(trace, snapshots):
        audit = record['tracking']
        if audit.get('training_trace_only') is not True or audit.get('offline_counterfactual_probes') is not True:
            raise ValueError('nonteacher event in trace')
        key = audit['sequence_id'], audit['event_id']
        if key in seen:
            raise ValueError('duplicate teacher event')
        seen.add(key)
        values, operations = [], {}
        beam = audit.get('beam_recovery_allocation') is True
        if beam and (plan['configuration'].get('enable_recovery') is not True
                or audit.get('allocation_trace_field') != 'recovery_allocation_trace'):
            raise ValueError('beam teacher trace/configuration differs')
        trace_field = 'recovery_allocation_trace' if beam else 'allocation_trace'
        work_field = 'kind' if beam else 'work_kind'
        for step in audit[trace_field]:
            operations[step[work_field]] = operations.get(step[work_field], 0)+1
            group = step['allocation_training']
            if group['feature_recipe'] != RECIPE or group['target_recipe'] != TARGET:
                raise ValueError('unknown teacher target or feature recipe')
            for candidate in group['candidates']:
                charged = candidate['charged_steps']
                value = candidate['target']
                expected = candidate['features'][0]*(candidate['model_bound_before']-candidate['model_bound_after'])/max(1, charged)
                if (type(charged) is not int or charged < 0 or not math.isfinite(value)
                        or not math.isfinite(expected) or abs(value-expected) > tolerance):
                    raise ValueError('teacher target arithmetic differs')
                values.append(value)
        events.append(dict(sequence_id=key[0], event_id=key[1], observation_count=audit['observation_count'],
            max_component_nodes=max((c['nodes'] for c in audit['components']), default=0),
            model_regret_upper=audit['model_regret_upper'], global_fallback_used=audit['global_fallback_used'],
            frontier_completion_candidates=audit.get('frontier_completion_candidates', 0),
            frontier_capped_components=sum(len(c['frontier']) >= plan['configuration']['state']['max_frontier']
                                          for c in audit['components']),
            search_steps=audit['recovery_search_steps'] if beam else audit['search_steps'],
            search_scope='extra_beam_recovery' if beam else 'component_allocation',
            operations=operations, target_rows=len(values),
            positive_targets=sum(value > tolerance for value in values),
            negative_targets=sum(value < -tolerance for value in values),
            near_zero_targets=sum(abs(value) <= tolerance for value in values),
            minimum_target=min(values, default=None), maximum_target=max(values, default=None),
            teacher_probe_executions=audit['teacher_probe_executions'], teacher_probe_cache_hits=audit['teacher_probe_cache_hits']))
    if sha_file(plan_path) != plan_hash:
        raise ValueError('teacher plan changed during analysis')
    scheduled = plan['scheduled_frames']
    if len(events) > scheduled:
        raise ValueError('teacher trace exceeds its schedule')
    if receipt is not None and (snapshots['incomplete_tail_bytes'] or snapshots['captured_prefix_sha256'] != receipt['tracking_sha256']
            or len(events) != receipt['completed_frames'] or len(events) != receipt['scheduled_frames'] or len(events) != scheduled):
        raise ValueError('complete trace coverage or identity differs')
    counts = {name: sum(event[name] for event in events) for name in
              ('target_rows', 'positive_targets', 'negative_targets', 'near_zero_targets', 'teacher_probe_executions', 'teacher_probe_cache_hits')}
    result = dict(kind='allocation_teacher_trace_diagnostic_v1', status='complete',
        partial_snapshot=partial_snapshot, producer_liveness_checked=False,
        plan_sha256=plan_hash, receipt_sha256=receipt_sha256, snapshot=snapshots,
        captured_events=len(events), scheduled_events=scheduled,
        full_official_train_claimed_by_producer=plan.get('full_official_train_verified') is True,
        full_official_train_completion_verified=False, numerical_target_tolerance=tolerance,
        counts=counts, events_with_positive_targets=sum(event['positive_targets'] > 0 for event in events),
        events_with_fallback=sum(event['global_fallback_used'] for event in events),
        events_with_bound_at_least_099=sum(event['model_regret_upper'] >= .99 for event in events),
        max_component_nodes=max((event['max_component_nodes'] for event in events), default=0),
        events=events, cached_target_repetitions_are_not_independent_samples=True,
        analyzer_sha256=sha_file(Path(__file__)), source_scorer=plan.get('identity_checkpoint_sha256'),
        real_tracking_evaluation=False, paper_eligible=False)
    output = _directory(output)
    _new_json(output/'summary.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'events'}, sort_keys=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--receipt-sha256')
    mode.add_argument('--partial-snapshot', action='store_true')
    args = parser.parse_args()
    summarize(args.replay, args.output, receipt_sha256=args.receipt_sha256, partial_snapshot=args.partial_snapshot)
