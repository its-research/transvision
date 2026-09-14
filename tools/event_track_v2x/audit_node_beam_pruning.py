#!/usr/bin/env python3
"""Exact, capped event-suffix enumeration from the SAME previous retained beam.

Post-hoc model-only diagnostic, not a new tracker, full posterior, MHT result,
GT oracle or uncharged inference. Refuses merges and fails at the work cap.
Historical factor rows are replayed from the sealed audit, not final DB scores.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x import evaluate_train_inference_diagnostic as evidence_tools
from tools.event_track_v2x.audit_beam_identity_divergence import choices_for, members_for, root_partition, tracking_events


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def logsumexp(values):
    values = tuple(values)
    if not values:
        return -math.inf
    largest = max(values)
    return largest + math.log(math.fsum(math.exp(v - largest) for v in values))


def transitions(choices, rows, slots):
    depth = len(choices); roots = root_partition(choices)
    blocked = {roots[i] for i in range(depth) if slots[i] == slots[depth]}
    groups = {}
    for parent, weight in rows[depth]:
        if type(parent) is not int or not -1 <= parent < depth or not math.isfinite(weight):
            raise ValueError('legal finite historical factor row required')
        root = -1 if parent < 0 else roots[parent]
        if root in blocked:
            continue
        groups.setdefault(root, []).append((parent, weight))
    for root, values in sorted(groups.items(), key=lambda item: min(p for p, _ in item[1])):
        yield min(p for p, _ in values), logsumexp(w for _, w in values)


def append(candidate, choice, increment):
    choices, score, sha = candidate
    roots = root_partition(choices); depth = len(choices)
    root = depth if choice == -1 else roots[choice]
    return choices + (choice,), math.fsum([score, increment]), digest([sha, depth, choice, root])


def score_prefix(choices, rows, slots, root_sha):
    result = (), 0., root_sha
    for choice in choices:
        options = dict(transitions(result[0], rows, slots))
        if choice not in options:
            raise ValueError('historical prefix is not a legal canonical class under event factors')
        result = append(result, choice, options[choice])
    return result


def enumerate_suffix(initial, rows, slots, width, target, *, cap):
    if (type(width) is not int or width < 1 or type(cap) is not int or cap < 1
            or not initial or len(rows) != len(slots) or len(target) != len(rows)
            or len({len(c[0]) for c in initial}) != 1 or len({c[0] for c in initial}) != len(initial)):
        raise ValueError('distinct common-depth predecessor beam and explicit work cap required')
    start = len(initial[0][0]); exact = list(initial); beam = list(initial); count = beam_count = 0; stages = []
    if not start < len(rows):
        raise ValueError('nonempty event suffix required')
    order = lambda c: (-c[1], c[2])
    for depth in range(start, len(rows)):
        expanded = []
        for candidate in exact:
            for choice, increment in transitions(candidate[0], rows, slots):
                count += 1
                if count > cap:
                    raise ValueError('exact suffix work cap exhausted; no partial oracle result')
                expanded.append(append(candidate, choice, increment))
        exact = expanded
        generated = sorted([append(c, p, w) for c in beam for p, w in transitions(c[0], rows, slots)], key=order)
        beam_count += len(generated); beam = generated[:width]
        prefix = tuple(target[:depth + 1]); found = [i + 1 for i, c in enumerate(generated) if c[0] == prefix]
        stages.append(dict(depth=depth + 1, generated_classes=len(generated), retained_classes=len(beam),
            target_prefix_generated=bool(found), target_prefix_rank=None if not found else found[0],
            target_prefix_retained=any(c[0] == prefix for c in beam)))
    exact.sort(key=order)
    positions = [i + 1 for i, c in enumerate(exact) if c[0] == tuple(target)]
    if len(positions) != 1:
        raise ValueError('actual recovery output is not one unique suffix of the previous retained beam')
    record = lambda c: dict(sha256=c[2], log_weight=c[1])
    return dict(initial_classes=len(initial), previous_depth=start, final_depth=len(rows),
        exact_generated_prefixes=count, exact_complete_classes=len(exact), node_beam_generated_prefixes=beam_count,
        target_exact_rank=positions[0], target_log_weight=exact[positions[0] - 1][1],
        exact_top_k=[record(c) for c in exact[:width]], node_beam_top_k=[record(c) for c in beam],
        node_pruning_stages=stages, exhaustive_only_within_previous_retained_classes=True,
        complete_history_posterior=False, work_cap=cap)


def historical_event(tracking, frame_id, component):
    rows = []; previous = None
    for event in tracking_events(tracking):
        if event['observation_count'] != len(rows) + len(event['appended_rows']):
            raise ValueError('audit factor append coverage differs')
        for i, row in event['rescored_rows']:
            if type(i) is not int or not 0 <= i < len(rows):
                raise ValueError('rescore references a nonhistorical observation')
            rows[i] = row
        rows.extend(event['appended_rows'])
        found = [s for s in event['components'] if s['component'] == component]
        if event['event_id'] == frame_id:
            if len(found) != 1 or previous is None or found[0]['merge_retained_beams_only']:
                raise ValueError('continuing unmerged component with a prior retained beam required')
            return event, found[0], previous, rows
        if found:
            previous = found[0]
    raise ValueError('selected event absent from complete inference')


def audit(replay, receipt_sha, frame_id, component, output, *, cap=100000):
    replay = Path(replay).absolute()
    outer_path = replay / 'development-inference-receipt.json'; outer = evidence_tools.read(outer_path, receipt_sha)
    receipt_path = replay / 'receipt.json'; receipt = evidence_tools.read(receipt_path, outer['replay_receipt_sha256'])
    plan_path = replay / 'plan.json'; plan = evidence_tools.read(plan_path, receipt['plan_sha256'])
    if (outer.get('status') != 'complete' or receipt.get('status') != 'complete'
            or outer.get('backend') != 'beam_recovery' or plan['class_scope'] != ['car']
            or plan['cohort_mode'] != 'real-train-development' or outer.get('training_performed') is not False
            or receipt['completed_frames'] != receipt['scheduled_frames'] or len(receipt['sequence_heads']) != 1):
        raise ValueError('complete car train recovery inference required')
    head = next(iter(receipt['sequence_heads'].values())); database = replay / head['database']
    tracking = replay / 'tracking.jsonl'
    inputs = {outer_path: receipt_sha, receipt_path: outer['replay_receipt_sha256'], plan_path: receipt['plan_sha256'],
        database: head['database_sha256'], tracking: receipt['tracking_sha256'],
        Path(__file__): evidence_tools.sha(__file__),
        Path(sys.modules[choices_for.__module__].__file__): evidence_tools.sha(sys.modules[choices_for.__module__].__file__),
        Path(evidence_tools.__file__): evidence_tools.sha(evidence_tools.__file__)}
    if any(evidence_tools.sha(p) != h for p, h in inputs.items()):
        raise ValueError('sealed inference inputs changed')
    event, current, previous, rows = historical_event(tracking, frame_id, component)
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        raw = db.execute('SELECT i,source,frame,raw,sha FROM observations WHERE i<? ORDER BY i', (len(rows),)).fetchall()
        if [r[0] for r in raw] != list(range(len(rows))):
            raise ValueError('complete historical raw observation prefix required')
        factor_hash = hashlib.sha256()
        for i, _, _, payload, raw_sha in raw:
            if hashlib.sha256(payload).hexdigest() != raw_sha:
                raise ValueError('raw observation hash differs')
            factor_hash.update(canonical([i, raw_sha, rows[i]]) + b'\n')
        if factor_hash.hexdigest() != event['factor_rows_sha256']:
            raise ValueError('event-time factor reconstruction differs; final scores cannot substitute')
        members = members_for(db, component, current['nodes']); inverse = {g: i for i, g in enumerate(members)}
        if any(g >= len(rows) for g in members):
            raise ValueError('future member in event component')
        local = [[(-1 if p < 0 else inverse[p], w) for p, w in rows[g]] for g in members]
        slots = [(raw[g][1], raw[g][2]) for g in members]
        root_sha = db.execute(f'SELECT sha FROM pc{component}_prefixes WHERE h=0').fetchone()[0]
        initial = [score_prefix(choices_for(db, component, r['handle'], previous['nodes']), local, slots, root_sha)
                   for r in previous['active']]
        target = choices_for(db, component, current['output_handle'], current['nodes'])
        result = enumerate_suffix(initial, local, slots, plan['configuration']['state']['active_limit'], target, cap=cap)
        expected = [dict(sha256=r['sha256'], log_weight=r['log_weight']) for r in current['backbone_active']]
        if result['node_beam_top_k'] != expected:
            raise ValueError('independent node pruning does not reproduce recorded backbone')
        chosen = next(r for r in current['active'] if r['handle'] == current['output_handle'])
        if (result['target_log_weight'] != chosen['log_weight']
                or any(a[k] != b[k] for a, b in zip(result['node_pruning_stages'], current['pruning'])
                       for k in ('depth', 'generated_classes', 'retained_classes'))
                or len(result['node_pruning_stages']) != len(current['pruning'])):
            raise ValueError('independent exact candidate score or pruning counts differ')
    if any(evidence_tools.sha(p) != h for p, h in inputs.items()):
        raise ValueError('inference or auditor sources changed during suffix enumeration')
    result.update(kind='train_node_beam_event_pruning_audit_v1', status='complete', frame_id=frame_id,
        component=component, sequence_id=event['sequence_id'], factor_rows_sha256=event['factor_rows_sha256'],
        historical_factors_reconstructed=True, backbone_reproduced_exactly=True,
        GT_read=False, predictions_modified=False, parameter_training=False, online_latency_measured=False,
        reproduced_classical_MHT=False, validation=False, paper_eligible=False,
        input_sha256={str(p): h for p, h in inputs.items()})
    destination = evidence_tools.new_directory(output)
    with (destination / 'node-pruning.json').open('xb') as stream:
        stream.write(canonical(result) + b'\n')
    print(json.dumps({k: result[k] for k in ('status', 'frame_id', 'component', 'exact_complete_classes',
        'target_exact_rank', 'backbone_reproduced_exactly', 'paper_eligible')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True); parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--frame-id', required=True); parser.add_argument('--component', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--cap', type=int, default=100000)
    args = parser.parse_args()
    audit(args.replay, args.receipt_sha256, args.frame_id, args.component, args.output, cap=args.cap)
