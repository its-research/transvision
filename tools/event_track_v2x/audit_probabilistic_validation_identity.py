#!/usr/bin/env python3
"""Prediction-only identity/state separation for three complete SPD val runs.

Read-only SQLite checks reconstruct every historical identity prefix and output
membership. This is not GT identity correctness, calibration or a recovery gain.
It neither imports a live tracker nor changes its database or past predictions.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import copy
import hashlib
from itertools import groupby, tee, zip_longest
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x import evaluate_probabilistic_validation as evaluation
from tools.event_track_v2x.audit_probabilistic_identity_invariance import fingerprint

native = evaluation.native


def digest(value):
    return hashlib.sha256(native.canonical(value)).hexdigest()


def non_state(box):
    return {k: v for k, v in box.items() if k not in ('mean', 'covariance')}


def checked_records(db, sequence, anchors, observations, pairs, state_config):
    """Yield validated records; retain only accumulated raw membership per root."""
    n = len(anchors)
    if (len(observations) != n or [r[0] for r in observations] != list(range(n))
            or any(i != k or type(root) is not int or not 0 <= root <= k
                   or anchors[root][1] != root for k, (i, root) in enumerate(anchors))):
        raise ValueError('contiguous observation and immutable root map required')
    groups, occupied, cursor = {}, set(), 0
    rows = db.execute('SELECT ordinal,event_id,prediction,audit FROM events ORDER BY ordinal')
    for ordinal, (saved, pair) in enumerate(zip_longest(rows, pairs)):
        if saved is None or pair is None:
            raise ValueError('database and published event coverage differs')
        audit, prediction = pair
        if (saved[:2] != (ordinal, audit['event_id'])
                or json.loads(saved[2]) != prediction or json.loads(saved[3]) != audit
                or audit['sequence_id'] != sequence or prediction['sequence_id'] != sequence
                or prediction['frame_id'] != audit['event_id']):
            raise ValueError('database event differs from published record')
        count = audit['observation_count']
        if type(count) is not int or not cursor <= count <= n:
            raise ValueError('nondecreasing observed prefix required')
        start = cursor
        for scan in audit['conditional_scans']:
            indices, roots = scan['indices'], scan['conditioned_track_roots']
            if (not indices or indices != list(range(cursor, cursor + len(indices)))
                    or cursor + len(indices) > count or len(set(roots)) != len(roots)
                    or any(type(r) is not int or r not in groups for r in roots)
                    or len(scan['anchors']) != len(indices)
                    or scan['reference_us'] != prediction['box_reference_timestamp_us']
                    or scan['decision_us'] != prediction['decision_timestamp_us']):
                raise ValueError('causal scan identity, prefix or clock differs')
            assigned = set()
            for j, (i, anchor) in enumerate(zip(indices, scan['anchors'])):
                _, node, source, frame, state_us, arrival_us, score = observations[i]
                root = anchors[i][1]; parent = anchor['representative_parent']
                if (anchor['index'] != i or anchor['root'] != root or root in assigned
                        or scan['source_slot'] != [source, frame]
                        or arrival_us > prediction['decision_timestamp_us']
                        or (root, source, frame) in occupied):
                    raise ValueError('anchor, source slot, arrival or exclusivity differs')
                if root == i:
                    if parent != -1:
                        raise ValueError('birth requires unmatched representative parent')
                elif (type(parent) is not int or not 0 <= parent < cursor
                        or anchors[parent][1] != root or root not in roots
                        or observations[parent][2:4] == (source, frame)
                        or scan['allowed'][roots.index(root)][j] is not True):
                    raise ValueError('matched root or representative parent is illegal')
                candidates = [p for (p,) in db.execute('SELECT p FROM potentials WHERE i=? ORDER BY p', (i,))
                    if (p == -1 and root == i or 0 <= p < cursor and anchors[p][1] == root
                        and observations[p][2:4] != (source, frame))]
                if not candidates or parent != min(candidates):
                    raise ValueError('representative parent is not a canonical legal factor edge')
                groups.setdefault(root, []).append((node, state_us, score))
                occupied.add((root, source, frame)); assigned.add(root)
            cursor += len(indices)
        if cursor != count or audit['new_observations'] != count - start:
            raise ValueError('conditional scans omit or repeat new observations')
        expected = []
        reference = prediction['box_reference_timestamp_us']
        for root, members in sorted(groups.items()):
            first = min(t for _, t, _ in members); last = max(t for _, t, _ in members)
            if (max(s for _, _, s in members) < state_config['birth_score']
                    or max(0, reference - last) > state_config['max_age_us']):
                continue
            score = max(s * state_config['survival_per_second'] ** (max(0, reference - t) / 1e6)
                        for _, t, s in members)
            if score < state_config['prune_score']:
                continue
            expected.append(dict(track_id=sequence + ':' + digest(['forest-birth', observations[root][1]])[:24],
                class_label='car', score=score, observation_ids=[node for node, _, _ in members],
                birth_state_us=first, last_update_us=last))
        if [non_state(box) for box in prediction['predictions']] != expected:
            raise ValueError('published IDs, membership, score or lifecycle differs from raw identity history')
        yield audit, prediction
    if cursor != n:
        raise ValueError('uncovered final observations')


def inspect_sequence(path, sequence, head, pairs, config):
    """No writable tracker open: SQLite mode=ro plus query_only only."""
    with closing(sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise ValueError('database integrity check failed')
        meta = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
        if (meta['schema'] != 'persistent_single_history_probabilistic_tracker_v1'
                or meta['sequence_id'] != sequence or meta['config'] != config
                or meta['state']['events'] != head['frames']
                or meta['state']['prediction_sha256'] != head['prediction_sha256']):
            raise ValueError('database final identity or configuration differs')
        anchors = list(db.execute('SELECT i,root FROM identity_anchors ORDER BY i'))
        observations = list(db.execute('SELECT i,node_id,source,frame,state_us,arrival_us,score FROM observations ORDER BY i'))
        left, right = tee(checked_records(db, sequence, anchors, observations, pairs, config['state']))
        result = fingerprint(anchors, (a for a, _ in left), (p for _, p in right))
        if result['frames'] != head['frames'] or result['observations'] != meta['state']['n']:
            raise ValueError('database head coverage differs')
        last = json.loads(db.execute('SELECT audit FROM events ORDER BY ordinal DESC LIMIT 1').fetchone()[0])
        if digest(last) != meta['state']['audit_sha256']:
            raise ValueError('database final audit chain head differs')
        return result


def fingerprint_run(directory, receipt, config):
    directory = Path(directory).absolute(); sequences = {}
    with (directory / 'tracking.jsonl').open('rb') as audits, (directory / 'predictions.jsonl').open('rb') as predictions:
        def records():
            for a, p in zip_longest(audits, predictions):
                if a is None or p is None:
                    raise ValueError('published stream lengths differ')
                yield json.loads(a)['tracking'], json.loads(p)
        for scene, pairs in groupby(records(), key=lambda r: r[0]['sequence_id']):
            if scene in sequences or scene not in receipt['sequence_heads']:
                raise ValueError('repeated or unknown sequence block')
            head = receipt['sequence_heads'][scene]
            path = evaluation.child(directory, head['database'])
            if native.sha(path) != head['database_sha256']:
                raise ValueError('sealed completed database required')
            sequences[scene] = inspect_sequence(path, scene, head, pairs, config)
            if native.sha(path) != head['database_sha256']:
                raise ValueError('identity database changed during audit')
    if (set(sequences) != set(receipt['sequence_heads'])
            or sum(r['frames'] for r in sequences.values()) != receipt['completed_frames']):
        raise ValueError('complete sequence and event coverage required')
    return sequences


def summarize(runs):
    if set(runs) != set(evaluation.RULES):
        raise ValueError('all three distinct state updaters required')
    first = runs['jpda-ci']
    if not first or any(set(r) != set(first) for r in runs.values()):
        raise ValueError('identical nonempty sequence coverage required')
    result = {}
    for scene, ref in first.items():
        cells = {rule: run[scene] for rule, run in runs.items()}
        keys = lambda r: [(e['sequence_id'], e['frame_id']) for e in r['events']]
        if any(keys(c) != keys(ref) or c['observations'] != ref['observations'] for c in cells.values()):
            raise ValueError('updater event or observation coverage differs')
        result[scene] = dict(frames=ref['frames'], observations=ref['observations'],
            conditional_scans_identical=len({c['conditional_scan_stream_sha256'] for c in cells.values()}) == 1,
            historical_identity_maps_identical=len({c['identity_anchor_stream_sha256'] for c in cells.values()}) == 1,
            non_state_output_identical=len({c['non_state_prediction_stream_sha256'] for c in cells.values()}) == 1,
            changed_prediction_payload_frames_relative_to_jpda_ci={rule: sum(
                a['payload_sha256'] != b['payload_sha256'] for a, b in zip(ref['events'], c['events']))
                for rule, c in cells.items()})
    return dict(sequences=result, frames=sum(r['frames'] for r in result.values()),
        observations=sum(r['observations'] for r in result.values()),
        all_conditional_scans_identical=all(r['conditional_scans_identical'] for r in result.values()),
        all_historical_identity_maps_identical=all(r['historical_identity_maps_identical'] for r in result.values()),
        all_non_state_outputs_identical=all(r['non_state_output_identical'] for r in result.values()))


def audit(references, output):
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new non-symlink audit output required')
    if len(references) != 3:
        raise ValueError('three complete run and independent-audit references required')
    bound = [evaluation.inspect_run(*r) for r in references]
    rules = [r['plan']['configuration']['update_rule'] for r in bound]
    if set(rules) != set(evaluation.RULES):
        raise ValueError('all three distinct updaters required')
    normalized = []
    for r in bound:
        plan = copy.deepcopy(r['plan']); plan['configuration'].pop('update_rule'); plan['runtime'].pop('pid')
        normalized.append(plan)
    if (any(p != normalized[0] for p in normalized[1:])
            or len({r['audit']['factor_stream_sha256'] for r in bound}) != 1):
        raise ValueError('updater-only configuration and identical actual factors required')
    evidence = {}
    for r in bound:
        for path, value in r['evidence'].items():
            if path in evidence and evidence[path] != value:
                raise ValueError('conflicting shared evidence')
            evidence[path] = value
    for p in (Path(__file__), Path(evaluation.__file__), Path(sys.modules[fingerprint.__module__].__file__), Path(native.__file__)):
        evidence[str(p.absolute())] = native.evidence(p)
    runs = {r['plan']['configuration']['update_rule']: fingerprint_run(ref[0], r['receipt'], r['plan']['configuration'])
            for ref, r in zip(references, bound)}
    result = summarize(runs)
    if result['frames'] != 3316 or len(result['sequences']) != 21:
        raise ValueError('full 3316-frame 21-sequence validation required')
    evaluation.unchanged(evidence)
    result.update(kind='probabilistic_full_val_identity_state_audit_v1', status='complete',
        seed=bound[0]['plan']['checkpoint_seed'], runs=runs,
        database_prefix_hashes_and_published_membership_reconstructed=True,
        inference_receipt_sha256={r['plan']['configuration']['update_rule']: ref[1] for ref, r in zip(references, bound)},
        factor_stream_sha256=bound[0]['audit']['factor_stream_sha256'], input_evidence=evidence,
        GT_payloads_read=False, metrics_recomputed=False, inference_rerun=False,
        hard_identity_correctness_claimed=False, recovery_gain_claimed=False,
        fair_resources_verified=False, three_seed_comparison=False, paper_eligible=False)
    output.mkdir(); native.write_json(output / 'identity-audit.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'seed', 'frames', 'observations',
        'all_conditional_scans_identical', 'all_historical_identity_maps_identical', 'all_non_state_outputs_identical')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs=4, action='append', required=True,
                        metavar=('DIRECTORY', 'RECEIPT_SHA256', 'AUDIT', 'AUDIT_SHA256'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.run, args.output)
