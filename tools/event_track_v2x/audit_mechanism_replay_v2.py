#!/usr/bin/env python3
"""Independent streaming replay audit; standard library only, no GT/model reads.

This verifies output bytes, commit chains and internal source/lifecycle
provenance. It does not recompute detector features, physical fusion or metric
engines, and does not make those independent audits implicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re

SEEDS = (1337, 2027, 3407)
RUNS = ([{'run_id': f'M0-seed-{s}', 'mode': 'M0', 'seed': s, 'deterministic_control': False} for s in SEEDS]
        + [{'run_id': 'M1-all-unmatched', 'mode': 'M1', 'seed': 1337, 'deterministic_control': True}]
        + [{'run_id': f'{m}-seed-{s}', 'mode': m, 'seed': s, 'deterministic_control': False}
           for m in ('M2', 'M3') for s in SEEDS])
FRAME_COUNT, SEQUENCE_COUNT = 3316, 21
ZERO = '0'*64
FILES = ('predictions.jsonl', 'association.jsonl', 'diagnostics.jsonl.gz', 'receipt.json')
CACHE_SHA = '66c58025bd79ea674ff676f1df8d76e7812c8b1d2b62bc54e8de3ced0bc3aca8'
SCHEDULE_SHA = '2c8999ecbe2ab98bedf13ba2da4b22bd6167eca07b368a99db148abf36de982a'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular evidence file required: '+str(path))
    for parent in path.parents:
        require(not parent.is_symlink(), 'symlink evidence ancestry forbidden')
    return path


def file_hash(path):
    h = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def evidence(path):
    path = regular(path)
    return {'sha256': file_hash(path), 'size': path.stat().st_size}


def distinct_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def decode(raw, *, line=False):
    if line:
        require(raw.endswith(b'\n'), 'JSONL must be newline terminated')
    value = json.loads(raw, object_pairs_hook=distinct_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    require(raw == canonical(value)+(b'\n' if line else b''), 'noncanonical JSON bytes')
    return value


def read_json(path):
    return decode(regular(path).read_bytes())


def close(a, b):
    require(isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a), 'invalid numeric value')
    require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), 'numeric provenance mismatch')


def sha_value(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA256 field')


def scan_inventory(root):
    expected = {'frozen-plan.json', 'summary.json'} | {r['run_id']+'/'+name for r in RUNS for name in FILES}
    paths = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in paths) and not root.is_symlink(), 'symlink in experiment')
    require({p.relative_to(root).as_posix() for p in paths if p.is_file()} == expected, 'experiment file inventory differs from 42-file contract')
    require({p.relative_to(root).as_posix() for p in paths if p.is_dir()} == {r['run_id'] for r in RUNS}, 'unexpected experiment directory')
    return {name: evidence(root/name) for name in sorted(expected)}


def compare_bytes(first, replay):
    left, right = hashlib.sha256(), hashlib.sha256()
    with regular(first).open('rb') as a, regular(replay).open('rb') as b:
        while True:
            aa, bb = a.read(4*1024**2), b.read(4*1024**2)
            require(aa == bb, 'first/replay bytes differ: '+str(first.name))
            if not aa:
                break
            left.update(aa); right.update(bb)
    require(left.hexdigest() == right.hexdigest(), 'replay digest mismatch')
    return left.hexdigest()


def validate_plan_summary(root, inventory):
    plan, summary = read_json(root/'frozen-plan.json'), read_json(root/'summary.json')
    require(plan['kind'] == 'mechanism_diagnostics_v2_plan' and plan['runs'] == RUNS, 'frozen ten-run plan differs')
    require(plan['official_validation_frames'] == FRAME_COUNT and plan['sequences'] == SEQUENCE_COUNT, 'frozen cohort size differs')
    require(plan['cache_sha256'] == CACHE_SHA and plan['schedule_sha256'] == SCHEDULE_SHA, 'cache or schedule identity differs')
    require(plan['device'] == 'cpu' and plan['cpu_threads'] == 1, 'not the frozen CPU runtime')
    for key in ('gt_model_inputs', 'test_payloads_read', 'val_parameter_fitting', 'seed_selection', 'paper_eligible'):
        require(plan[key] is False, 'invalid prediction boundary: '+key)
    require(plan['reporting_scope'] == 'car_only' and plan['config']['deadline_us'] == 100000, 'reporting/deadline changed')
    require(summary['kind'] == 'mechanism_diagnostics_v2_complete' and summary['status'] == 'completed', 'incomplete experiment')
    require(summary['plan_sha256'] == inventory['frozen-plan.json']['sha256'], 'summary plan binding differs')
    require(summary['reporting_scope'] == 'car_only' and summary['paper_eligible'] is False, 'invalid result qualification')
    require(summary['weights_unchanged'] is True and summary['all_required_sealed_streams_match'] is True, 'producer did not establish sealed parity')
    require(set(summary['runs']) == {r['run_id'] for r in RUNS}, 'summary run inventory differs')
    require(isinstance(summary['elapsed_seconds'], (int, float)) and math.isfinite(summary['elapsed_seconds']) and summary['elapsed_seconds'] >= 0, 'invalid elapsed time')
    for spec in RUNS:
        name = spec['run_id']; receipt = read_json(root/name/'receipt.json')
        require(receipt == summary['runs'][name], 'receipt differs from summary')
        require(all(receipt[k] == v for k, v in spec.items()), 'receipt mechanism/seed differs')
        require(receipt['frames'] == FRAME_COUNT and receipt['sequences'] == SEQUENCE_COUNT and receipt['weights_unchanged'] is True, 'receipt coverage/frozen weights differ')
        for base, key in [('predictions.jsonl', 'predictions_sha256'), ('association.jsonl', 'association_sha256'), ('diagnostics.jsonl.gz', 'diagnostics_gzip_sha256')]:
            require(receipt[key] == inventory[name+'/'+base]['sha256'], 'receipt stream digest differs')
        require(receipt['diagnostics_gzip_bytes'] == inventory[name+'/diagnostics.jsonl.gz']['size'], 'receipt diagnostic size differs')
        if spec['mode'] != 'M1':
            require(receipt['sealed_association_parity'] is True, 'missing sealed association declaration')
            require(inventory[name+'/association.jsonl'] == inventory[f"M0-seed-{spec['seed']}/association.jsonl"], 'M2/M3 association differs from same-seed M0')
        if spec['mode'] == 'M0':
            require(receipt['sealed_prediction_parity'] is True, 'missing sealed prediction declaration')
    return plan, summary


def validate_hypotheses(association, diagnostic, selected, mode):
    hypotheses = association['hypotheses']
    require(1 <= len(hypotheses) <= 4, 'invalid retained hypothesis count')
    require(association['full_posterior_omitted_mass'] is None and association['moment_matching'] is True, 'association interpretation differs')
    expected = 'forced_all_unmatched_control' if mode == 'M1' else 'normalized_truncated_joint_energy_with_all_unmatched'
    require(association['posterior_interpretation'] == expected, 'association interpretation differs')
    keys = set()
    for h in hypotheses:
        require(set(h) == {'pairs', 'unmatched_left', 'unmatched_right', 'weight', 'energy'}, 'unknown hypothesis fields')
        require(isinstance(h['weight'], (int, float)) and 0 < h['weight'] <= 1, 'invalid hypothesis mass')
        require(math.isfinite(h['energy']), 'nonfinite association energy')
        pairs = [tuple(pair) for pair in h['pairs']]
        require(all(len(pair) == 2 and all(type(i) is int for i in pair) for pair in pairs), 'invalid pair indices')
        left, right = [x[0] for x in pairs], [x[1] for x in pairs]
        require(len(set(left)) == len(left) and len(set(right)) == len(right), 'not one-to-one')
        for indices, un, count in [(left, h['unmatched_left'], selected[0]), (right, h['unmatched_right'], selected[1])]:
            require(all(type(i) is int for i in un), 'invalid unmatched indices')
            require(sorted(indices+un) == list(range(count)), 'matched/unmatched do not partition selected detections')
        key = tuple(sorted(pairs)); require(key not in keys, 'duplicate hypothesis'); keys.add(key)
    close(sum(h['weight'] for h in hypotheses), 1.)
    if mode == 'M1':
        require(hypotheses == [{'pairs': [], 'unmatched_left': list(range(selected[0])), 'unmatched_right': list(range(selected[1])), 'weight': 1., 'energy': 0.}], 'M1 is not forced all-unmatched')
    records = diagnostic['source_records']
    sources = {}
    for side, count, position in [('vehicle-side', selected[0], 0), ('infrastructure-side', selected[1], 1)]:
        items = [r for r in records if r['side'] == side]
        require([r['selected_index'] for r in items] == list(range(count)), 'source selected provenance incomplete')
        require(len({r['raw_index'] for r in items}) == count, 'duplicate raw source index')
        for r in items:
            require(type(r['raw_index']) is int and r['raw_index'] >= 0 and r['raw_score'] >= .05, 'invalid candidate provenance')
            require(r['frame_id'] == diagnostic['source_frame_ids'][position] and r['class_index'] in (0, 1, 2), 'source frame/class mismatch')
            sources[(side, r['selected_index'])] = r
    require(len(sources) == len(records) == sum(selected), 'unknown source record')
    for h in hypotheses:
        for i, j in h['pairs']:
            require(sources[('vehicle-side', i)]['class_index'] == sources[('infrastructure-side', j)]['class_index'], 'class-incompatible association')
    nodes = diagnostic['nodes']
    require([n['node_index'] for n in nodes] == list(range(len(nodes))), 'node indices are not contiguous')
    expected_nodes = selected[0] + sum(sum(h['weight'] for h in hypotheses if j in h['unmatched_right']) > 0 for j in range(selected[1]))
    require(len(nodes) == expected_nodes, 'compressed node count differs')
    for node in nodes:
        side, index = node['source_side'], node['source_selected_index']
        source = sources[(side, index)]
        require(node['source_frame_id'] == source['frame_id'] and node['source_raw_index'] == source['raw_index'], 'node raw-source provenance differs')
        require(node['class_index'] == source['class_index'], 'node class differs')
        unmatched_key = 'unmatched_left' if side == 'vehicle-side' else 'unmatched_right'
        mass = sum(h['weight'] for h in hypotheses if index in h[unmatched_key])
        close(node['unmatched_mass'], mass); close(node['matched_mass']+mass, 1.)
        if side == 'infrastructure-side':
            close(node['original_score'], source['score'])
            close(node['score'], source['score'] if mode == 'M3' else mass*source['score'])
            require(node['kind'] == 'road_residual' and node['components'] == [], 'road residual structure differs')
        else:
            require(node['kind'] == 'vehicle_anchored_mixture' and len(node['components']) == len(hypotheses), 'mixture components incomplete')
            expected_score = 0.
            for k, (component, h) in enumerate(zip(node['components'], hypotheses)):
                partner = dict(h['pairs']).get(index)
                require(component['hypothesis_index'] == k and component['partner_selected_index'] == partner, 'component association mismatch')
                close(component['weight'], h['weight'])
                wanted_score = source['score'] if partner is None else max(source['score'], sources[('infrastructure-side', partner)]['score'])
                close(component['score'], wanted_score)
                expected_score += h['weight'] * wanted_score
                require(component['partner_raw_index'] == (None if partner is None else sources[('infrastructure-side', partner)]['raw_index']), 'component partner raw index mismatch')
                kind = 'vehicle_unmatched' if partner is None else ('road_state_replacement' if mode == 'M2' else 'cross_source_ci')
                require(component['kind'] == kind, 'component intervention differs')
            close(node['score'], expected_score)


def scan_run(root, spec, plan_sha, deadline_us):
    name, mode = spec['run_id'], spec['mode']; folder = root/name
    receipt = read_json(folder/'receipt.json')
    frames, predictions, nodes, matches = 0, 0, 0, 0
    events, sensor_late, selected_total = Counter(), [0, 0], [0, 0]
    pred_tips, diag_tips, sequence_counts = {}, {}, Counter()
    seen_pairs, source_ids, retired = set(), [set(), set()], set()
    scene, previous_time, previous_ids = None, None, set()
    order_hash = hashlib.sha256()
    with (folder/'predictions.jsonl').open('rb') as p, (folder/'association.jsonl').open('rb') as a, gzip.open(folder/'diagnostics.jsonl.gz', 'rb') as d:
        for raw_pred, raw_assoc, raw_diag in zip_longest(p, a, d):
            require(None not in (raw_pred, raw_assoc, raw_diag), 'prediction/association/diagnostic line counts differ')
            pred, assoc, diag = decode(raw_pred, line=True), decode(raw_assoc, line=True), decode(raw_diag, line=True)
            sid, fid, timestamp = pred['sequence_id'], pred['frame_id'], pred['box_reference_timestamp_us']
            if sid != scene:
                require(sid not in pred_tips and (scene is None or sid > scene), 'sequences must be unique and contiguous sorted')
                scene, previous_time, previous_ids, retired = sid, None, set(), set()
                pred_tips[sid] = diag_tips[sid] = ZERO
            require(type(timestamp) is int and (previous_time is None or timestamp > previous_time), 'nonmonotonic state time')
            previous_time = timestamp
            require((sid, fid) not in seen_pairs, 'repeated reference frame'); seen_pairs.add((sid, fid))
            require(pred['previous_commit_sha256'] == pred_tips[sid], 'prediction predecessor differs')
            claimed = pred['commit_sha256']; sha_value(claimed)
            require(digest({k: v for k, v in pred.items() if k != 'commit_sha256'}) == claimed, 'prediction commit invalid')
            pred_tips[sid] = claimed
            require(diag['previous_diagnostic_commit_sha256'] == diag_tips[sid], 'diagnostic predecessor differs')
            diag_claimed = diag['diagnostic_commit_sha256']; sha_value(diag_claimed)
            require(digest({k: v for k, v in diag.items() if k != 'diagnostic_commit_sha256'}) == diag_claimed, 'diagnostic commit invalid')
            diag_tips[sid] = diag_claimed
            require(diag['kind'] == 'tracking_mechanism_v2_frame' and diag['mode'] == mode and diag['plan_sha256'] == plan_sha, 'diagnostic mode/plan differs')
            require(diag['prediction_commit_sha256'] == claimed and diag['association_frame_sha256'] == hashlib.sha256(raw_assoc[:-1]).hexdigest(), 'diagnostic stream binding invalid')
            for item in (assoc, diag):
                require(item['sequence_id'] == sid and item['frame_id'] == fid, 'three streams have different frames')
            require(pred['coordinate_frame'] == 'world' and pred['state_layout'] == 'gravity_xyz_length_width_height_yaw_vxy', 'output coordinate schema differs')
            require(pred['decision_timestamp_us'] == timestamp+deadline_us, 'deadline differs')
            for key in ('source_cache_sha256', 'source_available', 'selected_detections', 'box_reference_timestamp_us', 'decision_timestamp_us'):
                require(diag[key] == pred[key], 'source/timing/selection binding differs')
            for value in pred['source_cache_sha256']:
                sha_value(value)
            available, selected, information = pred['source_available'], pred['selected_detections'], pred['source_information_timestamp_us']
            require(len(available) == len(selected) == len(information) == len(pred['source_cache_sha256']) == 2, 'two sources required')
            require(len(diag['source_frame_ids']) == 2 and diag['source_frame_ids'][0] == fid, 'source frame binding differs')
            for i in (0, 1):
                require(type(available[i]) is bool and type(information[i]) is int, 'invalid source availability/time')
                require(available[i] == (information[i] <= timestamp+deadline_us), 'future source availability invalid')
                require(type(selected[i]) is int and 0 <= selected[i] <= 64 and (available[i] or selected[i] == 0), 'late/invalid source selection')
                require(diag['source_frame_ids'][i] not in source_ids[i], 'duplicate source frame ID')
                source_ids[i].add(diag['source_frame_ids'][i])
                sensor_late[i] += not available[i]; selected_total[i] += selected[i]
            validate_hypotheses(assoc, diag, selected, mode)
            ids = [item['track_id'] for item in pred['predictions']]
            require(ids == sorted(set(ids)) and all(tid.startswith(sid+':') for tid in ids), 'duplicate/noncanonical output track IDs')
            current_ids = set(ids)
            require(not current_ids & retired, 'retired track ID reappeared')
            require(diag['tracks_before'] == len(previous_ids) and diag['tracks_after'] == len(ids), 'track lifecycle counts differ')
            birth_ids = {event['track_id'] for event in diag['events'] if event['event'] == 'birth'}
            require(birth_ids == current_ids-previous_ids, 'birth events do not match new IDs')
            assigned_ids, assigned_nodes = set(), set()
            for assignment in diag['temporal_assignments']:
                tid, index = assignment['track_id'], assignment['node_index']
                require(tid in previous_ids and tid in current_ids and tid not in assigned_ids and index not in assigned_nodes, 'temporal assignment not one-to-one/live')
                assigned_ids.add(tid); assigned_nodes.add(index)
                require(0 <= index < len(diag['nodes']), 'temporal node index invalid')
                close(assignment['node_score'], diag['nodes'][index]['score'])
                close(assignment['assigned_cost'], .5*assignment['mahalanobis_squared']-math.log(max(assignment['node_score'], 1e-12)))
            retired |= previous_ids-current_ids; previous_ids = current_ids
            events.update(event['event'] for event in diag['events'])
            frames += 1; sequence_counts[sid] += 1; predictions += len(ids)
            nodes += len(diag['nodes']); matches += len(diag['temporal_assignments'])
            order_hash.update(canonical([sid, fid, timestamp, diag['source_frame_ids'], pred['source_cache_sha256'], available, selected])+b'\n')
    require(frames == FRAME_COUNT and len(sequence_counts) == SEQUENCE_COUNT, 'full official-validation cohort not present')
    expected = {'frames': frames, 'predictions': predictions, 'nodes': nodes, 'temporal_matches': matches,
                'events': dict(events), 'source_sensor_late': sensor_late, 'selected_detections': selected_total,
                'sequence_commits': pred_tips, 'diagnostic_sequence_commits': diag_tips, 'sequences': len(sequence_counts)}
    require(all(receipt[k] == v for k, v in expected.items()), 'recomputed stream totals/tips differ from receipt')
    return {**expected, 'sequence_frame_counts': dict(sequence_counts), 'input_order_sha256': order_hash.hexdigest(),
            'prediction_chain_verified': True, 'diagnostic_chain_verified': True, 'association_binding_verified': True,
            'source_and_node_provenance_verified': True, 'duplicate_frame_or_output_identity_count': 0}


def audit(first, replay, output):
    first, replay, output = Path(first), Path(replay), Path(output)
    require(not output.exists() and not output.is_symlink(), 'audit output is create-once')
    require(first.resolve() != replay.resolve(), 'first and replay must be distinct directories')
    require(all(root.resolve() not in output.resolve().parents for root in (first, replay)), 'audit output must be outside both 42-file roots')
    inventories = {key: scan_inventory(root) for key, root in [('first', first), ('replay', replay)]}
    plan, summary = validate_plan_summary(first, inventories['first'])
    replay_plan, replay_summary = validate_plan_summary(replay, inventories['replay'])
    require(plan == replay_plan, 'replay frozen plan differs')
    compare_bytes(first/'frozen-plan.json', replay/'frozen-plan.json')
    require({k: v for k, v in summary.items() if k != 'elapsed_seconds'} ==
            {k: v for k, v in replay_summary.items() if k != 'elapsed_seconds'}, 'summaries differ outside elapsed_seconds')
    results = {}; cohort = None
    for spec in RUNS:
        name = spec['run_id']
        for base in FILES:
            compare_bytes(first/name/base, replay/name/base)
        left = scan_run(first, spec, inventories['first']['frozen-plan.json']['sha256'], plan['config']['deadline_us'])
        right = scan_run(replay, spec, inventories['replay']['frozen-plan.json']['sha256'], plan['config']['deadline_us'])
        require(left == right, 'recomputed first/replay run evidence differs')
        if cohort is None:
            cohort = left['input_order_sha256']
        require(left['input_order_sha256'] == cohort, 'ten runs did not consume the same ordered cohort')
        results[name] = {**spec, **left, 'all_three_streams_byte_identical': True, 'receipt_byte_identical': True}
        print('MECHANISM_REPLAY_AUDITED '+json.dumps({'run_id': name, 'frames': left['frames'], 'sequences': left['sequences']}), flush=True)
    for key, root in [('first', first), ('replay', replay)]:
        require(scan_inventory(root) == inventories[key], 'input evidence changed during audit')
    result = {'kind': 'mechanism_diagnostics_independent_replay_audit_v1', 'status': 'verified',
              'first_root': str(first.resolve()), 'replay_root': str(replay.resolve()),
              'file_manifests': inventories, 'files_per_root': 42, 'run_count': 10, 'runs': results,
              'plan_byte_identical': True, 'summary_difference_allowed': ['elapsed_seconds'],
              'all_30_streams_byte_identical': True, 'all_10_receipts_byte_identical': True,
              'parsed_frames_per_execution': FRAME_COUNT*10, 'executions_scanned': 2,
              'prediction_commits_verified': FRAME_COUNT*20, 'diagnostic_commits_verified': FRAME_COUNT*20,
              'bytes_per_root': {key: sum(v['size'] for v in value.values()) for key, value in inventories.items()},
              'ground_truth_read': False, 'detector_or_model_executed': False, 'cache_payload_rehashed': False,
              'boundary': 'output replay and internal provenance audit, not an independent reconstruction of detector/fusion/metric engines',
              'reporting_scope': 'car_only_metrics; unchanged all-class prediction streams audited', 'paper_eligible': False,
              'auditor_source': evidence(__file__)}
    with output.open('xb') as stream:
        stream.write(canonical(result)+b'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('first', 'replay', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.first, args.replay, args.output)
    print('MECHANISM_REPLAY_AUDIT_COMPLETE '+json.dumps({'status': value['status'], 'runs': value['run_count'],
          'bytes_per_root': value['bytes_per_root']}), flush=True)


if __name__ == '__main__':
    main()
