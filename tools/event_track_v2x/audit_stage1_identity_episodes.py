#!/usr/bin/env python3
"""Full sealed Stage 1 GT-grounded observable identity and event-chain audit.

Geometric source correspondences are qualified diagnostics, not physical-ID
proofs or causal-effect estimates. Unknowns, gaps and ambiguity remain explicit.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
from itertools import zip_longest
import json
import math
from pathlib import Path
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def read(path, digest=None):
    p = Path(path).absolute()
    if not p.is_file() or any(x.is_symlink() for x in (p, *p.parents)): raise ValueError('ordinary original evidence required')
    if digest is not None and sha(p) != digest: raise ValueError('original evidence SHA differs')
    return json.loads(p.read_bytes())


def new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def line(stream, value): stream.write(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')


def module(path, digest, name):
    p = Path(path).absolute()
    if sha(p) != digest or p.is_symlink(): raise ValueError('frozen independent source differs')
    spec = importlib.util.spec_from_file_location(name, p); result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def similarities(evaluator, left, right):
    """Exact IoU kernel with conservative vectorized bounding-box rejection."""
    out = np.zeros((len(left), len(right)))
    if not left or not right: return out
    def arrays(boxes):
        centers = np.array([x['mean'][:2] for x in boxes]); extents = []
        for box in boxes:
            m = box['mean']; c, s = abs(math.cos(m[6])), abs(math.sin(m[6]))
            extents.append(((c * m[3] + s * m[4]) / 2, (s * m[3] + c * m[4]) / 2))
        return centers, np.asarray(extents)
    a, ae = arrays(left); b, be = arrays(right)
    candidates = np.all(abs(a[:, None, :] - b[None, :, :]) <= ae[:, None, :] + be[None, :, :] + 1e-10, axis=2)
    for i, j in zip(*np.nonzero(candidates)): out[i, j] = evaluator.bev_iou(left[i]['mean'], right[j]['mean'])
    return out


def full_id_correspondences(evaluator, oracle, gt, predictions, expected):
    groups = defaultdict(list); correspondence = {}; owners = {}; checks = []
    for g, p in zip(gt, predictions): groups[g['sid']].append((g, p))
    for sid, frames in groups.items():
        matches = []
        for g, p in frames:
            gs = [x['track_id'] for x in g['boxes']]; ps = [x['track_id'] for x in p['boxes']]
            matrix = similarities(evaluator, g['boxes'], p['boxes'])
            matches.append((gs, ps, matrix)); correspondence[(sid, g['fid'])] = oracle.unique_correspondences(matrix, gs, ps)
        gids = sorted({x for g, _, _ in matches for x in g}); pids = sorted({x for _, p, _ in matches for x in p})
        result = oracle.global_owners(matches, gids, pids)
        original = expected[sid]['Identity']
        if any(result[k] != original[k] for k in ('IDTP', 'IDFN', 'IDFP')):
            raise ValueError('complete native Identity objective counts differ: ' + sid)
        owners[sid] = result['owners']
        checks.append(dict(sequence_id=sid, original_Identity_counts_exact=True, IDTP=result['IDTP'], IDFN=result['IDFN'], IDFP=result['IDFP'],
                           ownership_status_counts=dict(Counter(x['status'] for x in result['owners'].values()))))
        print(json.dumps(dict(stage='global_identity_ownership', sequence_id=sid,
            checked_sequences=len(checks), expected_sequences=len(groups))), flush=True)
    return correspondence, owners, checks


def source_and_nodes(evaluator, oracle, diagnostic, raw_GT_objects):
    # Full original Car GT for source diagnostics; output errors remain car ROI.
    truth = [x for x in raw_GT_objects if x['class_label'] == 'car']
    source_correspondence = {}; all_nodes = {}
    for side in ('vehicle-side', 'infrastructure-side'):
        records = [x for x in diagnostic['source_records'] if x['side'] == side and x['class_index'] == 0]
        raw = [dict(track_id=str(x['selected_index']), mean=x['mean']) for x in records]
        qualified = oracle.unique_correspondences(similarities(evaluator, raw, truth),
            [x['track_id'] for x in raw], [x['track_id'] for x in truth])
        for record in records: source_correspondence[(side, record['selected_index'])] = qualified[str(record['selected_index'])]
    for node in diagnostic['nodes']:
        if node['class_index'] != 0: continue
        anchor = source_correspondence[(node['source_side'], node['source_selected_index'])]
        wrong = known_same = unknown = 0.; components = []
        for component in node['components']:
            partner = component['partner_selected_index']
            if partner is None: continue
            other = source_correspondence[('infrastructure-side', partner)]
            qualified = anchor['status'] == other['status'] == 'unique'
            status = 'same_qualified_GT' if qualified and anchor['other'] == other['other'] else ('different_qualified_GT' if qualified else 'unknown_source_correspondence')
            weight = component['weight']
            if status == 'different_qualified_GT': wrong += weight
            elif status == 'same_qualified_GT': known_same += weight
            else: unknown += weight
            components.append(dict(hypothesis_index=component['hypothesis_index'], weight=weight, kind=component['kind'],
                anchor_qualified_GT=anchor['other'], partner_qualified_GT=other['other'], correspondence_status=status,
                partner_raw_index=component['partner_raw_index'], partner_selected_index=partner,
                actual_component_mean=component['mean'], actual_component_covariance_sha256=component['covariance_sha256']))
        if node['kind'] == 'vehicle_anchored_mixture':
            if abs(wrong + known_same + unknown - node['matched_mass']) > 1e-12:
                raise ValueError('qualified/unknown paired masses do not cover actual matched mass')
            nonexecuted_pair_mass = 0.
        elif node['kind'] == 'road_residual':
            # A road residual is supported by the unmatched hypotheses. Its
            # matched_mass describes the other hypotheses, not executed CI
            # components of this node (which has an empty component list).
            if node['components']:
                raise ValueError('road residual must not claim active paired components')
            nonexecuted_pair_mass = node['matched_mass']
        else:
            raise ValueError('unknown original diagnostic node kind')
        all_nodes[node['node_index']] = dict(node_index=node['node_index'], kind=node['kind'],
            source_side=node['source_side'], source_raw_index=node['source_raw_index'], source_selected_index=node['source_selected_index'],
            source_frame_id=node['source_frame_id'], anchor_correspondence=anchor,
            score=node['score'], birth_effective_score=node.get('birth_effective_score'), actual_node_mean=node['mean'],
            actual_node_covariance_sha256=node['covariance_sha256'], matched_mass=node['matched_mass'], unmatched_mass=node['unmatched_mass'],
            qualified_different_GT_pair_mass=wrong, qualified_same_GT_pair_mass=known_same,
            unknown_pair_mass=unknown, active_component_pair_mass=wrong + known_same + unknown,
            nonexecuted_pair_mass_for_residual=nonexecuted_pair_mass, components=components)
    return all_nodes


def audit(args):
    root = Path(args.root).absolute(); out = Path(args.output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)): raise ValueError('fresh evaluator-only output directory required')
    proof = read(root / 'independent-full-provenance/provenance-receipt.json', args.provenance_sha256)
    if not proof['complete_original_cohort_scanned'] or not proof['all_internal_output_provenance_passed']: raise ValueError('full original output-chain admission required')
    run_id = args.run_id; campaign = 'birth' if run_id.startswith('M4-') else 'mechanism'
    records = [x for x in proof['runs'] if x['run_id'] == run_id and x['execution'] == 'first']
    if len(records) != 1: raise ValueError('original logical run required')
    stream_root = root / 'streams' / campaign / 'first' / run_id
    evaluator = module(args.evaluator_source, args.evaluator_sha256, 'frozen_independent_episode_geometry')
    oracle = module(args.oracle_source, args.oracle_sha256, 'frozen_independent_episode_ownership')
    metric_root = root / 'metric-original-readback-v2'
    metric_manifest_path = metric_root / 'original-byte-manifest.json'
    metric_manifest = read(metric_manifest_path, args.metrics_manifest_sha256)
    report_path = metric_root / campaign / run_id / 'report.json'
    metric_path = metric_root / campaign / run_id / 'metrics.json'
    report_digest = metric_manifest[f'{campaign}/{run_id}/report.json']['sha256']
    metric_digest = metric_manifest[f'{campaign}/{run_id}/metrics.json']['sha256']
    original = read(report_path, report_digest)
    if metric_digest != original['files']['metrics.json']['sha256']:
        raise ValueError('original metric manifest/report binding differs')
    metrics = read(metric_path, metric_digest)
    gt_path = root / 'evaluator-only-ground-truth/ground-truth.jsonl'; pred_path = stream_root / 'predictions.jsonl'; diag_path = stream_root / 'diagnostics.jsonl.gz'
    if original['predictions_sha256'] != proof['input_sha256'][str(pred_path)]:
        raise ValueError('original evaluator/preserved predictor stream binding differs')
    pins = {str(gt_path): evaluator.GT_SHA, str(pred_path): original['predictions_sha256'],
            str(diag_path): proof['input_sha256'][str(diag_path)], str(Path(args.evaluator_source).absolute()): args.evaluator_sha256,
            str(Path(args.oracle_source).absolute()): args.oracle_sha256, str(Path(__file__).absolute()): sha(__file__),
            str(root / 'evaluator-only-ground-truth/manifest.json'): evaluator.GT_MANIFEST_SHA,
            str(root / 'independent-full-provenance/provenance-receipt.json'): args.provenance_sha256,
            str(metric_manifest_path): args.metrics_manifest_sha256, str(report_path): report_digest, str(metric_path): metric_digest}
    for p, digest in pins.items():
        if sha(p) != digest: raise ValueError('frozen stream/GT/source SHA differs')
    out.mkdir(parents=True); started = time.monotonic()
    new(out / 'plan.json', dict(kind='rbf_stage1_GT_grounded_observable_episode_plan_v1', run_id=run_id,
        source_sha256=sha(__file__), independent_oracle_sha256=args.oracle_sha256, independent_geometry_sha256=args.evaluator_sha256,
        GT_future_used_only_after_sealed_prediction=True, global_identity_objective='original TrackEval Identity .5 IoU assignment; positive edges qualified by forbidden-edge integer cost margin',
        geometric_correspondence='mutually unique >=.5 oriented BEV IoU; ambiguity and absence unknown',
        source_diagnostic_GT='full original Car GT at reference frame; common-reference propagated source boxes; no physical identity proof',
        durations='consecutive native-frame wrong labels; preserve each actual interval and censor unknown/absent GT/sequence boundaries; no hidden-time continuity claim',
        event_chain='actual source component, node, birth/update/miss/kill and sealed output references; structural links do not identify causal effect',
        fragmented_observation='qualified GT changes predicted ID, with matching-gap count; distinct from official nuScenes Frag',
        strict_M3_2027_failure_preserved=True, stage1_qualified=False, paper_metrics_complete=False))
    gt, predictions, counts = evaluator.load_frames(gt_path, pred_path, original)
    correspondences, owners, identity_checks = full_id_correspondences(evaluator, oracle, gt, predictions, metrics['trackeval']['car']['sequences'])
    new(out / 'global-identity-ownership.json', dict(owners=owners, full_21_sequence_Identity_counts_exact=identity_checks,
        arbitrary_first_GT_anchor_used=False, predictions_updated=False, global_GT_used_only_by_evaluator=True))
    labels_count = Counter(); event_counts = Counter(); node_counts = Counter(); fragments = 0; frames = 0
    episodes = oracle.Episodes(); current_sid = None; frame_index = -1; last_detection = {}; latest_node = {}; birth_nodes = {}
    with gt_path.open() as raw_GT, pred_path.open() as raw_pred, gzip.open(diag_path, 'rt') as diag, \
         (out / 'identity-observations.jsonl').open('x') as observations, (out / 'node-component-links.jsonl').open('x') as nodes_file, \
         (out / 'lifecycle-links.jsonl').open('x') as lifecycle, (out / 'observed-fragments.jsonl').open('x') as fragment_file:
        for raw_g, raw_p, raw_d, g, p in zip_longest(raw_GT, raw_pred, diag, gt, predictions):
            if any(x is None for x in (raw_g, raw_p, raw_d, g, p)): raise ValueError('full frame/diagnostic coverage differs')
            original_GT = json.loads(raw_g); pf = json.loads(raw_p); df = json.loads(raw_d)
            sid, fid, timestamp = g['sid'], g['fid'], g['time']
            if (df['sequence_id'], df['frame_id'], df['box_reference_timestamp_us']) != (sid, fid, timestamp) or df['prediction_commit_sha256'] != pf['commit_sha256']:
                raise ValueError('actual diagnostic/frame/prediction commitment differs')
            if current_sid != sid:
                episodes.end_sequence(); current_sid = sid; frame_index = -1; last_detection = {}; latest_node = {}; birth_nodes = {}
            frame_index += 1; frames += 1
            nodes = source_and_nodes(evaluator, oracle, df, original_GT['objects'])
            prefix = dict(sequence_id=sid, frame_id=fid, frame_index=frame_index, timestamp_us=timestamp, prediction_commit_sha256=pf['commit_sha256'], diagnostic_commit_sha256=df['diagnostic_commit_sha256'])
            for node in nodes.values():
                node_counts['car_nodes'] += 1
                if node['qualified_different_GT_pair_mass'] > 0: node_counts['nodes_with_nonzero_qualified_different_GT_pair_mass'] += 1
                if node['unknown_pair_mass'] > 0: node_counts['nodes_with_unknown_pair_mass'] += 1
                line(nodes_file, {**prefix, **node, 'causal_effect_identified': False})
            direct = {}
            for event in df['events']:
                event_counts[event['event']] += 1; tid = event.get('track_id'); index = event.get('node_index')
                if tid is not None and index in nodes:
                    link = dict(source_frame_id=fid, node_index=index, qualified_different_GT_pair_mass=nodes[index]['qualified_different_GT_pair_mass'],
                                unknown_pair_mass=nodes[index]['unknown_pair_mass'])
                    direct[tid] = latest_node[tid] = link
                    if event['event'] == 'birth': birth_nodes[tid] = link
                line(lifecycle, {**prefix, **event, 'unique_global_identity_owner': owners[sid].get(tid),
                                 'current_node_reference': direct.get(tid), 'latest_observed_node_reference': latest_node.get(tid),
                                 'birth_node_reference': birth_nodes.get(tid), 'causal_effect_identified': False})
            for assignment in df['temporal_assignments']:
                tid, index = assignment['track_id'], assignment['node_index']
                if index in nodes:
                    link = dict(source_frame_id=fid, node_index=index, qualified_different_GT_pair_mass=nodes[index]['qualified_different_GT_pair_mass'], unknown_pair_mass=nodes[index]['unknown_pair_mass'])
                    direct[tid] = latest_node[tid] = link
                line(lifecycle, {**prefix, 'event': 'actual_temporal_assignment', **assignment,
                    'unique_global_identity_owner': owners[sid].get(tid), 'current_node_reference': direct.get(tid),
                    'birth_node_reference': birth_nodes.get(tid), 'causal_effect_identified': False})
            frame_labels = {}; links = {}
            for box in g['boxes']:
                gid = box['track_id']; c = correspondences[(sid, fid)][gid]
                label = oracle.identity_label(c, owners[sid], gid); frame_labels[gid] = label
                tid = label.get('prediction_id'); labels_count[label['label']] += 1
                links[gid] = dict(frame_id=fid, prediction_commit_sha256=pf['commit_sha256'], prediction_id=tid,
                    actual_output_GT_correspondence=c, current_node_reference=direct.get(tid), latest_observed_node_reference=latest_node.get(tid),
                    birth_node_reference=birth_nodes.get(tid), causal_effect_identified=False)
                line(observations, {**prefix, 'GT_identity': gid, **label, 'event_links': links[gid]})
                if c['status'] == 'unique':
                    observed_tid = c['other']; old = last_detection.get(gid)
                    if old is not None and old['prediction_id'] != observed_tid:
                        fragments += 1
                        line(fragment_file, {**prefix, 'GT_identity': gid, 'previous_qualified_observation': old,
                            'current_prediction_id': observed_tid, 'intervening_native_frames_without_same_qualified_detection': frame_index - old['frame_index'] - 1,
                            'official_nuScenes_Frag_equivalence_claimed': False, 'causal_effect_identified': False})
                    last_detection[gid] = dict(prediction_id=observed_tid, frame_id=fid, frame_index=frame_index, timestamp_us=timestamp)
            episodes.step(sid, frame_index, timestamp, frame_labels, links)
            if frames % 256 == 0: print(json.dumps(dict(stage='GT_grounded_identity_event_chain',run_id=run_id,frames=frames,total=3316)),flush=True)
    episodes.end_sequence()
    if frames != 3316 or len(counts) != 21 or len(identity_checks) != 21: raise ValueError('full native car scope required')
    new(out / 'observed-error-episodes.json', dict(episodes=episodes.finished, arbitrary_first_GT_anchor_used=False,
        hidden_time_continuity_claimed=False, physical_source_identity_proven=False, causal_effect_identified=False))
    for path, digest in pins.items():
        if sha(path) != digest: raise ValueError('sealed inputs changed during full episode audit')
    strict = next(x['all_three_streams_byte_identical'] for x in proof['strict_replay_byte_parity'] if x['run_id'] == run_id)
    receipt = dict(kind='rbf_stage1_full_native_GT_grounded_observable_identity_episode_and_structural_chain_v1', run_id=run_id,
        frames=frames, sequences=21, full_native_output_GT_diagnostic_alignment_verified=True, all_21_IDTP_IDFN_IDFP_original_objectives_exact=True,
        observation_label_counts=dict(labels_count), lifecycle_event_counts=dict(event_counts), node_counts=dict(node_counts),
        observed_error_episodes=len(episodes.finished), censored_error_episodes=sum(x['right_censored'] for x in episodes.finished),
        qualified_prediction_identity_changes=fragments, full_input_frames_categorized_including_unknowns=True,
        original_strict_replay_pass=strict, original_M3_2027_failure_preserved=True, arbitrary_first_GT_anchor_used=False,
        source_correspondence_is_geometric_diagnostic=True, physical_source_identity_proven=False, hidden_time_continuity_claimed=False,
        causal_effect_identified=False, complete_counterfactual_error_propagation_admitted=False, complete_stage1=False,
        paper_metrics_complete=False, input_sha256=pins, seconds=time.monotonic()-started,
        output_files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()})
    new(out / 'episode-receipt.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('input_sha256','output_files')}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','run-id','provenance-sha256','metrics-manifest-sha256','evaluator-source','evaluator-sha256','oracle-source','oracle-sha256','output'):
        p.add_argument('--' + name, required=True)
    audit(p.parse_args())
