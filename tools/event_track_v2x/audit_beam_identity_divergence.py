#!/usr/bin/env python3
"""Evaluator-only native SWITCH events and first recovery-output intervention.

Reads a sealed, complete single-sequence comparison. Reproduces native threshold
curves and event accumulation, without changing predictions or metric sources.
GT IDs are hashed in the report; no GT is supplied to an inference process.
History certificates distinguish current-event search from recovery of a class
outside the previous retained/output set. Neither is proof of correct identity.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x import evaluate_train_inference_diagnostic as evaluation

OFF, ON = 'beam_recovery_disabled', 'beam_recovery'


def key_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def root_partition(choices):
    roots = []
    for i, p in enumerate(choices):
        if type(p) is not int or not -1 <= p < i:
            raise ValueError('legal backward identity choices required')
        roots.append(i if p == -1 else roots[p])
    return tuple(roots)


def restrict_partition(roots, members, predecessor_members):
    if len(roots) != len(members) or len(set(members)) != len(members):
        raise ValueError('complete distinct historical member map required')
    local = {g: i for i, g in enumerate(members)}
    if len(set(predecessor_members)) != len(predecessor_members) or not set(predecessor_members) <= set(local):
        raise ValueError('predecessor must be a distinct subset of current members')
    first = {}; result = []
    for i, g in enumerate(predecessor_members):
        root = roots[local[g]]
        result.append(first.setdefault(root, i))
    return tuple(result)


def choices_for(db, component, handle, nodes):
    if any(type(v) is not int for v in (component, handle, nodes)) or component < 1 or handle < 0 or nodes < 0:
        raise ValueError('valid internal namespace and depth required')
    choices = []
    for depth in range(nodes, -1, -1):
        row = db.execute(f'SELECT parent,depth,choice FROM pc{component}_prefixes WHERE h=?', (handle,)).fetchone()
        if row is None or row[1] != depth:
            raise ValueError('historical output prefix depth differs')
        if depth == 0:
            if handle != 0 or row[0] is not None or row[2] is not None:
                raise ValueError('historical prefix did not end at root')
            break
        choices.append(row[2]); handle = row[0]
    return tuple(reversed(choices))


def members_for(db, component, nodes):
    rows = db.execute('SELECT local_i,global_i FROM component_members WHERE component=? AND local_i<? '
                      'ORDER BY local_i', (component, nodes)).fetchall()
    if [r[0] for r in rows] != list(range(nodes)):
        raise ValueError('historical member map depth differs')
    return [r[1] for r in rows]


def history_certificate(db, current, previous):
    """Independent root-equivalence projection; never reads final live metadata."""
    c, n = current['component'], current['nodes']
    roots = root_partition(choices_for(db, c, current['output_handle'], n))
    members = members_for(db, c, n)
    predecessors = []
    for p in current['predecessors']:
        if p not in previous:
            # A genuinely new component has no previous output to recover.
            if p != c:
                raise ValueError('missing historical predecessor summary')
            continue
        old = previous[p]; old_members = members_for(db, p, old['nodes'])
        projection = restrict_partition(roots, members, old_members)
        retained = {r['handle'] for r in old['active']} | {old['output_handle']}
        compatible = sorted(h for h in retained if
            root_partition(choices_for(db, p, h, old['nodes'])) == projection)
        predecessors.append(dict(component=p, historical_nodes=old['nodes'],
            compatible_previous_retained_or_output_handles=compatible,
            outside_previous_retained_and_output=not compatible))
    declared = {r['previous_component'] for r in current['recovery_events']}
    independent = {r['component'] for r in predecessors if r['outside_previous_retained_and_output']}
    if declared != independent:
        raise ValueError('recorded recovery differs from independent historical class projection')
    return dict(predecessors=predecessors, restored_previous_dropped_output_class=bool(independent),
                historical_summary_only=True, numerical_handles_only_compared_within_same_database=True)


def tracking_events(tracking):
    with Path(tracking).open('rb') as stream:
        for raw in stream:
            yield json.loads(raw)['tracking']


def first_intervention(tracking, db):
    previous = {}; first = None; direct = []; recovery_records = 0
    for event in tracking_events(tracking):
        for component in event['components']:
            recovery_records += len(component['recovery_events'])
            if component['output_handle'] == component['backbone_output_handle']:
                continue
            direct.append(dict(frame_id=event['event_id'], component=component['component']))
            if first is not None:
                continue
            certificate = history_certificate(db, component, previous)
            before = {r['handle']: r for r in component['backbone_active']}
            after = {r['handle']: r for r in component['active']}
            chosen = after.get(component['output_handle'])
            backbone = before.get(component['backbone_output_handle'])
            if chosen is None or backbone is None:
                raise ValueError('first intervention is not an explicitly weighted candidate')
            first = dict(frame_id=event['event_id'], sequence_id=event['sequence_id'],
                component=component['component'], nodes=component['nodes'],
                chosen=chosen, backbone=backbone, decision=component['decision'],
                chosen_in_current_backbone=component['output_handle'] in before,
                candidate_log_weight_difference=chosen['log_weight'] - backbone['log_weight'],
                recovery_steps=component['recovery_steps'], history_certificate=certificate)
        previous.update({r['component']: r for r in event['components']})
    return dict(first=first, direct_interventions=direct, direct_intervention_count=len(direct),
                declared_recovery_event_count=recovery_records,
                certificate_scope='first direct output intervention only; other records are not reclassified')


def selected_index(curves):
    values = np.array([np.nan if v is None else v for v in curves['mota']], dtype=float)
    if values.ndim != 1 or not len(values) or np.all(np.isnan(values)):
        raise ValueError('native best-MOTA threshold unavailable')
    index = int(np.nanargmax(values))
    threshold = curves['confidence'][index]
    if type(threshold) not in (int, float) or not math.isfinite(threshold):
        raise ValueError('native selected score threshold must be finite')
    return index, float(threshold)


def native_events(engine, frames, threshold):
    """Rebuild original-ID events and prove equality to unchanged native events."""
    import pandas as pd
    from sklearn.metrics.pairwise import euclidean_distances
    from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
    if len(engine.tracks_gt) != 1 or engine.class_name != 'car' or engine.dist_fcn.__name__ != 'center_distance':
        raise ValueError('one car sequence with native XY matching required')
    sid = next(iter(engine.tracks_gt)); ground = engine.tracks_gt[sid]; predicted = engine.tracks_pred[sid]
    if [(f['sequence_id'], f['box_reference_timestamp_us']) for f in frames] != [(sid, t) for t in ground]:
        raise ValueError('native scene timestamp order differs from sealed frames')
    accumulator = MOTAccumulatorCustom(); timeline = []
    for frame in frames:
        t = frame['box_reference_timestamp_us']
        gt = [b for b in ground[t] if b.tracking_name == 'car']
        pred = [b for b in predicted[t] if b.tracking_name == 'car' and b.tracking_score >= threshold]
        if not gt and not pred:
            continue
        distances = (euclidean_distances(np.array([b.translation[:2] for b in gt]),
                                        np.array([b.translation[:2] for b in pred])) if gt and pred else np.ones((0, 0)))
        distances[distances >= engine.dist_th_tp] = np.nan
        accumulator.update([b.tracking_id for b in gt], [b.tracking_id for b in pred], distances, frameid=len(timeline))
        timeline.append(dict(frame_id=frame['frame_id'], timestamp_us=t))
    reference, _ = engine.accumulate_threshold(threshold)
    reconstructed = MOTAccumulatorCustom.merge_event_dataframes([accumulator])
    pd.testing.assert_frame_equal(reconstructed, reference, check_exact=True)
    records = []
    for (frame, _), event in accumulator.events.iterrows():
        if event['Type'] not in ('MATCH', 'SWITCH', 'MISS'):
            continue
        records.append(dict(**timeline[frame], type=str(event['Type']), gt_id=event['OId'],
            pred_id=None if pd.isna(event['HId']) else str(event['HId']),
            distance_m=None if pd.isna(event['D']) else float(event['D'])))
    return records


def switches(records):
    previous = {}; result = []
    for record in records:
        if record['type'] == 'MISS':
            continue
        gt = record['gt_id']; old = previous.get(gt)
        if record['type'] == 'SWITCH':
            if old is None or old['pred_id'] == record['pred_id']:
                raise ValueError('native SWITCH has no different prior association')
            result.append(dict(frame_id=record['frame_id'], timestamp_us=record['timestamp_us'],
                gt_key_sha256=key_hash(gt), previous_pred_id=old['pred_id'], current_pred_id=record['pred_id'],
                previous_native_match_frame=old['frame_id'], distance_m=record['distance_m'],
                elapsed_since_previous_native_match_seconds=(record['timestamp_us'] - old['timestamp_us']) / 1e6))
        previous[gt] = record
    return result


def analyze_native(adapter, ground, predictions, metrics):
    from nuscenes.eval.tracking.algo import TrackingEvaluation
    from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingMetricData
    cfg = TrackingConfig.deserialize(metrics['nuscenes']['cfg'])
    if list(cfg.class_names) != ['car'] or cfg.dist_th_tp != 2. or TrackingMetricData.nelem != 40:
        raise ValueError('sealed native car metric configuration required')
    gt, pred = adapter._nuscenes_tracks(ground, predictions, ('car',))
    engine = TrackingEvaluation(gt, pred, 'car', cfg.dist_fcn_callable, cfg.dist_th_tp, cfg.min_recall,
                                TrackingMetricData.nelem, cfg.metric_worst, verbose=False)
    curves = adapter._clean(engine.accumulate().serialize())
    if curves != metrics['nuscenes_curves']['car']:
        raise ValueError('recomputed native threshold curves differ from sealed metrics')
    index, threshold = selected_index(curves)
    counts = {k: curves[k][index] for k in ('mota', 'motp', 'ids', 'fp', 'fn', 'frag', 'gt', 'tp')}
    if any(v != metrics['nuscenes'][k] for k, v in counts.items()):
        raise ValueError('native selected threshold differs from headline metrics')
    records = native_events(engine, ground, threshold); events = switches(records)
    if len(events) != counts['ids']:
        raise ValueError('native SWITCH records do not reproduce IDS count')
    return dict(selected_index=index, score_threshold=threshold, selected_native_metrics=counts,
                switch_events=events, native_curves_exact=True, native_event_dataframe_exact=True), records


def audit(comparison, digest, output):
    value = evaluation.read(comparison, digest)
    if (value.get('kind') != 'train_beam_recovery_controls_v1' or value.get('status') != 'complete'
            or value.get('actual_complete_factor_stream_identical') is not True
            or value.get('recovery_toggle_only_configuration_difference') is not True
            or value.get('validation') is not False or set(value['cells']) != {OFF, ON}):
        raise ValueError('complete car train recovery-toggle comparison required')
    evidence = {Path(p): h for p, h in value['input_sha256'].items()}
    for path, expected in ((Path(comparison), digest), (Path(__file__), evaluation.sha(__file__)),
                           (Path(evaluation.__file__), evaluation.sha(evaluation.__file__))):
        if evidence.setdefault(path, expected) != expected:
            raise ValueError('conflicting sealed evidence identity')
    if any(evaluation.sha(p) != h for p, h in evidence.items()):
        raise ValueError('sealed comparison evidence changed')
    gt_paths = [p for p in evidence if p.name == 'ground-truth.jsonl']
    if len(gt_paths) != 1:
        raise ValueError('exactly one sealed evaluator GT stream required')
    load = lambda p: [json.loads(line) for line in Path(p).read_bytes().splitlines()]
    ground = load(gt_paths[0]); module, adapter = evaluation.evaluator()
    runtime = adapter.runtime_evidence(); results = {}; records = {}
    for backend in (OFF, ON):
        cell = value['cells'][backend]; root = Path(cell['source_directory'])
        if cell['runtime'] != runtime or len(ground) != value['frames']:
            raise ValueError('current native evaluator runtime or frame coverage differs')
        pred_path = root / 'predictions.jsonl'
        if pred_path not in evidence:
            raise ValueError('unbound predictions')
        pred = load(pred_path); adapter.validate_predictions(pred, ground)
        metric_path = Path(cell['metric_path'])
        if evidence.get(metric_path) != cell['metric_sha256']:
            raise ValueError('sealed metric output identity differs')
        metric = evaluation.read(metric_path, cell['metric_sha256'])
        if (metric['backend'] != backend or metric['inference_receipt_sha256'] != cell['inference_receipt_sha256']
                or metric['protocol']['evaluated_classes'] != ['car'] or metric['counts']['sequences'] != 1):
            raise ValueError('metrics do not refer to this complete car replay')
        results[backend], records[backend] = analyze_native(adapter, ground, pred, metric['metrics'])
    cross = []
    off_lookup = {(r['timestamp_us'], key_hash(r['gt_id'])): r for r in records[OFF]}
    for event in results[ON]['switch_events']:
        other = off_lookup.get((event['timestamp_us'], event['gt_key_sha256']))
        cross.append(dict(frame_id=event['frame_id'], gt_key_sha256=event['gt_key_sha256'],
                          off_event=None if other is None else {k: other[k] for k in ('type', 'pred_id', 'distance_m')}))
    root = Path(value['cells'][ON]['source_directory']); receipt = evaluation.read(root / 'receipt.json', evidence[root / 'receipt.json'])
    if len(receipt['sequence_heads']) != 1:
        raise ValueError('one complete sequence database required')
    database = root / next(iter(receipt['sequence_heads'].values()))['database']
    if database not in evidence or root / 'tracking.jsonl' not in evidence:
        raise ValueError('unbound inference history')
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        intervention = first_intervention(root / 'tracking.jsonl', db)
    if adapter.runtime_evidence() != runtime or any(evaluation.sha(p) != h for p, h in evidence.items()):
        raise ValueError('native runtime or evidence changed during audit')
    result = dict(kind='train_beam_identity_divergence_audit_v1', status='complete', frames=len(ground),
        native=results, off_association_at_on_switches=cross, intervention=intervention,
        threshold_policy='unchanged native first nanargmax MOTA index per run; track-averaged scores are evaluator-only',
        inference_rerun=False, inference_modified=False, gt_for_evaluation_only=True, gt_ids_or_boxes_in_report=False,
        time_since_match_is_not_identity_error_duration=True, event_timing_is_not_causal_attribution=True,
        model_score_is_not_true_identity_probability=True, full_method=False, validation=False, paper_eligible=False,
        input_sha256={str(p): h for p, h in evidence.items()})
    destination = evaluation.new_directory(output); module.write_json(destination / 'identity-divergence.json', result)
    print(json.dumps(dict(status='complete', frames=len(ground), native_ids={b: len(results[b]['switch_events']) for b in results},
        first_intervention_frame=None if intervention['first'] is None else intervention['first']['frame_id'], paper_eligible=False)), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--comparison-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.comparison, args.comparison_sha256, args.output)
