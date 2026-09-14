#!/usr/bin/env python3
"""Evaluator-only, censored identity diagnostics for full frozen SPD val.

Reuse the source-pinned first-unique-ID diagnostic, NOT official IDS or a true
identity-error duration. No GT is sent to inference and no model is trained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
from itertools import groupby, zip_longest
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x import audit_full_mht_validation as mht

common, native = mht.common, mht.native
HELPER = ROOT / 'tools/event_track_v2x/evaluate_train_inference_diagnostic.py'
HELPER_SHA = '7a4a3f4e30596e28383bbf336e45a2ce7c4e696c1b8d012479f0dcf988dd68f9'
EVENT = ROOT / 'tools/event_track_v2x/analyze_mechanism_events_v2.py'
EVENT_SHA = 'e3ad276c1443538f5b7b6b99dce2d7908295335ede9d143a9249f5294cd48963'


def load_helper():
    if native.sha(HELPER) != HELPER_SHA or native.sha(EVENT) != EVENT_SHA:
        raise ValueError('source-pinned identity event definitions required')
    spec = importlib.util.spec_from_file_location('fixed_identity_event_helper', HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper


def aligned(gt, predictions, stream):
    for frame, prediction, raw in zip_longest(gt, predictions, stream):
        if frame is None or prediction is None or raw is None:
            raise ValueError('GT, prediction and audit coverage differs')
        key = lambda v: (v['sequence_id'], v['frame_id'], v['box_reference_timestamp_us'])
        if key(frame) != key(prediction):
            raise ValueError('identity event GT/prediction clock or frame differs')
        yield frame, prediction, raw


def diagnose_sequences(adapter, helper, gt, predictions, tracking_path, temporary_parent):
    """Small-cohort helper; only evaluate() certifies official full-val scope."""
    result = {}; order = []
    with tempfile.TemporaryDirectory(prefix='identity-event-slices-', dir=temporary_parent) as temp:
        with Path(tracking_path).open('rb') as stream:
            groups = groupby(aligned(gt, predictions, stream), key=lambda r: r[0]['sequence_id'])
            for ordinal, (sequence, records) in enumerate(groups):
                if sequence in result: raise ValueError('interleaved identity sequences')
                order.append(sequence)
                if order != sorted(order): raise ValueError('fixed sequence order required')
                frames, preds = [], []
                path = Path(temp) / f'sequence-{ordinal:04d}.jsonl'
                with path.open('xb') as target:
                    for frame, pred, raw in records:
                        frames.append(frame); preds.append(pred); target.write(raw)
                # This helper's event computation itself is split-independent:
                # it consumes one aligned sequence, not train conversion inputs.
                value = helper.identity_events(adapter, frames, preds, path)
                value['kind'] = 'spd_val_sequence_identity_event_diagnostic_v1'
                result[sequence] = value
                print(json.dumps(dict(event='identity_sequence_diagnosed', sequence=sequence,
                                      frames=len(frames))), flush=True)
    if not result: raise ValueError('nonempty identity diagnostic required')
    return result


def summarize(sequences):
    if not sequences: raise ValueError('nonempty sequence diagnostics required')
    counts = Counter(); episodes = []; first = next(iter(sequences.values()))
    for value in sequences.values():
        if value['protocol'] != first['protocol']:
            raise ValueError('identity diagnostic definitions differ across sequences')
        if value['counts']['frames'] != len(value['frames']):
            raise ValueError('identity frame count differs from actual observations')
        counts.update(value['counts']); episodes.extend(value['episodes'])
    if (counts['identity_unique_gt_frames'] + counts['identity_unknown_gt_frames'] != counts['roi_gt_frame_observations']
            or counts['anchor_agreement_gt_frames'] + counts['anchor_disagreement_gt_frames'] != counts['identity_unique_gt_frames']
            or counts['anchor_disagreement_episodes'] != len(episodes)):
        raise ValueError('identity denominators or episode coverage do not reconcile')
    spans = [native._finite(e['observed_error_span_seconds']) for e in episodes]
    if any(x < 0 for x in spans): raise ValueError('negative observed identity span')
    ratio = lambda n, d: None if not counts[d] else counts[n] / counts[d]
    return dict(counts=dict(counts), sequences=len(sequences), frames=counts['frames'],
        anchor_disagreement_given_unique=ratio('anchor_disagreement_gt_frames', 'identity_unique_gt_frames'),
        unknown_fraction_of_roi_gt_observations=ratio('identity_unknown_gt_frames', 'roi_gt_frame_observations'),
        observed_error_span_seconds_quantiles=None if not spans else np.quantile(spans, [.5, .95, .99, 1.]).tolist(),
        quantile_order=['p50', 'p95', 'p99', 'max'], quantiles_from_pooled_episodes_not_mean_sequence_quantiles=True,
        protocol=first['protocol'], identity_anchor_is_not_absolute_ground_truth=True,
        observed_span_is_not_continuous_error_duration=True, unknowns_are_not_assumed_correct=True,
        internal_recovery_is_not_gt_identity_recovery=True, model_bound_is_not_gt_error_probability=True,
        gt_error_calibration_computed=False, official_ids_computed=False, causal_effect_identified=False)


def evaluate(backend, run, receipt_sha, audit_path, audit_sha, ground_truth, output):
    if backend not in ('probabilistic', 'scan-mht'):
        raise ValueError('declared audited backend required')
    output, ground_truth = Path(output).absolute(), Path(ground_truth).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new non-symlink identity output required')
    # Bind independent full inference audit before opening GT or metric engines.
    bound = (common.inspect_run if backend == 'probabilistic' else mht.inspect_audited_run)(
        run, receipt_sha, audit_path, audit_sha)
    evidence = dict(bound['evidence']); helper = load_helper()
    for p in (Path(__file__), HELPER, EVENT, native.ADAPTER_PATH):
        evidence[str(p.absolute())] = native.evidence(p)
    for name, expected in (('manifest.json', native.GT_MANIFEST_SHA256), ('ground-truth.jsonl', native.GT_SHA256)):
        p = common.child(ground_truth, name); value = native.evidence(p)
        if value['sha256'] != expected: raise ValueError('frozen official-val GT required')
        evidence[str(p)] = value
    adapter = native.load_adapter(); runtime = adapter.runtime_evidence(); native.validate_runtime(runtime)
    golden = adapter.golden_cases()
    if golden.get('passed') is not True: raise ValueError('native geometry golden cases failed')
    _, gt = adapter.load_ground_truth(ground_truth)
    with common.child(run, 'predictions.jsonl').open('rb') as stream:
        predictions = [json.loads(line) for line in stream]
    coverage = adapter.validate_predictions(predictions, gt)
    if (coverage['frames'] != 3316 or coverage['sequences'] != 21
            or set(coverage['predictions_per_class']) - {'car'}):
        raise ValueError('full official 3316-frame car-only predictions required')
    details = diagnose_sequences(adapter, helper, gt, predictions, common.child(run, 'tracking.jsonl'), output.parent)
    summary = summarize(details)
    if summary['frames'] != 3316 or summary['sequences'] != 21:
        raise ValueError('full diagnostic coverage differs')
    common.unchanged(evidence)
    output.mkdir()
    for name, value in (('identity-events.json', details), ('runtime.json', runtime), ('golden-cases.json', golden)):
        native.write_json(output / name, value)
    plan = bound['plan']
    summary.update(kind='full_spd_val_identity_event_diagnostic_v1', status='complete', backend=backend,
        seed=plan['checkpoint_seed'], update_rule=plan['configuration'].get('update_rule'),
        configuration=plan['configuration'], run_directory=str(Path(run).absolute()),
        inference_receipt_sha256=receipt_sha, inference_audit_sha256=audit_sha,
        checkpoint_sha256=plan['checkpoint_sha256'], cache_sha256=plan['cache_sha256'],
        schedule_sha256=plan['schedule_sha256'], ground_truth_sha256=native.GT_SHA256,
        files={p.name: native.evidence(p) for p in output.iterdir()}, input_evidence=evidence,
        reporting_scope='car_only', gt_for_evaluation_only=True, validation_already_seen=True,
        validation_parameter_search=False, validation_checkpoint_selection=False,
        native_primary_metrics_computed=False, test_payloads_read=False, parameter_training=False,
        full_paper_comparison_completed=False, paper_eligible=False)
    native.write_json(output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'seed', 'backend', 'update_rule', 'frames', 'counts')}), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('probabilistic', 'scan-mht'), required=True)
    for key in ('run', 'audit', 'ground-truth', 'output'): p.add_argument('--' + key, type=Path, required=True)
    for key in ('receipt-sha256', 'audit-sha256'): p.add_argument('--' + key, required=True)
    a = p.parse_args(); evaluate(a.backend, a.run, a.receipt_sha256, a.audit, a.audit_sha256, a.ground_truth, a.output)
