#!/usr/bin/env python3
"""Compare all 12 frozen car SPD-val adaptations, never select favourable cells.

Three K=4 global scan-MHT runs and nine JPDA/PKF runs are required. Raw factor
streams, models, full evaluation protocol and shared runtime must agree. These
are descriptive seed comparisons, NOT equal-resource or recovery-only effects.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x import compare_probabilistic_validation as baseline
from tools.event_track_v2x import evaluate_mht_validation as mht

native, common, contract = baseline.native, baseline.evaluation, mht.contract
SEEDS, RULES = common.SEEDS, common.RULES
MHT = 'scan-mht-k4'
EXTRA_SOURCES = {
    'tools/event_track_v2x/run_mht_tracking_v2.py',
    'tools/event_track_v2x/persistent_mht_tracking.py',
    'tools/event_track_v2x/ranked_partial_assignment.py',
}
EXTRA_RUNTIME = {'assignment_binary_sha256', 'scipy', 'sqlite_version'}
RESOURCE_KEYS = ('elapsed_seconds', 'process_peak_rss_bytes', 'database_bytes',
    'latency_seconds_p50_p95_p99_max', 'frame_latency_seconds_p50_p95_p99_max',
    'latency_scope', 'frame_latency_scope', 'latency_clock', 'latency_quantile_method')


def resources(receipt, runtime):
    """Per-process measurements only; quantiles are never averaged into a tail."""
    result = {k: receipt[k] for k in RESOURCE_KEYS}
    for key in ('elapsed_seconds', 'process_peak_rss_bytes', 'database_bytes'):
        if native._finite(result[key]) <= 0:
            raise ValueError('positive actual resource measurement required')
    for key in ('latency_seconds_p50_p95_p99_max', 'frame_latency_seconds_p50_p95_p99_max'):
        values = result[key]
        if len(values) != 4 or any(native._finite(v) < 0 for v in values) or values != sorted(values):
            raise ValueError('ordered measured p50/p95/p99/max required')
    if (result['latency_clock'] != 'time.monotonic' or result['latency_quantile_method'] != 'numpy_linear'
            or any(not isinstance(result[k], str) or not result[k] for k in ('latency_scope', 'frame_latency_scope'))
            or runtime.get('exclusive_host') is not False
            or runtime.get('same_latency_or_memory_verified') is not False):
        raise ValueError('non-exclusive measured resource scope required')
    result.update(exclusive_host=False, same_latency_or_memory_verified=False,
                  campaign_peak_memory_measured=False, deployment_tail_latency_verified=False)
    return result


def factor_fingerprint(path, rows, audited_sequences):
    """Same ordered triplet digest as the JPDA audit, independently read back.

    Per-sequence hashes must also match the independent MHT SQLite ledger audit.
    A digest of digests is NOT the global ordered raw-factor-stream fingerprint.
    """
    total = hashlib.sha256(); per_sequence = {}; counts = {}; seen = set()
    with Path(path).open('rb') as stream:
        for row, line in zip_longest(rows, stream):
            if row is None or line is None:
                raise ValueError('factor stream and scheduled frame counts differ')
            a = json.loads(line)['tracking']; key = (a['sequence_id'], a['event_id'])
            if key != (row['sequence_id'], row['vehicle_frame']) or key in seen:
                raise ValueError('factor stream order or event identity differs')
            h = a['factor_rows_sha256']
            if not isinstance(h, str) or re.fullmatch('[0-9a-f]{64}', h) is None:
                raise ValueError('actual raw-factor digest required')
            seen.add(key); block = native.canonical([*key, h]) + b'\n'
            total.update(block); per_sequence.setdefault(key[0], hashlib.sha256()).update(block)
            counts[key[0]] = counts.get(key[0], 0) + 1
    if (not seen or set(per_sequence) != set(audited_sequences)
            or any(value.hexdigest() != audited_sequences[s]['factor_stream_sha256']
                   or counts[s] != audited_sequences[s]['frames'] for s, value in per_sequence.items())):
        raise ValueError('raw-factor stream differs from independently audited sequence ledger')
    return total.hexdigest()


def load_mht_report(path, expected_sha):
    path = Path(path).absolute()
    if native.sha(path) != expected_sha:
        raise ValueError('MHT evaluation report identity differs')
    report = native.read_json(path)
    if report.get('kind') != mht.KIND or report.get('status') != 'complete':
        raise ValueError('complete full MHT evaluation required')
    bound = contract.inspect_audited_run(report['run_directory'], report['inference_receipt_sha256'],
        report['inference_audit_path'], report['inference_audit_sha256'])
    plan, receipt = bound['plan'], bound['receipt']
    for key, value in (('seed', plan['checkpoint_seed']), ('width', 4),
            ('plan_sha256', receipt['plan_sha256']), ('configuration', plan['configuration']),
            ('inference_runtime', plan['runtime']), ('cache_sha256', plan['cache_sha256']),
            ('schedule_sha256', plan['schedule_sha256']), ('checkpoint_sha256', plan['checkpoint_sha256']),
            ('scorer_signature', plan['scorer_signature']), ('inference_source_sha256', plan['source_sha256']),
            ('predictions_sha256', receipt['predictions_sha256'])):
        if report.get(key) != value:
            raise ValueError('MHT evaluation-to-inference binding differs: ' + key)
    protocol = mht.protocol(native.load_adapter())
    if (report.get('protocol') != protocol or report.get('protocol_sha256') != contract.ledger.digest(protocol)
            or report.get('ground_truth_manifest_sha256') != native.GT_MANIFEST_SHA256
            or report.get('ground_truth_sha256') != native.GT_SHA256
            or report.get('coverage', {}).get('frames') != 3316 or report['coverage'].get('sequences') != 21
            or report.get('reporting_scope') != 'car_only' or report.get('validation_already_seen') is not True
            or report.get('test_payloads_read') is not False or report.get('parameter_training') is not False):
        raise ValueError('full official MHT car evaluation protocol differs')
    if set(report.get('files', {})) != {'metrics.json', 'runtime.json', 'golden-cases.json'}:
        raise ValueError('complete metrics, runtime and golden-case artifacts required')
    evidence = dict(bound['evidence']); payloads = {}
    for name, record in report['files'].items():
        p = common.child(path.parent, name)
        if native.evidence(p) != record:
            raise ValueError('MHT evaluation artifact changed')
        evidence[str(p)] = record; payloads[name] = native.read_json(p)
    native.validate_runtime(payloads['runtime.json'])
    if (payloads['golden-cases.json'].get('passed') is not True
            or native.car_metrics(payloads['metrics.json']) != report['primary_car']):
        raise ValueError('MHT summary differs from native metrics or golden cases failed')
    expected_inputs = dict(bound['evidence'])
    for p in (Path(mht.__file__), native.ADAPTER_PATH,
              common.child(report['ground_truth_directory'], 'manifest.json'),
              common.child(report['ground_truth_directory'], 'ground-truth.jsonl')):
        expected_inputs[str(p.absolute())] = native.evidence(p)
    if report.get('input_evidence') != expected_inputs:
        raise ValueError('MHT evaluation input inventory differs')
    factor = factor_fingerprint(common.child(bound['run'], 'tracking.jsonl'), bound['rows'], bound['audit']['sequences'])
    evidence.update(expected_inputs); evidence[str(path)] = native.evidence(path)
    common.unchanged(evidence)
    return dict(report=report, primary=baseline.primary_vector(report['primary_car']),
        sequences=native.sequence_vectors(payloads['metrics.json']), evaluator_runtime=payloads['runtime.json'],
        report_sha256=expected_sha, evidence=evidence, factor_stream_sha256=factor,
        resources=resources(receipt, plan['runtime']))


def load_probabilistic_report(path, expected_sha):
    cell = baseline.load_report(path, expected_sha)
    p = common.child(cell['report']['run_directory'], 'receipt.json')
    if cell['evidence'].get(str(p)) != native.evidence(p):
        raise ValueError('unbound probabilistic resource receipt')
    cell['resources'] = resources(native.read_json(p), cell['report']['inference_runtime'])
    cell['factor_stream_sha256'] = cell['report']['factor_stream_sha256']
    common.unchanged(cell['evidence'])
    return cell


def assemble(mht_campaign, probabilistic_campaign, mht_cells, probabilistic_cells):
    """Pure aggregation; file/SQLite/metric validation is required by loaders."""
    if len(mht_cells) != 3 or {c['report']['seed'] for c in mht_cells} != set(SEEDS):
        raise ValueError('all three distinct MHT seeds required; no failed-cell omission')
    old = baseline.assemble(probabilistic_campaign, probabilistic_cells)
    pmap = {(c['report']['seed'], c['report']['update_rule']): c for c in probabilistic_cells}
    mmap = {c['report']['seed']: c for c in mht_cells}
    jobs = mht_campaign.get('jobs', [])
    cps = mht_campaign.get('checkpoints', [])
    if (len(jobs) != 3 or {j['seed'] for j in jobs} != set(SEEDS)
            or len(cps) != 3 or {c['seed'] for c in cps} != set(SEEDS)
            or contract.ledger.digest(mht_campaign.get('configuration')) != contract.CONFIG_SHA):
        raise ValueError('complete frozen K=4 MHT campaign required')
    jobs, cps = {j['seed']: j for j in jobs}, {c['seed']: c for c in cps}
    pcps = {c['seed']: c for c in probabilistic_campaign['checkpoints']}
    ms, ps = mht_campaign['source_sha256'], probabilistic_campaign['source_sha256']
    if set(ms) - set(ps) != EXTRA_SOURCES or any(ms.get(k) != v for k, v in ps.items()):
        raise ValueError('shared inference sources differ between MHT and JPDA/PKF')
    reference_runtime = {k: v for k, v in mht_campaign['runtime'].items() if k != 'pid'}
    for seed in SEEDS:
        cell, pcell = mmap[seed], pmap[(seed, 'jpda-ci')]
        r, p, job, cp = cell['report'], pcell['report'], jobs[seed], cps[seed]
        a = job['arguments']
        if (any(cp.get(k) != pcps[seed].get(k) for k in ('sha256', 'model_sha256', 'scorer_signature'))
                or r['checkpoint_sha256'] != cp['sha256'] or r['scorer_signature'] != cp['scorer_signature']
                or r['configuration'] != mht_campaign['configuration'] or r['width'] != 4 or job['width'] != 4
                or a['width'] != 4 or a['checkpoint_sha256'] != cp['sha256'] or a['device'] != 'cpu'
                or r['run_directory'] != job['output'] or a['output'] != job['output']
                or any(a.get(k) != r['configuration'][k] for k in r['configuration']
                       if k.startswith('max_assignment_') or k == 'max_generated_candidates')
                or r['inference_source_sha256'] != ms
                or any(r[k] != p[k] or r[k] != a[k] for k in ('cache_sha256', 'schedule_sha256'))
                or r['configuration']['state'] != p['configuration']['state']):
            raise ValueError('MHT result differs from frozen campaign or common input/model')
        runtime = {k: v for k, v in r['inference_runtime'].items() if k != 'pid'}
        if (runtime != reference_runtime
                or {k: v for k, v in runtime.items() if k not in EXTRA_RUNTIME} !=
                   {k: v for k, v in p['inference_runtime'].items() if k != 'pid'}
                or cell['evaluator_runtime'] != pcell['evaluator_runtime']):
            raise ValueError('shared inference or native evaluation runtime differs')
        if (any(r[k] != p[k] for k in ('ground_truth_sha256', 'ground_truth_manifest_sha256'))
                or {k: v for k, v in r['protocol'].items() if k != 'kind'} !=
                   {k: v for k, v in p['protocol'].items() if k != 'kind'}
                or cell['factor_stream_sha256'] != p['factor_stream_sha256']):
            raise ValueError('native metric protocol or actual raw factors differ')
        for flag in ('paper_eligible', 'fair_resources_verified', 'reproduced_public_method',
                     'validation_checkpoint_selection', 'validation_parameter_search',
                     'same_input_selection_as_legacy_source_ablation'):
            if r.get(flag) is not False:
                raise ValueError('MHT adaptation or selection boundary changed')
    mapping = {**pmap, **{(s, MHT): c for s, c in mmap.items()}}
    sequence_ids = set(pmap[(SEEDS[0], 'jpda-ci')]['sequences'])
    for cell in mapping.values():
        if (set(cell['primary']) != set(baseline.METRICS) or len(sequence_ids) != 21
                or set(cell['sequences']) != sequence_ids):
            raise ValueError('complete primary and 21-sequence metrics required')
        for value in cell['primary'].values(): native._finite(value)
        for vector in cell['sequences'].values():
            if set(vector) != {'HOTA', 'AssA', 'DetA', 'IDF1'}:
                raise ValueError('complete per-sequence metrics required')
            for value in vector.values(): native._finite(value)
        resources(cell['resources'], cell['report']['inference_runtime'])
    descriptive = dict(old['three_seed_descriptive']); descriptive[MHT] = {}
    for metric in baseline.METRICS:
        values = {str(s): mmap[s]['primary'][metric] for s in SEEDS}
        descriptive[MHT][metric] = dict(mean=statistics.fmean(values.values()),
            sample_sd=statistics.stdev(values.values()), minimum=min(values.values()), maximum=max(values.values()), by_seed=values)
    deltas = {rule: {str(s): dict(
        primary={m: mmap[s]['primary'][m] - pmap[(s, rule)]['primary'][m] for m in baseline.METRICS},
        per_sequence={sid: {m: mmap[s]['sequences'][sid][m] - pmap[(s, rule)]['sequences'][sid][m]
            for m in ('HOTA', 'AssA', 'DetA', 'IDF1')} for sid in sorted(sequence_ids)}) for s in SEEDS} for rule in RULES}
    return dict(kind='fixed_mht_probabilistic_three_seed_full_val_comparison_v1', status='complete',
        completed_cells=12, frames_per_run=3316, sequences_per_run=21,
        three_seed_descriptive=descriptive, paired_mht_minus_baseline=deltas,
        per_run={f'{s}/{method}': dict(primary=cell['primary'], resources=cell['resources'],
            raw_factor_stream_sha256=cell['factor_stream_sha256']) for (s, method), cell in sorted(mapping.items())},
        metric_direction=old['metric_direction'], actual_factors_identical_within_each_seed=True,
        best_seed_selected=False, sample_sd_is_not_confidence_interval=True,
        sequences_are_not_independent_seed_replicates=True, primary_is_not_mean_of_sequence_metrics=True,
        method_semantics={MHT: 'irreversible_global_history_K4_separate_branch_states',
            **{r: 'single_committed_identity_history_collapsed_scan_states' for r in RULES}},
        same_state_time_protocol_verified=False, recovery_only_causal_effect_verified=False,
        deployment_tail_latency_verified=False, fair_resources_verified=False,
        strong_single_endpoint_controls_included=False, recoverable_method_included=False,
        public_methods_reproduced=False, full_paper_comparison_completed=False, paper_eligible=False,
        validation_already_seen=True, test_payloads_read=False, parameter_training=False)


def compare(mht_campaign_path, mht_sha, probabilistic_campaign_path, probabilistic_sha,
            mht_reports, probabilistic_reports, output):
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new non-symlink comparison output required')
    if (len(mht_reports) != 3 or len(probabilistic_reports) != 9
            or len({str(Path(p).absolute()) for p, _ in [*mht_reports, *probabilistic_reports]}) != 12):
        raise ValueError('exactly three MHT and nine distinct probabilistic reports required')
    campaigns = []; evidence = {}
    for path, expected in ((mht_campaign_path, mht_sha), (probabilistic_campaign_path, probabilistic_sha)):
        path = Path(path).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)) or native.sha(path) != expected:
            raise ValueError('frozen ordinary campaign file required')
        evidence[str(path)] = native.evidence(path); campaigns.append(native.read_json(path))
    mc, pc = campaigns
    if (mc.get('kind') != 'fixed_scan_mht_three_seed_full_val_campaign_v1'
            or pc.get('kind') != 'fixed_probabilistic_full_spd_val_campaign_preflight_v1'
            or mc.get('source_sha256') != contract.ledger.inference_sources()
            or pc.get('source_sha256') != common.inference_sources()):
        raise ValueError('current frozen MHT and probabilistic campaigns required')
    for path, expected in mc['contract_source_sha256'].items():
        p = common.child(ROOT, path); item = native.evidence(p)
        if item['sha256'] != expected: raise ValueError('frozen MHT evaluation contract changed')
        evidence[str(p)] = item
    for path, expected in mc['input_file_sha256'].items():
        item = native.evidence(path)
        if item['sha256'] != expected: raise ValueError('frozen MHT campaign input changed')
        evidence[path] = item
    m_cells = [load_mht_report(p, h) for p, h in mht_reports]
    p_cells = [load_probabilistic_report(p, h) for p, h in probabilistic_reports]
    result = assemble(mc, pc, m_cells, p_cells)
    for p in (Path(__file__), Path(baseline.__file__)):
        evidence[str(p)] = native.evidence(p)
    for cell in [*m_cells, *p_cells]:
        for p, value in cell['evidence'].items():
            if p in evidence and evidence[p] != value: raise ValueError('conflicting shared evidence')
            evidence[p] = value
    common.unchanged(evidence)
    result.update(mht_campaign_sha256=mht_sha, probabilistic_campaign_sha256=probabilistic_sha,
        report_sha256={f'{c["report"]["seed"]}/{method}': c['report_sha256']
            for method, c in [(MHT, c) for c in m_cells] + [(c['report']['update_rule'], c) for c in p_cells]},
        input_evidence=evidence)
    output.mkdir(); native.write_json(output / 'comparison.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'completed_cells', 'three_seed_descriptive', 'paper_eligible')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ('mht', 'probabilistic'):
        parser.add_argument('--' + prefix + '-campaign', type=Path, required=True)
        parser.add_argument('--' + prefix + '-campaign-sha256', required=True)
        parser.add_argument('--' + prefix + '-report', nargs=2, action='append', required=True, metavar=('REPORT', 'SHA256'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.mht_campaign, args.mht_campaign_sha256, args.probabilistic_campaign,
            args.probabilistic_campaign_sha256, args.mht_report, args.probabilistic_report, args.output)
