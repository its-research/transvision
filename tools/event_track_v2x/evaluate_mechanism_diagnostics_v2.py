#!/usr/bin/env python3
"""Create-once car-only evaluation and paired-seed mechanism comparisons.

The frozen source-ablation evaluator supplies evaluate(), not compare(). This
controller changes neither its metric protocol nor any prediction. M3 changes
residual scores used by temporal assignment, state survival and births; it is
not a pure birth intervention. No training, tuning or external publication.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE_EVALUATOR = ROOT / 'tools/event_track_v2x/evaluate_source_ablation_v2.py'
SOURCE_EVALUATOR_SHA = 'a9e80719682477c3f1a4ab6030d906d20f848d2584578e3fb8dfe45c6f399356'
REFERENCE_COMPARISON_SHA = 'c6b200f363a025674968b0947a32562f5d253677a05331c1ccc5e42d0e0f6d16'
PREDICTION_SOURCE_PINS = {
    'tools/event_track_v2x/run_mechanism_diagnostics_v2.py': '348d1e26eb810120fceafa6df8292b0de444ff86a1f1ec12d67adaf13c03c283',
    'transvision/models/event_track_v2x/tracking_mechanisms_v2.py': 'd9063684f54aa3f6bd083a13a05a4c18bdbf23f49855b387930cc89acb1adcf6',
}
SEEDS = (1337, 2027, 3407)
RUNS = ([{'run_id': f'M0-seed-{s}', 'mode': 'M0', 'seed': s, 'deterministic_control': False} for s in SEEDS]
        + [{'run_id': 'M1-all-unmatched', 'mode': 'M1', 'seed': 1337, 'deterministic_control': True}]
        + [{'run_id': f'{m}-seed-{s}', 'mode': m, 'seed': s, 'deterministic_control': False}
           for m in ('M2', 'M3') for s in SEEDS])
METRICS = {'AMOTA': ('nuscenes', 'amota'), 'AMOTP_m': ('nuscenes', 'amotp'),
           'MOTA': ('nuscenes', 'mota'), 'HOTA': ('trackeval', 'HOTA'),
           'AssA': ('trackeval', 'AssA'), 'DetA': ('trackeval', 'DetA'),
           'IDF1': ('trackeval', 'IDF1'), 'IDS': ('nuscenes', 'ids'),
           'Frag': ('nuscenes', 'frag'), 'FP': ('nuscenes', 'fp'), 'FN': ('nuscenes', 'fn')}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evidence(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('regular non-symlink evidence file required: ' + str(path))
    return {'sha256': sha(path), 'size': path.stat().st_size}


def read_json(path):
    evidence(path)
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value) + b'\n')


def load_evaluator():
    if evidence(SOURCE_EVALUATOR)['sha256'] != SOURCE_EVALUATOR_SHA:
        raise ValueError('frozen source evaluator changed')
    spec = importlib.util.spec_from_file_location('sealed_mechanism_car_evaluator', SOURCE_EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('missing/non-finite metric cannot be converted to zero')
    return value


def metric_vector(primary):
    return {k: finite(primary[group][name]) for k, (group, name) in METRICS.items()}


def descriptives(vectors):
    n = len(vectors)
    if not n:
        raise ValueError('at least one observation required')
    return {k: {'mean': statistics.fmean(v[k] for v in vectors),
                'sample_sd': statistics.stdev(v[k] for v in vectors) if n > 1 else None,
                'n': n} for k in METRICS}


def close_vectors(actual, expected):
    if set(actual) != set(expected) or any(not math.isclose(finite(actual[k]), finite(expected[k]),
                                                          abs_tol=1e-12, rel_tol=0) for k in actual):
        raise ValueError('M0/reference metric parity failed')


def validate_experiment(root):
    root = Path(root)
    plan, summary = read_json(root/'frozen-plan.json'), read_json(root/'summary.json')
    if (plan['kind'] != 'mechanism_diagnostics_v2_plan' or plan['runs'] != RUNS
            or plan['reporting_scope'] != 'car_only' or plan['paper_eligible'] is not False
            or plan['gt_model_inputs'] is not False or plan['test_payloads_read'] is not False
            or plan['val_parameter_fitting'] is not False or plan['seed_selection'] is not False
            or plan['official_validation_frames'] != 3316 or plan['sequences'] != 21
            or summary['kind'] != 'mechanism_diagnostics_v2_complete' or summary['status'] != 'completed'
            or summary['reporting_scope'] != 'car_only' or summary['paper_eligible'] is not False
            or summary['weights_unchanged'] is not True or summary['all_required_sealed_streams_match'] is not True
            or summary['plan_sha256'] != sha(root/'frozen-plan.json')
            or set(summary['runs']) != {r['run_id'] for r in RUNS}):
        raise ValueError('incomplete or incompatible ten-run mechanism experiment')
    for name, wanted in PREDICTION_SOURCE_PINS.items():
        if plan['current_source_hashes'].get(name) != wanted or evidence(ROOT/name)['sha256'] != wanted:
            raise ValueError('mechanism prediction source pin mismatch')
    for name, wanted in plan['current_source_hashes'].items():
        if evidence(ROOT/name)['sha256'] != wanted:
            raise ValueError('prediction dependency changed: ' + name)
    source_files = {'frozen-plan.json': evidence(root/'frozen-plan.json'), 'summary.json': evidence(root/'summary.json')}
    receipts = {}
    for spec in RUNS:
        name = spec['run_id']; folder = root/name
        receipt = read_json(folder/'receipt.json')
        if (receipt != summary['runs'][name] or any(receipt[k] != v for k, v in spec.items())
                or receipt['frames'] != 3316 or receipt['sequences'] != 21
                or len(receipt['sequence_commits']) != 21 or len(receipt['diagnostic_sequence_commits']) != 21
                or receipt['weights_unchanged'] is not True
                or (spec['mode'] != 'M1' and receipt['sealed_association_parity'] is not True)
                or (spec['mode'] == 'M0' and receipt['sealed_prediction_parity'] is not True)):
            raise ValueError('runner receipt mismatch: ' + name)
        files = {f: evidence(folder/f) for f in ('predictions.jsonl', 'association.jsonl', 'diagnostics.jsonl.gz', 'receipt.json')}
        for key in ('predictions', 'association'):
            if files[key+'.jsonl']['sha256'] != receipt[key+'_sha256']:
                raise ValueError('runner stream hash mismatch: ' + name)
        if (files['diagnostics.jsonl.gz']['sha256'] != receipt['diagnostics_gzip_sha256']
                or files['diagnostics.jsonl.gz']['size'] != receipt['diagnostics_gzip_bytes']):
            raise ValueError('diagnostic stream hash/size mismatch: ' + name)
        receipts[name] = receipt
        source_files[name] = files
    return plan, receipts, source_files


def validate_report(folder, evaluator, receipt=None):
    folder = Path(folder)
    report = read_json(folder/'report.json')
    adapter = evaluator.load_adapter()
    protocol = evaluator.protocol(adapter)
    if (report['kind'] != 'source_ablation_car_evaluation_v1' or report['status'] != 'completed'
            or report['reporting_scope'] != 'car_only' or report['protocol'] != protocol
            or report['protocol_sha256'] != hashlib.sha256(canonical(protocol)).hexdigest()
            or report['sealed_protocol_sha256'] != hashlib.sha256(canonical(adapter.PROTOCOL)).hexdigest()
            or report['ground_truth_manifest_sha256'] != evaluator.GT_MANIFEST_SHA256
            or report['ground_truth_sha256'] != evaluator.GT_SHA256
            or report['paper_eligible'] is not False or report['no_model_selection'] is not True
            or report['no_test_payload'] is not True or report['no_validation_parameter_fitting'] is not True
            or report['coverage']['frames'] != 3316 or report['coverage']['sequences'] != 21
            or report['input_sources']['evaluation_cli'] != evidence(SOURCE_EVALUATOR)
            or report['input_sources']['sealed_adapter'] != evidence(evaluator.ADAPTER_PATH)
            or report['input_sources']['predictions']['sha256'] != report['predictions_sha256']):
        raise ValueError('unsealed or incomplete car report: ' + str(folder))
    if receipt is not None and (report['predictions_sha256'] != receipt['predictions_sha256']
            or report['coverage']['final_sequence_commit_sha256'] != receipt['sequence_commits']):
        raise ValueError('report not bound to full prediction stream')
    if set(report['files']) != {'metrics.json', 'runtime.json', 'golden-cases.json'}:
        raise ValueError('unexpected evaluation artifact inventory')
    files = {name: evidence(folder/name) for name in report['files']}
    if files != report['files']:
        raise ValueError('evaluation artifact hash/size mismatch')
    evaluator.validate_runtime(read_json(folder/'runtime.json'))
    if read_json(folder/'golden-cases.json').get('passed') is not True:
        raise ValueError('golden car cases failed')
    metrics = read_json(folder/'metrics.json')
    if evaluator.car_metrics(metrics) != report['primary_car']:
        raise ValueError('reported metrics differ from official engine artifacts')
    sequences = evaluator.sequence_vectors(metrics)
    if set(sequences) != set(report['coverage']['final_sequence_commit_sha256']):
        raise ValueError('per-sequence coverage differs')
    files['report.json'] = evidence(folder/'report.json')
    return report, metric_vector(report['primary_car']), sequences, files


def validate_reference(path, evaluator):
    path = Path(path)
    if evidence(path)['sha256'] != REFERENCE_COMPARISON_SHA:
        raise ValueError('sealed source-comparison reference hash mismatch')
    comparison = read_json(path)
    manifest_path = path.parent/'source-manifest.json'
    manifest = read_json(manifest_path)
    if (comparison['kind'] != 'source_ablation_car_comparison_v1' or comparison['status'] != 'completed'
            or comparison['reporting_scope'] != 'car_only' or comparison['paper_eligible'] is not False
            or comparison['source_manifest_sha256'] != sha(manifest_path)):
        raise ValueError('invalid frozen source-comparison reference')
    sources = {'comparison': evidence(path), 'source_manifest': evidence(manifest_path), 'runs': {}}
    reference = {}
    for name in ['vehicle-only', 'infrastructure-only'] + [f'cooperative-seed-{s}' for s in SEEDS]:
        report, vector, sequences, files = validate_report(path.parent.parent/name, evaluator)
        if (files != manifest['runs'][name]['evaluation_files']
                or report['predictions_sha256'] != manifest['runs'][name]['runner_files']['predictions.jsonl']['sha256']):
            raise ValueError('source reference not bound to original runner/report artifacts')
        close_vectors({k: vector[k] for k in evaluator.PRIMARY_METRICS}, comparison['run_metrics'][name])
        reference[name] = {'report': report, 'metrics': vector, 'sequences': sequences,
                           'association_sha256': manifest['runs'][name]['runner_files']['association.jsonl']['sha256']}
        sources['runs'][name] = files
    return reference, sources


def contrast(current, baseline, current_sequences, baseline_sequences):
    if set(current_sequences) != set(baseline_sequences):
        raise ValueError('paired sequence cohorts differ')
    per_sequence = {sid: {k: current_sequences[sid][k] - baseline_sequences[sid][k]
                         for k in ('HOTA', 'AssA', 'DetA', 'IDF1')} for sid in sorted(current_sequences)}
    return {'metric_difference': {k: current[k] - baseline[k] for k in METRICS},
            'paired_sequence_differences': per_sequence,
            'paired_sequence_sign_counts': {k: {'positive': sum(v[k] > 0 for v in per_sequence.values()),
                                                'zero': sum(v[k] == 0 for v in per_sequence.values()),
                                                'negative': sum(v[k] < 0 for v in per_sequence.values())}
                                            for k in ('HOTA', 'AssA', 'DetA', 'IDF1')}}


def compare(root, reports_root, reference_source_comparison, output):
    root, reports_root, output = Path(root), Path(reports_root), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('comparison output is create-once')
    evaluator = load_evaluator()
    _, receipts, runner_sources = validate_experiment(root)
    reference, reference_sources = validate_reference(reference_source_comparison, evaluator)
    vectors, sequences, report_sources, parity, lifecycle = {}, {}, {}, {}, {}
    for spec in RUNS:
        name = spec['run_id']; receipt = receipts[name]
        report, vectors[name], sequences[name], report_sources[name] = validate_report(reports_root/name, evaluator, receipt)
        if report['input_sources']['predictions'] != runner_sources[name]['predictions.jsonl']:
            raise ValueError('prediction evidence size/hash differs')
        old = reference[f"cooperative-seed-{spec['seed']}"]
        if spec['mode'] != 'M1' and receipt['association_sha256'] != old['association_sha256']:
            raise ValueError('frozen association parity failed')
        if spec['mode'] == 'M0':
            if receipt['predictions_sha256'] != old['report']['predictions_sha256']:
                raise ValueError('M0 sealed prediction parity failed')
            close_vectors(vectors[name], old['metrics'])
            if sequences[name] != old['sequences']:
                raise ValueError('M0 sealed per-sequence metrics differ')
            parity[name] = True
        lifecycle[name] = {'events': receipt['events'], 'nodes': receipt['nodes'],
                           'temporal_matches': receipt['temporal_matches'], 'predictions': receipt['predictions'],
                           'scope': 'engineering counters over unchanged all-class candidates, not car metrics'}
    by_mode = {mode: descriptives([vectors[r['run_id']] for r in RUNS if r['mode'] == mode])
               for mode in ('M0', 'M1', 'M2', 'M3')}
    paired = {mode: {str(seed): contrast(vectors[f'{mode}-seed-{seed}'], vectors[f'M0-seed-{seed}'],
                                         sequences[f'{mode}-seed-{seed}'], sequences[f'M0-seed-{seed}'])
                     for seed in SEEDS} for mode in ('M2', 'M3')}
    paired_descriptive = {mode: descriptives([v['metric_difference'] for v in values.values()])
                          for mode, values in paired.items()}
    m1 = {str(seed): contrast(vectors['M1-all-unmatched'], vectors[f'M0-seed-{seed}'],
                              sequences['M1-all-unmatched'], sequences[f'M0-seed-{seed}']) for seed in SEEDS}
    against_road = {name: contrast(vector, reference['infrastructure-only']['metrics'],
                                   sequences[name], reference['infrastructure-only']['sequences'])
                    for name, vector in vectors.items()}
    sources = {'prediction_experiment': runner_sources, 'evaluation_reports': report_sources,
               'reference_source_ablation': reference_sources, 'comparison_cli': evidence(__file__),
               'source_evaluator': evidence(SOURCE_EVALUATOR)}
    result = {'kind': 'mechanism_diagnostics_car_comparison_v1', 'status': 'completed',
              'reporting_scope': 'car_only', 'coverage_per_run': {'frames': 3316, 'sequences': 21},
              'run_metrics': vectors, 'mode_descriptive': by_mode,
              'M2_M3_minus_same_seed_M0': paired, 'paired_difference_descriptive': paired_descriptive,
              'M1_single_run_minus_each_M0': m1, 'minus_infrastructure_only': against_road,
              'M0_matches_sealed_reference': parity, 'lifecycle_counters': lifecycle,
              'metric_direction': {k: 'lower_is_better' if k in ('AMOTP_m', 'IDS', 'Frag', 'FP', 'FN')
                                   else 'higher_is_better' for k in METRICS},
              'M1_repetitions': 1, 'M1_seed_interpretation': 'one deterministic control; comparisons reuse it, not three independent repeats',
              'M3_interpretation': 'restored residual-node score affects temporal costs, updates, pruning and births; not a birth-only treatment',
              'diagnostic_boundary': 'receipt counters only; detailed control-chain and causal event audit remains separate',
              'inference_boundary': 'exploratory descriptive validation; sample SD is not CI; no best-seed selection, tuning, p-values or causal proof',
              'difference_units': 'absolute metric units, not percentage or relative improvement',
              'independent_diagnostic_chain_audit_required': True, 'paper_eligible': False,
              'source_manifest_sha256': hashlib.sha256(canonical(sources)+b'\n').hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'source-manifest.json', sources)
    write_json(output/'comparison.json', result)
    return result


def worker(args):
    evaluator = load_evaluator()
    evaluator.evaluate(args.ground_truth, args.predictions, args.output)


def run(args):
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('evaluation output is create-once; inspect a previous result before retry')
    if type(args.max_workers) is not int or not 1 <= args.max_workers <= 3:
        raise ValueError('max-workers must be 1, 2 or 3')
    evaluator = load_evaluator()
    _, _, runner_sources = validate_experiment(args.root)
    _, reference_sources = validate_reference(args.reference_source_comparison, evaluator)
    gt_sources = {name: evidence(args.ground_truth/name) for name in ('manifest.json', 'ground-truth.jsonl')}
    if (gt_sources['manifest.json']['sha256'] != evaluator.GT_MANIFEST_SHA256
            or gt_sources['ground-truth.jsonl']['sha256'] != evaluator.GT_SHA256):
        raise ValueError('not the sealed official-validation ground truth')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    commands = {r['run_id']: [sys.executable, str(Path(__file__).resolve()), 'evaluate-one',
                '--ground-truth', str(args.ground_truth.resolve()), '--predictions',
                str((args.root/r['run_id']/'predictions.jsonl').resolve()), '--output',
                str((args.output/r['run_id']).resolve())] for r in RUNS}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'logs').mkdir()
    frozen = {'kind': 'mechanism_car_evaluation_execution_plan_v1', 'runs': RUNS,
              'runner_sources': runner_sources, 'reference_sources': reference_sources, 'ground_truth': gt_sources,
              'commands': commands, 'max_workers': args.max_workers, 'threads_per_worker': 1,
              'controller': evidence(__file__), 'source_evaluator': evidence(SOURCE_EVALUATOR),
              'reporting_scope': 'car_only', 'gpu_used': False, 'paper_eligible': False}
    write_json(args.output/'frozen-evaluation-plan.json', frozen)
    started = time.monotonic()
    def launch(name):
        with (args.output/'logs'/f'{name}.log').open('xb') as stream:
            status = subprocess.run(commands[name], env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    check=False).returncode
        return name, status
    codes = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(launch, r['run_id']) for r in RUNS]
        for future in as_completed(futures):
            name, status = future.result(); codes[name] = status
            print('MECHANISM_EVALUATED '+json.dumps({'run_id': name, 'exit_code': status}), flush=True)
    execution = {'kind': 'mechanism_car_evaluation_execution_v1', 'exit_codes': codes,
                 'elapsed_seconds': time.monotonic()-started,
                 'plan_sha256': sha(args.output/'frozen-evaluation-plan.json'),
                 'status': 'completed' if all(v == 0 for v in codes.values()) else 'failed'}
    write_json(args.output/'execution.json', execution)
    if any(codes.values()):
        raise RuntimeError('one or more car evaluations failed; partial reports and logs retained')
    if evidence(__file__) != frozen['controller'] or evidence(SOURCE_EVALUATOR) != frozen['source_evaluator']:
        raise ValueError('evaluation source changed during execution')
    compare(args.root, args.output, args.reference_source_comparison, args.output/'comparison')
    print('MECHANISM_EVALUATION_COMPLETE '+json.dumps({'runs': 10,
        'comparison_sha256': sha(args.output/'comparison'/'comparison.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    for key in ('root', 'ground-truth', 'reference-source-comparison', 'output'):
        run_parser.add_argument('--'+key, type=Path, required=True)
    run_parser.add_argument('--max-workers', type=int, choices=(1, 2, 3), default=3)
    compare_parser = sub.add_parser('compare')
    for key in ('root', 'reports-root', 'reference-source-comparison', 'output'):
        compare_parser.add_argument('--'+key, type=Path, required=True)
    single = sub.add_parser('evaluate-one')
    for key in ('ground-truth', 'predictions', 'output'):
        single.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    elif args.command == 'compare':
        compare(args.root, args.reports_root, args.reference_source_comparison, args.output)
    else:
        worker(args)


if __name__ == '__main__':
    main()
