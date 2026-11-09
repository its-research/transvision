#!/usr/bin/env python3
"""Independent arithmetic of preserved Stage 1 sequence sufficient statistics.

This is not a detector/matching engine rerun. Failed deterministic executions
stay excluded from qualified paired comparisons; all raw seed values survive.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS = (1337, 2027, 3407)
METRICS = ('HOTA', 'AssA', 'DetA', 'IDF1')
COUNTS = ('HOTA_TP', 'HOTA_FN', 'HOTA_FP')
AUTHOR = {'hota.py': 'bf85a9e4f2270fb3a52f94b05382250f4b8804a39a897ea9dfbf97ff85bf331e',
          'identity.py': '33b4ef7e11a24c47c6060eb2464cb92167aea42e8c0e6f652b10ca317872028c',
          '_base_metric.py': 'dac8b87bf3801269aa280b1cf91184f483de4f4e5c158bd07a4b1733b1a5722b'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path, digest=None):
    p = Path(path).absolute()
    if not p.is_file() or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('ordinary frozen evidence required')
    if digest is not None and sha(p) != digest:
        raise ValueError('evidence SHA differs')
    return json.loads(p.read_bytes())


def new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def sufficient(sequences, order):
    """Validate counts and produce weighted numerators, retaining empty tails."""
    if set(sequences) != set(order) or len(order) != len(set(order)):
        raise ValueError('complete unique paired sequence cohort required')
    arrays = {k: [] for k in (*COUNTS, 'AssA', 'AssPr', 'AssRe', 'LocA', 'IDTP', 'IDFN', 'IDFP')}
    for sid in order:
        row = sequences[sid]; h = row['HOTA']; identity = row['Identity']
        for field in COUNTS:
            v = np.asarray(h[field], dtype=np.float64)
            if v.shape != (19,) or not np.isfinite(v).all() or np.any(v < 0) or np.any(v != np.floor(v)):
                raise ValueError('invalid native 19-alpha integer counts')
            arrays[field].append(v)
        tp, fn, fp = (np.asarray(h[k], dtype=np.float64) for k in COUNTS)
        if np.any(tp + fn != row['gt_objects']) or np.any(tp + fp != row['predicted_objects']):
            raise ValueError('sequence count conservation differs')
        for field in ('AssA', 'AssPr', 'AssRe', 'LocA'):
            v = np.asarray(h[field], dtype=np.float64)
            if v.shape != (19,) or not np.isfinite(v).all() or np.any(v < 0) or np.any(v > 1 + 1e-12):
                raise ValueError('invalid sequence association/localization statistics')
            arrays[field].append(v * tp)
        for field in ('IDTP', 'IDFN', 'IDFP'):
            v = identity[field]
            if not np.isfinite(v) or v < 0 or v != int(v):
                raise ValueError('invalid identity counts')
            arrays[field].append(v)
        if identity['IDTP'] + identity['IDFN'] != row['gt_objects'] or identity['IDTP'] + identity['IDFP'] != row['predicted_objects']:
            raise ValueError('identity count conservation differs')
    return {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}


def aggregate(data, multiplicities):
    """Official sequence combination, with independent vectorized arithmetic."""
    weights = np.asarray(multiplicities, dtype=np.float64)
    n = data['IDTP'].shape[0]
    if weights.ndim != 2 or weights.shape[1] != n or not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights != np.floor(weights)) or np.any(weights.sum(1) == 0):
        raise ValueError('nonempty integer sequence multiplicities required')
    result = {field: weights @ data[field] for field in COUNTS}
    tp, fn, fp = (result[k] for k in COUNTS)
    for field in ('AssA', 'AssPr', 'AssRe'):
        result[field] = (weights @ data[field]) / np.maximum(1., tp)
    result['LocA'] = np.maximum(1e-10, weights @ data['LocA']) / np.maximum(1e-10, tp)
    result['DetRe'] = tp / np.maximum(1., tp + fn)
    result['DetPr'] = tp / np.maximum(1., tp + fp)
    result['DetA'] = tp / np.maximum(1., tp + fn + fp)
    result['HOTA'] = np.sqrt(result['DetA'] * result['AssA'])
    result['OWTA'] = np.sqrt(result['DetRe'] * result['AssA'])
    result['HOTA(0)'] = result['HOTA'][:, 0]
    result['LocA(0)'] = result['LocA'][:, 0]
    result['HOTALocA(0)'] = result['HOTA(0)'] * result['LocA(0)']
    for field in ('IDTP', 'IDFN', 'IDFP'):
        result[field] = weights @ data[field]
    itp, ifn, ifp = (result[k] for k in ('IDTP', 'IDFN', 'IDFP'))
    result['IDR'] = itp / np.maximum(1., itp + ifn)
    result['IDP'] = itp / np.maximum(1., itp + ifp)
    result['IDF1'] = itp / np.maximum(1., itp + .5 * ifn + .5 * ifp)
    return result


def draws(sequence_count):
    """One explicit PCG64 stream shared by every seed/model/comparison."""
    if sequence_count < 1:
        raise ValueError('nonempty sequence cohort required')
    indices = np.random.Generator(np.random.PCG64(1337)).integers(0, sequence_count, size=(10000, sequence_count), dtype=np.int64)
    weights = np.stack([np.bincount(x, minlength=sequence_count) for x in indices])
    return indices, weights


def metric_values(result):
    return np.column_stack([result[k].mean(1) if k != 'IDF1' else result[k] for k in METRICS])


def interval(values):
    low, high = np.quantile(values, [.025, .975], axis=0, method='linear')
    return {k: dict(low=float(a), high=float(b)) for k, a, b in zip(METRICS, low, high)}


def descriptive(rows):
    keys = set(rows[0])
    if any(set(x) != keys for x in rows):
        raise ValueError('complete metric fields required; no missing-field zeros')
    return {k: dict(mean=float(np.mean([x[k] for x in rows])),
                    sample_std=float(np.std([x[k] for x in rows], ddof=1)) if len(rows) > 1 else None)
            for k in sorted(keys)}


def execute(args):
    root = Path(args.root).absolute(); output = Path(args.output).absolute()
    if output.exists() or any(x.is_symlink() for x in (output, *output.parents)):
        raise ValueError('fresh output directory required')
    metrics_root = root / 'metric-original-readback-v2'
    readback = read(metrics_root / 'byte-readback-receipt.json', args.readback_sha256)
    provenance = read(root / 'independent-full-provenance/provenance-receipt.json', args.provenance_sha256)
    if readback['complete_actual_metric_files_read_back'] is not True or readback['run_count'] != 13 or provenance['complete_original_cohort_scanned'] is not True or provenance['all_internal_output_provenance_passed'] is not True:
        raise ValueError('complete original metric bytes and output provenance required')
    manifest = read(metrics_root / 'original-byte-manifest.json', readback['manifest_sha256'])
    for name, record in manifest.items():
        if sha(metrics_root / name) != record['sha256'] or (metrics_root / name).stat().st_size != record['bytes']:
            raise ValueError('original metric/author source bytes changed')
    for name, digest in AUTHOR.items():
        if manifest['author-trackeval/' + name]['sha256'] != digest:
            raise ValueError('original author aggregation source differs')
    parity = {x['run_id']: x['all_three_streams_byte_identical'] for x in provenance['strict_replay_byte_parity']}
    expected = {f'{m}-seed-{s}' for m in ('M0', 'M2', 'M3', 'M4') for s in SEEDS} | {'M1-all-unmatched'}
    if set(parity) != expected or {k for k, v in parity.items() if not v} != {'M3-seed-2027'}:
        raise ValueError('original complete cohort/failure boundary differs')
    output.mkdir(parents=True)
    runs = {}; order = None; frame_cohort = None; identity_gt_cohort = None; protocol = None
    gt_root = root / 'evaluator-only-ground-truth'
    gt_sha = sha(gt_root / 'ground-truth.jsonl'); gt_manifest_sha = sha(gt_root / 'manifest.json')
    for run in sorted(expected):
        campaign = 'birth' if run.startswith('M4-') else 'mechanism'; directory = metrics_root / campaign / run
        report = read(directory / 'report.json'); metrics = read(directory / 'metrics.json'); runtime = read(directory / 'runtime.json')
        golden = read(directory / 'golden-cases.json')
        raw = root / 'streams' / campaign / 'first' / run / 'predictions.jsonl'
        original = next(x for x in provenance['runs'] if x['run_id'] == run and x['execution'] == 'first')['result']
        if report['status'] != 'completed' or report['paper_eligible'] is not False or any(report[k] is not True for k in ('no_model_selection', 'no_test_payload', 'no_validation_parameter_fitting')):
            raise ValueError('seen exploratory evaluation boundary changed')
        if report['predictions_sha256'] != sha(raw) or report['ground_truth_sha256'] != gt_sha or report['ground_truth_manifest_sha256'] != gt_manifest_sha:
            raise ValueError('actual evaluator inputs differ from original streams/GT')
        coverage = report['coverage']
        if coverage['frames'] != 3316 or coverage['sequences'] != 21 or coverage['final_sequence_commit_sha256'] != original['sequence_commits']:
            raise ValueError('full evaluator/output commit coverage differs')
        for name, record in report['files'].items():
            if name not in ('metrics.json', 'runtime.json', 'golden-cases.json') or record['sha256'] != sha(directory / name) or record['size'] != (directory / name).stat().st_size:
                raise ValueError('metric report byte binding differs')
        if set(report['files']) != {'metrics.json', 'runtime.json', 'golden-cases.json'} or golden['passed'] is not True:
            raise ValueError('complete engine/golden-case outputs required')
        engine = {x['path']: x['sha256'] for x in runtime['source_trees']['trackeval']['files']}
        if any(engine['metrics/' + name] != digest for name, digest in AUTHOR.items()):
            raise ValueError('evaluation runtime author source mismatch')
        if protocol is None: protocol = report['protocol_sha256']
        if protocol != report['protocol_sha256'] or report['protocol']['split'] != 'official_validation' or report['reporting_scope'] != 'car_only':
            raise ValueError('same seen-SPD car protocol required')
        car = metrics['trackeval']['car']; sequences = car['sequences']
        candidate_order = sorted(sequences); candidate_frames = {s: sequences[s]['frames'] for s in candidate_order}
        candidate_gt = {s: sequences[s]['gt_objects'] for s in candidate_order}
        if order is None: order, frame_cohort, identity_gt_cohort = candidate_order, candidate_frames, candidate_gt
        if len(order) != 21 or sum(candidate_frames.values()) != 3316 or order != candidate_order or frame_cohort != candidate_frames or identity_gt_cohort != candidate_gt:
            raise ValueError('complete paired frame/GT cohort required')
        data = sufficient(sequences, order); combined = aggregate(data, np.ones((1, 21), dtype=int))
        errors = {}
        for field, value in car['HOTA_curves'].items():
            error = float(np.max(np.abs(np.asarray(value) - combined[field][0])))
            if error > 1e-12: raise ValueError('official HOTA cohort arithmetic differs: ' + field)
            errors[field] = error
        summary = {k: float(combined[k].mean()) if k in ('HOTA', 'AssA', 'DetA') else float(combined[k][0]) for k in car['summary']}
        if any(abs(summary[k] - v) > 1e-12 for k, v in car['summary'].items()) or car['summary'] != report['primary_car']['trackeval']:
            raise ValueError('official TrackEval summary differs')
        nuscenes = {k: value['car'] for k, value in metrics['nuscenes']['label_metrics'].items() if k not in ('lgd', 'tid')}
        if nuscenes != report['primary_car']['nuscenes'] or any(not np.isfinite(v) for v in nuscenes.values()):
            raise ValueError('complete native nuScenes metric readback differs')
        runs[run] = dict(data=data, raw=dict(trackeval=summary, nuscenes=nuscenes), strict_replay_pass=parity[run],
                         predictions_sha256=report['predictions_sha256'], report_sha256=sha(directory / 'report.json'), max_arithmetic_error=max(errors.values()))
    indices, weights = draws(len(order))
    plan = dict(kind='rbf_stage1_seen_SPD_sequence_statistics_plan_v1', seed=1337, draws=10000, bit_generator='PCG64',
                sequence_order=order, resample_unit='whole native sequence with replacement', shared_draws_across_all_seeds_and_models=True,
                metric_combination='author TP-weighted association and summed HOTA/identity counts; never arithmetic sequence mean',
                bootstrap_metrics=list(METRICS), confidence_interval='2.5 and 97.5 percentiles; NumPy linear interpolation',
                model_estimand='mean across original seed-matched runs on each shared sequence draw',
                M1='one deterministic execution; compared with three-seed M0 mean; no invented replicate or sample SD',
                M3='all three raw values retained; failed seed2027 prevents qualified model/bootstrap comparison',
                nuScenes_bootstrap_complete=False, metric_matching_engines_independently_rerun=False, seen_val_exploratory=True,
                physical_session_independence_claimed=False, stage1_qualified=False, paper_metrics_complete=False,
                readback_sha256=args.readback_sha256, provenance_sha256=args.provenance_sha256, source_sha256=sha(__file__))
    # Freeze actual protocol/draws before any bootstrap metric calculation.
    new(output / 'frozen-statistics-plan.json', plan)
    with (output / 'shared-sequence-draws.npy').open('xb') as stream: np.save(stream, indices, allow_pickle=False)
    distributions = {run: metric_values(aggregate(v['data'], weights)) for run, v in runs.items()}
    models = {}; comparisons = {}
    baseline_ids = [f'M0-seed-{s}' for s in SEEDS]
    baseline_draws = np.mean([distributions[x] for x in baseline_ids], axis=0)
    baseline_point = np.mean([[runs[x]['raw']['trackeval'][k] for k in METRICS] for x in baseline_ids], axis=0)
    for mode in ('M0', 'M1', 'M2', 'M3', 'M4'):
        ids = ['M1-all-unmatched'] if mode == 'M1' else [f'{mode}-seed-{s}' for s in SEEDS]
        valid = all(runs[x]['strict_replay_pass'] for x in ids)
        models[mode] = dict(original_run_ids=ids, original_run_count=len(ids), deterministic_control=mode == 'M1',
                           all_strict_replays_pass=valid, raw_descriptive_only=not valid,
                           trackeval=descriptive([runs[x]['raw']['trackeval'] for x in ids]),
                           nuscenes=descriptive([runs[x]['raw']['nuscenes'] for x in ids]),
                           sequence_bootstrap_95_percent_interval=None)
        if valid:
            combined = np.mean([distributions[x] for x in ids], axis=0)
            models[mode]['sequence_bootstrap_95_percent_interval'] = interval(combined)
            if mode != 'M0':
                point = np.mean([[runs[x]['raw']['trackeval'][k] for k in METRICS] for x in ids], axis=0) - baseline_point
                comparisons[mode + '-minus-M0'] = dict(candidate_run_count=len(ids), baseline_run_count=3,
                    deterministic_candidate=mode == 'M1', paired_whole_sequence_draws=True,
                    observed_difference=dict(zip(METRICS, map(float, point))), sequence_bootstrap_95_percent_interval=interval(combined - baseline_draws),
                    interpretation='conditional seen-val sequence uncertainty; not unseen/generalization or seed-resampling uncertainty')
    for name, record in manifest.items():
        if sha(metrics_root / name) != record['sha256']: raise ValueError('metric/source bytes changed during statistics')
    result = dict(kind='rbf_stage1_13_run_independent_sequence_arithmetic_and_partial_statistics_v1',
                  checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), complete_13_raw_metric_readback=True,
                  all_13_official_aggregate_arithmetic_verified=True, full_original_21_sequences_3316_frames=True,
                  models=models, qualified_strict_replay_paired_comparisons=comparisons,
                  per_run={k: {field: value for field, value in v.items() if field != 'data'} for k, v in runs.items()},
                  plan_sha256=sha(output / 'frozen-statistics-plan.json'), shared_draws_sha256=sha(output / 'shared-sequence-draws.npy'),
                  bootstrap_metric_scope=list(METRICS), nuScenes_bootstrap_complete=False,
                  independent_per_frame_matching_and_metric_engines_rerun=False, full_error_duration_causal_chain_complete=False,
                  original_M3_seed2027_failure_preserved=True, stage1_qualified=False, all_K_statistics_complete=False, paper_metrics_complete=False)
    new(output / 'statistics-receipt.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'readback-sha256', 'provenance-sha256', 'output'): p.add_argument('--' + name, required=True)
    execute(p.parse_args())
