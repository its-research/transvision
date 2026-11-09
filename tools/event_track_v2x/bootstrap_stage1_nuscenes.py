#!/usr/bin/env python3
"""Full paired sequence bootstrap after all author score-subset cache admissions."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def read(path, digest):
    p = Path(path).absolute()
    if not p.is_file() or any(x.is_symlink() for x in (p, *p.parents)) or sha(p) != digest:
        raise ValueError('ordinary admitted input and exact SHA required')
    return json.loads(p.read_bytes())


def new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def interval(values):
    finite = np.isfinite(values)
    if not finite.all():
        return dict(available=False, finite_replicates=int(finite.sum()), total_replicates=len(values),
                    reason='undefined author metric in at least one complete bootstrap replicate; no silent deletion')
    lo, hi = np.percentile(values, [2.5, 97.5], method='linear')
    return dict(available=True, low=float(lo), high=float(hi), total_replicates=len(values))


def evaluate(args):
    root = Path(args.cache_campaign).absolute(); output = Path(args.output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('fresh standalone bootstrap output required')
    proof_path = root / 'campaign-receipt.json'; proof = read(proof_path, args.cache_campaign_sha256)
    if proof['all13_complete_original_cohort_score_subset_caches_admitted'] is not True or len(proof['results']) != 13:
        raise ValueError('all13 actual author matching score-subset caches required')
    original_plan = read(args.shared_plan, args.shared_plan_sha256)
    if original_plan['seed'] != 1337 or original_plan['draws'] != 10000 or original_plan['bit_generator'] != 'PCG64':
        raise ValueError('original frozen shared resampling plan required')
    if sha(args.shared_draws) != args.shared_draws_sha256: raise ValueError('original shared draw bytes differ')
    draws = np.load(args.shared_draws, allow_pickle=False)
    if draws.shape != (10000, 21) or not np.issubdtype(draws.dtype, np.integer) or np.any(draws < 0) or np.any(draws >= 21):
        raise ValueError('whole native sequence draw coverage differs')
    expected_draws = np.random.Generator(np.random.PCG64(1337)).integers(0, 21, size=(10000, 21))
    if not np.array_equal(draws, expected_draws): raise ValueError('shared draws are not exact frozen seed1337/PCG64')
    oracle_path = Path(args.oracle_source).absolute()
    if args.oracle_sha256 != proof['oracle_sha256'] or sha(oracle_path) != args.oracle_sha256 or oracle_path.is_symlink():
        raise ValueError('original admitted frozen aggregation source differs')
    spec = importlib.util.spec_from_file_location('frozen_nuscenes_bootstrap_aggregation', oracle_path)
    oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
    run_ids = [x['run_id'] for x in proof['results']]
    expected_ids = {'M1-all-unmatched'} | {f'M{m}-seed-{s}' for m in (0,2,3,4) for s in (1337,2027,3407)}
    if set(run_ids) != expected_ids or len(set(run_ids)) != 13: raise ValueError('original13 logical configurations required')
    pins = {str(proof_path): args.cache_campaign_sha256, str(Path(args.shared_plan).absolute()): args.shared_plan_sha256,
            str(Path(args.shared_draws).absolute()): args.shared_draws_sha256, str(oracle_path): args.oracle_sha256,
            str(Path(__file__).absolute()): sha(__file__)}
    caches = {}; metadata = {}; metric_names = list(oracle.METRICS) + ['amota', 'amotp']
    for result in proof['results']:
        run = result['run_id']; path = root / run / 'actual-author-score-subset-cache.json'
        cache = read(path, result['cache_sha256']); pins[str(path)] = result['cache_sha256']
        if not result['full_native_all_actual_local_score_subsets_original_cohort_admitted'] or not cache['all21_actual_score_subset_caches_original_full_cohort_admitted']:
            raise ValueError('individual actual cache admission missing')
        if cache['source_sha256'] != proof['source_sha256'] or cache['sequence_ids'] != original_plan['sequence_order']:
            raise ValueError('actual cache source or original sequence ordering differs')
        if cache['input_evidence']['run_id'] != run or cache['strict_replay_pass'] != result['original_strict_replay_pass']:
            raise ValueError('actual cache run/strict-failure binding differs')
        if any(not s['all_distinct_prediction_score_subsets_actually_matched'] for s in cache['sequences'].values()):
            raise ValueError('all actual local score subsets required')
        prepared = oracle.prepare(cache); caches[run] = prepared
        full, _ = oracle.evaluate(prepared, np.ones(21, dtype=int))
        metadata[run] = dict(full_native_metrics=full, strict_replay_pass=cache['strict_replay_pass'],
            actual_all_local_score_subset_rows=sum(len(s['rows']) for s in cache['sequences'].values()),
            source_sha256=cache['source_sha256'], cache_sha256=result['cache_sha256'])
    if metadata['M3-seed-2027']['strict_replay_pass'] is not False:
        raise ValueError('original M3 seed2027 strict failure must remain failed')
    output.mkdir(parents=True); started = time.monotonic()
    plan = dict(kind='rbf_stage1_full_nuscenes_sequence_cluster_bootstrap_plan_v1', seed=1337, draws=10000,
        bit_generator='PCG64', shared_draws_sha256=args.shared_draws_sha256, sequence_order=original_plan['sequence_order'],
        recompute_global_TP_scores_GT_recall_thresholds_each_draw=True,
        exact_author_matching_subset_selected_for_every_scene_and_new_threshold=True,
        duplicated_sequences_independent_with_private_ID_namespace=True, sequence_macro_average_used=False,
        no_finite_replicate_deletion=True, M3_all_raw_draws_retained_but_no_qualified_inference=True,
        M1_one_deterministic_run_sample_std_unknown=True, amotp_unit='center distance meters; lower is better',
        source_sha256=sha(__file__), oracle_sha256=args.oracle_sha256, input_sha256=pins,
        numpy_version=np.__version__,
        physical_session_independence_claimed=False, seen_SPD_val_exploratory=True,
        all_K_statistics_complete=False, stage1_qualified=False, paper_metrics_complete=False)
    new(output / 'plan.json', plan)
    values = np.empty((13, 10000, len(metric_names)))
    for k, draw in enumerate(draws):
        weights = np.bincount(draw, minlength=21)
        for i, run in enumerate(run_ids):
            headline, _ = oracle.evaluate(caches[run], weights)
            values[i, k] = [headline[name] for name in metric_names]
        if (k + 1) % 500 == 0:
            print(json.dumps(dict(stage='full_shared_nuscenes_sequence_cluster_bootstrap', completed=k+1, total=10000)), flush=True)
    with (output / 'all13-complete-bootstrap-draw-metrics.npy').open('xb') as stream:
        np.save(stream, values, allow_pickle=False)
    model_values = {}; models = {}; comparisons = {}
    for model in ('M0','M1','M2','M3','M4'):
        ids = [r for r in run_ids if r.startswith(model+'-')]; indexes = [run_ids.index(r) for r in ids]
        mean_values = np.mean(values[indexes], axis=0); model_values[model] = mean_values
        full = np.asarray([[metadata[r]['full_native_metrics'][name] for name in metric_names] for r in ids])
        qualified = all(metadata[r]['strict_replay_pass'] for r in ids)
        models[model] = dict(run_ids=ids, original_run_count=len(ids), all_strict_replay_pass=qualified,
            raw_descriptive_metrics={name: dict(mean=float(full[:,j].mean()), sample_std=float(full[:,j].std(ddof=1)) if len(ids)>1 else None) for j,name in enumerate(metric_names)},
            qualified_10000_sequence_intervals={name: interval(mean_values[:,j]) for j,name in enumerate(metric_names)} if qualified else None,
            no_qualified_interval_reason=None if qualified else 'original strict replay failure retained')
        if model != 'M0' and qualified:
            delta = mean_values - model_values['M0']
            comparisons[model+'-minus-M0'] = {name: interval(delta[:,j]) for j,name in enumerate(metric_names)}
    for path, digest in pins.items():
        if sha(path) != digest: raise ValueError('original admitted inputs changed during complete bootstrap')
    receipt = dict(kind='rbf_stage1_all13_full_nuscenes_10000_paired_sequence_bootstrap_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), all13_complete_10000_sequence_draw_metrics_present=True,
        nuScenes_bootstrap_metric_scope=metric_names, models=models, per_run=metadata,
        qualified_strict_replay_paired_comparisons=comparisons, full_input_sha256=pins,
        source_sha256=sha(__file__), plan_sha256=sha(output/'plan.json'),
        actual_bootstrap_values_sha256=sha(output/'all13-complete-bootstrap-draw-metrics.npy'),
        original_M3_seed2027_failure_preserved=True, ambiguous_or_undefined_metrics_not_repaired=True,
        same_resource_costs_complete=False, all_K_statistics_complete=False,
        complete_stage1=False, paper_metrics_complete=False, goal_status='active', seconds=time.monotonic()-started)
    new(output / 'bootstrap-receipt.json', clean(receipt))
    print(json.dumps(dict(stage='complete_nuscenes_sequence_bootstrap', draws=10000, logical_runs=13,
        actual_values_sha256=receipt['actual_bootstrap_values_sha256'], complete_stage1=False, paper_metrics_complete=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-campaign','cache-campaign-sha256','shared-plan','shared-plan-sha256',
                 'shared-draws','shared-draws-sha256','oracle-source','oracle-sha256','output'):
        parser.add_argument('--'+name, required=True)
    evaluate(parser.parse_args())
