"""Read-only full fixed Top1 causal/cache203/fresh-state output admission.

All three numerical oracles are reused byte-for-byte. No recovery, exclusive
partition, posterior optimum, same-resource comparison or paper metric is
accepted by this driver. Old running Top1 auditors are separate and preserved.
"""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time

from rbf_final_refit_top1_binding import R, validate_output
from rbf_nested_seen_val_v2_common import new, register, sha

STRUCTURE = R/'source-freezes/rbf-independent-exclusive-forest-structure-recovery-v1-20261002'
sys.path.insert(0, str(STRUCTURE))
from causal import verify_database as causal_database
from final_cache203 import CacheAdmission, verify_database as feature_database

FRESH = R/'source-freezes/rbf-independent-fresh-branch-state-v3-normalized-admission-20261001/oracle.py'
FRESH_SHA = '46d5f6b474202961077e43f9c572f6ecfb0a98e992e9f4b19fcd37cccfce1c58'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--byte-admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.byte_admission.read_bytes())
    job, checkpoint, model_binding, source_binding = validate_output(value)
    for key, artifact in value['artifacts'].items():
        path = args.byte_admission.parent/(key+('.json' if key == 'receipt' else '.tar.gz'))
        assert sha(path) == artifact['sha256'] and path.stat().st_size == artifact['bytes']
    admission = CacheAdmission(value['seed'], checkpoint)
    assert admission.final_model_binding == model_binding
    assert sha(FRESH) == FRESH_SHA
    spec = importlib.util.spec_from_file_location('unchanged_final_Top1_independent_fresh_state', FRESH)
    fresh = importlib.util.module_from_spec(spec); spec.loader.exec_module(fresh)
    assert fresh.ATOL == fresh.RTOL == 1e-8
    assert not args.output.exists() and not any(p.is_symlink() for p in (args.output, *args.output.parents))
    args.output.mkdir(parents=True)
    binding = dict(task_id=value['task_id'], seed=value['seed'], method='topk', K=1,
        recipe_sha256=job['recipe_sha256'], byte_admission_sha256=sha(args.byte_admission),
        final_model_binding=model_binding, driver_source_binding=source_binding,
        fresh_oracle_sha256=FRESH_SHA, causal_oracle_sha256=sha(STRUCTURE/'causal.py'),
        atol=1e-8, rtol=1e-8, exclusive_partition_or_recovery_accepted=False,
        complete_online_method_accepted=False, same_resource_performance_accepted=False,
        learned_Stage2_complete=False, paper_performance_complete=False,
        old_model_full_forest_acceptance_inherited=False)
    new(args.output/'binding.json', binding)
    started = time.monotonic(); results = []
    try:
        for index, entry in enumerate(value['sequences']):
            paths = list(args.byte_admission.parent.glob(f'rank*-unpack/rank-*/{entry["sequence_id"]}/receipt.json'))
            assert len(paths) == 1
            receipt = json.loads(paths[0].read_bytes()); record = receipt['databases'][entry['sequence_id']]
            assert record['sha256'] == entry['database_sha256']
            database = paths[0].parent/record['path']
            assert database.resolve().is_relative_to(paths[0].parent.resolve()) and not database.is_symlink()
            def progress(data):
                payload = dict(data)
                payload.update(scope_stage='final_refit_full_Top1_causal_cache_fresh_state',
                    completed_sequences=index, total_sequences=46, ETA_seconds=None,
                    ETA_reason='heterogeneous raw-history and branch-state work; whole-cohort timing unknown')
                print(json.dumps(payload), flush=True)
            features = feature_database(database, entry['database_sha256'], admission, progress=progress)
            causal = causal_database(database, entry['database_sha256'])
            states = fresh.verify_database(database, entry['database_sha256'], progress)
            assert features['events'] == causal['events'] == states['events'] == entry['events']
            assert features['rows'] == causal['observations'] == entry['nodes']
            result = dict(sequence_id=entry['sequence_id'], database_sha256=entry['database_sha256'],
                runtime_raw_cache203_context=features, causal=causal, fresh_state=states)
            new(args.output/f'sequence-{index:02d}.json', result); results.append(result)
        assert len(results) == 46 and sum(v['causal']['events'] for v in results) == 7445
        assert sum(v['causal']['observations'] for v in results) == value['total_nodes']
        from rbf_final_refit_top1_binding import source_gate
        assert source_gate() == source_binding and sha(FRESH) == FRESH_SHA
        final = dict(binding, kind='rbf_final_refit_fixed_Top1_full_causal_cache203_fresh_state_admission_v1',
            completed_sequences=46, completed_events=7445, observations=value['total_nodes'],
            all_original_schedule_causal_raw_commits_verified=True,
            all_203_features_and_arrived_parent_contexts_independently_verified=True,
            all_stored_branch_states_and_chosen_outputs_numerically_verified=True,
            states=sum(v['fresh_state']['states'] for v in results),
            predictions=sum(v['fresh_state']['predictions'] for v in results),
            max_state_abs_error=max(v['fresh_state']['max_abs_error'] for v in results),
            max_cache203_abs_error=max(max(v['runtime_raw_cache203_context']['max_abs_error'].values()) for v in results),
            elapsed_seconds=time.monotonic()-started,
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        new(args.output/'acceptance.json', final); register(args.output/'acceptance.json', final['kind'])
        print(json.dumps(final), flush=True)
    except BaseException as error:
        failure = dict(binding, kind='rbf_final_refit_fixed_Top1_CPU_admission_failure_v1',
            type=type(error).__name__, message=str(error), completed_sequences=len(results), experiment_accepted=False)
        new(args.output/'failure.json', failure); register(args.output/'failure.json', failure['kind'])
        raise


if __name__ == '__main__':
    main()
