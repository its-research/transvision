"""Separate new-model TopK search/pruning gate after full byte/factor admission.

This supplements, rather than replaces, the frozen causal/cache/fresh-state
driver. It accepts no output selector, full online method or paper performance.
"""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time

import independent_fixed_top1_search as oracle
from rbf_nested_seen_val_v2_common import R, new, register, sha

TCPU = R/'source-freezes/rbf-final-refit-fixed-Top1-full-independent-CPU-v1-20261004'
sys.path.insert(0, str(TCPU))
from rbf_final_refit_top1_binding import validate_output, source_gate as input_source_gate


def source_gate():
    directory = Path(__file__).resolve().parent
    freeze = json.loads((directory/'source-freeze.json').read_bytes())
    assert freeze['kind'] == 'rbf_independent_fixed_Top1_search_pruning_source_v1'
    for name, spec in freeze['sources'].items():
        path = directory/name
        assert not path.is_symlink() and sha(path) == spec['sha256'] and path.stat().st_size == spec['bytes']
    for item in freeze['unchanged_references']:
        assert sha(item['path']) == item['sha256']
    qualification = json.loads(Path(freeze['qualification']['path']).read_bytes())
    assert sha(freeze['qualification']['path']) == freeze['qualification']['sha256']
    assert qualification['kind'] == 'rbf_independent_fixed_Top1_search_oracle_qualification_v1'
    assert qualification['oracle_sha256'] == sha(directory/'independent_fixed_top1_search.py')
    assert qualification['qualification_source_sha256'] == sha(directory/'qualify_independent_fixed_top1_search.py')
    assert qualification['root_oracle_sha256'] == oracle.ROOT_SHA
    assert qualification['controls_rejected'] == 20
    assert qualification['mathematical_controls']['exhaustive_global_product_cases'] == 60
    assert qualification['fixed_atol'] == qualification['fixed_rtol'] == oracle.ATOL == oracle.RTOL == 1e-8
    assert qualification['software_qualification_only'] is True
    assert qualification['new_final_model_experiment_accepted'] is qualification['paper_performance_complete'] is False
    return dict(source_freeze_sha256=sha(directory/'source-freeze.json'),
        oracle_sha256=sha(directory/'independent_fixed_top1_search.py'),
        driver_sha256=sha(__file__), qualification_sha256=freeze['qualification']['sha256'],
        input_gate_source_binding=input_source_gate())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--byte-admission', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-source', action='store_true')
    args = parser.parse_args()
    source_binding = source_gate()
    if args.check_source:
        assert args.byte_admission is None and args.output is None
        print(json.dumps(dict(source_gate_passed=True, **source_binding)), flush=True)
        return
    assert args.byte_admission is not None and args.output is not None
    assert args.byte_admission.resolve().is_relative_to(R/'artifacts') and not args.byte_admission.is_symlink()
    value = json.loads(args.byte_admission.read_bytes())
    job, _, model_binding, input_binding = validate_output(value)
    assert input_binding == source_binding['input_gate_source_binding']
    for key, artifact in value['artifacts'].items():
        path = args.byte_admission.parent/(key+('.json' if key == 'receipt' else '.tar.gz'))
        assert not path.is_symlink() and sha(path) == artifact['sha256'] and path.stat().st_size == artifact['bytes']
    assert args.output.resolve().is_relative_to(R/'artifacts') and not args.output.exists()
    assert not any(p.is_symlink() for p in (args.output, *args.output.parents))
    args.output.mkdir(parents=True)
    binding = dict(kind='rbf_final_refit_fixed_Top1_full_search_pruning_admission_v1',
        method='topk', K=1, task_id=value['task_id'], seed=value['seed'],
        recipe_sha256=job['recipe_sha256'], byte_admission_sha256=sha(args.byte_admission),
        final_model_binding=model_binding, source_binding=source_binding,
        old_model_search_or_forest_acceptance_inherited=False,
        conditional_output_selector_independently_accepted=False,
        continuous_states_independently_accepted=False,
        complete_online_method_accepted=False, learned_Stage2_complete=False,
        same_resource_performance_accepted=False, paper_performance_complete=False)
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
                if data['completed_events'] == data['total_events'] or data['completed_events'] % 20 == 0:
                    payload = dict(data, completed_sequences=index, total_sequences=46,
                                   whole_cohort_ETA_seconds=None,
                                   whole_cohort_ETA_reason='unequal raw-prefix and merge workloads')
                    print(json.dumps(payload), flush=True)
            result = oracle.verify_database(database, entry['database_sha256'], progress)
            assert result['events'] == entry['events'] and result['observations'] == entry['nodes']
            new(args.output/f'sequence-{index:02d}.json', result); results.append(result)
        assert len(results) == 46 and sum(v['events'] for v in results) == 7445
        assert sum(v['observations'] for v in results) == value['total_nodes']
        assert source_gate() == source_binding
        final = dict(binding, completed_sequences=46, completed_events=7445,
            observations=value['total_nodes'], all_final_model_cohort_search_pruning_independently_verified=True,
            node_pruning_steps=sum(v['node_pruning_steps'] for v in results),
            merged_components=sum(v['merged_components'] for v in results),
            max_abs_error=max(max(v['max_abs_error'].values()) for v in results), atol=1e-8, rtol=1e-8,
            elapsed_seconds=time.monotonic()-started,
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        new(args.output/'acceptance.json', final); register(args.output/'acceptance.json', final['kind'])
        print(json.dumps(final), flush=True)
    except BaseException as error:
        failure = dict(binding, kind='rbf_final_refit_fixed_Top1_search_pruning_admission_failure_v1',
                       type=type(error).__name__, message=str(error), completed_sequences=len(results), experiment_accepted=False)
        new(args.output/'failure.json', failure); register(args.output/'failure.json', failure['kind'])
        raise


if __name__ == '__main__': main()
