"""Full seen-val K1/K4 search or selector admission using unchanged frozen oracles.

This consumes all scheduled val sequences, including events without new queries.
The selector requires same-byte completed search evidence and never reruns it.
Continuous-state, complete-method and same-resource performance gates are separate.
"""
import argparse
import datetime
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

from rbf_nested_seen_val_v2_common import R, new, register, sha

CPU = R/'source-freezes/rbf-seen-val-fixed-K1-K4-full-independent-CPU-v1-20261005'
CPU_SHA = 'fabab95ea1dc7ff705ec7529b5de175d5106f64bd7acb7d90afced6cce9d916e'
ORIGINALS = {
    (1, 'search'): ('rbf-independent-fixed-Top1-search-pruning-v1-20261004', '5d64a621c4f77df2cd6957ba8f33b0673228ce83ff856742a9a2010bb7f89fb3'),
    (4, 'search'): ('rbf-independent-fixed-topK-search-pruning-v1-20261004', 'd65d00be0775b157312ac93d5d7b592cd129c4578053023a93f6dfd2500ef100'),
    (1, 'selector'): ('rbf-independent-fixed-Top1-conditional-selector-v1-20261004', '0e358b664da8013301de80537b8bdbbda675249d943aa13031ed83ac3fcd0c04'),
    (4, 'selector'): ('rbf-independent-fixed-topK-conditional-selector-v1-20261004', 'fce4ff0bd294ee214c07850975bb21634f2099b6c5fad974a238c5e2eb4b1cfc')}
FALSE_FLAGS = ('train_or_other_forest_acceptance_inherited', 'measured_network_arrival_history_verified',
               'continuous_states_independently_accepted', 'complete_online_method_accepted',
               'learned_Stage2_complete', 'same_resource_performance_accepted', 'paper_performance_complete')


def load_exact(path, name):
    path = Path(path)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    result = sys.modules[name]
    assert Path(result.__file__).resolve() == path
    return result


def frozen(directory, digest):
    assert sha(directory/'source-freeze.json') == digest
    value = json.loads((directory/'source-freeze.json').read_bytes())
    for name, item in value['sources'].items():
        path = directory/name
        assert path.resolve().is_relative_to(directory) and not path.is_symlink()
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
    for key in ('references', 'unchanged_references'):
        for item in value.get(key, []):
            assert sha(item['path']) == item['sha256']
    return value


def input_module():
    frozen(CPU, CPU_SHA)
    return load_exact(CPU/'rbf_seen_val_fixed_CPU_binding.py', 'unchanged_seen_val_fixed_CPU_for_search_selector')


def oracle_gate(width, stage):
    name, digest = ORIGINALS[width, stage]
    directory = R/'source-freezes'/name
    freeze = frozen(directory, digest)
    item = freeze['qualification']; assert sha(item['path']) == item['sha256']
    proof = json.loads(Path(item['path']).read_bytes())
    stem = 'top1' if width == 1 else 'topk'
    path = directory/f'independent_fixed_{stem}_{stage}.py'
    assert proof['oracle_sha256'] == sha(path)
    assert proof['qualification_source_sha256'] == sha(directory/f'qualify_independent_fixed_{stem}_{stage}.py')
    assert proof['software_qualification_only'] is True and proof['new_final_model_experiment_accepted'] is False
    oracle = load_exact(path, f'unchanged_seen_val_K{width}_{stage}_oracle')
    assert oracle.ATOL == oracle.RTOL == 1e-8
    if stage == 'search':
        assert proof['controls_rejected'] == (20 if width == 1 else 18)
        assert proof['mathematical_controls']['exhaustive_global_product_cases'] == 60
        assert proof['root_oracle_sha256'] == oracle.ROOT_SHA == sha(oracle.ROOT_ORACLE)
        assert proof['fixed_atol'] == proof['fixed_rtol'] == 1e-8
    else:
        assert proof['negative_controls_rejected'] == (22 if width == 1 else 21)
        assert proof['existing_component_decisions'] == 229
        assert proof['existing_MAP_and_Bayes_output_differences'] == (0 if width == 1 else 17)
        assert proof['atol'] == proof['rtol'] == 1e-8 and oracle.DECIMAL_PRECISION == 70
    return oracle, dict(path=str(path), sha256=sha(path), original_source_freeze_sha256=digest,
                        qualification_path=item['path'], qualification_sha256=item['sha256'])


def source_gate(width):
    directory = Path(__file__).resolve().parent
    own = json.loads((directory/'source-freeze.json').read_bytes())
    assert own['kind'] == 'rbf_seen_val_fixed_search_selector_source_v1'
    frozen(directory, sha(directory/'source-freeze.json'))
    originals = {stage: oracle_gate(width, stage)[1] for stage in ('search', 'selector')}
    return dict(source_freeze_sha256=sha(directory/'source-freeze.json'), K=width,
                unchanged_oracles=originals, input_gate_source_binding=input_module().source_gate(width))


def scope(width):
    assert type(width) is int and width in (1, 4)
    return f'SPD seen-val exploratory scheduled snapshots; fixed K{width} baseline'


def kind(width, stage):
    assert stage in ('search', 'selector')
    return f'rbf_seen_val_fixed_K{width}_full_{stage}_admission_v1'


def validate_search(search, value, model, sources, width):
    assert search['kind'] == kind(width, 'search') and search['stage'] == 'search'
    assert search['K'] == value['K'] == width and search['method'] == value['method'] == 'topk'
    assert search['scope'] == value['scope'] == scope(width)
    for key in ('task_id', 'seed', 'recipe_sha256'):
        assert search[key] == value[key]
    assert search['final_model_binding'] == model and search['source_binding'] == sources
    assert search['completed_sequences'] == len(value['sequences']) == 21
    assert search['completed_events'] == sum(v['events'] for v in value['sequences']) == 3316
    assert search['observations'] == search['node_pruning_steps'] == value['total_nodes']
    assert search['all_seen_val_search_pruning_independently_verified'] is True
    assert search['conditional_output_selector_independently_accepted'] is False
    assert search['atol'] == search['rtol'] == 1e-8
    assert type(search['max_abs_error']) in (float, int) and math.isfinite(search['max_abs_error']) and search['max_abs_error'] >= 0
    for flag in FALSE_FLAGS:
        assert search[flag] is False
    proofs = search['sequence_receipts']
    assert len(proofs) == len({p['sequence_id'] for p in proofs}) == 21
    assert [p['sequence_id'] for p in proofs] == [e['sequence_id'] for e in value['sequences']]
    assert [p['database_sha256'] for p in proofs] == [e['database_sha256'] for e in value['sequences']]


def search_receipts(path, byte_path, value, model, sources, width, selector):
    module = input_module(); search = module.registered(path)
    validate_search(search, value, model, sources, width)
    assert search['byte_admission_sha256'] == sha(byte_path)
    for index, (entry, item) in enumerate(zip(value['sequences'], search['sequence_receipts'])):
        expected = Path(path).parent/f'sequence-{index:02d}.json'
        assert item['path'] == str(expected)
        assert not expected.is_symlink() and sha(expected) == item['sha256']
        proof = json.loads(expected.read_bytes())
        selector.validate_search_sequence(proof, entry['database_sha256'], entry['sequence_id'], entry['events'], entry['nodes'])
    return search['sequence_receipts']


def validate_selector_result(result, entry, width):
    tag = 'Top1' if width == 1 else 'topK'
    assert result['kind'] == f'rbf_independent_fixed_{tag}_conditional_selector_audit_v1'
    for key, expected in (('sequence_id', entry['sequence_id']), ('database_sha256', entry['database_sha256']),
                          ('events', entry['events']), ('observations', entry['nodes'])):
        assert result[key] == expected
    assert result['atol'] == result['rtol'] == 1e-8 and result['decimal_precision'] == 70
    for flag in ('upstream_same_byte_search_proof_required', 'pairwise_conditional_risks_independently_verified',
                 'all_outputs_match_frozen_float64_minimum_and_SHA_tie_rule',
                 'complete_and_truncated_conditional_lower_semantics_verified', 'no_MAP_substitution_or_regret_fallback',
                 'conditional_output_selector_independently_accepted'):
        assert result[flag] is True
    for flag in ('search_pruning_rerun', 'continuous_states_independently_accepted', 'full_online_method_accepted',
                 'global_full_history_Bayes_accepted', 'true_posterior_certificate', 'same_resource_performance_accepted',
                 'paper_performance_complete'):
        assert result[flag] is False


def database_for(byte_path, entry):
    paths = list(byte_path.parent.glob(f'rank*-unpack/rank-*/{entry["sequence_id"]}/receipt.json'))
    assert len(paths) == 1
    receipt_path = paths[0]
    assert not any(p.is_symlink() for p in (receipt_path, *receipt_path.parents))
    record = json.loads(receipt_path.read_bytes())['databases'][entry['sequence_id']]
    assert record['sha256'] == entry['database_sha256']
    database = receipt_path.parent/record['path']
    assert database.resolve().is_relative_to(receipt_path.parent.resolve())
    assert not any(p.is_symlink() for p in (database, *database.parents))
    return database


def run(args):
    sources = source_gate(args.K); module = input_module()
    initial_sha = sha(args.byte_admission)
    job, checkpoint, model, input_sources = module.validate_output(args.byte_admission, width=args.K)
    assert input_sources == sources['input_gate_source_binding'] and sha(args.byte_admission) == initial_sha
    value = module.registered(args.byte_admission)
    search_oracle, _ = oracle_gate(args.K, 'search'); selector, _ = oracle_gate(args.K, 'selector')
    prerequisites = None; prerequisite_sha = None
    if args.stage == 'selector':
        assert args.search_admission is not None
        prerequisite_sha = sha(args.search_admission)
        prerequisites = search_receipts(args.search_admission, args.byte_admission, value, model, sources, args.K, selector)
        assert sha(args.search_admission) == prerequisite_sha
    else:
        assert args.search_admission is None
    assert args.output.resolve().is_relative_to(R/'artifacts') and not args.output.exists()
    assert not any(p.is_symlink() for p in (args.output, *args.output.parents))
    args.output.mkdir(parents=True)
    binding = dict(kind=kind(args.K, args.stage), stage=args.stage, K=args.K, method='topk', scope=scope(args.K),
        task_id=value['task_id'], seed=value['seed'], recipe_sha256=job['recipe_sha256'],
        byte_admission_path=str(args.byte_admission), byte_admission_sha256=initial_sha,
        final_model_binding=model, source_binding=sources, **{key:False for key in FALSE_FLAGS})
    if prerequisites is not None:
        binding.update(search_admission_path=str(args.search_admission), search_admission_sha256=prerequisite_sha,
                       admitted_search_sequence_receipts=prerequisites, search_pruning_rerun=False)
    new(args.output/'binding.json', binding)
    started = time.monotonic(); results = []; evidence = []
    try:
        for index, entry in enumerate(value['sequences']):
            database = database_for(args.byte_admission, entry)
            def progress(data):
                if data['completed_events'] == data['total_events'] or data['completed_events'] % 20 == 0:
                    print(json.dumps(dict(data, K=args.K, seed=value['seed'], sequence_id=entry['sequence_id'],
                        completed_sequences=index, total_sequences=21, whole_cohort_ETA_seconds=None,
                        whole_cohort_ETA_reason='unequal retained-prefix, component-merge and identity-scope work')), flush=True)
            if args.stage == 'search':
                result = search_oracle.verify_database(database, entry['database_sha256'], progress)
                selector.validate_search_sequence(result, entry['database_sha256'], entry['sequence_id'], entry['events'], entry['nodes'])
            else:
                item = prerequisites[index]; assert sha(item['path']) == item['sha256']
                result = selector.verify_database(database, entry['database_sha256'], json.loads(Path(item['path']).read_bytes()), progress)
                validate_selector_result(result, entry, args.K)
            assert result['events'] == entry['events'] and result['observations'] == entry['nodes']
            assert result['max_abs_error'] and all(type(e) in (float, int) and math.isfinite(e) and e >= 0 for e in result['max_abs_error'].values())
            path = args.output/f'sequence-{index:02d}.json'; new(path, result); results.append(result)
            evidence.append(dict(path=str(path), sha256=sha(path), sequence_id=entry['sequence_id'], database_sha256=entry['database_sha256']))
        assert len(results) == 21 and sum(v['events'] for v in results) == 3316
        assert sum(v['observations'] for v in results) == value['total_nodes']
        assert module.validate_output(args.byte_admission, width=args.K) == (job, checkpoint, model, input_sources)
        assert sha(args.byte_admission) == initial_sha and source_gate(args.K) == sources
        for item in evidence + (prerequisites or []): assert sha(item['path']) == item['sha256']
        if prerequisites is not None:
            assert sha(args.search_admission) == prerequisite_sha
            assert search_receipts(args.search_admission, args.byte_admission, value, model, sources, args.K, selector) == prerequisites
        final = dict(binding, completed_sequences=21, completed_events=3316, observations=value['total_nodes'],
            sequence_receipts=evidence, max_abs_error=max(max(r['max_abs_error'].values()) for r in results),
            atol=1e-8, rtol=1e-8, elapsed_seconds=time.monotonic()-started,
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if args.stage == 'search':
            final.update(all_seen_val_search_pruning_independently_verified=True,
                         conditional_output_selector_independently_accepted=False,
                         node_pruning_steps=sum(r['node_pruning_steps'] for r in results),
                         merged_components=sum(r.get('merged_components', 0) for r in results))
            validate_search(final, value, model, sources, args.K)
        else:
            final.update(all_seen_val_conditional_selectors_independently_verified=True,
                         conditional_output_selector_independently_accepted=True, decimal_precision=70,
                         component_decisions=sum(r.get('component_decisions', 0) for r in results),
                         MAP_and_output_differ=sum(r.get('posterior_MAP_and_output_differ', 0) for r in results),
                         scoped_identity_queries=sum(r.get('scoped_identity_queries', 0) for r in results))
        new(args.output/'acceptance.json', final); register(args.output/'acceptance.json', final['kind'])
        print(json.dumps(final), flush=True)
        return final
    except BaseException as error:
        failure = dict(binding, kind=kind(args.K, args.stage)+'_failure', error_type=type(error).__name__,
                       message=str(error), completed_sequences=len(results), automatic_retry=False, experiment_accepted=False)
        new(args.output/'failure.json', failure); register(args.output/'failure.json', failure['kind'])
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, choices=(1, 4), required=True)
    parser.add_argument('--stage', choices=('search', 'selector'))
    parser.add_argument('--byte-admission', type=Path)
    parser.add_argument('--search-admission', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-source', action='store_true')
    args = parser.parse_args()
    if args.check_source:
        assert args.stage is args.byte_admission is args.search_admission is args.output is None
        print(json.dumps(dict(source_gate_passed=True, **source_gate(args.K))), flush=True)
        return
    assert args.stage is not None and args.byte_admission is not None and args.output is not None
    run(args)


if __name__ == '__main__': main()
