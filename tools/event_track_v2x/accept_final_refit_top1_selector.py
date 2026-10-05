"""Full new-model conditional selector gate, consuming successful search only."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import sys
import time

import independent_fixed_top1_selector as oracle
from rbf_nested_seen_val_v2_common import R, new, register, sha

SEARCH_SOURCE = R/'source-freezes/rbf-independent-fixed-Top1-search-pruning-v1-20261004'
sys.path.insert(0, str(SEARCH_SOURCE))
from accept_final_refit_top1_search import source_gate as search_source_gate
from rbf_final_refit_top1_binding import validate_output


def source_gate():
    directory = Path(__file__).resolve().parent
    freeze = json.loads((directory/'source-freeze.json').read_bytes())
    assert freeze['kind'] == 'rbf_independent_fixed_Top1_conditional_selector_source_v1'
    for name, spec in freeze['sources'].items():
        path = directory/name
        assert not path.is_symlink() and sha(path) == spec['sha256'] and path.stat().st_size == spec['bytes']
    for item in freeze['unchanged_references']: assert sha(item['path']) == item['sha256']
    proof = json.loads(Path(freeze['qualification']['path']).read_bytes())
    assert sha(freeze['qualification']['path']) == freeze['qualification']['sha256']
    assert proof['oracle_sha256'] == sha(directory/'independent_fixed_top1_selector.py')
    assert proof['qualification_source_sha256'] == sha(directory/'qualify_independent_fixed_top1_selector.py')
    assert proof['negative_controls_rejected'] == 22
    assert proof['existing_component_decisions'] == 229 and proof['existing_MAP_and_Bayes_output_differences'] == 0
    assert proof['atol'] == proof['rtol'] == oracle.ATOL == oracle.RTOL == 1e-8
    assert proof['software_qualification_only'] is True and proof['new_final_model_experiment_accepted'] is False
    return dict(source_freeze_sha256=sha(directory/'source-freeze.json'),
                oracle_sha256=sha(directory/'independent_fixed_top1_selector.py'),
                driver_sha256=sha(__file__), qualification_sha256=freeze['qualification']['sha256'],
                unchanged_search_source_binding=search_source_gate())


def validate_local_search(search, value, model_binding, source_binding):
    """Pure real-cohort gate; small software examples cannot satisfy it."""
    assert search['kind'] == 'rbf_final_refit_fixed_Top1_full_search_pruning_admission_v1'
    assert search['method'] == 'topk' and search['K'] == 1
    for key in ('task_id','seed','recipe_sha256'): assert search[key] == value[key]
    assert search['final_model_binding'] == model_binding
    assert search['source_binding'] == source_binding['unchanged_search_source_binding']
    assert search['completed_sequences'] == 46 and search['completed_events'] == 7445
    assert search['observations'] == search['node_pruning_steps'] == value['total_nodes']
    assert search['all_final_model_cohort_search_pruning_independently_verified'] is True
    assert search['atol'] == search['rtol'] == 1e-8
    for key in ('old_model_search_or_forest_acceptance_inherited',
                'conditional_output_selector_independently_accepted',
                'complete_online_method_accepted','same_resource_performance_accepted','paper_performance_complete'):
        assert search[key] is False


def registered_receipt(path):
    ledger = R/'receipts/20260928-execution-ledger.json'
    with open(str(ledger)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        matches = [e for e in json.loads(ledger.read_bytes())['entries'] if e.get('receipt') == str(path)]
    assert len(matches) == 1 and matches[0]['receipt_sha256'] == sha(path), 'missing or changed completed-search registry'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--byte-admission', type=Path)
    parser.add_argument('--search-admission', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-source', action='store_true')
    args = parser.parse_args(); source_binding = source_gate()
    if args.check_source:
        assert args.byte_admission is args.search_admission is args.output is None
        print(json.dumps(dict(source_gate_passed=True,**source_binding)),flush=True); return
    assert args.byte_admission is not None and args.search_admission is not None and args.output is not None
    for path in (args.byte_admission,args.search_admission):
        assert path.resolve().is_relative_to(R/'artifacts') and not path.is_symlink()
    value = json.loads(args.byte_admission.read_bytes())
    job, _, model_binding, _ = validate_output(value)
    search = json.loads(args.search_admission.read_bytes())
    validate_local_search(search,value,model_binding,source_binding)
    assert search['byte_admission_sha256'] == sha(args.byte_admission)
    registered_receipt(args.search_admission)
    sequence_proofs = []
    for index, entry in enumerate(value['sequences']):
        path = args.search_admission.parent/f'sequence-{index:02d}.json'
        assert not path.is_symlink()
        proof = json.loads(path.read_bytes())
        oracle.validate_search_sequence(proof,entry['database_sha256'],entry['sequence_id'],entry['events'],entry['nodes'])
        sequence_proofs.append(dict(path=str(path),sha256=sha(path),sequence_id=entry['sequence_id']))
    assert not args.output.exists() and args.output.resolve().is_relative_to(R/'artifacts')
    assert not any(p.is_symlink() for p in (args.output,*args.output.parents))
    args.output.mkdir(parents=True)
    binding = dict(kind='rbf_final_refit_fixed_Top1_full_conditional_selector_admission_v1',
        method='topk',K=1,task_id=value['task_id'],seed=value['seed'],recipe_sha256=job['recipe_sha256'],
        byte_admission_sha256=sha(args.byte_admission),search_admission_path=str(args.search_admission),
        search_admission_sha256=sha(args.search_admission),admitted_search_sequence_receipts=sequence_proofs,
        final_model_binding=model_binding,source_binding=source_binding,
        upstream_same_byte_search_proofs_required=True,search_pruning_rerun=False,
        continuous_states_independently_accepted=False,complete_online_method_accepted=False,
        learned_Stage2_complete=False,same_resource_performance_accepted=False,paper_performance_complete=False)
    new(args.output/'binding.json',binding)
    started=time.monotonic();results=[]
    try:
        for index,entry in enumerate(value['sequences']):
            candidates=list(args.byte_admission.parent.glob(f'rank*-unpack/rank-*/{entry["sequence_id"]}/receipt.json'))
            assert len(candidates)==1
            receipt=json.loads(candidates[0].read_bytes());record=receipt['databases'][entry['sequence_id']]
            assert record['sha256']==entry['database_sha256']
            database=candidates[0].parent/record['path']
            assert database.resolve().is_relative_to(candidates[0].parent.resolve()) and not database.is_symlink()
            prerequisite=sequence_proofs[index];assert sha(prerequisite['path'])==prerequisite['sha256']
            def progress(data):
                if data['completed_events']==data['total_events'] or data['completed_events']%20==0:
                    print(json.dumps(dict(data,completed_sequences=index,total_sequences=46,
                        whole_cohort_ETA_seconds=None,whole_cohort_ETA_reason='unequal retained identity scopes')),flush=True)
            result=oracle.verify_database(database,entry['database_sha256'],json.loads(Path(prerequisite['path']).read_bytes()),progress)
            assert result['events']==entry['events'] and result['observations']==entry['nodes']
            new(args.output/f'sequence-{index:02d}.json',result);results.append(result)
        assert len(results)==46 and sum(v['events'] for v in results)==7445
        assert sum(v['observations'] for v in results)==value['total_nodes']
        assert source_gate()==source_binding and sha(args.search_admission)==binding['search_admission_sha256']
        for item in sequence_proofs:assert sha(item['path'])==item['sha256']
        final=dict(binding,completed_sequences=46,completed_events=7445,observations=value['total_nodes'],
            all_final_model_conditional_output_selectors_independently_verified=True,
            component_decisions=sum(v.get('component_decisions',0) for v in results),
            MAP_and_output_differ=sum(v.get('posterior_MAP_and_output_differ',0) for v in results),
            scoped_identity_queries=sum(v.get('scoped_identity_queries',0) for v in results),
            selector_sequence_receipts=[dict(path=str(args.output/f'sequence-{i:02d}.json'),sha256=sha(args.output/f'sequence-{i:02d}.json')) for i in range(46)],
            max_abs_error=max(max(v['max_abs_error'].values()) for v in results),atol=1e-8,rtol=1e-8,
            decimal_precision=70,elapsed_seconds=time.monotonic()-started,
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        new(args.output/'acceptance.json',final);register(args.output/'acceptance.json',final['kind'])
        print(json.dumps(final),flush=True)
    except BaseException as error:
        failure=dict(binding,kind='rbf_final_refit_fixed_Top1_selector_admission_failure_v1',
                     type=type(error).__name__,message=str(error),completed_sequences=len(results),experiment_accepted=False)
        new(args.output/'failure.json',failure);register(args.output/'failure.json',failure['kind']);raise


if __name__ == '__main__':main()
