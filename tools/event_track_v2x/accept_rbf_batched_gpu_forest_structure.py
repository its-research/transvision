"""Follow byte/factor/state admission with independent selected-sequence oracles."""
import argparse
import importlib.util
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R,new,register,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--admission',type=Path,required=True)
    args=parser.parse_args();path=args.admission.resolve()
    expected_root=R/'artifacts/rbf-batched-GPU-complete-sequence-independent-v1-20261004'
    assert path.is_relative_to(expected_root) and path.name=='independent-candidate-receipt.json'
    own=Path(__file__).resolve().parent;freeze=json.loads((own/'source-freeze.json').read_bytes())
    for name,record in freeze['sources'].items():assert sha(own/name)==record['sha256']
    for record in freeze['references']:assert sha(record['path'])==record['sha256']
    admitted=json.loads(path.read_bytes());assert admitted['kind']=='rbf_batched_GPU_selected_complete_sequences_independent_v1'
    assert admitted['task_id']==path.parent.name and admitted['structure_action_independent_acceptance_pending'] is True
    assert admitted['production_promotion_allowed'] is admitted['full_cohort_or_three_seed_acceptance'] is False
    report=json.loads((path.parent/'receipt.json').read_bytes());assert report['task_id']==admitted['task_id']
    assert sha(path.parent/'receipt.json')==admitted['artifacts']['receipt']['sha256']
    plan=report['plan'];assert len(admitted['outcomes'])==plan['world_size'] and plan['world_size'] in (4,8)
    pair_reader=Path(freeze['module_loader']['path'])
    spec=importlib.util.spec_from_file_location('batch_independent_module_loader',pair_reader)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    def read_oracle(key,dependencies=None):
        record=freeze['oracles'][key]
        return loader.module('batch_GPU_'+key,Path(record['path']),record['sha256'],dependencies)
    structure=read_oracle('structure');causal=read_oracle('causal',{'oracle':structure});action=read_oracle('action')
    destination=path.parent/'independent-structure-action'
    destination.mkdir(exist_ok=False)
    results=[]
    try:
        for outcome in admitted['outcomes']:
            rank=outcome['rank'];sequence=outcome['sequence'];events=outcome['events']
            disk=json.loads((path.parent/f'rank-{rank}-independent.json').read_bytes());assert disk==outcome
            directory=path.parent/f'rank{rank}-unpack/rank-{rank}'/sequence
            receipt=json.loads((directory/'receipt.json').read_bytes())
            dbinfo=receipt['databases'][sequence];database=(directory/dbinfo['path']).resolve()
            assert database.is_relative_to(directory.resolve()) and sha(database)==dbinfo['sha256']
            assert dbinfo['sha256']==outcome['fresh_state']['database_sha256']
            assert receipt['completed_events']==events and receipt['completed_sequences']==[sequence]
            def progress(value):
                print(json.dumps(dict(rank=rank,sequence=sequence,oracle_progress=value,ETA='unknown')),flush=True)
            result=dict(rank=rank,sequence=sequence,database_sha256=dbinfo['sha256'],
                structure=structure.audit_database(database,dbinfo['sha256'],progress),
                causal=causal.verify_database(database,dbinfo['sha256']),
                action=action.verify_database(database,dbinfo['sha256'],progress))
            assert all(result[key]['events']==events for key in ('structure','causal','action'))
            new(destination/f'rank-{rank}.json',result);results.append(result)
        final=destination/'receipt.json'
        new(final,dict(kind='rbf_batched_GPU_selected_sequences_structure_causal_action_independent_v1',
            task_id=admitted['task_id'],input_admission=str(path),input_admission_sha256=sha(path),
            source_freeze_sha256=sha(own/'source-freeze.json'),results=results,
            selected_complete_sequences_checked=True,full_46_sequence_three_seed_acceptance=False,
            GPU_memory_target_admitted=False,production_promotion_allowed=False,
            same_resource_or_paper_performance_complete=False))
        register(final,'rbf-batched-GPU-selected-sequence-structure-action-independent')
        print(json.dumps(dict(receipt=str(final))))
    except BaseException as error:
        failure=destination/'failure.json'
        new(failure,dict(error_type=type(error).__name__,message=str(error),input_admission_sha256=sha(path),
            completed_ranks=[r['rank'] for r in results],automatic_retry_allowed=False,accepted=False))
        register(failure,'rbf-batched-GPU-structure-action-independent-failure');raise


if __name__=='__main__':main()
