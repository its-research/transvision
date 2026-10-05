"""Derive a separate learned output reader while preserving the raw-factor audit.

No producer execution, source upload or remote task is performed here. The
result only admits bytes, original causal event coverage and fixed NN factors.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

PARENT = Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004/read_rbf_final_refit_forest_outputs.py')
PARENT_SHA = '53747489a446c8990cfc6d9310eb686a69b39d99bde226c8945665a974d83f61'


def build(parent):
    assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
    changes = [
        ('Independent bytes, full event coverage and source-bound raw-factor readback.',
         'Learned-priority output bytes, full event coverage and fixed raw-factor readback.'),
        ('import datetime\n', 'import datetime\nimport fcntl\n'),
        ('INDEX=R/', 'from rbf_final_refit_learned_output_binding import (arguments, source_gate, qualify_job, artifact_keys, contained, verify_priority_report, verify_priority_sequence, verify_task_unchanged, KIND)\n\nINDEX=R/'),
        ("    parser.add_argument('--seed',type=int,required=True,choices=(1337,2027,3407))\n    parser.add_argument('--method',required=True,choices=('rbf','topk'))\n    args=parser.parse_args()",
         "    arguments(parser)\n    args=parser.parse_args();args.method='rbf'\n    control=source_gate()"),
        ("    variant='exclusive-forest' if args.method=='rbf' else 'fixed-topK'\n    journal=R/f'receipts/rbf-final-refit-full-train-{variant}-GPU-dispatch-20261004.json'\n    job=next(v for v in json.loads(journal.read_bytes())['jobs'] if v['seed']==args.seed)",
         "    journal=R/'receipts/rbf-final-refit-learned-priority-full-train-GPU-dispatch-20261005.json'\n    jobs=json.loads(journal.read_bytes())['jobs'] if journal.exists() else []\n    matches=[v for v in jobs if v['seed']==args.seed]\n    assert len(matches)<=1\n    if not matches:\n        print(json.dumps(dict(seed=args.seed,status='no_learned_attempt_dispatched',no_readback_started=True,experiment_accepted=False,ETA='unknown')))\n        return\n    job=matches[0]"),
        ("    root=R/f'artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/{args.method}/seed{args.seed}'",
         "    publication_proof=qualify_job(args,Task,job,task,control)\n    root=R/f'artifacts/rbf-final-refit-learned-priority-independent-byte-factor-v1-20261005/seed{args.seed}'\n    assert not any(p.is_symlink() for p in (root,*root.parents))"),
        ("    expected_keys={'receipt',*(f'replay-rank{i}' for i in range(world))}\n    if args.method=='rbf':\n        expected_keys.add('exclusive-source-manifest')",
         "    reader_lock=(root/'.reader.lock').open('a')\n    fcntl.flock(reader_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    expected_keys=artifact_keys(world)"),
        ("        assert proof['task_id']==task.id and proof['recipe_sha256']==job['recipe_sha256']",
         "        assert proof['task_id']==task.id and proof['recipe_sha256']==job['recipe_sha256']\n        assert proof['kind'] in (KIND,'rbf_final_refit_learned_priority_failed_candidate_independent_bytes_v1')\n        assert proof['source_sha256']==sha(__file__)\n        assert proof['priority_publication_sha256']==sha(args.priority_publication)"),
        ("            assert task.artifacts[key].hash==spec['sha256'] and task.artifacts[key].size==spec['bytes']",
         "            assert task.artifacts[key].hash==spec['sha256'] and task.artifacts[key].size==spec['bytes']\n            local=root/(key+('.json' if key in ('receipt','exclusive-source-manifest','priority-input-binding') else '.tar.gz'))\n            assert local.stat().st_size==spec['bytes'] and sha(local)==spec['sha256']"),
        ("root/(key+('.json' if key in ('receipt','exclusive-source-manifest') else '.tar.gz'))",
         "root/(key+('.json' if key in ('receipt','exclusive-source-manifest','priority-input-binding') else '.tar.gz'))"),
        ("    assert report['kind']==('rbf_final_refit_exclusive_full_train_forest_candidate_v1' if args.method=='rbf'\n        else 'rbf_final_refit_full_train_fixed_topK_candidate_v1')",
         "    assert report['kind']=='rbf_final_refit_learned_priority_full_train_forest_candidate_v1'"),
        ("    if status=='failed':\n", "    if status=='failed':\n        verify_task_unchanged(task,job,inventory,status)\n"),
        ("kind='rbf_final_refit_forest_failed_candidate_independent_bytes_v1'",
         "kind='rbf_final_refit_learned_priority_failed_candidate_independent_bytes_v1'"),
        ("registered_failure_and_partial_output_bytes_verified=True,experiment_accepted=False,paper_performance_complete=False))",
         "registered_failure_and_partial_output_bytes_verified=True,source_sha256=sha(__file__),priority_publication_sha256=sha(args.priority_publication),experiment_accepted=False,paper_performance_complete=False))"),
        ("register(failure_path,'rbf_final_refit_forest_failed_candidate_independent_bytes_v1')",
         "register(failure_path,'rbf_final_refit_learned_priority_failed_candidate_independent_bytes_v1')"),
        ("    assert set(inventory)==expected_keys and report['failure'] is None",
         "    assert set(inventory)==expected_keys and report['failure'] is None\n    verify_priority_report(plan,report,json.loads((root/'priority-input-binding.json').read_bytes()),publication_proof)"),
        ("            assert per_plan['configuration']==plan['configuration'] and per_plan['fixture'] is False",
         "            verify_priority_sequence(plan,per_plan,receipt,sequence,len(groups[sequence]))\n            assert receipt['files']['plan.json']==sha(directory/'plan.json')\n            assert per_plan['configuration']==plan['configuration'] and per_plan['fixture'] is False"),
        ("assert sha(directory/name)==digest", "assert sha(contained(directory,name))==digest"),
        ("            assert sha(directory/database['path'])==database['sha256']\n            db=sqlite3.connect((directory/database['path']).resolve().as_uri()+'?mode=ro',uri=True)",
         "            database_path=contained(directory,database['path'])\n            assert sha(database_path)==database['sha256']\n            db=sqlite3.connect(database_path.resolve().as_uri()+'?mode=ro',uri=True)"),
        ("    new(accepted_path,dict(kind='rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1',",
         "    source_gate()\n    assert qualify_job(args,Task,job,task,control)==publication_proof\n    verify_task_unchanged(task,job,inventory,status)\n    new(accepted_path,dict(kind=KIND,priority_publication_sha256=sha(args.priority_publication),\n        priority_policy_signature=plan['priority_policy_signature'],priority_checkpoint_sha256=plan['priority_inputs']['checkpoint']['sha256'],\n        all_rank_and_sequence_priority_identities_verified=True,learned_expansion_order_independently_verified=False,"),
        ("register(accepted_path,'rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1')", "register(accepted_path,KIND)"),
    ]
    source=parent;edits=[]
    for before,after in changes:
        assert source.count(before)==1, 'frozen reader changed: '+before
        source=source.replace(before,after,1)
        edits.append(dict(before=before,after=after))
    compile(source,'<learned-independent-byte-factor-reader>','exec')
    before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef)}
    after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    unchanged=['canonical','read_artifact','unpack','normalized_factors']
    assert all(before[k]==after[k] for k in unchanged)
    original=parent[parent.index('                count=0;maximum=0.'):parent.index('            finally:')]
    assert source.count(original)==1
    return source,dict(parent_reader_sha256=PARENT_SHA,edits=edits,unchanged_function_AST=unchanged,
                       observation_factor_comparison_block_identical=True,NN_atol=1e-4,NN_rtol=1e-4)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    source,control=build(PARENT.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'read_final_refit_learned_outputs.py').write_text(source)
    (args.output/'source-control.json').write_text(json.dumps(control,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),bytes=len(source.encode()),sha256=hashlib.sha256(source.encode()).hexdigest(),actual_experiment_accepted=False)))


if __name__=='__main__':main()
