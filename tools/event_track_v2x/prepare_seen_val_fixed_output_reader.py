"""Derive and freeze the fixed K1/K4 val byte reader; never run experiments."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

from rbf_nested_seen_val_v2_common import R,new,register,sha
from rbf_seen_val_fixed_output_binding import DISPATCH,DISPATCH_SHA,SHARED,SHARED_SHA,SOURCE,SOURCE_SHA

NAME='rbf-seen-val-fixed-K1-K4-independent-output-reader-v1-20261005'


def derive():
    assert sha(SHARED/'source-freeze.json')==SHARED_SHA
    frozen=json.loads((SHARED/'source-freeze.json').read_bytes())
    path=SHARED/'read_rbf_seen_val_forest_outputs.py'
    assert sha(path)==frozen['sources'][path.name]['sha256']
    parent=path.read_text()
    edits=[
        ('Read all seen-val forest output bytes, events and admitted model factors.',
         'Read complete fixed K1/K4 seen-val bytes, events and admitted model factors.'),
        ('import rbf_seen_val_forest_output_binding as binding','import rbf_seen_val_fixed_output_binding as binding'),
        ('sha(binding.ORIGINAL_READER) == binding.ORIGINAL_READER_SHA','sha(binding.shared.ORIGINAL_READER) == binding.shared.ORIGINAL_READER_SHA'),
        ("'unchanged_train_byte_reader_helpers',binding.ORIGINAL_READER","'unchanged_train_byte_reader_helpers',binding.shared.ORIGINAL_READER"),
        ('references,plan,normalized_factors):','references,plan,normalized_factors,sources):'),
        ('binding.verify_sequence(plan,per_plan,receipt,sequence,events)','binding.verify_sequence(plan,per_plan,receipt,sequence,events,sources)'),
        ("                assert value['explicit_residual_partition'] is True and value['residual_partition_version']==1\n                assert all(c['representation']=='exclusive_root_partition_regions_v1' for c in value['components'])",
         "                binding.verify_audit(value,plan['baseline_K'])"),
        ("    assert plan['method']=='rbf' and plan['configuration']['allocation']=='bound'",
         "    assert plan['method']=='topk' and plan['configuration']['allocation']=='bound'\n    assert args.K==job['K']==plan['baseline_K'] and args.K in (1,4)\n    assert args.seed==job['seed']==plan['seed']"),
        ('rbf-seen-val-bound-forest-independent-byte-factor-v1-20261005/seed{args.seed}',
         'rbf-seen-val-fixed-K{args.K}-independent-byte-factor-v1-20261005/seed{args.seed}'),
        ("(binding.KIND,'rbf_seen_val_bound_forest_failed_candidate_independent_bytes_v1')",
         "(binding.KIND,binding.FAILED_KIND)\n            assert proof['K']==args.K and proof['seed']==args.seed"),
        ("assert report['kind']=='rbf_final_refit_SPD_seen_val_bound_forest_candidate_v1'",
         "assert report['kind']==f'rbf_final_refit_seen_val_fixed_K{args.K}_candidate_v1'"),
        ("kind='rbf_seen_val_bound_forest_failed_candidate_independent_bytes_v1',\n                seed=args.seed",
         "kind=binding.FAILED_KIND,K=args.K,\n                seed=args.seed"),
        ("binding.verify_report(plan,report,json.loads((root/'seen-val-input-binding.json').read_bytes()),published)",
         "binding.verify_report(plan,report,json.loads((root/'seen-val-input-binding.json').read_bytes()),\n            json.loads((root/'seen-val-baseline-binding.json').read_bytes()),published)"),
        ("        source=json.loads((root/'exclusive-source-manifest.json').read_bytes())\n        for a,b in (('patches','exclusive_patches'),('configuration','configuration'),('original_source','source'),\n                    ('bootstrap_sha256','bootstrap_sha256'),('source_replacements','source_replacements'),\n                    ('CPU_capacity_candidate_admission','CPU_capacity_candidate_admission')):\n            assert source[a]==plan[b]",
         '        sources=binding.source_map(plan)'),
        ('expected[sequence],plan,helper.normalized_factors))','expected[sequence],plan,helper.normalized_factors,sources))'),
        ("task_id=task.id,seed=args.seed,method='rbf',recipe_sha256", "task_id=task.id,seed=args.seed,method='topk',K=args.K,recipe_sha256"),
        ('sha(binding.INDEX)','sha(binding.shared.INDEX)'),
        ('scope=binding.publication.SCOPE','scope=plan[\'evaluation_scope\']'),
        ('full_forest_semantics_or_fresh_state_independently_accepted=False,learned_Stage2_complete=False,',
         'full_baseline_semantics_or_fresh_state_independently_accepted=False,learned_Stage2_complete=False,\n            main_or_train_full_forest_acceptance_inherited=False,source_package_sha256=binding.SOURCE_SHA,'),
        ("matches=[j for j in jobs if j['seed']==args.seed];assert len(matches)<=1",
         "matches=[j for j in jobs if j['seed']==args.seed and j['K']==args.K];assert len(matches)<=1"),
        ("status='no_seen_val_forest_attempt_dispatched'","K=args.K,status='no_seen_val_fixed_baseline_attempt_dispatched'"),
    ]
    source=parent
    for before,after in edits:
        assert source.count(before)==1,(before,source.count(before))
        source=source.replace(before,after,1)
    recovered=source
    for before,after in reversed(edits):
        assert recovered.count(after)==1,after
        recovered=recovered.replace(after,before,1)
    assert recovered==parent
    ast.parse(source)
    return source,dict(parent=str(path),parent_sha256=sha(path),exact_reversible_edits=len(edits),
        math_and_transfer_helpers_unchanged=True,edits=[dict(before=a,after=b) for a,b in edits])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--software-log',type=Path,required=True)
    args=parser.parse_args()
    out=R/'source-freezes'/NAME;assert not out.exists()
    log=args.software_log.read_text()
    assert '56 passed' in log and 'FAILED' not in log and 'ERROR' not in log
    assert sha(DISPATCH/'source-freeze.json')==DISPATCH_SHA and sha(SOURCE)==SOURCE_SHA
    own=Path(__file__).resolve().parent;repo=own.parents[1]
    source,control=derive()
    assert source==(own/'read_rbf_seen_val_fixed_outputs.py').read_text()
    paths={};refs={str(DISPATCH/'source-freeze.json'):DISPATCH_SHA,str(SHARED/'source-freeze.json'):SHARED_SHA,str(SOURCE):SOURCE_SHA}
    for root in (DISPATCH,SHARED):
        freeze=json.loads((root/'source-freeze.json').read_bytes())
        refs.update({item['path']:item['sha256'] for item in freeze['references']})
        for name,item in freeze['sources'].items():
            assert sha(root/name)==item['sha256']
            refs[str(root/name)]=item['sha256']
            if '/' not in name and name.endswith('.py') and not name.startswith(('test_','prepare_','read_')):
                if name in paths:assert sha(paths[name])==item['sha256']
                paths[name]=root/name
    paths['prepare_seen_val_fixed_baselines.py']=DISPATCH/'prepare_seen_val_fixed_baselines.py'
    for name in ('rbf_seen_val_fixed_output_binding.py','prepare_seen_val_fixed_output_reader.py','read_rbf_seen_val_fixed_outputs.py'):
        paths[name]=own/name
    for name in ('test_seen_val_fixed_output_reader.py','test_seen_val_forest_output_reader.py'):
        paths['tests/event_track_v2x/'+name]=repo/'tests/event_track_v2x'/name
    # The shared fixture imports its reader for pure helper access, not execution.
    paths['read_rbf_seen_val_forest_outputs.py']=SHARED/'read_rbf_seen_val_forest_outputs.py'
    for path in paths.values():ast.parse(path.read_text())
    for path,checksum in refs.items():assert sha(path)==checksum
    out.mkdir()
    for relative,path in paths.items():
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    shutil.copyfile(args.software_log,out/'software-tests.log')
    shutil.copyfile(repo/'docs/recover-before-fuse/remaining-experiment-code-preparation-20261004.md',out/'remaining-experiments.md')
    new(out/'source-control.json',control)
    value=dict(kind='rbf_seen_val_fixed_baseline_output_reader_source_v1',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()},
        references=[dict(path=p,sha256=h) for p,h in sorted(refs.items())],
        software_scope='fixed-width provenance, exact schedule/bytes/factors and corrupted-output rejection; no real val forest admission',
        software_tests=56,
        actual_GPU_tasks_created=0,actual_val_baseline_outputs_accepted=False,original_GPU_producers_unchanged=True,
        full_21_sequences_3316_events_required=True,NN_atol=1e-4,NN_rtol=1e-4,
        independent_full_baseline_CPU_required=True,full_Stage2_complete=False,paper_performance_complete=False)
    new(out/'source-freeze.json',value);register(out/'source-freeze.json',value['kind'])
    print(json.dumps(dict(source_freeze=str(out/'source-freeze.json'),sha256=sha(out/'source-freeze.json'))),flush=True)


if __name__=='__main__':main()
