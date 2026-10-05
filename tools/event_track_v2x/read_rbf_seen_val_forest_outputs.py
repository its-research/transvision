"""Read all seen-val forest output bytes, events and admitted model factors.

This gate is not a fresh-state, full forest, resource or metric acceptance.
Transfers resume exact registered bytes; failures and prior receipts persist.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import tarfile

from rbf_nested_seen_val_v2_common import R,new,register,sha
import rbf_seen_val_forest_output_binding as binding


def helpers():
    assert sha(binding.ORIGINAL_READER) == binding.ORIGINAL_READER_SHA
    spec=importlib.util.spec_from_file_location('unchanged_train_byte_reader_helpers',binding.ORIGINAL_READER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def local_artifact(root,key):
    suffix='.tar.gz' if key.startswith('replay-rank') else '.json'
    return root/(key+suffix)


def verify_extracted_archive(path,destination):
    """Bind reused extraction contents to archive bytes, not just a marker."""
    names=set();regular=set()
    with tarfile.open(path,'r:gz') as archive:
        for member in archive:
            assert member.name not in names;names.add(member.name)
            relative=Path(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert member.isfile() or member.isdir()
            assert not member.issym() and not member.islnk()
            if member.isdir():continue
            regular.add(relative.as_posix())
            local=binding.contained(destination,relative)
            assert local.stat().st_size==member.size
            digest=hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
            assert sha(local)==digest.hexdigest()
    entries=list(destination.rglob('*'))
    assert all(not p.is_symlink() for p in entries)
    actual={p.relative_to(destination).as_posix() for p in entries if p.is_file()}
    assert actual==regular|{'archive-byte-binding.json'}
    assert json.loads((destination/'archive-byte-binding.json').read_bytes())['archive_sha256']==sha(path)


def check_sequence(directory,sequence,events,origin,references,plan,normalized_factors):
    receipt_path=binding.contained(directory,'receipt.json')
    receipt=json.loads(receipt_path.read_bytes())
    per_plan=json.loads(binding.contained(directory,'plan.json').read_bytes())
    binding.verify_sequence(plan,per_plan,receipt,sequence,events)
    assert receipt['files']['plan.json']==sha(directory/'plan.json')
    assert set(receipt['databases'])=={sequence}
    for name,digest in receipt['files'].items():assert sha(binding.contained(directory,name))==digest
    database=receipt['databases'][sequence]
    path=binding.contained(directory,database['path']);assert sha(path)==database['sha256']
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        commits=db.execute('SELECT event_id,prediction,audit FROM events ORDER BY ordinal').fetchall()
        assert len(commits)==len(events)
        with (directory/'predictions.jsonl').open('rb') as predictions,(directory/'audit.jsonl').open('rb') as audits:
            count=0
            for event,(event_id,prediction,audit),p,a in zip(events,commits,predictions,audits):
                assert event_id==event['event_id'] and p.rstrip(b'\n')==prediction and a.rstrip(b'\n')==audit
                value=json.loads(audit)
                assert value['cache_ingestion']['new_deliveries']==event['deliveries']
                assert value['cache_ingestion'].get('old_rows_rescored',False) is False
                assert value['explicit_residual_partition'] is True and value['residual_partition_version']==1
                assert all(c['representation']=='exclusive_root_partition_regions_v1' for c in value['components'])
                count+=1
            assert count==len(commits) and not predictions.read(1) and not audits.read(1)
        count=0;maximum=0.
        for i,node,raw,digest in db.execute('SELECT i,node_id,raw,sha FROM observations ORDER BY i'):
            assert i==count and i<len(references) and hashlib.sha256(raw).hexdigest()==digest
            reference=references[i];observation=json.loads(raw)
            assert reference['row']==i and reference['node_id']==node
            assert observation['features'][141]==(observation['state_us']-origin)/1e8
            assert observation['node']['information_us']<=observation['node']['arrival_us']<=reference['decision_us']
            potentials=db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p',(i,)).fetchall()
            assert [p for p,w in potentials]==[-1]+reference['context_indices'][:-1]
            values=normalized_factors(tuple(float(v) for v in reference['logits']))
            assert len(potentials)==len(values)
            for (_,actual),value in zip(potentials,values):
                assert math.isfinite(actual) and abs(actual-value)<=1e-4+1e-4*abs(value)
                maximum=max(maximum,abs(actual-value))
            count+=1
        assert count==len(references)
        return dict(sequence_id=sequence,events=len(commits),nodes=count,database_sha256=database['sha256'],
            max_factor_difference_to_admitted_seen_val_forward=maximum)
    finally:db.close()


def read_job(args,job,task,Task):
    plan=job['plan'];world=plan['world_size'];status=str(task.status)
    assert plan['method']=='rbf' and plan['configuration']['allocation']=='bound'
    assert hashlib.sha256(binding.canonical(plan)).hexdigest()==job['recipe_sha256']
    if status not in ('completed','failed'):
        print(json.dumps(dict(seed=args.seed,task_id=task.id,status=status,no_readback_started=True,ETA='unknown')),flush=True);return
    published=binding.qualify_job(args,Task,job,task)
    root=R/f'artifacts/rbf-seen-val-bound-forest-independent-byte-factor-v1-20261005/seed{args.seed}'
    assert not any(p.is_symlink() for p in (root,*root.parents))
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.reader.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected_keys=binding.artifact_keys(world)
        assert 'receipt' in task.artifacts and set(task.artifacts)<=expected_keys
        accepted=root/'independent-byte-coverage-factor-admission.json'
        failed=root/'independent-failure-byte-readback.json'
        old=accepted if accepted.exists() else failed if failed.exists() else None
        if old is not None:
            proof=json.loads(old.read_bytes())
            assert proof['kind'] in (binding.KIND,'rbf_seen_val_bound_forest_failed_candidate_independent_bytes_v1')
            assert proof['source_sha256']==sha(__file__) and proof['task_id']==task.id and proof['recipe_sha256']==job['recipe_sha256']
            assert proof['publication_sha256']==sha(args.publication)
            for key,item in proof['artifacts'].items():
                p=local_artifact(root,key);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
            binding.verify_task_unchanged(task,job,proof['artifacts'],status)
            print(json.dumps(dict(task_id=task.id,existing_receipt=str(old),not_repeated_or_overwritten=True)),flush=True);return
        helper=helpers()
        inventory={}
        for key in sorted(task.artifacts):
            path=local_artifact(root,key)
            assert not path.is_symlink() and not path.with_name(path.name+'.partial').is_symlink()
            inventory[key]=helper.read_artifact(task,key,path)
        report=json.loads((root/'receipt.json').read_bytes())
        assert report['kind']=='rbf_final_refit_SPD_seen_val_bound_forest_candidate_v1'
        actual=dict(report['plan']);actual.pop('cache_relative_root',None)
        assert actual==plan and report['task_id']==task.id
        assert report['paper_performance_complete'] is report['same_resource_baseline_comparison_accepted'] is False
        if status=='failed':
            binding.verify_task_unchanged(task,job,inventory,status)
            value=dict(kind='rbf_seen_val_bound_forest_failed_candidate_independent_bytes_v1',
                seed=args.seed,task_id=task.id,recipe_sha256=job['recipe_sha256'],artifacts=inventory,
                source_sha256=sha(__file__),publication_sha256=sha(args.publication),producer_failure=report['failure'],
                rank_sequences=report['ranks'],automatic_retry_permitted=False,experiment_accepted=False)
            new(failed,value);register(failed,value['kind'])
            print(json.dumps(dict(task_id=task.id,failure_receipt=str(failed),failure_preserved=True)),flush=True);return
        assert set(inventory)==expected_keys and report['failure'] is None
        binding.verify_report(plan,report,json.loads((root/'seen-val-input-binding.json').read_bytes()),published)
        source=json.loads((root/'exclusive-source-manifest.json').read_bytes())
        for a,b in (('patches','exclusive_patches'),('configuration','configuration'),('original_source','source'),
                    ('bootstrap_sha256','bootstrap_sha256'),('source_replacements','source_replacements'),
                    ('CPU_capacity_candidate_admission','CPU_capacity_candidate_admission')):
            assert source[a]==plan[b]
        entry,checkpoint,expected,events,groups=binding.reference_inputs(args.seed,plan)
        checked=[]
        for rank in range(world):
            helper.unpack(root/f'replay-rank{rank}.tar.gz',root/f'rank{rank}-unpack')
            verify_extracted_archive(root/f'replay-rank{rank}.tar.gz',root/f'rank{rank}-unpack')
            rank_root=root/f'rank{rank}-unpack/rank-{rank}'
            sequences=sorted(groups)[rank::world]
            assert {p.parent.name for p in rank_root.glob('*/receipt.json')}==set(sequences)
            rank_report=next(r for r in report['ranks'] if r['rank']==rank)
            assert [v['sequence_id'] for v in rank_report['sequences']]==sequences
            assert rank_report['events_committed']==sum(len(groups[sid]) for sid in sequences)
            for sequence in sequences:
                checked.append(check_sequence(rank_root/sequence,sequence,groups[sequence],
                    events['origin_us_by_sequence'][sequence],expected[sequence],plan,helper.normalized_factors))
                print(json.dumps(dict(stage='independent_seen_val_bytes_events_factors',completed_sequences=len(checked),
                    total_sequences=21,ETA_seconds=None,ETA_reason='heterogeneous SQLite and archive costs')),flush=True)
        assert len(checked)==21 and len({x['sequence_id'] for x in checked})==21
        assert sum(x['events'] for x in checked)==3316 and sum(x['nodes'] for x in checked)==entry['rows']
        binding.source_gate()
        assert binding.qualify_job(args,Task,job,task)==published
        binding.verify_task_unchanged(task,job,inventory,status)
        value=dict(kind=binding.KIND,task_id=task.id,seed=args.seed,method='rbf',recipe_sha256=job['recipe_sha256'],
            artifacts=inventory,all_registered_bytes_verified=True,all_21_sequences_3316_events_and_final_model_factor_nodes_verified=True,
            sequences=checked,total_nodes=entry['rows'],world_size=world,final_checkpoint_sha256=plan['checkpoint']['sha256'],
            final_model_sha256=checkpoint['model_sha256'],input_numeric_index_sha256=sha(binding.INDEX),
            publication_sha256=sha(args.publication),source_sha256=sha(__file__),NN_atol=1e-4,NN_rtol=1e-4,
            scope=binding.publication.SCOPE,measured_network_arrival_history_verified=False,
            full_forest_semantics_or_fresh_state_independently_accepted=False,learned_Stage2_complete=False,
            same_resource_performance_accepted=False,paper_performance_complete=False,
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        new(accepted,value);register(accepted,binding.KIND)
        print(json.dumps(dict(task_id=task.id,receipt=str(accepted),bytes_events_factors_accepted=True,
            full_forest_independent_semantics_and_states_pending=True)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);binding.arguments(parser);args=parser.parse_args()
    binding.source_gate()
    path=binding.dispatch.JOURNAL
    jobs=json.loads(path.read_bytes())['jobs'] if path.exists() else []
    matches=[j for j in jobs if j['seed']==args.seed];assert len(matches)<=1
    if not matches:
        print(json.dumps(dict(seed=args.seed,status='no_seen_val_forest_attempt_dispatched',no_readback_started=True,ETA='unknown')),flush=True);return
    from clearml import Task
    read_job(args,matches[0],Task.get_task(task_id=matches[0]['task_id']),Task)


if __name__=='__main__':main()
