"""Independent bytes, full event coverage and source-bound raw-factor readback.

This admits neither fresh branch states nor forest semantics, resource parity,
physical recovery or paper metrics. A completed producer is only the first gate.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tarfile
import time
from urllib.parse import urlsplit

import requests

from rbf_nested_seen_val_v2_common import R,new,register,sha

INDEX=R/'receipts/rbf-final-refit-three-seed-all-row-full-independent-numeric-acceptance-20261004.json'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def read_artifact(task,key,destination):
    from clearml.backend_api.session import Session
    artifact=task.artifacts[key]
    if destination.exists():
        assert destination.stat().st_size==artifact.size and sha(destination)==artifact.hash
        return dict(sha256=artifact.hash,bytes=artifact.size)
    partial=destination.with_name(destination.name+'.partial')
    offset=partial.stat().st_size if partial.exists() else 0
    assert 0<=offset<=artifact.size
    digest=hashlib.sha256()
    if offset:
        with partial.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''):
                digest.update(block)
    url=urlsplit(artifact.url)
    assert url.scheme=='http' and url.netloc in ('10.100.34.118:8081','10.100.35.118:8081')
    assert not url.username and not url.password
    done=offset
    if done<artifact.size:
        headers={'Authorization':'Bearer '+Session().token}
        if offset:
            headers['Range']='bytes='+str(offset)+'-'
        start=last=time.monotonic()
        with requests.get(url._replace(netloc='10.100.34.118:8081').geturl(),headers=headers,
            timeout=(10,120),stream=True,allow_redirects=False) as response:
            assert response.status_code==(206 if offset else 200)
            if offset:
                assert response.headers.get('Content-Range')==f'bytes {offset}-{artifact.size-1}/{artifact.size}'
            with partial.open('ab' if partial.exists() else 'xb') as stream:
                for block in response.iter_content(1024**2):
                    if not block:
                        continue
                    done+=len(block)
                    assert done<=artifact.size
                    stream.write(block);digest.update(block)
                    now=time.monotonic()
                    if now-last>=30:
                        print(json.dumps(dict(stage='independent_final_forest_artifact_read',artifact=key,
                            completed_bytes=done,total_bytes=artifact.size,ETA_seconds=(now-start)*(artifact.size-done)/(done-offset),
                            ETA_scope='artifact transfer only; whole admission unknown')),flush=True)
                        last=now
    elif not partial.exists():
        partial.touch(exist_ok=False)
    assert done==artifact.size and digest.hexdigest()==artifact.hash
    partial.rename(destination)
    return dict(sha256=artifact.hash,bytes=artifact.size)


def unpack(path,destination):
    if destination.exists():
        proof=json.loads((destination/'archive-byte-binding.json').read_bytes())
        assert proof['archive_sha256']==sha(path)
        return
    temporary=destination.with_name(destination.name+'.partial')
    assert not temporary.exists(), 'interrupted extraction preserved; inspect it before resuming'
    temporary.mkdir()
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        assert len({m.name for m in members})==len(members)
        assert all((m.isdir() or m.isfile()) and not m.issym() and not m.islnk()
            and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        archive.extractall(temporary,filter='data')
    new(temporary/'archive-byte-binding.json',dict(archive_sha256=sha(path),members=len(members)))
    temporary.rename(destination)


def normalized_factors(values):
    assert values and all(math.isfinite(v) for v in values)
    maximum=max(values)
    total=maximum+math.log(sum(math.exp(v-maximum) for v in values))
    return tuple(v-total for v in values)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True,choices=(1337,2027,3407))
    parser.add_argument('--method',required=True,choices=('rbf','topk'))
    args=parser.parse_args()
    from clearml import Task
    variant='exclusive-forest' if args.method=='rbf' else 'fixed-topK'
    journal=R/f'receipts/rbf-final-refit-full-train-{variant}-GPU-dispatch-20261004.json'
    job=next(v for v in json.loads(journal.read_bytes())['jobs'] if v['seed']==args.seed)
    plan=job['plan'];world=plan['world_size']
    assert plan['method']==args.method and world in (4,8)
    assert hashlib.sha256(canonical(plan)).hexdigest()==job['recipe_sha256']
    task=Task.get_task(task_id=job['task_id'])
    status=str(task.status)
    if status not in ('completed','failed'):
        print(json.dumps(dict(seed=args.seed,method=args.method,task_id=task.id,status=status,
            no_readback_started=True,experiment_accepted=False,ETA='unknown')))
        return
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==plan['bootstrap_sha256']
    root=R/f'artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/{args.method}/seed{args.seed}'
    root.mkdir(parents=True,exist_ok=True)
    expected_keys={'receipt',*(f'replay-rank{i}' for i in range(world))}
    if args.method=='rbf':
        expected_keys.add('exclusive-source-manifest')
    assert 'receipt' in task.artifacts and set(task.artifacts)<=expected_keys
    accepted_path=root/'independent-byte-coverage-factor-admission.json'
    failure_path=root/'independent-failure-byte-readback.json'
    old_path=accepted_path if accepted_path.exists() else failure_path if failure_path.exists() else None
    if old_path is not None:
        proof=json.loads(old_path.read_bytes())
        assert proof['task_id']==task.id and proof['recipe_sha256']==job['recipe_sha256']
        assert set(proof['artifacts'])==set(task.artifacts)
        for key,spec in proof['artifacts'].items():
            assert task.artifacts[key].hash==spec['sha256'] and task.artifacts[key].size==spec['bytes']
        print(json.dumps(dict(task_id=task.id,existing_receipt=str(old_path),not_repeated_or_overwritten=True)))
        return
    inventory={key:read_artifact(task,key,root/(key+('.json' if key in ('receipt','exclusive-source-manifest') else '.tar.gz')))
        for key in sorted(task.artifacts)}
    report=json.loads((root/'receipt.json').read_bytes())
    assert report['kind']==('rbf_final_refit_exclusive_full_train_forest_candidate_v1' if args.method=='rbf'
        else 'rbf_final_refit_full_train_fixed_topK_candidate_v1')
    actual=dict(report['plan']);actual.pop('cache_relative_root',None)
    assert actual==plan and report['task_id']==task.id
    assert report['paper_performance_complete'] is False and report['same_resource_baseline_comparison_accepted'] is False
    if status=='failed':
        new(failure_path,dict(kind='rbf_final_refit_forest_failed_candidate_independent_bytes_v1',
            task_id=task.id,seed=args.seed,method=args.method,recipe_sha256=job['recipe_sha256'],artifacts=inventory,
            producer_failure=report['failure'],rank_sequences=report['ranks'],automatic_retry_permitted=False,
            registered_failure_and_partial_output_bytes_verified=True,experiment_accepted=False,paper_performance_complete=False))
        register(failure_path,'rbf_final_refit_forest_failed_candidate_independent_bytes_v1')
        print(json.dumps(dict(task_id=task.id,failure_receipt=str(failure_path),failure_preserved=True)))
        return
    assert set(inventory)==expected_keys and report['failure'] is None
    assert report['all_46_sequences_7445_events_completed'] is True and len(report['ranks'])==world
    assert sorted(v['rank'] for v in report['ranks'])==list(range(world))
    assert all(v['all_sequences_completed'] is True and v['world_size']==world
        and v['TF32_matmul'] is False and v['TF32_cudnn'] is False for v in report['ranks'])
    assert len({v['gpu_uuid'] for v in report['ranks']})==world
    if args.method=='rbf':
        source=json.loads((root/'exclusive-source-manifest.json').read_bytes())
        for a,b in (('patches','exclusive_patches'),('configuration','configuration'),('original_source','source'),
            ('bootstrap_sha256','bootstrap_sha256'),('source_replacements','source_replacements'),
            ('CPU_capacity_candidate_admission','CPU_capacity_candidate_admission')):
            assert source[a]==plan[b]
    entry=next(v for v in json.loads(INDEX.read_bytes())['seeds'] if v['seed']==args.seed)
    assert sha(INDEX)==plan['numeric_reference_admission_sha256']
    for key in ('training_byte_proof','prediction_byte_proof','numeric_completion'):
        assert sha(entry[key])==entry[key+'_sha256']
    assert entry['numeric_failures']==0
    ck_path=Path(entry['training_byte_proof']).parent/'checkpoint'
    assert sha(ck_path)==plan['checkpoint']['sha256']
    checkpoint=json.loads(ck_path.read_bytes())
    assert checkpoint['model_sha256']==plan['final_refit_model_sha256']
    forward=json.loads(Path(entry['prediction_byte_proof']).read_bytes())
    forward_root=Path(entry['prediction_byte_proof']).parent
    expected={}
    for spec in plan['forward_outputs']:
        assert spec==forward['artifacts'][spec['key']]
        path=forward_root/(spec['key']+'.jsonl')
        assert sha(path)==spec['sha256']
        for line in path.open('rb'):
            value=json.loads(line)
            sequence=value['sequence_id']
            values=expected.setdefault(sequence,[])
            assert value['row']==len(values)
            values.append(value)
    event_path=R/f'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed{args.seed}/events.json'
    assert sha(event_path)==plan['events']['sha256']
    events=json.loads(event_path.read_bytes())
    groups={s:[e for e in events['events'] if e['sequence_id']==s] for s in events['origin_us_by_sequence']}
    assert len(groups)==46 and set(groups)==set(expected) and sum(map(len,groups.values()))==7445
    checked=[]
    for rank in range(world):
        unpack(root/f'replay-rank{rank}.tar.gz',root/f'rank{rank}-unpack')
        for sequence in sorted(groups)[rank::world]:
            directory=root/f'rank{rank}-unpack/rank-{rank}'/sequence
            receipt=json.loads((directory/'receipt.json').read_bytes())
            per_plan=json.loads((directory/'plan.json').read_bytes())
            assert per_plan['configuration']==plan['configuration'] and per_plan['fixture'] is False
            assert per_plan['model_binding']['checkpoint_sha256']==plan['checkpoint']['sha256']
            assert per_plan['model_binding']['model_sha256']==checkpoint['model_sha256']
            assert per_plan['events_sha256']==hashlib.sha256(canonical(groups[sequence])).hexdigest()
            if args.method=='rbf':
                for name,digest in plan['exclusive_patches'].items():
                    assert per_plan['source_sha256'][name]==digest
                for name,spec in plan['source_replacements'].items():
                    assert per_plan['source_sha256'][name]==spec['sha256']
            assert receipt['status']=='software_replay_completed' and receipt['completed_events']==len(groups[sequence])
            for name,digest in receipt['files'].items():
                assert sha(directory/name)==digest
            database=receipt['databases'][sequence]
            assert sha(directory/database['path'])==database['sha256']
            db=sqlite3.connect((directory/database['path']).resolve().as_uri()+'?mode=ro',uri=True)
            try:
                assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                commits=db.execute('SELECT event_id,prediction,audit FROM events ORDER BY ordinal').fetchall()
                assert len(commits)==len(groups[sequence])
                with (directory/'predictions.jsonl').open('rb') as preds,(directory/'audit.jsonl').open('rb') as audits:
                    count=0
                    for event,(event_id,prediction,audit),p,a in zip(groups[sequence],commits,preds,audits):
                        assert event_id==event['event_id'] and p.rstrip(b'\n')==prediction and a.rstrip(b'\n')==audit
                        value=json.loads(audit)
                        assert value['cache_ingestion']['new_deliveries']==event['deliveries']
                        assert value['cache_ingestion'].get('old_rows_rescored',False) is False
                        if args.method=='rbf':
                            assert value['explicit_residual_partition'] is True and value['residual_partition_version']==1
                            assert all(c['representation']=='exclusive_root_partition_regions_v1' for c in value['components'])
                        count+=1
                    assert count==len(commits) and not preds.read(1) and not audits.read(1)
                count=0;maximum=0.
                for i,node,raw,digest in db.execute('SELECT i,node_id,raw,sha FROM observations ORDER BY i'):
                    assert i==count and hashlib.sha256(raw).hexdigest()==digest
                    reference=expected[sequence][i];observation=json.loads(raw)
                    assert reference['row']==i and reference['node_id']==node
                    assert observation['features'][141]==(observation['state_us']-events['origin_us_by_sequence'][sequence])/1e8
                    assert observation['node']['information_us']<=observation['node']['arrival_us']
                    potentials=db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p',(i,)).fetchall()
                    assert [p for p,w in potentials]==[-1]+reference['context_indices'][:-1]
                    values=normalized_factors(tuple(float(v) for v in reference['logits']))
                    assert len(potentials)==len(values)
                    for (_,actual),value in zip(potentials,values):
                        assert math.isfinite(actual) and abs(actual-value)<=1e-4+1e-4*abs(value)
                        maximum=max(maximum,abs(actual-value))
                    count+=1
                assert count==len(expected[sequence])
                checked.append(dict(sequence_id=sequence,events=len(commits),nodes=count,
                    max_factor_difference_to_admitted_final_refit_forward=maximum,database_sha256=database['sha256']))
            finally:
                db.close()
            print(json.dumps(dict(stage='independent_final_forest_bytes_events_factors',completed_sequences=len(checked),
                total_sequences=46,ETA_seconds=None,ETA_reason='heterogeneous SQLite and archive costs')),flush=True)
    assert len(checked)==46 and sum(v['events'] for v in checked)==7445
    assert sum(v['nodes'] for v in checked)==entry['rows']
    new(accepted_path,dict(kind='rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1',
        task_id=task.id,seed=args.seed,method=args.method,recipe_sha256=job['recipe_sha256'],artifacts=inventory,
        all_registered_bytes_verified=True,all_46_sequences_7445_events_and_final_model_factor_nodes_verified=True,
        sequences=checked,total_nodes=entry['rows'],world_size=world,
        final_checkpoint_sha256=plan['checkpoint']['sha256'],final_model_sha256=checkpoint['model_sha256'],
        input_numeric_index_sha256=sha(INDEX),NN_atol=1e-4,NN_rtol=1e-4,
        source_sha256=sha(__file__),full_forest_semantics_or_fresh_state_independently_accepted=False,
        learned_Stage2_complete=False,same_resource_performance_accepted=False,paper_performance_complete=False,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    register(accepted_path,'rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1')
    print(json.dumps(dict(task_id=task.id,receipt=str(accepted_path),bytes_events_factors_accepted=True,
        full_forest_independent_semantics_and_states_pending=True)))


if __name__=='__main__':
    main()
