#!/usr/bin/env python3
"""Deduplicated admitted fit-head training on any idle collision-free GPU4 queue."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from clearml import Task
from clearml.backend_api.session.client import APIClient

ROOT=Path('/Volumes/Data/test/recover-before-fuse')
SOURCE=ROOT/'source-freezes/spd-canonical-association-car-only-GPU-source-v3-20261001'
EXECUTION=ROOT/'artifacts/spd-canonical-association-car-only-GPU-execution-source-v3-20261001'
PRIOR_DIAGNOSTIC={0:'32ae16db5fc2401e84cff0e2f6e387d2',1:'2193532cb0a749988c9c41c2bf9dbc6a',2:'5315861001154da39de71be3284f625a',3:'882a10e95fd5492289c35c7612434874',4:'c8c680902ab6438e834b69df5206e1e1'}
IMAGE='gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb'
PROJECT='Thesis/EventTrack-V2X/Training'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def binding(w):
    m=re.fullmatch(r'([^:]+):gpu([0-9]+(?:,[0-9]+)*)',w['id'])
    if not m:return None
    devices=set(map(int,m.group(2).split(',')))
    return w.get('ip') or m.group(1),devices


def slots(api, own):
    now=datetime.now(timezone.utc);workers=[w.to_dict() for w in api.workers.get_all()];queues=[q.to_dict() for q in api.queues.get_all()];result=[]
    for q in queues:
        if not q['name'].startswith('GPU4-') or 'L40' in q['name']:continue
        pending=[e['task'] for e in q.get('entries',[])]
        if any(t not in own for t in pending):continue
        idle=[]
        for w in workers:
            b=binding(w)
            if b is None or len(b[1])!=4 or w.get('task',{}).get('id') or not any(x['id']==q['id'] for x in w.get('queues',[])):continue
            stamp=datetime.fromisoformat(str(w['last_activity_time']).replace('Z','+00:00'))
            if not 0<=(now-stamp).total_seconds()<90:continue
            if any(other.get('task',{}).get('id') and binding(other) and binding(other)[0]==b[0] and b[1]&binding(other)[1] for other in workers):continue
            if not any(binding(previous)[0]==b[0] and binding(previous)[1]&b[1] for previous in idle):idle.append(w)
        capacity=len(idle)-len(pending)
        if capacity>0:result.append(dict(queue_id=q['id'],queue_name=q['name'],capacity=capacity,eligible_workers=[w['id'] for w in idle]))
    result.sort(key=lambda r:(0 if 'A100' in r['queue_name'] else 1 if 'V100' in r['queue_name'] else 2,r['queue_name']))
    return result,[dict(id=w['id'],ip=w.get('ip'),task_id=w.get('task',{}).get('id'),queues=[q['name'] for q in w.get('queues',[])]) for w in workers]


def main():
    for r in json.loads((SOURCE/'source-inventory.json').read_text()):assert sha(SOURCE/r['path'])==r['sha256']
    execution_proof=EXECUTION/'independent-cloud-readback/acceptance-receipt.json'
    if not execution_proof.exists():print('COMPATIBLE_EXECUTION_SOURCE_CLOUD_ADMISSION_PENDING_NO_TASK_CREATED',flush=True);return
    ep=json.loads(execution_proof.read_text());assert ep['all_registered_source_and_runtime_admission_bytes_verified'] is True
    execution_task=Task.get_task(task_id=ep['task_id']);assert execution_task.status=='completed'
    for n,r in ep['artifacts'].items():assert execution_task.artifacts[n].hash==r['sha256'] and execution_task.artifacts[n].size==r['bytes']
    em=EXECUTION/'execution-source-manifest.json';assert sha(em)==ep['execution_source_manifest_sha256'];execution_manifest=json.loads(em.read_text());assert execution_manifest['runtime_contract_independently_admitted'] is True
    api=APIClient();receipt=ROOT/'receipts/spd-canonical-oof-fivefold-association-car-only-GPU-v3-dispatch-20261001.json'
    # Independent folds may start as soon as their own cloud bytes are admitted.
    # The append-only job journal retains waiting and failed/created records.
    rows=json.loads(receipt.read_text())['jobs'] if receipt.exists() else []
    own={r['task_id'] for r in rows if r.get('task_id')}
    def save():
        tmp=receipt.with_suffix('.json.tmp');tmp.write_text(json.dumps(dict(kind='canonical_fivefold_predicted_association_GPU4_dispatch_candidate',checked_at_utc=datetime.now(timezone.utc).isoformat(),dispatcher_sha256=sha(Path(__file__)),jobs=rows,paper_eligible=False),indent=2)+'\n');tmp.replace(receipt)
    for fold in range(5):
        B=ROOT/f'artifacts/spd-canonical-oof-fold{fold}-association-GPU-training-assets-v1-20261001';proof=B/'independent-cloud-readback/acceptance-receipt.json'
        if not proof.exists():
            if not any(r['fold_id']==fold and r['status']=='waiting_for_cloud_byte_readback' for r in rows):rows.append(dict(fold_id=fold,status='waiting_for_cloud_byte_readback',ETA='unknown'));save()
            continue
        accepted=json.loads(proof.read_text());assert accepted['all_registered_bytes_and_archive_members_verified'] is True and accepted['fold_id']==fold
        manifest=B/'asset-manifest.json';asset=json.loads(manifest.read_text());assert sha(manifest)==accepted['asset_manifest_sha256']
        parent=Task.get_task(task_id=accepted['task_id']);assert parent.status=='completed'
        for n,r in accepted['artifacts'].items():assert parent.artifacts[n].hash==r['sha256'] and parent.artifacts[n].size==r['bytes']
        bootstrap=SOURCE/'bootstrap_spd_canonical_oof_association_gpu4.py';assert sha(bootstrap)==execution_manifest['bootstrap_sha256']
        prior=Task.get_task(task_id=PRIOR_DIAGNOSTIC[fold]);assert prior.status=='completed' and len(prior.artifacts)==5
        prior_proof=ROOT/f'artifacts/spd-canonical-oof-fold{fold}-all-class-association-diagnostic-preserved-20261001/byte-readback-and-scope-exclusion-receipt.json';assert json.loads(prior_proof.read_text())['all_registered_artifact_bytes_verified']
        assert execution_manifest['class_scope']=='car' and execution_manifest['class_index']==0
        name=f'SPD canonical OOF fold-{fold} association car-only-v3 seed1337 fixed24 '+asset['configuration_sha256'][:12]+' '+execution_manifest['trainer_sha256'][:12]
        matches=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$');assert len(matches)<=1
        if matches:
            existing=matches[0];assert existing.get_parameters()['General/asset_manifest_sha256']==sha(manifest)
            if not any(r.get('task_id')==existing.id for r in rows):rows.append(dict(fold_id=fold,task_id=existing.id,status=str(existing.status),duplicate_not_created=True));save()
            own.add(existing.id);continue
        eligible,snapshot=slots(api,own)
        if not eligible:
            if not any(r['fold_id']==fold and r['status']=='waiting_for_idle_collision_free_GPU4' for r in rows):rows.append(dict(fold_id=fold,status='waiting_for_idle_collision_free_GPU4',ETA='unknown'));save()
            continue
        choice=eligible[0]
        task=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.training,binary='python3.12')
        task.output_uri='http://10.100.34.118:8081';task.set_script(repository='',branch='',commit='',working_dir='.',entry_point=bootstrap.name,diff=bootstrap.read_text());task.set_packages(['clearml==2.1.2'])
        task.set_base_docker(IMAGE,docker_arguments='-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 16g --env NCCL_P2P_DISABLE=1 --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env CLEARML_FILES_HOST=http://10.100.34.118:8081 --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env PYTHONDONTWRITEBYTECODE=1')
        task.set_parameters(dict(asset_task_id=parent.id,asset_manifest_sha256=sha(manifest),asset_independent_readback_sha256=sha(proof),bootstrap_sha256=sha(bootstrap),trainer_sha256=execution_manifest['trainer_sha256'],execution_source_task_id=execution_task.id,execution_source_manifest_sha256=sha(em),execution_source_independent_readback_sha256=sha(execution_proof),prior_all_class_diagnostic_task_id=prior.id,NCCL_P2P_DISABLE=1,output_route='10.100.34.118:8081',configuration_sha256=asset['configuration_sha256'],fold_id=fold,seed=1337,epochs=24,queue_name=choice['queue_name'],GPU_model_restriction=False,class_scope='car',class_index=0,car_view_independent_acceptance_sha256=execution_manifest['car_view_independent_acceptance_sha256'],paper_eligible=False))
        task.add_tags(['Recover-Before-Fuse','canonical-OOF','predicted-association','fit-only','candidate-not-paper-qualified'])
        task.reload();assert sha(bootstrap)==hashlib.sha256(task.data.script.diff.encode()).hexdigest()
        row=dict(fold_id=fold,task_id=task.id,status='created_before_enqueue',prior_all_class_diagnostic_task_id=prior.id,execution_source_task_id=execution_task.id,execution_source_manifest_sha256=sha(em),execution_source_independent_readback_sha256=sha(execution_proof),asset_task_id=parent.id,asset_manifest_sha256=sha(manifest),configuration_sha256=asset['configuration_sha256'],asset_independent_readback_sha256=sha(proof),queue=choice['queue_name'],eligible_workers_at_dispatch=choice['eligible_workers'],worker_snapshot=snapshot,ETA='unknown_until_transfer_and_training_progress')
        rows.append(row);save()
        fresh,_=slots(api,own);assert any(r['queue_id']==choice['queue_id'] for r in fresh), 'GPU capacity changed: preserve created task for inspection'
        Task.enqueue(task,queue_id=choice['queue_id']);own.add(task.id);task.reload();assert str(task.status) in ('queued','in_progress')
        row['status']=str(task.status);save();print(json.dumps(dict(fold_id=fold,task_id=task.id,queue=choice['queue_name'],status=str(task.status))),flush=True)


if __name__=='__main__':main()
