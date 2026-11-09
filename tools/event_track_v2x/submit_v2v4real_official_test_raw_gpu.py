#!/usr/bin/env python3
"""Deduplicate original official-test exports and use idle disjoint GPU workers."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from submit_rbf_final_identity import available, binding, IMAGE
from rbf_nested_seen_val_v2_common import R,new,register,sha
from run_v2v4real_official_test_gpu_shards import SOURCE_TASK,CHECKPOINT_TASKS,PROJECTION_ARCHIVE_SHA
from run_v2v4real_official_test_raw_features import CHECKPOINTS,PROJECTION_TASK,SOURCE_ARCHIVE_SHA,FRAME_SHA

PROJECT='Thesis/Recover-Before-Fuse/Inference'
from v2v4real_official_test_runtime import PACKAGES,OFFLINE_TASK,OFFLINE_ARCHIVE_SHA,OFFLINE_MANIFEST_SHA


def validate_failed_predecessors(predecessors,seed):
    if len(predecessors)>1:raise ValueError('ambiguous predecessor experiment')
    for previous in predecessors:
        params=previous.get_parameters()
        if (str(previous.status)!='failed' or 'native-features' in previous.artifacts
                or params.get('General/checkpoint_sha256')!=CHECKPOINTS[seed]
                or params.get('General/projection_frames_sha256')!=FRAME_SHA):
            raise ValueError('retry requires failed matching predecessor without prediction outputs')


def predecessor_versions(attempt):
    return {'original-point-reader-binding-v10':('runtime-offline-libraries-v9','runtime-system-libraries-v8','runtime-diagnostics-v7','runtime-attested-v6','original-publication-self-member-v5','original-publication-envelope-v4','authenticated-LAN34-v3','authenticated-download-v2','v1'),
            'runtime-offline-libraries-v9':('runtime-system-libraries-v8','runtime-diagnostics-v7','runtime-attested-v6','original-publication-self-member-v5','original-publication-envelope-v4','authenticated-LAN34-v3','authenticated-download-v2','v1'),
            'runtime-system-libraries-v8':('runtime-diagnostics-v7','runtime-attested-v6','original-publication-self-member-v5','original-publication-envelope-v4','authenticated-LAN34-v3','authenticated-download-v2','v1'),
            'runtime-diagnostics-v7':('runtime-attested-v6','original-publication-self-member-v5','original-publication-envelope-v4','authenticated-LAN34-v3','authenticated-download-v2','v1'),
            'runtime-attested-v6':('original-publication-self-member-v5','original-publication-envelope-v4','authenticated-LAN34-v3','authenticated-download-v2','v1'),
            'authenticated-download-v2':('v1',),
            'authenticated-LAN34-v3':('authenticated-download-v2','v1'),
            'original-publication-envelope-v4':('authenticated-LAN34-v3','authenticated-download-v2','v1'),
            'original-publication-self-member-v5':('original-publication-envelope-v4','authenticated-LAN34-v3','authenticated-download-v2','v1')}[attempt]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--attempt',choices=('v1','authenticated-download-v2','authenticated-LAN34-v3','original-publication-envelope-v4','original-publication-self-member-v5','runtime-attested-v6','runtime-diagnostics-v7','runtime-system-libraries-v8','runtime-offline-libraries-v9','original-point-reader-binding-v10'),default='v1')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    manifest=json.loads((a.source/'source-manifest.json').read_bytes());bootstrap=a.source/'bootstrap.py'
    if sha(bootstrap)!=manifest['bootstrap_sha256']:raise ValueError('frozen bootstrap bytes differ')
    gate=json.loads((a.source/'software-qualification.json').read_bytes())
    if gate['bootstrap_sha256']!=sha(bootstrap) or gate['software_checks_passed'] is not True:
        raise ValueError('qualified exact source required')
    for row in gate['evidence']:
        if sha(row['path'])!=row['sha256']:raise ValueError('software evidence changed')
    api=APIClient();jobs=[]
    for seed in (1337,2027,3407):
        original_name=f'V2V4Real official-test native raw seed{seed} checkpoint{CHECKPOINTS[seed][:12]} frames{FRAME_SHA[:12]} v1'
        name=original_name if a.attempt=='v1' else original_name[:-2]+a.attempt
        existing=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
        if existing:
            if len(existing)!=1:raise ValueError('ambiguous existing official-test experiment')
            task=existing[0];params=task.get_parameters()
            if params.get('General/checkpoint_sha256')!=CHECKPOINTS[seed] or params.get('General/projection_frames_sha256')!=FRAME_SHA:
                raise ValueError('existing task configuration differs')
            if params.get('General/execution_bundle_sha256')!=manifest['execution_bundle_sha256']:
                raise ValueError('existing attempt execution source differs')
            jobs.append(dict(seed=seed,task_id=task.id,status=str(task.status),duplicate_not_created=True));continue
        predecessors=[]
        if a.attempt!='v1':
            for version in predecessor_versions(a.attempt):
                prior_name=original_name[:-2]+version
                predecessors=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(prior_name)+'$')
                if predecessors:break
            if len(predecessors)==1 and str(predecessors[0].status)!='failed':
                previous=predecessors[0];params=previous.get_parameters()
                if params.get('General/checkpoint_sha256')!=CHECKPOINTS[seed] or params.get('General/projection_frames_sha256')!=FRAME_SHA:
                    raise ValueError('active or successful predecessor identity differs')
                bindings=[]
                if str(previous.status) not in ('completed','stopped','closed','published'):
                    workers=[w.to_dict() for w in api.workers.get_all()]
                    assigned=[binding(w) for w in workers if w['id']==previous.data.last_worker]
                    if len(assigned)!=1 or assigned[0] is None:raise ValueError('active predecessor physical binding unavailable')
                    bindings=[[assigned[0][0],sorted(assigned[0][1])]]
                jobs.append(dict(seed=seed,predecessor_task_id=previous.id,status=str(previous.status),
                                 new_attempt_not_created=True,bindings=bindings));continue
            validate_failed_predecessors(predecessors,seed)
        inputs=[(SOURCE_TASK,'source',SOURCE_ARCHIVE_SHA),(PROJECTION_TASK,'projection',PROJECTION_ARCHIVE_SHA),
                (PROJECTION_TASK,'publication-receipt','530f97f1d74a72dd7979a3999190b7e64972a3455d39d88795bc4061a8809231'),
                (CHECKPOINT_TASKS[seed],'best-checkpoint',CHECKPOINTS[seed])]
        if a.attempt=='runtime-offline-libraries-v9':
            inputs.extend([(OFFLINE_TASK,'libgl',OFFLINE_ARCHIVE_SHA),(OFFLINE_TASK,'manifest',OFFLINE_MANIFEST_SHA)])
        for tid,key,digest in inputs:
            owner=Task.get_task(task_id=tid)
            if str(owner.status)!='completed' or owner.artifacts[key].hash!=digest:raise ValueError('registered input changed')
        workers=[w.to_dict() for w in api.workers.get_all()];queues=[q.to_dict() for q in api.queues.get_all()]
        reserved=[(host,set(cards)) for j in jobs for host,cards in j.get('bindings',[])]
        choices=available(workers,queues,datetime.datetime.now(datetime.timezone.utc),reserved)
        if not choices:
            jobs.append(dict(seed=seed,status='waiting_for_idle_disjoint_GPU_worker',ETA='unknown'));continue
        choice=choices[0]
        job=dict(seed=seed,**choice,status='eligible_candidate',duplicate_not_created=False,
                 attempt=a.attempt,failed_predecessor_task_ids=[t.id for t in predecessors])
        if a.execute:
            task=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python')
            task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='official_test_bootstrap.py',diff=bootstrap.read_text())
            task.set_packages(PACKAGES)
            task.set_base_docker(IMAGE,docker_arguments='--network host --shm-size 64g '
                '-e CLEARML_AGENT_FORCE_TASK_INIT=0 --env NVIDIA_DRIVER_CAPABILITIES=compute,utility')
            task.set_parameters(dict(seed=seed,world_size=choice['world_size'],checkpoint_sha256=CHECKPOINTS[seed],
                projection_frames_sha256=FRAME_SHA,execution_bundle_sha256=manifest['execution_bundle_sha256'],
                bootstrap_sha256=sha(bootstrap),protocol_id='v2v4real-nominal-10hz-formal-v1',
                official_test_read=True,gt_read=False,paper_metric=False,TF32_enabled=False,
                attempt=a.attempt,failed_predecessor_task_ids=[t.id for t in predecessors]))
            task.add_tags(['Recover-Before-Fuse','V2V4Real','official-test-native-raw-'+a.attempt,f'seed-{seed}'])
            # Independently read the uploaded script and parameters before enqueue.
            admitted=Task.get_task(task_id=task.id)
            if hashlib.sha256(admitted.data.script.diff.encode()).hexdigest()!=sha(bootstrap):
                raise ValueError('cloud execution source readback differs')
            if admitted.get_parameters().get('General/execution_bundle_sha256')!=manifest['execution_bundle_sha256']:
                raise ValueError('cloud task configuration readback differs')
            # Recheck the fleet after source/config upload; queue state may have changed.
            current=available([w.to_dict() for w in api.workers.get_all()],[q.to_dict() for q in api.queues.get_all()],
                datetime.datetime.now(datetime.timezone.utc),reserved)
            if not any(x==choice for x in current):raise ValueError('physical GPU admission changed before enqueue')
            Task.enqueue(task,queue_id=choice['queue_id'])
            job.update(task_id=task.id,status='queued',cloud_script_independently_read_back=True)
        jobs.append(job)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        receipt=R/'receipts'/('v2v4real-official-test-raw-GPU-dispatch-'+stamp+'.json')
        value=dict(kind='v2v4real_official_test_raw_GPU_candidate_dispatch_v1',jobs=jobs,
            source_manifest_sha256=sha(a.source/'source-manifest.json'),bootstrap_sha256=sha(bootstrap),
            software_qualification_sha256=sha(a.source/'software-qualification.json'),execute=a.execute,
            GPU_model_restriction=True,allowed_multi_GPU_model="A100",L40S_CPU_only=True,paper_metric_complete=False)
        new(receipt,value);register(receipt,value['kind'])
        print(json.dumps(dict(receipt=str(receipt),jobs=jobs)),flush=True)
    if not jobs:print(json.dumps(dict(status='no_candidate',ETA='unknown')))


if __name__=='__main__':main()
