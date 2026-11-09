#!/usr/bin/env python3
"""Deduplicated corrected-target native training; read-only unless --execute.

Cloud input/member admission and conservative physical worker overlap checks
precede task creation. Actual UUID/collision/native-CUDA checks precede updates
inside the worker. Live tasks, failed tasks and unknown creation outcomes stay.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.submit_rbf_final_identity import available,IMAGE
from tools.event_track_v2x.gpu_worker_task_admission import query_fleet


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def new(p,v):Path(p).open('x').write(json.dumps(v,indent=2,sort_keys=True)+'\n')


def submit(args):
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    from clearml.backend_api.session import Session
    from tools.event_track_v2x.rbf_nested_seen_val_v2_common import register
    admission=Path(args.cloud_admission).absolute()
    if not admission.exists():return dict(status='waiting_for_complete_native_v2_cloud_input_readback',training_dispatched=False)
    if sha(admission)!=args.cloud_admission_sha256:raise ValueError('original full registered input byte proof required')
    proof=json.loads(admission.read_bytes())
    if (proof['kind']!='v2v4real_native_v2_training_input_registered_full_byte_and_member_readback_v1'
            or proof['all_registered_artifacts_and_archive_members_independently_read_back'] is not True
            or proof['CUDA_branch_state_or_priority_source_included'] is not False or proof['source_closure_module_count']!=27):
        raise ValueError('corrected native v2 training-only registered input qualification required')
    for spec in proof['registered_artifacts'].values():
        owner=Task.get_task(task_id=spec['task']);asset=owner.artifacts[spec['key']]
        if str(owner.status)!='completed' or asset.hash!=spec['sha256'] or asset.size!=spec['bytes']:
            raise ValueError('original registered corrected-target input changed')
    plan=Path(args.bundle_plan).absolute()
    if sha(plan)!=proof['registered_artifacts']['bundle-plan']['sha256']:raise ValueError('actual registered bundle plan required')
    pv=json.loads(plan.read_bytes());semantic=pv['semantic'];identity=hashlib.sha256(canonical(semantic).encode()).hexdigest()
    if (pv['kind']!='v2v4real_native_joint_identity_admitted_GPU_input_bundle_v2'
            or identity!=proof['semantic_training_identity'] or identity!=pv['semantic_training_identity']
            or semantic['seed'] not in (1337,2027,3407)
            or semantic['native_target_contract']!='author_native_mapped_id_one_explicit_association_v2'):
        raise ValueError('hardware-independent corrected-target training recipe differs')
    consumer=Path(args.consumer).absolute()
    if sha(consumer)!=args.consumer_sha256:raise ValueError('frozen native v2 GPU consumer required')
    api=APIClient();project='Thesis/Recover-Before-Fuse/Training';seed=semantic['seed']
    name=f'V2V native corrected-target joint identity seed{seed} '+identity[:16]
    matches=Task.get_tasks(project_name=project,task_name='^'+re.escape(name)+'$')
    if len(matches)>1:raise ValueError('ambiguous existing training tasks; retain all')
    if matches:
        task=matches[0];p=task.get_parameters()
        if p.get('General/semantic_training_identity')!=identity or p.get('General/consumer_sha256')!=args.consumer_sha256:
            raise ValueError('existing semantic candidate source differs; retain')
        return dict(task_id=task.id,status=str(task.status),duplicate_not_created=True,checkpoint_qualified=False)
    def fleet():
        return query_fleet(api)
    choices,snapshot=fleet()
    if not choices:return dict(status='waiting_for_disjoint_idle_physical_GPU4_or_GPU8',seed=seed,training_dispatched=False,ETA='unknown')
    choice=choices[0]
    if not args.execute:return dict(status='eligible_corrected_target_candidate',seed=seed,choice=choice,training_dispatched=False)
    R=Path('/Volumes/Data/test/recover-before-fuse');journal=R/'receipts'/('v2v4real-native-v2-training-dispatch-'+identity+'.json')
    with Path(str(journal)+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if journal.exists():raise ValueError('durable prior creation intent exists; reconcile original outcome before retry')
        # Requery under the intent lock; hardware/card count does not change ID.
        if Task.get_tasks(project_name=project,task_name='^'+re.escape(name)+'$'):
            raise ValueError('matching candidate appeared; preserve and re-read')
        recipe=dict(seed=seed,world_size=choice['world_size'],input_bundle=proof['registered_artifacts']['input-bundle'],
            bundle_plan_sha256=sha(plan),semantic_training_identity=identity,consumer_sha256=args.consumer_sha256,
            cloud_input_bytes_independently_read=True,disjoint_physical_worker_admitted=True,
            cloud_input_admission_sha256=sha(admission),native_target_contract=semantic['native_target_contract'])
        intent=dict(kind='v2v4real_native_v2_training_durable_intent_before_create',task_name=name,recipe=recipe,
            recipe_sha256=hashlib.sha256(canonical(recipe).encode()).hexdigest(),choice=choice,fleet=snapshot,
            source_sha256={str(Path(__file__)):sha(__file__),str(ROOT/'tools/event_track_v2x/submit_rbf_final_identity.py'):sha(ROOT/'tools/event_track_v2x/submit_rbf_final_identity.py')},creation_outcome_unknown=True,checkpoint_qualified=False)
        new(journal,intent);register(journal,intent['kind'])
        task=Task.create(project_name=project,task_name=name,task_type=Task.TaskTypes.training,binary='python3.12',detect_repository=False,packages=False,add_task_init_call=False)
        created=journal.with_name(journal.stem+'-created.json');new(created,dict(task_id=task.id,intent_sha256=sha(journal),created_before_enqueue=True));register(created,'v2v4real_native_v2_training_created_before_enqueue')
        task.output_uri=Session.get_files_server_host()
        task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='native_identity_v2.py',diff=consumer.read_text())
        task.set_packages(['clearml==2.1.2','numpy==1.26.4','scipy==1.14.1'])
        task.set_base_docker(IMAGE,docker_arguments='--network host --shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 -e NCCL_P2P_DISABLE=1 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_TF32_OVERRIDE=0 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
        params=dict(recipe=canonical(recipe),recipe_sha256=intent['recipe_sha256'],seed=seed,world_size=choice['world_size'],semantic_training_identity=identity,consumer_sha256=args.consumer_sha256,checkpoint_qualified=False,GPU_model_restriction=True,allowed_multi_GPU_model='A100')
        task.set_parameters(params);task.add_tags(['Recover-Before-Fuse','V2V-native-target-v2','joint-cross-source-temporal','independent-checkpoint-admission-pending'])
        task.reload()
        if hashlib.sha256(task.data.script.diff.encode()).hexdigest()!=args.consumer_sha256:raise ValueError('registered execution source differs; retain created task')
        actual=task.get_parameters()
        if any(actual.get('General/'+k)!=str(v) for k,v in params.items()):raise ValueError('registered parameters differ; retain created task')
        fresh,_=fleet()
        if not any(x['queue_id']==choice['queue_id'] and x['bindings']==choice['bindings'] for x in fresh):raise ValueError('physical capacity changed; retain created task without enqueue')
        Task.enqueue(task,queue_id=choice['queue_id']);task.reload()
        result=dict(kind='v2v4real_native_v2_joint_training_dispatch_candidate',task_id=task.id,seed=seed,status=str(task.status),choice=choice,recipe=recipe,intent_sha256=sha(journal),cloud_input_admission_sha256=sha(admission),checkpoint_qualified=False,paper_metrics_complete=False)
        p=journal.with_name(journal.stem+'-enqueued.json');new(p,result);register(p,result['kind']);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cloud-admission','cloud-admission-sha256','bundle-plan','consumer','consumer-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--execute',action='store_true');print(json.dumps(submit(p.parse_args()),sort_keys=True),flush=True)
