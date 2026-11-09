"""Dispatch complete exploratory seen-val bound forests after train admission.

No repeated cache construction, NN forward or passed experiment is scheduled.
GPU4/GPU8 workers must have disjoint physical bindings, including queued jobs.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re

import sys
PUBLICATION_SOURCE=Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-seen-val-complete-input-native-registered-route-v2-20261006T213931828146Z')
sys.path.insert(0,str(PUBLICATION_SOURCE))
from rbf_nested_seen_val_v2_common import R,new,register,sha
import publish_rbf_seen_val_forest_inputs_native_route_v2 as pub
from publish_rbf_seen_val_forest_inputs_native_route_v2 import (canonical,prerequisites,
    publication,PROJECT,local_inputs,manifest,remote_gate)
PRODUCER=R/'source-freezes/rbf-seen-val-bound-forest-native-registered-URL-GPU-v2-20261007'
BOOTSTRAP_SHA='9c81f2d1cd78f844c5b704e4d39b3b65c667ad2aa858c65a7f7bcba0b39330ef'
DESTINATION=None

from submit_rbf_final_identity import atomic,IMAGE
from submit_rbf_seen_val_joint_identity import snapshot

ALLOCATION_VARIANT = "final_refit_seen_val_bound_forest_registered_URL_v2"
JOURNAL=R/'receipts/rbf-seen-val-bound-forest-registered-URL-GPU-dispatch-v2-20261007.json'
CORE_JOURNALS=[R/'receipts'/name for name in (
    'rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json',
    'rbf-final-refit-full-train-fixed-topK-GPU-dispatch-20261004.json',
    'rbf-final-refit-full-train-fixed-Top1-GPU-dispatch-20261004.json')]
OTHER_JOURNALS=[R/'receipts'/name for name in (
    'rbf-batched-complete-sequence-GPU-dispatch-20261004.json',
    'rbf-branch-state-CUDA-collision-safe-dispatch-20261005.json',
    'rbf-final-refit-capacity-witness-teacher-GPU-dispatch-20261005.json',
    'rbf-final-refit-learned-priority-full-train-GPU-dispatch-20261005.json')]


def core_jobs():
    jobs=[];missing=[]
    for path in CORE_JOURNALS:
        entries=json.loads(path.read_bytes())['jobs'] if path.exists() else []
        assert len({j['seed'] for j in entries})==len(entries)
        assert {j['seed'] for j in entries}<={1337,2027,3407}
        missing.extend(dict(journal=str(path),seed=s) for s in (1337,2027,3407) if s not in {j['seed'] for j in entries})
        jobs.extend(entries)
    for path in OTHER_JOURNALS:
        if path.exists():jobs.extend(json.loads(path.read_bytes())['jobs'])
    assert len({j['task_id'] for j in jobs})==len(jobs)
    return jobs,missing


def semantic(plan):
    return hashlib.sha256(canonical({k:v for k,v in plan.items() if k!='world_size'})).hexdigest()


def verify_existing(task,job,base,identity):
    assert job is not None and job['task_id']==task.id, 'unjournaled existing task must be inspected'
    assert job['allocation_variant']==ALLOCATION_VARIANT and job['semantic_seen_val_identity']==identity
    assert job['recipe_sha256']==hashlib.sha256(canonical(job['plan'])).hexdigest()
    assert job['plan']==dict(base,world_size=job['plan']['world_size'])
    assert job['plan']['world_size'] in (4,8)
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==base['bootstrap_sha256']
    params=task.get_parameters()
    assert params['General/semantic_seen_val_identity']==identity
    assert params['General/recipe_sha256']==job['recipe_sha256']
    assert json.loads(params['General/plan'])==job['plan']


def dispatch(args,Task,api,base,publication_path):
    assert args.seed==base['seed']
    jobs=json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []
    assert len({j['seed'] for j in jobs})==len(jobs) and {j['seed'] for j in jobs}<={1337,2027,3407}
    prior=next((j for j in jobs if j['seed']==args.seed),None)
    identity=semantic(base)
    name=f'RBF final-refit complete seen-val bound forest seed{args.seed} '+identity[:16]
    matches=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
    assert len(matches)<=1
    if matches:
        task=matches[0];verify_existing(task,prior,base,identity)
        print(json.dumps(dict(seed=args.seed,existing_task_id=task.id,status=str(task.status),duplicate_not_created=True)),flush=True)
        return
    assert prior is None, 'preserve previously created seen-val forest rather than replace or retry'
    others,missing=core_jobs()
    if missing:
        print(json.dumps(dict(seed=args.seed,status='remaining_main_and_fixed_baselines_have_dispatch_priority',
            missing=missing,GPU_task_created=False,ETA='unknown')),flush=True);return
    def choices():
        candidates,fleet=snapshot(api,others+jobs)
        workers={w['id']:w for w in fleet['workers']}
        candidates=[c for c in candidates if c['world_size'] in (4,8)
            and all('L40' not in workers[w]['id'].upper() for w in c['eligible_workers'])]
        return candidates,fleet
    candidates,fleet=choices()
    if not candidates:
        print(json.dumps(dict(seed=args.seed,status='waiting_for_disjoint_idle_GPU',GPU_task_created=False,ETA='unknown')),flush=True);return
    choice=candidates[0];plan=dict(base,world_size=choice['world_size'])
    remote_gate(plan,publication_path,Task)
    if not args.execute:
        print(json.dumps(dict(seed=args.seed,choice=choice,recipe_sha256=hashlib.sha256(canonical(plan)).hexdigest(),
            semantic_seen_val_identity=identity,execute=False,GPU_task_created=False)),flush=True);return
    task=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python3.12',detect_repository=False,packages=False,add_task_init_call=False)
    job=dict(seed=args.seed,task_id=task.id,allocation_variant=ALLOCATION_VARIANT,
        semantic_seen_val_identity=identity,recipe_sha256=hashlib.sha256(canonical(plan)).hexdigest(),plan=plan,
        queue=choice['queue_name'],bindings=choice['bindings'],eligible_workers=choice['eligible_workers'],fleet=fleet,
        main_admission=str(args.main_admission),main_byte_admission=str(args.main_byte_admission),
        publication=str(publication_path),publication_sha256=sha(publication_path),
        status='created_before_configuration',ETA='unknown_until_task_level_progress')
    def save():
        now=datetime.datetime.now(datetime.timezone.utc)
        value=dict(kind='rbf_seen_val_bound_forest_GPU_dispatch_v1',jobs=jobs+[job],
            checked_at_utc=now.isoformat(),source_freeze_sha256=sha(Path(__file__).resolve().parent/'dispatch-source-freeze.json'),
            GPU_model_restriction=True,allowed_multi_GPU_model="A100",L40S_CPU_only=True,TF32_enabled=False,
            actual_memory_utilization_admitted=False,full_seen_val_forest_admission=False,paper_performance_complete=False)
        atomic(JOURNAL,value)
        path=R/'receipts'/('rbf-seen-val-bound-forest-dispatch-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        new(path,value);register(path,value['kind'])
    save()
    try:
        task.output_uri=DESTINATION
        task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='bootstrap.py',
            diff=(PRODUCER/'bootstrap.py').read_text())
        task.set_packages(['clearml==2.1.2'])
        task.set_base_docker(IMAGE,docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
            f'-e CLEARML_FILES_HOST={DESTINATION} -e NCCL_P2P_DISABLE=1 '
            '-e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_TF32_OVERRIDE=0 '
            '-e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
        task.set_parameters(dict(plan=canonical(plan).decode(),recipe_sha256=job['recipe_sha256'],
            semantic_seen_val_identity=identity,seed=args.seed,world_size=choice['world_size'],
            GPU_model_restriction=True,allowed_multi_GPU_model="A100",TF32_enabled=False,memory_target_percent='75-80',
            artificial_memory_padding=False,full_seen_val_forest_admission=False,
            ETA='unknown_until_actual_transfer_or_forest_event_progress',paper_performance_complete=False))
        task.add_tags(['Recover-Before-Fuse','SPD-seen-val-exploratory','all-class-top64',
            'bound-forest-baseline','full-independent-main-admitted','full-val-admission-pending'])
        task.reload();verify_existing(task,job,base,identity)
        remote_gate(plan,publication_path,Task)
        fresh,_=choices()
        assert choice in fresh, 'physical binding or capacity changed; created task preserved'
        Task.enqueue(task,queue_id=choice['queue_id'])
        task.reload();job['status']=str(task.status);save()
        print(json.dumps(dict(seed=args.seed,task_id=task.id,status=str(task.status),queue=choice['queue_name'],
            world_size=choice['world_size'],full_seen_val_forest_admission=False)),flush=True)
    except BaseException as error:
        job['dispatch_failure_type']=type(error).__name__;job['automatic_retry']=False;save();raise



def source_gate():
    own=Path(__file__).resolve().parent
    freeze=json.loads((own/'dispatch-source-freeze.json').read_bytes())
    for name,checksum in freeze['sources'].items():assert sha(own/name)==checksum
    for ref in freeze['references'].values():assert sha(ref['path'])==ref['sha256']
    for name,ref in freeze['resolved_imports'].items():
        module=sys.modules[name]
        assert Path(module.__file__).resolve()==Path(ref['path']).resolve()
        assert sha(module.__file__)==ref['sha256']
    pub.source_gate()
    assert sha(PRODUCER/'bootstrap.py')==BOOTSTRAP_SHA
    qualification=json.loads(Path(freeze['references']['native_registered_URL_reader']['path']).read_bytes())
    assert qualification['bootstrap_sha256']==BOOTSTRAP_SHA
    assert qualification['actual_completed_registered_asset_transport']['url_host_rewritten'] is False
    assert qualification['actual_completed_registered_asset_transport']['saved_file_independently_reread'] is True
    return freeze


def member_gate(path,publication_path,seed,freeze):
    value=json.loads(path.read_bytes());published=json.loads(publication_path.read_bytes())
    assert value['kind']=='rbf_seen_val_all_registered_cache_members_independent_readback'
    assert value['source_sha256']==freeze['references']['complete_cache_member_auditor']['sha256']
    assert value['seed']==published['seed']==seed and value['task_id']==published['task_id']
    assert value['publication_sha256']==sha(publication_path) and value['task_status']=='completed'
    assert value['registered_artifacts']==7 and value['member_count']==14379
    assert value['source_frames']==7189 and value['sequences']==21 and value['events']==3316
    assert value['all_registered_members_independently_streamed_and_hashed'] is True
    assert value['cache_or_neural_forward_rebuilt'] is False and value['seen_SPD_val_exploratory'] is True
    ledger=json.loads((R/'receipts/20260928-execution-ledger.json').read_bytes())['entries']
    assert any(e.get('receipt')==str(path) and e.get('receipt_sha256')==sha(path) for e in ledger)
    return dict(path=str(path),sha256=sha(path),task_id=value['task_id'])


def native_v2_training_priority(Task):
    root=R/'artifacts/v2v4real-native-identity-target-only-v2-20261006T204325976623Z'
    waiting=[]
    for seed in (1337,2027,3407):
        path=root/f'seed{seed}/cloud-input-publication/independent-readback-receipt.json'
        if not path.exists():continue
        proof=json.loads(path.read_bytes())
        assert proof['kind']=='v2v4real_native_v2_training_input_registered_full_byte_and_member_readback_v1'
        assert proof['all_registered_artifacts_and_archive_members_independently_read_back'] is True
        assert proof['CUDA_branch_state_or_priority_source_included'] is False
        identity=proof['semantic_training_identity']
        name=f'V2V native corrected-target joint identity seed{seed} '+identity[:16]
        matches=Task.get_tasks(project_name='Thesis/Recover-Before-Fuse/Training',task_name='^'+re.escape(name)+'$')
        assert len(matches)<=1
        if not matches or str(matches[0].status) not in ('queued','in_progress','completed','closed','published'):
            waiting.append(dict(seed=seed,admission=str(path),task_id=matches[0].id if matches else None,
                status=str(matches[0].status) if matches else 'not_dispatched'))
        else:
            assert matches[0].get_parameters()['General/semantic_training_identity']==identity
    return waiting


def main():
    global DESTINATION
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(1337,2027,3407),required=True)
    for name in ('main-admission','main-byte-admission','publication','member-admission'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    freeze=source_gate()
    if not all(p.is_file() for p in (args.main_admission,args.main_byte_admission,args.publication,args.member_admission)):
        print(json.dumps(dict(seed=args.seed,status='waiting_for_full_main_seven_asset_and_complete_registered_cache_members',GPU_task_created=False)),flush=True);return
    proof,main,_,_=prerequisites(args.main_admission,args.main_byte_admission,args.seed)
    records,bound=local_inputs(args.seed,args.main_admission,proof,main)
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    from clearml.backend_api.session import Session
    DESTINATION=Session.get_files_server_host();pub.DESTINATION=DESTINATION
    assert DESTINATION in ('http://10.100.34.118:8081','http://10.100.35.118:8081')
    expected=manifest(args.seed,records,proof,main,Task)
    published=publication(args.publication,expected,Task)
    members=member_gate(args.member_admission,args.publication,args.seed,freeze)
    base=pub.expected_plan(main,bound,published,4)
    base.update(bootstrap_sha256=BOOTSTRAP_SHA,artifact_files_host=DESTINATION,
        native_registered_URL_consumer_source_freeze_sha256=sha(PRODUCER/'source-freeze.json'),
        complete_registered_cache_member_admission=members)
    # Reject rerunning a prior same-seed seen-val forest under a new transport ID.
    prefix=f'RBF final-refit complete seen-val bound forest seed{args.seed} '
    matches=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(prefix))
    for task in matches:
        if task.get_parameters().get('General/semantic_seen_val_identity')!=semantic(base):
            print(json.dumps(dict(seed=args.seed,status='prior_same_seed_seen_val_forest_preserved',task_id=task.id,
                task_status=str(task.status),GPU_task_created=False,automatic_retry=False)),flush=True);return
    pending=native_v2_training_priority(Task)
    if pending:
        print(json.dumps(dict(seed=args.seed,status='qualified_native_v2_training_has_first_idle_GPU_priority',
            pending=pending,GPU_task_created=False)),flush=True);return
    remote_gate(base,args.publication,Task)
    if args.execute:
        with (R/'receipts/rbf-seen-val-bound-forest-dispatch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            dispatch(args,Task,APIClient(),base,args.publication)
    else:dispatch(args,Task,APIClient(),base,args.publication)


if __name__=='__main__':main()
