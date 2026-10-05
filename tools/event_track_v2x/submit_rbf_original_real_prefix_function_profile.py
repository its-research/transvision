"""A single distinct diagnostic after remaining real core jobs get priority.

No GPU model filter, no overlapping physical binding, no retry or memory
padding. The original body/weights and all numerical limits stay frozen.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from rbf_nested_seen_val_v2_common import R,new,register,sha
from submit_rbf_final_identity import atomic,IMAGE
from submit_rbf_seen_val_joint_identity import snapshot

ROOT=R/'source-freezes/rbf-original-real-prefix-function-profile-safe-dispatch-v1-20261004'
JOURNAL=R/'receipts/rbf-original-real-prefix-function-profile-GPU-dispatch-20261004.json'
PROJECT='Thesis/Recover-Before-Fuse/Inference'
CORE_JOURNALS=(
    R/'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json',
    R/'receipts/rbf-final-refit-full-train-fixed-topK-GPU-dispatch-20261004.json',
    R/'receipts/rbf-final-refit-full-train-fixed-Top1-GPU-dispatch-20261004.json')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def core_priority(journals):
    jobs=[];missing=[]
    for path in journals:
        entries=json.loads(path.read_bytes())['jobs'] if path.exists() else []
        assert len({job['seed'] for job in entries})==len(entries)
        assert {job['seed'] for job in entries} <= {1337,2027,3407}
        missing.extend(dict(journal=str(path),seed=seed) for seed in (2027,1337,3407)
            if seed not in {job['seed'] for job in entries})
        jobs.extend(entries)
    assert len({job['task_id'] for job in jobs})==len(jobs)
    return jobs,missing


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    prepared=json.loads((ROOT/'preparation.json').read_bytes())
    for name,record in prepared['sources'].items():assert sha(ROOT/name)==record['sha256']
    for record in prepared['references']:assert sha(record['path'])==record['sha256']
    executor=Path(prepared['executor_preparation'])
    contract=json.loads(executor.read_bytes());base=contract['plan']
    core,missing=core_priority(CORE_JOURNALS)
    if missing:
        print(json.dumps(dict(status='remaining_K4_and_Top1_have_priority',missing=missing,
            profile_task_created=False,ETA='unknown',goal_status='active')),flush=True)
        return
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    for job in core:
        task=Task.get_task(task_id=job['task_id'])
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==job['plan']['bootstrap_sha256']
        assert hashlib.sha256(canonical(job['plan'])).hexdigest()==job['recipe_sha256']
        assert task.get_parameters()['General/recipe_sha256']==job['recipe_sha256']
        assert json.loads(task.get_parameters()['General/plan'])==job['plan']
        if str(task.status) in ('created','unknown'):
            print(json.dumps(dict(status='preserve_and_finish_existing_core_dispatch_first',task_id=task.id,
                profile_task_created=False,ETA='unknown')),flush=True)
            return
    profile_jobs=json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []
    assert len(profile_jobs)<=1
    semantic=hashlib.sha256(canonical(dict(fixed_plan=base,
        independent_reader_source_freeze_sha256=prepared['reader_source_freeze_sha256'],
        scope='one real causal prefix cost diagnostic after core dispatch priority'))).hexdigest()
    name='RBF original real causal prefix function profile seed2027 '+semantic[:16]
    matches=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
    assert len(matches)<=1
    if matches:
        task=matches[0]
        assert task.get_parameters()['General/semantic_profile_identity']==semantic
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==base['bootstrap_sha256']
        assert len(profile_jobs)==1 and profile_jobs[0]['task_id']==task.id, 'preserve unknown existing Task; inspect journal'
        print(json.dumps(dict(existing_task_id=task.id,status=str(task.status),duplicate_not_created=True)),flush=True)
        return
    assert not profile_jobs, 'a journal Task is never silently replaced'
    for spec in [base[k] for k in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest')]+base['forward_outputs']:
        asset=Task.get_task(task_id=spec['task']);assert str(asset.status)=='completed'
        artifact=asset.artifacts[spec['key']]
        assert artifact.hash==spec['sha256'] and artifact.size==spec['bytes']
    api=APIClient();choices,fleet=snapshot(api,core+profile_jobs)
    if not choices:
        print(json.dumps(dict(status='waiting_for_disjoint_idle_GPU_after_core_priority',ETA='unknown',
            physical_non_L40_GPUs=fleet['physical_non_L40_GPUs'],physical_unbound_GPUs=fleet['physical_unbound_GPUs'],
            profile_task_created=False,goal_status='active')),flush=True)
        return
    choice=choices[0];plan=dict(base,world_size=choice['world_size'])
    reader_path=Path(prepared['reader_path'])
    module_spec=importlib.util.spec_from_file_location('frozen_prefix_profile_plan_gate',reader_path)
    reader=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(reader)
    reader.gate_plan(plan,contract)
    if not args.execute:
        print(json.dumps(dict(choice=choice,execute=False,plan_recipe_sha256=hashlib.sha256(canonical(plan)).hexdigest(),
            GPU_memory_target_percent=[75,80],actual_profile_or_memory_target_achieved=False)),flush=True)
        return
    digest=hashlib.sha256(canonical(plan)).hexdigest()
    task=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python3.12')
    task.output_uri='http://10.100.34.118:8081'
    task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='bootstrap.py',
        diff=(executor.parent/'bootstrap.py').read_text())
    task.set_packages(['clearml==2.1.2'])
    task.set_base_docker(IMAGE,docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
        '-e CLEARML_FILES_HOST=http://10.100.34.118:8081 -e NCCL_P2P_DISABLE=1 '
        '-e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_TF32_OVERRIDE=0 '
        '-e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
    task.set_parameters(dict(plan=canonical(plan).decode(),recipe_sha256=digest,
        semantic_profile_identity=semantic,seed=2027,world_size=choice['world_size'],GPU_model_restriction=False,
        actual_dispatcher_sha256=sha(__file__),historical_parent_dispatcher_sha256=base['dispatcher_sha256'],
        independent_reader_source_freeze_sha256=prepared['reader_source_freeze_sha256'],
        operational_GPU_memory_target_percent=[75,80],memory_target_reached=False,TF32_enabled=False,
        ETA='unknown_until_actual_transfer_or_real_event_progress',paper_performance_complete=False))
    task.add_tags(['Recover-Before-Fuse','all-class-top64','real-prefix-function-profile','diagnostic-only',
        'unchanged-original-replay','independent-prefix-admission-pending'])
    task.reload();assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==base['bootstrap_sha256']
    job=dict(seed=2027,task_id=task.id,semantic_profile_identity=semantic,recipe_sha256=digest,plan=plan,
        queue=choice['queue_name'],bindings=choice['bindings'],eligible_workers=choice['eligible_workers'],fleet=fleet,
        status='created_before_enqueue',ETA='unknown',actual_dispatcher_sha256=sha(__file__))
    profile_jobs.append(job)

    def save():
        now=datetime.datetime.now(datetime.timezone.utc)
        value=dict(kind='rbf_original_real_prefix_function_profile_disjoint_GPU_dispatch_v1',
            checked_at_utc=now.isoformat(),jobs=profile_jobs,preparation_sha256=sha(ROOT/'preparation.json'),
            GPU_model_restriction=False,L40S_CPU_only=True,physical_collision_forbidden=True,
            operational_GPU_memory_target_percent=[75,80],no_artificial_memory_padding=True,
            core_dispatch_priority_preserved=True,full_forest_or_latency_or_paper_accepted=False)
        atomic(JOURNAL,value)
        receipt=R/'receipts'/('rbf-original-real-prefix-function-profile-GPU-dispatch-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        new(receipt,value);register(receipt,value['kind'])

    save()
    fresh,_=snapshot(api,core)
    assert any(q['queue_id']==choice['queue_id'] for q in fresh), 'capacity changed; preserve created diagnostic Task'
    Task.enqueue(task,queue_id=choice['queue_id']);task.reload();job['status']=str(task.status);save()
    print(json.dumps(dict(task_id=task.id,status=str(task.status),queue=choice['queue_name'],
        world_size=choice['world_size'],full_forest_or_paper_accepted=False)),flush=True)


if __name__=='__main__':main()
