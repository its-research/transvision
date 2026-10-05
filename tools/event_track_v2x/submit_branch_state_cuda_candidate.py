"""One source-bound CUDA state measurement after core experiment dispatch.

GPU model is unrestricted. Prefer the smallest eligible worker, reserving its
entire binding while declaring that this candidate computes on one device.
No duplicate, restart, synthetic memory fill, or promotion is permitted.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re

from rbf_nested_seen_val_v2_common import R, new, register, sha
from submit_rbf_final_identity import atomic, binding, intersects, IMAGE
from submit_rbf_seen_val_joint_identity import active_reservations

TRANSPORT = R/'source-freezes/rbf-branch-state-CUDA-transport-producer-v2-witness-20261004'
TRANSPORT_SHA = '4b87a785fe0f60d01c80fd27872295857eff3a45d40c9caaae223eb06288c3e9'
CPU_ACCEPTANCE_SHA = 'a26c27c62f3eba4b76640d2ece2cbee5d5e1a151242da59e8271da79170069f1'
MANIFEST_SHA = 'ce8bf75c2b1c29aba37088aac6f286487097d8026a32ce3a5d718cc1d1313213'
ARCHIVE_SHA = 'e73b7f3a968f2a3a0bc1db7f17b6a22d0cf11dae9fb8b5655646ceaf53010ac1'
JOURNAL = R/'receipts/rbf-branch-state-CUDA-collision-safe-dispatch-20261005.json'
PROJECT = 'Thesis/Recover-Before-Fuse/Inference'
CORE_JOURNALS = [R/'receipts'/name for name in (
    'rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json',
    'rbf-final-refit-full-train-fixed-topK-GPU-dispatch-20261004.json',
    'rbf-final-refit-full-train-fixed-Top1-GPU-dispatch-20261004.json')]
OTHER = R/'receipts/rbf-batched-complete-sequence-GPU-dispatch-20261004.json'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def eligible(workers, queues, now, reserved=()):
    pending = {q['id']:bool(q.get('entries')) for q in queues}
    choices = []
    for queue in queues:
        match = re.match(r'^GPU(?:([1-9][0-9]*)-|-([1-9][0-9]*)$)',queue['name'])
        if not match or 'L40' in queue['name'] or pending[queue['id']] or 'force_workers:off' in queue.get('tags',[]): continue
        count = int(match[1] or match[2]); idle = []; unsafe = False
        for worker in workers:
            if not any(q['id'] == queue['id'] for q in worker.get('queues',[])) or worker.get('task',{}).get('id'): continue
            key = binding(worker)
            stamp = datetime.datetime.fromisoformat(str(worker['last_activity_time']).replace('Z','+00:00'))
            if 'L40' in worker['id'] or key is None or len(key[1]) != count or not 0 <= (now-stamp).total_seconds() < 90:
                unsafe = True; continue
            busy = any(intersects(key,binding(other)) and (other.get('task',{}).get('id') or
                any(pending.get(q['id'],False) for q in other.get('queues',[]))) for other in workers if other['id'] != worker['id'])
            if busy or any(intersects(key,item) for item in reserved): unsafe = True; continue
            if not any(intersects(key,binding(other)) for other in idle): idle.append(worker)
        if idle and not unsafe:
            choices.append(dict(queue_id=queue['id'],queue_name=queue['name'],worker_bound_GPU_count=count,
                eligible_workers=[w['id'] for w in idle],bindings=[[binding(w)[0],sorted(binding(w)[1])] for w in idle]))
    return sorted(choices,key=lambda q:(q['worker_bound_GPU_count'],q['queue_name']))


def core_jobs():
    jobs = []; missing = []
    for path in CORE_JOURNALS:
        entries = json.loads(path.read_bytes())['jobs'] if path.exists() else []
        assert len({j['seed'] for j in entries}) == len(entries)
        assert {j['seed'] for j in entries} <= {1337,2027,3407}
        missing.extend(dict(journal=str(path),seed=s) for s in (1337,2027,3407) if s not in {j['seed'] for j in entries})
        jobs.extend(entries)
    assert len({j['task_id'] for j in jobs}) == len(jobs)
    return jobs,missing


def plan_from_publication(path):
    proof = json.loads(path.read_bytes())
    assert proof['kind'] == 'rbf_branch_state_CUDA_input_full_independent_cloud_byte_readback_v1'
    assert proof['full_cloud_bytes_independently_read'] is True
    assert proof['source_freeze_sha256'] == TRANSPORT_SHA and proof['CPU_acceptance_sha256'] == CPU_ACCEPTANCE_SHA
    assert set(proof['artifacts']) == {'inputs','manifest','local-package-readback'}
    for key,spec in proof['artifacts'].items():
        assert spec['task'] == proof['task_id'] and spec['key'] == key
        cloud = path.parent/'independent-cloud-bytes'/key
        assert sha(cloud) == spec['sha256'] and cloud.stat().st_size == spec['bytes']
    assert proof['artifacts']['inputs']['sha256'] == ARCHIVE_SHA and proof['artifacts']['inputs']['bytes'] == 167431250
    assert proof['artifacts']['manifest']['sha256'] == MANIFEST_SHA
    return dict(recipe='rbf_one_complete_sequence_independent_root_state_CUDA_candidate_v1',
        inputs=proof['artifacts']['inputs'],manifest=proof['artifacts']['manifest'],
        CPU_acceptance_sha256=CPU_ACCEPTANCE_SHA,bootstrap_sha256=sha(TRANSPORT/'run_branch_state_cuda_candidate.py'),
        publication_sha256=sha(path),required_compute_GPUs=1,TF32_enabled=False,
        memory_target_percent=[75,80],no_artificial_memory_padding=True,
        sequence='0000',events=195,max_batch=64,full_cohort_or_paper_acceptance=False)


def run(args, prepared):
    own = Path(__file__).resolve().parent
    if not args.publication.is_file():
        print(json.dumps(dict(status='waiting_for_explicitly_authorized_independent_input_publication',GPU_task_created=False,ETA='unknown')),flush=True)
        return
    plan = plan_from_publication(args.publication)
    core,missing = core_jobs()
    if missing:
        print(json.dumps(dict(status='remaining_core_experiments_have_dispatch_priority',missing=missing,GPU_task_created=False,ETA='unknown')),flush=True)
        return
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    for job in core:
        task = Task.get_task(task_id=job['task_id'])
        assert json.loads(task.get_parameters()['General/plan']) == job['plan']
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == job['plan']['bootstrap_sha256']
        if str(task.status) in ('created','unknown'):
            print(json.dumps(dict(status='finish_existing_core_dispatch',task_id=task.id,GPU_task_created=False)),flush=True)
            return
    jobs = json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []
    assert len(jobs) <= 1
    identity = hashlib.sha256(canonical(plan)).hexdigest()
    name = 'RBF branch-state CUDA complete-sequence measurement '+identity[:16]
    matches = Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
    assert len(matches) <= 1
    if matches:
        task = matches[0]
        assert len(jobs) == 1 and jobs[0]['task_id'] == task.id, 'preserve unjournaled existing Task for inspection'
        assert json.loads(task.get_parameters()['General/plan']) == plan
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
        print(json.dumps(dict(existing_task_id=task.id,status=str(task.status),duplicate_not_created=True)),flush=True)
        return
    assert not jobs, 'existing Task must never be replaced'
    for key in ('inputs','manifest'):
        spec = plan[key]; source = Task.get_task(task_id=spec['task'])
        assert str(source.status) == 'completed'
        assert source.artifacts[spec['key']].hash == spec['sha256'] and source.artifacts[spec['key']].size == spec['bytes']
    api = APIClient()
    others = core + (json.loads(OTHER.read_bytes())['jobs'] if OTHER.exists() else [])
    def snapshot():
        workers = [w.to_dict() for w in api.workers.get_all()]
        queues = [q.to_dict() for q in api.queues.get_all()]
        now = datetime.datetime.now(datetime.timezone.utc)
        statuses = {job['task_id']:str(Task.get_task(task_id=job['task_id']).status) for job in others}
        choices = eligible(workers,queues,now,active_reservations(others,workers,statuses))
        fleet = dict(checked_at_utc=now.isoformat(),
            workers=[dict(id=w['id'],ip=w.get('ip'),task_id=w.get('task',{}).get('id'),
                queues=[q['id'] for q in w.get('queues',[])],last_activity_time=str(w['last_activity_time'])) for w in workers],
            queues=[dict(id=q['id'],name=q['name'],pending=[e['task'] for e in q.get('entries',[])]) for q in queues])
        return choices,fleet
    choices,fleet = snapshot()
    if not choices:
        print(json.dumps(dict(status='waiting_for_collision_free_worker',GPU_task_created=False,ETA='unknown')),flush=True); return
    choice = choices[0]
    if not args.execute:
        print(json.dumps(dict(choice=choice,plan=plan,execute=False,actual_compute_GPU_count=1)),flush=True); return
    task = Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python3.12')
    job = dict(task_id=task.id,plan=plan,recipe_sha256=identity,queue=choice['queue_name'],
        bindings=choice['bindings'],eligible_workers=choice['eligible_workers'],
        worker_bound_GPU_count=choice['worker_bound_GPU_count'],actual_compute_GPU_count=1,
        fleet=fleet,status='created_before_configuration',ETA='unknown')
    jobs.append(job)
    def save():
        now = datetime.datetime.now(datetime.timezone.utc)
        value = dict(kind='rbf_branch_state_CUDA_collision_safe_dispatch_v1',jobs=jobs,
            source_freeze_sha256=sha(own/'source-freeze.json'),checked_at_utc=now.isoformat(),
            GPU_model_restriction=False,L40S_CPU_only=True,physical_collision_forbidden=True,
            GPU_worker_binding_is_not_compute_GPU_count=True,full_cohort_or_paper_accepted=False)
        atomic(JOURNAL,value)
        path = R/'receipts'/('rbf-branch-state-CUDA-dispatch-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        new(path,value); register(path,value['kind'])
    save()
    task.output_uri = 'http://10.100.34.118:8081'
    task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='run_branch_state_cuda_candidate.py',
        diff=(TRANSPORT/'run_branch_state_cuda_candidate.py').read_text())
    task.set_packages(['clearml==2.1.2'])
    task.set_base_docker(IMAGE,docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
        '-e CLEARML_FILES_HOST=http://10.100.34.118:8081 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility '
        '-e NVIDIA_TF32_OVERRIDE=0 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
    task.set_parameters(dict(plan=canonical(plan).decode(),recipe_sha256=identity,
        semantic_experiment_identity=identity,required_compute_GPUs=1,
        worker_bound_GPU_count=choice['worker_bound_GPU_count'],GPU_model_restriction=False,TF32_enabled=False,
        independent_receiver_source_freeze_sha256=sha(own/'source-freeze.json'),
        ETA='unknown_until_real_task_progress',paper_performance_complete=False))
    task.add_tags(['Recover-Before-Fuse','branch-state-CUDA','complete-original-sequence','single-compute-device','independent-acceptance-pending'])
    task.reload(); assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
    fresh,_ = snapshot()
    # A changed eligible worker set requires a new bounded inspection, not an enqueue.
    assert choice in fresh, 'capacity or bindings changed; created Task retained'
    Task.enqueue(task,queue_id=choice['queue_id'])
    task.reload(); job['status'] = str(task.status); save()
    print(json.dumps(dict(task_id=task.id,status=str(task.status),queue=choice['queue_name'],
        actual_compute_GPU_count=1,worker_bound_GPU_count=choice['worker_bound_GPU_count'],accepted=False)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication',type=Path,default=R/'artifacts/rbf-branch-state-CUDA-input-cloud-publication-v2-20261004/independent-publication.json')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    own = Path(__file__).resolve().parent
    prepared = json.loads((own/'source-freeze.json').read_bytes())
    for name,record in prepared['sources'].items(): assert sha(own/name) == record['sha256']
    for record in prepared['references']: assert sha(record['path']) == record['sha256']
    assert sha(TRANSPORT/'source-freeze.json') == TRANSPORT_SHA
    # Serialize this dispatcher's create/dedup/journal/enqueue sequence locally.
    with open(str(JOURNAL)+'.dispatch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run(args,prepared)


if __name__ == '__main__': main()
