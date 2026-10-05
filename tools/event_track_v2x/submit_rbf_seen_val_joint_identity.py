"""Use disjoint idle GPU4/GPU8 capacity for distinct admitted val experiments."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re

from rbf_nested_seen_val_v2_common import R, new, register, sha
from submit_rbf_final_identity import available, binding, intersects, atomic, IMAGE

CONSUMER = R/'source-freezes/rbf-matching-seen-val-joint-identity-GPU-input-consumer-v1-20261004'
PUB = R/'artifacts/rbf-matching-seen-val-admitted-row-cloud-publication-v1-20261004'
CK = R/'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004'
PROJECT = 'Thesis/Recover-Before-Fuse/Inference'
JOURNAL = R/'receipts/rbf-matching-seen-val-three-seed-joint-identity-GPU-dispatch-20261004.json'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def active_reservations(jobs,workers,statuses):
    """Release finished tasks; reserve only actual cards once a worker binds."""
    reserved = []
    for job in jobs:
        assigned = [binding(w) for w in workers if w.get('task',{}).get('id') == job['task_id']]
        assigned = [item for item in assigned if item is not None]
        if assigned:
            reserved.extend(assigned)
        elif statuses[job['task_id']] not in ('completed','failed','stopped','closed','published'):
            reserved.extend((host,set(cards)) for host,cards in job['bindings'])
    return reserved


def snapshot(api,jobs,dry_reserved=()):
    from clearml import Task
    workers = [w.to_dict() for w in api.workers.get_all()]
    queues = [q.to_dict() for q in api.queues.get_all()]
    now = datetime.datetime.now(datetime.timezone.utc)
    statuses = {job['task_id']:str(Task.get_task(task_id=job['task_id']).status) for job in jobs}
    reserved = active_reservations(jobs,workers,statuses)+list(dry_reserved)
    choices = available(workers,queues,now,reserved)
    physical, busy = set(), set()
    for worker in workers:
        key = binding(worker)
        if key is None or 'L40' in worker['id']:
            continue
        stamp = datetime.datetime.fromisoformat(str(worker['last_activity_time']).replace('Z','+00:00'))
        if not 0 <= (now-stamp).total_seconds() < 90:
            continue
        for card in key[1]:
            physical.add((key[0],card))
            if worker.get('task',{}).get('id'):
                busy.add((key[0],card))
    fleet = dict(checked_at_utc=now.isoformat(),physical_non_L40_GPUs=len(physical),
        physical_worker_task_bound_GPUs=len(busy),physical_unbound_GPUs=len(physical-busy),
        worker_binding_is_not_utilization_measurement=True,
        pipeline_task_live_statuses=statuses,
        finished_task_reservations_released=True,assigned_worker_uses_actual_card_binding=True,
        workers=[dict(id=w['id'],ip=w.get('ip'),task_id=w.get('task',{}).get('id'),
            queues=[q['id'] for q in w.get('queues',[])],last_activity_time=str(w['last_activity_time'])) for w in workers],
        queues=[dict(id=q['id'],name=q['name'],pending=[e['task'] for e in q.get('entries',[])]) for q in queues])
    return choices,fleet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    frozen = json.loads((CONSUMER/'preparation.json').read_bytes())
    for name,record in frozen['sources'].items():
        assert sha(CONSUMER/name) == record['sha256']
    assert sha(CONSUMER/'software-gate.json') == frozen['software_gate_sha256']
    execution_root = Path(__file__).resolve().parent
    prepared = json.loads((execution_root/'preparation.json').read_bytes())
    for name,record in prepared['sources'].items():
        assert sha(execution_root/name) == record['sha256']
    api = APIClient()
    jobs = json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []
    dry_reserved = []
    def save():
        now = datetime.datetime.now(datetime.timezone.utc)
        value = dict(kind='rbf_matching_seen_val_generic_GPU_joint_identity_dispatch_v1',
            checked_at_utc=now.isoformat(),jobs=jobs,execution_source_preparation_sha256=sha(execution_root/'preparation.json'),
            GPU_model_restriction=False,L40S_CPU_only=True,TF32_enabled=False,paper_performance_complete=False)
        atomic(JOURNAL,value)
        path = R/'receipts'/('rbf-matching-seen-val-joint-GPU-dispatch-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        new(path,value);register(path,value['kind'])
    for seed in (2027,1337,3407):
        publication_path = PUB/f'seed{seed}/independent-publication-acceptance.json'
        if not publication_path.exists():
            print(json.dumps(dict(seed=seed,status='waiting_for_independent_cloud_input_admission',ETA='unknown')),flush=True)
            continue
        proof = json.loads(publication_path.read_bytes())
        assert proof['cloud_input_bytes_independently_read'] is True and proof['ready_for_generic_GPU_inference'] is True
        assert proof['GPU_consumer_source_sha256'] == sha(CONSUMER/'run_rbf_seen_val_identity_gpu.py')
        accepted = json.loads((CK/f'seed{seed}/acceptance.json').read_bytes())
        source = dict(task='7a7586375f1c467b91956abf3a681d21',key='code',bytes=1207564,
            sha256='f3649c4232ac29379b3a0eb39b4965bd61d284a7a0e365e40292001da8c1547f')
        inputs = {key:proof['artifacts'][key] for key in ('rows','row-manifest','row-independent-admission')}
        inputs.update(source=source,checkpoint=accepted['artifacts']['checkpoint'],
            weights_archive=accepted['artifacts']['identity-training'])
        item = dict(inputs,seed=seed,producer_sha256=proof['GPU_consumer_source_sha256'],
            input_independent_publication_sha256=sha(publication_path),
            independent_complete_feature_history_row_admission=True,checkpoint_validation_selection=False,
            scope='SPD seen-val exploratory target-free NN row forward candidate only')
        semantic = hashlib.sha256(canonical(item)).hexdigest()
        name = f'RBF matching seen-val joint identity forward seed{seed} '+semantic[:16]
        matches = Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
        assert len(matches) <= 1
        if matches:
            task = matches[0]
            assert task.get_parameters()['General/semantic_inference_identity'] == semantic
            assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == item['producer_sha256']
            print(json.dumps(dict(seed=seed,existing_task_id=task.id,status=str(task.status),duplicate_not_created=True)),flush=True)
            continue
        for spec in inputs.values():
            asset = Task.get_task(task_id=spec['task'])
            assert str(asset.status) == 'completed'
            assert asset.artifacts[spec['key']].hash == spec['sha256'] and asset.artifacts[spec['key']].size == spec['bytes']
        choices,fleet = snapshot(api,jobs,dry_reserved)
        if not choices:
            path = R/'receipts'/('rbf-matching-seen-val-waiting-for-GPU-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
            new(path,dict(seed=seed,fleet=fleet,ETA='unknown',goal_status='active',no_duplicate_created=True))
            register(path,'rbf_seen_val_independently_admitted_input_waiting_for_disjoint_idle_GPU_v1')
            print(json.dumps(dict(seed=seed,status='waiting_for_disjoint_idle_GPU',ETA='unknown')),flush=True)
            continue
        choice = choices[0]
        print(json.dumps(dict(seed=seed,choice=choice,execute=args.execute,
            physical_non_L40_GPUs=fleet['physical_non_L40_GPUs'],physical_unbound_GPUs=fleet['physical_unbound_GPUs'])),flush=True)
        if not args.execute:
            dry_reserved.extend((host,set(cards)) for host,cards in choice['bindings'])
            continue
        plan = dict(item,world_size=choice['world_size'])
        task = Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python3.12')
        task.output_uri = 'http://10.100.34.118:8081'
        task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='run_rbf_seen_val_identity_gpu.py',
            diff=(CONSUMER/'run_rbf_seen_val_identity_gpu.py').read_text())
        task.set_packages(['clearml==2.1.2'])
        task.set_base_docker(IMAGE,docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
            '-e CLEARML_FILES_HOST=http://10.100.34.118:8081 -e NCCL_P2P_DISABLE=1 '
            '-e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_TF32_OVERRIDE=0 '
            '-e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
        task.set_parameters(dict(plan=canonical(plan).decode(),recipe_sha256=hashlib.sha256(canonical(plan)).hexdigest(),
            semantic_inference_identity=semantic,seed=seed,world_size=choice['world_size'],GPU_model_restriction=False,
            TF32_enabled=False,ETA='unknown_until_real_input_transfer_or_NN_row_progress',paper_performance_complete=False))
        task.add_tags(['Recover-Before-Fuse','seen-val-exploratory','joint-cross-source-temporal','all-class-top64',
            'frozen-nested-selected-final-refit','full-input-admitted','independent-output-acceptance-pending'])
        task.reload()
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == item['producer_sha256']
        job = dict(seed=seed,task_id=task.id,semantic_inference_identity=semantic,recipe_sha256=hashlib.sha256(canonical(plan)).hexdigest(),
            plan=plan,queue=choice['queue_name'],eligible_workers=choice['eligible_workers'],bindings=choice['bindings'],
            fleet=fleet,status='created_before_enqueue',ETA='unknown_until_real_input_transfer_or_NN_row_progress')
        jobs.append(job);save()
        fresh,_ = snapshot(api,jobs[:-1],dry_reserved)
        assert any(q['queue_id'] == choice['queue_id'] for q in fresh), 'resource changed; retain created task'
        Task.enqueue(task,queue_id=choice['queue_id'])
        task.reload();job['status'] = str(task.status);save()
        print(json.dumps(dict(seed=seed,task_id=task.id,queue=choice['queue_name'],world_size=choice['world_size'],
            status=str(task.status),independent_output_accepted=False)),flush=True)


if __name__ == '__main__':
    main()
