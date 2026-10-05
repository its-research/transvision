"""Frozen-recipe all-class refits, using disjoint idle GPU4 or GPU8 workers."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

R = Path('/Volumes/Data/test/recover-before-fuse')
PROJECT = 'Thesis/Recover-Before-Fuse/Training'
IMAGE = 'gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def binding(worker):
    match = re.fullmatch(r'([^:]+):gpu([0-9]+(?:,[0-9]+)*)',worker['id'])
    return None if match is None else (worker.get('ip') or match[1],set(map(int,match[2].split(','))))


def intersects(left,right):
    return left is not None and right is not None and left[0] == right[0] and bool(left[1] & right[1])


def available(workers,queues,now,reserved=()):
    pending = {q['id']:bool(q.get('entries')) for q in queues}
    answer = []
    for queue in queues:
        match = re.match(r'^GPU(4|8)-',queue['name'])
        if not match or 'L40' in queue['name'] or pending[queue['id']] or 'force_workers:off' in queue.get('tags',[]):
            continue
        size = int(match[1])
        idle = []
        unsafe = False
        for worker in workers:
            if not any(q['id']==queue['id'] for q in worker.get('queues',[])) or worker.get('task',{}).get('id'):
                continue
            key = binding(worker)
            stamp = datetime.datetime.fromisoformat(str(worker['last_activity_time']).replace('Z','+00:00'))
            if key is None or len(key[1]) != size or not 0 <= (now-stamp).total_seconds() < 90:
                unsafe = True
                continue
            busy = any(intersects(key,binding(other)) and (other.get('task',{}).get('id')
                or any(pending.get(q['id'],False) for q in other.get('queues',[])))
                for other in workers if other['id'] != worker['id'])
            if busy or any(intersects(key,item) for item in reserved):
                unsafe = True
                continue
            if not any(intersects(key,binding(other)) for other in idle):
                idle.append(worker)
        if idle and not unsafe:
            answer.append(dict(queue_id=queue['id'],queue_name=queue['name'],world_size=size,
                eligible_workers=[w['id'] for w in idle],
                bindings=[[binding(w)[0],sorted(binding(w)[1])] for w in idle]))
    return sorted(answer,key=lambda x:(-x['world_size'],0 if 'A100' in x['queue_name'] else 1 if 'V100' in x['queue_name'] else 2,x['queue_name']))


def atomic(path,value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(temp,path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    prepared = json.loads(args.preparation.read_bytes())
    root = args.preparation.parent
    for name,digest in prepared['execution_sources'].items():
        assert sha(root/name) == digest
    assert prepared['full_train_inputs_and_frozen_selection_source_bound'] is True
    api = APIClient()
    def snapshot(reserved):
        workers = [w.to_dict() for w in api.workers.get_all()]
        queues = [q.to_dict() for q in api.queues.get_all()]
        now = datetime.datetime.now(datetime.timezone.utc)
        return available(workers,queues,now,reserved), dict(checked_at_utc=now.isoformat(),
            workers=[dict(id=w['id'],ip=w.get('ip'),task_id=w.get('task',{}).get('id'),
                          queues=[q['id'] for q in w.get('queues',[])],
                          last_activity_time=str(w['last_activity_time'])) for w in workers],
            queues=[dict(id=q['id'],name=q['name'],pending=[e['task'] for e in q.get('entries',[])]) for q in queues])
    journal = R/'receipts/rbf-nested-selected-all-class-full-train-refit-GPU-dispatch-20261004.json'
    jobs = json.loads(journal.read_bytes())['jobs'] if journal.exists() else []
    reserved = [(j['bindings'][0][0],set(j['bindings'][0][1])) for j in jobs]
    def save():
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        value = dict(kind='rbf_nested_selected_all_class_final_refit_GPU_dispatch_v1',
            checked_at_utc=now,preparation_sha256=sha(args.preparation),jobs=jobs,
            GPU_model_restriction=False,L40S_CPU_only=True,paper_performance_complete=False)
        atomic(journal,value)
        version = R/'receipts'/('rbf-nested-selected-all-class-full-train-refit-GPU-dispatch-'
            +datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        with version.open('x') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False)
            stream.write('\n')
        ledger_path = R/'receipts/20260928-execution-ledger.json'
        with open(str(ledger_path)+'.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger = json.loads(ledger_path.read_bytes())
            ledger['entries'].append(dict(kind=value['kind'],receipt=str(version),
                receipt_sha256=sha(version),checked_at_utc=now,task_ids=[j['task_id'] for j in jobs],
                journal_revision=len(jobs),goal_status='active'))
            ledger['updated_at_utc'] = now
            atomic(ledger_path,ledger)
    for seed in (2027,1337,3407):
        item = next(r for r in prepared['seeds'] if r['seed']==seed)
        selection = item['selection']
        assert selection['candidate_protocol']=='rbf-all-class-top64-v1'
        for key in ('source','checkpoint','weights_archive','manifest','rows'):
            spec = item['inputs'][key]
            task = Task.get_task(task_id=spec['task'])
            assert str(task.status)=='completed'
            artifact = task.artifacts[spec['key']]
            assert artifact.hash==spec['sha256'] and artifact.size==spec['bytes']
        # Semantic identity excludes GPU type/card count, so a different free
        # worker cannot generate a second training experiment for this recipe.
        identity = hashlib.sha256(canonical(item)).hexdigest()
        name = f'RBF all-class full-train frozen-nested-epoch refit seed{seed} '+identity[:16]
        matches = Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
        assert len(matches)<=1
        if matches:
            params = matches[0].get_parameters()
            assert params['General/semantic_training_identity']==identity
            prior_recipe=json.loads(params['General/recipe'])
            assert prior_recipe['runner_sha256']==prepared['runner_sha256'], 'existing source differs; preserve task'
            print(json.dumps(dict(seed=seed,existing_task_id=matches[0].id,status=str(matches[0].status),duplicate_not_created=True)),flush=True)
            continue
        choices,fleet = snapshot(reserved)
        if not choices:
            print(json.dumps(dict(seed=seed,status='waiting_for_disjoint_idle_GPU',ETA='unknown')),flush=True)
            continue
        choice = choices[0]
        print(json.dumps(dict(seed=seed,choice=choice,execute=args.execute)),flush=True)
        if not args.execute:
            reserved.extend((host,set(cards)) for host,cards in choice['bindings'])
            continue
        recipe = dict(item['inputs'],seed=seed,fit_config=selection['fit_config'],
            world_size=choice['world_size'],runner_sha256=prepared['runner_sha256'],
            frozen_selection_receipt_sha256=sha(args.preparation),
            source_binding='original_published_joint_training_modules_immutable',
            independent_checkpoint_acceptance=False,paper_performance_complete=False)
        digest = hashlib.sha256(canonical(recipe)).hexdigest()
        task = Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.training,binary='python3.12')
        task.output_uri = 'http://10.100.34.118:8081'
        task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='bootstrap.py',
                        diff=(root/'bootstrap.py').read_text())
        task.set_packages(['clearml==2.1.2'])
        task.set_base_docker(IMAGE,docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
            '-e CLEARML_FILES_HOST=http://10.100.34.118:8081 -e NCCL_P2P_DISABLE=1 '
            '-e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_TF32_OVERRIDE=0 '
            '-e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
        task.set_parameters(dict(recipe=canonical(recipe).decode(),recipe_sha256=digest,
            semantic_training_identity=identity,seed=seed,GPU_model_restriction=False,
            world_size=choice['world_size'],epochs=selection['fit_config']['epochs'],
            ETA='unknown_until_actual_optimizer_progress',paper_performance_complete=False))
        task.add_tags(['Recover-Before-Fuse','all-class-top64','joint-cross-source-temporal',
                       'full-official-train-refit','frozen-nested-selection','independent-acceptance-pending'])
        task.reload()
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==sha(root/'bootstrap.py')
        job = dict(seed=seed,task_id=task.id,recipe_sha256=digest,recipe=recipe,
            semantic_training_identity=identity,queue=choice['queue_name'],bindings=choice['bindings'],
            eligible_workers=choice['eligible_workers'],fleet=fleet,status='created_before_enqueue',
            ETA='unknown_until_actual_optimizer_progress')
        jobs.append(job)
        save()
        fresh,_ = snapshot(reserved)
        assert any(q['queue_id']==choice['queue_id'] for q in fresh),'capacity changed; created task retained'
        Task.enqueue(task,queue_id=choice['queue_id'])
        reserved.extend((host,set(cards)) for host,cards in choice['bindings'])
        task.reload()
        job['status'] = str(task.status)
        save()
        print(json.dumps(dict(seed=seed,task_id=task.id,queue=choice['queue_name'],world_size=choice['world_size'],status=str(task.status))),flush=True)


if __name__=='__main__':
    main()
