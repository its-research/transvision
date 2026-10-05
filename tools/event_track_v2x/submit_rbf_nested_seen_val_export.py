"""Deduplicate six matching val producers and use collision-safe idle GPU4s."""
import argparse
import base64
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile
from submit_rbf_final_identity import available, binding

R = Path('/Volumes/Data/test/recover-before-fuse')
PROJECT = 'Thesis/Recover-Before-Fuse/Training'
IMAGE = 'gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb'
JOURNAL = R/'receipts/rbf-nested-detector-matching-seen-val-raw-GPU-dispatch-20261004.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(temporary,path)


def save(jobs, preparation_sha256):
    now = datetime.datetime.now(datetime.timezone.utc)
    value = dict(kind='rbf_nested_detector_matching_seen_val_raw_GPU4_dispatch_v1',
        checked_at_utc=now.isoformat(),preparation_sha256=preparation_sha256,jobs=jobs,
        GPU_model_restriction=False,L40S_CPU_only=True,full_online_RBF_accepted=False,
        paper_performance_complete=False)
    write(JOURNAL,value)
    path = R/'receipts'/('rbf-nested-detector-matching-seen-val-raw-GPU-dispatch-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False)
        stream.write('\n')
    ledger_path = R/'receipts/20260928-execution-ledger.json'
    with open(str(ledger_path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_bytes())
        ledger['entries'].append(dict(kind=value['kind'],receipt=str(path),receipt_sha256=sha(path),
            task_ids=[j['task_id'] for j in jobs],checked_at_utc=now.isoformat(),goal_status='active'))
        ledger['updated_at_utc'] = now.isoformat()
        write(ledger_path,ledger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    prepared = json.loads(args.preparation.read_bytes())
    root = args.preparation.parent
    assert prepared['full_input_payload_and_schema_admitted'] is True
    gate = json.loads(Path(prepared['input_admission_receipt']).read_bytes())
    assert sha(prepared['input_admission_receipt']) == prepared['input_admission_receipt_sha256']
    assert gate['full_image_pose_input_coverage_verified'] is True
    assert gate['GT_payloads_included'] is False and gate['test_payloads_included'] is False
    for name,item in prepared['source_inventory'].items():
        assert sha(root/name)==item['sha256'] and (root/name).stat().st_size==item['bytes']
    source_names = ['run_seen_val_cache.py','run_rbf_nested_seen_val_export.py',
        'spd_cache_primitives.py','cooptrack_raw_query_decode.py','spd_export_training_binding.py',
        'run_cooptrack_a100.py']
    source_inventory = {name:prepared['source_inventory'][name] for name in source_names}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(source_names):
            info = zipfile.ZipInfo(name,date_time=(2026,10,4,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,(root/name).read_bytes())
    raw = stream.getvalue()
    bootstrap = (root/'bootstrap_rbf_nested_seen_val_export.py').read_text().replace('__SOURCE_BASE64__',base64.b64encode(raw).decode())
    jobs = json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []
    api = APIClient()
    for item in prepared['jobs']:
        recipe = dict(item,source_inventory=source_inventory,
            execution_source_zip_sha256=hashlib.sha256(raw).hexdigest(),world_size=4,
            runtime_manifest=prepared['runtime_manifest'],libgl=prepared['libgl'],
            input_manifest=prepared['input_manifest'],inputs=prepared['inputs'],
            input_dataset_sha256=gate['input_manifest_sha256'],
            input_admission_receipt_sha256=sha(prepared['input_admission_receipt']),
            evaluation_scope='SPD-seen-val-exploratory-cache-only',TF32_enabled=False)
        identity = hashlib.sha256(canonical(recipe)).hexdigest()
        name = 'RBF matching nested detector seen-val raw seed%d %s %s'%(item['seed'],item['side'],identity[:16])
        matches = Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
        assert len(matches)<=1
        if matches:
            parameters = matches[0].get_parameters()
            assert parameters['General/semantic_export_identity']==identity
            assert matches[0].data.script.diff==bootstrap
            print(json.dumps(dict(seed=item['seed'],side=item['side'],existing_task_id=matches[0].id,
                status=str(matches[0].status),duplicate_not_created=True)),flush=True)
            continue
        workers = [w.to_dict() for w in api.workers.get_all()]
        queues = [q.to_dict() for q in api.queues.get_all()]
        now = datetime.datetime.now(datetime.timezone.utc)
        reserved = []
        for job in jobs:
            prior = Task.get_task(task_id=job['task_id'])
            if str(prior.status) in ('completed','failed','stopped','closed','published','aborted'):
                continue
            worker_id = prior.data.last_worker
            observed = next((w for w in workers if w['id']==worker_id and w.get('task',{}).get('id')==prior.id),None)
            if observed:
                reserved.append(binding(observed))
            else:
                reserved.extend((host,set(cards)) for host,cards in job['possible_physical_bindings'])
        choices = [x for x in available(workers,queues,now,reserved) if x['world_size']==4]
        if not choices:
            print(json.dumps(dict(seed=item['seed'],side=item['side'],status='waiting_for_disjoint_idle_GPU4',ETA='unknown')),flush=True)
            continue
        choice = choices[0]
        print(json.dumps(dict(seed=item['seed'],side=item['side'],choice=choice,execute=args.execute)),flush=True)
        if not args.execute:
            continue
        # Runtime, detector and collection identities are checked again before mutation.
        for artifact_spec in (recipe['collection'],recipe['inputs'],recipe['input_manifest'],recipe['runtime_manifest'],recipe['libgl']):
            t = Task.get_task(task_id=artifact_spec['task'])
            a = t.artifacts[artifact_spec['key']]
            assert str(t.status)=='completed' and a.hash==artifact_spec['sha256'] and a.size==artifact_spec['bytes']
        training = Task.get_task(task_id=item['training_task_id'])
        assert str(training.status)=='completed'
        assert training.artifacts[item['side']+'-final-checkpoint'].hash==item['checkpoint_sha256']
        task = Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python3.12')
        task.output_uri='http://10.100.34.118:8081'
        task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='bootstrap.py',diff=bootstrap)
        task.set_packages(['clearml==2.1.2'])
        task.set_base_docker(IMAGE,docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
            '-e CLEARML_FILES_HOST=http://10.100.34.118:8081 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility '
            '-e NVIDIA_TF32_OVERRIDE=0 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e PYTHONDONTWRITEBYTECODE=1')
        task.set_parameters(dict(recipe=canonical(recipe).decode(),recipe_sha256=identity,
            semantic_export_identity=identity,seed=item['seed'],side=item['side'],world_size=4,
            GPU_model_restriction=False,TF32_enabled=False,ETA='unknown until actual shard frame progress',
            independent_acceptance=False,paper_performance_complete=False))
        task.add_tags(['Recover-Before-Fuse','all-class-raw-queries','nested-detector-fit',
            'SPD-seen-val-exploratory','no-GT-inputs','TF32-disabled','independent-acceptance-pending'])
        task.reload()
        assert task.data.script.diff==bootstrap
        job = dict(seed=item['seed'],side=item['side'],task_id=task.id,recipe=recipe,
            semantic_export_identity=identity,bootstrap_sha256=hashlib.sha256(bootstrap.encode()).hexdigest(),
            queue=choice['queue_name'],queue_id=choice['queue_id'],possible_physical_bindings=choice['bindings'],
            eligible_workers=choice['eligible_workers'],status='created_before_enqueue',
            fleet_snapshot=dict(checked_at_utc=now.isoformat(),
                workers=[dict(id=w['id'],ip=w.get('ip'),task_id=w.get('task',{}).get('id'),
                    queues=[q['id'] for q in w.get('queues',[])],
                    last_activity_time=str(w['last_activity_time'])) for w in workers],
                queues=[dict(id=q['id'],name=q['name'],pending=[e['task'] for e in q.get('entries',[])]) for q in queues]))
        jobs.append(job)
        save(jobs,sha(args.preparation))
        fresh_workers = [w.to_dict() for w in api.workers.get_all()]
        fresh_queues = [q.to_dict() for q in api.queues.get_all()]
        fresh = available(fresh_workers,fresh_queues,datetime.datetime.now(datetime.timezone.utc),reserved)
        assert any(c['queue_id']==choice['queue_id'] for c in fresh),'capacity changed; created Task retained'
        Task.enqueue(task,queue_id=choice['queue_id'])
        task.reload()
        job['status']=str(task.status)
        save(jobs,sha(args.preparation))
        print(json.dumps(dict(seed=item['seed'],side=item['side'],task_id=task.id,
            queue=choice['queue_name'],status=str(task.status),ETA='unknown until actual progress')),flush=True)


if __name__=='__main__':
    main()
