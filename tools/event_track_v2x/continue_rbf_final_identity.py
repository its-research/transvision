"""Per-seed byte admission and deduplicated new-checkpoint GPU forward.

Only the new full-train refit checkpoints are consumed. Accepted original
weights/outputs are never regenerated. No running or failed Task is restarted.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tarfile
import time

import requests
from clearml import Task
from clearml.backend_api.session import Session
from clearml.backend_api.session.client import APIClient

R=Path('/Volumes/Data/test/recover-before-fuse')
TRAIN_JOURNAL=R/'receipts/rbf-nested-selected-all-class-full-train-refit-GPU-dispatch-20261004.json'
PROJECT='Thesis/Recover-Before-Fuse/Inference'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def new(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False)
        f.write('\n')


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(tmp,path)


def register(receipt,kind):
    path=R/'receipts/20260928-execution-ledger.json'
    with open(str(path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=json.loads(path.read_bytes())
        if not any(x.get('receipt')==str(receipt) for x in value['entries']):
            value['entries'].append(dict(kind=kind,receipt=str(receipt),receipt_sha256=sha(receipt),
                checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active'))
            atomic(path,value)


def freeze(task,job):
    directory=R/f"artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed{job['seed']}"
    proof=directory/'acceptance.json'
    if proof.exists():
        admitted=json.loads(proof.read_bytes())
        assert admitted['task_id']==task.id and admitted['recipe_sha256']==job['recipe_sha256']
        for key,spec in admitted['artifacts'].items():
            assert task.artifacts[key].hash==spec['sha256'] and task.artifacts[key].size==spec['bytes']
            assert sha(directory/key)==spec['sha256']
        return admitted
    expected={'identity-training','recipe','runtime-admission','plan','receipt','checkpoint'}
    assert set(task.artifacts)==expected and str(task.status)=='completed'
    directory.mkdir(parents=True,exist_ok=False)
    specs={}
    for key in sorted(expected):
        artifact=task.artifacts[key]
        path=directory/key
        url=artifact.url.replace('10.100.35.118:8081','10.100.34.118:8081')
        with requests.get(url,headers={'Authorization':'Bearer '+Session().token},stream=True,timeout=(10,120)) as response:
            if response.status_code!=200:raise RuntimeError('artifact HTTP status '+str(response.status_code))
            with path.open('xb') as f:
                for chunk in response.iter_content(1024**2):f.write(chunk)
        assert sha(path)==artifact.hash and path.stat().st_size==artifact.size
        specs[key]=dict(task=task.id,key=key,sha256=artifact.hash,bytes=artifact.size)
    actual_recipe=json.loads((directory/'recipe').read_bytes())
    assert actual_recipe==job['recipe'] and hashlib.sha256(canonical(actual_recipe)).hexdigest()==job['recipe_sha256']
    checkpoint=json.loads((directory/'checkpoint').read_bytes())
    plan=json.loads((directory/'plan').read_bytes())
    receipt=json.loads((directory/'receipt').read_bytes())
    runtime=json.loads((directory/'runtime-admission').read_bytes())
    seed=job['seed'];epochs=job['recipe']['fit_config']['epochs'];world=job['recipe']['world_size']
    assert checkpoint['seed']==receipt['seed']==seed and checkpoint['selected_epoch']==receipt['epochs']==epochs
    assert checkpoint['plan_sha256']==sha(directory/'plan')
    assert checkpoint['row_protocol']['candidate_protocol']=='rbf-all-class-top64-v1'
    assert checkpoint['row_protocol']['class_scope']==['car','bicycle','pedestrian']
    assert checkpoint['labels_in_model_inputs'] is False and checkpoint['validation_or_test_selection'] is False
    assert checkpoint['selection_checkpoint_sha256']==job['recipe']['checkpoint']['sha256']
    assert checkpoint['selection']=='frozen_nested_selected_epoch_full_train_refit'
    assert plan['partition']=='all_46_official_train_sequences_after_frozen_internal_selection'
    assert len(checkpoint['partition']['fit'])==46 and checkpoint['partition']['holdout']==[]
    assert plan['global_batch']==64 and receipt['sequence_count']==46 and receipt['scheduled_frames']==7445
    assert runtime['world_size']==world and runtime['atol']==runtime['rtol']==1e-4 and runtime['optimizer_updates']==0
    assert len(runtime['runtime'])==world and len({x['uuid'] for x in runtime['runtime']})==world
    assert all(x['tf32_matmul'] is x['tf32_cudnn'] is False for x in runtime['runtime'])
    assert runtime['maximum_absolute_gradient_error']>=0
    assert receipt['independent_checkpoint_acceptance'] is False and receipt['paper_performance_complete'] is False
    root=directory/'archive-unpack';root.mkdir()
    with tarfile.open(directory/'identity-training') as archive:
        members=archive.getmembers()
        assert len({m.name for m in members})==len(members)
        assert all(not m.issym() and not m.islnk() and not m.name.startswith('/')
                   and '..' not in Path(m.name).parts and (m.isfile() or m.isdir()) for m in members)
        archive.extractall(root,filter='data')
    weights=root/f'training/seed-{seed}/weights.pt'
    assert sha(weights)==checkpoint['weights']['sha256']
    assert sha(root/'training/plan.json')==sha(directory/'plan')
    assert sha(root/f'training/seed-{seed}/checkpoint.json')==sha(directory/'checkpoint')
    history=json.loads((root/f'training/seed-{seed}/epochs.json').read_bytes())
    assert [x['epoch'] for x in history]==list(range(1,epochs+1))
    for row in history:
        assert all(row['counts'][k]>0 and math.isfinite(row['losses'][k]) and row['losses'][k]>=0
                   for k in ('cross_source','temporal'))
    manifest_path=R/f'artifacts/rbf-joint-identity-full-independent-numpy-v1-20261001/seed{seed}/rows-unpack/rows/manifest.json'
    manifest=json.loads(manifest_path.read_bytes())
    assert sha(manifest_path)==job['recipe']['manifest']['sha256']
    expected_steps=epochs*sum(math.ceil(x['supervised_rows']/64) for x in manifest['shards'])
    assert receipt['optimizer_steps']==expected_steps and checkpoint['partition']['fit']==manifest['sequences']
    accepted=dict(kind='rbf_new_final_refit_six_artifacts_independent_byte_metadata_admission_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_id=task.id,
        seed=seed,recipe_sha256=job['recipe_sha256'],artifacts=specs,
        all_six_registered_artifact_bytes_independently_read=True,
        frozen_nested_epoch_and_full_train_recipe_verified=True,
        two_head_training_and_producer_CPU_reference_gradient_report_bound=True,
        producer_runtime=runtime['runtime'],epochs=epochs,optimizer_steps=expected_steps,
        weights_sha256=checkpoint['weights']['sha256'],model_sha256=checkpoint['model_sha256'],
        independent_tensor_and_full_forward_acceptance=False,complete_online_method_accepted=False,
        paper_performance_complete=False,source_sha256=sha(__file__))
    new(proof,accepted);register(proof,accepted['kind'])
    print(json.dumps(dict(stage='new_refit_six_artifacts_independent_byte_admitted',seed=seed,
                         task_id=task.id,proof=str(proof))),flush=True)
    return accepted


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch-seconds',type=int,default=0)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    config=json.loads((root/'source-freeze.json').read_bytes())
    assert sha(__file__)==config['continuation_sha256']
    bootstrap=Path(config['original_forward_bootstrap'])
    assert sha(bootstrap)==config['original_forward_bootstrap_sha256']
    checker=Path(config['GPU_checker'])
    assert sha(checker)==config['GPU_checker_sha256']
    spec=importlib.util.spec_from_file_location('frozen_checker',checker)
    fleet=importlib.util.module_from_spec(spec);spec.loader.exec_module(fleet)
    journal=R/'receipts/rbf-new-final-refit-original-joint-all-row-GPU-forward-dispatch-20261004.json'
    jobs=json.loads(journal.read_bytes())['jobs'] if journal.exists() else []
    def save():
        now=datetime.datetime.now(datetime.timezone.utc)
        value=dict(kind='rbf_new_final_refit_original_joint_all_row_GPU_forward_dispatch_v1',
            checked_at_utc=now.isoformat(),jobs=jobs,continuation_sha256=sha(__file__),
            bootstrap_sha256=sha(bootstrap),completed_original_weights_repeated=False,
            independent_full_output_readback_pending=True,paper_performance_complete=False)
        atomic(journal,value)
        proof=R/'receipts'/('rbf-new-final-refit-GPU-forward-dispatch-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        new(proof,value);register(proof,value['kind'])
    started=time.monotonic();previous={}
    while True:
        terminal=0
        for job in json.loads(TRAIN_JOURNAL.read_bytes())['jobs']:
            task=Task.get_task(task_id=job['task_id']);task.reload();status=str(task.status)
            if previous.get(task.id)!=status:
                print(json.dumps(dict(stage='refit_dependency',seed=job['seed'],task_id=task.id,
                                     status=status,ETA='unknown_unless_logged_optimizer_progress')),flush=True)
                previous[task.id]=status
            if status in ('failed','stopped'):
                terminal+=1;continue
            if status!='completed':continue
            admitted=freeze(task,job)
            if any(x['seed']==job['seed'] for x in jobs):terminal+=1;continue
            plan=dict(seed=job['seed'],source=job['recipe']['source'],manifest=job['recipe']['manifest'],
                rows=job['recipe']['rows'],checkpoint=admitted['artifacts']['checkpoint'],
                weights_archive=admitted['artifacts']['identity-training'],bootstrap_sha256=sha(bootstrap),
                scope='new_frozen_nested_selected_full_train_refit_original_all_class_all_rows_forward_no_GT_scoring',
                numerical_atol=1e-4,numerical_rtol=1e-4,world_size=4)
            identity=hashlib.sha256(canonical(plan)).hexdigest()
            name=f"RBF final-refit original joint all-row GPU forward seed{job['seed']} "+identity[:16]
            matches=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
            assert len(matches)<=1
            if matches:
                assert matches[0].get_parameters()['General/recipe_sha256']==identity
                jobs.append(dict(seed=job['seed'],task_id=matches[0].id,plan=plan,recipe_sha256=identity,
                    status=str(matches[0].status),source_bound_byte_admission=str(R/f"artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed{job['seed']}/acceptance.json"),recovered_existing_not_recreated=True))
                save();terminal+=1;continue
            api=APIClient()
            def slots():
                workers=[w.to_dict() for w in api.workers.get_all()]
                queues=[q.to_dict() for q in api.queues.get_all()]
                now=datetime.datetime.now(datetime.timezone.utc)
                return [x for x in fleet.available(workers,queues,now) if x['world_size']==4],dict(checked_at_utc=now.isoformat(),workers=[dict(id=w['id'],ip=w.get('ip'),task_id=w.get('task',{}).get('id')) for w in workers])
            choices,snapshot=slots()
            if not choices:continue
            choice=choices[0]
            forward=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python3.12')
            forward.output_uri='http://10.100.34.118:8081'
            forward.set_script(repository='',branch='',commit='',working_dir='.',entry_point='bootstrap.py',diff=bootstrap.read_text())
            forward.set_packages(['clearml==2.1.2'])
            forward.set_base_docker(fleet.IMAGE,docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
                '-e CLEARML_FILES_HOST=http://10.100.34.118:8081 -e NCCL_P2P_DISABLE=1 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility '
                '-e NVIDIA_TF32_OVERRIDE=0 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
            forward.set_parameters(dict(plan=canonical(plan).decode(),recipe_sha256=identity,
                parent_refit_task_id=task.id,independent_byte_admission_sha256=sha(Path(R/f"artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed{job['seed']}/acceptance.json")),
                GPU_model_restriction=False,ETA='unknown_until_actual_row_progress',paper_performance_complete=False))
            forward.reload();assert hashlib.sha256(forward.data.script.diff.encode()).hexdigest()==sha(bootstrap)
            row=dict(seed=job['seed'],task_id=forward.id,plan=plan,recipe_sha256=identity,
                     queue=choice['queue_name'],eligible_workers=choice['eligible_workers'],fleet=snapshot,
                     status='created_before_enqueue',parent_refit_task_id=task.id,ETA='unknown_until_actual_row_progress')
            jobs.append(row);save()
            fresh,_=slots();assert any(x['queue_id']==choice['queue_id'] for x in fresh)
            Task.enqueue(forward,queue_id=choice['queue_id']);forward.reload();row['status']=str(forward.status);save()
            terminal+=1
            print(json.dumps(dict(stage='new_refit_GPU_forward_dispatched',seed=job['seed'],task_id=forward.id,queue=choice['queue_name'],status=str(forward.status))),flush=True)
        if terminal==3 or time.monotonic()-started>=args.watch_seconds:break
        time.sleep(30)


if __name__=='__main__':
    main()
