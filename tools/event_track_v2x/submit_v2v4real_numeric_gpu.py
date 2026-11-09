"""Source-bound, deduplicated dispatch for missing post-head numerical evidence."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re

from tools.event_track_v2x.submit_rbf_final_identity import available,binding,IMAGE
from tools.event_track_v2x.prepare_v2v4real_numeric_gpu_source import RAW_TASKS
from tools.event_track_v2x import run_v2v4real_official_test_gpu_shards as original
from tools.event_track_v2x import v2v4real_official_test_runtime as runtime

PROJECT='Thesis/Recover-Before-Fuse/Inference'
R=Path('/Volumes/Data/test/recover-before-fuse')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualify_source(root):
    root=Path(root);manifest=json.loads((root/'source-manifest.json').read_bytes())
    if sha(root/'bootstrap.py')!=manifest['bootstrap_sha256']:raise ValueError('bootstrap differs')
    for name,row in manifest['sources'].items():
        p=root/'execution-code'/name
        if p.is_symlink() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('execution closure differs')
    identity=hashlib.sha256(json.dumps(manifest['sources'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if identity!=manifest['execution_bundle_sha256']:raise ValueError('bundle identity differs')
    gate=json.loads((root/'dispatch-qualification.json').read_bytes())
    if gate.get('bootstrap_sha256')!=manifest['bootstrap_sha256'] or gate.get('software_ready_for_dispatch') is not True:
        raise ValueError('exact dispatch software qualification required')
    if gate.get('dispatcher_sha256')!=sha(__file__):raise ValueError('dispatcher differs from gate')
    checker=root/'independent-checker/preparation.json'
    if sha(checker)!=gate.get('checker_preparation_sha256'):raise ValueError('independent checker gate differs')
    for name,row in json.loads(checker.read_bytes())['sources'].items():
        p=root/'independent-checker'/name
        if p.is_symlink() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('independent checker source differs')
    for row in gate['evidence']:
        if sha(row['path'])!=row['sha256']:raise ValueError('dispatch qualification evidence changed')
    return manifest


def existing_candidate(tasks, *, seed, manifest):
    if len(tasks)>1:raise ValueError('ambiguous numeric candidate; retain all tasks')
    if not tasks:return None
    task=tasks[0];p=task.get_parameters()
    if (p.get('General/execution_bundle_sha256')!=manifest['execution_bundle_sha256']
            or int(p['General/seed'])!=seed or p.get('General/accepted_raw_task_id')!=RAW_TASKS[seed]):
        raise ValueError('existing numeric candidate source/input differs')
    return task


def verify_uploaded_parameters(actual, expected):
    # ClearML returns stored values as strings by default. Require the exact
    # serialization, rather than changing numerical tolerances or bool truthiness.
    for key,value in expected.items():
        observed=actual.get('General/'+key)
        if not ((type(observed) is type(value) and observed==value)
                or (type(observed) is str and observed==str(value))):
            raise ValueError('uploaded config differs; candidate retained')


def qualify_resume(task, *, seed, manifest):
    existing_candidate([task],seed=seed,manifest=manifest)
    if hashlib.sha256(task.data.script.diff.encode()).hexdigest()!=manifest['bootstrap_sha256']:
        raise ValueError('retained candidate execution bytes differ')
    if str(task.status)!='created':return False
    if task.artifacts:raise ValueError('created candidate already has runtime outputs; retain without enqueue')
    return True


def dispatch(root,seed,*,execute,per_rank_bytes,archive_bytes,raw_bytes,resume_task_id=None):
    if seed not in RAW_TASKS or any(type(v) is not int or v<=0 for v in (per_rank_bytes,archive_bytes,raw_bytes)):
        raise ValueError('registered seed and explicit storage budgets required')
    manifest=qualify_source(root)
    from clearml import Task
    from clearml.backend_api.session import Session
    from clearml.backend_api.session.client import APIClient
    Session();api=APIClient()
    task_name=f"V2V4Real original post-head numeric seed{seed} frames{original.export.FRAME_SHA[:12]} v1"
    existing=(Task.get_task(task_id=resume_task_id) if resume_task_id else
        existing_candidate(Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(task_name)+'$'),seed=seed,manifest=manifest))
    if existing and (not resume_task_id or not qualify_resume(existing,seed=seed,manifest=manifest)):
        return dict(task_id=existing.id,status=str(existing.status),duplicate_not_created=True)
    # Check the original registered inputs before publishing candidate code.
    pin=manifest['accepted_raw_proofs'][str(seed)]
    proof=json.loads((Path(root)/'execution-code'/pin['path']).read_bytes())
    required=[(original.SOURCE_TASK,'source',original.export.SOURCE_ARCHIVE_SHA),
        (original.export.PROJECTION_TASK,'projection',original.PROJECTION_ARCHIVE_SHA),
        (original.export.PROJECTION_TASK,'publication-receipt',original.PUBLICATION_SHA),
        (original.CHECKPOINT_TASKS[seed],'best-checkpoint',original.export.CHECKPOINTS[seed]),
        *[(RAW_TASKS[seed],name,proof['input_sha256'][name]) for name in ('native-features','native-feature-manifest','native-feature-index')],
        (runtime.OFFLINE_TASK,'libgl',runtime.OFFLINE_ARCHIVE_SHA),(runtime.OFFLINE_TASK,'manifest',runtime.OFFLINE_MANIFEST_SHA)]
    for owner,name,digest in required:
        task=Task.get_task(task_id=owner)
        if str(task.status)!='completed' or task.artifacts[name].hash!=digest:raise ValueError('original registered asset changed')
    choices=available([w.to_dict() for w in api.workers.get_all()],[q.to_dict() for q in api.queues.get_all()],datetime.datetime.now(datetime.timezone.utc))
    if existing:
        choices=[c for c in choices if c['world_size']==int(existing.get_parameters()['General/world_size'])]
    if not choices:return dict(status='waiting_for_idle_disjoint_GPU_worker',ETA=None,retained_task_id=existing.id if existing else None)
    choice=choices[0];result=dict(seed=seed,**choice,status='eligible_candidate',execute=execute)
    if not execute:return result
    parameters=dict(seed=seed,world_size=choice['world_size'],execution_bundle_sha256=manifest['execution_bundle_sha256'],
        accepted_raw_task_id=RAW_TASKS[seed],accepted_raw_proof_sha256=pin['sha256'],
        max_trace_bytes_per_rank=per_rank_bytes,max_archive_bytes=archive_bytes,max_raw_unpacked_bytes=raw_bytes,
        checkpoint_sha256=original.export.CHECKPOINTS[seed],projection_frames_sha256=original.export.FRAME_SHA,
        gt_read=False,TF32_enabled=False,network_forward_verified=False,paper_metric_complete=False)
    if existing:
        task=existing
        verify_uploaded_parameters(task.get_parameters(),parameters)
    else:
        task=Task.create(project_name=PROJECT,task_name=task_name,task_type=Task.TaskTypes.inference,binary='python')
        task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='numeric_bootstrap.py',diff=(Path(root)/'bootstrap.py').read_text())
        task.set_packages(runtime.PACKAGES)
        task.set_base_docker(IMAGE,docker_arguments='--network host --shm-size 64g -e CLEARML_AGENT_FORCE_TASK_INIT=0 --env NVIDIA_DRIVER_CAPABILITIES=compute,utility')
        task.set_parameters(parameters)
        task.add_tags(['Recover-Before-Fuse','original-post-head-numeric-v1',f'seed-{seed}'])
    remote=Task.get_task(task_id=task.id)
    if hashlib.sha256(remote.data.script.diff.encode()).hexdigest()!=manifest['bootstrap_sha256']:
        raise ValueError('uploaded execution bytes differ; candidate retained')
    read_parameters=remote.get_parameters()
    verify_uploaded_parameters(read_parameters,parameters)
    current=available([w.to_dict() for w in api.workers.get_all()],[q.to_dict() for q in api.queues.get_all()],datetime.datetime.now(datetime.timezone.utc))
    if choice not in current:raise ValueError('physical GPU fleet changed; candidate retained without enqueue')
    Task.enqueue(task,queue_id=choice['queue_id'])
    return dict(result,task_id=task.id,status='queued',uploaded_source_read_back=True,resumed_same_created_candidate=bool(existing))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--resume-task-id')
    p.add_argument('--seed',type=int,choices=sorted(RAW_TASKS),required=True);p.add_argument('--execute',action='store_true')
    for name in ('per-rank-bytes','archive-bytes','raw-bytes'):p.add_argument('--'+name,type=int,required=True)
    args=vars(p.parse_args());root=args.pop('source');seed=args.pop('seed')
    # Serialize our dispatchers across source/config readback and final fleet check.
    with (R/'receipts/numeric-dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=dispatch(root,seed,**args)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        receipt=R/'receipts'/f'v2v4real-numeric-dispatch-{stamp}.json'
        receipt.write_text(json.dumps(dict(result,source=str(root),bootstrap_sha256=sha(root/'bootstrap.py'),independent_numeric_acceptance=False),sort_keys=True)+'\n')
    print(json.dumps(dict(receipt=str(receipt),**result),sort_keys=True))

if __name__=='__main__':main()
