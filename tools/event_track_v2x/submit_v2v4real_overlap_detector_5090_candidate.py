#!/usr/bin/env python3
"""Publish and dispatch the isolated 5090 v1 diagnostic detector candidate."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse

PROJECT = 'Thesis/Recover-Before-Fuse/Training'
SOURCE_SHA256 = '122157453fa481ce27e9adb97528990bab439c8b9e30d5687229da6a57744f1a'
BOOTSTRAP_SHA256 = 'ff7017be47885d73f2a254f8830e7c071d869f7e06c2779c47dd562dee61a9ae'
QUEUE_ID = '5a84454c072349069e7b61af38637c6d'
IMAGE = ('gitlab.zhht.ai.com:5000/aitech/'
         'model_infer:py-3.12-cuda-12.6.2-torch-2.10-ultralytics-8.4.13-dvc-v2-onnx-clearml')
PACKAGES = ('clearml==2.1.2', 'numpy==1.26.4', 'scipy==1.14.1',
            'PyYAML==6.0.2', 'spconv-cu126==2.3.8', 'cumm-cu126==0.7.11',
            'open3d==0.19.0', 'shapely==2.0.7', 'tensorboardX==2.6.4',
            'einops==0.8.1')
SEEDS = (1337, 2027, 3407)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def existing_exact(Task, name):
    return Task.get_tasks(project_name=PROJECT, task_name='^' + re.escape(name) + '$')


def publish_source(Task, archive):
    name = 'V2V4Real overlap-controlled v1 5090 diagnostic source ' + SOURCE_SHA256[:12]
    existing = existing_exact(Task, name)
    if existing:
        if len(existing) != 1:
            raise ValueError('multiple diagnostic source tasks exist')
        task = existing[0]
        artifact = task.artifacts.get('source')
        if (str(task.status) != 'completed' or artifact is None
                or artifact.hash != SOURCE_SHA256 or artifact.size != archive.stat().st_size):
            raise ValueError('existing diagnostic source publication differs')
        return task, True
    task = Task.create(project_name=PROJECT, task_name=name,
                       task_type=Task.TaskTypes.data_processing)
    if not task.upload_artifact('source', artifact_object=archive, wait_on_upload=True):
        raise RuntimeError('source upload failed; inspect created task before retry')
    task.reload()
    artifact = task.artifacts.get('source')
    if artifact is None or artifact.hash != SOURCE_SHA256 or artifact.size != archive.stat().st_size:
        raise ValueError('uploaded source differs')
    task.mark_completed(force=True)
    return task, False


def idle_5090_slots():
    from clearml.backend_api.session.client import APIClient
    api = APIClient()
    q = api.queues.get_by_id(queue=QUEUE_ID)
    if q.name != 'GPU4-5090':
        raise ValueError('5090 queue identity differs')
    now = datetime.datetime.now(datetime.timezone.utc)
    idle = []
    for w in api.workers.get_all():
        if w.id not in ('10.100.34.130-5090:gpu0,1,2,3',
                        '10.100.34.130-5090:gpu4,5,6,7'):
            continue
        if (w.task is None and w.last_activity_time is not None
                and (now - w.last_activity_time).total_seconds() < 120):
            idle.append(w.id)
    return sorted(idle), len(q.entries or [])


def dispatch(Task, source, bootstrap, seeds):
    records = []
    for seed in seeds:
        name = (f'V2V4Real overlap-controlled v1 diagnostic PointPillar '
                f'seed{seed} {SOURCE_SHA256[:12]} 4x5090')
        existing = existing_exact(Task, name)
        if existing:
            records.append({'seed': seed, 'task_id': existing[0].id,
                            'status': str(existing[0].status), 'duplicate_not_created': True})
            continue
        idle, queued = idle_5090_slots()
        if len(idle) <= queued:
            records.append({'seed': seed, 'not_enqueued': 'no verified idle 5090 slot'})
            continue
        task = Task.create(project_name=PROJECT, task_name=name,
                           task_type=Task.TaskTypes.training, binary='python3')
        task.set_script(repository='', branch='', commit='', working_dir='.',
                        entry_point=bootstrap.name, diff=bootstrap.read_text())
        task.set_packages(list(PACKAGES))
        task.set_base_docker(IMAGE,
            docker_arguments='-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 64g --network host '
            '--env NCCL_P2P_DISABLE=1 --env NVIDIA_DRIVER_CAPABILITIES=compute,utility '
            '--env CLEARML_FILES_HOST=http://10.100.34.118:8081 '
            '--env CLEARML_APT_INSTALL=libgl1')
        task.set_parameters({'seed': seed, 'source_task_id': source.id,
            'source_sha256': SOURCE_SHA256, 'world_size': 4, 'workers': 8,
            'gpu_family': '5090', 'variant_id': 'v2v4real-train-overlap-controlled-v1',
            'protocol_id': 'v2v4real-nominal-10hz-formal-v1',
            'paper_eligible': False, 'formal_independent_test_eligible': False})
        task.add_tags(['Recover-Before-Fuse', 'V2V4Real', 'diagnostic-detector',
                       'overlap-controlled-v1', 'non-formal', 'PointPillar',
                       'DDP', '4x5090', f'seed-{seed}'])
        Task.enqueue(task, queue_id=QUEUE_ID)
        records.append({'seed': seed, 'task_id': task.id, 'status': str(task.status),
                        'queue_id': QUEUE_ID, 'duplicate_not_created': False})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--bootstrap', required=True, type=Path)
    p.add_argument('--receipt', required=True, type=Path)
    p.add_argument('--publish-only', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=[3407])
    a = p.parse_args()
    if sha(a.source) != SOURCE_SHA256 or sha(a.bootstrap) != BOOTSTRAP_SHA256:
        raise ValueError('candidate bytes differ from verified package')
    if a.receipt.exists():
        raise ValueError('fresh dispatch receipt required')
    from clearml import Task
    from clearml.backend_api import Session
    if urlparse(Session.get_api_server_host()).hostname != '10.100.35.118':
        raise ValueError('private ClearML API host differs')
    os.environ['CLEARML_FILES_HOST'] = 'http://10.100.35.118:8081'
    source, reused = publish_source(Task, a.source)
    rows = [] if a.publish_only else dispatch(Task, source, a.bootstrap, a.seeds)
    receipt = {'kind': 'v2v4real_overlap_controlled_v1_5090_diagnostic_dispatch_v1',
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_task_id': source.id, 'source_sha256': SOURCE_SHA256,
        'source_reused': reused, 'bootstrap_sha256': BOOTSTRAP_SHA256,
        'variant_id': 'v2v4real-train-overlap-controlled-v1',
        'paper_eligible': False, 'formal_independent_test_eligible': False,
        'tasks': rows, 'training_completed': False}
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
