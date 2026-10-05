#!/usr/bin/env python3
"""Deduplicate and queue one data-free SPD MHT Linux runtime probe on L40S."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

PROJECT = 'Thesis/Recover-Before-Fuse/Training'
QUEUE_ID = '8d0f8b54037249eeb0f1cc70cbfe73ab'
IMAGE = ('gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:'
         'e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb')
PACKAGES = ['clearml==2.1.2', 'numpy==1.26.4', 'scipy==1.14.1', 'torch==2.9.1']
THREAD_ENV = ('OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS '
              'VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS BLIS_NUM_THREADS '
              'PYTHONHASHSEED PYTHONDONTWRITEBYTECODE').split()


def submit(source):
    from clearml import Task
    from clearml.backend_api import Session
    from clearml.backend_api.session.client import APIClient
    if urlparse(Session.get_api_server_host()).hostname != '10.100.35.118':
        raise ValueError('private ClearML API host differs')
    q = APIClient().queues.get_by_id(queue=QUEUE_ID)
    if q.name != 'GPU3-L40S':
        raise ValueError('L40S queue identity differs')
    payload = source.read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    name = 'SPD MHT K4 Linux L40S CPU runtime probe ' + source_sha[:12]
    matches = Task.get_tasks(project_name=PROJECT,
                             task_name='^' + re.escape(name) + '$')
    if matches:
        if len(matches) != 1 or hashlib.sha256(
                (matches[0].data.script.diff or '').encode()).hexdigest() != source_sha:
            raise ValueError('existing runtime probe identity differs')
        return {'task_id': matches[0].id, 'status': str(matches[0].status),
                'source_sha256': source_sha, 'duplicate_not_created': True}
    task = Task.create(project_name=PROJECT, task_name=name,
                       task_type=Task.TaskTypes.testing, binary='python3.12')
    task.set_script(repository='', branch='', commit='', working_dir='.',
                    entry_point=source.name, diff=payload.decode())
    task.set_packages(PACKAGES)
    env = ' '.join('--env ' + key + '=1' for key in THREAD_ENV)
    # PYTHONHASHSEED must be zero; all other pinned thread values are one.
    env = env.replace('--env PYTHONHASHSEED=1', '--env PYTHONHASHSEED=0')
    task.set_base_docker(IMAGE, docker_arguments=(
        '-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 4g '
        '--env CUDA_VISIBLE_DEVICES= --env NVIDIA_DRIVER_CAPABILITIES=compute '
        '--env CLEARML_FILES_HOST=http://10.100.35.118:8081 ' + env))
    task.set_parameters({'probe_source_sha256': source_sha,
                         'runtime_packages': ','.join(PACKAGES),
                         'device': 'cpu', 'dataset_read': False,
                         'parameter_training': False,
                         'frozen_contract_changed': False})
    task.add_tags(['Recover-Before-Fuse', 'SPD', 'MHT-K4', 'L40S-CPU',
                   'runtime-probe', 'no-dataset', 'no-metrics'])
    task.reload()
    if (hashlib.sha256((task.data.script.diff or '').encode()).hexdigest() != source_sha
            or task.data.container is None):
        raise ValueError('created task differs; inspect before enqueue')
    Task.enqueue(task, queue_id=QUEUE_ID)
    return {'task_id': task.id, 'status': str(task.status),
            'source_sha256': source_sha, 'queue_id': QUEUE_ID,
            'duplicate_not_created': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    if a.receipt.exists() or a.receipt.is_symlink():
        raise ValueError('fresh dispatch receipt required')
    row = submit(a.source)
    receipt = {'kind': 'spd_mht_l40s_linux_cpu_runtime_probe_dispatch_v1',
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_path': str(a.source), 'packages': PACKAGES, 'image': IMAGE,
        'result': row, 'dataset_read': False, 'inference_started': False,
        'metrics_computed': False, 'frozen_contract_changed': False}
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
