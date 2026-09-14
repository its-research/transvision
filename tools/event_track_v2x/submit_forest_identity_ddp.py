#!/usr/bin/env python3
"""Submit independently seeded four-A100 jobs under the current hardware policy.

Default is a read-only resource report. --execute uploads a hash-pinned minimal
source/derived-train package and enqueues the selected jobs. The CLI is always
A100-only; legacy mixed-hardware helpers remain for historical reproduction.
Seeds may run serially when only one four-A100 group is free. No worker/queue
config, existing jobs, V100/5090 workers or eight-GPU workers are modified. New training
checkpoint retention is part of this deployment, not the historical 19-file
result-publication authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from collections import Counter

PROJECT = 'Thesis/Recover-Before-Fuse/Training'
IMAGE = 'gitlab.zhht.ai.com:5000/aitech/model_infer:py-3.12-cuda-12.6.2-torch-2.10-ultralytics-8.4.13-dvc-v2-onnx-clearml'
MATRIX = ((1337, 'A100', 'GPU4-A100'), (2027, '5090', 'GPU4-5090'), (3407, '5090', 'GPU4-5090'))
HOSTS = {'A100': '10.100.34.18-A100', '5090': '10.100.34.130-5090'}


def selected_jobs(seeds, *, a100_fallback=False):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s not in (1337,2027,3407) for s in seeds):
        raise ValueError('distinct predeclared seeds required')
    if a100_fallback:
        if 1337 in seeds:
            raise ValueError('A100 fallback is restricted to the remaining 2027/3407 seeds')
        return tuple((seed, 'A100', 'GPU4-A100') for seed,_,_ in MATRIX if seed in seeds)
    return tuple(row for row in MATRIX if row[0] in seeds)


def require_capacity(ready, jobs, *, a100_fallback=False):
    needed = Counter(family for _,family,_ in jobs)
    if a100_fallback:
        if set(needed) != {'A100'}:
            raise ValueError('serial fallback requires only four-A100 jobs')
        needed['A100'] = 1  # Existing queue serializes jobs while one group is free.
    actual = Counter(item['family'] for item in ready)
    if any(actual[family] < count for family,count in needed.items()):
        raise ValueError('insufficient disjoint idle four-GPU workers for selected seeds')


def reject_duplicate_seed(records, *, own_task_id=None):
    if any(record['id'] != own_task_id and record['status'] in ('queued','in_progress','completed')
           for record in records):
        raise ValueError('this package/seed already has an active or completed task')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True)


def slots(worker_id):
    match = re.fullmatch(r'(.+):gpu(\d+(?:,\d+)*)', worker_id)
    return (match[1], frozenset(map(int, match[2].split(',')))) if match else (worker_id, frozenset())


def available(workers, queues):
    busy = [slots(w['id']) for w in workers if (w.get('task') or {}).get('id')
            or any(queues.get(q['id'], {}).get('entries') for q in w.get('queues') or [])]
    answer = []
    for w in workers:
        host, cards = slots(w['id'])
        family = next((f for f,h in HOSTS.items() if h == host), None)
        if not family or len(cards) != 4 or (w.get('task') or {}).get('id'):
            continue
        if any(h == host and cards & used for h,used in busy):
            continue
        for q in w.get('queues') or []:
            if q['name'] == 'GPU4-'+family:
                state = queues.get(q['id'])
                if state is not None and not state.get('entries') and 'force_workers:off' not in state.get('tags', []):
                    answer.append(dict(worker=w['id'], family=family, queue=q['name'], cards=sorted(cards)))
    return answer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path)
    p.add_argument('--controller', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--attempt', type=int, default=1)
    p.add_argument('--seeds', type=int, nargs='+', choices=(1337,2027,3407), default=(1337,2027,3407))
    p.add_argument('--a100-fallback', action='store_true',
                   help='Legacy remaining-seed selector; all CLI submissions are now A100-only.')
    a = p.parse_args(argv)
    if a.attempt < 1:
        raise ValueError('positive explicit attempt required')
    selected = tuple((seed, 'A100', 'GPU4-A100') for seed, _, _ in
                     selected_jobs(a.seeds, a100_fallback=a.a100_fallback))
    from clearml import Task
    from clearml.backend_api import Session
    from clearml.backend_api.session.client import APIClient
    if urlparse(Session.get_api_server_host()).hostname != '10.100.35.118':
        raise ValueError('only the user-designated private ClearML server is allowed')
    client = APIClient()
    def snapshot():
        workers = [w.to_dict() for w in client.workers.get_all(last_seen=120)]
        queues = {q.id:q.to_dict() for q in client.queues.get_all()}
        return [row for row in available(workers, queues) if row['family'] == 'A100']
    ready = snapshot()
    print(json.dumps(dict(available_four_gpu_workers=ready, selected_identity_seeds=selected,
        excluded_families=['V100', '5090'], active_hardware_policy='A100-only',
        snapshot_is_reservation=False, training_submitted_by_this_read=False), sort_keys=True), flush=True)
    if not a.execute:
        return
    if a.package is None or a.controller is None:
        raise ValueError('explicit package and controller required to execute')
    metadata_path = a.package/'package.json'
    metadata, package_sha = json.loads(metadata_path.read_bytes()), sha(metadata_path)
    if (metadata['kind'] != 'forest_identity_ddp_package_v1' or metadata['split'] != 'train'
            or metadata['class_scope'] != ['car'] or metadata['raw_GT_included'] is not False
            or metadata['full_official_train'] is not True):
        raise ValueError('authorized derived full-train car package required')
    for item in metadata['artifacts']:
        if item['path'] not in ('source.tar.gz', 'train-rows.tar.gz'):
            raise ValueError('unexpected package payload')
        path = a.package/item['path']
        if path.is_symlink() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('package payload differs before upload')
    require_capacity(ready, selected, a100_fallback=True)
    package_receipt = a.package/'clearml-package.json'
    if package_receipt.exists():
        old = json.loads(package_receipt.read_bytes())
        if old['package_sha256'] != package_sha:
            raise ValueError('existing upload belongs to another package')
        package_task = Task.get_task(task_id=old['task_id'])
    else:
        package_task = Task.create(project_name=PROJECT, task_name='DDP derived-train package '+package_sha[:12],
                                   task_type=Task.TaskTypes.data_processing)
        save(package_receipt, dict(task_id=package_task.id, package_sha256=package_sha))
    uploads = [('package', metadata_path)]+[(r['path'], a.package/r['path']) for r in metadata['artifacts']]
    for name, path in uploads:
        current = Task.get_task(task_id=package_task.id)
        if name in current.artifacts and current.artifacts[name].hash == sha(path):
            continue
        if not package_task.upload_artifact(name, artifact_object=path, wait_on_upload=True):
            raise RuntimeError('package upload failed')
        if Task.get_task(task_id=package_task.id).artifacts[name].hash != sha(path):
            raise RuntimeError('package upload readback differs')
    package_task.mark_completed(force=True)
    ready = snapshot()
    require_capacity(ready, selected, a100_fallback=True)
    controller = a.controller.read_text()
    controller_sha = sha(a.controller)
    jobs = []
    for seed, family, queue in selected:
        suffix = '' if a.attempt == 1 else f'-attempt{a.attempt}'
        receipt_path = a.package/f'clearml-seed-{seed}{suffix}.json'
        name = f'RBF row-identity seed-{seed} four-{family} '+package_sha[:12]
        if a.attempt != 1:
            name += f' attempt-{a.attempt}'
        old = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else None
        if old is not None and (old.get('gpu_family') != family or old.get('queue') != queue):
            raise ValueError('saved submission violates current A100-only policy; do not reuse or migrate it')
        cohort = Task.get_tasks(project_name=PROJECT, task_name=
            '^'+re.escape(f'RBF row-identity seed-{seed} four-')+'(?:A100|5090) '
            +re.escape(package_sha[:12])+r'(?: attempt-\d+)?$')
        reject_duplicate_seed([dict(id=t.id,status=str(t.status)) for t in cohort],
                              own_task_id=old['task_id'] if old else None)
        if receipt_path.exists():
            if old['package_sha256'] != package_sha or old['controller_sha256'] != controller_sha:
                raise ValueError('saved submission source identity differs')
            task = Task.get_task(task_id=old['task_id'])
        else:
            existing = Task.get_tasks(project_name=PROJECT, task_name='^'+re.escape(name)+'$')
            if existing:
                raise ValueError('task with exact campaign name exists but local receipt is missing')
            task = Task.create(project_name=PROJECT, task_name=name, task_type=Task.TaskTypes.training, binary='python3.12')
            save(receipt_path, dict(task_id=task.id, seed=seed, gpu_family=family, queue=queue,
                                   package_sha256=package_sha, controller_sha256=controller_sha))
        if str(task.status) in ('queued', 'in_progress', 'completed'):
            jobs.append(dict(seed=seed, task_id=task.id, queue=queue, status=str(task.status)))
            continue
        if str(task.status) != 'created':
            raise ValueError('failed/stopped task requires a new explicit attempt, not silent retry')
        task.set_script(repository='', branch='', commit='', working_dir='.',
                        entry_point='run_clearml_forest_identity_ddp.py', diff=controller)
        task.set_packages(['clearml==2.1.2', 'numpy==1.26.4', 'scipy==1.14.1', 'cryptography==46.0.5'])
        task.set_base_docker(IMAGE, docker_arguments='-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 16g '
                             '--env NCCL_P2P_DISABLE=1 --env NVIDIA_DRIVER_CAPABILITIES=compute '
                             '--env CLEARML_FILES_HOST=http://10.100.35.118:8081')
        task.set_parameters(dict(package_task_id=package_task.id, package_sha256=package_sha, seed=seed,
            gpu_family=family, required_world_size=4, global_batch_size=64, epochs=10,
            class_scope='car', controller_sha256=controller_sha, paper_eligible=False,
            authorization='user-requested-ClearML-four-GPU-multi-machine-training-20260913',
            V100_excluded=True, RTX5090_excluded=True, a100_only=True,
            hardware_mixed_not_same_compute_benchmark=True,
            a100_fallback=a.a100_fallback, concurrent_execution_required=False))
        task.add_tags(['Recover-Before-Fuse', 'car-only', 'train-only', 'four-GPU-DDP',
                       'development-not-OOF', 'paper-evidence-pending', 'seed-'+str(seed)])
        Task.enqueue(task, queue_name=queue)
        jobs.append(dict(seed=seed, task_id=task.id, queue=queue, status=str(Task.get_task(task_id=task.id).status)))
        print('RBF_SUBMITTED '+json.dumps(jobs[-1]), flush=True)
    print('RBF_CAMPAIGN '+json.dumps(dict(jobs=jobs, gpus_per_job=4,
        requested_concurrent_gpus=None, concurrent_execution_guaranteed=False,
        a100_fallback=a.a100_fallback, a100_only=True,
        physical_hosts=len({HOSTS[f] for _,f,_ in selected}),
        complete_three_seed_submission=len(selected)==3, paper_eligible=False)), flush=True)


if __name__ == '__main__':
    main()
