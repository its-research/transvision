#!/usr/bin/env python3
"""Dispatch one diagnostic checkpoint forward check on an idle 4-GPU worker."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse


PROJECT = 'Thesis/Recover-Before-Fuse/Training'
IMAGE = ('gitlab.zhht.ai.com:5000/aitech/'
         'model_infer:py-3.12-cuda-12.6.2-torch-2.10-ultralytics-8.4.13-dvc-v2-onnx-clearml')
VERIFIER_SHA256 = '89a15b1e8a91ce3226da07041fc71f071c42c4c4fe5f88b6ce1a30499e7790f8'
TRAINING_BY_SEED = {
    1337: ('669833fb759742a4a857f99f87341e0b', '7f58517f4c62452cbef9b98c79ff1420',
           '93b3f63cb1058c725224e275c713566e2a11090812026060684c77867a3c3a69'),
    2027: ('f945af3976e34a479e40b56ef2a7be24', '7f58517f4c62452cbef9b98c79ff1420',
           '93b3f63cb1058c725224e275c713566e2a11090812026060684c77867a3c3a69'),
    3407: ('fd7a87ec5ffa43c3b28efc48bdc9b7e3', '79aeb14f8d4641f381150c3f912f623f',
           '122157453fa481ce27e9adb97528990bab439c8b9e30d5687229da6a57744f1a'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_freeze(seed, verifier, freeze):
    if verifier.is_symlink() or freeze.is_symlink():
        raise ValueError('ordinary frozen inputs required')
    if sha(verifier) != VERIFIER_SHA256:
        raise ValueError('frozen verifier source differs')
    receipt = json.loads(freeze.read_text())
    task_id, source_id, source_sha = TRAINING_BY_SEED[seed]
    if (receipt.get('kind') != 'v2v4real_overlap_controlled_v1_diagnostic_checkpoint_byte_freeze_v1'
            or receipt.get('seed') != seed or receipt.get('task_id') != task_id
            or receipt.get('source_task_id') != source_id
            or receipt.get('source_sha256') != source_sha
            or receipt.get('all_60_epochs_verified') is not True
            or receipt.get('checkpoint_bytes_frozen') is not True
            or receipt.get('paper_eligible') is not False
            or receipt.get('formal_independent_test_eligible') is not False):
        raise ValueError('completed diagnostic byte freeze required')
    row = receipt.get('artifacts', {}).get('best-checkpoint', {})
    checkpoint = Path(row.get('path', ''))
    if (not checkpoint.is_file() or checkpoint.is_symlink()
            or checkpoint.stat().st_size != row.get('bytes')
            or sha(checkpoint) != row.get('sha256')):
        raise ValueError('frozen checkpoint bytes differ')
    return receipt, row['sha256']


def idle_workers(queue_id):
    from clearml.backend_api.session.client import APIClient
    api = APIClient()
    queue = api.queues.get_by_id(queue=queue_id)
    if not queue.name.startswith('GPU4-'):
        raise ValueError('four-GPU queue required')
    now = datetime.datetime.now(datetime.timezone.utc)
    workers = api.workers.get_all()
    target = [worker for worker in workers if any(
        entry.id == queue_id for entry in worker.queues or [])]
    if not target:
        raise ValueError('requested four-GPU queue has no live worker')
    busy_devices = {}
    for worker in workers:
        if ':gpu' not in worker.id:
            continue
        host, suffix = worker.id.split(':gpu', 1)
        try:
            devices = {int(device) for device in suffix.split(',')}
        except ValueError as exc:
            raise ValueError('GPU worker device binding differs') from exc
        if not devices or not devices <= set(range(16)):
            raise ValueError('GPU worker device binding differs')
        if worker.task is not None:
            busy_devices.setdefault(host, set()).update(devices)
    idle = []
    for worker in target:
        if ':gpu' not in worker.id:
            continue
        host, suffix = worker.id.split(':gpu', 1)
        numbers = suffix.split(',')
        if len(numbers) != 4 or len(set(numbers)) != 4:
            continue
        devices = {int(device) for device in numbers}
        if (worker.task is None and worker.last_activity_time is not None
                and (now - worker.last_activity_time).total_seconds() < 120
                and not devices & busy_devices.get(host, set())):
            idle.append(worker.id)
    if len(idle) <= len(queue.entries or []):
        raise ValueError('no verified idle collision-free four-GPU worker')
    return sorted(idle), queue.name


def dispatch(seed, verifier, checkpoint_sha, attempt, queue_id):
    from clearml import Task
    from clearml.backend_api import Session
    if urlparse(Session.get_api_server_host()).hostname != '10.100.35.118':
        raise ValueError('private ClearML API host differs')
    os.environ['CLEARML_FILES_HOST'] = 'http://10.100.35.118:8081'
    task_id, source_id, source_sha = TRAINING_BY_SEED[seed]
    training = Task.get_task(task_id=task_id)
    source = Task.get_task(task_id=source_id)
    if (training.status != 'completed' or source.status != 'completed'
            or training.artifacts['best-checkpoint'].hash != checkpoint_sha
            or source.artifacts['source'].hash != source_sha):
        raise ValueError('completed ClearML checkpoint/source identity differs')
    name = (f'V2V4Real overlap v1 diagnostic checkpoint seed{seed} '
            f'{VERIFIER_SHA256[:12]} 4xGPU attempt{attempt}')
    existing = Task.get_tasks(project_name=PROJECT, task_name='^' + re.escape(name) + '$')
    if existing:
        if len(existing) != 1:
            raise ValueError('duplicate task identity is ambiguous')
        parameters = existing[0].get_parameters()
        if (parameters.get('General/seed') != str(seed)
                or parameters.get('General/checkpoint_sha256') != checkpoint_sha
                or parameters.get('General/verifier_source_sha256') != VERIFIER_SHA256):
            raise ValueError('existing task has different checkpoint identity')
        return {'task_id': existing[0].id, 'status': str(existing[0].status),
                'duplicate_not_created': True}
    available, queue_name = idle_workers(queue_id)
    task = Task.create(project_name=PROJECT, task_name=name,
                       task_type=Task.TaskTypes.testing, binary='python3')
    task.set_script(repository='', branch='', commit='', working_dir='.',
                    entry_point=verifier.name, diff=verifier.read_text())
    task.set_packages(['clearml==2.1.2'])
    task.set_base_docker(IMAGE,
        docker_arguments='-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 64g --network host '
        '--env NVIDIA_DRIVER_CAPABILITIES=compute,utility '
        '--env CLEARML_FILES_HOST=http://10.100.34.118:8081')
    task.set_parameters({'seed': seed, 'checkpoint_sha256': checkpoint_sha,
                         'checkpoint_task_id': task_id, 'source_task_id': source_id,
                         'source_sha256': source_sha, 'verifier_source_sha256': VERIFIER_SHA256,
                         'variant_id': 'v2v4real-train-overlap-controlled-v1',
                         'paper_eligible': False, 'formal_independent_test_eligible': False})
    task.add_tags(['Recover-Before-Fuse', 'V2V4Real', 'diagnostic-checkpoint',
                   'overlap-controlled-v1', 'non-formal', '4xGPU', f'seed-{seed}'])
    Task.enqueue(task, queue_id=queue_id)
    return {'task_id': task.id, 'status': str(task.status), 'queue_id': queue_id,
            'queue_name': queue_name, 'idle_workers_at_dispatch': available,
            'duplicate_not_created': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=TRAINING_BY_SEED, required=True)
    parser.add_argument('--verifier', type=Path, required=True)
    parser.add_argument('--freeze-receipt', type=Path, required=True)
    parser.add_argument('--dispatch-receipt', type=Path, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--queue-id', required=True)
    args = parser.parse_args()
    if args.attempt < 1 or args.dispatch_receipt.exists() or args.dispatch_receipt.is_symlink():
        raise ValueError('positive attempt and fresh dispatch receipt required')
    freeze, checkpoint_sha = verify_freeze(args.seed, args.verifier, args.freeze_receipt)
    result = dispatch(args.seed, args.verifier, checkpoint_sha, args.attempt,
                      args.queue_id)
    receipt = {'kind': 'v2v4real_overlap_diagnostic_checkpoint_gpu4_dispatch_v1',
               'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'seed': args.seed, 'attempt': args.attempt,
               'queue_id': args.queue_id,
               'training_task_id': freeze['task_id'],
               'checkpoint_sha256': checkpoint_sha,
               'byte_freeze_receipt_sha256': sha(args.freeze_receipt),
               'verifier_source_sha256': VERIFIER_SHA256,
               'paper_eligible': False, 'formal_independent_test_eligible': False,
               'gpu_forward_verified': False, 'dispatch': result}
    args.dispatch_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.dispatch_receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
