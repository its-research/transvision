#!/usr/bin/env python3
"""Deduplicate and dispatch data-free priority GPU readiness on idle 3090/V100."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

PROJECT = 'Thesis/Recover-Before-Fuse/Training'
REFERENCE_TASK = 'adc3b281eb6f4a299acc1ae7da1e4be2'
SCRIPT_SHA256 = '292d76736398f321cc79711e3d0cdf69f1171abd396bdcdcca677fdcc4e1ab00'
TARGETS = {
    '3090': {'worker': '10.100.34.27-3090:gpu4,5,6,7',
             'queue_id': 'a42b106a00ce4ac09a42454df7541e89', 'queue_name': 'GPU4-3090'},
    'V100': {'worker': '10.100.34.26-V100:gpu0,1,2,3',
             'queue_id': '3925e906ce484620a941e6ccedc4bdbd', 'queue_name': 'GPU4-V100'},
}


def clean_overlap(worker_id):
    host, mask = worker_id.split(':gpu')
    return host, {int(x) for x in mask.split(',')}


def available(api, family):
    target = TARGETS[family]
    queue = api.queues.get_by_id(queue=target['queue_id'])
    if queue.name != target['queue_name'] or queue.entries:
        return False, 'queue identity differs or has entries'
    workers = api.workers.get_all()
    matches = [w for w in workers if w.id == target['worker']]
    if len(matches) != 1:
        return False, 'target worker missing or ambiguous'
    worker = matches[0]
    now = datetime.datetime.now(datetime.timezone.utc)
    if (worker.task is not None or worker.last_activity_time is None
            or (now - worker.last_activity_time).total_seconds() >= 120
            or not any(q.id == target['queue_id'] for q in (worker.queues or []))):
        return False, 'target worker busy, stale or outside queue'
    host, mask = clean_overlap(target['worker'])
    for other in workers:
        if other.id == target['worker'] or other.task is None or ':gpu' not in other.id:
            continue
        other_host, other_mask = clean_overlap(other.id)
        if other_host == host and mask & other_mask:
            return False, 'physical GPU overlaps another live task'
    return True, 'verified idle non-overlapping four-GPU worker'


def dispatch(family):
    from clearml import Task
    from clearml.backend_api import Session
    from clearml.backend_api.session.client import APIClient
    if urlparse(Session.get_api_server_host()).hostname != '10.100.35.118':
        raise ValueError('ClearML API host differs')
    target = TARGETS[family]
    reference = Task.get_task(task_id=REFERENCE_TASK)
    script = reference.data.script.diff or ''
    if (str(reference.status) != 'completed'
            or hashlib.sha256(script.encode()).hexdigest() != SCRIPT_SHA256
            or reference.artifacts['readiness-receipt'].hash !=
            '92f00f9df25c7563383dd18af8450cd20dc5ec0299877b349c0b1f2ecb64bc59'):
        raise ValueError('successful reference readiness source differs')
    name = f'RBF priority float64 NCCL readiness {family} idle-four-v1 {SCRIPT_SHA256[:12]}'
    existing = Task.get_tasks(project_name=PROJECT, task_name='^' + re.escape(name) + '$')
    if existing:
        if len(existing) != 1:
            raise ValueError('multiple same-name readiness tasks exist')
        task = existing[0]
        if (hashlib.sha256((task.data.script.diff or '').encode()).hexdigest() != SCRIPT_SHA256
                or str(task.get_parameters().get('General/gpu_family')) != family):
            raise ValueError('existing readiness task differs')
        return {'task_id': task.id, 'status': str(task.status), 'duplicate_not_created': True}
    ok, reason = available(APIClient(), family)
    if not ok:
        return {'not_enqueued': reason, 'duplicate_not_created': True}
    task = Task.clone(source_task=reference, name=name, parent=reference.id)
    task.set_parameters({'gpu_family': family, 'script_sha256': SCRIPT_SHA256})
    task.add_tags(['Recover-Before-Fuse', 'priority-gpu-readiness', family,
                   'no-dataset', 'no-parameter-training'])
    task.reload()
    if (hashlib.sha256((task.data.script.diff or '').encode()).hexdigest() != SCRIPT_SHA256
            or str(task.get_parameters().get('General/gpu_family')) != family
            or task.data.container != reference.data.container):
        raise ValueError('cloned readiness task differs; inspect created task before retry')
    Task.enqueue(task, queue_id=target['queue_id'])
    return {'task_id': task.id, 'status': str(task.status),
            'queue_id': target['queue_id'], 'worker_target': target['worker'],
            'duplicate_not_created': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=TARGETS, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError('fresh dispatch receipt required')
    row = dispatch(args.family)
    receipt = {'kind': 'rbf_priority_secondary_gpu_readiness_dispatch_v1',
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu_family': args.family, 'reference_task_id': REFERENCE_TASK,
        'script_sha256': SCRIPT_SHA256, 'readiness': row,
        'dataset_read': False, 'parameter_training': False,
        'hardware_ready': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
