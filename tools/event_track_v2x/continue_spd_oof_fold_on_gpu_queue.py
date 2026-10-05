#!/usr/bin/env python3
"""Wait for an existing upload, accept bytes, then deduplicate/dispatch one OOF fit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--seed', type=int, choices=(1337, 2027, 3407), default=1337)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--upload-task-id', required=True)
    p.add_argument('--queue', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    from submit_spd_official_oof_fold_gpu4_libgl1_v5 import eligible_workers
    recorded = json.loads((a.package / 'clearml-package-task.json').read_bytes())
    if recorded['task_id'] != a.upload_task_id:
        raise ValueError('watcher upload identity mismatch')
    if a.receipt.exists():
        raise FileExistsError('continuation receipt is create-once')
    previous = None
    def report(stage):
        nonlocal previous
        if stage != previous:
            print(json.dumps({'stage': stage, 'fold_id': a.fold, 'queue': a.queue,
                              'eta': 'unknown', 'checked_at_utc': datetime.now(timezone.utc).isoformat()}),
                  flush=True)
            previous = stage
    while True:
        task = Task.get_task(task_id=a.upload_task_id)
        if task.status in ('failed', 'stopped', 'closed', 'published'):
            raise RuntimeError('upload terminal without required completed identity: ' + str(task.status))
        if task.status == 'completed' and (a.package / 'clearml-upload-acceptance.json').is_file():
            break
        report('waiting-existing-upload-' + str(task.status))
        time.sleep(30)
    folder = Path(__file__).parent
    if not (a.package / 'clearml-independent-readback.json').exists():
        report('independent-complete-byte-readback')
        subprocess.run([sys.executable, '-u', str(folder / 'accept_spd_oof_package_remote112_v3.py'),
                        '--fold', str(a.fold), '--package', str(a.package)], check=True)
    while not eligible_workers(APIClient(), a.queue):
        report('waiting-collision-free-four-gpu-worker')
        time.sleep(30)
    report('dispatching-admitted-fit')
    command = [sys.executable, '-u', str(folder / 'submit_spd_official_oof_fold_gpu4_libgl1_v5.py'),
               '--fold', str(a.fold), '--seed', str(a.seed), '--package', str(a.package), '--queue', a.queue]
    subprocess.run(command, check=True)
    acceptance = a.package / ('fold-%d-seed-%d-libgl1-v5-training-enqueue-acceptance.json' % (a.fold, a.seed))
    result = json.loads(acceptance.read_bytes())
    if result['fold_id'] != a.fold or result['seed'] != a.seed or result['queue'] != a.queue:
        raise ValueError('dispatch result differs from continuation request')
    receipt = {'kind': 'spd_oof_existing_upload_dependency_continuation_v1',
               'fold_id': a.fold, 'seed': a.seed, 'queue': a.queue,
               'upload_task_id': a.upload_task_id, 'training_task_id': result['task_id'],
               'enqueue_acceptance': str(acceptance), 'command': command,
               'status': 'dispatch-confirmed-only', 'training_complete': False,
               'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    with a.receipt.open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write('\n')
    report('dispatch-confirmed-not-training-complete')


if __name__ == '__main__':
    main()
