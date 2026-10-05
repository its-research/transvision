#!/usr/bin/env python3
"""Watch existing five fits; invoke immutable freezers only after completion.

Never create or restart training tasks. Partial/failed freezes require inspection.
This stage does not imply tensor, inference, calibration or paper acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
TASKS = ['517e10afb294404a96fc6c19f5ce88f6',
         '12b06f9674494dc381b4d6566d848b20',
         '6134c1c99e2141b3b919801f6874ff43',
         'ca63b926f4c9412b84ab2f77484fc7fe',
         'a4ad4a451bd94a79a96cea8f38b083c3']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def eligible(task):
    return str(task.status) == 'completed' and all(
        side + suffix in task.artifacts
        for side in ('vehicle-side', 'infrastructure-side')
        for suffix in ('-completion', '-final-checkpoint'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    inventory = json.loads((source / 'source-freeze-receipt.json').read_text())['inventory']
    def verify_sources():
        for r in inventory:
            if sha(source / r['name']) != r['sha256']:
                raise ValueError('immutable coordinator source differs: ' + r['name'])
    verify_sources()
    def event(kind, **fields):
        row = dict(kind=kind, checked_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
        with (a.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True), flush=True)
    (a.output / 'process.json').write_text(json.dumps({
        'pid': os.getpid(), 'argv': sys.argv, 'task_ids': TASKS,
        'source_freeze_sha256': sha(source / 'source-freeze-receipt.json'),
        'scope': 'completed detector byte freeze only', 'paper_eligible': False}) + '\n')
    from clearml import Task
    finished, last = set(), {}
    event('watch_started', pid=os.getpid(), overall_eta='unknown')
    while len(finished) < len(TASKS):
        for fold, task_id in enumerate(TASKS):
            if fold in finished:
                continue
            try:
                task = Task.get_task(task_id=task_id)
            except Exception as exc:
                # Error class only: SDK error bodies can contain authentication data.
                event('observation_error_nonterminal', fold=fold, task_id=task_id,
                      error_class=type(exc).__name__, eta='unknown')
                continue
            status = str(task.status)
            if last.get(fold) != status:
                event('training_state', fold=fold, task_id=task_id, status=status,
                      overall_eta='unknown')
                last[fold] = status
            if status in ('failed', 'stopped', 'closed', 'published'):
                event('requires_inspection_no_retry', fold=fold, task_id=task_id, status=status)
                finished.add(fold)
                continue
            if status != 'completed':
                continue
            if not eligible(task):
                event('completed_missing_both_side_artifacts', fold=fold, task_id=task_id)
                finished.add(fold)
                continue
            output = a.output / ('fold-%d-byte-freeze' % fold)
            if output.exists():
                event('partial_output_requires_inspection_no_retry', fold=fold, path=str(output))
                finished.add(fold)
                continue
            verify_sources()
            freezer = 'freeze_spd_official_oof_detector_bytes' + (
                '_offline_gl_v6.py' if fold >= 2 else '.py')
            package = ROOT / 'artifacts/spd-official-oof-fivefold-20260930' / (
                'fold-%d-package' % fold)
            cmd = [sys.executable, str(source / freezer), '--task-id', task_id,
                   '--package', str(package), '--output', str(output)]
            event('freezer_command', fold=fold, argv=cmd, eta='unknown')
            # Do not persist SDK stderr (it may include credentials). Existing
            # freezer writes full artifacts; persist only terminal return code.
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            receipt = output / 'acceptance-receipt.json'
            if result.returncode == 0 and receipt.is_file():
                accepted = json.loads(receipt.read_text())
                if (accepted.get('task_id') == task_id and accepted.get('fold_id') == fold
                        and accepted.get('seed') == 1337
                        and accepted.get('byte_freeze_accepted') is True):
                    event('detector_byte_freeze_accepted', fold=fold, task_id=task_id,
                          receipt=str(receipt), sha256=sha(receipt), tensor_forward_accepted=False)
                else:
                    event('freeze_receipt_mismatch_no_retry', fold=fold, task_id=task_id)
            else:
                event('freeze_failed_no_retry', fold=fold, task_id=task_id,
                      returncode=result.returncode, output=str(output))
            finished.add(fold)
        if a.once:
            break
        if len(finished) < len(TASKS):
            time.sleep(60)
    event('watch_exit', terminal_folds=sorted(finished), experiment_complete=False)


if __name__ == '__main__':
    main()
