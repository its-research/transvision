"""Continue existing frozen five-fold fits through independently accepted exports.

This process never creates training jobs, retries failures, or grants calibration
or paper eligibility. Existing terminal failures and partial outputs stop a fold.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
FREEZES = ROOT / 'artifacts/spd-oof-completed-byte-freeze-watch-20261001'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.output / 'process.json').exists():
        raise FileExistsError('monitor process identity already registered; inspect live handle')
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    inventory = json.loads((source / 'source-freeze-receipt.json').read_text())['inventory']
    def verify():
        for row in inventory:
            if sha(source / row['name']) != row['sha256']:
                raise ValueError('immutable continuation source differs')
    verify()
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    from submit_spd_oof_checkpoint_forward_gpu4 import safe_workers
    api = APIClient()
    stopped, finished, last = set(), set(), {}
    def event(kind, **fields):
        row = dict(kind=kind, checked_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
        with (args.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True), flush=True)
    def execute(cmd, fold, stage):
        verify()
        event('command_started', fold=fold, stage=stage, argv=cmd, eta='unknown')
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        existing = None
        with (args.output / ('fold-%d-%s-console.log' % (fold, stage))).open('a') as log:
            for line in proc.stdout:
                if line.startswith(('EXISTING_', 'FORWARD_ENQUEUED', 'RAW_EXPORT_ENQUEUED',
                                    'SAMPLED_FORWARD_', 'COMPLETE_RAW_', 'RAW_CLOUD_')):
                    log.write(line)
                    if line.startswith('EXISTING_'):
                        existing = line.split()[1]
        code = proc.wait()
        event('command_terminal', fold=fold, stage=stage, returncode=code)
        if code:
            stopped.add(fold)
            event('requires_inspection_no_retry', fold=fold, stage=stage)
        return code, existing
    def queue_for(freeze):
        # These queues have actually run the frozen runtime; no model whitelist.
        # The original training queue is preferred, then other admitted fit queues.
        ids = ['517e10afb294404a96fc6c19f5ce88f6','12b06f9674494dc381b4d6566d848b20',
               '6134c1c99e2141b3b919801f6874ff43','ca63b926f4c9412b84ab2f77484fc7fe',
               'a4ad4a451bd94a79a96cea8f38b083c3']
        order = [freeze['task_id']] + [i for i in ids if i != freeze['task_id']]
        queues = list(dict.fromkeys(Task.get_task(task_id=i).get_parameters()['General/queue'] for i in order))
        workers = [w.to_dict() for w in api.workers.get_all()]
        for name in queues:
            found = [q for q in api.queues.get_all(name=name) if q.name == name]
            if len(found) != 1 or found[0].entries:
                continue
            try:
                safe_workers(workers, found[0].id, datetime.now(timezone.utc))
            except ValueError:
                continue
            return name
        return None
    (args.output / 'process.json').write_text(json.dumps({
        'pid': os.getpid(), 'argv': sys.argv,
        'source_freeze_sha256': sha(source / 'source-freeze-receipt.json'),
        'scope': 'sampled forward only; raw exports await separately admitted transport revision',
        'paper_eligible': False, 'automatic_failure_retry': False}) + '\n')
    event('continuation_started', pid=os.getpid(), overall_eta='unknown')
    stages = [('forward', 'submit_spd_oof_checkpoint_forward_gpu4_mmcv14_iter_v2.py',
               'accept_spd_oof_checkpoint_forward_gpu4_mmcv14_iter_v2.py',
               'independent_bytes_and_sampled_tensor_forward_verified')]
    while len(stopped | finished) < 3:
        for fold in range(2, 5):
            if fold in stopped | finished:
                continue
            bf = FREEZES / ('fold-%d-byte-freeze/acceptance-receipt.json' % fold)
            if not bf.is_file():
                continue
            try:
                freeze = json.loads(bf.read_text())
                if freeze['fold_id'] != fold or freeze['seed'] != 1337 or not freeze['byte_freeze_accepted']:
                    raise ValueError('wrong original freeze identity')
                folder = args.output / ('fold-%d' % fold)
                folder.mkdir(exist_ok=True)
                package = ROOT / 'artifacts/spd-official-oof-fivefold-20260930' / ('fold-%d-package' % fold)
                held = ROOT / 'artifacts/spd-canonical-oof-heldout-inference-inputs-20260930' / ('fold-%d' % fold)
                fit = ROOT / 'artifacts/spd-canonical-oof-fit-feature-inputs-20261001' / ('fold-%d' % fold)
                forward_receipt = folder / 'forward-readback/acceptance-receipt.json'
                for stage, dispatch, acceptor, status in stages:
                    receipt = folder / (stage + '-readback/acceptance-receipt.json')
                    if receipt.exists():
                        accepted = json.loads(receipt.read_text())
                        if (accepted['status'] != status or accepted['training_task_id'] != freeze['task_id']
                                or accepted['byte_freeze_sha256'] != sha(bf) or accepted['fold_id'] != fold
                                or accepted['seed'] != 1337 or accepted['acceptor_sha256'] != sha(source / acceptor)):
                            raise ValueError('continuation acceptance identity differs')
                        continue
                    common = ['--byte-freeze', str(bf), '--package', str(package)]
                    common += ['--fit-inputs' if stage == 'fit-raw-v2' else '--heldout-inputs', str(fit if stage == 'fit-raw-v2' else held)]
                    if stage != 'forward':
                        common += ['--forward-acceptance', str(forward_receipt)]
                    dispatched = folder / (stage + '-dispatch.json')
                    if not dispatched.exists():
                        queue = queue_for(freeze)
                        if queue is None:
                            break
                        code, existing = execute([sys.executable, str(source / dispatch)] + common +
                                                 ['--queue', queue, '--output', str(dispatched)], fold, stage)
                        if code:
                            break
                        if not dispatched.exists():
                            if not existing:
                                raise ValueError('successful dispatcher lacks task identity')
                            with dispatched.open('x') as f:
                                json.dump({'task_id': existing, 'attached_existing': True}, f)
                    remote = Task.get_task(task_id=json.loads(dispatched.read_text())['task_id'])
                    key = (fold, stage)
                    if last.get(key) != str(remote.status):
                        event('remote_state', fold=fold, stage=stage, task_id=remote.id,
                              status=str(remote.status), eta='unknown')
                        last[key] = str(remote.status)
                    if remote.status in ('queued', 'in_progress'):
                        break
                    if remote.status != 'completed':
                        stopped.add(fold)
                        event('terminal_or_unqueued_requires_inspection_no_retry', fold=fold, stage=stage)
                        break
                    if receipt.parent.exists():
                        raise ValueError('partial readback requires inspection')
                    code, _ = execute([sys.executable, str(source / acceptor), '--task-id', remote.id] +
                                      common + ['--output', str(receipt.parent)], fold, stage)
                    if code:
                        break
                    # Revalidate a new receipt on the next polling cycle.
                    break
                else:
                    finished.add(fold)
                    event('fold_sampled_forward_accepted', fold=fold, complete_prediction_coverage_verified=False, paper_eligible=False)
            except ValueError as exc:
                stopped.add(fold)
                event('requires_inspection_no_retry', fold=fold, error_class=type(exc).__name__)
            except Exception as exc:
                # API observation failure is not task termination; keep same handles.
                event('observation_error_nonterminal', fold=fold, error_class=type(exc).__name__)
        if len(stopped | finished) < 3:
            time.sleep(60)
    event('continuation_exit', accepted_folds=sorted(finished), inspection_folds=sorted(stopped),
          experiment_complete=False, paper_eligible=False)


if __name__ == '__main__':
    main()
