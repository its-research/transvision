"""Start each new-weight independent reference once, after its byte gate."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

R = Path('/Volumes/Data/test/recover-before-fuse')
PY = '/Users/lbin/.local/share/recover-before-fuse/calibration-venv/bin/python'
ROOT = R / 'artifacts/rbf-final-refit-all-row-independent-numeric-v1-20261004'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def register(path):
    ledger = R / 'receipts/20260928-execution-ledger.json'
    with open(str(ledger) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(ledger.read_bytes())
        if not any(x.get('receipt') == str(path) for x in value['entries']):
            value['entries'].append(dict(kind='rbf_new_final_refit_independent_numeric_once_launch_v1',
                receipt=str(path), receipt_sha256=sha(path), goal_status='active',
                checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            tmp = ledger.with_suffix(ledger.suffix + '.tmp')
            tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
            os.replace(tmp, ledger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch-seconds', type=int, default=3600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'numeric-launch-source-freeze.json').read_bytes())
    assert sha(__file__) == config['launcher_sha256']
    assert sha(root / 'accept.py') == config['acceptance_sha256']
    started, children, previous = time.monotonic(), {}, {}
    while True:
        terminal = 0
        for seed in (1337, 2027, 3407):
            directory = ROOT / f'seed{seed}'
            if not (directory / 'independent-byte-coverage-receipt.json').exists():
                continue
            if seed not in children:
                log = directory / 'independent-numeric.log'
                if (directory / 'numeric-v1').exists() or log.exists():
                    children[seed] = None
                    print(json.dumps(dict(seed=seed, prior_attempt_preserved_no_restart=True)), flush=True)
                else:
                    command = [PY, '-B', str(root / 'accept.py'), '--stage', 'numeric', '--seed', str(seed)]
                    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    with log.open('x') as f:
                        child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=env,
                                                 start_new_session=True)
                    children[seed] = child
                    receipt = R / f'receipts/rbf-final-refit-seed{seed}-independent-numeric-launch-20261004.json'
                    new(receipt, dict(seed=seed, pid=child.pid, command=command, log=str(log),
                        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        launcher_sha256=sha(__file__), acceptance_sha256=sha(root / 'accept.py'),
                        byte_admission_sha256=sha(directory / 'independent-byte-coverage-receipt.json'),
                        atol=1e-4, rtol=1e-4, ETA='unknown_until_actual_row_progress',
                        numeric_acceptance_pass=False, whole_forest_replay_complete=False))
                    register(receipt)
                    print(json.dumps(dict(seed=seed, independent_numeric_pid=child.pid)), flush=True)
            child = children[seed]
            if child is None or child.poll() is not None:
                terminal += 1
                code = None if child is None else child.returncode
                if previous.get(seed) != (code, 'terminal'):
                    print(json.dumps(dict(seed=seed, numeric_process_terminal=True, exit_code=code,
                                         acceptance_requires_completion_receipt=True)), flush=True)
                    previous[seed] = (code, 'terminal')
        if terminal == 3 or time.monotonic() - started >= args.watch_seconds:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
