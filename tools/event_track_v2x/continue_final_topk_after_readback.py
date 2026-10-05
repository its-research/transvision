"""Wait for an existing reader; run the frozen verifier once, preserving failures."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

R = Path('/Volumes/Data/test/recover-before-fuse')
DRIVER = R/'source-freezes/rbf-final-refit-fixed-topK-full-independent-CPU-v2-receipt-rows-20261004/accept_final_refit_topk_cohort.py'
DRIVER_SHA = '29f5303e04a2d262d0d65c23d470fa9830100184370003ddb738558cad843f97'
PYTHON = '/Users/lbin/.local/share/recover-before-fuse/calibration-venv/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(1337, 2027), required=True)
    parser.add_argument('--reader-pid', type=int, required=True)
    args = parser.parse_args()
    freeze = DRIVER.parent/'source-freeze.json'
    assert sha(freeze) == '7dbc44e80ac6b13c232a59004fb3baaaa67ebde9b6e95c4890ce57a74f2f47f4'
    contract = json.loads(freeze.read_bytes())
    for name, spec in contract['sources'].items():
        assert sha(DRIVER.parent/name) == spec['sha256']
    for spec in contract['unchanged_references']:
        assert sha(Path(spec['path'])) == spec['sha256']
    assert sha(DRIVER) == DRIVER_SHA
    root = R/'artifacts/rbf-final-topk-after-readback-continuation-v3-20261004'/f'seed{args.seed}'
    root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root/'launch.json').exists(), 'existing attempt preserved; no automatic retry'
        gate = R/f'artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/topk/seed{args.seed}/independent-byte-coverage-factor-admission.json'
        output = root/'independent-acceptance'
        assert not output.exists(), 'existing output preserved'
        started = time.monotonic()
        expected = f'read_rbf_final_refit_forest_outputs.py --seed {args.seed} --method topk'
        while True:
            probe = subprocess.run(['ps', '-p', str(args.reader_pid), '-o', 'command='], capture_output=True, text=True)
            assert probe.returncode in (0, 1), 'process observation failed; do not infer termination'
            if probe.returncode == 1:
                break
            assert expected in probe.stdout, 'PID identity changed; inspect manually'
            if time.monotonic()-started > 172800:
                raise TimeoutError('bounded wait expired; original reader left untouched')
            time.sleep(30)
        assert gate.is_file(), 'reader ended without full admission; preserve failure and inspect'
        assert sha(DRIVER) == DRIVER_SHA
        value = json.loads(gate.read_bytes())
        assert value['seed'] == args.seed
        command = [PYTHON, str(DRIVER), '--byte-admission', str(gate), '--output', str(output)]
        record = dict(seed=args.seed, reader_pid=args.reader_pid, command=command,
                      gate=str(gate), gate_sha256=sha(gate), driver_sha256=DRIVER_SHA,
                      controller_sha256=sha(Path(__file__)), ETA='unknown',
                      experiment_accepted=False)
        with (root/'launch.json').open('x') as stream:
            json.dump(record, stream, sort_keys=True); stream.write('\n')
        with (root/'acceptance.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps(dict(phase='frozen_CPU_acceptance_started', seed=args.seed, pid=process.pid,
                                  output=str(output), ETA='unknown')), flush=True)
            code = process.wait()
        with (root/'process-exit.json').open('x') as stream:
            json.dump(dict(returncode=code, acceptance_exists=(output/'acceptance.json').is_file(),
                           failure_exists=(output/'failure.json').is_file(),
                           requires_independent_receipt_review=True), stream)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
