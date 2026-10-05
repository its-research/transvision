"""Join the existing seed1337 reader to the unchanged Top1 CPU verifier once.

No downloads, uploads, dispatch, retries or verifier changes. Observation errors
leave the original reader untouched; only an observed exit allows qualification.
"""
import argparse
import datetime
import fcntl
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from rbf_nested_seen_val_v2_common import R, new, register, sha

DRIVER = R / 'source-freezes/rbf-final-refit-fixed-Top1-full-independent-CPU-v2-receipt-rows-20261004/accept_final_refit_top1_cohort.py'
DRIVER_SHA = '05c817c0578d97d5152749102cc3fcf0fd332b4a05397017e7e0a2aa506b12b1'
FREEZE_SHA = '6ebb95230c82d5564000e911f3c4c45939783954f5937b27e4d553d03a171e35'
READER = R / 'source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004/read_rbf_final_refit_top1_outputs.py'
READER_SHA = 'd8a9542fddc0f661c723d3f8661dbc317d106d7c34f0b66c11d64c7d65a672e7'
PYTHON = '/Users/lbin/.local/share/recover-before-fuse/calibration-venv/bin/python'
ROOT = R / 'artifacts/rbf-final-top1-seed1337-after-readback-continuation-v1-20261005'
GATE = R / 'artifacts/rbf-final-refit-full-train-fixed-Top1-independent-byte-factor-v1-20261004/seed1337/independent-byte-coverage-factor-admission.json'
TASK = '237f30c74248439c91f07a7e868ca383'


def source_gate():
    assert sha(DRIVER) == DRIVER_SHA and sha(READER) == READER_SHA
    freeze = DRIVER.parent / 'source-freeze.json'
    assert sha(freeze) == FREEZE_SHA
    control = json.loads(freeze.read_bytes())
    for name, item in control['sources'].items():
        assert sha(DRIVER.parent / name) == item['sha256']
    for item in control['unchanged_references']:
        assert sha(item['path']) == item['sha256']
    reader_freeze = READER.parent / 'source-freeze.json'
    assert sha(reader_freeze) == '8f5c23555a4b151f6d434ecbdd25a3c6d1986bae0807966aed531e246e400222'
    for name, item in json.loads(reader_freeze.read_bytes())['sources'].items():
        assert sha(READER.parent / name) == item['sha256']
    for item in json.loads(reader_freeze.read_bytes())['references']:
        assert sha(item['path']) == item['sha256']
    own = Path(__file__).resolve().parent
    manifest = json.loads((own / 'source-freeze.json').read_bytes())
    assert manifest['kind'] == 'rbf_final_top1_seed1337_after_readback_source_v1'
    for name, item in manifest['sources'].items():
        assert sha(own / name) == item['sha256']
    return dict(driver_sha256=DRIVER_SHA, driver_freeze_sha256=FREEZE_SHA,
                reader_sha256=READER_SHA, controller_freeze_sha256=sha(own / 'source-freeze.json'))


def observe(pid):
    """None means confirmed absent; errors and timeouts are never absence."""
    try:
        p = subprocess.run(['ps', '-ww', '-p', str(pid), '-o', 'lstart=,command='],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return 'unknown', None
    if p.returncode == 1 and not p.stdout.strip() and not p.stderr.strip():
        return 'absent', None
    if p.returncode != 0 or p.stderr.strip() or not p.stdout.strip():
        return 'unknown', None
    return 'live', p.stdout.strip()


def bind_reader(identity):
    fields = identity.split(maxsplit=5)
    assert len(fields) == 6, 'missing process start identity'
    argv = shlex.split(fields[5])
    assert len(argv) == 5 and argv[1:] == ['-u', str(READER), '--seed', '1337'], 'reader PID belongs to another command'
    assert Path(argv[0]).name.startswith('python')
    return identity


def reject_duplicate_process():
    p = subprocess.run(['ps', '-ww', '-axo', 'command='], capture_output=True, text=True, timeout=10)
    assert p.returncode == 0 and not p.stderr.strip(), 'cannot verify CPU process uniqueness'
    for line in p.stdout.splitlines():
        if 'accept_final_refit_top1_cohort.py' not in line:
            continue
        argv = shlex.split(line)
        if '--byte-admission' in argv:
            index = argv.index('--byte-admission')
            assert index + 1 < len(argv)
            assert Path(argv[index + 1]).resolve() != GATE.resolve(), 'same seed already has a live CPU verifier'


def qualify():
    assert GATE.is_file() and not GATE.is_symlink(), 'reader exited without full byte/factor admission'
    value = json.loads(GATE.read_bytes())
    assert value['seed'] == 1337 and value['method'] == 'topk' and value['task_id'] == TASK
    assert value['source_sha256'] == READER_SHA
    ledger = json.loads((R / 'receipts/20260928-execution-ledger.json').read_bytes())
    assert any(e.get('receipt') == str(GATE) and e.get('receipt_sha256') == sha(GATE)
               and e.get('kind') == value['kind'] for e in ledger['entries']), 'unregistered or changed byte admission'
    # This original binding checks complete coverage, recipe/seed/model, source,
    # fixed tolerances, completed live task and every registered artifact.
    sys.path.insert(0, str(DRIVER.parent))
    spec = importlib.util.spec_from_file_location('frozen_Top1_binding_for_continuation', DRIVER.parent / 'rbf_final_refit_top1_binding.py')
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    job, _, model, _ = binding.validate_output(value)
    return dict(task_id=job['task_id'], seed=1337, byte_admission=str(GATE),
                byte_admission_sha256=sha(GATE), recipe_sha256=job['recipe_sha256'],
                final_model_binding=model)


def reject_prior_evidence():
    artifacts = R / 'artifacts'
    patterns = ('rbf-final-refit-Top1-seed1337*/binding.json',
                'rbf-final-top1-after-readback-continuation-*/seed1337/launch.json',
                'rbf-final-top1-after-readback-continuation-*/seed1337/independent-acceptance/binding.json')
    assert not any(list(artifacts.glob(pattern)) for pattern in patterns), 'prior seed1337 CPU attempt preserved'
    ledger = json.loads((R / 'receipts/20260928-execution-ledger.json').read_bytes())
    for entry in ledger['entries']:
        if entry.get('kind') not in ('rbf_final_refit_fixed_Top1_full_causal_cache203_fresh_state_admission_v1',
                                     'rbf_final_refit_fixed_Top1_CPU_admission_failure_v1'):
            continue
        value = json.loads(Path(entry['receipt']).read_bytes())
        assert value['task_id'] != TASK, 'prior registered CPU outcome preserved'


def run(reader_pid):
    control = source_gate()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (ROOT / 'started.json').exists(), 'existing controller attempt preserved; no automatic retry'
        assert not (ROOT / 'independent-acceptance').exists()
        reject_prior_evidence()
        reject_duplicate_process()
        status, identity = observe(reader_pid)
        assert status == 'live', 'initial live reader observation required'
        bind_reader(identity)
        started = dict(kind='rbf_final_top1_seed1337_CPU_continuation_started_v1',
                       reader_pid=reader_pid, reader_process_identity=identity,
                       controller_pid=__import__('os').getpid(), source_binding=control,
                       task_id=TASK, seed=1337, ETA='unknown', experiment_accepted=False,
                       checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        new(ROOT / 'started.json', started)
        register(ROOT / 'started.json', started['kind'])
        print(json.dumps(dict(phase='waiting_for_existing_reader', **started)), flush=True)
        try:
            begin = time.monotonic()
            while True:
                status, current = observe(reader_pid)
                if status == 'absent':
                    break
                if status == 'live':
                    assert current == identity, 'reader process identity changed; no automatic continuation'
                if time.monotonic() - begin > 172800:
                    raise TimeoutError('bounded observation expired; reader left untouched')
                print(json.dumps(dict(phase='waiting_for_existing_reader', observation=status,
                                      reader_pid=reader_pid, ETA='unknown')), flush=True)
                time.sleep(30)
            assert source_gate() == control
            qualified = qualify()
            reject_prior_evidence()
            reject_duplicate_process()
            output = ROOT / 'independent-acceptance'
            assert not output.exists()
            assert sha(GATE) == qualified['byte_admission_sha256']
            command = [PYTHON, str(DRIVER), '--byte-admission', str(GATE), '--output', str(output)]
            new(ROOT / 'launch.json', dict(qualified, command=command, source_binding=control,
                                          ETA='unknown', experiment_accepted=False))
            with (ROOT / 'acceptance.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                new(ROOT / 'child-process.json', dict(pid=child.pid, command=command))
                print(json.dumps(dict(phase='frozen_CPU_acceptance_started', pid=child.pid, seed=1337,
                                      ETA='unknown', output=str(output))), flush=True)
                code = child.wait()
            outcome = dict(kind='rbf_final_top1_seed1337_CPU_continuation_exit_v1', returncode=code,
                           acceptance_exists=(output / 'acceptance.json').is_file(),
                           failure_exists=(output / 'failure.json').is_file(),
                           requires_independent_receipt_review=True, experiment_accepted=False)
            new(ROOT / 'process-exit.json', outcome)
            register(ROOT / 'process-exit.json', outcome['kind'])
            if code:
                raise SystemExit(code)
        except BaseException as error:
            failure = dict(kind='rbf_final_top1_seed1337_CPU_continuation_failure_v1',
                           error_type=type(error).__name__, reader_left_untouched=True,
                           automatic_retry=False, experiment_accepted=False)
            new(ROOT / 'controller-failure.json', failure)
            register(ROOT / 'controller-failure.json', failure['kind'])
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-pid', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    if options.execute:
        run(options.reader_pid)
    else:
        control = source_gate()
        reject_prior_evidence()
        reject_duplicate_process()
        status, identity = observe(options.reader_pid)
        assert status == 'live', 'initial live reader observation required'
        bind_reader(identity)
        print(json.dumps(dict(execute=False, task_id=TASK, seed=1337, reader_pid=options.reader_pid,
                              observation=status, source_binding=control, experiment_accepted=False)))
