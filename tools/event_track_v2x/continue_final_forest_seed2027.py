"""Complete the existing final-model prefetch, readback and CPU admission chain.

This controller never dispatches, uploads, restarts a reader or changes numeric
code. Each child runs at most once after its original dependency is qualified.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from rbf_nested_seen_val_v2_common import R, new, register, sha
from continue_final_topk_seed3407 import observe

ROOT = R / 'artifacts/rbf-final-forest-seed2027-after-prefetch-continuation-v1-20261005'
CACHE = R / 'artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/rbf/seed2027'
GATE = CACHE / 'independent-byte-coverage-factor-admission.json'
TASK = '4e9019afec67453ba5a83e8401949991'
PREFETCH = R / 'source-freezes/rbf-final-refit-registered-rank-byte-prefetch-v1-20261005/prefetch_final_refit_rank_artifacts.py'
PREFETCH_SHA = '1d7ddcb37b0761093fd05221dd043d66d58788eacad33fe2c498b4c6abd41e32'
STARTED = R / 'receipts/rbf-final-refit-rank-byte-prefetch-seed2027-20261004T174532644652Z-started.json'
STARTED_SHA = '0d355f96637f1fc0642544d5ddd2fd19ee02591763e4c6b47d7e2980a5386d38'
FINISHED = STARTED.with_name(STARTED.name.replace('-started.json', '-finished.json'))
READER = R / 'source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004/read_rbf_final_refit_forest_outputs.py'
READER_SHA = '53747489a446c8990cfc6d9310eb686a69b39d99bde226c8945665a974d83f61'
DRIVER = R / 'source-freezes/rbf-final-refit-full-forest-independent-CPU-v4-receipt-rows-20261004/accept_final_refit_cohort.py'
DRIVER_SHA = '06fc3b18417c54169a327e69e2b6d5fafdd800445696d786c0d67b152adb0b2f'
DRIVER_FREEZE_SHA = 'f36f644f5f18fad3e5ab489edbc949e93bcab4efce22749dcb90211c8c8b27f7'
PYTHON = '/Users/lbin/.local/share/recover-before-fuse/calibration-venv/bin/python'
READER_PYTHON = '/Users/lbin/.local/share/clearml/venv/bin/python'
JOURNAL = R / 'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json'


def ledger_binding(path, kind):
    ledger = json.loads((R / 'receipts/20260928-execution-ledger.json').read_bytes())
    assert any(e.get('kind') == kind and e.get('receipt') == str(path)
               and e.get('receipt_sha256') == sha(path) for e in ledger['entries']), 'unregistered or changed dependency'


def source_gate():
    assert sha(PREFETCH) == PREFETCH_SHA and sha(READER) == READER_SHA and sha(DRIVER) == DRIVER_SHA
    assert sha(DRIVER.parent / 'source-freeze.json') == DRIVER_FREEZE_SHA
    freeze = json.loads((DRIVER.parent / 'source-freeze.json').read_bytes())
    for name, digest in freeze['sources'].items():
        assert sha(DRIVER.parent / name) == digest
    for spec in freeze['unchanged_independent_sources']:
        assert sha(spec['path']) == spec['sha256']
    own = Path(__file__).resolve().parent
    manifest = json.loads((own / 'source-freeze.json').read_bytes())
    assert manifest['kind'] == 'rbf_final_forest_seed2027_after_prefetch_source_v1'
    for name, spec in manifest['sources'].items():
        assert sha(own / name) == spec['sha256']
    assert sha(READER.parent / 'rbf_nested_seen_val_v2_common.py') == sha(own / 'rbf_nested_seen_val_v2_common.py')
    return dict(controller_freeze_sha256=sha(own / 'source-freeze.json'),
                reader_sha256=READER_SHA, CPU_driver_sha256=DRIVER_SHA,
                CPU_driver_freeze_sha256=DRIVER_FREEZE_SHA, prefetch_sha256=PREFETCH_SHA)


def admitted_start(pid):
    assert sha(STARTED) == STARTED_SHA
    ledger_binding(STARTED, 'rbf_final_refit_registered_rank_byte_prefetch_v1_started')
    value = json.loads(STARTED.read_bytes())
    assert value['pid'] == pid and value['seed'] == 2027 and value['task_id'] == TASK
    assert value['source_path'] == str(PREFETCH) and value['source_sha256'] == PREFETCH_SHA
    assert value['cache_directory'] == str(CACHE) and value['parallel_downloads'] == 2
    status, identity = observe(pid)
    assert status == 'live', 'initial live prefetch required'
    fields = identity.split(maxsplit=5)
    assert len(fields) == 6
    argv = shlex.split(fields[5])
    assert argv == [READER_PYTHON, str(PREFETCH), '--seed', '2027', '--parallel-downloads', '2']
    return value, identity


def wait_original(pid, identity):
    began = time.monotonic()
    while True:
        status, current = observe(pid)
        if status == 'absent':
            return
        if status == 'live':
            assert current == identity, 'prefetch PID was reused; preserve original work'
        if time.monotonic() - began > 172800:
            raise TimeoutError('bounded wait expired; original prefetch left untouched')
        print(json.dumps(dict(phase='waiting_for_original_prefetch', pid=pid,
                              observation=status, ETA='unknown')), flush=True)
        time.sleep(30)


def qualify_prefetch(start):
    assert FINISHED.is_file(), 'original prefetch ended without its final receipt'
    value = json.loads(FINISHED.read_bytes())
    ledger_binding(FINISHED, 'rbf_final_refit_registered_rank_byte_prefetch_v1_finished')
    assert all(value[k] == v for k, v in start.items()), 'prefetch dependency identity changed'
    assert value['all_snapshot_bytes_match'] is value['registered_snapshot_unchanged_at_end'] is True
    assert value['failures'] == {} and value['independently_read_artifacts'] == start['registered_snapshot']
    assert value['extraction_started'] is value['task_mutated'] is value['uploads_performed'] is False
    return value


def qualify_completed_task(prefetch):
    from clearml import Task
    jobs = [j for j in json.loads(JOURNAL.read_bytes())['jobs'] if j['task_id'] == TASK]
    assert len(jobs) == 1
    job = jobs[0]
    plan = job['plan']
    assert job['seed'] == plan['seed'] == 2027 and plan['method'] == 'rbf'
    assert job['recipe_sha256'] == prefetch['recipe_sha256']
    canonical = json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == job['recipe_sha256']
    task = Task.get_task(task_id=TASK)
    assert str(task.status) == 'completed', 'no terminal readback on an active or failed experiment'
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
    expected = {'receipt', 'exclusive-source-manifest', *(f'replay-rank{i}' for i in range(plan['world_size']))}
    assert set(task.artifacts) == expected
    for key, spec in prefetch['registered_snapshot'].items():
        assert task.artifacts[key].hash == spec['sha256'] and task.artifacts[key].size == spec['bytes']
    return job


def reject_competing_processes():
    p = subprocess.run(['ps', '-ww', '-axo', 'command='], capture_output=True, text=True, timeout=10)
    assert p.returncode == 0 and not p.stderr.strip(), 'cannot observe duplicate processes'
    for line in p.stdout.splitlines():
        if READER.name in line or DRIVER.name in line:
            argv = shlex.split(line)
            if '--byte-admission' in argv:
                i = argv.index('--byte-admission')
                assert i + 1 < len(argv) and Path(argv[i + 1]).resolve() != GATE.resolve(), 'existing CPU verifier'
            if str(READER) in argv and '--seed' in argv and '--method' in argv:
                assert (argv[argv.index('--seed') + 1], argv[argv.index('--method') + 1]) != ('2027', 'rbf'), 'existing terminal reader'
def reject_prior_CPU():
    for directory in (R / 'artifacts').glob('rbf-final-*'):
        candidates = [directory / 'binding.json', directory / 'independent-acceptance/binding.json',
                      directory / 'seed2027/independent-acceptance/binding.json']
        for binding in candidates:
            if not binding.exists():
                continue
            # Only tiny binding documents are inspected, never the large SQL/arrays.
            value = json.loads(binding.read_bytes())
            assert value.get('task_id') != TASK, 'prior full CPU attempt preserved'


def reject_existing():
    assert not GATE.exists(), 'existing final byte admission must be inspected, not repeated'
    assert not (CACHE / 'independent-failure-byte-readback.json').exists(), 'existing failed readback preserved'
    reject_competing_processes()
    reject_prior_CPU()


def run_child(name, command):
    new(ROOT / (name + '-launch.json'), dict(command=command, ETA='unknown', experiment_accepted=False))
    with (ROOT / (name + '.log')).open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        new(ROOT / (name + '-process.json'), dict(pid=child.pid, command=command))
        print(json.dumps(dict(phase=name + '_started', pid=child.pid, ETA='unknown')), flush=True)
        code = child.wait()
    new(ROOT / (name + '-exit.json'), dict(returncode=code, experiment_accepted=False))
    assert code == 0, name + ' failed; no automatic retry'


def qualify_bytes(job):
    value = json.loads(GATE.read_bytes())
    ledger_binding(GATE, 'rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1')
    assert value['source_sha256'] == READER_SHA and value['seed'] == 2027 and value['task_id'] == TASK
    sys.path.insert(0, str(DRIVER.parent))
    spec = importlib.util.spec_from_file_location('frozen_final_forest_binding_for_continuation', DRIVER.parent / 'rbf_final_refit_forest_binding.py')
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    assert binding.validate(value) == job
    index = json.loads(binding.INDEX.read_bytes())
    entry = next(v for v in index['seeds'] if v['seed'] == 2027)
    checkpoint = Path(entry['training_byte_proof']).parent / 'checkpoint'
    assert sha(checkpoint) == job['plan']['checkpoint']['sha256']
    return checkpoint


def run(pid):
    control = source_gate()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (ROOT / 'started.json').exists(), 'controller attempt exists; no retry'
        reject_existing()
        start, identity = admitted_start(pid)
        launched = dict(kind='rbf_final_forest_seed2027_after_prefetch_started_v1',
                        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        controller_pid=os.getpid(), prefetch_pid=pid, prefetch_identity=identity,
                        original_prefetch_started=str(STARTED), original_prefetch_started_sha256=STARTED_SHA,
                        source_binding=control, task_id=TASK, seed=2027, ETA='unknown', experiment_accepted=False)
        new(ROOT / 'started.json', launched)
        register(ROOT / 'started.json', launched['kind'])
        print(json.dumps(launched), flush=True)
        try:
            wait_original(pid, identity)
            assert source_gate() == control
            prefetch = qualify_prefetch(start)
            job = qualify_completed_task(prefetch)
            reject_existing()
            # The original prefetch uses this same lock. Hold it throughout the
            # terminal reader so no new staging helper can touch the same cache.
            with (CACHE / 'rank-byte-prefetch.lock').open('a') as staged:
                fcntl.flock(staged, fcntl.LOCK_EX | fcntl.LOCK_NB)
                new(ROOT / 'prefetch-qualification.json', dict(receipt=str(FINISHED), sha256=sha(FINISHED),
                    task_id=TASK, recipe_sha256=job['recipe_sha256'], source_binding=control))
                run_child('terminal-readback', [READER_PYTHON, str(READER), '--seed', '2027', '--method', 'rbf'])
            checkpoint = qualify_bytes(job)
            assert source_gate() == control
            reject_competing_processes()
            reject_prior_CPU()
            output = ROOT / 'independent-acceptance'
            assert not output.exists()
            new(ROOT / 'byte-qualification.json', dict(receipt=str(GATE), sha256=sha(GATE),
                checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), task_id=TASK))
            run_child('full-CPU-admission', [PYTHON, str(DRIVER), '--byte-admission', str(GATE),
                '--output', str(output), '--checkpoint', str(checkpoint)])
            result = dict(kind='rbf_final_forest_seed2027_continuation_exit_v1',
                          acceptance_exists=(output / 'acceptance.json').is_file(),
                          failure_exists=(output / 'failure.json').is_file(),
                          requires_independent_receipt_review=True, experiment_accepted=False)
            new(ROOT / 'process-exit.json', result)
            register(ROOT / 'process-exit.json', result['kind'])
        except BaseException as error:
            failure = dict(kind='rbf_final_forest_seed2027_continuation_failure_v1',
                           error_type=type(error).__name__, original_prefetch_left_untouched=True,
                           automatic_retry=False, experiment_accepted=False)
            new(ROOT / 'controller-failure.json', failure)
            register(ROOT / 'controller-failure.json', failure['kind'])
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefetch-pid', type=int, required=True)
    args = parser.parse_args()
    run(args.prefetch_pid)
