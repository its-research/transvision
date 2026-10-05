"""Join independently recovered rank bytes to the unchanged main admission chain.

Exactly one local continuation: retain original failed bytes and attempts, copy
only the two admitted diagnostic archives, then use the original terminal reader
and complete CPU driver. No uploads, task creation, experiment retries or changes
to frozen numerical code. Observation timeouts never mean the reader has stopped.
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
import shutil
import subprocess
import sys
import time

from rbf_nested_seen_val_v2_common import R, new, register, sha

NAME = 'rbf-final-forest-seed2027-after-isolated-diagnostic-v1-20261005'
ROOT = R/'artifacts'/NAME
OLD = R/'source-freezes/rbf-final-forest-seed2027-after-prefetch-continuation-v1-20261005'
OLD_SHA = 'e22815b20635446fce0d334622bbe137b42a386d20048c7ced075e3563b1462f'
DIAG_NAME = 'rbf-final-forest-seed2027-rank5-rank7-isolated-byte-diagnostic-v1-20261005'
DIAG_SOURCE = R/'source-freezes'/DIAG_NAME
DIAG_SHA = '91a4f83c241d542460427ae004566ef41824bf898e12f6927315bf33b9b8d890'
DIAG = R/'artifacts'/DIAG_NAME
TASK = '4e9019afec67453ba5a83e8401949991'
BAD_SHA = 'd030ea8dfa5f5f52dbe45d9ff8c8c195ae253747d57892a5165b049eb48f937f'
CACHE = R/'artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/rbf/seed2027'
GATE = CACHE/'independent-byte-coverage-factor-admission.json'


def base_module():
    assert sha(OLD/'source-freeze.json') == OLD_SHA
    manifest = json.loads((OLD/'source-freeze.json').read_bytes())
    for name, item in manifest['sources'].items(): assert sha(OLD/name) == item['sha256']
    sys.path.insert(0, str(OLD))
    name = 'unchanged_main_seed2027_continuation_contract'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, OLD/'continue_final_forest_seed2027.py')
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    module = sys.modules[name]; assert Path(module.__file__).parent == OLD
    assert Path(sys.modules['continue_final_topk_seed3407'].__file__).parent == OLD
    return module


def source_gate():
    base = base_module(); old_binding = base.source_gate()
    assert sha(DIAG_SOURCE/'source-freeze.json') == DIAG_SHA
    diagnostic = json.loads((DIAG_SOURCE/'source-freeze.json').read_bytes())
    for name, digest in diagnostic['sources'].items(): assert sha(DIAG_SOURCE/name) == digest
    for item in diagnostic['references']: assert sha(item['path']) == item['sha256']
    own = Path(__file__).resolve().parent
    freeze = json.loads((own/'source-freeze.json').read_bytes())
    assert freeze['kind'] == NAME
    for name, item in freeze['sources'].items(): assert sha(own/name) == item['sha256']
    for item in freeze['references']: assert sha(item['path']) == item['sha256']
    return dict(controller_source_freeze_sha256=sha(own/'source-freeze.json'),
                unchanged_original_chain=old_binding, diagnostic_source_freeze_sha256=DIAG_SHA)


def registered(path, kind=None):
    assert path.resolve().is_relative_to(R) and not any(p.is_symlink() for p in (path, *path.parents))
    value = json.loads(path.read_bytes())
    base_module().ledger_binding(path, kind or value['kind'])
    return value


def bind_diagnostic(pid):
    start = registered(DIAG/'started.json', DIAG_NAME+'_started')
    assert start['kind'] == DIAG_NAME and start['pid'] == pid
    assert start['task_id'] == TASK and start['seed'] == 2027 and start['source_freeze_sha256'] == DIAG_SHA
    script = DIAG_SOURCE/'read_final_forest_rank5_and_rank7_diagnostic.py'
    assert start['command'] == [str(script), '--execute']
    for key, kind in (('failure_receipt', 'rbf_final_refit_registered_rank_byte_prefetch_v1_finished'),
                      ('controller_failure', 'rbf_final_forest_seed2027_continuation_failure_v1')):
        item = start[key]
        assert sha(item['path']) == item['sha256']; registered(Path(item['path']), kind)
    assert start['original_partial'] == dict(path=str(CACHE/'replay-rank5.tar.gz.partial'), bytes=1918350389, sha256=BAD_SHA)
    assert start['automatic_retry'] is start['uploads_performed'] is start['extraction_started'] is start['experiment_accepted'] is False
    assert start['experiment_tasks_created'] == 0
    status, identity = base_module().observe(pid)
    if status == 'live':
        fields = identity.split(maxsplit=5); assert len(fields) == 6
        assert shlex.split(fields[5]) == [base_module().READER_PYTHON, str(script), '--execute']
    else:
        assert status == 'absent' and (DIAG/'finished.json').is_file(), 'no authoritative terminal diagnostic evidence'
    return start, status, identity


def wait_diagnostic(pid, identity, observe=None, sleep=time.sleep, clock=time.monotonic):
    observe = observe or base_module().observe; started = clock()
    while True:
        status, current = observe(pid)
        if status == 'absent': return
        if status == 'live': assert current == identity, 'diagnostic PID reused; no downstream action'
        assert status in ('unknown', 'live')
        if clock()-started > 172800: raise TimeoutError('bounded observation expired; original reader left untouched')
        print(json.dumps(dict(stage='waiting_for_isolated_rank_diagnostic', pid=pid, observation=status, ETA='unknown')), flush=True)
        sleep(30)


def validate_finished(start, finished, failed):
    assert all(finished[k] == v for k, v in start.items()), 'diagnostic input identity changed'
    assert finished['failures'] == {} and finished['all_selected_bytes_match'] is True
    assert finished['original_partial_preserved'] is True
    for flag in ('canonical_cache_promoted', 'full_byte_event_factor_admission', 'full_forest_CPU_admission', 'paper_performance_complete'):
        assert finished[flag] is False
    assert set(start['selected']) == {'replay-rank5', 'replay-rank7'}
    assert finished['independently_read_artifacts'] == start['selected']
    assert set(start['all_terminal_registered_artifacts']) == {'receipt','exclusive-source-manifest',*(f'replay-rank{i}' for i in range(8))}
    for key, item in start['selected'].items(): assert item == start['all_terminal_registered_artifacts'][key]
    assert failed['task_id'] == TASK and failed['seed'] == 2027 and failed['recipe_sha256'] == start['recipe_sha256']
    assert failed['failures'] == {'replay-rank5':'AssertionError'}
    assert failed['all_snapshot_bytes_match'] is False and failed['registered_snapshot_unchanged_at_end'] is True
    assert failed['independently_read_artifacts'] == start['independently_read_prior_archives_preserved']
    assert set(failed['independently_read_artifacts']) == {f'replay-rank{i}' for i in (0,1,2,3,4,6)}
    for key,item in failed['registered_snapshot'].items(): assert item == start['all_terminal_registered_artifacts'][key]
    assert start['original_partial']['sha256'] != start['selected']['replay-rank5']['sha256']


def qualify_finished(start):
    finished = registered(DIAG/'finished.json', DIAG_NAME+'_finished')
    failed = registered(Path(start['failure_receipt']['path']), 'rbf_final_refit_registered_rank_byte_prefetch_v1_finished')
    validate_finished(start, finished, failed)
    for item in (start['failure_receipt'], start['controller_failure']): assert sha(item['path']) == item['sha256']
    partial = Path(start['original_partial']['path'])
    assert partial.stat().st_size == start['original_partial']['bytes'] and sha(partial) == BAD_SHA
    for key,item in start['selected'].items():
        path = DIAG/(key+'.tar.gz')
        assert not path.is_symlink() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    # Original terminal reader rehashes all six prior archives itself. Do not
    # download, replace or otherwise mutate them in this recovery step.
    for key,item in failed['independently_read_artifacts'].items():
        path = CACHE/(key+'.tar.gz')
        assert not path.is_symlink() and path.stat().st_size == item['bytes']
    return finished


def qualify_live(start, Task=None):
    if Task is None:
        from clearml import Task
    base = base_module()
    jobs = [j for j in json.loads(base.JOURNAL.read_bytes())['jobs'] if j['task_id'] == TASK]; assert len(jobs) == 1
    job = jobs[0]; plan = job['plan']
    assert job['seed'] == plan['seed'] == 2027 and plan['method'] == 'rbf' and plan['world_size'] == 8
    assert hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == job['recipe_sha256'] == start['recipe_sha256']
    task = Task.get_task(task_id=TASK)
    assert str(task.status) == 'completed'
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
    params = task.get_parameters()
    assert json.loads(params['General/plan']) == plan and params['General/recipe_sha256'] == job['recipe_sha256']
    assert {key:dict(sha256=a.hash,bytes=a.size) for key,a in task.artifacts.items()} == start['all_terminal_registered_artifacts']
    return job


def copy_verified(source, destination, item):
    """Exclusive new file: any partial copy remains evidence and is never retried."""
    assert not source.is_symlink() and source.stat().st_size == item['bytes'] and sha(source) == item['sha256']
    assert not destination.exists() and not any(p.is_symlink() for p in (destination, *destination.parents))
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8*1024**2); dst.flush(); os.fsync(dst.fileno())
    assert destination.stat().st_size == item['bytes'] and sha(destination) == item['sha256']
    assert source.stat().st_size == item['bytes'] and sha(source) == item['sha256']
    return dict(source=str(source), destination=str(destination), **item, original_diagnostic_bytes_preserved=True)


def run_child(name, command):
    new(ROOT/(name+'-launch.json'),dict(command=command,ETA='unknown',automatic_retry=False,experiment_accepted=False))
    with (ROOT/(name+'.log')).open('x') as log:
        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        new(ROOT/(name+'-process.json'),dict(pid=child.pid,command=command))
        print(json.dumps(dict(stage=name+'_started',pid=child.pid,ETA='unknown')),flush=True)
        code = child.wait()
    new(ROOT/(name+'-exit.json'),dict(returncode=code,automatic_retry=False,experiment_accepted=False))
    assert code == 0, name+' failed; original and partial evidence retained'


def run(pid, execute=False):
    control = source_gate(); base = base_module()
    base.reject_existing()
    assert not ROOT.exists(), 'prior recovery attempt must be inspected, never replaced'
    start, status, identity = bind_diagnostic(pid)
    if not execute:
        print(json.dumps(dict(execute=False,diagnostic_pid=pid,observation=status,
            pending_dependency=str(DIAG/'finished.json'),source_binding=control,local_copy_or_child_started=False)),flush=True)
        return
    ROOT.mkdir()
    with (ROOT/'controller.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        launched=dict(kind=NAME+'_started',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pid=os.getpid(),command=sys.argv,diagnostic_pid=pid,diagnostic_identity=identity,diagnostic_start_sha256=sha(DIAG/'started.json'),
            task_id=TASK,seed=2027,source_binding=control,automatic_retry=False,experiment_accepted=False,ETA='unknown')
        new(ROOT/'started.json',launched);register(ROOT/'started.json',launched['kind'])
        print(json.dumps(launched),flush=True)
        try:
            if status == 'live': wait_diagnostic(pid,identity)
            assert base.observe(pid)[0] == 'absent'
            assert sha(DIAG/'started.json') == launched['diagnostic_start_sha256'] and source_gate() == control
            base.reject_existing()
            with (CACHE/'rank-byte-prefetch.lock').open('a') as staged:
                fcntl.flock(staged,fcntl.LOCK_EX|fcntl.LOCK_NB)
                finished = qualify_finished(start); job = qualify_live(start)
                base.reject_existing()
                assert all(not (CACHE/(key+'.tar.gz')).exists() for key in start['selected'])
                new(ROOT/'diagnostic-qualification.json',dict(finished_path=str(DIAG/'finished.json'),
                    finished_sha256=sha(DIAG/'finished.json'),task_id=TASK,recipe_sha256=job['recipe_sha256'],
                    preserved_failures=[start['failure_receipt'],start['controller_failure']],source_binding=control))
                promoted=[]
                for key,item in start['selected'].items():
                    new(ROOT/(key+'-copy-intent.json'),dict(artifact=key,source=str(DIAG/(key+'.tar.gz')),destination=str(CACHE/(key+'.tar.gz')),**item))
                    result=copy_verified(DIAG/(key+'.tar.gz'),CACHE/(key+'.tar.gz'),item)
                    new(ROOT/(key+'-copy-receipt.json'),result);promoted.append(result)
                assert sha(start['original_partial']['path']) == BAD_SHA and qualify_live(start) == job
                promotion=dict(kind=NAME+'_recovered_bytes_bound',copies=promoted,diagnostic_finished_sha256=sha(DIAG/'finished.json'),
                    original_partial_sha256=BAD_SHA,original_failed_attempts_preserved=True,
                    full_byte_event_factor_admission=False,full_forest_CPU_admission=False)
                new(ROOT/'recovered-byte-binding.json',promotion);register(ROOT/'recovered-byte-binding.json',promotion['kind'])
                assert source_gate() == control
                run_child('terminal-readback',[base.READER_PYTHON,str(base.READER),'--seed','2027','--method','rbf'])
            checkpoint=base.qualify_bytes(job)
            assert qualify_live(start) == job and source_gate() == control
            base.reject_competing_processes();base.reject_prior_CPU()
            output=ROOT/'independent-acceptance';assert not output.exists()
            new(ROOT/'byte-qualification.json',dict(path=str(GATE),sha256=sha(GATE),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
            run_child('full-CPU-admission',[base.PYTHON,str(base.DRIVER),'--byte-admission',str(GATE),'--output',str(output),'--checkpoint',str(checkpoint)])
            result=dict(kind=NAME+'_exit',acceptance_exists=(output/'acceptance.json').is_file(),
                failure_exists=(output/'failure.json').is_file(),requires_independent_receipt_review=True,experiment_accepted=False)
            new(ROOT/'process-exit.json',result);register(ROOT/'process-exit.json',result['kind'])
        except BaseException as error:
            failure=dict(kind=NAME+'_failure',error_type=type(error).__name__,
                original_diagnostic_left_untouched=True,original_failed_attempts_preserved=True,
                automatic_retry=False,experiment_accepted=False)
            new(ROOT/'controller-failure.json',failure);register(ROOT/'controller-failure.json',failure['kind']);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostic-pid',type=int,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args();run(args.diagnostic_pid,args.execute)
