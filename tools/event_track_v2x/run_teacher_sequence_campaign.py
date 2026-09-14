#!/usr/bin/env python3
"""Bounded continuation of the fixed 46-sequence CPU teacher, NOT training.

Existing live collectors count against two slots, identified by PID/start/argv.
Complete leaves are independently audited. No overwrite, automatic retry,
process termination, dataset upload, parameter training or partial full receipt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x import assemble_allocation_teachers as audit
from tools.event_track_v2x import collect_allocation_training as collector

CHECKPOINT_SHA = '70609d3a7bf330d0d56b92150d0f97db511dfe4545c07cc193e993d0b77a0e96'
POSE_SHA = 'de20f5dde0adeabcc2f2981c18ea08fc781ae166bf2f3a143d04520f8ce16793'
THREAD_ENV = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', BLIS_NUM_THREADS='1')
MIN_FREE = 4 * 1024**3


def ordinary(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('ordinary non-symlink path required')
    return path


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False); f.write('\n')


def process_identity(pid):
    """A failed observation is not a missing process or a successful exit."""
    if type(pid) is not int or pid <= 1:
        raise ValueError('specific positive process PID required')
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'command='],
                            capture_output=True, text=True, timeout=10, check=False)
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return None
    if result.returncode != 0 or result.stderr.strip():
        raise RuntimeError('process observation failed; no restart or exit inference permitted')
    parts = result.stdout.strip().split(None, 5)
    if len(parts) != 6:
        raise ValueError('unambiguous process start time and command required')
    return dict(pid=pid, start=' '.join(parts[:5]), command=parts[5])


def command_for(paths, sequence):
    return [paths['python'], 'tools/event_track_v2x/collect_allocation_training.py',
        '--cache', paths['cache'], '--cooperative-metadata', paths['metadata'],
        '--cooperative-metadata-sha256', audit.PAIR_SHA256,
        '--checkpoint', paths['checkpoint'], '--checkpoint-sha256', CHECKPOINT_SHA,
        '--beam-recovery', '--recovery-task-poses', paths['poses'],
        '--recovery-task-poses-sha256', POSE_SHA, '--active-limit', '4', '--expansion-budget', '256',
        '--development-sequence', sequence, '--output', str(Path(paths['root']) / leaf_name(sequence))]


def leaf_name(sequence):
    return f'beam-recovery-teacher-roi-seq{sequence}-a1001337-v1'


def matches_command(identity, expected):
    actual = shlex.split(identity['command'])
    if len(actual) < 2: return False
    if actual[1] == str(ROOT / expected[1]): actual[1] = expected[1]
    return actual == expected


def unchanged(plan):
    if any(audit.sha_file(p) != h for p, h in plan['input_sha256'].items()):
        raise ValueError('teacher campaign input, runtime or source changed')


def accept(directory, expected, pair_sha, common):
    directory = ordinary(directory)
    digest = audit.sha_file(directory / 'receipt.json')
    leaf = audit._leaf(directory, digest, expected, pair_sha, False)
    audit._unchanged([leaf])
    if audit._common(leaf['plan']) != common:
        raise ValueError('teacher leaf differs from fixed common configuration')
    # Do not retain every large event report in the scheduling process.
    return dict(leaf['input'], frames=len(leaf['report']['events']),
                artifact_sha256={str(p): h for p, h in leaf['artifacts'].items()},
                full_official_train_trace_completed=False)


@contextmanager
def campaign_lock(root):
    """OS-held lock, never a lock-file existence/liveness inference."""
    path = ordinary(root) / '.teacher-sequence-campaign.lock'
    flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)  # Only releases this controller's own lock; no deletion.


def prepare(root, checkpoint, poses, reference, reference_sha, external, output):
    root, checkpoint, poses, reference, output = map(ordinary, (root, checkpoint, poses, reference, output))
    if output.exists(): raise ValueError('new campaign output required; never overwrite or resume implicitly')
    if len(external) > 2 or len({s for s, _ in external}) != len(external) or len({p for _, p in external}) != len(external):
        raise ValueError('at most two distinct existing collectors may be adopted')
    if any(os.environ.get(k) != v for k, v in THREAD_ENV.items()):
        raise ValueError('fixed one-thread inference environment required')
    paths = dict(root=str(root), cache=str(root / 'cache'), metadata=str(root / 'projection/cooperative/data_info.json'),
                 checkpoint=str(checkpoint), poses=str(poses), python=sys.executable)
    expected, pair_sha = audit._pairs(paths['metadata'], False)
    if any(s not in expected for s, _ in external): raise ValueError('external collector must belong to official train')
    if audit.sha_file(reference / 'receipt.json') != reference_sha:
        raise ValueError('reference teacher receipt changed')
    reference_plan = json.loads((reference / 'plan.json').read_bytes())
    config = collector.BeamRecoveryConfig(state=collector.ForestTrackingConfig(active_limit=4),
        recovery_budget=256, recovery_allocation_scope=collector.SCOPE_MODE)
    if (reference_plan['configuration'] != asdict(config)
            or reference_plan['identity_checkpoint_sha256'] != CHECKPOINT_SHA
            or reference_plan['ego_pose_table_sha256'] != POSE_SHA
            or reference_plan['allocation_policy_signature'] is not None):
        raise ValueError('fixed learned beam-recovery teacher required')
    common = audit._common(reference_plan)
    accept(reference, expected, pair_sha, common)
    if audit.sha_file(checkpoint / 'checkpoint.json') != CHECKPOINT_SHA:
        raise ValueError('fixed identity checkpoint required')
    cp = json.loads((checkpoint / 'checkpoint.json').read_bytes())
    inputs = {str(root / 'cache/manifest.json'): audit.TRAIN_CACHE_SHA256,
        paths['metadata']: pair_sha, str(checkpoint / 'checkpoint.json'): CHECKPOINT_SHA,
        str(audit.contained_file(checkpoint, cp['weights']['path'])): cp['weights']['sha256'], str(poses): POSE_SHA,
        str(reference / 'receipt.json'): reference_sha, str(reference / 'plan.json'): audit.sha_file(reference / 'plan.json'),
        sys.executable: audit.sha_file(sys.executable), str(Path(__file__)): audit.sha_file(__file__),
        str(Path(audit.__file__)): audit.sha_file(audit.__file__)}
    inputs.update({str(ROOT / p): h for p, h in reference_plan['source_sha256'].items()})
    plan = dict(kind='fixed_train_teacher_sequence_campaign_v1', paths=paths, input_sha256=inputs,
        common=common, pair_sha256=pair_sha, sequence_frames={s: len(expected[s]) for s in sorted(expected)},
        max_parallel=2, minimum_free_bytes=MIN_FREE, thread_environment=THREAD_ENV,
        accepted={}, external={}, pending=[], parameter_training=False, raw_GT_read=False,
        upload_enabled=False, teacher_assembly_completed=False, paper_eligible=False)
    unchanged(plan); declared = dict(external)
    for sequence in sorted(expected):
        directory = root / leaf_name(sequence)
        identity = process_identity(declared[sequence]) if sequence in declared else None
        if identity is not None:
            if not directory.is_dir() or not matches_command(identity, command_for(paths, sequence)):
                raise ValueError('external process identity differs from declared collector')
            running_plan = json.loads((directory / 'plan.json').read_bytes())
            if audit._common(running_plan) != common:
                raise ValueError('existing live collector has different inputs or configuration')
            audit._sources(running_plan)
            plan['external'][sequence] = identity
        elif directory.exists():
            plan['accepted'][sequence] = accept(directory, expected, pair_sha, common)
            print(json.dumps(dict(event='teacher_existing_accepted', sequence=sequence)), flush=True)
        elif sequence in declared:
            raise ValueError('declared external collector missing without complete output; no retry')
        else:
            plan['pending'].append(sequence)
    unchanged(plan)
    output.mkdir(); write_json(output / 'campaign.json', plan)
    return plan, expected


class LiveCollectors:
    def __init__(self, plan, output):
        self.plan, self.output, self.owned, self.logs = plan, Path(output), {}, {}

    def poll(self, sequence, identity):
        if sequence in self.owned:
            return self.owned[sequence].poll()
        current = process_identity(identity['pid'])
        # PID reuse means the original process is gone, not permission to touch
        # the new process. A complete independent leaf audit is still required.
        return None if current == identity else 'external_exit_unobserved'

    def launch(self, sequence):
        unchanged(self.plan)
        if shutil.disk_usage(self.plan['paths']['root']).free < self.plan['minimum_free_bytes']:
            raise ValueError('private disk reserve exhausted; no launch or deletion')
        command = command_for(self.plan['paths'], sequence)
        if Path(command[-1]).exists(): raise ValueError('teacher destination appeared; no overwrite or retry')
        log = (self.output / f'collector-{sequence}.log').open('xb')
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            log.close(); raise
        self.owned[sequence], self.logs[sequence] = child, log
        return dict(pid=child.pid, owned=True)

    def close_log(self, sequence):
        if sequence in self.logs: self.logs.pop(sequence).close()


def drive(plan, launch, poll, accept_leaf, emit, wait):
    """Only this controller's new children are launched; never kill or retry.

    After an observed failure, drain already-running collectors and stop further
    launches. A process-observation error propagates, leaving processes intact.
    """
    accepted = dict(plan['accepted']); running = dict(plan['external'])
    pending = list(plan['pending']); failures = []
    if len(running) > plan['max_parallel']:
        raise ValueError('existing collectors exceed fixed concurrency')
    while running or pending:
        for sequence in list(running):
            status = poll(sequence, running[sequence])
            if status is None: continue
            del running[sequence]
            try:
                if status not in (0, 'external_exit_unobserved'):
                    raise ValueError('collector exited nonzero; automatic retry is disabled')
                accepted[sequence] = accept_leaf(sequence)
                emit(dict(event='teacher_accepted', sequence=sequence, exit_status=status, **accepted[sequence]))
            except Exception as error:
                failure = dict(sequence=sequence, error_type=type(error).__name__, error=str(error))
                failures.append(failure); emit(dict(event='teacher_failed', **failure))
        while pending and not failures and len(running) < plan['max_parallel']:
            sequence = pending[0]
            try:
                identity = launch(sequence)
            except Exception as error:
                failure = dict(sequence=sequence, error_type=type(error).__name__, error=str(error))
                failures.append(failure); emit(dict(event='teacher_launch_blocked', **failure)); break
            pending.pop(0); running[sequence] = identity
            emit(dict(event='teacher_started', sequence=sequence, process=identity))
        if failures and not running: break
        if running: wait()
    complete = (not failures and not pending and set(accepted) == set(plan['sequence_frames'])
                and all(v['frames'] == plan['sequence_frames'][s] for s, v in accepted.items()))
    return dict(status='complete' if complete else 'incomplete', accepted=accepted, pending=pending,
        failures=failures, completed_sequences=len(accepted), completed_frames=sum(v['frames'] for v in accepted.values()),
        all_sequence_teachers_audited=complete, full_teacher_assembly_completed=False,
        parameter_training=False, training_submitted=False, paper_eligible=False)


def execute(plan, expected, output):
    output = Path(output); live = LiveCollectors(plan, output)
    campaign_sha = audit.sha_file(output / 'campaign.json')
    if json.loads((output / 'campaign.json').read_bytes()) != plan:
        raise ValueError('saved campaign differs from the prepared launch plan')
    def emit(event):
        with (output / 'events.jsonl').open('a') as f:
            json.dump(event, f, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(event, sort_keys=True), flush=True)
    def accept_leaf(sequence):
        live.close_log(sequence)
        unchanged(plan)
        return accept(Path(plan['paths']['root']) / leaf_name(sequence), expected, plan['pair_sha256'], plan['common'])
    try:
        result = drive(plan, live.launch, live.poll, accept_leaf, emit, lambda: time.sleep(2))
        unchanged(plan)
        for accepted in result['accepted'].values():
            if any(audit.sha_file(p) != h for p, h in accepted['artifact_sha256'].items()):
                raise ValueError('previously accepted teacher changed before campaign receipt')
        if audit.sha_file(output / 'campaign.json') != campaign_sha:
            raise ValueError('frozen campaign file changed during execution')
        result.update(campaign_sha256=campaign_sha)
        write_json(output / 'receipt.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'completed_sequences', 'completed_frames', 'parameter_training')}), flush=True)
        return result
    except BaseException as error:
        emit(dict(event='controller_stopped', error_type=type(error).__name__,
                  active_owned_pids={s: p.pid for s, p in live.owned.items() if p.poll() is None},
                  existing_processes_not_terminated=True, automatic_retry=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('train-root', 'checkpoint', 'poses', 'reference', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--reference-sha256', required=True)
    p.add_argument('--external', nargs=2, action='append', default=[], metavar=('SEQUENCE', 'PID'))
    p.add_argument('--execute', action='store_true', help='Run bounded CPU collectors after the frozen plan is verified.')
    args = p.parse_args()
    with campaign_lock(args.train_root):
        plan, expected = prepare(args.train_root, args.checkpoint, args.poses, args.reference,
            args.reference_sha256, [(s, int(pid)) for s, pid in args.external], args.output)
        print(json.dumps(dict(event='teacher_campaign_frozen', accepted=len(plan['accepted']),
            existing_live=len(plan['external']), remaining=len(plan['pending']), max_parallel=2,
            parameter_training=False)), flush=True)
        if args.execute:
            result = execute(plan, expected, args.output)
            if result['status'] != 'complete': return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
