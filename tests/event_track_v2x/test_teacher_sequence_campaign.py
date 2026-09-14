"""Deterministic scheduling fixtures: no training, remote jobs or real datasets."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.event_track_v2x import run_teacher_sequence_campaign as tool


@pytest.fixture
def campaign():
    return dict(accepted={'0000': dict(sequence_id='0000', frames=2)},
        external={'0001': dict(pid=10), '0002': dict(pid=11)}, pending=['0003', '0004'],
        max_parallel=2, sequence_frames={f'{s:04d}': 2 for s in range(5)})


def simulate(plan, *, failures=None, audit_failures=(), launch_failure=None, observation_failure=None):
    time = [0]; events = []; launched = []; occupancy = [len(plan['external'])]
    ends = {'0001': 2, '0002': 4}; running = set(plan['external'])
    failures = failures or {}
    def launch(s):
        if s == launch_failure: raise ValueError('disk reserve')
        assert s not in launched and s not in plan['accepted'] and s not in plan['external']
        launched.append(s); running.add(s); ends[s] = time[0] + 3
        occupancy.append(len(running)); assert len(running) <= 2
        return dict(pid=20 + len(launched))
    def poll(s, identity):
        if s == observation_failure: raise RuntimeError('temporary observation failure')
        if time[0] < ends[s]: return None
        running.remove(s)
        return failures.get(s, 'external_exit_unobserved' if s in plan['external'] else 0)
    def accept(s):
        if s in audit_failures: raise ValueError('invalid complete leaf')
        return dict(sequence_id=s, frames=2)
    result = tool.drive(plan, launch, poll, accept, events.append, lambda: time.__setitem__(0, time[0]+1))
    return result, launched, events, occupancy, running


def test_existing_two_processes_consume_slots_and_pending_order_is_fixed(campaign):
    result, launched, events, occupancy, running = simulate(campaign)
    assert result['status'] == 'complete' and result['completed_sequences'] == 5
    assert result['completed_frames'] == 10 and not running
    assert launched == ['0003', '0004'] and max(occupancy) == 2
    assert [e['event'] for e in events[:2]] == ['teacher_accepted', 'teacher_started']
    for flag in ('full_teacher_assembly_completed', 'parameter_training', 'training_submitted', 'paper_eligible'):
        assert result[flag] is False
    assert campaign['pending'] == ['0003', '0004']  # Frozen plan never mutated.


@pytest.mark.parametrize('status', [1, 2, -9, 'unknown'])
def test_nonzero_exit_drains_existing_workers_without_retry_or_more_launches(campaign, status):
    result, launched, _, _, running = simulate(campaign, failures={'0001': status})
    assert result['status'] == 'incomplete' and not launched and not running
    assert result['pending'] == ['0003', '0004']
    assert set(result['accepted']) == {'0000', '0002'}
    assert result['failures'][0]['sequence'] == '0001'


def test_missing_or_bad_leaf_after_external_exit_never_counts_as_completion(campaign):
    result, launched, _, _, _ = simulate(campaign, audit_failures={'0001'})
    assert not launched and result['status'] == 'incomplete'
    assert '0001' not in result['accepted']


def test_launch_failure_drains_other_running_worker(campaign):
    result, launched, events, _, running = simulate(campaign, launch_failure='0003')
    assert result['status'] == 'incomplete' and not launched and not running
    assert set(result['accepted']) == {'0000', '0001', '0002'}
    assert any(e['event'] == 'teacher_launch_blocked' for e in events)


def test_observation_error_never_triggers_restart(campaign):
    with pytest.raises(RuntimeError, match='observation'):
        simulate(campaign, observation_failure='0001')


def test_wrong_accepted_coverage_cannot_be_full(campaign):
    campaign['accepted']['0000']['frames'] = 1
    result, *_ = simulate(campaign)
    assert result['status'] == 'incomplete' and not result['all_sequence_teachers_audited']


def test_third_existing_process_is_rejected_before_any_launch(campaign):
    campaign['external']['0009'] = {'pid': 19}
    with pytest.raises(ValueError, match='concurrency'):
        simulate(campaign)


@pytest.mark.parametrize('returncode,stdout,stderr,missing', [
    (1, '', '', True), (1, '', 'operation not permitted', False),
    (2, '', '', False), (0, '', '', False), (0, 'bad', '', False)])
def test_ps_missing_is_distinct_from_observation_failure(monkeypatch, returncode, stdout, stderr, missing):
    monkeypatch.setattr(tool.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr))
    if missing: assert tool.process_identity(123) is None
    else:
        with pytest.raises((ValueError, RuntimeError)): tool.process_identity(123)


def test_ps_timeout_does_not_mean_process_exited(monkeypatch):
    def timeout(*a, **kw): raise tool.subprocess.TimeoutExpired('ps', 10)
    monkeypatch.setattr(tool.subprocess, 'run', timeout)
    with pytest.raises(tool.subprocess.TimeoutExpired): tool.process_identity(123)


def test_process_start_and_exact_arguments_are_retained(monkeypatch):
    result = SimpleNamespace(returncode=0, stderr='', stdout=' Mon Sep 14 07:21:49 2026 /python collector --arg x\n')
    monkeypatch.setattr(tool.subprocess, 'run', lambda *a, **kw: result)
    assert tool.process_identity(123) == dict(pid=123, start='Mon Sep 14 07:21:49 2026', command='/python collector --arg x')


def test_reused_pid_never_touches_the_new_process(monkeypatch, tmp_path):
    old = dict(pid=123, start='old', command='old collector')
    monkeypatch.setattr(tool, 'process_identity', lambda p: dict(pid=p, start='new', command='unrelated'))
    live = tool.LiveCollectors({}, tmp_path)
    assert live.poll('0001', old) == 'external_exit_unobserved'


def test_adoption_requires_exact_collector_arguments():
    paths = dict(python='/python', root='/private/teacher', cache='/cache', metadata='/meta', checkpoint='/cp', poses='/poses')
    command = tool.command_for(paths, '0029')
    assert tool.matches_command(dict(command=tool.shlex.join(command)), command)
    absolute = command.copy(); absolute[1] = str(tool.ROOT / command[1])
    assert tool.matches_command(dict(command=tool.shlex.join(absolute)), command)
    wrong = command.copy(); wrong[wrong.index('--development-sequence')+1] = '0030'
    assert not tool.matches_command(dict(command=tool.shlex.join(wrong)), command)
    assert not tool.matches_command(dict(command=tool.shlex.join(command + ['--geometry-development'])), command)


def test_ordinary_output_rejects_symlinks(tmp_path):
    real = tmp_path / 'real'; real.mkdir(); link = tmp_path / 'link'; link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError): tool.ordinary(link / 'future')


def test_os_lock_excludes_a_second_controller_but_stale_file_does_not_block(tmp_path):
    with tool.campaign_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with tool.campaign_lock(tmp_path): pytest.fail('second controller acquired lock')
    assert (tmp_path / '.teacher-sequence-campaign.lock').exists()
    with tool.campaign_lock(tmp_path): pass


@pytest.mark.parametrize('reason', ['output_exists', 'low_disk', 'input_changed'])
def test_no_child_when_prelaunch_guard_fails(tmp_path, monkeypatch, reason):
    paths = dict(python='/python', root=str(tmp_path), cache='/cache', metadata='/meta', checkpoint='/cp', poses='/poses')
    plan = dict(paths=paths, minimum_free_bytes=10)
    monkeypatch.setattr(tool, 'unchanged', lambda p: None)
    monkeypatch.setattr(tool.shutil, 'disk_usage', lambda p: SimpleNamespace(free=0 if reason == 'low_disk' else 100))
    monkeypatch.setattr(tool.subprocess, 'Popen', lambda *a, **kw: pytest.fail('guard launched child'))
    if reason == 'output_exists': (tmp_path / tool.leaf_name('0030')).mkdir()
    if reason == 'input_changed':
        def changed(p): raise ValueError('input changed')
        monkeypatch.setattr(tool, 'unchanged', changed)
    with pytest.raises(ValueError): tool.LiveCollectors(plan, tmp_path).launch('0030')


def test_new_child_uses_fixed_env_no_shell_and_private_log(tmp_path, monkeypatch):
    paths = dict(python='/python', root=str(tmp_path), cache='/cache', metadata='/meta', checkpoint='/cp', poses='/poses')
    plan = dict(paths=paths, minimum_free_bytes=10)
    monkeypatch.setattr(tool, 'unchanged', lambda p: None)
    monkeypatch.setattr(tool.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100))
    def spawn(command, **kw):
        assert command == tool.command_for(paths, '0030')
        assert kw['cwd'] == tool.ROOT and kw['stdin'] == tool.subprocess.DEVNULL
        assert 'shell' not in kw and all(kw['env'][k] == v for k, v in tool.THREAD_ENV.items())
        assert kw['stdout'].name == str(tmp_path / 'collector-0030.log')
        return SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(tool.subprocess, 'Popen', spawn)
    live = tool.LiveCollectors(plan, tmp_path)
    assert live.launch('0030') == dict(pid=123, owned=True)
    live.close_log('0030')


def test_source_change_is_detected(tmp_path):
    p = tmp_path / 'source'; p.write_text('before')
    plan = dict(input_sha256={str(p): tool.audit.sha_file(p)})
    tool.unchanged(plan); p.write_text('after')
    with pytest.raises(ValueError, match='changed'): tool.unchanged(plan)


@pytest.mark.parametrize('bad', ['campaign', 'accepted_artifact', 'input'])
def test_changed_final_evidence_never_produces_success_receipt(tmp_path, monkeypatch, bad):
    source = tmp_path / 'source'; source.write_text('source')
    artifact = tmp_path / 'artifact'; artifact.write_text('teacher')
    plan = dict(input_sha256={str(source): tool.audit.sha_file(source)})
    tool.write_json(tmp_path / 'campaign.json', plan)
    result = dict(status='complete', completed_sequences=1, completed_frames=2, parameter_training=False,
                  accepted={'0000': dict(artifact_sha256={str(artifact): tool.audit.sha_file(artifact)})})
    def drive(*a, **kw):
        target = tmp_path / 'campaign.json' if bad == 'campaign' else source if bad == 'input' else artifact
        target.write_text('changed')
        return result
    monkeypatch.setattr(tool, 'drive', drive)
    with pytest.raises(ValueError, match='changed'): tool.execute(plan, {}, tmp_path)
    assert not (tmp_path / 'receipt.json').exists()
    stopped = json.loads((tmp_path / 'events.jsonl').read_text())
    assert stopped['existing_processes_not_terminated'] and stopped['automatic_retry'] is False


@pytest.mark.parametrize('externals', [[('0001', 2), ('0001', 3)], [('0001', 2), ('0002', 2)],
                                      [('0001', 2), ('0002', 3), ('0003', 4)]])
def test_prepare_rejects_ambiguous_or_excessive_external_ownership(tmp_path, externals):
    with pytest.raises(ValueError, match='distinct existing'):
        tool.prepare(tmp_path, tmp_path / 'cp', tmp_path / 'poses', tmp_path / 'reference',
                     'unused', externals, tmp_path / 'new-output')


def test_prepare_refuses_reuse_of_existing_campaign(tmp_path):
    with pytest.raises(ValueError, match='new campaign'):
        tool.prepare(tmp_path, tmp_path / 'cp', tmp_path / 'poses', tmp_path / 'reference',
                     'unused', [], tmp_path)
