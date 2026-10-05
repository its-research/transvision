"""Ensure observation errors cannot restart readers or bypass full admission."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as N

import pytest


@pytest.fixture
def code(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'tools/event_track_v2x'))
    m = importlib.import_module('continue_final_topk_seed3407')
    monkeypatch.setattr(m, 'ROOT', tmp_path / 'controller')
    monkeypatch.setattr(m, 'GATE', tmp_path / 'gate.json')
    monkeypatch.setattr(m, 'register', lambda *args: None)
    return m


@pytest.mark.parametrize('result,expected', [
    (N(returncode=1, stdout='', stderr=''), 'absent'),
    (N(returncode=1, stdout='', stderr='denied'), 'unknown'),
    (N(returncode=2, stdout='', stderr=''), 'unknown'),
    (N(returncode=0, stdout='', stderr=''), 'unknown'),
    (N(returncode=0, stdout='identity', stderr=''), 'live'),
])
def test_only_empty_successful_absence_is_terminal(code, monkeypatch, result, expected):
    monkeypatch.setattr(code.subprocess, 'run', lambda *a, **kw: result)
    assert code.observe(123)[0] == expected


@pytest.mark.parametrize('error', [OSError('probe failed'), __import__('subprocess').TimeoutExpired('ps', 10)])
def test_observer_exception_is_unknown(code, monkeypatch, error):
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(code.subprocess, 'run', fail)
    assert code.observe(123) == ('unknown', None)


def identity(m, seed=3407):
    return f'Sun Oct  4 19:12:00 2026 /python {m.READER} --seed {seed} --method topk'


def test_reader_identity_requires_exact_seed_method_and_source(code):
    assert code.bind_reader(identity(code)) == identity(code)
    for wrong in (identity(code, 2027), identity(code) + ' --extra', identity(code).replace('--method topk', '--method rbf')):
        with pytest.raises(AssertionError):
            code.bind_reader(wrong)


def setup_join(m, monkeypatch, states, qualifier_error=False):
    m.GATE.write_text('{}')
    monkeypatch.setattr(m, 'source_gate', lambda: {'frozen': True})
    monkeypatch.setattr(m, 'reject_prior_evidence', lambda: None)
    monkeypatch.setattr(m, 'reject_duplicate_process', lambda: None)
    observations = iter(states)
    monkeypatch.setattr(m, 'observe', lambda pid: next(observations))
    monkeypatch.setattr(m.time, 'sleep', lambda n: None)
    launched = []
    def qualify():
        if qualifier_error:
            raise AssertionError('full gate rejected')
        return dict(byte_admission_sha256=m.sha(m.GATE))
    monkeypatch.setattr(m, 'qualify', qualify)
    def popen(command, **kwargs):
        launched.append(command)
        return N(pid=456, wait=lambda: 0)
    monkeypatch.setattr(m.subprocess, 'Popen', popen)
    return launched


def test_unknown_observation_waits_then_runs_unchanged_driver_once(code, monkeypatch):
    live = ('live', identity(code))
    launched = setup_join(code, monkeypatch, [live, ('unknown', None), live, ('absent', None)])
    code.run(123)
    assert launched == [[code.PYTHON, str(code.DRIVER), '--byte-admission', str(code.GATE),
                         '--output', str(code.ROOT / 'independent-acceptance')]]
    outcome = json.loads((code.ROOT / 'process-exit.json').read_bytes())
    assert outcome['experiment_accepted'] is False and outcome['requires_independent_receipt_review'] is True
    with pytest.raises(AssertionError):
        code.run(123)
    assert len(launched) == 1


@pytest.mark.parametrize('condition', ['changed-pid', 'gate-failure', 'initial-unknown'])
def test_failed_preconditions_never_launch_or_restart(code, monkeypatch, condition):
    live = ('live', identity(code))
    states = [('unknown', None)] if condition == 'initial-unknown' else [live,
        ('live', identity(code).replace('19:12', '19:13')) if condition == 'changed-pid' else ('absent', None)]
    launched = setup_join(code, monkeypatch, states, condition == 'gate-failure')
    with pytest.raises(AssertionError):
        code.run(123)
    assert not launched
    if condition != 'initial-unknown':
        failure = json.loads((code.ROOT / 'controller-failure.json').read_bytes())
        assert failure['reader_left_untouched'] is True and failure['automatic_retry'] is False


def test_duplicate_live_cpu_refused(code, monkeypatch):
    duplicate = f'{code.PYTHON} {code.DRIVER} --byte-admission {code.GATE} --output /unused'
    monkeypatch.setattr(code.subprocess, 'run', lambda *a, **kw: N(returncode=0, stderr='', stdout=duplicate))
    with pytest.raises(AssertionError, match='already has a live'):
        code.reject_duplicate_process()
