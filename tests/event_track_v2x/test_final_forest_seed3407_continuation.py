"""Original prefetch must finish before any terminal reader or CPU verifier."""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import pytest


@pytest.fixture
def code(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'tools/event_track_v2x'))
    m = importlib.import_module('continue_final_forest_seed3407')
    monkeypatch.setattr(m, 'R', tmp_path)
    monkeypatch.setattr(m, 'ROOT', tmp_path / 'artifacts/chain')
    monkeypatch.setattr(m, 'CACHE', tmp_path / 'cache')
    m.CACHE.mkdir()
    monkeypatch.setattr(m, 'GATE', m.CACHE / 'admission.json')
    monkeypatch.setattr(m, 'FINISHED', tmp_path / 'finished.json')
    monkeypatch.setattr(m, 'JOURNAL', tmp_path / 'journal.json')
    monkeypatch.setattr(m, 'register', lambda *a: None)
    monkeypatch.setattr(m, 'ledger_binding', lambda *a: None)
    return m


def test_unknown_observation_does_not_complete_wait(code, monkeypatch):
    states = iter([('unknown', None), ('live', 'original'), ('absent', None)])
    waited = []
    monkeypatch.setattr(code, 'observe', lambda p: next(states))
    monkeypatch.setattr(code.time, 'sleep', lambda seconds: waited.append(seconds))
    code.wait_original(123, 'original')
    assert waited == [30, 30]


def test_reused_pid_cannot_complete_wait(code, monkeypatch):
    monkeypatch.setattr(code, 'observe', lambda p: ('live', 'different-start'))
    with pytest.raises(AssertionError, match='reused'):
        code.wait_original(123, 'original')


def snapshot():
    return dict(kind='rbf_final_refit_registered_rank_byte_prefetch_v1', seed=3407,
                task_id='fixture', pid=123, registered_snapshot={'replay-rank0': dict(bytes=8, sha256='a'*64)})


@pytest.mark.parametrize('mutation', ['none', 'partial', 'error', 'changed-snapshot', 'wrong-identity'])
def test_finished_prefetch_requires_exact_complete_snapshot(code, mutation):
    start = snapshot()
    value = dict(start, all_snapshot_bytes_match=True, registered_snapshot_unchanged_at_end=True,
                 failures={}, independently_read_artifacts=copy.deepcopy(start['registered_snapshot']),
                 extraction_started=False, task_mutated=False, uploads_performed=False)
    if mutation == 'partial': value['all_snapshot_bytes_match'] = False
    if mutation == 'error': value['failures']['download'] = 'TimeoutError'
    if mutation == 'changed-snapshot': value['independently_read_artifacts']['replay-rank0']['bytes'] = 7
    if mutation == 'wrong-identity': value['seed'] = 1337
    code.FINISHED.write_text(json.dumps(value))
    if mutation == 'none':
        assert code.qualify_prefetch(start) == value
    else:
        with pytest.raises(AssertionError): code.qualify_prefetch(start)


@pytest.mark.parametrize('world', [4, 8])
@pytest.mark.parametrize('change', ['none', 'active', 'failed', 'missing-rank', 'changed-bytes'])
def test_terminal_reader_requires_live_completed_full_artifacts(code, monkeypatch, change, world):
    plan = dict(seed=3407, method='rbf', world_size=world,
                bootstrap_sha256=code.hashlib.sha256(b'fixture').hexdigest())
    digest = code.hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    job = dict(task_id=code.TASK, seed=3407, recipe_sha256=digest, plan=plan)
    code.JOURNAL.write_text(json.dumps(dict(jobs=[job])))
    artifacts = {k:N(hash='a'*64, size=8) for k in ['receipt','exclusive-source-manifest'] + [f'replay-rank{i}' for i in range(world)]}
    task = N(status='completed', data=N(script=N(diff='fixture')), artifacts=artifacts)
    if change in ('active','failed'): task.status = 'in_progress' if change == 'active' else 'failed'
    if change == 'missing-rank': del artifacts[f'replay-rank{world-1}']
    if change == 'changed-bytes': artifacts['replay-rank0'].size = 9
    monkeypatch.setitem(sys.modules, 'clearml', N(Task=N(get_task=lambda **kw: task)))
    proof = dict(recipe_sha256=digest, registered_snapshot={'replay-rank0':dict(bytes=8,sha256='a'*64)})
    if change == 'none': assert code.qualify_completed_task(proof) == job
    else:
        with pytest.raises(AssertionError): code.qualify_completed_task(proof)


def chain(code, monkeypatch, fail_at=None):
    calls = []
    for name in ('source_gate','reject_existing','reject_competing_processes','reject_prior_CPU'):
        monkeypatch.setattr(code, name, lambda: {})
    monkeypatch.setattr(code, 'admitted_start', lambda p: ({}, 'identity'))
    checkpoint = code.R / 'checkpoint'
    checkpoint.write_text('{}')
    code.FINISHED.write_text('{}'); code.GATE.write_text('{}')
    def stage(name, result):
        def call(*args):
            calls.append(name)
            if name == fail_at: raise AssertionError(name)
            return result
        return call
    monkeypatch.setattr(code, 'wait_original', stage('wait', None))
    monkeypatch.setattr(code, 'qualify_prefetch', stage('prefetch', {}))
    monkeypatch.setattr(code, 'qualify_completed_task', stage('task', {'recipe_sha256':'r'}))
    monkeypatch.setattr(code, 'qualify_bytes', stage('bytes', checkpoint))
    children = []
    def child(name, argv):
        calls.append(name); children.append(argv)
        if name == fail_at: raise AssertionError(name)
    monkeypatch.setattr(code, 'run_child', child)
    return calls, children, checkpoint


def test_exact_dependency_order_and_frozen_commands(code, monkeypatch):
    calls, children, checkpoint = chain(code, monkeypatch)
    code.run(123)
    assert calls == ['wait','prefetch','task','terminal-readback','bytes','full-CPU-admission']
    assert children == [[code.READER_PYTHON,str(code.READER),'--seed','3407','--method','rbf'],
        [code.PYTHON,str(code.DRIVER),'--byte-admission',str(code.GATE),
         '--output',str(code.ROOT/'independent-acceptance'),'--checkpoint',str(checkpoint)]]
    value = json.loads((code.ROOT/'process-exit.json').read_bytes())
    assert value['experiment_accepted'] is False and value['requires_independent_receipt_review'] is True
    with pytest.raises(AssertionError): code.run(123)
    assert len(children) == 2


@pytest.mark.parametrize('fail_at', ['prefetch','task','terminal-readback','bytes'])
def test_upstream_failure_never_starts_CPU_or_retries(code, monkeypatch, fail_at):
    calls, children, _ = chain(code, monkeypatch, fail_at)
    with pytest.raises(AssertionError): code.run(123)
    assert 'full-CPU-admission' not in calls
    assert not (code.ROOT/'process-exit.json').exists()
    value = json.loads((code.ROOT/'controller-failure.json').read_bytes())
    assert value['automatic_retry'] is False and value['original_prefetch_left_untouched'] is True
    count = len(children)
    with pytest.raises(AssertionError): code.run(123)
    assert len(children) == count
