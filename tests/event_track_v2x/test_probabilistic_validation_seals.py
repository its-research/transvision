"""Full-val provenance gates; tiny mocks are not benchmark evidence."""
from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.event_track_v2x import run_probabilistic_tracking_v2 as tool


def test_source_inventory_covers_lazy_backends_and_initializers():
    sources = tool.validation_sources()
    for name in ('__init__.py', 'beam_recovery_allocation.py', 'persistent_joint_beam.py',
                 'persistent_cache_stream.py', 'jpda_filter.py', 'pkf_filter.py'):
        assert 'transvision/models/event_track_v2x/' + name in sources
    assert 'transvision/register.py' in sources
    assert 'tools/event_track_v2x/run_probabilistic_tracking_v2.py' in sources
    assert 'tools/event_track_v2x/evaluate_source_ablation_v2.py' not in sources


@pytest.mark.parametrize('change', ['add', 'remove', 'edit'])
def test_source_inventory_changes_cannot_receive_final_seal(monkeypatch, change):
    before = {'one.py': 'a' * 64, 'two.py': 'b' * 64}
    after = dict(before)
    if change == 'add':
        after['three.py'] = 'c' * 64
    elif change == 'remove':
        after.pop('two.py')
    else:
        after['two.py'] = 'c' * 64
    monkeypatch.setattr(tool, 'validation_sources', lambda: after)
    with pytest.raises(ValueError, match='sources or input'):
        tool.require_unchanged(before, {})


def test_checkpoint_weights_are_bound_and_rechecked(tmp_path, monkeypatch):
    cache, checkpoint = tmp_path/'cache', tmp_path/'checkpoint'
    cache.mkdir(); checkpoint.mkdir()
    for p in (cache/'manifest.json', tmp_path/'schedule.json',
              checkpoint/'checkpoint.json', checkpoint/'weights.pt'):
        p.write_bytes(b'fixture')
    args = Namespace(cache=cache, schedule=tmp_path/'schedule.json', checkpoint=checkpoint)
    inputs = tool.input_files(args, {'weights': {'path': 'weights.pt'}})
    assert len(inputs) == 4
    monkeypatch.setattr(tool, 'validation_sources', lambda: {})
    tool.require_unchanged({}, inputs)
    (checkpoint/'weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='sources or input'):
        tool.require_unchanged({}, inputs)


def test_runtime_rejects_unpinned_environment_and_non_cpu(monkeypatch):
    with pytest.raises(ValueError, match='CPU inference'):
        tool.runtime_evidence('cuda')
    monkeypatch.delenv('PYTHONHASHSEED', raising=False)
    with pytest.raises(ValueError, match='fresh-process'):
        tool.runtime_evidence('cpu')


def test_actual_fresh_process_records_one_thread_runtime():
    env = dict(os.environ, **tool.THREAD_ENV)
    result = subprocess.run([sys.executable, '-c',
        'import json; from tools.event_track_v2x.run_probabilistic_tracking_v2 import runtime_evidence; '
        'print(json.dumps(runtime_evidence("cpu")))'], cwd=tool.ROOT, env=env,
        text=True, capture_output=True, check=True, timeout=60)
    value = json.loads(result.stdout)
    assert value['torch_threads'] == value['torch_interop_threads'] == 1
    assert value['thread_environment'] == tool.THREAD_ENV
    assert value['pid'] != os.getpid()
    assert not value['exclusive_host'] and not value['same_latency_or_memory_verified']


@pytest.fixture
def val_gate(tmp_path, monkeypatch):
    cache = tmp_path/'cache'; cache.mkdir()
    manifest = dict(split='val', frame_count=7189, sequences=['fixture'])
    (cache/'manifest.json').write_text(json.dumps(manifest))
    schedule = tmp_path/'schedule.json'; schedule.write_text('fixture')
    args = Namespace(cache=cache, cache_sha256=tool.sha_file(cache/'manifest.json'),
        schedule=schedule, schedule_sha256=tool.sha_file(schedule), checkpoint=None,
        checkpoint_sha256=None, output=tmp_path/'out', device='cpu')
    monkeypatch.setattr(tool, 'schedule_rows', lambda *a: [{'sequence_id': 'fixture'}])
    monkeypatch.setattr(tool, 'VerifiedForestCache',
        lambda *a: SimpleNamespace(manifest_json=json.dumps(manifest)))
    monkeypatch.setattr(tool, 'runtime_evidence', lambda device: {'fixture': True})
    return args


@pytest.mark.parametrize('failure', ['short', 'source', 'input'])
def test_postflight_failure_never_writes_full_validation_receipt(val_gate, monkeypatch, failure):
    args = val_gate
    snapshot = {'fixture.py': 'a' * 64}
    monkeypatch.setattr(tool, 'validation_sources', lambda: dict(snapshot))
    def replay(cache, rows, output, config, **kwargs):
        output.mkdir()
        assert config.association_algorithm == 'lbp'
        assert config.anchor_decoder == 'joint-map'
        assert kwargs['plan']['runtime'] == {'fixture': True}
        if failure == 'source':
            snapshot['fixture.py'] = 'b' * 64
        elif failure == 'input':
            Path(args.schedule).write_text('changed')
        return dict(status='complete', completed_frames=2 if failure == 'short' else 3316,
                    scheduled_frames=3316, sequence_heads={str(i): {} for i in range(21)})
    monkeypatch.setattr(tool, 'replay_rows', replay)
    with pytest.raises(ValueError, match='changed during replay|complete 21-sequence'):
        tool.run(args)
    assert not (args.output/'full-validation-receipt.json').exists()
