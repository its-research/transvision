"""Launcher contract checks only; these tests do not claim CUDA execution."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def launcher(monkeypatch):
    directory = Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('state_gpu_entry', directory/'qualify_batched_branch_states_gpu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path):
    return ['--replay', str(tmp_path/'reference'), '--receipt-sha256', 'a'*64,
            '--sequence', '0000', '--events', '195',
            '--recorded-reference-acceptance', str(tmp_path/'reference.json'),
            '--reference-acceptance-sha256', 'b'*64,
            '--device', 'cuda:0', '--output', str(tmp_path/'candidate')]


def test_explicit_device_and_reference_required(launcher, tmp_path):
    command = args(tmp_path)
    command[command.index('cuda:0')] = 'cpu'
    with pytest.raises(SystemExit):
        launcher.arguments(command)
    command = args(tmp_path)
    i = command.index('--recorded-reference-acceptance')
    del command[i:i+2]
    with pytest.raises(SystemExit):
        launcher.arguments(command)


@pytest.mark.parametrize('fail', [False, True])
def test_measured_launcher_preserves_failure_and_restores_arguments(launcher, tmp_path, monkeypatch, fail):
    output = tmp_path/'candidate'
    original_argv = sys.argv
    def qualifier():
        assert '--no-profile' in sys.argv
        assert '--recorded-reference-acceptance' in sys.argv
        assert sys.argv[sys.argv.index('--device')+1] == 'cuda:0'
        output.mkdir()
        if fail:
            (output/'failure-retained.txt').write_text('intentional qualifier failure')
            raise ValueError('qualification failed')
        (output/'candidate-check.json').write_text(json.dumps(dict(
            device=dict(GPU_executed=True), complete_reference_sequence_checked=True, events=195)))
    def measure(replay, *, device, output):
        assert device == 'cuda:0'
        import torch
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        return replay(output=output)
    monkeypatch.setattr(launcher, 'qualify', qualifier)
    monkeypatch.setattr(launcher, 'measure_replay', measure)
    if fail:
        with pytest.raises(ValueError, match='qualification failed'):
            launcher.main(args(tmp_path))
        assert (output/'failure-retained.txt').exists()
    else:
        launcher.main(args(tmp_path))
    assert sys.argv is original_argv
    evidence = json.loads((output/'GPU-launch-binding.json').read_bytes())
    assert evidence['production_promotion_allowed'] is False
    assert evidence['memory_target_independently_accepted'] is False
    assert evidence['full_forest_on_GPU'] is False
    assert len(evidence['execution_sources_sha256']) == 3
    with pytest.raises(SystemExit):
        launcher.arguments(args(tmp_path))
