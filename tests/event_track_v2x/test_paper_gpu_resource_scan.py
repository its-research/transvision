"""CPU software fixtures with mocked CUDA counters; no GPU acceptance claims."""
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_paper_resource_scan import asset, spec  # noqa: F401
from tools.event_track_v2x import run_paper, scan_paper_resources as scan
from transvision.models.event_track_v2x import paper_gpu_resources as gpu
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file


def contract():
    return dict(kind=gpu.KIND, device='cuda:0', gpu_uuid='GPU-fixture-0', warmup_events=1,
                torch_threads=1, torch_interop_threads=1, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
                forward_batch_protocol='native_causal_frontier_one_call_at_a_time_v1',
                TF32=False, memory_target_percent=[75, 80])


def hardware():
    return dict(device='cuda:0', GPU_uuid='GPU-fixture-0', GPU_name='SOFTWARE FIXTURE',
                GPU_total_memory_bytes=1000, torch_version=torch.__version__, CUDA_version=torch.version.cuda,
                torch_threads=1, torch_interop_threads=1, TF32_matmul=False, TF32_cudnn=False)


@pytest.fixture
def mocked_counters(monkeypatch):
    props = SimpleNamespace(uuid='GPU-fixture-0', name='SOFTWARE FIXTURE', total_memory=1000)
    for name, function in dict(get_device_properties=lambda *_: props, synchronize=lambda *_: None,
                               set_device=lambda *_: None, reset_peak_memory_stats=lambda *_: None,
                               mem_get_info=lambda *_: (200, 1000), memory_allocated=lambda *_: 100,
                               memory_reserved=lambda *_: 200, max_memory_allocated=lambda *_: 100,
                               max_memory_reserved=lambda *_: 200).items():
        monkeypatch.setattr(torch.cuda, name, function)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(gpu, 'configure', lambda *_: hardware())


@pytest.mark.parametrize('key,value', [
    ('device', 'cuda'), ('device', 'cpu'), ('device', 'cuda:-1'), ('gpu_uuid', ''),
    ('gpu_uuid', 'unavailable'), ('warmup_events', 0), ('warmup_events', True),
    ('torch_threads', 0), ('torch_interop_threads', 1.5), ('OMP_NUM_THREADS', '1'),
    ('OPENBLAS_NUM_THREADS', -1), ('TF32', True), ('memory_target_percent', [1, 99]),
    ('forward_batch_protocol', 'auto-batch'), ('unexpected', True)])
def test_runtime_contract_rejects_implicit_or_changed_execution(key, value):
    changed = contract(); changed[key] = value
    with pytest.raises(ValueError): gpu.validate_contract(changed)


def test_contract_requires_hash_and_matching_cli_device(tmp_path):
    path = tmp_path/'GPU.json'; binding = asset(path, contract())
    assert gpu.load_contract(path, binding['sha256'], 'cuda:0') == contract()
    for digest, device in [('0'*64, 'cuda:0'), (binding['sha256'], 'cuda:1')]:
        with pytest.raises(ValueError): gpu.load_contract(path, digest, device)


@pytest.mark.parametrize('fault', ['unavailable', 'index', 'uuid', 'L40S', 'threads', None])
def test_configure_checks_actual_physical_device_and_threads(monkeypatch, fault):
    props = SimpleNamespace(uuid='wrong' if fault == 'uuid' else 'GPU-fixture-0',
                            name='NVIDIA L40S' if fault == 'L40S' else 'SOFTWARE FIXTURE', total_memory=1000)
    calls = []
    fake = SimpleNamespace(__version__='fixture', version=SimpleNamespace(cuda='fixture'), device=torch.device,
        cuda=SimpleNamespace(is_available=lambda: fault != 'unavailable', device_count=lambda: 0 if fault == 'index' else 1,
                             set_device=lambda d: calls.append(str(d)), get_device_properties=lambda d: props,
                             synchronize=lambda d: None),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                 cudnn=SimpleNamespace(allow_tf32=True)),
        set_num_threads=lambda n: calls.append(n), set_num_interop_threads=lambda n: calls.append(n),
        get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1,
        set_float32_matmul_precision=lambda value: calls.append(value))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setenv('OMP_NUM_THREADS', '2' if fault == 'threads' else '1')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '1')
    if fault:
        with pytest.raises(ValueError): gpu.configure(contract())
    else:
        result = gpu.configure(contract())
        assert result['GPU_uuid'] == props.uuid and not result['TF32_matmul'] and not result['TF32_cudnn']
        assert 'highest' in calls and 'cuda:0' in calls


@pytest.fixture
def trial(tmp_path, spec, mocked_counters, monkeypatch):
    root = tmp_path/'trial'; root.mkdir()
    runtime = asset(tmp_path/'runtime.json', contract())
    argv = ['replay', '--cache', str(Path(spec['cache_manifest']['path']).parent),
            '--cache-sha256', spec['cache_manifest']['sha256'], '--schedule', spec['schedule']['path'],
            '--schedule-sha256', spec['schedule']['sha256'], '--configuration', spec['candidates'][0]['configuration']['path'],
            '--configuration-sha256', spec['candidates'][0]['configuration']['sha256'],
            '--dataset', 'v2v4real', '--split', 'train', '--output', str(root/'replay'), '--fixture',
            '--device', 'cuda:0', '--resource-contract', runtime['path'], '--resource-contract-sha256', runtime['sha256']]
    monkeypatch.setattr(sys, 'argv', ['run_paper.py', *argv])
    assert run_paper.main() == 0
    events = [json.loads(line) for line in Path(spec['schedule']['path']).read_bytes().splitlines()]
    return root, runtime, events


def check_trial(trial):
    root, runtime, events = trial
    return gpu.validate_trial(root, contract(), runtime['sha256'], events,
                              json.loads((root/'replay/plan.json').read_bytes()),
                              json.loads((root/'replay/receipt.json').read_bytes()))


def test_warmup_uses_prefix_and_fresh_forest_with_full_formal_schedule(trial, capsys):
    root, _, _ = trial
    numbers, actual = check_trial(trial)
    assert actual == hardware() and numbers['peak_device_tensor_bytes'] == 100
    warm = json.loads((root/'replay-warmup/receipt.json').read_bytes())
    formal = json.loads((root/'replay/receipt.json').read_bytes())
    assert warm['completed_events'] == 1 and formal['completed_events'] == 2
    predictions = [json.loads(row) for row in (root/'replay/predictions.jsonl').read_bytes().splitlines()]
    assert len(predictions) == 2
    assert 'paper_replay_events' in (root/'replay-warmup.stdout').read_text()
    binding = json.loads((root/'replay/GPU-resource-binding.json').read_bytes())
    assert not binding['warmup_forest_state_reused'] and not binding['memory_target_independently_accepted']
    assert not binding['equal_resources_claimed']


@pytest.mark.parametrize('fault', ['uuid', 'TF32', 'sampler', 'occupancy', 'sample-time', 'peak',
                                 'tensor-reserve', 'throughput', 'warmup-prefix', 'warmup-config',
                                 'claim', 'measurement-claim', 'scope', 'contract', 'bytes'])
def test_rejects_contradictory_measurements_even_with_fresh_hashes(trial, fault):
    root, _, _ = trial
    mp = root/'replay/GPU-runtime-measurement.json'; bp = root/'replay/GPU-resource-binding.json'
    wp = root/'replay-warmup/plan.json'
    m = json.loads(mp.read_bytes()); b = json.loads(bp.read_bytes())
    if fault == 'uuid': m['GPU_uuid'] = 'GPU-wrong'
    elif fault == 'TF32': m['TF32_matmul'] = True
    elif fault == 'sampler': m['sampler_finished'] = False
    elif fault == 'occupancy': m['samples'][0]['device_used_percent'] = 79
    elif fault == 'sample-time': m['samples'][0]['elapsed_seconds'] = -1
    elif fault == 'peak': m['peak_process_tensor_allocated_bytes'] = 99
    elif fault == 'tensor-reserve': m['samples'][0]['process_tensor_allocated_bytes'] = 300
    elif fault == 'throughput': m['events_per_second_with_observer'] *= 2
    elif fault == 'warmup-prefix': b['warmup']['event_prefix_sha256'] = '0'*64
    elif fault == 'warmup-config':
        w = json.loads(wp.read_bytes()); w['configuration']['method'] = 'changed'
        wp.write_bytes(canonical(w)); b['warmup']['plan_sha256'] = sha_file(wp)
    elif fault == 'claim': b['equal_resources_claimed'] = True
    elif fault == 'measurement-claim': m['target_independently_accepted'] = True
    elif fault == 'scope': b['process_lifetime_host_peak_includes_setup_and_warmup'] = False
    elif fault == 'contract': b['contract']['warmup_events'] = 2
    mp.write_bytes(canonical(m)); b['measurement_sha256'] = sha_file(mp)
    if fault == 'bytes': b['measurement_sha256'] = '0'*64
    bp.write_bytes(canonical(b))
    with pytest.raises(ValueError): check_trial(trial)


def test_gpu_plan_binds_sources_and_failure_preserves_requested_device(tmp_path, spec, monkeypatch):
    spec['GPU_runtime'] = asset(tmp_path/'runtime.json', contract())
    spec['constraints']['peak_device_tensor_bytes'] = 500
    plan = scan.plan_scan(spec, tmp_path/'plan'); path = tmp_path/'plan/plan.json'
    assert plan['kind'] == 'rbf_resource_scan_plan_v2_GPU'
    observed = []
    def fail(argv, env, stdout_path, stderr_path, timeout, **kwargs):
        observed.append((argv, env)); Path(stdout_path).write_text(''); Path(stderr_path).write_text('fixture failure')
        return 19
    monkeypatch.setattr(scan, 'run_stage_with_eta', fail)
    with pytest.raises(RuntimeError, match='failed: 19'):
        scan.run_scan(path, sha_file(path), tmp_path/'run', python=sys.executable, evaluator_python=sys.executable)
    argv, env = observed[0]
    assert argv[argv.index('--device')+1] == 'cuda:0' and argv[argv.index('--resource-contract')+1] == spec['GPU_runtime']['path']
    assert env['OMP_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '1'
    assert (tmp_path/'run/failure.json').is_file() and not (tmp_path/'run/receipt.json').exists()
    changed = copy.deepcopy(plan); changed['GPU_source_sha256'][str(Path(gpu.__file__))] = '0'*64
    with pytest.raises(ValueError, match='sources changed'): scan.validate_plan_source(changed)


def test_gpu_scanner_freeze_with_mock_counters_and_real_independent_evaluator(tmp_path, spec, mocked_counters, monkeypatch):
    evaluator = os.environ.get('RBF_EVALUATOR_PYTHON')
    if not evaluator: pytest.skip('independent evaluator environment not supplied')
    spec['GPU_runtime'] = asset(tmp_path/'runtime.json', contract())
    spec['constraints'].update(peak_device_tensor_bytes=500, peak_device_reserved_bytes=500, replay_process_seconds=60)
    scan.plan_scan(spec, tmp_path/'plan'); plan_path = tmp_path/'plan/plan.json'
    original = scan.run_stage_with_eta
    def local_fixture(argv, env, stdout_path, stderr_path, timeout, **kwargs):
        if kwargs['stage'] != 'replay':
            assert 'PYTHONPATH' not in env and 'PYTHONHOME' not in env and env['PYTHONNOUSERSITE'] == '1'
            return original(argv, env, stdout_path, stderr_path, timeout, **kwargs)
        monkeypatch.setattr(sys, 'argv', argv[1:])
        Path(stdout_path).write_text('in-process CPU fixture; mocked GPU counters')
        Path(stderr_path).write_text('')
        return run_paper.main()
    monkeypatch.setattr(scan, 'run_stage_with_eta', local_fixture)
    scan.run_scan(plan_path, sha_file(plan_path), tmp_path/'run', python=sys.executable, evaluator_python=evaluator)
    receipt = tmp_path/'run/receipt.json'
    selected = scan.freeze_scan(plan_path, sha_file(plan_path), tmp_path/'run', sha_file(receipt), tmp_path/'freeze')
    assert selected['selected_candidate'] == 'geometry' and selected['warmup_measured_separately']
    assert not selected['equal_resources_claimed'] and not selected['memory_target_independently_accepted']
    assert len(selected['candidates'][0]['trials']) == 3
    data = json.loads(receipt.read_bytes()); del data['jobs'][0]['files']['replay/GPU-runtime-measurement.json']
    receipt.write_bytes(canonical(data))
    with pytest.raises(ValueError, match='incomplete artifact'):
        scan.freeze_scan(plan_path, sha_file(plan_path), tmp_path/'run', sha_file(receipt), tmp_path/'invalid')
