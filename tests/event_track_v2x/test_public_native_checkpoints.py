"""Native weight consumption, including DMSTrack's implicit second path."""
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.event_track_v2x import run_public_baseline as runner


def bind(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())


@pytest.fixture
def plan(tmp_path):
    result = dict(method='DMSTrack', source=str(tmp_path/'source'), python='/test/python',
                  table=3, original_protocol=True)
    result['checkpoint'] = bind(tmp_path/'weights/model_ego_epoch_3.pth', b'ego network')
    result['remote_checkpoint'] = bind(tmp_path/'weights/model_1_epoch_3.pth', b'remote network')
    bind(tmp_path/'source/DMSTrack/configs/v2v4real.yml',
         (Path(__file__).parent/'fixtures/dmstrack-v2v4real-d3b9949.yml').read_bytes())
    for key in ('dataset_manifest', 'split_mapping', 'environment_lock'):
        result[key] = bind(tmp_path/(key+'.json'), b'{}')
    return result


def test_exact_native_paths_are_recorded_and_command_remains_native(plan, tmp_path):
    actual = runner.consumed_checkpoints(plan)
    assert actual['checkpoint']['path'] == plan['checkpoint']['path']
    assert actual['remote_checkpoint']['path'] == plan['remote_checkpoint']['path']
    cwd, argv = runner.command(plan, tmp_path/'out')
    assert cwd == Path(plan['source'])/'DMSTrack'
    assert argv[argv.index('--load_model_path')+1] == plan['checkpoint']['path']
    assert '--use_multiple_nets' in argv
    assert '--remote-checkpoint' not in argv


def test_valid_hash_for_unused_remote_file_is_rejected(plan, tmp_path):
    plan['remote_checkpoint'] = bind(tmp_path/'unused-network.pth', b'unused but valid hash')
    with pytest.raises(ValueError, match='path replacement'):
        runner.consumed_checkpoints(plan)


def test_native_replacement_in_parent_directory_is_not_ignored(plan, tmp_path):
    plan['checkpoint'] = bind(tmp_path/'ego-source/model_ego.pth', b'ego')
    plan['remote_checkpoint'] = bind(tmp_path/'ego-source/model_1.pth', b'incorrect path')
    with pytest.raises(ValueError, match='path replacement'):
        runner.consumed_checkpoints(plan)
    plan['remote_checkpoint'] = bind(tmp_path/'1-source/model_1.pth', b'correct path')
    assert len(runner.consumed_checkpoints(plan)) == 2


@pytest.mark.parametrize('fault', ['no_ego_token', 'missing', 'changed', 'hardlink', 'symlink'])
def test_missing_changed_or_aliased_consumed_weights_fail(plan, tmp_path, fault):
    if fault == 'no_ego_token':
        plan['checkpoint'] = bind(tmp_path/'model.pth', b'ego')
        plan['remote_checkpoint'] = plan['checkpoint'].copy()
    else:
        remote = Path(plan['remote_checkpoint']['path'])
        remote.unlink()
        if fault == 'changed':
            remote.write_bytes(b'wrong network')
        elif fault == 'hardlink':
            os.link(plan['checkpoint']['path'], remote)
            plan['remote_checkpoint']['sha256'] = plan['checkpoint']['sha256']
        elif fault == 'symlink':
            remote.symlink_to(plan['checkpoint']['path'])
            plan['remote_checkpoint']['sha256'] = plan['checkpoint']['sha256']
    with pytest.raises((ValueError, OSError)):
        runner.consumed_checkpoints(plan)


@pytest.fixture
def official_version(monkeypatch):
    def git(argv, **kwargs):
        assert argv[:2] == ['git', '-C']
        if argv[3:] == ['rev-parse', 'HEAD']:
            return 'd3b9949499c8e68ea33060873bd1cb95b6d4d323\n'
        assert argv[3:] == ['diff', 'HEAD', '--']
        return ''
    monkeypatch.setattr(runner.subprocess, 'check_output', git)


def test_preflight_does_not_run_or_create_output(plan, tmp_path, monkeypatch, official_version):
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: pytest.fail('preflight executed'))
    output = tmp_path/'out'
    result = runner.run(plan, output)
    assert result['status'] == 'preflight_only'
    assert not output.exists()
    assert len(result['consumed_checkpoints']) == 2
    assert result['evaluated'] is False


@pytest.mark.parametrize('fault', ['wrong_path', 'changed_hash'])
def test_invalid_plan_never_launches_native_process(plan, tmp_path, monkeypatch, official_version, fault):
    if fault == 'wrong_path':
        plan['remote_checkpoint'] = bind(tmp_path/'unused.pth', b'valid but unused')
    else:
        Path(plan['remote_checkpoint']['path']).write_bytes(b'changed')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: pytest.fail('invalid plan executed'))
    with pytest.raises(ValueError):
        runner.run(plan, tmp_path/'out', execute=True)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('change_after_launch', [False, True])
def test_process_receipt_rechecks_both_consumed_weights(plan, tmp_path, monkeypatch, official_version,
                                                     change_after_launch):
    def process(argv, **kwargs):
        from tools.event_track_v2x.public_native_outputs import DMSTRACK_RUN
        run = output/'native-results'/DMSTRACK_RUN
        (run/'data_0').mkdir(parents=True)
        for index in range(9):
            (run/'data_0'/f'{index:04d}.txt').write_text('')
        (run/'summary_car_average_eval3D.txt').write_text(' sAMOTA  AMOTA  AMOTP\n0.5 0.4 0.6\n')
        if change_after_launch:
            Path(plan['remote_checkpoint']['path']).write_bytes(b'changed while running')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(runner.subprocess, 'run', process)
    output = tmp_path/'out'
    result = runner.run(plan, output, execute=True)
    assert result['status'] == ('failed' if change_after_launch else 'native_process_completed')
    assert result['consumed_checkpoints_unchanged'] is (not change_after_launch)
    assert result['official_results_verified'] is False and result['evaluated'] is False
    assert json.loads((output/'process-receipt.json').read_text()) == result


@pytest.mark.parametrize('method', ['CoopTrack', 'SparseCoop'])
def test_single_checkpoint_methods_do_not_invent_second_network(plan, method):
    plan['method'] = method
    del plan['remote_checkpoint']
    assert set(runner.consumed_checkpoints(plan)) == {'checkpoint'}


def test_binding_matches_expression_in_pinned_official_source(plan):
    source = os.environ.get('RBF_DMSTRACK_OFFICIAL_MAIN')
    if source is None:
        pytest.skip('provide the independently downloaded pinned official source')
    data = Path(source).read_bytes()
    assert hashlib.sha256(data).hexdigest() == '4a78f056d364795091e3a2b4f609a2b07a874874ff428aef1e56dde6ff904da6'
    tree = ast.parse(data)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'main_per_cat_multi_sensor_differentiable_kalman_filter')
    assignments = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == 'model_file']
    assert len(assignments) == 1
    expression = assignments[0].value
    assert ast.dump(expression) == ast.dump(ast.parse("load_model_path.replace('ego', cav_id)", mode='eval').body)
    # Evaluate only the verified string expression, not the downloaded program.
    expression = compile(ast.Expression(expression), '<pinned-native-path-expression>', 'eval')
    consumed = runner.consumed_checkpoints(plan)
    for cav, role in [('ego', 'checkpoint'), ('1', 'remote_checkpoint')]:
        path = eval(expression, {'__builtins__': {}},
                    dict(load_model_path=plan['checkpoint']['path'], cav_id=cav))
        assert path == consumed[role]['path']
