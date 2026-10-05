"""Pinned SparseCoop output-path behavior, without importing its GPU runtime."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools.event_track_v2x import run_public_baseline as runner


def bind(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    plan = dict(method='SparseCoop', table=3, original_protocol=True, python='/test/python',
                source=str(tmp_path/'source'))
    plan['configuration'] = bind(tmp_path/'source/projects/configs/cooperative.py', b'# native config\n')
    plan['checkpoint'] = bind(tmp_path/'weights/epoch24.pth', b'frozen network')
    for key in ('dataset_manifest', 'split_mapping', 'environment_lock'):
        plan[key] = bind(tmp_path/(key+'.json'), b'{}')
    def git(argv, **kwargs):
        assert argv[:2] == ['git', '-C']
        if argv[3:] == ['rev-parse', 'HEAD']:
            return '1f5741ccc62be8d8d5bedd296615b75b4c10fc7e\n'
        assert argv[3:] == ['diff', 'HEAD', '--']
        return ''
    monkeypatch.setattr(runner.subprocess, 'check_output', git)
    output = tmp_path/'source/work_dirs/cooperative'
    output.parent.mkdir(parents=True)
    return plan, output


def test_preflight_records_actual_root_and_does_not_execute(inputs, monkeypatch):
    plan, output = inputs
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: pytest.fail('preflight executed'))
    result = runner.run(plan, output)
    assert result['status'] == 'preflight_only' and not output.exists()
    contract = result['native_output_contract']
    assert contract['root'] == str(output)
    assert contract['raw_result_pattern'] == 'test/*/results.pkl'
    assert contract['out_argument_used_as_filename'] is False
    assert contract['native_code_changed'] is False


def test_arbitrary_out_flag_cannot_redirect_native_output(inputs, tmp_path, monkeypatch):
    plan, _ = inputs
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: pytest.fail('unsafe path executed'))
    with pytest.raises(ValueError, match='configuration-derived'):
        runner.run(plan, tmp_path/'requested-but-unused', execute=True)
    assert not (tmp_path/'requested-but-unused').exists()


def test_success_collects_actual_files_without_unpickling(inputs, monkeypatch):
    plan, output = inputs
    def process(argv, **kwargs):
        run = output/'test/Mon_Oct__5_07_40_00_2026'
        run.mkdir(parents=True)
        (run/'results.pkl').write_bytes(b'opaque native payload; deliberately not a pickle')
        (run/'metrics_summary.json').write_bytes(b'{"metric": 0.1}')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(runner.subprocess, 'run', process)
    result = runner.run(plan, output, execute=True)
    assert result['status'] == 'native_process_completed'
    actual = result['native_outputs']
    assert actual['unpickled'] is actual['official_metrics_verified'] is False
    assert actual['output_semantics_verified'] is False
    assert len(actual['files']) == 2
    assert actual['raw_predictions'].endswith('/results.pkl')
    assert not (output/'native-output.pkl').exists()
    assert json.loads((output/'process-receipt.json').read_bytes()) == result


@pytest.mark.parametrize('fault', ['missing', 'flag_path_only', 'empty', 'multiple_runs',
                                 'linked_raw', 'linked_directory'])
def test_exit_zero_cannot_hide_absent_or_unisolated_outputs(inputs, monkeypatch, tmp_path, fault):
    plan, output = inputs
    def process(argv, **kwargs):
        if fault == 'missing':
            return subprocess.CompletedProcess(argv, 0)
        if fault == 'flag_path_only':
            (output/'native-output.pkl').write_bytes(b'wrong layout')
            return subprocess.CompletedProcess(argv, 0)
        run = output/'test/time'
        run.mkdir(parents=True)
        raw = run/'results.pkl'
        raw.write_bytes(b'' if fault == 'empty' else b'native result')
        if fault == 'multiple_runs':
            (output/'test/another_time').mkdir()
        elif fault == 'linked_raw':
            remote = tmp_path/'outside-result.pkl'; remote.write_bytes(b'outside')
            raw.unlink(); raw.symlink_to(remote)
        elif fault == 'linked_directory':
            (run/'outside').symlink_to(tmp_path, target_is_directory=True)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(runner.subprocess, 'run', process)
    result = runner.run(plan, output, execute=True)
    assert result['returncode'] == 0 and result['status'] == 'failed'
    assert result['native_output_collection_error']
    assert (output/'process-receipt.json').is_file()
    assert result['official_results_verified'] is False


def test_output_root_formula_matches_pinned_official_source(inputs):
    path = os.environ.get('RBF_SPARSECOOP_OFFICIAL_TEST')
    if path is None:
        pytest.skip('provide independently downloaded pinned official tools/test.py')
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == 'de129d2a91eccda1fd6850eaf0c89b855f00e103db0d6dded772eb57602741f9'
    tree = ast.parse(data)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "checkpoint_dir.startswith('work_dirs/')")
    expected = ast.parse("if checkpoint_dir.startswith('work_dirs/'):\n    work_dir = checkpoint_dir\nelse:\n    work_dir = args.config.replace('projects/configs/', 'work_dirs/').replace('.py', '')").body[0]
    assert ast.dump(branch) == ast.dump(expected)
    plan, output = inputs
    scope = dict(checkpoint_dir=str(Path(plan['checkpoint']['path']).absolute().parent),
                 args=SimpleNamespace(config=str(Path(plan['configuration']['path']).absolute())))
    # Only this audited string-path branch executes, not any native imports/model.
    exec(compile(ast.Module(body=[branch], type_ignores=[]), '<pinned-path-branch>', 'exec'),
         {'__builtins__': {}}, scope)
    assert scope['work_dir'] == runner.native_output_contract(plan, output)['root']
    dumps = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == 'mmcv.dump']
    assert len(dumps) == 1
    assert ast.unparse(dumps[0].args[1]) == "osp.join(val_dir, 'results.pkl')"
