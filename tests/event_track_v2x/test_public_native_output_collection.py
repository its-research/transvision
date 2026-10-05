"""Native files are required independently of process exit status."""
import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.event_track_v2x import public_native_outputs as native
from tools.event_track_v2x import run_public_baseline as runner

ROW = '0 1 Car 0 0 0 -1 -1 -1 -1 1.5 2 4 0 1 10 0 .8\n'


@pytest.fixture
def dmstrack(tmp_path):
    config = tmp_path/'source/DMSTrack/configs/v2v4real.yml'
    config.parent.mkdir(parents=True)
    config.write_bytes((Path(__file__).parent/'fixtures/dmstrack-v2v4real-d3b9949.yml').read_bytes())
    contract = native.contract(dict(method='DMSTrack', source=str(tmp_path/'source')), tmp_path/'out')
    run = Path(contract['root'])/contract['run']
    (run/'data_0').mkdir(parents=True)
    for i in range(9):
        (run/'data_0'/f'{i:04d}.txt').write_text(ROW if i == 0 else '')
    (run/'summary_car_average_eval3D.txt').write_text('heading\n sAMOTA  AMOTA  AMOTP\n0.71 0.62 0.83\n')
    return contract, run


def test_native_complete_files_and_summary_are_not_independent_metrics(dmstrack):
    contract, run = dmstrack
    result = native.collect(contract)
    assert result['reported_metrics'] == dict(sAMOTA=.71, AMOTA=.62, AMOTP=.83)
    assert len(result['files']) == 10
    assert result['sequence_output_schema']['0001.txt']['rows'] == 0
    assert sum(x['declared_frames'] for x in result['sequence_output_schema'].values()) == 1993
    assert result['official_metrics_verified'] is False
    assert result['frame_coverage_independently_verified'] is False


@pytest.mark.parametrize('fault', ['absent', 'extra_sequence', 'missing_sequence', 'wrong_class',
    'nan', 'negative_dimension', 'duplicate', 'extra_field', 'past_frame', 'linked',
    'summary_absent', 'summary_nan', 'summary_duplicate', 'config_changed'])
def test_missing_or_invalid_outputs_rejected(dmstrack, fault):
    contract, run = dmstrack
    raw = run/'data_0/0000.txt'
    summary = run/'summary_car_average_eval3D.txt'
    if fault == 'absent':
        raw.unlink(); (run/'data_0').rename(run/'wrong')
    elif fault == 'extra_sequence':
        (run/'data_0/0009.txt').write_text('')
    elif fault == 'missing_sequence': raw.unlink()
    elif fault == 'wrong_class': raw.write_text(ROW.replace('Car', 'Van'))
    elif fault == 'nan': raw.write_text(ROW.replace('.8', 'nan'))
    elif fault == 'negative_dimension': raw.write_text(ROW.replace('1.5', '-1.5'))
    elif fault == 'duplicate': raw.write_text(ROW*2)
    elif fault == 'extra_field': raw.write_text(ROW.strip()+' 99\n')
    elif fault == 'past_frame': raw.write_text('147'+ROW[1:])
    elif fault == 'linked':
        (run/'other.txt').write_text(ROW); raw.unlink(); raw.symlink_to(run/'other.txt')
    elif fault == 'summary_absent': summary.unlink()
    elif fault == 'summary_nan': summary.write_text('sAMOTA AMOTA AMOTP\nnan .2 .3\n')
    elif fault == 'summary_duplicate': summary.write_text(summary.read_text()*2)
    else: Path(contract['configuration']['path']).write_text('changed')
    with pytest.raises((ValueError, OSError)): native.collect(contract)


def test_mutation_during_output_read_is_rejected(dmstrack, monkeypatch):
    contract, run = dmstrack
    original = native.read_summary
    def changing(path):
        result = original(path)
        (run/'data_0/0000.txt').write_text('')
        return result
    monkeypatch.setattr(native, 'read_summary', changing)
    with pytest.raises(ValueError, match='changed'): native.collect(contract)


@pytest.fixture
def cooptrack(tmp_path):
    plan = dict(method='CoopTrack', source=str(tmp_path/'source'), python='/python',
                configuration=dict(path=str(tmp_path/'source/projects/configs/model.py')),
                checkpoint=dict(path=str(tmp_path/'weights.pth')))
    output = tmp_path/'out'
    output.mkdir()
    contract = native.contract(plan, output)
    return plan, output, contract


def test_cooptrack_command_preserves_actual_raw_output(cooptrack):
    plan, output, contract = cooptrack
    _, argv = runner.command(plan, output)
    assert argv[argv.index('--out')+1] == str(output/'native-output.pkl')
    assert contract['evaluation_root'] == str(Path(plan['source'])/'test/model')
    native.validate_destination(contract)


def test_cooptrack_collects_native_files_without_executing_pickle(cooptrack):
    _, output, contract = cooptrack
    (output/'native-output.pkl').write_bytes(b'not a pickle')
    run = Path(contract['evaluation_root'])/'time'
    run.mkdir(parents=True)
    (run/'results_nusc.json').write_text('{}')
    result = native.collect(contract)
    assert len(result['files']) == 1 and result['unpickled'] is False
    with pytest.raises(ValueError, match='isolated'): native.validate_destination(contract)


@pytest.mark.parametrize('target', ['raw', 'evaluation', 'extra_run'])
def test_cooptrack_mutation_during_read_rejected(cooptrack, monkeypatch, target):
    _, output, contract = cooptrack
    raw = output/'native-output.pkl'; raw.write_bytes(b'raw')
    root = Path(contract['evaluation_root']); run = root/'time'; run.mkdir(parents=True)
    result_path = run/'results_nusc.json'; result_path.write_text('{}')
    original = native.tree_files
    changed = False
    def changing(path):
        nonlocal changed
        result = original(path)
        if not changed:
            changed = True
            if target == 'raw': raw.write_bytes(b'changed')
            elif target == 'evaluation': result_path.write_text('{"changed":true}')
            else: (root/'second').mkdir()
        return result
    monkeypatch.setattr(native,'tree_files',changing)
    with pytest.raises(ValueError,match='changed'): native.collect(contract)


@pytest.mark.parametrize('fault', ['missing_raw', 'missing_evaluation', 'multiple', 'link', 'empty'])
def test_cooptrack_does_not_accept_a_successful_empty_process(cooptrack, fault):
    _, output, contract = cooptrack
    raw = output/'native-output.pkl'; raw.write_bytes(b'raw')
    root = Path(contract['evaluation_root']); run = root/'time'; run.mkdir(parents=True)
    (run/'output.json').write_text('{}')
    if fault == 'missing_raw': raw.unlink()
    elif fault == 'missing_evaluation': run.rename(output/'elsewhere'); root.rmdir()
    elif fault == 'multiple': (root/'second').mkdir()
    elif fault == 'empty': (run/'output.json').unlink()
    else: (run/'linked').symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError): native.collect(contract)


def test_contract_paths_match_fixed_public_sources():
    paths = {key:os.environ.get(key) for key in ('RBF_DMSTRACK_OFFICIAL_MAIN','RBF_COOPTRACK_OFFICIAL_TEST')}
    if not all(paths.values()): pytest.skip('fixed public source snapshots required')
    dms = Path(paths['RBF_DMSTRACK_OFFICIAL_MAIN']).read_bytes()
    coop = Path(paths['RBF_COOPTRACK_OFFICIAL_TEST']).read_bytes()
    assert hashlib.sha256(dms).hexdigest() == '4a78f056d364795091e3a2b4f609a2b07a874874ff428aef1e56dde6ff904da6'
    assert hashlib.sha256(coop).hexdigest() == '709a1e682913a1b765c6e34f7a087808b36fb164ab29aba43335560e643689d5'
    dtree, ctree = ast.parse(dms), ast.parse(coop)
    fn = next(n for n in dtree.body if isinstance(n, ast.FunctionDef) and n.name == 'track_and_evaluate')
    assignments = [n for n in fn.body[1].body if isinstance(n, ast.Assign)]
    expected = ast.parse("evaluation_save_folder = 'evaluation_' + evaluation_config_dict['result_sha'] + '_%s' % seq_eval_mode + '_H%d' % cfg.num_hypo + '_epoch_%d' % epoch_idx").body[0]
    assert any(ast.dump(n) == ast.dump(expected) for n in assignments)
    expected = ast.parse("kwargs['jsonfile_prefix'] = osp.join('test', args.config.split('/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))").body[0]
    assert any(ast.dump(n) == ast.dump(expected) for n in ast.walk(ctree))


def test_parser_consumes_unmodified_native_save_results(dmstrack):
    import io
    source = os.environ.get('RBF_DMSTRACK_OFFICIAL_IO')
    if source is None: pytest.skip('fixed public native writer source required')
    data = Path(source).read_bytes()
    assert hashlib.sha256(data).hexdigest() == 'd65826c341c264873d2d12267863b5f260714987b015b94d572833b1d5a5a7ce'
    tree = ast.parse(data)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'save_results')
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<fixed-native-writer>', 'exec'), scope)
    visual, tracking = io.StringIO(), io.StringIO()
    scope['save_results']([1.5, 2., 4., 0., 1., 10., .1, 3, 0., 2, 0., 0., 0., 0., .81],
                          visual, tracking, {2:'Car'}, 6, -10000)
    contract, run = dmstrack
    (run/'data_0/0000.txt').write_text(tracking.getvalue())
    result = native.collect(contract)
    assert result['sequence_output_schema']['0000.txt']['rows'] == 1


def test_source_change_after_execution_is_failure(tmp_path, monkeypatch):
    # No GPU needed: exercise orchestration with bound native files.
    from tests.event_track_v2x.test_public_native_checkpoints import bind
    source = tmp_path/'source'
    plan = dict(method='DMSTrack', source=str(source), python='/python', table=3, original_protocol=True)
    plan['checkpoint'] = bind(tmp_path/'model_ego.pth', b'ego')
    plan['remote_checkpoint'] = bind(tmp_path/'model_1.pth', b'remote')
    for key in ('dataset_manifest','split_mapping','environment_lock'): plan[key] = bind(tmp_path/(key+'.json'),b'{}')
    bind(source/'DMSTrack/configs/v2v4real.yml',(Path(__file__).parent/'fixtures/dmstrack-v2v4real-d3b9949.yml').read_bytes())
    state = {'ran':False}
    monkeypatch.setattr(runner.subprocess,'check_output', lambda argv,**kw:
                        ('d3b9949499c8e68ea33060873bd1cb95b6d4d323' if 'rev-parse' in argv else ('diff' if state['ran'] else '')))
    def process(*args,**kwargs):
        import subprocess
        state['ran']=True
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(runner.subprocess,'run',process)
    monkeypatch.setattr(runner,'collect_native_outputs',lambda c: {'test_only':True})
    result = runner.run(plan,tmp_path/'out',execute=True)
    assert result['status']=='failed' and not result['source_unchanged']
