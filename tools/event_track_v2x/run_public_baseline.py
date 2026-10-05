#!/usr/bin/env python3
"""Source-locked native official commands.

A successful process is not a metric receipt.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def consumed_checkpoints(plan):
    """Bind the paths the pinned native entrypoint actually opens.

    DMSTrack d3b9949 main_dkf.py loads both CAVs with
    load_model_path.replace('ego', cav_id). There is no remote-checkpoint CLI
    argument. A hash for an unrelated second file cannot bind that load.
    """
    from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
    ego = Path(plan['checkpoint']['path']).absolute()
    roles = {'checkpoint': ego}
    if plan['method'] == 'DMSTrack':
        remote = Path(str(ego).replace('ego', '1'))
        declared = Path(plan['remote_checkpoint']['path']).absolute()
        if remote == ego:
            raise ValueError('DMSTrack ego path must derive a distinct CAV 1 checkpoint')
        if declared != remote:
            raise ValueError('DMSTrack remote_checkpoint differs from native ego-to-1 path replacement')
        if ego.samefile(remote):
            raise ValueError('DMSTrack CAV checkpoints must not alias the same file')
        roles['remote_checkpoint'] = remote
    result = {}
    for role, path in roles.items():
        actual = sha_file(path)
        if actual != plan[role]['sha256']:
            raise ValueError(role + ' consumed by native entrypoint is missing or changed')
        result[role] = dict(path=str(path), sha256=actual, bytes=path.stat().st_size)
    return result


def native_output_contract(plan, output):
    """Bind SparseCoop's actual output root without changing official code.

    At the pinned revision tools/test.py treats --out as a boolean; it writes
    test/<time.ctime()>/results.pkl under a path derived from args.config.
    Our command always supplies absolute checkpoint/configuration paths, so
    its relative 'work_dirs/' checkpoint-directory branch cannot be selected.
    """
    if plan['method'] != 'SparseCoop':
        from tools.event_track_v2x.public_native_outputs import contract
        return contract(plan, output)
    config = str(Path(plan['configuration']['path']).absolute())
    root = Path(config.replace('projects/configs/', 'work_dirs/').replace('.py', '')).absolute()
    if root != Path(output).absolute():
        raise ValueError('SparseCoop --output must equal native configuration-derived output root: '+str(root))
    return dict(kind='sparsecoop_native_output_paths_v1', root=str(root),
                raw_result_pattern='test/*/results.pkl', out_argument_used_as_filename=False,
                checkpoint_path_is_absolute=True, native_code_changed=False)


def collect_native_outputs(contract):
    """Hash actual output files only; never unpickle or claim metric validity."""
    if contract is None:
        raise ValueError('native output contract required')
    if contract['kind'] != 'sparsecoop_native_output_paths_v1':
        from tools.event_track_v2x.public_native_outputs import collect
        return collect(contract)
    from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
    root = Path(contract['root'])
    test = root/'test'
    if not test.is_dir() or any(p.is_symlink() for p in (test, *test.parents)):
        raise ValueError('missing or linked SparseCoop native test output')
    runs = list(test.iterdir())
    if len(runs) != 1 or not runs[0].is_dir() or runs[0].is_symlink():
        raise ValueError('one isolated SparseCoop native timestamp directory required')
    raw = runs[0]/'results.pkl'
    if not raw.is_file() or raw.is_symlink() or raw.stat().st_size == 0:
        raise ValueError('SparseCoop native results.pkl missing, linked or empty')
    files = {}
    for directory, names, filenames in os.walk(runs[0], followlinks=False):
        directory = Path(directory)
        if any((directory/name).is_symlink() for name in names):
            raise ValueError('linked directory in native output')
        for name in sorted(filenames):
            path = directory/name
            if not path.is_file() or path.is_symlink():
                raise ValueError('native output is not an isolated regular file')
            files[str(path.relative_to(root))] = dict(sha256=sha_file(path), bytes=path.stat().st_size)
    return dict(root=str(root), files=files, raw_predictions=str(raw.relative_to(root)),
                unpickled=False, official_metrics_verified=False, output_semantics_verified=False)


def command(plan, output):
    """Explicit adapters to verified official entrypoints, never
    substitutes."""
    name = plan['method']
    source = Path(plan['source']).absolute()
    python = plan['python']
    checkpoint = str(Path(plan['checkpoint']['path']).absolute())
    if name == 'CoopTrack':
        return source, [python, str(source / 'tools/test.py'), str(Path(plan['configuration']['path']).absolute()), checkpoint, '--eval', 'bbox', '--show-dir', str(output), '--out', str(output/'native-output.pkl')]
    if name == 'SparseCoop':
        return source, [
            python,
            str(source / 'tools/test.py'),
            str(Path(plan['configuration']['path']).absolute()), checkpoint, '--eval', 'bbox', '--out',
            str(output / 'native-output.pkl')
        ]
    if name == 'DMSTrack':
        return source / 'DMSTrack', [
            python, 'main_dkf.py', '--dataset', 'v2v4real', '--det_name', 'multi_sensor_differentiable_kalman_filter', '--num_frames_backprop', '10', '--num_frames_per_sub_seq',
            '-1', '--num_epochs', '0', '--use_multiple_nets', '--seq_eval_mode', 'all', '--run_evaluation_every_epoch', '--training_split', 'train', '--evaluation_split', 'val',
            '--regression_loss_weight', '1', '--association_loss_weight', '0', '--det_neg_log_likelihood_loss_weight', '0', '--feature', 'fusion', '--clip_grad_norm', '1',
            '--load_model_path', checkpoint, '--save_dir_prefix',
            str(output / 'native-results')
        ]
    raise ValueError('no complete verified official adapter for ' + name)


def run(plan, output, *, execute=False):
    from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
    registry = json.loads((ROOT / 'configs/event_track_v2x/paper/public-baselines.json').read_text())
    record = next(r for r in registry['baselines'] if r['name'] == plan['method'])
    if 'commit' not in record:
        raise ValueError('algorithm implementation incomplete: ' + plan['method'])
    source = Path(plan['source']).absolute()
    output = Path(output).absolute()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()

    if git('rev-parse', 'HEAD') != record['commit'] or git('diff', 'HEAD', '--'):
        raise ValueError('official source version differs or tracked source edited')
    if plan['table'] != 3 or plan['original_protocol'] is not True:
        raise ValueError('native baselines are not unified-input mechanism results')
    required = ('dataset_manifest', 'checkpoint', 'split_mapping', 'environment_lock')
    for key in required + (('configuration', ) if plan['method'] != 'DMSTrack' else ('remote_checkpoint', )):
        binding = plan[key]
        if sha_file(binding['path']) != binding['sha256']:
            raise ValueError(key + ' missing or changed')
    checkpoints = consumed_checkpoints(plan)
    output_contract = native_output_contract(plan, output)
    cwd, argv = command(plan, output)
    result = dict(
        kind='rbf_public_native_invocation_v1', method=plan['method'], source_commit=record['commit'], argv=argv, cwd=str(cwd), original_protocol=True, table=3, evaluated=False,
        consumed_checkpoints=checkpoints,
        checkpoint_binding_recipe='native_entrypoint_paths_and_full_file_hashes_v1',
        native_output_contract=output_contract)
    if not execute:
        return dict(result, status='preflight_only')
    if plan['method'] != 'SparseCoop':
        from tools.event_track_v2x.public_native_outputs import validate_destination
        validate_destination(output_contract)
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new isolated output required')
    output.mkdir()
    (output / 'plan.json').write_bytes(canonical(dict(plan, invocation=result)))
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
    with (output / 'process.log').open('xb') as log:
        completed = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    checkpoint_error = None
    try:
        checkpoints_unchanged = consumed_checkpoints(plan) == checkpoints
    except (ValueError, OSError, KeyError) as error:
        checkpoints_unchanged = False
        checkpoint_error = str(error)
    native_outputs, output_error = None, None
    try:
        native_outputs = collect_native_outputs(output_contract)
    except (ValueError, OSError) as error:
        output_error = str(error)
    source_unchanged = git('rev-parse', 'HEAD') == record['commit'] and not git('diff', 'HEAD', '--')
    result.update(
        status='native_process_completed' if completed.returncode == 0 and checkpoints_unchanged and output_error is None and source_unchanged else 'failed',
        returncode=completed.returncode,
        consumed_checkpoints_unchanged=checkpoints_unchanged,
        checkpoint_binding_error=checkpoint_error,
        native_outputs=native_outputs,
        native_output_collection_error=output_error,
        log_sha256=sha_file(output / 'process.log'),
        official_results_verified=False,
        source_unchanged=source_unchanged)
    (output / 'process-receipt.json').write_bytes(canonical(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(json.loads(Path(a.plan).read_text()), a.output, execute=a.execute), indent=2))


if __name__ == '__main__':
    main()
