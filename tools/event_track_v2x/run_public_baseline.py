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


def command(plan, output):
    """Explicit adapters to verified official entrypoints, never
    substitutes."""
    name = plan['method']
    source = Path(plan['source']).absolute()
    python = plan['python']
    checkpoint = str(Path(plan['checkpoint']['path']).absolute())
    if name == 'CoopTrack':
        return source, [python, str(source / 'tools/test.py'), str(Path(plan['configuration']['path']).absolute()), checkpoint, '--eval', 'bbox', '--show-dir', str(output)]
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
    cwd, argv = command(plan, output)
    result = dict(
        kind='rbf_public_native_invocation_v1', method=plan['method'], source_commit=record['commit'], argv=argv, cwd=str(cwd), original_protocol=True, table=3, evaluated=False)
    if not execute:
        return dict(result, status='preflight_only')
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new isolated output required')
    output.mkdir()
    (output / 'plan.json').write_bytes(canonical(dict(plan, invocation=result)))
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
    with (output / 'process.log').open('xb') as log:
        completed = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    result.update(
        status='native_process_completed' if completed.returncode == 0 else 'failed',
        returncode=completed.returncode,
        log_sha256=sha_file(output / 'process.log'),
        official_results_verified=False,
        source_unchanged=git('rev-parse', 'HEAD') == record['commit'] and not git('diff', 'HEAD', '--'))
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
