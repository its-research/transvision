#!/usr/bin/env python3
"""Freeze completed canonical OOF detector bytes; tensor/forward acceptance is separate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from run_cooptrack_official_oof_gpu4_offline_gl_v6 import (
    download_registered_artifact, validate_cohort,
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from clearml import Task
    task = Task.get_task(task_id=args.task_id)
    require(task.status == 'completed', 'task must be completed before byte freezing')
    manifest_path = args.package / 'package-manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    validate_cohort(manifest)
    admitted = json.loads((args.package / 'clearml-independent-readback.json').read_bytes())
    params = task.get_parameters()
    require(admitted['status'] == 'independent_bytes_verified'
            and admitted['manifest_sha256'] == sha(manifest_path)
            and admitted['fold_id'] == manifest['fold_id'], 'package admission mismatch')
    require(params.get('General/package_manifest_sha256') == sha(manifest_path)
            and params.get('General/package_task_id') == admitted['task_id']
            and params.get('General/package_independent_readback_sha256')
            == sha(args.package / 'clearml-independent-readback.json')
            and int(params['General/fold_id']) == manifest['fold_id']
            and int(params['General/epochs_per_side']) == 24
            and int(params['General/world_size']) == 4, 'training/package configuration mismatch')
    seed = int(params['General/seed'])
    require(seed in (1337, 2027, 3407), 'unexpected seed')
    controller = Path(__file__).with_name('run_cooptrack_official_oof_gpu4_offline_gl_v6.py')
    require(params['General/controller_sha256'] == sha(controller)
            and hashlib.sha256(task.data.script.diff.encode()).hexdigest() == sha(controller),
            'task controller differs from admitted local controller')
    names = ['batch-selection']
    for side in ('vehicle-side', 'infrastructure-side'):
        names += [side + suffix for suffix in (
            '-detector.py', '-launch-receipt.json', '-optimizer-startup.json',
            '-completion', '-final-checkpoint')]
    require(all(n in task.artifacts for n in names), 'completed task lacks required artifacts')
    require(not args.output.exists() and not args.output.is_symlink(), 'output is create-once')
    args.output.mkdir(parents=True)
    artifacts = {}
    # Read metadata/configs before downloading large checkpoint payloads.
    for name in [n for n in names if not n.endswith('-final-checkpoint')]:
        a = task.artifacts[name]
        filename = name if name.endswith(('.py', '.json')) else name + '.json'
        path = args.output / filename
        download_registered_artifact(a, a.hash, a.size, path)
        artifacts[name] = {'path': filename, 'sha256': sha(path), 'bytes': path.stat().st_size}
    selection = json.loads((args.output / 'batch-selection.json').read_bytes())
    require(selection['fold_id'] == manifest['fold_id'] and selection['seed'] == seed
            and selection['world_size'] == 4 and selection['base_learning_rate_unchanged']
            and selection['probe_weights_not_used_for_training'], 'batch identity mismatch')
    batch = selection['batch_per_gpu']
    require(batch in (2, 4, 8, 10) and batch * 4 <= len(manifest['fit_sequence_ids'])
            and selection['effective_batch_size'] == batch * 4
            and selection['sequence_stream_limit'] == len(manifest['fit_sequence_ids'])
            and selection['maximum_memory_fraction'] == 0.8, 'batch admission mismatch')
    profiles = selection['vehicle_profiles'] + [selection['infrastructure_profile']]
    selected_profiles = [p for p in profiles if p['batch_per_gpu'] == batch]
    require({p['side'] for p in selected_profiles} == {'vehicle-side', 'infrastructure-side'},
            'selected batch lacks both side probes')
    for profile in selected_profiles:
        require(profile['success'] and profile['returncode'] == 0
                and profile['batch_probe_only'] and profile['world_size'] == 4
                and sorted(r['rank'] for r in profile['ranks']) == list(range(4)),
                'selected batch probe not accepted')
        for rank in profile['ranks']:
            require(rank['iterations'] == 32 and rank['batch_probe_only']
                    and rank['world_size'] == 4 and rank['batch_per_gpu'] == batch
                    and 0 < rank['peak_reserved_bytes'] <= 0.8 * rank['total_memory_bytes'],
                    'rank probe exceeds memory or iteration boundary')
    sides = {}
    for side in ('vehicle-side', 'infrastructure-side'):
        launch = json.loads((args.output / (side + '-launch-receipt.json')).read_bytes())
        startup = json.loads((args.output / (side + '-optimizer-startup.json')).read_bytes())
        completion = json.loads((args.output / (side + '-completion.json')).read_bytes())
        require(launch['fit_sequence_ids'] == manifest['fit_sequence_ids']
                and not set(launch['fit_sequence_ids']) & set(manifest['held_out_sequence_ids'])
                and launch['fold_id'] == manifest['fold_id'] and launch['seed'] == seed
                and launch['side'] == side and launch['epochs'] == 24
                and launch['world_size'] == 4 and launch['batch_per_gpu'] == batch
                and launch['effective_batch_size'] == batch * 4
                and not launch['batch_probe_only'] and not launch['official_val_test_loaded']
                and not launch['paper_ranking_eligible'] and not launch['raw_labels_modified']
                and launch['pretrained_kind'] == 'ImageNet-R50-only-no-SPD-trained-weights',
                'launch split/training boundary mismatch')
        pretrained = next(r for r in manifest['inventory']
                          if r['path'] == 'resnet50-0676ba61.pth')
        require(launch['pretrained_sha256'] == pretrained['sha256'],
                'pretrained bytes differ from admitted package')
        require(startup['micro_iteration'] == 16 and not startup['batch_probe_only']
                and math.isfinite(startup['loss']) and startup['backbone_max_abs_update'] > 0
                and startup['optimizer_state_entries'] > 0
                and startup['resolved_config_sha256'] == artifacts[side + '-detector.py']['sha256']
                and startup['batch_per_gpu'] == batch and startup['world_size'] == 4,
                'optimizer/config identity mismatch')
        expected_iterations = launch['max_micro_iterations']
        require(isinstance(expected_iterations, int) and expected_iterations > 16
                and completion['checkpoint'] == f'iter_{expected_iterations}.pth'
                and completion['epochs'] == 24 and completion['side'] == side
                and completion['fold_id'] == manifest['fold_id'] and completion['seed'] == seed
                and completion['train_sequences'] == len(manifest['fit_sequence_ids'])
                and completion['cohort'] == manifest['cohort']
                and completion['batch_per_gpu'] == batch
                and completion['effective_batch_size'] == batch * 4
                and not completion['official_val_result_available'], 'completion identity mismatch')
        a = task.artifacts[side + '-final-checkpoint']
        require(a.hash == completion['sha256'] and a.size > 0, 'final checkpoint registration mismatch')
        path = args.output / (side + '-' + completion['checkpoint'])
        download_registered_artifact(a, completion['sha256'], a.size, path)
        artifacts[side + '-final-checkpoint'] = {
            'path': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size}
        sides[side] = {'expected_iterations': expected_iterations,
                       'checkpoint': path.name, 'actual_devices': completion['actual_cuda_devices']}
    task.reload()
    require(task.status == 'completed' and all(
        task.artifacts[n].hash == r['sha256'] and task.artifacts[n].size == r['bytes']
        for n, r in artifacts.items()), 'task/artifact changed during independent readback')
    receipt = {'kind': 'spd_canonical_oof_detector_independent_byte_freeze_v1',
               'task_id': task.id, 'fold_id': manifest['fold_id'], 'seed': seed,
               'package_manifest_sha256': sha(manifest_path),
               'package_independent_readback_sha256': sha(args.package / 'clearml-independent-readback.json'),
               'controller_sha256': sha(controller), 'freezer_sha256': sha(Path(__file__)),
               'artifacts': artifacts, 'sides': sides, 'byte_freeze_accepted': True,
               'checkpoint_tensors_or_forward_accepted': False,
               'heldout_inference_complete': False, 'formal_paper_eligible': False,
               'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    with (args.output / 'acceptance-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('OOF_DETECTOR_BYTES_FROZEN', sha(args.output / 'acceptance-receipt.json'), flush=True)


if __name__ == '__main__':
    main()
