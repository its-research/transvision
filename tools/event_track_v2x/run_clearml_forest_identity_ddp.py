#!/usr/bin/env python3
"""ClearML controller for one seed on exactly four allocated A100/5090 GPUs.

No CPU fallback, no V100, no test/val payload, no detector retraining. Source
and derived-row packages are hash-pinned. ClearML uploads only bounded training
receipts, logs and the newly trained identity checkpoint, never original GT or
full prediction streams. Nonzero torchrun exit cannot mean completed training.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import urlparse

FILES_HOST = 'http://10.100.35.118:8081'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def extract(source, destination, *, max_bytes):
    destination.mkdir()
    with tarfile.open(source, 'r:gz') as archive:
        seen, size = set(), 0
        for item in archive.getmembers():
            path = PurePosixPath(item.name)
            if (not item.isfile() or path.is_absolute() or '..' in path.parts or '\\' in item.name
                    or item.name in seen or len(seen) >= 1000):
                raise ValueError('unsafe or duplicate archive entry')
            seen.add(item.name); size += item.size
            if size > max_bytes:
                raise ValueError('archive exceeds predeclared size cap')
        archive.extractall(destination, filter='data')


def task_parameters(parameters):
    """ClearML set_parameters stores unqualified keys in the General section."""
    result = {}
    for key, value in parameters.items():
        key = key.removeprefix('General/')
        if key in result:
            raise ValueError('ambiguous duplicate task parameter')
        result[key] = value
    for key in ('package_task_id', 'package_sha256', 'seed', 'gpu_family'):
        if key not in result:
            raise ValueError('missing task parameter: '+key)
    if (int(result.get('required_world_size', 0)) != 4 or int(result.get('global_batch_size', 0)) != 64
            or int(result.get('epochs', 0)) != 10 or result.get('class_scope') != 'car'):
        raise ValueError('task hyperparameters differ from the fixed training contract')
    return result


def download_artifact(artifact):
    uri = urlparse(artifact.url)
    if (uri.scheme, uri.hostname, uri.port) != ('http', '10.100.35.118', 8081):
        raise ValueError('artifact must use the designated private file server')
    path = artifact.get_local_copy()
    if not path:
        raise RuntimeError('authenticated ClearML artifact download failed; no alternate credential or URL attempted')
    return Path(path)


def main():
    # Task-local official SDK configuration: keep authentication enabled. Some
    # workers advertise the server's other NIC (34.118), while artifact URLs
    # use the user-designated 35.118. Without this match SDK omits auth headers.
    os.environ['CLEARML_FILES_HOST'] = FILES_HOST
    from clearml import Task
    import torch

    task = Task.init(project_name='Thesis/Recover-Before-Fuse/Training', task_name='four-GPU identity',
                     reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False)
    params = task_parameters(task.get_parameters())
    package_id, expected = params['package_task_id'], params['package_sha256']
    seed, family = int(params['seed']), params['gpu_family']
    if seed not in (1337, 2027, 3407) or family not in ('A100', '5090'):
        raise ValueError('predeclared seed and non-V100 GPU family required')
    if torch.cuda.device_count() != 4 or not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
        raise ValueError('exactly four allocated CUDA GPUs with NCCL required')
    devices = []
    for i in range(4):
        name = torch.cuda.get_device_name(i)
        if family not in name:
            raise ValueError('worker device family differs from task assignment')
        with torch.cuda.device(i):
            x = torch.ones((128, 128), device='cuda')
            y = x @ x
            if not bool(torch.isfinite(y).all()) or float(y[0, 0]) != 128.:
                raise ValueError('assigned GPU compute preflight failed')
            torch.cuda.synchronize()
            devices.append(dict(index=i, name=name, free_bytes=torch.cuda.mem_get_info()[0]))
            del x, y
    print('RBF_GPU_PREFLIGHT '+json.dumps(dict(devices=devices, torch=torch.__version__,
          cuda=torch.version.cuda, architecture=torch.cuda.get_arch_list())), flush=True)
    package_task = Task.get_task(task_id=package_id)
    metadata_path = download_artifact(package_task.artifacts['package'])
    if sha(metadata_path) != expected:
        raise ValueError('package manifest changed')
    metadata = json.loads(metadata_path.read_bytes())
    if (metadata['kind'] != 'forest_identity_ddp_package_v1' or metadata['split'] != 'train'
            or metadata['class_scope'] != ['car'] or metadata['raw_GT_included'] is not False
            or metadata['full_official_train'] is not True):
        raise ValueError('full car-only derived train package required')
    root = Path(tempfile.mkdtemp(prefix='rbf-ddp-'))
    if shutil.disk_usage(root).free < 2*1024**3:
        raise ValueError('at least 2 GiB private runtime disk required')
    for record in metadata['artifacts']:
        name = record['path']
        if name not in ('source.tar.gz', 'train-rows.tar.gz'):
            raise ValueError('unexpected package artifact')
        source = download_artifact(package_task.artifacts[name])
        if source.stat().st_size != record['bytes'] or sha(source) != record['sha256']:
            raise ValueError('source/data archive identity differs')
        extract(source, root/('source' if name == 'source.tar.gz' else 'rows'),
                max_bytes=16*1024**2 if name == 'source.tar.gz' else 512*1024**2)
    for item in metadata['source_inventory']:
        path = PurePosixPath(item['path'])
        if path.is_absolute() or '..' in path.parts or sha(root/'source'/path) != item['sha256']:
            raise ValueError('unpacked source inventory differs')
    output = root/'fit'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    # Child ranks must not attach themselves as separate ClearML training jobs.
    env['CLEARML_AGENT_FORCE_TASK_INIT'] = '0'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
        'tools/event_track_v2x/train_forest_identity_ddp.py', '--data', str(root/'rows'),
        '--manifest-sha256', metadata['dataset_sha256'], '--output', str(output), '--seeds', str(seed),
        '--epochs', '10', '--global-batch-size', '64']
    print('RBF_DDP_LAUNCH '+json.dumps(dict(seed=seed, world_size=4, runtime=str(root), command=command)), flush=True)
    subprocess.run(command, cwd=root/'source', env=env, check=True)
    receipt = json.loads((output/'receipt.json').read_bytes())
    if (receipt['status'] != 'complete' or receipt['world_size'] != 4
            or [x['seed'] for x in receipt['seeds']] != [seed]):
        raise ValueError('incomplete four-rank training receipt')
    # Only this newly trained identity model and bounded metadata are retained.
    for name, path in [('training-receipt', output/'receipt.json'), ('training-plan', output/'plan.json'),
                       ('epoch-progress', output/f'seed-{seed}'/'epochs.jsonl'),
                       ('checkpoint-manifest', output/f'seed-{seed}'/'checkpoint.json'),
                       ('identity-checkpoint', output/f'seed-{seed}'/'weights.pt')]:
        if not task.upload_artifact(name, artifact_object=path, wait_on_upload=True):
            raise RuntimeError('training artifact upload failed')
        if Task.get_task(task_id=task.id).artifacts[name].hash != sha(path):
            raise RuntimeError('uploaded training artifact readback identity differs')
    print('RBF_DDP_TRAINING_COMPLETE '+json.dumps(dict(seed=seed, receipt_sha256=sha(output/'receipt.json'),
          checkpoint_artifact='identity-checkpoint',
          paper_eligible=False)), flush=True)
    task.close()


if __name__ == '__main__':
    main()
