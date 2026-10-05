#!/usr/bin/env python3
"""Record Linux CPU runtime identity for the frozen SPD MHT contract.

This probe reads no dataset, checkpoint, or ground truth and computes no metric.
It does not authorize a change to the frozen MHT runtime binding.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3

THREAD_ENV = {
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1', 'BLIS_NUM_THREADS': '1',
    'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1',
}
for key, value in THREAD_ENV.items():
    if os.environ.get(key) != value:
        raise RuntimeError(f'fresh-process thread environment differs: {key}')
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from clearml import Task
    task = Task.init(project_name='Thesis/Recover-Before-Fuse/Training',
                     task_name='SPD MHT L40S CPU runtime identity probe',
                     auto_connect_frameworks=False,
                     auto_connect_arg_parser=False)
    task.reload()
    worker = task.data.last_worker or ''
    if not worker.startswith('10.100.35.121-L40S:'):
        raise RuntimeError('runtime probe must execute on L40S worker')
    import numpy
    import scipy
    from scipy.optimize import _lsap
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only runtime probe initialized CUDA')
    row = {
        'kind': 'spd_mht_l40s_linux_cpu_runtime_identity_probe_v1',
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'task_id': task.id,
        'worker_id': worker,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': platform.python_version(),
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'torch': str(torch.__version__),
        'sqlite_version': sqlite3.sqlite_version,
        'assignment_binary_sha256': digest(_lsap.__file__),
        'assignment_binary_name': Path(_lsap.__file__).name,
        'thread_environment': THREAD_ENV,
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'device': 'cpu',
        'dataset_read': False,
        'checkpoint_read': False,
        'ground_truth_read': False,
        'inference_started': False,
        'metrics_computed': False,
        'frozen_contract_changed': False,
        'paper_eligible': False,
    }
    path = Path('/tmp/spd-mht-l40-runtime-probe.json')
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
    task.upload_artifact('runtime-probe', artifact_object=str(path), wait_on_upload=True)
    print(json.dumps({k: row[k] for k in ('task_id', 'worker_id', 'platform',
         'numpy', 'scipy', 'torch', 'sqlite_version', 'assignment_binary_sha256',
         'inference_started')}, sort_keys=True), flush=True)
    task.close()


if __name__ == '__main__':
    main()
