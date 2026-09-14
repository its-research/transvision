#!/usr/bin/env python3
"""Bounded four-A100 dependency inventory; no dataset, checkpoint or training.

Missing imports are diagnostic results, not permission to install packages.
A successful probe never certifies detector-forward or DDP readiness.
"""
from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys

PROJECT = "Thesis/Recover-Before-Fuse/Training"
MODULES = (
    ("numpy", "numpy"), ("scipy", "scipy"), ("numba", "numba"),
    ("yaml", "PyYAML"), ("open3d", "open3d"), ("spconv.pytorch", None),
    ("cumm", None), ("shapely", "shapely"), ("einops", "einops"),
    ("timm", "timm"), ("tensorboardX", "tensorboardX"),
    ("cv2", None), ("skimage", "scikit-image"), ("torchvision", "torchvision"),
    ("opencood", None),
)


def validate_devices(names):
    if len(names) != 4 or any(not isinstance(n, str) or "A100" not in n for n in names):
        raise ValueError("exactly four worker-assigned A100 devices required")


def inspect_import(module, distribution, *, run=subprocess.run):
    if (module, distribution) not in MODULES:
        raise ValueError("only predeclared runtime modules may be inspected")
    version = None
    if distribution is not None:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    code = (
        "import importlib,json\n"
        "try:\n"
        f" importlib.import_module({module!r})\n"
        " result={'importable':True,'error_type':None}\n"
        "except Exception as error:\n"
        " result={'importable':False,'error_type':type(error).__name__}\n"
        "print('RBF_IMPORT_PROBE '+json.dumps(result))\n"
    )
    try:
        done = run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        result = dict(importable=False, error_type="ImportTimeout")
    else:
        lines = [s.removeprefix("RBF_IMPORT_PROBE ") for s in done.stdout.splitlines()
                 if s.startswith("RBF_IMPORT_PROBE ")]
        if done.returncode or len(lines) != 1:
            result = dict(importable=False, error_type="ImportProcessFailure")
        else:
            result = json.loads(lines[0])
    return dict(module=module, distribution=distribution, version=version, **result)


def collect(torch):
    count = torch.cuda.device_count()
    if count != 4:
        raise ValueError("exactly four worker-assigned A100 devices required")
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    validate_devices(names)  # Reject the entire allocation before tensor work.
    devices = []
    for i, name in enumerate(names):
        with torch.cuda.device(i):
            left = torch.ones((64, 64), device=f"cuda:{i}")
            result = left @ left
            torch.cuda.synchronize()
            total = float(result.sum().item())
            if total != 262144.0:
                raise RuntimeError("bounded CUDA matrix check failed")
            free, capacity = torch.cuda.mem_get_info()
            devices.append(dict(index=i, name=name, free_bytes=free, total_bytes=capacity,
                                cuda_matrix_sum=total))
            del left, result
    modules = []
    for module, distribution in MODULES:
        row = inspect_import(module, distribution)
        modules.append(row)
        print("RBF_V2V4REAL_DEPENDENCY " + json.dumps(row, sort_keys=True), flush=True)
    return dict(kind="v2v4real_four_a100_runtime_inventory_v1", devices=devices,
                python=sys.version.split()[0], torch=str(torch.__version__), cuda=torch.version.cuda,
                modules=modules, nvcc_available=shutil.which("nvcc") is not None,
                disk_free_bytes=shutil.disk_usage(".").free,
                all_predeclared_imports_passed=all(row["importable"] for row in modules),
                gpu_assignment_verified=True, dataset_read=False, checkpoint_loaded=False,
                detector_forward_verified=False, ddp_verified=False, parameter_training=False,
                packages_installed_by_probe=False, paper_eligible=False)


def main():
    from clearml import Task
    import torch
    task = Task.init(project_name=PROJECT, task_name="V2V4Real four-A100 runtime inventory",
                     reuse_last_task_id=False, auto_connect_frameworks=False,
                     auto_connect_arg_parser=False)
    report = collect(torch)
    print("RBF_V2V4REAL_A100_RUNTIME " + json.dumps(report, sort_keys=True), flush=True)
    task.close()


if __name__ == "__main__":
    main()
