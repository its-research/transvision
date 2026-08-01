#!/usr/bin/env python3
"""Exercise CUDA compute and NCCL on every GPU assigned by ClearML."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any


def _worker() -> int:
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor(float(rank), device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = float(world_size * (world_size - 1) // 2)
    if value.item() != expected:
        raise RuntimeError(
            f"NCCL all-reduce mismatch on rank {rank}: {value.item()} != {expected}"
        )
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "event": "nccl_rank_pass",
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "value": value.item(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.destroy_process_group()
    return 0


def _probe() -> int:
    import torch

    expected_gpus = 4
    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise RuntimeError(f"expected {expected_gpus} visible GPUs, found {count}")

    result: dict[str, Any] = {
        "event": "rtx5090_probe",
        "python": sys.version.split()[0],
        "libc": platform.libc_ver(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": count,
        "cuda_arch_list": torch.cuda.get_arch_list(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "devices": [],
    }
    if not result["cuda_available"]:
        raise RuntimeError("torch.cuda.is_available() is false")
    if not result["nccl_available"]:
        raise RuntimeError("PyTorch NCCL backend is unavailable")

    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        device = torch.device(f"cuda:{index}")
        left = torch.arange(
            1024 * 1024,
            dtype=torch.float32,
            device=device,
        ).reshape(1024, 1024)
        product = left @ left.transpose(0, 1)
        checksum = product[0, 0].item()
        if not torch.isfinite(product).all().item():
            raise RuntimeError(f"non-finite CUDA matmul result on device {index}")
        torch.cuda.synchronize(index)
        result["devices"].append(
            {
                "index": index,
                "name": properties.name,
                "capability": [
                    properties.major,
                    properties.minor,
                ],
                "total_memory": properties.total_memory,
                "matmul_checksum": checksum,
            }
        )

    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_result = subprocess.run(
            [nvcc, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        result["nvcc"] = nvcc_result.stdout.strip().splitlines()[-1]
    else:
        result["nvcc"] = None

    print(json.dumps(result, sort_keys=True), flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={expected_gpus}",
            str(__file__),
            "--distributed-worker",
        ],
        check=True,
    )
    print("RESILIENT_V2X_5090_PROBE_PASS", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed-worker", action="store_true")
    args = parser.parse_args()
    return _worker() if args.distributed_worker else _probe()


if __name__ == "__main__":
    raise SystemExit(main())
