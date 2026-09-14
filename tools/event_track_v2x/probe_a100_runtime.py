#!/usr/bin/env python3
"""Read only the allocated A100 devices and runtime prerequisites."""

import json
from pathlib import Path
import shutil
import sys

from clearml import Task
import torch


def main():
    task = Task.init(project_name="Thesis/EventTrack-V2X/Training",
                     task_name="A100 runtime readiness", reuse_last_task_id=False,
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    count = torch.cuda.device_count()
    if count != 4:
        raise RuntimeError("expected exactly four worker-assigned GPUs")
    devices = []
    for index in range(count):
        name = torch.cuda.get_device_name(index)
        if "A100" not in name:
            raise RuntimeError("non-A100 GPU assignment rejected")
        with torch.cuda.device(index):
            tensor = torch.ones(1, device="cuda")
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            devices.append({"index": index, "name": name, "free_bytes": free,
                            "total_bytes": total, "tensor_value": float(tensor.item())})
            del tensor
    report = {"kind": "eventtrack_a100_runtime_readiness_v1", "devices": devices,
              "python": sys.version.split()[0], "torch": torch.__version__,
              "cuda": torch.version.cuda, "disk_free_bytes": shutil.disk_usage(".").free,
              "portable_prefix_exists": Path("/opt/cooptrack").exists(),
              "mmdet3d_prefix_exists": Path("/opt/mmdetection3d").exists(),
              "tar_available": shutil.which("tar") is not None,
              "gpu_data_or_training_loaded": False}
    print("EVENTTRACK_A100_READINESS " + json.dumps(report), flush=True)
    task.upload_artifact("a100-runtime-readiness", artifact_object=report, wait_on_upload=True)
    task.close()


if __name__ == "__main__":
    main()
