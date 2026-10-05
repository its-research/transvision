#!/usr/bin/env python3
"""Submit one deduplicated data-free Stage2 broad exact matrix to L40S CPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse

PROJECT = "Thesis/Recover-Before-Fuse/Training"
QUEUE_ID = "8d0f8b54037249eeb0f1cc70cbfe73ab"
IMAGE = ("gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:"
         "e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb")


def submit(source):
    from clearml import Task
    from clearml.backend_api import Session
    from clearml.backend_api.session.client import APIClient
    if urlparse(Session.get_api_server_host()).hostname != "10.100.35.118":
        raise ValueError("private ClearML API identity differs")
    queue = APIClient().queues.get_by_id(queue=QUEUE_ID)
    if queue.name != "GPU3-L40S":
        raise ValueError("L40S queue identity differs")
    payload = source.read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    name = "RBF Stage2 broad exact matrix L40S CPU " + source_sha[:12]
    matches = Task.get_tasks(project_name=PROJECT, task_name="^" + re.escape(name) + "$")
    if matches:
        if len(matches) != 1 or hashlib.sha256(
                (matches[0].data.script.diff or "").encode()).hexdigest() != source_sha:
            raise ValueError("existing broad exact matrix task identity differs")
        return {"task_id": matches[0].id, "status": str(matches[0].status),
                "source_sha256": source_sha, "duplicate_not_created": True}
    task = Task.create(project_name=PROJECT, task_name=name,
                       task_type=Task.TaskTypes.testing, binary="python3.12")
    task.set_script(repository="", branch="", commit="", working_dir=".",
                    entry_point=source.name, diff=payload.decode())
    task.set_packages(["clearml==2.1.2"])
    task.set_base_docker(IMAGE, docker_arguments=(
        "-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 4g "
        "--env CUDA_VISIBLE_DEVICES= --env NVIDIA_DRIVER_CAPABILITIES=compute "
        "--env CLEARML_FILES_HOST=http://10.100.35.118:8081 "
        "--env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 "
        "--env MKL_NUM_THREADS=1 --env PYTHONHASHSEED=0 "
        "--env PYTHONDONTWRITEBYTECODE=1"))
    task.set_parameters({"bootstrap_source_sha256": source_sha,
                         "frozen_oracle_source_task_id":
                         "a07847eb50c3447a8210bd3914a98bdd",
                         "device": "cpu", "dataset_read": False,
                         "parameter_training": False,
                         "full_stage_two_complete": False})
    task.add_tags(["Recover-Before-Fuse", "Stage2", "broad-exact-reference",
                   "L40S-CPU", "no-dataset", "no-paper-metrics"])
    task.reload()
    if hashlib.sha256((task.data.script.diff or "").encode()).hexdigest() != source_sha:
        raise ValueError("created task script differs; inspect before enqueue")
    Task.enqueue(task, queue_id=QUEUE_ID)
    return {"task_id": task.id, "status": str(task.status),
            "source_sha256": source_sha, "queue_id": QUEUE_ID,
            "duplicate_not_created": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    args = p.parse_args()
    if args.receipt.exists() or args.receipt.is_symlink():
        raise ValueError("fresh dispatch receipt required")
    row = submit(args.source)
    receipt = {"kind": "stage2_broad_exact_matrix_l40_cpu_dispatch_v1",
               "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "source_path": str(args.source), "image": IMAGE, "result": row,
               "dataset_read": False, "parameter_training": False,
               "full_stage_two_complete": False,
               "same_resource_baselines_complete": False,
               "paper_performance_complete": False,
               "eta": "unknown until task-level progress is reported"}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(args.receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
