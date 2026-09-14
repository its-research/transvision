#!/usr/bin/env python3
"""Default read-only; explicitly deploy a small no-data four-A100 probe.

Only the probe source and diagnostic console records go to private ClearML.
No datasets, model weights, predictions, queue settings or existing jobs change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.event_track_v2x.submit_forest_identity_ddp import IMAGE, PROJECT, available

PROBE = Path(__file__).with_name("probe_v2v4real_a100_runtime.py")
QUEUE = "GPU4-A100"
PROBES = {
    "inventory": (PROBE, "runtime-inventory"),
    "pointpillar": (Path(__file__).with_name("run_v2v4real_pointpillar_smoke.py"), "pointpillar-smoke"),
}


def deploy(Task, snapshot, source, expected_sha256, *, probe_kind="inventory"):
    if probe_kind not in PROBES:
        raise ValueError("only predeclared A100 probes may be submitted")
    probe, purpose = PROBES[probe_kind]
    digest = hashlib.sha256(source).hexdigest()
    if digest != expected_sha256 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("probe source SHA-256 mismatch")
    name = f"RBF V2V4Real {purpose} {digest[:12]} four-A100"
    existing = Task.get_tasks(project_name=PROJECT, task_name="^" + re.escape(name) + "$",
        task_filter={"status": ["created", "queued", "in_progress", "completed"]})
    if existing:
        return dict(task_ids=[t.id for t in existing], duplicate_not_created=True,
                    training_submitted=False)
    ready = snapshot()
    if not ready or any(r["family"] != "A100" or r["queue"] != QUEUE for r in ready):
        raise ValueError("an idle non-overlapping four-A100 worker is required")
    task = Task.create(project_name=PROJECT, task_name=name, task_type=Task.TaskTypes.testing)
    task.add_tags(["a100-only", "no-dataset-read", purpose, "not-training-results"])
    task.set_script(repository="", branch="", commit="", diff=source.decode("utf-8"),
                    working_dir=".", entry_point=probe.name)
    task.set_packages(["clearml==2.1.2"])
    task.set_base_docker(IMAGE, docker_arguments="-e CLEARML_AGENT_FORCE_TASK_INIT=1 --shm-size 32g")
    task.set_parameters(dict(required_gpu_count=4, required_device="A100", class_scope="car",
        source_sha256=digest, probe_kind=probe_kind, runtime_inventory_only=probe_kind == "inventory",
        detector_forward_planned=probe_kind == "pointpillar", environment_changes="task-private-venv-only",
        parameter_training=False, data_upload=False, V100_excluded=True, RTX5090_excluded=True))
    fresh = snapshot()
    if not fresh or any(r["family"] != "A100" or r["queue"] != QUEUE for r in fresh):
        raise RuntimeError(f"capacity changed; task {task.id} remains created, not enqueued")
    # An uncertain enqueue must be inspected by task id, never blindly retried.
    print("RBF_RUNTIME_TASK_CREATED " + json.dumps(dict(task_id=task.id)), flush=True)
    Task.enqueue(task, queue_name=QUEUE)
    return dict(task_id=task.id, queue=QUEUE, source_sha256=digest,
                status=str(task.get_status()), training_submitted=False,
                runtime_inventory_submitted=probe_kind == "inventory",
                pointpillar_smoke_submitted=probe_kind == "pointpillar", completion_verified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probe-sha256")
    parser.add_argument("--probe-kind", choices=tuple(PROBES), default="inventory")
    args = parser.parse_args(argv)
    source = PROBES[args.probe_kind][0].read_bytes()
    if args.execute and hashlib.sha256(source).hexdigest() != args.probe_sha256:
        raise ValueError("explicit current probe SHA-256 required for deployment")
    os.environ["CLEARML_FILES_HOST"] = "http://10.100.35.118:8081"
    from clearml import Task
    from clearml.backend_api import Session
    from clearml.backend_api.session.client import APIClient
    if urlparse(Session.get_api_server_host()).hostname != "10.100.35.118":
        raise ValueError("only the designated private ClearML server is allowed")
    client = APIClient()
    def snapshot():
        workers = [w.to_dict() for w in client.workers.get_all(last_seen=120)]
        queues = {q.id: q.to_dict() for q in client.queues.get_all()}
        return [row for row in available(workers, queues) if row["family"] == "A100"]
    if not args.execute:
        result = dict(available_four_gpu_workers=snapshot(), active_hardware_policy="A100-only",
                      snapshot_is_reservation=False, any_task_submitted=False)
    else:
        result = deploy(Task, snapshot, source, args.probe_sha256, probe_kind=args.probe_kind)
    print("RBF_RUNTIME_SUBMISSION " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
