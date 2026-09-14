#!/usr/bin/env python3
"""Submit a bounded four-A100 readiness task to the existing private queue."""

import hashlib
import json
from pathlib import Path


def main():
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    project = "Thesis/EventTrack-V2X/Training"
    name = "EventTrack A100 migration readiness 2026-09-11"
    queue = "GPU4-A100"
    active = Task.get_tasks(project_name=project, task_name=name,
                            task_filter={"status": ["created", "queued", "in_progress"]})
    if active:
        print(json.dumps({"existing_task_id": active[0].id, "duplicate_not_created": True}))
        return
    client = APIClient()
    queues = [item for item in client.queues.get_all(name=queue) if item.name == queue]
    if len(queues) != 1 or queues[0].entries:
        raise RuntimeError("A100 queue must exist and have no queued entries")
    ready = [worker for worker in client.workers.get_all()
             if "A100" in worker.id and any(item.id == queues[0].id for item in worker.queues or [])
             and not getattr(getattr(worker, "task", None), "id", None)]
    if not ready:
        raise RuntimeError("no idle A100 worker")
    script = Path(__file__).with_name("probe_a100_runtime.py").read_bytes()
    task = Task.create(project_name=project, task_name=name, task_type=Task.TaskTypes.testing)
    task.add_tags(["a100-only", "migration-readiness", "no-dataset-read", "not-training-results"])
    task.set_script(repository="", branch="", commit="", diff=script.decode(),
                    working_dir=".", entry_point="probe_a100_runtime.py")
    task.set_packages(["clearml==2.1.2"])
    task.set_base_docker(
        "gitlab.zhht.ai.com:5000/aitech/model_infer:py-3.12-cuda-12.6.2-torch-2.10-ultralytics-8.4.13-dvc-v2-onnx-clearml",
        docker_arguments="-e CLEARML_AGENT_FORCE_TASK_INIT=1 --shm-size 32g")
    task.set_parameters({"required_device": "A100", "required_gpu_count": 4,
                         "script_sha256": hashlib.sha256(script).hexdigest(),
                         "source_upload_authorization": "user-explicit-A100-only-2026-09-11"})
    Task.enqueue(task, queue_name=queue)
    print(json.dumps({"task_id": task.id, "queue": queue, "status": str(task.get_status())}))


if __name__ == "__main__":
    main()
