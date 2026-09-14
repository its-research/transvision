#!/usr/bin/env python3
"""Upload an authorized train-only package and submit the A100-only controller."""

import argparse
import hashlib
import json
from pathlib import Path

from run_cooptrack_a100 import ARTIFACT_NAMES, validate_archive


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def upload_package(root):
    from clearml import Task
    manifest_file = root / "package-manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    manifest_hash = sha256(manifest_file)
    if manifest["val_or_test_included"] is not False or manifest["train_sequence_count"] != 46:
        raise ValueError("only the authorized full-train package can be uploaded")
    for item in manifest["inventory"]:
        path = root / item["path"]
        if path.is_symlink() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("package artifact mismatch before upload")
        if item["path"] == "runtime.tar.gz":
            validate_archive(path, ["opt/cooptrack", "usr/local/cuda-11.8/targets/x86_64-linux/lib"], allow_links=True)
        elif item["path"] == "source.tar.gz":
            validate_archive(path, ["workspace/CoopTrack", "entrypoints"])
        elif item["path"] == "train-inputs.tar.gz":
            validate_archive(path, ["inputs", "converted"])
    receipt = root / "clearml-package-task.json"
    if receipt.exists():
        saved = json.loads(receipt.read_bytes())
        if saved["manifest_sha256"] != manifest_hash:
            raise ValueError("existing upload task belongs to a different package")
        task = Task.get_task(task_id=saved["task_id"])
    else:
        task = Task.create(project_name="Thesis/EventTrack-V2X/Training",
                           task_name="A100 train-only runtime package 2026-09-11 v1",
                           task_type=Task.TaskTypes.data_processing)
        task.output_uri = True
        task.add_tags(["A100-migration", "train-only", "user-authorized-necessary-source-upload"])
        task.set_parameters({"package_manifest_sha256": manifest_hash,
                             "val_or_test_included": False, "train_sequences": 46,
                             "source_upload_authorization": "user-explicit-A100-only-2026-09-11"})
        write_json(receipt, {"task_id": task.id, "manifest_sha256": manifest_hash})
    task.mark_started(force=True)
    items = [(ARTIFACT_NAMES[item["path"]], root / item["path"], item["sha256"])
             for item in manifest["inventory"]]
    items.append(("package-manifest", manifest_file, manifest_hash))
    for name, path, expected in items:
        current = Task.get_task(task_id=task.id)
        if name in current.artifacts and current.artifacts[name].hash == expected:
            print("EVENTTRACK_UPLOAD_ALREADY_VERIFIED " + name, flush=True)
            continue
        if not task.upload_artifact(name, artifact_object=path, wait_on_upload=True):
            raise RuntimeError("artifact upload returned failure: " + name)
        current = Task.get_task(task_id=task.id)
        if name not in current.artifacts or current.artifacts[name].hash != expected:
            raise RuntimeError("uploaded artifact hash does not match: " + name)
        print("EVENTTRACK_UPLOAD_VERIFIED " + name, flush=True)
    task.mark_completed(force=True)
    print("EVENTTRACK_PACKAGE_TASK " + json.dumps({"task_id": task.id, "manifest_sha256": manifest_hash}), flush=True)
    return task.id, manifest_hash


def submit(root, package_id, manifest_hash, attempt):
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    project = "Thesis/EventTrack-V2X/Training"
    name = f"D2 A100 full-train R50 seed-1337 batch-auto attempt-{attempt}"
    existing = Task.get_tasks(project_name=project, task_name=name,
                              task_filter={"status": ["created", "queued", "in_progress"]})
    if existing:
        print("EVENTTRACK_EXISTING_A100_TASK " + existing[0].id, flush=True)
        return existing[0].id
    client = APIClient()
    queues = [item for item in client.queues.get_all(name="GPU4-A100") if item.name == "GPU4-A100"]
    if len(queues) != 1 or queues[0].entries:
        raise RuntimeError("A100 queue must be empty before this launch")
    workers = [worker for worker in client.workers.get_all()
               if "A100" in worker.id and any(q.id == queues[0].id for q in worker.queues or [])
               and not getattr(getattr(worker, "task", None), "id", None)]
    if not workers:
        raise RuntimeError("no idle A100 worker; other jobs are not changed")
    script = Path(__file__).with_name("run_cooptrack_a100.py").read_bytes()
    task = Task.create(project_name=project, task_name=name, task_type=Task.TaskTypes.training,
                       binary="python3.12")
    task.set_script(repository="", branch="", commit="", working_dir=".",
                    entry_point="run_cooptrack_a100.py", diff=script.decode())
    task.set_packages(["clearml==2.1.2"])
    task.set_base_docker(
        "gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb",
        # The controller initializes its task with explicit connection settings.
        # GPU device binding remains owned by the assigned four-card worker.
        docker_arguments="-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 32g --env NCCL_P2P_DISABLE=1 --env NVIDIA_DRIVER_CAPABILITIES=compute")
    task.set_parameters({"package_task_id": package_id, "package_manifest_sha256": manifest_hash,
                         "controller_sha256": hashlib.sha256(script).hexdigest(),
                         "required_gpu": "A100", "world_size": 4,
                         "nvidia_driver_capabilities": "compute",
                         "batch_candidates_per_gpu": [2, 4, 8, 10], "epochs_per_side": 24,
                         "train_sequences": 46, "seed": 1337, "val_test_loaded": False,
                         "source_upload_authorization": "user-explicit-A100-only-2026-09-11"})
    task.add_tags(["a100-only", "full-train", "batch-auto", "seed-1337", "paper-evidence-pending"])
    receipt = {"task_id": task.id, "package_task_id": package_id,
               "package_manifest_sha256": manifest_hash, "controller_sha256": hashlib.sha256(script).hexdigest(),
               "queue": "GPU4-A100", "name": name, "stage": "D2-detector-training"}
    write_json(root / f"a100-training-attempt-{attempt}.json", receipt)
    Task.enqueue(task, queue_name="GPU4-A100")
    print("EVENTTRACK_A100_TASK_QUEUED " + json.dumps(receipt), flush=True)
    return task.id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    args = parser.parse_args()
    if args.upload_only and args.submit_only:
        raise ValueError("upload-only and submit-only are mutually exclusive")
    if args.submit_only:
        from clearml import Task
        saved = json.loads((args.package / "clearml-package-task.json").read_bytes())
        package_id, manifest_hash = saved["task_id"], saved["manifest_sha256"]
        if sha256(args.package / "package-manifest.json") != manifest_hash:
            raise ValueError("package manifest changed since verified upload")
        task = Task.get_task(task_id=package_id)
        if str(task.get_status()) != "completed":
            raise RuntimeError("package upload is not completed")
        manifest = json.loads((args.package / "package-manifest.json").read_bytes())
        for item in manifest["inventory"]:
            artifact = task.artifacts[ARTIFACT_NAMES[item["path"]]]
            if artifact.hash != item["sha256"] or artifact.size != item["bytes"]:
                raise ValueError("uploaded package identity changed")
    else:
        package_id, manifest_hash = upload_package(args.package)
    if not args.upload_only:
        submit(args.package, package_id, manifest_hash, args.attempt)


if __name__ == "__main__":
    main()
