#!/usr/bin/env python3
"""Run frozen protocol-boundary requests against accepted late-event artifacts."""

import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from urllib.parse import urlparse

PROJECT = "Thesis/Recover-Before-Fuse/Training"
SOURCE_TASK = "a07847eb50c3447a8210bd3914a98bdd"
SOURCE_SHA = "4ee688b675df384926468baa6aadc4deb22559685acd0c96c4306eaf0efc901a"
PARENTS = {1337: "88b25f8e71704e909a5ba50f3aaabb5b",
           2027: "939704d8e6d14dd78d213ed6e7a8dd28",
           3407: "5b7535e981054bbca8f5b2053fbea57c"}
HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def download(artifact):
    from clearml.storage.helper import StorageHelper
    url = artifact.url
    if urlparse(url).netloc not in HOSTS:
        raise ValueError("artifact outside approved ClearML file service")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    data = b"".join(StorageHelper.get(url).download_as_stream(url))
    if len(data) != artifact.size or sha(data) != artifact.hash:
        raise ValueError("artifact byte identity differs")
    return data


def extract_source(data, root):
    if sha(data) != SOURCE_SHA:
        raise ValueError("frozen source differs")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members or any(not m.isfile() or m.name.startswith("/") or
                              ".." in Path(m.name).parts for m in members):
            raise ValueError("unsafe source archive")
        for member in members:
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    return len(members)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from clearml import Task
    task = Task.init(project_name=PROJECT, task_name="RBF stage2 protocol boundary L40S CPU",
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    task.reload()
    worker = task.data.last_worker or ""
    if not worker.startswith("10.100.35.121-L40S:"):
        raise ValueError("L40S worker required")
    params = task.get_parameters()
    seed = int(params["General/seed"])
    if seed not in PARENTS or params["General/parent_task_id"] != PARENTS[seed]:
        raise ValueError("same-seed parent differs")
    bootstrap_sha = params["General/bootstrap_source_sha256"]
    entrypoint_sha = params["General/entrypoint_source_sha256"]
    if sha(Path(__file__).read_bytes()) != bootstrap_sha:
        raise ValueError("bootstrap source differs")
    source = Task.get_task(task_id=SOURCE_TASK)
    parent = Task.get_task(task_id=PARENTS[seed])
    if source.status != "completed" or source.artifacts["source"].hash != SOURCE_SHA:
        raise ValueError("source task differs")
    if parent.status != "completed":
        raise ValueError("parent task incomplete")
    with tempfile.TemporaryDirectory(prefix="rbf-stage2-boundary-") as directory:
        root = Path(directory)
        members = extract_source(download(source.artifacts["source"]), root)
        own = Task.get_task(task_id=task.id)
        entrypoint = download(own.artifacts["boundary-entrypoint"])
        if sha(entrypoint) != entrypoint_sha:
            raise ValueError("frozen boundary entrypoint differs")
        entrypoint_path = root / "run_stage_two_protocol_boundary_component_durable_v2.py"
        entrypoint_path.write_bytes(entrypoint)
        sys.path.insert(0, str(root / "equivalence-source"))
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("frozen_protocol_boundary_runner", entrypoint_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import torch
        if torch.cuda.is_initialized():
            raise ValueError("CPU task initialized CUDA")
        inputs = {}
        for key in ("event-results", "tracker-database"):
            data = download(parent.artifacts[key])
            path = root / f"parent-{key}"
            path.write_bytes(data)
            inputs[key] = {"path": path, "sha256": sha(data)}
        output = root / "boundary-result"
        receipt = module.run(inputs["event-results"]["path"],
                             inputs["tracker-database"]["path"], output)
        if (receipt["seed"] != seed or len(receipt["checks"]) != 4 or
                receipt["source_result_sha256"] != inputs["event-results"]["sha256"] or
                receipt["source_database_sha256"] != inputs["tracker-database"]["sha256"] or
                receipt["original_database_unchanged"] is not True or
                receipt["reopened_all_sql_tables_unchanged"] is not True or
                receipt["expected_error_paths_verified"] is not True):
            raise ValueError("protocol boundary check differs")
        combined = {"kind": "stage2_protocol_boundary_l40_cpu_execution_v1",
                    "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "task_id": task.id, "seed": seed, "worker_id": worker,
                    "source_task_id": SOURCE_TASK, "source_sha256": SOURCE_SHA,
                    "source_members": members, "parent_task_id": PARENTS[seed],
                    "parent_artifact_hashes": {k: v["sha256"] for k, v in inputs.items()},
                    "bootstrap_source_sha256": bootstrap_sha,
                    "entrypoint_source_sha256": entrypoint_sha,
                    "torch": str(torch.__version__), "device": "cpu",
                    "dataset_read": False, "ground_truth_read": False,
                    "parameter_training": False, "result": receipt,
                    "full_stage_two_complete": False,
                    "paper_performance_complete": False}
        path = root / "boundary-receipt.json"
        path.write_text(json.dumps(combined, sort_keys=True, indent=2) + "\n")
        task.upload_artifact("boundary-receipt", artifact_object=str(path), wait_on_upload=True)
        task.close()


if __name__ == "__main__":
    main()
