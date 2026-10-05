#!/usr/bin/env python3
"""Independently replay frozen stage-two empty/edge outputs on L40S CPU."""

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
import time
from urllib.parse import urlparse

PROJECT = "Thesis/Recover-Before-Fuse/Training"
SOURCE_TASK = "91e2a5402f9e4ec3bc7442ebe055ef80"
SOURCE_SHA = "91af36bb2baac61711ccd93283008d26fb9ffcc6a28bb1e5bb8360334c022218"
TASKS = {1337: "6dbdd050c5d847eebdd23b2e96d59eb9",
         2027: "e2b2c439a80a4c698db1a9d3871114b4",
         3407: "67e47f68da43497cad368fcd639cd92c"}
CASES = ("both_sources_empty", "source_zero_empty", "allowed_parent_edges_empty")
ALLOWED_HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def download(artifact):
    from clearml.storage.helper import StorageHelper
    url = artifact.url
    if urlparse(url).netloc not in ALLOWED_HOSTS:
        raise ValueError("artifact outside approved ClearML file service")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    data = b"".join(StorageHelper.get(url).download_as_stream(url))
    if len(data) != artifact.size or sha(data) != artifact.hash:
        raise ValueError("artifact byte identity differs")
    return data


def extract_source(data, root):
    if sha(data) != SOURCE_SHA:
        raise ValueError("frozen source archive differs")
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
    task = Task.init(project_name=PROJECT,
                     task_name="RBF stage2 empty-edge sealed replay L40S CPU",
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    task.reload()
    worker = task.data.last_worker or ""
    if not worker.startswith("10.100.35.121-L40S:"):
        raise ValueError("L40S worker required")
    params = task.get_parameters()
    bootstrap_sha = params["General/bootstrap_source_sha256"]
    verifier_sha = params["General/verifier_source_sha256"]
    if sha(Path(__file__).read_bytes()) != bootstrap_sha:
        raise ValueError("bootstrap source differs")
    source_task = Task.get_task(task_id=SOURCE_TASK)
    if source_task.status != "completed" or source_task.artifacts["source"].hash != SOURCE_SHA:
        raise ValueError("frozen source task differs")
    with tempfile.TemporaryDirectory(prefix="rbf-stage2-empty-edge-") as directory:
        root = Path(directory)
        members = extract_source(download(source_task.artifacts["source"]), root)
        own = Task.get_task(task_id=task.id)
        verifier_data = download(own.artifacts["sealed-replay-verifier"])
        if sha(verifier_data) != verifier_sha:
            raise ValueError("verifier source differs")
        verifier_path = root / "verify_stage_two_empty_edge_sealed_replay.py"
        verifier_path.write_bytes(verifier_data)
        sys.path.insert(0, str(root / "equivalence-source"))
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("frozen_empty_edge_replay_verifier", verifier_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import torch
        if torch.cuda.is_initialized():
            raise ValueError("CPU verifier initialized CUDA")
        checks = []
        start = time.monotonic()
        for seed, task_id in TASKS.items():
            original = Task.get_task(task_id=task_id)
            if original.status != "completed":
                raise ValueError("empty-edge task not complete")
            for case in CASES:
                inputs = {}
                hashes = {}
                for prefix in ("event-results", "tracker-database"):
                    key = prefix + "-" + case
                    artifact = original.artifacts[key]
                    path = root / f"seed{seed}-{key}"
                    path.write_bytes(download(artifact))
                    inputs[prefix] = path
                    hashes[key] = artifact.hash
                result = module.verify(inputs["event-results"], inputs["tracker-database"])
                if (result["seed"] != seed or result["case"] != case or
                        result["original_database_sha256"] != hashes["tracker-database-" + case] or
                        result["original_database_unchanged"] is not True or
                        result["reopened_initial_empty_prediction_byte_equal"] is not True or
                        result["all_persisted_tables_unchanged"] is not True):
                    raise ValueError("empty-edge sealed replay differs")
                checks.append({"seed": seed, "case": case, "task_id": task_id,
                               "input_hashes": hashes, "result": result})
                elapsed = time.monotonic() - start
                print(json.dumps({"verified": len(checks), "total": 9,
                                  "eta_seconds": round(elapsed / len(checks) * (9 - len(checks)), 1)}),
                      flush=True)
        receipt = {"kind": "stage2_empty_edge_l40_cpu_sealed_replay_v1",
                   "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "task_id": task.id, "worker_id": worker,
                   "source_task_id": SOURCE_TASK, "source_sha256": SOURCE_SHA,
                   "source_members": members, "bootstrap_source_sha256": bootstrap_sha,
                   "verifier_source_sha256": verifier_sha, "torch": str(torch.__version__),
                   "device": "cpu", "dataset_read": False, "ground_truth_read": False,
                   "parameter_training": False, "checks": checks,
                   "sealed_replay_verified": True, "full_stage_two_complete": False,
                   "paper_performance_complete": False}
        output = root / "sealed-replay-receipt.json"
        output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        task.upload_artifact("sealed-replay-receipt", artifact_object=str(output), wait_on_upload=True)
        task.close()


if __name__ == "__main__":
    main()
