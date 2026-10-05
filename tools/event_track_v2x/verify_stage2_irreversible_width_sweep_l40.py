#!/usr/bin/env python3
"""Independently replay frozen stage-two width-sweep outputs on L40S CPU."""

import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlparse


PROJECT = "Thesis/Recover-Before-Fuse/Training"
SOURCE_TASK = "2cb98df61fc9453ca00a3ecaa2d701a3"
SOURCE_SHA = "5499d5f3e7fc89a095e51be953ee2923efef79bc7238ddbf18d85e4e4a44471e"
TASKS = {
    1337: "250ff4592ae9418ca6f50b8d7ff0a72c",
    2027: "59e0cc40d0514beba590ef07f83f613f",
    3407: "294ba8a11c9c4348ac4527d824e7ec22",
}
ARTIFACT_NAMES = ("event-results", "tracker-database")
WIDTHS = (1, 2, 4)
ALLOWED_FILE_HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def download(artifact):
    from clearml.storage.helper import StorageHelper

    url = artifact.url
    if urlparse(url).netloc not in ALLOWED_FILE_HOSTS:
        raise ValueError("artifact outside approved file service")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    data = b"".join(StorageHelper.get(url).download_as_stream(url))
    if len(data) != artifact.size or sha_bytes(data) != artifact.hash:
        raise ValueError("ClearML artifact byte identity differs")
    return data


def extract_source(archive_bytes, root):
    import io

    if sha_bytes(archive_bytes) != SOURCE_SHA:
        raise ValueError("frozen source archive differs")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members or any(not m.isfile() or m.name.startswith("/") or ".." in Path(m.name).parts
                              for m in members):
            raise ValueError("unsafe frozen source archive")
        for member in members:
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    return len(members)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from clearml import Task

    task = Task.init(project_name=PROJECT, task_name="RBF stage2 irreversible width sweep sealed replay L40S CPU",
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    task.reload()
    worker = task.data.last_worker or ""
    if not worker.startswith("10.100.35.121-L40S:"):
        raise ValueError("L40S worker required")
    params = task.get_parameters()
    verifier_sha = params["General/verifier_source_sha256"]
    bootstrap_sha = params["General/bootstrap_source_sha256"]
    if sha_bytes(Path(__file__).read_bytes()) != bootstrap_sha:
        raise ValueError("bootstrap source differs")
    source_task = Task.get_task(task_id=SOURCE_TASK)
    if source_task.status != "completed" or source_task.artifacts["source"].hash != SOURCE_SHA:
        raise ValueError("frozen source task differs")
    with tempfile.TemporaryDirectory(prefix="rbf-stage2-width-replay-") as directory:
        root = Path(directory)
        source_members = extract_source(download(source_task.artifacts["source"]), root)
        self_task = Task.get_task(task_id=task.id)
        verifier_bytes = download(self_task.artifacts["sealed-replay-verifier"])
        if sha_bytes(verifier_bytes) != verifier_sha:
            raise ValueError("verifier source differs")
        verifier_path = root / "verify_stage_two_irreversible_sealed_replay.py"
        verifier_path.write_bytes(verifier_bytes)
        sys.path.insert(0, str(root / "equivalence-source"))
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("frozen_sealed_replay_verifier", verifier_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import torch

        if torch.cuda.is_initialized():
            raise ValueError("CPU verifier initialized CUDA")
        checks = []
        start = time.monotonic()
        for seed, task_id in TASKS.items():
            input_task = Task.get_task(task_id=task_id)
            if input_task.status != "completed":
                raise ValueError("source width sweep not complete")
            for width in WIDTHS:
                paths = {}
                hashes = {}
                for name in ARTIFACT_NAMES:
                    key = f"{name}-k{width}"
                    artifact = input_task.artifacts[key]
                    data = download(artifact)
                    path = root / f"seed{seed}-{key}"
                    path.write_bytes(data)
                    paths[name] = path
                    hashes[key] = artifact.hash
                outcome = module.verify(paths["event-results"], paths["tracker-database"])
                if (outcome["seed"] != seed or outcome["original_database_sha256"] != hashes[f"tracker-database-k{width}"]
                        or outcome["original_database_unchanged"] is not True
                        or outcome["reopened_backend_duplicate_prediction_byte_equal"] is not True
                        or outcome["all_event_prediction_and_audit_bytes_unchanged"] is not True):
                    raise ValueError("sealed replay verification differs")
                checks.append({"seed": seed, "width": width, "task_id": task_id,
                               "input_hashes": hashes, "result": outcome})
                elapsed = time.monotonic() - start
                remaining = len(TASKS) * len(WIDTHS) - len(checks)
                print(json.dumps({"verified": len(checks), "total": 9,
                                  "eta_seconds": round(elapsed / len(checks) * remaining, 1)}), flush=True)
        receipt = {"kind": "stage2_irreversible_width_sweep_l40_cpu_sealed_replay_v1",
                   "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "task_id": task.id, "worker_id": worker, "source_task_id": SOURCE_TASK,
                   "source_sha256": SOURCE_SHA, "source_members": source_members,
                   "bootstrap_source_sha256": bootstrap_sha, "verifier_source_sha256": verifier_sha,
                   "torch": str(torch.__version__), "device": "cpu", "dataset_read": False,
                   "ground_truth_read": False, "parameter_training": False, "checks": checks,
                   "sealed_replay_verified": True, "full_stage_two_complete": False,
                   "paper_performance_complete": False}
        output = root / "sealed-replay-receipt.json"
        output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        task.upload_artifact("sealed-replay-receipt", artifact_object=str(output), wait_on_upload=True)
        task.close()


if __name__ == "__main__":
    main()
