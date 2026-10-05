#!/usr/bin/env python3
"""Data-free, seeded finite-history exact-reference matrix on L40S CPU.

This is one Stage2 oracle component, not a tracker or a paper comparison.
"""
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlparse

SOURCE_TASK = "a07847eb50c3447a8210bd3914a98bdd"
SOURCE_SHA = "4ee688b675df384926468baa6aadc4deb22559685acd0c96c4306eaf0efc901a"
SEEDS = (1337, 2027, 3407)
PROJECT = "Thesis/Recover-Before-Fuse/Training"
FILE_HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def artifact_bytes(artifact):
    from clearml.storage.helper import StorageHelper
    url = artifact.url
    if urlparse(url).netloc not in FILE_HOSTS:
        raise ValueError("unapproved ClearML artifact host")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    data = b"".join(StorageHelper.get(url).download_as_stream(url))
    if len(data) != artifact.size or sha(data) != artifact.hash:
        raise ValueError("source artifact byte identity differs")
    return data


def frozen_oracle(data, directory):
    if sha(data) != SOURCE_SHA:
        raise ValueError("frozen source archive differs")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        member = archive.getmember("stage_two_exact_history_oracle.py")
        if not member.isfile():
            raise ValueError("oracle source is not a regular file")
        source = archive.extractfile(member).read()
    path = directory / member.name
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("frozen_stage2_exact_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.oracle, sha(source)


def node(node_id, source_id, frame_id, arrival, choices):
    return {"node_id": node_id, "source_id": source_id, "frame_id": frame_id,
            "information_us": arrival - 10, "arrival_us": arrival,
            "choices": [{"parent_id": parent, "log_potential": weight}
                        for parent, weight in choices]}


def cases(seed):
    rng = random.Random(seed)
    a = node("a", -1, "anchor", 100, [(None, 0.0)])
    b = node("b", 0, "scan0", 200, [(None, 0.0), ("a", rng.uniform(-0.01, 0.01))])
    c = node("c", 0, "scan0", 210, [(None, 0.0), ("a", rng.uniform(-0.01, 0.01))])
    d = node("d", 1, "scan1", 300, [(None, 0.0), ("a", 0.1), ("b", -0.1)])
    return [
        ("both_sides_empty", []),
        ("birth_only", [a]),
        ("one_sided_candidate", [a, b]),
        ("same_slot_conflict_near_equal", [a, b, c]),
        ("rectangular_two_sources", [a, b, c, d]),
        ("wide_finite_log_span", [a, node("b", 0, "scan0", 200,
                                         [(None, -500.0), ("a", 500.0)]),
                                  node("c", 1, "scan1", 300,
                                       [(None, 500.0), ("a", -500.0), ("b", 0.0)])]),
        ("later_disambiguation", [a, b, c,
                                  node("d", 1, "scan1", 300,
                                       [(None, 0.0), ("a", -8.0), ("b", 8.0), ("c", -8.0)])]),
    ]


def run(oracle, oracle_sha, task=None):
    started = time.monotonic()
    checks = []
    for seed in SEEDS:
        for name, nodes in cases(seed):
            result = oracle(nodes, max_configurations=100000)
            if result["production_modules_imported"] is not False or result["legal_histories"] < 1:
                raise ValueError("independent exact-reference contract differs")
            checks.append({"seed": seed, "case": name, "nodes": len(nodes),
                           "legal_histories": result["legal_histories"],
                           "cartesian_configurations": result["cartesian_configurations"],
                           "log_partition_decimal": result["log_partition_decimal"],
                           "bayes_risk_decimal": result["bayes_risk_decimal"],
                           "reference_sha256": sha(json.dumps(result, sort_keys=True,
                                                              separators=(",", ":")).encode())})
            elapsed = time.monotonic() - started
            eta = elapsed / len(checks) * (len(SEEDS) * 7 - len(checks))
            print(f"stage2 exact matrix {len(checks)}/21 seed={seed} case={name} "
                  f"ETA={eta:.1f}s", flush=True)
            if task is not None:
                task.get_logger().report_scalar("exact_matrix", "completed_cases",
                                                len(checks), iteration=len(checks))
    assert len(checks) == 21
    return {"kind": "stage2_exact_random_finite_history_matrix_v1",
            "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seeds": SEEDS, "case_count": len(checks), "checks": checks,
            "source_task_id": SOURCE_TASK, "source_sha256": SOURCE_SHA,
            "oracle_source_sha256": oracle_sha, "dataset_read": False,
            "ground_truth_read": False, "parameter_training": False,
            "production_modules_imported": False,
            "full_stage_two_complete": False, "same_resource_baselines_complete": False,
            "paper_performance_complete": False}


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from clearml import Task
    task = Task.init(project_name=PROJECT,
                     task_name="RBF stage2 exact random matrix L40S CPU",
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    task.reload()
    worker = task.data.last_worker or ""
    if not worker.startswith("10.100.35.121-L40S:"):
        raise ValueError("L40S worker required")
    expected = task.get_parameters()["General/bootstrap_source_sha256"]
    if sha(Path(__file__).read_bytes()) != expected:
        raise ValueError("bootstrap source differs")
    source = Task.get_task(task_id=SOURCE_TASK)
    if source.status != "completed" or source.artifacts["source"].hash != SOURCE_SHA:
        raise ValueError("frozen source task differs")
    with tempfile.TemporaryDirectory(prefix="rbf-stage2-exact-matrix-") as tmp:
        oracle, oracle_sha = frozen_oracle(artifact_bytes(source.artifacts["source"]), Path(tmp))
        result = run(oracle, oracle_sha, task)
        result.update({"task_id": task.id, "worker_id": worker,
                       "bootstrap_source_sha256": expected, "device": "cpu"})
        path = Path(tmp) / "exact-matrix-receipt.json"
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        task.upload_artifact("exact-matrix-receipt", artifact_object=str(path), wait_on_upload=True)
        task.close()


if __name__ == "__main__":
    main()
