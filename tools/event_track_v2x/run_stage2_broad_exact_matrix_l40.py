#!/usr/bin/env python3
"""Data-free randomized finite-history reference matrix; not a paper result."""
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import tarfile
import tempfile
import time
from urllib.parse import urlparse

SOURCE_TASK = "a07847eb50c3447a8210bd3914a98bdd"
SOURCE_SHA = "4ee688b675df384926468baa6aadc4deb22559685acd0c96c4306eaf0efc901a"
ORACLE_SHA = "29c038639fb1db1f5604a4f1ac150827752e22f86ccefc3bfcdd9356bc5821c0"
SEEDS = (1337, 2027, 3407)
CASES_PER_SEED = 64
FILE_HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def frozen_oracle(directory):
    from clearml import Task
    from clearml.storage.helper import StorageHelper
    source = Task.get_task(task_id=SOURCE_TASK)
    artifact = source.artifacts["source"]
    if source.status != "completed" or artifact.hash != SOURCE_SHA:
        raise ValueError("frozen oracle source task differs")
    url = artifact.url
    if urlparse(url).netloc not in FILE_HOSTS:
        raise ValueError("unapproved artifact host")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    data = b"".join(StorageHelper.get(url).download_as_stream(url))
    if len(data) != artifact.size or sha(data) != SOURCE_SHA:
        raise ValueError("frozen source archive differs")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        member = archive.getmember("stage_two_exact_history_oracle.py")
        if not member.isfile():
            raise ValueError("oracle member differs")
        payload = archive.extractfile(member).read()
    if sha(payload) != ORACLE_SHA:
        raise ValueError("oracle source differs")
    path = directory / member.name
    path.write_bytes(payload)
    spec = importlib.util.spec_from_file_location("frozen_stage2_broad_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.oracle


def case(seed, index):
    rng = random.Random(f"{seed}:{index}:stage2-broad-v1")
    regime = ("near_equal", "wide_finite", "mixed")[index % 3]
    count = index % 6
    nodes = []
    for position in range(count):
        if regime == "near_equal":
            weights = lambda: rng.uniform(-0.001, 0.001)
        elif regime == "wide_finite":
            weights = lambda: rng.uniform(-300.0, 300.0)
        else:
            weights = lambda: rng.uniform(-8.0, 8.0)
        parents = [None]
        if position:
            parents += rng.sample([n["node_id"] for n in nodes],
                                  k=rng.randint(0, min(position, 3)))
        nodes.append({"node_id": f"n{position}", "source_id": rng.randrange(2),
                      "frame_id": f"f{rng.randrange(3)}",
                      "information_us": position * 1000,
                      "arrival_us": position * 1000 + 100,
                      "choices": [{"parent_id": parent, "log_potential": weights()}
                                  for parent in parents]})
    return {"seed": seed, "index": index, "regime": regime, "nodes": nodes}


def run(oracle, task=None):
    rows = []
    start = time.monotonic()
    total = len(SEEDS) * CASES_PER_SEED
    for seed in SEEDS:
        for index in range(CASES_PER_SEED):
            item = case(seed, index)
            result = oracle(item["nodes"], max_configurations=100000)
            if result["production_modules_imported"] is not False or result["legal_histories"] < 1:
                raise ValueError("independent reference contract differs")
            item.update({"legal_histories": result["legal_histories"],
                         "cartesian_configurations": result["cartesian_configurations"],
                         "log_partition_decimal": result["log_partition_decimal"],
                         "bayes_risk_decimal": result["bayes_risk_decimal"],
                         "reference_sha256": sha(json.dumps(result, sort_keys=True,
                                                             separators=(",", ":")).encode())})
            rows.append(item)
            if len(rows) % 8 == 0:
                eta = (time.monotonic() - start) / len(rows) * (total - len(rows))
                print(f"stage2 broad exact {len(rows)}/{total} ETA={eta:.1f}s", flush=True)
                if task is not None:
                    task.get_logger().report_scalar("broad_exact", "completed_cases",
                                                    len(rows), iteration=len(rows))
    return {"kind": "stage2_broad_independent_exact_matrix_v1",
            "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_task_id": SOURCE_TASK, "source_sha256": SOURCE_SHA,
            "oracle_sha256": ORACLE_SHA, "case_count": len(rows), "seeds": SEEDS,
            "cases_per_seed": CASES_PER_SEED, "rows": rows, "dataset_read": False,
            "ground_truth_read": False, "parameter_training": False,
            "full_stage_two_complete": False, "same_resource_baselines_complete": False,
            "paper_performance_complete": False}


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from clearml import Task
    task = Task.init(project_name="Thesis/Recover-Before-Fuse/Training",
                     task_name="RBF stage2 broad exact matrix L40S CPU",
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    task.reload()
    worker = task.data.last_worker or ""
    if not worker.startswith("10.100.35.121-L40S:"):
        raise ValueError("L40S worker required")
    expected = task.get_parameters()["General/bootstrap_source_sha256"]
    if sha(Path(__file__).read_bytes()) != expected:
        raise ValueError("bootstrap source differs")
    with tempfile.TemporaryDirectory(prefix="rbf-stage2-broad-") as tmp:
        root = Path(tmp)
        result = run(frozen_oracle(root), task)
        result.update({"task_id": task.id, "worker_id": worker,
                       "bootstrap_source_sha256": expected, "device": "cpu"})
        path = root / "broad-exact-matrix-receipt.json"
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        task.upload_artifact("broad-exact-matrix-receipt", artifact_object=str(path),
                             wait_on_upload=True)
        task.close()


if __name__ == "__main__":
    main()
