#!/usr/bin/env python3
"""Independently read back and recompute each broad exact matrix case."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlparse

from clearml import Task
from clearml.storage.helper import StorageHelper

TASK_ID = "23f3b05694b04f3ab59654b0f55aa5f3"
SOURCE_SHA = "df558a7694e6095430a34818d2ef56b1ccb3247715556176fbc0b58a5740f87a"
ORACLE_SHA = "29c038639fb1db1f5604a4f1ac150827752e22f86ccefc3bfcdd9356bc5821c0"
ROOT = Path("test/recover-before-fuse")
SOURCE = Path("tools/event_track_v2x/run_stage2_broad_exact_matrix_l40.py")
ORACLE = ROOT / "historical/live-audit-20260926/stage_two_exact_history_oracle.py"
OUTPUT = ROOT / "artifacts/stage2-broad-exact-matrix-l40-20260930"


def must(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    must(not OUTPUT.exists() and not OUTPUT.is_symlink(), "acceptance output already exists")
    task = Task.get_task(task_id=TASK_ID)
    must(task.status == "completed" and task.get_last_iteration() == 192,
         "task not completed for all 192 cases")
    must(set(task.artifacts) == {"broad-exact-matrix-receipt"},
         "artifact inventory differs")
    must(sha(SOURCE.read_bytes()) == SOURCE_SHA and
         sha((task.data.script.diff or "").encode()) == SOURCE_SHA,
         "bootstrap source differs")
    must(sha(ORACLE.read_bytes()) == ORACLE_SHA, "local independent oracle differs")
    runner = load_module(SOURCE, "stage2_broad_matrix_runner")
    oracle = load_module(ORACLE, "stage2_broad_matrix_oracle")
    artifact = task.artifacts["broad-exact-matrix-receipt"]
    url = artifact.url
    must(urlparse(url).netloc in {"10.100.34.118:8081", "10.100.35.118:8081"},
         "artifact host differs")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    payload = b"".join(StorageHelper.get(url).download_as_stream(url))
    must(len(payload) == artifact.size and sha(payload) == artifact.hash,
         "artifact byte readback differs")
    report = json.loads(payload)
    must(report["task_id"] == TASK_ID and report["worker_id"] == task.data.last_worker
         and report["worker_id"].startswith("10.100.35.121-L40S:")
         and report["device"] == "cpu", "execution identity differs")
    must(report["bootstrap_source_sha256"] == SOURCE_SHA and
         report["source_task_id"] == runner.SOURCE_TASK and
         report["source_sha256"] == runner.SOURCE_SHA and
         report["oracle_sha256"] == ORACLE_SHA,
         "source lineage differs")
    rows = report["rows"]
    must(report["case_count"] == len(rows) == 192 and
         report["seeds"] == [1337, 2027, 3407] and
         report["cases_per_seed"] == 64, "matrix coverage differs")
    must([(r["seed"], r["index"]) for r in rows] ==
         [(seed, index) for seed in runner.SEEDS for index in range(64)],
         "case order or identity differs")
    for row in rows:
        expected = runner.case(row["seed"], row["index"])
        must(all(row[k] == v for k, v in expected.items()), "case input differs")
        result = oracle.oracle(row["nodes"], max_configurations=100000)
        must(row["legal_histories"] == result["legal_histories"] and
             row["cartesian_configurations"] == result["cartesian_configurations"] and
             row["log_partition_decimal"] == result["log_partition_decimal"] and
             row["bayes_risk_decimal"] == result["bayes_risk_decimal"] and
             row["reference_sha256"] == sha(json.dumps(result, sort_keys=True,
                                                       separators=(",", ":")).encode()),
             "independent exact recomputation differs")
    must(all(report[key] is False for key in
             ("dataset_read", "ground_truth_read", "parameter_training",
              "full_stage_two_complete", "same_resource_baselines_complete",
              "paper_performance_complete")), "scope differs")
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "broad-exact-matrix-receipt.json").open("xb") as stream:
        stream.write(payload)
    acceptance = {"kind": "stage2_broad_exact_matrix_l40_independent_readback_v1",
                  "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "task_id": TASK_ID, "worker_id": report["worker_id"],
                  "source_sha256": SOURCE_SHA, "oracle_sha256": ORACLE_SHA,
                  "artifact_sha256": sha(payload), "artifact_bytes": len(payload),
                  "case_count": len(rows),
                  "total_legal_histories": sum(r["legal_histories"] for r in rows),
                  "regimes": sorted({r["regime"] for r in rows}),
                  "independent_case_recomputation": True,
                  "full_stage_two_complete": False,
                  "same_resource_baselines_complete": False,
                  "paper_performance_complete": False}
    path = OUTPUT / "acceptance-receipt.json"
    with path.open("x") as stream:
        json.dump(acceptance, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"task_id": TASK_ID, "artifact_sha256": sha(payload),
                      "acceptance_sha256": sha(path.read_bytes()),
                      "case_count": len(rows), "status": "accepted_component"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
