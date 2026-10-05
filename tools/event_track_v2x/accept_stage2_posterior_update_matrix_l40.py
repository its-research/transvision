#!/usr/bin/env python3
"""Independently read back and accept the frozen Stage2 posterior matrix artifact."""
import datetime
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from clearml import Task
from clearml.storage.helper import StorageHelper

TASK_ID = "9bdec022720c4878949c60b40efcdf9c"
SOURCE_SHA = "b58536f3338d0198c9cc8e51aeec86d5aa7be282958fd12635c2cd1a8fb7ef85"
ORACLE_ARCHIVE_SHA = "4ee688b675df384926468baa6aadc4deb22559685acd0c96c4306eaf0efc901a"
ORACLE_SOURCE_SHA = "29c038639fb1db1f5604a4f1ac150827752e22f86ccefc3bfcdd9356bc5821c0"
ROOT = Path("test/recover-before-fuse")
OUTPUT = ROOT / "artifacts/stage2-posterior-update-matrix-l40-20260930"


def must(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    must(not OUTPUT.exists() and not OUTPUT.is_symlink(), "acceptance output already exists")
    task = Task.get_task(task_id=TASK_ID)
    must(task.status == "completed" and task.get_last_iteration() == 21,
         "task not complete with 21 cases")
    must(set(task.artifacts) == {"posterior-update-matrix-receipt"},
         "task artifact inventory differs")
    source = (ROOT / "source-freezes/run_stage2_posterior_update_matrix_l40_20260930.py").read_bytes()
    must(hashlib.sha256(source).hexdigest() == SOURCE_SHA and
         hashlib.sha256((task.data.script.diff or "").encode()).hexdigest() == SOURCE_SHA,
         "frozen bootstrap bytes differ")
    artifact = task.artifacts["posterior-update-matrix-receipt"]
    url = artifact.url
    must(urlparse(url).netloc in {"10.100.34.118:8081", "10.100.35.118:8081"},
         "artifact host differs")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    payload = b"".join(StorageHelper.get(url).download_as_stream(url))
    payload_sha = hashlib.sha256(payload).hexdigest()
    must(len(payload) == artifact.size and payload_sha == artifact.hash,
         "artifact independent byte readback differs")
    report = json.loads(payload)
    must(report["task_id"] == TASK_ID and report["worker_id"] == task.data.last_worker
         and report["worker_id"].startswith("10.100.35.121-L40S:")
         and report["device"] == "cpu", "execution identity differs")
    must(report["bootstrap_source_sha256"] == SOURCE_SHA and
         report["oracle_archive_sha256"] == ORACLE_ARCHIVE_SHA and
         report["oracle_source_sha256"] == ORACLE_SOURCE_SHA,
         "source lineage differs")
    expected_cases = ("both_sides_empty", "birth_only", "one_sided_candidate",
                      "same_slot_conflict_near_equal", "rectangular_two_sources",
                      "wide_finite_log_span", "later_disambiguation")
    expected = {(seed, case) for seed in (1337, 2027, 3407) for case in expected_cases}
    checks = report["checks"]
    must(report["case_count"] == len(checks) == 21 and report["seeds"] == [1337, 2027, 3407]
         and {(r["seed"], r["case"]) for r in checks} == expected,
         "case coverage differs")
    verified = 0
    with localcontext() as context:
        context.prec = 90
        for row in checks:
            if row["status"] == "vacuous_single_history":
                must(row["case"] in {"both_sides_empty", "birth_only"}
                     and row["legal_histories"] == 1, "vacuous case differs")
                continue
            must(row["status"] == "verified" and row["legal_histories"] > 1 and
                 row["retained_histories"] + row["omitted_histories"] == row["legal_histories"],
                 "nontrivial history partition differs")
            eta = Decimal(row["prior_omitted_mass_decimal"])
            posterior = Decimal(row["posterior_omitted_mass_decimal"])
            formula = Decimal(row["bayes_formula_decimal"])
            upper = Decimal(row["finite_kappa_upper_decimal"])
            counterexample = Decimal(row["invalid_kappa_one_counterexample_posterior_decimal"])
            must(0 < eta < 1 and 0 < posterior < 1 and
                 abs(posterior - formula) < Decimal("1e-65") and
                 posterior <= upper + Decimal("1e-65") and
                 counterexample > eta, "numeric acceptance differs")
            verified += 1
    must(verified == 15 and all(report[key] is False for key in
         ("dataset_read", "ground_truth_read", "production_modules_imported",
          "parameter_training", "full_stage_two_complete",
          "same_resource_baselines_complete", "paper_performance_complete")),
         "verified count or scope differs")
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "posterior-update-matrix-receipt.json").open("xb") as stream:
        stream.write(payload)
    acceptance = {"kind": "stage2_posterior_update_matrix_l40_independent_readback_v1",
                  "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "task_id": TASK_ID, "status": task.status,
                  "worker_id": report["worker_id"], "source_sha256": SOURCE_SHA,
                  "oracle_archive_sha256": ORACLE_ARCHIVE_SHA,
                  "oracle_source_sha256": ORACLE_SOURCE_SHA,
                  "artifact_sha256": payload_sha, "artifact_bytes": len(payload),
                  "case_count": 21, "verified_nontrivial_cases": verified,
                  "vacuous_single_history_cases": 6,
                  "bayes_update_and_finite_kappa_accepted": True,
                  "full_stage_two_complete": False,
                  "same_resource_baselines_complete": False,
                  "paper_performance_complete": False}
    path = OUTPUT / "acceptance-receipt.json"
    with path.open("x") as stream:
        json.dump(acceptance, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"task_id": TASK_ID, "artifact_sha256": payload_sha,
                      "acceptance_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "verified_nontrivial_cases": verified,
                      "status": "accepted_component"}, sort_keys=True))


if __name__ == "__main__":
    main()
