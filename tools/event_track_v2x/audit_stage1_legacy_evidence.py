#!/usr/bin/env python3
"""Verify the retained Stage 1 mechanism metadata and local evaluation files.

This deliberately does not promote legacy SPD val evidence to paper eligibility.
The original prediction streams are not part of this local readback.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


CAMPAIGNS = (
    ("mechanism-diagnostics-plan-20260912-v1.json",
     "mechanism-diagnostics-summary-20260912-v1.json", "mechanism-evaluation",
     "mechanism-event-audit"),
    ("birth-score-diagnostics-plan-20260912-v1.json",
     "birth-score-diagnostics-summary-20260912-v1.json",
     "birth-score-evaluation-20260912-v1", "birth-score-event-audit-20260912-v1"),
)


def digest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def read_json(path: Path):
    return json.loads(path.read_bytes())


def check(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def verify(root: Path) -> dict:
    campaigns = []
    all_ids = set()
    for plan_name, summary_name, evaluation_dir, audit_dir in CAMPAIGNS:
        plan_path, summary_path = root / plan_name, root / summary_name
        plan, summary = read_json(plan_path), read_json(summary_path)
        plan_sha, _ = digest(plan_path)
        summary_sha, _ = digest(summary_path)
        check(summary["plan_sha256"] == plan_sha, f"plan hash: {plan_name}")
        check(summary["status"] == "completed" and summary["all_required_sealed_streams_match"],
              f"summary status: {summary_name}")
        check(summary["reporting_scope"] == "car_only" and not summary["paper_eligible"],
              f"summary scope: {summary_name}")
        expected = {item["run_id"] for item in plan["runs"]}
        check(expected == set(summary["runs"]) and not (all_ids & expected),
              f"run membership: {summary_name}")
        all_ids |= expected
        runs = []
        for item in plan["runs"]:
            run_id = item["run_id"]
            run = summary["runs"][run_id]
            check((run["run_id"], run["seed"], run["mode"]) ==
                  (run_id, item["seed"], item["mode"]), f"run identity: {run_id}")
            check(run["frames"] == 3316 and run["sequences"] == 21 and run["weights_unchanged"],
                  f"run coverage: {run_id}")
            if run["mode"] == "M0":
                check(run["sealed_prediction_parity"] and run["sealed_association_parity"],
                      f"reported sealed parity: {run_id}")
            if run["mode"] == "M4":
                check(run["reference_M0_association_parity"], f"M4 association parity: {run_id}")
            base = root / evaluation_dir / run_id
            report_path = base / "report.json"
            report = read_json(report_path)
            coverage = report["coverage"]
            check(report["status"] == "completed" and not report["paper_eligible"],
                  f"evaluation status: {run_id}")
            check(report["reporting_scope"] == "car_only" and report["no_test_payload"]
                  and report["no_model_selection"] and report["no_validation_parameter_fitting"],
                  f"evaluation boundaries: {run_id}")
            check(report["predictions_sha256"] == run["predictions_sha256"],
                  f"prediction identity: {run_id}")
            check((coverage["frames"], coverage["sequences"], coverage["all_input_predictions"]) ==
                  (3316, 21, run["predictions"]), f"evaluation coverage: {run_id}")
            check(coverage["final_sequence_commit_sha256"] == run["sequence_commits"],
                  f"sequence commits: {run_id}")
            local_files = {}
            for filename, identity in report["files"].items():
                actual_sha, actual_size = digest(base / filename)
                check((actual_sha, actual_size) == (identity["sha256"], identity["size"]),
                      f"local evaluation file: {run_id}/{filename}")
                local_files[filename] = actual_sha
            runs.append({"run_id": run_id, "mode": run["mode"], "seed": run["seed"],
                         "frames": 3316, "sequences": 21,
                         "predictions_sha256_reported": run["predictions_sha256"],
                         "association_sha256_reported": run["association_sha256"],
                         "evaluation_report_sha256": digest(report_path)[0],
                         "local_evaluation_file_sha256": local_files})
        audit_root = root / audit_dir
        completion_path = audit_root / "completion.json"
        completion = read_json(completion_path)
        audit_report = read_json(audit_root / "report.json")
        check(completion["status"] == audit_report["status"] == "completed"
              and not audit_report["paper_eligible"], f"event audit status: {audit_dir}")
        check(not audit_report["causal_effect_identified"]
              and not audit_report["gt_full_text_exported"]
              and not audit_report["prediction_streams_exported"],
              f"event audit boundary: {audit_dir}")
        audit_files = {}
        for filename, identity in completion["files"].items():
            actual_sha, actual_size = digest(audit_root / filename)
            check((actual_sha, actual_size) == (identity["sha256"], identity["size_bytes"]),
                  f"event audit file: {audit_dir}/{filename}")
            audit_files[filename] = actual_sha
        campaigns.append({"plan": str(plan_path), "plan_sha256": plan_sha,
                          "summary": str(summary_path), "summary_sha256": summary_sha,
                          "run_count": len(runs), "runs": runs,
                          "event_audit_dir": str(audit_root),
                          "event_audit_completion_sha256": digest(completion_path)[0],
                          "event_audit_files_sha256": audit_files})
    check(len(all_ids) == 13 and sum(x["run_count"] for x in campaigns) == 13,
          "Stage 1 run count")
    check({r["mode"] for c in campaigns for r in c["runs"]} ==
          {"M0", "M1", "M2", "M3", "M4"}, "Stage 1 mode coverage")
    return {"kind": "recover_before_fuse_stage1_legacy_local_evidence_audit_v1",
            "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "local_metadata_and_evaluation_files_verified",
            "campaigns": campaigns, "run_count": 13,
            "raw_prediction_streams_independently_read_back": False,
            "clearml_task_ids_resolved": False,
            "causal_effect_identified": False,
            "paper_eligible": False,
            "full_recover_before_fuse_complete": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(f"verified {result['run_count']} historical runs; receipt={args.receipt}")


if __name__ == "__main__":
    main()
