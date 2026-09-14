#!/usr/bin/env python3
"""Prepare evaluator-only SPD val GT or run the pinned official metric engines."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

# Deliberately load only the evaluator adapter: preparing GT must not import
# tracker/cache modules or require a GPU/training environment through __init__.
_path = Path(__file__).resolve().parents[2] / "transvision/models/event_track_v2x/tracking_evaluation_v2.py"
_spec = importlib.util.spec_from_file_location("eventtrack_tracking_evaluation_v2", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
evaluate, golden_cases = _module.evaluate, _module.golden_cases
prepare_ground_truth, runtime_evidence, write_json = _module.prepare_ground_truth, _module.runtime_evidence, _module.write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-gt")
    prep.add_argument("--archive", required=True)
    prep.add_argument("--split", required=True)
    prep.add_argument("--vehicle-infos", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--schedule-output", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--ground-truth", required=True)
    run.add_argument("--predictions", required=True)
    run.add_argument("--output", required=True)
    golden = sub.add_parser("golden")
    golden.add_argument("--output", required=True)
    a = p.parse_args()
    if a.command == "prepare-gt":
        result = prepare_ground_truth(a.archive, a.split, a.vehicle_infos, a.output, a.schedule_output)
        print(json.dumps({k: result[k] for k in ("frames", "sequences", "ground_truth_sha256", "prediction_schedule_sha256", "roi_category_counts", "duplicate_ids")}))
    elif a.command == "golden":
        result = {"runtime": runtime_evidence(), "golden_cases": golden_cases()}
        write_json(a.output, result)
        print(json.dumps({"passed": result["golden_cases"]["passed"]}))
    else:
        result = evaluate(a.ground_truth, a.predictions, a.output)
        print(json.dumps({k: result[k] for k in ("status", "coverage", "primary_car")}))


if __name__ == "__main__":
    main()
