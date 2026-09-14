#!/usr/bin/env python3
"""Car-only source ablation with the sealed official tracking metric adapter.

This does not alter detector selection, tracker inputs or the sealed evaluator.
All conditions retain the vehicle reference clock/pose and the same native
official-validation cohort. Comparison is descriptive, without seed selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "transvision/models/event_track_v2x/tracking_evaluation_v2.py"
ADAPTER_SHA256 = "659fbf0cafad0e942eb2bb72023f3a4b9cc4c4cfca6883503b48f3c5c3c9b7ec"
GT_MANIFEST_SHA256 = "94675ac8585d893195b7e801b754f62ca4ee3524cf852e05e53121fe2cc4084a"
GT_SHA256 = "94908e0010003b42a5ed2c35ccc894005fe8d40cadfe57653871bbd95e0a9a76"
RUNTIME_VERSIONS = {"motmetrics": "1.4.0", "numpy": "1.26.4", "nuscenes-devkit": "1.2.0",
                    "pandas": "2.3.3", "pyquaternion": "0.9.9", "scipy": "1.15.3",
                    "shapely": "2.0.7", "trackeval": "1.0.0"}
RUNTIME_TREES = {"motmetrics": "d4a797e97023cb7147a98484030b454283f00e7e56a0bcd3ec737b1ee229c7e8",
                 "nuscenes": "49a8f50dcaae5cee5f025eafceab216a4e77d53771befb05a9624e7955af9d5a",
                 "trackeval": "90a4c6beb47735e74f1b785b55456f8e746daa938671dc9d3430f75cb96cbbf1"}
RUNS = {"vehicle-only": (1, 1337, True), "infrastructure-only": (2, 1337, True),
        **{f"cooperative-seed-{s}": (3, s, False) for s in (1337, 2027, 3407)}}
PRIMARY_METRICS = {"AMOTA": ("nuscenes", "amota"), "AMOTP_m": ("nuscenes", "amotp"),
                   "MOTA": ("nuscenes", "mota"), "HOTA": ("trackeval", "HOTA"),
                   "AssA": ("trackeval", "AssA"), "DetA": ("trackeval", "DetA"),
                   "IDF1": ("trackeval", "IDF1"), "IDS": ("nuscenes", "ids"),
                   "Frag": ("nuscenes", "frag")}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evidence(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"evidence must be a regular non-symlink file: {path}")
    return {"sha256": sha(path), "size": path.stat().st_size}


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def read_json(path):
    return json.loads(Path(path).read_bytes())


def load_adapter():
    if sha(ADAPTER_PATH) != ADAPTER_SHA256:
        raise ValueError("sealed tracking adapter source hash mismatch")
    spec = importlib.util.spec_from_file_location("sealed_source_ablation_evaluator", ADAPTER_PATH)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def protocol(adapter):
    result = copy.deepcopy(adapter.PROTOCOL)
    result.update(kind="spd_native_source_ablation_car_protocol_v1", evaluated_classes=["car"],
                  supplementary_classes=[], reporting_scope="car_only",
                  input_candidate_selection="unchanged all-class raw_score>=0.05 top64; car-only metrics",
                  source_ablation_reference="vehicle clock, box reference time and ego pose for all conditions; infrastructure-only is not autonomous infrastructure tracking")
    result["roi"]["class_range_m"] = {"car": 50.}
    return result


def validate_runtime(runtime):
    trees = {k: v["source_tree_sha256"] for k, v in runtime["source_trees"].items()}
    if (runtime["versions"] != RUNTIME_VERSIONS or trees != RUNTIME_TREES
            or runtime["adapter_source_sha256"] != ADAPTER_SHA256):
        raise ValueError("runtime differs from sealed official-evaluation source and version pins")
    for name, tree in runtime["source_trees"].items():
        if hashlib.sha256(canonical(tree["files"])).hexdigest() != RUNTIME_TREES[name]:
            raise ValueError("runtime source inventory hash mismatch")


def car_metrics(metrics):
    if set(metrics["trackeval"]) != {"car"}:
        raise ValueError("evaluation must contain car metrics only")
    labels = metrics["nuscenes"]["label_metrics"]
    if any(set(values) != {"car"} for values in labels.values()):
        raise ValueError("nuScenes labels must contain car only")
    return {"nuscenes": {k: v["car"] for k, v in labels.items()},
            "trackeval": metrics["trackeval"]["car"]["summary"]}


def evaluate(ground_truth, predictions_path, output):
    output, ground_truth, predictions_path = Path(output), Path(ground_truth), Path(predictions_path)
    if output.exists() or output.is_symlink():
        raise ValueError("evaluation output already exists; create-once evidence cannot be overwritten")
    adapter = load_adapter()
    inputs = {"ground_truth_manifest": evidence(ground_truth / "manifest.json"),
              "ground_truth": evidence(ground_truth / "ground-truth.jsonl"),
              "predictions": evidence(predictions_path), "sealed_adapter": evidence(ADAPTER_PATH),
              "evaluation_cli": evidence(__file__)}
    if (inputs["ground_truth_manifest"]["sha256"] != GT_MANIFEST_SHA256
            or inputs["ground_truth"]["sha256"] != GT_SHA256):
        raise ValueError("ground truth differs from the sealed official-validation cohort")
    runtime = adapter.runtime_evidence()
    validate_runtime(runtime)
    gold = adapter.golden_cases()  # The sealed function explicitly uses ("car",).
    if gold.get("passed") is not True:
        raise ValueError("official car golden cases failed")
    manifest, gt = adapter.load_ground_truth(ground_truth)
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    original_coverage = adapter.validate_predictions(predictions, gt)
    if original_coverage["frames"] != 3316 or original_coverage["sequences"] != 21:
        raise ValueError("complete official-validation coverage is required")
    metrics = adapter.compute_metrics(gt, predictions, classes=("car",))
    primary = car_metrics(metrics)
    coverage = {k: v for k, v in original_coverage.items() if k != "predictions_per_class"}
    coverage.update(car_predictions=original_coverage["predictions_per_class"].get("car", 0),
                    all_input_predictions=sum(original_coverage["predictions_per_class"].values()),
                    all_input_classes_validated=True)
    if (evidence(predictions_path) != inputs["predictions"]
            or evidence(ground_truth / "manifest.json") != inputs["ground_truth_manifest"]
            or evidence(ground_truth / "ground-truth.jsonl") != inputs["ground_truth"]):
        raise ValueError("evaluation input changed during computation")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("metrics.json", metrics), ("runtime.json", runtime), ("golden-cases.json", gold)):
        write_json(output / name, value)
    selected_protocol = protocol(adapter)
    report = {"kind": "source_ablation_car_evaluation_v1", "status": "completed",
              "protocol": selected_protocol,
              "protocol_sha256": hashlib.sha256(canonical(selected_protocol)).hexdigest(),
              "sealed_protocol_sha256": hashlib.sha256(canonical(adapter.PROTOCOL)).hexdigest(),
              "ground_truth_manifest_sha256": inputs["ground_truth_manifest"]["sha256"],
              "ground_truth_sha256": manifest["ground_truth_sha256"],
              "predictions_sha256": inputs["predictions"]["sha256"], "input_sources": inputs,
              "coverage": coverage, "primary_car": primary,
              "files": {p.name: evidence(p) for p in sorted(output.iterdir())},
              "reporting_scope": "car_only", "no_test_payload": True,
              "no_model_selection": True, "no_validation_parameter_fitting": True, "paper_eligible": False}
    write_json(output / "report.json", report)
    return report


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("missing or non-finite metric; do not replace it with zero")
    return value


def metric_vector(primary):
    return {k: _finite(primary[group][name]) for k, (group, name) in PRIMARY_METRICS.items()}


def sequence_vectors(metrics):
    result = {}
    for sid, sequence in metrics["trackeval"]["car"]["sequences"].items():
        result[sid] = {k: statistics.fmean(_finite(v) for v in sequence["HOTA"][k])
                       for k in ("HOTA", "AssA", "DetA")}
        result[sid]["IDF1"] = _finite(sequence["Identity"]["IDF1"])
    if len(result) != 21:
        raise ValueError("per-sequence metric coverage must be 21")
    return result


def compare(experiment_root, reports_root, output):
    root, reports_root, output = Path(experiment_root), Path(reports_root), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("comparison output already exists; create-once evidence cannot be overwritten")
    adapter = load_adapter()
    selected_protocol = protocol(adapter)
    summary, plan = read_json(root / "summary.json"), read_json(root / "frozen-plan.json")
    if (summary["kind"] != "source_ablation_v2_complete" or summary["status"] != "completed"
            or summary["reporting_scope"] != "car_only" or summary["paper_eligible"] is not False
            or summary["plan_sha256"] != sha(root / "frozen-plan.json")
            or summary["weights_unchanged"] is not True
            or summary["all_cooperative_streams_match_sealed"] is not True
            or set(summary["runs"]) != set(RUNS)
            or plan["kind"] != "source_ablation_v2_plan" or plan["reporting_scope"] != "car_only"):
        raise ValueError("incomplete or incompatible source-ablation runner evidence")
    expected_runs = [{"run_id": name, "agent_mask": mask, "seed": seed, "deterministic_control": deterministic}
                     for name, (mask, seed, deterministic) in RUNS.items()]
    if sorted(plan["runs"], key=lambda r: r["run_id"]) != sorted(expected_runs, key=lambda r: r["run_id"]):
        raise ValueError("frozen five-condition plan mismatch")
    sources = {"runner": {name: evidence(root / name) for name in ("summary.json", "frozen-plan.json")},
               "comparison_cli": evidence(__file__), "sealed_adapter": evidence(ADAPTER_PATH), "runs": {}}
    metrics_by_run, sequence_by_run = {}, {}
    cohort = None
    for run_id, (mask, seed, deterministic) in RUNS.items():
        run_root, report_root = root / run_id, reports_root / run_id
        receipt, report = read_json(run_root / "receipt.json"), read_json(report_root / "report.json")
        if (receipt != summary["runs"][run_id] or receipt["run_id"] != run_id
                or (receipt["agent_mask"], receipt["seed"], receipt["deterministic_control"]) != (mask, seed, deterministic)
                or receipt["frames"] != 3316 or receipt["sequences"] != 21
                or receipt["weights_unchanged"] is not True
                or (mask == 3 and receipt["sealed_cooperative_parity"] is not True)):
            raise ValueError(f"runner receipt mismatch: {run_id}")
        run_files = {name: evidence(run_root / name) for name in
                     ("predictions.jsonl", "association.jsonl", "control.jsonl", "receipt.json")}
        for key in ("predictions", "association", "control"):
            if run_files[f"{key}.jsonl"]["sha256"] != receipt[f"{key}_sha256"]:
                raise ValueError(f"runner stream hash mismatch: {run_id}/{key}")
        if (report["kind"] != "source_ablation_car_evaluation_v1" or report["status"] != "completed"
                or report["reporting_scope"] != "car_only" or report["protocol"] != selected_protocol
                or report["protocol_sha256"] != hashlib.sha256(canonical(selected_protocol)).hexdigest()
                or report["predictions_sha256"] != receipt["predictions_sha256"]
                or report["ground_truth_manifest_sha256"] != GT_MANIFEST_SHA256
                or report["ground_truth_sha256"] != GT_SHA256
                or report["no_model_selection"] is not True or report["paper_eligible"] is not False
                or report["no_test_payload"] is not True or report["no_validation_parameter_fitting"] is not True
                or report["sealed_protocol_sha256"] != hashlib.sha256(canonical(adapter.PROTOCOL)).hexdigest()
                or report["input_sources"]["predictions"] != run_files["predictions.jsonl"]
                or report["input_sources"]["sealed_adapter"] != evidence(ADAPTER_PATH)
                or report["input_sources"]["evaluation_cli"] != evidence(__file__)
                or report["input_sources"]["ground_truth_manifest"]["sha256"] != GT_MANIFEST_SHA256
                or report["input_sources"]["ground_truth"]["sha256"] != GT_SHA256
                or report["coverage"]["frames"] != 3316 or report["coverage"]["sequences"] != 21
                or report["coverage"]["final_sequence_commit_sha256"] != receipt["sequence_commits"]):
            raise ValueError(f"car evaluation does not bind the complete runner output: {run_id}")
        if set(report["files"]) != {"metrics.json", "runtime.json", "golden-cases.json"}:
            raise ValueError("unexpected evaluation artifact inventory")
        report_files = {name: evidence(report_root / name) for name in report["files"]}
        if report_files != report["files"]:
            raise ValueError(f"evaluation artifact hash or size mismatch: {run_id}")
        report_files["report.json"] = evidence(report_root / "report.json")
        validate_runtime(read_json(report_root / "runtime.json"))
        if read_json(report_root / "golden-cases.json").get("passed") is not True:
            raise ValueError("golden cases not passed")
        metrics = read_json(report_root / "metrics.json")
        if car_metrics(metrics) != report["primary_car"]:
            raise ValueError("reported car metrics differ from engine artifacts")
        metrics_by_run[run_id] = metric_vector(report["primary_car"])
        sequence_by_run[run_id] = sequence_vectors(metrics)
        if cohort is None:
            cohort = set(sequence_by_run[run_id])
        if set(sequence_by_run[run_id]) != cohort or cohort != set(receipt["sequence_commits"]):
            raise ValueError("paired sequence cohorts differ")
        sources["runs"][run_id] = {"runner_files": run_files, "evaluation_files": report_files}
    cooperative = [name for name in RUNS if name.startswith("cooperative-")]
    descriptive = {metric: {"mean": statistics.fmean(metrics_by_run[r][metric] for r in cooperative),
                            "sample_sd": statistics.stdev(metrics_by_run[r][metric] for r in cooperative), "n": 3}
                   for metric in PRIMARY_METRICS}
    differences = {}
    for run_id in cooperative:
        differences[run_id] = {}
        for baseline in ("vehicle-only", "infrastructure-only"):
            per_sequence = {sid: {metric: sequence_by_run[run_id][sid][metric] - sequence_by_run[baseline][sid][metric]
                                  for metric in ("HOTA", "AssA", "DetA", "IDF1")} for sid in sorted(cohort)}
            differences[run_id][baseline] = {
                "metric_difference": {k: metrics_by_run[run_id][k] - metrics_by_run[baseline][k] for k in PRIMARY_METRICS},
                "paired_sequence_differences": per_sequence,
                "paired_sequence_sign_counts": {k: {"positive": sum(r[k] > 0 for r in per_sequence.values()),
                                                       "zero": sum(r[k] == 0 for r in per_sequence.values()),
                                                       "negative": sum(r[k] < 0 for r in per_sequence.values())}
                                                for k in ("HOTA", "AssA", "DetA", "IDF1")}}
    comparison = {"kind": "source_ablation_car_comparison_v1", "status": "completed", "reporting_scope": "car_only",
                  "protocol": selected_protocol, "coverage_per_run": {"frames": 3316, "sequences": 21},
                  "run_metrics": metrics_by_run, "cooperative_three_seed_descriptive": descriptive,
                  "cooperative_minus_single_source": differences,
                  "metric_direction": {k: "lower_is_better" if k in ("AMOTP_m", "IDS", "Frag") else "higher_is_better"
                                       for k in PRIMARY_METRICS},
                  "difference_units": "absolute metric units; not percentages or relative improvement",
                  "single_source_repetitions": 1, "single_source_seed": "1337 only; cross-source association is bypassed by source mask",
                  "inference_boundary": "descriptive ablation on the same validation cohort; sample SD is not a confidence interval; no p-values, best-seed choice or validation fitting",
                  "independent_control_chain_audit_required": True, "paper_eligible": False,
                  "source_manifest_sha256": hashlib.sha256(canonical(sources) + b"\n").hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "source-manifest.json", sources)
    write_json(output / "comparison.json", comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("evaluate")
    for name in ("ground-truth", "predictions", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    comparison = sub.add_parser("compare")
    for name in ("experiment-root", "reports-root", "output"):
        comparison.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate(args.ground_truth, args.predictions, args.output)
        print(json.dumps({k: result[k] for k in ("status", "coverage", "primary_car")}))
    else:
        result = compare(args.experiment_root, args.reports_root, args.output)
        print(json.dumps({k: result[k] for k in ("status", "run_metrics", "cooperative_three_seed_descriptive")}))


if __name__ == "__main__":
    main()
