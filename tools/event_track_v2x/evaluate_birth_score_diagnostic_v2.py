#!/usr/bin/env python3
"""Sealed car evaluation of M4 birth-score-only runs, versus paired M0/M3.

Three CPU-only evaluations at most. No model training, predictor GT access,
parameter selection or external publication. M4 changes birth acceptance and
initial score, so later free-running states may differ. M3 minus M4 is not an
additive causal decomposition or a direct temporal-assignment effect.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "tools/event_track_v2x/evaluate_mechanism_diagnostics_v2.py"
BASE_SHA256 = "01a7e55b1b7a523c1344f195b1343b7a403837e8df360cf6d57580130e045ea7"
SOURCE_EVALUATOR = ROOT / "tools/event_track_v2x/evaluate_source_ablation_v2.py"
SOURCE_EVALUATOR_SHA256 = "a9e80719682477c3f1a4ab6030d906d20f848d2584578e3fb8dfe45c6f399356"
SEEDS = (1337, 2027, 3407)
RUNS = [{"run_id": f"M4-seed-{s}", "mode": "M4", "seed": s, "deterministic_control": False} for s in SEEDS]
BIRTH_SOURCE_PATHS = ("tools/event_track_v2x/run_birth_score_diagnostic_v2.py",
                      "transvision/models/event_track_v2x/tracking_birth_score_v2.py")
BIRTH_SOURCE_PINS = {
    "tools/event_track_v2x/run_birth_score_diagnostic_v2.py": "708af5d7aa42c108bdfe9cd6bca050c29cc9d351034938b27fa0866a56b34cc5",
    "transvision/models/event_track_v2x/tracking_birth_score_v2.py": "7f0f2748b72063c5ff255cdb8a27dab548badbcfc436ddf874c333fc9d4479a6",
}
COMMON_INPUT_KEYS = ("cache_sha256", "schedule_sha256", "split_sha256", "calibration_sha256",
                     "checkpoints", "config", "candidate_selection", "condition", "torch_version")
INTERPRETATION = {
    "M4": "road residual original score affects only birth acceptance and initial track score; common node score, pre-existing-track temporal cost and update expressions are unchanged",
    "free_running": "birth changes later populations and states; whole-sequence temporal-output equality is neither required nor claimed",
    "M3": "raw residual common score affects temporal assignment, updates, pruning and births",
    "M3_minus_M4": "descriptive paired contrast only; not an additive mediation decomposition or isolated direct temporal effect",
    "statistics": "all three fixed seeds; mean and sample SD are descriptive, not confidence intervals; no seed selection or tuning",
    "units": "absolute metric differences, not percentages or relative improvement",
    "events": "detailed M4 birth/node provenance remains in immutable diagnostics; a separate evaluator-only event audit is required",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def evidence(path):
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise ValueError("regular non-symlink evidence required: " + str(p))
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size}


def read_json(path):
    evidence(path)
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def load_base():
    if evidence(BASE_PATH)["sha256"] != BASE_SHA256 or evidence(SOURCE_EVALUATOR)["sha256"] != SOURCE_EVALUATOR_SHA256:
        raise ValueError("sealed evaluation source changed")
    spec = importlib.util.spec_from_file_location("sealed_birth_control_evaluation", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protect_output(output, *inputs):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("output is create-once; inspect prior results before retry")
    destination = output.resolve()
    for path in inputs:
        path = Path(path)
        if path.is_symlink():
            raise ValueError("input-root symlinks are forbidden")
        if destination == path.resolve() or path.resolve() in destination.parents:
            raise ValueError("output must not mutate an immutable input tree")


def source_snapshot():
    return {"controller": evidence(__file__), "comparison_helpers": evidence(BASE_PATH),
            "source_evaluator": evidence(SOURCE_EVALUATOR)}


def validate_birth_experiment(root, mechanism_root, expected_plan_sha256, base):
    root, mechanism_root = Path(root), Path(mechanism_root)
    if root.is_symlink() or mechanism_root.is_symlink():
        raise ValueError("input-root symlinks are forbidden")
    old_plan, old_receipts, old_sources = base.validate_experiment(mechanism_root)
    plan, summary = read_json(root / "frozen-plan.json"), read_json(root / "summary.json")
    root_sources = {name: evidence(root / name) for name in ("frozen-plan.json", "summary.json")}
    if (root_sources["frozen-plan.json"]["sha256"] != expected_plan_sha256
            or plan.get("kind") != "birth_score_diagnostics_v2_plan" or plan.get("runs") != RUNS
            or plan.get("reporting_scope") != "car_only" or plan.get("paper_eligible") is not False
            or plan.get("gt_model_inputs") is not False or plan.get("test_payloads_read") is not False
            or plan.get("val_parameter_fitting") is not False or plan.get("seed_selection") is not False
            or plan.get("official_validation_frames") != 3316 or plan.get("sequences") != 21
            or summary.get("kind") != "birth_score_diagnostics_v2_complete" or summary.get("status") != "completed"
            or summary.get("plan_sha256") != expected_plan_sha256 or summary.get("reporting_scope") != "car_only"
            or summary.get("paper_eligible") is not False or summary.get("weights_unchanged") is not True
            or summary.get("all_required_sealed_streams_match") is not True
            or set(summary.get("runs", {})) != {r["run_id"] for r in RUNS}):
        raise ValueError("incomplete or incompatible three-run M4 experiment")
    if (Path(plan["reference_mechanism_root"]).resolve() != mechanism_root.resolve()
            or plan["reference_mechanism_plan_sha256"] != old_sources["frozen-plan.json"]["sha256"]
            or plan["reference_mechanism_summary_sha256"] != old_sources["summary.json"]["sha256"]):
        raise ValueError("M4 reference mechanism root/plan/summary mismatch")
    for key in COMMON_INPUT_KEYS:
        if key not in plan or key not in old_plan or plan[key] != old_plan[key]:
            raise ValueError("M4 and M0/M3 frozen input/config differs: " + key)
    if not set(BIRTH_SOURCE_PATHS) <= set(plan["current_source_hashes"]):
        raise ValueError("M4 implementation sources missing from frozen plan")
    for name, wanted in plan["current_source_hashes"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or evidence(ROOT / name)["sha256"] != wanted:
            raise ValueError("M4 source dependency changed: " + name)
        if name in BIRTH_SOURCE_PINS and wanted != BIRTH_SOURCE_PINS[name]:
            raise ValueError("M4 prediction implementation pin mismatch")
    receipts = {}
    for spec in RUNS:
        name, seed = spec["run_id"], spec["seed"]
        folder = root / name
        if folder.is_symlink():
            raise ValueError("symlink M4 run directory")
        receipt = read_json(folder / "receipt.json")
        baseline = old_receipts[f"M0-seed-{seed}"]
        common = {filename: evidence(folder / filename) for filename in
                  ("predictions.jsonl", "association.jsonl", "diagnostics.jsonl.gz", "receipt.json")}
        if (receipt != summary["runs"][name] or any(receipt.get(k) != v for k, v in spec.items())
                or receipt.get("frames") != 3316 or receipt.get("sequences") != 21
                or receipt.get("weights_unchanged") is not True or receipt.get("sealed_association_parity") is not True
                or receipt.get("reference_M0_association_parity") is not True
                or "sealed_prediction_parity" not in receipt or receipt["sealed_prediction_parity"] is not None
                or len(receipt["sequence_commits"]) != 21 or len(receipt["diagnostic_sequence_commits"]) != 21
                or set(receipt["sequence_commits"]) != set(baseline["sequence_commits"])
                or set(receipt["diagnostic_sequence_commits"]) != set(baseline["sequence_commits"])):
            raise ValueError("M4 runner receipt mismatch: " + name)
        for key in ("predictions", "association"):
            if common[key + ".jsonl"]["sha256"] != receipt[key + "_sha256"]:
                raise ValueError("M4 stream hash mismatch: " + name)
        if (common["diagnostics.jsonl.gz"]["sha256"] != receipt["diagnostics_gzip_sha256"]
                or common["diagnostics.jsonl.gz"]["size"] != receipt["diagnostics_gzip_bytes"]):
            raise ValueError("M4 diagnostic hash/size mismatch")
        for key in ("selected_detections", "source_sensor_late"):
            if receipt.get(key) != baseline.get(key) or key not in receipt:
                raise ValueError("M4 source availability/candidate counts differ")
        if (receipt["association_sha256"] != baseline["association_sha256"]
                or receipt["association_sha256"] != old_receipts[f"M3-seed-{seed}"]["association_sha256"]):
            raise ValueError("M4 association differs from same-seed M0/M3")
        receipts[name], root_sources[name] = receipt, common
    return plan, receipts, root_sources, old_receipts, old_sources


def reference_reports(reports_root, old_receipts, old_sources, base, evaluator):
    root = Path(reports_root)
    if root.is_symlink():
        raise ValueError("reference report root symlink")
    comparison_path, manifest_path = root / "comparison/comparison.json", root / "comparison/source-manifest.json"
    comparison, manifest = read_json(comparison_path), read_json(manifest_path)
    if (comparison.get("kind") != "mechanism_diagnostics_car_comparison_v1"
            or comparison.get("status") != "completed" or comparison.get("reporting_scope") != "car_only"
            or comparison.get("paper_eligible") is not False
            or comparison.get("source_manifest_sha256") != evidence(manifest_path)["sha256"]
            or manifest["prediction_experiment"] != old_sources
            or set(comparison["run_metrics"]) != set(old_receipts)):
        raise ValueError("reference mechanism comparison/source-manifest mismatch")
    references, sources = {}, {"comparison": evidence(comparison_path), "source_manifest": evidence(manifest_path), "runs": {}}
    for mode in ("M0", "M3"):
        for seed in SEEDS:
            name = f"{mode}-seed-{seed}"
            report, vector, sequences, files = base.validate_report(root / name, evaluator, old_receipts[name])
            if (files != manifest["evaluation_reports"][name]
                    or report["input_sources"]["predictions"] != old_sources[name]["predictions.jsonl"]):
                raise ValueError("reference metric reports do not bind prediction evidence")
            base.close_vectors(vector, comparison["run_metrics"][name])
            references[name] = {"metrics": vector, "sequences": sequences}
            sources["runs"][name] = files
    return references, sources


def remember(snapshot, root, evidence_map):
    """Flatten runner inventories into exact read-only paths for final recheck."""
    for name, entry in evidence_map.items():
        if "sha256" in entry:
            snapshot[Path(root) / name] = entry
        else:
            for filename, value in entry.items():
                snapshot[Path(root) / name / filename] = value


def compare(root, reports_root, mechanism_root, mechanism_reports_root, expected_plan_sha256, output, *, expected_inputs=None):
    root, reports_root, mechanism_root, mechanism_reports_root, output = map(
        Path, (root, reports_root, mechanism_root, mechanism_reports_root, output))
    protect_output(output, root, mechanism_root, mechanism_reports_root)
    base = load_base(); evaluator = base.load_evaluator()
    code_sources = source_snapshot()
    _, receipts, runner_sources, old_receipts, old_sources = validate_birth_experiment(root, mechanism_root, expected_plan_sha256, base)
    references, reference_sources = reference_reports(mechanism_reports_root, old_receipts, old_sources, base, evaluator)
    if expected_inputs is not None and (
            runner_sources != expected_inputs["runner_sources"] or old_sources != expected_inputs["mechanism_sources"]
            or reference_sources != expected_inputs["reference_reports"] or code_sources != expected_inputs["evaluation_sources"]):
        raise ValueError("evaluation inputs changed since frozen execution plan")
    vectors, sequence_vectors, report_sources, lifecycle = {}, {}, {}, {}
    snapshot = {}
    remember(snapshot, root, runner_sources); remember(snapshot, mechanism_root, old_sources)
    snapshot[mechanism_reports_root / "comparison/comparison.json"] = reference_sources["comparison"]
    snapshot[mechanism_reports_root / "comparison/source-manifest.json"] = reference_sources["source_manifest"]
    for name, files in reference_sources["runs"].items():
        for filename, value in files.items():
            snapshot[mechanism_reports_root / name / filename] = value
    for spec in RUNS:
        name = spec["run_id"]
        report, vectors[name], sequence_vectors[name], files = base.validate_report(reports_root / name, evaluator, receipts[name])
        if report["input_sources"]["predictions"] != runner_sources[name]["predictions.jsonl"]:
            raise ValueError("M4 report prediction hash/size differs from runner")
        report_sources[name] = files
        for filename, value in files.items():
            snapshot[reports_root / name / filename] = value
        lifecycle[name] = {key: receipts[name][key] for key in ("events", "nodes", "temporal_matches", "predictions")}
        lifecycle[name]["scope"] = "all-class engineering counters; not car event counts or a mechanism causal decomposition"
    contrasts = {name: {} for name in ("M4_minus_same_seed_M0", "M4_minus_same_seed_M3", "M3_minus_same_seed_M4")}
    for seed in SEEDS:
        name = f"M4-seed-{seed}"
        for mode in ("M0", "M3"):
            reference = references[f"{mode}-seed-{seed}"]
            contrasts[f"M4_minus_same_seed_{mode}"][str(seed)] = base.contrast(
                vectors[name], reference["metrics"], sequence_vectors[name], reference["sequences"])
        reference = references[f"M3-seed-{seed}"]
        contrasts["M3_minus_same_seed_M4"][str(seed)] = base.contrast(
            reference["metrics"], vectors[name], reference["sequences"], sequence_vectors[name])
    summaries = {name: base.descriptives([v["metric_difference"] for v in values.values()]) for name, values in contrasts.items()}
    sources = {"M4_prediction_experiment": runner_sources, "M0_M3_prediction_experiment": old_sources,
               "M4_evaluation_reports": report_sources, "M0_M3_reference_reports": reference_sources, "evaluation_sources": code_sources}
    result = {"kind": "birth_score_diagnostics_car_comparison_v1", "status": "completed", "reporting_scope": "car_only",
              "coverage_per_run": {"frames": 3316, "sequences": 21}, "run_metrics": vectors,
              "M4_descriptive": base.descriptives(list(vectors.values())), "paired_contrasts": contrasts,
              "paired_difference_descriptive": summaries, "interpretation": INTERPRETATION,
              "lifecycle_counters": lifecycle, "independent_event_audit_required": True,
              "event_audit_contract": {"diagnostic_kind": "tracking_birth_score_v2_frame", "mode": "M4",
                  "common_node_score": "nodes[].score", "birth_score": "nodes[].birth_effective_score",
                  "birth_event_fields": ["score", "node_score", "birth_effective_score"],
                  "diagnostics_bound_by_runner_hash": True, "event_semantics_verified_here": False,
                  "future_gt_mapping_must_remain_evaluator_only": True},
              "metric_direction": {k: "lower_is_better" if k in ("AMOTP_m", "IDS", "Frag", "FP", "FN") else "higher_is_better"
                                   for k in base.METRICS}, "causal_additivity_assumed": False,
              "direct_temporal_effect_identified": False, "paper_eligible": False,
              "M4_plan_sha256": expected_plan_sha256,
              "source_manifest_sha256": hashlib.sha256(canonical(sources) + b"\n").hexdigest()}
    if source_snapshot() != code_sources or any(evidence(path) != before for path, before in snapshot.items()):
        raise ValueError("comparison evidence/source changed during computation")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "source-manifest.json", sources)
    write_json(output / "comparison.json", result)
    rows = ["# M4 birth-only car 对照", "", "M4 仅改变路端残余节点的 birth 接受分数和初始轨迹分数。",
            "后续 free-running 轨迹可因新生人口不同而分化；M3−M4 不代表可加分解或直接 temporal 效应。", "",
            "| seed | HOTA | IDF1 | AMOTA | ΔHOTA vs M0 | ΔHOTA vs M3 |", "|---|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        v = vectors[f"M4-seed-{seed}"]
        delta0 = contrasts["M4_minus_same_seed_M0"][str(seed)]["metric_difference"]["HOTA"]
        delta3 = contrasts["M4_minus_same_seed_M3"][str(seed)]["metric_difference"]["HOTA"]
        rows.append(f"| {seed} | {v['HOTA']:.6f} | {v['IDF1']:.6f} | {v['AMOTA']:.6f} | {delta0:+.6f} | {delta3:+.6f} |")
    rows += ["", "全部种子、11 项指标绝对差值及21序列配对结果见 comparison.json。",
             "样本标准差仅作描述，不是置信区间；未做参数选择、最优seed选择或因果效应识别。",
             "GT 只进入既有独立 car 评价器，不进入 predictor。逐节点 birth 事件须另行审计。", ""]
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(rows))
    return result


def worker(args):
    base = load_base()
    base.load_evaluator().evaluate(args.ground_truth, args.predictions, args.output)


def run(args):
    protect_output(args.output, args.root, args.mechanism_root, args.mechanism_reports_root, args.ground_truth)
    if type(args.max_workers) is not int or not 1 <= args.max_workers <= 3:
        raise ValueError("max-workers must be an integer between 1 and 3")
    base = load_base(); evaluator = base.load_evaluator()
    _, _, runner_sources, old_receipts, old_sources = validate_birth_experiment(
        args.root, args.mechanism_root, args.expected_plan_sha256, base)
    _, reference_sources = reference_reports(args.mechanism_reports_root, old_receipts, old_sources, base, evaluator)
    gt_sources = {name: evidence(args.ground_truth / name) for name in ("manifest.json", "ground-truth.jsonl")}
    if gt_sources["manifest.json"]["sha256"] != evaluator.GT_MANIFEST_SHA256 or gt_sources["ground-truth.jsonl"]["sha256"] != evaluator.GT_SHA256:
        raise ValueError("not the sealed official-validation ground truth")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    commands = {r["run_id"]: [sys.executable, str(Path(__file__).resolve()), "evaluate-one", "--ground-truth",
                str(args.ground_truth.resolve()), "--predictions", str((args.root / r["run_id"] / "predictions.jsonl").resolve()),
                "--output", str((args.output / r["run_id"]).resolve())] for r in RUNS}
    frozen = {"kind": "birth_score_car_evaluation_execution_plan_v1", "runs": RUNS, "max_workers": args.max_workers,
              "threads_per_worker": 1, "gpu_used": False, "M4_plan_sha256": args.expected_plan_sha256,
              "runner_sources": runner_sources, "mechanism_sources": old_sources, "reference_reports": reference_sources,
              "ground_truth": gt_sources, "evaluation_sources": source_snapshot(), "commands": commands,
              "reporting_scope": "car_only", "paper_eligible": False}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    write_json(args.output / "frozen-evaluation-plan.json", frozen)
    def launch(name):
        with (args.output / "logs" / f"{name}.log").open("xb") as stream:
            result = subprocess.run(commands[name], env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        return name, result.returncode
    started = time.monotonic(); codes = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        jobs = [pool.submit(launch, r["run_id"]) for r in RUNS]
        for job in as_completed(jobs):
            name, status = job.result(); codes[name] = status
            print("BIRTH_SCORE_EVALUATED " + json.dumps({"run_id": name, "exit_code": status}), flush=True)
    execution = {"kind": "birth_score_car_evaluation_execution_v1", "exit_codes": codes,
                 "elapsed_seconds": time.monotonic() - started,
                 "plan_sha256": evidence(args.output / "frozen-evaluation-plan.json")["sha256"],
                 "status": "completed" if all(code == 0 for code in codes.values()) else "failed"}
    write_json(args.output / "execution.json", execution)
    if any(codes.values()):
        raise RuntimeError("one or more M4 car evaluations failed; partial reports and logs retained")
    if source_snapshot() != frozen["evaluation_sources"]:
        raise ValueError("evaluation source changed during execution")
    if {name: evidence(args.ground_truth / name) for name in gt_sources} != gt_sources:
        raise ValueError("evaluation ground truth changed during execution")
    compare(args.root, args.output, args.mechanism_root, args.mechanism_reports_root,
            args.expected_plan_sha256, args.output / "comparison", expected_inputs=frozen)
    print("BIRTH_SCORE_EVALUATION_COMPLETE " + json.dumps({"runs": 3,
          "comparison_sha256": evidence(args.output / "comparison/comparison.json")["sha256"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for command in ("run", "compare"):
        sub = commands.add_parser(command)
        for name in ("root", "mechanism-root", "mechanism-reports-root", "output"):
            sub.add_argument("--" + name, required=True, type=Path)
        sub.add_argument("--expected-plan-sha256", required=True)
        if command == "run":
            sub.add_argument("--ground-truth", required=True, type=Path)
            sub.add_argument("--max-workers", type=int, choices=(1, 2, 3), default=3)
        else:
            sub.add_argument("--reports-root", required=True, type=Path)
    single = commands.add_parser("evaluate-one")
    for name in ("ground-truth", "predictions", "output"):
        single.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "compare":
        compare(args.root, args.reports_root, args.mechanism_root, args.mechanism_reports_root,
                args.expected_plan_sha256, args.output)
    else:
        worker(args)


if __name__ == "__main__":
    main()
