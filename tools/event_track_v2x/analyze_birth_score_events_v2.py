#!/usr/bin/env python3
"""Create-once, evaluator-only M4 car events with pinned M0/M3 comparisons.

Reuse the frozen event audit through an isolated function-global adapter. Never
rewrite its module, diagnostic evidence, common node score, or historical IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import platform
import statistics
from types import FunctionType

ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "tools/event_track_v2x/analyze_mechanism_events_v2.py"
CORE_SHA256 = "e3ad276c1443538f5b7b6b99dce2d7908295335ede9d143a9249f5294cd48963"
CONTROL_PATH = ROOT / "tools/event_track_v2x/evaluate_birth_score_diagnostic_v2.py"
CONTROL_SHA256 = "23844a66ca75c741064e30077b1dcd768852d4db29c823c709f9d58fec5f2e7d"
REFERENCE_FILES = {"report.json", "sources.json", "sequences.json", "identity-episodes.json",
                   "protocol.json", "README.md", "completion.json"}


def load_pinned(path, digest, name):
    if Path(path).is_symlink() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("frozen audit dependency changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    return (load_pinned(CORE_PATH, CORE_SHA256, "sealed_birth_event_core"),
            load_pinned(CONTROL_PATH, CONTROL_SHA256, "sealed_birth_event_inputs"))


def same_number(core, actual, expected):
    if not math.isclose(core.finite(actual), core.finite(expected), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("M4 birth/common score provenance mismatch")


def adapted_sequence(core, adapter, birth_score):
    """Return an isolated callable and M4-only counters; core globals stay intact."""
    if not 0 <= core.finite(birth_score) <= 1:
        raise ValueError("invalid frozen birth threshold")
    channel = Counter()

    def validate(p, a, d, g, mode, plan_sha, previous):
        # Validate the real M4 seal before constructing any compatibility view.
        if (mode != "M4" or d["mode"] != "M4" or d["kind"] != "tracking_birth_score_v2_frame"
                or d["previous_diagnostic_commit_sha256"] != previous
                or d["diagnostic_commit_sha256"] != hashlib.sha256(core.canonical(
                    {k: v for k, v in d.items() if k != "diagnostic_commit_sha256"})).hexdigest()):
            raise ValueError("M4 diagnostic identity or real commit-chain mismatch")
        view = dict(d, kind="tracking_mechanism_v2_frame", mode="M0")
        view["diagnostic_commit_sha256"] = hashlib.sha256(core.canonical(
            {k: v for k, v in view.items() if k != "diagnostic_commit_sha256"})).hexdigest()
        source = core.validate_frame(p, a, view, g, "M0", plan_sha, previous)
        # The view is never returned, exported, or installed into core globals.
        labels, _, _ = core.map_boxes([core.state_box(n, adapter) for n in d["nodes"]], g, adapter)
        for node, label in zip(d["nodes"], labels):
            score, effective = core.finite(node["score"]), core.finite(node["birth_effective_score"])
            if not (0 <= score <= 1 + 1e-10 and 0 <= effective <= 1 + 1e-10):
                raise ValueError("M4 node score outside probability-score range")
            if node["kind"] == "road_residual":
                record = source[(node["source_side"], node["source_selected_index"])]
                mass = sum(h["weight"] for h in a["hypotheses"]
                           if node["source_selected_index"] in h["unmatched_right"])
                if mass <= 0:
                    raise ValueError("road residual without retained unmatched mass")
                same_number(core, node["unmatched_mass"], mass)
                same_number(core, node["original_score"], record["score"])
                same_number(core, score, mass * record["score"])
                same_number(core, effective, record["score"])
            else:
                same_number(core, effective, score)
            if node["class_index"] == 0 and label["status"] != "outside_car_roi":
                channel["roi_car_nodes_measured"] += 1
                if node["kind"] == "road_residual":
                    channel["roi_residual_common_score_below_threshold"] += score < birth_score
                    channel["roi_residual_birth_effective_score_at_threshold"] += effective >= birth_score
                    raised = score < birth_score <= effective
                    channel["roi_residual_raised_birth_eligibility"] += raised
                    channel["roi_residual_raised_birth_eligibility_" + label["status"]] += raised
        seen = set()
        assigned = {e["node_index"] for e in d["temporal_assignments"]}
        for event in d["events"]:
            if event["event"] not in ("birth", "birth_rejected_score"):
                continue
            j = event["node_index"]
            if j in seen or j in assigned:
                raise ValueError("birth event duplicated or applied to assigned node")
            seen.add(j)
            node = d["nodes"][j]
            for key, expected in (("score", node["birth_effective_score"]),
                                  ("node_score", node["score"]),
                                  ("birth_effective_score", node["birth_effective_score"])):
                same_number(core, event[key], expected)
            accepted = event["event"] == "birth"
            if accepted != (node["birth_effective_score"] >= birth_score):
                raise ValueError("birth event contradicts effective threshold")
            if node["class_index"] == 0 and labels[j]["status"] != "outside_car_roi":
                channel["roi_car_" + event["event"] + "_score_channel_verified"] += 1
                if node["kind"] == "road_residual" and node["score"] < birth_score <= node["birth_effective_score"]:
                    channel["roi_residual_raised_eligibility_actual_birth"] += accepted
                    channel["roi_residual_raised_eligibility_actual_birth_" + labels[j]["status"]] += accepted
        if seen | assigned != set(range(len(d["nodes"]))):
            raise ValueError("node not covered by assignment or birth decision")
        return source

    function_globals = dict(core.analyze_sequence.__globals__, validate_frame=validate)
    function = FunctionType(core.analyze_sequence.__code__, function_globals,
                            core.analyze_sequence.__name__, core.analyze_sequence.__defaults__,
                            core.analyze_sequence.__closure__)
    return function, channel


def reference_audit(core, path, expected_report_sha256, old_sources):
    path = Path(path)
    if path.is_symlink() or {p.name for p in path.iterdir()} != REFERENCE_FILES:
        raise ValueError("reference event audit must contain the sealed seven-file bundle")
    files = {name: core.evidence(path / name) for name in sorted(REFERENCE_FILES)}
    complete = core.read_json(path / "completion.json")
    if (files["report.json"]["sha256"] != expected_report_sha256
            or (path / "completion.json").read_bytes() != core.canonical(complete) + b"\n"
            or complete.get("status") != "completed" or complete.get("audit_source_sha256") != CORE_SHA256
            or complete.get("files") != {k: v for k, v in files.items() if k != "completion.json"}):
        raise ValueError("reference event report pin or completion hashes mismatch")
    report, sources, sequences = [core.read_json(path / name) for name in ("report.json", "sources.json", "sequences.json")]
    if (report.get("kind") != "mechanism_car_event_audit_v1" or report.get("status") != "completed"
            or report.get("reporting_scope") != "car_only" or report.get("paper_eligible") is not False
            or report.get("causal_effect_identified") is not False
            or report.get("gt_full_text_exported") is not False or report.get("prediction_streams_exported") is not False
            or report.get("protocol_sha256") != hashlib.sha256(core.canonical(core.PROTOCOL)).hexdigest()
            or core.read_json(path / "protocol.json") != core.PROTOCOL
            or report.get("gt_manifest_sha256") != core.GT_MANIFEST_SHA256
            or report.get("gt_frames_sha256") != core.GT_SHA256
            or sources["audit_source"]["sha256"] != CORE_SHA256
            or sources["adapter"]["sha256"] != core.ADAPTER_SHA256
            or sources["gt_manifest"]["sha256"] != core.GT_MANIFEST_SHA256
            or sources["gt_frames"]["sha256"] != core.GT_SHA256
            or set(report["runs"]) != set(core.RUNS) or set(sequences) != set(core.RUNS)):
        raise ValueError("reference event audit protocol/source mismatch")
    convert = lambda record: {"sha256": record["sha256"], "size_bytes": record["size"]}
    for key, filename in (("plan", "frozen-plan.json"), ("summary", "summary.json")):
        if sources[key] != convert(old_sources[filename]):
            raise ValueError("reference event audit belongs to different mechanism inputs")
    if report["plan_sha256"] != old_sources["frozen-plan.json"]["sha256"]:
        raise ValueError("reference event plan differs from M4 reference")
    for name, (mode, seed, deterministic) in core.RUNS.items():
        expected = {k: convert(old_sources[name][filename]) for k, filename in
                    (("receipt", "receipt.json"), ("predictions", "predictions.jsonl"),
                     ("association", "association.jsonl"), ("diagnostics", "diagnostics.jsonl.gz"))}
        if sources["run:" + name] != expected:
            raise ValueError("reference event audit stream hashes differ: " + name)
        actual = report["runs"][name]
        if (actual["mode"], actual["seed"], actual["deterministic_control"]) != (mode, seed, deterministic):
            raise ValueError("reference event run identity mismatch")
        total = Counter()
        for sid, value in sequences[name].items():
            if value["sequence_id"] != sid:
                raise ValueError("reference event sequence identity mismatch")
            for number in value["counts"].values():
                if core.finite(number) < 0:
                    raise ValueError("negative reference event count")
            total.update(value["counts"])
        # Floating sums are compared within rounding only; count definitions are fixed.
        if set(total) != set(actual["counts"]) or any(not math.isclose(total[k], actual["counts"][k],
                rel_tol=1e-12, abs_tol=1e-9) for k in total):
            raise ValueError("reference sequence counts do not reproduce pinned report")
    return report, sequences, files


def paired_comparisons(core, runs, reference_sequences):
    output = []
    for seed in core.SEEDS:
        name = f"M4-seed-{seed}"
        for mode in ("M0", "M3"):
            reference = f"{mode}-seed-{seed}"
            target, control = runs[name]["sequences"], reference_sequences[reference]
            if set(target) != set(control):
                raise ValueError("paired event sequence coverage differs")
            differences = {}
            for sid in sorted(target):
                left, right = target[sid]["counts"], control[sid]["counts"]
                differences[sid] = {k: left.get(k, 0) - right.get(k, 0) for k in sorted(set(left) | set(right))}
            total = Counter()
            for values in differences.values():
                total.update(values)
            output.append({"run_id": name, "reference_run_id": reference, "by_sequence": differences,
                           "total_difference": dict(total), "scalar_difference_only": True,
                           "node_or_track_alignment_across_interventions": False,
                           "birth_channel_counters_compared": False, "additive_causal_decomposition": False})
    return output


def run(experiment_root, mechanism_root, reference_event_audit, ground_truth, output,
        expected_plan_sha256, expected_reference_report_sha256):
    root, old, reference, gt_dir, output = map(Path, (experiment_root, mechanism_root, reference_event_audit, ground_truth, output))
    core, control = load_dependencies()
    control.protect_output(output, root, old, reference, gt_dir)
    base = control.load_base()
    plan, receipts, runner_sources, _, old_sources = control.validate_birth_experiment(root, old, expected_plan_sha256, base)
    _, reference_sequences, reference_sources = reference_audit(core, reference, expected_reference_report_sha256, old_sources)
    paths = {"audit_source": Path(__file__), "core": CORE_PATH, "input_validator": CONTROL_PATH,
             "base_validator": control.BASE_PATH, "source_evaluator": control.SOURCE_EVALUATOR,
             "geometry": core.ADAPTER_PATH, "gt_manifest": gt_dir / "manifest.json", "gt_frames": gt_dir / "ground-truth.jsonl"}
    dependencies = {name: core.evidence(path) for name, path in paths.items()}
    if (dependencies["gt_manifest"]["sha256"] != core.GT_MANIFEST_SHA256
            or dependencies["gt_frames"]["sha256"] != core.GT_SHA256):
        raise ValueError("sealed GT input hashes mismatch")
    import numpy as np
    import shapely
    if np.__version__ != "1.26.4" or shapely.__version__ != "2.0.7":
        raise ValueError("geometry runtime must use numpy 1.26.4 and shapely 2.0.7")
    adapter = core.load_adapter()
    _, gt = adapter.load_ground_truth(gt_dir)
    if len(gt) != core.EXPECTED_FRAMES or len({g["sequence_id"] for g in gt}) != core.EXPECTED_SEQUENCES:
        raise ValueError("incomplete official validation GT")
    snapshot = {}
    control.remember(snapshot, root, runner_sources); control.remember(snapshot, old, old_sources)
    # Recheck source-pinned implementation dependencies at the end, not only on entry.
    old_plan = core.read_json(old / "frozen-plan.json")
    implementation = {name: core.evidence(ROOT / name) for name in
                      set(plan.get("current_source_hashes", {})) | set(old_plan.get("current_source_hashes", {}))}
    runs, episodes, source_cache = {}, {}, {}
    birth_threshold = core.finite(plan["config"]["birth_score"])
    for spec in control.RUNS:
        name = spec["run_id"]
        streams = [iter(core.records(root / name / filename)) for filename in
                   ("predictions.jsonl", "association.jsonl", "diagnostics.jsonl.gz")]
        sequences, errors, total, channels = {}, [], Counter(), Counter()
        try:
            for sid, group in itertools.groupby(gt, key=lambda g: g["sequence_id"]):
                gt_rows = list(group)
                if sid in sequences:
                    raise ValueError("noncontiguous frozen GT sequence")
                def rows():
                    for _ in gt_rows:
                        values = [next(stream, None) for stream in streams]
                        if any(value is None for value in values):
                            raise ValueError("truncated M4 audit stream")
                        yield tuple(values)
                analyze, channel = adapted_sequence(core, adapter, birth_threshold)
                result, events = analyze(sid, gt_rows, rows(), adapter, "M4", expected_plan_sha256, source_cache, birth_threshold)
                if (result["prediction_final_commit_sha256"] != receipts[name]["sequence_commits"][sid]
                        or result["diagnostic_final_commit_sha256"] != receipts[name]["diagnostic_sequence_commits"][sid]):
                    raise ValueError("M4 real sequence receipt tip mismatch")
                result["birth_channel_counts"] = dict(channel)
                sequences[sid] = result
                errors.extend(events); total.update(result["counts"]); channels.update(channel)
            if any(next(stream, None) is not None for stream in streams):
                raise ValueError("extra M4 frame outside official cohort")
        finally:
            for stream in streams:
                stream.close()
        spans = [e["observed_error_span_seconds"] for e in errors]
        denominator = total["car_pair_evaluable_occurrences"] + total["car_pair_unknown_occurrences"]
        runs[name] = {**spec, "counts": dict(total), "birth_channel_counts": dict(channels), "sequences": sequences,
                      "identity_observed_span_median_seconds": statistics.median(spans) if spans else None,
                      "identity_evaluable_fraction": total["identity_unique_gt_frames"] / total["roi_gt_frame_observations"]
                          if total["roi_gt_frame_observations"] else None,
                      "association_evaluable_fraction": total["car_pair_evaluable_occurrences"] / denominator if denominator else None}
        episodes[name] = errors
        print(f"BIRTH_SCORE_EVENT_AUDIT_RUN {name} frames={total['frames']} anchor_episodes={len(errors)}", flush=True)
    if (any(control.evidence(path) != value for path, value in snapshot.items())
            or any(core.evidence(path) != dependencies[name] for name, path in paths.items())
            or any(core.evidence(ROOT / name) != value for name, value in implementation.items())
            or any(core.evidence(reference / name) != value for name, value in reference_sources.items())):
        raise ValueError("immutable inputs or audit sources changed during analysis")
    protocol = {**core.PROTOCOL, "kind": "birth_score_car_event_audit_protocol_v1",
                "sealed_common_protocol_sha256": hashlib.sha256(core.canonical(core.PROTOCOL)).hexdigest(),
                "birth_score_threshold": birth_threshold,
                "common_node_score": "nodes[].score unchanged; original residual-suppression counters describe this channel, not actual birth rejection",
                "birth_channel": "nodes[].birth_effective_score; road source score only for birth gate and initial track score",
                "new_channel_comparison": "M4-only measurements; absent M0/M3 channels are not treated as zero",
                "paired_comparison": "M4 minus same-seed M0 and M3 common scalar counts; no cross-intervention ID alignment",
                "M4_minus_M3": "descriptive contrast; not additive mediation or an isolated temporal effect",
                "free_running": "birth changes future populations/states; later assignment equality is not asserted",
                "schema_adapter": "private function-global view validates shared schema after validating original M4 seal; no evidence or module mutation"}
    report = {"kind": "birth_score_car_event_audit_v1", "status": "completed", "reporting_scope": "car_only",
              "plan_sha256": expected_plan_sha256, "reference_report_sha256": expected_reference_report_sha256,
              "protocol_sha256": hashlib.sha256(core.canonical(protocol)).hexdigest(),
              "gt_manifest_sha256": core.GT_MANIFEST_SHA256, "gt_frames_sha256": core.GT_SHA256,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "shapely": shapely.__version__, "geos": shapely.geos_version_string},
              "paper_eligible": False, "causal_effect_identified": False, "gt_full_text_exported": False,
              "prediction_streams_exported": False, "published_to_clearml": False,
              "runs": {name: {k: v for k, v in result.items() if k != "sequences"} for name, result in runs.items()},
              "paired_comparisons": paired_comparisons(core, runs, reference_sequences)}
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("protocol.json", protocol), ("report.json", report),
                            ("sequences.json", {name: result["sequences"] for name, result in runs.items()}),
                            ("identity-episodes.json", episodes),
                            ("sources.json", {"dependencies": dependencies, "implementation": implementation,
                                              "M4_inputs": runner_sources, "original_inputs": old_sources, "reference_audit": reference_sources})):
        core.write_json(output / filename, value)
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write("# M4 car 出生分数事件审计\n\n仅为离线描述性诊断，不是官方 IDS 或因果效应证明。\n"
                     "复用冻结的 car ROI、oriented BEV IoU ≥ 0.5 双向唯一映射与时间戳删失定义；歧义保留 unknown。\n"
                     "counts 中残余分数压低统计指通用 node.score，不代表 M4 实际出生拒绝；birth_channel_counts 单独记录出生有效分数。\n"
                     "同种子 M4−M0 与 M4−M3 只比较共同统计量，不按节点编号或轨迹 ID 跨干预强行对齐，不构成可加性因果分解。\n"
                     "身份偏离相对首次唯一 ID 锚点，无法识别初始绝对错误；错误观测跨度不是无条件的连续错误时长界。\n"
                     "未知映射、GT 离开 ROI 与序列结束保留删失；零可评价样本为 null，不解释为零错误。\n"
                     "没有训练、参数选择、GT 入预测器、GT 全文或完整预测流导出，也没有 ClearML 发布。详情见 report.json 与 protocol.json。\n")
    core.write_json(output / "completion.json", {"status": "completed", "audit_source_sha256": dependencies["audit_source"]["sha256"],
                    "files": {p.name: core.evidence(p) for p in sorted(output.iterdir())}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment-root", "mechanism-root", "reference-event-audit", "ground-truth", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-reference-report-sha256", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
