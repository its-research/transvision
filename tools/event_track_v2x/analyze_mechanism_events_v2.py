#!/usr/bin/env python3
"""Evaluator-only car event audit for the ten frozen mechanism controls.

GT is never imported into a predictor. Geometry is reused from the source-pinned
sealed evaluator; event identities use strict unique overlap, not Hungarian
tie-breaking. This is descriptive diagnostic evidence, not causal identification
or an implementation of the official IDS metric.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import platform
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "transvision/models/event_track_v2x/tracking_evaluation_v2.py"
ADAPTER_SHA256 = "659fbf0cafad0e942eb2bb72023f3a4b9cc4c4cfca6883503b48f3c5c3c9b7ec"
GT_MANIFEST_SHA256 = "94675ac8585d893195b7e801b754f62ca4ee3524cf852e05e53121fe2cc4084a"
GT_SHA256 = "94908e0010003b42a5ed2c35ccc894005fe8d40cadfe57653871bbd95e0a9a76"
CACHE_SHA256 = "66c58025bd79ea674ff676f1df8d76e7812c8b1d2b62bc54e8de3ced0bc3aca8"
SCHEDULE_SHA256 = "2c8999ecbe2ab98bedf13ba2da4b22bd6167eca07b368a99db148abf36de982a"
EXPECTED_FRAMES, EXPECTED_SEQUENCES = 3316, 21
SEEDS = (1337, 2027, 3407)
RUNS = ({f"M0-seed-{s}": ("M0", s, False) for s in SEEDS}
        | {"M1-all-unmatched": ("M1", 1337, True)}
        | {f"{mode}-seed-{s}": (mode, s, False) for mode in ("M2", "M3") for s in SEEDS})
SIDES = ("vehicle-side", "infrastructure-side")
ZERO = "0" * 64
MAX_EXAMPLES_PER_SEQUENCE = 8
MAX_LINE_BYTES = 16 * 1024 * 1024

PROTOCOL = {
    "kind": "mechanism_car_event_audit_protocol_v1", "reporting_scope": "car_only",
    "iou_threshold": .5, "comparison": "greater_than_or_equal", "roi": "sealed car strict XY range <50m",
    "mapping": "per-side or prediction-set mutual degree-one in IoU>=0.5 graph; other overlaps ambiguous",
    "association_mass": "retained-hypothesis conditional weight only; not full posterior probability",
    "duplicates": "two or more car predictions each overlapping exactly the same sole GT; GT frame and excess counts",
    "identity_proxy": "fixed first strictly unique predicted ID per sequence/GT; deviations are anchor disagreements, not official IDS",
    "identity_censoring": "unknown mapping/GT absent or outside ROI/sequence end right-censor; re-entry after unknown left-censors onset",
    "identity_duration": "observed error span last_wrong_us-first_wrong_us; recovery-observation elapsed is separate; no guaranteed continuous-time bounds without a no-hidden-switch assumption",
    "initial_anchor": "first ID is arbitrary and may already be wrong; initial absolute identity correctness is unidentifiable",
    "pair_to_node": "same-frame source selected indices/components only; never align intervention nodes by node index",
    "lifecycle_scope": "birth counts include all car-class nodes plus ROI subset; kills use known prior car class, not inferred current ROI",
    "paired_comparison": "sequence-level scalar differences against seed-matched M0; deterministic M1 compared against all M0 seeds",
    "prediction_score_threshold": "none beyond sealed predictions", "gt_used_for_predictions": False,
    "test_payloads_read": False, "no_parameter_fitting": True, "causal_effect_identified": False,
    "unknown_policy": "report unknown and ambiguous denominators; never substitute a forced GT match",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evidence(path):
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise ValueError("expected regular non-symlink evidence: " + str(p))
    return {"sha256": sha(p), "size_bytes": p.stat().st_size}


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate JSON key: " + k)
        result[k] = v
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read_json(path):
    return decode(Path(path).read_bytes())


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                return
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError("diagnostic line exceeds explicit safety limit")
            value = decode(raw)
            if raw != canonical(value) + b"\n":
                raise ValueError("prediction/association/diagnostic JSONL must be canonical")
            yield value


def load_adapter():
    if evidence(ADAPTER_PATH)["sha256"] != ADAPTER_SHA256:
        raise ValueError("sealed adapter source hash mismatch")
    spec = importlib.util.spec_from_file_location("sealed_mechanism_event_geometry", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite numeric value")
    return value


def index(value, size):
    if type(value) is not int or not 0 <= value < size:
        raise ValueError("invalid local selected/node index")
    return value


def state_box(record, adapter):
    ci = index(record["class_index"], len(adapter.CLASSES))
    mean = np.asarray(record["mean"], float)
    if mean.shape != (9,) or not np.isfinite(mean).all() or np.any(mean[3:6] <= 0):
        raise ValueError("invalid diagnostic world state")
    return {"class_label": adapter.CLASSES[ci], "mean": record["mean"]}


def unique_overlap(iou):
    """No arbitrary winner: a row maps only if its unique column has degree one."""
    a = np.asarray(iou, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError("invalid IoU matrix")
    edge = a >= .5
    nr, nc = edge.sum(axis=1), edge.sum(axis=0)
    labels, singleton_groups = [], defaultdict(list)
    for i, n in enumerate(nr):
        if n == 0:
            labels.append({"status": "unmatched", "gt_index": None})
        elif n != 1:
            labels.append({"status": "ambiguous", "gt_index": None})
        else:
            j = int(np.flatnonzero(edge[i])[0])
            singleton_groups[j].append(i)
            labels.append({"status": "unique" if nc[j] == 1 else "ambiguous",
                           "gt_index": j if nc[j] == 1 else None})
    duplicates = {j: rows for j, rows in singleton_groups.items() if len(rows) > 1}
    return labels, duplicates


def map_boxes(boxes, frame, adapter):
    ground = adapter._roi(frame["objects"], frame["ego_translation_world"], "car")
    in_roi = {id(b) for b in adapter._roi(boxes, frame["ego_translation_world"], "car")}
    selected = [(i, b) for i, b in enumerate(boxes) if id(b) in in_roi]
    labels, duplicates = unique_overlap(adapter.oriented_bev_iou([b for _, b in selected], ground))
    result = [{"status": "outside_car_roi", "gt_id": None} for _ in boxes]
    for (i, _), label in zip(selected, labels):
        j = label["gt_index"]
        result[i] = {"status": label["status"], "gt_id": None if j is None else ground[j]["track_id"]}
    return result, {ground[j]["track_id"]: [selected[i][0] for i in rows] for j, rows in duplicates.items()}, ground


def pseudonym(sequence, gt_id):
    return hashlib.sha256(canonical([sequence, gt_id])).hexdigest()


class IdentityAudit:
    """Observable first-ID disagreements, with gaps never filled or interpolated."""
    def __init__(self, sequence):
        self.sequence = sequence
        self.states = {}
        self.episodes = []
        self.counts = Counter()
        self.previous_time = None

    def _close(self, state, timestamp, reason):
        episode = state["episode"]
        if episode is None:
            return
        episode["right_censored"] = reason != "anchor_return_observed"
        episode["closure_reason"] = reason
        episode["closure_observation_us"] = timestamp
        episode["observed_error_span_seconds"] = (episode["last_wrong_us"] - episode["first_wrong_us"]) / 1e6
        episode["recovery_observation_elapsed_seconds"] = (
            (timestamp - episode["first_wrong_us"]) / 1e6 if reason == "anchor_return_observed" else None)
        episode["onset_reference_to_return_observation_seconds"] = (
            (timestamp - episode["onset_lower_us"]) / 1e6
            if reason == "anchor_return_observed" and episode["onset_lower_us"] is not None else None)
        episode["other_track_ids"] = sorted(episode["other_track_ids"])
        self.episodes.append(episode)
        state["episode"] = None

    def step(self, timestamp, gt_ids, unique_predictions):
        if type(timestamp) is not int or (self.previous_time is not None and timestamp <= self.previous_time):
            raise ValueError("identity timestamps must strictly increase")
        gt_ids = set(gt_ids)
        if not set(unique_predictions) <= gt_ids:
            raise ValueError("identity mapping outside current GT ROI")
        for gid, state in self.states.items():
            if gid not in gt_ids:
                self._close(state, timestamp, "gt_absent_or_outside_roi")
                state["previous_status"] = "unknown"
        for gid in sorted(gt_ids, key=str):
            state = self.states.setdefault(gid, {"anchor": None, "previous_status": "unknown",
                                                "previous_known_us": None, "episode": None})
            tid = unique_predictions.get(gid)
            self.counts["roi_gt_frame_observations"] += 1
            if tid is None:
                self.counts["identity_unknown_gt_frames"] += 1
                self._close(state, timestamp, "mapping_unknown")
                state["previous_status"] = "unknown"
                continue
            self.counts["identity_unique_gt_frames"] += 1
            if state["anchor"] is None:
                state["anchor"] = tid
                self.counts["anchored_gt_identities"] += 1
            if tid == state["anchor"]:
                self.counts["anchor_agreement_gt_frames"] += 1
                self._close(state, timestamp, "anchor_return_observed")
                state["previous_status"] = "anchor"
            else:
                self.counts["anchor_disagreement_gt_frames"] += 1
                if state["episode"] is None:
                    state["episode"] = {
                        "sequence_id": self.sequence, "gt_key_sha256": pseudonym(self.sequence, gid),
                        "anchor_track_id": state["anchor"], "first_wrong_us": timestamp,
                        "last_wrong_us": timestamp, "wrong_observations": 0, "other_track_ids": set(),
                        "left_censored": state["previous_status"] != "anchor",
                        "onset_lower_us": state["previous_known_us"] if state["previous_status"] == "anchor" else None,
                        "onset_upper_us": timestamp,
                    }
                episode = state["episode"]
                episode["last_wrong_us"] = timestamp
                episode["wrong_observations"] += 1
                episode["other_track_ids"].add(tid)
                state["previous_status"] = "wrong"
            state["previous_known_us"] = timestamp
        self.previous_time = timestamp

    def finish(self):
        for state in self.states.values():
            self._close(state, self.previous_time, "sequence_end")
        return self.episodes


def validate_frame(p, a, d, g, mode, plan_sha, previous_diagnostic):
    for item in (p, a, d):
        if (item["sequence_id"], item["frame_id"]) != (g["sequence_id"], g["frame_id"]):
            raise ValueError("diagnostic/association/prediction/GT frame mismatch")
    if (d["kind"] != "tracking_mechanism_v2_frame" or d["mode"] != mode
            or d["plan_sha256"] != plan_sha or d["box_reference_timestamp_us"] != g["box_reference_timestamp_us"]
            or d["decision_timestamp_us"] != p["decision_timestamp_us"]
            or d["prediction_commit_sha256"] != p["commit_sha256"]
            or d["association_frame_sha256"] != hashlib.sha256(canonical(a)).hexdigest()
            or d["previous_diagnostic_commit_sha256"] != previous_diagnostic
            or d["diagnostic_commit_sha256"] != hashlib.sha256(canonical({k: v for k, v in d.items()
                                                                  if k != "diagnostic_commit_sha256"})).hexdigest()):
        raise ValueError("diagnostic identity or commit-chain mismatch")
    for name in ("source_available", "selected_detections", "source_cache_sha256"):
        if d[name] != p[name]:
            raise ValueError("diagnostic source boundary mismatch")
    selected = d["selected_detections"]
    if len(selected) != 2 or any(type(n) is not int or not 0 <= n <= 64 for n in selected):
        raise ValueError("invalid selected counts")
    source = {}
    for r in d["source_records"]:
        side = r["side"]
        if side not in SIDES:
            raise ValueError("unknown source side")
        key = (side, index(r["selected_index"], selected[SIDES.index(side)]))
        if key in source or r["frame_id"] != d["source_frame_ids"][SIDES.index(side)]:
            raise ValueError("duplicate or inconsistent source record")
        if type(r["raw_index"]) is not int or r["raw_index"] < 0:
            raise ValueError("invalid source raw index")
        if not (0 <= finite(r["score"]) <= 1 and 0 <= finite(r["raw_score"]) <= 1):
            raise ValueError("invalid source score")
        source[key] = r
    if len(source) != sum(selected):
        raise ValueError("source records do not cover selected inputs")
    hypotheses = a["hypotheses"]
    if not hypotheses or not math.isclose(sum(finite(h["weight"]) for h in hypotheses), 1., abs_tol=1e-8):
        raise ValueError("invalid retained association weights")
    for h in hypotheses:
        finite(h["energy"])
        if h["weight"] < 0:
            raise ValueError("negative retained weight")
        left, right = [], []
        for pair in h["pairs"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid association pair")
            left.append(index(pair[0], selected[0])); right.append(index(pair[1], selected[1]))
            if source[(SIDES[0], pair[0])]["class_index"] != source[(SIDES[1], pair[1])]["class_index"]:
                raise ValueError("cross-class association is forbidden by the frozen candidate contract")
        for used, unmatched, n in ((left, h["unmatched_left"], selected[0]), (right, h["unmatched_right"], selected[1])):
            complete = used + [index(i, n) for i in unmatched]
            if len(complete) != n or set(complete) != set(range(n)):
                raise ValueError("association is not a complete one-to-one partial matching")
    if mode == "M1" and (len(hypotheses) != 1 or hypotheses[0]["pairs"] or hypotheses[0]["weight"] != 1.):
        raise ValueError("M1 is not forced all-unmatched")
    if [n["node_index"] for n in d["nodes"]] != list(range(len(d["nodes"]))):
        raise ValueError("node indices are not local contiguous identities")
    for n in d["nodes"]:
        key = (n["source_side"], n["source_selected_index"])
        if key not in source or n["source_raw_index"] != source[key]["raw_index"]:
            raise ValueError("node provenance differs from source record")
        if n["kind"] == "vehicle_anchored_mixture":
            if n["source_side"] != SIDES[0] or len(n["components"]) != len(hypotheses):
                raise ValueError("invalid mixture component coverage")
            for k, (c, h) in enumerate(zip(n["components"], hypotheses)):
                partner = dict(h["pairs"]).get(n["source_selected_index"])
                if c["hypothesis_index"] != k or c["weight"] != h["weight"] or c["partner_selected_index"] != partner:
                    raise ValueError("component association provenance mismatch")
        elif n["kind"] != "road_residual" or n["source_side"] != SIDES[1]:
            raise ValueError("unknown node kind")
    for e in d["events"] + d["temporal_assignments"]:
        if "node_index" in e:
            index(e["node_index"], len(d["nodes"]))
    return source


def _example(examples, value):
    if len(examples) < MAX_EXAMPLES_PER_SEQUENCE:
        examples.append(value)


def analyze_sequence(sequence, gt_rows, rows, adapter, mode, plan_sha, source_cache, birth_score):
    counts, examples = Counter(), []
    identity = IdentityAudit(sequence)
    previous_diagnostic, prior_classes = ZERO, {}
    previous_unique_track_gt = {}
    prediction_batch = []
    for g, (p, a, d) in zip(gt_rows, rows):
        source = validate_frame(p, a, d, g, mode, plan_sha, previous_diagnostic)
        previous_diagnostic = d["diagnostic_commit_sha256"]
        prediction_batch.append(p)
        timestamp = g["box_reference_timestamp_us"]
        counts["frames"] += 1
        source_digest = hashlib.sha256(canonical(d["source_records"])).hexdigest()
        cache_key = (sequence, g["frame_id"])
        if cache_key in source_cache:
            cached = source_cache[cache_key]
            if cached["sha256"] != source_digest or cached["source_cache_sha256"] != d["source_cache_sha256"]:
                raise ValueError("source records differ between mechanism groups")
            source_labels = cached["labels"]
        else:
            source_labels = {}
            for side in SIDES:
                keys = sorted(k for k in source if k[0] == side)
                labels, _, _ = map_boxes([state_box(source[k], adapter) for k in keys], g, adapter)
                source_labels.update(zip(keys, labels))
            source_cache[cache_key] = {"sha256": source_digest, "source_cache_sha256": d["source_cache_sha256"],
                                      "labels": source_labels}
        for key, record in source.items():
            if record["class_index"] == 0:
                counts["source_car_" + source_labels[key]["status"]] += 1
        wrong_edges, unknown_edges = set(), set()
        wrong_mass_by_left = Counter()
        for h in a["hypotheses"]:
            for i, j in h["pairs"]:
                keys = ((SIDES[0], i), (SIDES[1], j))
                if any(source[k]["class_index"] != 0 for k in keys):
                    continue
                labels = [source_labels[k] for k in keys]
                counts["car_pair_occurrences"] += 1
                if any(x["status"] == "outside_car_roi" for x in labels):
                    counts["car_pair_outside_roi_occurrences"] += 1
                elif any(x["status"] != "unique" for x in labels):
                    counts["car_pair_unknown_occurrences"] += 1
                    counts["car_pair_unknown_conditional_mass"] += h["weight"]
                    unknown_edges.add((i, j))
                else:
                    counts["car_pair_evaluable_occurrences"] += 1
                    counts["car_pair_evaluable_conditional_mass"] += h["weight"]
                    if labels[0]["gt_id"] != labels[1]["gt_id"]:
                        counts["car_pair_wrong_occurrences"] += 1
                        counts["car_pair_wrong_conditional_mass"] += h["weight"]
                        wrong_edges.add((i, j)); wrong_mass_by_left[i] += h["weight"]
                        _example(examples, {"event": "uniquely_mapped_wrong_cross_source_pair", "frame_id": g["frame_id"],
                            "timestamp_us": timestamp, "source_raw_indices": [source[k]["raw_index"] for k in keys],
                            "gt_keys_sha256": [pseudonym(sequence, x["gt_id"]) for x in labels]})
        counts["unique_wrong_edges_per_frame_sum"] += len(wrong_edges)
        counts["unique_unknown_edges_per_frame_sum"] += len(unknown_edges)
        counts["frames_with_known_wrong_pair"] += bool(wrong_edges)
        node_labels, _, _ = map_boxes([state_box(n, adapter) for n in d["nodes"]], g, adapter)
        wrong_nodes = {n["node_index"] for n in d["nodes"] if n["source_side"] == SIDES[0]
                       and wrong_mass_by_left[n["source_selected_index"]] > 0}
        for n, label in zip(d["nodes"], node_labels):
            if n["class_index"] != 0 or label["status"] == "outside_car_roi":
                continue
            counts["roi_car_nodes"] += 1
            counts["node_car_" + label["status"]] += 1
            if n["kind"] == "road_residual":
                counts["roi_road_residual_nodes"] += 1
                if finite(n["original_score"]) >= birth_score:
                    counts["roi_residual_original_score_at_birth_threshold"] += 1
                    if finite(n["score"]) < birth_score:
                        counts["roi_residual_score_suppressed_below_birth_threshold"] += 1
                        counts["roi_residual_suppressed_unique_gt"] += label["status"] == "unique"
        pred_labels, duplicates, ground = map_boxes(p["predictions"], g, adapter)
        counts["duplicate_gt_frames"] += len(duplicates)
        counts["duplicate_excess_predictions"] += sum(len(v) - 1 for v in duplicates.values())
        counts["frames_with_duplicate_tracks"] += bool(duplicates)
        unique_predictions, unique_track_gt = {}, {}
        for box, label in zip(p["predictions"], pred_labels):
            if box["class_label"] != "car":
                continue
            counts["prediction_car_" + label["status"]] += 1
            if label["status"] == "unique":
                unique_predictions[label["gt_id"]] = box["track_id"]
                unique_track_gt[box["track_id"]] = label["gt_id"]
        identity.step(timestamp, [b["track_id"] for b in ground], unique_predictions)
        current_classes = {b["track_id"]: b["class_label"] for b in p["predictions"]}
        for event in d["events"]:
            kind = event["event"]
            if kind in ("birth", "birth_rejected_score"):
                node = d["nodes"][event["node_index"]]
                if node["class_index"] != 0:
                    continue
                counts["car_class_" + kind] += 1
                label = node_labels[event["node_index"]]
                if label["status"] != "outside_car_roi":
                    counts["roi_car_" + kind] += 1
                    counts["roi_car_" + kind + "_" + label["status"]] += 1
                    counts["roi_car_" + kind + "_touches_known_wrong_pair_node"] += event["node_index"] in wrong_nodes
                if kind == "birth" and current_classes.get(event["track_id"]) != "car":
                    raise ValueError("car birth absent from predictions")
            elif kind in ("kill_before_assignment", "kill_after_miss", "miss"):
                tid = event["track_id"]
                if tid not in prior_classes:
                    raise ValueError("lifecycle event has no prior track class")
                if prior_classes[tid] == "car":
                    counts["prior_car_class_" + kind] += 1
                    counts["prior_car_class_" + kind + "_with_previous_unique_gt"] += tid in previous_unique_track_gt
            else:
                raise ValueError("unknown lifecycle event")
        for event in d["temporal_assignments"]:
            if event["event"] != "matched" or event["track_id"] not in prior_classes:
                raise ValueError("invalid temporal assignment event")
            node = d["nodes"][event["node_index"]]
            if node["class_index"] == 0 and node_labels[event["node_index"]]["status"] != "outside_car_roi":
                counts["roi_car_temporal_assignments"] += 1
                counts["roi_car_temporal_assignments_touch_known_wrong_pair_node"] += event["node_index"] in wrong_nodes
        if d["tracks_after"] != len(p["predictions"]) or d["tracks_before"] != len(prior_classes):
            raise ValueError("lifecycle count mismatch")
        prior_classes, previous_unique_track_gt = current_classes, unique_track_gt
    adapter.validate_predictions(prediction_batch, gt_rows)
    episodes = identity.finish()
    counts.update(identity.counts)
    counts["anchor_disagreement_episodes"] = len(episodes)
    counts["anchor_episodes_left_censored"] = sum(e["left_censored"] for e in episodes)
    counts["anchor_episodes_right_censored"] = sum(e["right_censored"] for e in episodes)
    counts["anchor_episodes_return_observed"] = sum(not e["right_censored"] for e in episodes)
    counts["observed_error_span_seconds_sum"] = sum(e["observed_error_span_seconds"] for e in episodes)
    return {"sequence_id": sequence, "counts": dict(counts), "examples": examples,
            "prediction_final_commit_sha256": prediction_batch[-1]["commit_sha256"],
            "diagnostic_final_commit_sha256": previous_diagnostic}, episodes


def paired_comparisons(runs):
    output = []
    for run_id, (mode, seed, _) in RUNS.items():
        if mode == "M0":
            continue
        for control_seed in (SEEDS if mode == "M1" else (seed,)):
            control_id = f"M0-seed-{control_seed}"
            target, control = runs[run_id], runs[control_id]
            if set(target["sequences"]) != set(control["sequences"]):
                raise ValueError("paired sequence coverage differs")
            per_sequence = {}
            for sid in sorted(target["sequences"]):
                left, right = target["sequences"][sid]["counts"], control["sequences"][sid]["counts"]
                per_sequence[sid] = {k: left.get(k, 0) - right.get(k, 0) for k in sorted(set(left) | set(right))}
            keys = set().union(*(set(v) for v in per_sequence.values()))
            output.append({"run_id": run_id, "reference_run_id": control_id,
                "scalar_difference_only": True, "node_or_track_alignment_across_interventions": False,
                "by_sequence": per_sequence,
                "total_difference": {k: sum(v.get(k, 0) for v in per_sequence.values()) for k in sorted(keys)}})
    return output


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def run(experiment_root, ground_truth, output, expected_plan_sha256):
    root, gt_dir, output = map(Path, (experiment_root, ground_truth, output))
    if output.exists() or output.is_symlink():
        raise ValueError("event-audit output already exists; create-once only")
    if root.is_symlink() or gt_dir.is_symlink():
        raise ValueError("symlink input roots are forbidden")
    adapter = load_adapter()
    import shapely
    if np.__version__ != "1.26.4" or shapely.__version__ != "2.0.7":
        raise ValueError("geometry runtime must use numpy 1.26.4 and shapely 2.0.7")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "shapely": shapely.__version__,
               "geos": shapely.geos_version_string, "adapter_sha256": ADAPTER_SHA256}
    paths = {"plan": root / "frozen-plan.json", "summary": root / "summary.json",
             "gt_manifest": gt_dir / "manifest.json", "gt_frames": gt_dir / "ground-truth.jsonl",
             "adapter": ADAPTER_PATH, "audit_source": Path(__file__)}
    sources = {name: evidence(path) for name, path in paths.items()}
    if (sources["plan"]["sha256"] != expected_plan_sha256 or sources["gt_manifest"]["sha256"] != GT_MANIFEST_SHA256
            or sources["gt_frames"]["sha256"] != GT_SHA256):
        raise ValueError("frozen plan or sealed GT identity mismatch")
    plan, summary = read_json(paths["plan"]), read_json(paths["summary"])
    expected_runs = [{"run_id": name, "mode": mode, "seed": seed, "deterministic_control": deterministic}
                     for name, (mode, seed, deterministic) in RUNS.items()]
    if (plan.get("kind") != "mechanism_diagnostics_v2_plan" or plan.get("runs") != expected_runs
            or plan.get("reporting_scope") != "car_only" or plan.get("cache_sha256") != CACHE_SHA256
            or plan.get("schedule_sha256") != SCHEDULE_SHA256 or plan.get("gt_model_inputs") is not False
            or plan.get("test_payloads_read") is not False or plan.get("val_parameter_fitting") is not False
            or summary.get("kind") != "mechanism_diagnostics_v2_complete" or summary.get("status") != "completed"
            or summary.get("plan_sha256") != expected_plan_sha256 or set(summary["runs"]) != set(RUNS)
            or summary.get("all_required_sealed_streams_match") is not True or summary.get("paper_eligible") is not False):
        raise ValueError("mechanism experiment contract mismatch")
    manifest, gt = adapter.load_ground_truth(gt_dir)
    if len(gt) != EXPECTED_FRAMES or len({f["sequence_id"] for f in gt}) != EXPECTED_SEQUENCES:
        raise ValueError("incomplete official validation GT")
    source_cache, all_runs, all_episodes = {}, {}, {}
    for run_id, (mode, seed, deterministic) in RUNS.items():
        folder = root / run_id
        if folder.is_symlink():
            raise ValueError("symlink run directory")
        files = {"receipt": folder / "receipt.json", "predictions": folder / "predictions.jsonl",
                 "association": folder / "association.jsonl", "diagnostics": folder / "diagnostics.jsonl.gz"}
        inputs = {name: evidence(path) for name, path in files.items()}
        receipt = read_json(files["receipt"])
        if (receipt != summary["runs"][run_id] or receipt["frames"] != EXPECTED_FRAMES
                or receipt["sequences"] != EXPECTED_SEQUENCES or receipt["mode"] != mode
                or receipt["seed"] != seed or receipt["deterministic_control"] != deterministic
                or inputs["predictions"]["sha256"] != receipt["predictions_sha256"]
                or inputs["association"]["sha256"] != receipt["association_sha256"]
                or inputs["diagnostics"]["sha256"] != receipt["diagnostics_gzip_sha256"]):
            raise ValueError("run receipt or file identity mismatch: " + run_id)
        if mode in ("M2", "M3") and inputs["association"] != sources[f"run:M0-seed-{seed}"]["association"]:
            raise ValueError("M2/M3 association bytes differ from seed-matched M0")
        streams = [iter(records(files[name])) for name in ("predictions", "association", "diagnostics")]
        sequence_reports, episodes, counts = {}, [], Counter()
        try:
            for sid, group in itertools.groupby(gt, key=lambda x: x["sequence_id"]):
                gt_rows = list(group)
                if sid in sequence_reports:
                    raise ValueError("noncontiguous sequence in frozen GT")
                def rows():
                    for _ in gt_rows:
                        result = [next(stream, None) for stream in streams]
                        if any(item is None for item in result):
                            raise ValueError("truncated prediction/association/diagnostic stream")
                        yield tuple(result)
                result, errors = analyze_sequence(sid, gt_rows, rows(), adapter, mode, expected_plan_sha256,
                                                 source_cache, finite(plan["config"]["birth_score"]))
                sequence_reports[sid] = result
                episodes.extend(errors); counts.update(result["counts"])
                if (result["prediction_final_commit_sha256"] != receipt["sequence_commits"][sid]
                        or result["diagnostic_final_commit_sha256"] != receipt["diagnostic_sequence_commits"][sid]):
                    raise ValueError("sequence receipt tip mismatch")
            if any(next(stream, None) is not None for stream in streams):
                raise ValueError("extra frame after official validation cohort")
        finally:
            for stream in streams:
                stream.close()
        if {name: evidence(path) for name, path in files.items()} != inputs:
            raise ValueError("run inputs changed during event analysis")
        source_key = "run:" + run_id
        sources[source_key] = inputs
        spans = [e["observed_error_span_seconds"] for e in episodes]
        all_runs[run_id] = {"mode": mode, "seed": seed, "deterministic_control": deterministic,
            "counts": dict(counts), "sequences": sequence_reports,
            "identity_observed_span_median_seconds": statistics.median(spans) if spans else None,
            "identity_evaluable_fraction": counts["identity_unique_gt_frames"] / counts["roi_gt_frame_observations"]
                if counts["roi_gt_frame_observations"] else None,
            "association_evaluable_fraction": counts["car_pair_evaluable_occurrences"] /
                (counts["car_pair_evaluable_occurrences"] + counts["car_pair_unknown_occurrences"])
                if counts["car_pair_evaluable_occurrences"] + counts["car_pair_unknown_occurrences"] else None}
        all_episodes[run_id] = episodes
        print("MECHANISM_EVENT_AUDIT_RUN " + json.dumps({"run_id": run_id, "frames": counts["frames"],
              "anchor_disagreement_episodes": len(episodes)}), flush=True)
    if any(evidence(path) != sources[name] for name, path in paths.items()):
        raise ValueError("frozen inputs/source changed during event analysis")
    comparison = paired_comparisons(all_runs)
    protocol_sha = hashlib.sha256(canonical(PROTOCOL)).hexdigest()
    report = {"kind": "mechanism_car_event_audit_v1", "status": "completed", "plan_sha256": expected_plan_sha256,
        "protocol_sha256": protocol_sha, "reporting_scope": "car_only", "runtime": runtime,
        "gt_manifest_sha256": sources["gt_manifest"]["sha256"], "gt_frames_sha256": manifest["ground_truth_sha256"],
        "paper_eligible": False, "causal_effect_identified": False, "gt_full_text_exported": False,
        "prediction_streams_exported": False, "runs": {name: {k: v for k, v in result.items() if k != "sequences"}
                                                        for name, result in all_runs.items()},
        "paired_comparisons": comparison}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", PROTOCOL)
    write_json(output / "sources.json", sources)
    write_json(output / "sequences.json", {name: result["sequences"] for name, result in all_runs.items()})
    write_json(output / "identity-episodes.json", all_episodes)
    write_json(output / "report.json", report)
    lines = ["# Car-only 机制事件审计", "", "本报告仅为离线描述性诊断，不是官方 IDS 或因果效应证明。",
             "GT 映射固定为 car ROI 内 oriented BEV IoU ≥ 0.5 的双向唯一边；歧义不强行匹配。", "",
             "| 运行 | 可评价跨端配对 | 已知误配条件质量 | 重复 GT-帧 | 锚点偏离事件 | 右删失 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in all_runs.items():
        c = result["counts"]
        lines.append(f"| {name} | {c.get('car_pair_evaluable_occurrences', 0)} | "
                     f"{c.get('car_pair_wrong_conditional_mass', 0):.6f} | {c.get('duplicate_gt_frames', 0)} | "
                     f"{c.get('anchor_disagreement_episodes', 0)} | {c.get('anchor_episodes_right_censored', 0)} |")
    lines += ["", "配对质量是 retained Top-H 内条件权重之和，不是完整后验概率。",
              "身份代理使用真实微秒时间戳；报告首末错误观测之间的跨度与恢复观测elapsed，未知映射和序列结束均保留删失。",
              "帧间可能发生未观测的恢复或再次切换，因此这些观测跨度不是无条件的真实连续错误时长上下界。",
              "首个预测 ID 只作固定锚点，初始绝对身份错误无法从该代理量识别；不得把锚点偏离事件当官方 IDS。",
              "M0 配对比较只减每序列统计量，不跨干预对齐节点或轨迹编号。M1 仍是一个确定性对照。",
              "误配分量与 birth/temporal 节点的关联仅追溯同帧 provenance，不证明其导致最终身份错误。",
              "零可评价样本对应比例为 null/unknown，不解释为零错误。输出不包含 GT 坐标全文或完整预测流。", ""]
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    write_json(output / "completion.json", {"status": "completed", "files": {
        p.name: evidence(p) for p in sorted(output.iterdir())}, "audit_source_sha256": sources["audit_source"]["sha256"]})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", required=True, type=Path)
    p.add_argument("--ground-truth", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-plan-sha256", required=True)
    args = p.parse_args()
    run(args.experiment_root, args.ground_truth, args.output, args.expected_plan_sha256)


if __name__ == "__main__":
    main()
