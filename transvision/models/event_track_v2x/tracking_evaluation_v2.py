"""Frozen native-frame SPD adapter to the unmodified official metric engines.

GT preparation is intentionally separate from prediction/cache processing. The
adapter is not a claim of leaderboard parity: it preserves the original 3316
cooperative validation frames and never interpolates boxes or absent frames.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
from importlib import metadata
import inspect
import json
import math
from pathlib import Path
import pickle
import zipfile

import numpy as np


SPLIT_SHA256 = "4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3"
CLASSES = ("car", "bicycle", "pedestrian")
CLASS_MAP = {**dict.fromkeys(("Car", "Truck", "Van", "Bus"), "car"),
             **dict.fromkeys(("Motorcyclist", "Cyclist", "Tricyclist", "Barrowlist"), "bicycle"),
             "Pedestrian": "pedestrian"}
CLASS_RANGE = {"car": 50., "bicycle": 40., "pedestrian": 40.}
PROTOCOL = {
    "kind": "spd_native_tracking_evaluation_protocol_v2",
    "split_sha256": SPLIT_SHA256, "split": "official_validation",
    "expected_sequences": 21, "expected_frames": 3316,
    "primary_class": "car", "supplementary_classes": ["bicycle", "pedestrian"],
    "roi": {"type": "strict_global_xy_distance_from_vehicle_ego_origin",
            "class_range_m": CLASS_RANGE,
            "source": "CoopTrack 29f1c52 projects/configs_spd_coop/cooptrack/tiny_track_r50_stream_bs8_48epoch_3cls.py",
            "point_count_filter": "not_available_in_raw_spd_labels; no synthetic point counts or point filtering",
            "bike_rack_filter": "not_applicable; no bike-rack category in SPD labels"},
    "nuscenes": {"version": "1.2.0", "engine": "nuscenes.eval.tracking.evaluate.TrackingEval.evaluate",
                 "matching": "global XY center distance strictly less than 2m",
                 "recall_thresholds": 40, "min_recall": .1,
                 "scores": "mean score per sequence/track computed only in evaluator",
                 "mota_threshold": "standard engine maximum MOTA across recall thresholds; not model selection",
                 "excluded_headline_metrics": {"tid": "engine hardcodes 0.5s frame period",
                                                "lgd": "engine hardcodes 0.5s frame period"}},
    "trackeval": {"version": "1.0.0", "similarity": "upright oriented BEV rectangle IoU in world XY",
                  "hota_alphas": [i / 20 for i in range(1, 20)], "identity_iou_threshold": .5,
                  "prediction_score_threshold": "none beyond frozen tracker output",
                  "sequence_aggregation": "official metric combine_sequences; not arithmetic sequence mean"},
    "box_state": "world gravity xyz physical length width height yaw vx vy; upright heading projection",
    "temporal_policy": "native timestamp ordering; no resampling, absent-frame insertion, or box interpolation",
    "qualification": "official metric engines with source-pinned SPD adapter; not OOF or leaderboard parity",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def _integer(value, field):
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _identifier(value, field):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"invalid {field}")
    return value


def _pose(info):
    from pyquaternion import Quaternion
    le = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    ew = Quaternion(info["ego2global_rotation"]).rotation_matrix
    row_rotation = (ew @ le).T
    translation = np.asarray(info["lidar2ego_translation"]) @ ew.T + np.asarray(info["ego2global_translation"])
    return row_rotation, translation, np.asarray(info["ego2global_translation"])


def prepare_ground_truth(archive, split_path, vehicle_infos, output, schedule_path):
    """Read whitelisted official-val label members only, never train/test labels."""
    if sha(split_path) != SPLIT_SHA256:
        raise ValueError("official split hash mismatch")
    output, schedule_path = Path(output), Path(schedule_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("GT output must be empty; do not overwrite immutable evidence")
    if schedule_path.exists():
        raise ValueError("schedule already exists")
    split = json.loads(Path(split_path).read_text())
    val = set(split["cooperative_split"]["val"])
    sequences = set(split["batch_split"]["val"])
    if len(val) != 3316 or len(sequences) != 21:
        raise ValueError("unexpected official validation cohort")
    # This trusted local pickle is a previously generated validation-only pose
    # source. No annotation array is used in this adapter.
    with Path(vehicle_infos).open("rb") as stream:
        source_infos = pickle.load(stream)["infos"]
    infos = {str(i["token"]): i for i in source_infos}
    if len(infos) != len(source_infos):
        raise ValueError("duplicate frame in validation pose source")
    frames, schedule, sources, timings = [], [], [], defaultdict(list)
    raw_counts, kept_counts = Counter(), Counter()
    identities = defaultdict(set)
    with zipfile.ZipFile(archive) as z:
        metadata_name = "V2X-Seq-SPD/cooperative/data_info.json"
        metadata_bytes = z.read(metadata_name)
        # data_info is split metadata; payload reads below are strictly whitelisted.
        rows = [r for r in json.loads(metadata_bytes) if r["vehicle_frame"] in val]
        if len(rows) != 3316 or {r["vehicle_frame"] for r in rows} != val:
            raise ValueError("cooperative metadata does not exactly cover validation")
        for row in sorted(rows, key=lambda r: (r["vehicle_sequence"], r["vehicle_frame"])):
            sid, fid = row["vehicle_sequence"], row["vehicle_frame"]
            if sid not in sequences or sid != row["infrastructure_sequence"]:
                raise ValueError("invalid cooperative sequence boundary")
            info = infos[fid]
            if info["scene_token"] != sid or int(info["timestamp"]) != info["timestamp"]:
                raise ValueError("pose sequence/timestamp mismatch")
            timestamp = int(info["timestamp"])
            rot, trans, ego = _pose(info)
            label_name = f"V2X-Seq-SPD/cooperative/label/{fid}.json"
            label_bytes = z.read(label_name)  # Explicit val member, no archive extraction.
            annotations = json.loads(label_bytes)
            tids = [a["track_id"] for a in annotations]
            if len(tids) != len(set(tids)):
                raise ValueError(f"duplicate GT identity; no silent repair: {sid}/{fid}")
            objects = []
            for a in annotations:
                if int(a["veh_pointcloud_timestamp"]) != timestamp:
                    raise ValueError(f"GT timestamp mismatch: {sid}/{fid}")
                raw_counts[a["type"]] += 1
                name = CLASS_MAP.get(a["type"])
                if name is None:
                    raise ValueError(f"unexpected validation category: {a['type']}")
                loc = np.array([a["3d_location"][k] for k in ("x", "y", "z")])
                dims = np.array([a["3d_dimensions"][k] for k in ("l", "w", "h")])
                center = loc @ rot + trans
                heading = np.array([math.cos(a["rotation"]), math.sin(a["rotation"]), 0.]) @ rot
                state = np.r_[center, dims, math.atan2(heading[1], heading[0]), 0., 0.]
                if not np.isfinite(state).all() or np.any(dims <= 0):
                    raise ValueError(f"invalid GT dimensions/state: {sid}/{fid}/{a['track_id']}")
                identities[(sid, a["track_id"])].add(name)
                objects.append({"track_id": a["track_id"], "class_label": name,
                                "mean": state.tolist(), "annotation_token": a["token"],
                                "velocity_available": False})
                if np.linalg.norm(center[:2] - ego[:2]) < CLASS_RANGE[name]:
                    kept_counts[name] += 1
            frames.append({"sequence_id": sid, "frame_id": fid,
                           "box_reference_timestamp_us": timestamp,
                           "ego_translation_world": ego.tolist(),
                           "objects": sorted(objects, key=lambda a: a["track_id"])})
            schedule.append({"sequence_id": sid, "vehicle_frame": fid,
                             "infrastructure_frame": row["infrastructure_frame"],
                             "box_reference_timestamp_us": timestamp})
            sources.append({"path": label_name, "sha256": hashlib.sha256(label_bytes).hexdigest(),
                            "size": len(label_bytes), "objects": len(objects)})
            timings[sid].append(timestamp)
    gaps = {}
    for sid, values in timings.items():
        d = np.diff(values)
        if np.any(d <= 0):
            raise ValueError(f"non-increasing native GT time: {sid}")
        gaps[sid] = {"frames": len(values), "first_timestamp_us": values[0], "last_timestamp_us": values[-1],
                     "min_step_us": int(d.min()), "max_step_us": int(d.max()),
                     "steps_above_150ms": int(np.count_nonzero(d > 150000))}
    if set(timings) != sequences:
        raise ValueError("incomplete validation sequences")
    output.mkdir(parents=True, exist_ok=True)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path = output / "ground-truth.jsonl"
    with gt_path.open("wb") as stream:
        for frame in frames:
            stream.write(canonical(frame) + b"\n")
    # This schedule is deliberately outside the GT directory and contains no
    # system_error_offset, categories, counts, GT identity, poses or geometry.
    schedule_document = {"kind": "spd_official_validation_prediction_schedule_v1",
                         "split_sha256": SPLIT_SHA256, "frames": schedule,
                         "contains_ground_truth": False, "contains_system_error_offset": False}
    write_json(schedule_path, schedule_document)
    class_changes = [{"sequence_id": k[0], "track_id": k[1], "classes": sorted(v)}
                     for k, v in sorted(identities.items()) if len(v) > 1]
    manifest = {"kind": "spd_official_validation_evaluator_ground_truth_v2",
                "evaluator_only": True, "contains_ground_truth": True,
                "contains_train_payload": False, "contains_test_payload": False,
                "split_sha256": SPLIT_SHA256, "frames": 3316, "sequences": sorted(sequences),
                "ground_truth_file": gt_path.name, "ground_truth_sha256": sha(gt_path),
                "ground_truth_size": gt_path.stat().st_size,
                "prediction_schedule_sha256": sha(schedule_path),
                "pose_source_sha256": sha(vehicle_infos),
                "cooperative_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "label_members": sources, "raw_category_counts": dict(raw_counts),
                "roi_category_counts": dict(kept_counts), "duplicate_ids": 0,
                "class_changes": class_changes, "class_change_policy": "retain original labels; evaluate classes independently",
                "native_timing": gaps, "interpolated_frames": 0, "interpolated_objects": 0,
                "protocol": PROTOCOL, "adapter_source_sha256": sha(__file__)}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_ground_truth(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "ground-truth.jsonl"
    if (manifest["kind"] != "spd_official_validation_evaluator_ground_truth_v2"
            or manifest["split_sha256"] != SPLIT_SHA256
            or manifest["protocol"] != PROTOCOL or sha(path) != manifest["ground_truth_sha256"]):
        raise ValueError("GT/protocol evidence mismatch")
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    if (len(frames) != 3316 or len({f["sequence_id"] for f in frames}) != 21
            or len({(f["sequence_id"], f["frame_id"]) for f in frames}) != 3316):
        raise ValueError("GT frame coverage mismatch")
    return manifest, frames


def validate_predictions(predictions, gt):
    if len(predictions) != len(gt):
        raise ValueError("prediction frame coverage mismatch; empty frames must be explicit")
    previous, counts = {}, Counter()
    for p, g in zip(predictions, gt):
        key = (p["sequence_id"], p["frame_id"])
        if key != (g["sequence_id"], g["frame_id"]):
            raise ValueError(f"prediction/GT schedule mismatch: {key}")
        if _integer(p["box_reference_timestamp_us"], "box time") != g["box_reference_timestamp_us"]:
            raise ValueError("box-reference timestamp mismatch")
        if _integer(p["decision_timestamp_us"], "decision time") < p["box_reference_timestamp_us"]:
            raise ValueError("decision precedes the committed state time")
        if p.get("coordinate_frame") != "world" or p.get("state_layout") != "gravity_xyz_length_width_height_yaw_vxy":
            raise ValueError("prediction state convention is not explicit")
        seal = p.get("commit_sha256")
        if seal != hashlib.sha256(canonical({k: v for k, v in p.items() if k != "commit_sha256"})).hexdigest():
            raise ValueError("prediction commit seal mismatch")
        if p.get("previous_commit_sha256") != previous.get(p["sequence_id"], "0" * 64):
            raise ValueError("prediction commit chain mismatch")
        previous[p["sequence_id"]] = seal
        seen = set()
        for box in p["predictions"]:
            tid = _identifier(box["track_id"], "prediction track ID")
            if tid in seen:
                raise ValueError("duplicate predicted track in a frame")
            seen.add(tid)
            if box["class_label"] not in CLASSES:
                raise ValueError("unknown predicted class")
            mean, cov = np.asarray(box["mean"], float), np.asarray(box["covariance"], float)
            if (mean.shape != (9,) or cov.shape != (9, 9) or not np.isfinite(mean).all()
                    or not np.isfinite(cov).all() or np.any(mean[3:6] <= 0)
                    or not np.allclose(cov, cov.T, rtol=1e-6, atol=1e-7)
                    or np.linalg.eigvalsh(cov).min() < -1e-7):
                raise ValueError("invalid prediction state/covariance")
            score = box["score"]
            if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid prediction score")
            counts[box["class_label"]] += 1
    return {"frames": len(predictions), "sequences": len(previous), "predictions_per_class": dict(counts),
            "empty_frames": sum(not p["predictions"] for p in predictions),
            "final_sequence_commit_sha256": previous}


def oriented_bev_iou(left, right):
    """Exact convex-polygon area using GEOS; no axis-aligned-box shortcut."""
    from shapely.geometry import Polygon
    def polygons(boxes):
        result = []
        for box in boxes:
            m = np.asarray(box["mean"], float)
            corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * m[3:5] / 2
            c, s = math.cos(m[6]), math.sin(m[6])
            corners = corners @ np.array([[c, s], [-s, c]]) + m[:2]
            result.append(Polygon(corners))
        return result
    a, b = polygons(left), polygons(right)
    out = np.zeros((len(a), len(b)))
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            if p.intersects(q):
                intersection = p.intersection(q).area
                out[i, j] = min(1., max(0., intersection / (p.area + q.area - intersection)))
    return out


def _roi(boxes, ego, name):
    return [b for b in boxes if b["class_label"] == name
            and np.linalg.norm(np.asarray(b["mean"][:2]) - np.asarray(ego[:2])) < CLASS_RANGE[name]]


def _nuscenes_tracks(gt, predictions, classes):
    from nuscenes.eval.tracking.data_classes import TrackingBox
    ground, predicted = defaultdict(dict), defaultdict(dict)
    scores = defaultdict(list)
    for frame, pred in zip(gt, predictions):
        sid, time = frame["sequence_id"], frame["box_reference_timestamp_us"]
        ground[sid][time], predicted[sid][time] = [], []
        for name in classes:
            for source, target, is_gt in [(frame["objects"], ground[sid][time], True),
                                           (pred["predictions"], predicted[sid][time], False)]:
                for b in _roi(source, frame["ego_translation_world"], name):
                    m = b["mean"]
                    # Raw SPD IDs are sequence-scoped, whereas the official
                    # engine's worst-case ML denominator collects global IDs.
                    qualified_id = f"{sid}/{b['track_id']}"
                    target.append(TrackingBox(sample_token=f"{sid}/{frame['frame_id']}",
                        translation=tuple(m[:3]), size=(m[4], m[3], m[5]),
                        rotation=(math.cos(m[6] / 2), 0., 0., math.sin(m[6] / 2)),
                        velocity=tuple(m[7:9]), tracking_id=qualified_id, tracking_name=name,
                        tracking_score=-1. if is_gt else float(b["score"])))
                    if not is_gt:
                        scores[(sid, qualified_id)].append(b["score"])
    # Standard nuScenes score normalization, but deliberately no interpolation.
    avg = {k: float(np.mean(v)) for k, v in scores.items()}
    for sid, times in predicted.items():
        for boxes in times.values():
            for b in boxes:
                b.tracking_score = avg[(sid, b.tracking_id)]
    return ground, predicted


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def runtime_evidence():
    import nuscenes, trackeval, motmetrics, shapely
    expected = {"nuscenes-devkit": "1.2.0", "trackeval": "1.0.0", "motmetrics": "1.4.0", "numpy": "1.26.4"}
    observed = {k: metadata.version(k) for k in expected}
    if observed != expected:
        raise RuntimeError(f"pinned evaluation runtime mismatch: {observed}")
    trees = {}
    for name, module in [("nuscenes", nuscenes), ("trackeval", trackeval), ("motmetrics", motmetrics)]:
        root = Path(inspect.getfile(module)).parent
        inventory = [{"path": str(p.relative_to(root)), "sha256": sha(p)} for p in sorted(root.rglob("*.py"))]
        trees[name] = {"source_tree_sha256": hashlib.sha256(canonical(inventory)).hexdigest(), "files": inventory}
    return {"versions": {**observed, **{k: metadata.version(k) for k in ("scipy", "pandas", "shapely", "pyquaternion")}},
            "source_trees": trees, "adapter_source_sha256": sha(__file__)}


def compute_metrics(gt, predictions, classes=CLASSES):
    """Invoke official implementations unchanged; aggregate sequences officially."""
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.data_classes import TrackingConfig
    from nuscenes.eval.tracking.evaluate import TrackingEval
    import trackeval
    cfg = config_factory("tracking_nips_2019").serialize()
    cfg["tracking_names"] = list(classes)
    for key in ("class_range", "pretty_tracking_names", "tracking_colors"):
        cfg[key] = {name: cfg[key][name] for name in classes}
    engine = TrackingEval.__new__(TrackingEval)
    engine.cfg = TrackingConfig.deserialize(cfg)
    engine.tracks_gt, engine.tracks_pred = _nuscenes_tracks(gt, predictions, classes)
    engine.verbose, engine.output_dir, engine.render_classes = False, None, []
    official_summary, curve = engine.evaluate()
    summary = official_summary.serialize()
    excluded = {name: summary.pop(name, None) for name in ("tid", "lgd")}
    label = summary["label_metrics"]
    for name in ("tid", "lgd"):
        excluded[f"label_{name}"] = label.pop(name, None)
    hota = trackeval.metrics.HOTA()
    identity = trackeval.metrics.Identity({"THRESHOLD": .5, "PRINT_CONFIG": False})
    groups = defaultdict(list)
    for frame, pred in zip(gt, predictions):
        groups[frame["sequence_id"]].append((frame, pred))
    byclass = {}
    for name in classes:
        results = {}
        for sid, frames in groups.items():
            gs, ps, similarities = [], [], []
            for frame, pred in frames:
                g = _roi(frame["objects"], frame["ego_translation_world"], name)
                p = _roi(pred["predictions"], frame["ego_translation_world"], name)
                gs.append(g); ps.append(p); similarities.append(oriented_bev_iou(g, p))
            gmap = {k: i for i, k in enumerate(sorted({b["track_id"] for boxes in gs for b in boxes}))}
            pmap = {k: i for i, k in enumerate(sorted({b["track_id"] for boxes in ps for b in boxes}))}
            data = {"num_timesteps": len(frames), "num_gt_ids": len(gmap), "num_tracker_ids": len(pmap),
                    "num_gt_dets": sum(map(len, gs)), "num_tracker_dets": sum(map(len, ps)),
                    "gt_ids": [np.array([gmap[b["track_id"]] for b in boxes], dtype=int) for boxes in gs],
                    "tracker_ids": [np.array([pmap[b["track_id"]] for b in boxes], dtype=int) for boxes in ps],
                    "similarity_scores": similarities}
            results[sid] = {"HOTA": hota.eval_sequence(data), "Identity": identity.eval_sequence(data),
                            "frames": len(frames), "gt_objects": data["num_gt_dets"], "predicted_objects": data["num_tracker_dets"]}
        h = hota.combine_sequences({s: r["HOTA"] for s, r in results.items()})
        i = identity.combine_sequences({s: r["Identity"] for s, r in results.items()})
        byclass[name] = {"summary": {**{k: float(np.mean(h[k])) for k in ("HOTA", "AssA", "DetA")}, **i},
                         "HOTA_curves": h, "sequences": results}
    return _clean({"nuscenes": summary, "nuscenes_curves": curve.serialize(),
                   "nuscenes_inapplicable_2hz_time_metrics_not_valid_seconds": excluded,
                   "trackeval": byclass})


def golden_cases():
    """Exercise both real engines with perfect, empty and identity-switch cases."""
    gt, perfect = [], []
    for index, time in enumerate((1000000, 1100100, 1200200, 1750000)):
        box = {"track_id": "gt1", "class_label": "car", "mean": [index, 0., 0., 4., 2., 1.5, 0., 0., 0.]}
        gt.append({"sequence_id": "golden", "frame_id": str(index), "box_reference_timestamp_us": time,
                   "ego_translation_world": [0., 0., 0.], "objects": [box]})
        perfect.append({"predictions": [{**copy.deepcopy(box), "track_id": "pred1", "score": .9}]})
    empty = [{"predictions": []} for _ in gt]
    switched = copy.deepcopy(perfect)
    for f in switched[2:]:
        f["predictions"][0]["track_id"] = "pred2"
    result = {}
    for name, pred in (("perfect", perfect), ("empty", empty), ("identity_switch", switched)):
        metric = compute_metrics(gt, pred, ("car",))
        result[name] = {"nuscenes": metric["nuscenes"], "trackeval": metric["trackeval"]["car"]["summary"]}
    p, e, s = [result[n] for n in ("perfect", "empty", "identity_switch")]
    assert math.isclose(p["nuscenes"]["amota"], 1.)
    assert math.isclose(p["nuscenes"]["amotp"], 0., abs_tol=1e-10)
    assert p["nuscenes"]["ids"] == 0
    assert all(math.isclose(p["trackeval"][k], 1.) for k in ("HOTA", "AssA", "DetA", "IDF1"))
    assert e["nuscenes"]["amota"] == 0. and e["nuscenes"]["amotp"] == 2.
    assert all(e["trackeval"][k] == 0. for k in ("HOTA", "AssA", "DetA", "IDF1"))
    assert s["nuscenes"]["ids"] == 1
    assert math.isclose(s["trackeval"]["IDF1"], .5)
    assert math.isclose(s["trackeval"]["AssA"], .5)
    assert math.isclose(s["trackeval"]["HOTA"], math.sqrt(.5))
    assert s["trackeval"]["DetA"] == 1.
    return {"passed": True, "native_timestamps_us": [f["box_reference_timestamp_us"] for f in gt], "cases": result}


def evaluate(ground_truth_dir, prediction_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("evaluation output must be empty; immutable result cannot be overwritten")
    runtime = runtime_evidence()
    gold = golden_cases()
    manifest, gt = load_ground_truth(ground_truth_dir)
    prediction_path = Path(prediction_path)
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    coverage = validate_predictions(predictions, gt)
    metrics = compute_metrics(gt, predictions)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    write_json(output / "runtime.json", runtime)
    write_json(output / "golden-cases.json", gold)
    report = {"kind": "spd_complete_native_tracking_validation_v2", "status": "completed",
              "protocol": PROTOCOL, "protocol_sha256": hashlib.sha256(canonical(PROTOCOL)).hexdigest(),
              "ground_truth_manifest_sha256": sha(Path(ground_truth_dir) / "manifest.json"),
              "ground_truth_sha256": manifest["ground_truth_sha256"],
              "predictions_sha256": sha(prediction_path), "coverage": coverage,
              "files": {p.name: {"sha256": sha(p), "size": p.stat().st_size} for p in sorted(output.iterdir())},
              "primary_car": {"nuscenes": {k: v["car"] for k, v in metrics["nuscenes"]["label_metrics"].items()},
                              "trackeval": metrics["trackeval"]["car"]["summary"]},
              "no_test_payload": True, "no_model_selection": True, "paper_eligible": False}
    write_json(output / "report.json", report)
    return report
