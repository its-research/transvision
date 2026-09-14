"""Independent geometry, censoring, provenance and create-once audit checks."""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("mechanism_event_audit_test_module",
    ROOT / "tools/event_track_v2x/analyze_mechanism_events_v2.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def box(tid, x=0., yaw=0., label="car"):
    return {"track_id": tid, "class_label": label,
            "mean": [x, 0., 1., 4., 2., 2., yaw, 0., 0.]}


def pred_box(tid, x):
    return {**box(tid, x), "covariance": np.eye(9).tolist(), "score": .8,
            "identity_hypotheses": [{"track_id": tid, "probability": 1.}], "other_identity_probability": 0.}


def gt_frame(i, timestamp=None):
    return {"sequence_id": "seq", "frame_id": f"f{i:03d}",
            "box_reference_timestamp_us": timestamp if timestamp is not None else 1_000_000 + i * 100_000,
            "ego_translation_world": [0., 0., 0.], "objects": [box("g1", 0.), box("g2", 10.)]}


def seal(value, key):
    value[key] = hashlib.sha256(audit.canonical({k: v for k, v in value.items() if k != key})).hexdigest()
    return value


def make_frame(i, mode="M0", previous=None, diagnostic_previous=None, plan_sha="b" * 64, wrong=False):
    g = gt_frame(i)
    h = {"pairs": [] if mode == "M1" else ([[0, 1], [1, 0]] if wrong else [[0, 0], [1, 1]]),
         "unmatched_left": [0, 1] if mode == "M1" else [],
         "unmatched_right": [0, 1] if mode == "M1" else [], "weight": 1., "energy": 0.}
    a = {"sequence_id": "seq", "frame_id": g["frame_id"], "hypotheses": [h],
         "posterior_interpretation": "forced_all_unmatched_control" if mode == "M1" else "normalized_truncated_joint_energy_with_all_unmatched",
         "full_posterior_omitted_mass": None, "moment_matching": True}
    source = []
    nodes = []
    for side in audit.SIDES:
        for j in range(2):
            mean = box("", j * 10.)["mean"]
            source.append({"side": side, "frame_id": g["frame_id"] if side == audit.SIDES[0] else f"i{i:03d}",
                           "selected_index": j, "raw_index": j + 3, "raw_score": .8, "score": .8,
                           "class_index": 0, "mean": mean})
            if side == audit.SIDES[1] and mode != "M1":
                continue
            partner = dict(h["pairs"]).get(j)
            nodes.append({"node_index": len(nodes), "kind": "vehicle_anchored_mixture" if side == audit.SIDES[0] else "road_residual",
                          "source_side": side, "source_selected_index": j, "source_raw_index": j + 3,
                          "class_index": 0, "mean": mean, "score": .8, "original_score": .8,
                          "unmatched_mass": float(mode == "M1"), "matched_mass": float(mode != "M1"),
                          "components": [{"hypothesis_index": 0, "weight": 1., "partner_selected_index": partner}]
                              if side == audit.SIDES[0] else []})
    predictions = [pred_box(f"seq:{j:06d}", n["mean"][0]) for j, n in enumerate(nodes, 1)]
    p = {"sequence_id": "seq", "frame_id": g["frame_id"], "box_reference_timestamp_us": g["box_reference_timestamp_us"],
         "decision_timestamp_us": g["box_reference_timestamp_us"] + 100_000, "coordinate_frame": "world",
         "state_layout": "gravity_xyz_length_width_height_yaw_vxy", "source_cache_sha256": ["c" * 64, "d" * 64],
         "source_available": [True, True], "source_information_timestamp_us": [1, 1], "selected_detections": [2, 2],
         "predictions": predictions, "previous_commit_sha256": previous or audit.ZERO}
    seal(p, "commit_sha256")
    events = [{"event": "birth", "track_id": p["track_id"], "node_index": j, "score": .8}
              for j, p in enumerate(predictions)] if previous is None else []
    temporal = [] if previous is None else [{"event": "matched", "track_id": p["track_id"], "node_index": j}
                                           for j, p in enumerate(predictions)]
    d = {"kind": "tracking_mechanism_v2_frame", "mode": mode, "plan_sha256": plan_sha,
         "sequence_id": "seq", "frame_id": g["frame_id"], "box_reference_timestamp_us": g["box_reference_timestamp_us"],
         "decision_timestamp_us": p["decision_timestamp_us"], "source_frame_ids": [g["frame_id"], f"i{i:03d}"],
         "source_cache_sha256": p["source_cache_sha256"], "source_available": p["source_available"],
         "selected_detections": p["selected_detections"], "source_records": source, "nodes": nodes,
         "temporal_assignments": temporal, "events": events, "tracks_before": 0 if previous is None else len(predictions),
         "tracks_after": len(predictions), "tracks_eligible": 0 if previous is None else len(predictions),
         "prediction_commit_sha256": p["commit_sha256"], "association_frame_sha256": hashlib.sha256(audit.canonical(a)).hexdigest(),
         "previous_diagnostic_commit_sha256": diagnostic_previous or audit.ZERO}
    seal(d, "diagnostic_commit_sha256")
    return p, a, d


def test_unique_mapping_rejects_all_ambiguity_and_preserves_exact_threshold():
    labels, duplicates = audit.unique_overlap([[.5, 0], [.9, 0], [0, .8], [0, 0]])
    assert [x["status"] for x in labels] == ["ambiguous", "ambiguous", "unique", "unmatched"]
    assert duplicates == {0: [0, 1]}
    labels, duplicates = audit.unique_overlap([[.8, .7], [0., .6]])
    assert [x["status"] for x in labels] == ["ambiguous", "ambiguous"]
    assert duplicates == {}


@pytest.mark.parametrize("value", [[[float("nan")]], [[-1]], [[1.01]], [1, 2]])
def test_invalid_iou_rejected(value):
    with pytest.raises(ValueError, match="IoU"):
        audit.unique_overlap(value)


def test_real_sealed_oriented_geometry_and_strict_car_roi():
    adapter = audit.load_adapter()
    g = gt_frame(0)
    actual = [box("a", 0, np.pi / 2), box("b", 10), box("c", 50), box("d", 0, label="pedestrian")]
    labels, duplicates, ground = audit.map_boxes(actual, g, adapter)
    assert np.isclose(adapter.oriented_bev_iou([actual[0]], [g["objects"][0]])[0, 0], 1/3)
    assert [v["status"] for v in labels] == ["unmatched", "unique", "outside_car_roi", "outside_car_roi"]
    assert labels[1]["gt_id"] == "g2" and not duplicates and len(ground) == 2


def test_empty_overlap_is_not_zero_error_evidence():
    assert audit.unique_overlap(np.empty((0, 3))) == ([], {})
    labels, duplicates = audit.unique_overlap(np.empty((2, 0)))
    assert all(x["status"] == "unmatched" for x in labels) and not duplicates


def test_timestamp_duration_and_interval_censoring():
    instance = audit.IdentityAudit("s")
    for t, tid in [(1_000_000, "a"), (1_100_000, "b"), (1_700_000, "b"), (2_000_000, "a")]:
        instance.step(t, ["g"], {"g": tid})
    event, = instance.finish()
    assert event["observed_error_span_seconds"] == .6
    assert event["recovery_observation_elapsed_seconds"] == .9
    assert event["onset_reference_to_return_observation_seconds"] == 1.
    assert not event["left_censored"] and not event["right_censored"]
    assert event["wrong_observations"] == 2


def test_unknown_gaps_never_filled_and_reentry_is_left_censored():
    instance = audit.IdentityAudit("s")
    for t, mapping in [(0, {"g": "a"}), (10, {"g": "b"}), (100, {}), (1000, {"g": "b"})]:
        instance.step(t, ["g"], mapping)
    first, second = instance.finish()
    assert first["closure_reason"] == "mapping_unknown" and first["right_censored"]
    assert first["observed_error_span_seconds"] == 0.
    assert second["left_censored"] and second["right_censored"]
    assert second["onset_lower_us"] is None and second["onset_reference_to_return_observation_seconds"] is None
    assert instance.counts["identity_unknown_gt_frames"] == 1


def test_gt_exit_and_first_id_anchor_limitations():
    instance = audit.IdentityAudit("s")
    instance.step(0, ["g"], {})
    instance.step(10, ["g"], {"g": "possibly-wrong-initial-id"})
    instance.step(20, ["g"], {"g": "b"})
    instance.step(100, [], {})
    event, = instance.finish()
    assert event["closure_reason"] == "gt_absent_or_outside_roi"
    assert event["anchor_track_id"] == "possibly-wrong-initial-id"
    assert instance.counts["anchored_gt_identities"] == 1


def test_identity_refuses_nonincreasing_time():
    instance = audit.IdentityAudit("s")
    instance.step(4, [], {})
    with pytest.raises(ValueError, match="strictly increase"):
        instance.step(4, [], {})


def test_known_wrong_pair_links_only_same_frame_provenance():
    adapter = audit.load_adapter()
    rows = [make_frame(0, wrong=True)]
    result, episodes = audit.analyze_sequence("seq", [gt_frame(0)], rows, adapter, "M0", "b" * 64, {}, .3)
    counts = result["counts"]
    assert counts["car_pair_wrong_occurrences"] == 2
    assert counts["car_pair_wrong_conditional_mass"] == 2.
    assert counts["roi_car_birth_touches_known_wrong_pair_node"] == 2
    assert episodes == []
    assert "gt_id" not in result["examples"][0]


def test_ambiguous_source_maps_are_unknown_not_forced_wrong_pairs():
    adapter = audit.load_adapter()
    p, a, d = make_frame(0, wrong=True)
    d["source_records"][1]["mean"] = d["source_records"][0]["mean"][:]
    seal(d, "diagnostic_commit_sha256")
    result, _ = audit.analyze_sequence("seq", [gt_frame(0)], [(p, a, d)], adapter, "M0", "b" * 64, {}, .3)
    assert result["counts"]["car_pair_unknown_occurrences"] == 2
    assert result["counts"].get("car_pair_wrong_occurrences", 0) == 0
    assert result["counts"]["source_car_ambiguous"] == 2


def test_car_kill_uses_prior_class_and_not_current_prediction_roi():
    adapter = audit.load_adapter()
    first = make_frame(0)
    p, a, d = make_frame(1, previous=first[0]["commit_sha256"],
                         diagnostic_previous=first[2]["diagnostic_commit_sha256"])
    killed = p["predictions"].pop(0)["track_id"]
    seal(p, "commit_sha256")
    d["prediction_commit_sha256"] = p["commit_sha256"]
    d["tracks_after"] = 1
    d["temporal_assignments"] = [d["temporal_assignments"][1]]
    d["events"] = [{"event": "kill_before_assignment", "track_id": killed}]
    seal(d, "diagnostic_commit_sha256")
    result, _ = audit.analyze_sequence("seq", [gt_frame(0), gt_frame(1)], [first, (p, a, d)],
                                      adapter, "M0", "b" * 64, {}, .3)
    assert result["counts"]["prior_car_class_kill_before_assignment"] == 1
    assert result["counts"]["prior_car_class_kill_before_assignment_with_previous_unique_gt"] == 1


def test_cross_class_pair_is_rejected():
    p, a, d = make_frame(0)
    d["source_records"][2]["class_index"] = 2
    seal(d, "diagnostic_commit_sha256")
    with pytest.raises(ValueError, match="cross-class"):
        audit.validate_frame(p, a, d, gt_frame(0), "M0", "b" * 64, audit.ZERO)


@pytest.mark.parametrize("field", ["plan_sha256", "prediction_commit_sha256", "association_frame_sha256", "previous_diagnostic_commit_sha256"])
def test_changed_binding_is_rejected_even_when_resealed(field):
    p, a, d = make_frame(0)
    d[field] = "e" * 64
    seal(d, "diagnostic_commit_sha256")
    with pytest.raises(ValueError, match="diagnostic identity"):
        audit.validate_frame(p, a, d, gt_frame(0), "M0", "b" * 64, audit.ZERO)


def test_component_cannot_claim_another_hypothesis_partner():
    p, a, d = make_frame(0)
    d["nodes"][0]["components"][0]["partner_selected_index"] = 1
    seal(d, "diagnostic_commit_sha256")
    with pytest.raises(ValueError, match="component association"):
        audit.validate_frame(p, a, d, gt_frame(0), "M0", "b" * 64, audit.ZERO)


def test_cross_group_source_drift_is_rejected():
    adapter = audit.load_adapter()
    cache = {}
    row = make_frame(0)
    audit.analyze_sequence("seq", [gt_frame(0)], [row], adapter, "M0", "b" * 64, cache, .3)
    changed = copy.deepcopy(row)
    changed[2]["source_records"][0]["score"] = .7
    seal(changed[2], "diagnostic_commit_sha256")
    with pytest.raises(ValueError, match="source records differ"):
        audit.analyze_sequence("seq", [gt_frame(0)], [changed], adapter, "M0", "b" * 64, cache, .3)


def test_noncanonical_json_and_duplicate_keys_rejected(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_bytes(b'{"a": 1}\n')
    with pytest.raises(ValueError, match="canonical"):
        list(audit.records(p))
    with pytest.raises(ValueError, match="duplicate JSON"):
        audit.decode(b'{"a":1,"a":2}')


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root, ground, output = tmp_path / "experiment", tmp_path / "gt", tmp_path / "audit"
    root.mkdir(); ground.mkdir()
    gt = [gt_frame(i) for i in range(3)]
    (ground / "ground-truth.jsonl").write_bytes(b"".join(audit.canonical(g)+b"\n" for g in gt))
    manifest = {"ground_truth_sha256": audit.sha(ground / "ground-truth.jsonl")}
    audit.write_json(ground / "manifest.json", manifest)
    monkeypatch.setattr(audit, "GT_SHA256", manifest["ground_truth_sha256"])
    monkeypatch.setattr(audit, "GT_MANIFEST_SHA256", audit.sha(ground / "manifest.json"))
    monkeypatch.setattr(audit, "EXPECTED_FRAMES", 3)
    monkeypatch.setattr(audit, "EXPECTED_SEQUENCES", 1)
    adapter = audit.load_adapter()
    monkeypatch.setattr(adapter, "load_ground_truth", lambda directory: (manifest, gt))
    monkeypatch.setattr(audit, "load_adapter", lambda: adapter)
    plan = {"kind": "mechanism_diagnostics_v2_plan", "reporting_scope": "car_only",
            "runs": [{"run_id": name, "mode": mode, "seed": seed, "deterministic_control": deterministic}
                     for name, (mode, seed, deterministic) in audit.RUNS.items()],
            "cache_sha256": audit.CACHE_SHA256, "schedule_sha256": audit.SCHEDULE_SHA256,
            "gt_model_inputs": False, "test_payloads_read": False, "val_parameter_fitting": False,
            "config": {"birth_score": .3}}
    audit.write_json(root / "frozen-plan.json", plan)
    plan_sha = audit.sha(root / "frozen-plan.json")
    summary = {"kind": "mechanism_diagnostics_v2_complete", "status": "completed", "plan_sha256": plan_sha,
               "runs": {}, "all_required_sealed_streams_match": True, "paper_eligible": False}
    for name, (mode, seed, deterministic) in audit.RUNS.items():
        directory = root / name
        directory.mkdir()
        rows, previous, diag_previous = [], None, None
        for i in range(3):
            row = make_frame(i, mode, previous, diag_previous, plan_sha)
            rows.append(row)
            previous, diag_previous = row[0]["commit_sha256"], row[2]["diagnostic_commit_sha256"]
        for k, filename in enumerate(("predictions.jsonl", "association.jsonl")):
            (directory / filename).write_bytes(b"".join(audit.canonical(r[k])+b"\n" for r in rows))
        with gzip.open(directory / "diagnostics.jsonl.gz", "wb") as stream:
            stream.write(b"".join(audit.canonical(r[2])+b"\n" for r in rows))
        receipt = {"run_id": name, "mode": mode, "seed": seed, "deterministic_control": deterministic,
                   "frames": 3, "sequences": 1, "sequence_commits": {"seq": previous},
                   "diagnostic_sequence_commits": {"seq": diag_previous},
                   "predictions_sha256": audit.sha(directory / "predictions.jsonl"),
                   "association_sha256": audit.sha(directory / "association.jsonl"),
                   "diagnostics_gzip_sha256": audit.sha(directory / "diagnostics.jsonl.gz")}
        audit.write_json(directory / "receipt.json", receipt)
        summary["runs"][name] = receipt
    audit.write_json(root / "summary.json", summary)
    return root, ground, output, plan_sha


def test_complete_ten_group_audit_and_descriptive_paired_summary(experiment):
    root, ground, output, digest = experiment
    result = audit.run(root, ground, output, digest)
    assert len(result["runs"]) == 10 and len(result["paired_comparisons"]) == 9
    assert result["causal_effect_identified"] is False
    assert result["runs"]["M1-all-unmatched"]["identity_evaluable_fraction"] == 0.
    assert result["runs"]["M1-all-unmatched"]["association_evaluable_fraction"] is None
    assert result["runs"]["M1-all-unmatched"]["counts"]["duplicate_gt_frames"] == 6
    complete = audit.read_json(output / "completion.json")
    assert all(audit.evidence(output / name) == entry for name, entry in complete["files"].items())
    assert set(p.name for p in output.iterdir()) == {"report.json", "sources.json", "sequences.json", "identity-episodes.json",
                                                      "README.md", "protocol.json", "completion.json"}
    serialized = (output / "identity-episodes.json").read_text()
    assert "mean" not in serialized and "covariance" not in serialized


def test_create_once_does_not_touch_existing_output(experiment):
    root, ground, output, digest = experiment
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserved")
    with pytest.raises(ValueError, match="already exists"):
        audit.run(root, ground, output, digest)
    assert sentinel.read_text() == "preserved"


def test_wrong_plan_pin_has_no_output(experiment):
    root, ground, output, digest = experiment
    with pytest.raises(ValueError, match="identity mismatch"):
        audit.run(root, ground, output, "f" * 64)
    assert not output.exists()


def test_modified_gzip_receipt_identity_rejected(experiment):
    root, ground, output, digest = experiment
    path = root / "M0-seed-1337" / "diagnostics.jsonl.gz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="file identity"):
        audit.run(root, ground, output, digest)
    assert not output.exists()


def test_input_modification_during_analysis_prevents_output(experiment, monkeypatch):
    root, ground, output, digest = experiment
    original = audit.analyze_sequence
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "M0-seed-1337" / "receipt.json").write_text("changed")
        return result
    monkeypatch.setattr(audit, "analyze_sequence", changed)
    with pytest.raises(ValueError, match="changed during"):
        audit.run(root, ground, output, digest)
    assert not output.exists()


def test_runtime_pin_before_reading_production_data(experiment, monkeypatch):
    root, ground, output, digest = experiment
    import shapely
    monkeypatch.setattr(shapely, "__version__", "0.0")
    with pytest.raises(ValueError, match="geometry runtime"):
        audit.run(root, ground, output, digest)
