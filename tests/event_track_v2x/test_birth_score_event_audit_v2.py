"""M4-only schema adaptation; sealed shared event/geometry logic stays untouched."""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("birth_event_test_subject", "tools/event_track_v2x/analyze_birth_score_events_v2.py")
fixture_helpers = load("birth_event_frozen_fixtures", "tests/event_track_v2x/test_mechanism_event_audit_v2.py")
core = fixture_helpers.audit
control = audit.load_dependencies()[1]


def m4_frame(i=0, *, previous=None, diagnostic_previous=None, plan_sha="b" * 64, residual=False):
    p, a, d = fixture_helpers.make_frame(i, "M1" if residual else "M0", previous,
                                        diagnostic_previous, plan_sha)
    d["kind"], d["mode"] = "tracking_birth_score_v2_frame", "M4"
    if residual:
        unmatched = {**a["hypotheses"][0], "weight": .25}
        matched = {"pairs": [[0, 0], [1, 1]], "unmatched_left": [], "unmatched_right": [],
                   "weight": .75, "energy": 0.}
        a["hypotheses"] = [matched, unmatched]
        for node in d["nodes"]:
            node["matched_mass"], node["unmatched_mass"] = .75, .25
            if node["kind"] == "road_residual":
                node["score"] = .2
            else:
                node["components"] = [{"hypothesis_index": k, "weight": h["weight"],
                                       "partner_selected_index": dict(h["pairs"]).get(node["source_selected_index"])}
                                      for k, h in enumerate(a["hypotheses"])]
        d["association_frame_sha256"] = hashlib.sha256(core.canonical(a)).hexdigest()
    for node in d["nodes"]:
        node["birth_effective_score"] = node["original_score"] if node["kind"] == "road_residual" else node["score"]
    for event in d["events"]:
        if event["event"] in ("birth", "birth_rejected_score"):
            node = d["nodes"][event["node_index"]]
            event.update(score=node["birth_effective_score"], node_score=node["score"],
                         birth_effective_score=node["birth_effective_score"])
    fixture_helpers.seal(d, "diagnostic_commit_sha256")
    return p, a, d


def analyze(row):
    adapter = core.load_adapter()
    fn, channel = audit.adapted_sequence(core, adapter, .3)
    result, events = fn("seq", [fixture_helpers.gt_frame(0)], [row], adapter, "M4", "b" * 64, {}, .3)
    return result, events, channel


def test_dependencies_retain_frozen_hashes_and_do_not_import_tracker():
    imported, controller = audit.load_dependencies()
    assert imported.evidence(audit.CORE_PATH)["sha256"] == audit.CORE_SHA256
    assert imported.evidence(audit.CONTROL_PATH)["sha256"] == audit.CONTROL_SHA256
    assert "torch" not in imported.__dict__ and "torch" not in controller.__dict__


def test_private_adapter_preserves_common_score_real_seal_and_module_globals():
    original_validator = core.validate_frame
    original_globals = core.analyze_sequence.__globals__["validate_frame"]
    row = m4_frame(residual=True)
    before = copy.deepcopy(row)
    result, episodes, channel = analyze(row)
    assert row == before and core.validate_frame is original_validator
    assert core.analyze_sequence.__globals__["validate_frame"] is original_globals
    assert result["diagnostic_final_commit_sha256"] == row[2]["diagnostic_commit_sha256"]
    assert result["counts"]["roi_residual_score_suppressed_below_birth_threshold"] == 2
    assert result["counts"]["roi_car_birth"] == 4
    assert result["counts"]["duplicate_gt_frames"] == 2
    assert channel["roi_residual_raised_birth_eligibility"] == 2
    assert channel["roi_residual_raised_eligibility_actual_birth"] == 2
    assert channel["roi_residual_raised_birth_eligibility_ambiguous"] == 2
    assert channel["roi_residual_raised_birth_eligibility_unique"] == 0
    assert not episodes  # Duplication is unknown, not a forced identity assignment.


def test_real_seal_checked_before_compatibility_reseal():
    row = m4_frame()
    row[2]["nodes"][0]["birth_effective_score"] = .9
    with pytest.raises(ValueError, match="real commit-chain"):
        analyze(row)


@pytest.mark.parametrize("field", ["kind", "mode", "previous_diagnostic_commit_sha256", "prediction_commit_sha256",
                                   "association_frame_sha256", "plan_sha256"])
def test_resealed_invalid_source_binding_rejected(field):
    row = m4_frame()
    row[2][field] = "invalid"
    fixture_helpers.seal(row[2], "diagnostic_commit_sha256")
    with pytest.raises(ValueError, match="identity|commit-chain"):
        analyze(row)


@pytest.mark.parametrize("field,value", [("score", .8), ("original_score", .9),
                                         ("unmatched_mass", .4), ("birth_effective_score", .2)])
def test_road_birth_and_common_score_provenance_rejected(field, value):
    row = m4_frame(residual=True)
    row[2]["nodes"][2][field] = value
    fixture_helpers.seal(row[2], "diagnostic_commit_sha256")
    with pytest.raises(ValueError, match="provenance"):
        analyze(row)


@pytest.mark.parametrize("field", ["score", "node_score", "birth_effective_score"])
def test_birth_event_score_channels_are_checked(field):
    row = m4_frame(residual=True)
    row[2]["events"][2][field] = .6
    fixture_helpers.seal(row[2], "diagnostic_commit_sha256")
    with pytest.raises(ValueError, match="provenance"):
        analyze(row)


def test_wrong_effective_threshold_outcome_rejected():
    row = m4_frame()
    row[2]["events"][0]["event"] = "birth_rejected_score"
    fixture_helpers.seal(row[2], "diagnostic_commit_sha256")
    with pytest.raises(ValueError, match="effective threshold"):
        analyze(row)


@pytest.mark.parametrize("operation", ["duplicate", "omit", "assigned_birth"])
def test_birth_decision_coverage_and_exclusivity(operation):
    row = m4_frame()
    if operation == "duplicate":
        row[2]["events"].append(copy.deepcopy(row[2]["events"][0]))
    elif operation == "omit":
        row[2]["events"].pop()
    else:
        row[2]["temporal_assignments"] = [{"event": "matched", "node_index": 0, "track_id": "seq:000001"}]
    fixture_helpers.seal(row[2], "diagnostic_commit_sha256")
    with pytest.raises(ValueError, match="birth|covered"):
        analyze(row)


def test_m4_chain_and_nonuniform_microsecond_identity_spans():
    adapter = core.load_adapter()
    rows, ground = [], []
    for i, (timestamp, tid) in enumerate([(1000000, "seq:000001"), (1100000, "seq:000003"), (1800000, "seq:000003")]):
        p, a, d = m4_frame(i, previous=rows[-1][0]["commit_sha256"] if rows else None,
                          diagnostic_previous=rows[-1][2]["diagnostic_commit_sha256"] if rows else None)
        g = fixture_helpers.gt_frame(i, timestamp)
        p["box_reference_timestamp_us"], p["decision_timestamp_us"] = timestamp, timestamp + 100000
        p["predictions"][0] = fixture_helpers.pred_box(tid, 0)
        fixture_helpers.seal(p, "commit_sha256")
        d.update(box_reference_timestamp_us=timestamp, decision_timestamp_us=timestamp+100000,
                 prediction_commit_sha256=p["commit_sha256"])
        if i == 1:
            d["temporal_assignments"] = [d["temporal_assignments"][1]]
            d["events"] = [{"event": "kill_after_miss", "track_id": "seq:000001"},
                           {"event": "birth", "track_id": tid, "node_index": 0, "score": .8,
                            "node_score": .8, "birth_effective_score": .8}]
        elif i == 2:
            d["temporal_assignments"][0]["track_id"] = tid
        fixture_helpers.seal(d, "diagnostic_commit_sha256")
        rows.append((p, a, d)); ground.append(g)
    fn, _ = audit.adapted_sequence(core, adapter, .3)
    result, episodes = fn("seq", ground, rows, adapter, "M4", "b" * 64, {}, .3)
    event, = episodes
    assert event["observed_error_span_seconds"] == .7
    assert event["right_censored"] and not event["left_censored"]
    assert result["counts"]["prior_car_class_kill_after_miss"] == 1


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Small cohort overrides only the separately tested runner/GT cohort gate."""
    old, ground, reference, old_digest = fixture_helpers.experiment.__wrapped__(tmp_path, monkeypatch)
    core.run(old, ground, reference, old_digest)
    birth = tmp_path / "birth"
    birth.mkdir()
    core.write_json(birth / "frozen-plan.json", {"config": {"birth_score": .3}, "current_source_hashes": {}})
    digest = core.sha(birth / "frozen-plan.json")
    receipts = {}
    for spec in control.RUNS:
        name = spec["run_id"]
        directory = birth / name
        directory.mkdir()
        rows = []
        for i in range(3):
            rows.append(m4_frame(i, previous=rows[-1][0]["commit_sha256"] if rows else None,
                                diagnostic_previous=rows[-1][2]["diagnostic_commit_sha256"] if rows else None, plan_sha=digest))
        for k, filename in enumerate(("predictions.jsonl", "association.jsonl", "diagnostics.jsonl.gz")):
            data = b"".join(core.canonical(row[k])+b"\n" for row in rows)
            (directory / filename).write_bytes(gzip.compress(data) if k == 2 else data)
        receipts[name] = {"sequence_commits": {"seq": rows[-1][0]["commit_sha256"]},
                          "diagnostic_sequence_commits": {"seq": rows[-1][2]["diagnostic_commit_sha256"]}}
        core.write_json(directory / "receipt.json", receipts[name])
    core.write_json(birth / "summary.json", {"runs": receipts})
    def inventory(root, names):
        return {filename: control.evidence(root / filename) for filename in ("frozen-plan.json", "summary.json")} | {
            name: {filename: control.evidence(root / name / filename) for filename in
                   ("receipt.json", "predictions.jsonl", "association.jsonl", "diagnostics.jsonl.gz")} for name in names}
    new_sources, old_sources = inventory(birth, receipts), inventory(old, core.RUNS)
    monkeypatch.setattr(control, "validate_birth_experiment", lambda *args: (
        core.read_json(birth / "frozen-plan.json"), receipts, new_sources, {}, old_sources))
    monkeypatch.setattr(audit, "load_dependencies", lambda: (core, control))
    return {"experiment_root": birth, "mechanism_root": old, "reference_event_audit": reference,
            "ground_truth": ground, "output": tmp_path / "birth-audit", "expected_plan_sha256": digest,
            "expected_reference_report_sha256": core.sha(reference / "report.json")}


def test_complete_three_seed_audit_outputs_and_reference_paired_counts(bundle):
    result = audit.run(**bundle)
    assert len(result["runs"]) == 3 and len(result["paired_comparisons"]) == 6
    assert all(all(v == 0 for v in pair["total_difference"].values()) for pair in result["paired_comparisons"])
    assert all(not pair["birth_channel_counters_compared"] for pair in result["paired_comparisons"])
    assert not result["causal_effect_identified"] and not result["published_to_clearml"]
    output = bundle["output"]
    assert {p.name for p in output.iterdir()} == audit.REFERENCE_FILES
    complete = core.read_json(output / "completion.json")
    assert all(core.evidence(output / name) == entry for name, entry in complete["files"].items())
    assert "mean" not in (output / "identity-episodes.json").read_text()
    assert not any(output.rglob("*.jsonl*"))


@pytest.mark.parametrize("file", ["report.json", "sequences.json", "sources.json", "completion.json"])
def test_modified_reference_audit_rejected_before_output(bundle, file):
    path = bundle["reference_event_audit"] / file
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="pin|hashes"):
        audit.run(**bundle)
    assert not bundle["output"].exists()


def test_reference_sequence_sums_must_reproduce_pinned_report(bundle):
    root = bundle["reference_event_audit"]
    sequences = core.read_json(root / "sequences.json")
    sequences["M0-seed-1337"]["seq"]["counts"]["frames"] += 1
    (root / "sequences.json").write_bytes(core.canonical(sequences)+b"\n")
    complete = core.read_json(root / "completion.json")
    complete["files"]["sequences.json"] = core.evidence(root / "sequences.json")
    (root / "completion.json").write_bytes(core.canonical(complete)+b"\n")
    with pytest.raises(ValueError, match="reproduce pinned report"):
        audit.run(**bundle)


def test_create_once_preserves_existing_files(bundle):
    bundle["output"].mkdir()
    sentinel = bundle["output"] / "keep"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="create-once"):
        audit.run(**bundle)
    assert sentinel.read_text() == "keep"


def test_output_must_not_mutate_input_tree(bundle):
    bundle["output"] = bundle["reference_event_audit"] / "new"
    with pytest.raises(ValueError, match="immutable input"):
        audit.run(**bundle)


def test_late_input_change_prevents_output(bundle, monkeypatch):
    original = audit.adapted_sequence
    def changed(*args):
        function, channel = original(*args)
        def wrapped(*inner):
            result = function(*inner)
            (bundle["mechanism_root"] / "M0-seed-1337/receipt.json").write_text("changed")
            return result
        return wrapped, channel
    monkeypatch.setattr(audit, "adapted_sequence", changed)
    with pytest.raises(ValueError, match="changed during"):
        audit.run(**bundle)
    assert not bundle["output"].exists()


def test_wrong_gt_hash_prevents_output(bundle):
    path = bundle["ground_truth"] / "ground-truth.jsonl"
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="GT input hashes"):
        audit.run(**bundle)


def test_wrong_runtime_prevents_output(bundle, monkeypatch):
    import shapely
    monkeypatch.setattr(shapely, "__version__", "bad")
    with pytest.raises(ValueError, match="geometry runtime"):
        audit.run(**bundle)
