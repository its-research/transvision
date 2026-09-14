"""Create-once M4 car evaluation, frozen references and non-additive contrasts."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_mechanism_diagnostics_evaluation_v2 import bundle as old_bundle
from test_mechanism_diagnostics_evaluation_v2 import evaluation as old_evaluation
from test_mechanism_diagnostics_evaluation_v2 import metric_fixture, put

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("birth_score_evaluation_tests",
    ROOT / "tools/event_track_v2x/evaluate_birth_score_diagnostic_v2.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


@pytest.fixture
def bundle(old_bundle, tmp_path, monkeypatch):
    old = old_bundle
    monkeypatch.setattr(evaluation, "load_base", lambda: old_evaluation)
    fake_source = tmp_path / "source"
    fake_source.mkdir()
    for name in ("birth.py", "runner.py"):
        (fake_source / name).write_text("# immutable fixture\n")
    monkeypatch.setattr(evaluation, "ROOT", fake_source)
    monkeypatch.setattr(evaluation, "BIRTH_SOURCE_PATHS", ("birth.py", "runner.py"))
    monkeypatch.setattr(evaluation, "BIRTH_SOURCE_PINS", {name: evaluation.evidence(fake_source / name)["sha256"]
                                                       for name in ("birth.py", "runner.py")})
    common = {"cache_sha256": "c" * 64, "schedule_sha256": "d" * 64, "split_sha256": "e" * 64,
              "calibration_sha256": "a" * 64, "checkpoints": {str(s): "b" * 64 for s in evaluation.SEEDS},
              "config": {"birth_score": .3}, "candidate_selection": "unchanged all-class raw_score>=0.05 top64; car-only metrics",
              "condition": "clean-link", "torch_version": "frozen"}
    old_plan = old_evaluation.read_json(old.root / "frozen-plan.json")
    old_plan.update(common)
    put(old.root / "frozen-plan.json", old_plan)
    old_summary = old_evaluation.read_json(old.root / "summary.json")
    old_summary["plan_sha256"] = old_evaluation.sha(old.root / "frozen-plan.json")
    for name, receipt in old_summary["runs"].items():
        receipt["source_sensor_late"] = [1643, 0]
        receipt["selected_detections"] = [46867, 77194]
        put(old.root / name / "receipt.json", receipt)
    put(old.root / "summary.json", old_summary)
    old_evaluation.compare(old.root, old.reports, old.reference, old.reports / "comparison")
    root, reports = tmp_path / "M4", tmp_path / "M4-reports"
    plan = {**common, "kind": "birth_score_diagnostics_v2_plan", "runs": evaluation.RUNS,
            "reporting_scope": "car_only", "paper_eligible": False, "gt_model_inputs": False,
            "test_payloads_read": False, "val_parameter_fitting": False, "seed_selection": False,
            "official_validation_frames": 3316, "sequences": 21,
            "current_source_hashes": {name: evaluation.evidence(fake_source / name)["sha256"] for name in evaluation.BIRTH_SOURCE_PATHS},
            "reference_mechanism_root": str(old.root.resolve()),
            "reference_mechanism_plan_sha256": evaluation.evidence(old.root / "frozen-plan.json")["sha256"],
            "reference_mechanism_summary_sha256": evaluation.evidence(old.root / "summary.json")["sha256"]}
    put(root / "frozen-plan.json", plan)
    digest = evaluation.evidence(root / "frozen-plan.json")["sha256"]
    summary = {"kind": "birth_score_diagnostics_v2_complete", "status": "completed", "plan_sha256": digest,
               "reporting_scope": "car_only", "paper_eligible": False, "weights_unchanged": True,
               "all_required_sealed_streams_match": True, "runs": {}}
    for spec in evaluation.RUNS:
        name, seed = spec["run_id"], spec["seed"]
        baseline = old_summary["runs"][f"M0-seed-{seed}"]
        folder = root / name
        value = .35 + evaluation.SEEDS.index(seed) * .01
        put(folder / "predictions.jsonl", {"M4_prediction": value})
        (folder / "association.jsonl").write_bytes((old.root / f"M0-seed-{seed}" / "association.jsonl").read_bytes())
        (folder / "diagnostics.jsonl.gz").write_bytes(b"M4-diagnostic-fixture-not-read-by-metric-comparison")
        receipt = {**copy.deepcopy(baseline), **spec, "sealed_prediction_parity": None, "reference_M0_association_parity": True,
                   "predictions_sha256": evaluation.evidence(folder / "predictions.jsonl")["sha256"],
                   "diagnostics_gzip_sha256": evaluation.evidence(folder / "diagnostics.jsonl.gz")["sha256"],
                   "diagnostics_gzip_bytes": (folder / "diagnostics.jsonl.gz").stat().st_size}
        put(folder / "receipt.json", receipt)
        summary["runs"][name] = receipt
        report = old_evaluation.read_json(old.reports / f"M0-seed-{seed}" / "report.json")
        metrics = metric_fixture(value)
        put(reports / name / "metrics.json", metrics)
        put(reports / name / "runtime.json", {})
        put(reports / name / "golden-cases.json", {"passed": True})
        report["predictions_sha256"] = receipt["predictions_sha256"]
        report["input_sources"]["predictions"] = evaluation.evidence(folder / "predictions.jsonl")
        report["primary_car"] = old.source.car_metrics(metrics)
        report["files"] = {n: evaluation.evidence(reports / name / n) for n in ("metrics.json", "runtime.json", "golden-cases.json")}
        put(reports / name / "report.json", report)
    put(root / "summary.json", summary)
    return SimpleNamespace(root=root, reports=reports, mechanism_root=old.root, mechanism_reports=old.reports,
                           digest=digest, output=tmp_path / "M4-comparison", source=old.source, fake_source=fake_source)


def compare(b):
    return evaluation.compare(b.root, b.reports, b.mechanism_root, b.mechanism_reports, b.digest, b.output)


def reseal_plan(b, mutate):
    plan = evaluation.read_json(b.root / "frozen-plan.json")
    mutate(plan)
    put(b.root / "frozen-plan.json", plan)
    b.digest = evaluation.evidence(b.root / "frozen-plan.json")["sha256"]
    summary = evaluation.read_json(b.root / "summary.json")
    summary["plan_sha256"] = b.digest
    put(b.root / "summary.json", summary)


def test_three_seeds_and_both_paired_contrasts_are_descriptive(bundle):
    result = compare(bundle)
    assert len(result["run_metrics"]) == 3
    assert result["M4_descriptive"]["HOTA"]["n"] == 3
    assert result["M4_descriptive"]["HOTA"]["sample_sd"] == pytest.approx(.01)
    for seed in evaluation.SEEDS:
        for metric in ("HOTA", "FP", "FN", "AMOTP_m"):
            assert result["paired_contrasts"]["M4_minus_same_seed_M0"][str(seed)]["metric_difference"][metric] == pytest.approx(.05)
            assert result["paired_contrasts"]["M4_minus_same_seed_M3"][str(seed)]["metric_difference"][metric] == pytest.approx(.15)
            assert result["paired_contrasts"]["M3_minus_same_seed_M4"][str(seed)]["metric_difference"][metric] == pytest.approx(-.15)
        signs = result["paired_contrasts"]["M4_minus_same_seed_M0"][str(seed)]["paired_sequence_sign_counts"]["HOTA"]
        assert signs == {"positive": 21, "zero": 0, "negative": 0}
    assert result["causal_additivity_assumed"] is False and result["direct_temporal_effect_identified"] is False
    assert result["paper_eligible"] is False and result["independent_event_audit_required"] is True
    assert "not an additive" in result["interpretation"]["M3_minus_M4"]
    assert result["source_manifest_sha256"] == evaluation.evidence(bundle.output / "source-manifest.json")["sha256"]
    assert set(p.name for p in bundle.output.iterdir()) == {"comparison.json", "source-manifest.json", "README.md"}
    with pytest.raises(ValueError, match="create-once"):
        compare(bundle)


@pytest.mark.parametrize("field", ["reference_mechanism_plan_sha256", "reference_mechanism_summary_sha256", "reference_mechanism_root"])
def test_wrong_reference_binding_rejected(bundle, field):
    reseal_plan(bundle, lambda p: p.update({field: "wrong"}))
    with pytest.raises(ValueError, match="reference mechanism"):
        compare(bundle)
    assert not bundle.output.exists()


@pytest.mark.parametrize("field", evaluation.COMMON_INPUT_KEYS)
def test_frozen_input_difference_rejected(bundle, field):
    reseal_plan(bundle, lambda p: p.update({field: "different"}))
    with pytest.raises(ValueError, match="frozen input/config"):
        compare(bundle)


def test_wrong_plan_pin_prevents_output(bundle):
    bundle.digest = "0" * 64
    with pytest.raises(ValueError, match="incompatible"):
        compare(bundle)
    assert not bundle.output.exists()


@pytest.mark.parametrize("filename", ["predictions.jsonl", "association.jsonl", "diagnostics.jsonl.gz", "receipt.json"])
def test_modified_output_hash_or_receipt_rejected(bundle, filename):
    path = bundle.root / "M4-seed-2027" / filename
    if filename == "receipt.json":
        receipt = evaluation.read_json(path); receipt["frames"] = 3315
        put(path, receipt)
    else:
        with path.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        compare(bundle)
    assert not bundle.output.exists()


def test_m4_dependency_drift_rejected(bundle):
    (bundle.fake_source / "birth.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source dependency"):
        compare(bundle)


def test_m4_source_pin_cannot_be_bypassed_by_resealing_plan(bundle):
    path = bundle.fake_source / "birth.py"
    path.write_text("# new implementation\n")
    reseal_plan(bundle, lambda p: p["current_source_hashes"].update({"birth.py": evaluation.evidence(path)["sha256"]}))
    with pytest.raises(ValueError, match="implementation pin"):
        compare(bundle)


def test_reference_snapshot_cannot_change_between_launch_and_compare(bundle):
    with pytest.raises(ValueError, match="since frozen execution plan"):
        evaluation.compare(bundle.root, bundle.reports, bundle.mechanism_root, bundle.mechanism_reports,
                           bundle.digest, bundle.output, expected_inputs={"runner_sources": {}})
    assert not bundle.output.exists()


def test_candidate_count_difference_even_with_updated_receipts_rejected(bundle):
    path = bundle.root / "M4-seed-1337" / "receipt.json"
    receipt = evaluation.read_json(path)
    receipt["selected_detections"][0] -= 1
    put(path, receipt)
    summary = evaluation.read_json(bundle.root / "summary.json")
    summary["runs"]["M4-seed-1337"] = receipt
    put(bundle.root / "summary.json", summary)
    with pytest.raises(ValueError, match="availability/candidate"):
        compare(bundle)


def test_m4_association_must_match_both_controls(bundle):
    folder = bundle.root / "M4-seed-1337"
    put(folder / "association.jsonl", {"new_association": True})
    receipt = evaluation.read_json(folder / "receipt.json")
    receipt["association_sha256"] = evaluation.evidence(folder / "association.jsonl")["sha256"]
    put(folder / "receipt.json", receipt)
    summary = evaluation.read_json(bundle.root / "summary.json")
    summary["runs"]["M4-seed-1337"] = receipt
    put(bundle.root / "summary.json", summary)
    with pytest.raises(ValueError, match="same-seed M0/M3"):
        compare(bundle)


@pytest.mark.parametrize("which", ["M4", "M0", "M3"])
def test_metric_artifact_tampering_rejected(bundle, which):
    path = (bundle.reports if which == "M4" else bundle.mechanism_reports) / f"{which}-seed-1337" / "metrics.json"
    put(path, metric_fixture(.987))
    with pytest.raises(ValueError, match="artifact hash"):
        compare(bundle)


def test_reference_manifest_binding_rejected(bundle):
    path = bundle.mechanism_reports / "comparison/comparison.json"
    data = evaluation.read_json(path); data["source_manifest_sha256"] = "0" * 64
    put(path, data)
    with pytest.raises(ValueError, match="comparison/source-manifest"):
        compare(bundle)


def test_final_recheck_catches_reference_changed_after_validation(bundle, monkeypatch):
    original = old_evaluation.contrast
    mutated = False
    def changed(*args, **kwargs):
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            path = bundle.mechanism_reports / "M0-seed-1337" / "report.json"
            with path.open("ab") as stream:
                stream.write(b"changed")
            mutated = True
        return result
    monkeypatch.setattr(old_evaluation, "contrast", changed)
    with pytest.raises(ValueError, match="changed during"):
        compare(bundle)
    assert not bundle.output.exists()


def test_output_cannot_be_nested_in_immutable_inputs(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    with pytest.raises(ValueError, match="immutable input"):
        evaluation.protect_output(root / "output", root)


def test_evaluate_worker_only_calls_frozen_car_evaluate(tmp_path, monkeypatch):
    evaluate = Mock()
    monkeypatch.setattr(evaluation, "load_base", lambda: SimpleNamespace(load_evaluator=lambda: SimpleNamespace(evaluate=evaluate)))
    args = SimpleNamespace(ground_truth=tmp_path / "gt", predictions=tmp_path / "predictions", output=tmp_path / "output")
    evaluation.worker(args)
    evaluate.assert_called_once_with(args.ground_truth, args.predictions, args.output)


def args_for(b, tmp_path, workers=3):
    return SimpleNamespace(root=b.root, mechanism_root=b.mechanism_root, mechanism_reports_root=b.mechanism_reports,
                           ground_truth=tmp_path / "gt", output=tmp_path / "new-evaluation", max_workers=workers,
                           expected_plan_sha256=b.digest)


@pytest.mark.parametrize("workers", [0, 4, True])
def test_cpu_worker_limit_rejected_before_launch(bundle, tmp_path, workers, monkeypatch):
    launch = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(evaluation.subprocess, "run", launch)
    with pytest.raises(ValueError, match="max-workers"):
        evaluation.run(args_for(bundle, tmp_path, workers))
    launch.assert_not_called()


def test_controller_records_three_cpu_commands_and_preserves_failures(bundle, tmp_path, monkeypatch):
    args = args_for(bundle, tmp_path)
    args.ground_truth.mkdir()
    put(args.ground_truth / "manifest.json", {"kind": "fixture"})
    (args.ground_truth / "ground-truth.jsonl").write_bytes(b"fixture\n")
    monkeypatch.setattr(bundle.source, "GT_MANIFEST_SHA256", evaluation.evidence(args.ground_truth / "manifest.json")["sha256"])
    monkeypatch.setattr(bundle.source, "GT_SHA256", evaluation.evidence(args.ground_truth / "ground-truth.jsonl")["sha256"])
    monkeypatch.setattr(evaluation, "reference_reports", lambda *a: ({}, {}))
    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(evaluation.subprocess, "run", launch)
    with pytest.raises(RuntimeError, match="failed"):
        evaluation.run(args)
    assert len(calls) == 3
    assert all(env["CUDA_VISIBLE_DEVICES"] == "" and env["OMP_NUM_THREADS"] == "1" for _, env in calls)
    assert all("evaluate-one" in command for command, _ in calls)
    execution = evaluation.read_json(args.output / "execution.json")
    assert execution["status"] == "failed" and set(execution["exit_codes"].values()) == {2}
    assert len(list((args.output / "logs").iterdir())) == 3
    assert not (args.output / "comparison").exists()
