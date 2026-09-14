"""Car-only routing, sealed inputs and fail-closed five-condition comparison."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "source_ablation_eval_tests", ROOT / "tools/event_track_v2x/evaluate_source_ablation_v2.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def overwrite(path, value):
    path.write_bytes(evaluation.canonical(value) + b"\n")


def metric_fixture(value=.5):
    nuscenes = {key: {"car": value} for group, key in evaluation.PRIMARY_METRICS.values() if group == "nuscenes"}
    trackeval = {key: value for group, key in evaluation.PRIMARY_METRICS.values() if group == "trackeval"}
    return {"nuscenes": {"label_metrics": nuscenes},
            "trackeval": {"car": {"summary": trackeval,
                "sequences": {f"{n:04d}": {"HOTA": {key: [value] * 19 for key in ("HOTA", "AssA", "DetA")},
                                            "Identity": {"IDF1": value}} for n in range(21)}}}}


@pytest.fixture
def fake_adapter(monkeypatch):
    sealed = evaluation.load_adapter()
    adapter = SimpleNamespace(PROTOCOL=copy.deepcopy(sealed.PROTOCOL),
                              runtime_evidence=Mock(return_value={}),
                              golden_cases=Mock(return_value={"passed": True}),
                              load_ground_truth=Mock(return_value=({"ground_truth_sha256": evaluation.GT_SHA256}, ["GT"])),
                              validate_predictions=Mock(return_value={"frames": 3316, "sequences": 21,
                                  "predictions_per_class": {"car": 3, "bicycle": 2}, "empty_frames": 0,
                                  "final_sequence_commit_sha256": {f"{n:04d}": "f" * 64 for n in range(21)}}),
                              compute_metrics=Mock(return_value=metric_fixture()))
    monkeypatch.setattr(evaluation, "load_adapter", lambda: adapter)
    monkeypatch.setattr(evaluation, "validate_runtime", lambda runtime: None)
    return adapter


@pytest.fixture
def eval_inputs(tmp_path, monkeypatch):
    gt = tmp_path / "gt"
    gt.mkdir()
    overwrite(gt / "manifest.json", {"manifest": True})
    overwrite(gt / "ground-truth.jsonl", {"ground_truth": True})
    predictions = tmp_path / "predictions.jsonl"
    overwrite(predictions, {"prediction": True})
    monkeypatch.setattr(evaluation, "GT_MANIFEST_SHA256", evaluation.sha(gt / "manifest.json"))
    monkeypatch.setattr(evaluation, "GT_SHA256", evaluation.sha(gt / "ground-truth.jsonl"))
    return gt, predictions, tmp_path / "evaluated"


def test_car_only_without_changing_sealed_protocol(fake_adapter, eval_inputs):
    gt, predictions, output = eval_inputs
    original = copy.deepcopy(fake_adapter.PROTOCOL)
    fake_adapter.load_ground_truth.return_value = ({"ground_truth_sha256": evaluation.GT_SHA256}, ["GT"])
    report = evaluation.evaluate(gt, predictions, output)
    fake_adapter.compute_metrics.assert_called_once_with(["GT"], [{"prediction": True}], classes=("car",))
    fake_adapter.validate_predictions.assert_called_once_with([{"prediction": True}], ["GT"])
    assert fake_adapter.PROTOCOL == original
    assert report["protocol"]["evaluated_classes"] == ["car"]
    assert report["protocol"]["supplementary_classes"] == []
    assert report["protocol"]["roi"]["class_range_m"] == {"car": 50.}
    assert report["protocol"]["nuscenes"] == original["nuscenes"]
    assert report["protocol"]["trackeval"] == original["trackeval"]
    assert report["coverage"]["all_input_predictions"] == 5
    assert report["coverage"]["car_predictions"] == 3
    assert "predictions_per_class" not in report["coverage"]
    assert set(p.name for p in output.iterdir()) == {"report.json", "metrics.json", "runtime.json", "golden-cases.json"}
    assert all(evaluation.evidence(output / k) == v for k, v in report["files"].items())


def test_refuse_existing_even_empty_output(fake_adapter, eval_inputs):
    gt, predictions, output = eval_inputs
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        evaluation.evaluate(gt, predictions, output)
    fake_adapter.compute_metrics.assert_not_called()


def test_wrong_gt_pin_fails_before_metrics(fake_adapter, eval_inputs):
    gt, predictions, output = eval_inputs
    overwrite(gt / "ground-truth.jsonl", {"changed": True})
    with pytest.raises(ValueError, match="sealed official-validation"):
        evaluation.evaluate(gt, predictions, output)
    fake_adapter.compute_metrics.assert_not_called()
    assert not output.exists()


def test_input_change_during_metrics_prevents_publication(fake_adapter, eval_inputs):
    gt, predictions, output = eval_inputs
    def mutate(*args, **kwargs):
        overwrite(predictions, {"changed": True})
        return metric_fixture()
    fake_adapter.compute_metrics.side_effect = mutate
    with pytest.raises(ValueError, match="changed during"):
        evaluation.evaluate(gt, predictions, output)
    assert not output.exists()


@pytest.mark.parametrize("case", ["coverage", "golden", "class"])
def test_failed_prerequisite_or_class_scope_has_no_output(fake_adapter, eval_inputs, case):
    gt, predictions, output = eval_inputs
    if case == "coverage":
        fake_adapter.validate_predictions.return_value["frames"] = 3315
    elif case == "golden":
        fake_adapter.golden_cases.return_value = {"passed": False}
    else:
        fake_adapter.compute_metrics.return_value["trackeval"]["bicycle"] = {}
    with pytest.raises(ValueError):
        evaluation.evaluate(gt, predictions, output)
    assert not output.exists()


def test_sealed_adapter_hash_is_enforced(tmp_path, monkeypatch):
    altered = tmp_path / "adapter.py"
    altered.write_text("raise AssertionError('must not import')\n")
    monkeypatch.setattr(evaluation, "ADAPTER_PATH", altered)
    with pytest.raises(ValueError, match="adapter source hash"):
        evaluation.load_adapter()


def test_runtime_requires_versions_trees_and_inventory(monkeypatch):
    files = [{"path": "sample.py", "sha256": "a" * 64}]
    digest = hashlib.sha256(evaluation.canonical(files)).hexdigest()
    monkeypatch.setattr(evaluation, "RUNTIME_TREES", {"engine": digest})
    runtime = {"versions": dict(evaluation.RUNTIME_VERSIONS), "adapter_source_sha256": evaluation.ADAPTER_SHA256,
               "source_trees": {"engine": {"source_tree_sha256": digest, "files": files}}}
    evaluation.validate_runtime(runtime)
    changed = copy.deepcopy(runtime)
    changed["versions"]["numpy"] = "2.0.0"
    with pytest.raises(ValueError, match="runtime differs"):
        evaluation.validate_runtime(changed)
    runtime["source_trees"]["engine"]["files"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="inventory hash"):
        evaluation.validate_runtime(runtime)


@pytest.fixture
def comparison_inputs(tmp_path, fake_adapter):
    root, reports = tmp_path / "experiment", tmp_path / "reports"
    root.mkdir(); reports.mkdir()
    plan = {"kind": "source_ablation_v2_plan", "reporting_scope": "car_only",
            "runs": [{"run_id": name, "agent_mask": mask, "seed": seed, "deterministic_control": deterministic}
                     for name, (mask, seed, deterministic) in evaluation.RUNS.items()]}
    overwrite(root / "frozen-plan.json", plan)
    summary = {"kind": "source_ablation_v2_complete", "status": "completed", "reporting_scope": "car_only",
               "paper_eligible": False, "plan_sha256": evaluation.sha(root / "frozen-plan.json"),
               "weights_unchanged": True, "all_cooperative_streams_match_sealed": True, "runs": {}}
    for index, (run_id, (mask, seed, deterministic)) in enumerate(evaluation.RUNS.items()):
        run_dir, report_dir = root / run_id, reports / run_id
        run_dir.mkdir(); report_dir.mkdir()
        receipt = {"run_id": run_id, "agent_mask": mask, "seed": seed, "deterministic_control": deterministic,
                   "frames": 3316, "sequences": 21, "weights_unchanged": True,
                   "sealed_cooperative_parity": True if mask == 3 else None,
                   "sequence_commits": {f"{n:04d}": "f" * 64 for n in range(21)}}
        for kind in ("predictions", "association", "control"):
            overwrite(run_dir / f"{kind}.jsonl", {"fixture": run_id, "kind": kind})
            receipt[f"{kind}_sha256"] = evaluation.sha(run_dir / f"{kind}.jsonl")
        overwrite(run_dir / "receipt.json", receipt)
        summary["runs"][run_id] = receipt
        metrics = metric_fixture(.1 + index / 10)
        for name, value in (("metrics.json", metrics), ("runtime.json", {}), ("golden-cases.json", {"passed": True})):
            overwrite(report_dir / name, value)
        selected_protocol = evaluation.protocol(fake_adapter)
        report = {"kind": "source_ablation_car_evaluation_v1", "status": "completed", "reporting_scope": "car_only",
                  "protocol": selected_protocol, "protocol_sha256": hashlib.sha256(evaluation.canonical(selected_protocol)).hexdigest(),
                  "sealed_protocol_sha256": hashlib.sha256(evaluation.canonical(fake_adapter.PROTOCOL)).hexdigest(),
                  "predictions_sha256": receipt["predictions_sha256"],
                  "ground_truth_manifest_sha256": evaluation.GT_MANIFEST_SHA256, "ground_truth_sha256": evaluation.GT_SHA256,
                  "no_model_selection": True, "no_test_payload": True, "no_validation_parameter_fitting": True,
                  "paper_eligible": False, "coverage": {"frames": 3316, "sequences": 21,
                      "final_sequence_commit_sha256": receipt["sequence_commits"]},
                  "input_sources": {"predictions": evaluation.evidence(run_dir / "predictions.jsonl"),
                      "sealed_adapter": evaluation.evidence(evaluation.ADAPTER_PATH), "evaluation_cli": evaluation.evidence(evaluation.__file__),
                      "ground_truth_manifest": {"sha256": evaluation.GT_MANIFEST_SHA256}, "ground_truth": {"sha256": evaluation.GT_SHA256}},
                  "files": {p.name: evaluation.evidence(p) for p in report_dir.iterdir()},
                  "primary_car": evaluation.car_metrics(metrics)}
        overwrite(report_dir / "report.json", report)
    overwrite(root / "summary.json", summary)
    return root, reports, tmp_path / "comparison"


def test_five_groups_descriptive_sd_and_paired_differences(comparison_inputs):
    root, reports, output = comparison_inputs
    result = evaluation.compare(root, reports, output)
    assert set(result["run_metrics"]) == set(evaluation.RUNS)
    assert result["cooperative_three_seed_descriptive"]["AMOTA"]["mean"] == pytest.approx(.4)
    assert result["cooperative_three_seed_descriptive"]["AMOTA"]["sample_sd"] == pytest.approx(.1)
    paired = result["cooperative_minus_single_source"]["cooperative-seed-1337"]["vehicle-only"]
    assert paired["metric_difference"]["HOTA"] == pytest.approx(.2)
    assert len(paired["paired_sequence_differences"]) == 21
    assert paired["paired_sequence_sign_counts"]["HOTA"] == {"positive": 21, "zero": 0, "negative": 0}
    assert result["metric_direction"]["AMOTP_m"] == "lower_is_better"
    assert result["source_manifest_sha256"] == evaluation.sha(output / "source-manifest.json")
    assert result["single_source_repetitions"] == 1
    assert result["paper_eligible"] is False
    with pytest.raises(ValueError, match="already exists"):
        evaluation.compare(root, reports, output)


@pytest.mark.parametrize("case", ["missing_run", "unsealed_cooperative", "stream", "artifact", "metric", "cohort", "source", "class"])
def test_comparison_fails_closed_on_mutation(comparison_inputs, case):
    root, reports, output = comparison_inputs
    summary_path = root / "summary.json"
    report_path = reports / "vehicle-only" / "report.json"
    if case in ("missing_run", "unsealed_cooperative"):
        summary = evaluation.read_json(summary_path)
        if case == "missing_run":
            summary["runs"].pop("cooperative-seed-3407")
        else:
            summary["all_cooperative_streams_match_sealed"] = False
        overwrite(summary_path, summary)
    elif case == "stream":
        overwrite(root / "vehicle-only" / "control.jsonl", {"changed": True})
    elif case == "artifact":
        overwrite(reports / "vehicle-only" / "metrics.json", {"changed": True})
    elif case in ("metric", "cohort", "source"):
        report = evaluation.read_json(report_path)
        if case == "metric":
            report["primary_car"]["nuscenes"]["amota"] = .99
        elif case == "cohort":
            report["coverage"]["frames"] = 3315
        else:
            report["input_sources"]["evaluation_cli"]["sha256"] = "0" * 64
        overwrite(report_path, report)
    else:
        metric_path = reports / "vehicle-only" / "metrics.json"
        metrics = evaluation.read_json(metric_path)
        metrics["trackeval"]["bicycle"] = {}
        overwrite(metric_path, metrics)
        report = evaluation.read_json(report_path)
        report["files"]["metrics.json"] = evaluation.evidence(metric_path)
        overwrite(report_path, report)
    with pytest.raises(ValueError):
        evaluation.compare(root, reports, output)
    assert not output.exists()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_missing_metrics_are_not_silently_zero(value):
    primary = evaluation.car_metrics(metric_fixture())
    primary["nuscenes"]["amota"] = value
    with pytest.raises(ValueError, match="non-finite"):
        evaluation.metric_vector(primary)
