from __future__ import annotations

import copy

import pytest

from tools.resilient_v2x import build_final_single_seed_selection as selector
from tools.resilient_v2x import sota_selector


def _runs(value: float) -> list[dict[str, object]]:
    result = []
    for delay in selector.sota_gate.DELAYS_MS:
        for condition in selector.sota_gate.CONDITIONS:
            result.append(
                {
                    "condition_id": (
                        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
                    ),
                    "delay_ms": delay,
                    "condition": condition,
                    "agent_scope": "E+R",
                    "metrics": {
                        selector.AP_METRIC_KEYS[0]: value + 5.0,
                        selector.AP_METRIC_KEYS[1]: value,
                        selector.AP_METRIC_KEYS[2]: value - 2.0,
                        selector.AP_METRIC_KEYS[3]: value - 5.0,
                    },
                }
            )
    return result


def _formal_inputs(
    *, main_value: float = 51.0, baseline_value: float = 50.0
) -> dict[str, object]:
    entries = []
    for index, subject in enumerate(selector.EXPECTED_FORMAL_SUBJECTS, start=1):
        value = (
            baseline_value
            if subject in selector.sota_gate.BASELINE_SUBJECTS
            else main_value
        )
        entries.append(
            {
                "subject": subject,
                "identity_binding": {
                    "label": subject,
                    "training_task_id": f"{index:032x}",
                    "evaluation_task_id": f"{index + 100:032x}",
                    "model_id": f"{index + 200:032x}",
                    "model_name": f"ResilientV2X {subject} final checkpoint",
                    "checkpoint_sha256": f"{index:064x}",
                    "config_path": "configs/resilient_v2x/dair_resilient_v2x.py",
                    "config_sha256": "a" * 64,
                    "source_dataset_id": "b" * 32,
                    "source_revision_sha256": "b" * 64,
                    "source_archive_name": "fixture-source.tar.zst",
                    "source_archive_bytes": 10_000 + index,
                    "source_archive_sha256": "c" * 64,
                    "teacher_task_id": selector.candidate_queue.TEACHER_TASK_ID,
                    "teacher_model_id": selector.candidate_queue.TEACHER_MODEL_ID,
                    "teacher_checkpoint_sha256": (
                        selector.candidate_queue.TEACHER_CHECKPOINT_SHA256
                    ),
                    "training_dataset_id": (
                        selector.candidate_queue.TRAINING_DATASET_ID
                    ),
                    "checkpoint_filename": "epoch_50.pth",
                    "checkpoint_size_bytes": 2_000 + index,
                    "training_script_sha256": "d" * 64,
                    "evaluation_script_sha256": "e" * 64,
                    "run_contract_artifact_sha256": "1" * 64,
                    "initialization_audit_artifact_sha256": "2" * 64,
                    "final_checkpoint_contract_artifact_sha256": "3" * 64,
                    "evaluation_run_contract_artifact_sha256": "8" * 64,
                    "evaluation_plan_artifact_sha256": "4" * 64,
                    "metrics_artifact_sha256": "5" * 64,
                    "prediction_evidence_artifact_sha256": "6" * 64,
                    "prediction_evidence_archive_sha256": "7" * 64,
                },
                "runs": _runs(value),
            }
        )
    return selector._seal(
        {
            "schema_version": 2,
            "document_type": selector.FORMAL_INPUT_DOCUMENT_TYPE,
            "protocol_evidence_fingerprint": (
                selector.sota_gate.evidence_fingerprint()
            ),
            "subject_order": list(selector.EXPECTED_FORMAL_SUBJECTS),
            "training_seed": selector.candidate_queue.TRAINING_SEED,
            "checkpoint_policy": selector.candidate_queue.CHECKPOINT_POLICY,
            "authority_bindings": {
                "producer_task_id": f"{900:032x}",
                "producer_script_sha256": "8" * 64,
                "producer_artifact_name": "final_selector_formal_inputs",
                "training_controller_task_id": f"{901:032x}",
                "training_provenance_task_id": f"{902:032x}",
                "watcher_task_id": f"{903:032x}",
                "watcher_artifact_name": "formal_1337_evaluation_plan",
                "watcher_artifact_sha256": "f" * 64,
                "watcher_plan_seal_sha256": "9" * 64,
                "watcher_plan_content_sha256": "a" * 64,
                "leaderboard_task_id": f"{904:032x}",
                "leaderboard_artifact_name": "formal_1337_leaderboard",
                "leaderboard_artifact_sha256": "1" * 64,
                "leaderboard_seal_sha256": "b" * 64,
                "leaderboard_content_sha256": "c" * 64,
                "audit_task_id": f"{905:032x}",
                "audit_artifact_name": "formal_1337_comparability_audit",
                "audit_artifact_sha256": "2" * 64,
                "audit_seal_sha256": "d" * 64,
                "audit_content_sha256": "e" * 64,
            },
            "entries": entries,
        }
    )


def _candidate_manifest(
    values: dict[str, float] | None = None,
) -> dict[str, object]:
    entries = []
    if values is None:
        values = {
            "support_residual_linear": 52.0,
            "no_reliability_linear": 52.0,
            "support_residual_no_reliability": 49.0,
            "support_residual_no_reliability_linear": 49.0,
            "support_residual_no_reliability_linear_bbox25": 49.0,
        }
    for index, spec in enumerate(selector.candidate_queue.CANDIDATES, start=1):
        entries.append(
            {
                "index": index,
                "label": spec.label,
                "subject": spec.subject,
                "training_task_id": spec.training_task_id,
                "template_task_id": spec.template_task_id,
                "training_status": "completed",
                "evaluation_task_id": f"{index + 500:032x}",
                "evaluation_status": "completed",
                "evidence_status": "validated",
                "training_final": {
                    "model_id": f"{index + 600:032x}",
                    "checkpoint_sha256": f"{index + 700:064x}",
                    "checkpoint_size_bytes": 1_000 + index,
                    "run_contract_artifact_sha256": "1" * 64,
                    "initialization_audit_artifact_sha256": "2" * 64,
                    "final_checkpoint_contract_artifact_sha256": "3" * 64,
                },
                "formal_evidence": {
                    "metrics_artifact_sha256": "4" * 64,
                    "evaluation_plan_artifact_sha256": "5" * 64,
                    "prediction_evidence_artifact_sha256": "6" * 64,
                    "prediction_evidence_archive_sha256": "7" * 64,
                    "runs": _runs(values[spec.subject]),
                },
            }
        )
    return selector._seal(
        {
            "schema_version": 1,
            "document_type": (
                "resilient_v2x_formal_1337_candidate_evaluation_manifest"
            ),
            "controller_task_id": "f" * 32,
            "protocol_id": selector.sota_gate.PROTOCOL_ID,
            "training_seed": selector.candidate_queue.TRAINING_SEED,
            "training_dataset_id": selector.candidate_queue.TRAINING_DATASET_ID,
            "checkpoint_policy": selector.candidate_queue.CHECKPOINT_POLICY,
            "sample_count": selector.sota_gate.SAMPLE_COUNT,
            "ground_truth_count": selector.sota_gate.GROUND_TRUTH_COUNT,
            "unsupported_sample_count": selector.sota_gate.UNSUPPORTED_SAMPLE_COUNT,
            "sample_ids_sha256": selector.sota_gate.SAMPLE_IDS_SHA256,
            "delays_ms": list(selector.sota_gate.DELAYS_MS),
            "conditions": list(selector.sota_gate.CONDITIONS),
            "candidate_order": list(selector.CANDIDATE_ONLY_ORDER),
            "candidate_count": len(selector.CANDIDATE_ONLY_ORDER),
            "validated_candidate_count": len(selector.CANDIDATE_ONLY_ORDER),
            "all_candidates_validated": True,
            "entries": entries,
        }
    )


def _gate_runs(value: float) -> dict[tuple[int, str], dict[str, float]]:
    return {
        (delay, condition): {selector.sota_gate.LEADERSHIP_METRIC: value}
        for delay in selector.sota_gate.DELAYS_MS
        for condition in selector.sota_gate.CONDITIONS
    }


def _initial_selection(value: float = 49.0) -> dict[str, object]:
    candidates = {
        subject: _gate_runs(value)
        for subject in selector.INITIAL_TRAINED_CANDIDATE_ORDER
    }
    baselines = {
        subject: _gate_runs(50.0) for subject in selector.sota_gate.BASELINE_SUBJECTS
    }
    evidence = {
        subject: copy.deepcopy(selector.sota_gate.evidence_fingerprint())
        for subject in (*selector.sota_gate.BASELINE_SUBJECTS, *candidates)
    }
    return sota_selector.build_selection(
        candidate_runs_by_subject=candidates,
        baseline_runs_by_subject=baselines,
        evidence_fingerprints_by_subject=evidence,
    )


def _receipt_runs(value: float) -> list[dict[str, object]]:
    rows = _runs(value)
    for index, row in enumerate(rows, start=1):
        row["prediction_sha256"] = f"{index:064x}"
        row["prediction_content_sha256"] = f"{index + 100:064x}"
    return rows


def _receipt(
    candidate_id: str, value: float, *, baseline_value: float = 50.0
) -> dict[str, object]:
    index = selector.sequential.CANDIDATE_ORDER.index(candidate_id) + 1
    return selector.sequential.build_gate_receipt(
        candidate_id=candidate_id,
        source_dataset_id=f"{index:032x}",
        template_task_id=f"{index + 10:032x}",
        training_task_id=f"{index + 20:032x}",
        final_model_id=f"{index + 30:032x}",
        evaluation_task_id=f"{index + 40:032x}",
        final_checkpoint_sha256=f"{index:064x}",
        final_checkpoint_size_bytes=500_000_000 + index,
        run_contract_sha256=f"{index + 1:064x}",
        manifest_sha256=f"{index + 2:064x}",
        sealed_source_tree_sha256=f"{index + 3:064x}",
        teacher_audit_sha256=f"{index + 4:064x}",
        final_checkpoint_contract_sha256=f"{index + 5:064x}",
        evaluation_run_contract_sha256=f"{index + 6:064x}",
        evaluation_plan_sha256=f"{index + 7:064x}",
        metrics_artifact_sha256=f"{index + 8:064x}",
        gate_artifact_sha256=f"{index + 9:064x}",
        prediction_evidence_sha256=f"{index + 10:064x}",
        prediction_evidence_archive_sha256=f"{index + 11:064x}",
        evaluator_evidence_sha256=f"{index + 12:064x}",
        candidate_runs=_receipt_runs(value),
        baseline_runs_by_subject={
            subject: _receipt_runs(baseline_value)
            for subject in selector.sota_gate.BASELINE_SUBJECTS
        },
        baseline_evidence_sha256_by_subject={
            subject: f"{baseline_index + 20:064x}"
            for baseline_index, subject in enumerate(
                selector.sota_gate.BASELINE_SUBJECTS
            )
        },
    )


def _all_failed_manifest() -> dict[str, object]:
    return _candidate_manifest(
        {subject: 49.0 for subject in selector.CANDIDATE_ONLY_ORDER}
    )


def test_all_six_are_ranked_and_fixed_order_breaks_equal_margin() -> None:
    selection, identity = selector.build_final_selection(
        formal_inputs=_formal_inputs(),
        candidate_manifest=_candidate_manifest(),
    )

    assert selection["candidate_subjects"] == list(selector.FINAL_CANDIDATE_ORDER)
    assert selection["selected_candidate"] == "support_residual_linear"
    assert selection["performance_ranked_candidates"][:3] == [
        "support_residual_linear",
        "no_reliability_linear",
        "resilient_v2x",
    ]
    assert identity["status"] == "sealed"
    assert identity["selected_subject"] == "support_residual_linear"
    assert identity["selection_seal_sha256"] == selection["seal_sha256"]
    assert identity["method_binding"]["label"] == "E1"
    assert identity["method_binding"]["config_sha256"] == (
        selector.candidate_queue.CANDIDATES[0].config_sha256
    )
    assert selector._require_seal(selection, context="selection")
    assert selector._require_seal(identity, context="identity")


def test_candidate_protocol_drift_fails_closed() -> None:
    manifest = copy.deepcopy(_candidate_manifest())
    manifest["sample_count"] = 1_336
    manifest = selector._seal(manifest)

    with pytest.raises(selector.FinalSelectionError, match="sample_count drifted"):
        selector.build_final_selection(
            formal_inputs=_formal_inputs(), candidate_manifest=manifest
        )


def test_missing_candidate_entry_fails_closed() -> None:
    manifest = copy.deepcopy(_candidate_manifest())
    manifest["entries"].pop()
    manifest = selector._seal(manifest)

    with pytest.raises(selector.FinalSelectionError, match="entries are incomplete"):
        selector.build_final_selection(
            formal_inputs=_formal_inputs(), candidate_manifest=manifest
        )


def test_all_initial_failures_wait_for_sequential_evidence() -> None:
    selection, identity = selector.build_final_selection(
        formal_inputs=_formal_inputs(main_value=49.0),
        candidate_manifest=_all_failed_manifest(),
    )

    assert selection["status"] == "awaiting_sequential_evidence"
    assert selection["selection_is_final"] is False
    assert selection["selected_candidate"] is None
    assert selection["evaluated_candidate_count"] == 6
    assert selection["pending_candidate_subjects"] == list(
        selector.SEQUENTIAL_CANDIDATE_ORDER
    )
    assert identity["status"] == "no_selected_method"
    assert identity["method_binding"] is None


def test_trained_p1_receipt_can_be_selected_and_seals_identity() -> None:
    receipt = _receipt("P1", 51.0)
    chain = selector.sequential.seal_chain_evidence(
        initial_selection=_initial_selection(),
        sequential_gate_receipts=(receipt,),
    )

    selection, identity = selector.build_final_selection(
        formal_inputs=_formal_inputs(main_value=49.0),
        candidate_manifest=_all_failed_manifest(),
        sequential_evidence=chain,
    )

    assert selection["selected_candidate"] == "reliability_gated_residual"
    assert selection["pending_candidate_subjects"] == []
    assert selection["not_created_after_stop_subjects"] == list(
        selector.SEQUENTIAL_CANDIDATE_ORDER[1:]
    )
    rows = {row["subject"]: row for row in selection["candidate_results"]}
    assert rows[selector.SEQUENTIAL_CANDIDATE_ORDER[1]]["evidence_status"] == (
        "not_created_after_earlier_pass"
    )
    assert selection["selection_is_final"] is True
    assert identity["status"] == "sealed"
    assert identity["selected_candidate_label"] == "P1"
    assert identity["method_binding"]["config_sha256"] == (
        selector.sequential.CANDIDATE_BY_ID["P1"].config_sha256
    )
    assert identity["method_binding"]["training_dataset_id"] == (
        selector.candidate_queue.TRAINING_DATASET_ID
    )
    assert (
        identity["method_binding"]["sequential_gate_receipt_seal_sha256"]
        == (receipt["seal_sha256"])
    )
    assert (
        identity["gate_artifact"]["selection_seal_sha256"] == (selection["seal_sha256"])
    )
    assert identity["gate_result_content_sha256"] == selector._content_sha256(
        identity["gate_result"]
    )


def test_sequential_receipt_baselines_must_equal_formal_raw_runs() -> None:
    receipt = _receipt("P1", 51.0, baseline_value=50.1)
    chain = selector.sequential.seal_chain_evidence(
        initial_selection=_initial_selection(),
        sequential_gate_receipts=(receipt,),
    )

    with pytest.raises(
        selector.FinalSelectionError,
        match="baseline ffnet differs from formal raw runs",
    ):
        selector.build_final_selection(
            formal_inputs=_formal_inputs(main_value=49.0),
            candidate_manifest=_all_failed_manifest(),
            sequential_evidence=chain,
        )


def test_selected_method_must_match_fixed_teacher_contract() -> None:
    formal_inputs = _formal_inputs(main_value=52.0)
    formal_inputs["entries"][-1]["identity_binding"]["teacher_model_id"] = "f" * 32
    formal_inputs = selector._seal(formal_inputs)

    with pytest.raises(
        selector.FinalSelectionError,
        match="teacher_model_id differs from the fixed contract",
    ):
        selector.build_final_selection(
            formal_inputs=formal_inputs,
            candidate_manifest=_all_failed_manifest(),
        )


def test_raw_runs_require_explicit_e_plus_r_scope() -> None:
    formal_inputs = _formal_inputs()
    del formal_inputs["entries"][0]["runs"][0]["agent_scope"]
    formal_inputs = selector._seal(formal_inputs)

    with pytest.raises(selector.FinalSelectionError, match="scope drifted"):
        selector.build_final_selection(
            formal_inputs=formal_inputs,
            candidate_manifest=_candidate_manifest(),
        )


def test_selected_config_path_must_not_traverse_repo() -> None:
    formal_inputs = _formal_inputs(main_value=52.0)
    formal_inputs["entries"][-1]["identity_binding"]["config_path"] = (
        "configs/../outside.py"
    )
    formal_inputs = selector._seal(formal_inputs)

    with pytest.raises(selector.FinalSelectionError, match="config path is invalid"):
        selector.build_final_selection(
            formal_inputs=formal_inputs,
            candidate_manifest=_all_failed_manifest(),
        )


def test_generic_trained_gate_accepts_exact_full_deficit_boundary() -> None:
    baselines = {
        subject: _gate_runs(50.0) for subject in selector.sota_gate.BASELINE_SUBJECTS
    }
    candidate = _gate_runs(51.0)
    candidate[(0, "Full")][selector.sota_gate.LEADERSHIP_METRIC] = 49.5

    result = selector._evaluate_trained_candidate(
        subject="reliability_gated_residual",
        candidate_runs=candidate,
        baseline_runs=baselines,
    )

    assert result["aggregate_comparisons"]["full_0ms"]["margin"] == -0.5
    assert result["aggregate_comparisons"]["full_0ms"]["passes_gate"] is True
    assert result["gate_passed"] is False  # worst remains below the baseline


def test_ranking_precedence_is_worst_then_mean_then_full_then_fixed() -> None:
    def row(
        subject: str,
        *,
        worst: float,
        mean: float,
        full: float,
        order: int,
    ) -> dict[str, object]:
        return {
            "subject": subject,
            "gate_passed": True,
            "ranking_values": {
                "worst_12_margin": worst,
                "mean_12_margin": mean,
                "full_0ms_margin": full,
                "fixed_candidate_order": order,
            },
        }

    ranked = sorted(
        (
            row("fixed_later", worst=1.0, mean=1.0, full=1.0, order=2),
            row("full_worse", worst=1.0, mean=1.0, full=0.0, order=1),
            row("mean_worse", worst=1.0, mean=0.0, full=100.0, order=0),
            row("worst_best", worst=2.0, mean=-100.0, full=-100.0, order=3),
            row("fixed_earlier", worst=1.0, mean=1.0, full=1.0, order=1),
        ),
        key=selector._ranking_sort_key,
    )

    assert [item["subject"] for item in ranked] == [
        "worst_best",
        "fixed_earlier",
        "fixed_later",
        "full_worse",
        "mean_worse",
    ]
