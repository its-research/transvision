from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.resilient_v2x import (
    clearml_single_seed_sequential_candidates as sequential,
)
from tools.resilient_v2x import sota_gate, sota_selector


def _runs(value: float) -> dict[tuple[int, str], dict[str, float]]:
    return {
        (delay, condition): {sota_gate.LEADERSHIP_METRIC: value}
        for delay in sota_gate.DELAYS_MS
        for condition in sota_gate.CONDITIONS
    }


def _formal_runs(*, full: float, other: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (delay, condition) in enumerate(
        (delay, condition)
        for delay in sota_gate.DELAYS_MS
        for condition in sota_gate.CONDITIONS
    ):
        value = full if (delay, condition) == (0, "Full") else other
        rows.append(
            {
                "condition_id": (
                    f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
                ),
                "delay_ms": delay,
                "condition": condition,
                "agent_scope": "E+R",
                "prediction_sha256": f"{index + 1:x}" * 64,
                "prediction_content_sha256": f"{index + 2:x}" * 64,
                "metrics": {metric: value for metric in sequential.AP_METRIC_KEYS},
            }
        )
    return rows


def _baseline_formal_runs() -> dict[str, list[dict[str, object]]]:
    return {
        subject: _formal_runs(full=60.0, other=50.0)
        for subject in sota_gate.BASELINE_SUBJECTS
    }


def _selection(
    *,
    evaluated_subjects: tuple[str, ...],
    passing_subject: str | None = None,
) -> dict[str, object]:
    baselines = {subject: _runs(50.0) for subject in sota_gate.BASELINE_SUBJECTS}
    candidates = {
        subject: _runs(55.0 if subject == passing_subject else 40.0)
        for subject in sota_gate.CANDIDATE_ORDER
        if subject in evaluated_subjects
    }
    evidence = {
        subject: copy.deepcopy(sota_gate.evidence_fingerprint())
        for subject in (*sota_gate.BASELINE_SUBJECTS, *candidates)
    }
    return sota_selector.build_selection(
        candidate_runs_by_subject=candidates,
        baseline_runs_by_subject=baselines,
        evidence_fingerprints_by_subject=evidence,
    )


def _initial_all_failed() -> dict[str, object]:
    return _selection(evaluated_subjects=sota_gate.CANDIDATE_ORDER)


def _receipt(
    candidate_id: str,
    *,
    passes: bool = False,
    candidate_full: float | None = None,
    candidate_other: float | None = None,
) -> dict[str, object]:
    spec_index = sequential.CANDIDATE_ORDER.index(candidate_id) + 1
    candidate_runs = _formal_runs(
        full=(59.5 if passes else 59.4) if candidate_full is None else candidate_full,
        other=51.0 if candidate_other is None else candidate_other,
    )
    return sequential.build_gate_receipt(
        candidate_id=candidate_id,
        source_dataset_id=f"{spec_index:x}" * 32,
        template_task_id=f"{spec_index + 1:x}" * 32,
        training_task_id=f"{spec_index + 2:x}" * 32,
        final_model_id=f"{spec_index + 3:x}" * 32,
        evaluation_task_id=f"{spec_index + 4:x}" * 32,
        final_checkpoint_sha256=f"{spec_index:x}" * 64,
        final_checkpoint_size_bytes=570_000_000 + spec_index,
        run_contract_sha256=f"{spec_index + 1:x}" * 64,
        manifest_sha256=f"{spec_index + 2:x}" * 64,
        sealed_source_tree_sha256=f"{spec_index + 3:x}" * 64,
        teacher_audit_sha256=f"{spec_index + 4:x}" * 64,
        final_checkpoint_contract_sha256=f"{spec_index + 5:x}" * 64,
        evaluation_run_contract_sha256=f"{spec_index + 6:x}" * 64,
        evaluation_plan_sha256=f"{spec_index + 7:x}" * 64,
        metrics_artifact_sha256=f"{spec_index + 8:x}" * 64,
        gate_artifact_sha256=f"{spec_index + 5:x}" * 64,
        prediction_evidence_sha256=f"{spec_index + 6:x}" * 64,
        prediction_evidence_archive_sha256=f"{spec_index + 7:x}" * 64,
        evaluator_evidence_sha256=f"{spec_index + 7:x}" * 64,
        candidate_runs=candidate_runs,
        baseline_runs_by_subject=_baseline_formal_runs(),
        baseline_evidence_sha256_by_subject={
            subject: f"{index + 1:x}" * 64
            for index, subject in enumerate(sota_gate.BASELINE_SUBJECTS)
        },
    )


def _chain(*receipts: dict[str, object]) -> dict[str, object]:
    return sequential.seal_chain_evidence(
        initial_selection=_initial_all_failed(),
        sequential_gate_receipts=receipts,
    )


def test_queue_waits_until_every_existing_trained_candidate_is_evaluated() -> None:
    initial = _selection(evaluated_subjects=("resilient_v2x",))
    chain = sequential.seal_chain_evidence(
        initial_selection=initial,
        sequential_gate_receipts=(),
    )
    plan = sequential.build_plan(chain)
    assert plan["status"] == "await_existing_candidate_evidence"
    assert plan["create_new_training"] is False
    assert plan["next_candidate_id"] is None
    assert plan["initial_formal_barrier"]["ready_to_unlock_p1"] is False
    assert [row["label"] for row in plan["initial_formal_barrier"]["rows"]] == [
        "E1",
        "E2",
        "E3",
        "P0",
        "P2",
    ]


def test_existing_eligible_pass_stops_fallback_training() -> None:
    initial = _selection(
        evaluated_subjects=sota_gate.CANDIDATE_ORDER,
        passing_subject="support_residual_linear",
    )
    plan = sequential.build_plan(
        sequential.seal_chain_evidence(
            initial_selection=initial,
            sequential_gate_receipts=(),
        )
    )
    assert plan["status"] == "stop_existing_candidate_passed"
    assert plan["stop_create_new_training"] is True
    assert plan["create_new_training"] is False


def test_initial_formal_selector_gate_rows_are_recomputed_fail_closed() -> None:
    initial = _initial_all_failed()
    row = next(
        item
        for item in initial["candidate_results"]
        if item["subject"] == "support_residual_linear"
    )
    row["aggregate_comparisons"]["mean_12"]["margin"] = 1.0
    initial = sota_selector._sealed(initial)
    chain = sequential.seal_chain_evidence(
        initial_selection=initial,
        sequential_gate_receipts=(),
    )
    with pytest.raises(ValueError, match="mean_12 is inconsistent"):
        sequential.build_plan(chain)


def test_failed_receipts_unlock_exactly_one_successor_in_fixed_order() -> None:
    plan = sequential.build_plan(_chain())
    assert plan["next_candidate_id"] == "P1"
    assert plan["already_running_training_unaffected"] == ["E2", "E3", "P0", "P2"]
    assert plan["initial_formal_barrier"]["ready_to_unlock_p1"] is True
    assert plan["queue_mapping"] == {
        "training": {
            "name": "GPU4-A100",
            "id": sequential.QUEUE_IDS["GPU4-A100"],
        },
        "formal_evaluation": {
            "name": "GPU4-A100",
            "id": sequential.QUEUE_IDS["GPU4-A100"],
        },
    }
    assert plan["training_protocol"]["training_dataset_id"] == (
        sequential.TRAINING_DATASET_ID
    )
    assert plan["training_protocol"]["teacher_checkpoint_sha256"] == (
        sequential.TEACHER_CHECKPOINT_SHA256
    )

    receipts: list[dict[str, object]] = []
    for expected_current, expected_next in zip(
        sequential.CANDIDATE_ORDER,
        (*sequential.CANDIDATE_ORDER[1:], None),
        strict=True,
    ):
        receipts.append(_receipt(expected_current))
        plan = sequential.build_plan(_chain(*receipts))
        assert plan["next_candidate_id"] == expected_next
    assert plan["status"] == "predefined_candidates_exhausted_without_lead"
    assert plan["stop_create_new_training"] is True


def test_first_sequential_pass_stops_and_rejects_any_later_receipt() -> None:
    p1_pass = _receipt("P1", passes=True)
    plan = sequential.build_plan(_chain(p1_pass))
    assert plan["status"] == "stop_sequential_candidate_passed"
    assert plan["selected_candidate_id"] == "P1"
    assert plan["create_new_training"] is False

    with pytest.raises(ValueError, match="after the first eligible pass"):
        sequential.build_plan(_chain(p1_pass, _receipt("P3")))


def test_gate_boundaries_and_boolean_are_recomputed_fail_closed() -> None:
    assert _receipt("P1", passes=True)["gate"]["gate_passed"] is True
    assert _receipt("P1")["gate"]["gate_passed"] is False
    mean_boundary = _receipt(
        "P1",
        candidate_full=59.5,
        candidate_other=(610.0 - 59.5) / 11.0,
    )
    assert mean_boundary["gate"]["mean_12_margin"] == 0.0
    assert mean_boundary["gate"]["gate_passed"] is False
    receipt = _receipt("P1")
    receipt["gate"]["gate_passed"] = True
    receipt = sequential._sealed(receipt)
    with pytest.raises(ValueError, match="gate evidence is inconsistent"):
        sequential.build_plan(_chain(receipt))


def test_training_and_evaluation_must_both_be_completed_on_exact_queues() -> None:
    for section in ("producer", "evaluation"):
        receipt = _receipt("P1")
        receipt[section]["status"] = "in_progress"
        receipt = sequential._sealed(receipt)
        with pytest.raises(ValueError, match="lifecycle drifted"):
            sequential.build_plan(_chain(receipt))

    receipt = _receipt("P1")
    receipt["evaluation"]["queue_id"] = sequential.QUEUE_IDS["GPU4-V100"]
    receipt = sequential._sealed(receipt)
    with pytest.raises(ValueError, match="lifecycle drifted"):
        sequential.build_plan(_chain(receipt))


def test_formal_run_inventory_metrics_and_baseline_artifacts_are_fail_closed() -> None:
    receipt = _receipt("P1")
    receipt["evaluation"]["runs"].pop()
    receipt = sequential._sealed(receipt)
    with pytest.raises(ValueError, match="exactly 12 formal runs"):
        sequential.build_plan(_chain(receipt))

    receipt = _receipt("P1")
    del receipt["evaluation"]["runs"][0]["metrics"][sequential.AP_METRIC_KEYS[0]]
    receipt = sequential._sealed(receipt)
    with pytest.raises(ValueError, match="metrics keys mismatch"):
        sequential.build_plan(_chain(receipt))

    receipt = _receipt("P1")
    receipt["baseline_reference"]["formal_evidence_sha256_by_subject"].pop("ffnet")
    receipt = sequential._sealed(receipt)
    with pytest.raises(ValueError, match="baseline hashes drifted"):
        sequential.build_plan(_chain(receipt))


def test_sealed_receipt_survives_sorted_json_round_trip() -> None:
    serialized = json.dumps(_chain(_receipt("P1")), sort_keys=True)
    restored = json.loads(serialized)
    plan = sequential.build_plan(restored)
    assert plan["next_candidate_id"] == "P3"
    assert plan["completed_sequential_gate_count"] == 1


def test_candidate_registry_seals_config_and_dependency_semantics() -> None:
    sequential._validate_candidate_registry()
    assert [spec.parent_candidate_id for spec in sequential.CANDIDATE_SPECS] == [
        "P0",
        "P1",
        "P3",
        "P4",
        "P5",
        "P6",
    ]
    assert sequential.CANDIDATE_BY_ID["P4"].config_base.endswith("support_residual.py")
    assert sequential.CANDIDATE_BY_ID["P6"].config_base.endswith(
        "reliability_gated_residual_full_nonlinear.py"
    )


def test_p1_profile_wraps_zero_shot_guards_in_exact_serial_contract() -> None:
    _shared, profile = sequential._deployment_profile("P1", _chain())
    contract = profile.execution_contract
    assert contract["contract_type"] == "resilient_v2x_single_seed_p1_training_v2"
    assert contract["training_protocol"] == sequential.TRAINING_PROTOCOL
    assert contract["queue_mapping"]["training"] == {
        "name": "GPU4-A100",
        "id": sequential.QUEUE_IDS["GPU4-A100"],
    }
    assert contract["formal_evaluation_required_before_successor"] is True
    assert contract["successor_creation_requires_gate_passed_false"] is True
    assert contract["zero_shot_evidence_forbidden"] is True
    legacy = contract["p1_dual_candidate_execution_contract"]
    assert (
        legacy["candidate_variants"]["p1_zero_shot_p0_epoch50_final"]["mode"]
        == "inference_only"
    )


def test_out_of_order_or_zero_shot_or_protocol_drift_is_terminal() -> None:
    with pytest.raises(ValueError, match="P1 is not exact trained evidence"):
        sequential.build_plan(_chain(_receipt("P3")))

    zero_shot = _receipt("P1")
    zero_shot["candidate_subject"] = sota_gate.P1_ZERO_SHOT_SUBJECT
    zero_shot["weights_retrained"] = False
    zero_shot["evidence_class"] = "inference_only_checkpoint_reuse"
    zero_shot = sequential._sealed(zero_shot)
    with pytest.raises(ValueError, match="not exact trained evidence"):
        sequential.build_plan(_chain(zero_shot))

    drifted = _receipt("P1")
    drifted["protocol_evidence_fingerprint"]["sample_count"] = 1336
    drifted = sequential._sealed(drifted)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        sequential.build_plan(_chain(drifted))


def test_dry_run_is_default_and_remote_write_needs_all_explicit_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chain = _chain()
    evidence_path = tmp_path / "chain.json"
    evidence_path.write_text(json.dumps(chain), encoding="utf-8")
    calls: list[str] = []

    def fake_execute(**kwargs: object) -> int:
        calls.append(str(kwargs["phase"]))
        return 0

    monkeypatch.setattr(sequential, "_execute_phase", fake_execute)
    spec = sequential.CANDIDATE_SPECS[0]
    assert (
        sequential.main(
            [
                "upload-source",
                "--output-dir",
                str(spec.output_dir),
                "--execute-token",
                spec.upload_token,
            ]
        )
        == 0
    )
    no_evidence = json.loads(capsys.readouterr().out)
    assert no_evidence["status"] == "sealed_sequential_evidence_required"
    assert no_evidence["remote_state_changed"] is False
    assert calls == []

    assert sequential.main(["--evidence", str(evidence_path), "--phase", "launch"]) == 0
    assert calls == []
    assert json.loads(capsys.readouterr().out)["next_candidate_id"] == "P1"

    plan = sequential.build_plan(chain)
    common = [
        "--evidence",
        str(evidence_path),
        "--phase",
        "launch",
        "--execute",
        "--plan-seal-sha256",
        str(plan["seal_sha256"]),
    ]
    with pytest.raises(PermissionError, match="execution token"):
        sequential.main(common)
    with pytest.raises(PermissionError, match="remote-write token"):
        sequential.main([*common, "--execute-token", sequential.EXECUTE_TOKEN])

    assert (
        sequential.main(
            [
                *common,
                "--execute-token",
                sequential.EXECUTE_TOKEN,
                "--remote-write-token",
                spec.launch_token,
                "--source-dataset-id",
                "a" * 32,
                "--template-task-id",
                "b" * 32,
            ]
        )
        == 0
    )
    assert calls == ["launch"]
