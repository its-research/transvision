from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.resilient_v2x import build_post_winner_experiment_plan as plan


ROOT = Path(__file__).resolve().parents[2]
TRAINING_INDEX = ROOT / "artifacts/resilient_v2x/dair_v2/training_overlays.json"
EVALUATION_INDEX = ROOT / "artifacts/resilient_v2x/dair_v2/evaluation_overlays.json"
TEMPORAL_MANIFEST = ROOT / "artifacts/resilient_v2x/dair/temporal_manifest_v2.json"
VALIDATION_COHORT = ROOT / "artifacts/resilient_v2x/dair_v2/validation_cohort.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(config_path: str, checkpoint: Path) -> dict[str, object]:
    config = ROOT / config_path
    binding = {
        "config_path": config_path,
        "config_sha256": _sha256_file(config),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "training_task_id": "1" * 32,
        "evaluation_task_id": "2" * 32,
        "model_id": "3" * 32,
        "training_dataset_id": "4" * 32,
        "teacher_task_id": "5" * 32,
        "teacher_model_id": "6" * 32,
        "teacher_checkpoint_sha256": "7" * 64,
    }
    return plan._seal(
        {
            "schema_version": 2,
            "document_type": plan.SELECTED_IDENTITY_TYPE,
            "status": "sealed",
            "selected_subject": Path(config_path).stem,
            "selected_candidate_label": "fixture",
            "training_seed": plan.TRAINING_SEED,
            "checkpoint_policy": plan.CHECKPOINT_POLICY,
            "method_binding": binding,
            "gate_result": {"gate_passed": True},
        }
    )


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_path: str = "configs/resilient_v2x/dair_resilient_v2x.py",
) -> dict[str, object]:
    checkpoint = tmp_path / "epoch_50.pth"
    checkpoint.write_bytes(b"controlled-final-checkpoint")
    monkeypatch.setattr(plan, "_checkpoint_archive", lambda _subject: None)
    return plan.build_plan(
        selected_identity=_identity(config_path, checkpoint),
        checkpoint=checkpoint,
        training_index_path=TRAINING_INDEX,
        evaluation_index_path=EVALUATION_INDEX,
        temporal_manifest_path=TEMPORAL_MANIFEST,
        validation_cohort_path=VALIDATION_COHORT,
        paper_root=tmp_path / "paper-not-present",
    )


def test_main_winner_plan_is_sealed_complete_and_dependency_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _build(tmp_path, monkeypatch)

    plan.validate_plan(document)
    assert document["document_type"] == plan.DOCUMENT_TYPE
    assert document["mode"] == "local_plan_only_no_remote_writes"
    assert document["training_contract"] == plan.TRAINING_CONTRACT
    assert document["job_count"] == 49
    assert document["paper_interface_audit"]["status"] == "missing"

    jobs = document["jobs"]
    ids = [job["job_id"] for job in jobs]
    positions = {job_id: index for index, job_id in enumerate(ids)}
    for job in jobs:
        assert all(positions[parent] < positions[job["job_id"]] for parent in job["depends_on"])

    ablations = document["table_v_ablation_resolution"]
    assert [row["id"] for row in ablations] == [
        "no_ptf",
        "alternative_ptf",
        "router_static",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "no_distillation",
    ]
    assert all(
        row["reuse_audit"]["status"]
        == "reuse_candidate_pending_checkpoint_provenance"
        for row in ablations
    )


def test_no_reliability_winner_does_not_schedule_a_noop_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _build(
        tmp_path,
        monkeypatch,
        config_path="configs/resilient_v2x/improvements/no_reliability_linear.py",
    )
    ablations = {row["id"]: row for row in document["table_v_ablation_resolution"]}

    assert (
        ablations["no_reliability"]["reuse_audit"]["status"]
        == "not_applicable_winner_already_has_setting"
    )
    ids = {job["job_id"] for job in document["jobs"]}
    assert "table_v.no_reliability.resolve" in ids
    assert "table_v.no_reliability.train_or_reuse" not in ids
    assert "table_v.no_reliability.evaluate" not in ids
    assert (
        ablations["no_ptf"]["reuse_audit"]["status"]
        == "winner_derived_training_required"
    )
    actions = document["paper_interface_audit"][
        "winner_specific_table_v_actions"
    ]
    assert {
        (row["ablation_id"], row["action"]) for row in actions
    } >= {
        ("no_reliability", "remove_or_replace_noop_row"),
        ("alternative_ptf", "relabel_reference_and_alternative_ptf_rows"),
    }


def test_support_residual_winner_adds_exact_module_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _build(
        tmp_path,
        monkeypatch,
        config_path="configs/resilient_v2x/improvements/reliability_gated_residual_full_nonlinear.py",
    )
    removal = next(
        row
        for row in document["table_v_ablation_resolution"]
        if row["id"] == "winner_residual_module_removal"
    )
    assert removal["overrides"] == {
        "model.support_residual_weight": 0.0,
        "model.support_residual_reliability_gate": False,
    }
    assert removal["reuse_audit"]["status"] == "winner_derived_training_required"
    actions = document["paper_interface_audit"][
        "winner_specific_table_v_actions"
    ]
    addition = next(
        row
        for row in actions
        if row["ablation_id"] == "winner_residual_module_removal"
    )
    assert addition["action"] == "add_winner_module_removal_row_and_three_result_ids"
    assert len(addition["suggested_result_ids"]) == 3


def test_duration_and_agent_scope_jobs_never_enter_the_main_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _build(tmp_path, monkeypatch)
    jobs = {job["job_id"]: job for job in document["jobs"]}

    for duration in (1, 2, 3):
        for modality in ("lidar", "camera"):
            payload = jobs[
                f"table_vii.duration.q{duration}.{modality}.evaluate"
            ]["payload"]
            assert payload["agent_scope"] == "E+R"
            assert payload["duration_ticks"] == duration
            assert payload["paper_binding_status"] == "missing_modality_specific_result_id"
    boundary = jobs["table_vii.duration.q4_boundary"]["payload"]
    assert boundary["status"] == "unsupported"
    assert boundary["ap_result_id"] is None

    for scope in ("e_only", "r_only"):
        payload = jobs[f"diagnostic.{scope}.evaluate"]["payload"]
        assert payload["run_count"] == 8
        assert payload["main_table_membership"] is False
        assert payload["required_distinct_result_ids"] is True


def test_complexity_pair_requires_exact_capacity_and_excludes_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _build(tmp_path, monkeypatch)
    jobs = {job["job_id"]: job for job in document["jobs"]}

    concat = jobs["table_vii.complexity.concat_config"]["payload"]
    assert concat["overrides"] == {"model.routing_mode": "concat"}
    assert concat["must_retain_all_winner_parameters"] is True
    for subject in ("winner", "concat"):
        profile = jobs[f"table_vii.complexity.{subject}.profile"]["payload"]
        assert profile["exclude_module_paths"] == ["teacher"]
        assert profile["teacher_must_be_detached_before_cuda"] is True
        assert profile["flops_must_be_measured"] is True
    pair = jobs["table_vii.complexity.validate_pair"]["payload"]
    assert pair["parameter_count_must_match_exactly"] is True
    assert pair["teacher_cost_must_be_zero_for_both"] is True


def test_tampered_identity_and_checkpoint_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "epoch_50.pth"
    checkpoint.write_bytes(b"controlled-final-checkpoint")
    identity = _identity(
        "configs/resilient_v2x/dair_resilient_v2x.py", checkpoint
    )
    tampered = copy.deepcopy(identity)
    tampered["selected_candidate_label"] = "tampered"
    monkeypatch.setattr(plan, "_checkpoint_archive", lambda _subject: None)

    with pytest.raises(plan.PostWinnerPlanError, match="seal mismatch"):
        plan.build_plan(
            selected_identity=tampered,
            checkpoint=checkpoint,
            training_index_path=TRAINING_INDEX,
            evaluation_index_path=EVALUATION_INDEX,
            temporal_manifest_path=TEMPORAL_MANIFEST,
            validation_cohort_path=VALIDATION_COHORT,
            paper_root=tmp_path / "paper",
        )

    checkpoint.write_bytes(b"different-checkpoint")
    with pytest.raises(plan.PostWinnerPlanError, match="size differs|hash differs"):
        plan.build_plan(
            selected_identity=identity,
            checkpoint=checkpoint,
            training_index_path=TRAINING_INDEX,
            evaluation_index_path=EVALUATION_INDEX,
            temporal_manifest_path=TEMPORAL_MANIFEST,
            validation_cohort_path=VALIDATION_COHORT,
            paper_root=tmp_path / "paper",
        )


def test_tampered_plan_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _build(tmp_path, monkeypatch)
    document["jobs"][0]["payload"]["binding"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(plan.PostWinnerPlanError, match="seal mismatch"):
        plan.validate_plan(document)


def test_real_protocol_inputs_are_the_fixed_formal_contract() -> None:
    training = plan._validate_training_index(json.loads(TRAINING_INDEX.read_text()))
    evaluation = plan._validate_evaluation_index(json.loads(EVALUATION_INDEX.read_text()))

    assert training["p_lidar"] == training["p_camera"] == 0.2
    assert training["temporal_manifest_sha256"] == evaluation["temporal_manifest_sha256"]
    assert evaluation["sample_count"] == 1337
    assert evaluation["sample_ids_sha256"] == plan.SAMPLE_IDS_SHA256


def test_legacy_duration_overlays_are_not_protocol_compatible() -> None:
    current = json.loads(EVALUATION_INDEX.read_text())
    legacy = json.loads(
        (ROOT / "artifacts/resilient_v2x/dair/continuous_d2/evaluation_overlays.json").read_text()
    )

    assert legacy["temporal_manifest_sha256"] != current["temporal_manifest_sha256"]
    assert legacy["sample_ids_sha256"] != current["sample_ids_sha256"]
