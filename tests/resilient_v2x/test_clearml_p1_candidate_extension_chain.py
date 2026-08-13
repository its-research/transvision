from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.resilient_v2x import (
    clearml_1337_candidate_extension_leaderboard as leaderboard,
)
from tools.resilient_v2x import clearml_formal_candidate_selector as formal
from tools.resilient_v2x import clearml_formal_candidate_selector_round2 as round2
from tools.resilient_v2x import clearml_p1_zero_shot_candidate_extension as extension


CANONICAL_LEADERBOARD_TASK_ID = "a" * 32
CANONICAL_SELECTOR_TASK_ID = "b" * 32
EXTENSION_LEADERBOARD_TASK_ID = "c" * 32
CANDIDATE_EVALUATION_TASK_ID = "d" * 32
CANONICAL_LEADERBOARD_SEAL = "1" * 64


def _runs(value: float) -> dict[tuple[int, str], dict[str, float]]:
    return {
        (delay, condition): {
            key: value - metric_index * 0.1
            for metric_index, key in enumerate(formal.AP_METRIC_KEYS)
        }
        for delay in formal.DELAYS_MS
        for condition in formal.CONDITIONS
    }


def _all_runs(
    *,
    primary: float = 51.0,
    zero_shot: float = 52.0,
) -> tuple[
    dict[str, dict[tuple[int, str], dict[str, float]]],
    dict[tuple[int, str], dict[str, float]],
]:
    values = {subject: _runs(50.0) for subject in formal.BASELINE_SUBJECTS}
    candidate_values = {
        "resilient_v2x": primary,
        "support_residual": primary - 0.1,
        "linear_no_distillation": primary - 0.2,
        "no_distillation_peak_lr_3e4": primary - 0.3,
    }
    values.update(
        {subject: _runs(value) for subject, value in candidate_values.items()}
    )
    return values, _runs(zero_shot)


def _binding() -> dict[str, object]:
    return extension._sealed(
        {
            "schema_version": 1,
            "binding_type": "resilient_v2x_p1_p0_epoch50_checkpoint_reuse",
            "candidate_identity": extension.ZERO_SHOT_CANDIDATE_IDENTITY,
            "identity_alias_mapping_seal_sha256": (
                extension.identity_alias_mapping()["seal_sha256"]
            ),
            "evidence_class": extension.ZERO_SHOT_EVIDENCE_CLASS,
            "config_subject": extension.P1_CONFIG_SUBJECT,
            "config_experiment_identity": extension.P1_EXPERIMENT_IDENTITY,
            "config_path": extension.P1_CONFIG,
            "config_sha256": extension.P1_CONFIG_SHA256,
            "checkpoint_subject": extension.P0_CONFIG_SUBJECT,
            "checkpoint_experiment_identity": extension.P0_EXPERIMENT_IDENTITY,
            "checkpoint_policy": extension.CHECKPOINT_POLICY,
            "checkpoint_role": extension.CHECKPOINT_ROLE,
            "weights_retrained": False,
            "optimization_origin": extension.OPTIMIZATION_ORIGIN,
            "source_task": {
                "task_id": extension.P0_TASK_ID,
                "required_status": "completed",
                "run_contract_sha256": "2" * 64,
                "initialization_audit_sha256": "3" * 64,
                "training_seed": extension.TRAINING_SEED,
                "training_overlay_protocol_seed": (
                    extension.TRAINING_OVERLAY_PROTOCOL_SEED
                ),
                "max_epochs": 50,
                "precision": "FP32",
            },
            "checkpoint": {
                "model_id": "e" * 32,
                "model_name": extension.P0_FINAL_MODEL_NAME,
                "model_url": (
                    f"{extension.FILES_SERVER_URI}/models/"
                    f"{extension.P0_FINAL_REMOTE_FILENAME}"
                ),
                "source_filename": extension.P0_FINAL_SOURCE_FILENAME,
                "remote_filename": extension.P0_FINAL_REMOTE_FILENAME,
                "size_bytes": 123,
                "sha256": "4" * 64,
                "bytes_recomputed": True,
            },
            "forbidden_checkpoint_roles": list(extension.FORBIDDEN_CHECKPOINT_ROLES),
            "formal_evaluation": {
                "protocol_id": extension.PROTOCOL_ID,
                "sample_count": extension.SAMPLE_COUNT,
                "training_seed": extension.TRAINING_SEED,
                "overlay_protocol_seed": extension.TRAINING_OVERLAY_PROTOCOL_SEED,
                "delays_ms": list(extension.DELAYS_MS),
                "conditions": list(extension.CONDITIONS),
                "run_count": extension.RUN_COUNT,
            },
        }
    )


def _canonical_results() -> dict[str, dict[str, object]]:
    return {
        subject: {
            "subject": subject,
            "evaluation_task_id": f"{index + 100:032x}",
            "training_checkpoint_sha256": f"{index + 200:064x}",
            "metrics": {},
        }
        for index, subject in enumerate(formal.SUBJECT_ORDER)
    }


def _build_extension(
    monkeypatch: pytest.MonkeyPatch,
    *,
    zero_shot: float,
) -> dict[str, object]:
    all_runs, candidate_runs = _all_runs(zero_shot=zero_shot)
    canonical_results = _canonical_results()
    monkeypatch.setattr(
        leaderboard,
        "validate_canonical_leaderboard",
        lambda payload: (canonical_results, CANONICAL_LEADERBOARD_SEAL),
    )
    binding = _binding()
    return leaderboard.build_extension_leaderboard(
        canonical_leaderboard_task_id=CANONICAL_LEADERBOARD_TASK_ID,
        canonical_leaderboard={"unused": True},
        candidate_evaluation_task_id=CANDIDATE_EVALUATION_TASK_ID,
        candidate_binding=binding,
        candidate_runs=candidate_runs,
        candidate_provenance={
            "evaluation_task_id": CANDIDATE_EVALUATION_TASK_ID,
            "identity_alias_mapping_seal_sha256": (
                extension.identity_alias_mapping()["seal_sha256"]
            ),
            "checkpoint_reuse_binding_seal_sha256": binding["seal_sha256"],
            "checkpoint_reuse_equivalence_seal_sha256": "5" * 64,
            "evaluation_receipt_seal_sha256": "6" * 64,
            "metrics_sha256": "7" * 64,
        },
        canonical_baseline_runs={
            subject: all_runs[subject] for subject in formal.BASELINE_SUBJECTS
        },
    )


def _audit_chain() -> dict[str, object]:
    return {
        "training_provenance_task_id": "8" * 32,
        "training_progress_seal_sha256": "8" * 64,
        "training_provenance_seal_sha256": "9" * 64,
        "source_revision_equivalence": {},
        "source_revision_equivalence_seal_sha256": "a" * 64,
        "source_revision_subject_map": {},
        "source_revision_subject_map_seal_sha256": "b" * 64,
        "evaluation_source_revision_tree_sha256": "c" * 64,
        "evaluation_source_revision": {},
        "training_script_equivalence": {},
        "evaluation_script_sha256": "d" * 64,
    }


def _canonical_selection(*, primary: float = 51.0) -> dict[str, object]:
    all_runs, _ = _all_runs(primary=primary)
    return formal.build_selection(
        audit_task_id="e" * 32,
        leaderboard_task_id=CANONICAL_LEADERBOARD_TASK_ID,
        audit_seal="e" * 64,
        leaderboard_seal=CANONICAL_LEADERBOARD_SEAL,
        audit_chain=_audit_chain(),
        runs_by_subject=all_runs,
    )


def test_extension_leaderboard_references_but_never_mutates_canonical_26(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _build_extension(monkeypatch, zero_shot=52.0)
    extension._require_seal(payload, context="test extension leaderboard")
    canonical = payload["canonical_26"]
    candidate = payload["candidate"]

    assert canonical == {
        "leaderboard_task_id": CANONICAL_LEADERBOARD_TASK_ID,
        "artifact": "formal_1337_leaderboard",
        "seal_sha256": CANONICAL_LEADERBOARD_SEAL,
        "subject_order": list(formal.SUBJECT_ORDER),
        "subject_count": 26,
        "mutated": False,
    }
    assert extension.ZERO_SHOT_CANDIDATE_IDENTITY not in canonical["subject_order"]
    assert payload["extension_subject_order"] == [
        extension.ZERO_SHOT_CANDIDATE_IDENTITY
    ]
    assert payload["identity_alias_mapping"] == {
        "artifact": extension.IDENTITY_ALIAS_MAPPING_ARTIFACT,
        "seal_sha256": extension.identity_alias_mapping()["seal_sha256"],
        "canonical_identity": extension.ZERO_SHOT_CANDIDATE_IDENTITY,
        "display_subject_alias": extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS,
        "mapping_cardinality": "one_to_one",
        "alias_is_candidate_identity": False,
    }
    assert candidate["subject"] == extension.ZERO_SHOT_CANDIDATE_IDENTITY
    assert candidate["display_subject_alias"] == (
        extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS
    )
    assert candidate["kind"] == "diagnostic_candidate_extension"
    assert candidate["weights_retrained"] is False
    assert candidate["optimization_origin"] == "P0"
    assert candidate["eligible_for_trained_26_method_table"] is False
    assert candidate["config_subject"] == extension.P1_CONFIG_SUBJECT
    assert candidate["checkpoint_subject"] == extension.P0_CONFIG_SUBJECT
    assert candidate["source_task_id"] == extension.P0_TASK_ID
    assert candidate["evaluation_task_id"] == CANDIDATE_EVALUATION_TASK_ID
    assert len(candidate["runs"]) == 12
    assert payload["candidate_screening"]["gate_passed"] is True


@pytest.mark.xfail(
    strict=True,
    reason=(
        "immutable predecessor plan intentionally pins pre-successor selector bytes; "
        "the active successor receipt supersedes it"
    ),
)
def test_independent_local_deployment_plan_is_sealed_and_hash_pinned() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "artifacts/resilient_v2x/sota_round2_p1_zero_shot"
        / "p1-zero-shot-candidate-extension-plan.json"
    )
    plan = json.loads(path.read_text(encoding="utf-8"))
    extension._require_seal(plan, context="P1 zero-shot local deployment plan")
    assert plan["remote_state_changed"] is False
    assert plan["candidate"]["canonical_identity"] == (
        extension.ZERO_SHOT_CANDIDATE_IDENTITY
    )
    assert plan["candidate"]["display_subject_alias"] == (
        extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS
    )
    assert (
        plan["identity_alias_mapping"]["seal_sha256"]
        == (extension.identity_alias_mapping()["seal_sha256"])
    )
    for item in plan["local_implementation"]["files"]:
        source = root / item["path"]
        assert source.stat().st_size == item["bytes"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]
    for subject in ("p0", "p1", "p2"):
        reference = plan["immutable_inputs"][subject]
        upstream_path = root / reference["deployment_plan_path"]
        upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
        assert upstream["seal_sha256"] == reference["deployment_plan_seal_sha256"]
        assert (
            hashlib.sha256(upstream_path.read_bytes()).hexdigest()
            == (reference["deployment_plan_sha256"])
        )


def test_extension_leaderboard_validator_rejects_identity_conflation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _build_extension(monkeypatch, zero_shot=52.0)
    round2.validate_extension_leaderboard(payload)

    tampered = copy.deepcopy(payload)
    tampered["candidate"]["checkpoint_subject"] = extension.P1_CONFIG_SUBJECT
    tampered = extension._sealed(tampered)
    with pytest.raises(extension.P1CheckpointReuseError, match="checkpoint_subject"):
        round2.validate_extension_leaderboard(tampered)

    leaked = copy.deepcopy(payload)
    leaked["canonical_26"]["subject_order"].append(
        extension.ZERO_SHOT_CANDIDATE_IDENTITY
    )
    leaked = extension._sealed(leaked)
    with pytest.raises(extension.P1CheckpointReuseError, match="canonical-26"):
        round2.validate_extension_leaderboard(leaked)


def test_round2_selector_promotes_zero_shot_only_to_independent_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_payload = _build_extension(monkeypatch, zero_shot=52.0)
    canonical_payload = _canonical_selection(primary=51.0)
    result = round2.build_round2_selection(
        canonical_selector_task_id=CANONICAL_SELECTOR_TASK_ID,
        canonical_selection=canonical_payload,
        extension_leaderboard_task_id=EXTENSION_LEADERBOARD_TASK_ID,
        extension_leaderboard=extension_payload,
    )
    extension._require_seal(result, context="test round2 selection")

    assert result["selection_is_final"] is False
    assert not any(str(key).endswith("seed_confirmation") for key in result)
    assert result["requires_p1_independent_training"] is True
    assert result["checkpoint_reuse_can_enter_trained_table"] is False
    assert result["screening_selected_candidate"] == (
        extension.ZERO_SHOT_CANDIDATE_IDENTITY
    )
    assert result["screening_selected_candidate_eligible_for_final_claim"] is False
    assert result["recommended_action"] == "train_p1_50e_then_evaluate"
    assert result["candidate_subjects"] == list(round2.ROUND2_CANDIDATE_ORDER)
    assert result["candidate_count"] == 7
    assert result["evaluated_candidate_count"] == 2
    assert result["pending_candidate_count"] == 5
    assert result["selected_candidate"] is None
    assert result["canonical_selector"]["mutated"] is False
    assert result["canonical_leaderboard"]["mutated"] is False
    assert result["candidate_identity_alias_mapping"]["canonical_identity"] == (
        extension.ZERO_SHOT_CANDIDATE_IDENTITY
    )
    assert result["candidate_identity_alias_mapping"]["display_subject_alias"] == (
        extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS
    )
    zero_row = next(
        row
        for row in result["candidate_results"]
        if row["subject"] == extension.ZERO_SHOT_CANDIDATE_IDENTITY
    )
    assert zero_row["evidence_class"] == extension.ZERO_SHOT_EVIDENCE_CLASS
    assert zero_row["weights_retrained"] is False
    assert zero_row["eligible_for_final_paper_selection"] is False
    assert zero_row["display_subject_alias"] == (
        extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS
    )


def test_round2_selector_selects_trained_single_seed_winner_as_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_payload = _build_extension(monkeypatch, zero_shot=50.5)
    canonical_payload = _canonical_selection(primary=52.0)
    result = round2.build_round2_selection(
        canonical_selector_task_id=CANONICAL_SELECTOR_TASK_ID,
        canonical_selection=canonical_payload,
        extension_leaderboard_task_id=EXTENSION_LEADERBOARD_TASK_ID,
        extension_leaderboard=extension_payload,
    )
    assert result["screening_selected_candidate"] == "resilient_v2x"
    assert result["screening_selected_candidate_eligible_for_final_claim"] is True
    assert result["requires_p1_independent_training"] is False
    assert result["selection_is_final"] is True
    assert result["selected_candidate"] == "resilient_v2x"
    assert result["recommended_action"] == "select_single_seed_winner"


def test_round2_selector_rejects_tampered_candidate_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_payload = _build_extension(monkeypatch, zero_shot=52.0)
    tampered = copy.deepcopy(extension_payload)
    tampered["candidate_screening"]["gate_passed"] = False
    tampered = extension._sealed(tampered)
    with pytest.raises(extension.P1CheckpointReuseError, match="overall gate"):
        round2.validate_extension_leaderboard(tampered)


def test_canonical_selector_and_extension_seals_are_cross_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_payload = _build_extension(monkeypatch, zero_shot=52.0)
    canonical_payload = _canonical_selection(primary=51.0)
    drifted = copy.deepcopy(canonical_payload)
    drifted["leaderboard_seal_sha256"] = "f" * 64
    drifted = formal._sealed(drifted)
    with pytest.raises(extension.P1CheckpointReuseError, match="leaderboard_seal"):
        round2.build_round2_selection(
            canonical_selector_task_id=CANONICAL_SELECTOR_TASK_ID,
            canonical_selection=drifted,
            extension_leaderboard_task_id=EXTENSION_LEADERBOARD_TASK_ID,
            extension_leaderboard=extension_payload,
        )
