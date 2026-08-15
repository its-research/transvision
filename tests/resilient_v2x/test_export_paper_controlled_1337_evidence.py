from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import export_paper_controlled_1337_evidence as export


def test_artifact_payload_uses_authoritative_preview_when_download_is_unauthorized() -> None:
    payload = {"document_type": "sealed", "value": 1}

    class Artifact:
        def get(self) -> object:
            raise ValueError("files server returned 401")

    record = {
        "key": "evidence",
        "hash": _sha(1),
        "content_size": 1,
        "uri": "http://10.100.34.118:8081/task/evidence.json",
        "type_data": {"preview": json.dumps(payload)},
    }
    task = SimpleNamespace(
        artifacts={"evidence": Artifact()},
        data=SimpleNamespace(
            execution=SimpleNamespace(artifacts=[record])
        ),
    )

    observed, metadata = export._artifact_payload(
        task, "evidence", context="formal evidence"
    )

    assert observed == payload
    assert metadata == record


def test_artifact_payload_downloads_truncated_preview_with_authenticated_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"document_type": "sealed", "rows": list(range(100))}
    encoded = json.dumps(payload).encode()

    class Artifact:
        def get(self) -> object:
            raise ValueError("files server returned 401")

    record = {
        "key": "evidence",
        "hash": hashlib.sha256(encoded).hexdigest(),
        "content_size": len(encoded),
        "uri": "http://10.100.34.118:8081/task/evidence.json",
        "type_data": {"preview": encoded[:20].decode()},
    }
    task = SimpleNamespace(
        artifacts={"evidence": Artifact()},
        data=SimpleNamespace(execution=SimpleNamespace(artifacts=[record])),
    )
    monkeypatch.setattr(
        export,
        "_authenticated_json_artifact",
        lambda metadata, **_kwargs: json.loads(encoded),
    )

    observed, metadata = export._artifact_payload(
        task, "evidence", context="formal evidence"
    )

    assert observed == payload
    assert metadata == record


def test_training_config_path_accepts_legacy_declared_resolved_workspace_path() -> None:
    assert export._training_config_path(
        {
            "declared": None,
            "declared_resolved": (
                "/workspace/resilient-v2x-5090-runtime/"
                "configs/resilient_v2x/baselines/ffnet.py"
            ),
        },
        "ffnet",
    ) == "configs/resilient_v2x/baselines/ffnet.py"


def test_training_config_path_rejects_untrusted_absolute_path() -> None:
    with pytest.raises(export.PaperEvidenceExportError, match="unavailable"):
        export._training_config_path(
            {"declared": None, "declared_resolved": "/tmp/ffnet.py"}, "ffnet"
        )


@pytest.mark.parametrize(
    "path",
    [
        "/models/ffnet_epoch_50.pth",
        "/clearml/cache/0123456789abcdef0123456789abcdef.ffnet_epoch_50.pth",
    ],
)
def test_final_checkpoint_path_accepts_canonical_and_clearml_cache_names(
    path: str,
) -> None:
    assert export._is_final_checkpoint_path(path, "ffnet")


def test_final_checkpoint_path_rejects_nonfinal_checkpoint() -> None:
    assert not export._is_final_checkpoint_path("ffnet_epoch_20.pth", "ffnet")


def test_prediction_archive_extra_accepts_mmengine_diagnostics_only() -> None:
    assert export._is_allowed_prediction_archive_extra(
        "delay_000_full/20260812_175309/20260812_175309.log"
    )
    assert export._is_allowed_prediction_archive_extra(
        "delay_000_full/20260812_175309/vis_data/config.py"
    )
    assert not export._is_allowed_prediction_archive_extra(
        "delay_000_full/20260812_175309/arbitrary.bin"
    )


def _id(value: int) -> str:
    return f"{value:032x}"


def _sha(value: int) -> str:
    return f"{value:064x}"


def _runs(subject: str, *, full_0: float, other: float) -> list[dict[str, object]]:
    result = []
    for index, (delay, condition) in enumerate(
        (
            pair
            for delay in export.DELAYS_MS
            for pair in ((delay, c) for c in export.CONDITIONS)
        )
    ):
        value = full_0 if (delay, condition) == (0, "Full") else other
        result.append(
            {
                "condition_id": export._condition_id(delay, condition),
                "delay_ms": delay,
                "condition": condition,
                "agent_scope": export.AGENT_SCOPE,
                "sample_count": export.SAMPLE_COUNT,
                "ground_truth_count": export.GROUND_TRUTH_COUNT,
                "unsupported_sample_count": export.UNSUPPORTED_SAMPLE_COUNT,
                "sample_ids_sha256": export.SAMPLE_IDS_SHA256,
                "prediction_sha256": _sha(2000 + index),
                "prediction_content_sha256": _sha(3000 + index),
                "metrics": {key: value for key in export.AP_METRIC_KEYS},
            }
        )
    return result


def _identity(index: int, subject: str) -> dict[str, object]:
    return {
        "model_name": f"ResilientV2X {subject} final checkpoint",
        "modality": "LiDAR+Camera",
        "backbone": "PointPillars+ResNet-50/LSS",
        "training_task_id": _id(100 + index),
        "training_model_id": _id(200 + index),
        "evaluation_task_id": _id(300 + index),
        "checkpoint_sha256": _sha(400 + index),
        "checkpoint_size_bytes": 1000 + index,
        "checkpoint_bytes_verified": True,
        "source_revision_tree_sha256": _sha(500 + index),
        "source_dataset_id": _id(600 + index),
        "source_archive_sha256": _sha(700 + index),
        "config_path": f"configs/resilient_v2x/{subject}.py",
        "config_sha256": _sha(800 + index),
        "training_script_sha256": _sha(900 + index),
        "teacher_task_id": export.TEACHER_TASK_ID,
        "teacher_model_id": export.TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": export.TEACHER_CHECKPOINT_SHA256,
        "training_dataset_id": export.TRAINING_DATASET_ID,
        "metrics_artifact_sha256": _sha(1000 + index),
        "prediction_evidence_artifact_sha256": _sha(1100 + index),
        "prediction_evidence_archive_sha256": _sha(1100 + index),
    }


def _fixture() -> tuple[export.ChainEvidence, list[export.CollectedSubject]]:
    watcher_id = _id(1)
    leaderboard_id = _id(2)
    audit_id = _id(3)
    selector_id = _id(4)
    controller_id = _id(5)
    provenance_id = _id(6)
    plan_entries = []
    training_entries = []
    formal_runs: dict[str, list[dict[str, object]]] = {}
    for index, subject in enumerate(export.FORMAL_SUBJECT_ORDER, start=1):
        plan_entries.append(
            {
                "subject": subject,
                "evaluation_task_id": _id(300 + index),
                "queue": ("GPU4-A100", "GPU4-V100", "GPU4-5090")[index % 3],
            }
        )
        training_entries.append(
            {
                "index": index,
                "subject": subject,
                "kind": "baseline"
                if subject in export.FORMAL_LEADERBOARD_BASELINE_SUBJECTS
                else "improvement",
                "training_task_id": _id(100 + index),
                "training_predecessor_task_id": _id(1000 + index),
                "model_id": _id(200 + index),
                "model_name": f"ResilientV2X {subject} final checkpoint",
                "model_url": f"http://10.100.34.118:8081/models/{subject}_epoch_50.pth",
                "checkpoint_filename": "epoch_50.pth",
                "checkpoint_sha256": _sha(400 + index),
                "checkpoint_size_bytes": 1000 + index,
                "training_seed": export.TRAINING_SEED,
                "training_overlay_protocol_seed": export.TRAINING_SEED,
                "common_teacher_initialization_audit_artifact": "common_teacher_initialization_audit",
                "common_teacher_initialization_audit_sha256": _sha(1200 + index),
            }
        )
        if subject in export.FORMAL_LEADERBOARD_BASELINE_SUBJECTS:
            formal_runs[subject] = _runs(subject, full_0=50.0, other=40.0)
        elif subject == "resilient_v2x":
            formal_runs[subject] = _runs(subject, full_0=49.6, other=41.0)
        else:
            formal_runs[subject] = _runs(subject, full_0=45.0, other=35.0)
    plan = export._sealed(
        {
            "schema_version": 2,
            "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
            "training_controller_task_id": controller_id,
            "training_provenance_task_id": provenance_id,
            "protocol_id": export.PROTOCOL_ID,
            "sample_count": export.SAMPLE_COUNT,
            "delays_ms": list(export.DELAYS_MS),
            "conditions": list(export.CONDITIONS),
            "run_count": export.RUN_COUNT,
            "subject_order": list(export.FORMAL_SUBJECT_ORDER),
            "entries": plan_entries,
        }
    )
    manifest = export._sealed(
        {
            "schema_version": 1,
            "manifest_type": "resilient_v2x_formal_1337_training_inputs",
            "protocol_id": export.PROTOCOL_ID,
            "sample_count": export.SAMPLE_COUNT,
            "delays_ms": list(export.DELAYS_MS),
            "conditions": list(export.CONDITIONS),
            "run_count": export.RUN_COUNT,
            "checkpoint_policy": export.CHECKPOINT_POLICY,
            "training_seed": export.TRAINING_SEED,
            "training_overlay_protocol_seed": export.TRAINING_SEED,
            "evaluation_release_semantics": "formal_manifest_after_full_training_suite_completion",
            "subject_order": list(export.FORMAL_SUBJECT_ORDER),
            "subject_count": len(export.FORMAL_SUBJECT_ORDER),
            "entries": training_entries,
        }
    )
    results = []
    for index, subject in enumerate(export.FORMAL_SUBJECT_ORDER, start=1):
        training = training_entries[index - 1]
        results.append(
            {
                "index": index,
                "subject": subject,
                "kind": training["kind"],
                "training_task_id": training["training_task_id"],
                "training_model_id": training["model_id"],
                "training_checkpoint_sha256": training["checkpoint_sha256"],
                "source_revision_tree_sha256": _sha(500 + index),
                "evaluation_task_id": plan_entries[index - 1]["evaluation_task_id"],
                "metrics": {
                    key: export._metric_summary(formal_runs[subject], key)
                    for key in export.AP_METRIC_KEYS
                },
            }
        )
    leaderboard = export._sealed(
        {
            "schema_version": 3,
            "leaderboard_type": "resilient_v2x_formal_1337_leaderboard",
            "protocol_id": export.PROTOCOL_ID,
            "sample_count": export.SAMPLE_COUNT,
            "ground_truth_count": export.GROUND_TRUTH_COUNT,
            "unsupported_sample_count": export.UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(export.DELAYS_MS),
            "conditions": list(export.CONDITIONS),
            "run_count_per_subject": export.RUN_COUNT,
            "training_seed": export.TRAINING_SEED,
            "training_overlay_protocol_seed": export.TRAINING_SEED,
            "training_controller_task_id": controller_id,
            "watcher_task_id": watcher_id,
            "training_manifest_seal_sha256": manifest["seal_sha256"],
            "evaluation_plan_seal_sha256": plan["seal_sha256"],
            "subject_order": list(export.FORMAL_SUBJECT_ORDER),
            "subject_count": len(export.FORMAL_SUBJECT_ORDER),
            "baseline_subjects": list(export.FORMAL_LEADERBOARD_BASELINE_SUBJECTS),
            "baseline_count": len(export.FORMAL_LEADERBOARD_BASELINE_SUBJECTS),
            "metric_keys": list(export.AP_METRIC_KEYS),
            "results": results,
        }
    )
    audit = export._sealed(
        {
            "schema_version": 3,
            "document_type": "resilient_v2x_formal_1337_comparability_audit",
            "passed": True,
            "audit_task_id": audit_id,
            "training_controller_task_id": controller_id,
            "training_provenance_task_id": provenance_id,
            "watcher_task_id": watcher_id,
            "leaderboard_task_id": leaderboard_id,
            "protocol_id": export.PROTOCOL_ID,
            "training_dataset_id": export.TRAINING_DATASET_ID,
            "training_seed": export.TRAINING_SEED,
            "checkpoint_policy": export.CHECKPOINT_POLICY,
            "sample_count": export.SAMPLE_COUNT,
            "ground_truth_count": export.GROUND_TRUTH_COUNT,
            "unsupported_sample_count": export.UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(export.DELAYS_MS),
            "conditions": list(export.CONDITIONS),
            "run_count_per_subject": export.RUN_COUNT,
            "subject_order": list(export.FORMAL_SUBJECT_ORDER),
            "subject_count": len(export.FORMAL_SUBJECT_ORDER),
            "total_evaluation_run_count": len(export.FORMAL_SUBJECT_ORDER)
            * export.RUN_COUNT,
            "metric_keys": list(export.AP_METRIC_KEYS),
            "training_manifest_seal_sha256": manifest["seal_sha256"],
            "evaluation_plan_seal_sha256": plan["seal_sha256"],
            "leaderboard_seal_sha256": leaderboard["seal_sha256"],
            "training_tasks": [
                {"subject": subject} for subject in export.FORMAL_SUBJECT_ORDER
            ],
            "evaluation_tasks": [
                {"subject": subject} for subject in export.FORMAL_SUBJECT_ORDER
            ],
        }
    )
    paper_subjects = []
    for subject in (*export.BASELINE_SUBJECTS, "resilient_v2x"):
        index = export.FORMAL_SUBJECT_ORDER.index(subject) + 1
        paper_subjects.append(
            export.CollectedSubject(
                subject,
                export.DISPLAY_NAMES.get(subject, "Selected controlled method"),
                _identity(index, subject),
                formal_runs[subject],
            )
        )
    gate = export._gate_result(paper_subjects, "resilient_v2x")
    selector = export._sealed(
        {
            "schema_version": 4,
            "document_type": "resilient_v2x_formal_candidate_selection",
            "selection_is_final": True,
            "selected_candidate": "resilient_v2x",
            "protocol_id": export.PROTOCOL_ID,
            "sample_count": export.SAMPLE_COUNT,
            "ground_truth_count": export.GROUND_TRUTH_COUNT,
            "unsupported_sample_count": export.UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(export.DELAYS_MS),
            "conditions": list(export.CONDITIONS),
            "run_count_per_subject": export.RUN_COUNT,
            "baseline_subjects": list(export.BASELINE_SUBJECTS),
            "baseline_count": len(export.BASELINE_SUBJECTS),
            "audit_task_id": audit_id,
            "leaderboard_task_id": leaderboard_id,
            "watcher_task_id": watcher_id,
            "audit_seal_sha256": audit["seal_sha256"],
            "leaderboard_seal_sha256": leaderboard["seal_sha256"],
            "candidate_results": [
                {
                    "subject": "resilient_v2x",
                    "eligible_for_final_paper_selection": True,
                    **gate,
                }
            ],
        }
    )
    chain = export.ChainEvidence(
        watcher_id,
        leaderboard_id,
        audit_id,
        selector_id,
        plan,
        leaderboard,
        audit,
        selector,
        manifest,
    )
    return chain, paper_subjects


def _build() -> dict[str, object]:
    chain, subjects = _fixture()
    return export.build_paper_evidence(
        chain=chain,
        subjects=subjects,
        producer_task_id=_id(999),
        producer_script_sha256=_sha(999),
    )


def _v2_fixture() -> tuple[export.ChainEvidence, list[export.CollectedSubject]]:
    chain, subjects = _fixture()
    legacy_winner = chain.selector["candidate_results"][0]
    winner = {
        **legacy_winner,
        "candidate_label": "formal_main",
        "fixed_order_index": 0,
        "evidence_class": "formal_trained_50e_candidate",
        "weights_retrained": True,
        "evidence_status": "validated",
        "performance_rank": 1,
    }
    fingerprint_payload = {
        "schema_version": 1,
        "protocol_id": export.PROTOCOL_ID,
        "sample_count": export.SAMPLE_COUNT,
        "ground_truth_count": export.GROUND_TRUTH_COUNT,
        "unsupported_sample_count": export.UNSUPPORTED_SAMPLE_COUNT,
        "manifest_content_sha256": export.MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": export.OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": export.SAMPLE_IDS_SHA256,
        "delays_ms": list(export.DELAYS_MS),
        "conditions": list(export.CONDITIONS),
        "run_count": export.RUN_COUNT,
        "leadership_metric": export.LEADERSHIP_METRIC,
    }
    fingerprint = {
        **fingerprint_payload,
        "fingerprint_sha256": export._content_sha256(fingerprint_payload),
    }
    input_bindings = {
        "formal_inputs_seal_sha256": _sha(5000),
        "candidate_manifest_task_id": _id(5001),
        "candidate_manifest_seal_sha256": _sha(5002),
        "sequential_evidence_seal_sha256": None,
        "sequential_gate_receipt_count": 0,
    }
    selector = export._sealed(
        {
            "schema_version": 2,
            "document_type": export.FINAL_SELECTION_DOCUMENT_TYPE,
            "status": "selected",
            "selection_is_final": True,
            "selection_claim": "controlled_leader_selected",
            "claim_scope": (
                "single_seed_same_protocol_DAIR_controlled_leadership"
            ),
            "training_seed": export.TRAINING_SEED,
            "checkpoint_policy": export.CHECKPOINT_POLICY,
            "selected_method_identity_artifact": (
                export.SELECTED_METHOD_IDENTITY_ARTIFACT
            ),
            "selected_candidate": "resilient_v2x",
            "protocol_evidence_fingerprint": fingerprint,
            "baseline_subjects": list(export.BASELINE_SUBJECTS),
            "baseline_count": len(export.BASELINE_SUBJECTS),
            "candidate_results": [winner],
            "input_bindings": input_bindings,
        }
    )
    collected = subjects[-1].identity
    method_binding = {
        "config_path": collected["config_path"],
        "config_sha256": collected["config_sha256"],
        "source_dataset_id": collected["source_dataset_id"],
        "source_revision_sha256": collected["source_revision_tree_sha256"],
        "source_archive_sha256": collected["source_archive_sha256"],
        "training_dataset_id": collected["training_dataset_id"],
        "teacher_task_id": collected["teacher_task_id"],
        "teacher_model_id": collected["teacher_model_id"],
        "teacher_checkpoint_sha256": collected["teacher_checkpoint_sha256"],
        "training_task_id": collected["training_task_id"],
        "evaluation_task_id": collected["evaluation_task_id"],
        "model_id": collected["training_model_id"],
        "checkpoint_filename": "epoch_50.pth",
        "checkpoint_sha256": collected["checkpoint_sha256"],
        "checkpoint_size_bytes": collected["checkpoint_size_bytes"],
        "run_contract_artifact_sha256": _sha(5010),
        "initialization_audit_artifact_sha256": _sha(5011),
        "final_checkpoint_contract_artifact_sha256": _sha(5012),
        "evaluation_plan_artifact_sha256": _sha(5013),
        "metrics_artifact_sha256": collected["metrics_artifact_sha256"],
        "prediction_evidence_artifact_sha256": collected[
            "prediction_evidence_artifact_sha256"
        ],
        "prediction_evidence_archive_sha256": collected[
            "prediction_evidence_archive_sha256"
        ],
    }
    identity = export._sealed(
        {
            "schema_version": 2,
            "document_type": export.SELECTED_METHOD_IDENTITY_DOCUMENT_TYPE,
            "status": "sealed",
            "selected_subject": "resilient_v2x",
            "selected_candidate_label": "formal_main",
            "selected_fixed_order_index": 0,
            "selection_artifact": "final_single_seed_selection",
            "selection_seal_sha256": selector["seal_sha256"],
            "training_seed": export.TRAINING_SEED,
            "protocol_evidence_fingerprint": fingerprint,
            "checkpoint_policy": export.CHECKPOINT_POLICY,
            "method_binding": method_binding,
            "evidence_bindings": {
                "formal_inputs_seal_sha256": input_bindings[
                    "formal_inputs_seal_sha256"
                ],
                "candidate_manifest_task_id": input_bindings[
                    "candidate_manifest_task_id"
                ],
                "candidate_manifest_seal_sha256": input_bindings[
                    "candidate_manifest_seal_sha256"
                ],
                "sequential_evidence_seal_sha256": None,
                "sequential_gate_receipt_seal_sha256": None,
            },
            "gate_artifact": {
                "artifact_name": "final_single_seed_selection",
                "selection_seal_sha256": selector["seal_sha256"],
                "upstream_candidate_gate_artifact_sha256": None,
            },
            "gate_result": winner,
            "gate_result_content_sha256": export._content_sha256(winner),
        }
    )
    upgraded = export.ChainEvidence(
        chain.watcher_task_id,
        chain.leaderboard_task_id,
        chain.audit_task_id,
        chain.selector_task_id,
        chain.plan,
        chain.leaderboard,
        chain.audit,
        selector,
        chain.training_manifest,
        identity,
    )
    return upgraded, subjects


def test_builds_exact_sealed_288_cell_paper_document() -> None:
    result = _build()

    assert result["document_type"] == export.FORMAL_EVIDENCE_TYPE
    assert result["training_seed"] == export.TRAINING_SEED
    assert result["baseline_subjects"] == list(export.BASELINE_SUBJECTS)
    assert result["selected_candidate"] == "resilient_v2x"
    assert len(result["entries"]) == 6
    assert sum(len(row["runs"]) for row in result["entries"]) == 72
    assert 6 * 12 * 4 == 288
    assert export._sealed(result) == result


def test_v2_selector_consumes_sealed_selected_method_identity() -> None:
    chain, subjects = _v2_fixture()

    result = export.build_paper_evidence(
        chain=chain,
        subjects=subjects,
        producer_task_id=_id(999),
        producer_script_sha256=_sha(999),
    )

    assert result["schema_version"] == 2
    assert result["selected_method_identity_seal_sha256"] == (
        chain.selected_method_identity["seal_sha256"]
    )


def test_v2_selector_rejects_missing_selected_method_identity() -> None:
    chain, subjects = _v2_fixture()
    missing = export.ChainEvidence(
        chain.watcher_task_id,
        chain.leaderboard_task_id,
        chain.audit_task_id,
        chain.selector_task_id,
        chain.plan,
        chain.leaderboard,
        chain.audit,
        chain.selector,
        chain.training_manifest,
    )

    with pytest.raises(
        export.PaperEvidenceExportError,
        match="lacks the sealed selected-method identity",
    ):
        export.build_paper_evidence(
            chain=missing,
            subjects=subjects,
            producer_task_id=_id(999),
            producer_script_sha256=_sha(999),
        )


def test_v2_identity_checkpoint_must_match_collected_bytes() -> None:
    chain, subjects = _v2_fixture()
    identity = dict(chain.selected_method_identity)
    identity["method_binding"] = {
        **identity["method_binding"],
        "checkpoint_sha256": _sha(9999),
    }
    identity = export._sealed(identity)
    drifted = export.ChainEvidence(
        chain.watcher_task_id,
        chain.leaderboard_task_id,
        chain.audit_task_id,
        chain.selector_task_id,
        chain.plan,
        chain.leaderboard,
        chain.audit,
        chain.selector,
        chain.training_manifest,
        identity,
    )

    with pytest.raises(
        export.PaperEvidenceExportError,
        match="checkpoint_sha256 differs from independently collected evidence",
    ):
        export.build_paper_evidence(
            chain=drifted,
            subjects=subjects,
            producer_task_id=_id(999),
            producer_script_sha256=_sha(999),
        )


def test_rejects_missing_metric_cell() -> None:
    chain, subjects = _fixture()
    del subjects[-1].runs[-1]["metrics"][export.AP_METRIC_KEYS[-1]]

    with pytest.raises(export.PaperEvidenceExportError, match="metric inventory"):
        export.build_paper_evidence(
            chain=chain,
            subjects=subjects,
            producer_task_id=_id(999),
            producer_script_sha256=_sha(999),
        )


def test_rejects_unverified_checkpoint_bytes() -> None:
    chain, subjects = _fixture()
    subjects[0].identity["checkpoint_bytes_verified"] = False

    with pytest.raises(export.PaperEvidenceExportError, match="not verified"):
        export.build_paper_evidence(
            chain=chain,
            subjects=subjects,
            producer_task_id=_id(999),
            producer_script_sha256=_sha(999),
        )


def test_rejects_chain_seal_drift() -> None:
    chain, subjects = _fixture()
    chain.leaderboard["sample_count"] = 1336

    with pytest.raises(export.PaperEvidenceExportError, match="seal mismatch"):
        export.build_paper_evidence(
            chain=chain,
            subjects=subjects,
            producer_task_id=_id(999),
            producer_script_sha256=_sha(999),
        )


def test_rejects_selector_gate_values_not_bound_to_runs() -> None:
    chain, subjects = _fixture()
    winner = chain.selector["candidate_results"][0]
    winner["ranking_values"]["mean_12_margin"] += 0.01
    chain.selector.clear()
    chain.selector.update(
        export._sealed(
            {
                "schema_version": 4,
                "document_type": "resilient_v2x_formal_candidate_selection",
                "selection_is_final": True,
                "selected_candidate": "resilient_v2x",
                "protocol_id": export.PROTOCOL_ID,
                "sample_count": export.SAMPLE_COUNT,
                "ground_truth_count": export.GROUND_TRUTH_COUNT,
                "unsupported_sample_count": export.UNSUPPORTED_SAMPLE_COUNT,
                "delays_ms": list(export.DELAYS_MS),
                "conditions": list(export.CONDITIONS),
                "run_count_per_subject": export.RUN_COUNT,
                "baseline_subjects": list(export.BASELINE_SUBJECTS),
                "baseline_count": len(export.BASELINE_SUBJECTS),
                "audit_task_id": chain.audit_task_id,
                "leaderboard_task_id": chain.leaderboard_task_id,
                "watcher_task_id": chain.watcher_task_id,
                "audit_seal_sha256": chain.audit["seal_sha256"],
                "leaderboard_seal_sha256": chain.leaderboard["seal_sha256"],
                "candidate_results": [winner],
            }
        )
    )

    with pytest.raises(export.PaperEvidenceExportError, match="ranking is unbound"):
        export.build_paper_evidence(
            chain=chain,
            subjects=subjects,
            producer_task_id=_id(999),
            producer_script_sha256=_sha(999),
        )


def _prediction_bytes() -> bytes:
    cohort = json.loads(
        Path(
            "artifacts/resilient_v2x/dair_v2_complemented/validation_cohort.json"
        ).read_text(encoding="utf-8")
    )
    sample_ids = cohort["sample_ids"]
    assert export._content_sha256(sample_ids) == export.SAMPLE_IDS_SHA256
    samples = [
        {
            "sample_id": sample_id,
            "ground_truth_boxes_lidar_bottom_center": (
                [0] * export.GROUND_TRUTH_COUNT if index == 0 else []
            ),
            "ground_truth_labels": (
                [0] * export.GROUND_TRUTH_COUNT if index == 0 else []
            ),
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    payload = {
        "schema_version": 1,
        "sample_count": export.SAMPLE_COUNT,
        "samples": samples,
    }
    document = {**payload, "content_sha256": export._content_sha256(payload)}
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _archive_fixture(tmp_path: Path, *, corrupt_checkpoint: bool = False):
    subject = "ffnet"
    checkpoint_sha = _sha(77)
    config_raw = b"model = dict(type='fixture')\n"
    config_sha = hashlib.sha256(config_raw).hexdigest()
    prediction_raw = _prediction_bytes()
    prediction_sha = hashlib.sha256(prediction_raw).hexdigest()
    prediction_content = json.loads(prediction_raw)["content_sha256"]
    runs = []
    metric_runs = []
    plan_runs = []
    for delay in export.DELAYS_MS:
        for condition in export.CONDITIONS:
            condition_id = export._condition_id(delay, condition)
            runs.append(
                {
                    "condition_id": condition_id,
                    "delay_ms": delay,
                    "condition": condition,
                    "agent_scope": export.AGENT_SCOPE,
                    "sample_count": export.SAMPLE_COUNT,
                    "ground_truth_count": export.GROUND_TRUTH_COUNT,
                    "unsupported_sample_count": 0,
                    "sample_ids_sha256": export.SAMPLE_IDS_SHA256,
                    "prediction_sha256": prediction_sha,
                    "prediction_content_sha256": prediction_content,
                    "metrics": {key: 50.0 for key in export.AP_METRIC_KEYS},
                }
            )
            metric_runs.append(
                {
                    "condition_id": condition_id,
                    "delay_ms": delay,
                    "condition": condition,
                    "sample_count": export.SAMPLE_COUNT,
                    "sample_ids_sha256": export.SAMPLE_IDS_SHA256,
                    "ground_truth_count": export.GROUND_TRUTH_COUNT,
                    "unsupported_sample_count": 0,
                    "prediction_sha256": prediction_sha,
                    "prediction_content_sha256": prediction_content,
                    "metrics": {
                        **export.COUNT_METRICS,
                        **{key: 50.0 for key in export.AP_METRIC_KEYS},
                    },
                }
            )
            plan_runs.append(
                {
                    "condition_id": condition_id,
                    "delay_ms": delay,
                    "condition": condition,
                    "resolved_config_sha256": config_sha,
                }
            )
    metrics = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": export.RUN_COUNT,
        "baseline": subject,
        "protocol_id": export.PROTOCOL_ID,
        "checkpoint": "/tmp/ffnet_epoch_50.pth",
        "checkpoint_sha256": checkpoint_sha,
        "manifest_content_sha256": export.MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": export.OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": export.SAMPLE_IDS_SHA256,
        "expected_sample_count": export.SAMPLE_COUNT,
        "expected_ground_truth_count": export.GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": 0,
        "runs": metric_runs,
    }
    plan = {"checkpoint_sha256": checkpoint_sha, "runs": plan_runs}
    path = tmp_path / "evidence.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("evaluation_plan.json", json.dumps(plan))
        archive.writestr("metrics.json", json.dumps(metrics))
        for run in runs:
            condition_id = run["condition_id"]
            archive.writestr(f"{condition_id}/resolved_config.py", config_raw)
            archive.writestr(f"{condition_id}/predictions.json", prediction_raw)
            bound = (
                _sha(78)
                if corrupt_checkpoint and condition_id == "delay_000_full"
                else checkpoint_sha
            )
            archive.writestr(f"{condition_id}/checkpoint.sha256", f"{bound}\n")
    return path, subject, checkpoint_sha, plan, metrics, runs


def test_prediction_archive_binds_all_prediction_and_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    path, subject, checkpoint, plan, metrics, runs = _archive_fixture(tmp_path)

    assert export._validate_prediction_archive(
        path,
        subject=subject,
        checkpoint_sha256=checkpoint,
        plan=plan,
        metrics=metrics,
        runs=runs,
    ) == export._sha256_path(path)


def test_prediction_archive_rejects_checkpoint_drift(tmp_path: Path) -> None:
    path, subject, checkpoint, plan, metrics, runs = _archive_fixture(
        tmp_path, corrupt_checkpoint=True
    )

    with pytest.raises(export.PaperEvidenceExportError, match="checkpoint binding"):
        export._validate_prediction_archive(
            path,
            subject=subject,
            checkpoint_sha256=checkpoint,
            plan=plan,
            metrics=metrics,
            runs=runs,
        )


def test_model_record_recomputes_local_checkpoint_bytes(tmp_path: Path) -> None:
    subject = "ffnet"
    directory = tmp_path / subject
    directory.mkdir()
    checkpoint = directory / "ffnet_epoch_50.pth"
    checkpoint.write_bytes(b"verified checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "subject": subject,
        "source_task_id": _id(1),
        "models": [
            {
                "role": "canonical_final",
                "filename": checkpoint.name,
                "model_id": _id(2),
                "sha256": digest,
                "size_bytes": checkpoint.stat().st_size,
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    record = export._model_record(tmp_path, subject, _id(1))
    assert record["sha256"] == digest

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(export.PaperEvidenceExportError, match="bytes drifted"):
        export._model_record(tmp_path, subject, _id(1))
