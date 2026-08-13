#!/usr/bin/env python3
"""Merge formal-main and candidate raw runs into one sealed final selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

try:
    from tools.resilient_v2x import (
        clearml_formal_candidate_evaluation_queue as candidate_queue,
    )
    from tools.resilient_v2x import (
        clearml_single_seed_sequential_candidates as sequential,
    )
    from tools.resilient_v2x import sota_gate
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import clearml_formal_candidate_evaluation_queue as candidate_queue
    import clearml_single_seed_sequential_candidates as sequential
    import sota_gate


DOCUMENT_TYPE = "resilient_v2x_final_single_seed_selection"
IDENTITY_DOCUMENT_TYPE = "resilient_v2x_sealed_selected_method_identity"
FORMAL_INPUT_DOCUMENT_TYPE = "resilient_v2x_final_selector_formal_inputs"
SELECTION_ARTIFACT = "final_single_seed_selection"
IDENTITY_ARTIFACT = "selected_method_identity"
FORMAL_MAIN = "resilient_v2x"
INITIAL_TRAINED_CANDIDATE_ORDER = tuple(
    spec.subject
    for spec in sota_gate.CANDIDATE_SPECS
    if spec.eligible_for_final_paper_selection
)
CANDIDATE_ONLY_ORDER = INITIAL_TRAINED_CANDIDATE_ORDER[1:]
SEQUENTIAL_CANDIDATE_ORDER = tuple(
    spec.experiment for spec in sequential.CANDIDATE_SPECS
)
FINAL_CANDIDATE_ORDER = (
    *INITIAL_TRAINED_CANDIDATE_ORDER,
    *SEQUENTIAL_CANDIDATE_ORDER,
)
FINAL_CANDIDATE_LABELS = {
    **{
        spec.subject: spec.label
        for spec in sota_gate.CANDIDATE_SPECS
        if spec.eligible_for_final_paper_selection
    },
    **{spec.experiment: spec.candidate_id for spec in sequential.CANDIDATE_SPECS},
}
EXPECTED_FORMAL_SUBJECTS = (*sota_gate.BASELINE_SUBJECTS, FORMAL_MAIN)
AP_METRIC_KEYS = candidate_queue.AP_METRIC_KEYS


class FinalSelectionError(RuntimeError):
    """Raised when final selection evidence is incomplete or inconsistent."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise FinalSelectionError(f"value is not canonical JSON: {error}") from error


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = value.get("seal_sha256")
    if type(observed) is not str or len(observed) != 64:
        raise FinalSelectionError(f"{context} seal is invalid")
    if _seal(value)["seal_sha256"] != observed:
        raise FinalSelectionError(f"{context} seal mismatch")
    return observed


def _task_id(value: object, *, context: str) -> str:
    result = str(value or "")
    if len(result) != 32 or any(char not in "0123456789abcdef" for char in result):
        raise FinalSelectionError(f"{context} is not a ClearML task/model ID")
    return result


def _sha256(value: object, *, context: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise FinalSelectionError(f"{context} is not a SHA-256")
    return result


def _positive_int(value: object, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise FinalSelectionError(f"{context} is not a positive integer")
    return value


def _finite_ap(value: object, *, context: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise FinalSelectionError(f"{context} is not finite")
    result = float(value)
    if result < 0.0 or result > 100.0:
        raise FinalSelectionError(f"{context} is outside [0, 100]")
    return result


def _normalize_runs(
    value: object, *, subject: str
) -> dict[tuple[int, str], dict[str, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, Mapping)):
        raise FinalSelectionError(f"{subject} raw runs are invalid")
    expected = [
        (delay, condition)
        for delay in sota_gate.DELAYS_MS
        for condition in sota_gate.CONDITIONS
    ]
    if len(value) != len(expected):
        raise FinalSelectionError(f"{subject} raw run count is not 12")
    result: dict[tuple[int, str], dict[str, float]] = {}
    for index, (raw, (delay, condition)) in enumerate(
        zip(value, expected, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise FinalSelectionError(f"{subject} run {index} is invalid")
        expected_condition_id = (
            f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        )
        if any(
            raw.get(key) != expected_value
            for key, expected_value in {
                "condition_id": expected_condition_id,
                "delay_ms": delay,
                "condition": condition,
            }.items()
        ):
            raise FinalSelectionError(f"{subject} run {index} order drifted")
        if raw.get("agent_scope") != "E+R":
            raise FinalSelectionError(f"{subject} run {index} scope drifted")
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(AP_METRIC_KEYS):
            raise FinalSelectionError(f"{subject} run {index} metric inventory drifted")
        result[(delay, condition)] = {
            key: _finite_ap(metrics.get(key), context=f"{subject} run {index} {key}")
            for key in AP_METRIC_KEYS
        }
    return result


def _validate_formal_inputs(
    value: Mapping[str, object],
) -> tuple[
    dict[str, dict[tuple[int, str], dict[str, float]]],
    dict[str, dict[str, object]],
    str,
]:
    seal = _require_seal(value, context="formal selector inputs")
    expected_root_keys = {
        "schema_version",
        "document_type",
        "protocol_evidence_fingerprint",
        "subject_order",
        "training_seed",
        "checkpoint_policy",
        "authority_bindings",
        "entries",
        "seal_sha256",
    }
    if set(value) != expected_root_keys:
        raise FinalSelectionError("formal selector input inventory drifted")
    expected = {
        "schema_version": 2,
        "document_type": FORMAL_INPUT_DOCUMENT_TYPE,
        "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
        "subject_order": list(EXPECTED_FORMAL_SUBJECTS),
        "training_seed": candidate_queue.TRAINING_SEED,
        "checkpoint_policy": candidate_queue.CHECKPOINT_POLICY,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise FinalSelectionError(f"formal selector inputs {key} drifted")
    authority = value.get("authority_bindings")
    authority_keys = {
        "producer_task_id",
        "producer_script_sha256",
        "producer_artifact_name",
        "training_controller_task_id",
        "training_provenance_task_id",
        "watcher_task_id",
        "watcher_artifact_name",
        "watcher_artifact_sha256",
        "watcher_plan_seal_sha256",
        "watcher_plan_content_sha256",
        "leaderboard_task_id",
        "leaderboard_artifact_name",
        "leaderboard_artifact_sha256",
        "leaderboard_seal_sha256",
        "leaderboard_content_sha256",
        "audit_task_id",
        "audit_artifact_name",
        "audit_artifact_sha256",
        "audit_seal_sha256",
        "audit_content_sha256",
    }
    if not isinstance(authority, Mapping) or set(authority) != authority_keys:
        raise FinalSelectionError("formal selector authority bindings drifted")
    for key, expected_value in {
        "producer_artifact_name": "final_selector_formal_inputs",
        "watcher_artifact_name": "formal_1337_evaluation_plan",
        "leaderboard_artifact_name": "formal_1337_leaderboard",
        "audit_artifact_name": "formal_1337_comparability_audit",
    }.items():
        if authority.get(key) != expected_value:
            raise FinalSelectionError(f"formal selector authority {key} drifted")
    authority_id_keys = (
        "producer_task_id",
        "training_controller_task_id",
        "training_provenance_task_id",
        "watcher_task_id",
        "leaderboard_task_id",
        "audit_task_id",
    )
    authority_ids = {
        key: _task_id(authority.get(key), context=f"formal authority {key}")
        for key in authority_id_keys
    }
    if len(set(authority_ids.values())) != len(authority_ids):
        raise FinalSelectionError("formal selector authority task IDs alias")
    for key in (
        "producer_script_sha256",
        "watcher_artifact_sha256",
        "watcher_plan_seal_sha256",
        "watcher_plan_content_sha256",
        "leaderboard_seal_sha256",
        "leaderboard_artifact_sha256",
        "leaderboard_content_sha256",
        "audit_seal_sha256",
        "audit_artifact_sha256",
        "audit_content_sha256",
    ):
        _sha256(authority.get(key), context=f"formal authority {key}")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_FORMAL_SUBJECTS):
        raise FinalSelectionError("formal selector input entries are incomplete")
    runs: dict[str, dict[tuple[int, str], dict[str, float]]] = {}
    bindings: dict[str, dict[str, object]] = {}
    occupied_ids = set(authority_ids.values())
    observed_subject_ids: set[str] = set()
    formal_identity_keys = {
        "label",
        "config_path",
        "config_sha256",
        "source_dataset_id",
        "source_revision_sha256",
        "source_archive_name",
        "source_archive_bytes",
        "source_archive_sha256",
        "training_dataset_id",
        "teacher_task_id",
        "teacher_model_id",
        "teacher_checkpoint_sha256",
        "training_task_id",
        "evaluation_task_id",
        "model_id",
        "model_name",
        "checkpoint_filename",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "training_script_sha256",
        "evaluation_script_sha256",
        "run_contract_artifact_sha256",
        "initialization_audit_artifact_sha256",
        "final_checkpoint_contract_artifact_sha256",
        "evaluation_run_contract_artifact_sha256",
        "evaluation_plan_artifact_sha256",
        "metrics_artifact_sha256",
        "prediction_evidence_artifact_sha256",
        "prediction_evidence_archive_sha256",
    }
    for raw, subject in zip(entries, EXPECTED_FORMAL_SUBJECTS, strict=True):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"subject", "identity_binding", "runs"}
            or raw.get("subject") != subject
        ):
            raise FinalSelectionError(f"formal selector entry {subject} drifted")
        binding = raw.get("identity_binding")
        if not isinstance(binding, Mapping) or set(binding) != formal_identity_keys:
            raise FinalSelectionError(f"formal selector {subject} lacks identity")
        if binding.get("label") != subject:
            raise FinalSelectionError(f"formal selector {subject} label drifted")
        if not str(binding.get("model_name") or "").strip():
            raise FinalSelectionError(f"formal selector {subject} model name drifted")
        source_archive_name = binding.get("source_archive_name")
        if (
            type(source_archive_name) is not str
            or not source_archive_name
            or PurePosixPath(source_archive_name).name != source_archive_name
        ):
            raise FinalSelectionError(
                f"formal selector {subject} source archive name drifted"
            )
        _positive_int(
            binding.get("source_archive_bytes"),
            context=f"formal {subject} source archive",
        )
        training_id = _task_id(
            binding.get("training_task_id"), context=f"formal {subject} training"
        )
        evaluation_id = _task_id(
            binding.get("evaluation_task_id"),
            context=f"formal {subject} evaluation",
        )
        model_id = _task_id(binding.get("model_id"), context=f"formal {subject} model")
        if (
            any(
                item in occupied_ids | observed_subject_ids
                for item in (training_id, evaluation_id, model_id)
            )
            or len({training_id, evaluation_id, model_id}) != 3
        ):
            raise FinalSelectionError(f"formal selector {subject} identity aliases")
        observed_subject_ids.update((training_id, evaluation_id, model_id))
        for key in (
            "source_revision_sha256",
            "source_archive_sha256",
            "training_script_sha256",
            "evaluation_script_sha256",
            "evaluation_run_contract_artifact_sha256",
        ):
            _sha256(binding.get(key), context=f"formal {subject} {key}")
        raw_runs = raw.get("runs")
        expected_run_keys = {
            "condition_id",
            "delay_ms",
            "condition",
            "agent_scope",
            "metrics",
        }
        if not isinstance(raw_runs, list) or any(
            not isinstance(run, Mapping)
            or (
                set(run) != expected_run_keys
                and set(run) != expected_run_keys - {"agent_scope"}
            )
            for run in raw_runs
        ):
            raise FinalSelectionError(
                f"formal selector {subject} run inventory drifted"
            )
        runs[subject] = _normalize_runs(raw.get("runs"), subject=subject)
        bindings[subject] = _validate_selected_method_binding(binding, subject=subject)
    return runs, bindings, seal


def _validate_candidate_manifest(
    value: Mapping[str, object],
) -> tuple[
    dict[str, dict[tuple[int, str], dict[str, float]]],
    dict[str, dict[str, object]],
    str,
    str,
]:
    seal = _require_seal(value, context="candidate evaluation manifest")
    expected = {
        "schema_version": 1,
        "document_type": ("resilient_v2x_formal_1337_candidate_evaluation_manifest"),
        "protocol_id": sota_gate.PROTOCOL_ID,
        "training_seed": candidate_queue.TRAINING_SEED,
        "training_dataset_id": candidate_queue.TRAINING_DATASET_ID,
        "checkpoint_policy": candidate_queue.CHECKPOINT_POLICY,
        "sample_count": sota_gate.SAMPLE_COUNT,
        "ground_truth_count": sota_gate.GROUND_TRUTH_COUNT,
        "unsupported_sample_count": sota_gate.UNSUPPORTED_SAMPLE_COUNT,
        "sample_ids_sha256": sota_gate.SAMPLE_IDS_SHA256,
        "delays_ms": list(sota_gate.DELAYS_MS),
        "conditions": list(sota_gate.CONDITIONS),
        "candidate_order": list(CANDIDATE_ONLY_ORDER),
        "candidate_count": len(CANDIDATE_ONLY_ORDER),
        "validated_candidate_count": len(CANDIDATE_ONLY_ORDER),
        "all_candidates_validated": True,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise FinalSelectionError(f"candidate manifest {key} drifted")
    controller_task_id = _task_id(
        value.get("controller_task_id"), context="candidate controller"
    )
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != len(CANDIDATE_ONLY_ORDER):
        raise FinalSelectionError("candidate manifest entries are incomplete")
    spec_by_subject = {spec.subject: spec for spec in candidate_queue.CANDIDATES}
    runs: dict[str, dict[tuple[int, str], dict[str, float]]] = {}
    bindings: dict[str, dict[str, object]] = {}
    for raw, subject in zip(entries, CANDIDATE_ONLY_ORDER, strict=True):
        if not isinstance(raw, Mapping):
            raise FinalSelectionError(f"candidate {subject} entry is invalid")
        spec = spec_by_subject[subject]
        if any(
            raw.get(key) != expected_value
            for key, expected_value in {
                "label": spec.label,
                "subject": subject,
                "training_task_id": spec.training_task_id,
                "template_task_id": spec.template_task_id,
                "training_status": "completed",
                "evaluation_status": "completed",
                "evidence_status": "validated",
            }.items()
        ):
            raise FinalSelectionError(f"candidate {subject} entry drifted")
        evaluation_task_id = _task_id(
            raw.get("evaluation_task_id"),
            context=f"candidate {subject} evaluation",
        )
        final = raw.get("training_final")
        evidence = raw.get("formal_evidence")
        if not isinstance(final, Mapping) or not isinstance(evidence, Mapping):
            raise FinalSelectionError(f"candidate {subject} evidence is missing")
        model_id = _task_id(final.get("model_id"), context=f"{subject} model")
        checkpoint_sha = _sha256(
            final.get("checkpoint_sha256"), context=f"{subject} checkpoint"
        )
        checkpoint_size = _positive_int(
            final.get("checkpoint_size_bytes"), context=f"{subject} checkpoint size"
        )
        for key in (
            "run_contract_artifact_sha256",
            "initialization_audit_artifact_sha256",
            "final_checkpoint_contract_artifact_sha256",
        ):
            _sha256(final.get(key), context=f"{subject} {key}")
        for key in (
            "metrics_artifact_sha256",
            "evaluation_plan_artifact_sha256",
            "prediction_evidence_artifact_sha256",
            "prediction_evidence_archive_sha256",
        ):
            _sha256(evidence.get(key), context=f"{subject} {key}")
        runs[subject] = _normalize_runs(evidence.get("runs"), subject=subject)
        bindings[subject] = {
            "label": spec.label,
            "config_path": spec.config_path,
            "config_sha256": spec.config_sha256,
            "source_dataset_id": spec.source_dataset_id,
            "source_archive_name": spec.source_archive_name,
            "source_archive_bytes": spec.source_archive_bytes,
            "source_archive_sha256": spec.source_archive_sha256,
            "training_dataset_id": candidate_queue.TRAINING_DATASET_ID,
            "teacher_task_id": candidate_queue.TEACHER_TASK_ID,
            "teacher_model_id": candidate_queue.TEACHER_MODEL_ID,
            "teacher_checkpoint_sha256": (candidate_queue.TEACHER_CHECKPOINT_SHA256),
            "training_task_id": spec.training_task_id,
            "evaluation_task_id": evaluation_task_id,
            "model_id": model_id,
            "checkpoint_filename": "epoch_50.pth",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_size_bytes": checkpoint_size,
            "run_contract_artifact_sha256": final["run_contract_artifact_sha256"],
            "initialization_audit_artifact_sha256": final[
                "initialization_audit_artifact_sha256"
            ],
            "final_checkpoint_contract_artifact_sha256": final[
                "final_checkpoint_contract_artifact_sha256"
            ],
            "evaluation_plan_artifact_sha256": evidence[
                "evaluation_plan_artifact_sha256"
            ],
            "metrics_artifact_sha256": evidence["metrics_artifact_sha256"],
            "prediction_evidence_artifact_sha256": evidence[
                "prediction_evidence_artifact_sha256"
            ],
            "prediction_evidence_archive_sha256": evidence[
                "prediction_evidence_archive_sha256"
            ],
        }
    return runs, bindings, seal, controller_task_id


def _summary(
    values: Mapping[tuple[int, str], Mapping[str, float]], dimension: str
) -> float:
    metric = sota_gate.LEADERSHIP_METRIC
    ordered = [
        float(values[(delay, condition)][metric])
        for delay in sota_gate.DELAYS_MS
        for condition in sota_gate.CONDITIONS
    ]
    if dimension == "full_0ms":
        return float(values[(0, "Full")][metric])
    if dimension == "mean_12":
        return math.fsum(ordered) / len(ordered)
    if dimension == "worst_12":
        return min(ordered)
    raise FinalSelectionError(f"unsupported aggregate {dimension!r}")


def _evaluate_trained_candidate(
    *,
    subject: str,
    candidate_runs: Mapping[tuple[int, str], Mapping[str, float]],
    baseline_runs: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
) -> dict[str, object]:
    if subject not in FINAL_CANDIDATE_ORDER:
        raise FinalSelectionError(f"unknown trained candidate {subject!r}")
    if tuple(baseline_runs) != sota_gate.BASELINE_SUBJECTS:
        raise FinalSelectionError("fixed five-baseline inventory drifted")
    fixed_order = FINAL_CANDIDATE_ORDER.index(subject)
    comparisons: list[dict[str, object]] = []
    for delay in sota_gate.DELAYS_MS:
        for condition in sota_gate.CONDITIONS:
            candidate_value = float(
                candidate_runs[(delay, condition)][sota_gate.LEADERSHIP_METRIC]
            )
            baseline_values = {
                baseline: float(
                    baseline_runs[baseline][(delay, condition)][
                        sota_gate.LEADERSHIP_METRIC
                    ]
                )
                for baseline in sota_gate.BASELINE_SUBJECTS
            }
            best = max(baseline_values.values())
            best_subjects = [
                baseline
                for baseline in sota_gate.BASELINE_SUBJECTS
                if baseline_values[baseline] == best
            ]
            margin = candidate_value - best
            comparisons.append(
                {
                    "condition_id": (
                        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
                    ),
                    "delay_ms": delay,
                    "condition": condition,
                    "candidate_value": candidate_value,
                    "best_baseline_value": best,
                    "best_baseline_subject": best_subjects[0],
                    "best_baseline_subjects": best_subjects,
                    "margin": margin,
                    "strictly_leads": margin > 0.0,
                }
            )
    aggregate: dict[str, dict[str, object]] = {}
    for dimension in ("full_0ms", "mean_12", "worst_12"):
        baseline_values = {
            baseline: _summary(baseline_runs[baseline], dimension)
            for baseline in sota_gate.BASELINE_SUBJECTS
        }
        best = max(baseline_values.values())
        best_subjects = [
            baseline
            for baseline in sota_gate.BASELINE_SUBJECTS
            if baseline_values[baseline] == best
        ]
        candidate_value = _summary(candidate_runs, dimension)
        margin = candidate_value - best
        full = dimension == "full_0ms"
        aggregate[dimension] = {
            "candidate_value": candidate_value,
            "best_baseline_value": best,
            "best_baseline_subject": best_subjects[0],
            "best_baseline_subjects": best_subjects,
            "margin": margin,
            "strictly_leads": margin > 0.0,
            "gate_comparison": (
                "greater_than_or_equal_to_best_minus_0.5"
                if full
                else "strictly_greater_than_best"
            ),
            "passes_gate": (
                margin >= -sota_gate.FULL_0MS_MAX_DEFICIT if full else margin > 0.0
            ),
        }
    won = sum(bool(item["strictly_leads"]) for item in comparisons)
    gate_passed = all(bool(item["passes_gate"]) for item in aggregate.values())
    evidence_class = (
        sota_gate.CANDIDATE_BY_SUBJECT[subject].evidence_class
        if subject in sota_gate.CANDIDATE_BY_SUBJECT
        else next(
            spec.evidence_class
            for spec in sequential.CANDIDATE_SPECS
            if spec.experiment == subject
        )
    )
    return {
        "subject": subject,
        "candidate_label": FINAL_CANDIDATE_LABELS[subject],
        "fixed_order_index": fixed_order,
        "evidence_class": evidence_class,
        "weights_retrained": True,
        "eligible_for_final_paper_selection": True,
        "conditions_won": won,
        "conditions_total": sota_gate.RUN_COUNT,
        "conditions_won_fraction": f"{won}/{sota_gate.RUN_COUNT}",
        "strictly_leads_all_conditions": won == sota_gate.RUN_COUNT,
        "per_condition_lead_required": False,
        "condition_comparisons": comparisons,
        "aggregate_comparisons": aggregate,
        "gate_passed": gate_passed,
        "ranking_values": {
            "worst_12_margin": aggregate["worst_12"]["margin"],
            "mean_12_margin": aggregate["mean_12"]["margin"],
            "full_0ms_margin": aggregate["full_0ms"]["margin"],
            "fixed_candidate_order": fixed_order,
        },
    }


def _ranking_sort_key(
    value: Mapping[str, object],
) -> tuple[int, float, float, float, int]:
    if type(value.get("gate_passed")) is not bool:
        raise FinalSelectionError("candidate gate status is invalid")
    ranking = value.get("ranking_values")
    if not isinstance(ranking, Mapping):
        raise FinalSelectionError("candidate ranking values are invalid")
    worst = float(ranking["worst_12_margin"])
    mean = float(ranking["mean_12_margin"])
    full = float(ranking["full_0ms_margin"])
    order = ranking.get("fixed_candidate_order")
    if (
        not all(math.isfinite(item) for item in (worst, mean, full))
        or type(order) is not int
        or not 0 <= order < len(FINAL_CANDIDATE_ORDER)
    ):
        raise FinalSelectionError("candidate ranking values drifted")
    return (
        0 if value["gate_passed"] is True else 1,
        -worst,
        -mean,
        -full,
        order,
    )


def _validate_initial_selection_binding(
    *,
    sequential_evidence: Mapping[str, object],
    initial_results: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], str]:
    try:
        state = sequential.validate_chain_evidence(sequential_evidence)
    except (RuntimeError, ValueError) as error:
        raise FinalSelectionError(f"sequential evidence is invalid: {error}") from error
    selection = sequential_evidence.get("initial_selection")
    if not isinstance(selection, Mapping):
        raise FinalSelectionError("sequential evidence has no initial selection")
    rows = selection.get("candidate_results")
    if not isinstance(rows, list):
        raise FinalSelectionError("initial selection rows are unavailable")
    by_subject = {
        str(row.get("subject")): row for row in rows if isinstance(row, Mapping)
    }
    for subject in INITIAL_TRAINED_CANDIDATE_ORDER:
        observed = by_subject.get(subject)
        if not isinstance(observed, Mapping):
            raise FinalSelectionError(f"initial selection lacks {subject}")
        expected = initial_results[subject]
        for key in (
            "candidate_label",
            "fixed_order_index",
            "evidence_class",
            "weights_retrained",
            "eligible_for_final_paper_selection",
            "condition_comparisons",
            "aggregate_comparisons",
            "gate_passed",
            "ranking_values",
        ):
            if _canonical_json(observed.get(key)) != _canonical_json(expected[key]):
                raise FinalSelectionError(
                    f"initial selection {subject} {key} is not raw-run bound"
                )
    return state, _require_seal(
        sequential_evidence, context="sequential candidate evidence"
    )


def _validate_sequential_receipts(
    *,
    sequential_evidence: Mapping[str, object],
    initial_results: Mapping[str, Mapping[str, object]],
    baseline_runs: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
) -> tuple[
    dict[str, dict[tuple[int, str], dict[str, float]]],
    dict[str, dict[str, object]],
    dict[str, object],
    str,
]:
    state, seal = _validate_initial_selection_binding(
        sequential_evidence=sequential_evidence,
        initial_results=initial_results,
    )
    raw_receipts = sequential_evidence.get("sequential_gate_receipts")
    if not isinstance(raw_receipts, list):
        raise FinalSelectionError("sequential gate receipts are unavailable")
    runs: dict[str, dict[tuple[int, str], dict[str, float]]] = {}
    bindings: dict[str, dict[str, object]] = {}
    for raw in raw_receipts:
        if not isinstance(raw, Mapping):
            raise FinalSelectionError("sequential gate receipt is invalid")
        candidate_id = str(raw["candidate_id"])
        spec = sequential.CANDIDATE_BY_ID[candidate_id]
        subject = spec.experiment
        evaluation = raw["evaluation"]
        producer = raw["producer"]
        checkpoint = raw["checkpoint"]
        baseline_reference = raw["baseline_reference"]
        if not all(
            isinstance(item, Mapping)
            for item in (evaluation, producer, checkpoint, baseline_reference)
        ):
            raise FinalSelectionError(f"{candidate_id} receipt sections are invalid")
        candidate_runs = _normalize_runs(evaluation["runs"], subject=subject)
        receipt_baselines = baseline_reference["runs_by_subject"]
        if not isinstance(receipt_baselines, Mapping):
            raise FinalSelectionError(f"{candidate_id} baseline runs are invalid")
        for baseline in sota_gate.BASELINE_SUBJECTS:
            observed = _normalize_runs(
                receipt_baselines[baseline], subject=f"{candidate_id}/{baseline}"
            )
            if observed != baseline_runs[baseline]:
                raise FinalSelectionError(
                    f"{candidate_id} baseline {baseline} differs from formal raw runs"
                )
        training_protocol = raw["training_protocol"]
        if not isinstance(training_protocol, Mapping):
            raise FinalSelectionError(f"{candidate_id} training protocol is invalid")
        runs[subject] = candidate_runs
        bindings[subject] = {
            "candidate_id": candidate_id,
            "candidate_label": candidate_id,
            "config_path": spec.config,
            "config_sha256": spec.config_sha256,
            "source_dataset_id": producer["source_dataset_id"],
            "source_revision_sha256": producer["sealed_source_tree_sha256"],
            "training_dataset_id": training_protocol["training_dataset_id"],
            "teacher_task_id": training_protocol["teacher_task_id"],
            "teacher_model_id": training_protocol["teacher_model_id"],
            "teacher_checkpoint_sha256": training_protocol["teacher_checkpoint_sha256"],
            "training_task_id": producer["training_task_id"],
            "evaluation_task_id": evaluation["evaluation_task_id"],
            "model_id": checkpoint["model_id"],
            "checkpoint_filename": checkpoint["filename"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size_bytes": checkpoint["size_bytes"],
            "run_contract_artifact_sha256": producer["run_contract_sha256"],
            "initialization_audit_artifact_sha256": producer["teacher_audit_sha256"],
            "final_checkpoint_contract_artifact_sha256": checkpoint[
                "contract_artifact_sha256"
            ],
            "evaluation_run_contract_artifact_sha256": evaluation[
                "run_contract_sha256"
            ],
            "evaluation_plan_artifact_sha256": evaluation["evaluation_plan_sha256"],
            "metrics_artifact_sha256": evaluation["metrics_artifact_sha256"],
            "prediction_evidence_artifact_sha256": evaluation[
                "prediction_evidence_sha256"
            ],
            "prediction_evidence_archive_sha256": evaluation[
                "prediction_evidence_archive_sha256"
            ],
            "evaluator_evidence_sha256": evaluation["evaluator_evidence_sha256"],
            "source_gate_artifact_sha256": evaluation["gate_artifact_sha256"],
            "sequential_gate_receipt_seal_sha256": raw["seal_sha256"],
        }
    return runs, bindings, state, seal


def _validate_selected_method_binding(
    value: Mapping[str, object], *, subject: str
) -> dict[str, object]:
    required_ids = (
        "source_dataset_id",
        "training_dataset_id",
        "teacher_task_id",
        "teacher_model_id",
        "training_task_id",
        "evaluation_task_id",
        "model_id",
    )
    for key in required_ids:
        _task_id(value.get(key), context=f"selected {subject} {key}")
    required_hashes = (
        "config_sha256",
        "teacher_checkpoint_sha256",
        "checkpoint_sha256",
        "run_contract_artifact_sha256",
        "initialization_audit_artifact_sha256",
        "final_checkpoint_contract_artifact_sha256",
        "evaluation_plan_artifact_sha256",
        "metrics_artifact_sha256",
        "prediction_evidence_artifact_sha256",
        "prediction_evidence_archive_sha256",
    )
    for key in required_hashes:
        _sha256(value.get(key), context=f"selected {subject} {key}")
    source_hash_fields = (
        "source_revision_sha256",
        "source_archive_sha256",
    )
    present_source_hashes = [
        key for key in source_hash_fields if value.get(key) is not None
    ]
    if not present_source_hashes:
        raise FinalSelectionError(
            f"selected {subject} has no source revision/archive SHA-256"
        )
    for key in present_source_hashes:
        _sha256(value.get(key), context=f"selected {subject} {key}")
    config_path = value.get("config_path")
    parsed_config = PurePosixPath(config_path) if type(config_path) is str else None
    if (
        parsed_config is None
        or parsed_config.is_absolute()
        or tuple(parsed_config.parts)[:1] != ("configs",)
        or "." in parsed_config.parts
        or ".." in parsed_config.parts
    ):
        raise FinalSelectionError(f"selected {subject} config path is invalid")
    if value.get("checkpoint_filename") != "epoch_50.pth":
        raise FinalSelectionError(f"selected {subject} checkpoint is not epoch_50")
    _positive_int(
        value.get("checkpoint_size_bytes"), context=f"selected {subject} checkpoint"
    )
    fixed_contract = {
        "training_dataset_id": candidate_queue.TRAINING_DATASET_ID,
        "teacher_task_id": candidate_queue.TEACHER_TASK_ID,
        "teacher_model_id": candidate_queue.TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": candidate_queue.TEACHER_CHECKPOINT_SHA256,
    }
    for key, expected in fixed_contract.items():
        if value.get(key) != expected:
            raise FinalSelectionError(
                f"selected {subject} {key} differs from the fixed contract"
            )
    return dict(value)


def build_final_selection(
    *,
    formal_inputs: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    sequential_evidence: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    formal_runs, formal_bindings, formal_seal = _validate_formal_inputs(formal_inputs)
    candidate_runs, candidate_bindings, candidate_seal, controller_id = (
        _validate_candidate_manifest(candidate_manifest)
    )
    all_initial_runs = {**formal_runs, **candidate_runs}
    baseline_runs = {
        subject: all_initial_runs[subject] for subject in sota_gate.BASELINE_SUBJECTS
    }
    initial_results = {
        subject: _evaluate_trained_candidate(
            subject=subject,
            candidate_runs=all_initial_runs[subject],
            baseline_runs=baseline_runs,
        )
        for subject in INITIAL_TRAINED_CANDIDATE_ORDER
    }

    sequential_runs: dict[str, dict[tuple[int, str], dict[str, float]]] = {}
    sequential_bindings: dict[str, dict[str, object]] = {}
    sequential_state: dict[str, object] | None = None
    sequential_seal: str | None = None
    if sequential_evidence is not None:
        (
            sequential_runs,
            sequential_bindings,
            sequential_state,
            sequential_seal,
        ) = _validate_sequential_receipts(
            sequential_evidence=sequential_evidence,
            initial_results=initial_results,
            baseline_runs=baseline_runs,
        )
        expected_initial_state = (
            "pass"
            if any(row["gate_passed"] is True for row in initial_results.values())
            else "failed"
        )
        if sequential_state.get("initial_state") != expected_initial_state:
            raise FinalSelectionError(
                "sequential initial state differs from recomputed raw evidence"
            )

    evaluated_runs = {
        subject: all_initial_runs[subject]
        for subject in INITIAL_TRAINED_CANDIDATE_ORDER
    }
    evaluated_runs.update(sequential_runs)
    raw_results = [
        _evaluate_trained_candidate(
            subject=subject,
            candidate_runs=evaluated_runs[subject],
            baseline_runs=baseline_runs,
        )
        for subject in FINAL_CANDIDATE_ORDER
        if subject in evaluated_runs
    ]
    ranking = sorted(raw_results, key=_ranking_sort_key)
    passing = [
        row
        for row in ranking
        if row["gate_passed"] is True
        and row["eligible_for_final_paper_selection"] is True
    ]
    selected = str(passing[0]["subject"]) if passing else None
    rank_by_subject = {
        str(row["subject"]): index for index, row in enumerate(ranking, start=1)
    }
    result_by_subject = {str(row["subject"]): row for row in raw_results}
    fingerprint = sota_gate.evidence_fingerprint()
    fingerprint_sha = str(fingerprint["fingerprint_sha256"])
    enriched_results: list[dict[str, object]] = []
    pending_subjects: list[str] = []
    stopped_subjects: list[str] = []
    for subject in FINAL_CANDIDATE_ORDER:
        row = result_by_subject.get(subject)
        if row is not None:
            enriched_results.append(
                {
                    **row,
                    "evidence_status": "validated",
                    "evidence_fingerprint_sha256": fingerprint_sha,
                    "performance_rank": rank_by_subject[subject],
                }
            )
            continue
        evidence_status = (
            "not_created_after_earlier_pass" if selected is not None else "pending"
        )
        if selected is not None:
            stopped_subjects.append(subject)
        else:
            pending_subjects.append(subject)
        spec = next(
            item for item in sequential.CANDIDATE_SPECS if item.experiment == subject
        )
        enriched_results.append(
            {
                "subject": subject,
                "candidate_label": spec.candidate_id,
                "fixed_order_index": FINAL_CANDIDATE_ORDER.index(subject),
                "evidence_class": spec.evidence_class,
                "weights_retrained": True,
                "eligible_for_final_paper_selection": True,
                "evidence_status": evidence_status,
                "gate_passed": None,
                "ranking_values": None,
                "performance_rank": None,
            }
        )

    receipts = sequential_state.get("receipts") if sequential_state is not None else []
    if not isinstance(receipts, list):
        raise FinalSelectionError("validated sequential receipt state is invalid")
    exhausted = (
        sequential_state is not None
        and len(receipts) == len(SEQUENTIAL_CANDIDATE_ORDER)
        and selected is None
    )
    if selected is not None:
        status = "selected"
    elif exhausted:
        status = "predefined_candidates_exhausted_without_lead"
    elif sequential_evidence is None:
        status = "awaiting_sequential_evidence"
    else:
        status = "awaiting_sequential_candidate_evidence"

    registry: list[dict[str, object]] = []
    for subject in FINAL_CANDIDATE_ORDER:
        if subject in sota_gate.CANDIDATE_BY_SUBJECT:
            spec = sota_gate.CANDIDATE_BY_SUBJECT[subject]
            candidate_id: str | None = None if subject == FORMAL_MAIN else spec.label
            evidence_class = spec.evidence_class
        else:
            sequential_spec = next(
                item
                for item in sequential.CANDIDATE_SPECS
                if item.experiment == subject
            )
            candidate_id = sequential_spec.candidate_id
            evidence_class = sequential_spec.evidence_class
        registry.append(
            {
                "subject": subject,
                "candidate_label": FINAL_CANDIDATE_LABELS[subject],
                "candidate_id": candidate_id,
                "fixed_order_index": FINAL_CANDIDATE_ORDER.index(subject),
                "evidence_class": evidence_class,
                "weights_retrained": True,
                "eligible_for_final_paper_selection": True,
            }
        )

    selection = _seal(
        {
            "schema_version": 2,
            "document_type": DOCUMENT_TYPE,
            "status": status,
            "selection_is_final": selected is not None or exhausted,
            "selection_claim": (
                "controlled_leader_selected"
                if selected is not None
                else (
                    "controlled_leadership_not_reached"
                    if exhausted
                    else "selection_pending"
                )
            ),
            "claim_scope": "single_seed_same_protocol_DAIR_controlled_leadership",
            "training_seed": candidate_queue.TRAINING_SEED,
            "protocol_evidence_fingerprint": fingerprint,
            "checkpoint_policy": candidate_queue.CHECKPOINT_POLICY,
            "baseline_subjects": list(sota_gate.BASELINE_SUBJECTS),
            "baseline_count": len(sota_gate.BASELINE_SUBJECTS),
            "candidate_subjects": list(FINAL_CANDIDATE_ORDER),
            "candidate_registry": registry,
            "candidate_count": len(FINAL_CANDIDATE_ORDER),
            "evaluated_candidate_count": len(raw_results),
            "pending_candidate_count": len(pending_subjects),
            "pending_candidate_subjects": pending_subjects,
            "not_created_after_stop_count": len(stopped_subjects),
            "not_created_after_stop_subjects": stopped_subjects,
            "gate": {
                "full_0ms": "candidate >= best fixed baseline - 0.5",
                "mean_12": "candidate > best fixed baseline",
                "worst_12": "candidate > best fixed baseline",
                "per_condition_lead_required": False,
            },
            "ranking_key": [
                "gate_passed",
                "worst_12_margin",
                "mean_12_margin",
                "full_0ms_margin",
                "fixed_candidate_order",
            ],
            "performance_ranked_candidates": [str(row["subject"]) for row in ranking],
            "selected_candidate": selected,
            "selected_method_identity_artifact": IDENTITY_ARTIFACT,
            "candidate_results": enriched_results,
            "input_bindings": {
                "formal_inputs_seal_sha256": formal_seal,
                "candidate_manifest_task_id": controller_id,
                "candidate_manifest_seal_sha256": candidate_seal,
                "sequential_evidence_seal_sha256": sequential_seal,
                "sequential_gate_receipt_count": len(receipts),
            },
        }
    )
    identities = {
        **formal_bindings,
        **candidate_bindings,
        **sequential_bindings,
    }
    validated_identities = {
        subject: _validate_selected_method_binding(identities[subject], subject=subject)
        for subject in evaluated_runs
    }
    selected_binding = validated_identities[selected] if selected is not None else None
    selected_gate_result = (
        next(row for row in enriched_results if row["subject"] == selected)
        if selected is not None
        else None
    )
    identity = _seal(
        {
            "schema_version": 2,
            "document_type": IDENTITY_DOCUMENT_TYPE,
            "status": "sealed" if selected else "no_selected_method",
            "selected_subject": selected,
            "selected_candidate_label": (
                FINAL_CANDIDATE_LABELS[selected] if selected is not None else None
            ),
            "selected_fixed_order_index": (
                FINAL_CANDIDATE_ORDER.index(selected) if selected is not None else None
            ),
            "selection_artifact": SELECTION_ARTIFACT,
            "selection_seal_sha256": selection["seal_sha256"],
            "training_seed": candidate_queue.TRAINING_SEED,
            "protocol_evidence_fingerprint": fingerprint,
            "checkpoint_policy": candidate_queue.CHECKPOINT_POLICY,
            "method_binding": selected_binding,
            "evidence_bindings": {
                "formal_inputs_seal_sha256": formal_seal,
                "candidate_manifest_task_id": controller_id,
                "candidate_manifest_seal_sha256": candidate_seal,
                "sequential_evidence_seal_sha256": sequential_seal,
                "sequential_gate_receipt_seal_sha256": (
                    selected_binding.get("sequential_gate_receipt_seal_sha256")
                    if selected_binding is not None
                    else None
                ),
            },
            "gate_artifact": {
                "artifact_name": SELECTION_ARTIFACT,
                "selection_seal_sha256": selection["seal_sha256"],
                "upstream_candidate_gate_artifact_sha256": (
                    selected_binding.get("source_gate_artifact_sha256")
                    if selected_binding is not None
                    else None
                ),
            },
            "gate_result": selected_gate_result,
            "gate_result_content_sha256": (
                _content_sha256(selected_gate_result)
                if selected_gate_result is not None
                else None
            ),
        }
    )
    return selection, identity


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalSelectionError(f"cannot read {path} as JSON") from error
    if not isinstance(value, dict):
        raise FinalSelectionError(f"{path} is not a JSON object")
    return value


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-inputs", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--sequential-evidence", type=Path)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.selection_output.resolve() == args.identity_output.resolve():
        raise FinalSelectionError("selection and identity outputs must differ")
    selection, identity = build_final_selection(
        formal_inputs=_read_json(args.formal_inputs),
        candidate_manifest=_read_json(args.candidate_manifest),
        sequential_evidence=(
            _read_json(args.sequential_evidence)
            if args.sequential_evidence is not None
            else None
        ),
    )
    _write_new(args.selection_output, selection)
    _write_new(args.identity_output, identity)
    print(
        json.dumps(
            {
                "selection_output": str(args.selection_output),
                "identity_output": str(args.identity_output),
                "selected_candidate": selection["selected_candidate"],
                "selection_seal_sha256": selection["seal_sha256"],
                "identity_seal_sha256": identity["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
