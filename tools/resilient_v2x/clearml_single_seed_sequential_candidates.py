#!/usr/bin/env python3
"""Plan and explicitly deploy the fail-closed single-seed P1/P3--P7 queue.

The command is dry-run by default.  A deployment phase runs only when the
sealed evidence selects exactly one next candidate and the caller supplies the
explicit execution flag, the global confirmation token, the plan seal, and
(for remote phases) the candidate-and-phase-specific remote-write token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

try:
    from tools.resilient_v2x import sota_gate
except ModuleNotFoundError as error:  # pragma: no cover - standalone execution
    if error.name != "tools":
        raise
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.resilient_v2x import sota_gate


ROOT = Path(__file__).resolve().parents[2]
CHAIN_DOCUMENT_TYPE = "resilient_v2x_single_seed_sequential_candidate_evidence"
GATE_DOCUMENT_TYPE = "resilient_v2x_single_seed_formal_candidate_gate"
PLAN_DOCUMENT_TYPE = "resilient_v2x_single_seed_sequential_candidate_plan"
EXECUTE_TOKEN = "EXECUTE_EXACT_SINGLE_SEED_SEQUENTIAL_CANDIDATE"
ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

INITIAL_TRAINED_SUBJECTS = (
    "resilient_v2x",
    "support_residual_linear",
    "no_reliability_linear",
    "support_residual_no_reliability",
    "support_residual_no_reliability_linear",
    "support_residual_no_reliability_linear_bbox25",
)
INITIAL_FORMAL_BLOCKERS = (
    ("E1", "support_residual_linear"),
    ("E2", "no_reliability_linear"),
    ("E3", "support_residual_no_reliability"),
    ("P0", "support_residual_no_reliability_linear"),
    ("P2", "support_residual_no_reliability_linear_bbox25"),
)
ALREADY_RUNNING_UNAFFECTED = ("E2", "E3", "P0", "P2")

TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
TEACHER_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
TEACHER_MODEL_ID = "d962f6bae8474260b54e170a7a5f0418"
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
QUEUE_IDS = {
    "GPU4-A100": "9350f33af13a448da8339eb7bea52fdf",
    "GPU4-V100": "3925e906ce484620a941e6ccedc4bdbd",
    "GPU4-5090": "5a84454c072349069e7b61af38637c6d",
}
TRAINING_QUEUE = "GPU4-A100"
FORMAL_EVALUATION_QUEUE = "GPU4-A100"
AP_METRIC_KEYS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)

TRAINING_PROTOCOL = {
    "dataset": "DAIR training dataset sealed by the formal run contract",
    "training_dataset_id": TRAINING_DATASET_ID,
    "teacher": "unified frozen teacher sealed by the formal run contract",
    "teacher_task_id": TEACHER_TASK_ID,
    "teacher_model_id": TEACHER_MODEL_ID,
    "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
    "training_seed": 20_250_218,
    "precision": "FP32",
    "gpu_count": 4,
    "batch_size_per_gpu": 2,
    "global_batch_size": 8,
    "max_epochs": 50,
    "val_interval": 10,
}
GATE_POLICY = {
    "metric": sota_gate.LEADERSHIP_METRIC,
    "full_0ms": "candidate >= best controlled baseline - 0.5",
    "mean_12": "candidate > best controlled baseline",
    "worst_12": "candidate > best controlled baseline",
    "per_condition_lead_required": False,
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    experiment: str
    config: str
    identity: str
    evidence_class: str
    parent_candidate_id: str
    ptf_mode: str
    reliability_descriptor: str
    student_bbox_loss_weight: float
    upload_token: str
    template_token: str
    launch_token: str
    config_sha256: str
    config_base: str
    training_queue: str = TRAINING_QUEUE
    training_queue_id: str = QUEUE_IDS[TRAINING_QUEUE]
    formal_evaluation_queue: str = FORMAL_EVALUATION_QUEUE
    formal_evaluation_queue_id: str = QUEUE_IDS[FORMAL_EVALUATION_QUEUE]

    @property
    def output_dir(self) -> Path:
        suffix = self.candidate_id.lower()
        if self.candidate_id == "P1":
            return ROOT / "artifacts/resilient_v2x/sota_round2_p1"
        return ROOT / f"artifacts/resilient_v2x/sota_single_seed_{suffix}"


CANDIDATE_SPECS = (
    CandidateSpec(
        "P1",
        "reliability_gated_residual",
        "configs/resilient_v2x/improvements/reliability_gated_residual.py",
        "dair_improvement_reliability_gated_residual_trained_50e",
        "formal_p1_trained_candidate",
        "P0",
        "linear",
        "support-only router descriptor; raw reliability used only by residual",
        2.0,
        "UPLOAD_EXACT_ROUND2_P1_SOURCE",
        "CREATE_EXACT_ROUND2_P1_TEMPLATE",
        "LAUNCH_EXACT_ROUND2_P1_A100",
        "76f7164326c7de8f6c5e9148887fb02772c141e981ea42e579ff9bea010c0ef1",
        "configs/resilient_v2x/improvements/support_residual_no_reliability_linear.py",
    ),
    CandidateSpec(
        "P3",
        "reliability_gated_residual_bbox25",
        ("configs/resilient_v2x/improvements/reliability_gated_residual_bbox25.py"),
        "dair_improvement_reliability_gated_residual_bbox25",
        "formal_trained_50e_candidate",
        "P1",
        "linear",
        "support-only router descriptor; raw reliability used only by residual",
        2.5,
        "UPLOAD_EXACT_SINGLE_SEED_P3_SOURCE",
        "CREATE_EXACT_SINGLE_SEED_P3_TEMPLATE",
        "LAUNCH_EXACT_SINGLE_SEED_P3_A100",
        "66941f499a279a77aafdf08de2efbc38ef4a4745b9cfe50d5e2d2b1f21252bd9",
        "configs/resilient_v2x/improvements/reliability_gated_residual.py",
    ),
    CandidateSpec(
        "P4",
        "reliability_gated_residual_full_nonlinear",
        (
            "configs/resilient_v2x/improvements/"
            "reliability_gated_residual_full_nonlinear.py"
        ),
        "dair_improvement_reliability_gated_residual_full_nonlinear",
        "formal_trained_50e_candidate",
        "P3",
        "nonlinear",
        "full raw-reliability router descriptor",
        2.0,
        "UPLOAD_EXACT_SINGLE_SEED_P4_SOURCE",
        "CREATE_EXACT_SINGLE_SEED_P4_TEMPLATE",
        "LAUNCH_EXACT_SINGLE_SEED_P4_A100",
        "65a4013d72a951607c0761735d4a53a56c11a0c14b9d9815d0b8be358e02cf6a",
        "configs/resilient_v2x/improvements/support_residual.py",
    ),
    CandidateSpec(
        "P5",
        "reliability_gated_residual_full_nonlinear_bbox25",
        (
            "configs/resilient_v2x/improvements/"
            "reliability_gated_residual_full_nonlinear_bbox25.py"
        ),
        "dair_improvement_reliability_gated_residual_full_nonlinear_bbox25",
        "formal_trained_50e_candidate",
        "P4",
        "nonlinear",
        "full raw-reliability router descriptor",
        2.5,
        "UPLOAD_EXACT_SINGLE_SEED_P5_SOURCE",
        "CREATE_EXACT_SINGLE_SEED_P5_TEMPLATE",
        "LAUNCH_EXACT_SINGLE_SEED_P5_A100",
        "d5e6746723069c86370015c6a4cb6d1244bfcdcd22254973d2135ff9030b583f",
        (
            "configs/resilient_v2x/improvements/"
            "reliability_gated_residual_full_nonlinear.py"
        ),
    ),
    CandidateSpec(
        "P6",
        "reliability_gated_residual_full_linear",
        (
            "configs/resilient_v2x/improvements/"
            "reliability_gated_residual_full_linear.py"
        ),
        "dair_improvement_reliability_gated_residual_full_linear",
        "formal_trained_50e_candidate",
        "P5",
        "linear",
        "full raw-reliability router descriptor",
        2.0,
        "UPLOAD_EXACT_SINGLE_SEED_P6_SOURCE",
        "CREATE_EXACT_SINGLE_SEED_P6_TEMPLATE",
        "LAUNCH_EXACT_SINGLE_SEED_P6_A100",
        "9d5a798c6cd4e2495af74191139a63783cf7be156b6a9565280e8be2cfdeba68",
        (
            "configs/resilient_v2x/improvements/"
            "reliability_gated_residual_full_nonlinear.py"
        ),
    ),
    CandidateSpec(
        "P7",
        "reliability_gated_residual_full_linear_bbox25",
        (
            "configs/resilient_v2x/improvements/"
            "reliability_gated_residual_full_linear_bbox25.py"
        ),
        "dair_improvement_reliability_gated_residual_full_linear_bbox25",
        "formal_trained_50e_candidate",
        "P6",
        "linear",
        "full raw-reliability router descriptor",
        2.5,
        "UPLOAD_EXACT_SINGLE_SEED_P7_SOURCE",
        "CREATE_EXACT_SINGLE_SEED_P7_TEMPLATE",
        "LAUNCH_EXACT_SINGLE_SEED_P7_A100",
        "9e54224a39d7e15e7e6e682feffd605a9a002360910892cf8fed75e6a7ce9b76",
        (
            "configs/resilient_v2x/improvements/"
            "reliability_gated_residual_full_linear.py"
        ),
    ),
)
CANDIDATE_ORDER = tuple(spec.candidate_id for spec in CANDIDATE_SPECS)
CANDIDATE_BY_ID = {spec.candidate_id: spec for spec in CANDIDATE_SPECS}


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
        raise ValueError(f"value is outside canonical JSON: {error}") from error


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _require_seal(value: Mapping[str, object], *, context: str) -> None:
    observed = value.get("seal_sha256")
    expected = _sealed(value)["seal_sha256"]
    if observed != expected:
        raise ValueError(f"{context} seal mismatch")


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, context: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{context} keys mismatch; missing={missing!r}, extra={extra!r}"
        )


def _require_id(value: object, *, context: str) -> str:
    if type(value) is not str or ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase 32-character ClearML ID")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _require_ap(value: object, *, context: str) -> float:
    result = _finite(value, context=context)
    if not 0.0 <= result <= 100.0:
        raise ValueError(f"{context} must be in [0, 100]")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_candidate_registry() -> None:
    if tuple(CANDIDATE_BY_ID) != CANDIDATE_ORDER:
        raise RuntimeError("sequential candidate registry order drifted")
    seen_configs: set[str] = set()
    for index, spec in enumerate(CANDIDATE_SPECS):
        if spec.config in seen_configs:
            raise RuntimeError(f"{spec.candidate_id} config aliases a predecessor")
        seen_configs.add(spec.config)
        path = ROOT / spec.config
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{spec.candidate_id} config is not a regular file")
        if _sha256_file(path) != spec.config_sha256:
            raise RuntimeError(f"{spec.candidate_id} config SHA-256 drifted")
        base = ROOT / spec.config_base
        if not base.is_file() or base.is_symlink():
            raise RuntimeError(f"{spec.candidate_id} config base is invalid")
        if spec.training_queue != TRAINING_QUEUE or (
            spec.training_queue_id != QUEUE_IDS[TRAINING_QUEUE]
        ):
            raise RuntimeError(f"{spec.candidate_id} training queue drifted")
        if spec.formal_evaluation_queue != FORMAL_EVALUATION_QUEUE or (
            spec.formal_evaluation_queue_id != QUEUE_IDS[FORMAL_EVALUATION_QUEUE]
        ):
            raise RuntimeError(f"{spec.candidate_id} evaluation queue drifted")
        if index == 0:
            if spec.parent_candidate_id != "P0":
                raise RuntimeError("P1 must depend on P0")
        elif spec.parent_candidate_id != CANDIDATE_ORDER[index - 1]:
            raise RuntimeError(
                f"{spec.candidate_id} must depend on the immediately prior gate"
            )


def _condition_id(delay: int, condition: str) -> str:
    return f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"


def _normalize_formal_runs(
    value: Sequence[Mapping[str, object]], *, context: str
) -> list[dict[str, object]]:
    if isinstance(value, (str, bytes)) or len(value) != sota_gate.RUN_COUNT:
        raise ValueError(f"{context} must contain exactly 12 formal runs")
    normalized: list[dict[str, object]] = []
    expected_keys = {
        "condition_id",
        "delay_ms",
        "condition",
        "agent_scope",
        "prediction_sha256",
        "prediction_content_sha256",
        "metrics",
    }
    for index, (run, (delay, condition)) in enumerate(
        zip(
            value,
            (
                (delay, condition)
                for delay in sota_gate.DELAYS_MS
                for condition in sota_gate.CONDITIONS
            ),
            strict=True,
        )
    ):
        if not isinstance(run, Mapping):
            raise ValueError(f"{context} run {index} is not an object")
        _require_exact_keys(run, expected_keys, context=f"{context} run {index}")
        if (
            run.get("condition_id") != _condition_id(delay, condition)
            or run.get("delay_ms") != delay
            or run.get("condition") != condition
            or run.get("agent_scope") != "E+R"
        ):
            raise ValueError(f"{context} run {index} identity drifted")
        prediction_sha = _require_sha256(
            run.get("prediction_sha256"), context=f"{context} prediction SHA"
        )
        prediction_content_sha = _require_sha256(
            run.get("prediction_content_sha256"),
            context=f"{context} prediction content SHA",
        )
        metrics = run.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{context} run {index} metrics are invalid")
        _require_exact_keys(
            metrics, set(AP_METRIC_KEYS), context=f"{context} run {index} metrics"
        )
        normalized.append(
            {
                "condition_id": _condition_id(delay, condition),
                "delay_ms": delay,
                "condition": condition,
                "agent_scope": "E+R",
                "prediction_sha256": prediction_sha,
                "prediction_content_sha256": prediction_content_sha,
                "metrics": {
                    metric: _require_ap(
                        metrics[metric], context=f"{context} run {index} {metric}"
                    )
                    for metric in AP_METRIC_KEYS
                },
            }
        )
    return normalized


def _leadership_values(
    runs: Sequence[Mapping[str, object]], *, context: str
) -> dict[tuple[int, str], float]:
    normalized = _normalize_formal_runs(runs, context=context)
    return {
        (int(run["delay_ms"]), str(run["condition"])): float(
            run["metrics"][sota_gate.LEADERSHIP_METRIC]  # type: ignore[index]
        )
        for run in normalized
    }


def _aggregate(values: Mapping[tuple[int, str], float], dimension: str) -> float:
    ordered = [
        values[(delay, condition)]
        for delay in sota_gate.DELAYS_MS
        for condition in sota_gate.CONDITIONS
    ]
    if dimension == "full_0ms":
        return values[(0, "Full")]
    if dimension == "mean_12":
        return math.fsum(ordered) / len(ordered)
    if dimension == "worst_12":
        return min(ordered)
    raise ValueError(f"unknown gate aggregate {dimension!r}")


def _gate_from_formal_runs(
    candidate_runs: Sequence[Mapping[str, object]],
    baseline_runs_by_subject: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    if tuple(baseline_runs_by_subject) != sota_gate.BASELINE_SUBJECTS:
        raise ValueError("formal baseline order or membership drifted")
    candidate = _leadership_values(candidate_runs, context="candidate formal evidence")
    baselines = {
        subject: _leadership_values(
            baseline_runs_by_subject[subject], context=f"{subject} formal evidence"
        )
        for subject in sota_gate.BASELINE_SUBJECTS
    }
    aggregate: dict[str, dict[str, object]] = {}
    for dimension in ("full_0ms", "mean_12", "worst_12"):
        candidate_value = _aggregate(candidate, dimension)
        baseline_values = {
            subject: _aggregate(baselines[subject], dimension)
            for subject in sota_gate.BASELINE_SUBJECTS
        }
        best_value = max(baseline_values.values())
        best_subjects = [
            subject
            for subject in sota_gate.BASELINE_SUBJECTS
            if baseline_values[subject] == best_value
        ]
        margin = candidate_value - best_value
        aggregate[dimension] = {
            "candidate_value": candidate_value,
            "best_baseline_value": best_value,
            "best_baseline_subject": best_subjects[0],
            "best_baseline_subjects": best_subjects,
            "margin": margin,
            "passes_gate": (
                margin >= -sota_gate.FULL_0MS_MAX_DEFICIT
                if dimension == "full_0ms"
                else margin > 0.0
            ),
        }
    return {
        "policy": dict(GATE_POLICY),
        "aggregate_comparisons": aggregate,
        "full_0ms_margin": aggregate["full_0ms"]["margin"],
        "mean_12_margin": aggregate["mean_12"]["margin"],
        "worst_12_margin": aggregate["worst_12"]["margin"],
        "gate_passed": all(
            bool(aggregate[dimension]["passes_gate"])
            for dimension in ("full_0ms", "mean_12", "worst_12")
        ),
    }


def build_gate_receipt(
    *,
    candidate_id: str,
    source_dataset_id: str,
    template_task_id: str,
    training_task_id: str,
    final_model_id: str,
    evaluation_task_id: str,
    final_checkpoint_sha256: str,
    final_checkpoint_size_bytes: int,
    run_contract_sha256: str,
    manifest_sha256: str,
    sealed_source_tree_sha256: str,
    teacher_audit_sha256: str,
    final_checkpoint_contract_sha256: str,
    evaluation_run_contract_sha256: str,
    evaluation_plan_sha256: str,
    metrics_artifact_sha256: str,
    gate_artifact_sha256: str,
    prediction_evidence_sha256: str,
    prediction_evidence_archive_sha256: str,
    evaluator_evidence_sha256: str,
    candidate_runs: Sequence[Mapping[str, object]],
    baseline_runs_by_subject: Mapping[str, Sequence[Mapping[str, object]]],
    baseline_evidence_sha256_by_subject: Mapping[str, str],
) -> dict[str, object]:
    """Build a sealed trained-candidate gate receipt for the serial queue."""

    _validate_candidate_registry()
    spec = CANDIDATE_BY_ID.get(candidate_id)
    if spec is None:
        raise ValueError(f"unknown sequential candidate {candidate_id!r}")
    if type(final_checkpoint_size_bytes) is not int or final_checkpoint_size_bytes <= 0:
        raise ValueError("final checkpoint size must be a positive integer")
    candidate_formal_runs = _normalize_formal_runs(
        candidate_runs, context=f"{candidate_id} formal evidence"
    )
    if tuple(baseline_runs_by_subject) != sota_gate.BASELINE_SUBJECTS:
        raise ValueError("formal baseline run inventory drifted")
    baseline_formal_runs = {
        subject: _normalize_formal_runs(
            baseline_runs_by_subject[subject], context=f"{subject} formal evidence"
        )
        for subject in sota_gate.BASELINE_SUBJECTS
    }
    if tuple(baseline_evidence_sha256_by_subject) != sota_gate.BASELINE_SUBJECTS:
        raise ValueError("formal baseline artifact inventory drifted")
    baseline_hashes = {
        subject: _require_sha256(
            baseline_evidence_sha256_by_subject[subject],
            context=f"{subject} formal evidence SHA",
        )
        for subject in sota_gate.BASELINE_SUBJECTS
    }
    gate = _gate_from_formal_runs(candidate_formal_runs, baseline_formal_runs)
    payload = {
        "schema_version": 2,
        "document_type": GATE_DOCUMENT_TYPE,
        "candidate_id": candidate_id,
        "candidate_subject": spec.experiment,
        "candidate_identity": spec.identity,
        "evidence_class": spec.evidence_class,
        "weights_retrained": True,
        "training_protocol": dict(TRAINING_PROTOCOL),
        "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
        "baseline_subjects": list(sota_gate.BASELINE_SUBJECTS),
        "checkpoint": {
            "policy": "epoch_50_final_only",
            "filename": "epoch_50.pth",
            "epoch": 50,
            "model_id": final_model_id,
            "sha256": final_checkpoint_sha256,
            "size_bytes": final_checkpoint_size_bytes,
            "contract_artifact_sha256": final_checkpoint_contract_sha256,
        },
        "producer": {
            "status": "completed",
            "queue": spec.training_queue,
            "queue_id": spec.training_queue_id,
            "source_dataset_id": source_dataset_id,
            "template_task_id": template_task_id,
            "training_task_id": training_task_id,
            "config": spec.config,
            "config_sha256": spec.config_sha256,
            "run_contract_sha256": run_contract_sha256,
            "manifest_sha256": manifest_sha256,
            "sealed_source_tree_sha256": sealed_source_tree_sha256,
            "teacher_audit_sha256": teacher_audit_sha256,
        },
        "evaluation": {
            "status": "completed",
            "queue": spec.formal_evaluation_queue,
            "queue_id": spec.formal_evaluation_queue_id,
            "evaluation_task_id": evaluation_task_id,
            "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
            "agent_scope": "E+R",
            "run_count": sota_gate.RUN_COUNT,
            "ap_metric_keys": list(AP_METRIC_KEYS),
            "run_contract_sha256": evaluation_run_contract_sha256,
            "evaluation_plan_sha256": evaluation_plan_sha256,
            "metrics_artifact_sha256": metrics_artifact_sha256,
            "gate_artifact_sha256": gate_artifact_sha256,
            "prediction_evidence_sha256": prediction_evidence_sha256,
            "prediction_evidence_archive_sha256": (prediction_evidence_archive_sha256),
            "evaluator_evidence_sha256": evaluator_evidence_sha256,
            "runs": candidate_formal_runs,
        },
        "baseline_reference": {
            "subjects": list(sota_gate.BASELINE_SUBJECTS),
            "formal_evidence_sha256_by_subject": baseline_hashes,
            "runs_by_subject": baseline_formal_runs,
        },
        "gate": gate,
    }
    return _sealed(payload)


def seal_chain_evidence(
    *,
    initial_selection: Mapping[str, object],
    sequential_gate_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 2,
            "document_type": CHAIN_DOCUMENT_TYPE,
            "initial_selection": dict(initial_selection),
            "sequential_gate_receipts": [
                dict(receipt) for receipt in sequential_gate_receipts
            ],
        }
    )


def _validate_initial_validated_row(row: Mapping[str, object], *, subject: str) -> bool:
    expected_keys = {
        "subject",
        "candidate_label",
        "fixed_order_index",
        "evidence_class",
        "weights_retrained",
        "eligible_for_final_paper_selection",
        "conditions_won",
        "conditions_total",
        "conditions_won_fraction",
        "strictly_leads_all_conditions",
        "per_condition_lead_required",
        "condition_comparisons",
        "aggregate_comparisons",
        "gate_passed",
        "ranking_values",
        "evidence_status",
        "evidence_fingerprint_sha256",
    }
    _require_exact_keys(row, expected_keys, context=f"initial candidate {subject}")
    comparisons = row.get("condition_comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != sota_gate.RUN_COUNT:
        raise ValueError(f"initial candidate {subject} condition inventory drifted")
    won = 0
    for index, (comparison, (delay, condition)) in enumerate(
        zip(
            comparisons,
            (
                (delay, condition)
                for delay in sota_gate.DELAYS_MS
                for condition in sota_gate.CONDITIONS
            ),
            strict=True,
        )
    ):
        if not isinstance(comparison, Mapping):
            raise ValueError(f"initial candidate {subject} condition {index} invalid")
        _require_exact_keys(
            comparison,
            {
                "condition_id",
                "delay_ms",
                "condition",
                "candidate_value",
                "best_baseline_value",
                "best_baseline_subject",
                "best_baseline_subjects",
                "margin",
                "strictly_leads",
            },
            context=f"initial candidate {subject} condition {index}",
        )
        candidate_value = _require_ap(
            comparison.get("candidate_value"),
            context=f"initial candidate {subject} condition value",
        )
        baseline_value = _require_ap(
            comparison.get("best_baseline_value"),
            context=f"initial candidate {subject} baseline value",
        )
        margin = _finite(
            comparison.get("margin"), context=f"initial candidate {subject} margin"
        )
        strictly_leads = margin > 0.0
        best_subject = comparison.get("best_baseline_subject")
        best_subjects = comparison.get("best_baseline_subjects")
        if (
            comparison.get("condition_id") != _condition_id(delay, condition)
            or comparison.get("delay_ms") != delay
            or comparison.get("condition") != condition
            or not math.isclose(
                margin, candidate_value - baseline_value, rel_tol=0.0, abs_tol=1e-12
            )
            or comparison.get("strictly_leads") is not strictly_leads
            or type(best_subject) is not str
            or best_subject not in sota_gate.BASELINE_SUBJECTS
            or not isinstance(best_subjects, list)
            or not best_subjects
            or best_subjects[0] != best_subject
            or any(item not in sota_gate.BASELINE_SUBJECTS for item in best_subjects)
        ):
            raise ValueError(
                f"initial candidate {subject} condition {index} is inconsistent"
            )
        won += int(strictly_leads)
    if (
        row.get("conditions_won") != won
        or row.get("conditions_total") != sota_gate.RUN_COUNT
        or row.get("conditions_won_fraction") != f"{won}/{sota_gate.RUN_COUNT}"
        or row.get("strictly_leads_all_conditions") is not (won == sota_gate.RUN_COUNT)
        or row.get("per_condition_lead_required") is not False
    ):
        raise ValueError(f"initial candidate {subject} condition summary drifted")

    aggregate = row.get("aggregate_comparisons")
    if not isinstance(aggregate, Mapping) or set(aggregate) != {
        "full_0ms",
        "mean_12",
        "worst_12",
    }:
        raise ValueError(f"initial candidate {subject} aggregates drifted")
    margins: dict[str, float] = {}
    passes: list[bool] = []
    for dimension in ("full_0ms", "mean_12", "worst_12"):
        item = aggregate[dimension]
        if not isinstance(item, Mapping):
            raise ValueError(f"initial candidate {subject} {dimension} is invalid")
        _require_exact_keys(
            item,
            {
                "candidate_value",
                "best_baseline_value",
                "best_baseline_subject",
                "best_baseline_subjects",
                "margin",
                "strictly_leads",
                "gate_comparison",
                "passes_gate",
            },
            context=f"initial candidate {subject} {dimension}",
        )
        candidate_value = _require_ap(
            item.get("candidate_value"), context=f"{subject} {dimension} value"
        )
        baseline_value = _require_ap(
            item.get("best_baseline_value"),
            context=f"{subject} {dimension} baseline",
        )
        margin = _finite(item.get("margin"), context=f"{subject} {dimension} margin")
        expected_pass = (
            margin >= -sota_gate.FULL_0MS_MAX_DEFICIT
            if dimension == "full_0ms"
            else margin > 0.0
        )
        expected_comparison = (
            "greater_than_or_equal_to_best_minus_0.5"
            if dimension == "full_0ms"
            else "strictly_greater_than_best"
        )
        best_subject = item.get("best_baseline_subject")
        best_subjects = item.get("best_baseline_subjects")
        if (
            not math.isclose(
                margin, candidate_value - baseline_value, rel_tol=0.0, abs_tol=1e-12
            )
            or item.get("strictly_leads") is not (margin > 0.0)
            or item.get("passes_gate") is not expected_pass
            or item.get("gate_comparison") != expected_comparison
            or type(best_subject) is not str
            or best_subject not in sota_gate.BASELINE_SUBJECTS
            or not isinstance(best_subjects, list)
            or not best_subjects
            or best_subjects[0] != best_subject
        ):
            raise ValueError(f"initial candidate {subject} {dimension} is inconsistent")
        margins[dimension] = margin
        passes.append(expected_pass)
    expected_gate = all(passes)
    ranking = row.get("ranking_values")
    if not isinstance(ranking, Mapping) or ranking != {
        "worst_12_margin": margins["worst_12"],
        "mean_12_margin": margins["mean_12"],
        "full_0ms_margin": margins["full_0ms"],
        "fixed_candidate_order": sota_gate.CANDIDATE_ORDER.index(subject),
    }:
        raise ValueError(f"initial candidate {subject} ranking values drifted")
    if row.get("gate_passed") is not expected_gate:
        raise ValueError(f"initial candidate {subject} gate boolean drifted")
    return expected_gate


def _initial_selection_state(selection: Mapping[str, object]) -> str:
    expected_keys = {
        "schema_version",
        "document_type",
        "status",
        "selection_stage",
        "selection_is_final",
        "protocol_evidence_fingerprint",
        "baseline_subjects",
        "baseline_display_names",
        "candidate_registry",
        "candidate_subjects",
        "candidate_count",
        "evaluated_candidate_count",
        "pending_candidate_subjects",
        "pending_candidate_count",
        "gate",
        "ranking_key",
        "performance_ranked_candidates",
        "screening_selected_candidate",
        "screening_selected_candidate_eligible_for_final_claim",
        "selected_trained_candidate",
        "selected_candidate",
        "requires_p1_independent_training",
        "checkpoint_reuse_can_enter_trained_table",
        "recommended_action",
        "candidate_results",
        "seal_sha256",
    }
    _require_exact_keys(selection, expected_keys, context="initial selector")
    _require_seal(selection, context="initial selector")
    if (
        selection.get("schema_version") != 1
        or selection.get("document_type") != "resilient_v2x_sota_candidate_selection"
        or selection.get("selection_stage") != "single_seed_1337_sota_screening"
    ):
        raise ValueError("initial selector document type mismatch")
    fingerprint = selection.get("protocol_evidence_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ValueError("initial selector lacks a protocol evidence fingerprint")
    sota_gate.validate_evidence_fingerprint(fingerprint, context="initial selector")
    if selection.get("baseline_subjects") != list(sota_gate.BASELINE_SUBJECTS):
        raise ValueError("initial selector baseline set drifted")
    if selection.get("baseline_display_names") != dict(
        sota_gate.BASELINE_DISPLAY_NAMES
    ):
        raise ValueError("initial selector baseline labels drifted")
    if (
        selection.get("candidate_registry") != sota_gate.candidate_registry()
        or selection.get("candidate_subjects") != list(sota_gate.CANDIDATE_ORDER)
        or selection.get("candidate_count") != len(sota_gate.CANDIDATE_ORDER)
        or selection.get("ranking_key") != list(sota_gate.RANKING_KEY)
        or selection.get("checkpoint_reuse_can_enter_trained_table") is not False
    ):
        raise ValueError("initial selector candidate contract drifted")
    if selection.get("gate") != {
        "metric": sota_gate.LEADERSHIP_METRIC,
        "full_0ms": "candidate >= best controlled baseline - 0.5",
        "mean_12": "candidate > best controlled baseline",
        "worst_12": "candidate > best controlled baseline",
        "per_condition_lead_required": False,
    }:
        raise ValueError("initial selector gate policy drifted")
    rows = selection.get("candidate_results")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("initial selector candidate rows are invalid")
    by_subject: dict[str, Mapping[str, object]] = {}
    for row in rows:
        subject = row.get("subject")
        if type(subject) is not str or subject in by_subject:
            raise ValueError("initial selector candidate subjects are invalid")
        spec = sota_gate.CANDIDATE_BY_SUBJECT.get(subject)
        if spec is None or any(
            row.get(key) != expected
            for key, expected in {
                "candidate_label": spec.label,
                "fixed_order_index": sota_gate.CANDIDATE_ORDER.index(subject),
                "evidence_class": spec.evidence_class,
                "weights_retrained": spec.weights_retrained,
                "eligible_for_final_paper_selection": (
                    spec.eligible_for_final_paper_selection
                ),
            }.items()
        ):
            raise ValueError(f"initial selector row {subject!r} drifted")
        by_subject[subject] = row
    if tuple(by_subject) != sota_gate.CANDIDATE_ORDER:
        raise ValueError("initial selector candidate row order drifted")
    pending_subjects = [
        subject
        for subject in sota_gate.CANDIDATE_ORDER
        if by_subject[subject].get("evidence_status") == "pending"
    ]
    evaluated_subjects = [
        subject
        for subject in sota_gate.CANDIDATE_ORDER
        if by_subject[subject].get("evidence_status") == "validated"
    ]
    if (
        selection.get("pending_candidate_subjects") != pending_subjects
        or selection.get("pending_candidate_count") != len(pending_subjects)
        or selection.get("evaluated_candidate_count") != len(evaluated_subjects)
    ):
        raise ValueError("initial selector evidence counts drifted")
    missing = sorted(set(INITIAL_TRAINED_SUBJECTS) - set(by_subject))
    if missing:
        raise ValueError(f"initial selector is missing trained candidates: {missing!r}")
    pending = False
    passing: list[str] = []
    for subject in INITIAL_TRAINED_SUBJECTS:
        row = by_subject[subject]
        if row.get("weights_retrained") is not True:
            raise ValueError(f"initial candidate {subject} is not trained evidence")
        if row.get("eligible_for_final_paper_selection") is not True:
            raise ValueError(f"initial candidate {subject} is not claim-eligible")
        status = row.get("evidence_status")
        gate_passed = row.get("gate_passed")
        if status == "pending" and gate_passed is None:
            _require_exact_keys(
                row,
                {
                    "subject",
                    "candidate_label",
                    "fixed_order_index",
                    "evidence_status",
                    "evidence_class",
                    "weights_retrained",
                    "eligible_for_final_paper_selection",
                    "gate_passed",
                    "ranking_values",
                },
                context=f"pending initial candidate {subject}",
            )
            if row.get("ranking_values") is not None:
                raise ValueError(
                    f"pending initial candidate {subject} has ranking evidence"
                )
            pending = True
        elif status == "validated" and type(gate_passed) is bool:
            if (
                row.get("evidence_fingerprint_sha256")
                != fingerprint["fingerprint_sha256"]
            ):
                raise ValueError(
                    f"initial candidate {subject} formal fingerprint drifted"
                )
            if _validate_initial_validated_row(row, subject=subject):
                passing.append(subject)
        else:
            raise ValueError(f"initial candidate {subject} has invalid gate state")
    zero_shot = by_subject.get(sota_gate.P1_ZERO_SHOT_SUBJECT)
    if zero_shot is not None and (
        zero_shot.get("weights_retrained") is not False
        or zero_shot.get("eligible_for_final_paper_selection") is not False
    ):
        raise ValueError("P1 zero-shot evidence was mislabeled as trained evidence")
    for _label, subject in INITIAL_FORMAL_BLOCKERS:
        row = by_subject[subject]
        if (
            row.get("evidence_status") != "validated"
            or type(row.get("gate_passed")) is not bool
        ):
            pending = True
    selected = selection.get("selected_trained_candidate")
    ranked_rows = sorted(
        (by_subject[subject] for subject in evaluated_subjects),
        key=sota_gate.ranking_sort_key,
    )
    expected_ranking = [str(row["subject"]) for row in ranked_rows]
    if selection.get("performance_ranked_candidates") != expected_ranking:
        raise ValueError("initial selector performance ranking drifted")
    if passing:
        if selected not in passing:
            raise ValueError("initial selector trained winner is inconsistent")
        return "pass"
    if selected is not None:
        raise ValueError("initial selector names a trained winner without a pass")
    return "pending" if pending else "failed"


def _validate_gate_receipt(
    receipt: Mapping[str, object], *, expected_candidate_id: str
) -> bool:
    expected_keys = {
        "schema_version",
        "document_type",
        "candidate_id",
        "candidate_subject",
        "candidate_identity",
        "evidence_class",
        "weights_retrained",
        "training_protocol",
        "protocol_evidence_fingerprint",
        "baseline_subjects",
        "checkpoint",
        "producer",
        "evaluation",
        "baseline_reference",
        "gate",
        "seal_sha256",
    }
    _require_exact_keys(receipt, expected_keys, context=f"{expected_candidate_id} gate")
    _require_seal(receipt, context=f"{expected_candidate_id} gate")
    spec = CANDIDATE_BY_ID[expected_candidate_id]
    if (
        receipt.get("schema_version") != 2
        or receipt.get("document_type") != GATE_DOCUMENT_TYPE
        or receipt.get("candidate_id") != expected_candidate_id
        or receipt.get("candidate_subject") != spec.experiment
        or receipt.get("candidate_identity") != spec.identity
        or receipt.get("evidence_class") != spec.evidence_class
        or receipt.get("weights_retrained") is not True
    ):
        raise ValueError(f"{expected_candidate_id} is not exact trained evidence")
    if receipt.get("training_protocol") != TRAINING_PROTOCOL:
        raise ValueError(f"{expected_candidate_id} training protocol drifted")
    fingerprint = receipt.get("protocol_evidence_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ValueError(f"{expected_candidate_id} lacks evaluation fingerprint")
    sota_gate.validate_evidence_fingerprint(
        fingerprint, context=f"{expected_candidate_id} gate"
    )
    if receipt.get("baseline_subjects") != list(sota_gate.BASELINE_SUBJECTS):
        raise ValueError(f"{expected_candidate_id} baseline set drifted")

    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{expected_candidate_id} checkpoint evidence is invalid")
    _require_exact_keys(
        checkpoint,
        {
            "policy",
            "filename",
            "epoch",
            "model_id",
            "sha256",
            "size_bytes",
            "contract_artifact_sha256",
        },
        context=f"{expected_candidate_id} checkpoint",
    )
    if (
        checkpoint.get("policy") != "epoch_50_final_only"
        or checkpoint.get("filename") != "epoch_50.pth"
        or checkpoint.get("epoch") != 50
    ):
        raise ValueError(f"{expected_candidate_id} is not the final checkpoint")
    _require_id(checkpoint.get("model_id"), context="final model ID")
    _require_sha256(checkpoint.get("sha256"), context="final checkpoint SHA")
    size = checkpoint.get("size_bytes")
    if type(size) is not int or size <= 0:
        raise ValueError(f"{expected_candidate_id} checkpoint size is invalid")
    _require_sha256(
        checkpoint.get("contract_artifact_sha256"),
        context=f"{expected_candidate_id} final checkpoint contract",
    )

    producer = receipt.get("producer")
    if not isinstance(producer, Mapping):
        raise ValueError(f"{expected_candidate_id} producer evidence is invalid")
    _require_exact_keys(
        producer,
        {
            "status",
            "queue",
            "queue_id",
            "source_dataset_id",
            "template_task_id",
            "training_task_id",
            "config",
            "config_sha256",
            "run_contract_sha256",
            "manifest_sha256",
            "sealed_source_tree_sha256",
            "teacher_audit_sha256",
        },
        context=f"{expected_candidate_id} producer",
    )
    if (
        producer.get("status") != "completed"
        or producer.get("queue") != spec.training_queue
        or producer.get("queue_id") != spec.training_queue_id
        or producer.get("config") != spec.config
        or producer.get("config_sha256") != spec.config_sha256
    ):
        raise ValueError(f"{expected_candidate_id} producer lifecycle drifted")
    for field in ("source_dataset_id", "template_task_id", "training_task_id"):
        _require_id(producer.get(field), context=f"{expected_candidate_id} {field}")
    for field in (
        "run_contract_sha256",
        "manifest_sha256",
        "sealed_source_tree_sha256",
        "teacher_audit_sha256",
    ):
        _require_sha256(producer.get(field), context=f"{expected_candidate_id} {field}")

    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{expected_candidate_id} evaluator evidence is invalid")
    _require_exact_keys(
        evaluation,
        {
            "status",
            "queue",
            "queue_id",
            "evaluation_task_id",
            "protocol_evidence_fingerprint",
            "agent_scope",
            "run_count",
            "ap_metric_keys",
            "run_contract_sha256",
            "evaluation_plan_sha256",
            "metrics_artifact_sha256",
            "gate_artifact_sha256",
            "prediction_evidence_sha256",
            "prediction_evidence_archive_sha256",
            "evaluator_evidence_sha256",
            "runs",
        },
        context=f"{expected_candidate_id} evaluation",
    )
    if (
        evaluation.get("status") != "completed"
        or evaluation.get("queue") != spec.formal_evaluation_queue
        or evaluation.get("queue_id") != spec.formal_evaluation_queue_id
        or evaluation.get("agent_scope") != "E+R"
        or evaluation.get("run_count") != sota_gate.RUN_COUNT
        or evaluation.get("ap_metric_keys") != list(AP_METRIC_KEYS)
    ):
        raise ValueError(f"{expected_candidate_id} evaluation lifecycle drifted")
    _require_id(evaluation.get("evaluation_task_id"), context="evaluation task ID")
    evaluation_fingerprint = evaluation.get("protocol_evidence_fingerprint")
    if not isinstance(evaluation_fingerprint, Mapping):
        raise ValueError(f"{expected_candidate_id} evaluation fingerprint is invalid")
    sota_gate.validate_evidence_fingerprint(
        evaluation_fingerprint, context=f"{expected_candidate_id} evaluation"
    )
    for field in (
        "run_contract_sha256",
        "evaluation_plan_sha256",
        "metrics_artifact_sha256",
        "gate_artifact_sha256",
        "prediction_evidence_sha256",
        "prediction_evidence_archive_sha256",
        "evaluator_evidence_sha256",
    ):
        _require_sha256(
            evaluation.get(field), context=f"{expected_candidate_id} {field}"
        )
    raw_candidate_runs = evaluation.get("runs")
    if not isinstance(raw_candidate_runs, list):
        raise ValueError(f"{expected_candidate_id} formal runs are invalid")
    candidate_runs = _normalize_formal_runs(
        raw_candidate_runs, context=f"{expected_candidate_id} formal evidence"
    )

    baseline_reference = receipt.get("baseline_reference")
    if not isinstance(baseline_reference, Mapping):
        raise ValueError(f"{expected_candidate_id} baseline reference is invalid")
    _require_exact_keys(
        baseline_reference,
        {"subjects", "formal_evidence_sha256_by_subject", "runs_by_subject"},
        context=f"{expected_candidate_id} baseline reference",
    )
    if baseline_reference.get("subjects") != list(sota_gate.BASELINE_SUBJECTS):
        raise ValueError(f"{expected_candidate_id} baseline order drifted")
    raw_baseline_hashes = baseline_reference.get("formal_evidence_sha256_by_subject")
    raw_baseline_runs = baseline_reference.get("runs_by_subject")
    if not isinstance(raw_baseline_hashes, Mapping) or set(raw_baseline_hashes) != set(
        sota_gate.BASELINE_SUBJECTS
    ):
        raise ValueError(f"{expected_candidate_id} baseline hashes drifted")
    if not isinstance(raw_baseline_runs, Mapping) or set(raw_baseline_runs) != set(
        sota_gate.BASELINE_SUBJECTS
    ):
        raise ValueError(f"{expected_candidate_id} baseline runs drifted")
    baseline_runs: dict[str, list[dict[str, object]]] = {}
    for subject in sota_gate.BASELINE_SUBJECTS:
        _require_sha256(
            raw_baseline_hashes[subject], context=f"{subject} formal evidence SHA"
        )
        raw_runs = raw_baseline_runs[subject]
        if not isinstance(raw_runs, list):
            raise ValueError(f"{subject} formal runs are invalid")
        baseline_runs[subject] = _normalize_formal_runs(
            raw_runs, context=f"{subject} formal evidence"
        )

    gate = receipt.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError(f"{expected_candidate_id} gate payload is invalid")
    expected_gate = _gate_from_formal_runs(candidate_runs, baseline_runs)
    if _canonical_json(dict(gate)) != _canonical_json(expected_gate):
        raise ValueError(f"{expected_candidate_id} gate evidence is inconsistent")
    return bool(expected_gate["gate_passed"])


def validate_chain_evidence(value: Mapping[str, object]) -> dict[str, object]:
    _require_exact_keys(
        value,
        {
            "schema_version",
            "document_type",
            "initial_selection",
            "sequential_gate_receipts",
            "seal_sha256",
        },
        context="sequential evidence",
    )
    _require_seal(value, context="sequential evidence")
    if value.get("schema_version") != 2 or value.get("document_type") != (
        CHAIN_DOCUMENT_TYPE
    ):
        raise ValueError("sequential evidence identity drifted")
    initial = value.get("initial_selection")
    if not isinstance(initial, Mapping):
        raise ValueError("sequential evidence lacks the initial selector")
    initial_state = _initial_selection_state(initial)
    raw_receipts = value.get("sequential_gate_receipts")
    if not isinstance(raw_receipts, list) or not all(
        isinstance(receipt, Mapping) for receipt in raw_receipts
    ):
        raise ValueError("sequential gate receipts must be an object list")
    if len(raw_receipts) > len(CANDIDATE_ORDER):
        raise ValueError("too many sequential gate receipts")
    receipts: list[dict[str, object]] = []
    pass_seen = False
    for index, raw in enumerate(raw_receipts):
        if initial_state != "failed":
            raise ValueError("sequential receipts exist before all initial failures")
        if pass_seen:
            raise ValueError("a gate receipt exists after the first eligible pass")
        candidate_id = CANDIDATE_ORDER[index]
        passed = _validate_gate_receipt(raw, expected_candidate_id=candidate_id)
        receipts.append(dict(raw))
        pass_seen = passed
    return {
        "initial_state": initial_state,
        "receipts": receipts,
        "last_gate_passed": pass_seen,
        "evidence_seal_sha256": value["seal_sha256"],
    }


def build_plan(value: Mapping[str, object]) -> dict[str, object]:
    _validate_candidate_registry()
    state = validate_chain_evidence(value)
    initial_state = state["initial_state"]
    receipts = state["receipts"]
    assert isinstance(receipts, list)
    next_candidate_id: str | None = None
    selected_candidate_id: str | None = None
    if initial_state == "pending":
        status = "await_existing_candidate_evidence"
    elif initial_state == "pass":
        status = "stop_existing_candidate_passed"
    elif state["last_gate_passed"] is True:
        status = "stop_sequential_candidate_passed"
        selected_candidate_id = str(receipts[-1]["candidate_id"])
    elif len(receipts) == len(CANDIDATE_ORDER):
        status = "predefined_candidates_exhausted_without_lead"
    else:
        next_candidate_id = CANDIDATE_ORDER[len(receipts)]
        status = "ready_to_create_next_candidate"
    stop = status in {
        "stop_existing_candidate_passed",
        "stop_sequential_candidate_passed",
        "predefined_candidates_exhausted_without_lead",
    }
    next_spec = CANDIDATE_BY_ID.get(next_candidate_id or "")
    initial_selection = value["initial_selection"]
    assert isinstance(initial_selection, Mapping)
    initial_rows = initial_selection["candidate_results"]
    assert isinstance(initial_rows, list)
    initial_by_subject = {
        str(row["subject"]): row for row in initial_rows if isinstance(row, Mapping)
    }
    initial_barrier_rows = [
        {
            "label": label,
            "subject": subject,
            "formal_evidence_status": initial_by_subject[subject]["evidence_status"],
            "gate_passed": initial_by_subject[subject]["gate_passed"],
        }
        for label, subject in INITIAL_FORMAL_BLOCKERS
    ]
    payload = {
        "schema_version": 2,
        "document_type": PLAN_DOCUMENT_TYPE,
        "status": status,
        "candidate_order": list(CANDIDATE_ORDER),
        "candidate_registry": [asdict(spec) for spec in CANDIDATE_SPECS],
        "completed_sequential_gate_count": len(receipts),
        "next_candidate_id": next_candidate_id,
        "next_candidate": asdict(next_spec) if next_spec is not None else None,
        "selected_candidate_id": selected_candidate_id,
        "create_new_training": next_candidate_id is not None,
        "stop_create_new_training": stop,
        "already_running_training_unaffected": list(ALREADY_RUNNING_UNAFFECTED),
        "initial_formal_barrier": {
            "policy": (
                "E1/E2/E3/P0/P2 must each have completed formal 1337x12 "
                "E+R evidence and every trained initial candidate must fail"
            ),
            "required": [
                {"label": label, "subject": subject}
                for label, subject in INITIAL_FORMAL_BLOCKERS
            ],
            "rows": initial_barrier_rows,
            "ready_to_unlock_p1": initial_state == "failed",
        },
        "serial_lifecycle": {
            "order": list(CANDIDATE_ORDER),
            "current_candidate_id": next_candidate_id,
            "completed_training_and_formal_gate_count": len(receipts),
            "successor_rule": (
                "create exactly one successor only after the immediately prior "
                "candidate completed 50-epoch training, final checkpoint sealing, "
                "formal 1337x12 evaluation, and a failed recomputed gate"
            ),
            "first_pass_rule": (
                "stop creating new training after the first trained gate pass; "
                "already-running work may finish"
            ),
        },
        "p1_zero_shot_is_training_evidence": False,
        "training_protocol": dict(TRAINING_PROTOCOL),
        "evaluation_protocol": sota_gate.evidence_fingerprint(),
        "gate_policy": dict(GATE_POLICY),
        "queue_mapping": (
            {
                "training": {
                    "name": next_spec.training_queue,
                    "id": next_spec.training_queue_id,
                },
                "formal_evaluation": {
                    "name": next_spec.formal_evaluation_queue,
                    "id": next_spec.formal_evaluation_queue_id,
                },
            }
            if next_spec is not None
            else None
        ),
        "formal_evaluation_contract": (
            {
                "create_only_after_training_status": "completed",
                "checkpoint_policy": "epoch_50_final_only",
                "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
                "agent_scope": "E+R",
                "required_ap_metric_keys": list(AP_METRIC_KEYS),
                "required_artifacts": [
                    "run_contract",
                    "evaluation_plan",
                    "controlled_baseline_metrics",
                    "controlled_baseline_evidence",
                    "formal_gate_artifact",
                ],
                "successor_requires_gate_passed": False,
            }
            if next_spec is not None
            else None
        ),
        "evidence_seal_sha256": state["evidence_seal_sha256"],
        "execution": {
            "dry_run_default": True,
            "remote_state_changed": False,
            "explicit_execute_flag_required": True,
            "execute_token": EXECUTE_TOKEN,
            "remote_phases": ["upload-source", "create-template", "launch"],
            "candidate_phase_token_required": True,
            "formal_evaluation_is_separate_completed_evidence_barrier": True,
        },
    }
    return _sealed(payload)


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"JSON evidence is missing or not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return value


def _gate_receipt_by_id(
    evidence: Mapping[str, object], candidate_id: str
) -> Mapping[str, object]:
    receipts = evidence["sequential_gate_receipts"]
    assert isinstance(receipts, list)
    for receipt in receipts:
        assert isinstance(receipt, Mapping)
        if receipt.get("candidate_id") == candidate_id:
            return receipt
    raise ValueError(f"missing formal gate receipt for {candidate_id}")


def _deployment_profile(
    candidate_id: str, evidence: Mapping[str, object]
) -> tuple[object, object]:
    """Build the candidate profile through the existing sealed P0/P1 engine."""

    _validate_candidate_registry()
    from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared
    from tools.resilient_v2x import prepare_clearml_sota_round2_p1 as p1

    if candidate_id == "P1":
        spec = CANDIDATE_BY_ID[candidate_id]
        legacy_contract = p1.P1_PROFILE.execution_contract
        if not isinstance(legacy_contract, Mapping):
            raise RuntimeError("P1 dual-candidate execution contract is missing")
        _require_seal(legacy_contract, context="P1 dual-candidate contract")
        bindings = p1.P1_PROFILE.task_parameter_bindings
        if not isinstance(bindings, Mapping):
            raise RuntimeError("P1 exact task parameter bindings are missing")
        sequential_contract = _sealed(
            {
                "schema_version": 2,
                "contract_type": "resilient_v2x_single_seed_p1_training_v2",
                "candidate_id": "P1",
                "candidate_identity": spec.identity,
                "candidate_subject": spec.experiment,
                "config": {
                    "path": spec.config,
                    "sha256": spec.config_sha256,
                    "semantic_base": spec.config_base,
                },
                "dependency_parent_candidate_id": spec.parent_candidate_id,
                "training_protocol": dict(TRAINING_PROTOCOL),
                "task_parameter_bindings": dict(bindings),
                "queue_mapping": {
                    "training": {
                        "name": spec.training_queue,
                        "id": spec.training_queue_id,
                    },
                    "formal_evaluation": {
                        "name": spec.formal_evaluation_queue,
                        "id": spec.formal_evaluation_queue_id,
                    },
                },
                "final_checkpoint_contract": {
                    "policy": "epoch_50_final_only",
                    "filename": "epoch_50.pth",
                    "clean_best_is_diagnostic_only": True,
                    "model_and_contract_sha256_required": True,
                },
                "formal_evaluation_contract": {
                    "protocol_evidence_fingerprint": (sota_gate.evidence_fingerprint()),
                    "agent_scope": "E+R",
                    "ap_metric_keys": list(AP_METRIC_KEYS),
                    "candidate_and_five_baseline_raw_runs_required": True,
                    "prediction_evidence_required": True,
                },
                "formal_evaluation_required_before_successor": True,
                "successor_creation_requires_gate_passed_false": True,
                "zero_shot_evidence_forbidden": True,
                "p1_dual_candidate_execution_contract": dict(legacy_contract),
            }
        )
        return shared, replace(
            p1.P1_PROFILE,
            worker_queue=spec.training_queue,
            worker_queue_id=spec.training_queue_id,
            execution_contract=sequential_contract,
        )
    spec = CANDIDATE_BY_ID[candidate_id]
    parent_id = spec.parent_candidate_id
    shared_module, parent_profile = _deployment_profile(parent_id, evidence)
    parent_spec = CANDIDATE_BY_ID[parent_id]
    with shared_module.use_deployment_profile(parent_profile):
        parent_manifest = shared_module._verify_package(parent_spec.output_dir)
    parent_package = parent_manifest.get("target_source_package")
    if not isinstance(parent_package, Mapping):
        raise RuntimeError(f"{parent_id} source package is invalid")
    parent_receipt = _gate_receipt_by_id(evidence, parent_id)
    producer = parent_receipt.get("producer")
    if not isinstance(producer, Mapping):
        raise RuntimeError(f"{parent_id} producer receipt is invalid")
    parent_source_dataset_id = _require_id(
        producer.get("source_dataset_id"), context=f"{parent_id} source Dataset"
    )
    output_dir = spec.output_dir
    package_manifest_name = f"single-seed-{candidate_id.lower()}-source-package.json"
    deployment_plan_name = f"single-seed-{candidate_id.lower()}-deployment-plan.json"
    relative_script = Path(__file__).relative_to(ROOT)
    spec_anchor = shared_module._P0_SPEC_ANCHOR
    spec_patch = (
        "    ExperimentSpec(\n"
        f'        "{spec.experiment}",\n'
        '        "sota_candidate",\n'
        f'        "{spec.config}",\n'
        "        True,\n"
        "    ),\n" + spec_anchor
    )
    nested_anchor = f'        "{parent_spec.experiment}",\n        "resilient_v2x",\n'
    nested_patch = (
        f'        "{parent_spec.experiment}",\n'
        f'        "{spec.experiment}",\n'
        '        "resilient_v2x",\n'
    )
    execution_contract = _sealed(
        {
            "schema_version": 2,
            "contract_type": (
                f"resilient_v2x_single_seed_{candidate_id.lower()}_training_v2"
            ),
            "candidate_id": candidate_id,
            "candidate_identity": spec.identity,
            "candidate_subject": spec.experiment,
            "config": {
                "path": spec.config,
                "sha256": spec.config_sha256,
                "semantic_base": spec.config_base,
            },
            "dependency_parent_candidate_id": spec.parent_candidate_id,
            "training_protocol": dict(TRAINING_PROTOCOL),
            "queue_mapping": {
                "training": {
                    "name": spec.training_queue,
                    "id": spec.training_queue_id,
                },
                "formal_evaluation": {
                    "name": spec.formal_evaluation_queue,
                    "id": spec.formal_evaluation_queue_id,
                },
            },
            "final_checkpoint_contract": {
                "policy": "epoch_50_final_only",
                "filename": "epoch_50.pth",
                "clean_best_is_diagnostic_only": True,
                "model_and_contract_sha256_required": True,
            },
            "formal_evaluation_contract": {
                "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
                "agent_scope": "E+R",
                "ap_metric_keys": list(AP_METRIC_KEYS),
                "candidate_and_five_baseline_raw_runs_required": True,
                "prediction_evidence_required": True,
            },
            "formal_evaluation_required_before_successor": True,
            "successor_creation_requires_gate_passed_false": True,
            "zero_shot_evidence_forbidden": True,
        }
    )
    provenance = {
        "derivation": f"{candidate_id} exact additive config from {parent_id}",
        "parent_candidate_id": parent_id,
        "parent_candidate_identity": parent_spec.identity,
        "parent_source_dataset_id": parent_source_dataset_id,
        "parent_source_package_manifest_seal_sha256": parent_manifest["seal_sha256"],
        "parent_source_tree_sha256": parent_package["tree_sha256"],
        "parent_training_task_id": producer["training_task_id"],
        "parent_gate_artifact_sha256": parent_receipt["evaluation"][
            "gate_artifact_sha256"
        ],
        "parent_gate_passed": False,
    }
    profile = shared_module.Round2DeploymentProfile(
        label=f"single-seed {candidate_id}",
        output_dir=output_dir,
        package_manifest_name=package_manifest_name,
        deployment_plan_name=deployment_plan_name,
        package_type=f"resilient_v2x_single_seed_{candidate_id.lower()}_source",
        experiment=spec.experiment,
        config=spec.config,
        identity=spec.identity,
        base_source_dir=parent_spec.output_dir,
        base_source_inventory=parent_spec.output_dir / "source-inventory.json",
        base_source_archive=(
            parent_spec.output_dir / str(parent_package["archive_name"])
        ),
        base_source_dataset_id=parent_source_dataset_id,
        base_source_package=dict(parent_package),
        provenance=provenance,
        early_gate_policy=None,
        source_transition_artifact=(
            f"single_seed_{candidate_id.lower()}_source_transition"
        ),
        launch_receipt_artifact=f"single_seed_{candidate_id.lower()}_launch_receipt",
        source_dataset_prefix=f"ResilientV2X sealed single-seed {candidate_id} source",
        source_dataset_version_suffix=f"single-seed-{candidate_id.lower()}-v1",
        source_dataset_tags=(
            "ResilientV2X",
            "source",
            "single-seed-sota",
            candidate_id.lower(),
            f"derived-from-{parent_id.lower()}",
            "sealed",
        ),
        template_prefix=f"ResilientV2X single-seed {candidate_id} template",
        task_prefix=f"ResilientV2X single-seed {candidate_id}",
        worker_queue=spec.training_queue,
        worker_queue_id=spec.training_queue_id,
        gpu_count=shared_module.GPU_COUNT,
        batch_size_per_gpu=shared_module.BATCH_SIZE_PER_GPU,
        global_batch_size=shared_module.GLOBAL_BATCH_SIZE,
        max_epochs=shared_module.MAX_EPOCHS,
        val_interval=shared_module.VAL_INTERVAL,
        training_seed=shared_module.TRAINING_SEED,
        precision=shared_module.PRECISION,
        upload_token=spec.upload_token,
        template_token=spec.template_token,
        launch_token=spec.launch_token,
        spec_anchor=spec_anchor,
        spec_patch=spec_patch,
        nested_anchor=nested_anchor,
        nested_patch=nested_patch,
        bootstrap_parent=parent_profile,
        transition_type=(
            f"exact_additive_single_seed_{candidate_id.lower()}_from_"
            f"{parent_id.lower()}_source_revision"
        ),
        plan_type=f"resilient_v2x_single_seed_{candidate_id.lower()}_a100_v1",
        receipt_type=(f"resilient_v2x_single_seed_{candidate_id.lower()}_launch_v1"),
        template_status_message=f"sealed single-seed {candidate_id} template",
        transition_description=(
            f"{candidate_id} exact one-config additive source from {parent_id}"
        ),
        invariants=(
            f"{candidate_id} is created only after formal {parent_id} gate failure",
            "fixed seed 20250218, FP32, global batch 8, 50 epochs, val/10",
            "teacher and training Dataset remain sealed and unchanged",
            "P1 zero-shot evidence cannot authorize trained-candidate succession",
        ),
        duplicate_guard_description=(
            "exact task name plus predecessor parent before clone and exact "
            "current-clone acknowledgement before enqueue"
        ),
        duplicate_guard_stages=(
            "pre_clone_requires_zero_exact_name_plus_parent_matches",
            "post_receipt_requires_unique_current_clone_before_enqueue",
        ),
        relative_script=relative_script,
        cli_description=__doc__ or "",
        bootstrap_required_markers=(spec.experiment, spec.config),
        execution_contract=execution_contract,
        training_launch_enabled=True,
        strict_enqueue_acknowledgement=True,
        enqueue_acknowledgement_artifact=(
            f"single_seed_{candidate_id.lower()}_enqueue_acknowledgement"
        ),
    )
    return shared_module, profile


def _execute_phase(
    *,
    phase: str,
    plan: Mapping[str, object],
    evidence: Mapping[str, object],
    source_dataset_id: str | None,
    template_task_id: str | None,
    queue: str | None,
) -> int:
    candidate_id = plan.get("next_candidate_id")
    if type(candidate_id) is not str or candidate_id not in CANDIDATE_BY_ID:
        raise PermissionError("sealed plan does not authorize new training")
    spec = CANDIDATE_BY_ID[candidate_id]
    shared, profile = _deployment_profile(candidate_id, evidence)
    arguments = [phase, "--output-dir", str(spec.output_dir)]
    if phase == "upload-source":
        arguments.extend(["--execute-token", profile.upload_token])
    elif phase == "create-template":
        dataset_id = _require_id(source_dataset_id, context="source Dataset ID")
        arguments.extend(
            [
                "--source-dataset-id",
                dataset_id,
                "--execute-token",
                profile.template_token,
            ]
        )
    elif phase == "launch":
        dataset_id = _require_id(source_dataset_id, context="source Dataset ID")
        template_id = _require_id(template_task_id, context="template task ID")
        selected_queue = profile.worker_queue if queue is None else queue
        if selected_queue != profile.worker_queue:
            raise ValueError(f"queue must be exactly {profile.worker_queue}")
        arguments.extend(
            [
                "--source-dataset-id",
                dataset_id,
                "--template-task-id",
                template_id,
                "--queue",
                selected_queue,
                "--execute-token",
                profile.launch_token,
            ]
        )
    elif phase != "prepare":
        raise ValueError(f"unsupported execution phase {phase!r}")
    with shared.use_deployment_profile(profile):
        return int(shared.main(arguments))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = (
        "plan",
        "prepare",
        "upload-source",
        "create-template",
        "launch",
        "show-execution-contract",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=phases,
        help=(
            "compatibility form used by the shared sealed deployment engine; "
            "it remains dry-run unless --execute is also supplied"
        ),
    )
    parser.add_argument("--evidence", type=Path)
    parser.add_argument(
        "--phase",
        choices=phases,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--plan-seal-sha256", default="")
    parser.add_argument("--remote-write-token", default="")
    parser.add_argument("--source-dataset-id")
    parser.add_argument("--template-task-id")
    parser.add_argument("--queue")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.command is not None
        and args.phase is not None
        and args.command != args.phase
    ):
        raise ValueError("positional command and --phase disagree")
    phase = args.command or args.phase or "plan"
    if args.evidence is None:
        if args.execute:
            raise PermissionError(
                "sealed sequential evidence is required for execution"
            )
        print(
            json.dumps(
                _sealed(
                    {
                        "schema_version": 2,
                        "document_type": PLAN_DOCUMENT_TYPE,
                        "status": "sealed_sequential_evidence_required",
                        "requested_phase": phase,
                        "dry_run_default": True,
                        "remote_state_changed": False,
                    }
                ),
                sort_keys=True,
            )
        )
        return 0
    evidence = _load_json(args.evidence.resolve(strict=False))
    plan = build_plan(evidence)
    if args.output_dir is not None and plan.get("next_candidate_id") is not None:
        candidate_id = plan["next_candidate_id"]
        assert isinstance(candidate_id, str)
        expected_output = CANDIDATE_BY_ID[candidate_id].output_dir.resolve(strict=False)
        if args.output_dir.resolve(strict=False) != expected_output:
            raise ValueError("output directory does not match the authorized candidate")
    if phase == "show-execution-contract":
        candidate_id = plan.get("next_candidate_id")
        if type(candidate_id) is not str:
            raise PermissionError("sealed plan does not authorize a candidate contract")
        _shared, profile = _deployment_profile(candidate_id, evidence)
        print(
            json.dumps(
                {
                    "plan_seal_sha256": plan["seal_sha256"],
                    "candidate_id": candidate_id,
                    "execution_contract": profile.execution_contract,
                    "remote_state_changed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if phase == "plan" or not args.execute:
        print(json.dumps(plan, sort_keys=True))
        return 0
    if args.execute_token != EXECUTE_TOKEN:
        raise PermissionError("exact sequential execution token is required")
    if args.plan_seal_sha256 != plan["seal_sha256"]:
        raise PermissionError("exact current plan seal is required")
    if plan.get("status") != "ready_to_create_next_candidate":
        raise PermissionError("current evidence does not authorize new training")
    candidate_id = plan["next_candidate_id"]
    assert isinstance(candidate_id, str)
    spec = CANDIDATE_BY_ID[candidate_id]
    remote_tokens = {
        "upload-source": spec.upload_token,
        "create-template": spec.template_token,
        "launch": spec.launch_token,
    }
    if phase in remote_tokens and args.remote_write_token != remote_tokens[phase]:
        raise PermissionError(
            f"exact {candidate_id} {phase} remote-write token is required"
        )
    return _execute_phase(
        phase=phase,
        plan=plan,
        evidence=evidence,
        source_dataset_id=args.source_dataset_id,
        template_task_id=args.template_task_id,
        queue=args.queue,
    )


__all__ = (
    "ALREADY_RUNNING_UNAFFECTED",
    "AP_METRIC_KEYS",
    "CANDIDATE_ORDER",
    "CANDIDATE_SPECS",
    "EXECUTE_TOKEN",
    "FORMAL_EVALUATION_QUEUE",
    "GATE_POLICY",
    "INITIAL_FORMAL_BLOCKERS",
    "QUEUE_IDS",
    "TEACHER_CHECKPOINT_SHA256",
    "TEACHER_MODEL_ID",
    "TEACHER_TASK_ID",
    "TRAINING_DATASET_ID",
    "TRAINING_QUEUE",
    "TRAINING_PROTOCOL",
    "build_gate_receipt",
    "build_plan",
    "main",
    "seal_chain_evidence",
    "validate_chain_evidence",
)


if __name__ == "__main__":
    raise SystemExit(main())
