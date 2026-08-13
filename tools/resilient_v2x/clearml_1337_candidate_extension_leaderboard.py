#!/usr/bin/env python3
"""Build the sealed P1 zero-shot candidate-extension leaderboard.

The canonical 26-method leaderboard is consumed read-only and referenced by
task ID plus seal.  Its subject order, rows, artifact, and seal are never
rewritten.  This artifact contains exactly one separately identified extension
candidate and live-revalidated comparisons against the five controlled baselines.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from allegroai import Task
except ImportError:
    try:
        from clearml import Task
    except ImportError:
        Task = None  # type: ignore[assignment]

from tools.resilient_v2x import clearml_formal_candidate_selector as formal  # noqa: E402
from tools.resilient_v2x import sota_gate  # noqa: E402
from tools.resilient_v2x import (  # noqa: E402
    clearml_p1_zero_shot_candidate_extension as extension,
)


DEFAULT_PROJECT = "ResilientV2X/Training"
DEFAULT_CANONICAL_LEADERBOARD_TASK_ID = "f502bdd329ad4ef4b4b6cf5c5f52aba0"
CANONICAL_LEADERBOARD_ARTIFACT = "formal_1337_leaderboard"
EXTENSION_LEADERBOARD_ARTIFACT = "formal_1337_candidate_extension_leaderboard"
EXTENSION_LEADERBOARD_TYPE = "resilient_v2x_formal_1337_candidate_extension_leaderboard"
FAILED_STATUSES = extension.FAILED_STATUSES
WAITING_STATUSES = extension.WAITING_STATUSES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-leaderboard-task-id",
        default=DEFAULT_CANONICAL_LEADERBOARD_TASK_ID,
    )
    parser.add_argument("--candidate-evaluation-task-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=168.0)
    return parser


def _wait_for_completed(
    dependencies: Sequence[tuple[str, object]],
    *,
    deadline: float,
    poll_seconds: float,
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    while True:
        pending = False
        for context, task in dependencies:
            extension._reload(task, context=context)
            status = extension._status(task)
            if status == "completed":
                continue
            if status in FAILED_STATUSES:
                raise extension.P1CheckpointReuseError(f"{context} ended as {status!r}")
            if status not in WAITING_STATUSES:
                raise extension.P1CheckpointReuseError(
                    f"{context} has unexpected status {status!r}"
                )
            pending = True
        if not pending:
            return
        if monotonic() >= deadline:
            raise TimeoutError("timed out waiting for candidate-extension dependencies")
        sleeper(poll_seconds)


def validate_canonical_leaderboard(
    payload: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], str]:
    """Apply the original selector's exact 26-row parser without mutation."""

    results, seal, _, _, _, _ = formal._validate_leaderboard(payload)
    if list(results) != list(formal.SUBJECT_ORDER):
        raise extension.P1CheckpointReuseError(
            "canonical leaderboard subject order changed"
        )
    if extension.ZERO_SHOT_CANDIDATE_IDENTITY in results:
        raise extension.P1CheckpointReuseError(
            "zero-shot candidate leaked into the trained 26-method leaderboard"
        )
    return results, seal


def _validate_candidate_binding(value: Mapping[str, object]) -> str:
    seal = extension._require_seal(value, context="checkpoint reuse binding")
    alias_mapping_seal = extension.validate_identity_alias_mapping(
        extension.identity_alias_mapping()
    )
    expected = {
        "candidate_identity": extension.ZERO_SHOT_CANDIDATE_IDENTITY,
        "identity_alias_mapping_seal_sha256": alias_mapping_seal,
        "evidence_class": extension.ZERO_SHOT_EVIDENCE_CLASS,
        "config_subject": extension.P1_CONFIG_SUBJECT,
        "checkpoint_subject": extension.P0_CONFIG_SUBJECT,
        "checkpoint_policy": extension.CHECKPOINT_POLICY,
        "checkpoint_role": extension.CHECKPOINT_ROLE,
        "weights_retrained": False,
        "optimization_origin": extension.OPTIMIZATION_ORIGIN,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise extension.P1CheckpointReuseError(
                f"checkpoint reuse binding {key!r} mismatch"
            )
    source_task = value.get("source_task")
    checkpoint = value.get("checkpoint")
    evaluation = value.get("formal_evaluation")
    if not all(
        isinstance(item, Mapping) for item in (source_task, checkpoint, evaluation)
    ):
        raise extension.P1CheckpointReuseError(
            "checkpoint reuse binding nested contracts are invalid"
        )
    assert isinstance(source_task, Mapping)
    assert isinstance(checkpoint, Mapping)
    assert isinstance(evaluation, Mapping)
    if (
        source_task.get("task_id") != extension.P0_TASK_ID
        or source_task.get("required_status") != "completed"
        or source_task.get("training_seed") != extension.TRAINING_SEED
        or source_task.get("training_overlay_protocol_seed")
        != extension.TRAINING_OVERLAY_PROTOCOL_SEED
        or checkpoint.get("model_name") != extension.P0_FINAL_MODEL_NAME
        or checkpoint.get("source_filename") != extension.P0_FINAL_SOURCE_FILENAME
        or checkpoint.get("remote_filename") != extension.P0_FINAL_REMOTE_FILENAME
        or checkpoint.get("bytes_recomputed") is not True
        or evaluation.get("protocol_id") != extension.PROTOCOL_ID
        or evaluation.get("sample_count") != extension.SAMPLE_COUNT
        or evaluation.get("run_count") != extension.RUN_COUNT
    ):
        raise extension.P1CheckpointReuseError(
            "checkpoint reuse binding canonical provenance drifted"
        )
    extension._sha256(source_task.get("run_contract_sha256"), context="P0 run contract")
    extension._sha256(
        source_task.get("initialization_audit_sha256"),
        context="P0 initialization audit",
    )
    extension._task_id(checkpoint.get("model_id"), context="P0 final model")
    extension._sha256(checkpoint.get("sha256"), context="P0 final checkpoint")
    extension._positive_int(
        checkpoint.get("size_bytes"), context="P0 final checkpoint bytes"
    )
    extension._trusted_model_url(
        checkpoint.get("model_url"), context="P0 final checkpoint"
    )
    return seal


def _validate_equivalence(
    value: Mapping[str, object],
    *,
    binding_seal: str,
) -> str:
    seal = extension._require_seal(value, context="checkpoint reuse equivalence")
    expected = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_checkpoint_reuse_equivalence",
        "passed": True,
        "candidate_identity": extension.ZERO_SHOT_CANDIDATE_IDENTITY,
        "identity_alias_mapping_seal_sha256": (
            extension.identity_alias_mapping()["seal_sha256"]
        ),
        "evidence_class": extension.ZERO_SHOT_EVIDENCE_CLASS,
        "weights_retrained": False,
        "optimization_origin": extension.OPTIMIZATION_ORIGIN,
        "config_subject": extension.P1_CONFIG_SUBJECT,
        "checkpoint_subject": extension.P0_CONFIG_SUBJECT,
        "checkpoint_reuse_binding_seal_sha256": binding_seal,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise extension.P1CheckpointReuseError(
                f"checkpoint reuse equivalence {key!r} mismatch"
            )
    config = value.get("config_equivalence")
    state = value.get("state_schema")
    strict_load = value.get("strict_load")
    cuda = value.get("target_cuda")
    if not all(
        isinstance(item, Mapping) for item in (config, state, strict_load, cuda)
    ):
        raise extension.P1CheckpointReuseError(
            "checkpoint reuse equivalence subcontracts are invalid"
        )
    assert isinstance(config, Mapping)
    assert isinstance(state, Mapping)
    assert isinstance(strict_load, Mapping)
    assert isinstance(cuda, Mapping)
    if (
        config.get("only_deployment_model_delta") != "support_residual_reliability_gate"
        or config.get("p0_value") is not False
        or config.get("p1_value") is not True
        or state.get("missing_student_keys") != []
        or not isinstance(state.get("extra_checkpoint_keys"), list)
        or strict_load.get("strict_load") is not True
        or strict_load.get("missing_keys") != []
        or strict_load.get("unexpected_keys") != []
        or strict_load.get("loaded_tensor_mapping_equal") is not True
        or cuda.get("clean_fused_bitwise_equal") is not True
        or cuda.get("clean_routing_weights_bitwise_equal") is not True
        or cuda.get("fault_gate_activated") is not True
        or cuda.get("fault_forward_finite") is not True
        or cuda.get("fault_backward_finite") is not True
    ):
        raise extension.P1CheckpointReuseError(
            "checkpoint reuse equivalence did not prove strict reuse"
        )
    extras = state["extra_checkpoint_keys"]
    if not extras or any(
        type(key) is not str or not key.startswith("teacher.teacher.") for key in extras
    ):
        raise extension.P1CheckpointReuseError(
            "checkpoint reuse equivalence nested-teacher filter is invalid"
        )
    return seal


def _validate_receipt(
    value: Mapping[str, object],
    *,
    task_id: str,
    binding_seal: str,
    equivalence_seal: str,
) -> tuple[str, str]:
    seal = extension._require_seal(value, context="checkpoint reuse receipt")
    expected = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_checkpoint_reuse_evaluation_receipt",
        "task_id": task_id,
        "candidate_identity": extension.ZERO_SHOT_CANDIDATE_IDENTITY,
        "identity_alias_mapping_seal_sha256": (
            extension.identity_alias_mapping()["seal_sha256"]
        ),
        "config_subject": extension.P1_CONFIG_SUBJECT,
        "checkpoint_subject": extension.P0_CONFIG_SUBJECT,
        "checkpoint_reuse_binding_seal_sha256": binding_seal,
        "checkpoint_reuse_equivalence_seal_sha256": equivalence_seal,
        "complete": True,
        "run_count": extension.RUN_COUNT,
        "sample_count_per_run": extension.SAMPLE_COUNT,
        "weights_retrained": False,
        "optimization_origin": extension.OPTIMIZATION_ORIGIN,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise extension.P1CheckpointReuseError(
                f"checkpoint reuse receipt {key!r} mismatch"
            )
    metrics_sha = extension._sha256(
        value.get("metrics_sha256"), context="checkpoint reuse metrics"
    )
    extension._sha256(
        value.get("run_contract_seal_sha256"),
        context="checkpoint reuse run contract",
    )
    return seal, metrics_sha


def _input_model_ids(task: object) -> list[str]:
    models = extension._models(task, context="P1 zero-shot evaluation task")
    inputs = models.get("input")
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise extension.P1CheckpointReuseError(
            "P1 zero-shot evaluation input models are invalid"
        )
    return [
        extension._task_id(getattr(model, "id", None), context="evaluation input model")
        for model in inputs
    ]


def validate_candidate_evaluation(
    task: object,
    *,
    expected_task_id: str,
) -> tuple[
    dict[str, object],
    dict[tuple[int, str], dict[str, float]],
    dict[str, object],
]:
    task_id = extension._task_id(expected_task_id, context="candidate evaluation")
    extension._reload(task, context="P1 zero-shot evaluation task")
    if (
        extension._task_id(getattr(task, "id", None), context="candidate evaluation")
        != task_id
        or extension._status(task) != "completed"
    ):
        raise extension.P1CheckpointReuseError(
            "P1 zero-shot evaluation task identity/status mismatch"
        )
    parent = extension._task_parent(task)
    if parent != extension.P0_TASK_ID:
        raise extension.P1CheckpointReuseError(
            "P1 zero-shot evaluation parent is not P0"
        )
    alias_mapping = extension._artifact_mapping(
        task,
        extension.IDENTITY_ALIAS_MAPPING_ARTIFACT,
        context="P1 zero-shot evaluation task",
    )
    alias_mapping_seal = extension.validate_identity_alias_mapping(alias_mapping)
    binding = dict(
        extension._artifact_mapping(
            task,
            extension.CHECKPOINT_REUSE_BINDING_ARTIFACT,
            context="P1 zero-shot evaluation task",
        )
    )
    binding_seal = _validate_candidate_binding(binding)
    if binding.get("identity_alias_mapping_seal_sha256") != alias_mapping_seal:
        raise extension.P1CheckpointReuseError(
            "checkpoint binding is not cross-bound to the identity alias mapping"
        )
    extension.validate_evaluation_parameters(
        extension._parameters(task, context="P1 zero-shot evaluation task"),
        binding,
    )
    checkpoint = binding["checkpoint"]
    assert isinstance(checkpoint, Mapping)
    if _input_model_ids(task) != [checkpoint["model_id"]]:
        raise extension.P1CheckpointReuseError(
            "P1 zero-shot evaluation input model does not equal P0 final"
        )
    equivalence = extension._artifact_mapping(
        task,
        extension.CHECKPOINT_REUSE_EQUIVALENCE_ARTIFACT,
        context="P1 zero-shot evaluation task",
    )
    equivalence_seal = _validate_equivalence(
        equivalence,
        binding_seal=binding_seal,
    )
    receipt = extension._artifact_mapping(
        task,
        extension.CHECKPOINT_REUSE_EVALUATION_RECEIPT_ARTIFACT,
        context="P1 zero-shot evaluation task",
    )
    receipt_seal, expected_metrics_sha = _validate_receipt(
        receipt,
        task_id=task_id,
        binding_seal=binding_seal,
        equivalence_seal=equivalence_seal,
    )
    metrics = extension._artifact_mapping(
        task,
        extension.CONTROLLED_METRICS_ARTIFACT,
        context="P1 zero-shot evaluation task",
    )
    if extension._content_sha256(metrics) != expected_metrics_sha:
        raise extension.P1CheckpointReuseError(
            "P1 zero-shot evaluation metrics content SHA-256 mismatch"
        )
    runs = formal._parse_metrics(
        metrics,
        subject=extension.P1_CONFIG_SUBJECT,
        checkpoint_sha256=str(checkpoint["sha256"]),
    )
    provenance = {
        "evaluation_task_id": task_id,
        "identity_alias_mapping_seal_sha256": alias_mapping_seal,
        "checkpoint_reuse_binding_seal_sha256": binding_seal,
        "checkpoint_reuse_equivalence_seal_sha256": equivalence_seal,
        "evaluation_receipt_seal_sha256": receipt_seal,
        "metrics_sha256": expected_metrics_sha,
    }
    return binding, runs, provenance


def load_canonical_baseline_runs(
    *,
    task_class: object,
    canonical_results: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[tuple[int, str], dict[str, float]]]:
    result: dict[str, dict[tuple[int, str], dict[str, float]]] = {}
    for subject in formal.BASELINE_SUBJECTS:
        row = canonical_results[subject]
        task_id = str(row["evaluation_task_id"])
        task = task_class.get_task(task_id=task_id)
        extension._reload(task, context=f"canonical {subject} evaluation")
        if (
            extension._task_id(
                getattr(task, "id", None), context=f"canonical {subject} evaluation"
            )
            != task_id
            or extension._status(task) != "completed"
        ):
            raise extension.P1CheckpointReuseError(
                f"canonical {subject} evaluation identity/status drifted"
            )
        formal._validate_live_evaluation_provenance(
            task,
            subject=subject,
            leaderboard=row,
        )
        metrics = extension._artifact_mapping(
            task,
            extension.CONTROLLED_METRICS_ARTIFACT,
            context=f"canonical {subject} evaluation",
        )
        runs = formal._parse_metrics(
            metrics,
            subject=subject,
            checkpoint_sha256=str(row["training_checkpoint_sha256"]),
        )
        summaries = {
            key: formal._metric_summary(runs, key) for key in formal.AP_METRIC_KEYS
        }
        if formal._exact_json_equal(summaries, row["metrics"]) is not True:
            raise extension.P1CheckpointReuseError(
                f"canonical {subject} leaderboard summary drifted"
            )
        result[subject] = runs
    return result


def _candidate_screening(
    candidate_runs: Mapping[tuple[int, str], Mapping[str, float]],
    baseline_runs: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
) -> dict[str, object]:
    return sota_gate.evaluate_candidate(
        extension.ZERO_SHOT_CANDIDATE_IDENTITY,
        candidate_runs,
        baseline_runs,
    )


def build_extension_leaderboard(
    *,
    canonical_leaderboard_task_id: str,
    canonical_leaderboard: Mapping[str, object],
    candidate_evaluation_task_id: str,
    candidate_binding: Mapping[str, object],
    candidate_runs: Mapping[tuple[int, str], Mapping[str, float]],
    candidate_provenance: Mapping[str, object],
    canonical_baseline_runs: Mapping[
        str, Mapping[tuple[int, str], Mapping[str, float]]
    ],
) -> dict[str, object]:
    canonical_task_id = extension._task_id(
        canonical_leaderboard_task_id,
        context="canonical leaderboard task",
    )
    candidate_task_id = extension._task_id(
        candidate_evaluation_task_id,
        context="candidate evaluation task",
    )
    canonical_results, canonical_seal = validate_canonical_leaderboard(
        canonical_leaderboard
    )
    if tuple(canonical_baseline_runs) != formal.BASELINE_SUBJECTS:
        raise extension.P1CheckpointReuseError(
            "canonical baseline run set is incomplete"
        )
    binding_seal = _validate_candidate_binding(candidate_binding)
    extension._require_exact_keys(
        candidate_provenance,
        {
            "evaluation_task_id",
            "identity_alias_mapping_seal_sha256",
            "checkpoint_reuse_binding_seal_sha256",
            "checkpoint_reuse_equivalence_seal_sha256",
            "evaluation_receipt_seal_sha256",
            "metrics_sha256",
        },
        context="candidate evaluation provenance",
    )
    if (
        candidate_provenance.get("evaluation_task_id") != candidate_task_id
        or candidate_provenance.get("identity_alias_mapping_seal_sha256")
        != candidate_binding.get("identity_alias_mapping_seal_sha256")
        or candidate_provenance.get("checkpoint_reuse_binding_seal_sha256")
        != binding_seal
    ):
        raise extension.P1CheckpointReuseError(
            "candidate evaluation provenance is not cross-bound"
        )
    for key in (
        "identity_alias_mapping_seal_sha256",
        "checkpoint_reuse_equivalence_seal_sha256",
        "evaluation_receipt_seal_sha256",
        "metrics_sha256",
    ):
        extension._sha256(
            candidate_provenance.get(key), context=f"candidate provenance {key}"
        )
    checkpoint = candidate_binding["checkpoint"]
    assert isinstance(checkpoint, Mapping)
    summaries = {
        key: formal._metric_summary(candidate_runs, key)
        for key in formal.AP_METRIC_KEYS
    }
    ordered_runs = [
        {
            "condition_id": (
                f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
            ),
            "delay_ms": delay,
            "condition": condition,
            "metrics": dict(candidate_runs[(delay, condition)]),
        }
        for delay in formal.DELAYS_MS
        for condition in formal.CONDITIONS
    ]
    screening = _candidate_screening(candidate_runs, canonical_baseline_runs)
    return extension._sealed(
        {
            "schema_version": 2,
            "leaderboard_type": EXTENSION_LEADERBOARD_TYPE,
            "protocol_id": extension.PROTOCOL_ID,
            "sample_count": extension.SAMPLE_COUNT,
            "ground_truth_count": extension.GROUND_TRUTH_COUNT,
            "unsupported_sample_count": 0,
            "delays_ms": list(extension.DELAYS_MS),
            "conditions": list(extension.CONDITIONS),
            "run_count_per_subject": extension.RUN_COUNT,
            "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
            "canonical_26": {
                "leaderboard_task_id": canonical_task_id,
                "artifact": CANONICAL_LEADERBOARD_ARTIFACT,
                "seal_sha256": canonical_seal,
                "subject_order": list(formal.SUBJECT_ORDER),
                "subject_count": len(formal.SUBJECT_ORDER),
                "mutated": False,
            },
            "extension_subject_order": [extension.ZERO_SHOT_CANDIDATE_IDENTITY],
            "extension_subject_count": 1,
            "identity_alias_mapping": {
                "artifact": extension.IDENTITY_ALIAS_MAPPING_ARTIFACT,
                "seal_sha256": candidate_provenance[
                    "identity_alias_mapping_seal_sha256"
                ],
                "canonical_identity": extension.ZERO_SHOT_CANDIDATE_IDENTITY,
                "display_subject_alias": extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS,
                "mapping_cardinality": "one_to_one",
                "alias_is_candidate_identity": False,
            },
            "baseline_subjects": list(formal.BASELINE_SUBJECTS),
            "metric_keys": list(formal.AP_METRIC_KEYS),
            "candidate": {
                "subject": extension.ZERO_SHOT_CANDIDATE_IDENTITY,
                "display_subject_alias": extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS,
                "identity_alias_mapping_seal_sha256": candidate_provenance[
                    "identity_alias_mapping_seal_sha256"
                ],
                "kind": "diagnostic_candidate_extension",
                "evidence_class": extension.ZERO_SHOT_EVIDENCE_CLASS,
                "weights_retrained": False,
                "optimization_origin": extension.OPTIMIZATION_ORIGIN,
                "eligible_for_trained_26_method_table": False,
                "config_subject": extension.P1_CONFIG_SUBJECT,
                "checkpoint_subject": extension.P0_CONFIG_SUBJECT,
                "source_task_id": extension.P0_TASK_ID,
                "source_model_id": checkpoint["model_id"],
                "source_checkpoint_sha256": checkpoint["sha256"],
                "source_checkpoint_size_bytes": checkpoint["size_bytes"],
                "checkpoint_reuse_binding_seal_sha256": binding_seal,
                "checkpoint_reuse_equivalence_seal_sha256": candidate_provenance[
                    "checkpoint_reuse_equivalence_seal_sha256"
                ],
                "evaluation_receipt_seal_sha256": candidate_provenance[
                    "evaluation_receipt_seal_sha256"
                ],
                "metrics_sha256": candidate_provenance["metrics_sha256"],
                "evaluation_task_id": candidate_task_id,
                "metrics": summaries,
                "runs": ordered_runs,
            },
            "candidate_screening": screening,
            "canonical_result_count_revalidated": len(canonical_results),
            "conclusion_scope": (
                "single-seed inference-only checkpoint-reuse screening; "
                "requires independent P1 training and formal evaluation"
            ),
        }
    )


def _publish(task: object, payload: Mapping[str, object]) -> None:
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, Mapping) and EXTENSION_LEADERBOARD_ARTIFACT in artifacts:
        observed = extension._artifact_mapping(
            task,
            EXTENSION_LEADERBOARD_ARTIFACT,
            context="candidate-extension leaderboard task",
        )
        if extension._canonical_json(observed) != extension._canonical_json(payload):
            raise extension.P1CheckpointReuseError(
                "existing candidate-extension leaderboard drifted"
            )
        return
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader) or not uploader(
        EXTENSION_LEADERBOARD_ARTIFACT,
        artifact_object=dict(payload),
        wait_on_upload=True,
    ):
        raise extension.P1CheckpointReuseError(
            "failed to publish candidate-extension leaderboard"
        )
    flusher = getattr(task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)
    extension._reload(task, context="candidate-extension leaderboard task")
    observed = extension._artifact_mapping(
        task,
        EXTENSION_LEADERBOARD_ARTIFACT,
        context="candidate-extension leaderboard task",
    )
    if extension._canonical_json(observed) != extension._canonical_json(payload):
        raise extension.P1CheckpointReuseError(
            "candidate-extension leaderboard upload readback drifted"
        )


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    if task_class is None:
        raise extension.P1CheckpointReuseError("ClearML runtime is unavailable")
    if (
        type(args.poll_seconds) not in {int, float}
        or not math.isfinite(float(args.poll_seconds))
        or args.poll_seconds <= 0
        or type(args.timeout_hours) not in {int, float}
        or not math.isfinite(float(args.timeout_hours))
        or args.timeout_hours <= 0
    ):
        raise ValueError("poll interval and timeout must be finite and positive")
    canonical_id = extension._task_id(
        args.canonical_leaderboard_task_id,
        context="canonical leaderboard task",
    )
    evaluation_id = extension._task_id(
        args.candidate_evaluation_task_id,
        context="candidate evaluation task",
    )
    if canonical_id == evaluation_id:
        raise ValueError("canonical leaderboard and candidate evaluation must differ")
    if output_task is None:
        current_getter = getattr(task_class, "current_task", None)
        output_task = current_getter() if callable(current_getter) else None
    if output_task is None:
        raise extension.P1CheckpointReuseError(
            "candidate-extension leaderboard requires a current ClearML task"
        )
    canonical_task = task_class.get_task(task_id=canonical_id)
    candidate_task = task_class.get_task(task_id=evaluation_id)
    deadline = monotonic() + float(args.timeout_hours) * 3600.0
    _wait_for_completed(
        (
            ("canonical 26-method leaderboard", canonical_task),
            ("P1 zero-shot evaluation", candidate_task),
        ),
        deadline=deadline,
        poll_seconds=float(args.poll_seconds),
        monotonic=monotonic,
        sleeper=sleeper,
    )
    canonical_payload = extension._artifact_mapping(
        canonical_task,
        CANONICAL_LEADERBOARD_ARTIFACT,
        context="canonical leaderboard task",
    )
    canonical_results, initial_canonical_seal = validate_canonical_leaderboard(
        canonical_payload
    )
    binding, candidate_runs, provenance = validate_candidate_evaluation(
        candidate_task,
        expected_task_id=evaluation_id,
    )
    baseline_runs = load_canonical_baseline_runs(
        task_class=task_class,
        canonical_results=canonical_results,
    )
    payload = build_extension_leaderboard(
        canonical_leaderboard_task_id=canonical_id,
        canonical_leaderboard=canonical_payload,
        candidate_evaluation_task_id=evaluation_id,
        candidate_binding=binding,
        candidate_runs=candidate_runs,
        candidate_provenance=provenance,
        canonical_baseline_runs=baseline_runs,
    )

    # Final live readback closes the initial-read/build publication window.
    extension._reload(canonical_task, context="canonical leaderboard task")
    extension._reload(candidate_task, context="P1 zero-shot evaluation task")
    if (
        extension._status(canonical_task) != "completed"
        or extension._status(candidate_task) != "completed"
    ):
        raise extension.P1CheckpointReuseError(
            "candidate-extension dependency changed status before publication"
        )
    final_canonical = extension._artifact_mapping(
        canonical_task,
        CANONICAL_LEADERBOARD_ARTIFACT,
        context="canonical leaderboard task",
    )
    _, final_canonical_seal = validate_canonical_leaderboard(final_canonical)
    if final_canonical_seal != initial_canonical_seal or extension._canonical_json(
        final_canonical
    ) != extension._canonical_json(canonical_payload):
        raise extension.P1CheckpointReuseError(
            "canonical leaderboard changed before extension publication"
        )
    final_binding, final_runs, final_provenance = validate_candidate_evaluation(
        candidate_task, expected_task_id=evaluation_id
    )
    if (
        extension._canonical_json(final_binding) != extension._canonical_json(binding)
        or extension._canonical_json(final_runs)
        != extension._canonical_json(candidate_runs)
        or extension._canonical_json(final_provenance)
        != extension._canonical_json(provenance)
    ):
        raise extension.P1CheckpointReuseError(
            "candidate evaluation changed before extension publication"
        )
    final_baseline_runs = load_canonical_baseline_runs(
        task_class=task_class,
        canonical_results=canonical_results,
    )
    if extension._canonical_json(final_baseline_runs) != extension._canonical_json(
        baseline_runs
    ):
        raise extension.P1CheckpointReuseError(
            "canonical baseline metrics changed before extension publication"
        )
    _publish(output_task, payload)
    return payload


def main() -> int:
    args = _parser().parse_args()
    if Task is None:
        raise extension.P1CheckpointReuseError("ClearML runtime is unavailable")
    task = Task.init(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X P1 zero-shot candidate-extension leaderboard",
        reuse_last_task_id=False,
        output_uri=extension.FILES_SERVER_URI,
    )
    run(args, output_task=task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "EXTENSION_LEADERBOARD_ARTIFACT",
    "EXTENSION_LEADERBOARD_TYPE",
    "build_extension_leaderboard",
    "load_canonical_baseline_runs",
    "run",
    "validate_candidate_evaluation",
    "validate_canonical_leaderboard",
)
