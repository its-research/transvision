#!/usr/bin/env python3
"""Recover the exact candidate controller after the E1 runtime-default failure.

This is a deliberately narrow source-transition recovery.  It only accepts the
known failed controller task, the prior sealed recovery receipt, and the exact
``E1 evaluation parameters drifted`` failure.  Execution resets and reinstalls
that same task ID with the locally prepared controller source; no replacement
task can be created.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

try:
    from tools.resilient_v2x import (
        clearml_formal_candidate_evaluation_queue as candidate_queue,
    )
    from tools.resilient_v2x import (
        deploy_clearml_formal_candidate_evaluation_queue as deploy,
    )
    from tools.resilient_v2x import (
        recover_clearml_exact_eval_chain_runtime_failure as shared,
    )
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import clearml_formal_candidate_evaluation_queue as candidate_queue
    import deploy_clearml_formal_candidate_evaluation_queue as deploy
    import recover_clearml_exact_eval_chain_runtime_failure as shared


class CandidateE1RuntimeDefaultsRecoveryError(RuntimeError):
    """Raised when this exact source-transition recovery cannot be proven."""


TASK_ID = shared.CANDIDATE_CONTROLLER_TASK_ID
PRIOR_SOURCE_SHA256 = (
    "7650962daf1b217c57eb74908f57231e726728c5d079b551e5e5a4f356291f23"
)
PRIOR_PARAMETERS_SHA256 = (
    "3cae444c95ba8da3d288e4273b1baeff345c0482a3199077977cb80cbca8edba"
)
PRIOR_RECOVERY_RECEIPT_TYPE = (
    "resilient_v2x_candidate_e1_runtime_defaults_recovery"
)
PRIOR_RECOVERY_RECEIPT_SEAL_SHA256 = (
    "32291605a241b030fadf62b72842efa06b1d277cbfc9f3711cb68a8933f41c7d"
)
FAILURE_MESSAGE = "artifact evaluation_plan is not a JSON object"
EXECUTE_TOKEN = (
    "RESET_EXACT_CANDIDATE_CONTROLLER_AFTER_E1_RUNTIME_DEFAULTS_"
    "B6F0FBAB32A5478183A45B3CA833FC01"
)
ATTEMPT_RECEIPT_TYPE = (
    "resilient_v2x_candidate_e1_runtime_defaults_recovery_attempt"
)
FINAL_RECEIPT_TYPE = "resilient_v2x_candidate_e1_runtime_defaults_recovery"
E1_TASK_ID = "c6f26cc7902142c090ec238856409ac6"
EXPECTED_NEW_SOURCE_SHA256 = (
    "cb025c7741f38a45f759d1926b0b6bd2e663df285abee4eaff498be28efbdedc"
)
EXPECTED_NEW_PARAMETERS_SHA256 = (
    "e4ab9314f8fe35ea54fbda81d6e07a744a15e12a03ba7828ce6688a1271d02a6"
)
ALLOWED_E1_STATUSES = frozenset({"completed"})
REPAIR_SCOPE = "accept_exact_e1_formal_evidence_serialization_only"
ALLOWED_FAILED_ARTIFACTS = frozenset(
    {candidate_queue.CONTROLLER_ARTIFACT}
)


def _prior_binding(path: Path) -> dict[str, object]:
    value, resolved = shared._read_sealed(
        path, context="prior candidate recovery receipt"
    )
    expected = {
        "schema_version": 1,
        "receipt_type": PRIOR_RECOVERY_RECEIPT_TYPE,
        "status": "candidate_repaired_and_enqueued",
        "remote_state_changed": True,
        "same_id_only": True,
        "replacement_controller_created": False,
        "candidate_controller_task_id": TASK_ID,
        "seal_sha256": PRIOR_RECOVERY_RECEIPT_SEAL_SHA256,
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "prior candidate recovery receipt identity drifted"
        )
    authority = value.get("queued_authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("task_id") != TASK_ID
        or authority.get("source_sha256") != PRIOR_SOURCE_SHA256
        or authority.get("parameters_sha256") != PRIOR_PARAMETERS_SHA256
        or authority.get("queue") != deploy.SERVICES_QUEUE
        or authority.get("queue_id") != deploy.SERVICES_QUEUE_ID
    ):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "prior candidate queued authority drifted"
        )
    return {
        "path": str(resolved),
        "receipt_seal_sha256": value["seal_sha256"],
        "task_id": TASK_ID,
        "source_sha256": PRIOR_SOURCE_SHA256,
        "parameters_sha256": PRIOR_PARAMETERS_SHA256,
    }


def _failure_evidence(task: object) -> dict[str, object]:
    if shared._status(task) != "failed":
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate controller is not failed"
        )
    getter = getattr(task, "get_reported_console_output", None)
    if not callable(getter):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate controller console evidence is unavailable"
        )
    try:
        lines = [str(item) for item in (getter(200) or [])][-200:]
    except Exception as error:
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate controller console evidence cannot be read"
        ) from error
    raw = "\n".join(lines).encode("utf-8")[-262_144:]
    text = raw.decode("utf-8", errors="replace")
    runtime_messages = {
        line.split("RuntimeError:", 1)[1].strip()
        for line in text.splitlines()
        if "RuntimeError:" in line
    }
    data = getattr(task, "data", None)
    if (
        runtime_messages != {FAILURE_MESSAGE}
        or str(getattr(data, "status_reason", "") or "")
        != "worker execution exit code 1"
    ):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate controller failure reason drifted"
        )
    return {
        "content_stored": False,
        "tail_line_count": len(lines),
        "tail_bytes": len(raw),
        "tail_sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_error": FAILURE_MESSAGE,
        "status_reason": "worker execution exit code 1",
    }


def _e1_runtime_authority(task_class: object) -> dict[str, object]:
    task = shared._authoritative_task(
        task_class, E1_TASK_ID, context="E1 runtime-default recovery dependency"
    )
    spec = candidate_queue.CANDIDATES[0]
    if spec.label != "E1":
        raise CandidateE1RuntimeDefaultsRecoveryError("candidate order drifted")
    training = task_class.get_task(task_id=spec.training_task_id)
    template = task_class.get_task(task_id=spec.template_task_id)
    try:
        candidate_queue._validate_template(template, spec)
        candidate_queue._validate_static_training(training, spec)
        binding = candidate_queue._validate_completed_training(training, spec)
        status = candidate_queue._validate_evaluation_identity(
            task, template, spec, binding
        )
    except RuntimeError as error:
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "E1 runtime authority is invalid"
        ) from error
    if status not in ALLOWED_E1_STATUSES:
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "E1 is not running or completed during controller recovery"
        )
    queue_id = str(
        getattr(getattr(getattr(task, "data", None), "execution", None), "queue", "")
        or ""
    )
    return {
        "task_id": E1_TASK_ID,
        "status": status,
        "parent_task_id": spec.training_task_id,
        "queue": spec.queue,
        "queue_id": queue_id,
        "parameters_sha256": shared._content_sha256(
            candidate_queue._normalized_evaluation_parameters(
                candidate_queue._parameters(task)
            )
        ),
    }


def _failed_snapshot(task_class: object) -> dict[str, object]:
    task = shared._authoritative_task(
        task_class, TASK_ID, context="candidate E1-default failure target"
    )
    context = "candidate E1-default failure target"
    if (
        shared._status(task) != "failed"
        or str(getattr(task, "name", "") or "") != deploy.CONTROLLER_NAME
        or shared.successor_deploy._task_parent(task) != deploy.PARENT_TASK_ID
        or shared._exact_project_id(task, context=context) != deploy.PROJECT_ID
        or shared.successor_deploy._project_name(task) != deploy.PROJECT_NAME
        or str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
        != "controller"
    ):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate controller static identity drifted"
        )
    source = shared._source(task, entry_point=deploy.ENTRY_POINT, context=context)
    parameters = shared._parameters(task, context=context)
    queue, queue_id = shared._queue_identity(task, context=context)
    tags = sorted(shared._tags(task, field="tags", context=context))
    system_tags = sorted(
        shared._tags(task, field="system_tags", context=context)
    )
    artifacts = shared._artifact_inventory(task)
    models = shared._model_inventory(task, context=context)
    environment = shared._service_environment(task, context=context)
    execution = shared._execution_environment(task, context=context)
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    parameters_sha = shared._content_sha256(parameters)
    if (
        source_sha != PRIOR_SOURCE_SHA256
        or parameters_sha != PRIOR_PARAMETERS_SHA256
        or queue != deploy.SERVICES_QUEUE
        or queue_id != deploy.SERVICES_QUEUE_ID
        or tags != sorted(deploy.TAGS)
        or system_tags
        or {str(row.get("name") or "") for row in artifacts}
        != ALLOWED_FAILED_ARTIFACTS
        or len(artifacts) != len(ALLOWED_FAILED_ARTIFACTS)
        or models != {"input": [], "output": []}
    ):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate controller failed contract drifted"
        )
    return {
        "task_id": TASK_ID,
        "status": "failed",
        "project_id": deploy.PROJECT_ID,
        "task_name": deploy.CONTROLLER_NAME,
        "task_type": "controller",
        "parent_task_id": deploy.PARENT_TASK_ID,
        "entry_point": deploy.ENTRY_POINT,
        "source_sha256": source_sha,
        "parameters": parameters,
        "parameters_sha256": parameters_sha,
        "queue": queue,
        "queue_id": queue_id,
        "tags": tags,
        "system_tags": system_tags,
        "artifact_inventory": artifacts,
        "model_inventory": models,
        "service_environment": environment,
        "execution_environment": execution,
        "failure_evidence": _failure_evidence(task),
    }


def _target_material(prepared: Mapping[str, object]) -> dict[str, object]:
    source = prepared.get("source")
    parameters = prepared.get("parameters")
    record = prepared.get("source_record")
    if type(source) is not str or not isinstance(parameters, Mapping) or not isinstance(record, Mapping):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "prepared candidate deployment is invalid"
        )
    normalized = {str(key): str(value) for key, value in parameters.items()}
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    parameters_sha = shared._content_sha256(normalized)
    if (
        source_sha != EXPECTED_NEW_SOURCE_SHA256
        or record.get("sha256") != source_sha
        or parameters_sha != EXPECTED_NEW_PARAMETERS_SHA256
        or candidate_queue._RUNTIME_EMPTY_EVALUATION_PARAMETERS
        != {
            "Args/allow_failed_teacher_task": "False",
            "Args/experiment_from_task": "",
            "Args/student_checkpoint": "",
            "Args/student_checkpoint_sha256": "",
            "Args/student_model_id": "",
            "Args/student_task_id": "",
            "Args/teacher_checkpoint": "",
            "Args/teacher_checkpoint_sha256": "",
            "Args/teacher_model_id": "",
            "Args/teacher_task_id": "",
        }
    ):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate controller repair contract drifted"
        )
    return {
        "source": source,
        "source_sha256": source_sha,
        "source_bytes": len(source.encode("utf-8")),
        "parameters": normalized,
        "parameters_sha256": parameters_sha,
        "deployment_seal_sha256": prepared["deployment_seal_sha256"],
        "deployment_contract": prepared["deployment_contract"],
    }


def build_attempt(
    task_class: object, *, prior_recovery_receipt: Path
) -> tuple[dict[str, object], dict[str, object]]:
    prior = _prior_binding(prior_recovery_receipt)
    failed = _failed_snapshot(task_class)
    e1 = _e1_runtime_authority(task_class)
    prepared = deploy._prepare()
    target = _target_material(prepared)
    names = shared._exact_name_inventory(
        task_class,
        task_name=deploy.CONTROLLER_NAME,
        parent_task_id=deploy.PARENT_TASK_ID,
        expected_task_id=TASK_ID,
    )
    attempt = shared._seal(
        {
            "schema_version": 1,
            "receipt_type": ATTEMPT_RECEIPT_TYPE,
            "generated_at_utc": shared._utc_timestamp(),
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_id_only": True,
            "replacement_controller_created": False,
            "repair_scope": REPAIR_SCOPE,
            "prior_candidate_recovery": prior,
            "failed_authority": failed,
            "e1_authority": e1,
            "target_material": target,
            "exact_name_inventory": names,
            "planned_actions": [
                "durably_write_attempt_receipt",
                "revalidate_exact_failed_controller_and_e1_runtime",
                "reset_exact_controller_task_id",
                "install_repaired_standalone_source_and_contract",
                "enqueue_exact_controller_on_services",
            ],
        }
    )
    return attempt, prepared


def _validate_attempt(value: Mapping[str, object]) -> None:
    expected = {
        "schema_version": 1,
        "receipt_type": ATTEMPT_RECEIPT_TYPE,
        "status": "frozen_before_reset",
        "remote_state_changed": False,
        "same_id_only": True,
        "replacement_controller_created": False,
        "repair_scope": REPAIR_SCOPE,
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate repair attempt identity drifted"
        )
    if shared._seal(value) != dict(value):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate repair attempt seal drifted"
        )
    failed = value.get("failed_authority")
    target = value.get("target_material")
    if (
        not isinstance(failed, Mapping)
        or failed.get("task_id") != TASK_ID
        or failed.get("source_sha256") != PRIOR_SOURCE_SHA256
        or failed.get("parameters_sha256") != PRIOR_PARAMETERS_SHA256
        or not isinstance(target, Mapping)
        or target.get("source_sha256") != EXPECTED_NEW_SOURCE_SHA256
        or target.get("parameters_sha256") != EXPECTED_NEW_PARAMETERS_SHA256
        or value.get("exact_name_inventory") != [TASK_ID]
    ):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate repair attempt material drifted"
        )


def _validate_failed_matches(
    task_class: object, frozen: Mapping[str, object]
) -> object:
    current = _failed_snapshot(task_class)
    if current != dict(frozen):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate controller changed after repair attempt freeze"
        )
    return shared._authoritative_task(
        task_class, TASK_ID, context="candidate repair immediate reset target"
    )


def _wait_created(
    task_class: object, *, sleeper: Callable[[float], None]
) -> object:
    last = ""
    for index in range(5):
        task = shared._authoritative_task(
            task_class, TASK_ID, context="candidate repair reset readback"
        )
        last = shared._status(task)
        if last == "created":
            return task
        if index != 4:
            sleeper(1.0)
    raise CandidateE1RuntimeDefaultsRecoveryError(
        f"candidate controller did not reach created after reset: {last!r}"
    )


def _validate_created_shell(task: object) -> None:
    context = "candidate repair created shell"
    queue, queue_id = shared._queue_identity(task, context=context)
    if (
        shared._status(task) != "created"
        or queue
        or queue_id
        or shared._artifact_inventory(task)
        or shared._model_inventory(task, context=context)
        != {"input": [], "output": []}
        or str(getattr(task, "name", "") or "") != deploy.CONTROLLER_NAME
        or shared.successor_deploy._task_parent(task) != deploy.PARENT_TASK_ID
    ):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate reset did not produce the exact empty created shell"
        )


def execute_recovery(
    task_class: object,
    *,
    attempt: Mapping[str, object],
    attempt_path: Path,
    prepared: Mapping[str, object],
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    _validate_attempt(attempt)
    frozen, resolved = shared._read_sealed(
        attempt_path, context="candidate repair attempt receipt"
    )
    if resolved != attempt_path.resolve() or frozen != dict(attempt):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate repair attempt was not durably frozen"
        )
    e1_before = _e1_runtime_authority(task_class)
    frozen_e1 = attempt["e1_authority"]
    if (
        e1_before.get("task_id") != frozen_e1.get("task_id")
        or e1_before.get("parent_task_id") != frozen_e1.get("parent_task_id")
        or e1_before.get("queue_id") != frozen_e1.get("queue_id")
        or e1_before.get("parameters_sha256")
        != frozen_e1.get("parameters_sha256")
        or (
            frozen_e1.get("status") == "completed"
            and e1_before.get("status") != "completed"
        )
    ):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "E1 authority changed non-monotonically after repair freeze"
        )
    task = _validate_failed_matches(task_class, attempt["failed_authority"])
    resetter = getattr(task, "reset", None)
    if not callable(resetter):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate controller cannot be reset"
        )
    callback_error: Exception | None = None
    try:
        if resetter(force=True) is False:
            callback_error = CandidateE1RuntimeDefaultsRecoveryError(
                "candidate controller reset callback rejected"
            )
    except Exception as error:
        callback_error = error
    try:
        task = _wait_created(task_class, sleeper=sleeper)
    except Exception as error:
        if callback_error is not None:
            raise CandidateE1RuntimeDefaultsRecoveryError(
                "candidate controller reset authority readback failed"
            ) from callback_error
        raise error
    _validate_created_shell(task)
    deploy._configure_shell(task, prepared=prepared)
    created = deploy._validate_task(
        task_class,
        task_id=TASK_ID,
        prepared=prepared,
        expected_statuses=("created",),
    )
    if shared._exact_name_inventory(
        task_class,
        task_name=deploy.CONTROLLER_NAME,
        parent_task_id=deploy.PARENT_TASK_ID,
        expected_task_id=TASK_ID,
    ) != attempt["exact_name_inventory"]:
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate repair created a replacement task"
        )
    queued = deploy._enqueue(
        task_class,
        task,
        task_id=TASK_ID,
        prepared=prepared,
        sleeper=sleeper,
    )
    e1_after = _e1_runtime_authority(task_class)
    names = shared._exact_name_inventory(
        task_class,
        task_name=deploy.CONTROLLER_NAME,
        parent_task_id=deploy.PARENT_TASK_ID,
        expected_task_id=TASK_ID,
    )
    if names != attempt["exact_name_inventory"]:
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "candidate repair changed the exact-name inventory"
        )
    return shared._seal(
        {
            "schema_version": 1,
            "receipt_type": FINAL_RECEIPT_TYPE,
            "generated_at_utc": shared._utc_timestamp(),
            "status": "candidate_repaired_and_enqueued",
            "remote_state_changed": True,
            "same_id_only": True,
            "replacement_controller_created": False,
            "repair_scope": REPAIR_SCOPE,
            "attempt_receipt_path": str(attempt_path.resolve()),
            "attempt_receipt_seal_sha256": attempt["seal_sha256"],
            "prior_candidate_recovery": attempt["prior_candidate_recovery"],
            "candidate_controller_task_id": TASK_ID,
            "source_transition": {
                "from_sha256": PRIOR_SOURCE_SHA256,
                "to_sha256": EXPECTED_NEW_SOURCE_SHA256,
            },
            "parameter_transition": {
                "from_sha256": PRIOR_PARAMETERS_SHA256,
                "to_sha256": EXPECTED_NEW_PARAMETERS_SHA256,
            },
            "failure_evidence": attempt["failed_authority"]["failure_evidence"],
            "created_unqueued_authority": created,
            "queued_authority": queued,
            "e1_authority_before_reset": e1_before,
            "e1_authority_after_enqueue": e1_after,
            "exact_name_inventory": names,
            "e1_mutated": False,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--prior-candidate-recovery-receipt", type=Path, required=True)
    parser.add_argument("--attempt-receipt", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.execute_token != EXECUTE_TOKEN:
        raise CandidateE1RuntimeDefaultsRecoveryError(
            f"exact execute token required: {EXECUTE_TOKEN}"
        )
    if not args.execute and args.execute_token:
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "--execute-token requires --execute"
        )
    if args.execute != (args.attempt_receipt is not None):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "--attempt-receipt is required exactly with --execute"
        )
    outputs = [args.receipt]
    if args.attempt_receipt is not None:
        outputs.append(args.attempt_receipt)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise CandidateE1RuntimeDefaultsRecoveryError(
            "attempt and final receipt paths must differ"
        )
    for path in outputs:
        shared._require_new(path, context="candidate repair output")
    try:
        from clearml import Task
    except ImportError as error:
        raise CandidateE1RuntimeDefaultsRecoveryError("ClearML is required") from error
    attempt, prepared = build_attempt(
        Task, prior_recovery_receipt=args.prior_candidate_recovery_receipt
    )
    if not args.execute:
        plan = shared._seal(
            {
                **attempt,
                "receipt_type": "resilient_v2x_candidate_e1_runtime_defaults_recovery_plan",
                "status": "planned",
                "execute_token": EXECUTE_TOKEN,
            }
        )
        shared._write_new(args.receipt, plan)
        print(
            f"candidate controller repair plan: {args.receipt.resolve()} "
            f"seal={plan['seal_sha256']}"
        )
        return 0
    assert args.attempt_receipt is not None
    shared._write_new(args.attempt_receipt, attempt)
    result = execute_recovery(
        Task,
        attempt=attempt,
        attempt_path=args.attempt_receipt,
        prepared=prepared,
    )
    shared._write_new(args.receipt, result)
    print(
        f"candidate controller repaired: {args.receipt.resolve()} "
        f"task={TASK_ID} status={result['queued_authority']['status']} "
        f"seal={result['seal_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
