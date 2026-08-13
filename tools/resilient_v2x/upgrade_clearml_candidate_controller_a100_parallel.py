#!/usr/bin/env python3
"""Upgrade the exact candidate controller to the sealed A100 parallel policy."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from tools.resilient_v2x import clearml_formal_candidate_evaluation_queue as queue
    from tools.resilient_v2x import deploy_clearml_formal_candidate_evaluation_queue as deploy
    from tools.resilient_v2x import recover_clearml_exact_eval_chain_runtime_failure as shared
except ModuleNotFoundError:
    import clearml_formal_candidate_evaluation_queue as queue
    import deploy_clearml_formal_candidate_evaluation_queue as deploy
    import recover_clearml_exact_eval_chain_runtime_failure as shared


TASK_ID = "b6f0fbab32a5478183a45b3ca833fc01"
E2_TASK_ID = "c337353b188c41cba2ad92fcaabcaf35"
E3_TASK_ID = "27a82d39e6354697ad1532b6399bbde2"
PRIOR_SOURCE_SHA256 = "cb025c7741f38a45f759d1926b0b6bd2e663df285abee4eaff498be28efbdedc"
PRIOR_PARAMETERS_SHA256 = "e4ab9314f8fe35ea54fbda81d6e07a744a15e12a03ba7828ce6688a1271d02a6"
TARGET_SOURCE_SHA256 = "d57e2d82ad29d5323455b0d91ee24974d0f81c08b2128729606a58aa2e3c7a55"
TARGET_PARAMETERS_SHA256 = "1a820ca943d4bfd62b8e34c9bd58cc337759afe76a5a1d0ba12eddfb7fd5fe6a"
RECOVERY_SOURCE_SHA256 = "1fe760c9461c1a4f98d157597ba47f581f527e2790c7e368c00dd65c3a9599f8"
RECOVERY_PARAMETERS_SHA256 = "37d23b4e1a969999fa36ec89a95af8929268da6dd8ebf93ccc3134e6ef462d41"
EXECUTE_TOKEN = "UPGRADE_EXACT_CANDIDATE_CONTROLLER_A100_PARALLEL_B6F0FBAB32A5478183A45B3CA833FC01"


class UpgradeError(RuntimeError):
    pass


def _controller(task_class: object) -> tuple[object, dict[str, object]]:
    task = shared._authoritative_task(task_class, TASK_ID, context="candidate controller")
    source = shared._source(task, entry_point=deploy.ENTRY_POINT, context="candidate controller")
    parameters = shared._parameters(task, context="candidate controller")
    queue_name, queue_id = shared._queue_identity(task, context="candidate controller")
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    parameters_sha = shared._content_sha256(parameters)
    names = shared._exact_name_inventory(
        task_class,
        task_name=deploy.CONTROLLER_NAME,
        parent_task_id=deploy.PARENT_TASK_ID,
        expected_task_id=TASK_ID,
    )
    if (
        shared._status(task) != "in_progress"
        or source_sha != PRIOR_SOURCE_SHA256
        or parameters_sha != PRIOR_PARAMETERS_SHA256
        or (queue_name, queue_id) != (deploy.SERVICES_QUEUE, deploy.SERVICES_QUEUE_ID)
        or names != [TASK_ID]
        or shared._model_inventory(task, context="candidate controller") != {"input": [], "output": []}
    ):
        raise UpgradeError("candidate controller pre-upgrade authority drifted")
    return task, {
        "task_id": TASK_ID,
        "status": "in_progress",
        "source_sha256": source_sha,
        "parameters_sha256": parameters_sha,
        "queue": queue_name,
        "queue_id": queue_id,
        "artifact_inventory": shared._artifact_inventory(task),
        "exact_name_inventory": names,
    }


def _children(task_class: object) -> dict[str, object]:
    e2 = shared._authoritative_task(task_class, E2_TASK_ID, context="E2 evaluation")
    e3 = shared._authoritative_task(task_class, E3_TASK_ID, context="E3 evaluation")
    if shared._status(e2) != "in_progress" or shared._status(e3) != "created":
        raise UpgradeError("E2/E3 authority is not safe for parallel upgrade")
    if str(getattr(getattr(e2, "data", None), "last_worker", "") or "") != "10.100.34.26-V100:gpu4,5,6,7":
        raise UpgradeError("E2 worker drifted")
    execution = getattr(getattr(e2, "data", None), "execution", None)
    if str(getattr(execution, "queue", "") or "") != queue.QUEUE_IDS["GPU4-V100"]:
        raise UpgradeError("E2 execution queue drifted")
    e3_execution = getattr(getattr(e3, "data", None), "execution", None)
    if str(getattr(e3_execution, "queue", "") or ""):
        raise UpgradeError("E3 created shell already has a queue")
    return {
        "E2": {"task_id": E2_TASK_ID, "status": "in_progress", "worker": "10.100.34.26-V100:gpu4,5,6,7"},
        "E3": {"task_id": E3_TASK_ID, "status": "created", "queue": None},
    }


def _recover_failed(task_class: object, attempt_path: Path, receipt_path: Path) -> None:
    task = shared._authoritative_task(task_class, TASK_ID, context="failed candidate controller")
    source = shared._source(task, entry_point=deploy.ENTRY_POINT, context="failed candidate controller")
    parameters = shared._parameters(task, context="failed candidate controller")
    if (
        shared._status(task) != "failed"
        or hashlib.sha256(source.encode()).hexdigest() != TARGET_SOURCE_SHA256
        or shared._content_sha256(parameters) != TARGET_PARAMETERS_SHA256
    ):
        raise UpgradeError("failed controller authority drifted")
    children = {}
    expected = {
        E2_TASK_ID: ("in_progress", queue.QUEUE_IDS["GPU4-V100"]),
        E3_TASK_ID: ("in_progress", queue.QUEUE_IDS["GPU4-A100"]),
        "8d7d39dcc540475a8827d3cac4b9068e": ("created", ""),
        "908fe86179a24a64811d92b3af4d903f": ("in_progress", queue.QUEUE_IDS["GPU4-A100"]),
    }
    for task_id, (status, queue_id) in expected.items():
        child = shared._authoritative_task(task_class, task_id, context="candidate child")
        observed = (
            shared._status(child),
            str(getattr(getattr(getattr(child, "data", None), "execution", None), "queue", "") or ""),
        )
        if observed != (status, queue_id):
            raise UpgradeError("candidate child authority drifted")
        children[task_id] = {"status": status, "queue_id": queue_id}
    prepared = deploy._prepare()
    target_source = str(prepared["source_record"]["sha256"])
    target_parameters = deploy._content_sha256(deploy._normalized_parameters(prepared["parameters"]))
    if target_source != RECOVERY_SOURCE_SHA256 or target_parameters != RECOVERY_PARAMETERS_SHA256:
        raise UpgradeError("recovery target drifted")
    attempt = shared._seal({
        "schema_version": 1,
        "receipt_type": "resilient_v2x_candidate_a100_parallel_readback_recovery_attempt",
        "generated_at_utc": shared._utc_timestamp(),
        "status": "frozen_before_reset",
        "remote_state_changed": False,
        "same_id_only": True,
        "replacement_controller_created": False,
        "controller_source_sha256": TARGET_SOURCE_SHA256,
        "controller_parameters_sha256": TARGET_PARAMETERS_SHA256,
        "runtime_error": "E3 evaluation queue drifted",
        "child_tasks": children,
        "target_source_sha256": target_source,
        "target_parameters_sha256": target_parameters,
    })
    shared._write_new(attempt_path, attempt)
    task.reset(force=True)
    task = _wait_status(task_class, {"created"}, "candidate controller recovery reset")
    if shared._artifact_inventory(task):
        raise UpgradeError("candidate controller recovery shell is not empty")
    deploy._configure_shell(task, prepared=prepared)
    created = deploy._validate_task(task_class, task_id=TASK_ID, prepared=prepared, expected_statuses=("created",))
    queued = deploy._enqueue(task_class, task, task_id=TASK_ID, prepared=prepared, sleeper=time.sleep)
    result = shared._seal({
        "schema_version": 1,
        "receipt_type": "resilient_v2x_candidate_a100_parallel_readback_recovery",
        "generated_at_utc": shared._utc_timestamp(),
        "status": "recovered_and_enqueued",
        "remote_state_changed": True,
        "same_id_only": True,
        "replacement_controller_created": False,
        "attempt_receipt_path": str(attempt_path.resolve()),
        "attempt_receipt_seal_sha256": attempt["seal_sha256"],
        "created_authority": created,
        "queued_authority": queued,
        "child_tasks_not_reset": children,
    })
    shared._write_new(receipt_path, result)
    print(f"candidate controller readback recovered: task={TASK_ID} status={queued['status']} seal={result['seal_sha256']}")


def _target() -> tuple[dict[str, object], Mapping[str, object]]:
    prepared = deploy._prepare()
    source = prepared.get("source")
    parameters = prepared.get("parameters")
    if type(source) is not str or not isinstance(parameters, Mapping):
        raise UpgradeError("target deployment is invalid")
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    parameters_sha = deploy._content_sha256(deploy._normalized_parameters(parameters))
    e3 = next(spec for spec in queue.CANDIDATES if spec.label == "E3")
    if (
        source_sha != TARGET_SOURCE_SHA256
        or parameters_sha != TARGET_PARAMETERS_SHA256
        or prepared.get("deployment_seal_sha256") != "4e4e553c9ca7ec8db9d901b6a00fa8c1722ee23ba9d9e0e6052a09a82b872f3e"
        or queue.MAX_ACTIVE_CANDIDATE_EVALUATIONS != 3
        or e3.queue != "GPU4-A100"
    ):
        raise UpgradeError("target A100 parallel deployment drifted")
    return {
        "source_sha256": source_sha,
        "parameters_sha256": parameters_sha,
        "deployment_seal_sha256": prepared["deployment_seal_sha256"],
        "max_active_candidate_evaluations": 3,
        "E3_queue": "GPU4-A100",
    }, prepared


def _wait_status(task_class: object, expected: set[str], context: str) -> object:
    last = ""
    for index in range(20):
        task = shared._authoritative_task(task_class, TASK_ID, context=context)
        last = shared._status(task)
        if last in expected:
            return task
        if index != 19:
            time.sleep(1.0)
    raise UpgradeError(f"{context} did not reach {sorted(expected)}: {last!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--attempt-receipt", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--recover-readback", action="store_true")
    args = parser.parse_args(argv)
    if args.execute != (args.attempt_receipt is not None):
        raise UpgradeError("--attempt-receipt is required exactly with --execute")
    if args.execute and args.execute_token != EXECUTE_TOKEN:
        raise UpgradeError(f"exact execute token required: {EXECUTE_TOKEN}")
    outputs = [args.receipt] + ([args.attempt_receipt] if args.attempt_receipt else [])
    for path in outputs:
        assert path is not None
        shared._require_new(path, context="A100 parallel upgrade receipt")
    from clearml import Task

    if args.recover_readback:
        if not args.execute or args.attempt_receipt is None:
            raise UpgradeError("readback recovery requires execute and attempt receipt")
        _recover_failed(Task, args.attempt_receipt, args.receipt)
        return 0

    task, before = _controller(Task)
    children = _children(Task)
    target, prepared = _target()
    attempt = shared._seal({
        "schema_version": 1,
        "receipt_type": "resilient_v2x_candidate_a100_parallel_upgrade_attempt",
        "generated_at_utc": shared._utc_timestamp(),
        "status": "frozen_before_stop",
        "remote_state_changed": False,
        "same_id_only": True,
        "replacement_controller_created": False,
        "controller_before": before,
        "child_tasks_before": children,
        "target": target,
    })
    if not args.execute:
        plan = shared._seal({**attempt, "receipt_type": "resilient_v2x_candidate_a100_parallel_upgrade_plan", "status": "planned", "execute_token": EXECUTE_TOKEN})
        shared._write_new(args.receipt, plan)
        print(f"A100 parallel upgrade plan: {args.receipt.resolve()} seal={plan['seal_sha256']}")
        return 0
    assert args.attempt_receipt is not None
    shared._write_new(args.attempt_receipt, attempt)
    _, current = _controller(Task)
    if current != before or _children(Task) != children:
        raise UpgradeError("authority changed after upgrade freeze")
    task.mark_stopped(force=True, status_message="exact-ID A100 parallel policy upgrade")
    _wait_status(Task, {"stopped"}, "candidate controller stop")
    task = shared._authoritative_task(Task, TASK_ID, context="candidate controller reset target")
    task.reset(force=True)
    task = _wait_status(Task, {"created"}, "candidate controller reset")
    if shared._artifact_inventory(task) or shared._model_inventory(task, context="created controller") != {"input": [], "output": []}:
        raise UpgradeError("controller reset shell is not empty")
    deploy._configure_shell(task, prepared=prepared)
    created = deploy._validate_task(Task, task_id=TASK_ID, prepared=prepared, expected_statuses=("created",))
    queued = deploy._enqueue(Task, task, task_id=TASK_ID, prepared=prepared, sleeper=time.sleep)
    if _children(Task) != children:
        raise UpgradeError("child tasks changed during controller upgrade")
    result = shared._seal({
        "schema_version": 1,
        "receipt_type": "resilient_v2x_candidate_a100_parallel_upgrade",
        "generated_at_utc": shared._utc_timestamp(),
        "status": "upgraded_and_enqueued",
        "remote_state_changed": True,
        "same_id_only": True,
        "replacement_controller_created": False,
        "attempt_receipt_path": str(args.attempt_receipt.resolve()),
        "attempt_receipt_seal_sha256": attempt["seal_sha256"],
        "controller_before": before,
        "target": target,
        "created_authority": created,
        "queued_authority": queued,
        "child_tasks_unchanged": children,
    })
    shared._write_new(args.receipt, result)
    print(f"candidate controller upgraded: task={TASK_ID} status={queued['status']} seal={result['seal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
