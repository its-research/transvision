#!/usr/bin/env python3
"""Pause only the exact W3 plan and two exact candidate evaluations.

Dry-run is the default.  Execute mode may only dequeue a queued task back to
``created`` or stop an ``in_progress`` W3 evaluation.  It never mutates
training tasks, completed results, failed tasks, stopped tasks, or unrelated
evaluations.  All identities are bound to the W3 sealed plan or explicit
candidate task/source contracts and are authoritatively read back.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECONCILER_PATH = (
    ROOT / "tools/resilient_v2x/reconcile_clearml_formal_evaluation_orphans.py"
)
ARTIFACT_DIR = ROOT / "artifacts/resilient_v2x/formal-evaluation-pause"

PROJECT_ID = "6e43f972e5ea4cee901a7c8855fce8cd"
CONTROLLER_TASK_ID = "b6f0fbab32a5478183a45b3ca833fc01"
CONTROLLER_NAME = "ResilientV2X formal 1337 candidate evaluation queue"
CONTROLLER_PARENT_ID = "7e244a711751469b8cfdb25d77b05269"
CONTROLLER_SOURCE_SHA256 = (
    "9055b41235d3e8c34583a03fe64e2a2ab42f91e99995ffac13c5a7e77cf10adb"
)
W3_EVALUATION_SOURCE_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
ENTRY_POINT = "clearml_5090_bootstrap.py"
CONTROLLER_ENTRY_POINT = "clearml_formal_candidate_evaluation_queue.py"
EXECUTE_TOKEN = "PAUSE_EXACT_W3_AND_CANDIDATE_EVALUATIONS"
READ_ATTEMPTS = 5

CANDIDATES = (
    {
        "label": "E1",
        "subject": "support_residual_linear",
        "task_id": "c6f26cc7902142c090ec238856409ac6",
        "task_name": (
            "ResilientV2X formal1337 candidate eval E1 "
            "support_residual_linear [f0c3082f3aa3]"
        ),
        "parent_task_id": "f0c3082f3aa34a81805903e0ffdc8610",
        "source_sha256": (
            "6c67188351663db7083bea4b53dcd95b30855d10299aa41d3577a456ad468178"
        ),
    },
    {
        "label": "E3",
        "subject": "support_residual_no_reliability",
        "task_id": "27a82d39e6354697ad1532b6399bbde2",
        "task_name": (
            "ResilientV2X formal1337 candidate eval E3 "
            "support_residual_no_reliability [dc037315c068]"
        ),
        "parent_task_id": "dc037315c0684c3d854a2fd7c19a2a2f",
        "source_sha256": (
            "c0de5e7df30723d08829e6348355235919880749a248c5cdc7bcda80519d6763"
        ),
    },
)


class PauseError(RuntimeError):
    """Raised when a pause contract cannot be proven."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _sha(result)
    return result


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _load_reconciler() -> object:
    spec = importlib.util.spec_from_file_location(
        "_resilient_v2x_formal_pause_reconciler", RECONCILER_PATH
    )
    if spec is None or spec.loader is None:
        raise PauseError("cannot load the W3 plan reconciler")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _status(task: object) -> str:
    value = getattr(task, "status", None)
    value = str(getattr(value, "value", value) or "").lower()
    if not value:
        raise PauseError("task status is unavailable")
    return value


def _task_id(value: object, context: str) -> str:
    result = str(value or "")
    if re.fullmatch(r"[0-9a-f]{32}", result) is None:
        raise PauseError(f"{context} is not a ClearML task ID")
    return result


def _parent(task: object) -> str:
    value = getattr(task, "parent", None)
    if value in {None, ""}:
        value = getattr(getattr(task, "data", None), "parent", None)
    return str(value or "")


def _project(task: object) -> str:
    value = getattr(task, "project", None)
    if value in {None, ""}:
        value = getattr(getattr(task, "data", None), "project", None)
    return str(value or "")


def _script_identity(task: object, *, context: str) -> dict[str, str]:
    script = getattr(getattr(task, "data", None), "script", None)
    converter = getattr(script, "to_dict", None)
    script = converter() if callable(converter) else script
    if not isinstance(script, Mapping):
        raise PauseError(f"{context} script authority is unavailable")
    diff = script.get("diff")
    if type(diff) is not str or not diff:
        raise PauseError(f"{context} source bytes are unavailable")
    return {
        "entry_point": str(script.get("entry_point") or ""),
        "repository": str(script.get("repository") or ""),
        "working_dir": str(script.get("working_dir") or ""),
        "source_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def _parameters(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise PauseError(f"{context} cannot expose parameters")
    try:
        value = getter(backwards_compatibility=False, cast=False)
    except TypeError:
        try:
            value = getter(backwards_compatibility=False)
        except TypeError:
            value = getter()
    if not isinstance(value, Mapping):
        raise PauseError(f"{context} parameters are invalid")
    return {str(key): item for key, item in value.items()}


def _authoritative_task(task_class: object, task_id: str, *, context: str) -> object:
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise PauseError("ClearML Task cannot read authority")
    task = getter(task_id=task_id)
    if task is None or str(getattr(task, "id", "") or "") != task_id:
        raise PauseError(f"{context} authority ID mismatched")
    reloader = getattr(task, "_reload", None)
    if callable(reloader):
        reloader()
    return task


def _snapshot(
    task: object,
    *,
    expected_name: str,
    expected_parent: str,
    expected_source_sha256: str,
    expected_entry_point: str = ENTRY_POINT,
    expected_subject: str | None,
    context: str,
) -> dict[str, object]:
    task_id = _task_id(getattr(task, "id", ""), context)
    if str(getattr(task, "name", "") or "") != expected_name:
        raise PauseError(f"{context} name drifted")
    if _project(task) != PROJECT_ID or _parent(task) != expected_parent:
        raise PauseError(f"{context} project or parent drifted")
    script = _script_identity(task, context=context)
    if (
        script["entry_point"] != expected_entry_point
        or script["repository"] != ""
        or script["working_dir"] != "."
        or script["source_sha256"] != expected_source_sha256
    ):
        raise PauseError(f"{context} source identity drifted")
    parameters = _parameters(task, context=context)
    if expected_subject is not None:
        expected_parameters = {
            "Args/stage": "baseline_validate",
            "Args/controlled_baseline": expected_subject,
        }
        for key, value in expected_parameters.items():
            if str(parameters.get(key) or "") != value:
                raise PauseError(f"{context} parameter {key} drifted")
        baseline_task_id = _task_id(
            parameters.get("Args/controlled_baseline_task_id"),
            f"{context} controlled baseline",
        )
        if str(parameters.get("Args/predecessor_task_id") or "") != baseline_task_id:
            raise PauseError(f"{context} predecessor binding drifted")
        if context.startswith("candidate ") and baseline_task_id != expected_parent:
            raise PauseError(f"{context} training parent binding drifted")
    queue_getter = getattr(task, "get_executed_queue", None)
    if not callable(queue_getter):
        raise PauseError(f"{context} cannot expose queue authority")
    result = {
        "task_id": task_id,
        "task_name": expected_name,
        "project_id": PROJECT_ID,
        "parent_task_id": expected_parent,
        "status": _status(task),
        "queue_id": str(queue_getter(return_name=False) or ""),
        "source": script,
        "binding_parameters_sha256": _sha(
            {
                key: parameters.get(key)
                for key in (
                    "Args/stage",
                    "Args/controlled_baseline",
                    "Args/controlled_baseline_task_id",
                    "Args/controlled_baseline_model_id",
                    "Args/controlled_baseline_checkpoint_sha256",
                    "Args/predecessor_task_id",
                )
            }
        ),
    }
    result["authority_sha256"] = _sha(result)
    return result


def _validate_controller(task_class: object) -> dict[str, object]:
    task = _authoritative_task(
        task_class, CONTROLLER_TASK_ID, context="candidate controller"
    )
    snapshot = _snapshot(
        task,
        expected_name=CONTROLLER_NAME,
        expected_parent=CONTROLLER_PARENT_ID,
        expected_source_sha256=CONTROLLER_SOURCE_SHA256,
        expected_entry_point=CONTROLLER_ENTRY_POINT,
        expected_subject=None,
        context="candidate controller",
    )
    if snapshot["status"] != "stopped":
        raise PauseError("candidate controller must be stopped before pausing children")
    parameters = _parameters(task, context="candidate controller")
    expected = {
        "Deployment/source_sha256": CONTROLLER_SOURCE_SHA256,
        "Deployment/parent_task_id": CONTROLLER_PARENT_ID,
    }
    if any(str(parameters.get(key) or "") != value for key, value in expected.items()):
        raise PauseError("candidate controller deployment binding drifted")
    return snapshot


def _inventory(task_class: object) -> dict[str, object]:
    reconciler = _load_reconciler()
    plan, keep_by_subject, producer_status, artifact_authority = (
        reconciler._load_plan(task_class)
    )
    records: list[dict[str, object]] = []
    for index, subject in enumerate(reconciler.FORMAL_SUBJECT_ORDER, start=1):
        task_id = keep_by_subject[subject]
        task = _authoritative_task(
            task_class, task_id, context=f"W3 evaluation {subject}"
        )
        snapshot = _snapshot(
            task,
            expected_name=reconciler._planned_name(index, subject),
            expected_parent=reconciler.TRAINING_CONTROLLER_TASK_ID,
            expected_source_sha256=W3_EVALUATION_SOURCE_SHA256,
            expected_subject=subject,
            context=f"W3 evaluation {subject}",
        )
        status = str(snapshot["status"])
        if status not in {
            "created",
            "queued",
            "in_progress",
            "stopped",
            "failed",
            "completed",
        }:
            raise PauseError(f"W3 evaluation {subject} has unsafe status {status}")
        action = {
            "queued": "dequeue_to_created",
            "in_progress": "mark_stopped",
        }.get(status, "preserve_terminal_or_paused")
        records.append(
            {
                "scope": "w3_sealed_keep_set",
                "subject": subject,
                "action": action,
                "pre_authority": snapshot,
            }
        )
    candidate_records: list[dict[str, object]] = []
    for spec in CANDIDATES:
        task = _authoritative_task(
            task_class, str(spec["task_id"]), context=f"candidate {spec['label']}"
        )
        snapshot = _snapshot(
            task,
            expected_name=str(spec["task_name"]),
            expected_parent=str(spec["parent_task_id"]),
            expected_source_sha256=str(spec["source_sha256"]),
            expected_subject=str(spec["subject"]),
            context=f"candidate {spec['label']}",
        )
        status = str(snapshot["status"])
        if status not in {"created", "queued", "stopped", "failed", "completed"}:
            raise PauseError(
                f"candidate {spec['label']} is not safely dequeueable/preserved"
            )
        candidate_records.append(
            {
                "scope": "candidate_evaluation",
                "subject": spec["subject"],
                "action": (
                    "dequeue_to_created"
                    if status == "queued"
                    else "preserve_terminal_or_paused"
                ),
                "pre_authority": snapshot,
            }
        )
    return {
        "controller": _validate_controller(task_class),
        "w3_plan": {
            "plan_sha256": _sha(plan),
            "plan_seal_sha256": plan["seal_sha256"],
            "producer_status": producer_status,
            "artifact_authority": artifact_authority,
            "keep_task_ids": list(keep_by_subject.values()),
        },
        "records": records + candidate_records,
    }


def _mutate_one(
    task_class: object,
    *,
    record: Mapping[str, object],
    sleeper: Callable[[float], None],
) -> dict[str, object]:
    before = record["pre_authority"]
    if not isinstance(before, Mapping):
        raise PauseError("pause record has no authority snapshot")
    action = str(record["action"])
    task_id = str(before["task_id"])
    if action == "preserve_terminal_or_paused":
        return {"disposition": "preserved", "post_authority": dict(before)}
    task = _authoritative_task(task_class, task_id, context=f"pause {task_id}")
    callback_error: Exception | None = None
    try:
        if action == "dequeue_to_created":
            callback = getattr(task_class, "dequeue", None)
            if not callable(callback):
                raise PauseError("ClearML Task cannot dequeue")
            response = callback(task=task)
        elif action == "mark_stopped":
            callback = getattr(task, "mark_stopped", None)
            if not callable(callback):
                raise PauseError(f"task {task_id} cannot be stopped")
            response = callback(
                force=True,
                status_message="Paused by sealed formal evaluation pause contract",
            )
        else:
            raise PauseError(f"unknown pause action {action}")
        if response is False:
            callback_error = PauseError(f"task {task_id} pause callback failed")
    except Exception as error:
        callback_error = error
    expected_status = "created" if action == "dequeue_to_created" else "stopped"
    last_error: Exception | None = callback_error
    for attempt in range(READ_ATTEMPTS):
        try:
            current = _authoritative_task(
                task_class, task_id, context=f"paused {task_id}"
            )
            if _status(current) != expected_status:
                raise PauseError(
                    f"task {task_id} has not reached {expected_status}"
                )
            if (
                str(getattr(current, "name", "") or "") != before["task_name"]
                or _project(current) != before["project_id"]
                or _parent(current) != before["parent_task_id"]
                or _script_identity(current, context=f"paused {task_id}")
                != before["source"]
            ):
                raise PauseError(f"task {task_id} identity drifted while pausing")
            after = dict(before)
            after["status"] = expected_status
            after["authority_sha256"] = _sha(
                {key: value for key, value in after.items() if key != "authority_sha256"}
            )
            return {
                "disposition": (
                    f"{action}_confirmed_after_callback_error"
                    if callback_error is not None
                    else f"{action}_confirmed"
                ),
                "post_authority": after,
            }
        except Exception as error:
            last_error = error
        if attempt != READ_ATTEMPTS - 1:
            sleeper(1.0)
    raise PauseError(f"task {task_id} pause readback failed") from last_error


def _receipt(
    task_class: object,
    *,
    execute: bool,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    inventory = _inventory(task_class)
    records = []
    for source in inventory["records"]:
        record = dict(source)
        if execute:
            record.update(_mutate_one(task_class, record=record, sleeper=sleeper))
        else:
            record["disposition"] = f"would_{record['action']}"
        records.append(record)
    result = {
        "schema_version": 1,
        "receipt_type": "resilient_v2x_formal_evaluation_pause",
        "generated_at_utc": _utc_timestamp(),
        "mode": "execute" if execute else "dry_run",
        "status": "paused" if execute else "planned",
        "remote_state_changed": bool(
            execute
            and any(
                record["action"] in {"dequeue_to_created", "mark_stopped"}
                for record in records
            )
        ),
        "controller_authority": inventory["controller"],
        "w3_plan_binding": inventory["w3_plan"],
        "mutation_contract": {
            "training_tasks_mutated": False,
            "delete_or_archive_used": False,
            "queued_action": "dequeue_to_created",
            "w3_running_action": "mark_stopped",
            "candidate_running_policy": "fail_closed",
            "terminal_or_paused_policy": "preserve",
            "authority_read_attempts": READ_ATTEMPTS,
        },
        "summary": {
            "w3_count": 26,
            "candidate_count": 2,
            "would_or_did_dequeue": sum(
                record["action"] == "dequeue_to_created" for record in records
            ),
            "would_or_did_stop": sum(
                record["action"] == "mark_stopped" for record in records
            ),
            "preserved": sum(
                record["action"] == "preserve_terminal_or_paused"
                for record in records
            ),
        },
        "tasks": records,
    }
    return _seal(result)


def _default_receipt_path(*, execute: bool) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return ARTIFACT_DIR / (
        f"{'execute' if execute else 'dry-run'}-receipt-{timestamp}.json"
    )


def _write_new(path: Path, receipt: Mapping[str, object]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    payload = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.execute and args.execute_token:
        parser.error("--execute-token requires --execute")
    if args.execute and args.execute_token != EXECUTE_TOKEN:
        raise PauseError(f"exact execute token required: {EXECUTE_TOKEN}")
    try:
        from clearml import Task
    except ImportError as error:  # pragma: no cover
        raise PauseError("ClearML client is required") from error
    receipt = _receipt(Task, execute=args.execute)
    path = args.receipt or _default_receipt_path(execute=args.execute)
    _write_new(path, receipt)
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CANDIDATES",
    "EXECUTE_TOKEN",
    "PauseError",
    "_inventory",
    "_mutate_one",
    "_receipt",
    "main",
)
