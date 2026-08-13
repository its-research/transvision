#!/usr/bin/env python3
"""Archive only unbound, never-run formal-1337 evaluation task shells.

The W3 watcher evaluation-plan artifact is the sole keep-set authority.  The
default invocation performs read-only reconciliation and writes a sealed local
receipt.  Remote archival is recoverable and requires both ``--execute`` and
the exact token.  This tool never dequeues, stops, deletes, or mutates a task
that is queued, running, completed, or otherwise not a pristine created shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = (
    ROOT / "artifacts/resilient_v2x/formal-evaluation-orphan-reconciliation"
)

PROJECT_ID = "6e43f972e5ea4cee901a7c8855fce8cd"
PROJECT_NAME = "ResilientV2X/Training"
TRAINING_CONTROLLER_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
PROVENANCE_TASK_ID = "7e244a711751469b8cfdb25d77b05269"
W3_TASK_ID = "dbc05b28fbd044bf89edc3872742843a"
W3_TASK_NAME = "ResilientV2X formal 1337 dependency watcher successor"
PLAN_ARTIFACT = "formal_1337_evaluation_plan"
PLAN_TYPE = "resilient_v2x_formal_1337_evaluation_tasks"
PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
EXECUTE_TOKEN = "ARCHIVE_EXACT_FORMAL_1337_W3_ORPHAN_SHELLS"
ARCHIVED_TAG = "archived"
AUTHORITY_READ_ATTEMPTS = 5
MAX_JSON_ARTIFACT_BYTES = 64 * 1024 * 1024

FORMAL_SUBJECT_ORDER = (
    "support_residual",
    "ptf_none",
    "ptf_linear",
    "router_static",
    "no_distillation",
    "coformernet",
    "router_uniform",
    "no_reliability",
    "no_delay_metadata",
    "concat_capacity_matched",
    "ffnet",
    "bevfusion",
    "v2x_vit",
    "cobevt",
    "linear_no_distillation",
    "no_distillation_peak_lr_3e4",
    "ego_only",
    "fcooper",
    "attfuse",
    "v2vnet",
    "when2com",
    "where2comm",
    "late_fusion",
    "disconet",
    "how2comm",
    "resilient_v2x",
)


class ReconciliationError(RuntimeError):
    """Raised when archival safety cannot be proven."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _task_id(value: object, context: str) -> str:
    result = str(value or "")
    if re.fullmatch(r"[0-9a-f]{32}", result) is None:
        raise ReconciliationError(
            f"{context} must be a lowercase 32-hex ClearML ID"
        )
    return result


def _status(task: object, *, context: str) -> str:
    value = getattr(task, "status", None)
    if value is None:
        value = getattr(getattr(task, "data", None), "status", None)
    result = str(getattr(value, "value", value) or "").lower()
    if not result:
        raise ReconciliationError(f"{context} status is unavailable")
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


def _sequence(value: object, *, context: str) -> list[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise ReconciliationError(f"{context} is not a sequence")
    return list(value)


def _string_set(value: object, *, context: str) -> list[str]:
    values = [str(item) for item in _sequence(value, context=context)]
    if any(not item for item in values) or len(set(values)) != len(values):
        raise ReconciliationError(f"{context} is empty or duplicated")
    return sorted(values)


def _method_sequence(task: object, name: str, *, context: str) -> list[str]:
    getter = getattr(task, name, None)
    if not callable(getter):
        raise ReconciliationError(f"{context} cannot expose {name}")
    return _string_set(getter() or [], context=f"{context} {name}")


def _task_archived(task: object, *, context: str) -> bool:
    getter = getattr(task, "get_archived", None)
    if not callable(getter):
        raise ReconciliationError(f"{context} cannot expose archive status")
    value = getter()
    if type(value) is not bool:
        raise ReconciliationError(f"{context} returned invalid archive status")
    return value


def _to_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return {str(key): item for key, item in converted.items()}
    data = getattr(value, "__dict__", None)
    if isinstance(data, Mapping):
        return {
            str(key): item
            for key, item in data.items()
            if not str(key).startswith("_")
        }
    return {}


def _prune_empty(value: object) -> object:
    if isinstance(value, Mapping):
        result = {
            str(key): _prune_empty(item)
            for key, item in value.items()
            if item not in (None, "")
        }
        return {
            key: item for key, item in result.items() if item not in ([], {})
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        result = [_prune_empty(item) for item in value]
        return [item for item in result if item not in (None, "", [], {})]
    return value


def _date_value(value: object) -> str | None:
    if value in {None, ""}:
        return None
    formatter = getattr(value, "isoformat", None)
    return str(formatter()) if callable(formatter) else str(value)


def _artifact_names(task: object, *, context: str) -> list[str]:
    value = getattr(task, "artifacts", None)
    if not isinstance(value, Mapping):
        raise ReconciliationError(f"{context} artifact inventory is unavailable")
    return sorted(str(key) for key in value)


def _model_ids(task: object, *, context: str) -> dict[str, list[str]]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise ReconciliationError(f"{context} cannot expose models")
    models = getter()
    if not isinstance(models, Mapping):
        raise ReconciliationError(f"{context} model inventory is invalid")
    result: dict[str, list[str]] = {}
    for role in ("input", "output"):
        items = _sequence(models.get(role, []), context=f"{context} {role} models")
        ids = [_task_id(getattr(item, "id", ""), f"{context} {role} model") for item in items]
        if len(set(ids)) != len(ids):
            raise ReconciliationError(f"{context} repeats a {role} model")
        result[role] = sorted(ids)
    return result


def _reported_outputs(task: object, *, context: str) -> dict[str, object]:
    calls = (
        ("console", "get_reported_console_output", {"number_of_reports": 1}),
        ("scalars", "get_reported_scalars", {"max_samples": 1}),
        ("plots", "get_reported_plots", {"max_iterations": 1}),
        ("single_values", "get_reported_single_values", {}),
    )
    result: dict[str, object] = {}
    for key, method_name, kwargs in calls:
        getter = getattr(task, method_name, None)
        if not callable(getter):
            raise ReconciliationError(
                f"{context} cannot expose reported {key}"
            )
        value = getter(**kwargs)
        if key in {"console", "plots"}:
            result[key] = _sequence(value or [], context=f"{context} {key}")
        elif isinstance(value, Mapping):
            result[key] = {str(item): content for item, content in value.items()}
        else:
            raise ReconciliationError(f"{context} returned invalid {key}")
    return result


def _queue_id(task: object) -> str:
    execution = getattr(getattr(task, "data", None), "execution", None)
    value = getattr(execution, "queue", None)
    if value in {None, ""}:
        getter = getattr(task, "get_executed_queue", None)
        if callable(getter):
            value = getter(return_name=False)
    return str(value or "")


def _authority_snapshot(
    task: object,
    *,
    context: str,
    inspect_reported_outputs: bool,
) -> dict[str, object]:
    data = getattr(task, "data", None)
    if data is None:
        raise ReconciliationError(f"{context} authoritative data is unavailable")
    models = _model_ids(task, context=context)
    system_tags = _method_sequence(task, "get_system_tags", context=context)
    archived = _task_archived(task, context=context)
    if archived != (ARCHIVED_TAG in system_tags):
        raise ReconciliationError(f"{context} archive tag/status disagree")
    runtime = _prune_empty(_to_mapping(getattr(data, "runtime", None)))
    execution = _to_mapping(getattr(data, "execution", None))
    execution.pop("queue", None)
    if execution.get("test_split") == 0:
        execution.pop("test_split")
    execution = _prune_empty(execution)
    output = _to_mapping(getattr(data, "output", None))
    output.pop("destination", None)
    output = _prune_empty(output)
    snapshot: dict[str, object] = {
        "task_id": _task_id(getattr(task, "id", ""), context),
        "task_name": str(getattr(task, "name", "") or ""),
        "project_id": _project(task),
        "parent_task_id": _parent(task),
        "status": _status(task, context=context),
        "archived": archived,
        "tags": _method_sequence(task, "get_tags", context=context),
        "system_tags": system_tags,
        "queue_id": _queue_id(task),
        "started": _date_value(getattr(data, "started", None)),
        "completed": _date_value(getattr(data, "completed", None)),
        "published": _date_value(getattr(data, "published", None)),
        "last_worker": str(getattr(data, "last_worker", None) or ""),
        "runtime": runtime,
        "execution_without_queue": execution,
        "output_without_destination": output,
        "artifact_names": _artifact_names(task, context=context),
        "input_model_ids": models["input"],
        "output_model_ids": models["output"],
    }
    if inspect_reported_outputs:
        snapshot["reported_outputs"] = _reported_outputs(task, context=context)
    snapshot["authority_sha256"] = _content_sha256(snapshot)
    return snapshot


def _non_archive_state(snapshot: Mapping[str, object]) -> dict[str, object]:
    result = dict(snapshot)
    result.pop("authority_sha256", None)
    result.pop("archived", None)
    result.pop("system_tags", None)
    return result


def _validate_identity(
    snapshot: Mapping[str, object], *, name: str, task_id: str, context: str
) -> None:
    expected = {
        "task_id": task_id,
        "task_name": name,
        "project_id": PROJECT_ID,
        "parent_task_id": TRAINING_CONTROLLER_TASK_ID,
    }
    for key, value in expected.items():
        if snapshot.get(key) != value:
            raise ReconciliationError(f"{context} {key} drifted")


def _validate_pristine_orphan(
    snapshot: Mapping[str, object], *, allow_archived: bool
) -> None:
    task_id = str(snapshot["task_id"])
    if snapshot.get("status") != "created":
        raise ReconciliationError(
            f"orphan {task_id} is not created; refusing to archive"
        )
    if snapshot.get("archived") and not allow_archived:
        raise ReconciliationError(f"orphan {task_id} was unexpectedly archived")
    empty_fields = (
        "queue_id",
        "started",
        "completed",
        "published",
        "last_worker",
        "runtime",
        "execution_without_queue",
        "output_without_destination",
        "artifact_names",
        "output_model_ids",
    )
    if any(snapshot.get(key) not in (None, "", [], {}) for key in empty_fields):
        raise ReconciliationError(
            f"orphan {task_id} has queue, execution, or output evidence"
        )
    reported = snapshot.get("reported_outputs")
    if not isinstance(reported, Mapping) or any(reported.values()):
        raise ReconciliationError(f"orphan {task_id} has reported output evidence")


def _secure_json_path(path: Path, *, context: str) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
    except OSError as error:
        raise ReconciliationError(f"{context} cannot be resolved") from error
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ReconciliationError(f"{context} must be a single-link regular file")
    if before.st_size <= 0 or before.st_size > MAX_JSON_ARTIFACT_BYTES:
        raise ReconciliationError(f"{context} size is invalid")
    flags = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= int(getattr(os, name, 0))
    try:
        descriptor = os.open(resolved, flags)
        with os.fdopen(descriptor, "rb") as stream:
            payload = stream.read(MAX_JSON_ARTIFACT_BYTES + 1)
        after = resolved.stat()
    except OSError as error:
        raise ReconciliationError(f"{context} cannot be read safely") from error
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stable or len(payload) != before.st_size:
        raise ReconciliationError(f"{context} changed while being read")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReconciliationError(f"{context} is not UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ReconciliationError(f"{context} is not a JSON object")
    return {str(key): item for key, item in value.items()}


def _verified_artifact_preview(
    artifact: object, *, context: str
) -> tuple[dict[str, object], dict[str, object]] | None:
    preview = getattr(artifact, "preview", None)
    if preview is None:
        return None
    if type(preview) is not str:
        raise ReconciliationError(f"{context} preview is not text")
    size = getattr(artifact, "size", None)
    digest = str(getattr(artifact, "hash", "") or "")
    encoded = preview.encode("utf-8")
    if type(size) is not int or size != len(encoded):
        raise ReconciliationError(f"{context} preview size metadata mismatched")
    if (
        re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or hashlib.sha256(encoded).hexdigest() != digest
    ):
        raise ReconciliationError(f"{context} preview hash metadata mismatched")
    artifact_type = getattr(artifact, "type", None)
    artifact_mode = getattr(artifact, "mode", None)
    artifact_type = str(getattr(artifact_type, "value", artifact_type) or "")
    artifact_mode = str(getattr(artifact_mode, "value", artifact_mode) or "")
    if artifact_type != "dict" or artifact_mode != "output":
        raise ReconciliationError(f"{context} preview type or mode mismatched")
    try:
        value = json.loads(preview)
    except json.JSONDecodeError as error:
        raise ReconciliationError(f"{context} preview is not JSON") from error
    if not isinstance(value, Mapping):
        raise ReconciliationError(f"{context} preview is not a JSON object")
    return (
        {str(key): item for key, item in value.items()},
        {
            "acquisition": "clearml_authority_preview",
            "size_bytes": size,
            "artifact_bytes_sha256": digest,
        },
    )


def _artifact_mapping(
    artifact: object, *, context: str
) -> tuple[dict[str, object], dict[str, object]]:
    preview = _verified_artifact_preview(artifact, context=context)
    if preview is not None:
        return preview
    getter = getattr(artifact, "get", None)
    if not callable(getter):
        raise ReconciliationError(f"{context} cannot be downloaded")
    try:
        value = getter(force_download=True)
    except TypeError:
        value = getter()
    if isinstance(value, Mapping):
        result = {str(key): item for key, item in value.items()}
        return result, {
            "acquisition": "clearml_object_api",
            "canonical_json_sha256": _content_sha256(result),
        }
    if isinstance(value, (str, os.PathLike)):
        result = _secure_json_path(Path(value), context=context)
        return result, {
            "acquisition": "clearml_downloaded_json",
            "canonical_json_sha256": _content_sha256(result),
        }
    raise ReconciliationError(f"{context} did not yield an object or JSON path")


def _authoritative_task(task_class: object, task_id: str, *, context: str) -> object:
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise ReconciliationError("ClearML Task class cannot read task authority")
    task = getter(task_id=task_id)
    if task is None or str(getattr(task, "id", "") or "") != task_id:
        raise ReconciliationError(f"{context} authority readback mismatched")
    reloader = getattr(task, "_reload", None)
    if callable(reloader):
        reloader()
    return task


def _load_plan(
    task_class: object,
) -> tuple[dict[str, object], dict[str, str], str, dict[str, object]]:
    task = _authoritative_task(task_class, W3_TASK_ID, context="W3 watcher")
    if str(getattr(task, "name", "") or "") != W3_TASK_NAME:
        raise ReconciliationError("W3 watcher name drifted")
    if _project(task) != PROJECT_ID or _parent(task) != PROVENANCE_TASK_ID:
        raise ReconciliationError("W3 watcher project or parent drifted")
    producer_status = _status(task, context="W3 watcher")
    if producer_status not in {
        "queued",
        "in_progress",
        "completed",
        "failed",
    }:
        raise ReconciliationError("W3 watcher has no usable authoritative status")
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or PLAN_ARTIFACT not in artifacts:
        raise ReconciliationError("W3 evaluation plan artifact is missing")
    plan, artifact_authority = _artifact_mapping(
        artifacts[PLAN_ARTIFACT], context="W3 evaluation plan artifact"
    )
    seal = str(plan.get("seal_sha256") or "")
    unsigned = dict(plan)
    unsigned.pop("seal_sha256", None)
    if re.fullmatch(r"[0-9a-f]{64}", seal) is None or _content_sha256(unsigned) != seal:
        raise ReconciliationError("W3 evaluation plan seal is invalid")
    expected = {
        "schema_version": 2,
        "plan_type": PLAN_TYPE,
        "training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
        "training_provenance_task_id": PROVENANCE_TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "sample_count": 1337,
        "run_count": 12,
        "subject_order": list(FORMAL_SUBJECT_ORDER),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ReconciliationError(f"W3 evaluation plan {key} drifted")
    entries = plan.get("entries")
    if not isinstance(entries, list) or len(entries) != len(FORMAL_SUBJECT_ORDER):
        raise ReconciliationError("W3 evaluation plan entry count drifted")
    keep_by_subject: dict[str, str] = {}
    for subject, entry in zip(FORMAL_SUBJECT_ORDER, entries, strict=True):
        if not isinstance(entry, Mapping) or entry.get("subject") != subject:
            raise ReconciliationError("W3 evaluation plan subject order drifted")
        keep_by_subject[subject] = _task_id(
            entry.get("evaluation_task_id"), f"W3 {subject} evaluation task"
        )
    if len(set(keep_by_subject.values())) != len(FORMAL_SUBJECT_ORDER):
        raise ReconciliationError("W3 evaluation plan repeats a task ID")
    return plan, keep_by_subject, producer_status, artifact_authority


def _planned_name(index: int, subject: str) -> str:
    return (
        f"ResilientV2X formal1337 eval {index:02d} {subject} "
        f"[{TRAINING_CONTROLLER_TASK_ID}]"
    )


def _inventory(task_class: object) -> list[object]:
    getter = getattr(task_class, "get_tasks", None)
    if not callable(getter):
        raise ReconciliationError("ClearML Task class cannot list task authority")
    try:
        value = getter(
            allow_archived=True,
            task_filter={"parent": TRAINING_CONTROLLER_TASK_ID},
        )
    except Exception as error:
        raise ReconciliationError(
            "failed to list formal evaluation tasks by parent ID"
        ) from error
    if value is None or isinstance(value, (str, bytes, bytearray)):
        raise ReconciliationError("formal evaluation inventory is invalid")
    names = {
        _planned_name(index, subject)
        for index, subject in enumerate(FORMAL_SUBJECT_ORDER, start=1)
    }
    result = [task for task in value if str(getattr(task, "name", "") or "") in names]
    ids = [_task_id(getattr(task, "id", ""), "formal evaluation task") for task in result]
    if len(ids) != 52 or len(set(ids)) != 52:
        raise ReconciliationError(
            f"expected exactly 52 distinct formal evaluation tasks, found {len(ids)}"
        )
    return result


def _preflight(task_class: object) -> dict[str, object]:
    plan, keep_by_subject, producer_status, artifact_authority = _load_plan(
        task_class
    )
    inventory = _inventory(task_class)
    by_name: dict[str, list[str]] = {}
    for task in inventory:
        by_name.setdefault(str(getattr(task, "name", "") or ""), []).append(
            _task_id(getattr(task, "id", ""), "formal evaluation task")
        )
    records: list[dict[str, object]] = []
    orphan_ids: list[str] = []
    keep_ids: list[str] = []
    for index, subject in enumerate(FORMAL_SUBJECT_ORDER, start=1):
        name = _planned_name(index, subject)
        ids = sorted(by_name.get(name, []))
        keep_id = keep_by_subject[subject]
        if len(ids) != 2 or keep_id not in ids:
            raise ReconciliationError(
                f"{subject} does not have exactly one W3-bound and one orphan shell"
            )
        orphan_id = next(task_id for task_id in ids if task_id != keep_id)
        for task_id, binding in ((keep_id, "keep"), (orphan_id, "orphan")):
            task = _authoritative_task(
                task_class, task_id, context=f"{subject} {binding}"
            )
            snapshot = _authority_snapshot(
                task,
                context=f"{subject} {binding}",
                inspect_reported_outputs=binding == "orphan",
            )
            _validate_identity(
                snapshot,
                name=name,
                task_id=task_id,
                context=f"{subject} {binding}",
            )
            if binding == "keep":
                if snapshot["archived"]:
                    raise ReconciliationError(
                        f"W3-bound task {task_id} is unexpectedly archived"
                    )
                keep_ids.append(task_id)
                reason = "task_id_is_bound_by_w3_sealed_evaluation_plan"
            else:
                _validate_pristine_orphan(snapshot, allow_archived=True)
                orphan_ids.append(task_id)
                reason = (
                    "same-name_same-parent_shell_absent_from_w3_plan_and_"
                    "authority_proves_created_never_enqueued_no_output"
                )
            records.append(
                {
                    "subject": subject,
                    "index": index,
                    "task_id": task_id,
                    "task_name": name,
                    "plan_binding": binding,
                    "reason": reason,
                    "pre_authority": snapshot,
                }
            )
    if set(keep_ids) != set(keep_by_subject.values()) or len(orphan_ids) != 26:
        raise ReconciliationError("keep/orphan partition is invalid")
    return {
        "plan": plan,
        "plan_sha256": _content_sha256(plan),
        "plan_seal_sha256": plan["seal_sha256"],
        "plan_producer_status": producer_status,
        "plan_artifact_authority": artifact_authority,
        "keep_task_ids": keep_ids,
        "orphan_task_ids": orphan_ids,
        "records": records,
    }


def _archive_with_readback(
    task_class: object,
    *,
    task_id: str,
    name: str,
    sleeper: Callable[[float], None],
) -> tuple[dict[str, object], dict[str, object], str]:
    task = _authoritative_task(task_class, task_id, context=f"orphan {task_id}")
    before = _authority_snapshot(
        task,
        context=f"orphan {task_id}",
        inspect_reported_outputs=True,
    )
    _validate_identity(
        before, name=name, task_id=task_id, context=f"orphan {task_id}"
    )
    _validate_pristine_orphan(before, allow_archived=True)
    if before["archived"]:
        return before, before, "already_archived_exact_orphan"
    setter = getattr(task, "set_archived", None)
    if not callable(setter):
        raise ReconciliationError(f"orphan {task_id} cannot be archived")
    callback_error: Exception | None = None
    try:
        response = setter(True)
        if response is False:
            callback_error = ReconciliationError(
                f"orphan {task_id} archive callback was not confirmed"
            )
    except Exception as error:  # authoritative readback may prove acceptance
        callback_error = error
    last_error: Exception | None = callback_error
    for attempt in range(AUTHORITY_READ_ATTEMPTS):
        try:
            current = _authoritative_task(
                task_class, task_id, context=f"archived orphan {task_id}"
            )
            after = _authority_snapshot(
                current,
                context=f"archived orphan {task_id}",
                inspect_reported_outputs=True,
            )
            _validate_identity(
                after,
                name=name,
                task_id=task_id,
                context=f"archived orphan {task_id}",
            )
            _validate_pristine_orphan(after, allow_archived=True)
            if not after["archived"]:
                raise ReconciliationError(
                    f"orphan {task_id} archive tag is not authoritative"
                )
            expected_system_tags = sorted(
                set(before["system_tags"]) | {ARCHIVED_TAG}
            )
            if after["system_tags"] != expected_system_tags:
                raise ReconciliationError(
                    f"orphan {task_id} system tags drifted while archiving"
                )
            if _non_archive_state(after) != _non_archive_state(before):
                raise ReconciliationError(
                    f"orphan {task_id} non-archive state drifted"
                )
            disposition = (
                "archived_authority_confirmed_after_callback_error"
                if callback_error is not None
                else "archived_authority_confirmed"
            )
            return before, after, disposition
        except Exception as error:
            last_error = error
        if attempt != AUTHORITY_READ_ATTEMPTS - 1:
            sleeper(1.0)
    raise ReconciliationError(
        f"orphan {task_id} archive authority readback failed"
    ) from last_error


def _base_receipt(*, mode: str, status: str, preflight: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "receipt_type": "resilient_v2x_formal_evaluation_orphan_reconciliation",
        "generated_at_utc": _utc_timestamp(),
        "mode": mode,
        "status": status,
        "fixed_bindings": {
            "project_id": PROJECT_ID,
            "project_name": PROJECT_NAME,
            "training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "provenance_task_id": PROVENANCE_TASK_ID,
            "w3_task_id": W3_TASK_ID,
            "w3_task_name": W3_TASK_NAME,
            "plan_artifact": PLAN_ARTIFACT,
            "protocol_id": PROTOCOL_ID,
            "formal_subject_order": list(FORMAL_SUBJECT_ORDER),
        },
        "authority_contract": {
            "expected_total_task_count": 52,
            "expected_keep_count": 26,
            "expected_orphan_count": 26,
            "archive_is_recoverable": True,
            "delete_dequeue_stop_are_forbidden": True,
            "orphan_status_must_be_created": True,
            "orphan_queue_execution_and_output_must_be_empty": True,
            "bounded_post_write_authority_reads": AUTHORITY_READ_ATTEMPTS,
        },
        "plan_binding": {
            "plan_sha256": preflight["plan_sha256"],
            "plan_seal_sha256": preflight["plan_seal_sha256"],
            "plan_producer_status": preflight["plan_producer_status"],
            "plan_artifact_authority": preflight["plan_artifact_authority"],
            "keep_task_ids": list(preflight["keep_task_ids"]),
            "orphan_task_ids": list(preflight["orphan_task_ids"]),
        },
    }


def _dry_run_receipt(task_class: object) -> dict[str, object]:
    preflight = _preflight(task_class)
    receipt = _base_receipt(mode="dry_run", status="planned", preflight=preflight)
    records = []
    for record in preflight["records"]:
        item = dict(record)
        item["action"] = (
            "preserve_w3_bound_task"
            if item["plan_binding"] == "keep"
            else (
                "already_archived_noop"
                if item["pre_authority"]["archived"]
                else "would_archive_recoverably"
            )
        )
        records.append(item)
    receipt.update(
        {
            "remote_state_changed": False,
            "summary": {
                "total": 52,
                "keep": 26,
                "orphan": 26,
                "would_archive": sum(
                    item["action"] == "would_archive_recoverably"
                    for item in records
                ),
                "already_archived": sum(
                    item["action"] == "already_archived_noop"
                    for item in records
                ),
            },
            "tasks": records,
        }
    )
    return _seal(receipt)


def _execute_receipt(
    task_class: object, *, sleeper: Callable[[float], None] = time.sleep
) -> dict[str, object]:
    journal: dict[str, object] = {"archive_attempted_task_ids": []}
    try:
        preflight = _preflight(task_class)
        journal["preflight_plan_seal_sha256"] = preflight["plan_seal_sha256"]
        by_id = {
            str(item["task_id"]): item for item in preflight["records"]
        }
        actions: dict[str, dict[str, object]] = {}
        for task_id in preflight["orphan_task_ids"]:
            record = by_id[str(task_id)]
            journal["archive_attempted_task_ids"].append(task_id)
            before, after, disposition = _archive_with_readback(
                task_class,
                task_id=str(task_id),
                name=str(record["task_name"]),
                sleeper=sleeper,
            )
            actions[str(task_id)] = {
                "action": disposition,
                "immediate_pre_authority": before,
                "immediate_post_authority": after,
            }
        postflight = _preflight(task_class)
        if postflight["plan_sha256"] != preflight["plan_sha256"]:
            raise ReconciliationError("W3 evaluation plan drifted during archival")
        post_by_id = {
            str(item["task_id"]): item for item in postflight["records"]
        }
        for task_id in preflight["orphan_task_ids"]:
            if not post_by_id[str(task_id)]["pre_authority"]["archived"]:
                raise ReconciliationError(
                    f"orphan {task_id} was not archived in final readback"
                )
        for task_id in preflight["keep_task_ids"]:
            if post_by_id[str(task_id)]["pre_authority"]["archived"]:
                raise ReconciliationError(
                    f"W3-bound task {task_id} became archived"
                )
    except Exception as error:
        receipt = {
            "schema_version": 1,
            "receipt_type": "resilient_v2x_formal_evaluation_orphan_reconciliation",
            "generated_at_utc": _utc_timestamp(),
            "mode": "execute",
            "status": "failed_closed",
            "remote_state_may_have_changed": bool(
                journal["archive_attempted_task_ids"]
            ),
            "failure": {"type": type(error).__name__, "message": str(error)},
            "partial_journal": journal,
        }
        failure = ReconciliationError(str(error))
        setattr(failure, "sealed_receipt", _seal(receipt))
        raise failure from error
    receipt = _base_receipt(
        mode="execute", status="reconciled", preflight=preflight
    )
    records = []
    for record in preflight["records"]:
        item = dict(record)
        task_id = str(item["task_id"])
        if item["plan_binding"] == "keep":
            item["action"] = "preserved_w3_bound_task"
        else:
            item.update(actions[task_id])
        item["final_authority"] = post_by_id[task_id]["pre_authority"]
        records.append(item)
    receipt.update(
        {
            "remote_state_changed": any(
                item["action"].startswith("archived_authority_confirmed")
                for item in records
            ),
            "summary": {
                "total": 52,
                "kept": 26,
                "archived_or_previously_archived": 26,
                "mutated_now": sum(
                    item["action"].startswith("archived_authority_confirmed")
                    for item in records
                ),
            },
            "tasks": records,
        }
    )
    return _seal(receipt)


def _default_receipt_path(*, mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return ARTIFACT_DIR / f"{mode}-receipt-{timestamp}.json"


def _write_new_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink(missing_ok=True)
        finally:
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Recoverably archive the 26 exact unbound shells.",
    )
    parser.add_argument(
        "--execute-token",
        default="",
        help="Exact permission token; valid only with --execute.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Write-once sealed receipt path (defaults under artifacts/).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.execute and args.execute_token:
        parser.error("--execute-token requires --execute")
    if args.execute and args.execute_token != EXECUTE_TOKEN:
        raise ReconciliationError(f"exact execute token required: {EXECUTE_TOKEN}")
    try:
        from clearml import Task
    except ImportError as error:  # pragma: no cover - deployment environment only
        raise ReconciliationError("reconciliation requires the ClearML client") from error
    mode = "execute" if args.execute else "dry-run"
    receipt_path = args.receipt or _default_receipt_path(mode=mode)
    try:
        receipt = _execute_receipt(Task) if args.execute else _dry_run_receipt(Task)
    except ReconciliationError as error:
        failed = getattr(error, "sealed_receipt", None)
        if isinstance(failed, Mapping):
            _write_new_receipt(receipt_path, failed)
        raise
    _write_new_receipt(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "EXECUTE_TOKEN",
    "FORMAL_SUBJECT_ORDER",
    "ReconciliationError",
    "_archive_with_readback",
    "_dry_run_receipt",
    "_execute_receipt",
    "_preflight",
    "_write_new_receipt",
    "main",
)
