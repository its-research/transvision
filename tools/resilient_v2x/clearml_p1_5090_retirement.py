#!/usr/bin/env python3
"""Safely retire the sealed RTX 5090 P1 run before an all-A100 rerun.

The default invocation is local and read-only.  ``--preflight`` connects to
ClearML but performs no remote mutation.  The retirement state machine is
available only with ``--retire`` and the exact acknowledgement token.

This utility never stops a running task.  It will retire the known child only
after proving that it is still queued or already unqueued, has never acquired
a worker, has no artifacts or output models, and is the only scoped child.  A
controller-level quarantine barrier is installed before dequeue so that the
legacy executor cannot resume and create a replacement child.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType


RETIREMENT_TOKEN = "RETIRE_P1_5090_FOR_HOMOGENEOUS_A100"
RETIREMENT_TAG = "resilient-v2x-p1-retired-for-homogeneous-a100"
RETIREMENT_INTENT_ARTIFACT = "p1_5090_retirement_intent"
RETIREMENT_ARTIFACT = "p1_5090_retirement_manifest"
RETIREMENT_DOCUMENT = "resilient_v2x_p1_5090_retirement"

EXECUTION_KEY = "63edb447d0c7ce7d82071bffbac15ec111e37ff9a747453bec9b93fc744060cd"
CONTROLLER_ID = "3dd5ba86053f49b59477c6a41219600b"
CHILD_ID = "4713942e362c40b9b093b00845a74511"
CHILD_TASK_KEY = "train-r01-s02-resilient_v2x"
LEGACY_MUTEX_QUEUE_ID = "602fbb124f8d450491100c46959a6a6d"
LEGACY_MUTEX_NAME = "__p1_resilient_v2x_mutex_" + EXECUTION_KEY
LEGACY_EXECUTOR_SHA256 = (
    "709d7b272d6c575f8d81f2d6eeda3a42f44b13fec3da6fb622d5a3e76b2a3b1b"
)
A100_PLAN_SEAL_SHA256 = (
    "43c1076057999ecb48dfa979f4ec6d952a62c4e4c02307cc350e8489a3728446"
)
A100_PINSET_SEAL_SHA256 = (
    "3d44c31b0fd1f1c2ba4714a59828cd3f25f7ed2e1b64425618fb5e2e5d411eb7"
)
A100_EXECUTION_KEY = "c4dfc01775ce835819685112d5bf9f3713ef15e94da5d38717a5f1d443bdb793"
LEGACY_EXECUTOR_NAME = "clearml_p1_multiseed_executor.py"
A100_EXECUTOR_NAME = "clearml_p1_a100_multiseed_executor.py"
A100_EXECUTOR_SHA256 = (
    "c6dc31dd324f96b688b7dd82a2d3e50fb8a48cd14658dbf5ff96daed072b1298"
)
INTENT_TARGET_A100_EXECUTOR_SHA256 = (
    "5ffe844378b99aaf460e059d29c4c6e9ff6e9ba614a5894956a81c59cf7388fe"
)
A100_EXECUTOR_AMENDMENT_REASON = "clearml_dequeue_retains_historical_execution_queue"
MAX_LEGACY_QUEUE_ENTRY_COUNT = 10_000
LEGACY_LOCK_NAME = ".p1-scoped-multiseed-executor.lock"
LEGACY_MUTEX_JOURNAL_NAME = ".p1-scoped-multiseed-mutex.json"
OBSERVED_LEGACY_EXECUTOR_PID = 243444
OBSERVED_LEGACY_TMUX_SESSION = "resilientv2x-p1-multiseed"


class P15090RetirementError(RuntimeError):
    """Raised when safe retirement cannot be proven."""


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
        raise P15090RetirementError(
            f"value is outside canonical JSON: {error}"
        ) from None


def _seal(value: Mapping[str, object]) -> str:
    detached = dict(value)
    detached.pop("seal_sha256", None)
    return hashlib.sha256(_canonical_json(detached).encode("utf-8")).hexdigest()


def _exact_id(value: object, expected: str, context: str) -> str:
    observed = str(value or "")
    if observed != expected:
        raise P15090RetirementError(
            f"{context} ID drifted: expected {expected}, observed {observed!r}"
        )
    return observed


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise P15090RetirementError(
            f"{context} keys drifted: expected {sorted(expected)}, "
            f"observed {sorted(observed)}"
        )


def _task_id(task: object, context: str) -> str:
    value = str(getattr(task, "id", "") or "")
    if re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise P15090RetirementError(f"{context} has no exact ClearML ID")
    return value


def _task_name(task: object) -> str:
    return str(getattr(task, "name", "") or "")


def _output_models(task: object) -> list[object]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise P15090RetirementError("ClearML task cannot expose its model inventory")
    models = getter()
    if not isinstance(models, Mapping):
        raise P15090RetirementError("ClearML task model inventory is not a mapping")
    output = models.get("output") or []
    if isinstance(output, (str, bytes, Mapping)):
        raise P15090RetirementError("ClearML output-model inventory is invalid")
    try:
        return list(output)
    except TypeError:
        raise P15090RetirementError(
            "ClearML output-model inventory is not iterable"
        ) from None


def _legacy_path() -> Path:
    return Path(__file__).resolve().with_name(LEGACY_EXECUTOR_NAME)


def _a100_path() -> Path:
    return Path(__file__).resolve().with_name(A100_EXECUTOR_NAME)


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise P15090RetirementError(f"{path.name} cannot be a symlink")
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise P15090RetirementError(
            f"cannot open sealed {path.name}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise P15090RetirementError(
                f"{path.name} is outside the safe file contract"
            )
        payload = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(payload) != before.st_size or any(
        getattr(before, field) != getattr(after, field) for field in fields
    ):
        raise P15090RetirementError(f"{path.name} changed while being read")
    return payload


def _load_legacy() -> ModuleType:
    path = _legacy_path()
    source = _read_regular_file(path, max_bytes=2 * 1024 * 1024)
    observed = hashlib.sha256(source).hexdigest()
    if observed != LEGACY_EXECUTOR_SHA256:
        raise P15090RetirementError(
            "legacy executor SHA-256 drifted; retirement bindings are invalid"
        )
    module = ModuleType("_sealed_clearml_p1_multiseed_executor")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    if (
        module._execution_key(module.build_plan(), module.build_pinset())
        != EXECUTION_KEY
    ):
        raise P15090RetirementError("legacy execution key drifted")
    return module


def _load_a100_target_contract() -> dict[str, str]:
    path = _a100_path()
    source = _read_regular_file(path, max_bytes=2 * 1024 * 1024)
    observed_sha256 = hashlib.sha256(source).hexdigest()
    if observed_sha256 != A100_EXECUTOR_SHA256:
        raise P15090RetirementError(
            "A100 executor SHA-256 drifted; retirement target is invalid"
        )
    module = ModuleType("_sealed_clearml_p1_a100_multiseed_executor")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    plan = module.build_plan()
    pinset = module.build_pinset()
    observed = {
        "plan_seal_sha256": str(plan.get("seal_sha256") or ""),
        "pinset_seal_sha256": str(pinset.get("seal_sha256") or ""),
        "execution_key": str(module._execution_key(plan, pinset) or ""),
        "executor_sha256": observed_sha256,
    }
    expected = {
        "plan_seal_sha256": A100_PLAN_SEAL_SHA256,
        "pinset_seal_sha256": A100_PINSET_SEAL_SHA256,
        "execution_key": A100_EXECUTION_KEY,
        "executor_sha256": A100_EXECUTOR_SHA256,
    }
    if observed != expected:
        raise P15090RetirementError(
            "A100 plan, pinset, or execution-key binding drifted"
        )
    return observed


def build_plan() -> dict[str, object]:
    _load_a100_target_contract()
    plan: dict[str, object] = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_5090_retirement_plan",
        "default_mode": "dry_run",
        "remote_mutation_count": 0,
        "reason": "homogeneous_a100_migration",
        "bindings": {
            "execution_key": EXECUTION_KEY,
            "controller_task_id": CONTROLLER_ID,
            "queued_child_task_id": CHILD_ID,
            "queued_child_task_key": CHILD_TASK_KEY,
            "legacy_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
            "legacy_executor_sha256": LEGACY_EXECUTOR_SHA256,
            "legacy_queue_name": "GPU4-5090",
            "legacy_queue_id": "5a84454c072349069e7b61af38637c6d",
            "target_a100_plan_seal_sha256": A100_PLAN_SEAL_SHA256,
            "target_a100_pinset_seal_sha256": A100_PINSET_SEAL_SHA256,
            "target_a100_execution_key": A100_EXECUTION_KEY,
            "target_a100_executor_sha256": A100_EXECUTOR_SHA256,
        },
        "operator_gate": {
            "observed_executor_pid": OBSERVED_LEGACY_EXECUTOR_PID,
            "observed_tmux_session": OBSERVED_LEGACY_TMUX_SESSION,
            "required_before_retire": [
                "stop the exact legacy executor process",
                "wait until the legacy local execution lock is acquirable",
                "do not manually move or requeue the child",
            ],
        },
        "fail_closed_guarantees": [
            "never invoke task stop or archive APIs",
            "upload sealed intent before any controller tag or dequeue mutation",
            "quarantine the controller after intent and before dequeueing the child",
            "treat GET queue entries as residency authority and execution.queue as history",
            "require exact hidden stable-name and execution-tag inventories",
            "require no worker, artifacts, or output models on the child",
            "preserve a sealed controller retirement artifact",
            "release the legacy server mutex with force=false",
        ],
    }
    plan["seal_sha256"] = _seal(plan)
    return plan


def _cache_directory(legacy: ModuleType) -> Path:
    try:
        return legacy._validate_cache_directory()
    except Exception as error:
        raise P15090RetirementError(str(error)) from error


def _local_lock_snapshot(legacy: ModuleType) -> dict[str, object]:
    cache = _cache_directory(legacy)
    path = cache / LEGACY_LOCK_NAME
    if not path.exists() and not path.is_symlink():
        return {
            "path": str(path),
            "status": "absent",
            "acquirable": False,
            "reason": "legacy lock file is absent; operator must inspect deployment",
        }
    if path.is_symlink():
        raise P15090RetirementError("legacy execution lock became a symlink")
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise P15090RetirementError("legacy execution lock is not regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "path": str(path),
                "status": "held",
                "acquirable": False,
                "reason": "legacy executor still owns the local lock",
            }
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return {
            "path": str(path),
            "status": "available",
            "acquirable": True,
            "reason": None,
        }
    finally:
        os.close(descriptor)


def _mutex_journal_snapshot(legacy: ModuleType) -> dict[str, object]:
    path = _cache_directory(legacy) / LEGACY_MUTEX_JOURNAL_NAME
    if not path.exists() and not path.is_symlink():
        return {"path": str(path), "status": "absent"}
    receipt = legacy._validate_mutex_receipt(
        legacy._read_json_path(path, "local mutex journal"),
        execution_key=EXECUTION_KEY,
    )
    return {
        "path": str(path),
        "status": "present",
        "queue_id": str(receipt["queue_id"]),
        "owner_tag_sha256": hashlib.sha256(
            str(receipt["owner_tag"]).encode("utf-8")
        ).hexdigest(),
        "seal_sha256": str(receipt["seal_sha256"]),
    }


def _public_lease(value: Mapping[str, object]) -> dict[str, object]:
    owner = value.get("owner_tag")
    return {
        "name": value.get("name"),
        "status": value.get("status"),
        "queue_id": value.get("queue_id"),
        "owner_tag_sha256": (
            hashlib.sha256(str(owner).encode("utf-8")).hexdigest() if owner else None
        ),
    }


def _expected_names(legacy: ModuleType) -> dict[str, str]:
    names: dict[str, str] = {}
    for pair in legacy.build_plan()["pairs"]:
        for phase in ("training", "evaluation"):
            record = pair[phase]
            key = str(record["task_key"])
            names[key] = legacy._task_name(EXECUTION_KEY, key)
    if len(names) != 8 or CHILD_TASK_KEY not in names:
        raise P15090RetirementError("legacy stable child-name inventory drifted")
    return names


def _validate_child_never_ran(legacy: ModuleType, child: object) -> None:
    if legacy._worker_id(child):
        raise P15090RetirementError(
            "scoped child acquired a worker; never-ran retirement is forbidden"
        )
    if legacy._artifact_inventory(child):
        raise P15090RetirementError(
            "scoped child has artifacts; never-ran retirement is forbidden"
        )
    if _output_models(child):
        raise P15090RetirementError(
            "scoped child has output models; never-ran retirement is forbidden"
        )


def _validate_legacy_queue_snapshot(
    value: Mapping[str, object], legacy: ModuleType
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise P15090RetirementError("legacy queue receipt is not a mapping")
    snapshot = dict(value)
    _require_exact_keys(
        snapshot,
        {"queue_id", "queue_name", "entry_task_ids", "entry_count"},
        "legacy queue receipt",
    )
    if (
        snapshot.get("queue_id") != legacy.QUEUE_ID
        or snapshot.get("queue_name") != legacy.QUEUE_NAME
    ):
        raise P15090RetirementError("legacy queue identity drifted")
    raw_ids = snapshot.get("entry_task_ids")
    if not isinstance(raw_ids, list):
        raise P15090RetirementError("legacy queue entry IDs are not a list")
    if len(raw_ids) > MAX_LEGACY_QUEUE_ENTRY_COUNT:
        raise P15090RetirementError("legacy queue entry inventory exceeds safe limit")
    entry_task_ids = [str(item or "") for item in raw_ids]
    if any(re.fullmatch(r"[0-9a-f]{32}", item) is None for item in entry_task_ids):
        raise P15090RetirementError("legacy queue contains an invalid task ID")
    if entry_task_ids != sorted(entry_task_ids) or len(set(entry_task_ids)) != len(
        entry_task_ids
    ):
        raise P15090RetirementError("legacy queue entry IDs are not unique and sorted")
    entry_count = snapshot.get("entry_count")
    if (
        not isinstance(entry_count, int)
        or isinstance(entry_count, bool)
        or entry_count != len(entry_task_ids)
    ):
        raise P15090RetirementError("legacy queue entry count drifted")
    return {
        "queue_id": legacy.QUEUE_ID,
        "queue_name": legacy.QUEUE_NAME,
        "entry_task_ids": entry_task_ids,
        "entry_count": entry_count,
    }


def _read_legacy_queue_snapshot(
    legacy: ModuleType, task_class: object
) -> dict[str, object]:
    """Read the authoritative queue-entry inventory through a GET-only request."""

    session_getter = getattr(task_class, "_get_default_session", None)
    if not callable(session_getter):
        raise P15090RetirementError("ClearML cannot expose a read-only backend session")
    try:
        from clearml.backend_api.services.v2_13 import queues
    except ImportError as error:
        raise P15090RetirementError(
            f"ClearML queue API is unavailable: {error}"
        ) from None
    sender = getattr(session_getter(), "send", None)
    if not callable(sender):
        raise P15090RetirementError(
            "ClearML backend session cannot send read-only requests"
        )
    response = sender(
        queues.GetAllRequest(
            id=[legacy.QUEUE_ID],
            page_size=2,
            only_fields=["id", "name", "entries.task"],
        )
    )
    response_data = getattr(response, "response_data", None)
    if not isinstance(response_data, Mapping):
        raise P15090RetirementError("legacy queue lookup returned no response data")
    rows = response_data.get("queues")
    if not isinstance(rows, list) or len(rows) != 1:
        raise P15090RetirementError("legacy queue identity is missing or ambiguous")
    row = rows[0]
    if not isinstance(row, Mapping):
        raise P15090RetirementError("legacy queue response is not a mapping")
    entries = row.get("entries")
    if not isinstance(entries, list):
        raise P15090RetirementError("legacy queue entries are not readable")
    entry_task_ids: list[str] = []
    for item in entries:
        if isinstance(item, Mapping):
            raw_id = item.get("task") or item.get("id")
        else:
            raw_id = item
        task_id = str(raw_id or "")
        if re.fullmatch(r"[0-9a-f]{32}", task_id) is None:
            raise P15090RetirementError("legacy queue contains an invalid task ID")
        entry_task_ids.append(task_id)
    snapshot = {
        "queue_id": row.get("id"),
        "queue_name": row.get("name"),
        "entry_task_ids": sorted(entry_task_ids),
        "entry_count": len(entry_task_ids),
    }
    return _validate_legacy_queue_snapshot(snapshot, legacy)


def _validate_controller_artifacts(
    legacy: ModuleType,
    controller: object,
) -> tuple[dict[str, object], dict[str, object]]:
    artifacts = legacy._artifact_inventory(controller)
    expected = {
        "p1_multiseed_plan",
        "p1_multiseed_pinset",
        legacy.JOURNAL_ARTIFACT,
    }
    allowed_stages = (
        expected,
        expected | {RETIREMENT_INTENT_ARTIFACT},
        expected | {RETIREMENT_INTENT_ARTIFACT, RETIREMENT_ARTIFACT},
    )
    if set(artifacts) not in allowed_stages:
        raise P15090RetirementError(
            "controller artifact inventory differs from the sealed retirement stages"
        )
    plan = legacy.build_plan()
    pinset = legacy.build_pinset()
    if (
        legacy._artifact_mapping(artifacts["p1_multiseed_plan"], "P1 multiseed plan")
        != plan
    ):
        raise P15090RetirementError("controller plan artifact drifted")
    if (
        legacy._artifact_mapping(
            artifacts["p1_multiseed_pinset"], "P1 multiseed pinset"
        )
        != pinset
    ):
        raise P15090RetirementError("controller pinset artifact drifted")
    journal = legacy._validate_journal(
        legacy._artifact_mapping(
            artifacts[legacy.JOURNAL_ARTIFACT], "P1 execution journal"
        ),
        plan=plan,
        execution_key=EXECUTION_KEY,
        controller_id=CONTROLLER_ID,
    )
    rows = journal["tasks"]
    child_rows = [row for row in rows if row["task_key"] == CHILD_TASK_KEY]
    other_rows = [row for row in rows if row["task_key"] != CHILD_TASK_KEY]
    if (
        len(child_rows) != 1
        or child_rows[0]["task_id"] != CHILD_ID
        or child_rows[0]["state"] not in {"prepared", "enqueued", "active"}
        or child_rows[0]["server_status"] not in {"created", "queued"}
        or any(
            row["task_id"] is not None
            or row["state"] != "absent"
            or row["server_status"] != "absent"
            for row in other_rows
        )
    ):
        raise P15090RetirementError("legacy controller journal scope drifted")
    return artifacts, journal


def _retirement_intent(
    *,
    original_controller_artifacts: Sequence[str],
) -> dict[str, object]:
    intent: dict[str, object] = {
        "schema_version": 1,
        "document_type": RETIREMENT_DOCUMENT,
        "phase": "retiring",
        "status": "retiring",
        "reason": "homogeneous_a100_migration",
        "legacy": {
            "execution_key": EXECUTION_KEY,
            "executor_sha256": LEGACY_EXECUTOR_SHA256,
            "controller_task_id": CONTROLLER_ID,
            "child_task_id": CHILD_ID,
            "child_task_key": CHILD_TASK_KEY,
            "queue_id": "5a84454c072349069e7b61af38637c6d",
            "observed_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
            "mutex_name": LEGACY_MUTEX_NAME,
            "original_controller_artifacts": sorted(original_controller_artifacts),
        },
        "target_a100_plan_seal_sha256": A100_PLAN_SEAL_SHA256,
        "target_a100_pinset_seal_sha256": A100_PINSET_SEAL_SHA256,
        "target_a100_execution_key": A100_EXECUTION_KEY,
        "intent": {
            "install_controller_quarantine_barrier": True,
            "dequeue_only_if_never_claimed": True,
            "quarantine_created_legacy_scope": True,
            "never_stop_or_archive_tasks": True,
        },
    }
    intent["seal_sha256"] = _seal(intent)
    return intent


def _validate_intent(
    value: Mapping[str, object],
) -> dict[str, object]:
    intent = dict(value)
    _require_exact_keys(
        intent,
        {
            "document_type",
            "intent",
            "legacy",
            "phase",
            "reason",
            "schema_version",
            "seal_sha256",
            "status",
            "target_a100_execution_key",
            "target_a100_pinset_seal_sha256",
            "target_a100_plan_seal_sha256",
        },
        "retirement intent",
    )
    if intent.get("seal_sha256") != _seal(intent):
        raise P15090RetirementError("retirement intent seal mismatch")
    if (
        intent.get("schema_version") != 1
        or intent.get("document_type") != RETIREMENT_DOCUMENT
        or intent.get("phase") != "retiring"
        or intent.get("status") != "retiring"
        or intent.get("reason") != "homogeneous_a100_migration"
    ):
        raise P15090RetirementError("retirement intent header drifted")
    legacy = intent.get("legacy")
    expected_legacy = {
        "execution_key": EXECUTION_KEY,
        "executor_sha256": LEGACY_EXECUTOR_SHA256,
        "controller_task_id": CONTROLLER_ID,
        "child_task_id": CHILD_ID,
        "child_task_key": CHILD_TASK_KEY,
        "queue_id": "5a84454c072349069e7b61af38637c6d",
        "observed_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
        "mutex_name": LEGACY_MUTEX_NAME,
    }
    if not isinstance(legacy, Mapping) or any(
        legacy.get(field) != expected for field, expected in expected_legacy.items()
    ):
        raise P15090RetirementError("retirement intent legacy binding drifted")
    _require_exact_keys(
        legacy,
        set(expected_legacy) | {"original_controller_artifacts"},
        "retirement intent legacy binding",
    )
    if sorted(legacy.get("original_controller_artifacts") or []) != [
        "p1_multiseed_execution_journal",
        "p1_multiseed_pinset",
        "p1_multiseed_plan",
    ]:
        raise P15090RetirementError(
            "retirement intent original artifact inventory drifted"
        )
    expected_target = {
        "target_a100_plan_seal_sha256": A100_PLAN_SEAL_SHA256,
        "target_a100_pinset_seal_sha256": A100_PINSET_SEAL_SHA256,
        "target_a100_execution_key": A100_EXECUTION_KEY,
    }
    if any(
        intent.get(field) != expected for field, expected in expected_target.items()
    ):
        raise P15090RetirementError("retirement intent A100 target drifted")
    if intent.get("intent") != {
        "install_controller_quarantine_barrier": True,
        "dequeue_only_if_never_claimed": True,
        "quarantine_created_legacy_scope": True,
        "never_stop_or_archive_tasks": True,
    }:
        raise P15090RetirementError("retirement intent action set drifted")
    return intent


def _retirement_manifest(
    *,
    original_controller_artifacts: Sequence[str],
) -> dict[str, object]:
    intent = _retirement_intent(
        original_controller_artifacts=original_controller_artifacts
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "document_type": RETIREMENT_DOCUMENT,
        "phase": "retired",
        "status": "retired",
        "reason": "homogeneous_a100_migration",
        "execution_key": EXECUTION_KEY,
        "legacy_executor_sha256": LEGACY_EXECUTOR_SHA256,
        "controller_task_id": CONTROLLER_ID,
        "child_task_id": CHILD_ID,
        "child_task_key": CHILD_TASK_KEY,
        "legacy_queue_id": "5a84454c072349069e7b61af38637c6d",
        "legacy_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
        "legacy_mutex_name": LEGACY_MUTEX_NAME,
        "retirement_intent_seal_sha256": intent["seal_sha256"],
        "intent_target_a100_executor_sha256": INTENT_TARGET_A100_EXECUTOR_SHA256,
        "final_target_a100_executor_sha256": A100_EXECUTOR_SHA256,
        "target_a100_executor_amendment_reason": A100_EXECUTOR_AMENDMENT_REASON,
        "target_a100_plan_seal_sha256": A100_PLAN_SEAL_SHA256,
        "target_a100_pinset_seal_sha256": A100_PINSET_SEAL_SHA256,
        "target_a100_execution_key": A100_EXECUTION_KEY,
        "observed_initial_state": {
            "controller_status": "created",
            "child_status": "queued",
            "legacy_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
            "original_controller_artifacts": sorted(original_controller_artifacts),
        },
        "retirement_proofs": {
            "controller_barrier_tag": RETIREMENT_TAG,
            "legacy_quarantine_tag": "resilient-v2x-p1-quarantined-orphan",
            "child_dequeued_to_created_state": True,
            "child_never_claimed_by_worker": True,
            "child_artifact_count": 0,
            "child_output_model_count": 0,
            "other_seven_stable_children_absent": True,
            "no_queued_or_running_scoped_child": True,
            "queue_residency_authority": "clearml_queue_entries",
            "child_historical_execution_queue_id": "5a84454c072349069e7b61af38637c6d",
            "child_absent_from_legacy_queue_entries": True,
            "legacy_queue_entry_task_ids": [],
            "legacy_queue_entry_count": 0,
            "final_child_status": "created",
            "final_controller_status": "created",
            "final_tasks_archived": False,
            "final_tasks_quarantined": True,
        },
    }
    manifest["seal_sha256"] = _seal(manifest)
    return manifest


def _validate_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    manifest = dict(value)
    _require_exact_keys(
        manifest,
        {
            "child_task_id",
            "child_task_key",
            "controller_task_id",
            "document_type",
            "execution_key",
            "final_target_a100_executor_sha256",
            "intent_target_a100_executor_sha256",
            "legacy_executor_sha256",
            "legacy_mutex_name",
            "legacy_mutex_queue_id",
            "legacy_queue_id",
            "observed_initial_state",
            "phase",
            "reason",
            "retirement_intent_seal_sha256",
            "retirement_proofs",
            "schema_version",
            "seal_sha256",
            "status",
            "target_a100_execution_key",
            "target_a100_executor_amendment_reason",
            "target_a100_pinset_seal_sha256",
            "target_a100_plan_seal_sha256",
        },
        "retirement manifest",
    )
    if manifest.get("seal_sha256") != _seal(manifest):
        raise P15090RetirementError("retirement manifest seal mismatch")
    required = {
        "schema_version": 1,
        "document_type": RETIREMENT_DOCUMENT,
        "phase": "retired",
        "status": "retired",
        "reason": "homogeneous_a100_migration",
        "execution_key": EXECUTION_KEY,
        "legacy_executor_sha256": LEGACY_EXECUTOR_SHA256,
        "controller_task_id": CONTROLLER_ID,
        "child_task_id": CHILD_ID,
        "child_task_key": CHILD_TASK_KEY,
        "legacy_queue_id": "5a84454c072349069e7b61af38637c6d",
        "legacy_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
        "legacy_mutex_name": LEGACY_MUTEX_NAME,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise P15090RetirementError(f"retirement manifest {field} drifted")
    expected_target = {
        "target_a100_plan_seal_sha256": A100_PLAN_SEAL_SHA256,
        "target_a100_pinset_seal_sha256": A100_PINSET_SEAL_SHA256,
        "target_a100_execution_key": A100_EXECUTION_KEY,
        "intent_target_a100_executor_sha256": INTENT_TARGET_A100_EXECUTOR_SHA256,
        "final_target_a100_executor_sha256": A100_EXECUTOR_SHA256,
        "target_a100_executor_amendment_reason": A100_EXECUTOR_AMENDMENT_REASON,
    }
    if any(
        manifest.get(field) != expected for field, expected in expected_target.items()
    ):
        raise P15090RetirementError("retirement manifest A100 target drifted")
    initial = manifest.get("observed_initial_state")
    if initial != {
        "controller_status": "created",
        "child_status": "queued",
        "legacy_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
        "original_controller_artifacts": [
            "p1_multiseed_execution_journal",
            "p1_multiseed_pinset",
            "p1_multiseed_plan",
        ],
    }:
        raise P15090RetirementError("retirement manifest initial-state binding drifted")
    expected_intent = _retirement_intent(
        original_controller_artifacts=initial["original_controller_artifacts"]
    )
    if manifest.get("retirement_intent_seal_sha256") != expected_intent["seal_sha256"]:
        raise P15090RetirementError("retirement manifest intent link drifted")
    proofs = manifest.get("retirement_proofs")
    expected_proofs = {
        "controller_barrier_tag": RETIREMENT_TAG,
        "legacy_quarantine_tag": "resilient-v2x-p1-quarantined-orphan",
        "child_dequeued_to_created_state": True,
        "child_never_claimed_by_worker": True,
        "child_artifact_count": 0,
        "child_output_model_count": 0,
        "other_seven_stable_children_absent": True,
        "no_queued_or_running_scoped_child": True,
        "queue_residency_authority": "clearml_queue_entries",
        "child_historical_execution_queue_id": "5a84454c072349069e7b61af38637c6d",
        "child_absent_from_legacy_queue_entries": True,
        "legacy_queue_entry_task_ids": [],
        "legacy_queue_entry_count": 0,
        "final_child_status": "created",
        "final_controller_status": "created",
        "final_tasks_archived": False,
        "final_tasks_quarantined": True,
    }
    if not isinstance(proofs, Mapping) or dict(proofs) != expected_proofs:
        raise P15090RetirementError("retirement manifest proof set drifted")
    return manifest


def _inventory(
    legacy: ModuleType,
    task_class: object,
    *,
    queue_reader: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    controller_name = legacy._controller_name(EXECUTION_KEY)
    controller_matches = legacy._query_named_tasks(task_class, name=controller_name)
    if len(controller_matches) != 1:
        raise P15090RetirementError(
            "retirement requires exactly one hidden-inclusive stable controller"
        )
    controller = controller_matches[0]
    _exact_id(_task_id(controller, "controller"), CONTROLLER_ID, "controller")

    names = _expected_names(legacy)
    children_by_key: dict[str, list[object]] = {
        key: legacy._query_named_tasks(task_class, name=name)
        for key, name in names.items()
    }
    if len(children_by_key[CHILD_TASK_KEY]) != 1 or any(
        rows for key, rows in children_by_key.items() if key != CHILD_TASK_KEY
    ):
        raise P15090RetirementError(
            "hidden-inclusive stable child inventory is not exactly one plus seven absent"
        )
    child = children_by_key[CHILD_TASK_KEY][0]
    _exact_id(_task_id(child, "child"), CHILD_ID, "child")

    direct = legacy._query_direct_children(task_class, CONTROLLER_ID)
    if {_task_id(task, "controller child") for task in direct} != {CHILD_ID}:
        raise P15090RetirementError("controller direct-child inventory drifted")
    tagged = legacy._query_execution_tagged_tasks(task_class, EXECUTION_KEY)
    if {_task_id(task, "execution-tagged task") for task in tagged} != {
        CONTROLLER_ID,
        CHILD_ID,
    }:
        raise P15090RetirementError("global execution-tag inventory drifted")

    expected_tag = legacy._execution_tag(EXECUTION_KEY)
    controller_tags = legacy._task_tags(controller)
    child_tags = legacy._task_tags(child)
    if (
        legacy._task_project(controller) != legacy.PROJECT_ID
        or legacy._task_parent(controller) != legacy.SELECTOR_TASK_ID
        or _task_name(controller) != controller_name
        or legacy._task_queue(controller)
        or expected_tag not in controller_tags
        or legacy.CONTROLLER_TAG not in controller_tags
    ):
        raise P15090RetirementError("controller identity binding drifted")
    if (
        legacy._task_project(child) != legacy.PROJECT_ID
        or legacy._task_parent(child) != CONTROLLER_ID
        or _task_name(child) != names[CHILD_TASK_KEY]
        or expected_tag not in child_tags
        or f"p1-task-key:{CHILD_TASK_KEY}" not in child_tags
    ):
        raise P15090RetirementError("child identity binding drifted")

    controller_execution_tags = {
        item
        for item in controller_tags
        if item.startswith(legacy.EXECUTION_KEY_TAG_PREFIX)
    }
    child_execution_tags = {
        item for item in child_tags if item.startswith(legacy.EXECUTION_KEY_TAG_PREFIX)
    }
    child_key_tags = {item for item in child_tags if item.startswith("p1-task-key:")}
    if (
        controller_execution_tags != {expected_tag}
        or child_execution_tags != {expected_tag}
        or child_key_tags != {f"p1-task-key:{CHILD_TASK_KEY}"}
    ):
        raise P15090RetirementError("execution or child-key tags drifted")

    controller_status = legacy._task_status(controller)
    child_status = legacy._task_status(child)
    if controller_status != "created":
        raise P15090RetirementError(
            f"controller status {controller_status!r} is outside created quarantine"
        )
    if child_status not in {"queued", "created"}:
        raise P15090RetirementError(
            f"child status {child_status!r} forbids never-ran retirement"
        )
    child_execution_queue = legacy._task_queue(child)
    if child_execution_queue != legacy.QUEUE_ID:
        raise P15090RetirementError("child historical execution.queue binding drifted")
    legacy_queue = _validate_legacy_queue_snapshot(queue_reader(), legacy)
    queue_entry_task_ids = legacy_queue["entry_task_ids"]
    child_queue_resident = CHILD_ID in queue_entry_task_ids
    if (child_status == "queued") != child_queue_resident:
        raise P15090RetirementError(
            "child status conflicts with authoritative queue-entry residency"
        )
    _validate_child_never_ran(legacy, child)

    retirement_in_controller = RETIREMENT_TAG in controller_tags
    orphan_in_controller = legacy.ORPHAN_TAG in controller_tags
    retirement_in_child = RETIREMENT_TAG in child_tags
    orphan_in_child = legacy.ORPHAN_TAG in child_tags
    if retirement_in_controller != orphan_in_controller:
        raise P15090RetirementError(
            "controller retirement barrier tags are only partially present"
        )
    if retirement_in_child != orphan_in_child:
        raise P15090RetirementError(
            "child retirement barrier tags are only partially present"
        )
    if retirement_in_child and not retirement_in_controller:
        raise P15090RetirementError("child retirement preceded controller barrier")
    if child_status != "queued" and not retirement_in_controller:
        raise P15090RetirementError("unqueued child has no controller barrier")
    artifacts, journal = _validate_controller_artifacts(legacy, controller)
    has_intent = RETIREMENT_INTENT_ARTIFACT in artifacts
    has_manifest = RETIREMENT_ARTIFACT in artifacts
    if has_manifest and not has_intent:
        raise P15090RetirementError("retirement completion has no durable intent")
    if retirement_in_controller and not has_intent:
        raise P15090RetirementError("controller tags preceded retirement intent")
    if child_status != "queued" and not has_intent:
        raise P15090RetirementError("child mutation preceded retirement intent")
    if has_manifest and (
        child_status != "created"
        or child_queue_resident
        or legacy_queue["entry_count"] != 0
        or not retirement_in_controller
        or not retirement_in_child
    ):
        raise P15090RetirementError(
            "retirement completion preceded exact created quarantine"
        )

    controller_archived = legacy._task_archived(controller)
    child_archived = legacy._task_archived(child)
    if controller_archived or child_archived:
        raise P15090RetirementError(
            "quarantined-created retirement must not archive either task"
        )

    return {
        "controller": controller,
        "child": child,
        "controller_status": controller_status,
        "child_status": child_status,
        "child_execution_queue": child_execution_queue,
        "child_queue_resident": child_queue_resident,
        "legacy_queue": legacy_queue,
        "controller_tags": controller_tags,
        "child_tags": child_tags,
        "controller_archived": controller_archived,
        "child_archived": child_archived,
        "controller_artifacts": artifacts,
        "journal": journal,
        "has_manifest": has_manifest,
        "has_intent": has_intent,
        "barrier": retirement_in_controller,
        "child_tagged": retirement_in_child,
    }


def _receipt(
    legacy: ModuleType,
    task_class: object,
    *,
    readonly: bool,
    local_lock: Mapping[str, object] | None,
    lease: Mapping[str, object],
    mutex_journal: Mapping[str, object],
    queue_reader: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    state = _inventory(legacy, task_class, queue_reader=queue_reader)
    final = bool(
        state["controller_status"] == "created"
        and state["child_status"] == "created"
        and not state["child_queue_resident"]
        and state["legacy_queue"]["entry_count"] == 0
        and not state["controller_archived"]
        and not state["child_archived"]
        and state["barrier"]
        and state["child_tagged"]
        and state["has_manifest"]
    )
    manifest_seal = None
    intent_seal = None
    if state["has_intent"]:
        intent = _validate_intent(
            legacy._artifact_mapping(
                state["controller_artifacts"][RETIREMENT_INTENT_ARTIFACT],
                "P1 5090 retirement intent",
            )
        )
        intent_seal = intent["seal_sha256"]
    if state["has_manifest"]:
        manifest = _validate_manifest(
            legacy._artifact_mapping(
                state["controller_artifacts"][RETIREMENT_ARTIFACT],
                "P1 5090 retirement manifest",
            )
        )
        if manifest["retirement_intent_seal_sha256"] != intent_seal:
            raise P15090RetirementError(
                "retirement completion does not link the durable intent"
            )
        manifest_seal = manifest["seal_sha256"]
    blockers: list[str] = []
    if local_lock is not None and not local_lock.get("acquirable"):
        blockers.append(str(local_lock.get("reason") or "legacy lock unavailable"))
    lease_status = lease.get("status")
    journal_status = mutex_journal.get("status")
    if lease_status == "held" and (
        journal_status != "present"
        or lease.get("queue_id") != mutex_journal.get("queue_id")
    ):
        raise P15090RetirementError(
            "held legacy mutex has no matching sealed local journal"
        )
    if lease_status == "available" and journal_status == "present":
        blockers.append(
            "server mutex is absent but its sealed local journal still needs cleanup"
        )
    if lease_status not in {"available", "held"} or journal_status not in {
        "absent",
        "present",
    }:
        raise P15090RetirementError("legacy mutex or journal status drifted")
    unrelated_queue_entries = sorted(
        set(state["legacy_queue"]["entry_task_ids"]) - {CHILD_ID}
    )
    if unrelated_queue_entries:
        blockers.append("legacy RTX 5090 queue contains tasks outside the sealed child")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_5090_retirement_receipt",
        "status": "retired" if final else "validated",
        "readonly": readonly,
        "remote_mutation_count": 0 if readonly else None,
        "execution_key": EXECUTION_KEY,
        "controller": {
            "task_id": CONTROLLER_ID,
            "status": state["controller_status"],
            "archived": state["controller_archived"],
            "barrier_installed": state["barrier"],
            "retirement_intent_installed": state["has_intent"],
        },
        "child": {
            "task_id": CHILD_ID,
            "task_key": CHILD_TASK_KEY,
            "status": state["child_status"],
            "historical_execution_queue_id": state["child_execution_queue"],
            "queue_resident": state["child_queue_resident"],
            "archived": state["child_archived"],
            "retirement_tagged": state["child_tagged"],
            "last_worker": "",
            "artifact_count": 0,
            "output_model_count": 0,
        },
        "exact_inventory": {
            "controller_count": 1,
            "child_count": 1,
            "other_stable_child_count": 0,
            "execution_tagged_task_ids": [CONTROLLER_ID, CHILD_ID],
            "no_queued_or_running_child": (
                state["child_status"] == "created" and not state["child_queue_resident"]
            ),
        },
        "legacy_queue": dict(state["legacy_queue"]),
        "local_lock": dict(local_lock) if local_lock is not None else None,
        "global_mutex": _public_lease(lease),
        "local_mutex_journal": dict(mutex_journal),
        "retirement_manifest_seal_sha256": manifest_seal,
        "retirement_intent_seal_sha256": intent_seal,
        "ready_for_retirement": not blockers and not final,
        "blockers": blockers,
    }
    receipt["seal_sha256"] = _seal(receipt)
    return receipt


def preflight(
    *,
    legacy: ModuleType | None = None,
    task_class: object | None = None,
    api_client: object | None = None,
    a100_validator: Callable[[], Mapping[str, object]] | None = None,
    queue_reader: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Read only: validate the exact retirement inventory and local lock."""

    (a100_validator or _load_a100_target_contract)()
    module = legacy or _load_legacy()
    Task = task_class or module._load_clearml()
    client = api_client or module._api_client()
    read_queue = queue_reader or (lambda: _read_legacy_queue_snapshot(module, Task))
    local_lock = _local_lock_snapshot(module)
    lease = module._read_lease_snapshot(EXECUTION_KEY, api_client=client)
    journal = _mutex_journal_snapshot(module)
    receipt = _receipt(
        module,
        Task,
        readonly=True,
        local_lock=local_lock,
        lease=lease,
        mutex_journal=journal,
        queue_reader=read_queue,
    )
    if (
        receipt["status"] != "retired"
        and not receipt["controller"]["retirement_intent_installed"]
    ):
        # Before the irreversible barrier, retain the legacy executor's full
        # Source-D, teacher, anchor, queue, and global-duplicate validation.
        module.preflight(
            task_class=Task,
            lease_reader=lambda _key: lease,
        )
    return receipt


def _add_barrier_tags(
    legacy: ModuleType,
    task_class: object,
    task: object,
    *,
    task_id: str,
) -> object:
    tags = legacy._task_tags(task)
    required = {RETIREMENT_TAG, legacy.ORPHAN_TAG}
    if not required.issubset(tags):
        legacy._add_tags(task, sorted(required - tags))
        task = legacy._fresh(task_class, task_id)
    if not required.issubset(legacy._task_tags(task)):
        raise P15090RetirementError("retirement barrier tags did not round-trip")
    return task


def _dequeue_child(
    task_class: object,
    legacy: ModuleType,
    child: object,
    *,
    queue_reader: Callable[[], Mapping[str, object]],
) -> object:
    child = legacy._fresh(task_class, CHILD_ID)
    _validate_child_never_ran(legacy, child)
    status = legacy._task_status(child)
    if legacy._task_queue(child) != legacy.QUEUE_ID:
        raise P15090RetirementError("child historical execution.queue binding drifted")
    queue = _validate_legacy_queue_snapshot(queue_reader(), legacy)
    child_queue_resident = CHILD_ID in queue["entry_task_ids"]
    if status == "created" and not child_queue_resident:
        return child
    if status != "queued" or not child_queue_resident:
        raise P15090RetirementError(
            "child is not exactly queued on the sealed RTX 5090 queue"
        )
    dequeue = getattr(task_class, "dequeue", None)
    if not callable(dequeue):
        raise P15090RetirementError("ClearML Task.dequeue is unavailable")
    caught: Exception | None = None
    response: object | None = None
    try:
        response = dequeue(task=child)
    except Exception as error:  # Fresh server state remains authoritative.
        caught = error
    child = legacy._fresh(task_class, CHILD_ID)
    try:
        queue = _validate_legacy_queue_snapshot(queue_reader(), legacy)
        if (
            legacy._task_status(child) != "created"
            or legacy._task_queue(child) != legacy.QUEUE_ID
            or CHILD_ID in queue["entry_task_ids"]
        ):
            raise P15090RetirementError(
                "dequeue did not produce an exact created child absent from queue entries"
            )
        _validate_child_never_ran(legacy, child)
    except Exception as verification_error:
        if caught is not None:
            verification_error.add_note(f"Task.dequeue also raised: {caught}")
        raise
    if caught is not None:
        # ClearML can lose the response after committing.  The fresh, exact
        # never-ran state is a stronger acknowledgement than the transport.
        return child
    data = getattr(response, "response_data", response)
    if isinstance(data, Mapping):
        dequeued = data.get("dequeued")
        updated = data.get("updated")
        if dequeued not in (None, 1) or updated not in (None, 1):
            raise P15090RetirementError(
                "Task.dequeue response conflicts with the fresh server state"
            )
    return child


def _retirement_complete(state: Mapping[str, object]) -> bool:
    return bool(
        state["controller_status"] == "created"
        and state["child_status"] == "created"
        and not state["child_queue_resident"]
        and state["legacy_queue"]["entry_count"] == 0
        and not state["controller_archived"]
        and not state["child_archived"]
        and state["barrier"]
        and state["child_tagged"]
        and state["has_manifest"]
    )


def _reconcile_local_mutex_journal(
    legacy: ModuleType,
    *,
    lease: Mapping[str, object],
    journal: Mapping[str, object],
) -> dict[str, object]:
    lease_status = lease.get("status")
    journal_status = journal.get("status")
    if lease_status not in {"available", "held"}:
        raise P15090RetirementError("legacy mutex lease status is invalid")
    if journal_status not in {"absent", "present"}:
        raise P15090RetirementError("legacy local mutex journal status is invalid")
    if lease_status == "held":
        if journal_status != "present" or lease.get("queue_id") != journal.get(
            "queue_id"
        ):
            raise P15090RetirementError(
                "held legacy mutex has no matching sealed local journal"
            )
        return dict(journal)
    if journal_status == "absent":
        return dict(journal)

    # The server-side delete can commit while its response is lost, leaving
    # only the sealed local receipt.  Under the legacy local flock, an exact
    # globally-available lease proves this journal is stale and safe to remove.
    try:
        legacy._remove_mutex_receipt()
    except Exception as error:
        raise P15090RetirementError(
            f"cannot remove proven-stale local mutex journal: {error}"
        ) from error
    refreshed = _mutex_journal_snapshot(legacy)
    if refreshed.get("status") != "absent":
        raise P15090RetirementError("stale local mutex journal removal did not persist")
    return refreshed


@contextmanager
def _legacy_local_lock(legacy: ModuleType) -> Iterator[None]:
    try:
        with legacy._execution_lock():
            yield
    except P15090RetirementError:
        raise
    except Exception as error:
        raise P15090RetirementError(str(error)) from error


def retire(
    *,
    authorization_token: str,
    legacy: ModuleType | None = None,
    task_class: object | None = None,
    api_client: object | None = None,
    server_mutex: Callable[..., object] | None = None,
    a100_validator: Callable[[], Mapping[str, object]] | None = None,
    queue_reader: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Retire the exact never-run child; fail closed on any active task."""

    if authorization_token != RETIREMENT_TOKEN:
        raise P15090RetirementError("remote retirement token mismatch")
    validate_a100 = a100_validator or _load_a100_target_contract
    validate_a100()
    module = legacy or _load_legacy()
    Task = task_class or module._load_clearml()
    client = api_client or module._api_client()
    mutex_factory = server_mutex or module._server_execution_mutex
    read_queue = queue_reader or (lambda: _read_legacy_queue_snapshot(module, Task))

    with _legacy_local_lock(module):
        remote_mutation_count = 0
        initial = _inventory(module, Task, queue_reader=read_queue)
        initial_lease = module._read_lease_snapshot(EXECUTION_KEY, api_client=client)
        initial_journal = _mutex_journal_snapshot(module)
        initial_journal = _reconcile_local_mutex_journal(
            module,
            lease=initial_lease,
            journal=initial_journal,
        )
        if _retirement_complete(initial):
            if (
                initial_lease.get("status") == "available"
                and initial_journal.get("status") == "absent"
            ):
                result = _receipt(
                    module,
                    Task,
                    readonly=False,
                    local_lock=None,
                    lease=initial_lease,
                    mutex_journal=initial_journal,
                    queue_reader=read_queue,
                )
                result["remote_mutation_count"] = 0
                result["seal_sha256"] = _seal(result)
                return result
            if not (
                initial_lease.get("status") == "held"
                and initial_journal.get("status") == "present"
            ):
                raise P15090RetirementError(
                    "completed retirement has an inconsistent leftover mutex"
                )

        if not initial["has_intent"]:
            # This is the last point where the legacy full preflight is valid.
            module.preflight(
                task_class=Task,
                lease_reader=lambda _key: initial_lease,
            )

        remote_mutation_count += 2 if initial_lease.get("status") == "available" else 1
        with mutex_factory(EXECUTION_KEY, api_client=client) as mutex:
            mutex_queue_id = str(mutex.get("queue_id") or "")
            if re.fullmatch(r"[0-9a-f]{32}", mutex_queue_id) is None:
                raise P15090RetirementError(
                    "retirement mutex returned no exact queue ID"
                )
            validate_a100()
            state = _inventory(module, Task, queue_reader=read_queue)
            if not state["has_intent"]:
                # Re-run the full legacy read-only validation inside the
                # company-global mutex immediately before the first mutation.
                current_lease = module._read_lease_snapshot(
                    EXECUTION_KEY, api_client=client
                )
                module.preflight(
                    task_class=Task,
                    lease_reader=lambda _key: current_lease,
                )
                state = _inventory(module, Task, queue_reader=read_queue)
                intent = _retirement_intent(
                    original_controller_artifacts=sorted(state["controller_artifacts"]),
                )
                module._upload_artifact(
                    state["controller"], RETIREMENT_INTENT_ARTIFACT, intent
                )
                remote_mutation_count += 1
                state = _inventory(module, Task, queue_reader=read_queue)
                if not state["has_intent"]:
                    raise P15090RetirementError("retirement intent did not round-trip")
            observed_intent = module._artifact_mapping(
                state["controller_artifacts"][RETIREMENT_INTENT_ARTIFACT],
                "P1 5090 retirement intent",
            )
            _validate_intent(observed_intent)
            if not state["barrier"]:
                controller = _add_barrier_tags(
                    module,
                    Task,
                    state["controller"],
                    task_id=CONTROLLER_ID,
                )
                del controller
                remote_mutation_count += 1
                state = _inventory(module, Task, queue_reader=read_queue)
                if not state["barrier"]:
                    raise P15090RetirementError(
                        "controller anti-resume barrier is not durable"
                    )

            if state["child_status"] in {"queued", "created"}:
                child_was_queued = state["child_status"] == "queued"
                child = _dequeue_child(
                    Task,
                    module,
                    state["child"],
                    queue_reader=read_queue,
                )
                if child_was_queued:
                    remote_mutation_count += 1
                child_was_tagged = {
                    RETIREMENT_TAG,
                    module.ORPHAN_TAG,
                }.issubset(module._task_tags(child))
                child = _add_barrier_tags(module, Task, child, task_id=CHILD_ID)
                if not child_was_tagged:
                    remote_mutation_count += 1
                _validate_child_never_ran(module, child)
            state = _inventory(module, Task, queue_reader=read_queue)
            if (
                state["child_status"] != "created"
                or state["child_queue_resident"]
                or state["legacy_queue"]["entry_count"] != 0
                or not state["child_tagged"]
            ):
                raise P15090RetirementError(
                    "child did not reach exact created quarantine"
                )
            _validate_child_never_ran(module, state["child"])

            state = _inventory(module, Task, queue_reader=read_queue)
            manifest = _retirement_manifest(
                original_controller_artifacts=sorted(
                    set(state["controller_artifacts"])
                    - {RETIREMENT_INTENT_ARTIFACT, RETIREMENT_ARTIFACT}
                ),
            )
            existing = state["controller_artifacts"].get(RETIREMENT_ARTIFACT)
            if existing is None:
                module._upload_artifact(
                    state["controller"], RETIREMENT_ARTIFACT, manifest
                )
                remote_mutation_count += 1
            else:
                observed = module._artifact_mapping(
                    existing, "P1 5090 retirement manifest"
                )
                if _validate_manifest(observed) != manifest:
                    raise P15090RetirementError(
                        "existing retirement artifact differs from the exact manifest"
                    )

            state = _inventory(module, Task, queue_reader=read_queue)
            if not _retirement_complete(state):
                raise P15090RetirementError(
                    "retirement did not reach the exact final task state"
                )
            observed_manifest = module._artifact_mapping(
                state["controller_artifacts"][RETIREMENT_ARTIFACT],
                "P1 5090 retirement manifest",
            )
            _validate_manifest(observed_manifest)

        final_lease = module._read_lease_snapshot(EXECUTION_KEY, api_client=client)
        final_journal = _mutex_journal_snapshot(module)
        if (
            final_lease.get("status") != "available"
            or final_journal.get("status") != "absent"
        ):
            raise P15090RetirementError(
                "retirement completed but the execution mutex was not released"
            )
        receipt = _receipt(
            module,
            Task,
            readonly=False,
            local_lock=None,
            lease=final_lease,
            mutex_journal=final_journal,
            queue_reader=read_queue,
        )
        receipt["remote_mutation_count"] = remote_mutation_count
        receipt["seal_sha256"] = _seal(receipt)
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="connect read-only and print the exact retirement readiness receipt",
    )
    mode.add_argument(
        "--retire",
        action="store_true",
        help="retire the exact sealed RTX 5090 controller and queued child",
    )
    parser.add_argument(
        "--retirement-token",
        default="",
        help="exact acknowledgement required with --retire",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight:
        if args.retirement_token:
            raise P15090RetirementError(
                "--retirement-token is invalid with --preflight"
            )
        result = preflight()
    elif args.retire:
        if args.retirement_token != RETIREMENT_TOKEN:
            raise P15090RetirementError(
                "--retire requires the exact --retirement-token acknowledgement"
            )
        result = retire(authorization_token=args.retirement_token)
    else:
        if args.retirement_token:
            raise P15090RetirementError(
                "--retirement-token is invalid without --retire"
            )
        result = build_plan()
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            sort_keys=True,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except P15090RetirementError as error:
        print(f"P1-5090-RETIREMENT-ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
