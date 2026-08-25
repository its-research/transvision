#!/usr/bin/env python3
"""Safely resume the sealed P1 A100 campaign with two training slots.

This supervisor is an amendment layer over
``clearml_p1_a100_multiseed_executor.py``.  It never changes the sealed
scientific plan, pinset, execution key, controller, child names, or journal.
The default mode is local dry-run; ``--preflight`` is remote read-only; remote
mutation requires ``--execute`` plus the exact v2 acknowledgement token.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


BASE_EXECUTOR_FILENAME = "clearml_p1_a100_multiseed_executor.py"
FROZEN_BASE_SHA256 = "c6dc31dd324f96b688b7dd82a2d3e50fb8a48cd14658dbf5ff96daed072b1298"
BASE_EXECUTOR_SHA256 = FROZEN_BASE_SHA256
SEALED_PLAN_SHA256 = "43c1076057999ecb48dfa979f4ec6d952a62c4e4c02307cc350e8489a3728446"
SEALED_PINSET_SHA256 = (
    "3d44c31b0fd1f1c2ba4714a59828cd3f25f7ed2e1b64425618fb5e2e5d411eb7"
)
SEALED_EXECUTION_KEY = (
    "c4dfc01775ce835819685112d5bf9f3713ef15e94da5d38717a5f1d443bdb793"
)
EXECUTION_TOKEN = "ENQUEUE_P1_A100_PARALLEL_V2_6_TRAIN_6_EVAL"
AMENDMENT_ARTIFACT = "p1_a100_parallel_v2_amendment"
AMENDMENT_DOCUMENT_TYPE = "resilient_v2x_p1_a100_parallel_v2_amendment"
PREFLIGHT_DOCUMENT_TYPE = "resilient_v2x_p1_a100_parallel_v2_preflight"
PENDING_JOURNAL_DOCUMENT_TYPE = (
    "resilient_v2x_p1_a100_parallel_v2_pending_journal_write"
)
PENDING_JOURNAL_FILENAME = ".p1-a100-parallel-v2-journal-pending.json"
MAX_ACTIVE_TRAINING_TASKS = 2
MAX_ACTIVE_EVALUATION_TASKS = 1
INITIAL_IDLE_WORKER_ID = "10.100.34.18-A100:gpu0,1,2,3"
INITIAL_RUNNING_WORKER_ID = "10.100.34.18-A100:gpu4,5,6,7"
CONTROL_HOST_CONTRACT = {
    "route": "10.100.35.112",
    "hostname": "Ununtu",
    "machine_id_sha256": (
        "ddd6addb6b1401c12c2116b68df076fad9a64fa14cfd0507bf5ca52f46693c55"
    ),
    "clearml_cache_dir": (
        "/home/lbin/Desktop/transvision/work_dirs/orchestration/clearml_cache"
    ),
}


def _load_pinned_base() -> object:
    path = Path(__file__).resolve().with_name(BASE_EXECUTOR_FILENAME)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != BASE_EXECUTOR_SHA256:
        raise RuntimeError(
            "pinned P1 A100 base executor SHA-256 mismatch: "
            f"expected {BASE_EXECUTOR_SHA256}, observed {observed}"
        )
    spec = importlib.util.spec_from_file_location(
        "_resilientv2x_p1_a100_pinned_base", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the pinned P1 A100 base executor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_pinned_base()
P1ExecutorError = base.P1ExecutorError
P1RecoverableTimeout = base.P1RecoverableTimeout


class P1ParallelV2Error(P1ExecutorError):
    """The scheduling amendment or its live safety state is invalid."""


def _verify_frozen_bindings() -> tuple[dict[str, object], dict[str, object]]:
    path = Path(__file__).resolve().with_name(BASE_EXECUTOR_FILENAME)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_BASE_SHA256:
        raise P1ParallelV2Error("frozen base executor SHA-256 drifted")
    plan = base.build_plan()
    pinset = base.build_pinset()
    if plan.get("seal_sha256") != SEALED_PLAN_SHA256:
        raise P1ParallelV2Error("sealed P1 A100 plan drifted")
    if pinset.get("seal_sha256") != SEALED_PINSET_SHA256:
        raise P1ParallelV2Error("sealed P1 A100 pinset drifted")
    if base._execution_key(plan, pinset) != SEALED_EXECUTION_KEY:
        raise P1ParallelV2Error("sealed P1 A100 execution key drifted")
    return plan, pinset


def _supervisor_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _validate_control_host() -> dict[str, str]:
    machine_id_path = Path("/etc/machine-id")
    if not machine_id_path.is_file() or machine_id_path.is_symlink():
        raise P1ParallelV2Error("control host machine-id is unavailable")
    observed = {
        "route": CONTROL_HOST_CONTRACT["route"],
        "hostname": socket.gethostname(),
        "machine_id_sha256": hashlib.sha256(machine_id_path.read_bytes()).hexdigest(),
        "clearml_cache_dir": str(base._validate_cache_directory()),
    }
    if observed != CONTROL_HOST_CONTRACT:
        raise P1ParallelV2Error("parallel-v2 control host binding drifted")
    return observed


def _active_task_ids_from_journal(journal: Mapping[str, object]) -> list[str]:
    rows = journal.get("tasks")
    if not isinstance(rows, list):
        raise P1ParallelV2Error("source journal task inventory is unreadable")
    values: list[str] = []
    for raw in rows:
        row = base._mapping(raw, "source journal row")
        if row.get("server_status") not in {"queued", "in_progress"}:
            continue
        values.append(base._clearml_id(row.get("task_id"), "active child"))
    if len(values) != len(set(values)):
        raise P1ParallelV2Error("source journal contains duplicate active task IDs")
    return sorted(values)


def _build_migration_intent(
    *,
    controller_id: str,
    journal: Mapping[str, object],
    active_task_ids: Sequence[str],
) -> dict[str, object]:
    """Build the immutable scheduling-only amendment for the existing run."""

    plan, pinset = _verify_frozen_bindings()
    execution_key = base._execution_key(plan, pinset)
    controller_id = base._clearml_id(controller_id, "controller")
    validated_journal = base._validate_journal(
        journal,
        plan=plan,
        execution_key=execution_key,
        controller_id=controller_id,
    )
    active_ids = sorted(
        base._clearml_id(value, "active child") for value in active_task_ids
    )
    if len(active_ids) != len(set(active_ids)):
        raise P1ParallelV2Error("migration active task IDs are duplicated")
    if active_ids != _active_task_ids_from_journal(validated_journal):
        raise P1ParallelV2Error(
            "migration active task IDs differ from the sealed source journal"
        )
    amendment: dict[str, object] = {
        "schema_version": 1,
        "document_type": AMENDMENT_DOCUMENT_TYPE,
        "status": "active",
        "reason": "use_both_pinned_4xa100_workers_without_changing_scientific_scope",
        "base_executor_sha256": BASE_EXECUTOR_SHA256,
        "supervisor_sha256": _supervisor_sha256(),
        "plan_seal_sha256": plan["seal_sha256"],
        "pinset_seal_sha256": pinset["seal_sha256"],
        "execution_key": execution_key,
        "controller_task_id": controller_id,
        "control_host_contract": copy.deepcopy(CONTROL_HOST_CONTRACT),
        "source": {
            "execution_key": execution_key,
            "journal_artifact": base.JOURNAL_ARTIFACT,
            "journal_revision": validated_journal["revision"],
            "journal_seal_sha256": validated_journal["seal_sha256"],
            "active_task_ids": active_ids,
            "journal_snapshot": copy.deepcopy(validated_journal),
        },
        "preserved_identity": {
            "controller_name": base._controller_name(execution_key),
            "journal_artifact": base.JOURNAL_ARTIFACT,
            "manifest_artifact": base.MANIFEST_ARTIFACT,
            "child_names": [
                base._task_name(execution_key, str(record[phase]["task_key"]))
                for record in plan["pairs"]
                for phase in ("training", "evaluation")
            ],
        },
        "scheduling_override": {
            "base_max_parallel_tasks": 1,
            "max_active_or_queued_training_tasks": MAX_ACTIVE_TRAINING_TASKS,
            "max_active_or_queued_evaluation_tasks": MAX_ACTIVE_EVALUATION_TASKS,
            "phase_order": ["all_training", "all_evaluation"],
            "evaluation_barrier": "all_six_training_results_validated",
            "training_order": [
                str(record["training"]["task_key"]) for record in plan["pairs"]
            ],
            "evaluation_order": [
                str(record["evaluation"]["task_key"]) for record in plan["pairs"]
            ],
            "initial_second_training_gate": {
                "idle_worker_id": INITIAL_IDLE_WORKER_ID,
                "running_worker_id": INITIAL_RUNNING_WORKER_ID,
                "target_queue_has_no_foreign_entries": True,
                "overlapping_gpu8_queue_frozen": True,
            },
        },
    }
    amendment["seal_sha256"] = base._seal(amendment)
    return _validate_migration_intent(
        amendment,
        controller_id=controller_id,
        journal=validated_journal,
        active_task_ids=active_ids,
    )


def _validate_migration_intent(
    value: Mapping[str, object],
    *,
    controller_id: str,
    journal: Mapping[str, object],
    active_task_ids: Sequence[str],
) -> dict[str, object]:
    """Validate the amendment against the currently pinned base artifacts."""

    amendment = base._mapping(value, "parallel-v2 amendment")
    if amendment.get("seal_sha256") != base._seal(amendment):
        raise P1ParallelV2Error("parallel-v2 amendment seal mismatch")
    if amendment.get("document_type") != AMENDMENT_DOCUMENT_TYPE:
        raise P1ParallelV2Error("parallel-v2 amendment document type drifted")
    if amendment.get("base_executor_sha256") != BASE_EXECUTOR_SHA256:
        raise P1ParallelV2Error("parallel-v2 amendment base executor pin drifted")
    if amendment.get("supervisor_sha256") != _supervisor_sha256():
        raise P1ParallelV2Error("parallel-v2 amendment supervisor pin drifted")
    plan, pinset = _verify_frozen_bindings()
    execution_key = base._execution_key(plan, pinset)
    controller_id = base._clearml_id(controller_id, "controller")
    validated_journal = base._validate_journal(
        journal,
        plan=plan,
        execution_key=execution_key,
        controller_id=controller_id,
    )
    normalized_active_ids = sorted(
        base._clearml_id(value, "active child") for value in active_task_ids
    )
    if normalized_active_ids != _active_task_ids_from_journal(validated_journal):
        raise P1ParallelV2Error(
            "migration active task IDs differ from the sealed source journal"
        )
    bindings = {
        "plan_seal_sha256": plan["seal_sha256"],
        "pinset_seal_sha256": pinset["seal_sha256"],
        "execution_key": execution_key,
        "controller_task_id": controller_id,
    }
    for field, expected_value in bindings.items():
        if amendment.get(field) != expected_value:
            raise P1ParallelV2Error(f"parallel-v2 amendment {field} drifted")
    source = base._mapping(amendment.get("source"), "parallel-v2 source")
    expected_source = {
        "execution_key": execution_key,
        "journal_artifact": base.JOURNAL_ARTIFACT,
        "journal_revision": validated_journal["revision"],
        "journal_seal_sha256": validated_journal["seal_sha256"],
        "active_task_ids": normalized_active_ids,
        "journal_snapshot": copy.deepcopy(validated_journal),
    }
    if source != expected_source:
        raise P1ParallelV2Error("parallel-v2 source journal binding drifted")
    return amendment


def build_amendment(
    *,
    controller_id: str,
    journal: Mapping[str, object],
    active_task_ids: Sequence[str],
) -> dict[str, object]:
    return _build_migration_intent(
        controller_id=controller_id,
        journal=journal,
        active_task_ids=active_task_ids,
    )


def validate_amendment(
    value: Mapping[str, object],
    *,
    controller_id: str,
    journal: Mapping[str, object],
    active_task_ids: Sequence[str],
) -> dict[str, object]:
    return _validate_migration_intent(
        value,
        controller_id=controller_id,
        journal=journal,
        active_task_ids=active_task_ids,
    )


def _validate_installed_amendment(
    value: Mapping[str, object],
    *,
    controller_id: str,
    current_journal: Mapping[str, object],
) -> dict[str, object]:
    """Validate an immutable amendment while allowing journal progress.

    The amendment seals the exact handoff snapshot.  The mutable execution
    journal may advance after the handoff, but a task that already existed in
    the source snapshot may never silently change identity.
    """

    amendment = base._mapping(value, "installed parallel-v2 amendment")
    if amendment.get("seal_sha256") != base._seal(amendment):
        raise P1ParallelV2Error("installed parallel-v2 amendment seal mismatch")
    plan, pinset = _verify_frozen_bindings()
    controller_id = base._clearml_id(controller_id, "controller")
    expected_top = {
        "schema_version": 1,
        "document_type": AMENDMENT_DOCUMENT_TYPE,
        "status": "active",
        "reason": "use_both_pinned_4xa100_workers_without_changing_scientific_scope",
        "base_executor_sha256": FROZEN_BASE_SHA256,
        "supervisor_sha256": _supervisor_sha256(),
        "plan_seal_sha256": SEALED_PLAN_SHA256,
        "pinset_seal_sha256": SEALED_PINSET_SHA256,
        "execution_key": SEALED_EXECUTION_KEY,
        "controller_task_id": controller_id,
        "control_host_contract": CONTROL_HOST_CONTRACT,
    }
    base._exact_keys(
        amendment,
        set(expected_top)
        | {"source", "preserved_identity", "scheduling_override", "seal_sha256"},
        "installed parallel-v2 amendment",
    )
    for field, expected in expected_top.items():
        if amendment.get(field) != expected:
            raise P1ParallelV2Error(f"installed parallel-v2 amendment {field} drifted")

    source = base._mapping(amendment.get("source"), "installed amendment source")
    base._exact_keys(
        source,
        {
            "active_task_ids",
            "execution_key",
            "journal_artifact",
            "journal_revision",
            "journal_seal_sha256",
            "journal_snapshot",
        },
        "installed amendment source",
    )
    snapshot = base._validate_journal(
        base._mapping(source.get("journal_snapshot"), "amendment journal snapshot"),
        plan=plan,
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=controller_id,
    )
    expected_source = {
        "active_task_ids": _active_task_ids_from_journal(snapshot),
        "execution_key": SEALED_EXECUTION_KEY,
        "journal_artifact": base.JOURNAL_ARTIFACT,
        "journal_revision": snapshot["revision"],
        "journal_seal_sha256": snapshot["seal_sha256"],
        "journal_snapshot": snapshot,
    }
    if source != expected_source:
        raise P1ParallelV2Error("installed amendment source snapshot drifted")

    current = base._validate_journal(
        current_journal,
        plan=plan,
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=controller_id,
    )
    if int(current["revision"]) < int(snapshot["revision"]):
        raise P1ParallelV2Error("execution journal predates the migration source")
    current_rows = {
        str(row["task_key"]): base._mapping(row, "current journal row")
        for row in current["tasks"]
    }
    state_rank = {
        "absent": 0,
        "discovered": 1,
        "prepared": 2,
        "enqueued": 3,
        "active": 3,
        "completed": 4,
    }
    for raw in snapshot["tasks"]:
        prior = base._mapping(raw, "source journal row")
        prior_id = prior.get("task_id")
        if prior_id is None:
            continue
        now = current_rows[str(prior["task_key"])]
        if now.get("task_id") != prior_id:
            raise P1ParallelV2Error(
                f"source task identity changed for {prior['task_key']}"
            )
        prior_state = str(prior.get("state") or "")
        now_state = str(now.get("state") or "")
        if prior_state == "quarantined":
            if now_state not in {
                "quarantined",
                "discovered",
                "prepared",
                "active",
                "completed",
            }:
                raise P1ParallelV2Error(
                    f"source task state regressed for {prior['task_key']}"
                )
        elif now_state == "quarantined" or state_rank.get(
            now_state, -1
        ) < state_rank.get(prior_state, -1):
            raise P1ParallelV2Error(
                f"source task state regressed for {prior['task_key']}"
            )

    preserved = base._mapping(
        amendment.get("preserved_identity"), "installed preserved identity"
    )
    expected_preserved = {
        "controller_name": base._controller_name(SEALED_EXECUTION_KEY),
        "journal_artifact": base.JOURNAL_ARTIFACT,
        "manifest_artifact": base.MANIFEST_ARTIFACT,
        "child_names": [
            base._task_name(SEALED_EXECUTION_KEY, str(record[phase]["task_key"]))
            for record in plan["pairs"]
            for phase in ("training", "evaluation")
        ],
    }
    if preserved != expected_preserved:
        raise P1ParallelV2Error("installed amendment identity binding drifted")
    override = base._mapping(
        amendment.get("scheduling_override"), "installed scheduling override"
    )
    expected_override = _build_migration_intent(
        controller_id=controller_id,
        journal=snapshot,
        active_task_ids=expected_source["active_task_ids"],
    )["scheduling_override"]
    if override != expected_override:
        raise P1ParallelV2Error("installed scheduling override drifted")
    return amendment


def _queue_inventory(
    queue_receipt: Mapping[str, object], *, scoped_task_ids: set[str]
) -> dict[str, object]:
    """Normalize the live two-worker inventory without rejecting foreign work."""

    try:
        queue = base._validate_a100_queue_receipt(
            queue_receipt, context="parallel-v2 queue receipt"
        )
    except base.P1ExecutorError as error:
        raise P1ParallelV2Error(str(error)) from error
    queued = queue.get("queued_task_ids")
    count = queue.get("queued_entry_count")
    if (
        type(count) is not int
        or count < 0
        or not isinstance(queued, list)
        or count != len(queued)
    ):
        raise P1ParallelV2Error("target queue entry inventory is unreadable")
    queued_ids = [base._clearml_id(value, "queued task") for value in queued]
    if len(queued_ids) != len(set(queued_ids)):
        raise P1ParallelV2Error("target queue contains duplicate task IDs")

    workers = queue.get("workers")
    if not isinstance(workers, list):
        raise P1ParallelV2Error("target worker inventory is unreadable")
    running_by_worker: dict[str, str | None] = {}
    for raw in workers:
        worker = base._mapping(raw, "parallel-v2 worker")
        worker_id = str(worker.get("worker_id") or "")
        if worker_id not in base.ALLOWED_WORKER_IDS or worker_id in running_by_worker:
            raise P1ParallelV2Error("target worker binding drifted")
        raw_task_id = worker.get("running_task_id")
        running_by_worker[worker_id] = (
            base._clearml_id(raw_task_id, "running task")
            if raw_task_id not in (None, "")
            else None
        )
    if set(running_by_worker) != set(base.ALLOWED_WORKER_IDS):
        raise P1ParallelV2Error("target worker inventory drifted")
    running_ids = [value for value in running_by_worker.values() if value]
    if len(running_ids) != len(set(running_ids)):
        raise P1ParallelV2Error("one task is reported on multiple target workers")
    return {
        "queued_task_ids": queued_ids,
        "scoped_queued_task_ids": [
            value for value in queued_ids if value in scoped_task_ids
        ],
        "foreign_queued_task_ids": [
            value for value in queued_ids if value not in scoped_task_ids
        ],
        "running_by_worker": running_by_worker,
        "scoped_running_task_ids": [
            value for value in running_ids if value in scoped_task_ids
        ],
        "foreign_running_task_ids": [
            value for value in running_ids if value not in scoped_task_ids
        ],
        "idle_worker_ids": sorted(
            worker_id
            for worker_id, task_id in running_by_worker.items()
            if task_id is None
        ),
    }


def _scheduler_decision(
    *,
    journal: Mapping[str, object],
    queue_receipt: Mapping[str, object],
    require_initial_second_gate: bool = False,
) -> dict[str, object]:
    """Return one deterministic scheduling step for the single event loop.

    Foreign work is capacity, not corruption: it produces a blocked decision
    and is polled again.  Unreadable bindings and GPU8 freeze drift remain hard
    failures through ``_queue_inventory``.
    """

    plan, _pinset = _verify_frozen_bindings()
    controller_id = base._clearml_id(
        journal.get("controller_task_id"), "journal controller"
    )
    try:
        validated = base._validate_journal(
            journal,
            plan=plan,
            execution_key=SEALED_EXECUTION_KEY,
            controller_id=controller_id,
        )
    except base.P1ExecutorError as error:
        raise P1ParallelV2Error(str(error)) from error
    raw_rows = validated.get("tasks")
    if not isinstance(raw_rows, list):
        raise P1ParallelV2Error("journal task inventory is unreadable")
    rows = [base._mapping(value, "scheduler journal row") for value in raw_rows]
    by_key = {str(row["task_key"]): row for row in rows}
    training_keys = [str(pair["training"]["task_key"]) for pair in plan["pairs"]]
    evaluation_keys = [str(pair["evaluation"]["task_key"]) for pair in plan["pairs"]]
    scoped_ids = {
        base._clearml_id(row["task_id"], "journal child")
        for row in rows
        if row.get("task_id") is not None and row.get("state") != "quarantined"
    }
    live = _queue_inventory(queue_receipt, scoped_task_ids=scoped_ids)

    failure_statuses = set(base.FAILURE_STATUSES)
    for row in rows:
        status = str(row.get("server_status") or "")
        if row.get("state") == "quarantined":
            task_id = base._clearml_id(
                row.get("task_id"), f"quarantined {row['task_key']} task"
            )
            evidence = base._mapping(
                row.get("result"), f"quarantined {row['task_key']} evidence"
            )
            history = evidence.get("quarantined_task_ids")
            if (
                status not in {"created", "failed"}
                or not isinstance(history, list)
                or task_id not in history
            ):
                raise P1ParallelV2Error(
                    f"quarantined child {row['task_key']} evidence drifted"
                )
            continue
        if status in failure_statuses:
            raise P1ParallelV2Error(
                f"scoped child {row['task_key']} ended with status {status!r}"
            )

    def is_active(row: Mapping[str, object]) -> bool:
        return row.get("server_status") in {"queued", "in_progress"}

    def is_completed(row: Mapping[str, object]) -> bool:
        return (
            row.get("state") == "completed"
            and row.get("server_status") == "completed"
            and isinstance(row.get("result"), Mapping)
        )

    completed_training = [key for key in training_keys if is_completed(by_key[key])]
    active_training = [key for key in training_keys if is_active(by_key[key])]
    active_evaluation = [key for key in evaluation_keys if is_active(by_key[key])]
    if active_evaluation and len(completed_training) != len(training_keys):
        raise P1ParallelV2Error(
            "evaluation is active before all six scoped trainings were validated"
        )
    if len(active_training) > MAX_ACTIVE_TRAINING_TASKS:
        raise P1ParallelV2Error(
            "more than two scoped training tasks are live in parallel"
        )
    if len(active_evaluation) > MAX_ACTIVE_EVALUATION_TASKS:
        raise P1ParallelV2Error("scoped evaluation active/queued cap exceeded")

    if len(completed_training) == len(training_keys):
        phase = "evaluation"
        phase_keys = evaluation_keys
        active_keys = active_evaluation
        cap = MAX_ACTIVE_EVALUATION_TASKS
    else:
        phase = "training"
        phase_keys = training_keys
        active_keys = active_training
        cap = MAX_ACTIVE_TRAINING_TASKS

    decision: dict[str, object] = {
        "phase": phase,
        "active_task_keys": active_keys,
        "enqueue_task_keys": [],
        "blocked_reason": None,
        "foreign_running_task_ids": live["foreign_running_task_ids"],
        "foreign_queued_task_ids": live["foreign_queued_task_ids"],
        "idle_worker_ids": live["idle_worker_ids"],
    }
    journal_active_ids = {
        base._clearml_id(by_key[key].get("task_id"), "active journal child")
        for key in active_training + active_evaluation
    }
    receipt_active_ids = set(live["scoped_queued_task_ids"]) | set(
        live["scoped_running_task_ids"]
    )
    if journal_active_ids != receipt_active_ids:
        decision["blocked_reason"] = "scoped_queue_receipt_lag"
        return decision
    if len(active_keys) >= cap:
        decision["blocked_reason"] = f"{phase}_active_or_queued_cap_reached"
        return decision

    candidates = [
        key
        for key in phase_keys
        if not is_completed(by_key[key]) and not is_active(by_key[key])
    ]
    if not candidates:
        if all(is_completed(by_key[key]) for key in phase_keys):
            decision["blocked_reason"] = "phase_complete"
            return decision
        raise P1ParallelV2Error(
            f"{phase} journal contains no schedulable or active task"
        )

    foreign_queued = list(live["foreign_queued_task_ids"])
    if foreign_queued:
        decision["blocked_reason"] = "foreign_target_queue_entries"
        return decision

    scoped_queued = list(live["scoped_queued_task_ids"])
    idle_workers = list(live["idle_worker_ids"])
    available_dispatch_slots = len(idle_workers) - len(scoped_queued)
    if available_dispatch_slots <= 0:
        decision["blocked_reason"] = (
            "foreign_worker_occupancy"
            if live["foreign_running_task_ids"]
            else "target_worker_capacity_unavailable"
        )
        return decision

    launched_training = [
        key for key in training_keys if by_key[key].get("state") != "absent"
    ]
    initial_expansion = (
        phase == "training"
        and len(completed_training) == 0
        and len(active_training) == 1
        and len(launched_training) == 1
    )
    if initial_expansion and require_initial_second_gate:
        running = base._mapping(live["running_by_worker"], "running workers")
        active_id = base._clearml_id(
            by_key[active_training[0]].get("task_id"), "initial active training"
        )
        if (
            running.get(INITIAL_RUNNING_WORKER_ID) != active_id
            or running.get(INITIAL_IDLE_WORKER_ID) is not None
        ):
            decision["blocked_reason"] = "initial_second_training_gate_waiting"
            return decision

    decision["enqueue_task_keys"] = [candidates[0]]
    return decision


def _ensure_installed_amendment(
    task_class: object,
    controller_id: str,
    journal: Mapping[str, object],
    active_task_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Read the controller fresh and validate its already-installed amendment."""

    del active_task_ids  # The sealed source snapshot is authoritative after install.
    controller_id = base._clearml_id(controller_id, "controller")
    controller = base._fresh(task_class, controller_id)
    artifact = base._artifact_inventory(controller).get(AMENDMENT_ARTIFACT)
    if artifact is None:
        raise P1ParallelV2Error("parallel-v2 amendment is missing")
    try:
        value = base._artifact_mapping(artifact, "parallel-v2 amendment")
        return _validate_installed_amendment(
            value,
            controller_id=controller_id,
            current_journal=journal,
        )
    except base.P1ExecutorError as error:
        if isinstance(error, P1ParallelV2Error):
            raise
        raise P1ParallelV2Error(str(error)) from error


def _install_amendment(
    task_class: object,
    controller: object,
    journal: Mapping[str, object],
) -> dict[str, object]:
    """Install the immutable handoff artifact once, accepting ambiguous commit."""

    controller_id = base._clearml_id(getattr(controller, "id", None), "controller")
    existing = base._artifact_inventory(controller).get(AMENDMENT_ARTIFACT)
    if existing is not None:
        return _ensure_installed_amendment(task_class, controller_id, journal)
    active_ids = _active_task_ids_from_journal(journal)
    amendment = _build_migration_intent(
        controller_id=controller_id,
        journal=journal,
        active_task_ids=active_ids,
    )
    caught: Exception | None = None
    try:
        base._upload_artifact(controller, AMENDMENT_ARTIFACT, amendment)
    except Exception as error:  # A transport error may follow a committed upload.
        caught = error
    try:
        observed = _ensure_installed_amendment(task_class, controller_id, journal)
    except Exception:
        if caught is not None:
            raise P1ParallelV2Error(
                "parallel-v2 amendment upload failed before an auditable commit"
            ) from caught
        raise
    if observed != amendment:
        raise P1ParallelV2Error("parallel-v2 amendment round-trip drifted")
    return observed


def _build_execution_manifest(
    *, journal: Mapping[str, object], controller_id: str
) -> dict[str, object]:
    """Rebuild the base manifest in sealed plan order, never completion order."""

    plan, _pinset = _verify_frozen_bindings()
    controller_id = base._clearml_id(controller_id, "controller")
    validated = base._validate_journal(
        journal,
        plan=plan,
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=controller_id,
    )
    rows = {
        str(row["task_key"]): base._mapping(row, "manifest journal row")
        for row in validated["tasks"]
    }
    results: list[dict[str, object]] = []
    for pair_value in plan["pairs"]:
        pair = base._mapping(pair_value, "manifest pair")
        training_record = base._mapping(pair["training"], "manifest training")
        evaluation_record = base._mapping(pair["evaluation"], "manifest evaluation")
        training = rows[str(training_record["task_key"])]
        evaluation = rows[str(evaluation_record["task_key"])]
        if (
            training.get("state") != "completed"
            or training.get("server_status") != "completed"
            or evaluation.get("state") != "completed"
            or evaluation.get("server_status") != "completed"
        ):
            raise P1ParallelV2Error("cannot build a manifest before all tasks complete")
        model = base._mapping(training.get("result"), "manifest training result")
        evaluation_result = base._mapping(
            evaluation.get("result"), "manifest evaluation result"
        )
        results.append(
            {
                "subject": training_record["subject"],
                "seed_index": training_record["seed_index"],
                "training_seed": training_record["training_seed"],
                "training_task_id": base._clearml_id(
                    training.get("task_id"), "manifest training task"
                ),
                "model_id": model.get("model_id"),
                "checkpoint_sha256": model.get("checkpoint_sha256"),
                "training_runtime": model.get("runtime"),
                "evaluation": copy.deepcopy(evaluation_result),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "document_type": base.MANIFEST_DOCUMENT_TYPE,
        "execution_key": SEALED_EXECUTION_KEY,
        "plan_seal_sha256": SEALED_PLAN_SHA256,
        "controller_task_id": controller_id,
        "status": "completed",
        "runtime_contract": base._a100_runtime_contract(),
        "results": results,
    }
    manifest["seal_sha256"] = base._seal(manifest)
    try:
        return base._validate_execution_manifest(
            manifest,
            plan=plan,
            execution_key=SEALED_EXECUTION_KEY,
            controller_id=controller_id,
        )
    except base.P1ExecutorError as error:
        raise P1ParallelV2Error(str(error)) from error


def _journal_records(
    journal: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    rows = journal.get("tasks")
    if not isinstance(rows, list):
        raise P1ParallelV2Error("journal task inventory is unreadable")
    by_key = {
        str(row["task_key"]): base._mapping(row, "journal task row") for row in rows
    }
    plan, _pinset = _verify_frozen_bindings()
    records: dict[str, dict[str, object]] = {}
    paired_training: dict[str, dict[str, object]] = {}
    for pair_value in plan["pairs"]:
        pair = base._mapping(pair_value, "execution pair")
        training = base._mapping(pair["training"], "training record")
        evaluation = base._mapping(pair["evaluation"], "evaluation record")
        records[str(training["task_key"])] = training
        records[str(evaluation["task_key"])] = evaluation
        paired_training[str(evaluation["task_key"])] = training
    if set(by_key) != set(records):
        raise P1ParallelV2Error("journal and sealed task inventories differ")
    return by_key, paired_training


def _task_parameters(
    *, journal: Mapping[str, object], task_key: str
) -> dict[str, object]:
    by_key, paired_training = _journal_records(journal)
    row = by_key.get(task_key)
    if row is None:
        raise P1ParallelV2Error(f"unknown scoped task key {task_key}")
    subject = str(row["subject"])
    seed = int(row["training_seed"])
    if row["kind"] == "training":
        return base._training_parameters(subject, seed)
    training_record = paired_training[task_key]
    training = by_key[str(training_record["task_key"])]
    if (
        training.get("state") != "completed"
        or training.get("server_status") != "completed"
    ):
        raise P1ParallelV2Error(
            f"evaluation {task_key} cannot start before its training is validated"
        )
    model = base._mapping(training.get("result"), "paired training result")
    return base._evaluation_parameters(
        subject,
        seed,
        training_task_id=base._clearml_id(
            training.get("task_id"), "paired training task"
        ),
        model_id=base._clearml_id(model.get("model_id"), "paired model"),
        checkpoint_sha256=base._sha256(
            model.get("checkpoint_sha256"), "paired checkpoint"
        ),
    )


def _read_remote_journal(
    task_class: object, controller_id: str
) -> tuple[object, dict[str, object]]:
    plan, _pinset = _verify_frozen_bindings()
    controller = base._fresh(task_class, controller_id)
    artifact = base._artifact_inventory(controller).get(base.JOURNAL_ARTIFACT)
    if artifact is None:
        raise P1ParallelV2Error("controller execution journal is missing")
    journal = base._validate_journal(
        base._artifact_mapping(artifact, "P1 execution journal"),
        plan=plan,
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=controller_id,
    )
    return controller, journal


def _pending_journal_path() -> Path:
    return base._validate_cache_directory() / PENDING_JOURNAL_FILENAME


def _build_pending_journal_receipt(
    *,
    controller_id: str,
    prior: Mapping[str, object],
    candidate: Mapping[str, object],
    status: str,
    observed: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if status not in {"pending", "poisoned"}:
        raise P1ParallelV2Error("pending journal receipt status is invalid")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": PENDING_JOURNAL_DOCUMENT_TYPE,
        "status": status,
        "execution_key": SEALED_EXECUTION_KEY,
        "controller_task_id": base._clearml_id(controller_id, "controller"),
        "supervisor_sha256": _supervisor_sha256(),
        "control_host_contract": copy.deepcopy(CONTROL_HOST_CONTRACT),
        "prior_journal": copy.deepcopy(dict(prior)),
        "candidate_journal": copy.deepcopy(dict(candidate)),
        "observed_journal": (
            copy.deepcopy(dict(observed)) if observed is not None else None
        ),
    }
    receipt["seal_sha256"] = base._seal(receipt)
    return receipt


def _validate_pending_journal_receipt(
    value: Mapping[str, object], *, controller_id: str
) -> dict[str, object]:
    receipt = base._mapping(value, "pending journal receipt")
    base._exact_keys(
        receipt,
        {
            "candidate_journal",
            "control_host_contract",
            "controller_task_id",
            "document_type",
            "execution_key",
            "observed_journal",
            "prior_journal",
            "schema_version",
            "seal_sha256",
            "status",
            "supervisor_sha256",
        },
        "pending journal receipt",
    )
    if receipt.get("seal_sha256") != base._seal(receipt):
        raise P1ParallelV2Error("pending journal receipt seal mismatch")
    expected = {
        "schema_version": 1,
        "document_type": PENDING_JOURNAL_DOCUMENT_TYPE,
        "execution_key": SEALED_EXECUTION_KEY,
        "controller_task_id": base._clearml_id(controller_id, "controller"),
        "supervisor_sha256": _supervisor_sha256(),
        "control_host_contract": CONTROL_HOST_CONTRACT,
    }
    for field, item in expected.items():
        if receipt.get(field) != item:
            raise P1ParallelV2Error(f"pending journal receipt {field} drifted")
    if receipt.get("status") not in {"pending", "poisoned"}:
        raise P1ParallelV2Error("pending journal receipt status drifted")
    plan, _pinset = _verify_frozen_bindings()
    prior = base._validate_journal(
        base._mapping(receipt.get("prior_journal"), "pending prior journal"),
        plan=plan,
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=controller_id,
    )
    candidate = base._validate_journal(
        base._mapping(receipt.get("candidate_journal"), "pending candidate journal"),
        plan=plan,
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=controller_id,
    )
    if int(candidate["revision"]) != int(prior["revision"]) + 1:
        raise P1ParallelV2Error("pending journal revisions are not consecutive")
    observed_value = receipt.get("observed_journal")
    if observed_value is not None:
        observed = base._validate_journal(
            base._mapping(observed_value, "pending observed journal"),
            plan=plan,
            execution_key=SEALED_EXECUTION_KEY,
            controller_id=controller_id,
        )
        if receipt["status"] == "pending" and observed != prior:
            raise P1ParallelV2Error(
                "pending receipt observed journal is not the prior revision"
            )
    elif receipt["status"] == "poisoned":
        raise P1ParallelV2Error("poisoned journal receipt lacks observed evidence")
    return receipt


def _read_pending_journal_receipt(*, controller_id: str) -> dict[str, object] | None:
    path = _pending_journal_path()
    if not path.exists() and not path.is_symlink():
        return None
    return _validate_pending_journal_receipt(
        base._read_json_path(path, "pending journal receipt"),
        controller_id=controller_id,
    )


def _write_pending_journal_receipt(receipt: Mapping[str, object]) -> None:
    controller_id = base._clearml_id(
        receipt.get("controller_task_id"), "pending receipt controller"
    )
    value = _validate_pending_journal_receipt(receipt, controller_id=controller_id)
    path = _pending_journal_path()
    existing = _read_pending_journal_receipt(controller_id=controller_id)
    if existing is not None:
        if existing != value:
            raise P1ParallelV2Error(
                "a different durable pending journal receipt already exists"
            )
        return
    payload = (base._canonical_json(value) + "\n").encode("utf-8")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(temporary, flags, 0o600)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise P1ParallelV2Error("pending journal receipt write was incomplete")
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        descriptor = -1
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _remove_pending_journal_receipt(receipt: Mapping[str, object]) -> None:
    controller_id = base._clearml_id(
        receipt.get("controller_task_id"), "pending receipt controller"
    )
    path = _pending_journal_path()
    existing = _read_pending_journal_receipt(controller_id=controller_id)
    if existing != receipt:
        raise P1ParallelV2Error("pending journal receipt changed before removal")
    if path.is_symlink():
        raise P1ParallelV2Error("pending journal receipt became a symlink")
    path.unlink()
    directory = os.open(path.parent, os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _adopt_or_block_pending_journal_write(
    task_class: object,
    controller_id: str,
    journal: dict[str, object],
) -> None:
    receipt = _read_pending_journal_receipt(controller_id=controller_id)
    if receipt is None:
        return
    if receipt["status"] == "poisoned":
        raise P1ParallelV2Error("durable journal poison receipt requires manual audit")
    prior = base._mapping(receipt["prior_journal"], "pending prior journal")
    candidate = base._mapping(receipt["candidate_journal"], "pending candidate journal")
    _controller, observed = _read_remote_journal(task_class, controller_id)
    if observed == candidate:
        journal.clear()
        journal.update(copy.deepcopy(candidate))
        _remove_pending_journal_receipt(receipt)
        return
    if observed == prior:
        raise P1ParallelV2Error(
            "durable journal upload is still unresolved; refusing a different write"
        )
    raise P1ParallelV2Error(
        "remote journal diverged from a durable uncertain write; manual audit required"
    )


def _commit_journal_candidate(
    *,
    task_class: object,
    controller: object,
    journal: dict[str, object],
    candidate: dict[str, object],
) -> None:
    """Publish one revision without ever overwriting an uncertain revision."""

    controller_id = base._clearml_id(getattr(controller, "id", None), "controller")
    _adopt_or_block_pending_journal_write(task_class, controller_id, journal)
    fresh_controller, before = _read_remote_journal(task_class, controller_id)
    if before != journal:
        _write_pending_journal_receipt(
            _build_pending_journal_receipt(
                controller_id=controller_id,
                prior=journal,
                candidate=candidate,
                status="poisoned",
                observed=before,
            )
        )
        raise P1ParallelV2Error(
            "local journal is stale; refusing to overwrite the server journal"
        )
    if int(candidate.get("revision", -1)) != int(journal["revision"]) + 1:
        raise P1ParallelV2Error("journal candidate revision is not monotonic")
    plan, _pinset = _verify_frozen_bindings()
    base._validate_journal(
        candidate,
        plan=plan,
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=controller_id,
    )
    write_ahead = _build_pending_journal_receipt(
        controller_id=controller_id,
        prior=before,
        candidate=candidate,
        status="pending",
        observed=before,
    )
    # The local, fsync'd WAL must exist before the remote overwrite.  A crash
    # after this point can therefore never make the next process guess whether
    # this exact candidate was committed.
    _write_pending_journal_receipt(write_ahead)
    caught: Exception | None = None
    try:
        base._upload_artifact(fresh_controller, base.JOURNAL_ARTIFACT, candidate)
    except Exception as error:
        caught = error
    try:
        _latest_controller, observed = _read_remote_journal(task_class, controller_id)
    except Exception as read_error:
        if caught is not None:
            add_note = getattr(caught, "add_note", None)
            if callable(add_note):
                add_note(f"journal confirmation also failed: {read_error}")
            raise P1ParallelV2Error(
                "journal upload outcome is uncertain; no later revision may be written"
            ) from caught
        raise P1ParallelV2Error(
            "journal upload could not be confirmed; no later revision may be written"
        ) from read_error
    if observed == candidate:
        _remove_pending_journal_receipt(write_ahead)
        journal.clear()
        journal.update(copy.deepcopy(candidate))
        return
    if observed == before:
        raise P1ParallelV2Error(
            "journal upload outcome is unresolved; refusing same-revision overwrite"
        ) from caught
    raise P1ParallelV2Error(
        "journal changed unexpectedly; durable WAL requires manual audit"
    )


def _record_journal_task_safe(
    *,
    task_class: object,
    controller: object,
    journal: dict[str, object],
    task_key: str,
    task_id: str,
    state: str,
    server_status: str,
    result: Mapping[str, object] | None = None,
) -> None:
    controller_id = base._clearml_id(getattr(controller, "id", None), "controller")
    _adopt_or_block_pending_journal_write(task_class, controller_id, journal)
    candidate = copy.deepcopy(journal)
    row = base._journal_row(candidate, task_key)
    existing_id = row.get("task_id")
    if existing_id not in (None, task_id) and row.get("state") != "quarantined":
        raise P1ParallelV2Error(f"journal {task_key} task identity changed")
    prior_result = row.get("result")
    if existing_id not in (None, task_id) and row.get("state") == "quarantined":
        history: list[object] = []
        if isinstance(prior_result, Mapping):
            raw_history = prior_result.get("quarantined_task_ids", [])
            if isinstance(raw_history, list):
                history = list(raw_history)
        if existing_id not in history:
            history.append(existing_id)
        prior_result = {"quarantined_task_ids": history}
    row.update(
        {
            "task_id": base._clearml_id(task_id, f"journal {task_key}"),
            "state": state,
            "server_status": server_status,
            "result": dict(result) if result is not None else prior_result,
        }
    )
    candidate["status"] = "running"
    candidate["revision"] = int(journal["revision"]) + 1
    candidate["seal_sha256"] = base._seal(candidate)
    _commit_journal_candidate(
        task_class=task_class,
        controller=controller,
        journal=journal,
        candidate=candidate,
    )


def _persist_journal_status_safe(
    *,
    task_class: object,
    controller: object,
    journal: dict[str, object],
    status: str,
) -> None:
    controller_id = base._clearml_id(getattr(controller, "id", None), "controller")
    _adopt_or_block_pending_journal_write(task_class, controller_id, journal)
    candidate = copy.deepcopy(journal)
    candidate["status"] = status
    candidate["revision"] = int(journal["revision"]) + 1
    candidate["seal_sha256"] = base._seal(candidate)
    _commit_journal_candidate(
        task_class=task_class,
        controller=controller,
        journal=journal,
        candidate=candidate,
    )


def _reconcile_quarantined_journal_tasks_safe(
    *,
    task_class: object,
    controller: object,
    journal: dict[str, object],
    controller_id: str,
) -> None:
    rows = journal.get("tasks")
    if not isinstance(rows, list):
        raise P1ParallelV2Error(
            "journal task inventory is invalid during quarantine reconciliation"
        )
    for raw in list(rows):
        row = base._mapping(raw, "quarantine reconciliation row")
        if row.get("state") not in {"discovered", "prepared"}:
            continue
        task_key = str(row.get("task_key") or "")
        task_id = base._clearml_id(row.get("task_id"), f"quarantined {task_key} task")
        task = base._fresh(task_class, task_id)
        tags = base._task_tags(task)
        if base.ORPHAN_TAG not in tags:
            continue
        expected_execution_tag = base._execution_tag(SEALED_EXECUTION_KEY)
        expected_task_key_tag = f"p1-task-key:{task_key}"
        execution_tags = {
            item for item in tags if item.startswith(base.EXECUTION_KEY_TAG_PREFIX)
        }
        task_key_tags = {item for item in tags if item.startswith("p1-task-key:")}
        if (
            str(getattr(task, "name", "") or "")
            != base._task_name(SEALED_EXECUTION_KEY, task_key)
            or base._task_parent(task) != controller_id
            or base._task_project(task) != base.PROJECT_ID
            or not execution_tags.issubset({expected_execution_tag})
            or not task_key_tags.issubset({expected_task_key_tag})
            or base._task_queue(task)
            or base._task_status(task) not in {"created", "failed"}
        ):
            raise P1ParallelV2Error(f"quarantined task binding drifted for {task_key}")
        _record_journal_task_safe(
            task_class=task_class,
            controller=controller,
            journal=journal,
            task_key=task_key,
            task_id=task_id,
            state="quarantined",
            server_status=base._task_status(task),
            result={"quarantined_task_ids": [task_id]},
        )


def _record_discovery(
    *,
    task_class: object,
    controller: object,
    journal: dict[str, object],
    task_key: str,
    task_id: str,
    origin: str,
) -> None:
    row = base._journal_row(journal, task_key)
    if origin == "quarantined":
        _record_journal_task_safe(
            task_class=task_class,
            controller=controller,
            journal=journal,
            task_key=task_key,
            task_id=task_id,
            state="quarantined",
            server_status="failed",
            result={"quarantined_task_ids": [task_id]},
        )
        return
    if row.get("task_id") == task_id and row.get("state") != "absent":
        return
    _record_journal_task_safe(
        task_class=task_class,
        controller=controller,
        journal=journal,
        task_key=task_key,
        task_id=task_id,
        state="discovered",
        server_status="created",
    )


def _dispatch_candidate(
    *,
    task_class: object,
    controller: object,
    journal: dict[str, object],
    task_key: str,
    teacher: object,
    source_d: str,
    authorization_token: str,
    pre_enqueue_check: Callable[[], None] | None = None,
) -> object:
    """Create or resume one candidate, journal it first, then enqueue once."""

    if authorization_token != EXECUTION_TOKEN:
        raise P1ParallelV2Error("parallel-v2 remote execution token mismatch")
    controller_id = base._clearml_id(getattr(controller, "id", None), "controller")
    parameters = _task_parameters(journal=journal, task_key=task_key)

    def on_discovered(task_id: str, origin: str) -> None:
        _record_discovery(
            task_class=task_class,
            controller=controller,
            journal=journal,
            task_key=task_key,
            task_id=task_id,
            origin=origin,
        )

    try:
        task = base._create_task(
            task_class,
            teacher_task=teacher,
            controller_id=controller_id,
            task_key=task_key,
            source_d=source_d,
            parameters=parameters,
            execution_key=SEALED_EXECUTION_KEY,
            on_discovered=on_discovered,
        )
    except Exception as first_error:
        # A clone transport error can be ambiguous.  One immediate exact-name
        # recovery is safe because the base helper reuses or quarantines by the
        # sealed stable name and idempotency tags.
        try:
            matches = base._active_named_tasks(
                task_class, name=base._task_name(SEALED_EXECUTION_KEY, task_key)
            )
        except Exception:
            raise first_error
        if len(matches) != 1:
            raise first_error
        try:
            task = base._create_task(
                task_class,
                teacher_task=teacher,
                controller_id=controller_id,
                task_key=task_key,
                source_d=source_d,
                parameters=parameters,
                execution_key=SEALED_EXECUTION_KEY,
                on_discovered=on_discovered,
            )
        except Exception as recovery_error:
            add_note = getattr(recovery_error, "add_note", None)
            if callable(add_note):
                add_note(f"initial dispatch error: {first_error}")
            raise
    task_id = base._clearml_id(getattr(task, "id", None), "scoped child")
    base._require_server_unique_child(
        task_class,
        task=task,
        controller_id=controller_id,
        execution_key=SEALED_EXECUTION_KEY,
        task_key=task_key,
    )
    status = base._task_status(task)
    if status == "created":
        row = base._journal_row(journal, task_key)
        if row.get("state") != "prepared" or row.get("server_status") != "created":
            _record_journal_task_safe(
                task_class=task_class,
                controller=controller,
                journal=journal,
                task_key=task_key,
                task_id=task_id,
                state="prepared",
                server_status="created",
            )
        if pre_enqueue_check is not None:
            pre_enqueue_check()
        base._enqueue_once(task_class, task, authorization_token=base.EXECUTION_TOKEN)
        task = base._fresh(task_class, task_id)
        status = base._task_status(task)
    if status not in {"queued", "in_progress", "completed"}:
        raise P1ParallelV2Error(
            f"dispatched child {task_key} has invalid status {status!r}"
        )
    row = base._journal_row(journal, task_key)
    state = "completed" if status == "completed" else "active"
    if row.get("state") != state or row.get("server_status") != status:
        _record_journal_task_safe(
            task_class=task_class,
            controller=controller,
            journal=journal,
            task_key=task_key,
            task_id=task_id,
            state=state,
            server_status=status,
        )
    return task


def _reconcile_journal(
    *,
    task_class: object,
    controller: object,
    journal: dict[str, object],
    source_d: str,
    validated_completed_task_ids: set[str] | None = None,
    completion_guard: Callable[[], object] | None = None,
) -> dict[str, object]:
    """Converge journal rows to server truth without enqueueing any task."""

    validated = (
        validated_completed_task_ids
        if validated_completed_task_ids is not None
        else set()
    )
    controller_id = base._clearml_id(getattr(controller, "id", None), "controller")
    by_key, paired_training = _journal_records(journal)
    for task_key, row in by_key.items():
        task_id_value = row.get("task_id")
        if task_id_value is None or row.get("state") == "quarantined":
            continue
        task_id = base._clearml_id(task_id_value, f"{task_key} task")
        task = base._fresh(task_class, task_id)
        base._require_server_unique_child(
            task_class,
            task=task,
            controller_id=controller_id,
            execution_key=SEALED_EXECUTION_KEY,
            task_key=task_key,
        )
        if (
            base._task_parent(task) != controller_id
            or base._task_project(task) != base.PROJECT_ID
            or str(getattr(task, "name", "") or "")
            != base._task_name(SEALED_EXECUTION_KEY, task_key)
        ):
            raise P1ParallelV2Error(f"scoped child {task_key} identity drifted")
        status = base._task_status(task)
        if status in base.FAILURE_STATUSES:
            raise P1ParallelV2Error(
                f"scoped child {task_key} ended with status {status!r}"
            )
        if status == "created":
            if base._task_queue(task):
                raise P1ParallelV2Error(
                    f"created child {task_key} has an unexpected queue binding"
                )
            if row.get("state") in {"active", "completed"}:
                raise P1ParallelV2Error(f"scoped child {task_key} regressed to created")
            continue
        if status in {"queued", "in_progress"}:
            if base._task_queue(task) != base.QUEUE_ID:
                raise P1ParallelV2Error(
                    f"active child {task_key} queue binding drifted"
                )
            if row.get("state") == "completed":
                raise P1ParallelV2Error(
                    f"completed child {task_key} regressed to {status}"
                )
            if row.get("state") != "active" or row.get("server_status") != status:
                _record_journal_task_safe(
                    task_class=task_class,
                    controller=controller,
                    journal=journal,
                    task_key=task_key,
                    task_id=task_id,
                    state="active",
                    server_status=status,
                )
            continue
        if status != "completed":
            raise P1ParallelV2Error(
                f"scoped child {task_key} has unrecoverable status {status!r}"
            )
        if task_id in validated and row.get("state") == "completed":
            continue
        if completion_guard is not None:
            completion_guard()
        subject = str(row["subject"])
        seed = int(row["training_seed"])
        try:
            if row["kind"] == "training":
                result = base._validate_training_result(
                    task,
                    subject=subject,
                    seed=seed,
                    controller_id=controller_id,
                    source_d=source_d,
                )
            else:
                training_record = paired_training[task_key]
                training_row = base._journal_row(
                    journal, str(training_record["task_key"])
                )
                if training_row.get("state") != "completed":
                    raise P1ParallelV2Error(
                        f"evaluation {task_key} completed before paired training validation"
                    )
                model = base._mapping(
                    training_row.get("result"), "paired training result"
                )
                result = base._validate_evaluation_result(
                    task,
                    subject=subject,
                    seed=seed,
                    controller_id=controller_id,
                    training_task_id=base._clearml_id(
                        training_row.get("task_id"), "paired training task"
                    ),
                    model=model,
                    source_d=source_d,
                )
        except base.P1ExecutorError as error:
            if isinstance(error, P1ParallelV2Error):
                raise
            raise P1ParallelV2Error(str(error)) from error
        if completion_guard is not None:
            completion_guard()
        already_completed = (
            row.get("state") == "completed" and row.get("server_status") == "completed"
        )
        recorded_result = row.get("result")
        if already_completed and recorded_result is not None:
            if not isinstance(recorded_result, Mapping) or recorded_result != result:
                raise P1ParallelV2Error(
                    f"completed child {task_key} result evidence drifted"
                )
        if (
            row.get("state") != "completed"
            or row.get("server_status") != "completed"
            or row.get("result") != result
        ):
            _record_journal_task_safe(
                task_class=task_class,
                controller=controller,
                journal=journal,
                task_key=task_key,
                task_id=task_id,
                state="completed",
                server_status="completed",
                result=result,
            )
        validated.add(task_id)
    return journal


def _load_existing_execution(
    task_class: object,
    base_receipt: Mapping[str, object],
) -> tuple[object, dict[str, object]]:
    existing = base._mapping(
        base_receipt.get("existing_execution"), "base existing execution"
    )
    controller_id = existing.get("controller_id")
    if controller_id is None:
        raise P1ParallelV2Error(
            "parallel-v2 migration requires the existing sealed A100 controller"
        )
    controller_id = base._clearml_id(controller_id, "controller")
    controller = base._fresh(task_class, controller_id)
    plan, pinset = _verify_frozen_bindings()
    try:
        base._validate_controller(
            controller,
            plan=plan,
            pinset=pinset,
            execution_key=SEALED_EXECUTION_KEY,
        )
    except base.P1ExecutorError as error:
        raise P1ParallelV2Error(str(error)) from error
    artifacts = base._artifact_inventory(controller)
    allowed = {
        base.PLAN_ARTIFACT,
        base.PINSET_ARTIFACT,
        base.JOURNAL_ARTIFACT,
        base.MANIFEST_ARTIFACT,
        AMENDMENT_ARTIFACT,
    }
    if not set(artifacts).issubset(allowed):
        raise P1ParallelV2Error(
            "controller artifact inventory contains unknown entries"
        )
    for name, expected in (
        (base.PLAN_ARTIFACT, plan),
        (base.PINSET_ARTIFACT, pinset),
    ):
        artifact = artifacts.get(name)
        if artifact is None or base._artifact_mapping(artifact, name) != expected:
            raise P1ParallelV2Error(f"controller {name} drifted")
    if base.JOURNAL_ARTIFACT not in artifacts:
        raise P1ParallelV2Error("existing controller has no execution journal")
    journal = base._load_journal(
        controller,
        plan=plan,
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=controller_id,
    )
    return controller, journal


def _read_base_preflight(
    *,
    task_class: object | None = None,
    queue_reader: Callable[[object], Mapping[str, object]] | None = None,
    lease_reader: Callable[[str], Mapping[str, object]] | None = None,
    legacy_retirement_reader: Callable[
        [object, Mapping[str, object], Mapping[str, object], str],
        Mapping[str, object],
    ]
    | None = None,
) -> tuple[object, dict[str, object]]:
    task_type = task_class or base._load_clearml()
    try:
        receipt = base.preflight(
            task_class=task_type,
            queue_reader=queue_reader,
            lease_reader=lease_reader,
            legacy_retirement_reader=legacy_retirement_reader,
        )
    except base.P1ExecutorError as error:
        raise P1ParallelV2Error(str(error)) from error
    return task_type, receipt


def preflight(
    *,
    task_class: object | None = None,
    queue_reader: Callable[[object], Mapping[str, object]] | None = None,
    lease_reader: Callable[[str], Mapping[str, object]] | None = None,
    legacy_retirement_reader: Callable[
        [object, Mapping[str, object], Mapping[str, object], str],
        Mapping[str, object],
    ]
    | None = None,
) -> dict[str, object]:
    """Perform a remote read-only migration and scheduling preflight."""

    _verify_frozen_bindings()
    control_host = _validate_control_host()
    task_type, base_receipt = _read_base_preflight(
        task_class=task_class,
        queue_reader=queue_reader,
        lease_reader=lease_reader,
        legacy_retirement_reader=legacy_retirement_reader,
    )
    controller, journal = _load_existing_execution(task_type, base_receipt)
    controller_id = base._clearml_id(getattr(controller, "id", None), "controller")
    artifacts = base._artifact_inventory(controller)
    amendment_status = "not_installed"
    amendment_seal: str | None = None
    if AMENDMENT_ARTIFACT in artifacts:
        amendment = _ensure_installed_amendment(task_type, controller_id, journal)
        amendment_status = "installed"
        amendment_seal = str(amendment["seal_sha256"])
    queue_receipt = base._mapping(base_receipt.get("queue"), "base queue receipt")
    decision = _scheduler_decision(
        journal=journal,
        queue_receipt=queue_receipt,
        require_initial_second_gate=True,
    )
    pending = _read_pending_journal_receipt(controller_id=controller_id)
    if pending is not None:
        decision = dict(decision)
        decision["enqueue_task_keys"] = []
        decision["blocked_reason"] = "durable_pending_journal_write"
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": PREFLIGHT_DOCUMENT_TYPE,
        "status": "validated",
        "readonly": True,
        "remote_mutation_count": 0,
        "supervisor_sha256": _supervisor_sha256(),
        "base_executor_sha256": FROZEN_BASE_SHA256,
        "plan_seal_sha256": SEALED_PLAN_SHA256,
        "pinset_seal_sha256": SEALED_PINSET_SHA256,
        "execution_key": SEALED_EXECUTION_KEY,
        "controller_task_id": controller_id,
        "controller_status": base._task_status(controller),
        "control_host": control_host,
        "journal_revision": journal["revision"],
        "journal_seal_sha256": journal["seal_sha256"],
        "amendment_status": amendment_status,
        "amendment_seal_sha256": amendment_seal,
        "scheduler_decision": decision,
        "pending_journal_write": (
            {
                "status": pending["status"],
                "seal_sha256": pending["seal_sha256"],
                "candidate_revision": pending["candidate_journal"]["revision"],
            }
            if pending is not None
            else {"status": "absent", "seal_sha256": None, "candidate_revision": None}
        ),
        "base_preflight_seal_sha256": base_receipt["seal_sha256"],
        "global_mutex": copy.deepcopy(base_receipt["global_mutex"]),
        "queue": copy.deepcopy(queue_receipt),
    }
    receipt["seal_sha256"] = base._seal(receipt)
    return receipt


def _upload_manifest_once(
    task_class: object,
    controller: object,
    manifest: Mapping[str, object],
) -> None:
    controller_id = base._clearml_id(getattr(controller, "id", None), "controller")
    existing = base._artifact_inventory(controller).get(base.MANIFEST_ARTIFACT)
    if existing is not None:
        if base._artifact_mapping(existing, "P1 execution manifest") != manifest:
            raise P1ParallelV2Error("existing execution manifest drifted")
        return
    caught: Exception | None = None
    try:
        base._upload_artifact(controller, base.MANIFEST_ARTIFACT, manifest)
    except Exception as error:
        caught = error
    fresh = base._fresh(task_class, controller_id)
    artifact = base._artifact_inventory(fresh).get(base.MANIFEST_ARTIFACT)
    if artifact is None:
        if caught is not None:
            raise P1ParallelV2Error(
                "manifest upload failed before an auditable commit"
            ) from caught
        raise P1ParallelV2Error("execution manifest did not round-trip")
    if base._artifact_mapping(artifact, "P1 execution manifest") != manifest:
        raise P1ParallelV2Error("execution manifest round-trip drifted")


def _fresh_revalidate_completed(
    *,
    task_class: object,
    controller_id: str,
    source_d: str,
    completion_guard: Callable[[], object],
) -> tuple[object, dict[str, object], set[str]]:
    """Reload the journal and revalidate every completed child from server truth."""

    plan, _pinset = _verify_frozen_bindings()
    controller = base._fresh(task_class, controller_id)
    journal = base._load_journal(
        controller,
        plan=plan,
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=controller_id,
    )
    _adopt_or_block_pending_journal_write(task_class, controller_id, journal)
    _ensure_installed_amendment(task_class, controller_id, journal)
    completion_guard()
    validated: set[str] = set()
    _reconcile_journal(
        task_class=task_class,
        controller=controller,
        journal=journal,
        source_d=source_d,
        validated_completed_task_ids=validated,
        completion_guard=completion_guard,
    )
    return controller, journal, validated


def execute_plan(
    *,
    authorization_token: str,
    poll_seconds: float,
    timeout_hours: float,
    task_class: object | None = None,
    queue_reader: Callable[[object], Mapping[str, object]] | None = None,
    api_client: object | None = None,
    legacy_retirement_reader: Callable[
        [object, Mapping[str, object], Mapping[str, object], str],
        Mapping[str, object],
    ]
    | None = None,
) -> dict[str, object]:
    """Take over the existing campaign and run its two-slot training scheduler."""

    if authorization_token != EXECUTION_TOKEN:
        raise P1ParallelV2Error("remote execution token mismatch")
    if not math.isfinite(poll_seconds) or not 1.0 <= poll_seconds <= 60.0:
        raise P1ParallelV2Error("poll-seconds must be finite and within [1, 60]")
    if not math.isfinite(timeout_hours) or not 1.0 <= timeout_hours <= 72.0:
        raise P1ParallelV2Error("timeout-hours must be finite and within [1, 72]")
    plan, pinset = _verify_frozen_bindings()
    _validate_control_host()
    task_type = task_class or base._load_clearml()
    client = api_client or base._api_client()

    def read_lease(key: str) -> Mapping[str, object]:
        return base._read_lease_snapshot(key, api_client=client)

    def read_retirement(
        current_task_type: object,
        current_plan: Mapping[str, object],
        current_pinset: Mapping[str, object],
        current_key: str,
    ) -> Mapping[str, object]:
        if legacy_retirement_reader is not None:
            return legacy_retirement_reader(
                current_task_type, current_plan, current_pinset, current_key
            )
        return base._read_legacy_retirement_snapshot(
            current_task_type,
            plan=current_plan,
            pinset=current_pinset,
            execution_key=current_key,
            api_client=client,
        )

    def read_queue() -> dict[str, object]:
        try:
            return base._validate_a100_queue_receipt(
                (queue_reader or base._read_queue_snapshot)(task_type),
                context="parallel-v2 live queue receipt",
            )
        except base.P1ExecutorError as error:
            raise P1ParallelV2Error(str(error)) from error

    deadline = time.monotonic() + timeout_hours * 3600.0
    with base._execution_lock():
        _read_base_preflight(
            task_class=task_type,
            queue_reader=queue_reader,
            lease_reader=read_lease,
            legacy_retirement_reader=read_retirement,
        )
        with base._server_execution_mutex(SEALED_EXECUTION_KEY, api_client=client):
            _task_type, base_receipt = _read_base_preflight(
                task_class=task_type,
                queue_reader=queue_reader,
                lease_reader=read_lease,
                legacy_retirement_reader=read_retirement,
            )
            controller, journal = _load_existing_execution(task_type, base_receipt)
            controller_id = base._clearml_id(
                getattr(controller, "id", None), "controller"
            )
            _adopt_or_block_pending_journal_write(task_type, controller_id, journal)
            if base._task_status(controller) == "completed":
                artifact = base._artifact_inventory(controller).get(
                    base.MANIFEST_ARTIFACT
                )
                if artifact is None:
                    raise P1ParallelV2Error(
                        "completed controller has no execution manifest"
                    )
                return base._validate_execution_manifest(
                    base._artifact_mapping(artifact, "P1 execution manifest"),
                    plan=plan,
                    execution_key=SEALED_EXECUTION_KEY,
                    controller_id=controller_id,
                )

            # This is the first controller mutation after the handoff.  It is
            # also the barrier that makes the old serial executor fail closed.
            _install_amendment(task_type, controller, journal)
            controller = base._fresh(task_type, controller_id)
            source_d_task = task_type.get_task(task_id=base.SOURCE_D_TASK_ID)
            teacher = task_type.get_task(task_id=base.TEACHER_TASK_ID)
            source_d = base._validate_source_d(source_d_task)
            base._validate_teacher(teacher)
            _reconcile_quarantined_journal_tasks_safe(
                task_class=task_type,
                controller=controller,
                journal=journal,
                controller_id=controller_id,
            )
            validated_completed: set[str] = set()

            while True:
                if time.monotonic() >= deadline:
                    raise P1RecoverableTimeout(
                        "parallel-v2 timeout preserved all active and queued children"
                    )
                controller = base._fresh(task_type, controller_id)
                journal = base._load_journal(
                    controller,
                    plan=plan,
                    execution_key=SEALED_EXECUTION_KEY,
                    controller_id=controller_id,
                )
                _adopt_or_block_pending_journal_write(task_type, controller_id, journal)
                _ensure_installed_amendment(task_type, controller_id, journal)
                # Recheck the target and overlapping GPU8 queues before any
                # reconciliation can accept or persist a completed result.
                read_queue()
                _reconcile_journal(
                    task_class=task_type,
                    controller=controller,
                    journal=journal,
                    source_d=source_d,
                    validated_completed_task_ids=validated_completed,
                    completion_guard=read_queue,
                )
                by_key, _paired = _journal_records(journal)
                evaluation_keys = [
                    str(pair["evaluation"]["task_key"]) for pair in plan["pairs"]
                ]
                if all(
                    by_key[key].get("state") == "completed"
                    and by_key[key].get("server_status") == "completed"
                    for key in evaluation_keys
                ):
                    # The long-lived poll cache only avoids repeatedly reading
                    # immutable artifacts during normal scheduling.  Before
                    # sealing the manifest, independently revalidate all 12
                    # completed children and compare every result to the
                    # journal's recorded evidence.
                    controller, journal, final_validated = _fresh_revalidate_completed(
                        task_class=task_type,
                        controller_id=controller_id,
                        source_d=source_d,
                        completion_guard=read_queue,
                    )
                    final_rows, _final_pairs = _journal_records(journal)
                    expected_final_ids = {
                        base._clearml_id(row.get("task_id"), f"final {task_key} task")
                        for task_key, row in final_rows.items()
                        if row.get("state") == "completed"
                        and row.get("server_status") == "completed"
                    }
                    if (
                        len(expected_final_ids) != 12
                        or final_validated != expected_final_ids
                    ):
                        raise P1ParallelV2Error(
                            "final manifest revalidation did not cover all twelve tasks"
                        )
                    read_queue()
                    manifest = _build_execution_manifest(
                        journal=journal, controller_id=controller_id
                    )
                    _upload_manifest_once(task_type, controller, manifest)
                    if journal.get("status") != "completed":
                        _persist_journal_status_safe(
                            task_class=task_type,
                            controller=controller,
                            journal=journal,
                            status="completed",
                        )
                    base._complete_controller(task_type, controller)
                    return manifest

                queue_receipt = read_queue()
                decision = _scheduler_decision(
                    journal=journal,
                    queue_receipt=queue_receipt,
                    require_initial_second_gate=True,
                )
                candidates = list(decision["enqueue_task_keys"])
                if len(candidates) > 1:
                    raise P1ParallelV2Error(
                        "scheduler attempted more than one enqueue in one tick"
                    )
                if candidates:
                    candidate = str(candidates[0])
                    evaluation_candidate = decision.get("phase") == "evaluation"
                    if evaluation_candidate:
                        controller, journal, _validated_for_evaluation = (
                            _fresh_revalidate_completed(
                                task_class=task_type,
                                controller_id=controller_id,
                                source_d=source_d,
                                completion_guard=read_queue,
                            )
                        )
                        refreshed_decision = _scheduler_decision(
                            journal=journal,
                            queue_receipt=read_queue(),
                            require_initial_second_gate=True,
                        )
                        if refreshed_decision.get("enqueue_task_keys") != [candidate]:
                            continue

                    def recheck() -> None:
                        current_journal = journal
                        if evaluation_candidate:
                            _fresh_controller, current_journal, _validated = (
                                _fresh_revalidate_completed(
                                    task_class=task_type,
                                    controller_id=controller_id,
                                    source_d=source_d,
                                    completion_guard=read_queue,
                                )
                            )
                        refreshed = _scheduler_decision(
                            journal=current_journal,
                            queue_receipt=read_queue(),
                            require_initial_second_gate=True,
                        )
                        if refreshed.get("enqueue_task_keys") != [candidate]:
                            raise P1ParallelV2Error(
                                "capacity changed between task preparation and enqueue"
                            )

                    _dispatch_candidate(
                        task_class=task_type,
                        controller=controller,
                        journal=journal,
                        task_key=candidate,
                        teacher=teacher,
                        source_d=source_d,
                        authorization_token=authorization_token,
                        pre_enqueue_check=recheck,
                    )
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    continue
                time.sleep(min(poll_seconds, remaining))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="resume the sealed campaign with two training slots",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="connect read-only and print a sealed migration receipt",
    )
    parser.add_argument(
        "--enqueue-token",
        default="",
        help="exact acknowledgement token required with --execute",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=72.0)
    parser.add_argument("--pretty", action="store_true")
    return parser


def _print_json(value: Mapping[str, object], *, pretty: bool) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight:
        if args.enqueue_token:
            raise P1ParallelV2Error("--enqueue-token is invalid with --preflight")
        _print_json(preflight(), pretty=args.pretty)
        return 0
    if not args.execute:
        if args.enqueue_token:
            raise P1ParallelV2Error("--enqueue-token is invalid without --execute")
        plan, pinset = _verify_frozen_bindings()
        dry_run: dict[str, object] = {
            "schema_version": 1,
            "document_type": "resilient_v2x_p1_a100_parallel_v2_dry_run",
            "default_mode": "dry_run",
            "remote_mutation_authorized": False,
            "supervisor_sha256": _supervisor_sha256(),
            "base_executor_sha256": FROZEN_BASE_SHA256,
            "plan_seal_sha256": plan["seal_sha256"],
            "pinset_seal_sha256": pinset["seal_sha256"],
            "execution_key": SEALED_EXECUTION_KEY,
            "control_host_contract": copy.deepcopy(CONTROL_HOST_CONTRACT),
            "max_active_or_queued_training_tasks": MAX_ACTIVE_TRAINING_TASKS,
            "max_active_or_queued_evaluation_tasks": MAX_ACTIVE_EVALUATION_TASKS,
            "execution_token_sha256": hashlib.sha256(
                EXECUTION_TOKEN.encode("utf-8")
            ).hexdigest(),
        }
        dry_run["seal_sha256"] = base._seal(dry_run)
        _print_json(dry_run, pretty=args.pretty)
        return 0
    if args.enqueue_token != EXECUTION_TOKEN:
        raise P1ParallelV2Error(
            "--execute requires the exact --enqueue-token acknowledgement"
        )
    manifest = execute_plan(
        authorization_token=args.enqueue_token,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    _print_json(manifest, pretty=args.pretty)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except P1RecoverableTimeout as error:
        print(f"P1-PARALLEL-V2-RECOVERABLE-TIMEOUT: {error}", file=sys.stderr)
        raise SystemExit(3) from None
    except P1ExecutorError as error:
        print(f"P1-PARALLEL-V2-ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
