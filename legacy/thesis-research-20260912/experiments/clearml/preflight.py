#!/usr/bin/env python3
"""Fail-closed ClearML queue preflight with a topology-free public summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

ACTIVE_STATUSES = ["created", "queued", "in_progress"]


class PreflightError(RuntimeError):
    """A safe preflight failure that never contains internal object details."""


def value(record: object, name: str, default: object = None) -> object:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def worker_is_busy(worker: object) -> bool:
    """Return whether a worker reports a task without serializing that task."""
    task = value(worker, "task", None)
    if task is None:
        return False
    if isinstance(task, str):
        return bool(task.strip())
    task_id = value(task, "id", None)
    if task_id is None:
        # Unknown task shapes are treated as busy so preflight cannot overstate
        # available capacity.
        return True
    return bool(str(task_id).strip())


def inspect_preflight(
    *,
    client: object,
    task_api: object,
    queue_name: str,
    project_name: str,
    task_name: str,
    require_empty: bool,
) -> dict[str, object]:
    """Inspect queue state and return only non-sensitive aggregate fields."""
    queues_api = value(client, "queues")
    queues = list(queues_api.get_all(name=queue_name) or [])
    if len(queues) != 1:
        raise PreflightError(
            f"expected exactly one queue named {queue_name}, found {len(queues)}"
        )
    queue = queues[0]
    queue_id = str(value(queue, "id", ""))
    if not queue_id:
        raise PreflightError("matched queue has no stable identifier")
    entries = list(value(queue, "entries", []) or [])
    if require_empty and entries:
        raise PreflightError(f"queue {queue_name} is not empty: {len(entries)} entries")

    duplicates = task_api.get_tasks(
        project_name=project_name,
        task_name=task_name,
        task_filter={"status": ACTIVE_STATUSES},
    )
    if duplicates:
        raise PreflightError(
            f"active task with the same name already exists: {len(duplicates)}"
        )

    workers_api = value(client, "workers")
    workers = list(workers_api.get_all() or [])
    matching_workers = []
    for worker in workers:
        worker_queues = value(worker, "queues", []) or []
        queue_ids = {str(value(item, "id", item)) for item in worker_queues}
        if queue_id in queue_ids:
            matching_workers.append(worker)
    if not matching_workers:
        raise PreflightError(f"no online worker reports queue {queue_name}")

    busy_count = sum(worker_is_busy(worker) for worker in matching_workers)
    worker_count = len(matching_workers)
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "queue": {"name": queue_name, "entry_count": len(entries)},
        "workers": {
            "reporting": worker_count,
            "idle": worker_count - busy_count,
            "busy": busy_count,
        },
        "active_same_name_task_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--require-empty", action="store_true")
    args = parser.parse_args()

    # Import lazily so local unit tests and document builds do not require the
    # ClearML SDK or read a user's ClearML configuration.
    from clearml import Task
    from clearml.backend_api.session.client import APIClient

    try:
        report = inspect_preflight(
            client=APIClient(),
            task_api=Task,
            queue_name=args.queue,
            project_name=args.project,
            task_name=args.task_name,
            require_empty=args.require_empty,
        )
    except PreflightError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
