#!/usr/bin/env python3
"""Read a strict, URL-free whitelist of ClearML task evidence fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = re.compile(r"^[0-9a-f]{32}$")
SERVER_HASH = re.compile(r"^[0-9a-f]{64}$")
URL_LIKE = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|\bwww\.|"
    r"\b(?:10|127)\.(?:\d{1,3}\.){2}\d{1,3}\b|"
    r"\b192\.168\.(?:\d{1,3}\.)\d{1,3}\b|"
    r"\b172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3}\b)"
)
SAFE_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_STATUSES = frozenset(
    {
        "closed",
        "completed",
        "created",
        "failed",
        "in_progress",
        "published",
        "publishing",
        "queued",
        "stopped",
        "unknown",
    }
)
SAFE_LABEL_PUNCTUATION = frozenset(" _-./()+[]")
TASK_FIELDS = frozenset(
    {"artifacts", "id", "last_iteration", "name", "project", "source", "status"}
)
SOURCE_FIELDS = frozenset({"diff_sha256", "diff_size"})
ARTIFACT_FIELDS = frozenset({"name", "server_hash", "size"})


class InspectionError(RuntimeError):
    """Raised when a task cannot be represented by the safe output schema."""


def _member(record: object, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _validate_task_id(value: object) -> str:
    if not isinstance(value, str) or TASK_ID.fullmatch(value) is None:
        raise InspectionError("task id failed validation")
    return value


def _validate_label(value: object, field: str, *, maximum: int = 255) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise InspectionError(f"{field} failed validation")
    if value != value.strip() or URL_LIKE.search(value):
        raise InspectionError(f"{field} failed validation")
    if any(
        not (character.isalnum() or character in SAFE_LABEL_PUNCTUATION)
        for character in value
    ):
        raise InspectionError(f"{field} failed validation")
    return value


def _validate_status(value: object) -> str:
    enum_value = getattr(value, "value", value)
    if not isinstance(enum_value, str):
        raise InspectionError("task status failed validation")
    normalized = enum_value.lower()
    if normalized not in SAFE_STATUSES:
        raise InspectionError("task status failed validation")
    return normalized


def _validate_nonnegative_integer_or_none(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InspectionError(f"{field} failed validation")
    return value


def _validate_server_hash(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InspectionError("artifact server hash failed validation")
    normalized = value.lower()
    if SERVER_HASH.fullmatch(normalized) is None:
        raise InspectionError("artifact server hash failed validation")
    return normalized


def _validate_artifact_name(value: object) -> str:
    if not isinstance(value, str) or SAFE_ARTIFACT_NAME.fullmatch(value) is None:
        raise InspectionError("artifact name failed validation")
    return value


def _artifact_summary(name: object, artifact: object) -> dict[str, Any]:
    return {
        "name": _validate_artifact_name(name),
        "server_hash": _validate_server_hash(_member(artifact, "hash")),
        "size": _validate_nonnegative_integer_or_none(
            _member(artifact, "content_size"), "artifact size"
        ),
    }


def _load_task(task_id: str) -> object:
    try:
        from clearml import Task

        return Task.get_task(task_id=task_id)
    except Exception as exc:
        raise InspectionError("ClearML task lookup failed") from exc


def _task_summary(task_id: str) -> dict[str, Any]:
    requested_id = _validate_task_id(task_id)
    task = _load_task(requested_id)
    try:
        observed_id = _validate_task_id(getattr(task, "id", None))
        if observed_id != requested_id:
            raise InspectionError("returned task id does not match requested task id")
        script = _member(getattr(task, "data", None), "script")
        diff = _member(script, "diff")
        if diff is None:
            diff = ""
        if not isinstance(diff, str):
            raise InspectionError("source diff failed validation")
        diff_bytes = diff.encode("utf-8")

        artifacts = getattr(task, "artifacts", None)
        if not isinstance(artifacts, Mapping):
            raise InspectionError("task artifacts failed validation")
        artifact_records = [
            _artifact_summary(name, artifact) for name, artifact in artifacts.items()
        ]
        artifact_records.sort(key=lambda row: row["name"])

        payload = {
            "artifacts": artifact_records,
            "id": observed_id,
            "last_iteration": _validate_nonnegative_integer_or_none(
                task.get_last_iteration(), "last iteration"
            ),
            "name": _validate_label(getattr(task, "name", None), "task name"),
            "project": _validate_label(task.get_project_name(), "project name"),
            "source": {
                "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
                "diff_size": len(diff_bytes),
            },
            "status": _validate_status(getattr(task, "status", None)),
        }
    except InspectionError:
        raise
    except Exception as exc:
        raise InspectionError("ClearML task metadata inspection failed") from exc
    _assert_safe_schema(payload)
    return payload


def _assert_safe_schema(payload: Mapping[str, Any]) -> None:
    if frozenset(payload) != TASK_FIELDS:
        raise InspectionError("task output schema failed validation")
    source = payload.get("source")
    artifacts = payload.get("artifacts")
    if not isinstance(source, Mapping) or frozenset(source) != SOURCE_FIELDS:
        raise InspectionError("source output schema failed validation")
    if not isinstance(artifacts, list):
        raise InspectionError("artifact output schema failed validation")
    if any(
        not isinstance(row, Mapping) or frozenset(row) != ARTIFACT_FIELDS
        for row in artifacts
    ):
        raise InspectionError("artifact output schema failed validation")


def _atomic_write(output: Path, rendered: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a strict safe-field summary for one or more ClearML tasks."
    )
    parser.add_argument("task_id", nargs="+")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records = [_task_summary(task_id) for task_id in args.task_id]
        payload: Any = records[0] if len(records) == 1 else records
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            _atomic_write(args.output, rendered)
    except InspectionError as exc:
        print(f"inspection failed: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("inspection failed: output write failed", file=sys.stderr)
        return 2
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
