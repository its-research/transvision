#!/usr/bin/env python3
"""Publish one verified synthetic-canary ClearML task with a fail-closed gate.

Only a fixed, non-sensitive success receipt or a fixed error code is emitted.
ClearML calls run with process-level stdout and stderr suppression so native or
subprocess-backed SDK noise cannot disclose service details.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


CLEARML_PROJECT = "Thesis/RTP-V2X"
CLEARML_TASK_PREFIX = "rtpv2x__synth-causal-canary-v1__"
EXPECTED_ARTIFACTS = frozenset({"events", "metrics", "run_manifest"})
TASK_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_ARTIFACT_SIZE = (1 << 63) - 1


class PublishGateError(RuntimeError):
    """A fail-closed error carrying only a fixed, non-sensitive code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise PublishGateError(code)


def validate_task_id(value: object) -> str:
    if not isinstance(value, str) or TASK_ID_RE.fullmatch(value) is None:
        _fail("INVALID_TASK_ID")
    return value


def _status_value(value: object) -> str:
    normalized = getattr(value, "value", value)
    if not isinstance(normalized, str):
        _fail("INVALID_STATUS")
    return normalized.lower()


def _artifact_size(value: object) -> int | None:
    # ClearML may omit content_size/size for otherwise valid artifacts.  None is
    # retained in the signed snapshot and must remain unchanged after publish.
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("INVALID_ARTIFACT_SIZE")
    if value <= 0 or value > MAX_ARTIFACT_SIZE:
        _fail("INVALID_ARTIFACT_SIZE")
    return value


def _artifact_hash(value: object) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail("INVALID_ARTIFACT_HASH")
    return value


def _artifact_snapshot(task: object) -> tuple[dict[str, object], ...]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or frozenset(artifacts) != EXPECTED_ARTIFACTS:
        _fail("INVALID_ARTIFACT_SET")

    rows: list[dict[str, object]] = []
    for name in sorted(EXPECTED_ARTIFACTS):
        artifact = artifacts[name]
        rows.append(
            {
                "name": name,
                "size": _artifact_size(getattr(artifact, "size", None)),
                "sha256": _artifact_hash(getattr(artifact, "hash", None)),
            }
        )
    return tuple(rows)


def inspect_task(
    task: object, expected_task_id: str, expected_status: str
) -> tuple[dict[str, object], ...]:
    if getattr(task, "id", None) != expected_task_id:
        _fail("TASK_ID_MISMATCH")

    project = task.get_project_name()  # type: ignore[attr-defined]
    if project != CLEARML_PROJECT:
        _fail("PROJECT_MISMATCH")

    name = getattr(task, "name", None)
    if not isinstance(name, str) or not name.startswith(CLEARML_TASK_PREFIX):
        _fail("TASK_NAME_MISMATCH")

    if _status_value(getattr(task, "status", None)) != expected_status:
        _fail("TASK_STATUS_MISMATCH")
    return _artifact_snapshot(task)


def artifact_snapshot_sha256(snapshot: tuple[dict[str, object], ...]) -> str:
    encoded = json.dumps(
        snapshot,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@contextlib.contextmanager
def _suppress_process_output() -> Any:
    """Suppress Python and native writes to stdout/stderr for a short SDK call."""

    saved_stdout: int | None = None
    saved_stderr: int | None = None
    devnull: int | None = None
    previous_disable = logging.root.manager.disable
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(previous_disable)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            if saved_stdout is not None:
                os.dup2(saved_stdout, 1)
            if saved_stderr is not None:
                os.dup2(saved_stderr, 2)
            for descriptor in (devnull, saved_stdout, saved_stderr):
                if descriptor is not None:
                    os.close(descriptor)


def _quiet_call(code: str, function: Callable[..., Any], *args: object) -> Any:
    try:
        with _suppress_process_output():
            return function(*args)
    except PublishGateError:
        raise
    except Exception:
        _fail(code)


def load_task(task_id: str) -> object:
    def _load() -> object:
        from clearml import Task  # type: ignore

        return Task.get_task(task_id=task_id)

    return _quiet_call("TASK_LOOKUP_FAILED", _load)


def publish_task(
    task_id: str, *, loader: Callable[[str], object] = load_task
) -> tuple[str, str]:
    requested_id = validate_task_id(task_id)
    task = _quiet_call("TASK_LOOKUP_FAILED", loader, requested_id)
    before = _quiet_call(
        "TASK_INSPECTION_FAILED", inspect_task, task, requested_id, "completed"
    )

    _quiet_call("TASK_PUBLISH_FAILED", task.publish)  # type: ignore[attr-defined]

    refreshed = _quiet_call("TASK_REFRESH_FAILED", loader, requested_id)
    after = _quiet_call(
        "TASK_INSPECTION_FAILED", inspect_task, refreshed, requested_id, "published"
    )
    if after != before:
        _fail("ARTIFACT_SNAPSHOT_CHANGED")
    return "published", artifact_snapshot_sha256(after)


def parse_task_id(argv: Sequence[str]) -> str:
    if len(argv) != 1:
        _fail("INVALID_ARGUMENTS")
    return validate_task_id(argv[0])


def main(argv: Sequence[str] | None = None) -> int:
    try:
        task_id = parse_task_id(sys.argv[1:] if argv is None else argv)
        status, snapshot_sha = publish_task(task_id)
    except PublishGateError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        print("UNEXPECTED_FAILURE", file=sys.stderr)
        return 2

    print(f"task_id={task_id}")
    print(f"status={status}")
    print(f"artifact_snapshot_sha256={snapshot_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
