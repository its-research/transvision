#!/usr/bin/env python3
"""Freeze failed evaluation attempts and explicitly restore their exact task IDs.

The default mode is read-only and writes one local sealed plan receipt. Remote
mutation requires ``--execute`` plus the exact token and two distinct write-once
receipt paths: the attempt evidence is durably written before the first reset,
and the recovery receipt records every authoritative post-reset readback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from tools.resilient_v2x import clearml_1337_dependency_watcher as formal
    from tools.resilient_v2x import (
        clearml_formal_candidate_evaluation_queue as candidates,
    )
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import clearml_1337_dependency_watcher as formal
    import clearml_formal_candidate_evaluation_queue as candidates


EXECUTE_TOKEN = "RESET_EXACT_W3_AND_CANDIDATE_EVALUATION_ATTEMPTS"
PROJECT_ID = "6e43f972e5ea4cee901a7c8855fce8cd"
TRAINING_CONTROLLER_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
PROVENANCE_TASK_ID = "7e244a711751469b8cfdb25d77b05269"
PROVENANCE_SCRIPT_SHA256 = (
    "0bcc0b9ce383e5f2cfa35e87959caff9e388a988423e74797c7127a560eb007a"
)
EVALUATION_TEMPLATE_TASK_ID = "8b77a3674dfe405388aae39ef82d06ef"
CANDIDATE_EVALUATION_IDS = {
    "E1": "c6f26cc7902142c090ec238856409ac6",
    "E3": "27a82d39e6354697ad1532b6399bbde2",
}
RECOVERY_RECEIPT_TYPE = "resilient_v2x_evaluation_attempt_recovery"
EXPECTED_RECOVERY_TASKS = (
    ("formal_w3", "support_residual", "718ba3d3421249eb9f31d5216a8f160a"),
    ("formal_w3", "ptf_none", "e6ad9de8555e49acb6b7a7f869642c2c"),
    ("formal_w3", "ptf_linear", "d49754b3b7f84faeb39f0b454572375b"),
    ("formal_w3", "router_static", "1274721408a14a2e8c4fd3b15e7c71ad"),
    ("formal_w3", "no_distillation", "59d64499091040dfaf27097a22dd380a"),
    ("formal_w3", "coformernet", "d8fac86325e047e5aa25f5ce899902b6"),
    ("formal_w3", "router_uniform", "2819bf52c1d7433492544faf7870d163"),
    ("formal_w3", "no_reliability", "194271f304ba4d1ca5fa1cab3851b47f"),
    ("formal_w3", "no_delay_metadata", "ff90b801353241b285d89376dd0fb80b"),
    (
        "formal_w3",
        "concat_capacity_matched",
        "6c0e48337b2d4ebcaa7f877a8dc26b48",
    ),
    ("formal_w3", "ffnet", "144397bfa9c242bc9a92a1279922558b"),
    ("formal_w3", "bevfusion", "c3b87760b78742cb8e7de3a506a08999"),
    ("formal_w3", "v2x_vit", "cb2675d7e0e845268420f5c0d5248ece"),
    ("formal_w3", "cobevt", "23f4d7f082284aa2be098a90f0a596e8"),
    (
        "formal_w3",
        "linear_no_distillation",
        "f245587e92d54ed89e8eeb386ff63dcd",
    ),
    (
        "formal_w3",
        "no_distillation_peak_lr_3e4",
        "164d5157c04f4ba59a91fda19a2823e4",
    ),
    ("formal_w3", "ego_only", "cadbc85789c245ac9235e0c16f9de845"),
    ("formal_w3", "fcooper", "bfd69dc3877844ef890ab6f397786797"),
    ("formal_w3", "attfuse", "9a46b48da1444648b979a4a627ad9e8f"),
    ("formal_w3", "v2vnet", "dd32373cb1c94ece988a20a72d1eba95"),
    ("formal_w3", "when2com", "5420ac0dfb54476ba5ef0c8610de5cba"),
    ("formal_w3", "where2comm", "99fc7ea6f2884072a033206b1d123241"),
    ("formal_w3", "late_fusion", "242ac1668e37463db20b28530df9b8ed"),
    ("formal_w3", "disconet", "12a40d1188584fb0b0525a7a579416cc"),
    ("formal_w3", "how2comm", "f76cc1f899c140679353c7d31fce0eba"),
    ("formal_w3", "resilient_v2x", "7deb18e532324850bee1fb4279a838b7"),
    (
        "candidate",
        "support_residual_linear",
        "c6f26cc7902142c090ec238856409ac6",
    ),
    (
        "candidate",
        "support_residual_no_reliability",
        "27a82d39e6354697ad1532b6399bbde2",
    ),
)
ALLOWED_PRE_STATUSES = frozenset(
    {"created", "queued", "in_progress", "failed", "stopped"}
)
ATTEMPT_EVIDENCE_STATUSES = frozenset({"failed", "stopped"})
LEGACY_INERT_EVALUATION_PARAMETERS = {
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
LEGACY_FILES_SERVER_HOST = "10.100.34.118"
EXPECTED_FILES_SERVER_HOST = "10.100.35.118"
EXPECTED_FILES_SERVER_PORT = 8081


class RecoveryError(RuntimeError):
    """Raised when exact recovery cannot be proven."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        raise RecoveryError("authenticated artifact download refused a redirect")


class AuthenticatedArtifactDownloader:
    """Download each sealed artifact once using ephemeral ClearML auth headers."""

    def __init__(
        self,
        *,
        files_server_url: str,
        add_auth_headers: Callable[[dict[str, str]], object],
        opener: object | None = None,
    ) -> None:
        parsed = urlsplit(files_server_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname != EXPECTED_FILES_SERVER_HOST
            or parsed.port != EXPECTED_FILES_SERVER_PORT
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise RecoveryError("ClearML fileserver authority drifted")
        self._authority = parsed
        self._add_auth_headers = add_auth_headers
        self._opener = opener or build_opener(_RejectRedirects())
        self._cache: dict[tuple[str, str, str], bytes] = {}
        self.audit_records: list[dict[str, object]] = []

    def _target(self, task: object, name: str) -> tuple[str, int, str]:
        artifact, record, _ = formal._unique_artifact_metadata(task, name)
        urls = {
            str(value)
            for value in (
                formal._metadata_value(artifact, "uri", "url", "artifact_uri"),
                formal._metadata_value(record, "uri", "url", "artifact_uri"),
            )
            if value not in {None, ""}
        }
        if not urls:
            raise RecoveryError(f"artifact {name} lacks a fileserver URL")
        paths: set[str] = set()
        for value in urls:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname
                not in {LEGACY_FILES_SERVER_HOST, EXPECTED_FILES_SERVER_HOST}
                or parsed.port != EXPECTED_FILES_SERVER_PORT
                or parsed.username is not None
                or parsed.password is not None
                or not parsed.path.startswith("/")
                or parsed.query
                or parsed.fragment
            ):
                raise RecoveryError(f"artifact {name} URL authority drifted")
            paths.add(parsed.path)
        if len(paths) != 1:
            raise RecoveryError(f"artifact {name} URL path metadata drifted")
        sizes = {
            int(value)
            for value in (
                formal._metadata_value(
                    artifact, "content_size", "size", "size_bytes"
                ),
                formal._metadata_value(
                    record, "content_size", "size", "size_bytes"
                ),
            )
            if value not in {None, ""}
        }
        hashes = {
            str(value)
            for value in (
                formal._metadata_value(artifact, "hash", "sha256"),
                formal._metadata_value(record, "hash", "sha256"),
            )
            if value not in {None, ""}
        }
        if len(sizes) != 1 or next(iter(sizes)) <= 0:
            raise RecoveryError(f"artifact {name} size metadata drifted")
        if len(hashes) != 1 or formal.SHA256_PATTERN.fullmatch(
            next(iter(hashes))
        ) is None:
            raise RecoveryError(f"artifact {name} SHA-256 metadata drifted")
        target = urlunsplit(
            (
                self._authority.scheme,
                self._authority.netloc,
                next(iter(paths)),
                "",
                "",
            )
        )
        return target, next(iter(sizes)), next(iter(hashes))

    def __call__(self, task: object, name: str) -> bytes:
        task_id = formal._task_id(
            getattr(task, "id", ""), f"artifact {name} owner"
        )
        metadata_signature = formal._artifact_metadata_signature(task, name)
        key = (task_id, name, metadata_signature)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        target, expected_size, expected_sha256 = self._target(task, name)
        headers: dict[str, str] = {}
        self._add_auth_headers(headers)
        if not headers or any(not isinstance(value, str) for value in headers.values()):
            raise RecoveryError("ClearML authentication headers are unavailable")
        request = Request(target, headers=headers, method="GET")
        response = self._opener.open(request, timeout=30.0)
        try:
            status = int(getattr(response, "status", response.getcode()))
            final_url = str(response.geturl())
            if status != 200 or final_url != target:
                raise RecoveryError(
                    f"artifact {name} authenticated response identity drifted"
                )
            raw = response.read(expected_size + 1)
            if response.read(1):
                raise RecoveryError(f"artifact {name} exceeded its sealed size")
        finally:
            response.close()
        if len(raw) != expected_size:
            raise RecoveryError(f"artifact {name} authenticated size mismatch")
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        if observed_sha256 != expected_sha256:
            raise RecoveryError(f"artifact {name} authenticated SHA-256 mismatch")
        self._cache[key] = raw
        self.audit_records.append(
            {
                "task_id": task_id,
                "artifact_name": name,
                "authenticated_url": target,
                "content_size": expected_size,
                "sha256": expected_sha256,
                "redirects_allowed": False,
            }
        )
        return raw


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


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def validate_recovery_receipt(path: Path) -> dict[str, object]:
    """Validate and summarize the exact 26+2 recovery before deployment."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RecoveryError("evaluation recovery receipt cannot be resolved") from error
    if path.is_symlink() or not resolved.is_file():
        raise RecoveryError("evaluation recovery receipt must be a regular file")
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise RecoveryError("evaluation recovery receipt cannot be read") from error
    if not raw or len(raw) > 4_194_304:
        raise RecoveryError("evaluation recovery receipt size is invalid")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryError("evaluation recovery receipt is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise RecoveryError("evaluation recovery receipt is not a JSON object")
    seal = value.get("seal_sha256")
    if (
        type(seal) is not str
        or formal.SHA256_PATTERN.fullmatch(seal) is None
        or _seal(value)["seal_sha256"] != seal
    ):
        raise RecoveryError("evaluation recovery receipt seal mismatch")
    exact = {
        "schema_version": 1,
        "receipt_type": RECOVERY_RECEIPT_TYPE,
        "status": "recovered",
        "remote_state_changed": True,
        "authoritative_plan_producer_task_id": (
            formal.AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
        ),
        "authoritative_plan_seal_sha256": (
            formal.AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256
        ),
        "formal_task_count": 26,
        "candidate_task_ids": dict(CANDIDATE_EVALUATION_IDS),
        "all_tasks_created_unqueued": True,
        "replacement_tasks_created": False,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise RecoveryError(f"evaluation recovery receipt {key} drifted")
    attempt_seal = value.get("attempt_receipt_seal_sha256")
    if (
        type(attempt_seal) is not str
        or formal.SHA256_PATTERN.fullmatch(attempt_seal) is None
    ):
        raise RecoveryError("evaluation recovery attempt receipt seal is invalid")
    rows = value.get("recovered_tasks")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_RECOVERY_TASKS):
        raise RecoveryError("evaluation recovery task inventory is incomplete")
    observed: list[tuple[str, str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RecoveryError(f"evaluation recovery task {index} is invalid")
        identity = (
            str(row.get("scope") or ""),
            str(row.get("subject") or ""),
            str(row.get("task_id") or ""),
        )
        observed.append(identity)
        if row.get("status") != "created" or row.get("execution_queue_id") != "":
            raise RecoveryError(
                f"evaluation recovery task {identity[2]} is not created and unqueued"
            )
        for key in ("script_sha256", "parameters_sha256"):
            value_sha = row.get(key)
            if (
                type(value_sha) is not str
                or formal.SHA256_PATTERN.fullmatch(value_sha) is None
            ):
                raise RecoveryError(
                    f"evaluation recovery task {identity[2]} {key} is invalid"
                )
        inputs = row.get("input_model_ids")
        actions = row.get("actions")
        if (
            not isinstance(inputs, list)
            or any(
                type(item) is not str
                or len(item) != 32
                or any(character not in "0123456789abcdef" for character in item)
                for item in inputs
            )
            or not isinstance(actions, list)
            or "reset_force" not in actions
            or "install_gpu_preflight_source" not in actions
            or "restore_exact_parameters" not in actions
        ):
            raise RecoveryError(
                f"evaluation recovery task {identity[2]} action contract drifted"
            )
    if tuple(observed) != EXPECTED_RECOVERY_TASKS:
        raise RecoveryError("evaluation recovery exact task identity/order drifted")
    return {
        "path": str(resolved),
        "receipt_seal_sha256": seal,
        "attempt_receipt_seal_sha256": attempt_seal,
        "authoritative_plan_seal_sha256": (
            formal.AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256
        ),
        "task_count": len(rows),
        "task_ids_sha256": _content_sha256([item[2] for item in observed]),
        "all_tasks_created_unqueued": True,
        "replacement_tasks_created": False,
    }


def _task_project_id(task: object, *, context: str) -> str:
    values = {
        str(value)
        for value in (
            getattr(task, "project", None),
            getattr(getattr(task, "data", None), "project", None),
            getattr(getattr(task, "_data", None), "project", None),
        )
        if value not in {None, ""}
    }
    if values != {PROJECT_ID}:
        raise RecoveryError(f"{context} project identity drifted")
    return PROJECT_ID


def _task_script_source(task: object, *, context: str) -> str:
    script = getattr(getattr(task, "data", None), "script", None)
    if (
        str(getattr(script, "repository", "") or "") != ""
        or str(getattr(script, "working_dir", "") or "") != "."
        or str(getattr(script, "entry_point", "") or "")
        != "clearml_5090_bootstrap.py"
    ):
        raise RecoveryError(f"{context} script identity drifted")
    source = str(getattr(script, "diff", "") or "")
    if not source:
        raise RecoveryError(f"{context} source is empty")
    return source


def _input_models(task: object, *, context: str) -> list[dict[str, str]]:
    getter = getattr(task, "get_models", None)
    value = getter() if callable(getter) else None
    if not isinstance(value, Mapping):
        raise RecoveryError(f"{context} model inventory is unavailable")
    result: list[dict[str, str]] = []
    for model in value.get("input", []) or []:
        result.append(
            {
                "id": formal._task_id(
                    getattr(model, "id", ""), f"{context} input model"
                ),
                "name": str(getattr(model, "name", "") or ""),
                "task": str(getattr(model, "task", "") or ""),
            }
        )
    if len({item["id"] for item in result}) != len(result):
        raise RecoveryError(f"{context} input model inventory has duplicates")
    return result


def _artifact_inventory(task: object) -> list[dict[str, object]]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        return []
    result: list[dict[str, object]] = []
    for name, artifact in sorted(artifacts.items()):
        result.append(
            {
                "name": str(name),
                "hash": str(
                    getattr(artifact, "hash", None)
                    or getattr(artifact, "sha256", None)
                    or ""
                ),
                "size_bytes": getattr(artifact, "size", None)
                or getattr(artifact, "content_size", None),
                "url": str(getattr(artifact, "url", "") or ""),
            }
        )
    return result


def _console_evidence(task: object, *, status: str) -> dict[str, object] | None:
    if status not in ATTEMPT_EVIDENCE_STATUSES:
        return None
    getter = getattr(task, "get_reported_console_output", None)
    if not callable(getter):
        return {"available": False}
    try:
        lines = [str(line) for line in (getter(200) or [])][-200:]
    except Exception as error:
        return {"available": False, "read_error": type(error).__name__}
    encoded = "\n".join(lines).encode("utf-8")[-131_072:]
    folded = encoded.decode("utf-8", errors="replace").casefold()
    failure_markers = [
        marker
        for marker, fragments in {
            "cuda_out_of_memory": ("cuda out of memory", "cuda oom"),
            "traceback": ("traceback (most recent call last)",),
            "nonzero_exit": (
                "non-zero exit",
                "nonzero exit",
                "returned non-zero",
                "exit code",
            ),
        }.items()
        if any(fragment in folded for fragment in fragments)
    ]
    return {
        "available": True,
        # Remote console text can contain credentials printed by third-party
        # libraries. Its digest freezes the attempt without copying secrets
        # into a durable local receipt; the authoritative text remains in
        # ClearML and can be independently re-hashed when needed.
        "content_stored": False,
        "tail_line_count": len(lines),
        "tail_bytes": len(encoded),
        "tail_sha256": hashlib.sha256(encoded).hexdigest(),
        "failure_markers": failure_markers,
    }


def _attempt_snapshot(task: object, *, record: Mapping[str, object]) -> dict[str, object]:
    task_id = str(record["task_id"])
    context = f"{record['scope']} {record['subject']} evaluation"
    status = formal._status(task)
    data = getattr(task, "data", None)
    execution = getattr(data, "execution", None)
    parameters = formal._parameters(task)
    source = _task_script_source(task, context=context)
    tags = getattr(task, "tags", None)
    if tags is None:
        tags = getattr(data, "tags", None)
    result = {
        "scope": record["scope"],
        "subject": record["subject"],
        "task_id": task_id,
        "status": status,
        "name": str(getattr(task, "name", "") or ""),
        "parent_task_id": formal._task_parent(task),
        "project_id": _task_project_id(task, context=context),
        "execution_queue_id": str(getattr(execution, "queue", "") or ""),
        "last_worker": str(getattr(data, "last_worker", "") or ""),
        "status_reason": str(getattr(data, "status_reason", "") or ""),
        "status_message": str(getattr(data, "status_message", "") or ""),
        "created": str(getattr(data, "created", "") or ""),
        "started": str(getattr(data, "started", "") or ""),
        "completed": str(getattr(data, "completed", "") or ""),
        "published": str(getattr(data, "published", "") or ""),
        "script_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "parameters_sha256": _content_sha256(parameters),
        "input_models": _input_models(task, context=context),
        "tags": [str(value) for value in (tags or [])],
        "artifacts": _artifact_inventory(task),
        "console_evidence": _console_evidence(task, status=status),
    }
    result["attempt_evidence_sha256"] = _content_sha256(result)
    return result


def _validate_record_task(task: object, record: Mapping[str, object]) -> None:
    context = f"{record['scope']} {record['subject']} evaluation"
    if formal._task_id(getattr(task, "id", ""), context) != record["task_id"]:
        raise RecoveryError(f"{context} ID drifted")
    _task_project_id(task, context=context)
    if str(getattr(task, "name", "") or "") != record["task_name"]:
        raise RecoveryError(f"{context} name drifted")
    if formal._task_parent(task) != record["parent_task_id"]:
        raise RecoveryError(f"{context} parent drifted")
    status = formal._status(task)
    if status not in ALLOWED_PRE_STATUSES:
        raise RecoveryError(f"{context} status {status!r} is not recoverable")
    observed_parameters = formal._parameters(task)
    target_parameters = dict(record["parameters"])
    allowed_parameters = [target_parameters]
    if record.get("allow_legacy_inert_parameters") is True:
        allowed_parameters.append(
            {**target_parameters, **LEGACY_INERT_EVALUATION_PARAMETERS}
        )
    if observed_parameters not in allowed_parameters:
        raise RecoveryError(f"{context} parameters drifted")
    source = _task_script_source(task, context=context)
    if source not in {record["old_source"], record["target_source"]}:
        raise RecoveryError(f"{context} source drifted")
    queue_id = str(
        getattr(
            getattr(getattr(task, "data", None), "execution", None),
            "queue",
            "",
        )
        or ""
    )
    if queue_id not in {"", record["queue_id"]}:
        raise RecoveryError(f"{context} queue drifted")
    input_ids = [item["id"] for item in _input_models(task, context=context)]
    expected_model_id = str(record.get("input_model_id") or "")
    allowed_pre_model_id = str(
        record.get("allowed_pre_input_model_id") or expected_model_id
    )
    allowed_inputs = (
        [[]]
        if not allowed_pre_model_id
        else [[], [allowed_pre_model_id]]
    )
    if input_ids not in allowed_inputs:
        raise RecoveryError(f"{context} input binding drifted")


def _formal_inventory(
    task_class: object,
    *,
    authenticated_downloader: AuthenticatedArtifactDownloader | None = None,
) -> list[dict[str, object]]:
    controller = task_class.get_task(task_id=TRAINING_CONTROLLER_TASK_ID)
    manifest = formal._wait_for_controller_manifest(
        controller,
        deadline=time.monotonic() + 1.0,
        poll_seconds=0.01,
        metadata_only_artifact_gate=authenticated_downloader is None,
        authenticated_artifact_downloader=authenticated_downloader,
    )
    training = formal._parse_training_manifest(json.dumps(manifest))
    policies = formal._consume_training_provenance(
        task_class=task_class,
        provenance_task_id=PROVENANCE_TASK_ID,
        expected_provenance_script_sha256=PROVENANCE_SCRIPT_SHA256,
        controller=controller,
        controller_task_id=TRAINING_CONTROLLER_TASK_ID,
        manifest=manifest,
        training=training,
        metadata_only_artifact_gate=authenticated_downloader is None,
        authenticated_artifact_downloader=authenticated_downloader,
    )
    entries = formal._load_authoritative_evaluation_plan(
        task_class=task_class,
        producer_task_id=formal.AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID,
        expected_seal_sha256=formal.AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256,
        expected_project_id=policies.project_id,
        controller_task_id=TRAINING_CONTROLLER_TASK_ID,
        provenance_task_id=PROVENANCE_TASK_ID,
        template_task_id=EVALUATION_TEMPLATE_TASK_ID,
    )
    template = task_class.get_task(task_id=EVALUATION_TEMPLATE_TASK_ID)
    template_source = _task_script_source(template, context="formal template")
    target_source = formal._inject_gpu_memory_preflight(
        template_source, context="formal recovery template"
    )
    training_by_subject = {item["subject"]: item for item in training}
    result: list[dict[str, object]] = []
    for index, entry in enumerate(entries, start=1):
        subject = entry["subject"]
        training_item = training_by_subject[subject]
        record = {
            "scope": "formal_w3",
            "subject": subject,
            "task_id": entry["evaluation_task_id"],
            "task_name": formal._planned_evaluation_name(
                controller_task_id=TRAINING_CONTROLLER_TASK_ID,
                index=index,
                subject=subject,
            ),
            "parent_task_id": TRAINING_CONTROLLER_TASK_ID,
            "queue": entry["queue"],
            "queue_id": formal.EXPECTED_QUEUE_IDS[entry["queue"]],
            "parameters": formal._evaluation_parameters(
                template, training_item=training_item
            ),
            "input_model_id": "",
            "allowed_pre_input_model_id": training_item["training_model_id"],
            "allow_legacy_inert_parameters": True,
            "old_source": template_source,
            "target_source": target_source,
        }
        task = task_class.get_task(task_id=entry["evaluation_task_id"])
        _validate_record_task(task, record)
        record["task"] = task
        result.append(record)
    return result


def _candidate_inventory(task_class: object) -> list[dict[str, object]]:
    by_label = {spec.label: spec for spec in candidates.CANDIDATES}
    result: list[dict[str, object]] = []
    for label in ("E1", "E3"):
        spec = by_label[label]
        training = task_class.get_task(task_id=spec.training_task_id)
        template = task_class.get_task(task_id=spec.template_task_id)
        candidates._validate_template(template, spec)
        if candidates._validate_static_training(training, spec) != "completed":
            raise RecoveryError(f"candidate {label} training is not completed")
        binding = candidates._validate_completed_training(training, spec)
        template_source = candidates._script(template)["diff"]
        target_source = candidates._candidate_script_patch(spec, template_source)
        old_source = target_source.replace(
            candidates._GPU_PREFLIGHT_REPLACEMENT,
            candidates._GPU_PREFLIGHT_ANCHOR,
            1,
        )
        record = {
            "scope": "candidate",
            "subject": spec.subject,
            "label": label,
            "task_id": CANDIDATE_EVALUATION_IDS[label],
            "task_name": candidates._evaluation_name(spec),
            "parent_task_id": spec.training_task_id,
            "queue": spec.queue,
            "queue_id": candidates.QUEUE_IDS[spec.queue],
            "parameters": candidates._evaluation_parameters(
                template, spec, binding
            ),
            "input_model_id": binding["model_id"],
            "input_model_name": f"{spec.subject}_final_checkpoint",
            "old_source": old_source,
            "target_source": target_source,
        }
        task = task_class.get_task(task_id=CANDIDATE_EVALUATION_IDS[label])
        _validate_record_task(task, record)
        record["task"] = task
        result.append(record)
    return result


def _recovery_inventory(
    task_class: object,
    *,
    authenticated_downloader: AuthenticatedArtifactDownloader | None = None,
) -> list[dict[str, object]]:
    result = [
        *_formal_inventory(
            task_class, authenticated_downloader=authenticated_downloader
        ),
        *_candidate_inventory(task_class),
    ]
    ids = [str(record["task_id"]) for record in result]
    if len(result) != 28 or len(set(ids)) != 28:
        raise RecoveryError("recovery task inventory is not exactly 26+2 unique IDs")
    return result


def _public_record(record: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"task", "old_source", "target_source", "parameters"}
    } | {
        "old_script_sha256": hashlib.sha256(
            str(record["old_source"]).encode("utf-8")
        ).hexdigest(),
        "target_script_sha256": hashlib.sha256(
            str(record["target_source"]).encode("utf-8")
        ).hexdigest(),
        "parameters_sha256": _content_sha256(record["parameters"]),
    }


def build_attempt_receipt(
    task_class: object,
    *,
    authenticated_downloader: AuthenticatedArtifactDownloader | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    inventory = _recovery_inventory(
        task_class, authenticated_downloader=authenticated_downloader
    )
    attempts = [
        _attempt_snapshot(record["task"], record=record) for record in inventory
    ]
    receipt = _seal(
        {
            "schema_version": 1,
            "receipt_type": "resilient_v2x_evaluation_attempt_evidence",
            "generated_at_utc": _utc_timestamp(),
            "remote_state_changed": False,
            "authoritative_plan_producer_task_id": (
                formal.AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
            ),
            "authoritative_plan_seal_sha256": (
                formal.AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256
            ),
            "recovery_policy": (
                "reuse_exact_ids_no_clone_freeze_attempt_before_force_reset"
            ),
            "planned_tasks": [_public_record(record) for record in inventory],
            "attempts": attempts,
            "authenticated_artifact_downloads": (
                list(authenticated_downloader.audit_records)
                if authenticated_downloader is not None
                else []
            ),
        }
    )
    return receipt, inventory


def _authoritative_task(task_class: object, task_id: str) -> object:
    task = task_class.get_task(task_id=task_id)
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()
    if formal._task_id(getattr(task, "id", ""), "recovery task") != task_id:
        raise RecoveryError("recovery authoritative task identity drifted")
    return task


def _wait_status(
    task_class: object,
    task_id: str,
    expected: set[str],
    *,
    sleeper: Callable[[float], None],
) -> object:
    last = None
    for attempt in range(5):
        task = _authoritative_task(task_class, task_id)
        last = formal._status(task)
        if last in expected:
            return task
        if attempt != 4:
            sleeper(1.0)
    raise RecoveryError(
        f"task {task_id} did not reach {sorted(expected)}; observed {last!r}"
    )


def _restore_one(
    task_class: object,
    record: Mapping[str, object],
    *,
    sleeper: Callable[[float], None],
) -> dict[str, object]:
    task_id = str(record["task_id"])
    task = _authoritative_task(task_class, task_id)
    _validate_record_task(task, record)
    status = formal._status(task)
    actions: list[str] = []
    if status == "queued":
        response = task_class.dequeue(task)
        if response is False:
            raise RecoveryError(f"task {task_id} dequeue was not confirmed")
        actions.append("dequeue")
        task = _wait_status(
            task_class,
            task_id,
            {"created", "stopped"},
            sleeper=sleeper,
        )
    elif status == "in_progress":
        stopper = getattr(task, "mark_stopped", None)
        if not callable(stopper):
            raise RecoveryError(f"task {task_id} cannot be explicitly stopped")
        stopper(force=True, status_message="sealed evaluation attempt recovery")
        actions.append("mark_stopped_force")
        task = _wait_status(
            task_class, task_id, {"stopped"}, sleeper=sleeper
        )

    resetter = getattr(task, "reset", None)
    if not callable(resetter):
        raise RecoveryError(f"task {task_id} cannot be reset")
    resetter(force=True)
    actions.append("reset_force")
    task = _wait_status(task_class, task_id, {"created"}, sleeper=sleeper)

    inputs = _input_models(task, context=f"recovery task {task_id}")
    if inputs:
        remover = getattr(task, "remove_input_models", None)
        if not callable(remover):
            raise RecoveryError(f"task {task_id} cannot clear old input bindings")
        remover([item["id"] for item in inputs])
        actions.append("clear_input_models")
    model_id = str(record.get("input_model_id") or "")
    if model_id:
        setter = getattr(task, "set_input_model", None)
        if not callable(setter):
            raise RecoveryError(f"task {task_id} cannot restore its input model")
        setter(
            model_id=model_id,
            name=str(record["input_model_name"]),
            update_task_design=False,
            update_task_labels=False,
        )
        actions.append("restore_input_model")
    task.set_script(
        repository="",
        branch="",
        commit="",
        diff=str(record["target_source"]),
        working_dir=".",
        entry_point="clearml_5090_bootstrap.py",
    )
    actions.append("install_gpu_preflight_source")
    task.set_parameters(dict(record["parameters"]))
    actions.append("restore_exact_parameters")
    raw_tags = getattr(task, "tags", None)
    if raw_tags is None:
        raw_tags = getattr(getattr(task, "data", None), "tags", None)
    tags = [str(value) for value in (raw_tags or [])]
    if "dependency-released" in tags:
        setter = getattr(task, "set_tags", None)
        if not callable(setter) or setter(
            [value for value in tags if value != "dependency-released"]
        ) is False:
            raise RecoveryError(f"task {task_id} release tag cannot be cleared")
        actions.append("clear_dependency_released_tag")
    flusher = getattr(task, "flush", None)
    if callable(flusher) and flusher(wait_for_uploads=True) is False:
        raise RecoveryError(f"task {task_id} flush was not confirmed")
    task = _wait_status(task_class, task_id, {"created"}, sleeper=sleeper)
    _validate_record_task(task, record)
    observed_source = _task_script_source(task, context=f"recovered task {task_id}")
    if observed_source != record["target_source"]:
        raise RecoveryError(f"task {task_id} preflight source was not installed")
    queue_id = str(
        getattr(
            getattr(getattr(task, "data", None), "execution", None),
            "queue",
            "",
        )
        or ""
    )
    if queue_id:
        raise RecoveryError(f"task {task_id} retained a stale execution queue")
    input_ids = [
        item["id"] for item in _input_models(task, context=f"recovered task {task_id}")
    ]
    expected_inputs = [model_id] if model_id else []
    if input_ids != expected_inputs:
        raise RecoveryError(f"task {task_id} input binding was not restored")
    return {
        "task_id": task_id,
        "scope": record["scope"],
        "subject": record["subject"],
        "status": "created",
        "execution_queue_id": "",
        "script_sha256": hashlib.sha256(
            observed_source.encode("utf-8")
        ).hexdigest(),
        "parameters_sha256": _content_sha256(formal._parameters(task)),
        "input_model_ids": input_ids,
        "actions": actions,
    }


def execute_recovery(
    task_class: object,
    inventory: Sequence[Mapping[str, object]],
    *,
    attempt_receipt: Mapping[str, object],
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    recovered: list[dict[str, object]] = []
    for record in inventory:
        recovered.append(_restore_one(task_class, record, sleeper=sleeper))
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": "resilient_v2x_evaluation_attempt_recovery",
            "generated_at_utc": _utc_timestamp(),
            "status": "recovered",
            "remote_state_changed": True,
            "attempt_receipt_seal_sha256": attempt_receipt["seal_sha256"],
            "authoritative_plan_producer_task_id": (
                formal.AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
            ),
            "authoritative_plan_seal_sha256": (
                formal.AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256
            ),
            "formal_task_count": 26,
            "candidate_task_ids": dict(CANDIDATE_EVALUATION_IDS),
            "all_tasks_created_unqueued": True,
            "replacement_tasks_created": False,
            "recovered_tasks": recovered,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--attempt-receipt",
        type=Path,
        help="Required with --execute; written before the first remote mutation.",
    )
    return parser


def _print_receipt_summary(path: Path, receipt: Mapping[str, object]) -> None:
    attempts = receipt.get("attempts")
    recovered = receipt.get("recovered_tasks")
    print(
        json.dumps(
            {
                "receipt": str(path),
                "receipt_type": receipt.get("receipt_type"),
                "status": receipt.get("status"),
                "remote_state_changed": receipt.get("remote_state_changed"),
                "task_count": (
                    len(attempts)
                    if isinstance(attempts, list)
                    else len(recovered)
                    if isinstance(recovered, list)
                    else None
                ),
                "seal_sha256": receipt.get("seal_sha256"),
            },
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.execute_token != EXECUTE_TOKEN:
        raise RecoveryError(f"exact execute token required: {EXECUTE_TOKEN}")
    if args.execute and args.attempt_receipt is None:
        raise RecoveryError("--attempt-receipt is required with --execute")
    if not args.execute and args.execute_token:
        raise RecoveryError("--execute-token requires --execute")
    if not args.execute and args.attempt_receipt is not None:
        raise RecoveryError("--attempt-receipt requires --execute")
    if args.attempt_receipt is not None and (
        args.attempt_receipt.resolve() == args.receipt.resolve()
    ):
        raise RecoveryError("attempt and recovery receipts must use distinct paths")
    try:
        from clearml import Task
        from clearml.backend_api.session import Session
    except ImportError as error:
        raise RecoveryError("ClearML is required for recovery inspection") from error
    session = Session()
    downloader = AuthenticatedArtifactDownloader(
        files_server_url=str(Session.get_files_server_host()),
        add_auth_headers=session.add_auth_headers,
    )
    attempt, inventory = build_attempt_receipt(
        Task, authenticated_downloader=downloader
    )
    if not args.execute:
        plan = _seal(
            {
                **attempt,
                "receipt_type": "resilient_v2x_evaluation_attempt_recovery_plan",
                "status": "planned",
                "execute_token": EXECUTE_TOKEN,
            }
        )
        _write_new(args.receipt, plan)
        _print_receipt_summary(args.receipt, plan)
        return 0
    _write_new(args.attempt_receipt, attempt)
    try:
        recovered = execute_recovery(
            Task,
            inventory,
            attempt_receipt=attempt,
        )
    except Exception as error:
        failure = _seal(
            {
                "schema_version": 1,
                "receipt_type": "resilient_v2x_evaluation_attempt_recovery",
                "generated_at_utc": _utc_timestamp(),
                "status": "failed_closed",
                "remote_state_changed": True,
                "attempt_receipt_seal_sha256": attempt["seal_sha256"],
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        _write_new(args.receipt, failure)
        raise
    _write_new(args.receipt, recovered)
    _print_receipt_summary(args.receipt, recovered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
