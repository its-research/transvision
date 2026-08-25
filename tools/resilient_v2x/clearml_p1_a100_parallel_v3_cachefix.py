#!/usr/bin/env python3
"""Cache-safe continuation of the sealed P1 A100 parallel-v2 campaign.

This is a migration wrapper around the byte-pinned parallel-v2 supervisor.  It
does not change the scientific plan, pinset, execution key, controller, child
names, or journal schema.  Its only runtime change is that every mutable
journal read is authoritative: a fresh backend task descriptor is paired with
a forced artifact download and the descriptor hash/content size must match the
raw JSON bytes before the journal is accepted.

Default mode is local dry-run.  ``--preflight`` is remote read-only.  Remote
mutation requires ``--execute`` and the exact v3 acknowledgement token.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path


V2_FILENAME = "clearml_p1_a100_parallel_v2_supervisor.py"
FROZEN_V2_SHA256 = "124909587ef4866c615ca05427c9f5242d8230661b5b0ae98191e4589c477148"
V3_BARRIER_ARTIFACT = "p1_a100_parallel_v3_cachefix_barrier"
V3_BARRIER_DOCUMENT_TYPE = "resilient_v2x_p1_a100_parallel_v3_cachefix_barrier"
V3_PREFLIGHT_DOCUMENT_TYPE = "resilient_v2x_p1_a100_parallel_v3_cachefix_preflight"
V3_DRY_RUN_DOCUMENT_TYPE = "resilient_v2x_p1_a100_parallel_v3_cachefix_dry_run"
V3_EXECUTION_RECEIPT_DOCUMENT_TYPE = (
    "resilient_v2x_p1_a100_parallel_v3_cachefix_execution_receipt"
)
V3_CONTROLLER_TAG = "p1-a100-parallel-v3-cachefix"
V3_EXECUTION_TOKEN = "RECOVER_AND_EXECUTE_P1_A100_PARALLEL_V3_CACHEFIX"
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
LIVE_V2_AMENDMENT_SEAL_SHA256 = (
    "bff8d1f8ece8b37ac391b84c6a7a092604a2fb01d36f8b9ede551d6843228dfa"
)
LIVE_INITIAL_WAL_SEAL_SHA256 = (
    "eb74a4ebcde3e586b0ac681aa85b7b4ab4384afa37d15873672f71519ec0fd29"
)
LIVE_INITIAL_PRIOR_JOURNAL_SEAL_SHA256 = (
    "dd6f3c0b5310a8dc37d01630b4337509d3f70de87f563077e5ca36c329c80b49"
)
LIVE_INITIAL_CANDIDATE_JOURNAL_SEAL_SHA256 = (
    "d9e2ecd3ca9f84c8f70d75909f55490b25e65ba26e564da04adf0135589f2e3e"
)
LIVE_INITIAL_CANDIDATE_DESCRIPTOR_SHA256 = (
    "d1a02429035b26678b258647a403a44e2f2f6cd615988ef9bd8fc9457d8d37d2"
)
LIVE_INITIAL_CANDIDATE_DESCRIPTOR_SIZE = 4599


def _load_v2() -> object:
    path = Path(__file__).resolve().with_name(V2_FILENAME)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != FROZEN_V2_SHA256:
        raise RuntimeError(
            "pinned parallel-v2 supervisor SHA-256 mismatch: "
            f"expected {FROZEN_V2_SHA256}, observed {observed}"
        )
    spec = importlib.util.spec_from_file_location(
        "_resilientv2x_p1_a100_parallel_v2_pinned", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the pinned parallel-v2 supervisor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = _load_v2()
base = v2.base
P1ExecutorError = base.P1ExecutorError
P1RecoverableTimeout = base.P1RecoverableTimeout


class P1ParallelV3Error(P1ExecutorError):
    """The cache-fix migration or its authoritative evidence is invalid."""


def _v3_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _verify_frozen_bindings() -> tuple[dict[str, object], dict[str, object]]:
    path = Path(__file__).resolve().with_name(V2_FILENAME)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_V2_SHA256:
        raise P1ParallelV3Error("frozen parallel-v2 supervisor SHA-256 drifted")
    try:
        plan, pinset = v2._verify_frozen_bindings()
    except base.P1ExecutorError as error:
        raise P1ParallelV3Error(str(error)) from error
    if v2._supervisor_sha256() != FROZEN_V2_SHA256:
        raise P1ParallelV3Error("parallel-v2 self fingerprint drifted")
    return plan, pinset


def _validate_control_host() -> dict[str, str]:
    try:
        return v2._validate_control_host()
    except base.P1ExecutorError as error:
        raise P1ParallelV3Error(str(error)) from error


def _sdk_json_bytes(value: Mapping[str, object]) -> bytes:
    """Match ClearML 2.1.x dict-artifact serialization byte-for-byte."""

    return json.dumps(dict(value), sort_keys=True, indent=4).encode("utf-8")


def _descriptor_value(artifact: object, name: str) -> dict[str, object]:
    artifact_name = str(getattr(artifact, "name", "") or "")
    if artifact_name != name:
        raise P1ParallelV3Error(f"{name} descriptor key drifted")
    artifact_hash = str(getattr(artifact, "hash", "") or "")
    if len(artifact_hash) != 64 or any(
        item not in "0123456789abcdef" for item in artifact_hash
    ):
        raise P1ParallelV3Error(f"{name} descriptor hash is invalid")
    content_size = getattr(artifact, "size", None)
    if content_size is None:
        content_size = getattr(artifact, "content_size", None)
    if type(content_size) is not int or not 0 < content_size <= MAX_ARTIFACT_BYTES:
        raise P1ParallelV3Error(f"{name} descriptor content size is invalid")
    uri = str(getattr(artifact, "url", "") or getattr(artifact, "uri", "") or "")
    if not uri:
        raise P1ParallelV3Error(f"{name} descriptor URI is missing")
    artifact_type = str(getattr(artifact, "type", "") or "")
    if artifact_type not in {"dict", "JSON"}:
        raise P1ParallelV3Error(f"{name} descriptor type drifted")
    mode = str(getattr(artifact, "mode", "") or "")
    if mode != "output":
        raise P1ParallelV3Error(f"{name} descriptor mode drifted")
    if not uri.startswith(base.FILES_SERVER_URI + "/"):
        raise P1ParallelV3Error(f"{name} descriptor URI host drifted")
    return {
        "key": artifact_name,
        "hash": artifact_hash,
        "content_size": content_size,
        "uri": uri,
        "type": artifact_type,
        "mode": mode,
    }


def _read_regular_bytes(path_value: object, context: str) -> bytes:
    if not isinstance(path_value, (str, os.PathLike)):
        raise P1ParallelV3Error(f"{context} local copy is not a path")
    path = Path(path_value).absolute()
    if path.is_symlink():
        raise P1ParallelV3Error(f"{context} local copy is a symlink")
    path = path.resolve(strict=True)
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_ARTIFACT_BYTES
        ):
            raise P1ParallelV3Error(f"{context} local copy is outside safe bounds")
        payload = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) != before.st_size or any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise P1ParallelV3Error(f"{context} local copy changed while being read")
    return payload


def _force_artifact_mapping(
    artifact: object,
    *,
    name: str,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    """Force-download one fresh artifact and validate its backend descriptor."""

    descriptor = _descriptor_value(artifact, name)
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        raise P1ParallelV3Error(f"{name} has no forced-download interface")
    try:
        path = getter(
            extract_archive=False,
            raise_on_error=True,
            force_download=True,
        )
    except TypeError as error:
        raise P1ParallelV3Error(
            f"{name} SDK cannot guarantee a forced artifact download"
        ) from error
    raw = _read_regular_bytes(path, name)
    raw_hash = hashlib.sha256(raw).hexdigest()
    if raw_hash != descriptor["hash"] or len(raw) != descriptor["content_size"]:
        raise P1ParallelV3Error(f"{name} descriptor and forced bytes differ")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P1ParallelV3Error(f"{name} forced bytes are not JSON: {error}") from None
    return base._mapping(value, name), descriptor, raw


def _backend_artifact_descriptor(
    task_class: object, *, task_id: str, artifact_name: str
) -> object:
    """Read one artifact descriptor directly from tasks.get_by_id.

    A Task/Artifact object can retain both in-memory and disk-cached state.  The
    service response is therefore the commit authority; the returned descriptor
    is wrapped in a brand-new read-only Artifact only after this exact lookup.
    """

    task_id = base._clearml_id(task_id, "descriptor task")
    session_getter = getattr(task_class, "_get_default_session", None)
    if not callable(session_getter):
        raise P1ParallelV3Error("ClearML cannot expose a direct task session")
    try:
        from clearml.backend_api.services.v2_13 import tasks
    except ImportError as error:
        raise P1ParallelV3Error(f"ClearML task API is unavailable: {error}") from None
    session = session_getter()
    sender = getattr(session, "send", None)
    if not callable(sender):
        raise P1ParallelV3Error("ClearML task session cannot send get_by_id")
    response = sender(tasks.GetByIdRequest(task=task_id))
    response_object = getattr(response, "response", None)
    task_data = getattr(response_object, "task", None)
    if task_data is None:
        raise P1ParallelV3Error("tasks.get_by_id returned no task")
    if base._clearml_id(getattr(task_data, "id", None), "descriptor task") != task_id:
        raise P1ParallelV3Error("tasks.get_by_id task identity drifted")
    execution = getattr(task_data, "execution", None)
    artifacts = getattr(execution, "artifacts", None)
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise P1ParallelV3Error("tasks.get_by_id artifact inventory is unreadable")
    matches = [
        item for item in artifacts if getattr(item, "key", None) == artifact_name
    ]
    if len(matches) != 1:
        raise P1ParallelV3Error(
            f"tasks.get_by_id has missing or duplicate {artifact_name} descriptors"
        )
    return matches[0]


def _read_authoritative_artifact(
    task_class: object,
    *,
    controller_id: str,
    artifact_name: str,
    context: str | None = None,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    """Pair a direct backend descriptor with a brand-new forced Artifact read."""

    descriptor_object = _backend_artifact_descriptor(
        task_class,
        task_id=controller_id,
        artifact_name=artifact_name,
    )
    try:
        from clearml.binding.artifacts import Artifact
    except ImportError as error:
        raise P1ParallelV3Error(
            f"ClearML Artifact wrapper is unavailable: {error}"
        ) from None
    artifact = Artifact(descriptor_object)
    return _force_artifact_mapping(
        artifact,
        name=context or artifact_name,
    )


def _fresh_task(task_class: object, task_id: str) -> object:
    task_id = base._clearml_id(task_id, "task")
    task = task_class.get_task(task_id=task_id)
    if base._clearml_id(getattr(task, "id", None), "fresh task") != task_id:
        raise P1ParallelV3Error("fresh task identity drifted")
    return task


def _authoritative_artifact(
    task_class: object,
    *,
    task_id: str,
    artifact_name: str,
) -> tuple[object, dict[str, object], dict[str, object], bytes]:
    task = _fresh_task(task_class, task_id)
    value, descriptor, raw = _read_authoritative_artifact(
        task_class,
        controller_id=task_id,
        artifact_name=artifact_name,
        context=artifact_name,
    )
    return task, value, descriptor, raw


def _validate_authoritative_immutable_bindings(
    task_class: object,
    *,
    controller_id: str,
    plan: Mapping[str, object],
    pinset: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Validate the controller's sealed plan and pinset from backend bytes."""

    receipt: dict[str, dict[str, object]] = {}
    for artifact_name, expected in (
        (base.PLAN_ARTIFACT, plan),
        (base.PINSET_ARTIFACT, pinset),
    ):
        _task, observed, descriptor, raw = _authoritative_artifact(
            task_class,
            task_id=controller_id,
            artifact_name=artifact_name,
        )
        expected_raw = _sdk_json_bytes(expected)
        if (
            observed != expected
            or raw != expected_raw
            or descriptor["hash"] != hashlib.sha256(expected_raw).hexdigest()
            or descriptor["content_size"] != len(expected_raw)
        ):
            raise P1ParallelV3Error(f"controller {artifact_name} drifted")
        receipt[artifact_name] = copy.deepcopy(descriptor)
    return receipt


def _validate_journal(
    value: Mapping[str, object], *, controller_id: str
) -> dict[str, object]:
    plan, _pinset = _verify_frozen_bindings()
    try:
        return base._validate_journal(
            value,
            plan=plan,
            execution_key=v2.SEALED_EXECUTION_KEY,
            controller_id=controller_id,
        )
    except base.P1ExecutorError as error:
        raise P1ParallelV3Error(str(error)) from error


def _authoritative_remote_journal(
    task_class: object, controller_id: str
) -> tuple[object, dict[str, object]]:
    controller_id = base._clearml_id(controller_id, "controller")
    controller, value, _descriptor, _raw = _authoritative_artifact(
        task_class,
        task_id=controller_id,
        artifact_name=base.JOURNAL_ARTIFACT,
    )
    return controller, _validate_journal(value, controller_id=controller_id)


def _authoritative_load_journal(
    controller: object,
    *,
    plan: Mapping[str, object],
    execution_key: str,
    controller_id: str,
    task_class: object | None = None,
) -> dict[str, object]:
    if execution_key != v2.SEALED_EXECUTION_KEY:
        raise P1ParallelV3Error("mutable journal execution key drifted")
    if plan.get("seal_sha256") != v2.SEALED_PLAN_SHA256:
        raise P1ParallelV3Error("mutable journal plan drifted")
    if base._clearml_id(
        getattr(controller, "id", None), "controller"
    ) != base._clearml_id(controller_id, "controller"):
        raise P1ParallelV3Error("mutable journal controller drifted")
    task_type = task_class or type(controller)
    value, _descriptor, _raw = _read_authoritative_artifact(
        task_type,
        controller_id=controller_id,
        artifact_name=base.JOURNAL_ARTIFACT,
        context=base.JOURNAL_ARTIFACT,
    )
    return _validate_journal(value, controller_id=controller_id)


def _validate_v2_amendment(
    task_class: object, controller_id: str, journal: Mapping[str, object]
) -> dict[str, object]:
    controller_id = base._clearml_id(controller_id, "controller")
    _task, value, _descriptor, _raw = _authoritative_artifact(
        task_class,
        task_id=controller_id,
        artifact_name=v2.AMENDMENT_ARTIFACT,
    )
    try:
        amendment = v2._validate_installed_amendment(
            value,
            controller_id=controller_id,
            current_journal=journal,
        )
        if amendment.get("seal_sha256") != LIVE_V2_AMENDMENT_SEAL_SHA256:
            raise P1ParallelV3Error("live parallel-v2 amendment seal drifted")
        return amendment
    except base.P1ExecutorError as error:
        raise P1ParallelV3Error(str(error)) from error


def _journal_transition_kind(
    prior: Mapping[str, object], candidate: Mapping[str, object]
) -> str:
    """Accept only a single safe-writer revision from the sealed pending WAL."""

    if int(candidate.get("revision", -1)) != int(prior.get("revision", -1)) + 1:
        raise P1ParallelV3Error("pending journal revisions are not consecutive")
    top_changed = {
        key
        for key in set(prior) | set(candidate)
        if prior.get(key) != candidate.get(key)
    }
    if not top_changed.issubset({"revision", "seal_sha256", "status", "tasks"}):
        raise P1ParallelV3Error("pending journal changed immutable top-level fields")
    prior_rows = prior.get("tasks")
    candidate_rows = candidate.get("tasks")
    if not isinstance(prior_rows, list) or not isinstance(candidate_rows, list):
        raise P1ParallelV3Error("pending journal row inventory is invalid")
    changed_rows = [
        (before, after)
        for before, after in zip(prior_rows, candidate_rows, strict=True)
        if before != after
    ]
    if not changed_rows:
        if top_changed != {"revision", "seal_sha256", "status"}:
            raise P1ParallelV3Error("pending status revision has an invalid diff")
        return "journal_status"
    if len(changed_rows) != 1:
        raise P1ParallelV3Error("pending journal changed more than one task row")
    before = base._mapping(changed_rows[0][0], "pending prior row")
    after = base._mapping(changed_rows[0][1], "pending candidate row")
    immutable = {"kind", "seed_index", "subject", "task_key", "training_seed"}
    if any(before.get(field) != after.get(field) for field in immutable):
        raise P1ParallelV3Error("pending journal changed immutable task identity")
    row_changed = {
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    }
    if not row_changed.issubset({"task_id", "state", "server_status", "result"}):
        raise P1ParallelV3Error(
            "pending journal task diff is outside safe-writer fields"
        )
    if not {"revision", "seal_sha256", "tasks"}.issubset(top_changed):
        raise P1ParallelV3Error("pending task revision lacks mandatory revision fields")
    return "task_row"


def _read_pending_wal(controller_id: str) -> dict[str, object] | None:
    try:
        receipt = v2._read_pending_journal_receipt(controller_id=controller_id)
    except base.P1ExecutorError as error:
        raise P1ParallelV3Error(str(error)) from error
    if receipt is None:
        return None
    if receipt.get("status") != "pending":
        raise P1ParallelV3Error("v3 cannot recover a poisoned journal receipt")
    prior = _validate_journal(
        base._mapping(receipt.get("prior_journal"), "pending prior journal"),
        controller_id=controller_id,
    )
    candidate = _validate_journal(
        base._mapping(receipt.get("candidate_journal"), "pending candidate journal"),
        controller_id=controller_id,
    )
    _journal_transition_kind(prior, candidate)
    return receipt


def _build_v3_barrier(
    *,
    controller_id: str,
    amendment: Mapping[str, object],
    wal: Mapping[str, object],
) -> dict[str, object]:
    plan, pinset = _verify_frozen_bindings()
    controller_id = base._clearml_id(controller_id, "controller")
    prior = _validate_journal(
        base._mapping(wal.get("prior_journal"), "v3 source prior journal"),
        controller_id=controller_id,
    )
    candidate = _validate_journal(
        base._mapping(wal.get("candidate_journal"), "v3 source candidate journal"),
        controller_id=controller_id,
    )
    transition = _journal_transition_kind(prior, candidate)
    candidate_raw = _sdk_json_bytes(candidate)
    if (
        amendment.get("seal_sha256") != LIVE_V2_AMENDMENT_SEAL_SHA256
        or wal.get("seal_sha256") != LIVE_INITIAL_WAL_SEAL_SHA256
        or prior.get("seal_sha256") != LIVE_INITIAL_PRIOR_JOURNAL_SEAL_SHA256
        or candidate.get("seal_sha256") != LIVE_INITIAL_CANDIDATE_JOURNAL_SEAL_SHA256
        or prior.get("revision") != 5
        or candidate.get("revision") != 6
        or hashlib.sha256(candidate_raw).hexdigest()
        != LIVE_INITIAL_CANDIDATE_DESCRIPTOR_SHA256
        or len(candidate_raw) != LIVE_INITIAL_CANDIDATE_DESCRIPTOR_SIZE
    ):
        raise P1ParallelV3Error("initial live migration pins drifted")
    barrier: dict[str, object] = {
        "schema_version": 1,
        "document_type": V3_BARRIER_DOCUMENT_TYPE,
        "status": "active",
        "reason": "clearml_mutable_artifact_cache_requires_authoritative_forced_reads",
        "v2_supervisor_sha256": FROZEN_V2_SHA256,
        "v3_supervisor_sha256": _v3_sha256(),
        "base_executor_sha256": v2.FROZEN_BASE_SHA256,
        "plan_seal_sha256": plan["seal_sha256"],
        "pinset_seal_sha256": pinset["seal_sha256"],
        "execution_key": v2.SEALED_EXECUTION_KEY,
        "controller_task_id": controller_id,
        "v2_amendment_artifact": v2.AMENDMENT_ARTIFACT,
        "v2_amendment_seal_sha256": amendment["seal_sha256"],
        "source_pending_wal": {
            "wal_document_type": wal["document_type"],
            "wal_seal_sha256": wal["seal_sha256"],
            "prior_revision": prior["revision"],
            "prior_journal_seal_sha256": prior["seal_sha256"],
            "candidate_revision": candidate["revision"],
            "candidate_journal_seal_sha256": candidate["seal_sha256"],
            "transition_kind": transition,
        },
        "cache_fix_contract": {
            "backend_descriptor_required": True,
            "fresh_task_required": True,
            "force_download_required": True,
            "descriptor_hash_required": True,
            "descriptor_content_size_required": True,
            "raw_json_exact_required": True,
            "pending_candidate_retry_per_invocation_limit": 1,
            "pending_candidate_retry_must_be_byte_identical": True,
            "old_v2_execution_forbidden": True,
        },
        "journal_writer_attribution": {
            "scheduler_core_sha256": FROZEN_V2_SHA256,
            "runtime_cachefix_sha256": _v3_sha256(),
            "upload_protocol": "frozen_v2_clearml_sdk_plus_authoritative_confirm_v1",
        },
        "controller_tag": V3_CONTROLLER_TAG,
        "control_host_contract": copy.deepcopy(v2.CONTROL_HOST_CONTRACT),
    }
    barrier["seal_sha256"] = base._seal(barrier)
    return _validate_v3_barrier(
        barrier,
        controller_id=controller_id,
        amendment=amendment,
    )


def _validate_v3_barrier(
    value: Mapping[str, object],
    *,
    controller_id: str,
    amendment: Mapping[str, object],
) -> dict[str, object]:
    barrier = base._mapping(value, "parallel-v3 cache-fix barrier")
    base._exact_keys(
        barrier,
        {
            "base_executor_sha256",
            "cache_fix_contract",
            "control_host_contract",
            "controller_tag",
            "controller_task_id",
            "document_type",
            "execution_key",
            "journal_writer_attribution",
            "pinset_seal_sha256",
            "plan_seal_sha256",
            "reason",
            "schema_version",
            "seal_sha256",
            "source_pending_wal",
            "status",
            "v2_amendment_artifact",
            "v2_amendment_seal_sha256",
            "v2_supervisor_sha256",
            "v3_supervisor_sha256",
        },
        "parallel-v3 cache-fix barrier",
    )
    if barrier.get("seal_sha256") != base._seal(barrier):
        raise P1ParallelV3Error("parallel-v3 barrier seal mismatch")
    plan, pinset = _verify_frozen_bindings()
    expected = {
        "schema_version": 1,
        "document_type": V3_BARRIER_DOCUMENT_TYPE,
        "status": "active",
        "reason": "clearml_mutable_artifact_cache_requires_authoritative_forced_reads",
        "v2_supervisor_sha256": FROZEN_V2_SHA256,
        "v3_supervisor_sha256": _v3_sha256(),
        "base_executor_sha256": v2.FROZEN_BASE_SHA256,
        "plan_seal_sha256": plan["seal_sha256"],
        "pinset_seal_sha256": pinset["seal_sha256"],
        "execution_key": v2.SEALED_EXECUTION_KEY,
        "controller_task_id": base._clearml_id(controller_id, "controller"),
        "v2_amendment_artifact": v2.AMENDMENT_ARTIFACT,
        "v2_amendment_seal_sha256": amendment["seal_sha256"],
        "controller_tag": V3_CONTROLLER_TAG,
        "control_host_contract": v2.CONTROL_HOST_CONTRACT,
    }
    for field, item in expected.items():
        if barrier.get(field) != item:
            raise P1ParallelV3Error(f"parallel-v3 barrier {field} drifted")
    contract = base._mapping(barrier.get("cache_fix_contract"), "v3 cache contract")
    if contract != {
        "backend_descriptor_required": True,
        "fresh_task_required": True,
        "force_download_required": True,
        "descriptor_hash_required": True,
        "descriptor_content_size_required": True,
        "raw_json_exact_required": True,
        "pending_candidate_retry_per_invocation_limit": 1,
        "pending_candidate_retry_must_be_byte_identical": True,
        "old_v2_execution_forbidden": True,
    }:
        raise P1ParallelV3Error("parallel-v3 cache contract drifted")
    attribution = base._mapping(
        barrier.get("journal_writer_attribution"), "v3 writer attribution"
    )
    if attribution != {
        "scheduler_core_sha256": FROZEN_V2_SHA256,
        "runtime_cachefix_sha256": _v3_sha256(),
        "upload_protocol": "frozen_v2_clearml_sdk_plus_authoritative_confirm_v1",
    }:
        raise P1ParallelV3Error("parallel-v3 writer attribution drifted")
    source = base._mapping(barrier.get("source_pending_wal"), "v3 source WAL")
    base._exact_keys(
        source,
        {
            "candidate_journal_seal_sha256",
            "candidate_revision",
            "prior_journal_seal_sha256",
            "prior_revision",
            "transition_kind",
            "wal_document_type",
            "wal_seal_sha256",
        },
        "v3 source WAL",
    )
    if (
        source.get("wal_document_type") != v2.PENDING_JOURNAL_DOCUMENT_TYPE
        or source.get("wal_seal_sha256") != LIVE_INITIAL_WAL_SEAL_SHA256
        or source.get("prior_journal_seal_sha256")
        != LIVE_INITIAL_PRIOR_JOURNAL_SEAL_SHA256
        or source.get("candidate_journal_seal_sha256")
        != LIVE_INITIAL_CANDIDATE_JOURNAL_SEAL_SHA256
        or source.get("prior_revision") != 5
        or source.get("candidate_revision") != 6
        or source.get("transition_kind") not in {"journal_status", "task_row"}
        or type(source.get("prior_revision")) is not int
        or type(source.get("candidate_revision")) is not int
        or source["candidate_revision"] != source["prior_revision"] + 1
    ):
        raise P1ParallelV3Error("parallel-v3 source WAL drifted")
    for field in (
        "wal_seal_sha256",
        "prior_journal_seal_sha256",
        "candidate_journal_seal_sha256",
    ):
        base._sha256(source.get(field), f"parallel-v3 {field}")
    return barrier


def _read_installed_barrier(
    task_class: object,
    *,
    controller_id: str,
    amendment: Mapping[str, object],
) -> dict[str, object] | None:
    controller = _fresh_task(task_class, controller_id)
    if V3_BARRIER_ARTIFACT not in base._artifact_inventory(controller):
        return None
    _task, value, _descriptor, _raw = _authoritative_artifact(
        task_class,
        task_id=controller_id,
        artifact_name=V3_BARRIER_ARTIFACT,
    )
    barrier = _validate_v3_barrier(
        value,
        controller_id=controller_id,
        amendment=amendment,
    )
    return barrier


def _barrier_matches_wal(
    barrier: Mapping[str, object], wal: Mapping[str, object]
) -> None:
    source = base._mapping(barrier.get("source_pending_wal"), "v3 source WAL")
    prior = base._mapping(wal.get("prior_journal"), "pending prior journal")
    candidate = base._mapping(wal.get("candidate_journal"), "pending candidate journal")
    expected = {
        "wal_document_type": wal["document_type"],
        "wal_seal_sha256": wal["seal_sha256"],
        "prior_revision": prior["revision"],
        "prior_journal_seal_sha256": prior["seal_sha256"],
        "candidate_revision": candidate["revision"],
        "candidate_journal_seal_sha256": candidate["seal_sha256"],
        "transition_kind": _journal_transition_kind(prior, candidate),
    }
    if source != expected:
        raise P1ParallelV3Error("installed barrier and pending WAL differ")


def _wal_relation_to_barrier(
    barrier: Mapping[str, object], wal: Mapping[str, object]
) -> str:
    """Classify the immutable migration WAL or a later runtime WAL."""

    source = base._mapping(barrier.get("source_pending_wal"), "v3 source WAL")
    prior = base._mapping(wal.get("prior_journal"), "pending prior journal")
    candidate = base._mapping(wal.get("candidate_journal"), "pending candidate journal")
    if wal.get("seal_sha256") == source.get("wal_seal_sha256"):
        _barrier_matches_wal(barrier, wal)
        return "initial_migration"
    if (
        type(prior.get("revision")) is not int
        or type(candidate.get("revision")) is not int
        or prior["revision"] < source["candidate_revision"]
        or candidate["revision"] != prior["revision"] + 1
    ):
        raise P1ParallelV3Error("runtime WAL predates or conflicts with the v3 barrier")
    _journal_transition_kind(prior, candidate)
    return "post_barrier_runtime"


def _ensure_barrier_tag(task_class: object, controller_id: str) -> None:
    """Best-effort audit tag; the immutable artifact is the actual fence."""

    controller = _fresh_task(task_class, controller_id)
    if V3_CONTROLLER_TAG not in base._task_tags(controller):
        base._add_tags(controller, [V3_CONTROLLER_TAG])
    controller = _fresh_task(task_class, controller_id)
    if V3_CONTROLLER_TAG not in base._task_tags(controller):
        raise P1ParallelV3Error("parallel-v3 controller tag did not round-trip")


def _upload_mapping_confirmed(
    task_class: object,
    *,
    controller_id: str,
    artifact_name: str,
    value: Mapping[str, object],
) -> None:
    controller = _fresh_task(task_class, controller_id)
    caught: Exception | None = None
    try:
        base._upload_artifact(controller, artifact_name, value)
    except Exception as error:
        caught = error
    try:
        _fresh, observed, descriptor, raw = _authoritative_artifact(
            task_class,
            task_id=controller_id,
            artifact_name=artifact_name,
        )
    except Exception as read_error:
        if caught is not None:
            add_note = getattr(caught, "add_note", None)
            if callable(add_note):
                add_note(f"authoritative confirmation also failed: {read_error}")
            raise P1ParallelV3Error(
                f"{artifact_name} upload outcome is uncertain"
            ) from caught
        raise
    expected_raw = _sdk_json_bytes(value)
    if (
        observed != value
        or raw != expected_raw
        or descriptor["hash"] != hashlib.sha256(expected_raw).hexdigest()
        or descriptor["content_size"] != len(expected_raw)
    ):
        raise P1ParallelV3Error(
            f"{artifact_name} did not commit exact bytes"
        ) from caught


def _install_barrier(
    task_class: object,
    *,
    controller_id: str,
    amendment: Mapping[str, object],
    wal: Mapping[str, object],
) -> dict[str, object]:
    existing = _read_installed_barrier(
        task_class, controller_id=controller_id, amendment=amendment
    )
    if existing is not None:
        _wal_relation_to_barrier(existing, wal)
        _ensure_barrier_tag(task_class, controller_id)
        return existing
    barrier = _build_v3_barrier(
        controller_id=controller_id,
        amendment=amendment,
        wal=wal,
    )
    _upload_mapping_confirmed(
        task_class,
        controller_id=controller_id,
        artifact_name=V3_BARRIER_ARTIFACT,
        value=barrier,
    )
    _ensure_barrier_tag(task_class, controller_id)
    installed = _read_installed_barrier(
        task_class, controller_id=controller_id, amendment=amendment
    )
    if installed != barrier:
        raise P1ParallelV3Error("parallel-v3 barrier round-trip drifted")
    return installed


def _load_existing_execution(
    task_class: object, base_receipt: Mapping[str, object]
) -> tuple[object, dict[str, object]]:
    existing = base._mapping(
        base_receipt.get("existing_execution"), "base existing execution"
    )
    controller_id = base._clearml_id(existing.get("controller_id"), "controller")
    controller = _fresh_task(task_class, controller_id)
    plan, pinset = _verify_frozen_bindings()
    try:
        base._validate_controller(
            controller,
            plan=plan,
            pinset=pinset,
            execution_key=v2.SEALED_EXECUTION_KEY,
        )
    except base.P1ExecutorError as error:
        raise P1ParallelV3Error(str(error)) from error
    allowed = {
        base.PLAN_ARTIFACT,
        base.PINSET_ARTIFACT,
        base.JOURNAL_ARTIFACT,
        base.MANIFEST_ARTIFACT,
        v2.AMENDMENT_ARTIFACT,
        V3_BARRIER_ARTIFACT,
    }
    artifacts = base._artifact_inventory(controller)
    if not set(artifacts).issubset(allowed):
        raise P1ParallelV3Error(
            "controller artifact inventory contains unknown entries"
        )
    _validate_authoritative_immutable_bindings(
        task_class,
        controller_id=controller_id,
        plan=plan,
        pinset=pinset,
    )
    _fresh_controller, journal = _authoritative_remote_journal(
        task_class, controller_id
    )
    amendment = _validate_v2_amendment(task_class, controller_id, journal)
    barrier = _read_installed_barrier(
        task_class, controller_id=controller_id, amendment=amendment
    )
    if barrier is None:
        raise P1ParallelV3Error("parallel-v3 durable barrier is missing")
    source = base._mapping(barrier["source_pending_wal"], "v3 source WAL")
    if int(journal["revision"]) < int(source["candidate_revision"]):
        raise P1ParallelV3Error("execution journal predates the v3 recovery candidate")
    return controller, journal


def _journal_descriptor_state(
    task_class: object,
    *,
    controller_id: str,
    prior: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[str, object, dict[str, object], dict[str, object], bytes]:
    controller = _fresh_task(task_class, controller_id)
    descriptor_object = _backend_artifact_descriptor(
        task_class,
        task_id=controller_id,
        artifact_name=base.JOURNAL_ARTIFACT,
    )
    try:
        from clearml.binding.artifacts import Artifact
    except ImportError as error:
        raise P1ParallelV3Error(
            f"ClearML Artifact wrapper is unavailable: {error}"
        ) from None
    artifact = Artifact(descriptor_object)
    descriptor = _descriptor_value(artifact, base.JOURNAL_ARTIFACT)
    prior_raw = _sdk_json_bytes(prior)
    candidate_raw = _sdk_json_bytes(candidate)
    prior_hash = hashlib.sha256(prior_raw).hexdigest()
    candidate_hash = hashlib.sha256(candidate_raw).hexdigest()
    if descriptor["hash"] == prior_hash and descriptor["content_size"] == len(
        prior_raw
    ):
        descriptor_state = "prior"
    elif descriptor["hash"] == candidate_hash and descriptor["content_size"] == len(
        candidate_raw
    ):
        descriptor_state = "candidate"
    else:
        raise P1ParallelV3Error(
            "journal backend descriptor is neither prior nor candidate"
        )
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        raise P1ParallelV3Error("journal has no forced-download interface")
    path = getter(extract_archive=False, raise_on_error=True, force_download=True)
    raw = _read_regular_bytes(path, base.JOURNAL_ARTIFACT)
    if raw == prior_raw:
        raw_state = "prior"
    elif raw == candidate_raw:
        raw_state = "candidate"
    else:
        raise P1ParallelV3Error("journal forced bytes are neither prior nor candidate")
    if descriptor_state == "candidate" and raw_state != "candidate":
        raise P1ParallelV3Error(
            "candidate descriptor does not match forced journal bytes"
        )
    state = (
        descriptor_state
        if descriptor_state == raw_state
        else "prior_descriptor_candidate_file"
    )
    return (
        state,
        controller,
        descriptor,
        base._mapping(json.loads(raw.decode("utf-8")), "forced journal"),
        raw,
    )


def _recover_pending_candidate_once(
    task_class: object,
    *,
    controller_id: str,
    wal: Mapping[str, object],
) -> dict[str, object]:
    """Replay the exact sealed candidate at most once, then remove the WAL."""

    controller_id = base._clearml_id(controller_id, "controller")
    current_wal = _read_pending_wal(controller_id)
    if current_wal != wal:
        raise P1ParallelV3Error("pending WAL changed before exact recovery")
    prior = _validate_journal(
        base._mapping(wal["prior_journal"], "pending prior journal"),
        controller_id=controller_id,
    )
    candidate = _validate_journal(
        base._mapping(wal["candidate_journal"], "pending candidate journal"),
        controller_id=controller_id,
    )
    before_state, controller, before_descriptor, _before_value, _before_raw = (
        _journal_descriptor_state(
            task_class,
            controller_id=controller_id,
            prior=prior,
            candidate=candidate,
        )
    )
    upload_attempts = 0
    caught: Exception | None = None
    if before_state != "candidate":
        upload_attempts = 1
        try:
            # Keep the byte-pinned v2 SDK write path.  The cache fix is solely
            # the direct backend descriptor + forced-download confirmation.
            base._upload_artifact(controller, base.JOURNAL_ARTIFACT, candidate)
        except Exception as error:
            caught = error
    try:
        after_state, _fresh_controller, after_descriptor, after_value, after_raw = (
            _journal_descriptor_state(
                task_class,
                controller_id=controller_id,
                prior=prior,
                candidate=candidate,
            )
        )
    except Exception as read_error:
        if caught is not None:
            add_note = getattr(caught, "add_note", None)
            if callable(add_note):
                add_note(f"candidate confirmation also failed: {read_error}")
            raise P1ParallelV3Error(
                "exact candidate retry outcome is uncertain; WAL was retained"
            ) from caught
        raise
    expected_raw = _sdk_json_bytes(candidate)
    if (
        after_state != "candidate"
        or after_value != candidate
        or after_raw != expected_raw
        or after_descriptor["hash"] != hashlib.sha256(expected_raw).hexdigest()
        or after_descriptor["content_size"] != len(expected_raw)
    ):
        raise P1ParallelV3Error(
            "exact candidate did not commit authoritatively; WAL was retained"
        ) from caught
    if _read_pending_wal(controller_id) != wal:
        raise P1ParallelV3Error("pending WAL changed before deletion")
    v2._remove_pending_journal_receipt(wal)
    return {
        "before_state": before_state,
        "before_descriptor": before_descriptor,
        "after_descriptor": after_descriptor,
        "candidate_revision": candidate["revision"],
        "candidate_journal_seal_sha256": candidate["seal_sha256"],
        "upload_attempts": upload_attempts,
        "wal_removed": True,
    }


def _read_base_preflight(
    *,
    task_class: object | None = None,
    queue_reader: Callable[[object], Mapping[str, object]] | None = None,
    lease_reader: Callable[[str], Mapping[str, object]] | None = None,
    legacy_retirement_reader: Callable[
        [object, Mapping[str, object], Mapping[str, object], str], Mapping[str, object]
    ]
    | None = None,
) -> tuple[object, dict[str, object]]:
    return v2._read_base_preflight(
        task_class=task_class,
        queue_reader=queue_reader,
        lease_reader=lease_reader,
        legacy_retirement_reader=legacy_retirement_reader,
    )


def _controller_from_receipt(
    task_class: object, base_receipt: Mapping[str, object]
) -> tuple[object, str]:
    existing = base._mapping(
        base_receipt.get("existing_execution"), "base existing execution"
    )
    controller_id = base._clearml_id(existing.get("controller_id"), "controller")
    return _fresh_task(task_class, controller_id), controller_id


def preflight(
    *,
    task_class: object | None = None,
    queue_reader: Callable[[object], Mapping[str, object]] | None = None,
    lease_reader: Callable[[str], Mapping[str, object]] | None = None,
    legacy_retirement_reader: Callable[
        [object, Mapping[str, object], Mapping[str, object], str], Mapping[str, object]
    ]
    | None = None,
) -> dict[str, object]:
    """Read-only validation of the v3 migration or installed barrier."""

    plan, pinset = _verify_frozen_bindings()
    host = _validate_control_host()
    task_type, base_receipt = _read_base_preflight(
        task_class=task_class,
        queue_reader=queue_reader,
        lease_reader=lease_reader,
        legacy_retirement_reader=legacy_retirement_reader,
    )
    _controller, controller_id = _controller_from_receipt(task_type, base_receipt)
    immutable_bindings = _validate_authoritative_immutable_bindings(
        task_type,
        controller_id=controller_id,
        plan=plan,
        pinset=pinset,
    )
    wal = _read_pending_wal(controller_id)
    if wal is not None:
        prior = _validate_journal(
            base._mapping(wal["prior_journal"], "pending prior journal"),
            controller_id=controller_id,
        )
        candidate = _validate_journal(
            base._mapping(wal["candidate_journal"], "pending candidate journal"),
            controller_id=controller_id,
        )
        remote_state, _task, descriptor, _value, _raw = _journal_descriptor_state(
            task_type,
            controller_id=controller_id,
            prior=prior,
            candidate=candidate,
        )
        journal_for_amendment = prior if remote_state != "candidate" else candidate
    else:
        _task, journal_for_amendment = _authoritative_remote_journal(
            task_type, controller_id
        )
        remote_state = "current"
        descriptor = {}
    amendment = _validate_v2_amendment(task_type, controller_id, journal_for_amendment)
    barrier = _read_installed_barrier(
        task_type, controller_id=controller_id, amendment=amendment
    )
    if barrier is not None and wal is not None:
        _wal_relation_to_barrier(barrier, wal)
    if barrier is None and wal is None:
        raise P1ParallelV3Error(
            "parallel-v3 requires either the source pending WAL or an installed barrier"
        )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": V3_PREFLIGHT_DOCUMENT_TYPE,
        "status": "validated",
        "readonly": True,
        "remote_mutation_count": 0,
        "v2_supervisor_sha256": FROZEN_V2_SHA256,
        "v3_supervisor_sha256": _v3_sha256(),
        "plan_seal_sha256": v2.SEALED_PLAN_SHA256,
        "pinset_seal_sha256": v2.SEALED_PINSET_SHA256,
        "execution_key": v2.SEALED_EXECUTION_KEY,
        "controller_task_id": controller_id,
        "control_host": host,
        "immutable_bindings": immutable_bindings,
        "barrier_status": "installed" if barrier is not None else "not_installed",
        "barrier_seal_sha256": barrier["seal_sha256"] if barrier else None,
        "barrier_controller_tag_present": (
            V3_CONTROLLER_TAG in base._task_tags(_fresh_task(task_type, controller_id))
            if barrier is not None
            else False
        ),
        "pending_wal_status": "pending" if wal is not None else "absent",
        "pending_wal_seal_sha256": wal["seal_sha256"] if wal else None,
        "authoritative_journal_state": remote_state,
        "authoritative_journal_descriptor": descriptor,
        "base_preflight_seal_sha256": base_receipt["seal_sha256"],
        "queue": copy.deepcopy(base_receipt["queue"]),
        "global_mutex": copy.deepcopy(base_receipt["global_mutex"]),
    }
    receipt["seal_sha256"] = base._seal(receipt)
    return receipt


def _recover_and_install_barrier(
    *,
    task_class: object,
    api_client: object,
    queue_reader: Callable[[object], Mapping[str, object]] | None,
    legacy_retirement_reader: Callable[
        [object, Mapping[str, object], Mapping[str, object], str], Mapping[str, object]
    ]
    | None,
) -> dict[str, object]:
    def read_lease(key: str) -> Mapping[str, object]:
        return base._read_lease_snapshot(key, api_client=api_client)

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
            api_client=api_client,
        )

    with base._execution_lock():
        _read_base_preflight(
            task_class=task_class,
            queue_reader=queue_reader,
            lease_reader=read_lease,
            legacy_retirement_reader=read_retirement,
        )
        with base._server_execution_mutex(
            v2.SEALED_EXECUTION_KEY, api_client=api_client
        ):
            _task_type, base_receipt = _read_base_preflight(
                task_class=task_class,
                queue_reader=queue_reader,
                lease_reader=read_lease,
                legacy_retirement_reader=read_retirement,
            )
            _controller, controller_id = _controller_from_receipt(
                task_class, base_receipt
            )
            plan, pinset = _verify_frozen_bindings()
            _validate_authoritative_immutable_bindings(
                task_class,
                controller_id=controller_id,
                plan=plan,
                pinset=pinset,
            )
            wal = _read_pending_wal(controller_id)
            if wal is None:
                _task, journal = _authoritative_remote_journal(
                    task_class, controller_id
                )
                amendment = _validate_v2_amendment(task_class, controller_id, journal)
                barrier = _read_installed_barrier(
                    task_class, controller_id=controller_id, amendment=amendment
                )
                if barrier is None:
                    raise P1ParallelV3Error(
                        "source WAL is absent and the v3 barrier is not installed"
                    )
                return {
                    "controller_task_id": controller_id,
                    "barrier_seal_sha256": barrier["seal_sha256"],
                    "recovery": {
                        "candidate_revision": journal["revision"],
                        "upload_attempts": 0,
                        "wal_removed": False,
                        "status": "already_recovered",
                    },
                }
            _validate_journal(
                base._mapping(wal["prior_journal"], "pending prior journal"),
                controller_id=controller_id,
            )
            candidate = _validate_journal(
                base._mapping(wal["candidate_journal"], "pending candidate journal"),
                controller_id=controller_id,
            )
            # Fence the byte-pinned v2 process before the first forced journal
            # download.  The sealed WAL candidate is a valid monotonic current
            # journal for validating the already-installed v2 amendment.
            amendment = _validate_v2_amendment(task_class, controller_id, candidate)
            barrier = _install_barrier(
                task_class,
                controller_id=controller_id,
                amendment=amendment,
                wal=wal,
            )
            recovery = _recover_pending_candidate_once(
                task_class,
                controller_id=controller_id,
                wal=wal,
            )
            recovery["status"] = "recovered"
            return {
                "controller_task_id": controller_id,
                "barrier_seal_sha256": barrier["seal_sha256"],
                "recovery": recovery,
            }


@contextlib.contextmanager
def _authoritative_v2_runtime(*, task_class: object) -> Iterator[None]:
    """Patch every v2 mutable-journal entry point for this process only."""

    original_remote = v2._read_remote_journal
    original_load = base._load_journal
    original_existing = v2._load_existing_execution

    def authoritative_load(
        controller: object,
        *,
        plan: Mapping[str, object],
        execution_key: str,
        controller_id: str,
    ) -> dict[str, object]:
        return _authoritative_load_journal(
            controller,
            plan=plan,
            execution_key=execution_key,
            controller_id=controller_id,
            task_class=task_class,
        )

    v2._read_remote_journal = _authoritative_remote_journal
    base._load_journal = authoritative_load
    v2._load_existing_execution = _load_existing_execution
    try:
        yield
    finally:
        v2._read_remote_journal = original_remote
        base._load_journal = original_load
        v2._load_existing_execution = original_existing


def execute_plan(
    *,
    authorization_token: str,
    poll_seconds: float,
    timeout_hours: float,
    task_class: object | None = None,
    queue_reader: Callable[[object], Mapping[str, object]] | None = None,
    api_client: object | None = None,
    legacy_retirement_reader: Callable[
        [object, Mapping[str, object], Mapping[str, object], str], Mapping[str, object]
    ]
    | None = None,
) -> dict[str, object]:
    """Install the v3 barrier, recover the exact WAL candidate, then resume v2."""

    if authorization_token != V3_EXECUTION_TOKEN:
        raise P1ParallelV3Error("parallel-v3 execution token mismatch")
    if not math.isfinite(poll_seconds) or not 1.0 <= poll_seconds <= 60.0:
        raise P1ParallelV3Error("poll-seconds must be finite and within [1, 60]")
    if not math.isfinite(timeout_hours) or not 1.0 <= timeout_hours <= 72.0:
        raise P1ParallelV3Error("timeout-hours must be finite and within [1, 72]")
    _verify_frozen_bindings()
    _validate_control_host()
    task_type = task_class or base._load_clearml()
    client = api_client or base._api_client()
    migration = _recover_and_install_barrier(
        task_class=task_type,
        api_client=client,
        queue_reader=queue_reader,
        legacy_retirement_reader=legacy_retirement_reader,
    )
    with _authoritative_v2_runtime(task_class=task_type):
        manifest = v2.execute_plan(
            authorization_token=v2.EXECUTION_TOKEN,
            poll_seconds=poll_seconds,
            timeout_hours=timeout_hours,
            task_class=task_type,
            queue_reader=queue_reader,
            api_client=client,
            legacy_retirement_reader=legacy_retirement_reader,
        )
    result: dict[str, object] = {
        "schema_version": 1,
        "document_type": V3_EXECUTION_RECEIPT_DOCUMENT_TYPE,
        "status": "completed",
        "v3_supervisor_sha256": _v3_sha256(),
        "v2_supervisor_sha256": FROZEN_V2_SHA256,
        "barrier_seal_sha256": migration["barrier_seal_sha256"],
        "recovery": migration["recovery"],
        "execution_manifest": copy.deepcopy(manifest),
    }
    result["seal_sha256"] = base._seal(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-token", default="")
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
        if args.execution_token:
            raise P1ParallelV3Error("--execution-token is invalid with --preflight")
        _print_json(preflight(), pretty=args.pretty)
        return 0
    if not args.execute:
        if args.execution_token:
            raise P1ParallelV3Error("--execution-token is invalid without --execute")
        plan, pinset = _verify_frozen_bindings()
        value: dict[str, object] = {
            "schema_version": 1,
            "document_type": V3_DRY_RUN_DOCUMENT_TYPE,
            "default_mode": "dry_run",
            "remote_mutation_authorized": False,
            "v2_supervisor_sha256": FROZEN_V2_SHA256,
            "v3_supervisor_sha256": _v3_sha256(),
            "plan_seal_sha256": plan["seal_sha256"],
            "pinset_seal_sha256": pinset["seal_sha256"],
            "execution_key": v2.SEALED_EXECUTION_KEY,
            "barrier_artifact": V3_BARRIER_ARTIFACT,
            "execution_token_sha256": hashlib.sha256(
                V3_EXECUTION_TOKEN.encode("utf-8")
            ).hexdigest(),
        }
        value["seal_sha256"] = base._seal(value)
        _print_json(value, pretty=args.pretty)
        return 0
    if args.execution_token != V3_EXECUTION_TOKEN:
        raise P1ParallelV3Error(
            "--execute requires the exact --execution-token acknowledgement"
        )
    manifest = execute_plan(
        authorization_token=args.execution_token,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    _print_json(manifest, pretty=args.pretty)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except P1RecoverableTimeout as error:
        print(f"P1-PARALLEL-V3-RECOVERABLE-TIMEOUT: {error}", file=sys.stderr)
        raise SystemExit(3) from None
    except P1ExecutorError as error:
        print(f"P1-PARALLEL-V3-ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
