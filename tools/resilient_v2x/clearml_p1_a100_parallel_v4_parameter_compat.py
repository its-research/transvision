#!/usr/bin/env python3
"""Resume the sealed P1 campaign with exact ClearML runtime parameters.

This is a thin successor to the byte-pinned parallel-v3 cache-fix supervisor.
It preserves the scientific plan, pinset, execution key, controller, journal,
child identities, scheduler, and authoritative journal protocol.  The only
behavioral amendment is confined to completed-result validation: ClearML's
argparse binding materializes unused optional arguments in the ``Args``
section, so v4 validates the complete 28-key runtime schema instead of the
smaller pre-enqueue schema.

Default mode is local dry-run.  ``--preflight`` is remote read-only.  Remote
mutation requires ``--execute`` and the exact v4 acknowledgement token.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path


V3_FILENAME = "clearml_p1_a100_parallel_v3_cachefix.py"
FROZEN_V3_SHA256 = "baf914454f0b4152426c3dd7d3b36272db122e6734bde955ec7cd7d7c1755eea"
V4_BARRIER_ARTIFACT = "p1_a100_parallel_v4_parameter_compat_barrier"
V4_BARRIER_DOCUMENT_TYPE = "resilient_v2x_p1_a100_parallel_v4_parameter_compat_barrier"
V4_PREFLIGHT_DOCUMENT_TYPE = (
    "resilient_v2x_p1_a100_parallel_v4_parameter_compat_preflight"
)
V4_DRY_RUN_DOCUMENT_TYPE = "resilient_v2x_p1_a100_parallel_v4_parameter_compat_dry_run"
V4_EXECUTION_RECEIPT_DOCUMENT_TYPE = (
    "resilient_v2x_p1_a100_parallel_v4_parameter_compat_execution_receipt"
)
V4_EXECUTION_TOKEN = "RECOVER_AND_EXECUTE_P1_A100_PARALLEL_V4_PARAMETER_COMPAT"

LIVE_SOURCE_JOURNAL_REVISION = 10
LIVE_SOURCE_JOURNAL_SEAL_SHA256 = (
    "5121b1e5fccfbcbf4cdf6ca1abe8f4e996e0f4498881b822f707597494a7c616"
)
LIVE_V2_AMENDMENT_SEAL_SHA256 = (
    "bff8d1f8ece8b37ac391b84c6a7a092604a2fb01d36f8b9ede551d6843228dfa"
)
LIVE_V3_BARRIER_SEAL_SHA256 = (
    "13ef2f0045f4dee14688681678206899d73423f9929f65cd1ea8b0179a4e56ac"
)
LIVE_SOURCE_TASK_BINDINGS = {
    "train-r01-s01-bevfusion": {
        "task_id": "87377d1f37d34a2eb77e22a2c606bf9b",
        "state": "active",
        "server_status": "in_progress",
    },
    "train-r01-s01-resilient_v2x": {
        "task_id": "2c491c02491c45b4ac5deae3ff0bd44e",
        "state": "active",
        "server_status": "in_progress",
    },
}

TRAINING_RUNTIME_EXTRA_PARAMETERS = {
    "Args/controlled_baseline": "",
    "Args/controlled_baseline_checkpoint_sha256": "",
    "Args/controlled_baseline_model_id": "",
    "Args/controlled_baseline_task_id": "",
    "Args/student_checkpoint": "",
    "Args/student_checkpoint_sha256": "",
    "Args/student_model_id": "",
    "Args/student_task_id": "",
    "Args/teacher_checkpoint": "",
}
EVALUATION_RUNTIME_EXTRA_PARAMETERS = {
    "Args/allow_failed_teacher_task": False,
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


def _load_v3() -> object:
    path = Path(__file__).resolve().with_name(V3_FILENAME)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != FROZEN_V3_SHA256:
        raise RuntimeError(
            "pinned parallel-v3 supervisor SHA-256 mismatch: "
            f"expected {FROZEN_V3_SHA256}, observed {observed}"
        )
    spec = importlib.util.spec_from_file_location(
        "_resilientv2x_p1_a100_parallel_v3_pinned", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the pinned parallel-v3 supervisor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = _load_v3()
v2 = v3.v2
base = v3.base
P1ExecutorError = base.P1ExecutorError
P1RecoverableTimeout = base.P1RecoverableTimeout


class P1ParallelV4Error(P1ExecutorError):
    """The v4 parameter amendment or one of its bindings is invalid."""


def _v4_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _verify_frozen_bindings() -> tuple[dict[str, object], dict[str, object]]:
    path = Path(__file__).resolve().with_name(V3_FILENAME)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_V3_SHA256:
        raise P1ParallelV4Error("frozen parallel-v3 supervisor SHA-256 drifted")
    if v3._v3_sha256() != FROZEN_V3_SHA256:
        raise P1ParallelV4Error("parallel-v3 self fingerprint drifted")
    try:
        return v3._verify_frozen_bindings()
    except base.P1ExecutorError as error:
        raise P1ParallelV4Error(str(error)) from error


def _training_expected_keys() -> frozenset[str]:
    return frozenset(
        base._training_parameters(base.SUBJECTS[0], base.TRAINING_SEEDS[0])
    )


def _evaluation_expected_keys() -> frozenset[str]:
    return frozenset(
        base._evaluation_parameters(
            base.SUBJECTS[0],
            base.TRAINING_SEEDS[0],
            training_task_id="1" * 32,
            model_id="2" * 32,
            checkpoint_sha256="3" * 64,
        )
    )


TRAINING_EXPECTED_KEYS = _training_expected_keys()
EVALUATION_EXPECTED_KEYS = _evaluation_expected_keys()
FULL_RUNTIME_PARAMETER_KEYS = frozenset(
    set(TRAINING_EXPECTED_KEYS) | set(TRAINING_RUNTIME_EXTRA_PARAMETERS)
)


def _runtime_extra_parameters(phase: str) -> dict[str, object]:
    if phase == "training":
        return dict(TRAINING_RUNTIME_EXTRA_PARAMETERS)
    if phase == "evaluation":
        return dict(EVALUATION_RUNTIME_EXTRA_PARAMETERS)
    raise P1ParallelV4Error(f"unknown completion parameter phase {phase!r}")


def _logical_expected_keys(phase: str) -> frozenset[str]:
    if phase == "training":
        return TRAINING_EXPECTED_KEYS
    if phase == "evaluation":
        return EVALUATION_EXPECTED_KEYS
    raise P1ParallelV4Error(f"unknown completion parameter phase {phase!r}")


def _completion_runtime_expected(
    expected: Mapping[str, object], *, phase: str
) -> dict[str, object]:
    logical = base._mapping(expected, f"parallel-v4 {phase} logical parameters")
    expected_keys = _logical_expected_keys(phase)
    if set(logical) != set(expected_keys):
        raise P1ParallelV4Error(f"parallel-v4 {phase} logical parameter schema drifted")
    extras = _runtime_extra_parameters(phase)
    if set(logical) & set(extras):
        raise P1ParallelV4Error(
            f"parallel-v4 {phase} runtime extras overlap logical parameters"
        )
    result = dict(logical)
    result.update(extras)
    if set(result) != set(FULL_RUNTIME_PARAMETER_KEYS):
        raise P1ParallelV4Error(
            f"parallel-v4 {phase} full runtime parameter schema drifted"
        )
    return result


def _completion_parameter_contract() -> dict[str, object]:
    training_full = set(TRAINING_EXPECTED_KEYS) | set(TRAINING_RUNTIME_EXTRA_PARAMETERS)
    evaluation_full = set(EVALUATION_EXPECTED_KEYS) | set(
        EVALUATION_RUNTIME_EXTRA_PARAMETERS
    )
    if training_full != evaluation_full or training_full != set(
        FULL_RUNTIME_PARAMETER_KEYS
    ):
        raise P1ParallelV4Error("training and evaluation runtime schemas differ")
    return {
        "clearml_parameter_reader": "get_parameters(backwards_compatibility=False)",
        "normalization_function": "frozen_base._normalize_parameter",
        "validator_scope": ["training_result", "evaluation_result"],
        "created_task_validation_unchanged": True,
        "controller_validation_unchanged": True,
        "unknown_keys_forbidden": True,
        "missing_keys_forbidden": True,
        "normalized_value_match_required": True,
        "training": {
            "logical_expected_keys": sorted(TRAINING_EXPECTED_KEYS),
            "runtime_extra_parameters": copy.deepcopy(
                TRAINING_RUNTIME_EXTRA_PARAMETERS
            ),
            "full_runtime_keys": sorted(FULL_RUNTIME_PARAMETER_KEYS),
        },
        "evaluation": {
            "logical_expected_keys": sorted(EVALUATION_EXPECTED_KEYS),
            "runtime_extra_parameters": copy.deepcopy(
                EVALUATION_RUNTIME_EXTRA_PARAMETERS
            ),
            "full_runtime_keys": sorted(FULL_RUNTIME_PARAMETER_KEYS),
        },
    }


@contextlib.contextmanager
def _completion_parameter_guard(phase: str) -> Iterator[None]:
    """Expand expected parameters only while one result validator is running."""

    original = base._require_parameters
    call_count = 0

    def require_runtime_parameters(
        task: object, expected: Mapping[str, object], context: str
    ) -> None:
        nonlocal call_count
        call_count += 1
        if call_count != 1:
            raise P1ParallelV4Error(
                f"parallel-v4 {phase} validator checked parameters more than once"
            )
        original(
            task,
            _completion_runtime_expected(expected, phase=phase),
            context,
        )

    base._require_parameters = require_runtime_parameters
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        base._require_parameters = original
        if not failed and call_count != 1:
            raise P1ParallelV4Error(
                f"parallel-v4 {phase} validator did not check parameters exactly once"
            )


@contextlib.contextmanager
def _completion_validation_runtime() -> Iterator[None]:
    """Patch only the two completion validators and restore them on every exit."""

    original_training = base._validate_training_result
    original_evaluation = base._validate_evaluation_result

    def validate_training(*args: object, **kwargs: object) -> object:
        with _completion_parameter_guard("training"):
            return original_training(*args, **kwargs)

    def validate_evaluation(*args: object, **kwargs: object) -> object:
        with _completion_parameter_guard("evaluation"):
            return original_evaluation(*args, **kwargs)

    base._validate_training_result = validate_training
    base._validate_evaluation_result = validate_evaluation
    try:
        yield
    finally:
        base._validate_training_result = original_training
        base._validate_evaluation_result = original_evaluation


def _source_journal_binding(journal: Mapping[str, object]) -> dict[str, object]:
    value = base._mapping(journal, "parallel-v4 source journal")
    if (
        value.get("revision") != LIVE_SOURCE_JOURNAL_REVISION
        or value.get("seal_sha256") != LIVE_SOURCE_JOURNAL_SEAL_SHA256
        or value.get("status") != "running"
    ):
        raise P1ParallelV4Error("parallel-v4 live source journal drifted")
    rows = value.get("tasks")
    if not isinstance(rows, list):
        raise P1ParallelV4Error("parallel-v4 source journal task list is invalid")
    by_key = {
        str(
            base._mapping(row, "parallel-v4 source row").get("task_key")
        ): base._mapping(row, "parallel-v4 source row")
        for row in rows
    }
    if set(LIVE_SOURCE_TASK_BINDINGS) - set(by_key):
        raise P1ParallelV4Error("parallel-v4 source task binding is missing")
    observed: dict[str, dict[str, object]] = {}
    for task_key, expected in LIVE_SOURCE_TASK_BINDINGS.items():
        row = by_key[task_key]
        binding = {
            "task_id": row.get("task_id"),
            "state": row.get("state"),
            "server_status": row.get("server_status"),
        }
        if binding != expected:
            raise P1ParallelV4Error(
                f"parallel-v4 source task {task_key} binding drifted"
            )
        observed[task_key] = binding
    for task_key, row in by_key.items():
        if task_key in LIVE_SOURCE_TASK_BINDINGS:
            continue
        if (
            row.get("task_id") is not None
            or row.get("state") != "absent"
            or row.get("server_status") != "absent"
            or row.get("result") is not None
        ):
            raise P1ParallelV4Error(
                f"parallel-v4 unexpected live source task {task_key}"
            )
    return {
        "revision": LIVE_SOURCE_JOURNAL_REVISION,
        "seal_sha256": LIVE_SOURCE_JOURNAL_SEAL_SHA256,
        "status": "running",
        "task_bindings": copy.deepcopy(observed),
    }


def _expected_source_journal_binding() -> dict[str, object]:
    return {
        "revision": LIVE_SOURCE_JOURNAL_REVISION,
        "seal_sha256": LIVE_SOURCE_JOURNAL_SEAL_SHA256,
        "status": "running",
        "task_bindings": copy.deepcopy(LIVE_SOURCE_TASK_BINDINGS),
    }


def _build_v4_barrier(
    *,
    controller_id: str,
    amendment: Mapping[str, object],
    v3_barrier: Mapping[str, object],
    journal: Mapping[str, object],
) -> dict[str, object]:
    plan, pinset = _verify_frozen_bindings()
    controller_id = base._clearml_id(controller_id, "controller")
    if amendment.get("seal_sha256") != LIVE_V2_AMENDMENT_SEAL_SHA256:
        raise P1ParallelV4Error("parallel-v4 live v2 amendment drifted")
    if v3_barrier.get("seal_sha256") != LIVE_V3_BARRIER_SEAL_SHA256:
        raise P1ParallelV4Error("parallel-v4 live v3 barrier drifted")
    barrier: dict[str, object] = {
        "schema_version": 1,
        "document_type": V4_BARRIER_DOCUMENT_TYPE,
        "status": "active",
        "reason": "clearml_argparse_materializes_unused_optional_arguments",
        "v4_supervisor_sha256": _v4_sha256(),
        "v3_supervisor_sha256": FROZEN_V3_SHA256,
        "v2_supervisor_sha256": v3.FROZEN_V2_SHA256,
        "base_executor_sha256": v2.FROZEN_BASE_SHA256,
        "plan_seal_sha256": plan["seal_sha256"],
        "pinset_seal_sha256": pinset["seal_sha256"],
        "execution_key": v2.SEALED_EXECUTION_KEY,
        "controller_task_id": controller_id,
        "v2_amendment_artifact": v2.AMENDMENT_ARTIFACT,
        "v2_amendment_seal_sha256": amendment["seal_sha256"],
        "v3_barrier_artifact": v3.V3_BARRIER_ARTIFACT,
        "v3_barrier_seal_sha256": v3_barrier["seal_sha256"],
        "source_journal": _source_journal_binding(journal),
        "completion_parameter_contract": _completion_parameter_contract(),
    }
    barrier["seal_sha256"] = base._seal(barrier)
    return _validate_v4_barrier(
        barrier,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
    )


def _validate_v4_barrier(
    value: Mapping[str, object],
    *,
    controller_id: str,
    amendment: Mapping[str, object],
    v3_barrier: Mapping[str, object],
) -> dict[str, object]:
    barrier = base._mapping(value, "parallel-v4 parameter-compat barrier")
    base._exact_keys(
        barrier,
        {
            "base_executor_sha256",
            "completion_parameter_contract",
            "controller_task_id",
            "document_type",
            "execution_key",
            "pinset_seal_sha256",
            "plan_seal_sha256",
            "reason",
            "schema_version",
            "seal_sha256",
            "source_journal",
            "status",
            "v2_amendment_artifact",
            "v2_amendment_seal_sha256",
            "v2_supervisor_sha256",
            "v3_barrier_artifact",
            "v3_barrier_seal_sha256",
            "v3_supervisor_sha256",
            "v4_supervisor_sha256",
        },
        "parallel-v4 parameter-compat barrier",
    )
    if barrier.get("seal_sha256") != base._seal(barrier):
        raise P1ParallelV4Error("parallel-v4 barrier seal mismatch")
    plan, pinset = _verify_frozen_bindings()
    expected = {
        "schema_version": 1,
        "document_type": V4_BARRIER_DOCUMENT_TYPE,
        "status": "active",
        "reason": "clearml_argparse_materializes_unused_optional_arguments",
        "v4_supervisor_sha256": _v4_sha256(),
        "v3_supervisor_sha256": FROZEN_V3_SHA256,
        "v2_supervisor_sha256": v3.FROZEN_V2_SHA256,
        "base_executor_sha256": v2.FROZEN_BASE_SHA256,
        "plan_seal_sha256": plan["seal_sha256"],
        "pinset_seal_sha256": pinset["seal_sha256"],
        "execution_key": v2.SEALED_EXECUTION_KEY,
        "controller_task_id": base._clearml_id(controller_id, "controller"),
        "v2_amendment_artifact": v2.AMENDMENT_ARTIFACT,
        "v2_amendment_seal_sha256": amendment.get("seal_sha256"),
        "v3_barrier_artifact": v3.V3_BARRIER_ARTIFACT,
        "v3_barrier_seal_sha256": v3_barrier.get("seal_sha256"),
        "source_journal": _expected_source_journal_binding(),
        "completion_parameter_contract": _completion_parameter_contract(),
    }
    for field, expected_value in expected.items():
        if barrier.get(field) != expected_value:
            raise P1ParallelV4Error(f"parallel-v4 barrier {field} drifted")
    if amendment.get("seal_sha256") != LIVE_V2_AMENDMENT_SEAL_SHA256:
        raise P1ParallelV4Error("parallel-v4 live v2 amendment drifted")
    if v3_barrier.get("seal_sha256") != LIVE_V3_BARRIER_SEAL_SHA256:
        raise P1ParallelV4Error("parallel-v4 live v3 barrier drifted")
    return barrier


def _read_installed_v4_barrier(
    task_class: object,
    *,
    controller_id: str,
    amendment: Mapping[str, object],
    v3_barrier: Mapping[str, object],
) -> dict[str, object] | None:
    controller = v3._fresh_task(task_class, controller_id)
    if V4_BARRIER_ARTIFACT not in base._artifact_inventory(controller):
        return None
    _task, value, _descriptor, _raw = v3._authoritative_artifact(
        task_class,
        task_id=controller_id,
        artifact_name=V4_BARRIER_ARTIFACT,
    )
    return _validate_v4_barrier(
        value,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
    )


def _install_v4_barrier(
    task_class: object,
    *,
    controller_id: str,
    amendment: Mapping[str, object],
    v3_barrier: Mapping[str, object],
    journal: Mapping[str, object],
) -> dict[str, object]:
    existing = _read_installed_v4_barrier(
        task_class,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
    )
    if existing is not None:
        return existing
    if v3._read_pending_wal(controller_id) is not None:
        raise P1ParallelV4Error(
            "initial parallel-v4 barrier installation requires an absent pending WAL"
        )
    barrier = _build_v4_barrier(
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
        journal=journal,
    )
    v3._upload_mapping_confirmed(
        task_class,
        controller_id=controller_id,
        artifact_name=V4_BARRIER_ARTIFACT,
        value=barrier,
    )
    installed = _read_installed_v4_barrier(
        task_class,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
    )
    if installed != barrier:
        raise P1ParallelV4Error("parallel-v4 barrier round-trip drifted")
    return installed


def _validate_bound_barriers(
    task_class: object,
    *,
    controller_id: str,
    journal: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    amendment = v3._validate_v2_amendment(task_class, controller_id, journal)
    v3_barrier = v3._read_installed_barrier(
        task_class,
        controller_id=controller_id,
        amendment=amendment,
    )
    if v3_barrier is None:
        raise P1ParallelV4Error("parallel-v3 durable barrier is missing")
    v4_barrier = _read_installed_v4_barrier(
        task_class,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
    )
    if v4_barrier is None:
        raise P1ParallelV4Error("parallel-v4 durable barrier is missing")
    return amendment, v3_barrier, v4_barrier


def _load_existing_execution(
    task_class: object, base_receipt: Mapping[str, object]
) -> tuple[object, dict[str, object]]:
    existing = base._mapping(
        base_receipt.get("existing_execution"), "base existing execution"
    )
    controller_id = base._clearml_id(existing.get("controller_id"), "controller")
    controller = v3._fresh_task(task_class, controller_id)
    plan, pinset = _verify_frozen_bindings()
    try:
        base._validate_controller(
            controller,
            plan=plan,
            pinset=pinset,
            execution_key=v2.SEALED_EXECUTION_KEY,
        )
    except base.P1ExecutorError as error:
        raise P1ParallelV4Error(str(error)) from error
    allowed = {
        base.PLAN_ARTIFACT,
        base.PINSET_ARTIFACT,
        base.JOURNAL_ARTIFACT,
        base.MANIFEST_ARTIFACT,
        v2.AMENDMENT_ARTIFACT,
        v3.V3_BARRIER_ARTIFACT,
        V4_BARRIER_ARTIFACT,
    }
    artifacts = base._artifact_inventory(controller)
    if not set(artifacts).issubset(allowed):
        raise P1ParallelV4Error(
            "controller artifact inventory contains unknown entries"
        )
    v3._validate_authoritative_immutable_bindings(
        task_class,
        controller_id=controller_id,
        plan=plan,
        pinset=pinset,
    )
    fresh_controller, journal = v3._authoritative_remote_journal(
        task_class, controller_id
    )
    _validate_bound_barriers(
        task_class,
        controller_id=controller_id,
        journal=journal,
    )
    return fresh_controller, journal


def _validated_authoritative_load_journal(
    original: Callable[..., dict[str, object]],
    controller: object,
    *,
    plan: Mapping[str, object],
    execution_key: str,
    controller_id: str,
    task_class: object | None = None,
) -> dict[str, object]:
    journal = original(
        controller,
        plan=plan,
        execution_key=execution_key,
        controller_id=controller_id,
        task_class=task_class,
    )
    task_type = task_class or type(controller)
    _validate_bound_barriers(
        task_type,
        controller_id=controller_id,
        journal=journal,
    )
    return journal


def _validated_authoritative_remote_journal(
    original: Callable[..., tuple[object, dict[str, object]]],
    task_class: object,
    controller_id: str,
) -> tuple[object, dict[str, object]]:
    controller, journal = original(task_class, controller_id)
    _validate_bound_barriers(
        task_class,
        controller_id=controller_id,
        journal=journal,
    )
    return controller, journal


@contextlib.contextmanager
def _v4_runtime() -> Iterator[None]:
    """Layer v4 validation over v3 without changing any pinned source file."""

    original_existing = v3._load_existing_execution
    original_authoritative_load = v3._authoritative_load_journal
    original_authoritative_remote = v3._authoritative_remote_journal

    def authoritative_load(
        controller: object,
        *,
        plan: Mapping[str, object],
        execution_key: str,
        controller_id: str,
        task_class: object | None = None,
    ) -> dict[str, object]:
        return _validated_authoritative_load_journal(
            original_authoritative_load,
            controller,
            plan=plan,
            execution_key=execution_key,
            controller_id=controller_id,
            task_class=task_class,
        )

    def authoritative_remote(
        task_class: object, controller_id: str
    ) -> tuple[object, dict[str, object]]:
        return _validated_authoritative_remote_journal(
            original_authoritative_remote,
            task_class,
            controller_id,
        )

    v3._load_existing_execution = _load_existing_execution
    v3._authoritative_load_journal = authoritative_load
    v3._authoritative_remote_journal = authoritative_remote
    try:
        with _completion_validation_runtime():
            yield
    finally:
        v3._load_existing_execution = original_existing
        v3._authoritative_load_journal = original_authoritative_load
        v3._authoritative_remote_journal = original_authoritative_remote


def _read_current_bound_execution(
    task_class: object, base_receipt: Mapping[str, object]
) -> tuple[
    str,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    _controller, controller_id = v3._controller_from_receipt(task_class, base_receipt)
    plan, pinset = _verify_frozen_bindings()
    v3._validate_authoritative_immutable_bindings(
        task_class,
        controller_id=controller_id,
        plan=plan,
        pinset=pinset,
    )
    _fresh, journal = v3._authoritative_remote_journal(task_class, controller_id)
    amendment = v3._validate_v2_amendment(task_class, controller_id, journal)
    v3_barrier = v3._read_installed_barrier(
        task_class,
        controller_id=controller_id,
        amendment=amendment,
    )
    if v3_barrier is None:
        raise P1ParallelV4Error("parallel-v3 durable barrier is missing")
    wal = v3._read_pending_wal(controller_id)
    if wal is not None:
        v3._wal_relation_to_barrier(v3_barrier, wal)
    return controller_id, journal, amendment, v3_barrier


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
    """Read-only validation of v3 plus the optional installed v4 barrier."""

    _verify_frozen_bindings()
    task_type = task_class or base._load_clearml()
    v3_receipt = v3.preflight(
        task_class=task_type,
        queue_reader=queue_reader,
        lease_reader=lease_reader,
        legacy_retirement_reader=legacy_retirement_reader,
    )
    _task_type, base_receipt = v3._read_base_preflight(
        task_class=task_type,
        queue_reader=queue_reader,
        lease_reader=lease_reader,
        legacy_retirement_reader=legacy_retirement_reader,
    )
    controller_id, journal, amendment, v3_barrier = _read_current_bound_execution(
        task_type, base_receipt
    )
    v4_barrier = _read_installed_v4_barrier(
        task_type,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
    )
    if v4_barrier is None:
        if v3._read_pending_wal(controller_id) is not None:
            raise P1ParallelV4Error(
                "initial parallel-v4 preflight requires an absent pending WAL"
            )
        source_binding = _source_journal_binding(journal)
    else:
        source_binding = copy.deepcopy(v4_barrier["source_journal"])
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": V4_PREFLIGHT_DOCUMENT_TYPE,
        "status": "validated",
        "readonly": True,
        "remote_mutation_count": 0,
        "v4_supervisor_sha256": _v4_sha256(),
        "v3_supervisor_sha256": FROZEN_V3_SHA256,
        "v2_supervisor_sha256": v3.FROZEN_V2_SHA256,
        "base_executor_sha256": v2.FROZEN_BASE_SHA256,
        "plan_seal_sha256": v2.SEALED_PLAN_SHA256,
        "pinset_seal_sha256": v2.SEALED_PINSET_SHA256,
        "execution_key": v2.SEALED_EXECUTION_KEY,
        "controller_task_id": controller_id,
        "journal_revision": journal["revision"],
        "journal_seal_sha256": journal["seal_sha256"],
        "source_binding": source_binding,
        "v3_barrier_seal_sha256": v3_barrier["seal_sha256"],
        "v4_barrier_status": "installed" if v4_barrier is not None else "not_installed",
        "v4_barrier_seal_sha256": (
            v4_barrier["seal_sha256"] if v4_barrier is not None else None
        ),
        "completion_parameter_contract": _completion_parameter_contract(),
        "v3_preflight_seal_sha256": v3_receipt["seal_sha256"],
        "queue": copy.deepcopy(v3_receipt["queue"]),
        "global_mutex": copy.deepcopy(v3_receipt["global_mutex"]),
    }
    receipt["seal_sha256"] = base._seal(receipt)
    return receipt


def _install_v4_barrier_under_mutex(
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
        v3._read_base_preflight(
            task_class=task_class,
            queue_reader=queue_reader,
            lease_reader=read_lease,
            legacy_retirement_reader=read_retirement,
        )
        with base._server_execution_mutex(
            v2.SEALED_EXECUTION_KEY, api_client=api_client
        ):
            _task_type, base_receipt = v3._read_base_preflight(
                task_class=task_class,
                queue_reader=queue_reader,
                lease_reader=read_lease,
                legacy_retirement_reader=read_retirement,
            )
            controller_id, journal, amendment, v3_barrier = (
                _read_current_bound_execution(task_class, base_receipt)
            )
            barrier = _install_v4_barrier(
                task_class,
                controller_id=controller_id,
                amendment=amendment,
                v3_barrier=v3_barrier,
                journal=journal,
            )
            return {
                "controller_task_id": controller_id,
                "source_journal_revision": journal["revision"],
                "source_journal_seal_sha256": journal["seal_sha256"],
                "v3_barrier_seal_sha256": v3_barrier["seal_sha256"],
                "v4_barrier_seal_sha256": barrier["seal_sha256"],
            }


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
    """Install the v4 fence, then run the unchanged v3/v2 scheduler stack."""

    if authorization_token != V4_EXECUTION_TOKEN:
        raise P1ParallelV4Error("parallel-v4 execution token mismatch")
    if not math.isfinite(poll_seconds) or not 1.0 <= poll_seconds <= 60.0:
        raise P1ParallelV4Error("poll-seconds must be finite and within [1, 60]")
    if not math.isfinite(timeout_hours) or not 1.0 <= timeout_hours <= 72.0:
        raise P1ParallelV4Error("timeout-hours must be finite and within [1, 72]")
    _verify_frozen_bindings()
    try:
        v3._validate_control_host()
    except base.P1ExecutorError as error:
        raise P1ParallelV4Error(str(error)) from error
    task_type = task_class or base._load_clearml()
    client = api_client or base._api_client()
    migration = _install_v4_barrier_under_mutex(
        task_class=task_type,
        api_client=client,
        queue_reader=queue_reader,
        legacy_retirement_reader=legacy_retirement_reader,
    )
    with _v4_runtime():
        v3_receipt = v3.execute_plan(
            authorization_token=v3.V3_EXECUTION_TOKEN,
            poll_seconds=poll_seconds,
            timeout_hours=timeout_hours,
            task_class=task_type,
            queue_reader=queue_reader,
            api_client=client,
            legacy_retirement_reader=legacy_retirement_reader,
        )
    result: dict[str, object] = {
        "schema_version": 1,
        "document_type": V4_EXECUTION_RECEIPT_DOCUMENT_TYPE,
        "status": "completed",
        "v4_supervisor_sha256": _v4_sha256(),
        "v3_supervisor_sha256": FROZEN_V3_SHA256,
        "v2_supervisor_sha256": v3.FROZEN_V2_SHA256,
        "v4_barrier_seal_sha256": migration["v4_barrier_seal_sha256"],
        "migration": copy.deepcopy(migration),
        "v3_execution_receipt": copy.deepcopy(v3_receipt),
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
            raise P1ParallelV4Error("--execution-token is invalid with --preflight")
        _print_json(preflight(), pretty=args.pretty)
        return 0
    if not args.execute:
        if args.execution_token:
            raise P1ParallelV4Error("--execution-token is invalid without --execute")
        plan, pinset = _verify_frozen_bindings()
        value: dict[str, object] = {
            "schema_version": 1,
            "document_type": V4_DRY_RUN_DOCUMENT_TYPE,
            "default_mode": "dry_run",
            "remote_mutation_authorized": False,
            "v4_supervisor_sha256": _v4_sha256(),
            "v3_supervisor_sha256": FROZEN_V3_SHA256,
            "v2_supervisor_sha256": v3.FROZEN_V2_SHA256,
            "base_executor_sha256": v2.FROZEN_BASE_SHA256,
            "plan_seal_sha256": plan["seal_sha256"],
            "pinset_seal_sha256": pinset["seal_sha256"],
            "execution_key": v2.SEALED_EXECUTION_KEY,
            "barrier_artifact": V4_BARRIER_ARTIFACT,
            "completion_parameter_contract": _completion_parameter_contract(),
            "execution_token_sha256": hashlib.sha256(
                V4_EXECUTION_TOKEN.encode("utf-8")
            ).hexdigest(),
        }
        value["seal_sha256"] = base._seal(value)
        _print_json(value, pretty=args.pretty)
        return 0
    if args.execution_token != V4_EXECUTION_TOKEN:
        raise P1ParallelV4Error(
            "--execute requires the exact --execution-token acknowledgement"
        )
    receipt = execute_plan(
        authorization_token=args.execution_token,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    _print_json(receipt, pretty=args.pretty)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except P1RecoverableTimeout as error:
        print(f"P1-PARALLEL-V4-RECOVERABLE-TIMEOUT: {error}", file=sys.stderr)
        raise SystemExit(3) from None
    except P1ExecutorError as error:
        print(f"P1-PARALLEL-V4-ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
