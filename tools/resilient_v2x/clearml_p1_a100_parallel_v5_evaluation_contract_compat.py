#!/usr/bin/env python3
"""Resume P1 after sealing an exact legacy evaluation-contract amendment.

The byte-pinned Source-D evaluator records the training seed in the ClearML
task parameters and binds the evaluation to the exact training task, model,
and checkpoint, but its immutable ``run_contract`` omits two redundant fields:
``training_seed`` and ``training_overlay_protocol_seed``.  The frozen v4
supervisor therefore stops while accepting a successfully completed
evaluation.

This thin successor preserves the plan, pinset, execution key, controller,
journal, task identities, queue guards, scheduler, and every frozen predecessor
file.  It installs a durable v5 barrier and amends completed-evaluation
validation only.  The legacy run contract must have one exact key inventory;
the two fields must both be absent; the full task parameters must still bind
the expected seed and training task.  The missing values are added to an
in-memory copy for the frozen validator and are never written back to the task.

Default mode is local dry-run.  ``--preflight`` is remote read-only.  Remote
mutation requires ``--execute`` and the exact v5 acknowledgement token.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path


V4_FILENAME = "clearml_p1_a100_parallel_v4_parameter_compat.py"
FROZEN_V4_SHA256 = "e39dcde5187f99350f55e0210560890d505ca230da498fd1cb4d91c1d28a11c6"
V5_BARRIER_ARTIFACT = "p1_a100_parallel_v5_evaluation_contract_compat_barrier"
V5_BARRIER_DOCUMENT_TYPE = (
    "resilient_v2x_p1_a100_parallel_v5_evaluation_contract_compat_barrier"
)
V5_PREFLIGHT_DOCUMENT_TYPE = (
    "resilient_v2x_p1_a100_parallel_v5_evaluation_contract_compat_preflight"
)
V5_DRY_RUN_DOCUMENT_TYPE = (
    "resilient_v2x_p1_a100_parallel_v5_evaluation_contract_compat_dry_run"
)
V5_EXECUTION_RECEIPT_DOCUMENT_TYPE = (
    "resilient_v2x_p1_a100_parallel_v5_evaluation_contract_compat_execution_receipt"
)
V5_EXECUTION_TOKEN = (
    "RECOVER_AND_EXECUTE_P1_A100_PARALLEL_V5_EVALUATION_CONTRACT_COMPAT"
)

LIVE_SOURCE_JOURNAL_REVISION = 36
LIVE_SOURCE_JOURNAL_SEAL_SHA256 = (
    "b09222eec6478fd87088a0c5f6af1566f709443c73150c07b72cfe41cee96d5c"
)
LIVE_COMPLETED_EVALUATION_TASK_KEY = "eval-r01-s01-resilient_v2x"
LIVE_COMPLETED_EVALUATION_TASK_ID = "2fddc432826440389249f24fde8d3aa9"
LIVE_COMPLETED_EVALUATION_SUBJECT = "resilient_v2x"
LIVE_COMPLETED_EVALUATION_SEED = 20250218
LIVE_COMPLETED_EVALUATION_TRAINING_TASK_ID = "2c491c02491c45b4ac5deae3ff0bd44e"
LIVE_EVALUATION_RUN_CONTRACT_SHA256 = (
    "dab026363c1d2fae09e39287ba3174cd6dc8f7526e8f34bec79f7541672a265a"
)
LEGACY_MISSING_RUN_CONTRACT_FIELDS = frozenset(
    {"training_seed", "training_overlay_protocol_seed"}
)
LEGACY_EVALUATION_RUN_CONTRACT_KEYS = frozenset(
    {
        "baseline",
        "baseline_task_id",
        "checkpoint",
        "command",
        "dataset_root",
        "evaluator",
        "evaluator_sha256",
        "expected_conditions",
        "expected_delays_ms",
        "expected_ground_truth_count",
        "expected_manifest_content_sha256",
        "expected_overlay_index_content_sha256",
        "expected_run_count",
        "expected_sample_count",
        "expected_sample_ids_sha256",
        "mode",
        "overlay_index",
        "overlay_index_sha256",
        "predecessor_task_id",
        "protocol_id",
        "schema_version",
        "task_id",
        "training_dataset_id",
    }
)


def _load_v4() -> object:
    path = Path(__file__).resolve().with_name(V4_FILENAME)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != FROZEN_V4_SHA256:
        raise RuntimeError(
            "pinned parallel-v4 supervisor SHA-256 mismatch: "
            f"expected {FROZEN_V4_SHA256}, observed {observed}"
        )
    spec = importlib.util.spec_from_file_location(
        "_resilientv2x_p1_a100_parallel_v4_pinned", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the pinned parallel-v4 supervisor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4 = _load_v4()
v3 = v4.v3
v2 = v4.v2
base = v4.base
P1ExecutorError = base.P1ExecutorError
P1RecoverableTimeout = base.P1RecoverableTimeout


class P1ParallelV5Error(P1ExecutorError):
    """The v5 evaluation amendment or one of its bindings is invalid."""


def _v5_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _verify_frozen_bindings() -> tuple[dict[str, object], dict[str, object]]:
    path = Path(__file__).resolve().with_name(V4_FILENAME)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_V4_SHA256:
        raise P1ParallelV5Error("frozen parallel-v4 supervisor SHA-256 drifted")
    if v4._v4_sha256() != FROZEN_V4_SHA256:
        raise P1ParallelV5Error("parallel-v4 self fingerprint drifted")
    try:
        return v4._verify_frozen_bindings()
    except base.P1ExecutorError as error:
        raise P1ParallelV5Error(str(error)) from error


def _source_journal_binding(journal: Mapping[str, object]) -> dict[str, object]:
    value = base._mapping(journal, "parallel-v5 source journal")
    if (
        value.get("revision") != LIVE_SOURCE_JOURNAL_REVISION
        or value.get("seal_sha256") != LIVE_SOURCE_JOURNAL_SEAL_SHA256
        or value.get("status") != "running"
    ):
        raise P1ParallelV5Error("parallel-v5 source journal identity drifted")
    task_rows = value.get("tasks")
    if not isinstance(task_rows, list):
        raise P1ParallelV5Error("parallel-v5 source journal task list is invalid")
    rows = []
    for item in task_rows:
        row = base._mapping(item, "parallel-v5 source task")
        if row.get("task_key") == LIVE_COMPLETED_EVALUATION_TASK_KEY:
            rows.append(row)
    if len(rows) != 1:
        raise P1ParallelV5Error("parallel-v5 source evaluation binding is not unique")
    row = rows[0]
    expected = {
        "task_key": LIVE_COMPLETED_EVALUATION_TASK_KEY,
        "task_id": LIVE_COMPLETED_EVALUATION_TASK_ID,
        "state": "active",
        "server_status": "in_progress",
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise P1ParallelV5Error(f"parallel-v5 source evaluation {field} drifted")
    if row.get("result") is not None:
        raise P1ParallelV5Error("parallel-v5 source evaluation result is not empty")
    return {
        "revision": value["revision"],
        "seal_sha256": value["seal_sha256"],
        "status": value["status"],
        "evaluation_task_key": row["task_key"],
        "evaluation_task_id": row["task_id"],
        "evaluation_state": row["state"],
        "evaluation_server_status": row["server_status"],
    }


def _compatibility_contract() -> dict[str, object]:
    return {
        "scope": "completed_evaluation_result_only",
        "artifact": "run_contract",
        "legacy_artifact_sha256": LIVE_EVALUATION_RUN_CONTRACT_SHA256,
        "legacy_exact_keys": sorted(LEGACY_EVALUATION_RUN_CONTRACT_KEYS),
        "legacy_missing_fields": sorted(LEGACY_MISSING_RUN_CONTRACT_FIELDS),
        "task_parameter_training_seed_required": True,
        "task_parameter_training_task_binding_required": True,
        "scientific_plan_unchanged": True,
        "artifact_rewrite_forbidden": True,
        "created_task_validation_unchanged": True,
        "controller_validation_unchanged": True,
        "old_v4_execution_forbidden": True,
    }


def _expected_live_evaluation_contract_binding() -> dict[str, object]:
    return {
        "evaluation_task_id": LIVE_COMPLETED_EVALUATION_TASK_ID,
        "subject": LIVE_COMPLETED_EVALUATION_SUBJECT,
        "training_seed": LIVE_COMPLETED_EVALUATION_SEED,
        "training_task_id": LIVE_COMPLETED_EVALUATION_TRAINING_TASK_ID,
        "run_contract_artifact": "run_contract",
        "run_contract_sha256": LIVE_EVALUATION_RUN_CONTRACT_SHA256,
        "legacy_exact_keys": sorted(LEGACY_EVALUATION_RUN_CONTRACT_KEYS),
        "missing_fields": sorted(LEGACY_MISSING_RUN_CONTRACT_FIELDS),
    }


def _validate_live_evaluation_contract(
    task_class: object, *, controller_id: str
) -> dict[str, object]:
    task = v3._fresh_task(task_class, LIVE_COMPLETED_EVALUATION_TASK_ID)
    if base._task_status(task) != "completed" or base._task_parent(
        task
    ) != base._clearml_id(controller_id, "controller"):
        raise P1ParallelV5Error(
            "parallel-v5 source evaluation status or parent drifted"
        )
    fresh_task, run_contract, _descriptor, _raw = v3._authoritative_artifact(
        task_class,
        task_id=LIVE_COMPLETED_EVALUATION_TASK_ID,
        artifact_name="run_contract",
    )
    if str(getattr(fresh_task, "id", "")) != LIVE_COMPLETED_EVALUATION_TASK_ID:
        raise P1ParallelV5Error("parallel-v5 source evaluation identity drifted")
    artifact = base._artifact_inventory(fresh_task).get("run_contract")
    if artifact is None:
        raise P1ParallelV5Error("parallel-v5 source run contract is missing")
    if str(getattr(artifact, "hash", "") or "") != (
        LIVE_EVALUATION_RUN_CONTRACT_SHA256
    ):
        raise P1ParallelV5Error("parallel-v5 source run contract SHA-256 drifted")
    if set(run_contract).intersection(LEGACY_MISSING_RUN_CONTRACT_FIELDS):
        raise P1ParallelV5Error(
            "parallel-v5 source run contract is partially materialized"
        )
    if set(run_contract) != LEGACY_EVALUATION_RUN_CONTRACT_KEYS:
        raise P1ParallelV5Error("parallel-v5 source run contract key inventory drifted")
    parameters = base._parameters(fresh_task)
    expected_parameters = {
        "Args/training_seed": LIVE_COMPLETED_EVALUATION_SEED,
        "Args/predecessor_task_id": LIVE_COMPLETED_EVALUATION_TRAINING_TASK_ID,
        "Args/controlled_baseline_task_id": (
            LIVE_COMPLETED_EVALUATION_TRAINING_TASK_ID
        ),
        "Args/controlled_baseline": LIVE_COMPLETED_EVALUATION_SUBJECT,
        "Args/stage": "baseline_validate",
    }
    for key, expected_value in expected_parameters.items():
        if not base._normalize_parameter(parameters.get(key), expected_value):
            raise P1ParallelV5Error(
                f"parallel-v5 source evaluation parameter {key} drifted"
            )
    return _expected_live_evaluation_contract_binding()


def _build_v5_barrier(
    *,
    controller_id: str,
    journal: Mapping[str, object],
    amendment: Mapping[str, object],
    v3_barrier: Mapping[str, object],
    v4_barrier: Mapping[str, object],
    live_evaluation_contract: Mapping[str, object],
) -> dict[str, object]:
    plan, pinset = _verify_frozen_bindings()
    barrier: dict[str, object] = {
        "schema_version": 1,
        "document_type": V5_BARRIER_DOCUMENT_TYPE,
        "status": "active",
        "reason": "source_d_evaluation_run_contract_omits_redundant_seed_fields",
        "v5_supervisor_sha256": _v5_sha256(),
        "v4_supervisor_sha256": FROZEN_V4_SHA256,
        "v3_supervisor_sha256": v4.FROZEN_V3_SHA256,
        "v2_supervisor_sha256": v3.FROZEN_V2_SHA256,
        "base_executor_sha256": v2.FROZEN_BASE_SHA256,
        "plan_seal_sha256": plan["seal_sha256"],
        "pinset_seal_sha256": pinset["seal_sha256"],
        "execution_key": v2.SEALED_EXECUTION_KEY,
        "controller_task_id": base._clearml_id(controller_id, "controller"),
        "v2_amendment_artifact": v2.AMENDMENT_ARTIFACT,
        "v2_amendment_seal_sha256": amendment["seal_sha256"],
        "v3_barrier_artifact": v3.V3_BARRIER_ARTIFACT,
        "v3_barrier_seal_sha256": v3_barrier["seal_sha256"],
        "v4_barrier_artifact": v4.V4_BARRIER_ARTIFACT,
        "v4_barrier_seal_sha256": v4_barrier["seal_sha256"],
        "v5_barrier_artifact": V5_BARRIER_ARTIFACT,
        "source_journal": _source_journal_binding(journal),
        "source_evaluation_contract": base._mapping(
            live_evaluation_contract,
            "parallel-v5 source evaluation contract binding",
        ),
        "compatibility_contract": _compatibility_contract(),
    }
    barrier["seal_sha256"] = base._seal(barrier)
    return _validate_v5_barrier(
        barrier,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
        v4_barrier=v4_barrier,
    )


def _validate_v5_barrier(
    value: Mapping[str, object],
    *,
    controller_id: str,
    amendment: Mapping[str, object],
    v3_barrier: Mapping[str, object],
    v4_barrier: Mapping[str, object],
) -> dict[str, object]:
    barrier = base._mapping(value, "parallel-v5 evaluation-contract barrier")
    base._exact_keys(
        barrier,
        {
            "base_executor_sha256",
            "compatibility_contract",
            "controller_task_id",
            "document_type",
            "execution_key",
            "pinset_seal_sha256",
            "plan_seal_sha256",
            "reason",
            "schema_version",
            "seal_sha256",
            "source_journal",
            "source_evaluation_contract",
            "status",
            "v2_amendment_artifact",
            "v2_amendment_seal_sha256",
            "v2_supervisor_sha256",
            "v3_barrier_artifact",
            "v3_barrier_seal_sha256",
            "v3_supervisor_sha256",
            "v4_barrier_artifact",
            "v4_barrier_seal_sha256",
            "v4_supervisor_sha256",
            "v5_barrier_artifact",
            "v5_supervisor_sha256",
        },
        "parallel-v5 evaluation-contract barrier",
    )
    if barrier.get("seal_sha256") != base._seal(barrier):
        raise P1ParallelV5Error("parallel-v5 barrier seal mismatch")
    plan, pinset = _verify_frozen_bindings()
    expected = {
        "schema_version": 1,
        "document_type": V5_BARRIER_DOCUMENT_TYPE,
        "status": "active",
        "reason": "source_d_evaluation_run_contract_omits_redundant_seed_fields",
        "v5_supervisor_sha256": _v5_sha256(),
        "v4_supervisor_sha256": FROZEN_V4_SHA256,
        "v3_supervisor_sha256": v4.FROZEN_V3_SHA256,
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
        "v4_barrier_artifact": v4.V4_BARRIER_ARTIFACT,
        "v4_barrier_seal_sha256": v4_barrier.get("seal_sha256"),
        "v5_barrier_artifact": V5_BARRIER_ARTIFACT,
        "source_journal": {
            "revision": LIVE_SOURCE_JOURNAL_REVISION,
            "seal_sha256": LIVE_SOURCE_JOURNAL_SEAL_SHA256,
            "status": "running",
            "evaluation_task_key": LIVE_COMPLETED_EVALUATION_TASK_KEY,
            "evaluation_task_id": LIVE_COMPLETED_EVALUATION_TASK_ID,
            "evaluation_state": "active",
            "evaluation_server_status": "in_progress",
        },
        "source_evaluation_contract": _expected_live_evaluation_contract_binding(),
        "compatibility_contract": _compatibility_contract(),
    }
    for field, expected_value in expected.items():
        if barrier.get(field) != expected_value:
            raise P1ParallelV5Error(f"parallel-v5 barrier {field} drifted")
    return barrier


def _read_installed_v5_barrier(
    task_class: object,
    *,
    controller_id: str,
    amendment: Mapping[str, object],
    v3_barrier: Mapping[str, object],
    v4_barrier: Mapping[str, object],
) -> dict[str, object] | None:
    controller = v3._fresh_task(task_class, controller_id)
    if V5_BARRIER_ARTIFACT not in base._artifact_inventory(controller):
        return None
    _task, value, _descriptor, _raw = v3._authoritative_artifact(
        task_class,
        task_id=controller_id,
        artifact_name=V5_BARRIER_ARTIFACT,
    )
    return _validate_v5_barrier(
        value,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
        v4_barrier=v4_barrier,
    )


def _install_v5_barrier(
    task_class: object,
    *,
    controller_id: str,
    journal: Mapping[str, object],
    amendment: Mapping[str, object],
    v3_barrier: Mapping[str, object],
    v4_barrier: Mapping[str, object],
) -> dict[str, object]:
    existing = _read_installed_v5_barrier(
        task_class,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
        v4_barrier=v4_barrier,
    )
    if existing is not None:
        return existing
    if v3._read_pending_wal(controller_id) is not None:
        raise P1ParallelV5Error(
            "initial parallel-v5 barrier installation requires an absent pending WAL"
        )
    barrier = _build_v5_barrier(
        controller_id=controller_id,
        journal=journal,
        amendment=amendment,
        v3_barrier=v3_barrier,
        v4_barrier=v4_barrier,
        live_evaluation_contract=_validate_live_evaluation_contract(
            task_class, controller_id=controller_id
        ),
    )
    v3._upload_mapping_confirmed(
        task_class,
        controller_id=controller_id,
        artifact_name=V5_BARRIER_ARTIFACT,
        value=barrier,
    )
    installed = _read_installed_v5_barrier(
        task_class,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
        v4_barrier=v4_barrier,
    )
    if installed != barrier:
        raise P1ParallelV5Error("parallel-v5 barrier round-trip drifted")
    return installed


def _validate_bound_barriers(
    task_class: object,
    *,
    controller_id: str,
    journal: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    amendment, v3_barrier, v4_barrier = v4._validate_bound_barriers(
        task_class,
        controller_id=controller_id,
        journal=journal,
    )
    v5_barrier = _read_installed_v5_barrier(
        task_class,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
        v4_barrier=v4_barrier,
    )
    if v5_barrier is None:
        raise P1ParallelV5Error("parallel-v5 durable barrier is missing")
    return amendment, v3_barrier, v4_barrier, v5_barrier


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
        raise P1ParallelV5Error(str(error)) from error
    allowed = {
        base.PLAN_ARTIFACT,
        base.PINSET_ARTIFACT,
        base.JOURNAL_ARTIFACT,
        base.MANIFEST_ARTIFACT,
        v2.AMENDMENT_ARTIFACT,
        v3.V3_BARRIER_ARTIFACT,
        v4.V4_BARRIER_ARTIFACT,
        V5_BARRIER_ARTIFACT,
    }
    artifacts = base._artifact_inventory(controller)
    if not set(artifacts).issubset(allowed):
        raise P1ParallelV5Error(
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
    _validate_bound_barriers(
        task_class or type(controller),
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
def _legacy_evaluation_run_contract_runtime(
    task: object,
    *,
    subject: str,
    seed: int,
    training_task_id: str,
) -> Iterator[None]:
    original_mapping = base._artifact_mapping
    expected_context = f"{subject} seed {seed} evaluation run contract"

    def artifact_mapping(artifact: object, context: str) -> dict[str, object]:
        observed = original_mapping(artifact, context)
        if context != expected_context:
            return observed
        artifact_sha256 = str(getattr(artifact, "hash", "") or "")
        if artifact_sha256 != LIVE_EVALUATION_RUN_CONTRACT_SHA256:
            raise P1ParallelV5Error(f"{context} artifact SHA-256 drifted")
        if set(observed).intersection(LEGACY_MISSING_RUN_CONTRACT_FIELDS):
            raise P1ParallelV5Error(
                f"{context} contains a partially materialized seed binding"
            )
        if set(observed) != LEGACY_EVALUATION_RUN_CONTRACT_KEYS:
            raise P1ParallelV5Error(f"{context} legacy key inventory drifted")
        parameters = base._parameters(task)
        expected_parameters = {
            "Args/training_seed": seed,
            "Args/predecessor_task_id": training_task_id,
            "Args/controlled_baseline_task_id": training_task_id,
            "Args/controlled_baseline": subject,
            "Args/stage": "baseline_validate",
        }
        for key, expected_value in expected_parameters.items():
            if not base._normalize_parameter(parameters.get(key), expected_value):
                raise P1ParallelV5Error(
                    f"{context} authoritative task parameter {key} drifted"
                )
        amended = copy.deepcopy(observed)
        amended["training_seed"] = seed
        amended["training_overlay_protocol_seed"] = base.TRAINING_OVERLAY_PROTOCOL_SEED
        return amended

    base._artifact_mapping = artifact_mapping
    try:
        yield
    finally:
        base._artifact_mapping = original_mapping


def _validated_evaluation_result(
    original: Callable[..., dict[str, object]],
    task: object,
    *,
    subject: str,
    seed: int,
    controller_id: str,
    training_task_id: str,
    model: Mapping[str, object],
    source_d: str,
) -> dict[str, object]:
    with _legacy_evaluation_run_contract_runtime(
        task,
        subject=subject,
        seed=seed,
        training_task_id=training_task_id,
    ):
        return original(
            task,
            subject=subject,
            seed=seed,
            controller_id=controller_id,
            training_task_id=training_task_id,
            model=model,
            source_d=source_d,
        )


@contextlib.contextmanager
def _v5_runtime() -> Iterator[None]:
    original_existing = v4._load_existing_execution
    original_validated_load = v4._validated_authoritative_load_journal
    original_validated_remote = v4._validated_authoritative_remote_journal
    original_evaluation = base._validate_evaluation_result

    def validated_load(
        original: Callable[..., dict[str, object]],
        controller: object,
        *,
        plan: Mapping[str, object],
        execution_key: str,
        controller_id: str,
        task_class: object | None = None,
    ) -> dict[str, object]:
        return _validated_authoritative_load_journal(
            lambda *args, **kwargs: original_validated_load(original, *args, **kwargs),
            controller,
            plan=plan,
            execution_key=execution_key,
            controller_id=controller_id,
            task_class=task_class,
        )

    def validated_remote(
        original: Callable[..., tuple[object, dict[str, object]]],
        task_class: object,
        controller_id: str,
    ) -> tuple[object, dict[str, object]]:
        return _validated_authoritative_remote_journal(
            lambda *args, **kwargs: original_validated_remote(
                original, *args, **kwargs
            ),
            task_class,
            controller_id,
        )

    def validate_evaluation(task: object, **kwargs: object) -> dict[str, object]:
        return _validated_evaluation_result(
            original_evaluation,
            task,
            subject=str(kwargs["subject"]),
            seed=int(kwargs["seed"]),
            controller_id=str(kwargs["controller_id"]),
            training_task_id=str(kwargs["training_task_id"]),
            model=base._mapping(kwargs["model"], "parallel-v5 evaluation model"),
            source_d=str(kwargs["source_d"]),
        )

    v4._load_existing_execution = _load_existing_execution
    v4._validated_authoritative_load_journal = validated_load
    v4._validated_authoritative_remote_journal = validated_remote
    base._validate_evaluation_result = validate_evaluation
    try:
        yield
    finally:
        v4._load_existing_execution = original_existing
        v4._validated_authoritative_load_journal = original_validated_load
        v4._validated_authoritative_remote_journal = original_validated_remote
        base._validate_evaluation_result = original_evaluation


def _read_current_bound_execution(
    task_class: object, base_receipt: Mapping[str, object]
) -> tuple[
    str,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    controller_id, journal, amendment, v3_barrier = v4._read_current_bound_execution(
        task_class, base_receipt
    )
    v4_barrier = v4._read_installed_v4_barrier(
        task_class,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
    )
    if v4_barrier is None:
        raise P1ParallelV5Error("parallel-v4 durable barrier is missing")
    return controller_id, journal, amendment, v3_barrier, v4_barrier


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
    """Read-only validation of v4 plus the optional installed v5 barrier."""

    _verify_frozen_bindings()
    task_type = task_class or base._load_clearml()
    v4_receipt = v4.preflight(
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
    controller_id, journal, amendment, v3_barrier, v4_barrier = (
        _read_current_bound_execution(task_type, base_receipt)
    )
    v5_barrier = _read_installed_v5_barrier(
        task_type,
        controller_id=controller_id,
        amendment=amendment,
        v3_barrier=v3_barrier,
        v4_barrier=v4_barrier,
    )
    if v5_barrier is None:
        if v3._read_pending_wal(controller_id) is not None:
            raise P1ParallelV5Error(
                "initial parallel-v5 preflight requires an absent pending WAL"
            )
        source_binding = _source_journal_binding(journal)
        live_evaluation_contract = _validate_live_evaluation_contract(
            task_type, controller_id=controller_id
        )
    else:
        source_binding = copy.deepcopy(v5_barrier["source_journal"])
        live_evaluation_contract = copy.deepcopy(
            v5_barrier["source_evaluation_contract"]
        )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": V5_PREFLIGHT_DOCUMENT_TYPE,
        "status": "validated",
        "readonly": True,
        "remote_mutation_count": 0,
        "v5_supervisor_sha256": _v5_sha256(),
        "v4_supervisor_sha256": FROZEN_V4_SHA256,
        "v3_supervisor_sha256": v4.FROZEN_V3_SHA256,
        "v2_supervisor_sha256": v3.FROZEN_V2_SHA256,
        "base_executor_sha256": v2.FROZEN_BASE_SHA256,
        "plan_seal_sha256": v2.SEALED_PLAN_SHA256,
        "pinset_seal_sha256": v2.SEALED_PINSET_SHA256,
        "execution_key": v2.SEALED_EXECUTION_KEY,
        "controller_task_id": controller_id,
        "journal_revision": journal["revision"],
        "journal_seal_sha256": journal["seal_sha256"],
        "source_binding": source_binding,
        "source_evaluation_contract": live_evaluation_contract,
        "v4_barrier_seal_sha256": v4_barrier["seal_sha256"],
        "v5_barrier_status": "installed" if v5_barrier is not None else "not_installed",
        "v5_barrier_seal_sha256": (
            v5_barrier["seal_sha256"] if v5_barrier is not None else None
        ),
        "compatibility_contract": _compatibility_contract(),
        "v4_preflight_seal_sha256": v4_receipt["seal_sha256"],
        "queue": copy.deepcopy(v4_receipt["queue"]),
        "global_mutex": copy.deepcopy(v4_receipt["global_mutex"]),
    }
    receipt["seal_sha256"] = base._seal(receipt)
    return receipt


def _install_v5_barrier_under_mutex(
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
            controller_id, journal, amendment, v3_barrier, v4_barrier = (
                _read_current_bound_execution(task_class, base_receipt)
            )
            barrier = _install_v5_barrier(
                task_class,
                controller_id=controller_id,
                journal=journal,
                amendment=amendment,
                v3_barrier=v3_barrier,
                v4_barrier=v4_barrier,
            )
            return {
                "controller_task_id": controller_id,
                "source_journal_revision": journal["revision"],
                "source_journal_seal_sha256": journal["seal_sha256"],
                "v4_barrier_seal_sha256": v4_barrier["seal_sha256"],
                "v5_barrier_seal_sha256": barrier["seal_sha256"],
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
    """Install the v5 fence, then run the unchanged v4 scheduler stack."""

    if authorization_token != V5_EXECUTION_TOKEN:
        raise P1ParallelV5Error("parallel-v5 execution token mismatch")
    if not math.isfinite(poll_seconds) or not 1.0 <= poll_seconds <= 60.0:
        raise P1ParallelV5Error("poll-seconds must be finite and within [1, 60]")
    if not math.isfinite(timeout_hours) or not 1.0 <= timeout_hours <= 72.0:
        raise P1ParallelV5Error("timeout-hours must be finite and within [1, 72]")
    _verify_frozen_bindings()
    try:
        v3._validate_control_host()
    except base.P1ExecutorError as error:
        raise P1ParallelV5Error(str(error)) from error
    task_type = task_class or base._load_clearml()
    client = api_client or base._api_client()
    migration = _install_v5_barrier_under_mutex(
        task_class=task_type,
        api_client=client,
        queue_reader=queue_reader,
        legacy_retirement_reader=legacy_retirement_reader,
    )
    with _v5_runtime():
        v4_receipt = v4.execute_plan(
            authorization_token=v4.V4_EXECUTION_TOKEN,
            poll_seconds=poll_seconds,
            timeout_hours=timeout_hours,
            task_class=task_type,
            queue_reader=queue_reader,
            api_client=client,
            legacy_retirement_reader=legacy_retirement_reader,
        )
    result: dict[str, object] = {
        "schema_version": 1,
        "document_type": V5_EXECUTION_RECEIPT_DOCUMENT_TYPE,
        "status": "completed",
        "v5_supervisor_sha256": _v5_sha256(),
        "v4_supervisor_sha256": FROZEN_V4_SHA256,
        "v5_barrier_seal_sha256": migration["v5_barrier_seal_sha256"],
        "migration": copy.deepcopy(migration),
        "v4_execution_receipt": copy.deepcopy(v4_receipt),
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
            raise P1ParallelV5Error("--execution-token is invalid with --preflight")
        _print_json(preflight(), pretty=args.pretty)
        return 0
    if not args.execute:
        if args.execution_token:
            raise P1ParallelV5Error("--execution-token is invalid without --execute")
        plan, pinset = _verify_frozen_bindings()
        value: dict[str, object] = {
            "schema_version": 1,
            "document_type": V5_DRY_RUN_DOCUMENT_TYPE,
            "default_mode": "dry_run",
            "remote_mutation_authorized": False,
            "v5_supervisor_sha256": _v5_sha256(),
            "v4_supervisor_sha256": FROZEN_V4_SHA256,
            "plan_seal_sha256": plan["seal_sha256"],
            "pinset_seal_sha256": pinset["seal_sha256"],
            "execution_key": v2.SEALED_EXECUTION_KEY,
            "barrier_artifact": V5_BARRIER_ARTIFACT,
            "compatibility_contract": _compatibility_contract(),
            "execution_token_sha256": hashlib.sha256(
                V5_EXECUTION_TOKEN.encode("utf-8")
            ).hexdigest(),
        }
        value["seal_sha256"] = base._seal(value)
        _print_json(value, pretty=args.pretty)
        return 0
    if args.execution_token != V5_EXECUTION_TOKEN:
        raise P1ParallelV5Error(
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
    except (P1ExecutorError, RuntimeError, ValueError) as error:
        print(f"P1-PARALLEL-V5-ERROR: {error}")
        raise SystemExit(2) from None
