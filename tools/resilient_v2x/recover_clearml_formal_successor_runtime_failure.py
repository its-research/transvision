#!/usr/bin/env python3
"""Recover the exact failed formal W->L->A->S successor chain.

The tool never creates or clones a ClearML task.  Execution first freezes a
write-once attempt receipt, resets the four fixed task IDs in reverse dependency
order, reinstalls one coherently rendered source/parameter graph, proves that
all four tasks are created and unqueued, and only then enqueues W, L, A, S.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

try:
    from tools.resilient_v2x import deploy_clearml_formal_successor_chain as deploy
    from tools.resilient_v2x import recover_clearml_exact_eval_chain_runtime_failure as shared
    from tools.resilient_v2x import recover_clearml_original_queue_runtime_failure as original
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import deploy_clearml_formal_successor_chain as deploy
    import recover_clearml_exact_eval_chain_runtime_failure as shared
    import recover_clearml_original_queue_runtime_failure as original


SUCCESSOR_TASK_IDS = {
    "W": "74bf35decc2340b496167c11ec50f54b",
    "L": "cc7b54e4dcca4b92a4247f92987add60",
    "A": "dd287fe58e264bcca48fe222b6981048",
    "S": "d7ea54ce540d4b0486904c5910100885",
}
RESET_ORDER = ("S", "A", "L", "W")
CONFIGURE_ORDER = ("W", "L", "A", "S")
ENQUEUE_ORDER = CONFIGURE_ORDER
EXECUTE_TOKEN = (
    "RESET_EXACT_FORMAL_W_L_A_S_"
    "74BF35DECC2340B496167C11EC50F54B"
)
ATTEMPT_RECEIPT_TYPE = "resilient_v2x_formal_successor_runtime_recovery_attempt"
PLAN_RECEIPT_TYPE = "resilient_v2x_formal_successor_runtime_recovery_plan"
RECOVERY_RECEIPT_TYPE = "resilient_v2x_formal_successor_runtime_recovery"
EFFECTIVE_AUTHORITY_TYPE = "resilient_v2x_effective_formal_successor_authority"
MAX_RECEIPT_BYTES = 16 * 1024 * 1024

EXPECTED_FAILURE_MESSAGES = {
    "W": (
        "candidate E3 evaluation queue drifted",
        "queue occupancy execution queue must be a lowercase 32-hex ClearML ID",
    ),
    "L": "evaluation watcher ended as 'failed'",
    "A": "evaluation watcher ended as 'failed'",
    "S": "comparability audit ended as 'failed'",
}

_COMMON_RUNTIME_DEFAULTS = {
    "Args/evaluation_plan_amendment_producer_task_id": "",
    "Args/evaluation_plan_amendment_receipt_seal_sha256": "",
    "Args/evaluation_plan_amendment_revised_plan_seal_sha256": "",
    "Args/evaluation_plan_amendment_evidence_seal_sha256": "",
    "Args/evaluation_plan_amendment_worker_evidence_seal_sha256": "",
    "Args/evaluation_plan_amendment_task_ids_sha256": "",
    "Args/exact_eval_runtime_recovery_receipt_seal_sha256": "",
    "Args/exact_eval_runtime_recovery_attempt_seal_sha256": "",
    "Args/exact_eval_runtime_ffnet_source_sha256": "",
    "Args/exact_eval_runtime_ffnet_parameters_sha256": "",
    "Args/exact_eval_runtime_candidate_source_sha256": "",
    "Args/exact_eval_runtime_candidate_parameters_sha256": "",
}
_ROLE_RUNTIME_DEFAULTS = {
    "W": {
        "Args/dependencies_json": "",
        "Args/evaluation_plan_json": "",
        "Args/expected_training_script_sha256": "",
        "Args/expected_training_source_archive_sha256": "",
        "Args/expected_training_source_dataset_id": "",
        "Args/metadata_only_artifact_gate": "False",
        "Args/training_manifest_json": "",
        "Args/authoritative_evaluation_plan_amendment_receipt_seal_sha256": "",
        "Args/authoritative_evaluation_plan_producer_source_sha256": "",
        "Args/authoritative_evaluation_plan_amendment_evidence_seal_sha256": "",
        "Args/authoritative_evaluation_plan_worker_evidence_seal_sha256": "",
        "Args/authoritative_evaluation_plan_task_ids_sha256": "",
    },
    "L": {},
    "A": {},
    "S": {"Args/formal_inputs_artifact": "final_selector_formal_inputs"},
}


class SuccessorRuntimeRecoveryError(RuntimeError):
    """Raised when exact-ID successor recovery cannot be proven."""


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


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    raw = payload.encode("utf-8")
    if not raw or len(raw) > MAX_RECEIPT_BYTES:
        raise SuccessorRuntimeRecoveryError("write-once receipt size is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _require_new(path: Path, *, context: str) -> None:
    if path.is_symlink() or path.exists():
        raise SuccessorRuntimeRecoveryError(f"{context} must be a new path")


def _normalized_parameters(
    role: str, parameters: Mapping[str, object]
) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in parameters.items()}
    defaults = {**_COMMON_RUNTIME_DEFAULTS, **_ROLE_RUNTIME_DEFAULTS[role]}
    for key, expected in defaults.items():
        if key in normalized and normalized.pop(key) != expected:
            raise SuccessorRuntimeRecoveryError(
                f"role {role} runtime default {key} drifted"
            )
    return normalized


def _load_deployment(
    path: Path,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    value, resolved = shared._read_sealed(
        path, context="original successor deployment receipt"
    )
    evaluation_row = value.get("evaluation_recovery_receipt")
    original_row = value.get("original_queue_runtime_recovery")
    if not isinstance(evaluation_row, Mapping) or not isinstance(original_row, Mapping):
        raise SuccessorRuntimeRecoveryError("deployment recovery bindings are absent")
    evaluation = shared._evaluation_binding(Path(str(evaluation_row.get("path") or "")))
    ffnet = original.validate_ffnet_recovery_receipt(
        Path(str(original_row.get("path") or ""))
    )
    try:
        _, validated = original._successor_binding(
            path,
            evaluation_binding=evaluation,
            ffnet_binding=ffnet,
        )
    except original.OriginalQueueRecoveryError as error:
        raise SuccessorRuntimeRecoveryError(
            "original successor deployment receipt is invalid"
        ) from error
    expected_ids = {"P": deploy.COMPLETED_PROVENANCE_TASK_ID, **SUCCESSOR_TASK_IDS}
    if (
        validated.get("task_ids") != expected_ids
        or validated.get("created_task_ids") != SUCCESSOR_TASK_IDS
    ):
        raise SuccessorRuntimeRecoveryError("deployment exact task IDs drifted")
    return validated, resolved, ffnet


def _target_contract(deployment: Mapping[str, object]) -> dict[str, object]:
    task_ids = dict(deployment["task_ids"])
    base_sources = deploy._read_base_sources()
    rendered, replacements, embedded, embedded_records = (
        deploy._render_sources_with_bundle_evidence(base_sources, task_ids)
    )
    smoke = deploy._isolated_import_smoke(rendered, embedded_records)
    source_records = deploy._source_records(
        base_sources, rendered, replacements, embedded_records
    )
    deployment_sources = deployment.get("sources")
    if (
        not isinstance(deployment_sources, Mapping)
        or hashlib.sha256(base_sources["P"].encode("utf-8")).hexdigest()
        != deploy.COMPLETED_PROVENANCE_SOURCE_SHA256
        or _canonical_json(source_records["P"])
        != _canonical_json(deployment_sources.get("P"))
    ):
        raise SuccessorRuntimeRecoveryError(
            "sealed completed P local source authority drifted"
        )
    source_hashes = {
        role: str(source_records[role]["deployed_sha256"])
        for role in deploy.ROLE_ORDER
    }
    all_parameters = deploy._parameters(
        task_ids,
        source_hashes,
        authoritative_recovery=True,
        original_queue_recovery_binding=deployment[
            "original_queue_runtime_recovery"
        ],
    )
    all_parents = deploy._parent_ids(task_ids)
    return {
        "task_ids": task_ids,
        "parents": {role: all_parents[role] for role in CONFIGURE_ORDER},
        "sources": source_records,
        "rendered_sources": {role: rendered[role] for role in CONFIGURE_ORDER},
        "parameters": {
            role: deploy._normalized_expected_parameters(all_parameters[role])
            for role in CONFIGURE_ORDER
        },
        "raw_parameters": {role: all_parameters[role] for role in CONFIGURE_ORDER},
        "standalone_import_smoke": smoke,
        "base_sources": base_sources,
        "embedded_sources": embedded,
    }


def _console_evidence(task: object, *, role: str) -> dict[str, object]:
    getter = getattr(task, "get_reported_console_output", None)
    if not callable(getter):
        raise SuccessorRuntimeRecoveryError(
            f"successor {role} console evidence is unavailable"
        )
    try:
        lines = [str(item) for item in (getter(200) or [])][-200:]
    except Exception as error:
        raise SuccessorRuntimeRecoveryError(
            f"successor {role} console evidence cannot be read"
        ) from error
    raw = "\n".join(lines).encode("utf-8")[-262_144:]
    text = raw.decode("utf-8", errors="replace")
    expected = EXPECTED_FAILURE_MESSAGES[role]
    accepted = (expected,) if isinstance(expected, str) else tuple(expected)
    runtime_messages = {
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if "RuntimeError:" in line or "ValueError:" in line
    }
    if len(runtime_messages) != 1 or not runtime_messages.issubset(set(accepted)):
        raise SuccessorRuntimeRecoveryError(
            f"successor {role} failure reason is not an accepted exact signature"
        )
    observed = next(iter(runtime_messages))
    data = getattr(task, "data", None)
    status_reason = str(getattr(data, "status_reason", "") or "")
    if status_reason != "worker execution exit code 1":
        raise SuccessorRuntimeRecoveryError(
            f"successor {role} status reason drifted"
        )
    return {
        "content_stored": False,
        "tail_line_count": len(lines),
        "tail_bytes": len(raw),
        "tail_sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_error": observed,
        "status_reason": status_reason,
    }


def _reference_and_provenance_authority(
    task_class: object, deployment: Mapping[str, object]
) -> dict[str, object]:
    base = deploy._read_base_sources()
    provenance = deploy._validate_completed_provenance(
        task_class,
        task_id=deploy.COMPLETED_PROVENANCE_TASK_ID,
        local_source=base["P"],
    )
    controller_source = deploy._read_source(
        deploy.ROOT / "tools/resilient_v2x" / deploy.CONTROLLER_ENTRY_POINT,
        context="training controller",
    )
    references = {
        "training_controller": deploy._validate_reference_task(
            task_class,
            task_id=deploy.TRAINING_CONTROLLER_TASK_ID,
            entry_point=deploy.CONTROLLER_ENTRY_POINT,
            script_sha256=deploy._sha256_text(controller_source),
            context="training controller",
        ),
        "evaluation_template": deploy._validate_reference_task(
            task_class,
            task_id=deploy.EVALUATION_TEMPLATE_TASK_ID,
            entry_point="clearml_5090_bootstrap.py",
            script_sha256=deploy.EVALUATION_SCRIPT_SHA256,
            context="evaluation template",
        ),
    }
    if _canonical_json(references) != _canonical_json(deployment.get("references")):
        raise SuccessorRuntimeRecoveryError(
            "deployment reference authority drifted"
        )
    if _canonical_json(provenance) != _canonical_json(
        deployment.get("completed_provenance")
    ):
        raise SuccessorRuntimeRecoveryError("completed P authority drifted")
    return {"references": references, "completed_provenance": provenance}


def _formal_core_authority(
    task_class: object,
    deployment: Mapping[str, object],
    ffnet_binding: Mapping[str, object],
) -> dict[str, object]:
    row = deployment.get("original_queue_runtime_recovery")
    if not isinstance(row, Mapping):
        raise SuccessorRuntimeRecoveryError("FFNet recovery binding is absent")
    ffnet_receipt, resolved = shared._read_sealed(
        Path(str(row.get("path") or "")),
        context="original queue FFNet recovery receipt",
    )
    if (
        str(resolved) != str(Path(str(row.get("path") or "")).resolve())
        or ffnet_receipt.get("seal_sha256") != row.get("receipt_seal_sha256")
    ):
        raise SuccessorRuntimeRecoveryError(
            "FFNet recovery receipt binding drifted"
        )
    coformer = ffnet_receipt.get("coformer_completed_evidence")
    if not isinstance(coformer, Mapping):
        raise SuccessorRuntimeRecoveryError("CoFormer evidence is absent")
    return original._live_formal_core(
        task_class,
        ffnet_binding=ffnet_binding,
        coformer_expected=coformer,
    )


def _dependency_authority(
    task_class: object,
    deployment: Mapping[str, object],
    ffnet_binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        **_reference_and_provenance_authority(task_class, deployment),
        "formal_core": _formal_core_authority(
            task_class, deployment, ffnet_binding
        ),
    }


def _exact_name_inventory(
    task_class: object, *, role: str, task: object, parent_id: str
) -> list[str]:
    return shared._exact_name_inventory(
        task_class,
        task_name=str(getattr(task, "name", "") or ""),
        parent_task_id=parent_id,
        expected_task_id=SUCCESSOR_TASK_IDS[role],
    )


def _failed_snapshot(
    task_class: object,
    *,
    role: str,
    deployment: Mapping[str, object],
    prior_recovery: Mapping[str, object] | None = None,
) -> dict[str, object]:
    task_id = SUCCESSOR_TASK_IDS[role]
    expected = deployment["queued_authority"][role]
    task = shared._authoritative_task(
        task_class, task_id, context=f"failed successor {role}"
    )
    context = f"failed successor {role}"
    source = shared._source(
        task,
        entry_point=str(deploy.ROLE_DEFINITIONS[role]["entry_point"]),
        context=context,
    )
    parameters = _normalized_parameters(
        role, shared._parameters(task, context=context)
    )
    queue, queue_id = shared._queue_identity(task, context=context)
    artifacts = shared._artifact_inventory(task)
    models = shared._model_inventory(task, context=context)
    artifact_names = {
        str(item.get("name") or "")
        for item in artifacts
        if isinstance(item, Mapping)
    }
    allowed_artifact_names = set(deploy.ROLE_OUTPUT_ARTIFACTS[role])
    expected_source_sha256 = (
        str(prior_recovery["source_sha256"][role])
        if prior_recovery is not None
        else str(expected["script_sha256"])
    )
    if (
        shared._status(task) != "failed"
        or str(getattr(task, "name", "") or "")
        != deploy.ROLE_DEFINITIONS[role]["task_name"]
        or deploy._task_parent(task) != expected["parent_task_id"]
        or shared._exact_project_id(task, context=context)
        != shared.evaluation_recovery.PROJECT_ID
        or str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
        != "controller"
        or hashlib.sha256(source.encode("utf-8")).hexdigest()
        != expected_source_sha256
        or parameters
        != {str(key): str(value) for key, value in expected["parameters"].items()}
        or queue != deploy.SERVICES_QUEUE
        or queue_id != deploy.SERVICES_QUEUE_ID
        or len(artifact_names) != len(artifacts)
        or not artifact_names.issubset(allowed_artifact_names)
        or models != {"input": [], "output": []}
    ):
        raise SuccessorRuntimeRecoveryError(
            f"failed successor {role} authority drifted"
        )
    tags = sorted(shared._tags(task, field="tags", context=context))
    system_tags = sorted(shared._tags(task, field="system_tags", context=context))
    return {
        "role": role,
        "task_id": task_id,
        "task_name": str(getattr(task, "name", "") or ""),
        "parent_task_id": expected["parent_task_id"],
        "status": "failed",
        "queue": queue,
        "queue_id": queue_id,
        "source_sha256": expected_source_sha256,
        "parameters_sha256": _content_sha256(parameters),
        "tags": tags,
        "system_tags": system_tags,
        "artifact_inventory": artifacts,
        "model_inventory": models,
        "service_environment": shared._service_environment(task, context=context),
        "failure_evidence": _console_evidence(task, role=role),
        "exact_name_inventory": _exact_name_inventory(
            task_class,
            role=role,
            task=task,
            parent_id=str(expected["parent_task_id"]),
        ),
    }


def build_attempt(
    task_class: object,
    *,
    successor_deployment_receipt: Path,
    prior_recovery_receipt: Path | None = None,
) -> dict[str, object]:
    deployment, resolved, ffnet_binding = _load_deployment(
        successor_deployment_receipt
    )
    prior_recovery = None
    if prior_recovery_receipt is not None:
        prior_recovery = validate_successor_runtime_recovery_receipt(
            prior_recovery_receipt
        )
        if (
            prior_recovery["original_successor_deployment"]
            != {"path": str(resolved), "receipt_seal_sha256": deployment["seal_sha256"]}
            or prior_recovery["task_ids"] != SUCCESSOR_TASK_IDS
        ):
            raise SuccessorRuntimeRecoveryError(
                "prior successor recovery binding drifted"
            )
    target = _target_contract(deployment)
    dependency = _dependency_authority(task_class, deployment, ffnet_binding)
    failed = {
        role: _failed_snapshot(
            task_class,
            role=role,
            deployment=deployment,
            prior_recovery=prior_recovery,
        )
        for role in CONFIGURE_ORDER
    }
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": ATTEMPT_RECEIPT_TYPE,
            "generated_at_utc": shared._utc_timestamp(),
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "original_successor_deployment": {
                "path": str(resolved),
                "receipt_seal_sha256": deployment["seal_sha256"],
            },
            "prior_successor_runtime_recovery": prior_recovery,
            "task_ids": dict(SUCCESSOR_TASK_IDS),
            "reset_order": list(RESET_ORDER),
            "configure_order": list(CONFIGURE_ORDER),
            "enqueue_order": list(ENQUEUE_ORDER),
            "pre_reset_authority": failed,
            "dependency_authority": dependency,
            "target": {
                "parents": target["parents"],
                "sources": target["sources"],
                "parameters": target["parameters"],
                "standalone_import_smoke": target["standalone_import_smoke"],
            },
            "planned_actions": [
                "durably_write_attempt_receipt",
                "revalidate_all_four_failed_authorities",
                "reset_exact_ids_in_reverse_dependency_order_S_A_L_W",
                "configure_all_four_exact_ids_without_create_or_clone",
                "prove_all_four_created_unqueued",
                "enqueue_exact_ids_in_dependency_order_W_L_A_S",
            ],
        }
    )


def _validate_attempt(value: Mapping[str, object]) -> None:
    expected = {
        "schema_version": 1,
        "receipt_type": ATTEMPT_RECEIPT_TYPE,
        "status": "frozen_before_reset",
        "remote_state_changed": False,
        "same_ids_only": True,
        "replacement_tasks_created": False,
        "task_ids": SUCCESSOR_TASK_IDS,
        "reset_order": list(RESET_ORDER),
        "configure_order": list(CONFIGURE_ORDER),
        "enqueue_order": list(ENQUEUE_ORDER),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise SuccessorRuntimeRecoveryError("attempt receipt identity drifted")
    if _seal(value).get("seal_sha256") != value.get("seal_sha256"):
        raise SuccessorRuntimeRecoveryError("attempt receipt seal drifted")


def _wait_created(
    task_class: object,
    task_id: str,
    *,
    sleeper: Callable[[float], None],
) -> object:
    last = ""
    for attempt in range(5):
        task = shared._authoritative_task(
            task_class, task_id, context="successor reset readback"
        )
        last = shared._status(task)
        if last == "created":
            return task
        if attempt != 4:
            sleeper(1.0)
    raise SuccessorRuntimeRecoveryError(
        f"task {task_id} did not reach created after reset; observed {last!r}"
    )


def _reset_one(
    task_class: object,
    *,
    role: str,
    frozen: Mapping[str, object],
    deployment: Mapping[str, object],
    sleeper: Callable[[float], None],
    prior_recovery: Mapping[str, object] | None = None,
) -> object:
    if _failed_snapshot(
        task_class,
        role=role,
        deployment=deployment,
        prior_recovery=prior_recovery,
    ) != frozen:
        raise SuccessorRuntimeRecoveryError(
            f"successor {role} changed after attempt freeze"
        )
    task = shared._authoritative_task(
        task_class, SUCCESSOR_TASK_IDS[role], context=f"successor {role} reset"
    )
    resetter = getattr(task, "reset", None)
    if not callable(resetter):
        raise SuccessorRuntimeRecoveryError(f"successor {role} cannot reset")
    callback_error: Exception | None = None
    try:
        if resetter(force=True) is False:
            callback_error = SuccessorRuntimeRecoveryError(
                f"successor {role} reset callback rejected"
            )
    except Exception as error:  # authority readback can still prove acceptance
        callback_error = error
    try:
        created = _wait_created(
            task_class, SUCCESSOR_TASK_IDS[role], sleeper=sleeper
        )
    except Exception as error:
        if callback_error is not None:
            raise SuccessorRuntimeRecoveryError(
                f"successor {role} reset authority readback failed"
            ) from callback_error
        raise error
    queue, queue_id = shared._queue_identity(
        created, context=f"successor {role} reset shell"
    )
    if (
        queue
        or queue_id
        or shared._artifact_inventory(created)
        or shared._model_inventory(created, context=f"successor {role} reset shell")
        != {"input": [], "output": []}
    ):
        raise SuccessorRuntimeRecoveryError(
            f"successor {role} reset shell is not empty and unqueued"
        )
    return created


def _created_authority(
    task_class: object,
    *,
    role: str,
    target: Mapping[str, object],
    frozen: Mapping[str, object],
) -> dict[str, object]:
    task = shared._authoritative_task(
        task_class, SUCCESSOR_TASK_IDS[role], context=f"successor {role} created"
    )
    deploy._validate_configured_task(
        task_class,
        role=role,
        task_id=SUCCESSOR_TASK_IDS[role],
        parent_id=str(target["parents"][role]),
        source=str(target["rendered_sources"][role]),
        parameters=target["raw_parameters"][role],
        expected_statuses=("created",),
    )
    queue, queue_id = shared._queue_identity(
        task, context=f"successor {role} created"
    )
    tags = sorted(shared._tags(task, field="tags", context=f"successor {role}"))
    system_tags = sorted(
        shared._tags(task, field="system_tags", context=f"successor {role}")
    )
    if (
        queue
        or queue_id
        or tags != frozen["tags"]
        or system_tags != frozen["system_tags"]
        or shared._artifact_inventory(task)
        or shared._model_inventory(task, context=f"successor {role} created")
        != {"input": [], "output": []}
    ):
        raise SuccessorRuntimeRecoveryError(
            f"successor {role} created authority drifted"
        )
    return {
        "task_id": SUCCESSOR_TASK_IDS[role],
        "status": "created",
        "project_id": shared.evaluation_recovery.PROJECT_ID,
        "task_name": deploy.ROLE_DEFINITIONS[role]["task_name"],
        "task_type": "controller",
        "parent_task_id": target["parents"][role],
        "source_sha256": target["sources"][role]["deployed_sha256"],
        "parameters_sha256": _content_sha256(target["parameters"][role]),
        "queue": None,
        "queue_id": None,
        "artifact_inventory": [],
        "model_inventory": {"input": [], "output": []},
        "tags": tags,
        "system_tags": system_tags,
        "service_environment": shared._service_environment(
            task, context=f"successor {role} created"
        ),
    }


def _live_authority(
    task_class: object,
    *,
    role: str,
    target: Mapping[str, object],
) -> dict[str, object]:
    task = shared._authoritative_task(
        task_class, SUCCESSOR_TASK_IDS[role], context=f"successor {role} live"
    )
    context = f"successor {role} live"
    status = shared._status(task)
    source = shared._source(
        task,
        entry_point=str(deploy.ROLE_DEFINITIONS[role]["entry_point"]),
        context=context,
    )
    parameters = _normalized_parameters(
        role, shared._parameters(task, context=context)
    )
    queue, queue_id = shared._queue_identity(task, context=context)
    artifacts = shared._artifact_inventory(task)
    artifact_names = {str(item["name"]) for item in artifacts}
    expected_artifacts = set(deploy.ROLE_OUTPUT_ARTIFACTS[role])
    tags = sorted(shared._tags(task, field="tags", context=context))
    system_tags = sorted(
        shared._tags(task, field="system_tags", context=context)
    )
    if (
        status not in {"queued", "in_progress", "completed"}
        or str(getattr(task, "name", "") or "")
        != deploy.ROLE_DEFINITIONS[role]["task_name"]
        or deploy._task_parent(task) != target["parents"][role]
        or shared._exact_project_id(task, context=context)
        != shared.evaluation_recovery.PROJECT_ID
        or str(getattr(task, "task_type", "") or "")
        .rsplit(".", 1)[-1]
        .lower()
        != "controller"
        or hashlib.sha256(source.encode("utf-8")).hexdigest()
        != target["sources"][role]["deployed_sha256"]
        or parameters != target["parameters"][role]
        or queue != deploy.SERVICES_QUEUE
        or queue_id != deploy.SERVICES_QUEUE_ID
        or not artifact_names.issubset(expected_artifacts)
        or (status == "completed" and artifact_names != expected_artifacts)
        or shared._model_inventory(task, context=context)
        != {"input": [], "output": []}
    ):
        raise SuccessorRuntimeRecoveryError(
            f"successor {role} live authority drifted"
        )
    return {
        "task_id": SUCCESSOR_TASK_IDS[role],
        "status": status,
        "project_id": shared.evaluation_recovery.PROJECT_ID,
        "task_name": deploy.ROLE_DEFINITIONS[role]["task_name"],
        "task_type": "controller",
        "parent_task_id": target["parents"][role],
        "queue": queue,
        "queue_id": queue_id,
        "source_sha256": target["sources"][role]["deployed_sha256"],
        "parameters_sha256": _content_sha256(parameters),
        "artifact_inventory": artifacts,
        "model_inventory": {"input": [], "output": []},
        "tags": tags,
        "system_tags": system_tags,
        "service_environment": shared._service_environment(
            task, context=context
        ),
    }


def execute_recovery(
    task_class: object,
    *,
    attempt: Mapping[str, object],
    attempt_path: Path,
    successor_deployment_receipt: Path,
    sleeper: Callable[[float], None],
    prior_recovery_receipt: Path | None = None,
) -> dict[str, object]:
    _validate_attempt(attempt)
    frozen_attempt, resolved_attempt = shared._read_sealed(
        attempt_path, context="successor runtime recovery attempt"
    )
    if frozen_attempt != dict(attempt) or resolved_attempt != attempt_path.resolve():
        raise SuccessorRuntimeRecoveryError("attempt was not durably frozen")
    deployment, resolved_deployment, ffnet_binding = _load_deployment(
        successor_deployment_receipt
    )
    if (
        str(resolved_deployment)
        != attempt["original_successor_deployment"]["path"]
        or deployment["seal_sha256"]
        != attempt["original_successor_deployment"]["receipt_seal_sha256"]
    ):
        raise SuccessorRuntimeRecoveryError("deployment binding changed")
    prior_recovery = None
    if prior_recovery_receipt is not None:
        prior_recovery = validate_successor_runtime_recovery_receipt(
            prior_recovery_receipt
        )
    if prior_recovery != attempt.get("prior_successor_runtime_recovery"):
        raise SuccessorRuntimeRecoveryError(
            "prior successor recovery changed after freeze"
        )
    target = _target_contract(deployment)
    attempted_target = attempt["target"]
    comparable = {
        "parents": target["parents"],
        "sources": target["sources"],
        "parameters": target["parameters"],
        "standalone_import_smoke": target["standalone_import_smoke"],
    }
    if _canonical_json(comparable) != _canonical_json(attempted_target):
        raise SuccessorRuntimeRecoveryError("local target changed after freeze")
    dependencies_before_reset = _dependency_authority(
        task_class, deployment, ffnet_binding
    )
    if _canonical_json(dependencies_before_reset) != _canonical_json(
        attempt.get("dependency_authority")
    ):
        raise SuccessorRuntimeRecoveryError(
            "P/reference/formal-core authority changed after freeze"
        )
    # A single no-side-effect barrier must pass for all four tasks before the
    # first reset.  _reset_one retains a second immediate per-role readback.
    failed_before_reset = {
        role: _failed_snapshot(
            task_class,
            role=role,
            deployment=deployment,
            prior_recovery=prior_recovery,
        )
        for role in CONFIGURE_ORDER
    }
    if _canonical_json(failed_before_reset) != _canonical_json(
        attempt.get("pre_reset_authority")
    ):
        raise SuccessorRuntimeRecoveryError(
            "successor failure authority changed after freeze"
        )
    reset_shells: dict[str, object] = {}
    for role in RESET_ORDER:
        reset_shells[role] = _reset_one(
            task_class,
            role=role,
            frozen=attempt["pre_reset_authority"][role],
            deployment=deployment,
            sleeper=sleeper,
            prior_recovery=prior_recovery,
        )
    for role in CONFIGURE_ORDER:
        deploy._configure_shell(
            reset_shells[role],
            role=role,
            parent_id=str(target["parents"][role]),
            source=str(target["rendered_sources"][role]),
            parameters=target["raw_parameters"][role],
        )
        shared._set_tags(
            reset_shells[role],
            list(attempt["pre_reset_authority"][role]["tags"]),
            context=f"successor {role}",
        )
    deploy._assert_sources_unchanged(
        target["base_sources"], target["embedded_sources"]
    )
    created = {
        role: _created_authority(
            task_class,
            role=role,
            target=target,
            frozen=attempt["pre_reset_authority"][role],
        )
        for role in CONFIGURE_ORDER
    }
    # Required all-created barrier immediately before the first enqueue.
    for role in CONFIGURE_ORDER:
        if _created_authority(
            task_class,
            role=role,
            target=target,
            frozen=attempt["pre_reset_authority"][role],
        ) != created[role]:
            raise SuccessorRuntimeRecoveryError(
                f"successor {role} all-created readbacks differ"
            )
    dependencies_before_enqueue = _reference_and_provenance_authority(
        task_class, deployment
    )
    expected_references = {
        key: attempt["dependency_authority"][key]
        for key in ("references", "completed_provenance")
    }
    if _canonical_json(dependencies_before_enqueue) != _canonical_json(
        expected_references
    ):
        raise SuccessorRuntimeRecoveryError(
            "P/reference authority changed before enqueue"
        )
    deploy._assert_sources_unchanged(
        target["base_sources"], target["embedded_sources"]
    )
    queued: dict[str, object] = {}
    for role in ENQUEUE_ORDER:
        task = shared._authoritative_task(
            task_class, SUCCESSOR_TASK_IDS[role], context=f"successor {role} enqueue"
        )
        deploy._enqueue(
            task_class,
            task,
            role=role,
            task_id=SUCCESSOR_TASK_IDS[role],
            sleeper=sleeper,
        )
        queued[role] = _live_authority(task_class, role=role, target=target)
    final = {
        role: _live_authority(task_class, role=role, target=target)
        for role in CONFIGURE_ORDER
    }
    final_names: dict[str, list[str]] = {}
    for role in CONFIGURE_ORDER:
        task = shared._authoritative_task(
            task_class, SUCCESSOR_TASK_IDS[role], context=f"successor {role} final"
        )
        final_names[role] = _exact_name_inventory(
            task_class,
            role=role,
            task=task,
            parent_id=str(target["parents"][role]),
        )
        if final_names[role] != attempt["pre_reset_authority"][role][
            "exact_name_inventory"
        ]:
            raise SuccessorRuntimeRecoveryError(
                f"successor {role} exact-name inventory changed"
            )
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": RECOVERY_RECEIPT_TYPE,
            "generated_at_utc": shared._utc_timestamp(),
            "status": "recovered_and_enqueued",
            "remote_state_changed": True,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "original_successor_deployment": dict(
                attempt["original_successor_deployment"]
            ),
            "prior_successor_runtime_recovery": prior_recovery,
            "attempt_receipt_path": str(resolved_attempt),
            "attempt_receipt_seal_sha256": attempt["seal_sha256"],
            "task_ids": dict(SUCCESSOR_TASK_IDS),
            "reset_order": list(RESET_ORDER),
            "configure_order": list(CONFIGURE_ORDER),
            "enqueue_order": list(ENQUEUE_ORDER),
            "sources": target["sources"],
            "parameters": target["parameters"],
            "parents": target["parents"],
            "standalone_import_smoke": target["standalone_import_smoke"],
            "dependency_authority_before_reset": dependencies_before_reset,
            "pre_reset_authority_revalidated": failed_before_reset,
            "reference_authority_before_enqueue": dependencies_before_enqueue,
            "created_unqueued_authority": created,
            "queued_authority": queued,
            "final_authority": final,
            "exact_name_inventories": final_names,
            "no_new_exact_name_task_ids": True,
        }
    )


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SuccessorRuntimeRecoveryError(f"{context} is not an object")
    return dict(value)


def _allowed_artifacts(row: Mapping[str, object], role: str) -> bool:
    items = row.get("artifact_inventory")
    return isinstance(items, list) and all(
        isinstance(item, Mapping)
        and str(item.get("name") or "") in deploy.ROLE_OUTPUT_ARTIFACTS[role]
        for item in items
    )


def _load_validated_recovery(
    path: Path,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    value, resolved = shared._read_sealed(
        path, context="successor runtime recovery receipt"
    )
    expected = {
        "schema_version": 1,
        "receipt_type": RECOVERY_RECEIPT_TYPE,
        "status": "recovered_and_enqueued",
        "remote_state_changed": True,
        "same_ids_only": True,
        "replacement_tasks_created": False,
        "task_ids": SUCCESSOR_TASK_IDS,
        "reset_order": list(RESET_ORDER),
        "configure_order": list(CONFIGURE_ORDER),
        "enqueue_order": list(ENQUEUE_ORDER),
        "no_new_exact_name_task_ids": True,
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise SuccessorRuntimeRecoveryError("successor recovery identity drifted")
    attempt_path = value.get("attempt_receipt_path")
    if type(attempt_path) is not str or not attempt_path:
        raise SuccessorRuntimeRecoveryError("successor recovery attempt path is absent")
    attempt, attempt_resolved = shared._read_sealed(
        Path(attempt_path), context="successor runtime recovery attempt"
    )
    if str(attempt_resolved) != attempt_path:
        raise SuccessorRuntimeRecoveryError(
            "successor recovery attempt path is not canonical"
        )
    _validate_attempt(attempt)
    if value.get("attempt_receipt_seal_sha256") != attempt.get("seal_sha256"):
        raise SuccessorRuntimeRecoveryError("successor recovery attempt seal drifted")
    original_binding = _mapping(
        value.get("original_successor_deployment"),
        context="original deployment binding",
    )
    if (
        original_binding != attempt.get("original_successor_deployment")
        or set(original_binding) != {"path", "receipt_seal_sha256"}
    ):
        raise SuccessorRuntimeRecoveryError(
            "original deployment binding drifted"
        )
    deployment, deployment_resolved, _ = _load_deployment(
        Path(str(original_binding["path"]))
    )
    if (
        str(deployment_resolved) != original_binding["path"]
        or deployment.get("seal_sha256")
        != original_binding["receipt_seal_sha256"]
        or deployment.get("task_ids")
        != {"P": deploy.COMPLETED_PROVENANCE_TASK_ID, **SUCCESSOR_TASK_IDS}
    ):
        raise SuccessorRuntimeRecoveryError(
            "original deployment path, seal, or task IDs drifted"
        )
    target = _mapping(attempt.get("target"), context="attempt target")
    if set(target) != {
        "parents",
        "sources",
        "parameters",
        "standalone_import_smoke",
    }:
        raise SuccessorRuntimeRecoveryError("attempt target key inventory drifted")
    parents = _mapping(target["parents"], context="target parents")
    sources = _mapping(target["sources"], context="target sources")
    parameters = _mapping(target["parameters"], context="target parameters")
    if (
        set(parents) != set(CONFIGURE_ORDER)
        or set(sources) != set(deploy.ROLE_ORDER)
        or set(parameters) != set(CONFIGURE_ORDER)
        or parents != deployment.get("parents")
        or parameters != deployment.get("parameters")
    ):
        raise SuccessorRuntimeRecoveryError(
            "target parent or parameter graph drifted"
        )
    if (
        value.get("parents") != parents
        or value.get("sources") != sources
        or value.get("parameters") != parameters
        or value.get("standalone_import_smoke")
        != target["standalone_import_smoke"]
    ):
        raise SuccessorRuntimeRecoveryError(
            "recovery target does not match frozen attempt"
        )
    if (
        value.get("dependency_authority_before_reset")
        != attempt.get("dependency_authority")
        or value.get("pre_reset_authority_revalidated")
        != attempt.get("pre_reset_authority")
    ):
        raise SuccessorRuntimeRecoveryError(
            "pre-reset authority does not match frozen attempt"
        )
    dependency = _mapping(
        attempt.get("dependency_authority"), context="attempt dependency authority"
    )
    expected_pre_enqueue = {
        key: dependency[key]
        for key in ("references", "completed_provenance")
    }
    if value.get("reference_authority_before_enqueue") != expected_pre_enqueue:
        raise SuccessorRuntimeRecoveryError(
            "pre-enqueue reference authority drifted"
        )
    pre = _mapping(
        attempt.get("pre_reset_authority"), context="pre-reset authority"
    )
    exact_names = _mapping(
        value.get("exact_name_inventories"),
        context="exact-name inventories",
    )
    created_rows = _mapping(
        value.get("created_unqueued_authority"), context="created authority"
    )
    queued_rows = _mapping(
        value.get("queued_authority"), context="queued authority"
    )
    final_rows = _mapping(
        value.get("final_authority"), context="final authority"
    )
    expected_roles = set(CONFIGURE_ORDER)
    if any(
        set(rows) != expected_roles
        for rows in (pre, exact_names, created_rows, queued_rows, final_rows)
    ):
        raise SuccessorRuntimeRecoveryError(
            "successor recovery role inventory drifted"
        )
    for role in CONFIGURE_ORDER:
        source = _mapping(sources.get(role), context=f"{role} source")
        role_parameters = _mapping(
            parameters.get(role), context=f"{role} parameters"
        )
        created = _mapping(created_rows.get(role), context=f"{role} created")
        queued = _mapping(queued_rows.get(role), context=f"{role} queued")
        final = _mapping(final_rows.get(role), context=f"{role} final")
        source_sha = source.get("deployed_sha256")
        parameters_sha = _content_sha256(
            {str(key): str(item) for key, item in role_parameters.items()}
        )
        stable = {
            "task_id": SUCCESSOR_TASK_IDS[role],
            "project_id": shared.evaluation_recovery.PROJECT_ID,
            "task_name": deploy.ROLE_DEFINITIONS[role]["task_name"],
            "task_type": "controller",
            "parent_task_id": parents[role],
            "source_sha256": source_sha,
            "parameters_sha256": parameters_sha,
        }
        if any(
            any(row.get(key) != wanted for key, wanted in stable.items())
            for row in (created, queued, final)
        ):
            raise SuccessorRuntimeRecoveryError(
                f"successor recovery {role} stable authority drifted"
            )
        frozen_role = _mapping(pre.get(role), context=f"{role} pre-reset")
        expected_tags = frozen_role.get("tags")
        expected_system_tags = frozen_role.get("system_tags")
        if (
            created.get("status") != "created"
            or created.get("queue") is not None
            or created.get("queue_id") is not None
            or created.get("artifact_inventory") != []
            or created.get("model_inventory") != {"input": [], "output": []}
            or created.get("tags") != expected_tags
            or created.get("system_tags") != expected_system_tags
            or not isinstance(created.get("service_environment"), Mapping)
        ):
            raise SuccessorRuntimeRecoveryError(
                f"successor recovery {role} created barrier drifted"
            )
        for stage, row in (("queued", queued), ("final", final)):
            status = row.get("status")
            artifact_names = {
                str(item.get("name") or "")
                for item in row.get("artifact_inventory", [])
                if isinstance(item, Mapping)
            }
            if (
                status not in {"queued", "in_progress", "completed"}
                or row.get("queue") != deploy.SERVICES_QUEUE
                or row.get("queue_id") != deploy.SERVICES_QUEUE_ID
                or row.get("model_inventory") != {"input": [], "output": []}
                or row.get("tags") != expected_tags
                or row.get("system_tags") != expected_system_tags
                or not isinstance(row.get("service_environment"), Mapping)
                or not _allowed_artifacts(row, role)
                or (
                    status == "completed"
                    and artifact_names != set(deploy.ROLE_OUTPUT_ARTIFACTS[role])
                )
            ):
                raise SuccessorRuntimeRecoveryError(
                    f"successor recovery {role} {stage} authority drifted"
                )
        if exact_names.get(role) != frozen_role.get("exact_name_inventory"):
            raise SuccessorRuntimeRecoveryError(
                f"successor recovery {role} exact-name inventory drifted"
            )
    return value, resolved, attempt


def validate_successor_runtime_recovery_receipt(path: Path) -> dict[str, object]:
    value, resolved, _ = _load_validated_recovery(path)
    return {
        "path": str(resolved),
        "receipt_seal_sha256": value["seal_sha256"],
        "original_successor_deployment": dict(
            value["original_successor_deployment"]
        ),
        "attempt_receipt_seal_sha256": value["attempt_receipt_seal_sha256"],
        "task_ids": dict(value["task_ids"]),
        "source_sha256": {
            role: value["sources"][role]["deployed_sha256"]
            for role in CONFIGURE_ORDER
        },
        "parameters_sha256": {
            role: _content_sha256(value["parameters"][role])
            for role in CONFIGURE_ORDER
        },
        "same_ids_only": True,
        "replacement_tasks_created": False,
    }


def effective_successor_authority(
    path: Path, *, original_deployment_receipt: Mapping[str, object]
) -> dict[str, object]:
    value, _, _ = _load_validated_recovery(path)
    binding = validate_successor_runtime_recovery_receipt(path)
    if (
        binding["original_successor_deployment"]["receipt_seal_sha256"]
        != original_deployment_receipt.get("seal_sha256")
        or original_deployment_receipt.get("task_ids")
        != {"P": deploy.COMPLETED_PROVENANCE_TASK_ID, **SUCCESSOR_TASK_IDS}
    ):
        raise SuccessorRuntimeRecoveryError(
            "successor recovery does not bind the supplied deployment"
        )
    return _seal(
        {
            "schema_version": 1,
            "authority_type": EFFECTIVE_AUTHORITY_TYPE,
            "recovery_binding": binding,
            "original_successor_deployment": dict(
                value["original_successor_deployment"]
            ),
            "task_ids": {
                "P": deploy.COMPLETED_PROVENANCE_TASK_ID,
                **SUCCESSOR_TASK_IDS,
            },
            "parents": dict(value["parents"]),
            "sources": {
                role: value["sources"][role]["deployed_sha256"]
                for role in CONFIGURE_ORDER
            },
            "parameters": {
                role: {
                    "values": dict(value["parameters"][role]),
                    "sha256": _content_sha256(value["parameters"][role]),
                }
                for role in CONFIGURE_ORDER
            },
            "roles": {
                role: dict(value["final_authority"][role])
                for role in CONFIGURE_ORDER
            },
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--successor-deployment-receipt", type=Path, required=True)
    parser.add_argument("--prior-recovery-receipt", type=Path)
    parser.add_argument("--attempt-receipt", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.execute_token != EXECUTE_TOKEN:
        raise SuccessorRuntimeRecoveryError(
            f"exact execute token required: {EXECUTE_TOKEN}"
        )
    if not args.execute and args.execute_token:
        raise SuccessorRuntimeRecoveryError("--execute-token requires --execute")
    if args.execute != (args.attempt_receipt is not None):
        raise SuccessorRuntimeRecoveryError(
            "--attempt-receipt is required exactly with --execute"
        )
    outputs = [args.receipt]
    if args.attempt_receipt is not None:
        outputs.append(args.attempt_receipt)
    if len({item.resolve() for item in outputs}) != len(outputs):
        raise SuccessorRuntimeRecoveryError("attempt and final paths must differ")
    for path in outputs:
        _require_new(path, context="successor runtime recovery output")
    try:
        from clearml import Task
    except ImportError as error:
        raise SuccessorRuntimeRecoveryError("ClearML is required") from error
    attempt = build_attempt(
        Task,
        successor_deployment_receipt=args.successor_deployment_receipt,
        prior_recovery_receipt=args.prior_recovery_receipt,
    )
    if not args.execute:
        plan = _seal(
            {
                **attempt,
                "receipt_type": PLAN_RECEIPT_TYPE,
                "status": "planned",
                "execute_token": EXECUTE_TOKEN,
            }
        )
        _write_new(args.receipt, plan)
        print(json.dumps({"receipt": str(args.receipt), "seal_sha256": plan["seal_sha256"]}))
        return 0
    assert args.attempt_receipt is not None
    _write_new(args.attempt_receipt, attempt)
    try:
        result = execute_recovery(
            Task,
            attempt=attempt,
            attempt_path=args.attempt_receipt,
            successor_deployment_receipt=args.successor_deployment_receipt,
            prior_recovery_receipt=args.prior_recovery_receipt,
            sleeper=shared.time.sleep,
        )
    except Exception as error:
        failure = _seal(
            {
                "schema_version": 1,
                "receipt_type": RECOVERY_RECEIPT_TYPE,
                "generated_at_utc": shared._utc_timestamp(),
                "status": "failed_closed",
                "remote_state_changed": True,
                "attempt_receipt_path": str(args.attempt_receipt.resolve()),
                "attempt_receipt_seal_sha256": attempt["seal_sha256"],
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_new(args.receipt, failure)
        raise
    _write_new(args.receipt, result)
    print(json.dumps({"receipt": str(args.receipt), "seal_sha256": result["seal_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ATTEMPT_RECEIPT_TYPE",
    "EXECUTE_TOKEN",
    "RECOVERY_RECEIPT_TYPE",
    "SUCCESSOR_TASK_IDS",
    "SuccessorRuntimeRecoveryError",
    "build_attempt",
    "effective_successor_authority",
    "execute_recovery",
    "main",
    "validate_successor_runtime_recovery_receipt",
)
