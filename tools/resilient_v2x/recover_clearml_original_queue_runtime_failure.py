#!/usr/bin/env python3
"""Recover the exact FFNet/C tasks while retaining the sealed V100 plan.

This recovery is deliberately split into two remotely-mutating phases:

``ffnet``
    Freeze the failed FFNet attempt and all non-target evaluation evidence,
    reset/reinstall only task ``144397...``, and leave it created+unqueued.

``candidate``
    After a fresh W/L/A/S chain has been deployed from the completed P task,
    freeze that successor authority, reset/reinstall only controller
    ``b6f0...``, and enqueue it to ``services``.

The default mode is read-only and writes a sealed local plan.  Every execute
mode requires its exact token and distinct write-once attempt/final receipts.
No task is cloned and the formal evaluation plan remains GPU4-V100 for FFNet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from tools.resilient_v2x import (
        deploy_clearml_formal_candidate_evaluation_queue as candidate_deploy,
    )
    from tools.resilient_v2x import (
        deploy_clearml_formal_successor_chain as successor_deploy,
    )
    from tools.resilient_v2x import (
        recover_clearml_exact_eval_chain_runtime_failure as shared,
    )
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import deploy_clearml_formal_candidate_evaluation_queue as candidate_deploy
    import deploy_clearml_formal_successor_chain as successor_deploy
    import recover_clearml_exact_eval_chain_runtime_failure as shared


FFNET_TASK_ID = shared.FFNET_TASK_ID
COFORMER_TASK_ID = shared.COFORMER_TASK_ID
CANDIDATE_CONTROLLER_TASK_ID = shared.CANDIDATE_CONTROLLER_TASK_ID
ORIGINAL_QUEUE = "GPU4-V100"
ORIGINAL_QUEUE_ID = shared.formal.EXPECTED_QUEUE_IDS[ORIGINAL_QUEUE]
FFNET_EXECUTE_TOKEN = (
    "RESET_EXACT_FFNET_ON_ORIGINAL_GPU4_V100_"
    "144397BFA9C242BC9A92A1279922558B"
)
CANDIDATE_EXECUTE_TOKEN = (
    "RESET_EXACT_CANDIDATE_CONTROLLER_AFTER_ORIGINAL_WLAS_"
    "B6F0FBAB32A5478183A45B3CA833FC01"
)
FFNET_ATTEMPT_TYPE = "resilient_v2x_original_queue_ffnet_recovery_attempt"
FFNET_PLAN_TYPE = "resilient_v2x_original_queue_ffnet_recovery_plan"
FFNET_RECEIPT_TYPE = "resilient_v2x_original_queue_ffnet_recovery"
CANDIDATE_ATTEMPT_TYPE = (
    "resilient_v2x_original_queue_candidate_recovery_attempt"
)
CANDIDATE_PLAN_TYPE = "resilient_v2x_original_queue_candidate_recovery_plan"
CANDIDATE_RECEIPT_TYPE = "resilient_v2x_original_queue_candidate_recovery"
COFORMER_ARTIFACTS = frozenset(
    {
        "run_contract",
        "evaluation_plan",
        "controlled_baseline_metrics",
        "controlled_baseline_evidence",
    }
)


class OriginalQueueRecoveryError(RuntimeError):
    """Raised when an unchanged-plan recovery invariant cannot be proven."""


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    shared._write_new(path, value)


def _require_new(path: Path, *, context: str) -> None:
    shared._require_new(path, context=context)


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    return shared._seal(value)


def _content_sha256(value: object) -> str:
    return shared._content_sha256(value)


def _coformer_completed_record(
    evaluations: Mapping[str, object],
) -> dict[str, object]:
    rows = evaluations.get("records")
    if not isinstance(rows, list):
        raise OriginalQueueRecoveryError("evaluation inventory is unavailable")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("task_id") == COFORMER_TASK_ID
    ]
    if len(matches) != 1:
        raise OriginalQueueRecoveryError("CoFormer exact task evidence is absent")
    source = dict(matches[0])
    row = {
        key: source.get(key)
        for key in (
            "task_id",
            "subject",
            "project_id",
            "parent_task_id",
            "task_name",
            "task_type",
            "status",
            "queue",
            "queue_id",
            "script_sha256",
            "parameters_sha256",
            "input_model_ids",
            "output_model_ids",
            "tags",
            "system_tags",
        )
        if key in source
    }
    artifacts = source.get("artifact_inventory")
    if (
        row.get("status") != "completed"
        or row.get("queue") != "GPU4-A100"
        or not isinstance(artifacts, list)
        or len(artifacts) != len(COFORMER_ARTIFACTS)
        or {str(item.get("name") or "") for item in artifacts if isinstance(item, Mapping)}
        != COFORMER_ARTIFACTS
    ):
        raise OriginalQueueRecoveryError(
            "CoFormer must be completed with the exact four formal artifacts"
        )
    normalized: list[dict[str, object]] = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise OriginalQueueRecoveryError("CoFormer artifact record is invalid")
        digest = str(item.get("hash") or "")
        size = item.get("size_bytes")
        url = str(item.get("url") or "")
        if (
            shared.SHA256_RE.fullmatch(digest) is None
            or type(size) is not int
            or size <= 0
            or not url
        ):
            raise OriginalQueueRecoveryError(
                "CoFormer artifact hash/size/URL evidence is incomplete"
            )
        normalized.append(
            {
                "name": str(item["name"]),
                "hash": digest,
                "size_bytes": size,
                "url": url,
            }
        )
    normalized.sort(key=lambda item: str(item["name"]))
    row["artifact_inventory"] = normalized
    row["artifact_names"] = [str(item["name"]) for item in normalized]
    row["artifact_inventory_sha256"] = _content_sha256(normalized)
    return row


def _old_bindings(
    *,
    evaluation_recovery_receipt: Path,
    successor_deployment_receipt: Path,
    controller_recovery_receipt: Path,
    candidate_deployment_receipt: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    evaluation = shared._evaluation_binding(evaluation_recovery_receipt)
    successor, successor_receipt = shared._successor_binding(
        successor_deployment_receipt, evaluation_binding=evaluation
    )
    controller = shared._controller_recovery_binding(controller_recovery_receipt)
    candidate, _ = shared._candidate_deployment_binding(
        candidate_deployment_receipt,
        evaluation_binding=evaluation,
        controller_binding=controller,
    )
    return (
        {
            "evaluation_recovery": evaluation,
            "failed_successor_deployment": successor,
            "candidate_controller_recovery": controller,
            "failed_candidate_deployment": candidate,
        },
        successor_receipt,
    )


def _validate_original_evaluations(
    task_class: object, contracts: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    adjusted = [dict(contract) for contract in contracts]
    coformer = [
        contract for contract in adjusted if contract["task_id"] == COFORMER_TASK_ID
    ]
    if len(coformer) != 1 or coformer[0].get("input_model_ids") != []:
        raise OriginalQueueRecoveryError(
            "CoFormer sealed creation input-model contract drifted"
        )
    task = shared._authoritative_task(
        task_class, COFORMER_TASK_ID, context="completed CoFormer runtime binding"
    )
    if shared._status(task) != "completed":
        raise OriginalQueueRecoveryError("CoFormer is not completed")
    parameters = shared._parameters(task, context="completed CoFormer runtime binding")
    model_id = str(parameters.get("Args/controlled_baseline_model_id") or "")
    models = shared._model_inventory(
        task, context="completed CoFormer runtime binding"
    )
    if (
        len(model_id) != 32
        or models != {"input": [model_id], "output": []}
        or shared._evaluation_parameters_sha256(parameters)
        != coformer[0]["parameters_sha256"]
    ):
        raise OriginalQueueRecoveryError(
            "CoFormer runtime input is not the sealed controlled-baseline model"
        )
    coformer[0]["input_model_ids"] = [model_id]
    evaluations = shared._validate_evaluations(task_class, adjusted)
    row = evaluations["coformer_preservation"]
    if (
        row.get("status") != "completed"
        or row.get("input_model_ids") != [model_id]
        or set(row.get("artifact_names", [])) != COFORMER_ARTIFACTS
    ):
        raise OriginalQueueRecoveryError(
            "completed CoFormer monotonic runtime evidence is incomplete"
        )
    return evaluations


def build_ffnet_attempt(
    task_class: object,
    *,
    evaluation_recovery_receipt: Path,
    successor_deployment_receipt: Path,
    controller_recovery_receipt: Path,
    candidate_deployment_receipt: Path,
) -> dict[str, object]:
    bindings, successor_receipt = _old_bindings(
        evaluation_recovery_receipt=evaluation_recovery_receipt,
        successor_deployment_receipt=successor_deployment_receipt,
        controller_recovery_receipt=controller_recovery_receipt,
        candidate_deployment_receipt=candidate_deployment_receipt,
    )
    old_w = shared._authoritative_task(
        task_class, shared.SUCCESSOR_TASK_IDS["W"], context="old W plan producer"
    )
    plan = shared._validate_w_plan(old_w)
    contracts = shared._evaluation_contracts(
        evaluation_recovery_receipt, plan=plan
    )
    evaluations = _validate_original_evaluations(task_class, contracts)
    coformer = _coformer_completed_record(evaluations)
    ffnet_contract = next(
        item for item in contracts if item["task_id"] == FFNET_TASK_ID
    )
    ffnet, _ = shared._target_snapshot(
        task_class,
        label="FFNet",
        entry_point="clearml_5090_bootstrap.py",
        parent_task_id=str(ffnet_contract["parent_task_id"]),
        task_name=str(ffnet_contract["task_name"]),
        parameters=None,
        allowed_artifacts=frozenset(),
    )
    if ffnet["queue"] != ORIGINAL_QUEUE or ffnet["queue_id"] != ORIGINAL_QUEUE_ID:
        raise OriginalQueueRecoveryError("FFNet failed attempt is not on GPU4-V100")
    ffnet["pre_reset_queue"] = ffnet["queue"]
    ffnet["pre_reset_queue_id"] = ffnet["queue_id"]
    ffnet["pre_reset_tags"] = list(ffnet["tags"])
    ffnet["tags"] = [
        value for value in ffnet["tags"] if value != "dependency-released"
    ]
    candidate, _ = shared._target_snapshot(
        task_class,
        label="C",
        entry_point=candidate_deploy.ENTRY_POINT,
        parent_task_id=candidate_deploy.PARENT_TASK_ID,
        task_name=candidate_deploy.CONTROLLER_NAME,
        parameters=None,
        allowed_artifacts=frozenset(),
    )
    if set(candidate["tags"]) != candidate_deploy.TAGS:
        raise OriginalQueueRecoveryError("candidate controller tag contract drifted")
    parents = successor_receipt["parents"]
    parameters = successor_receipt["parameters"]
    old_successors: dict[str, object] = {}
    for role in ("W", "L", "A", "S"):
        material, _ = shared._target_snapshot(
            task_class,
            label=role,
            entry_point=str(successor_deploy.ROLE_DEFINITIONS[role]["entry_point"]),
            parent_task_id=str(parents[role]),
            task_name=str(successor_deploy.ROLE_DEFINITIONS[role]["task_name"]),
            parameters=parameters[role],
            allowed_artifacts=(
                frozenset({shared.formal.FORMAL_EVALUATION_PLAN_ARTIFACT})
                if role == "W"
                else frozenset()
            ),
        )
        old_successors[role] = {
            key: value
            for key, value in material.items()
            if key not in {"source", "parameters"}
        }
    names = {
        label: shared._exact_name_inventory(
            task_class,
            task_name=str(material["task_name"]),
            parent_task_id=str(material["parent_task_id"]),
            expected_task_id=(
                FFNET_TASK_ID if label == "FFNet" else CANDIDATE_CONTROLLER_TASK_ID
            ),
        )
        for label, material in {"FFNet": ffnet, "C": candidate}.items()
    }
    history = {
        label: shared._same_name_history_snapshot(
            task_class, label=label, inventory=names[label]
        )
        for label in ("FFNet", "C")
    }
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": FFNET_ATTEMPT_TYPE,
            "generated_at_utc": shared._utc_timestamp(),
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "plan_policy": "retain_original_gpu4_v100_no_amendment",
            "bindings": bindings,
            "evaluation_plan": {
                "producer_task_id": shared.SUCCESSOR_TASK_IDS["W"],
                "seal_sha256": shared.EXPECTED_W_PLAN_SEAL,
                "entry_count": len(plan["entries"]),
                "ffnet_queue": ORIGINAL_QUEUE,
            },
            "evaluation_inventory": evaluations,
            "coformer_completed_evidence": coformer,
            "ffnet_material": ffnet,
            "candidate_preserved_material": candidate,
            "old_successor_failure_evidence": old_successors,
            "exact_name_inventories": names,
            "preexisting_same_name_tasks": history,
            "planned_actions": [
                "durably_write_attempt_receipt",
                "freeze_completed_coformer_hash_size_status",
                "reset_reinstall_exact_ffnet_only",
                "leave_ffnet_created_unqueued",
                "preserve_candidate_failed_until_fresh_wlas",
                "preserve_all_other_exact_evaluations",
            ],
        }
    )


def _validate_ffnet_attempt(value: Mapping[str, object]) -> None:
    expected = {
        "schema_version": 1,
        "receipt_type": FFNET_ATTEMPT_TYPE,
        "status": "frozen_before_reset",
        "remote_state_changed": False,
        "same_ids_only": True,
        "replacement_tasks_created": False,
        "plan_policy": "retain_original_gpu4_v100_no_amendment",
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise OriginalQueueRecoveryError(f"FFNet attempt {key} drifted")
    ffnet = value.get("ffnet_material")
    if (
        not isinstance(ffnet, Mapping)
        or ffnet.get("task_id") != FFNET_TASK_ID
        or ffnet.get("queue") != ORIGINAL_QUEUE
        or ffnet.get("queue_id") != ORIGINAL_QUEUE_ID
        or ffnet.get("pre_reset_queue") != ORIGINAL_QUEUE
        or ffnet.get("pre_reset_queue_id") != ORIGINAL_QUEUE_ID
        or "dependency-released" in ffnet.get("tags", [])
        or hashlib.sha256(str(ffnet.get("source") or "").encode()).hexdigest()
        != shared.EXPECTED_SOURCE_SHA256["FFNet"]
        or _content_sha256(ffnet.get("parameters"))
        != shared.EXPECTED_PARAMETERS_SHA256["FFNet"]
    ):
        raise OriginalQueueRecoveryError("FFNet unchanged-plan material drifted")
    coformer = value.get("coformer_completed_evidence")
    if not isinstance(coformer, Mapping):
        raise OriginalQueueRecoveryError("CoFormer completed evidence is absent")
    _coformer_completed_record(
        {"records": [coformer]}
    )


def _candidate_still_failed(
    task_class: object, material: Mapping[str, object]
) -> dict[str, object]:
    shared._validate_failed_target_against_material(
        task_class,
        label="C",
        material=material,
        context="candidate preserved failed target",
    )
    result = {
        key: value
        for key, value in material.items()
        if key not in {"source", "parameters"}
    }
    result["status"] = "failed"
    return result


def execute_ffnet_recovery(
    task_class: object,
    *,
    attempt: Mapping[str, object],
    attempt_path: Path,
    sleeper: object,
) -> dict[str, object]:
    _validate_ffnet_attempt(attempt)
    frozen, resolved = shared._read_sealed(
        attempt_path, context="original queue FFNet attempt"
    )
    if resolved != attempt_path.resolve() or frozen != dict(attempt):
        raise OriginalQueueRecoveryError("FFNet attempt was not durably frozen")
    ffnet = attempt["ffnet_material"]
    candidate = attempt["candidate_preserved_material"]
    names = attempt["exact_name_inventories"]
    current_names = {
        label: shared._exact_name_inventory(
            task_class,
            task_name=str(material["task_name"]),
            parent_task_id=str(material["parent_task_id"]),
            expected_task_id=(
                FFNET_TASK_ID if label == "FFNet" else CANDIDATE_CONTROLLER_TASK_ID
            ),
        )
        for label, material in {"FFNet": ffnet, "C": candidate}.items()
    }
    if current_names != names:
        raise OriginalQueueRecoveryError("exact-name inventory changed before reset")
    old_w = shared._authoritative_task(
        task_class, shared.SUCCESSOR_TASK_IDS["W"], context="old W plan recheck"
    )
    if shared._validate_w_plan(old_w).get("seal_sha256") != (
        shared.EXPECTED_W_PLAN_SEAL
    ):
        raise OriginalQueueRecoveryError("original W plan drifted")
    shared._revalidate_old_successor_failures(
        task_class, evidence=attempt["old_successor_failure_evidence"]
    )
    _candidate_still_failed(task_class, candidate)
    preserved_before = shared._revalidate_preserved_evaluations(
        task_class, attempt_receipt=attempt
    )
    if _coformer_completed_record({"records": preserved_before["records"]}) != (
        attempt["coformer_completed_evidence"]
    ):
        raise OriginalQueueRecoveryError("CoFormer changed before FFNet reset")
    created = shared._reset_and_install(
        task_class,
        label="FFNet",
        material=ffnet,
        sleeper=sleeper,
    )
    if (
        created.get("status") != "created"
        or created.get("queue") is not None
        or created.get("queue_id") is not None
    ):
        raise OriginalQueueRecoveryError("FFNet was not left created+unqueued")
    preserved_final = shared._revalidate_preserved_evaluations(
        task_class, attempt_receipt=attempt
    )
    coformer_final = _coformer_completed_record(
        {"records": preserved_final["records"]}
    )
    if coformer_final != attempt["coformer_completed_evidence"]:
        raise OriginalQueueRecoveryError("CoFormer changed during FFNet recovery")
    candidate_final = _candidate_still_failed(task_class, candidate)
    old_successors = shared._revalidate_old_successor_failures(
        task_class, evidence=attempt["old_successor_failure_evidence"]
    )
    final_names = {
        label: shared._exact_name_inventory(
            task_class,
            task_name=str(material["task_name"]),
            parent_task_id=str(material["parent_task_id"]),
            expected_task_id=(
                FFNET_TASK_ID if label == "FFNet" else CANDIDATE_CONTROLLER_TASK_ID
            ),
        )
        for label, material in {"FFNet": ffnet, "C": candidate}.items()
    }
    if final_names != names:
        raise OriginalQueueRecoveryError("recovery created a replacement task")
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": FFNET_RECEIPT_TYPE,
            "generated_at_utc": shared._utc_timestamp(),
            "status": "ffnet_recovered_created_unqueued",
            "remote_state_changed": True,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "plan_policy": "retain_original_gpu4_v100_no_amendment",
            "attempt_receipt_path": str(attempt_path.resolve()),
            "attempt_receipt_seal_sha256": attempt["seal_sha256"],
            "bindings": dict(attempt["bindings"]),
            "ffnet_task_id": FFNET_TASK_ID,
            "ffnet_planned_queue": ORIGINAL_QUEUE,
            "ffnet_authority": created,
            "candidate_controller_task_id": CANDIDATE_CONTROLLER_TASK_ID,
            "candidate_preserved_failed": candidate_final,
            "coformer_completed_evidence": coformer_final,
            "preserved_evaluations": preserved_final,
            "old_successors_preserved": old_successors,
            "exact_name_inventories": final_names,
            "coformer_mutated": False,
            "candidate_mutated": False,
            "ffnet_enqueued": False,
        }
    )


def validate_ffnet_recovery_receipt(path: Path) -> dict[str, object]:
    value, resolved = shared._read_sealed(
        path, context="original queue FFNet recovery receipt"
    )
    expected = {
        "schema_version": 1,
        "receipt_type": FFNET_RECEIPT_TYPE,
        "status": "ffnet_recovered_created_unqueued",
        "remote_state_changed": True,
        "same_ids_only": True,
        "replacement_tasks_created": False,
        "plan_policy": "retain_original_gpu4_v100_no_amendment",
        "ffnet_task_id": FFNET_TASK_ID,
        "ffnet_planned_queue": ORIGINAL_QUEUE,
        "candidate_controller_task_id": CANDIDATE_CONTROLLER_TASK_ID,
        "coformer_mutated": False,
        "candidate_mutated": False,
        "ffnet_enqueued": False,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise OriginalQueueRecoveryError(f"FFNet recovery receipt {key} drifted")
    attempt_path = value.get("attempt_receipt_path")
    if type(attempt_path) is not str:
        raise OriginalQueueRecoveryError("FFNet attempt path is invalid")
    attempt, attempt_resolved = shared._read_sealed(
        Path(attempt_path), context="original queue FFNet attempt"
    )
    _validate_ffnet_attempt(attempt)
    if (
        value.get("attempt_receipt_seal_sha256") != attempt["seal_sha256"]
        or attempt_resolved != Path(attempt_path).resolve()
    ):
        raise OriginalQueueRecoveryError("FFNet attempt binding drifted")
    authority = value.get("ffnet_authority")
    coformer = value.get("coformer_completed_evidence")
    if (
        not isinstance(authority, Mapping)
        or authority.get("task_id") != FFNET_TASK_ID
        or authority.get("status") != "created"
        or authority.get("queue") is not None
        or authority.get("queue_id") is not None
        or authority.get("source_sha256") != shared.EXPECTED_SOURCE_SHA256["FFNet"]
        or authority.get("parameters_sha256")
        != shared.EXPECTED_PARAMETERS_SHA256["FFNet"]
        or not isinstance(coformer, Mapping)
    ):
        raise OriginalQueueRecoveryError("FFNet recovery authority drifted")
    normalized_coformer = _coformer_completed_record({"records": [coformer]})
    if (
        normalized_coformer != coformer
        or coformer != attempt.get("coformer_completed_evidence")
    ):
        raise OriginalQueueRecoveryError("CoFormer recovery evidence drifted")
    return {
        "path": str(resolved),
        "receipt_seal_sha256": value["seal_sha256"],
        "attempt_receipt_seal_sha256": attempt["seal_sha256"],
        "ffnet_task_id": FFNET_TASK_ID,
        "ffnet_source_sha256": authority["source_sha256"],
        "ffnet_parameters_sha256": authority["parameters_sha256"],
        "ffnet_planned_queue": ORIGINAL_QUEUE,
        "ffnet_created_unqueued": True,
        "coformer_task_id": COFORMER_TASK_ID,
        "coformer_artifact_inventory_sha256": coformer[
            "artifact_inventory_sha256"
        ],
        "coformer_completed": True,
        "same_ids_only": True,
        "replacement_tasks_created": False,
    }


def _successor_binding(
    path: Path,
    *,
    evaluation_binding: Mapping[str, object],
    ffnet_binding: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    value, resolved = shared._read_sealed(
        path, context="fresh original-plan successor receipt"
    )
    if (
        value.get("schema_version") != 1
        or value.get("receipt_type")
        != "resilient_v2x_formal_successor_chain_deployment"
        or value.get("mode") != "completed_provenance_successor_execute"
        or value.get("status") != "deployed"
        or value.get("remote_state_changed") is not True
        or value.get("created_roles") != ["W", "L", "A", "S"]
        or value.get("enqueue_order") != ["W", "L", "A", "S"]
        or value.get("evaluation_recovery_receipt") != evaluation_binding
        or value.get("original_queue_runtime_recovery") != ffnet_binding
        or "evaluation_plan_amendment_receipt" in value
        or "exact_eval_chain_runtime_recovery" in value
    ):
        raise OriginalQueueRecoveryError("fresh successor receipt contract drifted")
    task_ids = value.get("task_ids")
    parents = value.get("parents")
    sources = value.get("sources")
    parameters = value.get("parameters")
    authority = value.get("queued_authority")
    if not all(
        isinstance(item, Mapping)
        for item in (task_ids, parents, sources, parameters, authority)
    ):
        raise OriginalQueueRecoveryError("fresh successor receipt is incomplete")
    if task_ids.get("P") != successor_deploy.COMPLETED_PROVENANCE_TASK_ID:
        raise OriginalQueueRecoveryError("fresh successor P binding drifted")
    ids = [str(task_ids.get(role) or "") for role in ("W", "L", "A", "S")]
    if (
        len(set(ids)) != 4
        or any(len(item) != 32 for item in ids)
        or dict(parents)
        != {
            "W": task_ids["P"],
            "L": task_ids["W"],
            "A": task_ids["L"],
            "S": task_ids["A"],
        }
    ):
        raise OriginalQueueRecoveryError("fresh successor identity graph drifted")
    for role in ("W", "L", "A", "S"):
        source = sources.get(role)
        row = authority.get(role)
        role_parameters = parameters.get(role)
        if (
            not isinstance(source, Mapping)
            or not isinstance(row, Mapping)
            or not isinstance(role_parameters, Mapping)
            or row.get("task_id") != task_ids[role]
            or row.get("parent_task_id") != parents[role]
            or row.get("queue") != successor_deploy.SERVICES_QUEUE
            or row.get("script_sha256") != source.get("deployed_sha256")
            or row.get("parameters_sha256")
            != _content_sha256(
                {str(key): str(item) for key, item in role_parameters.items()}
            )
        ):
            raise OriginalQueueRecoveryError(
                f"fresh successor {role} receipt authority drifted"
            )
    return (
        {
            "path": str(resolved),
            "receipt_seal_sha256": value["seal_sha256"],
            "task_ids": {role: str(task_ids[role]) for role in ("W", "L", "A", "S")},
            "evaluation_recovery_receipt_seal_sha256": evaluation_binding[
                "receipt_seal_sha256"
            ],
            "ffnet_recovery_receipt_seal_sha256": ffnet_binding[
                "receipt_seal_sha256"
            ],
        },
        value,
    )


def _live_successors(
    task_class: object, receipt: Mapping[str, object]
) -> dict[str, object]:
    task_ids = receipt["task_ids"]
    parents = receipt["parents"]
    sources = receipt["sources"]
    parameters = receipt["parameters"]
    result: dict[str, object] = {}
    for role in ("W", "L", "A", "S"):
        task = shared._authoritative_task(
            task_class, str(task_ids[role]), context=f"fresh successor {role}"
        )
        status = shared._status(task)
        source = shared._source(
            task,
            entry_point=str(successor_deploy.ROLE_DEFINITIONS[role]["entry_point"]),
            context=f"fresh successor {role}",
        )
        observed_parameters = shared._parameters(
            task, context=f"fresh successor {role}"
        )
        queue, queue_id = shared._queue_identity(
            task, context=f"fresh successor {role}"
        )
        artifacts = shared._artifact_inventory(task)
        if (
            status not in {"queued", "in_progress", "completed"}
            or str(getattr(task, "name", "") or "")
            != successor_deploy.ROLE_DEFINITIONS[role]["task_name"]
            or successor_deploy._task_parent(task) != parents[role]
            or queue != successor_deploy.SERVICES_QUEUE
            or queue_id != successor_deploy.SERVICES_QUEUE_ID
            or hashlib.sha256(source.encode()).hexdigest()
            != sources[role]["deployed_sha256"]
            or _content_sha256(
                {str(key): str(item) for key, item in observed_parameters.items()}
            )
            != _content_sha256(
                {str(key): str(item) for key, item in parameters[role].items()}
            )
            or not {str(item["name"]) for item in artifacts}.issubset(
                set(successor_deploy.ROLE_OUTPUT_ARTIFACTS[role])
            )
        ):
            raise OriginalQueueRecoveryError(
                f"fresh successor {role} live authority drifted"
            )
        result[role] = {
            "task_id": str(task_ids[role]),
            "status": status,
            "queue": queue,
            "queue_id": queue_id,
            "source_sha256": sources[role]["deployed_sha256"],
            "parameters_sha256": _content_sha256(
                {str(key): str(item) for key, item in observed_parameters.items()}
            ),
            "artifact_inventory": artifacts,
        }
    return result


def _successors_advance_monotonically(
    frozen: Mapping[str, object], current: Mapping[str, object]
) -> bool:
    ranks = {"queued": 0, "in_progress": 1, "completed": 2}
    if set(frozen) != {"W", "L", "A", "S"} or set(current) != set(frozen):
        return False
    stable = (
        "task_id",
        "queue",
        "queue_id",
        "source_sha256",
        "parameters_sha256",
    )
    for role in ("W", "L", "A", "S"):
        before = frozen[role]
        after = current[role]
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            return False
        if (
            before.get("status") not in ranks
            or after.get("status") not in ranks
            or ranks[str(after["status"])] < ranks[str(before["status"])]
            or any(before.get(key) != after.get(key) for key in stable)
        ):
            return False
        before_artifacts = {
            _content_sha256(item)
            for item in before.get("artifact_inventory", [])
            if isinstance(item, Mapping)
        }
        after_artifacts = {
            _content_sha256(item)
            for item in after.get("artifact_inventory", [])
            if isinstance(item, Mapping)
        }
        if not before_artifacts.issubset(after_artifacts):
            return False
    return True


def _live_formal_core(
    task_class: object,
    *,
    ffnet_binding: Mapping[str, object],
    coformer_expected: Mapping[str, object],
) -> dict[str, object]:
    ffnet = shared._authoritative_task(
        task_class, FFNET_TASK_ID, context="original-plan FFNet"
    )
    status = shared._status(ffnet)
    queue, queue_id = shared._queue_identity(ffnet, context="original-plan FFNet")
    source = shared._source(
        ffnet,
        entry_point="clearml_5090_bootstrap.py",
        context="original-plan FFNet",
    )
    parameters = shared._parameters(ffnet, context="original-plan FFNet")
    artifacts = shared._artifact_inventory(ffnet)
    if (
        status not in {"created", "queued", "in_progress", "completed"}
        or hashlib.sha256(source.encode()).hexdigest()
        != ffnet_binding["ffnet_source_sha256"]
        or shared._evaluation_parameters_sha256(parameters)
        != ffnet_binding["ffnet_parameters_sha256"]
        or (
            status == "created" and (queue or queue_id)
        )
        or (
            status != "created"
            and (queue != ORIGINAL_QUEUE or queue_id != ORIGINAL_QUEUE_ID)
        )
        or not {str(item["name"]) for item in artifacts}.issubset(
            shared.FORMAL_EVALUATION_ARTIFACTS
        )
    ):
        raise OriginalQueueRecoveryError("FFNet live original-plan authority drifted")
    if status == "completed" and {
        str(item["name"]) for item in artifacts
    } != shared.FORMAL_EVALUATION_ARTIFACTS:
        raise OriginalQueueRecoveryError("completed FFNet lacks formal evidence")
    coformer_task = shared._authoritative_task(
        task_class, COFORMER_TASK_ID, context="completed CoFormer"
    )
    coformer_queue, coformer_queue_id = shared._queue_identity(
        coformer_task, context="completed CoFormer"
    )
    coformer_source = shared._source(
        coformer_task,
        entry_point="clearml_5090_bootstrap.py",
        context="completed CoFormer",
    )
    coformer_parameters = shared._parameters(
        coformer_task, context="completed CoFormer"
    )
    coformer_models = shared._model_inventory(
        coformer_task, context="completed CoFormer"
    )
    if (
        str(getattr(coformer_task, "name", "") or "")
        != coformer_expected.get("task_name")
        or successor_deploy._task_parent(coformer_task)
        != coformer_expected.get("parent_task_id")
        or shared._exact_project_id(
            coformer_task, context="completed CoFormer"
        )
        != coformer_expected.get("project_id")
        or str(getattr(coformer_task, "task_type", "") or "")
        .rsplit(".", 1)[-1]
        .lower()
        != coformer_expected.get("task_type")
        or coformer_queue != coformer_expected.get("queue")
        or coformer_queue_id != coformer_expected.get("queue_id")
        or hashlib.sha256(coformer_source.encode()).hexdigest()
        != coformer_expected.get("script_sha256")
        or shared._evaluation_parameters_sha256(coformer_parameters)
        != coformer_expected.get("parameters_sha256")
        or coformer_models.get("input")
        != coformer_expected.get("input_model_ids")
        or coformer_models.get("output")
        != coformer_expected.get("output_model_ids")
        or sorted(
            shared._tags(
                coformer_task, field="tags", context="completed CoFormer"
            )
        )
        != coformer_expected.get("tags")
        or sorted(
            shared._tags(
                coformer_task,
                field="system_tags",
                context="completed CoFormer",
            )
        )
        != coformer_expected.get("system_tags")
    ):
        raise OriginalQueueRecoveryError("CoFormer completed identity changed")
    coformer_record = {
        **dict(coformer_expected),
        "status": shared._status(coformer_task),
        "artifact_inventory": shared._artifact_inventory(coformer_task),
    }
    coformer_record["artifact_names"] = [
        str(item["name"]) for item in coformer_record["artifact_inventory"]
    ]
    coformer_record["artifact_inventory_sha256"] = _content_sha256(
        coformer_record["artifact_inventory"]
    )
    if _coformer_completed_record({"records": [coformer_record]}) != (
        coformer_expected
    ):
        raise OriginalQueueRecoveryError("CoFormer completed evidence changed")
    return {
        "ffnet": {
            "task_id": FFNET_TASK_ID,
            "status": status,
            "queue": queue,
            "queue_id": queue_id,
            "source_sha256": ffnet_binding["ffnet_source_sha256"],
            "parameters_sha256": ffnet_binding["ffnet_parameters_sha256"],
            "artifact_inventory": artifacts,
        },
        "coformer": coformer_expected,
    }


def build_candidate_attempt(
    task_class: object,
    *,
    evaluation_recovery_receipt: Path,
    ffnet_recovery_receipt: Path,
    successor_deployment_receipt: Path,
) -> dict[str, object]:
    evaluation = shared._evaluation_binding(evaluation_recovery_receipt)
    ffnet = validate_ffnet_recovery_receipt(ffnet_recovery_receipt)
    successor, successor_receipt = _successor_binding(
        successor_deployment_receipt,
        evaluation_binding=evaluation,
        ffnet_binding=ffnet,
    )
    live_successors = _live_successors(task_class, successor_receipt)
    ffnet_receipt, _ = shared._read_sealed(
        ffnet_recovery_receipt, context="original queue FFNet recovery receipt"
    )
    formal_core = _live_formal_core(
        task_class,
        ffnet_binding=ffnet,
        coformer_expected=ffnet_receipt["coformer_completed_evidence"],
    )
    candidate, _ = shared._target_snapshot(
        task_class,
        label="C",
        entry_point=candidate_deploy.ENTRY_POINT,
        parent_task_id=candidate_deploy.PARENT_TASK_ID,
        task_name=candidate_deploy.CONTROLLER_NAME,
        parameters=None,
        allowed_artifacts=frozenset(),
    )
    prepared = candidate_deploy._prepare()
    if (
        candidate["source"] != prepared["source"]
        or candidate["parameters"]
        != {str(key): str(value) for key, value in prepared["parameters"].items()}
        or set(candidate["tags"]) != candidate_deploy.TAGS
    ):
        raise OriginalQueueRecoveryError(
            "candidate target is not the unchanged original deployment"
        )
    names = shared._exact_name_inventory(
        task_class,
        task_name=str(candidate["task_name"]),
        parent_task_id=str(candidate["parent_task_id"]),
        expected_task_id=CANDIDATE_CONTROLLER_TASK_ID,
    )
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": CANDIDATE_ATTEMPT_TYPE,
            "generated_at_utc": shared._utc_timestamp(),
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_id_only": True,
            "replacement_controller_created": False,
            "plan_policy": "candidate_after_fresh_original_plan_wlas",
            "bindings": {
                "evaluation_recovery": evaluation,
                "ffnet_recovery": ffnet,
                "fresh_successor_deployment": successor,
            },
            "fresh_successor_authority": live_successors,
            "formal_core_authority": formal_core,
            "candidate_material": candidate,
            "exact_name_inventory": names,
            "preexisting_same_name_tasks": shared._same_name_history_snapshot(
                task_class, label="C", inventory=names
            ),
            "planned_actions": [
                "durably_write_attempt_receipt",
                "revalidate_fresh_wlas_before_candidate_reset",
                "revalidate_ffnet_original_queue_and_coformer_completed",
                "reset_reinstall_exact_candidate_controller_only",
                "enqueue_exact_candidate_controller_services",
            ],
        }
    )


def _validate_candidate_attempt(value: Mapping[str, object]) -> None:
    expected = {
        "schema_version": 1,
        "receipt_type": CANDIDATE_ATTEMPT_TYPE,
        "status": "frozen_before_reset",
        "remote_state_changed": False,
        "same_id_only": True,
        "replacement_controller_created": False,
        "plan_policy": "candidate_after_fresh_original_plan_wlas",
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise OriginalQueueRecoveryError(f"candidate attempt {key} drifted")
    material = value.get("candidate_material")
    if (
        not isinstance(material, Mapping)
        or material.get("task_id") != CANDIDATE_CONTROLLER_TASK_ID
        or material.get("queue") != candidate_deploy.SERVICES_QUEUE
        or material.get("queue_id") != candidate_deploy.SERVICES_QUEUE_ID
        or material.get("source_sha256") != shared.EXPECTED_SOURCE_SHA256["C"]
        or material.get("parameters_sha256")
        != shared.EXPECTED_PARAMETERS_SHA256["C"]
    ):
        raise OriginalQueueRecoveryError("candidate attempt material drifted")


def execute_candidate_recovery(
    task_class: object,
    *,
    attempt: Mapping[str, object],
    attempt_path: Path,
    successor_receipt: Mapping[str, object],
    ffnet_receipt: Mapping[str, object],
    sleeper: object,
) -> dict[str, object]:
    _validate_candidate_attempt(attempt)
    frozen, resolved = shared._read_sealed(
        attempt_path, context="original queue candidate attempt"
    )
    if resolved != attempt_path.resolve() or frozen != dict(attempt):
        raise OriginalQueueRecoveryError("candidate attempt was not durably frozen")
    material = attempt["candidate_material"]
    names = shared._exact_name_inventory(
        task_class,
        task_name=str(material["task_name"]),
        parent_task_id=str(material["parent_task_id"]),
        expected_task_id=CANDIDATE_CONTROLLER_TASK_ID,
    )
    if names != attempt["exact_name_inventory"]:
        raise OriginalQueueRecoveryError("candidate exact-name inventory changed")
    live_successors = _live_successors(task_class, successor_receipt)
    if not _successors_advance_monotonically(
        attempt["fresh_successor_authority"], live_successors
    ):
        raise OriginalQueueRecoveryError(
            "fresh successor authority did not advance monotonically"
        )
    formal_core_before = _live_formal_core(
        task_class,
        ffnet_binding=attempt["bindings"]["ffnet_recovery"],
        coformer_expected=ffnet_receipt["coformer_completed_evidence"],
    )
    shared._validate_failed_target_against_material(
        task_class,
        label="C",
        material=material,
        context="candidate immediate pre-reset guard",
    )
    created = shared._reset_and_install(
        task_class, label="C", material=material, sleeper=sleeper
    )
    prepared = candidate_deploy._prepare()
    if (
        created.get("status") != "created"
        or created.get("queue") is not None
        or created.get("queue_id") is not None
    ):
        raise OriginalQueueRecoveryError(
            "candidate controller was not created+unqueued before enqueue"
        )
    task = shared._authoritative_task(
        task_class,
        CANDIDATE_CONTROLLER_TASK_ID,
        context="candidate immediate pre-enqueue guard",
    )
    candidate_deploy._validate_task(
        task_class,
        task_id=CANDIDATE_CONTROLLER_TASK_ID,
        prepared=prepared,
        expected_statuses=("created",),
    )
    queued = candidate_deploy._enqueue(
        task_class,
        task,
        task_id=CANDIDATE_CONTROLLER_TASK_ID,
        prepared=prepared,
        sleeper=sleeper,
    )
    final_names = shared._exact_name_inventory(
        task_class,
        task_name=str(material["task_name"]),
        parent_task_id=str(material["parent_task_id"]),
        expected_task_id=CANDIDATE_CONTROLLER_TASK_ID,
    )
    if final_names != names:
        raise OriginalQueueRecoveryError("candidate recovery created a replacement")
    live_successors_final = _live_successors(task_class, successor_receipt)
    formal_core_final = _live_formal_core(
        task_class,
        ffnet_binding=attempt["bindings"]["ffnet_recovery"],
        coformer_expected=ffnet_receipt["coformer_completed_evidence"],
    )
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": CANDIDATE_RECEIPT_TYPE,
            "generated_at_utc": shared._utc_timestamp(),
            "status": "candidate_recovered_and_enqueued",
            "remote_state_changed": True,
            "same_id_only": True,
            "replacement_controller_created": False,
            "plan_policy": "candidate_after_fresh_original_plan_wlas",
            "attempt_receipt_path": str(attempt_path.resolve()),
            "attempt_receipt_seal_sha256": attempt["seal_sha256"],
            "bindings": dict(attempt["bindings"]),
            "candidate_controller_task_id": CANDIDATE_CONTROLLER_TASK_ID,
            "created_unqueued_authority": created,
            "queued_authority": queued,
            "formal_core_authority_before_reset": formal_core_before,
            "fresh_successor_authority": live_successors_final,
            "formal_core_authority": formal_core_final,
            "exact_name_inventory": final_names,
            "coformer_mutated": False,
            "ffnet_mutated": False,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("ffnet", "candidate"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--evaluation-recovery-receipt", type=Path, required=True)
    parser.add_argument("--successor-deployment-receipt", type=Path, required=True)
    parser.add_argument("--controller-recovery-receipt", type=Path)
    parser.add_argument("--candidate-deployment-receipt", type=Path)
    parser.add_argument("--ffnet-recovery-receipt", type=Path)
    parser.add_argument("--attempt-receipt", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def _summary(path: Path, value: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {
                "receipt": str(path),
                "receipt_type": value.get("receipt_type"),
                "status": value.get("status"),
                "remote_state_changed": value.get("remote_state_changed"),
                "seal_sha256": value.get("seal_sha256"),
            },
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    token = FFNET_EXECUTE_TOKEN if args.phase == "ffnet" else CANDIDATE_EXECUTE_TOKEN
    if args.execute and args.execute_token != token:
        raise OriginalQueueRecoveryError(f"exact execute token required: {token}")
    if not args.execute and args.execute_token:
        raise OriginalQueueRecoveryError("--execute-token requires --execute")
    if args.execute and args.attempt_receipt is None:
        raise OriginalQueueRecoveryError("--attempt-receipt is required with --execute")
    if not args.execute and args.attempt_receipt is not None:
        raise OriginalQueueRecoveryError("--attempt-receipt requires --execute")
    required = (
        (args.controller_recovery_receipt, args.candidate_deployment_receipt)
        if args.phase == "ffnet"
        else (args.ffnet_recovery_receipt,)
    )
    if any(item is None for item in required):
        raise OriginalQueueRecoveryError(f"{args.phase} phase inputs are incomplete")
    outputs = [args.receipt, *([] if args.attempt_receipt is None else [args.attempt_receipt])]
    if len({item.resolve() for item in outputs}) != len(outputs):
        raise OriginalQueueRecoveryError("attempt and final receipt paths must differ")
    for path in outputs:
        _require_new(path, context="original queue recovery output")
    try:
        from clearml import Task
    except ImportError as error:
        raise OriginalQueueRecoveryError("ClearML is required for inspection") from error
    if args.phase == "ffnet":
        attempt = build_ffnet_attempt(
            Task,
            evaluation_recovery_receipt=args.evaluation_recovery_receipt,
            successor_deployment_receipt=args.successor_deployment_receipt,
            controller_recovery_receipt=args.controller_recovery_receipt,
            candidate_deployment_receipt=args.candidate_deployment_receipt,
        )
        plan_type = FFNET_PLAN_TYPE
    else:
        attempt = build_candidate_attempt(
            Task,
            evaluation_recovery_receipt=args.evaluation_recovery_receipt,
            ffnet_recovery_receipt=args.ffnet_recovery_receipt,
            successor_deployment_receipt=args.successor_deployment_receipt,
        )
        plan_type = CANDIDATE_PLAN_TYPE
    if not args.execute:
        plan = _seal(
            {
                **attempt,
                "receipt_type": plan_type,
                "status": "planned",
                "execute_token": token,
            }
        )
        _write_new(args.receipt, plan)
        _summary(args.receipt, plan)
        return 0
    _write_new(args.attempt_receipt, attempt)
    if args.phase == "ffnet":
        result = execute_ffnet_recovery(
            Task,
            attempt=attempt,
            attempt_path=args.attempt_receipt,
            sleeper=shared.time.sleep,
        )
    else:
        successor_receipt, _ = shared._read_sealed(
            args.successor_deployment_receipt,
            context="fresh original-plan successor receipt",
        )
        ffnet_receipt, _ = shared._read_sealed(
            args.ffnet_recovery_receipt,
            context="original queue FFNet recovery receipt",
        )
        result = execute_candidate_recovery(
            Task,
            attempt=attempt,
            attempt_path=args.attempt_receipt,
            successor_receipt=successor_receipt,
            ffnet_receipt=ffnet_receipt,
            sleeper=shared.time.sleep,
        )
    _write_new(args.receipt, result)
    _summary(args.receipt, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CANDIDATE_CONTROLLER_TASK_ID",
    "CANDIDATE_EXECUTE_TOKEN",
    "COFORMER_TASK_ID",
    "FFNET_EXECUTE_TOKEN",
    "FFNET_TASK_ID",
    "OriginalQueueRecoveryError",
    "build_candidate_attempt",
    "build_ffnet_attempt",
    "execute_candidate_recovery",
    "execute_ffnet_recovery",
    "main",
    "validate_ffnet_recovery_receipt",
)
