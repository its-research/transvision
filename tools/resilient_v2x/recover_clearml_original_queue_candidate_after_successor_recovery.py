#!/usr/bin/env python3
"""Run original-queue candidate recovery against a recovered exact W/L/A/S chain."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from tools.resilient_v2x import (
        recover_clearml_formal_successor_runtime_failure as successor_recovery,
    )
    from tools.resilient_v2x import (
        recover_clearml_original_queue_runtime_failure as original,
    )
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import recover_clearml_formal_successor_runtime_failure as successor_recovery
    import recover_clearml_original_queue_runtime_failure as original


class CandidateAfterSuccessorRecoveryError(RuntimeError):
    """Raised when the recovered-successor bridge cannot be proven."""


_ROLES = ("W", "L", "A", "S")
_STATUS_RANK = {"queued": 0, "in_progress": 1, "completed": 2}


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CandidateAfterSuccessorRecoveryError(f"{context} is not an object")
    return dict(value)


def _service_environment_advances_monotonically(
    frozen: Mapping[str, object],
    current: Mapping[str, object],
    *,
    status: str,
) -> bool:
    """Accept only ClearML's canonical-to-materialized requirement transition."""

    stable = ("policy", "docker", "output_uri", "binary")
    if any(frozen.get(key) != current.get(key) for key in stable):
        return False
    before = frozen.get("requirements")
    after = current.get("requirements")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return False
    if before == after:
        return True
    policy = frozen.get("policy")
    return (
        isinstance(policy, Mapping)
        and before.get("form") == "canonical"
        and after.get("form") == "agent_materialized"
        and status in {"in_progress", "completed"}
        and before.get("canonical_requirement")
        == after.get("canonical_requirement")
        == policy.get("requirement")
    )


def _validated_effective_authority(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the standalone, sealed post-recovery W/L/A/S authority."""

    authority = dict(value)
    expected_keys = {
        "schema_version",
        "authority_type",
        "recovery_binding",
        "original_successor_deployment",
        "task_ids",
        "parents",
        "sources",
        "parameters",
        "roles",
        "seal_sha256",
    }
    if set(authority) != expected_keys:
        raise CandidateAfterSuccessorRecoveryError(
            "effective successor authority key inventory drifted"
        )
    if (
        authority.get("schema_version") != 1
        or authority.get("authority_type")
        != successor_recovery.EFFECTIVE_AUTHORITY_TYPE
        or successor_recovery._seal(authority) != authority
    ):
        raise CandidateAfterSuccessorRecoveryError(
            "effective successor authority identity or seal drifted"
        )
    task_ids = _mapping(authority.get("task_ids"), context="effective task IDs")
    expected_task_ids = {
        "P": successor_recovery.deploy.COMPLETED_PROVENANCE_TASK_ID,
        **successor_recovery.SUCCESSOR_TASK_IDS,
    }
    parents = _mapping(authority.get("parents"), context="effective parents")
    sources = _mapping(authority.get("sources"), context="effective sources")
    parameters = _mapping(
        authority.get("parameters"), context="effective parameters"
    )
    roles = _mapping(authority.get("roles"), context="effective roles")
    if (
        task_ids != expected_task_ids
        or set(parents) != set(_ROLES)
        or set(sources) != set(_ROLES)
        or set(parameters) != set(_ROLES)
        or set(roles) != set(_ROLES)
        or parents
        != {
            "W": task_ids["P"],
            "L": task_ids["W"],
            "A": task_ids["L"],
            "S": task_ids["A"],
        }
    ):
        raise CandidateAfterSuccessorRecoveryError(
            "effective successor identity graph drifted"
        )
    recovery_binding = _mapping(
        authority.get("recovery_binding"), context="effective recovery binding"
    )
    original_deployment = _mapping(
        authority.get("original_successor_deployment"),
        context="effective original deployment",
    )
    if (
        recovery_binding.get("original_successor_deployment")
        != original_deployment
        or recovery_binding.get("task_ids")
        != successor_recovery.SUCCESSOR_TASK_IDS
        or recovery_binding.get("same_ids_only") is not True
        or recovery_binding.get("replacement_tasks_created") is not False
    ):
        raise CandidateAfterSuccessorRecoveryError(
            "effective successor recovery binding drifted"
        )
    expected_role_keys = {
        "task_id",
        "status",
        "project_id",
        "task_name",
        "task_type",
        "parent_task_id",
        "queue",
        "queue_id",
        "source_sha256",
        "parameters_sha256",
        "artifact_inventory",
        "model_inventory",
        "tags",
        "system_tags",
        "service_environment",
    }
    source_binding = _mapping(
        recovery_binding.get("source_sha256"),
        context="effective recovery source binding",
    )
    parameter_binding = _mapping(
        recovery_binding.get("parameters_sha256"),
        context="effective recovery parameter binding",
    )
    for role in _ROLES:
        source_sha = sources[role]
        parameter_row = _mapping(
            parameters[role], context=f"effective {role} parameters"
        )
        values = _mapping(
            parameter_row.get("values"),
            context=f"effective {role} parameter values",
        )
        row = _mapping(roles[role], context=f"effective {role} authority")
        artifacts = row.get("artifact_inventory")
        models = row.get("model_inventory")
        tags = row.get("tags")
        system_tags = row.get("system_tags")
        environment = row.get("service_environment")
        if (
            type(source_sha) is not str
            or len(source_sha) != 64
            or any(character not in "0123456789abcdef" for character in source_sha)
            or set(parameter_row) != {"values", "sha256"}
            or any(
                type(key) is not str or type(item) is not str
                for key, item in values.items()
            )
            or parameter_row.get("sha256")
            != successor_recovery._content_sha256(values)
            or source_binding.get(role) != source_sha
            or parameter_binding.get(role) != parameter_row.get("sha256")
            or set(row) != expected_role_keys
            or row.get("task_id") != task_ids[role]
            or row.get("status") not in _STATUS_RANK
            or row.get("project_id")
            != original.shared.evaluation_recovery.PROJECT_ID
            or row.get("task_name")
            != successor_recovery.deploy.ROLE_DEFINITIONS[role]["task_name"]
            or row.get("task_type") != "controller"
            or row.get("parent_task_id") != parents[role]
            or row.get("queue") != successor_recovery.deploy.SERVICES_QUEUE
            or row.get("queue_id")
            != successor_recovery.deploy.SERVICES_QUEUE_ID
            or row.get("source_sha256") != source_sha
            or row.get("parameters_sha256") != parameter_row.get("sha256")
            or not isinstance(artifacts, list)
            or any(not isinstance(item, Mapping) for item in artifacts)
            or len({str(item.get("name") or "") for item in artifacts})
            != len(artifacts)
            or not {str(item.get("name") or "") for item in artifacts}.issubset(
                successor_recovery.deploy.ROLE_OUTPUT_ARTIFACTS[role]
            )
            or (
                row.get("status") == "completed"
                and {str(item.get("name") or "") for item in artifacts}
                != set(successor_recovery.deploy.ROLE_OUTPUT_ARTIFACTS[role])
            )
            or models != {"input": [], "output": []}
            or not isinstance(tags, list)
            or not isinstance(system_tags, list)
            or any(type(item) is not str for item in tags + system_tags)
            or tags != sorted(set(tags))
            or system_tags != sorted(set(system_tags))
            or not isinstance(environment, Mapping)
        ):
            raise CandidateAfterSuccessorRecoveryError(
                f"effective successor {role} authority drifted"
            )
    return authority


def _live_successors(
    task_class: object, effective_authority: Mapping[str, object]
) -> dict[str, object]:
    """Read and strictly validate the live recovered W/L/A/S authority."""

    effective = _validated_effective_authority(effective_authority)
    task_ids = effective["task_ids"]
    parents = effective["parents"]
    sources = effective["sources"]
    parameters = effective["parameters"]
    frozen_roles = effective["roles"]
    result: dict[str, object] = {}
    for role in _ROLES:
        context = f"effective successor {role}"
        task = original.shared._authoritative_task(
            task_class, str(task_ids[role]), context=context
        )
        status = original.shared._status(task)
        source = original.shared._source(
            task,
            entry_point=str(
                successor_recovery.deploy.ROLE_DEFINITIONS[role]["entry_point"]
            ),
            context=context,
        )
        try:
            observed_parameters = successor_recovery._normalized_parameters(
                role, original.shared._parameters(task, context=context)
            )
        except successor_recovery.SuccessorRuntimeRecoveryError as error:
            raise CandidateAfterSuccessorRecoveryError(
                f"effective successor {role} parameter authority drifted"
            ) from error
        queue, queue_id = original.shared._queue_identity(task, context=context)
        artifacts = original.shared._artifact_inventory(task)
        models = original.shared._model_inventory(task, context=context)
        tags = sorted(original.shared._tags(task, field="tags", context=context))
        system_tags = sorted(
            original.shared._tags(task, field="system_tags", context=context)
        )
        environment = original.shared._service_environment(task, context=context)
        frozen = frozen_roles[role]
        expected_parameters = parameters[role]["values"]
        artifact_names = {str(item.get("name") or "") for item in artifacts}
        if (
            status not in _STATUS_RANK
            or _STATUS_RANK[status] < _STATUS_RANK[str(frozen["status"])]
            or str(getattr(task, "name", "") or "")
            != successor_recovery.deploy.ROLE_DEFINITIONS[role]["task_name"]
            or successor_recovery.deploy._task_parent(task) != parents[role]
            or original.shared._exact_project_id(task, context=context)
            != original.shared.evaluation_recovery.PROJECT_ID
            or str(getattr(task, "task_type", "") or "")
            .rsplit(".", 1)[-1]
            .lower()
            != "controller"
            or hashlib.sha256(source.encode("utf-8")).hexdigest() != sources[role]
            or observed_parameters != expected_parameters
            or successor_recovery._content_sha256(observed_parameters)
            != parameters[role]["sha256"]
            or queue != successor_recovery.deploy.SERVICES_QUEUE
            or queue_id != successor_recovery.deploy.SERVICES_QUEUE_ID
            or not original.shared._artifact_inventory_is_monotonic(
                frozen["artifact_inventory"],
                artifacts,
                allowed_names=successor_recovery.deploy.ROLE_OUTPUT_ARTIFACTS[role],
            )
            or (
                status == "completed"
                and artifact_names
                != set(successor_recovery.deploy.ROLE_OUTPUT_ARTIFACTS[role])
            )
            or models != frozen["model_inventory"]
            or tags != frozen["tags"]
            or system_tags != frozen["system_tags"]
            or not _service_environment_advances_monotonically(
                frozen["service_environment"],
                environment,
                status=status,
            )
        ):
            raise CandidateAfterSuccessorRecoveryError(
                f"effective successor {role} live authority drifted"
            )
        result[role] = {
            "task_id": str(task_ids[role]),
            "status": status,
            "project_id": original.shared.evaluation_recovery.PROJECT_ID,
            "task_name": successor_recovery.deploy.ROLE_DEFINITIONS[role][
                "task_name"
            ],
            "task_type": "controller",
            "parent_task_id": parents[role],
            "queue": queue,
            "queue_id": queue_id,
            "source_sha256": sources[role],
            "parameters_sha256": parameters[role]["sha256"],
            "artifact_inventory": artifacts,
            "model_inventory": models,
            "tags": tags,
            "system_tags": system_tags,
            "service_environment": environment,
        }
    return result


def _effective_receipts(
    *,
    successor_deployment_receipt: Path,
    successor_runtime_recovery_receipt: Path,
    evaluation_recovery_receipt: Path,
    ffnet_recovery_receipt: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    evaluation = original.shared._evaluation_binding(evaluation_recovery_receipt)
    ffnet = original.validate_ffnet_recovery_receipt(ffnet_recovery_receipt)
    successor_binding, deployment = original._successor_binding(
        successor_deployment_receipt,
        evaluation_binding=evaluation,
        ffnet_binding=ffnet,
    )
    try:
        effective = successor_recovery.effective_successor_authority(
            successor_runtime_recovery_receipt,
            original_deployment_receipt=deployment,
        )
    except successor_recovery.SuccessorRuntimeRecoveryError as error:
        raise CandidateAfterSuccessorRecoveryError(
            "successor runtime recovery receipt is invalid"
        ) from error
    effective = _validated_effective_authority(effective)
    runtime_binding = dict(effective["recovery_binding"])
    successor_binding = dict(successor_binding)
    successor_binding["runtime_recovery"] = runtime_binding
    successor_binding["effective_successor_authority_seal_sha256"] = effective[
        "seal_sha256"
    ]
    return effective, successor_binding, ffnet


def build_candidate_attempt(
    task_class: object,
    *,
    evaluation_recovery_receipt: Path,
    ffnet_recovery_receipt: Path,
    successor_deployment_receipt: Path,
    successor_runtime_recovery_receipt: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    effective, successor_binding, ffnet = _effective_receipts(
        successor_deployment_receipt=successor_deployment_receipt,
        successor_runtime_recovery_receipt=successor_runtime_recovery_receipt,
        evaluation_recovery_receipt=evaluation_recovery_receipt,
        ffnet_recovery_receipt=ffnet_recovery_receipt,
    )
    live_successors = _live_successors(task_class, effective)
    ffnet_receipt, _ = original.shared._read_sealed(
        ffnet_recovery_receipt,
        context="original queue FFNet recovery receipt",
    )
    formal_core = original._live_formal_core(
        task_class,
        ffnet_binding=ffnet,
        coformer_expected=ffnet_receipt["coformer_completed_evidence"],
    )
    candidate, _ = original.shared._target_snapshot(
        task_class,
        label="C",
        entry_point=original.candidate_deploy.ENTRY_POINT,
        parent_task_id=original.candidate_deploy.PARENT_TASK_ID,
        task_name=original.candidate_deploy.CONTROLLER_NAME,
        parameters=None,
        allowed_artifacts=frozenset(),
    )
    prepared = original.candidate_deploy._prepare()
    if (
        candidate["source"] != prepared["source"]
        or candidate["parameters"]
        != {str(key): str(value) for key, value in prepared["parameters"].items()}
        or set(candidate["tags"]) != original.candidate_deploy.TAGS
    ):
        raise CandidateAfterSuccessorRecoveryError(
            "candidate target is not the unchanged original deployment"
        )
    names = original.shared._exact_name_inventory(
        task_class,
        task_name=str(candidate["task_name"]),
        parent_task_id=str(candidate["parent_task_id"]),
        expected_task_id=original.CANDIDATE_CONTROLLER_TASK_ID,
    )
    attempt = original._seal(
        {
            "schema_version": 1,
            "receipt_type": original.CANDIDATE_ATTEMPT_TYPE,
            "generated_at_utc": original.shared._utc_timestamp(),
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_id_only": True,
            "replacement_controller_created": False,
            "plan_policy": "candidate_after_recovered_exact_original_plan_wlas",
            "bindings": {
                "evaluation_recovery": original.shared._evaluation_binding(
                    evaluation_recovery_receipt
                ),
                "ffnet_recovery": ffnet,
                "fresh_successor_deployment": successor_binding,
            },
            "effective_successor_authority": effective,
            "fresh_successor_authority": live_successors,
            "formal_core_authority": formal_core,
            "candidate_material": candidate,
            "exact_name_inventory": names,
            "preexisting_same_name_tasks": (
                original.shared._same_name_history_snapshot(
                    task_class,
                    label="C",
                    inventory=names,
                )
            ),
            "planned_actions": [
                "durably_write_attempt_receipt",
                "revalidate_recovered_exact_wlas_before_candidate_reset",
                "revalidate_ffnet_original_queue_and_coformer_completed",
                "reset_reinstall_exact_candidate_controller_only",
                "enqueue_exact_candidate_controller_services",
            ],
        }
    )
    return attempt, effective


def _validate_candidate_attempt(value: Mapping[str, object]) -> None:
    expected = {
        "schema_version": 1,
        "receipt_type": original.CANDIDATE_ATTEMPT_TYPE,
        "status": "frozen_before_reset",
        "remote_state_changed": False,
        "same_id_only": True,
        "replacement_controller_created": False,
        "plan_policy": "candidate_after_recovered_exact_original_plan_wlas",
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise CandidateAfterSuccessorRecoveryError(
            "candidate recovery attempt identity drifted"
        )
    if original._seal(value) != dict(value):
        raise CandidateAfterSuccessorRecoveryError(
            "candidate recovery attempt seal drifted"
        )
    material = _mapping(
        value.get("candidate_material"), context="candidate material"
    )
    if (
        material.get("task_id") != original.CANDIDATE_CONTROLLER_TASK_ID
        or material.get("queue") != original.candidate_deploy.SERVICES_QUEUE
        or material.get("queue_id") != original.candidate_deploy.SERVICES_QUEUE_ID
        or material.get("source_sha256")
        != original.shared.EXPECTED_SOURCE_SHA256["C"]
        or material.get("parameters_sha256")
        != original.shared.EXPECTED_PARAMETERS_SHA256["C"]
    ):
        raise CandidateAfterSuccessorRecoveryError(
            "candidate recovery attempt material drifted"
        )
    effective = _validated_effective_authority(
        _mapping(
            value.get("effective_successor_authority"),
            context="frozen effective successor authority",
        )
    )
    bindings = _mapping(value.get("bindings"), context="candidate bindings")
    successor_binding = _mapping(
        bindings.get("fresh_successor_deployment"),
        context="candidate successor binding",
    )
    if (
        successor_binding.get("runtime_recovery")
        != effective["recovery_binding"]
        or successor_binding.get("effective_successor_authority_seal_sha256")
        != effective["seal_sha256"]
        or successor_binding.get("task_ids")
        != successor_recovery.SUCCESSOR_TASK_IDS
        or set(
            _mapping(
                value.get("fresh_successor_authority"),
                context="frozen live successor authority",
            )
        )
        != set(_ROLES)
    ):
        raise CandidateAfterSuccessorRecoveryError(
            "candidate effective successor binding drifted"
        )


def execute_candidate_recovery(
    task_class: object,
    *,
    attempt: Mapping[str, object],
    attempt_path: Path,
    successor_authority: Mapping[str, object],
    ffnet_receipt: Mapping[str, object],
    sleeper: object,
) -> dict[str, object]:
    """Run the original exact-ID candidate sequence with strict W/L/A/S reads."""

    _validate_candidate_attempt(attempt)
    frozen, resolved = original.shared._read_sealed(
        attempt_path, context="original queue candidate attempt"
    )
    if resolved != attempt_path.resolve() or frozen != dict(attempt):
        raise CandidateAfterSuccessorRecoveryError(
            "candidate attempt was not durably frozen"
        )
    effective = _validated_effective_authority(successor_authority)
    if effective != attempt["effective_successor_authority"]:
        raise CandidateAfterSuccessorRecoveryError(
            "effective successor authority changed after candidate freeze"
        )
    material = attempt["candidate_material"]
    names = original.shared._exact_name_inventory(
        task_class,
        task_name=str(material["task_name"]),
        parent_task_id=str(material["parent_task_id"]),
        expected_task_id=original.CANDIDATE_CONTROLLER_TASK_ID,
    )
    if names != attempt["exact_name_inventory"]:
        raise CandidateAfterSuccessorRecoveryError(
            "candidate exact-name inventory changed"
        )
    live_successors = _live_successors(task_class, effective)
    if not original._successors_advance_monotonically(
        attempt["fresh_successor_authority"], live_successors
    ):
        raise CandidateAfterSuccessorRecoveryError(
            "effective successor authority did not advance monotonically"
        )
    formal_core_before = original._live_formal_core(
        task_class,
        ffnet_binding=attempt["bindings"]["ffnet_recovery"],
        coformer_expected=ffnet_receipt["coformer_completed_evidence"],
    )
    original.shared._validate_failed_target_against_material(
        task_class,
        label="C",
        material=material,
        context="candidate immediate pre-reset guard",
    )
    created = original.shared._reset_and_install(
        task_class, label="C", material=material, sleeper=sleeper
    )
    prepared = original.candidate_deploy._prepare()
    if (
        created.get("status") != "created"
        or created.get("queue") is not None
        or created.get("queue_id") is not None
    ):
        raise CandidateAfterSuccessorRecoveryError(
            "candidate controller was not created+unqueued before enqueue"
        )
    task = original.shared._authoritative_task(
        task_class,
        original.CANDIDATE_CONTROLLER_TASK_ID,
        context="candidate immediate pre-enqueue guard",
    )
    original.candidate_deploy._validate_task(
        task_class,
        task_id=original.CANDIDATE_CONTROLLER_TASK_ID,
        prepared=prepared,
        expected_statuses=("created",),
    )
    queued = original.candidate_deploy._enqueue(
        task_class,
        task,
        task_id=original.CANDIDATE_CONTROLLER_TASK_ID,
        prepared=prepared,
        sleeper=sleeper,
    )
    final_names = original.shared._exact_name_inventory(
        task_class,
        task_name=str(material["task_name"]),
        parent_task_id=str(material["parent_task_id"]),
        expected_task_id=original.CANDIDATE_CONTROLLER_TASK_ID,
    )
    if final_names != names:
        raise CandidateAfterSuccessorRecoveryError(
            "candidate recovery created a replacement"
        )
    live_successors_final = _live_successors(task_class, effective)
    formal_core_final = original._live_formal_core(
        task_class,
        ffnet_binding=attempt["bindings"]["ffnet_recovery"],
        coformer_expected=ffnet_receipt["coformer_completed_evidence"],
    )
    return original._seal(
        {
            "schema_version": 1,
            "receipt_type": original.CANDIDATE_RECEIPT_TYPE,
            "generated_at_utc": original.shared._utc_timestamp(),
            "status": "candidate_recovered_and_enqueued",
            "remote_state_changed": True,
            "same_id_only": True,
            "replacement_controller_created": False,
            "plan_policy": "candidate_after_recovered_exact_original_plan_wlas",
            "attempt_receipt_path": str(attempt_path.resolve()),
            "attempt_receipt_seal_sha256": attempt["seal_sha256"],
            "bindings": dict(attempt["bindings"]),
            "effective_successor_authority_seal_sha256": effective[
                "seal_sha256"
            ],
            "candidate_controller_task_id": original.CANDIDATE_CONTROLLER_TASK_ID,
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
    parser = original._parser()
    parser.set_defaults(phase="candidate")
    parser.add_argument(
        "--successor-runtime-recovery-receipt",
        type=Path,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase != "candidate":
        raise CandidateAfterSuccessorRecoveryError("only candidate phase is supported")
    if args.execute and args.execute_token != original.CANDIDATE_EXECUTE_TOKEN:
        raise CandidateAfterSuccessorRecoveryError(
            f"exact execute token required: {original.CANDIDATE_EXECUTE_TOKEN}"
        )
    if not args.execute and args.execute_token:
        raise CandidateAfterSuccessorRecoveryError(
            "--execute-token requires --execute"
        )
    if args.ffnet_recovery_receipt is None:
        raise CandidateAfterSuccessorRecoveryError("FFNet recovery receipt is required")
    if args.execute != (args.attempt_receipt is not None):
        raise CandidateAfterSuccessorRecoveryError(
            "--attempt-receipt is required exactly with --execute"
        )
    outputs = [args.receipt]
    if args.attempt_receipt is not None:
        outputs.append(args.attempt_receipt)
    if len({item.resolve() for item in outputs}) != len(outputs):
        raise CandidateAfterSuccessorRecoveryError(
            "attempt and final receipt paths must differ"
        )
    for path in outputs:
        original._require_new(path, context="candidate recovery output")
    try:
        from clearml import Task
    except ImportError as error:
        raise CandidateAfterSuccessorRecoveryError("ClearML is required") from error
    attempt, effective = build_candidate_attempt(
        Task,
        evaluation_recovery_receipt=args.evaluation_recovery_receipt,
        ffnet_recovery_receipt=args.ffnet_recovery_receipt,
        successor_deployment_receipt=args.successor_deployment_receipt,
        successor_runtime_recovery_receipt=(
            args.successor_runtime_recovery_receipt
        ),
    )
    if not args.execute:
        plan = original._seal(
            {
                **attempt,
                "receipt_type": original.CANDIDATE_PLAN_TYPE,
                "status": "planned",
                "execute_token": original.CANDIDATE_EXECUTE_TOKEN,
            }
        )
        original._write_new(args.receipt, plan)
        original._summary(args.receipt, plan)
        return 0
    assert args.attempt_receipt is not None
    original._write_new(args.attempt_receipt, attempt)
    ffnet_receipt, _ = original.shared._read_sealed(
        args.ffnet_recovery_receipt,
        context="original queue FFNet recovery receipt",
    )
    result = execute_candidate_recovery(
        Task,
        attempt=attempt,
        attempt_path=args.attempt_receipt,
        successor_authority=effective,
        ffnet_receipt=ffnet_receipt,
        sleeper=original.shared.time.sleep,
    )
    original._write_new(args.receipt, result)
    original._summary(args.receipt, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CandidateAfterSuccessorRecoveryError",
    "build_candidate_attempt",
    "execute_candidate_recovery",
    "main",
)
