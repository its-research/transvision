from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import (
    recover_clearml_original_queue_candidate_after_successor_recovery as bridge,
)


def _effective_authority() -> dict[str, object]:
    task_ids = {
        "P": bridge.successor_recovery.deploy.COMPLETED_PROVENANCE_TASK_ID,
        **bridge.successor_recovery.SUCCESSOR_TASK_IDS,
    }
    parents = {
        "W": task_ids["P"],
        "L": task_ids["W"],
        "A": task_ids["L"],
        "S": task_ids["A"],
    }
    sources = {
        role: hashlib.sha256(f"source:{role}".encode()).hexdigest()
        for role in bridge._ROLES
    }
    values = {
        role: {"Args/value": role} for role in bridge._ROLES
    }
    parameter_hashes = {
        role: bridge.successor_recovery._content_sha256(values[role])
        for role in bridge._ROLES
    }
    original_deployment = {
        "path": "/tmp/original-successor-deployment.json",
        "receipt_seal_sha256": "d" * 64,
    }
    recovery_binding = {
        "path": "/tmp/successor-runtime-recovery.json",
        "receipt_seal_sha256": "e" * 64,
        "original_successor_deployment": original_deployment,
        "attempt_receipt_seal_sha256": "a" * 64,
        "task_ids": dict(bridge.successor_recovery.SUCCESSOR_TASK_IDS),
        "source_sha256": sources,
        "parameters_sha256": parameter_hashes,
        "same_ids_only": True,
        "replacement_tasks_created": False,
    }
    environment = {
        "policy": {"kind": "services"},
        "requirements": {"form": "agent_materialized"},
        "docker": "image",
        "output_uri": "http://files",
        "binary": "python",
    }
    roles = {
        role: {
            "task_id": task_ids[role],
            "status": "queued",
            "project_id": bridge.original.shared.evaluation_recovery.PROJECT_ID,
            "task_name": bridge.successor_recovery.deploy.ROLE_DEFINITIONS[role][
                "task_name"
            ],
            "task_type": "controller",
            "parent_task_id": parents[role],
            "queue": bridge.successor_recovery.deploy.SERVICES_QUEUE,
            "queue_id": bridge.successor_recovery.deploy.SERVICES_QUEUE_ID,
            "source_sha256": sources[role],
            "parameters_sha256": parameter_hashes[role],
            "artifact_inventory": [],
            "model_inventory": {"input": [], "output": []},
            "tags": ["formal"],
            "system_tags": ["development"],
            "service_environment": environment,
        }
        for role in bridge._ROLES
    }
    return bridge.successor_recovery._seal(
        {
            "schema_version": 1,
            "authority_type": bridge.successor_recovery.EFFECTIVE_AUTHORITY_TYPE,
            "recovery_binding": recovery_binding,
            "original_successor_deployment": original_deployment,
            "task_ids": task_ids,
            "parents": parents,
            "sources": sources,
            "parameters": {
                role: {
                    "values": values[role],
                    "sha256": parameter_hashes[role],
                }
                for role in bridge._ROLES
            },
            "roles": roles,
        }
    )


def _live_tasks(authority: dict[str, object]) -> dict[str, SimpleNamespace]:
    tasks: dict[str, SimpleNamespace] = {}
    for role in bridge._ROLES:
        row = authority["roles"][role]
        tasks[role] = SimpleNamespace(
            id=authority["task_ids"][role],
            name=row["task_name"],
            task_type="TaskTypes.controller",
            status=row["status"],
            parent=row["parent_task_id"],
            source=f"source:{role}",
            parameters=dict(authority["parameters"][role]["values"]),
            queue=row["queue"],
            queue_id=row["queue_id"],
            artifacts=copy.deepcopy(row["artifact_inventory"]),
            models=copy.deepcopy(row["model_inventory"]),
            tags=list(row["tags"]),
            system_tags=list(row["system_tags"]),
            environment=copy.deepcopy(row["service_environment"]),
        )
    return tasks


def _patch_live_reads(
    monkeypatch: pytest.MonkeyPatch,
    authority: dict[str, object],
    tasks: dict[str, SimpleNamespace],
) -> None:
    by_id = {task.id: task for task in tasks.values()}
    monkeypatch.setattr(
        bridge.original.shared,
        "_authoritative_task",
        lambda _task_class, task_id, **_kwargs: by_id[task_id],
    )
    monkeypatch.setattr(
        bridge.original.shared, "_status", lambda task: task.status
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_source",
        lambda task, **_kwargs: task.source,
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_parameters",
        lambda task, **_kwargs: task.parameters,
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_queue_identity",
        lambda task, **_kwargs: (task.queue, task.queue_id),
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_artifact_inventory",
        lambda task: task.artifacts,
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_model_inventory",
        lambda task, **_kwargs: task.models,
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_tags",
        lambda task, *, field, **_kwargs: getattr(task, field),
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_service_environment",
        lambda task, **_kwargs: task.environment,
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_exact_project_id",
        lambda _task, **_kwargs: bridge.original.shared.evaluation_recovery.PROJECT_ID,
    )
    monkeypatch.setattr(
        bridge.successor_recovery.deploy,
        "_task_parent",
        lambda task: task.parent,
    )


def test_live_validator_accepts_only_declared_runtime_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _effective_authority()
    tasks = _live_tasks(authority)
    tasks["L"].parameters[
        "Args/evaluation_plan_amendment_producer_task_id"
    ] = ""
    _patch_live_reads(monkeypatch, authority, tasks)
    observed = bridge._live_successors(object(), authority)
    assert set(observed) == set(bridge._ROLES)
    assert observed["L"]["parameters_sha256"] == authority["parameters"]["L"][
        "sha256"
    ]


def test_live_validator_accepts_agent_materialized_service_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _effective_authority()
    for role in bridge._ROLES:
        environment = authority["roles"][role]["service_environment"]
        environment["policy"][
            "requirement"
        ] = "clearml==2.1.11"
        environment["requirements"] = {
            "form": "canonical",
            "sha256": "c" * 64,
            "canonical_requirement": "clearml==2.1.11",
        }
    authority = bridge.successor_recovery._seal(authority)
    tasks = _live_tasks(authority)
    for task in tasks.values():
        task.status = "in_progress"
        task.environment["requirements"] = {
            "form": "agent_materialized",
            "sha256": "a" * 64,
            "canonical_requirement": "clearml==2.1.11",
            "resolved_count": 1,
            "resolved_packages": ["clearml==2.1.11"],
        }
    _patch_live_reads(monkeypatch, authority, tasks)
    observed = bridge._live_successors(object(), authority)
    assert all(
        observed[role]["service_environment"]["requirements"]["form"]
        == "agent_materialized"
        for role in bridge._ROLES
    )


@pytest.mark.parametrize(
    "drift",
    (
        "status",
        "parent",
        "source",
        "unknown_parameter",
        "known_parameter_default",
        "queue_id",
        "artifact",
        "model",
        "tags",
        "environment",
    ),
)
def test_live_validator_fails_closed_on_each_authority_plane(
    monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    authority = _effective_authority()
    tasks = _live_tasks(authority)
    task = tasks["W"]
    if drift == "status":
        task.status = "failed"
    elif drift == "parent":
        task.parent = "f" * 32
    elif drift == "source":
        task.source = "changed"
    elif drift == "unknown_parameter":
        task.parameters["Args/unknown_runtime_default"] = ""
    elif drift == "known_parameter_default":
        task.parameters["Args/exact_eval_runtime_recovery_receipt_seal_sha256"] = (
            "f" * 64
        )
    elif drift == "queue_id":
        task.queue_id = "f" * 32
    elif drift == "artifact":
        task.artifacts = [{"name": "unknown"}]
    elif drift == "model":
        task.models = {"input": [], "output": ["f" * 32]}
    elif drift == "tags":
        task.tags.append("unexpected")
    elif drift == "environment":
        task.environment["docker"] = "changed"
    _patch_live_reads(monkeypatch, authority, tasks)
    with pytest.raises(bridge.CandidateAfterSuccessorRecoveryError):
        bridge._live_successors(object(), authority)


def test_effective_authority_must_be_sealed_and_have_no_unknown_overlay() -> None:
    authority = _effective_authority()
    bridge._validated_effective_authority(authority)
    changed = copy.deepcopy(authority)
    changed["queued_authority"] = {"W": {"source_sha256": "0" * 64}}
    changed = bridge.successor_recovery._seal(changed)
    with pytest.raises(
        bridge.CandidateAfterSuccessorRecoveryError, match="key inventory"
    ):
        bridge._validated_effective_authority(changed)


def test_candidate_execution_preserves_exact_id_guard_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    effective = {"seal_sha256": "e" * 64}
    live = {
        role: {
            "task_id": task_id,
            "status": "in_progress",
            "queue": "services",
            "queue_id": bridge.successor_recovery.deploy.SERVICES_QUEUE_ID,
            "source_sha256": role.lower() * 64,
            "parameters_sha256": role.lower() * 64,
            "artifact_inventory": [],
        }
        for role, task_id in bridge.successor_recovery.SUCCESSOR_TASK_IDS.items()
    }
    material = {
        "task_id": bridge.original.CANDIDATE_CONTROLLER_TASK_ID,
        "task_name": "candidate",
        "parent_task_id": "c" * 32,
    }
    attempt = bridge.original._seal(
        {
            "candidate_material": material,
            "exact_name_inventory": [bridge.original.CANDIDATE_CONTROLLER_TASK_ID],
            "fresh_successor_authority": live,
            "effective_successor_authority": effective,
            "bindings": {"ffnet_recovery": {}},
        }
    )
    attempt_path = tmp_path / "attempt.json"
    bridge.original._write_new(attempt_path, attempt)
    events: list[str] = []
    monkeypatch.setattr(bridge, "_validate_candidate_attempt", lambda _value: None)
    monkeypatch.setattr(
        bridge, "_validated_effective_authority", lambda value: dict(value)
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_exact_name_inventory",
        lambda *_args, **_kwargs: [bridge.original.CANDIDATE_CONTROLLER_TASK_ID],
    )
    monkeypatch.setattr(
        bridge,
        "_live_successors",
        lambda *_args, **_kwargs: events.append("live-wlas") or live,
    )
    monkeypatch.setattr(
        bridge.original,
        "_successors_advance_monotonically",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        bridge.original,
        "_live_formal_core",
        lambda *_args, **_kwargs: events.append("live-core") or {},
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_validate_failed_target_against_material",
        lambda *_args, **_kwargs: events.append("failed-c-guard"),
    )
    monkeypatch.setattr(
        bridge.original.shared,
        "_reset_and_install",
        lambda *_args, **_kwargs: events.append("reset-c")
        or {"status": "created", "queue": None, "queue_id": None},
    )
    monkeypatch.setattr(
        bridge.original.candidate_deploy,
        "_prepare",
        lambda: {"source": "source", "parameters": {}},
    )
    controller = SimpleNamespace(id=bridge.original.CANDIDATE_CONTROLLER_TASK_ID)
    monkeypatch.setattr(
        bridge.original.shared,
        "_authoritative_task",
        lambda *_args, **_kwargs: controller,
    )
    monkeypatch.setattr(
        bridge.original.candidate_deploy,
        "_validate_task",
        lambda *_args, **_kwargs: events.append("created-c-guard"),
    )
    monkeypatch.setattr(
        bridge.original.candidate_deploy,
        "_enqueue",
        lambda *_args, **_kwargs: events.append("enqueue-c")
        or {"status": "in_progress", "queue": "services"},
    )
    result = bridge.execute_candidate_recovery(
        object(),
        attempt=attempt,
        attempt_path=attempt_path,
        successor_authority=effective,
        ffnet_receipt={"coformer_completed_evidence": {}},
        sleeper=lambda _seconds: None,
    )
    assert events.index("live-wlas") < events.index("reset-c")
    assert events.index("failed-c-guard") < events.index("reset-c")
    assert events.index("created-c-guard") < events.index("enqueue-c")
    assert result["candidate_controller_task_id"] == (
        bridge.original.CANDIDATE_CONTROLLER_TASK_ID
    )
    assert result["replacement_controller_created"] is False
