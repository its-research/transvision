from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "tools/resilient_v2x/recover_clearml_formal_successor_runtime_failure.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "recover_clearml_formal_successor_runtime_failure_under_test", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_exact_id_orders_and_token() -> None:
    module = _load_module()
    assert module.SUCCESSOR_TASK_IDS == {
        "W": "74bf35decc2340b496167c11ec50f54b",
        "L": "cc7b54e4dcca4b92a4247f92987add60",
        "A": "dd287fe58e264bcca48fe222b6981048",
        "S": "d7ea54ce540d4b0486904c5910100885",
    }
    assert module.RESET_ORDER == ("S", "A", "L", "W")
    assert module.CONFIGURE_ORDER == ("W", "L", "A", "S")
    assert module.ENQUEUE_ORDER == ("W", "L", "A", "S")
    assert module.EXECUTE_TOKEN.endswith(module.SUCCESSOR_TASK_IDS["W"].upper())


def test_w_failure_signatures_include_transient_queue_identity() -> None:
    module = _load_module()
    assert module.EXPECTED_FAILURE_MESSAGES["W"] == (
        "candidate E3 evaluation queue drifted",
        "queue occupancy execution queue must be a lowercase 32-hex ClearML ID",
    )


def test_runtime_parameter_defaults_are_narrow_and_fail_closed() -> None:
    module = _load_module()
    declared = {"Args/watcher_task_id": "a" * 32}
    observed = {
        **declared,
        "Args/evaluation_plan_amendment_producer_task_id": "",
        "Args/exact_eval_runtime_recovery_receipt_seal_sha256": "",
    }
    assert module._normalized_parameters("L", observed) == declared
    observed["Args/exact_eval_runtime_recovery_receipt_seal_sha256"] = "b" * 64
    with pytest.raises(module.SuccessorRuntimeRecoveryError, match="runtime default"):
        module._normalized_parameters("L", observed)


def test_failed_artifact_allowlist_is_role_specific() -> None:
    module = _load_module()
    allowed = {
        role: set(module.deploy.ROLE_OUTPUT_ARTIFACTS[role])
        for role in module.CONFIGURE_ORDER
    }
    assert allowed == {
        "W": {"formal_1337_evaluation_plan"},
        "L": {"formal_1337_leaderboard"},
        "A": {"formal_1337_comparability_audit"},
        "S": {"final_selector_formal_inputs", "formal_candidate_selection"},
    }
    assert "unexpected_artifact" not in set().union(*allowed.values())


def test_attempt_contract_requires_reverse_reset_and_same_ids() -> None:
    module = _load_module()
    attempt = module._seal(
        {
            "schema_version": 1,
            "receipt_type": module.ATTEMPT_RECEIPT_TYPE,
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "task_ids": dict(module.SUCCESSOR_TASK_IDS),
            "reset_order": list(module.RESET_ORDER),
            "configure_order": list(module.CONFIGURE_ORDER),
            "enqueue_order": list(module.ENQUEUE_ORDER),
        }
    )
    module._validate_attempt(attempt)
    changed = dict(attempt)
    changed["reset_order"] = list(module.CONFIGURE_ORDER)
    changed = module._seal(changed)
    with pytest.raises(module.SuccessorRuntimeRecoveryError, match="identity"):
        module._validate_attempt(changed)


def test_wrong_execute_token_fails_before_clearml_or_receipt_read(
    tmp_path: Path,
) -> None:
    module = _load_module()
    with pytest.raises(module.SuccessorRuntimeRecoveryError, match="exact execute token"):
        module.main(
            [
                "--execute",
                "--execute-token",
                "wrong",
                "--successor-deployment-receipt",
                str(tmp_path / "deployment.json"),
                "--attempt-receipt",
                str(tmp_path / "attempt.json"),
                "--receipt",
                str(tmp_path / "receipt.json"),
            ]
        )


def test_execute_order_has_all_created_barrier_before_first_enqueue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    events: list[str] = []
    attempt = module._seal(
        {
            "schema_version": 1,
            "receipt_type": module.ATTEMPT_RECEIPT_TYPE,
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "task_ids": dict(module.SUCCESSOR_TASK_IDS),
            "reset_order": list(module.RESET_ORDER),
            "configure_order": list(module.CONFIGURE_ORDER),
            "enqueue_order": list(module.ENQUEUE_ORDER),
            "original_successor_deployment": {
                "path": str((tmp_path / "deployment.json").resolve()),
                "receipt_seal_sha256": "d" * 64,
            },
                "pre_reset_authority": {
                role: {
                    "tags": [],
                    "system_tags": [],
                    "exact_name_inventory": [task_id],
                }
                    for role, task_id in module.SUCCESSOR_TASK_IDS.items()
                },
                "dependency_authority": {
                    "references": {},
                    "completed_provenance": {},
                    "formal_core": {},
                },
                "target": {"fixed": True},
        }
    )
    attempt_path = tmp_path / "attempt.json"
    module._write_new(attempt_path, attempt)
    deployment = {
        "seal_sha256": "d" * 64,
        "task_ids": {
            "P": module.deploy.COMPLETED_PROVENANCE_TASK_ID,
            **module.SUCCESSOR_TASK_IDS,
        },
    }
    target = {
        "parents": {role: "p" for role in module.CONFIGURE_ORDER},
        "sources": {
            role: {"deployed_sha256": role.lower() * 64}
            for role in module.CONFIGURE_ORDER
        },
        "rendered_sources": {role: role for role in module.CONFIGURE_ORDER},
        "parameters": {role: {} for role in module.CONFIGURE_ORDER},
        "raw_parameters": {role: {} for role in module.CONFIGURE_ORDER},
        "standalone_import_smoke": {},
        "base_sources": {},
        "embedded_sources": {},
    }
    monkeypatch.setattr(
        module,
        "_load_deployment",
        lambda *_args, **_kwargs: (
            deployment,
            Path(attempt["original_successor_deployment"]["path"]),
            {},
        ),
    )
    monkeypatch.setattr(module, "_target_contract", lambda _deployment: target)
    monkeypatch.setattr(
        module,
        "_dependency_authority",
        lambda *_args, **_kwargs: attempt["dependency_authority"],
    )
    monkeypatch.setattr(
        module,
        "_reference_and_provenance_authority",
        lambda *_args, **_kwargs: {
            "references": {},
            "completed_provenance": {},
        },
    )
    monkeypatch.setattr(
        module,
        "_failed_snapshot",
        lambda *_args, role, **_kwargs: attempt["pre_reset_authority"][role],
    )
    attempt["target"] = {
        "parents": target["parents"],
        "sources": target["sources"],
        "parameters": target["parameters"],
        "standalone_import_smoke": target["standalone_import_smoke"],
    }
    attempt = module._seal(attempt)
    attempt_path.unlink()
    module._write_new(attempt_path, attempt)
    monkeypatch.setattr(
        module,
        "_reset_one",
        lambda *_args, role, **_kwargs: events.append(f"reset:{role}") or object(),
    )
    monkeypatch.setattr(
        module.deploy,
        "_configure_shell",
        lambda *_args, role, **_kwargs: events.append(f"configure:{role}"),
    )
    monkeypatch.setattr(module.shared, "_set_tags", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.deploy, "_assert_sources_unchanged", lambda *_a: None)
    created_calls: dict[str, int] = {role: 0 for role in module.CONFIGURE_ORDER}

    def created(*_args, role: str, **_kwargs):
        created_calls[role] += 1
        events.append(f"created{created_calls[role]}:{role}")
        return {"role": role}

    monkeypatch.setattr(module, "_created_authority", created)
    monkeypatch.setattr(
        module.shared, "_authoritative_task", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        module.deploy,
        "_enqueue",
        lambda *_args, role, **_kwargs: events.append(f"enqueue:{role}"),
    )
    monkeypatch.setattr(
        module,
        "_live_authority",
        lambda *_args, role, **_kwargs: {
            "task_id": module.SUCCESSOR_TASK_IDS[role],
            "status": "queued",
        },
    )
    monkeypatch.setattr(
        module,
        "_exact_name_inventory",
        lambda *_args, role, **_kwargs: [module.SUCCESSOR_TASK_IDS[role]],
    )
    result = module.execute_recovery(
        object(),
        attempt=attempt,
        attempt_path=attempt_path,
        successor_deployment_receipt=tmp_path / "deployment.json",
        sleeper=lambda _seconds: None,
    )
    assert [item for item in events if item.startswith("reset:")] == [
        "reset:S",
        "reset:A",
        "reset:L",
        "reset:W",
    ]
    first_enqueue = min(i for i, item in enumerate(events) if item.startswith("enqueue:"))
    assert all(
        events.index(f"created2:{role}") < first_enqueue
        for role in module.CONFIGURE_ORDER
    )
    assert [item for item in events if item.startswith("enqueue:")] == [
        "enqueue:W",
        "enqueue:L",
        "enqueue:A",
        "enqueue:S",
    ]
    assert result["same_ids_only"] is True
    assert result["replacement_tasks_created"] is False


def test_execute_revalidates_all_four_before_any_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    deployment_path = (tmp_path / "deployment.json").resolve()
    dependency = {
        "references": {},
        "completed_provenance": {},
        "formal_core": {},
    }
    pre_reset = {
        role: {
            "tags": [],
            "system_tags": [],
            "exact_name_inventory": [task_id],
        }
        for role, task_id in module.SUCCESSOR_TASK_IDS.items()
    }
    target = {
        "parents": {role: "p" for role in module.CONFIGURE_ORDER},
        "sources": {
            role: {"deployed_sha256": role.lower() * 64}
            for role in module.CONFIGURE_ORDER
        },
        "rendered_sources": {role: role for role in module.CONFIGURE_ORDER},
        "parameters": {role: {} for role in module.CONFIGURE_ORDER},
        "raw_parameters": {role: {} for role in module.CONFIGURE_ORDER},
        "standalone_import_smoke": {},
        "base_sources": {},
        "embedded_sources": {},
    }
    attempt = module._seal(
        {
            "schema_version": 1,
            "receipt_type": module.ATTEMPT_RECEIPT_TYPE,
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "task_ids": dict(module.SUCCESSOR_TASK_IDS),
            "reset_order": list(module.RESET_ORDER),
            "configure_order": list(module.CONFIGURE_ORDER),
            "enqueue_order": list(module.ENQUEUE_ORDER),
            "original_successor_deployment": {
                "path": str(deployment_path),
                "receipt_seal_sha256": "d" * 64,
            },
            "pre_reset_authority": pre_reset,
            "dependency_authority": dependency,
            "target": {
                "parents": target["parents"],
                "sources": target["sources"],
                "parameters": target["parameters"],
                "standalone_import_smoke": target["standalone_import_smoke"],
            },
        }
    )
    attempt_path = tmp_path / "attempt.json"
    module._write_new(attempt_path, attempt)
    deployment = {
        "seal_sha256": "d" * 64,
        "task_ids": {
            "P": module.deploy.COMPLETED_PROVENANCE_TASK_ID,
            **module.SUCCESSOR_TASK_IDS,
        },
    }
    monkeypatch.setattr(
        module,
        "_load_deployment",
        lambda *_args, **_kwargs: (deployment, deployment_path, {}),
    )
    monkeypatch.setattr(module, "_target_contract", lambda _deployment: target)
    monkeypatch.setattr(
        module,
        "_dependency_authority",
        lambda *_args, **_kwargs: dependency,
    )
    events: list[str] = []

    def failed_snapshot(*_args, role: str, **_kwargs):
        events.append(f"preflight:{role}")
        if role == "S":
            raise module.SuccessorRuntimeRecoveryError("S drifted")
        return pre_reset[role]

    monkeypatch.setattr(module, "_failed_snapshot", failed_snapshot)
    monkeypatch.setattr(
        module,
        "_reset_one",
        lambda *_args, role, **_kwargs: events.append(f"reset:{role}"),
    )
    with pytest.raises(module.SuccessorRuntimeRecoveryError, match="S drifted"):
        module.execute_recovery(
            object(),
            attempt=attempt,
            attempt_path=attempt_path,
            successor_deployment_receipt=deployment_path,
            sleeper=lambda _seconds: None,
        )
    assert events == [
        "preflight:W",
        "preflight:L",
        "preflight:A",
        "preflight:S",
    ]


def _valid_recovery_receipts(module, tmp_path: Path):
    deployment_path = (tmp_path / "deployment.json").resolve()
    parents = {
        "W": module.deploy.COMPLETED_PROVENANCE_TASK_ID,
        "L": module.SUCCESSOR_TASK_IDS["W"],
        "A": module.SUCCESSOR_TASK_IDS["L"],
        "S": module.SUCCESSOR_TASK_IDS["A"],
    }
    sources = {
        role: {"deployed_sha256": (role.lower() or "p") * 64}
        for role in module.deploy.ROLE_ORDER
    }
    parameters = {role: {} for role in module.CONFIGURE_ORDER}
    pre_reset = {
        role: {
            "tags": [],
            "system_tags": [],
            "exact_name_inventory": [task_id],
        }
        for role, task_id in module.SUCCESSOR_TASK_IDS.items()
    }
    dependency = {
        "references": {},
        "completed_provenance": {},
        "formal_core": {},
    }
    target = {
        "parents": parents,
        "sources": sources,
        "parameters": parameters,
        "standalone_import_smoke": {},
    }
    original_binding = {
        "path": str(deployment_path),
        "receipt_seal_sha256": "d" * 64,
    }
    attempt = module._seal(
        {
            "schema_version": 1,
            "receipt_type": module.ATTEMPT_RECEIPT_TYPE,
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "task_ids": dict(module.SUCCESSOR_TASK_IDS),
            "reset_order": list(module.RESET_ORDER),
            "configure_order": list(module.CONFIGURE_ORDER),
            "enqueue_order": list(module.ENQUEUE_ORDER),
            "original_successor_deployment": original_binding,
            "pre_reset_authority": pre_reset,
            "dependency_authority": dependency,
            "target": target,
        }
    )
    attempt_path = (tmp_path / "attempt.json").resolve()
    module._write_new(attempt_path, attempt)

    created = {}
    queued = {}
    final = {}
    for role in module.CONFIGURE_ORDER:
        stable = {
            "task_id": module.SUCCESSOR_TASK_IDS[role],
            "project_id": module.shared.evaluation_recovery.PROJECT_ID,
            "task_name": module.deploy.ROLE_DEFINITIONS[role]["task_name"],
            "task_type": "controller",
            "parent_task_id": parents[role],
            "source_sha256": sources[role]["deployed_sha256"],
            "parameters_sha256": module._content_sha256(parameters[role]),
            "model_inventory": {"input": [], "output": []},
            "tags": [],
            "system_tags": [],
            "service_environment": {},
        }
        created[role] = {
            **stable,
            "status": "created",
            "queue": None,
            "queue_id": None,
            "artifact_inventory": [],
        }
        queued[role] = {
            **stable,
            "status": "queued",
            "queue": module.deploy.SERVICES_QUEUE,
            "queue_id": module.deploy.SERVICES_QUEUE_ID,
            "artifact_inventory": [],
        }
        final[role] = dict(queued[role])
    recovery = module._seal(
        {
            "schema_version": 1,
            "receipt_type": module.RECOVERY_RECEIPT_TYPE,
            "status": "recovered_and_enqueued",
            "remote_state_changed": True,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "original_successor_deployment": original_binding,
            "attempt_receipt_path": str(attempt_path),
            "attempt_receipt_seal_sha256": attempt["seal_sha256"],
            "task_ids": dict(module.SUCCESSOR_TASK_IDS),
            "reset_order": list(module.RESET_ORDER),
            "configure_order": list(module.CONFIGURE_ORDER),
            "enqueue_order": list(module.ENQUEUE_ORDER),
            "sources": sources,
            "parameters": parameters,
            "parents": parents,
            "standalone_import_smoke": {},
            "dependency_authority_before_reset": dependency,
            "pre_reset_authority_revalidated": pre_reset,
            "reference_authority_before_enqueue": {
                "references": {},
                "completed_provenance": {},
            },
            "created_unqueued_authority": created,
            "queued_authority": queued,
            "final_authority": final,
            "exact_name_inventories": {
                role: [task_id]
                for role, task_id in module.SUCCESSOR_TASK_IDS.items()
            },
            "no_new_exact_name_task_ids": True,
        }
    )
    deployment = {
        "seal_sha256": "d" * 64,
        "task_ids": {
            "P": module.deploy.COMPLETED_PROVENANCE_TASK_ID,
            **module.SUCCESSOR_TASK_IDS,
        },
        "parents": parents,
        "parameters": parameters,
    }
    return recovery, attempt, deployment, deployment_path


def test_resealed_recovery_authority_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    recovery, _attempt, deployment, deployment_path = _valid_recovery_receipts(
        module, tmp_path
    )
    monkeypatch.setattr(
        module,
        "_load_deployment",
        lambda *_args, **_kwargs: (deployment, deployment_path, {}),
    )
    valid_path = tmp_path / "recovery.json"
    module._write_new(valid_path, recovery)
    module.validate_successor_runtime_recovery_receipt(valid_path)

    tampered = copy.deepcopy(recovery)
    tampered["final_authority"]["W"]["source_sha256"] = "f" * 64
    tampered = module._seal(tampered)
    tampered_path = tmp_path / "tampered.json"
    module._write_new(tampered_path, tampered)
    with pytest.raises(
        module.SuccessorRuntimeRecoveryError, match="stable authority drifted"
    ):
        module.validate_successor_runtime_recovery_receipt(tampered_path)


def test_effective_authority_never_overlays_stale_deployment_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    recovered_sources = {
        role: {"deployed_sha256": role.lower() * 64}
        for role in module.CONFIGURE_ORDER
    }
    recovered_parameters = {
        role: {"Args/recovered": role} for role in module.CONFIGURE_ORDER
    }
    recovered_roles = {
        role: {
            "task_id": module.SUCCESSOR_TASK_IDS[role],
            "status": "in_progress",
            "source_sha256": recovered_sources[role]["deployed_sha256"],
        }
        for role in module.CONFIGURE_ORDER
    }
    recovery_value = {
        "original_successor_deployment": {
            "path": str((tmp_path / "deployment.json").resolve()),
            "receipt_seal_sha256": "d" * 64,
        },
        "parents": {role: "p" for role in module.CONFIGURE_ORDER},
        "sources": recovered_sources,
        "parameters": recovered_parameters,
        "final_authority": recovered_roles,
    }
    binding = {
        "original_successor_deployment": dict(
            recovery_value["original_successor_deployment"]
        ),
        "receipt_seal_sha256": "r" * 64,
    }
    monkeypatch.setattr(
        module,
        "_load_validated_recovery",
        lambda _path: (recovery_value, _path.resolve(), {}),
    )
    monkeypatch.setattr(
        module,
        "validate_successor_runtime_recovery_receipt",
        lambda _path: binding,
    )
    stale = "0" * 64
    original_deployment = {
        "seal_sha256": "d" * 64,
        "task_ids": {
            "P": module.deploy.COMPLETED_PROVENANCE_TASK_ID,
            **module.SUCCESSOR_TASK_IDS,
        },
        "sources": {
            role: {"deployed_sha256": stale}
            for role in module.CONFIGURE_ORDER
        },
        "parameters": {
            role: {"Args/stale": "true"} for role in module.CONFIGURE_ORDER
        },
        "queued_authority": {
            role: {"source_sha256": stale}
            for role in module.CONFIGURE_ORDER
        },
    }
    effective = module.effective_successor_authority(
        tmp_path / "recovery.json",
        original_deployment_receipt=original_deployment,
    )
    assert "queued_authority" not in effective
    assert "0" * 64 not in json.dumps(effective, sort_keys=True)
    assert effective["sources"] == {
        role: recovered_sources[role]["deployed_sha256"]
        for role in module.CONFIGURE_ORDER
    }
    assert effective["roles"] == recovered_roles
    assert module._seal(effective)["seal_sha256"] == effective["seal_sha256"]
