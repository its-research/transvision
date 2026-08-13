from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import deploy_clearml_formal_successor_chain as deploy
from tools.resilient_v2x import (
    recover_clearml_original_queue_runtime_failure as recovery,
)


def _artifact(name: str, index: int) -> dict[str, object]:
    return {
        "name": name,
        "hash": f"{index:x}" * 64,
        "size_bytes": index + 1,
        "url": f"http://10.100.35.118:8081/{name}.json",
    }


def _coformer() -> dict[str, object]:
    artifacts = [
        _artifact(name, index)
        for index, name in enumerate(sorted(recovery.COFORMER_ARTIFACTS), start=1)
    ]
    return {
        "task_id": recovery.COFORMER_TASK_ID,
        "subject": "coformernet",
        "status": "completed",
        "queue": "GPU4-A100",
        "queue_id": recovery.shared.formal.EXPECTED_QUEUE_IDS["GPU4-A100"],
        "artifact_names": [item["name"] for item in artifacts],
        "artifact_inventory": artifacts,
    }


def test_coformer_requires_completed_exact_hash_and_size_evidence() -> None:
    observed = recovery._coformer_completed_record({"records": [_coformer()]})
    assert observed["status"] == "completed"
    assert observed["artifact_inventory_sha256"] == recovery._content_sha256(
        observed["artifact_inventory"]
    )

    broken = _coformer()
    broken["artifact_inventory"][0]["size_bytes"] = 0
    with pytest.raises(recovery.OriginalQueueRecoveryError, match="hash/size/URL"):
        recovery._coformer_completed_record({"records": [broken]})

    broken = _coformer()
    broken["status"] = "in_progress"
    with pytest.raises(recovery.OriginalQueueRecoveryError, match="completed"):
        recovery._coformer_completed_record({"records": [broken]})


def _original_binding() -> dict[str, object]:
    return {
        "path": "/tmp/original-ffnet-recovery.json",
        "receipt_seal_sha256": "1" * 64,
        "attempt_receipt_seal_sha256": "2" * 64,
        "ffnet_task_id": recovery.FFNET_TASK_ID,
        "ffnet_source_sha256": "3" * 64,
        "ffnet_parameters_sha256": "4" * 64,
        "ffnet_planned_queue": "GPU4-V100",
        "ffnet_created_unqueued": True,
        "coformer_task_id": recovery.COFORMER_TASK_ID,
        "coformer_artifact_inventory_sha256": "5" * 64,
        "coformer_completed": True,
        "same_ids_only": True,
        "replacement_tasks_created": False,
    }


def test_successor_parameters_pin_original_recovery_for_every_role() -> None:
    ids = {
        "P": deploy.COMPLETED_PROVENANCE_TASK_ID,
        "W": "a" * 32,
        "L": "b" * 32,
        "A": "c" * 32,
        "S": "d" * 32,
    }
    hashes = {role: f"{index:x}" * 64 for index, role in enumerate(ids, start=1)}
    binding = _original_binding()
    values = deploy._parameters(
        ids,
        hashes,
        authoritative_recovery=True,
        original_queue_recovery_binding=binding,
    )
    for role in ("W", "L", "A", "S"):
        assert values[role][
            "Args/original_queue_runtime_recovery_receipt_path"
        ] == binding["path"]
        assert values[role][
            "Args/original_queue_runtime_recovery_receipt_seal_sha256"
        ] == binding["receipt_seal_sha256"]
        assert values[role]["Args/original_queue_runtime_ffnet_task_id"] == (
            recovery.FFNET_TASK_ID
        )
        assert values[role]["Args/original_queue_runtime_ffnet_planned_queue"] == (
            "GPU4-V100"
        )
        assert values[role]["Args/original_queue_runtime_coformer_task_id"] == (
            recovery.COFORMER_TASK_ID
        )


def test_original_and_amended_recovery_are_mutually_exclusive() -> None:
    ids = {
        "P": deploy.COMPLETED_PROVENANCE_TASK_ID,
        "W": "a" * 32,
        "L": "b" * 32,
        "A": "c" * 32,
        "S": "d" * 32,
    }
    hashes = {role: "a" * 64 for role in ids}
    with pytest.raises(deploy.DeploymentError, match="mutually exclusive"):
        deploy._parameters(
            ids,
            hashes,
            authoritative_recovery=True,
            amendment_binding={},
            original_queue_recovery_binding=_original_binding(),
        )


def _ffnet_attempt() -> dict[str, object]:
    coformer = recovery._coformer_completed_record({"records": [_coformer()]})
    return recovery._seal(
        {
            "ffnet_material": {
                "task_id": recovery.FFNET_TASK_ID,
                "task_name": "ffnet",
                "parent_task_id": "f" * 32,
            },
            "candidate_preserved_material": {
                "task_id": recovery.CANDIDATE_CONTROLLER_TASK_ID,
                "task_name": "candidate",
                "parent_task_id": "c" * 32,
            },
            "exact_name_inventories": {
                "FFNet": [recovery.FFNET_TASK_ID],
                "C": [recovery.CANDIDATE_CONTROLLER_TASK_ID],
            },
            "old_successor_failure_evidence": {
                role: {"task_id": role} for role in ("W", "L", "A", "S")
            },
            "evaluation_inventory": {"records": [coformer]},
            "coformer_completed_evidence": coformer,
            "bindings": {},
        }
    )


def test_ffnet_phase_mutates_only_ffnet_and_leaves_it_unqueued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _ffnet_attempt()
    attempt_path = tmp_path / "ffnet-attempt.json"
    recovery._write_new(attempt_path, attempt)
    events: list[str] = []
    monkeypatch.setattr(recovery, "_validate_ffnet_attempt", lambda _value: None)
    monkeypatch.setattr(
        recovery.shared,
        "_exact_name_inventory",
        lambda _tc, *, expected_task_id, **_kwargs: [expected_task_id],
    )
    monkeypatch.setattr(
        recovery.shared,
        "_authoritative_task",
        lambda *_args, **_kwargs: SimpleNamespace(id="old-w"),
    )
    monkeypatch.setattr(
        recovery.shared,
        "_validate_w_plan",
        lambda _task: {"seal_sha256": recovery.shared.EXPECTED_W_PLAN_SEAL},
    )
    frozen_successors = {
        role: {"task_id": role} for role in ("W", "L", "A", "S")
    }
    monkeypatch.setattr(
        recovery.shared,
        "_revalidate_old_successor_failures",
        lambda *_args, **_kwargs: frozen_successors,
    )
    monkeypatch.setattr(
        recovery,
        "_candidate_still_failed",
        lambda *_args, **_kwargs: {"status": "failed"},
    )
    preserved = {
        "records": [attempt["coformer_completed_evidence"]],
        "task_count": 27,
    }
    monkeypatch.setattr(
        recovery.shared,
        "_revalidate_preserved_evaluations",
        lambda *_args, **_kwargs: preserved,
    )

    def reset(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append(str(_kwargs["label"]))
        return {
            "task_id": recovery.FFNET_TASK_ID,
            "status": "created",
            "queue": None,
            "queue_id": None,
            "source_sha256": recovery.shared.EXPECTED_SOURCE_SHA256["FFNet"],
            "parameters_sha256": recovery.shared.EXPECTED_PARAMETERS_SHA256[
                "FFNet"
            ],
        }

    monkeypatch.setattr(recovery.shared, "_reset_and_install", reset)
    result = recovery.execute_ffnet_recovery(
        object(),
        attempt=attempt,
        attempt_path=attempt_path,
        sleeper=lambda _seconds: None,
    )
    assert events == ["FFNet"]
    assert result["ffnet_authority"]["status"] == "created"
    assert result["ffnet_enqueued"] is False
    assert result["candidate_mutated"] is False
    assert result["coformer_mutated"] is False


def _successors(status: str = "queued") -> dict[str, object]:
    return {
        role: {
            "task_id": role.lower() * 32,
            "status": status,
            "queue": "services",
            "queue_id": deploy.SERVICES_QUEUE_ID,
            "source_sha256": f"{index:x}" * 64,
            "parameters_sha256": f"{index + 4:x}" * 64,
            "artifact_inventory": [],
        }
        for index, role in enumerate(("W", "L", "A", "S"), start=1)
    }


def test_successor_binding_uses_sealed_script_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = {"receipt_seal_sha256": "e" * 64}
    ffnet = {"receipt_seal_sha256": "f" * 64}
    task_ids = {
        "P": deploy.COMPLETED_PROVENANCE_TASK_ID,
        "W": "1" * 32,
        "L": "2" * 32,
        "A": "3" * 32,
        "S": "4" * 32,
    }
    parents = {
        "W": task_ids["P"],
        "L": task_ids["W"],
        "A": task_ids["L"],
        "S": task_ids["A"],
    }
    sources = {
        role: {"deployed_sha256": f"{index:x}" * 64}
        for index, role in enumerate(("W", "L", "A", "S"), start=1)
    }
    parameters = {role: {"Args/value": role} for role in ("W", "L", "A", "S")}
    authority = {
        role: {
            "task_id": task_ids[role],
            "parent_task_id": parents[role],
            "queue": deploy.SERVICES_QUEUE,
            "script_sha256": sources[role]["deployed_sha256"],
            "parameters_sha256": recovery._content_sha256(parameters[role]),
        }
        for role in ("W", "L", "A", "S")
    }
    receipt = {
        "schema_version": 1,
        "receipt_type": "resilient_v2x_formal_successor_chain_deployment",
        "mode": "completed_provenance_successor_execute",
        "status": "deployed",
        "remote_state_changed": True,
        "created_roles": ["W", "L", "A", "S"],
        "enqueue_order": ["W", "L", "A", "S"],
        "evaluation_recovery_receipt": evaluation,
        "original_queue_runtime_recovery": ffnet,
        "task_ids": task_ids,
        "parents": parents,
        "sources": sources,
        "parameters": parameters,
        "queued_authority": authority,
        "seal_sha256": "a" * 64,
    }
    path = tmp_path / "successors.json"
    monkeypatch.setattr(
        recovery.shared,
        "_read_sealed",
        lambda candidate, **_kwargs: (receipt, candidate.resolve()),
    )
    binding, observed = recovery._successor_binding(
        path,
        evaluation_binding=evaluation,
        ffnet_binding=ffnet,
    )
    assert binding["task_ids"] == {role: task_ids[role] for role in ("W", "L", "A", "S")}
    assert observed is receipt


def test_candidate_phase_requires_fresh_wlas_before_reset_and_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = {
        "task_id": recovery.CANDIDATE_CONTROLLER_TASK_ID,
        "task_name": "candidate",
        "parent_task_id": "c" * 32,
    }
    frozen_successors = _successors()
    attempt = recovery._seal(
        {
            "candidate_material": candidate,
            "exact_name_inventory": [recovery.CANDIDATE_CONTROLLER_TASK_ID],
            "fresh_successor_authority": frozen_successors,
            "bindings": {"ffnet_recovery": _original_binding()},
        }
    )
    attempt_path = tmp_path / "candidate-attempt.json"
    recovery._write_new(attempt_path, attempt)
    events: list[str] = []
    monkeypatch.setattr(recovery, "_validate_candidate_attempt", lambda _value: None)
    monkeypatch.setattr(
        recovery.shared,
        "_exact_name_inventory",
        lambda *_args, **_kwargs: [recovery.CANDIDATE_CONTROLLER_TASK_ID],
    )

    def live_successors(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("live-wlas")
        return frozen_successors

    monkeypatch.setattr(recovery, "_live_successors", live_successors)
    monkeypatch.setattr(
        recovery,
        "_live_formal_core",
        lambda *_args, **_kwargs: events.append("live-core") or {"ready": False},
    )
    monkeypatch.setattr(
        recovery.shared,
        "_validate_failed_target_against_material",
        lambda *_args, **_kwargs: events.append("failed-c-guard"),
    )
    monkeypatch.setattr(
        recovery.shared,
        "_reset_and_install",
        lambda *_args, **_kwargs: events.append("reset-c")
        or {"status": "created", "queue": None, "queue_id": None},
    )
    monkeypatch.setattr(
        recovery.candidate_deploy,
        "_prepare",
        lambda: {"source": "source", "parameters": {}},
    )
    controller = SimpleNamespace(id=recovery.CANDIDATE_CONTROLLER_TASK_ID)
    monkeypatch.setattr(
        recovery.shared,
        "_authoritative_task",
        lambda *_args, **_kwargs: controller,
    )
    monkeypatch.setattr(
        recovery.candidate_deploy,
        "_validate_task",
        lambda *_args, **_kwargs: events.append("created-c-guard") or {},
    )
    monkeypatch.setattr(
        recovery.candidate_deploy,
        "_enqueue",
        lambda *_args, **_kwargs: events.append("enqueue-c")
        or {"status": "in_progress", "queue": "services"},
    )
    result = recovery.execute_candidate_recovery(
        object(),
        attempt=attempt,
        attempt_path=attempt_path,
        successor_receipt={},
        ffnet_receipt={"coformer_completed_evidence": _coformer()},
        sleeper=lambda _seconds: None,
    )
    assert events.index("live-wlas") < events.index("reset-c")
    assert events.index("failed-c-guard") < events.index("reset-c")
    assert events.index("created-c-guard") < events.index("enqueue-c")
    assert result["candidate_controller_task_id"] == (
        recovery.CANDIDATE_CONTROLLER_TASK_ID
    )
    assert result["replacement_controller_created"] is False


def test_wrong_tokens_fail_before_clearml_or_receipt_read(tmp_path: Path) -> None:
    common = [
        "--phase",
        "ffnet",
        "--execute",
        "--execute-token",
        "wrong",
        "--evaluation-recovery-receipt",
        str(tmp_path / "evaluation.json"),
        "--successor-deployment-receipt",
        str(tmp_path / "successor.json"),
        "--controller-recovery-receipt",
        str(tmp_path / "controller.json"),
        "--candidate-deployment-receipt",
        str(tmp_path / "candidate.json"),
        "--attempt-receipt",
        str(tmp_path / "attempt.json"),
        "--receipt",
        str(tmp_path / "final.json"),
    ]
    with pytest.raises(recovery.OriginalQueueRecoveryError, match="exact execute token"):
        recovery.main(common)
