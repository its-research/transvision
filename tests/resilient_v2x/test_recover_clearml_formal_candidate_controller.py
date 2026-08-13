from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from . import (
    test_deploy_clearml_formal_candidate_evaluation_queue as deploy_fakes,
)
from tools.resilient_v2x import (
    deploy_clearml_formal_candidate_evaluation_queue as deployment,
)
from tools.resilient_v2x import (
    recover_clearml_formal_candidate_controller as recovery,
)


def _evaluation_binding(tmp_path: Path) -> dict[str, object]:
    return {
        "path": str((tmp_path / "evaluation-recovery.json").resolve()),
        "receipt_seal_sha256": "1" * 64,
        "attempt_receipt_seal_sha256": "2" * 64,
        "authoritative_plan_seal_sha256": "3" * 64,
        "task_count": 28,
        "task_ids_sha256": "4" * 64,
        "all_tasks_created_unqueued": True,
        "replacement_tasks_created": False,
    }


def _install_legacy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reset_mode: str = "success",
) -> tuple[type[deploy_fakes._FakeTaskClass], deploy_fakes._FakeTask]:
    task_class = deploy_fakes._FakeTaskClass
    task_class.reset()
    task = deploy_fakes._FakeTask(
        task_class,
        task_id=recovery.CONTROLLER_TASK_ID,
        name=deployment.CONTROLLER_NAME,
        project_id=deployment.PROJECT_ID,
        project_name=deployment.PROJECT_NAME,
        status="stopped",
        task_type="controller",
    )
    task_class.registry[task.id] = task
    source = "# frozen legacy controller\n"
    source_sha = recovery.hashlib.sha256(source.encode("utf-8")).hexdigest()
    monkeypatch.setattr(recovery, "LEGACY_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(recovery, "LEGACY_SOURCE_BYTES", len(source.encode("utf-8")))
    monkeypatch.setattr(
        recovery,
        "_legacy_receipt_binding",
        lambda: {
            "path": "/frozen/legacy-receipt.json",
            "receipt_seal_sha256": recovery.LEGACY_DEPLOYMENT_RECEIPT_SEAL_SHA256,
            "deployment_seal_sha256": recovery.LEGACY_DEPLOYMENT_SEAL_SHA256,
            "source_sha256": source_sha,
        },
    )
    task._data.parent = deployment.PARENT_TASK_ID
    task._data.output.destination = deployment.FILES_SERVER_URI
    task._data.tags = sorted(deployment.TAGS)
    task._data.script.entry_point = deployment.ENTRY_POINT
    task._data.script.diff = source
    task._data.script.binary = "python"
    task._data.script.requirements = {
        "org_pip": "clearml==2.1.11",
        "pip": ["attrs==26.1.0", "clearml==2.1.11"],
    }
    task._docker = (
        f"{deployment.SERVICE_DOCKER_IMAGE} {deployment.SERVICE_DOCKER_ARGS}"
    )
    task._parameters = dict(recovery.LEGACY_PARAMETERS)
    task.artifacts = {
        recovery.LEGACY_ARTIFACT_NAMES[0]: SimpleNamespace(
            hash=recovery.LEGACY_MANIFEST_SHA256,
            size=recovery.LEGACY_MANIFEST_BYTES,
            timestamp="2026-08-12T05:00:19Z",
            type="dict",
            mode="output",
        )
    }
    task.get_models = MethodType(
        lambda _self: {"input": [], "output": []}, task
    )

    def reset(self: deploy_fakes._FakeTask, *, force: bool) -> None:
        assert force is True
        self.owner.events.append(("reset", self.id))
        if reset_mode == "never_accept":
            return
        self.status = "created"
        self.artifacts = {}
        if reset_mode == "raise_after":
            raise RuntimeError("callback failed after server acceptance")

    task.reset = MethodType(reset, task)
    task_class.events.clear()
    return task_class, task


@pytest.fixture(scope="module")
def prepared() -> dict[str, object]:
    return deployment._prepare()


def _assert_sealed(value: dict[str, object]) -> None:
    observed = str(value["seal_sha256"])
    unsealed = dict(value)
    unsealed.pop("seal_sha256")
    assert recovery._content_sha256(unsealed) == observed


def test_attempt_freezes_exact_stopped_singleton_before_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: dict[str, object],
) -> None:
    task_class, _ = _install_legacy(monkeypatch)
    monkeypatch.setattr(deployment, "_prepare", lambda: prepared)
    attempt, observed_prepared = recovery.build_attempt_receipt(
        task_class,
        evaluation_recovery_binding=_evaluation_binding(tmp_path),
    )
    assert observed_prepared is prepared
    assert attempt["status"] == "frozen_before_reset"
    assert attempt["pre_reset_authority"]["status"] == "stopped"
    assert attempt["pre_reset_authority"]["manifest_sha256"] == (
        recovery.LEGACY_MANIFEST_SHA256
    )
    assert attempt["target_deployment"]["source_sha256"] == (
        prepared["source_record"]["sha256"]
    )
    assert not [event for event in task_class.events if event[0] == "reset"]
    _assert_sealed(attempt)


def test_execute_resets_same_id_and_leaves_current_target_created_unqueued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: dict[str, object],
) -> None:
    task_class, task = _install_legacy(monkeypatch)
    monkeypatch.setattr(deployment, "_prepare", lambda: prepared)
    attempt, _ = recovery.build_attempt_receipt(
        task_class,
        evaluation_recovery_binding=_evaluation_binding(tmp_path),
    )
    receipt = recovery.execute_recovery(
        task_class,
        attempt_receipt=attempt,
        prepared=prepared,
        attempt_receipt_path=tmp_path / "attempt.json",
        sleeper=lambda _: None,
    )
    assert receipt["status"] == "recovered"
    assert receipt["controller_task_id"] == recovery.CONTROLLER_TASK_ID
    assert receipt["created_unqueued"] is True
    assert receipt["replacement_controller_created"] is False
    assert receipt["all_controller_models_absent"] is True
    assert task.status == "created"
    assert task.artifacts == {}
    assert task._data.script.diff == prepared["source"]
    assert task._parameters == {
        key: str(value) for key, value in prepared["parameters"].items()
    }
    assert [event for event in task_class.events if event[0] == "reset"] == [
        ("reset", recovery.CONTROLLER_TASK_ID)
    ]
    assert not [event for event in task_class.events if event[0] == "create"]
    _assert_sealed(receipt)


def test_reset_callback_exception_is_accepted_only_by_authoritative_created_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: dict[str, object],
) -> None:
    task_class, _ = _install_legacy(monkeypatch, reset_mode="raise_after")
    monkeypatch.setattr(deployment, "_prepare", lambda: prepared)
    attempt, _ = recovery.build_attempt_receipt(
        task_class,
        evaluation_recovery_binding=_evaluation_binding(tmp_path),
    )
    receipt = recovery.execute_recovery(
        task_class,
        attempt_receipt=attempt,
        prepared=prepared,
        attempt_receipt_path=tmp_path / "attempt.json",
        sleeper=lambda _: None,
    )
    assert receipt["created_unqueued"] is True


def test_reset_without_authoritative_acceptance_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: dict[str, object],
) -> None:
    task_class, _ = _install_legacy(monkeypatch, reset_mode="never_accept")
    monkeypatch.setattr(deployment, "_prepare", lambda: prepared)
    attempt, _ = recovery.build_attempt_receipt(
        task_class,
        evaluation_recovery_binding=_evaluation_binding(tmp_path),
    )
    with pytest.raises(recovery.ControllerRecoveryError, match="did not reach created"):
        recovery.execute_recovery(
            task_class,
            attempt_receipt=attempt,
            prepared=prepared,
            attempt_receipt_path=tmp_path / "attempt.json",
            sleeper=lambda _: None,
        )


def test_pre_reset_source_drift_fails_before_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, task = _install_legacy(monkeypatch)
    task._data.script.diff += "# drift\n"
    with pytest.raises(recovery.ControllerRecoveryError, match="legacy source drifted"):
        recovery.build_attempt_receipt(
            task_class,
            evaluation_recovery_binding=_evaluation_binding(tmp_path),
        )
    assert not [event for event in task_class.events if event[0] == "reset"]


def test_manifest_drift_fails_before_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, task = _install_legacy(monkeypatch)
    task.artifacts[recovery.LEGACY_ARTIFACT_NAMES[0]].hash = "0" * 64
    with pytest.raises(recovery.ControllerRecoveryError, match="legacy manifest drifted"):
        recovery.build_attempt_receipt(
            task_class,
            evaluation_recovery_binding=_evaluation_binding(tmp_path),
        )
    assert not [event for event in task_class.events if event[0] == "reset"]


def test_wrong_execute_token_fails_before_clearml_import(tmp_path: Path) -> None:
    with pytest.raises(recovery.ControllerRecoveryError, match="exact execute token"):
        recovery.main(
            [
                "--execute",
                "--execute-token",
                "wrong",
                "--evaluation-recovery-receipt",
                str(tmp_path / "missing.json"),
                "--attempt-receipt",
                str(tmp_path / "attempt.json"),
                "--receipt",
                str(tmp_path / "recovery.json"),
            ]
        )


def test_write_new_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    recovery._write_new(path, {"value": 1})
    with pytest.raises(FileExistsError):
        recovery._write_new(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_success_receipt_validator_binds_attempt_target_and_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: dict[str, object],
) -> None:
    task_class, _ = _install_legacy(monkeypatch)
    monkeypatch.setattr(deployment, "_prepare", lambda: prepared)
    attempt, _ = recovery.build_attempt_receipt(
        task_class,
        evaluation_recovery_binding=_evaluation_binding(tmp_path),
    )
    attempt_path = tmp_path / "attempt.json"
    recovery._write_new(attempt_path, attempt)
    receipt = recovery.execute_recovery(
        task_class,
        attempt_receipt=attempt,
        prepared=prepared,
        attempt_receipt_path=attempt_path,
        sleeper=lambda _: None,
    )
    receipt_path = tmp_path / "recovery.json"
    recovery._write_new(receipt_path, receipt)
    binding = recovery.validate_controller_recovery_receipt(receipt_path)
    assert binding["controller_task_id"] == recovery.CONTROLLER_TASK_ID
    assert binding["created_unqueued"] is True
    assert binding["evaluation_recovery_receipt_seal_sha256"] == "1" * 64
    assert binding["target_source_sha256"] == prepared["source_record"]["sha256"]

    tampered = dict(receipt)
    tampered["replacement_controller_created"] = True
    tampered = recovery._seal(tampered)
    tampered_path = tmp_path / "tampered.json"
    recovery._write_new(tampered_path, tampered)
    with pytest.raises(
        recovery.ControllerRecoveryError,
        match="replacement_controller_created drifted",
    ):
        recovery.validate_controller_recovery_receipt(tampered_path)
