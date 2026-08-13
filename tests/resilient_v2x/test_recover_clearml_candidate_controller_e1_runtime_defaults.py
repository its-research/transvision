from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.resilient_v2x import (
    recover_clearml_candidate_controller_e1_runtime_defaults as recovery,
)


class FailureTask:
    status = "failed"
    data = SimpleNamespace(status_reason="worker execution exit code 1")

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def get_reported_console_output(self, _count: int) -> list[str]:
        return list(self.lines)


def test_failure_evidence_accepts_only_exact_e1_runtime_default_failure() -> None:
    evidence = recovery._failure_evidence(
        FailureTask(["Traceback", f"RuntimeError: {recovery.FAILURE_MESSAGE}"])
    )
    assert evidence["runtime_error"] == recovery.FAILURE_MESSAGE
    with pytest.raises(
        recovery.CandidateE1RuntimeDefaultsRecoveryError,
        match="failure reason drifted",
    ):
        recovery._failure_evidence(
            FailureTask(
                [
                    f"RuntimeError: {recovery.FAILURE_MESSAGE}",
                    "RuntimeError: unknown failure",
                ]
            )
        )


def test_target_material_is_fixed_to_repaired_source_and_parameters() -> None:
    prepared = recovery.deploy._prepare()
    material = recovery._target_material(prepared)
    assert material["source_sha256"] == recovery.EXPECTED_NEW_SOURCE_SHA256
    assert (
        material["parameters_sha256"]
        == recovery.EXPECTED_NEW_PARAMETERS_SHA256
    )


class CreatedShell:
    status = "created"
    name = recovery.deploy.CONTROLLER_NAME
    data = SimpleNamespace(parent=recovery.deploy.PARENT_TASK_ID)


def test_created_shell_rejects_residual_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recovery.shared, "_status", lambda _task: "created")
    monkeypatch.setattr(
        recovery.shared, "_queue_identity", lambda _task, context: ("", "")
    )
    monkeypatch.setattr(
        recovery.shared,
        "_model_inventory",
        lambda _task, context: {"input": [], "output": []},
    )
    monkeypatch.setattr(
        recovery.shared.successor_deploy,
        "_task_parent",
        lambda _task: recovery.deploy.PARENT_TASK_ID,
    )
    monkeypatch.setattr(
        recovery.shared,
        "_artifact_inventory",
        lambda _task: [{"name": "stale"}],
    )
    with pytest.raises(
        recovery.CandidateE1RuntimeDefaultsRecoveryError,
        match="exact empty created shell",
    ):
        recovery._validate_created_shell(CreatedShell())

