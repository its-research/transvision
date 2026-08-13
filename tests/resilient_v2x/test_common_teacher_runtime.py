from __future__ import annotations

import runpy
from pathlib import Path

import pytest


RUNTIME = (
    Path(__file__).resolve().parents[2]
    / "configs/resilient_v2x/_base_/runtime.py"
)


def test_common_initialization_hook_is_disabled_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESILIENT_V2X_COMMON_INIT_CHECKPOINT", raising=False)
    monkeypatch.delenv("RESILIENT_V2X_COMMON_INIT_SHA256", raising=False)
    runtime = runpy.run_path(str(RUNTIME))
    assert runtime["load_from"] is None
    assert runtime["resume"] is False
    assert runtime["custom_hooks"] == []


@pytest.mark.parametrize("missing", ("checkpoint", "sha256"))
def test_common_initialization_environment_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    checkpoint = "/sealed/checkpoints/clean_teacher_best.pth"
    sha256 = "b" * 64
    if missing == "checkpoint":
        monkeypatch.delenv("RESILIENT_V2X_COMMON_INIT_CHECKPOINT", raising=False)
        monkeypatch.setenv("RESILIENT_V2X_COMMON_INIT_SHA256", sha256)
    else:
        monkeypatch.setenv("RESILIENT_V2X_COMMON_INIT_CHECKPOINT", checkpoint)
        monkeypatch.delenv("RESILIENT_V2X_COMMON_INIT_SHA256", raising=False)
    with pytest.raises(RuntimeError, match="must be provided together"):
        runpy.run_path(str(RUNTIME))


def test_common_initialization_hook_binds_checkpoint_and_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = "/sealed/checkpoints/clean_teacher_best.pth"
    sha256 = "c" * 64
    monkeypatch.setenv("RESILIENT_V2X_COMMON_INIT_CHECKPOINT", checkpoint)
    monkeypatch.setenv("RESILIENT_V2X_COMMON_INIT_SHA256", sha256)
    runtime = runpy.run_path(str(RUNTIME))
    assert runtime["load_from"] is None
    assert runtime["custom_hooks"] == [
        {
            "type": "CommonTeacherInitializationHook",
            "checkpoint": checkpoint,
            "expected_sha256": sha256,
            "priority": "HIGHEST",
        }
    ]

