from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from transvision.models.common_teacher_initialization import (
    COMMON_TEACHER_PREFIXES,
    EMPTY_TENSOR_MAPPING_SHA256,
    CommonTeacherInitializationError,
)


ROOT = Path(__file__).resolve().parents[2]
HOOK_SOURCE = ROOT / "transvision/models/hooks/common_teacher_initialization.py"


def _load_hook_class(monkeypatch: pytest.MonkeyPatch) -> type:
    class Registry:
        @staticmethod
        def register_module():
            return lambda implementation: implementation

    class CheckpointLoader:
        @staticmethod
        def load_checkpoint(filename, map_location=None, logger=None):
            del logger
            return torch.load(filename, map_location=map_location, weights_only=True)

    mmdet3d = ModuleType("mmdet3d")
    mmdet3d.__path__ = []  # type: ignore[attr-defined]
    registry = ModuleType("mmdet3d.registry")
    registry.HOOKS = Registry()
    mmengine = ModuleType("mmengine")
    mmengine.__path__ = []  # type: ignore[attr-defined]
    hooks = ModuleType("mmengine.hooks")
    hooks.Hook = object
    model = ModuleType("mmengine.model")
    model.is_model_wrapper = lambda candidate: False
    runner = ModuleType("mmengine.runner")
    runner.__path__ = []  # type: ignore[attr-defined]
    checkpoint = ModuleType("mmengine.runner.checkpoint")
    checkpoint.CheckpointLoader = CheckpointLoader
    for name, module in (
        ("mmdet3d", mmdet3d),
        ("mmdet3d.registry", registry),
        ("mmengine", mmengine),
        ("mmengine.hooks", hooks),
        ("mmengine.model", model),
        ("mmengine.runner", runner),
        ("mmengine.runner.checkpoint", checkpoint),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "_synthetic_common_teacher_initialization_hook",
        HOOK_SOURCE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CommonTeacherInitializationHook


class _CleanTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lidar_encoder = nn.Linear(1, 1, bias=False)
        self.camera_encoder = nn.Linear(1, 1, bias=False)
        self.bbox_head = nn.Linear(1, 1, bias=False)
        self.detection_projection = nn.Linear(1, 1, bias=False)
        self.resilient_fusion = nn.Linear(1, 1, bias=False)


class _Frozen(nn.Module):
    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher


class _Student(_CleanTeacher):
    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.teacher = _Frozen(teacher)


class _ZeroStateStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lidar_encoder = nn.Linear(1, 1, bias=False)
        self.camera_encoder = nn.Linear(1, 1, bias=False)
        self.bbox_head = nn.Linear(1, 1, bias=False)
        self.detection_projection = nn.Linear(1, 1, bias=False)
        self.fusion = nn.Identity()


def _fixture(tmp_path: Path) -> tuple[_CleanTeacher, _Student, Path, str]:
    teacher = _CleanTeacher()
    for index, parameter in enumerate(teacher.parameters(), start=1):
        parameter.data.fill_(float(index))
    student = _Student(copy.deepcopy(teacher))
    for name, parameter in student.named_parameters():
        if name.startswith(COMMON_TEACHER_PREFIXES):
            parameter.data.zero_()
    student.resilient_fusion.weight.data.fill_(-17.0)
    checkpoint = tmp_path / "teacher.pth"
    torch.save({"state_dict": teacher.state_dict()}, checkpoint)
    sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return teacher, student, checkpoint, sha256


def _runner(student: nn.Module, work_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        _load_from=None,
        _resume=False,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        model=student,
        work_dir=str(work_dir),
    )


def _hook(hook_class: type, checkpoint: Path, sha256: str):
    return hook_class(
        str(checkpoint),
        sha256,
        expected_teacher_keys=5,
        expected_common_keys=4,
        expected_teacher_fusion_keys=1,
    )


def test_hook_initializes_shared_only_and_publishes_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_class = _load_hook_class(monkeypatch)
    teacher, student, checkpoint, sha256 = _fixture(tmp_path)
    fusion_before = student.resilient_fusion.weight.detach().clone()
    runner = _runner(student, tmp_path / "work")

    _hook(hook_class, checkpoint, sha256).before_train(runner)

    for key, value in teacher.state_dict().items():
        if key.startswith(COMMON_TEACHER_PREFIXES):
            assert torch.equal(student.state_dict()[key], value)
    assert torch.equal(student.resilient_fusion.weight, fusion_before)
    audit = json.loads(
        (Path(runner.work_dir) / "common_teacher_initialization_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["result"] == "pass"
    assert audit["checkpoint"]["sha256"] == sha256
    assert audit["shared_initialization"]["keys"] == 4
    assert audit["method_specific_fusion"]["keys"] == 1
    assert audit["method_specific_fusion"]["numel"] == 1
    assert audit["method_specific_fusion"]["bytes"] > 0
    assert (
        audit["method_specific_fusion"]["sha256_before"] != EMPTY_TENSOR_MAPPING_SHA256
    )
    assert audit["method_specific_fusion"]["unchanged"] is True
    assert audit["target"]["nested_teacher_full_equality_verified"] is True


def test_hook_accepts_and_audits_zero_state_method_specific_fusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_class = _load_hook_class(monkeypatch)
    teacher, _, checkpoint, sha256 = _fixture(tmp_path)
    student = _ZeroStateStudent()
    for parameter in student.parameters():
        parameter.data.zero_()
    runner = _runner(student, tmp_path / "zero-state-work")

    _hook(hook_class, checkpoint, sha256).before_train(runner)

    for key, value in teacher.state_dict().items():
        if key.startswith(COMMON_TEACHER_PREFIXES):
            assert torch.equal(student.state_dict()[key], value)
    assert student.fusion.state_dict() == {}
    audit = json.loads(
        (Path(runner.work_dir) / "common_teacher_initialization_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["source"]["fusion_keys"] == 1
    assert audit["method_specific_fusion"] == {
        "keys": 0,
        "numel": 0,
        "bytes": 0,
        "sha256_before": EMPTY_TENSOR_MAPPING_SHA256,
        "sha256_after": EMPTY_TENSOR_MAPPING_SHA256,
        "unchanged": True,
    }
    assert audit["target"] == {
        "model_type": "_ZeroStateStudent",
        "target_key_count": 4,
        "target_common_key_count": 4,
        "target_fusion_key_count": 0,
        "nested_teacher_present": False,
        "nested_teacher_key_count": 0,
        "nested_teacher_full_equality_verified": False,
    }


def test_hook_fails_before_loading_on_sha_or_runner_checkpoint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_class = _load_hook_class(monkeypatch)
    _, student, checkpoint, sha256 = _fixture(tmp_path)
    common_before = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if name.startswith(COMMON_TEACHER_PREFIXES)
    }
    runner = _runner(student, tmp_path / "wrong-sha")
    with pytest.raises(CommonTeacherInitializationError, match="SHA-256 mismatch"):
        _hook(hook_class, checkpoint, "0" * 64).before_train(runner)
    for name, expected in common_before.items():
        assert torch.equal(student.state_dict()[name], expected)

    runner = _runner(student, tmp_path / "raw-load")
    runner._load_from = str(checkpoint)
    with pytest.raises(CommonTeacherInitializationError, match="load_from=None"):
        _hook(hook_class, checkpoint, sha256).before_train(runner)
