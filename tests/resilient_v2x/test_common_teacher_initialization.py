from __future__ import annotations

import copy
import hashlib
import json
import logging
import runpy
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from transvision.models.common_teacher_initialization import (
    COMMON_TEACHER_PREFIXES,
    EMPTY_TENSOR_MAPPING_SHA256,
    CommonTeacherInitializationError,
    assert_tensor_mapping_equal,
    normalize_checkpoint_state_dict,
    prepare_common_teacher_state,
    select_target_common_state,
    select_target_fusion_state,
    tensor_mapping_sha256,
    tensor_mapping_stats,
    validate_common_target,
)


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_TEACHER_KEYS = 5
SYNTHETIC_COMMON_KEYS = 4
SYNTHETIC_FUSION_KEYS = 1


def _synthetic_source(*, module_prefix: bool = False) -> dict[str, object]:
    state = OrderedDict(
        (
            ("lidar_encoder.block.weight", torch.tensor([[1.0, 2.0]])),
            ("camera_encoder.block.weight", torch.tensor([[3.0]])),
            ("bbox_head.cls.bias", torch.tensor([4.0])),
            ("detection_projection.norm.weight", torch.tensor([5.0])),
            ("resilient_fusion.router.weight", torch.tensor([[6.0]])),
        )
    )
    if module_prefix:
        state = OrderedDict((f"module.{key}", value) for key, value in state.items())
    return {"state_dict": state, "meta": {"epoch": 50}}


def _prepare(
    checkpoint: dict[str, object] | None = None,
) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    return prepare_common_teacher_state(
        _synthetic_source() if checkpoint is None else checkpoint,
        expected_teacher_keys=SYNTHETIC_TEACHER_KEYS,
        expected_common_keys=SYNTHETIC_COMMON_KEYS,
        expected_teacher_fusion_keys=SYNTHETIC_FUSION_KEYS,
    )


def _target_state(
    source: OrderedDict[str, torch.Tensor],
    *,
    nested_teacher: bool,
    baseline_fusion: bool = False,
    include_fusion: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    target = OrderedDict(
        (key, torch.zeros_like(value))
        for key, value in source.items()
        if key.startswith(COMMON_TEACHER_PREFIXES)
    )
    if include_fusion:
        fusion_key = (
            "fusion.method.weight"
            if baseline_fusion
            else "resilient_fusion.router.weight"
        )
        target[fusion_key] = torch.tensor([[-11.0]])
    if nested_teacher:
        target.update(
            (f"teacher.teacher.{key}", value.clone()) for key, value in source.items()
        )
    return target


def test_common_plan_normalizes_only_literal_module_prefix() -> None:
    source, common = _prepare(_synthetic_source(module_prefix=True))
    assert len(source) == SYNTHETIC_TEACHER_KEYS
    assert len(common) == SYNTHETIC_COMMON_KEYS
    assert set(common) == {
        "lidar_encoder.block.weight",
        "camera_encoder.block.weight",
        "bbox_head.cls.bias",
        "detection_projection.norm.weight",
    }
    assert not any(key.startswith("module.") for key in source)
    assert not any(key.startswith("resilient_fusion.") for key in common)


def test_checkpoint_normalization_rejects_colliding_keys() -> None:
    checkpoint = {
        "state_dict": OrderedDict(
            (
                ("lidar_encoder.weight", torch.ones(1)),
                ("module.lidar_encoder.weight", torch.zeros(1)),
            )
        )
    }
    with pytest.raises(CommonTeacherInitializationError, match="collision"):
        normalize_checkpoint_state_dict(checkpoint)


def test_shared_only_load_preserves_outer_fusion_and_verifies_nested_teacher() -> None:
    source, common = _prepare()
    target = _target_state(source, nested_teacher=True)
    contract = validate_common_target(source, common, target)
    assert contract["nested_teacher_present"] is True
    assert contract["nested_teacher_full_equality_verified"] is True
    assert contract["target_common_key_count"] == SYNTHETIC_COMMON_KEYS

    fusion_before = tensor_mapping_sha256(select_target_fusion_state(target))
    for key, value in common.items():
        target[key].copy_(value)
    assert_tensor_mapping_equal(
        common,
        select_target_common_state(target),
        context="synthetic shared load",
    )
    assert tensor_mapping_sha256(select_target_fusion_state(target)) == fusion_before


def test_controlled_baseline_target_accepts_only_common_plus_own_fusion() -> None:
    source, common = _prepare()
    target = _target_state(
        source,
        nested_teacher=False,
        baseline_fusion=True,
    )
    contract = validate_common_target(source, common, target)
    assert contract["nested_teacher_present"] is False
    assert contract["target_fusion_key_count"] == 1


def test_zero_state_fusion_has_canonical_empty_contract_and_remains_protected() -> None:
    source, common = _prepare()
    target = _target_state(
        source,
        nested_teacher=False,
        include_fusion=False,
    )

    contract = validate_common_target(source, common, target)
    fusion_before = select_target_fusion_state(target)
    assert contract == {
        "target_key_count": SYNTHETIC_COMMON_KEYS,
        "target_common_key_count": SYNTHETIC_COMMON_KEYS,
        "target_fusion_key_count": 0,
        "nested_teacher_present": False,
        "nested_teacher_key_count": 0,
        "nested_teacher_full_equality_verified": False,
    }
    assert tensor_mapping_stats(fusion_before) == {"keys": 0, "numel": 0, "bytes": 0}
    assert tensor_mapping_sha256(fusion_before) == EMPTY_TENSOR_MAPPING_SHA256
    assert EMPTY_TENSOR_MAPPING_SHA256 == hashlib.sha256(b"").hexdigest()

    for key, value in common.items():
        target[key].copy_(value)
    assert tensor_mapping_sha256(select_target_fusion_state(target)) == (
        EMPTY_TENSOR_MAPPING_SHA256
    )


def test_ego_only_config_and_real_fusion_model_accept_zero_state() -> None:
    from transvision.models.resilient_v2x.baselines import (
        EgoOnlyFusion,
        build_controlled_baseline_fusion,
    )

    config = runpy.run_path(str(ROOT / "configs/resilient_v2x/baselines/ego_only.py"))
    model_override = config["model"]
    assert isinstance(model_override, dict)
    assert model_override["baseline_name"] == "ego_only"
    assert model_override["enabled_agents"] == ("ego",)
    baseline_cfg = dict(model_override["baseline_cfg"])
    assert baseline_cfg.pop("_delete_") is True

    class EgoOnlyTarget(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lidar_encoder = nn.Linear(1, 1, bias=False)
            self.camera_encoder = nn.Linear(1, 1, bias=False)
            self.bbox_head = nn.Linear(1, 1, bias=False)
            self.detection_projection = nn.Linear(1, 1, bias=False)
            self.fusion = build_controlled_baseline_fusion(
                model_override["baseline_name"],
                channels=1,
                **baseline_cfg,
            )

    model = EgoOnlyTarget()
    assert isinstance(model.fusion, EgoOnlyFusion)
    assert model.fusion.state_dict() == OrderedDict()

    target = model.state_dict()
    target_common = select_target_common_state(target)
    target_fusion = select_target_fusion_state(target)
    teacher_state = OrderedDict(
        (key, value.detach().clone()) for key, value in target_common.items()
    )
    teacher_state["resilient_fusion.router.weight"] = torch.ones(1)
    source, common = prepare_common_teacher_state(
        {"state_dict": teacher_state},
        expected_teacher_keys=len(target_common) + 1,
        expected_common_keys=len(target_common),
        expected_teacher_fusion_keys=1,
    )

    contract = validate_common_target(source, common, target)
    assert contract["target_fusion_key_count"] == 0
    assert contract["target_key_count"] == len(target_common)
    assert contract["nested_teacher_present"] is False
    assert tensor_mapping_stats(target_fusion) == {
        "keys": 0,
        "numel": 0,
        "bytes": 0,
    }
    assert tensor_mapping_sha256(target_fusion) == EMPTY_TENSOR_MAPPING_SHA256

    incompatible = model.load_state_dict(common, strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys == []
    assert (
        tensor_mapping_sha256(select_target_fusion_state(model.state_dict()))
        == EMPTY_TENSOR_MAPPING_SHA256
    )


@pytest.mark.parametrize("failure", ("missing", "shape", "dtype"))
def test_common_target_fails_closed_on_shared_tensor_drift(failure: str) -> None:
    source, common = _prepare()
    target = _target_state(source, nested_teacher=False)
    key = "camera_encoder.block.weight"
    if failure == "missing":
        target.pop(key)
        match = "shared-key mismatch"
    elif failure == "shape":
        target[key] = torch.zeros(2)
        match = "shape mismatch"
    else:
        target[key] = target[key].double()
        match = "dtype mismatch"
    with pytest.raises(CommonTeacherInitializationError, match=match):
        validate_common_target(source, common, target)


def test_source_fails_closed_on_unknown_prefix_and_non_finite_tensor() -> None:
    unknown = _synthetic_source()
    unknown["state_dict"]["optimizer.shadow"] = torch.ones(1)  # type: ignore[index]
    with pytest.raises(CommonTeacherInitializationError, match="unsupported prefixes"):
        prepare_common_teacher_state(
            unknown,
            expected_teacher_keys=6,
            expected_common_keys=4,
            expected_teacher_fusion_keys=1,
        )

    non_finite = _synthetic_source()
    non_finite["state_dict"][  # type: ignore[index]
        "lidar_encoder.block.weight"
    ][0, 0] = float("nan")
    with pytest.raises(CommonTeacherInitializationError, match="non-finite"):
        _prepare(non_finite)


def test_nested_teacher_must_equal_all_source_tensors() -> None:
    source, common = _prepare()
    target = _target_state(source, nested_teacher=True)
    target["teacher.teacher.resilient_fusion.router.weight"].zero_()
    with pytest.raises(CommonTeacherInitializationError, match="value mismatch"):
        validate_common_target(source, common, target)


def test_hook_loads_only_shared_tensors_and_writes_audit(tmp_path: Path) -> None:
    pytest.importorskip("mmengine")
    pytest.importorskip("mmdet3d")
    from transvision.models.hooks.common_teacher_initialization import (
        CommonTeacherInitializationHook,
    )

    class CleanTeacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lidar_encoder = nn.Linear(1, 1, bias=False)
            self.camera_encoder = nn.Linear(1, 1, bias=False)
            self.bbox_head = nn.Linear(1, 1, bias=False)
            self.detection_projection = nn.Linear(1, 1, bias=False)
            self.resilient_fusion = nn.Linear(1, 1, bias=False)

    class Frozen(nn.Module):
        def __init__(self, teacher: nn.Module) -> None:
            super().__init__()
            self.teacher = teacher

    class Student(CleanTeacher):
        def __init__(self, teacher: nn.Module) -> None:
            super().__init__()
            self.teacher = Frozen(teacher)

    teacher = CleanTeacher()
    for index, parameter in enumerate(teacher.parameters(), start=1):
        parameter.data.fill_(float(index))
    student = Student(copy.deepcopy(teacher))
    for name, parameter in student.named_parameters():
        if name.startswith(COMMON_TEACHER_PREFIXES):
            parameter.data.zero_()
    student.resilient_fusion.weight.data.fill_(-13.0)
    fusion_before = student.resilient_fusion.weight.detach().clone()

    checkpoint = tmp_path / "teacher.pth"
    torch.save({"state_dict": teacher.state_dict()}, checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    logger = logging.getLogger("common-teacher-initialization-test")
    runner = SimpleNamespace(
        _load_from=None,
        _resume=False,
        logger=logger,
        model=student,
        work_dir=str(tmp_path / "work"),
    )
    hook = CommonTeacherInitializationHook(
        str(checkpoint),
        checkpoint_sha256,
        expected_teacher_keys=5,
        expected_common_keys=4,
        expected_teacher_fusion_keys=1,
    )
    hook.before_train(runner)

    for key, value in teacher.state_dict().items():
        if key.startswith(COMMON_TEACHER_PREFIXES):
            assert torch.equal(student.state_dict()[key], value)
    assert torch.equal(student.resilient_fusion.weight, fusion_before)
    audit_path = Path(runner.work_dir) / "common_teacher_initialization_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["result"] == "pass"
    assert audit["checkpoint"]["sha256"] == checkpoint_sha256
    assert audit["shared_initialization"]["keys"] == 4
    assert audit["method_specific_fusion"]["unchanged"] is True
    assert audit["target"]["nested_teacher_full_equality_verified"] is True
