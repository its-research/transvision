from __future__ import annotations

import ast
import inspect
import json
import runpy
from pathlib import Path

import pytest
import torch

from transvision.models.resilient_v2x import (
    BEVGridSpec,
    DynamicExpertRouter,
    ResilientV2XFeatureFusion,
)

from .test_configs import CONFIG_ROOT, MAIN, _load_config
from .test_feature_fusion import _all_supported
from .test_feature_fusion import _inputs as feature_fusion_inputs
from .test_routing import single_router_inputs


P0 = CONFIG_ROOT / "improvements" / "support_residual_no_reliability_linear.py"
P1 = CONFIG_ROOT / "improvements" / "reliability_gated_residual.py"
P1_IDENTITY = "dair_improvement_reliability_gated_residual"


def _router(
    *,
    weight: float = 0.0,
    reliability_gate: bool = False,
) -> DynamicExpertRouter:
    return DynamicExpertRouter(
        channels=256,
        hidden_channels=256,
        support_residual_weight=weight,
        support_residual_reliability_gate=reliability_gate,
    ).eval()


def _inputs(
    support: tuple[bool, bool, bool, bool],
    reliability: tuple[float, float, float, float],
) -> dict[str, object]:
    inputs = single_router_inputs(support)
    inputs["branch_reliability"] = torch.tensor([reliability])
    inputs["routing_mode"] = "uniform"
    inputs["use_reliability"] = False
    return inputs


def _model(path: Path) -> dict[str, object]:
    model = _load_config(path)["model"]
    assert isinstance(model, dict)
    return model


def _fusion(model: dict[str, object]) -> ResilientV2XFeatureFusion:
    grid = model["grid_spec"]
    assert isinstance(grid, dict)
    return ResilientV2XFeatureFusion(
        BEVGridSpec(**grid),
        ptf_mode=model["ptf_mode"],
        routing_mode=model["routing_mode"],
        use_reliability=model["use_reliability"],
        use_delay_metadata=model["use_delay_metadata"],
        support_residual_weight=model.get("support_residual_weight", 0.0),
        support_residual_reliability_gate=model.get(
            "support_residual_reliability_gate",
            False,
        ),
        delta_t_ms=model["delta_t_ms"],
    )


def _assert_same_routing_output(left: object, right: object) -> None:
    for field in (
        "fused",
        "weights",
        "descriptor",
        "expert_support",
        "lidar_expert",
        "camera_expert",
        "synergy_expert",
        "overall_support",
    ):
        assert torch.equal(
            getattr(left, field),
            getattr(right, field),
        ), field


@pytest.mark.parametrize("value", [None, 0, 1, "true", object()])
def test_reliability_gate_requires_an_exact_boolean(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="support_residual_reliability_gate",
    ):
        DynamicExpertRouter(
            channels=256,
            hidden_channels=256,
            support_residual_reliability_gate=value,
        )


def test_reliability_gate_is_default_off_at_every_public_boundary() -> None:
    assert (
        inspect.signature(DynamicExpertRouter)
        .parameters["support_residual_reliability_gate"]
        .default
        is False
    )
    assert (
        inspect.signature(ResilientV2XFeatureFusion)
        .parameters["support_residual_reliability_gate"]
        .default
        is False
    )
    detector_path = (
        CONFIG_ROOT.parents[1]
        / "transvision"
        / "models"
        / "detectors"
        / "resilient_v2x.py"
    )
    tree = ast.parse(detector_path.read_text(), filename=str(detector_path))
    detector = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResilientV2XNet"
    )
    initializer = next(
        node
        for node in detector.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    names = [argument.arg for argument in initializer.args.args]
    defaults = [None] * (len(names) - len(initializer.args.defaults))
    defaults.extend(initializer.args.defaults)
    detector_default = dict(zip(names, defaults))["support_residual_reliability_gate"]
    assert isinstance(detector_default, ast.Constant)
    assert detector_default.value is False


def test_default_off_is_bitwise_identical_for_the_p0_path() -> None:
    implicit = _router(weight=0.5)
    explicit = _router(weight=0.5, reliability_gate=False)
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    inputs = _inputs(
        (True, True, True, True),
        (0.8, 0.4, 0.2, 0.1),
    )

    _assert_same_routing_output(implicit(**inputs), explicit(**inputs))


def test_clean_full_reliability_is_bitwise_p0_equivalent() -> None:
    p0 = _router(weight=0.5)
    p1 = _router(weight=0.5, reliability_gate=True)
    p1.load_state_dict(p0.state_dict(), strict=True)
    inputs = _inputs(
        (True, True, True, True),
        (1.0, 1.0, 1.0, 1.0),
    )

    _assert_same_routing_output(p0(**inputs), p1(**inputs))


def test_diagnostic_marks_only_the_enabled_p1_behavior() -> None:
    selections = _all_supported()
    (
        lidar,
        camera,
        transforms,
        lidar_available,
        camera_available,
    ) = feature_fusion_inputs(selections)
    legacy = ResilientV2XFeatureFusion(
        BEVGridSpec(0.0, -2.0, 1.0, 4, 4),
        ptf_mode="none",
        routing_mode="uniform",
        use_reliability=False,
        support_residual_weight=0.5,
    ).eval()
    candidate = ResilientV2XFeatureFusion(
        BEVGridSpec(0.0, -2.0, 1.0, 4, 4),
        ptf_mode="none",
        routing_mode="uniform",
        use_reliability=False,
        support_residual_weight=0.5,
        support_residual_reliability_gate=True,
    ).eval()
    candidate.load_state_dict(legacy.state_dict(), strict=True)

    legacy_output = legacy(
        lidar,
        camera,
        transforms,
        transforms,
        lidar_available,
        camera_available,
        selections,
    )
    candidate_output = candidate(
        lidar,
        camera,
        transforms,
        transforms,
        lidar_available,
        camera_available,
        selections,
    )

    assert legacy_output.diagnostics[0].method == (
        "ResilientV2X(ptf=none,routing=uniform,reliability=0,delay_metadata=1)"
    )
    assert candidate_output.diagnostics[0].method == (
        "ResilientV2X(ptf=none,routing=uniform,reliability=0,"
        "delay_metadata=1,support_residual_reliability_gate=1)"
    )


def test_single_modality_reduces_residual_coefficient_from_half_to_quarter() -> None:
    baseline = _router()
    candidate = _router(weight=0.5, reliability_gate=True)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    inputs = _inputs(
        (True, True, False, False),
        (1.0, 1.0, 0.0, 0.0),
    )

    baseline_output = baseline(**inputs)
    candidate_output = candidate(**inputs)
    lidar = inputs["lidar_feature"]
    assert isinstance(lidar, torch.Tensor)
    expected = baseline_output.fused + 0.25 * (lidar - baseline_output.fused)

    torch.testing.assert_close(candidate_output.fused, expected)
    torch.testing.assert_close(
        candidate_output.descriptor[:, 768:770],
        torch.tensor([[1.0, 0.0]]),
        atol=0.0,
        rtol=0.0,
    )


def test_zero_raw_reliability_turns_the_residual_off_and_remains_finite() -> None:
    baseline = _router()
    candidate = _router(weight=0.5, reliability_gate=True)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    inputs = _inputs(
        (True, True, True, True),
        (0.0, 0.0, 0.0, 0.0),
    )

    baseline_output = baseline(**inputs)
    candidate_output = candidate(**inputs)

    assert torch.equal(candidate_output.fused, baseline_output.fused)
    assert torch.isfinite(candidate_output.fused).all().item()
    torch.testing.assert_close(
        candidate_output.descriptor[:, 768:770],
        torch.ones(1, 2),
        atol=0.0,
        rtol=0.0,
    )


def test_residual_mean_and_coefficient_use_raw_branch_reliability() -> None:
    baseline = _router()
    candidate = _router(weight=0.5, reliability_gate=True)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    inputs = _inputs(
        (True, True, True, True),
        (0.8, 0.4, 0.2, 0.0),
    )

    baseline_output = baseline(**inputs)
    candidate_output = candidate(**inputs)
    lidar = inputs["lidar_feature"]
    camera = inputs["camera_feature"]
    assert isinstance(lidar, torch.Tensor)
    assert isinstance(camera, torch.Tensor)
    # Raw modality reliability is [0.6, 0.1], so its fixed-slot mean gate is
    # 0.35 and the configured 0.5 residual coefficient becomes 0.175.
    raw_weighted_mean = (0.6 * lidar + 0.1 * camera) / 0.7
    expected = baseline_output.fused + 0.175 * (
        raw_weighted_mean - baseline_output.fused
    )

    torch.testing.assert_close(candidate_output.fused, expected)
    assert torch.equal(candidate_output.weights, baseline_output.weights)
    torch.testing.assert_close(
        candidate_output.descriptor[:, 768:770],
        torch.ones(1, 2),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.isfinite(candidate_output.fused).all().item()


def test_p1_source_is_a_minimal_overlay_of_the_immutable_p0_config() -> None:
    source = runpy.run_path(str(P1))

    assert source["_base_"] == ["./support_residual_no_reliability_linear.py"]
    assert source["model"] == {"support_residual_reliability_gate": True}
    assert source["experiment"]["name"] == P1_IDENTITY
    assert {key for key in source if not key.startswith("_")} == {
        "experiment",
        "implementation_choices_runtime",
        "model",
    }


def test_p1_changes_only_the_new_behavior_switch_from_p0() -> None:
    p0 = _load_config(P0)
    p1 = _load_config(P1)
    p0_model = p0["model"]
    p1_model = p1["model"]
    assert isinstance(p0_model, dict)
    assert isinstance(p1_model, dict)

    assert "support_residual_reliability_gate" not in p0_model
    assert {
        key
        for key in p0_model.keys() | p1_model.keys()
        if p0_model.get(key) != p1_model.get(key)
    } == {"support_residual_reliability_gate"}
    assert p1_model["ptf_mode"] == "linear"
    assert p1_model["use_reliability"] is False
    assert p1_model["support_residual_weight"] == 0.5
    assert p1_model["support_residual_reliability_gate"] is True


def test_p1_records_the_exact_raw_reliability_formula() -> None:
    choices = _load_config(P1)["implementation_choices_runtime"]
    assert choices["support_residual_reliability_gate"] is True
    assert choices["support_residual_reliability_source"] == (
        "raw branch reliability, independent of router descriptor inputs"
    )
    assert choices["support_residual_reliability_gate_formula"] == (
        "clamp(mean(s_lidar*(r_lidar_ego+r_lidar_rsu)/2, "
        "s_camera*(r_camera_ego+r_camera_rsu)/2),0,1)"
    )
    assert choices["support_residual_formula"] == (
        "router_fused + support_residual_weight * reliability_gate * "
        "(raw_reliability_weighted_mean - router_fused)"
    )


def test_p1_behavior_contract_is_json_serializable_without_loss() -> None:
    source = runpy.run_path(str(P1))
    payload = {
        "model": source["model"],
        "implementation_choices_runtime": source["implementation_choices_runtime"],
        "experiment": source["experiment"],
    }

    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_p1_preserves_teacher_distillation_ddp_seed_and_50e_val10() -> None:
    main = _load_config(MAIN)
    p0 = _load_config(P0)
    p1 = _load_config(P1)

    for key in (
        "custom_hooks",
        "find_unused_parameters",
        "load_from",
        "optim_wrapper",
        "param_scheduler",
        "paper_data_contract",
        "randomness",
        "train_cfg",
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
    ):
        assert p1[key] == p0[key] == main[key]

    main_model = main["model"]
    p1_model = p1["model"]
    assert isinstance(main_model, dict)
    assert isinstance(p1_model, dict)
    assert p1_model["teacher"] == main_model["teacher"]
    assert p1_model["teacher_checkpoint"] == main_model["teacher_checkpoint"]
    assert p1_model["distillation"] == main_model["distillation"]
    assert "support_residual_reliability_gate" not in p1_model["teacher"]
    assert p1["find_unused_parameters"] is True
    assert p1["train_cfg"] == {
        "type": "EpochBasedTrainLoop",
        "max_epochs": 50,
        "val_interval": 10,
    }
    assert p1["randomness"]["seed"] == 20250218
    assert p1["train_dataloader"]["dataset"]["seed"] == 20250218
    assert p1["train_dataloader"]["dataset"]["include_clean_teacher"] is True


def test_p1_keeps_the_exact_seeded_state_and_parameter_schema() -> None:
    p0_model = _model(P0)
    p1_model = _model(P1)

    torch.manual_seed(20250218)
    p0_fusion = _fusion(p0_model)
    torch.manual_seed(20250218)
    p1_fusion = _fusion(p1_model)

    p0_state = p0_fusion.state_dict()
    p1_state = p1_fusion.state_dict()
    assert tuple(p1_state) == tuple(p0_state)
    assert {
        name: (tuple(value.shape), value.dtype) for name, value in p1_state.items()
    } == {name: (tuple(value.shape), value.dtype) for name, value in p0_state.items()}
    assert tuple(
        (name, parameter.requires_grad)
        for name, parameter in p1_fusion.named_parameters()
    ) == tuple(
        (name, parameter.requires_grad)
        for name, parameter in p0_fusion.named_parameters()
    )
    assert sum(parameter.numel() for parameter in p1_fusion.parameters()) == sum(
        parameter.numel() for parameter in p0_fusion.parameters()
    )
    for name, value in p0_state.items():
        assert torch.equal(p1_state[name], value)
    assert p0_fusion.router.support_residual_reliability_gate is False
    assert p1_fusion.router.support_residual_reliability_gate is True
