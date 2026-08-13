from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from transvision.models.resilient_v2x.causal_repair import (
    CausalBranchRepair,
)
from transvision.models.resilient_v2x.contracts import (
    Agent,
    BranchSelection,
    Modality,
    SourceCandidate,
)
from transvision.models.resilient_v2x.geometry import BEVGridSpec
from transvision.models.resilient_v2x.ptf import PTFOutput
from transvision.models.resilient_v2x.routing import (
    DepthwiseSeparableResidualBlock,
    DynamicExpertRouter,
    ModalityAggregate,
    ModalityAggregator,
    RoutingOutput,
)


ROOT = Path(__file__).resolve().parents[2]


class _PretendDynamicMode:
    def __eq__(self, other: object) -> bool:
        return other == "dynamic"


class _SpoofedRoutingMode(str):
    def __new__(cls) -> "_SpoofedRoutingMode":
        return super().__new__(cls, "adaptive")

    def __eq__(self, other: object) -> bool:
        return other in ("dynamic", "uniform")


def make_router(
    *,
    support_residual_weight: float = 0.0,
) -> DynamicExpertRouter:
    return DynamicExpertRouter(
        channels=256,
        hidden_channels=256,
        support_residual_weight=support_residual_weight,
    )


def router_inputs(
    *,
    batch: int = 2,
    size: int = 2,
    requires_grad: bool = False,
) -> dict[str, object]:
    if batch != 2:
        raise ValueError("the canonical fixture has batch two")
    lidar = torch.randn(batch, 256, size, size).requires_grad_(requires_grad)
    camera = torch.randn(batch, 256, size, size).requires_grad_(requires_grad)
    return {
        "lidar_feature": lidar,
        "camera_feature": camera,
        "lidar_branch_support": torch.tensor(
            [[True, False], [True, True]]
        ),
        "camera_branch_support": torch.tensor(
            [[True, True], [False, False]]
        ),
        "branch_reliability": torch.tensor(
            [[1.0, 0.0, 0.8, 0.7], [1.0, 0.9, 0.0, 0.0]]
        ),
        "branch_observed": torch.tensor(
            [[True, False, True, False], [True, False, False, False]]
        ),
        "branch_propagated": torch.tensor(
            [[False, False, False, True], [False, True, False, False]]
        ),
        "branch_age_intervals": torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 0.0]]
        ),
        "rsu_delay_intervals": torch.tensor([[1.0], [2.0]]),
        "routing_mode": "dynamic",
        "use_reliability": True,
        "use_delay_metadata": True,
    }


def single_router_inputs(
    branch_support: tuple[bool, bool, bool, bool],
    *,
    size: int = 2,
) -> dict[str, object]:
    support = torch.tensor([branch_support], dtype=torch.bool)
    observed = torch.zeros(1, 4, dtype=torch.bool)
    propagated = torch.zeros(1, 4, dtype=torch.bool)
    for index, is_supported in enumerate(branch_support):
        if is_supported:
            if index in (0, 2):
                observed[0, index] = True
            else:
                propagated[0, index] = True
    reliability_values = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    reliability = reliability_values * support
    age_values = torch.tensor([[0.0, 1.0, 0.0, 2.0]])
    age = age_values * support
    has_rsu = branch_support[1] or branch_support[3]
    return {
        "lidar_feature": torch.randn(1, 256, size, size),
        "camera_feature": torch.randn(1, 256, size, size),
        "lidar_branch_support": support[:, :2],
        "camera_branch_support": support[:, 2:],
        "branch_reliability": reliability,
        "branch_observed": observed,
        "branch_propagated": propagated,
        "branch_age_intervals": age,
        "rsu_delay_intervals": torch.tensor([[1.0 if has_rsu else 0.0]]),
        "routing_mode": "dynamic",
        "use_reliability": True,
        "use_delay_metadata": True,
    }


def aggregation_inputs(
    *,
    batch: int = 2,
    size: int = 2,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        "ego_feature": torch.randn(
            batch, 256, size, size, requires_grad=requires_grad
        ),
        "rsu_feature": torch.randn(
            batch, 256, size, size, requires_grad=requires_grad
        ),
        "ego_support": torch.tensor([True, False]),
        "rsu_support": torch.tensor([False, True]),
        "ego_reliability": torch.tensor([0.8, 0.0]),
        "rsu_reliability": torch.tensor([0.0, 0.6]),
    }


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def test_routing_module_exposes_task_six_types() -> None:
    assert DepthwiseSeparableResidualBlock.__name__ == (
        "DepthwiseSeparableResidualBlock"
    )
    assert ModalityAggregate.__name__ == "ModalityAggregate"
    assert ModalityAggregator.__name__ == "ModalityAggregator"
    assert RoutingOutput.__name__ == "RoutingOutput"
    assert DynamicExpertRouter.__name__ == "DynamicExpertRouter"


@pytest.mark.parametrize("channels", [32, 64, 256])
def test_residual_block_accepts_future_widths_divisible_by_32(
    channels: int,
) -> None:
    block = DepthwiseSeparableResidualBlock(channels)
    value = torch.randn(2, channels, 2, 3)

    output = block(value)

    assert output.shape == value.shape
    assert output.dtype == value.dtype
    assert output.device == value.device


@pytest.mark.parametrize("channels", [0, -32, 31, 33, True, 256.0, "256"])
def test_residual_block_rejects_bad_channels(channels: object) -> None:
    with pytest.raises(ValueError, match="channels"):
        DepthwiseSeparableResidualBlock(channels)


def test_residual_block_has_exact_graph_and_parameter_golden() -> None:
    block = DepthwiseSeparableResidualBlock(256)
    convolutions = [
        module for module in block.modules() if isinstance(module, nn.Conv2d)
    ]
    normalizations = [
        module for module in block.modules() if isinstance(module, nn.GroupNorm)
    ]

    assert trainable_parameter_count(block) == 68_864
    assert len(convolutions) == 2
    depthwise, pointwise = convolutions
    assert depthwise.in_channels == depthwise.out_channels == 256
    assert depthwise.kernel_size == (3, 3)
    assert depthwise.padding == (1, 1)
    assert depthwise.groups == 256
    assert depthwise.bias is None
    assert pointwise.in_channels == pointwise.out_channels == 256
    assert pointwise.kernel_size == (1, 1)
    assert pointwise.groups == 1
    assert pointwise.bias is None
    assert len(normalizations) == 2
    assert all(module.num_groups == 32 for module in normalizations)
    assert all(module.affine for module in normalizations)
    assert sum(
        isinstance(module, nn.SiLU) for module in block.modules()
    ) == 2


@pytest.mark.parametrize(
    "value",
    [
        object(),
        torch.zeros(256, 2, 2),
        torch.zeros(0, 256, 2, 2),
        torch.zeros(1, 255, 2, 2),
        torch.zeros(1, 256, 0, 2),
        torch.zeros(1, 256, 2, 2, dtype=torch.int64),
        torch.full((1, 256, 2, 2), math.nan),
        torch.full((1, 256, 2, 2), math.inf),
    ],
    ids=[
        "not-tensor",
        "rank",
        "empty-batch",
        "channels",
        "empty-spatial",
        "integer",
        "nan",
        "inf",
    ],
)
def test_residual_block_fails_closed_on_invalid_input(value: object) -> None:
    with pytest.raises(ValueError):
        DepthwiseSeparableResidualBlock(256)(value)


def test_modality_aggregator_has_exact_graph_and_parameter_golden() -> None:
    aggregator = ModalityAggregator(channels=256)
    convolutions = [
        module
        for module in aggregator.modules()
        if isinstance(module, nn.Conv2d)
    ]
    normalizations = [
        module
        for module in aggregator.modules()
        if isinstance(module, nn.GroupNorm)
    ]
    blocks = [
        module
        for module in aggregator.modules()
        if isinstance(module, DepthwiseSeparableResidualBlock)
    ]

    assert trainable_parameter_count(aggregator) == 200_448
    assert len(convolutions) == 3
    projection = next(
        module
        for module in convolutions
        if module.in_channels == 512
    )
    assert projection.out_channels == 256
    assert projection.kernel_size == (1, 1)
    assert projection.bias is None
    assert len(normalizations) == 3
    assert len(blocks) == 1


@pytest.mark.parametrize("channels", [32, 255, 512, True, 256.0])
def test_modality_aggregator_accepts_only_production_width(
    channels: object,
) -> None:
    with pytest.raises(ValueError, match="channels"):
        ModalityAggregator(channels)


def test_router_has_exact_independent_graph_and_parameter_golden() -> None:
    router = make_router()
    convolutions = [
        module for module in router.modules() if isinstance(module, nn.Conv2d)
    ]
    normalizations = [
        module for module in router.modules() if isinstance(module, nn.GroupNorm)
    ]
    layer_norms = [
        module for module in router.modules() if isinstance(module, nn.LayerNorm)
    ]
    linears = [
        module for module in router.modules() if isinstance(module, nn.Linear)
    ]
    blocks = [
        module
        for module in router.modules()
        if isinstance(module, DepthwiseSeparableResidualBlock)
    ]

    assert router.descriptor_dim == 783
    assert trainable_parameter_count(router) == 747_809
    assert len(convolutions) == 13
    assert all(module.bias is None for module in convolutions)
    assert len(normalizations) == 13
    assert all(module.num_groups == 32 for module in normalizations)
    assert len(layer_norms) == 1
    assert layer_norms[0].normalized_shape == (783,)
    assert len(linears) == 2
    assert linears[0].in_features == 783
    assert linears[0].out_features == 256
    assert linears[1].in_features == 256
    assert linears[1].out_features == 3
    assert all(module.bias is not None for module in linears)
    assert len(blocks) == 6
    assert len({id(module) for module in blocks}) == 6
    assert not any(
        isinstance(module, ModalityAggregator) for module in router.modules()
    )

    synergy_projection = next(
        module
        for module in convolutions
        if module.in_channels == 512
    )
    assert synergy_projection.out_channels == 256
    assert synergy_projection.kernel_size == (1, 1)
    assert synergy_projection.bias is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("channels", 255),
        ("channels", 512),
        ("channels", True),
        ("channels", 256.0),
        ("hidden_channels", 128),
        ("hidden_channels", 512),
        ("hidden_channels", False),
        ("hidden_channels", 256.0),
    ],
)
def test_router_constructor_accepts_only_production_widths(
    field: str,
    value: object,
) -> None:
    values = {"channels": 256, "hidden_channels": 256}
    values[field] = value

    with pytest.raises(ValueError, match=field):
        DynamicExpertRouter(**values)


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, math.nan, math.inf, -math.inf, True, "0.5", None],
)
def test_router_support_residual_weight_fails_closed(value: object) -> None:
    with pytest.raises(ValueError, match="support_residual_weight"):
        DynamicExpertRouter(
            channels=256,
            hidden_channels=256,
            support_residual_weight=value,
        )


def test_modality_aggregator_masks_inputs_and_uses_exact_reliability_average() -> None:
    aggregator = ModalityAggregator(channels=256).eval()
    inputs = aggregation_inputs()
    first = aggregator(**inputs)
    changed = dict(inputs)
    changed_rsu = inputs["rsu_feature"].clone()
    changed_rsu[0].fill_(10_000.0)
    changed_ego = inputs["ego_feature"].clone()
    changed_ego[1].fill_(-10_000.0)
    changed["rsu_feature"] = changed_rsu
    changed["ego_feature"] = changed_ego
    second = aggregator(**changed)

    assert isinstance(first, ModalityAggregate)
    assert first.feature.shape == (2, 256, 2, 2)
    assert first.support.tolist() == [True, True]
    torch.testing.assert_close(
        first.reliability,
        torch.tensor([0.4, 0.3]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        first.feature[0],
        second.feature[0],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        first.feature[1],
        second.feature[1],
        atol=0.0,
        rtol=0.0,
    )


def test_modality_aggregator_concatenates_ego_before_rsu() -> None:
    aggregator = ModalityAggregator(channels=256).eval()
    captured: list[torch.Tensor] = []
    handle = aggregator.projection[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        aggregator(
            ego_feature=torch.ones(1, 256, 2, 2),
            rsu_feature=torch.full((1, 256, 2, 2), 2.0),
            ego_support=torch.tensor([True]),
            rsu_support=torch.tensor([True]),
            ego_reliability=torch.tensor([0.8]),
            rsu_reliability=torch.tensor([0.6]),
        )
    finally:
        handle.remove()

    assert len(captured) == 1
    torch.testing.assert_close(
        captured[0][:, :256],
        torch.ones_like(captured[0][:, :256]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        captured[0][:, 256:],
        torch.full_like(captured[0][:, 256:], 2.0),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("ego_support", "rsu_support"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_modality_aggregator_support_is_exact_or(
    ego_support: bool,
    rsu_support: bool,
) -> None:
    aggregator = ModalityAggregator(channels=256).eval()
    inputs = {
        "ego_feature": torch.randn(1, 256, 2, 2),
        "rsu_feature": torch.randn(1, 256, 2, 2),
        "ego_support": torch.tensor([ego_support]),
        "rsu_support": torch.tensor([rsu_support]),
        "ego_reliability": torch.tensor([0.8 if ego_support else 0.0]),
        "rsu_reliability": torch.tensor([0.6 if rsu_support else 0.0]),
    }

    output = aggregator(**inputs)

    assert output.support.tolist() == [ego_support or rsu_support]
    assert output.reliability.item() == pytest.approx(
        ((0.8 if ego_support else 0.0) + (0.6 if rsu_support else 0.0))
        / 2.0
    )
    if not ego_support and not rsu_support:
        assert torch.count_nonzero(output.feature).item() == 0


def test_aggregator_post_layer_mask_blocks_mutated_affine_leakage() -> None:
    aggregator = ModalityAggregator(channels=256).eval()
    captured: list[torch.Tensor] = []
    with torch.no_grad():
        for module in aggregator.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.zero_()
            if isinstance(module, nn.GroupNorm):
                module.weight.zero_()
                module.bias.fill_(2.0)

    handle = aggregator.residual_block.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().clone())
    )
    try:
        output = aggregator(
            ego_feature=torch.zeros(1, 256, 2, 2),
            rsu_feature=torch.zeros(1, 256, 2, 2),
            ego_support=torch.tensor([False]),
            rsu_support=torch.tensor([False]),
            ego_reliability=torch.tensor([0.0]),
            rsu_reliability=torch.tensor([0.0]),
        )
    finally:
        handle.remove()

    assert len(captured) == 1
    assert torch.count_nonzero(captured[0]).item() > 0
    assert torch.count_nonzero(output.feature).item() == 0


def test_aggregator_rejects_computation_produced_output_shape() -> None:
    aggregator = ModalityAggregator(channels=256).eval()
    handle = aggregator.residual_block.register_forward_hook(
        lambda _module, _inputs, output: output[:, :, :1, :1]
    )
    try:
        with pytest.raises(RuntimeError, match="shape"):
            aggregator(**aggregation_inputs())
    finally:
        handle.remove()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("ego_feature", object()),
        ("ego_feature", torch.zeros(2, 256, 2)),
        ("ego_feature", torch.zeros(0, 256, 2, 2)),
        ("ego_feature", torch.zeros(2, 255, 2, 2)),
        ("ego_feature", torch.zeros(2, 256, 0, 2)),
        ("ego_feature", torch.zeros(2, 256, 2, 2, dtype=torch.int64)),
        ("rsu_feature", torch.zeros(2, 256, 3, 2)),
        ("rsu_feature", torch.zeros(2, 256, 2, 2, dtype=torch.float64)),
        ("ego_support", torch.tensor([[True], [False]])),
        ("ego_support", torch.tensor([1, 0])),
        ("rsu_support", torch.tensor([1.0, 0.0])),
        ("ego_reliability", torch.zeros(2, 1)),
        ("ego_reliability", torch.zeros(2, dtype=torch.int64)),
        ("rsu_reliability", torch.zeros(2, dtype=torch.float64)),
    ],
)
def test_aggregator_rejects_invalid_shapes_and_dtypes(
    field: str,
    replacement: object,
) -> None:
    inputs = aggregation_inputs()
    inputs[field] = replacement

    with pytest.raises(ValueError):
        ModalityAggregator(channels=256)(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ego_feature", math.nan),
        ("rsu_feature", math.inf),
        ("ego_reliability", math.nan),
        ("ego_reliability", math.inf),
        ("ego_reliability", -0.1),
        ("ego_reliability", 1.1),
        ("rsu_reliability", math.nan),
        ("rsu_reliability", -0.1),
        ("rsu_reliability", 1.1),
    ],
)
def test_aggregator_rejects_nonfinite_or_out_of_range_values(
    field: str,
    value: float,
) -> None:
    inputs = aggregation_inputs()
    tensor = inputs[field].clone()
    tensor.flatten()[0] = value
    inputs[field] = tensor

    with pytest.raises(ValueError):
        ModalityAggregator(channels=256)(**inputs)


@pytest.mark.parametrize(
    ("support_field", "reliability_field"),
    [
        ("ego_support", "ego_reliability"),
        ("rsu_support", "rsu_reliability"),
    ],
)
def test_aggregator_requires_neutral_unsupported_reliability(
    support_field: str,
    reliability_field: str,
) -> None:
    inputs = aggregation_inputs()
    support = inputs[support_field].clone()
    reliability = inputs[reliability_field].clone()
    support[0] = False
    reliability[0] = 0.2
    inputs[support_field] = support
    inputs[reliability_field] = reliability

    with pytest.raises(ValueError, match="unsupported|neutral"):
        ModalityAggregator(channels=256)(**inputs)


def test_aggregator_preserves_inputs_and_masks_invalid_gradients() -> None:
    aggregator = ModalityAggregator(channels=256)
    inputs = aggregation_inputs(requires_grad=True)
    before = {
        name: value.detach().clone()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    output = aggregator(**inputs)
    output.feature.square().mean().backward()

    for name, expected in before.items():
        value = inputs[name]
        torch.testing.assert_close(value.detach(), expected, atol=0.0, rtol=0.0)
    ego_feature = inputs["ego_feature"]
    rsu_feature = inputs["rsu_feature"]
    assert ego_feature.grad is not None
    assert rsu_feature.grad is not None
    assert torch.isfinite(ego_feature.grad).all()
    assert torch.isfinite(rsu_feature.grad).all()
    assert torch.count_nonzero(ego_feature.grad[0]).item() > 0
    assert torch.count_nonzero(ego_feature.grad[1]).item() == 0
    assert torch.count_nonzero(rsu_feature.grad[0]).item() == 0
    assert torch.count_nonzero(rsu_feature.grad[1]).item() > 0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in aggregator.parameters()
    )


def test_descriptor_has_exact_frozen_layout() -> None:
    router = make_router().eval()
    inputs = router_inputs()

    output = router(**inputs)

    assert isinstance(output, RoutingOutput)
    assert output.fused.shape == (2, 256, 2, 2)
    assert output.weights.shape == (2, 3)
    assert output.descriptor.shape == (2, 783)
    assert output.expert_support.shape == (2, 3)
    assert output.lidar_expert.shape == (2, 256, 2, 2)
    assert output.camera_expert.shape == (2, 256, 2, 2)
    assert output.synergy_expert.shape == (2, 256, 2, 2)
    assert output.overall_support.shape == (2,)
    torch.testing.assert_close(
        output.descriptor[:, 0:256],
        output.lidar_expert.mean(dim=(-2, -1)),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.descriptor[:, 256:512],
        output.camera_expert.mean(dim=(-2, -1)),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.descriptor[:, 512:768],
        output.synergy_expert.mean(dim=(-2, -1)),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.descriptor[:, 768:770],
        torch.tensor([[0.5, 0.75], [0.95, 0.0]]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.descriptor[:, 770:778],
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.descriptor[:, 778:782],
        inputs["branch_age_intervals"],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.descriptor[:, 782:783],
        inputs["rsu_delay_intervals"],
        atol=0.0,
        rtol=0.0,
    )


def test_synergy_stem_concatenates_lidar_before_camera() -> None:
    router = make_router().eval()
    inputs = single_router_inputs((True, True, True, True))
    inputs["lidar_feature"] = torch.ones(1, 256, 2, 2)
    inputs["camera_feature"] = torch.full((1, 256, 2, 2), 2.0)
    captured: list[torch.Tensor] = []
    handle = router.synergy_stem[0].register_forward_pre_hook(
        lambda _module, values: captured.append(values[0].detach().clone())
    )
    try:
        router(**inputs)
    finally:
        handle.remove()

    assert len(captured) == 1
    torch.testing.assert_close(
        captured[0][:, :256],
        torch.ones_like(captured[0][:, :256]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        captured[0][:, 256:],
        torch.full_like(captured[0][:, 256:], 2.0),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "branch_support",
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, False, True, False),
        (False, True, False, True),
        (True, True, True, True),
    ],
)
def test_expert_and_overall_support_semantics(
    branch_support: tuple[bool, bool, bool, bool],
) -> None:
    inputs = single_router_inputs(branch_support)
    inputs["routing_mode"] = "uniform"

    output = make_router().eval()(**inputs)

    lidar_support = branch_support[0] or branch_support[1]
    camera_support = branch_support[2] or branch_support[3]
    expected = [lidar_support, camera_support, lidar_support and camera_support]
    assert output.expert_support.tolist() == [expected]
    assert output.overall_support.tolist() == [any(expected)]
    for index, is_supported in enumerate(expected):
        expert = (
            output.lidar_expert,
            output.camera_expert,
            output.synergy_expert,
        )[index]
        if not is_supported:
            assert torch.count_nonzero(expert).item() == 0
            assert (
                torch.count_nonzero(
                    output.descriptor[:, index * 256 : (index + 1) * 256]
                ).item()
                == 0
            )


def test_router_masks_invalid_experts_before_gap_and_weighted_sum() -> None:
    router = make_router().eval()
    captured: list[torch.Tensor] = []
    with torch.no_grad():
        for module in router.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.zero_()
            if isinstance(module, nn.GroupNorm):
                module.weight.zero_()
                module.bias.fill_(2.0)
    handle = router.lidar_expert.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().clone())
    )
    inputs = single_router_inputs((False, False, True, False))
    inputs["routing_mode"] = "uniform"
    try:
        output = router(**inputs)
    finally:
        handle.remove()

    assert len(captured) == 1
    assert torch.count_nonzero(captured[0]).item() > 0
    assert torch.count_nonzero(output.lidar_expert).item() == 0
    assert torch.count_nonzero(output.synergy_expert).item() == 0
    assert torch.count_nonzero(output.descriptor[:, :256]).item() == 0
    assert torch.count_nonzero(output.descriptor[:, 512:768]).item() == 0
    assert output.weights.tolist() == [[0.0, 1.0, 0.0]]
    torch.testing.assert_close(
        output.fused,
        output.camera_expert,
        atol=0.0,
        rtol=0.0,
    )


def test_reliability_ablation_changes_only_two_declared_fields() -> None:
    router = make_router().eval()
    inputs = router_inputs()
    inputs["routing_mode"] = "uniform"
    enabled = router(**inputs)
    inputs["use_reliability"] = False
    disabled = router(**inputs)

    torch.testing.assert_close(
        disabled.descriptor[:, 768:770],
        torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        enabled.descriptor[:, :768],
        disabled.descriptor[:, :768],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        enabled.descriptor[:, 770:],
        disabled.descriptor[:, 770:],
        atol=0.0,
        rtol=0.0,
    )


def test_delay_ablation_zeros_only_ages_and_delay_and_keeps_flags() -> None:
    router = make_router().eval()
    inputs = router_inputs()
    inputs["routing_mode"] = "uniform"
    enabled = router(**inputs)
    inputs["use_delay_metadata"] = False
    disabled = router(**inputs)

    assert torch.count_nonzero(disabled.descriptor[:, 778:783]).item() == 0
    torch.testing.assert_close(
        enabled.descriptor[:, :778],
        disabled.descriptor[:, :778],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        disabled.descriptor[:, 770:778],
        enabled.descriptor[:, 770:778],
        atol=0.0,
        rtol=0.0,
    )


def test_dynamic_weights_are_normalized_and_masked() -> None:
    output = make_router().eval()(**router_inputs())

    torch.testing.assert_close(
        output.weights.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
        rtol=0.0,
    )
    assert output.weights[1, 1].item() == 0.0
    assert output.weights[1, 2].item() == 0.0
    assert torch.isfinite(output.weights).all()
    assert output.weights.min().item() >= 0.0
    assert output.weights.max().item() <= 1.0


def test_uniform_mode_assigns_exact_inverse_valid_count() -> None:
    inputs = router_inputs()
    inputs["routing_mode"] = "uniform"

    output = make_router().eval()(**inputs)

    torch.testing.assert_close(
        output.weights,
        torch.tensor([[1.0 / 3.0] * 3, [1.0, 0.0, 0.0]]),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("branch_support", "expected_support"),
    [
        ((True, False, True, False), True),
        ((True, False, False, False), True),
        ((False, False, True, False), True),
        ((False, False, False, False), False),
    ],
)
def test_capacity_matched_concat_uses_masked_modality_experts(
    branch_support: tuple[bool, bool, bool, bool],
    expected_support: bool,
) -> None:
    router = make_router().eval()
    inputs = single_router_inputs(branch_support)
    inputs["routing_mode"] = "concat"

    output = router(**inputs)

    assert output.overall_support.tolist() == [expected_support]
    assert output.expert_support[0, 2].item() is expected_support
    assert output.weights.tolist() == [
        [0.0, 0.0, 1.0 if expected_support else 0.0]
    ]
    torch.testing.assert_close(
        output.fused,
        output.synergy_expert,
        atol=0.0,
        rtol=0.0,
    )


def test_concat_and_dynamic_router_have_identical_parameter_capacity() -> None:
    dynamic = make_router()
    concat = make_router()

    assert trainable_parameter_count(concat) == trainable_parameter_count(
        dynamic
    )


def test_concat_mode_does_not_execute_der_gate() -> None:
    router = make_router().eval()
    inputs = router_inputs()
    inputs["routing_mode"] = "concat"

    def forbid_gate(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
    ) -> None:
        raise AssertionError("concat baseline must not execute DER gate")

    handle = router.gate.register_forward_pre_hook(forbid_gate)
    try:
        output = router(**inputs)
    finally:
        handle.remove()

    assert output.weights.tolist() == [[0.0, 0.0, 1.0]] * 2


def test_static_mode_uses_only_final_gate_bias() -> None:
    router = make_router().eval()
    with torch.no_grad():
        for parameter in router.gate.parameters():
            parameter.fill_(math.nan)
        router.gate[-1].bias.copy_(
            torch.tensor([0.0, math.log(2.0), math.log(3.0)])
        )
    inputs = router_inputs()
    inputs["routing_mode"] = "static"

    output = router(**inputs)

    torch.testing.assert_close(
        output.weights[0],
        torch.tensor([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.weights[1],
        torch.tensor([1.0, 0.0, 0.0]),
        atol=0.0,
        rtol=0.0,
    )


def test_weighted_sum_uses_lidar_camera_synergy_expert_order() -> None:
    router = make_router().eval()
    with torch.no_grad():
        router.gate[-1].bias.copy_(
            torch.tensor([0.0, math.log(2.0), math.log(3.0)])
        )
    inputs = single_router_inputs((True, True, True, True))
    inputs["routing_mode"] = "static"

    output = router(**inputs)

    expected = (
        output.weights[:, 0, None, None, None] * output.lidar_expert
        + output.weights[:, 1, None, None, None] * output.camera_expert
        + output.weights[:, 2, None, None, None] * output.synergy_expert
    )
    torch.testing.assert_close(output.fused, expected)


def test_zero_support_residual_weight_is_bitwise_legacy_path() -> None:
    legacy = DynamicExpertRouter(channels=256, hidden_channels=256).eval()
    explicit_zero = make_router(support_residual_weight=0.0).eval()
    explicit_zero.load_state_dict(legacy.state_dict(), strict=True)
    inputs = router_inputs()

    legacy_output = legacy(**inputs)
    explicit_output = explicit_zero(**inputs)

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
            getattr(explicit_output, field),
            getattr(legacy_output, field),
        )


def test_support_residual_uses_exact_reliability_weighted_batch_formula() -> None:
    legacy = make_router().eval()
    candidate = make_router(support_residual_weight=0.5).eval()
    candidate.load_state_dict(legacy.state_dict(), strict=True)
    inputs = router_inputs()
    inputs["routing_mode"] = "uniform"

    legacy_output = legacy(**inputs)
    candidate_output = candidate(**inputs)

    branch_reliability = inputs["branch_reliability"]
    assert isinstance(branch_reliability, torch.Tensor)
    modality_scores = torch.stack(
        (
            branch_reliability[:, 0:2].sum(dim=1) / 2.0,
            branch_reliability[:, 2:4].sum(dim=1) / 2.0,
        ),
        dim=1,
    )
    lidar_feature = inputs["lidar_feature"]
    camera_feature = inputs["camera_feature"]
    assert isinstance(lidar_feature, torch.Tensor)
    assert isinstance(camera_feature, torch.Tensor)
    support_mean = (
        torch.stack((lidar_feature, camera_feature), dim=1)
        * modality_scores[:, :, None, None, None]
    ).sum(dim=1) / modality_scores.sum(dim=1)[:, None, None, None]
    expected = legacy_output.fused + 0.5 * (support_mean - legacy_output.fused)

    torch.testing.assert_close(candidate_output.fused, expected)
    assert torch.equal(candidate_output.weights, legacy_output.weights)
    assert torch.equal(candidate_output.descriptor, legacy_output.descriptor)
    assert torch.equal(candidate_output.lidar_expert, legacy_output.lidar_expert)
    assert torch.equal(candidate_output.camera_expert, legacy_output.camera_expert)
    assert torch.equal(candidate_output.synergy_expert, legacy_output.synergy_expert)
    # The second batch row has only one supported modality, so the fixed mean
    # is exactly that source regardless of its positive reliability value.
    torch.testing.assert_close(support_mean[1], lidar_feature[1])


def test_support_residual_masks_unsupported_source_and_empty_rows() -> None:
    candidate = make_router(support_residual_weight=0.5).eval()
    inputs = router_inputs()
    inputs["routing_mode"] = "uniform"
    inputs["lidar_branch_support"] = torch.tensor([[True, False], [False, False]])
    inputs["camera_branch_support"] = torch.tensor([[False, False], [False, False]])
    inputs["branch_reliability"] = torch.tensor(
        [[0.7, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    inputs["branch_observed"] = torch.tensor(
        [[True, False, False, False], [False, False, False, False]]
    )
    inputs["branch_propagated"] = torch.zeros(2, 4, dtype=torch.bool)
    inputs["branch_age_intervals"] = torch.zeros(2, 4)
    inputs["rsu_delay_intervals"] = torch.zeros(2, 1)
    first = candidate(**inputs)

    changed = dict(inputs)
    changed_camera = inputs["camera_feature"].clone()
    changed_camera.fill_(10_000.0)
    changed["camera_feature"] = changed_camera
    changed_lidar = inputs["lidar_feature"].clone()
    changed_lidar[1].fill_(-10_000.0)
    changed["lidar_feature"] = changed_lidar
    second = candidate(**changed)

    assert torch.equal(first.fused[0], second.fused[0])
    assert torch.count_nonzero(first.fused[1]).item() == 0
    assert torch.count_nonzero(second.fused[1]).item() == 0
    assert first.overall_support.tolist() == [True, False]
    assert second.overall_support.tolist() == [True, False]


def test_support_residual_state_dict_schema_is_unchanged() -> None:
    legacy = make_router()
    candidate = make_router(support_residual_weight=0.5)

    legacy_schema = tuple(
        (name, tuple(value.shape)) for name, value in legacy.state_dict().items()
    )
    candidate_schema = tuple(
        (name, tuple(value.shape)) for name, value in candidate.state_dict().items()
    )

    assert candidate_schema == legacy_schema
    assert trainable_parameter_count(candidate) == trainable_parameter_count(legacy)
    candidate.load_state_dict(legacy.state_dict(), strict=True)


def test_all_invalid_is_exact_zero_and_never_nan_in_every_mode() -> None:
    for mode in ("dynamic", "static", "uniform", "concat"):
        inputs = single_router_inputs((False, False, False, False))
        inputs["routing_mode"] = mode
        output = make_router().eval()(**inputs)

        assert torch.count_nonzero(output.weights).item() == 0
        assert torch.count_nonzero(output.fused).item() == 0
        assert torch.count_nonzero(output.lidar_expert).item() == 0
        assert torch.count_nonzero(output.camera_expert).item() == 0
        assert torch.count_nonzero(output.synergy_expert).item() == 0
        assert not torch.isnan(output.weights).any()
        assert output.overall_support.tolist() == [False]


class _ZeroPTF:
    def query(
        self,
        context: torch.Tensor,
        _query_agent_index: torch.Tensor,
        _horizon: torch.Tensor,
    ) -> PTFOutput:
        batch = context.shape[0]
        return PTFOutput(
            displacement=context.new_zeros(batch, 2, 4, 4),
            confidence=context.new_full((batch, 1, 4, 4), 0.5),
        )


def _selection(
    agent: Agent,
    modality: Modality,
    horizon: int,
) -> BranchSelection:
    tick = 10 - horizon
    source_tau_ms = tick * 100
    endpoint_tick = 10 if agent is Agent.EGO else tick
    observed = tick == endpoint_tick
    return BranchSelection(
        agent=agent,
        modality=modality,
        supported=True,
        source=SourceCandidate(
            packet_id=f"sequence:{modality.value}:{agent.value}:{tick}",
            n_s=tick,
            tau_s_ms=source_tau_ms,
            arrival_tau_ms=source_tau_ms + (100 if agent is Agent.RSU else 0),
            payload_valid=True,
            timestamp_valid=True,
            pose_valid=True,
            calibration_valid=True,
            faulted=False,
        ),
        horizon=horizon,
        endpoint_tick=endpoint_tick,
        observed=observed,
        propagated=not observed,
        rejected=(),
        reason=None,
    )


def test_paper_interval_ages_enter_descriptor_without_division_by_k() -> None:
    repair = CausalBranchRepair(
        grid_spec=BEVGridSpec(
            x_min=0.0,
            y_min=0.0,
            resolution=1.0,
            height=4,
            width=4,
        ),
        channels=256,
        alpha=0.9,
        history_limit=3,
    )
    selections = (
        _selection(Agent.EGO, Modality.LIDAR, 0),
        _selection(Agent.RSU, Modality.LIDAR, 1),
        _selection(Agent.EGO, Modality.CAMERA, 2),
        _selection(Agent.RSU, Modality.CAMERA, 3),
    )
    assert tuple(
        (selection.modality, selection.agent) for selection in selections
    ) == (
        (Modality.LIDAR, Agent.EGO),
        (Modality.LIDAR, Agent.RSU),
        (Modality.CAMERA, Agent.EGO),
        (Modality.CAMERA, Agent.RSU),
    )
    repaired_modalities = []
    for modality_selections in (selections[:2], selections[2:]):
        repaired_modalities.append(
            repair(
                selected_feature=torch.zeros(2, 256, 4, 4),
                source_to_target=torch.eye(4).repeat(2, 1, 1),
                ptf_context=torch.zeros(2, 1, 4, 4),
                ptf=_ZeroPTF(),
                query_agent_index=torch.tensor([0, 1]),
                selections=modality_selections,
            )
        )
    lidar_repaired, camera_repaired = repaired_modalities
    packed_support = torch.cat(
        (lidar_repaired.support, camera_repaired.support)
    ).reshape(1, 4)
    packed_reliability = torch.cat(
        (lidar_repaired.reliability, camera_repaired.reliability)
    ).reshape(1, 4)
    packed_observed = torch.cat(
        (lidar_repaired.observed, camera_repaired.observed)
    ).reshape(1, 4)
    packed_propagated = torch.cat(
        (lidar_repaired.propagated, camera_repaired.propagated)
    ).reshape(1, 4)
    packed_age = torch.cat(
        (lidar_repaired.age_intervals, camera_repaired.age_intervals)
    ).reshape(1, 4)
    inputs = {
        "lidar_feature": torch.randn(1, 256, 2, 2),
        "camera_feature": torch.randn(1, 256, 2, 2),
        "lidar_branch_support": packed_support[:, :2],
        "camera_branch_support": packed_support[:, 2:],
        "branch_reliability": packed_reliability,
        "branch_observed": packed_observed,
        "branch_propagated": packed_propagated,
        "branch_age_intervals": packed_age,
        "rsu_delay_intervals": torch.tensor([[1.0]]),
        "routing_mode": "uniform",
        "use_reliability": True,
        "use_delay_metadata": True,
    }

    output = make_router().eval()(**inputs)

    expected = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    torch.testing.assert_close(
        packed_age,
        expected,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.descriptor[:, 778:782],
        expected,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("lidar_feature", object()),
        ("lidar_feature", torch.zeros(2, 256, 2)),
        ("lidar_feature", torch.zeros(0, 256, 2, 2)),
        ("lidar_feature", torch.zeros(2, 255, 2, 2)),
        ("lidar_feature", torch.zeros(2, 256, 0, 2)),
        ("lidar_feature", torch.zeros(2, 256, 2, 2, dtype=torch.int64)),
        ("camera_feature", torch.zeros(2, 256, 3, 2)),
        ("camera_feature", torch.zeros(2, 256, 2, 2, dtype=torch.float64)),
        ("lidar_branch_support", torch.ones(2, 2)),
        ("lidar_branch_support", torch.ones(2, 1, dtype=torch.bool)),
        ("camera_branch_support", torch.ones(2, 3, dtype=torch.bool)),
        ("branch_reliability", torch.zeros(2, 3)),
        ("branch_reliability", torch.zeros(2, 4, dtype=torch.int64)),
        ("branch_observed", torch.zeros(2, 4)),
        ("branch_observed", torch.zeros(2, 3, dtype=torch.bool)),
        ("branch_propagated", torch.zeros(2, 4, dtype=torch.int64)),
        ("branch_age_intervals", torch.zeros(2, 3)),
        ("branch_age_intervals", torch.zeros(2, 4, dtype=torch.int64)),
        ("rsu_delay_intervals", torch.zeros(2)),
        ("rsu_delay_intervals", torch.zeros(2, 1, dtype=torch.int64)),
    ],
)
def test_router_rejects_invalid_shapes_and_dtypes(
    field: str,
    replacement: object,
) -> None:
    inputs = router_inputs()
    inputs[field] = replacement

    with pytest.raises(ValueError):
        make_router()(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lidar_feature", math.nan),
        ("camera_feature", math.inf),
        ("branch_reliability", math.nan),
        ("branch_reliability", math.inf),
        ("branch_reliability", -0.1),
        ("branch_reliability", 1.1),
        ("branch_age_intervals", math.nan),
        ("branch_age_intervals", math.inf),
        ("branch_age_intervals", -0.1),
        ("branch_age_intervals", 3.1),
        ("branch_age_intervals", 4.0),
        ("rsu_delay_intervals", math.nan),
        ("rsu_delay_intervals", math.inf),
        ("rsu_delay_intervals", -0.1),
        ("rsu_delay_intervals", 3.1),
    ],
)
def test_router_rejects_nonfinite_or_out_of_range_values(
    field: str,
    value: float,
) -> None:
    inputs = router_inputs()
    tensor = inputs[field].clone()
    tensor.flatten()[0] = value
    inputs[field] = tensor

    with pytest.raises(ValueError):
        make_router()(**inputs)


@pytest.mark.parametrize(
    ("field", "index", "value"),
    [
        ("branch_reliability", (0, 1), 0.2),
        ("branch_observed", (0, 1), True),
        ("branch_propagated", (0, 1), True),
        ("branch_age_intervals", (0, 1), 0.2),
        ("branch_reliability", (1, 2), 0.2),
        ("branch_observed", (1, 2), True),
        ("branch_propagated", (1, 2), True),
        ("branch_age_intervals", (1, 2), 0.2),
    ],
)
def test_router_requires_unsupported_branch_metadata_to_be_neutral(
    field: str,
    index: tuple[int, int],
    value: object,
) -> None:
    inputs = router_inputs()
    tensor = inputs[field].clone()
    tensor[index] = value
    inputs[field] = tensor

    with pytest.raises(ValueError, match="unsupported|neutral"):
        make_router()(**inputs)


@pytest.mark.parametrize(
    ("observed", "propagated"),
    [(False, False), (True, True)],
    ids=["neither", "both"],
)
def test_supported_branch_requires_exactly_one_flag(
    observed: bool,
    propagated: bool,
) -> None:
    inputs = router_inputs()
    branch_observed = inputs["branch_observed"].clone()
    branch_propagated = inputs["branch_propagated"].clone()
    branch_observed[0, 0] = observed
    branch_propagated[0, 0] = propagated
    inputs["branch_observed"] = branch_observed
    inputs["branch_propagated"] = branch_propagated

    with pytest.raises(ValueError, match="exactly one|observed|propagated"):
        make_router()(**inputs)


def test_arrived_rsu_delay_remains_visible_when_sensor_branches_are_unsupported() -> None:
    inputs = single_router_inputs((True, False, True, False))
    inputs["rsu_delay_intervals"] = torch.tensor([[0.2]])

    output = make_router()(**inputs)

    assert output.descriptor[0, 782].item() == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("routing_mode", "adaptive"),
        ("routing_mode", "Dynamic"),
        ("routing_mode", 1),
        ("routing_mode", torch.tensor([1, 2])),
        ("routing_mode", _PretendDynamicMode()),
        ("use_reliability", 1),
        ("use_reliability", None),
        ("use_delay_metadata", 0),
        ("use_delay_metadata", "yes"),
    ],
)
def test_router_rejects_invalid_mode_and_non_boolean_switches(
    field: str,
    value: object,
) -> None:
    inputs = router_inputs()
    inputs[field] = value

    with pytest.raises(ValueError):
        make_router()(**inputs)


def test_router_rejects_string_subclass_before_expert_or_gate_computation() -> None:
    router = make_router()
    inputs = router_inputs()
    inputs["routing_mode"] = _SpoofedRoutingMode()

    def forbid_computation(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
    ) -> None:
        raise AssertionError(
            "routing mode validation must precede expert and gate computation"
        )

    handles = [
        module.register_forward_pre_hook(forbid_computation)
        for module in (
            router.lidar_expert,
            router.camera_expert,
            router.synergy_stem,
            router.synergy_expert,
            router.gate,
        )
    ]
    try:
        with pytest.raises(ValueError, match="routing_mode"):
            router(**inputs)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize(
    "mode", ["dynamic", "static", "uniform", "concat"]
)
def test_router_accepts_only_each_builtin_routing_mode(mode: str) -> None:
    inputs = router_inputs()
    inputs["routing_mode"] = mode

    output = make_router().eval()(**inputs)

    assert output.weights.shape == (2, 3)


@pytest.mark.parametrize(
    "field",
    [
        "lidar_branch_support",
        "camera_branch_support",
        "branch_reliability",
        "branch_observed",
        "branch_propagated",
        "branch_age_intervals",
        "rsu_delay_intervals",
    ],
)
def test_router_rejects_metadata_device_mismatch(field: str) -> None:
    inputs = router_inputs()
    value = inputs[field]
    assert isinstance(value, torch.Tensor)
    inputs[field] = torch.empty(
        value.shape,
        dtype=value.dtype,
        device="meta",
    )

    with pytest.raises(ValueError, match="device"):
        make_router()(**inputs)


def test_router_rejects_feature_dtype_mismatch_with_parameters() -> None:
    router = make_router().double()
    inputs = router_inputs()

    with pytest.raises(ValueError, match="dtype"):
        router(**inputs)


def test_router_rejects_computation_produced_expert_shape() -> None:
    router = make_router().eval()
    handles = [
        expert.register_forward_hook(
            lambda _module, _inputs, output: output[:, :, :1, :1]
        )
        for expert in (
            router.lidar_expert,
            router.camera_expert,
            router.synergy_expert,
        )
    ]
    try:
        with pytest.raises(RuntimeError, match="shape"):
            router(**router_inputs())
    finally:
        for handle in handles:
            handle.remove()


def test_computation_produced_nonfinite_expert_is_an_error() -> None:
    router = make_router().eval()
    with torch.no_grad():
        router.lidar_expert[0].normalization1.bias[0] = math.nan

    with pytest.raises(RuntimeError, match="finite|expert"):
        router(**router_inputs())


def test_computation_produced_nonfinite_logits_are_an_error() -> None:
    router = make_router().eval()
    with torch.no_grad():
        router.gate[-1].bias[0] = math.nan

    with pytest.raises(RuntimeError, match="finite|logits"):
        router(**router_inputs())


def test_router_does_not_mutate_or_detach_inputs() -> None:
    router = make_router().eval()
    inputs = router_inputs(requires_grad=True)
    before = {
        name: value.detach().clone()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    output = router(**inputs)

    for name, expected in before.items():
        value = inputs[name]
        assert isinstance(value, torch.Tensor)
        torch.testing.assert_close(value.detach(), expected, atol=0.0, rtol=0.0)
    assert output.fused.requires_grad
    assert output.descriptor.requires_grad
    assert output.weights.requires_grad


def test_valid_gradients_are_finite_and_invalid_rows_contribute_zero() -> None:
    router = make_router()
    inputs = router_inputs(requires_grad=True)
    inputs["lidar_branch_support"] = torch.tensor(
        [[True, True], [False, False]]
    )
    inputs["camera_branch_support"] = torch.tensor(
        [[True, True], [False, False]]
    )
    inputs["branch_reliability"] = torch.tensor(
        [[0.9, 0.8, 0.7, 0.6], [0.0, 0.0, 0.0, 0.0]]
    )
    inputs["branch_observed"] = torch.tensor(
        [[True, False, True, False], [False, False, False, False]]
    )
    inputs["branch_propagated"] = torch.tensor(
        [[False, True, False, True], [False, False, False, False]]
    )
    inputs["branch_age_intervals"] = torch.tensor(
        [[0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0], [0.0, 0.0, 0.0, 0.0]]
    )
    inputs["rsu_delay_intervals"] = torch.tensor([[0.5], [0.0]])

    output = router(**inputs)
    loss = (
        output.fused.square().mean()
        + output.weights.square().sum()
        + output.descriptor[:, :768].square().mean()
    )
    loss.backward()

    for name in ("lidar_feature", "camera_feature"):
        value = inputs[name]
        assert isinstance(value, torch.Tensor)
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[0]).item() > 0
        assert torch.count_nonzero(value.grad[1]).item() == 0
    parameter_gradients = [
        parameter.grad for parameter in router.parameters()
    ]
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in parameter_gradients
        if gradient is not None
    )


def test_invalid_row_feature_values_cannot_change_any_output() -> None:
    router = make_router().eval()
    inputs = router_inputs()
    inputs["lidar_branch_support"] = torch.tensor(
        [[True, True], [False, False]]
    )
    inputs["camera_branch_support"] = torch.tensor(
        [[True, True], [False, False]]
    )
    inputs["branch_reliability"] = torch.tensor(
        [[0.9, 0.8, 0.7, 0.6], [0.0, 0.0, 0.0, 0.0]]
    )
    inputs["branch_observed"] = torch.tensor(
        [[True, False, True, False], [False, False, False, False]]
    )
    inputs["branch_propagated"] = torch.tensor(
        [[False, True, False, True], [False, False, False, False]]
    )
    inputs["branch_age_intervals"] = torch.tensor(
        [[0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0], [0.0, 0.0, 0.0, 0.0]]
    )
    inputs["rsu_delay_intervals"] = torch.tensor([[0.5], [0.0]])
    first = router(**inputs)
    changed = dict(inputs)
    changed_lidar = inputs["lidar_feature"].clone()
    changed_camera = inputs["camera_feature"].clone()
    changed_lidar[1].fill_(10_000.0)
    changed_camera[1].fill_(-10_000.0)
    changed["lidar_feature"] = changed_lidar
    changed["camera_feature"] = changed_camera
    second = router(**changed)

    torch.testing.assert_close(
        first.fused[0],
        second.fused[0],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        first.weights[0],
        second.weights[0],
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(first.fused[1]).item() == 0
    assert torch.count_nonzero(second.fused[1]).item() == 0
    assert torch.count_nonzero(first.weights[1]).item() == 0
    assert torch.count_nonzero(second.weights[1]).item() == 0


def test_task_six_package_import_does_not_load_custom_ops() -> None:
    code = (
        "import sys; "
        "from transvision.models.resilient_v2x import DynamicExpertRouter; "
        "assert 'transvision.models.bev_pool' not in sys.modules; "
        "assert 'transvision.models.voxel.voxel_layer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
