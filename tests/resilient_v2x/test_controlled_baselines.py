from __future__ import annotations

import pytest
import torch

from transvision.models.resilient_v2x.baselines import (
    AttFuseAdaptedFusion,
    BEVFusionAdaptedFusion,
    CoBEVTAdaptedFusion,
    CoFormerNetAdaptedFusion,
    FFNetAdaptedFusion,
    EgoOnlyFusion,
    FCooperAdaptedFusion,
    V2XViTAdaptedFusion,
    build_controlled_baseline_fusion,
)


BASELINE_TYPES = {
    "ego_only": EgoOnlyFusion,
    "fcooper": FCooperAdaptedFusion,
    "attfuse": AttFuseAdaptedFusion,
    "v2x_vit": V2XViTAdaptedFusion,
    "cobevt": CoBEVTAdaptedFusion,
    "coformernet": CoFormerNetAdaptedFusion,
    "bevfusion": BEVFusionAdaptedFusion,
    "ffnet": FFNetAdaptedFusion,
}


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branches = torch.randn(2, 4, 8, 3, 5, requires_grad=True)
    support = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, False],
        ]
    )
    ages = torch.tensor(
        [
            [0.0, 1.0, 0.0, 2.0],
            [1.0, float("nan"), 3.0, float("nan")],
        ]
    )
    return branches, support, ages


@pytest.mark.parametrize(("name", "expected_type"), BASELINE_TYPES.items())
def test_factory_builds_each_controlled_baseline(
    name: str,
    expected_type: type[torch.nn.Module],
) -> None:
    kwargs = {"window_sizes": (2, 3)} if name == "v2x_vit" else {}
    if name == "coformernet":
        kwargs["window_size"] = 3

    module = build_controlled_baseline_fusion(name, channels=8, **kwargs)

    assert isinstance(module, expected_type)


@pytest.mark.parametrize("name", BASELINE_TYPES)
def test_controlled_baseline_has_common_shape_and_backpropagates(
    name: str,
) -> None:
    torch.manual_seed(7)
    branches, support, ages = _inputs()
    module = build_controlled_baseline_fusion(name, channels=8)

    fused = module(branches, support, ages)
    fused.square().mean().backward()

    assert fused.shape == (2, 8, 3, 5)
    assert torch.isfinite(fused).all()
    assert branches.grad is not None
    assert torch.isfinite(branches.grad).all()
    assert torch.count_nonzero(branches.grad[1, 1]).item() == 0
    assert torch.count_nonzero(branches.grad[1, 3]).item() == 0


@pytest.mark.parametrize("name", BASELINE_TYPES)
def test_missing_branch_values_are_never_observed(name: str) -> None:
    torch.manual_seed(13)
    branches, support, ages = _inputs()
    branches = branches.detach()
    module = build_controlled_baseline_fusion(name, channels=8).eval()

    expected = module(branches, support, ages)
    poisoned = branches.clone()
    poisoned[1, 1] = float("nan")
    poisoned[1, 3] = float("inf")
    actual = module(poisoned, support, ages)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name", BASELINE_TYPES)
def test_all_missing_sample_fails_fast(name: str) -> None:
    branches, support, ages = _inputs()
    support[1] = False
    module = build_controlled_baseline_fusion(name, channels=8)

    with pytest.raises(
        ValueError,
        match=r"all branches are missing for sample indices \[1\]",
    ):
        module(branches, support, ages)


def test_ego_only_strictly_ignores_rsu_values_support_ages_and_gradients() -> None:
    torch.manual_seed(23)
    branches = torch.randn(1, 4, 3, 2, 2, requires_grad=True)
    support = torch.ones(1, 4, dtype=torch.bool)
    ages = torch.tensor(((0.0, 2.0, 1.0, 4.0),))
    module = EgoOnlyFusion(channels=3).eval()

    expected = module(branches, support, ages)
    poisoned = branches.detach().clone()
    poisoned[:, 1] = float("nan")
    poisoned[:, 3] = float("inf")
    torch.testing.assert_close(module(poisoned, support, ages), expected.detach())

    ego_support_only = support.clone()
    ego_support_only[:, (1, 3)] = False
    changed_ages = ages.clone()
    changed_ages[:, (1, 3)] = float("nan")
    torch.testing.assert_close(
        module(branches.detach(), ego_support_only, changed_ages),
        expected.detach(),
    )

    expected.sum().backward()
    assert branches.grad is not None
    assert torch.count_nonzero(branches.grad[:, 0]).item() > 0
    assert torch.count_nonzero(branches.grad[:, 2]).item() > 0
    assert torch.count_nonzero(branches.grad[:, 1]).item() == 0
    assert torch.count_nonzero(branches.grad[:, 3]).item() == 0


def test_ego_only_rejects_samples_with_only_rsu_support() -> None:
    branches = torch.zeros(1, 4, 2, 1, 1)
    support = torch.tensor(((False, True, False, True),))
    ages = torch.tensor(((float("nan"), 0.0, float("nan"), 2.0),))
    with pytest.raises(ValueError, match="both ego branches are missing"):
        EgoOnlyFusion(channels=2)(branches, support, ages)


def test_fcooper_is_exact_support_aware_spatial_max_and_delay_invariant() -> None:
    branches = torch.tensor(
        [[
            [[[1.0, -5.0], [3.0, 4.0]]],
            [[[99.0, 99.0], [99.0, 99.0]]],
            [[[2.0, -7.0], [1.0, 5.0]]],
            [[[0.0, -4.0], [8.0, 2.0]]],
        ]]
    )
    support = torch.tensor(((True, False, True, True),))
    ages = torch.tensor(((0.0, float("nan"), 200.0, 7.0),))
    module = FCooperAdaptedFusion(channels=1)
    expected = torch.tensor([[[[2.0, -4.0], [8.0, 5.0]]]])

    torch.testing.assert_close(module(branches, support, ages), expected)
    changed_ages = torch.tensor(((10.0, float("nan"), 0.0, 100.0),))
    torch.testing.assert_close(module(branches, support, changed_ages), expected)
    poisoned = branches.clone()
    poisoned[:, 1] = float("inf")
    torch.testing.assert_close(module(poisoned, support, ages), expected)


def test_attfuse_masks_weights_and_uses_age_and_per_position_scores() -> None:
    branches = torch.zeros(1, 4, 2, 1, 2)
    support = torch.tensor(((True, False, True, True),))
    ages = torch.tensor(((0.0, float("nan"), 2.0, 1.0),))
    module = AttFuseAdaptedFusion(channels=2, age_decay=1.0).eval()
    with torch.no_grad():
        module.score_projection.weight.zero_()
        module.score_projection.bias.zero_()
        module.branch_bias.zero_()

    clean, clean_support, clean_ages = module._prepare_inputs(
        branches, support, ages
    )
    weights = module._attention_weights(clean, clean_support, clean_ages)
    assert weights.shape == (1, 4, 1, 2)
    assert torch.count_nonzero(weights[:, 1]).item() == 0
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1, 1, 2))
    expected_supported = torch.softmax(torch.tensor((0.0, -2.0, -1.0)), dim=0)
    torch.testing.assert_close(weights[0, (0, 2, 3), 0, 0], expected_supported)

    with torch.no_grad():
        module.score_projection.weight[0, 0, 0, 0] = 1.0
    spatial = branches.clone()
    spatial[0, 0, 0, 0] = torch.tensor((4.0, -4.0))
    spatial[0, 2, 0, 0] = torch.tensor((-4.0, 4.0))
    clean, clean_support, clean_ages = module._prepare_inputs(
        spatial, support, ages
    )
    spatial_weights = module._attention_weights(
        clean, clean_support, clean_ages
    )
    assert spatial_weights[0, 0, 0, 0] > spatial_weights[0, 0, 0, 1]
    assert spatial_weights[0, 2, 0, 0] < spatial_weights[0, 2, 0, 1]


def test_common_contract_rejects_invalid_metadata() -> None:
    branches, support, ages = _inputs()
    module = build_controlled_baseline_fusion("bevfusion", channels=8)

    with pytest.raises(ValueError, match="support must be a bool tensor"):
        module(branches, support.float(), ages)
    ages[0, 0] = -1.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        module(branches, support, ages)


def test_factory_rejects_unknown_name_and_invalid_attention_shape() -> None:
    with pytest.raises(ValueError, match="name must be one of"):
        build_controlled_baseline_fusion("unknown")
    with pytest.raises(ValueError, match="divisible by num_heads"):
        build_controlled_baseline_fusion(
            "v2x_vit",
            channels=10,
            num_heads=4,
        )


def test_ffnet_only_predicts_supported_delayed_rsu_branches() -> None:
    class AddConstantPrediction(torch.nn.Module):
        def forward(
            self,
            features: torch.Tensor,
            ages: torch.Tensor,
        ) -> torch.Tensor:
            del ages
            return features + 10.0

    branches = torch.arange(
        2 * 4 * 2 * 2 * 3,
        dtype=torch.float32,
    ).reshape(2, 4, 2, 2, 3)
    support = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, True],
        ]
    )
    ages = torch.tensor(
        [
            [3.0, 2.0, 4.0, 0.0],
            [2.0, float("nan"), 1.0, 3.0],
        ]
    )
    module = FFNetAdaptedFusion(channels=2)
    module.lidar_feature_flow = AddConstantPrediction()
    module.camera_feature_flow = AddConstantPrediction()
    clean, support, ages = module._prepare_inputs(branches, support, ages)

    compensated = module._compensate_rsu_branches(
        clean,
        support,
        ages,
    )

    torch.testing.assert_close(compensated[:, 0], clean[:, 0])
    torch.testing.assert_close(compensated[:, 2], clean[:, 2])
    torch.testing.assert_close(compensated[0, 1], clean[0, 1] + 10.0)
    torch.testing.assert_close(compensated[0, 3], clean[0, 3])
    torch.testing.assert_close(compensated[1, 1], clean[1, 1])
    torch.testing.assert_close(compensated[1, 3], clean[1, 3] + 10.0)
