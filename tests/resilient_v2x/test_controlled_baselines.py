from __future__ import annotations

import pytest
import torch

from transvision.models.resilient_v2x.baselines import (
    BEVFusionAdaptedFusion,
    CoBEVTAdaptedFusion,
    CoFormerNetAdaptedFusion,
    FFNetAdaptedFusion,
    V2XViTAdaptedFusion,
    build_controlled_baseline_fusion,
)


BASELINE_TYPES = {
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
