from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("mmdet3d")

from mmdet3d.registry import MODELS
from mmengine.config import Config

from transvision.models.resilient_v2x.when2com_baseline import (
    When2comAdaptedFusion,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branches = torch.randn(2, 4, 8, 3, 5, requires_grad=True)
    support = torch.tensor(
        (
            (True, True, True, True),
            (True, False, True, False),
        )
    )
    ages = torch.tensor(
        (
            (0.0, 2.0, 1.0, 3.0),
            (2.0, float("nan"), 0.0, float("nan")),
        )
    )
    return branches, support, ages


def test_when2com_is_registered_and_config_is_explicitly_style_adapted() -> None:
    assert MODELS.get("When2comAdaptedFusion") is When2comAdaptedFusion
    built = MODELS.build(
        dict(
            type="When2comAdaptedFusion",
            channels=8,
            query_key_channels=4,
        )
    )
    assert isinstance(built, When2comAdaptedFusion)

    root = Path(__file__).resolve().parents[2]
    config = Config.fromfile(root / "configs/resilient_v2x/baselines/when2com.py")
    assert config.model.baseline_name == "when2com"
    assert config.model.baseline_cfg == {
        "query_key_channels": 64,
        "age_decay": 0.25,
        "temperature": 1.0,
    }
    assert "When2com-style" in config.experiment.method
    assert config.experiment.claim_status == (
        "controlled adaptation; not an exact source-paper reproduction"
    )


def test_when2com_aggregates_four_branches_to_two_agents_and_backpropagates() -> None:
    torch.manual_seed(7)
    branches, support, ages = _inputs()
    module = When2comAdaptedFusion(
        channels=8,
        query_key_channels=4,
    )

    fused = module(branches, support, ages)
    weights = module.communication_weights(branches, support, ages)
    fused.square().mean().backward()

    assert fused.shape == (2, 8, 3, 5)
    assert torch.isfinite(fused).all()
    assert weights.shape == (2, 2)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(weights[1], torch.tensor((1.0, 0.0)))
    assert branches.grad is not None
    assert torch.isfinite(branches.grad).all()
    assert torch.count_nonzero(branches.grad[1, 0]).item() > 0
    assert torch.count_nonzero(branches.grad[1, 2]).item() > 0
    assert torch.count_nonzero(branches.grad[1, 1]).item() == 0
    assert torch.count_nonzero(branches.grad[1, 3]).item() == 0


def test_unsupported_agents_cannot_pollute_values_weights_or_gradients() -> None:
    torch.manual_seed(13)
    branches = torch.randn(2, 4, 4, 2, 3, requires_grad=True)
    support = torch.tensor(
        (
            (True, False, True, False),
            (False, True, False, True),
        )
    )
    ages = torch.tensor(
        (
            (0.0, float("nan"), 1.0, float("inf")),
            (float("nan"), 2.0, float("inf"), 0.0),
        )
    )
    module = When2comAdaptedFusion(
        channels=4,
        query_key_channels=2,
    ).eval()

    expected = module(branches, support, ages)
    expected_weights = module.communication_weights(branches, support, ages)
    poisoned = branches.detach().clone()
    poisoned[0, (1, 3)] = float("nan")
    poisoned[1, (0, 2)] = float("inf")

    torch.testing.assert_close(module(poisoned, support, ages), expected.detach())
    torch.testing.assert_close(
        module.communication_weights(poisoned, support, ages),
        expected_weights.detach(),
    )
    torch.testing.assert_close(
        expected_weights,
        torch.tensor(((1.0, 0.0), (0.0, 1.0))),
    )

    expected.sum().backward()
    assert branches.grad is not None
    assert torch.count_nonzero(branches.grad[0, (1, 3)]).item() == 0
    assert torch.count_nonzero(branches.grad[1, (0, 2)]).item() == 0


def test_unsupported_modality_is_masked_inside_a_supported_agent() -> None:
    torch.manual_seed(17)
    branches = torch.randn(1, 4, 4, 2, 3, requires_grad=True)
    support = torch.tensor(((True, True, False, True),))
    ages = torch.tensor(((0.0, 1.0, float("nan"), 2.0),))
    module = When2comAdaptedFusion(
        channels=4,
        query_key_channels=2,
    ).eval()

    expected = module(branches, support, ages)
    poisoned = branches.detach().clone()
    poisoned[:, 2] = float("nan")
    torch.testing.assert_close(module(poisoned, support, ages), expected.detach())

    expected.sum().backward()
    assert branches.grad is not None
    assert torch.count_nonzero(branches.grad[:, 2]).item() == 0
    assert torch.count_nonzero(branches.grad[:, 0]).item() > 0


def test_support_mask_controls_query_key_gate_and_age_penalty() -> None:
    branches = torch.ones(1, 4, 2, 1, 1)
    support = torch.ones(1, 4, dtype=torch.bool)
    module = When2comAdaptedFusion(
        channels=2,
        query_key_channels=2,
        age_decay=1.0,
    ).eval()
    with torch.no_grad():
        module.query_projection.weight.copy_(torch.eye(2))
        module.key_projection.weight.copy_(torch.eye(2))

    equal_ages = torch.zeros(1, 4)
    equal_weights = module.communication_weights(branches, support, equal_ages)
    torch.testing.assert_close(equal_weights, torch.full((1, 2), 0.5))

    delayed_rsu = torch.tensor(((0.0, 4.0, 0.0, 4.0),))
    delayed_weights = module.communication_weights(branches, support, delayed_rsu)
    assert delayed_weights[0, 0] > delayed_weights[0, 1]

    ego_only_support = support.clone()
    ego_only_support[:, (1, 3)] = False
    unsupported_ages = delayed_rsu.clone()
    unsupported_ages[:, (1, 3)] = float("nan")
    torch.testing.assert_close(
        module.communication_weights(
            branches,
            ego_only_support,
            unsupported_ages,
        ),
        torch.tensor(((1.0, 0.0),)),
    )


def test_when2com_rejects_all_missing_samples_and_invalid_options() -> None:
    branches, support, ages = _inputs()
    support[1] = False
    with pytest.raises(
        ValueError,
        match=r"all branches are missing for sample indices \[1\]",
    ):
        When2comAdaptedFusion(channels=8)(branches, support, ages)

    with pytest.raises(ValueError, match="query_key_channels"):
        When2comAdaptedFusion(query_key_channels=0)
    with pytest.raises(ValueError, match="temperature"):
        When2comAdaptedFusion(temperature=0.0)
