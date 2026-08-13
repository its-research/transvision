from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

pytest.importorskip("mmdet3d")

from mmdet3d.registry import MODELS

from transvision.models.resilient_v2x.disconet_baseline import (
    DiscoNetAdaptedFusion,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "resilient_v2x" / "baselines" / "disconet.py"


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branches = torch.randn(2, 4, 4, 2, 3, requires_grad=True)
    support = torch.tensor(
        (
            (True, True, True, True),
            (True, False, True, False),
        )
    )
    ages = torch.tensor(
        (
            (0.0, 2.0, 1.0, 3.0),
            (1.0, float("nan"), 2.0, float("inf")),
        )
    )
    return branches, support, ages


def test_disconet_is_registered_and_config_declares_controlled_adaptation() -> None:
    assert MODELS.get("DiscoNetAdaptedFusion") is DiscoNetAdaptedFusion
    registered = MODELS.build(
        dict(
            type="DiscoNetAdaptedFusion",
            channels=4,
            edge_hidden_channels=2,
            age_decay=0.25,
        )
    )
    assert isinstance(registered, DiscoNetAdaptedFusion)

    config = runpy.run_path(str(CONFIG))
    assert config["model"] == {
        "baseline_name": "disconet",
        "baseline_cfg": {
            "_delete_": True,
            "edge_hidden_channels": 128,
            "age_decay": 0.25,
        },
    }
    assert config["custom_imports"] == {
        "imports": ("transvision.models.resilient_v2x.disconet_baseline",),
        "allow_failed_imports": False,
    }
    experiment = config["experiment"]
    assert experiment["method"] == (
        "DiscoNet-style cell-wise matrix-valued edge-attention L+C adaptation"
    )
    assert experiment["claim_status"] == (
        "controlled adaptation; not an exact source-paper reproduction"
    )


def test_shape_backprop_and_unsupported_branches_are_unobservable() -> None:
    torch.manual_seed(41)
    branches, support, ages = _inputs()
    module = DiscoNetAdaptedFusion(
        channels=4,
        edge_hidden_channels=4,
    ).eval()

    expected = module(branches, support, ages)
    poisoned = branches.detach().clone()
    poisoned[1, 1] = float("nan")
    poisoned[1, 3] = float("inf")
    poisoned_ages = ages.clone()
    poisoned_ages[1, 1] = float("-inf")
    poisoned_ages[1, 3] = float("nan")

    actual = module(poisoned, support, poisoned_ages)
    torch.testing.assert_close(actual, expected.detach())
    expected.square().mean().backward()

    assert expected.shape == (2, 4, 2, 3)
    assert torch.isfinite(expected).all()
    assert branches.grad is not None
    assert torch.isfinite(branches.grad).all()
    assert torch.count_nonzero(branches.grad[1, 0]).item() > 0
    assert torch.count_nonzero(branches.grad[1, 2]).item() > 0
    assert torch.count_nonzero(branches.grad[1, 1]).item() == 0
    assert torch.count_nonzero(branches.grad[1, 3]).item() == 0


def test_edge_encoder_produces_cell_wise_matrix_weights_normalized_by_sender() -> None:
    branches = torch.zeros(1, 4, 1, 1, 2)
    branches[:, 0, 0, 0] = torch.tensor((1.0, 1.0))
    branches[:, 1, 0, 0] = torch.tensor((0.0, 2.0))
    support = torch.tensor(((True, True, False, False),))
    ages = torch.tensor(((0.0, 0.0, float("nan"), float("inf")),))
    module = DiscoNetAdaptedFusion(
        channels=1,
        edge_hidden_channels=1,
        age_decay=0.0,
    ).eval()
    with torch.no_grad():
        convolutions = [
            layer for layer in module.edge_encoder if isinstance(layer, torch.nn.Conv2d)
        ]
        for convolution in convolutions:
            convolution.weight.fill_(1.0)
            if convolution.bias is not None:
                convolution.bias.zero_()

    weights = module.edge_weights(branches, support, ages)
    fused = module(branches, support, ages)

    assert weights.shape == (1, 2, 1, 2)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1, 1, 2))
    assert weights[0, 1, 0, 0] < 0.5
    assert weights[0, 1, 0, 1] > 0.5
    expected_fused = weights[0, 0, 0] + weights[0, 1, 0] * torch.tensor((0.0, 2.0))
    torch.testing.assert_close(fused[0, 0, 0], expected_fused)
    assert module.last_edge_weights is not None
    torch.testing.assert_close(module.last_edge_weights, weights)


def test_lidar_camera_aggregation_and_sender_age_penalty() -> None:
    branches = torch.tensor(
        [
            [
                [[[2.0]]],
                [[[10.0]]],
                [[[6.0]]],
                [[[14.0]]],
            ]
        ]
    )
    support = torch.ones(1, 4, dtype=torch.bool)
    ages = torch.tensor(((0.0, 3.0, 2.0, 1.0),))
    module = DiscoNetAdaptedFusion(
        channels=1,
        edge_hidden_channels=1,
        age_decay=1.0,
    ).eval()
    with torch.no_grad():
        for parameter in module.edge_encoder.parameters():
            parameter.zero_()

    clean = module._prepare_inputs(branches, support, ages)
    agents, agent_support, agent_ages = module._aggregate_agents(*clean)
    weights = module._edge_weights(agents, agent_support, agent_ages)

    ego_modality_weights = torch.softmax(torch.tensor((0.0, -2.0)), dim=0)
    rsu_modality_weights = torch.softmax(torch.tensor((-3.0, -1.0)), dim=0)
    torch.testing.assert_close(
        agents[0, 0, 0, 0, 0],
        2.0 * ego_modality_weights[0] + 6.0 * ego_modality_weights[1],
    )
    torch.testing.assert_close(
        agents[0, 1, 0, 0, 0],
        10.0 * rsu_modality_weights[0] + 14.0 * rsu_modality_weights[1],
    )
    expected_edges = torch.softmax(-agent_ages, dim=1)
    torch.testing.assert_close(weights[:, :, 0, 0], expected_edges)


def test_rsu_is_exact_fallback_when_ego_modalities_are_missing() -> None:
    branches = torch.tensor(
        [
            [
                [[[float("nan")]]],
                [[[3.0]]],
                [[[float("inf")]]],
                [[[5.0]]],
            ]
        ]
    )
    support = torch.tensor(((False, True, False, True),))
    ages = torch.tensor(((float("nan"), 0.0, float("inf"), 0.0),))
    module = DiscoNetAdaptedFusion(
        channels=1,
        edge_hidden_channels=1,
        age_decay=0.0,
    ).eval()

    fused = module(branches, support, ages)

    torch.testing.assert_close(fused, torch.tensor([[[[4.0]]]]))
    assert module.last_edge_weights is not None
    torch.testing.assert_close(
        module.last_edge_weights,
        torch.tensor([[[[0.0]], [[1.0]]]]),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"channels": 0}, "channels"),
        ({"edge_hidden_channels": 0}, "edge_hidden_channels"),
        ({"age_decay": -0.1}, "age_decay"),
    ),
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DiscoNetAdaptedFusion(**kwargs)


def test_all_missing_and_invalid_supported_metadata_are_rejected() -> None:
    module = DiscoNetAdaptedFusion(channels=2, edge_hidden_channels=2)
    branches = torch.zeros(1, 4, 2, 1, 1)
    support = torch.zeros(1, 4, dtype=torch.bool)
    ages = torch.full((1, 4), float("nan"))

    with pytest.raises(ValueError, match="all branches are missing"):
        module(branches, support, ages)

    support[0, 0] = True
    with pytest.raises(ValueError, match="finite and non-negative"):
        module(branches, support, ages)

    ages[0, 0] = 0.0
    branches[0, 0] = float("inf")
    with pytest.raises(ValueError, match="only finite values"):
        module(branches, support, ages)
