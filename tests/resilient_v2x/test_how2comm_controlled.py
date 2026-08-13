from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

pytest.importorskip("mmdet3d")

from mmdet3d.registry import MODELS

from transvision.models.resilient_v2x.how2comm_baseline import (
    How2commAdaptedFusion,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "resilient_v2x" / "baselines" / "how2comm.py"


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
            (1.0, float("nan"), 2.0, float("nan")),
        )
    )
    return branches, support, ages


def test_how2comm_is_registered_and_config_declares_controlled_adaptation() -> None:
    assert MODELS.get("How2commAdaptedFusion") is How2commAdaptedFusion
    built = MODELS.build(
        dict(
            type="How2commAdaptedFusion",
            channels=4,
            selection_reduction=2,
        )
    )
    assert isinstance(built, How2commAdaptedFusion)

    config = runpy.run_path(str(CONFIG))
    assert config["model"] == {
        "baseline_name": "how2comm",
        "baseline_cfg": {
            "_delete_": True,
            "selection_reduction": 4,
            "communication_threshold": 0.2,
            "age_decay": 0.25,
            "temporal_gain": 0.5,
        },
    }
    assert "clean-room How2comm-style" in config["experiment"]["method"]
    assert config["experiment"]["claim_status"] == (
        "controlled adaptation; not an exact source-paper reproduction"
    )


def test_four_branch_contract_backprop_and_poisoned_unsupported_invariance() -> None:
    torch.manual_seed(31)
    branches, support, ages = _inputs()
    module = How2commAdaptedFusion(
        channels=4,
        selection_reduction=2,
        communication_threshold=0.0,
    ).eval()

    expected = module(branches, support, ages)
    poisoned = branches.detach().clone()
    poisoned[1, 1] = float("nan")
    poisoned[1, 3] = float("inf")
    actual = module(poisoned, support, ages)
    torch.testing.assert_close(actual, expected.detach())

    expected.square().mean().backward()
    assert expected.shape == (2, 4, 2, 3)
    assert torch.isfinite(expected).all()
    assert branches.grad is not None
    assert torch.isfinite(branches.grad).all()
    assert torch.count_nonzero(branches.grad[1, 1]).item() == 0
    assert torch.count_nonzero(branches.grad[1, 3]).item() == 0
    assert torch.count_nonzero(branches.grad[1, 0]).item() > 0
    assert torch.count_nonzero(branches.grad[1, 2]).item() > 0


def test_spatial_channel_message_selection_and_ratio_are_exposed() -> None:
    branches = torch.zeros(2, 4, 1, 1, 4)
    branches[0, 1, 0, 0] = torch.tensor((-2.0, 2.0, -3.0, 3.0))
    support = torch.tensor(
        (
            (True, True, False, False),
            (True, False, True, False),
        )
    )
    ages = torch.tensor(
        (
            (0.0, 0.0, float("nan"), float("nan")),
            (0.0, float("nan"), 0.0, float("nan")),
        )
    )
    module = How2commAdaptedFusion(
        channels=1,
        selection_reduction=1,
        communication_threshold=0.2,
        age_decay=0.0,
        temporal_gain=0.0,
    ).eval()
    with torch.no_grad():
        module.spatial_attention.weight.fill_(1.0)
        module.spatial_attention.bias.zero_()
        for parameter in module.channel_attention.parameters():
            parameter.zero_()
        module.fusion_gate.weight.zero_()
        module.fusion_gate.bias.zero_()

    fused = module(branches, support, ages)
    spatial_mask = module.last_spatial_mask
    channel_mask = module.last_channel_mask
    ratio = module.last_communication_ratio

    assert fused.shape == (2, 1, 1, 4)
    assert spatial_mask is not None and spatial_mask.dtype is torch.bool
    assert spatial_mask.shape == (2, 1, 4)
    assert spatial_mask[0, 0].tolist() == [False, True, False, True]
    assert not spatial_mask[1].any()
    assert channel_mask is not None and channel_mask.tolist() == [[True], [False]]
    assert ratio is not None
    torch.testing.assert_close(ratio, torch.tensor((0.5, 0.0)))


def test_age_conditions_temporal_context_and_suppresses_stale_communication() -> None:
    module = How2commAdaptedFusion(
        channels=1,
        selection_reduction=1,
        communication_threshold=0.2,
        age_decay=1.0,
        temporal_gain=1.0,
    ).eval()
    with torch.no_grad():
        module.temporal_depthwise.weight.zero_()
        module.temporal_depthwise.weight[:, :, 1, 1] = 1.0
        module.temporal_projection.weight.fill_(1.0)
        module.spatial_attention.weight.zero_()
        module.spatial_attention.bias.zero_()
        for parameter in module.channel_attention.parameters():
            parameter.zero_()

    remote = torch.ones(2, 1, 1, 1)
    corrected = module._temporal_context(
        remote,
        torch.ones(2, dtype=torch.bool),
        torch.tensor((0.0, 2.0)),
    )
    torch.testing.assert_close(corrected[0], remote[0])
    assert corrected[1].item() > remote[1].item()

    branches = torch.ones(2, 4, 1, 1, 1)
    support = torch.ones(2, 4, dtype=torch.bool)
    ages = torch.tensor(
        (
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 4.0, 0.0, 4.0),
        )
    )
    module(branches, support, ages)
    assert module.last_spatial_mask is not None
    assert module.last_channel_mask is not None
    assert module.last_spatial_mask[0].all()
    assert module.last_channel_mask[0].all()
    assert not module.last_spatial_mask[1].any()
    assert not module.last_channel_mask[1].any()


def test_rsu_only_is_a_noncommunicating_local_fallback() -> None:
    branches = torch.tensor(
        [[
            [[[float("nan")]]],
            [[[3.0]]],
            [[[float("inf")]]],
            [[[5.0]]],
        ]]
    )
    support = torch.tensor(((False, True, False, True),))
    ages = torch.tensor(((float("nan"), 0.0, float("nan"), 0.0),))
    module = How2commAdaptedFusion(
        channels=1,
        selection_reduction=1,
        communication_threshold=0.0,
        age_decay=0.0,
        temporal_gain=0.0,
    ).eval()

    fused = module(branches, support, ages)

    torch.testing.assert_close(fused, torch.tensor([[[[4.0]]]]))
    assert module.last_communication_ratio is not None
    torch.testing.assert_close(module.last_communication_ratio, torch.zeros(1))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"channels": 0}, "positive integer"),
        ({"selection_reduction": 0}, "positive integer"),
        ({"communication_threshold": 1.1}, r"in \[0, 1\]"),
        ({"age_decay": -0.1}, "non-negative"),
        ({"temporal_gain": float("inf")}, "non-negative"),
    ),
)
def test_how2comm_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        How2commAdaptedFusion(**kwargs)


def test_how2comm_rejects_all_missing_sample() -> None:
    branches = torch.zeros(1, 4, 2, 1, 1)
    support = torch.zeros(1, 4, dtype=torch.bool)
    ages = torch.full((1, 4), float("nan"))

    with pytest.raises(ValueError, match="all branches are missing"):
        How2commAdaptedFusion(channels=2)(branches, support, ages)
