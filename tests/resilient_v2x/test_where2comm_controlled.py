from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

pytest.importorskip("mmdet3d")

from mmdet3d.registry import MODELS

from transvision.models.resilient_v2x.where2comm_baseline import (
    Where2commAdaptedFusion,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "resilient_v2x" / "baselines" / "where2comm.py"


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


def test_where2comm_is_registered_and_config_declares_clean_room_adaptation() -> None:
    assert MODELS.get("Where2commAdaptedFusion") is Where2commAdaptedFusion
    registered = MODELS.build(
        dict(
            type="Where2commAdaptedFusion",
            channels=4,
            communication_threshold=0.5,
            age_decay=0.25,
        )
    )
    assert isinstance(registered, Where2commAdaptedFusion)
    config = runpy.run_path(str(CONFIG))

    assert config["model"] == {
        "baseline_name": "where2comm",
        "baseline_cfg": {
            "_delete_": True,
            "communication_threshold": 0.5,
            "age_decay": 0.25,
        },
    }
    assert "clean-room Where2comm-style" in config["experiment"]["method"]
    assert config["experiment"]["claim_status"] == (
        "controlled adaptation; not an exact source-paper reproduction"
    )


def test_where2comm_shape_backprop_and_unsupported_values_are_unobservable() -> None:
    torch.manual_seed(17)
    branches, support, ages = _inputs()
    module = Where2commAdaptedFusion(
        channels=4,
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


def test_spatial_confidence_mask_and_ratio_are_exposed() -> None:
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
    module = Where2commAdaptedFusion(
        channels=1,
        communication_threshold=0.5,
        age_decay=0.0,
    ).eval()
    with torch.no_grad():
        module.confidence_head.weight.fill_(1.0)
        module.confidence_head.bias.zero_()
        module.agent_bias.zero_()

    fused = module(branches, support, ages)
    mask = module.last_communication_mask
    ratio = module.last_communication_ratio

    assert fused.shape == (2, 1, 1, 4)
    assert mask is not None and mask.dtype is torch.bool
    assert mask.shape == (2, 2, 1, 4)
    assert mask[0, 0].all()
    assert mask[0, 1, 0].tolist() == [False, True, False, True]
    assert mask[1, 0].all()
    assert not mask[1, 1].any()
    assert ratio is not None and ratio.shape == (2,)
    torch.testing.assert_close(ratio, torch.tensor((0.5, 0.0)))


def test_agent_aggregation_is_support_and_age_aware_before_communication() -> None:
    branches = torch.tensor(
        [[
            [[[2.0]]],
            [[[50.0]]],
            [[[6.0]]],
            [[[10.0]]],
        ]]
    )
    support = torch.tensor(((True, False, True, True),))
    ages = torch.tensor(((0.0, float("nan"), 0.0, 0.0),))
    module = Where2commAdaptedFusion(
        channels=1,
        communication_threshold=0.0,
        age_decay=0.0,
    )

    clean, clean_support, clean_ages = module._prepare_inputs(
        branches,
        support,
        ages,
    )
    agents, agent_support, agent_ages = module._aggregate_agents(
        clean,
        clean_support,
        clean_ages,
    )

    torch.testing.assert_close(agents[:, 0], torch.tensor([[[[4.0]]]]))
    torch.testing.assert_close(agents[:, 1], torch.tensor([[[[10.0]]]]))
    assert agent_support.tolist() == [[True, True]]
    torch.testing.assert_close(agent_ages, torch.zeros(1, 2))


def test_rsu_is_non_droppable_fallback_when_both_ego_modalities_are_missing() -> None:
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
    module = Where2commAdaptedFusion(
        channels=1,
        communication_threshold=1.0,
        age_decay=0.0,
    ).eval()

    fused = module(branches, support, ages)

    torch.testing.assert_close(fused, torch.tensor([[[[4.0]]]]))
    assert module.last_communication_mask is not None
    assert module.last_communication_mask.tolist() == [[[[False]], [[True]]]]
    assert module.last_communication_ratio is not None
    torch.testing.assert_close(
        module.last_communication_ratio,
        torch.ones(1),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"channels": 0}, "positive integer"),
        ({"communication_threshold": 1.1}, r"in \[0, 1\]"),
        ({"age_decay": -0.1}, "non-negative"),
    ),
)
def test_where2comm_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Where2commAdaptedFusion(**kwargs)


def test_where2comm_rejects_all_missing_sample() -> None:
    branches = torch.zeros(1, 4, 2, 1, 1)
    support = torch.zeros(1, 4, dtype=torch.bool)
    ages = torch.full((1, 4), float("nan"))

    with pytest.raises(ValueError, match="all branches are missing"):
        Where2commAdaptedFusion(channels=2)(branches, support, ages)
