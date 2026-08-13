from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

pytest.importorskip("mmdet3d")

from mmdet3d.registry import MODELS

from transvision.models.resilient_v2x.late_fusion_baseline import (
    LateFusionStyleFusion,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "resilient_v2x" / "baselines" / "late_fusion.py"
METHOD_LABEL = (
    "LateFusion-style L+C (controlled late-feature adaptation; not exact "
    "source-paper reproduction)"
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branches = torch.randn(3, 4, 4, 2, 3, requires_grad=True)
    support = torch.tensor(
        (
            (True, True, True, True),
            (True, False, True, False),
            (False, True, False, True),
        )
    )
    ages = torch.tensor(
        (
            (0.0, 2.0, 1.0, 3.0),
            (1.0, float("nan"), 2.0, float("inf")),
            (float("nan"), 2.0, float("inf"), 1.0),
        )
    )
    return branches, support, ages


def test_late_fusion_is_registered_and_config_claim_is_unambiguous() -> None:
    assert MODELS.get("LateFusionStyleFusion") is LateFusionStyleFusion
    built = MODELS.build(dict(type="LateFusionStyleFusion", channels=4, age_decay=0.25))
    assert isinstance(built, LateFusionStyleFusion)

    config = runpy.run_path(str(CONFIG))
    assert config["_base_"] == ["./_base_.py"]
    assert config["model"] == {
        "baseline_name": "late_fusion",
        "baseline_cfg": {"_delete_": True, "age_decay": 0.25},
    }
    assert config["experiment"]["method"] == METHOD_LABEL
    assert config["experiment"]["claim_status"] == (
        "controlled late-feature adaptation; not exact source-paper reproduction"
    )


def test_modalities_are_fused_inside_each_agent_before_terminal_gate() -> None:
    module = LateFusionStyleFusion(channels=1, age_decay=0.0)
    branches = torch.tensor([[[[[2.0]]], [[[50.0]]], [[[6.0]]], [[[10.0]]]]])
    support = torch.tensor(((True, False, True, True),))
    ages = torch.tensor(((0.0, float("nan"), 0.0, 2.0),))

    clean = module._prepare_inputs(branches, support, ages)
    agents, agent_support, agent_ages = module._merge_modalities(*clean)

    torch.testing.assert_close(agents[:, 0], torch.tensor([[[[4.0]]]]))
    torch.testing.assert_close(agents[:, 1], torch.tensor([[[[10.0]]]]))
    assert agent_support.tolist() == [[True, True]]
    torch.testing.assert_close(agent_ages, torch.tensor(((0.0, 2.0),)))


def test_terminal_gate_is_support_aware_age_aware_and_normalized() -> None:
    module = LateFusionStyleFusion(channels=1, age_decay=1.0).eval()
    with torch.no_grad():
        module.gate_projection.weight.zero_()
        module.gate_projection.bias.zero_()
        module.agent_bias.zero_()

    branches = torch.ones(2, 4, 1, 1, 2)
    support = torch.tensor(
        (
            (True, True, True, True),
            (False, True, False, True),
        )
    )
    ages = torch.tensor(
        (
            (0.0, 4.0, 0.0, 4.0),
            (float("nan"), 0.0, float("inf"), 0.0),
        )
    )

    weights = module.gate_weights(branches, support, ages)
    fused = module(branches, support, ages)

    assert weights.shape == (2, 2, 1, 2)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, 1, 2))
    assert (weights[0, 0] > weights[0, 1]).all()
    torch.testing.assert_close(weights[1, 0], torch.zeros(1, 2))
    torch.testing.assert_close(weights[1, 1], torch.ones(1, 2))
    torch.testing.assert_close(fused, torch.ones(2, 1, 1, 2))
    assert module.last_gate_weights is not None
    torch.testing.assert_close(module.last_gate_weights, weights)


def test_missing_branches_cannot_pollute_output_gate_or_gradients() -> None:
    torch.manual_seed(47)
    branches, support, ages = _inputs()
    module = LateFusionStyleFusion(channels=4, age_decay=0.25).eval()

    expected = module(branches, support, ages)
    expected_weights = module.gate_weights(branches, support, ages)
    poisoned = branches.detach().clone()
    poisoned[1, (1, 3)] = float("nan")
    poisoned[2, (0, 2)] = float("inf")

    torch.testing.assert_close(module(poisoned, support, ages), expected.detach())
    torch.testing.assert_close(
        module.gate_weights(poisoned, support, ages),
        expected_weights.detach(),
    )

    expected.square().mean().backward()
    assert expected.shape == (3, 4, 2, 3)
    assert torch.isfinite(expected).all()
    assert branches.grad is not None
    assert torch.isfinite(branches.grad).all()
    assert torch.count_nonzero(branches.grad[1, (1, 3)]).item() == 0
    assert torch.count_nonzero(branches.grad[2, (0, 2)]).item() == 0
    assert torch.count_nonzero(branches.grad[1, (0, 2)]).item() > 0
    assert torch.count_nonzero(branches.grad[2, (1, 3)]).item() > 0


def test_single_supported_agent_is_an_exact_non_droppable_fallback() -> None:
    module = LateFusionStyleFusion(channels=1, age_decay=0.5).eval()
    branches = torch.tensor(
        [
            [[[[2.0]]], [[[float("nan")]]], [[[6.0]]], [[[float("inf")]]]],
            [[[[float("nan")]]], [[[3.0]]], [[[float("inf")]]], [[[5.0]]]],
        ]
    )
    support = torch.tensor(
        (
            (True, False, True, False),
            (False, True, False, True),
        )
    )
    ages = torch.tensor(
        (
            (0.0, float("nan"), 0.0, float("inf")),
            (float("nan"), 0.0, float("inf"), 0.0),
        )
    )

    fused = module(branches, support, ages)
    weights = module.last_gate_weights

    torch.testing.assert_close(fused[:, 0, 0, 0], torch.tensor((4.0, 4.0)))
    assert weights is not None
    torch.testing.assert_close(
        weights[:, :, 0, 0],
        torch.tensor(((1.0, 0.0), (0.0, 1.0))),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"channels": 0}, "positive integer"),
        ({"age_decay": -0.1}, "non-negative"),
        ({"age_decay": float("inf")}, "finite"),
    ),
)
def test_constructor_rejects_invalid_options(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LateFusionStyleFusion(**kwargs)


def test_input_contract_rejects_all_missing_and_bad_supported_values() -> None:
    module = LateFusionStyleFusion(channels=2)
    branches = torch.zeros(2, 4, 2, 1, 1)
    support = torch.tensor(((True, False, False, False), (False, False, False, False)))
    ages = torch.zeros(2, 4)

    with pytest.raises(
        ValueError,
        match=r"all branches are missing for sample indices \[1\]",
    ):
        module(branches, support, ages)

    support[1, 1] = True
    ages[1, 1] = float("nan")
    with pytest.raises(ValueError, match="finite and non-negative"):
        module(branches, support, ages)

    ages[1, 1] = 0.0
    branches[1, 1] = float("inf")
    with pytest.raises(ValueError, match="only finite values"):
        module(branches, support, ages)
