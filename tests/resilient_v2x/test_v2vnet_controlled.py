from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

pytest.importorskip("mmdet3d")

from mmdet3d.registry import MODELS

from transvision.models.resilient_v2x.v2vnet_baseline import V2VNetStyleFusion


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "resilient_v2x" / "baselines" / "v2vnet.py"


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branches = torch.randn(2, 4, 4, 3, 5, requires_grad=True)
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


def test_v2vnet_style_fusion_is_registered_and_has_common_interface() -> None:
    assert MODELS.get("V2VNetStyleFusion") is V2VNetStyleFusion
    module = MODELS.build(
        dict(
            type="V2VNetStyleFusion",
            channels=4,
            age_decay=0.25,
            message_iterations=2,
            kernel_size=3,
        )
    )
    branches, support, ages = _inputs()

    fused = module(branches, support, ages)
    fused.square().mean().backward()

    assert fused.shape == (2, 4, 3, 5)
    assert torch.isfinite(fused).all()
    assert branches.grad is not None
    assert torch.isfinite(branches.grad).all()
    assert torch.count_nonzero(branches.grad[0, 1]).item() > 0
    assert torch.count_nonzero(branches.grad[0, 3]).item() > 0
    assert torch.count_nonzero(branches.grad[1, 1]).item() == 0
    assert torch.count_nonzero(branches.grad[1, 3]).item() == 0


def test_lidar_camera_are_merged_into_ego_and_rsu_before_messages() -> None:
    module = V2VNetStyleFusion(
        channels=1,
        age_decay=0.0,
        message_iterations=1,
        kernel_size=1,
    )
    branches = torch.tensor([[[[[2.0]]], [[[99.0]]], [[[6.0]]], [[[8.0]]]]])
    support = torch.tensor(((True, False, True, True),))
    ages = torch.tensor(((0.0, float("nan"), 4.0, 2.0),))
    clean = module._prepare_inputs(branches, support, ages)

    nodes, agent_support = module._merge_modalities(*clean)

    assert agent_support.tolist() == [[True, True]]
    torch.testing.assert_close(nodes[:, 0], torch.tensor([[[[4.0]]]]))
    torch.testing.assert_close(nodes[:, 1], torch.tensor([[[[8.0]]]]))


def test_unsupported_values_ages_and_gradients_are_strictly_masked() -> None:
    torch.manual_seed(31)
    module = V2VNetStyleFusion(
        channels=4,
        message_iterations=2,
        kernel_size=3,
    ).eval()
    branches, support, ages = _inputs()
    clean_output = module(branches, support, ages)
    poisoned = branches.detach().clone()
    poisoned[1, 1] = float("nan")
    poisoned[1, 3] = float("inf")
    poisoned_ages = ages.clone()
    poisoned_ages[1, 1] = float("-inf")
    poisoned_ages[1, 3] = float("nan")

    poisoned_output = module(poisoned, support, poisoned_ages)

    torch.testing.assert_close(poisoned_output, clean_output.detach())


def test_missing_peer_cannot_trigger_gru_bias_or_change_the_lone_node() -> None:
    torch.manual_seed(37)
    module = V2VNetStyleFusion(
        channels=2,
        age_decay=0.0,
        message_iterations=4,
        kernel_size=3,
    ).eval()
    branches = torch.randn(2, 4, 2, 2, 3)
    support = torch.tensor(
        (
            (True, False, True, False),
            (False, True, False, True),
        )
    )
    ages = torch.tensor(
        (
            (0.0, float("nan"), 0.0, float("nan")),
            (float("nan"), 0.0, float("nan"), 0.0),
        )
    )
    clean = module._prepare_inputs(branches, support, ages)
    nodes, agent_support = module._merge_modalities(*clean)

    states = module._message_pass(nodes, agent_support)
    output = module(branches, support, ages)

    torch.testing.assert_close(states, nodes)
    torch.testing.assert_close(output[0], nodes[0, 0])
    torch.testing.assert_close(output[1], nodes[1, 1])
    assert torch.count_nonzero(states[0, 1]).item() == 0
    assert torch.count_nonzero(states[1, 0]).item() == 0


def test_age_weighting_is_applied_within_each_agent_only() -> None:
    module = V2VNetStyleFusion(
        channels=1,
        age_decay=1.0,
        message_iterations=1,
        kernel_size=1,
    )
    branches = torch.tensor([[[[[2.0]]], [[[10.0]]], [[[6.0]]], [[[14.0]]]]])
    support = torch.ones(1, 4, dtype=torch.bool)
    ages = torch.tensor(((0.0, 3.0, 2.0, 1.0),))
    clean = module._prepare_inputs(branches, support, ages)

    nodes, _ = module._merge_modalities(*clean)

    ego_weights = torch.softmax(torch.tensor((0.0, -2.0)), dim=0)
    rsu_weights = torch.softmax(torch.tensor((-3.0, -1.0)), dim=0)
    torch.testing.assert_close(
        nodes[0, 0, 0, 0, 0],
        2.0 * ego_weights[0] + 6.0 * ego_weights[1],
    )
    torch.testing.assert_close(
        nodes[0, 1, 0, 0, 0],
        10.0 * rsu_weights[0] + 14.0 * rsu_weights[1],
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"channels": 0}, "channels"),
        ({"message_iterations": 0}, "message_iterations"),
        ({"kernel_size": 2}, "kernel_size must be odd"),
        ({"age_decay": -1.0}, "age_decay"),
    ),
)
def test_constructor_rejects_invalid_options(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        V2VNetStyleFusion(**kwargs)


def test_input_contract_rejects_all_missing_and_bad_supported_metadata() -> None:
    module = V2VNetStyleFusion(channels=2)
    branches = torch.zeros(1, 4, 2, 1, 1)
    support = torch.zeros(1, 4, dtype=torch.bool)
    ages = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="all branches are missing"):
        module(branches, support, ages)

    support[0, 0] = True
    ages[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite and non-negative"):
        module(branches, support, ages)

    ages[0, 0] = 0.0
    branches[0, 0] = float("inf")
    with pytest.raises(ValueError, match="only finite values"):
        module(branches, support, ages)


def test_config_labels_result_as_controlled_style_adaptation() -> None:
    config = runpy.run_path(str(CONFIG))

    assert config["_base_"] == ["./_base_.py"]
    assert config["model"] == {
        "baseline_name": "v2vnet",
        "baseline_cfg": {
            "_delete_": True,
            "age_decay": 0.25,
            "message_iterations": 3,
            "kernel_size": 3,
        },
    }
    experiment = config["experiment"]
    assert experiment["method"] == (
        "V2VNet-style two-agent ConvGRU message-passing L+C adaptation"
    )
    assert experiment["claim_status"] == (
        "controlled adaptation; not an exact source-paper reproduction"
    )
