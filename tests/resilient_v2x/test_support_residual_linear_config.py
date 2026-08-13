from __future__ import annotations

import copy
import runpy

import torch

from transvision.models.resilient_v2x import (
    BEVGridSpec,
    ResilientV2XFeatureFusion,
)

from .test_configs import CONFIG_ROOT, MAIN, _load_config


CANDIDATE = CONFIG_ROOT / "improvements" / "support_residual_linear.py"
RESIDUAL_FORMULA = "router_fused + weight * (support_weighted_mean - router_fused)"


def _changed_keys(left: dict[str, object], right: dict[str, object]) -> set[str]:
    return {
        key for key in left.keys() | right.keys() if left.get(key) != right.get(key)
    }


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
        delta_t_ms=model["delta_t_ms"],
    )


def test_support_residual_linear_source_has_one_minimal_base() -> None:
    source = runpy.run_path(str(CANDIDATE))

    assert source["_base_"] == ["../dair_resilient_v2x.py"]
    assert source["model"] == {
        "ptf_mode": "linear",
        "support_residual_weight": 0.5,
    }
    assert {key for key in source if not key.startswith("_")} == {
        "experiment",
        "implementation_choices_runtime",
        "model",
    }


def test_support_residual_linear_changes_only_requested_model_behavior() -> None:
    main = _load_config(MAIN)
    candidate = _load_config(CANDIDATE)
    main_model = main["model"]
    candidate_model = candidate["model"]
    assert isinstance(main_model, dict)
    assert isinstance(candidate_model, dict)

    assert _changed_keys(main_model, candidate_model) == {
        "ptf_mode",
        "support_residual_weight",
    }
    assert candidate_model["ptf_mode"] == "linear"
    assert candidate_model["support_residual_weight"] == 0.5

    assert candidate_model["teacher"] == main_model["teacher"]
    assert candidate_model["teacher"]["ptf_mode"] == "nonlinear"
    assert "support_residual_weight" not in candidate_model["teacher"]
    assert candidate_model["teacher_checkpoint"] == main_model["teacher_checkpoint"]
    assert candidate_model["distillation"] == main_model["distillation"]

    excluded = {"experiment", "implementation_choices_runtime", "model"}
    assert {key: value for key, value in candidate.items() if key not in excluded} == {
        key: value for key, value in main.items() if key not in excluded
    }

    expected_choices = copy.deepcopy(main["implementation_choices_runtime"])
    expected_choices.update(
        support_residual_weight=0.5,
        support_residual_formula=RESIDUAL_FORMULA,
    )
    assert candidate["implementation_choices_runtime"] == expected_choices


def test_support_residual_linear_preserves_training_and_evaluation_protocol() -> None:
    main = _load_config(MAIN)
    candidate = _load_config(CANDIDATE)

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
        assert candidate[key] == main[key]

    assert candidate["train_cfg"] == {
        "type": "EpochBasedTrainLoop",
        "max_epochs": 50,
        "val_interval": 10,
    }
    assert candidate["randomness"]["seed"] == 20250218
    assert candidate["train_dataloader"]["dataset"]["include_clean_teacher"] is True
    contract = candidate["paper_data_contract"]
    assert (
        len(contract["evaluation_delays_ms"]) * len(contract["fault_conditions"]) == 12
    )


def test_support_residual_linear_keeps_fusion_schema_and_seeded_initialization() -> (
    None
):
    main_model = _load_config(MAIN)["model"]
    candidate_model = _load_config(CANDIDATE)["model"]
    assert isinstance(main_model, dict)
    assert isinstance(candidate_model, dict)

    torch.manual_seed(20250218)
    main_fusion = _fusion(main_model)
    torch.manual_seed(20250218)
    candidate_fusion = _fusion(candidate_model)

    main_state = main_fusion.state_dict()
    candidate_state = candidate_fusion.state_dict()
    assert tuple(main_state) == tuple(candidate_state)
    assert {name: tuple(tensor.shape) for name, tensor in main_state.items()} == {
        name: tuple(tensor.shape) for name, tensor in candidate_state.items()
    }
    assert sum(parameter.numel() for parameter in main_fusion.parameters()) == sum(
        parameter.numel() for parameter in candidate_fusion.parameters()
    )
    for name, tensor in main_state.items():
        assert torch.equal(tensor, candidate_state[name])

    assert candidate_fusion.ptf_mode == "linear"
    assert candidate_fusion.router.support_residual_weight == 0.5
