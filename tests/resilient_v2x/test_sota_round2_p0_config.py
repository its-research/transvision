from __future__ import annotations

import runpy
from pathlib import Path

import torch

from transvision.models.resilient_v2x import (
    BEVGridSpec,
    ResilientV2XFeatureFusion,
)

from .test_configs import CONFIG_ROOT, MAIN, _load_config


P0 = (
    CONFIG_ROOT
    / "improvements"
    / "support_residual_no_reliability_linear.py"
)
P0_BASE = (
    CONFIG_ROOT / "improvements" / "support_residual_no_reliability.py"
)
P0_IDENTITY = "dair_improvement_support_residual_no_reliability_linear"


def _changed_keys(left: dict[str, object], right: dict[str, object]) -> set[str]:
    return {
        key for key in left.keys() | right.keys() if left.get(key) != right.get(key)
    }


def _model(path: Path) -> dict[str, object]:
    model = _load_config(path)["model"]
    assert isinstance(model, dict)
    return model


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


def test_p0_source_is_a_minimal_single_parameter_overlay() -> None:
    source = runpy.run_path(str(P0))

    assert source["_base_"] == ["./support_residual_no_reliability.py"]
    assert source["model"] == {"ptf_mode": "linear"}
    assert source["experiment"]["name"] == P0_IDENTITY
    assert {key for key in source if not key.startswith("_")} == {
        "experiment",
        "model",
    }


def test_p0_changes_only_ptf_mode_from_its_exact_base() -> None:
    base = _load_config(P0_BASE)
    candidate = _load_config(P0)
    base_model = base["model"]
    candidate_model = candidate["model"]
    assert isinstance(base_model, dict)
    assert isinstance(candidate_model, dict)

    assert _changed_keys(base_model, candidate_model) == {"ptf_mode"}
    assert base_model["ptf_mode"] == "nonlinear"
    assert candidate_model["ptf_mode"] == "linear"
    excluded = {"experiment", "model"}
    assert {key: value for key, value in candidate.items() if key not in excluded} == {
        key: value for key, value in base.items() if key not in excluded
    }


def test_p0_is_the_exact_three_factor_intersection() -> None:
    main_model = _model(MAIN)
    candidate_model = _model(P0)

    assert _changed_keys(main_model, candidate_model) == {
        "ptf_mode",
        "support_residual_weight",
        "use_reliability",
    }
    assert candidate_model["ptf_mode"] == "linear"
    assert candidate_model["support_residual_weight"] == 0.5
    assert candidate_model["use_reliability"] is False


def test_p0_has_no_resolved_model_alias() -> None:
    candidate_model = _model(P0)
    aliases = []
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        if path == P0:
            continue
        if _load_config(path).get("model") == candidate_model:
            aliases.append(path.relative_to(CONFIG_ROOT).as_posix())

    assert aliases == []


def test_p0_preserves_teacher_distillation_seed_and_training_protocol() -> None:
    main = _load_config(MAIN)
    candidate = _load_config(P0)

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

    candidate_model = candidate["model"]
    main_model = main["model"]
    assert isinstance(candidate_model, dict)
    assert isinstance(main_model, dict)
    assert candidate_model["teacher"] == main_model["teacher"]
    assert candidate_model["teacher_checkpoint"] == main_model["teacher_checkpoint"]
    assert candidate_model["distillation"] == main_model["distillation"]
    assert candidate["train_cfg"] == {
        "type": "EpochBasedTrainLoop",
        "max_epochs": 50,
        "val_interval": 10,
    }
    assert candidate["randomness"]["seed"] == 20250218
    assert candidate["train_dataloader"]["dataset"]["seed"] == 20250218
    assert candidate["train_dataloader"]["dataset"]["include_clean_teacher"] is True


def test_p0_keeps_seeded_fusion_parameter_schema() -> None:
    main_model = _model(MAIN)
    candidate_model = _model(P0)

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
