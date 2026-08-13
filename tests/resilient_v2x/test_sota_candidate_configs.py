from __future__ import annotations

import ast
import copy
import runpy
from pathlib import Path

import pytest
import torch

from transvision.models.resilient_v2x import (
    BEVGridSpec,
    ResilientV2XFeatureFusion,
)

from .test_configs import CONFIG_ROOT, MAIN, ROOT, _load_config


E1 = CONFIG_ROOT / "improvements" / "support_residual_linear.py"
E2 = CONFIG_ROOT / "improvements" / "no_reliability_linear.py"
E3 = CONFIG_ROOT / "improvements" / "support_residual_no_reliability.py"
NO_RELIABILITY = CONFIG_ROOT / "ablations" / "no_reliability.py"
PTF_LINEAR = CONFIG_ROOT / "ablations" / "ptf_linear.py"
SUPPORT_RESIDUAL = CONFIG_ROOT / "improvements" / "support_residual.py"
BOOTSTRAP = ROOT / "tools" / "resilient_v2x" / "clearml_5090_bootstrap.py"
RESIDUAL_FORMULA = "router_fused + weight * (support_weighted_mean - router_fused)"


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


def _literal_assignment(path: Path, name: str) -> object:
    tree = ast.parse(path.read_text(), filename=str(path))
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(matches) == 1, f"expected one assignment for {name}"
    return ast.literal_eval(matches[0])


@pytest.mark.parametrize(
    ("path", "base", "source_model", "identity"),
    (
        (
            E2,
            "../ablations/no_reliability.py",
            {"ptf_mode": "linear"},
            "dair_improvement_no_reliability_linear",
        ),
        (
            E3,
            "./support_residual.py",
            {"use_reliability": False},
            "dair_improvement_support_residual_no_reliability",
        ),
    ),
)
def test_new_candidate_sources_are_minimal_single_variable_overlays(
    path: Path,
    base: str,
    source_model: dict[str, object],
    identity: str,
) -> None:
    source = runpy.run_path(str(path))

    assert source["_base_"] == [base]
    assert source["model"] == source_model
    assert source["experiment"]["name"] == identity
    assert {key for key in source if not key.startswith("_")} == {
        "experiment",
        "model",
    }


@pytest.mark.parametrize(
    ("candidate_path", "base_path", "parameter", "value"),
    (
        (E2, NO_RELIABILITY, "ptf_mode", "linear"),
        (E3, SUPPORT_RESIDUAL, "use_reliability", False),
    ),
)
def test_new_candidates_change_exactly_one_model_parameter_from_their_base(
    candidate_path: Path,
    base_path: Path,
    parameter: str,
    value: object,
) -> None:
    base = _load_config(base_path)
    candidate = _load_config(candidate_path)
    base_model = base["model"]
    candidate_model = candidate["model"]
    assert isinstance(base_model, dict)
    assert isinstance(candidate_model, dict)

    assert _changed_keys(base_model, candidate_model) == {parameter}
    assert candidate_model[parameter] == value

    excluded = {"experiment", "model"}
    assert {key: value for key, value in candidate.items() if key not in excluded} == {
        key: value for key, value in base.items() if key not in excluded
    }
    assert candidate_model["teacher"] == base_model["teacher"]
    assert candidate_model["teacher_checkpoint"] == base_model["teacher_checkpoint"]
    assert candidate_model["distillation"] == base_model["distillation"]


def test_three_candidates_have_distinct_non_alias_model_identities() -> None:
    main_model = _model(MAIN)
    e1_model = _model(E1)
    e2_model = _model(E2)
    e3_model = _model(E3)

    assert _changed_keys(main_model, e1_model) == {
        "ptf_mode",
        "support_residual_weight",
    }
    assert _changed_keys(main_model, e2_model) == {
        "ptf_mode",
        "use_reliability",
    }
    assert _changed_keys(main_model, e3_model) == {
        "support_residual_weight",
        "use_reliability",
    }
    assert e1_model["ptf_mode"] == "linear"
    assert e1_model["support_residual_weight"] == 0.5
    assert e2_model["ptf_mode"] == "linear"
    assert e2_model["use_reliability"] is False
    assert e3_model["use_reliability"] is False
    assert e3_model["support_residual_weight"] == 0.5

    assert e1_model != _model(PTF_LINEAR)
    assert e1_model != _model(SUPPORT_RESIDUAL)
    assert e2_model != _model(PTF_LINEAR)
    assert e2_model != _model(NO_RELIABILITY)
    assert e3_model != _model(NO_RELIABILITY)
    assert e3_model != _model(SUPPORT_RESIDUAL)
    assert len({repr(e1_model), repr(e2_model), repr(e3_model)}) == 3


@pytest.mark.parametrize("candidate_path", (E1, E2, E3))
def test_candidate_model_is_not_an_alias_of_any_other_resolved_config(
    candidate_path: Path,
) -> None:
    candidate_model = _model(candidate_path)
    aliases = []
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        if path == candidate_path:
            continue
        resolved = _load_config(path)
        if resolved.get("model") == candidate_model:
            aliases.append(path.relative_to(CONFIG_ROOT).as_posix())

    assert aliases == []


@pytest.mark.parametrize("candidate_path", (E1, E2, E3))
def test_candidates_preserve_teacher_and_fixed_training_protocol(
    candidate_path: Path,
) -> None:
    main = _load_config(MAIN)
    candidate = _load_config(candidate_path)

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


def test_sealed_four_gpu_launcher_contract_is_global_batch_eight() -> None:
    assert _literal_assignment(BOOTSTRAP, "EXPECTED_GPU_COUNT") == 4
    assert _literal_assignment(BOOTSTRAP, "EXPERIMENT_MAX_EPOCHS") == 50
    assert _literal_assignment(BOOTSTRAP, "RTX5090_TRAIN_BATCH_SIZE_PER_GPU") == 2
    assert _literal_assignment(BOOTSTRAP, "RTX5090_VAL_INTERVAL") == 10
    assert _literal_assignment(BOOTSTRAP, "DEFAULT_TRAINING_SEED") == 20250218
    assert _literal_assignment(BOOTSTRAP, "TRAINING_OVERLAY_PROTOCOL_SEED") == 20250218
    assert (
        _literal_assignment(BOOTSTRAP, "EXPECTED_GPU_COUNT")
        * _literal_assignment(BOOTSTRAP, "RTX5090_TRAIN_BATCH_SIZE_PER_GPU")
        == 8
    )


@pytest.mark.parametrize("candidate_path", (E2, E3))
def test_new_candidates_keep_seeded_fusion_parameter_schema(
    candidate_path: Path,
) -> None:
    main_fusion_model = _model(MAIN)
    candidate_model = _model(candidate_path)

    torch.manual_seed(20250218)
    main_fusion = _fusion(main_fusion_model)
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


def test_support_residual_no_reliability_preserves_residual_audit_metadata() -> None:
    support = _load_config(SUPPORT_RESIDUAL)
    candidate = _load_config(E3)
    expected_choices = copy.deepcopy(support["implementation_choices_runtime"])

    assert candidate["implementation_choices_runtime"] == expected_choices
    assert candidate["implementation_choices_runtime"]["support_residual_weight"] == 0.5
    assert (
        candidate["implementation_choices_runtime"]["support_residual_formula"]
        == RESIDUAL_FORMULA
    )
