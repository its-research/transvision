from __future__ import annotations

import ast
import runpy
from pathlib import Path

import torch

from transvision.models.resilient_v2x import (
    BEVGridSpec,
    ResilientV2XFeatureFusion,
)

from .test_configs import CONFIG_ROOT, ROOT, _load_config


P0 = CONFIG_ROOT / "improvements" / "support_residual_no_reliability_linear.py"
P2 = CONFIG_ROOT / "improvements" / "support_residual_no_reliability_linear_bbox25.py"
P2_IDENTITY = "dair_improvement_support_residual_no_reliability_linear_bbox25"
P0_DEPLOYMENT = ROOT / "tools" / "resilient_v2x" / "prepare_clearml_sota_round2_p0.py"
STUDENT_BBOX_WEIGHT_PATH = ("bbox_head", "loss_bbox", "loss_weight")
_MISSING = object()


def _leaf_differences(
    left: object,
    right: object,
    path: tuple[str, ...] = (),
) -> dict[tuple[str, ...], tuple[object, object]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: dict[tuple[str, ...], tuple[object, object]] = {}
        for key in left.keys() | right.keys():
            assert isinstance(key, str)
            left_value = left[key] if key in left else _MISSING
            right_value = right[key] if key in right else _MISSING
            differences.update(_leaf_differences(left_value, right_value, (*path, key)))
        return differences
    if left != right:
        return {path: (left, right)}
    return {}


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


def test_p2_source_is_a_minimal_student_bbox_weight_overlay() -> None:
    source = runpy.run_path(str(P2))

    assert source["_base_"] == ["./support_residual_no_reliability_linear.py"]
    assert source["model"] == {"bbox_head": {"loss_bbox": {"loss_weight": 2.5}}}
    assert source["experiment"]["name"] == P2_IDENTITY
    assert {key for key in source if not key.startswith("_")} == {
        "experiment",
        "model",
    }


def test_p2_changes_exactly_one_resolved_student_model_leaf_from_p0() -> None:
    base = _load_config(P0)
    candidate = _load_config(P2)
    base_model = base["model"]
    candidate_model = candidate["model"]
    assert isinstance(base_model, dict)
    assert isinstance(candidate_model, dict)

    assert _leaf_differences(base_model, candidate_model) == {
        STUDENT_BBOX_WEIGHT_PATH: (2.0, 2.5)
    }
    excluded = {"experiment", "model"}
    assert {key: value for key, value in candidate.items() if key not in excluded} == {
        key: value for key, value in base.items() if key not in excluded
    }


def test_p2_has_no_resolved_model_alias() -> None:
    candidate_model = _model(P2)
    aliases = []
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        if path == P2:
            continue
        if _load_config(path).get("model") == candidate_model:
            aliases.append(path.relative_to(CONFIG_ROOT).as_posix())

    assert aliases == []


def test_p2_changes_student_bbox_weight_but_keeps_nested_teacher_at_two() -> None:
    base_model = _model(P0)
    candidate_model = _model(P2)

    assert base_model["bbox_head"]["loss_bbox"]["loss_weight"] == 2.0
    assert candidate_model["bbox_head"]["loss_bbox"]["loss_weight"] == 2.5
    assert candidate_model["teacher"] == base_model["teacher"]
    assert candidate_model["teacher"]["bbox_head"]["loss_bbox"]["loss_weight"] == 2.0
    assert candidate_model["teacher_checkpoint"] == base_model["teacher_checkpoint"]
    assert candidate_model["distillation"] == base_model["distillation"]


def test_p2_preserves_seed_training_evaluation_batch_and_fp32_protocol() -> None:
    base = _load_config(P0)
    candidate = _load_config(P2)

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
        assert candidate[key] == base[key]

    assert candidate["train_cfg"] == {
        "type": "EpochBasedTrainLoop",
        "max_epochs": 50,
        "val_interval": 10,
    }
    assert candidate["randomness"]["seed"] == 20250218
    assert candidate["train_dataloader"]["dataset"]["seed"] == 20250218
    assert candidate["train_dataloader"]["dataset"]["include_clean_teacher"] is True
    assert candidate["optim_wrapper"]["type"] == "OptimWrapper"
    contract = candidate["paper_data_contract"]
    assert (
        len(contract["evaluation_delays_ms"]) * len(contract["fault_conditions"]) == 12
    )

    assert _literal_assignment(P0_DEPLOYMENT, "GPU_COUNT") == 4
    assert _literal_assignment(P0_DEPLOYMENT, "BATCH_SIZE_PER_GPU") == 2
    assert _literal_assignment(P0_DEPLOYMENT, "GLOBAL_BATCH_SIZE") == 8
    assert _literal_assignment(P0_DEPLOYMENT, "MAX_EPOCHS") == 50
    assert _literal_assignment(P0_DEPLOYMENT, "VAL_INTERVAL") == 10
    assert _literal_assignment(P0_DEPLOYMENT, "TRAINING_SEED") == 20250218
    assert _literal_assignment(P0_DEPLOYMENT, "PRECISION") == "FP32"


def test_p2_keeps_p0_seeded_fusion_parameter_schema_and_initialization() -> None:
    base_model = _model(P0)
    candidate_model = _model(P2)

    torch.manual_seed(20250218)
    base_fusion = _fusion(base_model)
    torch.manual_seed(20250218)
    candidate_fusion = _fusion(candidate_model)

    base_state = base_fusion.state_dict()
    candidate_state = candidate_fusion.state_dict()
    assert tuple(base_state) == tuple(candidate_state)
    assert {name: tuple(tensor.shape) for name, tensor in base_state.items()} == {
        name: tuple(tensor.shape) for name, tensor in candidate_state.items()
    }
    assert sum(parameter.numel() for parameter in base_fusion.parameters()) == sum(
        parameter.numel() for parameter in candidate_fusion.parameters()
    )
    for name, tensor in base_state.items():
        assert torch.equal(tensor, candidate_state[name])
