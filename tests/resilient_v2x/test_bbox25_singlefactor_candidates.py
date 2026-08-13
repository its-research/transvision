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

from .test_configs import CONFIG_ROOT, ROOT, _load_config


NO_RELIABILITY = CONFIG_ROOT / "ablations" / "no_reliability.py"
SUPPORT_RESIDUAL = CONFIG_ROOT / "improvements" / "support_residual.py"
NO_RELIABILITY_BBOX25 = CONFIG_ROOT / "improvements" / "no_reliability_bbox25.py"
SUPPORT_RESIDUAL_BBOX25 = CONFIG_ROOT / "improvements" / "support_residual_bbox25.py"
BOOTSTRAP = ROOT / "tools" / "resilient_v2x" / "clearml_5090_bootstrap.py"
STUDENT_BBOX_WEIGHT_PATH = ("model", "bbox_head", "loss_bbox", "loss_weight")
MUTUAL_EXCLUSION_GROUP = "bbox25_single_factor_followup"
PROTOCOL_LOCK = {
    "training_seed": 20250218,
    "precision": "FP32",
    "gpu_count": 4,
    "train_batch_size_per_gpu": 2,
    "global_batch_size": 8,
    "max_epochs": 50,
    "val_interval": 10,
}
_MISSING = object()

CANDIDATES = (
    (
        NO_RELIABILITY_BBOX25,
        NO_RELIABILITY,
        "../ablations/no_reliability.py",
        "dair_improvement_no_reliability_bbox25",
        "dair_ablation_no_reliability",
        "configs/resilient_v2x/ablations/no_reliability.py",
        "dair_improvement_support_residual_bbox25",
    ),
    (
        SUPPORT_RESIDUAL_BBOX25,
        SUPPORT_RESIDUAL,
        "./support_residual.py",
        "dair_improvement_support_residual_bbox25",
        "dair_improvement_support_residual",
        "configs/resilient_v2x/improvements/support_residual.py",
        "dair_improvement_no_reliability_bbox25",
    ),
)
CANDIDATE_BASES = tuple((candidate[0], candidate[1]) for candidate in CANDIDATES)


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
        support_residual_reliability_gate=model.get(
            "support_residual_reliability_gate", False
        ),
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
    (
        "candidate_path",
        "base_path",
        "base_reference",
        "identity",
        "base_identity",
        "base_provenance",
        "peer_identity",
    ),
    CANDIDATES,
)
def test_bbox25_sources_are_minimal_student_only_overlays_with_explicit_provenance(
    candidate_path: Path,
    base_path: Path,
    base_reference: str,
    identity: str,
    base_identity: str,
    base_provenance: str,
    peer_identity: str,
) -> None:
    source = runpy.run_path(str(candidate_path))

    assert source["_base_"] == [base_reference]
    assert (candidate_path.parent / base_reference).resolve() == base_path.resolve()
    assert source["model"] == {"bbox_head": {"loss_bbox": {"loss_weight": 2.5}}}
    assert {key for key in source if not key.startswith("_")} == {
        "experiment",
        "model",
    }

    experiment = source["experiment"]
    assert experiment["_delete_"] is True
    assert experiment["name"] == identity
    assert experiment["stage"] == "resilient_improvement"
    assert experiment["candidate_provenance"] == {
        "base_config": base_provenance,
        "base_experiment": base_identity,
        "changed_model_leaf": "model.bbox_head.loss_bbox.loss_weight",
        "base_value": 2.0,
        "candidate_value": 2.5,
        "frozen_teacher_value": 2.0,
    }
    assert _load_config(base_path)["experiment"]["name"] == base_identity
    assert experiment["trigger"] == {
        "mode": "mutually_exclusive",
        "group": MUTUAL_EXCLUSION_GROUP,
        "peer_experiment": peer_identity,
    }
    assert experiment["protocol_lock"] == PROTOCOL_LOCK


@pytest.mark.parametrize("candidate_path,base_path", CANDIDATE_BASES)
def test_bbox25_candidates_change_exactly_one_resolved_operational_leaf(
    candidate_path: Path,
    base_path: Path,
) -> None:
    base = _load_config(base_path)
    candidate = _load_config(candidate_path)

    # Experiment identity/provenance is non-operational metadata.  Once aligned,
    # the complete resolved runtime configuration must differ at exactly one leaf.
    candidate_operational = copy.deepcopy(candidate)
    candidate_operational["experiment"] = copy.deepcopy(base["experiment"])
    assert _leaf_differences(base, candidate_operational) == {
        STUDENT_BBOX_WEIGHT_PATH: (2.0, 2.5)
    }


@pytest.mark.parametrize("candidate_path,base_path", CANDIDATE_BASES)
def test_bbox25_candidates_change_student_only_and_preserve_teacher_contract(
    candidate_path: Path,
    base_path: Path,
) -> None:
    base_model = _model(base_path)
    candidate_model = _model(candidate_path)

    assert base_model["bbox_head"]["loss_bbox"]["loss_weight"] == 2.0
    assert candidate_model["bbox_head"]["loss_bbox"]["loss_weight"] == 2.5
    assert candidate_model["teacher"] == base_model["teacher"]
    assert candidate_model["teacher"]["bbox_head"]["loss_bbox"]["loss_weight"] == 2.0
    assert candidate_model["teacher_checkpoint"] == base_model["teacher_checkpoint"]
    assert candidate_model["distillation"] == base_model["distillation"]


@pytest.mark.parametrize("candidate_path,base_path", CANDIDATE_BASES)
def test_bbox25_candidates_have_no_resolved_model_alias(
    candidate_path: Path,
    base_path: Path,
) -> None:
    del base_path
    candidate_model = _model(candidate_path)
    aliases = []
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        if path == candidate_path:
            continue
        if _load_config(path).get("model") == candidate_model:
            aliases.append(path.relative_to(CONFIG_ROOT).as_posix())

    assert aliases == []


def test_bbox25_candidate_triggers_are_reciprocal_and_exactly_one() -> None:
    experiments = {
        _load_config(path)["experiment"]["name"]: _load_config(path)["experiment"]
        for path in (NO_RELIABILITY_BBOX25, SUPPORT_RESIDUAL_BBOX25)
    }
    assert set(experiments) == {
        "dair_improvement_no_reliability_bbox25",
        "dair_improvement_support_residual_bbox25",
    }

    for identity, experiment in experiments.items():
        trigger = experiment["trigger"]
        assert trigger["mode"] == "mutually_exclusive"
        assert trigger["group"] == MUTUAL_EXCLUSION_GROUP
        assert trigger["peer_experiment"] in experiments
        assert trigger["peer_experiment"] != identity
        peer = experiments[trigger["peer_experiment"]]
        assert peer["trigger"]["peer_experiment"] == identity


@pytest.mark.parametrize("candidate_path,base_path", CANDIDATE_BASES)
def test_bbox25_candidates_preserve_seeded_state_and_parameter_schema(
    candidate_path: Path,
    base_path: Path,
) -> None:
    base_model = _model(base_path)
    candidate_model = _model(candidate_path)

    # The complete model structure is identical: the sole changed leaf remains
    # a float scalar, so the registry receives the same module/parameter schema.
    def structure(value: object) -> object:
        if isinstance(value, dict):
            return {key: structure(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(structure(item) for item in value)
        return type(value)

    assert structure(candidate_model) == structure(base_model)

    torch.manual_seed(20250218)
    base_fusion = _fusion(base_model)
    torch.manual_seed(20250218)
    candidate_fusion = _fusion(candidate_model)

    base_state = base_fusion.state_dict()
    candidate_state = candidate_fusion.state_dict()
    assert tuple(base_state) == tuple(candidate_state)
    assert {
        name: (tuple(tensor.shape), tensor.dtype) for name, tensor in base_state.items()
    } == {
        name: (tuple(tensor.shape), tensor.dtype)
        for name, tensor in candidate_state.items()
    }
    assert tuple(name for name, _ in base_fusion.named_parameters()) == tuple(
        name for name, _ in candidate_fusion.named_parameters()
    )
    assert sum(parameter.numel() for parameter in base_fusion.parameters()) == sum(
        parameter.numel() for parameter in candidate_fusion.parameters()
    )
    for name, tensor in base_state.items():
        assert torch.equal(tensor, candidate_state[name])
    candidate_fusion.load_state_dict(base_state, strict=True)


@pytest.mark.parametrize("candidate_path,base_path", CANDIDATE_BASES)
def test_bbox25_candidates_preserve_fixed_training_protocol(
    candidate_path: Path,
    base_path: Path,
) -> None:
    base = _load_config(base_path)
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
        assert candidate[key] == base[key]

    assert candidate["train_cfg"] == {
        "type": "EpochBasedTrainLoop",
        "max_epochs": 50,
        "val_interval": 10,
    }
    assert candidate["randomness"]["seed"] == 20250218
    for split in ("train", "val", "test"):
        dataloader = candidate[f"{split}_dataloader"]
        assert dataloader["sampler"]["seed"] == 20250218
        assert dataloader["dataset"]["seed"] == 20250218
    assert candidate["train_dataloader"]["dataset"]["include_clean_teacher"] is True
    assert candidate["optim_wrapper"]["type"] == "OptimWrapper"
    assert candidate["experiment"]["protocol_lock"] == PROTOCOL_LOCK


def test_bbox25_protocol_lock_matches_canonical_four_gpu_fp32_launcher() -> None:
    assert _literal_assignment(BOOTSTRAP, "EXPECTED_GPU_COUNT") == 4
    assert _literal_assignment(BOOTSTRAP, "RTX5090_TRAIN_BATCH_SIZE_PER_GPU") == 2
    assert _literal_assignment(BOOTSTRAP, "EXPERIMENT_MAX_EPOCHS") == 50
    assert _literal_assignment(BOOTSTRAP, "RTX5090_VAL_INTERVAL") == 10
    assert _literal_assignment(BOOTSTRAP, "DEFAULT_TRAINING_SEED") == 20250218
    assert (
        _literal_assignment(BOOTSTRAP, "EXPECTED_GPU_COUNT")
        * _literal_assignment(BOOTSTRAP, "RTX5090_TRAIN_BATCH_SIZE_PER_GPU")
        == 8
    )
    source = BOOTSTRAP.read_text()
    assert "requires FP32; --amp is forbidden" in source
    assert '"amp": False' in source
    assert '"precision": "FP32"' in source
