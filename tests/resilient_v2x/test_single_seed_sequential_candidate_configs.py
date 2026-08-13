from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest
import torch

from transvision.models.resilient_v2x import DynamicExpertRouter

from .test_configs import CONFIG_ROOT, _load_config
from .test_routing import single_router_inputs


IMPROVEMENTS = CONFIG_ROOT / "improvements"
P1 = IMPROVEMENTS / "reliability_gated_residual.py"
P3 = IMPROVEMENTS / "reliability_gated_residual_bbox25.py"
P4 = IMPROVEMENTS / "reliability_gated_residual_full_nonlinear.py"
P5 = IMPROVEMENTS / "reliability_gated_residual_full_nonlinear_bbox25.py"
P6 = IMPROVEMENTS / "reliability_gated_residual_full_linear.py"
P7 = IMPROVEMENTS / "reliability_gated_residual_full_linear_bbox25.py"
STUDENT_BBOX_LEAF = ("model", "bbox_head", "loss_bbox", "loss_weight")
MISSING = object()


def _leaf_differences(
    left: object,
    right: object,
    path: tuple[str, ...] = (),
) -> dict[tuple[str, ...], tuple[object, object]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: dict[tuple[str, ...], tuple[object, object]] = {}
        for key in left.keys() | right.keys():
            left_value = left[key] if key in left else MISSING
            right_value = right[key] if key in right else MISSING
            differences.update(_leaf_differences(left_value, right_value, (*path, key)))
        return differences
    if left != right:
        return {path: (left, right)}
    return {}


@pytest.mark.parametrize(
    ("candidate", "base", "base_reference", "candidate_id"),
    (
        (P3, P1, "./reliability_gated_residual.py", "P3"),
        (
            P5,
            P4,
            "./reliability_gated_residual_full_nonlinear.py",
            "P5",
        ),
        (P7, P6, "./reliability_gated_residual_full_linear.py", "P7"),
    ),
)
def test_bbox25_candidates_change_only_the_student_bbox_loss(
    candidate: Path,
    base: Path,
    base_reference: str,
    candidate_id: str,
) -> None:
    source = runpy.run_path(str(candidate))
    assert source["_base_"] == [base_reference]
    assert (candidate.parent / base_reference).resolve() == base.resolve()
    assert source["model"] == {"bbox_head": {"loss_bbox": {"loss_weight": 2.5}}}

    base_config = _load_config(base)
    candidate_config = _load_config(candidate)
    operational = copy.deepcopy(candidate_config)
    operational["experiment"] = copy.deepcopy(base_config["experiment"])
    assert _leaf_differences(base_config, operational) == {
        STUDENT_BBOX_LEAF: (2.0, 2.5)
    }

    base_model = base_config["model"]
    candidate_model = candidate_config["model"]
    assert candidate_model["teacher"] == base_model["teacher"]
    assert candidate_model["teacher_checkpoint"] == base_model["teacher_checkpoint"]
    assert candidate_model["distillation"] == base_model["distillation"]
    assert candidate_model["teacher"]["bbox_head"]["loss_bbox"]["loss_weight"] == 2.0
    provenance = candidate_config["experiment"]["candidate_provenance"]
    assert provenance["candidate_id"] == candidate_id
    assert provenance["changed_model_leaf"] == (
        "model.bbox_head.loss_bbox.loss_weight"
    )


@pytest.mark.parametrize(
    ("path", "ptf_mode", "use_reliability", "bbox_weight"),
    (
        (P1, "linear", False, 2.0),
        (P3, "linear", False, 2.5),
        (P4, "nonlinear", True, 2.0),
        (P5, "nonlinear", True, 2.5),
        (P6, "linear", True, 2.0),
        (P7, "linear", True, 2.5),
    ),
)
def test_all_sequential_candidates_preserve_the_fixed_training_contract(
    path: Path,
    ptf_mode: str,
    use_reliability: bool,
    bbox_weight: float,
) -> None:
    config = _load_config(path)
    model = config["model"]
    assert model["ptf_mode"] == ptf_mode
    assert model["use_reliability"] is use_reliability
    assert model["support_residual_weight"] == 0.5
    assert model["support_residual_reliability_gate"] is True
    assert model["bbox_head"]["loss_bbox"]["loss_weight"] == bbox_weight
    assert model["teacher"]["bbox_head"]["loss_bbox"]["loss_weight"] == 2.0
    assert config["randomness"]["seed"] == 20250218
    assert config["train_dataloader"]["dataset"]["seed"] == 20250218
    assert config["train_cfg"] == {
        "type": "EpochBasedTrainLoop",
        "max_epochs": 50,
        "val_interval": 10,
    }


def test_full_descriptor_uses_raw_reliability_not_support_only_values() -> None:
    p1_model = _load_config(P1)["model"]
    p4_model = _load_config(P4)["model"]
    p6_model = _load_config(P6)["model"]
    assert p1_model["use_reliability"] is False
    assert p4_model["use_reliability"] is True
    assert p6_model["use_reliability"] is True

    torch.manual_seed(20250218)
    router = DynamicExpertRouter(
        channels=256,
        hidden_channels=256,
        support_residual_weight=0.5,
        support_residual_reliability_gate=True,
    ).eval()
    inputs = single_router_inputs((True, True, True, True))
    inputs["branch_reliability"] = torch.tensor([[0.8, 0.4, 0.2, 0.0]])
    inputs["routing_mode"] = "uniform"

    full_inputs = dict(inputs)
    full_inputs["use_reliability"] = True
    support_only_inputs = dict(inputs)
    support_only_inputs["use_reliability"] = False
    full = router(**full_inputs)
    support_only = router(**support_only_inputs)

    torch.testing.assert_close(
        full.descriptor[:, 768:770], torch.tensor([[0.6, 0.1]])
    )
    torch.testing.assert_close(
        support_only.descriptor[:, 768:770], torch.tensor([[1.0, 1.0]])
    )
    assert torch.equal(full.descriptor[:, 770:], support_only.descriptor[:, 770:])

