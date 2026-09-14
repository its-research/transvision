from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from transvision.models.event_track_v2x.learned_association import (
    AssociationLogitsV1,
    LearnedAssociationModelV1,
    association_loss_v1,
    infer_association_v1,
    train_association_model_v1,
)
from transvision.models.event_track_v2x.learning_contracts import (
    ASSOCIATION_FEATURE_DIM_V1,
    ASSOCIATION_FEATURE_GROUPS_V1,
    AssociationTrainingConfigV1,
    association_feature_schema_document_v1,
)


@dataclass
class Frame:
    left_features: np.ndarray
    right_features: np.ndarray
    targets: np.ndarray
    right_identity_ids: tuple[str, ...]


def _frame(offset: float = 0.0) -> Frame:
    left = np.zeros((2, ASSOCIATION_FEATURE_DIM_V1), dtype=np.float32)
    right = np.zeros((3, ASSOCIATION_FEATURE_DIM_V1), dtype=np.float32)
    left[0, 0] = 0.1 + offset
    left[1, 0] = 0.8 + offset
    right[0, 0] = 0.1 + offset
    right[1, 0] = 0.8 + offset
    right[2, 0] = -0.9 + offset
    targets = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    return Frame(left, right, targets, ("right-a", "right-b", "right-c"))


def test_feature_schema_is_explicit_fixed_dimension() -> None:
    document = association_feature_schema_document_v1()

    assert document["dimension"] == ASSOCIATION_FEATURE_DIM_V1 == 208
    assert [name for name, _ in ASSOCIATION_FEATURE_GROUPS_V1] == [
        "geometry",
        "motion",
        "frozen_appearance",
        "class",
        "source_time",
        "covariance",
        "pose",
        "lineage",
    ]
    appearance = next(
        group for group in document["groups"] if group["name"] == "frozen_appearance"
    )
    assert appearance["dimension"] == 128
    assert document["appearance_contract"]["training_gradient_allowed"] is False


def test_pairwise_bce_and_bidirectional_assignment_backpropagate() -> None:
    torch.manual_seed(3)
    model = LearnedAssociationModelV1(hidden_dim=16, dropout=0.0)
    frame = _frame()
    logits = model(
        torch.from_numpy(frame.left_features),
        torch.from_numpy(frame.right_features),
    )
    loss = association_loss_v1(logits, torch.from_numpy(frame.targets).float())

    assert loss.total.item() > 0.0
    assert loss.pairwise_bce.item() > 0.0
    assert loss.left_assignment.item() > 0.0
    assert loss.right_assignment.item() > 0.0
    loss.total.backward()
    assert model.pair_scorer[-1].weight.grad is not None
    assert model.left_dustbin.weight.grad is not None
    assert model.right_dustbin.weight.grad is not None
    assert torch.isfinite(model.pair_scorer[-1].weight.grad).all()


def test_loss_rejects_non_one_to_one_supervision() -> None:
    model = LearnedAssociationModelV1(hidden_dim=8, dropout=0.0)
    logits = model(
        torch.zeros(1, ASSOCIATION_FEATURE_DIM_V1),
        torch.zeros(2, ASSOCIATION_FEATURE_DIM_V1),
    )
    with pytest.raises(ValueError, match="one-to-one"):
        association_loss_v1(logits, torch.ones(1, 2))


def test_right_assignment_has_exactly_one_dustbin_class() -> None:
    pair_logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    unmatched_left = torch.tensor([-0.2, 0.1])
    unmatched_right = torch.tensor([0.2, -0.4])
    targets = torch.eye(2)

    loss = association_loss_v1(
        AssociationLogitsV1(pair_logits, unmatched_left, unmatched_right),
        targets,
    )
    expected_logits = torch.cat(
        (pair_logits.transpose(0, 1), unmatched_right[:, None]), dim=1
    )

    assert expected_logits.shape == (2, 3)  # two left identities + one dustbin
    assert loss.right_assignment == pytest.approx(
        torch.nn.functional.cross_entropy(expected_logits, torch.tensor([0, 1])).item()
    )


def test_inference_exposes_dustbin_and_preserves_top_h_other_mass() -> None:
    model = LearnedAssociationModelV1(hidden_dim=8, dropout=0.0)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.pair_scorer[-1].bias.data.fill_(-4.0)
    model.left_dustbin.bias.data.fill_(4.0)
    model.right_dustbin.bias.data.fill_(4.0)
    frame = _frame()

    result = infer_association_v1(
        model,
        frame.left_features,
        frame.right_features,
        right_identity_ids=frame.right_identity_ids,
        top_h=2,
    )

    assert result.assignment.pairs == ()
    assert result.assignment.unmatched_left == (0, 1)
    assert result.assignment.unmatched_right == (0, 1, 2)
    assert len(result.left_identity_distributions[0].hypotheses) == 2
    for distribution in result.left_identity_distributions:
        assert distribution.unmatched_probability > 0.99
        assert distribution.other_probability >= distribution.unmatched_probability
        assert (
            sum(item.probability for item in distribution.hypotheses)
            + distribution.other_probability
        ) == pytest.approx(1.0)


def test_train_loop_updates_parameters_with_real_optimizer_steps() -> None:
    config = AssociationTrainingConfigV1(
        hidden_dim=8,
        dropout=0.0,
        learning_rate=5e-3,
        weight_decay=0.0,
        epochs=2,
        frames_per_step=1,
        top_h=2,
    )
    torch.manual_seed(17)
    initial = LearnedAssociationModelV1(hidden_dim=8, dropout=0.0)
    initial_state = {
        name: value.detach().clone() for name, value in initial.state_dict().items()
    }

    result = train_association_model_v1(
        (_frame(0.0), _frame(0.05)),
        (_frame(0.1),),
        config=config,
        seed=17,
    )

    assert len(result.history) == 2
    assert all(np.isfinite(item.train_loss) for item in result.history)
    assert all(np.isfinite(item.validation_loss) for item in result.history)
    assert any(
        not torch.equal(initial_state[name], value.cpu())
        for name, value in result.model.state_dict().items()
    )
