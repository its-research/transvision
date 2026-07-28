from __future__ import annotations

import torch
from torch import nn

from transvision.models.resilient_v2x import (
    FrozenTeacher,
    categorical_kl_from_logits,
    head_distillation_losses,
)


def test_categorical_kl_is_zero_for_identical_logits_and_has_student_gradient() -> None:
    teacher = torch.randn(2, 3, 4, 5)
    student = teacher.clone().requires_grad_()

    loss = categorical_kl_from_logits(teacher, student, temperature=2.0)

    assert loss.item() >= -1e-7
    assert loss.item() < 1e-6
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_categorical_head_distillation_masks_invalid_samples() -> None:
    teacher_feature = torch.randn(2, 4, 2, 2)
    student_feature = torch.randn(2, 4, 2, 2, requires_grad=True)
    teacher_logits = torch.randn(2, 3, 2, 2)
    student_logits = torch.randn(2, 3, 2, 2, requires_grad=True)
    mask = torch.tensor([True, False])

    losses = head_distillation_losses(
        teacher_feature,
        student_feature,
        teacher_logits,
        student_logits,
        temperature=2.0,
        lambda_feature=1.0,
        lambda_logit=1.0,
        valid_sample_mask=mask,
        head_type="categorical",
    )
    losses.total.backward()

    assert losses.head_type == "categorical"
    assert student_feature.grad is not None
    assert student_logits.grad is not None
    assert torch.count_nonzero(student_feature.grad[1]).item() == 0
    assert torch.count_nonzero(student_logits.grad[1]).item() == 0


def test_frozen_teacher_survives_parent_train_calls() -> None:
    teacher = nn.Sequential(nn.Linear(3, 4), nn.Dropout())
    wrapped = FrozenTeacher(teacher)
    parent = nn.ModuleDict({"teacher": wrapped, "student": nn.Linear(3, 4)})

    parent.train()
    output = wrapped(torch.randn(2, 3))

    assert output.shape == (2, 4)
    assert not wrapped.training
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
