from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from torch import nn

from transvision.models.resilient_v2x.distillation import (
    DistillationLosses,
    assert_teacher_frozen,
    bernoulli_kl_from_logits,
    distillation_losses,
    freeze_teacher,
)


ROOT = Path(__file__).resolve().parents[2]


def valid_loss_inputs(
    *,
    requires_grad: bool = False,
) -> dict[str, object]:
    return {
        "teacher_feature": torch.randn(
            2, 4, 3, 2, requires_grad=requires_grad
        ),
        "student_feature": torch.randn(
            2, 4, 3, 2, requires_grad=requires_grad
        ),
        "teacher_logits": torch.randn(
            2, 3, 2, 2, requires_grad=requires_grad
        ),
        "student_logits": torch.randn(
            2, 3, 2, 2, requires_grad=requires_grad
        ),
        "temperature": 4.0,
        "lambda_feature": 1.0,
        "lambda_logit": 1.0,
        "valid_sample_mask": torch.tensor([True, False]),
    }


def test_distillation_losses_dataclass_is_frozen() -> None:
    losses = DistillationLosses(
        feature=torch.tensor(1.0),
        bernoulli=torch.tensor(2.0),
        total=torch.tensor(3.0),
    )

    with pytest.raises(FrozenInstanceError):
        losses.total = torch.tensor(4.0)


def test_equal_logits_have_zero_bernoulli_kl() -> None:
    logits = torch.tensor([[-3.0, 0.0, 4.0]], dtype=torch.float64)

    loss = bernoulli_kl_from_logits(
        logits,
        logits,
        temperature=4.0,
        epsilon=1e-6,
    )

    torch.testing.assert_close(
        loss,
        torch.zeros_like(loss),
        atol=1e-14,
        rtol=0.0,
    )


def test_bernoulli_kl_locks_direction_and_negative_term() -> None:
    teacher = torch.tensor([0.0])
    student = torch.tensor([math.log(4.0)])

    loss = bernoulli_kl_from_logits(
        teacher,
        student,
        temperature=1.0,
        epsilon=1e-6,
    )

    expected = 0.5 * (
        math.log(0.5) - math.log(0.8)
        + math.log(0.5) - math.log(0.2)
    )
    assert loss.item() == pytest.approx(expected, abs=1e-7)


def test_bernoulli_kl_is_mean_over_every_element() -> None:
    teacher = torch.tensor([0.0, 0.0])
    student = torch.tensor([0.0, math.log(4.0)])

    pair_loss = bernoulli_kl_from_logits(
        teacher,
        student,
        temperature=1.0,
        epsilon=1e-6,
    )
    nonzero_loss = bernoulli_kl_from_logits(
        teacher[1:],
        student[1:],
        temperature=1.0,
        epsilon=1e-6,
    )

    torch.testing.assert_close(pair_loss, nonzero_loss / 2.0)


def test_bernoulli_kl_accepts_scalar_and_mixed_floating_dtypes() -> None:
    loss = bernoulli_kl_from_logits(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float32),
        temperature=2.0,
        epsilon=1e-6,
    )

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_extreme_logits_have_finite_loss_and_student_gradient_only() -> None:
    teacher = torch.tensor(
        [[-1000.0, 1000.0]],
        requires_grad=True,
    )
    student = torch.tensor(
        [[1000.0, -1000.0]],
        requires_grad=True,
    )

    loss = bernoulli_kl_from_logits(
        teacher,
        student,
        temperature=4.0,
        epsilon=1e-6,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert teacher.grad is None
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_tiny_valid_epsilon_keeps_extreme_kl_and_gradient_finite() -> None:
    teacher = torch.tensor([[-1000.0, 1000.0]])
    student = torch.tensor(
        [[1000.0, -1000.0]],
        requires_grad=True,
    )

    loss = bernoulli_kl_from_logits(
        teacher,
        student,
        temperature=1.0,
        epsilon=1e-12,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_feature_loss_uses_per_sample_element_mean_then_valid_mean() -> None:
    teacher_feature = torch.zeros(2, 2, 2, 2)
    student_feature = torch.stack(
        (
            torch.ones(2, 2, 2),
            torch.full((2, 2, 2), 2.0),
        )
    )
    logits = torch.zeros(2, 1, 1, 1)

    losses = distillation_losses(
        teacher_feature=teacher_feature,
        student_feature=student_feature,
        teacher_logits=logits,
        student_logits=logits,
        temperature=4.0,
        lambda_feature=1.0,
        lambda_logit=1.0,
        valid_sample_mask=torch.tensor([True, True]),
    )

    torch.testing.assert_close(losses.feature, torch.tensor(2.5))
    torch.testing.assert_close(
        losses.bernoulli,
        torch.zeros_like(losses.bernoulli),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_low_precision_feature_loss_uses_float32_and_backpropagates(
    dtype: torch.dtype,
) -> None:
    teacher_feature = torch.randn(2, 2, 2, 2, dtype=dtype)
    student_feature = torch.randn(
        2,
        2,
        2,
        2,
        dtype=dtype,
        requires_grad=True,
    )
    logits = torch.zeros(2, 1, 1, 1)

    losses = distillation_losses(
        teacher_feature=teacher_feature,
        student_feature=student_feature,
        teacher_logits=logits,
        student_logits=logits,
        temperature=4.0,
        lambda_feature=1.0,
        lambda_logit=1.0,
        valid_sample_mask=torch.tensor([True, False]),
    )
    losses.total.backward()

    expected = (
        (student_feature.detach()[0].float() - teacher_feature[0].float())
        .square()
        .mean()
    )
    assert losses.feature.dtype == torch.float32
    torch.testing.assert_close(losses.feature, expected)
    assert student_feature.grad is not None
    assert torch.isfinite(student_feature.grad).all()
    assert torch.count_nonzero(student_feature.grad[0]).item() > 0
    assert torch.count_nonzero(student_feature.grad[1]).item() == 0


def test_float64_feature_loss_preserves_small_difference_and_gradient() -> None:
    teacher_feature = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    student_feature = torch.tensor(
        [[[[1.0 + 1e-8]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    logits = torch.zeros(1, 1, 1, 1)

    losses = distillation_losses(
        teacher_feature=teacher_feature,
        student_feature=student_feature,
        teacher_logits=logits,
        student_logits=logits,
        temperature=4.0,
        lambda_feature=1.0,
        lambda_logit=0.0,
        valid_sample_mask=torch.tensor([True]),
    )
    losses.total.backward()

    expected = (student_feature.detach() - teacher_feature).square().mean()
    assert losses.feature.dtype == torch.float64
    assert losses.feature.item() == pytest.approx(1e-16, rel=1e-7)
    torch.testing.assert_close(losses.feature, expected)
    assert student_feature.grad is not None
    assert student_feature.grad.dtype == torch.float64
    assert torch.isfinite(student_feature.grad).all()
    assert torch.count_nonzero(student_feature.grad).item() == 1


def test_dense_background_elements_contribute_to_bernoulli_mean() -> None:
    teacher_feature = torch.zeros(1, 1, 1, 1)
    student_feature = torch.zeros_like(teacher_feature)
    teacher_logits = torch.zeros(1, 1, 2, 2)
    student_logits = torch.zeros_like(teacher_logits)
    student_logits[0, 0, 0, 0] = math.log(4.0)

    losses = distillation_losses(
        teacher_feature=teacher_feature,
        student_feature=student_feature,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        temperature=1.0,
        lambda_feature=1.0,
        lambda_logit=1.0,
        valid_sample_mask=torch.tensor([True]),
    )
    one_element = bernoulli_kl_from_logits(
        torch.tensor([0.0]),
        torch.tensor([math.log(4.0)]),
        temperature=1.0,
        epsilon=1e-6,
    )

    torch.testing.assert_close(losses.bernoulli, one_element / 4.0)


@pytest.mark.parametrize("poison", [math.nan, 1e20], ids=["nan", "huge"])
def test_invalid_poison_rows_do_not_change_losses_or_valid_gradients(
    poison: float,
) -> None:
    teacher_feature = torch.randn(2, 2, 2, 2)
    student_feature = torch.randn(2, 2, 2, 2, requires_grad=True)
    teacher_logits = torch.randn(2, 1, 2, 2)
    student_logits = torch.randn(2, 1, 2, 2, requires_grad=True)
    teacher_feature[1].fill_(poison)
    teacher_logits[1].fill_(poison)
    with torch.no_grad():
        student_feature[1].fill_(poison)
        student_logits[1].fill_(poison)

    masked = distillation_losses(
        teacher_feature=teacher_feature,
        student_feature=student_feature,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        temperature=4.0,
        lambda_feature=1.0,
        lambda_logit=1.0,
        valid_sample_mask=torch.tensor([True, False]),
    )
    reference_student_feature = (
        student_feature.detach()[:1].clone().requires_grad_()
    )
    reference_student_logits = (
        student_logits.detach()[:1].clone().requires_grad_()
    )
    reference = distillation_losses(
        teacher_feature=teacher_feature[:1],
        student_feature=reference_student_feature,
        teacher_logits=teacher_logits[:1],
        student_logits=reference_student_logits,
        temperature=4.0,
        lambda_feature=1.0,
        lambda_logit=1.0,
        valid_sample_mask=torch.tensor([True]),
    )

    masked.total.backward()
    reference.total.backward()

    torch.testing.assert_close(masked.feature, reference.feature)
    torch.testing.assert_close(masked.bernoulli, reference.bernoulli)
    torch.testing.assert_close(masked.total, reference.total)
    torch.testing.assert_close(
        student_feature.grad[0],
        reference_student_feature.grad[0],
    )
    torch.testing.assert_close(
        student_logits.grad[0],
        reference_student_logits.grad[0],
    )
    assert torch.count_nonzero(student_feature.grad[1]).item() == 0
    assert torch.count_nonzero(student_logits.grad[1]).item() == 0


def test_student_receives_finite_gradient_and_teacher_does_not() -> None:
    inputs = valid_loss_inputs(requires_grad=True)

    losses = distillation_losses(**inputs)
    losses.total.backward()

    teacher_feature = inputs["teacher_feature"]
    student_feature = inputs["student_feature"]
    teacher_logits = inputs["teacher_logits"]
    student_logits = inputs["student_logits"]
    assert isinstance(teacher_feature, torch.Tensor)
    assert isinstance(student_feature, torch.Tensor)
    assert isinstance(teacher_logits, torch.Tensor)
    assert isinstance(student_logits, torch.Tensor)
    assert teacher_feature.grad is None
    assert teacher_logits.grad is None
    assert student_feature.grad is not None
    assert student_logits.grad is not None
    assert torch.isfinite(student_feature.grad).all()
    assert torch.isfinite(student_logits.grad).all()
    assert torch.count_nonzero(student_feature.grad[0]).item() > 0
    assert torch.count_nonzero(student_logits.grad[0]).item() > 0
    assert torch.count_nonzero(student_feature.grad[1]).item() == 0
    assert torch.count_nonzero(student_logits.grad[1]).item() == 0


def test_total_uses_exact_weights_and_temperature_squared_only_once() -> None:
    inputs = valid_loss_inputs()
    inputs["temperature"] = 4.0
    inputs["lambda_feature"] = 2.0
    inputs["lambda_logit"] = 3.0

    losses = distillation_losses(**inputs)

    expected = (
        2.0 * losses.feature
        + 3.0 * (4.0**2) * losses.bernoulli
    )
    torch.testing.assert_close(losses.total, expected, atol=0.0, rtol=0.0)
    direct_bernoulli = bernoulli_kl_from_logits(
        inputs["teacher_logits"][:1],
        inputs["student_logits"][:1],
        temperature=4.0,
        epsilon=1e-6,
    )
    torch.testing.assert_close(
        losses.bernoulli,
        direct_bernoulli,
        atol=0.0,
        rtol=0.0,
    )


def test_outputs_are_finite_floating_scalars_connected_to_student() -> None:
    inputs = valid_loss_inputs(requires_grad=True)

    losses = distillation_losses(**inputs)

    for value in (losses.feature, losses.bernoulli, losses.total):
        assert value.shape == ()
        assert value.is_floating_point()
        assert torch.isfinite(value)
        assert value.requires_grad


def test_all_false_mask_raises_exact_error_before_reading_poison() -> None:
    inputs = valid_loss_inputs()
    for field in (
        "teacher_feature",
        "student_feature",
        "teacher_logits",
        "student_logits",
    ):
        value = inputs[field]
        assert isinstance(value, torch.Tensor)
        value.fill_(math.nan)
    inputs["valid_sample_mask"] = torch.tensor([False, False])

    with pytest.raises(
        RuntimeError,
        match="^no valid samples for distillation$",
    ):
        distillation_losses(**inputs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("teacher_feature", object()),
        ("teacher_feature", torch.zeros(2, 4, 3)),
        ("teacher_feature", torch.zeros(0, 4, 3, 2)),
        ("teacher_feature", torch.zeros(2, 4, 0, 2)),
        ("teacher_feature", torch.zeros(2, 4, 3, 2, dtype=torch.int64)),
        ("student_feature", torch.zeros(2, 5, 3, 2)),
        ("student_feature", torch.zeros(2, 4, 3, 2, dtype=torch.float64)),
        ("teacher_logits", object()),
        ("teacher_logits", torch.zeros(2, 3, 2)),
        ("teacher_logits", torch.zeros(0, 3, 2, 2)),
        ("teacher_logits", torch.zeros(2, 3, 0, 2)),
        ("teacher_logits", torch.zeros(2, 3, 2, 2, dtype=torch.int64)),
        ("student_logits", torch.zeros(2, 4, 2, 2)),
        ("student_logits", torch.zeros(2, 3, 2, 2, dtype=torch.float64)),
        ("student_logits", torch.zeros(3, 3, 2, 2)),
        ("valid_sample_mask", object()),
        ("valid_sample_mask", torch.tensor([[True], [False]])),
        ("valid_sample_mask", torch.tensor([1, 0])),
        ("valid_sample_mask", torch.tensor([True])),
    ],
)
def test_combined_losses_reject_invalid_structure(
    field: str,
    replacement: object,
) -> None:
    inputs = valid_loss_inputs()
    inputs[field] = replacement

    with pytest.raises(ValueError):
        distillation_losses(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("teacher_feature", math.nan),
        ("teacher_feature", math.inf),
        ("student_feature", math.nan),
        ("student_feature", math.inf),
        ("teacher_logits", math.nan),
        ("teacher_logits", math.inf),
        ("student_logits", math.nan),
        ("student_logits", math.inf),
    ],
)
def test_combined_losses_reject_nonfinite_valid_rows(
    field: str,
    value: float,
) -> None:
    inputs = valid_loss_inputs()
    tensor = inputs[field]
    assert isinstance(tensor, torch.Tensor)
    tensor[0].flatten()[0] = value

    with pytest.raises(ValueError, match="finite"):
        distillation_losses(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", True),
        ("temperature", 0.0),
        ("temperature", -1.0),
        ("temperature", math.nan),
        ("temperature", math.inf),
        ("lambda_feature", True),
        ("lambda_feature", -0.1),
        ("lambda_feature", math.nan),
        ("lambda_feature", math.inf),
        ("lambda_logit", False),
        ("lambda_logit", -0.1),
        ("lambda_logit", math.nan),
        ("lambda_logit", math.inf),
    ],
)
def test_combined_losses_reject_invalid_scalars(
    field: str,
    value: object,
) -> None:
    inputs = valid_loss_inputs()
    inputs[field] = value

    with pytest.raises(ValueError):
        distillation_losses(**inputs)


def test_combined_losses_reject_cross_pair_device_mismatch() -> None:
    inputs = valid_loss_inputs()
    inputs["student_logits"] = torch.empty(
        2,
        3,
        2,
        2,
        device="meta",
    )

    with pytest.raises(ValueError, match="device"):
        distillation_losses(**inputs)


def test_combined_losses_reject_mask_device_mismatch() -> None:
    inputs = valid_loss_inputs()
    inputs["valid_sample_mask"] = torch.empty(
        2,
        dtype=torch.bool,
        device="meta",
    )

    with pytest.raises(ValueError, match="device"):
        distillation_losses(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", True),
        ("temperature", 0.0),
        ("temperature", -1.0),
        ("temperature", math.nan),
        ("temperature", math.inf),
        ("epsilon", False),
        ("epsilon", 0.0),
        ("epsilon", -0.1),
        ("epsilon", 0.5),
        ("epsilon", 1.0),
        ("epsilon", math.nan),
        ("epsilon", math.inf),
    ],
)
def test_kl_rejects_invalid_scalars(field: str, value: object) -> None:
    values = {
        "teacher_logits": torch.zeros(2, 3),
        "student_logits": torch.zeros(2, 3),
        "temperature": 4.0,
        "epsilon": 1e-6,
    }
    values[field] = value

    with pytest.raises(ValueError):
        bernoulli_kl_from_logits(**values)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("teacher_logits", object()),
        ("student_logits", object()),
        ("teacher_logits", torch.zeros(0)),
        ("student_logits", torch.zeros(2, 2)),
        ("teacher_logits", torch.zeros(2, 3, dtype=torch.int64)),
        ("student_logits", torch.zeros(2, 3, dtype=torch.bool)),
    ],
)
def test_kl_rejects_invalid_tensors(
    field: str,
    replacement: object,
) -> None:
    values = {
        "teacher_logits": torch.zeros(2, 3),
        "student_logits": torch.zeros(2, 3),
        "temperature": 4.0,
        "epsilon": 1e-6,
    }
    values[field] = replacement

    with pytest.raises(ValueError):
        bernoulli_kl_from_logits(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("teacher_logits", math.nan),
        ("teacher_logits", math.inf),
        ("student_logits", math.nan),
        ("student_logits", math.inf),
    ],
)
def test_kl_rejects_nonfinite_logits(field: str, value: float) -> None:
    values = {
        "teacher_logits": torch.zeros(2, 3),
        "student_logits": torch.zeros(2, 3),
        "temperature": 4.0,
        "epsilon": 1e-6,
    }
    tensor = values[field].clone()
    tensor[0, 0] = value
    values[field] = tensor

    with pytest.raises(ValueError, match="finite"):
        bernoulli_kl_from_logits(**values)


def test_kl_rejects_device_mismatch() -> None:
    with pytest.raises(ValueError, match="device"):
        bernoulli_kl_from_logits(
            torch.zeros(2, 3),
            torch.empty(2, 3, device="meta"),
            temperature=4.0,
            epsilon=1e-6,
        )


def test_freeze_teacher_returns_same_object_and_clears_all_training_state() -> None:
    teacher = nn.Sequential(
        nn.Linear(4, 4),
        nn.Sequential(nn.BatchNorm1d(4), nn.Dropout()),
    )
    for parameter in teacher.parameters():
        parameter.grad = torch.ones_like(parameter)

    returned = freeze_teacher(teacher)

    assert returned is teacher
    assert all(not module.training for module in teacher.modules())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert_teacher_frozen(teacher)


def test_freeze_teacher_requires_module() -> None:
    with pytest.raises(ValueError, match="teacher"):
        freeze_teacher(object())


def test_assert_teacher_frozen_requires_module() -> None:
    with pytest.raises(ValueError, match="teacher"):
        assert_teacher_frozen(object())


def test_assert_teacher_frozen_rejects_training_nested_module() -> None:
    teacher = freeze_teacher(
        nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    )
    teacher[1].train()

    with pytest.raises(RuntimeError, match="eval|training"):
        assert_teacher_frozen(teacher)


def test_assert_teacher_frozen_rejects_reenabled_gradient() -> None:
    teacher = freeze_teacher(nn.Linear(4, 4))
    teacher.weight.requires_grad_(True)

    with pytest.raises(RuntimeError, match="gradient|requires_grad"):
        assert_teacher_frozen(teacher)


def test_assert_teacher_frozen_rejects_stale_gradient() -> None:
    teacher = freeze_teacher(nn.Linear(4, 4))
    teacher.weight.grad = torch.ones_like(teacher.weight)

    with pytest.raises(RuntimeError, match="gradient|grad"):
        assert_teacher_frozen(teacher)


def test_task_seven_package_import_does_not_load_custom_ops() -> None:
    code = (
        "import sys; "
        "from transvision.models.resilient_v2x import distillation_losses; "
        "assert 'transvision.models.bev_pool' not in sys.modules; "
        "assert 'transvision.models.voxel.voxel_layer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
