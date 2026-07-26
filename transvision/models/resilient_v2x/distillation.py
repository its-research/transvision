from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_DISTILLATION_EPSILON = 1e-6


def _require_module(value: object, name: str) -> nn.Module:
    if not isinstance(value, nn.Module):
        raise ValueError(f"{name} must be an nn.Module")
    return value


def _require_positive_scalar(value: object, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _require_nonnegative_scalar(value: object, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _require_epsilon(value: object) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 < float(value) < 0.5
    ):
        raise ValueError("epsilon must be finite and in (0,0.5)")
    return float(value)


def _require_logits_pair(
    teacher_logits: object,
    student_logits: object,
) -> tuple[Tensor, Tensor]:
    if not isinstance(teacher_logits, Tensor):
        raise ValueError("teacher_logits must be a tensor")
    if not isinstance(student_logits, Tensor):
        raise ValueError("student_logits must be a tensor")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher and student logits must have identical shapes"
        )
    if teacher_logits.numel() == 0:
        raise ValueError("teacher and student logits must be non-empty")
    if not teacher_logits.is_floating_point():
        raise ValueError("teacher_logits must be floating")
    if not student_logits.is_floating_point():
        raise ValueError("student_logits must be floating")
    if teacher_logits.device != student_logits.device:
        raise ValueError("teacher and student logits must share a device")
    if not torch.isfinite(teacher_logits.detach()).all().item():
        raise ValueError("teacher_logits must contain only finite values")
    if not torch.isfinite(student_logits.detach()).all().item():
        raise ValueError("student_logits must contain only finite values")
    return teacher_logits, student_logits


def _require_rank_four_pair(
    teacher: object,
    student: object,
    name: str,
) -> tuple[Tensor, Tensor]:
    if not isinstance(teacher, Tensor):
        raise ValueError(f"teacher_{name} must be a tensor")
    if not isinstance(student, Tensor):
        raise ValueError(f"student_{name} must be a tensor")
    if teacher.ndim != 4 or any(dimension <= 0 for dimension in teacher.shape):
        raise ValueError(
            f"teacher_{name} must have a positive rank-4 shape"
        )
    if student.ndim != 4 or any(dimension <= 0 for dimension in student.shape):
        raise ValueError(
            f"student_{name} must have a positive rank-4 shape"
        )
    if teacher.shape != student.shape:
        raise ValueError(
            f"teacher and student {name} tensors must have identical shapes"
        )
    if not teacher.is_floating_point() or not student.is_floating_point():
        raise ValueError(f"teacher and student {name} tensors must be floating")
    if teacher.dtype != student.dtype:
        raise ValueError(
            f"teacher and student {name} tensors must have identical dtype"
        )
    if teacher.device != student.device:
        raise ValueError(
            f"teacher and student {name} tensors must share a device"
        )
    return teacher, student


def _require_finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value.detach()).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _require_finite_output(value: Tensor, name: str) -> None:
    if (
        value.shape != ()
        or not value.is_floating_point()
        or not torch.isfinite(value.detach()).item()
    ):
        raise RuntimeError(f"{name} must be a finite floating scalar")


def _clamp_float32_probability(value: Tensor, epsilon: float) -> Tensor:
    zero = value.new_tensor(0.0)
    one = value.new_tensor(1.0)
    lower = torch.maximum(
        value.new_tensor(epsilon),
        torch.nextafter(zero, one),
    )
    upper = torch.minimum(
        value.new_tensor(1.0 - epsilon),
        torch.nextafter(one, zero),
    )
    return value.clamp(min=lower, max=upper)


@dataclass(frozen=True)
class DistillationLosses:
    feature: Tensor
    bernoulli: Tensor
    total: Tensor


def freeze_teacher(teacher: nn.Module) -> nn.Module:
    teacher = _require_module(teacher, "teacher")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return teacher


def assert_teacher_frozen(teacher: nn.Module) -> None:
    teacher = _require_module(teacher, "teacher")
    if any(module.training for module in teacher.modules()):
        raise RuntimeError("teacher and nested modules must remain in eval mode")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("teacher parameters must have requires_grad=False")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("teacher parameter gradients must be None")


def bernoulli_kl_from_logits(
    teacher_logits: Tensor,
    student_logits: Tensor,
    temperature: float,
    epsilon: float,
) -> Tensor:
    teacher_logits, student_logits = _require_logits_pair(
        teacher_logits,
        student_logits,
    )
    temperature = _require_positive_scalar(temperature, "temperature")
    epsilon = _require_epsilon(epsilon)

    teacher_probability = _clamp_float32_probability(
        torch.sigmoid(
            teacher_logits.detach().to(torch.float32) / temperature
        ),
        epsilon,
    )
    student_probability = _clamp_float32_probability(
        torch.sigmoid(student_logits.to(torch.float32) / temperature),
        epsilon,
    )
    positive = teacher_probability * (
        teacher_probability.log() - student_probability.log()
    )
    negative = (1.0 - teacher_probability) * (
        (1.0 - teacher_probability).log()
        - (1.0 - student_probability).log()
    )
    loss = (positive + negative).mean()
    _require_finite_output(loss, "Bernoulli KL")
    return loss


def distillation_losses(
    teacher_feature: Tensor,
    student_feature: Tensor,
    teacher_logits: Tensor,
    student_logits: Tensor,
    temperature: float,
    lambda_feature: float,
    lambda_logit: float,
    valid_sample_mask: Tensor,
) -> DistillationLosses:
    teacher_feature, student_feature = _require_rank_four_pair(
        teacher_feature,
        student_feature,
        "feature",
    )
    teacher_logits, student_logits = _require_rank_four_pair(
        teacher_logits,
        student_logits,
        "logits",
    )
    batch = teacher_feature.shape[0]
    if teacher_logits.shape[0] != batch:
        raise ValueError("feature and logit tensors must share batch size")
    if teacher_logits.device != teacher_feature.device:
        raise ValueError("all feature and logit tensors must share a device")
    if not isinstance(valid_sample_mask, Tensor):
        raise ValueError("valid_sample_mask must be a tensor")
    if valid_sample_mask.shape != (batch,):
        raise ValueError("valid_sample_mask must have shape [B]")
    if valid_sample_mask.dtype != torch.bool:
        raise ValueError("valid_sample_mask must be boolean")
    if valid_sample_mask.device != teacher_feature.device:
        raise ValueError(
            "valid_sample_mask device must match distillation tensors"
        )

    temperature = _require_positive_scalar(temperature, "temperature")
    lambda_feature = _require_nonnegative_scalar(
        lambda_feature,
        "lambda_feature",
    )
    lambda_logit = _require_nonnegative_scalar(lambda_logit, "lambda_logit")

    if not valid_sample_mask.any().item():
        raise RuntimeError("no valid samples for distillation")

    valid_teacher_feature = teacher_feature[valid_sample_mask].detach()
    valid_student_feature = student_feature[valid_sample_mask]
    valid_teacher_logits = teacher_logits[valid_sample_mask].detach()
    valid_student_logits = student_logits[valid_sample_mask]

    for name, value in (
        ("teacher_feature", valid_teacher_feature),
        ("student_feature", valid_student_feature),
        ("teacher_logits", valid_teacher_logits),
        ("student_logits", valid_student_logits),
    ):
        _require_finite(value, name)

    feature_dtype = torch.promote_types(
        valid_student_feature.dtype,
        torch.float32,
    )
    feature = F.mse_loss(
        valid_student_feature.to(feature_dtype),
        valid_teacher_feature.to(feature_dtype),
        reduction="none",
    ).flatten(start_dim=1).mean(dim=1).mean()
    bernoulli = bernoulli_kl_from_logits(
        valid_teacher_logits,
        valid_student_logits,
        temperature=temperature,
        epsilon=_DISTILLATION_EPSILON,
    )
    total = (
        lambda_feature * feature
        + lambda_logit * (temperature**2) * bernoulli
    )
    for name, value in (
        ("feature loss", feature),
        ("Bernoulli loss", bernoulli),
        ("total distillation loss", total),
    ):
        _require_finite_output(value, name)
    return DistillationLosses(
        feature=feature,
        bernoulli=bernoulli,
        total=total,
    )
