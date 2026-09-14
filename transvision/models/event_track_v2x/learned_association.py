"""Trainable cross-agent association head for EventTrack-V2X development.

The model consumes the fixed node schema from :mod:`learning_contracts` and
produces pair logits plus one unmatched (dustbin) logit per object.  Training
combines pairwise BCE with row-wise and column-wise assignment cross entropy;
inference keeps explicit unmatched mass and Top-H identity hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .association import Assignment, solve_one_to_one
from .learning_contracts import (
    ASSOCIATION_FEATURE_DIM_V1,
    AssociationTrainingConfigV1,
)


class AssociationFrameLike(Protocol):
    left_features: np.ndarray
    right_features: np.ndarray
    targets: np.ndarray
    right_identity_ids: tuple[str, ...]


@dataclass(frozen=True)
class AssociationLogitsV1:
    pair_logits: Tensor
    unmatched_left_logits: Tensor
    unmatched_right_logits: Tensor


@dataclass(frozen=True)
class AssociationLossV1:
    total: Tensor
    pairwise_bce: Tensor
    left_assignment: Tensor
    right_assignment: Tensor
    bidirectional_assignment: Tensor


@dataclass(frozen=True, slots=True)
class TopHIdentityV1:
    identity_id: str
    probability: float


@dataclass(frozen=True, slots=True)
class LeftIdentityDistributionV1:
    left_index: int
    hypotheses: tuple[TopHIdentityV1, ...]
    unmatched_probability: float
    other_probability: float

    def __post_init__(self) -> None:
        total = (
            sum(item.probability for item in self.hypotheses) + self.other_probability
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Top-H hypotheses and other mass must sum to one")
        if (
            self.unmatched_probability < 0.0
            or self.unmatched_probability > self.other_probability + 1e-7
        ):
            raise ValueError("unmatched mass must be contained in other mass")


@dataclass(frozen=True, slots=True)
class LearnedAssociationInferenceV1:
    assignment: Assignment
    pair_probabilities: np.ndarray
    unmatched_left_probabilities: np.ndarray
    unmatched_right_probabilities: np.ndarray
    left_identity_distributions: tuple[LeftIdentityDistributionV1, ...]


@dataclass(frozen=True, slots=True)
class AssociationEpochMetricsV1:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_pair_accuracy: float
    validation_assignment_accuracy: float


@dataclass(frozen=True)
class AssociationTrainingResultV1:
    model: "LearnedAssociationModelV1"
    history: tuple[AssociationEpochMetricsV1, ...]

    @property
    def final_metrics(self) -> dict[str, float]:
        final = self.history[-1]
        return {
            "train_loss": final.train_loss,
            "validation_assignment_accuracy": final.validation_assignment_accuracy,
            "validation_loss": final.validation_loss,
            "validation_pair_accuracy": final.validation_pair_accuracy,
        }


class LearnedAssociationModelV1(nn.Module):
    """Symmetric node encoder with learned pair and dustbin scorers."""

    def __init__(
        self,
        *,
        feature_dim: int = ASSOCIATION_FEATURE_DIM_V1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim != ASSOCIATION_FEATURE_DIM_V1:
            raise ValueError(
                f"feature_dim must equal frozen schema dimension {ASSOCIATION_FEATURE_DIM_V1}"
            )
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        self.left_dustbin = nn.Linear(hidden_dim, 1)
        self.right_dustbin = nn.Linear(hidden_dim, 1)

    def _features(self, value: Tensor, name: str) -> Tensor:
        if value.ndim != 2 or value.shape[1] != self.feature_dim:
            raise ValueError(f"{name} must have shape (N, {self.feature_dim})")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating point values")
        return value

    def forward(
        self, left_features: Tensor, right_features: Tensor
    ) -> AssociationLogitsV1:
        left_features = self._features(left_features, "left_features")
        right_features = self._features(right_features, "right_features")
        if left_features.device != right_features.device:
            raise ValueError("left and right features must use the same device")
        left = self.node_encoder(left_features)
        right = self.node_encoder(right_features)
        left_dustbin = self.left_dustbin(left).squeeze(-1)
        right_dustbin = self.right_dustbin(right).squeeze(-1)
        left_count, right_count = left.shape[0], right.shape[0]
        if left_count == 0 or right_count == 0:
            pair_logits = left.new_empty((left_count, right_count))
        else:
            expanded_left = left[:, None, :].expand(-1, right_count, -1)
            expanded_right = right[None, :, :].expand(left_count, -1, -1)
            pair_features = torch.cat(
                (
                    expanded_left,
                    expanded_right,
                    torch.abs(expanded_left - expanded_right),
                    expanded_left * expanded_right,
                ),
                dim=-1,
            )
            pair_logits = self.pair_scorer(pair_features).squeeze(-1)
        return AssociationLogitsV1(pair_logits, left_dustbin, right_dustbin)


def _validate_targets(targets: Tensor, shape: torch.Size) -> Tensor:
    if targets.shape != shape or targets.ndim != 2:
        raise ValueError(f"targets must have shape {tuple(shape)}")
    if not torch.is_floating_point(targets):
        targets = targets.float()
    if not torch.isfinite(targets).all() or not torch.all(
        (targets == 0) | (targets == 1)
    ):
        raise ValueError("targets must be a finite binary matrix")
    if targets.numel() and (
        torch.any(targets.sum(dim=1) > 1) or torch.any(targets.sum(dim=0) > 1)
    ):
        raise ValueError("targets must describe a one-to-one assignment")
    return targets


def association_loss_v1(
    logits: AssociationLogitsV1,
    targets: Tensor,
    *,
    pairwise_bce_weight: float = 1.0,
    assignment_weight: float = 1.0,
    max_positive_weight: float = 20.0,
) -> AssociationLossV1:
    """Pair BCE plus bidirectional dustbin-aware assignment cross entropy."""

    targets = _validate_targets(targets, logits.pair_logits.shape).to(
        device=logits.pair_logits.device,
        dtype=logits.pair_logits.dtype,
    )
    if targets.numel():
        positive_count = float(targets.sum().detach().cpu())
        negative_count = float(targets.numel() - positive_count)
        ratio = negative_count / max(positive_count, 1.0)
        positive_weight = logits.pair_logits.new_tensor(
            min(float(max_positive_weight), max(1.0, ratio))
        )
        pairwise_bce = F.binary_cross_entropy_with_logits(
            logits.pair_logits,
            targets,
            pos_weight=positive_weight,
        )
    else:
        pairwise_bce = (
            logits.pair_logits.sum()
            + logits.unmatched_left_logits.sum() * 0.0
            + logits.unmatched_right_logits.sum() * 0.0
        )

    left_count, right_count = targets.shape
    if left_count:
        left_labels = torch.full(
            (left_count,), right_count, dtype=torch.long, device=targets.device
        )
        positive_rows, positive_columns = torch.where(targets > 0.5)
        left_labels[positive_rows] = positive_columns
        left_logits = torch.cat(
            (logits.pair_logits, logits.unmatched_left_logits[:, None]), dim=1
        )
        left_assignment = F.cross_entropy(left_logits, left_labels)
    else:
        left_assignment = logits.unmatched_left_logits.sum() * 0.0

    if right_count:
        right_labels = torch.full(
            (right_count,), left_count, dtype=torch.long, device=targets.device
        )
        positive_rows, positive_columns = torch.where(targets > 0.5)
        right_labels[positive_columns] = positive_rows
        right_logits = torch.cat(
            (
                logits.pair_logits.transpose(0, 1),
                logits.unmatched_right_logits[:, None],
            ),
            dim=1,
        )
        right_assignment = F.cross_entropy(right_logits, right_labels)
    else:
        right_assignment = logits.unmatched_right_logits.sum() * 0.0

    directions = int(left_count > 0) + int(right_count > 0)
    bidirectional = (
        (left_assignment + right_assignment) / directions
        if directions
        else pairwise_bce * 0.0
    )
    total = (
        float(pairwise_bce_weight) * pairwise_bce
        + float(assignment_weight) * bidirectional
    )
    return AssociationLossV1(
        total=total,
        pairwise_bce=pairwise_bce,
        left_assignment=left_assignment,
        right_assignment=right_assignment,
        bidirectional_assignment=bidirectional,
    )


def _tensor_frame(
    sample: AssociationFrameLike, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    # Contract arrays are backed by immutable bytes.  Copying here avoids
    # PyTorch's undefined-behaviour warning for non-writeable NumPy buffers.
    left = torch.tensor(sample.left_features, dtype=torch.float32, device=device)
    right = torch.tensor(sample.right_features, dtype=torch.float32, device=device)
    targets = torch.tensor(sample.targets, dtype=torch.float32, device=device)
    return left, right, targets


def infer_association_v1(
    model: LearnedAssociationModelV1,
    left_features: Tensor | np.ndarray,
    right_features: Tensor | np.ndarray,
    *,
    right_identity_ids: Sequence[str],
    top_h: int,
) -> LearnedAssociationInferenceV1:
    """Return exact one-to-one matches and row-normalized Top-H identity mass."""

    if isinstance(top_h, bool) or not isinstance(top_h, int) or top_h <= 0:
        raise ValueError("top_h must be a positive integer")
    device = next(model.parameters()).device
    left = (
        left_features.to(device=device, dtype=torch.float32)
        if isinstance(left_features, Tensor)
        else torch.tensor(left_features, dtype=torch.float32, device=device)
    )
    right = (
        right_features.to(device=device, dtype=torch.float32)
        if isinstance(right_features, Tensor)
        else torch.tensor(right_features, dtype=torch.float32, device=device)
    )
    if len(right_identity_ids) != right.shape[0] or len(set(right_identity_ids)) != len(
        right_identity_ids
    ):
        raise ValueError(
            "right_identity_ids must be unique and align with right features"
        )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(left, right)
        left_softmax = torch.softmax(
            torch.cat(
                (logits.pair_logits, logits.unmatched_left_logits[:, None]), dim=1
            ),
            dim=1,
        )
        right_softmax = torch.softmax(
            torch.cat(
                (
                    logits.pair_logits.transpose(0, 1),
                    logits.unmatched_right_logits[:, None],
                ),
                dim=1,
            ),
            dim=1,
        )
    if was_training:
        model.train()
    left_count, right_count = logits.pair_logits.shape
    if left_count and right_count:
        pair_probabilities = torch.sqrt(
            torch.clamp(left_softmax[:, :right_count], min=0.0)
            * torch.clamp(right_softmax[:, :left_count].transpose(0, 1), min=0.0)
        )
    else:
        pair_probabilities = logits.pair_logits.new_empty((left_count, right_count))
    unmatched_left = (
        left_softmax[:, right_count] if left_count else left.new_empty((0,))
    )
    unmatched_right = (
        right_softmax[:, left_count] if right_count else right.new_empty((0,))
    )
    epsilon = torch.finfo(torch.float32).tiny
    pair_costs = -torch.log(torch.clamp(pair_probabilities, min=epsilon)).cpu().numpy()
    left_costs = -torch.log(torch.clamp(unmatched_left, min=epsilon)).cpu().numpy()
    right_costs = -torch.log(torch.clamp(unmatched_right, min=epsilon)).cpu().numpy()
    assignment = solve_one_to_one(
        pair_costs,
        unmatched_left_cost=left_costs,
        unmatched_right_cost=right_costs,
    )

    distributions: list[LeftIdentityDistributionV1] = []
    row_probabilities = left_softmax.cpu().numpy()
    for left_index in range(left_count):
        ranked = sorted(
            (
                (right_index, float(row_probabilities[left_index, right_index]))
                for right_index in range(right_count)
            ),
            key=lambda item: (-item[1], right_identity_ids[item[0]]),
        )[:top_h]
        hypotheses = tuple(
            TopHIdentityV1(right_identity_ids[right_index], probability)
            for right_index, probability in ranked
        )
        kept_probability = sum(item.probability for item in hypotheses)
        unmatched_probability = float(row_probabilities[left_index, right_count])
        distributions.append(
            LeftIdentityDistributionV1(
                left_index=left_index,
                hypotheses=hypotheses,
                unmatched_probability=unmatched_probability,
                other_probability=max(0.0, 1.0 - kept_probability),
            )
        )
    pair_array = np.asarray(pair_probabilities.cpu(), dtype=np.float64)
    left_array = np.asarray(unmatched_left.cpu(), dtype=np.float64)
    right_array = np.asarray(unmatched_right.cpu(), dtype=np.float64)
    for array in (pair_array, left_array, right_array):
        array.setflags(write=False)
    return LearnedAssociationInferenceV1(
        assignment=assignment,
        pair_probabilities=pair_array,
        unmatched_left_probabilities=left_array,
        unmatched_right_probabilities=right_array,
        left_identity_distributions=tuple(distributions),
    )


def evaluate_association_model_v1(
    model: LearnedAssociationModelV1,
    samples: Sequence[AssociationFrameLike],
    *,
    config: AssociationTrainingConfigV1,
    device: torch.device,
) -> tuple[float, float, float]:
    if not samples:
        raise ValueError("validation samples must not be empty")
    was_training = model.training
    model.eval()
    losses: list[float] = []
    pair_correct = 0
    pair_total = 0
    assignment_correct = 0
    with torch.no_grad():
        for sample in samples:
            left, right, targets = _tensor_frame(sample, device)
            logits = model(left, right)
            loss = association_loss_v1(
                logits,
                targets,
                pairwise_bce_weight=config.pairwise_bce_weight,
                assignment_weight=config.assignment_weight,
                max_positive_weight=config.max_positive_weight,
            )
            losses.append(float(loss.total.detach().cpu()))
            if targets.numel():
                predicted = logits.pair_logits >= 0.0
                pair_correct += int((predicted == (targets > 0.5)).sum().cpu())
                pair_total += targets.numel()
            inferred = infer_association_v1(
                model,
                left,
                right,
                right_identity_ids=sample.right_identity_ids,
                top_h=config.top_h,
            )
            expected = {
                (int(row), int(column))
                for row, column in zip(*np.where(np.asarray(sample.targets) == 1))
            }
            assignment_correct += int(set(inferred.assignment.pairs) == expected)
    if was_training:
        model.train()
    return (
        float(sum(losses) / len(losses)),
        float(pair_correct / pair_total) if pair_total else 1.0,
        float(assignment_correct / len(samples)),
    )


def train_association_model_v1(
    train_samples: Sequence[AssociationFrameLike],
    validation_samples: Sequence[AssociationFrameLike],
    *,
    config: AssociationTrainingConfigV1,
    seed: int,
    device: str | torch.device = "cpu",
) -> AssociationTrainingResultV1:
    """Run a real autograd/optimizer loop over train-derived frame matrices."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not train_samples or not validation_samples:
        raise ValueError("train and validation samples must both be non-empty")
    used_device = torch.device(device)
    if used_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = LearnedAssociationModelV1(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(used_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history: list[AssociationEpochMetricsV1] = []
    for epoch in range(config.epochs):
        model.train()
        ordering = torch.randperm(len(train_samples), generator=generator).tolist()
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        pending = 0
        for position, sample_index in enumerate(ordering):
            left, right, targets = _tensor_frame(
                train_samples[sample_index], used_device
            )
            logits = model(left, right)
            loss = association_loss_v1(
                logits,
                targets,
                pairwise_bce_weight=config.pairwise_bce_weight,
                assignment_weight=config.assignment_weight,
                max_positive_weight=config.max_positive_weight,
            )
            if not torch.isfinite(loss.total):
                raise RuntimeError("association training produced a non-finite loss")
            (loss.total / config.frames_per_step).backward()
            pending += 1
            step_losses.append(float(loss.total.detach().cpu()))
            final_sample = position + 1 == len(ordering)
            if pending == config.frames_per_step or final_sample:
                if pending != config.frames_per_step:
                    scale = config.frames_per_step / pending
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        validation_loss, pair_accuracy, assignment_accuracy = (
            evaluate_association_model_v1(
                model,
                validation_samples,
                config=config,
                device=used_device,
            )
        )
        history.append(
            AssociationEpochMetricsV1(
                epoch=epoch,
                train_loss=float(sum(step_losses) / len(step_losses)),
                validation_loss=validation_loss,
                validation_pair_accuracy=pair_accuracy,
                validation_assignment_accuracy=assignment_accuracy,
            )
        )
    return AssociationTrainingResultV1(model=model, history=tuple(history))


__all__ = [
    "AssociationEpochMetricsV1",
    "AssociationLogitsV1",
    "AssociationLossV1",
    "AssociationTrainingResultV1",
    "LearnedAssociationInferenceV1",
    "LearnedAssociationModelV1",
    "LeftIdentityDistributionV1",
    "TopHIdentityV1",
    "association_loss_v1",
    "evaluate_association_model_v1",
    "infer_association_v1",
    "train_association_model_v1",
]
