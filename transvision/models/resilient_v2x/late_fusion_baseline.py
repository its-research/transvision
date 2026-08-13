"""Controlled late-feature L+C fusion for aligned ego and RSU BEV inputs.

``LateFusionStyleFusion`` is the project's
``LateFusion-style L+C (controlled late-feature adaptation; not exact
source-paper reproduction)`` baseline.  It is deliberately not presented as
box-level late fusion or as a bit-exact reproduction of another codebase.

The input order is fixed to ``[L_E, L_R, C_E, C_R]``.  LiDAR and camera are
first fused independently inside each agent.  A support-aware, age-aware gate
then combines the resulting ego and RSU feature maps.  Unsupported branches
are removed before learned operations, and an all-missing sample is rejected.
"""

from __future__ import annotations

import math

import torch
from mmdet3d.registry import MODELS
from torch import Tensor, nn


BRANCH_ORDER = (
    "lidar_ego",
    "lidar_rsu",
    "camera_ego",
    "camera_rsu",
)
AGENT_ORDER = ("ego", "rsu")
_AGENT_BRANCH_INDICES = ((0, 2), (1, 3))


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_float(name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@MODELS.register_module()
class LateFusionStyleFusion(nn.Module):
    """Hierarchical agent-late feature fusion under the controlled protocol.

    This module operates on aligned BEV features, not detection boxes.  Within
    each agent, supported LiDAR and camera branches are combined with normalized
    exponential age weights.  The resulting ego and RSU features receive
    learned per-cell gate scores with an explicit age penalty.  Support masking
    is applied before both stages and after the final softmax.
    """

    branch_count = len(BRANCH_ORDER)
    agent_count = len(AGENT_ORDER)

    def __init__(
        self,
        channels: int = 256,
        *,
        age_decay: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.age_decay = _nonnegative_float("age_decay", age_decay)

        # The same scoring function is used for both agents so that agent
        # identity is represented only by the explicit, trainable bias.
        self.gate_projection = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.agent_bias = nn.Parameter(torch.zeros(1, self.agent_count, 1, 1))
        self._last_gate_weights: Tensor | None = None

    @property
    def last_gate_weights(self) -> Tensor | None:
        """Detached ``[B,2,H,W]`` ego/RSU weights from the latest forward."""

        return self._last_gate_weights

    def _prepare_inputs(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(branches, Tensor) or branches.ndim != 5:
            raise ValueError("branches must be a tensor with shape [B,4,C,H,W]")
        batch, branch_count, channels, height, width = branches.shape
        if batch <= 0 or height <= 0 or width <= 0:
            raise ValueError("branches must have positive B, H, and W dimensions")
        if branch_count != self.branch_count or channels != self.channels:
            raise ValueError(f"branches must have shape [B,4,{self.channels},H,W]")
        if not branches.is_floating_point():
            raise ValueError("branches must be floating point")

        metadata_shape = (batch, self.branch_count)
        if (
            not isinstance(support, Tensor)
            or support.shape != metadata_shape
            or support.dtype is not torch.bool
        ):
            raise ValueError("support must be a bool tensor with shape [B,4]")
        if support.device != branches.device:
            raise ValueError("branches and support must share a device")

        all_missing = (~support.any(dim=1)).nonzero(as_tuple=False).flatten()
        if all_missing.numel():
            indices = all_missing.detach().cpu().tolist()
            raise ValueError(
                "every sample must have at least one supported branch; "
                f"all branches are missing for sample indices {indices}"
            )

        if (
            not isinstance(ages, Tensor)
            or ages.shape != metadata_shape
            or not ages.is_floating_point()
        ):
            raise ValueError("ages must be a floating tensor with shape [B,4]")
        if ages.device != branches.device:
            raise ValueError("branches and ages must share a device")

        supported_ages = ages.masked_select(support)
        if not bool(torch.isfinite(supported_ages).all()) or bool(
            (supported_ages < 0).any()
        ):
            raise ValueError(
                "ages of supported branches must be finite and non-negative"
            )

        branch_mask = support[:, :, None, None, None]
        supported_values = branches.masked_select(branch_mask.expand_as(branches))
        if not bool(torch.isfinite(supported_values).all()):
            raise ValueError("supported branches must contain only finite values")

        clean_branches = torch.where(
            branch_mask,
            branches,
            torch.zeros((), dtype=branches.dtype, device=branches.device),
        )
        clean_ages = torch.where(
            support,
            ages,
            torch.zeros((), dtype=ages.dtype, device=ages.device),
        ).to(dtype=branches.dtype)
        return clean_branches, support, clean_ages

    def _merge_modalities(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Fuse supported L+C branches independently within ego and RSU."""

        agent_features: list[Tensor] = []
        agent_supports: list[Tensor] = []
        agent_ages: list[Tensor] = []
        for indices in _AGENT_BRANCH_INDICES:
            modality_support = support[:, indices]
            modality_ages = ages[:, indices]
            agent_support = modality_support.any(dim=1)

            logits = -self.age_decay * modality_ages.float()
            logits = logits.masked_fill(~modality_support, -torch.inf)
            safe_logits = torch.where(
                agent_support[:, None],
                logits,
                torch.zeros_like(logits),
            )
            weights = torch.softmax(safe_logits, dim=1)
            weights = torch.where(
                modality_support,
                weights,
                torch.zeros_like(weights),
            ).to(dtype=branches.dtype)

            feature = (branches[:, indices] * weights[:, :, None, None, None]).sum(
                dim=1
            )
            feature = torch.where(
                agent_support[:, None, None, None],
                feature,
                torch.zeros_like(feature),
            )
            effective_age = (modality_ages * weights).sum(dim=1)
            effective_age = torch.where(
                agent_support,
                effective_age,
                torch.zeros_like(effective_age),
            )

            agent_features.append(feature)
            agent_supports.append(agent_support)
            agent_ages.append(effective_age)

        return (
            torch.stack(agent_features, dim=1),
            torch.stack(agent_supports, dim=1),
            torch.stack(agent_ages, dim=1),
        )

    def _gate_weights(
        self,
        agent_features: Tensor,
        agent_support: Tensor,
        agent_ages: Tensor,
    ) -> Tensor:
        batch, agent_count, _, height, width = agent_features.shape
        scores = self.gate_projection(
            agent_features.reshape(
                batch * agent_count,
                self.channels,
                height,
                width,
            )
        ).reshape(batch, agent_count, height, width)
        scores = (
            scores
            + self.agent_bias
            - self.age_decay * agent_ages[:, :, None, None].to(dtype=scores.dtype)
        )
        support_map = agent_support[:, :, None, None]
        scores = scores.masked_fill(~support_map, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        return torch.where(support_map, weights, torch.zeros_like(weights))

    def gate_weights(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        """Return support-masked terminal ego/RSU gate weights."""

        branches, support, ages = self._prepare_inputs(branches, support, ages)
        agent_features, agent_support, agent_ages = self._merge_modalities(
            branches,
            support,
            ages,
        )
        return self._gate_weights(agent_features, agent_support, agent_ages)

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(branches, support, ages)
        agent_features, agent_support, agent_ages = self._merge_modalities(
            branches,
            support,
            ages,
        )
        weights = self._gate_weights(
            agent_features,
            agent_support,
            agent_ages,
        )
        fused = (
            agent_features * weights[:, :, None, :, :].to(agent_features.dtype)
        ).sum(dim=1)
        self._last_gate_weights = weights.detach()
        return fused


__all__ = ("LateFusionStyleFusion",)
