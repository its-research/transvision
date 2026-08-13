"""Clean-room Where2comm-style controlled L+C fusion adaptation.

This module implements only the high-level communication idea described by
Where2comm: spatial confidence maps decide which remote BEV cells communicate,
then supported agent features are fused with masked attention.  It is an
independent controlled adaptation for this project's already-aligned four
branches and is not a source-code or bit-exact reproduction.

Branch order is fixed to ego LiDAR, RSU LiDAR, ego camera, RSU camera.  The two
modalities are first aggregated within each agent.  Ego cells are always local;
RSU cells are admitted only when their learned confidence reaches the configured
threshold.  Unsupported values are removed before every learned operation.
"""

from __future__ import annotations

import math

import torch
from mmdet3d.registry import MODELS
from torch import Tensor, nn


AGENT_ORDER = ("ego", "rsu")
BRANCH_ORDER = (
    "lidar_ego",
    "lidar_rsu",
    "camera_ego",
    "camera_rsu",
)


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


def _probability(name: str, value: object) -> float:
    result = _nonnegative_float(name, value)
    if result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@MODELS.register_module()
class Where2commAdaptedFusion(nn.Module):
    """Support-safe spatial communication and attention over ego/RSU BEV.

    ``last_communication_mask`` has shape ``[B,2,H,W]`` in ``AGENT_ORDER``.
    Ego is local and therefore always selected when supported.  The RSU slice
    is the actual learned spatial transmission mask.  The per-sample
    ``last_communication_ratio`` is the transmitted fraction of RSU BEV cells;
    it is zero when RSU has no supported modality.
    """

    branch_count = len(BRANCH_ORDER)
    _agent_branch_indices = ((0, 2), (1, 3))

    def __init__(
        self,
        channels: int = 256,
        communication_threshold: float = 0.5,
        age_decay: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.communication_threshold = _probability(
            "communication_threshold",
            communication_threshold,
        )
        self.age_decay = _nonnegative_float("age_decay", age_decay)
        self.confidence_head = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.agent_bias = nn.Parameter(torch.zeros(1, len(AGENT_ORDER), 1, 1))
        self._last_communication_mask: Tensor | None = None
        self._last_communication_ratio: Tensor | None = None

    @property
    def last_communication_mask(self) -> Tensor | None:
        """Detached communication mask from the most recent forward pass."""

        return self._last_communication_mask

    @property
    def last_communication_ratio(self) -> Tensor | None:
        """Detached per-sample RSU transmission ratio from the latest pass."""

        return self._last_communication_ratio

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
        missing = (~support.any(dim=1)).nonzero(as_tuple=False).flatten()
        if missing.numel():
            indices = missing.detach().cpu().tolist()
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

    def _aggregate_agents(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        features: list[Tensor] = []
        agent_supports: list[Tensor] = []
        agent_ages: list[Tensor] = []
        for indices in self._agent_branch_indices:
            modality_support = support[:, indices]
            modality_ages = ages[:, indices]
            logits = -self.age_decay * modality_ages.float()
            logits = logits.masked_fill(~modality_support, -torch.inf)
            agent_support = modality_support.any(dim=1)
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
            age = (modality_ages * weights).sum(dim=1)
            age = torch.where(agent_support, age, torch.zeros_like(age))
            features.append(feature)
            agent_supports.append(agent_support)
            agent_ages.append(age)
        return (
            torch.stack(features, dim=1),
            torch.stack(agent_supports, dim=1),
            torch.stack(agent_ages, dim=1),
        )

    def _confidence_logits(self, agent_features: Tensor) -> Tensor:
        batch, agent_count, _, height, width = agent_features.shape
        return self.confidence_head(
            agent_features.reshape(
                batch * agent_count,
                self.channels,
                height,
                width,
            )
        ).reshape(batch, agent_count, height, width)

    def _communication_mask(
        self,
        confidence_logits: Tensor,
        agent_support: Tensor,
    ) -> Tensor:
        batch, agent_count, height, width = confidence_logits.shape
        if agent_count != len(AGENT_ORDER):
            raise RuntimeError("confidence logits must contain ego and RSU agents")
        support_map = agent_support[:, :, None, None].expand(
            batch,
            agent_count,
            height,
            width,
        )
        remote_selected = (
            torch.sigmoid(confidence_logits[:, 1]) >= self.communication_threshold
        ) & support_map[:, 1]
        ego_selected = support_map[:, 0]
        # If ego has no modality, RSU is the local fallback rather than a
        # droppable transmission.  This preserves the common non-empty contract.
        remote_selected = remote_selected | (~support_map[:, 0] & support_map[:, 1])
        return torch.stack((ego_selected, remote_selected), dim=1)

    @staticmethod
    def _rsu_communication_ratio(
        communication_mask: Tensor,
        agent_support: Tensor,
    ) -> Tensor:
        batch, _, height, width = communication_mask.shape
        transmitted = communication_mask[:, 1].flatten(1).sum(dim=1).float()
        denominator = torch.full(
            (batch,),
            height * width,
            dtype=torch.float32,
            device=communication_mask.device,
        )
        ratio = transmitted / denominator
        return torch.where(agent_support[:, 1], ratio, torch.zeros_like(ratio))

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(branches, support, ages)
        agent_features, agent_support, agent_ages = self._aggregate_agents(
            branches,
            support,
            ages,
        )
        confidence_logits = self._confidence_logits(agent_features)
        communication_mask = self._communication_mask(
            confidence_logits,
            agent_support,
        )

        scores = (
            confidence_logits
            + self.agent_bias
            - self.age_decay
            * agent_ages[:, :, None, None].to(dtype=confidence_logits.dtype)
        )
        scores = scores.masked_fill(~communication_mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(
            communication_mask,
            weights,
            torch.zeros_like(weights),
        )
        fused = (
            agent_features * weights[:, :, None, :, :].to(agent_features.dtype)
        ).sum(dim=1)

        self._last_communication_mask = communication_mask.detach()
        self._last_communication_ratio = self._rsu_communication_ratio(
            communication_mask,
            agent_support,
        ).detach()
        return fused


__all__ = (
    "AGENT_ORDER",
    "BRANCH_ORDER",
    "Where2commAdaptedFusion",
)
