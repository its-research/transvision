"""Clean-room How2comm-style controlled fusion for aligned L+C BEV inputs.

This module is an independent, protocol-controlled adaptation of two high-level
ideas from How2comm (NeurIPS 2023): sparse spatial-channel communication and
lightweight temporal compensation before pragmatic collaboration.  It is not
an exact reproduction of the source-paper system.  In particular, the module
does not implement or depend on ResilientV2X PTF, DER, a teacher, or any
distillation objective.

The input order is fixed to ``[L_E, L_R, C_E, C_R]``.  The caller supplies an
explicit support bit and non-negative age for every branch.  Unsupported values
are removed before learned operations and therefore cannot influence outputs,
selection diagnostics, or gradients.
"""

from __future__ import annotations

import math

import torch
from mmdet3d.registry import MODELS
from torch import Tensor, nn
from torch.nn import functional as F


BRANCH_ORDER = (
    "lidar_ego",
    "lidar_rsu",
    "camera_ego",
    "camera_rsu",
)
AGENT_ORDER = ("ego", "rsu")


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
class How2commAdaptedFusion(nn.Module):
    """Lightweight age-aware spatial-channel communication and L+C fusion.

    LiDAR and camera are first merged within each agent using support-masked,
    age-aware weights.  A small depthwise residual block gives the RSU feature
    an age-conditioned temporal-context correction.  Complementary ego-request
    and RSU-salience scores then select spatial cells and feature channels.  A
    learned residual gate fuses the selected RSU message into the ego feature.

    The temporal block is deliberately a bounded local residual; it performs no
    motion warping and is not the ResilientV2X PTF.  If ego support is absent,
    the RSU becomes the local fallback and is not counted as communication.
    """

    branch_count = len(BRANCH_ORDER)
    _agent_branch_indices = ((0, 2), (1, 3))

    def __init__(
        self,
        channels: int = 256,
        *,
        selection_reduction: int = 4,
        communication_threshold: float = 0.2,
        age_decay: float = 0.25,
        temporal_gain: float = 0.5,
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.selection_reduction = _positive_int(
            "selection_reduction",
            selection_reduction,
        )
        self.communication_threshold = _probability(
            "communication_threshold",
            communication_threshold,
        )
        self.age_decay = _nonnegative_float("age_decay", age_decay)
        self.temporal_gain = _nonnegative_float("temporal_gain", temporal_gain)

        hidden_channels = max(1, self.channels // self.selection_reduction)
        self.temporal_depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=False,
        )
        self.temporal_projection = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False,
        )
        self.spatial_attention = nn.Conv2d(
            self.channels,
            1,
            kernel_size=1,
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.channels, kernel_size=1),
        )
        self.fusion_gate = nn.Conv2d(
            2 * self.channels,
            self.channels,
            kernel_size=1,
        )

        self._last_spatial_mask: Tensor | None = None
        self._last_channel_mask: Tensor | None = None
        self._last_communication_ratio: Tensor | None = None

    @property
    def last_spatial_mask(self) -> Tensor | None:
        """Detached RSU spatial transmission mask from the latest pass."""

        return self._last_spatial_mask

    @property
    def last_channel_mask(self) -> Tensor | None:
        """Detached RSU channel transmission mask from the latest pass."""

        return self._last_channel_mask

    @property
    def last_communication_ratio(self) -> Tensor | None:
        """Detached selected fraction of the RSU ``H x W x C`` message."""

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

    def _aggregate_agents(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        agent_features: list[Tensor] = []
        agent_supports: list[Tensor] = []
        agent_ages: list[Tensor] = []
        for indices in self._agent_branch_indices:
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
            age = (modality_ages * weights).sum(dim=1)
            age = torch.where(agent_support, age, torch.zeros_like(age))
            agent_features.append(feature)
            agent_supports.append(agent_support)
            agent_ages.append(age)
        return (
            torch.stack(agent_features, dim=1),
            torch.stack(agent_supports, dim=1),
            torch.stack(agent_ages, dim=1),
        )

    def _temporal_context(
        self,
        remote: Tensor,
        remote_support: Tensor,
        remote_age: Tensor,
    ) -> Tensor:
        residual = self.temporal_projection(
            F.silu(self.temporal_depthwise(remote))
        )
        age_scale = self.temporal_gain * remote_age / (1.0 + remote_age)
        corrected = remote + residual * age_scale[:, None, None, None]
        return torch.where(
            remote_support[:, None, None, None],
            corrected,
            torch.zeros_like(corrected),
        )

    def _communication_selection(
        self,
        local: Tensor,
        remote: Tensor,
        communicating: Tensor,
        remote_age: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        local_spatial_request = 1.0 - torch.sigmoid(
            self.spatial_attention(local)
        )
        remote_spatial_salience = torch.sigmoid(self.spatial_attention(remote))
        local_channel_request = 1.0 - torch.sigmoid(
            self.channel_attention(local)
        )
        remote_channel_salience = torch.sigmoid(
            self.channel_attention(remote)
        )
        recency = torch.exp(
            -self.age_decay * remote_age.float()
        ).to(dtype=remote.dtype)
        spatial_probability = (
            local_spatial_request
            * remote_spatial_salience
            * recency[:, None, None, None]
        )
        channel_probability = (
            local_channel_request
            * remote_channel_salience
            * recency[:, None, None, None]
        )
        pair_mask = communicating[:, None, None, None]
        spatial_probability = torch.where(
            pair_mask,
            spatial_probability,
            torch.zeros_like(spatial_probability),
        )
        channel_probability = torch.where(
            pair_mask,
            channel_probability,
            torch.zeros_like(channel_probability),
        )

        spatial_mask = (
            spatial_probability >= self.communication_threshold
        ) & pair_mask
        channel_mask = (
            channel_probability >= self.communication_threshold
        ) & pair_mask
        spatial_gate = spatial_mask.to(remote.dtype)
        channel_gate = channel_mask.to(remote.dtype)
        if self.training:
            # Straight-through masks preserve discrete communication semantics
            # while allowing the compact selectors to learn from detection loss.
            spatial_gate = (
                spatial_gate
                + spatial_probability
                - spatial_probability.detach()
            )
            channel_gate = (
                channel_gate
                + channel_probability
                - channel_probability.detach()
            )
        selected = remote * spatial_gate * channel_gate
        return selected, spatial_mask[:, 0], channel_mask[:, :, 0, 0], recency

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(branches, support, ages)
        agents, agent_support, agent_ages = self._aggregate_agents(
            branches,
            support,
            ages,
        )
        ego = agents[:, 0]
        rsu = self._temporal_context(
            agents[:, 1],
            agent_support[:, 1],
            agent_ages[:, 1],
        )
        ego_supported = agent_support[:, 0]
        communicating = ego_supported & agent_support[:, 1]
        local = torch.where(
            ego_supported[:, None, None, None],
            ego,
            rsu,
        )
        selected, spatial_mask, channel_mask, _ = self._communication_selection(
            local,
            rsu,
            communicating,
            agent_ages[:, 1],
        )
        gate = torch.sigmoid(self.fusion_gate(torch.cat((local, selected), dim=1)))
        fused = local + gate * selected
        fused = torch.where(
            communicating[:, None, None, None],
            fused,
            local,
        )

        joint_mask = spatial_mask[:, None, :, :] & channel_mask[:, :, None, None]
        self._last_spatial_mask = spatial_mask.detach()
        self._last_channel_mask = channel_mask.detach()
        self._last_communication_ratio = joint_mask.flatten(1).float().mean(
            dim=1
        ).detach()
        return fused


__all__ = ("How2commAdaptedFusion",)
