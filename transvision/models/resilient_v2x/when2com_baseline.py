"""Controlled When2com-style fusion for aligned LiDAR and camera BEV features.

This is an independent, protocol-controlled adaptation of the query/key
communication idea from When2com.  It is not an exact reproduction of the
source-paper implementation.  The four aligned inputs use the fixed order
``[L_E, L_R, C_E, C_R]`` and are first reduced to ego and RSU agent features.
Unsupported inputs are masked before every learned operation.
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


def _positive_float(name: str, value: object) -> float:
    result = _nonnegative_float(name, value)
    if result == 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


@MODELS.register_module()
class When2comAdaptedFusion(nn.Module):
    """Two-agent query/key communication over multimodal BEV features.

    The ego query selects between ego and RSU keys whenever ego is available.
    If ego is unavailable, the RSU query is used instead.  This preserves a
    useful fallback for protocol samples that contain only RSU support while
    keeping the standard vehicle-centric communication direction.
    """

    branch_count = len(BRANCH_ORDER)
    agent_count = len(AGENT_ORDER)

    def __init__(
        self,
        channels: int = 256,
        *,
        query_key_channels: int = 64,
        age_decay: float = 0.25,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.query_key_channels = _positive_int(
            "query_key_channels",
            query_key_channels,
        )
        self.age_decay = _nonnegative_float("age_decay", age_decay)
        self.temperature = _positive_float("temperature", temperature)

        # Bias-free projections keep an unsupported, zeroed agent descriptor
        # exactly neutral even before the explicit support mask is applied.
        self.query_projection = nn.Linear(
            self.channels,
            self.query_key_channels,
            bias=False,
        )
        self.key_projection = nn.Linear(
            self.channels,
            self.query_key_channels,
            bias=False,
        )

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

    @staticmethod
    def _as_agents(values: Tensor) -> Tensor:
        """Map branch order [L_E,L_R,C_E,C_R] to [agent, modality]."""

        return torch.stack(
            (
                values[:, (0, 2)],
                values[:, (1, 3)],
            ),
            dim=1,
        )

    def _aggregate_agents(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        agent_branches = self._as_agents(branches)
        agent_branch_support = self._as_agents(support)
        agent_branch_ages = self._as_agents(ages)

        age_logits = -self.age_decay * agent_branch_ages.float()
        floor = torch.finfo(age_logits.dtype).min
        masked_logits = age_logits.masked_fill(~agent_branch_support, floor)
        centered_logits = masked_logits - masked_logits.amax(
            dim=2,
            keepdim=True,
        )
        raw_weights = (torch.exp(centered_logits) * agent_branch_support).to(
            dtype=branches.dtype
        )
        weight_sum = raw_weights.sum(dim=2)
        normalized = raw_weights / weight_sum.clamp_min(1.0)[:, :, None]
        agent_features = (agent_branches * normalized[:, :, :, None, None, None]).sum(
            dim=2
        )
        agent_support = agent_branch_support.any(dim=2)
        agent_ages = (
            agent_branch_ages * normalized.to(dtype=agent_branch_ages.dtype)
        ).sum(dim=2)
        return agent_features, agent_support, agent_ages

    def _communication_weights(
        self,
        agent_features: Tensor,
        agent_support: Tensor,
        agent_ages: Tensor,
    ) -> Tensor:
        descriptors = agent_features.mean(dim=(-2, -1))
        queries = self.query_projection(descriptors)
        keys = self.key_projection(descriptors)

        # Prefer a vehicle-centric receiver, with an RSU-only fallback.
        receiver_index = (~agent_support[:, 0]).to(dtype=torch.long)
        receiver_query = queries.gather(
            1,
            receiver_index[:, None, None].expand(
                -1,
                1,
                self.query_key_channels,
            ),
        ).squeeze(1)
        logits = torch.einsum("bd,bad->ba", receiver_query, keys)
        logits = logits / math.sqrt(self.query_key_channels)
        logits = logits / self.temperature
        logits = logits - self.age_decay * agent_ages.to(dtype=logits.dtype)
        logits = logits.masked_fill(~agent_support, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        return torch.where(agent_support, weights, torch.zeros_like(weights))

    def communication_weights(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        """Return the masked ego/RSU communication weights for diagnostics."""

        branches, support, ages = self._prepare_inputs(branches, support, ages)
        agent_features, agent_support, agent_ages = self._aggregate_agents(
            branches,
            support,
            ages,
        )
        return self._communication_weights(
            agent_features,
            agent_support,
            agent_ages,
        )

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
        weights = self._communication_weights(
            agent_features,
            agent_support,
            agent_ages,
        )
        return (agent_features * weights[:, :, None, None, None]).sum(dim=1)


__all__ = (
    "AGENT_ORDER",
    "BRANCH_ORDER",
    "When2comAdaptedFusion",
)
