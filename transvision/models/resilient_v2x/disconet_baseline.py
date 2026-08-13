"""Clean-room DiscoNet-style controlled L+C fusion adaptation.

This module adapts the inference-time collaboration graph idea from
"Learning Distilled Collaboration Graph for Multi-Agent Perception" to the
project's already aligned DAIR-V2X BEV branches.  It is not an exact
source-paper reproduction: the original early-collaboration teacher and its
feature-distillation training objective are outside this detection-only
controlled-baseline contract.

The fixed input order is ``[L_E, L_R, C_E, C_R]``.  Supported LiDAR and camera
branches are first reduced to ego and RSU agent features.  For the selected
receiver, a shared four-layer 1x1 edge encoder predicts one edge value per BEV
cell and sender.  A support-masked softmax then produces the matrix-valued edge
weights used for one-round message aggregation.  Unsupported values are
zeroed before every learned operation and can neither affect the output nor
receive gradients.
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


@MODELS.register_module()
class DiscoNetAdaptedFusion(nn.Module):
    """Two-agent matrix-valued collaboration graph over aligned L+C BEV.

    Args:
        channels: Channel count shared by every input branch.
        edge_hidden_channels: Width of the first edge-encoder hidden layer.
            The next two layers use progressively halved widths before the
            fourth 1x1 convolution emits a cell-wise scalar edge score.
        age_decay: Exponential age penalty used for modality aggregation and
            sender-edge scoring.

    The ego agent is the receiver whenever it has at least one supported
    modality.  An RSU-only sample uses the RSU as its local receiver, keeping
    the common controlled-baseline contract defined for every non-empty
    support pattern.
    """

    branch_count = len(BRANCH_ORDER)
    agent_count = len(AGENT_ORDER)
    _agent_branch_indices = ((0, 2), (1, 3))

    def __init__(
        self,
        channels: int = 256,
        *,
        edge_hidden_channels: int = 128,
        age_decay: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.edge_hidden_channels = _positive_int(
            "edge_hidden_channels",
            edge_hidden_channels,
        )
        self.age_decay = _nonnegative_float("age_decay", age_decay)

        second_hidden = max(1, self.edge_hidden_channels // 2)
        third_hidden = max(1, second_hidden // 2)
        # DiscoGraph predicts an HxW edge matrix by concatenating receiver and
        # aligned sender features, then applying four shared 1x1 convolutions.
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(
                self.channels * 2,
                self.edge_hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                self.edge_hidden_channels,
                second_hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                second_hidden,
                third_hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(third_hidden, 1, kernel_size=1),
        )
        self._last_edge_weights: Tensor | None = None

    @property
    def last_edge_weights(self) -> Tensor | None:
        """Detached ``[B,2,H,W]`` sender weights from the latest forward."""

        return self._last_edge_weights

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
        """Reduce the two supported modalities belonging to each agent."""

        features: list[Tensor] = []
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
            features.append(feature)
            agent_supports.append(agent_support)
            agent_ages.append(age)

        return (
            torch.stack(features, dim=1),
            torch.stack(agent_supports, dim=1),
            torch.stack(agent_ages, dim=1),
        )

    def _edge_weights(
        self,
        agent_features: Tensor,
        agent_support: Tensor,
        agent_ages: Tensor,
    ) -> Tensor:
        """Predict support-masked matrix-valued weights for one receiver."""

        batch, _, _, height, width = agent_features.shape
        receiver_index = (~agent_support[:, 0]).to(dtype=torch.long)
        receiver = agent_features.gather(
            1,
            receiver_index[:, None, None, None, None].expand(
                -1,
                1,
                self.channels,
                height,
                width,
            ),
        ).squeeze(1)

        receiver_by_sender = receiver[:, None].expand(
            -1,
            self.agent_count,
            -1,
            -1,
            -1,
        )
        edge_inputs = torch.cat((receiver_by_sender, agent_features), dim=2)
        logits = self.edge_encoder(
            edge_inputs.reshape(
                batch * self.agent_count,
                self.channels * 2,
                height,
                width,
            )
        ).reshape(batch, self.agent_count, height, width)
        logits = logits - self.age_decay * agent_ages[:, :, None, None].to(
            dtype=logits.dtype
        )
        support_map = agent_support[:, :, None, None]
        logits = logits.masked_fill(~support_map, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        return torch.where(support_map, weights, torch.zeros_like(weights))

    def edge_weights(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        """Return matrix-valued ego/RSU edge weights for diagnostics."""

        branches, support, ages = self._prepare_inputs(branches, support, ages)
        agent_features, agent_support, agent_ages = self._aggregate_agents(
            branches,
            support,
            ages,
        )
        return self._edge_weights(agent_features, agent_support, agent_ages)

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
        weights = self._edge_weights(agent_features, agent_support, agent_ages)
        fused = (
            agent_features * weights[:, :, None, :, :].to(agent_features.dtype)
        ).sum(dim=1)
        self._last_edge_weights = weights.detach()
        return fused


__all__ = (
    "AGENT_ORDER",
    "BRANCH_ORDER",
    "DiscoNetAdaptedFusion",
)
