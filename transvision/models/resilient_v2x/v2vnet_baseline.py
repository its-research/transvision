"""V2VNet-style two-agent fusion for controlled DAIR-V2X comparisons.

This is an independent L+C adaptation of the message-passing idea in
V2VNet.  It is not a source-paper reproduction: the shared experiment
encoders already provide four aligned BEV branches in the fixed order
``[L_E, L_R, C_E, C_R]``.  This module first reduces those branches to ego
and RSU nodes, then applies synchronous two-node ConvGRU message passing.

Unsupported branches are zeroed before every learned operation.  An edge is
active only when both its sender and receiver have at least one supported
modality, and unsupported nodes remain exactly zero at every iteration.
"""

from __future__ import annotations

import math

import torch
from mmdet3d.registry import MODELS
from torch import Tensor, nn


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


def _odd_kernel_size(value: object) -> int:
    result = _positive_int("kernel_size", value)
    if result % 2 == 0:
        raise ValueError("kernel_size must be odd")
    return result


class _ConvGRUCell(nn.Module):
    """Spatial GRU cell with one shared implementation for both directions."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.candidate = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, message: Tensor, hidden: Tensor) -> Tensor:
        reset, update = torch.sigmoid(
            self.gates(torch.cat((message, hidden), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((message, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


@MODELS.register_module()
class V2VNetStyleFusion(nn.Module):
    """Controlled two-node ConvGRU adaptation of V2VNet.

    Args:
        channels: Channel count shared by every aligned BEV branch.
        age_decay: Exponential age penalty used only while merging the L+C
            branches belonging to the same agent.
        message_iterations: Number of synchronous ego/RSU message rounds.
        kernel_size: Odd spatial kernel used by the message encoder and GRU.

    The return value is the updated ego node when ego is supported.  A sample
    with no ego modality falls back to its RSU node, which keeps the common
    controlled-baseline contract defined for every non-empty support pattern.
    """

    branch_count = 4
    agent_count = 2
    _agent_branch_indices = ((0, 2), (1, 3))

    def __init__(
        self,
        channels: int = 256,
        *,
        age_decay: float = 0.25,
        message_iterations: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.age_decay = _nonnegative_float("age_decay", age_decay)
        self.message_iterations = _positive_int(
            "message_iterations", message_iterations
        )
        self.kernel_size = _odd_kernel_size(kernel_size)
        padding = self.kernel_size // 2

        # V2VNet-style pairwise messages condition on both the receiver and
        # sender state.  The same weights are used for ego<-RSU and RSU<-ego.
        self.message_encoder = nn.Sequential(
            nn.Conv2d(
                self.channels * 2,
                self.channels,
                kernel_size=self.kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.ReLU(inplace=False),
        )
        self.conv_gru = _ConvGRUCell(self.channels, self.kernel_size)

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

        expanded_support = support[:, :, None, None, None]
        supported_values = branches.masked_select(expanded_support.expand_as(branches))
        if not bool(torch.isfinite(supported_values).all()):
            raise ValueError("supported branches must contain only finite values")

        clean_branches = torch.where(
            expanded_support,
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
    ) -> tuple[Tensor, Tensor]:
        """Reduce L+C to ego/RSU nodes without observing missing branches."""

        agent_features = []
        agent_support = []
        for lidar_index, camera_index in self._agent_branch_indices:
            indices = (lidar_index, camera_index)
            modality_features = branches[:, indices]
            modality_support = support[:, indices]
            modality_ages = ages[:, indices]
            node_support = modality_support.any(dim=1)

            logits = -self.age_decay * modality_ages.float()
            logits = logits.masked_fill(~modality_support, -torch.inf)
            # A missing agent has two -inf logits.  Replace those logits before
            # softmax, then mask the resulting node back to exact zero.
            safe_logits = torch.where(
                node_support[:, None],
                logits,
                torch.zeros_like(logits),
            )
            weights = torch.softmax(safe_logits, dim=1).to(branches.dtype)
            weights = torch.where(
                modality_support,
                weights,
                torch.zeros_like(weights),
            )
            node = (modality_features * weights[:, :, None, None, None]).sum(dim=1)
            node = node * node_support[:, None, None, None].to(node.dtype)
            agent_features.append(node)
            agent_support.append(node_support)

        return torch.stack(agent_features, dim=1), torch.stack(agent_support, dim=1)

    def _message_pass(self, nodes: Tensor, agent_support: Tensor) -> Tensor:
        """Run synchronous bidirectional updates on the fixed two-node graph."""

        node_mask = agent_support[:, :, None, None, None]
        states = torch.where(node_mask, nodes, torch.zeros_like(nodes))
        for _ in range(self.message_iterations):
            previous = states
            updated = []
            for receiver_index, sender_index in ((0, 1), (1, 0)):
                receiver = previous[:, receiver_index]
                sender = previous[:, sender_index]
                edge_support = (
                    agent_support[:, receiver_index] & agent_support[:, sender_index]
                )[:, None, None, None]
                message = self.message_encoder(torch.cat((receiver, sender), dim=1))
                message = torch.where(edge_support, message, torch.zeros_like(message))
                candidate = self.conv_gru(message, receiver)
                state = torch.where(edge_support, candidate, receiver)
                state = state * agent_support[:, receiver_index, None, None, None].to(
                    state.dtype
                )
                updated.append(state)
            states = torch.stack(updated, dim=1)
        return states

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(branches, support, ages)
        nodes, agent_support = self._merge_modalities(branches, support, ages)
        states = self._message_pass(nodes, agent_support)
        return torch.where(
            agent_support[:, 0, None, None, None],
            states[:, 0],
            states[:, 1],
        )


__all__ = ("V2VNetStyleFusion",)
