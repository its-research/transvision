from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn

from .contracts import Agent, Modality, ProtocolInvariantError
from .fusion import ResilientBatchSelections
from .geometry import BEVGridSpec, align_bev_to_target


CONTROLLED_BRANCH_KEYS = (
    "lidar_ego",
    "lidar_rsu",
    "camera_ego",
    "camera_rsu",
)

_BRANCH_LAYOUT = (
    ("lidar_ego", 0, 0, Agent.EGO, Modality.LIDAR),
    ("lidar_rsu", 0, 1, Agent.RSU, Modality.LIDAR),
    ("camera_ego", 1, 0, Agent.EGO, Modality.CAMERA),
    ("camera_rsu", 1, 1, Agent.RSU, Modality.CAMERA),
)


def _geometry_dtype(feature_dtype: torch.dtype) -> torch.dtype:
    if feature_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return feature_dtype


@dataclass(frozen=True)
class ControlledBaselineInputBatch:
    """Selected, geometry-aligned inputs for a controlled baseline.

    Branches always use the fixed ``[L_E,L_R,C_E,C_R]`` order. Unsupported
    branches and their ages are exact zero; ``support`` is the sole mask a
    baseline fusion module should use.
    """

    branches: Tensor
    support: Tensor
    ages: Tensor
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.branches, Tensor)
            or self.branches.ndim != 5
            or self.branches.shape[1] != 4
            or self.branches.shape[2] <= 0
            or self.branches.shape[3] <= 0
            or self.branches.shape[4] <= 0
            or not self.branches.is_floating_point()
        ):
            raise ProtocolInvariantError("branches must be floating [B,4,C,H,W]")
        batch = self.branches.shape[0]
        if batch <= 0:
            raise ProtocolInvariantError("branches batch must be positive")
        if (
            not isinstance(self.support, Tensor)
            or self.support.shape != (batch, 4)
            or self.support.dtype is not torch.bool
        ):
            raise ProtocolInvariantError("support must be boolean [B,4]")
        if (
            not isinstance(self.ages, Tensor)
            or self.ages.shape != (batch, 4)
            or not self.ages.is_floating_point()
        ):
            raise ProtocolInvariantError("ages must be floating [B,4]")
        if (
            self.support.device != self.branches.device
            or self.ages.device != self.branches.device
            or self.ages.dtype != self.branches.dtype
        ):
            raise ProtocolInvariantError(
                "branches, support, and ages must share placement and feature dtype"
            )
        if (
            type(self.sample_ids) is not tuple
            or len(self.sample_ids) != batch
            or any(type(value) is not str or not value for value in self.sample_ids)
        ):
            raise ProtocolInvariantError(
                "sample_ids must be a non-empty string tuple matching the batch"
            )
        if not torch.isfinite(self.branches.detach()).all().item():
            raise ProtocolInvariantError("selected branches must be finite")
        if not torch.isfinite(self.ages.detach()).all().item():
            raise ProtocolInvariantError("branch ages must be finite")
        unsupported = ~self.support
        if (
            self.branches.detach()
            .masked_select(unsupported[:, :, None, None, None].expand_as(self.branches))
            .ne(0)
            .any()
            .item()
        ):
            raise ProtocolInvariantError(
                "unsupported branches must remain exactly zero"
            )
        if self.ages.detach().masked_select(unsupported).ne(0).any().item():
            raise ProtocolInvariantError(
                "unsupported branch ages must remain exactly zero"
            )


class ControlledBaselineInputSelector(nn.Module):
    """Select causal horizons and perform geometry-only BEV alignment."""

    history_positions = 4

    def __init__(
        self,
        grid_spec: BEVGridSpec,
        channels: int = 256,
        enabled_branches: tuple[str, ...] = CONTROLLED_BRANCH_KEYS,
    ) -> None:
        super().__init__()
        if not isinstance(grid_spec, BEVGridSpec):
            raise ValueError("grid_spec must be a BEVGridSpec")
        if type(channels) is not int or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if (
            type(enabled_branches) is not tuple
            or not enabled_branches
            or any(key not in CONTROLLED_BRANCH_KEYS for key in enabled_branches)
            or len(set(enabled_branches)) != len(enabled_branches)
        ):
            raise ValueError(
                "enabled_branches must be a non-empty unique tuple of "
                "controlled branch keys"
            )
        self.grid_spec = grid_spec
        self.channels = channels
        self.enabled_branches = frozenset(enabled_branches)

    def _validate_histories(
        self,
        lidar_history: object,
        camera_history: object,
        selections: ResilientBatchSelections,
    ) -> tuple[Tensor, Tensor]:
        expected = (
            selections.batch_size,
            2,
            self.history_positions,
            self.channels,
            self.grid_spec.height,
            self.grid_spec.width,
        )
        for name, value in (
            ("lidar_history", lidar_history),
            ("camera_history", camera_history),
        ):
            if (
                not isinstance(value, Tensor)
                or tuple(value.shape) != expected
                or not value.is_floating_point()
            ):
                raise ValueError(f"{name} must be floating [B,2,4,C,H,W]")
        assert isinstance(lidar_history, Tensor)
        assert isinstance(camera_history, Tensor)
        if (
            camera_history.device != lidar_history.device
            or camera_history.dtype != lidar_history.dtype
        ):
            raise ValueError("LiDAR and camera histories must share dtype and device")
        return lidar_history, camera_history

    def forward(
        self,
        lidar_history: Tensor,
        camera_history: Tensor,
        source_to_target: Tensor,
        availability: Tensor,
        selections: ResilientBatchSelections,
    ) -> ControlledBaselineInputBatch:
        if not isinstance(selections, ResilientBatchSelections):
            raise ValueError("selections must be ResilientBatchSelections")
        lidar_history, camera_history = self._validate_histories(
            lidar_history,
            camera_history,
            selections,
        )
        batch = selections.batch_size
        if (
            not isinstance(source_to_target, Tensor)
            or source_to_target.shape != (batch, 2, 2, self.history_positions, 4, 4)
            or not source_to_target.is_floating_point()
        ):
            raise ValueError("source_to_target must be floating [B,2,2,4,4,4]")
        if (
            source_to_target.device != lidar_history.device
            or source_to_target.dtype != _geometry_dtype(lidar_history.dtype)
        ):
            raise ValueError(
                "source_to_target must share device and use the required geometry dtype"
            )
        if (
            not isinstance(availability, Tensor)
            or availability.shape != (batch, 2, 2, self.history_positions)
            or availability.dtype is not torch.bool
            or availability.device != lidar_history.device
        ):
            raise ValueError("availability must be boolean [B,2,2,4]")

        selected_features: list[Tensor] = []
        selected_transforms: list[Tensor] = []
        output_indices: list[int] = []
        support = torch.zeros(
            batch,
            4,
            dtype=torch.bool,
            device=lidar_history.device,
        )
        ages = lidar_history.new_zeros(batch, 4)

        histories = (lidar_history, camera_history)
        for branch_index, (
            key,
            modality_index,
            agent_index,
            expected_agent,
            expected_modality,
        ) in enumerate(_BRANCH_LAYOUT):
            if key not in self.enabled_branches:
                continue
            branch_selections = selections.branch(key)
            for batch_index, selection in enumerate(branch_selections):
                if (
                    selection.agent is not expected_agent
                    or selection.modality is not expected_modality
                ):
                    raise ProtocolInvariantError(
                        "selection branch does not match [L_E,L_R,C_E,C_R]"
                    )
                if not selection.supported:
                    continue
                horizon = selection.horizon
                if (
                    not isinstance(horizon, Integral)
                    or isinstance(horizon, bool)
                    or not 0 <= horizon < self.history_positions
                ):
                    raise ProtocolInvariantError(
                        "supported selection horizon must be in [0,3]"
                    )
                horizon = int(horizon)
                if not bool(
                    availability[
                        batch_index,
                        modality_index,
                        agent_index,
                        horizon,
                    ].item()
                ):
                    raise ProtocolInvariantError(
                        "selected source slot must be available"
                    )
                feature = histories[modality_index][
                    batch_index,
                    agent_index,
                    horizon,
                ]
                transform = source_to_target[
                    batch_index,
                    modality_index,
                    agent_index,
                    horizon,
                ]
                if not torch.isfinite(feature.detach()).all().item():
                    raise ValueError("selected history feature must be finite")
                selected_features.append(feature)
                selected_transforms.append(transform)
                output_indices.append(batch_index * 4 + branch_index)
                support[batch_index, branch_index] = True
                ages[batch_index, branch_index] = float(horizon)

        flat_branches = lidar_history.new_zeros(
            batch * 4,
            self.channels,
            self.grid_spec.height,
            self.grid_spec.width,
        )
        if selected_features:
            aligned = align_bev_to_target(
                torch.stack(selected_features, dim=0),
                torch.stack(selected_transforms, dim=0),
                self.grid_spec,
            )
            flat_branches = flat_branches.index_copy(
                0,
                torch.tensor(
                    output_indices,
                    dtype=torch.long,
                    device=lidar_history.device,
                ),
                aligned,
            )
        branches = flat_branches.view(
            batch,
            4,
            self.channels,
            self.grid_spec.height,
            self.grid_spec.width,
        )
        return ControlledBaselineInputBatch(
            branches=branches,
            support=support,
            ages=ages,
            sample_ids=selections.sample_ids,
        )


def select_controlled_baseline_inputs(
    *,
    lidar_history: Tensor,
    camera_history: Tensor,
    source_to_target: Tensor,
    availability: Tensor,
    selections: ResilientBatchSelections,
    grid_spec: BEVGridSpec,
    channels: int = 256,
) -> ControlledBaselineInputBatch:
    """Functional wrapper for callers that do not need to retain a module."""

    return ControlledBaselineInputSelector(
        grid_spec=grid_spec,
        channels=channels,
    )(
        lidar_history=lidar_history,
        camera_history=camera_history,
        source_to_target=source_to_target,
        availability=availability,
        selections=selections,
    )


__all__ = (
    "CONTROLLED_BRANCH_KEYS",
    "ControlledBaselineInputBatch",
    "ControlledBaselineInputSelector",
    "select_controlled_baseline_inputs",
)
