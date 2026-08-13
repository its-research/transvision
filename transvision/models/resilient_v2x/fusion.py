from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .causal_repair import CausalBranchRepair
from .contracts import (
    Agent,
    BranchSelection,
    Modality,
    ProtocolInvariantError,
    ResilientFeatureBatch,
    RoutingDiagnostics,
    SampleInferenceDiagnostics,
)
from .geometry import (
    BEVGridSpec,
    _geometry_transform_dtype,
    align_bev_to_target,
)
from .ptf import HorizonConditionedPTF, PTFOutput
from .routing import DynamicExpertRouter, ModalityAggregator


BRANCH_KEYS = (
    "lidar_ego",
    "lidar_rsu",
    "camera_ego",
    "camera_rsu",
)


@dataclass(frozen=True)
class ResilientBatchSelections:
    """Causal branch selections for one batched decision time."""

    sample_ids: tuple[str, ...]
    lidar_ego: tuple[BranchSelection, ...]
    lidar_rsu: tuple[BranchSelection, ...]
    camera_ego: tuple[BranchSelection, ...]
    camera_rsu: tuple[BranchSelection, ...]
    rsu_delay_intervals: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ProtocolInvariantError("sample_ids must not be empty")
        if any(
            type(sample_id) is not str
            or not sample_id
            or sample_id != sample_id.strip()
            for sample_id in self.sample_ids
        ):
            raise ProtocolInvariantError(
                "sample_ids must contain trimmed non-empty strings"
            )
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ProtocolInvariantError("sample_ids must be unique")

        expected = (
            ("lidar_ego", Agent.EGO, Modality.LIDAR),
            ("lidar_rsu", Agent.RSU, Modality.LIDAR),
            ("camera_ego", Agent.EGO, Modality.CAMERA),
            ("camera_rsu", Agent.RSU, Modality.CAMERA),
        )
        for name, agent, modality in expected:
            branch = getattr(self, name)
            if type(branch) is not tuple or len(branch) != len(self.sample_ids):
                raise ProtocolInvariantError(
                    f"{name} must be a tuple matching sample_ids"
                )
            if any(
                not isinstance(selection, BranchSelection)
                or selection.agent is not agent
                or selection.modality is not modality
                for selection in branch
            ):
                raise ProtocolInvariantError(
                    f"{name} contains a selection for the wrong branch"
                )
        if (
            type(self.rsu_delay_intervals) is not tuple
            or len(self.rsu_delay_intervals) != len(self.sample_ids)
            or any(
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 3.0
                for value in self.rsu_delay_intervals
            )
        ):
            raise ProtocolInvariantError(
                "rsu_delay_intervals must be a finite [0,3] tuple matching sample_ids"
            )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    def branch(self, key: str) -> tuple[BranchSelection, ...]:
        if key not in BRANCH_KEYS:
            raise KeyError(f"unknown branch key: {key}")
        return getattr(self, key)

    def sample(self, index: int) -> tuple[BranchSelection, ...]:
        return tuple(self.branch(key)[index] for key in BRANCH_KEYS)


class IdentityTrajectoryField(nn.Module):
    """No-PTF ablation: geometry only, with neutral confidence."""

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        if type(height) is not int or height <= 0:
            raise ValueError("height must be a positive integer")
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        self.height = height
        self.width = width

    def query(
        self,
        context: Tensor,
        query_agent_index: Tensor,
        horizon: Tensor,
    ) -> PTFOutput:
        if not isinstance(context, Tensor) or context.ndim != 4:
            raise ValueError("context must be a rank-4 tensor")
        batch = context.shape[0]
        if (
            not isinstance(query_agent_index, Tensor)
            or query_agent_index.shape != (batch,)
            or not isinstance(horizon, Tensor)
            or horizon.shape != (batch,)
        ):
            raise ValueError("query vectors must have shape [B]")
        return PTFOutput(
            displacement=context.new_zeros(
                batch,
                2,
                self.height,
                self.width,
            ),
            confidence=context.new_ones(
                batch,
                1,
                self.height,
                self.width,
            ),
        )


class ResilientV2XFeatureFusion(nn.Module):
    """Paper-level causal PTF repair and dynamic expert routing.

    Inputs contain shared-encoder BEV histories in each source-agent frame.
    Slot ``h`` is the source at logical horizon ``h`` relative to the current
    decision time, so the fixed four slots are ``h = 0, 1, 2, 3``.
    """

    channels = 256
    history_positions = 4
    history_limit = 3
    alpha = 0.9

    def __init__(
        self,
        grid_spec: BEVGridSpec,
        ptf_mode: Literal["nonlinear", "linear", "none"] = "nonlinear",
        routing_mode: Literal["dynamic", "static", "uniform", "concat"] = "dynamic",
        use_reliability: bool = True,
        use_delay_metadata: bool = True,
        support_residual_weight: float = 0.0,
        support_residual_reliability_gate: bool = False,
        delta_t_ms: int = 100,
    ) -> None:
        super().__init__()
        if not isinstance(grid_spec, BEVGridSpec):
            raise ValueError("grid_spec must be a BEVGridSpec")
        if grid_spec.height % 4 or grid_spec.width % 4:
            raise ValueError("grid height and width must be divisible by four")
        if ptf_mode not in ("nonlinear", "linear", "none"):
            raise ValueError("ptf_mode must be 'nonlinear', 'linear', or 'none'")
        if routing_mode not in ("dynamic", "static", "uniform", "concat"):
            raise ValueError(
                "routing_mode must be 'dynamic', 'static', 'uniform', or 'concat'"
            )
        if type(use_reliability) is not bool:
            raise ValueError("use_reliability must be a boolean")
        if type(use_delay_metadata) is not bool:
            raise ValueError("use_delay_metadata must be a boolean")
        if type(delta_t_ms) is not int or delta_t_ms <= 0:
            raise ValueError("delta_t_ms must be a positive integer")

        self.grid_spec = grid_spec
        self.ptf_mode = ptf_mode
        self.routing_mode = routing_mode
        self.use_reliability = use_reliability
        self.use_delay_metadata = use_delay_metadata
        self.delta_t_ms = delta_t_ms

        common = dict(
            in_channels=self.channels,
            history_positions=self.history_positions,
            history_limit=self.history_limit,
            projected_channels=64,
            context_channels=128,
            max_low_resolution_cells=8.0,
            mode="nonlinear" if ptf_mode == "none" else ptf_mode,
        )
        if ptf_mode == "none":
            self.lidar_ptf: nn.Module = IdentityTrajectoryField(
                grid_spec.height,
                grid_spec.width,
            )
            self.camera_ptf: nn.Module = IdentityTrajectoryField(
                grid_spec.height,
                grid_spec.width,
            )
            # The controlled ablation is parameter-capacity matched. These
            # dormant modules have exactly the full nonlinear PTF parameter
            # budget but are never called by the geometry-only path.
            self.lidar_ptf_capacity_match = HorizonConditionedPTF(**common)
            self.camera_ptf_capacity_match = HorizonConditionedPTF(**common)
        else:
            self.lidar_ptf = HorizonConditionedPTF(**common)
            self.camera_ptf = HorizonConditionedPTF(**common)

        self.repair = CausalBranchRepair(
            grid_spec=grid_spec,
            channels=self.channels,
            alpha=self.alpha,
            history_limit=self.history_limit,
        )
        self.lidar_aggregator = ModalityAggregator(self.channels)
        self.camera_aggregator = ModalityAggregator(self.channels)
        self.router = DynamicExpertRouter(
            channels=self.channels,
            hidden_channels=256,
            support_residual_weight=support_residual_weight,
            support_residual_reliability_gate=(support_residual_reliability_gate),
        )
        self.support_residual_reliability_gate = (
            self.router.support_residual_reliability_gate
        )

    def _validate_history(
        self,
        name: str,
        history: object,
        source_to_target: object,
        availability: object,
        batch: int,
        reference: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(history, Tensor) or history.ndim != 6:
            raise ValueError(f"{name}_history must have shape [B,2,4,256,H,W]")
        expected = (
            batch,
            2,
            self.history_positions,
            self.channels,
            self.grid_spec.height,
            self.grid_spec.width,
        )
        if tuple(history.shape) != expected:
            raise ValueError(f"{name}_history must have shape [B,2,4,256,H,W]")
        if not history.is_floating_point():
            raise ValueError(f"{name}_history must be floating")
        if reference is not None and (
            history.dtype != reference.dtype or history.device != reference.device
        ):
            raise ValueError("LiDAR and camera histories must share placement")

        if (
            not isinstance(source_to_target, Tensor)
            or source_to_target.shape != (batch, 2, self.history_positions, 4, 4)
            or not source_to_target.is_floating_point()
        ):
            raise ValueError(f"{name}_source_to_target must have shape [B,2,4,4,4]")
        if source_to_target.device != history.device:
            raise ValueError(f"{name} history and transforms must share device")
        if source_to_target.dtype != _geometry_transform_dtype(history.dtype):
            raise ValueError(
                f"{name} transforms must use the geometry dtype required by history"
            )
        if (
            not isinstance(availability, Tensor)
            or availability.shape != (batch, 2, self.history_positions)
            or availability.dtype is not torch.bool
            or availability.device != history.device
        ):
            raise ValueError(f"{name}_availability must be boolean [B,2,4]")

        flat_available = availability.reshape(-1)
        available_index = flat_available.nonzero(as_tuple=False).flatten()
        if available_index.numel():
            flat_history = history.reshape(
                -1,
                self.channels,
                self.grid_spec.height,
                self.grid_spec.width,
            )
            flat_transform = source_to_target.reshape(-1, 4, 4)
            if (
                not torch.isfinite(
                    flat_history.index_select(0, available_index).detach()
                )
                .all()
                .item()
            ):
                raise ValueError(f"available {name} history must contain finite values")
            if (
                not torch.isfinite(
                    flat_transform.index_select(0, available_index).detach()
                )
                .all()
                .item()
            ):
                raise ValueError(
                    f"available {name} transforms must contain finite values"
                )
        return history, source_to_target, availability

    def _validate_selected_slots(
        self,
        selections: ResilientBatchSelections,
        lidar_availability: Tensor,
        camera_availability: Tensor,
    ) -> None:
        for batch_index in range(selections.batch_size):
            for key, modality_index in (
                ("lidar_ego", 0),
                ("lidar_rsu", 0),
                ("camera_ego", 1),
                ("camera_rsu", 1),
            ):
                selection = selections.branch(key)[batch_index]
                if not selection.supported:
                    continue
                if selection.horizon is None:
                    raise ProtocolInvariantError(
                        "supported selection requires a horizon"
                    )
                agent_index = 0 if selection.agent is Agent.EGO else 1
                availability = (
                    lidar_availability if modality_index == 0 else camera_availability
                )
                if not bool(
                    availability[
                        batch_index,
                        agent_index,
                        selection.horizon,
                    ].item()
                ):
                    raise ProtocolInvariantError(
                        "selected source slot must be available"
                    )

    def _aligned_context(
        self,
        history: Tensor,
        source_to_target: Tensor,
        availability: Tensor,
        ptf: nn.Module,
    ) -> Tensor:
        batch = history.shape[0]
        if self.ptf_mode == "none":
            return history.new_zeros(
                batch,
                1,
                self.grid_spec.height // 4,
                self.grid_spec.width // 4,
            )

        flat_history = history.reshape(
            -1,
            self.channels,
            self.grid_spec.height,
            self.grid_spec.width,
        )
        flat_transform = source_to_target.reshape(-1, 4, 4)
        flat_available = availability.reshape(-1)
        available_index = flat_available.nonzero(as_tuple=False).flatten()
        aligned = history.new_zeros(flat_history.shape)
        if available_index.numel():
            aligned_valid = align_bev_to_target(
                flat_history.index_select(0, available_index),
                flat_transform.index_select(0, available_index),
                self.grid_spec,
            )
            aligned.index_copy_(0, available_index, aligned_valid)
        aligned = aligned.reshape_as(history)
        if not isinstance(ptf, HorizonConditionedPTF):
            raise RuntimeError("configured PTF has an unexpected type")
        return ptf.build_context(aligned, availability)

    def _repair_modality(
        self,
        modality: Modality,
        history: Tensor,
        source_to_target: Tensor,
        context: Tensor,
        ptf: nn.Module,
        ego_selections: Sequence[BranchSelection],
        rsu_selections: Sequence[BranchSelection],
    ):
        batch = history.shape[0]
        selected_features: list[Tensor] = []
        selected_transforms: list[Tensor] = []
        flat_selections: list[BranchSelection] = []
        identity = torch.eye(
            4,
            dtype=history.dtype,
            device=history.device,
        )
        for row in range(batch):
            for agent_index, selection in enumerate(
                (ego_selections[row], rsu_selections[row])
            ):
                if selection.modality is not modality:
                    raise ProtocolInvariantError(
                        "selection modality does not match repair call"
                    )
                flat_selections.append(selection)
                if selection.supported:
                    if selection.horizon is None:
                        raise ProtocolInvariantError(
                            "supported selection requires a horizon"
                        )
                    selected_features.append(
                        history[row, agent_index, selection.horizon]
                    )
                    selected_transforms.append(
                        source_to_target[
                            row,
                            agent_index,
                            selection.horizon,
                        ]
                    )
                else:
                    selected_features.append(torch.zeros_like(history[row, 0, 0]))
                    selected_transforms.append(identity)

        selected_feature = torch.stack(selected_features, dim=0)
        selected_transform = torch.stack(selected_transforms, dim=0)
        repeated_context = context.repeat_interleave(2, dim=0)
        query_agent_index = torch.tensor(
            [0, 1] * batch,
            dtype=torch.long,
            device=history.device,
        )
        repaired = self.repair(
            selected_feature=selected_feature,
            source_to_target=selected_transform,
            ptf_context=repeated_context,
            ptf=ptf,  # type: ignore[arg-type]
            query_agent_index=query_agent_index,
            selections=tuple(flat_selections),
        )
        feature = repaired.feature.view(
            batch,
            2,
            self.channels,
            self.grid_spec.height,
            self.grid_spec.width,
        )
        return repaired, feature

    def forward(
        self,
        lidar_history: Tensor,
        camera_history: Tensor,
        lidar_source_to_target: Tensor,
        camera_source_to_target: Tensor,
        lidar_availability: Tensor,
        camera_availability: Tensor,
        selections: ResilientBatchSelections,
    ) -> ResilientFeatureBatch:
        if not isinstance(selections, ResilientBatchSelections):
            raise ValueError("selections must be ResilientBatchSelections")
        batch = selections.batch_size
        (
            lidar_history,
            lidar_source_to_target,
            lidar_availability,
        ) = self._validate_history(
            "lidar",
            lidar_history,
            lidar_source_to_target,
            lidar_availability,
            batch,
            None,
        )
        (
            camera_history,
            camera_source_to_target,
            camera_availability,
        ) = self._validate_history(
            "camera",
            camera_history,
            camera_source_to_target,
            camera_availability,
            batch,
            lidar_history,
        )
        self._validate_selected_slots(
            selections,
            lidar_availability,
            camera_availability,
        )

        lidar_context = self._aligned_context(
            lidar_history,
            lidar_source_to_target,
            lidar_availability,
            self.lidar_ptf,
        )
        camera_context = self._aligned_context(
            camera_history,
            camera_source_to_target,
            camera_availability,
            self.camera_ptf,
        )
        lidar_repaired, lidar_branches = self._repair_modality(
            Modality.LIDAR,
            lidar_history,
            lidar_source_to_target,
            lidar_context,
            self.lidar_ptf,
            selections.lidar_ego,
            selections.lidar_rsu,
        )
        camera_repaired, camera_branches = self._repair_modality(
            Modality.CAMERA,
            camera_history,
            camera_source_to_target,
            camera_context,
            self.camera_ptf,
            selections.camera_ego,
            selections.camera_rsu,
        )

        lidar_support = lidar_repaired.support.view(batch, 2)
        camera_support = camera_repaired.support.view(batch, 2)
        lidar_reliability = lidar_repaired.reliability.view(batch, 2)
        camera_reliability = camera_repaired.reliability.view(batch, 2)
        lidar_aggregate = self.lidar_aggregator(
            lidar_branches[:, 0],
            lidar_branches[:, 1],
            lidar_support[:, 0],
            lidar_support[:, 1],
            lidar_reliability[:, 0],
            lidar_reliability[:, 1],
        )
        camera_aggregate = self.camera_aggregator(
            camera_branches[:, 0],
            camera_branches[:, 1],
            camera_support[:, 0],
            camera_support[:, 1],
            camera_reliability[:, 0],
            camera_reliability[:, 1],
        )

        branch_reliability = torch.cat(
            (lidar_reliability, camera_reliability),
            dim=1,
        )
        branch_observed = torch.cat(
            (
                lidar_repaired.observed.view(batch, 2),
                camera_repaired.observed.view(batch, 2),
            ),
            dim=1,
        )
        branch_propagated = torch.cat(
            (
                lidar_repaired.propagated.view(batch, 2),
                camera_repaired.propagated.view(batch, 2),
            ),
            dim=1,
        )
        branch_age = torch.cat(
            (
                lidar_repaired.age_intervals.view(batch, 2),
                camera_repaired.age_intervals.view(batch, 2),
            ),
            dim=1,
        )
        delay_intervals = lidar_history.new_tensor(selections.rsu_delay_intervals).view(
            batch, 1
        )

        routed = self.router(
            lidar_feature=lidar_aggregate.feature,
            camera_feature=camera_aggregate.feature,
            lidar_branch_support=lidar_support,
            camera_branch_support=camera_support,
            branch_reliability=branch_reliability,
            branch_observed=branch_observed,
            branch_propagated=branch_propagated,
            branch_age_intervals=branch_age,
            rsu_delay_intervals=delay_intervals,
            routing_mode=self.routing_mode,
            use_reliability=self.use_reliability,
            use_delay_metadata=self.use_delay_metadata,
        )

        lidar_diagnostics = lidar_repaired.diagnostics
        camera_diagnostics = camera_repaired.diagnostics
        reliability_gate_diagnostic = (
            ",support_residual_reliability_gate=1"
            if self.support_residual_reliability_gate
            else ""
        )
        method = (
            f"ResilientV2X(ptf={self.ptf_mode},"
            f"routing={self.routing_mode},"
            f"reliability={int(self.use_reliability)},"
            f"delay_metadata={int(self.use_delay_metadata)}"
            f"{reliability_gate_diagnostic})"
        )
        diagnostics = tuple(
            SampleInferenceDiagnostics(
                schema_version=1,
                sample_id=selections.sample_ids[index],
                method=method,
                branches=(
                    lidar_diagnostics[2 * index],
                    lidar_diagnostics[2 * index + 1],
                    camera_diagnostics[2 * index],
                    camera_diagnostics[2 * index + 1],
                ),
                routing=RoutingDiagnostics(
                    descriptor=routed.descriptor[index].detach(),
                    expert_support=routed.expert_support[index].detach(),
                    weights=routed.weights[index].detach(),
                    not_applicable_reason=None,
                ),
            )
            for index in range(batch)
        )
        return ResilientFeatureBatch(
            fused=routed.fused,
            overall_support=routed.overall_support,
            routing_weights=routed.weights,
            routing_descriptor=routed.descriptor,
            branch_features={
                "lidar_ego": lidar_branches[:, 0],
                "lidar_rsu": lidar_branches[:, 1],
                "camera_ego": camera_branches[:, 0],
                "camera_rsu": camera_branches[:, 1],
            },
            diagnostics=diagnostics,
        )
