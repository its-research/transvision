from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor, nn


def _require_positive_channels(
    value: object,
    name: str,
    *,
    approved: int | None = None,
) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive non-boolean integer")
    normalized = int(value)
    if approved is not None and normalized != approved:
        raise ValueError(f"{name} must be the approved integer {approved}")
    if approved is None and normalized % 32:
        raise ValueError(f"{name} must be divisible by 32")
    return normalized


def _is_finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _require_input_finite(value: Tensor, name: str) -> None:
    if not _is_finite(value):
        raise ValueError(f"{name} must contain only finite values")


def _require_runtime_finite(value: Tensor, name: str) -> None:
    if not _is_finite(value):
        raise RuntimeError(f"{name} must contain only finite values")


def _require_runtime_tensor(
    value: object,
    name: str,
    shape: tuple[int, ...],
    reference: Tensor,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise RuntimeError(f"{name} must be a tensor")
    if value.shape != shape:
        raise RuntimeError(f"{name} must have shape {list(shape)}")
    if not value.is_floating_point():
        raise RuntimeError(f"{name} must be floating")
    if value.dtype != reference.dtype:
        raise RuntimeError(f"{name} dtype must match feature inputs")
    if value.device != reference.device:
        raise RuntimeError(f"{name} device must match feature inputs")
    _require_runtime_finite(value, name)
    return value


def _require_feature(
    value: object,
    name: str,
    channels: int,
    parameter: Tensor,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [B,{channels},Y,X]")
    if (
        value.shape[0] <= 0
        or value.shape[1] != channels
        or value.shape[2] <= 0
        or value.shape[3] <= 0
    ):
        raise ValueError(
            f"{name} must have positive shape [B,{channels},Y,X]"
        )
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating")
    if value.dtype != parameter.dtype:
        raise ValueError(f"{name} dtype must match module parameters")
    if value.device != parameter.device:
        raise ValueError(f"{name} device must match module parameters")
    _require_input_finite(value, name)
    return value


def _require_bool_tensor(
    value: object,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean")
    if value.device != device:
        raise ValueError(f"{name} device must match feature inputs")
    return value


def _require_unit_interval_tensor(
    value: object,
    name: str,
    shape: tuple[int, ...],
    reference: Tensor,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating")
    if value.dtype != reference.dtype:
        raise ValueError(f"{name} dtype must match feature inputs")
    if value.device != reference.device:
        raise ValueError(f"{name} device must match feature inputs")
    _require_input_finite(value, name)
    detached = value.detach()
    if detached.lt(0).any().item() or detached.gt(1).any().item():
        raise ValueError(f"{name} must contain values in [0,1]")
    return value


def _mask_feature(value: Tensor, support: Tensor) -> Tensor:
    return value * support[:, None, None, None].to(dtype=value.dtype)


def _masked_softmax(logits: Tensor, support: Tensor) -> Tensor:
    masked = logits.masked_fill(
        ~support,
        torch.finfo(logits.dtype).min,
    )
    any_valid = support.any(dim=1, keepdim=True)
    safe_logits = torch.where(
        any_valid,
        masked,
        torch.zeros_like(masked),
    )
    weights = torch.softmax(safe_logits, dim=1)
    weights = weights * support.to(dtype=weights.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.where(
        any_valid,
        weights / denominator,
        torch.zeros_like(weights),
    )


class DepthwiseSeparableResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = _require_positive_channels(channels, "channels")
        self.depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=False,
        )
        self.normalization1 = nn.GroupNorm(32, self.channels)
        self.activation1 = nn.SiLU()
        self.pointwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False,
        )
        self.normalization2 = nn.GroupNorm(32, self.channels)
        self.activation2 = nn.SiLU()

    def forward(self, value: Tensor) -> Tensor:
        value = _require_feature(
            value,
            "value",
            self.channels,
            self.depthwise.weight,
        )
        transformed = self.depthwise(value)
        transformed = self.normalization1(transformed)
        transformed = self.activation1(transformed)
        transformed = self.pointwise(transformed)
        transformed = self.normalization2(transformed)
        output = self.activation2(value + transformed)
        return _require_runtime_tensor(
            output,
            "residual block output",
            tuple(value.shape),
            value,
        )


@dataclass
class ModalityAggregate:
    feature: Tensor
    support: Tensor
    reliability: Tensor


class ModalityAggregator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = _require_positive_channels(
            channels,
            "channels",
            approved=256,
        )
        self.projection = nn.Sequential(
            nn.Conv2d(
                2 * self.channels,
                self.channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(32, self.channels),
            nn.SiLU(),
        )
        self.residual_block = DepthwiseSeparableResidualBlock(self.channels)

    def forward(
        self,
        ego_feature: Tensor,
        rsu_feature: Tensor,
        ego_support: Tensor,
        rsu_support: Tensor,
        ego_reliability: Tensor,
        rsu_reliability: Tensor,
    ) -> ModalityAggregate:
        parameter = self.projection[0].weight
        ego_feature = _require_feature(
            ego_feature,
            "ego_feature",
            self.channels,
            parameter,
        )
        rsu_feature = _require_feature(
            rsu_feature,
            "rsu_feature",
            self.channels,
            parameter,
        )
        if rsu_feature.shape != ego_feature.shape:
            raise ValueError(
                "ego_feature and rsu_feature must have identical shape"
            )
        if rsu_feature.dtype != ego_feature.dtype:
            raise ValueError(
                "ego_feature and rsu_feature must have identical dtype"
            )
        if rsu_feature.device != ego_feature.device:
            raise ValueError(
                "ego_feature and rsu_feature must have identical device"
            )

        batch = ego_feature.shape[0]
        shape = (batch,)
        ego_support = _require_bool_tensor(
            ego_support,
            "ego_support",
            shape,
            ego_feature.device,
        )
        rsu_support = _require_bool_tensor(
            rsu_support,
            "rsu_support",
            shape,
            ego_feature.device,
        )
        ego_reliability = _require_unit_interval_tensor(
            ego_reliability,
            "ego_reliability",
            shape,
            ego_feature,
        )
        rsu_reliability = _require_unit_interval_tensor(
            rsu_reliability,
            "rsu_reliability",
            shape,
            ego_feature,
        )
        if ego_reliability.detach().masked_select(~ego_support).ne(0).any().item():
            raise ValueError(
                "unsupported Ego reliability must be neutral zero"
            )
        if rsu_reliability.detach().masked_select(~rsu_support).ne(0).any().item():
            raise ValueError(
                "unsupported RSU reliability must be neutral zero"
            )

        masked_ego = _mask_feature(ego_feature, ego_support)
        masked_rsu = _mask_feature(rsu_feature, rsu_support)
        projected = self.projection(
            torch.cat((masked_ego, masked_rsu), dim=1)
        )
        projected = _require_runtime_tensor(
            projected,
            "modality projection",
            tuple(ego_feature.shape),
            ego_feature,
        )
        transformed = self.residual_block(projected)
        transformed = _require_runtime_tensor(
            transformed,
            "modality aggregate",
            tuple(ego_feature.shape),
            ego_feature,
        )
        support = ego_support | rsu_support
        feature = _mask_feature(transformed, support)
        reliability = (ego_reliability + rsu_reliability) / 2.0
        feature = _require_runtime_tensor(
            feature,
            "masked modality aggregate",
            tuple(ego_feature.shape),
            ego_feature,
        )
        reliability = _require_runtime_tensor(
            reliability,
            "modality reliability",
            (batch,),
            ego_feature,
        )
        return ModalityAggregate(
            feature=feature,
            support=support,
            reliability=reliability,
        )


@dataclass
class RoutingOutput:
    fused: Tensor
    weights: Tensor
    descriptor: Tensor
    expert_support: Tensor
    lidar_expert: Tensor
    camera_expert: Tensor
    synergy_expert: Tensor
    overall_support: Tensor


class DynamicExpertRouter(nn.Module):
    descriptor_dim: int = 783

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.channels = _require_positive_channels(
            channels,
            "channels",
            approved=256,
        )
        self.hidden_channels = _require_positive_channels(
            hidden_channels,
            "hidden_channels",
            approved=256,
        )
        self.lidar_expert = nn.Sequential(
            DepthwiseSeparableResidualBlock(self.channels),
            DepthwiseSeparableResidualBlock(self.channels),
        )
        self.camera_expert = nn.Sequential(
            DepthwiseSeparableResidualBlock(self.channels),
            DepthwiseSeparableResidualBlock(self.channels),
        )
        self.synergy_stem = nn.Sequential(
            nn.Conv2d(
                2 * self.channels,
                self.channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(32, self.channels),
            nn.SiLU(),
        )
        self.synergy_expert = nn.Sequential(
            DepthwiseSeparableResidualBlock(self.channels),
            DepthwiseSeparableResidualBlock(self.channels),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(
                self.descriptor_dim,
                self.hidden_channels,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 3, bias=True),
        )

    def _validate_inputs(
        self,
        lidar_feature: object,
        camera_feature: object,
        lidar_branch_support: object,
        camera_branch_support: object,
        branch_reliability: object,
        branch_observed: object,
        branch_propagated: object,
        branch_normalized_age: object,
        normalized_rsu_delay: object,
        routing_mode: object,
        use_reliability: object,
        use_delay_metadata: object,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Literal["dynamic", "static", "uniform"],
        bool,
        bool,
    ]:
        parameter = self.synergy_stem[0].weight
        lidar_feature = _require_feature(
            lidar_feature,
            "lidar_feature",
            self.channels,
            parameter,
        )
        camera_feature = _require_feature(
            camera_feature,
            "camera_feature",
            self.channels,
            parameter,
        )
        if camera_feature.shape != lidar_feature.shape:
            raise ValueError(
                "lidar_feature and camera_feature must have identical shape"
            )
        if camera_feature.dtype != lidar_feature.dtype:
            raise ValueError(
                "lidar_feature and camera_feature must have identical dtype"
            )
        if camera_feature.device != lidar_feature.device:
            raise ValueError(
                "lidar_feature and camera_feature must have identical device"
            )
        batch = lidar_feature.shape[0]
        branch_shape = (batch, 4)
        modality_support_shape = (batch, 2)
        lidar_branch_support = _require_bool_tensor(
            lidar_branch_support,
            "lidar_branch_support",
            modality_support_shape,
            lidar_feature.device,
        )
        camera_branch_support = _require_bool_tensor(
            camera_branch_support,
            "camera_branch_support",
            modality_support_shape,
            lidar_feature.device,
        )
        branch_reliability = _require_unit_interval_tensor(
            branch_reliability,
            "branch_reliability",
            branch_shape,
            lidar_feature,
        )
        branch_observed = _require_bool_tensor(
            branch_observed,
            "branch_observed",
            branch_shape,
            lidar_feature.device,
        )
        branch_propagated = _require_bool_tensor(
            branch_propagated,
            "branch_propagated",
            branch_shape,
            lidar_feature.device,
        )
        branch_normalized_age = _require_unit_interval_tensor(
            branch_normalized_age,
            "branch_normalized_age",
            branch_shape,
            lidar_feature,
        )
        normalized_rsu_delay = _require_unit_interval_tensor(
            normalized_rsu_delay,
            "normalized_rsu_delay",
            (batch, 1),
            lidar_feature,
        )
        if not isinstance(routing_mode, str) or routing_mode not in (
            "dynamic",
            "static",
            "uniform",
        ):
            raise ValueError(
                "routing_mode must be 'dynamic', 'static', or 'uniform'"
            )
        if type(use_reliability) is not bool:
            raise ValueError("use_reliability must be a boolean")
        if type(use_delay_metadata) is not bool:
            raise ValueError("use_delay_metadata must be a boolean")

        branch_support = torch.cat(
            (lidar_branch_support, camera_branch_support),
            dim=1,
        )
        unsupported = ~branch_support
        if (
            branch_reliability.detach()
            .masked_select(unsupported)
            .ne(0)
            .any()
            .item()
        ):
            raise ValueError(
                "unsupported branch reliability must be neutral zero"
            )
        if branch_observed.masked_select(unsupported).any().item():
            raise ValueError(
                "unsupported branch observed flags must be neutral false"
            )
        if branch_propagated.masked_select(unsupported).any().item():
            raise ValueError(
                "unsupported branch propagated flags must be neutral false"
            )
        if (
            branch_normalized_age.detach()
            .masked_select(unsupported)
            .ne(0)
            .any()
            .item()
        ):
            raise ValueError(
                "unsupported branch normalized age must be neutral zero"
            )
        exactly_one_flag = branch_observed ^ branch_propagated
        if exactly_one_flag.ne(branch_support).any().item():
            raise ValueError(
                "supported branch requires exactly one observed/propagated flag"
            )
        rsu_supported = branch_support[:, (1, 3)].any(dim=1)
        if (
            normalized_rsu_delay.detach()[~rsu_supported]
            .ne(0)
            .any()
            .item()
        ):
            raise ValueError(
                "normalized RSU delay must be zero without RSU support"
            )
        return (
            lidar_feature,
            camera_feature,
            lidar_branch_support,
            camera_branch_support,
            branch_reliability,
            branch_observed,
            branch_propagated,
            branch_normalized_age,
            normalized_rsu_delay,
            routing_mode,
            use_reliability,
            use_delay_metadata,
        )

    def forward(
        self,
        lidar_feature: Tensor,
        camera_feature: Tensor,
        lidar_branch_support: Tensor,
        camera_branch_support: Tensor,
        branch_reliability: Tensor,
        branch_observed: Tensor,
        branch_propagated: Tensor,
        branch_normalized_age: Tensor,
        normalized_rsu_delay: Tensor,
        routing_mode: Literal["dynamic", "static", "uniform"],
        use_reliability: bool,
        use_delay_metadata: bool,
    ) -> RoutingOutput:
        (
            lidar_feature,
            camera_feature,
            lidar_branch_support,
            camera_branch_support,
            branch_reliability,
            branch_observed,
            branch_propagated,
            branch_normalized_age,
            normalized_rsu_delay,
            routing_mode,
            use_reliability,
            use_delay_metadata,
        ) = self._validate_inputs(
            lidar_feature,
            camera_feature,
            lidar_branch_support,
            camera_branch_support,
            branch_reliability,
            branch_observed,
            branch_propagated,
            branch_normalized_age,
            normalized_rsu_delay,
            routing_mode,
            use_reliability,
            use_delay_metadata,
        )

        lidar_support = lidar_branch_support.any(dim=1)
        camera_support = camera_branch_support.any(dim=1)
        synergy_support = lidar_support & camera_support
        expert_support = torch.stack(
            (lidar_support, camera_support, synergy_support),
            dim=1,
        )

        raw_lidar_expert = self.lidar_expert(lidar_feature)
        raw_camera_expert = self.camera_expert(camera_feature)
        raw_synergy_stem = self.synergy_stem(
            torch.cat((lidar_feature, camera_feature), dim=1)
        )
        expected_feature_shape = tuple(lidar_feature.shape)
        raw_synergy_stem = _require_runtime_tensor(
            raw_synergy_stem,
            "synergy stem output",
            expected_feature_shape,
            lidar_feature,
        )
        raw_synergy_expert = self.synergy_expert(raw_synergy_stem)
        for name, value in (
            ("LiDAR expert", raw_lidar_expert),
            ("Camera expert", raw_camera_expert),
            ("synergy expert", raw_synergy_expert),
        ):
            _require_runtime_tensor(
                value,
                name,
                expected_feature_shape,
                lidar_feature,
            )

        lidar_expert = _mask_feature(raw_lidar_expert, lidar_support)
        camera_expert = _mask_feature(raw_camera_expert, camera_support)
        synergy_expert = _mask_feature(raw_synergy_expert, synergy_support)
        for name, value in (
            ("masked LiDAR expert", lidar_expert),
            ("masked Camera expert", camera_expert),
            ("masked synergy expert", synergy_expert),
        ):
            _require_runtime_tensor(
                value,
                name,
                expected_feature_shape,
                lidar_feature,
            )

        expert_gap = torch.cat(
            (
                lidar_expert.mean(dim=(-2, -1)),
                camera_expert.mean(dim=(-2, -1)),
                synergy_expert.mean(dim=(-2, -1)),
            ),
            dim=1,
        )
        modality_reliability = torch.stack(
            (
                branch_reliability[:, 0:2].sum(dim=1) / 2.0,
                branch_reliability[:, 2:4].sum(dim=1) / 2.0,
            ),
            dim=1,
        )
        if not use_reliability:
            modality_reliability = torch.stack(
                (lidar_support, camera_support),
                dim=1,
            ).to(dtype=lidar_feature.dtype)
        flags = torch.stack(
            (branch_observed, branch_propagated),
            dim=2,
        ).reshape(lidar_feature.shape[0], 8)
        flags = flags.to(dtype=lidar_feature.dtype)
        if use_delay_metadata:
            ages = branch_normalized_age
            delay = normalized_rsu_delay
        else:
            ages = torch.zeros_like(branch_normalized_age)
            delay = torch.zeros_like(normalized_rsu_delay)
        descriptor = torch.cat(
            (
                expert_gap,
                modality_reliability,
                flags,
                ages,
                delay,
            ),
            dim=1,
        )
        if descriptor.shape != (
            lidar_feature.shape[0],
            self.descriptor_dim,
        ):
            raise RuntimeError("routing descriptor must have shape [B,783]")
        _require_runtime_finite(descriptor, "routing descriptor")
        metadata = descriptor[:, 768:]
        if (
            metadata.detach().lt(0).any().item()
            or metadata.detach().gt(1).any().item()
        ):
            raise RuntimeError(
                "routing descriptor metadata must be in [0,1]"
            )

        if routing_mode == "uniform":
            valid_count = expert_support.sum(
                dim=1,
                keepdim=True,
            ).to(dtype=lidar_feature.dtype)
            weights = (
                expert_support.to(dtype=lidar_feature.dtype)
                / valid_count.clamp_min(1.0)
            )
        else:
            if routing_mode == "dynamic":
                logits = self.gate(descriptor)
            else:
                final_gate = self.gate[-1]
                if not isinstance(final_gate, nn.Linear):
                    raise RuntimeError("final gate must be a Linear layer")
                if final_gate.bias is None:
                    raise RuntimeError("final gate must have a bias")
                logits = final_gate.bias.unsqueeze(0).expand(
                    lidar_feature.shape[0],
                    -1,
                )
            logits = _require_runtime_tensor(
                logits,
                "routing logits",
                (lidar_feature.shape[0], 3),
                lidar_feature,
            )
            weights = _masked_softmax(logits, expert_support)

        weights = _require_runtime_tensor(
            weights,
            "routing weights",
            (lidar_feature.shape[0], 3),
            lidar_feature,
        )
        detached_weights = weights.detach()
        if (
            detached_weights.lt(0).any().item()
            or detached_weights.gt(1).any().item()
        ):
            raise RuntimeError("routing weights must be in [0,1]")
        if detached_weights.masked_select(~expert_support).ne(0).any().item():
            raise RuntimeError("unsupported expert weights must be zero")
        expected_weight_sum = expert_support.any(dim=1).to(
            dtype=weights.dtype
        )
        tolerance = 10.0 * torch.finfo(weights.dtype).eps
        if not torch.allclose(
            detached_weights.sum(dim=1),
            expected_weight_sum,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise RuntimeError(
                "routing weights must sum to one for supported rows"
            )

        experts = torch.stack(
            (lidar_expert, camera_expert, synergy_expert),
            dim=1,
        )
        experts = experts * expert_support[
            :, :, None, None, None
        ].to(dtype=experts.dtype)
        fused = (
            experts
            * weights[:, :, None, None, None]
        ).sum(dim=1)
        fused = _require_runtime_tensor(
            fused,
            "fused expert output",
            expected_feature_shape,
            lidar_feature,
        )
        overall_support = expert_support.any(dim=1)
        return RoutingOutput(
            fused=fused,
            weights=weights,
            descriptor=descriptor,
            expert_support=expert_support,
            lidar_expert=lidar_expert,
            camera_expert=camera_expert,
            synergy_expert=synergy_expert,
            overall_support=overall_support,
        )
