from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PTFOutput:
    displacement: Tensor
    confidence: Tensor


class _FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.normalization1 = nn.GroupNorm(
            32,
            channels,
        )
        self.activation1 = nn.SiLU()
        self.convolution2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.normalization2 = nn.GroupNorm(
            32,
            channels,
        )
        self.activation2 = nn.SiLU()

    def forward(
        self,
        inputs: Tensor,
        scale: Tensor,
        bias: Tensor,
    ) -> Tensor:
        residual = inputs
        transformed = self.convolution1(inputs)
        transformed = self.normalization1(transformed)
        transformed = transformed * (1.0 + scale) + bias
        transformed = self.activation1(transformed)
        transformed = self.convolution2(transformed)
        transformed = self.normalization2(transformed)
        return self.activation2(residual + transformed)


class HorizonConditionedPTF(nn.Module):
    _AGENT_COUNT = 2
    _RESIDUAL_BLOCK_COUNT = 3

    def __init__(
        self,
        in_channels: int,
        history_positions: int,
        history_limit: int,
        projected_channels: int,
        context_channels: int,
        max_low_resolution_cells: float,
        mode: Literal["nonlinear", "linear"],
    ) -> None:
        super().__init__()
        self._require_approved_integer(in_channels, "in_channels", 256)
        self._require_approved_integer(
            history_positions,
            "history_positions",
            4,
        )
        self._require_approved_integer(history_limit, "history_limit", 3)
        self._require_approved_integer(
            projected_channels,
            "projected_channels",
            64,
        )
        self._require_approved_integer(
            context_channels,
            "context_channels",
            128,
        )
        if (
            not isinstance(max_low_resolution_cells, Real)
            or isinstance(max_low_resolution_cells, bool)
            or not math.isfinite(max_low_resolution_cells)
            or max_low_resolution_cells <= 0.0
            or float(max_low_resolution_cells) != 8.0
        ):
            raise ValueError(
                "max_low_resolution_cells must be the finite approved value 8.0"
            )
        if mode not in ("nonlinear", "linear"):
            raise ValueError("mode must be 'nonlinear' or 'linear'")

        self.in_channels = int(in_channels)
        self.history_positions = int(history_positions)
        self.history_limit = int(history_limit)
        self.projected_channels = int(projected_channels)
        self.context_channels = int(context_channels)
        self.max_low_resolution_cells = float(max_low_resolution_cells)
        self.mode = mode

        self.history_projection = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(
                128,
                self.projected_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(16, self.projected_channels),
            nn.SiLU(),
        )
        self.agent_embedding = nn.Embedding(
            self._AGENT_COUNT,
            self.projected_channels,
        )
        self.relative_time_embedding = nn.Embedding(
            self.history_positions,
            self.projected_channels,
        )

        slot_count = self._AGENT_COUNT * self.history_positions
        context_input_channels = slot_count * (self.projected_channels + 1)
        self.context_stem = nn.Sequential(
            nn.Conv2d(
                context_input_channels,
                self.context_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(32, self.context_channels),
            nn.SiLU(),
        )
        self.residual_blocks = nn.ModuleList(
            [
                _FiLMResidualBlock(self.context_channels)
                for _ in range(self._RESIDUAL_BLOCK_COUNT)
            ]
        )
        self.film_mlp = nn.Sequential(
            nn.Linear(self.projected_channels + 1, 128, bias=True),
            nn.SiLU(),
            nn.Linear(
                128,
                self._RESIDUAL_BLOCK_COUNT * 2 * self.context_channels,
                bias=True,
            ),
        )
        self.displacement_head = nn.Sequential(
            nn.Conv2d(
                self.context_channels,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, 2, kernel_size=1, bias=True),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(
                self.context_channels,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
        )

        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
        nn.init.zeros_(self.displacement_head[-1].weight)
        nn.init.zeros_(self.displacement_head[-1].bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

    @staticmethod
    def _require_approved_integer(
        value: object,
        name: str,
        expected: int,
    ) -> None:
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or int(value) != expected
        ):
            raise ValueError(f"{name} must be the approved integer {expected}")

    def projected_spatial_shape(
        self,
        spatial_shape: object,
    ) -> tuple[int, int]:
        if not isinstance(spatial_shape, (tuple, list)) or len(spatial_shape) != 2:
            raise ValueError("spatial shape must contain height and width")
        normalized: list[int] = []
        for dimension in spatial_shape:
            if (
                not isinstance(dimension, Integral)
                or isinstance(dimension, bool)
                or int(dimension) <= 0
            ):
                raise ValueError("spatial dimensions must be positive integers")
            integer_dimension = int(dimension)
            if integer_dimension % 4:
                raise ValueError("spatial dimensions must be divisible by four")
            normalized.append(integer_dimension // 4)
        return normalized[0], normalized[1]

    def build_context(
        self,
        aligned_history: Tensor,
        availability: Tensor,
    ) -> Tensor:
        if not isinstance(aligned_history, Tensor):
            raise ValueError("aligned_history must be a tensor")
        if aligned_history.ndim != 6:
            raise ValueError("aligned_history must have shape [B,2,4,256,H,W]")
        batch, agents, positions, channels, height, width = aligned_history.shape
        if batch <= 0:
            raise ValueError("aligned_history batch must be positive")
        if (
            agents != self._AGENT_COUNT
            or positions != self.history_positions
            or channels != self.in_channels
        ):
            raise ValueError("aligned_history must have shape [B,2,4,256,H,W]")
        if not aligned_history.is_floating_point():
            raise ValueError("aligned_history must be floating")
        projected_height, projected_width = self.projected_spatial_shape(
            (height, width)
        )

        if not isinstance(availability, Tensor):
            raise ValueError("availability must be a tensor")
        if availability.shape != (
            batch,
            self._AGENT_COUNT,
            self.history_positions,
        ):
            raise ValueError("availability must have shape [B,2,4]")
        if availability.dtype is not torch.bool:
            raise ValueError("availability must be boolean")
        if availability.device != aligned_history.device:
            raise ValueError("aligned_history and availability must share a device")
        if aligned_history.dtype != self.history_projection[0].weight.dtype:
            raise ValueError("aligned_history dtype must match the PTF parameter dtype")

        slot_count = self._AGENT_COUNT * self.history_positions
        flat = aligned_history.reshape(
            batch * slot_count,
            channels,
            height,
            width,
        )
        flat_mask = availability.reshape(batch * slot_count)
        valid_index = flat_mask.nonzero(as_tuple=False).flatten()
        valid_history = flat.index_select(0, valid_index)
        if (
            valid_index.numel() > 0
            and not torch.isfinite(valid_history.detach()).all().item()
        ):
            raise ValueError("available aligned_history must be finite")

        projected = flat.new_zeros(
            (
                batch * slot_count,
                self.projected_channels,
                projected_height,
                projected_width,
            )
        )
        if valid_index.numel() > 0:
            projected_valid = self.history_projection(valid_history)
            if projected_valid.shape[-2:] != (
                projected_height,
                projected_width,
            ):
                raise RuntimeError("PTF projection spatial shape mismatch")
            projected.index_copy_(0, valid_index, projected_valid)
        projected = projected.view(
            batch,
            self._AGENT_COUNT,
            self.history_positions,
            self.projected_channels,
            projected_height,
            projected_width,
        )

        agent_index = torch.arange(
            self._AGENT_COUNT,
            device=aligned_history.device,
            dtype=torch.long,
        ).view(1, self._AGENT_COUNT, 1)
        agent_index = agent_index.expand(
            batch,
            self._AGENT_COUNT,
            self.history_positions,
        )
        relative_time_index = torch.arange(
            self.history_positions,
            device=aligned_history.device,
            dtype=torch.long,
        ).view(1, 1, self.history_positions)
        relative_time_index = relative_time_index.expand(
            batch,
            self._AGENT_COUNT,
            self.history_positions,
        )
        agent_embedding = self.agent_embedding(agent_index)
        relative_time_embedding = self.relative_time_embedding(relative_time_index)
        slot_mask = availability.view(
            batch,
            self._AGENT_COUNT,
            self.history_positions,
            1,
            1,
            1,
        ).to(dtype=aligned_history.dtype)
        projected = (
            projected
            + agent_embedding.unsqueeze(-1).unsqueeze(-1)
            + relative_time_embedding.unsqueeze(-1).unsqueeze(-1)
        ) * slot_mask

        flattened_features = projected.reshape(
            batch,
            slot_count * self.projected_channels,
            projected_height,
            projected_width,
        )
        mask_planes = availability.reshape(
            batch,
            slot_count,
            1,
            1,
        ).to(dtype=aligned_history.dtype)
        mask_planes = mask_planes.expand(
            batch,
            slot_count,
            projected_height,
            projected_width,
        )
        return self.context_stem(torch.cat((flattened_features, mask_planes), dim=1))

    def query(
        self,
        context: Tensor,
        query_agent_index: Tensor,
        horizon: Tensor,
    ) -> PTFOutput:
        if not isinstance(context, Tensor):
            raise ValueError("context must be a tensor")
        if context.ndim != 4:
            raise ValueError("context must have shape [B,128,H,W]")
        batch, channels, height, width = context.shape
        if batch <= 0 or height <= 0 or width <= 0:
            raise ValueError("context must have positive batch and spatial size")
        if channels != self.context_channels:
            raise ValueError("context must have shape [B,128,H,W]")
        if not context.is_floating_point():
            raise ValueError("context must be floating")
        if not torch.isfinite(context.detach()).all().item():
            raise ValueError("context must be finite")
        if context.dtype != self.context_stem[0].weight.dtype:
            raise ValueError("context dtype must match the PTF parameter dtype")

        self._validate_index_vector(
            query_agent_index,
            "query_agent_index",
            batch,
        )
        self._validate_index_vector(horizon, "horizon", batch)
        if (
            query_agent_index.device != context.device
            or horizon.device != context.device
        ):
            raise ValueError(
                "context, query_agent_index, and horizon must share a device"
            )
        if (
            query_agent_index.lt(0).any().item()
            or query_agent_index.gt(self._AGENT_COUNT - 1).any().item()
        ):
            raise ValueError("query_agent_index must contain only 0 or 1")
        if horizon.lt(0).any().item() or horizon.gt(self.history_limit).any().item():
            raise ValueError("horizon must be in [0, 3]")

        query_embedding = self.agent_embedding(query_agent_index.to(dtype=torch.long))
        if self.mode == "linear":
            film_horizon = context.new_full(
                (batch, 1),
                1.0 / self.history_limit,
            )
        else:
            film_horizon = (
                horizon.to(dtype=context.dtype).unsqueeze(1) / self.history_limit
            )
        film_parameters = self.film_mlp(
            torch.cat((query_embedding, film_horizon), dim=1)
        )
        film_parameters = film_parameters.view(
            batch,
            self._RESIDUAL_BLOCK_COUNT,
            2,
            self.context_channels,
        )

        transformed = context
        for block_index, block in enumerate(self.residual_blocks):
            scale = film_parameters[:, block_index, 0].view(
                batch,
                self.context_channels,
                1,
                1,
            )
            bias = film_parameters[:, block_index, 1].view(
                batch,
                self.context_channels,
                1,
                1,
            )
            transformed = block(transformed, scale, bias)

        raw_displacement = self.displacement_head(transformed)
        raw_confidence = self.confidence_head(transformed)
        if self.mode == "linear":
            low_resolution_displacement = (
                self.max_low_resolution_cells
                / self.history_limit
                * torch.tanh(raw_displacement)
            )
            low_resolution_displacement = low_resolution_displacement * horizon.to(
                dtype=context.dtype
            ).view(batch, 1, 1, 1)
        else:
            low_resolution_displacement = self.max_low_resolution_cells * torch.tanh(
                raw_displacement
            )
        low_resolution_confidence = torch.sigmoid(raw_confidence)

        output_shape = (height * 4, width * 4)
        displacement = F.interpolate(
            low_resolution_displacement,
            size=output_shape,
            mode="bilinear",
            align_corners=False,
        )
        displacement = displacement * 4.0
        confidence = F.interpolate(
            low_resolution_confidence,
            size=output_shape,
            mode="bilinear",
            align_corners=False,
        )
        zero_horizon = horizon.eq(0).view(batch, 1, 1, 1)
        displacement = torch.where(
            zero_horizon,
            torch.zeros_like(displacement),
            displacement,
        )
        return PTFOutput(
            displacement=displacement,
            confidence=confidence,
        )

    @staticmethod
    def _validate_index_vector(
        values: object,
        name: str,
        batch: int,
    ) -> None:
        if not isinstance(values, Tensor):
            raise ValueError(f"{name} must be a tensor")
        if values.ndim != 1 or values.shape[0] != batch:
            raise ValueError(f"{name} must have shape [B]")
        if (
            values.dtype is torch.bool
            or values.is_floating_point()
            or values.is_complex()
        ):
            raise ValueError(f"{name} must contain non-boolean integers")
