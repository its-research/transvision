"""Controlled, adapted cooperative BEV fusion baselines.

These modules are independent implementations of paper-level architectural
ideas.  They share one experiment-facing contract and are not drop-in copies
of any upstream implementation:

* V2X-ViT, MIT, official repository commit
  ``f0e6c13f41e916548b2d8aba61e42a18ce980416``:
  https://github.com/DerrickXuNu/v2x-vit
* CoBEVT, Apache-2.0, official repository commit
  ``ed8fd11e663af74ab30f653cda3317b1a130cf53``:
  https://github.com/DerrickXuNu/CoBEVT
* BEVFusion, Apache-2.0, official repository commit
  ``326653dc06e0938edf1aae7d01efcd158ba83de5``:
  https://github.com/mit-han-lab/bevfusion
* CoFormerNet, paper DOI https://doi.org/10.3390/s24134101 and the local
  TransVision integration at commit
  ``8715c1e55d47f24caa081f63a951b5f0954f42d7``.  No CoFormerNet source code
  is reused here.
* FFNet, NeurIPS 2023 paper:
  https://papers.nips.cc/paper_files/paper/2023/file/
  6ca5d2665de83394f437dad0c3746907-Paper-Conference.pdf.  Its official
  repository was audited at commit
  ``52164dfe00764c9a9925539e99689cf25b88eace`` but contains no license file,
  so no repository source code was inspected or reused.  The adaptation below
  is written solely from Equations (2) and (4) and the fusion description in
  the published paper.

In particular, this file does not copy code from OpenCOOD or another
restrictively licensed implementation.  The four inputs are already aligned
BEV branches in this fixed order: ego LiDAR, RSU LiDAR, ego camera, RSU
camera.  Missing branches are masked before any learned operation.
"""

from __future__ import annotations

import math
from importlib import import_module
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


def _dropout_probability(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("dropout must be a finite number in [0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("dropout must be a finite number in [0, 1)")
    return result


def _mlp_hidden_channels(channels: int, ratio: object) -> int:
    if type(ratio) not in (int, float):
        raise ValueError("mlp_ratio must be a finite positive number")
    ratio_float = float(ratio)
    if not math.isfinite(ratio_float) or ratio_float <= 0.0:
        raise ValueError("mlp_ratio must be a finite positive number")
    return max(1, round(channels * ratio_float))


def _resolve_num_heads(channels: int, num_heads: int | None) -> int:
    if num_heads is None:
        for candidate in (8, 4, 2, 1):
            if candidate <= channels and channels % candidate == 0:
                return candidate
        return 1
    heads = _positive_int("num_heads", num_heads)
    if channels % heads:
        raise ValueError("channels must be divisible by num_heads")
    return heads


def _resolve_norm_groups(channels: int, norm_groups: int | None) -> int:
    if norm_groups is not None:
        groups = _positive_int("norm_groups", norm_groups)
        if channels % groups:
            raise ValueError("channels must be divisible by norm_groups")
        return groups
    for candidate in (32, 16, 8, 4, 2, 1):
        if candidate <= channels and channels % candidate == 0:
            return candidate
    return 1


def _window_sizes(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("window_sizes must be a non-empty sequence")
    result = tuple(_positive_int("window size", value) for value in values)
    if not result:
        raise ValueError("window_sizes must be a non-empty sequence")
    return result


class _ControlledFusionBase(nn.Module):
    """Shared validation and missing-branch semantics."""

    branch_count = len(BRANCH_ORDER)

    def __init__(self, channels: int = 256, age_decay: float = 0.25) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.age_decay = _nonnegative_float("age_decay", age_decay)

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

        expected_metadata_shape = (batch, self.branch_count)
        if (
            not isinstance(support, Tensor)
            or support.shape != expected_metadata_shape
            or support.dtype is not torch.bool
        ):
            raise ValueError("support must be a bool tensor with shape [B,4]")
        if support.device != branches.device:
            raise ValueError("branches and support must share a device")
        unsupported_samples = (~support.any(dim=1)).nonzero(as_tuple=False).flatten()
        if unsupported_samples.numel():
            indices = unsupported_samples.detach().cpu().tolist()
            raise ValueError(
                "every sample must have at least one supported branch; "
                f"all branches are missing for sample indices {indices}"
            )

        if (
            not isinstance(ages, Tensor)
            or ages.shape != expected_metadata_shape
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

    def _age_weights(self, support: Tensor, ages: Tensor) -> Tensor:
        logits = -self.age_decay * ages.float()
        logits = logits.masked_fill(~support, -torch.inf)
        return torch.softmax(logits, dim=1).to(dtype=ages.dtype)

    def _weighted_mean(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        weights = self._age_weights(support, ages)
        return (branches * weights[:, :, None, None, None]).sum(dim=1)


class EgoOnlyFusion(_ControlledFusionBase):
    """Non-cooperative ego LiDAR+camera control using the shared encoders."""

    _ego_indices = (0, 2)

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(branches, support, ages)
        ego_support = support[:, self._ego_indices]
        unsupported_samples = (
            (~ego_support.any(dim=1)).nonzero(as_tuple=False).flatten()
        )
        if unsupported_samples.numel():
            indices = unsupported_samples.detach().cpu().tolist()
            raise ValueError(
                f"both ego branches are missing for sample indices {indices}"
            )
        return self._weighted_mean(
            branches[:, self._ego_indices],
            ego_support,
            ages[:, self._ego_indices],
        )


class FCooperAdaptedFusion(_ControlledFusionBase):
    """Support-aware spatial-max L+C adaptation of F-Cooper SFF."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__(channels=channels, age_decay=0.0)

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, _ = self._prepare_inputs(branches, support, ages)
        branch_mask = support[:, :, None, None, None]
        floor = torch.finfo(branches.dtype).min
        return torch.where(branch_mask, branches, floor).amax(dim=1)


class AttFuseAdaptedFusion(_ControlledFusionBase):
    """Per-BEV-position masked-attention L+C adaptation of AttFuse."""

    def __init__(self, channels: int = 256, age_decay: float = 0.25) -> None:
        super().__init__(channels=channels, age_decay=age_decay)
        self.score_projection = nn.Conv2d(channels, 1, kernel_size=1)
        self.branch_bias = nn.Parameter(torch.zeros(1, self.branch_count, 1, 1))

    def _attention_weights(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        batch, _, _, height, width = branches.shape
        scores = self.score_projection(
            branches.reshape(batch * self.branch_count, self.channels, height, width)
        ).reshape(batch, self.branch_count, height, width)
        scores = (
            scores
            + self.branch_bias
            - self.age_decay * ages[:, :, None, None].to(dtype=scores.dtype)
        )
        support_map = support[:, :, None, None]
        scores = scores.masked_fill(~support_map, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        return torch.where(support_map, weights, torch.zeros_like(weights))

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(branches, support, ages)
        weights = self._attention_weights(branches, support, ages)
        return (branches * weights[:, :, None, :, :]).sum(dim=1)


class _RelativeAgeEncoding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, ages: Tensor) -> Tensor:
        return self.network(torch.log1p(ages).unsqueeze(-1))


class _BranchSelfAttention(nn.Module):
    """Self-attention over the four heterogeneous branches at every BEV cell."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, branches: Tensor, support: Tensor) -> Tensor:
        batch, branch_count, channels, height, width = branches.shape
        tokens = branches.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, branch_count, channels
        )
        key_padding_mask = (
            (~support)[:, None, None, :]
            .expand(batch, height, width, branch_count)
            .reshape(batch * height * width, branch_count)
        )
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.residual_dropout(attended)
        tokens = tokens + self.feed_forward(self.feed_forward_norm(tokens))
        result = tokens.reshape(
            batch,
            height,
            width,
            branch_count,
            channels,
        ).permute(0, 3, 4, 1, 2)
        return result * support[:, :, None, None, None].to(result.dtype)


def _partition_windows(
    features: Tensor,
    window_size: int,
) -> tuple[Tensor, Tensor, tuple[int, int, int, int, int]]:
    batch, channels, height, width = features.shape
    padded_height = math.ceil(height / window_size) * window_size
    padded_width = math.ceil(width / window_size) * window_size
    padded = F.pad(
        features,
        (0, padded_width - width, 0, padded_height - height),
    )
    rows = padded_height // window_size
    columns = padded_width // window_size
    windows = (
        padded.view(
            batch,
            channels,
            rows,
            window_size,
            columns,
            window_size,
        )
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch * rows * columns, window_size * window_size, channels)
    )

    valid = features.new_ones((batch, 1, height, width), dtype=torch.bool)
    valid = F.pad(
        valid,
        (0, padded_width - width, 0, padded_height - height),
        value=False,
    )
    valid_windows = (
        valid.view(
            batch,
            1,
            rows,
            window_size,
            columns,
            window_size,
        )
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch * rows * columns, window_size * window_size)
    )
    return (
        windows,
        valid_windows,
        (
            batch,
            height,
            width,
            padded_height,
            padded_width,
        ),
    )


def _unpartition_windows(
    windows: Tensor,
    window_size: int,
    shape: tuple[int, int, int, int, int],
) -> Tensor:
    batch, height, width, padded_height, padded_width = shape
    channels = windows.shape[-1]
    rows = padded_height // window_size
    columns = padded_width // window_size
    padded = (
        windows.view(
            batch,
            rows,
            columns,
            window_size,
            window_size,
            channels,
        )
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, padded_height, padded_width)
    )
    return padded[:, :, :height, :width]


class _MultiScaleWindowAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_sizes: tuple[int, ...],
        hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.window_sizes = window_sizes
        self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in window_sizes)
        self.attentions = nn.ModuleList(
            nn.MultiheadAttention(
                channels,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in window_sizes
        )
        self.scale_fuser = nn.Conv2d(
            channels * len(window_sizes),
            channels,
            kernel_size=1,
        )
        self.feed_forward_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        scale_outputs = []
        for window_size, norm, attention in zip(
            self.window_sizes,
            self.norms,
            self.attentions,
            strict=True,
        ):
            windows, valid, shape = _partition_windows(features, window_size)
            normalized = norm(windows)
            attended, _ = attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=~valid,
                need_weights=False,
            )
            scale_outputs.append(_unpartition_windows(attended, window_size, shape))
        features = features + self.residual_dropout(
            self.scale_fuser(torch.cat(scale_outputs, dim=1))
        )
        normalized = self.feed_forward_norm(features.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        return features + self.feed_forward(normalized)


class _FactorizedAxialAttention(nn.Module):
    """Row then column attention, the compact FAX adaptation used by CoBEVT."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.row_norm = nn.LayerNorm(channels)
        self.row_attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.column_norm = nn.LayerNorm(channels)
        self.column_attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        rows = features.permute(0, 2, 3, 1).reshape(
            batch * height,
            width,
            channels,
        )
        normalized_rows = self.row_norm(rows)
        attended_rows, _ = self.row_attention(
            normalized_rows,
            normalized_rows,
            normalized_rows,
            need_weights=False,
        )
        rows = rows + self.residual_dropout(attended_rows)
        features = rows.reshape(
            batch,
            height,
            width,
            channels,
        ).permute(0, 3, 1, 2)

        columns = features.permute(0, 3, 2, 1).reshape(
            batch * width,
            height,
            channels,
        )
        normalized_columns = self.column_norm(columns)
        attended_columns, _ = self.column_attention(
            normalized_columns,
            normalized_columns,
            normalized_columns,
            need_weights=False,
        )
        columns = columns + self.residual_dropout(attended_columns)
        features = columns.reshape(
            batch,
            width,
            height,
            channels,
        ).permute(0, 3, 2, 1)

        normalized = self.feed_forward_norm(features.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        return features + self.feed_forward(normalized)


class _WindowCrossAttention(nn.Module):
    """Local spatial cross-attention from one BEV query to all branches."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        branches: Tensor,
        support: Tensor,
    ) -> Tensor:
        batch, branch_count, channels, height, width = branches.shape
        query_windows, spatial_valid, shape = _partition_windows(
            query,
            self.window_size,
        )
        branch_windows, _, _ = _partition_windows(
            branches.reshape(
                batch * branch_count,
                channels,
                height,
                width,
            ),
            self.window_size,
        )
        window_count = query_windows.shape[0] // batch
        window_tokens = query_windows.shape[1]
        branch_windows = (
            branch_windows.reshape(
                batch,
                branch_count,
                window_count,
                window_tokens,
                channels,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(
                batch * window_count,
                branch_count * window_tokens,
                channels,
            )
        )
        key_valid = (
            support[:, None, :, None]
            & spatial_valid.reshape(
                batch,
                window_count,
                1,
                window_tokens,
            )
        ).reshape(
            batch * window_count,
            branch_count * window_tokens,
        )
        attended, _ = self.attention(
            self.query_norm(query_windows),
            self.memory_norm(branch_windows),
            self.memory_norm(branch_windows),
            key_padding_mask=~key_valid,
            need_weights=False,
        )
        query_windows = query_windows + self.residual_dropout(attended)
        return _unpartition_windows(query_windows, self.window_size, shape)


class V2XViTAdaptedFusion(_ControlledFusionBase):
    """V2X-ViT-style HMSA, relative time encoding, and MSwin fusion."""

    def __init__(
        self,
        channels: int = 256,
        *,
        age_decay: float = 0.25,
        num_heads: int | None = None,
        window_sizes: Sequence[int] = (2, 4),
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__(channels, age_decay)
        heads = _resolve_num_heads(self.channels, num_heads)
        windows = _window_sizes(window_sizes)
        dropout_value = _dropout_probability(dropout)
        hidden = _mlp_hidden_channels(self.channels, mlp_ratio)

        self.branch_embedding = nn.Parameter(
            torch.zeros(1, self.branch_count, self.channels)
        )
        nn.init.normal_(self.branch_embedding, std=0.02)
        self.relative_time_encoding = _RelativeAgeEncoding(self.channels)
        self.hmsa = _BranchSelfAttention(
            self.channels,
            heads,
            hidden,
            dropout_value,
        )
        self.multi_scale_window_attention = _MultiScaleWindowAttention(
            self.channels,
            heads,
            windows,
            hidden,
            dropout_value,
        )

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(
            branches,
            support,
            ages,
        )
        condition = (self.relative_time_encoding(ages) + self.branch_embedding)[
            :, :, :, None, None
        ]
        conditioned = branches + condition
        conditioned = conditioned * support[:, :, None, None, None].to(branches.dtype)
        attended = self.hmsa(conditioned, support)
        fused = self._weighted_mean(attended, support, ages)
        return self.multi_scale_window_attention(fused)


class CoBEVTAdaptedFusion(_ControlledFusionBase):
    """CoBEVT-style heterogeneous fusion followed by FAX attention."""

    def __init__(
        self,
        channels: int = 256,
        *,
        age_decay: float = 0.25,
        num_heads: int | None = None,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__(channels, age_decay)
        heads = _resolve_num_heads(self.channels, num_heads)
        dropout_value = _dropout_probability(dropout)
        hidden = _mlp_hidden_channels(self.channels, mlp_ratio)

        self.branch_embedding = nn.Parameter(
            torch.zeros(1, self.branch_count, self.channels)
        )
        nn.init.normal_(self.branch_embedding, std=0.02)
        self.relative_time_encoding = _RelativeAgeEncoding(self.channels)
        self.branch_fusion = _BranchSelfAttention(
            self.channels,
            heads,
            hidden,
            dropout_value,
        )
        self.fax = _FactorizedAxialAttention(
            self.channels,
            heads,
            hidden,
            dropout_value,
        )

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(
            branches,
            support,
            ages,
        )
        condition = (self.relative_time_encoding(ages) + self.branch_embedding)[
            :, :, :, None, None
        ]
        conditioned = branches + condition
        conditioned = conditioned * support[:, :, None, None, None].to(branches.dtype)
        attended = self.branch_fusion(conditioned, support)
        return self.fax(self._weighted_mean(attended, support, ages))


class CoFormerNetAdaptedFusion(_ControlledFusionBase):
    """Two-stage local-spatial then relative-temporal cross-attention."""

    def __init__(
        self,
        channels: int = 256,
        *,
        age_decay: float = 0.25,
        num_heads: int | None = None,
        window_size: int = 4,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__(channels, age_decay)
        heads = _resolve_num_heads(self.channels, num_heads)
        window = _positive_int("window_size", window_size)
        dropout_value = _dropout_probability(dropout)
        hidden = _mlp_hidden_channels(self.channels, mlp_ratio)

        self.branch_embedding = nn.Parameter(
            torch.zeros(1, self.branch_count, self.channels)
        )
        nn.init.normal_(self.branch_embedding, std=0.02)
        self.relative_time_encoding = _RelativeAgeEncoding(self.channels)
        self.spatial_cross_attention = _WindowCrossAttention(
            self.channels,
            heads,
            window,
            dropout_value,
        )
        self.temporal_query_norm = nn.LayerNorm(self.channels)
        self.temporal_memory_norm = nn.LayerNorm(self.channels)
        self.temporal_cross_attention = nn.MultiheadAttention(
            self.channels,
            heads,
            dropout=dropout_value,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(self.channels)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(self.channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(hidden, self.channels, kernel_size=1),
            nn.Dropout(dropout_value),
        )
        self.residual_dropout = nn.Dropout(dropout_value)

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(
            branches,
            support,
            ages,
        )
        batch, branch_count, channels, height, width = branches.shape
        condition = (self.relative_time_encoding(ages) + self.branch_embedding)[
            :, :, :, None, None
        ]
        memory = branches + condition
        memory = memory * support[:, :, None, None, None].to(branches.dtype)

        query = self._weighted_mean(branches, support, ages)
        query = self.spatial_cross_attention(query, memory, support)

        query_tokens = query.permute(0, 2, 3, 1).reshape(
            batch * height * width, 1, channels
        )
        memory_tokens = memory.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, branch_count, channels
        )
        key_padding_mask = (
            (~support)[:, None, None, :]
            .expand(batch, height, width, branch_count)
            .reshape(batch * height * width, branch_count)
        )
        attended, _ = self.temporal_cross_attention(
            self.temporal_query_norm(query_tokens),
            self.temporal_memory_norm(memory_tokens),
            self.temporal_memory_norm(memory_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + self.residual_dropout(attended)
        query = query_tokens.reshape(
            batch,
            height,
            width,
            channels,
        ).permute(0, 3, 1, 2)
        normalized = self.feed_forward_norm(query.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        return query + self.feed_forward(normalized)


class BEVFusionAdaptedFusion(_ControlledFusionBase):
    """BEVFusion ConvFuser adapted from two modalities to four V2X branches."""

    def __init__(
        self,
        channels: int = 256,
        *,
        age_decay: float = 0.25,
        hidden_channels: int | None = None,
        norm_groups: int | None = None,
    ) -> None:
        super().__init__(channels, age_decay)
        hidden = (
            self.channels
            if hidden_channels is None
            else _positive_int("hidden_channels", hidden_channels)
        )
        groups = _resolve_norm_groups(hidden, norm_groups)
        metadata_channels = self.branch_count * 2
        self.conv_fuser = nn.Sequential(
            nn.Conv2d(
                self.branch_count * self.channels + metadata_channels,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, self.channels, kernel_size=1),
        )

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(
            branches,
            support,
            ages,
        )
        batch, _, _, height, width = branches.shape
        age_weights = self._age_weights(support, ages)
        gated = branches * age_weights[:, :, None, None, None]
        feature_channels = gated.reshape(
            batch,
            self.branch_count * self.channels,
            height,
            width,
        )
        support_maps = (
            support[:, :, None, None]
            .expand(
                batch,
                self.branch_count,
                height,
                width,
            )
            .to(branches.dtype)
        )
        bounded_ages = torch.log1p(ages.float())
        bounded_ages = (bounded_ages / (1.0 + bounded_ages)).to(branches.dtype)
        age_maps = (
            bounded_ages[:, :, None, None].expand(
                batch, self.branch_count, height, width
            )
            * support_maps
        )
        fused = self.conv_fuser(
            torch.cat((feature_channels, support_maps, age_maps), dim=1)
        )
        return fused + self._weighted_mean(branches, support, ages)


def _warp_with_pixel_offsets(features: Tensor, offsets: Tensor) -> Tensor:
    """Backward-warp features with dense offsets expressed in BEV pixels."""

    if features.ndim != 4 or offsets.shape != (
        features.shape[0],
        2,
        features.shape[2],
        features.shape[3],
    ):
        raise ValueError("features and offsets have incompatible shapes")
    batch, _, height, width = features.shape
    y_coordinates = (
        torch.arange(height, device=features.device, dtype=features.dtype) + 0.5
    ) * (2.0 / height) - 1.0
    x_coordinates = (
        torch.arange(width, device=features.device, dtype=features.dtype) + 0.5
    ) * (2.0 / width) - 1.0
    y_grid, x_grid = torch.meshgrid(
        y_coordinates,
        x_coordinates,
        indexing="ij",
    )
    base_grid = torch.stack((x_grid, y_grid), dim=-1)
    base_grid = base_grid.unsqueeze(0).expand(batch, height, width, 2)
    normalized_offsets = torch.stack(
        (
            offsets[:, 0] * (2.0 / width),
            offsets[:, 1] * (2.0 / height),
        ),
        dim=-1,
    )
    return F.grid_sample(
        features,
        base_grid + normalized_offsets,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


class _SingleFrameFeatureFlowPredictor(nn.Module):
    """Predict a per-unit RSU flow and first-order feature derivative."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        norm_groups: int,
        max_displacement_per_age: float,
    ) -> None:
        super().__init__()
        self.max_displacement_per_age = max_displacement_per_age
        self.encoder = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.SiLU(),
        )
        self.offset_head = nn.Conv2d(
            hidden_channels,
            2,
            kernel_size=3,
            padding=1,
        )
        self.derivative_head = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.derivative_head.weight)
        nn.init.zeros_(self.derivative_head.bias)

    def forward(self, features: Tensor, ages: Tensor) -> Tensor:
        context = self.encoder(features)
        age_scale = ages[:, None, None, None].to(features.dtype)
        offsets = (
            torch.tanh(self.offset_head(context))
            * self.max_displacement_per_age
            * age_scale
        )
        predicted = _warp_with_pixel_offsets(features, offsets)
        return predicted + age_scale * self.derivative_head(context)


class FFNetAdaptedFusion(_ControlledFusionBase):
    """FFNet-style first-order delayed-RSU compensation and convolutional fusion.

    The paper estimates an infrastructure feature derivative from two
    consecutive RSU frames.  The controlled submission contract exposes only
    one already-selected feature per RSU modality, so that exact generator
    cannot be represented without adding hidden temporal inputs.  This minimal
    L+C adaptation instead predicts a per-unit dense offset and first-order
    derivative from each selected RSU feature itself, then evaluates that flow
    at its causal non-negative age.  It never predicts ego branches, never
    changes an age-zero branch, and never calls the ResilientV2X PTF or DER.
    """

    def __init__(
        self,
        channels: int = 256,
        *,
        age_decay: float = 0.0,
        hidden_channels: int | None = None,
        norm_groups: int | None = None,
        max_displacement_per_age: float = 1.0,
    ) -> None:
        super().__init__(channels, age_decay)
        hidden = (
            self.channels
            if hidden_channels is None
            else _positive_int("hidden_channels", hidden_channels)
        )
        groups = _resolve_norm_groups(hidden, norm_groups)
        max_displacement = _nonnegative_float(
            "max_displacement_per_age",
            max_displacement_per_age,
        )
        self.lidar_feature_flow = _SingleFrameFeatureFlowPredictor(
            self.channels,
            hidden,
            groups,
            max_displacement,
        )
        self.camera_feature_flow = _SingleFrameFeatureFlowPredictor(
            self.channels,
            hidden,
            groups,
            max_displacement,
        )
        self.conv_fuser = nn.Sequential(
            nn.Conv2d(
                self.branch_count * self.channels + self.branch_count,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, self.channels, kernel_size=1),
        )

    def _compensate_rsu_branches(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        compensated = [branches[:, 0]]
        for branch_index, predictor in (
            (1, self.lidar_feature_flow),
            (3, self.camera_feature_flow),
        ):
            feature = branches[:, branch_index]
            candidate = predictor(feature, ages[:, branch_index])
            propagate = (support[:, branch_index] & (ages[:, branch_index] > 0))[
                :, None, None, None
            ]
            predicted = torch.where(propagate, candidate, feature)
            if branch_index == 1:
                compensated.extend((predicted, branches[:, 2]))
            else:
                compensated.append(predicted)
        return torch.stack(compensated, dim=1)

    def forward(
        self,
        branches: Tensor,
        support: Tensor,
        ages: Tensor,
    ) -> Tensor:
        branches, support, ages = self._prepare_inputs(
            branches,
            support,
            ages,
        )
        batch, _, _, height, width = branches.shape
        compensated = self._compensate_rsu_branches(
            branches,
            support,
            ages,
        )
        feature_channels = compensated.reshape(
            batch,
            self.branch_count * self.channels,
            height,
            width,
        )
        support_maps = (
            support[:, :, None, None]
            .expand(
                batch,
                self.branch_count,
                height,
                width,
            )
            .to(branches.dtype)
        )
        fused = self.conv_fuser(torch.cat((feature_channels, support_maps), dim=1))
        return fused + self._weighted_mean(compensated, support, ages)


_CONTROLLED_BASELINES: dict[str, type[_ControlledFusionBase]] = {
    "ego_only": EgoOnlyFusion,
    "fcooper": FCooperAdaptedFusion,
    "attfuse": AttFuseAdaptedFusion,
    "v2x_vit": V2XViTAdaptedFusion,
    "cobevt": CoBEVTAdaptedFusion,
    "coformernet": CoFormerNetAdaptedFusion,
    "bevfusion": BEVFusionAdaptedFusion,
    "ffnet": FFNetAdaptedFusion,
}
_LAZY_CONTROLLED_BASELINES = {
    "late_fusion": (
        "transvision.models.resilient_v2x.late_fusion_baseline",
        "LateFusionStyleFusion",
    ),
    "v2vnet": (
        "transvision.models.resilient_v2x.v2vnet_baseline",
        "V2VNetStyleFusion",
    ),
    "disconet": (
        "transvision.models.resilient_v2x.disconet_baseline",
        "DiscoNetAdaptedFusion",
    ),
    "when2com": (
        "transvision.models.resilient_v2x.when2com_baseline",
        "When2comAdaptedFusion",
    ),
    "where2comm": (
        "transvision.models.resilient_v2x.where2comm_baseline",
        "Where2commAdaptedFusion",
    ),
    "how2comm": (
        "transvision.models.resilient_v2x.how2comm_baseline",
        "How2commAdaptedFusion",
    ),
}


def _controlled_baseline_type(name: str) -> type[nn.Module]:
    direct = _CONTROLLED_BASELINES.get(name)
    if direct is not None:
        return direct
    module_name, class_name = _LAZY_CONTROLLED_BASELINES[name]
    candidate = getattr(import_module(module_name), class_name, None)
    if not isinstance(candidate, type) or not issubclass(candidate, nn.Module):
        raise RuntimeError(f"controlled baseline {name!r} is not an nn.Module type")
    return candidate


def build_controlled_baseline_fusion(
    name: str,
    channels: int = 256,
    **kwargs: object,
) -> nn.Module:
    """Build one adapted fusion module with the common controlled contract."""

    choices_by_name = (*_CONTROLLED_BASELINES, *_LAZY_CONTROLLED_BASELINES)
    if type(name) is not str or name not in choices_by_name:
        choices = ", ".join(choices_by_name)
        raise ValueError(f"name must be one of: {choices}")
    return _controlled_baseline_type(name)(channels=channels, **kwargs)


__all__ = (
    "BRANCH_ORDER",
    "EgoOnlyFusion",
    "FCooperAdaptedFusion",
    "AttFuseAdaptedFusion",
    "V2XViTAdaptedFusion",
    "CoBEVTAdaptedFusion",
    "CoFormerNetAdaptedFusion",
    "BEVFusionAdaptedFusion",
    "FFNetAdaptedFusion",
    "build_controlled_baseline_fusion",
)
