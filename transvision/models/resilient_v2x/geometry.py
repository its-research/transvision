from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class BEVGridSpec:
    x_min: float
    y_min: float
    resolution: float
    height: int
    width: int

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "resolution"):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        for name in ("height", "width"):
            value = getattr(self, name)
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")


def _validate_spec(spec: BEVGridSpec) -> None:
    if not isinstance(spec, BEVGridSpec):
        raise ValueError("spec must be a BEVGridSpec")
    spec.__post_init__()


def _inverse_transform(transform: Tensor, name: str) -> Tensor:
    if not isinstance(transform, Tensor) or (
        transform.ndim != 3 or transform.shape[1:] != (4, 4)
    ):
        raise ValueError(f"{name} must have shape [B,4,4]")
    if not transform.is_floating_point():
        raise ValueError(f"{name} must be floating")
    if not torch.isfinite(transform).all().item():
        raise ValueError(f"{name} must contain only finite values")
    inverse, info = torch.linalg.inv_ex(transform)
    if info.ne(0).any().item() or not torch.isfinite(inverse).all().item():
        raise ValueError(f"{name} must be invertible")
    return inverse


def compose_source_to_target(
    source_world_from_agent: Tensor,
    target_world_from_ego: Tensor,
) -> Tensor:
    _inverse_transform(
        source_world_from_agent,
        "source_world_from_agent",
    )
    target_inverse = _inverse_transform(
        target_world_from_ego,
        "target_world_from_ego",
    )
    if source_world_from_agent.shape[0] != target_world_from_ego.shape[0]:
        raise ValueError("transform inputs must have matching batch sizes")
    if source_world_from_agent.dtype != target_world_from_ego.dtype:
        raise ValueError("transform inputs must have matching dtype")
    if source_world_from_agent.device != target_world_from_ego.device:
        raise ValueError("transform inputs must have matching device")
    return target_inverse @ source_world_from_agent


def build_backward_grid(
    source_to_target: Tensor,
    spec: BEVGridSpec,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    _validate_spec(spec)
    if not dtype.is_floating_point:
        raise ValueError("grid dtype must be floating")
    target_to_source = _inverse_transform(source_to_target, "source_to_target")
    if source_to_target.dtype != dtype:
        raise ValueError("transform dtype must match requested dtype")
    if source_to_target.device != device:
        raise ValueError("transform device must match requested device")

    rows = torch.arange(spec.height, dtype=dtype, device=device)
    cols = torch.arange(spec.width, dtype=dtype, device=device)
    y = spec.y_min + (rows + 0.5) * spec.resolution
    x = spec.x_min + (cols + 0.5) * spec.resolution
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    target_centers = torch.stack(
        (
            x_grid,
            y_grid,
            torch.zeros_like(x_grid),
            torch.ones_like(x_grid),
        ),
        dim=-1,
    )
    target_centers = target_centers.unsqueeze(0).expand(
        source_to_target.shape[0],
        -1,
        -1,
        -1,
    )
    source_centers = torch.einsum(
        "bij,bhwj->bhwi",
        target_to_source,
        target_centers,
    )
    source_cols = (source_centers[..., 0] - spec.x_min) / spec.resolution - 0.5
    source_rows = (source_centers[..., 1] - spec.y_min) / spec.resolution - 0.5
    u = 2.0 * (source_cols + 0.5) / spec.width - 1.0
    v = 2.0 * (source_rows + 0.5) / spec.height - 1.0
    grid = torch.stack((u, v), dim=-1)
    if not torch.isfinite(grid).all().item():
        raise ValueError("backward grid must contain only finite values")
    return grid


def align_bev_to_target(
    source: Tensor,
    source_to_target: Tensor,
    spec: BEVGridSpec,
) -> Tensor:
    _validate_spec(spec)
    if not isinstance(source, Tensor) or source.ndim != 4:
        raise ValueError("source must be a rank-4 tensor [B,C,Y,X]")
    if not source.is_floating_point():
        raise ValueError("source must be floating")
    if source.shape[2:] != (spec.height, spec.width):
        raise ValueError("source spatial shape must match the grid spec")
    if not isinstance(source_to_target, Tensor) or source_to_target.ndim != 3:
        raise ValueError("source_to_target must have shape [B,4,4]")
    if source.shape[0] != source_to_target.shape[0]:
        raise ValueError("source and transform must have matching batch sizes")
    if source.dtype != source_to_target.dtype:
        raise ValueError("source and transform must have matching dtype")
    if source.device != source_to_target.device:
        raise ValueError("source and transform must have matching device")
    grid = build_backward_grid(
        source_to_target,
        spec,
        dtype=source.dtype,
        device=source.device,
    )
    return F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def warp_with_displacement(
    source: Tensor,
    displacement_cells: Tensor,
) -> Tensor:
    if (
        not isinstance(source, Tensor)
        or not isinstance(displacement_cells, Tensor)
        or source.ndim != 4
        or displacement_cells.ndim != 4
    ):
        raise ValueError("source and displacement must be rank-4 tensors")
    batch, _, height, width = source.shape
    if displacement_cells.shape != (batch, 2, height, width):
        raise ValueError("displacement shape must be [B,2,Y,X]")
    if (
        not source.is_floating_point()
        or not displacement_cells.is_floating_point()
    ):
        raise ValueError("source and displacement must be floating")
    if source.dtype != displacement_cells.dtype:
        raise ValueError("source and displacement must have matching dtype")
    if source.device != displacement_cells.device:
        raise ValueError("source and displacement must have matching device")
    rows = torch.arange(height, dtype=source.dtype, device=source.device)
    cols = torch.arange(width, dtype=source.dtype, device=source.device)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    x = col_grid.unsqueeze(0) + displacement_cells[:, 0]
    y = row_grid.unsqueeze(0) + displacement_cells[:, 1]
    u = 2.0 * (x + 0.5) / width - 1.0
    v = 2.0 * (y + 0.5) / height - 1.0
    grid = torch.stack((u, v), dim=-1)
    return F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
