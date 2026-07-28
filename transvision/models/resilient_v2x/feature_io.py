from __future__ import annotations

import torch
from torch import Tensor


def bev_xy_to_yx(features: Tensor) -> Tensor:
    """Convert LSS ``[B,C,X,Y]`` output to the project ``[B,C,Y,X]`` grid."""

    if not isinstance(features, Tensor) or features.ndim != 4:
        raise ValueError("LSS BEV features must have shape [B,C,X,Y]")
    if not features.is_floating_point():
        raise ValueError("LSS BEV features must be floating")
    return features.transpose(-2, -1).contiguous()


def scatter_sparse_bev_history(
    encoded: Tensor,
    owner: Tensor,
    availability: Tensor,
    *,
    channels: int = 256,
) -> Tensor:
    """Scatter encoded causal payloads into ``[B,2,4,C,H,W]``.

    ``owner`` rows are ``[batch, agent, horizon]``. Exact coverage is
    required: every true availability slot has one encoded payload and no
    unavailable slot may be represented.
    """

    if not isinstance(encoded, Tensor) or encoded.ndim != 4:
        raise ValueError("encoded must have shape [K,C,H,W]")
    if not encoded.is_floating_point():
        raise ValueError("encoded must be floating")
    if type(channels) is not int or channels <= 0:
        raise ValueError("channels must be a positive integer")
    if encoded.shape[1] != channels:
        raise ValueError("encoded channel count does not match channels")
    if encoded.shape[2] <= 0 or encoded.shape[3] <= 0:
        raise ValueError("encoded spatial dimensions must be positive")
    if (
        not isinstance(owner, Tensor)
        or owner.shape != (encoded.shape[0], 3)
        or owner.dtype is torch.bool
        or owner.is_floating_point()
        or owner.is_complex()
        or owner.device != encoded.device
    ):
        raise ValueError("owner must be an integer [K,3] tensor beside encoded")
    if (
        not isinstance(availability, Tensor)
        or availability.ndim != 3
        or availability.shape[1:] != (2, 4)
        or availability.dtype is not torch.bool
        or availability.device != encoded.device
    ):
        raise ValueError("availability must be boolean [B,2,4]")
    batch = availability.shape[0]
    if batch <= 0:
        raise ValueError("availability batch must be positive")

    if owner.numel():
        if (
            owner[:, 0].lt(0).any().item()
            or owner[:, 0].ge(batch).any().item()
            or owner[:, 1].lt(0).any().item()
            or owner[:, 1].gt(1).any().item()
            or owner[:, 2].lt(0).any().item()
            or owner[:, 2].gt(3).any().item()
        ):
            raise ValueError("owner indices are out of range")
        linear_owner = owner[:, 0] * 8 + owner[:, 1] * 4 + owner[:, 2]
        if torch.unique(linear_owner).numel() != linear_owner.numel():
            raise ValueError("owner rows must be unique")
        expected = availability.reshape(-1).nonzero(as_tuple=False).flatten()
        if not torch.equal(torch.sort(linear_owner).values, expected):
            raise ValueError("owner must exactly cover available slots")
    elif availability.any().item():
        raise ValueError("available slots require encoded payloads")

    flat = encoded.new_zeros(
        batch * 8,
        channels,
        encoded.shape[2],
        encoded.shape[3],
    )
    if encoded.shape[0]:
        flat = flat.index_copy(0, linear_owner.to(torch.long), encoded)
    history = flat.view(
        batch,
        2,
        4,
        channels,
        encoded.shape[2],
        encoded.shape[3],
    )
    if not torch.equal(
        history.detach().ne(0).flatten(start_dim=3).any(dim=3) & ~availability,
        torch.zeros_like(availability),
    ):
        raise RuntimeError("unavailable history slots must remain zero")
    return history


__all__ = ("bev_xy_to_yx", "scatter_sparse_bev_history")
