from __future__ import annotations

import pytest
import torch

from transvision.models.resilient_v2x import (
    bev_xy_to_yx,
    scatter_sparse_bev_history,
)


def test_lss_xy_grid_is_transposed_to_shared_yx_convention() -> None:
    source = torch.arange(6, dtype=torch.float32).view(1, 1, 2, 3)

    converted = bev_xy_to_yx(source)

    assert converted.shape == (1, 1, 3, 2)
    assert torch.equal(converted[0, 0], source[0, 0].T)
    assert converted.is_contiguous()


def test_sparse_bev_scatter_exactly_places_features_and_gradients() -> None:
    encoded = torch.randn(3, 256, 2, 3, requires_grad=True)
    owner = torch.tensor([[0, 0, 0], [0, 1, 2], [1, 0, 3]])
    availability = torch.zeros(2, 2, 4, dtype=torch.bool)
    availability[0, 0, 0] = True
    availability[0, 1, 2] = True
    availability[1, 0, 3] = True

    history = scatter_sparse_bev_history(encoded, owner, availability)

    assert history.shape == (2, 2, 4, 256, 2, 3)
    assert torch.equal(history[0, 0, 0], encoded[0])
    assert torch.equal(history[0, 1, 2], encoded[1])
    assert torch.count_nonzero(history[1, 1]).item() == 0
    history.sum().backward()
    assert torch.equal(encoded.grad, torch.ones_like(encoded))


@pytest.mark.parametrize(
    "owner",
    (
        torch.tensor([[0, 0, 0], [0, 0, 0]]),
        torch.tensor([[0, 0, 0], [0, 1, 1]]),
        torch.tensor([[0, 0, 0], [0, 2, 0]]),
    ),
)
def test_sparse_bev_scatter_rejects_duplicate_missing_or_out_of_range_owner(
    owner: torch.Tensor,
) -> None:
    encoded = torch.randn(2, 256, 2, 2)
    availability = torch.zeros(1, 2, 4, dtype=torch.bool)
    availability[0, 0, 0] = True
    availability[0, 1, 0] = True

    with pytest.raises(ValueError):
        scatter_sparse_bev_history(encoded, owner, availability)


def test_sparse_bev_scatter_supports_all_empty_batch() -> None:
    encoded = torch.empty(0, 256, 2, 2)
    owner = torch.empty(0, 3, dtype=torch.long)
    availability = torch.zeros(2, 2, 4, dtype=torch.bool)

    history = scatter_sparse_bev_history(encoded, owner, availability)

    assert history.shape == (2, 2, 4, 256, 2, 2)
    assert torch.count_nonzero(history).item() == 0
