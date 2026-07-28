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


def test_lss_camera_projection_uses_general_affine_inverse() -> None:
    pytest.importorskip("mmdet3d")
    from transvision.models.detectors.resilient_v2x import (
        SharedResNetLSSBEVEncoder,
    )

    class CaptureViewTransform(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lidar_to_image: torch.Tensor | None = None

        def forward(
            self,
            image_features: torch.Tensor,
            _points: object,
            lidar_to_image: torch.Tensor,
            *_args: object,
        ) -> torch.Tensor:
            self.lidar_to_image = lidar_to_image.detach().clone()
            return image_features.new_zeros(image_features.shape[0], 256, 1, 1)

    capture = CaptureViewTransform()
    encoder = SharedResNetLSSBEVEncoder(
        image_backbone=torch.nn.Identity(),
        image_neck=torch.nn.Identity(),
        view_transform=capture,
        output_height=1,
        output_width=1,
        view_transform_output_order="yx",
    )
    intrinsics = torch.tensor(
        [[[2.0, 0.0, 0.5], [0.0, 3.0, 0.25], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    camera_to_agent = torch.tensor(
        [
            [
                [1.0, 0.2, 0.0, 2.0],
                [0.0, 0.8, 0.0, -1.0],
                [0.0, 0.0, 1.1, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float64,
    )

    output = encoder(
        torch.zeros(1, 3, 2, 2, dtype=torch.float64),
        intrinsics,
        camera_to_agent,
    )

    assert output.shape == (1, 256, 1, 1)
    assert capture.lidar_to_image is not None
    intrinsic4 = torch.eye(4, dtype=torch.float64)
    intrinsic4[:3, :3] = intrinsics[0]
    expected = intrinsic4 @ torch.linalg.inv(camera_to_agent[0])
    transpose_shortcut = intrinsic4 @ camera_to_agent[0].T
    assert torch.allclose(capture.lidar_to_image[0, 0], expected)
    assert not torch.allclose(capture.lidar_to_image[0, 0], transpose_shortcut)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lss_encoder_normalizes_fp32_bev_to_active_amp_dtype() -> None:
    pytest.importorskip("mmdet3d")
    from transvision.models.detectors.resilient_v2x import (
        SharedResNetLSSBEVEncoder,
    )

    class Float32ViewTransform(torch.nn.Module):
        def forward(
            self,
            image_features: torch.Tensor,
            *_args: object,
        ) -> torch.Tensor:
            return torch.zeros(
                image_features.shape[0],
                256,
                1,
                1,
                device=image_features.device,
                dtype=torch.float32,
            )

    encoder = SharedResNetLSSBEVEncoder(
        image_backbone=torch.nn.Identity(),
        image_neck=torch.nn.Identity(),
        view_transform=Float32ViewTransform(),
        output_height=1,
        output_width=1,
        view_transform_output_order="yx",
    ).cuda()
    images = torch.zeros(1, 3, 2, 2, device="cuda")
    intrinsics = torch.eye(3, device="cuda").unsqueeze(0)
    camera_to_agent = torch.eye(4, device="cuda").unsqueeze(0)

    with torch.autocast("cuda", dtype=torch.float16):
        output = encoder(images, intrinsics, camera_to_agent)

    assert output.dtype is torch.float16


def test_lss_fixed_geometry_is_registered_as_buffers() -> None:
    pytest.importorskip("mmdet3d")
    from transvision.dataset.transforms.depth_lss import LSSTransform

    transform = LSSTransform(
        in_channels=4,
        out_channels=4,
        image_size=(8, 12),
        feature_size=(2, 3),
        xbound=(0.0, 4.0, 1.0),
        ybound=(-2.0, 2.0, 1.0),
        zbound=(-1.0, 1.0, 2.0),
        dbound=(1.0, 3.0, 1.0),
    )

    parameters = dict(transform.named_parameters())
    buffers = dict(transform.named_buffers())
    assert {"dx", "bx", "nx", "frustum"}.issubset(buffers)
    assert not {"dx", "bx", "nx", "frustum"}.intersection(parameters)
    assert transform.nx.dtype == torch.long
    assert all(parameter.is_floating_point() for parameter in parameters.values())


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
