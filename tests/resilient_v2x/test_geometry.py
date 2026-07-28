import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import transvision.models.resilient_v2x.geometry as geometry_module
from transvision.models.resilient_v2x.geometry import (
    BEVGridSpec,
    align_bev_to_target,
    build_backward_grid,
    compose_source_to_target,
    warp_with_displacement,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC = BEVGridSpec(x_min=0.0, y_min=-2.5, resolution=1.0, height=5, width=5)


def impulse() -> torch.Tensor:
    value = torch.zeros(1, 1, 5, 5)
    value[0, 0, 2, 2] = 1.0
    return value


def test_grid_spec_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        SPEC.width = 6


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x_min", math.nan, "finite"),
        ("y_min", math.inf, "finite"),
        ("resolution", math.nan, "finite"),
        ("resolution", 0.0, "positive"),
        ("resolution", -1.0, "positive"),
        ("height", True, "positive integer"),
        ("height", 5.0, "positive integer"),
        ("height", 0, "positive integer"),
        ("width", False, "positive integer"),
        ("width", 5.0, "positive integer"),
        ("width", -1, "positive integer"),
    ],
)
def test_grid_spec_rejects_invalid_values(
    field: str,
    value: object,
    message: str,
) -> None:
    values = {
        "x_min": 0.0,
        "y_min": -2.5,
        "resolution": 1.0,
        "height": 5,
        "width": 5,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        BEVGridSpec(**values)


def test_identity_alignment_is_exact() -> None:
    actual = align_bev_to_target(impulse(), torch.eye(4).unsqueeze(0), SPEC)
    torch.testing.assert_close(actual, impulse(), atol=0.0, rtol=0.0)


def test_low_precision_alignment_builds_geometry_in_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = impulse().to(dtype=torch.float16)
    transform = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    captured: dict[str, torch.dtype] = {}

    def fake_grid_sample(
        value: torch.Tensor,
        grid: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["source"] = value.dtype
        captured["grid"] = grid.dtype
        return value.clone()

    monkeypatch.setattr(geometry_module.F, "grid_sample", fake_grid_sample)

    actual = align_bev_to_target(source, transform, SPEC)

    assert actual.dtype is torch.float16
    assert captured == {"source": torch.float16, "grid": torch.float16}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_amp_alignment_accepts_fp16_feature_and_fp32_transform() -> None:
    source = impulse().cuda().half().requires_grad_()
    transform = torch.eye(4, device="cuda", dtype=torch.float32).unsqueeze(0)

    with torch.autocast("cuda", dtype=torch.float16):
        actual = align_bev_to_target(source, transform, SPEC)
        loss = actual.float().square().sum()
    loss.backward()

    assert actual.dtype is torch.float16
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_positive_x_translation_moves_impulse_one_column() -> None:
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 3] = 1.0
    actual = align_bev_to_target(impulse(), transform, SPEC)
    assert actual[0, 0, 2, 3].item() == pytest.approx(1.0)


def test_positive_quarter_turn_rotates_source_counter_clockwise() -> None:
    symmetric = BEVGridSpec(
        x_min=-2.5,
        y_min=-2.5,
        resolution=1.0,
        height=5,
        width=5,
    )
    source = torch.zeros(1, 1, 5, 5)
    source[0, 0, 2, 3] = 1.0
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 0] = 0.0
    transform[:, 0, 1] = -1.0
    transform[:, 1, 0] = 1.0
    transform[:, 1, 1] = 0.0
    actual = align_bev_to_target(source, transform, symmetric)
    assert actual[0, 0, 3, 2].item() == pytest.approx(1.0)


def test_compose_source_to_target_uses_inverse_target_times_source() -> None:
    source_world_from_agent = torch.tensor(
        [
            [0.0, -1.0, 0.0, 13.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    target_world_from_ego = torch.tensor(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    expected = torch.tensor(
        [
            [0.0, -1.0, 0.0, 3.0],
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    actual = compose_source_to_target(
        source_world_from_agent,
        target_world_from_ego,
    )
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_backward_grid_uses_exact_identity_cell_centers_and_uv_order() -> None:
    small_spec = BEVGridSpec(
        x_min=0.0,
        y_min=0.0,
        resolution=1.0,
        height=2,
        width=3,
    )
    actual = build_backward_grid(
        torch.eye(4, dtype=torch.float64).unsqueeze(0),
        small_spec,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    expected = torch.tensor(
        [
            [
                [
                    [-0.6666666666666667, -0.5],
                    [0.0, -0.5],
                    [0.6666666666666667, -0.5],
                ],
                [
                    [-0.6666666666666667, 0.5],
                    [0.0, 0.5],
                    [0.6666666666666667, 0.5],
                ],
            ]
        ],
        dtype=torch.float64,
    )

    assert actual.shape == (1, 2, 3, 2)
    assert actual.dtype is torch.float64
    assert actual.device == torch.device("cpu")
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_backward_displacement_uses_x_then_y_channels() -> None:
    displacement = torch.zeros(1, 2, 5, 5)
    displacement[:, 0] = -1.0
    actual = warp_with_displacement(impulse(), displacement)
    assert actual[0, 0, 2, 3].item() == pytest.approx(1.0)


def test_non_finite_transform_fails_closed() -> None:
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        align_bev_to_target(impulse(), transform, SPEC)


def test_singular_transform_fails_closed() -> None:
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 3, 3] = 0.0
    with pytest.raises(ValueError, match="invertible"):
        align_bev_to_target(impulse(), transform, SPEC)


@pytest.mark.parametrize(
    "transform",
    [
        torch.eye(4),
        torch.ones(1, 3, 4),
        torch.eye(4, dtype=torch.int64).unsqueeze(0),
    ],
    ids=["rank", "shape", "dtype"],
)
def test_backward_grid_rejects_invalid_transform(transform: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"\[B,4,4\]|floating"):
        build_backward_grid(
            transform,
            SPEC,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_backward_grid_rejects_non_floating_requested_dtype() -> None:
    with pytest.raises(ValueError, match="floating"):
        build_backward_grid(
            torch.eye(4).unsqueeze(0),
            SPEC,
            dtype=torch.int64,
            device=torch.device("cpu"),
        )


def test_backward_grid_rejects_invalid_grid_spec_object() -> None:
    with pytest.raises(ValueError, match="BEVGridSpec"):
        build_backward_grid(
            torch.eye(4).unsqueeze(0),
            object(),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_backward_grid_requires_transform_to_match_requested_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        build_backward_grid(
            torch.eye(4, dtype=torch.float64).unsqueeze(0),
            SPEC,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        (
            torch.eye(4).repeat(2, 1, 1),
            torch.eye(4).unsqueeze(0),
            "batch",
        ),
        (
            torch.eye(4, dtype=torch.float64).unsqueeze(0),
            torch.eye(4).unsqueeze(0),
            "dtype",
        ),
    ],
    ids=["batch", "dtype"],
)
def test_composition_requires_matching_transform_inputs(
    source: torch.Tensor,
    target: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compose_source_to_target(source, target)


@pytest.mark.parametrize("singular_input", ["source", "target"])
def test_composition_requires_each_transform_to_be_invertible(
    singular_input: str,
) -> None:
    source = torch.eye(4).unsqueeze(0)
    target = torch.eye(4).unsqueeze(0)
    if singular_input == "source":
        source[:, 3, 3] = 0.0
    else:
        target[:, 3, 3] = 0.0

    with pytest.raises(ValueError, match="invertible"):
        compose_source_to_target(source, target)


@pytest.mark.parametrize(
    ("source", "transform", "message"),
    [
        (torch.ones(1, 5, 5), torch.eye(4).unsqueeze(0), "rank-4"),
        (
            torch.ones(1, 1, 4, 5),
            torch.eye(4).unsqueeze(0),
            "spatial shape",
        ),
        (
            torch.ones(2, 1, 5, 5),
            torch.eye(4).unsqueeze(0),
            "batch",
        ),
        (
            torch.ones(1, 1, 5, 5, dtype=torch.int64),
            torch.eye(4).unsqueeze(0),
            "floating",
        ),
        (
            torch.ones(1, 1, 5, 5, dtype=torch.float64),
            torch.eye(4).unsqueeze(0),
            "dtype",
        ),
    ],
    ids=["rank", "spatial-shape", "batch", "source-dtype", "dtype-mismatch"],
)
def test_alignment_rejects_invalid_inputs(
    source: torch.Tensor,
    transform: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        align_bev_to_target(source, transform, SPEC)


@pytest.mark.parametrize(
    ("source", "displacement", "message"),
    [
        (torch.ones(1, 5, 5), torch.zeros(1, 2, 5, 5), "rank-4"),
        (torch.ones(1, 1, 5, 5), torch.zeros(1, 5, 5), "rank-4"),
        (torch.ones(1, 1, 5, 5), torch.zeros(1, 3, 5, 5), r"\[B,2,Y,X\]"),
        (
            torch.ones(1, 1, 5, 5, dtype=torch.int64),
            torch.zeros(1, 2, 5, 5),
            "floating",
        ),
        (
            torch.ones(1, 1, 5, 5),
            torch.zeros(1, 2, 5, 5, dtype=torch.float64),
            "dtype",
        ),
    ],
    ids=[
        "source-rank",
        "displacement-rank",
        "shape",
        "source-dtype",
        "dtype-mismatch",
    ],
)
def test_displacement_warp_rejects_invalid_inputs(
    source: torch.Tensor,
    displacement: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        warp_with_displacement(source, displacement)


def test_displacement_warp_preserves_finite_source_gradient() -> None:
    source = impulse().requires_grad_()
    displacement = torch.zeros(1, 2, 5, 5)

    warp_with_displacement(source, displacement).square().sum().backward()

    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    assert source.grad[0, 0, 2, 2].item() == pytest.approx(2.0)


def test_package_import_does_not_load_voxel_or_bev_pool_custom_ops() -> None:
    code = (
        "import sys; "
        "import transvision.models.resilient_v2x; "
        "assert 'transvision.models.bev_pool' not in sys.modules; "
        "assert 'transvision.models.voxel.voxel_layer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
