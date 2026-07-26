import math
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from transvision.models.resilient_v2x.ptf import (
    HorizonConditionedPTF,
    PTFOutput,
)


def make_ptf(mode: str = "nonlinear") -> HorizonConditionedPTF:
    return HorizonConditionedPTF(
        in_channels=256,
        history_positions=4,
        history_limit=3,
        projected_channels=64,
        context_channels=128,
        max_low_resolution_cells=8.0,
        mode=mode,
    ).eval()


def make_history(batch: int = 1, size: int = 16) -> torch.Tensor:
    return torch.randn(batch, 2, 4, 256, size, size)


def make_context(batch: int = 1, size: int = 4) -> torch.Tensor:
    return torch.randn(batch, 128, size, size)


def set_nonzero_final_heads(model: HorizonConditionedPTF) -> None:
    with torch.no_grad():
        model.displacement_head[-1].weight.fill_(0.002)
        model.displacement_head[-1].bias.fill_(0.2)
        model.confidence_head[-1].weight.fill_(0.002)
        model.confidence_head[-1].bias.fill_(-0.1)


def test_ptf_output_is_frozen() -> None:
    output = PTFOutput(
        displacement=torch.zeros(1, 2, 4, 4),
        confidence=torch.zeros(1, 1, 4, 4),
    )

    with pytest.raises(FrozenInstanceError):
        output.displacement = torch.ones_like(output.displacement)


def test_projected_spatial_shape_locks_production_resolution() -> None:
    model = make_ptf()

    assert model.projected_spatial_shape((288, 288)) == (72, 72)
    assert model.projected_spatial_shape((16, 32)) == (4, 8)


@pytest.mark.parametrize(
    "shape",
    [
        object(),
        (16,),
        (16, 16, 16),
        (True, 16),
        (16.0, 16),
        (0, 16),
        (-4, 16),
        (15, 16),
    ],
    ids=[
        "not-sequence",
        "too-short",
        "too-long",
        "boolean",
        "non-integer",
        "zero",
        "negative",
        "not-divisible-by-four",
    ],
)
def test_projected_spatial_shape_rejects_invalid_shapes(shape: object) -> None:
    with pytest.raises(ValueError):
        make_ptf().projected_spatial_shape(shape)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("in_channels", 255),
        ("in_channels", True),
        ("history_positions", 3),
        ("history_positions", True),
        ("history_limit", 2),
        ("history_limit", True),
        ("projected_channels", 32),
        ("projected_channels", True),
        ("context_channels", 64),
        ("context_channels", True),
        ("max_low_resolution_cells", True),
        ("max_low_resolution_cells", 0.0),
        ("max_low_resolution_cells", -1.0),
        ("max_low_resolution_cells", math.nan),
        ("max_low_resolution_cells", math.inf),
        ("mode", "quadratic"),
        ("mode", 1),
    ],
)
def test_constructor_rejects_unapproved_values(
    field: str,
    value: object,
) -> None:
    values = {
        "in_channels": 256,
        "history_positions": 4,
        "history_limit": 3,
        "projected_channels": 64,
        "context_channels": 128,
        "max_low_resolution_cells": 8.0,
        "mode": "nonlinear",
    }
    values[field] = value

    with pytest.raises(ValueError):
        HorizonConditionedPTF(**values)


def test_exact_module_inventory_biases_and_zero_initialization() -> None:
    model = make_ptf()
    convolutions = [
        module for module in model.modules() if isinstance(module, nn.Conv2d)
    ]
    normalizations = [
        module for module in model.modules() if isinstance(module, nn.GroupNorm)
    ]
    linears = [
        module for module in model.modules() if isinstance(module, nn.Linear)
    ]
    embeddings = [
        module for module in model.modules() if isinstance(module, nn.Embedding)
    ]
    residual_blocks = [
        module
        for module in model.modules()
        if type(module).__name__ == "_FiLMResidualBlock"
    ]

    assert len(convolutions) == 13
    assert sum(module.bias is None for module in convolutions) == 11
    assert sum(module.bias is not None for module in convolutions) == 2
    assert len(normalizations) == 11
    assert len(linears) == 2
    assert all(module.bias is not None for module in linears)
    assert len(embeddings) == 2
    assert len(residual_blocks) == 3
    assert len({id(module) for module in residual_blocks}) == 3
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == 1_577_923

    projection_convolutions = [
        module
        for module in model.history_projection
        if isinstance(module, nn.Conv2d)
    ]
    assert [module.bias for module in projection_convolutions] == [None, None]
    assert model.context_stem[0].bias is None
    assert model.displacement_head[0].bias is None
    assert model.confidence_head[0].bias is None
    assert model.displacement_head[-1].bias is not None
    assert model.confidence_head[-1].bias is not None
    assert all(
        normalization.affine
        for block in model.residual_blocks
        for normalization in (block.normalization1, block.normalization2)
    )

    torch.testing.assert_close(
        model.film_mlp[-1].weight,
        torch.zeros_like(model.film_mlp[-1].weight),
    )
    torch.testing.assert_close(
        model.film_mlp[-1].bias,
        torch.zeros_like(model.film_mlp[-1].bias),
    )
    for head in (model.displacement_head, model.confidence_head):
        torch.testing.assert_close(
            head[-1].weight,
            torch.zeros_like(head[-1].weight),
        )
        torch.testing.assert_close(
            head[-1].bias,
            torch.zeros_like(head[-1].bias),
        )


def test_ptf_initial_output_and_shape() -> None:
    model = make_ptf()
    history = make_history(size=32)
    availability = torch.ones(1, 2, 4, dtype=torch.bool)

    context = model.build_context(history, availability)
    output = model.query(context, torch.tensor([0]), torch.tensor([3]))

    assert context.shape == (1, 128, 8, 8)
    assert output.displacement.shape == (1, 2, 32, 32)
    assert output.confidence.shape == (1, 1, 32, 32)
    torch.testing.assert_close(
        output.displacement,
        torch.zeros_like(output.displacement),
    )
    torch.testing.assert_close(
        output.confidence,
        torch.full_like(output.confidence, 0.5),
    )


def test_build_context_gathers_only_valid_rows_in_flatten_order() -> None:
    model = make_ptf()
    history = make_history()
    availability = torch.tensor(
        [[[True, False, True, False], [False, True, False, False]]]
    )
    captured: list[torch.Tensor] = []

    def capture_projection_input(
        _module: nn.Module,
        inputs: tuple[torch.Tensor],
    ) -> None:
        captured.append(inputs[0].detach().clone())

    handle = model.history_projection.register_forward_pre_hook(
        capture_projection_input
    )
    try:
        model.build_context(history, availability)
    finally:
        handle.remove()

    valid_index = availability.reshape(8).nonzero(as_tuple=False).flatten()
    expected = history.reshape(8, 256, 16, 16).index_select(0, valid_index)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected, atol=0.0, rtol=0.0)


def test_all_unavailable_history_never_calls_projection() -> None:
    model = make_ptf()
    history = torch.full((1, 2, 4, 256, 16, 16), math.nan)
    availability = torch.zeros(1, 2, 4, dtype=torch.bool)
    calls = 0

    def count_projection(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor],
    ) -> None:
        nonlocal calls
        calls += 1

    handle = model.history_projection.register_forward_pre_hook(count_projection)
    try:
        context = model.build_context(history, availability)
    finally:
        handle.remove()

    assert calls == 0
    assert context.shape == (1, 128, 4, 4)
    assert torch.isfinite(context).all()


@pytest.mark.parametrize("poison", [1000.0, math.nan], ids=["huge", "nan"])
def test_unavailable_payload_cannot_change_context(poison: float) -> None:
    model = make_ptf()
    first = make_history()
    second = first.clone()
    second[:, 1, 3] = poison
    availability = torch.ones(1, 2, 4, dtype=torch.bool)
    availability[:, 1, 3] = False

    torch.testing.assert_close(
        model.build_context(first, availability),
        model.build_context(second, availability),
        atol=0.0,
        rtol=0.0,
    )


def test_context_stem_input_preserves_slot_and_mask_order() -> None:
    model = make_ptf()
    history = torch.zeros(1, 2, 4, 256, 16, 16)
    availability = torch.tensor(
        [[[True, True, True, True], [True, True, False, True]]]
    )
    captured: list[torch.Tensor] = []

    with torch.no_grad():
        for parameter in model.history_projection.parameters():
            parameter.zero_()
        model.agent_embedding.weight[0].fill_(10.0)
        model.agent_embedding.weight[1].fill_(20.0)
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
            model.relative_time_embedding.weight[index].fill_(value)

    def capture_stem_input(
        _module: nn.Module,
        inputs: tuple[torch.Tensor],
    ) -> None:
        captured.append(inputs[0].detach().clone())

    handle = model.context_stem.register_forward_pre_hook(capture_stem_input)
    try:
        model.build_context(history, availability)
    finally:
        handle.remove()

    assert len(captured) == 1
    assert captured[0].shape == (1, 520, 4, 4)
    feature_slots = captured[0][:, :512].reshape(1, 8, 64, 4, 4)
    mask_planes = captured[0][:, 512:]
    expected_slot_values = (11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 0.0, 24.0)
    for index, value in enumerate(expected_slot_values):
        torch.testing.assert_close(
            feature_slots[:, index],
            torch.full_like(feature_slots[:, index], value),
            atol=0.0,
            rtol=0.0,
        )
    expected_masks = availability.reshape(1, 8, 1, 1).expand(1, 8, 4, 4)
    torch.testing.assert_close(
        mask_planes,
        expected_masks.to(dtype=mask_planes.dtype),
        atol=0.0,
        rtol=0.0,
    )


def test_projection_spatial_mismatch_raises_exact_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_ptf()
    monkeypatch.setattr(
        model,
        "projected_spatial_shape",
        lambda _shape: (5, 4),
    )

    with pytest.raises(
        RuntimeError,
        match="^PTF projection spatial shape mismatch$",
    ):
        model.build_context(
            make_history(),
            torch.ones(1, 2, 4, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    "history",
    [
        object(),
        torch.empty(2, 4, 256, 16, 16),
        torch.empty(0, 2, 4, 256, 16, 16),
        torch.empty(1, 1, 4, 256, 16, 16),
        torch.empty(1, 2, 3, 256, 16, 16),
        torch.empty(1, 2, 4, 128, 16, 16),
        torch.empty(1, 2, 4, 256, 0, 16),
        torch.empty(1, 2, 4, 256, 14, 16),
        torch.zeros(1, 2, 4, 256, 16, 16, dtype=torch.int64),
        torch.zeros(1, 2, 4, 256, 16, 16, dtype=torch.complex64),
    ],
    ids=[
        "not-tensor",
        "rank",
        "empty-batch",
        "agent-count",
        "history-count",
        "channel-count",
        "empty-spatial",
        "indivisible-spatial",
        "integer",
        "complex",
    ],
)
def test_build_context_rejects_invalid_history(history: object) -> None:
    with pytest.raises(ValueError):
        make_ptf().build_context(
            history,
            torch.ones(1, 2, 4, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    "availability",
    [
        object(),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(2, 2, 4, dtype=torch.bool),
        torch.ones(1, 2, 4),
        torch.ones(1, 2, 4, dtype=torch.int64),
    ],
    ids=["not-tensor", "shape", "batch", "floating", "integer"],
)
def test_build_context_rejects_invalid_availability(
    availability: object,
) -> None:
    with pytest.raises(ValueError):
        make_ptf().build_context(make_history(), availability)


@pytest.mark.parametrize("poison", [math.nan, math.inf], ids=["nan", "inf"])
def test_build_context_rejects_nonfinite_available_history(poison: float) -> None:
    history = make_history()
    history[:, 0, 0, 0, 0, 0] = poison

    with pytest.raises(ValueError, match="finite"):
        make_ptf().build_context(
            history,
            torch.ones(1, 2, 4, dtype=torch.bool),
        )


def test_build_context_rejects_device_mismatch() -> None:
    availability = torch.empty(
        (1, 2, 4),
        dtype=torch.bool,
        device="meta",
    )

    with pytest.raises(ValueError, match="device"):
        make_ptf().build_context(make_history(), availability)


def test_unavailable_history_has_zero_gradient() -> None:
    model = make_ptf()
    history = make_history().requires_grad_()
    availability = torch.tensor(
        [[[True, False, False, False], [False, False, False, False]]]
    )

    context = model.build_context(history, availability)
    context.square().sum().backward()

    assert history.grad is not None
    valid_gradient = history.grad[:, 0, 0]
    unavailable_gradient = history.grad.reshape(8, 256, 16, 16)[1:]
    assert torch.isfinite(valid_gradient).all()
    assert torch.count_nonzero(valid_gradient).item() > 0
    assert torch.count_nonzero(unavailable_gradient).item() == 0


@pytest.mark.parametrize(
    "context",
    [
        object(),
        torch.empty(128, 4, 4),
        torch.empty(0, 128, 4, 4),
        torch.empty(1, 127, 4, 4),
        torch.empty(1, 128, 0, 4),
        torch.zeros(1, 128, 4, 4, dtype=torch.int64),
        torch.zeros(1, 128, 4, 4, dtype=torch.complex64),
        torch.full((1, 128, 4, 4), math.nan),
        torch.full((1, 128, 4, 4), math.inf),
    ],
    ids=[
        "not-tensor",
        "rank",
        "empty-batch",
        "channels",
        "empty-spatial",
        "integer",
        "complex",
        "nan",
        "inf",
    ],
)
def test_query_rejects_invalid_context(context: object) -> None:
    with pytest.raises(ValueError):
        make_ptf().query(context, torch.tensor([0]), torch.tensor([1]))


@pytest.mark.parametrize(
    ("query_agent_index", "horizon"),
    [
        (object(), torch.tensor([1])),
        (torch.tensor([0.0]), torch.tensor([1])),
        (torch.tensor([True]), torch.tensor([1])),
        (torch.tensor([[0]]), torch.tensor([1])),
        (torch.tensor([0, 1]), torch.tensor([1])),
        (torch.tensor([-1]), torch.tensor([1])),
        (torch.tensor([2]), torch.tensor([1])),
        (torch.tensor([0]), object()),
        (torch.tensor([0]), torch.tensor([1.0])),
        (torch.tensor([0]), torch.tensor([True])),
        (torch.tensor([0]), torch.tensor([[1]])),
        (torch.tensor([0]), torch.tensor([1, 2])),
        (torch.tensor([0]), torch.tensor([-1])),
        (torch.tensor([0]), torch.tensor([4])),
    ],
    ids=[
        "agent-not-tensor",
        "agent-floating",
        "agent-boolean",
        "agent-rank",
        "agent-batch",
        "agent-negative",
        "agent-too-large",
        "horizon-not-tensor",
        "horizon-floating",
        "horizon-boolean",
        "horizon-rank",
        "horizon-batch",
        "horizon-negative",
        "horizon-too-large",
    ],
)
def test_query_rejects_invalid_indices(
    query_agent_index: object,
    horizon: object,
) -> None:
    with pytest.raises(ValueError):
        make_ptf().query(make_context(), query_agent_index, horizon)


@pytest.mark.parametrize("field", ["agent", "horizon"])
def test_query_rejects_device_mismatch(field: str) -> None:
    agent = torch.tensor([0])
    horizon = torch.tensor([1])
    if field == "agent":
        agent = torch.empty(1, dtype=torch.int64, device="meta")
    else:
        horizon = torch.empty(1, dtype=torch.int64, device="meta")

    with pytest.raises(ValueError, match="device"):
        make_ptf().query(make_context(), agent, horizon)


@pytest.mark.parametrize("invalid_horizon", [-1, 4])
def test_invalid_horizon_is_rejected_before_film(invalid_horizon: int) -> None:
    model = make_ptf()
    calls = 0

    def count_film(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor],
    ) -> None:
        nonlocal calls
        calls += 1

    handle = model.film_mlp.register_forward_pre_hook(count_film)
    try:
        with pytest.raises(ValueError):
            model.query(
                make_context(),
                torch.tensor([0]),
                torch.tensor([invalid_horizon]),
            )
    finally:
        handle.remove()

    assert calls == 0


def test_film_output_is_packed_block_major_scale_then_bias() -> None:
    model = make_ptf()
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    handles = []
    with torch.no_grad():
        model.film_mlp[-1].weight.zero_()
        for block_index in range(3):
            start = block_index * 256
            model.film_mlp[-1].bias[start : start + 128].fill_(
                float(block_index + 1)
            )
            model.film_mlp[-1].bias[start + 128 : start + 256].fill_(
                float(block_index + 11)
            )

    def capture_modulation(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        captured.append((inputs[1].detach().clone(), inputs[2].detach().clone()))

    for block in model.residual_blocks:
        handles.append(block.register_forward_pre_hook(capture_modulation))
    try:
        model.query(make_context(), torch.tensor([0]), torch.tensor([2]))
    finally:
        for handle in handles:
            handle.remove()

    assert len(captured) == 3
    for block_index, (scale, bias) in enumerate(captured):
        torch.testing.assert_close(
            scale,
            torch.full_like(scale, float(block_index + 1)),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            bias,
            torch.full_like(bias, float(block_index + 11)),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize("mode", ["nonlinear", "linear"])
def test_horizon_zero_forces_zero_after_nonzero_head(mode: str) -> None:
    model = make_ptf(mode)
    set_nonzero_final_heads(model)

    output = model.query(
        make_context(batch=2),
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
    )

    assert torch.count_nonzero(output.displacement).item() == 0


def test_post_upsample_displacement_scale_and_confidence_semantics() -> None:
    model = make_ptf()
    with torch.no_grad():
        model.displacement_head[-1].weight.zero_()
        model.displacement_head[-1].bias.fill_(math.atanh(0.25))
        model.confidence_head[-1].weight.zero_()
        model.confidence_head[-1].bias.fill_(math.log(1.0 / 3.0))

    output = model.query(
        make_context(size=4),
        torch.tensor([0]),
        torch.tensor([1]),
    )

    torch.testing.assert_close(
        output.displacement,
        torch.full_like(output.displacement, 8.0),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.confidence,
        torch.full_like(output.confidence, 0.25),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("mode", ["nonlinear", "linear"])
def test_output_values_stay_within_approved_bounds(mode: str) -> None:
    model = make_ptf(mode)
    with torch.no_grad():
        model.displacement_head[-1].weight.fill_(100.0)
        model.displacement_head[-1].bias.fill_(100.0)
        model.confidence_head[-1].weight.fill_(100.0)
        model.confidence_head[-1].bias.fill_(100.0)

    output = model.query(
        make_context(batch=2),
        torch.tensor([0, 1]),
        torch.tensor([3, 3]),
    )

    assert torch.isfinite(output.displacement).all()
    assert output.displacement.min().item() >= -32.0
    assert output.displacement.max().item() <= 32.0
    assert torch.isfinite(output.confidence).all()
    assert output.confidence.min().item() >= 0.0
    assert output.confidence.max().item() <= 1.0


def test_nonlinear_mode_is_not_forced_to_scale_linearly() -> None:
    model = make_ptf("nonlinear")
    set_nonzero_final_heads(model)
    context = make_context()

    first = model.query(context, torch.tensor([0]), torch.tensor([1]))
    second = model.query(context, torch.tensor([0]), torch.tensor([2]))

    assert first.displacement.abs().max().item() > 1e-3
    assert not torch.allclose(
        second.displacement,
        2.0 * first.displacement,
        atol=1e-5,
        rtol=1e-5,
    )


def test_linear_mode_scales_displacement_and_preserves_confidence() -> None:
    model = make_ptf("linear")
    set_nonzero_final_heads(model)
    context = make_context()

    first = model.query(context, torch.tensor([1]), torch.tensor([1]))
    second = model.query(context, torch.tensor([1]), torch.tensor([2]))
    third = model.query(context, torch.tensor([1]), torch.tensor([3]))

    assert first.displacement.abs().max().item() > 1e-3
    torch.testing.assert_close(
        second.displacement,
        2.0 * first.displacement,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        third.displacement,
        3.0 * first.displacement,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        second.confidence,
        first.confidence,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        third.confidence,
        first.confidence,
        atol=0.0,
        rtol=0.0,
    )


def test_other_batch_horizon_cannot_change_sample_zero_output() -> None:
    torch.manual_seed(7)
    model = make_ptf("nonlinear")
    with torch.no_grad():
        model.film_mlp[0].weight.zero_()
        model.film_mlp[0].bias.zero_()
        model.film_mlp[0].weight[0, -1] = 1.0
        model.film_mlp[-1].weight.zero_()
        model.film_mlp[-1].bias.zero_()
        model.film_mlp[-1].weight[:, 0] = 0.1
        set_nonzero_final_heads(model)
    context = make_context(batch=2)
    agents = torch.tensor([0, 1])

    first = model.query(context, agents, torch.tensor([1, 1]))
    second = model.query(context, agents, torch.tensor([1, 3]))

    torch.testing.assert_close(
        first.displacement[0],
        second.displacement[0],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        first.confidence[0],
        second.confidence[0],
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.allclose(
        first.displacement[1],
        second.displacement[1],
        atol=1e-5,
        rtol=1e-5,
    )
