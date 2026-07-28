from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from typing import Callable

import pytest
import torch
from torch import nn

import transvision.models.resilient_v2x.causal_repair as causal_repair_module
from transvision.models.resilient_v2x.causal_repair import CausalBranchRepair
from transvision.models.resilient_v2x.contracts import (
    Agent,
    BranchSelection,
    Modality,
    ProtocolInvariantError,
    SourceCandidate,
    UnsupportedReason,
)
from transvision.models.resilient_v2x.geometry import (
    BEVGridSpec,
    align_bev_to_target,
    warp_with_displacement,
)
from transvision.models.resilient_v2x.ptf import (
    HorizonConditionedPTF,
    PTFOutput,
)


SMALL_SPEC = BEVGridSpec(
    x_min=0.0,
    y_min=-2.56,
    resolution=0.32,
    height=16,
    width=16,
)


def make_repair() -> CausalBranchRepair:
    return CausalBranchRepair(
        grid_spec=SMALL_SPEC,
        channels=256,
        alpha=0.9,
        history_limit=3,
    )


def make_source(horizon: int) -> SourceCandidate:
    tick = 10 - horizon
    return SourceCandidate(
        packet_id=f"sequence:{tick}",
        n_s=tick,
        tau_s_ms=tick * 100,
        arrival_tau_ms=tick * 100 + 100,
        payload_valid=True,
        timestamp_valid=True,
        pose_valid=True,
        calibration_valid=True,
        faulted=False,
    )


def supported(
    agent: Agent,
    modality: Modality,
    horizon: int,
    *,
    endpoint_tick: int | None = None,
) -> BranchSelection:
    source = make_source(horizon)
    if endpoint_tick is None:
        endpoint_tick = 10
    observed = source.n_s == endpoint_tick
    return BranchSelection(
        agent=agent,
        modality=modality,
        supported=True,
        source=source,
        horizon=horizon,
        endpoint_tick=endpoint_tick,
        observed=observed,
        propagated=not observed,
        rejected=(),
        reason=None,
    )


def unsupported(
    agent: Agent,
    modality: Modality,
) -> BranchSelection:
    reason = (
        UnsupportedReason.EMPTY_MODALITY_HISTORY
        if agent is Agent.EGO
        else UnsupportedReason.EMPTY_ARRIVAL_SET
    )
    return BranchSelection.unsupported(agent, modality, reason, ())


def base_inputs(
    selections: tuple[BranchSelection, ...],
    *,
    requires_grad: bool = False,
) -> dict[str, object]:
    batch = len(selections)
    feature = torch.arange(
        batch * 256 * 16 * 16,
        dtype=torch.float32,
    ).reshape(batch, 256, 16, 16)
    feature = feature.div(feature.numel()).requires_grad_(requires_grad)
    context = torch.linspace(
        -0.2,
        0.2,
        batch * 4 * 16 * 16,
    ).reshape(batch, 4, 16, 16)
    context = context.requires_grad_(requires_grad)
    return {
        "selected_feature": feature,
        "source_to_target": torch.eye(4).repeat(batch, 1, 1),
        "ptf_context": context,
        "query_agent_index": torch.tensor(
            [0 if selection.agent is Agent.EGO else 1 for selection in selections],
            dtype=torch.int64,
        ),
        "selections": selections,
    }


class SpyPTF(nn.Module):
    def __init__(
        self,
        displacement_value: float = 0.0,
        confidence_value: float = 0.5,
    ) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(1.0))
        self.displacement_value = displacement_value
        self.confidence_value = confidence_value
        self.query_count = 0
        self.contexts: list[torch.Tensor] = []
        self.agent_indices: list[torch.Tensor] = []
        self.horizons: list[torch.Tensor] = []

    def query(
        self,
        context: torch.Tensor,
        query_agent_index: torch.Tensor,
        horizon: torch.Tensor,
    ) -> PTFOutput:
        self.query_count += 1
        self.contexts.append(context.detach().clone())
        self.agent_indices.append(query_agent_index.detach().clone())
        self.horizons.append(horizon.detach().clone())
        batch = context.shape[0]
        return PTFOutput(
            displacement=context.new_full(
                (batch, 2, SMALL_SPEC.height, SMALL_SPEC.width),
                self.displacement_value,
            ),
            confidence=context.new_full(
                (batch, 1, SMALL_SPEC.height, SMALL_SPEC.width),
                self.confidence_value,
            ),
        )


class PoisonPTF:
    def __init__(self) -> None:
        self.query_count = 0

    def query(self, *_args: object, **_kwargs: object) -> PTFOutput:
        self.query_count += 1
        raise AssertionError("unsupported branches must not query PTF")


class ReturningPTF:
    def __init__(self, factory: Callable[[torch.Tensor], object]) -> None:
        self.factory = factory
        self.query_count = 0

    def query(
        self,
        context: torch.Tensor,
        _query_agent_index: torch.Tensor,
        _horizon: torch.Tensor,
    ) -> object:
        self.query_count += 1
        return self.factory(context)


class GradientPTF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.05))

    def query(
        self,
        context: torch.Tensor,
        _query_agent_index: torch.Tensor,
        _horizon: torch.Tensor,
    ) -> PTFOutput:
        return PTFOutput(
            displacement=self.scale * context[:, :2],
            confidence=torch.sigmoid(context[:, 2:3]),
        )


def run_repair(
    repair: CausalBranchRepair,
    ptf: object,
    inputs: dict[str, object],
):
    return repair(ptf=ptf, **inputs)


def test_production_repair_shape_contract_without_feature_allocation() -> None:
    production = CausalBranchRepair(
        grid_spec=BEVGridSpec(
            x_min=0.0,
            y_min=-46.08,
            resolution=0.32,
            height=288,
            width=288,
        ),
        channels=256,
        alpha=0.9,
        history_limit=3,
    )

    assert production.expected_output_shape(batch_size=2) == (2, 256, 288, 288)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.0, "2"])
def test_expected_output_shape_rejects_nonpositive_or_noninteger_batch(
    batch_size: object,
) -> None:
    with pytest.raises(ProtocolInvariantError, match="batch_size"):
        make_repair().expected_output_shape(batch_size)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("channels", 255),
        ("channels", True),
        ("channels", 256.0),
        ("alpha", 0.8),
        ("alpha", True),
        ("alpha", math.nan),
        ("alpha", math.inf),
        ("history_limit", 2),
        ("history_limit", True),
        ("history_limit", 3.0),
    ],
)
def test_constructor_accepts_only_fixed_protocol_values(
    field: str,
    value: object,
) -> None:
    values = {
        "grid_spec": SMALL_SPEC,
        "channels": 256,
        "alpha": 0.9,
        "history_limit": 3,
    }
    values[field] = value

    with pytest.raises(ProtocolInvariantError):
        CausalBranchRepair(**values)


def test_observed_and_propagated_arithmetic_is_exact() -> None:
    selections = (
        supported(Agent.EGO, Modality.LIDAR, 0),
        supported(Agent.RSU, Modality.LIDAR, 3),
    )
    inputs = base_inputs(selections)
    source_to_target = inputs["source_to_target"]
    assert isinstance(source_to_target, torch.Tensor)
    source_to_target[1, 0, 3] = SMALL_SPEC.resolution
    spy = SpyPTF(displacement_value=-0.25, confidence_value=0.8)

    output = run_repair(make_repair(), spy, inputs)

    selected = inputs["selected_feature"]
    assert isinstance(selected, torch.Tensor)
    aligned = align_bev_to_target(selected, source_to_target, SMALL_SPEC)
    displacement = torch.full((2, 2, 16, 16), -0.25)
    aligned_warped = warp_with_displacement(aligned, displacement)
    expected = aligned_warped * torch.tensor([1.0, 0.9**3]).view(2, 1, 1, 1)
    assert output.feature.shape == (2, 256, 16, 16)
    torch.testing.assert_close(output.feature, expected)
    torch.testing.assert_close(output.feature[0], aligned_warped[0])
    assert output.diagnostics[0].gamma == 1.0
    assert output.diagnostics[0].reliability == 1.0
    assert output.diagnostics[1].gamma == pytest.approx(0.9**3)
    assert output.confidence is not None
    assert output.diagnostics[1].reliability == pytest.approx(
        (0.9**3) * output.confidence[1].mean().item()
    )
    torch.testing.assert_close(
        output.age_intervals,
        torch.tensor([0.0, 3.0]),
    )
    assert output.support.tolist() == [True, True]
    assert output.observed.tolist() == [True, False]
    assert output.propagated.tolist() == [False, True]
    assert spy.query_count == 1


def test_mixed_horizons_preserve_order_and_use_one_vectorized_query() -> None:
    selections = (
        supported(Agent.EGO, Modality.CAMERA, 0),
        supported(Agent.EGO, Modality.CAMERA, 1),
        supported(Agent.EGO, Modality.CAMERA, 2),
        supported(Agent.RSU, Modality.CAMERA, 3),
        unsupported(Agent.RSU, Modality.CAMERA),
    )
    inputs = base_inputs(selections)
    context = inputs["ptf_context"]
    assert isinstance(context, torch.Tensor)
    spy = SpyPTF(displacement_value=0.0, confidence_value=0.75)

    output = run_repair(make_repair(), spy, inputs)

    assert spy.query_count == 1
    torch.testing.assert_close(
        spy.contexts[0],
        context.index_select(0, torch.tensor([0, 1, 2, 3])),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        spy.agent_indices[0],
        torch.tensor([0, 0, 0, 1]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        spy.horizons[0],
        torch.tensor([0, 1, 2, 3]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        output.age_intervals,
        torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0]),
    )
    assert output.support.tolist() == [True, True, True, True, False]
    assert torch.count_nonzero(output.feature[4]).item() == 0
    assert output.displacement is not None
    assert output.confidence is not None
    assert torch.count_nonzero(output.displacement[4]).item() == 0
    assert torch.count_nonzero(output.confidence[4]).item() == 0
    assert output.diagnostics[4].displacement is None
    assert output.diagnostics[4].confidence is None


def test_rsu_endpoint_is_observed_but_still_uses_confidence() -> None:
    output = run_repair(
        make_repair(),
        SpyPTF(confidence_value=0.4),
        base_inputs((supported(Agent.RSU, Modality.LIDAR, 0),)),
    )

    assert output.observed.tolist() == [True]
    assert output.propagated.tolist() == [False]
    torch.testing.assert_close(output.reliability, torch.tensor([0.4]))
    assert output.diagnostics[0].observed is True
    assert output.diagnostics[0].propagated is False


def test_all_unsupported_short_circuits_every_numeric_poison() -> None:
    selections = (
        unsupported(Agent.EGO, Modality.LIDAR),
        unsupported(Agent.RSU, Modality.LIDAR),
    )
    inputs = base_inputs(selections)
    feature = inputs["selected_feature"]
    transform = inputs["source_to_target"]
    context = inputs["ptf_context"]
    agent_index = inputs["query_agent_index"]
    assert isinstance(feature, torch.Tensor)
    assert isinstance(transform, torch.Tensor)
    assert isinstance(context, torch.Tensor)
    assert isinstance(agent_index, torch.Tensor)
    feature.fill_(math.nan)
    context.fill_(math.nan)
    transform.fill_(math.nan)
    transform[0].zero_()
    agent_index[:] = torch.tensor([99, -7])
    spy = PoisonPTF()

    output = run_repair(make_repair(), spy, inputs)

    assert spy.query_count == 0
    assert torch.count_nonzero(output.feature).item() == 0
    assert torch.count_nonzero(output.reliability).item() == 0
    assert torch.count_nonzero(output.age_intervals).item() == 0
    assert output.support.tolist() == [False, False]
    assert output.observed.tolist() == [False, False]
    assert output.propagated.tolist() == [False, False]
    assert output.displacement is None
    assert output.confidence is None
    for diagnostics in output.diagnostics:
        assert diagnostics.ptf_queried is False
        assert diagnostics.gamma is None
        assert diagnostics.reliability == 0.0
        assert diagnostics.displacement is None
        assert diagnostics.confidence is None


def test_mixed_batch_never_validates_or_computes_unsupported_poison_row() -> None:
    selections = (
        supported(Agent.RSU, Modality.LIDAR, 1),
        unsupported(Agent.EGO, Modality.LIDAR),
    )
    inputs = base_inputs(selections)
    feature = inputs["selected_feature"]
    transform = inputs["source_to_target"]
    context = inputs["ptf_context"]
    agent_index = inputs["query_agent_index"]
    assert isinstance(feature, torch.Tensor)
    assert isinstance(transform, torch.Tensor)
    assert isinstance(context, torch.Tensor)
    assert isinstance(agent_index, torch.Tensor)
    feature[1].fill_(math.nan)
    context[1].fill_(math.nan)
    transform[1].zero_()
    agent_index[1] = 99
    spy = SpyPTF(confidence_value=0.6)

    output = run_repair(make_repair(), spy, inputs)

    assert spy.query_count == 1
    assert spy.contexts[0].shape[0] == 1
    assert torch.isfinite(output.feature[0]).all()
    assert torch.count_nonzero(output.feature[1]).item() == 0
    assert output.displacement is not None
    assert output.confidence is not None
    assert torch.count_nonzero(output.displacement[1]).item() == 0
    assert torch.count_nonzero(output.confidence[1]).item() == 0
    assert output.diagnostics[1].displacement is None
    assert output.diagnostics[1].confidence is None


def test_unsupported_diagnostics_are_json_null_not_zero_predictions() -> None:
    selection = unsupported(Agent.RSU, Modality.CAMERA)
    output = run_repair(
        make_repair(),
        PoisonPTF(),
        base_inputs((selection,)),
    )

    encoded = json.loads(json.dumps(asdict(output.diagnostics[0])))

    assert encoded["source_tick"] is None
    assert encoded["source_tau_ms"] is None
    assert encoded["horizon"] is None
    assert encoded["gamma"] is None
    assert encoded["displacement"] is None
    assert encoded["confidence"] is None
    assert encoded["reason"] == UnsupportedReason.EMPTY_ARRIVAL_SET.value


def test_supported_diagnostics_copy_source_and_detach_per_row_predictions() -> None:
    selection = supported(Agent.RSU, Modality.LIDAR, 2)
    inputs = base_inputs((selection,), requires_grad=True)

    output = run_repair(make_repair(), GradientPTF(), inputs)
    diagnostics = output.diagnostics[0]

    assert diagnostics.source_tick == 8
    assert diagnostics.source_tau_ms == 800
    assert diagnostics.horizon == 2
    assert diagnostics.ptf_queried is True
    assert diagnostics.reason is None
    assert diagnostics.displacement is not None
    assert diagnostics.confidence is not None
    assert diagnostics.displacement.shape == (2, 16, 16)
    assert diagnostics.confidence.shape == (1, 16, 16)
    assert diagnostics.displacement.requires_grad is False
    assert diagnostics.confidence.requires_grad is False
    assert output.displacement is not None
    assert output.confidence is not None
    assert output.displacement.requires_grad
    assert output.confidence.requires_grad


def test_distinct_modalities_use_only_the_explicit_injected_ptf() -> None:
    repair = make_repair()
    lidar = SpyPTF(displacement_value=0.0, confidence_value=0.2)
    camera = SpyPTF(displacement_value=0.5, confidence_value=0.9)
    lidar_output = run_repair(
        repair,
        lidar,
        base_inputs((supported(Agent.RSU, Modality.LIDAR, 1),)),
    )
    camera_output = run_repair(
        repair,
        camera,
        base_inputs((supported(Agent.RSU, Modality.CAMERA, 1),)),
    )

    assert lidar.query_count == 1
    assert camera.query_count == 1
    assert lidar_output.displacement is not None
    assert camera_output.displacement is not None
    assert lidar_output.confidence is not None
    assert camera_output.confidence is not None
    assert not torch.equal(
        lidar_output.displacement,
        camera_output.displacement,
    )
    assert not torch.equal(lidar_output.confidence, camera_output.confidence)
    assert list(repair.parameters()) == []
    assert list(repair.children()) == []
    assert repair.state_dict() == {}
    assert all(module is not lidar for module in repair.modules())
    assert all(module is not camera for module in repair.modules())
    assert all(parameter is not lidar.marker for parameter in repair.parameters())
    assert all(parameter is not camera.marker for parameter in repair.parameters())


def test_mixed_modalities_are_rejected_before_ptf_query() -> None:
    selections = (
        supported(Agent.EGO, Modality.LIDAR, 0),
        supported(Agent.RSU, Modality.CAMERA, 1),
    )
    spy = SpyPTF()

    with pytest.raises(ProtocolInvariantError, match="modality"):
        run_repair(make_repair(), spy, base_inputs(selections))

    assert spy.query_count == 0


def test_real_task_four_ptf_interface_integrates_on_small_cpu_fixture() -> None:
    ptf = HorizonConditionedPTF(
        in_channels=256,
        history_positions=4,
        history_limit=3,
        projected_channels=64,
        context_channels=128,
        max_low_resolution_cells=8.0,
        mode="nonlinear",
    ).eval()
    selection = supported(Agent.RSU, Modality.CAMERA, 1)
    inputs = base_inputs((selection,))
    inputs["ptf_context"] = torch.randn(1, 128, 4, 4)

    output = run_repair(make_repair(), ptf, inputs)

    assert output.feature.shape == (1, 256, 16, 16)
    assert output.displacement is not None
    assert output.confidence is not None
    torch.testing.assert_close(
        output.displacement,
        torch.zeros_like(output.displacement),
    )
    torch.testing.assert_close(
        output.confidence,
        torch.full_like(output.confidence, 0.5),
    )
    torch.testing.assert_close(
        output.reliability,
        torch.tensor([0.9 * 0.5]),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_feature", object(), "selected_feature"),
        ("selected_feature", torch.zeros(1, 256, 16), "rank-4"),
        ("selected_feature", torch.zeros(0, 256, 16, 16), "batch"),
        ("selected_feature", torch.zeros(1, 255, 16, 16), "channel"),
        ("selected_feature", torch.zeros(1, 256, 8, 16), "spatial"),
        (
            "selected_feature",
            torch.zeros(1, 256, 16, 16, dtype=torch.int64),
            "floating",
        ),
        ("source_to_target", object(), "source_to_target"),
        ("source_to_target", torch.eye(4), r"\[N,4,4\]"),
        ("source_to_target", torch.zeros(1, 3, 4), r"\[N,4,4\]"),
        (
            "source_to_target",
            torch.eye(4, dtype=torch.float64).unsqueeze(0),
            "dtype",
        ),
        ("ptf_context", object(), "ptf_context"),
        ("ptf_context", torch.zeros(1, 4, 16), "rank-4"),
        (
            "ptf_context",
            torch.zeros(1, 4, 16, 16, dtype=torch.int64),
            "floating",
        ),
        ("query_agent_index", object(), "query_agent_index"),
        ("query_agent_index", torch.tensor([[0]]), r"\[N\]"),
        ("query_agent_index", torch.tensor([0.0]), "integer"),
        ("query_agent_index", torch.tensor([True]), "integer"),
    ],
)
def test_structural_tensor_contract_rejects_malformed_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    inputs = base_inputs((supported(Agent.EGO, Modality.LIDAR, 0),))
    inputs[field] = value

    with pytest.raises(ValueError, match=message):
        run_repair(make_repair(), SpyPTF(), inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_to_target", torch.eye(4).repeat(2, 1, 1)),
        ("ptf_context", torch.zeros(2, 4, 16, 16)),
        ("query_agent_index", torch.tensor([0, 0])),
        ("selections", ()),
    ],
)
def test_batch_and_selection_count_must_match(
    field: str,
    value: object,
) -> None:
    inputs = base_inputs((supported(Agent.EGO, Modality.LIDAR, 0),))
    inputs[field] = value
    expected_error = ProtocolInvariantError if field == "selections" else ValueError

    with pytest.raises(expected_error, match="batch|selection"):
        run_repair(make_repair(), SpyPTF(), inputs)


def test_tensor_devices_must_match_before_query() -> None:
    inputs = base_inputs((supported(Agent.EGO, Modality.LIDAR, 0),))
    inputs["query_agent_index"] = torch.empty(
        1,
        dtype=torch.int64,
        device="meta",
    )

    with pytest.raises(ValueError, match="device"):
        run_repair(make_repair(), SpyPTF(), inputs)


@pytest.mark.parametrize(
    "selections",
    [
        object(),
        ("not-a-selection",),
    ],
    ids=["not-sequence", "invalid-element"],
)
def test_selection_container_and_elements_are_validated(
    selections: object,
) -> None:
    inputs = base_inputs((supported(Agent.EGO, Modality.LIDAR, 0),))
    inputs["selections"] = selections

    with pytest.raises(ProtocolInvariantError, match="selection"):
        run_repair(make_repair(), SpyPTF(), inputs)


@pytest.mark.parametrize("horizon", [-1, 4, True, 1.5, math.nan, math.inf])
def test_supported_horizon_must_be_exact_integer_range(
    horizon: object,
) -> None:
    selection = supported(Agent.RSU, Modality.LIDAR, 1)
    object.__setattr__(selection, "horizon", horizon)

    with pytest.raises(ProtocolInvariantError, match="horizon"):
        run_repair(
            make_repair(),
            SpyPTF(),
            base_inputs((selection,)),
        )


@pytest.mark.parametrize(
    ("selection", "observed", "propagated"),
    [
        (supported(Agent.EGO, Modality.LIDAR, 0), False, True),
        (supported(Agent.EGO, Modality.LIDAR, 1), True, False),
        (supported(Agent.RSU, Modality.LIDAR, 0), False, True),
    ],
    ids=["ego-zero-propagated", "ego-history-observed", "rsu-zero-propagated"],
)
def test_observed_and_propagated_semantics_are_enforced(
    selection: BranchSelection,
    observed: bool,
    propagated: bool,
) -> None:
    object.__setattr__(selection, "observed", observed)
    object.__setattr__(selection, "propagated", propagated)
    with pytest.raises(ProtocolInvariantError, match="observed|propagated"):
        run_repair(
            make_repair(),
            SpyPTF(),
            base_inputs((selection,)),
        )


@pytest.mark.parametrize(
    ("selection", "agent_index"),
    [
        (supported(Agent.EGO, Modality.LIDAR, 0), 1),
        (supported(Agent.RSU, Modality.LIDAR, 0), 0),
    ],
)
def test_supported_agent_index_must_match_selection(
    selection: BranchSelection,
    agent_index: int,
) -> None:
    inputs = base_inputs((selection,))
    inputs["query_agent_index"] = torch.tensor([agent_index])

    with pytest.raises(ProtocolInvariantError, match="agent"):
        run_repair(make_repair(), SpyPTF(), inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", object()),
        ("n_s", True),
        ("n_s", 8.0),
        ("tau_s_ms", False),
        ("tau_s_ms", 800.0),
    ],
    ids=[
        "source-type",
        "tick-boolean",
        "tick-noninteger",
        "time-boolean",
        "time-noninteger",
    ],
)
def test_supported_source_contract_fails_before_ptf_query(
    field: str,
    value: object,
) -> None:
    selection = supported(Agent.RSU, Modality.LIDAR, 2)
    if field == "source":
        object.__setattr__(selection, "source", value)
    else:
        assert selection.source is not None
        object.__setattr__(selection.source, field, value)
    spy = SpyPTF()

    with pytest.raises(ProtocolInvariantError, match="source"):
        run_repair(make_repair(), spy, base_inputs((selection,)))

    assert spy.query_count == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("packet_id", 3),
        ("arrival_tau_ms", 900.5),
        ("arrival_tau_ms", True),
        ("arrival_tau_ms", None),
        ("arrival_tau_ms", 799),
        ("payload_valid", False),
        ("payload_valid", 1),
        ("timestamp_valid", False),
        ("timestamp_valid", 1),
        ("pose_valid", False),
        ("pose_valid", 1),
        ("calibration_valid", False),
        ("calibration_valid", 1),
        ("faulted", True),
        ("faulted", 0),
    ],
    ids=[
        "packet-id-type",
        "arrival-noninteger",
        "arrival-boolean",
        "rsu-arrival-null",
        "rsu-negative-delay",
        "payload-invalid",
        "payload-nonboolean",
        "timestamp-invalid",
        "timestamp-nonboolean",
        "pose-invalid",
        "pose-nonboolean",
        "calibration-invalid",
        "calibration-nonboolean",
        "faulted",
        "faulted-nonboolean",
    ],
)
def test_supported_source_local_invariants_fail_before_geometry_and_ptf(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = supported(Agent.RSU, Modality.LIDAR, 2)
    assert selection.source is not None
    object.__setattr__(selection.source, field, value)
    spy = SpyPTF()
    alignment_calls = 0
    real_alignment = causal_repair_module.align_bev_to_target

    def counted_alignment(
        source: torch.Tensor,
        source_to_target: torch.Tensor,
        spec: BEVGridSpec,
    ) -> torch.Tensor:
        nonlocal alignment_calls
        alignment_calls += 1
        return real_alignment(source, source_to_target, spec)

    monkeypatch.setattr(
        causal_repair_module,
        "align_bev_to_target",
        counted_alignment,
    )

    with pytest.raises(ProtocolInvariantError):
        run_repair(make_repair(), spy, base_inputs((selection,)))

    assert alignment_calls == 0
    assert spy.query_count == 0


def test_supported_source_preserves_task_three_compatible_local_boundaries() -> None:
    ego = supported(Agent.EGO, Modality.LIDAR, 0)
    rsu = supported(Agent.RSU, Modality.LIDAR, 2)
    assert ego.source is not None
    assert rsu.source is not None
    object.__setattr__(ego.source, "packet_id", "")
    object.__setattr__(ego.source, "arrival_tau_ms", None)
    object.__setattr__(
        rsu.source,
        "arrival_tau_ms",
        rsu.source.tau_s_ms,
    )
    spy = SpyPTF()

    output = run_repair(
        make_repair(),
        spy,
        base_inputs((ego, rsu)),
    )

    assert output.support.tolist() == [True, True]
    assert spy.query_count == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed", 0),
        ("propagated", 0),
        ("reason", "bad-reason"),
    ],
)
def test_unsupported_contract_requires_exact_flags_and_reason_type_before_query(
    field: str,
    value: object,
) -> None:
    selection = unsupported(Agent.RSU, Modality.LIDAR)
    object.__setattr__(selection, field, value)
    spy = SpyPTF()

    with pytest.raises(ProtocolInvariantError, match="unsupported"):
        run_repair(make_repair(), spy, base_inputs((selection,)))

    assert spy.query_count == 0


def test_supported_nonfinite_and_singular_inputs_fail_closed() -> None:
    selection = supported(Agent.RSU, Modality.LIDAR, 1)
    inputs = base_inputs((selection,))
    feature = inputs["selected_feature"]
    assert isinstance(feature, torch.Tensor)
    feature[0, 0, 0, 0] = math.nan
    with pytest.raises(RuntimeError, match="aligned|finite"):
        run_repair(make_repair(), SpyPTF(), inputs)

    inputs = base_inputs((selection,))
    transform = inputs["source_to_target"]
    assert isinstance(transform, torch.Tensor)
    transform[0, 3, 3] = 0.0
    with pytest.raises(ValueError, match="invertible"):
        run_repair(make_repair(), SpyPTF(), inputs)

    inputs = base_inputs((selection,))
    transform = inputs["source_to_target"]
    assert isinstance(transform, torch.Tensor)
    transform[0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        run_repair(make_repair(), SpyPTF(), inputs)


def test_supported_nonfinite_feature_fails_before_out_of_bounds_alignment_masks_it() -> None:
    selection = supported(Agent.EGO, Modality.LIDAR, 0)
    inputs = base_inputs((selection,))
    feature = inputs["selected_feature"]
    transform = inputs["source_to_target"]
    assert isinstance(feature, torch.Tensor)
    assert isinstance(transform, torch.Tensor)
    feature.fill_(math.nan)
    transform[0, 0, 3] = 1_000_000.0
    spy = SpyPTF()

    with pytest.raises(RuntimeError, match="feature.*finite"):
        run_repair(make_repair(), spy, inputs)

    assert spy.query_count == 0


def valid_ptf_output(context: torch.Tensor) -> PTFOutput:
    return PTFOutput(
        displacement=context.new_zeros(1, 2, 16, 16),
        confidence=context.new_full((1, 1, 16, 16), 0.5),
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda _context: (torch.zeros(1), torch.zeros(1)), "PTFOutput"),
        (
            lambda context: PTFOutput(
                object(),
                context.new_full((1, 1, 16, 16), 0.5),
            ),
            "displacement",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 16, 16),
                object(),
            ),
            "confidence",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(2, 16, 16),
                context.new_full((1, 1, 16, 16), 0.5),
            ),
            "displacement",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 16, 16),
                context.new_full((1, 16, 16), 0.5),
            ),
            "confidence",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 3, 16, 16),
                context.new_full((1, 1, 16, 16), 0.5),
            ),
            "displacement",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(2, 2, 16, 16),
                context.new_full((2, 1, 16, 16), 0.5),
            ),
            "batch",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 8, 16),
                context.new_full((1, 1, 8, 16), 0.5),
            ),
            "spatial",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 16, 16),
                context.new_full((1, 2, 16, 16), 0.5),
            ),
            "confidence",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 16, 16),
                context.new_full((2, 1, 16, 16), 0.5),
            ),
            "batch",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 16, 16),
                context.new_full((1, 1, 8, 16), 0.5),
            ),
            "spatial",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(
                    1,
                    2,
                    16,
                    16,
                    dtype=torch.int64,
                ),
                context.new_full((1, 1, 16, 16), 0.5),
            ),
            "floating",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 16, 16),
                context.new_zeros(
                    1,
                    1,
                    16,
                    16,
                    dtype=torch.int64,
                ),
            ),
            "floating",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 16, 16, dtype=torch.float64),
                context.new_full((1, 1, 16, 16), 0.5),
            ),
            "dtype",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 16, 16),
                context.new_full(
                    (1, 1, 16, 16),
                    0.5,
                    dtype=torch.float64,
                ),
            ),
            "dtype",
        ),
        (
            lambda context: PTFOutput(
                torch.zeros(1, 2, 16, 16, device="meta"),
                context.new_full((1, 1, 16, 16), 0.5),
            ),
            "device",
        ),
        (
            lambda context: PTFOutput(
                context.new_zeros(1, 2, 16, 16),
                torch.zeros(1, 1, 16, 16, device="meta"),
            ),
            "device",
        ),
    ],
)
def test_ptf_output_type_shape_and_dtype_contract(
    factory: Callable[[torch.Tensor], object],
    message: str,
) -> None:
    selection = supported(Agent.RSU, Modality.LIDAR, 1)
    ptf = ReturningPTF(factory)

    with pytest.raises(RuntimeError, match=message):
        run_repair(make_repair(), ptf, base_inputs((selection,)))

    assert ptf.query_count == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("displacement", math.nan, "displacement"),
        ("displacement", math.inf, "displacement"),
        ("confidence", math.nan, "confidence"),
        ("confidence", math.inf, "confidence"),
        ("confidence", -0.1, r"\[0,1\]"),
        ("confidence", 1.1, r"\[0,1\]"),
    ],
)
def test_ptf_output_finite_and_confidence_range_contract(
    field: str,
    value: float,
    message: str,
) -> None:
    def factory(context: torch.Tensor) -> PTFOutput:
        output = valid_ptf_output(context)
        tensor = getattr(output, field).clone()
        tensor[0, 0, 0, 0] = value
        return replace(output, **{field: tensor})

    with pytest.raises(RuntimeError, match=message):
        run_repair(
            make_repair(),
            ReturningPTF(factory),
            base_inputs((supported(Agent.RSU, Modality.LIDAR, 1),)),
        )


def test_nonfinite_warped_feature_is_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def poison_warp(
        source: torch.Tensor,
        _displacement: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full_like(source, math.nan)

    monkeypatch.setattr(
        causal_repair_module,
        "warp_with_displacement",
        poison_warp,
    )

    with pytest.raises(RuntimeError, match="warped|finite"):
        run_repair(
            make_repair(),
            SpyPTF(),
            base_inputs((supported(Agent.RSU, Modality.LIDAR, 1),)),
        )


def test_ptf_computation_errors_propagate_unchanged() -> None:
    class ExpectedFailure(RuntimeError):
        """Sentinel proving injected computation failures are not rewritten."""

    class RaisingPTF:
        def query(self, *_args: object, **_kwargs: object) -> PTFOutput:
            raise ExpectedFailure("ptf computation failed")

    with pytest.raises(ExpectedFailure, match="ptf computation failed"):
        run_repair(
            make_repair(),
            RaisingPTF(),
            base_inputs((supported(Agent.RSU, Modality.LIDAR, 1),)),
        )


def test_forward_does_not_mutate_inputs() -> None:
    selections = (
        supported(Agent.EGO, Modality.CAMERA, 0),
        unsupported(Agent.RSU, Modality.CAMERA),
    )
    inputs = base_inputs(selections)
    tensor_copies = {
        name: value.clone()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    run_repair(make_repair(), SpyPTF(), inputs)

    for name, before in tensor_copies.items():
        value = inputs[name]
        assert isinstance(value, torch.Tensor)
        torch.testing.assert_close(value, before, atol=0.0, rtol=0.0)
    assert inputs["selections"] == selections


def test_supported_gradients_are_finite_and_unsupported_rows_are_zero() -> None:
    selections = (
        supported(Agent.RSU, Modality.LIDAR, 2),
        unsupported(Agent.EGO, Modality.LIDAR),
    )
    inputs = base_inputs(selections, requires_grad=True)

    ptf = GradientPTF()
    output = run_repair(make_repair(), ptf, inputs)
    assert output.displacement is not None
    assert output.confidence is not None
    loss = (
        output.feature.sum()
        + output.reliability.sum()
        + output.displacement.sum()
        + output.confidence.sum()
    )
    loss.backward()

    feature = inputs["selected_feature"]
    context = inputs["ptf_context"]
    assert isinstance(feature, torch.Tensor)
    assert isinstance(context, torch.Tensor)
    assert feature.grad is not None
    assert context.grad is not None
    assert ptf.scale.grad is not None
    assert torch.isfinite(feature.grad[0]).all()
    assert torch.isfinite(context.grad[0]).all()
    assert torch.isfinite(ptf.scale.grad).all()
    assert torch.count_nonzero(feature.grad[0]).item() > 0
    assert torch.count_nonzero(context.grad[0]).item() > 0
    assert torch.count_nonzero(ptf.scale.grad).item() > 0
    assert torch.count_nonzero(feature.grad[1]).item() == 0
    assert torch.count_nonzero(context.grad[1]).item() == 0
    assert output.displacement.grad_fn is not None
    assert output.confidence.grad_fn is not None
