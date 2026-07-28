from __future__ import annotations

import math

import pytest
import torch

from transvision.models.resilient_v2x.causal_repair import (
    age_decay,
    arrived_candidates,
    branch_reliability,
    latest_arrived_rsu_age_intervals,
    select_causal_source,
)
from transvision.models.resilient_v2x.contracts import (
    Agent,
    BranchSelection,
    Modality,
    ProtocolInvariantError,
    RejectedCandidate,
    SourceCandidate,
    UnsupportedReason,
)


def candidate(
    n_s: int = 10,
    *,
    packet_id: str | None = None,
    delay_ms: int = 0,
    arrival_tau_ms: int | None = None,
    payload_valid: bool = True,
    timestamp_valid: bool = True,
    pose_valid: bool = True,
    calibration_valid: bool = True,
    faulted: bool = False,
) -> SourceCandidate:
    tau_s_ms = n_s * 100
    return SourceCandidate(
        packet_id=packet_id or f"seq:{n_s}",
        n_s=n_s,
        tau_s_ms=tau_s_ms,
        arrival_tau_ms=(
            tau_s_ms + delay_ms
            if arrival_tau_ms is None
            else arrival_tau_ms
        ),
        payload_valid=payload_valid,
        timestamp_valid=timestamp_valid,
        pose_valid=pose_valid,
        calibration_valid=calibration_valid,
        faulted=faulted,
    )


def history(delay_ms: int, fault_target: bool) -> list[SourceCandidate]:
    return [
        candidate(
            n,
            delay_ms=delay_ms,
            faulted=fault_target and n == 10,
        )
        for n in range(7, 11)
    ]


def select(
    agent: Agent,
    candidates: list[SourceCandidate] | tuple[SourceCandidate, ...],
    *,
    modality: Modality = Modality.LIDAR,
    n_t: int = 10,
    target_tau_ms: int = 1000,
    delta_t_ms: int = 100,
    history_limit: int = 3,
) -> BranchSelection:
    return select_causal_source(
        agent=agent,
        modality=modality,
        n_t=n_t,
        target_tau_ms=target_tau_ms,
        delta_t_ms=delta_t_ms,
        history_limit=history_limit,
        candidates=candidates,
    )


@pytest.mark.parametrize("agent", [Agent.EGO, Agent.RSU])
@pytest.mark.parametrize("delay_ms", [0, 100, 200, 300])
@pytest.mark.parametrize("fault_target", [False, True], ids=["full", "fault"])
def test_complete_agent_delay_fault_horizon_table(
    agent: Agent,
    delay_ms: int,
    fault_target: bool,
) -> None:
    result = select(agent, history(delay_ms, fault_target))
    expected_horizon = (
        int(fault_target)
        if agent is Agent.EGO
        else max(delay_ms // 100, int(fault_target))
    )
    expected_endpoint = 10 if agent is Agent.EGO else 10 - delay_ms // 100
    expected_observed = not fault_target if agent is Agent.EGO else not (
        delay_ms == 0 and fault_target
    )

    assert result.supported
    assert result.horizon == expected_horizon
    assert result.endpoint_tick == expected_endpoint
    assert result.observed is expected_observed
    assert result.propagated is not result.observed


def test_arrived_candidates_applies_agent_cutoff_and_sorts_newest_first() -> None:
    inputs = [
        candidate(9, arrival_tau_ms=2000),
        candidate(11, arrival_tau_ms=900),
        candidate(10, arrival_tau_ms=1000),
    ]

    ego = arrived_candidates(Agent.EGO, 1000, inputs)
    rsu = arrived_candidates(Agent.RSU, 1000, inputs)

    assert tuple(item.n_s for item in ego) == (10, 9)
    assert tuple(item.n_s for item in rsu) == (10,)


def test_arrived_candidates_rejects_negative_rsu_delay() -> None:
    with pytest.raises(ProtocolInvariantError, match="delay"):
        arrived_candidates(
            Agent.RSU,
            1000,
            [candidate(10, arrival_tau_ms=999)],
        )


def test_rsu_empty_arrived_set_has_specific_reason() -> None:
    null_arrival = vars(candidate(9)).copy()
    null_arrival["arrival_tau_ms"] = None
    result = select(
        Agent.RSU,
        [
            SourceCandidate(**null_arrival),
            candidate(10, arrival_tau_ms=1001),
        ],
    )

    assert result.reason is UnsupportedReason.EMPTY_ARRIVAL_SET
    assert [item.reason for item in result.rejected] == [
        UnsupportedReason.EMPTY_ARRIVAL_SET,
        UnsupportedReason.EMPTY_ARRIVAL_SET,
    ]


@pytest.mark.parametrize("agent", [Agent.EGO, Agent.RSU])
def test_all_candidates_faulted_has_empty_modality_history(agent: Agent) -> None:
    result = select(
        agent,
        [candidate(n, faulted=True) for n in range(7, 11)],
    )

    assert result.reason is UnsupportedReason.EMPTY_MODALITY_HISTORY


def test_empty_ego_history_has_empty_modality_history() -> None:
    result = select(Agent.EGO, [])

    assert result.reason is UnsupportedReason.EMPTY_MODALITY_HISTORY
    assert result.rejected == ()


@pytest.mark.parametrize(
    ("n_s", "supported", "reason"),
    [
        (7, True, None),
        (6, False, UnsupportedReason.UNSUPPORTED_HORIZON),
    ],
    ids=["h-equals-k", "h-equals-k-plus-one"],
)
def test_history_limit_boundary_is_not_clipped(
    n_s: int,
    supported: bool,
    reason: UnsupportedReason | None,
) -> None:
    result = select(Agent.EGO, [candidate(n_s)])

    assert result.supported is supported
    assert result.reason is reason
    if supported:
        assert result.horizon == 3


def test_ego_ignores_future_arrivals_but_rejects_future_source_ticks() -> None:
    result = select(
        Agent.EGO,
        [
            candidate(9, arrival_tau_ms=5000),
            candidate(11, arrival_tau_ms=900),
            candidate(10, arrival_tau_ms=5000),
        ],
    )

    assert result.source is not None
    assert result.source.n_s == 10
    assert result.observed
    assert result.rejected == (
        RejectedCandidate(
            packet_id="seq:11",
            n_s=11,
            reason=UnsupportedReason.INVALID_TIMESTAMP,
        ),
    )


def test_out_of_order_candidates_select_greatest_causal_valid_tick() -> None:
    result = select(
        Agent.EGO,
        [candidate(8), candidate(10), candidate(7), candidate(9)],
    )

    assert result.source is not None
    assert result.source.n_s == 10


def test_same_tick_packet_id_breaks_ties_deterministically() -> None:
    result = select(
        Agent.EGO,
        [
            candidate(10, packet_id="seq:10:a"),
            candidate(10, packet_id="seq:10:z"),
        ],
    )

    assert result.source is not None
    assert result.source.packet_id == "seq:10:z"


def test_duplicate_candidate_identity_fails_closed() -> None:
    duplicate = candidate(10, packet_id="duplicate")

    with pytest.raises(ProtocolInvariantError, match="duplicate"):
        select(Agent.EGO, [duplicate, duplicate])


def test_exact_arrival_cutoff_has_no_future_packet_interpolation() -> None:
    result = select(
        Agent.RSU,
        [
            candidate(10, arrival_tau_ms=1001),
            candidate(9, arrival_tau_ms=1000),
        ],
    )

    assert result.source is not None
    assert result.source.n_s == 9
    assert result.horizon == 1
    assert result.endpoint_tick == 9
    assert result.observed
    assert result.rejected[0].packet_id == "seq:10"
    assert result.rejected[0].reason is UnsupportedReason.EMPTY_ARRIVAL_SET


def test_newest_invalid_candidate_falls_back_to_older_valid_source() -> None:
    result = select(
        Agent.EGO,
        [
            candidate(9),
            candidate(10, pose_valid=False),
        ],
        modality=Modality.CAMERA,
    )

    assert result.source is not None
    assert result.source.n_s == 9
    assert result.endpoint_tick == 10
    assert result.propagated
    assert result.reason is None
    assert result.rejected[0] == RejectedCandidate(
        packet_id="seq:10",
        n_s=10,
        reason=UnsupportedReason.INVALID_POSE,
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"timestamp_valid": False}, UnsupportedReason.INVALID_TIMESTAMP),
        ({"payload_valid": False}, UnsupportedReason.MISSING_PAYLOAD),
        ({"pose_valid": False}, UnsupportedReason.INVALID_POSE),
        ({"calibration_valid": False}, UnsupportedReason.INVALID_CALIBRATION),
    ],
)
def test_each_metadata_failure_becomes_the_branch_reason(
    changes: dict[str, bool],
    reason: UnsupportedReason,
) -> None:
    result = select(Agent.EGO, [candidate(**changes)])

    assert result.reason is reason
    assert result.rejected[0].reason is reason


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {
                "timestamp_valid": False,
                "payload_valid": False,
                "pose_valid": False,
                "calibration_valid": False,
            },
            UnsupportedReason.INVALID_TIMESTAMP,
        ),
        (
            {
                "payload_valid": False,
                "pose_valid": False,
                "calibration_valid": False,
            },
            UnsupportedReason.MISSING_PAYLOAD,
        ),
        (
            {"pose_valid": False, "calibration_valid": False},
            UnsupportedReason.INVALID_POSE,
        ),
    ],
)
def test_metadata_reason_uses_first_failed_check_within_candidate(
    changes: dict[str, bool],
    reason: UnsupportedReason,
) -> None:
    result = select(Agent.EGO, [candidate(**changes)])

    assert result.reason is reason


def test_newest_metadata_failure_wins_across_candidates() -> None:
    result = select(
        Agent.EGO,
        [
            candidate(9, pose_valid=False),
            candidate(10, calibration_valid=False),
        ],
    )

    assert result.reason is UnsupportedReason.INVALID_CALIBRATION


def test_metadata_reason_precedes_unsupported_older_horizon() -> None:
    result = select(
        Agent.EGO,
        [
            candidate(6),
            candidate(10, payload_valid=False),
        ],
    )

    assert result.reason is UnsupportedReason.MISSING_PAYLOAD


def test_newest_metadata_valid_beyond_horizon_preserves_horizon_reason() -> None:
    result = select(
        Agent.EGO,
        [
            candidate(5, timestamp_valid=False),
            candidate(6),
        ],
    )

    assert result.reason is UnsupportedReason.UNSUPPORTED_HORIZON


def test_rejections_are_permutation_stable_and_newest_first() -> None:
    values = [
        candidate(11),
        candidate(10, arrival_tau_ms=1001),
        candidate(9, faulted=True),
        candidate(8, timestamp_valid=False),
        candidate(7),
        candidate(6),
    ]
    expected = (
        RejectedCandidate(
            "seq:11",
            11,
            UnsupportedReason.INVALID_TIMESTAMP,
        ),
        RejectedCandidate(
            "seq:10",
            10,
            UnsupportedReason.EMPTY_ARRIVAL_SET,
        ),
        RejectedCandidate(
            "seq:9",
            9,
            UnsupportedReason.EMPTY_MODALITY_HISTORY,
        ),
        RejectedCandidate(
            "seq:8",
            8,
            UnsupportedReason.INVALID_TIMESTAMP,
        ),
        RejectedCandidate(
            "seq:6",
            6,
            UnsupportedReason.UNSUPPORTED_HORIZON,
        ),
    )

    for ordered in (values, list(reversed(values)), [values[i] for i in (3, 0, 5, 2, 4, 1)]):
        result = select(Agent.RSU, ordered)
        assert result.source is not None
        assert result.source.n_s == 7
        assert result.rejected == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_t", True),
        ("n_t", 10.0),
        ("target_tau_ms", False),
        ("target_tau_ms", 1000.0),
        ("delta_t_ms", True),
        ("delta_t_ms", 100.0),
        ("delta_t_ms", 0),
        ("delta_t_ms", -100),
        ("history_limit", False),
        ("history_limit", 3.0),
        ("history_limit", -1),
    ],
)
def test_protocol_scalars_fail_closed(field: str, value: object) -> None:
    inputs = {
        "agent": Agent.EGO,
        "modality": Modality.LIDAR,
        "n_t": 10,
        "target_tau_ms": 1000,
        "delta_t_ms": 100,
        "history_limit": 3,
        "candidates": [candidate()],
    }
    inputs[field] = value

    with pytest.raises(ProtocolInvariantError):
        select_causal_source(**inputs)


def test_target_timestamp_must_match_target_tick() -> None:
    with pytest.raises(ProtocolInvariantError, match="target"):
        select(
            Agent.EGO,
            [candidate()],
            target_tau_ms=1001,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_s", True),
        ("n_s", 10.0),
        ("tau_s_ms", False),
        ("tau_s_ms", 1000.0),
        ("arrival_tau_ms", True),
        ("arrival_tau_ms", 1000.0),
    ],
)
def test_candidate_protocol_scalars_fail_closed(field: str, value: object) -> None:
    values = vars(candidate()).copy()
    values[field] = value

    with pytest.raises(ProtocolInvariantError):
        select(Agent.RSU, [SourceCandidate(**values)])


@pytest.mark.parametrize(
    "tau_s_ms",
    [800, 950],
    ids=["tick-inconsistent", "non-integral-horizon"],
)
def test_candidate_timestamp_and_horizon_must_be_exact(tau_s_ms: int) -> None:
    values = vars(candidate(9)).copy()
    values["tau_s_ms"] = tau_s_ms

    with pytest.raises(ProtocolInvariantError):
        select(Agent.EGO, [SourceCandidate(**values)])


def test_arrived_rsu_packet_cannot_have_negative_realized_delay() -> None:
    with pytest.raises(ProtocolInvariantError, match="delay"):
        select(
            Agent.RSU,
            [candidate(10, arrival_tau_ms=999)],
        )


def test_age_decay_matches_exact_decay_vector_and_preserves_autograd() -> None:
    horizon = torch.tensor(
        [0.0, 1.0, 2.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = age_decay(horizon, 0.9)
    result.sum().backward()

    torch.testing.assert_close(
        result,
        torch.tensor([1.0, 0.9, 0.81, 0.729], dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert result.dtype is horizon.dtype
    assert result.device == horizon.device
    assert horizon.grad is not None
    assert torch.isfinite(horizon.grad).all()


def test_age_decay_converts_integer_horizons_to_floating() -> None:
    result = age_decay(torch.tensor([0, 1, 2, 3]), 0.9)

    assert result.is_floating_point()
    torch.testing.assert_close(
        result,
        torch.tensor([1.0, 0.9, 0.81, 0.729]),
    )


@pytest.mark.parametrize(
    "horizon",
    [
        torch.tensor([math.nan]),
        torch.tensor([math.inf]),
        torch.tensor([-1.0]),
        torch.tensor([0.5]),
        torch.tensor([-1]),
    ],
)
def test_age_decay_rejects_invalid_horizons(horizon: torch.Tensor) -> None:
    with pytest.raises(ProtocolInvariantError, match="horizon"):
        age_decay(horizon, 0.9)


@pytest.mark.parametrize("alpha", [math.nan, math.inf, 0.0, -0.1, 1.1, True])
def test_age_decay_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ProtocolInvariantError, match="alpha"):
        age_decay(torch.tensor([0.0]), alpha)


def supported_selection(
    agent: Agent,
    horizon: int,
    *,
    modality: Modality = Modality.LIDAR,
    delay_ms: int = 0,
    endpoint_tick: int | None = None,
) -> BranchSelection:
    source = candidate(10 - horizon, delay_ms=delay_ms)
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


def test_unsupported_branch_short_circuits_poison_confidence() -> None:
    selection = BranchSelection.unsupported(
        Agent.RSU,
        Modality.CAMERA,
        UnsupportedReason.EMPTY_ARRIVAL_SET,
        (),
    )
    poison = torch.tensor([[[[math.nan]]]], dtype=torch.float64)

    result = branch_reliability(selection, poison, 0.9, 0)

    assert result.shape == ()
    assert result.dtype is poison.dtype
    assert result.item() == 0.0


@pytest.mark.parametrize(
    "confidence",
    [None, torch.tensor([[[[math.nan]]]], dtype=torch.float64)],
)
def test_observed_current_ego_is_one_without_reading_confidence(
    confidence: torch.Tensor | None,
) -> None:
    result = branch_reliability(
        supported_selection(Agent.EGO, 0),
        confidence,
        0.9,
        0,
    )

    assert result.shape == ()
    assert result.item() == 1.0
    if confidence is not None:
        assert result.dtype is confidence.dtype


def test_propagated_reliability_uses_selected_batch_mean_and_decay() -> None:
    confidence = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[0.2, 0.4], [0.6, 0.8]]],
        ],
        dtype=torch.float64,
    )

    result = branch_reliability(
        supported_selection(Agent.RSU, 2),
        confidence,
        0.9,
        1,
    )

    assert result.shape == ()
    assert result.dtype is confidence.dtype
    assert result.item() == pytest.approx(0.405)


def test_propagated_reliability_preserves_confidence_autograd() -> None:
    confidence = torch.full(
        (2, 1, 2, 2),
        0.5,
        dtype=torch.float64,
        requires_grad=True,
    )

    result = branch_reliability(
        supported_selection(Agent.RSU, 1),
        confidence,
        0.9,
        1,
    )
    result.backward()

    assert confidence.grad is not None
    torch.testing.assert_close(confidence.grad[0], torch.zeros_like(confidence.grad[0]))
    torch.testing.assert_close(
        confidence.grad[1],
        torch.full_like(confidence.grad[1], 0.9 / 4),
    )


def test_propagated_reliability_requires_confidence() -> None:
    with pytest.raises(ProtocolInvariantError, match="confidence"):
        branch_reliability(
            supported_selection(Agent.RSU, 1),
            None,
            0.9,
            0,
        )


@pytest.mark.parametrize(
    "confidence",
    [
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2, 2),
        torch.ones(0, 1, 2, 2),
        torch.ones(1, 1, 0, 2),
        torch.ones(1, 1, 2, 2, dtype=torch.int64),
    ],
    ids=["rank", "channel", "empty-batch", "empty-spatial", "integer"],
)
def test_propagated_reliability_rejects_invalid_confidence_shape_or_dtype(
    confidence: torch.Tensor,
) -> None:
    with pytest.raises(ProtocolInvariantError, match="confidence"):
        branch_reliability(
            supported_selection(Agent.RSU, 1),
            confidence,
            0.9,
            0,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1])
def test_propagated_reliability_rejects_invalid_confidence_values(
    value: float,
) -> None:
    confidence = torch.full((1, 1, 2, 2), 0.5)
    confidence[0, 0, 0, 0] = value

    with pytest.raises(ProtocolInvariantError, match="confidence"):
        branch_reliability(
            supported_selection(Agent.RSU, 1),
            confidence,
            0.9,
            0,
        )


@pytest.mark.parametrize("batch_index", [True, 0.0, -1, 2])
def test_propagated_reliability_rejects_invalid_batch_index(
    batch_index: int,
) -> None:
    with pytest.raises(ProtocolInvariantError, match="batch"):
        branch_reliability(
            supported_selection(Agent.RSU, 1),
            torch.ones(2, 1, 2, 2),
            0.9,
            batch_index,
        )


def test_rsu_age_uses_latest_arrived_packet_independent_of_sensor_fault() -> None:
    candidates = [
        candidate(10, arrival_tau_ms=1100, faulted=True),
        candidate(9, arrival_tau_ms=1000, faulted=True),
        candidate(8, arrival_tau_ms=950),
    ]

    result = latest_arrived_rsu_age_intervals(
        candidates,
        target_tau_ms=1000,
        history_limit=3,
        delta_t_ms=100,
    )

    assert result == 1.0


def test_rsu_age_is_zero_without_an_arrived_timestamp_valid_packet() -> None:
    result = latest_arrived_rsu_age_intervals(
        [
            candidate(10, arrival_tau_ms=1001),
            candidate(9, arrival_tau_ms=1000, timestamp_valid=False),
        ],
        target_tau_ms=1000,
        history_limit=3,
        delta_t_ms=100,
    )

    assert result == 0.0


def test_rsu_age_requires_same_age_for_latest_multimodal_packets() -> None:
    candidates = [
        candidate(9, packet_id="lidar:9", arrival_tau_ms=1000),
        candidate(9, packet_id="camera:9", arrival_tau_ms=950),
    ]

    assert latest_arrived_rsu_age_intervals(
        candidates,
        target_tau_ms=1000,
        history_limit=3,
        delta_t_ms=100,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("history_limit", "delta_t_ms"),
    [(0, 100), (-1, 100), (3, 0), (3, -100), (True, 100), (3, False)],
)
def test_rsu_delay_intervals_rejects_invalid_denominator_inputs(
    history_limit: int,
    delta_t_ms: int,
) -> None:
    with pytest.raises(ProtocolInvariantError):
        latest_arrived_rsu_age_intervals(
            [],
            target_tau_ms=1000,
            history_limit=history_limit,
            delta_t_ms=delta_t_ms,
        )


def test_rsu_age_rejects_arrival_before_source_time() -> None:
    with pytest.raises(ProtocolInvariantError, match="precede"):
        latest_arrived_rsu_age_intervals(
            [candidate(9, arrival_tau_ms=899)],
            target_tau_ms=1000,
            history_limit=3,
            delta_t_ms=100,
        )


def test_rsu_age_rejects_inconsistent_timestamp_marked_valid() -> None:
    values = vars(candidate(9, arrival_tau_ms=1000)).copy()
    values["tau_s_ms"] = 850
    malformed = SourceCandidate(**values)

    with pytest.raises(ProtocolInvariantError, match="timestamp"):
        latest_arrived_rsu_age_intervals(
            [malformed],
            target_tau_ms=1000,
            history_limit=3,
            delta_t_ms=100,
        )
