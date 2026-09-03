import numpy as np
import pytest

from transvision.models.event_track_v2x import (
    AppendOnlyCommitLog,
    AffineClockMap,
    BudgetBasis,
    CausalScoreEvidenceV1,
    CorrelatedBeliefRequiresCI,
    CorrelatedTrackBelief,
    FixedLagReplay,
    IndependentIncrement,
    IngestStatus,
    Lineage,
    LocalTimestamps,
    OutOfWindowPolicy,
    ReceivedMessage,
    ScheduleCandidate,
    WireMessage,
    associate_gaussians,
    chi_square_gate,
    chi_square_quantile,
    constant_velocity_dynamics,
    covariance_intersection,
    encode_message,
    gaussian_nll,
    select_exact_budget,
    solve_one_to_one,
    wire_digest,
)


IDENTITY_CLOCK = AffineClockMap(1.0, 0.0, np.zeros((2, 2)))


def message(
    message_id: str,
    *,
    sequence: int,
    event_time: float,
    measurement: float,
    information_cutoff: float | None = None,
    state_reference: float | None = None,
    factor_id: str | None = None,
    ancestor_message_ids: tuple[str, ...] = (),
    received_at: float = 2.0,
    target_id: str = "track",
    deadline: float = 3.0,
    ttl: float = 5.0,
) -> ReceivedMessage:
    wire = WireMessage(
        message_id=message_id,
        source="sender",
        sequence=sequence,
        timestamps=LocalTimestamps(
            event_time if information_cutoff is None else information_cutoff,
            event_time if state_reference is None else state_reference,
            event_time + 0.01,
            event_time + 0.02,
        ),
        deadline=deadline,
        coordinate_frame="world",
        ttl=ttl,
        payload=IndependentIncrement(
            target_id=target_id,
            measurement=np.array([measurement]),
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.2]]),
            lineage=Lineage(
                (factor_id or f"factor-{message_id}",),
                True,
                ancestor_message_ids,
            ),
        ),
    )
    return ReceivedMessage(wire, received_at=received_at)


def tracker(
    *, lag: float = 5.0, policy: OutOfWindowPolicy = OutOfWindowPolicy.DROP
) -> FixedLagReplay:
    return FixedLagReplay(
        track_id="track",
        initial_time=0.0,
        initial_mean=np.array([0.0, 1.0]),
        initial_covariance=np.diag([1.0, 0.5]),
        dynamics=constant_velocity_dynamics(
            1, acceleration_spectral_density=0.1
        ),
        lag=lag,
        out_of_window_policy=policy,
    )


def manual_predict(
    mean: np.ndarray, covariance: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    dynamics = constant_velocity_dynamics(1, acceleration_spectral_density=0.1)
    transition = dynamics.transition(delta)
    return (
        transition @ mean,
        transition @ covariance @ transition.T + dynamics.process_covariance(delta),
    )


def manual_update(
    mean: np.ndarray, covariance: np.ndarray, measurement: float
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array([[1.0, 0.0]])
    noise = np.array([[0.2]])
    innovation_covariance = matrix @ covariance @ matrix.T + noise
    gain = np.linalg.solve(innovation_covariance, matrix @ covariance).T
    updated_mean = mean + gain @ (np.array([measurement]) - matrix @ mean)
    residual = np.eye(2) - gain @ matrix
    updated_covariance = residual @ covariance @ residual.T + gain @ noise @ gain.T
    return updated_mean, updated_covariance


def test_gaussian_nll_chi_square_gate_and_quantile() -> None:
    covariance = np.diag([4.0, 1.0])
    innovation = np.array([2.0, 1.0])
    expected = 0.5 * 2.0 + 0.5 * np.log(4.0)
    assert gaussian_nll(innovation, covariance) == pytest.approx(expected)
    assert chi_square_quantile(2, 0.95) == pytest.approx(5.991464547, rel=1e-8)
    accepted = chi_square_gate(np.array([1.0, 1.0]), np.eye(2), probability=0.95)
    rejected = chi_square_gate(np.array([3.0, 3.0]), np.eye(2), probability=0.95)
    assert accepted.accepted
    assert not rejected.accepted


def test_one_to_one_assignment_has_real_unmatched_options() -> None:
    costs = np.array([[0.1, 20.0], [0.2, 30.0]])
    result = solve_one_to_one(
        costs, unmatched_left_cost=np.array([2.0, 2.0]), unmatched_right_cost=1.0
    )
    assert result.pairs == ((0, 0),)
    assert result.unmatched_left == (1,)
    assert result.unmatched_right == (1,)
    assert result.total_cost == pytest.approx(3.1)

    all_unmatched = solve_one_to_one(
        np.array([[100.0]]), unmatched_left_cost=1.0, unmatched_right_cost=1.0
    )
    assert all_unmatched.pairs == ()
    assert all_unmatched.total_cost == 2.0


def test_gaussian_association_gates_then_enforces_one_to_one() -> None:
    result = associate_gaussians(
        [np.array([0.0]), np.array([0.1])],
        [np.array([[0.1]]), np.array([[0.1]])],
        [np.array([0.05]), np.array([50.0])],
        [np.array([[0.1]]), np.array([[0.1]])],
        gate_probability=0.95,
        unmatched_left_cost=3.0,
        unmatched_right_cost=3.0,
        assume_independent=True,
    )
    assert len(result.assignment.pairs) == 1
    assert result.assignment.unmatched_left in {(0,), (1,)}
    assert result.assignment.unmatched_right == (1,)
    assert np.isinf(result.costs[:, 1]).all()


def test_association_requires_explicit_covariance_semantics_and_scales_past_24() -> None:
    with pytest.raises(ValueError, match="innovation_covariances"):
        associate_gaussians(
            [np.array([0.0])],
            [np.array([[0.1]])],
            [np.array([0.0])],
            [np.array([[0.1]])],
            unmatched_left_cost=1.0,
            unmatched_right_cost=1.0,
        )
    count = 30
    diagonal = np.full((count, count), np.inf)
    np.fill_diagonal(diagonal, 0.0)
    assignment = solve_one_to_one(
        diagonal, unmatched_left_cost=2.0, unmatched_right_cost=2.0
    )
    assert assignment.pairs == tuple((index, index) for index in range(count))
    explicit = associate_gaussians(
        [np.array([0.0])],
        [np.array([[0.1]])],
        [np.array([1.0])],
        [np.array([[0.1]])],
        innovation_covariances=np.array([[[[4.0]]]]),
        unmatched_left_cost=3.0,
        unmatched_right_cost=3.0,
    )
    assert explicit.assignment.pairs == ((0, 0),)


def test_covariance_intersection_matches_closed_form_and_never_uses_independence() -> None:
    result = covariance_intersection(
        np.array([0.0]),
        np.array([[2.0]]),
        np.array([2.0]),
        np.array([[8.0]]),
        weight_first=0.5,
    )
    expected_covariance = 1.0 / (0.5 / 2.0 + 0.5 / 8.0)
    expected_mean = expected_covariance * (0.5 * 0.0 / 2.0 + 0.5 * 2.0 / 8.0)
    assert result.covariance[0, 0] == pytest.approx(expected_covariance)
    assert result.mean[0] == pytest.approx(expected_mean)
    optimized = covariance_intersection(
        np.array([0.0]), np.array([[1.0]]), np.array([10.0]), np.array([[9.0]])
    )
    assert optimized.weight_first == pytest.approx(1.0)
    assert optimized.covariance[0, 0] == pytest.approx(1.0)


def test_real_delta_t_chronological_replay_matches_event_order_filter() -> None:
    replay = tracker()
    received = message("late", sequence=1, event_time=0.5, measurement=0.4)
    result = replay.ingest_increment(
        received, clock_map=IDENTITY_CLOCK, decision_time=2.0
    )
    assert result.status is IngestStatus.APPLIED

    mean, covariance = manual_predict(
        np.array([0.0, 1.0]), np.diag([1.0, 0.5]), 0.5
    )
    mean, covariance = manual_update(mean, covariance, 0.4)
    mean, covariance = manual_predict(mean, covariance, 1.5)
    snapshot = replay.snapshot()
    assert snapshot.time == 2.0
    assert snapshot.mean == pytest.approx(mean)
    assert snapshot.covariance == pytest.approx(covariance)


def test_replay_uses_state_reference_not_information_cutoff() -> None:
    replay = tracker()
    received = message(
        "state-reference",
        sequence=1,
        event_time=0.75,
        information_cutoff=0.25,
        state_reference=0.75,
        measurement=0.4,
    )
    assert replay.ingest_increment(
        received, clock_map=IDENTITY_CLOCK, decision_time=2.0
    ).applied
    mean, covariance = manual_predict(
        np.array([0.0, 1.0]), np.diag([1.0, 0.5]), 0.75
    )
    mean, covariance = manual_update(mean, covariance, 0.4)
    mean, covariance = manual_predict(mean, covariance, 1.25)
    assert replay.snapshot().mean == pytest.approx(mean)
    assert replay.snapshot().covariance == pytest.approx(covariance)


def test_replay_rejects_future_state_reference_even_with_old_information() -> None:
    received = message(
        "future-state",
        sequence=1,
        event_time=2.5,
        information_cutoff=1.0,
        state_reference=2.5,
        measurement=1.0,
        received_at=2.0,
        deadline=3.0,
    )
    assert (
        tracker().ingest_increment(
            received, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).status
        is IngestStatus.FUTURE_INFORMATION
    )


def test_same_message_set_is_deterministic_under_different_arrival_orders() -> None:
    older = message("older", sequence=9, event_time=0.5, measurement=0.2)
    newer = message("newer", sequence=2, event_time=1.2, measurement=1.4)
    first = tracker()
    second = tracker()
    for received, _event_time in ((newer, 1.2), (older, 0.5)):
        assert first.ingest_increment(
            received, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).applied
    for received, _event_time in ((older, 0.5), (newer, 1.2)):
        assert second.ingest_increment(
            received, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).applied
    assert first.snapshot().mean.tobytes() == second.snapshot().mean.tobytes()
    assert (
        first.snapshot().covariance.tobytes()
        == second.snapshot().covariance.tobytes()
    )


def test_duplicate_message_and_duplicate_lineage_are_no_ops() -> None:
    replay = tracker()
    original = message("first", sequence=1, event_time=1.0, measurement=0.9)
    assert replay.ingest_increment(
        original, clock_map=IDENTITY_CLOCK, decision_time=2.0
    ).applied
    before = replay.snapshot()
    duplicate = replay.ingest_increment(
        original, clock_map=IDENTITY_CLOCK, decision_time=2.0
    )
    assert duplicate.status is IngestStatus.DUPLICATE
    overlap = message(
        "different-envelope",
        sequence=2,
        event_time=1.0,
        measurement=100.0,
        factor_id="factor-first",
    )
    assert (
        replay.ingest_increment(
            overlap, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).status
        is IngestStatus.LINEAGE_OVERLAP
    )
    after = replay.snapshot()
    assert after.mean.tobytes() == before.mean.tobytes()
    assert after.covariance.tobytes() == before.covariance.tobytes()


def test_shared_ancestor_messages_cannot_be_absorbed_twice() -> None:
    replay = tracker()
    first = message(
        "child-a",
        sequence=1,
        event_time=1.0,
        measurement=0.9,
        ancestor_message_ids=("common-parent",),
    )
    second = message(
        "child-b",
        sequence=2,
        event_time=1.0,
        measurement=100.0,
        ancestor_message_ids=("common-parent",),
    )
    assert replay.ingest_increment(
        first, clock_map=IDENTITY_CLOCK, decision_time=2.0
    ).applied
    assert (
        replay.ingest_increment(
            second, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).status
        is IngestStatus.LINEAGE_OVERLAP
    )


def test_boundary_lineage_wrong_target_deadline_and_ttl_fail_closed() -> None:
    prior = FixedLagReplay(
        track_id="track",
        initial_time=0.0,
        initial_mean=np.array([0.0, 1.0]),
        initial_covariance=np.diag([1.0, 0.5]),
        dynamics=constant_velocity_dynamics(1),
        lag=5.0,
        absorbed_factor_ids=("already-in-prior",),
    )
    repeated = message(
        "repeated", sequence=1, event_time=1.0, measurement=1.0,
        factor_id="already-in-prior",
    )
    assert (
        prior.ingest_increment(
            repeated, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).status
        is IngestStatus.LINEAGE_OVERLAP
    )
    wrong = message(
        "wrong", sequence=2, event_time=1.0, measurement=100.0,
        target_id="other-track",
    )
    assert (
        tracker().ingest_increment(
            wrong, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).status
        is IngestStatus.WRONG_TARGET
    )
    late = message(
        "late-deadline", sequence=3, event_time=1.0, measurement=1.0,
        deadline=1.5,
    )
    assert (
        tracker().ingest_increment(
            late, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).status
        is IngestStatus.MISSED_DEADLINE
    )
    expired = message(
        "expired", sequence=4, event_time=1.0, measurement=1.0, ttl=0.25
    )
    assert (
        tracker().ingest_increment(
            expired, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).status
        is IngestStatus.EXPIRED_TTL
    )


def test_correlated_belief_is_forbidden_from_independent_kalman_path() -> None:
    base = message("belief", sequence=1, event_time=1.0, measurement=1.0)
    belief = WireMessage(
        message_id="belief",
        source="sender",
        sequence=1,
        timestamps=base.message.timestamps,
        deadline=3.0,
        coordinate_frame="world",
        ttl=5.0,
        payload=CorrelatedTrackBelief(
            track_id="track",
            existence_probability=0.9,
            mean=np.array([1.0, 0.0]),
            covariance=np.eye(2),
            identity_probabilities=(("track", 1.0),),
            lineage=Lineage(("possibly-shared",), False),
        ),
    )
    with pytest.raises(CorrelatedBeliefRequiresCI):
        tracker().ingest_increment(
            ReceivedMessage(belief, received_at=2.0),
            clock_map=IDENTITY_CLOCK,
            decision_time=2.0,
        )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (OutOfWindowPolicy.DROP, IngestStatus.OUT_OF_WINDOW_DROPPED),
        (OutOfWindowPolicy.REJECT, IngestStatus.OUT_OF_WINDOW_REJECTED),
    ],
)
def test_out_of_window_policy_is_explicit(
    policy: OutOfWindowPolicy, expected: IngestStatus
) -> None:
    replay = tracker(lag=1.0, policy=policy)
    replay.advance_to(3.0)
    stale = message(
        "stale", sequence=1, event_time=1.0, measurement=1.0, received_at=3.0
    )
    result = replay.ingest_increment(
        stale, clock_map=IDENTITY_CLOCK, decision_time=3.0
    )
    assert result.status is expected
    assert replay.factor_count == 0
    assert (
        replay.ingest_increment(
            stale, clock_map=IDENTITY_CLOCK, decision_time=3.0
        ).status
        is IngestStatus.DUPLICATE
    )


def test_replay_rejects_future_arrival_and_conservatively_future_information() -> None:
    future_arrival = message(
        "arrival", sequence=1, event_time=1.0, measurement=0.0, received_at=3.0
    )
    replay = tracker()
    assert (
        replay.ingest_increment(
            future_arrival, clock_map=IDENTITY_CLOCK, decision_time=2.0
        ).status
        is IngestStatus.FUTURE_ARRIVAL
    )
    uncertain = message("clock", sequence=2, event_time=1.9, measurement=0.0)
    uncertain_clock = AffineClockMap(
        1.0, 0.0, np.diag([0.0, 0.04])
    )
    assert (
        replay.ingest_increment(
            uncertain, clock_map=uncertain_clock, decision_time=2.0
        ).status
        is IngestStatus.FUTURE_INFORMATION
    )


def test_append_only_commit_prefix_hash_is_stable_and_snapshot_is_copied() -> None:
    log = AppendOnlyCommitLog()
    mutable_snapshot = {"tracks": [1, 2], "mean": np.array([0.0, 1.0])}
    first = log.commit(
        decision_time=1.0, snapshot=mutable_snapshot, message_ids=["m2", "m1"]
    )
    prefix = log.prefix_hash()
    mutable_snapshot["tracks"].append(99)
    second = log.commit(decision_time=2.0, snapshot={"tracks": [3]})
    assert log.prefix_hash(1) == prefix == first.record_hash
    assert second.previous_hash == first.record_hash
    assert log.records[0].snapshot["tracks"] == [1, 2]
    assert log.verify()
    with pytest.raises(ValueError, match="strictly increasing"):
        log.commit(decision_time=2.0, snapshot={})


def scheduling_message(index: str, padding: int = 0) -> WireMessage:
    received = message(
        index,
        sequence=ord(index[0]),
        event_time=1.0,
        measurement=1.0,
        factor_id=f"factor-{index}-{'x' * padding}",
    )
    return received.message


def scheduling_candidate(
    item: WireMessage, risk_reduction: float, on_time_probability: float
) -> ScheduleCandidate:
    network_time = 2.0
    return ScheduleCandidate(
        message=item,
        network_transmit_time=network_time,
        score_evidence=CausalScoreEvidenceV1(
            as_of_network_time=network_time,
            risk_reduction=risk_reduction,
            on_time_probability=on_time_probability,
            confidence=0.0,
            age_seconds=0.0,
            tracker_state_sha256="a" * 64,
            channel_model_sha256="b" * 64,
            scorer_config_sha256="c" * 64,
            on_time_estimate_sha256="d" * 64,
            candidate_message_sha256=wire_digest(item),
            candidate_network_transmit_time=network_time,
            candidate_deadline=item.deadline,
            ground_truth_free=True,
        ),
    )


def test_exact_scheduler_uses_on_time_risk_and_application_payload_bytes() -> None:
    first = scheduling_candidate(scheduling_message("a"), 6.0, 1.0)
    second = scheduling_candidate(scheduling_message("b"), 12.0, 0.5)
    # The tempting item has higher singleton value, but its larger canonical
    # application payload prevents combining it with either six-value item.
    tempting = scheduling_candidate(
        scheduling_message("c", padding=100), 11.0, 1.0
    )
    budget = len(encode_message(first.message)) + len(encode_message(second.message))
    result = select_exact_budget([tempting, second, first], budget_bytes=budget)
    assert result.message_ids == ("a", "b")
    assert result.expected_risk_reduction == pytest.approx(12.0)
    assert result.total_bytes == sum(
        len(encode_message(candidate.message)) for candidate in result.selected
    )
    assert result.total_bytes <= budget
    assert result.budget_basis is BudgetBasis.APPLICATION

    too_small = select_exact_budget(
        [first], budget_bytes=len(encode_message(first.message)) - 1
    )
    assert too_small.message_ids == ()
    assert too_small.total_bytes == 0


def test_scheduler_budget_counts_unicode_utf8_and_protocol_overhead() -> None:
    unicode_candidate = scheduling_candidate(scheduling_message("车端"), 1.0, 1.0)
    encoded = encode_message(unicode_candidate.message)
    assert len(encoded) > len(encoded.decode("utf-8"))
    assert select_exact_budget(
        [unicode_candidate], budget_bytes=len(encoded)
    ).message_ids == ("车端",)
    assert select_exact_budget(
        [unicode_candidate], budget_bytes=len(encoded) - 1
    ).message_ids == ()
