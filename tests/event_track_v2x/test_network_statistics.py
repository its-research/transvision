from dataclasses import replace

import numpy as np
import pytest

from transvision.models.event_track_v2x.network import (
    ByteAccountV1,
    DEFAULT_NETWORK_SEEDS_V1,
    JitterModel,
    NetworkConditionId,
    NetworkTraceEventV1,
    NetworkTraceV1,
    PacketRequest,
    WireAccountingConfigV1,
    account_wire_bytes_v1,
    condition_plan_v1,
    default_condition_plans_v1,
    generate_condition_trace_v1,
)
from transvision.models.event_track_v2x.scheduler import (
    BudgetBasis,
    CausalScoreEvidenceV1,
    ScheduleCandidate,
    SchedulerKind,
    schedule_baseline,
    schedule_rate_limited_v1,
)
from transvision.models.event_track_v2x.schema import (
    IndependentIncrement,
    Lineage,
    LocalTimestamps,
    WireMessage,
)
from transvision.models.event_track_v2x.wire import encode_message, wire_digest
from transvision.models.event_track_v2x.statistics import (
    Alternative,
    MetricObservationV1,
    aggregate_robust_macro_v1,
    holm_correction_v1,
    paired_sequence_bca_v1,
    paired_sequence_permutation_v1,
)


def packets(count: int, *, sources: int = 1) -> list[PacketRequest]:
    return [
        PacketRequest(
            message_id=f"m-{index:04d}",
            source=f"agent-{index % sources}",
            sequence=index,
            transmitted_at=index * 0.1,
            deadline=index * 0.1 + 1.0,
            encoded_bytes=100 + index % 7,
        )
        for index in range(count)
    ]


def message(index: int) -> WireMessage:
    time = 1.0 + index * 0.1
    return WireMessage(
        message_id=f"candidate-{index}",
        source="roadside",
        sequence=index,
        timestamps=LocalTimestamps(time, time, time, time),
        deadline=time + 1.0,
        coordinate_frame="world",
        ttl=2.0,
        payload=IndependentIncrement(
            target_id=f"track-{index}",
            measurement=np.asarray([float(index)]),
            measurement_matrix=np.asarray([[1.0]]),
            measurement_covariance=np.asarray([[1.0]]),
            lineage=Lineage((f"factor-{index}",), True),
        ),
    )


def _candidate(
    item: WireMessage,
    risk_reduction: float,
    on_time_probability: float,
    *,
    confidence: float = 0.0,
    age_seconds: float = 0.0,
    projected_on_wire_bytes: int | None = None,
    network_transmit_time: float | None = None,
) -> ScheduleCandidate:
    network_time = (
        item.timestamps.transmitted
        if network_transmit_time is None
        else network_transmit_time
    )
    return ScheduleCandidate(
        message=item,
        network_transmit_time=network_time,
        score_evidence=CausalScoreEvidenceV1(
            as_of_network_time=network_time,
            risk_reduction=risk_reduction,
            on_time_probability=on_time_probability,
            confidence=confidence,
            age_seconds=age_seconds,
            tracker_state_sha256="a" * 64,
            channel_model_sha256="b" * 64,
            scorer_config_sha256="c" * 64,
            on_time_estimate_sha256="d" * 64,
            candidate_message_sha256=wire_digest(item),
            candidate_network_transmit_time=network_time,
            candidate_deadline=item.deadline,
            ground_truth_free=True,
        ),
        projected_on_wire_bytes=projected_on_wire_bytes,
    )


def test_preregistered_condition_plans_are_complete_and_canonical() -> None:
    plans = default_condition_plans_v1()
    assert tuple(plan.condition_id.value for plan in plans) == tuple(
        f"C{index}" for index in range(9)
    )
    assert all(plan.seeds == DEFAULT_NETWORK_SEEDS_V1 for plan in plans)
    assert len({plan.content_sha256 for plan in plans}) == 9
    assert condition_plan_v1("C2").jitter_model is JitterModel.TRUNCATED_NORMAL
    ge = condition_plan_v1("C4").gilbert_elliott
    assert ge is not None
    assert ge.probability_bad_to_good == pytest.approx(0.2)
    stationary_bad = (1.0 - ge.probability_good_to_good) / (
        1.0 - ge.probability_good_to_good + ge.probability_bad_to_good
    )
    assert stationary_bad == pytest.approx(0.3)


def test_condition_traces_are_order_invariant_and_hashable() -> None:
    requests = packets(80, sources=2)
    first = generate_condition_trace_v1(requests, "C8", seed=1001)
    second = generate_condition_trace_v1(list(reversed(requests)), "C8", seed=1001)
    assert first == second
    assert first.canonical_bytes == second.canonical_bytes
    assert first.content_sha256 == second.content_sha256
    assert len(first.content_sha256) == 64
    with pytest.raises(ValueError, match="not registered"):
        generate_condition_trace_v1(requests, "C8", seed=999)


def test_every_registered_c0_c8_seed_generates_a_distinct_sealed_trace() -> None:
    requests = packets(20, sources=2)
    traces = [
        generate_condition_trace_v1(requests, plan, seed=seed)
        for plan in default_condition_plans_v1()
        for seed in DEFAULT_NETWORK_SEEDS_V1
    ]
    assert len(traces) == 90
    assert len({trace.content_sha256 for trace in traces}) == 90


def test_c0_c2_delay_contract_and_exact_trace_byte_totals() -> None:
    requests = packets(40)
    clean = generate_condition_trace_v1(requests, NetworkConditionId.C0, seed=1001)
    assert all(not event.dropped for event in clean.events)
    assert [event.arrival_time for event in clean.events] == [
        request.transmitted_at for request in requests
    ]
    assert clean.application_bytes == sum(request.encoded_bytes for request in requests)
    assert clean.on_wire_bytes == sum(
        event.byte_account.total_on_wire_bytes for event in clean.events
    )
    assert clean.on_wire_bytes > clean.application_bytes
    assert clean.on_time_application_bytes == clean.application_bytes

    jittered = generate_condition_trace_v1(requests, "C2", seed=1001)
    delays = [
        event.arrival_time - event.packet.transmitted_at
        for event in jittered.events
        if event.arrival_time is not None
    ]
    assert min(delays) >= 0.15 - 1e-12
    assert max(delays) <= 0.45 + 1e-12
    assert np.std(delays) > 0.0


def test_loss_outage_clock_and_pose_conditions_materialize_expected_fields() -> None:
    requests = packets(320, sources=2)
    iid = generate_condition_trace_v1(requests, "C3", seed=1002)
    assert 60 < sum(event.dropped for event in iid.events) < 140

    burst = generate_condition_trace_v1(requests, "C4", seed=1002)
    dropped = [event.dropped for event in burst.events]
    assert any(left and right for left, right in zip(dropped, dropped[1:]))

    outage = generate_condition_trace_v1(requests, "C5", seed=1002)
    outage_indices = [index for index, event in enumerate(outage.events) if event.dropped]
    assert len(outage_indices) >= 20
    assert any(right == left + 1 for left, right in zip(outage_indices, outage_indices[1:]))

    clock = generate_condition_trace_v1(requests, "C6", seed=1002)
    for source in ("agent-0", "agent-1"):
        source_events = [event for event in clock.events if event.packet.source == source]
        assert len({event.clock_offset_seconds for event in source_events}) == 1
        assert len({event.clock_drift_ppm for event in source_events}) == 1
        assert abs(source_events[0].clock_offset_seconds) <= 0.1
        assert abs(source_events[0].clock_drift_ppm) <= 50.0
        assert source_events[-1].clock_error_at_transmit_seconds != pytest.approx(
            source_events[0].clock_error_at_transmit_seconds
        )

    pose = generate_condition_trace_v1(requests, "C7", seed=1002)
    assert any(
        any(value != 0.0 for value in event.translation_noise_metres)
        for event in pose.events
    )
    assert any(event.yaw_noise_degrees != 0.0 for event in pose.events)


def test_wire_accounting_includes_fragment_headers_retries_and_acks() -> None:
    config = WireAccountingConfigV1(
        max_frame_bytes=100,
        per_fragment_header_bytes=10,
        per_message_header_bytes=5,
        ack_frame_bytes=20,
    )
    account = account_wire_bytes_v1(
        175,
        config,
        data_transmissions=2,
        ack_transmissions=2,
    )
    assert account.fragments_per_transmission == 2
    assert account.data_on_wire_bytes == 400
    assert account.ack_on_wire_bytes == 40
    assert account.total_on_wire_bytes == 440
    assert account.retransmissions == 1
    with pytest.raises(ValueError, match="smaller"):
        ByteAccountV1(
            application_bytes=100,
            data_transmissions=1,
            fragments_per_transmission=1,
            data_on_wire_bytes=99,
            ack_transmissions=0,
            ack_on_wire_bytes=0,
            total_on_wire_bytes=99,
        )


def test_all_scheduler_baselines_are_deterministic_and_byte_capped() -> None:
    candidates = (
        _candidate(
            message(0),
            6.0,
            1.0,
            confidence=0.2,
            age_seconds=3.0,
            projected_on_wire_bytes=600,
        ),
        _candidate(
            message(1),
            6.0,
            1.0,
            confidence=0.9,
            age_seconds=2.0,
            projected_on_wire_bytes=600,
        ),
        _candidate(
            message(2),
            11.0,
            1.0,
            confidence=0.5,
            age_seconds=1.0,
            projected_on_wire_bytes=1200,
        ),
    )
    full = schedule_baseline(
        candidates,
        SchedulerKind.FULL_SEND,
        budget_bytes=1200,
    )
    assert full.message_ids == ("candidate-0", "candidate-1", "candidate-2")
    assert not full.budget_compliant

    periodic = schedule_baseline(
        candidates,
        SchedulerKind.PERIODIC,
        budget_bytes=1200,
        period=2,
        phase=0,
    )
    assert periodic.message_ids == ("candidate-0",)

    fifo = schedule_baseline(
        candidates,
        SchedulerKind.FIFO_AOI,
        budget_bytes=600,
    )
    assert fifo.message_ids == ("candidate-0",)
    confidence = schedule_baseline(
        candidates,
        SchedulerKind.CONFIDENCE_TOP_K,
        budget_bytes=600,
    )
    assert confidence.message_ids == ("candidate-1",)
    voi = schedule_baseline(
        candidates,
        SchedulerKind.MARGINAL_VOI,
        budget_bytes=1200,
    )
    assert voi.message_ids == ("candidate-0", "candidate-1")
    assert voi.total_on_wire_bytes == 1200
    assert voi.expected_risk_reduction == pytest.approx(12.0)

    random_first = schedule_baseline(
        candidates,
        SchedulerKind.RANDOM,
        budget_bytes=1200,
        seed=1001,
    )
    random_second = schedule_baseline(
        tuple(reversed(candidates)),
        SchedulerKind.RANDOM,
        budget_bytes=1200,
        seed=1001,
    )
    assert random_first == random_second
    assert random_first.budget_basis is BudgetBasis.ON_WIRE
    assert random_first.budget_compliant


def test_scheduler_uses_network_monotonic_time_not_unsynchronised_sender_clock() -> None:
    first_message = replace(
        message(0),
        source="agent-a",
        timestamps=LocalTimestamps(100.0, 100.0, 100.0, 100.0),
    )
    second_message = replace(
        message(1),
        source="agent-b",
        timestamps=LocalTimestamps(0.0, 0.0, 0.0, 0.0),
    )
    first = _candidate(
        first_message,
        1.0,
        1.0,
        network_transmit_time=1.0,
        projected_on_wire_bytes=600,
    )
    second = _candidate(
        second_message,
        1.0,
        1.0,
        network_transmit_time=2.0,
        projected_on_wire_bytes=600,
    )
    result = schedule_baseline(
        (second, first), SchedulerKind.FULL_SEND, budget_bytes=None
    )
    assert result.message_ids == (first_message.message_id, second_message.message_id)


def test_scheduler_rejects_future_or_ground_truth_score_evidence() -> None:
    item = message(0)
    with pytest.raises(ValueError, match="ground-truth-free"):
        CausalScoreEvidenceV1(
            as_of_network_time=1.0,
            risk_reduction=1.0,
            on_time_probability=1.0,
            confidence=1.0,
            age_seconds=0.0,
            tracker_state_sha256="a" * 64,
            channel_model_sha256="b" * 64,
            scorer_config_sha256="c" * 64,
            on_time_estimate_sha256="d" * 64,
            candidate_message_sha256=wire_digest(item),
            candidate_network_transmit_time=1.0,
            candidate_deadline=item.deadline,
            ground_truth_free=False,
        )
    evidence = CausalScoreEvidenceV1(
        as_of_network_time=2.0,
        risk_reduction=1.0,
        on_time_probability=1.0,
        confidence=1.0,
        age_seconds=0.0,
        tracker_state_sha256="a" * 64,
        channel_model_sha256="b" * 64,
        scorer_config_sha256="c" * 64,
        on_time_estimate_sha256="d" * 64,
        candidate_message_sha256=wire_digest(item),
        candidate_network_transmit_time=2.0,
        candidate_deadline=item.deadline,
        ground_truth_free=True,
    )
    with pytest.raises(ValueError, match="future"):
        ScheduleCandidate(
            message=item,
            network_transmit_time=1.0,
            score_evidence=evidence,
            projected_on_wire_bytes=600,
        )


def test_on_wire_budget_never_falls_back_to_application_bytes() -> None:
    candidate = _candidate(message(0), 1.0, 1.0)
    with pytest.raises(ValueError, match="silent fallback"):
        schedule_baseline(
            (candidate,), SchedulerKind.CONFIDENCE_TOP_K, budget_bytes=10_000
        )
    with pytest.raises(ValueError, match="silent fallback"):
        schedule_baseline(
            (candidate,),
            SchedulerKind.CONFIDENCE_TOP_K,
            budget_bytes=10_000,
            budget_basis=BudgetBasis.APPLICATION,
        )


def test_stream_scheduler_enforces_bytes_per_second_token_bucket() -> None:
    messages = tuple(message(index) for index in (0, 1, 11))
    wire_accounting = WireAccountingConfigV1()
    trace = generate_condition_trace_v1(
        tuple(
            PacketRequest(
                message_id=item.message_id,
                source=item.source,
                sequence=item.sequence,
                transmitted_at=item.timestamps.transmitted,
                deadline=item.deadline,
                encoded_bytes=len(encode_message(item)),
            )
            for item in messages
        ),
        condition_plan_v1("C0", wire_accounting=wire_accounting),
        seed=1001,
    )
    event_by_id = {event.packet.message_id: event for event in trace.events}
    candidates = tuple(
        _candidate(
            item,
            1.0,
            1.0,
            confidence=1.0,
            projected_on_wire_bytes=account_wire_bytes_v1(
                len(encode_message(item)),
                wire_accounting,
                data_transmissions=1,
                ack_transmissions=1,
            ).total_on_wire_bytes,
        )
        for item in messages
    )
    result = schedule_rate_limited_v1(
        candidates,
        SchedulerKind.CONFIDENCE_TOP_K,
        network_trace=trace,
        wire_accounting=wire_accounting,
        byte_rate_limit_per_second=600,
    )
    assert result.message_ids == ("candidate-0", "candidate-11")
    assert result.total_on_wire_bytes == sum(
        event_by_id[message_id].byte_account.total_on_wire_bytes
        for message_id in result.message_ids
    )
    assert result.total_reserved_on_wire_bytes == sum(
        candidate.on_wire_byte_cost
        for candidate in candidates
        if candidate.message.message_id in result.message_ids
    )
    assert result.network_trace_sha256 == trace.content_sha256
    assert result.budget_compliant

    with pytest.raises(ValueError, match="disagrees with causal reservation"):
        schedule_rate_limited_v1(
            (replace(
                candidates[0],
                projected_on_wire_bytes=candidates[0].on_wire_byte_cost + 1,
            ),),
            SchedulerKind.CONFIDENCE_TOP_K,
            network_trace=trace,
            wire_accounting=wire_accounting,
            byte_rate_limit_per_second=600,
        )

    full = schedule_rate_limited_v1(
        candidates,
        SchedulerKind.FULL_SEND,
        network_trace=trace,
        wire_accounting=wire_accounting,
        byte_rate_limit_per_second=600,
    )
    assert full.message_ids == ("candidate-0", "candidate-1", "candidate-11")
    assert not full.budget_compliant


def test_stream_scheduler_does_not_use_realised_ack_or_drop_to_select() -> None:
    messages = tuple(message(index) for index in (0, 1))
    wire_accounting = WireAccountingConfigV1()
    packets = tuple(
        PacketRequest(
            message_id=item.message_id,
            source=item.source,
            sequence=item.sequence,
            transmitted_at=item.timestamps.transmitted,
            deadline=item.deadline,
            encoded_bytes=len(encode_message(item)),
        )
        for item in messages
    )
    delivered_trace = generate_condition_trace_v1(
        packets,
        condition_plan_v1("C0", wire_accounting=wire_accounting),
        seed=1001,
    )
    first = delivered_trace.events[0]
    dropped_first = replace(
        first,
        arrival_time=None,
        dropped=True,
        byte_account=account_wire_bytes_v1(
            first.packet.encoded_bytes,
            wire_accounting,
            data_transmissions=1,
            ack_transmissions=0,
        ),
    )
    dropped_trace = NetworkTraceV1(
        condition_id=delivered_trace.condition_id,
        seed=delivered_trace.seed,
        condition_plan=delivered_trace.condition_plan,
        condition_plan_sha256=delivered_trace.condition_plan_sha256,
        events=(dropped_first, delivered_trace.events[1]),
    )
    candidates = tuple(
        _candidate(
            item,
            1.0,
            1.0,
            confidence=1.0,
            projected_on_wire_bytes=account_wire_bytes_v1(
                len(encode_message(item)),
                wire_accounting,
                data_transmissions=1,
                ack_transmissions=1,
            ).total_on_wire_bytes,
        )
        for item in messages
    )
    reservation = candidates[0].on_wire_byte_cost
    delivered = schedule_rate_limited_v1(
        candidates,
        SchedulerKind.CONFIDENCE_TOP_K,
        network_trace=delivered_trace,
        wire_accounting=wire_accounting,
        byte_rate_limit_per_second=reservation,
    )
    dropped = schedule_rate_limited_v1(
        candidates,
        SchedulerKind.CONFIDENCE_TOP_K,
        network_trace=dropped_trace,
        wire_accounting=wire_accounting,
        byte_rate_limit_per_second=reservation,
    )
    assert delivered.message_ids == dropped.message_ids == ("candidate-0",)
    assert delivered.total_reserved_on_wire_bytes == dropped.total_reserved_on_wire_bytes
    assert delivered.total_on_wire_bytes > dropped.total_on_wire_bytes


def test_stream_scheduler_fails_closed_on_trace_or_realised_byte_overrun() -> None:
    item = message(0)
    wire_accounting = WireAccountingConfigV1()
    trace = generate_condition_trace_v1(
        (
            PacketRequest(
                message_id=item.message_id,
                source=item.source,
                sequence=item.sequence,
                transmitted_at=item.timestamps.transmitted,
                deadline=item.deadline,
                encoded_bytes=len(encode_message(item)),
            ),
        ),
        condition_plan_v1("C0", wire_accounting=wire_accounting),
        seed=1001,
    )
    reservation = account_wire_bytes_v1(
        len(encode_message(item)),
        wire_accounting,
        data_transmissions=1,
        ack_transmissions=1,
    ).total_on_wire_bytes
    candidate = _candidate(
        item,
        1.0,
        1.0,
        confidence=1.0,
        projected_on_wire_bytes=reservation,
    )
    with pytest.raises(ValueError, match="condition plan disagrees"):
        schedule_rate_limited_v1(
            (candidate,),
            SchedulerKind.CONFIDENCE_TOP_K,
            network_trace=trace,
            wire_accounting=WireAccountingConfigV1(ack_frame_bytes=65),
            byte_rate_limit_per_second=reservation,
        )

    event = trace.events[0]
    with pytest.raises(ValueError, match="legacy event transmissions"):
        NetworkTraceEventV1(
            packet=event.packet,
            arrival_time=event.arrival_time,
            dropped=event.dropped,
            channel_state=event.channel_state,
            clock_offset_seconds=event.clock_offset_seconds,
            clock_drift_ppm=event.clock_drift_ppm,
            clock_error_at_transmit_seconds=event.clock_error_at_transmit_seconds,
            translation_noise_metres=event.translation_noise_metres,
            yaw_noise_degrees=event.yaw_noise_degrees,
            byte_account=account_wire_bytes_v1(
                event.packet.encoded_bytes,
                wire_accounting,
                data_transmissions=2,
                ack_transmissions=1,
            ),
            wire_accounting_config=wire_accounting,
        )


def metric_observations() -> list[MetricObservationV1]:
    values = {
        ("s1", "C1", 1001): 1.0,
        ("s1", "C1", 1002): 3.0,
        ("s1", "C2", 1001): 5.0,
        ("s1", "C2", 1002): 7.0,
        ("s2", "C1", 1001): 2.0,
        ("s2", "C1", 1002): 4.0,
        ("s2", "C2", 1001): 6.0,
        ("s2", "C2", 1002): 8.0,
    }
    return [
        MetricObservationV1(
            method="eventtrack",
            metric="AssA",
            sequence_id=sequence,
            condition_id=condition,
            network_seed=network_seed,
            training_seed=1337,
            value=value,
        )
        for (sequence, condition, network_seed), value in values.items()
    ]


def test_robust_macro_averages_seeds_then_conditions_and_sequences() -> None:
    observations = metric_observations()
    result = aggregate_robust_macro_v1(
        observations,
        method="eventtrack",
        metric="AssA",
        expected_sequence_ids=("s1", "s2"),
        condition_ids=("C1", "C2"),
        network_seeds=(1001, 1002),
        training_seeds=(1337,),
    )
    assert result.sequence_values == (("s1", 4.0), ("s2", 5.0))
    assert result.condition_values == (("C1", 2.5), ("C2", 6.5))
    assert result.value == pytest.approx(4.5)

    with pytest.raises(ValueError, match="missing robust metric cells"):
        aggregate_robust_macro_v1(
            observations[:-1],
            method="eventtrack",
            metric="AssA",
            expected_sequence_ids=("s1", "s2"),
            condition_ids=("C1", "C2"),
            network_seeds=(1001, 1002),
            training_seeds=(1337,),
        )

    only_first_sequence = [
        item for item in observations if item.sequence_id == "s1"
    ]
    with pytest.raises(ValueError, match="missing robust metric cells"):
        aggregate_robust_macro_v1(
            only_first_sequence,
            method="eventtrack",
            metric="AssA",
            expected_sequence_ids=("s1", "s2"),
            condition_ids=("C1", "C2"),
            network_seeds=(1001, 1002),
            training_seeds=(1337,),
        )


def test_failed_runs_are_zero_scored_but_missing_runs_are_not() -> None:
    observations = metric_observations()
    failed = observations[-1]
    observations[-1] = MetricObservationV1(
        method=failed.method,
        metric=failed.metric,
        sequence_id=failed.sequence_id,
        condition_id=failed.condition_id,
        network_seed=failed.network_seed,
        training_seed=failed.training_seed,
        value=None,
        failed=True,
    )
    result = aggregate_robust_macro_v1(
        observations,
        method="eventtrack",
        metric="AssA",
        expected_sequence_ids=("s1", "s2"),
        condition_ids=("C1", "C2"),
        network_seeds=(1001, 1002),
        training_seeds=(1337,),
    )
    assert result.failed_runs == 1
    assert result.value == pytest.approx(3.5)


def test_c1_c9_use_condition_specific_replicates_and_equal_condition_weight() -> None:
    observations: list[MetricObservationV1] = []
    for sequence in ("s1", "s2"):
        for condition_number in range(1, 9):
            for network_seed in (1001, 1002):
                observations.append(
                    MetricObservationV1(
                        method="eventtrack",
                        metric="AssA",
                        sequence_id=sequence,
                        condition_id=f"C{condition_number}",
                        network_seed=network_seed,
                        training_seed=1337,
                        value=float(condition_number),
                    )
                )
        for trace_id in ("heldout-01", "heldout-02", "heldout-03"):
            observations.append(
                MetricObservationV1(
                    method="eventtrack",
                    metric="AssA",
                    sequence_id=sequence,
                    condition_id="C9",
                    network_seed=None,
                    replicate_id=trace_id,
                    training_seed=1337,
                    value=9.0,
                )
            )

    with pytest.raises(ValueError, match="C9 held-out trace replicate IDs"):
        aggregate_robust_macro_v1(
            observations,
            method="eventtrack",
            metric="AssA",
            expected_sequence_ids=("s1", "s2"),
            network_seeds=(1001, 1002),
            training_seeds=(1337,),
        )

    result = aggregate_robust_macro_v1(
        observations,
        method="eventtrack",
        metric="AssA",
        expected_sequence_ids=("s1", "s2"),
        network_seeds=(1001, 1002),
        condition_replicates={
            "C9": ("heldout-01", "heldout-02", "heldout-03")
        },
        training_seeds=(1337,),
    )
    assert result.observations == 38
    assert result.value == pytest.approx(5.0)
    assert dict(result.condition_values)["C9"] == pytest.approx(9.0)


def test_paired_bca_permutation_and_holm_are_reproducible() -> None:
    control = {f"s{index}": 0.5 for index in range(1, 5)}
    treatment = {
        "s1": 0.6,
        "s2": 0.7,
        "s3": 0.8,
        "s4": 0.9,
    }
    first = paired_sequence_bca_v1(
        treatment,
        control,
        resamples=2_000,
        seed=17,
    )
    second = paired_sequence_bca_v1(
        treatment,
        control,
        resamples=2_000,
        seed=17,
    )
    assert first == second
    assert first.estimate == pytest.approx(0.25)
    assert first.confidence_lower > 0.0

    permutation = paired_sequence_permutation_v1(
        treatment,
        control,
        alternative=Alternative.GREATER,
        permutations=10_000,
    )
    assert permutation.exact
    assert permutation.permutations == 16
    assert permutation.p_value == pytest.approx(1.0 / 16.0)

    holm = holm_correction_v1({"a": 0.01, "b": 0.03, "c": 0.04})
    assert [result.adjusted_p_value for result in holm] == pytest.approx(
        [0.03, 0.06, 0.06]
    )
    assert [result.reject for result in holm] == [True, False, False]


def test_holm_preregistered_boundary_is_strict() -> None:
    result = holm_correction_v1({"comparison": 0.05}, alpha=0.05)
    assert result[0].adjusted_p_value == pytest.approx(0.05)
    assert not result[0].reject
