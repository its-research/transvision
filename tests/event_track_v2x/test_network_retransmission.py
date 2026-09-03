from dataclasses import replace
import json

import numpy as np
import pytest

from transvision.models.event_track_v2x.network import (
    SYNTHETIC_LEDGER_SOURCE_V1,
    GilbertElliott,
    LossModel,
    NetworkConditionId,
    NetworkConditionPlanV1,
    NetworkTraceError,
    NetworkTraceEventV1,
    NetworkTraceV1,
    PacketRequest,
    SyntheticTransmissionAttemptV1,
    WireAccountingConfigV1,
    account_wire_bytes_v1,
    bounded_wire_reservation_v1,
    condition_plan_v1,
    decode_network_trace_v1,
    generate_condition_trace_v1,
)
from transvision.models.event_track_v2x.scheduler import (
    CausalScoreEvidenceV1,
    ScheduleCandidate,
    SchedulerKind,
    schedule_rate_limited_v1,
)
from transvision.models.event_track_v2x.schema import (
    IndependentIncrement,
    Lineage,
    LocalTimestamps,
    WireMessage,
)
from transvision.models.event_track_v2x.wire import encode_message, wire_digest


def _packet(message_id: str, *, transmitted_at: float = 0.0) -> PacketRequest:
    return PacketRequest(
        message_id=message_id,
        source="roadside",
        sequence=int(message_id.rsplit("-", 1)[-1]),
        transmitted_at=transmitted_at,
        deadline=transmitted_at + 2.0,
        encoded_bytes=175,
    )


def _message() -> WireMessage:
    time = 1.0
    return WireMessage(
        message_id="candidate-0",
        source="roadside",
        sequence=0,
        timestamps=LocalTimestamps(time, time, time, time),
        deadline=3.0,
        coordinate_frame="world",
        ttl=2.0,
        payload=IndependentIncrement(
            target_id="track-0",
            measurement=np.asarray([0.0]),
            measurement_matrix=np.asarray([[1.0]]),
            measurement_covariance=np.asarray([[1.0]]),
            lineage=Lineage(("factor-0",), True),
        ),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _ge_retransmission_trace() -> NetworkTraceV1:
    wire = WireAccountingConfigV1(
        max_frame_bytes=100,
        per_fragment_header_bytes=10,
        per_message_header_bytes=5,
        ack_frame_bytes=20,
        ack_timeout_seconds=0.05,
        max_retransmissions=2,
    )
    ge = GilbertElliott(
        probability_good_to_good=0.8,
        probability_bad_to_good=0.2,
        success_probability_good=0.0,
        success_probability_bad=0.0,
        initial_good_probability=0.7,
    )
    plan = NetworkConditionPlanV1(
        NetworkConditionId.C4,
        seeds=(1001,),
        loss_model=LossModel.GILBERT_ELLIOTT,
        gilbert_elliott=ge,
        wire_accounting=wire,
    )
    return generate_condition_trace_v1(
        (_packet("message-0", transmitted_at=1.0),),
        plan,
        seed=1001,
    )


def test_default_configuration_preserves_no_retry_behavior() -> None:
    packets = (_packet("message-0"), _packet("message-1", transmitted_at=0.1))
    first = generate_condition_trace_v1(packets, "C1", seed=1001)
    second = generate_condition_trace_v1(tuple(reversed(packets)), "C1", seed=1001)

    assert first.content_sha256 == second.content_sha256
    assert all(event.byte_account.data_transmissions == 1 for event in first.events)
    assert all(not event.attempt_timeline for event in first.events)
    assert WireAccountingConfigV1().max_retransmissions == 0


def test_all_dropped_attempts_stop_at_bound_and_account_exact_bytes() -> None:
    wire = WireAccountingConfigV1(
        max_frame_bytes=100,
        per_fragment_header_bytes=10,
        per_message_header_bytes=5,
        ack_frame_bytes=20,
        ack_timeout_seconds=0.05,
        max_retransmissions=2,
    )
    plan = NetworkConditionPlanV1(
        NetworkConditionId.C3,
        seeds=(1001,),
        loss_model=LossModel.IID,
        iid_loss_probability=1.0,
        wire_accounting=wire,
    )
    packet = _packet("message-0")

    first = generate_condition_trace_v1((packet,), plan, seed=1001)
    second = generate_condition_trace_v1((packet,), plan, seed=1001)
    event = first.events[0]
    attempts = event.attempt_timeline

    assert first.content_sha256 == second.content_sha256
    assert tuple(attempt.attempt_index for attempt in attempts) == (0, 1, 2)
    assert [attempt.data_sent_at for attempt in attempts] == pytest.approx(
        [0.0, 0.05, 0.10]
    )
    assert [attempt.ack_timeout_at for attempt in attempts] == pytest.approx(
        [0.05, 0.10, 0.15]
    )
    assert all(attempt.data_dropped for attempt in attempts)
    assert all(attempt.ack_timed_out for attempt in attempts)
    assert attempts[-1].retry_exhausted
    assert all(
        attempt.to_dict()["ledger_source"] == SYNTHETIC_LEDGER_SOURCE_V1
        for attempt in attempts
    )
    assert event.dropped
    assert event.byte_account.data_transmissions == 3
    assert event.byte_account.retransmissions == 2
    assert event.byte_account.ack_transmissions == 0
    assert event.byte_account.data_on_wire_bytes == 600
    assert event.byte_account.total_on_wire_bytes == 600
    assert first.on_wire_bytes == 600
    reservation = bounded_wire_reservation_v1(packet.encoded_bytes, wire)
    assert reservation.data_transmissions == 3
    assert reservation.ack_transmissions == 3
    assert reservation.total_on_wire_bytes == 660
    assert event.byte_account.total_on_wire_bytes <= reservation.total_on_wire_bytes


def test_retransmission_trace_hash_is_input_order_invariant() -> None:
    wire = WireAccountingConfigV1(
        ack_timeout_seconds=0.15,
        max_retransmissions=1,
    )
    plan = NetworkConditionPlanV1(
        NetworkConditionId.C3,
        seeds=(1001,),
        loss_model=LossModel.IID,
        iid_loss_probability=1.0,
        wire_accounting=wire,
    )
    packets = (
        _packet("message-0", transmitted_at=1.0),
        _packet("message-1", transmitted_at=1.0),
    )

    first = generate_condition_trace_v1(packets, plan, seed=1001)
    second = generate_condition_trace_v1(
        tuple(reversed(packets)),
        plan,
        seed=1001,
    )

    assert first == second
    assert first.content_sha256 == second.content_sha256
    assert all(event.attempt_timeline for event in first.events)


def test_late_ack_causes_one_retransmission_with_auditable_arrivals() -> None:
    wire = WireAccountingConfigV1(
        ack_timeout_seconds=0.15,
        max_retransmissions=1,
    )
    packet = _packet("message-0", transmitted_at=1.0)
    one_attempt = account_wire_bytes_v1(
        packet.encoded_bytes,
        wire,
        data_transmissions=1,
        ack_transmissions=1,
    )
    initial_timeout = packet.transmitted_at + wire.ack_timeout_seconds
    retry_timeout = initial_timeout + wire.ack_timeout_seconds
    initial = SyntheticTransmissionAttemptV1(
        attempt_index=0,
        data_sent_at=packet.transmitted_at,
        data_dropped=False,
        data_arrival_time=1.05,
        data_channel_state=None,
        ack_sent_at=1.05,
        ack_dropped=False,
        ack_arrival_time=1.20,
        ack_channel_state=None,
        ack_timeout_at=initial_timeout,
        ack_timed_out=True,
        retry_exhausted=False,
        data_on_wire_bytes=one_attempt.data_on_wire_bytes,
        ack_on_wire_bytes=wire.ack_frame_bytes,
    )
    retry = SyntheticTransmissionAttemptV1(
        attempt_index=1,
        data_sent_at=initial_timeout,
        data_dropped=False,
        data_arrival_time=1.16,
        data_channel_state=None,
        ack_sent_at=1.16,
        ack_dropped=False,
        ack_arrival_time=1.17,
        ack_channel_state=None,
        ack_timeout_at=retry_timeout,
        ack_timed_out=False,
        retry_exhausted=False,
        data_on_wire_bytes=one_attempt.data_on_wire_bytes,
        ack_on_wire_bytes=wire.ack_frame_bytes,
    )
    event = NetworkTraceEventV1(
        packet=packet,
        arrival_time=initial.data_arrival_time,
        dropped=False,
        channel_state=None,
        clock_offset_seconds=0.0,
        clock_drift_ppm=0.0,
        clock_error_at_transmit_seconds=0.0,
        translation_noise_metres=(0.0, 0.0, 0.0),
        yaw_noise_degrees=0.0,
        byte_account=account_wire_bytes_v1(
            packet.encoded_bytes,
            wire,
            data_transmissions=2,
            ack_transmissions=2,
        ),
        wire_accounting_config=wire,
        attempt_timeline=(initial, retry),
    )

    assert initial.ack_arrival_time > initial.ack_timeout_at
    assert initial.ack_timed_out
    assert retry.is_retransmission
    assert retry.data_sent_at == pytest.approx(initial.ack_timeout_at)
    assert retry.ack_arrival_time <= retry.ack_timeout_at
    assert not retry.ack_timed_out
    assert event.arrival_time == pytest.approx(1.05)
    assert event.byte_account.data_transmissions == 2
    assert event.byte_account.ack_transmissions == 2
    assert event.byte_account.total_on_wire_bytes == sum(
        attempt.data_on_wire_bytes + attempt.ack_on_wire_bytes
        for attempt in event.attempt_timeline
    )
    with pytest.raises(ValueError, match="ACK timeout flag"):
        replace(
            event,
            byte_account=account_wire_bytes_v1(
                packet.encoded_bytes,
                wire,
                data_transmissions=1,
                ack_transmissions=1,
            ),
            attempt_timeline=(replace(initial, ack_timed_out=False),),
        )


def test_ack_drop_and_final_timeout_are_explicit() -> None:
    wire = WireAccountingConfigV1(
        ack_timeout_seconds=0.2,
        max_retransmissions=1,
    )
    plan = NetworkConditionPlanV1(
        NetworkConditionId.C3,
        seeds=(1001,),
        loss_model=LossModel.IID,
        iid_loss_probability=0.4,
        wire_accounting=wire,
    )
    trace = generate_condition_trace_v1(
        (_packet("message-0"),),
        plan,
        seed=1001,
    )
    initial, retry = trace.events[0].attempt_timeline

    assert initial.data_dropped
    assert initial.ack_sent_at is None
    assert initial.ack_timed_out
    assert not retry.data_dropped
    assert retry.ack_sent_at == retry.data_arrival_time
    assert retry.ack_dropped
    assert retry.ack_arrival_time is None
    assert retry.ack_timed_out
    assert retry.retry_exhausted


def test_attempt_timeline_rejects_retransmission_undercount() -> None:
    wire = WireAccountingConfigV1(
        ack_timeout_seconds=0.15,
        max_retransmissions=1,
    )
    plan = NetworkConditionPlanV1(
        NetworkConditionId.C3,
        seeds=(1001,),
        loss_model=LossModel.IID,
        iid_loss_probability=1.0,
        wire_accounting=wire,
    )
    trace = generate_condition_trace_v1(
        (_packet("message-0", transmitted_at=1.0),),
        plan,
        seed=1001,
    )
    event = trace.events[0]
    understated = account_wire_bytes_v1(
        event.packet.encoded_bytes,
        wire,
        data_transmissions=1,
        ack_transmissions=0,
    )

    assert understated.total_on_wire_bytes < event.byte_account.total_on_wire_bytes
    with pytest.raises(ValueError, match="attempt timeline"):
        replace(event, byte_account=understated)


def test_bounded_reservation_keeps_capped_scheduler_budget_compliant() -> None:
    message = _message()
    wire = WireAccountingConfigV1(
        ack_timeout_seconds=0.15,
        max_retransmissions=1,
    )
    application_bytes = len(encode_message(message))
    trace = generate_condition_trace_v1(
        (
            PacketRequest(
                message_id=message.message_id,
                source=message.source,
                sequence=message.sequence,
                transmitted_at=message.timestamps.transmitted,
                deadline=message.deadline,
                encoded_bytes=application_bytes,
            ),
        ),
        condition_plan_v1("C3", wire_accounting=wire),
        seed=1001,
    )
    reservation = bounded_wire_reservation_v1(
        application_bytes,
        wire,
    ).total_on_wire_bytes
    candidate = ScheduleCandidate(
        message=message,
        network_transmit_time=message.timestamps.transmitted,
        score_evidence=CausalScoreEvidenceV1(
            as_of_network_time=message.timestamps.transmitted,
            risk_reduction=1.0,
            on_time_probability=1.0,
            confidence=1.0,
            age_seconds=0.0,
            tracker_state_sha256="a" * 64,
            channel_model_sha256="b" * 64,
            scorer_config_sha256="c" * 64,
            on_time_estimate_sha256="d" * 64,
            candidate_message_sha256=wire_digest(message),
            candidate_network_transmit_time=message.timestamps.transmitted,
            candidate_deadline=message.deadline,
            ground_truth_free=True,
        ),
        projected_on_wire_bytes=reservation,
    )

    result = schedule_rate_limited_v1(
        (candidate,),
        SchedulerKind.CONFIDENCE_TOP_K,
        network_trace=trace,
        wire_accounting=wire,
        byte_rate_limit_per_second=reservation,
    )

    assert result.message_ids == (message.message_id,)
    assert trace.on_wire_bytes <= reservation
    assert result.total_on_wire_bytes <= result.total_reserved_on_wire_bytes
    assert result.budget_violations == 0
    assert result.budget_compliant


def test_trace_rejects_plan_digest_and_event_wire_config_mismatch() -> None:
    trace = generate_condition_trace_v1(
        (_packet("message-0", transmitted_at=1.0),),
        condition_plan_v1("C1"),
        seed=1001,
    )
    with pytest.raises(ValueError, match="condition_plan_sha256"):
        replace(trace, condition_plan_sha256="f" * 64)

    mismatched_plan = condition_plan_v1(
        "C1",
        wire_accounting=WireAccountingConfigV1(ack_frame_bytes=65),
    )
    with pytest.raises(ValueError, match="wire accounting config"):
        NetworkTraceV1(
            condition_id=trace.condition_id,
            seed=trace.seed,
            condition_plan=mismatched_plan,
            condition_plan_sha256=mismatched_plan.content_sha256,
            events=trace.events,
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"ack_timeout_seconds": 0.0}, "ack_timeout_seconds"),
        (
            {"ack_on_success": False, "max_retransmissions": 1},
            "require ack_on_success",
        ),
        (
            {"ack_frame_bytes": 0, "max_retransmissions": 1},
            "non-zero ACK",
        ),
    ),
)
def test_invalid_retransmission_configuration_fails_closed(
    kwargs: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        WireAccountingConfigV1(**kwargs)


def test_network_trace_v1_strict_decoder_round_trip_reconstructs_full_ledger() -> None:
    trace = _ge_retransmission_trace()

    decoded = decode_network_trace_v1(trace.canonical_bytes)

    assert decoded == trace
    assert decoded.content_sha256 == trace.content_sha256
    assert decoded.condition_plan.gilbert_elliott is not None
    assert decoded.condition_plan.wire_accounting.max_retransmissions == 2
    assert decoded.events[0].packet == trace.events[0].packet
    assert decoded.events[0].byte_account == trace.events[0].byte_account
    assert decoded.events[0].attempt_timeline == trace.events[0].attempt_timeline


def test_network_trace_v1_strict_decoder_replays_ledger_invariants() -> None:
    document = _ge_retransmission_trace().to_dict()
    account = document["events"][0]["byte_account"]
    account["total_on_wire_bytes"] += 1

    with pytest.raises(NetworkTraceError, match="total_on_wire_bytes"):
        decode_network_trace_v1(_canonical_json(document))

    document = _ge_retransmission_trace().to_dict()
    attempt = document["events"][0]["attempt_timeline"][1]
    attempt["is_retransmission"] = False
    with pytest.raises(NetworkTraceError, match="is_retransmission"):
        decode_network_trace_v1(_canonical_json(document))


def test_network_trace_v1_strict_decoder_rejects_unknown_and_missing_fields() -> None:
    document = _ge_retransmission_trace().to_dict()
    del document["condition_plan"]["wire_accounting"]["ack_frame_bytes"]
    with pytest.raises(NetworkTraceError, match="missing=.*ack_frame_bytes"):
        decode_network_trace_v1(_canonical_json(document))

    document = _ge_retransmission_trace().to_dict()
    document["events"][0]["byte_account"]["unregistered_total"] = 0
    with pytest.raises(NetworkTraceError, match="unknown=.*unregistered_total"):
        decode_network_trace_v1(_canonical_json(document))


def test_network_trace_v1_strict_decoder_rejects_duplicate_json_keys() -> None:
    canonical = _ge_retransmission_trace().canonical_bytes
    duplicate = canonical.replace(
        b'{"condition_id":"C4"',
        b'{"condition_id":"C4","condition_id":"C4"',
        1,
    )
    assert duplicate != canonical

    with pytest.raises(NetworkTraceError, match="duplicate JSON key: condition_id"):
        decode_network_trace_v1(duplicate)


def test_network_trace_v1_strict_decoder_rejects_noncanonical_and_nonfinite() -> None:
    canonical = _ge_retransmission_trace().canonical_bytes
    with pytest.raises(NetworkTraceError, match="not canonical"):
        decode_network_trace_v1(canonical + b"\n")

    nonfinite = canonical.replace(
        b'"clock_drift_limit_ppm":0.0',
        b'"clock_drift_limit_ppm":NaN',
        1,
    )
    assert nonfinite != canonical
    with pytest.raises(NetworkTraceError, match="non-finite JSON constant"):
        decode_network_trace_v1(nonfinite)
