from dataclasses import replace
import json

import pytest

from transvision.models.event_track_v2x.measured_packet_trace import (
    MeasuredPacketRecordV1,
    MeasuredPacketTraceError,
    MeasuredPacketTraceV1,
    SynchronizationStatus,
    decode_measured_packet_trace_v1,
    validate_measured_packet_trace_segment_v1,
)
from transvision.models.event_track_v2x.measured_trace import (
    REQUIRED_PACKET_FIELDS_V1,
    ClockReference,
    MeasuredTraceSegmentV1,
    TraceRole,
)


def _record(
    monotonic_time: float,
    *,
    sequence_number: int,
    send_time: float | None = None,
    arrival_time: float | None,
    loss: bool,
    retransmission: bool = False,
    ack: bool = False,
    synchronization_status: SynchronizationStatus | str = (
        SynchronizationStatus.SYNCHRONIZED
    ),
) -> MeasuredPacketRecordV1:
    return MeasuredPacketRecordV1(
        monotonic_time=monotonic_time,
        wall_time="2026-09-02T00:00:01Z",
        sequence_number=sequence_number,
        send_time=monotonic_time if send_time is None else send_time,
        arrival_time=arrival_time,
        packet_bytes=64 if ack else 512,
        loss=loss,
        retransmission=retransmission,
        ack=ack,
        synchronization_status=synchronization_status,
    )


@pytest.fixture
def trace() -> MeasuredPacketTraceV1:
    records = (
        _record(
            1001.0,
            sequence_number=1,
            arrival_time=1001.1,
            loss=False,
        ),
        _record(
            1001.2,
            sequence_number=1,
            arrival_time=1001.25,
            loss=False,
            ack=True,
        ),
        _record(
            1002.0,
            sequence_number=2,
            arrival_time=None,
            loss=True,
        ),
        _record(
            1003.0,
            sequence_number=2,
            arrival_time=1003.1,
            loss=False,
            retransmission=True,
        ),
        _record(
            1003.2,
            sequence_number=2,
            arrival_time=1003.25,
            loss=False,
            ack=True,
        ),
    )
    return MeasuredPacketTraceV1(
        trace_id="setup-trace-00",
        collection_start_monotonic_time=1000.0,
        collection_end_monotonic_time=1600.0,
        clock_reference=ClockReference.PTP,
        records=records,
    )


def _segment(trace: MeasuredPacketTraceV1) -> MeasuredTraceSegmentV1:
    return MeasuredTraceSegmentV1(
        trace_id=trace.trace_id,
        role=TraceRole.SETUP,
        date_cohort_id="date-00",
        road_coverage_type="urban-covered",
        duration_seconds=600,
        packet_count=len(trace.records),
        packet_metadata_sha256=trace.content_sha256,
        packet_fields=REQUIRED_PACKET_FIELDS_V1,
        clock_reference=trace.clock_reference,
    )


def test_canonical_round_trip_and_exact_segment_binding(
    trace: MeasuredPacketTraceV1,
) -> None:
    assert decode_measured_packet_trace_v1(trace.canonical_bytes) == trace
    assert len(trace.content_sha256) == 64
    assert trace.supports_clock_claims
    assert tuple(trace.records[0].to_primitive()) == tuple(
        sorted(REQUIRED_PACKET_FIELDS_V1)
    )
    validate_measured_packet_trace_segment_v1(trace, _segment(trace))

    with pytest.raises(MeasuredPacketTraceError, match="does not match"):
        validate_measured_packet_trace_segment_v1(
            trace,
            replace(_segment(trace), packet_metadata_sha256="f" * 64),
        )


def test_no_clock_reference_cannot_support_clock_claims(
    trace: MeasuredPacketTraceV1,
) -> None:
    unsupported = replace(trace, clock_reference=ClockReference.NONE)
    assert not unsupported.supports_clock_claims
    segment = _segment(unsupported)
    assert not segment.supports_clock_claims
    validate_measured_packet_trace_segment_v1(unsupported, segment)


def test_unsynchronized_record_disables_clock_claims_and_status_is_closed(
    trace: MeasuredPacketTraceV1,
) -> None:
    records = (
        replace(
            trace.records[0],
            synchronization_status=SynchronizationStatus.UNSYNCHRONIZED,
        ),
        *trace.records[1:],
    )
    unsynchronized = replace(trace, records=records)

    assert unsynchronized.clock_reference is ClockReference.PTP
    assert not unsynchronized.supports_clock_claims
    assert (
        unsynchronized.to_primitive()["records"][0]["synchronization_status"]
        == "unsynchronized"
    )
    with pytest.raises(MeasuredPacketTraceError, match="does not match"):
        validate_measured_packet_trace_segment_v1(
            unsynchronized,
            _segment(unsynchronized),
        )
    with pytest.raises(MeasuredPacketTraceError, match="synchronization_status"):
        replace(trace.records[0], synchronization_status="holdover")


def test_strict_schema_rejects_privacy_fields_duplicates_nan_and_noncanonical_json(
    trace: MeasuredPacketTraceV1,
) -> None:
    record = trace.records[0].to_primitive()
    record["device_id"] = "secret-device"
    with pytest.raises(MeasuredPacketTraceError, match="unknown"):
        MeasuredPacketRecordV1.from_mapping(record)

    document = trace.to_primitive()
    document["precise_coordinates"] = [1.0, 2.0]
    with pytest.raises(MeasuredPacketTraceError, match="unknown"):
        MeasuredPacketTraceV1.from_mapping(document)

    with pytest.raises(MeasuredPacketTraceError, match="duplicate JSON key"):
        decode_measured_packet_trace_v1(b'{"kind":"x","kind":"y"}')
    with pytest.raises(MeasuredPacketTraceError, match="not canonical"):
        decode_measured_packet_trace_v1(json.dumps(trace.to_primitive()).encode())
    nan_document = trace.canonical_bytes.replace(b"1000.0", b"NaN", 1)
    with pytest.raises(MeasuredPacketTraceError, match="non-finite"):
        decode_measured_packet_trace_v1(nan_document)


def test_window_order_bytes_and_loss_arrival_semantics_fail_closed(
    trace: MeasuredPacketTraceV1,
) -> None:
    with pytest.raises(MeasuredPacketTraceError, match="exactly 600"):
        replace(trace, collection_end_monotonic_time=1599.0)
    with pytest.raises(MeasuredPacketTraceError, match="stable tie-break"):
        replace(trace, records=tuple(reversed(trace.records)))
    with pytest.raises(MeasuredPacketTraceError, match="positive integer"):
        replace(trace.records[0], packet_bytes=0)
    with pytest.raises(MeasuredPacketTraceError, match="loss must be true"):
        replace(trace.records[0], loss=True)
    with pytest.raises(MeasuredPacketTraceError, match="inside the 600-second"):
        outside = replace(trace.records[0], monotonic_time=999.0, send_time=999.0)
        replace(trace, records=(outside, *trace.records[1:]))

    same_time_first = _record(
        1010.0,
        sequence_number=1,
        arrival_time=1010.1,
        loss=False,
    )
    same_time_second = _record(
        1010.0,
        sequence_number=2,
        arrival_time=1010.1,
        loss=False,
    )
    with pytest.raises(MeasuredPacketTraceError, match="stable tie-break"):
        replace(trace, records=(same_time_second, same_time_first))


def test_retransmission_and_ack_protocol_semantics_fail_closed(
    trace: MeasuredPacketTraceV1,
) -> None:
    first_marked_retry = _record(
        1010.0,
        sequence_number=10,
        arrival_time=1010.1,
        loss=False,
        retransmission=True,
    )
    with pytest.raises(MeasuredPacketTraceError, match="first data attempt"):
        replace(trace, records=(first_marked_retry,))

    orphan_ack = _record(
        1010.0,
        sequence_number=10,
        arrival_time=1010.1,
        loss=False,
        ack=True,
    )
    with pytest.raises(MeasuredPacketTraceError, match="previously delivered"):
        replace(trace, records=(orphan_ack,))

    retransmit_after_ack = _record(
        1001.3,
        sequence_number=1,
        arrival_time=1001.4,
        loss=False,
        retransmission=True,
    )
    changed = (*trace.records[:2], retransmit_after_ack, *trace.records[2:])
    with pytest.raises(MeasuredPacketTraceError, match="after a delivered ACK"):
        replace(trace, records=changed)


def test_protocol_relations_use_physical_send_and_arrival_times(
    trace: MeasuredPacketTraceV1,
) -> None:
    # Record iteration order may differ from physical packet-event order.  This
    # ACK is listed first by monotonic_time but is physically sent only after
    # the data arrival, so it remains a valid pair.
    ack_logged_first = _record(
        1010.0,
        sequence_number=20,
        send_time=1011.1,
        arrival_time=1011.2,
        loss=False,
        ack=True,
    )
    data_logged_second = _record(
        1011.0,
        sequence_number=20,
        send_time=1011.0,
        arrival_time=1011.1,
        loss=False,
    )
    physically_valid = replace(
        trace,
        records=(ack_logged_first, data_logged_second),
    )
    assert len(physically_valid.records) == 2

    # Conversely, a delivered ACK physically precedes the retry even though
    # its record is listed later.  Iteration-order validation used to accept
    # this impossible retransmission.
    first = _record(
        1020.0,
        sequence_number=21,
        send_time=1020.0,
        arrival_time=1020.1,
        loss=False,
    )
    retry = _record(
        1021.0,
        sequence_number=21,
        send_time=1022.0,
        arrival_time=1022.1,
        loss=False,
        retransmission=True,
    )
    ack_logged_last = _record(
        1022.0,
        sequence_number=21,
        send_time=1020.1,
        arrival_time=1020.2,
        loss=False,
        ack=True,
    )
    with pytest.raises(MeasuredPacketTraceError, match="after a delivered ACK"):
        replace(trace, records=(first, retry, ack_logged_last))

    simultaneous_attempt = replace(
        retry,
        monotonic_time=1020.5,
        send_time=first.send_time,
    )
    with pytest.raises(MeasuredPacketTraceError, match="unique send_time"):
        replace(trace, records=(first, simultaneous_attempt))
