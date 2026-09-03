from dataclasses import replace

import pytest

from transvision.models.event_track_v2x.measured_trace import (
    ClockReference,
    MeasuredTraceError,
    MeasuredTraceReceiptV1,
    MeasuredTraceSegmentV1,
    REQUIRED_PACKET_FIELDS_V1,
    TraceRole,
    decode_measured_trace_receipt,
)


def _segment(index: int) -> MeasuredTraceSegmentV1:
    return MeasuredTraceSegmentV1(
        trace_id=f"trace-{index:02d}",
        role=TraceRole.SETUP if index < 10 else TraceRole.HELD_OUT,
        date_cohort_id=f"date-{index % 3}",
        road_coverage_type=f"coverage-{index % 3}",
        duration_seconds=600,
        packet_count=100 + index,
        packet_metadata_sha256=f"{index + 1:064x}",
        packet_fields=REQUIRED_PACKET_FIELDS_V1,
        clock_reference=ClockReference.PTP,
    )


def _receipt() -> MeasuredTraceReceiptV1:
    return MeasuredTraceReceiptV1(
        receipt_id="c9-heldout-v1",
        segments=tuple(_segment(index) for index in range(30)),
        collection_protocol_sha256="a" * 64,
    )


def test_measured_trace_receipt_round_trip_and_exact_split() -> None:
    receipt = _receipt()
    assert decode_measured_trace_receipt(receipt.canonical_bytes) == receipt
    assert len(receipt.setup_trace_ids) == 10
    assert len(receipt.held_out_trace_ids) == 20
    assert receipt.held_out_packet_metadata_sha256s == tuple(
        (f"trace-{index:02d}", f"{index + 1:064x}")
        for index in range(10, 30)
    )
    assert receipt.supports_clock_claims
    assert receipt.content_sha256 == _receipt().content_sha256


def test_measured_trace_receipt_rejects_privacy_and_coverage_failures() -> None:
    with pytest.raises(MeasuredTraceError, match="must not contain"):
        replace(_segment(0), contains_precise_coordinates=True)

    segments = list(_receipt().segments)
    segments[10] = replace(segments[10], role=TraceRole.SETUP)
    with pytest.raises(MeasuredTraceError, match="10 setup and 20"):
        replace(_receipt(), segments=tuple(segments))

    one_date = tuple(replace(item, date_cohort_id="date-0") for item in _receipt().segments)
    with pytest.raises(MeasuredTraceError, match="three date"):
        replace(_receipt(), segments=one_date)


def test_measured_trace_receipt_rejects_missing_packet_field_and_noncanonical_json() -> None:
    with pytest.raises(MeasuredTraceError, match="packet_fields"):
        replace(_segment(0), packet_fields=REQUIRED_PACKET_FIELDS_V1[:-1])
    noncanonical = b'{"kind":"measured_trace_receipt_v1"}'
    with pytest.raises(MeasuredTraceError):
        decode_measured_trace_receipt(noncanonical)


def test_clock_claims_require_a_reference_on_every_segment() -> None:
    segments = list(_receipt().segments)
    segments[0] = replace(segments[0], clock_reference=ClockReference.NONE)
    receipt = replace(_receipt(), segments=tuple(segments))
    assert not receipt.supports_clock_claims
