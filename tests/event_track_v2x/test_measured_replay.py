from dataclasses import replace
import json

import numpy as np
import pytest

from transvision.models.event_track_v2x.contracts import DetectionCacheV1
from transvision.models.event_track_v2x.measured_packet_trace import (
    MeasuredPacketRecordV1,
    MeasuredPacketTraceV1,
    SynchronizationStatus,
)
from transvision.models.event_track_v2x.measured_replay import (
    C9ApplicationMessageBindingV1,
    C9MeasuredReplayError,
    C9ReplayCapability,
    application_network_events_v1,
    build_c9_measured_replay_v1,
    decode_c9_measured_replay_v1,
    validate_c9_measured_replay_v1,
)
from transvision.models.event_track_v2x.measured_trace import (
    REQUIRED_PACKET_FIELDS_V1,
    ClockReference,
    MeasuredTraceReceiptV1,
    MeasuredTraceSegmentV1,
    TraceRole,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes


def _record(
    monotonic_time: float,
    *,
    sequence_number: int,
    arrival_time: float | None,
    packet_bytes: int,
    retransmission: bool = False,
    ack: bool = False,
) -> MeasuredPacketRecordV1:
    return MeasuredPacketRecordV1(
        monotonic_time=monotonic_time,
        wall_time="2026-09-02T00:00:01Z",
        sequence_number=sequence_number,
        send_time=monotonic_time,
        arrival_time=arrival_time,
        packet_bytes=packet_bytes,
        loss=arrival_time is None,
        retransmission=retransmission,
        ack=ack,
        synchronization_status=SynchronizationStatus.SYNCHRONIZED,
    )


@pytest.fixture
def packet_trace() -> MeasuredPacketTraceV1:
    return MeasuredPacketTraceV1(
        trace_id="trace-10",
        collection_start_monotonic_time=1000.0,
        collection_end_monotonic_time=1600.0,
        clock_reference=ClockReference.PTP,
        records=(
            _record(
                1001.0,
                sequence_number=1,
                arrival_time=1001.1,
                packet_bytes=2048,
            ),
            _record(
                1001.2,
                sequence_number=1,
                arrival_time=1001.25,
                packet_bytes=64,
                ack=True,
            ),
            _record(
                1002.0,
                sequence_number=2,
                arrival_time=None,
                packet_bytes=2048,
            ),
            _record(
                1003.0,
                sequence_number=2,
                arrival_time=1003.1,
                packet_bytes=2048,
                retransmission=True,
            ),
            _record(
                1003.2,
                sequence_number=2,
                arrival_time=1003.25,
                packet_bytes=64,
                ack=True,
            ),
        ),
    )


def _segment(index: int, packet_trace: MeasuredPacketTraceV1) -> MeasuredTraceSegmentV1:
    is_target = index == 10
    return MeasuredTraceSegmentV1(
        trace_id=f"trace-{index:02d}",
        role=TraceRole.SETUP if index < 10 else TraceRole.HELD_OUT,
        date_cohort_id=f"date-{index % 3}",
        road_coverage_type=f"coverage-{index % 3}",
        duration_seconds=600,
        packet_count=len(packet_trace.records) if is_target else 100 + index,
        packet_metadata_sha256=(
            packet_trace.content_sha256 if is_target else f"{index + 1:064x}"
        ),
        packet_fields=REQUIRED_PACKET_FIELDS_V1,
        clock_reference=packet_trace.clock_reference if is_target else ClockReference.PTP,
    )


@pytest.fixture
def receipt(packet_trace: MeasuredPacketTraceV1) -> MeasuredTraceReceiptV1:
    return MeasuredTraceReceiptV1(
        receipt_id="c9-receipt-v1",
        segments=tuple(_segment(index, packet_trace) for index in range(30)),
        collection_protocol_sha256="a" * 64,
    )


def _cache(*, event_time: float = 1000.9, frame: int = 1) -> DetectionCacheV1:
    return DetectionCacheV1(
        sequence_id="sequence-001",
        frame_id=f"frame-{frame:06d}",
        event_time=event_time,
        agent_id="roadside",
        coordinate_frame="world",
        boxes_3d=np.asarray([[0.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.0]]),
        scores=np.asarray([0.9]),
        class_labels=("car",),
        velocities=np.asarray([[1.0, 0.0]]),
        covariances=np.asarray([np.eye(9)]),
        dataset_sha256="b" * 64,
        detector_config_sha256="c" * 64,
        checkpoint_sha256="d" * 64,
    )


def _binding(
    cache: DetectionCacheV1,
    *,
    message_id: str = "message-001",
    sequences: tuple[int, ...] = (1, 2),
    source: str | None = None,
) -> C9ApplicationMessageBindingV1:
    payload = canonical_json_bytes(cache.to_primitive())
    return C9ApplicationMessageBindingV1(
        message_id=message_id,
        source=cache.agent_id if source is None else source,
        sequence_numbers=sequences,
        detection_cache_sha256=cache.digest(),
        payload_sha256=cache.digest(),
        payload_bytes=len(payload),
        deadline=1004.0,
    )


def test_timing_only_is_canonical_and_cannot_invent_application_messages(
    packet_trace: MeasuredPacketTraceV1,
    receipt: MeasuredTraceReceiptV1,
) -> None:
    replay = build_c9_measured_replay_v1(packet_trace, receipt)

    assert replay.capability is C9ReplayCapability.NETWORK_TIMING_ONLY
    assert replay.application_messages == ()
    assert decode_c9_measured_replay_v1(replay.canonical_bytes) == replay
    assert replay.packet_trace_sha256 == packet_trace.content_sha256
    assert replay.receipt_sha256 == receipt.content_sha256
    assert replay.data_on_wire_bytes == 2048 * 3
    assert replay.ack_on_wire_bytes == 64 * 2
    validate_c9_measured_replay_v1(replay, packet_trace, receipt)

    with pytest.raises(C9MeasuredReplayError, match="cannot produce"):
        application_network_events_v1(replay)
    with pytest.raises(C9MeasuredReplayError, match="both bindings"):
        build_c9_measured_replay_v1(
            packet_trace,
            receipt,
            bindings=(_binding(_cache()),),
        )


def test_payload_binding_deduplicates_ack_and_retransmission(
    packet_trace: MeasuredPacketTraceV1,
    receipt: MeasuredTraceReceiptV1,
) -> None:
    cache = _cache()
    replay = build_c9_measured_replay_v1(
        packet_trace,
        receipt,
        bindings=(_binding(cache),),
        caches_by_message_id={"message-001": cache},
    )

    assert replay.capability is C9ReplayCapability.APPLICATION_MESSAGES
    assert len(replay.application_messages) == 1
    message = replay.application_messages[0]
    assert message.arrival_time == 1003.1
    assert not message.dropped
    assert replay.application_bytes == len(canonical_json_bytes(cache.to_primitive()))
    assert replay.sequence_outcomes[1].data_attempts == 2
    assert replay.sequence_outcomes[1].retransmissions == 1
    assert replay.sequence_outcomes[1].ack_transmissions == 1

    events = application_network_events_v1(replay)
    assert len(events) == 1
    assert events[0].packet.message_id == "message-001"
    assert events[0].arrival_time == 1003.1
    validate_c9_measured_replay_v1(
        replay,
        packet_trace,
        receipt,
        caches_by_message_id={"message-001": cache},
    )


def test_application_mapping_is_exact_causal_and_cache_bound(
    packet_trace: MeasuredPacketTraceV1,
    receipt: MeasuredTraceReceiptV1,
) -> None:
    cache = _cache()
    with pytest.raises(C9MeasuredReplayError, match="cover every"):
        build_c9_measured_replay_v1(
            packet_trace,
            receipt,
            bindings=(_binding(cache, sequences=(1,)),),
            caches_by_message_id={"message-001": cache},
        )

    with pytest.raises(C9MeasuredReplayError, match="source"):
        build_c9_measured_replay_v1(
            packet_trace,
            receipt,
            bindings=(_binding(cache, source="vehicle"),),
            caches_by_message_id={"message-001": cache},
        )

    future = _cache(event_time=1001.01)
    with pytest.raises(C9MeasuredReplayError, match="future"):
        build_c9_measured_replay_v1(
            packet_trace,
            receipt,
            bindings=(_binding(future),),
            caches_by_message_id={"message-001": future},
        )

    wrong_hash = replace(_binding(cache), detection_cache_sha256="e" * 64, payload_sha256="e" * 64)
    with pytest.raises(C9MeasuredReplayError, match="exact detection cache"):
        build_c9_measured_replay_v1(
            packet_trace,
            receipt,
            bindings=(wrong_hash,),
            caches_by_message_id={"message-001": cache},
        )


def test_replay_cold_read_and_external_reproduction_reject_tampering(
    packet_trace: MeasuredPacketTraceV1,
    receipt: MeasuredTraceReceiptV1,
) -> None:
    cache = _cache()
    replay = build_c9_measured_replay_v1(
        packet_trace,
        receipt,
        bindings=(_binding(cache),),
        caches_by_message_id={"message-001": cache},
    )
    document = replay.to_primitive()
    document["application_messages"][0]["on_time"] = False
    with pytest.raises(C9MeasuredReplayError, match="on_time"):
        decode_c9_measured_replay_v1(canonical_json_bytes(document))
    with pytest.raises(C9MeasuredReplayError, match="not canonical"):
        decode_c9_measured_replay_v1(json.dumps(replay.to_primitive()).encode())

    changed_trace = replace(
        packet_trace,
        records=(
            replace(packet_trace.records[0], packet_bytes=2049),
            *packet_trace.records[1:],
        ),
    )
    with pytest.raises(C9MeasuredReplayError, match="receipt segment"):
        validate_c9_measured_replay_v1(
            replay,
            changed_trace,
            receipt,
            caches_by_message_id={"message-001": cache},
        )


def test_retransmission_size_change_is_ambiguous_and_rejected(
    packet_trace: MeasuredPacketTraceV1,
    receipt: MeasuredTraceReceiptV1,
) -> None:
    changed = replace(
        packet_trace,
        records=(
            *packet_trace.records[:3],
            replace(packet_trace.records[3], packet_bytes=2047),
            packet_trace.records[4],
        ),
    )
    segments = tuple(
        replace(
            item,
            packet_metadata_sha256=changed.content_sha256,
            packet_count=len(changed.records),
        )
        if item.trace_id == changed.trace_id
        else item
        for item in receipt.segments
    )
    changed_receipt = replace(receipt, segments=segments)
    with pytest.raises(C9MeasuredReplayError, match="packet size changed"):
        build_c9_measured_replay_v1(changed, changed_receipt)
