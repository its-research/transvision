from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from transvision.models.event_track_v2x.contracts import DetectionCacheV1
from transvision.models.event_track_v2x.network import (
    NetworkConditionId,
    NetworkTraceEventV1,
    NetworkTraceV1,
    PacketRequest,
    WireAccountingConfigV1,
    account_wire_bytes_v1,
    condition_plan_v1,
)
from transvision.models.event_track_v2x.runner import (
    ReferenceSequenceRunnerError,
    ReferenceSequenceRunnerV1,
)
from transvision.models.event_track_v2x.tracker import ReferenceMultiTargetTracker
from transvision.models.event_track_v2x.wire import canonical_json_bytes


def _cache(
    frame: int,
    *,
    source: str = "roadside",
    sequence_id: str = "sequence-001",
    event_time: float = 0.0,
) -> DetectionCacheV1:
    return DetectionCacheV1(
        sequence_id=sequence_id,
        frame_id=f"frame-{frame:06d}",
        event_time=event_time,
        agent_id=source,
        coordinate_frame="world",
        boxes_3d=np.asarray([[float(frame), 0.0, 0.5, 4.0, 2.0, 1.5, 0.0]]),
        scores=np.asarray([0.9]),
        class_labels=("car",),
        velocities=np.asarray([[1.0, 0.0]]),
        covariances=np.asarray([np.eye(9)]),
        dataset_sha256="a" * 64,
        detector_config_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
    )


def _event(
    cache: DetectionCacheV1,
    *,
    message_id: str,
    packet_sequence: int,
    transmitted_at: float,
    arrival_time: float | None,
    clock_offset_seconds: float = 0.0,
    clock_drift_ppm: float = 0.0,
    clock_error_at_transmit_seconds: float = 0.0,
) -> NetworkTraceEventV1:
    encoded_bytes = len(canonical_json_bytes(cache.to_primitive()))
    dropped = arrival_time is None
    packet = PacketRequest(
        message_id=message_id,
        source=cache.agent_id,
        sequence=packet_sequence,
        transmitted_at=transmitted_at,
        deadline=1.0,
        encoded_bytes=encoded_bytes,
    )
    return NetworkTraceEventV1(
        packet=packet,
        arrival_time=arrival_time,
        dropped=dropped,
        channel_state=None,
        clock_offset_seconds=clock_offset_seconds,
        clock_drift_ppm=clock_drift_ppm,
        clock_error_at_transmit_seconds=clock_error_at_transmit_seconds,
        translation_noise_metres=(0.0, 0.0, 0.0),
        yaw_noise_degrees=0.0,
        byte_account=account_wire_bytes_v1(
            encoded_bytes,
            WireAccountingConfigV1(),
            ack_transmissions=0 if dropped else 1,
        ),
    )


def _trace(
    condition: str,
    events: tuple[NetworkTraceEventV1, ...],
) -> NetworkTraceV1:
    plan = condition_plan_v1(condition)
    return NetworkTraceV1(
        condition_id=condition,
        seed=1001,
        condition_plan=plan,
        condition_plan_sha256=plan.content_sha256,
        events=events,
    )


def _single_prediction_result():
    cache = _cache(1, event_time=0.05)
    trace = _trace(
        "C0",
        (
            _event(
                cache,
                message_id="message-validation",
                packet_sequence=1,
                transmitted_at=0.0,
                arrival_time=0.1,
            ),
        ),
    )
    return ReferenceSequenceRunnerV1(ReferenceMultiTargetTracker()).run(
        caches_by_message_id={"message-validation": cache},
        network_trace=trace,
        run_config_sha256="d" * 64,
        decision_times=(0.2,),
    )


def test_c0_canary_commits_every_decision_and_skips_drops() -> None:
    delivered = _cache(1, event_time=0.05)
    dropped = _cache(2, event_time=0.10)
    late = _cache(3, event_time=0.15)
    trace = _trace(
        "C0",
        (
            _event(
                delivered,
                message_id="message-001",
                packet_sequence=1,
                transmitted_at=0.0,
                arrival_time=0.1,
            ),
            _event(
                dropped,
                message_id="message-002",
                packet_sequence=2,
                transmitted_at=0.1,
                arrival_time=None,
            ),
            _event(
                late,
                message_id="message-003",
                packet_sequence=3,
                transmitted_at=0.2,
                arrival_time=0.5,
            ),
        ),
    )
    tracker = ReferenceMultiTargetTracker()
    result = ReferenceSequenceRunnerV1(tracker).run(
        caches_by_message_id={
            "message-001": delivered,
            "message-002": dropped,
            "message-003": late,
        },
        network_trace=trace,
        run_config_sha256="d" * 64,
        decision_times=(0.2, 0.4),
    )

    assert result.condition_id is NetworkConditionId.C0
    assert result.processed_message_ids == ("message-001",)
    assert result.dropped_message_ids == ("message-002",)
    assert result.late_message_ids == ("message-003",)
    assert result.condition_input_manifest is None
    assert len(tracker.commit_records) == 2
    assert result.predictions and all(item.committed for item in result.predictions)
    rerun = ReferenceSequenceRunnerV1(ReferenceMultiTargetTracker()).run(
        caches_by_message_id={
            "message-001": delivered,
            "message-002": dropped,
            "message-003": late,
        },
        network_trace=trace,
        run_config_sha256="d" * 64,
        decision_times=(0.2, 0.4),
    )
    assert result.content_sha256 == rerun.content_sha256


def test_result_predictions_are_bound_to_sequence_schedule_and_unique_key() -> None:
    result = _single_prediction_result()
    prediction = result.predictions[0]

    with pytest.raises(ReferenceSequenceRunnerError, match="sequence"):
        replace(
            result,
            predictions=(replace(prediction, sequence_id="sequence-wrong"),),
        )
    with pytest.raises(ReferenceSequenceRunnerError, match="outside.*schedule"):
        replace(
            result,
            predictions=(replace(prediction, decision_time=0.3),),
        )
    with pytest.raises(ReferenceSequenceRunnerError, match="duplicate"):
        replace(result, predictions=(prediction, prediction))


def test_out_of_order_arrival_is_processed_only_when_causally_available() -> None:
    first_sent = _cache(1, event_time=0.05)
    second_sent = _cache(2, event_time=0.10)
    trace = _trace(
        "C2",
        (
            _event(
                first_sent,
                message_id="message-first",
                packet_sequence=1,
                transmitted_at=0.0,
                arrival_time=0.30,
            ),
            _event(
                second_sent,
                message_id="message-second",
                packet_sequence=2,
                transmitted_at=0.1,
                arrival_time=0.20,
            ),
        ),
    )
    result = ReferenceSequenceRunnerV1(ReferenceMultiTargetTracker()).run(
        caches_by_message_id={
            "message-first": first_sent,
            "message-second": second_sent,
        },
        network_trace=trace,
        run_config_sha256="d" * 64,
        decision_times=(0.25, 0.35),
    )

    assert result.processed_message_ids == (
        "message-second",
        "message-first",
    )
    assert result.condition_input_manifest is None


def test_duplicate_cache_cannot_be_counted_as_two_messages() -> None:
    cache = _cache(1, event_time=0.05)
    trace = _trace(
        "C0",
        (
            _event(
                cache,
                message_id="message-001",
                packet_sequence=1,
                transmitted_at=0.0,
                arrival_time=0.1,
            ),
            _event(
                cache,
                message_id="message-002",
                packet_sequence=2,
                transmitted_at=0.1,
                arrival_time=0.2,
            ),
        ),
    )
    with pytest.raises(ReferenceSequenceRunnerError, match="double-count"):
        ReferenceSequenceRunnerV1(ReferenceMultiTargetTracker()).run(
            caches_by_message_id={"message-001": cache, "message-002": cache},
            network_trace=trace,
            run_config_sha256="d" * 64,
            decision_times=(0.3,),
        )


def test_c6_realizes_event_time_and_emits_input_manifest() -> None:
    cache = _cache(1, event_time=0.10)
    event = _event(
        cache,
        message_id="message-c6",
        packet_sequence=1,
        transmitted_at=0.0,
        arrival_time=0.20,
        clock_offset_seconds=0.04,
        clock_drift_ppm=25.0,
        clock_error_at_transmit_seconds=0.05,
    )
    trace = _trace("C6", (event,))
    result = ReferenceSequenceRunnerV1(ReferenceMultiTargetTracker()).run(
        caches_by_message_id={"message-c6": cache},
        network_trace=trace,
        run_config_sha256="d" * 64,
        decision_times=(0.30,),
    )

    assert result.predictions[0].event_time == pytest.approx(0.15)
    assert result.condition_input_manifest is not None
    assert result.condition_input_manifest.condition_id is NetworkConditionId.C6
    assert result.condition_input_manifest.network_trace_sha256 == trace.content_sha256
    assert len(result.condition_input_manifest.evidence_sha256s) == 1

    future = replace(event, clock_error_at_transmit_seconds=0.15)
    future_trace = _trace("C6", (future,))
    with pytest.raises(ValueError, match="future input rejected"):
        ReferenceSequenceRunnerV1(ReferenceMultiTargetTracker()).run(
            caches_by_message_id={"message-c6": cache},
            network_trace=future_trace,
            run_config_sha256="d" * 64,
            decision_times=(0.30,),
        )


def test_simultaneous_arrivals_use_reference_batch_ingestion() -> None:
    roadside_a = _cache(1, source="roadside-a", event_time=0.05)
    roadside_b = _cache(2, source="roadside-b", event_time=0.05)
    trace = _trace(
        "C0",
        (
            _event(
                roadside_b,
                message_id="message-b",
                packet_sequence=1,
                transmitted_at=0.0,
                arrival_time=0.2,
            ),
            _event(
                roadside_a,
                message_id="message-a",
                packet_sequence=1,
                transmitted_at=0.0,
                arrival_time=0.2,
            ),
        ),
    )
    class RecordingTracker(ReferenceMultiTargetTracker):
        def __init__(self) -> None:
            super().__init__()
            self.batch_calls = 0

        def ingest_batch(self, batch):
            self.batch_calls += 1
            return super().ingest_batch(batch)

    tracker = RecordingTracker()
    result = ReferenceSequenceRunnerV1(tracker).run(
        caches_by_message_id={"message-a": roadside_a, "message-b": roadside_b},
        network_trace=trace,
        run_config_sha256="d" * 64,
        decision_times=(0.3,),
    )

    assert result.processed_message_ids == ("message-a", "message-b")
    assert tracker.batch_calls == 1


def test_c9_is_explicitly_outside_the_reference_runner() -> None:
    with pytest.raises(ReferenceSequenceRunnerError, match="C9 measured traces"):
        ReferenceSequenceRunnerV1(ReferenceMultiTargetTracker()).run(
            caches_by_message_id={"message": _cache(1)},
            network_trace=SimpleNamespace(condition_id="C9"),
            run_config_sha256="d" * 64,
            decision_times=(0.3,),
        )
