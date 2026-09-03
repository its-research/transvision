from dataclasses import replace
import json

import numpy as np
import pytest

from transvision.models.event_track_v2x.contracts import DetectionCacheV1
from transvision.models.event_track_v2x.network import (
    NetworkConditionId,
    NetworkTraceEventV1,
    NetworkTraceV1,
    PacketRequest,
    account_wire_bytes_v1,
    condition_plan_v1,
)
from transvision.models.event_track_v2x.network_disturbance import (
    REALIZATION_CONTRACT_SHA256_V1,
    ConditionInputManifestV1,
    build_condition_input_manifest_v1,
    decode_condition_input_manifest_v1,
    realize_condition_input_v1,
    require_causal_at_arrival_v1,
)


def _cache(*, agent_id: str = "agent-0", event_time: float = 10.0) -> DetectionCacheV1:
    boxes = np.asarray(
        [
            [1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.25],
            [-2.0, 0.5, 1.0, 3.0, 1.5, 1.2, np.pi - 0.1],
        ],
        dtype=np.float64,
    )
    covariances = np.stack(
        (
            np.diag([1.0, 4.0, 2.0, 0.2, 0.3, 0.4, 0.1, 9.0, 16.0]),
            np.diag([2.0, 5.0, 3.0, 0.3, 0.4, 0.5, 0.2, 10.0, 17.0]),
        )
    )
    return DetectionCacheV1(
        sequence_id="sequence-001",
        frame_id="frame-000100",
        event_time=event_time,
        agent_id=agent_id,
        coordinate_frame="world",
        boxes_3d=boxes,
        scores=np.asarray([0.95, 0.80]),
        class_labels=("car", "car"),
        velocities=np.asarray([[3.0, 4.0], [-1.0, 2.0]]),
        covariances=covariances,
        dataset_sha256="a" * 64,
        detector_config_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
    )


def _event_and_trace(
    condition_id: NetworkConditionId | str,
    *,
    source: str = "agent-0",
    message_id: str = "detection-message-1",
    transmitted_at: float = 9.9,
    arrival_time: float | None = 10.5,
    dropped: bool = False,
    clock_offset_seconds: float = 0.0,
    clock_drift_ppm: float = 0.0,
    clock_error_at_transmit_seconds: float = 0.0,
    translation_noise_metres: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_noise_degrees: float = 0.0,
) -> tuple[NetworkTraceEventV1, NetworkTraceV1]:
    condition = NetworkConditionId(condition_id)
    plan = condition_plan_v1(condition)
    wire = plan.wire_accounting
    packet = PacketRequest(
        message_id=message_id,
        source=source,
        sequence=7,
        transmitted_at=transmitted_at,
        deadline=11.0,
        encoded_bytes=100,
    )
    account = account_wire_bytes_v1(
        100,
        wire,
        ack_transmissions=0 if dropped else 1,
    )
    event = NetworkTraceEventV1(
        packet=packet,
        arrival_time=arrival_time,
        dropped=dropped,
        channel_state=None,
        clock_offset_seconds=clock_offset_seconds,
        clock_drift_ppm=clock_drift_ppm,
        clock_error_at_transmit_seconds=clock_error_at_transmit_seconds,
        translation_noise_metres=translation_noise_metres,
        yaw_noise_degrees=yaw_noise_degrees,
        byte_account=account,
        wire_accounting_config=wire,
    )
    trace = NetworkTraceV1(
        condition_id=condition,
        seed=1001,
        condition_plan=plan,
        condition_plan_sha256=plan.content_sha256,
        events=(event,),
    )
    return event, trace


def test_c0_is_exact_identity_deterministic_and_keeps_source_cache_immutable() -> None:
    cache = _cache()
    source_snapshot = cache.to_primitive()
    event, trace = _event_and_trace("C0")

    first = realize_condition_input_v1(cache, trace, event)
    second = realize_condition_input_v1(cache, trace, event)

    assert first.detection_cache is cache
    assert first.detection_cache.digest() == cache.digest()
    assert first.content_sha256 == second.content_sha256
    assert first.canonical_bytes == second.canonical_bytes
    assert first.evidence.original_cache_sha256 == cache.digest()
    assert first.evidence.network_trace_sha256 == trace.content_sha256
    assert first.evidence.message_id == event.packet.message_id
    assert first.evidence.realization_contract_sha256 == REALIZATION_CONTRACT_SHA256_V1
    assert first.evidence.ground_truth_free
    assert require_causal_at_arrival_v1(first) is cache
    assert cache.to_primitive() == source_snapshot
    with pytest.raises(ValueError):
        cache.boxes_3d[0, 0] = 99.0


def test_c6_applies_only_recorded_clock_error_to_tracker_event_time() -> None:
    cache = _cache()
    event, trace = _event_and_trace(
        "C6",
        clock_offset_seconds=0.04,
        clock_drift_ppm=25.0,
        clock_error_at_transmit_seconds=0.05,
    )

    realized = realize_condition_input_v1(cache, trace, event)

    assert realized.detection_cache.event_time == 10.05
    np.testing.assert_array_equal(realized.detection_cache.boxes_3d, cache.boxes_3d)
    np.testing.assert_array_equal(realized.detection_cache.velocities, cache.velocities)
    np.testing.assert_array_equal(realized.detection_cache.covariances, cache.covariances)
    assert realized.evidence.clock_error_at_transmit_seconds == 0.05
    assert realized.detection_cache.digest() != cache.digest()


def test_c7_applies_se2_to_centres_yaw_velocity_and_covariance() -> None:
    cache = _cache()
    translation = (3.0, -1.0, 0.5)
    event, trace = _event_and_trace(
        "C7",
        translation_noise_metres=translation,
        yaw_noise_degrees=90.0,
    )

    realized = realize_condition_input_v1(cache, trace, event)
    output = realized.detection_cache
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    expected_boxes = np.array(cache.boxes_3d, copy=True)
    expected_boxes[:, :2] = cache.boxes_3d[:, :2] @ rotation.T + np.asarray(
        translation[:2]
    )
    expected_boxes[:, 2] += translation[2]
    expected_boxes[:, 6] = (
        cache.boxes_3d[:, 6] + np.pi / 2.0 + np.pi
    ) % (2.0 * np.pi) - np.pi
    expected_velocities = cache.velocities @ rotation.T
    jacobian = np.eye(9)
    jacobian[np.ix_((0, 1), (0, 1))] = rotation
    jacobian[np.ix_((7, 8), (7, 8))] = rotation
    expected_covariances = np.einsum(
        "ij,njk,lk->nil",
        jacobian,
        np.asarray(cache.covariances),
        jacobian,
    )

    np.testing.assert_allclose(output.boxes_3d, expected_boxes, atol=1e-15)
    np.testing.assert_allclose(output.velocities, expected_velocities, atol=1e-15)
    np.testing.assert_allclose(output.covariances, expected_covariances, atol=1e-15)
    assert np.all(output.boxes_3d[:, 6] >= -np.pi)
    assert np.all(output.boxes_3d[:, 6] < np.pi)
    assert output.event_time == cache.event_time
    np.testing.assert_array_equal(cache.boxes_3d, _cache().boxes_3d)


def test_c8_materializes_clock_and_pose_disturbances_together() -> None:
    cache = _cache()
    event, trace = _event_and_trace(
        "C8",
        clock_offset_seconds=-0.02,
        clock_drift_ppm=30.0,
        clock_error_at_transmit_seconds=-0.015,
        translation_noise_metres=(0.2, -0.1, 0.05),
        yaw_noise_degrees=-1.0,
    )

    realized = realize_condition_input_v1(cache, trace, event)

    assert realized.detection_cache.event_time == 9.985
    assert not np.array_equal(realized.detection_cache.boxes_3d, cache.boxes_3d)
    assert not np.array_equal(realized.detection_cache.velocities, cache.velocities)
    assert not np.array_equal(realized.detection_cache.covariances, cache.covariances)
    assert realized.evidence.condition_id is NetworkConditionId.C8
    assert realized.evidence.translation_noise_metres == (0.2, -0.1, 0.05)


def test_condition_input_manifest_binds_exact_ordered_c6_run_evidence() -> None:
    cache = _cache()
    first_event, first_trace = _event_and_trace(
        "C6",
        message_id="detection-message-1",
        clock_error_at_transmit_seconds=0.01,
    )
    second_event, _ = _event_and_trace(
        "C6",
        message_id="detection-message-2",
        clock_error_at_transmit_seconds=0.02,
    )
    trace = replace(first_trace, events=(first_event, second_event))
    realized = (
        realize_condition_input_v1(cache, trace, first_event),
        realize_condition_input_v1(cache, trace, second_event),
    )
    manifest = build_condition_input_manifest_v1(
        run_id="run-c6-001",
        trace=trace,
        run_config_sha256="d" * 64,
        detection_cache_sha256=cache.digest(),
        evidence=(item.evidence for item in realized),
    )

    assert manifest.condition_id is NetworkConditionId.C6
    assert manifest.network_trace_sha256 == trace.content_sha256
    assert manifest.condition_plan_sha256 == trace.condition_plan_sha256
    assert manifest.evidence_sha256s == tuple(
        item.evidence.content_sha256 for item in realized
    )
    assert decode_condition_input_manifest_v1(manifest.canonical_bytes) == manifest

    reversed_manifest = build_condition_input_manifest_v1(
        run_id="run-c6-001",
        trace=trace,
        run_config_sha256="d" * 64,
        detection_cache_sha256=cache.digest(),
        evidence=(item.evidence for item in reversed(realized)),
    )
    assert reversed_manifest.content_sha256 != manifest.content_sha256
    with pytest.raises(ValueError, match="must be unique"):
        replace(
            manifest,
            evidence_sha256s=(
                manifest.evidence_sha256s[0],
                manifest.evidence_sha256s[0],
            ),
        )
    evidence_from_other_trace = realize_condition_input_v1(
        cache, first_trace, first_event
    ).evidence
    with pytest.raises(ValueError, match="exact run trace"):
        build_condition_input_manifest_v1(
            run_id="run-c6-001",
            trace=trace,
            run_config_sha256="d" * 64,
            detection_cache_sha256=cache.digest(),
            evidence=(evidence_from_other_trace,),
        )
    with pytest.raises(ValueError, match="restricted to C6-C8"):
        ConditionInputManifestV1(
            run_id="run-c5-001",
            condition_id="C5",
            network_trace_sha256=trace.content_sha256,
            condition_plan_sha256=condition_plan_v1("C5").content_sha256,
            run_config_sha256="d" * 64,
            detection_cache_sha256=cache.digest(),
            evidence_sha256s=manifest.evidence_sha256s,
        )

    noncanonical = json.dumps(manifest.to_primitive()).encode()
    with pytest.raises(ValueError, match="not canonical"):
        decode_condition_input_manifest_v1(noncanonical)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        decode_condition_input_manifest_v1(b'{"kind":"x","kind":"y"}')


def test_source_mismatch_is_rejected() -> None:
    cache = _cache(agent_id="vehicle")
    event, trace = _event_and_trace("C7", source="roadside")

    with pytest.raises(ValueError, match="source must match"):
        realize_condition_input_v1(cache, trace, event)


def test_dropped_event_cannot_be_materialized() -> None:
    cache = _cache()
    event, trace = _event_and_trace(
        "C8",
        arrival_time=None,
        dropped=True,
        clock_error_at_transmit_seconds=0.01,
        translation_noise_metres=(0.1, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="only a delivered"):
        realize_condition_input_v1(cache, trace, event)


def test_future_realized_time_is_preserved_but_causal_gate_rejects_it() -> None:
    cache = _cache(event_time=10.0)
    event, trace = _event_and_trace(
        "C6",
        transmitted_at=9.9,
        arrival_time=10.05,
        clock_offset_seconds=0.1,
        clock_error_at_transmit_seconds=0.1,
    )

    realized = realize_condition_input_v1(cache, trace, event)

    assert realized.detection_cache.event_time == 10.1
    assert realized.evidence.arrival_time == 10.05
    assert not realized.causal_at_arrival
    with pytest.raises(ValueError, match="future input rejected"):
        require_causal_at_arrival_v1(realized)
    assert realized.detection_cache.event_time != realized.evidence.arrival_time


def test_event_must_exactly_match_message_identity_and_bytes_sealed_by_trace() -> None:
    cache = _cache()
    event, trace = _event_and_trace(
        "C7",
        translation_noise_metres=(0.1, 0.0, 0.0),
        yaw_noise_degrees=1.0,
    )
    altered = replace(event, yaw_noise_degrees=2.0)

    with pytest.raises(ValueError, match="exact event sealed"):
        realize_condition_input_v1(cache, trace, altered)


def test_condition_label_cannot_hide_an_unregistered_disturbance_type() -> None:
    cache = _cache()
    event, trace = _event_and_trace(
        "C0",
        clock_offset_seconds=0.01,
        clock_error_at_transmit_seconds=0.01,
    )

    with pytest.raises(ValueError, match="cannot contain a clock disturbance"):
        realize_condition_input_v1(cache, trace, event)
