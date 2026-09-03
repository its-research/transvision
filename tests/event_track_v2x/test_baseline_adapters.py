from __future__ import annotations

import numpy as np
import pytest

from transvision.models.event_track_v2x.baseline_adapters import (
    AsynchronousFusionMode,
    AsynchronousFusionTrackerAdapter,
    AtomicBatchUnsupportedError,
    DetectionInputLock,
    DetectionInputMismatchError,
    UnavailableBaselineError,
    VehicleOnlyTrackerAdapter,
    build_official_backend,
    official_backend_statuses,
)
from transvision.models.event_track_v2x.contracts import DetectionCacheV1
from transvision.models.event_track_v2x.network import (
    NetworkTraceEventV1,
    NetworkTraceV1,
    PacketRequest,
    WireAccountingConfigV1,
    account_wire_bytes_v1,
    condition_plan_v1,
)
from transvision.models.event_track_v2x.runner import ReferenceSequenceRunnerV1
from transvision.models.event_track_v2x.tracker import (
    ReferenceMultiTargetTracker,
    TrackerAdapter,
    TrackerIngestStatus,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def cache(
    *,
    sequence_id: str = "0087",
    frame_id: str = "000001",
    event_time: float = 0.0,
    agent_id: str = "vehicle",
    position_x: float = 0.0,
    velocity_x: float = 0.0,
    dataset_sha256: str = SHA_A,
    detector_config_sha256: str = SHA_B,
    checkpoint_sha256: str = SHA_C,
) -> DetectionCacheV1:
    return DetectionCacheV1(
        sequence_id=sequence_id,
        frame_id=frame_id,
        event_time=event_time,
        agent_id=agent_id,
        coordinate_frame="world",
        boxes_3d=np.array([[position_x, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0]]),
        scores=np.array([0.9]),
        class_labels=("car",),
        velocities=np.array([[velocity_x, 0.0]]),
        covariances=np.repeat(np.eye(9)[None, :, :] * 0.2, 1, axis=0),
        dataset_sha256=dataset_sha256,
        detector_config_sha256=detector_config_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )


def detection_lock(
    frame: DetectionCacheV1 | None = None,
    *additional_frames: DetectionCacheV1,
) -> DetectionInputLock:
    return DetectionInputLock.from_caches(
        (frame or cache(), *additional_frames), cache_manifest_sha256=SHA_D
    )


class _SingleIngestBackend:
    """Protocol-compatible backend intentionally lacking atomic batching."""

    def __init__(self) -> None:
        self._tracker = ReferenceMultiTargetTracker()

    @property
    def adapter_name(self) -> str:
        return "single-ingest-test-backend"

    @property
    def current_time(self) -> float:
        return self._tracker.current_time

    def reset(self, *, sequence_id: str, initial_time: float) -> None:
        self._tracker.reset(sequence_id=sequence_id, initial_time=initial_time)

    def ingest(self, detections: DetectionCacheV1, *, arrival_time: float):
        return self._tracker.ingest(detections, arrival_time=arrival_time)

    def advance(self, decision_time: float):
        return self._tracker.advance(decision_time)

    def commit(self, decision_time: float):
        return self._tracker.commit(decision_time)

    def finalize(self):
        return self._tracker.finalize()


def _delivered_event(
    frame: DetectionCacheV1,
    *,
    message_id: str,
    arrival_time: float,
) -> NetworkTraceEventV1:
    wire = WireAccountingConfigV1()
    encoded_bytes = len(canonical_json_bytes(frame.to_primitive()))
    return NetworkTraceEventV1(
        packet=PacketRequest(
            message_id=message_id,
            source=frame.agent_id,
            sequence=1,
            transmitted_at=0.0,
            deadline=1.0,
            encoded_bytes=encoded_bytes,
        ),
        arrival_time=arrival_time,
        dropped=False,
        channel_state=None,
        clock_offset_seconds=0.0,
        clock_drift_ppm=0.0,
        clock_error_at_transmit_seconds=0.0,
        translation_noise_metres=(0.0, 0.0, 0.0),
        yaw_noise_degrees=0.0,
        byte_account=account_wire_bytes_v1(
            encoded_bytes,
            wire,
            ack_transmissions=1,
        ),
        wire_accounting_config=wire,
    )


def test_vehicle_only_adapter_implements_lifecycle_and_records_shared_input() -> None:
    frame = cache()
    adapter = VehicleOnlyTrackerAdapter(detection_lock(frame))
    adapter.reset(sequence_id="0087", initial_time=0.0)
    assert isinstance(adapter, TrackerAdapter)
    result = adapter.ingest(frame, arrival_time=0.0)
    assert result.applied
    assert result.cache_sha256 == frame.digest()
    assert adapter.input_cache_sha256s == (frame.digest(),)

    committed = adapter.commit(0.0)
    assert committed[0].committed
    assert adapter.finalize() == committed
    with pytest.raises(RuntimeError, match="finalized"):
        adapter.advance(0.1)


def test_vehicle_only_rejects_remote_cache_before_mutating_backend() -> None:
    vehicle = cache()
    remote = cache(agent_id="infrastructure")
    adapter = VehicleOnlyTrackerAdapter(detection_lock(vehicle, remote))
    adapter.reset(sequence_id="0087", initial_time=0.0)
    with pytest.raises(DetectionInputMismatchError, match="non-vehicle"):
        adapter.ingest(remote, arrival_time=0.0)
    assert adapter.input_cache_sha256s == ()

    result = adapter.ingest(vehicle, arrival_time=0.0)
    assert result.born_track_ids == ("0087:000001",)


def test_detector_lock_and_single_sequence_fail_closed() -> None:
    sequence_mismatch = cache(sequence_id="0099", frame_id="000002")
    adapter = VehicleOnlyTrackerAdapter(detection_lock(cache(), sequence_mismatch))
    adapter.reset(sequence_id="0087", initial_time=0.0)

    with pytest.raises(DetectionInputMismatchError, match="checkpoint_sha256"):
        adapter.ingest(cache(checkpoint_sha256=SHA_D), arrival_time=0.0)
    assert adapter.input_cache_sha256s == ()

    mismatch = adapter.ingest(sequence_mismatch, arrival_time=0.0)
    assert mismatch.status is TrackerIngestStatus.SEQUENCE_MISMATCH
    assert adapter.input_cache_sha256s == ()


def test_vehicle_only_preserves_no_future_and_duplicate_idempotence() -> None:
    frame = cache()
    future_event = cache(frame_id="000002", event_time=0.2)
    adapter = VehicleOnlyTrackerAdapter(detection_lock(frame, future_event))
    adapter.reset(sequence_id="0087", initial_time=0.0)

    future_arrival = adapter.ingest(frame, arrival_time=0.1)
    assert future_arrival.status is TrackerIngestStatus.FUTURE_ARRIVAL
    assert adapter.input_cache_sha256s == ()

    applied = adapter.ingest(frame, arrival_time=0.0)
    assert applied.applied
    before = adapter.advance(0.0)[0].digest()
    duplicate = adapter.ingest(frame, arrival_time=0.0)
    assert duplicate.status is TrackerIngestStatus.DUPLICATE_CACHE
    assert duplicate.cache_sha256 == frame.digest()
    assert adapter.advance(0.0)[0].digest() == before
    assert adapter.input_cache_sha256s == (frame.digest(),)

    result = adapter.ingest(future_event, arrival_time=0.0)
    assert result.status is TrackerIngestStatus.FUTURE_EVENT
    assert adapter.input_cache_sha256s == (frame.digest(),)


def test_asynchronous_fusion_modes_distinguish_stale_and_compensated_remote() -> None:
    remote = cache(
        agent_id="infrastructure",
        event_time=0.0,
        velocity_x=10.0,
    )
    lock = detection_lock(remote)
    stale = AsynchronousFusionTrackerAdapter(
        lock, mode=AsynchronousFusionMode.DIRECT_STALE
    )
    compensated = AsynchronousFusionTrackerAdapter(
        lock, mode=AsynchronousFusionMode.CONSTANT_VELOCITY
    )
    for adapter in (stale, compensated):
        adapter.reset(sequence_id="0087", initial_time=1.0)
        assert adapter.ingest(remote, arrival_time=1.0).applied

    stale_prediction = stale.advance(1.0)[0]
    compensated_prediction = compensated.advance(1.0)[0]
    assert stale_prediction.mean[0] == pytest.approx(0.0)
    assert compensated_prediction.mean[0] == pytest.approx(10.0)
    assert stale.input_cache_sha256s == compensated.input_cache_sha256s
    assert stale.input_cache_sha256s == (remote.digest(),)

    duplicate = stale.ingest(remote, arrival_time=1.0)
    assert duplicate.status is TrackerIngestStatus.DUPLICATE_CACHE
    assert stale.advance(1.0)[0].digest() == stale_prediction.digest()


def test_stale_view_collision_is_rejected_instead_of_counted_as_duplicate() -> None:
    first = cache(agent_id="infrastructure", velocity_x=10.0)
    second = cache(agent_id="infrastructure", velocity_x=20.0)
    adapter = AsynchronousFusionTrackerAdapter(
        detection_lock(first, second), mode=AsynchronousFusionMode.DIRECT_STALE
    )
    adapter.reset(sequence_id="0087", initial_time=0.0)
    assert adapter.ingest(first, arrival_time=0.0).applied
    with pytest.raises(DetectionInputMismatchError, match="collapse"):
        adapter.ingest(second, arrival_time=0.0)
    assert adapter.input_cache_sha256s == (first.digest(),)


def test_atomic_batch_is_permutation_invariant_for_two_agents() -> None:
    vehicle = cache(agent_id="vehicle", position_x=0.0)
    remote = cache(agent_id="infrastructure", position_x=20.0)
    lock = detection_lock(vehicle, remote)
    committed_digests: list[tuple[str, ...]] = []
    receipt_orders: list[tuple[str, ...]] = []
    result_orders: list[tuple[str, ...]] = []

    for ordered in ((vehicle, remote), (remote, vehicle)):
        adapter = AsynchronousFusionTrackerAdapter(
            lock,
            mode=AsynchronousFusionMode.CONSTANT_VELOCITY,
        )
        adapter.reset(sequence_id="0087", initial_time=0.0)
        assert adapter.supports_atomic_ingest_batch
        assert not adapter.ranking_eligible
        results = adapter.ingest_batch((frame, 0.0) for frame in ordered)
        committed = adapter.commit(0.0)
        committed_digests.append(tuple(item.digest() for item in committed))
        receipt_orders.append(adapter.input_cache_sha256s)
        result_orders.append(tuple(item.cache_sha256 for item in results))

    assert committed_digests[0] == committed_digests[1]
    assert receipt_orders[0] == receipt_orders[1]
    assert result_orders[0] == result_orders[1] == receipt_orders[0]


def test_batch_role_failure_preflights_without_partial_backend_mutation() -> None:
    vehicle = cache(agent_id="vehicle")
    remote = cache(agent_id="infrastructure", position_x=20.0)
    adapter = VehicleOnlyTrackerAdapter(detection_lock(vehicle, remote))
    adapter.reset(sequence_id="0087", initial_time=0.0)

    with pytest.raises(DetectionInputMismatchError, match="non-vehicle"):
        adapter.ingest_batch(((vehicle, 0.0), (remote, 0.0)))
    assert adapter.input_cache_sha256s == ()
    assert adapter.ingest(vehicle, arrival_time=0.0).born_track_ids == (
        "0087:000001",
    )


def test_batch_rejects_transformed_view_collision_before_mutation() -> None:
    first = cache(agent_id="infrastructure", velocity_x=10.0)
    second = cache(agent_id="infrastructure", velocity_x=20.0)
    adapter = AsynchronousFusionTrackerAdapter(
        detection_lock(first, second),
        mode=AsynchronousFusionMode.DIRECT_STALE,
    )
    adapter.reset(sequence_id="0087", initial_time=0.0)

    with pytest.raises(DetectionInputMismatchError, match="collapse"):
        adapter.ingest_batch(((first, 0.0), (second, 0.0)))
    assert adapter.input_cache_sha256s == ()
    assert adapter.ingest(first, arrival_time=0.0).applied


def test_backend_without_atomic_batch_fails_closed_without_sequential_fallback() -> None:
    frame = cache()
    adapter = VehicleOnlyTrackerAdapter(
        detection_lock(frame),
        backend=_SingleIngestBackend(),
    )
    adapter.reset(sequence_id="0087", initial_time=0.0)
    assert not adapter.supports_atomic_ingest_batch

    with pytest.raises(AtomicBatchUnsupportedError, match="sequential fallback"):
        adapter.ingest_batch(((frame, 0.0),))
    assert adapter.input_cache_sha256s == ()
    assert adapter.ingest(frame, arrival_time=0.0).applied


def test_runner_uses_detector_locked_atomic_batch_for_same_arrival_agents() -> None:
    vehicle = cache(agent_id="vehicle", position_x=0.0)
    remote = cache(agent_id="infrastructure", position_x=20.0)
    plan = condition_plan_v1("C1")
    trace = NetworkTraceV1(
        condition_id="C1",
        seed=1001,
        condition_plan=plan,
        condition_plan_sha256=plan.content_sha256,
        events=(
            _delivered_event(
                remote,
                message_id="message-infrastructure",
                arrival_time=0.1,
            ),
            _delivered_event(
                vehicle,
                message_id="message-vehicle",
                arrival_time=0.1,
            ),
        ),
    )
    adapter = AsynchronousFusionTrackerAdapter(
        detection_lock(vehicle, remote),
        mode=AsynchronousFusionMode.CONSTANT_VELOCITY,
    )

    result = ReferenceSequenceRunnerV1(adapter).run(
        caches_by_message_id={
            "message-infrastructure": remote,
            "message-vehicle": vehicle,
        },
        network_trace=trace,
        run_config_sha256=SHA_D,
        decision_times=(0.2,),
    )

    assert result.processed_message_ids == (
        "message-infrastructure",
        "message-vehicle",
    )
    assert result.predictions
    assert set(adapter.input_cache_sha256s) == {
        vehicle.digest(),
        remote.digest(),
    }


def test_detector_lock_rejects_unlisted_frame_before_backend_mutation() -> None:
    admitted = cache()
    unlisted = cache(frame_id="000002", event_time=0.1)
    adapter = VehicleOnlyTrackerAdapter(detection_lock(admitted))
    adapter.reset(sequence_id="0087", initial_time=0.0)

    with pytest.raises(DetectionInputMismatchError, match="verified manifest"):
        adapter.ingest(unlisted, arrival_time=0.1)
    assert adapter.input_cache_sha256s == ()

    assert adapter.ingest(admitted, arrival_time=0.0).applied
    assert adapter.input_cache_sha256s == (admitted.digest(),)


def test_official_baselines_fail_closed_without_verified_bridges() -> None:
    statuses = official_backend_statuses()
    assert {status.name for status in statuses} == {
        "ab3dmot",
        "immortaltracker",
        "simpletrack",
    }
    for status in statuses:
        assert not status.ready
        assert not status.ranking_eligible
        assert status.bridge_module in status.missing_modules
        with pytest.raises(UnavailableBaselineError, match=status.name):
            build_official_backend(status.name)


def test_unknown_official_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown official baseline"):
        build_official_backend("not-a-tracker")
