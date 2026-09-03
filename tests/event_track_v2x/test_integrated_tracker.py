from __future__ import annotations

import numpy as np
import pytest

from transvision.models.event_track_v2x import (
    EVENTTRACK_REFERENCE_CAPABILITIES_V1,
    AffineClockMap,
    CooperativeIngestStatus,
    CorrelatedTrackBelief,
    DetectionCacheV1,
    EventTrackReferenceCapabilityError,
    EventTrackReferenceConfig,
    EventTrackReferenceTracker,
    FrozenOnTimeProbabilityV1,
    IndependentIncrement,
    Lineage,
    LocalTimestamps,
    ReceivedMessage,
    ReferenceTrackerConfig,
    TrackerAdapter,
    WireMessage,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
IDENTITY_CLOCK = AffineClockMap(1.0, 0.0, np.zeros((2, 2)))


def _cache(
    *,
    frame_id: str = "000001",
    event_time: float = 0.0,
    agent_id: str = "vehicle",
    x: float = 0.0,
    score: float = 0.9,
) -> DetectionCacheV1:
    return DetectionCacheV1(
        sequence_id="0087",
        frame_id=frame_id,
        event_time=event_time,
        agent_id=agent_id,
        coordinate_frame="world",
        boxes_3d=np.array([[x, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0]]),
        scores=np.array([score]),
        class_labels=("car",),
        velocities=np.array([[1.0, 0.0]]),
        covariances=np.repeat((np.eye(9) * 0.4)[None, :, :], 1, axis=0),
        dataset_sha256=SHA_A,
        detector_config_sha256=SHA_B,
        checkpoint_sha256=SHA_C,
    )


def _tracker(*, lag: float = 3.0, current_time: float = 0.0) -> EventTrackReferenceTracker:
    tracker = EventTrackReferenceTracker(
        EventTrackReferenceConfig(
            tracker=ReferenceTrackerConfig(max_age_seconds=10.0),
            fixed_lag_seconds=lag,
        )
    )
    tracker.reset(sequence_id="0087", initial_time=0.0)
    tracker.ingest(_cache(), arrival_time=0.0)
    tracker.commit(0.0)
    if current_time:
        tracker.advance(current_time)
    return tracker


def _increment(
    message_id: str,
    *,
    sequence: int,
    event_time: float,
    received_at: float,
    target_id: str = "0087:000001",
    full_state: bool = False,
    factor_id: str | None = None,
) -> ReceivedMessage:
    if full_state:
        measurement = np.zeros(9)
        measurement[0] = event_time + 0.25
        measurement[7] = 1.0
        matrix = np.eye(9)
        covariance = np.eye(9) * 0.1
    else:
        measurement = np.array([event_time + 0.25])
        matrix = np.zeros((1, 9))
        matrix[0, 0] = 1.0
        covariance = np.array([[0.1]])
    return ReceivedMessage(
        message=WireMessage(
            message_id=message_id,
            source="roadside",
            sequence=sequence,
            timestamps=LocalTimestamps(
                event_time,
                event_time,
                event_time + 0.01,
                event_time + 0.02,
            ),
            deadline=3.0,
            coordinate_frame="world",
            ttl=5.0,
            payload=IndependentIncrement(
                target_id=target_id,
                measurement=measurement,
                measurement_matrix=matrix,
                measurement_covariance=covariance,
                lineage=Lineage((factor_id or f"factor-{message_id}",), True),
            ),
        ),
        received_at=received_at,
    )


def _belief(
    message_id: str,
    *,
    event_time: float,
    received_at: float,
    factor_id: str,
) -> ReceivedMessage:
    mean = np.zeros(9)
    mean[0] = 2.0
    mean[7] = 1.0
    return ReceivedMessage(
        message=WireMessage(
            message_id=message_id,
            source="roadside",
            sequence=10,
            timestamps=LocalTimestamps(
                event_time,
                event_time,
                event_time + 0.01,
                event_time + 0.02,
            ),
            deadline=3.0,
            coordinate_frame="world",
            ttl=5.0,
            payload=CorrelatedTrackBelief(
                track_id="0087:000001",
                existence_probability=0.8,
                mean=mean,
                covariance=np.eye(9) * 0.05,
                identity_probabilities=(
                    ("0087:000001", 0.7),
                    ("remote-alternative", 0.3),
                ),
                lineage=Lineage((factor_id,), False),
            ),
        ),
        received_at=received_at,
    )


def test_capabilities_fail_closed_and_tracker_implements_adapter() -> None:
    tracker = EventTrackReferenceTracker()
    assert isinstance(tracker, TrackerAdapter)
    capabilities = EVENTTRACK_REFERENCE_CAPABILITIES_V1
    capabilities.require(
        "multi_target_birth_death",
        "point_time_fixed_lag_oosm",
        "correlated_belief_ci",
        "analytic_counterfactual_voi",
    )
    with pytest.raises(EventTrackReferenceCapabilityError, match="unsupported"):
        capabilities.require("learned_voi")
    with pytest.raises(EventTrackReferenceCapabilityError, match="unknown"):
        capabilities.require("invented_capability")
    assert capabilities.formal_experiment_or_sota_evidence is False


def test_point_time_oosm_replay_is_arrival_order_invariant() -> None:
    newer = _increment(
        "newer",
        sequence=2,
        event_time=1.1,
        received_at=1.5,
    )
    older = _increment(
        "older",
        sequence=1,
        event_time=0.6,
        received_at=1.5,
    )
    first = _tracker(current_time=1.5)
    second = _tracker(current_time=1.5)

    assert first.ingest_cooperative(newer, clock_map=IDENTITY_CLOCK).status is (
        CooperativeIngestStatus.APPLIED_POINT_REPLAY
    )
    assert first.ingest_cooperative(older, clock_map=IDENTITY_CLOCK).applied
    assert second.ingest_cooperative(older, clock_map=IDENTITY_CLOCK).applied
    assert second.ingest_cooperative(newer, clock_map=IDENTITY_CLOCK).applied

    first_prediction = first.prediction_for_track("0087:000001")
    second_prediction = second.prediction_for_track("0087:000001")
    assert first_prediction.mean.tobytes() == second_prediction.mean.tobytes()
    assert (
        first_prediction.covariance.tobytes()
        == second_prediction.covariance.tobytes()
    )
    assert first_prediction.lineage.factor_ids == second_prediction.lineage.factor_ids


def test_detector_cache_and_cooperative_factor_share_one_append_only_path() -> None:
    tracker = _tracker(current_time=0.5)
    first_record = tracker.commit_records[0]
    detector = _cache(frame_id="000002", event_time=0.4, x=0.5)
    assert tracker.ingest(detector, arrival_time=0.5).applied
    increment = _increment(
        "late-independent",
        sequence=1,
        event_time=0.25,
        received_at=0.5,
    )
    assert tracker.ingest_cooperative(increment, clock_map=IDENTITY_CLOCK).applied
    prediction = tracker.commit(0.5)[0]

    assert tracker.commit_records[0] == first_record
    assert tracker.commit_records[1].previous_hash == first_record.record_hash
    assert detector.digest() in tracker.commit_records[1].message_ids
    assert "late-independent" in tracker.commit_records[1].message_ids
    assert "factor-late-independent" in prediction.lineage.factor_ids
    assert "late-independent" in prediction.lineage.ancestor_message_ids


def test_cooperative_only_commit_records_message_without_detector_miss_decay() -> None:
    tracker = EventTrackReferenceTracker(
        EventTrackReferenceConfig(
            tracker=ReferenceTrackerConfig(
                max_age_seconds=10.0,
                missed_detection_factor=0.5,
            ),
            fixed_lag_seconds=3.0,
        )
    )
    tracker.reset(sequence_id="0087", initial_time=0.0)
    two_tracks = DetectionCacheV1(
        sequence_id="0087",
        frame_id="000001",
        event_time=0.0,
        agent_id="vehicle",
        coordinate_frame="world",
        boxes_3d=np.array(
            [
                [0.0, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0],
                [20.0, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0],
            ]
        ),
        scores=np.array([0.9, 0.9]),
        class_labels=("car", "car"),
        velocities=np.array([[1.0, 0.0], [1.0, 0.0]]),
        covariances=np.repeat((np.eye(9) * 0.4)[None, :, :], 2, axis=0),
        dataset_sha256=SHA_A,
        detector_config_sha256=SHA_B,
        checkpoint_sha256=SHA_C,
    )
    tracker.ingest(two_tracks, arrival_time=0.0)
    tracker.commit(0.0)
    tracker.advance(1.0)
    increment = _increment(
        "cooperative-only",
        sequence=1,
        event_time=0.75,
        received_at=1.0,
        target_id="0087:000001",
    )
    assert tracker.ingest_cooperative(increment, clock_map=IDENTITY_CLOCK).applied
    before_commit = {
        item.track_id: item.existence_probability for item in tracker.advance(1.0)
    }

    committed = {item.track_id: item for item in tracker.commit(1.0)}

    assert committed["0087:000002"].existence_probability == pytest.approx(
        before_commit["0087:000002"]
    )
    assert "cooperative-only" in tracker.commit_records[-1].message_ids


def test_uncertain_source_time_uses_full_state_moment_match_only() -> None:
    clock = AffineClockMap(
        1.0,
        0.0,
        np.zeros((2, 2)),
        timestamp_variance=0.01,
    )
    tracker = _tracker(current_time=1.5)
    full = _increment(
        "mixture-full",
        sequence=1,
        event_time=1.0,
        received_at=1.5,
        full_state=True,
    )
    assert tracker.ingest_cooperative(full, clock_map=clock).status is (
        CooperativeIngestStatus.APPLIED_MIXTURE_MOMENT
    )

    unsupported = _increment(
        "mixture-partial",
        sequence=2,
        event_time=1.0,
        received_at=1.5,
    )
    result = tracker.ingest_cooperative(unsupported, clock_map=clock)
    assert result.status is CooperativeIngestStatus.UNSUPPORTED_PAYLOAD_SHAPE
    assert (
        tracker.ingest_cooperative(unsupported, clock_map=clock).status
        is CooperativeIngestStatus.DUPLICATE
    )


def test_correlated_belief_uses_ci_merges_lineage_and_rebases_replay() -> None:
    tracker = _tracker(current_time=1.5)
    belief = _belief(
        "belief-a",
        event_time=1.0,
        received_at=1.5,
        factor_id="correlated-a",
    )
    before = tracker.prediction_for_track("0087:000001")
    result = tracker.ingest_cooperative(belief, clock_map=IDENTITY_CLOCK)
    assert result.status is CooperativeIngestStatus.APPLIED_CORRELATED_CI
    after = tracker.prediction_for_track("0087:000001")
    assert after.covariance[0, 0] <= before.covariance[0, 0]
    assert after.lineage.complete is False
    assert "correlated-a" in after.lineage.factor_ids
    assert "belief-a" in after.lineage.ancestor_message_ids
    assert len(after.identity_hypotheses) <= tracker.config.top_h
    assert (
        sum(item.probability for item in after.identity_hypotheses)
        + after.other_identity_probability
        == pytest.approx(1.0)
    )
    assert (
        tracker.ingest_cooperative(belief, clock_map=IDENTITY_CLOCK).status
        is CooperativeIngestStatus.DUPLICATE
    )

    shared = ReceivedMessage(
        message=WireMessage(
            message_id="belief-b",
            source="roadside",
            sequence=11,
            timestamps=LocalTimestamps(1.4, 1.4, 1.41, 1.42),
            deadline=3.0,
            coordinate_frame="world",
            ttl=5.0,
            payload=CorrelatedTrackBelief(
                track_id="0087:000001",
                existence_probability=after.existence_probability,
                mean=after.mean,
                covariance=after.covariance,
                identity_probabilities=tuple(
                    (item.identity_id, item.probability)
                    for item in after.identity_hypotheses
                )
                + (
                    ("lineage-other", after.other_identity_probability),
                )
                if after.other_identity_probability
                else tuple(
                    (item.identity_id, item.probability)
                    for item in after.identity_hypotheses
                ),
                lineage=Lineage(
                    ("correlated-a",),
                    False,
                    ("belief-a",),
                ),
            ),
        ),
        received_at=1.5,
    )
    shared_result = tracker.ingest_cooperative(shared, clock_map=IDENTITY_CLOCK)
    assert shared_result.status is CooperativeIngestStatus.APPLIED_CORRELATED_CI
    after_shared = tracker.prediction_for_track("0087:000001")
    np.testing.assert_allclose(
        after_shared.covariance,
        after.covariance,
        rtol=1e-10,
        atol=1e-12,
    )
    pre_boundary = _increment(
        "pre-ci-late",
        sequence=3,
        event_time=1.25,
        received_at=1.5,
    )
    assert (
        tracker.ingest_cooperative(pre_boundary, clock_map=IDENTITY_CLOCK).status
        is CooperativeIngestStatus.OUT_OF_WINDOW
    )


def test_analytic_voi_counterfactual_does_not_mutate_tracker() -> None:
    tracker = _tracker(current_time=1.0)
    candidate = _increment(
        "voi-candidate",
        sequence=1,
        event_time=1.0,
        received_at=1.0,
        full_state=True,
    )
    before = tracker.prediction_for_track("0087:000001")
    score = tracker.score_cooperative_candidate(
        candidate,
        clock_map=IDENTITY_CLOCK,
        on_time_estimate=FrozenOnTimeProbabilityV1(
            probability=0.9,
            as_of_network_time=1.0,
            network_transmit_time=1.0,
            message_deadline=3.0,
            channel_model_sha256=SHA_A,
        ),
    )
    after = tracker.prediction_for_track("0087:000001")

    assert after.digest() == before.digest()
    assert score.before_prediction_sha256 == before.digest()
    assert score.causal_score_evidence.candidate_message_sha256
    assert score.expected_value >= 0.0


def test_batch_is_atomic_when_one_cache_conflicts() -> None:
    tracker = EventTrackReferenceTracker()
    tracker.reset(sequence_id="0087", initial_time=0.0)
    valid = _cache()
    conflict = _cache(score=0.8)

    with pytest.raises((TypeError, ValueError)):
        tracker.ingest_batch(((valid, 0.0), (conflict, 0.0)))
    assert tracker.active_track_ids == ()
