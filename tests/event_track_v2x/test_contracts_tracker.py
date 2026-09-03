from __future__ import annotations

import json

import numpy as np
import pytest

from transvision.models.event_track_v2x.contracts import (
    ArtifactDigestV1,
    DetectionCacheV1,
    EvaluatorContractV1,
    EvidenceBundleV1,
    IdentityHypothesisV1,
    TrackingPredictionV1,
    contract_digest,
    decode_contract,
    encode_contract,
)
from transvision.models.event_track_v2x.schema import Lineage
from transvision.models.event_track_v2x.tracker import (
    ReferenceMultiTargetTracker,
    ReferenceTrackerConfig,
    TrackerAdapter,
    TrackerIngestStatus,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def detection_cache(
    *,
    sequence_id: str = "0087",
    frame_id: str = "000001",
    event_time: float = 0.0,
    boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    labels: tuple[str, ...] | None = None,
    velocities: np.ndarray | None = None,
    covariances: np.ndarray | None = None,
    coordinate_frame: str = "world",
    agent_id: str = "vehicle",
) -> DetectionCacheV1:
    if boxes is None:
        boxes = np.array([[1.0, 2.0, 0.0, 4.5, 1.8, 1.6, 0.0]])
    count = len(boxes)
    return DetectionCacheV1(
        sequence_id=sequence_id,
        frame_id=frame_id,
        event_time=event_time,
        agent_id=agent_id,
        coordinate_frame=coordinate_frame,
        boxes_3d=boxes,
        scores=np.full(count, 0.9) if scores is None else scores,
        class_labels=("car",) * count if labels is None else labels,
        velocities=np.zeros((count, 2)) if velocities is None else velocities,
        covariances=(
            np.repeat(np.eye(9)[None, :, :] * 0.2, count, axis=0)
            if covariances is None
            else covariances
        ),
        dataset_sha256=SHA_A,
        detector_config_sha256=SHA_B,
        checkpoint_sha256=SHA_C,
    )


def test_detection_cache_contract_is_canonical_strict_and_gt_free() -> None:
    cache = detection_cache()
    encoded = encode_contract(cache)
    assert decode_contract(encoded, DetectionCacheV1) == cache
    assert contract_digest(cache) == cache.digest()
    assert cache.state_vectors().shape == (1, 9)
    assert not cache.boxes_3d.flags.writeable
    assert not cache.covariances.flags.writeable
    digest = cache.digest()
    with pytest.raises(ValueError, match="WRITEABLE"):
        cache.scores.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        cache.boxes_3d.setflags(write=True)
    assert cache.digest() == digest

    unknown = cache.to_primitive()
    unknown["ground_truth"] = []
    with pytest.raises(ValueError, match="unknown=.*ground_truth"):
        DetectionCacheV1.from_mapping(unknown)

    wrong_version = cache.to_primitive()
    wrong_version["schema_version"] = True
    with pytest.raises(ValueError, match="schema version"):
        DetectionCacheV1.from_mapping(wrong_version)

    noncanonical = json.dumps(
        cache.to_primitive(), ensure_ascii=False, sort_keys=False
    ).encode("utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        decode_contract(noncanonical, DetectionCacheV1)


def test_detection_cache_rejects_malformed_shapes_scores_and_covariance() -> None:
    with pytest.raises(ValueError, match="ending in"):
        detection_cache(boxes=np.zeros((1, 8)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        detection_cache(scores=np.array([1.2]))
    invalid_covariance = np.repeat(np.eye(9)[None, :, :], 1, axis=0)
    invalid_covariance[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="positive definite"):
        detection_cache(covariances=invalid_covariance)
    invalid_dimensions = np.array(
        [[1.0, 2.0, 0.0, -4.5, 1.8, 1.6, 0.0]]
    )
    with pytest.raises(ValueError, match="must be positive"):
        detection_cache(boxes=invalid_dimensions)
    invalid_yaw = np.array(
        [[1.0, 2.0, 0.0, 4.5, 1.8, 1.6, np.pi]]
    )
    with pytest.raises(ValueError, match="canonical"):
        detection_cache(boxes=invalid_yaw)
    with pytest.raises(ValueError, match="canonical identifiers"):
        detection_cache(labels=("car|gt-track-id=secret",))
    with pytest.raises(ValueError, match="trimmed"):
        detection_cache(sequence_id=" 0087")


def test_prediction_evaluator_and_evidence_contracts_round_trip() -> None:
    prediction = TrackingPredictionV1(
        sequence_id="0087",
        frame_id="000001",
        track_id="0087:000001",
        class_label="car",
        event_time=1.0,
        arrival_time=1.1,
        decision_time=1.2,
        mean=np.zeros(9),
        covariance=np.eye(9),
        existence_probability=0.9,
        identity_hypotheses=(
            IdentityHypothesisV1("0087:000001", 0.8),
            IdentityHypothesisV1("0087:000002", 0.2),
        ),
        lineage=Lineage(("factor-1",), True),
        committed=True,
    )
    assert decode_contract(
        encode_contract(prediction), TrackingPredictionV1
    ) == prediction
    with pytest.raises(ValueError, match="event_time"):
        TrackingPredictionV1(
            sequence_id="0087",
            frame_id="000001",
            track_id="0087:000001",
            class_label="car",
            event_time=2.0,
            arrival_time=1.0,
            decision_time=3.0,
            mean=np.zeros(9),
            covariance=np.eye(9),
            existence_probability=1.0,
            identity_hypotheses=(IdentityHypothesisV1("id", 1.0),),
            lineage=Lineage(("factor",), True),
            committed=False,
        )

    evaluator = EvaluatorContractV1(
        dataset_name="V2X-Seq-SPD",
        frequency_hz=10.0,
        split_name="val",
        sample_count=3316,
        roi=(-100.0, -40.0, -5.0, 100.0, 40.0, 5.0),
        class_mapping=(("Car", "car"),),
        matching_thresholds=(("center_distance_m", 2.0),),
        evaluator_name="TrackEval",
        evaluator_version="1.0.0",
        cohort_sha256=SHA_A,
    )
    assert decode_contract(
        encode_contract(evaluator), EvaluatorContractV1
    ) == evaluator

    artifacts = {
        name: ArtifactDigestV1(f"artifact://run/{name}", SHA_B, 10)
        for name in EvidenceBundleV1._ARTIFACT_FIELDS
    }
    evidence = EvidenceBundleV1(run_id="run-001", **artifacts)
    decoded = decode_contract(encode_contract(evidence), EvidenceBundleV1)
    assert decoded == evidence
    assert decoded.digest() == contract_digest(evidence)
    malformed = evidence.to_primitive()
    malformed["logs"]["extra"] = True
    with pytest.raises(ValueError, match="unknown=.*extra"):
        EvidenceBundleV1.from_mapping(malformed)


def test_reference_tracker_satisfies_protocol_and_birth_order_is_deterministic() -> None:
    boxes = np.array(
        [
            [8.0, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0],
            [2.0, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0],
        ]
    )
    first = ReferenceMultiTargetTracker()
    second = ReferenceMultiTargetTracker()
    assert isinstance(first, TrackerAdapter)
    first.reset(sequence_id="0087", initial_time=0.0)
    second.reset(sequence_id="0087", initial_time=0.0)
    result_first = first.ingest(detection_cache(boxes=boxes), arrival_time=0.0)
    result_second = second.ingest(
        detection_cache(boxes=boxes[::-1].copy()), arrival_time=0.0
    )
    assert result_first.born_track_ids == (
        "0087:000001",
        "0087:000002",
    )
    assert result_second.born_track_ids == result_first.born_track_ids
    first_positions = [item.mean[0] for item in first.advance(0.0)]
    second_positions = [item.mean[0] for item in second.advance(0.0)]
    assert first_positions == second_positions == [2.0, 8.0]


def test_reference_tracker_matches_tracks_and_keeps_top_h_identity_hypotheses() -> None:
    config = ReferenceTrackerConfig(top_h=2, gate_probability=0.999)
    tracker = ReferenceMultiTargetTracker(config)
    tracker.reset(sequence_id="0087", initial_time=0.0)
    close_boxes = np.array(
        [
            [0.0, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0],
            [0.2, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0],
        ]
    )
    tracker.ingest(detection_cache(boxes=close_boxes), arrival_time=0.0)
    tracker.commit(0.0)
    tracker.advance(0.1)
    result = tracker.ingest(
        detection_cache(
            frame_id="000002",
            event_time=0.1,
            boxes=np.array([[0.1, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0]]),
        ),
        arrival_time=0.1,
    )
    assert result.status is TrackerIngestStatus.APPLIED
    assert len(result.matched_track_ids) == 1
    matched = {
        item.track_id: item for item in tracker.advance(0.1)
    }[result.matched_track_ids[0]]
    assert len(matched.identity_hypotheses) == 2
    assert sum(item.probability for item in matched.identity_hypotheses) == pytest.approx(
        1.0
    )
    assert matched.other_identity_probability == 0.0


def test_reference_tracker_no_future_duplicate_and_frame_checks_fail_closed() -> None:
    tracker = ReferenceMultiTargetTracker()
    tracker.reset(sequence_id="0087", initial_time=0.0)

    future_event = detection_cache(event_time=1.0)
    assert (
        tracker.ingest(future_event, arrival_time=0.0).status
        is TrackerIngestStatus.FUTURE_EVENT
    )
    assert tracker.active_track_ids == ()

    cache = detection_cache()
    assert (
        tracker.ingest(cache, arrival_time=1.0).status
        is TrackerIngestStatus.FUTURE_ARRIVAL
    )
    assert tracker.ingest(cache, arrival_time=0.0).applied
    before = tracker.advance(0.0)[0].digest()
    assert (
        tracker.ingest(cache, arrival_time=0.0).status
        is TrackerIngestStatus.DUPLICATE_CACHE
    )
    assert tracker.advance(0.0)[0].digest() == before

    conflicting = detection_cache(scores=np.array([0.8]))
    conflict_result = tracker.ingest(conflicting, arrival_time=0.0)
    assert (
        conflict_result.status
        is TrackerIngestStatus.CONFLICTING_CACHE_IDENTITY
    )
    assert tracker.advance(0.0)[0].digest() == before

    wrong_sequence = detection_cache(sequence_id="other", frame_id="2")
    assert (
        tracker.ingest(wrong_sequence, arrival_time=0.0).status
        is TrackerIngestStatus.SEQUENCE_MISMATCH
    )
    wrong_frame = detection_cache(coordinate_frame="agent", frame_id="3")
    assert (
        tracker.ingest(wrong_frame, arrival_time=0.0).status
        is TrackerIngestStatus.COORDINATE_FRAME_MISMATCH
    )


def test_top_h_preserves_omitted_identity_probability_mass() -> None:
    tracker = ReferenceMultiTargetTracker(
        ReferenceTrackerConfig(top_h=3, gate_probability=0.999)
    )
    tracker.reset(sequence_id="0087", initial_time=0.0)
    boxes = np.repeat(
        np.array([[0.0, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0]]),
        4,
        axis=0,
    )
    tracker.ingest(detection_cache(boxes=boxes), arrival_time=0.0)
    tracker.commit(0.0)
    tracker.advance(0.1)
    result = tracker.ingest(
        detection_cache(
            frame_id="000002",
            event_time=0.1,
            boxes=boxes[:1],
        ),
        arrival_time=0.1,
    )
    matched = {
        item.track_id: item for item in tracker.advance(0.1)
    }[result.matched_track_ids[0]]
    assert len(matched.identity_hypotheses) == 3
    assert sum(
        item.probability for item in matched.identity_hypotheses
    ) == pytest.approx(0.75)
    assert matched.other_identity_probability == pytest.approx(0.25)


def test_reference_tracker_survival_pruning_commit_and_finalize_are_explicit() -> None:
    tracker = ReferenceMultiTargetTracker(
        ReferenceTrackerConfig(max_age_seconds=0.5)
    )
    tracker.reset(sequence_id="0087", initial_time=0.0)
    tracker.ingest(detection_cache(), arrival_time=0.0)
    first = tracker.commit(0.0)
    assert first[0].committed
    committed_digest = first[0].digest()
    with pytest.raises(ValueError, match="WRITEABLE"):
        first[0].mean.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        first[0].mean[0] = 999.0
    assert first[0].digest() == committed_digest
    first_hash = tracker.commit_records[0].record_hash
    first_bytes = tracker.commit_records[0].snapshot_bytes

    tracker.advance(0.25)
    tracker.ingest(
        detection_cache(frame_id="000002", event_time=0.25), arrival_time=0.25
    )
    second = tracker.commit(0.25)
    assert second[0].track_id == first[0].track_id
    assert tracker.commit_records[0].record_hash == first_hash
    assert tracker.commit_records[0].snapshot_bytes == first_bytes
    assert tracker.commit_records[1].previous_hash == first_hash

    tracker.advance(1.0)
    assert tracker.active_track_ids == ()
    with pytest.raises(RuntimeError, match="uncommitted"):
        tracker.finalize()
    assert tracker.commit(1.0) == ()
    history = tracker.finalize()
    assert history == first + second
    assert history[0].digest() == committed_digest
    with pytest.raises(RuntimeError, match="finalized"):
        tracker.advance(2.0)


def test_reference_tracker_time_fallback_frame_id_is_archive_path_safe() -> None:
    tracker = ReferenceMultiTargetTracker()
    tracker.reset(sequence_id="0087", initial_time=0.0)
    tracker.ingest(detection_cache(), arrival_time=0.0)
    tracker.commit(0.0)

    prediction = tracker.commit(0.25)[0]

    assert prediction.frame_id.startswith("time-")
    assert len(prediction.frame_id) == len("time-") + 16
    assert prediction.frame_id[5:].isalnum()


def test_tracker_reset_cannot_erase_committed_history() -> None:
    tracker = ReferenceMultiTargetTracker()
    tracker.reset(sequence_id="0087", initial_time=0.0)
    tracker.ingest(detection_cache(), arrival_time=0.0)
    tracker.commit(0.0)
    records = tracker.commit_records

    with pytest.raises(RuntimeError, match="single-sequence"):
        tracker.reset(sequence_id="0099", initial_time=0.0)
    assert tracker.commit_records == records


def test_tracker_cannot_finalize_after_uncommitted_late_input() -> None:
    tracker = ReferenceMultiTargetTracker()
    tracker.reset(sequence_id="0087", initial_time=0.0)
    tracker.ingest(detection_cache(), arrival_time=0.0)
    with pytest.raises(RuntimeError, match="commit before finalize"):
        tracker.finalize()


def test_multi_agent_caches_apply_one_miss_and_preserve_reference_frame() -> None:
    tracker = ReferenceMultiTargetTracker()
    tracker.reset(sequence_id="0087", initial_time=0.0)
    tracker.ingest(detection_cache(), arrival_time=0.0)
    tracker.commit(0.0)

    tracker.advance(0.1)
    matched = detection_cache(frame_id="000002", event_time=0.1)
    empty_remote = detection_cache(
        frame_id="remote-0009",
        event_time=0.1,
        agent_id="infrastructure",
        boxes=np.empty((0, 7)),
    )
    assert tracker.ingest(matched, arrival_time=0.1).applied
    assert tracker.ingest(empty_remote, arrival_time=0.1).applied
    committed = tracker.commit(0.1)

    assert committed[0].frame_id == "000002"
    survived = 0.9 * 0.95**0.1
    assert committed[0].existence_probability == pytest.approx(
        1.0 - (1.0 - survived) * 0.1
    )


def test_multiple_empty_agent_caches_count_as_one_missed_decision() -> None:
    def run(agent_ids: tuple[str, ...]) -> float:
        tracker = ReferenceMultiTargetTracker()
        tracker.reset(sequence_id="0087", initial_time=0.0)
        tracker.ingest(detection_cache(), arrival_time=0.0)
        tracker.commit(0.0)
        tracker.advance(0.1)
        for index, agent_id in enumerate(agent_ids):
            tracker.ingest(
                detection_cache(
                    frame_id=f"empty-{index}",
                    event_time=0.1,
                    agent_id=agent_id,
                    boxes=np.empty((0, 7)),
                ),
                arrival_time=0.1,
            )
        return tracker.commit(0.1)[0].existence_probability

    assert run(("infrastructure", "vehicle")) == pytest.approx(run(("vehicle",)))
