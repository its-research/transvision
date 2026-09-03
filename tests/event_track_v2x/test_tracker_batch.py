from __future__ import annotations

import itertools

import numpy as np
import pytest

from transvision.models.event_track_v2x.contracts import DetectionCacheV1
from transvision.models.event_track_v2x.tracker import ReferenceMultiTargetTracker


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _cache(
    *,
    agent_id: str,
    frame_id: str,
    event_time: float = 0.0,
    sequence_id: str = "0087",
    coordinate_frame: str = "world",
    x_positions: tuple[float, ...] = (0.0,),
    score: float = 0.9,
) -> DetectionCacheV1:
    boxes = np.asarray(
        [[x, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0] for x in x_positions],
        dtype=np.float64,
    ).reshape((-1, 7))
    count = len(boxes)
    return DetectionCacheV1(
        sequence_id=sequence_id,
        frame_id=frame_id,
        event_time=event_time,
        agent_id=agent_id,
        coordinate_frame=coordinate_frame,
        boxes_3d=boxes,
        scores=np.full(count, score, dtype=np.float64),
        class_labels=("car",) * count,
        velocities=np.zeros((count, 2), dtype=np.float64),
        covariances=np.repeat(
            (np.eye(9, dtype=np.float64) * 0.2)[None, :, :], count, axis=0
        ),
        dataset_sha256=SHA_A,
        detector_config_sha256=SHA_B,
        checkpoint_sha256=SHA_C,
    )


def _new_tracker(*, initial_time: float = 0.0) -> ReferenceMultiTargetTracker:
    tracker = ReferenceMultiTargetTracker()
    tracker.reset(sequence_id="0087", initial_time=initial_time)
    return tracker


def test_two_nonempty_agents_have_permutation_invariant_output_and_commit() -> None:
    caches = (
        (_cache(agent_id="infrastructure", frame_id="rsu-1"), 0.0),
        (_cache(agent_id="vehicle", frame_id="000001"), 0.0),
    )
    observations: list[tuple[tuple[object, ...], str]] = []

    for permutation in itertools.permutations(caches):
        tracker = _new_tracker()
        results = tracker.ingest_batch(permutation)
        predictions = tracker.commit(0.0)
        observations.append(
            (
                tuple(
                    (
                        result.cache_sha256,
                        result.matched_track_ids,
                        result.born_track_ids,
                    )
                    for result in results
                )
                + tuple(prediction.digest() for prediction in predictions),
                tracker.commit_records[-1].record_hash,
            )
        )

    assert observations[0] == observations[1]
    assert len(observations[0][0]) == 3


def test_three_nonempty_agents_update_permutation_invariant_existing_tracks() -> None:
    updates = (
        (
            _cache(
                agent_id="roadside-b",
                frame_id="b-2",
                event_time=0.1,
                x_positions=(0.2, 10.2),
                score=0.7,
            ),
            0.1,
        ),
        (
            _cache(
                agent_id="vehicle",
                frame_id="000002",
                event_time=0.1,
                x_positions=(0.0, 10.0),
                score=0.95,
            ),
            0.1,
        ),
        (
            _cache(
                agent_id="roadside-a",
                frame_id="a-2",
                event_time=0.1,
                x_positions=(0.1, 10.1),
                score=0.8,
            ),
            0.1,
        ),
    )
    observations: set[tuple[tuple[str, ...], str]] = set()

    for permutation in itertools.permutations(updates):
        tracker = _new_tracker()
        tracker.ingest_batch(
            (
                (
                    _cache(
                        agent_id="vehicle",
                        frame_id="000001",
                        x_positions=(0.0, 10.0),
                    ),
                    0.0,
                ),
            )
        )
        tracker.commit(0.0)
        tracker.advance(0.1)
        results = tracker.ingest_batch(permutation)
        predictions = tracker.commit(0.1)
        observations.add(
            (
                tuple(result.cache_sha256 for result in results)
                + tuple(prediction.digest() for prediction in predictions),
                tracker.commit_records[-1].record_hash,
            )
        )

    assert len(observations) == 1


@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        (
            _cache(agent_id="roadside", frame_id="bad", sequence_id="other"),
            "sequence_mismatch",
        ),
        (
            _cache(agent_id="roadside", frame_id="bad", event_time=0.1),
            "future_event",
        ),
        (
            _cache(
                agent_id="roadside",
                frame_id="bad",
                coordinate_frame="agent",
            ),
            "coordinate_frame_mismatch",
        ),
    ],
)
def test_invalid_batch_is_rejected_before_any_tracker_mutation(
    invalid: DetectionCacheV1,
    error: str,
) -> None:
    tracker = _new_tracker()
    valid = _cache(agent_id="vehicle", frame_id="000001")

    with pytest.raises(ValueError, match=error):
        tracker.ingest_batch(((valid, 0.0), (invalid, 0.0)))

    assert tracker.active_track_ids == ()
    assert tracker.commit_records == ()
    assert tracker.ingest(valid, arrival_time=0.0).applied
    assert len(tracker.commit(0.0)) == 1


def test_conflicting_identity_and_reference_frames_are_atomic() -> None:
    tracker = _new_tracker()
    first = _cache(agent_id="vehicle", frame_id="000001", score=0.9)
    conflict = _cache(agent_id="vehicle", frame_id="000001", score=0.8)
    next_frame = _cache(agent_id="vehicle", frame_id="000002")

    with pytest.raises(ValueError, match="conflicting_cache_identity"):
        tracker.ingest_batch(((first, 0.0), (conflict, 0.0)))
    assert tracker.active_track_ids == ()

    with pytest.raises(ValueError, match="multiple reference-agent frames"):
        tracker.ingest_batch(((first, 0.0), (next_frame, 0.0)))
    assert tracker.active_track_ids == ()
    assert tracker.ingest(first, arrival_time=0.0).applied
