from dataclasses import FrozenInstanceError
import hashlib

import numpy as np
import pytest

from transvision.models.event_track_v2x.contracts import (
    IdentityHypothesisV1,
    TrackingPredictionV1,
)
from transvision.models.event_track_v2x.evaluator_adapter import (
    EvaluatorFrameScheduleEntryV1,
    EvaluatorInputArchiveError,
    EvaluatorInputArchiveV1,
    build_evaluator_input_archive_v1,
    decode_evaluator_input_archive,
    evaluator_input_archive_to_predictions_v1,
    to_neutral_trackeval_nuscenes_input_rows_v1,
    validate_evaluator_input_archive_v1,
)
from transvision.models.event_track_v2x.schema import Lineage
from transvision.models.event_track_v2x.wire import canonical_json_bytes


SCHEDULE = (
    EvaluatorFrameScheduleEntryV1("0087", "000001", 1.2),
    EvaluatorFrameScheduleEntryV1("0087", "000002", 2.2),
)


def _prediction(
    *,
    frame_id: str = "000001",
    track_id: str = "track-1",
    event_time: float = 1.0,
    arrival_time: float = 1.1,
    decision_time: float = 1.2,
    committed: bool = True,
) -> TrackingPredictionV1:
    return TrackingPredictionV1(
        sequence_id="0087",
        frame_id=frame_id,
        track_id=track_id,
        class_label="car",
        event_time=event_time,
        arrival_time=arrival_time,
        decision_time=decision_time,
        mean=np.arange(9, dtype=np.float64) + (0.0 if track_id == "track-1" else 1.0),
        covariance=np.eye(9, dtype=np.float64) * (
            1.0 if track_id == "track-1" else 2.0
        ),
        existence_probability=0.9,
        identity_hypotheses=(
            IdentityHypothesisV1(f"identity-{track_id}", 0.75),
            IdentityHypothesisV1("identity-other", 0.20),
        ),
        other_identity_probability=0.05,
        lineage=Lineage(
            (f"factor-{track_id}",),
            True,
            (f"message-{track_id}",),
        ),
        committed=committed,
    )


def _golden_archive() -> EvaluatorInputArchiveV1:
    return build_evaluator_input_archive_v1(
        (_prediction(track_id="track-1"), _prediction(track_id="track-2")),
        SCHEDULE,
    )


def test_golden_two_frame_archive_is_lossless_numeric_and_hash_stable() -> None:
    archive = _golden_archive()
    decoded = decode_evaluator_input_archive(archive.canonical_bytes())
    restored = evaluator_input_archive_to_predictions_v1(
        decoded, expected_frame_schedule=SCHEDULE
    )

    assert decoded == archive
    assert restored == (
        _prediction(track_id="track-1"),
        _prediction(track_id="track-2"),
    )
    assert len(decoded.frames) == 2
    assert decoded.frames[1].predictions == ()
    assert restored[0].mean.tolist() == [float(index) for index in range(9)]
    assert restored[1].covariance.tolist() == (np.eye(9) * 2.0).tolist()
    assert restored[0].identity_hypotheses[0].probability == 0.75
    assert restored[0].lineage.ancestor_message_ids == ("message-track-1",)
    assert (
        archive.content_sha256
        == "f308268e657f6bd09b2ce84caa57a4f16cf63ee551956ca1ee3821c55310e953"
    )
    assert (
        archive.digest()
        == "30203b4e29c0c941583a2b8c854d698b3168a9785cab03fba11167239e74fa0f"
    )
    assert _golden_archive().canonical_bytes() == archive.canonical_bytes()


def test_neutral_rows_preserve_empty_frames_and_are_read_only() -> None:
    rows = to_neutral_trackeval_nuscenes_input_rows_v1(_golden_archive())
    assert len(rows) == 2
    assert rows[0].sequence_id == "0087"
    assert tuple(track.track_id for track in rows[0].tracks) == (
        "track-1",
        "track-2",
    )
    assert rows[0].tracks[0].state == tuple(float(index) for index in range(9))
    assert rows[1].tracks == ()
    with pytest.raises(FrozenInstanceError):
        rows[0].frame_id = "tampered"  # type: ignore[misc]


def test_builder_rejects_duplicate_track_ids_and_uncommitted_predictions() -> None:
    prediction = _prediction()
    with pytest.raises(EvaluatorInputArchiveError, match="unique, sorted"):
        build_evaluator_input_archive_v1((prediction, prediction), SCHEDULE)
    with pytest.raises(EvaluatorInputArchiveError, match="only committed"):
        build_evaluator_input_archive_v1((_prediction(committed=False),), SCHEDULE)


def test_builder_rejects_prediction_frame_missing_from_explicit_schedule() -> None:
    with pytest.raises(EvaluatorInputArchiveError, match="missing from"):
        build_evaluator_input_archive_v1(
            (
                _prediction(
                    frame_id="000002",
                    event_time=2.0,
                    arrival_time=2.1,
                    decision_time=2.2,
                ),
            ),
            (SCHEDULE[0],),
        )


def test_schedule_and_prediction_frame_decision_order_cannot_move_backwards() -> None:
    with pytest.raises(EvaluatorInputArchiveError, match="frame/decision order"):
        build_evaluator_input_archive_v1((), tuple(reversed(SCHEDULE)))
    with pytest.raises(EvaluatorInputArchiveError, match="frame/decision order"):
        build_evaluator_input_archive_v1(
            (),
            (
                EvaluatorFrameScheduleEntryV1("0087", "000001", 2.2),
                EvaluatorFrameScheduleEntryV1("0087", "000002", 1.2),
            ),
        )
    with pytest.raises(EvaluatorInputArchiveError, match="moves backwards"):
        build_evaluator_input_archive_v1(
            (
                _prediction(
                    frame_id="000002",
                    track_id="track-2",
                    event_time=2.0,
                    arrival_time=2.1,
                    decision_time=2.2,
                ),
                _prediction(track_id="track-1"),
            ),
            SCHEDULE,
        )


def test_explicit_schedule_validator_detects_missing_empty_frame() -> None:
    archive_without_second_frame = build_evaluator_input_archive_v1(
        (_prediction(),), (SCHEDULE[0],)
    )
    with pytest.raises(EvaluatorInputArchiveError, match="missing, extra"):
        validate_evaluator_input_archive_v1(archive_without_second_frame, SCHEDULE)


def test_decoder_rejects_duplicate_unknown_nonfinite_and_noncanonical_json() -> None:
    archive = _golden_archive()
    duplicate = archive.canonical_bytes().replace(
        b'{"content_sha256":',
        b'{"content_sha256":"0","content_sha256":',
        1,
    )
    with pytest.raises(EvaluatorInputArchiveError, match="duplicate JSON key"):
        decode_evaluator_input_archive(duplicate)

    document = archive.sealed_document()
    document["unknown"] = True
    with pytest.raises(EvaluatorInputArchiveError, match="missing or unknown"):
        EvaluatorInputArchiveV1.from_mapping(document)

    nonfinite = archive.canonical_bytes().replace(
        b'"decision_time":1.2', b'"decision_time":NaN', 1
    )
    with pytest.raises(EvaluatorInputArchiveError, match="non-finite JSON"):
        decode_evaluator_input_archive(nonfinite)

    noncanonical = b" " + archive.canonical_bytes()
    with pytest.raises(EvaluatorInputArchiveError, match="not canonical"):
        decode_evaluator_input_archive(noncanonical)


def test_resealed_duplicate_track_id_is_rejected_during_decode() -> None:
    document = _golden_archive().payload()
    first_prediction = document["frames"][0]["predictions"][0]  # type: ignore[index]
    document["frames"][0]["predictions"].append(first_prediction)  # type: ignore[index, union-attr]
    document["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    with pytest.raises(EvaluatorInputArchiveError, match="unique, sorted"):
        decode_evaluator_input_archive(canonical_json_bytes(document))
