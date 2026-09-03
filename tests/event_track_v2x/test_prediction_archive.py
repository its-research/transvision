from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from transvision.models.event_track_v2x.commit import AppendOnlyCommitLog, CommitRecord
from transvision.models.event_track_v2x.contracts import (
    IdentityHypothesisV1,
    TrackingPredictionV1,
)
from transvision.models.event_track_v2x.prediction_archive import (
    PredictionArchiveError,
    build_prediction_archive,
    verify_prediction_archive,
)
from transvision.models.event_track_v2x.schema import Lineage
from transvision.models.event_track_v2x.wire import canonical_json_bytes


BINDINGS = {
    "experiment_plan_sha256": "a" * 64,
    "cohort_sha256": "f" * 64,
    "detector_cache_sha256": "b" * 64,
    "network_trace_sha256": "c" * 64,
    "tracker_config_sha256": "d" * 64,
    "evaluator_sha256": "e" * 64,
}
DEFAULT_EXPECTED_FRAMES = [("0087", "000001", 1.2)]


def _prediction(
    frame_id: str = "000001",
    track_id: str = "track-1",
    *,
    sequence_id: str = "0087",
    event_time: float = 1.0,
    arrival_time: float = 1.1,
    decision_time: float = 1.2,
    committed: bool = True,
) -> TrackingPredictionV1:
    return TrackingPredictionV1(
        sequence_id=sequence_id,
        frame_id=frame_id,
        track_id=track_id,
        class_label="car",
        event_time=event_time,
        arrival_time=arrival_time,
        decision_time=decision_time,
        mean=np.arange(9, dtype=np.float64),
        covariance=np.eye(9),
        existence_probability=0.9,
        identity_hypotheses=(IdentityHypothesisV1(track_id, 1.0),),
        lineage=Lineage((f"factor-{track_id}",), True),
        committed=committed,
    )


def _build(
    directory: Path,
    predictions: list[TrackingPredictionV1],
    *,
    expected_frames: list[tuple[str, str, float]] | None = None,
    commit_records: tuple[CommitRecord, ...] | None = None,
) -> str:
    if expected_frames is None:
        expected_frames = sorted(
            {
                (item.sequence_id, item.frame_id, item.decision_time)
                for item in predictions
            }
        )
    if commit_records is None:
        log = AppendOnlyCommitLog()
        for sequence_id, frame_id, decision_time in sorted(
            expected_frames, key=lambda item: item[2]
        ):
            snapshot = [
                item.to_primitive()
                for item in sorted(
                    (
                        prediction
                        for prediction in predictions
                        if prediction.sequence_id == sequence_id
                        and prediction.frame_id == frame_id
                        and prediction.decision_time == decision_time
                    ),
                    key=lambda item: item.track_id,
                )
            ]
            log.commit(decision_time=decision_time, snapshot=snapshot)
        commit_records = log.records
    return build_prediction_archive(
        predictions,
        commit_records,
        directory,
        expected_frames=expected_frames,
        archive_id="eventtrack-test-c0-seed1001",
        **BINDINGS,
    )


def _tree_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _verify(
    directory: Path,
    *,
    expected_frames: list[tuple[str, str, float]] | None = None,
    expected_bindings: dict[str, str] | None = None,
):
    return verify_prediction_archive(
        directory,
        expected_frames=(
            DEFAULT_EXPECTED_FRAMES if expected_frames is None else expected_frames
        ),
        expected_bindings=expected_bindings,
    )


def _reseal_frame(directory: Path, mutate: object) -> None:
    frame_path = next((directory / "frames").rglob("*.json"))
    frame = json.loads(frame_path.read_bytes())
    mutate(frame)
    raw = canonical_json_bytes(frame)
    frame_path.write_bytes(raw)

    manifest_path = directory / "prediction-archive-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    entry = manifest["entries"][0]
    entry["byte_count"] = len(raw)
    entry["sha256"] = hashlib.sha256(raw).hexdigest()
    entry["prediction_count"] = len(frame["predictions"])
    track_ids = [item["track_id"] for item in frame["predictions"]]
    entry["track_ids_sha256"] = hashlib.sha256(
        canonical_json_bytes(track_ids)
    ).hexdigest()
    manifest["prediction_count"] = len(frame["predictions"])
    payload = {key: value for key, value in manifest.items() if key != "content_sha256"}
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def test_prediction_archive_is_deterministic_bound_and_gt_free(
    tmp_path: Path,
) -> None:
    predictions = [
        _prediction(
            "000002",
            "track-3",
            event_time=2.0,
            arrival_time=2.1,
            decision_time=2.2,
        ),
        _prediction("000001", "track-2"),
        _prediction("000001", "track-1"),
    ]
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_digest = _build(first, predictions)
    second_digest = _build(second, list(reversed(predictions)))

    assert first_digest == second_digest
    assert _tree_bytes(first) == _tree_bytes(second)
    observed, decoded = _verify(
        first,
        expected_frames=[
            ("0087", "000001", 1.2),
            ("0087", "000002", 2.2),
        ],
        expected_bindings=BINDINGS,
    )
    assert observed == first_digest
    assert [(item.frame_id, item.track_id) for item in decoded] == [
        ("000001", "track-1"),
        ("000001", "track-2"),
        ("000002", "track-3"),
    ]
    assert all(item.committed for item in decoded)

    manifest = json.loads(
        (first / "prediction-archive-manifest.json").read_bytes()
    )
    assert manifest["ground_truth_included"] is False
    assert manifest["prediction_count"] == 3
    assert manifest["commit_chain_tip_sha256"] == manifest["commit_records"][-1][
        "record_hash"
    ]
    assert {name: manifest[name] for name in BINDINGS} == BINDINGS
    for path in first.rglob("*.json"):
        assert b"ground_truth\":" not in path.read_bytes()


def test_prediction_archive_rejects_uncommitted_duplicate_and_future_records(
    tmp_path: Path,
) -> None:
    with pytest.raises(PredictionArchiveError, match="not committed"):
        _build(
            tmp_path / "uncommitted",
            [_prediction(committed=False)],
            expected_frames=[("0087", "000001", 1.2)],
        )

    duplicate = _prediction()
    with pytest.raises(PredictionArchiveError, match="duplicate"):
        _build(tmp_path / "duplicate", [duplicate, duplicate])

    future = _prediction()
    object.__setattr__(future, "arrival_time", future.decision_time + 1.0)
    with pytest.raises(PredictionArchiveError, match="invalid prediction"):
        _build(
            tmp_path / "future",
            [future],
            expected_frames=[("0087", "000001", 1.2)],
        )

    with pytest.raises(PredictionArchiveError, match="identities|decision time"):
        _build(
            tmp_path / "decision-drift",
            [
                _prediction(track_id="track-1"),
                _prediction(track_id="track-2", decision_time=1.3),
            ],
        )


def test_prediction_archive_preserves_zero_prediction_frames(tmp_path: Path) -> None:
    destination = tmp_path / "empty-frame"
    digest = _build(
        destination,
        [],
        expected_frames=[("0087", "000001", 1.2)],
    )
    observed, predictions = _verify(
        destination, expected_frames=[("0087", "000001", 1.2)]
    )
    assert observed == digest
    assert predictions == ()
    manifest = json.loads(
        (destination / "prediction-archive-manifest.json").read_bytes()
    )
    assert manifest["frame_count"] == 1
    assert manifest["prediction_count"] == 0
    assert manifest["entries"][0]["prediction_count"] == 0


def test_prediction_archive_rejects_committed_flag_without_matching_commit_snapshot(
    tmp_path: Path,
) -> None:
    log = AppendOnlyCommitLog()
    log.commit(decision_time=1.2, snapshot=[])
    with pytest.raises(PredictionArchiveError, match="commit snapshot"):
        _build(
            tmp_path / "fabricated-committed",
            [_prediction()],
            expected_frames=[("0087", "000001", 1.2)],
            commit_records=log.records,
        )


@pytest.mark.parametrize(
    "attack", ["tamper", "extra", "extra-directory", "manifest-space"]
)
def test_prediction_archive_rejects_file_set_and_byte_mutation(
    tmp_path: Path, attack: str
) -> None:
    destination = tmp_path / "archive"
    _build(destination, [_prediction()])
    if attack == "tamper":
        frame_path = next((destination / "frames").rglob("*.json"))
        frame_path.write_bytes(frame_path.read_bytes() + b" ")
    elif attack == "extra":
        (destination / "ground-truth.json").write_text("{}")
    elif attack == "extra-directory":
        (destination / "unsealed-empty-directory").mkdir()
    else:
        manifest = destination / "prediction-archive-manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(PredictionArchiveError):
        _verify(destination)


@pytest.mark.parametrize("attack", ["future", "uncommitted", "duplicate", "gt"])
def test_prediction_archive_rejects_semantic_attack_after_resealing(
    tmp_path: Path, attack: str
) -> None:
    destination = tmp_path / "archive"
    _build(destination, [_prediction()])

    def mutate(frame: dict[str, object]) -> None:
        predictions = frame["predictions"]
        assert isinstance(predictions, list)
        if attack == "future":
            predictions[0]["arrival_time"] = predictions[0]["decision_time"] + 1.0
        elif attack == "uncommitted":
            predictions[0]["committed"] = False
        elif attack == "duplicate":
            predictions.append(dict(predictions[0]))
        else:
            predictions[0]["ground_truth"] = "forbidden"

    _reseal_frame(destination, mutate)
    with pytest.raises(PredictionArchiveError):
        _verify(destination)


def test_prediction_archive_rejects_binding_substitution(tmp_path: Path) -> None:
    destination = tmp_path / "archive"
    _build(destination, [_prediction()])
    mismatched = {**BINDINGS, "network_trace_sha256": "f" * 64}
    with pytest.raises(PredictionArchiveError, match="network_trace_sha256"):
        _verify(destination, expected_bindings=mismatched)


def test_prediction_archive_rejects_symbolic_links(tmp_path: Path) -> None:
    destination = tmp_path / "archive"
    _build(destination, [_prediction()])
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    try:
        (destination / "leak").symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable on this filesystem")

    with pytest.raises(PredictionArchiveError, match="symbolic link"):
        _verify(destination)


def test_prediction_archive_verifier_rejects_self_consistent_frame_subset(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "subset"
    _build(destination, [_prediction()])
    with pytest.raises(PredictionArchiveError, match="frozen expected_frames"):
        _verify(
            destination,
            expected_frames=[
                ("0087", "000001", 1.2),
                ("0087", "000002", 2.2),
            ],
        )
