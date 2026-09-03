"""Lossless evaluator-input exchange for committed tracking predictions.

``EvaluatorInputArchiveV1`` is a neutral, canonical interchange layer.  It
preserves EventTrack state, covariance, existence, Top-H identity hypotheses,
lineage, and explicit empty frames.  The read-only row projection contains the
information normally needed by TrackEval or nuScenes adapter code, but this
module neither invokes either official evaluator nor computes metrics.  A
successful round trip therefore proves serialization invariants only; it is
not evidence of official evaluator parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Mapping

import numpy as np

from .contracts import TrackingPredictionV1
from .wire import canonical_json_bytes


_IDENTIFIER = re.compile(r"[^\s]+")


class EvaluatorInputArchiveError(ValueError):
    """Raised when evaluator exchange data is lossy or non-canonical."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise EvaluatorInputArchiveError(f"{name} must be a non-empty identifier")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluatorInputArchiveError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise EvaluatorInputArchiveError(f"{name} must be finite")
    return result


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise EvaluatorInputArchiveError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        raise EvaluatorInputArchiveError(f"{name} has missing or unknown fields")
    return value


@dataclass(frozen=True, slots=True)
class EvaluatorFrameScheduleEntryV1:
    """One explicit committed frame, including frames with zero predictions."""

    sequence_id: str
    frame_id: str
    decision_time: float

    _FIELDS = frozenset({"decision_time", "frame_id", "sequence_id"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence_id", _identifier(self.sequence_id, "sequence_id")
        )
        object.__setattr__(self, "frame_id", _identifier(self.frame_id, "frame_id"))
        object.__setattr__(
            self, "decision_time", _finite(self.decision_time, "decision_time")
        )

    @property
    def key(self) -> tuple[str, str, float]:
        return (self.sequence_id, self.frame_id, self.decision_time)

    def to_primitive(self) -> dict[str, object]:
        return {
            "decision_time": self.decision_time,
            "frame_id": self.frame_id,
            "sequence_id": self.sequence_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "EvaluatorFrameScheduleEntryV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            sequence_id=item["sequence_id"],
            frame_id=item["frame_id"],
            decision_time=item["decision_time"],
        )


def _validated_schedule(
    values: Iterable[EvaluatorFrameScheduleEntryV1],
) -> tuple[EvaluatorFrameScheduleEntryV1, ...]:
    entries = tuple(values)
    if not entries or not all(
        isinstance(item, EvaluatorFrameScheduleEntryV1) for item in entries
    ):
        raise EvaluatorInputArchiveError(
            "frame schedule must contain EvaluatorFrameScheduleEntryV1"
        )
    if len({item.key for item in entries}) != len(entries):
        raise EvaluatorInputArchiveError("frame schedule contains a duplicate frame")
    seen_frame_keys: set[tuple[str, str]] = set()
    previous: EvaluatorFrameScheduleEntryV1 | None = None
    closed_sequences: set[str] = set()
    for entry in entries:
        frame_key = (entry.sequence_id, entry.frame_id)
        if frame_key in seen_frame_keys:
            raise EvaluatorInputArchiveError(
                "frame schedule repeats a sequence/frame identifier"
            )
        seen_frame_keys.add(frame_key)
        if previous is not None:
            if entry.sequence_id == previous.sequence_id:
                if (
                    entry.frame_id <= previous.frame_id
                    or entry.decision_time <= previous.decision_time
                ):
                    raise EvaluatorInputArchiveError(
                        "frame/decision order must increase within each sequence"
                    )
            else:
                closed_sequences.add(previous.sequence_id)
                if (
                    entry.sequence_id <= previous.sequence_id
                    or entry.sequence_id in closed_sequences
                ):
                    raise EvaluatorInputArchiveError(
                        "sequence/frame order must be canonical and contiguous"
                    )
        previous = entry
    return entries


@dataclass(frozen=True, slots=True)
class EvaluatorInputFrameV1:
    """One canonical frame record; ``predictions`` may be explicitly empty."""

    sequence_id: str
    frame_id: str
    decision_time: float
    predictions: tuple[TrackingPredictionV1, ...]

    _FIELDS = frozenset(
        {"decision_time", "frame_id", "predictions", "sequence_id"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence_id", _identifier(self.sequence_id, "sequence_id")
        )
        object.__setattr__(self, "frame_id", _identifier(self.frame_id, "frame_id"))
        decision_time = _finite(self.decision_time, "decision_time")
        object.__setattr__(self, "decision_time", decision_time)
        if not isinstance(self.predictions, (list, tuple)):
            raise EvaluatorInputArchiveError("predictions must be an array")
        predictions = tuple(self.predictions)
        if not all(isinstance(item, TrackingPredictionV1) for item in predictions):
            raise EvaluatorInputArchiveError(
                "predictions must contain TrackingPredictionV1"
            )
        track_ids: list[str] = []
        for prediction in predictions:
            if not prediction.committed:
                raise EvaluatorInputArchiveError(
                    "evaluator input may contain only committed predictions"
                )
            if (
                prediction.sequence_id != self.sequence_id
                or prediction.frame_id != self.frame_id
                or prediction.decision_time != decision_time
            ):
                raise EvaluatorInputArchiveError(
                    "prediction does not match its sequence/frame/decision group"
                )
            track_ids.append(prediction.track_id)
        if track_ids != sorted(set(track_ids)):
            raise EvaluatorInputArchiveError(
                "each frame must have unique, sorted track identifiers"
            )
        object.__setattr__(self, "predictions", predictions)

    @property
    def schedule_entry(self) -> EvaluatorFrameScheduleEntryV1:
        return EvaluatorFrameScheduleEntryV1(
            self.sequence_id, self.frame_id, self.decision_time
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "decision_time": self.decision_time,
            "frame_id": self.frame_id,
            "predictions": [item.to_primitive() for item in self.predictions],
            "sequence_id": self.sequence_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "EvaluatorInputFrameV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        raw_predictions = item["predictions"]
        if not isinstance(raw_predictions, list):
            raise EvaluatorInputArchiveError("predictions must be an array")
        return cls(
            sequence_id=item["sequence_id"],
            frame_id=item["frame_id"],
            decision_time=item["decision_time"],
            predictions=tuple(
                TrackingPredictionV1.from_mapping(prediction)
                for prediction in raw_predictions
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluatorInputArchiveV1:
    """Lossless canonical exchange archive, not an evaluator result.

    ``content_sha256`` seals the payload without the seal field. ``digest()``
    hashes the final canonical sealed document; the two hashes are deliberately
    distinct evidence identifiers.
    """

    frames: tuple[EvaluatorInputFrameV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frames, (list, tuple)):
            raise EvaluatorInputArchiveError("frames must be an array")
        frames = tuple(self.frames)
        if not frames or not all(
            isinstance(item, EvaluatorInputFrameV1) for item in frames
        ):
            raise EvaluatorInputArchiveError(
                "frames must contain EvaluatorInputFrameV1"
            )
        _validated_schedule(frame.schedule_entry for frame in frames)
        object.__setattr__(self, "frames", frames)

    @property
    def frame_schedule(self) -> tuple[EvaluatorFrameScheduleEntryV1, ...]:
        return tuple(frame.schedule_entry for frame in self.frames)

    def payload(self) -> dict[str, object]:
        return {
            "frames": [frame.to_primitive() for frame in self.frames],
            "kind": "evaluator_input_archive_v1",
            "schema_version": 1,
        }

    @property
    def content_sha256(self) -> str:
        """Return the SHA-256 seal of the unsealed archive payload."""

        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()

    def sealed_document(self) -> dict[str, object]:
        return {**self.payload(), "content_sha256": self.content_sha256}

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.sealed_document())

    def digest(self) -> str:
        """Return the SHA-256 of the final canonical sealed document."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "EvaluatorInputArchiveV1":
        fields = frozenset(
            {"content_sha256", "frames", "kind", "schema_version"}
        )
        item = _strict_fields(value, fields, cls.__name__)
        if item["kind"] != "evaluator_input_archive_v1" or item[
            "schema_version"
        ] != 1:
            raise EvaluatorInputArchiveError("unsupported evaluator-input archive")
        if type(item["content_sha256"]) is not str or re.fullmatch(
            r"[0-9a-f]{64}", item["content_sha256"]
        ) is None:
            raise EvaluatorInputArchiveError(
                "content_sha256 must be a lowercase SHA-256"
            )
        payload = {key: item[key] for key in fields if key != "content_sha256"}
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if item["content_sha256"] != expected:
            raise EvaluatorInputArchiveError("archive content SHA-256 mismatch")
        raw_frames = item["frames"]
        if not isinstance(raw_frames, list):
            raise EvaluatorInputArchiveError("frames must be an array")
        return cls(
            frames=tuple(
                EvaluatorInputFrameV1.from_mapping(frame) for frame in raw_frames
            )
        )


def build_evaluator_input_archive_v1(
    predictions: Iterable[TrackingPredictionV1],
    frame_schedule: Iterable[EvaluatorFrameScheduleEntryV1],
) -> EvaluatorInputArchiveV1:
    """Build an archive from canonical committed predictions and a full schedule.

    Prediction frames must appear in schedule order and track identifiers must
    be sorted within a frame.  Reordering is rejected instead of being hidden by
    serialization.
    """

    schedule = _validated_schedule(frame_schedule)
    schedule_positions = {entry.key: index for index, entry in enumerate(schedule)}
    grouped: dict[tuple[str, str, float], list[TrackingPredictionV1]] = {
        entry.key: [] for entry in schedule
    }
    values = tuple(predictions)
    if not all(isinstance(item, TrackingPredictionV1) for item in values):
        raise EvaluatorInputArchiveError(
            "predictions must contain TrackingPredictionV1"
        )
    previous_position = -1
    previous_track_id: str | None = None
    for prediction in values:
        if not prediction.committed:
            raise EvaluatorInputArchiveError(
                "evaluator input may contain only committed predictions"
            )
        key = (
            prediction.sequence_id,
            prediction.frame_id,
            prediction.decision_time,
        )
        try:
            position = schedule_positions[key]
        except KeyError as exc:
            raise EvaluatorInputArchiveError(
                "prediction frame is missing from the explicit frame schedule"
            ) from exc
        if position < previous_position:
            raise EvaluatorInputArchiveError(
                "prediction frame/decision order moves backwards"
            )
        if position == previous_position:
            assert previous_track_id is not None
            if prediction.track_id <= previous_track_id:
                raise EvaluatorInputArchiveError(
                    "each frame must have unique, sorted track identifiers"
                )
        else:
            previous_track_id = None
        grouped[key].append(prediction)
        previous_position = position
        previous_track_id = prediction.track_id
    return EvaluatorInputArchiveV1(
        frames=tuple(
            EvaluatorInputFrameV1(
                sequence_id=entry.sequence_id,
                frame_id=entry.frame_id,
                decision_time=entry.decision_time,
                predictions=tuple(grouped[entry.key]),
            )
            for entry in schedule
        )
    )


def validate_evaluator_input_archive_v1(
    archive: EvaluatorInputArchiveV1,
    expected_frame_schedule: Iterable[EvaluatorFrameScheduleEntryV1],
) -> None:
    """Reject any missing, extra, repeated, or reordered expected frame."""

    if not isinstance(archive, EvaluatorInputArchiveV1):
        raise TypeError("archive must be EvaluatorInputArchiveV1")
    expected = _validated_schedule(expected_frame_schedule)
    if archive.frame_schedule != expected:
        raise EvaluatorInputArchiveError(
            "archive frame schedule is missing, extra, or reordered"
        )


def evaluator_input_archive_to_predictions_v1(
    archive: EvaluatorInputArchiveV1,
    *,
    expected_frame_schedule: Iterable[EvaluatorFrameScheduleEntryV1] | None = None,
) -> tuple[TrackingPredictionV1, ...]:
    """Recover committed predictions without losing explicit archive frames."""

    if not isinstance(archive, EvaluatorInputArchiveV1):
        raise TypeError("archive must be EvaluatorInputArchiveV1")
    if expected_frame_schedule is not None:
        validate_evaluator_input_archive_v1(archive, expected_frame_schedule)
    return tuple(
        prediction for frame in archive.frames for prediction in frame.predictions
    )


@dataclass(frozen=True, slots=True)
class NeutralEvaluatorTrackRowV1:
    """Read-only tracker row for downstream adapter code."""

    track_id: str
    class_label: str
    event_time: float
    arrival_time: float
    state: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    existence_probability: float
    identity_hypotheses: tuple[tuple[str, float], ...]
    other_identity_probability: float
    lineage_factor_ids: tuple[str, ...]
    lineage_complete: bool
    lineage_ancestor_message_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NeutralEvaluatorFrameRowV1:
    """Read-only neutral frame; an empty ``tracks`` tuple preserves empty frames."""

    sequence_id: str
    frame_id: str
    decision_time: float
    tracks: tuple[NeutralEvaluatorTrackRowV1, ...]


def to_neutral_trackeval_nuscenes_input_rows_v1(
    archive: EvaluatorInputArchiveV1,
) -> tuple[NeutralEvaluatorFrameRowV1, ...]:
    """Project neutral read-only input rows for future evaluator adapters.

    No TrackEval or nuScenes metric is computed here, and this function does not
    establish parity with either official implementation.
    """

    if not isinstance(archive, EvaluatorInputArchiveV1):
        raise TypeError("archive must be EvaluatorInputArchiveV1")
    return tuple(
        NeutralEvaluatorFrameRowV1(
            sequence_id=frame.sequence_id,
            frame_id=frame.frame_id,
            decision_time=frame.decision_time,
            tracks=tuple(
                NeutralEvaluatorTrackRowV1(
                    track_id=prediction.track_id,
                    class_label=prediction.class_label,
                    event_time=prediction.event_time,
                    arrival_time=prediction.arrival_time,
                    state=tuple(float(value) for value in prediction.mean),
                    covariance=tuple(
                        tuple(float(value) for value in row)
                        for row in prediction.covariance
                    ),
                    existence_probability=prediction.existence_probability,
                    identity_hypotheses=tuple(
                        (item.identity_id, item.probability)
                        for item in prediction.identity_hypotheses
                    ),
                    other_identity_probability=(
                        prediction.other_identity_probability
                    ),
                    lineage_factor_ids=prediction.lineage.factor_ids,
                    lineage_complete=prediction.lineage.complete,
                    lineage_ancestor_message_ids=(
                        prediction.lineage.ancestor_message_ids
                    ),
                )
                for prediction in frame.predictions
            ),
        )
        for frame in archive.frames
    )


def decode_evaluator_input_archive(data: bytes) -> EvaluatorInputArchiveV1:
    """Decode canonical archive bytes and reject duplicate JSON keys."""

    if type(data) is not bytes:
        raise TypeError("archive data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise EvaluatorInputArchiveError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                EvaluatorInputArchiveError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluatorInputArchiveError("invalid EvaluatorInputArchiveV1 JSON") from exc
    archive = EvaluatorInputArchiveV1.from_mapping(value)
    if archive.canonical_bytes() != data:
        raise EvaluatorInputArchiveError(
            "EvaluatorInputArchiveV1 JSON is not canonical"
        )
    return archive


__all__ = [
    "EvaluatorFrameScheduleEntryV1",
    "EvaluatorInputArchiveError",
    "EvaluatorInputArchiveV1",
    "EvaluatorInputFrameV1",
    "NeutralEvaluatorFrameRowV1",
    "NeutralEvaluatorTrackRowV1",
    "build_evaluator_input_archive_v1",
    "decode_evaluator_input_archive",
    "evaluator_input_archive_to_predictions_v1",
    "to_neutral_trackeval_nuscenes_input_rows_v1",
    "validate_evaluator_input_archive_v1",
]
