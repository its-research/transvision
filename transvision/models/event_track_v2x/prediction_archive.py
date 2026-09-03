"""Deterministic, ground-truth-free archives for committed tracker outputs.

The archive is a final experiment artifact.  It therefore accepts only
``TrackingPredictionV1`` records that have already been committed, writes one
canonical JSON document per sequence/frame, and seals the exact file set in a
canonical manifest.  Verification starts from bytes on disk and rejects
semantic invalidity even when an attacker recomputes the unkeyed SHA-256 seals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable, Mapping

from .commit import GENESIS_HASH, CommitRecord
from .contracts import TrackingPredictionV1, encode_contract
from .wire import canonical_json_bytes


_MANIFEST_NAME = "prediction-archive-manifest.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PATH_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_BINDING_FIELDS = (
    "experiment_plan_sha256",
    "cohort_sha256",
    "detector_cache_sha256",
    "network_trace_sha256",
    "tracker_config_sha256",
    "evaluator_sha256",
)


class PredictionArchiveError(ValueError):
    """Raised when a prediction archive is unsafe or fails its contract."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PredictionArchiveError(f"{name} must be a lowercase SHA-256")
    return value


def _path_identifier(value: object, name: str) -> str:
    if type(value) is not str or _PATH_IDENTIFIER.fullmatch(value) is None:
        raise PredictionArchiveError(
            f"{name} must be a non-empty ASCII path-safe identifier"
        )
    if value in {".", ".."}:
        raise PredictionArchiveError(f"{name} is not a safe identifier")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise PredictionArchiveError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionArchiveError(f"{name} must be a finite number") from exc
    if not -float("inf") < result < float("inf"):
        raise PredictionArchiveError(f"{name} must be a finite number")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PredictionArchiveError(f"{name} must be a non-negative integer")
    return value


def _load_json(data: bytes, name: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise PredictionArchiveError(
                    f"duplicate JSON key in {name}: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PredictionArchiveError(
            f"non-finite JSON constant in {name}: {value}"
        )

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionArchiveError(f"invalid JSON in {name}") from exc


def _prediction_from_mapping(value: object, name: str) -> TrackingPredictionV1:
    try:
        prediction = TrackingPredictionV1.from_mapping(value)
    except (TypeError, ValueError) as exc:
        raise PredictionArchiveError(f"invalid prediction in {name}: {exc}") from exc
    primitive = prediction.to_primitive()
    if encode_contract(prediction) != canonical_json_bytes(value):
        raise PredictionArchiveError(f"prediction in {name} is not canonical")
    if primitive["event_time"] > primitive["decision_time"]:
        raise PredictionArchiveError(
            f"prediction in {name} uses future event information"
        )
    if primitive["arrival_time"] > primitive["decision_time"]:
        raise PredictionArchiveError(
            f"prediction in {name} arrives after its decision"
        )
    if prediction.committed is not True:
        raise PredictionArchiveError(
            f"prediction in {name} is not committed and cannot be archived"
        )
    return prediction


@dataclass(frozen=True, slots=True)
class PredictionArchiveIndexEntry:
    sequence_id: str
    frame_id: str
    decision_time: float
    relative_path: str
    byte_count: int
    sha256: str
    prediction_count: int
    track_ids_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_id",
            _path_identifier(self.sequence_id, "sequence_id"),
        )
        object.__setattr__(
            self, "frame_id", _path_identifier(self.frame_id, "frame_id")
        )
        object.__setattr__(
            self, "decision_time", _finite(self.decision_time, "decision_time")
        )
        if type(self.relative_path) is not str:
            raise PredictionArchiveError("relative_path must be a string")
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_path
            or path.suffix != ".json"
        ):
            raise PredictionArchiveError(
                "relative_path must be a canonical relative JSON path"
            )
        expected_path = f"frames/{self.sequence_id}/{self.frame_id}.json"
        if self.relative_path != expected_path:
            raise PredictionArchiveError("relative_path disagrees with frame identity")
        object.__setattr__(
            self, "byte_count", _nonnegative_int(self.byte_count, "byte_count")
        )
        if self.byte_count == 0:
            raise PredictionArchiveError("prediction frame must not be empty")
        object.__setattr__(
            self,
            "prediction_count",
            _nonnegative_int(self.prediction_count, "prediction_count"),
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        object.__setattr__(
            self,
            "track_ids_sha256",
            _sha256(self.track_ids_sha256, "track_ids_sha256"),
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "decision_time": self.decision_time,
            "frame_id": self.frame_id,
            "prediction_count": self.prediction_count,
            "relative_path": self.relative_path,
            "sequence_id": self.sequence_id,
            "sha256": self.sha256,
            "track_ids_sha256": self.track_ids_sha256,
        }


def _entry_from_mapping(value: object) -> PredictionArchiveIndexEntry:
    expected = {
        "byte_count",
        "decision_time",
        "frame_id",
        "prediction_count",
        "relative_path",
        "sequence_id",
        "sha256",
        "track_ids_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise PredictionArchiveError(
            "prediction index entry has missing or unknown fields"
        )
    return PredictionArchiveIndexEntry(**value)


def _validate_bindings(values: Mapping[str, object]) -> dict[str, str]:
    if set(values) != set(_BINDING_FIELDS):
        missing = sorted(set(_BINDING_FIELDS) - set(values))
        unknown = sorted(set(values) - set(_BINDING_FIELDS))
        raise PredictionArchiveError(
            f"archive bindings mismatch; missing={missing}, unknown={unknown}"
        )
    return {name: _sha256(values[name], name) for name in _BINDING_FIELDS}


def _frame_document(
    sequence_id: str,
    frame_id: str,
    decision_time: float,
    predictions: tuple[TrackingPredictionV1, ...],
) -> dict[str, object]:
    return {
        "decision_time": decision_time,
        "frame_id": frame_id,
        "ground_truth_included": False,
        "kind": "tracking_prediction_frame_v1",
        "predictions": [prediction.to_primitive() for prediction in predictions],
        "schema_version": 1,
        "sequence_id": sequence_id,
    }


def _frame_contract(
    expected_frames: Iterable[tuple[str, str, float]],
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for index, item in enumerate(expected_frames):
        if not isinstance(item, tuple) or len(item) != 3:
            raise PredictionArchiveError(
                f"expected frame {index} must be (sequence_id, frame_id, decision_time)"
            )
        sequence_id = _path_identifier(item[0], "sequence_id")
        frame_id = _path_identifier(item[1], "frame_id")
        decision_time = _finite(item[2], "decision_time")
        key = (sequence_id, frame_id)
        if key in result:
            raise PredictionArchiveError("expected frame identities must be unique")
        result[key] = decision_time
    if not result:
        raise PredictionArchiveError("expected_frames must contain the frozen cohort")
    if len({key[0] for key in result}) != 1:
        raise PredictionArchiveError(
            "one prediction archive must contain exactly one sequence"
        )
    decision_times = tuple(result.values())
    if len(set(decision_times)) != len(decision_times):
        raise PredictionArchiveError(
            "prediction archive frame decision times must be unique"
        )
    return result


def _commit_record_primitive(record: CommitRecord) -> dict[str, object]:
    if not isinstance(record, CommitRecord):
        raise PredictionArchiveError("commit_records must contain CommitRecord")
    if type(record.index) is not int or record.index < 0:
        raise PredictionArchiveError("commit record index must be non-negative")
    decision_time = _finite(record.decision_time, "commit decision_time")
    if not isinstance(record.snapshot_bytes, bytes):
        raise PredictionArchiveError("commit snapshot_bytes must be bytes")
    try:
        snapshot = _load_json(record.snapshot_bytes, "commit snapshot")
    except PredictionArchiveError as exc:
        raise PredictionArchiveError("commit snapshot is not canonical JSON") from exc
    if canonical_json_bytes(snapshot) != record.snapshot_bytes:
        raise PredictionArchiveError("commit snapshot is not canonical JSON")
    if not isinstance(record.message_ids, tuple):
        raise PredictionArchiveError("commit message_ids must be a tuple")
    message_ids = tuple(
        _path_identifier(item, "commit message_id") for item in record.message_ids
    )
    if message_ids != tuple(sorted(set(message_ids))):
        raise PredictionArchiveError("commit message_ids must be sorted and unique")
    return {
        "decision_time": decision_time,
        "index": record.index,
        "message_ids": list(message_ids),
        "previous_hash": _sha256(record.previous_hash, "previous_hash"),
        "record_hash": _sha256(record.record_hash, "record_hash"),
        "snapshot_sha256": _digest(record.snapshot_bytes),
    }


def _validate_commit_chain_primitives(
    values: object,
) -> tuple[dict[str, object], ...]:
    fields = {
        "decision_time",
        "index",
        "message_ids",
        "previous_hash",
        "record_hash",
        "snapshot_sha256",
    }
    if not isinstance(values, (list, tuple)) or not values:
        raise PredictionArchiveError("commit_records must be a non-empty array")
    result: list[dict[str, object]] = []
    previous_hash = GENESIS_HASH
    previous_time = -float("inf")
    for expected_index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) != fields:
            raise PredictionArchiveError(
                "commit record has missing or unknown fields"
            )
        index = value["index"]
        if type(index) is not int or index != expected_index:
            raise PredictionArchiveError("commit record indices must be contiguous")
        decision_time = _finite(value["decision_time"], "commit decision_time")
        if decision_time <= previous_time:
            raise PredictionArchiveError(
                "commit decision times must be strictly increasing"
            )
        message_values = value["message_ids"]
        if not isinstance(message_values, (list, tuple)):
            raise PredictionArchiveError("commit message_ids must be an array")
        message_ids = tuple(
            _path_identifier(item, "commit message_id") for item in message_values
        )
        if message_ids != tuple(sorted(set(message_ids))):
            raise PredictionArchiveError(
                "commit message_ids must be sorted and unique"
            )
        observed_previous = _sha256(value["previous_hash"], "previous_hash")
        if observed_previous != previous_hash:
            raise PredictionArchiveError("commit chain previous hash mismatch")
        snapshot_sha256 = _sha256(value["snapshot_sha256"], "snapshot_sha256")
        body = canonical_json_bytes(
            {
                "decision_time": decision_time,
                "index": index,
                "message_ids": message_ids,
                "previous_hash": observed_previous,
                "snapshot_sha256": snapshot_sha256,
            }
        )
        record_hash = _sha256(value["record_hash"], "record_hash")
        if _digest(body) != record_hash:
            raise PredictionArchiveError("commit record hash mismatch")
        primitive = {
            "decision_time": decision_time,
            "index": index,
            "message_ids": list(message_ids),
            "previous_hash": observed_previous,
            "record_hash": record_hash,
            "snapshot_sha256": snapshot_sha256,
        }
        result.append(primitive)
        previous_hash = record_hash
        previous_time = decision_time
    return tuple(result)


def build_prediction_archive(
    predictions: Iterable[TrackingPredictionV1],
    commit_records: Iterable[CommitRecord],
    output_directory: Path,
    *,
    expected_frames: Iterable[tuple[str, str, float]],
    archive_id: str,
    experiment_plan_sha256: str,
    cohort_sha256: str,
    detector_cache_sha256: str,
    network_trace_sha256: str,
    tracker_config_sha256: str,
    evaluator_sha256: str,
) -> str:
    """Create an immutable final prediction archive and return its seal."""

    archive_id = _path_identifier(archive_id, "archive_id")
    bindings = _validate_bindings(
        {
            "experiment_plan_sha256": experiment_plan_sha256,
            "cohort_sha256": cohort_sha256,
            "detector_cache_sha256": detector_cache_sha256,
            "network_trace_sha256": network_trace_sha256,
            "tracker_config_sha256": tracker_config_sha256,
            "evaluator_sha256": evaluator_sha256,
        }
    )
    validated: list[TrackingPredictionV1] = []
    for index, item in enumerate(predictions):
        if not isinstance(item, TrackingPredictionV1):
            raise PredictionArchiveError(
                f"prediction {index} must be TrackingPredictionV1"
            )
        prediction = _prediction_from_mapping(
            item.to_primitive(), f"prediction {index}"
        )
        _path_identifier(prediction.sequence_id, "sequence_id")
        _path_identifier(prediction.frame_id, "frame_id")
        validated.append(prediction)
    frame_contract = _frame_contract(expected_frames)

    keys = [
        (item.sequence_id, item.frame_id, item.track_id) for item in validated
    ]
    if len(set(keys)) != len(keys):
        raise PredictionArchiveError(
            "duplicate sequence/frame/track prediction in final archive"
        )
    validated.sort(
        key=lambda item: (
            item.sequence_id,
            item.decision_time,
            item.frame_id,
            item.track_id,
        )
    )

    grouped: dict[tuple[str, str], list[TrackingPredictionV1]] = {
        key: [] for key in frame_contract
    }
    for prediction in validated:
        key = (prediction.sequence_id, prediction.frame_id)
        if key not in frame_contract:
            raise PredictionArchiveError("prediction frame is outside the frozen cohort")
        if prediction.decision_time != frame_contract[key]:
            raise PredictionArchiveError("prediction decision time violates frame contract")
        grouped[key].append(prediction)
    for key, group in grouped.items():
        if group and len({item.decision_time for item in group}) != 1:
            raise PredictionArchiveError(
                f"frame {key[0]}/{key[1]} has multiple decision times"
            )

    raw_commit_records = tuple(commit_records)
    commit_primitives = _validate_commit_chain_primitives(
        tuple(_commit_record_primitive(item) for item in raw_commit_records)
    )
    ordered_frame_keys = tuple(
        key
        for key, _ in sorted(
            frame_contract.items(), key=lambda item: (item[1], item[0][1])
        )
    )
    if len(commit_primitives) != len(ordered_frame_keys):
        raise PredictionArchiveError(
            "commit chain must contain exactly one record per frozen frame"
        )
    for commit, frame_key in zip(commit_primitives, ordered_frame_keys):
        if commit["decision_time"] != frame_contract[frame_key]:
            raise PredictionArchiveError(
                "commit decision time does not match the frozen frame"
            )
        snapshot = canonical_json_bytes(
            [
                item.to_primitive()
                for item in sorted(grouped[frame_key], key=lambda item: item.track_id)
            ]
        )
        if commit["snapshot_sha256"] != _digest(snapshot):
            raise PredictionArchiveError(
                "commit snapshot does not match archived frame predictions"
            )

    output_directory = Path(output_directory)
    if output_directory.exists() or output_directory.is_symlink():
        raise PredictionArchiveError("output_directory must not already exist")
    output_directory.mkdir(parents=True, mode=0o700)

    entries: list[PredictionArchiveIndexEntry] = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            frame_contract[item[0]],
            item[0][1],
        ),
    )
    for (sequence_id, frame_id), group in ordered_groups:
        ordered_predictions = tuple(sorted(group, key=lambda item: item.track_id))
        decision_time = frame_contract[(sequence_id, frame_id)]
        relative_path = f"frames/{sequence_id}/{frame_id}.json"
        raw = canonical_json_bytes(
            _frame_document(
                sequence_id,
                frame_id,
                decision_time,
                ordered_predictions,
            )
        )
        destination = output_directory / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with destination.open("xb") as stream:
            stream.write(raw)
        track_ids = [prediction.track_id for prediction in ordered_predictions]
        entries.append(
            PredictionArchiveIndexEntry(
                sequence_id=sequence_id,
                frame_id=frame_id,
                decision_time=decision_time,
                relative_path=relative_path,
                byte_count=len(raw),
                sha256=_digest(raw),
                prediction_count=len(ordered_predictions),
                track_ids_sha256=_digest(canonical_json_bytes(track_ids)),
            )
        )

    payload: dict[str, object] = {
        "archive_id": archive_id,
        **bindings,
        "commit_chain_tip_sha256": commit_primitives[-1]["record_hash"],
        "commit_records": list(commit_primitives),
        "entries": [entry.to_primitive() for entry in entries],
        "frame_count": len(entries),
        "ground_truth_included": False,
        "kind": "tracking_prediction_archive_manifest_v1",
        "prediction_count": len(validated),
        "schema_version": 1,
    }
    content_sha256 = _digest(canonical_json_bytes(payload))
    document = {**payload, "content_sha256": content_sha256}
    with (output_directory / _MANIFEST_NAME).open("xb") as stream:
        stream.write(canonical_json_bytes(document))
    return content_sha256


def verify_prediction_archive(
    directory: Path,
    *,
    expected_frames: Iterable[tuple[str, str, float]],
    expected_bindings: Mapping[str, str] | None = None,
) -> tuple[str, tuple[TrackingPredictionV1, ...]]:
    """Verify the sealed directory and return its hash and predictions.

    Passing ``expected_bindings`` prevents substitution of a self-consistent
    archive from another experiment.  When supplied, the mapping must contain
    exactly the fields in :data:`_BINDING_FIELDS`.
    """

    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise PredictionArchiveError("archive directory must be a real directory")
    manifest_path = directory / _MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PredictionArchiveError(
            "prediction archive manifest is missing or is a symbolic link"
        )
    manifest_raw = manifest_path.read_bytes()
    value = _load_json(manifest_raw, _MANIFEST_NAME)
    expected_fields = {
        "archive_id",
        "content_sha256",
        "entries",
        "frame_count",
        "ground_truth_included",
        "kind",
        "prediction_count",
        "schema_version",
        "commit_chain_tip_sha256",
        "commit_records",
        *_BINDING_FIELDS,
    }
    if type(value) is not dict or set(value) != expected_fields:
        raise PredictionArchiveError(
            "prediction archive manifest has missing or unknown fields"
        )
    if (
        value["schema_version"] != 1
        or value["kind"] != "tracking_prediction_archive_manifest_v1"
    ):
        raise PredictionArchiveError("unsupported prediction archive schema")
    if value["ground_truth_included"] is not False:
        raise PredictionArchiveError(
            "prediction archive must not contain ground truth"
        )
    _path_identifier(value["archive_id"], "archive_id")
    commit_records = _validate_commit_chain_primitives(value["commit_records"])
    if _sha256(
        value["commit_chain_tip_sha256"], "commit_chain_tip_sha256"
    ) != commit_records[-1]["record_hash"]:
        raise PredictionArchiveError("commit chain tip does not match final record")
    bindings = _validate_bindings(
        {name: value[name] for name in _BINDING_FIELDS}
    )
    if expected_bindings is not None:
        expected = _validate_bindings(expected_bindings)
        for name in _BINDING_FIELDS:
            if bindings[name] != expected[name]:
                raise PredictionArchiveError(
                    f"prediction archive binding mismatch for {name}"
                )
    content_sha256 = _sha256(value["content_sha256"], "content_sha256")
    if type(value["entries"]) is not list:
        raise PredictionArchiveError("entries must be an array")
    frame_count = _nonnegative_int(value["frame_count"], "frame_count")
    prediction_count = _nonnegative_int(
        value["prediction_count"], "prediction_count"
    )
    if frame_count == 0:
        raise PredictionArchiveError("prediction archive must not be empty")
    if frame_count != len(value["entries"]):
        raise PredictionArchiveError("frame_count does not match entries")
    if len(commit_records) != frame_count:
        raise PredictionArchiveError(
            "commit chain must contain exactly one record per frozen frame"
        )

    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    if _digest(canonical_json_bytes(payload)) != content_sha256:
        raise PredictionArchiveError(
            "prediction archive manifest content SHA-256 mismatch"
        )
    if canonical_json_bytes(value) != manifest_raw:
        raise PredictionArchiveError(
            "prediction archive manifest is not canonical JSON"
        )

    entries = tuple(_entry_from_mapping(item) for item in value["entries"])
    if [entry.to_primitive() for entry in entries] != value["entries"]:
        raise PredictionArchiveError("prediction index entries are not canonical")
    entry_keys = [(entry.sequence_id, entry.frame_id) for entry in entries]
    if len(set(entry_keys)) != len(entry_keys):
        raise PredictionArchiveError("duplicate sequence/frame archive entry")
    order = [
        (entry.sequence_id, entry.decision_time, entry.frame_id)
        for entry in entries
    ]
    if order != sorted(order):
        raise PredictionArchiveError(
            "prediction archive entries are not in canonical order"
        )
    frame_contract = _frame_contract(expected_frames)
    expected_order = sorted(
        (sequence_id, decision_time, frame_id)
        for (sequence_id, frame_id), decision_time in frame_contract.items()
    )
    if order != expected_order:
        raise PredictionArchiveError(
            "prediction archive frame cohort does not match frozen expected_frames"
        )
    if sum(entry.prediction_count for entry in entries) != prediction_count:
        raise PredictionArchiveError("prediction_count does not match entries")

    expected_files = {_MANIFEST_NAME, *(entry.relative_path for entry in entries)}
    expected_directories = {"frames"}
    for entry in entries:
        parent = PurePosixPath(entry.relative_path).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise PredictionArchiveError(f"symbolic link rejected: {path}")
        if path.is_file():
            observed_files.add(path.relative_to(directory).as_posix())
        elif path.is_dir():
            observed_directories.add(path.relative_to(directory).as_posix())
        else:
            raise PredictionArchiveError(
                f"non-regular filesystem object rejected: {path}"
            )
    if observed_files != expected_files:
        missing = sorted(expected_files - observed_files)
        extra = sorted(observed_files - expected_files)
        raise PredictionArchiveError(
            f"prediction archive file set mismatch; missing={missing}, extra={extra}"
        )
    if observed_directories != expected_directories:
        missing = sorted(expected_directories - observed_directories)
        extra = sorted(observed_directories - expected_directories)
        raise PredictionArchiveError(
            "prediction archive directory set mismatch; "
            f"missing={missing}, extra={extra}"
        )

    all_predictions: list[TrackingPredictionV1] = []
    frame_snapshots: list[bytes] = []
    observed_prediction_keys: set[tuple[str, str, str]] = set()
    for entry in entries:
        raw = (directory / entry.relative_path).read_bytes()
        if len(raw) != entry.byte_count or _digest(raw) != entry.sha256:
            raise PredictionArchiveError(
                f"prediction frame byte mismatch: {entry.relative_path}"
            )
        frame_value = _load_json(raw, entry.relative_path)
        frame_fields = {
            "decision_time",
            "frame_id",
            "ground_truth_included",
            "kind",
            "predictions",
            "schema_version",
            "sequence_id",
        }
        if type(frame_value) is not dict or set(frame_value) != frame_fields:
            raise PredictionArchiveError(
                f"prediction frame has missing or unknown fields: {entry.relative_path}"
            )
        if canonical_json_bytes(frame_value) != raw:
            raise PredictionArchiveError(
                f"prediction frame is not canonical JSON: {entry.relative_path}"
            )
        if (
            frame_value["schema_version"] != 1
            or frame_value["kind"] != "tracking_prediction_frame_v1"
        ):
            raise PredictionArchiveError("unsupported prediction frame schema")
        if frame_value["ground_truth_included"] is not False:
            raise PredictionArchiveError(
                "prediction frame must not contain ground truth"
            )
        if (
            frame_value["sequence_id"],
            frame_value["frame_id"],
            frame_value["decision_time"],
        ) != (entry.sequence_id, entry.frame_id, entry.decision_time):
            raise PredictionArchiveError(
                "prediction frame header disagrees with its index entry"
            )
        if type(frame_value["predictions"]) is not list:
            raise PredictionArchiveError("predictions must be an array")

        frame_predictions = tuple(
            _prediction_from_mapping(item, entry.relative_path)
            for item in frame_value["predictions"]
        )
        if len(frame_predictions) != entry.prediction_count:
            raise PredictionArchiveError(
                "prediction frame count disagrees with its index entry"
            )
        track_ids = [prediction.track_id for prediction in frame_predictions]
        if track_ids != sorted(track_ids) or len(set(track_ids)) != len(track_ids):
            raise PredictionArchiveError(
                "prediction frame track IDs must be unique and canonically ordered"
            )
        if _digest(canonical_json_bytes(track_ids)) != entry.track_ids_sha256:
            raise PredictionArchiveError("prediction frame track ID seal mismatch")
        for prediction in frame_predictions:
            if (
                prediction.sequence_id,
                prediction.frame_id,
                prediction.decision_time,
            ) != (entry.sequence_id, entry.frame_id, entry.decision_time):
                raise PredictionArchiveError(
                    "prediction content disagrees with its frame header"
                )
            key = (
                prediction.sequence_id,
                prediction.frame_id,
                prediction.track_id,
            )
            if key in observed_prediction_keys:
                raise PredictionArchiveError(
                    "duplicate sequence/frame/track prediction in final archive"
                )
            observed_prediction_keys.add(key)
        all_predictions.extend(frame_predictions)
        frame_snapshots.append(
            canonical_json_bytes(
                [prediction.to_primitive() for prediction in frame_predictions]
            )
        )

    if len(all_predictions) != prediction_count:
        raise PredictionArchiveError("verified prediction count mismatch")
    for entry, commit, snapshot in zip(entries, commit_records, frame_snapshots):
        if commit["decision_time"] != entry.decision_time:
            raise PredictionArchiveError(
                "commit decision time does not match archived frame"
            )
        if commit["snapshot_sha256"] != _digest(snapshot):
            raise PredictionArchiveError(
                "commit snapshot does not match archived frame predictions"
            )
    return content_sha256, tuple(all_predictions)


__all__ = [
    "PredictionArchiveError",
    "PredictionArchiveIndexEntry",
    "build_prediction_archive",
    "verify_prediction_archive",
]
