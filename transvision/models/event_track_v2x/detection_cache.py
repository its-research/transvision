"""Immutable, ground-truth-free storage for :class:`DetectionCacheV1` frames.

The cache directory is an experiment input, not a convenient model dump.  It
is created once, contains only decoded detector instances, and is sealed by a
manifest that binds every relative path, byte count, and SHA-256 digest.
Verification rejects symbolic links, extra files, and non-canonical JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable

from .contracts import DetectionCacheV1, decode_contract, encode_contract
from .wire import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_NAME = "detection-cache-manifest.json"


class DetectionCacheError(ValueError):
    """Raised when a detector cache is unsafe, incomplete, or inconsistent."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DetectionCacheError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise DetectionCacheError(f"{name} must be a trimmed non-empty string")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise DetectionCacheError(f"{name} must not contain path separators")
    if value in {".", ".."}:
        raise DetectionCacheError(f"{name} is not a safe identifier")
    return value


def _load_json(data: bytes, name: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DetectionCacheError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: _reject_constant(value, name),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectionCacheError(f"invalid JSON in {name}") from exc


def _reject_constant(value: str, name: str) -> None:
    raise DetectionCacheError(f"non-finite JSON constant in {name}: {value}")


@dataclass(frozen=True, slots=True)
class DetectionCacheIndexEntry:
    sequence_id: str
    frame_id: str
    agent_id: str
    event_time: float
    relative_path: str
    byte_count: int
    sha256: str
    detection_count: int

    def __post_init__(self) -> None:
        for field_name in ("sequence_id", "frame_id", "agent_id"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name),
            )
        event_time = float(self.event_time)
        if not (-float("inf") < event_time < float("inf")):
            raise DetectionCacheError("event_time must be finite")
        object.__setattr__(self, "event_time", event_time)
        if type(self.relative_path) is not str:
            raise DetectionCacheError("relative_path must be a string")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
            raise DetectionCacheError("relative_path must be a safe JSON path")
        if path.as_posix() != self.relative_path:
            raise DetectionCacheError("relative_path must be canonical POSIX syntax")
        for field_name in ("byte_count", "detection_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise DetectionCacheError(f"{field_name} must be non-negative int")
        if self.byte_count == 0:
            raise DetectionCacheError("cache frame must not be empty")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))

    def to_primitive(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "byte_count": self.byte_count,
            "detection_count": self.detection_count,
            "event_time": self.event_time,
            "frame_id": self.frame_id,
            "relative_path": self.relative_path,
            "sequence_id": self.sequence_id,
            "sha256": self.sha256,
        }


def _entry_from_mapping(value: object) -> DetectionCacheIndexEntry:
    expected = {
        "agent_id",
        "byte_count",
        "detection_count",
        "event_time",
        "frame_id",
        "relative_path",
        "sequence_id",
        "sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise DetectionCacheError("cache index entry has missing or unknown fields")
    return DetectionCacheIndexEntry(**value)


def _frame_relative_path(frame: DetectionCacheV1) -> str:
    sequence = _identifier(frame.sequence_id, "sequence_id")
    agent = _identifier(frame.agent_id, "agent_id")
    frame_id = _identifier(frame.frame_id, "frame_id")
    return f"frames/{sequence}/{agent}/{frame_id}.json"


def detection_frame_contract_sha256(
    frames: Iterable[tuple[str, str, str, float]],
) -> str:
    """Hash the exact ordered frame identities expected from a cohort."""

    contract = _expected_frame_contract(frames)
    payload = {
        "frames": [
            {
                "agent_id": agent_id,
                "event_time": event_time,
                "frame_id": frame_id,
                "sequence_id": sequence_id,
            }
            for sequence_id, frame_id, agent_id, event_time in contract
        ],
        "kind": "detection_frame_contract_v1",
        "schema_version": 1,
    }
    return _digest(canonical_json_bytes(payload))


def build_detection_cache(
    frames: Iterable[DetectionCacheV1],
    output_directory: Path,
    *,
    cache_id: str,
    cohort_sha256: str,
    allowed_class_names: tuple[str, ...],
) -> str:
    """Create a new immutable cache and return its manifest content digest.

    ``output_directory`` must not exist.  This prevents a cache regeneration
    from silently mixing old and new detector outputs.
    """

    cache_id = _identifier(cache_id, "cache_id")
    cohort_sha256 = _sha256(cohort_sha256, "cohort_sha256")
    if not isinstance(allowed_class_names, tuple) or not allowed_class_names:
        raise DetectionCacheError(
            "allowed_class_names must be a non-empty tuple"
        )
    class_names = tuple(
        sorted(_identifier(value, "allowed class name") for value in allowed_class_names)
    )
    if len(set(class_names)) != len(class_names):
        raise DetectionCacheError("allowed_class_names must be unique")
    output_directory = Path(output_directory)
    if output_directory.exists() or output_directory.is_symlink():
        raise DetectionCacheError("output_directory must not already exist")

    ordered = sorted(
        tuple(frames),
        key=lambda frame: (
            frame.sequence_id,
            frame.event_time,
            frame.frame_id,
            frame.agent_id,
        ),
    )
    if not ordered:
        raise DetectionCacheError("detection cache must contain at least one frame")
    identities = [
        (frame.sequence_id, frame.frame_id, frame.agent_id) for frame in ordered
    ]
    if len(set(identities)) != len(identities):
        raise DetectionCacheError("cache frame identities must be unique")
    frame_contract_sha256 = detection_frame_contract_sha256(
        (
            frame.sequence_id,
            frame.frame_id,
            frame.agent_id,
            frame.event_time,
        )
        for frame in ordered
    )

    dataset_hashes = {frame.dataset_sha256 for frame in ordered}
    config_hashes = {frame.detector_config_sha256 for frame in ordered}
    checkpoint_hashes = {frame.checkpoint_sha256 for frame in ordered}
    if len(dataset_hashes) != 1:
        raise DetectionCacheError("all frames must bind the same dataset SHA-256")
    if len(config_hashes) != 1 or len(checkpoint_hashes) != 1:
        raise DetectionCacheError("all frames must bind one detector and checkpoint")
    unknown_labels = sorted(
        {
            label
            for frame in ordered
            for label in frame.class_labels
            if label not in set(class_names)
        }
    )
    if unknown_labels:
        raise DetectionCacheError(
            f"cache contains labels outside the frozen class vocabulary: {unknown_labels}"
        )

    output_directory.mkdir(parents=True, mode=0o700)

    entries: list[DetectionCacheIndexEntry] = []
    for frame in ordered:
        relative_path = _frame_relative_path(frame)
        destination = output_directory / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw = encode_contract(frame)
        with destination.open("xb") as handle:
            handle.write(raw)
        entries.append(
            DetectionCacheIndexEntry(
                sequence_id=frame.sequence_id,
                frame_id=frame.frame_id,
                agent_id=frame.agent_id,
                event_time=frame.event_time,
                relative_path=relative_path,
                byte_count=len(raw),
                sha256=_digest(raw),
                detection_count=frame.count,
            )
        )

    payload: dict[str, object] = {
        "cache_id": cache_id,
        "class_names": list(class_names),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "cohort_sha256": cohort_sha256,
        "dataset_sha256": next(iter(dataset_hashes)),
        "detector_config_sha256": next(iter(config_hashes)),
        "entries": [entry.to_primitive() for entry in entries],
        "frame_contract_sha256": frame_contract_sha256,
        "frame_count": len(entries),
        "ground_truth_included": False,
        "kind": "detection_cache_manifest_v1",
        "schema_version": 1,
    }
    content_sha256 = _digest(canonical_json_bytes(payload))
    document = {**payload, "content_sha256": content_sha256}
    manifest_raw = canonical_json_bytes(document)
    with (output_directory / _MANIFEST_NAME).open("xb") as handle:
        handle.write(manifest_raw)
    return content_sha256


def _expected_frame_contract(
    frames: Iterable[tuple[str, str, str, float]],
) -> tuple[tuple[str, str, str, float], ...]:
    result: list[tuple[str, str, str, float]] = []
    for index, item in enumerate(frames):
        if not isinstance(item, tuple) or len(item) != 4:
            raise DetectionCacheError(
                f"expected frame {index} must be "
                "(sequence_id, frame_id, agent_id, event_time)"
            )
        sequence_id = _identifier(item[0], "expected sequence_id")
        frame_id = _identifier(item[1], "expected frame_id")
        agent_id = _identifier(item[2], "expected agent_id")
        try:
            event_time = float(item[3])
        except (TypeError, ValueError) as exc:
            raise DetectionCacheError("expected event_time must be finite") from exc
        if not (-float("inf") < event_time < float("inf")):
            raise DetectionCacheError("expected event_time must be finite")
        result.append((sequence_id, frame_id, agent_id, event_time))
    if not result or len(set(result)) != len(result):
        raise DetectionCacheError("expected frames must be non-empty and unique")
    return tuple(sorted(result, key=lambda item: (item[0], item[3], item[1], item[2])))


def verify_detection_cache(
    directory: Path,
    *,
    expected_cohort_sha256: str,
    expected_frames: Iterable[tuple[str, str, str, float]],
    expected_frame_contract_sha256: str,
    expected_class_names: tuple[str, ...],
) -> tuple[str, tuple[DetectionCacheV1, ...]]:
    """Verify an entire cache against its frozen cohort and return its frames."""

    directory = Path(directory)
    expected_cohort_sha256 = _sha256(
        expected_cohort_sha256, "expected_cohort_sha256"
    )
    frame_contract = _expected_frame_contract(expected_frames)
    expected_frame_contract_sha256 = _sha256(
        expected_frame_contract_sha256, "expected_frame_contract_sha256"
    )
    if detection_frame_contract_sha256(frame_contract) != expected_frame_contract_sha256:
        raise DetectionCacheError(
            "expected frame identities do not match frozen frame-contract SHA-256"
        )
    if not isinstance(expected_class_names, tuple) or not expected_class_names:
        raise DetectionCacheError(
            "expected_class_names must be a non-empty tuple"
        )
    class_names = tuple(
        sorted(
            _identifier(value, "expected class name")
            for value in expected_class_names
        )
    )
    if len(set(class_names)) != len(class_names):
        raise DetectionCacheError("expected_class_names must be unique")
    if not directory.is_dir() or directory.is_symlink():
        raise DetectionCacheError("cache directory must be a real directory")
    manifest_path = directory / _MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise DetectionCacheError("cache manifest is missing or is a symbolic link")
    manifest_raw = manifest_path.read_bytes()
    value = _load_json(manifest_raw, _MANIFEST_NAME)
    expected = {
        "cache_id",
        "checkpoint_sha256",
        "class_names",
        "cohort_sha256",
        "content_sha256",
        "dataset_sha256",
        "detector_config_sha256",
        "entries",
        "frame_contract_sha256",
        "frame_count",
        "ground_truth_included",
        "kind",
        "schema_version",
    }
    if type(value) is not dict or set(value) != expected:
        raise DetectionCacheError("cache manifest has missing or unknown fields")
    if value["schema_version"] != 1 or value["kind"] != "detection_cache_manifest_v1":
        raise DetectionCacheError("unsupported cache manifest schema")
    if value["ground_truth_included"] is not False:
        raise DetectionCacheError("detection cache must not contain ground truth")
    if value["class_names"] != list(class_names):
        raise DetectionCacheError("cache class vocabulary does not match evaluator")
    if value["frame_contract_sha256"] != expected_frame_contract_sha256:
        raise DetectionCacheError(
            "cache frame-contract SHA-256 does not match frozen plan"
        )
    for field_name in (
        "checkpoint_sha256",
        "cohort_sha256",
        "dataset_sha256",
        "detector_config_sha256",
        "content_sha256",
    ):
        _sha256(value[field_name], field_name)
    if value["cohort_sha256"] != expected_cohort_sha256:
        raise DetectionCacheError("cache cohort SHA-256 does not match frozen cohort")
    _identifier(value["cache_id"], "cache_id")
    if type(value["entries"]) is not list:
        raise DetectionCacheError("entries must be an array")
    if type(value["frame_count"]) is not int or value["frame_count"] != len(
        value["entries"]
    ):
        raise DetectionCacheError("frame_count does not match entries")

    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    content_sha256 = _digest(canonical_json_bytes(payload))
    if content_sha256 != value["content_sha256"]:
        raise DetectionCacheError("cache manifest content SHA-256 mismatch")
    if canonical_json_bytes(value) != manifest_raw:
        raise DetectionCacheError("cache manifest is not canonical JSON")

    entries = tuple(_entry_from_mapping(item) for item in value["entries"])
    if not entries:
        raise DetectionCacheError("detection cache manifest must not be empty")
    if tuple(entry.to_primitive() for entry in entries) != tuple(value["entries"]):
        raise DetectionCacheError("cache index entries are not canonical")
    keys = [(entry.sequence_id, entry.frame_id, entry.agent_id) for entry in entries]
    order = [
        (entry.sequence_id, entry.event_time, entry.frame_id, entry.agent_id)
        for entry in entries
    ]
    if order != sorted(order):
        raise DetectionCacheError("cache entries are not in canonical order")
    if len(set(keys)) != len(keys):
        raise DetectionCacheError("cache entry identities must be unique")
    observed_contract = tuple(
        (entry.sequence_id, entry.frame_id, entry.agent_id, entry.event_time)
        for entry in entries
    )
    if observed_contract != frame_contract:
        missing = sorted(set(frame_contract) - set(observed_contract))
        extra = sorted(set(observed_contract) - set(frame_contract))
        raise DetectionCacheError(
            "cache frame cohort mismatch; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    expected_files = {_MANIFEST_NAME, *(entry.relative_path for entry in entries)}
    observed_files: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise DetectionCacheError(f"symbolic link rejected: {path}")
        if path.is_file():
            observed_files.add(path.relative_to(directory).as_posix())
    if observed_files != expected_files:
        missing = sorted(expected_files - observed_files)
        extra = sorted(observed_files - expected_files)
        raise DetectionCacheError(
            f"cache file set mismatch; missing={missing}, extra={extra}"
        )

    frames: list[DetectionCacheV1] = []
    for entry in entries:
        raw = (directory / entry.relative_path).read_bytes()
        if len(raw) != entry.byte_count or _digest(raw) != entry.sha256:
            raise DetectionCacheError(f"cache frame byte mismatch: {entry.relative_path}")
        contract = decode_contract(raw, DetectionCacheV1)
        if (
            contract.sequence_id,
            contract.frame_id,
            contract.agent_id,
            contract.event_time,
            contract.count,
        ) != (
            entry.sequence_id,
            entry.frame_id,
            entry.agent_id,
            entry.event_time,
            entry.detection_count,
        ):
            raise DetectionCacheError("cache frame content disagrees with index")
        if contract.dataset_sha256 != value["dataset_sha256"]:
            raise DetectionCacheError("cache frame dataset hash drift")
        if contract.detector_config_sha256 != value["detector_config_sha256"]:
            raise DetectionCacheError("cache frame detector hash drift")
        if contract.checkpoint_sha256 != value["checkpoint_sha256"]:
            raise DetectionCacheError("cache frame checkpoint hash drift")
        if any(label not in class_names for label in contract.class_labels):
            raise DetectionCacheError("cache frame label is outside class vocabulary")
        frames.append(contract)
    return content_sha256, tuple(frames)


__all__ = [
    "DetectionCacheError",
    "DetectionCacheIndexEntry",
    "build_detection_cache",
    "detection_frame_contract_sha256",
    "verify_detection_cache",
]
