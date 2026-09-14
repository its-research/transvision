#!/usr/bin/env python3
"""Fail-closed real V2X-Seq canary orchestration.

This module supplies the dataset/protocol/evidence plumbing only.  It does not
contain CenterPoint, RTP-V2X, or any other thesis model.  A committed adapter is
mandatory and is responsible for the actual forward pass, backward pass,
optimizer step, metric implementation, and checkpoint serialization.

The canary is intentionally unable to turn itself into a scientific result:
``scientific_claim_allowed`` must be false and its evidence manifest records
``diagnostic_only``.  Main experiments need a separate frozen configuration and
the repository result gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from provenance_seals import (
    SealError,
    loads_json_strict,
    validate_git_seal,
    validate_installed_environment_manifest,
    verify_head_blob,
)


SCHEMA_VERSION = 1
CANARY_TYPE = "real_v2xseq"
DATASET_NAME = "V2X-Seq-SPD"
TEMPLATE_CONFIG_PATH = "experiments/clearml/configs/v2xseq_track_canary.json"
OFFICIAL_REPOSITORY = "https://github.com/AIR-THU/DAIR-V2X-Seq"
OFFICIAL_ASSET_LAYOUT = {
    ("vehicle", "lidar"): ("vehicle-side", "velodyne", ".pcd"),
    ("infrastructure", "lidar"): ("infrastructure-side", "velodyne", ".pcd"),
    ("vehicle", "camera"): ("vehicle-side", "image", ".jpg"),
    ("infrastructure", "camera"): ("infrastructure-side", "image", ".jpg"),
}
REQUIRED_ARTIFACTS = (
    "checkpoint.bin",
    "environment.json",
    "metrics.json",
    "predictions.jsonl",
    "stdout.log",
)
REQUIRED_SOURCE_SEAL_PATHS = frozenset(
    {
        TEMPLATE_CONFIG_PATH,
        "experiments/clearml/protocols/v2xseq-track-v1.json",
        "experiments/rtp_v2x/freeze_v2xseq_inputs.py",
        "experiments/rtp_v2x/provenance_seals.py",
        "experiments/rtp_v2x/v2xseq_real_canary.py",
    }
)


class ContractError(RuntimeError):
    """Raised when a real-data run cannot be identified or replayed safely."""


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_json_snapshot(path: Path) -> tuple[Any, str]:
    """Parse and hash one immutable read of a control or dataset JSON file."""

    try:
        payload = path.read_bytes()
        document = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot read JSON {path}: {type(exc).__name__}") from exc
    return document, sha256_bytes(payload)


def load_json(path: Path) -> Any:
    return load_json_snapshot(path)[0]


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_copy(value: object, label: str) -> Any:
    """Deep-copy finite JSON so adapter-side mutation cannot change evidence."""

    try:
        return json.loads(
            canonical_bytes(value),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not finite JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_dataset_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ContractError(f"unsafe dataset path: {relative!r}")
    resolved_root = root.resolve()
    cursor = resolved_root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ContractError(
                f"dataset evidence may not traverse a symlink: {relative}"
            )
    target = cursor.resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"dataset path escapes root: {relative!r}") from exc
    return target


def validate_official_asset_path(
    agent: str, modality: str, relative: str, label: str
) -> None:
    expected = OFFICIAL_ASSET_LAYOUT.get((agent, modality))
    if expected is None:
        raise ContractError(f"{label} uses an unsupported SPD agent/modality")
    pure = PurePosixPath(relative)
    prefix = expected[:2]
    suffix = expected[2]
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or len(pure.parts) != 3
        or pure.parts[:2] != prefix
        or pure.suffix.lower() != suffix
    ):
        raise ContractError(
            f"{label} does not match the official SPD {agent}/{modality} layout"
        )


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value


def validate_digest(value: Any, label: str) -> str:
    digest = require_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def validate_identity(root: Path, document: object) -> dict[str, dict[str, Any]]:
    identity = require_object(document, "dataset identity")
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("dataset identity schema_version must be 1")
    if identity.get("dataset") != "V2X-Seq-SPD":
        raise ContractError("dataset identity must name V2X-Seq-SPD")
    if identity.get("source_repository") != OFFICIAL_REPOSITORY:
        raise ContractError("dataset identity must point to the official repository")
    if identity.get("source_kind") != "official_release":
        raise ContractError("dataset identity source_kind must be official_release")
    if identity.get("license_acknowledged") is not True:
        raise ContractError("dataset license must be acknowledged explicitly")
    require_string(identity.get("release_id"), "dataset identity release_id")
    if identity.get("scope") not in {"canary_subset", "complete_release"}:
        raise ContractError(
            "dataset identity scope must be canary_subset or complete_release"
        )
    sequence_ids = identity.get("sequence_ids")
    if (
        not isinstance(sequence_ids, list)
        or not sequence_ids
        or any(not isinstance(value, str) or not value for value in sequence_ids)
        or sequence_ids != sorted(set(sequence_ids))
    ):
        raise ContractError(
            "dataset identity sequence_ids must be a sorted unique non-empty array"
        )
    entries = identity.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ContractError("dataset identity entries must be a non-empty array")

    index: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(entries):
        entry = require_object(raw, f"dataset identity entry {position}")
        relative = require_string(entry.get("path"), f"entry {position}.path")
        if relative in index:
            raise ContractError(f"duplicate dataset identity path: {relative}")
        expected_size = entry.get("size")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
        ):
            raise ContractError(f"entry {position}.size must be a positive integer")
        expected_hash = validate_digest(entry.get("sha256"), f"entry {position}.sha256")
        target = safe_dataset_path(root, relative)
        if not target.is_file():
            raise ContractError(f"dataset identity file is missing: {relative}")
        if target.stat().st_size != expected_size:
            raise ContractError(f"dataset identity size mismatch: {relative}")
        if sha256_file(target) != expected_hash:
            raise ContractError(f"dataset identity hash mismatch: {relative}")
        role = require_string(entry.get("role"), f"entry {position}.role")
        common_fields = {"path", "size", "sha256", "role"}
        role_fields = {
            "cooperative_index": common_fields,
            "vehicle_index": common_fields,
            "label": common_fields | {"sequence", "vehicle_frame"},
            "raw_sensor": common_fields
            | {
                "sequence",
                "vehicle_frame",
                "infrastructure_frame",
                "agent",
                "modality",
            },
        }
        if role not in role_fields:
            raise ContractError(f"entry {position}.role is unsupported: {role}")
        if set(entry) != role_fields[role]:
            raise ContractError(
                f"entry {position} fields do not match the strict {role} schema"
            )
        if role in {"label", "raw_sensor"}:
            entry_sequence = require_string(
                entry.get("sequence"), f"entry {position}.sequence"
            )
            if entry_sequence not in sequence_ids:
                raise ContractError(
                    f"entry {position}.sequence is absent from identity sequence_ids"
                )
            require_string(
                entry.get("vehicle_frame"), f"entry {position}.vehicle_frame"
            )
        if role == "raw_sensor":
            require_string(
                entry.get("infrastructure_frame"),
                f"entry {position}.infrastructure_frame",
            )
            agent = require_string(entry.get("agent"), f"entry {position}.agent")
            modality = require_string(
                entry.get("modality"), f"entry {position}.modality"
            )
            validate_official_asset_path(
                agent, modality, relative, f"entry {position}.path"
            )
        index[relative] = {**entry, "role": role}
    if sha256_bytes(
        canonical_bytes(sorted(index.values(), key=lambda row: row["path"]))
    ) != validate_digest(
        identity.get("content_sha256"), "dataset identity content_sha256"
    ):
        raise ContractError("dataset identity content_sha256 mismatch")
    cooperative_indexes = [
        entry for entry in index.values() if entry["role"] == "cooperative_index"
    ]
    if len(cooperative_indexes) != 1:
        raise ContractError("dataset identity must seal exactly one cooperative_index")
    vehicle_indexes = [
        entry for entry in index.values() if entry["role"] == "vehicle_index"
    ]
    if len(vehicle_indexes) != 1:
        raise ContractError("dataset identity must seal exactly one vehicle_index")
    return index


def validate_split(document: object, *, protocol_id: str) -> dict[str, list[str]]:
    split = require_object(document, "frozen split")
    if split.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("frozen split schema_version must be 1")
    if split.get("dataset") != "V2X-Seq-SPD":
        raise ContractError("frozen split must name V2X-Seq-SPD")
    if split.get("protocol_id") != protocol_id:
        raise ContractError("frozen split protocol_id does not match protocol")
    validate_digest(split.get("source_sha256"), "frozen split source_sha256")
    partitions = split.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        "train",
        "validation",
        "test",
    }:
        raise ContractError(
            "frozen split must contain train, validation, and test exactly"
        )
    seen: set[str] = set()
    normalized: dict[str, list[str]] = {}
    for name in ("train", "validation", "test"):
        values = partitions[name]
        if not isinstance(values, list) or any(
            not isinstance(item, str) or not item for item in values
        ):
            raise ContractError(f"frozen split {name} must be an array of sequence IDs")
        if values != sorted(set(values)):
            raise ContractError(f"frozen split {name} must be sorted and unique")
        overlap = seen.intersection(values)
        if overlap:
            raise ContractError(f"frozen split partitions overlap: {sorted(overlap)}")
        seen.update(values)
        normalized[name] = list(values)
    if not seen:
        raise ContractError("frozen split contains no sequences")
    expected_hash = validate_digest(split.get("partition_sha256"), "partition_sha256")
    if sha256_bytes(canonical_bytes(normalized)) != expected_hash:
        raise ContractError("frozen split partition_sha256 mismatch")
    return normalized


def validate_sequence_request(
    data: Mapping[str, Any],
    key: str,
    *,
    expected_partition: str,
    partitions: Mapping[str, list[str]],
) -> tuple[str, int]:
    request = require_object(data.get(key), f"config.data.{key}")
    if set(request) != {"partition", "sequence_id", "max_frames"}:
        raise ContractError(
            f"config.data.{key} must contain partition, sequence_id, and max_frames exactly"
        )
    partition = require_string(request.get("partition"), f"config.data.{key}.partition")
    if partition != expected_partition:
        raise ContractError(f"config.data.{key}.partition must be {expected_partition}")
    sequence_id = require_string(
        request.get("sequence_id"), f"config.data.{key}.sequence_id"
    )
    if sequence_id not in partitions[partition]:
        raise ContractError(
            f"config.data.{key}.sequence_id is absent from the frozen {partition} partition"
        )
    max_frames = request.get("max_frames")
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames <= 0
    ):
        raise ContractError(f"config.data.{key}.max_frames must be a positive integer")
    return sequence_id, max_frames


def validate_protocol(document: object) -> dict[str, Any]:
    protocol = require_object(document, "protocol")
    require_string(protocol.get("protocol_id"), "protocol_id")
    if protocol.get("status") != "frozen":
        raise ContractError("protocol status must be frozen")
    if protocol.get("dataset") != "V2X-Seq-SPD":
        raise ContractError("tracking protocol must name V2X-Seq-SPD")
    if protocol.get("causal_rule") != "arrival_time <= decision_time":
        raise ContractError(
            "protocol causal rule is not frozen to the inclusive cutoff"
        )
    unresolved = protocol.get("unresolved")
    if unresolved != []:
        raise ContractError("protocol unresolved list must be empty")
    if any(str(key).endswith("_not_frozen") for key in protocol):
        raise ContractError("protocol still contains a not-frozen section")
    require_string(protocol.get("coordinate_frame"), "protocol coordinate_frame")
    classes = protocol.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(item, str) or not item for item in classes)
    ):
        raise ContractError("protocol classes must be a non-empty string array")
    fault_grid = protocol.get("fault_grid")
    if not isinstance(fault_grid, dict) or not fault_grid:
        raise ContractError("protocol fault_grid must be frozen")
    time_model = require_object(protocol.get("time_model"), "protocol time_model")
    require_string(time_model.get("native_timestamp_unit"), "native_timestamp_unit")
    interval = time_model.get("sampling_interval_ns")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ContractError("sampling_interval_ns must be a positive integer")
    metrics = protocol.get("tracking_metrics")
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(not isinstance(item, str) for item in metrics)
    ):
        raise ContractError("tracking_metrics must be a non-empty string array")
    return protocol


@dataclass(frozen=True)
class Frame:
    sequence_id: str
    frame_id: str
    timestamp: int
    assets: tuple[dict[str, Any], ...]
    labels: tuple[dict[str, Any], ...]

    def observation_payload(self, root: Path) -> dict[str, Any]:
        """Return the model-visible observation without any evaluation target."""

        assets = []
        for entry in self.assets:
            assets.append(
                {
                    "agent": entry["agent"],
                    "modality": entry["modality"],
                    "path": entry["path"],
                    "size": entry["size"],
                    "sha256": entry["sha256"],
                    "absolute_path": str(safe_dataset_path(root, entry["path"])),
                }
            )
        return canonical_copy(
            {
                "sequence_id": self.sequence_id,
                "frame_id": self.frame_id,
                "timestamp": self.timestamp,
                "assets": assets,
            },
            "frame observation",
        )

    def target_payload(self) -> dict[str, Any]:
        """Return the evaluator-only target without local sensor paths."""

        return canonical_copy(
            {
                "sequence_id": self.sequence_id,
                "frame_id": self.frame_id,
                "timestamp": self.timestamp,
                "labels": list(self.labels),
            },
            "frame target",
        )

    def training_payload(self, root: Path) -> dict[str, Any]:
        """Return the sole training frame with its supervised target attached."""

        return canonical_copy(
            {**self.observation_payload(root), "labels": list(self.labels)},
            "frame training payload",
        )


def load_sequence(
    root: Path,
    identity_index: Mapping[str, dict[str, Any]],
    *,
    sequence_id: str,
    max_frames: int,
    required_assets: Sequence[tuple[str, str]],
) -> list[Frame]:
    cooperative_entries = [
        entry
        for entry in identity_index.values()
        if entry.get("role") == "cooperative_index"
    ]
    if len(cooperative_entries) != 1:
        raise ContractError("identity must contain exactly one cooperative_index")
    cooperative_path = safe_dataset_path(root, cooperative_entries[0]["path"])
    data_info = load_json(cooperative_path)
    if not isinstance(data_info, list):
        raise ContractError("cooperative data_info must be an array")
    vehicle_entries = [
        entry
        for entry in identity_index.values()
        if entry.get("role") == "vehicle_index"
    ]
    if len(vehicle_entries) != 1:
        raise ContractError("identity must contain exactly one vehicle_index")
    vehicle_info = load_json(safe_dataset_path(root, vehicle_entries[0]["path"]))
    if not isinstance(vehicle_info, list):
        raise ContractError("vehicle data_info must be an array")
    vehicle_index: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(vehicle_info):
        row = require_object(raw, f"vehicle data_info row {position}")
        frame_id = require_string(
            row.get("frame_id"), f"vehicle row {position}.frame_id"
        )
        if frame_id in vehicle_index:
            raise ContractError(f"vehicle data_info has duplicate frame: {frame_id}")
        vehicle_index[frame_id] = row
    selected = [
        row
        for row in data_info
        if isinstance(row, dict) and row.get("vehicle_sequence") == sequence_id
    ]
    selected.sort(key=lambda row: str(row.get("vehicle_frame", "")))
    if not selected:
        raise ContractError(
            f"sequence is absent from cooperative data_info: {sequence_id}"
        )
    if max_frames <= 0:
        raise ContractError("max_frames must be positive")
    selected = selected[:max_frames]
    frames: list[Frame] = []
    for pair in selected:
        frame_id = require_string(pair.get("vehicle_frame"), "vehicle_frame")
        matching_labels = [
            entry
            for entry in identity_index.values()
            if entry.get("role") == "label"
            and entry.get("vehicle_frame") == frame_id
            and entry.get("sequence") == sequence_id
        ]
        if len(matching_labels) != 1:
            raise ContractError(
                f"identity must seal exactly one label file for {frame_id}"
            )
        labels = load_json(safe_dataset_path(root, matching_labels[0]["path"]))
        if not isinstance(labels, list) or any(
            not isinstance(row, dict) for row in labels
        ):
            raise ContractError(f"label file is not an object array for {frame_id}")
        vehicle_row = vehicle_index.get(frame_id)
        if vehicle_row is None:
            raise ContractError(f"vehicle data_info is missing frame {frame_id}")
        if vehicle_row.get("sequence_id") != sequence_id:
            raise ContractError(
                f"vehicle data_info sequence mismatch for frame {frame_id}"
            )
        timestamp_value = vehicle_row.get("pointcloud_timestamp")
        if isinstance(timestamp_value, bool) or not isinstance(
            timestamp_value, (str, int)
        ):
            raise ContractError(
                f"frame {frame_id} lacks an observation-side pointcloud_timestamp"
            )
        try:
            timestamp = int(timestamp_value)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"frame {frame_id} has a non-integer observation timestamp"
            ) from exc
        label_timestamps: set[int] = set()
        for row in labels:
            value = row.get("veh_pointcloud_timestamp")
            if value in (None, ""):
                continue
            try:
                label_timestamps.add(int(value))
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    f"frame {frame_id} has a non-integer label timestamp"
                ) from exc
        if label_timestamps and label_timestamps != {timestamp}:
            raise ContractError(
                f"frame {frame_id} label timestamp disagrees with observation metadata"
            )

        frame_assets = [
            entry
            for entry in identity_index.values()
            if entry.get("role") == "raw_sensor"
            and entry.get("vehicle_frame") == frame_id
            and entry.get("sequence") == sequence_id
        ]
        asset_index: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in frame_assets:
            key = (str(entry.get("agent")), str(entry.get("modality")))
            if key in asset_index:
                raise ContractError(
                    f"frame {frame_id} has duplicate sealed raw sensor: {key}"
                )
            asset_index[key] = entry
        observed_assets = set(asset_index)
        expected_assets = set(required_assets)
        if observed_assets != expected_assets:
            missing_assets = sorted(expected_assets - observed_assets)
            extra_assets = sorted(observed_assets - expected_assets)
            raise ContractError(
                f"frame {frame_id} raw sensor set mismatch; "
                f"missing={missing_assets}, extra={extra_assets}"
            )
        frame_assets = [asset_index[key] for key in sorted(asset_index)]
        for entry in frame_assets:
            target = safe_dataset_path(root, entry["path"])
            with target.open("rb") as stream:
                if not stream.read(1):
                    raise ContractError(f"raw sensor file is empty: {entry['path']}")
        frames.append(
            Frame(
                sequence_id=sequence_id,
                frame_id=frame_id,
                timestamp=timestamp,
                assets=tuple(sorted(frame_assets, key=lambda row: row["path"])),
                labels=tuple(labels),
            )
        )
    frames.sort(key=lambda frame: (frame.timestamp, frame.frame_id))
    for previous, current in zip(frames, frames[1:]):
        if current.timestamp <= previous.timestamp:
            raise ContractError(
                f"sequence timestamps are not strictly increasing at {current.frame_id}"
            )
    return frames


def resolve_repository_file(repository_root: Path, relative: str, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ContractError(f"unsafe {label} path")
    resolved_root = repository_root.resolve()
    cursor = resolved_root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ContractError(f"{label} may not traverse a symlink")
    path = cursor.resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"{label} escapes repository") from exc
    if not path.is_file():
        raise ContractError(f"{label} must be a regular repository file")
    return path


def git_identity(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("repository has no readable Git commit") from exc
    if len(commit) not in {40, 64} or any(c not in "0123456789abcdef" for c in commit):
        raise ContractError("Git commit is not a full lowercase object ID")
    return commit, bool(status.strip())


def probe_gpu(required_name_substring: str) -> dict[str, Any]:
    """Require at least one visible GPU matching the frozen device family."""

    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ContractError("nvidia-smi GPU preflight failed") from exc
    rows = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not rows or not any(
        required_name_substring in row.split(",", 1)[0] for row in rows
    ):
        raise ContractError(
            f"no visible GPU matches required substring: {required_name_substring}"
        )
    return {"query": "name,driver_version,memory.total", "rows": rows}


def ensure_read_only_dataset(root: Path) -> None:
    if not root.is_dir():
        raise ContractError("SPD dataset root is not a directory")
    if os.access(root, os.W_OK):
        raise ContractError("SPD dataset root must be mounted read-only for the canary")


def ensure_read_only_contracts_root(root: Path) -> None:
    if not root.is_dir():
        raise ContractError("external contracts root is not a directory")
    if os.access(root, os.W_OK):
        raise ContractError("external contracts root must be mounted read-only")


def validate_clearml_task_id(value: Any) -> str:
    task_id = require_string(value, "ClearML task ID")
    if len(task_id) != 32 or any(c not in "0123456789abcdef" for c in task_id):
        raise ContractError(
            "ClearML task ID must be a 32-character lowercase hexadecimal ID"
        )
    environment_id = os.environ.get("CLEARML_TASK_ID") or os.environ.get(
        "TRAINS_TASK_ID"
    )
    if environment_id != task_id:
        raise ContractError(
            "ClearML task ID must come from the active ClearML agent environment"
        )
    return task_id


def read_stable_regular_file(path: Path, label: str) -> bytes:
    """Read one ordinary file and reject replacement during the read."""

    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise ContractError(f"{label} must be a non-symlink regular file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ContractError(f"{label} could not be read") from exc
    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )
    if stable_before != stable_after or not stat.S_ISREG(after.st_mode):
        raise ContractError(f"{label} changed while it was read")
    return payload


def resolve_contract_file(contracts_root: Path, relative: str, label: str) -> Path:
    """Resolve a regular file below an external, non-symlink contracts root."""

    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise ContractError(f"unsafe {label} path: {relative!r}")
    if pure.as_posix() != relative:
        raise ContractError(f"non-canonical {label} path: {relative!r}")
    raw_root = Path(contracts_root)
    if raw_root.is_symlink():
        raise ContractError("contracts root may not be a symlink")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise ContractError("contracts root is unavailable") from exc
    if not root.is_dir():
        raise ContractError("contracts root must be a directory")
    cursor = root
    for component in pure.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ContractError(f"{label} may not traverse a symlink")
    try:
        metadata = cursor.lstat()
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ContractError(f"{label} is unavailable below the contracts root") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} must be a regular file")
    return resolved


def _resolve_repository_directory(
    repository_root: Path, relative: str, label: str
) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise ContractError(f"unsafe {label} path: {relative!r}")
    if pure.as_posix() != relative:
        raise ContractError(f"non-canonical {label} path: {relative!r}")
    root = repository_root.resolve()
    cursor = root
    for component in pure.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ContractError(f"{label} may not traverse a symlink")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ContractError(
            f"{label} is unavailable below the repository root"
        ) from exc
    if not resolved.is_dir():
        raise ContractError(f"{label} must be a directory")
    return resolved


def _package_tree_paths(repository_root: Path, package_root_relative: str) -> set[str]:
    package_root = _resolve_repository_directory(
        repository_root, package_root_relative, "adapter.package_root"
    )
    paths: set[str] = set()
    for current, directory_names, file_names in os.walk(
        package_root, followlinks=False
    ):
        current_path = Path(current)
        for name in directory_names:
            candidate = current_path / name
            if name == ".git":
                raise ContractError(
                    "adapter package_root may not contain nested Git metadata"
                )
            if candidate.is_symlink():
                raise ContractError(
                    "adapter package_root may not contain directory symlinks"
                )
        for name in file_names:
            candidate = current_path / name
            if name == ".git":
                raise ContractError(
                    "adapter package_root may not contain nested Git metadata"
                )
            try:
                metadata = candidate.lstat()
            except OSError as exc:
                raise ContractError("adapter package file became unavailable") from exc
            if not stat.S_ISREG(metadata.st_mode) or candidate.is_symlink():
                raise ContractError(
                    "adapter package_root may contain regular files only"
                )
            paths.add(candidate.relative_to(repository_root.resolve()).as_posix())
    if not paths:
        raise ContractError("adapter package_root must contain regular files")
    return paths


def _load_external_seal(
    contracts_root: Path,
    reference_value: Any,
    label: str,
) -> tuple[dict[str, Any], str]:
    reference = require_object(reference_value, label)
    if set(reference) != {"manifest_path", "manifest_file_sha256"}:
        raise ContractError(
            f"{label} must contain manifest_path and manifest_file_sha256 exactly"
        )
    relative = require_string(reference.get("manifest_path"), f"{label}.manifest_path")
    expected = validate_digest(
        reference.get("manifest_file_sha256"), f"{label}.manifest_file_sha256"
    )
    path = resolve_contract_file(contracts_root, relative, label)
    payload = read_stable_regular_file(path, label)
    observed = sha256_bytes(payload)
    if observed != expected:
        raise ContractError(f"{label} file SHA-256 does not match the resolved config")
    try:
        document = require_object(loads_json_strict(payload, label=label), label)
    except SealError as exc:
        raise ContractError(f"{label} is not a valid strict JSON seal: {exc}") from exc
    return document, observed


def _seal_entry_paths(document: Mapping[str, Any]) -> set[str]:
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ContractError("Git seal entries must be an array")
    return {
        require_string(require_object(entry, "Git seal entry").get("path"), "path")
        for entry in entries
    }


def enforce_production_execution_gate(
    model: Mapping[str, Any],
    adapter: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
    contracts_root: Path | None = None,
    resolved_config_path: Path | None = None,
    resolved_config_sha256: str | None = None,
    template_config: Mapping[str, Any] | None = None,
    required_source_paths: Iterable[str] = REQUIRED_SOURCE_SEAL_PATHS,
    required_package_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Bind enabled SPD execution to clean source, package, and environment seals."""

    if model.get("implementation_status") != "ready":
        raise ContractError("SPD canary model implementation_status must be ready")
    if (
        adapter.get("execution_enabled") is not True
        or adapter.get("execution_status") != "enabled"
    ):
        raise ContractError(
            "SPD adapter execution must set execution_enabled=true and status=enabled"
        )
    if (
        repository_root is None
        or contracts_root is None
        or resolved_config_path is None
        or resolved_config_sha256 is None
    ):
        raise ContractError(
            "enabled SPD adapter execution requires an external resolved config root"
        )
    repository_root = repository_root.resolve()
    if contracts_root.is_symlink():
        raise ContractError("contracts root may not be a symlink")
    ensure_read_only_contracts_root(contracts_root)
    contracts_root = contracts_root.resolve()
    resolved_config_path = resolved_config_path.resolve()
    try:
        resolved_config_path.relative_to(contracts_root)
    except ValueError as exc:
        raise ContractError(
            "enabled SPD config must be below the external contracts root"
        ) from exc
    expected_config_sha = validate_digest(
        resolved_config_sha256, "resolved SPD config SHA-256"
    )
    if sha256_file(resolved_config_path) != expected_config_sha:
        raise ContractError("resolved SPD config changed after it was parsed")
    commit, dirty = git_identity(repository_root)
    if dirty:
        raise ContractError("enabled SPD execution requires a clean Git working tree")

    source_document, source_file_sha = _load_external_seal(
        contracts_root, model.get("source_tree_seal"), "model.source_tree_seal"
    )
    package_document, package_file_sha = _load_external_seal(
        contracts_root, adapter.get("package_seal"), "adapter.package_seal"
    )
    environment_document, environment_file_sha = _load_external_seal(
        contracts_root,
        model.get("installed_environment_seal"),
        "model.installed_environment_seal",
    )
    try:
        source_validated = validate_git_seal(
            repository_root, source_document, expected_kind="source_tree"
        )
        package_validated = validate_git_seal(
            repository_root, package_document, expected_kind="package"
        )
        environment_validated = validate_installed_environment_manifest(
            environment_document, verify_current=True
        )
    except SealError as exc:
        raise ContractError(f"SPD execution provenance seal rejected: {exc}") from exc
    if (
        source_validated["git_commit"] != commit
        or package_validated["git_commit"] != commit
    ):
        raise ContractError("SPD source/package seals must bind the current Git HEAD")

    source_paths = _seal_entry_paths(source_validated)
    missing_sources = sorted(set(required_source_paths) - source_paths)
    if missing_sources:
        raise ContractError(f"SPD source-tree seal omits required paths: {missing_sources}")

    package_root_relative = require_string(
        adapter.get("package_root"), "adapter.package_root"
    )
    observed_package_paths = _package_tree_paths(repository_root, package_root_relative)
    package_paths = _seal_entry_paths(package_validated)
    if package_paths != observed_package_paths:
        raise ContractError(
            "SPD package seal must cover the complete adapter.package_root file set"
        )
    missing_package_paths = sorted(set(required_package_paths) - package_paths)
    if missing_package_paths:
        raise ContractError(
            f"SPD package seal omits required runtime paths: {missing_package_paths}"
        )

    template = require_object(template_config, "template_config")
    if set(template) != {"path", "sha256"}:
        raise ContractError("template_config must contain path and sha256 exactly")
    template_relative = require_string(template.get("path"), "template_config.path")
    if template_relative != TEMPLATE_CONFIG_PATH:
        raise ContractError(
            "template_config.path must name the repository SPD pending template"
        )
    template_path = resolve_repository_file(
        repository_root, template_relative, "SPD pending template config"
    )
    template_sha = validate_digest(template.get("sha256"), "template_config.sha256")
    template_payload = read_stable_regular_file(
        template_path, "SPD pending template config"
    )
    if sha256_bytes(template_payload) != template_sha:
        raise ContractError("SPD pending template config SHA-256 mismatch")
    try:
        template_document = require_object(
            loads_json_strict(template_payload, label="SPD pending template config"),
            "SPD pending template config",
        )
    except SealError as exc:
        raise ContractError("SPD pending template config is not strict JSON") from exc
    if (
        template_document.get("schema_version") != SCHEMA_VERSION
        or template_document.get("canary_type") != CANARY_TYPE
        or template_document.get("stage") != "canary"
    ):
        raise ContractError("repository SPD template identity is invalid")
    template_adapter = require_object(
        template_document.get("adapter"), "SPD pending template adapter"
    )
    template_execution_status = require_string(
        template_adapter.get("execution_status"),
        "SPD pending template adapter execution_status",
    )
    if (
        template_adapter.get("execution_enabled") is not False
        or not template_execution_status.startswith("disabled_")
    ):
        raise ContractError("repository SPD template must remain execution-disabled")
    if template_document.get("scientific_claim_allowed") is not False:
        raise ContractError("repository SPD template must remain diagnostic-only")
    template_evidence = require_object(
        template_document.get("evidence"), "SPD pending template evidence"
    )
    if template_evidence.get("registry_write_allowed") is not False:
        raise ContractError("repository SPD template must forbid registry writes")
    if template_relative not in source_paths:
        raise ContractError("SPD source-tree seal must include the pending template config")

    return {
        "resolved_config_sha256": expected_config_sha,
        "template_config_sha256": template_sha,
        "source_tree": {
            "file_sha256": source_file_sha,
            "manifest_sha256": source_validated["manifest_sha256"],
            "git_commit": source_validated["git_commit"],
        },
        "adapter_package": {
            "file_sha256": package_file_sha,
            "manifest_sha256": package_validated["manifest_sha256"],
            "git_commit": package_validated["git_commit"],
        },
        "installed_environment": {
            "file_sha256": environment_file_sha,
            "manifest_sha256": environment_validated["manifest_sha256"],
        },
    }


def verify_canary_head_sources(
    repository_root: Path,
    paths: Mapping[str, Path],
    *,
    expected_commit: str,
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    resolved_root = repository_root.resolve()
    for label, path in sorted(paths.items()):
        try:
            relative = path.resolve().relative_to(resolved_root).as_posix()
            observed = verify_head_blob(resolved_root, relative)
        except (ValueError, SealError) as exc:
            raise ContractError(
                f"{label} is not an unchanged ordinary blob at Git HEAD"
            ) from exc
        if observed.get("git_commit") != expected_commit:
            raise ContractError(f"{label} Git commit does not match the canary commit")
        evidence[label] = observed
    return evidence


def load_adapter(path: Path, factory_name: str, context: Mapping[str, Any]) -> Any:
    spec = importlib.util.spec_from_file_location("rtpv2x_real_canary_adapter", path)
    if spec is None or spec.loader is None:
        raise ContractError("cannot construct model adapter module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # adapter failures are evidence, not contract errors
        raise ContractError(
            f"model adapter import failed: {type(exc).__name__}"
        ) from exc
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ContractError(f"model adapter factory is not callable: {factory_name}")
    adapter = factory(dict(context))
    for method in (
        "forward",
        "backward",
        "optimizer_step",
        "evaluate",
        "save_checkpoint",
        "runtime_environment",
    ):
        if not callable(getattr(adapter, method, None)):
            raise ContractError(f"model adapter lacks callable method: {method}")
    return adapter


def json_safe(value: Any, label: str) -> Any:
    try:
        encoded = canonical_bytes(value)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not finite JSON") from exc
    sensitive_fragments = {
        "password",
        "secret",
        "api_key",
        "access_key",
        "authorization",
        "credential",
    }

    def inspect(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = str(key).lower()
                if any(fragment in lowered for fragment in sensitive_fragments):
                    raise ContractError(f"{label} contains a sensitive field")
                inspect(child)
        elif isinstance(node, list):
            for child in node:
                inspect(child)
        elif isinstance(node, str) and (
            node.startswith("/") or node.startswith("file://")
        ):
            raise ContractError(f"{label} contains a local absolute path")

    inspect(normalized)
    return normalized


def validate_metrics(
    metrics: object, expected_names: Iterable[str]
) -> dict[str, float | int]:
    result = require_object(metrics, "adapter metrics")
    expected = set(expected_names)
    if set(result) != expected:
        raise ContractError("adapter metrics do not match the frozen protocol exactly")
    normalized: dict[str, float | int] = {}
    for name, value in result.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ContractError(f"adapter metric is not finite numeric: {name}")
        normalized[name] = value
    return normalized


def evidence_hashes(output: Path) -> dict[str, str]:
    return {name: sha256_file(output / name) for name in REQUIRED_ARTIFACTS}


def run_canary(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(args.repository_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    ensure_read_only_dataset(dataset_root)
    contracts_root_value = getattr(args, "contracts_root", None)
    contracts_root = (
        Path(contracts_root_value) if contracts_root_value is not None else None
    )
    if contracts_root is None:
        config_path = resolve_repository_file(
            repository_root, args.config, "SPD canary config"
        )
    else:
        config_path = resolve_contract_file(
            contracts_root, args.config, "resolved SPD canary config"
        )
    config_payload = read_stable_regular_file(config_path, "SPD canary config")
    config_snapshot_sha256 = sha256_bytes(config_payload)
    try:
        config = require_object(
            loads_json_strict(config_payload, label="SPD canary config"),
            "SPD canary config",
        )
    except SealError as exc:
        raise ContractError(f"SPD canary config is not strict JSON: {exc}") from exc
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("SPD canary config schema_version must be 1")
    if config.get("canary_type") != CANARY_TYPE:
        raise ContractError(f"config canary_type must be {CANARY_TYPE}")
    if config.get("scientific_claim_allowed") is not False:
        raise ContractError("real-data canary must set scientific_claim_allowed=false")
    if config.get("stage") != "canary":
        raise ContractError("SPD diagnostic runner accepts stage=canary only")
    if getattr(args, "allow_dirty_diagnostic", False):
        raise ContractError(
            "enabled SPD production canary does not allow dirty diagnostics"
        )

    clearml = require_object(config.get("clearml"), "config.clearml")
    require_string(clearml.get("project"), "config.clearml.project")
    require_string(clearml.get("queue"), "config.clearml.queue")
    if clearml.get("require_task_id") is not True:
        raise ContractError("SPD canary config must require a ClearML task ID")
    clearml_task_id = validate_clearml_task_id(
        getattr(args, "clearml_task_id", None)
    )

    data = require_object(config.get("data"), "config.data")
    if data.get("dataset") != DATASET_NAME:
        raise ContractError(f"config.data.dataset must be {DATASET_NAME}")
    if data.get("read_only_mount_required") is not True:
        raise ContractError("config.data must require a read-only dataset mount")
    protocol_path = resolve_repository_file(
        repository_root,
        require_string(data.get("protocol_path"), "protocol_path"),
        "protocol",
    )
    identity_path = Path(
        require_string(data.get("identity_manifest"), "identity_manifest")
    ).resolve()
    split_path = Path(
        require_string(data.get("frozen_split"), "frozen_split")
    ).resolve()
    identity_expected_sha256 = validate_digest(
        data.get("identity_manifest_sha256"), "identity_manifest_sha256"
    )
    split_expected_sha256 = validate_digest(
        data.get("frozen_split_sha256"), "frozen_split_sha256"
    )
    protocol_document, protocol_snapshot_sha256 = load_json_snapshot(protocol_path)
    protocol = validate_protocol(protocol_document)
    protocol_id = protocol["protocol_id"]
    if config.get("protocol") != protocol_id:
        raise ContractError("config protocol does not match protocol document")
    identity, identity_snapshot_sha256 = load_json_snapshot(identity_path)
    if identity_snapshot_sha256 != identity_expected_sha256:
        raise ContractError(
            "SPD identity manifest SHA-256 does not match the resolved config"
        )
    identity_index = validate_identity(dataset_root, identity)
    split_document, split_snapshot_sha256 = load_json_snapshot(split_path)
    if split_snapshot_sha256 != split_expected_sha256:
        raise ContractError("SPD split SHA-256 does not match the resolved config")
    partitions = validate_split(split_document, protocol_id=protocol_id)
    training_sequence_id, training_max_frames = validate_sequence_request(
        data,
        "training",
        expected_partition="train",
        partitions=partitions,
    )
    evaluation_sequence_id, evaluation_max_frames = validate_sequence_request(
        data,
        "evaluation",
        expected_partition="validation",
        partitions=partitions,
    )
    if training_sequence_id == evaluation_sequence_id:
        raise ContractError("training and evaluation sequence IDs must differ")
    if training_max_frames != 1:
        raise ContractError(
            "config.data.training.max_frames must be 1 for the single-step canary"
        )
    sealed_sequences = identity.get("sequence_ids")
    if not isinstance(sealed_sequences, list) or not {
        training_sequence_id,
        evaluation_sequence_id,
    }.issubset(sealed_sequences):
        raise ContractError(
            "dataset identity must seal both training and evaluation sequences"
        )
    raw_required_assets = data.get("required_assets")
    if not isinstance(raw_required_assets, list) or not raw_required_assets:
        raise ContractError("config.data.required_assets must be a non-empty array")
    required_assets: list[tuple[str, str]] = []
    for index, value in enumerate(raw_required_assets):
        asset = require_object(value, f"required_assets[{index}]")
        if set(asset) != {"agent", "modality"}:
            raise ContractError(
                "each required asset must contain agent and modality exactly"
            )
        key = (
            require_string(asset["agent"], f"required_assets[{index}].agent"),
            require_string(asset["modality"], f"required_assets[{index}].modality"),
        )
        if key not in OFFICIAL_ASSET_LAYOUT:
            raise ContractError(
                f"required_assets[{index}] is not an official SPD sensor key"
            )
        required_assets.append(key)
    if len(required_assets) != len(set(required_assets)):
        raise ContractError("config.data.required_assets contains duplicates")
    training_frames = load_sequence(
        dataset_root,
        identity_index,
        sequence_id=training_sequence_id,
        max_frames=training_max_frames,
        required_assets=required_assets,
    )
    evaluation_frames = load_sequence(
        dataset_root,
        identity_index,
        sequence_id=evaluation_sequence_id,
        max_frames=evaluation_max_frames,
        required_assets=required_assets,
    )
    training_observation = training_frames[0].observation_payload(dataset_root)
    training_target = training_frames[0].target_payload()
    training_payload = training_frames[0].training_payload(dataset_root)
    inference_payloads = [
        frame.observation_payload(dataset_root) for frame in evaluation_frames
    ]
    evaluation_targets = [frame.target_payload() for frame in evaluation_frames]

    commit, dirty = git_identity(repository_root)
    if dirty:
        raise ContractError(
            "enabled SPD canary requires a clean committed working tree"
        )
    model = require_object(config.get("model"), "config.model")
    require_string(model.get("name"), "config.model.name")
    if model.get("implementation_status") != "ready":
        raise ContractError("SPD canary model implementation_status must be ready")
    if model.get("initialization") != "from_scratch":
        raise ContractError(
            "SPD canary model initialization must be frozen to from_scratch"
        )
    if (
        model.get("checkpoint_path") is not None
        or model.get("checkpoint_sha256") is not None
    ):
        raise ContractError(
            "from_scratch SPD canary cannot declare an input checkpoint"
        )
    adapter_config = require_object(config.get("adapter"), "config.adapter")
    if set(adapter_config) != {
        "execution_enabled",
        "execution_status",
        "factory",
        "package_root",
        "package_seal",
        "path",
        "settings",
        "sha256",
    }:
        raise ContractError("config.adapter fields do not match the SPD execution schema")
    adapter_relative = require_string(adapter_config.get("path"), "adapter.path")
    adapter_path = resolve_repository_file(
        repository_root,
        adapter_relative,
        "adapter",
    )
    adapter_expected_sha256 = validate_digest(
        adapter_config.get("sha256"), "adapter.sha256"
    )
    if sha256_file(adapter_path) != adapter_expected_sha256:
        raise ContractError("SPD adapter SHA-256 does not match the resolved config")
    execution_provenance = enforce_production_execution_gate(
        model,
        adapter_config,
        repository_root=repository_root,
        contracts_root=contracts_root,
        resolved_config_path=config_path,
        resolved_config_sha256=config_snapshot_sha256,
        template_config=config.get("template_config"),
        required_package_paths={adapter_relative},
    )
    runtime = require_object(config.get("runtime"), "config.runtime")
    if runtime.get("epochs") != 1 or runtime.get("batch_size") != 1:
        raise ContractError("SPD canary must freeze epochs=1 and batch_size=1")
    required_gpu = require_string(
        runtime.get("require_gpu_name_substring"), "require_gpu_name_substring"
    )
    gpu_environment = probe_gpu(required_gpu)
    evidence = require_object(config.get("evidence"), "config.evidence")
    required_evidence_flags = {
        "save_checkpoint",
        "save_metrics",
        "save_predictions",
        "sha256_manifest",
    }
    if any(evidence.get(key) is not True for key in required_evidence_flags):
        raise ContractError("SPD canary evidence flags must all be true")
    if evidence.get("registry_write_allowed") is not False:
        raise ContractError("SPD canary evidence must set registry_write_allowed=false")
    source_evidence = verify_canary_head_sources(
        repository_root,
        {
            "adapter": adapter_path,
            "provenance": Path(__file__).with_name("provenance_seals.py").resolve(),
            "protocol": protocol_path,
            "runner": Path(__file__).resolve(),
        },
        expected_commit=commit,
    )
    factory = require_string(adapter_config.get("factory"), "adapter.factory")
    adapter_settings = json_safe(
        require_object(adapter_config.get("settings", {}), "adapter.settings"),
        "adapter settings",
    )
    protocol_context = canonical_copy(
        {
            "protocol_id": protocol["protocol_id"],
            "status": protocol["status"],
            "dataset": protocol["dataset"],
            "causal_rule": protocol["causal_rule"],
            "time_model": protocol["time_model"],
            "coordinate_frame": protocol["coordinate_frame"],
            "classes": protocol["classes"],
            "fault_grid": protocol["fault_grid"],
            "tracking_metrics": protocol["tracking_metrics"],
        },
        "adapter protocol context",
    )
    context = canonical_copy(
        {
            "protocol": protocol_context,
            "adapter_settings": adapter_settings,
            "dataset_release_id": identity["release_id"],
            "training_sequence_id": training_sequence_id,
            "evaluation_sequence_id": evaluation_sequence_id,
        },
        "adapter context",
    )
    control_paths = {
        "adapter": adapter_path,
        "config": config_path,
        "dataset_identity": identity_path,
        "frozen_split": split_path,
        "protocol": protocol_path,
        "provenance": Path(__file__).with_name("provenance_seals.py").resolve(),
        "runner": Path(__file__).resolve(),
    }
    for label, path in control_paths.items():
        if not path.is_file() or path.is_symlink():
            raise ContractError(
                f"control file must be regular and non-symlink: {label}"
            )
    control_hashes = {
        "adapter": sha256_file(adapter_path),
        "config": config_snapshot_sha256,
        "dataset_identity": identity_snapshot_sha256,
        "frozen_split": split_snapshot_sha256,
        "protocol": protocol_snapshot_sha256,
        "provenance": sha256_file(control_paths["provenance"]),
        "runner": sha256_file(control_paths["runner"]),
    }
    payload_evidence = {
        "training_observation_sha256": sha256_bytes(
            canonical_bytes(training_observation)
        ),
        "training_target_sha256": sha256_bytes(canonical_bytes(training_target)),
        "evaluation_observation_sha256": [
            sha256_bytes(canonical_bytes(payload)) for payload in inference_payloads
        ],
        "evaluation_target_sha256": [
            sha256_bytes(canonical_bytes(target)) for target in evaluation_targets
        ],
    }
    adapter = load_adapter(adapter_path, factory, context)
    adapter_environment = require_object(
        json_safe(adapter.runtime_environment(), "adapter runtime environment"),
        "adapter runtime environment",
    )
    if set(adapter_environment) != {
        "device_name",
        "framework",
        "framework_version",
    }:
        raise ContractError(
            "adapter runtime environment must contain device_name, framework, and framework_version exactly"
        )
    if required_gpu not in require_string(
        adapter_environment.get("device_name"),
        "adapter runtime environment device_name",
    ):
        raise ContractError(
            "adapter runtime environment does not identify the required GPU"
        )
    require_string(
        adapter_environment.get("framework"),
        "adapter runtime environment framework",
    )
    require_string(
        adapter_environment.get("framework_version"),
        "adapter runtime environment framework_version",
    )

    output_argument = Path(args.output)
    if output_argument.is_symlink():
        raise ContractError("output directory may not be a symlink")
    output = output_argument.resolve()
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise ContractError("output path must be a regular directory")
        if any(output.iterdir()):
            raise ContractError("output directory must not already contain files")
    else:
        output.mkdir(parents=True)
    events: list[str] = []
    events.append("preflight_complete")
    train_output = adapter.forward(
        canonical_copy(training_payload, "adapter training payload"), training=True
    )
    events.append("forward_complete")
    backward_report = json_safe(adapter.backward(train_output), "backward report")
    events.append("backward_complete")
    adapter.optimizer_step()
    events.append("optimizer_step_complete")

    predictions = []
    for frame in inference_payloads:
        prediction = json_safe(
            adapter.forward(
                canonical_copy(frame, "adapter inference payload"), training=False
            ),
            "prediction",
        )
        predictions.append({"frame_id": frame["frame_id"], "prediction": prediction})
    events.append("evaluation_forward_complete")

    frozen_predictions = canonical_copy(predictions, "frozen predictions")
    predictions_payload = b"".join(canonical_bytes(row) for row in frozen_predictions)
    expected_predictions_sha256 = sha256_bytes(predictions_payload)
    predictions_path = output / "predictions.jsonl"
    atomic_write(predictions_path, predictions_payload)
    if sha256_file(predictions_path) != expected_predictions_sha256:
        raise ContractError(
            "persisted predictions do not match frozen prediction bytes"
        )

    checkpoint_path = output / "checkpoint.bin"
    adapter.save_checkpoint(str(checkpoint_path))
    if (
        not checkpoint_path.is_file()
        or checkpoint_path.is_symlink()
        or checkpoint_path.stat().st_size <= 0
    ):
        raise ContractError("adapter did not create a non-empty checkpoint.bin")
    if sha256_file(predictions_path) != expected_predictions_sha256:
        raise ContractError(
            "adapter modified frozen predictions while saving checkpoint"
        )
    events.append("checkpoint_complete")
    pre_target_artifact_hashes = {
        "checkpoint.bin": sha256_file(checkpoint_path),
        "predictions.jsonl": expected_predictions_sha256,
    }
    events.append("pre_target_artifacts_frozen")

    metric_values = validate_metrics(
        adapter.evaluate(
            canonical_copy(frozen_predictions, "evaluator prediction copy"),
            canonical_copy(evaluation_targets, "evaluator target copy"),
        ),
        protocol["tracking_metrics"],
    )
    events.append("evaluation_complete")
    for name, expected_hash in pre_target_artifact_hashes.items():
        path = output / name
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected_hash
        ):
            raise ContractError(f"adapter modified frozen pre-target artifact: {name}")

    post_commit, post_dirty = git_identity(repository_root)
    if post_commit != commit or post_dirty != dirty:
        raise ContractError("repository identity changed during canary execution")
    post_execution_provenance = enforce_production_execution_gate(
        model,
        adapter_config,
        repository_root=repository_root,
        contracts_root=contracts_root,
        resolved_config_path=config_path,
        resolved_config_sha256=config_snapshot_sha256,
        template_config=config.get("template_config"),
        required_package_paths={adapter_relative},
    )
    if canonical_bytes(post_execution_provenance) != canonical_bytes(
        execution_provenance
    ):
        raise ContractError("SPD execution provenance changed during canary execution")
    ensure_read_only_dataset(dataset_root)
    if validate_clearml_task_id(getattr(args, "clearml_task_id", None)) != clearml_task_id:
        raise ContractError("active ClearML task identity changed during execution")
    if canonical_bytes(probe_gpu(required_gpu)) != canonical_bytes(gpu_environment):
        raise ContractError("visible A100 GPU identity changed during execution")
    post_source_evidence = verify_canary_head_sources(
        repository_root,
        {
            "adapter": adapter_path,
            "provenance": Path(__file__).with_name("provenance_seals.py").resolve(),
            "protocol": protocol_path,
            "runner": Path(__file__).resolve(),
        },
        expected_commit=commit,
    )
    if post_source_evidence != source_evidence:
        raise ContractError("Git HEAD source evidence changed during execution")
    for label, path in control_paths.items():
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"control file changed type during execution: {label}")
        if sha256_file(path) != control_hashes[label]:
            raise ContractError(f"control file changed during execution: {label}")
    validate_identity(
        dataset_root, canonical_copy(identity, "dataset identity snapshot")
    )
    metrics_document = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "claim_scope": "diagnostic_only",
        "scientific_claim_allowed": False,
        "training_sequence_id": training_sequence_id,
        "evaluation_sequence_id": evaluation_sequence_id,
        "training_frame_count": len(training_frames),
        "evaluation_frame_count": len(evaluation_frames),
        "backward_report": backward_report,
        "metrics": metric_values,
    }
    atomic_write(output / "metrics.json", canonical_bytes(metrics_document))
    environment = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "diagnostic_only",
        "scientific_claim_allowed": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "adapter_sha256": control_hashes["adapter"],
        "gpu": gpu_environment,
        "adapter": adapter_environment,
        "installed_environment_seal": execution_provenance[
            "installed_environment"
        ],
    }
    atomic_write(output / "environment.json", canonical_bytes(environment))
    atomic_write(output / "stdout.log", ("\n".join(events) + "\n").encode("utf-8"))

    artifact_hashes = evidence_hashes(output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "diagnostic_only",
        "scientific_claim_allowed": False,
        "clearml_task_id": clearml_task_id,
        "protocol_id": protocol_id,
        "dataset_release_id": identity["release_id"],
        "dataset_identity_sha256": control_hashes["dataset_identity"],
        "frozen_split_sha256": control_hashes["frozen_split"],
        "protocol_sha256": control_hashes["protocol"],
        "config_sha256": control_hashes["config"],
        "template_config_sha256": execution_provenance[
            "template_config_sha256"
        ],
        "adapter_sha256": control_hashes["adapter"],
        "code_commit": commit,
        "execution_provenance": execution_provenance,
        "source_head_evidence": source_evidence,
        "dirty_diagnostic": dirty,
        "registry_write_allowed": False,
        "training_sequence_id": training_sequence_id,
        "evaluation_sequence_id": evaluation_sequence_id,
        "training_frame_ids": [frame.frame_id for frame in training_frames],
        "evaluation_frame_ids": [frame.frame_id for frame in evaluation_frames],
        "payload_evidence": payload_evidence,
        "pre_target_artifacts": pre_target_artifact_hashes,
        "artifacts": artifact_hashes,
    }
    atomic_write(output / "run_manifest.json", canonical_bytes(manifest))
    sums = "".join(
        f"{sha256_file(output / name)}  {name}\n"
        for name in sorted((*REQUIRED_ARTIFACTS, "run_manifest.json"))
    )
    atomic_write(output / "SHA256SUMS", sums.encode("ascii"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root", default=str(Path(__file__).resolve().parents[2])
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--config",
        required=True,
        help="repository-relative pending config, or contracts-root-relative resolved config",
    )
    parser.add_argument(
        "--contracts-root",
        help="external read-only root containing an enabled resolved config and provenance seals",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--clearml-task-id",
        default=os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"),
    )
    parser.add_argument(
        "--allow-dirty-diagnostic",
        action="store_true",
        help="deprecated fail-closed flag; enabled SPD execution always rejects dirty state",
    )
    args = parser.parse_args()
    try:
        manifest = run_canary(args)
    except ContractError as exc:
        raise SystemExit(f"real V2X-Seq canary rejected: {exc}") from exc
    print(
        json.dumps(
            {
                "claim_scope": manifest["claim_scope"],
                "evaluation_frame_count": len(manifest["evaluation_frame_ids"]),
                "protocol_id": manifest["protocol_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
