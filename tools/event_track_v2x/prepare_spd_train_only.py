#!/usr/bin/env python3
"""Materialize a fail-closed camera-only V2X-Seq-SPD training projection.

The source release stores train and validation metadata together.  This tool
therefore never uses ``ZipFile.extract``.  It audits every central-directory
entry, builds an exact train whitelist from the pinned official split, writes
only whitelisted members, and creates filtered copies of all three
``data_info.json`` tables.  Point-cloud payloads are never opened or written.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from typing import BinaryIO
import zipfile


SCHEMA_VERSION = 1
DOCUMENT_TYPE = "event_track_v2x_spd_train_only_manifest"
DATASET_NAME = "V2X-Seq-SPD"
OFFICIAL_SPLIT_SHA256 = (
    "4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3"
)
OFFICIAL_TRAIN_COUNTS = {
    "batch_split": 46,
    "vehicle_split": 8504,
    "infrastructure_split": 7834,
    "cooperative_split": 7445,
}

METADATA_ARCHIVE = "V2X-Seq-SPD.zip"
INFRASTRUCTURE_IMAGE_ARCHIVES = (
    "V2X-Seq-SPD-infrastructure-side-image-00_11616163645886464.zip",
    "V2X-Seq-SPD-infrastructure-side-image-01_11616163645886464.zip",
)
VEHICLE_IMAGE_ARCHIVE = "V2X-Seq-SPD-vehicle-side-image_11616163645886464.zip"
SOURCE_ARCHIVES = (
    *INFRASTRUCTURE_IMAGE_ARCHIVES,
    VEHICLE_IMAGE_ARCHIVE,
    METADATA_ARCHIVE,
)

_SPLIT_SECTIONS = (
    "batch_split",
    "vehicle_split",
    "infrastructure_split",
    "cooperative_split",
)
_SPLIT_NAMES = ("train", "val", "test", "test_A")
_FRAME_ID = re.compile(r"[0-9]{6}")
_SEQUENCE_ID = re.compile(r"[0-9]{4}")
_MAP_MEMBER = re.compile(r"V2X-Seq-SPD/maps/yizhuang[0-9]{2}\.json")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ALLOWED_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_MAX_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 250_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 64 * 1024 * 1024 * 1024
_COPY_CHUNK_BYTES = 8 * 1024 * 1024

_SIDE_PATHS: dict[str, dict[str, str]] = {
    "vehicle": {
        "image_path": "image/{frame}.jpg",
        "pointcloud_path": "velodyne/{frame}.pcd",
        "calib_camera_intrinsic_path": "calib/camera_intrinsic/{frame}.json",
        "calib_lidar_to_camera_path": "calib/lidar_to_camera/{frame}.json",
        "calib_lidar_to_novatel_path": "calib/lidar_to_novatel/{frame}.json",
        "calib_novatel_to_world_path": "calib/novatel_to_world/{frame}.json",
        "label_camera_std_path": "label/camera/{frame}.json",
        "label_lidar_std_path": "label/lidar/{frame}.json",
    },
    "infrastructure": {
        "image_path": "image/{frame}.jpg",
        "pointcloud_path": "velodyne/{frame}.pcd",
        "calib_camera_intrinsic_path": "calib/camera_intrinsic/{frame}.json",
        "calib_virtuallidar_to_camera_path": (
            "calib/virtuallidar_to_camera/{frame}.json"
        ),
        "calib_virtuallidar_to_world_path": (
            "calib/virtuallidar_to_world/{frame}.json"
        ),
        "label_camera_std_path": "label/camera/{frame}.json",
        "label_lidar_std_path": "label/virtuallidar/{frame}.json",
    },
}


class TrainOnlyPreparationError(ValueError):
    """Raised before a source can produce an unsafe or ambiguous projection."""


@dataclass(frozen=True)
class SourceFingerprint:
    path: Path
    name: str
    size_bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int

    def manifest_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ZipCatalog:
    files: Mapping[str, zipfile.ZipInfo]
    directories: frozenset[str]


@dataclass(frozen=True)
class PlannedMember:
    archive_name: str
    member_name: str
    target: str
    kind: str
    size_bytes: int
    crc32: str

    def plan_record(self) -> dict[str, object]:
        return {
            "archive": self.archive_name,
            "crc32": self.crc32,
            "member": self.member_name,
            "path": self.target,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class PreparationPlan:
    archive_dir: Path
    output: Path
    split_name: str
    split_size_bytes: int
    split_sha256: str
    split_document: Mapping[str, Mapping[str, tuple[str, ...]]]
    sources: tuple[SourceFingerprint, ...]
    catalogs: Mapping[str, ZipCatalog]
    members: tuple[PlannedMember, ...]
    data_info_payloads: Mapping[str, bytes]
    counts: Mapping[str, int]
    plan_sha256: str
    planned_payload_bytes: int


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainOnlyPreparationError("value is outside canonical JSON") from exc


def _content_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_json(raw: bytes, context: str) -> object:
    def reject_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise TrainOnlyPreparationError(
                    f"duplicate JSON key in {context}: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise TrainOnlyPreparationError(
            f"non-finite JSON constant in {context}: {value}"
        )

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainOnlyPreparationError(f"invalid UTF-8 JSON: {context}") from exc


def _regular_file(path: Path, context: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise TrainOnlyPreparationError(f"missing {context}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TrainOnlyPreparationError(
            f"{context} must be a regular non-symlink file: {path}"
        )
    return metadata


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(_COPY_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path, expected_name: str) -> SourceFingerprint:
    if path.name != expected_name:
        raise TrainOnlyPreparationError(
            f"source archive basename mismatch: expected {expected_name}"
        )
    before = _regular_file(path, f"source archive {expected_name}")
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise TrainOnlyPreparationError(
                    f"source archive changed while opening: {expected_name}"
                )
            digest = _sha256_stream(stream)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise TrainOnlyPreparationError(
            f"cannot read source archive: {expected_name}"
        ) from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(opened, field) != getattr(after, field) for field in stable_fields):
        raise TrainOnlyPreparationError(
            f"source archive changed while hashing: {expected_name}"
        )
    return SourceFingerprint(
        path=path,
        name=expected_name,
        size_bytes=after.st_size,
        sha256=digest,
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
    )


def _assert_source_unchanged(source: SourceFingerprint) -> None:
    metadata = _regular_file(source.path, f"source archive {source.name}")
    observed = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
    expected = (
        source.device,
        source.inode,
        source.size_bytes,
        source.mtime_ns,
    )
    if observed != expected:
        raise TrainOnlyPreparationError(
            f"source archive changed after hashing: {source.name}"
        )


def _canonical_relative_path(value: object, context: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise TrainOnlyPreparationError(f"{context} must be a trimmed non-empty string")
    if (
        value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or "\x00" in value
        or "//" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise TrainOnlyPreparationError(
            f"{context} is not a canonical POSIX relative path"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise TrainOnlyPreparationError(
            f"{context} is not a canonical POSIX relative path"
        )
    return value


def _member_name(info: zipfile.ZipInfo, archive_name: str) -> tuple[str, bool]:
    raw = info.filename
    is_directory = raw.endswith("/")
    candidate = raw[:-1] if is_directory else raw
    name = _canonical_relative_path(candidate, f"member in {archive_name}")
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if is_directory:
        if file_type not in (0, stat.S_IFDIR):
            raise TrainOnlyPreparationError(
                f"special or mismatched directory member in {archive_name}: {raw}"
            )
    elif file_type not in (0, stat.S_IFREG):
        raise TrainOnlyPreparationError(
            f"special archive member in {archive_name}: {raw}"
        )
    if info.flag_bits & 0x1:
        raise TrainOnlyPreparationError(
            f"encrypted archive member is forbidden in {archive_name}: {raw}"
        )
    if info.compress_type not in _ALLOWED_COMPRESSION:
        raise TrainOnlyPreparationError(
            f"unsupported compression method in {archive_name}: {raw}"
        )
    if info.file_size < 0 or info.file_size > _MAX_MEMBER_BYTES:
        raise TrainOnlyPreparationError(
            f"archive member size is unsafe in {archive_name}: {raw}"
        )
    if info.file_size and info.compress_size <= 0:
        raise TrainOnlyPreparationError(
            f"archive member has an invalid compressed size: {raw}"
        )
    if info.compress_size and info.file_size / info.compress_size > 10_000:
        raise TrainOnlyPreparationError(
            f"archive member compression ratio is unsafe: {raw}"
        )
    return name, is_directory


def _catalog(source: SourceFingerprint) -> ZipCatalog:
    _assert_source_unchanged(source)
    try:
        with zipfile.ZipFile(source.path, "r") as archive:
            infos = archive.infolist()
    except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise TrainOnlyPreparationError(
            f"cannot read ZIP central directory: {source.name}"
        ) from exc
    if not infos or len(infos) > _MAX_ARCHIVE_MEMBERS:
        raise TrainOnlyPreparationError(
            f"archive member count is unsafe: {source.name}"
        )
    files: dict[str, zipfile.ZipInfo] = {}
    directories: set[str] = set()
    seen_casefold: set[str] = set()
    total_size = 0
    for info in infos:
        name, is_directory = _member_name(info, source.name)
        folded = name.casefold()
        if folded in seen_casefold:
            raise TrainOnlyPreparationError(
                f"duplicate or case-colliding member in {source.name}: {name}"
            )
        seen_casefold.add(folded)
        if is_directory:
            directories.add(name)
        else:
            files[name] = info
            total_size += info.file_size
    if total_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise TrainOnlyPreparationError(
            f"archive uncompressed size is unsafe: {source.name}"
        )
    _assert_source_unchanged(source)
    return ZipCatalog(files=files, directories=frozenset(directories))


def _read_zip_members(
    source: SourceFingerprint,
    catalog: ZipCatalog,
    names: Sequence[str],
) -> dict[str, bytes]:
    missing = sorted(set(names) - set(catalog.files))
    if missing:
        raise TrainOnlyPreparationError(
            f"required members missing from {source.name}: {missing[:5]}"
        )
    result: dict[str, bytes] = {}
    _assert_source_unchanged(source)
    try:
        with zipfile.ZipFile(source.path, "r") as archive:
            for name in names:
                info = catalog.files[name]
                with archive.open(info, "r") as stream:
                    raw = stream.read(_MAX_MEMBER_BYTES + 1)
                if len(raw) != info.file_size or len(raw) > _MAX_MEMBER_BYTES:
                    raise TrainOnlyPreparationError(
                        f"member size mismatch while reading {source.name}: {name}"
                    )
                result[name] = raw
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
        raise TrainOnlyPreparationError(
            f"cannot read required members from {source.name}"
        ) from exc
    _assert_source_unchanged(source)
    return result


def _identifier(
    value: object,
    pattern: re.Pattern[str],
    context: str,
) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise TrainOnlyPreparationError(f"{context} is not canonical")
    return value


def _validate_split(
    document: object,
    *,
    enforce_official_counts: bool,
) -> dict[str, dict[str, tuple[str, ...]]]:
    if type(document) is not dict or set(document) != set(_SPLIT_SECTIONS):
        raise TrainOnlyPreparationError(
            "split document has missing or unknown sections"
        )
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for section in _SPLIT_SECTIONS:
        raw_section = document[section]
        if type(raw_section) is not dict or set(raw_section) != set(_SPLIT_NAMES):
            raise TrainOnlyPreparationError(
                f"split section {section} has missing or unknown partitions"
            )
        pattern = _SEQUENCE_ID if section == "batch_split" else _FRAME_ID
        partitions: dict[str, tuple[str, ...]] = {}
        for split_name in _SPLIT_NAMES:
            raw_ids = raw_section[split_name]
            if type(raw_ids) is not list or not raw_ids:
                raise TrainOnlyPreparationError(
                    f"{section}.{split_name} must be a non-empty array"
                )
            ids = tuple(
                _identifier(item, pattern, f"{section}.{split_name} ID")
                for item in raw_ids
            )
            if len(ids) != len(set(ids)):
                raise TrainOnlyPreparationError(
                    f"{section}.{split_name} contains duplicate IDs"
                )
            partitions[split_name] = ids
        train = set(partitions["train"])
        val = set(partitions["val"])
        test = set(partitions["test"])
        test_a = set(partitions["test_A"])
        if train & (val | test | test_a) or val & (test | test_a):
            raise TrainOnlyPreparationError(
                f"{section} has cross-partition train/validation leakage"
            )
        if not test_a <= test:
            raise TrainOnlyPreparationError(
                f"{section}.test_A must be a subset of test"
            )
        if enforce_official_counts and len(train) != OFFICIAL_TRAIN_COUNTS[section]:
            raise TrainOnlyPreparationError(f"official {section}.train count mismatch")
        result[section] = partitions
    return result


def _data_info_rows(value: object, relative: str) -> list[dict[str, object]]:
    if type(value) is not list or not value:
        raise TrainOnlyPreparationError(f"{relative} must be a non-empty array")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(value):
        if type(row) is not dict or not all(type(key) is str for key in row):
            raise TrainOnlyPreparationError(f"{relative}[{index}] must be an object")
        rows.append(row)
    return rows


def _side_rows(
    value: object,
    *,
    side: str,
    split: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    set[str],
]:
    relative = f"{side}-side/data_info.json"
    rows = _data_info_rows(value, relative)
    expected_paths = _SIDE_PATHS[side]
    observed: dict[str, dict[str, object]] = {}
    all_split_ids = set().union(*map(set, split[f"{side}_split"].values()))
    for index, row in enumerate(rows):
        context = f"{relative}[{index}]"
        frame = _identifier(row.get("frame_id"), _FRAME_ID, f"{context}.frame_id")
        sequence = _identifier(
            row.get("sequence_id"), _SEQUENCE_ID, f"{context}.sequence_id"
        )
        if frame in observed:
            raise TrainOnlyPreparationError(f"duplicate {side} frame: {frame}")
        if frame not in all_split_ids:
            raise TrainOnlyPreparationError(
                f"{context} is absent from the official split"
            )
        row_path_keys = {key for key in row if key.endswith("_path")}
        if row_path_keys != set(expected_paths):
            raise TrainOnlyPreparationError(f"{context} path fields mismatch")
        for key, template in expected_paths.items():
            path = _canonical_relative_path(row[key], f"{context}.{key}")
            if path != template.format(frame=frame):
                raise TrainOnlyPreparationError(
                    f"{context}.{key} disagrees with frame_id"
                )
        row["sequence_id"] = sequence
        observed[frame] = row
    train_ids = set(split[f"{side}_split"]["train"])
    missing_train = sorted(train_ids - set(observed))
    if missing_train:
        raise TrainOnlyPreparationError(
            f"{relative} is missing train frames: {missing_train[:5]}"
        )
    filtered = [row for row in rows if row["frame_id"] in train_ids]
    train_sequences = {str(row["sequence_id"]) for row in filtered}
    expected_sequences = set(split["batch_split"]["train"])
    if train_sequences != expected_sequences:
        raise TrainOnlyPreparationError(
            f"{relative} train sequences disagree with batch_split.train"
        )
    return filtered, observed, train_ids


def _cooperative_rows(
    value: object,
    *,
    split: Mapping[str, Mapping[str, tuple[str, ...]]],
    vehicle_rows: Mapping[str, Mapping[str, object]],
    infrastructure_rows: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], set[str]]:
    relative = "cooperative/data_info.json"
    rows = _data_info_rows(value, relative)
    required = {
        "vehicle_frame",
        "infrastructure_frame",
        "vehicle_sequence",
        "infrastructure_sequence",
        "system_error_offset",
    }
    observed: dict[str, dict[str, object]] = {}
    observed_infrastructure: set[str] = set()
    all_split_ids = set().union(*map(set, split["cooperative_split"].values()))
    for index, row in enumerate(rows):
        context = f"{relative}[{index}]"
        if not required <= set(row):
            raise TrainOnlyPreparationError(f"{context} is missing required fields")
        vehicle_frame = _identifier(
            row["vehicle_frame"], _FRAME_ID, f"{context}.vehicle_frame"
        )
        infrastructure_frame = _identifier(
            row["infrastructure_frame"],
            _FRAME_ID,
            f"{context}.infrastructure_frame",
        )
        vehicle_sequence = _identifier(
            row["vehicle_sequence"], _SEQUENCE_ID, f"{context}.vehicle_sequence"
        )
        infrastructure_sequence = _identifier(
            row["infrastructure_sequence"],
            _SEQUENCE_ID,
            f"{context}.infrastructure_sequence",
        )
        if vehicle_sequence != infrastructure_sequence:
            raise TrainOnlyPreparationError(f"{context} joins different sequences")
        if vehicle_frame in observed:
            raise TrainOnlyPreparationError(
                f"duplicate cooperative vehicle frame: {vehicle_frame}"
            )
        if infrastructure_frame in observed_infrastructure:
            raise TrainOnlyPreparationError(
                f"duplicate cooperative infrastructure frame: {infrastructure_frame}"
            )
        if vehicle_frame not in all_split_ids:
            raise TrainOnlyPreparationError(
                f"{context} is absent from cooperative_split"
            )
        vehicle_row = vehicle_rows.get(vehicle_frame)
        infrastructure_row = infrastructure_rows.get(infrastructure_frame)
        if vehicle_row is None or infrastructure_row is None:
            raise TrainOnlyPreparationError(
                f"{context} references an unknown side frame"
            )
        if vehicle_row["sequence_id"] != vehicle_sequence:
            raise TrainOnlyPreparationError(
                f"{context}.vehicle_sequence disagrees with vehicle data_info"
            )
        if infrastructure_row["sequence_id"] != infrastructure_sequence:
            raise TrainOnlyPreparationError(
                f"{context}.infrastructure_sequence disagrees with infrastructure data_info"
            )
        observed[vehicle_frame] = row
        observed_infrastructure.add(infrastructure_frame)
    train_ids = set(split["cooperative_split"]["train"])
    missing_train = sorted(train_ids - set(observed))
    if missing_train:
        raise TrainOnlyPreparationError(
            f"cooperative data_info is missing train frames: {missing_train[:5]}"
        )
    filtered = [row for row in rows if row["vehicle_frame"] in train_ids]
    train_sequences = {str(row["vehicle_sequence"]) for row in filtered}
    if train_sequences != set(split["batch_split"]["train"]):
        raise TrainOnlyPreparationError(
            "cooperative train sequences disagree with batch_split.train"
        )
    return filtered, observed, train_ids


def _image_frames(
    catalog: ZipCatalog,
    *,
    archive_name: str,
    member_root: str,
) -> dict[str, zipfile.ZipInfo]:
    pattern = re.compile(re.escape(member_root) + r"/([0-9]{6})\.jpg")
    frames: dict[str, zipfile.ZipInfo] = {}
    for name, info in catalog.files.items():
        match = pattern.fullmatch(name)
        if match is None:
            raise TrainOnlyPreparationError(
                f"unexpected image archive member in {archive_name}: {name}"
            )
        frame = match.group(1)
        if frame in frames:
            raise TrainOnlyPreparationError(
                f"duplicate image frame in {archive_name}: {frame}"
            )
        frames[frame] = info
    allowed_directories = {member_root}
    if not catalog.directories <= allowed_directories:
        unexpected = sorted(catalog.directories - allowed_directories)
        raise TrainOnlyPreparationError(
            f"unexpected image archive directories in {archive_name}: {unexpected[:5]}"
        )
    return frames


def _all_parent_directories(paths: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        pure = PurePosixPath(path)
        for index in range(1, len(pure.parts)):
            result.add(PurePosixPath(*pure.parts[:index]).as_posix())
    return result


def _hash_records(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json_bytes(record))
        digest.update(b"\n")
    return digest.hexdigest()


def _prepare_output_path(output: str | Path) -> Path:
    requested = Path(output).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    if requested.exists() or requested.is_symlink():
        raise TrainOnlyPreparationError(f"refusing to overwrite output: {requested}")
    destination = requested.resolve(strict=False)
    try:
        parent = destination.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TrainOnlyPreparationError(
            f"output parent must already exist: {destination.parent}"
        ) from exc
    if not parent.is_dir():
        raise TrainOnlyPreparationError(f"output parent is not a directory: {parent}")
    destination = parent / destination.name
    if destination.exists() or destination.is_symlink():
        raise TrainOnlyPreparationError(f"refusing to overwrite output: {destination}")
    return destination


def build_preparation_plan(
    archive_dir: str | Path,
    split_path: str | Path,
    output: str | Path,
    *,
    expected_split_sha256: str | None = OFFICIAL_SPLIT_SHA256,
) -> PreparationPlan:
    """Audit all sources and return an immutable exact-write plan."""

    archive_root_input = Path(archive_dir).expanduser()
    if archive_root_input.is_symlink():
        raise TrainOnlyPreparationError("archive directory must not be a symlink")
    try:
        archive_root = archive_root_input.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TrainOnlyPreparationError(
            f"archive directory does not exist: {archive_root_input}"
        ) from exc
    if not archive_root.is_dir():
        raise TrainOnlyPreparationError("archive directory must be a directory")
    destination = _prepare_output_path(output)

    split_file = Path(split_path).expanduser()
    split_metadata = _regular_file(split_file, "official split file")
    try:
        split_raw = split_file.read_bytes()
    except OSError as exc:
        raise TrainOnlyPreparationError("cannot read official split file") from exc
    split_after = _regular_file(split_file, "official split file")
    if (
        split_metadata.st_dev,
        split_metadata.st_ino,
        split_metadata.st_size,
        split_metadata.st_mtime_ns,
    ) != (
        split_after.st_dev,
        split_after.st_ino,
        split_after.st_size,
        split_after.st_mtime_ns,
    ):
        raise TrainOnlyPreparationError("official split changed while reading")
    split_sha256 = hashlib.sha256(split_raw).hexdigest()
    if expected_split_sha256 is not None:
        if _SHA256.fullmatch(expected_split_sha256) is None:
            raise TrainOnlyPreparationError("expected split SHA-256 is invalid")
        if split_sha256 != expected_split_sha256:
            raise TrainOnlyPreparationError(
                "official split SHA-256 mismatch; refusing an unpinned train definition"
            )
    split = _validate_split(
        _load_json(split_raw, "official split file"),
        enforce_official_counts=expected_split_sha256 is not None,
    )

    sources = tuple(_fingerprint(archive_root / name, name) for name in SOURCE_ARCHIVES)
    source_by_name = {source.name: source for source in sources}
    catalogs = {source.name: _catalog(source) for source in sources}

    metadata_members = tuple(
        f"V2X-Seq-SPD/{relative}"
        for relative in (
            "vehicle-side/data_info.json",
            "infrastructure-side/data_info.json",
            "cooperative/data_info.json",
        )
    )
    metadata_raw = _read_zip_members(
        source_by_name[METADATA_ARCHIVE],
        catalogs[METADATA_ARCHIVE],
        metadata_members,
    )
    vehicle_filtered, vehicle_all, vehicle_train = _side_rows(
        _load_json(metadata_raw[metadata_members[0]], metadata_members[0]),
        side="vehicle",
        split=split,
    )
    infrastructure_filtered, infrastructure_all, infrastructure_train = _side_rows(
        _load_json(metadata_raw[metadata_members[1]], metadata_members[1]),
        side="infrastructure",
        split=split,
    )
    cooperative_filtered, cooperative_all, cooperative_train = _cooperative_rows(
        _load_json(metadata_raw[metadata_members[2]], metadata_members[2]),
        split=split,
        vehicle_rows=vehicle_all,
        infrastructure_rows=infrastructure_all,
    )
    for row in cooperative_filtered:
        if row["vehicle_frame"] not in vehicle_train:
            raise TrainOnlyPreparationError(
                "cooperative train row references a non-train vehicle frame"
            )
        if row["infrastructure_frame"] not in infrastructure_train:
            raise TrainOnlyPreparationError(
                "cooperative train row references a non-train infrastructure frame"
            )

    expected_metadata_files = set(metadata_members)
    for side, rows in (
        ("vehicle", vehicle_all.values()),
        ("infrastructure", infrastructure_all.values()),
    ):
        for row in rows:
            for key in _SIDE_PATHS[side]:
                if key.startswith("calib_") or key.startswith("label_"):
                    expected_metadata_files.add(f"V2X-Seq-SPD/{side}-side/{row[key]}")
    expected_metadata_files.update(
        f"V2X-Seq-SPD/cooperative/label/{frame}.json" for frame in cooperative_all
    )
    metadata_catalog = catalogs[METADATA_ARCHIVE]
    ignored_metadata_files = {
        name
        for name in metadata_catalog.files
        if name == "V2X-Seq-SPD/README.md" or _MAP_MEMBER.fullmatch(name)
    }
    unexpected_metadata = sorted(
        set(metadata_catalog.files) - expected_metadata_files - ignored_metadata_files
    )
    missing_metadata = sorted(expected_metadata_files - set(metadata_catalog.files))
    if unexpected_metadata:
        raise TrainOnlyPreparationError(
            f"unexpected metadata archive members: {unexpected_metadata[:5]}"
        )
    if missing_metadata:
        raise TrainOnlyPreparationError(
            f"required frame metadata members are missing: {missing_metadata[:5]}"
        )
    allowed_directories = _all_parent_directories(
        sorted(expected_metadata_files | ignored_metadata_files)
    )
    if not metadata_catalog.directories <= allowed_directories:
        unexpected = sorted(metadata_catalog.directories - allowed_directories)
        raise TrainOnlyPreparationError(
            f"unexpected metadata archive directories: {unexpected[:5]}"
        )

    infrastructure_images: dict[str, tuple[str, str, zipfile.ZipInfo]] = {}
    for archive_name in INFRASTRUCTURE_IMAGE_ARCHIVES:
        member_root = archive_name.split("_11616163645886464.zip", maxsplit=1)[0]
        frames = _image_frames(
            catalogs[archive_name],
            archive_name=archive_name,
            member_root=member_root,
        )
        for frame, info in frames.items():
            if frame in infrastructure_images:
                raise TrainOnlyPreparationError(
                    f"duplicate infrastructure image frame across archives: {frame}"
                )
            infrastructure_images[frame] = (
                archive_name,
                f"{member_root}/{frame}.jpg",
                info,
            )
    vehicle_member_root = "V2X-Seq-SPD-vehicle-side-image"
    vehicle_catalog_frames = _image_frames(
        catalogs[VEHICLE_IMAGE_ARCHIVE],
        archive_name=VEHICLE_IMAGE_ARCHIVE,
        member_root=vehicle_member_root,
    )
    vehicle_images = {
        frame: (
            VEHICLE_IMAGE_ARCHIVE,
            f"{vehicle_member_root}/{frame}.jpg",
            info,
        )
        for frame, info in vehicle_catalog_frames.items()
    }
    if set(infrastructure_images) != set(infrastructure_all):
        raise TrainOnlyPreparationError(
            "infrastructure image archives disagree with infrastructure data_info"
        )
    if set(vehicle_images) != set(vehicle_all):
        raise TrainOnlyPreparationError(
            "vehicle image archive disagrees with vehicle data_info"
        )

    planned: list[PlannedMember] = []
    counts: dict[str, int] = {
        "infrastructure_calibration": 0,
        "infrastructure_images": 0,
        "infrastructure_labels": 0,
        "vehicle_calibration": 0,
        "vehicle_images": 0,
        "vehicle_labels": 0,
        "cooperative_labels": 0,
        "data_info": 3,
    }

    def add_member(
        archive_name: str,
        member_name: str,
        target: str,
        kind: str,
    ) -> None:
        info = catalogs[archive_name].files.get(member_name)
        if info is None:
            raise TrainOnlyPreparationError(
                f"planned source member is missing from {archive_name}: {member_name}"
            )
        planned.append(
            PlannedMember(
                archive_name=archive_name,
                member_name=member_name,
                target=_canonical_relative_path(target, "planned output path"),
                kind=kind,
                size_bytes=info.file_size,
                crc32=f"{info.CRC:08x}",
            )
        )
        counts[kind] += 1

    for frame in sorted(infrastructure_train):
        archive_name, member_name, _ = infrastructure_images[frame]
        add_member(
            archive_name,
            member_name,
            f"V2X-Seq-SPD/infrastructure-side/image/{frame}.jpg",
            "infrastructure_images",
        )
    for frame in sorted(vehicle_train):
        archive_name, member_name, _ = vehicle_images[frame]
        add_member(
            archive_name,
            member_name,
            f"V2X-Seq-SPD/vehicle-side/image/{frame}.jpg",
            "vehicle_images",
        )
    for side, filtered in (
        ("infrastructure", infrastructure_filtered),
        ("vehicle", vehicle_filtered),
    ):
        for row in filtered:
            for key in _SIDE_PATHS[side]:
                if not (key.startswith("calib_") or key.startswith("label_")):
                    continue
                relative = f"{side}-side/{row[key]}"
                kind = (
                    f"{side}_{'calibration' if key.startswith('calib_') else 'labels'}"
                )
                add_member(
                    METADATA_ARCHIVE,
                    f"V2X-Seq-SPD/{relative}",
                    f"V2X-Seq-SPD/{relative}",
                    kind,
                )
    for frame in sorted(cooperative_train):
        relative = f"V2X-Seq-SPD/cooperative/label/{frame}.json"
        add_member(
            METADATA_ARCHIVE,
            relative,
            relative,
            "cooperative_labels",
        )

    planned.sort(key=lambda item: item.target)
    target_casefold: set[str] = set()
    for item in planned:
        folded = item.target.casefold()
        if folded in target_casefold:
            raise TrainOnlyPreparationError(
                f"duplicate or case-colliding output path: {item.target}"
            )
        target_casefold.add(folded)
        if "velodyne" in PurePosixPath(item.target).parts or item.target.endswith(
            ".pcd"
        ):
            raise TrainOnlyPreparationError(
                f"point-cloud output entered the write plan: {item.target}"
            )

    data_info_payloads = {
        "V2X-Seq-SPD/vehicle-side/data_info.json": (
            _canonical_json_bytes(vehicle_filtered) + b"\n"
        ),
        "V2X-Seq-SPD/infrastructure-side/data_info.json": (
            _canonical_json_bytes(infrastructure_filtered) + b"\n"
        ),
        "V2X-Seq-SPD/cooperative/data_info.json": (
            _canonical_json_bytes(cooperative_filtered) + b"\n"
        ),
    }
    if target_casefold & {path.casefold() for path in data_info_payloads}:
        raise TrainOnlyPreparationError("data_info path collides with archive payload")

    plan_records = [item.plan_record() for item in planned]
    plan_records.extend(
        {
            "archive": None,
            "crc32": None,
            "member": None,
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for path, payload in sorted(data_info_payloads.items())
    )
    plan_records.sort(key=lambda item: str(item["path"]))
    plan_sha256 = _hash_records(plan_records)
    planned_payload_bytes = sum(item.size_bytes for item in planned) + sum(
        len(payload) for payload in data_info_payloads.values()
    )
    counts["archive_payload_files"] = len(planned)
    counts["output_payload_files"] = len(planned) + len(data_info_payloads)
    counts["infrastructure_train_frames"] = len(infrastructure_train)
    counts["vehicle_train_frames"] = len(vehicle_train)
    counts["cooperative_train_frames"] = len(cooperative_train)
    counts["frame_json"] = (
        counts["infrastructure_calibration"]
        + counts["infrastructure_labels"]
        + counts["vehicle_calibration"]
        + counts["vehicle_labels"]
        + counts["cooperative_labels"]
    )

    return PreparationPlan(
        archive_dir=archive_root,
        output=destination,
        split_name=split_file.name,
        split_size_bytes=len(split_raw),
        split_sha256=split_sha256,
        split_document=split,
        sources=sources,
        catalogs=catalogs,
        members=tuple(planned),
        data_info_payloads=data_info_payloads,
        counts=counts,
        plan_sha256=plan_sha256,
        planned_payload_bytes=planned_payload_bytes,
    )


def _manifest(
    plan: PreparationPlan,
    *,
    status: str,
    output_tree_sha256: str | None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "document_type": DOCUMENT_TYPE,
        "dataset": DATASET_NAME,
        "subset": "official-train-camera-only",
        "status": status,
        "release_identity_status": "unverified-source-bytes",
        "source_split": {
            "name": plan.split_name,
            "size_bytes": plan.split_size_bytes,
            "sha256": plan.split_sha256,
        },
        "source_archives": [
            source.manifest_record()
            for source in sorted(plan.sources, key=lambda item: item.name)
        ],
        "policy": {
            "included_partition": "train",
            "excluded_partitions": ["val", "test", "test_A"],
            "included_payloads": ["image", "calibration_json", "label_json"],
            "point_cloud_payloads_included": False,
            "data_info_projection": "train-records-only",
        },
        "counts": dict(sorted(plan.counts.items())),
        "planned_payload_bytes": plan.planned_payload_bytes,
        "plan_sha256": plan.plan_sha256,
        "plan_hash_algorithm": "sha256-canonical-json-lines-v1",
        "output_tree_sha256": output_tree_sha256,
        "output_tree_hash_algorithm": (
            "sha256-canonical-json-lines-path-size-sha256-v1"
        ),
    }
    document["content_sha256"] = _content_sha256(document)
    return document


def _write_bytes_exclusive(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            digest.update(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise TrainOnlyPreparationError(f"cannot write output file: {path}") from exc
    return {
        "path": path.as_posix(),
        "sha256": digest.hexdigest(),
        "size_bytes": len(payload),
    }


def _copy_member(
    archive: zipfile.ZipFile,
    catalog: ZipCatalog,
    member: PlannedMember,
    destination: Path,
) -> dict[str, object]:
    info = catalog.files[member.member_name]
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with archive.open(info, "r") as input_stream:
            with destination.open("xb") as output_stream:
                while True:
                    chunk = input_stream.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > info.file_size:
                        raise TrainOnlyPreparationError(
                            f"member expanded beyond declared size: {member.member_name}"
                        )
                    output_stream.write(chunk)
                    digest.update(chunk)
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
        raise TrainOnlyPreparationError(
            f"cannot copy ZIP member: {member.member_name}"
        ) from exc
    if size != info.file_size:
        raise TrainOnlyPreparationError(
            f"member size mismatch after copy: {member.member_name}"
        )
    return {
        "path": member.target,
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def _actual_relative_files(root: Path) -> set[str]:
    result: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            candidate = current / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise TrainOnlyPreparationError(
                    f"non-directory or symlink entered output tree: {candidate}"
                )
        for name in file_names:
            candidate = current / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise TrainOnlyPreparationError(
                    f"special or symlink file entered output tree: {candidate}"
                )
            result.add(candidate.relative_to(root).as_posix())
    return result


def _verify_projected_data_info(plan: PreparationPlan, root: Path) -> None:
    checks = (
        (
            "V2X-Seq-SPD/vehicle-side/data_info.json",
            "frame_id",
            "vehicle_split",
        ),
        (
            "V2X-Seq-SPD/infrastructure-side/data_info.json",
            "frame_id",
            "infrastructure_split",
        ),
        (
            "V2X-Seq-SPD/cooperative/data_info.json",
            "vehicle_frame",
            "cooperative_split",
        ),
    )
    for relative, key, section in checks:
        raw = (root / relative).read_bytes()
        rows = _data_info_rows(_load_json(raw, relative), relative)
        observed = [row.get(key) for row in rows]
        if len(observed) != len(set(observed)):
            raise TrainOnlyPreparationError(f"duplicate frames in projected {relative}")
        train = set(plan.split_document[section]["train"])
        excluded = set().union(
            *(
                set(plan.split_document[section][name])
                for name in ("val", "test", "test_A")
            )
        )
        if set(observed) != train or set(observed) & excluded:
            raise TrainOnlyPreparationError(
                f"non-train records detected in projected {relative}"
            )


def materialize_preparation_plan(plan: PreparationPlan) -> dict[str, object]:
    """Write a previously audited plan into a newly created output directory."""

    if plan.output.exists() or plan.output.is_symlink():
        raise TrainOnlyPreparationError(f"refusing to overwrite output: {plan.output}")
    try:
        plan.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise TrainOnlyPreparationError(
            f"cannot create isolated output directory: {plan.output}"
        ) from exc
    owned_metadata = plan.output.lstat()
    try:
        tree_records: list[dict[str, object]] = []
        members_by_archive = {
            source.name: [
                member for member in plan.members if member.archive_name == source.name
            ]
            for source in plan.sources
        }
        for source in plan.sources:
            _assert_source_unchanged(source)
            try:
                with zipfile.ZipFile(source.path, "r") as archive:
                    for member in members_by_archive[source.name]:
                        tree_records.append(
                            _copy_member(
                                archive,
                                plan.catalogs[source.name],
                                member,
                                plan.output.joinpath(
                                    *PurePosixPath(member.target).parts
                                ),
                            )
                        )
            except (OSError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
                raise TrainOnlyPreparationError(
                    f"cannot extract planned members from {source.name}"
                ) from exc
            _assert_source_unchanged(source)
        for relative, payload in sorted(plan.data_info_payloads.items()):
            record = _write_bytes_exclusive(
                plan.output.joinpath(*PurePosixPath(relative).parts), payload
            )
            record["path"] = relative
            tree_records.append(record)
        tree_records.sort(key=lambda item: str(item["path"]))
        output_tree_sha256 = _hash_records(tree_records)
        manifest = _manifest(
            plan,
            status="materialized",
            output_tree_sha256=output_tree_sha256,
        )
        manifest_path = plan.output / "train-only-manifest.json"
        _write_bytes_exclusive(manifest_path, _canonical_json_bytes(manifest) + b"\n")

        expected_files = {member.target for member in plan.members}
        expected_files.update(plan.data_info_payloads)
        expected_files.add("train-only-manifest.json")
        actual_files = _actual_relative_files(plan.output)
        if actual_files != expected_files:
            raise TrainOnlyPreparationError(
                "output file inventory differs from the exact train-only plan"
            )
        if any(
            "velodyne" in PurePosixPath(path).parts or path.endswith(".pcd")
            for path in actual_files
        ):
            raise TrainOnlyPreparationError("point-cloud payload detected in output")
        _verify_projected_data_info(plan, plan.output)
        return manifest
    except BaseException:
        try:
            current = plan.output.lstat()
        except FileNotFoundError:
            current = None
        if (
            current is not None
            and stat.S_ISDIR(current.st_mode)
            and not stat.S_ISLNK(current.st_mode)
            and (current.st_dev, current.st_ino)
            == (owned_metadata.st_dev, owned_metadata.st_ino)
        ):
            shutil.rmtree(plan.output)
        raise


def prepare_spd_train_only(
    archive_dir: str | Path,
    split_path: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
    expected_split_sha256: str | None = OFFICIAL_SPLIT_SHA256,
) -> dict[str, object]:
    """Audit sources and optionally materialize the exact train-only projection.

    Passing ``expected_split_sha256=None`` is intended only for synthetic tests.
    The CLI always enforces :data:`OFFICIAL_SPLIT_SHA256`.
    """

    plan = build_preparation_plan(
        archive_dir,
        split_path,
        output,
        expected_split_sha256=expected_split_sha256,
    )
    if dry_run:
        return _manifest(plan, status="audit-only", output_tree_sha256=None)
    return materialize_preparation_plan(plan)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        required=True,
        help="directory containing the four fixed SPD metadata/image ZIP files",
    )
    parser.add_argument(
        "--split",
        type=Path,
        required=True,
        help="pinned official cooperative-split-data-spd.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new isolated output directory; existing paths are rejected",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="audit and print the plan manifest without creating output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = prepare_spd_train_only(
            args.archive_dir,
            args.split,
            args.output,
            dry_run=args.dry_run,
        )
    except TrainOnlyPreparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json_bytes(manifest) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "OFFICIAL_SPLIT_SHA256",
    "TrainOnlyPreparationError",
    "build_preparation_plan",
    "materialize_preparation_plan",
    "prepare_spd_train_only",
)
