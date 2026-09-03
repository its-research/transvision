"""Canonical byte manifest for the local V2X-Seq-SPD archive set."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEMA_VERSION = 1
DOCUMENT_TYPE = "event_track_v2x_spd_archive_manifest"
DATASET_NAME = "V2X-Seq-SPD"
DEFAULT_RELEASE_IDENTITY_STATUS = "unverified-source-bytes"
EXPECTED_ARCHIVE_COUNT = 8
EXPECTED_ARCHIVE_NAMES = (
    "V2X-Seq-SPD-infrastructure-side-image-00_11616163645886464.zip",
    "V2X-Seq-SPD-infrastructure-side-image-01_11616163645886464.zip",
    "V2X-Seq-SPD-infrastructure-side-velodyne.z01",
    "V2X-Seq-SPD-infrastructure-side-velodyne.zip",
    "V2X-Seq-SPD-vehicle-side-image_11616163645886464.zip",
    "V2X-Seq-SPD-vehicle-side-velodyne.z01",
    "V2X-Seq-SPD-vehicle-side-velodyne.zip",
    "V2X-Seq-SPD.zip",
)
ALLOWED_RELEASE_IDENTITY_STATUSES = frozenset({DEFAULT_RELEASE_IDENTITY_STATUS})
# This local-mirror schema intentionally has no scientifically publishable
# identity status.  A trusted official-release inventory needs a separate,
# independently authenticated ingestion path rather than a caller-supplied
# status string.
SCIENTIFICALLY_PUBLISHABLE_RELEASE_IDENTITY_STATUSES: frozenset[str] = frozenset()
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARCHIVE_SUFFIX = re.compile(r".*(?:\.zip|\.z[0-9]{2})", re.IGNORECASE)


class ArchiveManifestError(ValueError):
    """Raised when an archive set or its byte manifest is invalid."""


def _validate_json_domain(value: object, context: str = "manifest") -> None:
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise ArchiveManifestError(f"{context} keys must be strings")
        for key, item in value.items():
            _validate_json_domain(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_domain(item, f"{context}[{index}]")
        return
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    raise ArchiveManifestError(f"{context} is outside the canonical JSON domain")


def canonical_json_bytes(value: object) -> bytes:
    _validate_json_domain(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_name(name: object, context: str) -> str:
    if type(name) is not str or not name or name != Path(name).name:
        raise ArchiveManifestError(f"{context} must be a basename")
    if name.startswith(".") or any(ord(character) < 32 for character in name):
        raise ArchiveManifestError(f"{context} is not canonical")
    if not name.startswith("V2X-Seq-SPD") or _ARCHIVE_SUFFIX.fullmatch(name) is None:
        raise ArchiveManifestError(f"{context} is not an SPD archive name")
    return name


def discover_archives(
    archive_dir: str | Path,
    *,
    expected_count: int = EXPECTED_ARCHIVE_COUNT,
) -> tuple[Path, ...]:
    if type(expected_count) is not int or expected_count <= 0:
        raise ArchiveManifestError("expected_count must be a positive integer")
    root = Path(archive_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ArchiveManifestError(f"archive directory is not a directory: {root}")
    archives: list[Path] = []
    for entry in root.iterdir():
        if not entry.name.startswith("V2X-Seq-SPD"):
            continue
        if _ARCHIVE_SUFFIX.fullmatch(entry.name) is None:
            continue
        _archive_name(entry.name, "archive name")
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArchiveManifestError(
                f"archive must be a regular non-symlink file: {entry.name}"
            )
        archives.append(entry)
    archives.sort(key=lambda item: item.name)
    if len(archives) != expected_count:
        raise ArchiveManifestError(
            f"expected {expected_count} SPD archives, found {len(archives)}"
        )
    if expected_count == EXPECTED_ARCHIVE_COUNT and tuple(
        path.name for path in archives
    ) != EXPECTED_ARCHIVE_NAMES:
        raise ArchiveManifestError("SPD local-mirror archive filenames mismatch")
    return tuple(archives)


def build_archive_manifest(
    archive_dir: str | Path,
    *,
    release_identity_status: str = DEFAULT_RELEASE_IDENTITY_STATUS,
    expected_count: int = EXPECTED_ARCHIVE_COUNT,
) -> dict[str, object]:
    if (
        type(release_identity_status) is not str
        or not release_identity_status
        or release_identity_status != release_identity_status.strip()
        or release_identity_status not in ALLOWED_RELEASE_IDENTITY_STATUSES
    ):
        raise ArchiveManifestError(
            "release_identity_status is not allowed for this local-mirror tool"
        )
    records = [
        {
            "name": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": _sha256_file(archive),
        }
        for archive in discover_archives(archive_dir, expected_count=expected_count)
    ]
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "document_type": DOCUMENT_TYPE,
        "dataset": DATASET_NAME,
        "release_identity_status": release_identity_status,
        "archive_count": len(records),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "archives": records,
    }
    document["content_sha256"] = _content_sha256(document)
    return verify_manifest_document(document, expected_count=expected_count)


def verify_manifest_document(
    document: Mapping[str, object],
    *,
    expected_count: int = EXPECTED_ARCHIVE_COUNT,
) -> dict[str, object]:
    if not isinstance(document, Mapping):
        raise ArchiveManifestError("archive manifest must be an object")
    expected_fields = {
        "schema_version",
        "document_type",
        "dataset",
        "release_identity_status",
        "archive_count",
        "total_size_bytes",
        "archives",
        "content_sha256",
    }
    if set(document) != expected_fields:
        raise ArchiveManifestError("archive manifest fields mismatch")
    plain = dict(document)
    if plain["schema_version"] != SCHEMA_VERSION:
        raise ArchiveManifestError("unsupported archive manifest schema_version")
    if plain["document_type"] != DOCUMENT_TYPE or plain["dataset"] != DATASET_NAME:
        raise ArchiveManifestError("archive manifest identity mismatch")
    status = plain["release_identity_status"]
    if (
        type(status) is not str
        or status not in ALLOWED_RELEASE_IDENTITY_STATUSES
    ):
        raise ArchiveManifestError("invalid release_identity_status")
    archives_raw = plain["archives"]
    if isinstance(archives_raw, (str, bytes)) or not isinstance(
        archives_raw, Sequence
    ):
        raise ArchiveManifestError("archives must be an array")
    records: list[dict[str, object]] = []
    for index, raw in enumerate(archives_raw):
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "size_bytes",
            "sha256",
        }:
            raise ArchiveManifestError(f"archives[{index}] fields mismatch")
        name = _archive_name(raw["name"], f"archives[{index}].name")
        size = raw["size_bytes"]
        digest = raw["sha256"]
        if type(size) is not int or size < 0:
            raise ArchiveManifestError(f"archives[{index}].size_bytes is invalid")
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ArchiveManifestError(f"archives[{index}].sha256 is invalid")
        records.append({"name": name, "size_bytes": size, "sha256": digest})
    names = [str(record["name"]) for record in records]
    if names != sorted(set(names)):
        raise ArchiveManifestError("archive records must be unique and sorted")
    if len(records) != expected_count or plain["archive_count"] != len(records):
        raise ArchiveManifestError("archive count mismatch")
    expected_total = sum(int(record["size_bytes"]) for record in records)
    if plain["total_size_bytes"] != expected_total:
        raise ArchiveManifestError("archive total_size_bytes mismatch")
    digest = plain["content_sha256"]
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        raise ArchiveManifestError("content_sha256 is invalid")
    _validate_json_domain(plain)
    if _content_sha256(plain) != digest:
        raise ArchiveManifestError("archive manifest content hash mismatch")
    plain["archives"] = records
    return plain


def write_archive_manifest(path: str | Path, document: Mapping[str, object]) -> Path:
    verified = verify_manifest_document(document)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(verified) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_archive_manifest(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve(strict=True)
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ArchiveManifestError("archive manifest must be a regular file")
    raw = source.read_bytes()
    canonical = raw[:-1] if raw.endswith(b"\n") else raw
    try:
        decoded = json.loads(canonical)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveManifestError("invalid archive manifest JSON") from error
    verified = verify_manifest_document(decoded)
    if canonical_json_bytes(verified) != canonical:
        raise ArchiveManifestError("archive manifest JSON is not canonical")
    return verified


def verify_archive_files(
    archive_dir: str | Path,
    document: Mapping[str, object],
) -> tuple[Path, ...]:
    verified = verify_manifest_document(document)
    root = Path(archive_dir).expanduser().resolve(strict=True)
    discovered = discover_archives(root, expected_count=int(verified["archive_count"]))
    by_name = {path.name: path for path in discovered}
    verified_paths: list[Path] = []
    for record in verified["archives"]:  # type: ignore[union-attr]
        path = by_name.get(str(record["name"]))
        if path is None:
            raise ArchiveManifestError(f"archive is missing: {record['name']}")
        if path.stat().st_size != record["size_bytes"]:
            raise ArchiveManifestError(f"archive size mismatch: {path.name}")
        if _sha256_file(path) != record["sha256"]:
            raise ArchiveManifestError(f"archive SHA-256 mismatch: {path.name}")
        verified_paths.append(path)
    return tuple(verified_paths)
