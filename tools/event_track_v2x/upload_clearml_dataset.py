#!/usr/bin/env python3
"""Dry-run or upload, finalize, and byte-readback the local SPD archive set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.archive_manifest import (  # noqa: E402
    canonical_json_bytes,
    read_archive_manifest,
    verify_archive_files,
)


DEFAULT_PROJECT = "Thesis/EventTrack-V2X/Datasets"
DEFAULT_NAME = "V2X-Seq-SPD local smoke unverified source bytes"
EXECUTE_TOKEN = "EVENTTRACK_V2X_UPLOAD_ARCHIVES"
REQUIRED_TAGS = (
    "V2X-Seq-SPD",
    "EventTrack-V2X",
    "local-smoke",
    "unverified-source-bytes",
    "restricted",
    "scientific-claim-forbidden",
)
STAGED_MANIFEST_NAME = "archive-manifest.json"


class ClearMLDatasetError(RuntimeError):
    """Raised when publication cannot satisfy the fail-closed contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument(
        "--execute-token",
        help=(
            "Publication is disabled unless this exactly equals "
            f"{EXECUTE_TOKEN!r}."
        ),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--chunk-size-mb", type=int, default=512)
    parser.add_argument(
        "--readback-dir",
        type=Path,
        help="Independent byte-readback target; must be absent or empty.",
    )
    parser.add_argument(
        "--defer-byte-readback",
        action="store_true",
        help=(
            "Finalize without local byte readback. This is only an intermediate "
            "state and must be followed by verify_clearml_dataset.py elsewhere."
        ),
    )
    return parser


def _trimmed(value: object, context: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ClearMLDatasetError(f"{context} must be trimmed and non-empty")
    return value


def _positive_integer(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ClearMLDatasetError(f"{context} must be a positive integer")
    return value


def _description(manifest: Mapping[str, object]) -> str:
    return json.dumps(
        {
            "archive_count": manifest["archive_count"],
            "dataset": manifest["dataset"],
            "manifest_content_sha256": manifest["content_sha256"],
            "purpose": "Local EventTrack-V2X data smoke and reproducibility staging",
            "release_identity_status": manifest["release_identity_status"],
            "scientific_claims_allowed": False,
            "total_size_bytes": manifest["total_size_bytes"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def publication_plan(
    manifest: Mapping[str, object],
    *,
    project: str,
    name: str,
    version: str,
) -> dict[str, object]:
    archives = manifest["archives"]
    if not isinstance(archives, Sequence):
        raise ClearMLDatasetError("manifest archives must be an array")
    return {
        "mode": "dry-run",
        "project": _trimmed(project, "project"),
        "name": _trimmed(name, "name"),
        "version": _trimmed(version, "version"),
        "tags": list(REQUIRED_TAGS),
        "release_identity_status": manifest["release_identity_status"],
        "manifest_content_sha256": manifest["content_sha256"],
        "archive_count": manifest["archive_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "dataset_files": [str(record["name"]) for record in archives]
        + [STAGED_MANIFEST_NAME],
        "execute_token_required": EXECUTE_TOKEN,
    }


def _copy_regular(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o400)


def stage_dataset(
    stage: Path,
    archives: Sequence[Path],
    manifest_path: Path,
) -> tuple[str, ...]:
    if stage.exists() and any(stage.iterdir()):
        raise ClearMLDatasetError("staging directory must be empty")
    stage.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for archive in archives:
        destination = stage / archive.name
        _copy_regular(archive, destination)
        names.append(destination.name)
    _copy_regular(manifest_path, stage / STAGED_MANIFEST_NAME)
    names.append(STAGED_MANIFEST_NAME)
    if len(names) != 9 or len(names) != len(set(names)):
        raise ClearMLDatasetError("staged dataset must contain eight archives and one manifest")
    if sorted(path.name for path in stage.iterdir()) != sorted(names):
        raise ClearMLDatasetError("staging directory contains unexpected files")
    return tuple(sorted(names))


def _verify_dataset_directory(
    root: Path,
    manifest: Mapping[str, object],
    expected_names: Sequence[str],
) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ClearMLDatasetError("ClearML readback is not a directory")
    observed = tuple(sorted(path.name for path in root.iterdir()))
    if observed != tuple(sorted(expected_names)):
        raise ClearMLDatasetError(
            f"ClearML dataset file inventory mismatch: {observed!r}"
        )
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ClearMLDatasetError(
                f"ClearML dataset contains a non-regular entry: {path.name}"
            )
    readback_manifest = read_archive_manifest(root / STAGED_MANIFEST_NAME)
    if readback_manifest != dict(manifest):
        raise ClearMLDatasetError("ClearML manifest readback mismatch")
    return verify_archive_files(root, manifest)


def _dataset_identity(dataset: Any) -> dict[str, object]:
    return {
        "id": dataset.id,
        "project": dataset.project,
        "name": dataset.name,
        "version": dataset.version,
        "finalized": bool(dataset.is_final()),
    }


def _require_identity(
    identity: Mapping[str, object],
    *,
    dataset_id: str,
    project: str,
    name: str,
    version: str,
    finalized: bool,
) -> None:
    expected = {
        "id": dataset_id,
        "project": project,
        "name": name,
        "version": version,
        "finalized": finalized,
    }
    if dict(identity) != expected:
        raise ClearMLDatasetError(
            f"ClearML dataset readback mismatch: expected {expected!r}, got {dict(identity)!r}"
        )


def _execute(
    *,
    dataset_cls: Any,
    archives: Sequence[Path],
    manifest_path: Path,
    manifest: Mapping[str, object],
    project: str,
    name: str,
    version: str,
    max_workers: int,
    chunk_size_mb: int,
    readback_dir: Path | None,
) -> dict[str, object]:
    dataset = dataset_cls.create(
        dataset_project=project,
        dataset_name=name,
        dataset_version=version,
        dataset_tags=list(REQUIRED_TAGS),
        description=_description(manifest),
    )
    dataset_id = _trimmed(dataset.id, "created ClearML dataset ID")
    stage = Path(tempfile.mkdtemp(prefix="eventtrack-v2x-clearml-archives-"))
    stage_removed = False
    try:
        staged_names = stage_dataset(stage, archives, manifest_path)
        _verify_dataset_directory(stage, manifest, staged_names)
        dataset.add_files(
            stage,
            local_base_folder=str(stage),
            max_workers=max_workers,
        )
        dataset.upload(
            show_progress=True,
            verbose=True,
            chunk_size=chunk_size_mb,
            max_workers=max_workers,
            preview=False,
        )
        _verify_dataset_directory(stage, manifest, staged_names)
        before_finalize = dataset_cls.get(dataset_id=dataset_id)
        _require_identity(
            _dataset_identity(before_finalize),
            dataset_id=dataset_id,
            project=project,
            name=name,
            version=version,
            finalized=False,
        )
        if tuple(sorted(before_finalize.list_files())) != staged_names:
            raise ClearMLDatasetError("ClearML pre-finalize file inventory mismatch")
        finalized = dataset.finalize(verbose=True, raise_on_error=True)
        if finalized is not True:
            raise ClearMLDatasetError("ClearML finalize did not report success")
        readback = dataset_cls.get(dataset_id=dataset_id, only_completed=True)
        identity = _dataset_identity(readback)
        _require_identity(
            identity,
            dataset_id=dataset_id,
            project=project,
            name=name,
            version=version,
            finalized=True,
        )
        if tuple(sorted(readback.list_files())) != staged_names:
            raise ClearMLDatasetError("ClearML finalized file inventory mismatch")
        # Upload has severed ClearML's local-path references.  Remove the
        # isolated source copy before downloading the independent readback so
        # a full dataset does not require source + stage + cache + readback at
        # the same time.
        shutil.rmtree(stage)
        stage_removed = True
        result: dict[str, object] = {
            "mode": (
                "executed" if readback_dir is not None
                else "executed-pending-byte-readback"
            ),
            "dataset": identity,
            "manifest_content_sha256": manifest["content_sha256"],
            "staged_files": list(staged_names),
            "byte_readback_verified": False,
        }
        if readback_dir is None:
            return result
        if readback_dir.exists() and any(readback_dir.iterdir()):
            raise ClearMLDatasetError("readback directory must be absent or empty")
        local_copy = readback.get_mutable_local_copy(
            target_folder=str(readback_dir),
            overwrite=False,
            raise_on_error=True,
            max_workers=max_workers,
        )
        if not local_copy:
            raise ClearMLDatasetError("ClearML did not return a local readback path")
        readback_files = _verify_dataset_directory(
            Path(local_copy).resolve(strict=True), manifest, staged_names
        )
        result["byte_readback_verified"] = True
        result["readback_archive_count"] = len(readback_files)
        return result
    finally:
        if not stage_removed:
            shutil.rmtree(stage)


def main(
    argv: Sequence[str] | None = None,
    *,
    dataset_cls: Any | None = None,
) -> int:
    args = _parser().parse_args(argv)
    project = _trimmed(args.project, "project")
    name = _trimmed(args.name, "name")
    version = _trimmed(args.version, "version")
    max_workers = _positive_integer(args.max_workers, "max_workers")
    chunk_size_mb = _positive_integer(args.chunk_size_mb, "chunk_size_mb")
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = read_archive_manifest(manifest_path)
    archives = verify_archive_files(args.archive_dir, manifest)
    plan = publication_plan(
        manifest,
        project=project,
        name=name,
        version=version,
    )
    if args.execute_token is None:
        print(canonical_json_bytes(plan).decode("utf-8"))
        return 0
    if args.execute_token != EXECUTE_TOKEN:
        raise ClearMLDatasetError("incorrect execute token; publication refused")
    if args.readback_dir is not None and args.defer_byte_readback:
        raise ClearMLDatasetError(
            "choose --readback-dir or --defer-byte-readback, not both"
        )
    if args.readback_dir is None and not args.defer_byte_readback:
        raise ClearMLDatasetError(
            "execution requires --readback-dir or explicit --defer-byte-readback"
        )
    readback_dir = (
        args.readback_dir.expanduser().resolve()
        if args.readback_dir is not None
        else None
    )
    if dataset_cls is None:
        from clearml import Dataset

        dataset_cls = Dataset
    result = _execute(
        dataset_cls=dataset_cls,
        archives=archives,
        manifest_path=manifest_path,
        manifest=manifest,
        project=project,
        name=name,
        version=version,
        max_workers=max_workers,
        chunk_size_mb=chunk_size_mb,
        readback_dir=readback_dir,
    )
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
