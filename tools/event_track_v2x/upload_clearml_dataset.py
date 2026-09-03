#!/usr/bin/env python3
"""Stage and byte-verify a scientifically unverified local SPD archive set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
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
DEFAULT_FORMAL_NAME = "V2X-Seq-SPD formal integrity candidate"
EXECUTE_TOKEN = "EVENTTRACK_V2X_UPLOAD_ARCHIVES"
PUBLICATION_MODES = ("smoke", "formal")
COMMON_TAGS = (
    "V2X-Seq-SPD",
    "EventTrack-V2X",
    "unverified-source-bytes",
    "restricted",
    "scientific-claim-forbidden",
)
SMOKE_TAGS = (*COMMON_TAGS, "local-smoke")
FORMAL_TAGS = (*COMMON_TAGS, "formal-candidate", "cold-readback-required")
# Kept for callers that imported the original smoke-only contract.
REQUIRED_TAGS = SMOKE_TAGS
STAGED_MANIFEST_NAME = "archive-manifest.json"


class ClearMLDatasetError(RuntimeError):
    """Raised when ClearML staging or verification violates its contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name")
    parser.add_argument(
        "--publication-mode",
        choices=PUBLICATION_MODES,
        default="smoke",
        help=(
            "smoke keeps the local-smoke safety label; formal creates a "
            "finalized integrity candidate for independent cold readback. "
            "Neither mode makes these unverified local-mirror bytes "
            "scientifically publishable."
        ),
    )
    parser.add_argument(
        "--execute-token",
        help=(
            "Remote upload execution is disabled unless this exactly equals "
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


def _description(
    manifest: Mapping[str, object],
    *,
    publication_mode: str,
) -> str:
    purpose = (
        "Local EventTrack-V2X data smoke and reproducibility staging"
        if publication_mode == "smoke"
        else "EventTrack-V2X formal integrity candidate pending cold readback"
    )
    return json.dumps(
        {
            "archive_count": manifest["archive_count"],
            "dataset": manifest["dataset"],
            "manifest_content_sha256": manifest["content_sha256"],
            "purpose": purpose,
            "release_identity_status": manifest["release_identity_status"],
            "scientific_claims_allowed": False,
            "total_size_bytes": manifest["total_size_bytes"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _tags(publication_mode: str) -> tuple[str, ...]:
    if publication_mode == "smoke":
        return SMOKE_TAGS
    if publication_mode == "formal":
        return FORMAL_TAGS
    raise ClearMLDatasetError(f"unsupported publication mode: {publication_mode!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publication_plan(
    manifest: Mapping[str, object],
    *,
    project: str,
    name: str,
    version: str,
    publication_mode: str = "smoke",
) -> dict[str, object]:
    archives = manifest["archives"]
    if not isinstance(archives, Sequence):
        raise ClearMLDatasetError("manifest archives must be an array")
    return {
        "mode": "dry-run",
        "project": _trimmed(project, "project"),
        "name": _trimmed(name, "name"),
        "version": _trimmed(version, "version"),
        "publication_mode": publication_mode,
        "target_state": "smoke" if publication_mode == "smoke" else "finalized",
        "scientific_claims_allowed": False,
        "tags": list(_tags(publication_mode)),
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
    *,
    external_manifest_sha256: str,
) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
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
    if _sha256_file(root / STAGED_MANIFEST_NAME) != external_manifest_sha256:
        raise ClearMLDatasetError("ClearML external manifest SHA-256 mismatch")
    return verify_archive_files(root, manifest)


def _dataset_tags(dataset: Any) -> tuple[str, ...]:
    tags = dataset.tags
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise ClearMLDatasetError("ClearML dataset tags are not an array")
    normalized = tuple(sorted(_trimmed(tag, "ClearML dataset tag") for tag in tags))
    if len(normalized) != len(set(normalized)):
        raise ClearMLDatasetError("ClearML dataset tags contain duplicates")
    return normalized


def _clearml_status(dataset: Any) -> str:
    task = getattr(dataset, "_task", None)
    get_status = getattr(task, "get_status", None)
    if not callable(get_status):
        raise ClearMLDatasetError("ClearML dataset task status is unavailable")
    raw = get_status()
    status = getattr(raw, "value", raw)
    return _trimmed(str(status), "ClearML dataset task status")


def _publication_state(dataset: Any, *, tags: Sequence[str], status: str) -> str:
    if status == "published":
        return "published"
    if "local-smoke" in tags:
        return "smoke"
    if bool(dataset.is_final()):
        return "finalized"
    return "draft"


def _dataset_identity(dataset: Any) -> dict[str, object]:
    tags = _dataset_tags(dataset)
    status = _clearml_status(dataset)
    return {
        "id": dataset.id,
        "project": dataset.project,
        "name": dataset.name,
        "version": dataset.version,
        "finalized": bool(dataset.is_final()),
        "clearml_status": status,
        "publication_state": _publication_state(dataset, tags=tags, status=status),
        "tags": list(tags),
    }


def _require_identity(
    identity: Mapping[str, object],
    *,
    dataset_id: str,
    project: str,
    name: str,
    version: str,
    finalized: bool,
    publication_state: str,
    tags: Sequence[str],
) -> None:
    expected = {
        "id": dataset_id,
        "project": project,
        "name": name,
        "version": version,
        "finalized": finalized,
        "publication_state": publication_state,
        "tags": list(sorted(tags)),
    }
    observed = {
        key: identity.get(key)
        for key in expected
    }
    if observed != expected:
        raise ClearMLDatasetError(
            f"ClearML dataset readback mismatch: expected {expected!r}, got {observed!r}"
        )


def _wait_for_all_upload_results(
    dataset: Any,
    *,
    chunk_size_mb: int,
    max_workers: int,
) -> int:
    """Run ClearML upload and surface failures from every artifact future.

    ClearML 2.1.3 waits for its ThreadPoolExecutor on context exit, but it does
    not call ``result()`` on the final batch of futures.  The audited wrapper
    records every synchronous ``upload_artifact`` result so a trailing False or
    exception cannot be hidden by that implementation detail.
    """

    task = getattr(dataset, "_task", None)
    original = getattr(task, "upload_artifact", None)
    flush = getattr(task, "flush", None)
    if not callable(original) or not callable(flush):
        raise ClearMLDatasetError("ClearML upload barrier API is unavailable")
    outcomes: list[bool] = []
    failures: list[BaseException] = []
    lock = Lock()

    def audited_upload_artifact(*args: Any, **kwargs: Any) -> bool:
        if kwargs.get("wait_on_upload") is not True:
            error = ClearMLDatasetError(
                "ClearML artifact upload was not configured to wait"
            )
            with lock:
                failures.append(error)
            raise error
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            with lock:
                failures.append(error)
            raise
        with lock:
            outcomes.append(result is True)
        return result

    had_instance_override = "upload_artifact" in vars(task)
    previous_override = vars(task).get("upload_artifact")
    setattr(task, "upload_artifact", audited_upload_artifact)
    upload_error: BaseException | None = None
    try:
        dataset.upload(
            show_progress=True,
            verbose=True,
            chunk_size=chunk_size_mb,
            max_workers=max_workers,
            preview=False,
        )
    except BaseException as error:
        upload_error = error
    finally:
        if had_instance_override:
            setattr(task, "upload_artifact", previous_override)
        else:
            delattr(task, "upload_artifact")
    if upload_error is not None:
        raise upload_error
    if failures:
        raise failures[0]
    if not outcomes:
        raise ClearMLDatasetError("ClearML upload produced no audited artifact futures")
    if not all(outcomes):
        raise ClearMLDatasetError("ClearML artifact upload future returned failure")
    flushed = flush(wait_for_uploads=True)
    if flushed is not True:
        raise ClearMLDatasetError("ClearML upload flush did not report success")
    return len(outcomes)


def _execute(
    *,
    dataset_cls: Any,
    archives: Sequence[Path],
    manifest_path: Path,
    manifest: Mapping[str, object],
    project: str,
    name: str,
    version: str,
    publication_mode: str,
    max_workers: int,
    chunk_size_mb: int,
    readback_dir: Path | None,
) -> dict[str, object]:
    required_tags = _tags(publication_mode)
    external_manifest_sha256 = _sha256_file(manifest_path)
    dataset = dataset_cls.create(
        dataset_project=project,
        dataset_name=name,
        dataset_version=version,
        dataset_tags=list(required_tags),
        description=_description(manifest, publication_mode=publication_mode),
    )
    dataset_id = _trimmed(dataset.id, "created ClearML dataset ID")
    stage = Path(tempfile.mkdtemp(prefix="eventtrack-v2x-clearml-archives-"))
    stage_removed = False
    try:
        staged_names = stage_dataset(stage, archives, manifest_path)
        _verify_dataset_directory(
            stage,
            manifest,
            staged_names,
            external_manifest_sha256=external_manifest_sha256,
        )
        dataset.add_files(
            stage,
            local_base_folder=str(stage),
            max_workers=max_workers,
        )
        upload_artifact_count = _wait_for_all_upload_results(
            dataset,
            chunk_size_mb=chunk_size_mb,
            max_workers=max_workers,
        )
        _verify_dataset_directory(
            stage,
            manifest,
            staged_names,
            external_manifest_sha256=external_manifest_sha256,
        )
        before_finalize = dataset_cls.get(dataset_id=dataset_id)
        _require_identity(
            _dataset_identity(before_finalize),
            dataset_id=dataset_id,
            project=project,
            name=name,
            version=version,
            finalized=False,
            publication_state=(
                "smoke" if publication_mode == "smoke" else "draft"
            ),
            tags=required_tags,
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
            publication_state=(
                "smoke" if publication_mode == "smoke" else "finalized"
            ),
            tags=required_tags,
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
            "external_manifest_sha256": external_manifest_sha256,
            "publication_mode": publication_mode,
            "publication_state": identity["publication_state"],
            "staged_files": list(staged_names),
            "upload_artifact_count": upload_artifact_count,
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
        local_copy_path = Path(local_copy)
        if local_copy_path.is_symlink():
            raise ClearMLDatasetError("ClearML readback path must not be a symlink")
        readback_files = _verify_dataset_directory(
            local_copy_path.resolve(strict=True),
            manifest,
            staged_names,
            external_manifest_sha256=external_manifest_sha256,
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
    publication_mode = _trimmed(args.publication_mode, "publication_mode")
    default_name = (
        DEFAULT_NAME if publication_mode == "smoke" else DEFAULT_FORMAL_NAME
    )
    name = _trimmed(args.name or default_name, "name")
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
        publication_mode=publication_mode,
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
    if publication_mode == "formal" and not args.defer_byte_readback:
        raise ClearMLDatasetError(
            "formal integrity staging requires --defer-byte-readback and independent "
            "verify_clearml_dataset.py execution"
        )
    readback_dir = None
    if args.readback_dir is not None:
        expanded_readback = args.readback_dir.expanduser()
        if expanded_readback.is_symlink():
            raise ClearMLDatasetError("readback directory must not be a symlink")
        readback_dir = expanded_readback.resolve()
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
        publication_mode=publication_mode,
        max_workers=max_workers,
        chunk_size_mb=chunk_size_mb,
        readback_dir=readback_dir,
    )
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
