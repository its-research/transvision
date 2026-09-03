#!/usr/bin/env python3
"""Cold-cache verify exact ClearML SPD bytes without publishing local mirrors."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.archive_manifest import (  # noqa: E402
    SCIENTIFICALLY_PUBLISHABLE_RELEASE_IDENTITY_STATUSES,
    canonical_json_bytes,
    read_archive_manifest,
)
from tools.event_track_v2x.upload_clearml_dataset import (  # noqa: E402
    DEFAULT_FORMAL_NAME,
    DEFAULT_NAME,
    DEFAULT_PROJECT,
    FORMAL_TAGS,
    PUBLICATION_MODES,
    SMOKE_TAGS,
    STAGED_MANIFEST_NAME,
    ClearMLDatasetError,
    _copy_regular,
    _dataset_identity,
    _require_identity,
    _sha256_file,
    _trimmed,
    _verify_dataset_directory,
)


PUBLISH_TOKEN = "EVENTTRACK_V2X_PUBLISH_VERIFIED_DATASET"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--readback-dir", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name")
    parser.add_argument(
        "--publication-mode",
        choices=PUBLICATION_MODES,
        default="smoke",
    )
    parser.add_argument(
        "--cold-cache-dir",
        type=Path,
        required=True,
        help="Dedicated absent-or-empty CLEARML_CACHE_DIR for this readback.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Request scientific publication. This fails closed for the current "
            "unverified local-mirror manifest; cold verification remains "
            "available without this flag."
        ),
    )
    parser.add_argument(
        "--execute-token",
        help=f"Required with --publish; must equal {PUBLISH_TOKEN!r}.",
    )
    return parser


def _checked_empty_target(path: Path, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ClearMLDatasetError(f"{context} must not be a symlink")
    resolved = expanded.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ClearMLDatasetError(f"{context} must be a directory")
        if any(resolved.iterdir()):
            raise ClearMLDatasetError(f"{context} must be absent or empty")
    return resolved


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _copy_readback(
    source: Path,
    destination: Path,
    expected_names: Sequence[str],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in expected_names:
        _copy_regular(source / name, destination / name)


def main(argv: Sequence[str] | None = None, *, dataset_cls: object | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_id = _trimmed(args.dataset_id, "dataset_id")
    project = _trimmed(args.project, "project")
    version = _trimmed(args.version, "version")
    publication_mode = _trimmed(args.publication_mode, "publication_mode")
    name = _trimmed(
        args.name
        or (DEFAULT_NAME if publication_mode == "smoke" else DEFAULT_FORMAL_NAME),
        "name",
    )
    if type(args.max_workers) is not int or args.max_workers <= 0:
        raise ClearMLDatasetError("max_workers must be a positive integer")
    if args.publish:
        if publication_mode != "formal":
            raise ClearMLDatasetError("only a formal dataset may be published")
        if args.execute_token != PUBLISH_TOKEN:
            raise ClearMLDatasetError("incorrect publish execute token; publication refused")
    elif args.execute_token is not None:
        raise ClearMLDatasetError("--execute-token is only valid with --publish")

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = read_archive_manifest(manifest_path)
    if args.publish:
        release_identity_status = str(manifest["release_identity_status"])
        if (
            release_identity_status
            not in SCIENTIFICALLY_PUBLISHABLE_RELEASE_IDENTITY_STATUSES
        ):
            raise ClearMLDatasetError(
                "scientific publication refused for local-mirror manifest with "
                f"release_identity_status={release_identity_status!r}; obtain an "
                "independently authenticated official-release inventory"
            )
        raise ClearMLDatasetError(
            "scientific publication is unavailable through the local-mirror verifier"
        )
    external_manifest_sha256 = _sha256_file(manifest_path)
    expected_names = tuple(
        sorted(
            [str(record["name"]) for record in manifest["archives"]]
            + [STAGED_MANIFEST_NAME]
        )
    )
    readback_dir = _checked_empty_target(args.readback_dir, "readback directory")
    cold_cache_dir = _checked_empty_target(
        args.cold_cache_dir, "cold cache directory"
    )
    if _overlap(readback_dir, cold_cache_dir):
        raise ClearMLDatasetError("readback directory and cold cache must not overlap")

    previous_cache = os.environ.get("CLEARML_CACHE_DIR")
    os.environ["CLEARML_CACHE_DIR"] = str(cold_cache_dir)
    required_tags = SMOKE_TAGS if publication_mode == "smoke" else FORMAL_TAGS
    try:
        if dataset_cls is None:
            from clearml import Dataset

            dataset_cls = Dataset
        dataset = dataset_cls.get(  # type: ignore[attr-defined]
            dataset_id=dataset_id,
            only_completed=True,
        )
        identity = _dataset_identity(dataset)
        observed_state = str(identity["publication_state"])
        allowed_states = {"smoke"} if publication_mode == "smoke" else {"finalized"}
        if observed_state not in allowed_states:
            raise ClearMLDatasetError(
                f"unexpected ClearML publication state: {observed_state!r}"
            )
        _require_identity(
            identity,
            dataset_id=dataset_id,
            project=project,
            name=name,
            version=version,
            finalized=True,
            publication_state=observed_state,
            tags=required_tags,
        )
        if tuple(sorted(dataset.list_files())) != expected_names:
            raise ClearMLDatasetError("ClearML finalized file inventory mismatch")
        local_copy = dataset.get_local_copy(
            use_soft_links=False,
            raise_on_error=True,
            max_workers=args.max_workers,
        )
        if not local_copy:
            raise ClearMLDatasetError("ClearML did not return a cold-cache path")
        local_copy_path = Path(local_copy)
        if local_copy_path.is_symlink():
            raise ClearMLDatasetError("ClearML cold-cache path must not be a symlink")
        local_copy_path = local_copy_path.resolve(strict=True)
        cold_cache_dir.mkdir(parents=True, exist_ok=True)
        cold_cache_root = cold_cache_dir.resolve(strict=True)
        if cold_cache_root != local_copy_path and cold_cache_root not in local_copy_path.parents:
            raise ClearMLDatasetError(
                "ClearML local copy was not materialized inside the dedicated cold cache"
            )
        internal_mismatches = dataset.verify_dataset_hash(
            local_copy_path=str(local_copy_path),
            skip_hash=False,
            verbose=True,
        )
        if isinstance(internal_mismatches, (str, bytes)) or not isinstance(
            internal_mismatches, Sequence
        ):
            raise ClearMLDatasetError(
                "ClearML verify_dataset_hash did not return a mismatch array"
            )
        if internal_mismatches:
            raise ClearMLDatasetError(
                "ClearML verify_dataset_hash mismatch: "
                f"{tuple(sorted(str(item) for item in internal_mismatches))!r}"
            )
        archives = _verify_dataset_directory(
            local_copy_path,
            manifest,
            expected_names,
            external_manifest_sha256=external_manifest_sha256,
        )
        _copy_readback(local_copy_path, readback_dir, expected_names)
        _verify_dataset_directory(
            readback_dir,
            manifest,
            expected_names,
            external_manifest_sha256=external_manifest_sha256,
        )
    finally:
        if previous_cache is None:
            os.environ.pop("CLEARML_CACHE_DIR", None)
        else:
            os.environ["CLEARML_CACHE_DIR"] = previous_cache
    print(
        canonical_json_bytes(
            {
                "archive_count": len(archives),
                "byte_readback_verified": True,
                "cold_cache_verified": True,
                "dataset": identity,
                "external_manifest_sha256": external_manifest_sha256,
                "manifest_content_sha256": manifest["content_sha256"],
                "publication_state": identity["publication_state"],
                "published_by_this_run": False,
                "release_identity_status": manifest["release_identity_status"],
                "scientific_claims_allowed": False,
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
