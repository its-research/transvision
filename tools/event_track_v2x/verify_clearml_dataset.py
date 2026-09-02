#!/usr/bin/env python3
"""Download a finalized ClearML SPD dataset and verify every archived byte."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.archive_manifest import (  # noqa: E402
    canonical_json_bytes,
    read_archive_manifest,
)
from tools.event_track_v2x.upload_clearml_dataset import (  # noqa: E402
    DEFAULT_NAME,
    DEFAULT_PROJECT,
    STAGED_MANIFEST_NAME,
    ClearMLDatasetError,
    _dataset_identity,
    _require_identity,
    _verify_dataset_directory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--readback-dir", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None, *, dataset_cls: object | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = read_archive_manifest(args.manifest)
    expected_names = tuple(
        sorted(
            [str(record["name"]) for record in manifest["archives"]]
            + [STAGED_MANIFEST_NAME]
        )
    )
    readback_dir = args.readback_dir.expanduser().resolve()
    if readback_dir.exists() and any(readback_dir.iterdir()):
        raise ClearMLDatasetError("readback directory must be absent or empty")
    if dataset_cls is None:
        from clearml import Dataset

        dataset_cls = Dataset
    dataset = dataset_cls.get(dataset_id=args.dataset_id, only_completed=True)  # type: ignore[attr-defined]
    identity = _dataset_identity(dataset)
    _require_identity(
        identity,
        dataset_id=args.dataset_id,
        project=args.project,
        name=args.name,
        version=args.version,
        finalized=True,
    )
    if tuple(sorted(dataset.list_files())) != expected_names:
        raise ClearMLDatasetError("ClearML finalized file inventory mismatch")
    local_copy = dataset.get_mutable_local_copy(
        target_folder=str(readback_dir),
        overwrite=False,
        raise_on_error=True,
        max_workers=args.max_workers,
    )
    if not local_copy:
        raise ClearMLDatasetError("ClearML did not return a local readback path")
    archives = _verify_dataset_directory(
        Path(local_copy).resolve(strict=True), manifest, expected_names
    )
    print(
        canonical_json_bytes(
            {
                "archive_count": len(archives),
                "byte_readback_verified": True,
                "dataset": identity,
                "manifest_content_sha256": manifest["content_sha256"],
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
