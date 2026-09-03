#!/usr/bin/env python3
"""Build a canonical byte manifest for eight local V2X-Seq-SPD archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.archive_manifest import (  # noqa: E402
    DEFAULT_RELEASE_IDENTITY_STATUS,
    build_archive_manifest,
    write_archive_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--release-identity-status",
        default=DEFAULT_RELEASE_IDENTITY_STATUS,
        help=(
            "Local-mirror evidence status. This tool accepts only "
            f"{DEFAULT_RELEASE_IDENTITY_STATUS!r}; it cannot assert official "
            "release identity."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_archive_manifest(
        args.archive_dir,
        release_identity_status=args.release_identity_status,
    )
    output = write_archive_manifest(args.output, document)
    print(
        json.dumps(
            {
                "archive_count": document["archive_count"],
                "content_sha256": document["content_sha256"],
                "manifest": str(output),
                "release_identity_status": document["release_identity_status"],
                "total_size_bytes": document["total_size_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
