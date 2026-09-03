#!/usr/bin/env python3
"""Build the fixed five-fold V2X-Seq-SPD train development manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.development_split import (  # noqa: E402
    DevelopmentSplitError,
    build_development_split_manifest_v1,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.split.read_bytes()
    try:
        document = json.loads(raw)
        sequence_ids = document["batch_split"]["train"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DevelopmentSplitError("invalid official SPD split document") from exc
    if not isinstance(sequence_ids, list):
        raise DevelopmentSplitError("batch_split.train must be an array")
    manifest = build_development_split_manifest_v1(
        sequence_ids,
        split_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(manifest.canonical_bytes)
    print(manifest.content_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
