#!/usr/bin/env python3
"""Audit an SPD metadata archive against the configured cooperative splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.dataset.event_track_v2x_release_audit import (  # noqa: E402
    audit_spd_release,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_spd_release(args.archive, args.split)
    raw = canonical_json_bytes(report)
    if args.output is None:
        sys.stdout.buffer.write(raw + b"\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(raw)
        print(report["content_sha256"])
    split_reports = report["split_reports"]
    required = tuple(split_reports.get(name, {}) for name in ("train", "val"))
    return 0 if all(item.get("protocol_ready") for item in required) else 2


if __name__ == "__main__":
    raise SystemExit(main())
