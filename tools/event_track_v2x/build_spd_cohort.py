#!/usr/bin/env python3
"""Build a sealed native-10 Hz or frozen-2 Hz V2X-Seq-SPD cohort."""

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

from transvision.dataset.event_track_v2x_cohort import (  # noqa: E402
    SPDCohortError,
    build_spd_cohort_manifest,
)
from transvision.dataset.event_track_v2x_spd import load_spd_metadata  # noqa: E402
from transvision.models.event_track_v2x.wire import canonical_json_bytes  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--rate", type=int, choices=(2, 10), required=True)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _split(path: Path, name: str) -> tuple[list[str], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
        ids = value["cooperative_split"][name]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SPDCohortError("invalid cooperative split document") from exc
    if type(ids) is not list:
        raise SPDCohortError("cooperative split must be an array")
    return ids, hashlib.sha256(raw).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    split_ids, split_sha256 = _split(args.split_path, args.split)
    metadata = load_spd_metadata(args.dataset_root)
    manifest = build_spd_cohort_manifest(
        metadata,
        split_name=args.split,
        split_frame_ids=split_ids,
        split_sha256=split_sha256,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        target_rate_hz=args.rate,
        phase=args.phase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical_json_bytes(manifest))
    print(manifest["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
