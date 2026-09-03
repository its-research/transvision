#!/usr/bin/env python3
"""Verify an EventTrack-V2X EvidenceBundleV1 against a local or cold cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.evidence import (  # noqa: E402
    verify_evidence_bundle,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = verify_evidence_bundle(args.bundle, artifact_root=args.artifact_root)
    print(bundle.digest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
