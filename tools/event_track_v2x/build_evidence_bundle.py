#!/usr/bin/env python3
"""Build a canonical EventTrack-V2X EvidenceBundleV1 from local artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.evidence import (  # noqa: E402
    EVIDENCE_ROLES,
    build_evidence_bundle,
    write_evidence_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for role in EVIDENCE_ROLES:
        parser.add_argument(f"--{role.replace('_', '-')}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    role_paths = {role: getattr(args, role) for role in EVIDENCE_ROLES}
    bundle = build_evidence_bundle(
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        role_paths=role_paths,
    )
    print(write_evidence_bundle(bundle, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
