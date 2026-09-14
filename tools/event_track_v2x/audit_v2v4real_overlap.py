#!/usr/bin/env python3
"""Audit native train/test sequence, session and exact-PCD overlap without GT.

Exit 0: no overlap under the implemented checks (NOT training qualification).
Exit 3: overlap found; the new audit file is preserved for review.
Exit 2: invalid inputs/configuration or output already exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transvision.models.event_track_v2x.v2v4real_inputs import V2V4RealInputError
from transvision.models.event_track_v2x.v2v4real_overlap import audit_native_overlap


def _read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise V2V4RealInputError("session map must be a bounded regular JSON file")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise V2V4RealInputError("duplicate session-map JSON key")
            result[key] = value
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--test-inputs", type=Path, required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--test-manifest-sha256", required=True)
    parser.add_argument("--session-map", type=Path, required=True)
    parser.add_argument("--session-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise V2V4RealInputError("output already exists; choose a new audit file")
        if any(root.resolve() == args.output.resolve() or root.resolve() in args.output.resolve().parents
               for root in (args.train_inputs, args.test_inputs)):
            raise V2V4RealInputError("audit output must be outside the immutable input trees")
        if (args.session_evidence.is_symlink() or not args.session_evidence.is_file()
                or args.session_evidence.stat().st_size > 16 * 1024 * 1024):
            raise V2V4RealInputError("session evidence must be a bounded regular file")
        report = audit_native_overlap(
            args.train_inputs, args.test_inputs, train_manifest_sha256=args.train_manifest_sha256,
            test_manifest_sha256=args.test_manifest_sha256, session_map=_read_json(args.session_map),
            session_evidence=args.session_evidence.read_bytes(),
        )
        payload = (json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(payload)
        print(json.dumps({
            "output": str(args.output.resolve()), "report_sha256": report["report_sha256"],
            "checked_overlap_absent": report["checked_overlap_absent"],
            "affected_train_sequences": len(report["affected_train_component_closure"]),
            "training_eligibility_verified": False, "paper_eligible": False,
        }, sort_keys=True))
        return 0 if report["checked_overlap_absent"] else 3
    except (V2V4RealInputError, OSError, ValueError) as exc:
        print(f"V2V4Real overlap audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
