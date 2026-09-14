#!/usr/bin/env python3
"""Export completed TRAIN teacher traces or fit three fixed-seed priority heads.

Production fitting requires a verified full-train TRACE cohort. A sequence
holdout within train is excluded from optimizer updates and never selects a
checkpoint. This is not a full-train refit or an isolated upstream pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.allocation_training import export_training, fit_priority


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--replay', type=Path, required=True)
    export.add_argument('--receipt-sha256', required=True)
    export.add_argument('--output', type=Path, required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--data', type=Path, required=True)
    fit.add_argument('--manifest-sha256', required=True)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    result = (export_training(args.replay, args.receipt_sha256, args.output) if args.command == 'export'
        else fit_priority(args.data, args.manifest_sha256, args.output, epochs=args.epochs))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
