#!/usr/bin/env python3
"""Predeclared, fresh-process CPU resource sweeps on full SPD val, car only.

Does not access GT, choose checkpoints, fit models or publish any artifacts.
Each backend/configuration requires the three full-train seeds; each job runs
at least three times in randomized complete blocks. No equal-budget claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from transvision.models.event_track_v2x.resource_sweep import run_sweep, worker
    from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--spec', type=Path)
    group.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--spec-sha256')
    parser.add_argument('--worker-plan-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--job-index', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.spec is not None:
        if args.job_index is not None or args.worker_plan_sha256 is not None or sha_file(args.spec) != args.spec_sha256:
            parser.error('spec digest required; worker arguments cannot be mixed')
        result = run_sweep(json.loads(args.spec.read_bytes()), args.output)
        print(json.dumps(dict(status=result['status'], completed_runs=len(result['runs'])), sort_keys=True))
    else:
        if args.spec_sha256 is not None or sha_file(args.worker_plan) != args.worker_plan_sha256:
            parser.error('worker plan digest required')
        plan = json.loads(args.worker_plan.read_bytes())
        if args.job_index is None or not 0 <= args.job_index < len(plan['jobs']):
            parser.error('valid worker job index required')
        worker(plan, plan['jobs'][args.job_index], args.output)


if __name__ == '__main__':
    main()
