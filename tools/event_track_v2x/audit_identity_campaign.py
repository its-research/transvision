#!/usr/bin/env python3
"""Aggregate three independently verified DDP identity runs, without publishing.

An offline campaign receipt proves artifact consistency, not live ClearML
status, machine concurrency, tracking quality, or pipeline-isolated selection.
The original single-seed receipts remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.audit_forest_identity_ddp import verify
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json

SEEDS = (1337, 2027, 3407)


def verify_campaign(runs, data_manifest, data_manifest_sha256, *, require_full_train=True):
    """runs is a sequence of (seed, directory, independently expected receipt SHA)."""
    runs = list(runs)
    seeds = [row[0] for row in runs]
    if (len(runs) != len(SEEDS) or any(type(seed) is not int for seed in seeds)
            or set(seeds) != set(SEEDS)):
        raise ValueError('exactly one run for each required campaign seed required')
    roots = [Path(row[1]).resolve() for row in runs]
    if len(set(roots)) != len(SEEDS):
        raise ValueError('three independent single-seed run directories required')

    reference = None
    reports, bindings = [], []
    files = [(Path(data_manifest), data_manifest_sha256)]
    for seed, root, receipt_sha256 in sorted(runs, key=lambda row: row[0]):
        root = Path(root)
        report = verify(root, receipt_sha256, data_manifest, data_manifest_sha256,
                        seed=seed, require_full_train=require_full_train)
        receipt_path = contained_file(root, 'receipt.json')
        plan_path = contained_file(root, 'plan.json')
        receipt = json.loads(receipt_path.read_bytes())
        plan = json.loads(plan_path.read_bytes())
        if (plan['seeds'] != [seed] or [row['seed'] for row in receipt['seeds']] != [seed]
                or plan['required_campaign_seeds'] != list(SEEDS)
                or receipt['required_campaign_seeds'] != list(SEEDS)):
            raise ValueError('single-seed run and common required campaign contract required')
        # Hardware/runtime identifiers may differ across valid four-card workers.
        # All fitting/data/source/selection fields must match, including new ones.
        common = {key: value for key, value in plan.items() if key not in ('seeds', 'rank_runtime')}
        if reference is None:
            reference = common
        elif common != reference:
            raise ValueError('campaign data, source, fit or selection plans differ')
        result = receipt['seeds'][0]
        checkpoint = contained_file(root, result['checkpoint_manifest'])
        metadata = json.loads(checkpoint.read_bytes())
        weights = contained_file(checkpoint.parent, metadata['weights']['path'])
        files.extend(((receipt_path, receipt_sha256), (plan_path, receipt['plan_sha256']),
                      (checkpoint, report['checkpoint_sha256']),
                      (checkpoint.parent/'epochs.jsonl', result['epochs_sha256']),
                      (weights, report['weights_sha256'])))
        reports.append(report)
        bindings.append(dict(seed=seed, run_directory=str(root.resolve()),
            plan_sha256=receipt['plan_sha256'], epochs_sha256=result['epochs_sha256'],
            elapsed_seconds=receipt['elapsed_seconds'], rank_runtime=receipt['rank_runtime']))
    if len({report['model_sha256'] for report in reports}) != len(SEEDS):
        raise ValueError('independent seeds produced identical saved model digests')
    if any(sha_file(path) != digest for path, digest in files):
        raise ValueError('campaign artifacts changed during verification')
    return dict(kind='forest_identity_ddp_campaign_offline_audit_v1', verified=True,
        complete_three_seed_campaign=True, seeds=list(SEEDS), full_official_train=require_full_train,
        local_fixture_only=not require_full_train, data_manifest_sha256=data_manifest_sha256,
        common_plan=reference, runs=reports, run_bindings=bindings,
        remote_live_status_checked=False, multi_machine_concurrency_verified=False,
        strict_pipeline_isolated_selection=False, tracking_validation_performed=False, paper_eligible=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs=3, action='append', required=True,
                        metavar=('SEED', 'DIRECTORY', 'RECEIPT_SHA256'))
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--data-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [(int(seed), path, digest) for seed, path, digest in args.run]
    result = verify_campaign(runs, args.data_manifest, args.data_manifest_sha256)
    _new_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('common_plan', 'run_bindings')}, sort_keys=True))


if __name__ == '__main__':
    main()
