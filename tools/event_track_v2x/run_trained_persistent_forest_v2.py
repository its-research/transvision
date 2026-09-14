#!/usr/bin/env python3
"""Full SPD-val development replay with a train-fitted row identity checkpoint.

This validates integration, not the still-incomplete paper method. No GT/test,
parameter sweep, checkpoint selection, or publication is performed here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows, _new
from tools.event_track_v2x.run_tracking_v2 import schedule_rows, SPLIT_SHA
from tools.event_track_v2x.train_forest_identity import training_sources
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig


def run(args):
    config = PersistentForestConfig()
    scorer, checkpoint = load_identity_checkpoint(args.checkpoint, args.checkpoint_sha256, config=config.state, device=args.device)
    if checkpoint.get('full_official_train') is not True:
        raise ValueError('fixture checkpoint cannot be used as full-train validation')
    rows = schedule_rows(args.schedule, args.schedule_sha256)
    cache = VerifiedForestCache(args.cache, args.cache_sha256)
    manifest = json.loads(cache.manifest_json)
    if frozen_cache_identity(cache) != checkpoint['frozen_cache_identity']:
        raise ValueError('validation frozen detector/appearance/calibration differs from training')
    if (manifest['split'] != 'val' or manifest['frame_count'] != 7189
            or set(manifest['sequences']) != {r['sequence_id'] for r in rows}):
        raise ValueError('full sealed SPD validation cache required')
    sources = training_sources()
    for path in (Path(__file__), ROOT/'tools/event_track_v2x/run_persistent_forest_v2.py',
                 ROOT/'tools/event_track_v2x/run_tracking_v2.py',
                 ROOT/'transvision/models/event_track_v2x/persistent_cache_stream.py',
                 ROOT/'transvision/models/event_track_v2x/persistent_forest.py',
                 ROOT/'transvision/models/event_track_v2x/persistent_component_store.py',
                 ROOT/'transvision/models/event_track_v2x/persistent_component_tracking.py',
                 ROOT/'transvision/models/event_track_v2x/persistent_beam_tracking.py',
                 ROOT/'transvision/models/event_track_v2x/persistent_joint_beam.py',
                 ROOT/'transvision/models/event_track_v2x/allocation_policy.py',
                 ROOT/'transvision/models/event_track_v2x/learned_component_allocation.py',
                 ROOT/'transvision/models/event_track_v2x/forest_cache_stream.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha_file(path)
    plan = dict(kind='train_fitted_persistent_identity_spd_val_plan_v1', cache_sha256=args.cache_sha256,
        schedule_sha256=args.schedule_sha256, split_sha256=SPLIT_SHA, checkpoint_sha256=args.checkpoint_sha256,
        scorer_signature=scorer.signature, configuration=asdict(config), source_sha256=sources,
        class_scope=['car'], full_train_identity_fitted=True, checkpoint_seed=checkpoint['seed'],
        val_seen_during_research=True, validation_parameter_search=False, validation_checkpoint_selection=False,
        strict_pipeline_isolated_selection=False, paper_eligible=False, all_three_seeds_required_for_comparison=True)
    result = replay_rows(cache, rows, args.output, config, plan=plan, learned_scorer=scorer)
    if any(sha_file(ROOT/p) != sha for p, sha in sources.items()):
        raise ValueError('sources changed during validation; receipt not frozen')
    _new(args.output/'full-validation-receipt.json', dict(result,
         full_official_validation_schedule_completed=result['completed_frames'] == 3316,
         checkpoint_sha256=args.checkpoint_sha256, checkpoint_seed=checkpoint['seed']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'schedule', 'checkpoint', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('cache-sha256', 'schedule-sha256', 'checkpoint-sha256'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--device', default='cpu')
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
