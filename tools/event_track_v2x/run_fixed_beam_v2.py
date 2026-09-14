#!/usr/bin/env python3
"""Full SPD official-val car replay: declared fixed-width irreversible beam.

Uses identical raw V2 selection, candidate-row context and state updates. Width
is an explicit predeclared experimental parameter, not automatically selected
on validation. No GT/test, metric lookup, checkpoint selection or publication.
This local beam implementation is not a reproduced classical MHT baseline.
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
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.persistent_beam_tracking import (
    PersistentBeamConfig, PersistentBeamTracker, PersistentRankedBeamConfig, PersistentRankedBeamTracker,
)
from transvision.models.event_track_v2x.persistent_joint_beam import PersistentJointBeamTracker


def run(args):
    selection = getattr(args, 'selection', 'joint')
    choices = dict(node=PersistentBeamTracker, batch=PersistentRankedBeamTracker, joint=PersistentJointBeamTracker)
    if selection not in choices:
        raise ValueError('beam selection must be node, batch or joint')
    tracker_type = choices[selection]
    config = tracker_type.CONFIG_TYPE(state=ForestTrackingConfig(active_limit=args.width))
    if (args.checkpoint is None) != (args.checkpoint_sha256 is None):
        raise ValueError('checkpoint path and hash must be supplied together')
    scorer, checkpoint = None, None
    if args.checkpoint is not None:
        scorer, checkpoint = load_identity_checkpoint(args.checkpoint, args.checkpoint_sha256, config=config.state, device=args.device)
        if checkpoint.get('full_official_train') is not True:
            raise ValueError('fixture checkpoint cannot represent full-train beam validation')
    rows = schedule_rows(args.schedule, args.schedule_sha256)
    cache = VerifiedForestCache(args.cache, args.cache_sha256)
    manifest = json.loads(cache.manifest_json)
    if (manifest['split'] != 'val' or manifest['frame_count'] != 7189
            or set(manifest['sequences']) != {r['sequence_id'] for r in rows}):
        raise ValueError('full sealed SPD validation cache required')
    if checkpoint is not None and frozen_cache_identity(cache) != checkpoint['frozen_cache_identity']:
        raise ValueError('validation upstream differs from frozen training producers')
    sources = training_sources()
    for path in (Path(__file__), ROOT/'tools/event_track_v2x/run_persistent_forest_v2.py',
                 ROOT/'tools/event_track_v2x/run_tracking_v2.py',
                 *(ROOT/'transvision/models/event_track_v2x'/name for name in
                   ('persistent_beam_tracking.py', 'persistent_joint_beam.py', 'persistent_component_store.py', 'persistent_component_tracking.py',
                    'allocation_policy.py', 'learned_component_allocation.py',
                    'persistent_cache_stream.py', 'persistent_forest.py', 'forest_cache_stream.py'))):
        sources[path.relative_to(ROOT).as_posix()] = sha_file(path)
    plan = dict(kind='persistent_fixed_beam_spd_val_development_plan_v1',
        cache_sha256=args.cache_sha256, schedule_sha256=args.schedule_sha256, split_sha256=SPLIT_SHA,
        checkpoint_sha256=args.checkpoint_sha256, scorer_signature=None if scorer is None else scorer.signature,
        configuration=asdict(config), source_sha256=sources, class_scope=['car'],
        geometry_baseline=scorer is None, checkpoint_seed=None if checkpoint is None else checkpoint['seed'],
        beam_width=args.width, irreversible=True, reproduced_classical_mht=False,
        pruning_policy=tracker_type.PRUNING_POLICY, selection=selection,
        component_merge_policy=('full_retained_cartesian_product_jointly_with_new_batch' if selection == 'joint'
                                else 'top_k_prior_product_before_new_batch_extension'),
        input_context_and_states_shared_with_recoverable=True, same_latency_or_memory_verified=False,
        val_seen_during_research=True, validation_parameter_search=False, validation_checkpoint_selection=False,
        strict_pipeline_isolated_selection=False, learned_allocation=False, paper_eligible=False)
    result = replay_rows(cache, rows, args.output, config, plan=plan, learned_scorer=scorer)
    if any(sha_file(ROOT/p) != sha for p, sha in sources.items()):
        raise ValueError('beam validation sources changed during replay')
    _new(args.output/'full-validation-receipt.json', dict(result,
        full_official_validation_schedule_completed=result['completed_frames'] == 3316,
        checkpoint_sha256=args.checkpoint_sha256, beam_width=args.width))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'schedule', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('cache-sha256', 'schedule-sha256'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--selection', choices=('joint', 'batch', 'node'), default='joint')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--device', default='cpu')
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
