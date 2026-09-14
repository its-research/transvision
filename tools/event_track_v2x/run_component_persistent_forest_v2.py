#!/usr/bin/env python3
"""Development full-SPD-val replay with persistent exact-support components.

Geometry is the explicit baseline. A supplied checkpoint must have been fitted
on full official train with identical frozen upstream producers. An optional
priority checkpoint must bind the same factors/configuration and full train
teacher cohort; its sequence holdout is excluded from head fitting. No GT/test,
validation parameter search, verified tracking gain, or ClearML publication.
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
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig
from transvision.models.event_track_v2x.completion_component_tracking import PersistentCompletionConfig
from transvision.models.event_track_v2x.allocation_training import allocation_sources, load_priority, training_binding
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer


def run(args):
    defaults = ForestTrackingConfig()
    config_type = PersistentCompletionConfig if getattr(args,'frontier_completion',False) else PersistentComponentConfig
    config = config_type(state=ForestTrackingConfig(**{key: getattr(args, key, getattr(defaults, key))
        for key in ('active_limit', 'expansion_budget', 'max_model_regret')}))
    if (args.checkpoint is None) != (args.checkpoint_sha256 is None):
        raise ValueError('checkpoint path and hash must be supplied together')
    allocation_path, allocation_sha = getattr(args, 'allocation_checkpoint', None), getattr(args, 'allocation_checkpoint_sha256', None)
    if (allocation_path is None) != (allocation_sha is None):
        raise ValueError('allocation checkpoint path and hash must be supplied together')
    scorer, checkpoint = None, None
    if args.checkpoint is not None:
        scorer, checkpoint = load_identity_checkpoint(args.checkpoint, args.checkpoint_sha256, config=config.state, device=args.device)
        if checkpoint.get('full_official_train') is not True:
            raise ValueError('fixture checkpoint cannot represent full-train component validation')
    rows = schedule_rows(args.schedule, args.schedule_sha256)
    cache = VerifiedForestCache(args.cache, args.cache_sha256)
    manifest = json.loads(cache.manifest_json)
    if (manifest['split'] != 'val' or manifest['frame_count'] != 7189
            or set(manifest['sequences']) != {r['sequence_id'] for r in rows}):
        raise ValueError('full sealed SPD validation cache required')
    if checkpoint is not None and frozen_cache_identity(cache) != checkpoint['frozen_cache_identity']:
        raise ValueError('validation upstream differs from frozen training producers')
    allocation_policy = None
    if allocation_path is not None:
        signature = scorer.signature if scorer is not None else GeometryForestScorer(process_noise=config.state.process_noise).signature
        allocation_policy, _ = load_priority(allocation_path, allocation_sha,
            binding=training_binding(config, signature, frozen_cache_identity(cache)))
    sources = dict(training_sources(), **allocation_sources())
    for path in (Path(__file__), ROOT/'tools/event_track_v2x/run_persistent_forest_v2.py',
                 ROOT/'tools/event_track_v2x/run_tracking_v2.py',
                 *(ROOT/'transvision/models/event_track_v2x'/name for name in
                   ('persistent_component_store.py', 'persistent_component_tracking.py', 'persistent_beam_tracking.py', 'persistent_joint_beam.py',
                    'persistent_cache_stream.py', 'persistent_forest.py', 'forest_cache_stream.py'))):
        sources[path.relative_to(ROOT).as_posix()] = sha_file(path)
    plan = dict(kind='persistent_component_identity_spd_val_development_plan_v1',
        cache_sha256=args.cache_sha256, schedule_sha256=args.schedule_sha256, split_sha256=SPLIT_SHA,
        checkpoint_sha256=args.checkpoint_sha256, scorer_signature=None if scorer is None else scorer.signature,
        configuration=asdict(config), source_sha256=sources, class_scope=['car'],
        geometry_baseline=scorer is None, checkpoint_seed=None if checkpoint is None else checkpoint['seed'],
        factor_representation='exact_root_partition_classes_with_seeded_lower_mass_v1',
        conditional_bayes_comparison_without_regret_fallback=config.state.max_model_regret == 1.,
        val_seen_during_research=True, validation_parameter_search=False, validation_checkpoint_selection=False,
        strict_pipeline_isolated_selection=False, learned_allocation=allocation_policy is not None,
        allocation_checkpoint_sha256=allocation_sha, paper_eligible=False)
    result = replay_rows(cache, rows, args.output, config, plan=plan, learned_scorer=scorer, allocation_policy=allocation_policy)
    if any(sha_file(ROOT/p) != sha for p, sha in sources.items()):
        raise ValueError('component validation sources changed during replay')
    _new(args.output/'full-validation-receipt.json', dict(result,
        full_official_validation_schedule_completed=result['completed_frames'] == 3316,
        checkpoint_sha256=args.checkpoint_sha256))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'schedule', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('cache-sha256', 'schedule-sha256'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--allocation-checkpoint', type=Path)
    parser.add_argument('--allocation-checkpoint-sha256')
    parser.add_argument('--device', default='cpu')
    defaults = ForestTrackingConfig()
    parser.add_argument('--active-limit', type=int, default=defaults.active_limit)
    parser.add_argument('--expansion-budget', type=int, default=defaults.expansion_budget)
    parser.add_argument('--max-model-regret', type=float, default=defaults.max_model_regret)
    parser.add_argument('--frontier-completion',action='store_true',help='Use the separately versioned frontier-completion variant.')
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
