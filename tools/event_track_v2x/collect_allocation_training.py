#!/usr/bin/env python3
"""Offline counterfactual search traces on the sealed full SPD TRAIN cache.

No GT, official val/test, external writes or actual network-delay claim. Solver
caps fail the run; raw nodes/edges are never silently removed to pass capacity.
Teacher overhead is not online latency. Geometry is an explicit optional mode.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.prepare_forest_training import TRAIN_CACHE_SHA256
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows, _new
from tools.event_track_v2x.train_forest_identity import training_sources
from transvision.models.event_track_v2x.allocation_training import allocation_sources, training_binding
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig
from transvision.models.event_track_v2x.completion_component_tracking import PersistentCompletionConfig
from transvision.models.event_track_v2x.covered_completion_tracking import PersistentCoveredCompletionConfig
from transvision.models.event_track_v2x.beam_recovery_tracking import BeamRecoveryConfig
from transvision.models.event_track_v2x.recovery_task_scope import SCOPE_MODE, VerifiedEgoPoseTable


def select_teacher_schedule(rows, development_sequence=None):
    if development_sequence is None:
        return rows, True
    if not isinstance(development_sequence,str) or development_sequence not in {r['sequence_id'] for r in rows}:
        raise ValueError('development sequence must belong to the verified train schedule')
    return [r for r in rows if r['sequence_id']==development_sequence], False


def teacher_config_type(*, frontier_completion=False, coverage_aware_admission=False, beam_recovery=False):
    if any(type(v) is not bool for v in (frontier_completion, coverage_aware_admission, beam_recovery)):
        raise ValueError('explicit boolean teacher variants required')
    if beam_recovery:
        if frontier_completion or coverage_aware_admission:
            raise ValueError('beam recovery uses its own completion policy; component flags are exclusive')
        return BeamRecoveryConfig
    if coverage_aware_admission and not frontier_completion:
        raise ValueError('coverage-aware admission requires frontier completion')
    return (PersistentCoveredCompletionConfig if coverage_aware_admission else
            PersistentCompletionConfig if frontier_completion else PersistentComponentConfig)


def verified_train_rows(cache, cooperative_metadata, metadata_sha256, *, require_full_train=True):
    """Shared prediction-only schedule check; relaxed size is test-only."""
    if type(cache) is not VerifiedForestCache or type(require_full_train) is not bool:
        raise TypeError('verified cache and explicit cohort mode required')
    metadata = json.loads(cache.manifest_json)
    if (metadata['split'] != 'train' or require_full_train and
            (cache.manifest_sha256 != TRAIN_CACHE_SHA256 or len(metadata['sequences']) != 46 or len(cache.index) != 16338)):
        raise ValueError('sealed full official train cache required')
    cooperative_metadata = Path(cooperative_metadata)
    if sha_file(cooperative_metadata) != metadata_sha256:
        raise ValueError('train cooperative metadata identity differs')
    pairs = json.loads(cooperative_metadata.read_bytes())
    if (not isinstance(pairs,list) or not pairs or require_full_train and len(pairs) != 7445
            or any(len({p[field] for p in pairs}) != len(pairs)
            for field in ('vehicle_frame', 'infrastructure_frame'))):
        raise ValueError('full train pair cohort and unique frames required')
    rows = []
    for pair in pairs:
        scene = pair['vehicle_sequence']
        if scene != pair['infrastructure_sequence']:
            raise ValueError('train cooperative pair crosses sequences')
        for side, name in (('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')):
            if (scene, side, pair[name]) not in cache.index:
                raise ValueError('cooperative pair outside sealed train cache')
        _, meta = cache.index[(scene, 'vehicle-side', pair['vehicle_frame'])]
        rows.append(dict(sequence_id=scene, vehicle_frame=pair['vehicle_frame'],
            infrastructure_frame=pair['infrastructure_frame'],
            box_reference_timestamp_us=json.loads(meta)['box_reference_timestamp_us']))
    rows.sort(key=lambda r: (r['sequence_id'], r['box_reference_timestamp_us']))
    if {r['sequence_id'] for r in rows} != set(metadata['sequences']):
        raise ValueError('train pair schedule omits a sequence')
    if sha_file(cooperative_metadata) != metadata_sha256:
        raise ValueError('train cooperative metadata changed during schedule verification')
    return rows


def run(args):
    config_type = teacher_config_type(frontier_completion=getattr(args,'frontier_completion',False),
        coverage_aware_admission=getattr(args,'coverage_aware_admission',False),
        beam_recovery=getattr(args,'beam_recovery',False))
    pose_path,pose_sha=getattr(args,'recovery_task_poses',None),getattr(args,'recovery_task_poses_sha256',None)
    if (pose_path is None) != (pose_sha is None) or pose_path is not None and config_type is not BeamRecoveryConfig:
        raise ValueError('task pose path/hash require beam recovery together')
    if config_type is BeamRecoveryConfig:
        config = config_type(state=ForestTrackingConfig(active_limit=args.active_limit,
            max_model_regret=args.max_model_regret),recovery_budget=args.expansion_budget,
            recovery_allocation_scope=SCOPE_MODE if pose_path is not None else 'all_recent')
    else:
        config = config_type(state=ForestTrackingConfig(active_limit=args.active_limit,
            expansion_budget=args.expansion_budget, max_model_regret=args.max_model_regret))
    if (args.checkpoint is None) != (args.checkpoint_sha256 is None):
        raise ValueError('identity checkpoint and hash required together')
    cache = VerifiedForestCache(args.cache, TRAIN_CACHE_SHA256)
    ego_poses=None if pose_path is None else VerifiedEgoPoseTable(pose_path,pose_sha,cache)
    rows = verified_train_rows(cache,args.cooperative_metadata,args.cooperative_metadata_sha256)
    rows,full_trace = select_teacher_schedule(rows,getattr(args,'development_sequence',None))
    frozen = frozen_cache_identity(cache)
    scorer, checkpoint = None, None
    if args.checkpoint is not None:
        scorer, checkpoint = load_identity_checkpoint(args.checkpoint, args.checkpoint_sha256, config=config.state)
        if not checkpoint['full_official_train'] or checkpoint['frozen_cache_identity'] != frozen:
            raise ValueError('full-train identity checkpoint and matching upstream required')
    signature = scorer.signature if scorer is not None else GeometryForestScorer(process_noise=config.state.process_noise).signature
    sources = dict(training_sources(), **allocation_sources())
    for path in (Path(__file__), ROOT/'tools/event_track_v2x/run_persistent_forest_v2.py',
                 ROOT/'transvision/models/event_track_v2x/experiment_progress.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha_file(path)
    plan = dict(kind='component_allocation_teacher_full_train_plan_v1', cache_sha256=TRAIN_CACHE_SHA256,
        cooperative_metadata_sha256=args.cooperative_metadata_sha256, configuration=asdict(config),
        full_official_train_verified=full_trace,input_full_official_train_verified=True,
        development_sequence=getattr(args,'development_sequence',None),scheduled_frames=len(rows),
        source_sha256=sources, class_scope=['car'],
        allocation_training_binding=training_binding(config, signature, frozen),
        identity_checkpoint_sha256=args.checkpoint_sha256, geometry_development=scorer is None,
        upstream_in_sample=True, strict_pipeline_isolated_selection=False, paper_eligible=False)
    result = replay_rows(cache, rows, args.output, config, plan=plan, learned_scorer=scorer,
                         allocation_teacher=True,ego_poses=ego_poses)
    if (sha_file(args.cooperative_metadata) != args.cooperative_metadata_sha256
            or any(sha_file(ROOT/p) != h for p, h in sources.items())):
        raise ValueError('teacher sources or metadata changed during replay')
    if full_trace:
        _new(args.output/'full-train-teacher-receipt.json', dict(result, full_official_train_trace_completed=True))
    else:
        _new(args.output/'development-teacher-receipt.json',dict(result,full_official_train_trace_completed=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'cooperative-metadata', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--cooperative-metadata-sha256', required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--active-limit', type=int, default=4)
    parser.add_argument('--expansion-budget', type=int, default=256)
    parser.add_argument('--max-model-regret', type=float, default=.05)
    parser.add_argument('--frontier-completion',action='store_true',help='Enable the separately versioned frontier-completion variant.')
    parser.add_argument('--coverage-aware-admission',action='store_true',
        help='Use the separate coverage-capacity variant; requires --frontier-completion.')
    parser.add_argument('--beam-recovery',action='store_true',
        help='Teacher on the shared node-beam recovery backend; expansion-budget is extra recovery work.')
    parser.add_argument('--recovery-task-poses',type=Path,
        help='Optional arrived-vehicle-pose allocation scope; requires --beam-recovery.')
    parser.add_argument('--recovery-task-poses-sha256')
    parser.add_argument('--development-sequence',help='Replay one complete TRAIN sequence for diagnostics; never marks full train complete.')
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
