#!/usr/bin/env python3
"""One complete TRAIN sequence, frozen learned factors, no teacher probes or GT.

Run each backend in a fresh process. This is development inference, not full
train, tracking validation, a three-seed result, or an equal-resource benchmark.
The CLI requires the real sealed full-train cache and trained checkpoint; the
separate low-level test API cannot label fixtures as a verified real cohort.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from tools.event_track_v2x.collect_allocation_training import verified_train_rows, select_teacher_schedule
from tools.event_track_v2x.prepare_forest_training import TRAIN_CACHE_SHA256
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows, _new
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.resource_sweep import configuration, source_snapshot, THREAD_ENV
from transvision.models.event_track_v2x.recovery_task_scope import SCOPE_MODE, VerifiedEgoPoseTable

BACKENDS = ('component_completion','component_covered_completion','joint_beam')
ADDITIONAL_BACKENDS = ('class_bound_joint_beam','slot_bound_joint_beam','sparse_slot_bound_joint_beam',
                       'reachable_slot_bound_joint_beam','node_beam',
                       'beam_recovery','beam_recovery_disabled')
REPLAY_TOOLS = frozenset(('run_train_inference_diagnostic.py','collect_allocation_training.py',
    'prepare_forest_training.py','run_persistent_forest_v2.py','run_tracking_v2.py','train_forest_identity.py'))


def diagnostic_sources():
    # All model modules/initializers plus the tool dependency closure. Unrelated
    # report/evaluation tools may be developed without changing live inference.
    return {path:digest for path,digest in source_snapshot().items()
            if not path.startswith('tools/') or Path(path).name in REPLAY_TOOLS}


def diagnostic_configuration(backend, max_model_regret=None, *, task_scoped_recovery=False):
    values={}
    if type(task_scoped_recovery) is not bool or task_scoped_recovery and backend not in ('beam_recovery','beam_recovery_disabled'):
        raise ValueError('explicit task allocation control only for beam recovery on/off')
    if task_scoped_recovery:values['recovery_allocation_scope']=SCOPE_MODE
    if max_model_regret is not None:
        if (backend not in ('component_completion','component_covered_completion')
                or type(max_model_regret) not in (int,float)):
            raise ValueError('explicit numeric regret override is only for recoverable controls')
        values['state']=dict(max_model_regret=max_model_regret)
    return configuration(dict(backend=backend,configuration=values))


def run(cache, cooperative_metadata, metadata_sha256, checkpoint, checkpoint_sha256,
        output, *, sequence, backend, allow_fixture=False, max_model_regret=None,
        ego_pose_table=None, ego_pose_table_sha256=None):
    import torch
    if type(allow_fixture) is not bool or backend not in BACKENDS+ADDITIONAL_BACKENDS or not isinstance(sequence,str) or not sequence:
        raise ValueError('known inference backend and explicit train sequence required')
    if any(os.environ.get(key) != value for key,value in THREAD_ENV.items()):
        raise ValueError('fresh-process thread environment is not explicitly pinned')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = verified_train_rows(cache,cooperative_metadata,metadata_sha256,require_full_train=not allow_fixture)
    rows,_ = select_teacher_schedule(rows,sequence)
    if (ego_pose_table is None) != (ego_pose_table_sha256 is None):
        raise ValueError('pose sidecar and pinned hash required together')
    config = diagnostic_configuration(backend,max_model_regret,task_scoped_recovery=ego_pose_table is not None)
    poses = None if ego_pose_table is None else VerifiedEgoPoseTable(ego_pose_table,ego_pose_table_sha256,cache)
    scorer,metadata = load_identity_checkpoint(checkpoint,checkpoint_sha256,config=config.state)
    if ((not allow_fixture and not metadata['full_official_train'])
            or metadata['frozen_cache_identity'] != frozen_cache_identity(cache)):
        raise ValueError('matching frozen full-train identity checkpoint required')
    sources = diagnostic_sources()
    plan = dict(kind='train_sequence_inference_diagnostic_plan_v1',backend=backend,
        configuration=asdict(config),source_sha256=sources,cache_sha256=cache.manifest_sha256,
        source_scope='model_modules_initializers_and_declared_replay_tool_dependencies',
        cooperative_metadata_sha256=metadata_sha256,identity_checkpoint_sha256=checkpoint_sha256,
        identity_seed=metadata['seed'],scorer_signature=scorer.signature,
        cohort_mode='fixture-only' if allow_fixture else 'real-train-development',
        complete_input_train_cohort_verified=not allow_fixture,selected_sequence=sequence,
        selected_schedule=rows,scheduled_frames=len(rows),class_scope=['car'],
        full_official_train_trace_completed=False,upstream_in_sample=True,
        strict_pipeline_isolated_selection=False,real_tracking_evaluation=False,
        max_model_regret_override=max_model_regret,
        conditional_bayes_without_threshold_fallback=config.state.max_model_regret==1.,
        physically_equal_resources_claimed=False,three_seed_comparison=False,paper_eligible=False,
        thread_environment={key:os.environ[key] for key in THREAD_ENV},
        runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
            host=platform.node(),pid=os.getpid(),torch=torch.__version__))
    output = Path(output)
    result = replay_rows(cache,rows,output,config,plan=plan,learned_scorer=scorer,ego_poses=poses)
    # The outer diagnostic is accepted only with this separate final receipt.
    if (diagnostic_sources() != sources or sha_file(cooperative_metadata) != metadata_sha256
            or sha_file(Path(checkpoint)/'checkpoint.json') != checkpoint_sha256
            or ego_pose_table is not None and sha_file(ego_pose_table)!=ego_pose_table_sha256):
        _new(output/'diagnostic-failure.json',dict(status='failed',reason='source_or_input_changed',paper_eligible=False))
        raise ValueError('diagnostic source or input changed; no final diagnostic receipt')
    final = dict(kind='train_sequence_inference_diagnostic_v1',status='complete',backend=backend,
        replay_receipt_sha256=sha_file(output/'receipt.json'),
        plan_sha256=sha_file(output/'plan.json'),completed_frames=result['completed_frames'],
        cohort_mode=plan['cohort_mode'],selected_sequence=sequence,
        complete_selected_sequence_verified=not allow_fixture,
        full_official_train_trace_completed=False,training_performed=False,
        offline_teacher_probes=False,real_tracking_evaluation=False,paper_eligible=False)
    _new(output/'development-inference-receipt.json',final)
    print(json.dumps(final,sort_keys=True))
    return final


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','cooperative-metadata','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cooperative-metadata-sha256',required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--sequence',required=True)
    parser.add_argument('--backend',choices=BACKENDS+ADDITIONAL_BACKENDS,required=True)
    parser.add_argument('--ego-pose-table',type=Path,help='Explicit allocation-only raw-XY50 control with arrived vehicle poses.')
    parser.add_argument('--ego-pose-table-sha256')
    parser.add_argument('--max-model-regret',type=float,
        help='Explicit train-only decision control; 1 disables threshold fallback, not uncertainty auditing. Default unchanged.')
    args=parser.parse_args()
    cache=VerifiedForestCache(args.cache,TRAIN_CACHE_SHA256)
    run(cache,args.cooperative_metadata,args.cooperative_metadata_sha256,args.checkpoint,args.checkpoint_sha256,
        args.output,sequence=args.sequence,backend=args.backend,max_model_regret=args.max_model_regret,
        ego_pose_table=args.ego_pose_table,ego_pose_table_sha256=args.ego_pose_table_sha256)
