#!/usr/bin/env python3
"""Reproduce ONLY the next failed train event on a copy of its closed database.

No capacity change or live code instrumentation. Whitelisted numeric traceback
locals explain the exhausted frontier after the normal transaction rollback.
Original artifacts are hash-pinned and untouched; no complete-sequence metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from tools.event_track_v2x.audit_train_inference_comparison import inspect_failure
from tools.event_track_v2x.run_train_inference_diagnostic import diagnostic_sources
from transvision.models.event_track_v2x.allocation_training import _directory
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,sha_file
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery,VerifiedForestCache
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.persistent_class_bound_beam import PersistentClassBoundBeamTracker
from transvision.models.event_track_v2x.persistent_class_bound_beam import _RankingWork
from transvision.models.event_track_v2x.persistent_slot_bound_beam import PersistentSlotBoundBeamTracker, _SlotEnvelope
from transvision.models.event_track_v2x.persistent_sparse_slot_bound_beam import PersistentSparseSlotBoundBeamTracker
from transvision.models.event_track_v2x.persistent_beam_tracking import PersistentRankedBeamTracker, PersistentBeamTracker
from transvision.models.event_track_v2x.persistent_forest import PersistentForestTracker
from transvision.models.event_track_v2x.persistent_joint_beam import PersistentJointBeamTracker
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentTracker
from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream
from transvision.models.event_track_v2x.resource_sweep import THREAD_ENV


def state_failure_context(frames):
    replay=next((f for f in reversed(frames) if f.f_code is PersistentForestTracker._ensure_state.__code__),None)
    beam=next((f for f in frames if f.f_code is PersistentBeamTracker._component_inference.__code__),None)
    if replay is None or beam is None:return None
    data,outer=replay.f_locals,beam.f_locals;kernel=data['self'];change=outer.get('change')
    prefix=data.get('prefix');current=data.get('current');old=data.get('state')
    return dict(failure_stage='branch_state_replay',component=outer['component'],component_nodes=kernel.n,
        state_updates=kernel.state_updates,state_update_cap=kernel.config.state.max_replay_operations,
        prior_components_state_updates=outer['state_work'],retained_classes=len(outer['active']),
        requested_branch_index=outer['active'].index(outer['h']),
        pending_root_prefixes=len(data['pending']),current_prefix_depth=None if prefix is None else prefix.depth,
        late_raw_replay_active='query' in data,component_merge=outer['component'] in outer['products'],
        predecessor_components=[] if change is None else list(change.predecessors),
        new_component_nodes=0 if change is None else len(change.added_indices),
        old_cached_state_last_us=None if old is None else old['last_us'],
        current_replay_observation_us=None if current is None else current.state_us,
        traceback_state_read_after_normal_rollback=True,full_raw_payloads_recorded=False)


def failure_context(error):
    frames=[];tb=error.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame);tb=tb.tb_next
    ranked=next((f for f in reversed(frames) if f.f_code in (
        PersistentJointBeamTracker._advance_component.__code__,PersistentRankedBeamTracker._extend.__code__)),None)
    event=next((f for f in frames if f.f_code is PersistentComponentTracker.step.__code__),None)
    if ranked is None or event is None: return state_failure_context(frames)
    data=ranked.f_locals;state=event.f_locals
    queue=data.get('queue',[]);retained=data.get('retained',[])
    advance=next((f for f in reversed(frames) if 'change' in f.f_locals
                  and f.f_locals.get('kernel') is data['kernel']),None)
    change=data.get('change') if advance is None else advance.f_locals['change']
    indices=None if change is None else set(change.added_indices)
    slots=Counter((o.node.source_id,o.node.frame_id) for i,o in enumerate(state['new'],state['old_n'])
                  if indices is None or i in indices)
    context=dict(ranking_function=ranked.f_code.co_name,start_depth=data['start'],
        component_nodes=data['kernel'].n,current_prefix_depth=data['prefix'].depth if 'prefix' in data else None,
        component=data.get('component') if advance is None else advance.f_locals.get('component'),
        queue_entries=len(queue),frontier_peak=data.get('peak'),
        next_choice_count=len(data.get('choices',[])),retained_complete_classes=len(retained),
        visited_complete_classes=data.get('terminal_count'),width=data['self'].config.state.active_limit,
        best_frontier_log_upper=-queue[0][0] if queue else None,
        kth_complete_log_weight=-retained[-1][0] if len(retained)==data['self'].config.state.active_limit else None,
        new_component_source_frame_counts=[dict(source=s,frame=f,n=n) for (s,f),n in sorted(slots.items())],
        source_frame_exclusion_in_ranking_bound=isinstance(data['self'],PersistentSlotBoundBeamTracker),
        traceback_state_read_after_normal_rollback=True)
    cursor=data.get('cursor')
    if cursor is not None:
        context.update(prior_combinations_materialized=cursor.emitted,
            cartesian_queue_entries=len(cursor.queue),cartesian_states_seen=len(cursor.seen),
            predecessor_widths=[len(options) for _,options in cursor.groups],
            log_prior_combinations=math.fsum(math.log(len(options)) for _,options in cursor.groups))
    upper=data.get('prefix_upper',data.get('upper'))
    envelope=getattr(upper,'__self__',None)
    if envelope is not None and hasattr(envelope,'work'):
        work=envelope.work
        context.update(ranking_operations=work.total,ranking_work_counts=dict(work.counts),
            ranking_operation_cap=work.config.max_ranking_operations,
            ranking_catalog_terms_peak=work.catalog_peak,ranking_root_cache_entries_peak=work.root_cache_peak)
        if isinstance(envelope,_SlotEnvelope):
            context.update(assignment_matrix_cells_peak=getattr(work,'assignment_matrix_peak',0),
                assignment_matrix_cells_cap=work.config.max_assignment_matrix_cells,
                assignment_solves=work.counts['assignment_solves'],
                assignment_solves_cap=work.config.max_assignment_solves,
                assignment_dual_gap_max=getattr(work,'assignment_dual_gap_max',0.))
    charge=next((f for f in reversed(frames) if f.f_code is _RankingWork.charge.__code__),None)
    if charge is not None:
        context.update(rejected_ranking_work_kind=charge.f_locals['kind'],
            rejected_ranking_work_amount=charge.f_locals['amount'])
    assignment=next((f for f in reversed(frames) if f.f_code is _SlotEnvelope._assignment.__code__),None)
    if assignment is not None:
        a=assignment.f_locals
        context.update(current_assignment_rows=a.get('n'),current_assignment_cells=a.get('cells'),
            current_assignment_available_roots=len(a['available']),
            current_assignment_unknown_rows=sum(math.isfinite(row[2]) for row in a['details']),
            current_assignment_direct_known_entries=sum(len(row[1]) for row in a['details']))
    return context


def diagnose(failed_replay,failure_sha256,database_sha256,cache_root,checkpoint,checkpoint_sha256,output):
    import torch
    if any(os.environ.get(k)!=v for k,v in THREAD_ENV.items()): raise ValueError('pinned fresh-process threads required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    failed=Path(failed_replay);report=inspect_failure(failed,failure_sha256);plan=report['plan']
    tracker_types={'class_bound_joint_beam':PersistentClassBoundBeamTracker,
                   'slot_bound_joint_beam':PersistentSlotBoundBeamTracker,
                   'sparse_slot_bound_joint_beam':PersistentSparseSlotBoundBeamTracker,
                   'node_beam':PersistentBeamTracker}
    if (plan['backend'] not in tracker_types or plan['class_scope']!=['car']
            or plan['source_sha256']!=diagnostic_sources()
            or plan['identity_checkpoint_sha256']!=checkpoint_sha256):
        raise ValueError('matching failed ranking-bound train source/checkpoint required')
    database=failed/'sequence-0000.sqlite'
    if sha_file(database)!=database_sha256 or any(Path(str(database)+s).exists() for s in ('-wal','-journal')):
        raise ValueError('pinned closed database required')
    previous=json.loads((failed/'tracking.jsonl').read_bytes().splitlines()[-1])['tracking']
    index=report['failure']['completed_frames'];row=plan['selected_schedule'][index]
    if index<1 or row['sequence_id']!=previous['sequence_id']:
        raise ValueError('same-sequence committed prefix and next event required')
    cache=VerifiedForestCache(cache_root,plan['cache_sha256'])
    output=_directory(output);copy=output/'diagnostic-copy.sqlite';shutil.copy2(database,copy)
    tracker=tracker_types[plan['backend']].open(copy,expected_database_sha256=database_sha256,
        expected_prediction_sha256=previous['prediction_sha256'])
    inputs={failed/name:digest for name,digest in report['captured_artifact_sha256'].items()}
    inputs.update({ROOT/path:digest for path,digest in plan['source_sha256'].items()})
    inputs[Path(__file__)]=sha_file(__file__)
    inputs[Path(checkpoint)/'checkpoint.json']=checkpoint_sha256
    before=dict(tracker.meta);result=None
    try:
        scorer,_=load_identity_checkpoint(checkpoint,checkpoint_sha256,config=tracker.config.state)
        if scorer.signature!=plan['scorer_signature']: raise ValueError('frozen scoring signature differs')
        origin=min(json.loads(meta)['box_reference_timestamp_us'] for key,(_,meta) in cache.index.items()
                   if key[0]==tracker.sequence_id)
        stream=PersistentForestCacheStream(cache,tracker,scorer,origin_us=origin)
        decision=row['box_reference_timestamp_us']+100_000;deliveries=[]
        for side,key in (('vehicle-side','vehicle_frame'),('infrastructure-side','infrastructure_frame')):
            entry,meta=(json.loads(v) for v in cache.index[(tracker.sequence_id,side,row[key])])
            if max(meta['box_reference_timestamp_us'],meta['source_image_timestamp_us'])<=decision:
                deliveries.append(CacheDelivery(tracker.sequence_id,side,row[key],decision,entry['frame_sha256']))
        try:
            stream.step(deliveries,frame_id=row['vehicle_frame'],reference_us=row['box_reference_timestamp_us'],
                        decision_us=decision,event_id=row['vehicle_frame'])
        except ValueError as error:
            context=failure_context(error)
            result=dict(kind='isolated_failed_event_diagnostic_v1',status='diagnostic_complete',
                backend=plan['backend'],expected_error=report['failure']['error'],actual_error=str(error),
                expected_failure_reproduced=str(error)==report['failure']['error'] and context is not None,
                failed_event_index_zero_based=index,frame_id=row['vehicle_frame'],context=context,
                prior_event_state_unchanged=tracker.meta==before,original_failure_sha256=failure_sha256,
                original_database_sha256=database_sha256,ground_truth_used=False,parameter_training=False,
                live_instrumentation=False,complete_sequence_replay=False,performance_benchmark=False,paper_eligible=False)
        else:
            raise ValueError('expected failed event succeeded; diagnostic assumption invalid')
    finally:
        tracker.close()
    if any(sha_file(path)!=digest for path,digest in inputs.items()): raise ValueError('source/input changed during diagnosis')
    result.update(input_sha256={str(p):h for p,h in inputs.items()},
                  diagnostic_copy_sha256=sha_file(copy),copy_bytes_equal_original=sha_file(copy)==database_sha256)
    with (output/'diagnostic.json').open('xb') as stream: stream.write(canonical(result))
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},sort_keys=True))
    if not result['expected_failure_reproduced'] or not result['prior_event_state_unchanged']:
        raise ValueError('failed event or rollback did not reproduce')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('failed-replay','cache','checkpoint','output'): parser.add_argument('--'+name,type=Path,required=True)
    for name in ('failure-sha256','database-sha256','checkpoint-sha256'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    diagnose(args.failed_replay,args.failure_sha256,args.database_sha256,args.cache,args.checkpoint,args.checkpoint_sha256,args.output)
