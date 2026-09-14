#!/usr/bin/env python3
"""Bounded train-prefix ranking probe; never a complete sequence result."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from tools.event_track_v2x.collect_allocation_training import verified_train_rows,select_teacher_schedule
from tools.event_track_v2x.prepare_forest_training import TRAIN_CACHE_SHA256
from tools.event_track_v2x.run_train_inference_diagnostic import diagnostic_sources
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows,_new
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.resource_sweep import THREAD_ENV,configuration


def run_probe(cache,metadata,metadata_sha,checkpoint,checkpoint_sha,output,*,sequence,events,allow_fixture=False,
              backend='slot_bound_joint_beam'):
    import torch
    if type(allow_fixture) is not bool or type(events) is not int or events<1:
        raise ValueError('explicit positive prefix length required')
    if backend not in ('slot_bound_joint_beam','sparse_slot_bound_joint_beam','reachable_slot_bound_joint_beam'):
        raise ValueError('declared slot-bound backend required')
    if any(os.environ.get(k)!=v for k,v in THREAD_ENV.items()): raise ValueError('pinned fresh-process threads required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    complete=verified_train_rows(cache,metadata,metadata_sha,require_full_train=not allow_fixture)
    complete,_=select_teacher_schedule(complete,sequence)
    if events>len(complete) or not allow_fixture and events==len(complete):
        raise ValueError('proper train prefix required; use the full diagnostic for a complete sequence')
    config=configuration(dict(backend=backend))
    scorer,trained=load_identity_checkpoint(checkpoint,checkpoint_sha,config=config.state)
    if (not allow_fixture and trained['full_official_train'] is not True
            or trained['frozen_cache_identity']!=frozen_cache_identity(cache)):
        raise ValueError('matching frozen train checkpoint required')
    def sources():
        return dict(diagnostic_sources(),**{Path(__file__).relative_to(ROOT).as_posix():sha_file(__file__)})
    frozen=sources();selected=complete[:events]
    plan=dict(kind='train_prefix_ranking_probe_plan_v1',backend=backend,configuration=asdict(config),
        source_sha256=frozen,cache_sha256=cache.manifest_sha256,cooperative_metadata_sha256=metadata_sha,
        identity_checkpoint_sha256=checkpoint_sha,scorer_signature=scorer.signature,identity_seed=trained['seed'],
        class_scope=['car'],cohort_mode='fixture-only' if allow_fixture else 'real-train-prefix-probe',
        complete_input_train_cohort_verified=not allow_fixture,selected_sequence=sequence,
        complete_sequence_event_count=len(complete),complete_sequence_schedule_sha256=hashlib.sha256(canonical(complete)).hexdigest(),
        selected_schedule=selected,scheduled_frames=events,complete_selected_sequence_verified=False,
        prediction_only=True,training_performed=False,ground_truth_used=False,paper_eligible=False,
        thread_environment=THREAD_ENV)
    output=Path(output)
    result=replay_rows(cache,selected,output,config,plan=plan,learned_scorer=scorer)
    if (sources()!=frozen or sha_file(metadata)!=metadata_sha
            or sha_file(Path(checkpoint)/'checkpoint.json')!=checkpoint_sha):
        raise ValueError('prefix probe sources or inputs changed; no acceptance receipt')
    final=dict(kind='train_prefix_ranking_probe_v1',status='complete',completed_prefix_events=result['completed_frames'],
        plan_sha256=sha_file(output/'plan.json'),replay_receipt_sha256=sha_file(output/'receipt.json'),
        complete_sequence_event_count=len(complete),cohort_mode=plan['cohort_mode'],
        complete_sequence_run=False,complete_selected_sequence_verified=False,
        training_performed=False,ground_truth_used=False,paper_eligible=False)
    _new(output/'probe-receipt.json',final)
    print(json.dumps(final,sort_keys=True))
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','cooperative-metadata','checkpoint','output'): parser.add_argument('--'+name,type=Path,required=True)
    for name in ('cooperative-metadata-sha256','checkpoint-sha256','sequence'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--events',type=int,required=True)
    parser.add_argument('--backend',choices=('slot_bound_joint_beam','sparse_slot_bound_joint_beam',
        'reachable_slot_bound_joint_beam'),default='slot_bound_joint_beam')
    args=parser.parse_args()
    run_probe(VerifiedForestCache(args.cache,TRAIN_CACHE_SHA256),args.cooperative_metadata,args.cooperative_metadata_sha256,
        args.checkpoint,args.checkpoint_sha256,args.output,sequence=args.sequence,events=args.events,backend=args.backend)
