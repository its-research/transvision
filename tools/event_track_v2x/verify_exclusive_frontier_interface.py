"""Synthetic CPU runtime matrix for the separately named exclusive backend.

Exports raw finite cases, actual commits and SQLite histories for independent
readback. No real dataset, model training, metric or full Stage2 claim.
"""
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from transvision.models.event_track_v2x.exclusive_completion_tracking import (
    ExclusiveCompletionTracker, ExclusiveCompletionTeacher, ExclusiveCompletionLearned,
    PersistentExclusiveCompletionConfig,
)
from transvision.models.event_track_v2x.allocation_policy import FrozenPriorityPolicy, FEATURES
from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection, PaperForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import IdentityNode
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file


def scene(seed, regime):
    rng = np.random.default_rng(seed)
    raw=[]
    for i in range(5):
        source = (0,0,1,1,0)[i]
        state = (1_000_000,1_000_000,1_200_000,1_200_000,1_400_000)[i]
        arrival = state
        if regime == 'late' and i == 2:
            state,arrival = 900_000,1_200_000
        feature=np.zeros(203);feature[138+i%3]=1.;feature[200]=feature[201]=.8
        same = regime == 'conflicts'
        frame=('a' if i<2 else 'b' if i<4 else 'c') if same else str(i)
        raw.append(RawIdentityDetection('fixture',IdentityNode(str(i),source,state,arrival,frame),
                  i%2,state,[i*.05,0.,1.,4.,2.,1.5,0.,0.,0.],np.eye(9)*.2,.8,feature,'a'*64))
    support = [(-1,),(-1,0),(-1,0,1),(-1,0,1,2),(-1,0,1,2,3)]
    if regime == 'disconnected':
        support=[(-1,),(-1,),(-1,0),(-1,1),(-1,0,2)]
    rows=[]
    for options in support:
        rows.append(tuple((p,float(rng.normal()* (30. if regime=='extreme' else 1.))) for p in options))
    return tuple(raw),tuple(rows)


def handle_path(kernel,h):
    path=[]
    while h:
        p=kernel._prefix(h);path.append(p.choice);h=p.parent_handle
    return list(reversed(path))


def enrich(tracker,commit):
    value={'audit':commit.audit,'prediction':commit.prediction}
    paths={}
    for s in value['audit']['components']:
        c=s['component'];k=tracker.kernels[c]
        handles={s['output_handle'],s['fallback_handle']}
        handles.update(r['handle'] for r in s['active'])
        handles.update(r['handle'] for r in s['frontier'])
        handles.update(e['handle'] for r in s['frontier'] for e in r['excluded_leaves'])
        paths[str(c)]={'members':list(tracker.store.members(c)),
                       'handles':{str(h):handle_path(k,h) for h in sorted(handles)}}
    value['independent_path_exports']=paths
    return value


def run_case(root,seed,regime,width,budget,variant):
    raw,rows=scene(seed,regime)
    cfg=PersistentExclusiveCompletionConfig(state=PaperForestTrackingConfig(
        active_limit=width,expansion_budget=budget,max_frontier=4,
        decision_mode='all-legal-hamming',max_model_regret=1.),max_total_frontier=16)
    cls={'bound':ExclusiveCompletionTracker,'teacher':ExclusiveCompletionTeacher,
         'learned':ExclusiveCompletionLearned}[variant]
    options={}
    if variant=='learned':
        options['allocation_policy']=FrozenPriorityPolicy((np.zeros((4,len(FEATURES))),np.zeros(4),np.zeros((1,4)),np.zeros(1)))
    key=f'{variant}-{seed}-{regime}-K{width}-B{budget}'
    path=root/(key+'.sqlite')
    t=cls(path,sequence_id='fixture',config=cfg,**options)
    snapshots=[];first=None
    groups=((0,2,1_100_000),(2,4,1_300_000),(4,5,1_500_000),(5,5,1_700_000))
    try:
        for ordinal,(a,b,clock) in enumerate(groups):
            commit=t.step(raw[a:b],rows[a:b],frame_id=str(ordinal),event_id=str(ordinal),reference_us=clock,decision_us=clock)
            first=first or commit
            snapshots.append(enrich(t,commit))
            assert commit.audit['search_steps']<=budget
            assert commit.audit['residual_exclusion_references']<=len(commit.audit['components'])*width
            assert commit.audit['residual_regions_counted_in_database_storage'] is True
        duplicate=t.step(raw[:2],rows[:2],frame_id='0',event_id='0',reference_us=1_100_000,decision_us=1_100_000)
        assert duplicate.prediction_json==first.prediction_json and duplicate.audit_json==first.audit_json
        last=snapshots[-1]['prediction']['commit_sha256']
        sealed=t.close()
        t=cls.open(path,expected_prediction_sha256=last,expected_database_sha256=sealed,**options)
        assert t.meta['events']==4
        duplicate=t.step(raw[:2],rows[:2],frame_id='0',event_id='0',reference_us=1_100_000,decision_us=1_100_000)
        assert duplicate.prediction_json==first.prediction_json and duplicate.audit_json==first.audit_json
        fifth=t.step((),(),frame_id='4',event_id='4',reference_us=1_900_000,decision_us=1_900_000)
        snapshots.append(enrich(t,fifth))
        final=t.close();t=None
        return dict(key=key,seed=seed,regime=regime,width=width,budget=budget,variant=variant,
                    configuration=asdict(cfg),raw=[asdict(r) for r in raw],rows=rows,snapshots=snapshots,
                    database=dict(path=path.name,sha256=final,bytes=path.stat().st_size),
                    duplicate_and_reopen_immutable_commits_verified=True,trained_priority_policy=False)
    finally:
        if t is not None:t.close()


def run(root, progress=lambda x:None):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    cases=list(itertools.product((1337,2027,3407),('conflicts','disconnected','late','extreme'),(1,4),(0,1,4,16),('bound','teacher','learned')))
    result=[];started=time.monotonic()
    for seed,regime,width,budget,variant in cases:
        result.append(run_case(root,seed,regime,width,budget,variant))
        done=len(result);eta=(time.monotonic()-started)*(len(cases)-done)/done
        progress(dict(stage='exclusive_frontier_cpu_interface_matrix',completed_cases=done,total_cases=len(cases),
                      ETA_seconds=eta,ETA_scope='finite interface matrix production only; independent acceptance excluded'))
    return dict(kind='rbf_exclusive_frontier_cpu_interface_matrix_v1',cases=result,case_count=len(result),
                completed_events=sum(len(c['snapshots']) for c in result),device='cpu',dataset_read=False,
                full_stage_two_complete=False,complete_online_method_accepted=False,trained_priority_policy=False,
                same_resource_performance_comparison=False,paper_performance_complete=False,
                elapsed_seconds=time.monotonic()-started)
