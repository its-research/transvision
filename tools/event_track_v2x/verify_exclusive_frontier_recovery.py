"""Causal finite recovery scenario, including a branch initially outside Top-1.

The target identity partition is evaluated offline. The online tracker receives
only the current arrivals and their finite raw potentials, never the target.
"""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np

from tools.event_track_v2x.verify_exclusive_frontier_interface import enrich
from transvision.models.event_track_v2x.exclusive_completion_tracking import (
    ExclusiveCompletionTracker, ExclusiveCompletionTeacher, ExclusiveCompletionLearned,
    PersistentExclusiveCompletionConfig,
)
from transvision.models.event_track_v2x.allocation_policy import FrozenPriorityPolicy, FEATURES
from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection, PaperForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import IdentityNode


def inputs():
    raw=[]
    for i in range(4):
        source=1 if i==1 else 0
        state=1_000_000 if i<2 else 1_200_000
        frame=('a','b','c','c')[i]
        f=np.zeros(203);f[138]=1.;f[200]=f[201]=.8
        raw.append(RawIdentityDetection('fixture',IdentityNode(str(i),source,state,state,frame),
                   1 if i==3 else 0,state,[float(i%2),0.,1.,4.,2.,1.5,0.,0.,0.],np.eye(9)*.2,.8,f,'a'*64))
    # Initially joining cross-source observations is more likely. Later two
    # same-frame detections support their distinct old identities.
    rows= (((-1,0.),),((-1,-2.),(0,0.)),
           ((-1,-20.),(0,0.),(1,-20.)),((-1,-20.),(0,-20.),(1,0.)))
    return tuple(raw),rows


def run(root,progress=lambda x:None):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    raw,rows=inputs();result=[];started=time.monotonic()
    for variant in ('bound','teacher','learned'):
        for budget in (4,8,16):
            key=f'{variant}-causal-return-K1-B{budget}'
            cfg=PersistentExclusiveCompletionConfig(state=PaperForestTrackingConfig(
                active_limit=1,expansion_budget=budget,max_frontier=4,
                decision_mode='all-legal-hamming',max_model_regret=1.),max_total_frontier=16)
            cls={'bound':ExclusiveCompletionTracker,'teacher':ExclusiveCompletionTeacher,
                 'learned':ExclusiveCompletionLearned}[variant]
            options={}
            if variant=='learned':options['allocation_policy']=FrozenPriorityPolicy((np.zeros((4,len(FEATURES))),np.zeros(4),np.zeros((1,4)),np.zeros(1)))
            path=root/(key+'.sqlite');t=cls(path,sequence_id='fixture',config=cfg,**options)
            snapshots=[]
            try:
                for event in range(12):
                    a,b=(0,2) if event==0 else (2,4) if event==1 else (4,4)
                    clock=1_100_000 if event==0 else 1_200_000+event*100_000
                    commit=t.step(raw[a:b],rows[a:b],frame_id=str(event),event_id=str(event),reference_us=clock,decision_us=clock)
                    snapshots.append(enrich(t,commit))
                    assert commit.audit['search_steps']<=budget
                sealed=t.close();t=None
                result.append(dict(key=key,budget=budget,width=1,variant=variant,configuration=asdict(cfg),
                    raw=[asdict(r) for r in raw],rows=rows,snapshots=snapshots,
                    database=dict(path=path.name,sha256=sealed,bytes=path.stat().st_size),
                    declared_target_roots=[0,1,0,1],target_never_passed_to_online_tracker=True,
                    trained_priority_policy=False,physical_identity_or_tracking_metric_claim=False))
            finally:
                if t is not None:t.close()
            done=len(result)
            progress(dict(stage='exclusive_frontier_causal_recovery_cpu',completed_cases=done,total_cases=9,
                          ETA_seconds=(time.monotonic()-started)*(9-done)/done,
                          ETA_scope='finite production scenarios only; independent acceptance excluded'))
    return dict(kind='rbf_exclusive_frontier_causal_recovery_CPU_v1',cases=result,case_count=9,
                completed_events=108,device='cpu',dataset_read=False,full_stage_two_complete=False,
                complete_online_method_accepted=False,trained_priority_policy=False,
                same_resource_performance_comparison=False,paper_performance_complete=False,
                elapsed_seconds=time.monotonic()-started)
