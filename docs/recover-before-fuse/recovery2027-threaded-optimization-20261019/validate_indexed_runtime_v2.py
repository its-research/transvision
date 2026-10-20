"""Frozen full tracker baseline/candidate event parity, not real-cohort acceptance."""
import hashlib,json,os,pathlib,sys,types
ROOT=pathlib.Path('/runtime')
# Match registered bootstrap package loading; avoid unrelated top-level imports.
for name,rel in [('transvision','transvision'),('transvision.models','transvision/models'),('transvision.models.event_track_v2x','transvision/models/event_track_v2x')]:
 m=types.ModuleType(name);m.__path__=[str(ROOT/rel)];sys.modules[name]=m
import numpy as np
from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig,RawIdentityDetection
from transvision.models.event_track_v2x.identity_forest import IdentityNode
from transvision.models.event_track_v2x.recovery_off_tracking import RecoveryOffConfig,RecoveryOffTracker
from transvision.models.event_track_v2x.recovery_off_tracking import HistoryRestriction
from incremental_projection_candidate import projected_kernel_choice
from prepared_threaded_factors_v2_candidate import PreparedThreadedFactors
from transvision.models.event_track_v2x import threaded_action_adapter
from indexed_observation_slots_candidate import install
from transvision.models.event_track_v2x.persistent_component_store import _Namespace
original_execute=_Namespace.execute
calls=0
original=HistoryRestriction.kernel_choice
def counted(self,kernel,handle,root):
 global calls
 calls+=1
 return projected_kernel_choice(self,kernel,handle,root)

def obs(name,source=0,time=1000000,arrival=None):
 features=np.zeros(203);features[138],features[200],features[201]=1,.8,.8
 return RawIdentityDetection('0003',IdentityNode(name,source,time,time+10 if arrival is None else arrival,name),0,time,[0.,0.,1.,4.,2.,1.5,0.,0.,0.],np.eye(9)*.2,.8,features,'a'*64)
checks=0
for threads in (4,8):
 for width in (1,4):
  for budget in (0,100):
   config=RecoveryOffConfig(state=PaperForestTrackingConfig(active_limit=width,expansion_budget=budget,decision_mode='all-legal-hamming',max_model_regret=1.))
   paths=[pathlib.Path('/output')/f't{threads}-w{width}-b{budget}-{kind}.sqlite' for kind in ('base','candidate')]
   trackers=[RecoveryOffTracker(p,sequence_id='0003',config=config) for p in paths]
   events=[([obs('a'),obs('separate',1),obs('c',time=1010000)],[((-1,0.),),((-1,0.),),((-1,0.),(0,-10.))],1100000,{}),([obs('bridge',1,900000,1200000)],[((-1,-3.),(1,1.),(2,1.))],1300000,{'rescored_rows':[(2,((-1,-20.),(0,20.)))]}),([],[],1400000,{})]
   try:
    for ordinal,(raw,rows,clock,kwargs) in enumerate(events):
     results=[]
     for mode,t in enumerate(trackers):
      for key in ('RBF_THREADED_ACTION_LIBRARY','RBF_THREADED_ACTION_THREADS'):os.environ.pop(key,None)
      HistoryRestriction.kernel_choice=counted if mode else original
      _Namespace.execute=original_execute
      if mode:install(_Namespace)
      threaded_action_adapter.wrap_action_factors=(lambda factors: PreparedThreadedFactors(factors,'/native/native.so',threads)) if mode else (lambda factors:factors)
      results.append(t.step(raw,rows,frame_id=str(ordinal),event_id=str(ordinal),reference_us=clock,decision_us=clock,**kwargs))
     assert results[0].prediction_json==results[1].prediction_json
     assert results[0].audit==results[1].audit
     checks+=1
     if ordinal==1:
      # Exercise the real immutable database restore after component merge.
      for mode,t in enumerate(trackers):
       prediction_sha=results[mode].prediction['commit_sha256'];database_sha=t.close()
       trackers[mode]=RecoveryOffTracker.open(paths[mode],expected_prediction_sha256=prediction_sha,expected_database_sha256=database_sha)
   finally:
    for t in trackers:t.close()
assert calls>0
assert checks==24
result={'passed':True,'paired_events':checks,'projection_calls':calls,'close_reopen_after_merge':True,'scope':'isolated history-projection candidate: full tracker merge/rescore/close-reopen/empty-event parity; indexed slot lookup and prepared native adapter 4 and 8 threads','real_seed2027_qualified':False,'full_replay_qualified':False}
pathlib.Path('/output/runtime-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
