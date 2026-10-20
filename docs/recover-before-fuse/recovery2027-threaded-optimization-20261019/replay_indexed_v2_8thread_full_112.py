"""All 452 real events with identical admitted reference factors in both arms.
This validates optimization parity, not GPU model forward or full 46-sequence acceptance.
"""
import pathlib,sys,types,json,hashlib,time,os,shutil,threading,datetime
ROOT=pathlib.Path('/runtime')
# Verify the complete frozen/candidate package before importing executable code.
manifest=json.loads(pathlib.Path('/inputs/runtime-package-manifest.json').read_text())
for relative,expected_sha in manifest['files'].items():
 p=pathlib.Path(relative)
 assert not p.is_absolute() and '..' not in p.parts
 assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==expected_sha,relative
assert hashlib.sha256(pathlib.Path('/inputs/real2027-plan.json').read_bytes()).hexdigest()=='8b7c6a72a33e20f5192941cddfd9286a12f0b2f42f159661cfbbfcadc8a491ec'
assert hashlib.sha256(pathlib.Path('/context/context-validation.json').read_bytes()).hexdigest()=='eda6b81af53f9899ac808ea3a27cf12b716cdfdc5093e26dc925f18d1ca526b1'
for name,rel in [('transvision','transvision'),('transvision.models','transvision/models'),('transvision.models.event_track_v2x','transvision/models/event_track_v2x')]:
 m=types.ModuleType(name);m.__path__=[str(ROOT/rel)];sys.modules[name]=m
import numpy as np
from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection,PaperForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import IdentityNode
from transvision.models.event_track_v2x.recovery_off_tracking import RecoveryOffConfig,RecoveryOffTracker

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
raw=pathlib.Path('/raw/raw-observations.jsonl');assert sha(raw)=='1caf3ae225e490dcfc4bc59bb60dcad0c30f37f17469cb711f1f894748d72300'
context=json.loads(pathlib.Path('/context/context-validation.json').read_text());assert context['passed'] and context['rows']==21278 and context['context_indices_exact']
ep=pathlib.Path('/inputs/real2027-inputs/sequence0002-events.json');fp=pathlib.Path('/inputs/real2027-inputs/sequence0002-frozen-logits.json')
assert sha(ep)=='3c2db2445be9c26c9f9c657731278e025d840937df40e86b21709ffd3f03cb13';assert sha(fp)=='b2a6a3833d730c5feb8bdc7dc3e5705f6fba63a38858a683acfdb93a53b28b68'
events=json.loads(ep.read_text());factors=json.loads(fp.read_text());plan=json.loads(pathlib.Path('/inputs/real2027-plan.json').read_text());config=RecoveryOffConfig(state=PaperForestTrackingConfig(**plan['configuration']['state']),**plan['configuration']['limits'])
assert len(events)==452 and len(factors)==21278
groups={}
for line in raw.open():
 r=json.loads(line);v=r['observation'];v['node']=IdentityNode(**v['node']);obs=RawIdentityDetection(**v);f=factors[r['row']];assert obs.node.node_id==f['node_id']
 a=np.asarray(f['logits'],dtype=np.float64);m=float(a.max());w=a-(m+float(np.log(np.exp(a-m).sum())))
 parents=[-1]+f['context_indices'][:-1];groups.setdefault(r['event_index'],[]).append((obs,tuple(zip(parents,map(float,w)))))

import sqlite3
from incremental_projection_candidate import projected_kernel_choice
from transvision.models.event_track_v2x.recovery_off_tracking import HistoryRestriction
HistoryRestriction.kernel_choice=projected_kernel_choice
OUT=pathlib.Path('/output')
from prepared_threaded_factors_v2_candidate import PreparedThreadedFactors, METRICS
from transvision.models.event_track_v2x import threaded_action_adapter
threaded_action_adapter.wrap_action_factors=lambda factors:PreparedThreadedFactors(factors,'/native/native.so',8)
from indexed_observation_slots_candidate import install
from transvision.models.event_track_v2x.persistent_component_store import _Namespace
install(_Namespace)
verified=0
unverified=[]
tracker=RecoveryOffTracker(OUT/'candidate.sqlite',sequence_id='0002',config=config)
state={'stage':'starting','completed_events':0,'total_events':452,'ETA_seconds':None,'ETA_reason':'unknown until real expensive events are measured','native_threads':8,'allowed_CPUs':sorted(os.sched_getaffinity(0))}
lock=threading.Lock();done=threading.Event()
def emit(**kw):
 with lock:
  state.update(kw);r=dict(state,at=datetime.datetime.now(datetime.timezone.utc).isoformat())
  with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(r)+'\n');f.flush()
  print(json.dumps(r),flush=True)
def heartbeat():
 while not done.wait(60):emit(observation='heartbeat only, not internal progress')
threading.Thread(target=heartbeat,daemon=True).start()
try:
 for index,e in enumerate(events):
  assert shutil.disk_usage(OUT).free>8*1024**3,'disk floor: retain partial evidence'
  db=sqlite3.connect('file:/baseline/baseline.sqlite?mode=ro',uri=True,timeout=5)
  expected=db.execute('SELECT prediction,audit FROM events WHERE event_id=?',(e['event_id'],)).fetchone();db.close()
  if index<=313: assert expected is not None,('baseline event missing',index)
  group=groups.get(index,[])
  emit(observation='actual event entry',stage='event_begin',event_index=index,event_id=e['event_id'])
  start=time.monotonic();cpu=time.process_time()
  result=tracker.step([x[0] for x in group],[x[1] for x in group],**{k:e[k] for k in ('frame_id','reference_us','decision_us','event_id')})
  wall=time.monotonic()-start;used=time.process_time()-cpu
  if expected is not None:
   assert result.prediction_json==expected[0],('prediction mismatch',index)
   assert result.audit_json==expected[1],('audit mismatch',index)
   verified+=1
  else: unverified.append(index)
  emit(observation='actual event returned',baseline_verified=expected is not None,verified_events=verified,native_metrics=dict(METRICS),stage='event_verified' if expected is not None else 'event_completed_unverified',completed_events=index+1,event_seconds=wall,process_CPU_seconds=used,CPU_wall_ratio=used/wall,prediction_sha256=hashlib.sha256(result.prediction_json).hexdigest(),audit_sha256=hashlib.sha256(result.audit_json).hexdigest())
 database_sha=tracker.close();tracker=None
 report=dict(execution_completed=True,parity_passed=not unverified,verified_events=verified,unverified_event_indices=unverified,events=452,includes_event313=True,database_sha256=database_sha,full_452_events_qualified=not unverified,full_scientific_acceptance=False,native_threads=8)
 (OUT/'result.json').write_text(json.dumps(report,indent=2));emit(stage='completed')
except BaseException as error:
 (OUT/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error),state=state),indent=2));raise
finally:
 done.set()
 if tracker is not None:tracker.close()
