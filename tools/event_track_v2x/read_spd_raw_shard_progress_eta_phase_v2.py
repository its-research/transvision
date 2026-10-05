"""Read parallel raw-export ETA from producer counters; never accept a result."""
import argparse,json,math,re
from pathlib import Path
from datetime import datetime,timezone
PATTERN=re.compile(r'^OOF_RAW_SHARD_PROGRESS (vehicle-side|infrastructure-side) ([01]) EVENTTRACK_CACHE_PROGRESS (.+)$')

def estimate(messages):
 samples={};phase=None
 for message in messages:
  for line in message.splitlines():
   if line.startswith("EVENTTRACK_PHASE_ETA "):
    phase=json.loads(line.split(" ",1)[1]).get("phase")
   match=PATTERN.fullmatch(line)
   if not match:continue
   side,shard,payload=match.groups();r=json.loads(payload);key=(side,int(shard))
   if r.get('side')!=side or type(r.get('shard')) is not int or r['shard']!=key[1]:raise ValueError('shard progress identity differs')
   if type(r.get('frames')) is not int or type(r.get('total')) is not int or not 0<r['frames']<=r['total']:raise ValueError('invalid frame counters')
   elapsed=r.get('elapsed_seconds')
   if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<=0:raise ValueError('invalid producer elapsed time')
   values=samples.setdefault(key,[])
   point=(r['frames'],r['total'],float(elapsed))
   if values and point==values[-1]:continue
   if values and (point[1]!=values[-1][1] or point[0]<=values[-1][0] or point[2]<=values[-1][2]):values.clear()
   values.append(point)
 result={'eta_seconds':None,'eta_status':'unknown','scope':'parallel frame production at latest producer sample; excludes publication/readback','overall_eta':'unknown','experiment_complete':False,'shards':[]}
 for key,values in sorted(samples.items()):
  last=values[-1];row={'side':key[0],'shard':key[1],'frames':last[0],'total':last[1],'producer_elapsed_seconds':last[2],'samples':len(values),'eta_seconds':None}
  if last[0]==last[1]:row['eta_seconds']=0.0
  elif len(values)>=2:
   rates=[last[0]/last[2]]
   first=values[max(0,len(values)-5)]
   if last[2]>first[2] and last[0]>first[0]:rates.append((last[0]-first[0])/(last[2]-first[2]))
   row['eta_seconds']=(last[1]-last[0])/min(rates)
  result['shards'].append(row)
 if len(result['shards'])==4 and all(r['eta_seconds'] is not None for r in result['shards']):result.update(eta_seconds=max(r['eta_seconds'] for r in result['shards']),eta_status='estimated_from_producer_elapsed')
 result["latest_observed_phase"]=phase
 if phase is not None and phase not in ("complete-heldout-raw-export","fit-feature-raw-export"):
  result.update(eta_seconds=None,eta_status="unknown_non_production_phase")
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 from clearml import Task
 t=Task.get_task(task_id=a.task_id);r=dict(estimate(t.get_reported_console_output(number_of_reports=100)),task_id=t.id,status=str(t.status),checked_at_utc=datetime.now(timezone.utc).isoformat())
 if str(t.status)!='in_progress':r.update(eta_seconds=None,eta_status='unknown_terminal_requires_acceptance')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(r,allow_nan=False))
if __name__=='__main__':main()
