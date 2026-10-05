#!/usr/bin/env python3
"""Wait for existing full fit raw acceptance, dispatch CPU once, retain failures.

Scope ends at remote task completion/failure. No cloud output acceptance or V2
claim is made; a separate independent readback must follow completed jobs.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,json,hashlib,subprocess,sys,time,os
R=Path('/Volumes/Data/test/recover-before-fuse')
F=R/'source-freezes/spd-canonical-calibration-linux126-cloud-execution-candidate-20261001'
RAW=R/'artifacts/spd-oof-runtime-v3-folds01-raw-dependency-watch-20261001'
QUEUE='8d0f8b54037249eeb0f1cc70cbfe73ab'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 from clearml import Task
 from clearml.backend_api.session.client import APIClient
 states={};terminal=set();hashes=json.loads((F/'source-freeze-receipt.json').read_text())['inventory']
 def event(kind,**d):
  with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,checked_at_utc=datetime.now(timezone.utc).isoformat(),**d))+'\n')
 event('started',pid=os.getpid(),scope='dispatch_and_remote_status_only_no_output_acceptance')
 while len(terminal)<2:
  for fold in (0,1):
   if fold in terminal:continue
   receipt=a.output/f'fold-{fold}-dispatch.json'
   if receipt.exists():
    d=json.loads(receipt.read_text());task=Task.get_task(task_id=d['task_id']);status=str(task.status)
    if states.get(fold)!=status:event('remote_status',fold=fold,task_id=task.id,status=status,eta='unknown_until_task_level_progress');states[fold]=status
    if status in ('completed','failed','stopped'):
     terminal.add(fold);event('remote_terminal_not_independent_acceptance',fold=fold,task_id=task.id,status=status)
    continue
   raw=RAW/f'fold-{fold}/fit-raw-v2-readback/acceptance-receipt.json'
   if not raw.exists():continue
   # Do not enqueue while CPU queue has pending work or no fresh idle listener.
   api=APIClient();q=api.queues.get_by_id(queue=QUEUE);workers=[w.to_dict() for w in api.workers.get_all()];now=datetime.now(timezone.utc)
   eligible=[w['id'] for w in workers if 'L40S' in w['id'] and not w.get('task',{}).get('id') and any(x['id']==QUEUE for x in w.get('queues',[])) and w.get('last_report_time') and now-datetime.fromisoformat(str(w['last_report_time']))<timedelta(seconds=120)]
   if q.entries or not eligible:continue
   for row in hashes:
    path=F/row['path']
    if path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('frozen CPU dispatcher/bootstrap changed')
   command=[sys.executable,str(F/'submit_spd_canonical_calibration_linux126_cpu_candidate.py'),'--fold',str(fold),'--raw-readback',str(raw),'--bootstrap',str(F/'bootstrap_spd_canonical_calibration_linux126_cpu_candidate.py'),'--receipt',str(receipt)]
   event('dispatch_command_started',fold=fold,argv=command)
   with (a.output/f'fold-{fold}-dispatch.log').open('xb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
   event('dispatch_command_terminal',fold=fold,returncode=result.returncode)
   if result.returncode:
    terminal.add(fold);event('dispatch_failure_preserved_no_retry',fold=fold)
  if len(terminal)<2:time.sleep(60)
 event('monitor_scope_finished',independent_calibration_outputs_verified=False,formal_v2_ready=False)
if __name__=='__main__':main()
