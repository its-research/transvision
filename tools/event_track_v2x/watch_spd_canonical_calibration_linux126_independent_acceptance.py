#!/usr/bin/env python3
"""Accept completed existing CPU calibrations once, retaining all partial failures."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,subprocess,time,sys,os
R=Path('/Volumes/Data/test/recover-before-fuse')
DISPATCH=R/'artifacts/spd-canonical-calibration-linux126-cpu-dependency-watch-20261001'
ACCEPTOR=R/'source-freezes/spd-canonical-calibration-linux126-independent-acceptor-candidate-20261001'
CALIBRATION_PYTHON=Path('/Users/lbin/.local/share/recover-before-fuse/calibration-venv/bin/python')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 from clearml import Task
 hashes=json.loads((ACCEPTOR/'source-freeze-receipt.json').read_bytes())['inventory'];terminal=set();states={}
 def event(kind,**d):
  with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,checked_at_utc=datetime.now(timezone.utc).isoformat(),**d))+'\n')
 event('started',pid=os.getpid(),scope='independent full cloud bytes/raw-GT/examples/parameter readback')
 while len(terminal)<2:
  for fold in (0,1):
   if fold in terminal:continue
   dispatch=DISPATCH/f'fold-{fold}-dispatch.json'
   if not dispatch.exists():continue
   d=json.loads(dispatch.read_bytes());task=Task.get_task(task_id=d['task_id']);status=str(task.status)
   if states.get(fold)!=status:states[fold]=status;event('remote_status',fold=fold,task_id=task.id,status=status,eta='unknown_without_task_level_progress')
   if status in ('failed','stopped'):terminal.add(fold);event('remote_failure_preserved_no_retry',fold=fold,task_id=task.id,status=status);continue
   if status!='completed':continue
   for row in hashes:
    source=ACCEPTOR/row['path']
    if source.stat().st_size!=row['bytes'] or hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('independent acceptor closure changed')
   command=[str(CALIBRATION_PYTHON),str(ACCEPTOR/'accept_spd_canonical_calibration_linux126_cpu_candidate.py'),'--task-id',task.id,'--dispatch',str(dispatch),'--raw-readback',str(R/f'artifacts/spd-oof-runtime-v3-folds01-raw-dependency-watch-20261001/fold-{fold}/fit-raw-v2-readback/acceptance-receipt.json'),'--fit-inputs',str(R/f'artifacts/spd-single-fit-overlay-materializer-readback-20261001/job-fold-{fold}/fold-{fold}'),'--output',str(a.output/f'fold-{fold}-independent-readback')]
   event('acceptance_command_started',fold=fold,argv=command)
   with (a.output/f'fold-{fold}-acceptance.log').open('xb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
   terminal.add(fold);event('acceptance_command_terminal',fold=fold,returncode=result.returncode,failed_attempts_retried=False)
  if len(terminal)<2:time.sleep(60)
 event('monitor_scope_terminal',formal_v2_ready=False,paper_eligible=False)
if __name__=='__main__':main()
