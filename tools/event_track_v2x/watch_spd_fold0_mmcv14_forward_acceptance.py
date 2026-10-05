"""Wait on the existing corrected fold0 task and independently accept it once.

No task creation, retries, training, raw inference or paper qualification.
"""
import json,hashlib,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
TASK='e5a7474c7cd34891a5aacc8b7bd2de8d'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=Path(__file__).parent
 inventory=json.loads((source/'source-freeze-receipt.json').read_text())['inventory']
 def verify():
  for r in inventory:
   if sha(source/r['name'])!=r['sha256']:raise ValueError('immutable acceptor monitor changed')
 verify()
 from clearml import Task
 out=ROOT/'artifacts/spd-fold0-mmcv14-forward-acceptance-watch-20261001';out.mkdir(exist_ok=False)
 (out/'process.json').write_text(json.dumps({'pid':os.getpid(),'task_id':TASK,'argv':sys.argv,'source_freeze_sha256':sha(source/'source-freeze-receipt.json'),'automatic_retry':False})+'\n')
 def event(kind,**fields):
  row=dict(kind=kind,checked_at_utc=datetime.now(timezone.utc).isoformat(),**fields)
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 last=None
 while True:
  verify();task=Task.get_task(task_id=TASK);p=task.get_parameters()
  if p.get('General/source_task_id')!='2652410a467f49d982115537a3ceca5a' or p.get('General/verifier_sha256')!='d7bc1571454521b94bf677a2f5e22acd95734bd08c58ee4154ffa244924fb1d3':raise ValueError('existing task identity changed')
  state=(str(task.status),tuple(sorted(task.artifacts)))
  if state!=last:event('live_existing_task',task_id=TASK,status=state[0],artifacts=list(state[1]),eta='unknown_without_task_level_progress');last=state
  if state[0]=='completed':
   cmd=[sys.executable,str(source/'accept_spd_oof_checkpoint_forward_gpu4_mmcv14_iter_v2.py'),'--task-id',TASK,'--byte-freeze',str(ROOT/'artifacts/spd-oof-completed-byte-freeze-watch-20261001/fold-0-byte-freeze/acceptance-receipt.json'),'--package',str(ROOT/'artifacts/spd-official-oof-fivefold-20260930/fold-0-package'),'--heldout-inputs',str(ROOT/'artifacts/spd-canonical-oof-heldout-inference-inputs-20260930/fold-0'),'--output',str(out/'independent-readback')]
   event('acceptance_command_started',argv=cmd);z=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
   safe=[x for x in z.stdout.splitlines() if x.startswith('SAMPLED_FORWARD_INDEPENDENTLY_ACCEPTED')]
   event('acceptance_terminal',returncode=z.returncode,accepted_markers=safe,automatic_retry=False)
   if z.returncode:raise RuntimeError('independent acceptance failed; preserve output and inspect')
   receipt=out/'independent-readback/acceptance-receipt.json';r=json.loads(receipt.read_text())
   if r['task_id']!=TASK or r['status']!='independent_bytes_and_sampled_tensor_forward_verified':raise ValueError('unexpected acceptance')
   event('independent_forward_accepted',receipt=str(receipt),sha256=sha(receipt),paper_eligible=False);return
  if state[0] in ('failed','stopped','aborted','closed'):
   event('terminal_requires_inspection',status=state[0],automatic_retry=False);return
  time.sleep(60)
if __name__=='__main__':main()
