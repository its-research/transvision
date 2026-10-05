#!/usr/bin/env python3
"""Deduplicate actual fold0/1 CPU calibration; require complete fit raw admission."""
import argparse,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
QUEUE='8d0f8b54037249eeb0f1cc70cbfe73ab'
PROJECT='Thesis/EventTrack-V2X/Training'
IMAGE='gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb'
RUNNER_SHA='f8b9fb992d0830a1a890c70f2a05ea80c7044a93ac2a4a13d41a76b8b1424455'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
def admit(receipt,fold):
 need(not receipt.is_symlink(),'fit receipt is symlinked');r=json.loads(receipt.read_bytes())
 need(r.get('kind')=='spd_canonical_oof_fit_feature_raw_pose_v2_cloud_independent_content_readback' and r.get('status')=='all_cloud_bytes_frames_arrays_raw_poses_verified' and r.get('fold_id')==fold and r.get('seed')==1337 and r.get('source_task_id')=='5d218685995548948ccb37b92dd337de' and r.get('acceptor_sha256')=='fefd651a33f6dabf56ea0612eebc80673af5e5d50b7466d0e928644c7569f878' and r.get('held_out_selection_scoring_eligible') is False,'complete matching fit raw admission missing')
 from clearml import Task
 t=Task.get_task(task_id=r['task_id']);need(t.status=='completed' and set(t.artifacts)==set(r['artifacts']),'fit export scope/status differs')
 for name,row in r['artifacts'].items():
  relative=Path(row['path']);need(not relative.is_absolute() and '..' not in relative.parts,'unsafe fit artifact path');p=receipt.parent/relative
  need(not p.is_symlink() and p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'] and t.artifacts[name].hash==row['sha256'] and t.artifacts[name].size==row['bytes'],'fit cloud/local artifact changed')
 return r
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',type=int,choices=(0,1),required=True);p.add_argument('--raw-readback',type=Path,required=True);p.add_argument('--bootstrap',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
 need(not a.receipt.exists(),'dispatch receipt exists; inspect existing task');r=admit(a.raw_readback,a.fold)
 frozen=Path('/Volumes/Data/test/recover-before-fuse/source-freezes/spd-canonical-calibration-linux126-cloud-execution-candidate-20261001');need(a.bootstrap.read_bytes()==(frozen/a.bootstrap.name).read_bytes() and Path(__file__).read_bytes()==(frozen/Path(__file__).name).read_bytes(),'CPU dispatch/bootstrap differ from freeze')
 from clearml import Task
 from clearml.backend_api.session.client import APIClient
 for tid,expected in [('641448cbbb83421ab6c63b8537c3bf99','2d01f5d0bfc94c4837eedd574ef6b2176f8211dc428352b49d27573a8533c517'),('c5ca5855aed84f6e83ceac022578f5ed','907e5a0986daa4c46a747d0f7344d5a5f3edf0f7dd1ed718fb37c4710ae28492')]:
  t=Task.get_task(task_id=tid);need(t.status=='completed' and t.artifacts['source-manifest'].hash==expected,'source/metadata registration differs')
 probe=Task.get_task(task_id='969e56c034714dfab10e16471d68d7e1');need(probe.status=='completed' and probe.artifacts['runtime-probe'].hash=='2fc885c76f46de4ba7756aec537daae960dc2ca043802ac237e1bce77395b62a','runtime probe registration differs')
 name=f'SPD canonical OOF fold-{a.fold} seed1337 calibration Linux126 CPU '+sha(a.raw_readback)[:12]+' '+sha(a.bootstrap)[:12];matches=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
 if matches:
  need(len(matches)==1,'ambiguous matching calibration tasks');t=matches[0];need(hashlib.sha256(t.data.script.diff.encode()).hexdigest()==sha(a.bootstrap) and t.get_parameters()['General/raw_fit_readback_sha256']==sha(a.raw_readback),'existing calibration identity differs');result=dict(task_id=t.id,status=str(t.status),duplicate_not_created=True)
  if a.check_only:print(json.dumps(dict(result,experiment_started=False)));return
 else:
  api=APIClient();q=api.queues.get_by_id(queue=QUEUE);need(q.name=='GPU3-L40S' and not q.entries,'L40S CPU queue unavailable/pending')
  now=datetime.now(timezone.utc);workers=[w.to_dict() for w in api.workers.get_all()];eligible=[]
  for w in workers:
   stamp=w.get('last_report_time');stamp=datetime.fromisoformat(str(stamp)) if stamp else None
   if 'L40S' in w['id'] and not w.get('task',{}).get('id') and any(x['id']==QUEUE for x in w.get('queues',[])) and stamp and now-stamp<timedelta(seconds=120):eligible.append(w['id'])
  need(eligible,'no fresh idle L40S CPU listener')
  if a.check_only:print(json.dumps(dict(status='ready_to_dispatch',fold=a.fold,workers=eligible,raw_fit_task_id=r['task_id'],experiment_started=False)));return
  # Broad search prevents a changed bootstrap label from silently repeating a fit.
  for t in Task.get_tasks(project_name=PROJECT,task_name='^SPD canonical OOF fold-'+str(a.fold)+' seed1337 calibration Linux126 CPU '):
   need(t.get_parameters().get('General/raw_fit_task_id')!=r['task_id'],'another calibration attempt for this exact fit export exists; inspect it')
  t=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.training,binary='python3.12');t.set_script(repository='',branch='',commit='',working_dir='.',entry_point=a.bootstrap.name,diff=a.bootstrap.read_text());t.set_packages(['clearml==2.1.5','numpy==1.26.4','scipy==1.14.1'])
  t.set_base_docker(IMAGE,docker_arguments='-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 4g --env CUDA_VISIBLE_DEVICES= --env CLEARML_FILES_HOST=http://10.100.35.118:8081 --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 --env PYTHONHASHSEED=0 --env PYTHONDONTWRITEBYTECODE=1')
  t.set_parameters(dict(fold_id=a.fold,seed=1337,raw_fit_task_id=r['task_id'],raw_fit_readback_sha256=sha(a.raw_readback),raw_fit_readback_text=a.raw_readback.read_text(),runner_sha256=RUNNER_SHA,bootstrap_sha256=sha(a.bootstrap),device='cpu',held_out_GT_used_for_fitting=False,val_test_read=False,formal_v2_ready=False,paper_eligible=False));t.add_tags(['Recover-Before-Fuse','SPD','canonical-calibration','fit-only','Linux126','L40S-CPU',f'fold-{a.fold}'])
  t.reload();need(hashlib.sha256(t.data.script.diff.encode()).hexdigest()==sha(a.bootstrap) and t.get_parameters()['General/raw_fit_readback_sha256']==sha(a.raw_readback),'new task source/parameters differ')
  # Persist the handle before enqueue so an interrupted observation cannot duplicate it.
  a.receipt.parent.mkdir(parents=True,exist_ok=True)
  with a.receipt.open('x') as f:json.dump(dict(kind='canonical_calibration_linux126_cpu_dispatch',task_id=t.id,fold_id=a.fold,raw_fit_task_id=r['task_id'],raw_fit_readback_sha256=sha(a.raw_readback),bootstrap_sha256=sha(a.bootstrap),eligible_workers=eligible,enqueue_confirmed=False),f,indent=2);f.write('\n')
  Task.enqueue(t,queue_id=QUEUE);result=dict(task_id=t.id,status=str(t.status),duplicate_not_created=False,enqueue_confirmed=True)
 out=a.receipt.with_name(a.receipt.stem+'-enqueue-acceptance.json')
 with out.open('x') as f:json.dump(dict(result,checked_at_utc=datetime.now(timezone.utc).isoformat(),formal_v2_ready=False,paper_eligible=False),f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
