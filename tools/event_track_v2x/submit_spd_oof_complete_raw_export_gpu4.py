"""Dispatch full raw held-out inference only after independent forward acceptance."""
import argparse, hashlib, json, re
from pathlib import Path
from datetime import datetime, timezone
from spd_canonical_oof_export_binding import validate_binding
from submit_spd_official_oof_fold_gpu4_offline_gl_nvml_v7 import DOCKER
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
PROJECT='Thesis/EventTrack-V2X/Training'
SOURCE_TASK='fc3d2af1feb849eea4aa2222ba15dc5b'
SOURCE_MANIFEST='d3901dc04fc3606c5441a939f40fa200bf169ede5d7889fb2f011b2d66587fd7'
SOURCE_READBACK='bf6ea32a77ac930a3f8a7ede6ffa23bf1b1178cdcd814f1fa9cf0c8013c234f3'
CLOUD_SHA='eca6277da199e198fbcf07b8098421787c8eca08136c67e05761497106cdada1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def physical(worker):
 m=re.fullmatch(r'([^:]+):gpu([0-9]+(?:,[0-9]+)*)',worker['id'])
 need(m is not None,'unknown GPU worker binding')
 devices=[int(x) for x in m.group(2).split(',')];need(len(set(devices))==len(devices),'duplicate GPU indices')
 return worker.get('ip') or m.group(1),set(devices)
def safe_workers(workers,queue_id,now):
 eligible=[]
 for w in workers:
  if not any(q['id']==queue_id for q in w.get('queues',[])) or w.get('task',{}).get('id'):continue
  host,devices=physical(w);need(len(devices)==4,'idle queue listener is not four GPUs')
  stamp=datetime.fromisoformat(str(w['last_activity_time']).replace('Z','+00:00'))
  need(0<=(now-stamp).total_seconds()<90,'idle queue listener heartbeat is stale')
  for other in workers:
   if not other.get('task',{}).get('id'):continue
   if other.get('ip')!=host and other['id'].split(':',1)[0]!=w['id'].split(':',1)[0]:continue
   _,busy=physical(other);need(not devices.intersection(busy),'idle queue listener overlaps active physical GPUs')
  eligible.append(w['id'])
 need(bool(eligible),'no idle four-GPU listener')
 return eligible

def completed_evidence(task,freeze,manifest):
 need(task.status=='completed','original two-side training is not completed')
 need(freeze['kind']=='spd_canonical_oof_detector_independent_byte_freeze_v1' and freeze['byte_freeze_accepted'] is True,'byte freeze not admitted')
 need(task.id==freeze['task_id'] and freeze['fold_id']==manifest['fold_id'],'wrong fold/training identity')
 p=task.get_parameters()
 need(p.get('General/package_manifest_sha256')==freeze['package_manifest_sha256'] and p.get('General/package_independent_readback_sha256')==freeze['package_independent_readback_sha256'] and int(p['General/seed'])==freeze['seed'],'training configuration differs')
 need(hashlib.sha256(task.data.script.diff.encode()).hexdigest()==freeze['controller_sha256'],'training controller differs')
 names={'batch-selection'}
 for side in ('vehicle-side','infrastructure-side'):
  names.update(side+s for s in ('-detector.py','-launch-receipt.json','-optimizer-startup.json','-completion','-final-checkpoint'))
 need(set(freeze['artifacts'])==names,'incomplete two-side freeze')
 for n,r in freeze['artifacts'].items():
  a=task.artifacts[n];need(a.hash==r['sha256'] and a.size==r['bytes'],'training registered artifact differs')
 return p

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('byte-freeze','package','heldout-inputs','forward-acceptance','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--queue',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
 own=ROOT/'source-freezes/spd-canonical-oof-complete-raw-dispatch-20261001'/Path(__file__).name
 need(own.read_bytes()==Path(__file__).read_bytes(),'dispatcher differs from frozen source')
 for dependency in ('spd_canonical_oof_export_binding.py','run_cooptrack_official_oof_gpu4_filehost_v4.py','submit_spd_official_oof_fold_gpu4_offline_gl_nvml_v7.py','run_cooptrack_official_oof_gpu4_offline_gl_v6.py'):
  need((own.parent/dependency).read_bytes()==Path(__file__).with_name(dependency).read_bytes(),'dispatcher dependency changed')
 # Missing or partial freezes are rejected before creating any ClearML task.
 f=json.loads(a.byte_freeze.read_bytes());mp=a.package/'package-manifest.json';m=json.loads(mp.read_bytes())
 need(sha(mp)==f['package_manifest_sha256'] and sha(a.package/'clearml-independent-readback.json')==f['package_independent_readback_sha256'],'local package binding differs')
 for r in f['artifacts'].values():
  path=Path(r['path']);need(len(path.parts)==1 and not path.is_absolute(),'unsafe evidence path')
  local=a.byte_freeze.parent/path;need(not local.is_symlink() and local.stat().st_size==r['bytes'] and sha(local)==r['sha256'],'local evidence bytes differ')
 cloud=ROOT/'receipts/spd-canonical-oof-fivefold-cloud-inference-input-publication-20261001.json';need(sha(cloud)==CLOUD_SHA,'cloud input index differs')
 held=next(r for r in json.loads(cloud.read_bytes())['heldout_folds'] if r['fold_id']==f['fold_id'])
 im=a.heldout_inputs/'input-manifest.json';need(sha(im)==held['input_manifest_sha256'],'heldout input manifest differs')
 inputs=json.loads(im.read_bytes())
 for side in ('vehicle-side','infrastructure-side'):
  def path(suffix):return a.byte_freeze.parent/f['artifacts'][side+suffix]['path']
  launch=json.loads(path('-launch-receipt.json').read_bytes());startup=json.loads(path('-optimizer-startup.json').read_bytes());completion=json.loads(path('-completion').read_bytes())
  validate_binding(m,inputs,f,launch,startup,completion,side=side,seed=f['seed'],checkpoint_sha256=f['artifacts'][side+'-final-checkpoint']['sha256'],config_sha256=sha(path('-detector.py')),package_manifest_sha256=sha(mp),input_manifest_sha256=sha(im),evidence_payloads={side+s:path(s).read_bytes() for s in ('-launch-receipt.json','-optimizer-startup.json','-completion')})
 accepted=json.loads(a.forward_acceptance.read_bytes())
 need(accepted['kind']=='spd_canonical_oof_gpu4_sampled_forward_independent_readback_v1' and accepted['status']=='independent_bytes_and_sampled_tensor_forward_verified' and accepted['training_task_id']==f['task_id'] and accepted['byte_freeze_sha256']==sha(a.byte_freeze) and accepted['fold_id']==f['fold_id'] and accepted['seed']==f['seed'] and accepted['acceptor_sha256']=='fcafab64a2c4edfe958cf749adb69d30fc3231760d510579e93c4b0e083659b1','corresponding independent sampled forward not admitted')
 need(len(accepted['shards'])==4 and {(r['side'],r['shard_index']) for r in accepted['shards']}=={(side,k) for side in ('vehicle-side','infrastructure-side') for k in (0,1)},'forward admission incomplete')
 expected_forward={'sampled-forward-summary'}
 for side in ('vehicle-side','infrastructure-side'):
  for k in (0,1):expected_forward.update((side+'-shard-%d-forward'%k,side+'-shard-%d-forward-log'%k))
 need(set(accepted['artifacts'])==expected_forward and accepted['verifier_sha256']=='8669c5c064c1ac3da8221413cd5f7756b1045049a825793e054d3945515b6a1e','forward artifact/verifier admission incomplete')
 for r in accepted['artifacts'].values():
  q=Path(r['path']);need(len(q.parts)==1 and not q.is_absolute(),'unsafe forward artifact path');q=a.forward_acceptance.parent/q
  need(not q.is_symlink() and q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'independently read forward bytes differ')
 bundle=ROOT/'artifacts/spd-canonical-oof-gpu4-complete-raw-source-20261001';smp=bundle/'source-manifest.json';sr=bundle/'clearml-independent-readback.json'
 need(sha(smp)==SOURCE_MANIFEST and sha(sr)==SOURCE_READBACK,'execution source admission differs')
 sm=json.loads(smp.read_bytes());readback=json.loads(sr.read_bytes());need(readback['status']=='independent_bytes_verified' and readback['task_id']==SOURCE_TASK,'source not admitted')
 inventory={Path(r['path']).name:r for r in sm['inventory']}
 bootstrap=Path(__file__).with_name('bootstrap_spd_oof_complete_raw_export_gpu4.py');need(sha(bootstrap)==inventory[bootstrap.name]['sha256'],'bootstrap changed')
 from clearml import Task
 from clearml.backend_api.session.client import APIClient
 training=Task.get_task(task_id=f['task_id']);completed_evidence(training,f,m)
 forward=Task.get_task(task_id=accepted['task_id']);need(forward.status=='completed','forward no longer completed')
 fp=forward.get_parameters()
 need(fp['General/byte_freeze_sha256']==sha(a.byte_freeze) and fp['General/verifier_sha256']==accepted['verifier_sha256'] and fp['General/source_task_id']=='72068e936baf4f61a0c51bb2065167e7' and fp['General/execution_controller_sha256']=='a01f619549101d809d8ce2ed194784b12200534268790c1acaec84955e303681' and hashlib.sha256(forward.data.script.diff.encode()).hexdigest()=='a435e3efdd76a8d1f063e1d48471aa6f8deb606ffe1e496a890dcc9015fd2110','forward execution identity changed')
 for n,r in accepted['artifacts'].items():
  item=forward.artifacts[n];need(item.hash==r['sha256'] and item.size==r['bytes'],'forward artifact changed')
 source=Task.get_task(task_id=SOURCE_TASK);need(source.status=='completed','source publication not completed')
 for n,r in readback['artifacts'].items():
  item=source.artifacts[n];need(item.hash==r['sha256'] and item.size==r['bytes'],'source registered artifact differs')
 name='SPD canonical OOF fold-%d seed-%d complete raw export '%(f['fold_id'],f['seed']);identity=name+sha(a.byte_freeze)[:12]+' '+inventory['run_spd_canonical_oof_raw_cache.py']['sha256'][:12]
 existing=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'.*$')
 if existing:
  need(len(existing)==1 and existing[0].name==identity,'existing forward evidence differs or duplicated')
  ep=existing[0].get_parameters();need(ep.get('General/byte_freeze_sha256')==sha(a.byte_freeze) and ep.get('General/source_manifest_sha256')==SOURCE_MANIFEST and ep.get('General/forward_acceptance_sha256')==sha(a.forward_acceptance),'existing task identity differs')
  print('EXISTING_RAW_EXPORT_TASK',existing[0].id,str(existing[0].status),flush=True);return
 need(not a.output.exists(),'create-once dispatch receipt already exists; inspect before retry')
 api=APIClient();queues=[q for q in api.queues.get_all(name=a.queue) if q.name==a.queue];need(len(queues)==1,'queue must resolve uniquely');q=queues[0];need(not q.entries,'queue has pending tasks')
 workers=[w.to_dict() for w in api.workers.get_all()];eligible=safe_workers(workers,q.id,datetime.now(timezone.utc))
 if a.check_only:print('RAW_EXPORT_PREREQUISITES_PASSED_NO_TASK_CREATED',json.dumps(eligible));return
 # Recheck the original completed task immediately before creation.
 completed_evidence(Task.get_task(task_id=f['task_id']),f,m)
 task=Task.create(project_name=PROJECT,task_name=identity,task_type=Task.TaskTypes.testing,binary='python3.12')
 task.set_script(repository='',branch='',commit='',working_dir='.',entry_point=bootstrap.name,diff=bootstrap.read_text());task.set_packages(['clearml==2.1.2'])
 task.set_base_docker(DOCKER,docker_arguments='-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 32g --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env CLEARML_FILES_HOST=http://10.100.34.118:8081')
 task.set_parameters({'source_task_id':SOURCE_TASK,'source_manifest_sha256':SOURCE_MANIFEST,'source_archive_sha256':sm['archive']['sha256'],'execution_controller_sha256':inventory['run_spd_oof_complete_raw_export_gpu4.py']['sha256'],'producer_sha256':inventory['run_spd_canonical_oof_raw_cache.py']['sha256'],'cache_verifier_sha256':inventory['spd_canonical_oof_cache_primitives.py']['sha256'],'forward_acceptance_text':a.forward_acceptance.read_text(),'forward_acceptance_sha256':sha(a.forward_acceptance),'byte_freeze_text':a.byte_freeze.read_text(),'byte_freeze_sha256':sha(a.byte_freeze),'cloud_input_evidence_text':cloud.read_text(),'fold_id':f['fold_id'],'seed':f['seed'],'queue':a.queue,'paper_eligible':False})
 task.add_tags(['SPD','canonical-OOF','complete-raw-export','uncalibrated-non-formal'])
 receipt={'kind':'spd_canonical_oof_gpu4_complete_raw_dispatch_v1','task_id':task.id,'training_task_id':training.id,'forward_task_id':forward.id,'forward_acceptance_sha256':sha(a.forward_acceptance),'fold_id':f['fold_id'],'seed':f['seed'],'byte_freeze_sha256':sha(a.byte_freeze),'source_task_id':SOURCE_TASK,'source_manifest_sha256':SOURCE_MANIFEST,'queue':a.queue,'eligible_workers_at_dispatch':eligible,'worker_snapshot':workers,'status':'created_before_enqueue','actual_gpu_export_verified':False,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
 with a.output.open('x') as out:json.dump(receipt,out,indent=2,default=str);out.write('\n')
 # Queue listeners may overlap other queues; fail closed on a new collision.
 fresh=[w.to_dict() for w in api.workers.get_all()];safe_workers(fresh,q.id,datetime.now(timezone.utc))
 Task.enqueue(task,queue_name=a.queue)
 t=Task.get_task(task_id=task.id);need(t.status in ('queued','in_progress'),'enqueue not confirmed; preserve created task')
 accepted=a.output.with_name(a.output.stem+'-enqueue-acceptance.json')
 with accepted.open('x') as out:json.dump(dict(receipt,status=str(t.status)),out,indent=2,default=str);out.write('\n')
 print('RAW_EXPORT_ENQUEUED',task.id,flush=True)
if __name__=='__main__':main()
