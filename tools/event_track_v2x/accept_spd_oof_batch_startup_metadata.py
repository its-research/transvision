"""Independently read batch probes and startup metadata; no final-weight claims."""
import argparse,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
from run_cooptrack_official_oof_gpu4_offline_gl_v6 import validate_cohort,download_registered_artifact

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-id',required=True);p.add_argument('--package',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 from clearml import Task
 task=Task.get_task(task_id=a.task_id);params=task.get_parameters()
 mp=a.package/'package-manifest.json';m=json.loads(mp.read_bytes());validate_cohort(m)
 rb=a.package/'clearml-independent-readback.json';admitted=json.loads(rb.read_bytes())
 need(admitted['status']=='independent_bytes_verified' and admitted['fold_id']==m['fold_id'] and admitted['manifest_sha256']==sha(mp),'package admission differs')
 need(params['General/package_manifest_sha256']==sha(mp) and params['General/package_independent_readback_sha256']==sha(rb) and params['General/package_task_id']==admitted['task_id'] and int(params['General/fold_id'])==m['fold_id'] and int(params['General/world_size'])==4 and int(params['General/epochs_per_side'])==24,'training/package identity differs')
 controller=Path(__file__).with_name('run_cooptrack_official_oof_gpu4_offline_gl_v6.py');need(params['General/controller_sha256']==sha(controller) and hashlib.sha256(task.data.script.diff.encode()).hexdigest()==sha(controller),'controller differs')
 seed=int(params['General/seed']);need(seed in (1337,2027,3407),'unadmitted seed')
 names=['batch-selection','vehicle-side-detector.py','vehicle-side-launch-receipt.json','vehicle-side-optimizer-startup.json']
 need(all(n in task.artifacts for n in names),'batch/startup evidence not ready')
 need(not a.output.exists(),'receipt output is create-once');a.output.mkdir(parents=True)
 records={};parsed={}
 for n in names:
  artifact=task.artifacts[n];q=a.output/(n if n.endswith(('.py','.json')) else n+'.json')
  download_registered_artifact(artifact,artifact.hash,artifact.size,q);records[n]={'path':q.name,'sha256':sha(q),'bytes':q.stat().st_size}
  if not n.endswith('.py'):parsed[n]=json.loads(q.read_bytes())
 selection=parsed['batch-selection'];batch=selection['batch_per_gpu'];fit=m['fit_sequence_ids']
 need(selection['fold_id']==m['fold_id'] and selection['seed']==seed and selection['world_size']==4 and selection['sequence_stream_limit']==len(fit) and selection['maximum_memory_fraction']==.8 and selection['effective_batch_size']==batch*4 and batch in (2,4,8,10) and batch*4<=len(fit) and selection['base_learning_rate_unchanged'] and selection['probe_weights_not_used_for_training'],'batch boundary differs')
 profiles=selection['vehicle_profiles']+[selection['infrastructure_profile']]
 selected=[r for r in profiles if r['batch_per_gpu']==batch];need({r['side'] for r in selected}=={'vehicle-side','infrastructure-side'},'chosen batch lacks both side probes')
 for r in selected:
  need(r['success'] and r['returncode']==0 and r['batch_probe_only'] and r['world_size']==4 and sorted(x['rank'] for x in r['ranks'])==list(range(4)),'chosen probe did not complete four ranks')
  for x in r['ranks']:need(x['iterations']==32 and x['world_size']==4 and x['batch_per_gpu']==batch and x['batch_probe_only'] and 0<x['peak_reserved_bytes']<=.8*x['total_memory_bytes'],'selected rank exceeded sizing gate')
 launch=parsed['vehicle-side-launch-receipt.json'];startup=parsed['vehicle-side-optimizer-startup.json']
 need(launch['fit_sequence_ids']==fit and not set(fit)&set(m['held_out_sequence_ids']) and launch['fold_id']==m['fold_id'] and launch['seed']==seed and launch['side']=='vehicle-side' and launch['epochs']==24 and launch['world_size']==4 and launch['batch_per_gpu']==batch and launch['effective_batch_size']==batch*4 and not launch['batch_probe_only'] and not launch['official_val_test_loaded'] and not launch['raw_labels_modified'] and not launch['paper_ranking_eligible'] and launch['pretrained_kind']=='ImageNet-R50-only-no-SPD-trained-weights','formal startup split/cohort differs')
 pretrained=next(r for r in m['inventory'] if r['path']=='resnet50-0676ba61.pth');need(launch['pretrained_sha256']==pretrained['sha256'],'pretrained identity differs')
 need(startup['kind']=='detector_optimizer_startup_v1' and startup['micro_iteration']==16 and not startup['batch_probe_only'] and math.isfinite(startup['loss']) and startup['backbone_max_abs_update']>0 and math.isfinite(startup['backbone_max_abs_update']) and startup['optimizer_state_entries']>0 and startup['resolved_config_sha256']==records['vehicle-side-detector.py']['sha256'] and startup['batch_per_gpu']==batch and startup['world_size']==4,'optimizer/config metadata differs')
 current=Task.get_task(task_id=task.id)
 need(all(current.artifacts[n].hash==r['sha256'] and current.artifacts[n].size==r['bytes'] for n,r in records.items()),'artifact changed during independent readback')
 result={'kind':'spd_canonical_oof_batch_and_vehicle_startup_metadata_independent_readback_v1','task_id':task.id,'fold_id':m['fold_id'],'seed':seed,'controller_sha256':sha(controller),'package_manifest_sha256':sha(mp),'package_admission_sha256':sha(rb),'artifacts':records,'batch_per_gpu':batch,'effective_batch_size':batch*4,'actual_devices':selection['actual_cuda_devices'],'scope':'both-side selected memory probes and vehicle formal startup metadata only','startup_checkpoint_tensors_verified':False,'full_training_complete':False,'paper_eligible':False,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
 q=a.output/'acceptance-receipt.json'
 with q.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('BATCH_STARTUP_METADATA_INDEPENDENTLY_ACCEPTED',sha(q),flush=True)
if __name__=='__main__':main()
