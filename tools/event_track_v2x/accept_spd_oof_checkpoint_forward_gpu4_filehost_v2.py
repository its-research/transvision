"""Read back completed sampled-forward artifacts; full inference is separate."""
import argparse,hashlib,json,pickle
from pathlib import Path
from datetime import datetime,timezone
from spd_canonical_oof_export_binding import validate_binding
from run_cooptrack_official_oof_gpu4_offline_gl_v6 import download_registered_artifact
from submit_spd_oof_checkpoint_forward_gpu4_filehost_v2 import completed_evidence,SOURCE_TASK,SOURCE_MANIFEST
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
VERIFIER='8669c5c064c1ac3da8221413cd5f7756b1045049a825793e054d3945515b6a1e'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def expected_samples(inputs,side,shard):
 m=json.loads((inputs/'input-manifest.json').read_bytes());spec=m[side]
 info=inputs/side/'image-pose-infos.pkl';index=inputs/side/'frame-index.json'
 need(sha(info)==spec['infos_sha256'] and sha(index)==spec['frame_index_sha256'],'sample inputs differ')
 scenes=sorted(m['train_sequences'])[shard::2]
 payload=pickle.loads(info.read_bytes());values=payload['infos']
 rows=sorted((r for r in values if r['scene_token'] in scenes),key=lambda r:(r['scene_token'],r['timestamp'],r['token']))
 indices={0,len(rows)//2,len(rows)-1}
 for scene in scenes:indices.add(next(i for i,r in enumerate(rows) if r['scene_token']==scene))
 frame_rows={r['frame_id']:r for r in json.loads(index.read_bytes())}
 return [{'sequence_id':rows[i]['scene_token'],'frame_id':rows[i]['token'],'frame_index_row':frame_rows[rows[i]['token']]} for i in sorted(indices)]
def validate_report(report,freeze,freeze_sha,binding,side,shard,samples):
 need(report['kind']=='spd_canonical_oof_final_checkpoint_sampled_forward_acceptance_v1','wrong sampled forward contract')
 need(report['fold_id']==freeze['fold_id'] and report['seed']==freeze['seed'] and report['side']==side and report['shard_index']==shard and report['shard_count']==2 and report['byte_freeze_sha256']==freeze_sha and report['verifier_sha256']==VERIFIER,'forward identity differs')
 need(report['binding']==binding,'forward training/input binding differs')
 need(all(report[k] is True for k in ('strict_state_dict_load_verified','raw_head_forward_verified','weights_unchanged')) and report['complete_prediction_coverage_verified'] is False and report['formal_paper_eligible'] is False,'forward scope or state check differs')
 tr=report['tensor_report'];need(tr['all_state_tensors_finite'] is True and tr['checkpoint_iteration']==freeze['sides'][side]['expected_iterations'] and type(tr['state_tensor_count']) is int and tr['state_tensor_count']>0 and type(tr['state_elements']) is int and tr['state_elements']>0,'tensor inspection incomplete')
 records=report['records'];need(len(records)==len(samples),'sample count differs')
 for actual,expected in zip(records,samples):
  need(all(actual[k]==v for k,v in expected.items()) and actual['arrays_finite'] is True and type(actual['queries']) is int and actual['queries']>0,'sample frame/pose/finite query evidence differs')
 need(isinstance(report['actual_device'],str) and bool(report['actual_device']),'actual device missing')
 return {'side':side,'shard_index':shard,'sample_frames':len(samples),'actual_device':report['actual_device'],'state_tensor_count':tr['state_tensor_count'],'state_elements':tr['state_elements']}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-id',required=True)
 for k in ('byte-freeze','package','heldout-inputs','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();own=ROOT/'source-freezes/spd-canonical-oof-forward-filehost-v2-acceptor-20261001'
 for file in (Path(__file__),*[Path(__file__).with_name(n) for n in ('spd_canonical_oof_export_binding.py','run_cooptrack_official_oof_gpu4_filehost_v4.py','run_cooptrack_official_oof_gpu4_offline_gl_v6.py','submit_spd_oof_checkpoint_forward_gpu4_filehost_v2.py','submit_spd_official_oof_fold_gpu4_offline_gl_nvml_v7.py')]):need(file.read_bytes()==(own/file.name).read_bytes(),'acceptor dependency differs')
 from clearml import Task
 task=Task.get_task(task_id=a.task_id);need(task.status=='completed','forward task not completed')
 f=json.loads(a.byte_freeze.read_bytes());mp=a.package/'package-manifest.json';m=json.loads(mp.read_bytes());need(sha(mp)==f['package_manifest_sha256'],'package differs')
 training=Task.get_task(task_id=f['task_id']);completed_evidence(training,f,m)
 params=task.get_parameters();need(params.get('General/transport_revision')=='filehost-v2' and params.get('General/bootstrap_sha256')=='a9def6d8838c595e702736ffaee8d93b25087af539e4a7da86d1861849939053','transport revision differs');bootstrap=ROOT/'source-freezes/spd-canonical-oof-forward-filehost-v2-acceptor-20261001/bootstrap_spd_oof_checkpoint_forward_gpu4_filehost_v2.py'
 need(hashlib.sha256(task.data.script.diff.encode()).hexdigest()==sha(bootstrap),'executed bootstrap differs')
 source_manifest=json.loads((ROOT/'artifacts/spd-canonical-oof-gpu4-forward-source-20261001/source-manifest.json').read_bytes())
 need(sha(ROOT/'artifacts/spd-canonical-oof-gpu4-forward-source-20261001/source-manifest.json')==SOURCE_MANIFEST,'canonical source manifest differs')
 controller=next(r for r in source_manifest['inventory'] if r['path']=='code/run_spd_oof_checkpoint_forward_gpu4.py')
 need(params['General/execution_controller_sha256']==controller['sha256'] and params['General/source_archive_sha256']==source_manifest['archive']['sha256'] and hashlib.sha256(params['General/cloud_input_evidence_text'].encode()).hexdigest()=='eca6277da199e198fbcf07b8098421787c8eca08136c67e05761497106cdada1','task execution/input source differs')
 need(params['General/source_task_id']==SOURCE_TASK and params['General/source_manifest_sha256']==SOURCE_MANIFEST and params['General/verifier_sha256']==VERIFIER and params['General/byte_freeze_sha256']==sha(a.byte_freeze) and params['General/byte_freeze_text'].encode()==a.byte_freeze.read_bytes(),'task source/checkpoint binding differs')
 expected={'sampled-forward-summary'}
 for side in ('vehicle-side','infrastructure-side'):
  for k in (0,1):expected.update((side+'-shard-%d-forward'%k,side+'-shard-%d-forward-log'%k))
 need(set(task.artifacts)==expected,'missing or unexpected forward artifacts')
 need(not a.output.exists(),'output is create-once');a.output.mkdir(parents=True)
 records={}
 for n in sorted(expected):
  item=task.artifacts[n];dest=a.output/(n+('.log' if n.endswith('-log') else '.json'));download_registered_artifact(item,item.hash,item.size,dest);records[n]={'path':dest.name,'bytes':dest.stat().st_size,'sha256':sha(dest)}
 summary=json.loads((a.output/'sampled-forward-summary.json').read_bytes());need(summary['kind']=='spd_canonical_oof_gpu4_sampled_forward_summary_v1' and summary['training_task_id']==training.id and summary['byte_freeze_sha256']==sha(a.byte_freeze) and summary['fold_id']==f['fold_id'] and summary['seed']==f['seed'] and summary['sampled_tensor_forward_verified'] is True and summary['train_labels_downloaded'] is False and summary['complete_prediction_coverage_verified'] is False and summary['paper_eligible'] is False,'summary binding differs')
 need(len(summary['receipts'])==4 and len(summary['actual_devices'])==4 and len(summary['cuda_visible_devices'])==4 and len(set(summary['cuda_visible_devices']))==4,'four shard/device summary missing')
 im=a.heldout_inputs/'input-manifest.json';inputs=json.loads(im.read_bytes());accepted=[]
 for side in ('vehicle-side','infrastructure-side'):
  def ep(s):return a.byte_freeze.parent/f['artifacts'][side+s]['path']
  evidence={side+s:ep(s).read_bytes() for s in ('-launch-receipt.json','-optimizer-startup.json','-completion')}
  binding=validate_binding(m,inputs,f,json.loads(ep('-launch-receipt.json').read_bytes()),json.loads(ep('-optimizer-startup.json').read_bytes()),json.loads(ep('-completion').read_bytes()),side=side,seed=f['seed'],checkpoint_sha256=f['artifacts'][side+'-final-checkpoint']['sha256'],config_sha256=sha(ep('-detector.py')),package_manifest_sha256=sha(mp),input_manifest_sha256=sha(im),evidence_payloads=evidence)
  for k in (0,1):
   name=side+'-shard-%d-forward'%k;report=json.loads((a.output/records[name]['path']).read_bytes())
   accepted.append(validate_report(report,f,sha(a.byte_freeze),binding,side,k,expected_samples(a.heldout_inputs,side,k)))
   row=next(r for r in summary['receipts'] if r['side']==side and r['shard_index']==k);need(row['sha256']==records[name]['sha256'] and row['bytes']==records[name]['bytes'],'summary shard byte identity differs')
 need(summary['actual_devices']==[r['actual_device'] for r in accepted],'summary actual devices differ')
 current=Task.get_task(task_id=task.id);need(current.status=='completed' and all(current.artifacts[n].hash==r['sha256'] and current.artifacts[n].size==r['bytes'] for n,r in records.items()),'task artifacts changed during readback')
 result={'kind':'spd_canonical_oof_gpu4_sampled_forward_independent_readback_v1','status':'independent_bytes_and_sampled_tensor_forward_verified','task_id':task.id,'training_task_id':training.id,'fold_id':f['fold_id'],'seed':f['seed'],'byte_freeze_sha256':sha(a.byte_freeze),'verifier_sha256':VERIFIER,'acceptor_sha256':sha(Path(__file__)),'transport_revision':'filehost-v2','prior_failed_task_id':params['General/prior_failed_task_id'],'artifacts':records,'shards':accepted,'complete_prediction_coverage_verified':False,'paper_eligible':False,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
 out=a.output/'acceptance-receipt.json'
 with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print('SAMPLED_FORWARD_INDEPENDENTLY_ACCEPTED',sha(out),flush=True)
if __name__=='__main__':main()
