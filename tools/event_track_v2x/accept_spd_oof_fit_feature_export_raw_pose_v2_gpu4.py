"""Full independent cloud byte/content readback of uncalibrated OOF raw exports."""
import argparse,hashlib,json,tarfile,shutil
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from audit_spd_oof_fit_feature_raw_cache_raw_pose_v2 import audit,sha,need
from spd_canonical_oof_fit_feature_export_binding import validate_binding
from run_cooptrack_official_oof_gpu4_offline_gl_v6 import download_registered_artifact
from submit_spd_oof_fit_feature_export_raw_pose_v2_gpu4 import completed_evidence,SOURCE_TASK,SOURCE_MANIFEST,SOURCE_READBACK,ROOT

def extract_cache(archive,destination,manifest,manifest_bytes):
 expected={'raw-cache-manifest.json':{'bytes':len(manifest_bytes),'sha256':hashlib.sha256(manifest_bytes).hexdigest()},'launch-receipt.json':None,'resolved-cache-config.py':{'sha256':manifest['resolved_config_sha256']}}
 for frame in manifest['frames']:
  for key in ('arrays','metadata'):
   row=frame[key];relative=PurePosixPath(row['path']);need(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() not in expected,'unsafe or duplicate cache member')
   expected[relative.as_posix()]=row
 need(not destination.exists(),'cache extraction is create-once');destination.mkdir();seen=set()
 with tarfile.open(archive,'r:gz') as tf:
  for member in tf:
   relative=PurePosixPath(member.name);need(member.isfile() and not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='cache','non-file or unsafe archive member')
   name=PurePosixPath(*relative.parts[1:]).as_posix();need(name in expected and name not in seen,'extra or duplicate archive file')
   row=expected[name]
   if row and 'bytes' in row:need(member.size==row['bytes'],'archive member size differs')
   else:need(member.size<8*1024*1024,'oversized cache header')
   target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
   with tf.extractfile(member) as source,target.open('xb') as out:
    for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block);count+=len(block);out.write(block)
   need(count==member.size,'truncated archive member')
   if row and 'sha256' in row:need(h.hexdigest()==row['sha256'],'archive member bytes differ')
   seen.add(name)
 need(seen==set(expected),'archive inventory incomplete')
 return len(seen)

def expected_binding(package,inputs,freeze,freeze_folder,side,package_sha,input_sha):
 def ep(s):return freeze_folder/freeze['artifacts'][side+s]['path']
 evidence={side+s:ep(s).read_bytes() for s in ('-launch-receipt.json','-optimizer-startup.json','-completion')}
 return validate_binding(package,inputs,freeze,json.loads(ep('-launch-receipt.json').read_bytes()),json.loads(ep('-optimizer-startup.json').read_bytes()),json.loads(ep('-completion').read_bytes()),side=side,seed=freeze['seed'],checkpoint_sha256=freeze['artifacts'][side+'-final-checkpoint']['sha256'],config_sha256=sha(ep('-detector.py')),package_manifest_sha256=package_sha,input_manifest_sha256=input_sha,evidence_payloads=evidence)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-id',required=True)
 for key in ('byte-freeze','package','fit-inputs','forward-acceptance','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();frozen_source=ROOT/'source-freezes/spd-fit-feature-raw-pose-v2-cloud-acceptor-20261001'
 source_freeze=json.loads((frozen_source/'source-freeze-receipt.json').read_bytes())
 for r in source_freeze['inventory']:need(sha(Path(__file__).with_name(r['name']))==r['sha256'],'acceptor dependency changed')
 from clearml import Task
 task=Task.get_task(task_id=a.task_id);need(task.status=='completed','raw export task not completed')
 f=json.loads(a.byte_freeze.read_bytes());mp=a.package/'package-manifest.json';m=json.loads(mp.read_bytes());need(sha(mp)==f['package_manifest_sha256'] and sha(a.package/'clearml-independent-readback.json')==f['package_independent_readback_sha256'],'package binding differs')
 training=Task.get_task(task_id=f['task_id']);completed_evidence(training,f,m)
 for row in f['artifacts'].values():
  local=a.byte_freeze.parent/row['path'];need(Path(row['path']).name==row['path'] and not local.is_symlink() and local.stat().st_size==row['bytes'] and sha(local)==row['sha256'],'local training frozen bytes differ')
 forward_acceptance=json.loads(a.forward_acceptance.read_bytes());need(forward_acceptance['status']=='independent_bytes_and_sampled_tensor_forward_verified' and forward_acceptance['training_task_id']==training.id and forward_acceptance['byte_freeze_sha256']==sha(a.byte_freeze),'independent forward differs')
 forward=Task.get_task(task_id=forward_acceptance['task_id']);need(forward.status=='completed','forward no longer completed')
 need(forward_acceptance['kind']=='spd_canonical_oof_gpu4_sampled_forward_independent_readback_v1' and forward_acceptance['fold_id']==f['fold_id'] and forward_acceptance['seed']==f['seed'] and forward_acceptance['acceptor_sha256']=='fcafab64a2c4edfe958cf749adb69d30fc3231760d510579e93c4b0e083659b1' and forward_acceptance['verifier_sha256']=='8669c5c064c1ac3da8221413cd5f7756b1045049a825793e054d3945515b6a1e','forward acceptance kind/identity differs')
 fp=forward.get_parameters()
 need(fp['General/byte_freeze_sha256']==sha(a.byte_freeze) and fp['General/source_task_id']=='72068e936baf4f61a0c51bb2065167e7' and fp['General/source_manifest_sha256']=='7da98b6ada7114f001faf173fe0552427ddb27fd237f5c037eda10bd6271b353' and fp['General/execution_controller_sha256']=='a01f619549101d809d8ce2ed194784b12200534268790c1acaec84955e303681' and fp['General/verifier_sha256']==forward_acceptance['verifier_sha256'] and hashlib.sha256(forward.data.script.diff.encode()).hexdigest()=='a435e3efdd76a8d1f063e1d48471aa6f8deb606ffe1e496a890dcc9015fd2110','forward execution source differs')
 forward_artifacts={'sampled-forward-summary'}
 for side in ('vehicle-side','infrastructure-side'):
  for shard in (0,1):forward_artifacts.update((side+'-shard-%d-forward'%shard,side+'-shard-%d-forward-log'%shard))
 need(set(forward_acceptance['artifacts'])==forward_artifacts and len(forward_acceptance['shards'])==4 and {(r['side'],r['shard_index']) for r in forward_acceptance['shards']}=={(side,k) for side in ('vehicle-side','infrastructure-side') for k in (0,1)},'forward admission incomplete')
 for name,row in forward_acceptance['artifacts'].items():
  local=a.forward_acceptance.parent/row['path'];need(Path(row['path']).name==row['path'] and not local.is_symlink() and local.stat().st_size==row['bytes'] and sha(local)==row['sha256'],'forward independently read bytes differ')
  need(forward.artifacts[name].hash==row['sha256'] and forward.artifacts[name].size==row['bytes'],'forward registration changed')
 bundle=ROOT/'artifacts/spd-oof-gpu4-fit-feature-raw-pose-v2-source-20261001';smp=bundle/'source-manifest.json';need(sha(smp)==SOURCE_MANIFEST and sha(bundle/'clearml-independent-readback.json')==SOURCE_READBACK,'raw source admission differs')
 sm=json.loads(smp.read_bytes());inventory={Path(r['path']).name:r for r in sm['inventory']};params=task.get_parameters()
 expected_parameters={'source_task_id':SOURCE_TASK,'source_manifest_sha256':SOURCE_MANIFEST,'source_archive_sha256':sm['archive']['sha256'],'execution_controller_sha256':inventory['run_spd_oof_fit_feature_export_raw_pose_v2_gpu4.py']['sha256'],'producer_sha256':inventory['run_spd_canonical_oof_fit_feature_raw_cache_raw_pose_v2.py']['sha256'],'cache_verifier_sha256':inventory['spd_canonical_oof_fit_feature_cache_primitives.py']['sha256'],'pose_metadata_helper_sha256':inventory['spd_canonical_oof_raw_pose_metadata_v2.py']['sha256'],'byte_freeze_sha256':sha(a.byte_freeze),'byte_freeze_text':a.byte_freeze.read_text(),'forward_acceptance_sha256':sha(a.forward_acceptance),'forward_acceptance_text':a.forward_acceptance.read_text()}
 need(all(params.get('General/'+k)==v for k,v in expected_parameters.items()),'raw executed configuration differs')
 need(hashlib.sha256(task.data.script.diff.encode()).hexdigest()==inventory['bootstrap_spd_oof_fit_feature_export_raw_pose_v2_gpu4.py']['sha256'],'executed raw bootstrap differs')
 expected={'complete-fit-feature-export-summary','fit-input-materialization'}
 for side in ('vehicle-side','infrastructure-side'):
  for shard in (0,1):expected.update(side+'-shard-%d-'%shard+suffix for suffix in ('export-log','raw-cache','raw-manifest','raw-readback'))
 need(set(task.artifacts)==expected,'raw export artifact inventory incomplete/extra')
 need(not a.output.exists(),'acceptance output is create-once')
 need(shutil.disk_usage(a.output.parent).free>3*sum(task.artifacts[n].size for n in expected),'insufficient local space for independent raw cache readback')
 a.output.mkdir(parents=True);records={}
 # Read headers before cache archives so full extraction can enforce exact inventory.
 for name in sorted(expected,key=lambda n:n.endswith('raw-cache')):
  item=task.artifacts[name];suffix='.tar.gz' if name.endswith('raw-cache') else '.log' if name.endswith('export-log') else '.json';dest=a.output/(name+suffix)
  download_registered_artifact(item,item.hash,item.size,dest);records[name]={'path':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size}
 summary=json.loads((a.output/records['complete-fit-feature-export-summary']['path']).read_bytes())
 need(summary['kind']=='spd_canonical_oof_gpu4_fit_feature_raw_pose_v2_summary' and summary['training_task_id']==training.id and summary['forward_task_id']==forward.id and summary['forward_acceptance_sha256']==sha(a.forward_acceptance) and summary['byte_freeze_sha256']==sha(a.byte_freeze) and summary['fold_id']==f['fold_id'] and summary['seed']==f['seed'] and summary['metadata_pose_source']=='raw-calibration-composition-float64-v2' and summary['pose_metadata_helper_sha256']==expected_parameters['pose_metadata_helper_sha256'],'raw summary differs')
 need(summary['job_local_all_payloads_and_archive_bytes_verified'] is True and all(summary[k] is False for k in ('independent_cloud_readback_verified','train_labels_downloaded','covariance_calibrated','formal_v2_ready','paper_eligible','held_out_selection_scoring_eligible')),'raw summary overclaims scope')
 need(len(summary['shards'])==4 and len(summary['actual_devices'])==4 and len(summary['cuda_visible_devices'])==4 and len(set(summary['cuda_visible_devices']))==4,'four shard/device summary incomplete')
 im=a.fit_inputs/'input-manifest.json';inputs=json.loads(im.read_bytes());totals={s:0 for s in ('vehicle-side','infrastructure-side')};accepted=[]
 need(params.get('General/fit_input_manifest_sha256')==sha(im) and summary['fit_input_manifest_sha256']==sha(im) and summary['fit_sequence_ids']==inputs['fit_sequence_ids']==m['fit_sequence_ids'] and summary['excluded_held_out_sequence_ids']==inputs['excluded_held_out_sequence_ids']==m['held_out_sequence_ids'],'fit input summary/provenance differs')
 cloud=ROOT/'receipts/spd-canonical-oof-fivefold-cloud-inference-input-publication-20261001.json'
 overlay=ROOT/'artifacts/spd-canonical-oof-fit-feature-overlay-20261001/clearml-independent-readback.json'
 need(sha(cloud)=='eca6277da199e198fbcf07b8098421787c8eca08136c67e05761497106cdada1' and params.get('General/cloud_input_evidence_text')==cloud.read_text() and sha(overlay)=='0a2d53e84f346383852efadb9dec05a7064d8e0543c3bc4c2156f4bd1d04e889' and params.get('General/fit_overlay_admission_text')==overlay.read_text(),'fit cloud download admission differs')
 materialization=json.loads((a.output/records['fit-input-materialization']['path']).read_bytes())
 need(materialization['kind']=='spd_canonical_oof_single_fit_feature_overlay_materialization_actual_readback_v1' and materialization['fold_id']==f['fold_id'] and materialization['status']=='passed' and materialization['all_reconstructed_payload_bytes_verified'] is True and materialization['predictions_generated'] is False and materialization['paper_eligible'] is False and materialization['overlay_admission_sha256']=='a5707455ccd7310e0315378b388e5d9d3172a5829715d1be0005a882f228817b','fit reconstruction receipt differs')
 need(len(materialization['folds'])==1 and materialization['folds'][0]['fold_id']==f['fold_id'] and materialization['folds'][0]['input_manifest_sha256']==sha(im) and materialization['folds'][0]['frames']==inputs['frames'] and materialization['folds'][0]['held_out_selection_scoring_eligible'] is False,'fit reconstruction fold/role differs')
 poses=ROOT/'artifacts/spd-canonical-oof-heldout-raw-poses-20260930'
 for side in totals:
  binding=expected_binding(m,inputs,f,a.byte_freeze.parent,side,sha(mp),sha(im))
  for shard in (0,1):
   prefix=side+'-shard-%d'%shard;cm_bytes=(a.output/records[prefix+'-raw-manifest']['path']).read_bytes();cm=json.loads(cm_bytes);rb=json.loads((a.output/records[prefix+'-raw-readback']['path']).read_bytes());row=next(r for r in summary['shards'] if r['side']==side and r['shard_index']==shard)
   need(row==dict(rb,manifest_sha256=records[prefix+'-raw-manifest']['sha256'],actual_device=cm['physical_device']) and rb['archive']=={k:records[prefix+'-raw-cache'][k] for k in ('sha256','bytes')} and rb['all_archive_members_read'] is True,'shard summary/archive bytes differ')
   need(cm['side']==side and cm['fold_id']==f['fold_id'] and cm['seed']==f['seed'] and cm['shard_index']==shard and cm['shard_count']==2 and cm['sequences']==sorted(inputs['fit_sequence_ids'])[shard::2] and cm['byte_freeze_sha256']==sha(a.byte_freeze) and cm['training_binding']==binding and cm['producer_code_sha256']==expected_parameters['producer_sha256'] and cm['pose_metadata_helper_sha256']==expected_parameters['pose_metadata_helper_sha256'] and cm['verifier_code_sha256']==expected_parameters['cache_verifier_sha256'],'cache provenance differs')
   need(cm['kind']=='eventtrack_fit_feature_raw_detector_cache_v1' and cm['canonical_oof_fit_feature_export'] is True and cm['held_out_selection_scoring_eligible'] is False,'fit cache scope differs')
   directory=a.output/(prefix+'-cache');count=extract_cache(a.output/records[prefix+'-raw-cache']['path'],directory,cm,cm_bytes);need(count==rb['archive_files'],'archive file count differs')
   checked=audit(directory,a.fit_inputs,mp,poses)
   need(checked['frames_verified']==cm['frame_count'] and checked['detections_verified']==cm['detection_count'] and all(rb['cache'][k]==checked[k] for k in rb['cache']),'independent full content count differs')
   totals[side]+=checked['frames_verified'];accepted.append(dict(checked,side=side,shard_index=shard,actual_device=cm['physical_device']))
   print('RAW_CLOUD_CONTENT_READBACK',side,shard,checked['frames_verified'],'ETA=unknown',flush=True)
 need(totals==inputs['frames']==summary['frame_totals'] and summary['actual_devices']==[r['actual_device'] for r in accepted],'complete fold side frame/device totals differ')
 current=Task.get_task(task_id=task.id);need(current.status=='completed' and all(current.artifacts[n].hash==r['sha256'] and current.artifacts[n].size==r['bytes'] for n,r in records.items()),'task changed during independent readback')
 result={'kind':'spd_canonical_oof_fit_feature_raw_pose_v2_cloud_independent_content_readback','status':'all_cloud_bytes_frames_arrays_raw_poses_verified','task_id':task.id,'training_task_id':training.id,'forward_task_id':forward.id,'fold_id':f['fold_id'],'seed':f['seed'],'byte_freeze_sha256':sha(a.byte_freeze),'forward_acceptance_sha256':sha(a.forward_acceptance),'source_task_id':SOURCE_TASK,'executing_worker':task.data.last_worker,'cuda_visible_devices':summary['cuda_visible_devices'],'acceptor_sha256':sha(Path(__file__)),'artifacts':records,'shards':accepted,'frame_totals':totals,'covariance_calibrated':False,'formal_v2_ready':False,'held_out_selection_scoring_eligible':False,'paper_eligible':False,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
 out=a.output/'acceptance-receipt.json'
 with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print('COMPLETE_RAW_CLOUD_INDEPENDENTLY_ACCEPTED',sha(out),flush=True)
if __name__=='__main__':main()
