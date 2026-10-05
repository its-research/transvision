"""Publish admitted held-out image/pose inputs, never labels or test splits."""
import argparse,hashlib,json,re
from pathlib import Path
from clearml import Task
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
PROJECT='Thesis/EventTrack-V2X/Training'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--independent-receipt-sha256',required=True);a=p.parse_args()
 source=Path(__file__);freeze=ROOT/'source-freezes/spd-oof-heldout-inference-package-upload-20261001'/source.name
 if source.read_bytes()!=freeze.read_bytes():raise ValueError('publisher source differs from freeze')
 receipt=a.package/'independent-archive-readback.json'
 if sha(receipt)!=a.independent_receipt_sha256:raise ValueError('independent admission receipt changed')
 accepted=json.loads(receipt.read_bytes());mp=a.package/'package-manifest.json';m=json.loads(mp.read_bytes());ar=a.package/'heldout-inputs.tar.gz'
 if (accepted.get('kind')!='spd_canonical_oof_heldout_inference_package_independent_readback_v1'
     or accepted.get('all_archive_payloads_verified') is not True
     or accepted['source_input_admission_sha256']!='29214bb602a4637637a479636109e6bec6591282d7e2647c2d628bc8ebd967df'
     or accepted['package_manifest_sha256']!=sha(mp) or accepted['archive']!=m['archive']
     or accepted['fold_id']!=m['fold_id'] or accepted['frames']!=m['frames']
     or accepted['files_verified']!=len(m['input_inventory'])
     or m['kind']!='spd_canonical_oof_heldout_inference_package_v1'
     or m['gt_or_val_test_included'] is not False or m['predictions_generated'] is not False
     or ar.stat().st_size!=m['archive']['bytes'] or sha(ar)!=m['archive']['sha256']):
  raise ValueError('held-out package lacks full immutable byte admission')
 name='SPD canonical OOF fold-%d held-out GT-free inputs %s'%(m['fold_id'],sha(mp)[:12])
 pin=a.package/'clearml-package-task.json'
 if pin.exists():
  prior=json.loads(pin.read_bytes())
  if prior['manifest_sha256']!=sha(mp) or prior['independent_receipt_sha256']!=a.independent_receipt_sha256:raise ValueError('existing task belongs to different admission')
  task=Task.get_task(task_id=prior['task_id'])
 else:
  same=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
  if len(same)>1:raise ValueError('duplicate held-out input tasks')
  task=same[0] if same else Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.data_processing)
  if same and task.get_parameters().get('General/package_manifest_sha256')!=sha(mp):raise ValueError('existing task identity differs')
  if not same:
   task.output_uri=True;task.add_tags(['SPD','canonical-OOF','held-out-inference-inputs','GT-free','fold-%d'%m['fold_id']])
   task.set_parameters({'package_manifest_sha256':sha(mp),'independent_receipt_sha256':a.independent_receipt_sha256,'fold_id':m['fold_id'],'archive_sha256':m['archive']['sha256'],'input_manifest_sha256':m['input_manifest_sha256'],'gt_or_val_test_included':False,'predictions_generated':False,'source_upload_authorization':'user-explicit-2026-09-29'})
  with pin.open('x') as f:json.dump({'task_id':task.id,'manifest_sha256':sha(mp),'independent_receipt_sha256':a.independent_receipt_sha256},f,indent=2);f.write('\n')
 if task.status!='completed':task.mark_started(force=True)
 for name,path in [('heldout-inputs',ar),('package-manifest',mp),('independent-archive-readback',receipt)]:
  current=Task.get_task(task_id=task.id);expected=sha(path)
  if name not in current.artifacts:
   if not task.upload_artifact(name,artifact_object=path,wait_on_upload=True):raise RuntimeError('input upload failed')
  current=Task.get_task(task_id=task.id)
  if current.artifacts[name].hash!=expected or current.artifacts[name].size!=path.stat().st_size:raise ValueError('published input bytes differ')
  print('HELDOUT_INPUT_ARTIFACT_REGISTERED',name,expected,flush=True)
 task.mark_completed(force=True)
 output=a.package/'clearml-upload-acceptance.json'
 result={'kind':'spd_canonical_oof_heldout_inference_input_upload_v1','task_id':task.id,'fold_id':m['fold_id'],'manifest_sha256':sha(mp),'archive':m['archive'],'independent_receipt_sha256':a.independent_receipt_sha256,'status':'uploaded_and_registered_hash_verified','remote_independent_byte_readback_passed':False,'predictions_generated':False,'paper_eligible':False}
 if output.exists():
  if json.loads(output.read_bytes())!=result:raise ValueError('prior upload acceptance changed')
 else:
  with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('HELDOUT_INPUT_PACKAGE_UPLOAD_COMPLETE',task.id,flush=True)
if __name__=='__main__':main()
