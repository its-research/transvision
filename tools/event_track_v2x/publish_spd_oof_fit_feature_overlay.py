"""Publish a reconstruction-accepted fit metadata overlay; never training results."""
import argparse,json,re
from pathlib import Path
from clearml import Task
from spd_canonical_oof_input_gate import digest

PROJECT='Thesis/EventTrack-V2X/Training'
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--materialization-receipt',type=Path,required=True);p.add_argument('--materialization-sha256',required=True);a=p.parse_args()
 frozen=ROOT/'source-freezes/spd-canonical-oof-fit-feature-overlay-publication-20261001'/Path(__file__).name
 if frozen.read_bytes()!=Path(__file__).read_bytes():raise ValueError('publisher source changed')
 proof=a.package/'independent-overlay-readback.json';mp=a.package/'overlay-manifest.json';m=json.loads(mp.read_bytes());ar=a.package/'fit-feature-overlay.tar.gz'
 if digest(proof)!='a5707455ccd7310e0315378b388e5d9d3172a5829715d1be0005a882f228817b':raise ValueError('overlay independent admission changed')
 z=json.loads(proof.read_bytes())
 if z['overlay_manifest_sha256']!=digest(mp) or z['archive']!=m['archive'] or digest(ar)!=m['archive']['sha256'] or ar.stat().st_size!=m['archive']['bytes']:raise ValueError('overlay bytes changed')
 if digest(a.materialization_receipt)!=a.materialization_sha256:raise ValueError('materialization receipt changed')
 accepted=json.loads(a.materialization_receipt.read_bytes())
 if accepted.get('kind')!='spd_canonical_oof_fit_feature_overlay_materialization_actual_readback_v1' or accepted.get('status')!='passed' or accepted.get('all_reconstructed_payload_bytes_verified') is not True or accepted.get('overlay_admission_sha256')!=digest(proof):raise ValueError('fit inputs not reconstructed and accepted')
 rows=accepted['folds']
 if sorted(r['fold_id'] for r in rows)!=list(range(5)) or sum(sum(r['frames'].values()) for r in rows)!=65352:raise ValueError('incomplete fivefold reconstruction')
 for r in rows:
  original=next(x for x in m['folds'] if x['fold_id']==r['fold_id'])
  if r['input_manifest_sha256']!=original['input_manifest_sha256'] or r['frames']!=original['frames'] or r['held_out_selection_scoring_eligible'] is not False:raise ValueError('reconstruction role/identity differs')
 name='SPD canonical OOF fit feature overlay '+digest(mp)[:12];pin=a.package/'clearml-overlay-task.json'
 same=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
 if len(same)>1:raise ValueError('duplicate overlay tasks')
 if pin.exists():
  prior=json.loads(pin.read_bytes());task=Task.get_task(task_id=prior['task_id'])
  if prior['overlay_manifest_sha256']!=digest(mp):raise ValueError('pinned publication differs')
 elif same:task=same[0]
 else:
  task=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.data_processing);task.output_uri=True
  task.set_parameters({'overlay_manifest_sha256':digest(mp),'overlay_archive_sha256':digest(ar),'overlay_admission_sha256':digest(proof),'materialization_sha256':a.materialization_sha256,'held_out_selection_scoring_eligible':False,'predictions_generated':False,'paper_eligible':False,'source_upload_authorization':'user-explicit-2026-09-29'})
  task.add_tags(['SPD','canonical-OOF','fit-feature-input-overlay','GT-free','not-selection-scoring','non-formal'])
  with pin.open('x') as f:json.dump({'task_id':task.id,'overlay_manifest_sha256':digest(mp)},f,indent=2);f.write('\n')
  task.mark_started(force=True)
  for key,path in [('fit-overlay',ar),('overlay-manifest',mp),('independent-overlay-readback',proof),('materialization-acceptance',a.materialization_receipt)]:
   if not task.upload_artifact(key,artifact_object=path,wait_on_upload=True):raise RuntimeError('publication failed; retain task and inspect before retry')
  task.mark_completed(force=True)
 task=Task.get_task(task_id=task.id)
 if task.status!='completed' or task.get_parameters().get('General/overlay_manifest_sha256')!=digest(mp):raise ValueError('existing publication is not complete or differs; do not duplicate')
 records=[]
 for key,path in [('fit-overlay',ar),('overlay-manifest',mp),('independent-overlay-readback',proof),('materialization-acceptance',a.materialization_receipt)]:
  item=task.artifacts[key]
  if item.hash!=digest(path) or item.size!=path.stat().st_size:raise ValueError('published artifact differs')
  records.append({'name':key,'sha256':digest(path),'bytes':path.stat().st_size})
 result={'kind':'spd_canonical_oof_fit_feature_overlay_clearml_publication_v1','task_id':task.id,'status':'completed_registered_identity_verified','overlay_manifest_sha256':digest(mp),'artifacts':records,'independent_remote_bytes_verified':False,'predictions_generated':False,'paper_eligible':False}
 out=a.package/'clearml-upload-acceptance.json'
 if out.exists():
  if json.loads(out.read_bytes())!=result:raise ValueError('existing receipt differs')
 else:
  with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('FIT_OVERLAY_SOURCE_PUBLISHED',task.id,digest(out),flush=True)

if __name__=='__main__':main()
