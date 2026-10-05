"""Publish byte-audited forward source only; does not launch GPU work."""
import argparse, json, re, tarfile, hashlib
from pathlib import Path
from clearml import Task
from spd_canonical_oof_input_gate import digest
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
MANIFEST='04681ea607045761eea078de04a62e764e351a0d64aa2d0b07760d0107fc0dd7'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);a=p.parse_args()
 frozen=ROOT/'source-freezes/spd-canonical-oof-complete-raw-pose-v2-source-publication-20261001'/Path(__file__).name
 if frozen.read_bytes()!=Path(__file__).read_bytes():raise ValueError('publisher changed')
 mp=a.package/'source-manifest.json';ar=a.package/'source.tar.gz';proof=a.package/'independent-source-archive-readback.json'
 m=json.loads(mp.read_bytes());z=json.loads(proof.read_bytes())
 if digest(mp)!=MANIFEST or digest(ar)!=m['archive']['sha256'] or ar.stat().st_size!=m['archive']['bytes'] or z['source_manifest_sha256']!=MANIFEST or z['all_members_read'] is not True or z['members_verified']!=len(m['inventory']):raise ValueError('source admission differs')
 expected={r['path']:r for r in m['inventory']}
 with tarfile.open(ar,'r:gz') as tf:
  members=tf.getmembers()
  if len(members)!=len(expected) or set(x.name for x in members)!=set(expected):raise ValueError('archive inventory differs')
  for x in members:
   if not x.isfile():raise ValueError('non-file source member')
   b=tf.extractfile(x).read();r=expected[x.name]
   if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('source bytes differ')
 files=[('source',ar),('source-manifest',mp),('independent-source-archive-readback',proof)]
 name='SPD canonical OOF GPU4 complete raw pose-v2 source '+MANIFEST[:12];pin=a.package/'clearml-source-task.json'
 same=Task.get_tasks(project_name='Thesis/EventTrack-V2X/Training',task_name='^'+re.escape(name)+'$')
 if len(same)>1:raise ValueError('duplicate source tasks')
 if pin.exists():
  prior=json.loads(pin.read_bytes())
  if prior['source_manifest_sha256']!=MANIFEST:raise ValueError('pin differs')
  task=Task.get_task(task_id=prior['task_id'])
 elif same:task=same[0]
 else:
  task=Task.create(project_name='Thesis/EventTrack-V2X/Training',task_name=name,task_type=Task.TaskTypes.data_processing);task.output_uri=True
  task.set_parameters({'source_manifest_sha256':MANIFEST,'source_archive_sha256':digest(ar),'source_admission_sha256':digest(proof),'actual_gpu_execution_verified':False,'paper_eligible':False,'source_upload_authorization':'user-explicit'})
  with pin.open('x') as f:json.dump({'task_id':task.id,'source_manifest_sha256':MANIFEST},f,indent=2);f.write('\n')
  task.mark_started(force=True)
  for key,path in files:
   if not task.upload_artifact(key,artifact_object=path,wait_on_upload=True):raise RuntimeError('upload failed; preserve task')
  task.mark_completed(force=True)
 task=Task.get_task(task_id=task.id)
 if task.status!='completed' or task.get_parameters().get('General/source_manifest_sha256')!=MANIFEST:raise ValueError('existing source incomplete or differs; no restart')
 records=[]
 for key,path in files:
  item=task.artifacts[key]
  if item.hash!=digest(path) or item.size!=path.stat().st_size:raise ValueError('registered artifact differs')
  records.append({'name':key,'sha256':digest(path),'bytes':path.stat().st_size})
 result={'kind':'spd_canonical_oof_complete_raw_pose_v2_source_publication','status':'completed_registered_identity_verified','task_id':task.id,'source_manifest_sha256':MANIFEST,'artifacts':records,'independent_remote_bytes_verified':False,'actual_gpu_execution_verified':False,'paper_eligible':False}
 out=a.package/'clearml-upload-acceptance.json'
 if out.exists():
  if json.loads(out.read_bytes())!=result:raise ValueError('receipt differs')
 else:
  with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('COMPLETE_RAW_POSE_V2_SOURCE_PUBLISHED',task.id,digest(out),flush=True)
if __name__=='__main__':main()
