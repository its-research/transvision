"""Publish independently verified OpenCV dependency bytes, without training."""
import hashlib,json
from pathlib import Path
from clearml import Task
from run_cooptrack_official_oof_gpu4_filehost_v4 import download_registered_artifact
ROOT=Path('/Volumes/Data/test/recover-before-fuse/artifacts/spd-offline-libgl-runtime-20261001')
SHA='5ac68c58a292e435a0ee55c98f0bd2720a9b088343afc813c262bdc552cc0e10'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 probe=json.loads((ROOT/'target-runtime-probe.json').read_bytes())
 assert probe['network']=='none' and probe['fixed_returncode']==0 and probe['baseline_returncode']!=0 and 'libGL.so.1' in probe['baseline_stderr']
 assert sha(ROOT/'libgl.tar.gz')==SHA
 manifest=ROOT/'manifest.json';assert sha(manifest)=='2905061e79777c2858471fa2ee3ccbfaf3cf064fa62429dfa4e1424d34530f82'
 name='SPD offline GL dependency bundle '+SHA[:12];project='Thesis/EventTrack-V2X/Training'
 prior=Task.get_tasks(project_name=project,task_name='^'+name+'$');assert len(prior)<=1
 t=prior[0] if prior else Task.create(project_name=project,task_name=name,task_type=Task.TaskTypes.data_processing)
 t.output_uri=True
 if t.status!='completed':t.mark_started(force=True)
 for key,p in [('libgl',ROOT/'libgl.tar.gz'),('manifest',manifest),('target-runtime-probe',ROOT/'target-runtime-probe.json')]:
  t.reload()
  if key not in t.artifacts:assert t.upload_artifact(key,artifact_object=p,wait_on_upload=True)
  t.reload();assert t.artifacts[key].hash==sha(p) and t.artifacts[key].size==p.stat().st_size
 if t.status!='completed':t.mark_completed(force=True)
 t.reload();out=ROOT/'clearml-independent-readback';out.mkdir()
 records={}
 for key in ('libgl','manifest','target-runtime-probe'):
  a=t.artifacts[key];p=out/key;download_registered_artifact(a,a.hash,a.size,p)
  records[key]={'bytes':p.stat().st_size,'sha256':sha(p)}
 receipt={'kind':'spd_offline_libgl_dependency_clearml_independent_readback_v1','status':'independent_bytes_verified','task_id':t.id,'artifacts':records,'target_runtime_import_accepted':True,'gpu_forward_accepted':False}
 p=ROOT/'clearml-upload-acceptance.json'
 with p.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print('OFFLINE_GL_PUBLISHED_AND_READ_BACK',t.id,flush=True)
if __name__=='__main__':main()
