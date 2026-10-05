"""Fetch the separately frozen execution source before the GPU forward wrapper."""
import hashlib,json,runpy,sys,tarfile,time
from pathlib import Path,PurePosixPath
from urllib.parse import urlparse,urlunparse
from clearml.backend_api.session import Session
from clearml import Task
from clearml.storage.helper import StorageHelper

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(task,name,expected,path):
 item=task.artifacts[name];url=urlparse(item.url)
 if item.hash!=expected or url.scheme!='http' or url.netloc not in {'10.100.35.118:8081','10.100.34.118:8081'} or url.username or url.password or url.query:raise ValueError('unadmitted source identity/file service')
 active=urlparse(Session.get_files_server_host())
 if active.scheme!='http' or active.netloc not in {'10.100.35.118:8081','10.100.34.118:8081'} or active.username or active.password or active.query or active.fragment or url.fragment:raise ValueError('unadmitted active file service')
 routed=urlunparse(url._replace(netloc=active.netloc))
 h=hashlib.sha256();count=0;start=time.monotonic()
 with path.open('xb') as out:
  for block in StorageHelper.get(routed).download_as_stream(routed,chunk_size=1024*1024):
   out.write(block);h.update(block);count+=len(block)
   if count>item.size:raise ValueError('source exceeds registered size')
 if count!=item.size or h.hexdigest()!=expected:raise ValueError('source bytes differ')
 print('SOURCE_DOWNLOAD_ETA '+json.dumps({'artifact':name,'bytes':count,'elapsed_seconds':time.monotonic()-start,'eta_seconds':0.0}),flush=True)

def main():
 task=Task.init(project_name='Thesis/EventTrack-V2X/Training',task_name='SPD canonical OOF final checkpoint sampled forward GPU4',reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False)
 p={k.split('/',1)[-1]:v for k,v in task.get_parameters().items() if k.startswith('General/')}
 owner=Task.get_task(task_id=p['source_task_id'])
 if owner.status!='completed':raise ValueError('execution source task not completed')
 root=Path('/eventtrack-forward-code');root.mkdir()
 manifest=root/'source-manifest.json';archive=root/'source.tar.gz'
 fetch(owner,'source-manifest',p['source_manifest_sha256'],manifest);fetch(owner,'source',p['source_archive_sha256'],archive)
 m=json.loads(manifest.read_bytes())
 if m['kind']!='spd_canonical_oof_gpu4_forward_source_bundle_v1' or m['archive']['sha256']!=digest(archive) or m['archive']['bytes']!=archive.stat().st_size or m['actual_gpu_execution_verified'] is not False:raise ValueError('wrong source contract')
 expected={r['path']:r for r in m['inventory']}
 if len(expected)!=len(m['inventory']):raise ValueError('duplicate source inventory')
 seen=set()
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   path=PurePosixPath(member.name)
   if not member.isfile() or member.name not in expected or member.name in seen or path.is_absolute() or '..' in path.parts or len(path.parts)!=2 or path.parts[0]!='code' or not path.name.endswith('.py'):raise ValueError('unsafe source archive member')
   row=expected[member.name];raw=tar.extractfile(member).read()
   if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('source payload differs')
   target=root/member.name;target.parent.mkdir(exist_ok=True)
   with target.open('xb') as out:out.write(raw)
   seen.add(member.name)
 if seen!=set(expected):raise ValueError('incomplete execution source')
 controller=root/'code/run_spd_oof_checkpoint_forward_gpu4_mmcv14_iter_v2.py';verifier=root/'code/verify_spd_canonical_oof_checkpoint_forward_mmcv14_iter_v2.py'
 if digest(controller)!=p['execution_controller_sha256'] or digest(verifier)!=p['verifier_sha256']:raise ValueError('execution entrypoint differs')
 sys.path.insert(0,str(controller.parent));runpy.run_path(str(controller),run_name='__main__')

if __name__=='__main__':main()
