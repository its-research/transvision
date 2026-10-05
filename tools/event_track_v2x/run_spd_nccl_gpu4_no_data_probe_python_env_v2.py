"""Bounded four-GPU NCCL communication diagnostics; no training/data access."""
import json,os,selectors,signal,subprocess,tarfile,time,hashlib
from pathlib import Path,PurePosixPath
from urllib.parse import urlparse,urlunparse
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def validate_archive(path, prefixes, allow_links=False):
    """Validate paths before extracting a package with a separately pinned hash."""
    count = 0
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe archive member path")
            if not any(str(name) == prefix or str(name).startswith(prefix + "/")
                       for prefix in prefixes):
                raise ValueError("archive member outside the package prefixes")
            if entry.isdev() or entry.isfifo():
                raise ValueError("archive contains special device or FIFO")
            if entry.issym() or entry.islnk():
                if not allow_links:
                    raise ValueError("unexpected link in train or source archive")
                link = PurePosixPath(entry.linkname)
                target = str(link) if link.is_absolute() or entry.islnk() else str(name.parent / link)
                normalized = os.path.normpath("/" + target.lstrip("/"))
                if not any(normalized == "/" + prefix or normalized.startswith("/" + prefix + "/")
                           for prefix in prefixes):
                    raise ValueError("archive link escapes the authorized runtime prefix")
            count += 1
    if not count:
        raise ValueError("empty archive")
    return count

def download_registered_artifact(artifact, expected_sha, expected_bytes, target):
    from clearml.storage.helper import StorageHelper
    from clearml.backend_api.session import Session
    registered = urlparse(artifact.url)
    active = urlparse(Session.get_files_server_host())
    hosts = {"10.100.35.118:8081", "10.100.34.118:8081"}
    if (registered.scheme != "http" or active.scheme != "http"
            or registered.netloc not in hosts or active.netloc not in hosts
            or registered.username or registered.query or registered.fragment):
        raise ValueError("unregistered artifact/file-service route")
    if artifact.hash != expected_sha or artifact.size != expected_bytes:
        raise ValueError("registered artifact identity differs")
    url = urlunparse(registered._replace(netloc=active.netloc))
    target = Path(target)
    digest = hashlib.sha256()
    received = 0
    started = time.monotonic()
    print("EVENTTRACK_DOWNLOAD_ETA " + json.dumps({"artifact": target.name,
          "eta_seconds": None, "eta_status": "unknown", "file_service": active.netloc}), flush=True)
    with target.open("xb") as stream:
        for block in StorageHelper.get(url).download_as_stream(url):
            stream.write(block)
            digest.update(block)
            prior = received
            received += len(block)
            if received > expected_bytes:
                raise ValueError("artifact longer than registered byte count")
            if received // (256 * 1024 * 1024) > prior // (256 * 1024 * 1024):
                elapsed = time.monotonic() - started
                print("EVENTTRACK_DOWNLOAD_ETA " + json.dumps({"artifact": target.name,
                      "received_bytes": received, "expected_bytes": expected_bytes,
                      "eta_seconds": (expected_bytes-received)*elapsed/received}), flush=True)
    if received != expected_bytes or digest.hexdigest() != expected_sha:
        raise ValueError("downloaded byte count or SHA differs")
    return target

WORKER="import argparse,json,os\nfrom datetime import timedelta\nfrom pathlib import Path\nimport torch\nimport torch.distributed as dist\np=argparse.ArgumentParser();p.add_argument('--local_rank',type=int,default=int(os.environ.get('LOCAL_RANK','0')));p.add_argument('--output',required=True);a=p.parse_args()\ntorch.cuda.set_device(a.local_rank);assert torch.cuda.device_count()==4\ndist.init_process_group('nccl',timeout=timedelta(seconds=45));rank=dist.get_rank();assert dist.get_world_size()==4\nx=torch.tensor([1337 if rank==0 else 0],dtype=torch.int32,device='cuda');dist.broadcast(x,src=0);assert x.item()==1337\nfor count in (1,262144):\n x=torch.full((count,),float(rank+1),device='cuda');dist.all_reduce(x);assert torch.equal(x,torch.full_like(x,10.0))\ndist.barrier();torch.cuda.synchronize()\nreport={'rank':rank,'local_rank':a.local_rank,'device':torch.cuda.get_device_name(a.local_rank),'torch':torch.__version__,'broadcast_passed':True,'all_reduce_passed':True,'payload_elements':[1,262144]}\nwith (Path(a.output)/('rank-%d.json'%rank)).open('x') as f:json.dump(report,f)\ndist.destroy_process_group();print('NCCL_RANK_ACCEPTED '+json.dumps(report),flush=True)\n"

def main():
 from clearml import Task
 task=Task.init(project_name='Thesis/EventTrack-V2X/Training',task_name='SPD four-GPU no-data NCCL diagnostic',reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False)
 root=Path('/eventtrack-nccl-probe');root.mkdir()
 runtime=Task.get_task(task_id='9a7e7a9213954b57a35403f777e74561')
 archive=download_registered_artifact(runtime.artifacts['runtime'],'b6b39c66eec6921c0f0f51d2b9cd4e5571fcafdc44d3605112dfb349af8fab32',2504425555,root/'runtime.tar.gz')
 for p in ('/opt/cooptrack','/usr/local/cuda-11.8/targets/x86_64-linux/lib'):
  if Path(p).exists():raise FileExistsError('runtime destination must be fresh')
 validate_archive(archive,['opt/cooptrack','usr/local/cuda-11.8/targets/x86_64-linux/lib'],allow_links=True)
 subprocess.run(['tar','--no-same-owner','-xzf',str(archive),'-C','/'],check=True)
 dependency=Task.get_task(task_id='1e0730c280e846bebd92cc4de49893e3');assert dependency.status=='completed'
 archive=download_registered_artifact(dependency.artifacts['libgl'],'5ac68c58a292e435a0ee55c98f0bd2720a9b088343afc813c262bdc552cc0e10',1141794,root/'libgl.tar.gz')
 validate_archive(archive,['libgl','manifest.json']);subprocess.run(['tar','--no-same-owner','-xzf',str(archive),'-C',str(root)],check=True)
 if sha256(root/'manifest.json')!='2905061e79777c2858471fa2ee3ccbfaf3cf064fa62429dfa4e1424d34530f82':raise ValueError('dependency manifest changed')
 for r in json.loads((root/'manifest.json').read_bytes())['inventory']:
  p=root/r['path'];assert p.stat().st_size==r['bytes'] and sha256(p)==r['sha256']
 script=root/'worker.py';script.write_text(WORKER)
 env=dict(os.environ);env.pop('VIRTUAL_ENV',None);env.pop('PYTHONHOME',None)
 env.update({'PATH':'/opt/cooptrack/bin:'+env.get('PATH',''),'PYTHONNOUSERSITE':'1','PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','NCCL_P2P_DISABLE':'1','NCCL_IB_DISABLE':'1','NCCL_DEBUG':'INFO','LD_LIBRARY_PATH':str(root/'libgl')+':/opt/cooptrack/lib:/opt/cooptrack/lib/python3.8/site-packages/torch/lib:/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'})
 readiness=subprocess.check_output(['/opt/cooptrack/bin/python','-c',"import sys,torch,json; assert sys.version_info[:2]==(3,8); assert torch.cuda.device_count()==4; print(json.dumps({'python':sys.version,'torch':torch.__version__,'devices':[torch.cuda.get_device_name(i) for i in range(4)]}))"],env=env,text=True)
 print('NCCL_LEGACY_READINESS '+readiness,flush=True)
 cases=[('baseline',{}),('shm-disabled',{'NCCL_SHM_DISABLE':'1'}),('shm-disabled-loopback',{'NCCL_SHM_DISABLE':'1','NCCL_SOCKET_IFNAME':'lo'})];results=[]
 for index,(name,overrides) in enumerate(cases):
  case=root/name;case.mkdir();case_env=dict(env)
  for key in ('NCCL_SHM_DISABLE','NCCL_SOCKET_IFNAME'):case_env.pop(key,None)
  case_env.update(overrides)
  cmd=['/opt/cooptrack/bin/python','-m','torch.distributed.launch','--nproc_per_node=4','--master_port='+str(29580+index),str(script),'--output',str(case)]
  print('NCCL_CASE_STARTED '+json.dumps({'case':name,'environment_overrides':overrides,'eta':'unknown'}),flush=True)
  process=subprocess.Popen(cmd,env=case_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
  selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ);started=time.monotonic();lines=[];timed_out=False
  while process.poll() is None or selector.get_map():
   if time.monotonic()-started>90 and process.poll() is None:
    timed_out=True;os.killpg(process.pid,signal.SIGKILL)
   for key,mask in selector.select(timeout=1):
    line=key.fileobj.readline()
    if not line:selector.unregister(key.fileobj);continue
    lines.append(line);print(line,end='',flush=True)
  status=process.wait();selector.close()
  (case/'console.log').write_text(''.join(lines));ranks=[json.loads(p.read_bytes()) for p in sorted(case.glob('rank-*.json'))]
  accepted=status==0 and len(ranks)==4 and sorted(r['rank'] for r in ranks)==list(range(4)) and all(r['broadcast_passed'] and r['all_reduce_passed'] for r in ranks)
  result={'case':name,'environment_overrides':overrides,'returncode':status,'timeout':timed_out,'ranks':ranks,'communication_accepted':accepted,'elapsed_seconds':time.monotonic()-started};results.append(result)
  task.upload_artifact(name+'-console',artifact_object=case/'console.log',wait_on_upload=True)
  print('NCCL_CASE_FINISHED '+json.dumps(result),flush=True)
 report={'kind':'spd_nccl_gpu4_no_data_runtime_diagnostic_v1','runtime_sha256':'b6b39c66eec6921c0f0f51d2b9cd4e5571fcafdc44d3605112dfb349af8fab32','results':results,'gt_or_dataset_loaded':False,'model_or_optimizer_created':False,'training_accepted':False,'paper_eligible':False,'worker_source_sha256':sha256(script),'controller_source_sha256':sha256(Path(__file__)),'legacy_readiness':json.loads(readiness),'supersedes_invalid_probe_task_id':'501461a3f38646c0a8c4f3203e1a1f2b','pythonpath_cleared':True,'pythonhome_removed':True,'physical_worker_id':os.environ.get('CLEARML_WORKER_ID','unknown')}
 p=root/'nccl-diagnostic.json';p.write_text(json.dumps(report,indent=2)+'\n');assert task.upload_artifact('nccl-diagnostic',artifact_object=p,wait_on_upload=True)
 print('NCCL_DIAGNOSTIC_COMPLETE '+json.dumps({'communication_accepted_cases':[r['case'] for r in results if r['communication_accepted']],'training_accepted':False}),flush=True)
if __name__=='__main__':main()
