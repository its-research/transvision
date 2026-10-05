"""Candidate four-GPU execution wrapper for completed canonical OOF checkpoints.

Loads runtime/source and GT-free held-out assets only; never downloads fit labels.
This wrapper is not an accepted GPU experiment until its outputs are read back.
"""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from clearml import Task
from run_cooptrack_official_oof_gpu4_offline_gl_v6 import download_registered_artifact,validate_archive,validate_cohort

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def download(task,name,record,path):
 require(task.status=='completed','artifact owner not completed')
 item=task.artifacts[name];require(item.hash==record['sha256'] and item.size==record['bytes'],'artifact registration differs')
 return download_registered_artifact(item,record['sha256'],record['bytes'],path)

def main():
 task=Task.init(project_name='Thesis/EventTrack-V2X/Training',task_name='SPD canonical OOF final checkpoint sampled forward GPU4',reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False)
 p={k.split('/',1)[-1]:v for k,v in task.get_parameters().items() if k.startswith('General/')}
 raw=p['byte_freeze_text'].encode();require(hashlib.sha256(raw).hexdigest()==p['byte_freeze_sha256'],'byte freeze text differs')
 frozen=json.loads(raw);require(frozen['kind']=='spd_canonical_oof_detector_independent_byte_freeze_v1' and frozen['byte_freeze_accepted'] is True,'training byte freeze not accepted')
 training=Task.get_task(task_id=frozen['task_id']);require(training.status=='completed','training still incomplete')
 require(hashlib.sha256(training.data.script.diff.encode()).hexdigest()==frozen['controller_sha256'],'training controller differs')
 params=training.get_parameters();package=Task.get_task(task_id=params['General/package_task_id'])
 require(params['General/package_manifest_sha256']==frozen['package_manifest_sha256'] and params['General/package_independent_readback_sha256']==frozen['package_independent_readback_sha256'],'fit package provenance differs')
 root=Path('/eventtrack-oof-forward');root.mkdir();downloads=root/'downloads';downloads.mkdir()
 bf=root/'byte-freeze.json';bf.write_bytes(raw)
 mp=downloads/'package-manifest.json';item=package.artifacts['package-manifest'];download(package,'package-manifest',{'bytes':item.size,'sha256':frozen['package_manifest_sha256']},mp)
 manifest=json.loads(mp.read_bytes());validate_cohort(manifest);require(manifest['fold_id']==frozen['fold_id'],'wrong fold')
 # Do not fetch train-inputs.tar.gz: inference has no access to fit supervision.
 for item in manifest['inventory']:
  if item['path']=='train-inputs.tar.gz':continue
  require(item['path'] in ('runtime.tar.gz','source.tar.gz','resnet50-0676ba61.pth'),'unexpected runtime artifact')
  owner=Task.get_task(task_id=item['artifact_task_id']) if item.get('artifact_task_id') else package
  name={'runtime.tar.gz':'runtime','source.tar.gz':'source','resnet50-0676ba61.pth':'pretrained'}[item['path']]
  path=download(owner,name,item,downloads/item['path'])
  if name=='pretrained':
   target=Path('/pretrained');require(not target.exists(),'pretrained destination already exists');target.mkdir();shutil.copy2(path,target/'resnet50.pth');continue
  prefixes=['opt/cooptrack','usr/local/cuda-11.8/targets/x86_64-linux/lib'] if name=='runtime' else ['workspace/CoopTrack','entrypoints']
  for destination in prefixes:
   if destination!='usr/local/cuda-11.8/targets/x86_64-linux/lib':require(not Path('/'+destination).exists(),'runtime/source destination already exists')
  validate_archive(path,prefixes,allow_links=name=='runtime');subprocess.run(['tar','--no-same-owner','-xzf',str(path),'-C','/'],check=True)
 cloud=p['cloud_input_evidence_text'].encode();require(hashlib.sha256(cloud).hexdigest()=='eca6277da199e198fbcf07b8098421787c8eca08136c67e05761497106cdada1','cloud input admission differs')
 held=next(r for r in json.loads(cloud)['heldout_folds'] if r['fold_id']==frozen['fold_id']);inputs_task=Task.get_task(task_id=held['task_id'])
 hp=download(inputs_task,'package-manifest',{'sha256':held['package_manifest_sha256'],'bytes':inputs_task.artifacts['package-manifest'].size},downloads/'heldout-package-manifest.json')
 hm=json.loads(hp.read_bytes());require(hm['input_manifest_sha256']==held['input_manifest_sha256'] and hm['archive']==held['archive'] and hm['gt_or_val_test_included'] is False,'held-out archive binding differs')
 ha=download(inputs_task,'heldout-inputs',held['archive'],downloads/'heldout-inputs.tar.gz');validate_archive(ha,['inputs'])
 subprocess.run(['tar','--no-same-owner','-xzf',str(ha),'-C',str(root)],check=True)
 gl=Task.get_task(task_id='1e0730c280e846bebd92cc4de49893e3');lib=root/'offline-gl';lib.mkdir()
 for name,record,file in [('libgl',{'sha256':'5ac68c58a292e435a0ee55c98f0bd2720a9b088343afc813c262bdc552cc0e10','bytes':1141794},'libgl.tar.gz'),('manifest',{'sha256':'2905061e79777c2858471fa2ee3ccbfaf3cf064fa62429dfa4e1424d34530f82','bytes':3005},'manifest.json')]:download(gl,name,record,lib/file)
 validate_archive(lib/'libgl.tar.gz',['libgl','manifest.json']);subprocess.run(['tar','--no-same-owner','-xzf',str(lib/'libgl.tar.gz'),'-C',str(lib)],check=True)
 require(sha(lib/'manifest.json')=='2905061e79777c2858471fa2ee3ccbfaf3cf064fa62429dfa4e1424d34530f82','extracted GL manifest differs')
 for row in json.loads((lib/'manifest.json').read_bytes())['inventory']:
  q=lib/row['path'];require(not q.is_symlink() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'GL payload differs')
 evidence=root/'evidence';evidence.mkdir()
 for name,record in frozen['artifacts'].items():download(training,name,record,evidence/record['path'])
 env=dict(os.environ);env.pop('VIRTUAL_ENV',None);env.update({'PATH':'/opt/cooptrack/bin:'+env.get('PATH',''),'PYTHONPATH':'/workspace/CoopTrack:'+str(Path(__file__).parent),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','LD_LIBRARY_PATH':str(lib/'libgl')+':/opt/cooptrack/lib:/opt/cooptrack/lib/python3.8/site-packages/torch/lib:/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64','MPLCONFIGDIR':'/tmp/eventtrack-forward-matplotlib'})
 python='/opt/cooptrack/bin/python'
 ready=json.loads(subprocess.check_output([python,'-c',"import torch,json;assert torch.cuda.device_count()==4;print(json.dumps({'torch':torch.__version__,'devices':[torch.cuda.get_device_name(i) for i in range(4)]}))"],env=env,text=True))
 devices=env.get('CUDA_VISIBLE_DEVICES','').split(',') if env.get('CUDA_VISIBLE_DEVICES','') else ['0','1','2','3']
 require(len(devices)==4 and len(set(devices))==4,'four unique assigned CUDA devices required')
 verifier=Path(__file__).with_name('verify_spd_canonical_oof_checkpoint_forward.py');require(sha(verifier)==p['verifier_sha256'],'verifier source differs')
 children=[];receipts=[]
 for index,(side,shard) in enumerate((s,k) for s in ('vehicle-side','infrastructure-side') for k in (0,1)):
  out=root/('%s-shard-%d-forward.json'%(side,shard));log=root/('%s-shard-%d-forward.log'%(side,shard));child_env=dict(env,CUDA_VISIBLE_DEVICES=devices[index]);a=frozen['artifacts']
  command=[python,str(verifier),'--inputs',str(root/'inputs'),'--upstream','/workspace/CoopTrack','--package-manifest',str(mp),'--byte-freeze',str(bf),'--checkpoint',str(evidence/a[side+'-final-checkpoint']['path']),'--training-config',str(evidence/a[side+'-detector.py']['path']),'--training-launch',str(evidence/a[side+'-launch-receipt.json']['path']),'--training-startup',str(evidence/a[side+'-optimizer-startup.json']['path']),'--training-completion',str(evidence/a[side+'-completion']['path']),'--output',str(out),'--checkpoint-sha256',a[side+'-final-checkpoint']['sha256'],'--package-manifest-sha256',frozen['package_manifest_sha256'],'--byte-freeze-sha256',p['byte_freeze_sha256'],'--input-manifest-sha256',held['input_manifest_sha256'],'--seed',str(frozen['seed']),'--side',side,'--shard-index',str(shard)]
  handle=log.open('w');process=subprocess.Popen(command,env=child_env,stdout=handle,stderr=subprocess.STDOUT);children.append((process,handle,log,out,side,shard));print('OOF_FORWARD_CHILD_STARTED',side,shard,'assigned_cuda',devices[index],flush=True)
 started=time.monotonic();offsets={str(c[2]):0 for c in children}
 while any(c[0].poll() is None for c in children):
  for process,handle,log,out,side,shard in children:
   with log.open(errors='replace') as stream:
    stream.seek(offsets[str(log)])
    text=stream.read();offsets[str(log)]=stream.tell()
    for line in text.splitlines():
     if line.startswith('OOF_CHECKPOINT_FORWARD_PROGRESS '):print('OOF_FORWARD_SHARD_PROGRESS',side,shard,line.strip(),flush=True)
  print('EVENTTRACK_PHASE_ETA '+json.dumps({'phase':'sampled-checkpoint-forward','finished_shards':sum(c[0].poll() is not None for c in children),'total_shards':4,'elapsed_seconds':time.monotonic()-started,'eta_seconds':None,'eta_status':'unknown','overall_eta':'unknown'}),flush=True);time.sleep(15)
 for process,handle,log,out,side,shard in children:
  handle.close();task.upload_artifact(side+'-shard-%d-forward-log'%shard,artifact_object=log,wait_on_upload=True)
  if out.exists():task.upload_artifact(side+'-shard-%d-forward'%shard,artifact_object=out,wait_on_upload=True)
 require(all(c[0].returncode==0 and c[3].exists() for c in children),'sampled forward failed; retain all four logs')
 for process,handle,log,out,side,shard in children:
  x=json.loads(out.read_bytes());require(x['kind']=='spd_canonical_oof_final_checkpoint_sampled_forward_acceptance_v1' and x['fold_id']==frozen['fold_id'] and x['seed']==frozen['seed'] and x['side']==side and x['shard_index']==shard and x['byte_freeze_sha256']==p['byte_freeze_sha256'] and x['raw_head_forward_verified'] is True and x['strict_state_dict_load_verified'] is True and x['weights_unchanged'] is True and x['tensor_report']['all_state_tensors_finite'] is True and x['verifier_sha256']==p['verifier_sha256'],'forward receipt differs')
  receipts.append({'side':side,'shard_index':shard,'sha256':sha(out),'bytes':out.stat().st_size})
 summary={'kind':'spd_canonical_oof_gpu4_sampled_forward_summary_v1','training_task_id':training.id,'byte_freeze_sha256':p['byte_freeze_sha256'],'fold_id':frozen['fold_id'],'seed':frozen['seed'],'actual_devices':ready['devices'],'cuda_visible_devices':devices,'receipts':receipts,'sampled_tensor_forward_verified':True,'train_labels_downloaded':False,'complete_prediction_coverage_verified':False,'paper_eligible':False}
 task.upload_artifact('sampled-forward-summary',artifact_object=summary,wait_on_upload=True);task.close()

if __name__=='__main__':main()
