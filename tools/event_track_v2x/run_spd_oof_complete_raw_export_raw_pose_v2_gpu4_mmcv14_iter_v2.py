"""Candidate four-GPU complete held-out raw export after sampled forward admission.

Loads runtime/source and GT-free held-out assets only; never downloads fit labels.
This wrapper is not an accepted GPU experiment until its outputs are read back.
"""
import hashlib,json,os,shutil,subprocess,time,tarfile
from pathlib import Path
from clearml import Task
from run_cooptrack_official_oof_gpu4_offline_gl_v6 import download_registered_artifact,validate_archive,validate_cohort

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def download(task,name,record,path):
 require(task.status=='completed','artifact owner not completed')
 item=task.artifacts[name];require(item.hash==record['sha256'] and item.size==record['bytes'],'artifact registration differs')
 return download_registered_artifact(item,record['sha256'],record['bytes'],path)

def main():
 task=Task.init(project_name='Thesis/EventTrack-V2X/Training',task_name='SPD canonical OOF complete held-out raw export raw-pose-v2 GPU4',reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False)
 p={k.split('/',1)[-1]:v for k,v in task.get_parameters().items() if k.startswith('General/')}
 raw=p['byte_freeze_text'].encode();require(hashlib.sha256(raw).hexdigest()==p['byte_freeze_sha256'],'byte freeze text differs')
 frozen=json.loads(raw);require(frozen['kind']=='spd_canonical_oof_detector_independent_byte_freeze_v1' and frozen['byte_freeze_accepted'] is True,'training byte freeze not accepted')
 training=Task.get_task(task_id=frozen['task_id']);require(training.status=='completed','training still incomplete')
 require(hashlib.sha256(training.data.script.diff.encode()).hexdigest()==frozen['controller_sha256'],'training controller differs')
 params=training.get_parameters();package=Task.get_task(task_id=params['General/package_task_id'])
 require(params['General/package_manifest_sha256']==frozen['package_manifest_sha256'] and params['General/package_independent_readback_sha256']==frozen['package_independent_readback_sha256'],'fit package provenance differs')
 admitted=p['forward_acceptance_text'].encode();require(hashlib.sha256(admitted).hexdigest()==p['forward_acceptance_sha256'],'forward admission bytes differ')
 acceptance=json.loads(admitted);require(acceptance['kind']=='spd_canonical_oof_gpu4_sampled_forward_independent_readback_v1' and acceptance['status']=='independent_bytes_and_sampled_tensor_forward_verified' and acceptance['training_task_id']==training.id and acceptance['byte_freeze_sha256']==p['byte_freeze_sha256'] and acceptance['fold_id']==frozen['fold_id'] and acceptance['seed']==frozen['seed'] and acceptance['acceptor_sha256']=='5867c8477c2d038db39bea3903972c27a2b9c11236c6bc935d62c1ee896f4522','independent sampled forward not admitted')
 forward=Task.get_task(task_id=acceptance['task_id']);require(forward.status=='completed','sampled forward task not completed')
 forward_params=forward.get_parameters();require(forward_params['General/byte_freeze_sha256']==p['byte_freeze_sha256'] and forward_params['General/verifier_sha256']==acceptance['verifier_sha256'],'sampled forward configuration differs')
 require(acceptance['verifier_sha256']=='d7bc1571454521b94bf677a2f5e22acd95734bd08c58ee4154ffa244924fb1d3' and forward_params['General/source_task_id']=='2652410a467f49d982115537a3ceca5a' and forward_params['General/source_manifest_sha256']=='22540575808bc6cbad33f2c33a42b3972f112b8a1a7bb0a9d9eb3ca1f88ef1af' and forward_params['General/execution_controller_sha256']=='3c51ccc6f871a67bbd2428078f90f9341786a6793c379bb89c951bb5a0cffc36','forward execution source differs')
 require(hashlib.sha256(forward.data.script.diff.encode()).hexdigest()=='db38ca815f480a6499957003e6c0df9ecf7011312ba5887d5730973fb7062ca5','forward bootstrap differs')
 expected={'sampled-forward-summary'}
 for side in ('vehicle-side','infrastructure-side'):
  for k in (0,1):expected.update((side+'-shard-%d-forward'%k,side+'-shard-%d-forward-log'%k))
 require(set(acceptance['artifacts'])==expected,'forward artifact admission incomplete')
 require(len(acceptance['shards'])==4 and {(r['side'],r['shard_index']) for r in acceptance['shards']}=={(side,k) for side in ('vehicle-side','infrastructure-side') for k in (0,1)},'forward admission lacks four shards')
 for name,row in acceptance['artifacts'].items():
  item=forward.artifacts[name];require(item.hash==row['sha256'] and item.size==row['bytes'],'forward accepted artifact registration differs')
 root=Path('/eventtrack-oof-raw-pose-v2');root.mkdir();downloads=root/'downloads';downloads.mkdir()
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
 producer=Path(__file__).with_name('run_spd_canonical_oof_raw_cache_raw_pose_v2.py');require(sha(producer)==p['producer_sha256'],'raw producer source differs')
 cache_verifier=Path(__file__).with_name('spd_canonical_oof_cache_primitives.py');require(sha(cache_verifier)==p['cache_verifier_sha256'],'cache verifier source differs')
 pose_helper=Path(__file__).with_name('spd_canonical_oof_raw_pose_metadata_v2.py');require(sha(pose_helper)==p['pose_metadata_helper_sha256'],'raw pose metadata helper differs')
 children=[]
 for index,(side,shard) in enumerate((s,k) for s in ('vehicle-side','infrastructure-side') for k in (0,1)):
  out=root/('%s-shard-%d-cache'%(side,shard));log=root/('%s-shard-%d-cache.log'%(side,shard));child_env=dict(env,CUDA_VISIBLE_DEVICES=devices[index]);a=frozen['artifacts']
  command=[python,str(producer),'--inputs',str(root/'inputs'),'--upstream','/workspace/CoopTrack','--package-manifest',str(mp),'--byte-freeze',str(bf),'--checkpoint',str(evidence/a[side+'-final-checkpoint']['path']),'--appearance-checkpoint','/pretrained/resnet50.pth','--training-config',str(evidence/a[side+'-detector.py']['path']),'--training-launch',str(evidence/a[side+'-launch-receipt.json']['path']),'--training-startup',str(evidence/a[side+'-optimizer-startup.json']['path']),'--training-completion',str(evidence/a[side+'-completion']['path']),'--output',str(out),'--checkpoint-sha256',a[side+'-final-checkpoint']['sha256'],'--package-manifest-sha256',frozen['package_manifest_sha256'],'--byte-freeze-sha256',p['byte_freeze_sha256'],'--input-manifest-sha256',held['input_manifest_sha256'],'--seed',str(frozen['seed']),'--side',side,'--shard-index',str(shard)]
  handle=log.open('w');process=subprocess.Popen(command,env=child_env,stdout=handle,stderr=subprocess.STDOUT);children.append((process,handle,log,out,side,shard));print('OOF_RAW_CHILD_STARTED',side,shard,'assigned_cuda',devices[index],flush=True)
 started=time.monotonic();offsets={str(c[2]):0 for c in children}
 while any(c[0].poll() is None for c in children):
  for process,handle,log,out,side,shard in children:
   with log.open(errors='replace') as stream:
    stream.seek(offsets[str(log)]);text=stream.read();offsets[str(log)]=stream.tell()
    for line in text.splitlines():
     if line.startswith('EVENTTRACK_CACHE_PROGRESS '):print('OOF_RAW_SHARD_PROGRESS',side,shard,line.strip(),flush=True)
  print('EVENTTRACK_PHASE_ETA '+json.dumps({'phase':'complete-heldout-raw-export','finished_shards':sum(c[0].poll() is not None for c in children),'total_shards':4,'elapsed_seconds':time.monotonic()-started,'eta_seconds':None,'eta_status':'unknown','overall_eta':'unknown'}),flush=True);time.sleep(15)
 for process,handle,log,out,side,shard in children:
  handle.close();require(task.upload_artifact(side+'-shard-%d-export-log'%shard,artifact_object=log,wait_on_upload=True),'log publication failed')
 require(all(c[0].returncode==0 and (c[3]/'raw-cache-manifest.json').is_file() for c in children),'raw export failed; preserve logs')
 inputs=json.loads((root/'inputs/input-manifest.json').read_bytes());records=[];totals={side:0 for side in ('vehicle-side','infrastructure-side')}
 for process,handle,log,out,side,shard in children:
  # Reopen every array/metadata in an independent CPU process again at publication.
  verified=json.loads(subprocess.check_output([python,str(cache_verifier),str(out),'--inputs',str(root/'inputs'),'--package-manifest',str(mp)],env=env,text=True))
  cm=json.loads((out/'raw-cache-manifest.json').read_bytes())
  require(verified['all_payloads_read'] is True and verified['expected_frame_coverage_verified'] is True and verified['manifest_sha256']==sha(out/'raw-cache-manifest.json') and verified['frames_verified']==cm['frame_count'] and verified['detections_verified']==cm['detection_count'],'independent cache readback differs')
  require(cm['fold_id']==frozen['fold_id'] and cm['seed']==frozen['seed'] and cm['side']==side and cm['shard_index']==shard and cm['shard_count']==2 and cm['sequences']==sorted(inputs['train_sequences'])[shard::2] and cm['byte_freeze_sha256']==p['byte_freeze_sha256'] and cm['checkpoint_sha256']==frozen['artifacts'][side+'-final-checkpoint']['sha256'] and cm['metadata_pose_source']=='raw-calibration-composition-float64-v2' and cm['pose_metadata_helper_sha256']==p['pose_metadata_helper_sha256'] and cm['producer_code_sha256']==p['producer_sha256'] and cm['verifier_code_sha256']==p['cache_verifier_sha256'] and cm['weights_unchanged'] is True and cm['raw_head_frames_verified']==cm['frame_count'] and cm['covariance_calibrated'] is False and cm['formal_v2_ready'] is False,'raw completion scope/identity differs')
  totals[side]+=cm['frame_count'];archive=root/('%s-shard-%d-cache.tar.gz'%(side,shard))
  files=sorted(x for x in out.rglob('*') if x.is_file());require(not any(x.is_symlink() for x in out.rglob('*')),'cache symlink forbidden')
  print('EVENTTRACK_PHASE_ETA '+json.dumps({'phase':'cache-archive-publication','side':side,'shard':shard,'files':len(files),'eta_seconds':None,'eta_status':'unknown'}),flush=True)
  with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
   for file in files:tar.add(file,arcname='cache/'+file.relative_to(out).as_posix(),recursive=False)
  # Verify the archive contains exactly the already-read payload bytes.
  expected={'cache/'+file.relative_to(out).as_posix():{'sha256':sha(file),'bytes':file.stat().st_size} for file in files}
  seen=set()
  with tarfile.open(archive,'r:gz') as tar:
   for member in tar:
    require(member.isfile() and member.name in expected and member.name not in seen,'archive inventory differs')
    stream=tar.extractfile(member);h=hashlib.sha256();count=0
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block);count+=len(block)
    require(count==expected[member.name]['bytes'] and h.hexdigest()==expected[member.name]['sha256'],'archive payload differs');seen.add(member.name)
  require(seen==set(expected),'archive missing files')
  prefix=side+'-shard-%d'%shard
  readback={'kind':'spd_canonical_oof_raw_shard_cpu_and_archive_readback_v1','side':side,'shard_index':shard,'cache':verified,'archive':{'sha256':sha(archive),'bytes':archive.stat().st_size},'archive_files':len(files),'all_archive_members_read':True,'formal_v2_ready':False}
  receipt=root/(prefix+'-readback.json');receipt.write_text(json.dumps(readback,indent=2)+'\n')
  for name,file in ((prefix+'-raw-cache',archive),(prefix+'-raw-manifest',out/'raw-cache-manifest.json'),(prefix+'-raw-readback',receipt)):
   require(task.upload_artifact(name,artifact_object=file,wait_on_upload=True),'cache artifact upload failed')
  records.append(dict(readback,manifest_sha256=sha(out/'raw-cache-manifest.json'),actual_device=cm['physical_device']))
 require(totals==inputs['frames'],'both side shard totals differ from full held-out input')
 summary={'kind':'spd_canonical_oof_gpu4_complete_raw_export_raw_pose_v2_summary','training_task_id':training.id,'forward_task_id':forward.id,'forward_acceptance_sha256':p['forward_acceptance_sha256'],'byte_freeze_sha256':p['byte_freeze_sha256'],'fold_id':frozen['fold_id'],'seed':frozen['seed'],'actual_devices':ready['devices'],'cuda_visible_devices':devices,'shards':records,'frame_totals':totals,'metadata_pose_source':'raw-calibration-composition-float64-v2','pose_metadata_helper_sha256':p['pose_metadata_helper_sha256'],'job_local_all_payloads_and_archive_bytes_verified':True,'independent_cloud_readback_verified':False,'train_labels_downloaded':False,'covariance_calibrated':False,'formal_v2_ready':False,'paper_eligible':False}
 require(task.upload_artifact('complete-raw-export-summary',artifact_object=summary,wait_on_upload=True),'summary publication failed');task.close()

if __name__=='__main__':main()
