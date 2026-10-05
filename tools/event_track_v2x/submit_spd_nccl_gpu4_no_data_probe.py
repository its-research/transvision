"""Deduplicate and queue one bounded communication diagnostic on compatible GPUs."""
import hashlib,json,re
from pathlib import Path
from clearml import Task
from clearml.backend_api.session.client import APIClient
from submit_spd_official_oof_fold_gpu4_offline_gl_v6 import eligible_workers,DOCKER
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
def main():
 source=Path(__file__).with_name('run_spd_nccl_gpu4_no_data_probe.py');raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 freeze=ROOT/'source-freezes/spd-nccl-gpu4-no-data-probe-20261001'/source.name
 if freeze.read_bytes()!=raw:raise ValueError('diagnostic differs from frozen source')
 name='SPD four-GPU no-data NCCL diagnostic '+digest[:12];project='Thesis/EventTrack-V2X/Training'
 same=Task.get_tasks(project_name=project,task_name='^'+re.escape(name)+'$')
 if same:
  if len(same)!=1:raise ValueError('duplicate diagnostic tasks')
  print('EXISTING_NCCL_PROBE',same[0].id,same[0].status);return
 candidates=eligible_workers(APIClient(),'GPU4-V100')
 if not candidates:raise RuntimeError('no collision-free four-GPU V100 worker')
 task=Task.create(project_name=project,task_name=name,task_type=Task.TaskTypes.testing,binary='python3.12')
 task.set_script(repository='',branch='',commit='',working_dir='.',entry_point=source.name,diff=raw.decode())
 task.set_packages(['clearml==2.1.2'])
 task.set_base_docker(DOCKER,docker_arguments='--shm-size 32g -e CLEARML_AGENT_FORCE_TASK_INIT=0 -e NVIDIA_DRIVER_CAPABILITIES=compute -e CLEARML_FILES_HOST=http://10.100.34.118:8081')
 task.set_parameters({'controller_sha256':digest,'gt_or_dataset_loaded':False,'model_or_optimizer_created':False,'world_size':4,'cases':['baseline','shm-disabled','shm-disabled-loopback'],'failed_training_task_id':'d0d2b630acdb47adbdb297173d687d9f','paper_eligible':False})
 task.add_tags(['SPD','runtime-probe','NCCL','no-data','not-training'])
 receipt={'kind':'spd_nccl_gpu4_no_data_probe_dispatch_v1','task_id':task.id,'controller_sha256':digest,'eligible_physical_workers_at_dispatch':candidates,'queue':'GPU4-V100','communication_accepted':False,'training_accepted':False}
 p=ROOT/'receipts/spd-nccl-gpu4-no-data-probe-dispatch-20261001.json'
 with p.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 Task.enqueue(task,queue_name='GPU4-V100');current=Task.get_task(task_id=task.id)
 if current.status not in ('queued','in_progress'):raise RuntimeError('diagnostic enqueue not confirmed')
 print('NCCL_PROBE_ENQUEUED',task.id,current.status,flush=True)
if __name__=='__main__':main()
