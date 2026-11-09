#!/usr/bin/env python3
"""Registered full native train inputs -> GPU selection/refit/row candidates.

Cloud publication, independent input readback and disjoint physical admission
must precede dispatch. Completed execution still needs independent acceptance.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def fetch(spec,path):
    import requests
    from clearml import Task
    from clearml.backend_api.session import Session
    owner=Task.get_task(task_id=spec['task'])
    if str(owner.status)!='completed':raise ValueError('completed registered native input owner required')
    asset=owner.artifacts[spec['key']]
    if asset.hash!=spec['sha256'] or asset.size!=spec['bytes']:raise ValueError('registered native input bytes changed')
    url=asset.url.replace('10.100.35.118:8081','10.100.34.118:8081');done=0;started=time.monotonic()
    with requests.get(url,headers={'Authorization':'Bearer '+Session().token},stream=True,timeout=(10,120)) as response:
        if response.status_code!=200:raise RuntimeError('native input HTTP status '+str(response.status_code))
        with path.open('xb') as stream:
            for block in response.iter_content(1024**2):
                stream.write(block);done+=len(block)
                if done%(32*1024**2)<len(block):print(json.dumps(dict(stage='registered_native_training_input_transfer',
                    completed_bytes=done,total_bytes=spec['bytes'],ETA_seconds=(time.monotonic()-started)*(spec['bytes']-done)/done,
                    ETA_scope='current native input transfer only')),flush=True)
    if done!=spec['bytes'] or sha(path)!=spec['sha256']:raise ValueError('native input transfer byte identity differs')


def unpack(path,destination):
    destination.mkdir()
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        if (len({m.name for m in members})!=len(members) or any(not m.isfile() or Path(m.name).is_absolute()
                or '..' in Path(m.name).parts or Path(m.name).as_posix()!=m.name for m in members)):
            raise ValueError('ordinary unique canonical native training archive required')
        archive.extractall(destination,filter='data')


def physical_runtime(world):
    import torch
    if world not in (4,8) or not torch.cuda.is_available() or torch.cuda.device_count()!=world:
        raise ValueError('exactly four/eight allocated CUDA devices required')
    devices=[]
    for i in range(world):
        p=torch.cuda.get_device_properties(i);uuid=str(getattr(p,'uuid','unavailable'))
        capability=torch.cuda.get_device_capability(i)
        if uuid in ('unavailable','None') or 'L40' in p.name or 'sm_'+str(capability[0])+str(capability[1]) not in torch.cuda.get_arch_list():
            raise ValueError('eligible physical UUID and native CUDA architecture required')
        devices.append(dict(local_device=i,uuid=uuid,name=p.name,capability=list(capability)))
    if len({r['uuid'] for r in devices})!=world:raise ValueError('distinct physical GPUs required')
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True,text=True,capture_output=True)
    assigned={r['uuid'] for r in devices};collisions=[]
    for line in result.stdout.splitlines():
        if not line.strip():continue
        uuid,pid=(v.strip() for v in line.split(','))
        if uuid in assigned and int(pid)!=os.getpid():collisions.append(dict(uuid=uuid,pid=int(pid)))
    if collisions:raise ValueError('pre-existing live compute process intersects allocated physical GPUs')
    return dict(devices=devices,no_preexisting_compute_process_on_allocated_UUIDs=True,
        disjoint_worker_enqueue_admission_remains_separate=True,formal_cost_isolation_claimed=False)


def main():
    from clearml import Task
    task=Task.init(project_name='Thesis/Recover-Before-Fuse/Training',task_name='V2V native joint identity selection full refit',
        reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False,output_uri='http://10.100.34.118:8081')
    params=task.get_parameters();recipe=json.loads(params['General/recipe'])
    if hashlib.sha256(canonical(recipe)).hexdigest()!=params['General/recipe_sha256']:
        raise ValueError('frozen native execution recipe differs')
    if recipe['seed'] not in (1337,2027,3407) or recipe['world_size'] not in (4,8):raise ValueError('frozen native seed/world required')
    if not all(recipe[k] is True for k in ('cloud_input_bytes_independently_read','disjoint_physical_worker_admitted')):
        raise ValueError('independent cloud input readback and physical worker admission required')
    if sha(__file__)!=recipe['consumer_sha256']:raise ValueError('frozen GPU consumer source differs')
    base=Path('v2v4real-native-joint-identity');base.mkdir();output=base/'training'
    (base/'recipe.json').write_bytes(canonical(recipe)+b'\n')
    try:
        archive=base/'input-bundle.tar.gz';fetch(recipe['input_bundle'],archive);inputs=base/'inputs';unpack(archive,inputs)
        plan_path=inputs/'bundle-plan.json'
        if sha(plan_path)!=recipe['bundle_plan_sha256']:raise ValueError('frozen native bundle plan differs')
        plan=json.loads(plan_path.read_bytes());semantic=plan['semantic'];identity=hashlib.sha256(canonical(semantic)).hexdigest()
        if (plan['kind']!='v2v4real_native_joint_identity_admitted_GPU_input_bundle_v1'
                or identity!=plan['semantic_training_identity'] or identity!=recipe['semantic_training_identity']
                or semantic['seed']!=recipe['seed'] or semantic['dataset']!='v2v4real'
                or plan['raw_YAML_or_GT_boxes_or_IDs_or_official_test_arrays_in_bundle'] is not False
                or plan['blocked_priority_or_CUDA_branch_state_package_included'] is not False):
            raise ValueError('training-only native bundle semantic identity differs')
        expected={'bundle-plan.json','row-admission.json','partition.json','rows/manifest.json'}
        manifest=json.loads((inputs/'rows/manifest.json').read_bytes())
        for name,key in [('rows/manifest.json','manifest_sha256'),('row-admission.json','row_admission_sha256'),('partition.json','partition_sha256')]:
            if sha(inputs/name)!=semantic[key]:raise ValueError('native admitted declaration bytes differ')
        for name,h in semantic['source_sha256'].items():
            expected.add('source/'+name)
            if sha(inputs/'source'/name)!=h:raise ValueError('native training/scoring source bytes differ')
        for record in manifest['shards']:
            name='rows/'+record['path'];expected.add(name)
            if Path(record['path']).is_absolute() or '..' in Path(record['path']).parts or sha(inputs/name)!=record['sha256']:
                raise ValueError('complete native training shard bytes differ')
        actual={p.relative_to(inputs).as_posix() for p in inputs.rglob('*') if p.is_file()}
        if actual!=expected:raise ValueError('unexpected native GPU input payloads')
        runtime=physical_runtime(recipe['world_size'])
        (base/'runtime-before-launch.json').write_text(json.dumps(runtime,indent=2)+'\n')
        os.environ.update(PYTHONDONTWRITEBYTECODE='1',NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NCCL_P2P_DISABLE='1')
        runner=inputs/'source/tools/event_track_v2x/train_v2v4real_joint_identity.py'
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(recipe['world_size']),str(runner),
            '--seed',str(recipe['seed']),'--data',str(inputs/'rows'),'--manifest-sha256',semantic['manifest_sha256'],
            '--row-admission',str(inputs/'row-admission.json'),'--row-admission-sha256',semantic['row_admission_sha256'],
            '--partition',str(inputs/'partition.json'),'--partition-sha256',semantic['partition_sha256'],'--output',str(output)]
        result=subprocess.run(command)
        if result.returncode:raise RuntimeError('native selection/refit runner exited '+str(result.returncode))
        receipt=json.loads((output/'receipt.json').read_bytes())
        if receipt['completed'] is not True or receipt['seed']!=recipe['seed']:raise ValueError('complete native candidate receipt required')
        package=base/'identity-training.tar.gz'
        with tarfile.open(package,'x:gz') as archive:archive.add(output,arcname='training')
        files={'identity-training':package,'receipt':output/'receipt.json','checkpoint':output/'final/checkpoint.json',
            'plan':output/'plan-before-updates.json','recipe':base/'recipe.json','runtime-before-launch':base/'runtime-before-launch.json'}
        for name,p in files.items():
            if not task.upload_artifact(name,artifact_object=p,wait_on_upload=True):raise RuntimeError('native candidate artifact upload did not complete')
        print(json.dumps(dict(stage='native_joint_identity_candidate_execution_uploaded',seed=recipe['seed'],
            semantic_training_identity=identity,checkpoint_independently_qualified=False,paper_metrics_complete=False)),flush=True)
        task.close()
    except BaseException as e:
        failure=base/'GPU-execution-failure.json'
        with failure.open('x') as stream:stream.write(json.dumps(dict(error_type=type(e).__name__,error=str(e),recipe=recipe,
            independent_checkpoint_accepted=False,partials_and_failures_preserved=True),indent=2)+'\n')
        task.upload_artifact('failure',artifact_object=failure,wait_on_upload=True)
        raise


if __name__=='__main__':main()
