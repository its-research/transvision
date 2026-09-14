#!/usr/bin/env python3
"""One independent priority-model seed on one exactly-four-GPU ClearML worker.

Standalone bootstrap; uploads only this task's newly trained priority weights,
checkpoint and bounded training evidence. No V100, CPU, GT/prediction uploads,
alternate server, missing full-teacher receipt or successful-status shortcut.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import urlparse

HOST='10.100.35.118'
FIT_CONFIG=dict(epochs=10,hidden=32,batch_groups=64,learning_rate=.001,gradient_clip=5.)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def parameters(values):
    result={}
    for name,value in values.items():
        name=name.removeprefix('General/')
        if name in result:raise ValueError('ambiguous task parameter')
        result[name]=value
    for name,allowed in (('seed',('1337','2027','3407')),('required_world_size',('4',)),
            ('global_batch_groups',('64',)),('epochs',('10',))):
        if type(result.get(name)) not in (str,int) or str(result[name]) not in allowed:
            raise ValueError('priority task parameter differs: '+name)
        result[name]=int(result[name])
    if result.get('class_scope')!='car' or result.get('gpu_family') not in ('A100','5090'):
        raise ValueError('car-only A100/5090 assignment required')
    if (re.fullmatch(r'[0-9a-f]{32}',result.get('package_task_id','')) is None
            or any(re.fullmatch(r'[0-9a-f]{64}',result.get(k,'')) is None
                   for k in ('package_sha256','controller_sha256'))):
        raise ValueError('pinned package task and manifest required')
    return result


def check_package(meta):
    if (meta.get('kind')!='component_priority_ddp_package_v1' or meta.get('split')!='train'
            or meta.get('class_scope')!=['car'] or meta.get('full_official_train_trace') is not True
            or meta.get('teacher_frames')!=7445 or meta.get('teacher_sequences')!=46
            or any(meta.get(k) is not False for k in ('raw_GT_included','teacher_traces_included',
                'predictions_included','original_detection_stream_included','trained_models_included'))):
        raise ValueError('verified complete car priority teacher package required')
    records=meta.get('artifacts',[])
    if len(records)!=2 or {r['path'] for r in records}!={'source.tar.gz','priority-groups.tar.gz'}:
        raise ValueError('exact source and numerical priority artifacts required')
    if any(type(r['bytes']) is not int or r['bytes']<1 or re.fullmatch(r'[0-9a-f]{64}',r['sha256']) is None for r in records):
        raise ValueError('invalid bounded artifact identity')


def download(artifact):
    uri=urlparse(artifact.url)
    if (uri.scheme,uri.hostname,uri.port)!=('http',HOST,8081):
        raise ValueError('only designated private file server allowed')
    path=artifact.get_local_copy()
    if not path:raise RuntimeError('authenticated package download failed')
    return Path(path)


def extract_checked(path,destination,inventory,max_bytes):
    expected={r['path']:r for r in inventory}
    if len(expected)!=len(inventory) or not expected or len(expected)>1000:
        raise ValueError('distinct bounded inventory required')
    with tarfile.open(path,'r:gz') as archive:
        seen=set();total=0
        for item in archive.getmembers():
            name=PurePosixPath(item.name);total+=item.size
            if (not item.isfile() or name.is_absolute() or '..' in name.parts or '\\' in item.name
                    or item.name in seen or item.name not in expected or item.size!=expected[item.name]['bytes']):
                raise ValueError('unsafe, duplicate or unexpected archive member')
            seen.add(item.name)
            if total>max_bytes:raise ValueError('archive exceeds declared unpacked limit')
        if seen!=set(expected):raise ValueError('archive omitted an inventory file')
        destination.mkdir();archive.extractall(destination,filter='data')
    if any(sha(destination/name)!=r['sha256'] for name,r in expected.items()):
        raise ValueError('unpacked inventory identity differs')


def validate_result(output,metadata,seed,family,*,expected_devices=None):
    """Called only after checked torchrun exit; validates every saved epoch."""
    output=Path(output);receipt=json.loads((output/'receipt.json').read_bytes())
    plan=json.loads((output/'plan.json').read_bytes())
    directory=output/str(seed);cp=json.loads((directory/'checkpoint.json').read_bytes())
    if (receipt.get('kind')!='component_priority_ddp_receipt_v1' or receipt.get('status')!='complete'
            or receipt.get('seed')!=seed or receipt.get('world_size')!=4
            or receipt.get('full_official_train_trace') is not True or receipt.get('local_fixture_only') is not False
            or receipt['plan_sha256']!=sha(output/'plan.json') or cp['plan_sha256']!=receipt['plan_sha256']
            or receipt['checkpoint_manifest']!=str(seed)+'/checkpoint.json'
            or receipt['checkpoint_sha256']!=sha(directory/'checkpoint.json')
            or receipt['epochs_sha256']!=sha(directory/'epochs.jsonl')
            or cp['weights_sha256']!=sha(directory/'weights.npz')):
        raise ValueError('complete hash-bound four-rank priority result required')
    if (plan['fit_config']!=FIT_CONFIG or plan['seed']!=seed
            or plan['training_manifest_sha256']!=metadata['training_manifest_sha256']
            or cp['training_manifest_sha256']!=metadata['training_manifest_sha256']
            or plan['training_source_sha256']!=metadata['source_sha256']
            or cp['training_source_sha256']!=metadata['source_sha256']
            or plan['statistics']!=metadata['statistics'] or cp['seed']!=seed
            or plan['binding']!=metadata['binding'] or cp['binding']!=metadata['binding']
            or plan['fit_sequences']!=metadata['fit_sequences'] or plan['holdout_sequences']!=metadata['holdout_sequences']
            or cp['fit_sequences']!=plan['fit_sequences'] or cp['holdout_sequences']!=plan['holdout_sequences']
            or cp['initial_policy_signature']==cp['policy_signature'] or cp['fixed_final_epoch']!=10
            or cp['distributed_world_size']!=4 or cp['distributed_backend']!='nccl'):
        raise ValueError('priority training configuration or weights did not match assignment')
    from tools.event_track_v2x.train_allocation_policy_ddp import validate_runtime
    from transvision.models.event_track_v2x.allocation_training import load_priority
    validate_runtime(receipt['rank_runtime'],require_full_train=True)
    if (receipt['rank_runtime']!=plan['rank_runtime'] or any(family not in r['gpu_name'] for r in receipt['rank_runtime'])):
        raise ValueError('priority result used another GPU assignment')
    if expected_devices is not None and {r['gpu_uuid'] for r in receipt['rank_runtime']}!={r['uuid'] for r in expected_devices}:
        raise ValueError('priority result GPU UUIDs differ from actual preflight')
    frozen,_=load_priority(directory,receipt['checkpoint_sha256'],binding=metadata['binding'])
    if frozen.signature!=cp['policy_signature'] or frozen.weights[0].shape!=(FIT_CONFIG['hidden'],18):
        raise ValueError('saved priority weights differ')
    epochs=[json.loads(line) for line in (directory/'epochs.jsonl').read_bytes().splitlines()]
    if [e['epoch'] for e in epochs]!=list(range(1,11)):raise ValueError('all ten training epochs required')
    for epoch in epochs:
        ranks=epoch['rank_progress'];fit=metadata['statistics']['fit'];held=metadata['statistics']['holdout']
        batches=(fit['groups']+63)//64
        if (len(ranks)!=4 or sorted(r['rank'] for r in ranks)!=list(range(4))
                or any(r['batches']!=batches or r['nonempty_batches']<1 or r['positive_gradient_batches']<1 for r in ranks)
                or epoch['global_batches']!=batches or epoch['fit_groups']!=fit['groups']
                or epoch['holdout_groups']!=held['groups'] or len({r['policy_signature'] for r in ranks})!=1
                or sum(r['fit_groups'] for r in ranks)!=fit['groups'] or sum(r['fit_rows'] for r in ranks)!=fit['rows']
                or sum(r['holdout_groups'] for r in ranks)!=held['groups'] or sum(r['holdout_rows'] for r in ranks)!=held['rows']
                or epoch['official_validation_or_test_read'] is not False
                or any(not math.isfinite(epoch[k]) or epoch[k]<0 for k in ('training_mse','train_sequence_holdout_mse'))):
            raise ValueError('epoch group coverage or actual synchronized rank work differs')
        for key,loss,count in (('training_mse','group_loss_sum',fit['groups']),
                              ('train_sequence_holdout_mse','holdout_group_loss_sum',held['groups'])):
            if (any(not math.isfinite(r[loss]) or r[loss]<0 for r in ranks)
                    or not math.isclose(epoch[key],sum(r[loss] for r in ranks)/count,rel_tol=1e-12,abs_tol=1e-12)):
                raise ValueError('epoch objective differs from actual rank loss sums')
    if epochs[-1]['rank_progress'][0]['policy_signature']!=frozen.signature:
        raise ValueError('checkpoint is not the final synchronized epoch')
    return [('training-receipt',output/'receipt.json'),('training-plan',output/'plan.json'),
        ('epoch-progress',directory/'epochs.jsonl'),('checkpoint-manifest',directory/'checkpoint.json'),
        ('priority-checkpoint',directory/'weights.npz')]


def main():
    os.environ['CLEARML_API_HOST']='http://'+HOST+':8008'
    os.environ['CLEARML_FILES_HOST']='http://'+HOST+':8081'
    for port in (8008,8081):
        with socket.create_connection((HOST,port),timeout=5):pass
    from clearml import Task
    import torch
    task=Task.init(project_name='Thesis/Recover-Before-Fuse/Training',task_name='four-GPU priority',
        reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False)
    params=parameters(task.get_parameters());seed=params['seed'];family=params['gpu_family']
    if sha(__file__)!=params['controller_sha256']:raise ValueError('executed priority controller source differs')
    if torch.cuda.device_count()!=4 or not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
        raise ValueError('exactly four allocated CUDA GPUs and NCCL required')
    devices=[]
    for index in range(4):
        props=torch.cuda.get_device_properties(index)
        if family not in props.name:raise ValueError('assigned GPU family differs')
        with torch.cuda.device(index):
            x=torch.ones((128,128),device='cuda');y=x@x;torch.cuda.synchronize()
            if not bool(torch.isfinite(y).all()) or float(y[0,0])!=128.:raise ValueError('GPU arithmetic preflight failed')
            devices.append(dict(index=index,name=props.name,uuid=str(getattr(props,'uuid','unavailable'))));del x,y
    if len({r['uuid'] for r in devices})!=4 or any(r['uuid'] in ('None','unavailable') for r in devices):
        raise ValueError('four distinct real GPU UUIDs required')
    print('RBF_PRIORITY_GPU_PREFLIGHT '+json.dumps(devices),flush=True)
    package=Task.get_task(task_id=params['package_task_id']);path=download(package.artifacts['package'])
    if sha(path)!=params['package_sha256']:raise ValueError('priority package manifest changed')
    metadata=json.loads(path.read_bytes());check_package(metadata)
    root=Path(tempfile.mkdtemp(prefix='rbf-priority-ddp-'))
    if shutil.disk_usage(root).free<2*sum(r['bytes'] for r in metadata['data_inventory'])+2*1024**3:
        raise ValueError('insufficient private runtime disk for priority package')
    for record in metadata['artifacts']:
        source=download(package.artifacts[record['path']])
        if source.stat().st_size!=record['bytes'] or sha(source)!=record['sha256']:
            raise ValueError('priority archive differs')
        code=record['path']=='source.tar.gz'
        extract_checked(source,root/('source' if code else 'groups'),
            metadata['source_inventory' if code else 'data_inventory'],16*1024**2 if code else 8*1024**3)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CLEARML_AGENT_FORCE_TASK_INIT='0')
    output=root/'fit'
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
        'tools/event_track_v2x/train_allocation_policy_ddp.py','--data',str(root/'groups'),
        '--manifest-sha256',metadata['training_manifest_sha256'],'--output',str(output),'--seed',str(seed),
        '--epochs','10','--global-batch-groups','64']
    print('RBF_PRIORITY_DDP_LAUNCH '+json.dumps(dict(seed=seed,world_size=4,command=command)),flush=True)
    subprocess.run(command,cwd=root/'source',env=env,check=True)
    sys.path.insert(0,str(root/'source'))
    artifacts=validate_result(output,metadata,seed,family,expected_devices=devices)
    if any(p.stat().st_size>16*1024**2 for _,p in artifacts):raise ValueError('priority output exceeds retention cap')
    for name,path in artifacts:
        if not task.upload_artifact(name,artifact_object=path,wait_on_upload=True):raise RuntimeError('priority result upload failed')
        if Task.get_task(task_id=task.id).artifacts[name].hash!=sha(path):raise RuntimeError('priority upload readback differs')
    print('RBF_PRIORITY_DDP_COMPLETE '+json.dumps(dict(seed=seed,receipt_sha256=sha(output/'receipt.json'),paper_eligible=False)),flush=True)
    task.close()


if __name__=='__main__':main()
