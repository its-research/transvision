#!/usr/bin/env python3
"""Candidate Linux GPU execution of the pinned original official-test schedule."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlparse
from tools.event_track_v2x import run_v2v4real_official_test_raw_features as export
from tools.event_track_v2x import merge_v2v4real_official_test_raw_shards as merger

PROJECTION_ARCHIVE_SHA='e1e0e3652425d351be86983946bf46b6e7c0ac8529a768dced936ef93f420f8d'
SOURCE_TASK='474da80b76cb48279dad699d6d5e1c82'
CHECKPOINT_TASKS={1337:'64f21d6043654a6192f1ce0809b60c1e',2027:'3ba32ad9592a412a8fb4d3ec6eb15faa',3407:'7c72f02af4704341aaad866f7bad8c15'}


def download(owner,name,digest,path):
    """Use only registered artifact URLs and verify the full streamed bytes."""
    import requests
    from clearml.backend_api.session import Session
    if str(owner.status)!='completed':raise ValueError('completed input task required')
    item=owner.artifacts[name];url=urlparse(item.url)
    if (item.hash!=digest or url.scheme!='http' or url.netloc not in {'10.100.34.118:8081','10.100.35.118:8081'}
            or url.username or url.password or url.query or url.fragment):
        raise ValueError('registered input identity or file service differs')
    path=Path(path);h=hashlib.sha256();count=0;start=time.monotonic();last=start
    # The GPU worker LAN reaches the same registered file service on its .34
    # NIC. Keep the exact artifact path; registered size and SHA remain binding.
    transport_url=url._replace(netloc='10.100.34.118:8081').geturl()
    # The SDK storage helper does not reliably attach the local ClearML token
    # to these file-service hosts. Never forward that token on redirects.
    with requests.get(transport_url,headers={'Authorization':'Bearer '+Session().token},
                      stream=True,timeout=(10,120),allow_redirects=False) as response:
        if response.status_code != 200:
            raise ValueError('registered file service HTTP status '+str(response.status_code))
        with path.open('xb') as out:
          for block in response.iter_content(chunk_size=8*1024**2):
            if not block:continue
            out.write(block);h.update(block);count+=len(block)
            if count>item.size:raise ValueError('registered artifact size exceeded')
            now=time.monotonic()
            if now-last>=15:
                print(json.dumps(dict(stage='official_test_input_transfer',artifact=name,completed_bytes=count,
                    total_bytes=item.size,ETA_seconds=(now-start)/count*(item.size-count),eta_scope='this transfer only')),flush=True)
                last=now
    if count!=item.size or h.hexdigest()!=digest:raise ValueError('registered input bytes differ')
    return dict(task_id=owner.id,artifact=name,bytes=count,sha256=digest,
                registered_file_host=url.netloc,transport_file_host='10.100.34.118:8081')


PUBLICATION_SHA='530f97f1d74a72dd7979a3999190b7e64972a3455d39d88795bc4061a8809231'


def unpack_projection(archive,directory,publication_receipt):
    """Extract regular members only, then verify every original publication entry."""
    export.asset(archive,PROJECTION_ARCHIVE_SHA)
    export.asset(publication_receipt,PUBLICATION_SHA)
    publication=json.loads(Path(publication_receipt).read_bytes())
    if (publication.get('kind')!='v2v4real_formal_projection_publication_v1'
            or publication.get('split')!='official_test' or publication.get('gt_in_projection') is not False
            or publication.get('projection_manifest_sha256')!=export.PROJECTION_SHA
            or publication.get('split_admission_sha256')!=export.ADMISSION_SHA):
        raise ValueError('original GT-free official-test publication required')
    inventory={'projection/'+r['path']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in publication['inventory']}
    if len(inventory)!=len(publication['inventory']):raise ValueError('duplicate publication inventory')
    receipt_member='projection/publication-receipt.json'
    if receipt_member in inventory:raise ValueError('publication inventory must not contain itself')
    # Original publisher writes the receipt before adding its byte identity to
    # the tar inventory. Its JSON cannot recursively list its own hash.
    inventory[receipt_member]=dict(bytes=Path(publication_receipt).stat().st_size,sha256=PUBLICATION_SHA)
    directory=Path(directory);directory.mkdir();seen={}
    with tarfile.open(archive) as stream:
        for member in stream:
            name=PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or '..' in name.parts or member.name in seen
                    or member.name not in inventory or member.size!=inventory[member.name]['bytes']):
                raise ValueError('unsafe projection member')
            target=directory/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();count=0
            with target.open('xb') as out:
                data=stream.extractfile(member)
                for block in iter(lambda:data.read(8*1024**2),b''):
                    count+=len(block);h.update(block);out.write(block)
            if count!=member.size:raise ValueError('truncated projection member')
            seen[member.name]=dict(bytes=count,sha256=h.hexdigest())
    # Publication is a separate registered artifact; archive members retain
    # their original projection/ prefix, as recorded by the original readback.
    if set(seen)!=set(inventory):
        raise ValueError('projection inventory differs')
    if any(seen[name]!=row for name,row in inventory.items()):raise ValueError('projection member bytes differ')
    payload=directory/'projection'
    export.asset(payload/'inputs/manifest.json',export.PROJECTION_SHA)
    export.asset(payload/'inputs/frames.jsonl',export.FRAME_SHA)
    export.asset(payload/'split-admission.json',export.ADMISSION_SHA)
    export.asset(archive,PROJECTION_ARCHIVE_SHA)
    return payload/'inputs',payload/'split-admission.json'


def launch(root,source,inputs,admission,checkpoint,seed,world_size):
    if type(world_size) is not int or world_size not in (4,8):raise ValueError('four or eight admitted GPU cards required')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=world_size:raise ValueError('assigned CUDA card count differs')
    script=Path(export.__file__).absolute();processes=[];logs=[];outputs=[]
    try:
        for rank in range(world_size):
            out=root/f'rank{rank}';outputs.append(out)
            log=(root/f'rank{rank}.log').open('xb');logs.append(log)
            command=[sys.executable,str(script),'--source-archive',str(source),'--inputs-root',str(inputs),
                '--admission',str(admission),'--checkpoint',str(checkpoint),'--seed',str(seed),'--output',str(out),
                '--device',f'cuda:{rank}','--rank',str(rank),'--world-size',str(world_size)]
            print(json.dumps(dict(stage='official_test_shard_launch',rank=rank,command=command,ETA_seconds=None)),flush=True)
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            for rank,p in enumerate(processes):
                if p.poll() not in (None,0):raise RuntimeError(f'official test rank {rank} failed; original log preserved')
            for rank in range(world_size):
                lines=(root/f'rank{rank}.log').read_text(errors='replace').splitlines()
                values=[]
                for line in lines:
                    try:v=json.loads(line)
                    except ValueError:continue
                    if isinstance(v,dict) and v.get('stage')=='official_test_raw_export':values.append(v)
                if values:print(json.dumps(dict(values[-1],rank=rank,whole_experiment_ETA='unknown')),flush=True)
            time.sleep(15)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('failed official-test shard')
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            p.wait()
        for log in logs:log.close()
    return merger.merge(outputs,inputs/'frames.jsonl',root/'merged')


def main(task=None):
    from clearml import Task
    if task is None:
        task=Task.init(project_name='Thesis/Recover-Before-Fuse/Inference',task_name='V2V4Real official-test raw shards',
            reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False)
    p={k.split('/',1)[-1]:v for k,v in task.get_parameters().items() if k.startswith('General/')}
    seed=int(p['seed']);world=int(p['world_size'])
    if seed not in CHECKPOINT_TASKS:raise ValueError('frozen native checkpoint seed required')
    root=Path('/tmp')/('rbf-official-test-raw-'+task.id);root.mkdir()
    try:
        source=root/'original-source.tar.gz';projection=root/'projection.tar.gz';checkpoint=root/'best-checkpoint'
        publication=root/'original-publication-receipt.json'
        inputs=[download(Task.get_task(task_id=SOURCE_TASK),'source',export.SOURCE_ARCHIVE_SHA,source),
                download(Task.get_task(task_id=export.PROJECTION_TASK),'publication-receipt',PUBLICATION_SHA,publication),
                download(Task.get_task(task_id=export.PROJECTION_TASK),'projection',PROJECTION_ARCHIVE_SHA,projection),
                download(Task.get_task(task_id=CHECKPOINT_TASKS[seed]),'best-checkpoint',export.CHECKPOINTS[seed],checkpoint)]
        prepared,admission=unpack_projection(projection,root/'projection',publication)
        result=launch(root,source,prepared,admission,checkpoint,seed,world)
        result['registered_inputs']=inputs
        result['whole_experiment_ETA']='unknown';result['independent_raw_acceptance']=False
        (root/'execution-receipt.json').write_bytes(export.canonical(result))
        # Upload a single complete merged output, not four partial test cohorts.
        for name in ('native-features','native-feature-manifest','native-feature-index','projection-manifest','split-admission','raw-merge-receipt.json','raw-export-receipt.json'):
            task.upload_artifact(name,artifact_object=str(root/'merged'/name),wait_on_upload=True)
        task.upload_artifact('projection-frames',artifact_object=str(prepared/'frames.jsonl'),wait_on_upload=True)
        task.upload_artifact('execution-receipt',artifact_object=str(root/'execution-receipt.json'),wait_on_upload=True)
        task.close()
    except Exception as exc:
        failure=dict(kind='v2v4real_official_test_raw_candidate_failure_v1',exception_type=type(exc).__name__,
                     seed=seed,world_size=world,independent_raw_acceptance=False,paper_metric=False,ETA_seconds=None)
        (root/'failure.json').write_bytes(export.canonical(failure))
        task.upload_artifact('failure-receipt',artifact_object=str(root/'failure.json'),wait_on_upload=True)
        for log in root.glob('rank*.log'):task.upload_artifact(log.stem,artifact_object=str(log),wait_on_upload=True)
        raise


if __name__=='__main__':main()
