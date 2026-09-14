#!/usr/bin/env python3
"""Bounded, non-training check of four 5090 devices and the existing package.

Only the pinned small package manifest is downloaded from the original private
server. No dataset/model download, alternate route, credential print, global
configuration change or artifact upload. The child SDK operation has a 45 s cap.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.parse import urlparse

PACKAGE_TASK='79fa0cea63774a4b959e6ef4fd09deed'
PACKAGE_SHA='63169e2f526e18fee0c5fcd05984de39e6a370a46c9804ca599e4c2d87697c52'
FILES_HOST='http://10.100.35.118:8081'
API_HOST='http://10.100.35.118:8008'
MARKER='RBF_FILES_PROBE '


def checked_manifest(artifact):
    uri=urlparse(artifact.url)
    if (uri.scheme,uri.hostname,uri.port)!=('http','10.100.35.118',8081):
        raise ValueError('designated private file server required')
    path=artifact.get_local_copy()
    if not path:raise RuntimeError('authenticated download failed')
    path=Path(path)
    if path.stat().st_size>1024*1024:raise ValueError('manifest exceeds one MiB')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PACKAGE_SHA:raise ValueError('manifest identity differs')
    manifest=json.loads(raw)
    if (manifest['kind']!='forest_identity_ddp_package_v1' or manifest['split']!='train'
            or manifest['class_scope']!=['car'] or manifest['raw_GT_included'] is not False):
        raise ValueError('existing derived car-only train package required')
    return dict(download_verified=True,bytes=len(raw),sha256=PACKAGE_SHA)


def sdk_child():
    try:
        from clearml import Task
        result=checked_manifest(Task.get_task(task_id=PACKAGE_TASK).artifacts['package'])
    except Exception as error:
        # Do not expose URLs, authorization headers or arbitrary SDK payloads.
        result=dict(download_verified=False,error_type=type(error).__name__)
    print(MARKER+json.dumps(result,sort_keys=True),flush=True)
    return 0 if result['download_verified'] else 2


def bounded_download():
    try:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--package-check'],
            env=dict(os.environ,CLEARML_FILES_HOST=FILES_HOST,CLEARML_API_HOST=API_HOST,
                     CLEARML_AGENT_FORCE_TASK_INIT='0'),
            capture_output=True,text=True,timeout=45,check=False)
    except subprocess.TimeoutExpired:
        return dict(download_verified=False,error_type='TimeoutExpired',timeout_seconds=45)
    records=[json.loads(line[len(MARKER):]) for line in result.stdout.splitlines() if line.startswith(MARKER)]
    if len(records)!=1 or result.returncode not in (0,2):
        return dict(download_verified=False,error_type='ChildProbeFailed',returncode=result.returncode)
    record=records[0]
    if result.returncode!=0 or record.get('download_verified') is not True:
        return dict(download_verified=False,error_type=record.get('error_type','ChildProbeFailed'))
    if record.get('sha256')!=PACKAGE_SHA or not 0<record.get('bytes',0)<=1024*1024:
        return dict(download_verified=False,error_type='InvalidChildReceipt')
    return record


def endpoint_reachability():
    results={}
    for name,port in (('api',8008),('files',8081)):
        try:
            with socket.create_connection(('10.100.35.118',port),timeout=5):pass
            results[name]=dict(reachable=True,port=port)
        except OSError as error:
            results[name]=dict(reachable=False,port=port,error_type=type(error).__name__,errno=error.errno)
    return results


def main():
    os.environ['CLEARML_FILES_HOST']=FILES_HOST
    os.environ['CLEARML_API_HOST']=API_HOST
    os.environ['CLEARML_AGENT_FORCE_TASK_INIT']='0'
    if sys.argv[1:]==['--package-check']:return sdk_child()
    if sys.argv[1:]:raise ValueError('unexpected probe arguments')
    import torch
    # The agent captures stdout and task status. Keep SDK authentication inside
    # the bounded child, using only the explicitly authorized API/files host.
    if torch.cuda.device_count()!=4:raise ValueError('exactly four allocated GPUs required')
    devices=[]
    for i in range(4):
        props=torch.cuda.get_device_properties(i)
        if '5090' not in props.name:raise ValueError('only the assigned 5090 family is allowed')
        devices.append(dict(index=i,name=props.name,uuid=str(getattr(props,'uuid','unavailable'))))
    if (len({d['uuid'] for d in devices})!=4
            or any(d['uuid'] in ('unavailable','None','') for d in devices)):
        raise ValueError('four distinct GPU UUIDs required')
    print('RBF_FILES_PROBE_STARTED '+json.dumps(dict(devices=devices,package_task=PACKAGE_TASK)),flush=True)
    endpoints=endpoint_reachability()
    result=(bounded_download() if all(r['reachable'] for r in endpoints.values()) else
            dict(download_verified=False,error_type='DesignatedEndpointUnreachable'))
    print('RBF_WORKER_FILES_READINESS '+json.dumps(dict(result,devices=devices,endpoints=endpoints,parameter_training=False,
        dataset_or_model_downloaded=False,package_task=PACKAGE_TASK,paper_eligible=False),sort_keys=True),flush=True)
    if result['download_verified'] is not True:raise RuntimeError('bounded private package-manifest download failed')
    return 0


if __name__=='__main__':sys.exit(main())
