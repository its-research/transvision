#!/usr/bin/env python3
"""Independently accept uploaded package bytes over the registered LAN path."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from urllib.parse import urlparse
from run_cooptrack_official_oof_gpu4 import validate_cohort


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=int,choices=range(5),required=True)
    parser.add_argument('--package',type=Path,required=True)
    a=parser.parse_args();out=a.package/'clearml-independent-readback.json'
    if out.exists():raise FileExistsError('accepted receipt is create-once')
    from clearml import Task
    from clearml.storage.helper import StorageHelper
    mp=a.package/'package-manifest.json';manifest=json.loads(mp.read_bytes());validate_cohort(manifest)
    upload=json.loads((a.package/'clearml-upload-acceptance.json').read_bytes())
    if (upload['status']!='uploaded_and_hash_verified' or upload['fold_id']!=a.fold
            or manifest['fold_id']!=a.fold or upload['manifest_sha256']!=sha(mp)):
        raise ValueError('uploaded fold/package identity differs')
    task=Task.get_task(task_id=upload['task_id'])
    if task.status!='completed':raise ValueError('upload task not completed')
    archive=next(r for r in manifest['inventory'] if r['path']=='train-inputs.tar.gz')
    expected=[('package-manifest',mp.stat().st_size,sha(mp)),('train-inputs',archive['bytes'],archive['sha256'])]
    artifacts=[]
    for name,size,digest in expected:
        item=task.artifacts[name]
        if item.size!=size or item.hash!=digest:raise ValueError('registered identity differs')
        parsed=urlparse(item.url)
        if parsed.scheme!='http' or parsed.netloc!='10.100.35.118:8081':raise ValueError('registered file host differs')
        helper=StorageHelper.get(item.url);container=helper._driver._containers['http://'+parsed.netloc]
        artifacts.append({'name':name,'url':item.url,'bytes':size,'sha256':digest,
                          'headers':dict(container.get_headers(item.url))})
    source=Path(__file__).with_name('readback_spd_oof_package_remote_bytes.py')
    freeze=Path('/Volumes/Data/test/recover-before-fuse/source-freezes/spd-oof-package-remote-readback-v3-20260930')
    if (freeze/source.name).read_bytes()!=source.read_bytes() or (freeze/Path(__file__).name).read_bytes()!=Path(__file__).read_bytes():
        raise ValueError('remote readback code differs from source freeze')
    process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',
        '10.100.35.112','python3 -u -c '+shlex.quote(source.read_text())],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    process.stdin.write(json.dumps({'artifacts':artifacts}));process.stdin.close();result=None
    for line in process.stdout:
        if line.startswith('REMOTE_READBACK_RESULT '):result=json.loads(line.split(' ',1)[1])
        elif line.startswith('REMOTE_READBACK_PROGRESS '):print(line,end='',flush=True)
    if process.wait()!=0 or result is None:raise RuntimeError('remote independent byte readback failed; no acceptance')
    if result['artifacts']!={name:{'bytes':size,'sha256':digest} for name,size,digest in expected}:
        raise ValueError('independent byte result differs')
    current=Task.get_task(task_id=task.id)
    if current.status!='completed' or any(current.artifacts[n].hash!=h or current.artifacts[n].size!=s for n,s,h in expected):
        raise ValueError('registered artifact changed during readback')
    receipt={'kind':'spd_official_oof_fold_clearml_independent_readback_v1','status':'independent_bytes_verified',
        'task_id':task.id,'fold_id':a.fold,'manifest_sha256':sha(mp),'artifacts':result['artifacts'],
        'detector_training_started':False,'readback_execution_host':result['execution_host'],
        'readback_execution_platform':result['execution_platform'],'remote_source_sha256':sha(source),
        'acceptor_sha256':sha(Path(__file__)),'credentials_persisted':False,
        'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    with out.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    print('OOF_PACKAGE_INDEPENDENTLY_ACCEPTED '+sha(out),flush=True)


if __name__=='__main__':
    main()
