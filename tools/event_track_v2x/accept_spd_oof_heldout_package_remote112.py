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
    mp=a.package/'package-manifest.json';manifest=json.loads(mp.read_bytes())
    admission_path=a.package/'independent-archive-readback.json'
    admission=json.loads(admission_path.read_bytes())
    if (manifest.get('kind')!='spd_canonical_oof_heldout_inference_package_v1'
            or manifest.get('gt_or_val_test_included') is not False
            or manifest.get('predictions_generated') is not False
            or admission.get('kind')!='spd_canonical_oof_heldout_inference_package_independent_readback_v1'
            or admission.get('all_archive_payloads_verified') is not True
            or admission.get('source_input_admission_sha256')!='29214bb602a4637637a479636109e6bec6591282d7e2647c2d628bc8ebd967df'
            or admission['fold_id']!=a.fold or admission['package_manifest_sha256']!=sha(mp)
            or admission['archive']!=manifest['archive']):
        raise ValueError('held-out archive lacks independent input byte admission')
    upload=json.loads((a.package/'clearml-upload-acceptance.json').read_bytes())
    if (upload['status']!='uploaded_and_registered_hash_verified' or upload['fold_id']!=a.fold
            or manifest['fold_id']!=a.fold or upload['manifest_sha256']!=sha(mp) or upload['independent_receipt_sha256']!=sha(admission_path)):
        raise ValueError('uploaded fold/package identity differs')
    collective=Path('/Volumes/Data/test/recover-before-fuse/receipts/spd-canonical-oof-fivefold-heldout-inference-package-admission-20261001.json')
    if sha(collective)!='a38ed021adb55158c9546d33f94280efe83cbd44e88dd68cefad79f7c374b98c':
        raise ValueError('canonical all-fold archive admission changed')
    row=next(r for r in json.loads(collective.read_bytes())['folds'] if r['fold_id']==a.fold)
    if row['independent_receipt_sha256']!=sha(admission_path) or row['package_manifest_sha256']!=sha(mp) or row['archive']!=manifest['archive']:
        raise ValueError('archive differs from canonical fivefold independent admission')
    task=Task.get_task(task_id=upload['task_id'])
    params=task.get_parameters()
    if params.get('General/package_manifest_sha256')!=sha(mp) or params.get('General/independent_receipt_sha256')!=sha(admission_path):
        raise ValueError('uploaded task admission binding differs')
    if task.status!='completed':raise ValueError('upload task not completed')
    archive=manifest['archive']
    expected=[('package-manifest',mp.stat().st_size,sha(mp)),('heldout-inputs',archive['bytes'],archive['sha256']),('independent-archive-readback',admission_path.stat().st_size,sha(admission_path))]
    artifacts=[]
    for name,size,digest in expected:
        item=task.artifacts[name]
        if item.size!=size or item.hash!=digest:raise ValueError('registered identity differs')
        parsed=urlparse(item.url)
        if parsed.scheme!='http' or parsed.netloc not in {'10.100.35.118:8081','10.100.34.118:8081'}:raise ValueError('registered file host differs')
        helper=StorageHelper.get(item.url);container=helper._driver._containers['http://'+parsed.netloc]
        artifacts.append({'name':name,'url':item.url,'bytes':size,'sha256':digest,
                          'headers':dict(container.get_headers(item.url))})
    source=Path(__file__).with_name('readback_spd_oof_package_remote_bytes.py')
    freeze=Path('/Volumes/Data/test/recover-before-fuse/source-freezes/spd-oof-heldout-package-remote-readback-20261001')
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
    receipt={'kind':'spd_canonical_oof_heldout_inference_package_clearml_independent_readback_v1','status':'independent_bytes_verified',
        'task_id':task.id,'fold_id':a.fold,'manifest_sha256':sha(mp),'artifacts':result['artifacts'],
        'predictions_generated':False,'paper_eligible':False,'gt_or_val_test_included':False,'readback_execution_host':result['execution_host'],
        'readback_execution_platform':result['execution_platform'],'remote_source_sha256':sha(source),
        'acceptor_sha256':sha(Path(__file__)),'credentials_persisted':False,
        'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    with out.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    print('OOF_HELDOUT_INPUT_PACKAGE_INDEPENDENTLY_ACCEPTED '+sha(out),flush=True)


if __name__=='__main__':
    main()
