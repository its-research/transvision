"""Registered source/data fetch, strict byte admission, then four-GPU training."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlparse,urlunparse

from clearml import Task
from clearml.backend_api.session import Session
from clearml.storage.helper import StorageHelper


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def fetch(task,name,expected,path):
    item=task.artifacts[name];registered=urlparse(item.url);active=urlparse(Session.get_files_server_host())
    hosts={'10.100.35.118:8081','10.100.34.118:8081'}
    assert item.hash==expected and registered.scheme==active.scheme=='http' and registered.netloc in hosts and active.netloc in hosts
    assert not registered.username and not registered.query and not registered.fragment
    for attempt,host in enumerate(dict.fromkeys([active.netloc,'10.100.34.118:8081','10.100.35.118:8081'])):
        url=urlunparse(registered._replace(netloc=host));received=0;start=time.monotonic();partial=path.with_name(path.name+f'.route-attempt-{attempt}')
        print(json.dumps(dict(stage='association_registered_input_transfer',artifact=name,file_service=host,eta_seconds=None)),flush=True)
        try:
            blocks=StorageHelper.get(url).download_as_stream(url,chunk_size=1024**2)
            if blocks is None:raise FileNotFoundError('registered file transfer unavailable')
            with partial.open('xb') as stream:
                for block in blocks:
                    stream.write(block);received+=len(block);assert received<=item.size
                    if received%(16*1024**2)<len(block):print(json.dumps(dict(stage='association_input_transfer',artifact=name,completed_bytes=received,total_bytes=item.size,eta_seconds=(item.size-received)*(time.monotonic()-start)/received)),flush=True)
            assert received==item.size and sha(partial)==expected
            assert not path.exists();partial.replace(path);return path
        except Exception as error:
            print(json.dumps(dict(stage='registered_transfer_route_failed_preserved',artifact=name,file_service=host,error_type=type(error).__name__,received_bytes=received,eta_seconds=None)),flush=True)
    raise RuntimeError('all admitted file-service routes failed byte admission')


def main():
    task=Task.init(project_name='Thesis/EventTrack-V2X/Training',task_name='canonical OOF association GPU training',reuse_last_task_id=True,auto_connect_frameworks=False,auto_connect_arg_parser=False)
    task.output_uri='http://10.100.34.118:8081'
    params=task.get_parameters();get=lambda name:params['General/'+name]
    root=Path('canonical-association-car-only-job');root.mkdir(exist_ok=False)
    source=Task.get_task(task_id=get('asset_task_id'));assert source.status=='completed'
    asset=fetch(source,'asset-manifest',get('asset_manifest_sha256'),root/'asset-manifest.json');binding=json.loads(asset.read_text())
    assert binding['fold_id']==int(get('fold_id')) and binding['independent_full_fit_example_acceptance_passed'] is True
    archive=fetch(source,'fit-example-archive',binding['archive_sha256'],root/'examples.tar.gz')
    execution=Task.get_task(task_id=get('execution_source_task_id'));assert execution.status=='completed'
    execution_manifest=fetch(execution,'execution-source-manifest',get('execution_source_manifest_sha256'),root/'execution-source-manifest.json');execution_binding=json.loads(execution_manifest.read_text())
    assert execution_binding['runtime_contract_independently_admitted'] is True
    runner=fetch(execution,'GPU-trainer-source',execution_binding['trainer_sha256'],root/'train_spd_canonical_oof_association_gpu4.py')
    configpath=fetch(source,'training-configuration',binding['configuration_sha256'],root/'configuration.json');config=json.loads(configpath.read_text())
    assert config['fold_id']==binding['fold_id'] and config['seed']==1337
    assert execution_binding['class_scope']=='car' and execution_binding['class_index']==0
    config.update(class_scope='car',class_index=0,training_view='car-only-after-frozen-all-class-candidate-cap-v1',original_configuration_sha256=binding['configuration_sha256'],car_view_independent_acceptance_sha256=execution_binding['car_view_independent_acceptance_sha256'])
    configpath=root/'car-only-configuration.json';configpath.write_text(json.dumps(config,sort_keys=True,indent=2)+'\n')
    expected={r['path']:r for r in binding['inventory']};assert len(expected)==len(binding['inventory'])
    data=root/'data';data.mkdir();seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            rel=PurePosixPath(member.name)
            assert member.isfile() and member.name in expected and member.name not in seen and not rel.is_absolute() and '..' not in rel.parts
            row=expected[member.name];assert member.size==row['bytes'];p=data/member.name;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as stream:
                f=tar.extractfile(member)
                for b in iter(lambda:f.read(8*1024**2),b''):stream.write(b)
            assert sha(p)==row['sha256'];seen.add(member.name)
    assert seen==set(expected)
    subprocess.run([sys.executable,str(runner),'--data',str(data),'--config',str(configpath),'--output',str(root/'training')],check=True)
    completion=json.loads((root/'training/completion.json').read_text());assert completion['epoch']==24 and completion['configuration']==config
    for name,filename in [('association-checkpoint','checkpoint.pt'),('training-completion','completion.json'),('training-epochs','epochs.jsonl'),('actual-GPU-runtime','runtime.json'),('training-coverage','training-coverage.json')]:
        assert task.upload_artifact(name,artifact_object=root/'training'/filename,wait_on_upload=True)
    print(json.dumps(dict(stage='association_training_artifacts_uploaded_pending_independent_acceptance',fold_id=config['fold_id'],seed=config['seed'],epoch=24,completion_sha256=sha(root/'training/completion.json'),paper_eligible=False)),flush=True)


if __name__=='__main__':main()
