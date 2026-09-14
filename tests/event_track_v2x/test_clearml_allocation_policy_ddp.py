"""Synthetic controller evidence is not real GPU training or authorization."""
import copy
from contextlib import nullcontext
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from tools.event_track_v2x import run_clearml_allocation_policy_ddp as tool
from tools.event_track_v2x.train_allocation_policy_ddp import priority_ddp_sources
from transvision.models.event_track_v2x.allocation_training import KIND,allocation_sources
from transvision.models.event_track_v2x.allocation_policy import FEATURES,RECIPE,TARGET,FrozenPriorityPolicy
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,sha_file


def task_params():
    return dict(seed='1337',required_world_size='4',global_batch_groups='64',epochs='10',class_scope='car',
                gpu_family='A100',package_task_id='a'*32,package_sha256='b'*64,controller_sha256='c'*64)


@pytest.mark.parametrize('prefix',['','General/'])
def test_exact_four_gpu_group_objective_contract(prefix):
    result=tool.parameters({prefix+k:v for k,v in task_params().items()})
    assert result['seed']==1337 and result['global_batch_groups']==64


@pytest.mark.parametrize('change',[dict(seed=True),dict(seed='1'),dict(required_world_size='3'),
    dict(global_batch_groups='32'),dict(epochs='1'),dict(class_scope='pedestrian'),dict(gpu_family='V100'),
    dict(package_task_id='missing'),dict(controller_sha256='')])
def test_wrong_assignment_rejected(change):
    with pytest.raises(ValueError):tool.parameters(dict(task_params(),**change))


def test_duplicate_namespaced_parameters_rejected():
    with pytest.raises(ValueError,match='ambiguous'):tool.parameters(dict(task_params(),**{'General/seed':'1337'}))


def test_download_is_authenticated_and_cannot_use_an_alternate_host(tmp_path):
    artifact=SimpleNamespace(url='http://10.100.35.118:8081/package',get_local_copy=lambda:str(tmp_path/'package'))
    assert tool.download(artifact)==tmp_path/'package'
    artifact.url='http://10.100.34.118:8081/package'
    with pytest.raises(ValueError,match='designated'):tool.download(artifact)
    artifact.url='http://10.100.35.118:8081/package';artifact.get_local_copy=lambda:None
    with pytest.raises(RuntimeError,match='authenticated'):tool.download(artifact)


@pytest.mark.parametrize('bad',[None,'unexpected','omitted','link','escape','cap','hash'])
def test_archive_inventory_cannot_hide_extra_payloads(tmp_path,bad):
    payload=b'numeric';record=dict(path='groups.jsonl',bytes=len(payload),sha256=__import__('hashlib').sha256(payload).hexdigest())
    path=tmp_path/'archive.tar.gz'
    with tarfile.open(path,'w:gz') as archive:
        member=tarfile.TarInfo('../escape' if bad=='escape' else 'groups.jsonl');member.size=len(payload)
        if bad=='link':member.type=tarfile.SYMTYPE;member.linkname='/tmp'
        if bad!='omitted':archive.addfile(member,io.BytesIO(payload))
        if bad=='unexpected':
            extra=tarfile.TarInfo('GT.json');extra.size=2;archive.addfile(extra,io.BytesIO(b'{}'))
    if bad=='hash':record['sha256']='0'*64
    if bad:
        with pytest.raises(ValueError):tool.extract_checked(path,tmp_path/'out',[record],1 if bad=='cap' else 100)
    else:
        tool.extract_checked(path,tmp_path/'out',[record],100)
        assert (tmp_path/'out/groups.jsonl').read_bytes()==payload


@pytest.fixture
def result_fixture(tmp_path):
    output=tmp_path/'fit';directory=output/'1337';directory.mkdir(parents=True)
    rng=np.random.default_rng(1337)
    frozen=FrozenPriorityPolicy([rng.normal(size=(32,18)),rng.normal(size=32),rng.normal(size=(1,32)),rng.normal(size=1)])
    np.savez(directory/'weights.npz',**{'w'+str(i):w for i,w in enumerate(frozen.weights)})
    metadata=dict(training_manifest_sha256='a'*64,source_sha256=priority_ddp_sources(),binding={'fixed':'binding'},
        statistics=dict(fit=dict(groups=8,rows=16),holdout=dict(groups=4,rows=8)),
        fit_sequences=['fit'],holdout_sequences=['held'])
    runtime=[dict(rank=i,world_size=4,host='synthetic',device=f'cuda:{i}',backend='nccl',
                  gpu_name='NVIDIA A100',gpu_uuid=f'uuid-{i}') for i in range(4)]
    plan=dict(fit_config=tool.FIT_CONFIG,seed=1337,training_manifest_sha256=metadata['training_manifest_sha256'],
        training_source_sha256=metadata['source_sha256'],statistics=metadata['statistics'],binding=metadata['binding'],
        fit_sequences=metadata['fit_sequences'],holdout_sequences=metadata['holdout_sequences'],rank_runtime=runtime)
    cp=dict(kind=KIND,feature_recipe=RECIPE,target_recipe=TARGET,feature_names=FEATURES,split='train',
        full_official_train_trace=True,seed=1337,training_manifest_sha256=metadata['training_manifest_sha256'],
        training_source_sha256=metadata['source_sha256'],source_sha256=allocation_sources(),binding=metadata['binding'],
        fit_sequences=metadata['fit_sequences'],holdout_sequences=metadata['holdout_sequences'],
        initial_policy_signature='initial',policy_signature=frozen.signature,fixed_final_epoch=10,
        distributed_world_size=4,distributed_backend='nccl',weights_sha256=sha_file(directory/'weights.npz'))
    rank_progress=[dict(rank=i,batches=1,nonempty_batches=1,positive_gradient_batches=1,
        fit_groups=2,fit_rows=4,holdout_groups=1,holdout_rows=2,group_loss_sum=.2,holdout_group_loss_sum=.1,
        policy_signature=frozen.signature) for i in range(4)]
    epochs=[dict(epoch=i,rank_progress=copy.deepcopy(rank_progress),global_batches=1,fit_groups=8,holdout_groups=4,
        official_validation_or_test_read=False,training_mse=.1,train_sequence_holdout_mse=.1) for i in range(1,11)]
    receipt=dict(kind='component_priority_ddp_receipt_v1',status='complete',seed=1337,world_size=4,
        full_official_train_trace=True,local_fixture_only=False,checkpoint_manifest='1337/checkpoint.json',rank_runtime=runtime)
    def seal():
        (output/'plan.json').write_bytes(canonical(plan));cp['plan_sha256']=sha_file(output/'plan.json')
        (directory/'checkpoint.json').write_bytes(canonical(cp))
        (directory/'epochs.jsonl').write_bytes(b''.join(canonical(e)+b'\n' for e in epochs))
        receipt.update(plan_sha256=sha_file(output/'plan.json'),checkpoint_sha256=sha_file(directory/'checkpoint.json'),
                       epochs_sha256=sha_file(directory/'epochs.jsonl'))
        (output/'receipt.json').write_bytes(canonical(receipt))
    seal()
    return output,metadata,plan,cp,epochs,receipt,seal


def test_every_saved_epoch_and_actual_npz_policy_are_validated(result_fixture):
    output,meta,*_=result_fixture
    artifacts=tool.validate_result(output,meta,1337,'A100',expected_devices=[dict(uuid=f'uuid-{i}') for i in range(4)])
    assert len(artifacts)==5 and artifacts[-1][0]=='priority-checkpoint'
    assert all(path.is_file() for _,path in artifacts)
    assert not any('GT' in str(path) or 'predictions' in str(path) for _,path in artifacts)


@pytest.mark.parametrize('bad',['status','world','fixture','seed','epochs','rank','unused_rank','coverage','signature',
    'objective','nonfinite','holdout','hyperparameters','source','unchanged_weights','gpu_family','uuid','preflight'])
def test_resealed_but_invalid_training_results_cannot_be_uploaded(result_fixture,bad):
    output,meta,plan,cp,epochs,receipt,seal=result_fixture
    expected=[dict(uuid=f'uuid-{i}') for i in range(4)]
    if bad=='status':receipt['status']='failed'
    elif bad=='world':receipt['world_size']=1
    elif bad=='fixture':receipt['local_fixture_only']=True
    elif bad=='seed':cp['seed']=2027
    elif bad=='epochs':epochs.pop()
    elif bad=='rank':epochs[0]['rank_progress'][0]['rank']=1
    elif bad=='unused_rank':epochs[0]['rank_progress'][0]['positive_gradient_batches']=0
    elif bad=='coverage':epochs[0]['rank_progress'][0]['fit_groups']-=1
    elif bad=='signature':epochs[-1]['rank_progress'][0]['policy_signature']='different'
    elif bad=='objective':epochs[0]['training_mse']=.9
    elif bad=='nonfinite':epochs[0]['training_mse']=float('nan')
    elif bad=='holdout':plan['holdout_sequences']=['fit']
    elif bad=='hyperparameters':plan['fit_config']=dict(plan['fit_config'],epochs=1)
    elif bad=='source':cp['training_source_sha256']={}
    elif bad=='unchanged_weights':cp['initial_policy_signature']=cp['policy_signature']
    elif bad=='gpu_family':receipt['rank_runtime'][0]['gpu_name']='V100'
    elif bad=='uuid':receipt['rank_runtime'][0]['gpu_uuid']='uuid-1'
    else:expected[0]['uuid']='different'
    # Canonical JSON deliberately forbids NaN; raw corrupt logs are still rejected.
    if bad=='nonfinite':
        path=output/'1337/epochs.jsonl';path.write_text('\n'.join(json.dumps(e) for e in epochs)+'\n')
        receipt['epochs_sha256']=sha_file(path);(output/'receipt.json').write_bytes(canonical(receipt))
    else:seal()
    with pytest.raises(ValueError):tool.validate_result(output,meta,1337,'A100',expected_devices=expected)


def test_failed_torchrun_never_validates_or_uploads_results(tmp_path,monkeypatch):
    import torch
    source=tmp_path/'source';source.write_bytes(b'fixture source')
    groups=tmp_path/'groups';groups.write_bytes(b'fixture groups')
    artifacts=[dict(path=name,bytes=p.stat().st_size,sha256=sha_file(p)) for name,p in
               (('source.tar.gz',source),('priority-groups.tar.gz',groups))]
    metadata=dict(kind='component_priority_ddp_package_v1',split='train',class_scope=['car'],
        full_official_train_trace=True,teacher_frames=7445,teacher_sequences=46,
        raw_GT_included=False,teacher_traces_included=False,predictions_included=False,
        original_detection_stream_included=False,trained_models_included=False,
        artifacts=artifacts,source_inventory=[],data_inventory=[dict(bytes=10)],training_manifest_sha256='a'*64)
    package=tmp_path/'package.json';package.write_bytes(canonical(metadata))
    remote=SimpleNamespace(artifacts={name:SimpleNamespace(url='http://10.100.35.118:8081/'+name,
        get_local_copy=lambda p=p:str(p)) for name,p in
        (('package',package),('source.tar.gz',source),('priority-groups.tar.gz',groups))})
    params=dict(task_params(),package_sha256=sha_file(package),controller_sha256=sha_file(tool.__file__))
    task=SimpleNamespace(get_parameters=lambda:params,
        upload_artifact=lambda *a,**k:pytest.fail('failed training must not upload'),
        close=lambda:pytest.fail('failed training must not close as a completed task'))
    Task=SimpleNamespace(init=lambda **kw:task,get_task=lambda **kw:remote)
    monkeypatch.setitem(sys.modules,'clearml',SimpleNamespace(Task=Task))
    monkeypatch.setattr(tool.socket,'create_connection',lambda *a,**k:nullcontext())
    monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.distributed,'is_nccl_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(name='NVIDIA A100',uuid=f'uuid-{i}'))
    monkeypatch.setattr(torch.cuda,'device',lambda i:nullcontext())
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ones=torch.ones;monkeypatch.setattr(torch,'ones',lambda shape,**kw:ones(shape))
    runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(tool.tempfile,'mkdtemp',lambda **kw:str(runtime))
    monkeypatch.setattr(tool,'extract_checked',lambda p,d,i,m:d.mkdir())
    monkeypatch.setattr(tool,'validate_result',lambda *a,**kw:pytest.fail('must check torchrun exit first'))
    def failed(command,**kwargs):
        assert '--nproc_per_node=4' in command and '--global-batch-groups' in command
        assert kwargs['check'] is True and kwargs['env']['CLEARML_API_HOST']=='http://10.100.35.118:8008'
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(tool.subprocess,'run',failed)
    with pytest.raises(subprocess.CalledProcessError):tool.main()
