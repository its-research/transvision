from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from tools.event_track_v2x.train_allocation_policy_ddp import (
    GroupLoss,PriorityFitConfig,audit_data,fit_distributed,global_groups,policy,priority_model,
    sequence_partition,validate_runtime,
)
from transvision.models.event_track_v2x.allocation_policy import FEATURES
from transvision.models.event_track_v2x.allocation_training import load_priority
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,sha_file
from test_allocation_training import priority_data
from test_forest_training_data import prepared_rows


def _worker(rank,root,rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        rng=np.random.default_rng(71)
        batch=[(rng.normal(size=(n,len(FEATURES))),rng.uniform(-1,1,size=n)) for n in (1,7,3)]
        torch.manual_seed(82);model=priority_model(4);full=GroupLoss(model)
        full(batch,len(batch),1).backward()
        gradients={name:p.grad.clone() for name,p in model.named_parameters()}
        model.zero_grad(set_to_none=True);ddp=DistributedDataParallel(full)
        ddp(batch[rank::4],len(batch),4).backward()
        for name,p in model.named_parameters():
            torch.testing.assert_close(p.grad,gradients[name],rtol=1e-12,atol=1e-12)
        del ddp,full,model
        data=Path(root)/'priority-data';config=PriorityFitConfig(epochs=2,hidden=4,batch_groups=4)
        first=fit_distributed(data,sha_file(data/'manifest.json'),Path(root)/'fit-first',device='cpu',
            seed=1337,config=config,require_full_train=False)
        assert first['world_size']==4 and not first['complete_three_seed_campaign']
        dist.barrier()
        if rank==0:
            path=data/'manifest.json';manifest=json.loads(path.read_bytes())
            _,held=sequence_partition(manifest['shards'])
            for record in held:
                file=data/record['path'];rows=[json.loads(x) for x in file.read_bytes().splitlines()]
                for row in rows:row['targets']=[.9]*len(row['targets'])
                file.write_bytes(b''.join(canonical(row)+b'\n' for row in rows))
                record['sha256']=sha_file(file)
            path.write_bytes(canonical(manifest))
        dist.barrier()
        second=fit_distributed(data,sha_file(data/'manifest.json'),Path(root)/'fit-heldout-changed',device='cpu',
            seed=1337,config=config,require_full_train=False)
        assert not second['full_official_train_trace']
        for field in ('hidden','seed','manifest'):
            divergent=dict(device='cpu',seed=1337,config=config,require_full_train=False)
            manifest_sha=sha_file(data/'manifest.json')
            if field=='hidden':
                divergent['config']=PriorityFitConfig(epochs=2,hidden=4+rank,batch_groups=4)
            if field=='seed':divergent['seed']=1337 if rank==0 else 2027
            if field=='manifest' and rank:manifest_sha='a'*64
            with pytest.raises(ValueError,match='rank execution contracts differ'):
                fit_distributed(data,manifest_sha,Path(root)/('divergent-'+field),**divergent)
        with pytest.raises(ValueError,match='CUDA/NCCL'):
            fit_distributed(data,sha_file(data/'manifest.json'),Path(root)/'not-production',device='cpu')
    finally:dist.destroy_process_group()


def test_four_actual_processes_group_gradient_empty_rank_and_holdout_isolation(priority_data,tmp_path):
    data,_,binding,*_=priority_data
    mp.spawn(_worker,args=(str(tmp_path),str(tmp_path/'rendezvous')),nprocs=4,join=True)
    policies=[];held_losses=[]
    for name in ('fit-first','fit-heldout-changed'):
        output=tmp_path/name;receipt=json.loads((output/'receipt.json').read_bytes())
        frozen,cp=load_priority(output/'1337',receipt['checkpoint_sha256'],binding=binding,require_full_train=False)
        assert frozen.signature!=cp['initial_policy_signature']
        assert cp['distributed_world_size']==4 and not cp['head_fit_includes_holdout']
        assert not cp['strict_pipeline_isolated_selection'] and not receipt['paper_eligible']
        epochs=[json.loads(x) for x in (output/'1337/epochs.jsonl').read_bytes().splitlines()]
        assert [r['epoch'] for r in epochs]==[1,2]
        for row in epochs:
            assert sum(r['fit_groups'] for r in row['rank_progress'])==row['fit_groups']
            assert sum(r['holdout_groups'] for r in row['rank_progress'])==row['holdout_groups']
            assert len({r['policy_signature'] for r in row['rank_progress']})==1
        policies.append(frozen.signature);held_losses.append(epochs[-1]['train_sequence_holdout_mse'])
        with pytest.raises(ValueError,match='provenance'):
            load_priority(output/'1337',receipt['checkpoint_sha256'],binding=binding)
    assert policies[0]==policies[1] and held_losses[0]!=held_losses[1]
    assert not (tmp_path/'not-production').exists()
    assert not any((tmp_path/('divergent-'+field)).exists() for field in ('hidden','seed','manifest'))


def test_equal_group_objective_is_not_equal_candidate_row_objective():
    model=priority_model(4)
    for p in model.parameters():p.data.zero_()
    batch=[(np.zeros((1,len(FEATURES))),np.array([1.])),
           (np.zeros((9,len(FEATURES))),np.zeros(9))]
    assert float(GroupLoss(model)(batch,2,1).detach())==pytest.approx(.5)


@pytest.mark.parametrize('size',[1,2,4,64])
def test_streamed_batches_keep_every_group_and_candidate_count(priority_data,size):
    data,*_=priority_data;manifest,fit,held,stats=audit_data(data,sha_file(data/'manifest.json'),require_full_train=False)
    for name,records in (('fit',fit),('holdout',held)):
        batches=list(global_groups(data,records,size))
        assert all(1<=len(b)<=size for b in batches)
        assert sum(map(len,batches))==stats[name]['groups']
        assert sum(len(y) for b in batches for _,y in b)==stats[name]['rows']


@pytest.mark.parametrize('bad',['fixture','false_full_flag','duplicate_path','wrong_source','gt_input','invalid_group'])
def test_data_preflight_rejects_wrong_provenance_before_output(priority_data,bad):
    data,*_=priority_data;path=data/'manifest.json';manifest=json.loads(path.read_bytes())
    if bad=='false_full_flag':manifest['full_official_train_trace']=True
    if bad=='duplicate_path':manifest['shards'][1]['path']=manifest['shards'][0]['path']
    if bad=='wrong_source':manifest['source_sha256']={}
    if bad=='gt_input':manifest['future_or_gt_inputs']=True
    if bad=='invalid_group':
        record=manifest['shards'][0];p=data/record['path'];p.write_bytes(b'{"GT":"not a training field"}\n')
        record['sha256']=sha_file(p)
    path.write_bytes(canonical(manifest))
    with pytest.raises(ValueError):
        audit_data(data,sha_file(path),require_full_train=bad in ('fixture','false_full_flag'))


@pytest.mark.parametrize('bad',['world','ranks','hosts','device','uuid','missing_uuid','family','cpu'])
def test_gpu_inventory_requires_four_distinct_a100_or_5090_cards(bad):
    rows=[dict(rank=i,world_size=4,host='worker',device='cuda:'+str(i),backend='nccl',
               gpu_uuid='uuid-'+str(i),gpu_name='NVIDIA A100') for i in range(4)]
    validate_runtime(rows,require_full_train=True)
    if bad=='world':rows=rows[:3]
    if bad=='ranks':rows[0]['rank']=1
    if bad=='hosts':rows[0]['host']='another'
    if bad=='device':rows[0]['device']='cuda:1'
    if bad=='uuid':rows[0]['gpu_uuid']='uuid-1'
    if bad=='missing_uuid':rows[0]['gpu_uuid']='None'
    if bad=='family':rows[0]['gpu_name']='Tesla V100'
    if bad=='cpu':rows[0].update(device='cpu',backend='gloo')
    with pytest.raises(ValueError):validate_runtime(rows,require_full_train=True)


@pytest.mark.parametrize('kwargs',[dict(epochs=0),dict(hidden=True),dict(batch_groups=0),
                                  dict(learning_rate=float('nan')),dict(gradient_clip=True)])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):PriorityFitConfig(**kwargs)


def test_requires_initialized_real_process_group(tmp_path):
    with pytest.raises(ValueError,match='initialized'):
        fit_distributed(tmp_path,'a'*64,tmp_path/'out',device='cpu',require_full_train=False)
