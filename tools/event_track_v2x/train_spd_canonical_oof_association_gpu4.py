#!/usr/bin/env python3
"""Four-GPU candidate training on independently admitted complementary-fit data.

No held-out labels, checkpoint selection, scenario certification or paper claim.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler


class PredictedAssociation(nn.Module):
    def __init__(self, hidden=128, dropout=.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(203, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.pair = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.left_dustbin = nn.Linear(hidden, 1)
        self.right_dustbin = nn.Linear(hidden, 1)

    def forward(self, left, right):
        l, r = self.encoder(left), self.encoder(right)
        a, b = l[:, :, None, :].expand(-1, -1, r.shape[1], -1), r[:, None, :, :].expand(-1, l.shape[1], -1, -1)
        pair = self.pair(torch.cat([a, b, torch.abs(a - b), a * b], -1)).squeeze(-1)
        return pair, self.left_dustbin(l).squeeze(-1), self.right_dustbin(r).squeeze(-1)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def car_only_view(x, y):
    """Apply class-0 view after frozen all-class candidate cap; preserve raw data."""
    n, m = len(x['left']), len(x['right'])
    li = np.flatnonzero(x['left_classes'] == 0)
    ri = np.flatnonzero(x['right_classes'] == 0)
    assert not (x['geometry_gate'] & (x['left_classes'][:, None] != x['right_classes'][None, :])).any()
    left_map = np.full(n + 1, -1, np.int64); left_map[li] = np.arange(len(li)); left_map[n] = len(li)
    right_map = np.full(m + 1, -1, np.int64); right_map[ri] = np.arange(len(ri)); right_map[m] = len(ri)
    a, b = {}, {}
    for key in ['left', 'left_classes', 'left_query_indices']: a[key] = x[key][li]
    for key in ['right', 'right_classes', 'right_query_indices']: a[key] = x[key][ri]
    for key in ['geometry_gate', 'innovation_distance_squared']: a[key] = x[key][np.ix_(li, ri)]
    for key in ['targets', 'supervised_pair_mask']: b[key] = y[key][np.ix_(li, ri)]
    for side, selected, mapping in [('left', li, right_map), ('right', ri, left_map)]:
        b[side + '_assignment_mask'] = y[side + '_assignment_mask'][selected]
        original = y[side + '_assignment'][selected]
        remapped = mapping[original]
        assert not (b[side + '_assignment_mask'] & (remapped < 0)).any()
        b[side + '_assignment'] = np.where(remapped < 0, len(ri) if side == 'left' else len(li), remapped)
    assert (a['left_classes'] == 0).all() and (a['right_classes'] == 0).all()
    return a, b


def load_data(root, config):
    manifest=json.loads((root/'manifest.json').read_text());accept=json.loads((root/'independent-full-readback-acceptance.json').read_text())
    assert sha(root/'manifest.json')==config['data_manifest_sha256']==accept['manifest_sha256']
    assert sha(root/'independent-full-readback-acceptance.json')==config['data_acceptance_sha256']
    assert manifest['fold_id']==config['fold_id']==accept['fold_id'] and accept['all_example_bytes_features_gates_targets_and_deadline_availability_verified'] is True
    assert manifest['fit_sequence_ids']==config['fit_sequence_ids'] and manifest['excluded_held_out_sequence_ids']==config['held_out_sequence_ids']
    assert sha(root/'examples.json')==manifest['example_index_sha256']
    assert config['class_scope']=='car' and config['class_index']==0
    examples=[];used=[];skipped=[]
    for row in json.loads((root/'examples.json').read_text()):
        values=[]
        for key in ['features','offline_targets']:
            p=root/row[key]['path'];assert sha(p)==row[key]['sha256'] and p.stat().st_size==row[key]['bytes']
            with np.load(p,allow_pickle=False) as z:values.append({k:z[k] for k in z.files})
        x,y=car_only_view(*values);n,m=len(x['left']),len(x['right']);identity=(row['sequence_id'],row['vehicle_frame_id'],row['infrastructure_frame_id'])
        if not n or not m or not (y['supervised_pair_mask'].any() or y['left_assignment_mask'].any() or y['right_assignment_mask'].any()):skipped.append(identity);continue
        assert n<=64 and m<=64 and x['left'].shape==(n,203) and x['right'].shape==(m,203)
        left=np.zeros((64,203),np.float32);right=np.zeros_like(left);left[:n]=x['left'];right[:m]=x['right']
        gate=np.zeros((64,64),bool);gate[:n,:m]=x['geometry_gate'];pair_mask=np.zeros_like(gate);pair_mask[:n,:m]=y['supervised_pair_mask']
        target=np.zeros((64,64),np.float32);target[:n,:m]=y['targets'];lm=np.zeros(64,bool);rm=np.zeros(64,bool);lm[:n]=y['left_assignment_mask'];rm[:m]=y['right_assignment_mask']
        la=np.full(64,64,np.int64);ra=np.full(64,64,np.int64);la[:n]=np.where(y['left_assignment']==m,64,y['left_assignment']);ra[:m]=np.where(y['right_assignment']==n,64,y['right_assignment'])
        assert np.isfinite(left).all() and np.isfinite(right).all()
        examples.append(tuple(torch.from_numpy(v) for v in [left,right,gate,pair_mask,target,lm,rm,la,ra]));used.append(identity)
    assert examples
    return examples,dict(used_pairs=used,skipped_empty_or_unsupervised_pairs=skipped,original_pairs=manifest['counts']['pairs'],class_scope='car',class_index=0,candidate_policy='class-0 view after unchanged raw-score>=0.05/all-class-top64',fit_sequences=config['fit_sequence_ids'])


def global_mean(numerator, local_count):
    denominator=local_count.detach().to(dtype=torch.float32).clone();dist.all_reduce(denominator)
    return numerator*dist.get_world_size()/denominator.clamp_min(1)


def worker(rank, world, port, data, output, config):
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    random.seed(config['seed']+rank);np.random.seed(config['seed']+rank);torch.manual_seed(config['seed'])
    if rank==0:print(json.dumps(dict(stage='association_runtime_admission',torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,python=platform.python_version(),NCCL_P2P_DISABLE=os.environ.get('NCCL_P2P_DISABLE'),eta_seconds=None)),flush=True)
    assert os.environ.get('NCCL_P2P_DISABLE')=='1'
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world,timeout=timedelta(seconds=120))
    assert platform.system()=='Linux' and torch.__version__=='2.6.0+cu124' and torch.version.cuda=='12.4'
    assert np.__version__=='1.26.4' and platform.python_version()=='3.12.3'
    device=torch.cuda.get_device_properties(rank)
    runtime=dict(rank=rank,host=socket.gethostname(),GPU_name=device.name,GPU_uuid=str(device.uuid),capability=list(torch.cuda.get_device_capability(rank)),torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,python=platform.python_version(),NCCL_P2P_DISABLE=1,collective_timeout_seconds=120)
    identities=[None]*world;dist.all_gather_object(identities,runtime)
    assert len({r['GPU_uuid'] for r in identities})==world and len({r['host'] for r in identities})==1
    # Check globally normalized masked BCE and the padded dustbin index before
    # using real data. Unknown alternatives receive zero supervision weight.
    fixture=torch.zeros((2,2),device=f'cuda:{rank}',requires_grad=True)
    fixture_mask=torch.tensor([[True,False],[False,True]],device=f'cuda:{rank}')
    fixture_loss=global_mean((F.binary_cross_entropy_with_logits(fixture,torch.zeros_like(fixture),reduction='none')*fixture_mask).sum(),fixture_mask.sum())
    assert torch.allclose(fixture_loss,torch.tensor(np.log(2),device=f'cuda:{rank}',dtype=torch.float32))
    fixture_loss.backward();assert torch.equal(fixture.grad[~fixture_mask],torch.zeros(2,device=f'cuda:{rank}'))
    fixture_logits=torch.full((1,65),-1e4,device=f'cuda:{rank}');fixture_logits[:,64]=0
    assert F.cross_entropy(fixture_logits,torch.tensor([64],device=f'cuda:{rank}'))==0
    dataset,coverage=load_data(Path(data),config)
    coverage_sha=hashlib.sha256(json.dumps(coverage,sort_keys=True).encode()).hexdigest();bindings=[None]*world;dist.all_gather_object(bindings,coverage_sha);assert len(set(bindings))==1
    if rank==0:
        (Path(output)/'runtime.json').write_text(json.dumps(dict(devices=identities,world_size=world,configuration=config),indent=2)+'\n')
        (Path(output)/'training-coverage.json').write_text(json.dumps(dict(**coverage,coverage_sha256=coverage_sha,distributed_sampler_padding_possible=True),indent=2)+'\n')
    model=nn.parallel.DistributedDataParallel(PredictedAssociation().cuda(rank),device_ids=[rank])
    torch.manual_seed(config['seed']+rank)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=config['seed'],drop_last=False)
    loader=DataLoader(dataset,batch_size=config['global_batch_size']//world,sampler=sampler,num_workers=0,drop_last=False)
    started=time.monotonic();total=config['epochs']*len(loader);step=0
    for epoch in range(1,config['epochs']+1):
        sampler.set_epoch(epoch);model.train();loss_sum=0.;epoch_steps=0
        for batch in loader:
            left,right,gate,mask,target,lm,rm,la,ra=[v.cuda(rank) for v in batch]
            pair,ld,rd=model(left,right)
            pair_loss=global_mean((F.binary_cross_entropy_with_logits(pair,target,reduction='none')*mask).sum(),mask.sum())
            gated=pair.masked_fill(~gate,-1e4)
            left_logits=torch.cat([gated,ld.unsqueeze(-1)],dim=-1);right_logits=torch.cat([gated.transpose(1,2),rd.unsqueeze(-1)],dim=-1)
            left_loss=global_mean((F.cross_entropy(left_logits.flatten(0,1),la.flatten(),reduction='none').reshape_as(lm)*lm).sum(),lm.sum())
            right_loss=global_mean((F.cross_entropy(right_logits.flatten(0,1),ra.flatten(),reduction='none').reshape_as(rm)*rm).sum(),rm.sum())
            loss=pair_loss+left_loss+right_loss
            assert torch.isfinite(loss), 'nonfinite training loss'
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip']);optimizer.step()
            loss_sum+=float(loss.detach());epoch_steps+=1;step+=1
            if rank==0 and (step%25==0 or step==total):
                elapsed=time.monotonic()-started;print(json.dumps(dict(kind='rbf_experiment_progress_v1',stage='canonical_oof_association_GPU_training',fold_id=config['fold_id'],seed=config['seed'],epoch=epoch,completed=step,total=total,eta_seconds=elapsed/step*(total-step),loss=float(loss.detach()))),flush=True)
        aggregate=torch.tensor(loss_sum,device=f'cuda:{rank}');dist.all_reduce(aggregate)
        if rank==0:
            with (Path(output)/'epochs.jsonl').open('a') as log:log.write(json.dumps(dict(epoch=epoch,steps=epoch_steps,mean_loss=float(aggregate)/(epoch_steps*world),elapsed_seconds=time.monotonic()-started))+'\n')
        dist.barrier()
    if rank==0:
        checkpoint=dict(kind='canonical_OOF_predicted_association_candidate_checkpoint_v1',model={k:v.cpu() for k,v in model.module.state_dict().items()},configuration=config,epoch=config['epochs'],seed=config['seed'],fold_id=config['fold_id'],feature_dimension=203,coverage_sha256=coverage_sha)
        torch.save(checkpoint,Path(output)/'checkpoint.pt')
        completion=dict(kind='canonical_OOF_association_full_fixed_epoch_candidate_training_completion',configuration=config,epoch=config['epochs'],checkpoint_sha256=sha(Path(output)/'checkpoint.pt'),training_coverage_sha256=sha(Path(output)/'training-coverage.json'),runtime_sha256=sha(Path(output)/'runtime.json'),epochs_sha256=sha(Path(output)/'epochs.jsonl'),complete_fit_input_inventory_admitted=True,held_out_GT_used=False,hard_negative_scenario_coverage_certified=False,pooled_OOF_selection_complete=False,paper_eligible=False)
        (Path(output)/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());assert config['global_batch_size']%4==0 and config['epochs']==24
    assert torch.cuda.device_count()==4;output=Path(a.output);output.mkdir(exist_ok=False)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(4,port,a.data,a.output,config),nprocs=4,join=True)
