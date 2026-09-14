#!/usr/bin/env python3
"""Four-GPU priority-head fitting on complete, train-only teacher exports.

Independent seeds may run on different four-GPU workers. One job stays on one
host; V100/CPU fallback is forbidden by the production CLI. CPU/Gloo is a
low-level test-only path. Equal-weight selection GROUPS, not candidate rows,
are partitioned without padding. Sequence holdout never updates parameters or
selects an epoch. This is not strict upstream-isolated training or final refit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tools.event_track_v2x.train_forest_identity_ddp import _gather,_rank_zero,ddp_sources,rank_rows
from transvision.models.event_track_v2x.allocation_policy import FEATURES,RECIPE,TARGET,FrozenPriorityPolicy
from transvision.models.event_track_v2x.allocation_training import DATA_KIND,KIND,SEEDS,allocation_sources,groups
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,contained_file,sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json


@dataclass(frozen=True)
class PriorityFitConfig:
    epochs:int=10
    hidden:int=32
    batch_groups:int=64
    learning_rate:float=.001
    gradient_clip:float=5.

    def __post_init__(self):
        if (type(self.epochs) is not int or self.epochs<1 or type(self.hidden) is not int
                or not 1<=self.hidden<=256 or type(self.batch_groups) is not int
                or not 1<=self.batch_groups<=1024
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
                       for v in (self.learning_rate,self.gradient_clip))):
            raise ValueError('positive bounded priority fitting configuration required')


def priority_ddp_sources():
    # The existing DDP helpers import the identity-training dependency closure.
    # Bind it explicitly rather than pretending this file is self-contained.
    return dict(ddp_sources(),**allocation_sources(),
                **{Path(__file__).relative_to(ROOT).as_posix():sha_file(__file__)})


def sequence_partition(records):
    scenes=[r['sequence_id'] for r in records]
    if (len(scenes)<2 or any(not isinstance(s,str) or not s for s in scenes)
            or len(set(scenes))!=len(scenes)):
        raise ValueError('at least two distinct nonempty train sequence IDs required')
    ordered=sorted(scenes,key=lambda s:(hashlib.sha256(('priority-holdout-v1:'+s).encode()).hexdigest(),s))
    held=set(ordered[:max(1,len(scenes)//5)])
    return tuple([r for r in records if (r['sequence_id'] in held)==v] for v in (False,True))


def audit_data(data,manifest_sha256,*,require_full_train=True):
    if type(require_full_train) is not bool: raise TypeError('explicit full-train requirement required')
    data=Path(data);path=contained_file(data,'manifest.json')
    if sha_file(path)!=manifest_sha256: raise ValueError('priority data manifest changed')
    manifest=json.loads(path.read_bytes())
    if (manifest['kind']!=DATA_KIND or manifest['split']!='train'
            or manifest['source_sha256']!=allocation_sources()
            or manifest['feature_recipe']!=RECIPE or manifest['target_recipe']!=TARGET
            or manifest['feature_names']!=list(FEATURES)
            or manifest['labels_are_model_not_true_risk'] is not True
            or manifest['future_or_gt_inputs'] is not False):
        raise ValueError('current-source, model-progress-only train data required')
    records=manifest['shards'];fit,held=sequence_partition(records)
    if (len({r['path'] for r in records})!=len(records)
            or any(type(r[k]) is not int or r[k]<0 for r in records for k in ('groups','rows'))):
        raise ValueError('distinct shard files and nonnegative integer counts required')
    if require_full_train and (manifest['full_official_train_trace'] is not True or len(records)!=46
            or any(re.fullmatch(r'\d{4}',r['sequence_id']) is None for r in records)
            or re.fullmatch(r'[0-9a-f]{64}',manifest['replay_receipt_sha256']) is None):
        raise ValueError('complete 46-sequence SPD train teacher export required; fixture cannot be relabelled')
    statistics={}
    for name,part in (('fit',fit),('holdout',held)):
        count=rows=signal=positive=negative=0;maximum=0.
        for record in part:
            for x,y in groups(data,record):
                count+=1;rows+=len(y);signal+=int(np.any(np.abs(y)>1e-12))
                positive+=int(np.count_nonzero(y>1e-12));negative+=int(np.count_nonzero(y< -1e-12))
                maximum=max(maximum,float(np.abs(y).max()))
        if count<1: raise ValueError('fit and held-out sequences both require nonempty groups')
        statistics[name]=dict(groups=count,rows=rows,nonzero_target_groups=signal,
            positive_targets=positive,negative_targets=negative,max_abs_target=maximum)
    if require_full_train and statistics['fit']['nonzero_target_groups']==0:
        raise ValueError('all fitted targets are numerically zero; no informative priority training claim')
    if sha_file(path)!=manifest_sha256 or manifest['source_sha256']!=allocation_sources():
        raise ValueError('priority data or sources changed during audit')
    return manifest,fit,held,statistics


def global_groups(data,records,size):
    if type(size) is not int or not 1<=size<=1024: raise ValueError('bounded group batch required')
    batch=[]
    for record in records:
        for group in groups(data,record):
            batch.append(group)
            if len(batch)==size:
                yield batch;batch=[]
    if batch: yield batch


def priority_model(hidden):
    return torch.nn.Sequential(torch.nn.Linear(len(FEATURES),hidden),torch.nn.Tanh(),
        torch.nn.Linear(hidden,1),torch.nn.Tanh()).double()


def policy(model):
    return FrozenPriorityPolicy([v.detach().cpu().numpy() for v in
        (model[0].weight,model[0].bias,model[2].weight,model[2].bias)])


class GroupLoss(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,local_groups,global_count,world_size):
        if (type(global_count) is not int or global_count<1 or type(world_size) is not int
                or world_size<1 or len(local_groups)>global_count):
            raise ValueError('valid global group count and world size required')
        if not local_groups:
            return sum(p.sum()*0. for p in self.model.parameters())
        device=next(self.model.parameters()).device
        values=[(self.model(torch.as_tensor(x,dtype=torch.float64,device=device))[:,0]
                 -torch.as_tensor(y,dtype=torch.float64,device=device)).square().mean() for x,y in local_groups]
        return torch.stack(values).sum()*(world_size/global_count)


def validate_runtime(runtimes,*,require_full_train):
    if not runtimes or any(r['world_size']!=len(runtimes) for r in runtimes):
        raise ValueError('complete actual rank inventory required')
    if sorted(r['rank'] for r in runtimes)!=list(range(len(runtimes))) or len({r['host'] for r in runtimes})!=1:
        raise ValueError('distinct ranks on one worker required')
    if require_full_train:
        if (len(runtimes)<4 or any(r['backend']!='nccl' or not r['device'].startswith('cuda:') for r in runtimes)
                or len({r['device'] for r in runtimes})!=len(runtimes)
                or len({r.get('gpu_uuid') for r in runtimes})!=len(runtimes)
                or any(not r.get('gpu_uuid') or r['gpu_uuid'] in ('unavailable','None') for r in runtimes)
                or any(not any(f in r.get('gpu_name','') for f in ('A100','5090')) for r in runtimes)):
            raise ValueError('at least four distinct actual A100/5090 CUDA/NCCL GPUs required; V100 excluded')


def fit_distributed(data,manifest_sha256,output,*,device,seed=1337,config=None,require_full_train=True):
    if not dist.is_initialized(): raise ValueError('initialized distributed process group required')
    if type(require_full_train) is not bool or type(seed) is not int or seed not in SEEDS:
        raise ValueError('explicit cohort and one predeclared seed required')
    config=config or PriorityFitConfig()
    if type(config) is not PriorityFitConfig: raise TypeError('validated priority configuration required')
    rank,world=dist.get_rank(),dist.get_world_size();device=torch.device(device)
    if require_full_train and (world<4 or device.type!='cuda' or dist.get_backend()!='nccl'):
        raise ValueError('full priority training requires at least four actual CUDA/NCCL ranks')
    if config.batch_groups<world: raise ValueError('global group batch must cover all ranks')
    runtime=dict(rank=rank,world_size=world,device=str(device),host=socket.gethostname(),
                 backend=dist.get_backend(),torch=torch.__version__,torch_cuda=torch.version.cuda)
    if device.type=='cuda':
        props=torch.cuda.get_device_properties(device)
        runtime.update(gpu_name=props.name,gpu_uuid=str(getattr(props,'uuid','unavailable')),
                       capability=[props.major,props.minor],total_memory_bytes=props.total_memory)
    runtimes=_gather(runtime);validate_runtime(runtimes,require_full_train=require_full_train)
    contract=dict(seed=seed,fit_config=asdict(config),manifest_sha256=manifest_sha256,
                  require_full_train=require_full_train)
    if any(value!=contract for value in _gather(contract)):
        raise ValueError('rank execution contracts differ before output or optimizer initialization')
    data,output=Path(data),Path(output).absolute();sources=priority_ddp_sources()
    if any(s!=sources for s in _gather(sources)): raise ValueError('rank sources differ')
    manifest,fit,held,statistics=_rank_zero(lambda:audit_data(data,manifest_sha256,require_full_train=require_full_train))
    if any(h!=manifest_sha256 for h in _gather(sha_file(contained_file(data,'manifest.json')))):
        raise ValueError('rank data manifests differ')
    plan=dict(kind='component_priority_ddp_plan_v1',training_manifest_sha256=manifest_sha256,
        training_source_sha256=sources,source_sha256=manifest['source_sha256'],seed=seed,fit_config=asdict(config),
        rank_runtime=runtimes,world_size=world,statistics=statistics,binding=manifest['binding'],
        fit_sequences=sorted(r['sequence_id'] for r in fit),holdout_sequences=sorted(r['sequence_id'] for r in held),
        global_batch_groups=config.batch_groups,partition='strided_global_groups_no_padding_no_drop',
        objective='equal_group_mean_squared_one_operation_model_progress',dtype='float64',
        gradient_reduction='DDP_mean_of_world_scaled_local_group_loss_sums_over_global_group_count',
        sequence_order='manifest_order_no_shuffle',selection='fixed_final_epoch_no_holdout_checkpoint_selection',
        full_official_train_trace=require_full_train,local_fixture_only=not require_full_train,
        upstream_may_be_in_sample=True,strict_pipeline_isolated_selection=False,final_full_train_refit=False,
        physical_speedup_claimed=False,paper_eligible=False)
    def create_output():
        if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
            raise ValueError('new nonsymlink priority output required')
        output.mkdir();_new_json(output/'plan.json',plan);return sha_file(output/'plan.json')
    plan_sha=_rank_zero(create_output);started=time.monotonic();directory=output/str(seed)
    try:
        _rank_zero(lambda:directory.mkdir());torch.manual_seed(seed)
        model=priority_model(config.hidden).to(device);initial=policy(model).signature
        ddp=DistributedDataParallel(GroupLoss(model),device_ids=[device.index] if device.type=='cuda' else None)
        optimizer=torch.optim.Adam(ddp.parameters(),lr=config.learning_rate)
        for epoch in range(1,config.epochs+1):
            epoch_start=time.monotonic();count=rows=batches=nonempty=positive=0;total=0.
            ddp.train()
            for batch in global_groups(data,fit,config.batch_groups):
                local=rank_rows(batch,rank,world);optimizer.zero_grad(set_to_none=True)
                loss=ddp(local,len(batch),world);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),config.gradient_clip,error_if_nonfinite=True)
                optimizer.step();count+=len(local);rows+=sum(len(y) for _,y in local);batches+=1
                nonempty+=bool(local);positive+=float(norm)>0;total+=float(loss.detach())*len(batch)/world
            model.eval();held_total=0.;held_count=held_rows=0
            with torch.no_grad():
                for batch in global_groups(data,held,config.batch_groups):
                    local=rank_rows(batch,rank,world)
                    held_total+=float(ddp.module(local,len(batch),world))*len(batch)/world
                    held_count+=len(local);held_rows+=sum(len(y) for _,y in local)
            progress=_gather(dict(rank=rank,fit_groups=count,fit_rows=rows,batches=batches,
                nonempty_batches=nonempty,positive_gradient_batches=positive,group_loss_sum=total,
                holdout_groups=held_count,holdout_rows=held_rows,holdout_group_loss_sum=held_total,
                policy_signature=policy(model).signature))
            if (sum(r['fit_groups'] for r in progress)!=statistics['fit']['groups']
                    or sum(r['fit_rows'] for r in progress)!=statistics['fit']['rows']
                    or sum(r['holdout_groups'] for r in progress)!=statistics['holdout']['groups']
                    or sum(r['holdout_rows'] for r in progress)!=statistics['holdout']['rows']
                    or len({r['batches'] for r in progress})!=1
                    or len({r['policy_signature'] for r in progress})!=1
                    or require_full_train and any(r['nonempty_batches']==0 or r['positive_gradient_batches']==0 for r in progress)):
                raise ValueError('group/row coverage or synchronized actual rank work failed')
            entry=dict(epoch=epoch,rank_progress=progress,fit_groups=statistics['fit']['groups'],
                holdout_groups=statistics['holdout']['groups'],global_batches=batches,
                training_mse=sum(r['group_loss_sum'] for r in progress)/statistics['fit']['groups'],
                train_sequence_holdout_mse=sum(r['holdout_group_loss_sum'] for r in progress)/statistics['holdout']['groups'],
                elapsed_seconds=time.monotonic()-epoch_start,official_validation_or_test_read=False)
            def save_epoch():
                with (directory/'epochs.jsonl').open('ab') as stream:stream.write(canonical(entry)+b'\n')
                print(json.dumps(entry,sort_keys=True),flush=True)
            _rank_zero(save_epoch)
        frozen=policy(model)
        if frozen.signature==initial: raise ValueError('optimizer did not change priority weights')
        if any(s!=sources for s in _gather(priority_ddp_sources())): raise ValueError('priority training sources changed')
        _rank_zero(lambda:audit_data(data,manifest_sha256,require_full_train=require_full_train))
        def save_final():
            with (directory/'weights.npz').open('xb') as stream:
                np.savez(stream,**{'w'+str(i):w for i,w in enumerate(frozen.weights)})
            checkpoint=dict(kind=KIND,feature_recipe=RECIPE,feature_names=FEATURES,target_recipe=TARGET,
                seed=seed,split='train',binding=manifest['binding'],source_sha256=allocation_sources(),
                training_source_sha256=sources,weights_sha256=sha_file(directory/'weights.npz'),
                policy_signature=frozen.signature,initial_policy_signature=initial,
                full_official_train_trace=require_full_train,head_fit_sequence_isolated=True,head_fit_includes_holdout=False,
                fit_sequences=plan['fit_sequences'],holdout_sequences=plan['holdout_sequences'],
                strict_pipeline_isolated_selection=False,paper_eligible=False,final_full_train_refit=False,
                training_manifest_sha256=manifest_sha256,plan_sha256=plan_sha,
                distributed_world_size=world,distributed_backend=dist.get_backend(),fixed_final_epoch=config.epochs)
            _new_json(directory/'checkpoint.json',checkpoint)
            receipt=dict(kind='component_priority_ddp_receipt_v1',status='complete',seed=seed,
                plan_sha256=plan_sha,checkpoint_manifest=directory.name+'/checkpoint.json',
                checkpoint_sha256=sha_file(directory/'checkpoint.json'),epochs_sha256=sha_file(directory/'epochs.jsonl'),
                rank_runtime=runtimes,world_size=world,full_official_train_trace=require_full_train,
                local_fixture_only=not require_full_train,elapsed_seconds=time.monotonic()-started,
                complete_three_seed_campaign=False,tracking_validation_performed=False,paper_eligible=False)
            _new_json(output/'receipt.json',receipt);return receipt
        return _rank_zero(save_final)
    except BaseException as error:
        if rank==0:
            _new_json(output/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
                partial_outputs_not_final_results=True,paper_eligible=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--global-batch-groups',type=int,default=64)
    a=p.parse_args();world=int(os.environ.get('WORLD_SIZE','0'));local=int(os.environ.get('LOCAL_WORLD_SIZE','0'))
    rank=int(os.environ.get('LOCAL_RANK','-1'))
    if world<4 or local!=world or not 0<=rank<world or torch.cuda.device_count()<world:
        raise ValueError('torchrun on one worker with at least four allocated CUDA GPUs required')
    torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    try:
        print(json.dumps(fit_distributed(a.data,a.manifest_sha256,a.output,device=f'cuda:{rank}',seed=a.seed,
            config=PriorityFitConfig(epochs=a.epochs,batch_groups=a.global_batch_groups)),sort_keys=True))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
