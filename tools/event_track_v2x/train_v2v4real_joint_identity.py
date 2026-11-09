#!/usr/bin/env python3
"""Actual native train-name-group selection and full-train joint identity refit.

Requires independently admitted full real V2V rows and exactly 4/8 CUDA ranks.
No fixture/CPU fallback, changed partition, test-driven choice or SPD reuse.
"""
import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import types
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    # Load the frozen tracking modules directly, without executing unrelated
    # detection registries or requiring a full MMDetection environment.
    for name in ('transvision','transvision.models','transvision.models.event_track_v2x'):
        module=types.ModuleType(name);module.__path__=[str(ROOT.joinpath(*name.split('.')))]
        sys.modules[name]=module
from tools.event_track_v2x.v2v4real_motion_fit_inputs import read,sha
from tools.event_track_v2x.v2v4real_identity_training_contract import FIT_CONFIG,HEADS,admitted_roles,macro_score,select_epoch,local_global_head_loss
from tools.event_track_v2x.train_forest_identity import FitConfig
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file
from transvision.models.event_track_v2x.forest_training_data import TrainingShard,row_protocol
from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig
from transvision.models.event_track_v2x.forest_training import batched_row_logits
from transvision.models.event_track_v2x.forest_row_context import ForestRowContext
from transvision.models.event_track_v2x.paper_calibration import separate_identity_losses
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from transvision.models.event_track_v2x.recoverable_identity import model_digest
from transvision.models.event_track_v2x.forest_training_checkpoint import CHECKPOINT_KIND


def new(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def cpu_state(model):
    # Preserve the established independent weights reader's CPU-only storage
    # contract; copying device placement does not change parameter values.
    return {name:value.detach().cpu() for name,value in model.state_dict().items()}


def rank_zero(operation):
    answer=[None]
    if dist.get_rank()==0:
        try:answer[0]=dict(value=operation(),error=None)
        except Exception as e:answer[0]=dict(error=type(e).__name__+': '+str(e))
    dist.broadcast_object_list(answer,src=0)
    if answer[0]['error']:raise ValueError(answer[0]['error'])
    return answer[0]['value']


def gather(value):
    result=[None]*dist.get_world_size();dist.all_gather_object(result,value);return result


class JointHeadLoss(torch.nn.Module):
    def __init__(self,model,config,protocol):
        super().__init__();self.model=model;self.config=config;self.protocol=protocol

    def forward(self,examples):
        selected=examples[dist.get_rank()::dist.get_world_size()]
        zero=sum(p.sum()*0 for p in self.model.parameters())
        if selected:
            contexts,targets=zip(*selected)
            logits=batched_row_logits(self.model,contexts,max_nodes=9,max_batch=self.config.batch_size,
                geometry_weight=self.config.geometry_weight,process_noise=self.protocol['process_noise'])
            value=separate_identity_losses(logits,contexts,targets)
        else:value=dict(cross_source=zero,temporal=zero,counts=dict(cross_source=0,temporal=0))
        counts=torch.tensor([value['counts'][h] for h in HEADS],device=zero.device,dtype=torch.int64)
        dist.all_reduce(counts);global_counts=dict(zip(HEADS,counts.cpu().tolist()))
        loss=local_global_head_loss({h:value[h] for h in HEADS},value['counts'],global_counts,dist.get_world_size())
        sums=torch.stack([value[h].detach()*value['counts'][h] for h in HEADS]);dist.all_reduce(sums)
        return dict(loss=loss+zero,global_counts=global_counts,global_sums=dict(zip(HEADS,sums.cpu().tolist())),local_rows=len(selected))


def phase(model,ddp,optimizer,records,data,config,protocol,*,train,epoch,stage):
    model.train(train);ddp.train(train);results=[];rows=local_rows=batches=updates=0;maximum_gradient=0.
    for record in records:
        shard=TrainingShard(contained_file(data,record['path']),record,protocol['parent_limit'])
        stats={h:dict(loss_sum=0.,samples=0) for h in HEADS};seen=0
        for start in range(0,len(shard.valid_indices),config.batch_size):
            examples=[shard.example(int(i)) for i in shard.valid_indices[start:start+config.batch_size]]
            if train:optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                value=ddp(examples)
                if train and sum(value['global_counts'].values())>0:
                    value['loss'].backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip,error_if_nonfinite=True)
                    maximum_gradient=max(maximum_gradient,float(norm));optimizer.step();updates+=1
            for h in HEADS:
                stats[h]['loss_sum']+=value['global_sums'][h];stats[h]['samples']+=value['global_counts'][h]
            seen+=len(examples);local_rows+=value['local_rows'];batches+=1
        if seen!=record['supervised_rows']:raise ValueError('phase omitted supervised native rows')
        rows+=seen;results.append(dict(sequence_id=record['sequence_id'],supervised_rows=seen,**stats))
        if dist.get_rank()==0:print(json.dumps(dict(stage=stage,phase='fit' if train else 'holdout',epoch=epoch,
            sequence_id=record['sequence_id'],supervised_rows=seen,completed_sequences=len(results),total_sequences=len(records))),flush=True)
        del shard
    rank_counts=gather(dict(rank=dist.get_rank(),local_rows=local_rows,batches=batches,model_sha256=model_digest(model)))
    if (sum(x['local_rows'] for x in rank_counts)!=rows or len({x['model_sha256'] for x in rank_counts})!=1):
        raise ValueError('strided native rows or synchronized rank parameters differ')
    return dict(sequences=results,supervised_rows=rows,global_batches=batches,optimizer_steps=updates,
        maximum_preclip_gradient_norm=maximum_gradient,rank_coverage=rank_counts,**macro_score(results))


def initialize(seed,rank,config,protocol,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=RecoverableIdentityModel(hidden=config.hidden,heads=config.heads,dropout=config.dropout).to(device)
    initial=model_digest(model);module=JointHeadLoss(model,config,protocol)
    ddp=DistributedDataParallel(module,device_ids=[rank])
    torch.manual_seed(seed+rank);torch.cuda.manual_seed_all(seed+rank)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    return model,ddp,optimizer,initial


def forward_all(model,records,data,out,config,protocol):
    """Every raw row, including unknown targets; target masks select no rows."""
    rank=dist.get_rank();world=dist.get_world_size();count=0;sequences=[]
    destination=out/f'predictions-rank{rank}.jsonl'
    with destination.open('x') as stream,torch.inference_mode():
        for record in records[rank::world]:
            shard=TrainingShard(contained_file(data,record['path']),record,protocol['parent_limit']);a=shard.arrays
            for start in range(0,record['nodes'],config.batch_size):
                indices=list(range(start,min(start+config.batch_size,record['nodes'])));contexts=[]
                for i in indices:
                    length=int(a['lengths'][i]);support=tuple(map(int,a['contexts'][i,:length]))
                    contexts.append(ForestRowContext(support,tuple(shard._observation(j) for j in support),int(a['decision_us'][i])))
                logits=batched_row_logits(model,contexts,max_nodes=protocol['parent_limit']+1,
                    max_batch=config.batch_size,geometry_weight=config.geometry_weight,process_noise=protocol['process_noise'])
                for i,c,v in zip(indices,contexts,logits):
                    stream.write(json.dumps(dict(sequence_id=record['sequence_id'],row=i,node_id=str(a['node_id'][i]),
                        decision_us=int(a['decision_us'][i]),context_indices=list(c.indices),logits=v.cpu().tolist()),allow_nan=False)+'\n')
                count+=len(indices)
            sequences.append(dict(sequence_id=record['sequence_id'],rows=record['nodes'],source_shard_sha256=record['sha256']))
            print(json.dumps(dict(stage='all_native_rows_final_weight_forward',rank=rank,sequence_id=record['sequence_id'],
                completed_rows=count,independent_numeric_accepted=False)),flush=True)
            del shard
    results=gather(dict(rank=rank,world_size=world,rows=count,sequences=sequences,path=destination.name,
        sha256=sha(destination),model_sha256=model_digest(model),target_masks_used_to_select_forward_rows=False))
    if (sum(r['rows'] for r in results)!=sum(r['nodes'] for r in records)
            or len({r['model_sha256'] for r in results})!=1):raise ValueError('full original row forward coverage or rank weights differ')
    return results


def run(args):
    # Split/qualification admission occurs before any training-row payload or
    # optimizer update. Raw YAML and original test arrays are never transported.
    pins={};data=Path(args.data).absolute();out=Path(args.output).absolute()
    manifest=json.loads(read(data/'manifest.json',args.manifest_sha256,pins))
    proof=json.loads(read(args.row_admission,args.row_admission_sha256,pins))
    partition=json.loads(read(args.partition,args.partition_sha256,pins))
    records=admitted_roles(manifest,proof,partition,args.seed,args.manifest_sha256)
    if manifest['row_protocol']!=row_protocol(PaperForestTrackingConfig()):
        raise ValueError('full native row feature/context/arrival protocol differs')
    if manifest['provenance']['original_nested_partition_sha256']!=args.partition_sha256:
        raise ValueError('original partition SHA changed')
    for record in manifest['shards']:
        p=contained_file(data,record['path'])
        if sha(p)!=record['sha256']:raise ValueError('admitted full native training shard bytes differ')
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world not in (4,8) or not torch.cuda.is_available() or torch.cuda.device_count()!=world:
        raise ValueError('exactly four/eight allocated CUDA devices required; no CPU fallback')
    if any('L40' in torch.cuda.get_device_name(i) for i in range(world)):
        raise ValueError('L40S allocation remains CPU-only')
    torch.cuda.set_device(rank);torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=30));device=torch.device('cuda',rank)
    props=torch.cuda.get_device_properties(rank);uuid=str(getattr(props,'uuid','unavailable'))
    if uuid=='unavailable':raise ValueError('actual physical GPU UUID required')
    runtime=dict(rank=rank,world_size=world,device=str(device),gpu_name=props.name,gpu_uuid=uuid,
        host=socket.gethostname(),torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
    runtimes=gather(runtime)
    if len({r['gpu_uuid'] for r in runtimes})!=world or len({r['host'] for r in runtimes})!=1:
        raise ValueError('one worker with distinct physical GPUs required')
    config=FitConfig(**FIT_CONFIG);protocol=manifest['row_protocol']
    source_paths=[Path(__file__),ROOT/'tools/event_track_v2x/v2v4real_identity_training_contract.py',ROOT/'tools/event_track_v2x/v2v4real_motion_fit_inputs.py',ROOT/'tools/event_track_v2x/train_forest_identity.py']+sorted((ROOT/'transvision/models/event_track_v2x').glob('*.py'))
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}
    if len({json.dumps(x,sort_keys=True) for x in gather(sources)})!=1:raise ValueError('rank source identities differ')
    plan=dict(kind='v2v4real_native_joint_identity_selection_full_refit_plan_v1',seed=args.seed,fit_config=asdict(config),
        manifest_sha256=args.manifest_sha256,row_admission_sha256=args.row_admission_sha256,partition_sha256=args.partition_sha256,
        original_sequence_to_name_group=partition['sequence_to_name_group'],fit_sequences=[r['sequence_id'] for r in records['fit']],
        holdout_sequences=[r['sequence_id'] for r in records['holdout']],row_protocol=protocol,source_sha256=sources,
        objective='sum_of_named_cross_source_and_temporal_global_mean_partial_label_surrogates',
        checkpoint_selection='minimum_sum_of_two_head_train_holdout_sequence_macro_means_then_earliest_epoch',
        refit='same_seed_new_initialization_all_32_original_train_sequences_selected_epoch_count',
        row_order='chronological_within_sequence_strided_global_batches_no_padding_no_drop',
        zero_informative_batch_policy='skip_optimizer_update_after_all_ranks_agree',rank_runtime=runtimes,
        nominal_rate_hz=10,upstream_sequence_isolated=partition['upstream_sequence_isolated'],
        train_test_overlap_disclosed=True,physical_session_independence_claimed=False,
        strict_pipeline_isolated_selection=False,official_test_labels_read=False,checkpoint_independently_qualified=False,paper_metrics_complete=False)
    def create():
        if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('fresh separate native training output required')
        out.mkdir(parents=True);new(out/'plan-before-updates.json',plan);(out/'selection').mkdir();(out/'final').mkdir()
    rank_zero(create);started=time.monotonic();history=[]
    try:
        model,ddp,optimizer,initial=initialize(args.seed,rank,config,protocol,device)
        rank_zero(lambda:torch.save(cpu_state(model),out/'initial-weights.pt'))
        for epoch in range(1,11):
            fit=phase(model,ddp,optimizer,records['fit'],data,config,protocol,train=True,epoch=epoch,stage='native_checkpoint_selection')
            holdout=phase(model,ddp,optimizer,records['holdout'],data,config,protocol,train=False,epoch=epoch,stage='native_checkpoint_selection')
            if fit['optimizer_steps']<1:raise ValueError('epoch made no informative optimizer updates')
            entry=dict(epoch=epoch,fit=fit,holdout=holdout,model_sha256=model_digest(model));history.append(entry)
            def save_selection():
                folder=out/'selection'/f'epoch-{epoch:04d}';folder.mkdir()
                torch.save(cpu_state(model),folder/'weights.pt');new(folder/'epoch.json',entry)
                return sha(folder/'weights.pt')
            rank_zero(save_selection)
        selected=select_epoch(history)
        rank_zero(lambda:new(out/'selection/selected-epoch.json',dict(selected_epoch=selected,history=history,official_test_read=False)))
        del optimizer,ddp,model;torch.cuda.empty_cache()
        model,ddp,optimizer,refit_initial=initialize(args.seed,rank,config,protocol,device)
        if refit_initial!=initial:raise ValueError('full refit initialization differs from same-seed selection')
        refit_history=[]
        for epoch in range(1,selected+1):
            stat=phase(model,ddp,optimizer,manifest['shards'],data,config,protocol,train=True,epoch=epoch,stage='native_full_train_refit')
            if stat['optimizer_steps']<1:raise ValueError('full refit made no informative optimizer update')
            entry=dict(epoch=epoch,fit=stat,model_sha256=model_digest(model));refit_history.append(entry)
            rank_zero(lambda:new(out/'final'/f'epoch-{epoch:04d}.json',entry))
        model.eval().requires_grad_(False)
        if model_digest(model)==initial:raise ValueError('native final weights unchanged from initialization')
        forward=forward_all(model,manifest['shards'],data,out,config,protocol)
        for p,h in pins.items():
            if sha(p)!=h:raise ValueError('native training declarations changed')
        for name,h in sources.items():
            if sha(ROOT/name)!=h:raise ValueError('native training/scoring source changed')
        def save_final():
            folder=out/'final';torch.save(cpu_state(model),folder/'weights.pt')
            checkpoint=dict(kind=CHECKPOINT_KIND,architecture=dict(hidden=128,heads=4,dropout=.1),seed=args.seed,dataset='v2v4real',
                row_protocol=protocol,data_split='train',dataset_sha256=args.manifest_sha256,model_sha256=model_digest(model),
                initial_model_sha256=initial,weights=dict(path='weights.pt',sha256=sha(folder/'weights.pt')),geometry_weight=1.,
                labels_in_model_inputs=False,local_partial_label_objective=True,frozen_cache_identity=manifest['frozen_cache_identity'],
                source_sha256=sources,plan_sha256=sha(out/'plan-before-updates.json'),selected_epoch=selected,
                selection='original_train_name_group_holdout_then_full_train_refit',partition=dict(fit=manifest['sequences'],holdout=[]),
                selection_partition=dict(fit=plan['fit_sequences'],holdout=plan['holdout_sequences']),
                selection_receipt_sha256=sha(out/'selection/selected-epoch.json'),distributed_world_size=world,
                native_class_scope=['vehicle'],network_class_slot_mapping={'car':'vehicle'},strict_pipeline_isolated_selection=False,
                validation_or_test_selection=False,train_test_overlap_disclosed=True,exact_joint_nll=False,fixture=False,paper_eligible=False)
            new(folder/'checkpoint.json',checkpoint)
            new(out/'receipt.json',dict(kind='v2v4real_native_joint_identity_selection_full_refit_candidate_v1',seed=args.seed,
                completed=True,selected_epoch=selected,checkpoint_sha256=sha(folder/'checkpoint.json'),weights_sha256=sha(folder/'weights.pt'),
                plan_sha256=sha(out/'plan-before-updates.json'),refit_history=refit_history,runtime=runtimes,all_row_forward=forward,
                initial_weights_sha256=sha(out/'initial-weights.pt'),seconds=time.monotonic()-started,
                checkpoint_independently_qualified=False,paper_metrics_complete=False))
        rank_zero(save_final)
    except Exception as e:
        rank_zero(lambda:new(out/'failure.json',dict(error_type=type(e).__name__,error=str(e),completed=False,
            selection_epochs_completed=len(history),original_failure_preserved=True)))
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[1337,2027,3407],required=True)
    for name in ('data','manifest-sha256','row-admission','row-admission-sha256','partition','partition-sha256','output'):
        p.add_argument('--'+name,required=True)
    run(p.parse_args())
