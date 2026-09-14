#!/usr/bin/env python3
"""Actual multi-GPU row-identity training; launch with torchrun on one worker.

Global batches are partitioned without padding or dropping rows. DDP averages
rank gradients, so each local loss sum is scaled by world_size/global_rows.
An empty tail rank joins backward with zero contribution, not duplicated data.
Only rank zero writes final artifacts. GPU training is development evidence,
not strict OOF selection or tracking validation. CPU/Gloo is test-only.
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tools.event_track_v2x.train_forest_identity import FitConfig, SEEDS, audit_dataset, training_sources
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_supervision import set_valued_parent_loss
from transvision.models.event_track_v2x.forest_training import batched_row_logits
from transvision.models.event_track_v2x.forest_training_checkpoint import CHECKPOINT_KIND
from transvision.models.event_track_v2x.forest_training_data import TrainingShard, _new_json
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from transvision.models.event_track_v2x.recoverable_identity import model_digest


def ddp_sources():
    return dict(training_sources(), **{Path(__file__).relative_to(ROOT).as_posix(): sha_file(Path(__file__))})


def rank_rows(indices, rank, world_size):
    if type(world_size) is not int or world_size < 1 or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError('valid rank and world size required')
    return indices[rank::world_size]


class RowLoss(torch.nn.Module):
    def __init__(self, model, config, protocol):
        super().__init__()
        self.model, self.config, self.protocol = model, config, protocol

    def forward(self, examples, global_rows, world_size):
        if not 1 <= global_rows <= self.config.batch_size or world_size < 1:
            raise ValueError('positive bounded global batch required')
        if not examples:
            # No artificial example, no label, and no sample counted twice.
            return sum(p.sum()*0. for p in self.model.parameters())
        contexts, targets = zip(*examples)
        logits = batched_row_logits(self.model, contexts, max_nodes=self.protocol['parent_limit']+1,
            max_batch=self.config.batch_size, geometry_weight=self.config.geometry_weight,
            process_noise=self.protocol['process_noise'])
        loss = set_valued_parent_loss(logits, targets)
        if loss['supervised_rows'] != len(examples):
            raise ValueError('DDP partition contains unsupervised rows')
        return loss['loss']*(len(examples)*world_size/global_rows)


def _gather(value):
    result = [None]*dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def _rank_zero(operation):
    message = [None]
    if dist.get_rank() == 0:
        try:
            message[0] = dict(value=operation(), error=None)
        except Exception as error:
            message[0] = dict(error=f'{type(error).__name__}: {error}')
    dist.broadcast_object_list(message, src=0)
    if message[0]['error']:
        raise ValueError(message[0]['error'])
    return message[0]['value']


def fit_distributed(data, manifest_sha256, output, *, config=None, device, require_full_train=True, seeds=SEEDS):
    if not dist.is_initialized():
        raise ValueError('initialized distributed process group required')
    config, data, output = config or FitConfig(), Path(data), Path(output).absolute()
    device = torch.device(device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if type(config) is not FitConfig or type(require_full_train) is not bool:
        raise TypeError('validated training configuration required')
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError('distinct predeclared identity seeds required')
    if require_full_train and (world < 4 or device.type != 'cuda' or dist.get_backend() != 'nccl'):
        raise ValueError('full training requires at least four actual CUDA/NCCL ranks')
    if config.batch_size < world:
        raise ValueError('global batch must allow a nonempty partition on every rank')
    runtime = dict(rank=rank, world_size=world, device=str(device), host=socket.gethostname(),
                   backend=dist.get_backend(), torch=torch.__version__, torch_cuda=torch.version.cuda)
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        if not any(name in props.name for name in ('A100', '5090')):
            raise ValueError('only A100 or RTX 5090 allowed; V100 explicitly excluded')
        runtime.update(gpu_name=props.name, gpu_uuid=str(getattr(props, 'uuid', 'unavailable')),
                       total_memory_bytes=props.total_memory, capability=[props.major, props.minor])
    runtimes = _gather(runtime)
    if len({(r['host'], r['device']) for r in runtimes}) != world and device.type == 'cuda':
        raise ValueError('ranks must use distinct allocated GPUs')
    if len({r['host'] for r in runtimes}) != 1:
        raise ValueError('this contract requires one multi-GPU worker, not cross-host training')
    sources = ddp_sources()
    if any(s != sources for s in _gather(sources)):
        raise ValueError('rank source identities differ')
    manifest, _, total_rows, informative = _rank_zero(lambda: audit_dataset(data, manifest_sha256,
                                                        require_full_train=require_full_train))
    protocol, records = manifest['row_protocol'], manifest['shards']
    if any(s != manifest_sha256 for s in _gather(sha_file(contained_file(data, 'manifest.json')))):
        raise ValueError('rank data manifest identities differ')
    plan = dict(kind='persistent_forest_identity_ddp_plan_v1', dataset_sha256=manifest_sha256,
        source_sha256=sources, seeds=seeds, required_campaign_seeds=SEEDS,
        fit_config=asdict(config), world_size=world, rank_runtime=runtimes,
        global_batch_size=config.batch_size, partition='strided_global_batch_no_padding_no_drop',
        gradient_reduction='DDP_mean_of_world_scaled_local_sums_over_global_rows',
        row_protocol=protocol, supervised_rows=total_rows, informative_rows=informative,
        dataset_sequences=manifest['sequences'], full_official_train=require_full_train,
        local_fixture_only=not require_full_train, strict_pipeline_isolated_selection=False,
        upstream_provenance=manifest['provenance'], selection='fixed_final_epoch_no_validation_search',
        paper_eligible=False, GPU_bitwise_reproducibility_claimed=False)

    def create_output():
        if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError('new output without symlink traversal required')
        output.mkdir()
        _new_json(output/'plan.json', plan)
        return sha_file(output/'plan.json')

    plan_sha = _rank_zero(create_output)
    started, results = time.monotonic(), []
    try:
        for seed in seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = RecoverableIdentityModel(hidden=config.hidden, heads=config.heads, dropout=config.dropout).to(device)
            initial_sha = model_digest(model)
            module = RowLoss(model, config, protocol)
            ddp = DistributedDataParallel(module, device_ids=[device.index] if device.type == 'cuda' else None)
            # Rank-specific stochastic masks, same synchronized initialization.
            torch.manual_seed(seed+rank)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed+rank)
            optimizer = torch.optim.AdamW(ddp.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            generator = np.random.default_rng(seed)
            directory = output/f'seed-{seed}'
            _rank_zero(lambda: directory.mkdir())
            for epoch in range(1, config.epochs+1):
                epoch_start = time.monotonic()
                count = batches = nonempty_batches = positive_gradient_batches = 0
                summed_loss = max_norm = 0.
                ddp.train()
                for index in generator.permutation(len(records)):
                    record = records[int(index)]
                    shard = TrainingShard(contained_file(data, record['path']), record, protocol['parent_limit'])
                    indices = generator.permutation(shard.valid_indices)
                    for start in range(0, len(indices), config.batch_size):
                        group = indices[start:start+config.batch_size]
                        local = rank_rows(group, rank, world)
                        examples = [shard.example(int(i)) for i in local]
                        optimizer.zero_grad(set_to_none=True)
                        loss = ddp(examples, len(group), world)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), config.gradient_clip, error_if_nonfinite=True)
                        optimizer.step()
                        count += len(local); batches += 1; nonempty_batches += bool(len(local))
                        positive_gradient_batches += float(norm) > 0
                        summed_loss += loss.detach().item()*len(group)/world
                        max_norm = max(max_norm, float(norm))
                    del shard
                progress = _gather(dict(rank=rank, supervised_rows=count, batches=batches,
                    nonempty_batches=nonempty_batches, positive_gradient_batches=positive_gradient_batches,
                    loss_sum=summed_loss, max_preclip_gradient_norm=max_norm, model_sha256=model_digest(model)))
                if (sum(r['supervised_rows'] for r in progress) != total_rows
                        or len({r['batches'] for r in progress}) != 1
                        or len({r['model_sha256'] for r in progress}) != 1
                        or (require_full_train and any(r['nonempty_batches'] == 0
                            or r['positive_gradient_batches'] == 0 for r in progress))):
                    raise ValueError('row coverage, actual rank work or synchronized weights failed')
                entry = dict(seed=seed, epoch=epoch, supervised_rows=total_rows, global_batches=batches,
                    local_training_loss=sum(r['loss_sum'] for r in progress)/total_rows,
                    rank_progress=progress, elapsed_seconds=time.monotonic()-epoch_start,
                    validation_metrics_read=False)
                def save_epoch():
                    with (directory/'epochs.jsonl').open('ab') as log:
                        log.write((json.dumps(entry, sort_keys=True, allow_nan=False)+'\n').encode())
                    print(json.dumps(entry, sort_keys=True), flush=True)
                _rank_zero(save_epoch)
            model.eval().requires_grad_(False)
            final_sha = model_digest(model)
            if final_sha == initial_sha:
                raise ValueError('optimizer did not change identity weights')
            def save_model():
                with (directory/'weights.pt').open('xb') as stream:
                    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, stream)
                checkpoint = dict(kind=CHECKPOINT_KIND, architecture=dict(hidden=config.hidden, heads=config.heads,
                    dropout=config.dropout), seed=seed, row_protocol=protocol, data_split='train',
                    dataset_sha256=manifest_sha256, model_sha256=final_sha, initial_model_sha256=initial_sha,
                    weights=dict(path='weights.pt', sha256=sha_file(directory/'weights.pt')),
                    geometry_weight=config.geometry_weight, local_partial_label_objective=True, labels_in_model_inputs=False,
                    frozen_cache_identity=manifest['frozen_cache_identity'], source_sha256=sources,
                    plan_sha256=plan_sha, full_official_train=require_full_train, paper_eligible=False,
                    calibration_guarantee=False, validation_or_test_selection=False,
                    strict_pipeline_isolated_selection=False, fixed_final_epoch=config.epochs,
                    distributed_world_size=world, distributed_backend=dist.get_backend())
                _new_json(directory/'checkpoint.json', checkpoint)
                return dict(seed=seed, checkpoint_manifest=directory.name+'/checkpoint.json',
                    checkpoint_sha256=sha_file(directory/'checkpoint.json'), epochs_sha256=sha_file(directory/'epochs.jsonl'))
            results.append(_rank_zero(save_model))
            del ddp, module, optimizer, model
        if any(s != sources for s in _gather(ddp_sources())):
            raise ValueError('training sources changed during distributed fitting')
        if any(s != manifest_sha256 for s in _gather(sha_file(contained_file(data, 'manifest.json')))):
            raise ValueError('training manifest changed during distributed fitting')
        receipt = dict(kind='persistent_forest_identity_ddp_receipt_v1', status='complete', seeds=results,
            plan_sha256=plan_sha, world_size=world, rank_runtime=runtimes,
            full_official_train=require_full_train, local_fixture_only=not require_full_train,
            required_campaign_seeds=SEEDS, complete_three_seed_campaign=set(seeds) == set(SEEDS),
            elapsed_seconds=time.monotonic()-started, paper_eligible=False, tracking_validation_performed=False)
        _rank_zero(lambda: _new_json(output/'receipt.json', receipt))
        return receipt
    except BaseException as error:
        if rank == 0:
            _new_json(output/'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error),
                completed_seeds=len(results), partial_outputs_not_final_results=True, paper_eligible=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--global-batch-size', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=SEEDS)
    args = parser.parse_args()
    world, local_world = int(os.environ.get('WORLD_SIZE', '0')), int(os.environ.get('LOCAL_WORLD_SIZE', '0'))
    rank = int(os.environ.get('LOCAL_RANK', '-1'))
    if world < 4 or local_world != world or not 0 <= rank < world or torch.cuda.device_count() < world:
        raise ValueError('torchrun on one worker with at least four allocated CUDA GPUs required')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    try:
        fit_distributed(args.data, args.manifest_sha256, args.output,
            config=FitConfig(epochs=args.epochs, batch_size=args.global_batch_size), device=f'cuda:{rank}', seeds=args.seeds)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
