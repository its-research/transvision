"""Refit frozen nested-selected all-class identity recipes on full SPD train.

The published training rows and model remain immutable. The separate runner
uses the original two-head loss and strided DDP partition, with no GPU-family
restriction and no new model selection. Runtime admission precedes updates.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import types


def install_source(root):
    sys.path.insert(0, str(root))
    for name in ('transvision', 'transvision.models', 'transvision.models.event_track_v2x'):
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split('.'))) ]
        sys.modules[name] = module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(base, recipe):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    install_source(base / 'source')
    from tools.event_track_v2x.train_forest_identity import FitConfig, audit_dataset
    from tools.event_track_v2x.train_forest_identity_ddp import _rank_zero, _gather, rank_rows
    from transvision.models.event_track_v2x.forest_training import batched_row_logits
    from transvision.models.event_track_v2x.forest_training_data import TrainingShard
    from transvision.models.event_track_v2x.paper_calibration import separate_identity_losses
    from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
    from transvision.models.event_track_v2x.forest_training_checkpoint import CHECKPOINT_KIND
    from transvision.models.event_track_v2x.recoverable_identity import model_digest

    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl')
    world = dist.get_world_size()
    assert world == recipe['world_size'] and world in (4, 8)
    assert torch.cuda.device_count() == world
    seed = recipe['seed']
    selected = json.loads((base / 'selected-checkpoint.json').read_bytes())
    config = FitConfig(**recipe['fit_config'])
    assert config.epochs == selected['selected_epoch']
    assert selected['strict_pipeline_isolated_selection'] is True
    assert selected['selection'] == 'official_train_internal_holdout'
    assert selected['dataset_sha256'] == recipe['manifest']['sha256']
    assert selected['row_protocol']['candidate_protocol'] == 'rbf-all-class-top64-v1'
    assert selected['row_protocol']['class_scope'] == ['car', 'bicycle', 'pedestrian']
    for name, digest in selected['source_sha256'].items():
        assert sha(base / 'source' / name) == digest, name
    data = base / 'rows' / 'rows'
    manifest, sources, total_rows, informative = _rank_zero(
        lambda: audit_dataset(data, recipe['manifest']['sha256'], require_full_train=False))
    assert sha(data / 'manifest.json') == recipe['manifest']['sha256']
    assert manifest['row_protocol'] == selected['row_protocol']
    assert len(manifest['sequences']) == len(manifest['shards']) == 46
    assert manifest['scheduled_frames'] == 7445 and manifest['sealed_source_frames'] == 16338
    provenance = manifest['provenance']
    assert provenance['full_official_train_verified'] is True
    assert provenance['upstream_sequence_isolated'] is True
    assert provenance['official_val_or_test_read'] is False
    assert provenance['upstream_identity_partition'] == selected['partition']
    props = torch.cuda.get_device_properties(rank)
    runtime = dict(rank=rank, world_size=world, device=props.name,
                   uuid=str(props.uuid), memory_bytes=props.total_memory,
                   torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                   host=os.uname().nodename, tf32_matmul=False, tf32_cudnn=False)
    runtimes = _gather(runtime)
    assert len({r['uuid'] for r in runtimes}) == world
    assert len({r['host'] for r in runtimes}) == 1
    output = base / 'training'
    _rank_zero(lambda: output.mkdir())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    architecture = dict(hidden=config.hidden, heads=config.heads, dropout=config.dropout)
    model = RecoverableIdentityModel(**architecture).cuda(rank)
    initial = model_digest(model)
    assert initial == selected['initial_model_sha256'], 'fresh seeded initialization differs'

    class RowLoss(torch.nn.Module):
        def __init__(self, identity):
            super().__init__()
            self.identity = identity

        def forward(self, examples):
            local = rank_rows(examples, dist.get_rank(), world)
            zero = sum(p.sum() * 0 for p in self.identity.parameters())
            if local:
                contexts, targets = zip(*local)
                logits = batched_row_logits(self.identity, contexts,
                    max_nodes=manifest['row_protocol']['parent_limit'] + 1,
                    max_batch=config.batch_size, geometry_weight=config.geometry_weight,
                    process_noise=manifest['row_protocol']['process_noise'])
                result = separate_identity_losses(logits, contexts, targets)
            else:
                result = dict(cross_source=zero, temporal=zero,
                              counts=dict(cross_source=0, temporal=0))
            heads = ('cross_source', 'temporal')
            counts = torch.tensor([result['counts'][k] for k in heads],
                                  device=zero.device, dtype=torch.long)
            dist.all_reduce(counts)
            global_counts = dict(zip(heads, counts.cpu().tolist()))
            loss = zero
            for key in heads:
                loss = loss + result[key] * (world * result['counts'][key] / max(1, global_counts[key]))
            sums = torch.stack([result[k].detach() * result['counts'][k] for k in heads])
            dist.all_reduce(sums)
            reported = {k: float(sums[i] / max(1, global_counts[k])) for i, k in enumerate(heads)}
            return dict(loss=loss, counts=global_counts, heads=reported)

    wrapped = DistributedDataParallel(RowLoss(model), device_ids=[rank])
    # Independent single-process CPU functional loss/gradient for a real train
    # batch; no optimizer or validation/GT-based model selection in this probe.
    first = manifest['shards'][0]
    shard = TrainingShard(data / first['path'], first, manifest['row_protocol']['parent_limit'])
    examples = [shard.example(int(i)) for i in shard.valid_indices[:config.batch_size]]
    model.eval()
    admitted = wrapped(examples)
    admitted['loss'].backward()

    def reference():
        cpu = RecoverableIdentityModel(**architecture)
        cpu.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
        cpu.eval()
        contexts, targets = zip(*examples)
        logits = batched_row_logits(cpu, contexts, max_nodes=9, max_batch=config.batch_size,
                                   geometry_weight=config.geometry_weight, process_noise=.1)
        losses = separate_identity_losses(logits, contexts, targets)
        losses['loss'].backward()
        assert losses['counts'] == admitted['counts'] and sum(losses['counts'].values()) > 0
        for key in ('cross_source', 'temporal'):
            assert np.isclose(float(losses[key].detach()), admitted['heads'][key], atol=1e-4, rtol=1e-4)
        maximum = 0.
        for (name, actual), (ref_name, expected) in zip(model.named_parameters(), cpu.named_parameters()):
            assert name == ref_name and actual.grad is not None and expected.grad is not None
            a, b = actual.grad.detach().cpu().numpy(), expected.grad.detach().numpy()
            assert np.isfinite(a).all() and np.allclose(a, b, atol=1e-4, rtol=1e-4), name
            maximum = max(maximum, float(np.max(np.abs(a - b))))
        receipt = dict(kind='rbf_final_refit_real_batch_DDP_two_head_CPU_gradient_admission_v1',
            rows=len(examples), head_counts=losses['counts'], GPU_heads=admitted['heads'],
            CPU_heads={k: float(losses[k].detach()) for k in ('cross_source', 'temporal')},
            maximum_absolute_gradient_error=maximum, atol=1e-4, rtol=1e-4,
            world_size=world, runtime=runtimes, optimizer_updates=0)
        write_new(output / 'runtime-admission.json', receipt)
        return receipt

    admission = _rank_zero(reference)
    wrapped.zero_grad(set_to_none=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    plan = dict(kind='rbf_nested_selected_epoch_all_class_full_train_refit_candidate_v1',
        recipe=recipe, source_sha256=selected['source_sha256'], runtime=runtimes,
        partition='all_46_official_train_sequences_after_frozen_internal_selection',
        selection='frozen_per_seed_nested_selected_epoch_no_new_validation_search',
        selected_checkpoint_sha256=recipe['checkpoint']['sha256'], full_train=True,
        total_supervised_rows=total_rows, informative_rows=informative,
        objective='cross_source_plus_temporal_set_valued_row_surrogates', exact_joint_nll=False,
        batch_partition='strided_global_batch_no_padding_no_drop', global_batch=config.batch_size,
        labels_in_model_inputs=False, paper_eligible=False)
    _rank_zero(lambda: write_new(output / 'plan.json', plan))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = config.epochs * sum(math.ceil(r['supervised_rows'] / config.batch_size) for r in manifest['shards'])
    done = 0
    started = time.monotonic()
    epochs = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        sums = dict(cross_source=0., temporal=0.)
        counts = dict(cross_source=0, temporal=0)
        for rec in manifest['shards']:
            shard = TrainingShard(data / rec['path'], rec, manifest['row_protocol']['parent_limit'])
            assert len(shard.valid_indices) == rec['supervised_rows']
            for start in range(0, len(shard.valid_indices), config.batch_size):
                examples = [shard.example(int(i)) for i in shard.valid_indices[start:start + config.batch_size]]
                result = wrapped(examples)
                optimizer.zero_grad(set_to_none=True)
                result['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                for key in counts:
                    counts[key] += result['counts'][key]
                    sums[key] += result['heads'][key] * result['counts'][key]
                done += 1
                if rank == 0 and (done % 100 == 0 or done == total_steps):
                    elapsed = time.monotonic() - started
                    print(json.dumps(dict(stage='full_train_final_identity_optimizer_updates',seed=seed,
                        epoch=epoch,completed_steps=done,total_steps=total_steps,
                        ETA_seconds=elapsed * (total_steps - done) / done,
                        ETA_scope='remaining optimizer updates only; checkpoint and independent acceptance excluded')),
                        flush=True)
        epochs.append(dict(epoch=epoch, counts=counts,
            losses={k: sums[k] / max(1, counts[k]) for k in counts}))
    assert done == total_steps
    model.eval().requires_grad_(False)
    final = model_digest(model)
    assert final != initial and len(set(_gather(final))) == 1

    def finish():
        seed_dir = output / f'seed-{seed}'
        seed_dir.mkdir()
        with (seed_dir / 'weights.pt').open('xb') as stream:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, stream)
        checkpoint = dict(selected, model_sha256=final, initial_model_sha256=initial,
            weights=dict(path='weights.pt',sha256=sha(seed_dir/'weights.pt')),
            plan_sha256=sha(output/'plan.json'), partition=dict(fit=manifest['sequences'],holdout=[]),
            kind=CHECKPOINT_KIND, selected_epoch=config.epochs,
            selection='frozen_nested_selected_epoch_full_train_refit',
            selection_checkpoint_sha256=recipe['checkpoint']['sha256'],
            selection_source_task_id=recipe['checkpoint']['task'],
            strict_pipeline_isolated_selection=True, validation_or_test_selection=False,
            runner_sha256=recipe['runner_sha256'], paper_eligible=False)
        write_new(seed_dir/'checkpoint.json', checkpoint)
        write_new(seed_dir/'epochs.json', epochs)
        write_new(output/'receipt.json',dict(kind='rbf_all_class_full_train_final_identity_refit_execution_v1',
            seed=seed,epochs=config.epochs,optimizer_steps=done,sequence_count=46,
            scheduled_frames=7445,model_sha256=final,checkpoint_sha256=sha(seed_dir/'checkpoint.json'),
            runtime_admission=admission,independent_checkpoint_acceptance=False,
            paper_performance_complete=False,elapsed_seconds=time.monotonic()-started))
    _rank_zero(finish)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    args = parser.parse_args()
    run(args.base.resolve(),json.loads(args.recipe.read_bytes()))
