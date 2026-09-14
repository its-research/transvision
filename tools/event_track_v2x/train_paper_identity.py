#!/usr/bin/env python3
"""Sequence-isolated, chronological row-surrogate training on official train
only."""
import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.event_track_v2x.train_forest_identity import FitConfig, audit_dataset  # noqa: E402
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file  # noqa: E402
from transvision.models.event_track_v2x.forest_training import batched_row_logits  # noqa: E402
from transvision.models.event_track_v2x.forest_training_checkpoint import CHECKPOINT_KIND  # noqa: E402
from transvision.models.event_track_v2x.forest_training_data import TrainingShard, _new_json  # noqa: E402
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel  # noqa: E402
from transvision.models.event_track_v2x.paper_calibration import separate_identity_losses  # noqa: E402
from transvision.models.event_track_v2x.paper_protocol import PAPER, SEEDS, PaperProtocol, training_partition  # noqa: E402
from transvision.models.event_track_v2x.recoverable_identity import model_digest  # noqa: E402


def fit(data, expected_sha256, output, *, protocol, groups, config=None, seeds=SEEDS, device='cpu', fixture=False):
    protocol.require_train()
    if protocol.candidates != PAPER or type(fixture) is not bool:
        raise ValueError('explicit main candidate protocol/evidence status required')
    config = config or FitConfig()
    import torch.distributed as dist

    from transvision.models.event_track_v2x.paper_ddp import PaperDDP
    parallel = PaperDDP(device, fixture=fixture) if dist.is_initialized() else None
    write = parallel.write if parallel else lambda operation: operation()
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError('declared unique training seeds required')
    data, output = Path(data), Path(output).absolute()
    manifest, sources, _, _ = audit_dataset(data, expected_sha256, require_full_train=False)
    if manifest['row_protocol'].get('candidate_protocol') != PAPER:
        raise ValueError('legacy candidate shards cannot train paper model')
    if set(groups) != set(manifest['sequences']) or any(not isinstance(g, str) or not g for g in groups.values()):
        raise ValueError('explicit sequence-to-recording mapping required')
    provenance = manifest['provenance']
    if not fixture and (provenance.get('dataset') != protocol.dataset or provenance.get('full_official_train_verified') is not True
                        or provenance.get('upstream_sequence_isolated') is not True):
        raise ValueError('missing full train and sequence-isolated upstream detector/calibration provenance')
    partition = training_partition(groups.values())
    records = {key: [r for r in manifest['shards'] if groups[r['sequence_id']] in ids] for key, ids in partition.items()}
    sources = dict(sources, **{str(Path(__file__).relative_to(ROOT)): sha_file(__file__)})
    for name in ('paper_calibration.py', 'paper_protocol.py', 'paper_ddp.py'):
        sources['transvision/models/event_track_v2x/' + name] = sha_file(ROOT / 'transvision/models/event_track_v2x' / name)
    sources['tools/event_track_v2x/train_forest_identity_ddp.py'] = sha_file(ROOT / 'tools/event_track_v2x/train_forest_identity_ddp.py')

    def create_output():
        if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError('new training output required')
        output.mkdir()

    write(create_output)
    write(lambda: _new_json(
        output / 'plan.json',
        dict(
            kind='rbf_sequence_identity_fit_v1',
            protocol=asdict(protocol),
            groups=groups,
            partition=partition,
            seeds=seeds,
            config=asdict(config),
            fixture=fixture,
            dataset_sha256=expected_sha256,
            source_sha256=sources,
            selection='minimum_train_holdout_macro_row_surrogate',
            sequence_order='causal_row_order_within_sequence',
            objective='cross_source_plus_temporal_row_surrogates',
            exact_joint_nll=False,
            torch_version=torch.__version__,
            device=str(device),
            world_size=parallel.world_size if parallel else 1,
            gradient_reduction='DDP_global_denominator_per_head_no_padding_no_drop' if parallel else 'single_process')))
    results = []
    try:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            architecture = dict(hidden=config.hidden, heads=config.heads, dropout=config.dropout)
            model = RecoverableIdentityModel(**architecture).to(device)
            initial = model_digest(model)
            distributed_loss = parallel.bind(model, config, manifest['row_protocol']) if parallel else None
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            best, selected, epoch_rows = float('inf'), None, []
            directory = output / f'seed-{seed}'
            write(lambda: directory.mkdir())
            for epoch in range(1, config.epochs + 1):
                stats = {}
                for phase in ('fit', 'holdout'):
                    model.train(phase == 'fit')
                    sequence_losses, counts = [], {'cross_source': 0, 'temporal': 0}
                    for record in records[phase]:
                        shard = TrainingShard(contained_file(data, record['path']), record, manifest['row_protocol']['parent_limit'])
                        summed, number = 0., 0
                        for start in range(0, len(shard.valid_indices), config.batch_size):
                            examples = [shard.example(int(i)) for i in shard.valid_indices[start:start + config.batch_size]]
                            contexts, targets = zip(*examples)
                            with torch.set_grad_enabled(phase == 'fit'):
                                if phase == 'fit' and parallel:
                                    loss = distributed_loss(examples)
                                else:
                                    logits = batched_row_logits(
                                        model,
                                        contexts,
                                        max_nodes=manifest['row_protocol']['parent_limit'] + 1,
                                        max_batch=config.batch_size,
                                        geometry_weight=config.geometry_weight,
                                        process_noise=manifest['row_protocol']['process_noise'])
                                    loss = separate_identity_losses(logits, contexts, targets)
                                if phase == 'fit':
                                    optimizer.zero_grad(set_to_none=True)
                                    loss['loss'].backward()
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                                    optimizer.step()
                            value = parallel.reported_loss(loss['loss']) if parallel and phase == 'fit' else float(loss['loss'].detach())
                            summed += value * len(examples)
                            number += len(examples)
                            for key in counts:
                                counts[key] += loss['counts'][key]
                        if not number:
                            raise ValueError('empty sequence supervision')
                        sequence_losses.append(summed / number)
                    if not sequence_losses or not sum(counts.values()):
                        raise ValueError('partition lacks informative supervision')
                    stats[phase] = dict(macro_row_surrogate=float(np.mean(sequence_losses)), informative_rows=counts)
                epoch_rows.append(dict(epoch=epoch, **stats))
                score = stats['holdout']['macro_row_surrogate']
                if score < best:
                    best, selected = score, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.cpu().load_state_dict(best_state)
            model.eval().requires_grad_(False)
            if model_digest(model) == initial:
                raise ValueError('training made no parameter update')

            def save_weights():
                with (directory / 'weights.pt').open('xb') as stream:
                    torch.save(model.state_dict(), stream)
                _new_json(directory / 'epochs.json', epoch_rows)

            write(save_weights)
            checkpoint = dict(
                kind=CHECKPOINT_KIND,
                architecture=architecture,
                seed=seed,
                row_protocol=manifest['row_protocol'],
                data_split='train',
                dataset=protocol.dataset,
                dataset_sha256=expected_sha256,
                model_sha256=model_digest(model),
                initial_model_sha256=initial,
                weights=dict(path='weights.pt', sha256=sha_file(directory / 'weights.pt')),
                geometry_weight=config.geometry_weight,
                local_partial_label_objective=True,
                labels_in_model_inputs=False,
                frozen_cache_identity=manifest['frozen_cache_identity'],
                source_sha256=sources,
                plan_sha256=sha_file(output / 'plan.json'),
                selected_epoch=selected,
                partition=partition,
                fixture=fixture,
                paper_eligible=False,
                calibration_guarantee=False,
                selection='official_train_internal_holdout',
                strict_pipeline_isolated_selection=not fixture,
                validation_or_test_selection=False,
                exact_joint_nll=False)
            write(lambda: _new_json(directory / 'checkpoint.json', checkpoint))
            results.append(dict(seed=seed, path=directory.name, sha256=sha_file(directory / 'checkpoint.json')))
        if sha_file(data / 'manifest.json') != expected_sha256 or any(sha_file(ROOT / p) != s for p, s in sources.items()):
            raise ValueError('training input/source changed')
        receipt = dict(
            status='software_training_completed',
            fixture=fixture,
            seeds=results,
            three_seed_training_complete=set(seeds) == set(SEEDS),
            paper_results_verified=False,
            plan_sha256=sha_file(output / 'plan.json'))
        write(lambda: _new_json(output / 'receipt.json', receipt))
        return receipt
    except BaseException as error:
        if not parallel or dist.get_rank() == 0:
            _new_json(output / 'failure.json', dict(status='failed', error=str(error), completed_seeds=results))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest-sha256', 'output', 'dataset', 'groups'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--config')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, action='append')
    p.add_argument('--fixture', action='store_true')
    a = p.parse_args()
    import torch.distributed as dist
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        from datetime import timedelta
        if a.device.startswith('cuda'):
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            a.device = f'cuda:{local_rank}'
        dist.init_process_group('nccl' if a.device.startswith('cuda') else 'gloo', timeout=timedelta(seconds=60))
    result = fit(
        a.data,
        a.manifest_sha256,
        a.output,
        protocol=PaperProtocol(a.dataset, 'train'),
        groups=json.loads(Path(a.groups).read_text()),
        config=FitConfig(**json.loads(Path(a.config).read_text())) if a.config else None,
        seeds=a.seed or SEEDS,
        device=a.device,
        fixture=a.fixture)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(json.dumps(result, sort_keys=True))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
