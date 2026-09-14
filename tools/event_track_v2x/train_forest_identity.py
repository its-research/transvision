#!/usr/bin/env python3
"""Three-seed train-only row identity fitting; no validation selection/publication.

Current detector/calibration caches are in-sample, not OOF. This fixed-final-epoch
development run is not the paper's strict sequence-isolated model-selection run.
All hyperparameters and source hashes are saved before optimizer updates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_supervision import set_valued_parent_loss
from transvision.models.event_track_v2x.forest_training import batched_row_logits
from transvision.models.event_track_v2x.forest_training_checkpoint import CHECKPOINT_KIND
from transvision.models.event_track_v2x.forest_training_data import DATA_KIND, TrainingShard, _new_json
from transvision.models.event_track_v2x.forest_training_data import row_protocol
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from transvision.models.event_track_v2x.recoverable_identity import model_digest


SEEDS = (1337, 2027, 3407)


@dataclass(frozen=True)
class FitConfig:
    epochs: int = 10
    batch_size: int = 64
    hidden: int = 128
    heads: int = 4
    dropout: float = .1
    learning_rate: float = .0003
    weight_decay: float = .0001
    gradient_clip: float = 5.
    geometry_weight: float = 1.

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in (self.epochs, self.batch_size, self.hidden, self.heads))
                or self.hidden % self.heads or self.batch_size > 256
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1
                or any(not math.isfinite(v) or v < 0 for v in
                       (self.learning_rate, self.weight_decay, self.gradient_clip, self.geometry_weight))
                or self.learning_rate == 0 or self.gradient_clip == 0):
            raise ValueError('invalid fixed-epoch fitting configuration')


def training_sources():
    module = ROOT/'transvision/models/event_track_v2x'
    paths = [Path(__file__), ROOT/'tools/event_track_v2x/prepare_forest_training.py']+[
        module/name for name in ('forest_training.py', 'forest_training_data.py', 'forest_training_checkpoint.py',
        'forest_supervision.py', 'forest_row_context.py', 'forest_potentials.py', 'forest_tracking.py',
        'learned_identity.py', 'recoverable_identity.py', 'identity_forest.py', 'detection_cache_v2.py', 'forest_cache_stream.py',
        'prediction_features.py', 'tracking_v2.py', 'recoverable_states.py', 'fusion.py', 'arrays.py')]
    return {p.relative_to(ROOT).as_posix(): sha_file(p) for p in paths}


def audit_dataset(data, manifest_sha256, *, require_full_train=True):
    """Shared, read-only full-data preflight for single-process and DDP fits."""
    data = Path(data)
    if type(require_full_train) is not bool:
        raise TypeError('explicit full-train requirement required')
    path = contained_file(data, 'manifest.json')
    if sha_file(path) != manifest_sha256:
        raise ValueError('training manifest identity differs')
    manifest = json.loads(path.read_bytes())
    if (manifest['kind'] != DATA_KIND or manifest['split'] != 'train'
            or manifest['labels_in_model_inputs'] is not False or manifest['row_protocol']['class_scope'] != ['car']
            or not manifest['shards']
            or [s['sequence_id'] for s in manifest['shards']] != manifest['sequences']
            or manifest['sequences'] != sorted(set(manifest['sequences']))
            or len({s['path'] for s in manifest['shards']}) != len(manifest['shards'])):
        raise ValueError('invalid train-only identity dataset contract')
    if require_full_train and (len(manifest['sequences']) != 46 or manifest['scheduled_frames'] != 7445
            or manifest['sealed_source_frames'] != 16338
            or manifest['provenance'].get('full_official_train_verified') is not True):
        raise ValueError('full official train artifacts required; fixture is not a training run')
    protocol = manifest['row_protocol']
    gate_config = ForestTrackingConfig(**{k: protocol[k] for k in
        ('parent_limit', 'max_parent_gap_us', 'gate_distance_m', 'process_noise')})
    if protocol != row_protocol(gate_config):
        raise ValueError('unsupported training feature, selection or context recipe')
    sources = training_sources()
    if require_full_train:
        producer = manifest['provenance'].get('producer_source_sha256')
        if not isinstance(producer, dict) or not producer or any(sha != sources.get(name) for name, sha in producer.items()):
            raise ValueError('training preparation sources differ; rebuild rather than silently change contexts')
    records = manifest['shards']
    # Audit every shard/example before the first optimizer update. At most one
    # sequence is resident; all unusable rows remain in the audit, not deleted.
    informative, total_rows = 0, 0
    for record in records:
        shard = TrainingShard(contained_file(data, record['path']), record, protocol['parent_limit'])
        for row in range(record['nodes']):
            _, target = shard.example(row)
            informative += int(any(target.positives) and sum(target.known) > sum(target.positives))
        total_rows += record['supervised_rows']
    if not total_rows or not informative:
        raise ValueError('no informative supervised decisions; cannot claim identity training')
    del shard
    if sha_file(path) != manifest_sha256 or training_sources() != sources:
        raise ValueError('training inputs or sources changed during input audit')
    return manifest, sources, total_rows, informative


def fit_dataset(data, manifest_sha256, output, *, config=None, device='cpu', require_full_train=True):
    config, data, output = config or FitConfig(), Path(data), Path(output).absolute()
    if type(config) is not FitConfig or type(require_full_train) is not bool:
        raise TypeError('validated training configuration required')
    manifest, sources, total_rows, informative = audit_dataset(data, manifest_sha256,
                                                            require_full_train=require_full_train)
    protocol, records, path = manifest['row_protocol'], manifest['shards'], contained_file(data, 'manifest.json')
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new training output without symlink traversal required')
    output.mkdir()
    if training_sources() != sources:
        raise ValueError('training sources changed during input audit')
    plan = dict(kind='persistent_forest_identity_fit_plan_v1', dataset_sha256=manifest_sha256,
        source_sha256=sources, seeds=SEEDS, fit_config=asdict(config), device=str(device), row_protocol=protocol,
        supervised_rows=total_rows, informative_rows=informative, dataset_sequences=manifest['sequences'],
        full_official_train=require_full_train, local_fixture_only=not require_full_train,
        selection='fixed_final_epoch_no_validation_search', strict_pipeline_isolated_selection=False,
        upstream_provenance=manifest['provenance'], paper_eligible=False,
        torch_version=torch.__version__, numpy_version=np.__version__,
        determinism='seeded_CPU_fixture_verified_GPU_bitwise_reproducibility_not_claimed')
    _new_json(output/'plan.json', plan)
    started, results = time.monotonic(), []
    try:
        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if str(device).startswith('cuda'):
                torch.cuda.manual_seed_all(seed)
            generator = np.random.default_rng(seed)
            architecture = dict(hidden=config.hidden, heads=config.heads, dropout=config.dropout)
            model = RecoverableIdentityModel(**architecture).to(device=device)
            initial_sha = model_digest(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            directory = output/f'seed-{seed}'
            directory.mkdir()
            with (directory/'epochs.jsonl').open('xb') as log:
                for epoch in range(config.epochs):
                    epoch_start, count, summed_loss, batches, max_norm = time.monotonic(), 0, 0., 0, 0.
                    model.train()
                    for index in generator.permutation(len(records)):
                        record = records[int(index)]
                        shard = TrainingShard(contained_file(data, record['path']), record, protocol['parent_limit'])
                        indices = generator.permutation(shard.valid_indices)
                        for start in range(0, len(indices), config.batch_size):
                            examples = [shard.example(int(i)) for i in indices[start:start+config.batch_size]]
                            contexts, targets = zip(*examples)
                            optimizer.zero_grad(set_to_none=True)
                            logits = batched_row_logits(model, contexts, max_nodes=protocol['parent_limit']+1,
                                max_batch=config.batch_size, geometry_weight=config.geometry_weight, process_noise=protocol['process_noise'])
                            loss = set_valued_parent_loss(logits, targets)['loss']
                            loss.backward()
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                            optimizer.step()
                            count += len(examples); batches += 1
                            summed_loss += loss.item()*len(examples)
                            max_norm = max(max_norm, float(norm))
                        del shard
                    if count != total_rows:
                        raise ValueError('epoch omitted or repeated supervised rows')
                    entry = dict(seed=seed, epoch=epoch+1, supervised_rows=count, batches=batches,
                        local_training_loss=summed_loss/count, max_preclip_gradient_norm=max_norm,
                        elapsed_seconds=time.monotonic()-epoch_start, validation_metrics_read=False)
                    log.write((json.dumps(entry, sort_keys=True, allow_nan=False)+'\n').encode()); log.flush()
                    print(json.dumps(entry, sort_keys=True), flush=True)
            model.cpu().eval().requires_grad_(False)
            final_sha = model_digest(model)
            if final_sha == initial_sha:
                raise ValueError('optimizer did not change identity weights')
            with (directory/'weights.pt').open('xb') as stream:
                torch.save(model.state_dict(), stream)
            checkpoint = dict(kind=CHECKPOINT_KIND, architecture=architecture, seed=seed, row_protocol=protocol,
                data_split='train', dataset_sha256=manifest_sha256, model_sha256=final_sha, initial_model_sha256=initial_sha,
                weights=dict(path='weights.pt', sha256=sha_file(directory/'weights.pt')),
                geometry_weight=config.geometry_weight, local_partial_label_objective=True, labels_in_model_inputs=False,
                frozen_cache_identity=manifest['frozen_cache_identity'],
                source_sha256=sources, plan_sha256=sha_file(output/'plan.json'), full_official_train=require_full_train,
                paper_eligible=False, calibration_guarantee=False, validation_or_test_selection=False,
                strict_pipeline_isolated_selection=False, fixed_final_epoch=config.epochs)
            _new_json(directory/'checkpoint.json', checkpoint)
            results.append(dict(seed=seed, checkpoint_manifest=directory.name+'/checkpoint.json',
                checkpoint_sha256=sha_file(directory/'checkpoint.json'), epochs_sha256=sha_file(directory/'epochs.jsonl')))
        if sha_file(path) != manifest_sha256 or training_sources() != sources:
            raise ValueError('training inputs or sources changed during fitting')
        receipt = dict(kind='persistent_forest_identity_fit_receipt_v1', status='complete', seeds=results,
            full_official_train=require_full_train, local_fixture_only=not require_full_train,
            plan_sha256=sha_file(output/'plan.json'), elapsed_seconds=time.monotonic()-started,
            paper_eligible=False, tracking_validation_performed=False)
        _new_json(output/'receipt.json', receipt)
        return receipt
    except BaseException as error:
        _new_json(output/'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error),
            completed_seeds=len(results), partial_outputs_not_final_results=True, paper_eligible=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(fit_dataset(args.data, args.manifest_sha256, args.output,
        config=FitConfig(epochs=args.epochs, batch_size=args.batch_size), device=args.device), sort_keys=True))


if __name__ == '__main__':
    main()
