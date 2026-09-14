#!/usr/bin/env python3
"""Verify downloaded DDP artifacts against their sealed TRAIN manifest.

This is an offline evidence check, not a new training or tracking evaluation.
ClearML's live status must be checked separately; a file alone cannot prove
the remote task completed. No artifact is published by this command.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.forest_training_data import _new_json


def verify(root, receipt_sha256, data_manifest, data_manifest_sha256, *, seed, require_full_train=True):
    root, data_manifest = Path(root), Path(data_manifest)
    receipt_path, plan_path = contained_file(root,'receipt.json'), contained_file(root,'plan.json')
    if sha_file(receipt_path) != receipt_sha256 or sha_file(data_manifest) != data_manifest_sha256:
        raise ValueError('receipt or original train manifest identity differs')
    receipt, plan, data = [json.loads(p.read_bytes()) for p in (receipt_path,plan_path,data_manifest)]
    if (receipt['kind'] != 'persistent_forest_identity_ddp_receipt_v1' or receipt['status'] != 'complete'
            or receipt['plan_sha256'] != sha_file(plan_path) or receipt['tracking_validation_performed'] is not False
            or receipt['paper_eligible'] is not False or plan['paper_eligible'] is not False
            or receipt['full_official_train'] != require_full_train or plan['full_official_train'] != require_full_train):
        raise ValueError('inconsistent DDP completion or research eligibility contract')
    world = receipt['world_size']
    if type(world) is not int or world < 4 or plan['world_size'] != world:
        raise ValueError('at least four matching ranks required')
    if (plan['dataset_sha256'] != data_manifest_sha256 or data['split'] != 'train'
            or data['row_protocol']['class_scope'] != ['car'] or plan['row_protocol'] != data['row_protocol']
            or plan['dataset_sequences'] != data['sequences'] or plan['upstream_provenance'] != data['provenance']):
        raise ValueError('training dataset or row-protocol binding differs')
    if require_full_train and (len(data['sequences']) != 46 or data['scheduled_frames'] != 7445
            or data['sealed_source_frames'] != 16338 or plan['fit_config']['epochs'] != 10
            or plan['global_batch_size'] != 64):
        raise ValueError('fixed full-train cohort or fitting configuration differs')
    runtimes = receipt['rank_runtime']
    if (runtimes != plan['rank_runtime'] or sorted(r['rank'] for r in runtimes) != list(range(world))
            or any(r['world_size'] != world for r in runtimes) or len({r['host'] for r in runtimes}) != 1):
        raise ValueError('rank runtime identities differ')
    if require_full_train and (len({r['device'] for r in runtimes}) != world
            or len({r.get('gpu_uuid') for r in runtimes}) != world
            or any(r.get('gpu_uuid') in (None,'','unavailable') for r in runtimes)
            or any(r['backend'] != 'nccl' or not r['device'].startswith('cuda:')
                   or not any(f in r.get('gpu_name','') for f in ('A100','5090')) for r in runtimes)):
        raise ValueError('actual distinct A100/5090 CUDA/NCCL ranks required')
    matches = [r for r in receipt['seeds'] if r['seed'] == seed]
    if len(matches) != 1 or [r['seed'] for r in receipt['seeds']] != list(plan['seeds']):
        raise ValueError('unique requested seed and declared seed cohort required')
    if receipt['complete_three_seed_campaign'] != (set(plan['seeds']) == {1337,2027,3407}):
        raise ValueError('three-seed campaign completion claim differs from actual seeds')
    result = matches[0]
    checkpoint_path = contained_file(root,result['checkpoint_manifest'])
    epoch_path = contained_file(checkpoint_path.parent,'epochs.jsonl')
    if sha_file(epoch_path) != result['epochs_sha256']:
        raise ValueError('epoch-log identity differs')
    gate = ForestTrackingConfig(**{k:data['row_protocol'][k] for k in
        ('parent_limit','max_parent_gap_us','gate_distance_m','process_noise')})
    scorer, model = load_identity_checkpoint(checkpoint_path.parent,result['checkpoint_sha256'],config=gate)
    if (model['plan_sha256'] != receipt['plan_sha256'] or model['dataset_sha256'] != data_manifest_sha256
            or model['source_sha256'] != plan['source_sha256'] or model['seed'] != seed
            or model['distributed_world_size'] != world or model['model_sha256'] == model['initial_model_sha256']
            or model['fixed_final_epoch'] != plan['fit_config']['epochs']
            or model['frozen_cache_identity'] != data['frozen_cache_identity']
            or any(m.training for m in scorer.model.modules()) or any(p.requires_grad for p in scorer.model.parameters())):
        raise ValueError('loaded model or fit-plan binding differs')
    entries = [json.loads(line) for line in epoch_path.read_text().splitlines()]
    total = sum(r['supervised_rows'] for r in data['shards'])
    expected_batches = sum(math.ceil(r['supervised_rows']/plan['global_batch_size']) for r in data['shards'])
    if [r['epoch'] for r in entries] != list(range(1,plan['fit_config']['epochs']+1)):
        raise ValueError('missing or repeated epoch')
    for entry in entries:
        ranks = entry['rank_progress']
        if (entry['seed'] != seed or entry['supervised_rows'] != total or entry['global_batches'] != expected_batches
                or entry['validation_metrics_read'] is not False or not math.isfinite(entry['local_training_loss'])
                or sorted(r['rank'] for r in ranks) != list(range(world))
                or sum(r['supervised_rows'] for r in ranks) != total
                or any(r['batches'] != expected_batches or not math.isfinite(r['max_preclip_gradient_norm']) for r in ranks)
                or len({r['model_sha256'] for r in ranks}) != 1
                or (require_full_train and any(not 0 < r['nonempty_batches'] <= r['batches']
                    or not 0 < r['positive_gradient_batches'] <= r['batches'] for r in ranks))):
            raise ValueError('epoch coverage, rank work or synchronized model evidence differs')
    if any(r['model_sha256'] != model['model_sha256'] for r in entries[-1]['rank_progress']):
        raise ValueError('last synchronized rank model differs from saved checkpoint')
    return dict(kind='forest_identity_ddp_offline_artifact_audit_v1', verified=True, seed=seed,
        receipt_sha256=receipt_sha256, checkpoint_sha256=result['checkpoint_sha256'],
        weights_sha256=model['weights']['sha256'], model_sha256=model['model_sha256'],
        data_manifest_sha256=data_manifest_sha256, world_size=world, epochs=len(entries),
        supervised_rows_per_epoch=total, global_batches_per_epoch=expected_batches,
        first_training_loss=entries[0]['local_training_loss'], last_training_loss=entries[-1]['local_training_loss'],
        full_official_train=require_full_train, remote_live_status_checked=False,
        complete_three_seed_campaign=receipt['complete_three_seed_campaign'],
        strict_pipeline_isolated_selection=False, tracking_validation_performed=False, paper_eligible=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--data-manifest',type=Path,required=True)
    p.add_argument('--data-manifest-sha256',required=True)
    p.add_argument('--seed',type=int,choices=(1337,2027,3407),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=verify(a.run,a.receipt_sha256,a.data_manifest,a.data_manifest_sha256,seed=a.seed)
    _new_json(a.output,result)
    print(json.dumps(result,sort_keys=True))
