"""Bind fit-only feature export to canonical fold-specific detector evidence.

This validates provenance metadata only. It does not accept checkpoint tensors,
execute inference, or grant formal dataset/paper qualification.
"""
import math
import re
import hashlib
import json

from run_cooptrack_official_oof_gpu4_filehost_v4 import validate_cohort


def need(value, message):
    if not value:
        raise ValueError(message)


def validate_binding(package, inputs, byte_freeze, launch, startup, completion, *,
                     side, seed, checkpoint_sha256, config_sha256,
                     package_manifest_sha256, input_manifest_sha256,
                     evidence_payloads):
    validate_cohort(package)
    for digest in (checkpoint_sha256, config_sha256, package_manifest_sha256,
                   input_manifest_sha256):
        need(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest),
             'invalid bound SHA-256')
    need(side in ('vehicle-side', 'infrastructure-side')
         and type(seed) is int and seed in (1337, 2027, 3407), 'side/seed mismatch')
    fold = package['fold_id']
    fit, held = package['fit_sequence_ids'], package['held_out_sequence_ids']
    need(inputs.get('kind') == 'spd_canonical_oof_fit_feature_inputs_v1'
         and inputs.get('cohort') == 'canonical-oof-fit-feature-inputs'
         and inputs.get('fold_id') == fold
         and inputs.get('fit_sequence_ids') == fit
         and inputs.get('excluded_held_out_sequence_ids') == held
         and inputs.get('training_package_manifest_sha256') == package_manifest_sha256
         and inputs.get('held_out_selection_scoring_eligible') is False
         and inputs.get('predictions_generated') is False
         and inputs.get('paper_eligible') is False
         and inputs.get('source_all_train_input_admission_sha256') == '29214bb602a4637637a479636109e6bec6591282d7e2647c2d628bc8ebd967df'
         and inputs.get('official_split_sha256') == package['official_split_sha256']
         and inputs.get('canonical_fivefold_manifest_sha256')
         == package['canonical_fivefold_manifest_sha256']
         and all(inputs.get(k) is False for k in (
             'gt_payloads_in_package', 'gt_payloads_read', 'val_payloads_read',
             'test_payloads_read')), 'input is not exact label-free fit complement')
    need(byte_freeze.get('kind') == 'spd_canonical_oof_detector_independent_byte_freeze_v1'
         and byte_freeze.get('byte_freeze_accepted') is True
         and byte_freeze.get('fold_id') == fold and byte_freeze.get('seed') == seed
         and byte_freeze.get('package_manifest_sha256') == package_manifest_sha256,
         'training lacks matching completed-task byte freeze')
    artifacts = byte_freeze['artifacts']
    for suffix, parsed in (('-launch-receipt.json', launch),
                           ('-optimizer-startup.json', startup),
                           ('-completion', completion)):
        raw = evidence_payloads[side + suffix]
        record = artifacts[side + suffix]
        need(type(raw) is bytes and len(raw) == record['bytes']
             and hashlib.sha256(raw).hexdigest() == record['sha256']
             and json.loads(raw) == parsed, 'parsed training evidence differs from frozen bytes')
    need(artifacts[side + '-final-checkpoint']['sha256'] == checkpoint_sha256
         and artifacts[side + '-detector.py']['sha256'] == config_sha256,
         'checkpoint/config differs from byte freeze')
    need(launch.get('kind') == 'cooptrack_detector_training_launch_v1'
         and launch.get('cohort') == 'fit-fold'
         and launch.get('fit_sequence_ids') == fit
         and launch.get('fold_id') == fold and launch.get('seed') == seed
         and launch.get('side') == side and launch.get('epochs') == 24
         and launch.get('batch_probe_only') is False
         and launch.get('official_val_test_loaded') is False
         and launch.get('raw_labels_modified') is False
         and launch.get('paper_ranking_eligible') is False
         and launch.get('pretrained_kind') == 'ImageNet-R50-only-no-SPD-trained-weights',
         'launch is not corresponding canonical fold fit')
    pretrained = next(r for r in package['inventory'] if r['path'] == 'resnet50-0676ba61.pth')
    need(launch.get('pretrained_sha256') == pretrained['sha256'], 'pretrained identity mismatch')
    for key in ('loss', 'backbone_max_abs_update'):
        value = startup.get(key)
        need(type(value) in (int, float) and math.isfinite(value), 'nonfinite optimizer evidence')
    need(startup.get('kind') == 'detector_optimizer_startup_v1'
         and startup.get('batch_probe_only') is False
         and startup.get('micro_iteration') == 16
         and startup.get('resolved_config_sha256') == config_sha256
         and startup['backbone_max_abs_update'] > 0
         and startup.get('optimizer_state_entries', 0) > 0, 'optimizer startup mismatch')
    total = launch.get('max_micro_iterations')
    need(type(total) is int and total > 16
         and completion.get('side') == side
         and completion.get('cohort') == 'official-oof-fold-fit'
         and completion.get('train_sequences') == len(fit)
         and completion.get('epochs') == 24
         and completion.get('fold_id') == fold and completion.get('seed') == seed
         and completion.get('sha256') == checkpoint_sha256
         and completion.get('checkpoint') == 'iter_%d.pth' % total
         and completion.get('stage') == 'D2-detector-training-only'
         and completion.get('official_val_result_available') is False,
         'final checkpoint is not the completed canonical fold fit')
    need(byte_freeze['sides'][side]['expected_iterations'] == total,
         'byte-freeze iteration identity mismatch')
    for key in ('batch_per_gpu', 'effective_batch_size'):
        need(launch.get(key) == startup.get(key) == completion.get(key),
             'resource identity mismatch')
    batch = launch.get('batch_per_gpu')
    need(type(batch) is int and batch in (2, 4, 8, 10) and batch * 4 <= len(fit)
         and launch.get('effective_batch_size') == batch * 4
         and launch.get('world_size') == startup.get('world_size') == 4,
         'world-size/batch identity mismatch')
    return {'kind': 'spd-canonical-oof-fit-feature-export-binding-v1',
            'task_id': byte_freeze['task_id'], 'fold_id': fold, 'seed': seed, 'side': side,
            'fit_sequence_ids': fit, 'held_out_sequence_ids': held,
            'package_manifest_sha256': package_manifest_sha256,
            'input_manifest_sha256': input_manifest_sha256,
            'checkpoint_sha256': checkpoint_sha256, 'resolved_config_sha256': config_sha256,
            'epochs': 24, 'metadata_binding_accepted': True,
            'checkpoint_tensors_or_forward_accepted': False,
            'fit_feature_inference_complete': False,
            'held_out_selection_scoring_eligible': False,
            'export_sequence_ids': fit, 'formal_paper_eligible': False}
