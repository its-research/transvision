"""Canonical SPD calibration membership gate, before fitting or sealing V2.

This gate proves metadata isolation only. It does not establish raw-cache
acceptance, fitted numerical calibration, or DetectionCacheV2 completion.
The historical generic cache builder must not bypass this gate for OOF use.
"""
from run_cooptrack_official_oof_gpu4_filehost_v4 import validate_cohort


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_input_boundary(package, fit_inputs, held_inputs):
    validate_cohort(package)
    fold = package['fold_id']
    fit, held = package['fit_sequence_ids'], package['held_out_sequence_ids']
    for inputs in (fit_inputs, held_inputs):
        require(type(inputs.get('fold_id')) is int and inputs['fold_id'] == fold,
                'calibration input fold differs')
        for key in ('official_split_sha256', 'canonical_fivefold_manifest_sha256'):
            require(inputs.get(key) == package[key], 'calibration input partition differs')
        require(all(inputs.get(key) is False for key in (
            'gt_payloads_in_package', 'gt_payloads_read', 'val_payloads_read',
            'test_payloads_read')), 'calibration prediction inputs contain labels or val/test')
    require(fit_inputs.get('kind') == 'spd_canonical_oof_fit_feature_inputs_v1'
            and fit_inputs.get('cohort') == 'canonical-oof-fit-feature-inputs'
            and fit_inputs.get('fit_sequence_ids') == fit
            and fit_inputs.get('excluded_held_out_sequence_ids') == held
            and fit_inputs.get('held_out_selection_scoring_eligible') is False,
            'calibration feature input is not exact fit complement')
    require(held_inputs.get('kind') == 'eventtrack_train_image_pose_inputs_v1'
            and held_inputs.get('cohort') == 'canonical-oof-held-out'
            and held_inputs.get('train_sequences') == held
            and held_inputs.get('excluded_fit_sequence_ids') == fit,
            'calibration application input is not exact held-out fold')
    require(not set(fit) & set(held), 'calibration fit and held-out overlap')
    return {'fold_id': fold, 'fit_sequence_ids': fit, 'held_out_sequence_ids': held,
            'metadata_isolation_verified': True,
            'numerical_calibration_verified': False, 'formal_v2_ready': False}


def validate_calibration_boundary(package, fit_inputs, held_inputs, calibration):
    """Reject all-train, historical nested, or held-out fitted calibration.

    The frozen calibration must record the exact canonical partition binding;
    an old generic train label or a disjoint subset does not prove this contract.
    Callers must separately verify source bytes, per-side checkpoints, actual
    fitted examples, parameters and independent numerical readback.
    """
    boundary = validate_input_boundary(package, fit_inputs, held_inputs)
    require(calibration.get('kind') == 'eventtrack_train_calibration_v1',
            'unsupported calibration schema')
    require(calibration.get('fit_sequences') == boundary['fit_sequence_ids'],
            'calibration must fit exactly the corresponding canonical complement')
    evidence = calibration.get('evidence', {})
    require(all(evidence.get(key) is False for key in (
        'official_validation_used_for_selection', 'test_payloads_read')),
        'calibration used official val or test')
    binding = calibration.get('canonical_oof_binding', {})
    require(type(binding.get('fold_id')) is int
            and binding['fold_id'] == package['fold_id']
            and binding.get('held_out_sequence_ids') == boundary['held_out_sequence_ids']
            and all(binding.get(key) == package[key] for key in (
                'official_split_sha256', 'canonical_fivefold_manifest_sha256')),
            'calibration lacks matching canonical OOF partition binding')
    return boundary
