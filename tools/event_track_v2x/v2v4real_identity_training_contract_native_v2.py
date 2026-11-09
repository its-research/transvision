"""Native v2 admission adds positive cross/temporal targets in BOTH train roles.

Preserves every geometry, row support, feature, partition and fit recipe check
from the original contract. A legacy zero-cross-positive proof is rejected.
"""
from copy import deepcopy
from tools.event_track_v2x.v2v4real_identity_training_contract import (
    FIT_CONFIG,HEADS,macro_score,select_epoch,local_global_head_loss,
    admitted_roles as original_roles,
)


def admitted_roles(manifest,proof,partition,seed,manifest_sha256):
    if (proof.get('kind')!='v2v4real_identity_rows_independent_native_target_v2_admission'
            or proof.get('all_prediction_context_fields_byte_identical_to_admitted_rows') is not True
            or proof.get('target_and_byte_checks_exact') is not True
            or len(proof.get('unchanged_prediction_context_fields',[]))!=15
            or manifest['provenance'].get('native_target_contract')!='author_native_mapped_id_one_explicit_association_v2'
            or manifest['annotation_audit'].get('cross_source_label_basis')!='author_native_mapped_id_equality_at_paired_frame_at_least_one_explicit_association_v2'):
        raise ValueError('independent author-native v2 targets and exact feature/support reuse required')
    # The strengthened v2 proof supplies all of the original admission fields;
    # delegate their checks without pretending the stored proof has old targets.
    original=deepcopy(proof)
    original['kind']='v2v4real_identity_rows_independent_feature_support_target_admission_v1'
    roles=original_roles(manifest,original,partition,seed,manifest_sha256)
    results={r['sequence_id']:r for r in proof['results']}
    for field in ('cross_source_positive_parent_edges','temporal_positive_parent_edges'):
        if proof.get(field,0)<=0 or proof[field]!=sum(r[field] for r in proof['results']):
            raise ValueError('complete nonzero independently counted '+field+' required')
        for role,records in roles.items():
            if any(r.get(field)!=results[r['sequence_id']][field] for r in records) or sum(r[field] for r in records)<=0:
                raise ValueError('both original fit/holdout roles need real '+field)
    return roles
