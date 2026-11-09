"""Full native row admission and fixed train-only joint identity recipe.

The original name-group partition is authoritative. Its upstream isolation and
physical-session limitations remain disclosed, without fixture substitution.
"""
import math

HEADS=('cross_source','temporal')
FIT_CONFIG=dict(epochs=10,batch_size=64,hidden=128,heads=4,dropout=.1,learning_rate=.0003,
    weight_decay=.0001,gradient_clip=5.,geometry_weight=1.)


def admitted_roles(manifest,proof,partition,seed,manifest_sha256):
    if (seed not in (1337,2027,3407) or proof.get('kind')!='v2v4real_identity_rows_independent_feature_support_target_admission_v1'
            or proof.get('seed')!=seed or proof.get('sequences')!=32 or proof.get('official_test_labels_read') is not False
            or any(proof.get(k) is not True for k in ['all_original_cache_nodes_and_prediction_only_features_verified',
                'all_causal_parent_contexts_independently_verified','all_native_temporal_cross_source_partial_targets_independently_verified'])):
        raise ValueError('complete seed-bound independent native row admission required')
    if (manifest.get('kind')!='persistent_forest_train_rows_v1' or manifest.get('split')!='train'
            or manifest.get('labels_in_model_inputs') is not False or manifest.get('gt_boxes_or_ids_in_shards') is not False
            or manifest.get('scheduled_frames')!=7105 or manifest.get('sealed_source_frames')!=14210
            or manifest.get('ingested_source_frames')!=14210 or manifest.get('source_unavailable')!=0):
        raise ValueError('full 32-sequence native train row cohort required')
    p=manifest['provenance'];protocol=manifest['row_protocol']
    if (p.get('dataset')!='v2v4real' or p.get('seed')!=seed or p.get('full_official_train_verified') is not True
            or p.get('native_class_scope')!=['vehicle'] or p.get('network_class_slot_mapping')!={'car':'vehicle'}
            or p.get('official_test_labels_read') is not False or p.get('nominal_rate_hz')!=10
            or protocol.get('candidate_protocol')!='rbf-all-class-top64-v1' or protocol.get('parent_limit')!=8
            or protocol.get('feature_recipe')!='recover-before-fuse-physical-world-cache203-v1'
            or protocol.get('max_parent_gap_us')!=2000000 or protocol.get('gate_distance_m')!=8.
            or protocol.get('process_noise')!=.1 or protocol.get('minimum_raw_score')!=.05 or protocol.get('maximum_detections')!=64
            or manifest['frozen_cache_identity']['feature_method']!='rbf-pointpillar-bev128-bilinear-l2-v1'):
        raise ValueError('dataset-specific native feature and temporal contract differs')
    if (partition.get('official_split')!='train' or partition.get('official_test_used_for_partition') is not False
            or partition.get('sequence_to_name_group')!=p['sequence_to_name_group']):
        raise ValueError('original frozen official-train name-group partition required')
    sequences=manifest['sequences'];records=manifest['shards']
    if (len(sequences)!=32 or sequences!=sorted(set(sequences)) or [r['sequence_id'] for r in records]!=sequences
            or len({r['path'] for r in records})!=32
            or set(sequences)!=set(partition['sequence_to_name_group']) or sum(r['nodes'] for r in records)!=proof['nodes']
            or not any(k.endswith('/manifest.json') and h==manifest_sha256 for k,h in proof['input_sha256'].items())):
        raise ValueError('original manifest or complete independently verified row support differs')
    if ([r['sequence_id'] for r in proof['results']]!=sequences
            or [(r['nodes'],r['reason_counts']) for r in proof['results']]!=[(r['nodes'],r['reason_counts']) for r in records]):
        raise ValueError('complete per-sequence independent row counts differ')
    fit_groups=set(partition['identity_fit']);holdout_groups=set(partition['identity_selection'])
    if not fit_groups or not holdout_groups or fit_groups&holdout_groups:raise ValueError('disjoint original identity fit/selection groups required')
    roles={}
    for s,g in partition['sequence_to_name_group'].items():
        role='identity_fit' if g in fit_groups else 'identity_selection' if g in holdout_groups else None
        if role is None:raise ValueError('native sequence lacks original identity role')
        roles[s]=role
    if roles!=p['sequence_identity_roles']:raise ValueError('producer changed original frozen identity roles')
    return {name:[r for r in records if roles[r['sequence_id']]==role] for name,role in [('fit','identity_fit'),('holdout','identity_selection')]}


def macro_score(sequences):
    """Sum of two head-wise sequence macro means, without fake zero losses."""
    result={};counts={}
    for head in HEADS:
        available=[r[head]['loss_sum']/r[head]['samples'] for r in sequences if r[head]['samples']>0]
        if not available or not all(math.isfinite(v) for v in available):
            raise ValueError('phase lacks finite informative '+head+' supervision')
        result[head]=math.fsum(available)/len(available);counts[head]=len(available)
    return dict(head_macro_means=result,informative_sequences=counts,selection_score=math.fsum(result.values()),
        sequences_without_any_informative_head=[r['sequence_id'] for r in sequences if not any(r[h]['samples'] for h in HEADS)])


def select_epoch(history):
    if not history or [r['epoch'] for r in history]!=list(range(1,len(history)+1)):
        raise ValueError('complete ordered selection epoch history required')
    if any(not math.isfinite(r['holdout']['selection_score']) for r in history):raise ValueError('finite original train holdout scores required')
    return min(history,key=lambda r:(r['holdout']['selection_score'],r['epoch']))['epoch']


def local_global_head_loss(head_means,local_counts,global_counts,world_size):
    """DDP average yields exactly a global mean per named head, including tails."""
    if (type(world_size) is not int or world_size<1 or set(head_means)!=set(HEADS)
            or set(local_counts)!=set(HEADS) or set(global_counts)!=set(HEADS)
            or any(type(local_counts[h]) is not int or type(global_counts[h]) is not int
                or not 0<=local_counts[h]<=global_counts[h] for h in HEADS)):
        raise ValueError('valid local/global informative head counts required')
    zero=sum(v*0 for v in head_means.values());loss=zero
    for head in HEADS:
        if global_counts[head]>0:
            loss=loss+head_means[head]*(world_size*local_counts[head]/global_counts[head])
    return loss
