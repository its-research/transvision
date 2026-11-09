import math
import pytest
import torch
from tools.event_track_v2x.v2v4real_identity_training_contract import HEADS,FIT_CONFIG,admitted_roles,macro_score,select_epoch,local_global_head_loss


def test_head_macro_does_not_turn_absent_supervision_into_zero_loss():
    data=[dict(sequence_id='a',cross_source=dict(loss_sum=9.,samples=3),temporal=dict(loss_sum=0.,samples=0)),
        dict(sequence_id='b',cross_source=dict(loss_sum=0.,samples=0),temporal=dict(loss_sum=8.,samples=4))]
    assert macro_score(data)['selection_score']==5.
    assert macro_score(data)['informative_sequences']==dict(cross_source=1,temporal=1)
    with pytest.raises(ValueError,match='temporal'):macro_score(data[:1])


def test_earliest_frozen_train_holdout_epoch_wins_exact_tie():
    data=[dict(epoch=i,holdout=dict(selection_score=x)) for i,x in enumerate([3.,1.,1.],1)]
    assert select_epoch(data)==2
    with pytest.raises(ValueError,match='ordered'):select_epoch([data[-1]])
    with pytest.raises(ValueError,match='finite'):select_epoch([dict(epoch=1,holdout=dict(selection_score=math.nan))])


def test_unequal_head_denominators_and_empty_rank_match_full_batch_gradient():
    parameter=torch.tensor(.6,dtype=torch.float64,requires_grad=True)
    # Four ranks, two differently populated heads, including a completely empty
    # rank. Compare against the direct full-population objective and derivative.
    cross=[(parameter-i).square() for i in [0.,1.,2.]]
    temporal=[(parameter+i).square() for i in [1.,3.]]
    zero=parameter*0
    populations=[dict(cross_source=cross[:2],temporal=[]),dict(cross_source=cross[2:],temporal=temporal[:1]),
        dict(cross_source=[],temporal=temporal[1:]),dict(cross_source=[],temporal=[])]
    losses=[]
    for population in populations:
        means={h:torch.stack(population[h]).mean() if population[h] else zero for h in HEADS}
        losses.append(local_global_head_loss(means,{h:len(population[h]) for h in HEADS},dict(cross_source=3,temporal=2),4))
    distributed=sum(losses)/4;direct=torch.stack(cross).mean()+torch.stack(temporal).mean()
    torch.testing.assert_close(distributed,direct,atol=1e-15,rtol=0)
    g1=torch.autograd.grad(distributed,parameter,retain_graph=True)[0];g2=torch.autograd.grad(direct,parameter)[0]
    torch.testing.assert_close(g1,g2,atol=1e-15,rtol=0)


@pytest.mark.parametrize('changes',[{'seed':2027},{'sequences':31},{'official_test_labels_read':True},
    {'all_causal_parent_contexts_independently_verified':False}])
def test_unqualified_or_wrong_seed_rows_never_reach_native_optimizer(changes):
    proof=dict(kind='v2v4real_identity_rows_independent_feature_support_target_admission_v1',seed=1337,sequences=32,
        official_test_labels_read=False,all_original_cache_nodes_and_prediction_only_features_verified=True,
        all_causal_parent_contexts_independently_verified=True,all_native_temporal_cross_source_partial_targets_independently_verified=True)
    proof.update(changes)
    with pytest.raises(ValueError,match='seed-bound'):admitted_roles({},proof,{},1337,'0'*64)


def test_fixed_recipe_is_the_declared_ten_epoch_joint_model():
    assert FIT_CONFIG==dict(epochs=10,batch_size=64,hidden=128,heads=4,dropout=.1,learning_rate=.0003,
        weight_decay=.0001,gradient_clip=5.,geometry_weight=1.)
