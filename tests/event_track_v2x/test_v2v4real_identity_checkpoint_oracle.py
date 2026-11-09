import copy
import pytest
from tools.event_track_v2x.verify_v2v4real_identity_checkpoint import phase_score,contained


def phase():
    records=[dict(sequence_id='a',supervised_rows=65),dict(sequence_id='b',supervised_rows=1)]
    rows=[dict(sequence_id='a',supervised_rows=65,cross_source=dict(samples=2,loss_sum=6.),temporal=dict(samples=0,loss_sum=0.)),
        dict(sequence_id='b',supervised_rows=1,cross_source=dict(samples=0,loss_sum=0.),temporal=dict(samples=1,loss_sum=2.))]
    p=dict(sequences=rows,head_macro_means=dict(cross_source=3.,temporal=2.),informative_sequences=dict(cross_source=1,temporal=1),
        selection_score=5.,supervised_rows=66,global_batches=3,optimizer_steps=2,maximum_preclip_gradient_norm=2.,
        rank_coverage=[dict(rank=i,local_rows=n,batches=3,model_sha256='a'*64) for i,n in enumerate([18,16,16,16])])
    return p,records


def test_macro_oracle_reconstructs_missing_head_and_sequence_tail_counts():
    p,records=phase()
    assert phase_score(p,records,4,training=True)==(5.,'a'*64)


@pytest.mark.parametrize('corruption',['macro','missing_rows','extra_batch','unequal_model','wrong_tail','bad_loss','no_head'])
def test_independent_selection_oracle_rejects_corrupted_history(corruption):
    p,records=phase();p=copy.deepcopy(p)
    if corruption=='macro':p['selection_score']=2.5
    elif corruption=='missing_rows':p['sequences'][0]['supervised_rows']=64
    elif corruption=='extra_batch':p['global_batches']=4
    elif corruption=='unequal_model':p['rank_coverage'][3]['model_sha256']='b'*64
    elif corruption=='wrong_tail':
        p['rank_coverage'][0]['local_rows']=17;p['rank_coverage'][1]['local_rows']=17
    elif corruption=='bad_loss':p['sequences'][0]['cross_source']['loss_sum']=float('nan')
    else:p['sequences'][1]['temporal']=dict(samples=0,loss_sum=0.)
    with pytest.raises(ValueError):phase_score(p,records,4,training=True)


def test_independent_payload_read_rejects_absolute_traversal_and_symlink(tmp_path):
    p=tmp_path/'data';p.write_text('ordinary')
    link=tmp_path/'link';link.symlink_to(p)
    assert contained(tmp_path,'data')==p
    for bad in ['../data',str(p),'link']:
        with pytest.raises(ValueError):contained(tmp_path,bad)
