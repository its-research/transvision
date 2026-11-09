from tools.event_track_v2x.v2v4real_identity_target_oracle import identities,match_native,supervision
from test_v2v4real_identity_supervision import row


def test_independent_components_preserve_missing_link_and_conflict():
    data=[row(0,0,['1'],['7']),row(1,0,['9'],['7']),row(0,1,['1'],['8']),row(1,1,['10'],['8'])]
    assert all(x[3]=='ambiguous' for x in identities(data).values())
    plain=identities([row(0,0,['101'],['-1']),row(1,0,['1'],['-1'])])
    assert not any(x[2] for x in plain.values())


def test_score_order_first_gt_tie_and_strict_two_metre_gate():
    assert match_native([[0,0],[0,0],[3,0]],[.2,.9,.8],[[0,0],[0,0]])==[1,0,-1]
    assert match_native([[2,0]],[.9],[[0,0]])==[-1]


def test_partial_supervision_never_turns_missing_cross_link_into_negative():
    q=(0,('scene',0,'1'),False,'matched');other=(1,('scene',1,'9'),False,'matched')
    assert supervision(q,[other],set())==([False,False],[False,False],'uncertain_birth')
    assert supervision(q,[],{q})==([False],[False],'out_of_support')
    assert supervision(q,[q],{q})==([False,True],[True,True],'parent')
