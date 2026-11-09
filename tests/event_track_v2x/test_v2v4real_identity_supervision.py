import math
import numpy as np
from tools.event_track_v2x.v2v4real_identity_supervision import source_supervision
from transvision.models.event_track_v2x.forest_supervision import relation


def row(source,ordinal,oids,aids,mapped=None):
    return dict(sequence_id='scene',side='vehicle-side' if source==0 else 'infrastructure-side',cav_id=str(source),
        frame_key=str(ordinal),frame_ordinal=ordinal,object_ids=oids,association_ids=aids,
        native_mapped_ids=mapped if mapped is not None else [int(a) if a!='-1' else int(o)+100*source for o,a in zip(oids,aids)],
        states_source_lwh=[[0,0,0,4,2,1,0] for _ in oids])


def targets(index):return list(index.targets.values())


def test_fallback_namespace_collision_does_not_create_cross_link():
    _,index,_=source_supervision([row(0,0,['101'],['-1']),row(1,0,['1'],['-1'])])
    a,b=targets(index);assert relation(a,b) is None
    assert index.audit['exact_native_cross_source_links']==0


def test_real_temporal_source_id_and_explicit_cross_link_with_box_conversion():
    frames,index,schedule=source_supervision([row(0,0,['1'],['7']),row(1,0,['9'],['7']),row(0,1,['1'],['7']),row(1,1,['9'],['7'])])
    t=targets(index);assert all(relation(t[0],x) is True for x in t)
    np.testing.assert_array_equal(frames['scene','vehicle-side','0'].boxes[0,3:6],[2,4,1])
    assert frames['scene','vehicle-side','0'].boxes[0,6]==-math.pi/2
    assert schedule[1]['box_reference_timestamp_us']==100000


def test_changed_association_invalidates_joined_component():
    _,index,_=source_supervision([row(0,0,['1'],['7']),row(1,0,['9'],['7']),row(0,1,['1'],['8']),row(1,1,['10'],['8'])])
    assert all(x.status=='ambiguous' for x in targets(index))


def test_duplicate_native_alias_is_unknown_and_empty_pair_preserved():
    frames,index,schedule=source_supervision([row(0,0,['1','2'],['7','-1'],[7,7]),row(1,0,[],[]),row(0,1,[],[]),row(1,1,[],[])])
    assert all(x.status=='ambiguous' for x in targets(index))
    assert len(schedule)==2 and frames['scene','vehicle-side','1'].boxes.shape==(0,7)
