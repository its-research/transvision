"""Offline test GT coverage must match the original source schedule and egos."""
import pytest
from tools.event_track_v2x.prepare_v2v4real_official_test_vehicle_gt import coverage


def original():
    return [dict(sequence_id='scene',frame_key=str(i),frame_ordinal=i,is_ego=ego,cav_id='1' if ego else '0')
            for i in range(3) for ego in (True,False)]


def frames():
    return [dict(sequence_id='scene',frame_key=str(i),frame_ordinal=i,ego_cav='1',objects=[]) for i in range(3)]


def test_empty_GT_frames_retained_and_original_ego_is_used():
    coverage(frames(),original())


@pytest.mark.parametrize('mutation',['missing','duplicate','wrong-key','wrong-ego','wrong-ordinal'])
def test_no_renaming_filtering_or_ego_substitution(mutation):
    rows=frames()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(dict(rows[0]))
    elif mutation=='wrong-key':rows[0]['frame_key']='other'
    elif mutation=='wrong-ego':rows[0]['ego_cav']='0'
    else:rows[0]['frame_ordinal']=9
    with pytest.raises(ValueError):coverage(rows,original())


def test_duplicate_original_ego_rows_are_not_silently_collapsed():
    rows=original();rows.append(dict(rows[0]))
    with pytest.raises(ValueError):coverage(frames(),rows)


def test_native_Pedestrian_exclusion_precedes_dimension_validation_without_mutation():
    import copy
    import numpy as np
    from test_v2v4real_ground_truth import pair,obj
    from transvision.models.event_track_v2x.v2v4real_ground_truth import prepare_vehicle_frame,prepare_frame,native_vehicle_annotations
    from tools.event_track_v2x.v2v4real_gt_oracle import native_oracle
    metadata=pair();bad=obj(kind='Pedestrian');bad['extent']=[-.3847775757864492,-.5025336885882745,.9331549033522606]
    metadata['0']['vehicles'][57]=bad;before=copy.deepcopy(metadata)
    actual=prepare_vehicle_frame(metadata,ego_cav='0')
    annotations,counts=native_vehicle_annotations(metadata['0'])
    assert counts['Pedestrian']==1 and all(a.raw_class!='Pedestrian' for a in annotations)
    assert metadata['0']['vehicles']==before['0']['vehicles']
    with native_oracle('/private/tmp/rbf-vehicle-gt-oracle-pinned-20261005',vehicle=True) as reference:
        expected=reference(metadata,ego_cav='0')
    assert {r['track_id'] for r in actual['objects']}==set(expected)
    for row in actual['objects']:np.testing.assert_allclose(row['corners_ego'],expected[row['track_id']],atol=1e-4,rtol=0)
    with pytest.raises(ValueError):prepare_frame(metadata,ego_cav='0')
    metadata['0']['vehicles'][57]['obj_type']='Car'
    with pytest.raises(ValueError):prepare_vehicle_frame(metadata,ego_cav='0')


@pytest.mark.parametrize('first_x',[10.,200.])
def test_same_CAV_duplicate_identity_matches_native_first_retained_occurrence(first_x):
    import numpy as np
    from test_v2v4real_ground_truth import pair,obj
    from transvision.models.event_track_v2x.v2v4real_ground_truth import prepare_vehicle_frame,prepare_frame
    from tools.event_track_v2x.v2v4real_gt_oracle import native_oracle
    metadata=pair()
    metadata['0']['vehicles']={1:obj(first_x,aid=17),3:obj(20.,aid=17,kind='Truck')}
    actual=prepare_vehicle_frame(metadata,ego_cav='0')
    with native_oracle('/private/tmp/rbf-vehicle-gt-oracle-pinned-20261005',vehicle=True) as reference:
        expected=reference(metadata,ego_cav='0')
    assert {r['track_id'] for r in actual['objects']}==set(expected)
    for row in actual['objects']:np.testing.assert_allclose(row['corners_ego'],expected[row['track_id']],atol=1e-4,rtol=0)
    if first_x==10.:
        assert any(d['kept_cav']==d['suppressed_cav']=='0' for d in actual['audit']['duplicate_ids'])
        metadata['0']['vehicles'][3]['obj_type']='Car'
        with pytest.raises(ValueError):prepare_frame(metadata,ego_cav='0')
