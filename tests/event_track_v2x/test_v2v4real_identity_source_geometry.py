import numpy as np
import pytest
from tools.event_track_v2x.v2v4real_identity_source_geometry import source_boxes


def metadata(extent):
    return dict(vehicles={'1':dict(location=[0,0,0],center=[0,0,0],angle=[0,0,0],extent=extent,obj_type='Truck',ass_id=7)})


def test_native_signed_width_is_measured_from_unchanged_corners():
    a,counts,signed=source_boxes(metadata([2,-1,.5]),'0')
    np.testing.assert_allclose(a[0]['box_lwh'],[0,0,0,4,2,1,0])
    assert signed==[dict(object_id='1',original_signed_extent=[2,-1,.5])]
    assert a[0]['track_id']==7 and counts=={'Truck':1}


def test_zero_extent_is_not_repaired_or_silently_dropped():
    with pytest.raises(ValueError,match='degenerate'):
        source_boxes(metadata([0,1,.5]),'0')
