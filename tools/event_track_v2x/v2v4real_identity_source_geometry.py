"""Native signed-extent geometry for full-train offline identity supervision.

Some release annotations contain negative extents. Author geometry multiplies
the signed values into ordered corners, then measures upright edge lengths.
Preserve that operation exactly; do not take absolute values before projection.
The earlier strict source/motion parser remains unchanged.
"""
from collections import Counter
import numpy as np
from transvision.models.event_track_v2x.v2v4real_inputs import _vector,pose_to_world
from transvision.models.event_track_v2x.v2v4real_ground_truth import CORNER_SIGNS,GT_RANGE,native_id,upright_lwh,upright_corners
from transvision.models.event_track_v2x.paper_evaluation_policy import native_vehicle


def source_boxes(metadata,cav):
    records=[];counts=Counter();signed=[]
    for oid,item in metadata['vehicles'].items():
        if not isinstance(item,dict) or not isinstance(item.get('obj_type'),str) or not item['obj_type']:
            raise ValueError('explicit native annotation class required')
        counts[item['obj_type']]+=1
        if not native_vehicle(item['obj_type']):continue
        identity=native_id(oid,item.get('ass_id',oid),cav)
        location=_vector(item.get('location'),3,'location');center=_vector(item.get('center'),3,'center')
        angles=_vector(item.get('angle'),3,'angle');extent=_vector(item.get('extent'),3,'extent')
        if np.any(extent<0):signed.append(dict(object_id=str(oid),original_signed_extent=extent.tolist()))
        pose=pose_to_world((*list(location+center),*list(angles)))
        corners=(CORNER_SIGNS*extent)@pose[:3,:3].T+pose[:3,3]
        box=upright_lwh(corners)
        mask=upright_corners([box])[0]
        if ((mask>=GT_RANGE[:3])&(mask<=GT_RANGE[3:])).all(axis=1).sum()>=2:
            records.append(dict(object_id=str(oid),associated_id=str(item.get('ass_id',oid)),
                track_id=identity,raw_class=item['obj_type'],box_lwh=box))
    return records,dict(counts),signed
