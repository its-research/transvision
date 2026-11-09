"""Offline native per-source training geometry; never the merged ego union.

Caller separately admits train raw rows and their pose/clock/PCD lineage. This
adapter only checks the supplied byte/pose bindings and parses source geometry.
Duplicate mapped IDs remain explicit so the target builder breaks their history.
"""
import hashlib

import numpy as np

from transvision.models.event_track_v2x.v2v4real_inputs import load_raw_yaml
from transvision.models.event_track_v2x.v2v4real_ground_truth import _local_boxes, matrix_pose


def source_train_frame(manifest, label_bytes, binding):
    if manifest.get('split') != 'train' or manifest.get('official_test_read') is not False:
        raise ValueError('train-only source geometry required before label parsing')
    if type(label_bytes) is not bytes or hashlib.sha256(label_bytes).hexdigest() != binding['label_sha256']:
        raise ValueError('admitted source label SHA differs')
    cav = binding['cav_id']
    side = binding['side']
    if cav not in ('0', '1') or side not in ('vehicle-side', 'infrastructure-side'):
        raise ValueError('explicit native CAV and cache side required')
    ordinal = binding['frame_ordinal']
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError('nonnegative nominal frame ordinal required')
    metadata = load_raw_yaml(label_bytes)
    pose = matrix_pose(metadata['lidar_pose'])
    raw_pose = np.asarray(binding['source_to_world'], dtype=np.float64)
    if not np.array_equal(pose, raw_pose):
        raise ValueError('native label and admitted raw source pose differ')
    boxes, counts = _local_boxes(metadata, cav, vehicle=True)
    return dict(sequence_id=binding['sequence_id'], side=side, cav_id=cav,
                frame_key=binding['frame_key'], frame_ordinal=ordinal,
                source_to_world=pose, label_sha256=binding['label_sha256'],
                states_source=np.asarray([b['box_lwh'] for b in boxes], dtype=np.float64).reshape(-1, 7),
                track_ids=[b['track_id'] for b in boxes],
                raw_classes=[b['raw_class'] for b in boxes],
                object_ids=[b['object_id'] for b in boxes],
                native_annotation_counts=counts,
                source_roi='native_at_least_two_upright_corners_inside_xyz',
                duplicate_identity_policy='preserve_for_motion_history_break',
                clock='nominal_10hz')
