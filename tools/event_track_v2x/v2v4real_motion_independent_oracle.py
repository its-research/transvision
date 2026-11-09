"""Independent original per-source geometry and scalar past-history oracle.

No producer source-frame/target module is imported. Restricted YAML decoding
is shared solely for numerical data transport. Geometry executes SHA-pinned
original AST definitions; velocity uses scalar fsum rather than producer arrays.
"""
import math
import sys
import types
from collections import Counter

import numpy as np
import torch

from tools.event_track_v2x.v2v4real_gt_oracle import verified_sources, definitions


def original_source_geometry(root):
    trees = verified_sources(root)
    transform = dict(np=np, torch=torch)
    definitions(trees['transformation_utils.py'], ('x_to_world', 'x1_to_x2'), transform, 'original/transformation_utils.py')
    common = dict(np=np, torch=torch)
    definitions(trees['common_utils.py'], ('check_numpy_to_torch', 'rotate_points_along_z'), common, 'original/common_utils.py')
    box = dict(np=np, torch=torch, sys=sys, x1_to_x2=transform['x1_to_x2'], common_utils=types.SimpleNamespace(**common))
    definitions(trees['box_utils.py'], ('corner_to_center', 'boxes_to_corners_3d',
        'get_mask_for_boxes_within_range_torch', 'mask_boxes_outside_range_numpy',
        'create_bbx', 'project_world_objects'), box, 'original/box_utils.py')

    def geometry(metadata, cav):
        objects = metadata['vehicles']
        # Exclude before decoding geometry; original function accesses location
        # before its Pedestrian check, while the declared native vehicle route
        # excludes exact Pedestrian before selected annotation validation.
        selected = {k: v for k, v in objects.items() if v['obj_type'] != 'Pedestrian'}
        output = {}
        box['project_world_objects'](selected, output, np.eye(4), [-100, -40, -5, 100, 40, 3], 'lwh')
        ids, classes, states = [], [], []
        for oid, record in output.items():
            aid = int(record['ass_id'])
            identity = aid if aid != -1 else int(oid) + 100 * int(cav)
            ids.append(identity); classes.append(objects[oid]['obj_type']); states.append(record['coord'][0])
        return np.asarray(states, dtype=np.float64).reshape(-1, 7), ids, classes
    return geometry


def scalar_target_rows(frames):
    history, order = {}, {}
    for frame in frames:
        key = frame['sequence_id'], frame['side']
        ordinal = frame['frame_ordinal']
        if type(ordinal) is not int or ordinal < 0 or ordinal <= order.get(key, -1):
            raise ValueError('noncausal source ordinal')
        order[key] = ordinal
        pose = frame['source_to_world']
        identities = frame['track_ids']
        count = Counter(identities)
        values, valid, predecessors = [], [], []
        for state, identity in zip(frame['states_source'], identities):
            hkey = key + (identity,)
            if count[identity] != 1:
                history.pop(hkey, None)
                values.append([math.nan, math.nan]); valid.append(False); predecessors.append(None)
                continue
            world = [math.fsum([float(pose[a][b]) * float(state[b]) for b in range(3)] + [float(pose[a][3])]) for a in range(3)]
            previous = history.get(hkey)
            if previous is None:
                values.append([math.nan, math.nan]); valid.append(False); predecessors.append(None)
            else:
                dt = (ordinal - previous[0]) / 10.0
                velocity = [(world[a] - previous[1][a]) / dt for a in range(3)]
                values.append([math.fsum(velocity[a] * float(pose[a][b]) for a in range(3)) for b in range(2)])
                valid.append(True); predecessors.append(previous[0])
            history[hkey] = ordinal, world
        yield dict(velocity_source=np.asarray(values, dtype=np.float64).reshape(-1, 2),
                   velocity_valid=np.asarray(valid, dtype=bool), predecessor_ordinals=predecessors)
