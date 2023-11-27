import math

import numpy as np

from transvision.config import name2id
from transvision.dataset.dataset_utils import load_json
from transvision.v2x_utils import get_3d_8points


def get_label_lidar_rotation(lidar_3d_8_points):
    """3D box in LiDAR coordinate system:

      4 -------- 5
     /|         /|
    7 -------- 6 .
    | |        | |
    . 0 -------- 1
    |/         |/
    3 -------- 2

    x: 3->0
    y: 1->0
    z: 0->4

    Args:
        lidar_3d_8_points: eight point list [[x,y,z],...]
    Returns:
        rotation_z: (-pi,pi) rad
    """
    x0, y0 = lidar_3d_8_points[0][0], lidar_3d_8_points[0][1]
    x3, y3 = lidar_3d_8_points[3][0], lidar_3d_8_points[3][1]
    dx, dy = x0 - x3, y0 - y3
    rotation_z = math.atan2(dy, dx)
    return rotation_z


class Label(dict):

    def __init__(self, path, filt):
        raw_labels = load_json(path)
        # path.replace('label_world', 'label/lidar')

        boxes = []
        class_types = []
        lwhs = []
        for label in raw_labels:
            size = label['3d_dimensions']
            if size['l'] == 0 or size['w'] == 0 or size['h'] == 0:
                continue
            if 'world_8_points' in label:
                box = label['world_8_points']
            else:
                pos = label['3d_location']
                box = get_3d_8points(
                    [float(size['l']), float(size['w']), float(size['h'])],
                    float(label['rotation']),
                    [float(pos['x']), float(pos['y']), float(pos['z']) - float(size['h']) / 2],
                ).tolist()
            lwh = [float(size['l']), float(size['w']), float(size['h'])]
            # determine if box is in extended range
            if filt is None or filt(box):
                boxes.append(box)
                class_types.append(name2id[label['type'].lower()])
                lwhs.append(lwh)
        boxes = np.array(boxes)
        class_types = np.array(class_types)
        # if len(class_types) == 1:
        #     boxes = boxes[np.newaxis, :]
        self.__setitem__('lwhs', lwhs)
        self.__setitem__('boxes_3d', boxes)
        self.__setitem__('labels_3d', class_types)
        self.__setitem__('scores_3d', np.ones_like(class_types))
