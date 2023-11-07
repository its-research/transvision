from typing import Union

import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.utils import array_converter
from torch import Tensor

# from mmdet3d.visualization import Det3DLocalVisualizer
from tools.visual.local_visualizer import Det3DLocalVisualizer2


@array_converter(apply_to=('val', ))
def limit_period(val: Union[np.ndarray, Tensor], offset: float = 0.5, period: float = np.pi) -> Union[np.ndarray, Tensor]:
    """Limit the value into a period for periodic function.

    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val


points = np.fromfile('./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/velodyne/015372.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer2(save_dir='./')
# set point cloud in visualizer
visualizer.set_points(points)
# (x, y, z, x_size, y_size, z_size, yaw, ...)
# boxes = item['box3d_lidar'].copy()
# # swap l, w (or dx, dy)
# item['box3d_lidar'][3] = boxes[4]
# item['box3d_lidar'][4] = boxes[3]
# # change yaw
# item['box3d_lidar'][6] = -boxes[6] - np.pi / 2
# item['box3d_lidar'][6] = limit_period(item['box3d_lidar'][6], period=np.pi * 2)
yaw = -1.143976 - np.pi / 2
yaw = limit_period(yaw, period=np.pi * 2)
# hwl l, w, h
# bboxes_3d = LiDARInstance3DBoxes(torch.tensor([[27.3455, -25.56276, -0.5044948, 1.882665, 1.776011,  4.521148,-1.143976]]))
bboxes_3d = LiDARInstance3DBoxes(torch.tensor([[27.3455, -25.56276, -0.5044948, 4.521148, 1.776011, 1.882665, -1.143976]]))

# "3d_dimensions": {
#     "h": 2.701104,
#     "w": 2.918645,
#     "l": 10.638364
# },
# "3d_location": {
#     "x": 47.89272,
#     "y": -24.70498,
#     "z": 0.6321486
# },
# "rotation": 1.994932
bboxes_3d = LiDARInstance3DBoxes(torch.tensor([[47.89272, -24.70498, 0.6321486, 10.638364, 2.918645, 2.701104, 1.994932]]))
# Draw 3D bboxes
visualizer.draw_bboxes_3d(bboxes_3d)
visualizer.show()
