# Coordinate system refactoring(v1.0.0rc0)

In this version, we did a major code refactoring which improved the consistency among the three coordinate systems (and corresponding box representation), LiDAR, Camera, and Depth. A brief summary for this refactoring is as follows:

- The three coordinate systems are all right-handed now (which means the yaw angle increases in the counterclockwise direction).
- The LiDAR system `(x_size, y_size, z_size)` corresponds to `(l, w, h)` instead of `(w, l, h)`. This is more natural since `l` is parallel with the direction where the yaw angle is zero, and we prefer using the positive direction of the `x` axis as that direction, which is exactly how we define yaw angle in Depth and Camera coordinate systems.
- The APIs for box-related operations are improved and now are more user-friendly.

## ***NOTICE!!***

Since definitions of box representation have changed, the annotation data of most datasets require updating:

- SUN RGB-D: Yaw angles in the annotation should be reversed.
- KITTI: For LiDAR boxes in GT databases, (x_size, y_size, z_size, yaw) out of (x, y, z, x_size, y_size, z_size) should be converted from the old LiDAR coordinate system to the new one. The training/validation data annotations should be left unchanged since they are under the Camera coordinate system, which is unmodified after the refactoring.
- Waymo: Same as KITTI.
- nuScenes: For LiDAR boxes in training/validation data and GT databases, (x_size, y_size, z_size, yaw) out of (x, y, z, x_size, y_size, z_size) should be converted.
- Lyft: Same as nuScenes.

Please regenerate the data annotation/GT database files or use [`update_data_coords.py`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc0/tools/update_data_coords.py) to update the data.

To use boxes under Depth and LiDAR coordinate systems, or to convert boxes between different coordinate systems, users should be aware of the difference between the old and new definitions. For example, the rotation, flipping, and bev functions of [`DepthInstance3DBoxes`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc0/mmdet3d/core/bbox/structures/depth_box3d.py) and [`LiDARInstance3DBoxes`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc0/mdet3d/core/bbox/structures/lidar_box3d.py) and box conversion [functions](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc0/mmdet3d/core/bbox/structures/box_3d_mode.py) have all been reimplemented in the refactoring.

Consequently, functions like [`output_to_lyft_box`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc0/mmdet3d/datasets/lyft_dataset.py) undergo small modification to adapt to the new LiDAR/Depth box.

Since the LiDAR system `(x_size, y_size, z_size)` now corresponds to `(l, w, h)` instead of `(w, l, h)`, the anchor sizes for LiDAR boxes are also changed, e.g., from `[1.6, 3.9, 1.56]` to `[3.9, 1.6, 1.56]`.

Functions only involving points are generally unaffected except if they rely on some refactored utility functions such as `rotation_3d_in_axis`.
