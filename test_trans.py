import os

import numpy as np
import torch
from mmdet3d.structures import CameraInstance3DBoxes
from mmengine import load

from tools.dataset_converters.ffnet_preprocess import get_label_lidar_rotation, trans_point_w2l
from tools.dataset_converters.gen_kitti.label_lidarcoord_to_cameracoord import convert_point, get_camera_3d_8points, get_lidar2cam
from tools.dataset_converters.label_world2v import get_rotation, get_world_8_points, trans_point_world2v, write_world_8_points


def convert(box, src, dst, rt_mat=None):
    arr = torch.from_numpy(np.asarray(box)).clone()

    # convert box from `src` mode to `dst` mode.
    x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
    if src == 'lidar' and dst == 'cam':
        if rt_mat is None:
            rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        xyz_size = torch.cat([y_size, z_size, x_size], dim=-1)
    elif src == 'cam' and dst == 'lidar':
        if rt_mat is None:
            rt_mat = arr.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        xyz_size = torch.cat([z_size, x_size, y_size], dim=-1)

    if not isinstance(rt_mat, torch.Tensor):
        rt_mat = arr.new_tensor(rt_mat)
    if rt_mat.size(1) == 4:
        extended_xyz = torch.cat([arr[:3], arr.new_ones(1)], dim=-1)
        xyz = extended_xyz @ rt_mat.t()
    else:
        xyz = arr[:3] @ rt_mat.t()

    remains = arr[..., 6:]
    arr = torch.cat([xyz[:3], xyz_size, remains], dim=-1)

    # convert arr to the original type
    return arr.numpy()


# label_world = {
#     "alpha": 1.755473172162272,
#     "3d_dimensions": {"h": 1.59085, "w": 1.855412, "l": 4.288774},
#     "3d_location": {
#         "x": 3954.7612483013086,
#         "y": 2402.414303699318,
#         "z": 20.041707738008963,
#     },
#     "rotation": 3.114823,
#     "world_8_points": [
#         [3956.385905256698, 2404.0967987862227, 19.2533960847273],
#         [3957.1030444331914, 2402.3856064320607, 19.262668234546823],
#         [3953.1476883717287, 2400.7278394009254, 19.239213047358756],
#         [3952.430549195236, 2402.4390317550874, 19.229940897539233],
#         [3956.3748082308875, 2404.1007679977106, 20.844202428659166],
#         [3957.0919474073808, 2402.3895756435486, 20.853474578478693],
#         [3953.1365913459185, 2400.7318086124133, 20.830019391290627],
#         [3952.4194521694253, 2402.4430009665757, 20.8207472414711],
#     ],
# }
label_world = {
    '2d_box': {
        'xmin': 75.893105,
        'ymin': 601.654663,
        'xmax': 529.742012,
        'ymax': 748.949097,
    },
    '3d_dimensions': {
        'h': 1.537562,
        'w': 1.764941,
        'l': 4.534194
    },
    '3d_location': {
        'x': 2651.310234532717,
        'y': 1735.613395854671,
        'z': 20.515609803372765,
    },
    'rotation':
    0.04081635,
    'world_8_points': [
        [2653.3120116458444, 1734.228135196805, 19.751771756688676],
        [2651.845055912768, 1733.2471318248156, 19.725503548057258],
        [2649.324357287641, 1737.0160467217788, 19.74206641441238],
        [2650.7913130207176, 1737.9970500937677, 19.768334623043803],
        [2653.2961117777936, 1734.210744987563, 21.28915319233315],
        [2651.8291560447174, 1733.2297416155736, 21.262884983701728],
        [2649.30845741959, 1736.9986565125369, 21.279447850056854],
        [2650.7754131526667, 1737.979659884526, 21.305716058688276],
    ],
}
data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure'

path_novatel2world = os.path.join(data_root, 'vehicle-side/calib/novatel_to_world', '000010.json')
path_lidar2novatel = os.path.join(data_root, 'vehicle-side/calib/lidar_to_novatel', '000010.json')

# World to Vehicle
world_8_points_old = label_world['world_8_points']
world_8_points = []
for point in world_8_points_old:
    point_new = trans_point_w2l(point, path_novatel2world, path_lidar2novatel)
    world_8_points.append(point_new)
lidar_3d_data_veh_ff = {}
lidar_3d_data_veh_ff['3d_dimensions'] = label_world['3d_dimensions']
lidar_3d_data_veh_ff['3d_location'] = {}
lidar_3d_data_veh_ff['3d_location']['x'] = (world_8_points[0][0] + world_8_points[2][0]) / 2
lidar_3d_data_veh_ff['3d_location']['y'] = (world_8_points[0][1] + world_8_points[2][1]) / 2
lidar_3d_data_veh_ff['3d_location']['z'] = (world_8_points[0][2] + world_8_points[4][2]) / 2
lidar_3d_data_veh_ff['rotation'] = get_label_lidar_rotation(world_8_points)
lidar_3d_data_veh_ff['world_8_points'] = world_8_points
# World to Vehicle end
print('###ffnet convert###')
# print("origin lidar:", lidar_3d_data_veh_ff)

x, y, z = (
    lidar_3d_data_veh_ff['3d_location']['x'],
    lidar_3d_data_veh_ff['3d_location']['y'],
    lidar_3d_data_veh_ff['3d_location']['z'],
)
l, w, h = (
    lidar_3d_data_veh_ff['3d_dimensions']['l'],
    lidar_3d_data_veh_ff['3d_dimensions']['w'],
    lidar_3d_data_veh_ff['3d_dimensions']['h'],
)
z = z - h / 2

lidar_old = np.array([x, y, z, w, l, h, -lidar_3d_data_veh_ff['rotation']])

calib_v_lidar2cam_filename = os.path.join(data_root, 'vehicle-side/calib/lidar_to_camera', '000010.json')
calib_v_lidar2cam = load(calib_v_lidar2cam_filename)
calib_v_cam_intrinsic_filename = os.path.join(data_root, 'vehicle-side/calib/camera_intrinsic', '000010.json')
calib_v_cam_intrinsic = load(calib_v_cam_intrinsic_filename)
rect = np.identity(4)
Trv2c = np.identity(4)
Trv2c[0:3, 0:3] = calib_v_lidar2cam['rotation']
Trv2c[0:3, 3] = [
    calib_v_lidar2cam['translation'][0][0],
    calib_v_lidar2cam['translation'][1][0],
    calib_v_lidar2cam['translation'][2][0],
]
P2 = np.identity(4)
P2[0:3, 0:3] = np.array(calib_v_cam_intrinsic['cam_K']).reshape(3, 3)

extended_xyz = np.array([x, y, z, 1])
cam_location = extended_xyz @ Trv2c.T
cam_location = cam_location[:3]
cam_dimension = [l, h, w]
cam_rotation = -lidar_3d_data_veh_ff['rotation']

camera_3d_data_ff = {}
camera_3d_data_ff['3d_location'] = np.array(cam_location)
camera_3d_data_ff['3d_dimensions'] = np.array(cam_dimension)
camera_3d_data_ff['rotation'] = np.array([cam_rotation])

camera_old = np.array([cam_location[0], cam_location[1], cam_location[2], l, h, w, cam_rotation])
# print("kitti camera:", camera_3d_data_ff)
gt_bboxes_3d_cam = np.concatenate(
    [
        camera_3d_data_ff['3d_location'],
        camera_3d_data_ff['3d_dimensions'],
        camera_3d_data_ff['rotation'],
    ],
    axis=0,
).astype(np.float32)

gt_bboxes_3d_lidar_ff = convert(gt_bboxes_3d_cam, 'cam', 'lidar', np.linalg.inv(rect @ Trv2c))
print(lidar_old.tolist())
print(camera_old.tolist())
print(gt_bboxes_3d_lidar_ff.tolist())

gt_bboxes_3d = CameraInstance3DBoxes(np.array([camera_old])).convert_to(0, np.linalg.inv(rect @ Trv2c))
print(gt_bboxes_3d[0].numpy().tolist())

# DAIR official code
print('###dair convert###')

my_3d_point = np.array([
    label_world['3d_location']['x'],
    label_world['3d_location']['y'],
    label_world['3d_location']['z'],
]).reshape(3, 1)
new_3d_point = trans_point_world2v(my_3d_point, path_novatel2world, path_lidar2novatel)

world_8_points = get_world_8_points(label_world)
my_world_8_points = []
for j in range(8):
    point = world_8_points[j]
    point = trans_point_world2v(point, path_novatel2world, path_lidar2novatel)
    my_world_8_points.append(point)
new_world_8_points = write_world_8_points(my_world_8_points)

l = label_world['3d_dimensions']['l']
w = label_world['3d_dimensions']['w']
h = label_world['3d_dimensions']['h']
rotation = get_rotation(new_world_8_points, new_3d_point, l, w)

x = new_3d_point[0]
y = new_3d_point[1]
z = new_3d_point[2]
print(
    lidar_3d_data_veh_ff['3d_location']['x'],
    lidar_3d_data_veh_ff['3d_location']['y'],
    lidar_3d_data_veh_ff['3d_location']['z'],
    lidar_3d_data_veh_ff['3d_dimensions']['w'],
    lidar_3d_data_veh_ff['3d_dimensions']['h'],
    lidar_3d_data_veh_ff['3d_dimensions']['l'],
    lidar_3d_data_veh_ff['rotation'],
)
print(x, y, z, w, h, l, rotation)
# print(lidar_3d_data_veh_ff["world_8_points"])
# print(my_world_8_points)
print('#### origin lidar end')

calib_lidar2cam = load(os.path.join(data_root, 'vehicle-side/calib/lidar_to_camera', '004105.json'))
r_velo2cam, t_velo2cam = get_lidar2cam(calib_lidar2cam)
Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))

z = z - h / 2
bottom_center = [x, y, z]
obj_size = [l, w, h]
bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam

alpha, yaw = get_camera_3d_8points(obj_size, rotation, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam)
cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)
print(
    camera_3d_data_ff['3d_location'],
    camera_3d_data_ff['3d_dimensions'],  # lhw
    camera_3d_data_ff['rotation'],
)
print(cam_x, cam_y, cam_z, l, h, w, yaw)
print('###kitti camera end')

gt_bboxes_3d = np.concatenate([np.array([cam_x, cam_y, cam_z]), np.array([l, h, w]), np.array([yaw])], axis=0).astype(np.float32)
gt_bboxes_3d = np.array([gt_bboxes_3d])
gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(0, np.linalg.inv(rect @ Trv2c))
gt_bboxes_3d_17 = convert(
    np.concatenate([np.array([cam_x, cam_y, cam_z]), np.array([w, h, l]), np.array([yaw])], axis=0).astype(np.float32),
    'cam',
    'lidar',
    np.linalg.inv(rect @ Trv2c),
)
# set_label(label, h, w, l, cam_x, cam_y, cam_z, alpha, yaw)
print(gt_bboxes_3d_lidar_ff.tolist())
print(gt_bboxes_3d_17.tolist())
print(gt_bboxes_3d.numpy()[0].tolist())
