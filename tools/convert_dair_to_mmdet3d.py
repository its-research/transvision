import argparse
import os

import numpy as np
from dataset_converters.gen_kitti.gen_calib2kitti import convert_calib_v2x_to_kitti, get_cam_D_and_cam_K, get_velo2cam
from dataset_converters.gen_kitti.label_lidarcoord_to_cameracoord import convert_point, get_camera_3d_8points, get_label
from dataset_converters.kitti_data_utils import _extend_matrix
from dataset_converters.update_infos_to_v2 import get_empty_instance
from mmdet3d.structures import points_cam2img
from mmdet3d.structures.ops import box_np_ops
from mmengine.fileio import dump, load
from skimage import io
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', type=str, default='vehicle-side')
parser.add_argument('--root-path', type=str, default='./data/DAIR-V2X/cooperative-vehicle-infrastructure/', help='specify the root path of dataset')
args = parser.parse_args()


def get_metainfo():
    metainfo = dict()

    metainfo['dataset'] = 'dair_v2x_dataset'
    metainfo['info_version'] = '1.0'
    metainfo['classes'] = ['Pedestrian', 'Cyclist', 'Car']
    metainfo['categories'] = {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2}
    return metainfo


if __name__ == '__main__':
    # dair_infos_train_path = os.path.join(args.root_path, 'dair_infos_train.pkl')
    # dair_infos_val_path = os.path.join(args.root_path, 'dair_infos_val.pkl')
    root_path = os.path.join(args.root_path, args.dataset)
    dair_infos_trainval_path = os.path.join(root_path, 'dair_infos_trainval.pkl')

    data_infos = {}
    data_infos['metainfo'] = get_metainfo()
    data_infos['data_list'] = []

    ori_dair_infos_train = load(os.path.join(root_path, 'data_info.json'))

    sample_idx = 0

    for ori_info in tqdm(ori_dair_infos_train):
        data_info = {'sample_idx': 0, 'images': {}, 'lidar_points': {}, 'instances': {}, 'cam_instances': {}}

        images = {}
        defalut_cam = 'CAM2'
        for cam_idx in range(4):
            images['CAM%d' % cam_idx] = {}
            images['CAM%d' % cam_idx]['img_path'] = ''
            images['CAM%d' % cam_idx]['height'] = 0
            images['CAM%d' % cam_idx]['width'] = 0
            images['CAM%d' % cam_idx]['cam2img'] = []
            images['CAM%d' % cam_idx]['lidar2img'] = []
            images['CAM%d' % cam_idx]['lidar2cam'] = []
        images[defalut_cam]['img_path'] = ori_info['image_path'].split('/')[-1]
        img_path = os.path.join(root_path, 'image', images[defalut_cam]['img_path'])
        image_shape = np.array(io.imread(img_path).shape[:2], dtype=np.int32)
        images[defalut_cam]['height'] = image_shape[0]
        images[defalut_cam]['width'] = image_shape[1]

        sensor_view = args.dataset
        path_camera_intrinsic = os.path.join(root_path, 'calib/camera_intrinsic')
        if sensor_view == 'vehicle-side' or sensor_view == 'cooperative':
            path_lidar_to_camera = os.path.join(root_path, 'calib/lidar_to_camera')
        else:
            path_lidar_to_camera = os.path.join(root_path, 'calib/virtuallidar_to_camera')
        camera_intrisinc_path = os.path.join(path_camera_intrinsic, images[defalut_cam]['img_path'].replace('.jpg', '.json'))
        cam_D, cam_K = get_cam_D_and_cam_K(camera_intrisinc_path)
        lidar_to_camera_path = os.path.join(path_lidar_to_camera, images[defalut_cam]['img_path'].replace('.jpg', '.json'))
        t_velo2cam, r_velo2cam = get_velo2cam(lidar_to_camera_path)

        t_velo2cam = np.array(t_velo2cam).reshape(3, 1)
        r_velo2cam = np.array(r_velo2cam).reshape(3, 3)
        P2, Tr_velo_to_cam = convert_calib_v2x_to_kitti(cam_D, cam_K, t_velo2cam, r_velo2cam)
        P2 = _extend_matrix(np.array(P2).reshape(3, 4))
        Tr_velo_to_cam = _extend_matrix(np.array(Tr_velo_to_cam).reshape(3, 4))
        images[defalut_cam]['cam2img'] = P2

        R0_rect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
        R0_rect = rect_4x4
        images['R0_rect'] = R0_rect
        lidar2cam = R0_rect @ Tr_velo_to_cam
        images[defalut_cam]['lidar2cam'] = lidar2cam
        images[defalut_cam]['lidar2img'] = P2 @ lidar2cam

        lidar_points = {}
        lidar_points['num_pts_feats'] = 4  # default value
        lidar_points['lidar_path'] = images[defalut_cam]['img_path'].replace('.jpg', '.bin')
        lidar_points['Tr_velo_to_cam'] = Tr_velo_to_cam.tolist()
        lidar_points['Tr_imu_to_velo'] = Tr_velo_to_cam.tolist()  # equal to Tr_velo_to_cam

        cam_instances = {}
        for cam_idx in range(4):
            cam_instances['CAM%d' % cam_idx] = []

        cam_labels_path = os.path.join(root_path, 'label/camera', images[defalut_cam]['img_path'].replace('.jpg', '.json'))
        label_infos = load(cam_labels_path)
        if label_infos == []:
            continue
        for label_info in label_infos:

            cam_instance = {}
            cam_instance['bbox'] = [label_info['2d_box']['xmin'], label_info['2d_box']['ymin'], label_info['2d_box']['xmax'], label_info['2d_box']['ymax']]
            # TODO 考虑转换Truck、Van和Bus
            if label_info['type'] in ['Truck', 'Van', 'Bus']:
                label_info['type'] == 'Car'

            if label_info['type'] in data_infos['metainfo']['classes']:
                cam_instance['bbox_label'] = data_infos['metainfo']['classes'].index(label_info['type'])
                cam_instance['bbox_label_3d'] = cam_instance['bbox_label']
            else:
                cam_instance['bbox_label'] = -1
                cam_instance['bbox_label_3d'] = -1
                continue

            h, w, l, x, y, z, yaw_lidar = get_label(label_info)
            z = z - h / 2
            bottom_center = [x, y, z]
            obj_size = [l, w, h]

            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam)
            [cam_x, cam_y, cam_z, _] = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

            cam_instance['bbox_3d'] = [cam_x, cam_y, cam_z, l, w, h, yaw]
            cam_instance['velocity'] = -1

            loc = np.array([cam_x, cam_y, cam_z])
            dims = np.array([l, w, h])

            dst = np.array([0.5, 0.5, 0.5])
            src = np.array([0.5, 1.0, 0.5])

            center_3d = loc + dims * (dst - src)
            center_2d = points_cam2img(center_3d.reshape([1, 3]), images[defalut_cam]['cam2img'], with_depth=True)
            center_2d = center_2d.squeeze().tolist()

            cam_instance['center_2d'] = center_2d[:2]
            cam_instance['depth'] = center_2d[2]

            # loc_center = loc + dims * (dst - src)

            # gt_bbox_3d = np.concatenate([loc_center, dims, [yaw]]).astype(np.float32)

            # corners_3d = box_np_ops.center_to_corner_box3d(gt_bbox_3d[:, :3], gt_bbox_3d[:, 3:6], gt_bbox_3d[:, 6], (0.5, 0.5, 0.5), axis=1)
            # corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
            # in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            # corners_3d = corners_3d[:, in_front]

            # # Project 3d box to 2d.
            # camera_intrinsic = data_info['images']["CAM%d"]['cam2img']
            # corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

            # # Keep only corners that fall within the image.
            # final_coords = post_process_coords(corner_coords, imsize=(data_info['images'][defalut_cam]['width'], data_info['images'][defalut_cam]['height']))

            # # Skip if the convex hull of the re-projected corners
            # # does not intersect the image canvas.
            # if final_coords is None:
            #     continue
            # else:
            #     min_x, min_y, max_x, max_y = final_coords

            # # Generate dictionary record to be included in the .json file.
            # cam_instance['bbox'] = [min_x, min_y, max_x, max_y]
            # cam_instance['bbox_3d_isvalid'] = True

            # # If mono3d=True, add 3D annotations in camera coordinates
            # # use bottom center to represent the bbox_3d
            # cam_instance['bbox_3d'] = np.concatenate([loc, dims, yaw], axis=1).astype(np.float32).squeeze().tolist()
            # cam_instance['velocity'] = -1  # no velocity in KITTI

            # center_3d = np.array(loc_center).reshape([1, 3])
            # center_2d_with_depth = points_cam2img(
            #     center_3d, camera_intrinsic, with_depth=True)
            # center_2d_with_depth = center_2d_with_depth.squeeze().tolist()

            # cam_instance['center_2d'] = center_2d_with_depth[:2]
            # cam_instance['depth'] = center_2d_with_depth[2]

            cam_instances[defalut_cam].append(cam_instance)

        instances = []

        lidar_labels_path = os.path.join(root_path, 'label/lidar', images[defalut_cam]['img_path'].replace('.jpg', '.json'))
        label_infos = load(lidar_labels_path)
        if label_infos == []:
            continue

        object_idx = 0
        for label_info in label_infos:
            instance = get_empty_instance()

            instance['bbox'] = [label_info['2d_box']['xmin'], label_info['2d_box']['ymin'], label_info['2d_box']['xmax'], label_info['2d_box']['ymax']]

            if label_info['type'] in ['Truck', 'Van', 'Bus']:
                label_info['type'] = 'Car'

            if label_info['type'] in data_infos['metainfo']['classes']:
                instance['bbox_label'] = data_infos['metainfo']['classes'].index(label_info['type'])
                instance['bbox_label_3d'] = instance['bbox_label']
                instance['attr_label'] = instance['bbox_label']
            else:
                instance['bbox_label'] = -1
                instance['bbox_label_3d'] = -1
                instance['attr_label'] = -1
                continue

            h, w, l, x, y, z, yaw_lidar = get_label(label_info)
            if w == 0.0 or h == 0.0 or l == 0.0:
                continue
            z = z - h / 2
            bottom_center = [x, y, z]
            obj_size = [l, w, h]

            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam)
            [cam_x, cam_y, cam_z, _] = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

            dst = np.array([0.5, 0.5, 0.5])
            src = np.array([0.5, 1.0, 0.5])

            loc = np.array([cam_x, cam_y, cam_z])
            dims = np.array([w, h, l])
            rots = np.array([-yaw_lidar])

            center_3d = loc + dims * (dst - src)
            center_2d = points_cam2img(center_3d.reshape([1, 3]), images[defalut_cam]['cam2img'], with_depth=True)
            center_2d = center_2d.squeeze().tolist()
            instance['center_2d'] = center_2d[:2]
            instance['depth'] = center_2d[2]
            gt_bboxes_3d = np.concatenate([loc, dims, rots]).tolist()
            instance['bbox_3d'] = gt_bboxes_3d
            instance['truncated'] = label_info['truncated_state']
            instance['occluded'] = label_info['occluded_state']
            instance['alpha'] = alpha
            instance['score'] = 0.0
            instance['index'] = object_idx
            instance['group_id'] = 0
            instance['difficulty'] = 2
            object_idx += 1

            v_path = os.path.join(root_path, 'velodyne', lidar_points['lidar_path'])
            points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, lidar_points['num_pts_feats']])
            rect = images['R0_rect']
            Trv2c = lidar_points['Tr_velo_to_cam']
            P2 = images[defalut_cam]['cam2img']
            image_shape = [images[defalut_cam]['height'], images[defalut_cam]['width']]
            points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2, image_shape)

            gt_boxes_camera = np.concatenate([loc, dims, rots])
            gt_boxes_camera = gt_boxes_camera[np.newaxis, :]
            gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c)
            indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
            num_points_in_gt = indices.sum(0)
            num_ignored = 0
            num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
            instance['num_lidar_pts'] = num_points_in_gt[0]

            instances.append(instance)

        if instances != []:
            data_info['sample_idx'] = sample_idx
            data_info['images'] = images
            data_info['lidar_points'] = lidar_points
            data_info['cam_instances'] = cam_instances
            data_info['instances'] = instances
            sample_idx += 1

            data_infos['data_list'].append(data_info)

    dump(data_infos, dair_infos_trainval_path)
