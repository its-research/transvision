import argparse
import copy
import os

import numpy as np
# from gen_kitti.gen_calib2kitti import convert_calib_v2x_to_kitti, get_cam_D_and_cam_K, get_velo2cam
# from gen_kitti.label_lidarcoord_to_cameracoord import convert_point, get_camera_3d_8points
from gen_kitti.label_lidarcoord_to_cameracoord import get_label
# from kitti_data_utils import _extend_matrix
from mmdet3d.structures.bbox_3d.utils import limit_period
# from mmdet3d.structures import points_cam2img
from mmdet3d.structures.ops import box_np_ops
from mmengine.fileio import dump, load
from skimage import io
from tqdm import tqdm
from update_infos_to_v2 import get_empty_instance

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', type=str, default='cooperative')
parser.add_argument('--root-path', type=str, default='./data/DAIR-V2X/cooperative-vehicle-infrastructure/', help='specify the root path of dataset')
parser.add_argument('--dst-root-path', type=str, default='./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/ffnet', help='specify the root path of dataset')
parser.add_argument('--split_file_path', type=str, default='data/split_datas/cooperative-split-data.json', help='specify the split file')
args = parser.parse_args()


def get_split_list(split_path):
    ori_list = load(split_path)

    split_list = {}
    split_list['train'] = ori_list['cooperative_split']['train']
    split_list['val'] = ori_list['cooperative_split']['val']
    split_list['trainval'] = split_list['train'] + split_list['val']
    return split_list


def get_metainfo():
    metainfo = dict()

    metainfo['dataset'] = 'dair_v2x_dataset'
    metainfo['info_version'] = '1.0'
    metainfo['classes'] = ['Pedestrian', 'Cyclist', 'Car']
    metainfo['categories'] = {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2}
    return metainfo


def get_calibs(calib_path):
    calib = load(calib_path)
    if 'transform' in calib.keys():
        calib = calib['transform']
    rotation = calib['rotation']
    translation = calib['translation']
    return rotation, translation


def rev_matrix(rotation, translation):
    rotation = np.matrix(rotation)
    rev_R = rotation.I
    rev_R = np.array(rev_R)
    rev_T = -np.dot(rev_R, translation)
    return rev_R, rev_T


def inverse_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R


def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    rotation_1 = np.matrix(rotation_1)
    translation_1 = np.matrix(translation_1)
    rotation_2 = np.matrix(rotation_2)
    translation_2 = np.matrix(translation_2)

    rotation = rotation_2 * rotation_1
    translation = rotation_2 * translation_1 + translation_2
    rotation = np.array(rotation)
    translation = np.array(translation)

    return rotation, translation


def trans_lidar_i2v(inf_lidar2world_path, veh_lidar2novatel_path, veh_novatel2world_path, system_error_offset=None):
    inf_lidar2world_r, inf_lidar2world_t = get_calibs(inf_lidar2world_path)
    if system_error_offset is not None:
        inf_lidar2world_t[0][0] = inf_lidar2world_t[0][0] + system_error_offset['delta_x']
        inf_lidar2world_t[1][0] = inf_lidar2world_t[1][0] + system_error_offset['delta_y']

    veh_novatel2world_r, veh_novatel2world_t = get_calibs(veh_novatel2world_path)
    veh_world2novatel_r, veh_world2novatel_t = rev_matrix(veh_novatel2world_r, veh_novatel2world_t)
    inf_lidar2novatel_r, inf_lidar2novatel_t = mul_matrix(inf_lidar2world_r, inf_lidar2world_t, veh_world2novatel_r, veh_world2novatel_t)

    veh_lidar2novatel_r, veh_lidar2novatel_t = get_calibs(veh_lidar2novatel_path)
    veh_novatel2lidar_r, veh_novatel2lidar_t = rev_matrix(veh_lidar2novatel_r, veh_lidar2novatel_t)
    inf_lidar2lidar_r, inf_lidar2lidar_t = mul_matrix(inf_lidar2novatel_r, inf_lidar2novatel_t, veh_novatel2lidar_r, veh_novatel2lidar_t)

    return inf_lidar2lidar_r, inf_lidar2lidar_t


def get_calib_info(ori_info, root_path):
    calib = {}

    veh_idx = ori_info['vehicle_image_path'].split('/')[-1].replace('.jpg', '')
    calib_lidar_i2v_path = os.path.join(root_path, 'cooperative/calib/lidar_i2v/' + veh_idx + '.json')
    calib_lidar_i2v = load(calib_lidar_i2v_path)

    calib['lidar_i2v'] = calib_lidar_i2v
    return calib


def get_v2x_info(ori_info, root_path, info_file):
    veh_idx = ori_info['vehicle_image_path'].split('/')[-1].replace('.jpg', '')
    info_path = os.path.join(root_path, 'flow_data_jsons', info_file)
    info = load(info_path)

    data_list = info['data_list']
    datas = []
    for data in data_list:
        if data['vehicle_idx'] == veh_idx:
            datas.append(data)

    return datas


def get_images_info(ori_info, defalut_cam, root_path, veh_idx, inf_idx, sensor_view='vehicle'):
    images = {}
    for cam_idx in range(4):
        images['CAM%d' % cam_idx] = {}
        images['CAM%d' % cam_idx]['img_path'] = ''
        images['CAM%d' % cam_idx]['height'] = 0
        images['CAM%d' % cam_idx]['width'] = 0
        images['CAM%d' % cam_idx]['cam2img'] = []
        images['CAM%d' % cam_idx]['lidar2img'] = []
        images['CAM%d' % cam_idx]['lidar2cam'] = []
    images[defalut_cam]['img_path'] = ori_info[sensor_view + '_image_path']
    img_path = os.path.join(root_path, images[defalut_cam]['img_path'])
    image_shape = np.array(io.imread(img_path).shape[:2], dtype=np.int32)
    images[defalut_cam]['height'] = image_shape[0]
    images[defalut_cam]['width'] = image_shape[1]

    calib_v_lidar2cam_filename = os.path.join(root_path, 'vehicle-side/calib/lidar_to_camera', veh_idx + '.json')
    calib_v_cam_intrinsic_filename = os.path.join(root_path, 'vehicle-side/calib/camera_intrinsic/', veh_idx + '.json')

    # from ffnet
    calib_v_lidar2cam = load(calib_v_lidar2cam_filename)
    calib_v_cam_intrinsic = load(calib_v_cam_intrinsic_filename)
    rect = np.identity(4)
    Trv2c = np.identity(4)
    Trv2c[0:3, 0:3] = calib_v_lidar2cam['rotation']
    Trv2c[0:3, 3] = [calib_v_lidar2cam['translation'][0][0], calib_v_lidar2cam['translation'][1][0], calib_v_lidar2cam['translation'][2][0]]
    P2 = np.identity(4)
    P2[0:3, 0:3] = np.array(calib_v_cam_intrinsic['cam_K']).reshape(3, 3)

    images[defalut_cam]['cam2img'] = P2.tolist()
    images[defalut_cam]['tr_velo_to_cam'] = Trv2c.astype(np.float32).tolist()

    images['R0_rect'] = rect.tolist()
    lidar2cam = rect.astype(np.float32) @ Trv2c.astype(np.float32)
    images[defalut_cam]['lidar2cam'] = lidar2cam.tolist()
    images[defalut_cam]['lidar2img'] = (P2 @ lidar2cam).tolist()
    # from ffnet end

    # cam_D, cam_K = get_cam_D_and_cam_K(calib_v_cam_intrinsic_filename)
    # t_velo2cam, r_velo2cam = get_velo2cam(calib_v_lidar2cam_filename)

    # t_velo2cam = np.array(t_velo2cam).reshape(3, 1)
    # r_velo2cam = np.array(r_velo2cam).reshape(3, 3)
    # P2, Tr_velo_to_cam = convert_calib_v2x_to_kitti(cam_D, cam_K, t_velo2cam, r_velo2cam)
    # P2 = _extend_matrix(np.array(P2).reshape(3, 4))
    # Tr_velo_to_cam = _extend_matrix(np.array(Tr_velo_to_cam).reshape(3, 4))

    # R0_rect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
    # rect_4x4[3, 3] = 1.
    # rect_4x4[:3, :3] = R0_rect
    # R0_rect = rect_4x4
    # R0_rect = R0_rect

    # images[defalut_cam]['cam2img'] = P2.tolist()
    # images[defalut_cam]['tr_velo_to_cam'] = Tr_velo_to_cam.astype(np.float32).tolist()
    # images[defalut_cam]['t_velo2cam'] = t_velo2cam.tolist()
    # images[defalut_cam]['r_velo2cam'] = r_velo2cam.tolist()
    # images['R0_rect'] = R0_rect.tolist()
    # lidar2cam = R0_rect.astype(np.float32) @ Tr_velo_to_cam.astype(np.float32)
    # images[defalut_cam]['lidar2cam'] = lidar2cam.tolist()
    # images[defalut_cam]['lidar2img'] = (P2 @ lidar2cam).tolist()

    return images


def get_cam_instances(images, metainfo, root_path):
    cam_instances = {}
    for cam_idx in range(4):
        cam_instances['CAM%d' % cam_idx] = []

    return cam_instances


def add_difficulty_to_annos(dims, bbox, occlusion, truncation):
    min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
    height = bbox[3] - bbox[1]

    diff = []
    easy_mask = np.ones((len(dims), ), dtype=bool)
    moderate_mask = np.ones((len(dims), ), dtype=bool)
    hard_mask = np.ones((len(dims), ), dtype=bool)
    if occlusion > max_occlusion[0] or height <= min_height[0] or truncation > max_trunc[0]:
        easy_mask[0] = False
    if occlusion > max_occlusion[1] or height <= min_height[1] or truncation > max_trunc[1]:
        moderate_mask[0] = False
    if occlusion > max_occlusion[2] or height <= min_height[2] or truncation > max_trunc[2]:
        hard_mask[0] = False

    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    return diff[0]


def get_instances(images, lidar_points, metainfo, root_path):
    instances = []

    # TODO 直接从label_world中读取
    lidar_labels_path = os.path.join(root_path, 'cooperative/label/lidar', images[defalut_cam]['img_path'].replace('.jpg', '.json').split('/')[-1])

    label_infos = load(lidar_labels_path)
    if label_infos == []:
        return instances

    object_idx = 0
    for label_info in label_infos:
        instance = get_empty_instance()

        instance['bbox'] = [label_info['2d_box']['xmin'], label_info['2d_box']['ymin'], label_info['2d_box']['xmax'], label_info['2d_box']['ymax']]
        #
        # if label_info['type'] in ['truck', 'van', 'bus', 'car']:
        if label_info['type'] in ['car']:
            label_info['type'] = 'Car'

        if label_info['type'] in metainfo['classes']:
            instance['bbox_label'] = metainfo['classes'].index(label_info['type'])
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
        # bottom_center = [x, y, z]
        # obj_size = [l, h, w]  # [l, w, h]

        # bottom_center_in_cam = images[defalut_cam]['r_velo2cam'] * np.matrix(bottom_center).T + images[defalut_cam]['t_velo2cam']
        # alpha, yaw = get_camera_3d_8points(obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, images[defalut_cam]['r_velo2cam'], images[defalut_cam]['t_velo2cam'])
        # [cam_x, cam_y, cam_z, _] = convert_point(np.array([x, y, z, 1]).T, np.array(images[defalut_cam]['tr_velo_to_cam']).astype(np.float32))

        # dst = np.array([0.5, 0.5, 0.5])
        # src = np.array([0.5, 1.0, 0.5])

        # loc = np.array([cam_x, cam_y, cam_z])
        # dims = np.array([w, h, l])
        # rots = np.array([-yaw_lidar])

        extended_xyz = np.array([x, y, z, 1])
        loc = extended_xyz @ np.array(images[defalut_cam]['tr_velo_to_cam']).astype(np.float32).T
        loc = loc[:3]

        dims = np.array([l, h, w])  # 初始为wlh, 交换lhw
        # yaw = -yaw_lidar - np.pi / 2
        # yaw = -yaw_lidar
        # yaw = yaw_lidar - np.pi / 2  # Wrong
        yaw = -yaw_lidar - np.pi / 2
        yaw = limit_period(yaw, period=np.pi * 2)
        rots = np.array([yaw])

        gt_bboxes_3d = np.concatenate([loc, dims, rots]).tolist()  # camera coord
        instance['bbox_3d'] = gt_bboxes_3d

        # center_3d = loc + dims * (dst - src)
        # center_2d = points_cam2img(center_3d.reshape([1, 3]), images[defalut_cam]['cam2img'], with_depth=True)
        # center_2d = center_2d.squeeze().tolist()
        # instance['center_2d'] = center_2d[:2]
        # instance['depth'] = center_2d[2]
        # instance['truncated'] = int(label_info['truncated_state'])
        # instance['occluded'] = int(label_info['occluded_state'])
        # instance['alpha'] = alpha
        # TODO Fake data
        instance['alpha'] = -10
        instance['bbox'] = [0, 0, 100, 100]
        instance['truncated'] = 0
        instance['occluded'] = 0

        instance['score'] = 0.0
        instance['index'] = object_idx
        instance['group_id'] = len(label_infos)
        instance['difficulty'] = add_difficulty_to_annos(dims, instance['bbox'], instance['occluded'], instance['truncated'])
        object_idx += 1

        v_path = os.path.join(root_path, lidar_points['lidar_path'])
        points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, lidar_points['num_pts_feats']])
        rect = np.array(images['R0_rect'])
        Trv2c = np.array(lidar_points['Tr_velo_to_cam'])
        P2 = np.array(images[defalut_cam]['cam2img'])
        image_shape = [images[defalut_cam]['height'], images[defalut_cam]['width']]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2, image_shape)

        # dims_cam_kitti = np.array([l, h, w])
        # yaw_lidar = -yaw_lidar - np.pi / 2
        # rots_cam_kitti = limit_period(yaw_lidar, period=np.pi * 2)
        # rots_cam_kitti = np.array([rots_cam_kitti])  # -rots_cam_kitti 31071
        # rots_cam_kitti = np.array([-yaw_lidar])

        gt_boxes_camera = np.concatenate([loc, dims, rots])
        gt_boxes_camera = gt_boxes_camera[np.newaxis, :]
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = 0
        num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
        instance['num_lidar_pts'] = num_points_in_gt[0].astype(np.int32)

        instances.append(instance)
    return instances


if __name__ == '__main__':
    root_path = args.root_path
    v2x_info_gen = True
    if v2x_info_gen is False:
        dair_infos_trainval_path = os.path.join(args.dst_root_path, 'dair_infos_trainval.pkl')
        dair_infos_train_path = os.path.join(args.dst_root_path, 'dair_infos_train.pkl')
        dair_infos_val_path = os.path.join(args.dst_root_path, 'dair_infos_val.pkl')
    else:
        dair_infos_trainval_path = os.path.join(args.dst_root_path, 'dair_infos_flow_trainval.pkl')
        dair_infos_train_path = os.path.join(args.dst_root_path, 'dair_infos_flow_train.pkl')
        dair_infos_val_path = os.path.join(args.dst_root_path, 'dair_infos_flow_val.pkl')

    split_list = get_split_list(args.split_file_path)

    metainfo = get_metainfo()

    data_infos = {}
    data_infos['metainfo'] = metainfo
    data_infos['data_list'] = []

    data_infos_train = {}
    data_infos_train['metainfo'] = metainfo
    data_infos_train['data_list'] = []

    data_infos_val = {}
    data_infos_val['metainfo'] = metainfo
    data_infos_val['data_list'] = []

    ori_dair_infos_train = load(os.path.join(root_path, args.dataset, 'data_info.json'))
    defalut_cam = 'CAM2'

    flow_sample_idx = 0

    for ori_info in tqdm(ori_dair_infos_train):
        data_info = {
            'sample_idx': 0,
            'veh_sample_idx': 0,
            'inf_sample_idx': 0,
            'calib': {},
            'images': {},
            'lidar_points': {},
            'inf_images': {},
            'inf_lidar_points': {},
            'instances': {},
            'cam_instances': {}
        }
        data_info['veh_sample_idx'] = ori_info['vehicle_image_path'].split('/')[-1].replace('.jpg', '')
        data_info['inf_sample_idx'] = ori_info['infrastructure_image_path'].split('/')[-1].replace('.jpg', '')

        images = get_images_info(ori_info, defalut_cam, root_path, data_info['veh_sample_idx'], data_info['inf_sample_idx'])

        lidar_points = {}
        lidar_points['num_pts_feats'] = 4  # default value
        lidar_points['lidar_path'] = os.path.join('vehicle-side/velodyne', images[defalut_cam]['img_path'].split('/')[-1].replace('.jpg', '.bin'))
        lidar_points['Tr_velo_to_cam'] = images[defalut_cam]['tr_velo_to_cam']
        lidar_points['Tr_imu_to_velo'] = images[defalut_cam]['tr_velo_to_cam']  # equal to Tr_velo_to_cam

        lidar_points['inf_lidar_path'] = os.path.join('infrastructure-side/velodyne', data_info['inf_sample_idx'] + '.bin')

        cam_instances = get_cam_instances(images, metainfo, root_path)

        instances = get_instances(images, lidar_points, metainfo, root_path)

        calib = get_calib_info(ori_info, args.dst_root_path)

        if instances != []:
            frame_id = images[defalut_cam]['img_path'].split('/')[-1].replace('.jpg', '')
            sample_idx = int(frame_id)

            data_info['img_path'] = images[defalut_cam]['img_path']
            data_info['sample_idx'] = sample_idx
            data_info['images'] = images
            data_info['lidar_points'] = lidar_points
            data_info['cam_instances'] = cam_instances
            data_info['instances'] = instances
            data_info['calib'] = calib

            if v2x_info_gen:
                v2x_info_train_file = 'flow_data_info_train_2.json'
                v2x_info_val_file = 'flow_data_info_val_2.json'
            else:
                data_infos['data_list'].append(data_info)
                v2x_info_train_file = 'flow_data_info_train.json'
                v2x_info_val_file = 'flow_data_info_val_0.json'

            if frame_id in split_list['train']:
                v2x_infos = get_v2x_info(ori_info, args.dst_root_path, v2x_info_train_file)
                if v2x_infos is None:
                    continue
                for v2x_info in v2x_infos:
                    data_info_new = copy.deepcopy(data_info)
                    data_info_new['sample_idx'] = flow_sample_idx
                    data_info_new['v2x_info'] = v2x_info
                    data_infos_train['data_list'].append(data_info_new)
                    flow_sample_idx = flow_sample_idx + 1
                    if not v2x_info_gen:
                        data_infos['data_list'].append(data_info_new)
            else:
                v2x_infos = get_v2x_info(ori_info, args.dst_root_path, v2x_info_val_file)
                if v2x_infos is None:
                    continue
                for v2x_info in v2x_infos:
                    data_info_new = copy.deepcopy(data_info)
                    data_info_new['sample_idx'] = flow_sample_idx
                    flow_sample_idx = flow_sample_idx + 1
                    data_info_new['v2x_info'] = v2x_info
                    data_infos_val['data_list'].append(data_info_new)
                    if not v2x_info_gen:
                        data_infos['data_list'].append(data_info_new)

    dump(data_infos, dair_infos_trainval_path)
    dump(data_infos_train, dair_infos_train_path)
    dump(data_infos_val, dair_infos_val_path)
