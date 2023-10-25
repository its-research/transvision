import pickle

import numpy as np


def parse_pkl_file(file_path, version='1.2.0-info'):
    """Parse the pkl file and return the data."""
    with open(file_path, 'rb') as pkl_data:
        data = pickle.load(pkl_data)

        if version == '1.2.0-info':
            # print(data.keys())
            # print(data["metainfo"])
            print(data['data_list'][0].keys())
            # print(data["data_list"][0]["images"].keys())
            print(data['data_list'][10]['lidar_points']['lidar_path'])
            print(data['data_list'][0]['instances'][0].keys())
            # print(data["data_list"][0]["cam_instances"].keys())

            for ins in data['data_list'][10]['instances']:
                print(ins['bbox_3d'])
                break

        if version == '1.2.0-dbinfo':
            # print(data["Car"][0].keys())
            print(data['Car'][0]['box3d_lidar'])

        if version == '0.17.1-info':
            # print(data[0].keys())
            # print(data[0]["annos"].keys())
            print(data[0]['annos']['location'][0])
            print(data[0]['annos']['dimensions'][0])
            print(data[0]['annos']['rotation_y'][0])

        if version == '0.17.1-dbinfo':
            # print(data["Car"][0].keys())
            print(data['Car'][0]['box3d_lidar'])


# print("1.2.0")
# parse_pkl_file("1.2.0/kitti_infos_train.pkl", version="1.2.0-info")
# parse_pkl_file("1.2.0/kitti_dbinfos_train.pkl", version="1.2.0-dbinfo")

# print("0.17.1 to 1.2.0")
# parse_pkl_file("update/kitti_infos_train.pkl", version="1.2.0-info")
# parse_pkl_file("update/kitti_dbinfos_train.pkl", version="1.2.0-dbinfo")

# print("0.17.1")
# parse_pkl_file("0.17.1/kitti_infos_train.pkl", version="0.17.1-info")
# parse_pkl_file("0.17.1/kitti_dbinfos_train.pkl", version="0.17.1-dbinfo")

# with open('dair_infos_trainval.pkl', "rb") as pkl_data:
#     data = pickle.load(pkl_data)
# print("data['data_list']:", data['data_list'][0].keys())
# print('sample_idx: ', data['data_list'][0]['sample_idx'])
# print('images: ', data['data_list'][0]['images'].keys())
# print('images: ', data['data_list'][0]['images']["CAM2"])
# print('R0_rect: ', data['data_list'][0]['images']["R0_rect"])
# for key in data['data_list'][0]['images'].keys():
#     if key != 'R0_rect':
#         print(key, ': ', data['data_list'][0]['images'][key].keys())

# print('lidar_points: ', data['data_list'][0]['lidar_points'].keys())
# print('lidar_points: ', data['data_list'][0]['lidar_points'])

# print('cam_instances: ', data['data_list'][0]['cam_instances'].keys())
# print('cam_instances: ', data['data_list'][0]['cam_instances']['CAM2'])
# for key in data['data_list'][0]['cam_instances']['CAM2'][0].keys():
#     print(key, ': ', data['data_list'][0]['cam_instances']['CAM2'][0][key])

# print('instances: ', data['data_list'][0]['instances'][0].keys())
# for key in data['data_list'][0]['instances'][0].keys():
#     print(key, ': ', data['data_list'][0]['instances'][0][key])
# count = 0
# for d in data['data_list']:
#     # if d['instances'] == []:
#     #     print(d['sample_idx'])
#     #     # print(d)
#     #     count+=1
#     for d1 in d['instances']:
#         if d1['bbox_label'] == -1:
#             print(d['sample_idx'])

# if d['lidar_points']['lidar_path'] == '012784.bin':
#     print(d)
#     print(d['sample_idx'])

# print("1.2.0")
# parse_pkl_file("kitti_infos_train.pkl", version="1.2.0-info")
# parse_pkl_file("1.2.0/kitti_dbinfos_train.pkl", version="1.2.0-dbinfo")


def get_dair_data(data, idx):
    """根据sample_idx获取dair_data中的数据."""
    for d in data:
        if d['sample_idx'] == idx:
            return d
    return None


dair_data_file = open('data/DAIR-V2X/cooperative-vehicle-infrastructure/dair_infos_trainval.pkl', 'rb')
dair_data = pickle.load(dair_data_file)['data_list']

kitti_data_file = open('data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/kitti_infos_trainval.pkl', 'rb')
kitti_data = pickle.load(kitti_data_file)['data_list']

pt_diff_count = 0
for kitti in kitti_data:
    sample_idx = kitti['sample_idx']
    dair = get_dair_data(dair_data, sample_idx)
    # print('sample_idx:', sample_idx)

    # print('keys:', kitti.keys())
    # check images
    # print('images key:', kitti['images'].keys())

    if kitti['images']['R0_rect'] != dair['images']['R0_rect']:
        print('R0_rect is different')
        print(kitti['images']['R0_rect'])
        print(dair['images']['R0_rect'])
    kitti_cam2 = kitti['images']['CAM2']
    dair_cam2 = dair['images']['CAM2']
    # print('CAM2 keys:', kitti_cam2.keys())
    if kitti_cam2['img_path'] != dair_cam2['img_path'].split('/')[-1]:
        print('img_path is different')
        print(kitti_cam2['img_path'])
        print(dair_cam2['img_path'])
    if kitti_cam2['height'] != dair_cam2['height']:
        print('height is different')
        print(kitti_cam2['height'])
        print(dair_cam2['height'])
    if kitti_cam2['width'] != dair_cam2['width']:
        print('width is different')
        print(kitti_cam2['width'])
        print(dair_cam2['width'])
    if kitti_cam2['cam2img'] != dair_cam2['cam2img']:
        print('cam2img is different')
        print(kitti_cam2['cam2img'])
        print(dair_cam2['cam2img'])

    # kitti_cam2['lidar2img'] = np.array(kitti_cam2['lidar2img'])
    # dair_cam2['lidar2img'] = np.array(dair_cam2['lidar2img'])
    # comp = np.isclose(kitti_cam2['lidar2img'], dair_cam2['lidar2img'])
    # print(comp.all())

    if kitti_cam2['lidar2img'] != dair_cam2['lidar2img']:
        print('lidar2img is different')
        print(kitti_cam2['lidar2img'])
        print(dair_cam2['lidar2img'])
    if kitti_cam2['lidar2cam'] != dair_cam2['lidar2cam']:
        print('lidar2cam is different')
        print(kitti_cam2['lidar2cam'])
        print(dair_cam2['lidar2cam'])

    kitti_lidar_points = kitti['lidar_points']
    dair_lidar_points = dair['lidar_points']
    # print('lidar_points keys:', kitti_lidar_points.keys())

    if kitti_lidar_points['num_pts_feats'] != dair_lidar_points['num_pts_feats']:
        print('num_pts_feats is different')
        print(kitti_lidar_points['num_pts_feats'])
        print(dair_lidar_points['num_pts_feats'])

    if kitti_lidar_points['lidar_path'] != dair_lidar_points['lidar_path'].split('/')[-1]:
        print('lidar_path is different')
        print(kitti_lidar_points['lidar_path'])
        print(dair_lidar_points['lidar_path'])

    if kitti_lidar_points['Tr_velo_to_cam'] != dair_lidar_points['Tr_velo_to_cam']:
        print('Tr_velo_to_cam is different')
        print(kitti_lidar_points['Tr_velo_to_cam'])
        print(dair_lidar_points['Tr_velo_to_cam'])

    if kitti_lidar_points['Tr_imu_to_velo'] != dair_lidar_points['Tr_imu_to_velo']:
        print('Tr_imu_to_velo is different')
        print(kitti_lidar_points['Tr_imu_to_velo'])
        print(dair_lidar_points['Tr_imu_to_velo'])

    kitti_instances = kitti['instances']
    dair_instances = dair['instances']
    # print('instances keys:', kitti_instances[0].keys())

    isinstance_num = len(kitti_instances)

    for i in range(isinstance_num):
        # ['bbox', 'bbox_label', 'bbox_3d', 'bbox_label_3d', 'depth', 'center_2d', 'num_lidar_pts', 'difficulty', 'truncated', 'occluded', 'alpha', 'score', 'index', 'group_id'])
        if kitti_instances[i]['bbox'] != dair_instances[i]['bbox']:
            print('bbox is different')
            print(kitti_instances[i]['bbox'])
            print(dair_instances[i]['bbox'])
        if kitti_instances[i]['bbox_label'] != dair_instances[i]['bbox_label']:
            print('bbox_label is different')
            print(kitti_instances[i]['bbox_label'])
            print(dair_instances[i]['bbox_label'])

        kitti_instances[i]['bbox_3d'] = np.array(kitti_instances[i]['bbox_3d'])
        dair_instances[i]['bbox_3d'] = np.array(dair_instances[i]['bbox_3d'])
        comp = np.isclose(kitti_instances[i]['bbox_3d'], dair_instances[i]['bbox_3d'])
        if comp.all() is False:
            print('bbox_3d is different')
            print(kitti_instances[i]['bbox_3d'])
            print(dair_instances[i]['bbox_3d'])
        if kitti_instances[i]['bbox_label_3d'] != dair_instances[i]['bbox_label_3d']:
            print('bbox_label_3d is different')
            print(kitti_instances[i]['bbox_label_3d'])
            print(dair_instances[i]['bbox_label_3d'])
        if np.isclose(kitti_instances[i]['depth'], dair_instances[i]['depth']) is False:
            print('depth is different')
            print(kitti_instances[i]['depth'])
            print(dair_instances[i]['depth'])

        kitti_instances[i]['center_2d'] = np.array(kitti_instances[i]['center_2d'])
        dair_instances[i]['center_2d'] = np.array(dair_instances[i]['center_2d'])
        comp = np.isclose(kitti_instances[i]['center_2d'], dair_instances[i]['center_2d'])
        if comp.all() is False:
            print('center_2d is different')
            print(kitti_instances[i]['center_2d'])
            print(dair_instances[i]['center_2d'])
        if kitti_instances[i]['num_lidar_pts'] != dair_instances[i]['num_lidar_pts']:
            print('sample_idx:', sample_idx)
            print('num_lidar_pts is different')
            print(kitti_instances[i]['num_lidar_pts'])
            print(dair_instances[i]['num_lidar_pts'])
            pt_diff_count += 1
        if kitti_instances[i]['difficulty'] != dair_instances[i]['difficulty']:
            print('difficulty is different')
            print(kitti_instances[i]['difficulty'])
            print(dair_instances[i]['difficulty'])
        if kitti_instances[i]['truncated'] != dair_instances[i]['truncated']:
            print('truncated is different')
            print(kitti_instances[i]['truncated'])
            print(dair_instances[i]['truncated'])
        if kitti_instances[i]['occluded'] != dair_instances[i]['occluded']:
            print('occluded is different')
            print(kitti_instances[i]['occluded'])
            print(dair_instances[i]['occluded'])
        if np.isclose(kitti_instances[i]['alpha'], dair_instances[i]['alpha']) is False:
            print('alpha is different')
            print(kitti_instances[i]['alpha'])
            print(dair_instances[i]['alpha'])
        if kitti_instances[i]['score'] != dair_instances[i]['score']:
            print('score is different')
            print(kitti_instances[i]['score'])
            print(dair_instances[i]['score'])
        if kitti_instances[i]['index'] != dair_instances[i]['index']:
            print('index is different')
            print(kitti_instances[i]['index'])
            print(dair_instances[i]['index'])
        # if kitti_instances[i]['group_id'] != dair_instances[i]['group_id']:
        #     print('group_id is different')
        #     print(kitti_instances[i]['group_id'])
        #     print(dair_instances[i]['group_id'])
print(pt_diff_count)
