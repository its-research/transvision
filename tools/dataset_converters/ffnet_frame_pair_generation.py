import argparse
import copy
import os
import random

from mmengine import dump as mmengine_dump
from mmengine import load as mmengine_load


def idx_batch_mapping(inf_data_infos):
    idx_batch_mappings = {}
    for inf_data_info in inf_data_infos:
        inf_idx = inf_data_info['pointcloud_path'].split('/')[-1].replace('.pcd', '')
        idx_batch_mappings[inf_idx] = inf_data_info['batch_id']

    return idx_batch_mappings


def split_datas(data_infos, split_datas, split='val'):
    data_infos_split = []

    inf_split_datas = split_datas['infrastructure_split'][split]
    for data_info in data_infos:
        infrastructure_frame = data_info['infrastructure_image_path'].split('/')[-1].replace('.jpg', '')
        if infrastructure_frame in inf_split_datas:
            data_infos_split.append(data_info)

    return data_infos_split


def data_info_flow_train(data_infos, inf_idx_batch_mappings):
    infrastructure_idxs = []
    for data_info in data_infos:
        infrastructure_idxs.append(data_info['infrastructure_idx'])

    data_infos_flow = []
    for data_info in data_infos:
        random_num = 3
        for ii in range(random_num):
            data_info_flow = copy.deepcopy(data_info)

            infrastructure_idx = data_info['infrastructure_idx']
            t_0_1 = random.randint(-3, 3)
            t_1 = str(int(infrastructure_idx) - t_0_1).zfill(6)
            if t_1 not in infrastructure_idxs or inf_idx_batch_mappings[infrastructure_idx] != inf_idx_batch_mappings[t_1]:
                continue
            data_info_flow['infrastructure_idx_t_1'] = t_1
            data_info_flow['infrastructure_pointcloud_bin_path_t_1'] = 'infrastructure-side/velodyne/' + t_1 + '.bin'

            t_0 = str(int(t_1) - 1).zfill(6)
            if t_0 not in infrastructure_idxs or inf_idx_batch_mappings[infrastructure_idx] != inf_idx_batch_mappings[t_0]:
                continue
            data_info_flow['infrastructure_idx_t_0'] = t_0
            data_info_flow['infrastructure_pointcloud_bin_path_t_0'] = 'infrastructure-side/velodyne/' + t_0 + '.bin'
            data_info_flow['infrastructure_t_0_1'] = 1.0

            delta_t_2 = random.randint(1, 2)
            t_2 = str(int(t_1) + delta_t_2).zfill(6)
            if t_2 not in infrastructure_idxs or inf_idx_batch_mappings[infrastructure_idx] != inf_idx_batch_mappings[t_2]:
                continue
            data_info_flow['infrastructure_idx_t_2'] = t_2
            data_info_flow['infrastructure_pointcloud_bin_path_t_2'] = 'infrastructure-side/velodyne/' + t_2 + '.bin'
            data_info_flow['infrastructure_t_1_2'] = delta_t_2 + 0.0

            data_infos_flow.append(data_info_flow)

    metainfo = dict()

    metainfo['dataset'] = 'v2x_dataset'
    metainfo['info_version'] = '1.1'
    metainfo['classes'] = ['Pedestrian', 'Cyclist', 'Car']
    metainfo['categories'] = [{'id': 0, 'name': 'Pedestrian'}, {'id': 1, 'name': 'Cyclist'}, {'id': 2, 'name': 'Car'}]
    converted_data_info = dict(metainfo=metainfo, data_list=data_infos_flow)

    return converted_data_info


def data_info_flow_val(data_infos, inf_idx_batch_mappings, async_k=1):
    infrastructure_idxs = []
    for data_info in data_infos:
        infrastructure_idxs.append(data_info['infrastructure_idx'])

    data_infos_flow = []
    for data_info in data_infos:

        data_info_flow = copy.deepcopy(data_info)

        infrastructure_idx = data_info['infrastructure_idx']
        t_0_1 = async_k
        t_1 = str(int(infrastructure_idx) - t_0_1).zfill(6)
        if t_1 not in infrastructure_idxs or inf_idx_batch_mappings[infrastructure_idx] != inf_idx_batch_mappings[t_1]:
            continue
        data_info_flow['infrastructure_idx_t_1'] = t_1
        data_info_flow['infrastructure_pointcloud_bin_path_t_1'] = 'infrastructure-side/velodyne/' + t_1 + '.bin'

        t_0 = str(int(t_1) - 1).zfill(6)
        if t_0 not in infrastructure_idxs or inf_idx_batch_mappings[infrastructure_idx] != inf_idx_batch_mappings[t_0]:
            continue
        data_info_flow['infrastructure_idx_t_0'] = t_0
        data_info_flow['infrastructure_pointcloud_bin_path_t_0'] = 'infrastructure-side/velodyne/' + t_0 + '.bin'
        delta_t_1 = 1.0
        data_info_flow['infrastructure_t_0_1'] = delta_t_1

        delta_t_2 = int(infrastructure_idx) - int(t_1)
        t_2 = str(int(t_1) + delta_t_2).zfill(6)
        if t_2 != infrastructure_idx:
            raise Exception('Index Setting Error', t_0, t_1, t_2)
        data_info_flow['infrastructure_idx_t_2'] = t_2
        data_info_flow['infrastructure_pointcloud_bin_path_t_2'] = 'infrastructure-side/velodyne/' + t_2 + '.bin'
        data_info_flow['infrastructure_t_1_2'] = delta_t_2 + 0.0

        data_infos_flow.append(data_info_flow)

    metainfo = dict()

    metainfo['dataset'] = 'v2x_dataset'
    metainfo['info_version'] = '1.1'
    metainfo['classes'] = ['Pedestrian', 'Cyclist', 'Car']
    metainfo['categories'] = [{'id': 0, 'name': 'Pedestrian'}, {'id': 1, 'name': 'Cyclist'}, {'id': 2, 'name': 'Car'}]
    converted_data_info = dict(metainfo=metainfo, data_list=data_infos_flow)

    return converted_data_info


parser = argparse.ArgumentParser('Preprocess the DAIR-V2X-C for FFNET.')
parser.add_argument('--source-root', type=str, default='./data/DAIR-V2X/cooperative-vehicle-infrastructure', help='Raw data root of DAIR-V2X-C.')
parser.add_argument('--target-root', type=str, default='./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/ffnet/', help='Target data root of DAIR-V2X-C.')

if __name__ == '__main__':
    args = parser.parse_args()
    dair_v2x_c_root = args.source_root
    dair_v2x_c_root_target = args.target_root
    if not os.path.exists(dair_v2x_c_root_target):
        os.makedirs(dair_v2x_c_root_target)

    inf_data_infos_path = os.path.join(dair_v2x_c_root, 'infrastructure-side/data_info.json')
    inf_data_infos = mmengine_load(inf_data_infos_path)
    inf_idx_batch_mappings = idx_batch_mapping(inf_data_infos)

    # You should split the data_info_new.json generated from preprocessing into train/val.
    split_json_path = os.path.join('data/split_datas', 'cooperative-split-data.json')
    split_jsons = mmengine_load(split_json_path)

    # Generate training part
    data_infos_path = os.path.join(dair_v2x_c_root, 'cooperative/data_info_new.json')
    data_infos = mmengine_load(data_infos_path)
    data_infos_train = split_datas(data_infos, split_jsons, split='train')

    data_infos_train_path = 'flow_data_info_train.json'
    flow_data_jsons_dir = os.path.join(dair_v2x_c_root_target, 'flow_data_jsons')
    if not os.path.exists(flow_data_jsons_dir):
        os.makedirs(flow_data_jsons_dir)

    metainfo = dict()
    metainfo['dataset'] = 'v2x_dataset'
    metainfo['info_version'] = '1.1'
    metainfo['classes'] = ['Pedestrian', 'Cyclist', 'Car']
    metainfo['categories'] = [{'id': 0, 'name': 'Pedestrian'}, {'id': 1, 'name': 'Cyclist'}, {'id': 2, 'name': 'Car'}]
    converted_data_info = dict(metainfo=metainfo, data_list=data_infos_train)
    data_infos_train_path = os.path.join(flow_data_jsons_dir, data_infos_train_path)
    mmengine_dump(data_infos_train, data_infos_train_path)
    data_infos_flow_train = data_info_flow_train(data_infos_train, inf_idx_batch_mappings)
    data_infos_flow_path = 'flow_data_info_train_2.json'
    data_infos_flow_path = os.path.join(flow_data_jsons_dir, data_infos_flow_path)
    mmengine_dump(data_infos_flow_train, data_infos_flow_path)

    # Generate val part
    data_infos_path = os.path.join(dair_v2x_c_root, 'cooperative/data_info_new.json')
    data_infos = mmengine_load(data_infos_path)
    data_infos_val = split_datas(data_infos, split_jsons, split='val')

    for async_k in range(0, 6):
        data_infos_flow_val = data_info_flow_val(data_infos_val, inf_idx_batch_mappings, async_k=async_k)
        print('The length of data_infos_flow_val is: ', async_k, len(data_infos_flow_val['data_list']))
        data_infos_flow_path = 'flow_data_info_val_' + str(async_k) + '.json'
        data_infos_flow_path = os.path.join(flow_data_jsons_dir, data_infos_flow_path)
        mmengine_dump(data_infos_flow_val, data_infos_flow_path)
