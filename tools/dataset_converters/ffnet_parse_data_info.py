import os

from mmengine.fileio import load

data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/'
veh_data_info_path = 'vehicle-side/data_info.json'
inf_data_info_path = 'infrastructure-side/data_info.json'
coo_data_info_path = 'cooperative/data_info.json'


def parse_data_info(file_path):
    infos = load(file_path)
    print(infos[0].keys())
    print('The length of infos is: ', len(infos))


parse_data_info(os.path.join(data_root, veh_data_info_path))
parse_data_info(os.path.join(data_root, inf_data_info_path))
parse_data_info(os.path.join(data_root, coo_data_info_path))
