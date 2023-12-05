_base_ = ['./config_basemodel_fusion.py']

work_dir = './work_dirs/mmdet3d_1.3.0/ffnet-vic3d/basemodel/inf_only'
data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/ffnet/'
data_info_train_path = 'dair_infos_train.pkl'
data_info_val_path = 'dair_infos_val.pkl'

point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]
voxel_size = [0.16, 0.16, 4]

model = dict(mode='inf_only')

val_evaluator = [
    dict(type='KittiMetric', ann_file=data_root + data_info_val_path, metric='bbox', pcd_limit_range=point_cloud_range, backend_args=None),
    dict(
        type='DAIRV2XMetric',
        ann_file=data_root + data_info_val_path,
        veh_config_path='configs/ffnet/config_basemodel_inf_only.py',
        work_dir=work_dir,
        split_data_path='data/split_datas/cooperative-split-data.json',
        model='feature_fusion',
        input='data/DAIR-V2X/cooperative-vehicle-infrastructure',
        test_mode=None,
        val_data_path=None,
        pcd_limit_range=point_cloud_range,
        backend_args=None)
]
test_evaluator = val_evaluator

find_unused_parameters = True
load_from = None
