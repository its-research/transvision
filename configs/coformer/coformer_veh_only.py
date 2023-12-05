_base_ = ['./coformer.py']
work_dir = './work_dirs/mmdet3d_1.3.0/coformer/basemodel/veh_only'

model = dict(mode='veh_only')  # veh_only, inf_only, fusion
data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/ffnet/'
data_info_val_path = 'dair_infos_val.pkl'

val_evaluator = [
    dict(type='KittiMetric', ann_file=data_root + data_info_val_path, metric='bbox', pcd_limit_range={{_base_.point_cloud_range}}, backend_args=None),
    dict(
        type='DAIRV2XMetric',
        ann_file=data_root + data_info_val_path,
        veh_config_path='configs/coformer/coformer_veh_only.py',
        work_dir=work_dir,
        split_data_path='data/split_datas/cooperative-split-data.json',
        model='coformer',
        input='data/DAIR-V2X/cooperative-vehicle-infrastructure',
        test_mode=None,
        val_data_path=None,
        pcd_limit_range={{_base_.point_cloud_range}},
        backend_args=None)
]
test_evaluator = val_evaluator
