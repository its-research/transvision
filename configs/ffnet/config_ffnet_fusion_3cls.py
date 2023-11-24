_base_ = [
    '../__base__/schedules/cyclic-40e.py',
    '../__base__/default_runtime.py',
    '../__base__/models/feature_flownet_3cls.py',
]
dataset_type = 'V2XDatasetV2'
data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/ffnet/'
# flownet_test_mode: {'FlowPred', 'OriginFeat', 'Async'}
# FlowPred: Use feature flow to compensate for the temporary asynchrony
# OriginFeat: Do not introduce the simulated temporal asychrony
# Async: Introduce the temporal asynchrony and do not use feature flow to compensate for the temporary asynchrony

data_info_train_path = 'dair_infos_flow_train.pkl'
data_info_val_path = 'dair_infos_flow_val.pkl'

work_dir = './work_dirs/mmdet3d_1.2.0/ffnet-vic3d/flow/fusion'

input_modality = dict(use_lidar=True, use_camera=False)
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None
point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]
voxel_size = [0.16, 0.16, 4]
l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
h = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
output_shape = [h, l]

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='vehicle',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure_t0',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure_t1',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure_t2',
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetDAIRInputs',
        keys=[
            'points',
            'infrastructure_points',
            'infrastructure_t0_points',
            'infrastructure_t1_points',
            'infrastructure_t2_points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        # fmt:off
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'inf2veh',
                   'infrastructure_pointcloud_bin_path_t_0', 'infrastructure_pointcloud_bin_path_t_1', 'infrastructure_pointcloud_bin_path_t_2', 'infrastructure_t_0_1',
                   'infrastructure_t_1_2', 'calib', 'v2x_info'),
        # fmt:on
    ),
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='vehicle',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure_t0',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure_t1',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure_t2',
    ),
    dict(
        type='Pack3DDetDAIRInputs',
        keys=['points', 'infrastructure_points', 'infrastructure_t0_points', 'infrastructure_t1_points', 'infrastructure_t2_points'],
        # fmt:off
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'inf2veh',
                   'infrastructure_pointcloud_bin_path_t_0', 'infrastructure_pointcloud_bin_path_t_1', 'infrastructure_pointcloud_bin_path_t_2', 'infrastructure_t_0_1',
                   'infrastructure_t_1_2', 'calib', 'v2x_info'),
        # fmt:on
    ),
]
eval_pipeline = [
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='vehicle'),
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='infrastructure'),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure_t0',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure_t1',
    ),
    dict(
        type='LoadPointsFromFile_w_sensor_view',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sensor_view='infrastructure_t2',
    ),
    dict(
        type='Pack3DDetDAIRInputs',
        keys=['points', 'infrastructure_points', 'infrastructure_t0_points', 'infrastructure_t1_points', 'infrastructure_t2_points', 'gt_bboxes_3d', 'gt_labels_3d'],
        # fmt:off
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'inf2veh',
                   'calib', 'v2x_info'),
        # fmt:on
    )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_info_train_path,
            data_prefix=dict(pts=''),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            pcd_limit_range=point_cloud_range,
            box_type_3d='LiDAR',
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_info_val_path,
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        pcd_limit_range=point_cloud_range,
        box_type_3d='LiDAR',
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_info_val_path,
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        pcd_limit_range=point_cloud_range,
        box_type_3d='LiDAR',
    ),
)

val_evaluator = [
    dict(type='KittiMetric', ann_file=data_root + data_info_val_path, metric='bbox', pcd_limit_range=point_cloud_range, backend_args=backend_args),
    dict(
        type='DAIRV2XMetric',
        ann_file=data_root + data_info_val_path,
        veh_config_path='configs/ffnet/config_ffnet_fusion_3cls.py',
        work_dir=work_dir,
        split_data_path='data/split_datas/cooperative-split-data.json',
        model='feature_flow',
        input='data/DAIR-V2X/cooperative-vehicle-infrastructure',
        test_mode='FlowPred',
        val_data_path='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/ffnet/flow_data_jsons/flow_data_info_val_1.json',
        pcd_limit_range=point_cloud_range,
        backend_args=backend_args)
]
test_evaluator = val_evaluator

lr = 0.0008
epoch_num = 10
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01), clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=epoch_num * 0.4, eta_min=lr * 10, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=epoch_num * 0.6, eta_min=lr * 1e-4, begin=epoch_num * 0.4, end=epoch_num * 1, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=epoch_num * 0.4, eta_min=0.85 / 0.95, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=epoch_num * 0.6, eta_min=1, begin=epoch_num * 0.4, end=epoch_num * 1, convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=10)
find_unused_parameters = True
