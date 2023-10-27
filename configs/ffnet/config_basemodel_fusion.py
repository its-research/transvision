_base_ = ['../__base__/schedules/cyclic-40e.py', '../__base__/default_runtime.py', '../__base__/models/v2x_voxelnet.py']

dataset_type = 'V2XDatasetV2'
data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/ffnet/'
data_info_train_path = 'dair_infos_train.pkl'
data_info_val_path = 'dair_infos_val.pkl'
work_dir = './work_dirs/mmdet3d_1.2.0/ffnet-vic3d/basemodel/fusion'

point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]

input_modality = dict(use_lidar=True, use_camera=False)
class_names = ['Car']
metainfo = dict(classes=class_names)
backend_args = None

train_pipeline = [
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='vehicle'),
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='infrastructure'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetDAIRInputs',
        keys=['points', 'infrastructure_points', 'gt_bboxes_3d', 'gt_labels_3d'],
        # fmt:off
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'inf2veh',
                   'calib'),
        # fmt:on
    )
]
test_pipeline = [
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='vehicle'),
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='infrastructure'),
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1920, 1080),  # (1333, 800)
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1., 1.], translation_std=[0, 0, 0]),
    #         dict(type='RandomFlip3D'),
    #         dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range)
    #     ]),
    dict(
        type='Pack3DDetDAIRInputs',
        keys=['points', 'infrastructure_points', 'gt_bboxes_3d', 'gt_labels_3d'],
        # fmt:off
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'inf2veh',
                   'calib'),
        # fmt:on
    )
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='vehicle'),
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='infrastructure'),
    dict(
        type='Pack3DDetDAIRInputs',
        keys=['points', 'infrastructure_points', 'gt_bboxes_3d', 'gt_labels_3d'],
        # fmt:off
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'inf2veh',
                   'calib'),
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
            pcd_limit_range=point_cloud_range,  # not sure
            box_type_3d='LiDAR',
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=''),
        ann_file=data_info_val_path,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        pcd_limit_range=point_cloud_range,  # not sure
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=''),
        ann_file=data_info_val_path,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        pcd_limit_range=point_cloud_range,  # not sure
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(type='KittiMetric', ann_file=data_root + data_info_val_path, metric='bbox', pcd_limit_range=point_cloud_range, backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
epoch_num = 40
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01), clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=epoch_num * 0.4, eta_min=lr * 10, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=epoch_num * 0.6, eta_min=lr * 1e-4, begin=epoch_num * 0.4, end=epoch_num * 1, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=epoch_num * 0.4, eta_min=0.85 / 0.95, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=epoch_num * 0.6, eta_min=1, begin=epoch_num * 0.4, end=epoch_num * 1, convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
find_unused_parameters = True
