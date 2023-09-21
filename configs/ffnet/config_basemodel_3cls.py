default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

dataset_type = 'V2XDataset'
data_root = './data/DAIR-V2X/cooperative-vehicle-infrastructure/'
data_info_train_path = 'flow_data_jsons/flow_data_info_train.json'
data_info_val_path = 'flow_data_jsons/flow_data_info_val_0.json'
work_dir = './ffnet_work_dir/work_dir_baseline-V1'

input_modality = dict(use_lidar=True, use_camera=False)
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]
voxel_size = [0.16, 0.16, 4]
l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
h = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
output_shape = [h, l]
z_center_pedestrian = -0.6
z_center_cyclist = -0.6
z_center_car = -2.66

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
                   'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
                   'inf2veh'),
        # fmt:on
    )
]
test_pipeline = [
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='vehicle'),
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='infrastructure'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(h, l),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='Pack3DDetDAIRInputs',
                keys=['points', 'infrastructure_points'],
                # fmt:off
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
                           'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'inf2veh'),
                # fmt:on
            ),
        ],
    ),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='vehicle'),
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='infrastructure'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(h, l),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='Pack3DDetDAIRInputs',
                keys=['points', 'infrastructure_points'],
                # fmt:off
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
                           'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'inf2veh'),
                # fmt:on
            ),
        ],
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
            split='training',
            data_prefix=dict(pts='velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            pcd_limit_range=point_cloud_range,  # not sure
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
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
        data_prefix=dict(pts='velodyne_reduced'),
        ann_file=data_info_val_path,
        split='training',
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
        data_prefix=dict(pts='velodyne_reduced'),
        ann_file=data_info_val_path,
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        pcd_limit_range=point_cloud_range,  # not sure
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(type='KittiMetric', ann_file=data_info_val_path, metric='bbox', backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model = dict(
    type='V2XVoxelNet',
    voxel_layer=dict(
        max_num_points=100,  # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(40000, 40000)),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=100,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(40000, 40000))),
    voxel_encoder=dict(type='PillarFeatureNet', in_channels=4, feat_channels=[64], with_distance=False, voxel_size=voxel_size, point_cloud_range=point_cloud_range),
    middle_encoder=dict(type='PointPillarsScatter', in_channels=64, output_shape=output_shape),
    backbone=dict(type='SECOND', in_channels=64, layer_nums=[3, 5, 5], layer_strides=[2, 2, 2], out_channels=[64, 128, 256]),
    neck=dict(type='SECONDFPN', in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [point_cloud_range[0], point_cloud_range[1], z_center_pedestrian, point_cloud_range[3], point_cloud_range[4], z_center_pedestrian],
                [point_cloud_range[0], point_cloud_range[1], z_center_cyclist, point_cloud_range[3], point_cloud_range[4], z_center_cyclist],
                [point_cloud_range[0], point_cloud_range[1], z_center_car, point_cloud_range[3], point_cloud_range[4], z_center_car],
            ],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(use_rotate_nms=False, nms_across_levels=False, nms_thr=0.01, score_thr=0.2, min_bbox_size=0, nms_pre=1000, max_num=300))

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
epoch_num = 80
# optim_wrapper = dict(optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=epoch_num * 0.4, eta_min=lr * 10, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=epoch_num * 0.6, eta_min=lr * 1e-4, begin=epoch_num * 0.4, end=epoch_num * 1, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=epoch_num * 0.4, eta_min=0.85 / 0.95, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=epoch_num * 0.6, eta_min=1, begin=epoch_num * 0.4, end=epoch_num * 1, convert_to_iter_based=True)
]

# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.

# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.

# Runtime settings，training schedule for 40e
# Although the max_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual max epoch
# number could be Nx40
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (6 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=48)

file_client_args = dict(backend='disk')
find_unused_parameters = True
