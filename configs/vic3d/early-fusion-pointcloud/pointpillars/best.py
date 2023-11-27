auto_scale_lr = dict(base_batch_size=48, enable=False)
backend_args = None
class_names = [
    'Car',
]
data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/early-fusion/'
dataset_type = 'KittiDataset'
db_sampler = dict(
    backend_args=None,
    classes=[
        'Car',
    ],
    data_root='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/early-fusion/',
    info_path='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/early-fusion/kitti_dbinfos_train.pkl',
    points_loader=dict(backend_args=None, coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
    prepare=dict(filter_by_difficulty=[
        -1,
    ], filter_by_min_points=dict(Car=5)),
    rate=1.0,
    sample_groups=dict(Car=15))
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, type='CheckpointHook'),
    logger=dict(interval=300, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(cudnn_benchmark=False, dist_cfg=dict(backend='nccl'), mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
epoch_num = 80
eval_pipeline = [
    dict(backend_args=None, coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
file_client_args = dict(backend='disk')
height = 576
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'pytorch'
length = 576
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=100)
lr = 0.0018
metainfo = dict(classes=[
    'Car',
])
model = dict(
    backbone=dict(in_channels=64, layer_nums=[
        3,
        5,
        5,
    ], layer_strides=[
        2,
        2,
        2,
    ], out_channels=[
        64,
        128,
        256,
    ], type='SECOND'),
    bbox_head=dict(
        anchor_generator=dict(
            ranges=[
                [
                    0,
                    -46.08,
                    -2.66,
                    92.16,
                    46.08,
                    -2.66,
                ],
            ], reshape_out=False, rotations=[
                0,
                1.57,
            ], sizes=[
                [
                    3.9,
                    1.6,
                    1.56,
                ],
            ], type='Anchor3DRangeGenerator'),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        diff_rad_by_sin=True,
        feat_channels=384,
        in_channels=384,
        loss_bbox=dict(beta=0.1111111111111111, loss_weight=2.0, type='mmdet.SmoothL1Loss'),
        loss_cls=dict(alpha=0.25, gamma=2.0, loss_weight=1.0, type='mmdet.FocalLoss', use_sigmoid=True),
        loss_dir=dict(loss_weight=0.2, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        num_classes=1,
        type='Anchor3DHead',
        use_direction_classifier=True),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(max_num_points=100, max_voxels=(
            16000,
            40000,
        ), point_cloud_range=[
            0,
            -46.08,
            -3,
            92.16,
            46.08,
            1,
        ], voxel_size=[
            0.16,
            0.16,
            4,
        ])),
    middle_encoder=dict(in_channels=64, output_shape=[
        576,
        576,
    ], type='PointPillarsScatter'),
    neck=dict(in_channels=[
        64,
        128,
        256,
    ], out_channels=[
        128,
        128,
        128,
    ], type='SECONDFPN', upsample_strides=[
        1,
        2,
        4,
    ]),
    test_cfg=dict(max_num=300, min_bbox_size=0, nms_across_levels=False, nms_pre=1000, nms_thr=0.01, score_thr=0.2, use_rotate_nms=False),
    train_cfg=dict(
        allowed_border=0,
        assigner=[
            dict(ignore_iof_thr=-1, iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'), min_pos_iou=0.45, neg_iou_thr=0.45, pos_iou_thr=0.6, type='Max3DIoUAssigner'),
        ],
        debug=False,
        pos_weight=-1),
    type='VoxelNet',
    voxel_encoder=dict(
        feat_channels=[
            64,
        ], in_channels=4, point_cloud_range=[
            0,
            -46.08,
            -3,
            92.16,
            46.08,
            1,
        ], type='PillarFeatureNet', voxel_size=[
            0.16,
            0.16,
            4,
        ], with_distance=False))
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2), loss_scale='dynamic', optimizer=dict(betas=(
        0.95,
        0.99,
    ), lr=0.0018, type='AdamW', weight_decay=0.01), type='AmpOptimWrapper')
output_shape = [
    576,
    576,
]
param_scheduler = [
    dict(T_max=32.0, begin=0, by_epoch=True, convert_to_iter_based=True, end=32.0, eta_min=0.018, type='CosineAnnealingLR'),
    dict(T_max=48.0, begin=32.0, by_epoch=True, convert_to_iter_based=True, end=80, eta_min=1.8e-07, type='CosineAnnealingLR'),
    dict(T_max=32.0, begin=0, by_epoch=True, convert_to_iter_based=True, end=32.0, eta_min=0.8947368421052632, type='CosineAnnealingMomentum'),
    dict(T_max=48.0, begin=32.0, convert_to_iter_based=True, end=80, eta_min=1, type='CosineAnnealingMomentum'),
]
point_cloud_range = [
    0,
    -46.08,
    -3,
    92.16,
    46.08,
    1,
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/early-fusion/',
        metainfo=dict(classes=[
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(backend_args=None, coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    496,
                    576,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(rot_range=[
                        0,
                        0,
                    ], scale_ratio_range=[
                        1.0,
                        1.0,
                    ], translation_std=[
                        0,
                        0,
                        0,
                    ], type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(point_cloud_range=[
                        0,
                        -46.08,
                        -3,
                        92.16,
                        46.08,
                        1,
                    ], type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/early-fusion/kitti_infos_val.pkl', backend_args=None, metric='bbox', type='KittiMetric')
test_pipeline = [
    dict(backend_args=None, coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
    dict(
        flip=False,
        img_scale=(
            496,
            576,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(rot_range=[
                0,
                0,
            ], scale_ratio_range=[
                1.0,
                1.0,
            ], translation_std=[
                0,
                0,
                0,
            ], type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(point_cloud_range=[
                0,
                -46.08,
                -3,
                92.16,
                46.08,
                1,
            ], type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=2)
train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(pts='training/velodyne_reduced'),
            data_root='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/early-fusion/',
            metainfo=dict(classes=[
                'Car',
            ]),
            modality=dict(use_camera=False, use_lidar=True),
            pipeline=[
                dict(backend_args=None, coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
                dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
                dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
                dict(rot_range=[
                    -0.78539816,
                    0.78539816,
                ], scale_ratio_range=[
                    0.95,
                    1.05,
                ], type='GlobalRotScaleTrans'),
                dict(point_cloud_range=[
                    0,
                    -46.08,
                    -3,
                    92.16,
                    46.08,
                    1,
                ], type='PointsRangeFilter'),
                dict(point_cloud_range=[
                    0,
                    -46.08,
                    -3,
                    92.16,
                    46.08,
                    1,
                ], type='ObjectRangeFilter'),
                dict(type='PointShuffle'),
                dict(keys=[
                    'points',
                    'gt_labels_3d',
                    'gt_bboxes_3d',
                ], type='Pack3DDetInputs'),
            ],
            test_mode=False,
            type='KittiDataset'),
        times=2,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(rot_range=[
        -0.78539816,
        0.78539816,
    ], scale_ratio_range=[
        0.95,
        1.05,
    ], type='GlobalRotScaleTrans'),
    dict(point_cloud_range=[
        0,
        -46.08,
        -3,
        92.16,
        46.08,
        1,
    ], type='PointsRangeFilter'),
    dict(point_cloud_range=[
        0,
        -46.08,
        -3,
        92.16,
        46.08,
        1,
    ], type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(keys=[
        'points',
        'gt_labels_3d',
        'gt_bboxes_3d',
    ], type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/early-fusion/',
        metainfo=dict(classes=[
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(backend_args=None, coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    496,
                    576,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(rot_range=[
                        0,
                        0,
                    ], scale_ratio_range=[
                        1.0,
                        1.0,
                    ], translation_std=[
                        0,
                        0,
                        0,
                    ], type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(point_cloud_range=[
                        0,
                        -46.08,
                        -3,
                        92.16,
                        46.08,
                        1,
                    ], type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/early-fusion/kitti_infos_val.pkl', backend_args=None, metric='bbox', type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer', type='Det3DLocalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.16,
    0.16,
    4,
]
work_dir = './work_dirs/ffnet-vic3d/basemodel/mmdet3d_1.3.0'
z_center_car = -2.66
z_center_cyclist = -0.6
z_center_pedestrian = -0.6
