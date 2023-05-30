dataset_type = 'KittiDataset'
data_root = '../../../data/DAIR-V2X/single-vehicle-side/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
z_center_pedestrian = -0.6
z_center_cyclist = -0.6
z_center_car = -1.78

mean = [103.53, 116.28, 123.675]
std = [1.0, 1.0, 1.0]
img_scale = (960, 540)
to_rgb = False
img_norm_cfg = dict(mean=mean, std=std, to_rgb=to_rgb)
input_modality = dict(use_lidar=True, use_camera=True)

lr = 0.003
optimizer = dict(type='AdamW', lr=0.003, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=1000, warmup_ratio=0.1, min_lr_ratio=1e-05)
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class'
load_from = (
    'https://download.openmmlab.com/mmdetection3d/pretrain_models/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth')
resume_from = None
workflow = [('train', 1)]

model = dict(
    type='DynamicMVXFasterRCNN',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
    ),
    img_neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    pts_voxel_layer=dict(max_num_points=-1, point_cloud_range=point_cloud_range, voxel_size=voxel_size, max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        fusion_layer=dict(
            type='PointFusion',
            img_channels=256,
            pts_channels=64,
            mid_channels=128,
            out_channels=128,
            img_levels=[0, 1, 2, 3, 4],
            align_corners=False,
            activate_out=True,
            fuse_out=False,
        ),
    ),
    pts_middle_encoder=dict(type='SparseEncoder', in_channels=128, sparse_shape=[41, 1600, 1408], order=('conv', 'norm', 'act')),
    pts_backbone=dict(type='SECOND', in_channels=256, layer_nums=[5, 5], layer_strides=[1, 2], out_channels=[128, 256]),
    pts_neck=dict(type='SECONDFPN', in_channels=[128, 256], upsample_strides=[1, 2], out_channels=[256, 256]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40, -0.6, 70.4, 40, -0.6], [0, -40, -0.6, 70.4, 40, -0.6], [0, -40, -1.78, 70.4, 40, -1.78]],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False,
        ),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=2.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
    ),
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1,
                ),
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1,
                ),
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1,
                ),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        )),
    test_cfg=dict(pts=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50,
    )),
)
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='Resize', img_scale=[(480, 270), (1920, 1080)], multiscale_mode='range', keep_ratio=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2],
    ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(960, 540),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='Normalize', mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['points', 'img']),
        ],
    ),
]
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points', 'img']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root=data_root,
            ann_file=data_root + '/kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=[
                dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
                dict(type='Resize', img_scale=[(480, 270), (1920, 1080)], multiscale_mode='range', keep_ratio=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0.2, 0.2, 0.2],
                ),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
                dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
                dict(type='PointShuffle'),
                dict(type='Normalize', mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle3D', class_names=class_names),
                dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']),
            ],
            modality=dict(use_lidar=True, use_camera=True),
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
        ),
    ),
    val=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(960, 540),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='Resize', multiscale_mode='value', keep_ratio=True),
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0],
                    ),
                    dict(type='RandomFlip3D'),
                    dict(type='Normalize', mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
                    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
                    dict(type='Collect3D', keys=['points', 'img']),
                ],
            ),
        ],
        modality=dict(use_lidar=True, use_camera=True),
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
    test=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file=data_root + '/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(960, 540),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='Resize', multiscale_mode='value', keep_ratio=True),
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0],
                    ),
                    dict(type='RandomFlip3D'),
                    dict(type='Normalize', mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
                    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
                    dict(type='Collect3D', keys=['points', 'img']),
                ],
            ),
        ],
        modality=dict(use_lidar=True, use_camera=True),
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
)
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
        dict(type='LoadImageFromFile'),
        dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
        dict(type='Collect3D', keys=['points', 'img']),
    ],
)
gpu_ids = range(0, 1)