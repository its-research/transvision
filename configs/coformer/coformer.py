_base_ = ['../__base__/schedules/cyclic-40e.py', '../__base__/default_runtime.py']

dataset_type = 'V2XDataset'
data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/ffnet/'
data_info_train_path = 'dair_infos_train.pkl'
data_info_val_path = 'dair_infos_val.pkl'
work_dir = './work_dirs/mmdet3d_1.3.0/coformer/basemodel'

input_modality = dict(use_lidar=True, use_camera=True)
class_names = ['Car']
metainfo = dict(classes=class_names)
backend_args = None

point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]  # [-75.2, -75.2, -4, 75.2, 75.2, 2]
voxel_size = [0.16, 0.16, 4]
out_size_factor = 8
grid_size = [1440, 1440, 41]
sparse_shape = [1440, 1440, 41]
pc_range = [0, -46.08],  # [-75.2, -75.2]

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(max_num_points=10, point_cloud_range=point_cloud_range, voxel_size=voxel_size, max_voxels=[16000, 40000], voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=4,
        sparse_shape=[1440, 1440, 41],  # [41, 1600, 1408]
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=1,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        train_cfg=dict(
            dataset='Kitti',
            point_cloud_range=point_cloud_range,
            grid_size=[1440, 1440, 41],  # [1600, 1408, 40], [1504, 1504, 40]
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='mmdet.FocalLossCost', gamma=2.0, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(dataset='Kitti', grid_size=[1440, 1440, 41], out_size_factor=8, voxel_size=voxel_size, pc_range=[0, -46.08], nms_type=None),
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[0, -46.08],  # [-75.2, -75.2]
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # [0, -46.08, -3, 92.16, 46.08, 1][-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size,
            code_size=8),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)))

train_pipeline = [
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='vehicle'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='GlobalRotScaleTrans', scale_ratio_range=[0.9, 1.1], rot_range=[-0.78539816, 0.78539816], translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetDAIRInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar', 'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx', 'lidar_path', 'img_path',
            'transformation_3d_flow', 'pcd_rotation', 'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix', 'lidar_aug_matrix'
        ])
]

test_pipeline = [
    dict(type='LoadPointsFromFile_w_sensor_view', coord_type='LIDAR', load_dim=4, use_dim=4, sensor_view='vehicle'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetDAIRInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar', 'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx', 'lidar_path', 'img_path',
            'num_pts_feats', 'num_views'
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_info_train_path,
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=dict(pts=''),
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_info_val_path,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=dict(pts=''),
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='KittiMetric', ann_file=data_root + data_info_val_path, metric='bbox', pcd_limit_range=point_cloud_range, backend_args=backend_args),
    # dict(
    #     type='DAIRV2XMetric',
    #     ann_file=data_root + data_info_val_path,
    #     veh_config_path='configs/ffnet/config_basemodel_fusion.py',
    #     work_dir=work_dir,
    #     split_data_path='data/split_datas/cooperative-split-data.json',
    #     model='feature_fusion',
    #     input='data/DAIR-V2X/cooperative-vehicle-infrastructure',
    #     test_mode=None,
    #     val_data_path=None,
    #     pcd_limit_range=point_cloud_range,
    #     backend_args=backend_args)
]
test_evaluator = val_evaluator

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# learning rate
lr = 0.0001
epoch_num = 20
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=8, eta_min=lr * 10, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=12, eta_min=lr * 1e-4, begin=8, end=epoch_num, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=8, eta_min=0.85 / 0.95, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=12, eta_min=1, begin=8, end=epoch_num, by_epoch=True, convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=10)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01), clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
log_processor = dict(window_size=50)

default_hooks = dict(logger=dict(type='LoggerHook', interval=50), checkpoint=dict(type='CheckpointHook', interval=5))
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]
