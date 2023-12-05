# 对于基于体素化的检测器如SECOND, PointPillars 及 CenterPoint, 点云范围(point cloud range)和体素大小(voxel size)应该根据您的数据集做调整。
# 理论上, `voxel_size` 和 `point_cloud_range` 的设置是相关联的。设置较小的 `voxel_size` 将增加体素数以及相应的内存消耗。
# 此外, 需要注意以下问题：
# 如果将 `point_cloud_range` 和 `voxel_size` 分别设置成 `[0, -40, -3, 70.4, 40, 1]` 和 `[0.05, 0.05, 0.1]`,
# 那么中间特征图的形状应该为 `[(1-(-3))/0.1+1, (40-(-40))/0.05, (70.4-0)/0.05]=[41, 1600, 1408]`
# 更改 `point_cloud_range` 时, 请记得依据 `voxel_size` 更改 `middle_encoder` 里中间特征图的形状。
# 关于 `anchor_range` 的设置, 一般需要根据数据集做调整。需要注意的是, `z` 值需要根据点云的位置做相应调整,
# 具体请参考此 [issue](https://github.com/open-mmlab/mmdetection3d/issues/986)。
# 关于 `anchor_size` 的设置, 通常需要计算整个训练集中目标的长、宽、高的平均值作为 `anchor_size`, 以获得最好的结果。

# For voxel based detectors such as SECOND, PV-RCNN and CenterPoint, the point cloud range and voxel size should follow:
# Point cloud range along z-axis / voxel_size is 40
# Point cloud range along x&y-axis / voxel_size is the multiple of 16.

point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]
voxel_size = [0.16, 0.16, 0.1]

out_size_factor = 8
sparse_shape = [576, 576, 41]
# grid_size = [576, 576, 41]
grid_size = [576, 576, 41]  # TODO: why 40 or 41?
pc_range = [0, -46.08]

model = dict(
    type='CoFormerNet',
    mode='fusion',  # veh_only, inf_only, fusion
    data_preprocessor=dict(
        type='Det3DDataDAIRPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(max_num_points=10, point_cloud_range=point_cloud_range, voxel_size=voxel_size, max_voxels=[16000, 40000], voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=4,
        sparse_shape=sparse_shape,
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
        num_proposals=300,
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
            grid_size=grid_size,
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
        test_cfg=dict(dataset='Kitti', grid_size=grid_size, out_size_factor=8, voxel_size=voxel_size, pc_range=[0, -46.08], nms_type=None),
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[0, -46.08],  # pc_range=point_cloud_range[:2],
            post_center_range=[0, -50, -5, 100, 50, 3],
            score_threshold=0.2,
            out_size_factor=8,
            voxel_size=voxel_size,
            code_size=8),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)))
