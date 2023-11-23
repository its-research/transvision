point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]
voxel_size = [0.16, 0.16, 4]
l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
h = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
flownet_test_mode = 'FlowPred'  # {'FlowPred', 'OriginFeat', 'Async'}
data_root = ('./')
pretrained_basemodel = 'models/ffnet_basemodel_1.2.0.pth'
z_center_pedestrian = -0.6
z_center_cyclist = -0.6
z_center_car = -2.66

model = dict(
    type='FeatureFlowNet',
    voxel_layer=dict(max_num_points=100, point_cloud_range=point_cloud_range, voxel_size=voxel_size, max_voxels=(40000, 40000)),
    data_preprocessor=dict(
        type='Det3DDataDAIRPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=100,  # max_points_per_voxel 100
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(40000, 40000))),
    voxel_encoder=dict(type='PillarFeatureNet', in_channels=4, feat_channels=[64], with_distance=False, voxel_size=voxel_size, point_cloud_range=point_cloud_range),
    middle_encoder=dict(type='PointPillarsScatter', in_channels=64, output_shape=[h, l]),
    backbone=dict(type='SECOND', in_channels=64, layer_nums=[3, 5, 5], layer_strides=[2, 2, 2], out_channels=[64, 128, 256]),
    neck=dict(type='SECONDFPN', in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[point_cloud_range[0], point_cloud_range[1], z_center_pedestrian, point_cloud_range[3], point_cloud_range[4], z_center_pedestrian],
                    [point_cloud_range[0], point_cloud_range[1], z_center_cyclist, point_cloud_range[3], point_cloud_range[4], z_center_cyclist],
                    [point_cloud_range[0], point_cloud_range[1], z_center_car, point_cloud_range[3], point_cloud_range[4], z_center_car]],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False,
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.1111111111111111, loss_weight=2.0),
        loss_dir=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
    ),
    init_cfg=dict(type='Pretrained', checkpoint=pretrained_basemodel),
    train_cfg=dict(
        assigner=[
            dict(type='Max3DIoUAssigner', iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'), pos_iou_thr=0.5, neg_iou_thr=0.35, min_pos_iou=0.35, ignore_iof_thr=-1),
            dict(type='Max3DIoUAssigner', iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'), pos_iou_thr=0.5, neg_iou_thr=0.35, min_pos_iou=0.35, ignore_iof_thr=-1),
            dict(type='Max3DIoUAssigner', iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'), pos_iou_thr=0.6, neg_iou_thr=0.45, min_pos_iou=0.45, ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False,
        pretrained_model=pretrained_basemodel,
        data_root=data_root,
    ),
    test_cfg=dict(
        use_rotate_nms=False,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.2,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=300,
        test_mode=flownet_test_mode,
        pretrained_model='',
    ),
)
