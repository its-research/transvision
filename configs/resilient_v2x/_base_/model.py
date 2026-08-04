"""ResilientV2X architecture shared by teacher, student, and ablations.

The paper fixes the four causal branches, 256-channel BEV features,
PointPillars, ResNet-50 + LSS, PTF, and the three-expert router.  Numeric
choices not reported by the paper are recorded in ``implementation_choices``.
"""

point_cloud_range = [0.0, -40.0, -3.0, 80.0, 40.0, 1.0]
# Keep the detector head at 0.625 m rather than the previous 0.8 m.  The old
# 0.4 m pillars followed by the fixed x2 SECOND stride left too few anchors
# for the strict Car assigner and made high-IoU recall collapse.  0.3125 m
# divides the fixed 80 m ranges exactly and produces a PTF-compatible 128x128
# feature grid.
voxel_size = [0.3125, 0.3125, 4.0]
group_norm = dict(type="GN", num_groups=32, eps=0.001)
resnet50_checkpoint = __import__("os").getenv(
    "RESILIENT_V2X_RESNET50_CHECKPOINT",
    "https://download.pytorch.org/models/resnet50-0676ba61.pth",
)
bev_grid = dict(
    x_min=0.0,
    y_min=-40.0,
    resolution=0.625,
    height=128,
    width=128,
)

implementation_choices_model = dict(
    status="implementation choice; the paper does not report these values",
    point_cloud_range=point_cloud_range,
    voxel_size=voxel_size,
    bev_resolution_m=0.625,
    image_size=(256, 704),
    camera_depth_bins=(1.0, 81.0, 1.0),
    anchor_size_lwh=(4.35, 1.91, 1.59),
    anchor_z_bottom=-1.8,
    car_assigner_iou=(0.5, 0.35, 0.35),
    lidar_normalization="GroupNorm; independent of sparse payload batch composition",
    camera_normalization="frozen ImageNet BatchNorm plus GroupNorm LSS neck",
    image_backbone_checkpoint=resnet50_checkpoint,
    distillation_temperature=2.0,
    lambda_feature=1.0,
    lambda_logit=1.0,
)

paper_model_contract = dict(
    feature_channels=256,
    branch_order=("L_E", "L_R", "C_E", "C_R"),
    lidar_encoder="shared PointPillars",
    camera_encoder="shared ResNet-50 + LSS",
    experts=("lidar", "camera", "synergy"),
    ptf_mode="nonlinear",
    routing_mode="dynamic",
)

lidar_encoder = dict(
    type="SharedPointPillarsBEVEncoder",
    voxelize_cfg=dict(
        max_num_points=32,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000),
    ),
    voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=group_norm,
    ),
    middle_encoder=dict(
        type="PointPillarsScatter",
        in_channels=64,
        output_shape=(256, 256),
    ),
    backbone=dict(
        type="SECOND",
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
        norm_cfg=group_norm,
    ),
    neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[64, 96, 96],
        norm_cfg=group_norm,
    ),
    output_height=128,
    output_width=128,
)

camera_encoder = dict(
    type="SharedResNetLSSBEVEncoder",
    image_backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint=resnet50_checkpoint),
    ),
    image_neck=dict(
        type="GeneralizedLSSFPN",
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=group_norm,
        act_cfg=dict(type="ReLU", inplace=True),
        upsample_cfg=dict(mode="bilinear", align_corners=False),
    ),
    view_transform=dict(
        type="LSSTransform",
        in_channels=256,
        out_channels=256,
        image_size=(256, 704),
        feature_size=(32, 88),
        xbound=(0.0, 80.0, 0.625),
        ybound=(-40.0, 40.0, 0.625),
        zbound=(-5.0, 3.0, 8.0),
        dbound=(1.0, 81.0, 1.0),
        downsample=1,
    ),
    output_height=128,
    output_width=128,
    view_transform_output_order="xy",
)

bbox_head = dict(
    type="Anchor3DHead",
    num_classes=1,
    in_channels=256,
    feat_channels=256,
    use_direction_classifier=True,
    anchor_generator=dict(
        type="Anchor3DRangeGenerator",
        ranges=[[0.0, -40.0, -1.8, 80.0, 40.0, -1.8]],
        sizes=[[4.35, 1.91, 1.59]],
        rotations=[0.0, 1.5707963267948966],
        reshape_out=False,
    ),
    diff_rad_by_sin=True,
    bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
    loss_cls=dict(
        type="mmdet.FocalLoss",
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0,
    ),
    loss_bbox=dict(
        type="mmdet.SmoothL1Loss",
        beta=0.1111111111111111,
        loss_weight=2.0,
    ),
    loss_dir=dict(
        type="mmdet.CrossEntropyLoss",
        use_sigmoid=False,
        loss_weight=0.2,
    ),
    train_cfg=dict(
        assigner=[
            dict(
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            )
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=2000,
        max_num=100,
    ),
)

data_preprocessor = dict(
    type="ResilientV2XDataPreprocessor",
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
    non_blocking=True,
)

# This clean architecture is embedded once as the frozen teacher.  It contains
# no teacher of its own, which prevents recursion and makes strict checkpoint
# loading possible.
clean_teacher_model = dict(
    type="ResilientV2XNet",
    grid_spec=bev_grid,
    lidar_encoder=lidar_encoder,
    camera_encoder=camera_encoder,
    bbox_head=bbox_head,
    data_preprocessor=data_preprocessor,
    ptf_mode="nonlinear",
    routing_mode="dynamic",
    use_reliability=True,
    use_delay_metadata=True,
    delta_t_ms=100,
)

model = dict(
    **clean_teacher_model,
    teacher=clean_teacher_model,
    teacher_checkpoint=__import__("os").getenv(
        "RESILIENT_V2X_TEACHER_CHECKPOINT",
        "work_dirs/resilient_v2x_dair_clean_teacher/best_teacher.pth",
    ),
    distillation=dict(
        temperature=2.0,
        lambda_feature=1.0,
        lambda_logit=1.0,
        head_type="bernoulli",
        logit_path=(0, 0),
    ),
)
