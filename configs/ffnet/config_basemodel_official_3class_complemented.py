"""Strict FFNet v0.17.1 three-class fusion baseline on complemented data."""

_base_ = ["./config_basemodel_veh_only_complemented.py"]

point_cloud_range = [0.0, -46.08, -3.0, 92.16, 46.08, 1.0]
class_names = ("Pedestrian", "Cyclist", "Car")
metainfo = dict(classes=class_names)
z_center_pedestrian = -0.6
z_center_cyclist = -0.6
z_center_car = -2.66

model = dict(
    mode="fusion",
    legacy_voxel_coordinate_order=True,
    data_preprocessor=dict(infrastructure_intensity_scale=1.0),
    voxel_encoder=dict(
        _delete_=True,
        type="FFNetLegacyPillarFeatureNet",
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=[0.16, 0.16, 4.0],
        point_cloud_range=point_cloud_range,
        legacy=True,
    ),
    bbox_head=dict(
        _delete_=True,
        type="Anchor3DHead",
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        dir_offset=0.0,
        dir_limit_offset=1.0,
        assign_per_class=False,
        anchor_generator=dict(
            type="Anchor3DRangeGenerator",
            ranges=[
                [0.0, -46.08, z_center_pedestrian, 92.16, 46.08, z_center_pedestrian],
                [0.0, -46.08, z_center_cyclist, 92.16, 46.08, z_center_cyclist],
                [0.0, -46.08, z_center_car, 92.16, 46.08, z_center_car],
            ],
            sizes=[
                [0.6, 0.8, 1.73],
                [0.6, 1.76, 1.73],
                [1.6, 3.9, 1.56],
            ],
            rotations=[0.0, 1.57],
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
            beta=1.0 / 9.0,
            loss_weight=2.0,
        ),
        loss_dir=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.2,
        ),
    ),
    train_cfg=dict(
        _delete_=True,
        assigner=[
            dict(
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1,
            ),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=False,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.2,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=300,
    ),
)

train_dataloader = dict(
    dataset=dict(dataset=dict(metainfo=metainfo)),
)
val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = dict(dataset=dict(metainfo=metainfo))

val_evaluator = dict(
    type="ResilientV2XMetric",
    iou_thresholds=(0.5, 0.7),
    max_detections=300,
    car_label_index=2,
    point_cloud_range=point_cloud_range,
)
test_evaluator = val_evaluator
test_evaluator["prediction_output"] = __import__("os").getenv(
    "RESILIENT_V2X_FFNET_PREDICTION_OUTPUT"
)

ffnet_reproduction_contract = dict(
    model="official V2XVoxelNet fusion three-class head",
    inference_mode="fusion",
    fusion_calibration=(
        "ego_lidar_from_rsu_lidar derived from sealed per-sample poses"
    ),
    classes=class_names,
    car_label_index=2,
    official_checkpoint_head_channels=dict(cls=18, reg=42, direction=12),
    recipe="official cyclic-40e with RepeatDataset(times=2)",
    source_protocol="current complemented temporal manifest",
    train_samples=4823,
    validation_samples=1789,
    training_world_size=1,
    global_batch_size=2,
    expected_optimizer_steps=192920,
    point_cloud_range=point_cloud_range,
    score_threshold=0.2,
    max_detections=300,
    prepared_infrastructure_intensity_scale=1.0,
    voxel_coordinate_order="z-y-x (MMDetection3D v0.17)",
)
