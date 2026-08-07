"""FFNet-B-V on the sealed complemented manifest and official split."""

_base_ = [
    "../__base__/schedules/cyclic-40e.py",
    "../__base__/default_runtime.py",
    "../__base__/models/v2x_voxelnet.py",
]

data_root = __import__("os").environ.get(
    "RESILIENT_V2X_DATA_ROOT",
    "data/DAIR-V2X/cooperative-vehicle-infrastructure",
)
info_root = __import__("os").environ.get(
    "RESILIENT_V2X_FFNET_INFO_ROOT",
    "work_dirs/ffnet_manifest_infos",
)
train_info = __import__("os").path.join(info_root, "dair_infos_train.pkl")
val_info = __import__("os").path.join(info_root, "dair_infos_val.pkl")

point_cloud_range = [0.0, -46.08, -3.0, 92.16, 46.08, 1.0]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=["Car"])

train_pipeline = [
    dict(
        type="LoadPointsFromFile_w_sensor_view",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        sensor_view="vehicle",
    ),
    dict(
        type="LoadPointsFromFile_w_sensor_view",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        sensor_view="infrastructure",
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="Pack3DDetDAIRInputs",
        keys=[
            "points",
            "infrastructure_points",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=(
            "sample_idx",
            "sample_id",
            "calib",
            "box_mode_3d",
            "box_type_3d",
        ),
    ),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile_w_sensor_view",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        sensor_view="vehicle",
    ),
    dict(
        type="LoadPointsFromFile_w_sensor_view",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        sensor_view="infrastructure",
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="Pack3DDetDAIRInputs",
        keys=[
            "points",
            "infrastructure_points",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=(
            "sample_idx",
            "sample_id",
            "calib",
            "box_mode_3d",
            "box_type_3d",
        ),
    ),
]

dataset = dict(
    type="V2XDataset",
    data_root=data_root,
    data_prefix=dict(pts=""),
    modality=input_modality,
    metainfo=metainfo,
    pcd_limit_range=point_cloud_range,
    box_type_3d="LiDAR",
    boxes_in_lidar=True,
)
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            **dataset,
            ann_file=train_info,
            pipeline=train_pipeline,
            test_mode=False,
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        **dataset,
        ann_file=val_info,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

model = dict(mode="veh_only", legacy_voxel_coordinate_order=True)

# Preserve FFNet's published v0.17.1 basemodel optimization recipe.  The
# generic v1.3 cyclic-40e base uses 1e-4, but the original FFNet config uses
# 1e-3 and cycles to 1e-2 before decaying to 1e-7.
ffnet_base_lr = 1e-3
optim_wrapper = dict(optimizer=dict(lr=ffnet_base_lr))
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=16,
        eta_min=ffnet_base_lr * 10,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=24,
        eta_min=ffnet_base_lr * 1e-4,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=16,
        eta_min=0.85 / 0.95,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=24,
        eta_min=1,
        begin=16,
        end=40,
        convert_to_iter_based=True,
    ),
]
val_evaluator = dict(
    type="ResilientV2XMetric",
    iou_thresholds=(0.5, 0.7),
    max_detections=300,
    point_cloud_range=point_cloud_range,
)
test_evaluator = val_evaluator

find_unused_parameters = True
randomness = dict(seed=20250218, deterministic=False, diff_rank_seed=False)
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=10,
        max_keep_ckpts=4,
        filename_tmpl="epoch_{}.pth",
    )
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Det3DLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

ffnet_reproduction_contract = dict(
    model="official V2XVoxelNet veh_only",
    recipe="official cyclic-40e with RepeatDataset(times=2)",
    source_protocol="current complemented temporal manifest",
    coordinate_convention=(
        "MMDetection3D v0.17 LiDAR bottom center "
        "[x,y,z,width,length,height,-yaw]"
    ),
    train_samples=4823,
    validation_samples=1789,
    global_batch_size=8,
    expected_optimizer_steps=48240,
    initial_learning_rate=ffnet_base_lr,
    peak_learning_rate=ffnet_base_lr * 10,
    final_learning_rate=ffnet_base_lr * 1e-4,
    point_cloud_range=point_cloud_range,
    score_threshold=0.2,
    max_detections=300,
)
