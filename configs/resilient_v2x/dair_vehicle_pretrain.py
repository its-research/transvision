"""Stage 0: FFNet-resolution pretraining on current vehicle LiDAR scans.

The pretraining grid and Car anchors intentionally match the repository's
FFNet-B-V reference. Learnable tensor shapes remain identical to the paper
model, so the resulting LiDAR encoder and bbox head state is a strict
parameter-compatible subset of the clean teacher.
"""

_copy = __import__("copy")

_components = __import__(
    "configs.resilient_v2x._base_.model",
    fromlist=("bbox_head", "data_preprocessor", "lidar_encoder"),
)
bbox_head = _copy.deepcopy(_components.bbox_head)
data_preprocessor = _components.data_preprocessor
lidar_encoder = _copy.deepcopy(_components.lidar_encoder)
del _components

vehicle_point_cloud_range = [0.0, -46.08, -3.0, 92.16, 46.08, 1.0]
vehicle_voxel_size = [0.16, 0.16, 4.0]

# FFNet-B-V uses 576x576 pillars and a stride-2 detection map (288x288,
# 0.32 m/cell). The previous pretrain used a 128x128 map (0.625 m/cell),
# which removed roughly three quarters of the spatial anchor locations.
lidar_encoder["voxelize_cfg"].update(
    max_num_points=100,
    point_cloud_range=vehicle_point_cloud_range,
    voxel_size=vehicle_voxel_size,
    max_voxels=(40000, 40000),
)
lidar_encoder["voxel_encoder"].update(
    voxel_size=vehicle_voxel_size,
    point_cloud_range=vehicle_point_cloud_range,
)
lidar_encoder["middle_encoder"]["output_shape"] = (576, 576)
lidar_encoder["output_height"] = 288
lidar_encoder["output_width"] = 288
lidar_encoder["output_projection"] = None
lidar_encoder["output_channels"] = 384

# Vehicle validation uses the original FFNet 384-channel neck-to-head path.
# The clean teacher keeps the same transferable neck/head weights and adds
# 384-to-256-to-384 adapters around the paper's 256-channel fusion.
# Match working FFNet-B-V anchors (z_center_car=-2.66 in v2x_voxelnet.py).
# Using -1.78 starved MaxIoU positives and left ep10 with ~1 prediction.
bbox_head["anchor_generator"] = dict(
    type="AlignedAnchor3DRangeGenerator",
    ranges=[
        [
            vehicle_point_cloud_range[0],
            vehicle_point_cloud_range[1],
            -2.66,
            vehicle_point_cloud_range[3],
            vehicle_point_cloud_range[4],
            -2.66,
        ]
    ],
    sizes=[[3.9, 1.6, 1.56]],
    rotations=[0.0, 1.5707963267948966],
    reshape_out=False,
)
bbox_head["train_cfg"]["assigner"][0].update(
    pos_iou_thr=0.6,
    neg_iou_thr=0.45,
    min_pos_iou=0.45,
)
bbox_head["assign_per_class"] = True
bbox_head["test_cfg"].update(
    use_rotate_nms=False,
    score_thr=0.2,
    nms_pre=1000,
    max_num=300,
)

_base_ = [
    "./_base_/dataset.py",
    "./_base_/runtime.py",
]

model = dict(
    type="VehiclePointPillarsPretrainNet",
    lidar_encoder=lidar_encoder,
    bbox_head=bbox_head,
    data_preprocessor=data_preprocessor,
)

train_dataloader = dict(
    dataset=dict(
        load_camera=False,
        point_cloud_range=vehicle_point_cloud_range,
        include_clean_teacher=False,
        transport_overlay_path=None,
        transport_overlay_sha256=None,
        fault_overlay_path=None,
        fault_overlay_sha256=None,
    )
)
val_dataloader = dict(
    dataset=dict(
        load_camera=False,
        point_cloud_range=vehicle_point_cloud_range,
    )
)
test_dataloader = dict(
    dataset=dict(
        load_camera=False,
        point_cloud_range=vehicle_point_cloud_range,
    )
)

val_evaluator = dict(
    point_cloud_range=vehicle_point_cloud_range,
    max_detections=300,
)
test_evaluator = val_evaluator

# FFNet runs 40 epochs over RepeatDataset(times=2). This loader does not repeat
# the dataset, so 80 epochs preserve the same number of data passes and place
# the cyclic schedule boundary after the same amount of training data.
# Mirror FFNet's published cyclic recipe: climb 1e-3 -> 1e-2, then decay.
ffnet_base_lr = 0.001
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=32,
        eta_min=ffnet_base_lr * 10,
        by_epoch=True,
        begin=0,
        end=32,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=48,
        eta_min=ffnet_base_lr * 1e-4,
        by_epoch=True,
        begin=32,
        end=80,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=32,
        eta_min=0.85 / 0.95,
        by_epoch=True,
        begin=0,
        end=32,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=48,
        eta_min=1,
        begin=32,
        end=80,
        convert_to_iter_based=True,
    ),
]

# Validation every ten epochs follows the FFNet protocol and avoids spending
# most of the run evaluating 1,789 frames.
train_cfg = dict(max_epochs=80, val_interval=10)

# FFNet-B-V uses AdamW lr=1e-3. The shared runtime profile defaults to 1e-4 for
# the multimodal teacher/student; without this override the vehicle pretrain
# inherits 1e-4 and stays near-zero AP while the official FFNet recipe learns.
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.001,
        betas=(0.95, 0.99),
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

default_hooks = dict(
    checkpoint=dict(
        filename_tmpl="vehicle_epoch_{}.pth",
    )
)

vehicle_pretrain_contract = dict(
    source="current vehicle LiDAR only",
    agent="ego",
    horizon=0,
    reference="FFNet-B-V repository reimplementation",
    point_cloud_range=vehicle_point_cloud_range,
    voxel_size=vehicle_voxel_size,
    detection_grid=(288, 288),
    detection_resolution_m=0.32,
    expected_global_batch_size=8,
    expected_dataset_passes=80,
    expected_optimizer_steps_80e=48240,
    shared_checkpoint_prefixes=("lidar_encoder.", "bbox_head."),
    transfer="all learnable tensors are shape-compatible with clean teacher",
    checkpoint_selection="exact final epoch for downstream teacher transfer",
)

experiment = dict(
    name="resilient_v2x_dair_vehicle_pretrain",
    stage="vehicle_pretrain",
    protocol="current vehicle LiDAR only; zero latency and no injected faults",
    required_external_inputs=("RESILIENT_V2X_SPLIT_SHA256",),
    transfer_contract=("lidar_encoder", "bbox_head"),
)

del _copy
