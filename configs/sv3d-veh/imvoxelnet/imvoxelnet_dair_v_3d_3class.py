_base_ = ["../../mmdet_schedule_1x.py", "../../default_runtime.py"]

work_dir = "./work_dirs/imvoxelnet_veh"

dataset_type = "KittiDataset"
data_root = "./data/DAIR-V2X/single-vehicle-side/"
class_names = ["Pedestrian", "Cyclist", "Car"]
input_modality = dict(use_lidar=False, use_camera=True)
metainfo = dict(classes=class_names)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size = [0.32, 0.32, 0.32]
n_voxels = [int((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i]) for i in range(3)]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1280, 384)
img_resize_scale = [(1173, 352), (1387, 416)]

ped_center = -0.6
cyc_center = -0.6
car_center = -1.78

anchor_range_ped = [
    point_cloud_range[0],
    point_cloud_range[1],
    ped_center,
    point_cloud_range[3] - voxel_size[0],
    point_cloud_range[4] - voxel_size[1],
    ped_center,
]
anchor_range_cyc = [
    point_cloud_range[0],
    point_cloud_range[1],
    cyc_center,
    point_cloud_range[3] - voxel_size[0],
    point_cloud_range[4] - voxel_size[1],
    cyc_center,
]
anchor_range_car = [
    point_cloud_range[0],
    point_cloud_range[1],
    car_center,
    point_cloud_range[3] - voxel_size[0],
    point_cloud_range[4] - voxel_size[1],
    car_center,
]

anchor_size_pred = [0.6, 0.8, 1.73]
anchor_size_cyc = [0.6, 1.76, 1.73]
anchor_size_car = [1.6, 3.9, 1.56]


model = dict(
    type="ImVoxelNet",
    data_preprocessor=dict(type="Det3DDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True, pad_size_divisor=32),
    backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        style="pytorch",
    ),
    neck=dict(type="mmdet.FPN", in_channels=[256, 512, 1024, 2048], out_channels=64, num_outs=4),
    neck_3d=dict(type="OutdoorImVoxelNeck", in_channels=64, out_channels=256),
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=len(class_names),
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="AlignedAnchor3DRangeGenerator",
            ranges=[anchor_range_ped, anchor_range_cyc, anchor_range_car],
            sizes=[anchor_size_pred, anchor_size_cyc, anchor_size_car],
            rotations=[0, 1.57],
            reshape_out=False,
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        loss_cls=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    n_voxels=n_voxels,
    coord_type="LIDAR",
    prior_generator=dict(type="AlignedAnchor3DRangeGenerator", ranges=[[0, -39.68, -3.08, 69.12, 39.68, 0.76]], rotations=[0.0]),
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(  # for Cyclist
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(  # for Car
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
    test_cfg=dict(use_rotate_nms=True, nms_across_levels=False, nms_thr=0.01, score_thr=0.01, min_bbox_size=0, nms_pre=300, max_num=100),
)


backend_args = None

train_pipeline = [
    dict(type="LoadAnnotations3D", backend_args=backend_args),
    dict(type="LoadImageFromFileMono3D", backend_args=backend_args),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="RandomResize", scale=[(1173, 352), (1387, 416)], keep_ratio=True),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="Pack3DDetInputs", keys=["img", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(type="LoadImageFromFileMono3D", backend_args=backend_args),
    dict(type="Resize", scale=(1280, 384), keep_ratio=True),
    dict(type="Pack3DDetInputs", keys=["img"]),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file="kitti_infos_train.pkl",
            data_prefix=dict(img="training/image_2"),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d="LiDAR",
            backend_args=backend_args,
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="kitti_infos_val.pkl",
        data_prefix=dict(img="training/image_2"),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="KittiMetric", ann_file=data_root + "kitti_infos_val.pkl", metric="bbox", backend_args=backend_args)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(_delete_=True, type="AdamW", lr=0.0001, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35.0, norm_type=2),
)
param_scheduler = [dict(type="MultiStepLR", begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)]

# hooks
default_hooks = dict(checkpoint=dict(type="CheckpointHook", max_keep_ckpts=1))

# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")
