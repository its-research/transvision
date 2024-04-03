_base_ = ['./bevformer_base_occ.py']

dataset_type = 'NuSceneOcc'
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For OpenOcc v2 we have 17 classes (including `free`)
occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True)

_dim_ = 256

bev_h_ = 200
bev_w_ = 200

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(type='CustomCollect3D', keys=['img', 'voxel_semantics', 'voxel_flow'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # dict(type='LoadOccGTFromFile'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[dict(type='DefaultFormatBundle3D', class_names=occ_class_names, with_label=False),
                    dict(type='CustomCollect3D', keys=['img'])])
]

nusc_dataset_type = 'NuSceneOcc'
nusc_data_root = 'data/nuscenes/'
light_dataset_type = 'NuSceneOcc'
light_data_root = 'data/lightwheelocc/'
file_client_args = dict(backend='disk')

trainset_nusc = dict(
    type=nusc_dataset_type,
    data_root=nusc_data_root,
    ann_file=nusc_data_root + 'nuscenes_infos_train_occ.pkl',
    pipeline=train_pipeline,
    classes=occ_class_names,
    modality=input_modality,
    test_mode=False,
    use_valid_flag=True,
    filter_empty_gt=False,
    # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    box_type_3d='LiDAR')

trainset_lightwheel = dict(
    type=light_dataset_type,
    data_root=light_data_root,
    ann_file=light_data_root + 'lightwheel_occ_infos_train.pkl',
    pipeline=train_pipeline,
    classes=occ_class_names,
    modality=input_modality,
    test_mode=False,
    use_valid_flag=True,
    filter_empty_gt=False,
    box_type_3d='LiDAR')

testset_nusc = dict(
    type=nusc_dataset_type,
    data_root=nusc_data_root,
    ann_file=nusc_data_root + 'nuscenes_infos_test_occ.pkl',
    pipeline=test_pipeline,
    classes=occ_class_names,
    modality=input_modality,
    filter_empty_gt=False)

valset_nusc = dict(
    type=nusc_dataset_type,
    data_root=nusc_data_root,
    ann_file=nusc_data_root + 'nuscenes_infos_val_occ.pkl',
    pipeline=test_pipeline,
    classes=occ_class_names,
    modality=input_modality,
    filter_empty_gt=False)

testset_lightwheel = dict(
    type=light_dataset_type,
    data_root=light_data_root,
    ann_file=light_data_root + 'lightwheel_occ_infos_test.pkl',
    pipeline=test_pipeline,
    classes=occ_class_names,
    modality=input_modality,
    filter_empty_gt=False,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='ConcatDataset',
        datasets=[trainset_nusc, trainset_lightwheel],
    ),
    val=testset_lightwheel,
    test=testset_lightwheel,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

load_from = 'epoch_24.pth'
