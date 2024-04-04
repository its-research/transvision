_base_ = ['./bevformer_base_occ_w_lightwheel.py']

input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True)
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
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
    val=testset_lightwheel,
    test=testset_lightwheel,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
