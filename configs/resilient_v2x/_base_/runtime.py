"""Deterministic single-GPU reference profile.

The paper does not report a memory budget.  Hardware fit must therefore be
measured for the selected dataset history density and training stage.
"""

default_scope = "mmdet3d"

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_best="resilient_v2x/car_3d_ap_r40_0.70",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(
        type="Det3DVisualizationHook",
        draw=False,
        draw_gt=False,
        draw_pred=False,
    ),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
randomness = dict(seed=20250218, deterministic=True, diff_rank_seed=False)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Det3DLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        end=50,
        T_max=50,
        by_epoch=True,
        eta_min=0.000001,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
auto_scale_lr = dict(enable=False, base_batch_size=1)

load_from = None
resume = False

implementation_choices_runtime = dict(
    status="implementation choice; the paper does not report these values",
    optimizer="AdamW",
    learning_rate=0.0001,
    weight_decay=0.01,
    max_epochs=50,
    warmup_iterations=500,
    gradient_clip_norm=35,
    single_gpu_reference_batch_size=1,
)
