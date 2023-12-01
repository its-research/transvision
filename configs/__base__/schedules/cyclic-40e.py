# The schedule is usually used by models trained on KITTI dataset
# The learning rate set in the cyclic schedule is the initial learning rate
# rather than the max learning rate. Since the target_ratio is (10, 1e-4),
# the learning rate will change from 0.0018 to 0.018, than go to 0.0018*1e-4
lr = 0.0001
epoch_num = 40
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01), clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=epoch_num * 0.4, eta_min=lr * 10, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=epoch_num * 0.6, eta_min=lr * 1e-4, begin=epoch_num * 0.4, end=epoch_num * 1, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=epoch_num * 0.4, eta_min=0.85 / 0.95, begin=0, end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=epoch_num * 0.6, eta_min=1, begin=epoch_num * 0.4, end=epoch_num * 1, convert_to_iter_based=True)
]

# Runtime settings，training schedule for 40e
# Although the max_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual max epoch
# number could be Nx40
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (6 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=48)
