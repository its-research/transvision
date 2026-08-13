"""No-distillation candidate with peak learning rate reduced to 3e-4."""

_base_ = ["../ablations/no_distillation.py"]

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=20,
        eta_min=0.0003,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=20,
        end=50,
        T_max=30,
        by_epoch=True,
        eta_min=0.000001,
        convert_to_iter_based=True,
    ),
]
implementation_choices_runtime = dict(peak_learning_rate=0.0003)
experiment = dict(
    name="dair_improvement_no_distillation_peak_lr_3e4",
    stage="resilient_improvement",
    improvement=dict(
        component="optimizer schedule",
        setting="no distillation, peak learning rate 0.0003",
    ),
)
