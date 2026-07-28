"""Training ablation: train the resilient student without a clean teacher."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(teacher=None, teacher_checkpoint=None, distillation=None)
train_dataloader = dict(dataset=dict(include_clean_teacher=False))
experiment = dict(
    name="dair_ablation_no_distillation",
    stage="ablation",
    ablation=dict(component="distillation", setting="none"),
)
