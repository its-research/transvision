"""ResilientV2X candidate combining linear PTF with no distillation."""

_base_ = ["../ablations/no_distillation.py"]

model = dict(ptf_mode="linear")
experiment = dict(
    name="dair_improvement_linear_no_distillation",
    stage="resilient_improvement",
    improvement=dict(
        component="PTF+distillation",
        setting="linear PTF, no distillation",
    ),
)
