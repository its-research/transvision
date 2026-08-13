"""Linear-PTF candidate with magnitude-matched feature distillation."""

_base_ = ["./weak_feature_distillation.py"]

model = dict(ptf_mode="linear")
experiment = dict(
    name="dair_improvement_linear_weak_feature_distillation",
    stage="resilient_improvement",
    improvement=dict(
        component="PTF+distillation",
        setting="linear PTF, lambda_feature=0.05, lambda_logit=1.0",
    ),
)
