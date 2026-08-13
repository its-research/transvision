"""ResilientV2X candidate with feature distillation rescaled to loss magnitude."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(
    distillation=dict(
        lambda_feature=0.05,
        lambda_logit=1.0,
    ),
)
experiment = dict(
    name="dair_improvement_weak_feature_distillation",
    stage="resilient_improvement",
    improvement=dict(
        component="distillation",
        setting="lambda_feature=0.05, lambda_logit=1.0",
    ),
)
