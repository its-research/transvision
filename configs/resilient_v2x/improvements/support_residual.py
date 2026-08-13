"""Scale-preserving support residual candidate for ResilientV2X."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(support_residual_weight=0.5)

implementation_choices_runtime = dict(
    support_residual_weight=0.5,
    support_residual_formula=(
        "router_fused + weight * (support_weighted_mean - router_fused)"
    ),
)
experiment = dict(
    name="dair_improvement_support_residual",
    stage="resilient_improvement",
    improvement=dict(
        component="dynamic expert router output",
        setting="scale-preserving support residual, weight 0.5",
    ),
)
