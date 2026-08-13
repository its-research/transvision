"""Linear-PTF ResilientV2X candidate with a scale-preserving residual."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(
    ptf_mode="linear",
    support_residual_weight=0.5,
)

implementation_choices_runtime = dict(
    support_residual_weight=0.5,
    support_residual_formula=(
        "router_fused + weight * (support_weighted_mean - router_fused)"
    ),
)
experiment = dict(
    name="dair_improvement_support_residual_linear",
    stage="resilient_improvement",
    improvement=dict(
        component="PTF+dynamic expert router output",
        setting="linear PTF, scale-preserving support residual weight 0.5",
    ),
)
