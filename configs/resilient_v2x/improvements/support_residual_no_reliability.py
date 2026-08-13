"""No-reliability candidate based exactly on the support residual variant."""

_base_ = ["./support_residual.py"]

model = dict(use_reliability=False)
experiment = dict(
    name="dair_improvement_support_residual_no_reliability",
    stage="resilient_improvement",
    improvement=dict(
        component="dynamic expert router descriptor+output",
        setting=(
            "no branch-reliability features, scale-preserving support residual "
            "weight 0.5"
        ),
    ),
)
