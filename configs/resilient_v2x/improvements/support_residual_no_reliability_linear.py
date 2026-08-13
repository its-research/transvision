"""Linear-PTF candidate with support residual and no reliability inputs."""

_base_ = ["./support_residual_no_reliability.py"]

model = dict(ptf_mode="linear")
experiment = dict(
    name="dair_improvement_support_residual_no_reliability_linear",
    stage="resilient_improvement",
    improvement=dict(
        component="PTF+dynamic expert router descriptor+output",
        setting=(
            "linear PTF, no branch-reliability features, scale-preserving "
            "support residual weight 0.5"
        ),
    ),
)
