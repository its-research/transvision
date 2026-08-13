"""Linear-PTF candidate based exactly on the no-reliability ablation."""

_base_ = ["../ablations/no_reliability.py"]

model = dict(ptf_mode="linear")
experiment = dict(
    name="dair_improvement_no_reliability_linear",
    stage="resilient_improvement",
    improvement=dict(
        component="PTF+router descriptor",
        setting="linear PTF, no branch-reliability features",
    ),
)
