"""PTF ablation: replace nonlinear PTF by its linear variant."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(ptf_mode="linear")
experiment = dict(
    name="dair_ablation_ptf_linear",
    stage="ablation",
    ablation=dict(component="PTF", setting="linear"),
)
