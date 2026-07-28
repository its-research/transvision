"""PTF ablation: remove learned temporal residual propagation."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(ptf_mode="none")
experiment = dict(
    name="dair_ablation_ptf_none",
    stage="ablation",
    ablation=dict(component="PTF", setting="none"),
)
