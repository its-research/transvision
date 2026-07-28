"""Descriptor ablation: remove branch reliability features."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(use_reliability=False)
experiment = dict(
    name="dair_ablation_no_reliability",
    stage="ablation",
    ablation=dict(component="router_descriptor", setting="no_reliability"),
)
