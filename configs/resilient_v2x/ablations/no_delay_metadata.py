"""Descriptor ablation: remove branch age/delay metadata."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(use_delay_metadata=False)
experiment = dict(
    name="dair_ablation_no_delay_metadata",
    stage="ablation",
    ablation=dict(component="router_descriptor", setting="no_delay_metadata"),
)
