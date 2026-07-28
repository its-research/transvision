"""Router ablation: equal weights for the three experts."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(routing_mode="uniform")
experiment = dict(
    name="dair_ablation_router_uniform",
    stage="ablation",
    ablation=dict(component="router", setting="uniform"),
)
