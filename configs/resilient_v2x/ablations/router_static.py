"""Router ablation: learned sample-independent expert weights."""

_base_ = ["../dair_resilient_v2x.py"]

model = dict(routing_mode="static")
experiment = dict(
    name="dair_ablation_router_static",
    stage="ablation",
    ablation=dict(component="router", setting="static"),
)
