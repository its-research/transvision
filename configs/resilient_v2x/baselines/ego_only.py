"""Controlled non-cooperative ego LiDAR+camera baseline on DAIR-V2X."""

_base_ = ["./_base_.py"]

model = dict(
    enabled_agents=("ego",),
    baseline_name="ego_only",
    baseline_cfg=dict(
        _delete_=True,
        age_decay=0.25,
    ),
)
experiment = dict(
    name="dair_controlled_baseline_ego_only",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method="Ego-only L+C fusion; RSU branches strictly ignored",
    claim_status="controlled adaptation; not an exact source-paper reproduction",
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
        "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256",
        "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
        "RESILIENT_V2X_TRAIN_FAULT_SHA256",
    ),
)
