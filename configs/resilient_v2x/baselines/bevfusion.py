"""Controlled BEVFusion-style fusion baseline on DAIR-V2X."""

_base_ = ["./_base_.py"]

model = dict(
    baseline_name="bevfusion",
    baseline_cfg=dict(
        _delete_=True,
        age_decay=0.25,
        hidden_channels=256,
        norm_groups=32,
    ),
)
experiment = dict(
    name="dair_controlled_baseline_bevfusion",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method="BEVFusion-style adapted fusion",
    claim_status="controlled adaptation; not an exact source-paper reproduction",
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
        "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256",
        "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
        "RESILIENT_V2X_TRAIN_FAULT_SHA256",
    ),
)
