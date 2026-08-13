"""Controlled V2VNet-style L+C fusion baseline on DAIR-V2X."""

_base_ = ["./_base_.py"]

model = dict(
    baseline_name="v2vnet",
    baseline_cfg=dict(
        _delete_=True,
        age_decay=0.25,
        message_iterations=3,
        kernel_size=3,
    ),
)
experiment = dict(
    name="dair_controlled_baseline_v2vnet",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method="V2VNet-style two-agent ConvGRU message-passing L+C adaptation",
    claim_status="controlled adaptation; not an exact source-paper reproduction",
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
        "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256",
        "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
        "RESILIENT_V2X_TRAIN_FAULT_SHA256",
    ),
)
