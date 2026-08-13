"""Controlled F-Cooper-style L+C fusion baseline on DAIR-V2X."""

_base_ = ["./_base_.py"]

model = dict(
    baseline_name="fcooper",
    baseline_cfg=dict(_delete_=True),
)
experiment = dict(
    name="dair_controlled_baseline_fcooper",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method="F-Cooper-style support-aware spatial-max L+C adaptation",
    claim_status="controlled adaptation; not an exact source-paper reproduction",
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
        "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256",
        "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
        "RESILIENT_V2X_TRAIN_FAULT_SHA256",
    ),
)
