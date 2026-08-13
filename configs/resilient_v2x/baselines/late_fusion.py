"""Controlled LateFusion-style L+C feature baseline on DAIR-V2X."""

_base_ = ["./_base_.py"]

custom_imports = dict(
    imports=["transvision.models.resilient_v2x.late_fusion_baseline"],
    allow_failed_imports=False,
)

model = dict(
    baseline_name="late_fusion",
    baseline_cfg=dict(
        _delete_=True,
        age_decay=0.25,
    ),
)
experiment = dict(
    name="dair_controlled_baseline_late_fusion",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method=(
        "LateFusion-style L+C (controlled late-feature adaptation; not exact "
        "source-paper reproduction)"
    ),
    claim_status="controlled adaptation; not an exact source-paper reproduction",
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
        "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256",
        "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
        "RESILIENT_V2X_TRAIN_FAULT_SHA256",
    ),
)
