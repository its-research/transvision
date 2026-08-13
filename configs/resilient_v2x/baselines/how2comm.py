"""Clean-room How2comm-style L+C controlled fusion baseline."""

_base_ = ["./_base_.py"]

custom_imports = dict(
    imports=["transvision.models.resilient_v2x.how2comm_baseline"],
    allow_failed_imports=False,
)

model = dict(
    baseline_name="how2comm",
    baseline_cfg=dict(
        _delete_=True,
        selection_reduction=4,
        communication_threshold=0.2,
        age_decay=0.25,
        temporal_gain=0.5,
    ),
)
experiment = dict(
    name="dair_controlled_baseline_how2comm",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method=(
        "clean-room How2comm-style spatial-channel communication selection, "
        "lightweight age-conditioned temporal context, and pragmatic L+C fusion"
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
