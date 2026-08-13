"""Clean-room Where2comm-style L+C controlled fusion baseline."""

_base_ = ["./_base_.py"]

custom_imports = dict(
    imports=["transvision.models.resilient_v2x.where2comm_baseline"],
    allow_failed_imports=False,
)

model = dict(
    baseline_name="where2comm",
    baseline_cfg=dict(
        _delete_=True,
        communication_threshold=0.5,
        age_decay=0.25,
    ),
)
experiment = dict(
    name="dair_controlled_baseline_where2comm",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method=(
        "clean-room Where2comm-style spatial confidence communication and "
        "support-aware attention L+C adaptation"
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
