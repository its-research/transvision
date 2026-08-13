"""Clean-room DiscoNet-style matrix-edge controlled L+C baseline."""

_base_ = ["./_base_.py"]

custom_imports = dict(
    imports=["transvision.models.resilient_v2x.disconet_baseline"],
    allow_failed_imports=False,
)

model = dict(
    baseline_name="disconet",
    baseline_cfg=dict(
        _delete_=True,
        edge_hidden_channels=128,
        age_decay=0.25,
    ),
)
experiment = dict(
    name="dair_controlled_baseline_disconet",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method=("DiscoNet-style cell-wise matrix-valued edge-attention L+C adaptation"),
    claim_status="controlled adaptation; not an exact source-paper reproduction",
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
        "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256",
        "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
        "RESILIENT_V2X_TRAIN_FAULT_SHA256",
    ),
)
