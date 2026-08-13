"""Controlled When2com-style L+C fusion baseline on DAIR-V2X."""

_base_ = ["./_base_.py"]

custom_imports = dict(
    imports=["transvision.models.resilient_v2x.when2com_baseline"],
    allow_failed_imports=False,
)

model = dict(
    baseline_name="when2com",
    baseline_cfg=dict(
        _delete_=True,
        query_key_channels=64,
        age_decay=0.25,
        temperature=1.0,
    ),
)
experiment = dict(
    name="dair_controlled_baseline_when2com",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method=("When2com-style two-agent query/key communication-gated L+C adaptation"),
    claim_status="controlled adaptation; not an exact source-paper reproduction",
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
        "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256",
        "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
        "RESILIENT_V2X_TRAIN_FAULT_SHA256",
    ),
)
