"""Main two-stage ResilientV2X student configuration on DAIR-V2X."""

_base_ = [
    "./_base_/model.py",
    "./_base_/dataset.py",
    "./_base_/runtime.py",
]

experiment = dict(
    name="resilient_v2x_dair",
    stage="distilled_student",
    protocol="causal multimodal cooperative detection",
    claim_status=(
        "implementation profile only; the paper publishes no verifiable "
        "reference code or result artifact"
    ),
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
        "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256",
        "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
        "RESILIENT_V2X_TRAIN_FAULT_SHA256",
        "RESILIENT_V2X_TEACHER_CHECKPOINT or the documented default checkpoint",
    ),
)
