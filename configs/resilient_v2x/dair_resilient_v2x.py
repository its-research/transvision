"""Main two-stage ResilientV2X student configuration on DAIR-V2X."""

_base_ = [
    "./_base_/model.py",
    "./_base_/dataset.py",
    "./_base_/runtime.py",
]

# The runtime hook initializes only the four shared encoder/projection/head
# prefixes. The outer resilient fusion remains method-specific and random;
# ResilientV2XNet independently strict-loads the complete nested frozen teacher.

experiment = dict(
    name="resilient_v2x_dair",
    stage="distilled_student",
    protocol="causal multimodal cooperative detection",
    student_initialization="selected clean-teacher shared prefixes only",
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
        "RESILIENT_V2X_COMMON_INIT_CHECKPOINT",
        "RESILIENT_V2X_COMMON_INIT_SHA256",
        "RESILIENT_V2X_TEACHER_CHECKPOINT or the documented default checkpoint",
    ),
)
