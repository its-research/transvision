"""Controlled FFNet-style fusion baseline on DAIR-V2X.

The source paper uses two consecutive RSU frames to supervise its feature-flow
derivative.  The shared four-selected-branch contract exposes one causal source
per branch, so this is an explicitly named single-frame L+C adaptation rather
than an exact reproduction of the paper's two-stage LiDAR-only system.
"""

_base_ = ["./_base_.py"]

model = dict(
    baseline_name="ffnet",
    baseline_cfg=dict(
        _delete_=True,
        age_decay=0.0,
        hidden_channels=256,
        norm_groups=32,
        max_displacement_per_age=1.0,
    ),
)
experiment = dict(
    name="dair_controlled_baseline_ffnet",
    stage="controlled_baseline",
    protocol="causal multimodal cooperative detection",
    method="FFNet-style single-frame L+C adapted fusion",
    claim_status="controlled adaptation; not an exact source-paper reproduction",
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
        "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256",
        "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
        "RESILIENT_V2X_TRAIN_FAULT_SHA256",
    ),
)
