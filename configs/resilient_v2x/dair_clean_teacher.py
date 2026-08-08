"""Stage 1: train the clean, zero-latency teacher on DAIR-V2X."""

_os = __import__("os")

_base_ = ["./dair_resilient_v2x.py"]

model = dict(
    teacher=None,
    teacher_checkpoint=None,
    distillation=None,
)
load_from = _os.getenv("RESILIENT_V2X_VEHICLE_PRETRAIN_CHECKPOINT") or None
del _os
train_dataloader = dict(
    dataset=dict(
        include_clean_teacher=False,
        transport_overlay_path=None,
        transport_overlay_sha256=None,
        fault_overlay_path=None,
        fault_overlay_sha256=None,
    )
)
default_hooks = dict(
    checkpoint=dict(
        filename_tmpl="teacher_epoch_{}.pth",
    )
)
experiment = dict(
    name="resilient_v2x_dair_clean_teacher",
    stage="clean_teacher",
    protocol="zero latency and no injected faults",
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "optional RESILIENT_V2X_VEHICLE_PRETRAIN_CHECKPOINT",
    ),
)
