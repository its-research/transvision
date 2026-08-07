"""Stage 1: train the clean, zero-latency teacher on DAIR-V2X."""

_os = __import__("os")

_base_ = ["./dair_resilient_v2x.py"]

model = dict(
    teacher=None,
    teacher_checkpoint=None,
    distillation=None,
)


def _resolve_vehicle_pretrain_checkpoint() -> str | None:
    env_path = _os.getenv("RESILIENT_V2X_VEHICLE_PRETRAIN_CHECKPOINT")
    if env_path:
        return env_path
    # Resume teacher-only from the durable xyz2zyx vehicle80 OutputModel when
    # the mid-epoch teacher weights were not uploaded before stop.
    model_id = _os.getenv(
        "RESILIENT_V2X_VEHICLE_PRETRAIN_MODEL_ID",
        "af1f02070a6b4f7c8a75d5ce3a91ba6e",
    )
    try:
        from clearml import Model

        local = Model(model_id=model_id).get_local_copy()
    except Exception:
        return None
    return str(local) if local else None


load_from = _resolve_vehicle_pretrain_checkpoint()
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
# Full multimodal val every epoch dominated wall-clock; match vehicle cadence.
train_cfg = dict(max_epochs=50, val_interval=10)
default_hooks = dict(
    checkpoint=dict(
        filename_tmpl="teacher_epoch_{}.pth",
        interval=1,
        max_keep_ckpts=5,
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
