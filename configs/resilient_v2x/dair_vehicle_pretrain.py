"""Stage 0: pretrain the shared LiDAR encoder and Car head on vehicle scans."""

_components = __import__(
    "configs.resilient_v2x._base_.model",
    fromlist=("bbox_head", "data_preprocessor", "lidar_encoder"),
)
bbox_head = _components.bbox_head
data_preprocessor = _components.data_preprocessor
lidar_encoder = _components.lidar_encoder
del _components

_base_ = [
    "./_base_/dataset.py",
    "./_base_/runtime.py",
]

model = dict(
    type="VehiclePointPillarsPretrainNet",
    lidar_encoder=lidar_encoder,
    bbox_head=bbox_head,
    data_preprocessor=data_preprocessor,
)

train_dataloader = dict(
    dataset=dict(
        load_camera=False,
        include_clean_teacher=False,
        transport_overlay_path=None,
        transport_overlay_sha256=None,
        fault_overlay_path=None,
        fault_overlay_sha256=None,
    )
)
val_dataloader = dict(dataset=dict(load_camera=False))
test_dataloader = dict(dataset=dict(load_camera=False))

default_hooks = dict(
    checkpoint=dict(
        filename_tmpl="vehicle_epoch_{}.pth",
    )
)

vehicle_pretrain_contract = dict(
    source="current vehicle LiDAR only",
    agent="ego",
    horizon=0,
    shared_checkpoint_prefixes=("lidar_encoder.", "bbox_head."),
    checkpoint_selection="exact final epoch for downstream teacher transfer",
)

experiment = dict(
    name="resilient_v2x_dair_vehicle_pretrain",
    stage="vehicle_pretrain",
    protocol="current vehicle LiDAR only; zero latency and no injected faults",
    required_external_inputs=("RESILIENT_V2X_SPLIT_SHA256",),
    transfer_contract=("lidar_encoder", "bbox_head"),
)
