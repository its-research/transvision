"""Shared controlled-comparison profile on the ResilientV2X protocol.

The detector reuses the exact DAIR-V2X encoders, head, dataset, runtime, and
evaluator from the main experiment. Only the cooperative fusion block is
selected by each child config. These are protocol-controlled adaptations,
not claims of bit-exact reproduction of the source papers.
"""

_model_components = __import__(
    "configs.resilient_v2x._base_.model",
    fromlist=(
        "bbox_head",
        "bev_grid",
        "camera_encoder",
        "data_preprocessor",
        "detection_projection",
        "lidar_encoder",
    ),
)
bbox_head = _model_components.bbox_head
bev_grid = _model_components.bev_grid
camera_encoder = _model_components.camera_encoder
data_preprocessor = _model_components.data_preprocessor
detection_projection = _model_components.detection_projection
lidar_encoder = _model_components.lidar_encoder
del _model_components


_base_ = [
    "../_base_/dataset.py",
    "../_base_/runtime.py",
]

model = dict(
    type="ControlledCooperativeBaselineNet",
    grid_spec=bev_grid,
    lidar_encoder=lidar_encoder,
    camera_encoder=camera_encoder,
    bbox_head=bbox_head,
    detection_projection=detection_projection,
    data_preprocessor=data_preprocessor,
    baseline_name="v2x_vit",
    baseline_cfg=dict(
        age_decay=0.25,
        num_heads=8,
        window_sizes=(2, 4),
        dropout=0.0,
        mlp_ratio=2.0,
    ),
)

# Baselines are trained directly against detection targets. They do not
# instantiate the clean teacher or request clean-teacher tensors from data.
train_dataloader = dict(dataset=dict(include_clean_teacher=False))

controlled_baseline_contract = dict(
    status="controlled adaptation; not an exact source-paper reproduction",
    dataset="DAIR-V2X cooperative vehicle-infrastructure",
    modalities=("LiDAR", "camera"),
    branch_order=("L_E", "L_R", "C_E", "C_R"),
    causal_history=True,
    shared_encoders_and_detection_head=True,
    distillation=False,
    ptf=False,
    der=False,
)
