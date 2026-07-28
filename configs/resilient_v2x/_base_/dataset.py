"""Strict DAIR-V2X temporal dataset configuration.

All artifact identities are external inputs.  No placeholder SHA-256 is
invented: unset hashes remain ``None`` and the dataset fails before I/O.
"""



dataset_type = "ResilientTemporalDataset"
evaluation_point_cloud_range = [0.0, -40.0, -3.0, 80.0, 40.0, 1.0]
data_root = __import__("os").getenv(
    "RESILIENT_V2X_DATA_ROOT",
    "data/DAIR-V2X/cooperative-vehicle-infrastructure",
)
manifest_path = __import__("os").getenv(
    "RESILIENT_V2X_MANIFEST",
    "artifacts/resilient_v2x/dair/temporal_manifest.json",
)
expected_split_hash = __import__("os").getenv("RESILIENT_V2X_SPLIT_SHA256") or None

train_transport_overlay_path = __import__("os").getenv(
    "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY",
    "artifacts/resilient_v2x/dair/train_transport.jsonl.zst",
)
train_transport_overlay_sha256 = (
    __import__("os").getenv("RESILIENT_V2X_TRAIN_TRANSPORT_SHA256") or None
)
train_fault_overlay_path = __import__("os").getenv(
    "RESILIENT_V2X_TRAIN_FAULT_OVERLAY",
    "artifacts/resilient_v2x/dair/train_fault.jsonl.zst",
)
train_fault_overlay_sha256 = (
    __import__("os").getenv("RESILIENT_V2X_TRAIN_FAULT_SHA256") or None
)

implementation_choices_dataset = dict(
    status="implementation choice; the paper does not report these values",
    train_batch_size_per_gpu=1,
    workers_per_gpu=4,
    camera_image_size=(256, 704),
    global_seed=20250218,
    training_transport_overlay=(
        "external protocol artifact selected by the reproducer"
    ),
    training_fault_overlay=(
        "external protocol artifact selected by the reproducer"
    ),
)

paper_data_contract = dict(
    dataset="DAIR-V2X cooperative vehicle-infrastructure",
    target_class=("Car",),
    delta_t_ms=100,
    history_limit=3,
    decision_is_causal=True,
    evaluation_delays_ms=(0, 100, 200, 300),
    fault_conditions=("Full", "L-Fail", "C-Fail"),
)

common_dataset = dict(
    type=dataset_type,
    manifest_path=manifest_path,
    data_root=data_root,
    expected_split_hash=expected_split_hash,
    allow_fixture=False,
    seed=20250218,
    load_camera=True,
    load_lidar=True,
    camera_image_size=(256, 704),
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(
        type="EpochIndexSampler",
        shuffle=True,
        seed=20250218,
        drop_last=False,
    ),
    collate_fn=dict(type="collate_resilient_samples"),
    dataset=dict(
        **common_dataset,
        split="train",
        include_clean_teacher=True,
        transport_overlay_path=train_transport_overlay_path,
        transport_overlay_sha256=train_transport_overlay_sha256,
        fault_overlay_path=train_fault_overlay_path,
        fault_overlay_sha256=train_fault_overlay_sha256,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(
        type="EpochIndexSampler",
        shuffle=False,
        seed=20250218,
        drop_last=False,
    ),
    collate_fn=dict(type="collate_resilient_samples"),
    dataset=dict(
        **common_dataset,
        split="val",
        include_clean_teacher=False,
        transport_overlay_path=None,
        transport_overlay_sha256=None,
        fault_overlay_path=None,
        fault_overlay_sha256=None,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(
        type="EpochIndexSampler",
        shuffle=False,
        seed=20250218,
        drop_last=False,
    ),
    collate_fn=dict(type="collate_resilient_samples"),
    dataset=dict(
        **common_dataset,
        split="test",
        include_clean_teacher=False,
        transport_overlay_path=None,
        transport_overlay_sha256=None,
        fault_overlay_path=None,
        fault_overlay_sha256=None,
    ),
)

val_evaluator = dict(
    type="ResilientV2XMetric",
    iou_thresholds=(0.5, 0.7),
    max_detections=100,
    point_cloud_range=evaluation_point_cloud_range,
    prediction_output=__import__("os").getenv("RESILIENT_V2X_PREDICTION_OUTPUT") or None,
)
test_evaluator = val_evaluator
