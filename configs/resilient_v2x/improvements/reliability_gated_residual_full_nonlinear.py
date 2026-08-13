"""P4: nonlinear PTF, full reliability descriptor, and gated residual."""

_base_ = ["./support_residual.py"]

model = dict(support_residual_reliability_gate=True)

implementation_choices_runtime = dict(
    support_residual_weight=0.5,
    support_residual_reliability_gate=True,
    router_reliability_descriptor="full raw modality reliability",
    support_residual_reliability_source=(
        "raw branch reliability, shared with the full router descriptor"
    ),
    support_residual_reliability_gate_formula=(
        "clamp(mean(s_lidar*(r_lidar_ego+r_lidar_rsu)/2, "
        "s_camera*(r_camera_ego+r_camera_rsu)/2),0,1)"
    ),
    support_residual_formula=(
        "router_fused + support_residual_weight * reliability_gate * "
        "(raw_reliability_weighted_mean - router_fused)"
    ),
)

experiment = dict(
    _delete_=True,
    name="dair_improvement_reliability_gated_residual_full_nonlinear",
    stage="resilient_improvement",
    improvement=dict(
        component="PTF+full reliability descriptor+gated residual",
        setting=(
            "nonlinear PTF, full raw-reliability router descriptor, and "
            "raw-reliability-gated support residual"
        ),
    ),
    candidate_provenance=dict(
        candidate_id="P4",
        base_config="configs/resilient_v2x/improvements/support_residual.py",
        base_experiment="dair_improvement_support_residual",
        changed_model_leaf="model.support_residual_reliability_gate",
        base_value=False,
        candidate_value=True,
    ),
    protocol_lock=dict(
        dataset="DAIR training dataset sealed by the formal run contract",
        teacher="unified frozen teacher sealed by the formal run contract",
        training_seed=20250218,
        precision="FP32",
        gpu_count=4,
        train_batch_size_per_gpu=2,
        global_batch_size=8,
        max_epochs=50,
        val_interval=10,
    ),
)
