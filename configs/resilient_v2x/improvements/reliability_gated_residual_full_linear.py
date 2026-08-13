"""P6: linear PTF, full reliability descriptor, and gated residual."""

_base_ = ["./reliability_gated_residual_full_nonlinear.py"]

model = dict(ptf_mode="linear")
experiment = dict(
    _delete_=True,
    name="dair_improvement_reliability_gated_residual_full_linear",
    stage="resilient_improvement",
    improvement=dict(
        component="PTF",
        setting=(
            "P4 full reliability descriptor and gated residual with linear PTF"
        ),
    ),
    candidate_provenance=dict(
        candidate_id="P6",
        base_config=(
            "configs/resilient_v2x/improvements/"
            "reliability_gated_residual_full_nonlinear.py"
        ),
        base_experiment=(
            "dair_improvement_reliability_gated_residual_full_nonlinear"
        ),
        changed_model_leaf="model.ptf_mode",
        base_value="nonlinear",
        candidate_value="linear",
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
