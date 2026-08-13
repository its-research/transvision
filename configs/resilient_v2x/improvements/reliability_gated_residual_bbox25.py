"""P3: trained P1 with only the student bbox loss raised to 2.5."""

_base_ = ["./reliability_gated_residual.py"]

model = dict(bbox_head=dict(loss_bbox=dict(loss_weight=2.5)))
experiment = dict(
    _delete_=True,
    name="dair_improvement_reliability_gated_residual_bbox25",
    stage="resilient_improvement",
    improvement=dict(
        component="student detection bbox regression loss",
        setting="trained P1 with student bbox loss 2.5; frozen teacher remains 2.0",
    ),
    candidate_provenance=dict(
        candidate_id="P3",
        base_config=(
            "configs/resilient_v2x/improvements/reliability_gated_residual.py"
        ),
        base_experiment="dair_improvement_reliability_gated_residual",
        changed_model_leaf="model.bbox_head.loss_bbox.loss_weight",
        base_value=2.0,
        candidate_value=2.5,
        frozen_teacher_value=2.0,
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
