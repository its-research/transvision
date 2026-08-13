"""No-reliability single-factor candidate with student bbox loss weight 2.5."""

_base_ = ["../ablations/no_reliability.py"]

model = dict(bbox_head=dict(loss_bbox=dict(loss_weight=2.5)))

experiment = dict(
    _delete_=True,
    name="dair_improvement_no_reliability_bbox25",
    stage="resilient_improvement",
    improvement=dict(
        component="student detection bbox regression loss",
        setting=(
            "no-reliability base with student bbox loss weight 2.5; "
            "frozen teacher remains 2.0"
        ),
    ),
    candidate_provenance=dict(
        base_config="configs/resilient_v2x/ablations/no_reliability.py",
        base_experiment="dair_ablation_no_reliability",
        changed_model_leaf="model.bbox_head.loss_bbox.loss_weight",
        base_value=2.0,
        candidate_value=2.5,
        frozen_teacher_value=2.0,
    ),
    trigger=dict(
        mode="mutually_exclusive",
        group="bbox25_single_factor_followup",
        peer_experiment="dair_improvement_support_residual_bbox25",
    ),
    protocol_lock=dict(
        training_seed=20250218,
        precision="FP32",
        gpu_count=4,
        train_batch_size_per_gpu=2,
        global_batch_size=8,
        max_epochs=50,
        val_interval=10,
    ),
)
