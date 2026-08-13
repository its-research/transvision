"""P0 candidate with a student-only bbox regression loss weight of 2.5."""

_base_ = ["./support_residual_no_reliability_linear.py"]

model = dict(bbox_head=dict(loss_bbox=dict(loss_weight=2.5)))
experiment = dict(
    name="dair_improvement_support_residual_no_reliability_linear_bbox25",
    stage="resilient_improvement",
    improvement=dict(
        component="student detection bbox regression loss",
        setting="P0 with student bbox loss weight 2.5; frozen teacher remains 2.0",
    ),
)
