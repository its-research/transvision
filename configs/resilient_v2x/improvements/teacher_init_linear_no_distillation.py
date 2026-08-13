"""Teacher-initialized linear-PTF candidate without distillation losses."""

_base_ = ["./linear_no_distillation.py"]

experiment = dict(
    name="dair_improvement_teacher_init_linear_no_distillation",
    stage="resilient_improvement",
    improvement=dict(
        component="initialization+PTF+distillation",
        setting=(
            "selected clean-teacher initialization, linear PTF, "
            "no distillation loss"
        ),
    ),
)
