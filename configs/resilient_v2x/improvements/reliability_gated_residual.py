"""P1: P0 router descriptor with a raw-reliability-gated residual."""

_base_ = ["./support_residual_no_reliability_linear.py"]

model = dict(support_residual_reliability_gate=True)

implementation_choices_runtime = dict(
    support_residual_reliability_gate=True,
    support_residual_reliability_source=(
        "raw branch reliability, independent of router descriptor inputs"
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
    name="dair_improvement_reliability_gated_residual",
    stage="resilient_improvement",
    improvement=dict(
        component="dynamic expert router residual output",
        setting=(
            "P0 linear PTF and support-only router reliability descriptor; "
            "raw-reliability-weighted residual with fixed-slot reliability gate"
        ),
    ),
)
