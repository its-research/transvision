"""Paper evaluation condition: Full sensors, fixed RSU delay 0 ms."""

_base_ = ["../dair_resilient_v2x.py"]

test_dataloader = dict(
    dataset=dict(
        transport_overlay_path=None,
        transport_overlay_sha256=None,
        fault_overlay_path=None,
        fault_overlay_sha256=None,
    )
)
experiment = dict(
    name="dair_global_delay_000_full",
    stage="evaluation",
    condition=dict(type="fixed_delay", delay_ms=0, fault="Full"),
    required_external_inputs=("RESILIENT_V2X_SPLIT_SHA256",),
)
