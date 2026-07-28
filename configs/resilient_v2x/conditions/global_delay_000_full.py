"""Paper evaluation condition: Full sensors, fixed RSU delay 0 ms."""

_base_ = ["../dair_resilient_v2x.py"]

_transport_path = __import__("os").getenv(
    "RESILIENT_V2X_TEST_TRANSPORT_DELAY_000_OVERLAY",
    "artifacts/resilient_v2x/dair/val_transport_delay_000.jsonl.zst",
)
_transport_sha256 = (
    __import__("os").getenv("RESILIENT_V2X_TEST_TRANSPORT_DELAY_000_SHA256") or None
)

test_dataloader = dict(
    dataset=dict(
        transport_overlay_path=_transport_path,
        transport_overlay_sha256=_transport_sha256,
        fault_overlay_path=None,
        fault_overlay_sha256=None,
    )
)
experiment = dict(
    name="dair_global_delay_000_full",
    stage="evaluation",
    condition=dict(type="fixed_delay", delay_ms=0, fault="Full"),
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TEST_TRANSPORT_DELAY_000_OVERLAY",
        "RESILIENT_V2X_TEST_TRANSPORT_DELAY_000_SHA256",
    ),
)
