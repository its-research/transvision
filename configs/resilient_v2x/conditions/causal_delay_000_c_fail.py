"""Causal endpoint C-Fail at fixed RSU delay 0 ms (E+R scope)."""

_base_ = ["./global_delay_000_full.py"]

_fault_path = __import__("os").getenv(
    "RESILIENT_V2X_TEST_CAUSAL_DELAY_000_C_FAIL_OVERLAY",
    "artifacts/resilient_v2x/dair/val_causal_delay_000_c_fail.jsonl.zst",
)
_fault_sha256 = (
    __import__("os").getenv("RESILIENT_V2X_TEST_CAUSAL_DELAY_000_C_FAIL_SHA256") or None
)

test_dataloader = dict(
    dataset=dict(
        fault_overlay_path=_fault_path,
        fault_overlay_sha256=_fault_sha256,
    )
)
experiment = dict(
    name="dair_causal_delay_000_c_fail",
    condition=dict(
        type="causal_endpoint",
        delay_ms=0,
        fault="C-Fail",
        scope="E+R",
        duration_ticks=1,
        order="arrival selection before fault masking",
    ),
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_TEST_TRANSPORT_DELAY_000_OVERLAY",
        "RESILIENT_V2X_TEST_TRANSPORT_DELAY_000_SHA256",
        "RESILIENT_V2X_TEST_CAUSAL_DELAY_000_C_FAIL_OVERLAY",
        "RESILIENT_V2X_TEST_CAUSAL_DELAY_000_C_FAIL_SHA256",
    ),
)
