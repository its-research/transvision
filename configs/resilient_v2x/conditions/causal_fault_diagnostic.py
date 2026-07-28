"""Parameterized causal fault diagnostic for scope and duration sweeps.

Set the environment variables listed in ``required_external_inputs``.  The
fault overlay must have been generated after causal arrival selection for the
same transport overlay; this config never synthesizes protocol records.
"""



_base_ = ["../dair_resilient_v2x.py"]

_delay_ms = int(__import__("os").getenv("RESILIENT_V2X_DIAGNOSTIC_DELAY_MS", "300"))
_scope = __import__("os").getenv("RESILIENT_V2X_DIAGNOSTIC_SCOPE", "R-only")
_modality = __import__("os").getenv("RESILIENT_V2X_DIAGNOSTIC_MODALITY", "lidar")
_duration_ticks = int(
    __import__("os").getenv("RESILIENT_V2X_DIAGNOSTIC_DURATION_TICKS", "1")
)
if _delay_ms not in (0, 100, 200, 300):
    raise ValueError("diagnostic delay must be 0, 100, 200, or 300 ms")
if _scope not in ("E+R", "E-only", "R-only"):
    raise ValueError("diagnostic scope must be E+R, E-only, or R-only")
if _modality not in ("lidar", "camera"):
    raise ValueError("diagnostic modality must be lidar or camera")
if _duration_ticks not in (1, 2, 3, 4):
    raise ValueError("diagnostic duration must be one to four ticks")

_transport_path = None
_transport_sha256 = None
if _delay_ms > 0:
    _transport_path = __import__("os").getenv(
        "RESILIENT_V2X_DIAGNOSTIC_TRANSPORT_OVERLAY",
        f"artifacts/resilient_v2x/dair/test_transport_delay_{_delay_ms:03d}.jsonl.zst",
    )
    _transport_sha256 = (
        __import__("os").getenv("RESILIENT_V2X_DIAGNOSTIC_TRANSPORT_SHA256") or None
    )
_fault_path = __import__("os").getenv(
    "RESILIENT_V2X_DIAGNOSTIC_FAULT_OVERLAY",
    (
        "artifacts/resilient_v2x/dair/"
        f"test_causal_{_delay_ms:03d}_{_scope}_{_modality}_d{_duration_ticks}.jsonl.zst"
    ),
)
_fault_sha256 = __import__("os").getenv("RESILIENT_V2X_DIAGNOSTIC_FAULT_SHA256") or None

test_dataloader = dict(dataset=dict(
    transport_overlay_path=_transport_path,
    transport_overlay_sha256=_transport_sha256,
    fault_overlay_path=_fault_path,
    fault_overlay_sha256=_fault_sha256,
))
experiment = dict(
    name="dair_causal_fault_diagnostic",
    stage="diagnostic",
    condition=dict(
        type="causal_endpoint_continuous",
        delay_ms=_delay_ms,
        fault=("L-Fail" if _modality == "lidar" else "C-Fail"),
        scope=_scope,
        duration_ticks=_duration_ticks,
        order="arrival selection before fault masking",
    ),
    supported_matrix=dict(
        delay_ms=(0, 100, 200, 300),
        scope=("E+R", "E-only", "R-only"),
        modality=("lidar", "camera"),
        duration_ticks=(1, 2, 3, 4),
    ),
    required_external_inputs=(
        "RESILIENT_V2X_SPLIT_SHA256",
        "RESILIENT_V2X_DIAGNOSTIC_TRANSPORT_OVERLAY when delay > 0",
        "RESILIENT_V2X_DIAGNOSTIC_TRANSPORT_SHA256 when delay > 0",
        "RESILIENT_V2X_DIAGNOSTIC_FAULT_OVERLAY",
        "RESILIENT_V2X_DIAGNOSTIC_FAULT_SHA256",
    ),
)
