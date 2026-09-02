"""Lightweight EventTrack-V2X protocol and estimator core.

This package intentionally imports only the Python standard library and NumPy;
it is usable without MMDetection3D, SciPy, AB3DMOT, or CUDA.
"""

from .association import (
    Assignment,
    GateResult,
    GaussianAssociation,
    associate_gaussians,
    chi_square_gate,
    chi_square_quantile,
    gaussian_nll,
    solve_one_to_one,
)
from .clock import (
    AffineClockMap,
    MappedEventTimes,
    TimeEstimate,
    conservative_no_future_gate,
)
from .commit import AppendOnlyCommitLog, CommitRecord, GENESIS_HASH
from .fusion import CIFusion, covariance_intersection
from .network import (
    GilbertElliott,
    LossModel,
    NetworkConfig,
    NetworkEvent,
    NetworkTrace,
    PacketRequest,
    generate_network_trace,
)
from .replay import (
    CorrelatedBeliefRequiresCI,
    FixedLagReplay,
    IngestResult,
    IngestStatus,
    LinearGaussianDynamics,
    OutOfWindowPolicy,
    ReplaySnapshot,
    constant_velocity_dynamics,
)
from .scheduler import ScheduleCandidate, ScheduleResult, select_exact_budget
from .schema import (
    CorrelatedTrackBelief,
    EventTimes,
    IndependentIncrement,
    Lineage,
    LocalTimestamps,
    ReceivedMessage,
    WireMessage,
)
from .wire import decode_message, encode_message, wire_digest, wire_size

__all__ = [
    "AffineClockMap",
    "AppendOnlyCommitLog",
    "Assignment",
    "CIFusion",
    "CommitRecord",
    "CorrelatedBeliefRequiresCI",
    "CorrelatedTrackBelief",
    "EventTimes",
    "FixedLagReplay",
    "GENESIS_HASH",
    "GateResult",
    "GaussianAssociation",
    "GilbertElliott",
    "IndependentIncrement",
    "IngestResult",
    "IngestStatus",
    "Lineage",
    "LinearGaussianDynamics",
    "LocalTimestamps",
    "LossModel",
    "MappedEventTimes",
    "NetworkConfig",
    "NetworkEvent",
    "NetworkTrace",
    "OutOfWindowPolicy",
    "PacketRequest",
    "ReceivedMessage",
    "ReplaySnapshot",
    "ScheduleCandidate",
    "ScheduleResult",
    "TimeEstimate",
    "WireMessage",
    "associate_gaussians",
    "chi_square_gate",
    "chi_square_quantile",
    "constant_velocity_dynamics",
    "conservative_no_future_gate",
    "covariance_intersection",
    "decode_message",
    "encode_message",
    "gaussian_nll",
    "generate_network_trace",
    "select_exact_budget",
    "solve_one_to_one",
    "wire_digest",
    "wire_size",
]
