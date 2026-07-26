from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Agent(str, Enum):
    EGO = "ego"
    RSU = "rsu"


class Modality(str, Enum):
    LIDAR = "lidar"
    CAMERA = "camera"


class UnsupportedReason(str, Enum):
    EMPTY_ARRIVAL_SET = "empty_arrival_set"
    EMPTY_MODALITY_HISTORY = "empty_modality_history"
    UNSUPPORTED_HORIZON = "unsupported_horizon"
    INVALID_TIMESTAMP = "invalid_timestamp"
    INVALID_POSE = "invalid_pose"
    INVALID_CALIBRATION = "invalid_calibration"
    MISSING_PAYLOAD = "missing_payload"
    METHOD_NOT_APPLICABLE = "method_not_applicable"


class ProtocolInvariantError(ValueError):
    """Raised when a protocol object is internally inconsistent."""


@dataclass(frozen=True)
class SourceCandidate:
    packet_id: str
    n_s: int
    tau_s_ms: int
    arrival_tau_ms: int | None
    payload_valid: bool
    timestamp_valid: bool
    pose_valid: bool
    calibration_valid: bool
    faulted: bool


@dataclass(frozen=True)
class RejectedCandidate:
    packet_id: str
    n_s: int
    reason: UnsupportedReason


@dataclass(frozen=True)
class BranchSelection:
    agent: Agent
    modality: Modality
    supported: bool
    source: SourceCandidate | None
    horizon: int | None
    observed: bool
    propagated: bool
    rejected: tuple[RejectedCandidate, ...]
    reason: UnsupportedReason | None

    def __post_init__(self) -> None:
        if self.supported:
            if self.source is None or self.horizon is None or self.reason is not None:
                raise ProtocolInvariantError(
                    "supported branch requires source and horizon and forbids reason"
                )
            if self.horizon < 0:
                raise ProtocolInvariantError("horizon must be non-negative")
            if self.observed == self.propagated:
                raise ProtocolInvariantError(
                    "supported branch requires exactly one observed/propagated flag"
                )
        elif (
            self.source is not None
            or self.horizon is not None
            or self.reason is None
            or self.observed
            or self.propagated
        ):
            raise ProtocolInvariantError(
                "unsupported branch requires null source/horizon, false flags, and reason"
            )

    @classmethod
    def unsupported(
        cls,
        agent: Agent,
        modality: Modality,
        reason: UnsupportedReason,
        rejected: tuple[RejectedCandidate, ...],
    ) -> "BranchSelection":
        return cls(
            agent=agent,
            modality=modality,
            supported=False,
            source=None,
            horizon=None,
            observed=False,
            propagated=False,
            rejected=rejected,
            reason=reason,
        )
