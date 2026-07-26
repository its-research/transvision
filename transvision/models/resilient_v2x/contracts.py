from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Literal

import torch
from torch import Tensor


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


@dataclass(frozen=True)
class BranchDiagnostics:
    agent: Agent
    modality: Modality
    supported: bool
    source_tick: int | None
    source_tau_ms: int | None
    horizon: int | None
    observed: bool
    propagated: bool
    gamma: float | None
    reliability: float
    ptf_queried: bool
    displacement: Tensor | None
    confidence: Tensor | None
    reason: UnsupportedReason | None


@dataclass
class RepairedBranch:
    feature: Tensor
    reliability: Tensor
    support: Tensor
    observed: Tensor
    propagated: Tensor
    normalized_age: Tensor
    displacement: Tensor | None
    confidence: Tensor | None
    diagnostics: tuple[BranchDiagnostics, ...]


def _require_trimmed_text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProtocolInvariantError(
            f"{name} must be a non-empty, non-whitespace string"
        )
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolInvariantError(f"{name} must be boolean")
    return value


def _require_integer(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ProtocolInvariantError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ProtocolInvariantError(f"{name} is below its minimum")
    if maximum is not None and normalized > maximum:
        raise ProtocolInvariantError(f"{name} exceeds its maximum")
    return normalized


def _require_unit_scalar(value: object, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ProtocolInvariantError(f"{name} must be finite and in [0,1]")
    return float(value)


def _require_finite_tensor(
    value: object,
    name: str,
    shape: tuple[int, ...],
    *,
    floating: bool,
    detached: bool,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ProtocolInvariantError(f"{name} must be a tensor")
    if value.shape != shape:
        raise ProtocolInvariantError(f"{name} has an invalid shape")
    if floating:
        if not value.is_floating_point():
            raise ProtocolInvariantError(f"{name} must be floating")
    elif value.dtype is not torch.bool:
        raise ProtocolInvariantError(f"{name} must be boolean")
    if detached and value.requires_grad:
        raise ProtocolInvariantError(f"{name} must be detached")
    if value.device.type == "meta":
        raise ProtocolInvariantError(
            f"{name} device must contain materialized values"
        )
    if floating and not torch.isfinite(value).all().item():
        raise ProtocolInvariantError(f"{name} must contain only finite values")
    return value


def _validate_routing_diagnostics(value: object) -> "RoutingDiagnostics":
    if not isinstance(value, RoutingDiagnostics):
        raise ProtocolInvariantError(
            "routing must be a RoutingDiagnostics record"
        )
    fields = (value.descriptor, value.expert_support, value.weights)
    present = tuple(field is not None for field in fields)
    if any(present) and not all(present):
        raise ProtocolInvariantError(
            "routing tensor fields must be all present or all absent"
        )
    if not any(present):
        _require_trimmed_text(
            value.not_applicable_reason,
            "routing not-applicable reason",
        )
        return value
    if value.not_applicable_reason is not None:
        raise ProtocolInvariantError(
            "applicable routing forbids a not-applicable reason"
        )

    descriptor = _require_finite_tensor(
        value.descriptor,
        "routing descriptor",
        (783,),
        floating=True,
        detached=True,
    )
    expert_support = _require_finite_tensor(
        value.expert_support,
        "routing expert support",
        (3,),
        floating=False,
        detached=True,
    )
    weights = _require_finite_tensor(
        value.weights,
        "routing weights",
        (3,),
        floating=True,
        detached=True,
    )
    if (
        descriptor.device != expert_support.device
        or descriptor.device != weights.device
    ):
        raise ProtocolInvariantError(
            "routing diagnostic tensors must share a device"
        )
    if weights.lt(0).any().item() or weights.gt(1).any().item():
        raise ProtocolInvariantError("routing weights must be in [0,1]")
    if weights.masked_select(~expert_support).ne(0).any().item():
        raise ProtocolInvariantError(
            "unsupported expert weights must be exactly zero"
        )
    if expert_support.any().item():
        tolerance = 10.0 * torch.finfo(weights.dtype).eps
        if not torch.allclose(
            weights.sum(),
            weights.new_tensor(1.0),
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ProtocolInvariantError(
                "supported routing weights must sum to one"
            )
    elif weights.ne(0).any().item():
        raise ProtocolInvariantError(
            "all-invalid routing weights must be exactly zero"
        )
    return value


@dataclass(frozen=True)
class RoutingDiagnostics:
    descriptor: Tensor | None
    expert_support: Tensor | None
    weights: Tensor | None
    not_applicable_reason: str | None

    def __post_init__(self) -> None:
        _validate_routing_diagnostics(self)


def _validate_branch_diagnostics(
    branch: object,
    expected_agent: Agent,
    expected_modality: Modality,
) -> BranchDiagnostics:
    if not isinstance(branch, BranchDiagnostics):
        raise ProtocolInvariantError(
            "branches must contain BranchDiagnostics records"
        )
    if (
        branch.agent is not expected_agent
        or branch.modality is not expected_modality
    ):
        raise ProtocolInvariantError(
            "branches must use fixed [L_E,L_R,C_E,C_R] order"
        )
    supported = _require_bool(branch.supported, "branch supported")
    observed = _require_bool(branch.observed, "branch observed")
    propagated = _require_bool(branch.propagated, "branch propagated")
    ptf_queried = _require_bool(branch.ptf_queried, "branch ptf_queried")
    reliability = _require_unit_scalar(
        branch.reliability,
        "branch reliability",
    )

    if not supported:
        if (
            branch.source_tick is not None
            or branch.source_tau_ms is not None
            or branch.horizon is not None
            or observed
            or propagated
            or branch.gamma is not None
            or reliability != 0.0
            or ptf_queried
            or branch.displacement is not None
            or branch.confidence is not None
            or not isinstance(branch.reason, UnsupportedReason)
        ):
            raise ProtocolInvariantError(
                "unsupported branch requires null algorithm fields, "
                "false flags, zero reliability, and a typed reason"
            )
        return branch

    source_tick = _require_integer(
        branch.source_tick,
        "supported branch source tick",
        minimum=0,
    )
    source_tau_ms = _require_integer(
        branch.source_tau_ms,
        "supported branch source time",
    )
    horizon = _require_integer(
        branch.horizon,
        "supported branch horizon",
        minimum=0,
        maximum=3,
    )
    if source_tick < 0 or not isinstance(source_tau_ms, int):
        raise ProtocolInvariantError("supported branch source metadata is invalid")
    if observed == propagated:
        raise ProtocolInvariantError(
            "supported branch requires exactly one observed/propagated flag"
        )
    expected_observed = expected_agent is Agent.EGO and horizon == 0
    if observed is not expected_observed or propagated is expected_observed:
        raise ProtocolInvariantError(
            "supported branch observed/propagated semantics are invalid"
        )
    if branch.reason is not None:
        raise ProtocolInvariantError("supported branch forbids a reason")

    if not ptf_queried:
        if (
            branch.gamma is not None
            or branch.displacement is not None
            or branch.confidence is not None
        ):
            raise ProtocolInvariantError(
                "branch without PTF requires null gamma/displacement/confidence"
            )
        return branch

    _require_unit_scalar(branch.gamma, "supported PTF branch gamma")
    displacement = branch.displacement
    confidence = branch.confidence
    if not isinstance(displacement, Tensor):
        raise ProtocolInvariantError(
            "supported PTF branch requires displacement"
        )
    if (
        displacement.ndim != 3
        or displacement.shape[0] != 2
        or displacement.shape[1] <= 0
        or displacement.shape[2] <= 0
    ):
        raise ProtocolInvariantError(
            "branch displacement must be [2,Y,X]"
        )
    displacement = _require_finite_tensor(
        displacement,
        "branch displacement",
        tuple(displacement.shape),
        floating=True,
        detached=True,
    )
    expected_confidence_shape = (
        1,
        displacement.shape[1],
        displacement.shape[2],
    )
    confidence = _require_finite_tensor(
        confidence,
        "branch confidence",
        expected_confidence_shape,
        floating=True,
        detached=True,
    )
    if (
        confidence.dtype != displacement.dtype
        or confidence.device != displacement.device
    ):
        raise ProtocolInvariantError(
            "branch displacement/confidence dtype and device must match"
        )
    if confidence.lt(0).any().item() or confidence.gt(1).any().item():
        raise ProtocolInvariantError(
            "branch confidence must be in [0,1]"
        )
    return branch


@dataclass(frozen=True)
class SampleInferenceDiagnostics:
    schema_version: Literal[1]
    sample_id: str
    method: str
    branches: tuple[
        BranchDiagnostics,
        BranchDiagnostics,
        BranchDiagnostics,
        BranchDiagnostics,
    ]
    routing: RoutingDiagnostics

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, Integral)
            or isinstance(self.schema_version, bool)
            or int(self.schema_version) != 1
        ):
            raise ProtocolInvariantError("schema_version must be integer 1")
        _require_trimmed_text(self.sample_id, "sample_id")
        _require_trimmed_text(self.method, "method")
        if type(self.branches) is not tuple or len(self.branches) != 4:
            raise ProtocolInvariantError(
                "branches must be a tuple of exactly four records"
            )
        expected_order = (
            (Agent.EGO, Modality.LIDAR),
            (Agent.RSU, Modality.LIDAR),
            (Agent.EGO, Modality.CAMERA),
            (Agent.RSU, Modality.CAMERA),
        )
        for branch, (agent, modality) in zip(
            self.branches,
            expected_order,
        ):
            _validate_branch_diagnostics(branch, agent, modality)
        _validate_routing_diagnostics(self.routing)


@dataclass
class ResilientFeatureBatch:
    fused: Tensor
    overall_support: Tensor
    routing_weights: Tensor
    routing_descriptor: Tensor
    branch_features: Mapping[str, Tensor]
    diagnostics: tuple[SampleInferenceDiagnostics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fused, Tensor):
            raise ProtocolInvariantError("fused feature must be a tensor")
        if (
            self.fused.ndim != 4
            or any(dimension <= 0 for dimension in self.fused.shape)
            or self.fused.shape[1] != 256
        ):
            raise ProtocolInvariantError(
                "fused feature must be positive [B,256,Y,X]"
            )
        if not self.fused.is_floating_point():
            raise ProtocolInvariantError("fused feature must be floating")
        if self.fused.device.type == "meta":
            raise ProtocolInvariantError("fused feature must be materialized")
        if not torch.isfinite(self.fused).all().item():
            raise ProtocolInvariantError(
                "fused feature must contain only finite values"
            )

        batch = self.fused.shape[0]
        overall_support = _require_finite_tensor(
            self.overall_support,
            "overall support",
            (batch,),
            floating=False,
            detached=False,
        )
        routing_weights = _require_finite_tensor(
            self.routing_weights,
            "routing weights",
            (batch, 3),
            floating=True,
            detached=False,
        )
        routing_descriptor = _require_finite_tensor(
            self.routing_descriptor,
            "routing descriptor",
            (batch, 783),
            floating=True,
            detached=False,
        )
        if (
            overall_support.device != self.fused.device
            or routing_weights.device != self.fused.device
            or routing_descriptor.device != self.fused.device
        ):
            raise ProtocolInvariantError(
                "feature batch tensors must share a device"
            )
        if routing_weights.lt(0).any().item():
            raise ProtocolInvariantError(
                "routing weights must be nonnegative"
            )
        unsupported_rows = ~overall_support
        if (
            routing_weights.masked_select(
                unsupported_rows[:, None].expand_as(routing_weights)
            )
            .ne(0)
            .any()
            .item()
        ):
            raise ProtocolInvariantError(
                "unsupported rows require exact zero routing weights"
            )
        if overall_support.any().item():
            supported_sums = routing_weights[overall_support].sum(dim=1)
            tolerance = 10.0 * torch.finfo(routing_weights.dtype).eps
            if not torch.allclose(
                supported_sums,
                torch.ones_like(supported_sums),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ProtocolInvariantError(
                    "supported row routing weights must sum to one"
                )

        if type(self.diagnostics) is not tuple:
            raise ProtocolInvariantError("diagnostics must be a tuple")
        if len(self.diagnostics) != batch:
            raise ProtocolInvariantError(
                "diagnostics count must equal batch size"
            )
        sample_ids: list[str] = []
        for index, diagnostic in enumerate(self.diagnostics):
            if not isinstance(diagnostic, SampleInferenceDiagnostics):
                raise ProtocolInvariantError(
                    "diagnostics must contain sample records"
                )
            diagnostic.__post_init__()
            sample_ids.append(diagnostic.sample_id)
            routing = _validate_routing_diagnostics(diagnostic.routing)
            if (
                routing.descriptor is None
                or routing.expert_support is None
                or routing.weights is None
            ):
                raise ProtocolInvariantError(
                    "Resilient feature diagnostics require applicable routing"
                )
            if (
                routing.descriptor.dtype != routing_descriptor.dtype
                or routing.descriptor.device != routing_descriptor.device
                or routing.weights.dtype != routing_weights.dtype
                or routing.weights.device != routing_weights.device
            ):
                raise ProtocolInvariantError(
                    "diagnostic routing dtype and device must match batch rows"
                )
            if not torch.equal(
                routing.descriptor,
                routing_descriptor[index],
            ):
                raise ProtocolInvariantError(
                    "diagnostic routing descriptor must match batch row"
                )
            if not torch.equal(routing.weights, routing_weights[index]):
                raise ProtocolInvariantError(
                    "diagnostic routing weights must match batch row"
                )
            if bool(routing.expert_support.any().item()) is not bool(
                overall_support[index].item()
            ):
                raise ProtocolInvariantError(
                    "diagnostic expert support must match overall support"
                )
        if len(sample_ids) != len(set(sample_ids)):
            raise ProtocolInvariantError("sample IDs must be unique")

        if not isinstance(self.branch_features, Mapping):
            raise ProtocolInvariantError(
                "branch_features must be a mapping"
            )
        for key, value in self.branch_features.items():
            if type(key) is not str or not key:
                raise ProtocolInvariantError(
                    "branch feature keys must be non-empty strings"
                )
            if not isinstance(value, Tensor):
                raise ProtocolInvariantError(
                    "branch feature values must be tensors"
                )
            if value.shape != self.fused.shape:
                raise ProtocolInvariantError(
                    "branch features must match fused feature shape"
                )
            if (
                value.dtype != self.fused.dtype
                or value.device != self.fused.device
            ):
                raise ProtocolInvariantError(
                    "branch features must match fused dtype and device"
                )
            if value.device.type == "meta" or not torch.isfinite(value).all().item():
                raise ProtocolInvariantError(
                    "branch features must contain finite materialized values"
                )
