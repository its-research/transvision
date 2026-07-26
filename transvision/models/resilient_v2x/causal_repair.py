from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import torch
from torch import Tensor, nn

from .contracts import (
    Agent,
    BranchDiagnostics,
    BranchSelection,
    Modality,
    ProtocolInvariantError,
    RejectedCandidate,
    RepairedBranch,
    SourceCandidate,
    UnsupportedReason,
)
from .geometry import BEVGridSpec, align_bev_to_target, warp_with_displacement
from .ptf import HorizonConditionedPTF, PTFOutput


def _require_integer(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ProtocolInvariantError(f"{name} must be an integer")
    return int(value)


def _normalize_candidates(
    candidates: Sequence[SourceCandidate],
) -> tuple[SourceCandidate, ...]:
    normalized: list[SourceCandidate] = []
    identities: set[tuple[int, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, SourceCandidate):
            raise ProtocolInvariantError(
                "candidates must contain SourceCandidate values"
            )
        n_s = _require_integer(candidate.n_s, "candidate n_s")
        _require_integer(candidate.tau_s_ms, "candidate tau_s_ms")
        if candidate.arrival_tau_ms is not None:
            _require_integer(
                candidate.arrival_tau_ms,
                "candidate arrival_tau_ms",
            )
        if not isinstance(candidate.packet_id, str):
            raise ProtocolInvariantError("candidate packet_id must be a string")
        identity = (n_s, candidate.packet_id)
        if identity in identities:
            raise ProtocolInvariantError(
                "duplicate candidate (n_s, packet_id) identity"
            )
        identities.add(identity)
        normalized.append(candidate)
    return tuple(
        sorted(
            normalized,
            key=lambda item: (item.n_s, item.packet_id),
            reverse=True,
        )
    )


def arrived_candidates(
    agent: Agent,
    target_tau_ms: int,
    candidates: Sequence[SourceCandidate],
) -> tuple[SourceCandidate, ...]:
    target_tau_ms = _require_integer(target_tau_ms, "target_tau_ms")
    if not isinstance(agent, Agent):
        raise ProtocolInvariantError("agent must be an Agent")
    normalized = _normalize_candidates(candidates)
    for candidate in normalized:
        if (
            agent is Agent.RSU
            and candidate.tau_s_ms <= target_tau_ms
            and candidate.arrival_tau_ms is not None
            and candidate.arrival_tau_ms <= target_tau_ms
            and candidate.arrival_tau_ms < candidate.tau_s_ms
        ):
            raise ProtocolInvariantError(
                "arrived RSU candidate has negative delay"
            )
    return tuple(
        candidate
        for candidate in normalized
        if candidate.tau_s_ms <= target_tau_ms
        and (
            agent is Agent.EGO
            or (
                candidate.arrival_tau_ms is not None
                and candidate.arrival_tau_ms <= target_tau_ms
            )
        )
    )


def _metadata_rejection(
    candidate: SourceCandidate,
) -> UnsupportedReason | None:
    if not candidate.timestamp_valid:
        return UnsupportedReason.INVALID_TIMESTAMP
    if not candidate.payload_valid:
        return UnsupportedReason.MISSING_PAYLOAD
    if not candidate.pose_valid:
        return UnsupportedReason.INVALID_POSE
    if not candidate.calibration_valid:
        return UnsupportedReason.INVALID_CALIBRATION
    return None


def unsupported_reason(
    agent: Agent,
    arrived_count: int,
    unmasked_count: int,
    within_horizon_count: int,
    newest_metadata_rejection: UnsupportedReason | None,
) -> UnsupportedReason:
    if agent is Agent.RSU and arrived_count == 0:
        return UnsupportedReason.EMPTY_ARRIVAL_SET
    if unmasked_count == 0:
        return UnsupportedReason.EMPTY_MODALITY_HISTORY
    if newest_metadata_rejection is not None:
        return newest_metadata_rejection
    if within_horizon_count == 0:
        return UnsupportedReason.UNSUPPORTED_HORIZON
    raise ProtocolInvariantError(
        "unsupported reason requested with valid candidate"
    )


def select_causal_source(
    agent: Agent,
    modality: Modality,
    n_t: int,
    target_tau_ms: int,
    delta_t_ms: int,
    history_limit: int,
    candidates: Sequence[SourceCandidate],
) -> BranchSelection:
    if not isinstance(agent, Agent):
        raise ProtocolInvariantError("agent must be an Agent")
    if not isinstance(modality, Modality):
        raise ProtocolInvariantError("modality must be a Modality")
    n_t = _require_integer(n_t, "n_t")
    target_tau_ms = _require_integer(target_tau_ms, "target_tau_ms")
    delta_t_ms = _require_integer(delta_t_ms, "delta_t_ms")
    history_limit = _require_integer(history_limit, "history_limit")
    if delta_t_ms <= 0:
        raise ProtocolInvariantError("delta_t_ms must be positive")
    if history_limit < 0:
        raise ProtocolInvariantError("history_limit must be non-negative")
    if target_tau_ms != n_t * delta_t_ms:
        raise ProtocolInvariantError(
            "target_tau_ms must equal n_t * delta_t_ms"
        )

    normalized = _normalize_candidates(candidates)
    horizons: dict[tuple[int, str], int] = {}
    for candidate in normalized:
        if candidate.tau_s_ms != candidate.n_s * delta_t_ms:
            raise ProtocolInvariantError(
                "candidate tau_s_ms must equal n_s * delta_t_ms"
            )
        if candidate.n_s <= n_t:
            elapsed_ms = target_tau_ms - candidate.tau_s_ms
            if elapsed_ms < 0 or elapsed_ms % delta_t_ms:
                raise ProtocolInvariantError(
                    "candidate time horizon must be a non-negative integer"
                )
            logical_horizon = n_t - candidate.n_s
            time_horizon = elapsed_ms // delta_t_ms
            if logical_horizon < 0 or logical_horizon != time_horizon:
                raise ProtocolInvariantError(
                    "logical and time-derived horizons must match"
                )
            horizons[(candidate.n_s, candidate.packet_id)] = logical_horizon
        if (
            agent is Agent.RSU
            and candidate.n_s <= n_t
            and candidate.arrival_tau_ms is not None
            and candidate.arrival_tau_ms <= target_tau_ms
            and candidate.arrival_tau_ms < candidate.tau_s_ms
        ):
            raise ProtocolInvariantError(
                "arrived RSU candidate has negative delay"
            )

    arrived_count = 0
    unmasked_count = 0
    within_horizon_count = 0
    newest_metadata_rejection: UnsupportedReason | None = None
    newest_unmasked_seen = False
    valid: list[tuple[SourceCandidate, int]] = []
    rejected: list[RejectedCandidate] = []

    for candidate in normalized:
        if candidate.n_s > n_t:
            rejected.append(
                RejectedCandidate(
                    candidate.packet_id,
                    candidate.n_s,
                    UnsupportedReason.INVALID_TIMESTAMP,
                )
            )
            continue
        if agent is Agent.RSU and (
            candidate.arrival_tau_ms is None
            or candidate.arrival_tau_ms > target_tau_ms
        ):
            rejected.append(
                RejectedCandidate(
                    candidate.packet_id,
                    candidate.n_s,
                    UnsupportedReason.EMPTY_ARRIVAL_SET,
                )
            )
            continue

        arrived_count += 1
        if candidate.faulted:
            rejected.append(
                RejectedCandidate(
                    candidate.packet_id,
                    candidate.n_s,
                    UnsupportedReason.EMPTY_MODALITY_HISTORY,
                )
            )
            continue

        unmasked_count += 1
        metadata_rejection = _metadata_rejection(candidate)
        if not newest_unmasked_seen:
            newest_unmasked_seen = True
            newest_metadata_rejection = metadata_rejection
        if metadata_rejection is not None:
            rejected.append(
                RejectedCandidate(
                    candidate.packet_id,
                    candidate.n_s,
                    metadata_rejection,
                )
            )
            continue

        horizon = horizons[(candidate.n_s, candidate.packet_id)]
        if horizon > history_limit:
            rejected.append(
                RejectedCandidate(
                    candidate.packet_id,
                    candidate.n_s,
                    UnsupportedReason.UNSUPPORTED_HORIZON,
                )
            )
            continue

        within_horizon_count += 1
        valid.append((candidate, horizon))

    rejected_tuple = tuple(rejected)
    if valid:
        source, horizon = valid[0]
        observed = agent is Agent.EGO and horizon == 0
        return BranchSelection(
            agent=agent,
            modality=modality,
            supported=True,
            source=source,
            horizon=horizon,
            observed=observed,
            propagated=not observed,
            rejected=rejected_tuple,
            reason=None,
        )

    return BranchSelection.unsupported(
        agent,
        modality,
        unsupported_reason(
            agent,
            arrived_count,
            unmasked_count,
            within_horizon_count,
            newest_metadata_rejection,
        ),
        rejected_tuple,
    )


def _validated_alpha(alpha: float) -> float:
    if (
        not isinstance(alpha, Real)
        or isinstance(alpha, bool)
        or not math.isfinite(alpha)
        or not 0.0 < alpha <= 1.0
    ):
        raise ProtocolInvariantError("alpha must be finite and in (0, 1]")
    return float(alpha)


def age_decay(horizon: Tensor, alpha: float) -> Tensor:
    if not isinstance(horizon, Tensor):
        raise ProtocolInvariantError("horizon must be a tensor")
    if horizon.is_complex() or horizon.dtype is torch.bool:
        raise ProtocolInvariantError("horizon must be real numeric values")
    detached = horizon.detach()
    if not torch.isfinite(detached).all().item():
        raise ProtocolInvariantError("horizon must be finite")
    if detached.lt(0).any().item():
        raise ProtocolInvariantError("horizon must be non-negative")
    if detached.is_floating_point() and detached.ne(detached.round()).any().item():
        raise ProtocolInvariantError("horizon must contain integral values")

    alpha = _validated_alpha(alpha)
    exponent = (
        horizon
        if horizon.is_floating_point()
        else horizon.to(dtype=torch.get_default_dtype())
    )
    return torch.pow(exponent.new_tensor(alpha), exponent)


def _placed_scalar(confidence: Tensor | None, value: float) -> Tensor:
    if isinstance(confidence, Tensor):
        return confidence.new_full((), value)
    return torch.tensor(value)


def branch_reliability(
    selection: BranchSelection,
    confidence: Tensor | None,
    alpha: float,
    batch_index: int,
) -> Tensor:
    if not isinstance(selection, BranchSelection):
        raise ProtocolInvariantError("selection must be a BranchSelection")
    if not selection.supported:
        return _placed_scalar(confidence, 0.0)
    if (
        selection.agent is Agent.EGO
        and selection.horizon == 0
        and selection.observed
    ):
        return _placed_scalar(confidence, 1.0)

    if not isinstance(confidence, Tensor):
        raise ProtocolInvariantError(
            "propagated branch requires a confidence tensor"
        )
    if (
        confidence.ndim != 4
        or confidence.shape[1] != 1
        or any(size <= 0 for size in confidence.shape)
    ):
        raise ProtocolInvariantError(
            "confidence must have non-empty shape [B,1,Y,X]"
        )
    if not confidence.is_floating_point():
        raise ProtocolInvariantError("confidence must be floating")
    detached = confidence.detach()
    if not torch.isfinite(detached).all().item():
        raise ProtocolInvariantError("confidence must be finite")
    if detached.lt(0).any().item() or detached.gt(1).any().item():
        raise ProtocolInvariantError("confidence values must be in [0,1]")
    batch_index = _require_integer(batch_index, "batch_index")
    if batch_index < 0 or batch_index >= confidence.shape[0]:
        raise ProtocolInvariantError("batch_index is out of range")
    if selection.horizon is None:
        raise ProtocolInvariantError(
            "supported branch requires a horizon"
        )

    horizon = confidence.new_tensor(selection.horizon)
    return age_decay(horizon, alpha) * confidence[batch_index].mean()


def normalized_selected_rsu_delay(
    selections: Sequence[BranchSelection],
    history_limit: int,
    delta_t_ms: int,
) -> float:
    history_limit = _require_integer(history_limit, "history_limit")
    delta_t_ms = _require_integer(delta_t_ms, "delta_t_ms")
    if history_limit <= 0 or delta_t_ms <= 0:
        raise ProtocolInvariantError(
            "delay denominator inputs must be positive"
        )

    realized_delays: list[int] = []
    for selection in selections:
        if (
            selection.supported
            and selection.agent is Agent.RSU
            and selection.source is not None
            and selection.source.arrival_tau_ms is not None
        ):
            delay = (
                selection.source.arrival_tau_ms
                - selection.source.tau_s_ms
            )
            if delay < 0:
                raise ProtocolInvariantError(
                    "selected RSU delay cannot be negative"
                )
            realized_delays.append(delay)
    if not realized_delays:
        return 0.0
    denominator = history_limit * delta_t_ms
    return min(max(max(realized_delays) / denominator, 0.0), 1.0)


class CausalBranchRepair(nn.Module):
    """Compose causal selection, geometric alignment, PTF, and age decay."""

    _CHANNELS = 256
    _ALPHA = 0.9
    _HISTORY_LIMIT = 3

    def __init__(
        self,
        grid_spec: BEVGridSpec,
        channels: int,
        alpha: float,
        history_limit: int,
    ) -> None:
        super().__init__()
        if not isinstance(grid_spec, BEVGridSpec):
            raise ValueError("grid_spec must be a BEVGridSpec")
        if (
            not isinstance(channels, Integral)
            or isinstance(channels, bool)
            or int(channels) != self._CHANNELS
        ):
            raise ProtocolInvariantError(
                "channels must be the approved integer 256"
            )
        if (
            not isinstance(alpha, Real)
            or isinstance(alpha, bool)
            or not math.isfinite(alpha)
            or float(alpha) != self._ALPHA
        ):
            raise ProtocolInvariantError(
                "alpha must be the approved finite value 0.9"
            )
        if (
            not isinstance(history_limit, Integral)
            or isinstance(history_limit, bool)
            or int(history_limit) != self._HISTORY_LIMIT
        ):
            raise ProtocolInvariantError(
                "history_limit must be the approved integer 3"
            )
        self.grid_spec = grid_spec
        self.channels = int(channels)
        self.alpha = float(alpha)
        self.history_limit = int(history_limit)

    def expected_output_shape(
        self,
        batch_size: int,
    ) -> tuple[int, int, int, int]:
        if (
            not isinstance(batch_size, Integral)
            or isinstance(batch_size, bool)
            or int(batch_size) <= 0
        ):
            raise ProtocolInvariantError(
                "batch_size must be a positive integer"
            )
        return (
            int(batch_size),
            self.channels,
            self.grid_spec.height,
            self.grid_spec.width,
        )

    @staticmethod
    def _require_floating_tensor(
        value: object,
        name: str,
        rank: int,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise ValueError(f"{name} must be a tensor")
        if value.ndim != rank:
            if name == "source_to_target":
                raise ValueError(
                    "source_to_target must have shape [N,4,4]"
                )
            raise ValueError(f"{name} must be rank-{rank}")
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating")
        return value

    def _validate_inputs(
        self,
        selected_feature: object,
        source_to_target: object,
        ptf_context: object,
        query_agent_index: object,
        selections: object,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        tuple[BranchSelection, ...],
    ]:
        feature = self._require_floating_tensor(
            selected_feature,
            "selected_feature",
            4,
        )
        batch = feature.shape[0]
        if batch <= 0:
            raise ValueError("selected_feature batch must be positive")
        if feature.shape[1] != self.channels:
            raise ValueError(
                "selected_feature channel count must equal channels"
            )
        if feature.shape[2:] != (
            self.grid_spec.height,
            self.grid_spec.width,
        ):
            raise ValueError(
                "selected_feature spatial shape must match grid_spec"
            )

        transform = self._require_floating_tensor(
            source_to_target,
            "source_to_target",
            3,
        )
        if transform.shape[1:] != (4, 4):
            raise ValueError("source_to_target must have shape [N,4,4]")
        if transform.shape[0] != batch:
            raise ValueError(
                "source_to_target batch must match selected_feature"
            )
        context = self._require_floating_tensor(
            ptf_context,
            "ptf_context",
            4,
        )
        if context.shape[0] != batch:
            raise ValueError("ptf_context batch must match selected_feature")
        if any(dimension <= 0 for dimension in context.shape[1:]):
            raise ValueError(
                "ptf_context channel and spatial dimensions must be positive"
            )

        if not isinstance(query_agent_index, Tensor):
            raise ValueError("query_agent_index must be a tensor")
        if (
            query_agent_index.ndim != 1
            or query_agent_index.shape[0] != batch
        ):
            raise ValueError(
                "query_agent_index must have shape [N] matching batch"
            )
        if (
            query_agent_index.dtype is torch.bool
            or query_agent_index.is_floating_point()
            or query_agent_index.is_complex()
        ):
            raise ValueError(
                "query_agent_index must contain non-boolean integers"
            )

        for name, value in (
            ("source_to_target", transform),
            ("ptf_context", context),
        ):
            if value.dtype != feature.dtype:
                raise ValueError(
                    f"{name} dtype must match selected_feature"
                )
            if value.device != feature.device:
                raise ValueError(
                    f"{name} device must match selected_feature"
                )
        if query_agent_index.device != feature.device:
            raise ValueError(
                "query_agent_index device must match selected_feature"
            )

        if (
            not isinstance(selections, Sequence)
            or isinstance(selections, (str, bytes))
        ):
            raise ProtocolInvariantError("selections must be a sequence")
        normalized_selections = tuple(selections)
        if len(normalized_selections) != batch:
            raise ProtocolInvariantError(
                "selection count must match selected_feature batch"
            )
        if any(
            not isinstance(selection, BranchSelection)
            for selection in normalized_selections
        ):
            raise ProtocolInvariantError(
                "selections must contain BranchSelection values"
            )
        modality = normalized_selections[0].modality
        if not isinstance(modality, Modality) or any(
            selection.modality is not modality
            for selection in normalized_selections
        ):
            raise ProtocolInvariantError(
                "all selections in a call must have one modality"
            )
        return (
            feature,
            transform,
            context,
            query_agent_index,
            normalized_selections,
        )

    def _collect_supported(
        self,
        selections: tuple[BranchSelection, ...],
        query_agent_index: Tensor,
    ) -> tuple[list[int], list[int], list[bool], list[bool]]:
        supported_indices: list[int] = []
        horizons: list[int] = []
        observed: list[bool] = []
        propagated: list[bool] = []
        expected_agent_indices: list[int] = []

        for row, selection in enumerate(selections):
            if not isinstance(selection.agent, Agent):
                raise ProtocolInvariantError(
                    "selection agent must be an Agent"
                )
            if not isinstance(selection.supported, bool):
                raise ProtocolInvariantError(
                    "selection supported flag must be boolean"
                )
            if not selection.supported:
                if (
                    selection.source is not None
                    or selection.horizon is not None
                    or selection.observed is not False
                    or selection.propagated is not False
                    or not isinstance(
                        selection.reason,
                        UnsupportedReason,
                    )
                ):
                    raise ProtocolInvariantError(
                        "unsupported selection is internally inconsistent"
                    )
                continue

            horizon_value = selection.horizon
            if (
                not isinstance(horizon_value, Integral)
                or isinstance(horizon_value, bool)
                or not 0 <= int(horizon_value) <= self.history_limit
            ):
                raise ProtocolInvariantError(
                    "supported horizon must be an integer in [0,3]"
                )
            horizon = int(horizon_value)
            expected_observed = (
                selection.agent is Agent.EGO and horizon == 0
            )
            if (
                not isinstance(selection.observed, bool)
                or not isinstance(selection.propagated, bool)
                or selection.observed is not expected_observed
                or selection.propagated is expected_observed
            ):
                raise ProtocolInvariantError(
                    "selection observed/propagated semantics are inconsistent"
                )
            source = selection.source
            if not isinstance(source, SourceCandidate):
                raise ProtocolInvariantError(
                    "supported selection source must be a SourceCandidate"
                )
            _require_integer(source.n_s, "selection source n_s")
            _require_integer(
                source.tau_s_ms,
                "selection source tau_s_ms",
            )
            if selection.reason is not None:
                raise ProtocolInvariantError(
                    "supported selection forbids a reason"
                )

            supported_indices.append(row)
            horizons.append(horizon)
            observed.append(expected_observed)
            propagated.append(not expected_observed)
            expected_agent_indices.append(
                0 if selection.agent is Agent.EGO else 1
            )

        if supported_indices:
            supported_index = torch.tensor(
                supported_indices,
                device=query_agent_index.device,
                dtype=torch.long,
            )
            actual = query_agent_index.index_select(0, supported_index)
            expected = query_agent_index.new_tensor(expected_agent_indices)
            if not torch.equal(actual, expected):
                raise ProtocolInvariantError(
                    "query agent index must agree with each selection agent"
                )
        return supported_indices, horizons, observed, propagated

    @staticmethod
    def _all_finite(value: Tensor) -> bool:
        return torch.isfinite(value.detach()).all().item()

    @classmethod
    def _require_runtime_finite(cls, value: Tensor, name: str) -> None:
        if not cls._all_finite(value):
            raise RuntimeError(f"{name} must contain only finite values")

    def _validate_ptf_output(
        self,
        output: object,
        supported_count: int,
        feature: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(output, PTFOutput):
            raise RuntimeError("ptf.query() must return PTFOutput")
        displacement = output.displacement
        confidence = output.confidence
        if not isinstance(displacement, Tensor):
            raise RuntimeError("PTF displacement must be a tensor")
        if not isinstance(confidence, Tensor):
            raise RuntimeError("PTF confidence must be a tensor")
        if displacement.ndim != 4:
            raise RuntimeError(
                "PTF displacement must have shape [S,2,Y,X]"
            )
        if displacement.shape[0] != supported_count:
            raise RuntimeError(
                "PTF displacement batch must match supported rows"
            )
        if displacement.shape[1] != 2:
            raise RuntimeError(
                "PTF displacement channel count must be two"
            )
        if displacement.shape[2:] != (
            self.grid_spec.height,
            self.grid_spec.width,
        ):
            raise RuntimeError(
                "PTF displacement spatial shape must match grid_spec"
            )
        if confidence.ndim != 4:
            raise RuntimeError(
                "PTF confidence must have shape [S,1,Y,X]"
            )
        if confidence.shape[0] != supported_count:
            raise RuntimeError(
                "PTF confidence batch must match supported rows"
            )
        if confidence.shape[1] != 1:
            raise RuntimeError(
                "PTF confidence channel count must be one"
            )
        if confidence.shape[2:] != (
            self.grid_spec.height,
            self.grid_spec.width,
        ):
            raise RuntimeError(
                "PTF confidence spatial shape must match grid_spec"
            )
        for name, value in (
            ("displacement", displacement),
            ("confidence", confidence),
        ):
            if not value.is_floating_point():
                raise RuntimeError(f"PTF {name} must be floating")
            if value.dtype != feature.dtype:
                raise RuntimeError(
                    f"PTF {name} dtype must match selected_feature"
                )
            if value.device != feature.device:
                raise RuntimeError(
                    f"PTF {name} device must match selected_feature"
                )
            self._require_runtime_finite(value, f"PTF {name}")
        detached_confidence = confidence.detach()
        if (
            detached_confidence.lt(0).any().item()
            or detached_confidence.gt(1).any().item()
        ):
            raise RuntimeError("PTF confidence must be in [0,1]")
        return displacement, confidence

    @staticmethod
    def _unsupported_diagnostics(
        selection: BranchSelection,
    ) -> BranchDiagnostics:
        return BranchDiagnostics(
            agent=selection.agent,
            modality=selection.modality,
            supported=False,
            source_tick=None,
            source_tau_ms=None,
            horizon=None,
            observed=False,
            propagated=False,
            gamma=None,
            reliability=0.0,
            ptf_queried=False,
            displacement=None,
            confidence=None,
            reason=selection.reason,
        )

    def forward(
        self,
        selected_feature: Tensor,
        source_to_target: Tensor,
        ptf_context: Tensor,
        ptf: HorizonConditionedPTF,
        query_agent_index: Tensor,
        selections: Sequence[BranchSelection],
    ) -> RepairedBranch:
        (
            selected_feature,
            source_to_target,
            ptf_context,
            query_agent_index,
            normalized_selections,
        ) = self._validate_inputs(
            selected_feature,
            source_to_target,
            ptf_context,
            query_agent_index,
            selections,
        )
        (
            supported_indices,
            horizon_values,
            observed_values,
            propagated_values,
        ) = self._collect_supported(
            normalized_selections,
            query_agent_index,
        )
        batch = selected_feature.shape[0]
        feature_shape = self.expected_output_shape(batch)

        if not supported_indices:
            return RepairedBranch(
                feature=selected_feature.new_zeros(feature_shape),
                reliability=selected_feature.new_zeros(batch),
                support=torch.zeros(
                    batch,
                    dtype=torch.bool,
                    device=selected_feature.device,
                ),
                observed=torch.zeros(
                    batch,
                    dtype=torch.bool,
                    device=selected_feature.device,
                ),
                propagated=torch.zeros(
                    batch,
                    dtype=torch.bool,
                    device=selected_feature.device,
                ),
                normalized_age=selected_feature.new_zeros(batch),
                displacement=None,
                confidence=None,
                diagnostics=tuple(
                    self._unsupported_diagnostics(selection)
                    for selection in normalized_selections
                ),
            )

        supported_index = torch.tensor(
            supported_indices,
            dtype=torch.long,
            device=selected_feature.device,
        )
        supported_feature = selected_feature.index_select(
            0,
            supported_index,
        )
        supported_transform = source_to_target.index_select(
            0,
            supported_index,
        )
        supported_context = ptf_context.index_select(
            0,
            supported_index,
        )
        supported_agent_index = query_agent_index.index_select(
            0,
            supported_index,
        )
        horizon = torch.tensor(
            horizon_values,
            dtype=torch.long,
            device=selected_feature.device,
        )

        if not self._all_finite(supported_context):
            raise ValueError(
                "supported ptf_context must contain only finite values"
            )
        aligned_feature = align_bev_to_target(
            supported_feature,
            supported_transform,
            self.grid_spec,
        )
        self._require_runtime_finite(aligned_feature, "aligned feature")
        ptf_output = ptf.query(
            supported_context,
            supported_agent_index,
            horizon,
        )
        displacement, confidence = self._validate_ptf_output(
            ptf_output,
            len(supported_indices),
            selected_feature,
        )
        warped = warp_with_displacement(
            aligned_feature,
            displacement,
        )
        self._require_runtime_finite(warped, "warped feature")

        gamma = torch.pow(
            selected_feature.new_tensor(self.alpha),
            horizon.to(dtype=selected_feature.dtype),
        )
        normalized_age = (
            horizon.to(dtype=selected_feature.dtype) / self.history_limit
        )
        observed = torch.tensor(
            observed_values,
            dtype=torch.bool,
            device=selected_feature.device,
        )
        propagated = torch.tensor(
            propagated_values,
            dtype=torch.bool,
            device=selected_feature.device,
        )
        reliability = torch.where(
            observed,
            torch.ones_like(gamma),
            gamma * confidence.mean(dim=(-3, -2, -1)),
        )
        repaired = gamma[:, None, None, None] * warped

        for name, value in (
            ("gamma", gamma),
            ("normalized age", normalized_age),
            ("reliability", reliability),
            ("repaired feature", repaired),
        ):
            self._require_runtime_finite(value, name)
        for name, value in (
            ("gamma", gamma),
            ("normalized age", normalized_age),
            ("reliability", reliability),
        ):
            detached = value.detach()
            if (
                detached.lt(0).any().item()
                or detached.gt(1).any().item()
            ):
                raise RuntimeError(f"{name} must be in [0,1]")

        full_feature = selected_feature.new_zeros(feature_shape).index_copy(
            0,
            supported_index,
            repaired,
        )
        full_reliability = selected_feature.new_zeros(batch).index_copy(
            0,
            supported_index,
            reliability,
        )
        full_normalized_age = selected_feature.new_zeros(batch).index_copy(
            0,
            supported_index,
            normalized_age,
        )
        full_support = torch.zeros(
            batch,
            dtype=torch.bool,
            device=selected_feature.device,
        ).index_fill(0, supported_index, True)
        full_observed = torch.zeros(
            batch,
            dtype=torch.bool,
            device=selected_feature.device,
        ).index_copy(0, supported_index, observed)
        full_propagated = torch.zeros(
            batch,
            dtype=torch.bool,
            device=selected_feature.device,
        ).index_copy(0, supported_index, propagated)
        full_displacement = displacement.new_zeros(
            (
                batch,
                2,
                self.grid_spec.height,
                self.grid_spec.width,
            )
        ).index_copy(0, supported_index, displacement)
        full_confidence = confidence.new_zeros(
            (
                batch,
                1,
                self.grid_spec.height,
                self.grid_spec.width,
            )
        ).index_copy(0, supported_index, confidence)

        supported_positions = {
            row: position
            for position, row in enumerate(supported_indices)
        }
        diagnostics: list[BranchDiagnostics] = []
        for row, selection in enumerate(normalized_selections):
            if not selection.supported:
                diagnostics.append(
                    self._unsupported_diagnostics(selection)
                )
                continue
            position = supported_positions[row]
            source = selection.source
            if source is None:
                raise ProtocolInvariantError(
                    "supported selection requires source"
                )
            diagnostics.append(
                BranchDiagnostics(
                    agent=selection.agent,
                    modality=selection.modality,
                    supported=True,
                    source_tick=source.n_s,
                    source_tau_ms=source.tau_s_ms,
                    horizon=horizon_values[position],
                    observed=observed_values[position],
                    propagated=propagated_values[position],
                    gamma=float(gamma[position].detach().item()),
                    reliability=float(
                        reliability[position].detach().item()
                    ),
                    ptf_queried=True,
                    displacement=displacement[position].detach(),
                    confidence=confidence[position].detach(),
                    reason=None,
                )
            )

        return RepairedBranch(
            feature=full_feature,
            reliability=full_reliability,
            support=full_support,
            observed=full_observed,
            propagated=full_propagated,
            normalized_age=full_normalized_age,
            displacement=full_displacement,
            confidence=full_confidence,
            diagnostics=tuple(diagnostics),
        )
