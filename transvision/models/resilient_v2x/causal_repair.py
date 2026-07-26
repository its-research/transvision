from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import torch
from torch import Tensor

from .contracts import (
    Agent,
    BranchSelection,
    Modality,
    ProtocolInvariantError,
    RejectedCandidate,
    SourceCandidate,
    UnsupportedReason,
)


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
