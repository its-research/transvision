"""Deadline-conditioned exact-byte additive proxy scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

import numpy as np

from .network import (
    NetworkTraceV1,
    WireAccountingConfigV1,
    bounded_wire_reservation_v1,
    condition_plan_v1,
)
from .schema import WireMessage
from .wire import encode_message, wire_digest


class BudgetBasis(str, Enum):
    APPLICATION = "application"
    ON_WIRE = "on_wire"


class SchedulerKind(str, Enum):
    FULL_SEND = "full_send"
    PERIODIC = "periodic"
    RANDOM = "random"
    FIFO_AOI = "fifo_aoi"
    CONFIDENCE_TOP_K = "confidence_top_k"
    MARGINAL_VOI = "marginal_voi"


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CausalScoreEvidenceV1:
    """Ground-truth-free score inputs frozen before a network-domain decision."""

    as_of_network_time: float
    risk_reduction: float
    on_time_probability: float
    confidence: float
    age_seconds: float
    tracker_state_sha256: str
    channel_model_sha256: str
    scorer_config_sha256: str
    on_time_estimate_sha256: str
    candidate_message_sha256: str
    candidate_network_transmit_time: float
    candidate_deadline: float
    ground_truth_free: bool

    def __post_init__(self) -> None:
        as_of = float(self.as_of_network_time)
        risk = float(self.risk_reduction)
        probability = float(self.on_time_probability)
        confidence = float(self.confidence)
        age = float(self.age_seconds)
        candidate_transmit = float(self.candidate_network_transmit_time)
        candidate_deadline = float(self.candidate_deadline)
        if not np.isfinite(as_of):
            raise ValueError("as_of_network_time must be finite")
        if not np.isfinite(risk) or risk < 0.0:
            raise ValueError("risk_reduction must be finite and non-negative")
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("on_time_probability must be in [0, 1]")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not np.isfinite(age) or age < 0.0:
            raise ValueError("age_seconds must be finite and non-negative")
        if not np.isfinite(candidate_transmit):
            raise ValueError("candidate_network_transmit_time must be finite")
        if not np.isfinite(candidate_deadline) or candidate_deadline < candidate_transmit:
            raise ValueError(
                "candidate_deadline must be finite and not precede candidate transmit"
            )
        if as_of > candidate_transmit:
            raise ValueError("score evidence is from the future of candidate transmit")
        for name in (
            "tracker_state_sha256",
            "channel_model_sha256",
            "scorer_config_sha256",
            "on_time_estimate_sha256",
            "candidate_message_sha256",
        ):
            if (
                type(getattr(self, name)) is not str
                or _SHA256.fullmatch(getattr(self, name)) is None
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.ground_truth_free is not True:
            raise ValueError("scheduler score evidence must be ground-truth-free")
        object.__setattr__(self, "as_of_network_time", as_of)
        object.__setattr__(self, "risk_reduction", risk)
        object.__setattr__(self, "on_time_probability", probability)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "age_seconds", age)
        object.__setattr__(
            self, "candidate_network_transmit_time", candidate_transmit
        )
        object.__setattr__(self, "candidate_deadline", candidate_deadline)


@dataclass(frozen=True, slots=True)
class ScheduleCandidate:
    message: WireMessage
    network_transmit_time: float
    score_evidence: CausalScoreEvidenceV1
    projected_on_wire_bytes: int | None = None

    def __post_init__(self) -> None:
        network_time = float(self.network_transmit_time)
        if not np.isfinite(network_time):
            raise ValueError("network_transmit_time must be finite")
        if not isinstance(self.score_evidence, CausalScoreEvidenceV1):
            raise TypeError("score_evidence must be CausalScoreEvidenceV1")
        if self.score_evidence.as_of_network_time > network_time:
            raise ValueError("scheduler score evidence is from the future")
        if self.score_evidence.candidate_message_sha256 != wire_digest(self.message):
            raise ValueError("scheduler score evidence is bound to another message")
        if self.score_evidence.candidate_network_transmit_time != network_time:
            raise ValueError("scheduler score evidence is bound to another transmit time")
        if self.score_evidence.candidate_deadline != self.message.deadline:
            raise ValueError("scheduler score evidence is bound to another deadline")
        if self.projected_on_wire_bytes is not None:
            if (
                isinstance(self.projected_on_wire_bytes, bool)
                or not isinstance(self.projected_on_wire_bytes, int)
            ):
                raise TypeError("projected_on_wire_bytes must be an integer")
            if self.projected_on_wire_bytes <= 0:
                raise ValueError("projected_on_wire_bytes must be positive")
            if self.projected_on_wire_bytes < len(encode_message(self.message)):
                raise ValueError(
                    "projected_on_wire_bytes cannot be smaller than application bytes"
                )
        object.__setattr__(self, "network_transmit_time", network_time)

    @property
    def risk_reduction(self) -> float:
        return self.score_evidence.risk_reduction

    @property
    def on_time_probability(self) -> float:
        return self.score_evidence.on_time_probability

    @property
    def confidence(self) -> float:
        return self.score_evidence.confidence

    @property
    def age_seconds(self) -> float:
        return self.score_evidence.age_seconds

    @property
    def encoded_bytes(self) -> bytes:
        return encode_message(self.message)

    @property
    def byte_cost(self) -> int:
        """Legacy application-level encoded byte cost."""

        return len(self.encoded_bytes)

    @property
    def application_byte_cost(self) -> int:
        return self.byte_cost

    @property
    def on_wire_byte_cost(self) -> int:
        if self.projected_on_wire_bytes is None:
            raise ValueError(
                "on-wire scheduling requires projected_on_wire_bytes; "
                "application bytes cannot be used as a silent fallback"
            )
        return self.projected_on_wire_bytes

    @property
    def expected_value(self) -> float:
        return self.risk_reduction * self.on_time_probability


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    selected: tuple[ScheduleCandidate, ...]
    total_bytes: int
    expected_risk_reduction: float
    budget_basis: BudgetBasis

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(candidate.message.message_id for candidate in self.selected)


@dataclass(frozen=True, slots=True)
class BaselineScheduleResult:
    """A scheduler decision with both byte ledgers kept explicit."""

    scheduler: SchedulerKind
    selected: tuple[ScheduleCandidate, ...]
    total_application_bytes: int
    total_on_wire_bytes: int
    expected_risk_reduction: float
    budget_bytes: int | None
    budget_basis: BudgetBasis

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(candidate.message.message_id for candidate in self.selected)

    @property
    def budget_used_bytes(self) -> int:
        if self.budget_basis is BudgetBasis.APPLICATION:
            return self.total_application_bytes
        return self.total_on_wire_bytes

    @property
    def budget_compliant(self) -> bool:
        return self.budget_bytes is None or self.budget_used_bytes <= self.budget_bytes


@dataclass(frozen=True, slots=True)
class StreamScheduleResultV1:
    scheduler: SchedulerKind
    selected: tuple[ScheduleCandidate, ...]
    total_application_bytes: int
    total_reserved_on_wire_bytes: int
    total_on_wire_bytes: int
    byte_rate_limit_per_second: int
    burst_seconds: float
    network_trace_sha256: str
    budget_violations: int

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(candidate.message.message_id for candidate in self.selected)

    @property
    def budget_compliant(self) -> bool:
        return self.budget_violations == 0


def select_exact_budget(
    candidates: list[ScheduleCandidate] | tuple[ScheduleCandidate, ...],
    *,
    budget_bytes: int,
    budget_basis: BudgetBasis | str = BudgetBasis.APPLICATION,
) -> ScheduleResult:
    """Solve the additive 0-1 proxy under one explicitly named byte ledger.

    The default is the canonical EventTrack application payload size for
    backwards-compatible unit tests.  Confirmatory line-rate experiments must
    pass ``ON_WIRE`` with trace-derived projections, or use
    :func:`schedule_rate_limited_v1` for the frozen bytes-per-second contract.
    """

    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
        raise TypeError("budget_bytes must be an integer")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    basis = BudgetBasis(budget_basis)
    ordered = tuple(sorted(candidates, key=lambda item: item.message.message_id))
    identifiers = [candidate.message.message_id for candidate in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate message IDs must be unique")

    # used bytes -> (expected value, selected candidate indices)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, candidate in enumerate(ordered):
        cost = _candidate_cost(candidate, basis)
        value = candidate.expected_value
        next_states = dict(states)
        for used, (total_value, selected) in states.items():
            new_used = used + cost
            if new_used > budget_bytes:
                continue
            proposal = (total_value + value, selected + (index,))
            incumbent = next_states.get(new_used)
            if incumbent is None or proposal[0] > incumbent[0] + 1e-12 or (
                abs(proposal[0] - incumbent[0]) <= 1e-12
                and tuple(identifiers[item] for item in proposal[1])
                < tuple(identifiers[item] for item in incumbent[1])
            ):
                next_states[new_used] = proposal
        states = next_states

    best_used, (best_value, best_indices) = min(
        states.items(),
        key=lambda item: (
            -item[1][0],
            item[0],
            tuple(identifiers[index] for index in item[1][1]),
        ),
    )
    selected = tuple(ordered[index] for index in best_indices)
    return ScheduleResult(
        selected=selected,
        total_bytes=best_used,
        expected_risk_reduction=best_value,
        budget_basis=basis,
    )


def _candidate_cost(candidate: ScheduleCandidate, basis: BudgetBasis) -> int:
    if basis is BudgetBasis.APPLICATION:
        return candidate.application_byte_cost
    return candidate.on_wire_byte_cost


def _canonical_candidates(
    candidates: list[ScheduleCandidate] | tuple[ScheduleCandidate, ...],
) -> tuple[ScheduleCandidate, ...]:
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.network_transmit_time,
                item.message.source,
                item.message.sequence,
                item.message.message_id,
            ),
        )
    )
    identifiers = [candidate.message.message_id for candidate in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate message IDs must be unique")
    return ordered


def _ranked_pack(
    ordered: tuple[ScheduleCandidate, ...],
    *,
    budget_bytes: int,
    budget_basis: BudgetBasis,
    max_messages: int | None,
) -> tuple[ScheduleCandidate, ...]:
    selected: list[ScheduleCandidate] = []
    used = 0
    for candidate in ordered:
        if max_messages is not None and len(selected) >= max_messages:
            break
        cost = _candidate_cost(candidate, budget_basis)
        if used + cost <= budget_bytes:
            selected.append(candidate)
            used += cost
    return tuple(selected)


def _exact_value_pack(
    candidates: tuple[ScheduleCandidate, ...],
    *,
    budget_bytes: int,
    budget_basis: BudgetBasis,
    max_messages: int | None,
) -> tuple[ScheduleCandidate, ...]:
    identifiers = [candidate.message.message_id for candidate in candidates]
    # used bytes -> (expected value, selected candidate indices)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, candidate in enumerate(candidates):
        cost = _candidate_cost(candidate, budget_basis)
        next_states = dict(states)
        for used, (total_value, selected) in states.items():
            if max_messages is not None and len(selected) >= max_messages:
                continue
            new_used = used + cost
            if new_used > budget_bytes:
                continue
            proposal = (total_value + candidate.expected_value, selected + (index,))
            incumbent = next_states.get(new_used)
            if incumbent is None or proposal[0] > incumbent[0] + 1e-12 or (
                abs(proposal[0] - incumbent[0]) <= 1e-12
                and tuple(identifiers[item] for item in proposal[1])
                < tuple(identifiers[item] for item in incumbent[1])
            ):
                next_states[new_used] = proposal
        states = next_states
    _, (_, best_indices) = min(
        states.items(),
        key=lambda item: (
            -item[1][0],
            item[0],
            tuple(identifiers[index] for index in item[1][1]),
        ),
    )
    return tuple(candidates[index] for index in best_indices)


def schedule_baseline(
    candidates: list[ScheduleCandidate] | tuple[ScheduleCandidate, ...],
    scheduler: SchedulerKind | str,
    *,
    budget_bytes: int | None,
    budget_basis: BudgetBasis | str = BudgetBasis.ON_WIRE,
    seed: int = 0,
    period: int = 1,
    phase: int = 0,
    max_messages: int | None = None,
) -> BaselineScheduleResult:
    """Run a deterministic preregistered scheduler baseline.

    Full-send deliberately ignores the cap and reports ``budget_compliant``;
    every other scheduler is hard-capped.  Random priority is reproducible and
    independent of the input container order.
    """

    kind = SchedulerKind(scheduler)
    basis = BudgetBasis(budget_basis)
    ordered = _canonical_candidates(candidates)
    if budget_bytes is not None:
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
            raise TypeError("budget_bytes must be an integer or None")
        if budget_bytes < 0:
            raise ValueError("budget_bytes must be non-negative")
    if kind is not SchedulerKind.FULL_SEND and budget_bytes is None:
        raise ValueError("capped schedulers require budget_bytes")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")
    if isinstance(phase, bool) or not isinstance(phase, int) or not 0 <= phase < period:
        raise ValueError("phase must satisfy 0 <= phase < period")
    if max_messages is not None:
        if (
            isinstance(max_messages, bool)
            or not isinstance(max_messages, int)
            or max_messages < 0
        ):
            raise ValueError("max_messages must be a non-negative integer or None")

    if kind is SchedulerKind.FULL_SEND:
        selected = ordered
    else:
        assert budget_bytes is not None
        if kind is SchedulerKind.PERIODIC:
            ranked = tuple(
                candidate
                for candidate in ordered
                if candidate.message.sequence % period == phase
            )
        elif kind is SchedulerKind.RANDOM:
            rng = np.random.default_rng(seed)
            priorities = rng.random(len(ordered))
            ranked = tuple(
                candidate
                for _, candidate in sorted(
                    zip(priorities, ordered),
                    key=lambda item: (item[0], item[1].message.message_id),
                )
            )
        elif kind is SchedulerKind.FIFO_AOI:
            ranked = tuple(
                sorted(
                    ordered,
                    key=lambda candidate: (
                        -candidate.age_seconds,
                        candidate.network_transmit_time,
                        candidate.message.message_id,
                    ),
                )
            )
        elif kind is SchedulerKind.CONFIDENCE_TOP_K:
            ranked = tuple(
                sorted(
                    ordered,
                    key=lambda candidate: (
                        -candidate.confidence,
                        candidate.message.message_id,
                    ),
                )
            )
        else:
            ranked = tuple(sorted(ordered, key=lambda item: item.message.message_id))

        if kind is SchedulerKind.MARGINAL_VOI:
            selected = _exact_value_pack(
                ranked,
                budget_bytes=budget_bytes,
                budget_basis=basis,
                max_messages=max_messages,
            )
        else:
            selected = _ranked_pack(
                ranked,
                budget_bytes=budget_bytes,
                budget_basis=basis,
                max_messages=max_messages,
            )

    return BaselineScheduleResult(
        scheduler=kind,
        selected=selected,
        total_application_bytes=sum(
            candidate.application_byte_cost for candidate in selected
        ),
        total_on_wire_bytes=sum(candidate.on_wire_byte_cost for candidate in selected),
        expected_risk_reduction=sum(candidate.expected_value for candidate in selected),
        budget_bytes=budget_bytes,
        budget_basis=basis,
    )


def schedule_rate_limited_v1(
    candidates: list[ScheduleCandidate] | tuple[ScheduleCandidate, ...],
    scheduler: SchedulerKind | str,
    *,
    network_trace: NetworkTraceV1,
    wire_accounting: WireAccountingConfigV1,
    byte_rate_limit_per_second: int,
    burst_seconds: float = 1.0,
    seed: int = 0,
    period: int = 1,
    phase: int = 0,
    max_messages_per_decision: int | None = None,
) -> StreamScheduleResultV1:
    """Schedule a time-ordered stream under an on-wire token-bucket budget.

    The bucket refills at exactly ``byte_rate_limit_per_second`` and has a
    capacity of ``rate * burst_seconds``.  Candidates sharing a transmission
    time form one decision batch.  Selection causally reserves every bounded
    data attempt and, when ``ack_on_success`` is enabled, one possible ACK for
    every attempt.  Realised drop, ACK, and retransmission outcomes are read
    only after selection for the evidence ledger and fail-closed bound checks.
    Full-send remains an uncapped diagnostic and records every bucket overdraft
    as a budget violation.
    """

    kind = SchedulerKind(scheduler)
    if not isinstance(network_trace, NetworkTraceV1):
        raise TypeError("network_trace must be NetworkTraceV1")
    if not isinstance(wire_accounting, WireAccountingConfigV1):
        raise TypeError("wire_accounting must be WireAccountingConfigV1")
    expected_plan_sha256 = condition_plan_v1(
        network_trace.condition_id,
        wire_accounting=wire_accounting,
    ).content_sha256
    if network_trace.condition_plan_sha256 != expected_plan_sha256:
        raise ValueError(
            "network trace condition plan disagrees with wire_accounting"
        )
    if (
        isinstance(byte_rate_limit_per_second, bool)
        or not isinstance(byte_rate_limit_per_second, int)
        or byte_rate_limit_per_second <= 0
    ):
        raise ValueError("byte_rate_limit_per_second must be a positive integer")
    burst_seconds = float(burst_seconds)
    if not np.isfinite(burst_seconds) or burst_seconds <= 0.0:
        raise ValueError("burst_seconds must be finite and positive")
    ordered = _canonical_candidates(candidates)
    trace_events = {
        event.packet.message_id: event for event in network_trace.events
    }
    reservation_by_id: dict[str, int] = {}
    for candidate in ordered:
        event = trace_events.get(candidate.message.message_id)
        if event is None:
            raise ValueError("scheduler candidate is absent from NetworkTraceV1")
        packet = event.packet
        message = candidate.message
        if (
            packet.source != message.source
            or packet.sequence != message.sequence
            or packet.transmitted_at != candidate.network_transmit_time
            or packet.deadline != message.deadline
            or packet.encoded_bytes != candidate.application_byte_cost
        ):
            raise ValueError("scheduler candidate metadata disagrees with network trace")
        reservation = bounded_wire_reservation_v1(
            candidate.application_byte_cost,
            wire_accounting,
        ).total_on_wire_bytes
        if candidate.on_wire_byte_cost != reservation:
            raise ValueError(
                "scheduler projected on-wire cost disagrees with causal reservation"
            )
        reservation_by_id[candidate.message.message_id] = reservation
    if not ordered:
        return StreamScheduleResultV1(
            scheduler=kind,
            selected=(),
            total_application_bytes=0,
            total_reserved_on_wire_bytes=0,
            total_on_wire_bytes=0,
            byte_rate_limit_per_second=byte_rate_limit_per_second,
            burst_seconds=burst_seconds,
            network_trace_sha256=network_trace.content_sha256,
            budget_violations=0,
        )

    capacity = byte_rate_limit_per_second * burst_seconds
    tokens = capacity
    previous_time = ordered[0].network_transmit_time
    selected: list[ScheduleCandidate] = []
    violations = 0
    groups: dict[float, list[ScheduleCandidate]] = {}
    for candidate in ordered:
        groups.setdefault(candidate.network_transmit_time, []).append(candidate)
    for decision_index, (transmitted, group) in enumerate(sorted(groups.items())):
        elapsed = transmitted - previous_time
        if elapsed < -1e-12:  # pragma: no cover - canonical sort prevents this.
            raise ValueError("candidate transmission times must not move backwards")
        tokens = min(capacity, tokens + max(0.0, elapsed) * byte_rate_limit_per_second)
        result = schedule_baseline(
            tuple(group),
            kind,
            budget_bytes=max(0, int(np.floor(tokens))),
            budget_basis=BudgetBasis.ON_WIRE,
            seed=seed + decision_index,
            period=period,
            phase=phase,
            max_messages=max_messages_per_decision,
        )
        reserved = result.total_on_wire_bytes
        if reserved > tokens + 1e-12:
            violations += 1
        tokens = max(0.0, tokens - reserved)
        selected.extend(result.selected)
        previous_time = transmitted
    selected_tuple = tuple(selected)
    total_reserved = sum(
        reservation_by_id[candidate.message.message_id]
        for candidate in selected_tuple
    )
    total_realised = sum(
        trace_events[candidate.message.message_id].byte_account.total_on_wire_bytes
        for candidate in selected_tuple
    )
    violations += sum(
        trace_events[candidate.message.message_id].byte_account.total_on_wire_bytes
        > reservation_by_id[candidate.message.message_id]
        for candidate in selected_tuple
    )
    maximum_data_transmissions = wire_accounting.max_retransmissions + 1
    maximum_ack_transmissions = (
        maximum_data_transmissions if wire_accounting.ack_on_success else 0
    )
    for candidate in selected_tuple:
        account = trace_events[candidate.message.message_id].byte_account
        if (
            account.data_transmissions < 1
            or account.data_transmissions > maximum_data_transmissions
            or account.ack_transmissions > maximum_ack_transmissions
            or account.ack_transmissions > account.data_transmissions
        ):
            violations += 1
    accounting_duration = burst_seconds + (
        ordered[-1].network_transmit_time
        - ordered[0].network_transmit_time
    )
    if total_realised > byte_rate_limit_per_second * accounting_duration + 1e-12:
        violations += 1
    return StreamScheduleResultV1(
        scheduler=kind,
        selected=selected_tuple,
        total_application_bytes=sum(
            candidate.application_byte_cost for candidate in selected_tuple
        ),
        total_reserved_on_wire_bytes=total_reserved,
        total_on_wire_bytes=total_realised,
        byte_rate_limit_per_second=byte_rate_limit_per_second,
        burst_seconds=burst_seconds,
        network_trace_sha256=network_trace.content_sha256,
        budget_violations=violations,
    )


__all__ = [
    "BaselineScheduleResult",
    "BudgetBasis",
    "CausalScoreEvidenceV1",
    "ScheduleCandidate",
    "ScheduleResult",
    "SchedulerKind",
    "StreamScheduleResultV1",
    "schedule_baseline",
    "schedule_rate_limited_v1",
    "select_exact_budget",
]
