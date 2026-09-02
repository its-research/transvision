"""Deadline-conditioned exact-byte additive proxy scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import WireMessage
from .wire import encode_message


@dataclass(frozen=True, slots=True)
class ScheduleCandidate:
    message: WireMessage
    risk_reduction: float
    on_time_probability: float

    def __post_init__(self) -> None:
        risk = float(self.risk_reduction)
        probability = float(self.on_time_probability)
        if not np.isfinite(risk) or risk < 0.0:
            raise ValueError("risk_reduction must be finite and non-negative")
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("on_time_probability must be in [0, 1]")
        object.__setattr__(self, "risk_reduction", risk)
        object.__setattr__(self, "on_time_probability", probability)

    @property
    def encoded_bytes(self) -> bytes:
        return encode_message(self.message)

    @property
    def byte_cost(self) -> int:
        return len(self.encoded_bytes)

    @property
    def expected_value(self) -> float:
        return self.risk_reduction * self.on_time_probability


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    selected: tuple[ScheduleCandidate, ...]
    total_bytes: int
    expected_risk_reduction: float

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(candidate.message.message_id for candidate in self.selected)


def select_exact_budget(
    candidates: list[ScheduleCandidate] | tuple[ScheduleCandidate, ...],
    *,
    budget_bytes: int,
) -> ScheduleResult:
    """Solve the frozen additive 0-1 proxy exactly in actual wire bytes."""

    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
        raise TypeError("budget_bytes must be an integer")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    ordered = tuple(sorted(candidates, key=lambda item: item.message.message_id))
    identifiers = [candidate.message.message_id for candidate in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate message IDs must be unique")

    # used bytes -> (expected value, selected candidate indices)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, candidate in enumerate(ordered):
        cost = candidate.byte_cost
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
    )


__all__ = ["ScheduleCandidate", "ScheduleResult", "select_exact_budget"]
