"""Exact byte-budgeted, deadline-aware message selection for RTP-V2X."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence


SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GRANULARITIES = frozenset({"bev", "query", "trajectory"})
SOLVER_ID = "exact-sparse-pareto-dp-v1"
MAX_CANDIDATES = 256
MAX_FRONTIER_STATES = 65_536
MAX_BYTE_COUNT = 2**63 - 1
MAX_UTILITY = 1e100


class SelectionError(ValueError):
    """Raised when communication candidates are not directly comparable."""


def _as_float(value: int | float, *, label: str) -> float:
    try:
        return float(value)
    except (OverflowError, ValueError) as exc:
        raise SelectionError(f"{label} exceeds the numeric bound") from exc


@dataclass(frozen=True, slots=True)
class CommunicationCandidate:
    message_id: str
    granularity: str
    encoded_bytes: int
    net_utility: float
    predicted_arrival_time: int
    deadline: int
    wire_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.message_id, str)
            or SAFE_ID_RE.fullmatch(self.message_id) is None
        ):
            raise SelectionError("message_id must be a safe non-empty identifier")
        if self.granularity not in GRANULARITIES:
            raise SelectionError(f"granularity must be one of {sorted(GRANULARITIES)}")
        if (
            isinstance(self.encoded_bytes, bool)
            or not isinstance(self.encoded_bytes, int)
            or self.encoded_bytes <= 0
            or self.encoded_bytes > MAX_BYTE_COUNT
        ):
            raise SelectionError(
                "encoded_bytes must be a positive signed 64-bit integer"
            )
        if isinstance(self.net_utility, bool) or not isinstance(
            self.net_utility, (int, float)
        ):
            raise SelectionError("net_utility is outside the finite numeric bound")
        net_utility = _as_float(self.net_utility, label="net_utility")
        if not math.isfinite(net_utility) or abs(net_utility) > MAX_UTILITY:
            raise SelectionError("net_utility is outside the finite numeric bound")
        for name in ("predicted_arrival_time", "deadline"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SelectionError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.wire_sha256, str)
            or SHA256_RE.fullmatch(self.wire_sha256) is None
        ):
            raise SelectionError(
                "wire_sha256 must be 64 lowercase hexadecimal characters"
            )


@dataclass(frozen=True, slots=True)
class CommunicationSelection:
    selected: tuple[CommunicationCandidate, ...]
    rejected: tuple[tuple[str, str], ...]
    bytes_used: int
    total_utility: float
    candidate_count: int
    eligible_count: int
    frontier_peak: int
    solver_id: str = SOLVER_ID


def _better_state(
    left: tuple[float, tuple[str, ...]],
    right: tuple[float, tuple[str, ...]],
) -> bool:
    """Return whether ``left`` is preferable to ``right`` deterministically."""

    if left[0] != right[0]:
        return left[0] > right[0]
    return left[1] < right[1]


def _prune_dominated_states(
    states: dict[int, tuple[float, tuple[str, ...]]],
) -> dict[int, tuple[float, tuple[str, ...]]]:
    """Remove states using more bytes without a strictly larger utility."""

    result: dict[int, tuple[float, tuple[str, ...]]] = {}
    best_utility = -math.inf
    for used in sorted(states):
        state = states[used]
        if state[0] <= best_utility:
            continue
        result[used] = state
        best_utility = state[0]
    return result


def select_budgeted_messages(
    candidates: Sequence[CommunicationCandidate], *, budget_bytes: int
) -> CommunicationSelection:
    """Select the maximum-utility on-time subset under an exact byte budget.

    The dynamic program uses measured ``encoded_bytes`` and never substitutes
    tensor element counts or a compression estimate.  Non-positive utility and
    predicted deadline misses are rejected before optimization.
    """

    if (
        isinstance(budget_bytes, bool)
        or not isinstance(budget_bytes, int)
        or budget_bytes < 0
        or budget_bytes > MAX_BYTE_COUNT
    ):
        raise SelectionError(
            "budget_bytes must be a non-negative signed 64-bit integer"
        )
    rows = list(candidates)
    if any(not isinstance(row, CommunicationCandidate) for row in rows):
        raise SelectionError("every candidate must be a CommunicationCandidate")
    if len(rows) > MAX_CANDIDATES:
        raise SelectionError(
            f"candidate count exceeds exact solver limit {MAX_CANDIDATES}"
        )
    identifiers = [row.message_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise SelectionError("message_id values must be unique")

    eligible: list[CommunicationCandidate] = []
    rejected: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: item.message_id):
        if row.predicted_arrival_time > row.deadline:
            rejected[row.message_id] = "predicted_deadline_miss"
        elif row.net_utility <= 0.0:
            rejected[row.message_id] = "non_positive_utility"
        elif row.encoded_bytes > budget_bytes:
            rejected[row.message_id] = "larger_than_budget"
        else:
            eligible.append(row)

    by_id = {row.message_id: row for row in eligible}
    # byte_count -> (utility, sorted message IDs)
    states: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, ())}
    frontier_peak = 1
    for row in eligible:
        updated = dict(states)
        for used, state in states.items():
            new_used = used + row.encoded_bytes
            if new_used > budget_bytes:
                continue
            candidate_state = (
                state[0] + float(row.net_utility),
                tuple(sorted((*state[1], row.message_id))),
            )
            if not math.isfinite(candidate_state[0]) or abs(candidate_state[0]) > (
                MAX_UTILITY * MAX_CANDIDATES
            ):
                raise SelectionError("aggregate utility exceeds numeric bound")
            current = updated.get(new_used)
            if current is None or _better_state(candidate_state, current):
                updated[new_used] = candidate_state
        states = _prune_dominated_states(updated)
        frontier_peak = max(frontier_peak, len(states))
        if len(states) > MAX_FRONTIER_STATES:
            raise SelectionError(
                "exact solver frontier exceeds the frozen operational limit"
            )

    best_bytes = 0
    best_state = states[0]
    for used in sorted(states):
        state = states[used]
        if state[0] > best_state[0]:
            best_bytes = used
            best_state = state
        elif state[0] == best_state[0] and (
            used < best_bytes or (used == best_bytes and state[1] < best_state[1])
        ):
            best_bytes = used
            best_state = state

    selected_ids = set(best_state[1])
    for row in eligible:
        if row.message_id not in selected_ids:
            rejected[row.message_id] = "not_selected_by_exact_budget_optimization"
    selected = tuple(by_id[message_id] for message_id in best_state[1])
    observed_bytes = sum(row.encoded_bytes for row in selected)
    if observed_bytes != best_bytes or observed_bytes > budget_bytes:
        raise SelectionError("internal byte accounting mismatch")
    return CommunicationSelection(
        selected=selected,
        rejected=tuple(sorted(rejected.items())),
        bytes_used=observed_bytes,
        total_utility=float(best_state[0]),
        candidate_count=len(rows),
        eligible_count=len(eligible),
        frontier_peak=frontier_peak,
    )
