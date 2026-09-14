"""Causal, replay-stable message memory for RTP-V2X.

The memory separates event time from complete-message arrival time.  A
snapshot at decision time ``t`` can only contain messages whose event and
arrival times are both no later than ``t``.  Inserting a late message therefore
cannot rewrite an earlier snapshot.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
PAYLOAD_KINDS = frozenset({"bev", "query", "trajectory", "signal"})


class CausalMemoryError(ValueError):
    """Raised when message identity or causal time semantics are ambiguous."""


def _safe_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise CausalMemoryError(f"{label} must be a safe non-empty identifier")
    return value


def _integer_time(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CausalMemoryError(f"{label} must be a non-negative integer")
    return value


def _finite_unit_interval(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalMemoryError(f"{label} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise CausalMemoryError(f"{label} exceeds the numeric bound") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CausalMemoryError(f"{label} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """One content-addressed message available to the online method.

    ``encoded_bytes`` includes the protocol header and is measured after the
    exact encoder used by the transport.  ``payload_sha256`` binds the decoded
    payload without retaining dataset bytes in this protocol object.
    """

    message_id: str
    agent_id: str
    payload_kind: str
    event_time: int
    arrival_time: int
    encoded_bytes: int
    payload_sha256: str
    reliability: float

    def __post_init__(self) -> None:
        _safe_id(self.message_id, label="message_id")
        _safe_id(self.agent_id, label="agent_id")
        if self.payload_kind not in PAYLOAD_KINDS:
            raise CausalMemoryError(
                f"payload_kind must be one of {sorted(PAYLOAD_KINDS)}"
            )
        _integer_time(self.event_time, label="event_time")
        _integer_time(self.arrival_time, label="arrival_time")
        if (
            isinstance(self.encoded_bytes, bool)
            or not isinstance(self.encoded_bytes, int)
            or self.encoded_bytes <= 0
        ):
            raise CausalMemoryError("encoded_bytes must be a positive integer")
        if (
            not isinstance(self.payload_sha256, str)
            or SHA256_RE.fullmatch(self.payload_sha256) is None
        ):
            raise CausalMemoryError(
                "payload_sha256 must be 64 lowercase hexadecimal characters"
            )
        _finite_unit_interval(self.reliability, label="reliability")


class CausalMessageMemory:
    """Store messages by stable identity and expose immutable causal snapshots."""

    def __init__(
        self,
        *,
        max_age: int,
        max_messages_per_agent: int,
        max_total_records: int = 1_000_000,
    ) -> None:
        self._max_age = _integer_time(max_age, label="max_age")
        if self._max_age == 0:
            raise CausalMemoryError("max_age must be positive")
        if (
            isinstance(max_messages_per_agent, bool)
            or not isinstance(max_messages_per_agent, int)
            or max_messages_per_agent <= 0
        ):
            raise CausalMemoryError("max_messages_per_agent must be a positive integer")
        if (
            isinstance(max_total_records, bool)
            or not isinstance(max_total_records, int)
            or max_total_records <= 0
        ):
            raise CausalMemoryError("max_total_records must be a positive integer")
        self._max_messages_per_agent = max_messages_per_agent
        self._max_total_records = max_total_records
        self._messages: dict[str, MessageRecord] = {}
        self._latest_committed_decision: int | None = None

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def latest_committed_decision(self) -> int | None:
        return self._latest_committed_decision

    def ingest(self, message: MessageRecord) -> bool:
        """Insert one message.

        Returns ``True`` for a new identity and ``False`` for a byte-identical
        replay.  Reusing an identity with different metadata is rejected.
        """

        if not isinstance(message, MessageRecord):
            raise CausalMemoryError("message must be a MessageRecord")
        existing = self._messages.get(message.message_id)
        if existing is not None:
            if existing == message:
                return False
            raise CausalMemoryError(
                f"message_id {message.message_id!r} was reused with different content"
            )
        if (
            self._latest_committed_decision is not None
            and message.arrival_time <= self._latest_committed_decision
        ):
            raise CausalMemoryError(
                "a newly observed message cannot backfill an already committed "
                "arrival boundary"
            )
        if len(self._messages) >= self._max_total_records:
            raise CausalMemoryError(
                "message memory reached max_total_records before safe pruning"
            )
        self._messages[message.message_id] = message
        return True

    def ingest_many(self, messages: Iterable[MessageRecord]) -> int:
        inserted = 0
        for message in messages:
            inserted += int(self.ingest(message))
        return inserted

    def snapshot(
        self,
        *,
        decision_time: int,
    ) -> tuple[MessageRecord, ...]:
        """Return the deterministic message view available at ``decision_time``.

        Age is measured from event time.  Capacity is applied independently to
        each agent after causal filtering, across all payload kinds.
        """

        decision = _integer_time(decision_time, label="decision_time")
        streams: dict[str, list[MessageRecord]] = {}
        for message in self._messages.values():
            if message.arrival_time > decision or message.event_time > decision:
                continue
            if decision - message.event_time > self._max_age:
                continue
            streams.setdefault(message.agent_id, []).append(message)

        selected: list[MessageRecord] = []
        for key in sorted(streams):
            rows = sorted(
                streams[key],
                key=lambda item: (
                    item.arrival_time,
                    item.event_time,
                    item.message_id,
                ),
                reverse=True,
            )[: self._max_messages_per_agent]
            selected.extend(rows)
        return tuple(
            sorted(
                selected,
                key=lambda item: (
                    item.arrival_time,
                    item.event_time,
                    item.agent_id,
                    item.payload_kind,
                    item.message_id,
                ),
            )
        )

    def commit_output(self, *, decision_time: int) -> None:
        """Record an emitted output boundary without changing stored messages."""

        decision = _integer_time(decision_time, label="decision_time")
        if (
            self._latest_committed_decision is not None
            and decision <= self._latest_committed_decision
        ):
            raise CausalMemoryError("committed decision times must increase strictly")
        self._latest_committed_decision = decision
