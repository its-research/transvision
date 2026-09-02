"""Append-only, hash-chained committed output snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .wire import canonical_json_bytes


GENESIS_HASH = "0" * 64


@dataclass(frozen=True, slots=True)
class CommitRecord:
    index: int
    decision_time: float
    snapshot_bytes: bytes
    message_ids: tuple[str, ...]
    previous_hash: str
    record_hash: str

    @property
    def snapshot(self) -> Any:
        return json.loads(self.snapshot_bytes.decode("utf-8"))


class AppendOnlyCommitLog:
    """In-memory reference implementation of the immutable commit contract."""

    def __init__(self) -> None:
        self._records: list[CommitRecord] = []

    @property
    def records(self) -> tuple[CommitRecord, ...]:
        return tuple(self._records)

    def commit(
        self,
        *,
        decision_time: float,
        snapshot: Any,
        message_ids: tuple[str, ...] | list[str] = (),
    ) -> CommitRecord:
        decision_time = float(decision_time)
        if not np.isfinite(decision_time):
            raise ValueError("decision_time must be finite")
        if self._records and decision_time <= self._records[-1].decision_time:
            raise ValueError("commit decision_time must be strictly increasing")
        identifiers = tuple(sorted(message_ids))
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError("message_ids must be non-empty strings")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("message_ids must be unique")
        snapshot_bytes = canonical_json_bytes(snapshot)
        index = len(self._records)
        previous_hash = self.prefix_hash()
        body = canonical_json_bytes(
            {
                "decision_time": decision_time,
                "index": index,
                "message_ids": identifiers,
                "previous_hash": previous_hash,
                "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            }
        )
        record_hash = hashlib.sha256(body).hexdigest()
        record = CommitRecord(
            index=index,
            decision_time=decision_time,
            snapshot_bytes=snapshot_bytes,
            message_ids=identifiers,
            previous_hash=previous_hash,
            record_hash=record_hash,
        )
        self._records.append(record)
        return record

    def prefix_hash(self, length: int | None = None) -> str:
        if length is None:
            length = len(self._records)
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError("prefix length must be an integer")
        if not 0 <= length <= len(self._records):
            raise ValueError("prefix length is outside the commit log")
        return GENESIS_HASH if length == 0 else self._records[length - 1].record_hash

    def verify(self) -> bool:
        previous_hash = GENESIS_HASH
        previous_time = -np.inf
        for index, record in enumerate(self._records):
            if (
                record.index != index
                or record.previous_hash != previous_hash
                or record.decision_time <= previous_time
            ):
                return False
            body = canonical_json_bytes(
                {
                    "decision_time": record.decision_time,
                    "index": record.index,
                    "message_ids": record.message_ids,
                    "previous_hash": record.previous_hash,
                    "snapshot_sha256": hashlib.sha256(record.snapshot_bytes).hexdigest(),
                }
            )
            if hashlib.sha256(body).hexdigest() != record.record_hash:
                return False
            previous_hash = record.record_hash
            previous_time = record.decision_time
        return True


__all__ = ["AppendOnlyCommitLog", "CommitRecord", "GENESIS_HASH"]
