"""Deterministic sequence-level OOF split for SPD development experiments.

The official V2X-Seq-SPD ``test`` and ``test_A`` cohorts are deliberately
outside the EventTrack-V2X protocol.  Model and hyperparameter selection use
pooled out-of-fold predictions from the 46 official training sequences.  The
21 official validation sequences remain sealed until the confirmatory run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .wire import canonical_json_bytes


DEVELOPMENT_DATASET_ID_V1 = "v2x-seq-spd"
DEVELOPMENT_SPLIT_NAME_V1 = "train"
DEVELOPMENT_FOLD_COUNT_V1 = 5
DEVELOPMENT_SEQUENCE_COUNT_V1 = 46
DEVELOPMENT_FOLD_SALT_V1 = "eventtrack-v2x-spd-development-5fold-v1"
DEVELOPMENT_FOLD_SIZES_V1 = (10, 9, 9, 9, 9)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class DevelopmentSplitError(ValueError):
    """Raised when a development fold manifest is malformed or noncanonical."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise DevelopmentSplitError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DevelopmentSplitError(f"{name} must be a lowercase SHA-256")
    return value


def _sequence_ids(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise DevelopmentSplitError(f"{name} must be an array")
    result = tuple(_identifier(value, f"{name} item") for value in values)
    if result != tuple(sorted(set(result))):
        raise DevelopmentSplitError(f"{name} must be unique and sorted")
    return result


def _rank_key(sequence_id: str, split_sha256: str) -> tuple[str, str]:
    payload = {
        "salt": DEVELOPMENT_FOLD_SALT_V1,
        "sequence_id": sequence_id,
        "split_sha256": split_sha256,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest(), sequence_id


def _expected_folds(
    sequence_ids: tuple[str, ...], split_sha256: str
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    ranked = tuple(sorted(sequence_ids, key=lambda item: _rank_key(item, split_sha256)))
    result: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    full = frozenset(sequence_ids)
    for fold_index in range(DEVELOPMENT_FOLD_COUNT_V1):
        held_out = tuple(
            sorted(
                sequence_id
                for position, sequence_id in enumerate(ranked)
                if position % DEVELOPMENT_FOLD_COUNT_V1 == fold_index
            )
        )
        fit = tuple(sorted(full - frozenset(held_out)))
        result.append((fit, held_out))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DevelopmentFoldV1:
    """One fit/held-out partition in the deterministic OOF protocol."""

    fold_id: int
    fit_sequence_ids: tuple[str, ...]
    held_out_sequence_ids: tuple[str, ...]

    _FIELDS = frozenset(
        {"fit_sequence_ids", "fold_id", "held_out_sequence_ids"}
    )

    def __post_init__(self) -> None:
        if type(self.fold_id) is not int or not 0 <= self.fold_id < DEVELOPMENT_FOLD_COUNT_V1:
            raise DevelopmentSplitError("fold_id must be an integer in [0, 5)")
        fit = _sequence_ids(self.fit_sequence_ids, "fit_sequence_ids")
        held_out = _sequence_ids(self.held_out_sequence_ids, "held_out_sequence_ids")
        if not fit or not held_out:
            raise DevelopmentSplitError("fit and held-out cohorts must be non-empty")
        if set(fit).intersection(held_out):
            raise DevelopmentSplitError("fit and held-out cohorts must not overlap")
        object.__setattr__(self, "fit_sequence_ids", fit)
        object.__setattr__(self, "held_out_sequence_ids", held_out)

    def to_primitive(self) -> dict[str, object]:
        return {
            "fit_sequence_ids": list(self.fit_sequence_ids),
            "fold_id": self.fold_id,
            "held_out_sequence_ids": list(self.held_out_sequence_ids),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "DevelopmentFoldV1":
        if not isinstance(value, Mapping) or frozenset(value) != cls._FIELDS:
            raise DevelopmentSplitError(
                "DevelopmentFoldV1 has missing or unknown fields"
            )
        return cls(
            fold_id=value["fold_id"],  # type: ignore[arg-type]
            fit_sequence_ids=tuple(value["fit_sequence_ids"]),  # type: ignore[arg-type]
            held_out_sequence_ids=tuple(value["held_out_sequence_ids"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DevelopmentSplitManifestV1:
    """Canonical 5-fold sequence assignment for official SPD train."""

    dataset_id: str
    split_name: str
    split_sha256: str
    sequence_ids: tuple[str, ...]
    folds: tuple[DevelopmentFoldV1, ...]

    _PAYLOAD_FIELDS = frozenset(
        {
            "dataset_id",
            "fold_count",
            "fold_salt",
            "folds",
            "kind",
            "schema_version",
            "sequence_ids",
            "split_name",
            "split_sha256",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _identifier(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "split_name", _identifier(self.split_name, "split_name"))
        object.__setattr__(self, "split_sha256", _sha256(self.split_sha256, "split_sha256"))
        sequences = _sequence_ids(self.sequence_ids, "sequence_ids")
        if self.dataset_id != DEVELOPMENT_DATASET_ID_V1:
            raise DevelopmentSplitError("dataset_id must be v2x-seq-spd")
        if self.split_name != DEVELOPMENT_SPLIT_NAME_V1:
            raise DevelopmentSplitError("development split_name must be train")
        if len(sequences) != DEVELOPMENT_SEQUENCE_COUNT_V1:
            raise DevelopmentSplitError(
                "development sequence_ids must contain exactly 46 sequences"
            )
        if not isinstance(self.folds, (list, tuple)) or not all(
            isinstance(item, DevelopmentFoldV1) for item in self.folds
        ):
            raise DevelopmentSplitError("folds must contain DevelopmentFoldV1")
        folds = tuple(self.folds)
        if tuple(item.fold_id for item in folds) != tuple(
            range(DEVELOPMENT_FOLD_COUNT_V1)
        ):
            raise DevelopmentSplitError("folds must contain canonical fold IDs 0..4")
        expected = _expected_folds(sequences, self.split_sha256)
        observed = tuple(
            (item.fit_sequence_ids, item.held_out_sequence_ids) for item in folds
        )
        if observed != expected:
            raise DevelopmentSplitError(
                "fold assignments do not match the fixed hash-sort round-robin rule"
            )
        if tuple(len(item.held_out_sequence_ids) for item in folds) != DEVELOPMENT_FOLD_SIZES_V1:
            raise DevelopmentSplitError("development fold sizes must be 10/9/9/9/9")
        held_out_union = tuple(
            sorted(sequence_id for fold in folds for sequence_id in fold.held_out_sequence_ids)
        )
        if held_out_union != sequences:
            raise DevelopmentSplitError(
                "every development sequence must be held out exactly once"
            )
        object.__setattr__(self, "sequence_ids", sequences)
        object.__setattr__(self, "folds", folds)

    def fold_for_sequence(self, sequence_id: str) -> int:
        sequence = _identifier(sequence_id, "sequence_id")
        for fold in self.folds:
            if sequence in fold.held_out_sequence_ids:
                return fold.fold_id
        raise DevelopmentSplitError("sequence_id is outside the development cohort")

    def payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "fold_count": DEVELOPMENT_FOLD_COUNT_V1,
            "fold_salt": DEVELOPMENT_FOLD_SALT_V1,
            "folds": [item.to_primitive() for item in self.folds],
            "kind": "development_split_manifest_v1",
            "schema_version": 1,
            "sequence_ids": list(self.sequence_ids),
            "split_name": self.split_name,
            "split_sha256": self.split_sha256,
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()

    def sealed_document(self) -> dict[str, object]:
        return {**self.payload(), "content_sha256": self.content_sha256}

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.sealed_document())

    @classmethod
    def from_mapping(cls, value: object) -> "DevelopmentSplitManifestV1":
        if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
            raise DevelopmentSplitError("development manifest must be a string-keyed object")
        if frozenset(value) != cls._PAYLOAD_FIELDS | {"content_sha256"}:
            raise DevelopmentSplitError("development manifest has missing or unknown fields")
        if (
            value["kind"] != "development_split_manifest_v1"
            or value["schema_version"] != 1
            or value["fold_count"] != DEVELOPMENT_FOLD_COUNT_V1
            or value["fold_salt"] != DEVELOPMENT_FOLD_SALT_V1
        ):
            raise DevelopmentSplitError("unsupported development split manifest schema")
        observed = _sha256(value["content_sha256"], "content_sha256")
        payload = {key: value[key] for key in cls._PAYLOAD_FIELDS}
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if observed != expected:
            raise DevelopmentSplitError("development manifest content SHA-256 mismatch")
        raw_folds = value["folds"]
        if not isinstance(raw_folds, list):
            raise DevelopmentSplitError("folds must be an array")
        return cls(
            dataset_id=value["dataset_id"],  # type: ignore[arg-type]
            split_name=value["split_name"],  # type: ignore[arg-type]
            split_sha256=value["split_sha256"],  # type: ignore[arg-type]
            sequence_ids=tuple(value["sequence_ids"]),  # type: ignore[arg-type]
            folds=tuple(DevelopmentFoldV1.from_mapping(item) for item in raw_folds),
        )


def build_development_split_manifest_v1(
    sequence_ids: Iterable[str], *, split_sha256: str
) -> DevelopmentSplitManifestV1:
    """Build the one permitted SPD development assignment."""

    sequences = tuple(sorted(sequence_ids))
    split_digest = _sha256(split_sha256, "split_sha256")
    folds = tuple(
        DevelopmentFoldV1(
            fold_id=fold_id,
            fit_sequence_ids=fit,
            held_out_sequence_ids=held_out,
        )
        for fold_id, (fit, held_out) in enumerate(
            _expected_folds(sequences, split_digest)
        )
    )
    return DevelopmentSplitManifestV1(
        dataset_id=DEVELOPMENT_DATASET_ID_V1,
        split_name=DEVELOPMENT_SPLIT_NAME_V1,
        split_sha256=split_digest,
        sequence_ids=sequences,
        folds=folds,
    )


def decode_development_split_manifest_v1(
    data: bytes,
) -> DevelopmentSplitManifestV1:
    if type(data) is not bytes:
        raise TypeError("development manifest data must be bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise DevelopmentSplitError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentSplitError("invalid development manifest JSON") from exc
    manifest = DevelopmentSplitManifestV1.from_mapping(value)
    if manifest.canonical_bytes != data:
        raise DevelopmentSplitError("development manifest JSON is not canonical")
    return manifest


__all__ = [
    "DEVELOPMENT_DATASET_ID_V1",
    "DEVELOPMENT_FOLD_COUNT_V1",
    "DEVELOPMENT_FOLD_SALT_V1",
    "DEVELOPMENT_FOLD_SIZES_V1",
    "DEVELOPMENT_SEQUENCE_COUNT_V1",
    "DEVELOPMENT_SPLIT_NAME_V1",
    "DevelopmentFoldV1",
    "DevelopmentSplitError",
    "DevelopmentSplitManifestV1",
    "build_development_split_manifest_v1",
    "decode_development_split_manifest_v1",
]
