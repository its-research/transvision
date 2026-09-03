"""Scientific dataset-release and independent cold-read attestation contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

from .wire import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
OFFICIAL_RELEASE_STATUS_V1 = "independently-authenticated-official-release"


class DatasetReleaseError(ValueError):
    """Raised when data identity or cold-cache evidence is insufficient."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise DatasetReleaseError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DatasetReleaseError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_mapping(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise DatasetReleaseError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        raise DatasetReleaseError(f"{name} has missing or unknown fields")
    return value


@dataclass(frozen=True, slots=True)
class DatasetReleaseReceiptV1:
    """Hash binding for official source bytes and a fresh ClearML readback."""

    receipt_id: str
    dataset_id: str
    split_name: str
    dataset_manifest_sha256: str
    official_inventory_sha256: str
    license_evidence_sha256: str
    split_sha256: str
    cohort_sha256: str
    frame_contract_sha256: str
    sequence_ids: tuple[str, ...]
    clearml_dataset_id: str
    clearml_project_id: str
    clearml_version: str
    cold_cache_verification_sha256: str
    release_identity_status: str
    license_use_authorized: bool
    source_bytes_verified: bool
    cold_cache_verified: bool
    scientific_claims_allowed: bool

    _FIELDS = frozenset(
        {
            "clearml_dataset_id",
            "clearml_project_id",
            "clearml_version",
            "cohort_sha256",
            "cold_cache_verification_sha256",
            "cold_cache_verified",
            "dataset_id",
            "dataset_manifest_sha256",
            "frame_contract_sha256",
            "kind",
            "license_evidence_sha256",
            "license_use_authorized",
            "official_inventory_sha256",
            "receipt_id",
            "release_identity_status",
            "schema_version",
            "scientific_claims_allowed",
            "sequence_ids",
            "source_bytes_verified",
            "split_name",
            "split_sha256",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "receipt_id",
            "dataset_id",
            "split_name",
            "clearml_dataset_id",
            "clearml_project_id",
            "clearml_version",
            "release_identity_status",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "dataset_manifest_sha256",
            "official_inventory_sha256",
            "license_evidence_sha256",
            "split_sha256",
            "cohort_sha256",
            "frame_contract_sha256",
            "cold_cache_verification_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if not isinstance(self.sequence_ids, (list, tuple)):
            raise TypeError("sequence_ids must be an array")
        sequence_ids = tuple(
            _identifier(item, "sequence_id") for item in self.sequence_ids
        )
        if not sequence_ids or sequence_ids != tuple(sorted(set(sequence_ids))):
            raise DatasetReleaseError("sequence_ids must be non-empty, unique, and sorted")
        object.__setattr__(self, "sequence_ids", sequence_ids)
        for name in (
            "license_use_authorized",
            "source_bytes_verified",
            "cold_cache_verified",
            "scientific_claims_allowed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.release_identity_status != OFFICIAL_RELEASE_STATUS_V1:
            raise DatasetReleaseError(
                "dataset release is not independently authenticated as official"
            )
        if not (
            self.license_use_authorized
            and self.source_bytes_verified
            and self.cold_cache_verified
            and self.scientific_claims_allowed
        ):
            raise DatasetReleaseError(
                "dataset release receipt requires verified source bytes, cold cache, "
                "and scientific claim authorization"
            )
        if self.dataset_id == "v2x-seq-spd":
            expected_counts = {"train": 46, "val": 21}
            if self.split_name not in expected_counts:
                raise DatasetReleaseError(
                    "EventTrack-V2X excludes V2X-Seq-SPD test and test_A; "
                    "release receipts may bind only train or val"
                )
            expected_count = expected_counts[self.split_name]
            if len(sequence_ids) != expected_count:
                raise DatasetReleaseError(
                    f"V2X-Seq-SPD {self.split_name} receipt requires exactly "
                    f"{expected_count} sequences"
                )
        elif self.dataset_id == "griffin-25m" and self.split_name != "val":
            raise DatasetReleaseError("Griffin external receipt requires val split")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "clearml_dataset_id": self.clearml_dataset_id,
            "clearml_project_id": self.clearml_project_id,
            "clearml_version": self.clearml_version,
            "cohort_sha256": self.cohort_sha256,
            "cold_cache_verification_sha256": self.cold_cache_verification_sha256,
            "cold_cache_verified": self.cold_cache_verified,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "frame_contract_sha256": self.frame_contract_sha256,
            "kind": "dataset_release_receipt_v1",
            "license_evidence_sha256": self.license_evidence_sha256,
            "license_use_authorized": self.license_use_authorized,
            "official_inventory_sha256": self.official_inventory_sha256,
            "receipt_id": self.receipt_id,
            "release_identity_status": self.release_identity_status,
            "schema_version": 1,
            "scientific_claims_allowed": self.scientific_claims_allowed,
            "sequence_ids": list(self.sequence_ids),
            "source_bytes_verified": self.source_bytes_verified,
            "split_name": self.split_name,
            "split_sha256": self.split_sha256,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "DatasetReleaseReceiptV1":
        result = _strict_mapping(value, cls._FIELDS, cls.__name__)
        if result["kind"] != "dataset_release_receipt_v1" or result["schema_version"] != 1:
            raise DatasetReleaseError("unsupported dataset release receipt schema")
        return cls(
            receipt_id=result["receipt_id"],
            dataset_id=result["dataset_id"],
            split_name=result["split_name"],
            dataset_manifest_sha256=result["dataset_manifest_sha256"],
            official_inventory_sha256=result["official_inventory_sha256"],
            license_evidence_sha256=result["license_evidence_sha256"],
            split_sha256=result["split_sha256"],
            cohort_sha256=result["cohort_sha256"],
            frame_contract_sha256=result["frame_contract_sha256"],
            sequence_ids=tuple(result["sequence_ids"]),  # type: ignore[arg-type]
            clearml_dataset_id=result["clearml_dataset_id"],
            clearml_project_id=result["clearml_project_id"],
            clearml_version=result["clearml_version"],
            cold_cache_verification_sha256=result[
                "cold_cache_verification_sha256"
            ],
            release_identity_status=result["release_identity_status"],
            license_use_authorized=result["license_use_authorized"],
            source_bytes_verified=result["source_bytes_verified"],
            cold_cache_verified=result["cold_cache_verified"],
            scientific_claims_allowed=result["scientific_claims_allowed"],
        )


def decode_dataset_release_receipt(data: bytes) -> DatasetReleaseReceiptV1:
    """Decode exact canonical bytes for a future official-release verifier."""

    if type(data) is not bytes:
        raise TypeError("dataset release receipt data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DatasetReleaseError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                DatasetReleaseError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetReleaseError("invalid dataset release receipt JSON") from exc
    receipt = DatasetReleaseReceiptV1.from_mapping(value)
    if receipt.canonical_bytes != data:
        raise DatasetReleaseError("dataset release receipt JSON is not canonical")
    return receipt


def require_scientific_clearml_verification(value: Mapping[str, object]) -> str:
    """Reject the current smoke verifier and return a formal receipt digest.

    The existing local-mirror verifier intentionally emits
    ``scientific_claims_allowed=false``.  This bridge therefore cannot turn a
    byte-identical smoke mirror into publication evidence.
    """

    if value.get("scientific_claims_allowed") is not True:
        raise DatasetReleaseError(
            "ClearML verification does not authorize scientific claims"
        )
    if value.get("cold_cache_verified") is not True or value.get(
        "byte_readback_verified"
    ) is not True:
        raise DatasetReleaseError("ClearML verification is not a complete cold read")
    raw = canonical_json_bytes(dict(value))
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "DatasetReleaseError",
    "DatasetReleaseReceiptV1",
    "OFFICIAL_RELEASE_STATUS_V1",
    "decode_dataset_release_receipt",
    "require_scientific_clearml_verification",
]
