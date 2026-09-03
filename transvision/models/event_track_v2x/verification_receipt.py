"""Trusted detached receipt for the independent publication verifier.

The receipt is not a substitute for verification.  It is the signed output of
an independent process that has cold-read the evidence, recomputed evaluator
outputs and wire ledgers, and executed the preregistered invariant suite.  The
publication gate accepts it only under an out-of-band trusted public-key hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from .publication_gate import REQUIRED_INVARIANTS_V1
from .results_registry import RunResultRegistryV1
from .wire import canonical_json_bytes


_HEX_32 = re.compile(r"[0-9a-f]{64}")
_HEX_64 = re.compile(r"[0-9a-f]{128}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class VerificationReceiptError(ValueError):
    """Raised when independent verification evidence is malformed or untrusted."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise VerificationReceiptError(f"{name} must be a canonical identifier")
    return value


def _hex(value: object, pattern: re.Pattern[str], name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise VerificationReceiptError(f"{name} is not canonical lowercase hex")
    return value


def _verified_at(value: object) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise VerificationReceiptError("verified_at_utc must be RFC3339 UTC")
    try:
        observed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise VerificationReceiptError("verified_at_utc must be RFC3339 UTC") from exc
    if observed.utcoffset() is None or observed.utcoffset().total_seconds() != 0:
        raise VerificationReceiptError("verified_at_utc must use UTC")
    return value


def _strict_mapping(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise VerificationReceiptError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        raise VerificationReceiptError(f"{name} has missing or unknown fields")
    return value


def run_evidence_inventory_sha256(registry: RunResultRegistryV1) -> str:
    """Hash every semantic run/evidence binding independently of cell ordering."""

    if not isinstance(registry, RunResultRegistryV1):
        raise TypeError("registry must be RunResultRegistryV1")
    inventory = [
        {
            "channel_outcome_trace_sha256": cell.channel_outcome_trace_sha256,
            "checkpoint_sha256": cell.checkpoint_sha256,
            "evidence_bundle_sha256": cell.evidence_bundle_sha256,
            "key": list(cell.key),
            "metrics_artifact_sha256": cell.metrics_artifact_sha256,
            "network_trace_sha256": cell.network_trace_sha256,
            "on_wire_bytes_total": cell.on_wire_bytes_total,
            "wire_ledger_sha256": cell.wire_ledger_sha256,
            "wire_measurement_duration_seconds": (
                cell.wire_measurement_duration_seconds
            ),
        }
        for cell in registry.cells
    ]
    return hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


@dataclass(frozen=True, slots=True)
class TrustedVerificationReceiptV1:
    provider_id: str
    log_entry_id: str
    verified_at_utc: str
    experiment_plan_sha256: str
    run_result_registry_sha256: str
    aggregate_evidence_bundle_sha256: str
    qualification_registry_sha256: str
    validation_result_registry_sha256: str
    development_dataset_release_receipt_sha256: str
    primary_dataset_release_receipt_sha256: str
    griffin_dataset_release_receipt_sha256: str
    measured_trace_receipt_sha256: str
    run_evidence_inventory_sha256: str
    attempt_manifest_sha256: str
    independent_verification_report_sha256: str
    invariant_report_sha256: str
    evaluator_recompute_report_sha256: str
    wire_ledger_replay_report_sha256: str
    cold_cache_inventory_sha256: str
    invariant_ids: tuple[str, ...]
    cold_cache_verified: bool
    dataset_release_identity_and_license_verified: bool
    measured_trace_semantics_verified: bool
    validation_evidence_semantics_verified: bool
    run_evidence_semantics_verified: bool
    execution_chronology_verified: bool
    evaluator_outputs_recomputed: bool
    wire_ledgers_replayed: bool
    scientific_claims_allowed: bool
    public_key_hex: str
    signature_hex: str

    _FIELDS = frozenset(
        {
            "aggregate_evidence_bundle_sha256",
            "attempt_manifest_sha256",
            "cold_cache_inventory_sha256",
            "cold_cache_verified",
            "dataset_release_identity_and_license_verified",
            "development_dataset_release_receipt_sha256",
            "evaluator_outputs_recomputed",
            "evaluator_recompute_report_sha256",
            "experiment_plan_sha256",
            "griffin_dataset_release_receipt_sha256",
            "invariant_ids",
            "invariant_report_sha256",
            "independent_verification_report_sha256",
            "kind",
            "log_entry_id",
            "measured_trace_receipt_sha256",
            "measured_trace_semantics_verified",
            "primary_dataset_release_receipt_sha256",
            "provider_id",
            "public_key_hex",
            "qualification_registry_sha256",
            "validation_result_registry_sha256",
            "run_evidence_inventory_sha256",
            "run_evidence_semantics_verified",
            "run_result_registry_sha256",
            "schema_version",
            "scientific_claims_allowed",
            "signature_algorithm",
            "signature_hex",
            "validation_evidence_semantics_verified",
            "verified_at_utc",
            "wire_ledger_replay_report_sha256",
            "wire_ledgers_replayed",
            "execution_chronology_verified",
        }
    )

    def __post_init__(self) -> None:
        for name in ("provider_id", "log_entry_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "verified_at_utc", _verified_at(self.verified_at_utc))
        for name in (
            "experiment_plan_sha256",
            "run_result_registry_sha256",
            "aggregate_evidence_bundle_sha256",
            "attempt_manifest_sha256",
            "qualification_registry_sha256",
            "validation_result_registry_sha256",
            "development_dataset_release_receipt_sha256",
            "primary_dataset_release_receipt_sha256",
            "griffin_dataset_release_receipt_sha256",
            "measured_trace_receipt_sha256",
            "run_evidence_inventory_sha256",
            "independent_verification_report_sha256",
            "invariant_report_sha256",
            "evaluator_recompute_report_sha256",
            "wire_ledger_replay_report_sha256",
            "cold_cache_inventory_sha256",
            "public_key_hex",
        ):
            object.__setattr__(
                self, name, _hex(getattr(self, name), _HEX_32, name)
            )
        object.__setattr__(
            self,
            "signature_hex",
            _hex(self.signature_hex, _HEX_64, "signature_hex"),
        )
        if not isinstance(self.invariant_ids, (list, tuple)):
            raise TypeError("invariant_ids must be an array")
        invariants = tuple(_identifier(item, "invariant_id") for item in self.invariant_ids)
        if invariants != REQUIRED_INVARIANTS_V1:
            raise VerificationReceiptError(
                "verification receipt does not cover every preregistered invariant"
            )
        object.__setattr__(self, "invariant_ids", invariants)
        for name in (
            "cold_cache_verified",
            "dataset_release_identity_and_license_verified",
            "measured_trace_semantics_verified",
            "validation_evidence_semantics_verified",
            "run_evidence_semantics_verified",
            "execution_chronology_verified",
            "evaluator_outputs_recomputed",
            "wire_ledgers_replayed",
            "scientific_claims_allowed",
        ):
            if getattr(self, name) is not True:
                raise VerificationReceiptError(f"{name} must be true")

    def signed_payload(self) -> dict[str, Any]:
        return {
            "aggregate_evidence_bundle_sha256": (
                self.aggregate_evidence_bundle_sha256
            ),
            "attempt_manifest_sha256": self.attempt_manifest_sha256,
            "cold_cache_inventory_sha256": self.cold_cache_inventory_sha256,
            "cold_cache_verified": self.cold_cache_verified,
            "dataset_release_identity_and_license_verified": (
                self.dataset_release_identity_and_license_verified
            ),
            "evaluator_outputs_recomputed": self.evaluator_outputs_recomputed,
            "evaluator_recompute_report_sha256": (
                self.evaluator_recompute_report_sha256
            ),
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "development_dataset_release_receipt_sha256": (
                self.development_dataset_release_receipt_sha256
            ),
            "griffin_dataset_release_receipt_sha256": (
                self.griffin_dataset_release_receipt_sha256
            ),
            "invariant_ids": list(self.invariant_ids),
            "independent_verification_report_sha256": (
                self.independent_verification_report_sha256
            ),
            "invariant_report_sha256": self.invariant_report_sha256,
            "kind": "eventtrack_v2x_verification_payload_v1",
            "log_entry_id": self.log_entry_id,
            "measured_trace_receipt_sha256": self.measured_trace_receipt_sha256,
            "measured_trace_semantics_verified": (
                self.measured_trace_semantics_verified
            ),
            "primary_dataset_release_receipt_sha256": (
                self.primary_dataset_release_receipt_sha256
            ),
            "provider_id": self.provider_id,
            "qualification_registry_sha256": self.qualification_registry_sha256,
            "validation_result_registry_sha256": (
                self.validation_result_registry_sha256
            ),
            "run_evidence_inventory_sha256": self.run_evidence_inventory_sha256,
            "run_evidence_semantics_verified": (
                self.run_evidence_semantics_verified
            ),
            "execution_chronology_verified": self.execution_chronology_verified,
            "run_result_registry_sha256": self.run_result_registry_sha256,
            "schema_version": 1,
            "scientific_claims_allowed": self.scientific_claims_allowed,
            "verified_at_utc": self.verified_at_utc,
            "validation_evidence_semantics_verified": (
                self.validation_evidence_semantics_verified
            ),
            "wire_ledger_replay_report_sha256": (
                self.wire_ledger_replay_report_sha256
            ),
            "wire_ledgers_replayed": self.wire_ledgers_replayed,
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            **self.signed_payload(),
            "kind": "trusted_verification_receipt_v1",
            "public_key_hex": self.public_key_hex,
            "signature_algorithm": "ed25519",
            "signature_hex": self.signature_hex,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def public_key_sha256(self) -> str:
        return hashlib.sha256(bytes.fromhex(self.public_key_hex)).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "TrustedVerificationReceiptV1":
        result = _strict_mapping(value, cls._FIELDS, cls.__name__)
        if (
            result["kind"] != "trusted_verification_receipt_v1"
            or result["schema_version"] != 1
            or result["signature_algorithm"] != "ed25519"
        ):
            raise VerificationReceiptError("unsupported verification receipt schema")
        kwargs = {
            key: result[key]
            for key in cls._FIELDS
            if key not in {"kind", "schema_version", "signature_algorithm"}
        }
        kwargs["invariant_ids"] = tuple(result["invariant_ids"])  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]


def sign_trusted_verification_receipt_v1(
    *, private_key: Ed25519PrivateKey, **fields: object
) -> TrustedVerificationReceiptV1:
    """Sign verifier output; production callers must protect the verifier key."""

    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be Ed25519PrivateKey")
    public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    unsigned = TrustedVerificationReceiptV1(
        **fields,
        public_key_hex=public.hex(),
        signature_hex="0" * 128,
    )
    signature = private_key.sign(canonical_json_bytes(unsigned.signed_payload()))
    return TrustedVerificationReceiptV1(
        **fields,
        public_key_hex=public.hex(),
        signature_hex=signature.hex(),
    )


def verify_trusted_verification_receipt_v1(
    receipt: TrustedVerificationReceiptV1,
    *, trusted_public_key_sha256: str,
) -> None:
    if not isinstance(receipt, TrustedVerificationReceiptV1):
        raise TypeError("receipt must be TrustedVerificationReceiptV1")
    trusted = _hex(
        trusted_public_key_sha256, _HEX_32, "trusted_public_key_sha256"
    )
    if receipt.public_key_sha256 != trusted:
        raise VerificationReceiptError("publication verifier is not the trusted key")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(receipt.public_key_hex)).verify(
            bytes.fromhex(receipt.signature_hex),
            canonical_json_bytes(receipt.signed_payload()),
        )
    except (InvalidSignature, ValueError) as exc:
        raise VerificationReceiptError("invalid publication verifier signature") from exc


def decode_trusted_verification_receipt(
    data: bytes,
) -> TrustedVerificationReceiptV1:
    if type(data) is not bytes:
        raise TypeError("verification receipt data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise VerificationReceiptError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationReceiptError("invalid verification receipt JSON") from exc
    receipt = TrustedVerificationReceiptV1.from_mapping(value)
    if receipt.canonical_bytes != data:
        raise VerificationReceiptError("verification receipt JSON is not canonical")
    return receipt


__all__ = [
    "TrustedVerificationReceiptV1",
    "VerificationReceiptError",
    "decode_trusted_verification_receipt",
    "run_evidence_inventory_sha256",
    "sign_trusted_verification_receipt_v1",
    "verify_trusted_verification_receipt_v1",
]
