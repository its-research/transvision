"""Detached Ed25519 receipt for an immutable confirmatory experiment plan.

Signing proves that the exact plan bytes existed when an external registry
issued its timestamp.  Trust in that time still depends on an independently
approved registry public-key digest; a self-generated key is not publication
evidence.
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

from .experiment import ExperimentPlanV1
from .wire import canonical_json_bytes


_HEX_32 = re.compile(r"[0-9a-f]{64}")
_HEX_64 = re.compile(r"[0-9a-f]{128}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class PreregistrationError(ValueError):
    """Raised when a plan registration receipt is malformed or untrusted."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise PreregistrationError(f"{name} must be a canonical identifier")
    return value


def _hex(value: object, pattern: re.Pattern[str], name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise PreregistrationError(f"{name} is not canonical lowercase hex")
    return value


def _registered_at(value: object) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise PreregistrationError("registered_at_utc must be RFC3339 UTC")
    try:
        observed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PreregistrationError("registered_at_utc must be RFC3339 UTC") from exc
    if observed.utcoffset() is None or observed.utcoffset().total_seconds() != 0:
        raise PreregistrationError("registered_at_utc must use UTC")
    return value


def _strict_mapping(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise PreregistrationError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        raise PreregistrationError(f"{name} has missing or unknown fields")
    return value


@dataclass(frozen=True, slots=True)
class PreregistrationReceiptV1:
    provider_id: str
    log_entry_id: str
    registered_at_utc: str
    experiment_plan_sha256: str
    baseline_qualification_registry_sha256: str
    public_key_hex: str
    signature_hex: str

    _FIELDS = frozenset(
        {
            "baseline_qualification_registry_sha256",
            "experiment_plan_sha256",
            "kind",
            "log_entry_id",
            "provider_id",
            "public_key_hex",
            "registered_at_utc",
            "schema_version",
            "signature_algorithm",
            "signature_hex",
        }
    )

    def __post_init__(self) -> None:
        for name in ("provider_id", "log_entry_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "registered_at_utc", _registered_at(self.registered_at_utc))
        for name in (
            "experiment_plan_sha256",
            "baseline_qualification_registry_sha256",
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

    def signed_payload(self) -> dict[str, object]:
        return {
            "baseline_qualification_registry_sha256": (
                self.baseline_qualification_registry_sha256
            ),
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "kind": "eventtrack_v2x_preregistration_payload_v1",
            "log_entry_id": self.log_entry_id,
            "provider_id": self.provider_id,
            "registered_at_utc": self.registered_at_utc,
            "schema_version": 1,
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            **self.signed_payload(),
            "kind": "preregistration_receipt_v1",
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
    def from_mapping(cls, value: object) -> "PreregistrationReceiptV1":
        result = _strict_mapping(value, cls._FIELDS, cls.__name__)
        if (
            result["kind"] != "preregistration_receipt_v1"
            or result["schema_version"] != 1
            or result["signature_algorithm"] != "ed25519"
        ):
            raise PreregistrationError("unsupported preregistration receipt schema")
        return cls(
            provider_id=result["provider_id"],
            log_entry_id=result["log_entry_id"],
            registered_at_utc=result["registered_at_utc"],
            experiment_plan_sha256=result["experiment_plan_sha256"],
            baseline_qualification_registry_sha256=result[
                "baseline_qualification_registry_sha256"
            ],
            public_key_hex=result["public_key_hex"],
            signature_hex=result["signature_hex"],
        )


def sign_preregistration_receipt_v1(
    *,
    experiment_plan: ExperimentPlanV1,
    provider_id: str,
    log_entry_id: str,
    registered_at_utc: str,
    private_key: Ed25519PrivateKey,
) -> PreregistrationReceiptV1:
    """Create a detached receipt after baseline qualification and plan freeze."""

    if not isinstance(experiment_plan, ExperimentPlanV1):
        raise TypeError("experiment_plan must be ExperimentPlanV1")
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be Ed25519PrivateKey")
    public_bytes = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    unsigned = PreregistrationReceiptV1(
        provider_id=provider_id,
        log_entry_id=log_entry_id,
        registered_at_utc=registered_at_utc,
        experiment_plan_sha256=experiment_plan.content_sha256,
        baseline_qualification_registry_sha256=(
            experiment_plan.baseline_qualification_registry_sha256
        ),
        public_key_hex=public_bytes.hex(),
        signature_hex="0" * 128,
    )
    signature = private_key.sign(canonical_json_bytes(unsigned.signed_payload()))
    return PreregistrationReceiptV1(
        provider_id=unsigned.provider_id,
        log_entry_id=unsigned.log_entry_id,
        registered_at_utc=unsigned.registered_at_utc,
        experiment_plan_sha256=unsigned.experiment_plan_sha256,
        baseline_qualification_registry_sha256=(
            unsigned.baseline_qualification_registry_sha256
        ),
        public_key_hex=unsigned.public_key_hex,
        signature_hex=signature.hex(),
    )


def verify_preregistration_receipt_v1(
    receipt: PreregistrationReceiptV1,
    *,
    experiment_plan: ExperimentPlanV1,
    trusted_public_key_sha256: str,
) -> None:
    """Verify receipt parity, signature, and an out-of-band trust anchor."""

    if not isinstance(receipt, PreregistrationReceiptV1):
        raise TypeError("receipt must be PreregistrationReceiptV1")
    if not isinstance(experiment_plan, ExperimentPlanV1):
        raise TypeError("experiment_plan must be ExperimentPlanV1")
    trusted = _hex(
        trusted_public_key_sha256, _HEX_32, "trusted_public_key_sha256"
    )
    if receipt.public_key_sha256 != trusted:
        raise PreregistrationError("preregistration signer is not the trusted key")
    if (
        receipt.experiment_plan_sha256 != experiment_plan.content_sha256
        or receipt.baseline_qualification_registry_sha256
        != experiment_plan.baseline_qualification_registry_sha256
    ):
        raise PreregistrationError("preregistration receipt does not bind this plan")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(receipt.public_key_hex)).verify(
            bytes.fromhex(receipt.signature_hex),
            canonical_json_bytes(receipt.signed_payload()),
        )
    except (InvalidSignature, ValueError) as exc:
        raise PreregistrationError("invalid preregistration signature") from exc


def decode_preregistration_receipt(data: bytes) -> PreregistrationReceiptV1:
    if type(data) is not bytes:
        raise TypeError("preregistration receipt data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise PreregistrationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreregistrationError("invalid preregistration receipt JSON") from exc
    receipt = PreregistrationReceiptV1.from_mapping(value)
    if receipt.canonical_bytes != data:
        raise PreregistrationError("preregistration receipt JSON is not canonical")
    return receipt


__all__ = [
    "PreregistrationError",
    "PreregistrationReceiptV1",
    "decode_preregistration_receipt",
    "sign_preregistration_receipt_v1",
    "verify_preregistration_receipt_v1",
]
