#!/usr/bin/env python3
"""Validate the immutable RTP-V2X fault contract and derive keyed RNG draws.

This module does not inject faults into a dataset.  It supplies the fail-closed
contract boundary shared by future tracking, forecasting, and communication
runners.  In particular, it keeps random decisions independent of traversal
order and provides a literal byte-identity reference for the nominal control.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SCHEMA_VERSION = 1
CONTRACT_ID = "RTPV2X-FAULT-INJECTION-v1"
CONTRACT_STATUS = "frozen"
CONTRACT_SCOPE = "six_required_thesis_conditions_only"
CONDITION_IDS = (
    "nominal",
    "latency_300ms",
    "independent_packet_loss_0p3",
    "gilbert_elliott_burst_loss",
    "pose_noise_0p5m_0p5deg",
    "observation_retention_0p5",
)
FIXED_BASE_SEEDS = (3407, 4909, 6203)
SEED_DOMAIN = "RTPV2X-FAULT-SEED-v1"
RNG_DOMAIN = "RTPV2X-FAULT-RNG-v1"
EXPECTED_CANONICAL_SHA256 = (
    "154ae11b499570783d8eae6cf4906fea3b0d6397782375065b044e0300ed7fe5"
)
ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "status",
        "scope",
        "claim_boundary",
        "fixed_base_seeds",
        "run_composition",
        "time_semantics",
        "seed_derivation",
        "rng",
        "nominal_invariants",
        "conditions",
        "trace_policy",
    }
)
REFERENCE_FIELDS = frozenset({"contract_id", "path", "sha256", "condition_ids"})


class FaultContractError(RuntimeError):
    """Raised when a fault contract or deterministic draw is not reproducible."""


def _reject_constant(value: str) -> None:
    raise FaultContractError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FaultContractError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def loads_json_strict(payload: bytes | str) -> object:
    """Parse UTF-8 JSON while rejecting duplicates and non-finite numbers."""

    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except FaultContractError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FaultContractError(
            f"fault contract is not strict UTF-8 JSON: {type(exc).__name__}"
        ) from exc


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical semantic representation used by the v1 contract."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FaultContractError(
            f"value cannot be represented as canonical JSON: {type(exc).__name__}"
        ) from exc
    return (text + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FaultContractError(
            f"cannot hash fault-contract file: {type(exc).__name__}"
        ) from exc
    return digest.hexdigest()


def validate_contract(document: object) -> dict[str, Any]:
    """Validate the exact immutable semantic content of contract version 1."""

    if not isinstance(document, dict):
        raise FaultContractError("fault contract root must be an object")
    if set(document) != ROOT_FIELDS:
        raise FaultContractError(
            "fault contract root fields do not match the v1 schema"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise FaultContractError("fault contract schema_version must be 1")
    if document.get("contract_id") != CONTRACT_ID:
        raise FaultContractError(f"fault contract id must be {CONTRACT_ID}")
    if document.get("status") != CONTRACT_STATUS:
        raise FaultContractError("fault contract status must be frozen")
    if document.get("scope") != CONTRACT_SCOPE:
        raise FaultContractError("fault contract scope does not match v1")
    if document.get("fixed_base_seeds") != list(FIXED_BASE_SEEDS):
        raise FaultContractError("fault contract fixed seeds do not match v1")

    conditions = document.get("conditions")
    if not isinstance(conditions, list):
        raise FaultContractError("fault contract conditions must be an array")
    observed_ids = [
        row.get("id") if isinstance(row, dict) else None for row in conditions
    ]
    if observed_ids != list(CONDITION_IDS):
        raise FaultContractError(
            "fault contract must contain the six required conditions in fixed order"
        )

    semantic_sha256 = sha256_bytes(canonical_json_bytes(document))
    if semantic_sha256 != EXPECTED_CANONICAL_SHA256:
        raise FaultContractError(
            "fault contract content differs from immutable v1; create a new version"
        )
    return document


def load_contract(path: Path | str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FaultContractError("fault contract must be a regular non-symlink file")
    try:
        payload = candidate.read_bytes()
    except OSError as exc:
        raise FaultContractError(
            f"cannot read fault contract: {type(exc).__name__}"
        ) from exc
    return validate_contract(loads_json_strict(payload))


def _safe_repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise FaultContractError("fault contract path must be repository-relative")
    resolved_root = root.resolve()
    cursor = resolved_root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise FaultContractError("fault contract path may not traverse a symlink")
    path = cursor.resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise FaultContractError("fault contract path escapes repository") from exc
    if not path.is_file():
        raise FaultContractError("fault contract path is not a regular file")
    return path


def validate_protocol_reference(
    protocol: Mapping[str, object], repository_root: Path | str
) -> dict[str, Any]:
    """Resolve and verify a tracking/forecasting protocol's pinned v1 reference."""

    reference = protocol.get("fault_grid")
    if not isinstance(reference, dict) or set(reference) != REFERENCE_FIELDS:
        raise FaultContractError(
            "protocol fault_grid reference has an invalid schema"
        )
    if reference.get("contract_id") != CONTRACT_ID:
        raise FaultContractError("protocol references the wrong fault contract id")
    if reference.get("condition_ids") != list(CONDITION_IDS):
        raise FaultContractError("protocol fault condition ids do not match v1")
    raw_path = reference.get("path")
    expected_sha256 = reference.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise FaultContractError("protocol fault contract path must be non-empty")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise FaultContractError("protocol fault contract SHA-256 is invalid")
    path = _safe_repository_file(Path(repository_root), raw_path)
    if sha256_file(path) != expected_sha256:
        raise FaultContractError("protocol fault contract SHA-256 does not match")
    return load_contract(path)


def _seed_component(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FaultContractError(f"{label} must be a non-empty string")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise FaultContractError(f"{label} contains a forbidden control character")
    return value


def derive_seed_digest(
    *,
    base_seed: int,
    condition_id: str,
    scope_id: str,
    source_id: str,
    item_id: str,
    stream_name: str,
) -> bytes:
    """Derive one 256-bit decision key without shared mutable RNG state."""

    if isinstance(base_seed, bool) or base_seed not in FIXED_BASE_SEEDS:
        raise FaultContractError("base_seed is not one of the three fixed seeds")
    if condition_id not in CONDITION_IDS:
        raise FaultContractError("condition_id is not in the fault contract")
    components = [
        CONTRACT_ID,
        base_seed,
        condition_id,
        _seed_component(scope_id, "scope_id"),
        _seed_component(source_id, "source_id"),
        _seed_component(item_id, "item_id"),
        _seed_component(stream_name, "stream_name"),
    ]
    payload = json.dumps(
        components,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(SEED_DOMAIN.encode("ascii") + b"\x00" + payload).digest()


def seed_u64(seed_digest: bytes) -> int:
    if not isinstance(seed_digest, bytes) or len(seed_digest) != 32:
        raise FaultContractError("seed digest must contain exactly 32 bytes")
    return int.from_bytes(seed_digest[:8], "big", signed=False)


def uniform_open_01(seed_digest: bytes, counter: int) -> float:
    """Return the v1 counter-based open-interval binary64 uniform draw."""

    if not isinstance(seed_digest, bytes) or len(seed_digest) != 32:
        raise FaultContractError("seed digest must contain exactly 32 bytes")
    if (
        isinstance(counter, bool)
        or not isinstance(counter, int)
        or counter < 0
        or counter >= 2**64
    ):
        raise FaultContractError("counter must be an unsigned 64-bit integer")
    block = hashlib.sha256(
        RNG_DOMAIN.encode("ascii")
        + b"\x00"
        + seed_digest
        + counter.to_bytes(8, "big", signed=False)
    ).digest()
    top_53_bits = int.from_bytes(block[:8], "big", signed=False) >> 11
    result = (top_53_bits + 0.5) / 2**53
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise FaultContractError("counter RNG produced an invalid uniform value")
    return result


def bernoulli(seed_digest: bytes, counter: int, probability: float) -> bool:
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise FaultContractError("Bernoulli probability must be finite in [0, 1]")
    return uniform_open_01(seed_digest, counter) < float(probability)


def normal_standard(seed_digest: bytes, counter: int) -> float:
    """Return Box--Muller z0 from counters ``counter`` and ``counter + 1``."""

    if (
        isinstance(counter, bool)
        or not isinstance(counter, int)
        or counter < 0
        or counter >= 2**64 - 1
    ):
        raise FaultContractError(
            "normal start counter must leave room for two unsigned 64-bit draws"
        )
    u1 = uniform_open_01(seed_digest, counter)
    u2 = uniform_open_01(seed_digest, counter + 1)
    result = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    if not math.isfinite(result):
        raise FaultContractError("Box--Muller RNG produced a non-finite normal value")
    return result


def nominal_identity(payload: bytes) -> bytes:
    """Return the exact input object; callers must not parse or reserialize it."""

    if not isinstance(payload, bytes):
        raise FaultContractError("nominal payload must be immutable bytes")
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    document = load_contract(args.contract)
    print(
        json.dumps(
            {
                "contract_id": document["contract_id"],
                "condition_count": len(document["conditions"]),
                "semantic_sha256": EXPECTED_CANONICAL_SHA256,
                "status": document["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
