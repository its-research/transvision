"""Canonical, content-addressed experiment evidence for ResilientV2X.

This module intentionally has no MMEngine or detector dependency.  It packages
metrics that were already produced by the evaluation runner; it never invokes
a model and never recomputes a metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ROLE_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
DOCUMENT_SCHEMA_VERSION = 1
PROFILE_DOCUMENT_TYPE = "resilient_v2x_complexity_profile"
EVIDENCE_DOCUMENT_TYPE = "resilient_v2x_experiment_evidence"


class EvidenceError(ValueError):
    """Raised when experiment evidence is malformed or unverifiable."""


def _validate_json_domain(value: object, context: str = "document") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise EvidenceError(f"{context} mapping keys must be strings")
            _validate_json_domain(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_domain(item, f"{context}[{index}]")
        return
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise EvidenceError(f"{context} must be finite")
        return
    raise EvidenceError(f"{context} is outside the canonical JSON domain")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a plain JSON value in the repository's canonical form."""

    _validate_json_domain(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of an existing regular file without loading it all."""

    if type(chunk_size) is not int or chunk_size <= 0:
        raise EvidenceError("chunk_size must be a positive integer")
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise EvidenceError(f"artifact is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactDigest:
    """Identity of a concrete input or output artifact."""

    role: str
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.role) is not str or ROLE_PATTERN.fullmatch(self.role) is None:
            raise EvidenceError("artifact role is not canonical")
        if type(self.path) is not str or not self.path:
            raise EvidenceError("artifact path must be non-empty")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise EvidenceError("artifact size_bytes must be a non-negative integer")
        if (
            type(self.sha256) is not str
            or SHA256_PATTERN.fullmatch(self.sha256) is None
        ):
            raise EvidenceError("artifact sha256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def digest_artifact(
    path: str | os.PathLike[str],
    role: str,
) -> ArtifactDigest:
    """Resolve and hash one artifact."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise EvidenceError(f"artifact is not a regular file: {resolved}")
    return ArtifactDigest(
        role=role,
        path=str(resolved),
        size_bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved),
    )


def _document_without_hash(document: Mapping[str, object]) -> dict[str, object]:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return payload


def document_content_sha256(document: Mapping[str, object]) -> str:
    """Hash all document fields except ``content_sha256`` itself."""

    return hashlib.sha256(
        canonical_json_bytes(_document_without_hash(document))
    ).hexdigest()


def seal_document(
    document_type: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Create a versioned document whose hash authenticates all other fields."""

    if type(document_type) is not str or ROLE_PATTERN.fullmatch(document_type) is None:
        raise EvidenceError("document_type is not canonical")
    if not isinstance(payload, Mapping):
        raise EvidenceError("payload must be a mapping")
    reserved = {"schema_version", "document_type", "content_sha256"}
    if reserved.intersection(payload):
        raise EvidenceError("payload contains a reserved document field")
    document: dict[str, object] = {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "document_type": document_type,
        **dict(payload),
    }
    _validate_json_domain(document)
    document["content_sha256"] = document_content_sha256(document)
    return document


def verify_document(
    document: Mapping[str, object],
    *,
    expected_type: str | None = None,
) -> dict[str, object]:
    """Validate structure and content hash, returning a detached plain mapping."""

    if not isinstance(document, Mapping):
        raise EvidenceError("document must be a mapping")
    plain = dict(document)
    if plain.get("schema_version") != DOCUMENT_SCHEMA_VERSION:
        raise EvidenceError("unsupported evidence schema_version")
    document_type = plain.get("document_type")
    if type(document_type) is not str or ROLE_PATTERN.fullmatch(document_type) is None:
        raise EvidenceError("document_type is not canonical")
    if expected_type is not None and document_type != expected_type:
        raise EvidenceError(
            f"expected document_type {expected_type!r}, got {document_type!r}"
        )
    digest = plain.get("content_sha256")
    if type(digest) is not str or SHA256_PATTERN.fullmatch(digest) is None:
        raise EvidenceError("content_sha256 must be lowercase hexadecimal")
    _validate_json_domain(plain)
    if document_content_sha256(plain) != digest:
        raise EvidenceError("evidence content hash mismatch")
    return plain


def write_document(
    path: str | os.PathLike[str],
    document: Mapping[str, object],
) -> Path:
    """Atomically write a verified canonical document."""

    verified = verify_document(document)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(verified) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_document(
    path: str | os.PathLike[str],
    *,
    expected_type: str | None = None,
) -> dict[str, object]:
    """Read a canonical document and reject formatting or hash drift."""

    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    canonical_raw = raw[:-1] if raw.endswith(b"\n") else raw
    try:
        decoded = json.loads(canonical_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid evidence JSON: {source}") from error
    if not isinstance(decoded, Mapping):
        raise EvidenceError("evidence JSON must contain an object")
    if canonical_json_bytes(decoded) != canonical_raw:
        raise EvidenceError("evidence JSON is not canonical")
    return verify_document(decoded, expected_type=expected_type)


def _normalize_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metrics, Mapping) or not metrics:
        raise EvidenceError("metrics must be a non-empty mapping")
    normalized = dict(metrics)
    _validate_json_domain(normalized, "metrics")
    return normalized


def _normalize_artifacts(
    artifacts: Sequence[ArtifactDigest | Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized: list[ArtifactDigest] = []
    for index, artifact in enumerate(artifacts):
        if isinstance(artifact, ArtifactDigest):
            normalized.append(artifact)
            continue
        if not isinstance(artifact, Mapping):
            raise EvidenceError(f"artifacts[{index}] is invalid")
        try:
            normalized.append(
                ArtifactDigest(
                    role=artifact["role"],  # type: ignore[arg-type]
                    path=artifact["path"],  # type: ignore[arg-type]
                    size_bytes=artifact["size_bytes"],  # type: ignore[arg-type]
                    sha256=artifact["sha256"],  # type: ignore[arg-type]
                )
            )
        except KeyError as error:
            raise EvidenceError(
                f"artifacts[{index}] is missing {error.args[0]}"
            ) from error
    roles = [artifact.role for artifact in normalized]
    if len(roles) != len(set(roles)):
        raise EvidenceError("artifact roles must be unique")
    return [
        artifact.to_dict()
        for artifact in sorted(normalized, key=lambda item: item.role)
    ]


def build_evidence_document(
    *,
    run_id: str,
    metrics: Mapping[str, object],
    complexity_profile: Mapping[str, object],
    artifacts: Sequence[ArtifactDigest | Mapping[str, object]],
    conditions: Mapping[str, object] | None = None,
    measured_at_utc: str | None = None,
    prediction_artifact_role: str | None = None,
) -> dict[str, object]:
    """Package existing results without invoking inference or metric code.

    ``prediction_artifact_role`` points at an artifact produced by the same
    evaluation run.  The field is optional because some runners retain only
    aggregate metrics, but its presence gives sample-level auditability.
    """

    if type(run_id) is not str or not run_id or run_id != run_id.strip():
        raise EvidenceError("run_id must be trimmed and non-empty")
    verified_profile = verify_document(
        complexity_profile,
        expected_type=PROFILE_DOCUMENT_TYPE,
    )
    normalized_artifacts = _normalize_artifacts(artifacts)
    artifact_roles = {item["role"] for item in normalized_artifacts}
    if prediction_artifact_role is not None:
        if prediction_artifact_role not in artifact_roles:
            raise EvidenceError(
                "prediction_artifact_role does not identify an artifact"
            )
    normalized_conditions = dict(conditions or {})
    _validate_json_domain(normalized_conditions, "conditions")
    payload: dict[str, object] = {
        "run_id": run_id,
        "metrics": _normalize_metrics(metrics),
        "complexity_profile": verified_profile,
        "artifacts": normalized_artifacts,
        "conditions": normalized_conditions,
        "metric_provenance": {
            "inference_performed_by_evidence_builder": False,
            "metric_recomputation_performed_by_evidence_builder": False,
            "prediction_artifact_role": prediction_artifact_role,
        },
    }
    if measured_at_utc is not None:
        if (
            type(measured_at_utc) is not str
            or not measured_at_utc
            or measured_at_utc != measured_at_utc.strip()
        ):
            raise EvidenceError("measured_at_utc must be trimmed and non-empty")
        payload["measured_at_utc"] = measured_at_utc
    return seal_document(EVIDENCE_DOCUMENT_TYPE, payload)


def capture_git_state(repository: str | os.PathLike[str]) -> dict[str, object]:
    """Capture committed revision plus hashes of tracked and untracked drift."""

    root = Path(repository).expanduser().resolve(strict=True)

    def git(*arguments: str) -> bytes:
        try:
            result = subprocess.run(
                ("git", "-C", str(root), *arguments),
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise EvidenceError(f"cannot inspect Git repository: {root}") from error
        return result.stdout

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise EvidenceError("Git HEAD is not a canonical object ID")
    tracked_patch = git("diff", "--binary", "HEAD", "--")
    untracked_raw = git("ls-files", "--others", "--exclude-standard", "-z")
    untracked_paths = [
        item.decode("utf-8") for item in untracked_raw.split(b"\0") if item
    ]
    untracked: list[dict[str, object]] = []
    for relative in sorted(untracked_paths):
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise EvidenceError("untracked Git path escapes repository") from error
        if not candidate.is_file():
            continue
        untracked.append(
            {
                "path": relative,
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    state: dict[str, object] = {
        "commit": commit,
        "tracked_patch_sha256": hashlib.sha256(tracked_patch).hexdigest(),
        "tracked_patch_size_bytes": len(tracked_patch),
        "untracked_files": untracked,
    }
    state["working_tree_sha256"] = hashlib.sha256(
        canonical_json_bytes(state)
    ).hexdigest()
    state["dirty"] = bool(tracked_patch or untracked)
    return state


__all__ = (
    "ArtifactDigest",
    "DOCUMENT_SCHEMA_VERSION",
    "EVIDENCE_DOCUMENT_TYPE",
    "EvidenceError",
    "PROFILE_DOCUMENT_TYPE",
    "build_evidence_document",
    "canonical_json_bytes",
    "capture_git_state",
    "digest_artifact",
    "document_content_sha256",
    "read_document",
    "seal_document",
    "sha256_file",
    "verify_document",
    "write_document",
)
