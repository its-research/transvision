"""Local construction and independent verification of EvidenceBundleV1.

The bundle uses relative POSIX paths.  This lets the same manifest verify the
producer directory, an uploaded ClearML dataset, and a newly downloaded cold
cache without embedding machine-specific absolute paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Mapping

from .contracts import (
    ArtifactDigestV1,
    EvidenceBundleV1,
    decode_contract,
    encode_contract,
)


EVIDENCE_ROLES = (
    "source",
    "dataset",
    "detection_cache",
    "network_trace",
    "tracker_config",
    "evaluator_contract",
    "checkpoint",
    "predictions",
    "per_sequence_metrics",
    "logs",
    "environment",
)


class EvidenceError(ValueError):
    """Raised when an evidence root cannot prove exact artifact identity."""


def _safe_relative_path(value: object, name: str) -> PurePosixPath:
    if type(value) is not str or not value or value != value.strip():
        raise EvidenceError(f"{name} must be a trimmed relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise EvidenceError(f"{name} must be a canonical relative POSIX path")
    return path


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_size += len(chunk)
    return digest.hexdigest(), byte_size


def _regular_file(root: Path, relative: PurePosixPath, role: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise EvidenceError("evidence root must be a real directory")
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise EvidenceError(f"symbolic link rejected for {role}: {relative}")
    if not current.is_file():
        raise EvidenceError(f"artifact for {role} is not a regular file: {relative}")
    return current


def build_evidence_bundle(
    *,
    run_id: str,
    artifact_root: Path,
    role_paths: Mapping[str, str],
) -> EvidenceBundleV1:
    """Hash the exact eleven required artifacts under ``artifact_root``."""

    if set(role_paths) != set(EVIDENCE_ROLES):
        missing = sorted(set(EVIDENCE_ROLES) - set(role_paths))
        unknown = sorted(set(role_paths) - set(EVIDENCE_ROLES))
        raise EvidenceError(
            f"evidence roles do not match contract; missing={missing}, unknown={unknown}"
        )
    root = Path(artifact_root)
    artifacts: dict[str, ArtifactDigestV1] = {}
    observed_paths: set[str] = set()
    for role in EVIDENCE_ROLES:
        relative = _safe_relative_path(role_paths[role], role)
        if relative.as_posix() in observed_paths:
            raise EvidenceError("each evidence role must reference a distinct artifact")
        observed_paths.add(relative.as_posix())
        path = _regular_file(root, relative, role)
        sha256, byte_size = _hash_file(path)
        artifacts[role] = ArtifactDigestV1(
            uri=relative.as_posix(), sha256=sha256, byte_size=byte_size
        )
    return EvidenceBundleV1(run_id=run_id, **artifacts)


def write_evidence_bundle(bundle: EvidenceBundleV1, output_path: Path) -> str:
    """Write a bundle once and return the SHA-256 of its canonical bytes."""

    raw = encode_contract(bundle)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def verify_evidence_bundle(
    bundle_path: Path, *, artifact_root: Path
) -> EvidenceBundleV1:
    """Verify canonical bundle bytes and every bound artifact from scratch."""

    bundle_file = Path(bundle_path)
    if bundle_file.is_symlink() or not bundle_file.is_file():
        raise EvidenceError("evidence bundle must be a regular file")
    bundle = decode_contract(bundle_file.read_bytes(), EvidenceBundleV1)
    root = Path(artifact_root)
    observed_paths: set[str] = set()
    for role in EVIDENCE_ROLES:
        artifact = getattr(bundle, role)
        relative = _safe_relative_path(artifact.uri, f"{role}.uri")
        if relative.as_posix() in observed_paths:
            raise EvidenceError("evidence bundle reuses an artifact across roles")
        observed_paths.add(relative.as_posix())
        path = _regular_file(root, relative, role)
        sha256, byte_size = _hash_file(path)
        if (sha256, byte_size) != (artifact.sha256, artifact.byte_size):
            raise EvidenceError(f"artifact digest mismatch for role {role}")
    return bundle


__all__ = [
    "EVIDENCE_ROLES",
    "EvidenceError",
    "build_evidence_bundle",
    "verify_evidence_bundle",
    "write_evidence_bundle",
]
