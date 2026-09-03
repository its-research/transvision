from pathlib import Path

import pytest

from transvision.models.event_track_v2x.evidence import (
    EVIDENCE_ROLES,
    EvidenceError,
    build_evidence_bundle,
    verify_evidence_bundle,
    write_evidence_bundle,
)


def _artifacts(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, role in enumerate(EVIDENCE_ROLES):
        relative = f"artifacts/{index:02d}-{role}.bin"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}-evidence".encode())
        result[role] = relative
    return result


def test_evidence_bundle_round_trip_and_independent_root(tmp_path: Path) -> None:
    producer = tmp_path / "producer"
    role_paths = _artifacts(producer)
    bundle = build_evidence_bundle(
        run_id="eventtrack-test-run", artifact_root=producer, role_paths=role_paths
    )
    bundle_path = producer / "evidence-bundle.json"
    assert len(write_evidence_bundle(bundle, bundle_path)) == 64
    assert verify_evidence_bundle(bundle_path, artifact_root=producer) == bundle

    cold = tmp_path / "cold"
    cold.mkdir()
    for relative in role_paths.values():
        source = producer / relative
        destination = cold / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    assert verify_evidence_bundle(bundle_path, artifact_root=cold) == bundle


def test_evidence_bundle_rejects_missing_unknown_or_reused_roles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    role_paths = _artifacts(root)
    role_paths.pop("logs")
    with pytest.raises(EvidenceError, match="roles"):
        build_evidence_bundle(
            run_id="run", artifact_root=root, role_paths=role_paths
        )

    role_paths = _artifacts(root)
    role_paths["logs"] = role_paths["environment"]
    with pytest.raises(EvidenceError, match="distinct"):
        build_evidence_bundle(
            run_id="run", artifact_root=root, role_paths=role_paths
        )


def test_evidence_bundle_rejects_tampering_and_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    role_paths = _artifacts(root)
    bundle = build_evidence_bundle(
        run_id="run", artifact_root=root, role_paths=role_paths
    )
    bundle_path = root / "bundle.json"
    write_evidence_bundle(bundle, bundle_path)
    (root / role_paths["predictions"]).write_bytes(b"tampered")
    with pytest.raises(EvidenceError, match="predictions"):
        verify_evidence_bundle(bundle_path, artifact_root=root)

    escaped = _artifacts(tmp_path / "escape-root")
    escaped["source"] = "../outside"
    with pytest.raises(EvidenceError, match="relative"):
        build_evidence_bundle(
            run_id="run", artifact_root=tmp_path / "escape-root", role_paths=escaped
        )


def test_evidence_bundle_rejects_symlinked_artifact(tmp_path: Path) -> None:
    root = tmp_path / "root"
    role_paths = _artifacts(root)
    source = root / role_paths["source"]
    outside = tmp_path / "outside"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable on this filesystem")
    with pytest.raises(EvidenceError, match="symbolic link"):
        build_evidence_bundle(
            run_id="run", artifact_root=root, role_paths=role_paths
        )
