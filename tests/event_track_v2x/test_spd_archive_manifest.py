from __future__ import annotations

from pathlib import Path

import pytest

from tools.event_track_v2x.archive_manifest import (
    ArchiveManifestError,
    EXPECTED_ARCHIVE_NAMES,
    build_archive_manifest,
    read_archive_manifest,
    verify_archive_files,
    write_archive_manifest,
)


ARCHIVE_NAMES = EXPECTED_ARCHIVE_NAMES


def _archives(root: Path) -> Path:
    root.mkdir()
    for index, name in enumerate(ARCHIVE_NAMES):
        (root / name).write_bytes(f"archive-{index}".encode("ascii"))
    return root


def test_manifest_round_trip_and_byte_verification(tmp_path: Path) -> None:
    archives = _archives(tmp_path / "archives")
    document = build_archive_manifest(archives)
    manifest_path = write_archive_manifest(tmp_path / "manifest.json", document)

    readback = read_archive_manifest(manifest_path)
    verified = verify_archive_files(archives, readback)

    assert readback["archive_count"] == 8
    assert readback["release_identity_status"] == "unverified-source-bytes"
    assert readback["total_size_bytes"] == sum(path.stat().st_size for path in verified)
    assert [record["name"] for record in readback["archives"]] == sorted(
        ARCHIVE_NAMES
    )
    assert manifest_path.read_bytes().endswith(b"\n")


def test_manifest_detects_archive_tampering(tmp_path: Path) -> None:
    archives = _archives(tmp_path / "archives")
    document = build_archive_manifest(archives)
    (archives / ARCHIVE_NAMES[0]).write_bytes(b"tampered")

    with pytest.raises(ArchiveManifestError, match="mismatch"):
        verify_archive_files(archives, document)


def test_manifest_rejects_archive_symlink(tmp_path: Path) -> None:
    archives = _archives(tmp_path / "archives")
    target = archives / ARCHIVE_NAMES[0]
    target.unlink()
    target.symlink_to(archives / ARCHIVE_NAMES[1])

    with pytest.raises(ArchiveManifestError, match="non-symlink"):
        build_archive_manifest(archives)


def test_manifest_rejects_noncanonical_json(tmp_path: Path) -> None:
    archives = _archives(tmp_path / "archives")
    document = build_archive_manifest(archives)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(str(document), encoding="utf-8")

    with pytest.raises(ArchiveManifestError, match="invalid archive manifest JSON"):
        read_archive_manifest(manifest_path)


def test_manifest_rejects_disguised_suffix_and_identity_escalation(
    tmp_path: Path,
) -> None:
    archives = _archives(tmp_path / "archives")
    disguised = archives / ARCHIVE_NAMES[0]
    disguised.rename(archives / f"{ARCHIVE_NAMES[0]}.txt")
    with pytest.raises(ArchiveManifestError, match="expected 8"):
        build_archive_manifest(archives)
    (archives / f"{ARCHIVE_NAMES[0]}.txt").rename(disguised)
    with pytest.raises(ArchiveManifestError, match="not allowed"):
        build_archive_manifest(
            archives, release_identity_status="official-byte-verified"
        )
