#!/usr/bin/env python3
"""Create and verify a canonical source snapshot for a ClearML worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Sequence


SENSITIVE_NAMES = {
    ".env",
    ".netrc",
    ".clearml.conf",
    "clearml.conf",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pfx", ".pem"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run_git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _sha256_stream(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)[1]


def _tracked_modes(root: Path) -> dict[bytes, bytes]:
    records = _run_git(root, "ls-files", "-s", "-z").split(b"\0")
    result: dict[bytes, bytes] = {}
    for record in records:
        if not record:
            continue
        prefix, path = record.split(b"\t", 1)
        mode = prefix.split(b" ", 1)[0]
        if path in result:
            raise ValueError(f"duplicate Git index path: {path!r}")
        result[path] = mode
    return result


def _validate_relative_path(raw_path: bytes) -> str:
    path = raw_path.decode("utf-8", errors="strict")
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        raise ValueError(f"unsafe source path: {path!r}")
    lowered = {part.casefold() for part in pure.parts}
    name = pure.name.casefold()
    if (
        lowered & SENSITIVE_NAMES
        or name in SENSITIVE_NAMES
        or PurePosixPath(name).suffix in SENSITIVE_SUFFIXES
    ):
        raise ValueError(f"sensitive source path is forbidden: {path!r}")
    return path


def _snapshot(root: Path) -> list[dict[str, object]]:
    root = root.resolve(strict=True)
    tracked_modes = _tracked_modes(root)
    raw_paths = _run_git(
        root,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ).split(b"\0")
    raw_paths = sorted({path for path in raw_paths if path})
    entries: list[dict[str, object]] = []
    for raw_path in raw_paths:
        relative_path = _validate_relative_path(raw_path)
        source = root.joinpath(*PurePosixPath(relative_path).parts)
        metadata = source.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"source entry is not a regular file: {relative_path}")
        resolved = source.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"source entry escapes root: {relative_path}") from error
        git_mode = tracked_modes.get(raw_path)
        mode = 0o755 if git_mode == b"100755" else 0o644
        entries.append(
            {
                "path": relative_path,
                "mode": mode,
                "size": metadata.st_size,
                "sha256": _sha256(source),
            }
        )
    if not entries:
        raise RuntimeError("source inventory is empty")
    return entries


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _tree_sha256(entries: list[dict[str, object]]) -> str:
    return hashlib.sha256(_canonical_json(entries)).hexdigest()


def _write_archive(
    root: Path,
    entries: list[dict[str, object]],
    destination: Path,
) -> None:
    import zstandard

    compressor = zstandard.ZstdCompressor(
        level=19,
        threads=1,
        write_checksum=True,
    )
    with destination.open("xb") as compressed:
        with compressor.stream_writer(compressed, closefd=False) as stream:
            with tarfile.open(
                fileobj=stream, mode="w|", format=tarfile.GNU_FORMAT
            ) as bundle:
                for entry in entries:
                    relative_path = str(entry["path"])
                    source = root.joinpath(*PurePosixPath(relative_path).parts)
                    member = tarfile.TarInfo(relative_path)
                    member.type = tarfile.REGTYPE
                    member.mode = int(entry["mode"])
                    member.uid = 0
                    member.gid = 0
                    member.uname = ""
                    member.gname = ""
                    member.mtime = 0
                    member.size = int(entry["size"])
                    with source.open("rb") as payload:
                        bundle.addfile(member, payload)


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(8 * 1024 * 1024)
            right_chunk = right_stream.read(8 * 1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _verify_archive(
    archive: Path,
    entries: list[dict[str, object]],
) -> None:
    import zstandard

    observed = 0
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as bundle:
                for member in bundle:
                    if observed >= len(entries):
                        raise ValueError(f"unexpected archive member: {member.name!r}")
                    expected = entries[observed]
                    if (
                        not member.isfile()
                        or member.name != expected["path"]
                        or member.mode != expected["mode"]
                        or member.size != expected["size"]
                        or member.uid != 0
                        or member.gid != 0
                        or member.uname
                        or member.gname
                        or member.mtime != 0
                    ):
                        raise ValueError(
                            f"archive member contract mismatch: {member.name!r}"
                        )
                    payload = bundle.extractfile(member)
                    if payload is None:
                        raise ValueError(
                            f"archive member has no payload: {member.name!r}"
                        )
                    with payload:
                        size, digest = _sha256_stream(payload)
                    if size != expected["size"] or digest != expected["sha256"]:
                        raise ValueError(f"archive payload mismatch: {member.name!r}")
                    observed += 1
    if observed != len(entries):
        raise ValueError(
            f"archive entry count mismatch: expected {len(entries)}, got {observed}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    initial_entries = _snapshot(root)
    tree_sha256 = _tree_sha256(initial_entries)
    archive_name = f"resilient-v2x-source-{tree_sha256[:12]}.tar.zst"
    first = output_dir / f".{archive_name}.first"
    second = output_dir / f".{archive_name}.second"
    _write_archive(root, initial_entries, first)
    _write_archive(root, initial_entries, second)
    if _sha256(first) != _sha256(second) or not _files_equal(first, second):
        raise RuntimeError("canonical source archive is not byte deterministic")

    final_entries = _snapshot(root)
    if final_entries != initial_entries:
        raise RuntimeError("source tree changed while creating the archive")
    _verify_archive(first, initial_entries)
    _verify_archive(second, initial_entries)

    archive = output_dir / archive_name
    first.replace(archive)
    second.unlink()
    inventory = {
        "schema_version": 1,
        "artifact_type": "resilient_v2x_source_inventory",
        "algorithm": {
            "inventory": "git-ls-files-v1",
            "tar": "gnu-regular-files-uid0-gid0-mtime0",
            "zstd": "level19-threads1-checksum",
        },
        "git_head": _run_git(root, "rev-parse", "HEAD").decode().strip(),
        "file_count": len(initial_entries),
        "source_bytes": sum(int(entry["size"]) for entry in initial_entries),
        "tree_sha256": tree_sha256,
        "files": initial_entries,
    }
    inventory_path = output_dir / "source-inventory.json"
    inventory_path.write_bytes(_canonical_json(inventory) + b"\n")

    summary = {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "inventory": str(inventory_path),
        "inventory_bytes": inventory_path.stat().st_size,
        "inventory_sha256": _sha256(inventory_path),
        "file_count": inventory["file_count"],
        "source_bytes": inventory["source_bytes"],
        "tree_sha256": tree_sha256,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
