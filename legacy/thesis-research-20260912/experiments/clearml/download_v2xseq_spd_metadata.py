#!/usr/bin/env python3
"""Download a pinned, non-official V2X-Seq-SPD metadata mirror safely.

The mirror is used only to unblock protocol and association-pipeline checks. It
does not replace an authoritative dataset root, raw sensor data, or the full TFD
dataset. Maps are excluded by default because three duplicated map directories
account for most of the mirrored metadata size.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO = "apdoa/DAIR-V2X"
DEFAULT_REVISION = "daa36a1fa2b848c38b4bb63188adfe660531131b"
DEFAULT_PREFIX = "SPD/train_val_encode/V2X-Seq-SPD-New/cooperative/"
DEFAULT_MAX_BYTES = 200 * 1024 * 1024
USER_AGENT = "rtp-v2x-protocol-audit/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--include-maps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("dataset API response must be an object")
    return payload


def safe_target(destination: Path, relative_path: str) -> Path:
    root = destination.resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"mirror path escapes destination: {relative_path}") from exc
    return target


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, row: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size != row.get("size"):
        return False
    lfs = row.get("lfs")
    if isinstance(lfs, dict) and isinstance(lfs.get("sha256"), str):
        return file_sha256(path) == lfs["sha256"]
    blob_id = row.get("blobId")
    return isinstance(blob_id, str) and git_blob_sha1(path) == blob_id


def download_one(
    *,
    row: dict[str, Any],
    destination: Path,
    repo: str,
    revision: str,
    prefix: str,
) -> dict[str, Any]:
    source_path = row["rfilename"]
    relative_path = source_path[len(prefix) :]
    target = safe_target(destination, relative_path)
    if verify_file(target, row):
        return {
            "path": relative_path,
            "size": target.stat().st_size,
            "sha256": file_sha256(target),
            "status": "reused",
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in source_path.split("/"))
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{encoded_path}"
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if not verify_file(temporary, row):
                raise RuntimeError(f"downloaded file failed size/hash verification: {source_path}")
            os.replace(temporary, target)
            return {
                "path": relative_path,
                "size": target.stat().st_size,
                "sha256": file_sha256(target),
                "status": "downloaded",
            }
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {source_path}: {last_error}")


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 32:
        raise SystemExit("--workers must be in [1, 32]")
    if args.max_bytes <= 0:
        raise SystemExit("--max-bytes must be positive")
    if not args.prefix.endswith("/"):
        raise SystemExit("--prefix must end with a slash")

    api_url = f"https://huggingface.co/api/datasets/{args.repo}?blobs=true"
    info = request_json(api_url)
    if info.get("sha") != args.revision:
        raise SystemExit(
            f"mirror revision changed: expected {args.revision}, observed {info.get('sha')}"
        )
    siblings = info.get("siblings")
    if not isinstance(siblings, list):
        raise SystemExit("mirror API response lacks siblings")
    rows = [
        row
        for row in siblings
        if isinstance(row, dict)
        and isinstance(row.get("rfilename"), str)
        and row["rfilename"].startswith(args.prefix)
        and (args.include_maps or "/maps/" not in row["rfilename"])
    ]
    rows.sort(key=lambda row: row["rfilename"])
    if not rows:
        raise SystemExit("no files matched the requested prefix")
    total_bytes = sum(int(row.get("size", -1)) for row in rows)
    if total_bytes < 0 or total_bytes > args.max_bytes:
        raise SystemExit(
            f"refusing download of {total_bytes} bytes; limit is {args.max_bytes}"
        )

    plan = {
        "source": f"https://huggingface.co/datasets/{args.repo}",
        "official_source": False,
        "revision": args.revision,
        "prefix": args.prefix,
        "maps_included": args.include_maps,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "scientific_claim_allowed": False,
        "purpose": "protocol and association-pipeline audit only",
    }
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        return

    args.destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                download_one,
                row=row,
                destination=args.destination,
                repo=args.repo,
                revision=args.revision,
                prefix=args.prefix,
            )
            for row in rows
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            entries.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"verified {index}/{len(futures)} files", flush=True)

    entries.sort(key=lambda row: row["path"])
    manifest = {**plan, "entries": entries}
    manifest_path = args.destination / "mirror-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={file_sha256(manifest_path)}")


if __name__ == "__main__":
    main()
