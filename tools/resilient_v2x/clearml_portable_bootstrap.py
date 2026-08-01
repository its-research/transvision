#!/usr/bin/env python3
"""Materialize the locked ResilientV2X runtime before starting a ClearML run."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


RUNTIME_ARCHIVE = "resilient-v2x-runtime-a9112cf7.tar.zst"
RUNTIME_ARCHIVE_BYTES = 5_740_702_308
RUNTIME_ARCHIVE_SHA256 = (
    "1d58e605cc6ef8c467c739ecd8837c18cf7458c58cda9c17d8f230213d0966e7"
)
ZSTD_BINARY = "zstd"
ZSTD_BINARY_BYTES = 1_022_760
ZSTD_BINARY_SHA256 = "7c5468b370f7c47eda07281e3437fafc568f95d10420051e3aa522709f9342c5"
RUNNER_SHA256 = "a9112cf7ecd80c13c6a803daa041bda19bee2d8f9793df7307cb3c6501320352"
RUNNER_BYTES = 33_458
IMAGE_DIGEST = "sha256:f1de6a66fd626761661cfe36b9538c90f0056628962d7faca308578e6abdad38"
PYTHON = Path("/opt/resilient-v2x/bin/python")
WORKSPACE = Path("/workspace/transvision")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dataset-id", required=True)
    parser.add_argument("--training-dataset-id", required=True)
    parser.add_argument("--gpus", type=int, choices=(4,), default=4)
    parser.add_argument(
        "--stage",
        choices=("all", "teacher", "student", "validate"),
        default="all",
    )
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> Path:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"file size mismatch: {path}")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"file SHA-256 mismatch: {path}")
    return path


def _require_empty_target(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty runtime target: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if os.environ.get("RESILIENT_V2X_CONTAINER_IMAGE_DIGEST") != IMAGE_DIGEST:
        raise ValueError("portable runtime image digest contract is missing or wrong")

    from clearml import Dataset

    runtime_root = Path(
        Dataset.get(
            dataset_id=args.runtime_dataset_id,
            only_completed=True,
        ).get_local_copy()
    ).resolve(strict=True)
    archive = _verify_file(
        runtime_root / "runtime" / RUNTIME_ARCHIVE,
        expected_bytes=RUNTIME_ARCHIVE_BYTES,
        expected_sha256=RUNTIME_ARCHIVE_SHA256,
    )
    source_zstd = _verify_file(
        runtime_root / "runtime" / ZSTD_BINARY,
        expected_bytes=ZSTD_BINARY_BYTES,
        expected_sha256=ZSTD_BINARY_SHA256,
    )

    _require_empty_target(PYTHON.parent.parent)
    _require_empty_target(WORKSPACE)
    local_zstd = Path("/tmp/resilient-v2x-zstd")
    if local_zstd.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap tool: {local_zstd}")
    shutil.copyfile(source_zstd, local_zstd)
    local_zstd.chmod(0o500)
    if _sha256(local_zstd) != ZSTD_BINARY_SHA256:
        raise ValueError("copied zstd binary SHA-256 mismatch")

    subprocess.run(
        [
            "tar",
            f"--use-compress-program={local_zstd}",
            "-xf",
            str(archive),
            "-C",
            "/",
        ],
        check=True,
    )
    runner = _verify_file(
        WORKSPACE / "tools/resilient_v2x/clearml_train.py",
        expected_bytes=RUNNER_BYTES,
        expected_sha256=RUNNER_SHA256,
    )
    if not PYTHON.is_file():
        raise FileNotFoundError(f"portable Python is missing: {PYTHON}")

    env = dict(os.environ)
    torch_lib = Path("/opt/resilient-v2x/lib/python3.10/site-packages/torch/lib")
    env.update(
        {
            "PATH": (
                "/opt/resilient-v2x/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "PYTHONPATH": str(WORKSPACE),
            "LD_LIBRARY_PATH": ":".join(
                filter(
                    None,
                    (
                        "/opt/resilient-v2x/lib",
                        str(torch_lib),
                        env.get("LD_LIBRARY_PATH", ""),
                    ),
                )
            ),
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    command = [
        str(PYTHON),
        str(runner),
        "--dataset-id",
        args.training_dataset_id,
        "--gpus",
        str(args.gpus),
        "--stage",
        args.stage,
        "--max-epochs",
        str(args.max_epochs),
    ]
    if args.amp:
        command.append("--amp")
    os.chdir(WORKSPACE)
    os.execve(str(PYTHON), command, env)
    raise AssertionError("os.execve unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
