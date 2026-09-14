#!/usr/bin/env python3
"""Package only the authorized runtime, train inputs and necessary runner."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("package directory is create-once")
    root = args.runtime_root
    original = json.loads((root / "full-train-inputs/input-manifest.json").read_bytes())
    if original.get("cohort") != "full-train" or len(original["fit_sequence_ids"]) != 46:
        raise ValueError("expected all 46 official train sequences")
    args.output.mkdir()
    image = "sha256:85525aefed5d9a5d5f6d72d7853f1a9ba3744206c889c767a8dfac8202c1d2ab"
    subprocess.run([
        "docker", "run", "--rm", "--network", "none",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={args.output},dst=/export",
        "--entrypoint", "tar", image,
        "-czf", "/export/runtime.tar.gz", "-C", "/", "opt/cooptrack",
        "usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so.11.0",
        "usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so.11.8.89",
    ], check=True)
    print("EVENTTRACK_PACKAGE_RUNTIME_DONE", flush=True)
    subprocess.run([
        "tar", "-czf", str(args.output / "train-inputs.tar.gz"),
        "--transform=s,^full-train-inputs,inputs,",
        "--transform=s,^conversion-runs/full-train-converted-attempt-2,converted,",
        "-C", str(root), "full-train-inputs",
        "conversion-runs/full-train-converted-attempt-2",
    ], check=True)
    print("EVENTTRACK_PACKAGE_TRAIN_DONE", flush=True)
    source = args.output / "source.tar"
    with source.open("xb") as stream:
        subprocess.run(["git", "-C", str(args.upstream_root), "archive", "--format=tar",
                        "--prefix=workspace/CoopTrack/", "29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399"],
                       stdout=stream, check=True)
    subprocess.run(["tar", "-rf", str(source), "--transform=s,^,entrypoints/,",
                    "-C", str(root), "run_cooptrack_detector.py", "Dockerfile"], check=True)
    with source.open("rb") as src, gzip.open(args.output / "source.tar.gz", "xb", compresslevel=5) as dst:
        shutil.copyfileobj(src, dst)
    source.unlink()  # Only this newly created intermediate; archive retained.
    shutil.copy2(args.pretrained, args.output / "resnet50-0676ba61.pth")
    inventory = [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
                 for path in sorted(args.output.iterdir())]
    manifest = {"kind": "eventtrack_a100_migration_package_v1", "inventory": inventory,
                "source_image_id": image, "requires_glibc_minimum": "2.32",
                "upstream_commit": "29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399",
                "train_sequence_count": 46, "val_or_test_included": False,
                "upload_authorization": "user-explicit-A100-only-2026-09-11"}
    with (args.output / "package-manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print("EVENTTRACK_A100_PACKAGE " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
