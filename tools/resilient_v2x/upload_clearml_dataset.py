#!/usr/bin/env python3
"""Publish the exact ResilientV2X train/validation payload to ClearML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OFFICIAL_SPLIT_SHA256 = (
    "d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
)
DATASET_PREFIX = Path("cooperative-vehicle-infrastructure")
MANIFEST_DATASET_PATH = Path("manifests/temporal_manifest_v2.json")
SPLIT_DATASET_PATH = Path("metadata/cooperative-split-data.json")
RESNET_DATASET_PATH = Path("models/resnet50-0676ba61.pth")
TRAINING_ARTIFACTS_PREFIX = Path("protocols/dair_v2")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts/resilient_v2x/dair/temporal_manifest_v2.json",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT / "data/split_datas/cooperative-split-data.json",
    )
    parser.add_argument(
        "--protocol-artifacts",
        type=Path,
        default=ROOT / "artifacts/resilient_v2x/dair_v2",
    )
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        default=ROOT / "models/resnet50-0676ba61.pth",
    )
    parser.add_argument("--project", default="ResilientV2X/Datasets")
    parser.add_argument(
        "--name",
        default="DAIR-V2X-C ResilientV2X v2 runtime",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--chunk-size-mb", type=int, default=512)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[object, dict[str, object]]:
    from transvision.dataset.resilient_v2x_manifest import load_temporal_manifest

    manifest = load_temporal_manifest(
        path,
        expected_split_hash=OFFICIAL_SPLIT_SHA256,
        allow_fixture=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("temporal manifest must be a JSON object")
    return manifest, payload


def _runtime_relative_paths(manifest: object) -> tuple[str, ...]:
    prepared = {item.prepared_relative_path for item in manifest.prepared_artifacts}
    cameras = {
        source.relative_path
        for sample in manifest.samples
        for source in sample.source_slices
        if source.modality == "camera"
    }
    paths = tuple(sorted(prepared | cameras))
    if len(paths) != len(prepared) + len(cameras):
        raise ValueError("prepared LiDAR and camera paths unexpectedly overlap")
    return paths


def _link(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"dataset source is not a regular non-symlink file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _link_tree(
    source_root: Path,
    destination_root: Path,
    *,
    include: Iterable[Path] | None = None,
) -> int:
    paths = include
    if paths is None:
        paths = (
            path.relative_to(source_root)
            for path in source_root.rglob("*")
            if path.is_file()
        )
    count = 0
    for relative in paths:
        _link(source_root / relative, destination_root / relative)
        count += 1
    return count


def _stage_dataset(
    stage: Path,
    *,
    data_root: Path,
    manifest_path: Path,
    split_file: Path,
    protocol_artifacts: Path,
    resnet_checkpoint: Path,
    runtime_paths: Sequence[str],
) -> int:
    count = _link_tree(
        data_root,
        stage / DATASET_PREFIX,
        include=(Path(value) for value in runtime_paths),
    )
    _link(manifest_path, stage / MANIFEST_DATASET_PATH)
    _link(split_file, stage / SPLIT_DATASET_PATH)
    _link(resnet_checkpoint, stage / RESNET_DATASET_PATH)
    count += 3
    count += _link_tree(
        protocol_artifacts,
        stage / TRAINING_ARTIFACTS_PREFIX,
    )
    return count


def _description(
    *,
    manifest_payload: dict[str, object],
    manifest_path: Path,
    split_file: Path,
    runtime_paths: Sequence[str],
) -> str:
    return json.dumps(
        {
            "purpose": (
                "Exact runtime payload for ResilientV2X clean-teacher, "
                "distilled-student, and validation on DAIR-V2X-C"
            ),
            "layout": {
                "data_root": DATASET_PREFIX.as_posix(),
                "manifest": MANIFEST_DATASET_PATH.as_posix(),
                "split": SPLIT_DATASET_PATH.as_posix(),
                "resnet50": RESNET_DATASET_PATH.as_posix(),
                "protocol_artifacts": TRAINING_ARTIFACTS_PREFIX.as_posix(),
            },
            "manifest_content_sha256": manifest_payload["content_sha256"],
            "manifest_file_sha256": _sha256(manifest_path),
            "dataset_release_sha256": manifest_payload["dataset_release_sha256"],
            "split_sha256": _sha256(split_file),
            "runtime_payload_file_count": len(runtime_paths),
            "selection": (
                "prepared_artifacts[].prepared_relative_path plus unique camera "
                "source_slices[].relative_path from the sealed manifest"
            ),
        },
        sort_keys=True,
        indent=2,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_root = args.data_root.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    split_file = args.split_file.resolve(strict=True)
    protocol_artifacts = args.protocol_artifacts.resolve(strict=True)
    resnet_checkpoint = args.resnet_checkpoint.resolve(strict=True)

    manifest, manifest_payload = _load_manifest(manifest_path)
    if _sha256(split_file) != OFFICIAL_SPLIT_SHA256:
        raise ValueError("split file does not match the official cooperative split")
    runtime_paths = _runtime_relative_paths(manifest)

    from clearml import Dataset

    description = _description(
        manifest_payload=manifest_payload,
        manifest_path=manifest_path,
        split_file=split_file,
        runtime_paths=runtime_paths,
    )
    dataset = Dataset.create(
        dataset_project=args.project,
        dataset_name=args.name,
        dataset_version=args.version,
        dataset_tags=[
            "DAIR-V2X-C",
            "ResilientV2X",
            "train",
            "validation",
            "manifest-selected",
        ],
        description=description,
    )
    print(f"CLEARML_DATASET_ID={dataset.id}", flush=True)

    stage = Path(tempfile.mkdtemp(prefix="resilient-v2x-clearml-dataset-"))
    completed = False
    try:
        staged_count = _stage_dataset(
            stage,
            data_root=data_root,
            manifest_path=manifest_path,
            split_file=split_file,
            protocol_artifacts=protocol_artifacts,
            resnet_checkpoint=resnet_checkpoint,
            runtime_paths=runtime_paths,
        )
        print(
            json.dumps(
                {
                    "staging_path": str(stage),
                    "staged_file_count": staged_count,
                    "runtime_payload_file_count": len(runtime_paths),
                    "manifest_content_sha256": manifest_payload["content_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        dataset.add_files(
            stage,
            local_base_folder=str(stage),
            max_workers=args.max_workers,
        )
        dataset.upload(
            show_progress=True,
            verbose=True,
            chunk_size=args.chunk_size_mb,
            max_workers=args.max_workers,
            preview=False,
        )
        dataset.finalize(verbose=True, raise_on_error=True)
        completed = True
        print(
            f"CLEARML_DATASET_URL={dataset._task.get_output_log_web_page()}",
            flush=True,
        )
    finally:
        if completed:
            shutil.rmtree(stage)
        else:
            print(
                f"Upload did not complete; retained staging directory: {stage}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
