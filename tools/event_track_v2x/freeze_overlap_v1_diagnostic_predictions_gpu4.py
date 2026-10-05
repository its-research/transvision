#!/usr/bin/env python3
"""Independently freeze and validate GT-free overlap-v1 diagnostic predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from clearml import Task
from clearml.storage.helper import StorageHelper


SOURCE_TASK_ID = "2cdbb41f7504457585d0c6be052c3f47"
SOURCE_SHA256 = "e511ea54695ccd5154561331f9d6f1ca32f7a9d9d4c59c55825a4efb215b9d6f"
TRAINING_BY_SEED = {
    1337: "669833fb759742a4a857f99f87341e0b",
    2027: "f945af3976e34a479e40b56ef2a7be24",
}
FORMAL_CONFIG_SHA256 = "40479859c407db27cc7e609c889a447057319905f94e887769ef919d57004334"
SELECTION_TASK_ID = "35ae26a544d24a9e9a314f4789e9648f"
SELECTION_SHA256 = "266593c647c99cd33778de5e35a2c5fb14493f6808631309fb949f0b67d4efaa"
SELECTION_MANIFEST_SHA256 = "b5e775cb22ef48ff66ed819d865620bc9f07236aea794561b6ee3510c57d8977"
PROJECTION_TASK_ID = "72599160425d47ca9b7e8916bbb77b42"
PROJECTION_SHA256 = "2a022771710632db4d1011ded447fd6677e241bdff7c51e4f6d572a0c18523d5"
ALLOWED_HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}
ARTIFACTS = {"diagnostic-prediction-manifest": "manifest.json",
             "diagnostic-predictions": "predictions.jsonl",
             "diagnostic-publication-receipt": "publication-receipt.json"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def download(artifact, destination):
    parsed = urlparse(artifact.url)
    if parsed.scheme not in ("http", "https") or parsed.netloc not in ALLOWED_HOSTS:
        raise ValueError("artifact is outside approved ClearML file service")
    alternate = artifact.url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1) \
        if "10.100.34.118:8081" in artifact.url else artifact.url.replace(
            "10.100.35.118:8081", "10.100.34.118:8081", 1)
    urls = ([alternate, artifact.url]
            if urlparse(alternate).netloc == "10.100.35.118:8081"
            else [artifact.url, alternate])
    failures = []
    for url in urls:
        partial = destination.with_name(destination.name + ".part")
        if partial.exists():
            partial.unlink()
        try:
            size = 0
            with partial.open("xb") as output:
                blocks = StorageHelper.get(url).download_as_stream(url)
                if blocks is None:
                    raise RuntimeError("file service returned no stream")
                for block in blocks:
                    output.write(block)
                    size += len(block)
            if (size != artifact.size or sha256(partial) != artifact.hash):
                raise ValueError("streamed artifact size or SHA256 differs")
            partial.replace(destination)
            return
        except Exception as exc:
            failures.append(type(exc).__name__)
            if partial.exists():
                partial.unlink()
    raise RuntimeError("approved ClearML file-service readback failed: " + ",".join(failures))


def valid_finite_numbers(values):
    return isinstance(values, list) and all(type(value) in (int, float)
                                            and math.isfinite(value) for value in values)


def verify_rows(path, selection_frames):
    expected = {(row["sequence_id"], row["frame_ordinal"],
                 "vehicle-side" if row["is_ego"] else "infrastructure-side"):
                row["frame_key"] for row in selection_frames}
    if len(expected) != 2034:
        raise ValueError("selected projection coverage differs")
    identities = set()
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {"sequence_id", "frame_id", "frame_ordinal", "side",
                            "states_ego", "raw_scores", "class_indices"}:
                raise ValueError("prediction row schema differs")
            identity = (row["sequence_id"], row["frame_ordinal"], row["side"])
            if identity not in expected or identity in identities \
                    or row["frame_id"] != expected[identity]:
                raise ValueError("prediction coverage or uniqueness differs")
            identities.add(identity)
            scores = row["raw_scores"]
            states = row["states_ego"]
            classes = row["class_indices"]
            if (not valid_finite_numbers(scores) or len(scores) > 64
                    or any(not 0 <= score <= 1 for score in scores)
                    or not isinstance(states, list) or len(states) != len(scores)
                    or any(not valid_finite_numbers(state) or len(state) != 9
                           for state in states)
                    or not isinstance(classes, list) or len(classes) != len(scores)
                    or any(type(value) is not int or value != 0 for value in classes)):
                raise ValueError("prediction candidate schema or finiteness differs")
            count += 1
    if count != 2034 or identities != set(expected):
        raise ValueError("prediction row coverage differs")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=TRAINING_BY_SEED, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--dispatch-receipt", type=Path, required=True)
    parser.add_argument("--selection-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("fresh diagnostic prediction freeze directory required")
    selection = args.selection_frames.absolute()
    if (selection.is_symlink() or not selection.is_file()
            or sha256(selection) != "1cf182737f04fe6ddd16777165e919f47eacd4375670c0c7c001f6596c48129e"):
        raise ValueError("frozen train-only selection frames differ")
    frames = [json.loads(line) for line in selection.read_bytes().splitlines()]
    dispatch_path = args.dispatch_receipt.absolute()
    if dispatch_path.is_symlink() or not dispatch_path.is_file():
        raise ValueError("ordinary inference dispatch receipt required")
    dispatch = json.loads(dispatch_path.read_bytes())
    checkpoint_task = TRAINING_BY_SEED[args.seed]
    checkpoint_sha = dispatch.get("checkpoint_sha256")
    if (dispatch.get("kind") != "v2v4real_overlap_v1_diagnostic_inference_protocol_gated_dispatch_v1"
            or dispatch.get("seed") != args.seed
            or dispatch.get("dispatch", {}).get("task_id") != args.task_id
            or dispatch.get("source_task_id") != SOURCE_TASK_ID
            or dispatch.get("source_sha256") != SOURCE_SHA256
            or dispatch.get("formal_config_sha256") != FORMAL_CONFIG_SHA256
            or dispatch.get("training_task_id") != checkpoint_task
            or not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64
            or dispatch.get("paper_eligible") is not False
            or dispatch.get("formal_independent_test_eligible") is not False):
        raise ValueError("protocol-gated inference dispatch lineage differs")
    task = Task.get_task(task_id=args.task_id)
    source = Task.get_task(task_id=SOURCE_TASK_ID)
    acceptance_task = Task.get_task(task_id=dispatch["checkpoint_acceptance_task_id"])
    params = task.get_parameters()
    if (str(task.status) != "completed" or str(source.status) != "completed"
            or source.artifacts["source"].hash != SOURCE_SHA256
            or str(acceptance_task.status) != "completed"
            or acceptance_task.artifacts["checkpoint-acceptance"].hash != dispatch["checkpoint_acceptance_sha256"]
            or params.get("General/source_task_id") != SOURCE_TASK_ID
            or params.get("General/source_sha256") != SOURCE_SHA256
            or params.get("General/formal_config_sha256") != FORMAL_CONFIG_SHA256
            or params.get("General/checkpoint_seed") != str(args.seed)
            or params.get("General/checkpoint_task_id") != checkpoint_task
            or params.get("General/checkpoint_sha256") != checkpoint_sha
            or params.get("General/checkpoint_acceptance_task_id") != dispatch["checkpoint_acceptance_task_id"]
            or params.get("General/checkpoint_acceptance_sha256") != dispatch["checkpoint_acceptance_sha256"]
            or set(task.artifacts) != set(ARTIFACTS)):
        raise ValueError("completed diagnostic inference and frozen source required")
    output.mkdir(parents=True)
    for key, filename in ARTIFACTS.items():
        download(task.artifacts[key], output / filename)
    manifest = json.loads((output / "manifest.json").read_bytes())
    receipt = json.loads((output / "publication-receipt.json").read_bytes())
    expected_manifest = {
        "kind": "v2v4real_overlap_v1_diagnostic_predictions_v1",
        "variant_id": "v2v4real-train-overlap-controlled-v1",
        "protocol_id": "v2v4real-nominal-10hz-formal-v1",
        "dataset_split": "train", "data_role": "train_internal_validate",
        "projection_task_id": PROJECTION_TASK_ID, "projection_sha256": PROJECTION_SHA256,
        "selection_task_id": SELECTION_TASK_ID, "selection_sha256": SELECTION_SHA256,
        "selected_projection_sha256": SELECTION_MANIFEST_SHA256,
        "checkpoint_task_id": checkpoint_task, "checkpoint_sha256": checkpoint_sha,
        "checkpoint_seed": args.seed,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
        "sequence_ids": sorted({row["sequence_id"] for row in frames}),
        "rows": 2034, "gt_read": False, "official_test_read": False,
        "paper_metric": False, "paper_eligible": False,
        "formal_independent_test_eligible": False,
    }
    if (not isinstance(manifest, dict) or set(manifest) != set(expected_manifest) | {"rows_sha256"}
            or any(manifest.get(key) != value for key, value in expected_manifest.items())
            or manifest["rows_sha256"] != sha256(output / "predictions.jsonl")
            or receipt.get("kind") != "v2v4real_overlap_v1_diagnostic_prediction_publication_v1"
            or receipt.get("task_id") != args.task_id
            or receipt.get("checkpoint_task_id") != checkpoint_task
            or receipt.get("checkpoint_sha256") != checkpoint_sha
            or receipt.get("checkpoint_seed") != args.seed
            or receipt.get("formal_config_sha256") != FORMAL_CONFIG_SHA256
            or receipt.get("training_receipt_sha256") != dispatch["training_receipt_sha256"]
            or receipt.get("projection_task_id") != PROJECTION_TASK_ID
            or receipt.get("projection_sha256") != PROJECTION_SHA256
            or receipt.get("selection_task_id") != SELECTION_TASK_ID
            or receipt.get("selection_sha256") != SELECTION_SHA256
            or receipt.get("prediction_rows_sha256") != manifest["rows_sha256"]
            or receipt.get("prediction_manifest_sha256") != sha256(output / "manifest.json")
            or receipt.get("gt_read") is not False
            or receipt.get("official_test_read") is not False
            or receipt.get("paper_eligible") is not False
            or receipt.get("formal_independent_test_eligible") is not False):
        raise ValueError("diagnostic prediction manifest or publication receipt differs")
    count = verify_rows(output / "predictions.jsonl", frames)
    frozen = {"kind": "v2v4real_overlap_v1_diagnostic_predictions_byte_freeze_v1",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "task_id": args.task_id, "source_task_id": SOURCE_TASK_ID,
              "source_sha256": SOURCE_SHA256,
              "seed": args.seed,
              "formal_config_sha256": FORMAL_CONFIG_SHA256,
              "dispatch_receipt_sha256": sha256(dispatch_path),
              "selection_task_id": SELECTION_TASK_ID,
              "selection_sha256": SELECTION_SHA256,
              "projection_task_id": PROJECTION_TASK_ID,
              "projection_sha256": PROJECTION_SHA256,
              "checkpoint_task_id": checkpoint_task,
              "checkpoint_sha256": checkpoint_sha,
              "prediction_rows": count,
              "prediction_rows_sha256": manifest["rows_sha256"],
              "manifest_sha256": sha256(output / "manifest.json"),
              "publication_receipt_sha256": sha256(output / "publication-receipt.json"),
              "all_artifacts_independently_streamed_and_hashed": True,
              "exact_train_internal_validate_coverage_verified": True,
              "gt_read": False, "official_test_read": False,
              "paper_eligible": False, "formal_independent_test_eligible": False}
    destination = output / "diagnostic-prediction-byte-freeze-receipt.json"
    destination.write_text(json.dumps(frozen, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"task_id": args.task_id, "rows": count,
                      "freeze_receipt_sha256": sha256(destination),
                      "paper_eligible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
