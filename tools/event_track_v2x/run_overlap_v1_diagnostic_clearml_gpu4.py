#!/usr/bin/env python3
"""Four-GPU GT-free diagnostic validate inference for approved overlap v1."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


PROJECT = "Thesis/Recover-Before-Fuse/Training"
PROJECTION_TASK = "72599160425d47ca9b7e8916bbb77b42"
PROJECTION_SHA = "2a022771710632db4d1011ded447fd6677e241bdff7c51e4f6d572a0c18523d5"
SELECTION_TASK = "35ae26a544d24a9e9a314f4789e9648f"
SELECTION_SHA = "266593c647c99cd33778de5e35a2c5fb14493f6808631309fb949f0b67d4efaa"
SELECTION_MANIFEST_SHA = "b5e775cb22ef48ff66ed819d865620bc9f07236aea794561b6ee3510c57d8977"
CHECKPOINT_TASK_BY_SEED = {
    1337: "669833fb759742a4a857f99f87341e0b",
    2027: "f945af3976e34a479e40b56ef2a7be24",
    3407: "fd7a87ec5ffa43c3b28efc48bdc9b7e3",
}
FORMAL_CONFIG_SHA256 = "40479859c407db27cc7e609c889a447057319905f94e887769ef919d57004334"


def selected_checkpoint(parameters):
    value = parameters.get("General/checkpoint_seed")
    if value not in {"1337", "2027", "3407"}:
        raise ValueError("supported diagnostic seed required")
    seed = int(value)
    task_id = parameters.get("General/checkpoint_task_id")
    checkpoint_sha = parameters.get("General/checkpoint_sha256")
    receipt_sha = parameters.get("General/training_receipt_sha256")
    if (task_id != CHECKPOINT_TASK_BY_SEED[seed]
            or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                   for digest in (checkpoint_sha, receipt_sha))):
        raise ValueError("frozen diagnostic checkpoint selection differs")
    return seed, task_id, checkpoint_sha, receipt_sha


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def status(stage, **details):
    print(json.dumps({"kind": "rbf_experiment_progress_v1", "stage": stage,
                      "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                      "eta_seconds": None, "estimated_finish_utc": None,
                      "eta_status": "warming_up", "experiment_acceptance_proven": False,
                      **details}, sort_keys=True), flush=True)


def validate_task(task, ident, key, digest):
    from clearml import Task
    upstream = Task.get_task(task_id=ident)
    artifact = upstream.artifacts.get(key)
    if (str(upstream.status) != "completed" or artifact is None
            or artifact.hash != digest):
        raise ValueError("upstream ClearML artifact differs: " + ident + "/" + key)
    return upstream


def materialize_selection(full_inputs, selection_root, selected_inputs):
    from transvision.models.event_track_v2x.v2v4real_inputs import load_prepared_frames

    manifest_path = selection_root / "manifest.json"
    frames_path = selection_root / "frames.jsonl"
    if (sha256(manifest_path) != SELECTION_MANIFEST_SHA
            or selected_inputs.exists() or selected_inputs.is_symlink()):
        raise ValueError("selected manifest identity or fresh output differs")
    selected_inputs.mkdir()
    for name in ("manifest.json", "frames.jsonl"):
        (selected_inputs / name).write_bytes((selection_root / name).read_bytes())
    frames = [json.loads(line) for line in frames_path.read_bytes().splitlines()]
    if len(frames) != 2034:
        raise ValueError("selected frame count differs")
    paths = set()
    for row in frames:
        raw = row.get("pcd_path")
        if not isinstance(raw, str):
            raise ValueError("invalid selected point cloud path")
        pure = PurePosixPath(raw)
        if (pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 4
                or pure.parts[0] != "pcd" or not raw.endswith(".pcd")
                or raw in paths):
            raise ValueError("invalid or duplicate selected point cloud path")
        paths.add(raw)
        source = full_inputs / raw
        target = selected_inputs / raw
        if source.is_symlink() or not source.is_file():
            raise ValueError("selected point cloud is missing from published train projection")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target, follow_symlinks=False)
    manifest, checked = load_prepared_frames(selected_inputs,
                                              expected_manifest_sha256=SELECTION_MANIFEST_SHA)
    if (manifest["dataset_split"] != "train" or len(checked) != 2034
            or manifest["paired_frame_count"] != 1017
            or manifest["gt_in_projection"] is not False):
        raise ValueError("selected train-only projection validation differs")
    return manifest


def merge(shards, output, selected_inputs, seed, checkpoint_task, checkpoint_sha):
    if output.exists() or output.is_symlink():
        raise ValueError("fresh merged output required")
    output.mkdir()
    selected_frames = [json.loads(line) for line in
                       (selected_inputs / "frames.jsonl").read_bytes().splitlines()]
    expected = {(row["sequence_id"], row["frame_ordinal"],
                 "vehicle-side" if row["is_ego"] else "infrastructure-side")
                for row in selected_frames}
    if len(expected) != 2034:
        raise ValueError("selected projection frame identities differ")
    all_rows = []
    seen = set()
    sequences = set()
    for rank, shard in enumerate(shards):
        manifest = json.loads((shard / "manifest.json").read_bytes())
        raw = (shard / "predictions.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        if (manifest.get("kind") != "v2v4real_overlap_v1_diagnostic_prediction_shard_v1"
                or manifest.get("shard_rank") != rank
                or manifest.get("shard_world_size") != 4
                or manifest.get("checkpoint_seed") != seed
                or manifest.get("checkpoint_sha256") != checkpoint_sha
                or manifest.get("formal_config_sha256") != FORMAL_CONFIG_SHA256
                or manifest.get("selected_projection_sha256") != SELECTION_MANIFEST_SHA
                or manifest.get("rows") != len(rows)
                or manifest.get("rows_sha256") != hashlib.sha256(raw).hexdigest()
                or manifest.get("gt_read") is not False
                or manifest.get("official_test_read") is not False
                or manifest.get("paper_metric") is not False
                or manifest.get("paper_eligible") is not False
                or manifest.get("formal_independent_test_eligible") is not False):
            raise ValueError("diagnostic prediction shard differs")
        expected_shard_sequences = sorted({row["sequence_id"] for row in selected_frames})[rank::4]
        if manifest.get("sequence_ids") != expected_shard_sequences:
            raise ValueError("diagnostic prediction shard sequence assignment differs")
        for row in rows:
            identity = (row["sequence_id"], row["frame_ordinal"], row["side"])
            if identity in seen:
                raise ValueError("duplicate prediction row")
            seen.add(identity)
            sequences.add(row["sequence_id"])
        all_rows.extend(rows)
    if len(all_rows) != 2034 or len(sequences) != 6 or seen != expected:
        raise ValueError("merged train-internal validate prediction coverage differs")
    all_rows.sort(key=lambda row: (row["sequence_id"], row["frame_ordinal"], row["side"]))
    raw = b"".join(canonical(row) + b"\n" for row in all_rows)
    (output / "predictions.jsonl").write_bytes(raw)
    manifest = {"kind": "v2v4real_overlap_v1_diagnostic_predictions_v1",
                "variant_id": "v2v4real-train-overlap-controlled-v1",
                "protocol_id": "v2v4real-nominal-10hz-formal-v1",
                "dataset_split": "train", "data_role": "train_internal_validate",
                "projection_task_id": PROJECTION_TASK,
                "projection_sha256": PROJECTION_SHA,
                "selection_task_id": SELECTION_TASK,
                "selection_sha256": SELECTION_SHA,
                "selected_projection_sha256": SELECTION_MANIFEST_SHA,
                "checkpoint_task_id": checkpoint_task,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_seed": seed,
                "formal_config_sha256": FORMAL_CONFIG_SHA256,
                "sequence_ids": sorted(sequences), "rows": len(all_rows),
                "rows_sha256": hashlib.sha256(raw).hexdigest(),
                "gt_read": False, "official_test_read": False,
                "paper_metric": False, "paper_eligible": False,
                "formal_independent_test_eligible": False}
    (output / "manifest.json").write_bytes(canonical(manifest))
    return manifest


def main():
    from clearml import Task
    import torch
    from transvision.models.event_track_v2x.paper_nominal_clock import validate_formal_variant
    from tools.event_track_v2x.run_v2v4real_calibration_clearml import (
        download, extract_regular, install_headless_open3d_stub)

    project = Path(__file__).resolve().parents[2]
    protocol_path = project / "configs/event_track_v2x/v2v4real-nominal-10hz-formal-v1.json"
    if (protocol_path.is_symlink() or not protocol_path.is_file()
            or sha256(protocol_path) != FORMAL_CONFIG_SHA256):
        raise ValueError("frozen nominal variant config differs before task initialization")
    protocol = validate_formal_variant(json.loads(protocol_path.read_bytes()))
    if protocol["protocol_id"] != "v2v4real-nominal-10hz-formal-v1":
        raise ValueError("frozen nominal variant ID differs")
    task = Task.init(project_name=PROJECT,
                     task_name="V2V4Real overlap v1 diagnostic validate inference",
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    task.reload()
    seed, checkpoint_task, checkpoint_sha, training_receipt_sha = selected_checkpoint(
        task.get_parameters())
    worker = os.environ.get("CLEARML_WORKER_ID", "")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    binding = re.fullmatch(r"[^:]+:gpu(\d+),(\d+),(\d+),(\d+)", worker)
    if (binding is None or len(set(binding.groups())) != 4 or len(names) != 4
            or any(not name for name in names)):
        raise ValueError("exactly four bound CUDA GPUs required")
    for ident, key, digest in ((PROJECTION_TASK, "projection", PROJECTION_SHA),
                               (SELECTION_TASK, "selection", SELECTION_SHA),
                               (checkpoint_task, "best-checkpoint", checkpoint_sha),
                               (checkpoint_task, "training-receipt", training_receipt_sha)):
        validate_task(task, ident, key, digest)
    root = Path(tempfile.mkdtemp(prefix="rbf-overlap-v1-diagnostic-inference-"))
    status("download_train_projection", completed=0, total=4)
    projection = download(PROJECTION_TASK, "projection", PROJECTION_SHA,
                          root / "projection.tar.gz")
    status("download_train_projection", completed=1, total=4)
    selection = download(SELECTION_TASK, "selection", SELECTION_SHA,
                         root / "selection.tar.gz")
    checkpoint = download(checkpoint_task, "best-checkpoint", checkpoint_sha,
                          root / "best.pth")
    training_receipt = download(checkpoint_task, "training-receipt",
                                training_receipt_sha, root / "training-receipt.json")
    status("download_frozen_inputs", completed=4, total=4)
    extracted = root / "extracted"
    extract_regular(projection, extracted, maximum_bytes=64 * 1024**3)
    full_inputs = extracted / "projection/inputs"
    if not full_inputs.is_dir():
        raise ValueError("published train projection lacks canonical inputs")
    selection_root = root / "selected-archive"
    extract_regular(selection, selection_root, maximum_bytes=5 * 1024**2,
                    allowed_roots={"manifest.json", "frames.jsonl"})
    sys.path.insert(0, str(project))
    selected_inputs = root / "selected-inputs"
    materialize_selection(full_inputs, selection_root, selected_inputs)
    status("selected_train_projection_hash_verified", completed=2034, total=2034)
    install_headless_open3d_stub(project)
    official = project.parent / "official"
    runner = project / "tools/event_track_v2x/run_overlap_v1_diagnostic_predictions.py"
    shards = []
    processes = []
    for rank in range(4):
        output = root / f"shard-{rank}"
        shards.append(output)
        command = [sys.executable, str(runner), "--source", str(official),
                   "--inputs-root", str(selected_inputs), "--checkpoint", str(checkpoint),
                   "--checkpoint-sha256", checkpoint_sha,
                   "--training-receipt", str(training_receipt),
                   "--training-receipt-sha256", training_receipt_sha,
                   "--seed", str(seed), "--output", str(output),
                   "--device", "cuda:0", "--rank", str(rank), "--world-size", "4"]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), PYTHONPATH=str(project))
        processes.append(subprocess.Popen(command, env=environment))
    failures = [process.wait() for process in processes]
    if any(failures):
        raise RuntimeError("one or more diagnostic inference shards failed: " + repr(failures))
    output = root / "merged"
    manifest = merge(shards, output, selected_inputs, seed, checkpoint_task, checkpoint_sha)
    receipt = {"kind": "v2v4real_overlap_v1_diagnostic_prediction_publication_v1",
               "task_id": task.id, "worker_id": worker, "gpu_names": names,
               "command_entry": runner.name, "checkpoint_task_id": checkpoint_task,
               "checkpoint_sha256": checkpoint_sha, "checkpoint_seed": seed,
               "training_receipt_sha256": training_receipt_sha,
               "formal_config_sha256": FORMAL_CONFIG_SHA256,
               "projection_task_id": PROJECTION_TASK,
               "projection_sha256": PROJECTION_SHA,
               "selection_task_id": SELECTION_TASK, "selection_sha256": SELECTION_SHA,
               "prediction_rows_sha256": manifest["rows_sha256"],
               "prediction_manifest_sha256": sha256(output / "manifest.json"),
               "gt_read": False, "official_test_read": False,
               "paper_eligible": False, "formal_independent_test_eligible": False}
    receipt_path = output / "publication-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    for key, path in (("diagnostic-prediction-manifest", output / "manifest.json"),
                      ("diagnostic-predictions", output / "predictions.jsonl"),
                      ("diagnostic-publication-receipt", receipt_path)):
        if not task.upload_artifact(key, artifact_object=path, wait_on_upload=True):
            raise RuntimeError("diagnostic output upload failed: " + key)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
