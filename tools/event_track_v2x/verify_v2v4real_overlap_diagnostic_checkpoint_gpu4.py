#!/usr/bin/env python3
"""Four-CUDA-GPU tensor and forward acceptance for a diagnostic checkpoint.

This task reads no dataset or GT and reports no paper metric.  It verifies the
diagnostic checkpoint envelope, loads the frozen official model, and runs the same
fixed voxel fixture independently on every assigned GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import time
import venv
from datetime import datetime, timedelta, timezone

TRAINING_BY_SEED = {
    1337: ("669833fb759742a4a857f99f87341e0b", "7f58517f4c62452cbef9b98c79ff1420",
           "93b3f63cb1058c725224e275c713566e2a11090812026060684c77867a3c3a69"),
    2027: ("f945af3976e34a479e40b56ef2a7be24", "7f58517f4c62452cbef9b98c79ff1420",
           "93b3f63cb1058c725224e275c713566e2a11090812026060684c77867a3c3a69"),
    3407: ("fd7a87ec5ffa43c3b28efc48bdc9b7e3", "79aeb14f8d4641f381150c3f912f623f",
           "122157453fa481ce27e9adb97528990bab439c8b9e30d5687229da6a57744f1a"),
}
OFFICIAL_COMMIT = "5a821e13753bafc611f95c47bc1a306acdcb0f7c"
PROTOCOL_ID = "v2v4real-nominal-10hz-formal-v1"
VARIANT_ID = "v2v4real-train-overlap-controlled-v1"
CONFIG_SHA256 = "138c4ad3508fdd7061f5b290c83ad8f0772fea33f0af0e2be56330423a3c92d1"
RUNTIME_PACKAGES = ("numpy==1.26.4", "scipy==1.14.1", "PyYAML==6.0.2",
                    "spconv-cu126==2.3.8", "cumm-cu126==0.7.11",
                    "open3d==0.19.0", "shapely==2.0.7")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection(parameters):
    seed = parameters.get("General/seed")
    checkpoint_sha = parameters.get("General/checkpoint_sha256")
    if str(seed) not in {str(value) for value in TRAINING_BY_SEED}:
        raise ValueError("supported diagnostic seed required")
    if (not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint_sha)):
        raise ValueError("frozen checkpoint SHA256 required")
    return int(seed), checkpoint_sha


def progress(done, started):
    elapsed = time.monotonic() - started
    rate = done / elapsed if done and elapsed > 0 else None
    remaining = (4 - done) / rate if rate else None
    print(json.dumps({"kind": "rbf_experiment_progress_v1",
                      "stage": "diagnostic_checkpoint_gpu_forward_devices",
                      "completed": done, "total": 4, "progress_percent": 25 * done,
                      "elapsed_seconds": elapsed, "units_per_second": rate,
                      "eta_seconds": remaining,
                      "estimated_finish_utc": ((datetime.now(timezone.utc)
                          + timedelta(seconds=remaining)).isoformat()
                          if remaining is not None else None),
                      "eta_status": "estimated" if rate else "warming_up",
                      "eta_method": "cumulative_wall_clock_rate_variable_workload",
                      "experiment_acceptance_proven": False}, sort_keys=True), flush=True)


def download(artifact, expected, target):
    from clearml.storage.helper import StorageHelper
    url = artifact.url
    if "10.100.35.118:8081" in url:
        url = url.replace("10.100.35.118:8081", "10.100.34.118:8081", 1)
    elif "10.100.34.118:8081" not in url:
        raise ValueError("artifact is outside the pinned ClearML file service")
    target = Path(target)
    size = 0
    with target.open("xb") as stream:
        for chunk in StorageHelper.get(url).download_as_stream(url):
            stream.write(chunk)
            size += len(chunk)
    if artifact.hash != expected or size != artifact.size or sha256(target) != expected:
        raise ValueError("downloaded artifact identity differs")
    return target


def extract_source(archive, target):
    target = Path(target)
    target.mkdir(exist_ok=False)
    total = 0
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or not member.isfile()
                    or not name.parts or name.parts[0] not in {"project", "official"}):
                raise ValueError("unsafe source archive member")
            total += member.size
            if total > 128 * 1024**2:
                raise ValueError("source archive exceeds bound")
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, path.open("xb") as output:
                while chunk := source.read(1024**2):
                    output.write(chunk)
    marker = target / "official/.official-commit"
    if marker.read_text().strip() != OFFICIAL_COMMIT:
        raise ValueError("official source commit differs")


def fixture_cloud():
    import numpy as np
    close = [[0.05, 0.05, 0.0, index / 40.0] for index in range(35)]
    return np.asarray(close + [[1.0, 1.0, 0.1, 0.2], [11.0, 2.0, 0.2, 0.4]],
                      dtype=np.float32)


def ensure_runtime():
    if os.environ.get("RBF_CHECKPOINT_RUNTIME") == "1":
        return
    root = Path(tempfile.mkdtemp(prefix="rbf-v2v4real-checkpoint-runtime-"))
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root / "venv")
    python = root / "venv/bin/python"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("PIP_")}
    environment.update(PIP_CONFIG_FILE=os.devnull, RBF_CHECKPOINT_RUNTIME="1",
                       LD_LIBRARY_PATH=("/opt/nvidia/nsight-compute/2024.3.2/host/"
                                        "linux-desktop-glibc_2_11_3-x64/Mesa"))
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                    "--index-url", "http://10.100.34.109:3141/root/pypi/+simple/",
                    "--trusted-host", "10.100.34.109", "--only-binary=:all:",
                    *RUNTIME_PACKAGES], env=environment, check=True, timeout=1800)
    os.execve(str(python), [str(python), str(Path(__file__).resolve())], environment)


def main():
    ensure_runtime()
    from clearml import Task
    import numpy as np
    import torch
    import yaml

    task = Task.init(project_name="Thesis/Recover-Before-Fuse/Training",
                     task_name="V2V4Real diagnostic checkpoint 4xGPU acceptance",
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    task.reload()
    seed, checkpoint_sha = selection(task.get_parameters())
    checkpoint_id, source_id, source_sha = TRAINING_BY_SEED[seed]
    worker = os.environ.get("CLEARML_WORKER_ID", "")
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    binding = re.fullmatch(r"[^:]+:gpu(\d+),(\d+),(\d+),(\d+)", worker)
    if (binding is None or len(set(binding.groups())) != 4 or len(names) != 4
            or any(not name for name in names)):
        raise ValueError("exactly four bound CUDA GPUs required")
    root = Path(tempfile.mkdtemp(prefix="rbf-v2v4real-checkpoint-"))
    source_task = Task.get_task(task_id=source_id)
    checkpoint_task = Task.get_task(task_id=checkpoint_id)
    if str(source_task.status) != "completed" or str(checkpoint_task.status) != "completed":
        raise ValueError("upstream source or checkpoint task is incomplete")
    source = download(source_task.artifacts["source"], source_sha, root / "source.tar.gz")
    checkpoint = download(checkpoint_task.artifacts["best-checkpoint"], checkpoint_sha,
                          root / "best.pth")
    extract_source(source, root / "source")
    official = root / "source/official"
    sys.path.insert(0, str(official))
    from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
    from opencood.hypes_yaml.yaml_utils import load_point_pillar_params
    from opencood.models.point_pillar import PointPillar

    config_path = official / "opencood/hypes_yaml/point_pillar_late_fusion.yaml"
    if sha256(config_path) != CONFIG_SHA256:
        raise ValueError("frozen diagnostic detector configuration differs")
    config = load_point_pillar_params(yaml.safe_load(config_path.read_text()))
    preprocessor = SpVoxelPreprocessor(config["preprocess"], train=False)
    processed = preprocessor.collate_batch([preprocessor.preprocess(fixture_cloud())])
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True)
    receipt = download(checkpoint_task.artifacts["training-receipt"],
                       checkpoint_task.artifacts["training-receipt"].hash,
                       root / "training-receipt.json")
    training = json.loads(receipt.read_text())
    selected = training.get("selected")
    if (training.get("seed") != seed or training.get("variant_id") != VARIANT_ID
            or training.get("paper_eligible") is not False
            or training.get("formal_independent_test_eligible") is not False
            or not isinstance(selected, dict)):
        raise ValueError("diagnostic training receipt differs")
    if (set(envelope) != {"model", "optimizer", "epoch", "seed", "validation_loss",
                          "protocol_id", "official_commit", "variant_id", "paper_eligible",
                          "formal_independent_test_eligible"}
            or envelope["seed"] != seed or envelope["epoch"] != selected.get("epoch")
            or envelope["validation_loss"] != selected.get("validation_loss")
            or envelope["protocol_id"] != PROTOCOL_ID
            or envelope["official_commit"] != OFFICIAL_COMMIT
            or envelope["variant_id"] != VARIANT_ID
            or envelope["paper_eligible"] is not False
            or envelope["formal_independent_test_eligible"] is not False
            or not isinstance(envelope["model"], dict)
            or not envelope["model"]
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                   for value in envelope["model"].values())):
        raise ValueError("diagnostic checkpoint envelope differs")
    reports = []
    reference = None
    started = time.monotonic()
    progress(0, started)
    for index, name in enumerate(names):
        model = PointPillar(config["model"]["args"]).to(f"cuda:{index}").eval()
        model.load_state_dict(envelope["model"], strict=True)
        batch = {key: torch.as_tensor(value, device=f"cuda:{index}")
                 for key, value in processed.items()}
        with torch.inference_mode():
            output = model({"processed_lidar": batch})
        torch.cuda.synchronize(index)
        arrays = {key: value.detach().cpu().numpy() for key, value in output.items()}
        if set(arrays) != {"psm", "rm"} or not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("detector raw heads differ")
        if reference is None:
            reference = arrays
        maximum = {key: float(np.max(np.abs(value - reference[key]))) for key, value in arrays.items()}
        if any(not np.allclose(value, reference[key], atol=1e-6, rtol=1e-5)
               for key, value in arrays.items()):
            raise ValueError("cross-device checkpoint forward differs")
        reports.append({"index": index, "name": name, "head_shapes":
                        {key: list(value.shape) for key, value in arrays.items()},
                        "max_abs_difference_to_first": maximum})
        progress(index + 1, started)
    report = {"kind": "v2v4real_overlap_diagnostic_checkpoint_gpu4_acceptance_v1",
              "task_id": task.id, "worker_id": worker, "source_task_id": source_id,
              "source_sha256": source_sha, "checkpoint_task_id": checkpoint_id,
              "checkpoint_sha256": checkpoint_sha, "seed": seed,
              "epoch": envelope["epoch"], "validation_loss": envelope["validation_loss"],
              "protocol_id": PROTOCOL_ID, "variant_id": VARIANT_ID,
              "official_commit": OFFICIAL_COMMIT,
              "devices": reports, "dataset_read": False, "gt_read": False,
              "paper_metric_reported": False, "diagnostic_checkpoint_envelope_verified": True,
              "strict_state_dict_load_verified": True, "raw_head_forward_verified": True,
              "paper_eligible": False, "formal_independent_test_eligible": False}
    path = root / "checkpoint-acceptance.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if not task.upload_artifact("checkpoint-acceptance", artifact_object=path, wait_on_upload=True):
        raise RuntimeError("acceptance artifact upload failed")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
