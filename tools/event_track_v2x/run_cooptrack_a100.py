#!/usr/bin/env python3
"""Run batch sizing and full-train detector fits on four assigned A100s."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import signal
import subprocess
import tarfile
import time


ARTIFACT_NAMES = {
    "runtime.tar.gz": "runtime",
    "train-inputs.tar.gz": "train-inputs",
    "source.tar.gz": "source",
    "resnet50-0676ba61.pth": "pretrained",
}
LEGACY_PYTHON = "/opt/cooptrack/bin/python"
RUNNER = "/entrypoints/run_cooptrack_detector.py"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def validate_archive(path, prefixes, allow_links=False):
    """Validate paths before extracting a package with a separately pinned hash."""
    count = 0
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe archive member path")
            if not any(str(name) == prefix or str(name).startswith(prefix + "/")
                       for prefix in prefixes):
                raise ValueError("archive member outside the package prefixes")
            if entry.isdev() or entry.isfifo():
                raise ValueError("archive contains special device or FIFO")
            if entry.issym() or entry.islnk():
                if not allow_links:
                    raise ValueError("unexpected link in train or source archive")
                link = PurePosixPath(entry.linkname)
                target = str(link) if link.is_absolute() or entry.islnk() else str(name.parent / link)
                normalized = os.path.normpath("/" + target.lstrip("/"))
                if not any(normalized == "/" + prefix or normalized.startswith("/" + prefix + "/")
                           for prefix in prefixes):
                    raise ValueError("archive link escapes the authorized runtime prefix")
            count += 1
    if not count:
        raise ValueError("empty archive")
    return count


def choose_batch(profiles, memory_fraction=0.8):
    eligible = []
    for profile in profiles:
        reports = profile.get("ranks", [])
        if profile.get("success") and len(reports) == 4 and all(
                item["peak_reserved_bytes"] <= item["total_memory_bytes"] * memory_fraction
                for item in reports):
            eligible.append(profile["batch_per_gpu"])
    if not eligible:
        raise RuntimeError("no candidate batch passed four-rank memory and optimizer checks")
    return max(eligible)


def materialize(package_task, manifest_sha256):
    manifest_file = Path(package_task.artifacts["package-manifest"].get_local_copy())
    if sha256(manifest_file) != manifest_sha256:
        raise ValueError("package manifest does not match submitted SHA-256")
    manifest = json.loads(manifest_file.read_bytes())
    if (manifest.get("kind") != "eventtrack_a100_migration_package_v1"
            or manifest.get("train_sequence_count") != 46
            or manifest.get("val_or_test_included") is not False):
        raise ValueError("package is not the authorized full-train-only package")
    if {item["path"] for item in manifest["inventory"]} != set(ARTIFACT_NAMES):
        raise ValueError("unexpected package inventory")
    destinations = ["/opt/cooptrack", "/inputs", "/converted", "/workspace/CoopTrack", "/entrypoints", "/pretrained"]
    for destination in destinations:
        if Path(destination).exists():
            raise FileExistsError("refusing to replace existing runtime destination: " + destination)
    for item in manifest["inventory"]:
        source = Path(package_task.artifacts[ARTIFACT_NAMES[item["path"]]].get_local_copy())
        if source.stat().st_size != item["bytes"] or sha256(source) != item["sha256"]:
            raise ValueError("package artifact hash or size mismatch")
        if item["path"] == "runtime.tar.gz":
            prefixes = ["opt/cooptrack", "usr/local/cuda-11.8/targets/x86_64-linux/lib"]
            validate_archive(source, prefixes, allow_links=True)
        elif item["path"] == "train-inputs.tar.gz":
            validate_archive(source, ["inputs", "converted"])
        elif item["path"] == "source.tar.gz":
            validate_archive(source, ["workspace/CoopTrack", "entrypoints"])
        else:
            Path("/pretrained").mkdir()
            shutil.copy2(source, "/pretrained/resnet50.pth")
            continue
        subprocess.run(["tar", "--no-same-owner", "-xzf", str(source), "-C", "/"], check=True)
        print("EVENTTRACK_MATERIALIZED " + item["path"], flush=True)
    return manifest


def training_command(side, batch, output, probe_iters):
    command = [LEGACY_PYTHON, "-m", "torch.distributed.launch", "--nproc_per_node=4",
               "--master_port=29571", RUNNER, "--upstream-root", "/workspace/CoopTrack",
               "--inputs", "/inputs", "--converted", "/converted", "--pretrained", "/pretrained/resnet50.pth",
               "--output", str(output), "--side", side, "--batch-size", str(batch),
               "--accumulation", "1", "--epochs", "24", "--seed", "1337",
               "--allow-larger-batch", "--require-device", "A100"]
    if probe_iters:
        command += ["--batch-probe-iters", str(probe_iters)]
    return command


def run_phase(task, side, batch, output, env, probe_iters=0):
    label = f"probe-b{batch}-{side}" if probe_iters else f"train-{side}"
    log_path = output.with_suffix(".log")
    command = training_command(side, batch, output, probe_iters)
    last_uploaded = None
    startup_uploaded = False
    started = time.monotonic()
    process = subprocess.Popen(command, env=env, cwd="/workspace/CoopTrack",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    with log_path.open("x") as log:
        try:
            while process.poll() is None or selector.get_map():
                if probe_iters and time.monotonic() - started > 1800:
                    raise TimeoutError("batch sizing exceeded 30 minutes")
                for key, _ in selector.select(timeout=1):
                    line = key.fileobj.readline()
                    if not line:
                        selector.unregister(key.fileobj)
                        continue
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    match = re.search(r"Iter \[(\d+)/\d+\].*?\bloss: ([0-9.eE+-]+)", line)
                    if match:
                        task.get_logger().report_scalar(label, "loss", iteration=int(match[1]), value=float(match[2]))
                    if not probe_iters and "EVENTTRACK_OPTIMIZER_STARTED " in line and not startup_uploaded:
                        for name in ["optimizer-startup.json", "startup-iter-16.pth", "launch-receipt.json", "detector.py"]:
                            task.upload_artifact(side + "-" + name, artifact_object=output / name, wait_on_upload=True)
                        startup_uploaded = True
                    if not probe_iters and match:
                        checkpoints = sorted(output.glob("iter_*.pth"), key=lambda item: int(item.stem.split("_")[1]))
                        if checkpoints:
                            latest = checkpoints[-1]
                            if latest.name != last_uploaded and time.time() - latest.stat().st_mtime > 1:
                                task.upload_artifact(side + "-latest-checkpoint", artifact_object=latest, wait_on_upload=True)
                                last_uploaded = latest.name
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            selector.close()
    status = process.wait()
    reports = [json.loads(path.read_bytes()) for path in sorted(output.glob("gpu-memory-rank-*.json"))]
    profile = {"side": side, "batch_per_gpu": batch, "world_size": 4,
               "effective_batch_size": batch * 4, "success": status == 0,
               "returncode": status, "elapsed_seconds": time.monotonic() - started,
               "ranks": reports, "batch_probe_only": bool(probe_iters)}
    if probe_iters:
        profile["success"] = profile["success"] and (output / "optimizer-startup.json").is_file()
    else:
        profile["success"] = profile["success"] and startup_uploaded
    write_json(output.with_suffix(".result.json"), profile)
    task.upload_artifact(label + "-profile", artifact_object=profile, wait_on_upload=True)
    return profile


def parse_package_arguments(parameters, argv=None):
    """Accept ClearML's General section as well as legacy unsectioned values."""
    parser = argparse.ArgumentParser()
    for name in ["package_task_id", "package_manifest_sha256"]:
        default = parameters.get("General/" + name, parameters.get(name))
        parser.add_argument("--" + name.replace("_", "-"), default=default)
    args = parser.parse_args(argv)
    if not args.package_task_id or not args.package_manifest_sha256:
        raise ValueError("package ID and manifest SHA-256 are required")
    return args


def main():
    from clearml import Task
    task = Task.init(project_name="Thesis/EventTrack-V2X/Training",
                     task_name="D2 A100 full-train detector training",
                     reuse_last_task_id=False, auto_connect_frameworks=False,
                     auto_connect_arg_parser=False)
    args = parse_package_arguments(task.get_parameters())
    package = Task.get_task(task_id=args.package_task_id)
    materialize(package, args.package_manifest_sha256)
    env = dict(os.environ)
    env.update({"PATH": "/opt/cooptrack/bin:" + env.get("PATH", ""),
                "PYTHONPATH": "/workspace/CoopTrack", "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
                "LD_LIBRARY_PATH": "/opt/cooptrack/lib:/opt/cooptrack/lib/python3.8/site-packages/torch/lib:/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
                "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1", "MPLCONFIGDIR": "/tmp/eventtrack-matplotlib"})
    # Explicitly remove controller-venv activation from the legacy subprocess.
    env.pop("VIRTUAL_ENV", None)
    root = Path("/eventtrack-a100-runs")
    root.mkdir()
    readiness = subprocess.check_output([
        LEGACY_PYTHON, "-c",
        "import torch,mmcv,mmdet,mmdet3d; import mmcv._ext; "
        "assert torch.cuda.device_count()==4; "
        "assert all('A100' in torch.cuda.get_device_name(i) for i in range(4)); "
        "print(torch.__version__,mmcv.__version__,mmdet3d.__version__)"], env=env, text=True)
    print("EVENTTRACK_A100_LEGACY_READY " + readiness.strip(), flush=True)
    profiles = []
    # 4 ranks x 10 frames is the largest tested batch below 46 causal streams.
    for batch in [2, 4, 8, 10]:
        profile = run_phase(task, "vehicle-side", batch, root / f"probe-vehicle-b{batch}", env, probe_iters=32)
        profiles.append(profile)
        if not profile["success"]:
            log = (root / f"probe-vehicle-b{batch}.log").read_text(errors="replace")
            if "out of memory" not in log.lower():
                raise RuntimeError("batch probe failed for a non-memory error")
            break
    selected = choose_batch(profiles)
    # Infrastructure has a different object count; it must pass the same sizing gate.
    while True:
        infra_profile = run_phase(task, "infrastructure-side", selected,
                                  root / f"probe-infrastructure-b{selected}", env, probe_iters=32)
        if not infra_profile["success"]:
            log = (root / f"probe-infrastructure-b{selected}.log").read_text(errors="replace")
            if "out of memory" not in log.lower():
                raise RuntimeError("infrastructure batch probe failed for a non-memory error")
        try:
            infra_eligible = choose_batch([infra_profile]) == selected
        except RuntimeError:
            infra_eligible = False
        if infra_eligible:
            break
        candidates = [item for item in profiles if item["batch_per_gpu"] < selected]
        selected = choose_batch(candidates)
    selection = {"batch_per_gpu": selected, "world_size": 4, "effective_batch_size": selected * 4,
                 "maximum_memory_fraction": 0.8, "sequence_stream_limit": 46,
                 "base_learning_rate_unchanged": True, "vehicle_profiles": profiles,
                 "infrastructure_profile": infra_profile,
                 "probe_weights_not_used_for_training": True}
    write_json(root / "batch-selection.json", selection)
    task.upload_artifact("batch-selection", artifact_object=selection, wait_on_upload=True)
    print("EVENTTRACK_A100_BATCH_SELECTED " + json.dumps(selection), flush=True)
    for side in ["vehicle-side", "infrastructure-side"]:
        output = root / side
        result = run_phase(task, side, selected, output, env)
        if not result["success"]:
            raise RuntimeError(side + " full-train detector training failed")
        checkpoints = sorted(output.glob("iter_*.pth"), key=lambda item: int(item.stem.split("_")[1]))
        if not checkpoints:
            raise RuntimeError("completed run has no trained checkpoint")
        final = checkpoints[-1]
        task.upload_artifact(side + "-final-checkpoint", artifact_object=final, wait_on_upload=True)
        receipt = {"side": side, "checkpoint": final.name, "sha256": sha256(final),
                   "epochs": 24, "train_sequences": 46, "batch_per_gpu": selected,
                   "effective_batch_size": selected * 4, "seed": 1337,
                   "stage": "D2-detector-training-only", "official_val_result_available": False}
        task.upload_artifact(side + "-completion", artifact_object=receipt, wait_on_upload=True)
    task.close()


if __name__ == "__main__":
    main()
