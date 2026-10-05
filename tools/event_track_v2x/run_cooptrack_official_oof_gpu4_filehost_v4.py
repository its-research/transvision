#!/usr/bin/env python3
"""Run SPD official-train canonical OOF detector fits on four compatible GPUs."""

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
from urllib.parse import urlparse, urlunparse


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


def validate_cohort(manifest):
    if manifest.get("val_or_test_included") is not False:
        raise ValueError("validation/test payload is forbidden")
    if manifest.get("kind") == "eventtrack_spd_official_oof_fold_package_v1":
        fold_id = manifest.get("fold_id")
        sequences = manifest.get("official_train_sequence_ids")
        fit = manifest.get("fit_sequence_ids")
        held = manifest.get("held_out_sequence_ids")
        split_sha = "0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd"
        if (manifest.get("cohort") != "official-oof-fold-fit"
                or type(fold_id) is not int or not 0 <= fold_id < 5
                or not isinstance(sequences, list)
                or not all(isinstance(item, str) and item for item in sequences)
                or len(sequences) != 46 or sequences != sorted(set(sequences))
                or not isinstance(fit, list) or not isinstance(held, list)
                or manifest.get("train_sequence_count") != len(fit)
                or manifest.get("official_split_sha256") != split_sha
                or manifest.get("canonical_fivefold_manifest_sha256")
                   != "1c9d23f291be09e686a2bdcc60ef19ea58a5c2894978cc5f6a6b2a81f43c2bd6"):
            raise ValueError("invalid canonical OOF package identity")
        ranked = sorted(sequences, key=lambda sequence_id: (
            hashlib.sha256(json.dumps({
                "salt": "eventtrack-v2x-spd-development-5fold-v1",
                "sequence_id": sequence_id,
                "split_sha256": split_sha,
            }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            sequence_id))
        expected_held = sorted(item for position, item in enumerate(ranked)
                               if position % 5 == fold_id)
        expected_fit = sorted(set(sequences) - set(expected_held))
        if (held != expected_held or fit != expected_fit
                or len(held) != (10 if fold_id == 0 else 9)
                or len(fit) != (36 if fold_id == 0 else 37)):
            raise ValueError("OOF fit and held-out membership differs from fixed fivefold")
        return "official-oof-fold-fit", len(fit)
    raise ValueError("only a canonical SPD official OOF fold package is admitted")


def batch_candidates(sequence_count):
    return [batch for batch in (2, 4, 8, 10) if batch * 4 <= sequence_count]


def download_registered_artifact(artifact, expected_sha, expected_bytes, target):
    from clearml.storage.helper import StorageHelper
    from clearml.backend_api.session import Session
    registered = urlparse(artifact.url)
    active = urlparse(Session.get_files_server_host())
    hosts = {"10.100.35.118:8081", "10.100.34.118:8081"}
    if (registered.scheme != "http" or active.scheme != "http"
            or registered.netloc not in hosts or active.netloc not in hosts
            or registered.username or registered.query or registered.fragment):
        raise ValueError("unregistered artifact/file-service route")
    if artifact.hash != expected_sha or artifact.size != expected_bytes:
        raise ValueError("registered artifact identity differs")
    url = urlunparse(registered._replace(netloc=active.netloc))
    target = Path(target)
    digest = hashlib.sha256()
    received = 0
    started = time.monotonic()
    print("EVENTTRACK_DOWNLOAD_ETA " + json.dumps({"artifact": target.name,
          "eta_seconds": None, "eta_status": "unknown", "file_service": active.netloc}), flush=True)
    with target.open("xb") as stream:
        for block in StorageHelper.get(url).download_as_stream(url):
            stream.write(block)
            digest.update(block)
            prior = received
            received += len(block)
            if received > expected_bytes:
                raise ValueError("artifact longer than registered byte count")
            if received // (256 * 1024 * 1024) > prior // (256 * 1024 * 1024):
                elapsed = time.monotonic() - started
                print("EVENTTRACK_DOWNLOAD_ETA " + json.dumps({"artifact": target.name,
                      "received_bytes": received, "expected_bytes": expected_bytes,
                      "eta_seconds": (expected_bytes-received)*elapsed/received}), flush=True)
    if received != expected_bytes or digest.hexdigest() != expected_sha:
        raise ValueError("downloaded byte count or SHA differs")
    return target


def materialize(package_task, manifest_sha256):
    downloads = Path("/eventtrack-oof-verified-downloads")
    downloads.mkdir()
    manifest_artifact = package_task.artifacts["package-manifest"]
    manifest_file = download_registered_artifact(manifest_artifact, manifest_sha256,
        manifest_artifact.size, downloads / "package-manifest.json")
    if sha256(manifest_file) != manifest_sha256:
        raise ValueError("package manifest does not match submitted SHA-256")
    manifest = json.loads(manifest_file.read_bytes())
    validate_cohort(manifest)
    if {item["path"] for item in manifest["inventory"]} != set(ARTIFACT_NAMES):
        raise ValueError("unexpected package inventory")
    destinations = ["/opt/cooptrack", "/inputs", "/converted", "/workspace/CoopTrack", "/entrypoints", "/pretrained"]
    for destination in destinations:
        if Path(destination).exists():
            raise FileExistsError("refusing to replace existing runtime destination: " + destination)
    for item in manifest["inventory"]:
        owner = package_task
        if item.get("artifact_task_id"):
            from clearml import Task
            owner = Task.get_task(task_id=item["artifact_task_id"])
        source = download_registered_artifact(owner.artifacts[ARTIFACT_NAMES[item["path"]]],
            item["sha256"], item["bytes"], downloads / item["path"])
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
    inputs = json.loads(Path("/inputs/input-manifest.json").read_bytes())
    converted = json.loads(Path("/converted/conversion-manifest.json").read_bytes())
    if (sha256(Path("/inputs/input-manifest.json")) != manifest["input_manifest_sha256"]
            or sha256(Path("/converted/conversion-manifest.json"))
               != manifest["conversion_manifest_sha256"]
            or inputs["fit_sequence_ids"] != manifest["fit_sequence_ids"]
            or converted["fit_sequence_ids"] != manifest["fit_sequence_ids"]
            or inputs["excluded_held_out_sequence_ids"] != manifest["held_out_sequence_ids"]
            or inputs.get("cohort") != "fit-fold"
            or converted.get("cohort") != "fit-fold"
            or inputs.get("fold_id") != manifest["fold_id"]
            or converted.get("fold_id") != manifest["fold_id"]
            or inputs.get("official_split_sha256") != manifest["official_split_sha256"]
            or inputs.get("development_manifest_sha256")
               != manifest["canonical_fivefold_manifest_sha256"]):
        raise ValueError("materialized OOF fold input or conversion binding differs")
    return manifest


def training_command(side, batch, output, probe_iters, seed=1337):
    if seed not in (1337, 2027, 3407):
        raise ValueError("seed must be one of the three paper seeds")
    command = [LEGACY_PYTHON, "-m", "torch.distributed.launch", "--nproc_per_node=4",
               "--master_port=29571", RUNNER, "--upstream-root", "/workspace/CoopTrack",
               "--inputs", "/inputs", "--converted", "/converted", "--pretrained", "/pretrained/resnet50.pth",
               "--output", str(output), "--side", side, "--batch-size", str(batch),
               "--accumulation", "1", "--epochs", "24", "--seed", str(seed),
               "--allow-larger-batch"]
    if probe_iters:
        command += ["--batch-probe-iters", str(probe_iters)]
    return command


def run_phase(task, side, batch, output, env, probe_iters=0, seed=1337):
    label = f"probe-b{batch}-{side}" if probe_iters else f"train-{side}"
    log_path = output.with_suffix(".log")
    command = training_command(side, batch, output, probe_iters, seed=seed)
    last_uploaded = None
    startup_uploaded = False
    started = time.monotonic()
    previous_progress = None
    last_eta_report = started
    print("EVENTTRACK_PHASE_ETA " + json.dumps({"phase": label, "eta_seconds": None,
          "eta_status": "unknown", "scope": "current phase only",
          "reason": "awaiting iteration progress", "overall_eta": "unknown"}), flush=True)
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
                    progress = re.search(r"Iter \[([0-9]+)/([0-9]+)\]", line)
                    if progress:
                        iteration, total = map(int, progress.groups())
                        now = time.monotonic()
                        eta = None
                        if previous_progress is not None:
                            prior_iteration, prior_time = previous_progress
                            delta = iteration - prior_iteration
                            if delta > 0 and now > prior_time and 0 < iteration <= total:
                                eta = (total - iteration) * (now - prior_time) / delta
                        if previous_progress is None or iteration > previous_progress[0]:
                            previous_progress = (iteration, now)
                        last_eta_report = now
                        print("EVENTTRACK_PHASE_ETA " + json.dumps({"phase": label,
                              "iteration": iteration, "total_iterations": total,
                              "eta_seconds": eta, "eta_status": "unknown" if eta is None else "estimated",
                              "scope": "current phase only; excludes publication and other side",
                              "overall_eta": "unknown"}), flush=True)
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
                if time.monotonic() - last_eta_report >= 60:
                    print("EVENTTRACK_PHASE_ETA " + json.dumps({"phase": label,
                          "eta_seconds": None, "eta_status": "unknown",
                          "reason": "no recent iteration progress", "overall_eta": "unknown"}), flush=True)
                    last_eta_report = time.monotonic()
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
    print("EVENTTRACK_PHASE_ETA " + json.dumps({"phase": label,
          "eta_seconds": 0 if status == 0 else None,
          "eta_status": "process_finished" if status == 0 else "process_failed",
          "returncode": status, "overall_eta": "unknown",
          "scope": "subprocess only; artifact acceptance pending"}), flush=True)
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
    parser.add_argument("--seed", type=int, choices=(1337, 2027, 3407),
                        default=parameters.get("General/seed", parameters.get("seed", 1337)))
    args = parser.parse_args(argv)
    args.seed = int(args.seed)
    if args.seed not in (1337, 2027, 3407):
        raise ValueError("seed must be one of the three paper seeds")
    if not args.package_task_id or not args.package_manifest_sha256:
        raise ValueError("package ID and manifest SHA-256 are required")
    return args


def main():
    from clearml import Task
    task = Task.init(project_name="Thesis/EventTrack-V2X/Training",
                     task_name="D2 SPD official OOF fold detector training",
                     reuse_last_task_id=False, auto_connect_frameworks=False,
                     auto_connect_arg_parser=False)
    args = parse_package_arguments(task.get_parameters())
    package = Task.get_task(task_id=args.package_task_id)
    manifest = materialize(package, args.package_manifest_sha256)
    cohort, sequence_count = validate_cohort(manifest)
    env = dict(os.environ)
    env.update({"PATH": "/opt/cooptrack/bin:" + env.get("PATH", ""),
                "PYTHONPATH": "/workspace/CoopTrack", "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
                "LD_LIBRARY_PATH": "/opt/cooptrack/lib:/opt/cooptrack/lib/python3.8/site-packages/torch/lib:/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
                "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1", "MPLCONFIGDIR": "/tmp/eventtrack-matplotlib"})
    # Explicitly remove controller-venv activation from the legacy subprocess.
    env.pop("VIRTUAL_ENV", None)
    root = Path("/eventtrack-oof-gpu4-runs")
    root.mkdir()
    readiness = subprocess.check_output([
        LEGACY_PYTHON, "-c",
        "import torch,mmcv,mmdet,mmdet3d,json,os; import mmcv._ext; "
        "assert torch.cuda.device_count()==4; "
        "print(json.dumps({'torch':torch.__version__,'mmcv':mmcv.__version__,"
        "'mmdet3d':mmdet3d.__version__,'cuda_devices':"
        "[torch.cuda.get_device_name(i) for i in range(4)],"
        "'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','')}))"],
        env=env, text=True)
    runtime = json.loads(readiness.strip())
    print("EVENTTRACK_OOF_GPU4_LEGACY_READY " + json.dumps(runtime), flush=True)
    profiles = []
    # Never allocate more simultaneous streams than the admitted cohort.
    for batch in batch_candidates(sequence_count):
        profile = run_phase(task, "vehicle-side", batch, root / f"probe-vehicle-b{batch}", env, probe_iters=32, seed=args.seed)
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
                                  root / f"probe-infrastructure-b{selected}", env, probe_iters=32, seed=args.seed)
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
                 "maximum_memory_fraction": 0.8, "sequence_stream_limit": sequence_count,
                 "cohort": cohort, "seed": args.seed, "fold_id": manifest["fold_id"],
                 "actual_cuda_devices": runtime["cuda_devices"],
                 "cuda_visible_devices": runtime["cuda_visible_devices"],
                 "base_learning_rate_unchanged": True, "vehicle_profiles": profiles,
                 "infrastructure_profile": infra_profile,
                 "probe_weights_not_used_for_training": True}
    write_json(root / "batch-selection.json", selection)
    task.upload_artifact("batch-selection", artifact_object=selection, wait_on_upload=True)
    print("EVENTTRACK_OOF_GPU4_BATCH_SELECTED " + json.dumps(selection), flush=True)
    for side in ["vehicle-side", "infrastructure-side"]:
        output = root / side
        result = run_phase(task, side, selected, output, env, seed=args.seed)
        if not result["success"]:
            raise RuntimeError(side + " OOF fit detector training failed")
        checkpoints = sorted(output.glob("iter_*.pth"), key=lambda item: int(item.stem.split("_")[1]))
        if not checkpoints:
            raise RuntimeError("completed run has no trained checkpoint")
        final = checkpoints[-1]
        task.upload_artifact(side + "-final-checkpoint", artifact_object=final, wait_on_upload=True)
        receipt = {"side": side, "checkpoint": final.name, "sha256": sha256(final),
                   "epochs": 24, "train_sequences": sequence_count, "cohort": cohort,
                   "fold_id": manifest["fold_id"],
                   "actual_cuda_devices": runtime["cuda_devices"],
                   "cuda_visible_devices": runtime["cuda_visible_devices"],
                   "batch_per_gpu": selected,
                   "effective_batch_size": selected * 4, "seed": args.seed,
                   "stage": "D2-detector-training-only", "official_val_result_available": False}
        task.upload_artifact(side + "-completion", artifact_object=receipt, wait_on_upload=True)
    task.close()


if __name__ == "__main__":
    main()
