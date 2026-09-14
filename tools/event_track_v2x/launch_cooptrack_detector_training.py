#!/usr/bin/env python3
"""Track sequential vehicle/infrastructure fold training in private ClearML.

The controller uses Python 3.10+ on the GPU host; models use the pinned legacy
container. It never reads official val/test, changes dataset tags, or writes the
paper result registry. The caller must first finish the input conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import traceback


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--side", choices=("vehicle-side", "infrastructure-side", "both"), default="both")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("controller output is create-once; inspect the existing run")
    conversion = json.loads((args.converted / "conversion-manifest.json").read_bytes())
    cohort = conversion.get("cohort", "fit-fold")
    cohort_name = "full-train" if cohort == "full-train" else f"fold-{conversion['fold_id']}"
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    source_commit = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True).strip()
    if source_commit != "29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399":
        raise ValueError("upstream source mismatch")
    if subprocess.check_output(
            ["git", "-C", str(args.upstream_root), "status", "--porcelain"], text=True):
        raise ValueError("upstream checkout is dirty")
    args.output.mkdir(parents=True)
    (args.output / "runs").mkdir()
    entrypoints = args.output / "entrypoints"
    entrypoints.mkdir()
    for name in ("run_cooptrack_detector.py", "launch_cooptrack_detector_training.py", "Dockerfile"):
        shutil.copy2(args.runtime_root / name, entrypoints / name)
    from clearml import Task
    sides = ("vehicle-side", "infrastructure-side") if args.side == "both" else (args.side,)
    for side in sides:
        task = Task.init(project_name="Thesis/EventTrack-V2X/Training",
                         task_name=f"D2 CoopTrack-R50 {cohort_name} {side} seed-1337",
                         task_type=Task.TaskTypes.training, reuse_last_task_id=False,
                         auto_connect_frameworks=False, auto_connect_arg_parser=False,
                         output_uri=True)
        task.set_tags(["formal-detector-training", "spd-train-only", cohort_name, "seed-1337",
                       "user-confirmed-official-source", "paper-evidence-pending"])
        parameters = {
            "stage": "D2-detector-fit", "side": side, "seed": 1337, "epochs": 24,
            "fold_id": conversion["fold_id"], "fit_sequences": conversion["fit_sequence_ids"],
            "cohort": cohort,
            "frame_count": conversion["counts"][side]["frames"],
            "source_commit": source_commit, "image_id": image_id,
            "conversion_manifest_sha256": sha256(args.converted / "conversion-manifest.json"),
            "input_manifest_sha256": sha256(args.inputs / "input-manifest.json"),
            "entrypoint_sha256": sha256(entrypoints / "run_cooptrack_detector.py"),
            "controller_sha256": sha256(entrypoints / "launch_cooptrack_detector_training.py"),
            "dockerfile_sha256": sha256(entrypoints / "Dockerfile"),
            "pretrained_sha256": sha256(args.pretrained),
            "effective_batch_size": 8, "micro_batch_size": 1, "gradient_accumulation": 8,
            "runtime_library_path": "/opt/cooptrack/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
            "source_identity": "user-confirmed-official-download-2026-09-11",
            "official_val_read": False, "test_or_test_A_read": False,
            "paper_registry_writable": False,
        }
        task.set_parameters(parameters)
        write_json(args.output / f"{side}-task.json", {"task_id": task.id, **parameters})
        task.upload_artifact("training-input-manifest", artifact_object=args.inputs / "input-manifest.json")
        task.upload_artifact("conversion-manifest", artifact_object=args.converted / "conversion-manifest.json")
        container_name = f"eventtrack-det-{cohort_name}-{side}-1337-{str(task.id)[:8]}"
        command = [
            "docker", "run", "--rm", "--name", container_name,
            "--gpus", "device=0", "--network", "none", "--ipc", "private", "--shm-size", "4g",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OMP_NUM_THREADS=4",
            "--env", "LD_LIBRARY_PATH=/opt/cooptrack/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
            "--env", "MPLCONFIGDIR=/tmp/matplotlib", "--env", "GLOO_SOCKET_IFNAME=lo",
            "--env", "NCCL_SOCKET_IFNAME=lo", "--env", "NCCL_IB_DISABLE=1",
            "--mount", f"type=bind,src={args.upstream_root},dst=/workspace/CoopTrack,readonly",
            "--mount", f"type=bind,src={entrypoints},dst=/entrypoints,readonly",
            "--mount", f"type=bind,src={args.inputs},dst=/inputs,readonly",
            "--mount", f"type=bind,src={args.converted},dst=/converted,readonly",
            "--mount", f"type=bind,src={args.pretrained},dst=/pretrained/resnet50.pth,readonly",
            "--mount", f"type=bind,src={args.output / 'runs'},dst=/runs",
            "--entrypoint", "python", image_id,
            "-m", "torch.distributed.launch", "--nproc_per_node=1", "--master_port=29571",
            "/entrypoints/run_cooptrack_detector.py", "--upstream-root", "/workspace/CoopTrack",
            "--inputs", "/inputs", "--converted", "/converted", "--pretrained", "/pretrained/resnet50.pth",
            "--output", f"/runs/{side}", "--side", side,
        ]
        print("EVENTTRACK_TASK_STARTED " + json.dumps({"task_id": task.id, "side": side}), flush=True)
        proof_uploaded = False
        try:
            with (args.output / f"{side}.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                write_json(args.output / f"{side}-process.json",
                           {"controller_pid": os.getpid(), "docker_pid": process.pid,
                            "container": container_name, "task_id": task.id})
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    match = re.search(r"Iter \[(\d+)/\d+\].*?\bloss: ([0-9.eE+-]+)", line)
                    if match:
                        task.get_logger().report_scalar("train", "loss", value=float(match[2]),
                                                        iteration=int(match[1]))
                    if "EVENTTRACK_OPTIMIZER_STARTED " in line and not proof_uploaded:
                        run = args.output / "runs" / side
                        task.upload_artifact("optimizer-startup", artifact_object=run / "optimizer-startup.json")
                        task.upload_artifact("startup-checkpoint", artifact_object=run / "startup-iter-16.pth")
                        task.upload_artifact("launch-receipt", artifact_object=run / "launch-receipt.json")
                        task.upload_artifact("resolved-config", artifact_object=run / "detector.py")
                        proof_uploaded = True
                result = process.wait()
            if result:
                raise RuntimeError(f"detector container exited with code {result}")
            if not proof_uploaded:
                raise RuntimeError("training ended without optimizer-update evidence")
            run = args.output / "runs" / side
            checkpoints = sorted(run.glob("iter_*.pth"), key=lambda path: int(path.stem.split("_")[1]))
            final = checkpoints[-1]
            task.upload_artifact("final-checkpoint", artifact_object=final, wait_on_upload=True)
            write_json(args.output / f"{side}-completed.json",
                       {"task_id": task.id, "checkpoint": str(final), "sha256": sha256(final)})
            task.close()
        except BaseException as exc:
            write_json(args.output / f"{side}-failed.json", {"task_id": task.id, "error": str(exc)})
            task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
            task.close()
            traceback.print_exc()
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
