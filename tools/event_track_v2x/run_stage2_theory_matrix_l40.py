#!/usr/bin/env python3
"""Run the frozen Stage2 exact-bound verifier for three seeds on L40S CPU."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlparse

SOURCE_TASK = "a07847eb50c3447a8210bd3914a98bdd"
SOURCE_SHA = "4ee688b675df384926468baa6aadc4deb22559685acd0c96c4306eaf0efc901a"
PROJECT = "Thesis/Recover-Before-Fuse/Training"
SEEDS = (1337, 2027, 3407)
TRIALS = 256
HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def download(artifact):
    from clearml.storage.helper import StorageHelper
    url = artifact.url
    if urlparse(url).netloc not in HOSTS:
        raise ValueError("unapproved ClearML artifact host")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    data = b"".join(StorageHelper.get(url).download_as_stream(url))
    if len(data) != artifact.size or sha(data) != artifact.hash:
        raise ValueError("frozen source bytes differ")
    return data


def extract(data, root):
    if sha(data) != SOURCE_SHA:
        raise ValueError("source archive differs")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members or any(not m.isfile() or m.name.startswith("/") or
                              ".." in Path(m.name).parts for m in members):
            raise ValueError("unsafe source archive")
        for member in members:
            path = root / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    return len(members)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from clearml import Task
    task = Task.init(project_name=PROJECT,
                     task_name="RBF stage2 theory matrix L40S CPU",
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    task.reload()
    worker = task.data.last_worker or ""
    if not worker.startswith("10.100.35.121-L40S:"):
        raise ValueError("L40S worker required")
    expected = task.get_parameters()["General/bootstrap_source_sha256"]
    if sha(Path(__file__).read_bytes()) != expected:
        raise ValueError("bootstrap source differs")
    source = Task.get_task(task_id=SOURCE_TASK)
    if source.status != "completed" or source.artifacts["source"].hash != SOURCE_SHA:
        raise ValueError("frozen source task differs")
    with tempfile.TemporaryDirectory(prefix="rbf-stage2-theory-") as tmp:
        root = Path(tmp)
        members = extract(download(source.artifacts["source"]), root)
        project = root / "equivalence-source"
        verifier = project / "tools/event_track_v2x/verify_recoverable_hypotheses.py"
        if not verifier.is_file():
            raise ValueError("frozen verifier missing")
        verifier_sha = sha(verifier.read_bytes())
        reports = []
        started = time.monotonic()
        for seed in SEEDS:
            output = root / f"seed{seed}-theory.json"
            print(f"stage2 theory seed={seed} status=running ETA=unknown", flush=True)
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
            run = subprocess.run([sys.executable, str(verifier), "--output", str(output),
                                  "--seed", str(seed), "--trials", str(TRIALS)],
                                 cwd=project, env=env, capture_output=True, text=True,
                                 timeout=900)
            if run.returncode != 0 or not output.is_file():
                raise ValueError(f"frozen verifier failed for seed {seed}: {run.stderr[-1000:]}")
            report = json.loads(output.read_text())
            if (report.get("status") != "verified" or report.get("seed") != seed or
                    len(report.get("partitions", [])) != TRIALS or
                    report["algebra"]["random_cases"] != TRIALS * 2 or
                    report["scope"]["real_dataset_validation"] is not False):
                raise ValueError("theory verifier scope or trial count differs")
            for relative, actual in report["source_hashes"].items():
                if sha((project / relative).read_bytes()) != actual:
                    raise ValueError("theory verifier source inventory differs")
            reports.append({"seed": seed, "sha256": sha(output.read_bytes()),
                            "bytes": output.stat().st_size,
                            "partitions": len(report["partitions"]),
                            "algebra_trials": report["algebra"]["random_cases"],
                            "counterexamples": sorted(report["algebra"]["counterexamples"])})
            elapsed = time.monotonic() - started
            eta = elapsed / len(reports) * (len(SEEDS) - len(reports))
            print(f"stage2 theory {len(reports)}/3 seed={seed} ETA={eta:.1f}s", flush=True)
            task.get_logger().report_scalar("theory_matrix", "completed_seeds",
                                            len(reports), iteration=len(reports))
            task.upload_artifact(f"seed{seed}-theory", artifact_object=str(output),
                                 wait_on_upload=True)
        combined = {"kind": "stage2_theory_matrix_three_seed_l40_cpu_execution_v1",
                    "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "task_id": task.id, "worker_id": worker, "device": "cpu",
                    "source_task_id": SOURCE_TASK, "source_sha256": SOURCE_SHA,
                    "source_members": members, "bootstrap_source_sha256": expected,
                    "verifier_source_sha256": verifier_sha,
                    "trials_per_seed": TRIALS, "reports": reports,
                    "dataset_read": False, "ground_truth_read": False,
                    "parameter_training": False, "full_stage_two_complete": False,
                    "same_resource_baselines_complete": False,
                    "paper_performance_complete": False}
        receipt = root / "theory-matrix-receipt.json"
        receipt.write_text(json.dumps(combined, sort_keys=True, indent=2) + "\n")
        task.upload_artifact("theory-matrix-receipt", artifact_object=str(receipt),
                             wait_on_upload=True)
        task.close()


if __name__ == "__main__":
    main()
