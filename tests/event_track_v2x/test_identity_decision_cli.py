"""External-process reproducibility and publication boundaries of Bayes checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_independent_process_bayes_report_replay_and_create_once(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "tools/event_track_v2x/verify_identity_decision.py"), "--trials", "25"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    outputs = []
    for name in ("first", "second"):
        path = tmp_path / (name + ".json")
        result = subprocess.run(command + ["--output", str(path)], cwd=root, env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0])
    assert report["status"] == "verified" and len(report["verification"]["random_cases"]) == 25
    assert report["scope"]["formal_interval_certificate"] is False
    assert report["scope"]["decoder_integrated_with_branch_states"] is False
    assert report["scope"]["frozen_pipeline_modified"] is False
    absent = report["verification"]["absent_support_bayes_action"]
    assert absent["decision"]["action_in_retained_set"] is False
    extreme = report["verification"]["extreme_common_offset"]
    assert extreme["corrected_decision"]["choices"] == [-1]
    assert extreme["corrected_decision"]["model_truncation_regret_upper_estimate"] is None
    for relative, expected in report["source_hashes"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    repeat = subprocess.run(command + ["--output", str(tmp_path / "first.json")], cwd=root, env=env,
                            capture_output=True, text=True, timeout=60)
    assert repeat.returncode != 0 and "create-once" in repeat.stderr
    assert (tmp_path / "first.json").read_bytes() == outputs[0]
