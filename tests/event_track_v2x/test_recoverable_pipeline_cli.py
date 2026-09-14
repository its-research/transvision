"""External fresh-process replay of the synthetic learned/state pipeline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_two_fresh_processes_match_and_report_is_create_once(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "tools/event_track_v2x/verify_recoverable_pipeline.py")]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    outputs = []
    for name in ("first", "replay"):
        path = tmp_path / (name + ".json")
        result = subprocess.run(command + ["--output", str(path)], cwd=root, env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0])
    assert report["status"] == "verified" and report["fresh_instance_replay_byte_equal"]
    assert report["scope"]["trained_model"] is False
    assert report["scope"]["real_data_read"] is False
    assert report["scope"]["complete_lifecycle_tracker"] is False
    assert len(report["trace"]["allocations"]) == 3
    assert report["trace"]["checks"]["distinct_conditional_branch_state_histories"] >= 2
    counterexample = report["trace"]["checks"]["component_independence_assumption_counterexample"]
    assert counterexample["independent_model_partition"] == 4.
    assert counterexample["true_coupled_partition"] == 103.
    assert counterexample["invalid_product_omitted_mass_bound"] < counterexample["true_coupled_omitted_mass"]
    for relative, expected in report["source_hashes"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    duplicate = subprocess.run(command + ["--output", str(tmp_path / "first.json")], cwd=root, env=env,
                               capture_output=True, text=True, timeout=60)
    assert duplicate.returncode != 0 and "create-once" in duplicate.stderr
    assert (tmp_path / "first.json").read_bytes() == outputs[0]
