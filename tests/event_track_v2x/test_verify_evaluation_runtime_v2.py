"""Verifier control-flow fixtures; real-engine conformance is a separate CLI run."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.event_track_v2x import verify_evaluation_runtime_v2 as verification
from tools.event_track_v2x.evaluate_source_ablation_v2 import RUNTIME_VERSIONS


@pytest.fixture
def alias_fixture(monkeypatch):
    reference = b"a = np.float(1); b = np.int(2); c = np.float64(3)\n"
    installed = b"a = float(1); b = int(2); c = np.float64(3)\n"
    digest = lambda value: hashlib.sha256(value).hexdigest()
    monkeypatch.setattr(verification, "REFERENCE", {
        "fixture.py": (digest(reference), digest(installed), 2)})
    return reference, installed


def test_only_declared_alias_transform_is_accepted(alias_fixture):
    evidence = verification.check_reference_bytes("fixture.py", *alias_fixture)
    assert evidence["scalar_alias_replacements"] == 2
    assert evidence["declared_compatibility_transform_identical"] is True
    assert evidence["byte_identical"] is False


@pytest.mark.parametrize("side", [0, 1])
def test_changed_reference_or_installed_code_fails(alias_fixture, side):
    contents = list(alias_fixture)
    contents[side] += b"# different source\n"
    with pytest.raises(ValueError, match="beyond declared"):
        verification.check_reference_bytes("fixture.py", *contents)


def test_even_pinned_non_alias_edit_is_rejected(monkeypatch, alias_fixture):
    reference, installed = alias_fixture
    installed = installed.replace(b"float(1)", b"float(9)")
    monkeypatch.setattr(verification, "REFERENCE", {"fixture.py": (
        hashlib.sha256(reference).hexdigest(), hashlib.sha256(installed).hexdigest(), 2)})
    with pytest.raises(ValueError, match="beyond declared"):
        verification.check_reference_bytes("fixture.py", reference, installed)


def test_alias_count_mismatch_fails(monkeypatch, alias_fixture):
    reference, installed = alias_fixture
    original = verification.REFERENCE["fixture.py"]
    monkeypatch.setitem(verification.REFERENCE, "fixture.py", (*original[:2], 1))
    with pytest.raises(ValueError, match="beyond declared"):
        verification.check_reference_bytes("fixture.py", reference, installed)


@pytest.mark.parametrize("actual", [float("nan"), float("inf"), -float("inf"), .51, True, "0.5", None])
def test_numeric_check_fails_closed(actual):
    with pytest.raises(ValueError, match="expected"):
        verification.close("fixture", actual, .5)


def test_numeric_check_accepts_tolerance():
    verification.close("fixture", .5 + 1e-12, .5)


@pytest.fixture
def fake_runtime(monkeypatch):
    # No metric-engine assertion here: exercise receipts/failure paths only.
    adapter = SimpleNamespace(runtime_evidence=Mock(return_value={"fixture": "runtime"}),
                              golden_cases=Mock(return_value={"passed": True}))
    monkeypatch.setattr(verification, "sys", SimpleNamespace(version_info=(3, 10), executable=sys.executable))
    monkeypatch.setattr(verification, "load_adapter", lambda: adapter)
    monkeypatch.setattr(verification, "validate_runtime", Mock())
    monkeypatch.setattr(verification, "reference_evidence", Mock(return_value={"fixture": True}))
    monkeypatch.setattr(verification, "extended_cases", Mock(return_value={
        "passed": True, "cases": {str(i): {"passed": True} for i in range(4)}}))
    return adapter


def test_fixture_success_receipt_binds_outputs_and_excludes_dataset_claims(tmp_path, fake_runtime):
    output = tmp_path / "success"
    receipt = verification.run(output)
    assert receipt["status"] == "complete"
    assert receipt["original_cases"] == 3
    assert receipt["additional_cases"] == 4
    for field in ("real_dataset_read", "test_payloads_read", "real_tracking_method_result", "paper_eligible"):
        assert receipt[field] is False
    for name, digest in receipt["files"].items():
        assert verification.sha(output / name) == digest
    assert not (output / "failure.json").exists()
    assert fake_runtime.runtime_evidence.call_count == 2


@pytest.mark.parametrize("failure", ["load", "runtime", "reference", "golden", "extended", "changed_runtime", "changed_source"])
def test_failed_checks_never_produce_complete_receipt(tmp_path, monkeypatch, fake_runtime, failure):
    def fail(*args, **kwargs):
        raise ValueError("intentional fixture failure")
    if failure == "load":
        monkeypatch.setattr(verification, "load_adapter", fail)
    elif failure == "runtime":
        monkeypatch.setattr(verification, "validate_runtime", fail)
    elif failure == "reference":
        monkeypatch.setattr(verification, "reference_evidence", fail)
    elif failure == "golden":
        fake_runtime.golden_cases.return_value = {"passed": False}
    elif failure == "extended":
        monkeypatch.setattr(verification, "extended_cases", fail)
    elif failure == "changed_runtime":
        fake_runtime.runtime_evidence.side_effect = [{"fixture": 1}, {"fixture": 2}]
    else:
        monkeypatch.setattr(verification, "sha", Mock(side_effect=["a" * 64, "b" * 64, "c" * 64]))
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        verification.run(output)
    assert not (output / "receipt.json").exists()
    assert json.loads((output / "failure.json").read_text())["partial_outputs_not_conformance_results"] is True


@pytest.mark.parametrize("version", [(3, 8), (3, 12), (4, 0)])
def test_unsupported_python_refused_before_output(tmp_path, monkeypatch, version):
    monkeypatch.setattr(verification, "sys", SimpleNamespace(version_info=version))
    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="separate Python"):
        verification.run(output)
    assert not output.exists()


def test_existing_output_preserved(tmp_path, fake_runtime):
    marker = tmp_path / "keep"
    marker.write_text("unchanged")
    with pytest.raises(ValueError, match="new output directory"):
        verification.run(tmp_path)
    assert marker.read_text() == "unchanged"
    fake_runtime.runtime_evidence.assert_not_called()


def test_symlink_parent_refused(tmp_path, fake_runtime):
    parent = tmp_path / "actual"
    parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verification.run(link / "new")
    assert not (parent / "new").exists()


def test_optimized_python_refused_without_evaluation_imports(tmp_path):
    output = tmp_path / "optimized"
    result = subprocess.run([sys.executable, "-O", str(Path(verification.__file__)), "--output", str(output)],
        cwd=verification.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode != 0
    assert "optimization disables sealed golden assertions" in result.stderr
    assert not output.exists()


def test_synthetic_cases_are_car_only_with_independent_duplicate_states():
    gt, predictions = verification.sequence("fixture", 4, duplicates=True)
    before = copy.deepcopy(gt)
    assert [r["box_reference_timestamp_us"] for r in gt] == [1_000_000 + i * 150_100 for i in range(4)]
    assert {obj["class_label"] for r in predictions for obj in r["predictions"]} == {"car"}
    predictions[0]["predictions"][0]["mean"][0] = 100.
    assert predictions[0]["predictions"][1]["mean"][0] == 0.
    assert gt == before


def test_wheel_lock_matches_install_records_and_sealed_versions():
    root = verification.ROOT / "environments/event_track_v2x"
    audit = json.loads((root / "evaluation-wheel-install-audit-20260913.json").read_text())
    lock = (root / "requirements-evaluation-macos-arm64-py310.lock").read_text()
    lines = [line for line in lock.splitlines() if line and not line.startswith("#")]
    assert len(lines) == len(audit["wheels"]) == 50
    assert len({row["name"].lower() for row in audit["wheels"]}) == 50
    for row in audit["wheels"]:
        assert row["url"].startswith("https://files.pythonhosted.org/")
        assert row["url"].endswith(".whl")
        assert len(row["sha256"]) == 64
        assert f'{row["name"]} @ {row["url"]} --hash=sha256:{row["sha256"]}' in lines
    versions = {row["name"].lower(): row["version"] for row in audit["wheels"]}
    for name, version in RUNTIME_VERSIONS.items():
        assert versions[name] == version
    scipy = next(row for row in audit["wheels"] if row["name"] == "scipy")
    assert "macosx_12_0_arm64" in scipy["url"]
    assert scipy["sha256"] != audit["replacement"]["discarded_sha256"]
