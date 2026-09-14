import json
from pathlib import Path
import sys

import pytest

from tools.event_track_v2x import launch_cooptrack_detector_training as launcher


def arguments(monkeypatch, root):
    argv = ["launch_cooptrack_detector_training.py"]
    for option in ("runtime-root", "upstream-root", "inputs", "converted", "pretrained", "output"):
        argv.extend(["--" + option, str(root / option)])
    argv.extend(["--image", "pinned-image"])
    monkeypatch.setattr(sys, "argv", argv)


def test_duplicate_output_is_rejected_before_external_calls(tmp_path, monkeypatch):
    arguments(monkeypatch, tmp_path)
    (tmp_path / "output").mkdir()

    def unexpected(*args, **kwargs):
        pytest.fail("duplicate run must not invoke Docker, Git or ClearML")

    monkeypatch.setattr(launcher.subprocess, "check_output", unexpected)
    with pytest.raises(FileExistsError, match="create-once"):
        launcher.main()


@pytest.mark.parametrize("commit,dirty,error", [
    ("wrong-commit", "", "source mismatch"),
    ("29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399", " M detector.py", "dirty"),
])
def test_unpinned_or_dirty_upstream_is_rejected_before_task_creation(
        tmp_path, monkeypatch, commit, dirty, error):
    arguments(monkeypatch, tmp_path)
    converted = tmp_path / "converted"
    converted.mkdir()
    (converted / "conversion-manifest.json").write_text(json.dumps({"fold_id": 0}))

    def check_output(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return "sha256:pinned-image\n"
        if command[-2:] == ["rev-parse", "HEAD"]:
            return commit + "\n"
        if command[-2:] == ["status", "--porcelain"]:
            return dirty
        pytest.fail("unexpected external command: " + repr(command))

    monkeypatch.setattr(launcher.subprocess, "check_output", check_output)
    with pytest.raises(ValueError, match=error):
        launcher.main()
    assert not Path(tmp_path / "output").exists()
