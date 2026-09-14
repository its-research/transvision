import io
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

from tools.event_track_v2x import run_cooptrack_a100
from tools.event_track_v2x.run_cooptrack_a100 import (
    choose_batch, parse_package_arguments, training_command, validate_archive,
)


@pytest.mark.parametrize("prefix", ["", "General/"])
def test_package_arguments_accept_clearml_and_legacy_sections(prefix):
    args = parse_package_arguments({prefix + "package_task_id": "package-task",
                                    prefix + "package_manifest_sha256": "a" * 64}, [])
    assert args.package_task_id == "package-task"
    assert args.package_manifest_sha256 == "a" * 64


def test_package_arguments_prefer_clearml_general_section():
    args = parse_package_arguments({"package_task_id": "legacy-task",
                                    "package_manifest_sha256": "a" * 64,
                                    "General/package_task_id": "clearml-task",
                                    "General/package_manifest_sha256": "b" * 64}, [])
    assert args.package_task_id == "clearml-task"
    assert args.package_manifest_sha256 == "b" * 64


def test_explicit_package_arguments_override_clearml_values():
    args = parse_package_arguments({"General/package_task_id": "clearml-task",
                                    "General/package_manifest_sha256": "a" * 64},
                                   ["--package-task-id", "cli-task",
                                    "--package-manifest-sha256", "b" * 64])
    assert args.package_task_id == "cli-task"
    assert args.package_manifest_sha256 == "b" * 64


@pytest.mark.parametrize("parameters", [{}, {"General/package_task_id": "task"},
                                        {"General/package_manifest_sha256": "a" * 64},
                                        {"General/package_task_id": "",
                                         "General/package_manifest_sha256": "a" * 64}])
def test_package_arguments_reject_missing_values(parameters):
    with pytest.raises(ValueError, match="package ID and manifest"):
        parse_package_arguments(parameters, [])


def test_main_uses_sectioned_parameters_before_materialization(monkeypatch):
    parameters = {"General/package_task_id": "package-task",
                  "General/package_manifest_sha256": "a" * 64}
    package = object()
    task_api = SimpleNamespace(
        init=lambda **kwargs: SimpleNamespace(get_parameters=lambda: parameters),
        get_task=lambda task_id: package if task_id == "package-task" else None,
    )
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=task_api))
    monkeypatch.setattr(sys, "argv", ["run_cooptrack_a100.py"])

    def stop_at_materialization(actual_package, manifest_hash):
        assert actual_package is package
        assert manifest_hash == "a" * 64
        raise RuntimeError("verified materialization boundary")

    monkeypatch.setattr(run_cooptrack_a100, "materialize", stop_at_materialization)
    with pytest.raises(RuntimeError, match="verified materialization boundary"):
        run_cooptrack_a100.main()


def profile(batch, used=70, success=True, rank_count=4):
    return {"batch_per_gpu": batch, "success": success,
            "ranks": [{"peak_reserved_bytes": used, "total_memory_bytes": 100}] * rank_count}


def test_batch_selection_requires_all_ranks_and_twenty_percent_headroom():
    assert choose_batch([profile(2), profile(4), profile(8, used=85), profile(10, rank_count=3)]) == 4
    with pytest.raises(RuntimeError):
        choose_batch([profile(8, success=False)])


def test_training_command_is_a100_only_full_train_and_no_probe_by_default():
    cmd = training_command("vehicle-side", 8, Path("/runs/train"), 0)
    assert "--nproc_per_node=4" in cmd
    assert cmd[cmd.index("--require-device") + 1] == "A100"
    assert cmd[cmd.index("--batch-size") + 1] == "8"
    assert cmd[cmd.index("--accumulation") + 1] == "1"
    assert "--batch-probe-iters" not in cmd
    assert all("test_A" not in value for value in cmd)


def archive(tmp_path, name, link=None):
    path = tmp_path / "example.tar.gz"
    with tarfile.open(path, "w:gz") as output:
        item = tarfile.TarInfo(name)
        if link is not None:
            item.type = tarfile.SYMTYPE
            item.linkname = link
            output.addfile(item)
        else:
            item.size = 3
            output.addfile(item, io.BytesIO(b"abc"))
    return path


def test_archive_whitelists_destinations(tmp_path):
    path = archive(tmp_path, "inputs/vehicle-side/image/000001.jpg")
    assert validate_archive(path, ["inputs", "converted"]) == 1


def test_archive_rejects_traversal(tmp_path):
    path = archive(tmp_path, "inputs/../../outside")
    with pytest.raises(ValueError, match="unsafe"):
        validate_archive(path, ["inputs"])


def test_runtime_symlink_must_stay_inside_runtime(tmp_path):
    path = archive(tmp_path, "opt/cooptrack/bin/python", "python3.8")
    assert validate_archive(path, ["opt/cooptrack"], allow_links=True) == 1
    path.unlink()
    path = archive(tmp_path, "opt/cooptrack/bin/python", "../../../outside")
    with pytest.raises(ValueError, match="escapes"):
        validate_archive(path, ["opt/cooptrack"], allow_links=True)


def test_training_data_cannot_have_symlinks(tmp_path):
    path = archive(tmp_path, "inputs/link", "target")
    with pytest.raises(ValueError, match="unexpected link"):
        validate_archive(path, ["inputs"])
