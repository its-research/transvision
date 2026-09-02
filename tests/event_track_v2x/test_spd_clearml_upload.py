from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.event_track_v2x.archive_manifest import (
    ArchiveManifestError,
    EXPECTED_ARCHIVE_NAMES,
    build_archive_manifest,
    write_archive_manifest,
)
from tools.event_track_v2x.upload_clearml_dataset import (
    DEFAULT_NAME,
    DEFAULT_PROJECT,
    EXECUTE_TOKEN,
    ClearMLDatasetError,
    main,
)
from tools.event_track_v2x.verify_clearml_dataset import main as verify_main


ARCHIVE_NAMES = EXPECTED_ARCHIVE_NAMES


@pytest.fixture(autouse=True)
def _reset_fake_dataset() -> None:
    _FakeDataset.instance = None
    _FakeDataset.fail_upload = False
    _FakeDataset.mutate_on_create = None
    _FakeDataset.corrupt_readback = False


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    archives = tmp_path / "archives"
    archives.mkdir()
    for index, name in enumerate(ARCHIVE_NAMES):
        (archives / name).write_bytes(bytes([index]))
    manifest = write_archive_manifest(
        tmp_path / "archive-manifest.json",
        build_archive_manifest(archives),
    )
    return archives, manifest


class _FakeDataset:
    instance: "_FakeDataset | None" = None
    fail_upload = False
    mutate_on_create: Path | None = None
    corrupt_readback = False

    def __init__(self, *, project: str, name: str, version: str) -> None:
        self.id = "dataset-id"
        self.project = project
        self.name = name
        self.version = version
        self.finalized = False
        self.calls: list[str] = []
        self.staged_names: tuple[str, ...] = ()
        self.description = ""
        self.tags: tuple[str, ...] = ()
        self.staged_bytes: dict[str, bytes] = {}

    @classmethod
    def create(cls, **kwargs: Any) -> "_FakeDataset":
        if cls.mutate_on_create is not None:
            cls.mutate_on_create.write_bytes(b"changed-after-initial-verification")
        instance = cls(
            project=kwargs["dataset_project"],
            name=kwargs["dataset_name"],
            version=kwargs["dataset_version"],
        )
        instance.description = kwargs["description"]
        instance.tags = tuple(kwargs["dataset_tags"])
        instance.calls.append("create")
        cls.instance = instance
        return instance

    @classmethod
    def get(cls, **_kwargs: Any) -> "_FakeDataset":
        assert cls.instance is not None
        cls.instance.calls.append("get")
        return cls.instance

    def add_files(self, path: Path, **_kwargs: Any) -> None:
        self.calls.append("add_files")
        self.staged_names = tuple(sorted(item.name for item in Path(path).iterdir()))
        self.staged_bytes = {
            item.name: item.read_bytes() for item in Path(path).iterdir()
        }

    def upload(self, **_kwargs: Any) -> None:
        self.calls.append("upload")
        if self.fail_upload:
            raise RuntimeError("upload failed")

    def finalize(self, **_kwargs: Any) -> bool:
        self.calls.append("finalize")
        self.finalized = True
        return True

    def is_final(self) -> bool:
        return self.finalized

    def list_files(self) -> list[str]:
        self.calls.append("list_files")
        return list(self.staged_names)

    def get_mutable_local_copy(self, *, target_folder: str, **_kwargs: Any) -> str:
        self.calls.append("get_mutable_local_copy")
        target = Path(target_folder)
        target.mkdir(parents=True, exist_ok=True)
        for name, contents in self.staged_bytes.items():
            (target / name).write_bytes(contents)
        if self.corrupt_readback:
            (target / ARCHIVE_NAMES[0]).write_bytes(b"corrupt-readback")
        return str(target)


def _arguments(
    archives: Path, manifest: Path, *, readback: Path | None = None
) -> list[str]:
    result = [
        "--archive-dir",
        str(archives),
        "--manifest",
        str(manifest),
        "--version",
        "smoke-v1",
    ]
    if readback is not None:
        result.extend(("--readback-dir", str(readback)))
    return result


def test_default_is_dry_run_and_does_not_create_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archives, manifest = _inputs(tmp_path)
    _FakeDataset.instance = None

    assert main(_arguments(archives, manifest), dataset_cls=_FakeDataset) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "dry-run"
    assert output["project"] == DEFAULT_PROJECT
    assert output["name"] == DEFAULT_NAME
    assert output["archive_count"] == 8
    assert len(output["dataset_files"]) == 9
    assert _FakeDataset.instance is None


def test_wrong_execute_token_fails_closed(tmp_path: Path) -> None:
    archives, manifest = _inputs(tmp_path)

    with pytest.raises(ClearMLDatasetError, match="incorrect execute token"):
        main(
            _arguments(archives, manifest) + ["--execute-token", "wrong"],
            dataset_cls=_FakeDataset,
        )


def test_execute_uploads_only_archives_and_manifest_then_reads_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archives, manifest = _inputs(tmp_path)
    _FakeDataset.instance = None
    _FakeDataset.fail_upload = False
    _FakeDataset.mutate_on_create = None
    _FakeDataset.corrupt_readback = False

    assert (
        main(
            _arguments(archives, manifest, readback=tmp_path / "readback")
            + ["--execute-token", EXECUTE_TOKEN],
            dataset_cls=_FakeDataset,
        )
        == 0
    )

    instance = _FakeDataset.instance
    assert instance is not None
    assert instance.calls == [
        "create",
        "add_files",
        "upload",
        "get",
        "list_files",
        "finalize",
        "get",
        "list_files",
        "get_mutable_local_copy",
    ]
    assert instance.staged_names == tuple(sorted((*ARCHIVE_NAMES, "archive-manifest.json")))
    assert instance.finalized
    assert "restricted" in instance.tags
    assert "scientific-claim-forbidden" in instance.tags
    assert "/private/" not in instance.description
    output = json.loads(capsys.readouterr().out)
    assert output["dataset"]["finalized"] is True
    assert output["byte_readback_verified"] is True


def test_upload_failure_never_finalizes(tmp_path: Path) -> None:
    archives, manifest = _inputs(tmp_path)
    _FakeDataset.instance = None
    _FakeDataset.fail_upload = True
    _FakeDataset.mutate_on_create = None
    _FakeDataset.corrupt_readback = False

    with pytest.raises(RuntimeError, match="upload failed"):
        main(
            _arguments(archives, manifest, readback=tmp_path / "readback")
            + ["--execute-token", EXECUTE_TOKEN],
            dataset_cls=_FakeDataset,
        )

    instance = _FakeDataset.instance
    assert instance is not None
    assert "finalize" not in instance.calls
    assert not instance.finalized


def test_source_mutation_after_initial_hash_is_detected_before_upload(
    tmp_path: Path,
) -> None:
    archives, manifest = _inputs(tmp_path)
    _FakeDataset.instance = None
    _FakeDataset.fail_upload = False
    _FakeDataset.mutate_on_create = archives / ARCHIVE_NAMES[0]
    _FakeDataset.corrupt_readback = False
    with pytest.raises(ArchiveManifestError, match="mismatch"):
        main(
            _arguments(archives, manifest, readback=tmp_path / "readback")
            + ["--execute-token", EXECUTE_TOKEN],
            dataset_cls=_FakeDataset,
        )
    assert _FakeDataset.instance is not None
    assert "add_files" not in _FakeDataset.instance.calls


def test_corrupt_remote_readback_is_not_reported_as_success(tmp_path: Path) -> None:
    archives, manifest = _inputs(tmp_path)
    _FakeDataset.instance = None
    _FakeDataset.fail_upload = False
    _FakeDataset.mutate_on_create = None
    _FakeDataset.corrupt_readback = True
    with pytest.raises(ArchiveManifestError, match="mismatch"):
        main(
            _arguments(archives, manifest, readback=tmp_path / "readback")
            + ["--execute-token", EXECUTE_TOKEN],
            dataset_cls=_FakeDataset,
        )


def test_deferred_upload_requires_separate_byte_verifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archives, manifest = _inputs(tmp_path)
    assert (
        main(
            _arguments(archives, manifest)
            + [
                "--execute-token",
                EXECUTE_TOKEN,
                "--defer-byte-readback",
            ],
            dataset_cls=_FakeDataset,
        )
        == 0
    )
    upload_result = json.loads(capsys.readouterr().out)
    assert upload_result["mode"] == "executed-pending-byte-readback"
    assert upload_result["byte_readback_verified"] is False
    assert (
        verify_main(
            [
                "--dataset-id",
                "dataset-id",
                "--manifest",
                str(manifest),
                "--version",
                "smoke-v1",
                "--readback-dir",
                str(tmp_path / "independent-readback"),
            ],
            dataset_cls=_FakeDataset,
        )
        == 0
    )
    verify_result = json.loads(capsys.readouterr().out)
    assert verify_result["byte_readback_verified"] is True
