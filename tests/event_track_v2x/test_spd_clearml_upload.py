from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
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
    DEFAULT_FORMAL_NAME,
    DEFAULT_NAME,
    DEFAULT_PROJECT,
    EXECUTE_TOKEN,
    ClearMLDatasetError,
    main,
)
from tools.event_track_v2x.verify_clearml_dataset import (
    PUBLISH_TOKEN,
    main as verify_main,
)


ARCHIVE_NAMES = EXPECTED_ARCHIVE_NAMES


@pytest.fixture(autouse=True)
def _reset_fake_dataset() -> None:
    _FakeDataset.instance = None
    _FakeDataset.fail_upload = False
    _FakeDataset.upload_returns_false = False
    _FakeDataset.mutate_on_create = None
    _FakeDataset.corrupt_readback = False
    _FakeDataset.corrupt_manifest_readback = False
    _FakeDataset.symlink_readback = False
    _FakeDataset.internal_hash_mismatches = ()
    _FakeDataset.get_requests = []


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
    upload_returns_false = False
    mutate_on_create: Path | None = None
    corrupt_readback = False
    corrupt_manifest_readback = False
    symlink_readback = False
    internal_hash_mismatches: tuple[str, ...] = ()
    get_requests: list[dict[str, Any]] = []

    def __init__(self, *, project: str, name: str, version: str) -> None:
        self.id = "dataset-id"
        self.project = project
        self.name = name
        self.version = version
        self.finalized = False
        self.status = "created"
        self._task = self
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
    def get(cls, **kwargs: Any) -> "_FakeDataset":
        assert cls.instance is not None
        cls.get_requests.append(dict(kwargs))
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
        # Match the ClearML 2.1.3 trailing-future behavior: the executor waits
        # for completion, but this method intentionally never calls result().
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(
                self.upload_artifact,
                name="data",
                wait_on_upload=True,
            )

    def upload_artifact(self, **_kwargs: Any) -> bool:
        self.calls.append("upload_artifact")
        if self.fail_upload:
            raise RuntimeError("upload failed")
        return not self.upload_returns_false

    def flush(self, *, wait_for_uploads: bool) -> bool:
        self.calls.append(f"flush:{wait_for_uploads}")
        return True

    def finalize(self, **_kwargs: Any) -> bool:
        self.calls.append("finalize")
        self.finalized = True
        self.status = "completed"
        return True

    def is_final(self) -> bool:
        return self.finalized

    def get_status(self) -> str:
        return self.status

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

    def get_local_copy(self, **_kwargs: Any) -> str:
        self.calls.append("get_local_copy")
        cache_root = Path(os.environ["CLEARML_CACHE_DIR"])
        target = cache_root / "storage_manager" / "datasets" / "ds_dataset-id"
        target.mkdir(parents=True, exist_ok=True)
        for name, contents in self.staged_bytes.items():
            path = target / name
            if self.symlink_readback and name == ARCHIVE_NAMES[0]:
                source = cache_root / "symlink-source"
                source.write_bytes(contents)
                path.symlink_to(source)
            else:
                path.write_bytes(contents)
        if self.corrupt_readback:
            (target / ARCHIVE_NAMES[0]).write_bytes(b"corrupt-readback")
        if self.corrupt_manifest_readback:
            manifest_path = target / "archive-manifest.json"
            manifest_path.write_bytes(manifest_path.read_bytes().rstrip(b"\n"))
        return str(target)

    def verify_dataset_hash(self, **kwargs: Any) -> list[str]:
        self.calls.append(
            "verify_dataset_hash:"
            f"{kwargs.get('skip_hash')}:{kwargs.get('verbose')}"
        )
        return list(self.internal_hash_mismatches)

    def publish(self, **_kwargs: Any) -> bool:
        self.calls.append("publish")
        self.status = "published"
        return True


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
        "upload_artifact",
        "flush:True",
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
    assert output["dataset"]["publication_state"] == "smoke"
    assert output["upload_artifact_count"] == 1
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


def test_false_trailing_upload_future_never_finalizes(tmp_path: Path) -> None:
    archives, manifest = _inputs(tmp_path)
    _FakeDataset.upload_returns_false = True

    with pytest.raises(ClearMLDatasetError, match="future returned failure"):
        main(
            _arguments(archives, manifest, readback=tmp_path / "readback")
            + ["--execute-token", EXECUTE_TOKEN],
            dataset_cls=_FakeDataset,
        )

    assert _FakeDataset.instance is not None
    assert "upload_artifact" in _FakeDataset.instance.calls
    assert "finalize" not in _FakeDataset.instance.calls


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
                "--cold-cache-dir",
                str(tmp_path / "cold-cache"),
            ],
            dataset_cls=_FakeDataset,
        )
        == 0
    )
    verify_result = json.loads(capsys.readouterr().out)
    assert verify_result["byte_readback_verified"] is True
    assert verify_result["cold_cache_verified"] is True
    assert verify_result["dataset"]["publication_state"] == "smoke"
    assert _FakeDataset.get_requests[-1] == {
        "dataset_id": "dataset-id",
        "only_completed": True,
    }
    assert "verify_dataset_hash:False:True" in _FakeDataset.instance.calls


def test_formal_upload_finalizes_but_does_not_publish(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archives, manifest = _inputs(tmp_path)
    assert (
        main(
            _arguments(archives, manifest)
            + [
                "--publication-mode",
                "formal",
                "--execute-token",
                EXECUTE_TOKEN,
                "--defer-byte-readback",
            ],
            dataset_cls=_FakeDataset,
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["dataset"]["name"] == DEFAULT_FORMAL_NAME
    assert result["dataset"]["publication_state"] == "finalized"
    assert "formal-candidate" in result["dataset"]["tags"]
    assert "local-smoke" not in result["dataset"]["tags"]
    assert "publish" not in _FakeDataset.instance.calls


def test_formal_mode_rejects_same_process_readback(tmp_path: Path) -> None:
    archives, manifest = _inputs(tmp_path)
    with pytest.raises(ClearMLDatasetError, match="independent"):
        main(
            _arguments(archives, manifest, readback=tmp_path / "readback")
            + [
                "--publication-mode",
                "formal",
                "--execute-token",
                EXECUTE_TOKEN,
            ],
            dataset_cls=_FakeDataset,
        )


def test_formal_cold_readback_verifies_without_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archives, manifest = _inputs(tmp_path)
    assert (
        main(
            _arguments(archives, manifest)
            + [
                "--publication-mode",
                "formal",
                "--execute-token",
                EXECUTE_TOKEN,
                "--defer-byte-readback",
            ],
            dataset_cls=_FakeDataset,
        )
        == 0
    )
    capsys.readouterr()

    assert (
        verify_main(
            [
                "--dataset-id",
                "dataset-id",
                "--manifest",
                str(manifest),
                "--version",
                "smoke-v1",
                "--publication-mode",
                "formal",
                "--readback-dir",
                str(tmp_path / "formal-readback"),
                "--cold-cache-dir",
                str(tmp_path / "formal-cold-cache"),
            ],
            dataset_cls=_FakeDataset,
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["published_by_this_run"] is False
    assert result["scientific_claims_allowed"] is False
    assert result["release_identity_status"] == "unverified-source-bytes"
    assert result["dataset"]["publication_state"] == "finalized"
    assert "verify_dataset_hash:False:True" in _FakeDataset.instance.calls
    assert "publish" not in _FakeDataset.instance.calls
    assert _FakeDataset.get_requests[-1] == {
        "dataset_id": "dataset-id",
        "only_completed": True,
    }


def test_unverified_local_mirror_publish_fails_before_clearml_access(
    tmp_path: Path,
) -> None:
    _, manifest = _inputs(tmp_path)

    with pytest.raises(ClearMLDatasetError, match="unverified-source-bytes"):
        verify_main(
            [
                "--dataset-id",
                "dataset-id",
                "--manifest",
                str(manifest),
                "--version",
                "smoke-v1",
                "--publication-mode",
                "formal",
                "--readback-dir",
                str(tmp_path / "readback"),
                "--cold-cache-dir",
                str(tmp_path / "cold-cache"),
                "--publish",
                "--execute-token",
                PUBLISH_TOKEN,
            ],
            dataset_cls=_FakeDataset,
        )

    assert _FakeDataset.instance is None
    assert _FakeDataset.get_requests == []


def test_publish_requires_dedicated_token(tmp_path: Path) -> None:
    _, manifest = _inputs(tmp_path)
    with pytest.raises(ClearMLDatasetError, match="publish execute token"):
        verify_main(
            [
                "--dataset-id",
                "dataset-id",
                "--manifest",
                str(manifest),
                "--version",
                "smoke-v1",
                "--publication-mode",
                "formal",
                "--readback-dir",
                str(tmp_path / "readback"),
                "--cold-cache-dir",
                str(tmp_path / "cold-cache"),
                "--publish",
                "--execute-token",
                "wrong",
            ],
            dataset_cls=_FakeDataset,
        )


def test_clearml_internal_full_hash_mismatch_blocks_cold_verification(
    tmp_path: Path,
) -> None:
    archives, manifest = _inputs(tmp_path)
    main(
        _arguments(archives, manifest)
        + [
            "--publication-mode",
            "formal",
            "--execute-token",
            EXECUTE_TOKEN,
            "--defer-byte-readback",
        ],
        dataset_cls=_FakeDataset,
    )
    _FakeDataset.internal_hash_mismatches = (ARCHIVE_NAMES[0],)
    with pytest.raises(ClearMLDatasetError, match="verify_dataset_hash mismatch"):
        verify_main(
            [
                "--dataset-id",
                "dataset-id",
                "--manifest",
                str(manifest),
                "--version",
                "smoke-v1",
                "--publication-mode",
                "formal",
                "--readback-dir",
                str(tmp_path / "readback"),
                "--cold-cache-dir",
                str(tmp_path / "cold-cache"),
            ],
            dataset_cls=_FakeDataset,
        )
    assert "publish" not in _FakeDataset.instance.calls


def test_external_manifest_raw_sha_mismatch_blocks_verification(tmp_path: Path) -> None:
    archives, manifest = _inputs(tmp_path)
    main(
        _arguments(archives, manifest)
        + ["--execute-token", EXECUTE_TOKEN, "--defer-byte-readback"],
        dataset_cls=_FakeDataset,
    )
    _FakeDataset.corrupt_manifest_readback = True
    with pytest.raises(ClearMLDatasetError, match="external manifest SHA-256"):
        verify_main(
            [
                "--dataset-id",
                "dataset-id",
                "--manifest",
                str(manifest),
                "--version",
                "smoke-v1",
                "--readback-dir",
                str(tmp_path / "readback"),
                "--cold-cache-dir",
                str(tmp_path / "cold-cache"),
            ],
            dataset_cls=_FakeDataset,
        )


def test_symlink_in_cold_cache_is_rejected(tmp_path: Path) -> None:
    archives, manifest = _inputs(tmp_path)
    main(
        _arguments(archives, manifest)
        + ["--execute-token", EXECUTE_TOKEN, "--defer-byte-readback"],
        dataset_cls=_FakeDataset,
    )
    _FakeDataset.symlink_readback = True
    with pytest.raises(ClearMLDatasetError, match="non-regular entry"):
        verify_main(
            [
                "--dataset-id",
                "dataset-id",
                "--manifest",
                str(manifest),
                "--version",
                "smoke-v1",
                "--readback-dir",
                str(tmp_path / "readback"),
                "--cold-cache-dir",
                str(tmp_path / "cold-cache"),
            ],
            dataset_cls=_FakeDataset,
        )
