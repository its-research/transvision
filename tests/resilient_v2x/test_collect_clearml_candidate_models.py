from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tools.resilient_v2x import collect_clearml_candidate_models as collector


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


class _Artifact:
    def __init__(self, name: str, payload: dict[str, object]) -> None:
        self.name = name
        self.url = f"http://10.100.34.118:8081/task/artifacts/{name}/{name}.json"
        self.refresh(payload)

    def refresh(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.content = _json_bytes(payload)
        self.size = len(self.content)
        self.hash = _sha_bytes(self.content)


class _Model:
    def __init__(self, *, model_id: str, task_id: str, name: str, url: str) -> None:
        self.id = model_id
        self.task = task_id
        self.name = name
        self.url = url
        self.cache_calls = 0

    def get_local_copy(self, **_: object) -> str:
        self.cache_calls += 1
        raise AssertionError("candidate collector must not use the ClearML model cache")


class _Task:
    def __init__(
        self,
        *,
        task_id: str,
        name: str,
        parent: str,
        status: str,
        parameters: dict[str, object],
        artifacts: dict[str, _Artifact],
        models: list[_Model],
    ) -> None:
        self.id = task_id
        self.name = name
        self.parent = parent
        self.status = status
        self._parameters = parameters
        self.artifacts = artifacts
        self._models = models
        self.reload_calls = 0

    def reload(self) -> None:
        self.reload_calls += 1

    def get_parameters(self) -> dict[str, object]:
        return dict(self._parameters)

    def get_models(self) -> dict[str, list[_Model]]:
        return {"input": [], "output": list(self._models)}


class _Tasks:
    def __init__(self, task: _Task) -> None:
        self.task = task
        self.calls: list[str] = []

    def get_task(self, *, task_id: str) -> _Task:
        self.calls.append(task_id)
        assert task_id == self.task.id
        return self.task


@dataclass
class _Fixture:
    spec: collector.CandidateSpec
    task: _Task
    tasks: _Tasks
    final_bytes: bytes
    best_bytes: bytes

    @property
    def auth(self) -> dict[str, Any]:
        return {
            "auth_header_provider": lambda: {"Authorization": "Bearer test"},
            "auth_refresher": lambda: None,
            "artifact_bytes_loader": (
                lambda artifact, _reference, _provider, _refresher: artifact.content
            ),
        }


def _make_fixture(
    tmp_path: Path,
    *,
    status: str = "completed",
    label: str = "T1",
    subject: str = "test_candidate",
) -> _Fixture:
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True)
    source_file = b"source-file"
    inventory = {
        "schema_version": 1,
        "tree_sha256": "3" * 64,
        "file_count": 1,
        "source_bytes": len(source_file),
        "files": [
            {
                "path": "candidate.py",
                "size_bytes": len(source_file),
                "sha256": _sha_bytes(source_file),
                "mode": 420,
            }
        ],
    }
    inventory_path = source_root / "source-inventory.json"
    inventory_path.write_bytes(_json_bytes(inventory))
    archive_path = source_root / "source.tar.zst"
    archive_path.write_bytes(b"sealed-source-archive")
    source = collector.SourceIdentity(
        dataset_id="4" * 32,
        archive_name=archive_path.name,
        archive_size_bytes=archive_path.stat().st_size,
        archive_sha256=_sha_bytes(archive_path.read_bytes()),
        tree_sha256=str(inventory["tree_sha256"]),
        file_count=1,
        source_bytes=len(source_file),
        inventory_size_bytes=inventory_path.stat().st_size,
        inventory_sha256=_sha_bytes(inventory_path.read_bytes()),
        archive_path=archive_path,
        inventory_path=inventory_path,
    )
    task_id = "a" * 32
    config_path = f"configs/resilient_v2x/improvements/{subject}.py"
    config_sha = "5" * 64
    spec = collector.CandidateSpec(
        label=label,
        subject=subject,
        task_id=task_id,
        task_name=f"ResilientV2X candidate {label}",
        config_path=config_path,
        config_sha256=config_sha,
        source=source,
    )
    final_bytes = b"epoch-50-final-weights"
    best_bytes = b"clean-best-diagnostic-weights"
    final_url = (
        "http://10.100.34.118:8081/Training/Candidate Task/models/"
        f"{subject}_epoch_50.pth"
    )
    best_url = (
        "http://10.100.34.118:8081/Training/Candidate Task/models/"
        f"{subject}_clean_val_best_epoch_30.pth"
    )
    final_contract = {
        "filename": "epoch_50.pth",
        "model_id": "1" * 32,
        "name": f"ResilientV2X {subject} final checkpoint",
        "sha256": _sha_bytes(final_bytes),
        "size_bytes": len(final_bytes),
        "url": final_url,
    }
    best_contract = {
        "claim_role": "diagnostic checkpoint candidate; final remains canonical",
        "epoch": 30,
        "filename": "best_resilient_v2x_car_bev_ap_r40_0.70_epoch_30.pth",
        "model_id": "2" * 32,
        "name": f"ResilientV2X {subject} clean-val best checkpoint",
        "selection_metric": collector.CLEAN_SELECTION_METRIC,
        "selection_protocol": collector.CLEAN_SELECTION_PROTOCOL,
        "sha256": _sha_bytes(best_bytes),
        "size_bytes": len(best_bytes),
        "url": best_url,
    }
    run_contract = {
        "schema_version": 1,
        "task_id": task_id,
        "experiment": subject,
        "experiment_kind": "sota_candidate",
        "source_dataset_id": source.dataset_id,
        "training_dataset_id": collector.TRAINING_DATASET_ID,
        "seed": collector.TRAINING_SEED,
        "gpus": 4,
        "ddp_processes": 4,
        "global_batch_size": 8,
        "train_batch_size_per_gpu": 2,
        "max_epochs": 50,
        "val_interval": 10,
        "amp": False,
        "precision": "FP32",
        "auto_scale_lr": False,
        "condition_evaluation": False,
        "predecessor_task_id": collector.PREDECESSOR_TASK_ID,
        "source_archive": {
            "name": source.archive_name,
            "sha256": source.archive_sha256,
            "size_bytes": source.archive_size_bytes,
        },
        "config": {
            "declared": config_path,
            "config_sha256": config_sha,
            "resolved_config_sha256": config_sha,
        },
        "teacher": {
            "task_id": collector.TEACHER_TASK_ID,
            "model_id": collector.TEACHER_MODEL_ID,
            "name": collector.TEACHER_MODEL_NAME,
            "expected_sha256": collector.TEACHER_CHECKPOINT_SHA256,
            "sha256": collector.TEACHER_CHECKPOINT_SHA256,
            "size_bytes": collector.TEACHER_CHECKPOINT_BYTES,
        },
        "common_teacher_initialization": {
            "audit_artifact_name": collector.TEACHER_AUDIT_ARTIFACT,
            "contract": "shared-only-clean-teacher-initialization-v1",
            "policy": "shared-only",
            "teacher_checkpoint_sha256": collector.TEACHER_CHECKPOINT_SHA256,
        },
    }
    teacher_audit = {
        "schema_version": 1,
        "contract": "shared-only-clean-teacher-initialization-v1",
        "result": "pass",
        "checkpoint": {
            "expected_sha256": collector.TEACHER_CHECKPOINT_SHA256,
            "sha256": collector.TEACHER_CHECKPOINT_SHA256,
            "size_bytes": collector.TEACHER_CHECKPOINT_BYTES,
        },
        "shared_initialization": {
            "exact_tensor_equality_verified": True,
            "shape_dtype_verified": True,
            "keys": 468,
            "expected_keys": 468,
        },
        "method_specific_fusion": {
            "unchanged": True,
            "sha256_before": "6" * 64,
            "sha256_after": "6" * 64,
        },
        "target": {
            "nested_teacher_present": True,
            "nested_teacher_full_equality_verified": True,
            "model_type": "ResilientV2XNet",
        },
    }
    artifacts = {
        collector.RUN_ARTIFACT: _Artifact(collector.RUN_ARTIFACT, run_contract),
        collector.TEACHER_AUDIT_ARTIFACT: _Artifact(
            collector.TEACHER_AUDIT_ARTIFACT, teacher_audit
        ),
        collector.FINAL_ARTIFACT: _Artifact(collector.FINAL_ARTIFACT, final_contract),
        collector.BEST_ARTIFACT: _Artifact(collector.BEST_ARTIFACT, best_contract),
    }
    parameters = {
        "Args/experiment_from_task": subject,
        "Args/source_dataset_id": source.dataset_id,
        "Args/source_archive_name": source.archive_name,
        "Args/source_archive_bytes": str(source.archive_size_bytes),
        "Args/source_archive_sha256": source.archive_sha256,
        "Args/training_dataset_id": collector.TRAINING_DATASET_ID,
        "Args/training_seed": str(collector.TRAINING_SEED),
        "Args/gpus": "4",
        "Args/max_epochs": "50",
        "Args/amp": "False",
        "Args/teacher_task_id": collector.TEACHER_TASK_ID,
        "Args/teacher_model_id": collector.TEACHER_MODEL_ID,
        "Args/teacher_checkpoint_sha256": collector.TEACHER_CHECKPOINT_SHA256,
    }
    models = [
        _Model(
            model_id=str(final_contract["model_id"]),
            task_id=task_id,
            name=str(final_contract["name"]),
            url=final_url,
        ),
        _Model(
            model_id=str(best_contract["model_id"]),
            task_id=task_id,
            name=str(best_contract["name"]),
            url=best_url,
        ),
    ]
    task = _Task(
        task_id=task_id,
        name=spec.task_name,
        parent=spec.parent_task_id,
        status=status,
        parameters=parameters,
        artifacts=artifacts,
        models=models,
    )
    return _Fixture(
        spec=spec,
        task=task,
        tasks=_Tasks(task),
        final_bytes=final_bytes,
        best_bytes=best_bytes,
    )


def _inspect(fixture: _Fixture) -> dict[str, object]:
    return collector.inspect_candidate(
        task_class=fixture.tasks,
        spec=fixture.spec,
        **fixture.auth,
    )


def _downloader(fixture: _Fixture, calls: list[str]):
    def download(**kwargs: object) -> tuple[Path, bool]:
        destination = Path(str(kwargs["destination"]))
        url = str(kwargs["model_url"])
        value = fixture.best_bytes if "clean_val_best" in url else fixture.final_bytes
        destination.write_bytes(value)
        calls.append(url)
        return destination.resolve(), False

    return download


def _write_legacy_archive(
    fixture: _Fixture, output_root: Path, *, empty_clearml_tree: bool = False
) -> Path:
    directory = output_root / fixture.spec.subject
    directory.mkdir(parents=True)
    snapshot = _inspect(fixture)
    checkpoints = snapshot["checkpoints"]
    assert isinstance(checkpoints, list)
    values = {
        "canonical_final": fixture.final_bytes,
        "clean_val_best_diagnostic": fixture.best_bytes,
    }
    models = []
    for item in checkpoints:
        assert isinstance(item, dict)
        (directory / str(item["filename"])).write_bytes(values[str(item["role"])])
        models.append(
            {
                "role": item["role"],
                "filename": item["filename"],
                "model_id": item["model_id"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "subject": fixture.spec.subject,
                "source_task_id": fixture.spec.task_id,
                "models": models,
            },
            indent=2,
        )
        + "\n"
    )
    if empty_clearml_tree:
        (directory / "ResilientV2X/Training/task/models").mkdir(parents=True)
    return directory


def test_catalog_pins_initial_five_candidates() -> None:
    assert list(collector.SPEC_BY_LABEL) == ["E1", "E2", "E3", "P0", "P2"]
    assert [spec.task_id for spec in collector.CANDIDATE_SPECS] == [
        "f0c3082f3aa34a81805903e0ffdc8610",
        "969c8fce6d24446299561772b3955274",
        "dc037315c0684c3d854a2fd7c19a2a2f",
        "8883c51ced4f4951a45edbaefe6342d4",
        "f5d3820b4cdf416183c8f1fee566abe3",
    ]


def test_inspect_accepts_completed_single_seed_contract(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    snapshot = _inspect(fixture)
    assert snapshot["status"] == "completed"
    assert snapshot["run_contract"] == {
        "task_id": fixture.spec.task_id,
        "experiment": fixture.spec.subject,
        "config_path": fixture.spec.config_path,
        "config_sha256": fixture.spec.config_sha256,
        "training_dataset_id": collector.TRAINING_DATASET_ID,
        "seed": 20250218,
        "global_batch_size": 8,
        "gpus": 4,
        "max_epochs": 50,
        "val_interval": 10,
        "precision": "FP32",
        "amp": False,
    }
    checkpoints = snapshot["checkpoints"]
    assert isinstance(checkpoints, list)
    assert [item["role"] for item in checkpoints] == [
        "canonical_final",
        "clean_val_best_diagnostic",
    ]
    assert [item["paper_eligible"] for item in checkpoints] == [True, False]


def test_not_ready_fails_before_reading_artifacts(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, status="in_progress")
    calls = 0

    def loader(*_: object) -> bytes:
        nonlocal calls
        calls += 1
        raise AssertionError

    with pytest.raises(collector.CandidateNotReady, match="not completed"):
        collector.inspect_candidate(
            task_class=fixture.tasks,
            spec=fixture.spec,
            auth_header_provider=lambda: {"Authorization": "Bearer test"},
            auth_refresher=lambda: None,
            artifact_bytes_loader=loader,
        )
    assert calls == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seed", 7),
        ("global_batch_size", 4),
        ("precision", "AMP"),
        ("training_dataset_id", "9" * 32),
        ("max_epochs", 49),
        ("val_interval", 1),
        ("amp", True),
    ],
)
def test_run_contract_drift_fails_closed(
    tmp_path: Path, key: str, value: object
) -> None:
    fixture = _make_fixture(tmp_path)
    artifact = fixture.task.artifacts[collector.RUN_ARTIFACT]
    payload = dict(artifact.payload)
    payload[key] = value
    artifact.refresh(payload)
    with pytest.raises(RuntimeError, match=f"run contract drifted: {key}"):
        _inspect(fixture)


def test_source_teacher_and_final_contract_drift_fail_closed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run = fixture.task.artifacts[collector.RUN_ARTIFACT]
    payload = dict(run.payload)
    teacher = dict(payload["teacher"])
    teacher["model_id"] = "9" * 32
    payload["teacher"] = teacher
    run.refresh(payload)
    with pytest.raises(RuntimeError, match="teacher drifted: model_id"):
        _inspect(fixture)

    fixture = _make_fixture(tmp_path / "second")
    final = fixture.task.artifacts[collector.FINAL_ARTIFACT]
    payload = dict(final.payload)
    payload["filename"] = "epoch_40.pth"
    final.refresh(payload)
    with pytest.raises(RuntimeError, match="not epoch_50"):
        _inspect(fixture)

    fixture = _make_fixture(tmp_path / "third")
    fixture.spec.source.archive_path.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="source archive identity mismatch"):
        _inspect(fixture)


def test_teacher_audit_failure_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    artifact = fixture.task.artifacts[collector.TEACHER_AUDIT_ARTIFACT]
    payload = dict(artifact.payload)
    payload["result"] = "fail"
    artifact.refresh(payload)
    with pytest.raises(RuntimeError, match="audit did not pass"):
        _inspect(fixture)


def test_artifact_bytes_must_match_clearml_hash(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    def corrupt(artifact: _Artifact, *_: object) -> bytes:
        return artifact.content + b"x"

    with pytest.raises(RuntimeError, match="artifact size mismatch"):
        collector.inspect_candidate(
            task_class=fixture.tasks,
            spec=fixture.spec,
            auth_header_provider=lambda: {"Authorization": "Bearer test"},
            auth_refresher=lambda: None,
            artifact_bytes_loader=corrupt,
        )


def test_collect_streams_models_and_writes_bound_manifest(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output_root = tmp_path / "completed-live"
    calls: list[str] = []
    manifest = collector.collect_candidate(
        task_class=fixture.tasks,
        spec=fixture.spec,
        output_root=output_root,
        model_downloader=_downloader(fixture, calls),
        **fixture.auth,
    )
    directory = output_root / fixture.spec.subject
    assert len(calls) == 2
    assert manifest["checkpoint_policy"] == "epoch_50_final_only_for_paper"
    assert manifest["official_checkpoint_role"] == "canonical_final"
    assert [item["paper_eligible"] for item in manifest["models"]] == [True, False]
    assert manifest["content_sha256"] == collector._manifest_content(manifest)
    assert (directory / f"{fixture.spec.subject}_epoch_50.pth").read_bytes() == (
        fixture.final_bytes
    )
    assert all(model.cache_calls == 0 for model in fixture.task._models)


def test_existing_legacy_archive_is_reused_without_download(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output_root = tmp_path / "completed-live"
    directory = _write_legacy_archive(fixture, output_root, empty_clearml_tree=True)

    def forbidden(**_: object) -> tuple[Path, bool]:
        raise AssertionError("existing E1-style files must not be downloaded again")

    manifest = collector.collect_candidate(
        task_class=fixture.tasks,
        spec=fixture.spec,
        output_root=output_root,
        model_downloader=forbidden,
        **fixture.auth,
    )
    assert manifest["schema_version"] == 2
    assert all(item["reused_existing_file"] for item in manifest["models"])
    assert manifest["empty_legacy_directories_ignored"] == [
        "ResilientV2X/Training/task/models"
    ]
    assert json.loads((directory / "manifest.json").read_text()) == manifest


def test_second_collection_is_idempotent(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output_root = tmp_path / "completed-live"
    calls: list[str] = []
    collector.collect_candidate(
        task_class=fixture.tasks,
        spec=fixture.spec,
        output_root=output_root,
        model_downloader=_downloader(fixture, calls),
        **fixture.auth,
    )
    calls.clear()
    collector.collect_candidate(
        task_class=fixture.tasks,
        spec=fixture.spec,
        output_root=output_root,
        model_downloader=_downloader(fixture, calls),
        **fixture.auth,
    )
    assert calls == []
    assert len(list((output_root / fixture.spec.subject).glob("*.pth"))) == 2


def test_insufficient_space_fails_before_any_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    output_root = tmp_path / "completed-live"
    calls: list[str] = []
    monkeypatch.setattr(collector.formal, "_available_bytes", lambda _path: 0)
    with pytest.raises(RuntimeError, match="insufficient free space"):
        collector.collect_candidate(
            task_class=fixture.tasks,
            spec=fixture.spec,
            output_root=output_root,
            model_downloader=_downloader(fixture, calls),
            **fixture.auth,
        )
    assert calls == []
    assert not (output_root / fixture.spec.subject).exists()


def test_bad_download_is_not_published(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output_root = tmp_path / "completed-live"

    def bad(**kwargs: object) -> tuple[Path, bool]:
        destination = Path(str(kwargs["destination"]))
        destination.write_bytes(b"bad")
        return destination.resolve(), False

    with pytest.raises(RuntimeError, match="staged checkpoint identity mismatch"):
        collector.collect_candidate(
            task_class=fixture.tasks,
            spec=fixture.spec,
            output_root=output_root,
            model_downloader=bad,
            **fixture.auth,
        )
    assert not (output_root / fixture.spec.subject).exists()
    assert list((output_root / ".candidate-staging").iterdir()) == []


def test_remote_evidence_drift_during_download_is_not_published(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    output_root = tmp_path / "completed-live"
    calls: list[str] = []

    def drifting(**kwargs: object) -> tuple[Path, bool]:
        result = _downloader(fixture, calls)(**kwargs)
        if len(calls) == 1:
            artifact = fixture.task.artifacts[collector.RUN_ARTIFACT]
            payload = dict(artifact.payload)
            payload["non_contract_note"] = "changed"
            artifact.refresh(payload)
        return result

    with pytest.raises(RuntimeError, match="evidence changed during download"):
        collector.collect_candidate(
            task_class=fixture.tasks,
            spec=fixture.spec,
            output_root=output_root,
            model_downloader=drifting,
            **fixture.auth,
        )
    assert not (output_root / fixture.spec.subject).exists()


def test_matching_blob_is_hardlinked_instead_of_downloaded(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output_root = tmp_path / "completed-live"
    source_dir = output_root / "existing_method"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "existing_epoch_50.pth"
    source_file.write_bytes(fixture.final_bytes)
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "filename": source_file.name,
                        "size_bytes": len(fixture.final_bytes),
                        "sha256": _sha_bytes(fixture.final_bytes),
                    }
                ],
            }
        )
    )
    calls: list[str] = []
    manifest = collector.collect_candidate(
        task_class=fixture.tasks,
        spec=fixture.spec,
        output_root=output_root,
        model_downloader=_downloader(fixture, calls),
        **fixture.auth,
    )
    assert len(calls) == 1
    assert "clean_val_best" in calls[0]
    final = output_root / fixture.spec.subject / f"{fixture.spec.subject}_epoch_50.pth"
    assert os.stat(final).st_ino == os.stat(source_file).st_ino
    records = {item["role"]: item for item in manifest["models"]}
    assert records["canonical_final"]["storage_deduplicated_via_hardlink"] is True


def test_recomputed_manifest_tampering_is_detected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output_root = tmp_path / "completed-live"
    calls: list[str] = []
    collector.collect_candidate(
        task_class=fixture.tasks,
        spec=fixture.spec,
        output_root=output_root,
        model_downloader=_downloader(fixture, calls),
        **fixture.auth,
    )
    path = output_root / fixture.spec.subject / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["models"][0]["paper_eligible"] = False
    manifest["content_sha256"] = collector._manifest_content(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="model drifted"):
        collector.audit_candidate(
            task_class=fixture.tasks,
            spec=fixture.spec,
            output_root=output_root,
            **fixture.auth,
        )


def test_custom_future_candidate_spec_uses_same_contract(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, label="P7", subject="future_p7")
    snapshot = _inspect(fixture)
    assert snapshot["label"] == "P7"
    assert snapshot["subject"] == "future_p7"


def test_collect_cli_requires_exact_execution_token() -> None:
    with pytest.raises(RuntimeError, match=collector.EXECUTE_TOKEN):
        collector.main(["--candidate", "E1", "--collect"])
