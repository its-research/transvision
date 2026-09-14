from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.event_track_v2x import run_clearml_development_canary as runtime
from tools.event_track_v2x import submit_clearml_development_canary as submit
from transvision.models.event_track_v2x.development_split import (
    build_development_split_manifest_v1,
)
from transvision.models.event_track_v2x.learning_contracts import (
    AssociationTrainingArtifactManifestV1,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes


SOURCE_CLEAN = {
    "branch": "EventTrackV2X",
    "commit": "a" * 40,
    "dirty": False,
    "worktree_state_sha256": "b" * 64,
}


def _arguments(
    root: Path, *, execute: bool = False, execute_local: bool = False
) -> list[str]:
    result = [
        "--dataset-root",
        "/worker/train-only/V2X-Seq-SPD",
        "--split",
        str(root / "protocols/split.json"),
        "--development-manifest",
        str(root / "protocols/development-5fold.json"),
        "--config",
        str(root / "configs/association-canary.json"),
        "--appearance-cache",
        str(root / "fixtures/appearance.npz"),
        "--fold-id",
        "2",
        "--seed",
        "1337",
        "--max-frame-pairs-per-sequence",
        "4",
    ]
    if execute:
        result.extend(
            ("--runtime-image", "registry.invalid/eventtrack:py310", "--execute-remote")
        )
    if execute_local:
        result.append("--execute-local")
    return result


def _fixture(root: Path) -> None:
    (root / "protocols").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures/appearance.npz").write_bytes(b"appearance-cache")
    sequences = [f"sequence-{index:02d}" for index in range(46)]
    split = root / "protocols/split.json"
    split.write_text(
        json.dumps(
            {
                "batch_split": {
                    "train": sequences,
                    "val": ["sealed-val"],
                    "test": ["forbidden-test"],
                    "test_A": ["forbidden-test-a"],
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    split_sha256 = hashlib.sha256(split.read_bytes()).hexdigest()
    submit.OFFICIAL_SPLIT_SHA256 = split_sha256
    runtime.OFFICIAL_SPLIT_SHA256 = split_sha256
    manifest = build_development_split_manifest_v1(
        sequences,
        split_sha256=split_sha256,
    )
    (root / "protocols/development-5fold.json").write_bytes(manifest.canonical_bytes)
    (root / "configs/association-canary.json").write_text(
        '{"epochs":1}\n', encoding="utf-8"
    )


def _write_train_only_manifest(dataset_root: Path, split_sha256: str) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "document_type": "event_track_v2x_spd_train_only_manifest",
        "dataset": "V2X-Seq-SPD",
        "subset": "official-train-camera-only",
        "status": "materialized",
        "output_tree_sha256": hashlib.sha256(b"").hexdigest(),
        "output_tree_hash_algorithm": (
            "sha256-canonical-json-lines-path-size-sha256-v1"
        ),
        "source_split": {"sha256": split_sha256},
        "policy": {
            "included_partition": "train",
            "excluded_partitions": ["val", "test", "test_A"],
            "point_cloud_payloads_included": False,
            "data_info_projection": "train-records-only",
        },
    }
    payload["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    (dataset_root.parent / "train-only-manifest.json").write_bytes(
        canonical_json_bytes(payload) + b"\n"
    )


def _write_training_artifacts(output: Path, parameters: dict[str, object]) -> None:
    output.mkdir(parents=True)
    checkpoint = output / "association-checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    manifest = AssociationTrainingArtifactManifestV1(
        fold_id=int(parameters["fold_id"]),
        training_seed=int(parameters["seed"]),
        development_manifest_sha256=str(
            parameters["expected_development_manifest_sha256"]
        ),
        official_split_sha256=str(parameters["expected_split_sha256"]),
        cohort_sha256="c" * 64,
        config_sha256=str(parameters["expected_config_sha256"]),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        appearance_source=(
            f"frozen_npz_{parameters['appearance_cache_sha256']}_v1"
            if parameters.get("appearance_cache_sha256")
            else f"frozen_resnet50_{parameters['appearance_checkpoint_sha256']}_128_v1"
        ),
        covariance_source="fixture-covariance-v1",
        train_sequence_count=1,
        validation_sequence_count=1,
        train_frame_count=1,
        validation_frame_count=1,
        train_object_count=1,
        validation_object_count=1,
        metrics={
            "train_loss": 0.1,
            "validation_assignment_accuracy": 0.5,
            "validation_loss": 0.2,
            "validation_pair_accuracy": 0.5,
        },
    )
    (output / "association-training-manifest.json").write_bytes(
        manifest.canonical_bytes
    )


def _plan(root: Path, *, source: dict[str, object] | None = None) -> dict[str, object]:
    args = submit._parser().parse_args(_arguments(root))
    return submit.build_submission_plan(
        args,
        root=root,
        source=source or SOURCE_CLEAN,
    )


def test_default_is_network_free_dry_run_with_complete_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fixture(tmp_path)

    assert submit.main(_arguments(tmp_path), root=tmp_path, source=SOURCE_CLEAN) == 0

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["mode"] == "dry-run"
    assert receipt["submission_performed"] is False
    assert receipt["task_created"] is False
    assert receipt["task_enqueued"] is False
    assert receipt["project"] == "Thesis/EventTrack-V2X/Development"
    assert receipt["queue"] == "GPU4-A100"
    assert "development" in receipt["task_name"]
    assert "train-only" in receipt["task_name"]
    assert "seed-1337" in receipt["task_name"]
    assert "fold-2" in receipt["task_name"]
    assert receipt["dataset"] == {
        "task_id": "691397743f934284b9419582adcce0f6",
        "required_tag": "scientific-claim-forbidden",
    }
    assert receipt["source"] == SOURCE_CLEAN
    assert set(receipt["bindings"]) == {
        "split_sha256",
        "development_manifest_sha256",
        "development_manifest_content_sha256",
        "config_file_sha256",
        "config_sha256",
    }
    assert receipt["scope"] == {
        "split": "train",
        "forbidden_splits": ["val", "test", "test_A"],
        "development_only": True,
        "ranking_eligible": False,
        "formal_evidence": False,
    }
    command = receipt["training_command"]
    assert "tools/event_track_v2x/train_association.py" in command
    assert command[command.index("--fold-id") + 1] == "2"
    assert command[command.index("--seed") + 1] == "1337"
    assert not any(item in command for item in ("val", "test", "test_A"))
    assert receipt["runtime"]["dependency_bootstrap"] == []
    assert receipt["runtime"]["preinstalled_dependencies"] == ["numpy", "torch"]
    assert receipt["runtime"]["observed_a100_image_compatible"] is False


def test_plan_accepts_materialized_train_only_directory_name(tmp_path: Path) -> None:
    _fixture(tmp_path)
    arguments = _arguments(tmp_path)
    arguments[arguments.index("--dataset-root") + 1] = (
        "/worker/V2X-Seq-SPD-train-only/V2X-Seq-SPD"
    )

    plan = submit.build_submission_plan(
        submit._parser().parse_args(arguments),
        root=tmp_path,
        source=SOURCE_CLEAN,
    )

    assert plan["parameters"]["Args/dataset_root"] == (
        "/worker/V2X-Seq-SPD-train-only/V2X-Seq-SPD"
    )


def test_plan_rejects_non_train_only_root_and_mismatched_split(tmp_path: Path) -> None:
    _fixture(tmp_path)
    arguments = _arguments(tmp_path)
    arguments[arguments.index("--dataset-root") + 1] = "/worker/V2X-Seq-SPD"
    with pytest.raises(
        submit.DevelopmentCanarySubmissionError, match="isolated train-only"
    ):
        submit.build_submission_plan(
            submit._parser().parse_args(arguments),
            root=tmp_path,
            source=SOURCE_CLEAN,
        )

    split = tmp_path / "protocols/split.json"
    value = json.loads(split.read_text(encoding="utf-8"))
    value["batch_split"]["train"] = value["batch_split"]["train"][:-1]
    split.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        submit.DevelopmentCanarySubmissionError,
        match="pinned official V2X-Seq-SPD split",
    ):
        _plan(tmp_path)


class _DatasetAPI:
    tags = ["scientific-claim-forbidden", "restricted"]
    requests: list[dict[str, object]] = []
    local_copy: Path | None = None

    @classmethod
    def get(cls, **kwargs: object) -> object:
        cls.requests.append(dict(kwargs))
        return SimpleNamespace(
            tags=list(cls.tags),
            get_local_copy=lambda: (
                str(cls.local_copy) if cls.local_copy is not None else ""
            ),
        )


class _Task:
    def __init__(self, owner: type["_TaskAPI"]) -> None:
        self.owner = owner
        self.id = "1" * 32
        self.status = "created"
        self.script: dict[str, object] = {}
        self.parameters: dict[str, object] = {}
        self.tags: list[str] = []
        self.packages: list[str] = []
        self.docker: tuple[str, str] | None = None
        self.closed = False
        self.failure: dict[str, object] | None = None

    def set_script(self, **kwargs: object) -> None:
        self.script = dict(kwargs)

    def set_packages(self, packages: list[str]) -> None:
        self.packages = list(packages)

    def set_base_docker(self, *, docker_image: str, docker_arguments: str) -> None:
        self.docker = (docker_image, docker_arguments)

    def set_parameters(self, parameters: dict[str, object]) -> None:
        self.parameters = dict(parameters)

    def set_tags(self, tags: list[str]) -> bool:
        self.tags = list(tags)
        return True

    def flush(self, *, wait_for_uploads: bool) -> bool:
        assert wait_for_uploads
        return True

    def close(self) -> None:
        self.closed = True

    def mark_failed(self, **kwargs: object) -> None:
        self.failure = dict(kwargs)


class _TaskAPI:
    TaskTypes = SimpleNamespace(training="training")
    created: _Task | None = None
    create_count = 0
    enqueue_count = 0

    @classmethod
    def reset(cls) -> None:
        cls.created = None
        cls.create_count = 0
        cls.enqueue_count = 0

    @classmethod
    def get_tasks(cls, **_kwargs: object) -> list[object]:
        return []

    @classmethod
    def create(
        cls,
        *,
        project_name: str,
        task_name: str,
        task_type: object,
        detect_repository: bool,
    ) -> _Task:
        assert project_name == submit.PROJECT
        assert task_name
        assert task_type
        assert detect_repository is False
        cls.create_count += 1
        cls.created = _Task(cls)
        return cls.created

    @classmethod
    def init(cls, **kwargs: object) -> _Task:
        assert kwargs["project_name"] == submit.PROJECT
        cls.create_count += 1
        cls.created = _Task(cls)
        return cls.created

    @classmethod
    def enqueue(cls, task: _Task, *, queue_name: str) -> None:
        assert queue_name == submit.QUEUE
        cls.enqueue_count += 1
        task.status = "queued"

    @classmethod
    def get_task(cls, *, task_id: str) -> _Task:
        assert cls.created is not None and task_id == cls.created.id
        return cls.created


def test_execute_remote_configures_and_enqueues_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fixture(tmp_path)
    clearml_config = tmp_path / "clearml.conf"
    clearml_config.write_text("api {}\n", encoding="utf-8")
    monkeypatch.setenv("CLEARML_CONFIG_FILE", str(clearml_config))
    _TaskAPI.reset()
    _DatasetAPI.requests = []
    _DatasetAPI.tags = ["scientific-claim-forbidden", "restricted"]

    assert (
        submit.main(
            _arguments(tmp_path, execute=True),
            task_api=_TaskAPI,
            dataset_api=_DatasetAPI,
            root=tmp_path,
            source=SOURCE_CLEAN,
        )
        == 0
    )

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["mode"] == "execute-remote"
    assert receipt["task_id"] == "1" * 32
    assert receipt["status"] == "queued"
    assert _TaskAPI.create_count == 1
    assert _TaskAPI.enqueue_count == 1
    task = _TaskAPI.created
    assert task is not None
    assert task.script == {
        "repository": submit.REPOSITORY,
        "branch": "EventTrackV2X",
        "commit": "a" * 40,
        "diff": "",
        "working_dir": ".",
        "entry_point": submit.ENTRY_POINT,
    }
    assert task.parameters["Args/dataset_task_id"] == submit.DATASET_TASK_ID
    assert task.parameters["Args/development_only"] is True
    assert task.parameters["Args/ranking_eligible"] is False
    assert task.parameters["Args/formal_evidence"] is False
    assert task.packages == []
    assert "scientific-claim-forbidden" in task.tags
    assert _DatasetAPI.requests == [
        {"dataset_id": submit.DATASET_TASK_ID, "only_completed": True}
    ]


def test_execute_remote_fails_before_creation_for_dirty_source_or_dataset_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path)
    clearml_config = tmp_path / "clearml.conf"
    clearml_config.write_text("api {}\n", encoding="utf-8")
    monkeypatch.setenv("CLEARML_CONFIG_FILE", str(clearml_config))
    dirty = {**SOURCE_CLEAN, "dirty": True}
    _TaskAPI.reset()
    with pytest.raises(
        submit.DevelopmentCanarySubmissionError, match="clean source worktree"
    ):
        submit.main(
            _arguments(tmp_path, execute=True),
            task_api=_TaskAPI,
            dataset_api=_DatasetAPI,
            root=tmp_path,
            source=dirty,
        )
    assert _TaskAPI.create_count == 0

    _DatasetAPI.tags = ["restricted"]
    with pytest.raises(
        submit.DevelopmentCanarySubmissionError,
        match="scientific-claim-forbidden",
    ):
        submit.main(
            _arguments(tmp_path, execute=True),
            task_api=_TaskAPI,
            dataset_api=_DatasetAPI,
            root=tmp_path,
            source=SOURCE_CLEAN,
        )
    assert _TaskAPI.create_count == 0


def _runtime_parameters(plan: dict[str, object]) -> dict[str, object]:
    raw = plan["parameters"]
    assert isinstance(raw, dict)
    return {key.removeprefix("Args/"): value for key, value in raw.items()}


class _RuntimeTask:
    id = "2" * 32

    def __init__(self) -> None:
        self.uploads: list[dict[str, object]] = []

    def upload_artifact(self, **kwargs: object) -> bool:
        self.uploads.append(dict(kwargs))
        return True

    def flush(self, *, wait_for_uploads: bool) -> bool:
        return wait_for_uploads


def test_remote_runner_calls_train_association_and_preserves_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path)
    dataset_root = tmp_path / "worker/train-only/V2X-Seq-SPD"
    dataset_root.mkdir(parents=True)
    arguments = _arguments(tmp_path)
    arguments[arguments.index("--dataset-root") + 1] = str(dataset_root)
    plan = submit.build_submission_plan(
        submit._parser().parse_args(arguments),
        root=tmp_path,
        source=SOURCE_CLEAN,
    )
    parameters = _runtime_parameters(plan)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(
        runtime,
        "TRAIN_ENTRY_POINT",
        tmp_path / "tools/event_track_v2x/train_association.py",
    )
    monkeypatch.setattr(
        runtime,
        "PREPARE_ENTRY_POINT",
        tmp_path / "tools/event_track_v2x/prepare_spd_train_only.py",
    )
    (tmp_path / "tools/event_track_v2x").mkdir(parents=True)
    runtime.TRAIN_ENTRY_POINT.write_text("# fixture\n", encoding="utf-8")
    runtime.PREPARE_ENTRY_POINT.write_text("# fixture\n", encoding="utf-8")
    _DatasetAPI.tags = ["scientific-claim-forbidden", "restricted"]
    _DatasetAPI.local_copy = tmp_path / "archives"
    _DatasetAPI.local_copy.mkdir()
    task = _RuntimeTask()
    observed: dict[str, Any] = {}

    def runner(
        command: list[str], *, cwd: Path, env: dict[str, str], check: bool
    ) -> None:
        if Path(command[1]).name == "prepare_spd_train_only.py":
            output = Path(command[command.index("--output") + 1])
            (output / "V2X-Seq-SPD").mkdir(parents=True)
            _write_train_only_manifest(
                output / "V2X-Seq-SPD",
                str(parameters["expected_split_sha256"]),
            )
            observed["preparation_command"] = command
            return
        observed.update(command=command, cwd=cwd, env=env, check=check)
        output = Path(command[command.index("--output") + 1])
        _write_training_artifacts(output, parameters)

    receipt = runtime.run_remote(
        parameters,
        task=task,
        dataset_api=_DatasetAPI,
        runner=runner,
    )

    command = observed["command"]
    assert Path(observed["preparation_command"][1]).name == "prepare_spd_train_only.py"
    assert Path(command[1]).name == "train_association.py"
    assert command[command.index("--fold-id") + 1] == "2"
    assert command[command.index("--seed") + 1] == "1337"
    assert not any(item in command for item in ("val", "test", "test_A"))
    assert observed["env"]["EVENTTRACK_V2X_ALLOWED_SPLIT"] == "train"
    assert observed["env"]["EVENTTRACK_V2X_FORBIDDEN_SPLITS"] == "val,test,test_A"
    assert receipt["development_only"] is True
    assert receipt["ranking_eligible"] is False
    assert receipt["formal_evidence"] is False
    assert len(task.uploads) == 2
    assert all(
        upload["metadata"]["development_only"] is True
        and upload["metadata"]["ranking_eligible"] is False
        and upload["metadata"]["formal_evidence"] is False
        for upload in task.uploads
    )


def test_remote_runner_rejects_forbidden_scope_and_unlabelled_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path)
    dataset_root = tmp_path / "worker/train-only/V2X-Seq-SPD"
    dataset_root.mkdir(parents=True)
    arguments = _arguments(tmp_path)
    arguments[arguments.index("--dataset-root") + 1] = str(dataset_root)
    plan = submit.build_submission_plan(
        submit._parser().parse_args(arguments),
        root=tmp_path,
        source=SOURCE_CLEAN,
    )
    parameters = _runtime_parameters(plan)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(
        runtime,
        "TRAIN_ENTRY_POINT",
        tmp_path / "tools/event_track_v2x/train_association.py",
    )
    monkeypatch.setattr(
        runtime,
        "PREPARE_ENTRY_POINT",
        tmp_path / "tools/event_track_v2x/prepare_spd_train_only.py",
    )
    (tmp_path / "tools/event_track_v2x").mkdir(parents=True)
    runtime.TRAIN_ENTRY_POINT.write_text("# fixture\n", encoding="utf-8")
    runtime.PREPARE_ENTRY_POINT.write_text("# fixture\n", encoding="utf-8")
    _DatasetAPI.tags = ["scientific-claim-forbidden", "restricted"]
    _DatasetAPI.local_copy = tmp_path / "archives"
    _DatasetAPI.local_copy.mkdir()

    forbidden = dict(parameters)
    forbidden["split_name"] = "val"
    with pytest.raises(runtime.DevelopmentCanaryRuntimeError, match="only the train"):
        runtime.build_training_command(forbidden)

    def bad_runner(
        command: list[str], *, cwd: Path, env: dict[str, str], check: bool
    ) -> None:
        if Path(command[1]).name == "prepare_spd_train_only.py":
            output = Path(command[command.index("--output") + 1])
            (output / "V2X-Seq-SPD").mkdir(parents=True)
            _write_train_only_manifest(
                output / "V2X-Seq-SPD",
                str(parameters["expected_split_sha256"]),
            )
            return
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "association-checkpoint.pt").write_bytes(b"weights")
        (output / "association-training-manifest.json").write_text(
            json.dumps(
                {
                    "development_only": True,
                    "ranking_eligible": True,
                    "formal_evidence": False,
                }
            ),
            encoding="utf-8",
        )

    task = _RuntimeTask()
    with pytest.raises(
        runtime.DevelopmentCanaryRuntimeError,
        match="training manifest is invalid",
    ):
        runtime.run_remote(
            parameters,
            task=task,
            dataset_api=_DatasetAPI,
            runner=bad_runner,
        )
    assert task.uploads == []


def test_execute_local_initializes_tracked_task_without_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fixture(tmp_path)
    dataset_root = tmp_path / "worker/train-only/V2X-Seq-SPD"
    dataset_root.mkdir(parents=True)
    arguments = _arguments(tmp_path, execute_local=True)
    arguments[arguments.index("--dataset-root") + 1] = str(dataset_root)
    cache_index = arguments.index("--appearance-cache")
    del arguments[cache_index : cache_index + 2]
    checkpoint = tmp_path / "external/resnet50.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"frozen-resnet50")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    arguments.extend(
        (
            "--appearance-checkpoint",
            str(checkpoint),
            "--appearance-checkpoint-sha256",
            checkpoint_sha256,
        )
    )
    clearml_config = tmp_path / "clearml.conf"
    clearml_config.write_text("api {}\n", encoding="utf-8")
    monkeypatch.setenv("CLEARML_CONFIG_FILE", str(clearml_config))
    _TaskAPI.reset()
    _DatasetAPI.tags = ["scientific-claim-forbidden", "restricted"]
    monkeypatch.setattr(
        runtime,
        "run_remote",
        lambda parameters, **kwargs: {
            "mode": "remote-completed",
            **runtime.BOUNDARY,
        },
    )

    assert (
        submit.main(
            arguments,
            task_api=_TaskAPI,
            dataset_api=_DatasetAPI,
            root=tmp_path,
            source={**SOURCE_CLEAN, "dirty": True},
        )
        == 0
    )

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["mode"] == "execute-local"
    assert receipt["task_created"] is True
    assert receipt["task_enqueued"] is False
    assert receipt["status"] == "local-completed"
    assert receipt["appearance"] == {
        "mode": "checkpoint",
        "path": str(checkpoint),
        "sha256": checkpoint_sha256,
    }
    assert _TaskAPI.create_count == 1
    assert _TaskAPI.enqueue_count == 0
    assert _TaskAPI.created is not None and _TaskAPI.created.closed is True


def test_submitter_has_no_top_level_clearml_import() -> None:
    source = Path(submit.__file__).read_text(encoding="utf-8")
    prefix = source.split("def main(", 1)[0]
    assert "from clearml import" not in prefix
    assert "import clearml" not in prefix
