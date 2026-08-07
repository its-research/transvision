from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
TEACHER_TASK_ID = "a" * 32
TEACHER_MODEL_ID = "b" * 32
STUDENT_TASK_ID = "c" * 32
STUDENT_MODEL_ID = "d" * 32
VALIDATION_TASK_ID = "e" * 32
DATASET_ID = "f" * 32
SOURCE_DATASET_ID = "1" * 32
PROJECT_ID = "2" * 32


def _load_controller():
    path = ROOT / "tools/resilient_v2x/clearml_5090_paper_controller.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Artifact:
    def __init__(self, value: object, *, url: str | None = None) -> None:
        self.value = value
        self.url = url
        self.downloads = 0

    def get(self, *, force_download: bool = False):
        assert force_download is True
        self.downloads += 1
        return self.value


class _Model:
    def __init__(
        self,
        *,
        name: str,
        task: str,
        model_id: str,
        url: str,
        local_copy: Path,
        events: list[str] | None = None,
    ) -> None:
        self.name = name
        self.task = task
        self.id = model_id
        self.url = url
        self.local_copy = local_copy
        self.events = events

    def get_local_copy(
        self,
        *,
        extract_archive: bool,
        raise_on_error: bool,
        force_download: bool,
    ) -> str:
        assert extract_archive is False
        assert raise_on_error is True
        assert force_download is True
        if self.events is not None:
            self.events.append("student_model_downloaded")
        return str(self.local_copy)


class _Task:
    def __init__(
        self,
        *,
        task_id: str,
        status: str,
        parameters: dict[str, object],
        models: list[_Model] | None = None,
        artifacts: dict[str, _Artifact] | None = None,
        output_uri: str = "http://10.100.34.118:8081",
        on_reload=None,
    ) -> None:
        self.id = task_id
        self.status = status
        self.parameters = dict(parameters)
        self.models = [] if models is None else list(models)
        self.artifacts = {} if artifacts is None else dict(artifacts)
        self.output_uri = output_uri
        self.data = SimpleNamespace(
            script=SimpleNamespace(
                entry_point=("tools/resilient_v2x/clearml_5090_bootstrap.py")
            )
        )
        self.on_reload = on_reload

    def reload(self) -> None:
        if self.on_reload is not None:
            self.on_reload(self)

    def get_parameters(self):
        return dict(self.parameters)

    def set_parameter(self, *, name: str, value: object) -> None:
        self.parameters[name] = value

    def get_models(self):
        return {"output": list(self.models)}


def _student_parameters(
    module,
    *,
    teacher_sha256: str,
    allow_failed_teacher_task: bool,
) -> dict[str, object]:
    return {
        "Args/source_dataset_id": SOURCE_DATASET_ID,
        "Args/source_archive_name": "source.tar.zst",
        "Args/source_archive_bytes": 123,
        "Args/source_archive_sha256": "2" * 64,
        "Args/training_dataset_id": DATASET_ID,
        "Args/native_bundle_bytes": 456,
        "Args/native_bundle_sha256": "3" * 64,
        "Args/build_manifest_sha256": "4" * 64,
        "Args/gpus": module.EXPECTED_GPUS,
        "Args/stage": "student",
        "Args/max_epochs": module.EXPECTED_MAX_EPOCHS,
        "Args/amp": False,
        "Args/teacher_task_id": TEACHER_TASK_ID,
        "Args/teacher_model_id": TEACHER_MODEL_ID,
        "Args/teacher_checkpoint_sha256": teacher_sha256,
        "Args/allow_failed_teacher_task": allow_failed_teacher_task,
    }


def _seal(module, document_type: str, payload: dict[str, object]):
    document = {
        "schema_version": 1,
        "document_type": document_type,
        **payload,
    }
    document["content_sha256"] = module._document_content_sha256(document)
    return document


def _condition_document(
    module,
    *,
    condition_name: str,
    teacher_sha256: str,
    student_sha256: str,
):
    metrics = {
        "resilient_v2x/sample_count": 1337,
        "resilient_v2x/car_ground_truth_count": 9000,
        "resilient_v2x/car_bev_ap_r40_0.50": 20.0,
        "resilient_v2x/car_bev_ap_r40_0.70": 10.0,
        "resilient_v2x/car_3d_ap_r40_0.50": 15.0,
        "resilient_v2x/car_3d_ap_r40_0.70": 5.0,
    }
    return _seal(
        module,
        module.CONDITION_RESULT_DOCUMENT_TYPE,
        {
            "task_id": VALIDATION_TASK_ID,
            "dataset_id": DATASET_ID,
            "runtime_profile": "rtx5090",
            "artifact_name": f"validation_{condition_name}",
            "condition_id": condition_name,
            "evaluation_sample_count": 1337,
            "checkpoints": {
                "teacher": {"sha256": teacher_sha256},
                "student": {"sha256": student_sha256},
            },
            "metrics": metrics,
        },
    )


def _write_summary(
    module,
    tmp_path: Path,
    *,
    teacher_sha256: str,
    student_sha256: str,
    missing_condition: str | None = None,
) -> tuple[Path, dict[str, _Artifact]]:
    conditions = {
        name: _condition_document(
            module,
            condition_name=name,
            teacher_sha256=teacher_sha256,
            student_sha256=student_sha256,
        )
        for name in module.EXPECTED_CONDITIONS
        if name != missing_condition
    }
    summary = _seal(
        module,
        module.VALIDATION_SUMMARY_DOCUMENT_TYPE,
        {
            "task_id": VALIDATION_TASK_ID,
            "dataset_id": DATASET_ID,
            "runtime_profile": "rtx5090",
            "evaluation_sample_count": 1337,
            "student_checkpoint": {
                "filename": "epoch_50.pth",
                "size_bytes": 7,
                "sha256": student_sha256,
            },
            "teacher_checkpoint": {
                "filename": "teacher_epoch_50.pth",
                "size_bytes": 7,
                "sha256": teacher_sha256,
            },
            "condition_count": len(conditions),
            "ground_truth_count": 9000,
            "conditions": conditions,
        },
    )
    path = tmp_path / "validation_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    artifacts = {
        "validation_summary": _Artifact(
            path,
            url=("http://10.100.34.118:8081/task/artifacts/validation_summary.json"),
        )
    }
    for condition_name in conditions:
        artifact_name = f"validation_{condition_name}"
        artifacts[artifact_name] = _Artifact(
            tmp_path,
            url=(f"http://10.100.34.118:8081/task/artifacts/{artifact_name}.zip"),
        )
    return path, artifacts


def _workflow(
    tmp_path: Path,
    *,
    initial_student_status: str = "in_progress",
    duplicate_student_model: bool = False,
    wrong_student_url: bool = False,
    missing_condition: str | None = None,
    validation_fails: bool = False,
    corrupt_draft: bool = False,
):
    module = _load_controller()
    events: list[str] = []
    teacher_checkpoint = tmp_path / "teacher_epoch_50.pth"
    teacher_checkpoint.write_bytes(b"teacher")
    student_checkpoint = tmp_path / "epoch_50.pth"
    student_checkpoint.write_bytes(b"student")
    teacher_sha256 = hashlib.sha256(b"teacher").hexdigest()
    student_sha256 = hashlib.sha256(b"student").hexdigest()

    teacher_model = _Model(
        name=module.CLEAN_TEACHER_MODEL_NAME,
        task=TEACHER_TASK_ID,
        model_id=TEACHER_MODEL_ID,
        url=("http://10.100.34.118:8081/models/teacher_epoch_50.pth"),
        local_copy=teacher_checkpoint,
    )
    teacher_task = _Task(
        task_id=TEACHER_TASK_ID,
        status="failed",
        parameters={},
        models=[teacher_model],
        artifacts={
            "run_contract": _Artifact(
                {
                    "task_id": TEACHER_TASK_ID,
                    "dataset_id": DATASET_ID,
                    "runtime_profile": "rtx5090",
                    "gpus": 4,
                    "max_epochs": 50,
                    "checkpoint_policy": "final_epoch",
                    "stage": "all",
                }
            )
        },
    )
    student_url = "http://10.100.34.118:8081/models/epoch_50.pth"
    if wrong_student_url:
        student_url = "http://10.100.35.118:8081/models/epoch_50.pth"
    student_model = _Model(
        name=module.DISTILLED_STUDENT_MODEL_NAME,
        task=STUDENT_TASK_ID,
        model_id=STUDENT_MODEL_ID,
        url=student_url,
        local_copy=student_checkpoint,
        events=events,
    )
    student_models = [student_model]
    if duplicate_student_model:
        student_models.append(
            _Model(
                name=module.DISTILLED_STUDENT_MODEL_NAME,
                task=STUDENT_TASK_ID,
                model_id="9" * 32,
                url=student_url,
                local_copy=student_checkpoint,
            )
        )

    student_parameters = _student_parameters(
        module,
        teacher_sha256=teacher_sha256,
        allow_failed_teacher_task=True,
    )
    student_artifacts = {
        "run_contract": _Artifact(
            {
                "task_id": STUDENT_TASK_ID,
                "dataset_id": DATASET_ID,
                "runtime_profile": "rtx5090",
                "gpus": 4,
                "global_batch_size": 8,
                "max_epochs": 50,
                "amp": False,
                "checkpoint_policy": "final_epoch",
                "evaluation_condition_count": 12,
                "stage": "student",
            }
        ),
        "checkpoint_handoff_contract": _Artifact(
            {
                "teacher": {
                    "task_id": TEACHER_TASK_ID,
                    "model_id": TEACHER_MODEL_ID,
                    "name": module.CLEAN_TEACHER_MODEL_NAME,
                    "url": ("http://10.100.34.118:8081/models/teacher_epoch_50.pth"),
                    "sha256": teacher_sha256,
                    "expected_sha256": teacher_sha256,
                },
                "failed_task_salvage": True,
            }
        ),
    }

    def finish_student(task: _Task) -> None:
        if task.status != "completed":
            events.append("student_completed")
            task.status = "completed"

    student_task = _Task(
        task_id=STUDENT_TASK_ID,
        status=initial_student_status,
        parameters=student_parameters,
        models=student_models,
        artifacts=student_artifacts,
        on_reload=finish_student,
    )
    tasks = {
        TEACHER_TASK_ID: teacher_task,
        STUDENT_TASK_ID: student_task,
    }

    class Tasks:
        draft: _Task | None = None
        enqueued_queue: str | None = None
        project_requests: list[str] = []

        @staticmethod
        def get_task(*, task_id: str):
            return tasks[task_id]

        @classmethod
        def get_project_id(
            cls,
            *,
            project_name: str,
            search_hidden: bool = True,
        ):
            assert search_hidden is True
            cls.project_requests.append(project_name)
            return PROJECT_ID

        @classmethod
        def clone(cls, **kwargs):
            assert kwargs["source_task"] is student_task
            assert kwargs["project"] == PROJECT_ID
            assert student_task.status == "completed"
            assert events[-1] == "student_model_downloaded"
            events.append("validation_cloned")

            def finish_validation(task: _Task) -> None:
                if task.status != "queued":
                    return
                if validation_fails:
                    task.status = "failed"
                    events.append("validation_failed")
                    return
                _, artifacts = _write_summary(
                    module,
                    tmp_path,
                    teacher_sha256=teacher_sha256,
                    student_sha256=student_sha256,
                    missing_condition=missing_condition,
                )
                task.artifacts = artifacts
                task.status = "completed"
                events.append("validation_completed")

            cls.draft = _Task(
                task_id=VALIDATION_TASK_ID,
                status="created",
                parameters=student_parameters,
                models=[],
                artifacts={},
                on_reload=finish_validation,
            )
            if corrupt_draft:
                original = cls.draft.set_parameter

                def corrupt(*, name: str, value: object) -> None:
                    original(name=name, value=value)
                    if name == "Args/stage":
                        cls.draft.parameters[name] = "student"

                cls.draft.set_parameter = corrupt
            return cls.draft

        @classmethod
        def enqueue(cls, *, task, queue_name: str, force: bool):
            assert task is cls.draft
            assert task.status == "created"
            assert force is False
            cls.enqueued_queue = queue_name
            events.append("validation_enqueued")
            task.status = "queued"
            return {"queued": 1, "updated": 1}

    return (
        module,
        Tasks,
        events,
        teacher_sha256,
        student_sha256,
    )


def _run(module, tasks, teacher_sha256: str, **kwargs):
    return module.run_paper_controller(
        tasks,
        student_task_id=STUDENT_TASK_ID,
        teacher_task_id=TEACHER_TASK_ID,
        teacher_model_id=TEACHER_MODEL_ID,
        teacher_checkpoint_sha256=teacher_sha256,
        expected_student_model_id=STUDENT_MODEL_ID,
        allow_failed_teacher_task=True,
        poll_seconds=0.01,
        timeout_seconds=1.0,
        sleep_fn=lambda _seconds: None,
        **kwargs,
    )


def test_controller_runs_only_after_student_completion_and_verifies_results(
    tmp_path: Path,
) -> None:
    module, tasks, events, teacher_sha256, student_sha256 = _workflow(tmp_path)
    result = _run(module, tasks, teacher_sha256)

    assert events == [
        "student_completed",
        "student_model_downloaded",
        "validation_cloned",
        "validation_enqueued",
        "validation_completed",
    ]
    assert tasks.enqueued_queue == module.WORKER_QUEUE == "GPU4-5090"
    assert tasks.project_requests == ["ResilientV2X/Training"]
    assert result.validation_task_id == VALIDATION_TASK_ID
    assert result.student_model_id == STUDENT_MODEL_ID
    assert result.student_checkpoint_sha256 == student_sha256
    assert result.condition_count == 12
    assert tasks.draft is not None
    parameters = tasks.draft.parameters
    assert parameters["Args/stage"] == "validate"
    assert parameters["Args/gpus"] == 4
    assert parameters["Args/teacher_task_id"] == TEACHER_TASK_ID
    assert parameters["Args/teacher_model_id"] == TEACHER_MODEL_ID
    assert parameters["Args/teacher_checkpoint_sha256"] == teacher_sha256
    assert parameters["Args/student_task_id"] == STUDENT_TASK_ID
    assert parameters["Args/student_model_id"] == STUDENT_MODEL_ID
    assert parameters["Args/student_checkpoint_sha256"] == student_sha256


@pytest.mark.parametrize("failure", ["duplicate", "url"])
def test_invalid_student_model_is_rejected_before_clone(
    tmp_path: Path,
    failure: str,
) -> None:
    workflow = _workflow(
        tmp_path,
        duplicate_student_model=failure == "duplicate",
        wrong_student_url=failure == "url",
    )
    module, tasks, events, teacher_sha256, _ = workflow
    pattern = "exactly one" if failure == "duplicate" else "trusted files server"
    with pytest.raises(RuntimeError, match=pattern):
        _run(module, tasks, teacher_sha256)
    assert "validation_cloned" not in events
    assert tasks.enqueued_queue is None


def test_expected_student_sha_is_checked_before_clone(tmp_path: Path) -> None:
    module, tasks, events, teacher_sha256, _ = _workflow(tmp_path)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _run(
            module,
            tasks,
            teacher_sha256,
            expected_student_checkpoint_sha256="0" * 64,
        )
    assert "validation_cloned" not in events


def test_draft_is_rechecked_before_enqueue(tmp_path: Path) -> None:
    module, tasks, events, teacher_sha256, _ = _workflow(
        tmp_path,
        corrupt_draft=True,
    )
    with pytest.raises(RuntimeError, match="Args/stage mismatch"):
        _run(module, tasks, teacher_sha256)
    assert "validation_cloned" in events
    assert "validation_enqueued" not in events


def test_validation_must_complete_and_contain_exactly_twelve_conditions(
    tmp_path: Path,
) -> None:
    module, tasks, _, teacher_sha256, _ = _workflow(
        tmp_path,
        validation_fails=True,
    )
    with pytest.raises(RuntimeError, match="ended without completion"):
        _run(module, tasks, teacher_sha256)

    module, tasks, _, teacher_sha256, _ = _workflow(
        tmp_path,
        missing_condition="delay_300_c_fail",
    )
    with pytest.raises(RuntimeError, match="condition_count mismatch"):
        _run(module, tasks, teacher_sha256)


def test_cli_defaults_worker_queue_to_gpu4_and_allows_override() -> None:
    module = _load_controller()
    destinations = {action.dest for action in module._parser()._actions}
    assert "worker_queue" in destinations
    assert module.WORKER_QUEUE == "GPU4-5090"
    args = module._parser().parse_args(
        [
            "--student-task-id",
            "a" * 32,
            "--teacher-task-id",
            "b" * 32,
            "--teacher-model-id",
            "c" * 32,
            "--teacher-checkpoint-sha256",
            "d" * 64,
        ]
    )
    assert args.worker_queue == "GPU4-5090"
    args_gpu8 = module._parser().parse_args(
        [
            "--student-task-id",
            "a" * 32,
            "--teacher-task-id",
            "b" * 32,
            "--teacher-model-id",
            "c" * 32,
            "--teacher-checkpoint-sha256",
            "d" * 64,
            "--worker-queue",
            "GPU8-5090",
        ]
    )
    assert args_gpu8.worker_queue == "GPU8-5090"


def test_bootstrap_entry_point_accepts_only_reviewed_path_or_basename() -> None:
    module = _load_controller()
    for entry_point in (
        "clearml_5090_bootstrap.py",
        "tools/resilient_v2x/clearml_5090_bootstrap.py",
        "/workspace/tools/resilient_v2x/clearml_5090_bootstrap.py",
    ):
        task = SimpleNamespace(
            data=SimpleNamespace(script=SimpleNamespace(entry_point=entry_point))
        )
        assert (
            module._require_bootstrap_entry_point(
                task,
                context="test task",
            )
            == entry_point
        )

    wrong = SimpleNamespace(
        data=SimpleNamespace(script=SimpleNamespace(entry_point="other_bootstrap.py"))
    )
    with pytest.raises(RuntimeError, match="sealed RTX5090 bootstrap"):
        module._require_bootstrap_entry_point(wrong, context="test task")


def test_main_publishes_durable_controller_handoff(
    monkeypatch,
    capsys,
) -> None:
    module = _load_controller()
    result = module.ControllerResult(
        validation_task_id=VALIDATION_TASK_ID,
        student_model_id=STUDENT_MODEL_ID,
        student_checkpoint_sha256="5" * 64,
        validation_summary_sha256="6" * 64,
        condition_count=12,
    )

    class CurrentController:
        id = "7" * 32
        output_uri = None

        def __init__(self) -> None:
            self.parameters: dict[str, object] = {}
            self.summary: dict[str, object] | None = None
            self.flush_calls = 0

        def set_parameter(self, *, name: str, value: object) -> None:
            self.parameters[name] = value

        def upload_artifact(
            self,
            name: str,
            *,
            artifact_object: str,
            wait_on_upload: bool,
        ) -> bool:
            assert name == "paper_controller_summary"
            assert wait_on_upload is True
            self.summary = json.loads(Path(artifact_object).read_text(encoding="utf-8"))
            return True

        def flush(self, *, wait_for_uploads: bool) -> None:
            assert wait_for_uploads is True
            self.flush_calls += 1

    current = CurrentController()

    class Tasks:
        @staticmethod
        def current_task():
            return current

    observed: dict[str, object] = {}

    def fake_run(task_class, **kwargs):
        observed["task_class"] = task_class
        observed.update(kwargs)
        return result

    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=Tasks))
    monkeypatch.setattr(module, "run_paper_controller", fake_run)
    assert (
        module.main(
            [
                "--student-task-id",
                STUDENT_TASK_ID,
                "--teacher-task-id",
                TEACHER_TASK_ID,
                "--teacher-model-id",
                TEACHER_MODEL_ID,
                "--teacher-checkpoint-sha256",
                "8" * 64,
            ]
        )
        == 0
    )

    assert observed["task_class"] is Tasks
    assert "controller_task" not in observed
    assert current.parameters["Controller/validation_task_id"] == VALIDATION_TASK_ID
    assert current.output_uri == module.FILES_SERVER_URI
    assert current.flush_calls == 2
    assert current.summary is not None
    summary = module._require_sealed_document(
        current.summary,
        expected_type=module.PAPER_CONTROLLER_SUMMARY_DOCUMENT_TYPE,
        context="paper controller summary",
    )
    assert summary["controller_task_id"] == current.id
    assert summary["student_task_id"] == STUDENT_TASK_ID
    assert summary["validation_task_id"] == VALIDATION_TASK_ID
    assert summary["student_model"]["model_id"] == STUDENT_MODEL_ID
    assert summary["condition_count"] == 12
    printed = json.loads(capsys.readouterr().out)
    assert printed["event"] == "resilient_v2x_paper_validation_complete"


def test_remote_controller_connects_before_argparse() -> None:
    module = _load_controller()
    current = object()
    calls: list[dict[str, object]] = []

    class Tasks:
        @staticmethod
        def current_task():
            return None

        @staticmethod
        def init(**kwargs):
            calls.append(dict(kwargs))
            return current

    assert module._connect_controller_task(Tasks) is current
    assert calls == [
        {
            "project_name": "ResilientV2X/Training",
            "task_name": "ResilientV2X paper validation controller",
            "reuse_last_task_id": False,
            "output_uri": module.FILES_SERVER_URI,
            "auto_connect_arg_parser": True,
        }
    ]
