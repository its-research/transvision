from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import clearml_formal_candidate_evaluation_queue as queue


@pytest.fixture(autouse=True)
def _formal_core_gate_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        queue,
        "_formal_core_phase_snapshot",
        lambda _task_class: {
            "policy": "formal_core_six_completed_before_candidate_e1",
            "ready": True,
            "entries": [],
        },
    )


class Artifact:
    def __init__(
        self,
        name: str,
        value: object,
        *,
        digest: str | None = None,
        size: int = 100,
        path: Path | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.path = path
        preview = json.dumps(value) if isinstance(value, dict) else str(value)
        suffix = ".zip" if path is not None else ".json"
        self.record = {
            "key": name,
            "hash": digest or hashlib.sha256(preview.encode()).hexdigest(),
            "content_size": path.stat().st_size if path is not None else size,
            "uri": f"http://{queue.FILES_SERVER_HOST}:{queue.FILES_SERVER_PORT}/artifacts/{name}{suffix}",
            "type_data": {"preview": preview},
        }

    def get(self) -> object:
        return self.value

    def get_local_copy(self, **_kwargs: object) -> str:
        assert self.path is not None
        return str(self.path)


class Record:
    def __init__(self, artifact: Artifact) -> None:
        self.artifact = artifact

    def to_dict(self) -> dict[str, object]:
        return dict(self.artifact.record)


class FileArtifact(Artifact):
    def get(self) -> object:
        assert self.path is not None
        return str(self.path)


class Model:
    def __init__(self, model_id: str, name: str, task: str, url: str) -> None:
        self.id = model_id
        self.name = name
        self.task = task
        self.url = url


class FakeTask:
    def __init__(
        self,
        task_id: str,
        *,
        name: str,
        status: str,
        parent: str,
        parameters: dict[str, object],
        diff: str,
        models: dict[str, list[Model]] | None = None,
        artifacts: dict[str, Artifact] | None = None,
        queue_id: str = "",
        last_worker: str = "",
    ) -> None:
        self.id = task_id
        self.name = name
        self.status = status
        self.parent = parent
        self.project = queue.PROJECT_ID
        self.parameters = dict(parameters)
        self._models = models or {"input": [], "output": []}
        self.artifacts = artifacts or {}
        self.data = SimpleNamespace(
            parent=parent,
            project=queue.PROJECT_ID,
            script={
                "repository": "",
                "working_dir": ".",
                "entry_point": queue.TEMPLATE_ENTRY_POINT,
                "diff": diff,
            },
            execution=SimpleNamespace(
                artifacts=[Record(value) for value in self.artifacts.values()],
                queue=queue_id,
            ),
            last_worker=last_worker,
        )

    def reload(self) -> None:
        self.data.execution.artifacts = [
            Record(value) for value in self.artifacts.values()
        ]

    def get_parameters(self, **_kwargs: object) -> dict[str, object]:
        return dict(self.parameters)

    def set_parameters(self, value: dict[str, object]) -> None:
        self.parameters = dict(value)

    def get_models(self) -> dict[str, list[Model]]:
        return self._models

    def remove_input_models(self, ids: list[str]) -> None:
        self._models["input"] = [
            model for model in self._models["input"] if model.id not in ids
        ]

    def set_input_model(self, model_id: str, name: str, **_kwargs: object) -> None:
        self._models["input"] = [Model(model_id, name, "", "")]

    def get_project_name(self) -> str:
        return queue.PROJECT_NAME


class FakeTaskClass:
    tasks: dict[str, FakeTask]
    clone_calls: int

    @classmethod
    def reset(cls, tasks: list[FakeTask]) -> None:
        cls.tasks = {task.id: task for task in tasks}
        cls.clone_calls = 0

    @classmethod
    def get_task(cls, *, task_id: str) -> FakeTask:
        return cls.tasks[task_id]

    @classmethod
    def get_tasks(cls, **_kwargs: object) -> list[FakeTask]:
        parent = (_kwargs.get("task_filter") or {}).get("parent")
        return [task for task in cls.tasks.values() if task.parent == parent]

    @classmethod
    def query_tasks(cls, **_kwargs: object) -> list[str]:
        task_filter = _kwargs.get("task_filter") or {}
        statuses = set(task_filter.get("status") or [])
        parents = task_filter.get("parent")
        parent_values = (
            {str(value) for value in parents}
            if isinstance(parents, (list, tuple, set))
            else ({str(parents)} if parents else set())
        )
        projects = {str(value) for value in task_filter.get("project") or []}
        return [
            task.id
            for task in cls.tasks.values()
            if (not statuses or task.status in statuses)
            and (not parent_values or task.parent in parent_values)
            and (not projects or task.project in projects)
        ]

    @classmethod
    def clone(
        cls, *, source_task: FakeTask, name: str, parent: str
    ) -> FakeTask:
        cls.clone_calls += 1
        task_id = f"{9000 + cls.clone_calls:032x}"
        task = FakeTask(
            task_id,
            name=name,
            status="created",
            parent=parent,
            parameters=source_task.parameters,
            diff=source_task.data.script["diff"],
        )
        cls.tasks[task_id] = task
        return task

    @classmethod
    def enqueue(cls, *, task: FakeTask, queue_name: str) -> bool:
        task.status = "queued"
        task.data.execution.queue = queue.QUEUE_IDS[queue_name]
        return True


class StaleEnqueueTaskClass(FakeTaskClass):
    @classmethod
    def enqueue(cls, *, task: FakeTask, queue_name: str) -> bool:
        stale = FakeTask(
            task.id,
            name=task.name,
            status="created",
            parent=task.parent,
            parameters=task.parameters,
            diff=task.data.script["diff"],
            models=task.get_models(),
        )
        stale.status = "queued"
        stale.data.execution.queue = queue.QUEUE_IDS[queue_name]
        cls.tasks[task.id] = stale
        return True


def fake_template_diff() -> str:
    return (
        "import hashlib\nimport os\nfrom pathlib import Path\n"
        "def _capture_gpu_runtime() -> dict[str, object]:\n"
        "    import torch\n"
        "    return {}\n\n"
        "def _apply_controlled_evaluator_headless_compatibility(source_root):\n"
        "    return source_root\n\n"
        "def _execute_controlled_baseline_validation(\n"
        "    args, *, source_root, python, runtime_env, dataset_class, task_class\n"
        "):\n"
        "    baseline = str(args.controlled_baseline)\n"
        "    evaluator = _apply_controlled_evaluator_headless_compatibility(source_root)\n"
        "    command = [\n"
        "        str(python), str(evaluator), baseline\n"
        "    ]\n"
    )


def template_parameters(spec: queue.CandidateSpec) -> dict[str, object]:
    return {
        "Args/source_dataset_id": spec.source_dataset_id,
        "Args/source_archive_name": spec.source_archive_name,
        "Args/source_archive_bytes": str(spec.source_archive_bytes),
        "Args/source_archive_sha256": spec.source_archive_sha256,
        "Args/training_dataset_id": queue.TRAINING_DATASET_ID,
        "Args/native_bundle_sha256": queue.NATIVE_BUNDLE_SHA256,
        "Args/build_manifest_sha256": queue.BUILD_MANIFEST_SHA256,
        "Args/gpus": "4",
        "Args/max_epochs": "50",
        "Args/amp": "False",
        "Args/stage": "teacher",
        "Args/experiment_from_task": "",
        "Args/teacher_task_id": "",
        "Args/teacher_model_id": "",
        "Args/teacher_checkpoint_sha256": "",
    }


def training_parameters(spec: queue.CandidateSpec) -> dict[str, object]:
    return {
        **template_parameters(spec),
        "Args/stage": "all",
        "Args/experiment_from_task": spec.subject,
        "Args/predecessor_task_id": queue.PREDECESSOR_TASK_ID,
        "Args/teacher_task_id": queue.TEACHER_TASK_ID,
        "Args/teacher_model_id": queue.TEACHER_MODEL_ID,
        "Args/teacher_checkpoint_sha256": queue.TEACHER_CHECKPOINT_SHA256,
        "Args/training_seed": str(queue.TRAINING_SEED),
    }


def run_contract(spec: queue.CandidateSpec) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": spec.training_task_id,
        "experiment": spec.subject,
        "source_dataset_id": spec.source_dataset_id,
        "training_dataset_id": queue.TRAINING_DATASET_ID,
        "predecessor_task_id": queue.PREDECESSOR_TASK_ID,
        "gpus": 4,
        "global_batch_size": 8,
        "max_epochs": 50,
        "val_interval": 10,
        "precision": "FP32",
        "amp": False,
        "seed": queue.TRAINING_SEED,
        "native_bundle_sha256": queue.NATIVE_BUNDLE_SHA256,
        "build_manifest_sha256": queue.BUILD_MANIFEST_SHA256,
        "source_archive": {
            "name": spec.source_archive_name,
            "sha256": spec.source_archive_sha256,
            "size_bytes": spec.source_archive_bytes,
        },
        "config": {
            "declared": spec.config_path,
            "config_sha256": spec.config_sha256,
            "resolved_config_sha256": spec.config_sha256,
        },
        "teacher": {
            "task_id": queue.TEACHER_TASK_ID,
            "model_id": queue.TEACHER_MODEL_ID,
            "sha256": queue.TEACHER_CHECKPOINT_SHA256,
            "expected_sha256": queue.TEACHER_CHECKPOINT_SHA256,
        },
        "common_teacher_initialization": {
            "audit_artifact_name": "common_teacher_initialization_audit",
            "contract": "shared-only-clean-teacher-initialization-v1",
            "expected_nested_teacher": True,
            "teacher_checkpoint_sha256": queue.TEACHER_CHECKPOINT_SHA256,
        },
    }


def launch_receipt(spec: queue.CandidateSpec) -> dict[str, object]:
    common: dict[str, object] = {
        "source_dataset_id": spec.source_dataset_id,
        "template_task_id": spec.template_task_id,
    }
    if spec.label.startswith("E"):
        common["tasks"] = [
            {
                "label": spec.label,
                "experiment": spec.subject,
                "config": spec.config_path,
                "config_sha256": spec.config_sha256,
                "task_id": spec.training_task_id,
                "parent_task_id": queue.PREDECESSOR_TASK_ID,
                "training_predecessor_task_id": queue.PREDECESSOR_TASK_ID,
                "script_sha256": spec.training_script_sha256,
            }
        ]
    else:
        common.update(
            {
                "task_id": spec.training_task_id,
                "parent_task_id": queue.PREDECESSOR_TASK_ID,
                "patched_script_sha256": spec.training_script_sha256,
                "teacher_task_id": queue.TEACHER_TASK_ID,
                "teacher_model_id": queue.TEACHER_MODEL_ID,
                "teacher_checkpoint_sha256": queue.TEACHER_CHECKPOINT_SHA256,
            }
        )
    return queue._sealed(common)


def world(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
    final_filename: str = "epoch_50.pth",
) -> tuple[queue.CandidateSpec, FakeTask, FakeTask]:
    diff = fake_template_diff()
    spec = replace(
        queue.CANDIDATES[0],
        training_script_sha256=hashlib.sha256(b"training diff").hexdigest(),
    )
    receipt = launch_receipt(spec)
    spec = replace(spec, launch_receipt_seal_sha256=receipt["seal_sha256"])
    monkeypatch.setattr(queue, "CANDIDATES", (spec,))
    monkeypatch.setattr(queue, "CANDIDATE_ORDER", (spec.subject,))
    monkeypatch.setattr(
        queue,
        "TEMPLATE_SCRIPT_SHA256",
        hashlib.sha256(diff.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        queue,
        "_live_worker_rows",
        lambda: [
            {
                "id": f"10.0.0.{index}-{name}:gpu4,5,6,7",
                "queues": [{"id": queue_id}],
            }
            for index, (name, queue_id) in enumerate(
                (
                    ("A100", queue.QUEUE_IDS["GPU4-A100"]),
                    ("V100", queue.QUEUE_IDS["GPU4-V100"]),
                    ("5090", queue.QUEUE_IDS["GPU4-5090"]),
                ),
                start=1,
            )
        ],
    )
    transition = Artifact(
        spec.source_transition_artifact,
        {"sealed": True},
        digest=spec.source_transition_artifact_sha256,
        size=spec.source_transition_artifact_bytes,
    )
    template = FakeTask(
        spec.template_task_id,
        name="template",
        status="completed",
        parent=queue.TEMPLATE_PARENT_TASK_ID,
        parameters=template_parameters(spec),
        diff=diff,
        artifacts={spec.source_transition_artifact: transition},
    )
    artifacts = {
        "run_contract": Artifact("run_contract", run_contract(spec)),
        spec.launch_receipt_artifact: Artifact(spec.launch_receipt_artifact, receipt),
    }
    outputs: list[Model] = []
    if status == "completed":
        model_id = "2" * 32
        model_name = f"ResilientV2X {spec.subject} final checkpoint"
        model_url = (
            f"http://{queue.FILES_SERVER_HOST}:{queue.FILES_SERVER_PORT}/models/"
            f"{spec.subject}_epoch_50.pth"
        )
        outputs.append(Model(model_id, model_name, spec.training_task_id, model_url))
        artifacts.update(
            {
                "common_teacher_initialization_audit": Artifact(
                    "common_teacher_initialization_audit", "audit"
                ),
                "final_checkpoint_contract": Artifact(
                    "final_checkpoint_contract",
                    {
                        "filename": final_filename,
                        "model_id": model_id,
                        "name": model_name,
                        "url": model_url,
                        "sha256": "3" * 64,
                        "size_bytes": 1234,
                    },
                ),
            }
        )
    training = FakeTask(
        spec.training_task_id,
        name="training",
        status=status,
        parent=queue.PREDECESSOR_TASK_ID,
        parameters=training_parameters(spec),
        diff="training diff",
        models={"input": [], "output": outputs},
        artifacts=artifacts,
    )
    receipt = launch_receipt(spec)
    training.artifacts[spec.launch_receipt_artifact] = Artifact(
        spec.launch_receipt_artifact, receipt
    )
    training.reload()
    FakeTaskClass.reset([template, training])
    return spec, template, training


def script_editor(task_id: str, diff: str) -> None:
    FakeTaskClass.tasks[task_id].data.script["diff"] = diff


def append_completed_candidate_world(
    monkeypatch: pytest.MonkeyPatch, first: queue.CandidateSpec
) -> queue.CandidateSpec:
    second = replace(
        first,
        label="E2",
        subject="second_candidate",
        training_task_id="4" * 32,
        template_task_id="5" * 32,
        queue="GPU4-V100",
        config_path="configs/resilient_v2x/improvements/second_candidate.py",
        config_sha256="6" * 64,
    )
    receipt = launch_receipt(second)
    second = replace(second, launch_receipt_seal_sha256=receipt["seal_sha256"])
    transition = Artifact(
        second.source_transition_artifact,
        {"sealed": True},
        digest=second.source_transition_artifact_sha256,
        size=second.source_transition_artifact_bytes,
    )
    template = FakeTask(
        second.template_task_id,
        name="second template",
        status="completed",
        parent=queue.TEMPLATE_PARENT_TASK_ID,
        parameters=template_parameters(second),
        diff=fake_template_diff(),
        artifacts={second.source_transition_artifact: transition},
    )
    model_id = "7" * 32
    model_name = f"ResilientV2X {second.subject} final checkpoint"
    model_url = (
        f"http://{queue.FILES_SERVER_HOST}:{queue.FILES_SERVER_PORT}/models/"
        f"{second.subject}_epoch_50.pth"
    )
    training = FakeTask(
        second.training_task_id,
        name="second training",
        status="completed",
        parent=queue.PREDECESSOR_TASK_ID,
        parameters=training_parameters(second),
        diff="training diff",
        models={
            "input": [],
            "output": [
                Model(model_id, model_name, second.training_task_id, model_url)
            ],
        },
        artifacts={
            "run_contract": Artifact("run_contract", run_contract(second)),
            second.launch_receipt_artifact: Artifact(
                second.launch_receipt_artifact, receipt
            ),
            "common_teacher_initialization_audit": Artifact(
                "common_teacher_initialization_audit", "audit"
            ),
            "final_checkpoint_contract": Artifact(
                "final_checkpoint_contract",
                {
                    "filename": "epoch_50.pth",
                    "model_id": model_id,
                    "name": model_name,
                    "url": model_url,
                    "sha256": "8" * 64,
                    "size_bytes": 2345,
                },
            ),
        },
    )
    FakeTaskClass.tasks[template.id] = template
    FakeTaskClass.tasks[training.id] = training
    monkeypatch.setattr(queue, "CANDIDATES", (first, second))
    monkeypatch.setattr(queue, "CANDIDATE_ORDER", (first.subject, second.subject))
    return second


def test_incomplete_training_stays_pending_and_does_not_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world(monkeypatch, status="in_progress")
    manifest = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    assert FakeTaskClass.clone_calls == 0
    assert manifest["entries"][0]["evidence_status"] == "pending_training"


def test_formal_core_barrier_prevents_candidate_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world(monkeypatch, status="completed")
    monkeypatch.setattr(
        queue,
        "_formal_core_phase_snapshot",
        lambda _task_class: {
            "policy": "formal_core_six_completed_before_candidate_e1",
            "ready": False,
            "entries": [
                {
                    "subject": "ffnet",
                    "task_id": queue.FORMAL_CORE_EVALUATIONS[0]["task_id"],
                    "status": "created",
                }
            ],
        },
    )
    manifest = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    assert FakeTaskClass.clone_calls == 0
    assert manifest["formal_core_release_barrier"]["ready"] is False
    assert manifest["entries"][0]["release_gate"] == "waiting_for_formal_core"


def test_completed_training_creates_binds_and_enqueues_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _, _ = world(monkeypatch, status="completed")
    first = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    second = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    assert FakeTaskClass.clone_calls == 1
    assert first["entries"][0]["evaluation_status"] == "queued"
    assert second["entries"][0]["evaluation_task_id"] == first["entries"][0][
        "evaluation_task_id"
    ]
    evaluation = FakeTaskClass.tasks[first["entries"][0]["evaluation_task_id"]]
    assert [model.id for model in evaluation.get_models()["input"]] == ["2" * 32]
    assert evaluation.data.execution.queue == queue.QUEUE_IDS[spec.queue]


def test_enqueue_reloads_authoritative_task_before_queue_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world(monkeypatch, status="completed")
    StaleEnqueueTaskClass.tasks = FakeTaskClass.tasks
    StaleEnqueueTaskClass.clone_calls = FakeTaskClass.clone_calls
    manifest = queue.reconcile_once(
        task_class=StaleEnqueueTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    assert manifest["entries"][0]["evaluation_status"] == "queued"


def test_running_evaluation_accepts_only_known_materialized_empty_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, template, training = world(monkeypatch, status="completed")
    binding = queue._validate_completed_training(training, spec)
    manifest = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    evaluation = FakeTaskClass.tasks[manifest["entries"][0]["evaluation_task_id"]]
    evaluation.status = "in_progress"
    evaluation.parameters.update(queue._RUNTIME_EMPTY_EVALUATION_PARAMETERS)
    assert queue._validate_evaluation_identity(
        evaluation, template, spec, binding
    ) == "in_progress"

    evaluation.parameters["Args/teacher_task_id"] = "unexpected"
    with pytest.raises(RuntimeError, match="teacher_task_id"):
        queue._validate_evaluation_identity(evaluation, template, spec, binding)


def test_candidate_release_waits_for_every_training_final_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _, _ = world(monkeypatch, status="completed")
    second = append_completed_candidate_world(monkeypatch, first)
    FakeTaskClass.tasks[second.training_task_id].status = "in_progress"

    manifest = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )

    assert FakeTaskClass.clone_calls == 0
    assert manifest["training_release_barrier"] == {
        "policy": "all_five_completed_and_final_contract_verified",
        "ready": False,
        "required_labels": [first.label, second.label],
    }
    assert manifest["entries"][0]["release_gate"] == (
        "waiting_for_all_training_final_contracts"
    )


def test_candidates_release_in_parallel_when_disjoint_workers_are_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _, _ = world(monkeypatch, status="completed")
    second = append_completed_candidate_world(monkeypatch, first)
    initial = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    assert FakeTaskClass.clone_calls == 2
    assert initial["entries"][0]["evaluation_status"] == "queued"
    assert initial["entries"][1]["evaluation_status"] == "queued"

    waiting = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    assert FakeTaskClass.clone_calls == 2
    assert waiting["entries"][1]["release_gate"] == "evaluation_active"

    first_evaluation = FakeTaskClass.tasks[
        str(initial["entries"][0]["evaluation_task_id"])
    ]
    first_evaluation.status = "completed"
    monkeypatch.setattr(
        queue,
        "_validate_completed_evaluation",
        lambda _task, _spec, _binding: {"runs": []},
    )
    continued = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    assert FakeTaskClass.clone_calls == 2
    assert continued["entries"][0]["evidence_status"] == "validated"
    assert continued["entries"][1]["subject"] == second.subject
    assert continued["entries"][1]["evaluation_status"] == "queued"


def test_external_target_queue_work_blocks_release_until_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _, _ = world(monkeypatch, status="completed")
    selected_queue_id = queue.QUEUE_IDS[spec.queue]
    external = FakeTask(
        "a" * 32,
        name="external training",
        status="queued",
        parent="b" * 32,
        parameters={},
        diff="external",
        queue_id=selected_queue_id,
    )
    FakeTaskClass.tasks[external.id] = external
    blocked = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    assert FakeTaskClass.clone_calls == 1
    assert blocked["entries"][0]["evaluation_status"] == "created"
    assert blocked["entries"][0]["release_gate"] == "waiting_for_gpu_resources"
    assert blocked["entries"][0]["resource_gate"]["external_blockers"] == [
        {
            "task_id": external.id,
            "status": "queued",
            "queue_id": selected_queue_id,
            "last_worker": None,
            "reasons": ["target_queue_waiting_task"],
        }
    ]

    external.status = "completed"
    released = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    assert FakeTaskClass.clone_calls == 1
    assert released["entries"][0]["evaluation_status"] == "queued"


def test_candidate_same_queue_gpu0_3_does_not_block_idle_gpu4_7() -> None:
    queue_id = queue.QUEUE_IDS["GPU4-5090"]
    external = FakeTask(
        "c" * 32,
        name="external disjoint training",
        status="in_progress",
        parent="d" * 32,
        parameters={},
        diff="external",
        queue_id=queue_id,
        last_worker="10.0.0.9-5090:gpu0,1,2,3",
    )
    FakeTaskClass.reset([external])

    snapshot = queue._resource_gate_snapshot(
        FakeTaskClass,
        candidate_evaluation_task_ids=set(),
        target_queue_id=queue_id,
        worker_rows=[
            {
                "id": "10.0.0.9-5090:gpu0,1,2,3",
                "queues": [{"id": queue_id}],
            },
            {
                "id": "10.0.0.9-5090:gpu4,5,6,7",
                "queues": [{"id": queue_id}],
            },
        ],
    )

    assert snapshot["ready"] is True
    assert snapshot["selected_idle_target_worker_id"] == (
        "10.0.0.9-5090:gpu4,5,6,7"
    )
    assert snapshot["external_blockers"] == []


def test_candidate_overlapping_active_gpu_blocks_without_idle_worker() -> None:
    queue_id = queue.QUEUE_IDS["GPU4-5090"]
    external = FakeTask(
        "e" * 32,
        name="external overlapping training",
        status="in_progress",
        parent="f" * 32,
        parameters={},
        diff="external",
        queue_id=queue_id,
        last_worker="10.0.0.9-5090:gpu4,5,6,7",
    )
    FakeTaskClass.reset([external])

    snapshot = queue._resource_gate_snapshot(
        FakeTaskClass,
        candidate_evaluation_task_ids=set(),
        target_queue_id=queue_id,
        worker_rows=[
            {
                "id": "10.0.0.9-5090:gpu4,5,6,7",
                "queues": [{"id": queue_id}],
            },
        ],
    )

    assert snapshot["ready"] is False
    assert snapshot["selected_idle_target_worker_id"] is None
    assert snapshot["external_blockers"][0]["reasons"] == [
        "overlapping_active_worker"
    ]


def test_queue_occupancy_api_unavailable_fails_closed_before_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world(monkeypatch, status="completed")
    monkeypatch.setattr(FakeTaskClass, "query_tasks", None)
    with pytest.raises(RuntimeError, match="authoritatively inspect queue occupancy"):
        queue.reconcile_once(
            task_class=FakeTaskClass,
            controller_task=None,
            execute=True,
            script_editor=script_editor,
            sleeper=lambda _seconds: None,
        )
    assert FakeTaskClass.clone_calls == 0
    assert not [
        task
        for task in FakeTaskClass.tasks.values()
        if task.name.startswith("ResilientV2X formal1337 candidate eval")
    ]


def test_clean_best_cannot_satisfy_final_checkpoint_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _, training = world(
        monkeypatch,
        status="completed",
        final_filename="clean_val_best_epoch_30.pth",
    )
    with pytest.raises(RuntimeError, match="epoch_50"):
        queue._validate_completed_training(training, spec)


def _evidence_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[queue.CandidateSpec, dict[str, object], FakeTask]:
    spec, template, training = world(monkeypatch, status="completed")
    sample_ids = ["sample-a", "sample-b"]
    monkeypatch.setattr(queue, "SAMPLE_COUNT", 2)
    monkeypatch.setattr(queue, "GROUND_TRUTH_COUNT", 3)
    monkeypatch.setattr(
        queue,
        "SAMPLE_IDS_SHA256",
        hashlib.sha256(queue._canonical_json(sample_ids).encode()).hexdigest(),
    )
    monkeypatch.setattr(
        queue,
        "COUNT_METRICS",
        {
            "resilient_v2x/sample_count": 2,
            "resilient_v2x/car_ground_truth_count": 3,
            "resilient_v2x/unsupported_sample_count": 0,
        },
    )
    binding = queue._validate_completed_training(training, spec)
    runs = []
    plan_runs = []
    archive_path = tmp_path / "evidence.zip"
    files: dict[str, bytes] = {
        "evaluation_plan.json": b"{}",
        "metrics.json": b"{}",
    }
    for delay in queue.DELAYS_MS:
        for condition in queue.CONDITIONS:
            condition_id = f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
            prediction = {
                "sample_count": 2,
                "samples": [
                    {
                        "sample_id": "sample-a",
                        "ground_truth_boxes_lidar_bottom_center": [[0], [1]],
                        "ground_truth_labels": [0, 0],
                    },
                    {
                        "sample_id": "sample-b",
                        "ground_truth_boxes_lidar_bottom_center": [[2]],
                        "ground_truth_labels": [0],
                    },
                ],
            }
            prediction["content_sha256"] = hashlib.sha256(
                queue._canonical_json(prediction).encode()
            ).hexdigest()
            raw = queue._canonical_json(prediction).encode()
            prediction_sha = hashlib.sha256(raw).hexdigest()
            files[f"{condition_id}/predictions.json"] = raw
            files[f"{condition_id}/resolved_config.py"] = b"model = {}\n"
            files[f"{condition_id}/checkpoint.sha256"] = (
                f"{binding['checkpoint_sha256']}\n".encode()
            )
            run_metrics = {
                **{key: 50.0 for key in queue.AP_METRIC_KEYS},
                **queue.COUNT_METRICS,
            }
            runs.append(
                {
                    "condition_id": condition_id,
                    "delay_ms": delay,
                    "condition": condition,
                    "metrics": run_metrics,
                    "predictions": f"/{condition_id}/predictions.json",
                    "prediction_sha256": prediction_sha,
                    "prediction_content_sha256": prediction["content_sha256"],
                    "sample_count": 2,
                    "sample_ids_sha256": queue.SAMPLE_IDS_SHA256,
                    "ground_truth_count": 3,
                    "unsupported_sample_count": 0,
                }
            )
            plan_runs.append(
                {
                    "condition_id": condition_id,
                    "delay_ms": delay,
                    "condition": condition,
                    "agent_scope": "E+R",
                    "duration_ticks": 1,
                }
            )
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, value in files.items():
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = 0o100600 << 16
            archive.writestr(member, value)
    metrics = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": 12,
        "baseline": spec.subject,
        "protocol_id": queue.PROTOCOL_ID,
        "checkpoint": f"/models/{spec.subject}_epoch_50.pth",
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "manifest_content_sha256": queue.MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": queue.OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": queue.SAMPLE_IDS_SHA256,
        "expected_sample_count": 2,
        "expected_ground_truth_count": 3,
        "expected_unsupported_sample_count": 0,
        "runs": runs,
    }
    plan = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_evaluation",
        "protocol_id": queue.PROTOCOL_ID,
        "baseline": spec.subject,
        "evaluation_subject_type": "improvement",
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "manifest_content_sha256": queue.MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": queue.OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": queue.SAMPLE_IDS_SHA256,
        "expected_sample_count": 2,
        "expected_ground_truth_count": 3,
        "expected_unsupported_sample_count": 0,
        "delays_ms": list(queue.DELAYS_MS),
        "conditions": list(queue.CONDITIONS),
        "runs": plan_runs,
    }
    task_id = "9" * 32
    checkpoint = {
        "task_id": spec.training_task_id,
        "model_id": binding["model_id"],
        "name": binding["model_name"],
        "sha256": binding["checkpoint_sha256"],
        "size_bytes": binding["checkpoint_size_bytes"],
    }
    run = {
        "schema_version": 1,
        "mode": "baseline_validate",
        "task_id": task_id,
        "baseline": spec.subject,
        "baseline_task_id": spec.training_task_id,
        "predecessor_task_id": spec.training_task_id,
        "training_dataset_id": queue.TRAINING_DATASET_ID,
        "protocol_id": queue.PROTOCOL_ID,
        "expected_sample_count": 2,
        "expected_ground_truth_count": 3,
        "expected_run_count": 12,
        "expected_manifest_content_sha256": queue.MANIFEST_CONTENT_SHA256,
        "expected_overlay_index_content_sha256": queue.OVERLAY_INDEX_CONTENT_SHA256,
        "expected_sample_ids_sha256": queue.SAMPLE_IDS_SHA256,
        "checkpoint": checkpoint,
    }
    evidence_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    artifacts = {
        "run_contract": Artifact("run_contract", run),
        "evaluation_plan": Artifact("evaluation_plan", plan),
        "controlled_baseline_metrics": Artifact(
            "controlled_baseline_metrics", metrics
        ),
        "controlled_baseline_evidence": Artifact(
            "controlled_baseline_evidence",
            "archive",
            digest=evidence_digest,
            path=archive_path,
        ),
    }
    evaluation = FakeTask(
        task_id,
        name=queue._evaluation_name(spec),
        status="completed",
        parent=spec.training_task_id,
        parameters=queue._evaluation_parameters(template, spec, binding),
        diff=queue._candidate_script_patch(spec, template.data.script["diff"]),
        models={
            "input": [
                Model(str(binding["model_id"]), "final", spec.training_task_id, "")
            ],
            "output": [],
        },
        artifacts=artifacts,
        queue_id=queue.QUEUE_IDS[spec.queue],
    )
    return spec, binding, evaluation


def test_completed_evidence_validates_prediction_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, binding, evaluation = _evidence_fixture(monkeypatch, tmp_path)
    result = queue._validate_completed_evaluation(evaluation, spec, binding)
    assert len(result["runs"]) == 12
    assert result["prediction_evidence_archive_sha256"] == hashlib.sha256(
        (tmp_path / "evidence.zip").read_bytes()
    ).hexdigest()


def test_metrics_accepts_only_exact_clearml_checkpoint_cache_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, binding, evaluation = _evidence_fixture(monkeypatch, tmp_path)
    metrics = evaluation.artifacts["controlled_baseline_metrics"].value
    metrics["checkpoint"] = (
        f"/cache/{'a' * 32}.{spec.subject}_epoch_50.pth"
    )
    assert len(queue._validate_metrics(metrics, spec, binding)) == 12
    metrics["checkpoint"] = (
        f"/cache/{'g' * 32}.{spec.subject}_epoch_50.pth"
    )
    with pytest.raises(RuntimeError, match="not final-only"):
        queue._validate_metrics(metrics, spec, binding)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("protocol_id",), "wrong", "protocol_id"),
        (("expected_sample_count",), 999, "sample_count"),
        (("sample_ids_sha256",), "f" * 64, "sample_ids_sha256"),
        (("checkpoint_sha256",), "e" * 64, "checkpoint_sha256"),
        (("runs", 0, "ground_truth_count"), 999, "ground_truth_count"),
    ],
)
def test_metrics_protocol_count_hash_and_model_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: tuple[object, ...],
    value: object,
    match: str,
) -> None:
    spec, binding, evaluation = _evidence_fixture(monkeypatch, tmp_path)
    metrics = evaluation.artifacts["controlled_baseline_metrics"].value
    target = metrics
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    evaluation.artifacts["controlled_baseline_metrics"].record["type_data"] = {
        "preview": json.dumps(metrics)
    }
    evaluation.reload()
    with pytest.raises(RuntimeError, match=match):
        queue._validate_completed_evaluation(evaluation, spec, binding)


def test_in_progress_evaluation_remains_pending_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world(monkeypatch, status="completed")
    first = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=True,
        script_editor=script_editor,
        sleeper=lambda _seconds: None,
    )
    evaluation = FakeTaskClass.tasks[first["entries"][0]["evaluation_task_id"]]
    evaluation.status = "in_progress"
    manifest = queue.reconcile_once(
        task_class=FakeTaskClass,
        controller_task=None,
        execute=False,
        script_editor=script_editor,
    )
    assert manifest["entries"][0]["evidence_status"] == "pending_evaluation"


def test_custom_json_file_artifact_is_read_with_exact_hash(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_plan.json"
    raw = b'{"schema_version":1,"plan_type":"test"}'
    path.write_bytes(raw)
    artifact = FileArtifact(
        "evaluation_plan",
        "custom",
        digest=hashlib.sha256(raw).hexdigest(),
        path=path,
    )
    task = FakeTask(
        "8" * 32,
        name="file artifact",
        status="completed",
        parent="7" * 32,
        parameters={},
        diff="pass\n",
        artifacts={"evaluation_plan": artifact},
    )
    assert queue._artifact_payload(task, "evaluation_plan") == {
        "schema_version": 1,
        "plan_type": "test",
    }
    artifact.record["hash"] = "f" * 64
    task.reload()
    with pytest.raises(RuntimeError, match="changed during read"):
        queue._artifact_payload(task, "evaluation_plan")


def test_custom_json_file_artifact_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_plan.json"
    raw = b"[]"
    path.write_bytes(raw)
    artifact = FileArtifact(
        "evaluation_plan",
        "custom",
        digest=hashlib.sha256(raw).hexdigest(),
        path=path,
    )
    task = FakeTask(
        "8" * 32,
        name="file artifact",
        status="completed",
        parent="7" * 32,
        parameters={},
        diff="pass\n",
        artifacts={"evaluation_plan": artifact},
    )
    with pytest.raises(RuntimeError, match="not a JSON object"):
        queue._artifact_payload(task, "evaluation_plan")


def test_default_plan_is_no_write_and_execute_token_is_exact() -> None:
    args = SimpleNamespace(
        execute=False,
        execute_token="",
        execute_remotely=False,
        service_queue="services",
        poll_seconds=1.0,
        timeout_hours=1.0,
        single_pass=True,
    )
    plan = queue.run(args, task_class=None)
    assert plan["remote_state_changed"] is False
    args.execute = True
    with pytest.raises(PermissionError, match="exact execute token"):
        queue.run(args, task_class=FakeTaskClass)
