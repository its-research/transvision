from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_ID = "1" * 32
GATE_ID = "2" * 32
TEMPLATE_ID = "3" * 32
TEACHER_TASK_ID = "4" * 32
TEACHER_MODEL_ID = "5" * 32
TEACHER_SHA = "a" * 64
SOURCE_DATASET_ID = "6" * 32
TRAINING_DATASET_ID = "7" * 32
SOURCE_SHA = "b" * 64
NATIVE_SHA = "c" * 64
BUILD_SHA = "d" * 64


def _load_controller():
    path = ROOT / "tools/resilient_v2x/clearml_5090_training_controller.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Artifact:
    def __init__(self, payload: object) -> None:
        self.payload = copy.deepcopy(payload)

    def get(self):
        return copy.deepcopy(self.payload)


class _Model:
    def __init__(self, *, name: str, task: str, model_id: str, url: str) -> None:
        self.name = name
        self.task = task
        self.id = model_id
        self.url = url


class _FakeTask:
    def __init__(
        self,
        owner: "_FakeTasks",
        *,
        task_id: str,
        name: str,
        status: str,
        parameters: dict[str, object] | None = None,
        script: dict[str, object] | None = None,
        docker: str = "",
        parent: str = "",
        project: str = "ResilientV2X/Training",
        models: list[_Model] | None = None,
    ) -> None:
        self.owner = owner
        self.id = task_id
        self.name = name
        self.status = status
        self.parameters = dict(parameters or {})
        self.script = dict(script or {})
        self.docker = docker
        self.parent = parent
        self.project = project
        self.artifacts: dict[str, _Artifact] = {}
        self.models = list(models or [])
        self.data = SimpleNamespace(
            script=SimpleNamespace(to_dict=lambda: dict(self.script))
        )
        self.output_uri = None

    def get_parameters(self, **_kwargs):
        return dict(self.parameters)

    def set_parameters(self, parameters: dict[str, object]) -> None:
        self.parameters = dict(parameters)
        experiment = self.parameters["Args/experiment_from_task"]
        self.owner.events.append(f"override:{experiment}")

    def get_script(self):
        return {
            key: self.script.get(key)
            for key in ("working_dir", "entry_point", "branch", "repository")
        }

    def get_base_docker(self) -> str:
        return self.docker

    def get_models(self):
        return {"output": list(self.models)}

    def reload(self) -> None:
        if self.status == "queued":
            self.status = "in_progress"
            experiment = self.parameters["Args/experiment_from_task"]
            self.owner.events.append(f"running:{experiment}")
        elif self.status == "in_progress":
            experiment = self.parameters["Args/experiment_from_task"]
            if experiment == self.owner.fail_experiment:
                self.status = "failed"
                self.owner.active.discard(self.id)
                self.owner.events.append(f"failed:{experiment}")
            else:
                self.status = "completed"
                self.owner.active.discard(self.id)
                self.owner.events.append(f"complete:{experiment}")
                self._publish_experiment_outputs(experiment)

    def _publish_experiment_outputs(self, experiment: str) -> None:
        params = self.parameters
        requires_teacher = experiment in self.owner.module.TEACHER_DEPENDENT_EXPERIMENTS
        teacher = None
        if requires_teacher:
            teacher = {
                "task_id": TEACHER_TASK_ID,
                "model_id": TEACHER_MODEL_ID,
                "sha256": TEACHER_SHA,
            }
        run_contract = {
            "schema_version": 1,
            "mode": "experiment_from_task",
            "task_id": self.id,
            "experiment": experiment,
            "source_dataset_id": params["Args/source_dataset_id"],
            "training_dataset_id": params["Args/training_dataset_id"],
            "native_build_task_id": self.owner.module.BUILD_TASK_ID,
            "native_bundle_sha256": params["Args/native_bundle_sha256"],
            "build_manifest_sha256": params["Args/build_manifest_sha256"],
            "base_image_manifest_digest": (
                self.owner.module.BASE_IMAGE_AMD64_MANIFEST_DIGEST
            ),
            "source_archive": {
                "name": params["Args/source_archive_name"],
                "size_bytes": int(str(params["Args/source_archive_bytes"])),
                "sha256": params["Args/source_archive_sha256"],
            },
            "predecessor_task_id": params["Args/predecessor_task_id"],
            "gpus": 4,
            "ddp_processes": 4,
            "max_epochs": 50,
            "amp": False,
            "precision": "FP32",
            "runtime_profile": "rtx5090",
            "val_interval": 10,
            "per_epoch_validation": False,
            "condition_evaluation": False,
            "teacher": teacher,
        }
        model_id = f"{int(self.id, 16) + 1000:032x}"
        model_name = f"ResilientV2X {experiment} final checkpoint"
        if experiment == self.owner.invalid_model_experiment:
            model_name = "wrong model"
        model_url = f"http://10.100.34.118:8081/models/{experiment}_epoch_50.pth"
        model = _Model(
            name=model_name,
            task=self.id,
            model_id=model_id,
            url=model_url,
        )
        self.models = [model]
        self.artifacts = {
            "run_contract": _Artifact(run_contract),
            "final_checkpoint_contract": _Artifact(
                {
                    "model_id": model_id,
                    "name": f"ResilientV2X {experiment} final checkpoint",
                    "url": model_url,
                    "filename": "epoch_50.pth",
                    "size_bytes": 123456,
                    "sha256": "e" * 64,
                }
            ),
        }

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        assert wait_on_upload is True
        self.artifacts[name] = _Artifact(artifact_object)
        self.owner.events.append(f"artifact:{name}")
        return True

    def flush(self, *, wait_for_uploads: bool) -> None:
        assert wait_for_uploads is True


class _FakeTasks:
    def __init__(self, module, *, gate_status: str = "completed") -> None:
        self.module = module
        self.registry: dict[str, _FakeTask] = {}
        self.events: list[str] = []
        self.active: set[str] = set()
        self.clone_counts: dict[str, int] = {}
        self.fail_experiment: str | None = None
        self.invalid_model_experiment: str | None = None
        self.enqueue_failures = 0
        self.next_id = 100
        self.controller = self._add(
            task_id=CONTROLLER_ID,
            name="controller",
            status="in_progress",
        )
        self.gate = self._add(
            task_id=GATE_ID,
            name="validation gate",
            status=gate_status,
        )
        teacher_model = _Model(
            name="ResilientV2X clean teacher",
            task=TEACHER_TASK_ID,
            model_id=TEACHER_MODEL_ID,
            url="http://10.100.34.118:8081/models/teacher_epoch_50.pth",
        )
        self.teacher = self._add(
            task_id=TEACHER_TASK_ID,
            name="teacher",
            status="failed",
            models=[teacher_model],
        )
        self.template = self._add(
            task_id=TEMPLATE_ID,
            name="verified bootstrap template",
            status="completed",
            parameters=_template_parameters(),
            script=_template_script(module),
            docker=_template_docker(module),
        )

    def _add(self, **kwargs) -> _FakeTask:
        task = _FakeTask(self, **kwargs)
        self.registry[task.id] = task
        return task

    def get_task(self, *, task_id: str):
        return self.registry[task_id]

    def get_tasks(
        self,
        *,
        project_name: str,
        task_name: str,
        allow_archived: bool = True,
        task_filter: dict[str, object] | None = None,
    ):
        assert allow_archived is True
        parent = (task_filter or {}).get("parent")
        return [
            task
            for task in self.registry.values()
            if task.project == project_name
            and re.fullmatch(task_name, task.name)
            and (parent is None or task.parent == parent)
        ]

    def clone(
        self,
        *,
        source_task: _FakeTask,
        name: str,
        parent: str,
    ):
        experiment = name.split()[3]
        self.events.append(f"clone:{experiment}")
        self.clone_counts[experiment] = self.clone_counts.get(experiment, 0) + 1
        task_id = f"{self.next_id:032x}"
        self.next_id += 1
        docker = source_task.docker
        if experiment == getattr(self, "drift_clone_experiment", None):
            docker = docker.replace("--network host", "--network bridge")
        return self._add(
            task_id=task_id,
            name=name,
            status="created",
            parameters=source_task.parameters,
            script=source_task.script,
            docker=docker,
            parent=parent,
            project=source_task.project,
        )

    def enqueue(self, *, task: _FakeTask, queue_name: str):
        assert queue_name in {"GPU4-5090", "GPU4-A100", "GPU4-V100"}
        if self.enqueue_failures:
            self.enqueue_failures -= 1
            self.events.append("enqueue-error")
            raise RuntimeError("simulated enqueue interruption")
        if getattr(self, "allow_parallel", False):
            pass
        else:
            assert not self.active, "the controller attempted parallel training"
        experiment = task.parameters["Args/experiment_from_task"]
        self.events.append(f"enqueue:{experiment}")
        self.active.add(task.id)
        task.status = "queued"
        return {"queued": 1, "updated": 1}


def _template_parameters() -> dict[str, object]:
    return {
        "Args/source_dataset_id": SOURCE_DATASET_ID,
        "Args/source_archive_name": "source.tar.zst",
        "Args/source_archive_bytes": "123",
        "Args/source_archive_sha256": SOURCE_SHA,
        "Args/training_dataset_id": TRAINING_DATASET_ID,
        "Args/native_bundle_bytes": "456",
        "Args/native_bundle_sha256": NATIVE_SHA,
        "Args/build_manifest_sha256": BUILD_SHA,
        "Args/stage": "validate",
        "Args/gpus": 4,
        "Args/max_epochs": 50,
        "Args/student_task_id": "8" * 32,
        "Args/student_model_id": "9" * 32,
        "Args/student_checkpoint_sha256": "f" * 64,
        "Args/teacher_task_id": TEACHER_TASK_ID,
        "Args/teacher_model_id": TEACHER_MODEL_ID,
        "Args/teacher_checkpoint_sha256": TEACHER_SHA,
        "General/retained_template_parameter": "sealed",
    }


def _template_script(module) -> dict[str, object]:
    markers = "\n".join(
        (
            module.BASE_IMAGE_AMD64_MANIFEST_DIGEST,
            module.BASE_IMAGE_CONFIG_DIGEST,
            module.BUILD_TASK_ID,
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        )
    )
    return {
        "repository": "",
        "branch": "",
        "version_num": "",
        "working_dir": ".",
        "entry_point": "clearml_5090_bootstrap.py",
        "diff": markers,
    }


def _template_docker(module) -> str:
    return " ".join(
        (
            module.EXPECTED_DOCKER_IMAGE,
            "--network host",
            "--env PYTHONSAFEPATH=1",
        )
    )


def _args(**overrides):
    values = {
        "gate_task_id": GATE_ID,
        "paper_controller_task_id": None,
        "paper_controller_summary_artifact": "paper_controller_summary",
        "template_task_id": TEMPLATE_ID,
        "teacher_task_id": TEACHER_TASK_ID,
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "allow_failed_teacher_task": True,
        "worker_queue": "GPU4-5090",
        "worker_queues": "",
        "max_parallel": 1,
        "canary_first": False,
        "adopt_experiment": [],
        "project": "ResilientV2X/Training",
        "poll_seconds": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_evidence_priority_order_is_exact_and_fixed() -> None:
    module = _load_controller()
    assert module.EXPERIMENT_ORDER == (
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
        "coformernet",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "concat_capacity_matched",
        "ffnet",
        "bevfusion",
        "v2x_vit",
        "cobevt",
    )


def test_parameter_override_is_complete_and_removes_student_handoff() -> None:
    module = _load_controller()
    template = _template_parameters()
    teacher = module._experiment_parameters(
        template,
        experiment="ptf_none",
        predecessor_task_id=GATE_ID,
        teacher_task_id=TEACHER_TASK_ID,
        teacher_model_id=TEACHER_MODEL_ID,
        teacher_checkpoint_sha256=TEACHER_SHA,
        allow_failed_teacher_task=True,
    )
    assert teacher["Args/experiment_from_task"] == "ptf_none"
    assert teacher["Args/predecessor_task_id"] == GATE_ID
    assert teacher["Args/gpus"] == 4
    assert teacher["Args/stage"] == "all"
    assert teacher["Args/max_epochs"] == 50
    assert teacher["Args/amp"] is False
    assert teacher["Args/teacher_task_id"] == TEACHER_TASK_ID
    assert teacher["Args/teacher_model_id"] == TEACHER_MODEL_ID
    assert teacher["Args/teacher_checkpoint_sha256"] == TEACHER_SHA
    assert teacher["Args/allow_failed_teacher_task"] is True
    assert not module.STUDENT_HANDOFF_KEYS & set(teacher)

    teacher_free = module._experiment_parameters(
        template,
        experiment="no_distillation",
        predecessor_task_id="a" * 32,
        teacher_task_id=TEACHER_TASK_ID,
        teacher_model_id=TEACHER_MODEL_ID,
        teacher_checkpoint_sha256=TEACHER_SHA,
        allow_failed_teacher_task=True,
    )
    assert not any(key.startswith("Args/teacher_") for key in teacher_free)
    assert "Args/allow_failed_teacher_task" not in teacher_free
    assert not module.STUDENT_HANDOFF_KEYS & set(teacher_free)


def test_full_suite_clones_queues_and_validates_strictly_one_at_a_time() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    summary = module.run_training_suite(
        _args(),
        task_class=tasks,
        controller_task=tasks.controller,
        sleeper=lambda _seconds: None,
    )
    assert summary["status"] == "completed"
    assert summary["task_count"] == 13
    assert [item["experiment"] for item in summary["results"]] == list(
        module.EXPERIMENT_ORDER
    )
    module._require_valid_seal(summary, context="summary")
    progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    module._require_valid_seal(progress, context="progress")
    assert [step["state"] for step in progress["steps"]] == ["completed"] * 13
    assert tasks.active == set()

    previous_complete = -1
    for experiment in module.EXPERIMENT_ORDER:
        clone_index = tasks.events.index(f"clone:{experiment}")
        enqueue_index = tasks.events.index(f"enqueue:{experiment}")
        complete_index = tasks.events.index(f"complete:{experiment}")
        assert previous_complete < clone_index < enqueue_index < complete_index
        previous_complete = complete_index
    clones = [task for task in tasks.registry.values() if task.parent == CONTROLLER_ID]
    assert len(clones) == 13
    predecessor = GATE_ID
    for experiment, task in zip(module.EXPERIMENT_ORDER, clones, strict=True):
        assert task.parameters["Args/predecessor_task_id"] == predecessor
        assert task.parameters["Args/experiment_from_task"] == experiment
        assert not module.STUDENT_HANDOFF_KEYS & set(task.parameters)
        if experiment in module.TEACHER_DEPENDENT_EXPERIMENTS:
            assert task.parameters["Args/teacher_model_id"] == TEACHER_MODEL_ID
        else:
            assert "Args/teacher_task_id" not in task.parameters
        predecessor = task.id


def test_failed_experiment_stops_before_any_later_clone() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.fail_experiment = "ptf_none"
    with pytest.raises(RuntimeError, match="ended without completion|failed"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {"ptf_none": 1}
    progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    assert progress["steps"][0]["state"] == "failed"
    assert all(step["state"] == "pending" for step in progress["steps"][1:])
    assert module.SUMMARY_ARTIFACT not in tasks.controller.artifacts


def test_enqueue_interruption_recovers_created_clone_without_duplication() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.enqueue_failures = 1
    with pytest.raises(RuntimeError, match="simulated enqueue interruption"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    assert progress["steps"][0]["state"] == "created"
    first_task_id = progress["steps"][0]["task_id"]

    summary = module.run_training_suite(
        _args(),
        task_class=tasks,
        controller_task=tasks.controller,
        sleeper=lambda _seconds: None,
    )
    assert summary["status"] == "completed"
    assert tasks.clone_counts["ptf_none"] == 1
    assert summary["results"][0]["task_id"] == first_task_id


def test_crash_window_clone_is_discovered_by_exact_name_and_not_repeated() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    recovered_id = "a1" * 16
    tasks._add(
        task_id=recovered_id,
        name=module._task_name(1, "ptf_none", CONTROLLER_ID),
        status="created",
        # Simulate a process death immediately after Task.clone(), before the
        # inherited student/validation parameters were replaced.
        parameters=tasks.template.parameters,
        script=tasks.template.script,
        docker=tasks.template.docker,
        parent=CONTROLLER_ID,
    )
    summary = module.run_training_suite(
        _args(),
        task_class=tasks,
        controller_task=tasks.controller,
        sleeper=lambda _seconds: None,
    )
    assert summary["results"][0]["task_id"] == recovered_id
    assert tasks.clone_counts.get("ptf_none", 0) == 0
    assert tasks.events.count("enqueue:ptf_none") == 1


@pytest.mark.parametrize("drift", ["entrypoint", "docker", "source"])
def test_template_identity_drift_fails_before_cloning(drift: str) -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    if drift == "entrypoint":
        tasks.template.script["entry_point"] = "/tmp/wrong.py"
    elif drift == "docker":
        tasks.template.docker += " ".join(
            (
                " --env CLEARML_API_HOST=http://10.100.35.118:8008",
                "--env CLEARML_WEB_HOST=http://10.100.34.118:8080",
                "--env CLEARML_FILES_HOST=http://10.100.34.118:8081",
            )
        )
    else:
        tasks.template.parameters["Args/source_archive_sha256"] = "invalid"
    with pytest.raises((RuntimeError, ValueError)):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {}


def test_clone_identity_drift_is_rejected_before_enqueue() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.drift_clone_experiment = "ptf_none"
    with pytest.raises(RuntimeError, match="Docker command drifted"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {"ptf_none": 1}
    assert not any(event.startswith("enqueue:") for event in tasks.events)


def test_invalid_output_model_blocks_the_next_experiment() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.invalid_model_experiment = "ptf_none"
    with pytest.raises(RuntimeError, match="exactly one final OutputModel"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {"ptf_none": 1}
    assert "clone:ptf_linear" not in tasks.events
    progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    assert progress["steps"][0]["state"] == "failed"
    assert progress["steps"][0]["failure_status"] == ("completion_validation_failed")


def test_failed_gate_prevents_all_cloning() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module, gate_status="failed")
    with pytest.raises(RuntimeError, match="validation gate.*failed"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {}


def test_gate_can_be_resolved_from_paper_controller_summary() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    paper_id = "f" * 32
    paper = tasks._add(
        task_id=paper_id,
        name="paper controller",
        status="completed",
    )
    paper.artifacts["paper_controller_summary"] = _Artifact(
        {"validation": {"task_id": GATE_ID}}
    )
    args = _args(gate_task_id=None, paper_controller_task_id=paper_id)
    assert (
        module._resolve_gate_task_id(
            args,
            task_class=tasks,
            sleeper=lambda _seconds: None,
        )
        == GATE_ID
    )


def test_controller_source_never_uses_pipeline_controller() -> None:
    source = (
        ROOT / "tools/resilient_v2x/clearml_5090_training_controller.py"
    ).read_text(encoding="utf-8")
    assert "from clearml import PipelineController" not in source
    assert "PipelineController(" not in source




def test_experiment_syspath_patch_allows_a100_v100_smoke_capability() -> None:
    module = _load_controller()
    smoke = (
        "if any(torch.cuda.get_device_capability(i) != (12, 0) for i in range(gpu_count)):\n"
        '    raise RuntimeError("all GPUs must have compute capability 12.0")\n'
    )
    patched = module._apply_experiment_syspath_patch(smoke)
    assert module._EXPERIMENT_SMOKE_CAPABILITY_MARKER in patched
    assert "all GPUs must have compute capability 12.0" not in patched
    assert "(8, 0)" in patched and "(7, 0)" in patched


def test_find_recoverable_clone_skips_failed_tasks() -> None:
    module = _load_controller()

    class Tasks:
        @staticmethod
        def get_tasks(**_kwargs):
            return [
                SimpleNamespace(
                    name="ResilientV2X post-main 03 router_static [ctrl]",
                    parent="ctrl",
                    status="failed",
                )
            ]

    found = module._find_recoverable_clone(
        Tasks,
        project="ResilientV2X/Training",
        name="ResilientV2X post-main 03 router_static [ctrl]",
        controller_task_id="ctrl",
    )
    assert found is None


def test_canary_unblocks_after_first_training_iteration() -> None:
    module = _load_controller()

    class Task:
        def __init__(self, last_iteration):
            self._last = last_iteration

        def get_last_iteration(self):
            return self._last

    class Tasks:
        @staticmethod
        def get_task(task_id):
            return Task(1 if task_id == "canary" else 0)

    steps = [
        {"state": "completed", "adopted": True, "task_id": "old"},
        {"state": "running", "adopted": False, "task_id": "canary"},
    ]
    assert module._canary_blocks_extra_slots(
        steps, canary_first=True, task_class=Tasks
    ) is False
    assert module._canary_blocks_extra_slots(
        [{"state": "running", "adopted": False, "task_id": "x"}],
        canary_first=True,
        task_class=type(
            "T",
            (),
            {"get_task": staticmethod(lambda task_id: Task(0))},
        ),
    ) is True

def test_remote_controller_initialization_enables_argparse_connection() -> None:
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

    assert module._current_controller_task(
        Tasks,
        auto_connect_arg_parser=True,
    ) is current
    assert calls == [
        {
            "project_name": module.DEFAULT_PROJECT,
            "task_name": "ResilientV2X post-main sequential training controller",
            "reuse_last_task_id": False,
            "output_uri": module.FILES_SERVER_URI,
            "auto_connect_arg_parser": True,
        }
    ]
