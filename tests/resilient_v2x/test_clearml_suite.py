from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64
MAIN_TASK_ID = "a" * 32


def _load_script(name: str):
    path = ROOT / "tools/resilient_v2x" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bootstrap_args(*extra: str):
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    argv = [
        "--source-dataset-id",
        "source-id",
        "--source-archive-name",
        "source.tar.zst",
        "--source-archive-bytes",
        "123",
        "--source-archive-sha256",
        SHA,
        "--training-dataset-id",
        "training-id",
        "--native-bundle-bytes",
        "456",
        "--native-bundle-sha256",
        SHA,
        "--build-manifest-sha256",
        SHA,
        *extra,
    ]
    return bootstrap, bootstrap._parser().parse_args(argv)


def _draft_ids(controller) -> dict[str, str]:
    return {
        experiment: f"{index:032x}"
        for index, experiment in enumerate(controller.EXPERIMENT_ORDER, start=1)
    }


class _Model:
    def __init__(
        self,
        *,
        name: str,
        task: str = MAIN_TASK_ID,
        url: str = "http://10.100.34.118:8081/models/teacher.pth",
        model_id: str = "b" * 32,
        local_copy: Path | None = None,
    ) -> None:
        self.name = name
        self.task = task
        self.url = url
        self.id = model_id
        self._local_copy = local_copy

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
        assert self._local_copy is not None
        return str(self._local_copy)


class _MainTask:
    def __init__(self, models: list[_Model], status: str = "completed") -> None:
        self.status = status
        self._models = models

    def get_models(self):
        return {"output": self._models}


def test_fixed_suite_is_exactly_the_reviewed_thirteen() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    assert bootstrap.EXPERIMENT_ORDER == (
        "ptf_linear",
        "ptf_none",
        "router_static",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "no_distillation",
        "concat_capacity_matched",
        "v2x_vit",
        "cobevt",
        "coformernet",
        "bevfusion",
        "ffnet",
    )
    assert bootstrap.TEACHER_DEPENDENT_EXPERIMENTS == {
        "ptf_linear",
        "ptf_none",
        "router_static",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "concat_capacity_matched",
    }
    assert bootstrap.BASELINE_EXPERIMENTS == {
        "v2x_vit",
        "cobevt",
        "coformernet",
        "bevfusion",
        "ffnet",
    }
    assert bootstrap.BUILD_TASK_ID == "86a3ee30dcc749408ba19ba2088adb4c"


def test_original_stage_arguments_remain_valid_and_unchanged() -> None:
    bootstrap, args = _bootstrap_args(
        "--stage",
        "student",
        "--teacher-checkpoint",
        "teacher.pth",
    )
    bootstrap._validate_arguments(args)
    command = bootstrap._runner_command(
        args,
        python=Path("/venv/python"),
        runner=Path("clearml_train.py"),
    )
    assert command[-2:] == ["--teacher-checkpoint", "teacher.pth"]
    assert "--experiment-from-task" not in command


def test_experiment_arguments_enforce_predecessor_and_teacher_policy() -> None:
    bootstrap, missing_predecessor = _bootstrap_args(
        "--experiment-from-task",
        "ptf_linear",
        "--teacher-task-id",
        MAIN_TASK_ID,
    )
    with pytest.raises(ValueError, match="predecessor"):
        bootstrap._validate_arguments(missing_predecessor)

    _, valid = _bootstrap_args(
        "--experiment-from-task",
        "ptf_linear",
        "--teacher-task-id",
        MAIN_TASK_ID,
        "--predecessor-task-id",
        MAIN_TASK_ID,
    )
    bootstrap._validate_arguments(valid)

    _, forbidden = _bootstrap_args(
        "--experiment-from-task",
        "no_distillation",
        "--teacher-task-id",
        MAIN_TASK_ID,
        "--predecessor-task-id",
        MAIN_TASK_ID,
    )
    with pytest.raises(ValueError, match="forbids teacher"):
        bootstrap._validate_arguments(forbidden)


def test_training_commands_are_4gpu_ddp_and_training_only(tmp_path: Path) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    command = bootstrap._ddp_training_command(
        Path("/venv/python"),
        gpus=4,
        config=tmp_path / "config.py",
        work_dir=tmp_path / "work",
        max_epochs=50,
    )
    assert "--nproc_per_node=4" in command
    assert command[command.index("--module") + 1 :] [0] == (
        "tools.resilient_v2x.run_deterministic"
    )
    assert "tools/train.py" in command
    assert "tools/test.py" not in command
    assert not any("conditions/" in value for value in command)
    assert "train_cfg.val_interval" not in " ".join(command)
    assert set(bootstrap.RTX5090_HEADLESS_CFG_OPTIONS) <= set(command)


def test_baseline_plan_is_single_process_dry_run_then_sealed(tmp_path: Path) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    work_dir = tmp_path / "baseline"
    work_dir.mkdir()
    resolved = work_dir / "resolved_config.py"
    resolved.write_text("launcher = 'none'\n", encoding="utf-8")
    baseline_config = tmp_path / "v2x_vit.py"
    baseline_config.write_text("model = dict(type='baseline')\n", encoding="utf-8")
    plan_path = work_dir / "training_plan.json"
    plan = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_training",
        "baseline": "v2x_vit",
        "work_dir": str(work_dir.resolve()),
        "plan_path": str(plan_path.resolve()),
        "resolved_config": str(resolved.resolve()),
        "resolved_config_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "baseline_config": str(baseline_config.resolve()),
        "baseline_config_sha256": hashlib.sha256(
            baseline_config.read_bytes()
        ).hexdigest(),
        "resume": False,
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    command = bootstrap._baseline_plan_command(
        Path("/venv/python"),
        source_root=ROOT,
        baseline="v2x_vit",
        training_index=tmp_path / "training_overlays.json",
        work_dir=work_dir,
    )
    assert "torch.distributed.run" not in command
    assert command[-1] == "--dry-run"
    observed_plan, observed_config, _ = bootstrap._validate_baseline_dry_run(
        spec=bootstrap.EXPERIMENT_BY_NAME["v2x_vit"],
        work_dir=work_dir,
    )
    assert observed_plan == plan_path.resolve()
    assert observed_config == resolved.resolve()


def test_run_contract_binds_command_config_and_training_only_policy(
    tmp_path: Path,
) -> None:
    bootstrap, args = _bootstrap_args(
        "--experiment-from-task",
        "ptf_none",
        "--teacher-task-id",
        MAIN_TASK_ID,
        "--predecessor-task-id",
        MAIN_TASK_ID,
    )
    config = tmp_path / "config.py"
    config.write_bytes(b"train_cfg = dict(max_epochs=50, val_interval=1)\n")
    command = ["/venv/python", "-m", "torch.distributed.run", "tools/train.py"]
    contract = bootstrap._experiment_run_contract(
        args,
        task_id="c" * 32,
        spec=bootstrap.EXPERIMENT_BY_NAME["ptf_none"],
        dataset_root=tmp_path,
        teacher_contract={"model_id": "b" * 32},
        predecessor_task_id=MAIN_TASK_ID,
        config_path=config,
        training_command=command,
        baseline_plan=None,
    )
    assert contract["training_command"] == command
    assert contract["config"]["config_sha256"] == hashlib.sha256(
        config.read_bytes()
    ).hexdigest()
    assert contract["config"]["resolved_config_sha256"] == hashlib.sha256(
        config.read_bytes()
    ).hexdigest()
    assert contract["source_archive"]["sha256"] == SHA
    assert contract["predecessor_task_id"] == MAIN_TASK_ID
    assert contract["gpus"] == contract["ddp_processes"] == 4
    assert contract["seed"] == 20250218
    assert contract["learning_rate"] == 0.0001
    assert contract["auto_scale_lr"] is False
    assert contract["precision"] == "FP32"
    assert contract["per_epoch_validation"] is True
    assert contract["condition_evaluation"] is False


def test_teacher_handoff_is_unique_owned_durable_and_registered(
    tmp_path: Path,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    checkpoint = tmp_path / "teacher.pth"
    checkpoint.write_bytes(b"teacher")
    model = _Model(name=bootstrap.CLEAN_TEACHER_MODEL_NAME, local_copy=checkpoint)
    main = _MainTask([model])

    class Tasks:
        @staticmethod
        def get_task(*, task_id: str):
            assert task_id == MAIN_TASK_ID
            return main

    class Current:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def set_input_model(self, **kwargs) -> None:
            self.calls.append(kwargs)

    current = Current()
    env: dict[str, str] = {}
    contract = bootstrap._prepare_teacher_handoff(
        spec=bootstrap.EXPERIMENT_BY_NAME["ptf_linear"],
        teacher_task_id=MAIN_TASK_ID,
        task=current,
        task_class=Tasks,
        env=env,
    )
    assert current.calls == [
        {
            "model_id": "b" * 32,
            "name": "clean_teacher",
            "update_task_design": False,
            "update_task_labels": False,
        }
    ]
    assert env["RESILIENT_V2X_TEACHER_CHECKPOINT"] == str(checkpoint.resolve())
    assert contract is not None and contract["sha256"] == hashlib.sha256(
        b"teacher"
    ).hexdigest()

    wrong_port = _Model(
        name=bootstrap.CLEAN_TEACHER_MODEL_NAME,
        url="http://10.100.34.118:8080/model.pth",
    )
    with pytest.raises(RuntimeError, match="8081"):
        bootstrap._require_unique_teacher_output_model(
            _MainTask([wrong_port]),
            expected_task_id=MAIN_TASK_ID,
        )

    empty_checkpoint = tmp_path / "empty_teacher.pth"
    empty_checkpoint.touch()
    with pytest.raises(ValueError, match="empty"):
        bootstrap._download_teacher_checkpoint(
            _Model(
                name=bootstrap.CLEAN_TEACHER_MODEL_NAME,
                local_copy=empty_checkpoint,
            )
        )


@pytest.mark.parametrize(
    "experiment",
    ["no_distillation", "v2x_vit", "cobevt", "coformernet", "bevfusion", "ffnet"],
)
def test_teacher_free_experiments_never_call_teacher_api(experiment: str) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")

    class ExplodingTasks:
        @staticmethod
        def get_task(**_kwargs):
            raise AssertionError("teacher API must not be called")

    class ExplodingTask:
        def set_input_model(self, **_kwargs):
            raise AssertionError("input model must not be set")

    env = {"RESILIENT_V2X_TEACHER_CHECKPOINT": "stale.pth"}
    assert (
        bootstrap._prepare_teacher_handoff(
            spec=bootstrap.EXPERIMENT_BY_NAME[experiment],
            teacher_task_id=None,
            task=ExplodingTask(),
            task_class=ExplodingTasks,
            env=env,
        )
        is None
    )
    assert "RESILIENT_V2X_TEACHER_CHECKPOINT" not in env


def test_experiment_task_uses_current_or_initializes_with_fixed_output_uri() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    existing = object()

    class Existing:
        @staticmethod
        def current_task():
            return existing

    assert (
        bootstrap._current_or_init_experiment_task(
            Existing,
            bootstrap.EXPERIMENT_BY_NAME["ptf_linear"],
        )
        is existing
    )

    class Initializing:
        kwargs = None

        @staticmethod
        def current_task():
            return None

        @classmethod
        def init(cls, **kwargs):
            cls.kwargs = kwargs
            return "initialized"

    assert (
        bootstrap._current_or_init_experiment_task(
            Initializing,
            bootstrap.EXPERIMENT_BY_NAME["v2x_vit"],
        )
        == "initialized"
    )
    assert Initializing.kwargs["reuse_last_task_id"] is False
    assert Initializing.kwargs["output_uri"] == "http://10.100.34.118:8081"


def test_final_checkpoint_upload_is_nonempty_named_iteration_50(
    tmp_path: Path,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    checkpoint = tmp_path / "epoch_50.pth"
    checkpoint.write_bytes(b"checkpoint")

    class OutputModel:
        instances = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.id = "d" * 32
            self.update_kwargs = None
            self.instances.append(self)

        def update_weights(self, **kwargs):
            self.update_kwargs = kwargs
            return "http://10.100.34.118:8081/models/final.pth"

    contract = bootstrap._upload_experiment_checkpoint(
        task=object(),
        output_model_class=OutputModel,
        spec=bootstrap.EXPERIMENT_BY_NAME["ptf_linear"],
        checkpoint=checkpoint,
    )
    update = OutputModel.instances[0].update_kwargs
    assert update["target_filename"] == "ptf_linear_epoch_50.pth"
    assert update["iteration"] == 50
    assert update["async_enable"] is False
    assert contract["size_bytes"] == len(b"checkpoint")

    checkpoint.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        bootstrap._upload_experiment_checkpoint(
            task=object(),
            output_model_class=OutputModel,
            spec=bootstrap.EXPERIMENT_BY_NAME["ptf_linear"],
            checkpoint=checkpoint,
        )


def test_controller_plan_is_linear_uses_prebuilt_drafts_and_exact_chain() -> None:
    controller = _load_script("clearml_5090_sequence_controller.py")
    drafts = _draft_ids(controller)
    plan = controller.build_step_plan(
        draft_task_ids=drafts,
        main_task_id=MAIN_TASK_ID,
        worker_queue="GPU4-5090",
    )
    assert len(plan) == 13
    for index, step in enumerate(plan):
        experiment = controller.EXPERIMENT_ORDER[index]
        assert step["experiment"] == experiment
        assert step["base_task_id"] == drafts[experiment]
        expected_parent = [] if index == 0 else [plan[index - 1]["name"]]
        assert step["parents"] == expected_parent
        expected_predecessor = (
            MAIN_TASK_ID
            if index == 0
            else drafts[controller.EXPERIMENT_ORDER[index - 1]]
        )
        override = step["parameter_override"]
        assert override["Args/predecessor_task_id"] == expected_predecessor
        assert override["Args/experiment_from_task"] == experiment
        assert override["Args/gpus"] == 4
        assert not any("source" in key or "build" in key for key in override)
        if controller.EXPERIMENT_BY_NAME[experiment].requires_teacher:
            assert override["Args/teacher_task_id"] == MAIN_TASK_ID
        else:
            assert "Args/teacher_task_id" not in override


def test_pipeline_controller_does_not_clone_the_thirteen_drafts() -> None:
    module = _load_script("clearml_5090_sequence_controller.py")
    plan = module.build_step_plan(
        draft_task_ids=_draft_ids(module),
        main_task_id=MAIN_TASK_ID,
        worker_queue="GPU4-5090",
    )

    class FakeController:
        def __init__(self, **kwargs) -> None:
            self.init_kwargs = kwargs
            self.steps = []
            self.default_queue = None

        def set_default_execution_queue(self, queue: str) -> None:
            self.default_queue = queue

        def add_step(self, **kwargs) -> bool:
            self.steps.append(kwargs)
            return True

    guards = {}
    for step in plan:
        def guard(*_args) -> bool:
            return True

        guards[step["name"]] = guard

    controller = module.build_pipeline_controller(
        FakeController,
        step_plan=plan,
        project="project",
        name="name",
        version="1",
        worker_queue="GPU4-5090",
        step_guards=guards,
    )
    assert len(controller.steps) == 13
    assert all(step["clone_base_task"] is False for step in controller.steps)
    assert all(step["output_uri"] == module.FILES_SERVER_URI for step in controller.steps)
    assert all(
        step["pre_execute_callback"] is guards[step["name"]]
        for step in controller.steps
    )


def test_draft_validation_requires_bound_experiment_teacher_and_predecessor() -> None:
    module = _load_script("clearml_5090_sequence_controller.py")
    drafts = _draft_ids(module)
    plan = module.build_step_plan(
        draft_task_ids=drafts,
        main_task_id=MAIN_TASK_ID,
        worker_queue="GPU4-5090",
    )
    tasks = {}
    parameters_by_experiment = {}
    for step in plan:
        experiment = step["experiment"]
        parameters = dict(step["parameter_override"])
        parameters_by_experiment[experiment] = parameters
        tasks[drafts[experiment]] = SimpleNamespace(
            status="created",
            get_parameters=lambda parameters=parameters: parameters,
        )

    class Tasks:
        @staticmethod
        def get_task(*, task_id: str):
            return tasks[task_id]

    module._validate_draft_tasks(
        Tasks,
        drafts,
        main_task_id=MAIN_TASK_ID,
    )
    parameters_by_experiment["cobevt"]["Args/predecessor_task_id"] = "f" * 32
    with pytest.raises(RuntimeError, match="predecessor"):
        module._validate_draft_tasks(
            Tasks,
            drafts,
            main_task_id=MAIN_TASK_ID,
        )


def test_every_step_release_guard_rechecks_created_immutable_draft() -> None:
    module = _load_script("clearml_5090_sequence_controller.py")
    drafts = _draft_ids(module)
    plan = module.build_step_plan(
        draft_task_ids=drafts,
        main_task_id=MAIN_TASK_ID,
        worker_queue="GPU4-5090",
    )
    parameters = {
        step["experiment"]: dict(step["parameter_override"])
        for step in plan
    }
    draft_tasks = {
        drafts[step["experiment"]]: SimpleNamespace(
            status="created",
            get_parameters=(
                lambda experiment=step["experiment"]: parameters[experiment]
            ),
        )
        for step in plan
    }
    teacher = SimpleNamespace(
        name=module.CLEAN_TEACHER_MODEL_NAME,
        task=MAIN_TASK_ID,
        url="http://10.100.34.118:8081/models/teacher.pth",
    )
    main = SimpleNamespace(
        status="completed",
        get_models=lambda: {"output": [teacher]},
    )

    class Tasks:
        @staticmethod
        def get_task(*, task_id: str):
            return main if task_id == MAIN_TASK_ID else draft_tasks[task_id]

    guards = module._step_release_guards(
        Tasks,
        step_plan=plan,
        main_task_id=MAIN_TASK_ID,
        poll_seconds=1.0,
    )
    assert set(guards) == {step["name"] for step in plan}
    for step in plan:
        assert guards[step["name"]](object(), object(), {}) is True

    second = plan[1]
    draft_tasks[second["base_task_id"]].status = "queued"
    with pytest.raises(RuntimeError, match="must be created"):
        guards[second["name"]](object(), object(), {})
    draft_tasks[second["base_task_id"]].status = "created"

    third = plan[2]
    parameters[third["experiment"]]["Args/gpus"] = 8
    with pytest.raises(RuntimeError, match="Args/gpus mismatch"):
        guards[third["name"]](object(), object(), {})

    first = plan[0]
    main.status = "in_progress"

    def finish_main(**_kwargs) -> None:
        main.status = "completed"
        draft_tasks[first["base_task_id"]].status = "queued"

    main.wait_for_status = finish_main
    main.reload = lambda: None
    with pytest.raises(RuntimeError, match="must be created"):
        guards[first["name"]](object(), object(), {})


def test_remote_checkpoint_arguments_are_complete_and_stage_scoped() -> None:
    bootstrap, valid = _bootstrap_args(
        "--stage",
        "student",
        "--teacher-task-id",
        MAIN_TASK_ID,
        "--teacher-model-id",
        "b" * 32,
        "--teacher-checkpoint-sha256",
        SHA,
        "--allow-failed-teacher-task",
    )
    bootstrap._validate_arguments(valid)

    _, incomplete = _bootstrap_args(
        "--stage",
        "student",
        "--teacher-task-id",
        MAIN_TASK_ID,
    )
    with pytest.raises(ValueError, match="task ID, model ID, and checkpoint SHA"):
        bootstrap._validate_arguments(incomplete)

    _, missing_student = _bootstrap_args(
        "--stage",
        "validate",
        "--teacher-task-id",
        MAIN_TASK_ID,
        "--teacher-model-id",
        "b" * 32,
        "--teacher-checkpoint-sha256",
        SHA,
    )
    with pytest.raises(ValueError, match="teacher and student"):
        bootstrap._validate_arguments(missing_student)


def test_failed_teacher_and_completed_student_handoffs_are_sha_pinned(
    tmp_path: Path,
) -> None:
    bootstrap, args = _bootstrap_args(
        "--stage",
        "validate",
        "--teacher-task-id",
        MAIN_TASK_ID,
        "--teacher-model-id",
        "b" * 32,
        "--teacher-checkpoint-sha256",
        hashlib.sha256(b"teacher").hexdigest(),
        "--allow-failed-teacher-task",
        "--student-task-id",
        "c" * 32,
        "--student-model-id",
        "d" * 32,
        "--student-checkpoint-sha256",
        hashlib.sha256(b"student").hexdigest(),
    )
    bootstrap._validate_arguments(args)

    teacher_path = tmp_path / "teacher_epoch_50.pth"
    teacher_path.write_bytes(b"teacher")
    student_path = tmp_path / "student_epoch_50.pth"
    student_path.write_bytes(b"student")
    teacher_model = _Model(
        name=bootstrap.CLEAN_TEACHER_MODEL_NAME,
        task=MAIN_TASK_ID,
        model_id="b" * 32,
        url=(
            "http://10.100.34.118:8081/models/"
            "teacher_epoch_50.pth"
        ),
        local_copy=teacher_path,
    )
    student_model = _Model(
        name=bootstrap.DISTILLED_STUDENT_MODEL_NAME,
        task="c" * 32,
        model_id="d" * 32,
        url="http://10.100.34.118:8081/models/student_epoch_50.pth",
        local_copy=student_path,
    )
    teacher_task = _MainTask([teacher_model], status="failed")
    teacher_task.artifacts = {
        "run_contract": SimpleNamespace(
            get=lambda: {
                "task_id": MAIN_TASK_ID,
                "runtime_profile": "rtx5090",
                "gpus": 4,
                "max_epochs": 50,
                "checkpoint_policy": "final_epoch",
                "stage": "all",
                "dataset_id": "training-id",
            }
        )
    }
    student_task = _MainTask([student_model], status="completed")

    class Tasks:
        @staticmethod
        def get_task(*, task_id: str):
            if task_id == MAIN_TASK_ID:
                return teacher_task
            if task_id == "c" * 32:
                return student_task
            raise AssertionError(task_id)

    class Current:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def set_input_model(self, **kwargs) -> None:
            self.calls.append(kwargs)

    current = Current()
    teacher, student, contract = bootstrap._prepare_remote_checkpoint_handoffs(
        args,
        task=current,
        task_class=Tasks,
    )
    assert teacher == teacher_path.resolve()
    assert student == student_path.resolve()
    assert [call["model_id"] for call in current.calls] == ["b" * 32, "d" * 32]
    assert contract["failed_task_salvage"] is True
    assert contract["teacher"]["sha256"] == hashlib.sha256(b"teacher").hexdigest()
    assert contract["student"]["sha256"] == hashlib.sha256(b"student").hexdigest()

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        bootstrap._download_teacher_checkpoint(
            teacher_model,
            expected_sha256="f" * 64,
        )


def test_bootstrap_runtime_enables_trusted_mmengine_loading() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    env = bootstrap._runtime_environment(
        {"PATH": "/usr/bin", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "0"},
        venv_root=Path("/venv"),
        source_root=Path("/source"),
    )
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"

    with pytest.raises(RuntimeError, match="TORCH_FORCE_WEIGHTS_ONLY_LOAD"):
        bootstrap._runtime_environment(
            {"PATH": "/usr/bin", "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "yes"},
            venv_root=Path("/venv"),
            source_root=Path("/source"),
        )
