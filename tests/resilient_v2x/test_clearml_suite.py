from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64
MAIN_TASK_ID = "a" * 32
EMPTY_MAPPING_SHA = hashlib.sha256(b"").hexdigest()
FORMAL_CONFIG_SUBJECTS = (
    "support_residual",
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
    "linear_no_distillation",
    "no_distillation_peak_lr_3e4",
    "ego_only",
    "fcooper",
    "attfuse",
    "v2vnet",
    "when2com",
    "where2comm",
    "late_fusion",
    "disconet",
    "how2comm",
    "resilient_v2x",
)
EXPECTED_NESTED_TEACHER_SUBJECTS = frozenset(
    {
        "support_residual",
        "ptf_none",
        "ptf_linear",
        "router_static",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "concat_capacity_matched",
        "resilient_v2x",
    }
)


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


def _nvidia_runner(gpu_output: str, process_output: str):
    outputs = iter((gpu_output, process_output))

    def run(command, *, capture=False, **_kwargs):
        assert command[0] == "nvidia-smi"
        assert capture is True
        return SimpleNamespace(stdout=next(outputs), stderr="", returncode=0)

    return run


def _idle_gpu_rows(*, used_mib: int = 100) -> str:
    return "\n".join(
        f"{index}, GPU-{index}, 81920, {used_mib}, {81920 - used_mib}"
        for index in range(4)
    )


def test_bootstrap_gpu_memory_preflight_accepts_four_idle_visible_gpus() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    result = bootstrap._capture_gpu_memory_preflight(
        4,
        environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        runner=_nvidia_runner(_idle_gpu_rows(), ""),
    )

    assert result["event"] == "gpu_memory_preflight_pass"
    assert result["gpu_count"] == 4
    assert result["compute_process_count"] == 0


def test_bootstrap_gpu_memory_preflight_rejects_unregistered_memory_use() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    with pytest.raises(RuntimeError, match="idle-memory preflight threshold"):
        bootstrap._capture_gpu_memory_preflight(
            4,
            environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            runner=_nvidia_runner(_idle_gpu_rows(used_mib=26_000), ""),
        )


def test_bootstrap_gpu_memory_preflight_rejects_compute_process() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    with pytest.raises(RuntimeError, match="already have compute processes"):
        bootstrap._capture_gpu_memory_preflight(
            4,
            environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            runner=_nvidia_runner(_idle_gpu_rows(), "GPU-2, 12345, 512\n"),
        )


def test_bootstrap_memory_preflight_precedes_torch_cuda_capture() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    source = inspect.getsource(bootstrap.main)
    assert source.index("_capture_gpu_memory_preflight") < source.index(
        "_capture_gpu_runtime"
    )


def _draft_ids(controller) -> dict[str, str]:
    return {
        experiment: f"{index:032x}"
        for index, experiment in enumerate(controller.EXPERIMENT_ORDER, start=1)
    }


def _common_teacher_initialization_audit(
    *,
    nested_teacher: bool,
    fusion_keys: int = 7,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": "shared-only-clean-teacher-initialization-v1",
        "result": "pass",
        "checkpoint": {
            "path": "/tmp/teacher.pth",
            "filename": "teacher.pth",
            "size_bytes": 143_246_244,
            "sha256": SHA,
            "expected_sha256": SHA,
        },
        "source": {
            "keys": 617,
            "numel": 35_811_485,
            "bytes": 143_246_244,
            "expected_keys": 617,
            "common_keys": 468,
            "expected_common_keys": 468,
            "fusion_keys": 149,
            "expected_fusion_keys": 149,
            "state_sha256": "b" * 64,
        },
        "shared_initialization": {
            "prefixes": [
                "lidar_encoder.",
                "camera_encoder.",
                "bbox_head.",
                "detection_projection.",
            ],
            "keys": 468,
            "numel": 31_506_934,
            "bytes": 126_028_040,
            "expected_keys": 468,
            "state_sha256": "c" * 64,
            "shape_dtype_verified": True,
            "exact_tensor_equality_verified": True,
        },
        "method_specific_fusion": {
            "keys": fusion_keys,
            "numel": 0 if fusion_keys == 0 else 1_024,
            "bytes": 0 if fusion_keys == 0 else 4_096,
            "sha256_before": EMPTY_MAPPING_SHA if fusion_keys == 0 else "d" * 64,
            "sha256_after": EMPTY_MAPPING_SHA if fusion_keys == 0 else "d" * 64,
            "unchanged": True,
        },
        "target": {
            "model_type": "ControlledModel",
            "target_key_count": (468 + fusion_keys + (617 if nested_teacher else 0)),
            "target_common_key_count": 468,
            "target_fusion_key_count": fusion_keys,
            "nested_teacher_present": nested_teacher,
            "nested_teacher_key_count": 617 if nested_teacher else 0,
            "nested_teacher_full_equality_verified": nested_teacher,
        },
    }


def test_official_ffnet_single_gpu_stage_is_closed_and_parseable() -> None:
    _, bootstrap_args = _bootstrap_args("--stage", "ffnet_official_train")
    runner = _load_script("clearml_train.py")
    train_args = runner._parser().parse_args(
        ["--dataset-id", "dataset", "--stage", "ffnet_official_train"]
    )

    assert bootstrap_args.stage == "ffnet_official_train"
    assert train_args.stage == "ffnet_official_train"


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


def test_fixed_suite_preserves_existing_order_and_appends_primary_method() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    assert bootstrap.CORE_EXPERIMENT_ORDER == (
        "support_residual",
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
    assert bootstrap.EXPERIMENT_ORDER == bootstrap.CORE_EXPERIMENT_ORDER + (
        "linear_no_distillation",
        "no_distillation_peak_lr_3e4",
        "ego_only",
        "fcooper",
        "attfuse",
        "v2vnet",
        "when2com",
        "where2comm",
        "late_fusion",
        "disconet",
        "how2comm",
        "resilient_v2x",
    )
    assert bootstrap.TEACHER_DEPENDENT_EXPERIMENTS == set(bootstrap.EXPERIMENT_ORDER)
    assert bootstrap.BASELINE_EXPERIMENTS == {
        "v2x_vit",
        "cobevt",
        "coformernet",
        "bevfusion",
        "ffnet",
        "ego_only",
        "fcooper",
        "attfuse",
        "v2vnet",
        "when2com",
        "where2comm",
        "late_fusion",
        "disconet",
        "how2comm",
    }
    assert bootstrap.PRIMARY_METHOD_EXPERIMENTS == {"resilient_v2x"}
    assert bootstrap.NESTED_TEACHER_EXPERIMENTS == (EXPECTED_NESTED_TEACHER_SUBJECTS)
    primary = bootstrap.EXPERIMENT_BY_NAME["resilient_v2x"]
    assert primary.kind == "primary_method"
    assert primary.config == "configs/resilient_v2x/dair_resilient_v2x.py"
    assert primary.requires_teacher is True
    assert bootstrap.BUILD_TASK_ID == "9055c0d3c4dd450c8a75dddfb21a56bd"


def test_vehicle_runtime_reuses_native_bundle_only_for_audited_python_changes() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    assert bootstrap.NATIVE_BUILD_SOURCE_DATASET_ID == (
        "bcbd15ae7e454e9885bc4250a3de774e"
    )
    assert len(bootstrap.NATIVE_BUILD_INPUT_SHA256) == 14
    assert set(bootstrap.NATIVE_BUILD_INPUT_SHA256) >= {
        "setup.py",
        "transvision/models/bev_pool/src/bev_pool_cuda.cu",
        "transvision/models/voxel/src/voxelization_cuda.cu",
    }
    assert bootstrap.NATIVE_COMPATIBLE_PYTHON_ONLY_CHANGES
    assert (
        "configs/resilient_v2x/dair_resilient_v2x.py"
        in bootstrap.NATIVE_COMPATIBLE_PYTHON_ONLY_CHANGES
    )
    assert (
        "configs/resilient_v2x/improvements/support_residual.py"
        in bootstrap.NATIVE_COMPATIBLE_PYTHON_ONLY_CHANGES
    )
    assert all(
        path.endswith(".py") for path in bootstrap.NATIVE_COMPATIBLE_PYTHON_ONLY_CHANGES
    )
    assert not any(
        path.endswith((".cu", ".cpp", ".cuh", ".h"))
        for path in bootstrap.NATIVE_COMPATIBLE_PYTHON_ONLY_CHANGES
    )


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


def test_training_dataset_materialization_forbids_soft_links(tmp_path: Path) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    destination = tmp_path / "task-dataset"

    class Dataset:
        pass

    class Runner:
        calls: list[tuple[object, Path]] = []

        @staticmethod
        def _materialize_clearml_dataset(dataset: object, path: Path) -> Path:
            Runner.calls.append((dataset, path))
            return path.resolve()

    dataset = Dataset()
    assert (
        bootstrap._materialize_training_dataset(
            dataset,
            destination=destination,
            runner=Runner,
        )
        == destination.resolve()
    )
    assert Runner.calls == [(dataset, destination)]


def test_baseline_validate_arguments_are_isolated_and_complete() -> None:
    bootstrap, args = _bootstrap_args(
        "--stage",
        "baseline_validate",
        "--controlled-baseline",
        "ffnet",
        "--controlled-baseline-task-id",
        "b" * 32,
        "--controlled-baseline-model-id",
        "d" * 32,
        "--controlled-baseline-checkpoint-sha256",
        "e" * 64,
        "--predecessor-task-id",
        "c" * 32,
    )

    bootstrap._validate_arguments(args)

    assert args.experiment_from_task is None
    assert args.controlled_baseline == "ffnet"
    assert args.controlled_baseline_task_id == "b" * 32
    assert args.controlled_baseline_model_id == "d" * 32
    assert args.controlled_baseline_checkpoint_sha256 == "e" * 64
    assert args.predecessor_task_id == "c" * 32


def test_ablation_validate_arguments_use_the_same_dependency_contract() -> None:
    bootstrap, args = _bootstrap_args(
        "--stage",
        "baseline_validate",
        "--controlled-baseline",
        "ptf_none",
        "--controlled-baseline-task-id",
        "b" * 32,
        "--controlled-baseline-model-id",
        "d" * 32,
        "--controlled-baseline-checkpoint-sha256",
        "e" * 64,
        "--predecessor-task-id",
        "c" * 32,
    )

    bootstrap._validate_arguments(args)

    assert args.controlled_baseline == "ptf_none"
    assert args.controlled_baseline_task_id == "b" * 32
    assert args.controlled_baseline_model_id == "d" * 32
    assert args.controlled_baseline_checkpoint_sha256 == "e" * 64
    assert args.predecessor_task_id == "c" * 32


@pytest.mark.parametrize(
    "missing",
    ("baseline", "baseline_task", "baseline_model", "baseline_sha", "predecessor"),
)
def test_baseline_validate_rejects_missing_dependency_fields(missing: str) -> None:
    extra = ["--stage", "baseline_validate"]
    if missing != "baseline":
        extra.extend(["--controlled-baseline", "ffnet"])
    if missing != "baseline_task":
        extra.extend(["--controlled-baseline-task-id", "b" * 32])
    if missing != "baseline_model":
        extra.extend(["--controlled-baseline-model-id", "d" * 32])
    if missing != "baseline_sha":
        extra.extend(["--controlled-baseline-checkpoint-sha256", "e" * 64])
    if missing != "predecessor":
        extra.extend(["--predecessor-task-id", "c" * 32])
    bootstrap, args = _bootstrap_args(*extra)

    with pytest.raises(ValueError, match="baseline_validate requires"):
        bootstrap._validate_arguments(args)


def test_experiment_rejects_controlled_baseline_parameters() -> None:
    bootstrap, args = _bootstrap_args(
        "--experiment-from-task",
        "ffnet",
        "--predecessor-task-id",
        "c" * 32,
        "--controlled-baseline",
        "ffnet",
        "--controlled-baseline-task-id",
        "b" * 32,
        "--controlled-baseline-model-id",
        "d" * 32,
        "--controlled-baseline-checkpoint-sha256",
        "e" * 64,
    )

    with pytest.raises(
        ValueError,
        match="controlled-baseline options cannot be combined with an experiment",
    ):
        bootstrap._validate_arguments(args)


def test_controlled_baseline_metrics_upload_as_json_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    metrics = {"complete": True, "planned_run_count": 12, "runs": []}
    stage = bootstrap.ControlledEvidenceStage(tmp_path, (), ())
    monkeypatch.setattr(
        bootstrap,
        "_verify_controlled_evidence_stage",
        lambda _stage: ({}, metrics),
    )
    monkeypatch.setattr(
        bootstrap,
        "_verify_uploaded_controlled_evidence",
        lambda _task, _stage: None,
    )

    class Task:
        def __init__(self) -> None:
            self.uploads: list[tuple[str, object, bool]] = []
            self.artifacts: dict[str, object] = {}

        def upload_artifact(
            self,
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            self.uploads.append((name, artifact_object, wait_on_upload))
            return True

        def flush(self, *, wait_for_uploads: bool) -> None:
            assert wait_for_uploads is True

        def reload(self) -> None:
            pass

    task = Task()
    bootstrap._upload_controlled_baseline_artifacts(
        task,
        evidence_stage=stage,
    )

    assert task.uploads == [
        ("evaluation_plan", str(tmp_path / "evaluation_plan.json"), True),
        ("controlled_baseline_metrics", metrics, True),
        ("controlled_baseline_evidence", str(tmp_path), True),
    ]
    assert isinstance(task.uploads[1][1], dict)


def test_controlled_evaluator_is_headless_and_sealed() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    evaluator = ROOT / "tools/resilient_v2x/evaluate_controlled_baselines.py"

    assert (
        bootstrap._sha256(evaluator)
        == bootstrap.CONTROLLED_BASELINE_PROTOCOL_EVALUATOR_SHA256
    )
    assert (
        bootstrap._apply_controlled_evaluator_headless_compatibility(ROOT)
        == evaluator.resolve()
    )


def test_vehicle_pretrain_stages_forbid_checkpoint_handoffs() -> None:
    bootstrap, vehicle = _bootstrap_args("--stage", "vehicle")
    bootstrap._validate_arguments(vehicle)
    _, vehicle_teacher = _bootstrap_args("--stage", "vehicle_teacher")
    bootstrap._validate_arguments(vehicle_teacher)

    _, invalid = _bootstrap_args(
        "--stage",
        "vehicle_teacher",
        "--teacher-checkpoint",
        "teacher.pth",
    )
    with pytest.raises(ValueError, match="forbids checkpoint handoff"):
        bootstrap._validate_arguments(invalid)


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

    _, valid_primary = _bootstrap_args(
        "--experiment-from-task",
        "resilient_v2x",
        "--teacher-task-id",
        MAIN_TASK_ID,
        "--predecessor-task-id",
        MAIN_TASK_ID,
    )
    bootstrap._validate_arguments(valid_primary)

    _, primary_without_teacher = _bootstrap_args(
        "--experiment-from-task",
        "resilient_v2x",
        "--predecessor-task-id",
        MAIN_TASK_ID,
    )
    with pytest.raises(ValueError, match="teacher-dependent"):
        bootstrap._validate_arguments(primary_without_teacher)

    _, valid_shared_initialization = _bootstrap_args(
        "--experiment-from-task",
        "no_distillation",
        "--teacher-task-id",
        MAIN_TASK_ID,
        "--predecessor-task-id",
        MAIN_TASK_ID,
    )
    bootstrap._validate_arguments(valid_shared_initialization)

    _, missing_shared_initialization = _bootstrap_args(
        "--experiment-from-task",
        "no_distillation",
        "--predecessor-task-id",
        MAIN_TASK_ID,
    )
    with pytest.raises(ValueError, match="teacher-dependent"):
        bootstrap._validate_arguments(missing_shared_initialization)


def test_training_commands_are_4gpu_ddp_and_training_only(tmp_path: Path) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    command = bootstrap._ddp_training_command(
        Path("/venv/python"),
        gpus=4,
        config=tmp_path / "config.py",
        work_dir=tmp_path / "work",
        max_epochs=50,
        training_seed=7,
    )
    assert "--nproc_per_node=4" in command
    assert command[command.index("--module") + 1 :][0] == (
        "tools.resilient_v2x.run_deterministic"
    )
    assert "tools/train.py" in command
    assert "tools/test.py" not in command
    assert not any("conditions/" in value for value in command)
    assert "train_cfg.val_interval=10" in command
    assert set(bootstrap.RTX5090_HEADLESS_CFG_OPTIONS) <= set(command)
    assert "train_dataloader.batch_size=2" in command
    assert "val_dataloader.batch_size=4" in command
    assert "test_dataloader.batch_size=4" in command
    assert "find_unused_parameters=True" in command
    assert "randomness.seed=7" in command
    assert "train_dataloader.sampler.seed=7" in command
    assert "train_dataloader.dataset.seed=7" in command
    assert "implementation_choices_dataset.global_seed=7" in command
    assert set(bootstrap.EXPERIMENT_CHECKPOINT_CFG_OPTIONS) <= set(command)


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
        "seed": 7,
        "training_index_protocol_seed": 20250218,
        "resume": False,
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    command = bootstrap._baseline_plan_command(
        Path("/venv/python"),
        source_root=ROOT,
        baseline="v2x_vit",
        training_index=tmp_path / "training_overlays.json",
        work_dir=work_dir,
        training_seed=7,
    )
    assert "torch.distributed.run" not in command
    assert command[-1] == "--dry-run"
    observed_plan, observed_config, _ = bootstrap._validate_baseline_dry_run(
        spec=bootstrap.EXPERIMENT_BY_NAME["v2x_vit"],
        work_dir=work_dir,
        training_seed=7,
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
    overlay_dir = tmp_path / "protocols/dair_v2"
    overlay_dir.mkdir(parents=True)
    (overlay_dir / "training_overlays.json").write_text(
        json.dumps({"protocol_seed": 20250218}),
        encoding="utf-8",
    )
    command = ["/venv/python", "-m", "torch.distributed.run", "tools/train.py"]
    contract = bootstrap._experiment_run_contract(
        args,
        task_id="c" * 32,
        spec=bootstrap.EXPERIMENT_BY_NAME["ptf_none"],
        dataset_root=tmp_path,
        teacher_contract={"model_id": "b" * 32, "sha256": SHA},
        predecessor_task_id=MAIN_TASK_ID,
        config_path=config,
        training_command=command,
        baseline_plan=None,
    )
    assert contract["training_command"] == command
    assert (
        contract["config"]["config_sha256"]
        == hashlib.sha256(config.read_bytes()).hexdigest()
    )
    assert (
        contract["config"]["resolved_config_sha256"]
        == hashlib.sha256(config.read_bytes()).hexdigest()
    )
    assert contract["source_archive"]["sha256"] == SHA
    assert contract["predecessor_task_id"] == MAIN_TASK_ID
    assert contract["gpus"] == contract["ddp_processes"] == 4
    assert contract["seed"] == contract["training_seed"] == 20250218
    assert contract["training_overlay_protocol_seed"] == 20250218
    assert contract["learning_rate"] == 0.0001
    assert contract["auto_scale_lr"] is False
    assert contract["global_batch_size"] == 8
    assert contract["train_batch_size_per_gpu"] == 2
    assert contract["eval_batch_size_per_gpu"] == 4
    assert contract["precision"] == "FP32"
    assert contract["val_interval"] == 10
    assert bootstrap.RTX5090_VAL_INTERVAL == 10
    assert contract["per_epoch_validation"] is False
    assert contract["condition_evaluation"] is False
    assert contract["common_teacher_initialization"] == {
        "policy": "shared-only",
        "contract": "shared-only-clean-teacher-initialization-v1",
        "shared_prefixes": [
            "lidar_encoder.",
            "camera_encoder.",
            "bbox_head.",
            "detection_projection.",
        ],
        "expected_source_keys": 617,
        "expected_source_numel": 35_811_485,
        "expected_source_bytes": 143_246_244,
        "expected_shared_keys": 468,
        "expected_shared_numel": 31_506_934,
        "expected_shared_bytes": 126_028_040,
        "expected_teacher_fusion_keys": 149,
        "teacher_checkpoint_sha256": SHA,
        "audit_artifact_name": "common_teacher_initialization_audit",
        "audit_filename": "common_teacher_initialization_audit.json",
        "expected_nested_teacher": True,
    }

    support_args = _bootstrap_args(
        "--experiment-from-task",
        "support_residual",
        "--teacher-task-id",
        MAIN_TASK_ID,
        "--predecessor-task-id",
        MAIN_TASK_ID,
    )[1]
    support_contract = bootstrap._experiment_run_contract(
        support_args,
        task_id="d" * 32,
        spec=bootstrap.EXPERIMENT_BY_NAME["support_residual"],
        dataset_root=tmp_path,
        teacher_contract={"model_id": "b" * 32, "sha256": SHA},
        predecessor_task_id=MAIN_TASK_ID,
        config_path=config,
        training_command=command,
        baseline_plan=None,
    )
    assert (
        support_contract["common_teacher_initialization"]["expected_nested_teacher"]
        is True
    )

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        bootstrap._experiment_run_contract(
            args,
            task_id="e" * 32,
            spec=bootstrap.EXPERIMENT_BY_NAME["ptf_none"],
            dataset_root=tmp_path,
            teacher_contract={"model_id": "b" * 32},
            predecessor_task_id=MAIN_TASK_ID,
            config_path=config,
            training_command=command,
            baseline_plan=None,
        )


@pytest.mark.parametrize("subject", FORMAL_CONFIG_SUBJECTS)
def test_nested_teacher_expectation_matches_real_resolved_config(subject: str) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    evaluator = _load_script("evaluate_controlled_baselines.py")
    assert set(FORMAL_CONFIG_SUBJECTS) == set(bootstrap.EXPERIMENT_ORDER)
    spec = bootstrap.EXPERIMENT_BY_NAME[subject]
    relative_config = spec.config or f"configs/resilient_v2x/baselines/{subject}.py"
    resolved = evaluator._load_python_config(ROOT / relative_config)
    model = resolved.get("model")
    assert isinstance(model, dict)
    actual_nested_teacher = model.get("teacher") is not None
    assert actual_nested_teacher is (subject in bootstrap.NESTED_TEACHER_EXPERIMENTS)
    assert actual_nested_teacher is (subject in EXPECTED_NESTED_TEACHER_SUBJECTS)


@pytest.mark.parametrize(
    ("nested_teacher", "fusion_keys"),
    ((False, 0), (False, 7), (True, 7)),
)
def test_common_teacher_initialization_audit_accepts_exact_role_schema(
    tmp_path: Path,
    nested_teacher: bool,
    fusion_keys: int,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    assert bootstrap.EMPTY_TENSOR_MAPPING_SHA256 == EMPTY_MAPPING_SHA
    audit_path = tmp_path / bootstrap.COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME
    audit_path.write_text(
        json.dumps(
            _common_teacher_initialization_audit(
                nested_teacher=nested_teacher,
                fusion_keys=fusion_keys,
            )
        ),
        encoding="utf-8",
    )

    assert (
        bootstrap._validate_common_teacher_initialization_audit(
            audit_path,
            expected_teacher_sha256=SHA,
            expect_nested_teacher=nested_teacher,
        )
        == audit_path.resolve()
    )


@pytest.mark.parametrize(
    ("field_path", "bad_value", "message"),
    (
        (("result",), "fail", r"audit.result mismatch"),
        (("checkpoint", "sha256"), "e" * 64, r"checkpoint SHA-256"),
        (("source", "keys"), 616, r"source.keys mismatch"),
        (("source", "numel"), 35_811_484, r"source.numel mismatch"),
        (
            ("shared_initialization", "prefixes"),
            ["lidar_encoder."],
            r"shared_initialization.prefixes mismatch",
        ),
        (
            ("shared_initialization", "keys"),
            467,
            r"shared_initialization.keys mismatch",
        ),
        (
            ("method_specific_fusion", "unchanged"),
            False,
            r"method_specific_fusion.unchanged mismatch",
        ),
        (
            ("method_specific_fusion", "keys"),
            -1,
            r"method_specific_fusion.keys must be a non-negative integer",
        ),
        (
            ("method_specific_fusion", "keys"),
            0,
            r"zero-state method-specific fusion",
        ),
        (
            ("method_specific_fusion", "numel"),
            0,
            r"non-empty method-specific fusion",
        ),
        (
            ("method_specific_fusion", "sha256_after"),
            "e" * 64,
            r"fusion changed",
        ),
        (
            ("target", "nested_teacher_present"),
            True,
            r"target.nested_teacher_present mismatch",
        ),
        (
            ("target", "target_key_count"),
            1,
            r"target.target_key_count mismatch",
        ),
    ),
)
def test_common_teacher_initialization_audit_rejects_contract_drift(
    tmp_path: Path,
    field_path: tuple[str, ...],
    bad_value: object,
    message: str,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    audit = _common_teacher_initialization_audit(nested_teacher=False)
    cursor = audit
    for field in field_path[:-1]:
        nested = cursor[field]
        assert isinstance(nested, dict)
        cursor = nested
    cursor[field_path[-1]] = bad_value
    audit_path = tmp_path / bootstrap.COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        bootstrap._validate_common_teacher_initialization_audit(
            audit_path,
            expected_teacher_sha256=SHA,
            expect_nested_teacher=False,
        )


@pytest.mark.parametrize(
    "failure",
    ("nonzero_numel", "nonempty_digest", "target_fusion_count", "target_key_count"),
)
def test_zero_state_fusion_audit_rejects_internal_inconsistency(
    tmp_path: Path,
    failure: str,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    audit = _common_teacher_initialization_audit(
        nested_teacher=False,
        fusion_keys=0,
    )
    fusion = audit["method_specific_fusion"]
    target = audit["target"]
    assert isinstance(fusion, dict)
    assert isinstance(target, dict)
    if failure == "nonzero_numel":
        fusion["numel"] = 1
        match = "keys=numel=bytes=0"
    elif failure == "nonempty_digest":
        fusion["sha256_before"] = fusion["sha256_after"] = "d" * 64
        match = "canonical empty-mapping SHA-256"
    elif failure == "target_fusion_count":
        target["target_fusion_key_count"] = 1
        match = "target_fusion_key_count mismatch"
    else:
        target["target_key_count"] = 469
        match = "target_key_count mismatch"
    audit_path = tmp_path / bootstrap.COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        bootstrap._validate_common_teacher_initialization_audit(
            audit_path,
            expected_teacher_sha256=SHA,
            expect_nested_teacher=False,
        )


def test_common_teacher_initialization_audit_requires_exact_complete_schema(
    tmp_path: Path,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    missing_path = tmp_path / bootstrap.COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME
    with pytest.raises(FileNotFoundError, match="audit is missing"):
        bootstrap._validate_common_teacher_initialization_audit(
            missing_path,
            expected_teacher_sha256=SHA,
            expect_nested_teacher=False,
        )

    audit = _common_teacher_initialization_audit(nested_teacher=False)
    shared = audit["shared_initialization"]
    assert isinstance(shared, dict)
    del shared["bytes"]
    missing_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match=r"schema mismatch: missing=\['bytes'\]"):
        bootstrap._validate_common_teacher_initialization_audit(
            missing_path,
            expected_teacher_sha256=SHA,
            expect_nested_teacher=False,
        )

    audit = _common_teacher_initialization_audit(nested_teacher=False)
    audit["unexpected"] = True
    missing_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match=r"extra=\['unexpected'\]"):
        bootstrap._validate_common_teacher_initialization_audit(
            missing_path,
            expected_teacher_sha256=SHA,
            expect_nested_teacher=False,
        )


def test_common_teacher_initialization_audit_uploads_then_flushes(
    tmp_path: Path,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    audit_path = work_dir / bootstrap.COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME
    audit_path.write_text(
        json.dumps(
            _common_teacher_initialization_audit(
                nested_teacher=False,
                fusion_keys=0,
            )
        ),
        encoding="utf-8",
    )

    class Task:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def upload_artifact(self, name: str, **kwargs) -> bool:
            self.events.append(("upload", (name, kwargs)))
            return True

        def flush(self, **kwargs) -> None:
            self.events.append(("flush", kwargs))

    task = Task()
    assert (
        bootstrap._validate_and_upload_common_teacher_initialization_audit(
            task,
            work_dir=work_dir,
            expected_teacher_sha256=SHA,
            expect_nested_teacher=False,
        )
        == audit_path.resolve()
    )
    assert task.events == [
        (
            "upload",
            (
                "common_teacher_initialization_audit",
                {
                    "artifact_object": str(audit_path.resolve()),
                    "wait_on_upload": True,
                },
            ),
        ),
        ("flush", {"wait_for_uploads": True}),
    ]

    class FailingTask:
        def __init__(self) -> None:
            self.flushed = False

        def upload_artifact(self, *_args, **_kwargs) -> bool:
            return False

        def flush(self, **_kwargs) -> None:
            self.flushed = True

    failing_task = FailingTask()
    with pytest.raises(RuntimeError, match="failed to upload"):
        bootstrap._validate_and_upload_common_teacher_initialization_audit(
            failing_task,
            work_dir=work_dir,
            expected_teacher_sha256=SHA,
            expect_nested_teacher=False,
        )
    assert failing_task.flushed is False


def test_experiment_validates_audit_between_training_and_final_model_upload() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    source = inspect.getsource(bootstrap._execute_experiment_from_task)
    training = source.index("_run_logged(training_command")
    audit = source.index("_validate_and_upload_common_teacher_initialization_audit(")
    final_model = source.index("_upload_experiment_checkpoint(")
    assert training < audit < final_model


def test_evaluator_compatibility_template_includes_all_improvements() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    source = inspect.getsource(
        bootstrap._apply_controlled_evaluator_headless_compatibility
    )
    assert "IMPROVEMENTS = (" in source
    assert '"support_residual"' in source
    assert '"linear_no_distillation"' in source
    assert '"no_distillation_peak_lr_3e4"' in source
    assert "EVALUATION_SUBJECTS = BASELINES + ABLATIONS + IMPROVEMENTS" in source


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
    assert (
        contract is not None
        and contract["sha256"] == hashlib.sha256(b"teacher").hexdigest()
    )

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


def test_every_formal_experiment_requires_shared_teacher_initialization() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    assert bootstrap.TEACHER_DEPENDENT_EXPERIMENTS == set(bootstrap.EXPERIMENT_ORDER)


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


def test_clean_val_best_checkpoint_is_resolved_and_uploaded_as_diagnostic(
    tmp_path: Path,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    checkpoint = tmp_path / "best_resilient_v2x_car_bev_ap_r40_0.70_epoch_30.pth"
    checkpoint.write_bytes(b"best checkpoint")
    resolved, epoch = bootstrap._resolve_clean_val_best_checkpoint(tmp_path)
    assert resolved == checkpoint.resolve()
    assert epoch == 30

    class OutputModel:
        instances = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.id = "e" * 32
            self.update_kwargs = None
            self.instances.append(self)

        def update_weights(self, **kwargs):
            self.update_kwargs = kwargs
            return "http://10.100.34.118:8081/models/best.pth"

    contract = bootstrap._upload_experiment_best_checkpoint(
        task=object(),
        output_model_class=OutputModel,
        spec=bootstrap.EXPERIMENT_BY_NAME["linear_no_distillation"],
        checkpoint=checkpoint,
        epoch=epoch,
    )
    instance = OutputModel.instances[0]
    assert "not-final-claim" in instance.kwargs["tags"]
    assert instance.update_kwargs["iteration"] == 30
    assert instance.update_kwargs["target_filename"] == (
        "linear_no_distillation_clean_val_best_epoch_30.pth"
    )
    assert contract["selection_protocol"] == "DAIR-CLEAN-PAIR1789-v1"
    assert contract["claim_role"].startswith("diagnostic")


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
    assert all(
        step["output_uri"] == module.FILES_SERVER_URI for step in controller.steps
    )
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
    parameters = {step["experiment"]: dict(step["parameter_override"]) for step in plan}
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
        url=("http://10.100.34.118:8081/models/teacher_epoch_50.pth"),
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
