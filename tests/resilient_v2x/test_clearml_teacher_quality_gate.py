from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_teacher_quality_gate.py"
TEACHER_TASK_ID = "a" * 32
MODEL_ID = "b" * 32
OUTPUT_TASK_ID = "c" * 32
DATASET_ID = "d" * 32


def _verify_document(
    document: dict[str, object],
    *,
    expected_type: str,
) -> dict[str, object]:
    assert document["schema_version"] == 1
    assert document["document_type"] == expected_type
    observed = document["content_sha256"]
    unhashed = dict(document)
    unhashed.pop("content_sha256")
    canonical = json.dumps(
        unhashed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert observed == hashlib.sha256(canonical).hexdigest()
    return dict(document)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "clearml_teacher_quality_gate",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Artifact:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
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
    ) -> None:
        self.name = name
        self.task = task
        self.id = model_id
        self.url = url
        self.local_copy = local_copy
        self.download_calls: list[dict[str, object]] = []

    def get_local_copy(self, **kwargs: object) -> str:
        self.download_calls.append(dict(kwargs))
        return str(self.local_copy)


class _TeacherTask:
    def __init__(
        self,
        *,
        status: str,
        parameters: dict[str, object],
        artifacts: dict[str, _Artifact],
        model: _Model,
        console: list[str],
        reload_statuses: list[str] | None = None,
    ) -> None:
        self.id = TEACHER_TASK_ID
        self.status = status
        self.parameters = parameters
        self.artifacts = artifacts
        self.model = model
        self.console = console
        self.data = SimpleNamespace(
            script=SimpleNamespace(to_dict=lambda: {"diff": "sealed-task-script"})
        )
        self.reload_statuses = list(reload_statuses or [])
        self.reload_calls = 0
        self.console_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def reload(self) -> None:
        self.reload_calls += 1
        if self.reload_statuses:
            self.status = self.reload_statuses.pop(0)

    def get_parameters(self, **_kwargs: object) -> dict[str, object]:
        return dict(self.parameters)

    def get_models(self) -> dict[str, list[object]]:
        return {"input": [], "output": [self.model]}

    def get_reported_console_output(
        self,
        *args: object,
        **kwargs: object,
    ) -> list[str]:
        self.console_calls.append((args, dict(kwargs)))
        return list(self.console)


class _OutputTask:
    def __init__(self) -> None:
        self.id = OUTPUT_TASK_ID
        self.artifacts: dict[str, _Artifact] = {}
        self.tags: list[str] = []
        self.uploads: list[tuple[str, object, bool]] = []
        self.flush_calls = 0

    def set_tags(self, tags: list[str]) -> None:
        self.tags = list(tags)

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        self.uploads.append((name, artifact_object, wait_on_upload))
        self.artifacts[name] = _Artifact(artifact_object)
        return True

    def flush(self, *, wait_for_uploads: bool) -> None:
        assert wait_for_uploads is True
        self.flush_calls += 1


class _Tasks:
    def __init__(self, teacher: _TeacherTask, output: _OutputTask) -> None:
        self.teacher = teacher
        self.output = output

    def get_task(self, *, task_id: str) -> _TeacherTask:
        assert task_id == TEACHER_TASK_ID
        return self.teacher

    def current_task(self) -> _OutputTask:
        return self.output


def _run_contract(module, *, seeded: bool = False) -> dict[str, object]:
    contract = {
        "task_id": TEACHER_TASK_ID,
        "dataset_id": DATASET_ID,
        "dataset_local_copy": "/clearml/cache/training-dataset",
        "gpus": 4,
        "training_world_size": 4,
        "global_batch_size": 8,
        "train_batch_size_per_gpu": 2,
        "eval_batch_size_per_gpu": 4,
        "vehicle_global_batch_size": 8,
        "vehicle_train_batch_size_per_gpu": 2,
        "vehicle_eval_batch_size_per_gpu": 4,
        "learning_rate": 0.0001,
        "auto_scale_lr": False,
        "expected_optimizer_steps": None,
        "max_epochs": 50,
        "val_interval": 10,
        "amp": False,
        "runtime_profile": "rtx5090",
        "python_safe_path": "1",
        "checkpoint_policy": module.CLEAN_TEACHER_CHECKPOINT_POLICY,
        "manifest_content_sha256": module.EXPECTED_MANIFEST_CONTENT_SHA256,
        "split_sha256": module.OFFICIAL_SPLIT_SHA256,
        "evaluation_index_content_sha256": (
            module.EXPECTED_EVALUATION_INDEX_CONTENT_SHA256
        ),
        "evaluation_sample_ids_sha256": (module.EXPECTED_EVALUATION_SAMPLE_IDS_SHA256),
        "evaluation_sample_count": module.EXPECTED_EVALUATION_SAMPLE_COUNT,
        "evaluation_condition_count": (module.EXPECTED_EVALUATION_CONDITION_COUNT),
        "output_uri_scheme": "http",
        "stage": "teacher",
    }
    if seeded:
        contract.update(
            {
                "training_seed": module.DEFAULT_TRAINING_SEED,
                "training_overlay_protocol_seed": (
                    module.TRAINING_OVERLAY_PROTOCOL_SEED
                ),
                "seed": module.DEFAULT_TRAINING_SEED,
            }
        )
        assert set(contract) == module.SEEDED_RUN_CONTRACT_KEYS
    else:
        assert set(contract) == module.LEGACY_SOURCE_C_RUN_CONTRACT_KEYS
    return contract


def _metrics_by_epoch() -> dict[int, tuple[float, float, float, float]]:
    return {
        10: (68.0, 60.8, 58.0, 38.1),
        20: (69.0, 61.2, 59.0, 38.5),
        30: (70.0, 62.0, 60.0, 40.0),
        40: (69.5, 61.8, 59.5, 39.8),
        50: (69.0, 61.0, 59.0, 39.0),
    }


def _console_lines(
    values: dict[int, tuple[float, float, float, float]] | None = None,
) -> list[str]:
    values = values or _metrics_by_epoch()
    result = ["Epoch(train) [1][20/603] loss: 2.0"]
    for epoch, (bev50, bev70, three_d50, three_d70) in values.items():
        result.append(f"Epoch(val) [{epoch}][20/448] time: 0.2")
        result.append(
            "\x1b[32mEpoch(val) "
            f"[{epoch}][448/448]    "
            "resilient_v2x/sample_count: 1789.0000  "
            "resilient_v2x/car_ground_truth_count: 15337.0000  "
            f"resilient_v2x/car_bev_ap_r40_0.50: {bev50:.4f}  "
            f"resilient_v2x/car_bev_ap_r40_0.70: {bev70:.4f}  "
            f"resilient_v2x/car_3d_ap_r40_0.50: {three_d50:.4f}  "
            f"resilient_v2x/car_3d_ap_r40_0.70: {three_d70:.4f}  "
            "resilient_v2x/unsupported_sample_count: 0.0000  "
            "time: 0.3\x1b[0m"
        )
    return result


def _checkpoint_contract(
    module,
    *,
    selected_epoch: int,
    checkpoint: Path,
    model_url: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "selection_protocol": module.CLEAN_TEACHER_SELECTION_PROTOCOL,
        "selection_metric": module.CLEAN_TEACHER_SELECTION_METRIC,
        "selection_rule": module.CLEAN_TEACHER_SELECTION_RULE,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": {
            "model_id": MODEL_ID,
            "name": module.CLEAN_TEACHER_MODEL_NAME,
            "url": model_url,
            "filename": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "trained_epochs": 50,
        "final_epoch": 50,
        "final_checkpoint": {
            "filename": "teacher_epoch_50.pth",
            "size_bytes": 123,
            "sha256": "f" * 64,
        },
        "downstream_role": ("frozen teacher and trainable student initialization"),
    }


def _fixture(
    tmp_path: Path,
    *,
    selected_epoch: int = 30,
    metrics: dict[int, tuple[float, float, float, float]] | None = None,
    status: str = "completed",
    reload_statuses: list[str] | None = None,
):
    module = _load_module()
    module.EXPECTED_TEACHER_SCRIPT_DIFF_SHA256 = hashlib.sha256(
        b"sealed-task-script"
    ).hexdigest()
    filename = (
        f"best_resilient_v2x_car_bev_ap_r40_0.70_teacher_epoch_{selected_epoch}.pth"
    )
    checkpoint = tmp_path / filename
    checkpoint.write_bytes(b"controlled-clean-teacher")
    model_url = f"http://10.100.34.118:8081/models/{filename}"
    model = _Model(
        name=module.CLEAN_TEACHER_MODEL_NAME,
        task=TEACHER_TASK_ID,
        model_id=MODEL_ID,
        url=model_url,
        local_copy=checkpoint,
    )
    parameters = {
        "Args/source_dataset_id": module.EXPECTED_SOURCE_DATASET_ID,
        "Args/source_archive_name": module.EXPECTED_SOURCE_ARCHIVE_NAME,
        "Args/source_archive_bytes": str(module.EXPECTED_SOURCE_ARCHIVE_BYTES),
        "Args/source_archive_sha256": module.EXPECTED_SOURCE_ARCHIVE_SHA256,
        "Args/training_dataset_id": DATASET_ID,
        "Args/native_bundle_bytes": str(module.EXPECTED_NATIVE_BUNDLE_BYTES),
        "Args/native_bundle_sha256": module.EXPECTED_NATIVE_BUNDLE_SHA256,
        "Args/build_manifest_sha256": module.EXPECTED_BUILD_MANIFEST_SHA256,
        "Args/stage": "teacher",
        "Args/max_epochs": "50",
        "Args/gpus": 4,
        "Args/amp": "False",
    }
    module.EXPECTED_TRAINING_DATASET_ID = DATASET_ID
    artifacts = {
        module.RUN_CONTRACT_ARTIFACT: _Artifact(_run_contract(module)),
        module.TEACHER_CHECKPOINT_ARTIFACT: _Artifact(
            _checkpoint_contract(
                module,
                selected_epoch=selected_epoch,
                checkpoint=checkpoint,
                model_url=model_url,
            )
        ),
    }
    teacher = _TeacherTask(
        status=status,
        parameters=parameters,
        artifacts=artifacts,
        model=model,
        console=_console_lines(metrics),
        reload_statuses=reload_statuses,
    )
    output = _OutputTask()
    tasks = _Tasks(teacher, output)
    args = SimpleNamespace(
        teacher_task_id=TEACHER_TASK_ID,
        min_bev_ap70=module.DEFAULT_MIN_BEV_AP70,
        min_3d_ap70=module.DEFAULT_MIN_3D_AP70,
        poll_seconds=1.0,
        timeout_hours=1.0,
    )
    return module, teacher, output, tasks, args


def test_parser_uses_ffnet_noninferiority_defaults() -> None:
    module = _load_module()
    args = module._parser().parse_args(["--teacher-task-id", TEACHER_TASK_ID])
    assert args.min_bev_ap70 == 59.6257
    assert args.min_3d_ap70 == 30.0


def test_gate_script_has_no_runtime_transvision_import() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.startswith("transvision") for name in imported)


def test_console_reader_supports_clearml_single_argument_api() -> None:
    module = _load_module()

    class SingleArgumentConsoleTask:
        calls: list[int] = []

        def get_reported_console_output(
            self,
            number_of_reports: int,
        ) -> list[str]:
            self.calls.append(number_of_reports)
            return ["one", "two"]

    task = SingleArgumentConsoleTask()
    assert module._console_chunks(task) == ["one", "two"]
    assert task.calls == [module.CONSOLE_REPORT_LIMIT]


def test_gate_waits_validates_downloads_and_publishes_sealed_json(
    tmp_path: Path,
) -> None:
    module, teacher, output, tasks, args = _fixture(
        tmp_path,
        status="queued",
        reload_statuses=["in_progress", "completed"],
    )
    sleeps: list[float] = []
    payload = module.run(
        args,
        task_class=tasks,
        output_task=output,
        monotonic=lambda: 0.0,
        sleeper=sleeps.append,
    )

    assert sleeps == [1.0]
    assert teacher.reload_calls == 2
    assert teacher.model.download_calls == [
        {
            "extract_archive": False,
            "raise_on_error": True,
            "force_download": True,
        }
    ]
    assert teacher.console_calls == [
        (
            (),
            {
                "number_of_reports": module.CONSOLE_REPORT_LIMIT,
                "max_line_length": module.CONSOLE_LINE_LIMIT,
                "order": "asc",
            },
        )
    ]
    assert payload["passed"] is True
    assert payload["run_contract_schema"] == "legacy_source_c"
    assert payload["training_seed_evidence"] == "legacy_fixed_by_sealed_source"
    assert payload["validation_count"] == 5
    assert payload["expected_validation_epochs"] == [10, 20, 30, 40, 50]
    assert payload["best"]["selected_epoch"] == 30
    assert payload["best"]["selected_metrics"] == {
        "car_bev_ap_r40_0.50": 70.0,
        "car_bev_ap_r40_0.70": 62.0,
        "car_3d_ap_r40_0.50": 60.0,
        "car_3d_ap_r40_0.70": 40.0,
    }
    verified = _verify_document(
        payload,
        expected_type=module.QUALITY_GATE_DOCUMENT_TYPE,
    )
    assert verified == payload
    assert len(output.uploads) == 1
    assert output.uploads[0] == (module.QUALITY_GATE_ARTIFACT, payload, True)
    assert output.flush_calls == 1
    assert "teacher-quality-gate" in output.tags


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (1, 59.0, "BEV AP70 failed"),
        (3, 29.5, "3D AP70 failed"),
    ),
)
def test_gate_rejects_selected_checkpoint_below_either_threshold(
    tmp_path: Path,
    field: int,
    value: float,
    message: str,
) -> None:
    metrics = _metrics_by_epoch()
    selected = list(metrics[30])
    selected[field] = value
    metrics[30] = tuple(selected)  # type: ignore[assignment]
    if field == 1:
        for epoch in metrics:
            row = list(metrics[epoch])
            row[1] = min(row[1], value)
            metrics[epoch] = tuple(row)  # type: ignore[assignment]
    module, _teacher, output, tasks, args = _fixture(
        tmp_path,
        metrics=metrics,
    )
    with pytest.raises(RuntimeError, match=message):
        module.run(args, task_class=tasks, output_task=output)
    failure = output.uploads[-1][1]
    assert failure["passed"] is False
    assert failure["validation_count"] == 5
    assert failure["error_type"] == "RuntimeError"


def test_gate_rejects_checkpoint_epoch_that_is_not_bev70_best(
    tmp_path: Path,
) -> None:
    metrics = _metrics_by_epoch()
    metrics[40] = (70.5, 63.0, 60.5, 40.5)
    module, _teacher, output, tasks, args = _fixture(
        tmp_path,
        metrics=metrics,
    )
    with pytest.raises(RuntimeError, match="selected_epoch does not match"):
        module.run(args, task_class=tasks, output_task=output)
    assert output.uploads[-1][1]["passed"] is False


def test_bev70_best_alignment_allows_four_decimal_log_rounding(
    tmp_path: Path,
) -> None:
    metrics = _metrics_by_epoch()
    metrics[30] = (70.0, 62.0000, 60.0, 40.0)
    metrics[40] = (70.1, 62.0001, 60.1, 40.1)
    module, _teacher, output, tasks, args = _fixture(
        tmp_path,
        metrics=metrics,
    )
    payload = module.run(args, task_class=tasks, output_task=output)
    assert payload["best"]["rounding_equivalent_best_epochs"] == [30, 40]


def test_gate_requires_all_five_scheduled_terminal_validations(
    tmp_path: Path,
) -> None:
    metrics = _metrics_by_epoch()
    del metrics[20]
    module, _teacher, output, tasks, args = _fixture(
        tmp_path,
        metrics=metrics,
    )
    with pytest.raises(RuntimeError, match=r"missing=\[20\]"):
        module.run(args, task_class=tasks, output_task=output)
    assert output.uploads[-1][1]["passed"] is False


def test_gate_rejects_conflicting_duplicate_validation_lines(
    tmp_path: Path,
) -> None:
    module, teacher, output, tasks, args = _fixture(tmp_path)
    duplicate = _console_lines({30: (70.0, 61.9, 60.0, 40.0)})[-1]
    teacher.console.append(duplicate)
    with pytest.raises(RuntimeError, match="conflicting results"):
        module.run(args, task_class=tasks, output_task=output)
    assert output.uploads[-1][1]["passed"] is False


@pytest.mark.parametrize("contract_name", ("run", "checkpoint"))
def test_gate_rejects_contract_or_checkpoint_byte_drift(
    tmp_path: Path,
    contract_name: str,
) -> None:
    module, teacher, output, tasks, args = _fixture(tmp_path)
    if contract_name == "run":
        contract = teacher.artifacts[module.RUN_CONTRACT_ARTIFACT].value
        assert isinstance(contract, dict)
        contract["unexpected"] = True
        message = "run contract schema mismatch"
    else:
        contract = teacher.artifacts[module.TEACHER_CHECKPOINT_ARTIFACT].value
        assert isinstance(contract, dict)
        selected = contract["selected_checkpoint"]
        assert isinstance(selected, dict)
        selected["sha256"] = "0" * 64
        message = "checkpoint SHA-256 mismatch"
    with pytest.raises(RuntimeError, match=message):
        module.run(args, task_class=tasks, output_task=output)
    failure = output.uploads[-1][1]
    assert failure["passed"] is False
    assert failure["validation_count"] == (0 if contract_name == "run" else 5)


def test_seeded_run_contract_remains_supported(tmp_path: Path) -> None:
    module, teacher, output, tasks, args = _fixture(tmp_path)
    teacher.artifacts[module.RUN_CONTRACT_ARTIFACT] = _Artifact(
        _run_contract(module, seeded=True)
    )
    payload = module.run(args, task_class=tasks, output_task=output)
    assert payload["run_contract_schema"] == "seeded"
    assert payload["training_seed_evidence"] == "run_contract_explicit"
    assert payload["training_seed_binding"] == {
        "training_seed": 20250218,
        "training_overlay_protocol_seed": 20250218,
    }


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("15337.0000", "15336.0000", "car_ground_truth_count mismatch"),
        (
            "unsupported_sample_count: 0.0000",
            "unsupported_sample_count: 1.0000",
            "unsupported_sample_count mismatch",
        ),
    ),
)
def test_gate_rejects_clean_validation_count_drift(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    module, teacher, output, tasks, args = _fixture(tmp_path)
    teacher.console = [line.replace(old, new) for line in teacher.console]
    with pytest.raises(RuntimeError, match=message):
        module.run(args, task_class=tasks, output_task=output)
    assert output.uploads[-1][1]["passed"] is False


def test_gate_fails_immediately_if_teacher_does_not_complete(
    tmp_path: Path,
) -> None:
    module, _teacher, output, tasks, args = _fixture(
        tmp_path,
        status="failed",
    )
    with pytest.raises(RuntimeError, match="ended as 'failed'"):
        module.run(args, task_class=tasks, output_task=output)
    assert output.uploads == []
