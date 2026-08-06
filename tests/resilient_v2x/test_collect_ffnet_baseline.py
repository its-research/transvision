from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/collect_ffnet_baseline.py"


def _module():
    spec = importlib.util.spec_from_file_location("collect_ffnet_baseline", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_line(module, offset: float = 0.0) -> str:
    return "  ".join(
        f"resilient_v2x/{name}: {index + offset:.4f}"
        for index, name in enumerate(module.REQUIRED_METRICS, start=1)
    )


def test_extract_final_metrics_requires_epoch_40_and_uses_last_validation() -> None:
    module = _module()
    reports = [
        "Epoch(train) [39][4823/4823]",
        _metric_line(module, 10.0),
        "Epoch(train) [40][4823/4823] loss: 0.1",
        _metric_line(module, 20.0),
        _metric_line(module, 30.0),
    ]

    metrics, source = module._extract_final_metrics(reports)

    assert metrics[module.REQUIRED_METRICS[0]] == 31.0
    assert metrics[module.REQUIRED_METRICS[-1]] == 45.0
    assert "45.0000" in source


def test_extract_final_metrics_rejects_incomplete_validation() -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="incomplete"):
        module._extract_final_metrics(
            [
                "Epoch(train) [40][4823/4823]",
                "resilient_v2x/sample_count: 1789.0000",
            ]
        )


def test_run_contract_is_exact_for_official_single_gpu_recipe() -> None:
    module = _module()
    task_id = "a" * 32
    contract = {
        "task_id": task_id,
        "stage": "ffnet_official_train",
        "dataset_id": module.EXPECTED_DATASET_ID,
        "training_world_size": 1,
        "global_batch_size": 2,
        "train_batch_size_per_gpu": 2,
        "eval_batch_size_per_gpu": 4,
        "max_epochs": 40,
        "expected_optimizer_steps": 192920,
        "checkpoint_policy": "final_epoch",
        "learning_rate": 0.001,
        "amp": False,
        "auto_scale_lr": False,
        "runtime_profile": "rtx5090",
    }

    assert module._require_run_contract(contract, task_id) == contract
    contract["global_batch_size"] = 8
    with pytest.raises(RuntimeError, match="global_batch_size mismatch"):
        module._require_run_contract(contract, task_id)


def test_collector_refuses_an_incomplete_task(tmp_path: Path) -> None:
    module = _module()

    class Task:
        id = "b" * 32

        @staticmethod
        def get_status() -> str:
            return "in_progress"

    with pytest.raises(RuntimeError, match="not completed"):
        module.collect(
            Task(),
            task_id=Task.id,
            out_dir=tmp_path,
            console_reports=10,
        )


def test_task_parameters_seal_source_and_native_build_identity() -> None:
    module = _module()

    class Task:
        @staticmethod
        def get_parameters(**_kwargs):
            return dict(module.EXPECTED_TASK_PARAMETERS)

    assert module._require_task_parameters(Task()) == module.EXPECTED_TASK_PARAMETERS

    class DriftedTask:
        @staticmethod
        def get_parameters(**_kwargs):
            parameters = dict(module.EXPECTED_TASK_PARAMETERS)
            parameters["Args/source_archive_sha256"] = "0" * 64
            return parameters

    with pytest.raises(RuntimeError, match="source_archive_sha256 mismatch"):
        module._require_task_parameters(DriftedTask())
