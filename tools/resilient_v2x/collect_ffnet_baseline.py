#!/usr/bin/env python3
"""Collect a completed FFNet-B-F ClearML reproduction as sealed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit


TASK_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
FINAL_TRAIN_PATTERN = re.compile(r"Epoch\(train\)\s+\[40\]\[\s*4823/4823\]")
METRIC_PATTERN = re.compile(
    r"resilient_v2x/([a-z0-9_.]+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+))"
)
EXPECTED_MODEL_NAME = "FFNet-B-F official three-class single-GPU complemented baseline"
EXPECTED_FILES_HOST = "10.100.34.118"
EXPECTED_FILES_PORT = 8081
EXPECTED_DATASET_ID = "fc242933c3ac43c2b47aaa3bd7f4a920"
EXPECTED_TASK_PARAMETERS = {
    "Args/source_dataset_id": "6ab792bb6a944da4857448cb6f5c9774",
    "Args/source_archive_name": "resilient-v2x-source-db5ee26d85d0.tar.zst",
    "Args/source_archive_bytes": "1156241",
    "Args/source_archive_sha256": (
        "29120e9555e036355f55c27816a559020fb715b7a7202eef17492da472dccbcf"
    ),
    "Args/training_dataset_id": EXPECTED_DATASET_ID,
    "Args/native_bundle_bytes": "740957535",
    "Args/native_bundle_sha256": (
        "71a9fc1f481b313309ef76949b931078b133c35127a65614d3aa5647907c3d9a"
    ),
    "Args/build_manifest_sha256": (
        "ea46a43f341bdcc951f1de66ec15005e4d15577a929cff243136f3a4a2a119ed"
    ),
    "Args/stage": "ffnet_official_train",
    "Args/gpus": "4",
    "Args/max_epochs": "40",
    "Args/amp": "False",
}
REQUIRED_METRICS = (
    "sample_count",
    "car_ground_truth_count",
    "car_prediction_count",
    "car_bev_ap_r40_0.50",
    "car_bev_ap_r40_0.70",
    "car_3d_ap_r40_0.50",
    "car_3d_ap_r40_0.70",
    "diagnostic_pred_z_bottom_p50",
    "diagnostic_gt_z_bottom_p50",
    "diagnostic_pred_height_p50",
    "diagnostic_gt_height_p50",
    "diagnostic_bev_match_050_count",
    "diagnostic_bev_match_050_abs_z_error_p50",
    "diagnostic_bev_match_050_vertical_iou_p50",
    "diagnostic_bev_match_050_3d_iou_p50",
)
REPOSITORY_REFERENCE = {
    "car_3d_ap_r40_0.50": 55.48,
    "car_3d_ap_r40_0.70": 31.54,
    "car_bev_ap_r40_0.50": 63.15,
    "car_bev_ap_r40_0.70": 54.27,
}
PAPER_REFERENCE = {
    "car_3d_ap_r40_0.50": 55.81,
    "car_3d_ap_r40_0.70": 30.23,
    "car_bev_ap_r40_0.50": 63.54,
    "car_bev_ap_r40_0.70": 54.16,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--console-reports", type=int, default=5000)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _seal(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _write_json(path: Path, value: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    os.replace(temporary, path)
    return path.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _status(task: object) -> str:
    getter = getattr(task, "get_status", None)
    value = getter() if callable(getter) else getattr(task, "status", None)
    value = getattr(value, "value", value)
    return str(value).rsplit(".", 1)[-1].lower()


def _flatten_console(reports: Sequence[str]) -> list[str]:
    return [
        ANSI_PATTERN.sub("", line).strip()
        for report in reports
        for line in report.splitlines()
        if line.strip()
    ]


def _extract_final_metrics(reports: Sequence[str]) -> tuple[dict[str, float], str]:
    lines = _flatten_console(reports)
    final_train_indices = [
        index for index, line in enumerate(lines) if FINAL_TRAIN_PATTERN.search(line)
    ]
    if not final_train_indices:
        raise RuntimeError("console does not prove epoch 40 reached 4823/4823")
    final_train_index = final_train_indices[-1]
    candidates: list[tuple[dict[str, float], str]] = []
    for line in lines[final_train_index + 1 :]:
        values = {name: float(value) for name, value in METRIC_PATTERN.findall(line)}
        if values:
            candidates.append((values, line))
    if not candidates:
        raise RuntimeError("console has no validation metrics after epoch 40")
    metrics, source_line = candidates[-1]
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise RuntimeError(f"final validation metrics are incomplete: {missing}")
    return {name: metrics[name] for name in REQUIRED_METRICS}, source_line


def _require_run_contract(value: object, task_id: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("run_contract is not a JSON object")
    contract = dict(value)
    expected = {
        "task_id": task_id,
        "stage": "ffnet_official_train",
        "dataset_id": EXPECTED_DATASET_ID,
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
    for key, expected_value in expected.items():
        if contract.get(key) != expected_value:
            raise RuntimeError(
                f"run_contract {key} mismatch: "
                f"expected {expected_value!r}, got {contract.get(key)!r}"
            )
    return contract


def _artifact_mapping(task: object, name: str) -> dict[str, object]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"task is missing artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise RuntimeError(f"artifact {name!r} cannot be downloaded")
    value = getter()
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).resolve(strict=True)
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise RuntimeError(f"artifact {name!r} is not a JSON object")
    return dict(decoded)


def _require_task_parameters(task: object) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError("task cannot enumerate parameters")
    try:
        parameters = getter(backwards_compatibility=False, cast=False)
    except TypeError:
        parameters = getter()
    if not isinstance(parameters, Mapping):
        raise RuntimeError("task returned invalid parameters")
    for key, expected in EXPECTED_TASK_PARAMETERS.items():
        actual = parameters.get(key)
        if str(actual) != expected:
            raise RuntimeError(
                f"task parameter {key} mismatch: expected {expected!r}, got {actual!r}"
            )
    return {key: parameters[key] for key in EXPECTED_TASK_PARAMETERS}


def _require_unique_model(task: object, task_id: str) -> object:
    getter = getattr(task, "get_models", None)
    models = getter() if callable(getter) else None
    outputs = models.get("output") if isinstance(models, Mapping) else None
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise RuntimeError("completed task has no output model sequence")
    candidates = [
        model
        for model in outputs
        if getattr(model, "name", None) == EXPECTED_MODEL_NAME
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "completed FFNet task must expose exactly one official output model; "
            f"found {len(candidates)}"
        )
    model = candidates[0]
    if str(getattr(model, "task", "") or "") != task_id:
        raise RuntimeError("output model ownership mismatch")
    model_id = str(getattr(model, "id", "") or "")
    if TASK_ID_PATTERN.fullmatch(model_id) is None:
        raise RuntimeError("output model has an invalid ClearML ID")
    url = str(getattr(model, "url", "") or "")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != EXPECTED_FILES_HOST
        or parsed.port != EXPECTED_FILES_PORT
        or Path(unquote(parsed.path)).name != "epoch_40.pth"
    ):
        raise RuntimeError(f"output model is not a durable epoch_40.pth URL: {url!r}")
    return model


def _download_model(model: object, destination: Path) -> Path:
    getter = getattr(model, "get_local_copy", None)
    if not callable(getter):
        raise RuntimeError("output model cannot be downloaded")
    try:
        local = getter(raise_on_error=True)
    except TypeError:
        local = getter()
    source = Path(local).resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError("downloaded output model is missing or empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    return destination.resolve(strict=True)


def _comparison(metrics: Mapping[str, float]) -> dict[str, object]:
    return {
        "official_repository": {
            key: {
                "reference": reference,
                "reproduction": metrics[key],
                "delta": round(metrics[key] - reference, 4),
            }
            for key, reference in REPOSITORY_REFERENCE.items()
        },
        "paper_table_2_zero_latency": {
            key: {
                "reference": reference,
                "reproduction": metrics[key],
                "delta": round(metrics[key] - reference, 4),
            }
            for key, reference in PAPER_REFERENCE.items()
        },
    }


def collect(task: object, *, task_id: str, out_dir: Path, console_reports: int) -> Path:
    if TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise ValueError("task_id must be a lowercase 32-hex ClearML ID")
    if console_reports <= 0:
        raise ValueError("console_reports must be positive")
    if str(getattr(task, "id", "") or "") != task_id:
        raise RuntimeError("loaded task ID mismatch")
    if _status(task) != "completed":
        raise RuntimeError("FFNet task is not completed")

    source_identity = _require_task_parameters(task)
    run_contract = _require_run_contract(
        _artifact_mapping(task, "run_contract"), task_id
    )
    info_contract = _artifact_mapping(task, "ffnet_info_contract")
    reporter = getattr(task, "get_reported_console_output", None)
    if not callable(reporter):
        raise RuntimeError("task cannot provide console output")
    metrics, metric_log_line = _extract_final_metrics(
        reporter(number_of_reports=console_reports)
    )
    model = _require_unique_model(task, task_id)

    output = out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = _download_model(model, output / "epoch_40.pth")
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "run_contract.json", run_contract)
    _write_json(output / "ffnet_info_contract.json", info_contract)

    checkpoint_record = {
        "clearml_model_id": str(getattr(model, "id", "") or ""),
        "name": EXPECTED_MODEL_NAME,
        "url": str(getattr(model, "url", "") or ""),
        "filename": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }
    evidence = _seal(
        {
            "schema_version": 1,
            "document_type": "ffnet_b_f_official_reproduction",
            "collected_at_utc": _utc_now(),
            "clearml_task": {
                "id": task_id,
                "status": "completed",
                "url": (
                    "http://10.100.34.118:8080/projects/"
                    "6e43f972e5ea4cee901a7c8855fce8cd/experiments/"
                    f"{task_id}/output/log"
                ),
            },
            "checkpoint": checkpoint_record,
            "source_identity": source_identity,
            "metrics": metrics,
            "metric_provenance": {
                "source": "ClearML console, final validation after epoch 40",
                "evaluator": "ResilientV2XMetric",
                "recomputed_by_collector": False,
                "source_log_line": metric_log_line,
            },
            "comparison": _comparison(metrics),
            "contracts": {
                "run_contract_sha256": _sha256(output / "run_contract.json"),
                "ffnet_info_contract_sha256": _sha256(
                    output / "ffnet_info_contract.json"
                ),
            },
        }
    )
    return _write_json(output / "evidence.json", evidence)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cache = args.out_dir.expanduser().resolve() / ".clearml-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CLEARML_CACHE_DIR", str(cache))
    from clearml import Task

    task = Task.get_task(task_id=args.task_id)
    evidence = collect(
        task,
        task_id=args.task_id,
        out_dir=args.out_dir,
        console_reports=args.console_reports,
    )
    print(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
