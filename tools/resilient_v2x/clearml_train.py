#!/usr/bin/env python3
"""Run the controlled ResilientV2X two-stage experiment on one ClearML worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SPLIT_SHA256 = (
    "d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
)
EXPECTED_MANIFEST_CONTENT_SHA256 = (
    "05c247d77a5bc38130d7ecdc54fd8edb23cc5cafa74b81f6a350c73a2e836572"
)
EXPECTED_RESNET_SHA256 = (
    "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
)
EXPECTED_EVALUATION_INDEX_CONTENT_SHA256 = (
    "04a0ac5065ebaad829f4dfe7663aa0a4d436e7d587078063e09b316b46cc14b6"
)
EXPECTED_EVALUATION_SAMPLE_IDS_SHA256 = (
    "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
)
EXPECTED_EVALUATION_SAMPLE_COUNT = 1_337
MAX_DETECTIONS = 100
MIN_REFERENCE_BEV_AP_R40_070 = 1.0
MIN_REFERENCE_3D_AP_R40_070_EXCLUSIVE = 0.0
PREDICTION_DOCUMENT_TYPE = "resilient_v2x_predictions"
CONDITION_RESULT_DOCUMENT_TYPE = "resilient_v2x_condition_result"
VALIDATION_SUMMARY_DOCUMENT_TYPE = "resilient_v2x_validation_summary"
CONDITIONS = ("full", "l_fail", "c_fail")
DELAYS_MS = (0, 100, 200, 300)
CONDITION_LABELS = {
    "full": "Full",
    "l_fail": "L-Fail",
    "c_fail": "C-Fail",
}
EXPECTED_POINT_CLOUD_RANGE = [0.0, -40.0, -3.0, 80.0, 40.0, 1.0]
REQUIRED_METRIC_KEYS = {
    "resilient_v2x/sample_count",
    "resilient_v2x/car_ground_truth_count",
    "resilient_v2x/car_prediction_count",
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
    "resilient_v2x/unsupported_sample_count",
}
RUNTIME_PROFILES = ("legacy", "rtx5090")
RTX5090_EXPECTED_PACKAGES = {
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
    "mmdet": "3.2.0",
    "mmdet3d": "1.3.0",
}
RTX5090_CUSTOM_OP_MODULES = (
    "transvision.models.voxel.voxel_layer",
    "transvision.models.bev_pool.bev_pool_ext",
)
RTX5090_HEADLESS_CFG_OPTIONS = (
    "visualizer._scope_=mmengine",
    "visualizer.type=Visualizer",
    "visualizer.vis_backends.0._scope_=mmengine",
)
RTX5090_VEHICLE_TRAIN_BATCH_SIZE_PER_GPU = 8
RTX5090_VEHICLE_EVAL_BATCH_SIZE_PER_GPU = 16
RTX5090_TRAIN_BATCH_SIZE_PER_GPU = 2
RTX5090_EVAL_BATCH_SIZE_PER_GPU = 4
LEGACY_TASK_TAGS = (
    "4gpu",
    "A100",
    "DDP",
    "global-batch-4",
    "lr-1e-4-unscaled",
    "manifest-05c247d7",
)
RTX5090_TASK_TAGS = (
    "4gpu",
    "RTX5090",
    "sm120",
    "FP32",
    "DDP",
    "train-global-batch-8",
    "vehicle-global-batch-32",
    "lr-1e-4-unscaled",
    "manifest-05c247d7",
)


def _batch_profile(runtime_profile: str) -> dict[str, int]:
    if runtime_profile == "legacy":
        return {
            "vehicle_train_per_gpu": 1,
            "vehicle_eval_per_gpu": 1,
            "train_per_gpu": 1,
            "eval_per_gpu": 1,
        }
    if runtime_profile == "rtx5090":
        return {
            "vehicle_train_per_gpu": RTX5090_VEHICLE_TRAIN_BATCH_SIZE_PER_GPU,
            "vehicle_eval_per_gpu": RTX5090_VEHICLE_EVAL_BATCH_SIZE_PER_GPU,
            "train_per_gpu": RTX5090_TRAIN_BATCH_SIZE_PER_GPU,
            "eval_per_gpu": RTX5090_EVAL_BATCH_SIZE_PER_GPU,
        }
    raise ValueError(f"unknown runtime profile: {runtime_profile!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--gpus", type=int, choices=(4,), default=4)
    parser.add_argument(
        "--stage",
        choices=("all", "vehicle", "vehicle_teacher", "teacher", "student", "validate"),
        default="all",
    )
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--runtime-profile",
        choices=RUNTIME_PROFILES,
        default="legacy",
    )
    return parser


def _runtime_tags(runtime_profile: str) -> list[str]:
    if runtime_profile == "legacy":
        return list(LEGACY_TASK_TAGS)
    if runtime_profile == "rtx5090":
        return list(RTX5090_TASK_TAGS)
    raise ValueError(f"unknown runtime profile: {runtime_profile!r}")


def _runtime_cfg_options(runtime_profile: str) -> tuple[str, ...]:
    if runtime_profile == "legacy":
        return ()
    if runtime_profile == "rtx5090":
        return RTX5090_HEADLESS_CFG_OPTIONS
    raise ValueError(f"unknown runtime profile: {runtime_profile!r}")


def _runtime_environment(
    base_env: Mapping[str, str],
    runtime_profile: str,
) -> dict[str, str]:
    env = dict(base_env)
    if runtime_profile == "legacy":
        return env
    if runtime_profile == "rtx5090":
        env["PYTHONSAFEPATH"] = "1"
        return env
    raise ValueError(f"unknown runtime profile: {runtime_profile!r}")


def _merged_task_tags(
    existing_tags: Sequence[str],
    runtime_profile: str,
) -> list[str]:
    tags = list(existing_tags)
    if runtime_profile == "rtx5090":
        tags = [tag for tag in tags if "a100" not in tag.casefold()]
    return list(dict.fromkeys(tags + _runtime_tags(runtime_profile)))


def _capture_rtx5090_runtime_contract() -> dict[str, object]:
    import torch

    packages: dict[str, str | None] = {}
    for name in RTX5090_EXPECTED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    custom_ops: dict[str, bool] = {}
    for module_name in RTX5090_CUSTOM_OP_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:
            custom_ops[module_name] = False
        else:
            custom_ops[module_name] = True

    gpu_count = torch.cuda.device_count()
    return {
        "python": list(sys.version_info[:2]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count,
        "capabilities": [
            list(torch.cuda.get_device_capability(index)) for index in range(gpu_count)
        ],
        "torch_arch_list": list(torch.cuda.get_arch_list()),
        "packages": packages,
        "custom_ops": custom_ops,
    }


def _validate_rtx5090_runtime_contract(contract: Mapping[str, object]) -> None:
    expected_scalars = {
        "python": [3, 12],
        "torch": "2.10.0+cu128",
        "torch_cuda": "12.8",
        "cuda_available": True,
        "gpu_count": 4,
    }
    for field, expected in expected_scalars.items():
        if contract.get(field) != expected:
            raise RuntimeError(
                f"RTX5090 runtime {field} mismatch: "
                f"expected {expected!r}, got {contract.get(field)!r}"
            )
    if contract.get("capabilities") != [[12, 0]] * 4:
        raise RuntimeError("RTX5090 runtime requires four compute-capability 12.0 GPUs")
    arch_list = contract.get("torch_arch_list")
    if not isinstance(arch_list, list) or "sm_120" not in arch_list:
        raise RuntimeError("RTX5090 PyTorch runtime does not contain sm_120")
    if contract.get("packages") != RTX5090_EXPECTED_PACKAGES:
        raise RuntimeError(
            "RTX5090 OpenMMLab package versions do not match the sealed runtime"
        )
    expected_custom_ops = {name: True for name in RTX5090_CUSTOM_OP_MODULES}
    if contract.get("custom_ops") != expected_custom_ops:
        raise RuntimeError("RTX5090 custom operation imports failed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_evaluation_contract(path: Path) -> dict[str, object]:
    from transvision.dataset.resilient_v2x_manifest import (
        canonical_json_bytes,
        content_sha256,
    )

    raw = path.read_bytes()
    canonical_raw = raw[:-1] if raw.endswith(b"\n") else raw
    value = json.loads(canonical_raw)
    if not isinstance(value, dict):
        raise ValueError("evaluation overlay index must be a JSON object")
    if canonical_json_bytes(value) != canonical_raw:
        raise ValueError("evaluation overlay index is not canonical JSON")
    if value.get("schema_version") != 1:
        raise ValueError("evaluation overlay index schema_version must be 1")
    if value.get("artifact_type") != "resilient_v2x_evaluation_overlays":
        raise ValueError("evaluation overlay index has the wrong artifact type")
    if value.get("content_sha256") != EXPECTED_EVALUATION_INDEX_CONTENT_SHA256:
        raise ValueError("evaluation overlay index has the wrong declared hash")
    if content_sha256(value) != EXPECTED_EVALUATION_INDEX_CONTENT_SHA256:
        raise ValueError("evaluation overlay index content hash mismatch")
    if value.get("temporal_manifest_sha256") != EXPECTED_MANIFEST_CONTENT_SHA256:
        raise ValueError("evaluation overlays do not match the v2 temporal manifest")
    if value.get("split") != "val":
        raise ValueError("evaluation overlay index must bind the val split")
    if value.get("delays_ms") != list(DELAYS_MS):
        raise ValueError("evaluation overlay index has the wrong delay matrix")
    if value.get("conditions") != [CONDITION_LABELS[item] for item in CONDITIONS]:
        raise ValueError("evaluation overlay index has the wrong condition matrix")

    sample_ids = value.get("sample_ids")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != EXPECTED_EVALUATION_SAMPLE_COUNT
        or any(type(sample_id) is not str or not sample_id for sample_id in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("evaluation overlay index has an invalid sample cohort")
    sample_ids_sha256 = hashlib.sha256(canonical_json_bytes(sample_ids)).hexdigest()
    if value.get("sample_ids_sha256") != EXPECTED_EVALUATION_SAMPLE_IDS_SHA256:
        raise ValueError("evaluation overlay index has the wrong sample cohort hash")
    if sample_ids_sha256 != EXPECTED_EVALUATION_SAMPLE_IDS_SHA256:
        raise ValueError("evaluation sample cohort hash mismatch")
    return value


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_box_rows(value: object, context: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    for index, row in enumerate(value):
        if (
            not isinstance(row, list)
            or len(row) != 7
            or any(not _finite_number(coordinate) for coordinate in row)
        ):
            raise ValueError(f"{context}[{index}] must contain seven finite numbers")


def _validate_prediction_document(
    path: Path,
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    from transvision.dataset.resilient_v2x_manifest import canonical_json_bytes
    from transvision.evaluation.resilient_v2x_evidence import read_document

    document = read_document(path, expected_type=PREDICTION_DOCUMENT_TYPE)
    required_fields = {
        "schema_version",
        "document_type",
        "coordinate_convention",
        "point_cloud_range",
        "iou_thresholds",
        "max_detections",
        "sample_count",
        "samples",
        "content_sha256",
    }
    if set(document) != required_fields:
        raise ValueError("prediction document fields do not match the contract")
    if document.get("coordinate_convention") != (
        "[x,y,z_bottom,length,width,height,yaw] in current ego LiDAR"
    ):
        raise ValueError("prediction coordinate convention is invalid")
    if document.get("point_cloud_range") != EXPECTED_POINT_CLOUD_RANGE:
        raise ValueError("prediction point cloud range is invalid")
    if document.get("iou_thresholds") != [0.5, 0.7]:
        raise ValueError("prediction IoU thresholds are invalid")
    if document.get("max_detections") != MAX_DETECTIONS:
        raise ValueError("prediction max_detections is invalid")

    expected_sample_ids = evaluation.get("sample_ids")
    if not isinstance(expected_sample_ids, list):
        raise ValueError("evaluation contract is missing sample_ids")
    samples = document.get("samples")
    if (
        type(document.get("sample_count")) is not int
        or document["sample_count"] != len(expected_sample_ids)
        or not isinstance(samples, list)
        or len(samples) != len(expected_sample_ids)
    ):
        raise ValueError("prediction sample count does not match the evaluation cohort")

    sample_fields = {
        "sample_id",
        "predicted_boxes_lidar_bottom_center",
        "predicted_scores",
        "predicted_labels",
        "ground_truth_boxes_lidar_bottom_center",
        "ground_truth_labels",
        "diagnostic",
    }
    observed_sample_ids: list[str] = []
    expected_branch_order = (
        ("ego", "lidar"),
        ("rsu", "lidar"),
        ("ego", "camera"),
        ("rsu", "camera"),
    )
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != sample_fields:
            raise ValueError(f"prediction sample {index} fields are invalid")
        sample_id = sample.get("sample_id")
        if type(sample_id) is not str or not sample_id:
            raise ValueError(f"prediction sample {index} has an invalid sample_id")
        observed_sample_ids.append(sample_id)

        predicted_boxes = sample.get("predicted_boxes_lidar_bottom_center")
        ground_truth_boxes = sample.get("ground_truth_boxes_lidar_bottom_center")
        _validate_box_rows(predicted_boxes, f"prediction sample {index} boxes")
        _validate_box_rows(
            ground_truth_boxes, f"prediction sample {index} ground truth"
        )
        predicted_scores = sample.get("predicted_scores")
        predicted_labels = sample.get("predicted_labels")
        ground_truth_labels = sample.get("ground_truth_labels")
        if (
            not isinstance(predicted_scores, list)
            or not isinstance(predicted_labels, list)
            or not isinstance(ground_truth_labels, list)
            or not isinstance(predicted_boxes, list)
            or not isinstance(ground_truth_boxes, list)
            or len(predicted_boxes) != len(predicted_scores)
            or len(predicted_boxes) != len(predicted_labels)
            or len(predicted_boxes) > MAX_DETECTIONS
            or len(ground_truth_boxes) != len(ground_truth_labels)
            or any(not _finite_number(score) for score in predicted_scores)
            or any(type(label) is not int or label < 0 for label in predicted_labels)
            or any(type(label) is not int or label < 0 for label in ground_truth_labels)
        ):
            raise ValueError(f"prediction sample {index} array shapes are invalid")

        diagnostic = sample.get("diagnostic")
        if (
            not isinstance(diagnostic, dict)
            or type(diagnostic.get("method")) is not str
            or not diagnostic["method"]
            or type(diagnostic.get("overall_supported")) is not bool
            or not isinstance(diagnostic.get("routing"), dict)
        ):
            raise ValueError(f"prediction sample {index} diagnostic is invalid")
        branches = diagnostic.get("branches")
        if not isinstance(branches, list) or len(branches) != 4:
            raise ValueError(
                f"prediction sample {index} branch diagnostics are invalid"
            )
        branch_order: list[tuple[object, object]] = []
        for branch in branches:
            if (
                not isinstance(branch, dict)
                or type(branch.get("supported")) is not bool
            ):
                raise ValueError(
                    f"prediction sample {index} branch diagnostic is invalid"
                )
            branch_order.append((branch.get("agent"), branch.get("modality")))
        if tuple(branch_order) != expected_branch_order:
            raise ValueError(f"prediction sample {index} branch order is invalid")

    if observed_sample_ids != expected_sample_ids:
        raise ValueError("prediction sample order does not match the evaluation cohort")
    observed_sha256 = hashlib.sha256(
        canonical_json_bytes(observed_sample_ids)
    ).hexdigest()
    if observed_sha256 != evaluation.get("sample_ids_sha256"):
        raise ValueError(
            "prediction sample cohort hash does not match the evaluation contract"
        )
    return document


def _metrics_from_scalars(
    work_dir: Path,
    expected_sample_count: int,
) -> tuple[Path, dict[str, object]]:
    candidates = sorted(work_dir.rglob("scalars.json"))
    metric_source = "scalars.json"
    if not candidates:
        candidates = sorted(
            path
            for path in work_dir.rglob("*.json")
            if len(path.name) == 20
            and path.name[8] == "_"
            and path.name.endswith(".json")
            and (path.name[:8] + path.name[9:15]).isdigit()
        )
        metric_source = "MMEngine timestamped metric log"
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one {metric_source} under {work_dir}, "
            f"found {len(candidates)}"
        )
    scalars = candidates[0].resolve(strict=True)
    metric_rows: list[dict[str, object]] = []
    metric_content = scalars.read_text(encoding="utf-8")
    metric_lines = (
        metric_content.splitlines()
        if metric_source == "scalars.json"
        else [metric_content]
    )
    for line_number, line in enumerate(metric_lines, start=1):
        if not line:
            raise ValueError(f"empty JSONL row in {scalars}:{line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row in {scalars}:{line_number}")
        metrics = {
            key: value for key, value in row.items() if key.startswith("resilient_v2x/")
        }
        if metrics:
            metric_rows.append(metrics)
    if len(metric_rows) != 1:
        raise ValueError(
            f"expected exactly one ResilientV2X metric row, found {len(metric_rows)}"
        )
    metrics = metric_rows[0]
    if set(metrics) != REQUIRED_METRIC_KEYS:
        missing = sorted(REQUIRED_METRIC_KEYS - set(metrics))
        extra = sorted(set(metrics) - REQUIRED_METRIC_KEYS)
        raise ValueError(
            f"metric keys are incomplete: missing={missing}, extra={extra}"
        )
    if any(not _finite_number(value) for value in metrics.values()):
        raise ValueError("all ResilientV2X metrics must be finite numbers")

    count_keys = {
        "resilient_v2x/sample_count",
        "resilient_v2x/car_ground_truth_count",
        "resilient_v2x/car_prediction_count",
        "resilient_v2x/unsupported_sample_count",
    }
    counts: dict[str, int] = {}
    for key in count_keys:
        numeric = float(metrics[key])
        if numeric < 0.0 or not numeric.is_integer():
            raise ValueError(f"{key} must be a non-negative integer-valued metric")
        counts[key] = int(numeric)
    if counts["resilient_v2x/sample_count"] != expected_sample_count:
        raise ValueError("scalar sample count does not match the evaluation cohort")
    if counts["resilient_v2x/car_ground_truth_count"] <= 0:
        raise ValueError("evaluation cohort must contain car ground truth")
    if (
        counts["resilient_v2x/car_prediction_count"]
        > expected_sample_count * MAX_DETECTIONS
    ):
        raise ValueError("prediction count exceeds max_detections times sample count")
    if counts["resilient_v2x/unsupported_sample_count"] > expected_sample_count:
        raise ValueError("unsupported sample count exceeds sample count")
    for key in REQUIRED_METRIC_KEYS - count_keys:
        value = float(metrics[key])
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{key} must be an AP percentage in [0, 100]")
    return scalars, metrics


def _validate_formal_reference_metrics(metrics: Mapping[str, object]) -> None:
    """Reject the known high-IoU collapse before publishing formal evidence."""

    bev_ap70 = float(metrics["resilient_v2x/car_bev_ap_r40_0.70"])
    three_d_ap70 = float(metrics["resilient_v2x/car_3d_ap_r40_0.70"])
    if bev_ap70 < MIN_REFERENCE_BEV_AP_R40_070:
        raise RuntimeError(
            "delay_000_full BEV AP@0.7 failed the recovery gate: "
            f"{bev_ap70:.6f} < {MIN_REFERENCE_BEV_AP_R40_070:.6f}"
        )
    if three_d_ap70 <= MIN_REFERENCE_3D_AP_R40_070_EXCLUSIVE:
        raise RuntimeError(
            "delay_000_full 3D AP@0.7 failed the recovery gate: "
            f"{three_d_ap70:.6f} must be greater than "
            f"{MIN_REFERENCE_3D_AP_R40_070_EXCLUSIVE:.6f}"
        )


def _write_sealed_document(
    path: Path,
    document_type: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    from transvision.evaluation.resilient_v2x_evidence import (
        seal_document,
        write_document,
    )

    document = seal_document(document_type, payload)
    write_document(path, document)
    return document


def _run(command: Sequence[str], *, env: Mapping[str, str]) -> None:
    print(
        json.dumps({"command": list(command), "cwd": str(ROOT)}, sort_keys=True),
        flush=True,
    )
    subprocess.run(command, cwd=ROOT, env=dict(env), check=True)


def _torchrun(gpus: int, entry_point: str, *arguments: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "--module",
        "tools.resilient_v2x.run_deterministic",
        entry_point,
        *arguments,
    ]


def _checkpoint(work_dir: Path, prefix: str, epoch: int) -> Path:
    filename = (
        f"teacher_epoch_{epoch}.pth" if prefix == "teacher" else f"epoch_{epoch}.pth"
    )
    checkpoint = work_dir / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(f"final epoch checkpoint is missing: {checkpoint}")
    return checkpoint.resolve(strict=True)


def _best_checkpoint(work_dir: Path, prefix: str) -> Path:
    candidates = sorted(
        work_dir.glob(
            "best_resilient_v2x_car_bev_ap_r40_0.70_"
            f"{prefix}_epoch_*.pth"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one best {prefix} checkpoint, found {len(candidates)}"
        )
    checkpoint = candidates[0]
    if checkpoint.is_symlink():
        checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError(f"best {prefix} checkpoint is invalid: {checkpoint}")
    return checkpoint.resolve(strict=True)


def _overlay_environment(
    dataset_root: Path,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    artifact_root = dataset_root / "protocols/dair_v2"
    training = _read_json(artifact_root / "training_overlays.json")
    evaluation = _load_evaluation_contract(artifact_root / "evaluation_overlays.json")
    if training.get("temporal_manifest_sha256") != EXPECTED_MANIFEST_CONTENT_SHA256:
        raise ValueError("training overlays do not match the v2 temporal manifest")
    if evaluation.get("temporal_manifest_sha256") != EXPECTED_MANIFEST_CONTENT_SHA256:
        raise ValueError("evaluation overlays do not match the v2 temporal manifest")

    overlays = training.get("overlays")
    if not isinstance(overlays, dict):
        raise ValueError("training overlay index is missing overlays")
    transport = overlays.get("transport")
    fault = overlays.get("fault")
    if not isinstance(transport, dict) or not isinstance(fault, dict):
        raise ValueError("training overlay records are invalid")

    env = dict(base_env)
    env.update(
        {
            "RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY": str(
                artifact_root / str(transport["path"])
            ),
            "RESILIENT_V2X_TRAIN_TRANSPORT_SHA256": str(
                transport["uncompressed_sha256"]
            ),
            "RESILIENT_V2X_TRAIN_FAULT_OVERLAY": str(
                artifact_root / str(fault["path"])
            ),
            "RESILIENT_V2X_TRAIN_FAULT_SHA256": str(fault["uncompressed_sha256"]),
        }
    )

    transports = evaluation.get("transport_overlays")
    faults = evaluation.get("fault_overlays")
    if not isinstance(transports, list) or not isinstance(faults, list):
        raise ValueError("evaluation overlay records are invalid")
    for record in transports:
        if not isinstance(record, dict) or not isinstance(record.get("overlay"), dict):
            raise ValueError("invalid evaluation transport overlay")
        delay = int(record["delay_ms"])
        overlay = record["overlay"]
        stem = f"RESILIENT_V2X_TEST_TRANSPORT_DELAY_{delay:03d}"
        env[f"{stem}_OVERLAY"] = str(artifact_root / str(overlay["path"]))
        env[f"{stem}_SHA256"] = str(overlay["uncompressed_sha256"])
    for record in faults:
        if (
            not isinstance(record, dict)
            or record.get("agent_scope") != "E+R"
            or record.get("duration") != 1
            or record.get("condition") not in ("L-Fail", "C-Fail")
        ):
            continue
        overlay = record.get("overlay")
        if not isinstance(overlay, dict):
            raise ValueError("invalid evaluation fault overlay")
        delay = int(record["delay_ms"])
        condition = str(record["condition"]).replace("-", "_").upper()
        stem = f"RESILIENT_V2X_TEST_CAUSAL_DELAY_{delay:03d}_{condition}"
        env[f"{stem}_OVERLAY"] = str(artifact_root / str(overlay["path"]))
        env[f"{stem}_SHA256"] = str(overlay["uncompressed_sha256"])
    return env


def _condition_config(delay_ms: int, condition: str) -> Path:
    if condition == "full":
        name = f"global_delay_{delay_ms:03d}_full.py"
    else:
        name = f"causal_delay_{delay_ms:03d}_{condition}.py"
    return ROOT / "configs/resilient_v2x/conditions" / name


def _upload_model(
    task: object,
    name: str,
    checkpoint: Path,
    runtime_profile: str,
) -> None:
    from clearml import OutputModel

    tags = ["ResilientV2X", "DDP", "A100"]
    if runtime_profile == "rtx5090":
        tags = ["ResilientV2X", "DDP", "RTX5090", "sm120", "FP32"]
    model = OutputModel(
        task=task,
        name=name,
        framework="PyTorch",
        tags=tags,
    )
    uri = model.update_weights(
        weights_filename=str(checkpoint),
        auto_delete_file=False,
        async_enable=False,
    )
    if urlsplit(uri).scheme not in {"http", "https", "s3", "gs", "azure"}:
        raise RuntimeError(f"model was not uploaded to durable storage: {uri!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if args.runtime_profile == "rtx5090" and args.amp:
        raise ValueError("RTX5090 first-run profile requires FP32; --amp is forbidden")

    if args.runtime_profile == "rtx5090":
        _validate_rtx5090_runtime_contract(_capture_rtx5090_runtime_contract())
    batch_profile = _batch_profile(args.runtime_profile)

    from clearml import Dataset, Task, TaskTypes

    task = Task.current_task() or Task.init(
        project_name="ResilientV2X/Training",
        task_name=(
            "ResilientV2X RTX5090 4-GPU train and validation"
            if args.runtime_profile == "rtx5090"
            else "ResilientV2X controlled 4-GPU train and validation"
        ),
        task_type=TaskTypes.training,
        reuse_last_task_id=False,
        output_uri=True,
    )
    if not task.output_uri:
        task.output_uri = True
    task.set_tags(_merged_task_tags(task.get_tags(), args.runtime_profile))

    dataset = Dataset.get(dataset_id=args.dataset_id, only_completed=True)
    dataset_root = Path(dataset.get_local_copy()).resolve(strict=True)
    data_root = dataset_root / "cooperative-vehicle-infrastructure"
    manifest_path = dataset_root / "manifests/temporal_manifest_v2.json"
    evaluation_index_path = dataset_root / "protocols/dair_v2/evaluation_overlays.json"
    resnet_checkpoint = dataset_root / "models/resnet50-0676ba61.pth"
    manifest = _read_json(manifest_path)
    evaluation = _load_evaluation_contract(evaluation_index_path)
    if manifest.get("content_sha256") != EXPECTED_MANIFEST_CONTENT_SHA256:
        raise ValueError("ClearML dataset contains the wrong temporal manifest")
    if _sha256(resnet_checkpoint) != EXPECTED_RESNET_SHA256:
        raise ValueError("ClearML dataset contains the wrong ResNet-50 checkpoint")

    import torch

    if torch.cuda.device_count() != args.gpus:
        raise RuntimeError(
            f"expected {args.gpus} visible GPUs, got {torch.cuda.device_count()}"
        )

    env = _runtime_environment(os.environ, args.runtime_profile)
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "NVIDIA_TF32_OVERRIDE": "0",
            "RESILIENT_V2X_DATA_ROOT": str(data_root),
            "RESILIENT_V2X_MANIFEST": str(manifest_path),
            "RESILIENT_V2X_SPLIT_SHA256": OFFICIAL_SPLIT_SHA256,
            "RESILIENT_V2X_RESNET50_CHECKPOINT": str(resnet_checkpoint),
        }
    )
    env = _overlay_environment(dataset_root, env)

    task_id = task.id
    work_root = ROOT / "work_dirs/clearml" / task_id
    work_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task_id": task_id,
        "dataset_id": args.dataset_id,
        "dataset_local_copy": str(dataset_root),
        "gpus": args.gpus,
        "global_batch_size": args.gpus * batch_profile["train_per_gpu"],
        "train_batch_size_per_gpu": batch_profile["train_per_gpu"],
        "eval_batch_size_per_gpu": batch_profile["eval_per_gpu"],
        "vehicle_global_batch_size": (
            args.gpus * batch_profile["vehicle_train_per_gpu"]
        ),
        "vehicle_train_batch_size_per_gpu": batch_profile["vehicle_train_per_gpu"],
        "vehicle_eval_batch_size_per_gpu": batch_profile["vehicle_eval_per_gpu"],
        "learning_rate": 0.0001,
        "auto_scale_lr": False,
        "max_epochs": args.max_epochs,
        "amp": args.amp,
        "runtime_profile": args.runtime_profile,
        "python_safe_path": env.get("PYTHONSAFEPATH"),
        "checkpoint_policy": "final_epoch",
        "manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
        "split_sha256": OFFICIAL_SPLIT_SHA256,
        "evaluation_index_content_sha256": EXPECTED_EVALUATION_INDEX_CONTENT_SHA256,
        "evaluation_sample_ids_sha256": EXPECTED_EVALUATION_SAMPLE_IDS_SHA256,
        "evaluation_sample_count": EXPECTED_EVALUATION_SAMPLE_COUNT,
        "evaluation_condition_count": len(DELAYS_MS) * len(CONDITIONS),
        "output_uri_scheme": urlsplit(str(task.output_uri)).scheme or "clearml-default",
        "stage": args.stage,
    }
    if not task.upload_artifact(
        "run_contract",
        artifact_object=metadata,
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to upload the run contract")

    if args.runtime_profile == "legacy":
        _run(
            [
                sys.executable,
                "tools/resilient_v2x/run_deterministic.py",
                "tools/resilient_v2x/check_environment.py",
                "--mode",
                "controlled",
                "--world-size",
                str(args.gpus),
                "--require-cuda",
                "--require-custom-ops",
            ],
            env=env,
        )

    runtime_cfg_options = _runtime_cfg_options(args.runtime_profile)
    def train_args(train_batch_size: int, eval_batch_size: int) -> list[str]:
        result = [
            "--launcher",
            "pytorch",
            "--cfg-options",
            f"train_cfg.max_epochs={args.max_epochs}",
            f"train_dataloader.batch_size={train_batch_size}",
            f"val_dataloader.batch_size={eval_batch_size}",
            f"test_dataloader.batch_size={eval_batch_size}",
            *runtime_cfg_options,
        ]
        if args.amp:
            result.insert(0, "--amp")
        return result

    vehicle_train_args = train_args(
        batch_profile["vehicle_train_per_gpu"],
        batch_profile["vehicle_eval_per_gpu"],
    )
    model_train_args = train_args(
        batch_profile["train_per_gpu"],
        batch_profile["eval_per_gpu"],
    )

    vehicle_checkpoint: Path | None = None
    if args.stage in ("vehicle", "vehicle_teacher"):
        vehicle_dir = work_root / "vehicle"
        _run(
            _torchrun(
                args.gpus,
                "tools/train.py",
                "configs/resilient_v2x/dair_vehicle_pretrain.py",
                "--work-dir",
                str(vehicle_dir),
                *vehicle_train_args,
            ),
            env=env,
        )
        vehicle_checkpoint = _best_checkpoint(vehicle_dir, "vehicle")
        _upload_model(
            task,
            "ResilientV2X vehicle PointPillars pretrain",
            vehicle_checkpoint,
            args.runtime_profile,
        )

    if args.stage == "vehicle":
        task.flush(wait_for_uploads=True)
        return 0

    teacher_checkpoint = (
        args.teacher_checkpoint.resolve(strict=True)
        if args.teacher_checkpoint is not None
        else None
    )
    if args.stage in ("all", "teacher", "vehicle_teacher"):
        teacher_dir = work_root / "teacher"
        teacher_env = dict(env)
        if vehicle_checkpoint is not None:
            teacher_env["RESILIENT_V2X_VEHICLE_PRETRAIN_CHECKPOINT"] = str(
                vehicle_checkpoint
            )
        _run(
            _torchrun(
                args.gpus,
                "tools/train.py",
                "configs/resilient_v2x/dair_clean_teacher.py",
                "--work-dir",
                str(teacher_dir),
                *model_train_args,
            ),
            env=teacher_env,
        )
        teacher_checkpoint = _checkpoint(teacher_dir, "teacher", args.max_epochs)
        _upload_model(
            task,
            "ResilientV2X clean teacher",
            teacher_checkpoint,
            args.runtime_profile,
        )

    if args.stage in ("teacher", "vehicle_teacher"):
        task.flush(wait_for_uploads=True)
        return 0
    if teacher_checkpoint is None and args.stage in ("all", "student", "validate"):
        raise ValueError("student training and validation require a teacher checkpoint")

    student_checkpoint = (
        args.student_checkpoint.resolve(strict=True)
        if args.student_checkpoint is not None
        else None
    )
    if args.stage in ("all", "student"):
        student_dir = work_root / "student"
        student_env = dict(env)
        student_env["RESILIENT_V2X_TEACHER_CHECKPOINT"] = str(teacher_checkpoint)
        _run(
            _torchrun(
                args.gpus,
                "tools/train.py",
                "configs/resilient_v2x/dair_resilient_v2x.py",
                "--work-dir",
                str(student_dir),
                *model_train_args,
            ),
            env=student_env,
        )
        student_checkpoint = _checkpoint(student_dir, "student", args.max_epochs)
        _upload_model(
            task,
            "ResilientV2X distilled student",
            student_checkpoint,
            args.runtime_profile,
        )

    if args.stage == "student":
        task.flush(wait_for_uploads=True)
        return 0
    if student_checkpoint is None:
        raise ValueError("validation requires a student checkpoint")

    student_checkpoint_sha256 = _sha256(student_checkpoint)
    teacher_checkpoint_sha256 = _sha256(teacher_checkpoint)
    condition_results: dict[str, dict[str, object]] = {}
    for delay_ms in DELAYS_MS:
        for condition in CONDITIONS:
            condition_name = f"delay_{delay_ms:03d}_{condition}"
            artifact_name = f"validation_{condition_name}"
            condition_dir = work_root / "validation" / condition_name
            if condition_dir.exists():
                raise FileExistsError(
                    f"refusing to reuse validation evidence directory: {condition_dir}"
                )
            condition_dir.mkdir(parents=True)
            prediction_path = condition_dir / "predictions.json"
            condition_config = _condition_config(delay_ms, condition).resolve(
                strict=True
            )
            condition_env = dict(env)
            condition_env["RESILIENT_V2X_TEACHER_CHECKPOINT"] = str(teacher_checkpoint)
            condition_env["RESILIENT_V2X_PREDICTION_OUTPUT"] = str(prediction_path)
            _run(
                _torchrun(
                    args.gpus,
                    "tools/test.py",
                    str(condition_config),
                    str(student_checkpoint),
                    "--work-dir",
                    str(condition_dir),
                    "--launcher",
                    "pytorch",
                    "--cfg-options",
                    f"test_dataloader.batch_size={batch_profile['eval_per_gpu']}",
                    *runtime_cfg_options,
                ),
                env=condition_env,
            )

            prediction = _validate_prediction_document(prediction_path, evaluation)
            scalars_path, metrics = _metrics_from_scalars(
                condition_dir,
                EXPECTED_EVALUATION_SAMPLE_COUNT,
            )
            if metrics["resilient_v2x/sample_count"] != prediction["sample_count"]:
                raise ValueError(
                    f"{condition_name} prediction and scalar sample counts differ"
                )
            if condition_name == "delay_000_full":
                _validate_formal_reference_metrics(metrics)

            transport_stem = f"RESILIENT_V2X_TEST_TRANSPORT_DELAY_{delay_ms:03d}"
            transport_sha256 = condition_env[f"{transport_stem}_SHA256"]
            fault_sha256 = None
            if condition != "full":
                fault_stem = (
                    f"RESILIENT_V2X_TEST_CAUSAL_DELAY_{delay_ms:03d}_"
                    f"{condition.upper()}"
                )
                fault_sha256 = condition_env[f"{fault_stem}_SHA256"]

            condition_payload: dict[str, object] = {
                "task_id": task_id,
                "dataset_id": args.dataset_id,
                "runtime_profile": args.runtime_profile,
                "artifact_name": artifact_name,
                "condition_id": condition_name,
                "delay_ms": delay_ms,
                "condition": CONDITION_LABELS[condition],
                "manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
                "evaluation_index_content_sha256": (
                    EXPECTED_EVALUATION_INDEX_CONTENT_SHA256
                ),
                "evaluation_sample_ids_sha256": (EXPECTED_EVALUATION_SAMPLE_IDS_SHA256),
                "evaluation_sample_count": EXPECTED_EVALUATION_SAMPLE_COUNT,
                "checkpoints": {
                    "student": {
                        "filename": student_checkpoint.name,
                        "size_bytes": student_checkpoint.stat().st_size,
                        "sha256": student_checkpoint_sha256,
                    },
                    "teacher": {
                        "filename": teacher_checkpoint.name,
                        "size_bytes": teacher_checkpoint.stat().st_size,
                        "sha256": teacher_checkpoint_sha256,
                    },
                },
                "condition_config": {
                    "path": condition_config.relative_to(ROOT).as_posix(),
                    "size_bytes": condition_config.stat().st_size,
                    "sha256": _sha256(condition_config),
                },
                "overlays": {
                    "transport_uncompressed_sha256": transport_sha256,
                    "fault_uncompressed_sha256": fault_sha256,
                },
                "prediction": {
                    "path": prediction_path.relative_to(work_root).as_posix(),
                    "size_bytes": prediction_path.stat().st_size,
                    "file_sha256": _sha256(prediction_path),
                    "content_sha256": prediction["content_sha256"],
                    "sample_count": prediction["sample_count"],
                    "sample_ids_sha256": EXPECTED_EVALUATION_SAMPLE_IDS_SHA256,
                },
                "scalars": {
                    "path": scalars_path.relative_to(work_root).as_posix(),
                    "size_bytes": scalars_path.stat().st_size,
                    "file_sha256": _sha256(scalars_path),
                },
                "metrics": metrics,
            }
            condition_result = _write_sealed_document(
                condition_dir / "metrics.json",
                CONDITION_RESULT_DOCUMENT_TYPE,
                condition_payload,
            )
            if not task.upload_artifact(
                artifact_name,
                artifact_object=str(condition_dir),
                wait_on_upload=True,
            ):
                raise RuntimeError(
                    f"failed to synchronously upload validation evidence: {artifact_name}"
                )
            condition_results[condition_name] = condition_result

    expected_condition_names = {
        f"delay_{delay_ms:03d}_{condition}"
        for delay_ms in DELAYS_MS
        for condition in CONDITIONS
    }
    if set(condition_results) != expected_condition_names:
        raise RuntimeError(
            "validation did not produce the complete 12-condition matrix"
        )
    ground_truth_counts = {
        int(float(result["metrics"]["resilient_v2x/car_ground_truth_count"]))
        for result in condition_results.values()
    }
    if len(ground_truth_counts) != 1:
        raise ValueError("ground-truth count differs across validation conditions")

    summary_path = work_root / "validation_summary.json"
    _write_sealed_document(
        summary_path,
        VALIDATION_SUMMARY_DOCUMENT_TYPE,
        {
            "task_id": task_id,
            "dataset_id": args.dataset_id,
            "runtime_profile": args.runtime_profile,
            "manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
            "evaluation_index_content_sha256": (
                EXPECTED_EVALUATION_INDEX_CONTENT_SHA256
            ),
            "evaluation_sample_ids_sha256": EXPECTED_EVALUATION_SAMPLE_IDS_SHA256,
            "evaluation_sample_count": EXPECTED_EVALUATION_SAMPLE_COUNT,
            "student_checkpoint": {
                "filename": student_checkpoint.name,
                "size_bytes": student_checkpoint.stat().st_size,
                "sha256": student_checkpoint_sha256,
            },
            "teacher_checkpoint": {
                "filename": teacher_checkpoint.name,
                "size_bytes": teacher_checkpoint.stat().st_size,
                "sha256": teacher_checkpoint_sha256,
            },
            "condition_count": len(condition_results),
            "ground_truth_count": next(iter(ground_truth_counts)),
            "conditions": condition_results,
        },
    )
    if not task.upload_artifact(
        "validation_summary",
        artifact_object=str(summary_path),
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to upload the validation summary")
    task.flush(wait_for_uploads=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
