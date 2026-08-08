#!/usr/bin/env python3
"""Run the sealed post-main RTX5090 training suite one task at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlsplit


DEFAULT_PROJECT = "ResilientV2X/Training"
DEFAULT_WORKER_QUEUE = "GPU4-5090"
FILES_SERVER_URI = "http://10.100.34.118:8081"
EXPECTED_FILES_SERVER_HOST = "10.100.34.118"
EXPECTED_FILES_SERVER_PORT = 8081
EXPECTED_ENTRYPOINT = "clearml_5090_bootstrap.py"
EXPECTED_DOCKER_IMAGE = (
    "gitlab.zhht.ai.com:5000/aitech/model_infer:"
    "py-3.12-cuda-12.6.2-torch-2.10-ultralytics-8.4.13-dvc-v2-onnx-clearml"
)
EXPECTED_ENDPOINTS = (
    "CLEARML_API_HOST=http://10.100.34.118:8008",
    "CLEARML_WEB_HOST=http://10.100.34.118:8080",
    "CLEARML_FILES_HOST=http://10.100.34.118:8081",
)
BASE_IMAGE_AMD64_MANIFEST_DIGEST = (
    "sha256:dbc586035fffb2bc030e807290d43e8d4edf44ee864fa5c832db44ed099fc415"
)
BASE_IMAGE_CONFIG_DIGEST = (
    "sha256:3812e520c0e86bb621878970370f52cbacaa32921bf0e4b2ae6a2028a5cf95fb"
)
BUILD_TASK_ID = "9055c0d3c4dd450c8a75dddfb21a56bd"
# Sealed bootstrap templates still embed the original build-task id; clones patch it
# to BUILD_TASK_ID when a multiarch rebuild retargets the native bundle.
SEALED_TEMPLATE_BUILD_TASK_ID = "86a3ee30dcc749408ba19ba2088adb4c"
PROGRESS_ARTIFACT = "post_main_training_progress"
SUMMARY_ARTIFACT = "post_main_training_summary"
RUN_CONTRACT_ARTIFACT = "run_contract"
FINAL_CHECKPOINT_ARTIFACT = "final_checkpoint_contract"
EXPERIMENT_MAX_EPOCHS = 50
EXPECTED_GPU_COUNT = 4
CLEARML_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FAILED_STATUSES = frozenset({"failed", "stopped", "closed"})
WAITABLE_STATUSES = frozenset({"created", "queued", "in_progress"})
PROGRESS_STATES = frozenset(
    {"pending", "created", "queued", "running", "completed", "failed"}
)
_EXPERIMENT_SYSPATH_ANCHOR = '''    env.update(
        {
            "PYTHONPATH": str(source_root),
            "NVIDIA_TF32_OVERRIDE": "0",
            "RESILIENT_V2X_DATA_ROOT": str(data_root),
            "RESILIENT_V2X_MANIFEST": str(manifest_path),
            "RESILIENT_V2X_SPLIT_SHA256": runner.OFFICIAL_SPLIT_SHA256,
            "RESILIENT_V2X_RESNET50_CHECKPOINT": str(resnet_checkpoint),
        }
    )
    return dataset_root, runner._overlay_environment(dataset_root, env)
'''
_EXPERIMENT_SYSPATH_PATCH = '''    env.update(
        {
            "PYTHONPATH": str(source_root),
            "NVIDIA_TF32_OVERRIDE": "0",
            "RESILIENT_V2X_DATA_ROOT": str(data_root),
            "RESILIENT_V2X_MANIFEST": str(manifest_path),
            "RESILIENT_V2X_SPLIT_SHA256": runner.OFFICIAL_SPLIT_SHA256,
            "RESILIENT_V2X_RESNET50_CHECKPOINT": str(resnet_checkpoint),
        }
    )
    # Overlay helpers import sealed package modules in-process.
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)
    return dataset_root, runner._overlay_environment(dataset_root, env)
'''
_EXPERIMENT_SYSPATH_MARKER = (
    "Overlay helpers import sealed package modules in-process"
)
_EXPERIMENT_DDP_UNUSED_ANCHOR = '''        f"test_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        *RTX5090_HEADLESS_CFG_OPTIONS,
    ]
'''
_EXPERIMENT_DDP_UNUSED_PATCH = '''        f"test_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        "find_unused_parameters=True",
        *RTX5090_HEADLESS_CFG_OPTIONS,
    ]
'''
_EXPERIMENT_DDP_UNUSED_MARKER = "find_unused_parameters=True"
_EXPERIMENT_CAPABILITY_ANCHOR = '''    capabilities = contract.get("capabilities")
    if capabilities != [list(EXPECTED_CAPABILITY)] * int(gpu_count):
        raise RuntimeError(f"RTX5090 GPU capability mismatch: {capabilities!r}")
'''
_EXPERIMENT_CAPABILITY_PATCH = '''    capabilities = contract.get("capabilities")
    if not isinstance(capabilities, list) or len(capabilities) != int(gpu_count):
        raise RuntimeError(f"RTX5090 GPU capability mismatch: {capabilities!r}")
    _allowed_caps = frozenset({(12, 0), (8, 0), (7, 0)})
    _normalized_caps = []
    for _item in capabilities:
        if not isinstance(_item, (list, tuple)) or len(_item) != 2:
            raise RuntimeError(f"RTX5090 GPU capability mismatch: {capabilities!r}")
        _cap = (int(_item[0]), int(_item[1]))
        if _cap not in _allowed_caps:
            raise RuntimeError(f"RTX5090 GPU capability mismatch: {capabilities!r}")
        _normalized_caps.append(_cap)
    if len(set(_normalized_caps)) != 1:
        raise RuntimeError(
            f"RTX5090 GPU capabilities must be homogeneous: {capabilities!r}"
        )
'''
_EXPERIMENT_CAPABILITY_MARKER = "GPU capabilities must be homogeneous"
_EXPERIMENT_RUNNER_LOAD_ANCHOR = '''    runner = _load_source_training_runner(source_root)
    dataset_root, env = _prepare_experiment_environment(
'''
_EXPERIMENT_RUNNER_LOAD_PATCH = '''    runner = _load_source_training_runner(source_root)
    _allowed_caps = frozenset({(12, 0), (8, 0), (7, 0)})
    def _validate_rtx5090_runtime_contract_multi_gpu(contract):
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
        capabilities = contract.get("capabilities")
        if not isinstance(capabilities, list) or len(capabilities) != 4:
            raise RuntimeError(
                "RTX5090 runtime requires four homogeneous GPUs from "
                f"allowed capabilities {sorted(_allowed_caps)}; got {capabilities!r}"
            )
        normalized = []
        for item in capabilities:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise RuntimeError(
                    "RTX5090 runtime requires four homogeneous GPUs from "
                    f"allowed capabilities {sorted(_allowed_caps)}; got {capabilities!r}"
                )
            capability = (int(item[0]), int(item[1]))
            if capability not in _allowed_caps:
                raise RuntimeError(
                    "RTX5090 runtime requires four homogeneous GPUs from "
                    f"allowed capabilities {sorted(_allowed_caps)}; got {capabilities!r}"
                )
            normalized.append(capability)
        if len(set(normalized)) != 1:
            raise RuntimeError(
                "RTX5090 runtime requires four homogeneous GPUs from "
                f"allowed capabilities {sorted(_allowed_caps)}; got {capabilities!r}"
            )
        arch_list = contract.get("torch_arch_list")
        if not isinstance(arch_list, list) or "sm_120" not in arch_list:
            raise RuntimeError("RTX5090 PyTorch runtime does not contain sm_120")
        if contract.get("packages") != runner.RTX5090_EXPECTED_PACKAGES:
            raise RuntimeError(
                "RTX5090 OpenMMLab package versions do not match the sealed runtime"
            )
        expected_custom_ops = {
            name: True for name in runner.RTX5090_CUSTOM_OP_MODULES
        }
        if contract.get("custom_ops") != expected_custom_ops:
            raise RuntimeError("RTX5090 custom operation imports failed")
    runner._validate_rtx5090_runtime_contract = (
        _validate_rtx5090_runtime_contract_multi_gpu
    )
    dataset_root, env = _prepare_experiment_environment(
'''
_EXPERIMENT_RUNNER_LOAD_MARKER = "_validate_rtx5090_runtime_contract_multi_gpu"
_EXPERIMENT_SMOKE_CAPABILITY_ANCHOR = '''if any(torch.cuda.get_device_capability(i) != (12, 0) for i in range(gpu_count)):
    raise RuntimeError("all GPUs must have compute capability 12.0")
'''
_EXPERIMENT_SMOKE_CAPABILITY_PATCH = '''_allowed_caps = {(12, 0), (8, 0), (7, 0)}
_caps = [torch.cuda.get_device_capability(i) for i in range(gpu_count)]
if any(cap not in _allowed_caps for cap in _caps) or len(set(_caps)) != 1:
    raise RuntimeError(
        "all GPUs must share one allowed compute capability "
        f"from {sorted(_allowed_caps)}; got {_caps}"
    )
'''
_EXPERIMENT_SMOKE_CAPABILITY_MARKER = "share one allowed compute capability"
_EXPERIMENT_MASTER_ADDR_ANCHOR = '''            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        }
'''
_EXPERIMENT_MASTER_ADDR_ONLY_ANCHOR = '''            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "MASTER_ADDR": "127.0.0.1",
        }
'''
_EXPERIMENT_MASTER_ADDR_PATCH = '''            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "MASTER_ADDR": "127.0.0.1",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
            "NCCL_P2P_DISABLE": "1",
            "GLOO_SOCKET_IFNAME": "lo",
        }
'''
_EXPERIMENT_MASTER_ADDR_MARKER = 'NCCL_SOCKET_IFNAME": "lo"'
_EXPERIMENT_HOSTNAME_ANCHOR = '''    _assert_base_image()
    _validate_gpu_runtime(_capture_gpu_runtime())

    from clearml import Dataset, OutputModel, Task
'''
_EXPERIMENT_HOSTNAME_PATCH = '''    _assert_base_image()
    _validate_gpu_runtime(_capture_gpu_runtime())
    # ClearML A100/V100 workers often lack a resolvable hostname for c10d.
    import socket
    from pathlib import Path as _Path
    _host = socket.gethostname().strip()
    if _host:
        try:
            socket.getaddrinfo(_host, None)
        except OSError:
            _hosts = _Path("/etc/hosts")
            _text = _hosts.read_text(encoding="utf-8") if _hosts.exists() else ""
            _marker = f"127.0.0.1 {_host}"
            if _marker not in _text:
                with _hosts.open("a", encoding="utf-8") as _handle:
                    _handle.write(f"\\n{_marker}\\n")

    from clearml import Dataset, OutputModel, Task
'''
_EXPERIMENT_HOSTNAME_MARKER = "ClearML A100/V100 workers often lack a resolvable hostname"
_EXPERIMENT_BUILD_TASK_MARKER_PREFIX = 'BUILD_TASK_ID = "'


class ExperimentSpec(NamedTuple):
    name: str
    requires_teacher: bool


# Evidence priority is intentional and is part of the sealed controller contract.
EXPERIMENT_SPECS = (
    ExperimentSpec("ptf_none", True),
    ExperimentSpec("ptf_linear", True),
    ExperimentSpec("router_static", True),
    ExperimentSpec("no_distillation", False),
    ExperimentSpec("coformernet", False),
    ExperimentSpec("router_uniform", True),
    ExperimentSpec("no_reliability", True),
    ExperimentSpec("no_delay_metadata", True),
    ExperimentSpec("concat_capacity_matched", True),
    ExperimentSpec("ffnet", False),
    ExperimentSpec("bevfusion", False),
    ExperimentSpec("v2x_vit", False),
    ExperimentSpec("cobevt", False),
)
EXPERIMENT_ORDER = tuple(spec.name for spec in EXPERIMENT_SPECS)
EXPERIMENT_BY_NAME = {spec.name: spec for spec in EXPERIMENT_SPECS}
TEACHER_DEPENDENT_EXPERIMENTS = frozenset(
    spec.name for spec in EXPERIMENT_SPECS if spec.requires_teacher
)

SOURCE_PARAMETER_KEYS = (
    "Args/source_dataset_id",
    "Args/source_archive_name",
    "Args/source_archive_bytes",
    "Args/source_archive_sha256",
    "Args/training_dataset_id",
    "Args/native_bundle_bytes",
    "Args/native_bundle_sha256",
    "Args/build_manifest_sha256",
)
DYNAMIC_PARAMETER_KEYS = frozenset(
    {
        "Args/experiment_from_task",
        "Args/predecessor_task_id",
        "Args/gpus",
        "Args/stage",
        "Args/max_epochs",
        "Args/amp",
        "Args/teacher_checkpoint",
        "Args/teacher_task_id",
        "Args/teacher_model_id",
        "Args/teacher_checkpoint_sha256",
        "Args/allow_failed_teacher_task",
        "Args/student_checkpoint",
        "Args/student_task_id",
        "Args/student_model_id",
        "Args/student_checkpoint_sha256",
    }
)
STUDENT_HANDOFF_KEYS = frozenset(
    {
        "Args/student_checkpoint",
        "Args/student_task_id",
        "Args/student_model_id",
        "Args/student_checkpoint_sha256",
    }
)
SCRIPT_IDENTITY_KEYS = (
    "binary",
    "repository",
    "branch",
    "version_num",
    "tag",
    "working_dir",
    "entry_point",
    "diff",
    "requirements",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sealed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _require_valid_seal(payload: Mapping[str, object], *, context: str) -> None:
    observed = payload.get("seal_sha256")
    if type(observed) is not str or SHA256_PATTERN.fullmatch(observed) is None:
        raise RuntimeError(f"{context} has no valid seal_sha256")
    expected = _sealed(payload)["seal_sha256"]
    if observed != expected:
        raise RuntimeError(f"{context} seal_sha256 mismatch")


def _clearml_id(value: object, context: str) -> str:
    result = str(value or "")
    if CLEARML_ID_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase 32-hex ClearML ID")
    return result


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return result


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument(
        "--gate-task-id",
        help="explicit completed 12-condition validation task ID",
    )
    gate.add_argument(
        "--paper-controller-task-id",
        help="completed paper controller that publishes the validation task ID",
    )
    parser.add_argument(
        "--paper-controller-summary-artifact",
        default="paper_controller_summary",
    )
    parser.add_argument("--template-task-id", required=True)
    parser.add_argument("--teacher-task-id", required=True)
    parser.add_argument("--teacher-model-id", required=True)
    parser.add_argument("--teacher-checkpoint-sha256", required=True)
    parser.add_argument("--allow-failed-teacher-task", action="store_true")
    parser.add_argument("--worker-queue", default=DEFAULT_WORKER_QUEUE)
    parser.add_argument(
        "--worker-queues",
        default="",
        help="comma-separated queues for parallel slots (overrides --worker-queue)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="max concurrent training tasks (must be <= number of worker queues)",
    )
    parser.add_argument(
        "--canary-first",
        action="store_true",
        help="enqueue only the first new experiment until it completes, then fill slots",
    )
    parser.add_argument(
        "--adopt-experiment",
        action="append",
        default=[],
        metavar="NAME=TASK_ID",
        help="adopt an existing ClearML task into suite progress (repeatable)",
    )
    parser.add_argument(
        "--build-task-id",
        default="",
        help="override native BUILD_TASK_ID (multiarch rebuild retarget)",
    )
    parser.add_argument(
        "--native-bundle-bytes",
        type=int,
        default=0,
        help="override Args/native_bundle_bytes from the template",
    )
    parser.add_argument(
        "--native-bundle-sha256",
        default="",
        help="override Args/native_bundle_sha256 from the template",
    )
    parser.add_argument(
        "--build-manifest-sha256",
        default="",
        help="override Args/build_manifest_sha256 from the template",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--poll-seconds", type=_positive_float, default=30.0)
    return parser


def _resolve_worker_queues(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "worker_queues", "") or ""
    if type(raw) is str and raw.strip():
        queues = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        queue = getattr(args, "worker_queue", "")
        if type(queue) is not str or not queue.strip():
            raise ValueError("worker queue must be non-empty")
        queues = [queue.strip()]
    if not queues:
        raise ValueError("worker queues must be non-empty")
    if len(set(queues)) != len(queues):
        raise ValueError("worker queues must be unique")
    return queues


def _parse_adopt_experiments(values: Sequence[object]) -> dict[str, str]:
    adopted: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or "=" not in raw:
            raise ValueError(f"invalid --adopt-experiment value: {raw!r}")
        name, task_id = raw.split("=", 1)
        name = name.strip()
        task_id = task_id.strip()
        if name not in EXPERIMENT_BY_NAME:
            raise ValueError(f"unknown adopt experiment: {name!r}")
        if name in adopted:
            raise ValueError(f"duplicate adopt experiment: {name!r}")
        adopted[name] = _clearml_id(task_id, f"adopted {name} task")
    return adopted


def _normalized_task_status(task: object) -> str:
    status = getattr(task, "status", None)
    if callable(status):
        status = status()
    if status is None:
        getter = getattr(task, "get_status", None)
        if callable(getter):
            status = getter()
    value = getattr(status, "value", status)
    return str(value).rsplit(".", 1)[-1].lower()


def _reload(task: object) -> None:
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()


def _task_parameters(task: object) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError("task cannot enumerate parameters")
    try:
        parameters = getter(backwards_compatibility=False, cast=False)
    except TypeError:
        parameters = getter()
    if not isinstance(parameters, Mapping):
        raise RuntimeError("task returned invalid parameters")
    return {str(key): value for key, value in parameters.items()}


def _task_script(task: object) -> dict[str, object]:
    data = getattr(task, "data", None)
    script_object = getattr(data, "script", None)
    to_dict = getattr(script_object, "to_dict", None)
    script = to_dict() if callable(to_dict) else None
    if not isinstance(script, Mapping):
        getter = getattr(task, "get_script", None)
        if not callable(getter):
            raise RuntimeError("task cannot enumerate its script")
        script = getter()
    if not isinstance(script, Mapping):
        raise RuntimeError("task returned an invalid script mapping")
    return {key: script.get(key) for key in SCRIPT_IDENTITY_KEYS}


def _task_docker(task: object) -> str:
    getter = getattr(task, "get_base_docker", None)
    if not callable(getter):
        raise RuntimeError("task cannot enumerate its base Docker command")
    value = getter()
    if type(value) is not str or not value.strip():
        raise RuntimeError("task has no base Docker command")
    return value.strip()


def _parameter_matches(actual: object, expected: object) -> bool:
    if type(actual) is bool or type(expected) is bool:
        return str(actual).casefold() == str(expected).casefold()
    if type(actual) is int or type(expected) is int:
        return str(actual) == str(expected)
    return actual == expected


def _require_files_server_url(value: object, *, context: str) -> str:
    result = str(value or "")
    parsed = urlsplit(result)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != EXPECTED_FILES_SERVER_HOST
        or parsed.port != EXPECTED_FILES_SERVER_PORT
    ):
        raise RuntimeError(
            f"{context} must use {EXPECTED_FILES_SERVER_HOST}:"
            f"{EXPECTED_FILES_SERVER_PORT}: {result!r}"
        )
    return result


def _validate_docker_command(command: str) -> None:
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise RuntimeError(f"invalid template Docker command: {error}") from error
    if not tokens or tokens[0] != EXPECTED_DOCKER_IMAGE:
        raise RuntimeError("template Docker image mismatch")
    has_host_network = "--network=host" in tokens or any(
        tokens[index : index + 2] == ["--network", "host"]
        for index in range(max(0, len(tokens) - 1))
    )
    if not has_host_network:
        raise RuntimeError("template Docker command must use host networking")
    expected_endpoints = {
        endpoint.partition("=")[0]: endpoint for endpoint in EXPECTED_ENDPOINTS
    }
    observed_endpoints: dict[str, list[str]] = {}
    for token in tokens:
        candidate = token
        for prefix in ("--env=", "-e"):
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :]
                break
        key = candidate.partition("=")[0]
        if key in expected_endpoints:
            observed_endpoints.setdefault(key, []).append(candidate)
    # Agents inherit ClearML server configuration when these flags are absent.
    # If a task spells out any endpoint, require the complete exact .34 set.
    if observed_endpoints:
        if set(observed_endpoints) != set(expected_endpoints):
            raise RuntimeError(
                "template Docker command has incomplete ClearML endpoints"
            )
        for key, expected in expected_endpoints.items():
            if any(value != expected for value in observed_endpoints[key]):
                raise RuntimeError(f"template Docker endpoint mismatch: {key}")


def _validate_source_parameters(parameters: Mapping[str, object]) -> dict[str, object]:
    source = {}
    for key in SOURCE_PARAMETER_KEYS:
        value = parameters.get(key)
        if value in (None, ""):
            raise RuntimeError(f"template is missing sealed source parameter {key}")
        source[key] = value
    _clearml_id(source["Args/source_dataset_id"], "source dataset")
    _clearml_id(source["Args/training_dataset_id"], "training dataset")
    for key in ("Args/source_archive_bytes", "Args/native_bundle_bytes"):
        try:
            value = int(str(source[key]))
        except ValueError as error:
            raise RuntimeError(f"template {key} is not an integer") from error
        if value <= 0:
            raise RuntimeError(f"template {key} must be positive")
    for key in (
        "Args/source_archive_sha256",
        "Args/native_bundle_sha256",
        "Args/build_manifest_sha256",
    ):
        _sha256(source[key], key)
    archive_name = str(source["Args/source_archive_name"])
    if Path(archive_name).name != archive_name or not archive_name:
        raise RuntimeError("template source archive name must be a plain filename")
    return source


def _apply_experiment_syspath_patch(diff: str) -> str:
    """Ensure experiment bootstrap can import sealed package modules in-process."""

    if type(diff) is not str:
        raise RuntimeError("standalone script diff must be a string")
    if _EXPERIMENT_SYSPATH_MARKER in diff:
        patched = diff
    else:
        anchor_count = diff.count(_EXPERIMENT_SYSPATH_ANCHOR)
        if anchor_count == 0:
            # Unit-test fixtures and unrelated scripts stay unchanged.
            patched = diff
        elif anchor_count != 1:
            raise RuntimeError("experiment sys.path compatibility anchor is not unique")
        else:
            patched = diff.replace(
                _EXPERIMENT_SYSPATH_ANCHOR, _EXPERIMENT_SYSPATH_PATCH, 1
            )
    if _EXPERIMENT_DDP_UNUSED_MARKER not in patched:
        unused_count = patched.count(_EXPERIMENT_DDP_UNUSED_ANCHOR)
        if unused_count == 1:
            patched = patched.replace(
                _EXPERIMENT_DDP_UNUSED_ANCHOR, _EXPERIMENT_DDP_UNUSED_PATCH, 1
            )
        elif unused_count > 1:
            raise RuntimeError("experiment DDP unused-parameter anchor is not unique")
    if _EXPERIMENT_CAPABILITY_MARKER not in patched:
        capability_count = patched.count(_EXPERIMENT_CAPABILITY_ANCHOR)
        if capability_count == 1:
            patched = patched.replace(
                _EXPERIMENT_CAPABILITY_ANCHOR, _EXPERIMENT_CAPABILITY_PATCH, 1
            )
        elif capability_count > 1:
            raise RuntimeError("experiment GPU capability anchor is not unique")
    if _EXPERIMENT_RUNNER_LOAD_MARKER not in patched:
        runner_count = patched.count(_EXPERIMENT_RUNNER_LOAD_ANCHOR)
        if runner_count == 1:
            patched = patched.replace(
                _EXPERIMENT_RUNNER_LOAD_ANCHOR, _EXPERIMENT_RUNNER_LOAD_PATCH, 1
            )
        elif runner_count > 1:
            raise RuntimeError("experiment runner-load anchor is not unique")
    if _EXPERIMENT_SMOKE_CAPABILITY_MARKER not in patched:
        smoke_count = patched.count(_EXPERIMENT_SMOKE_CAPABILITY_ANCHOR)
        if smoke_count == 1:
            patched = patched.replace(
                _EXPERIMENT_SMOKE_CAPABILITY_ANCHOR,
                _EXPERIMENT_SMOKE_CAPABILITY_PATCH,
                1,
            )
        elif smoke_count > 1:
            raise RuntimeError("experiment smoke capability anchor is not unique")
    if _EXPERIMENT_MASTER_ADDR_MARKER not in patched:
        master_count = patched.count(_EXPERIMENT_MASTER_ADDR_ANCHOR)
        master_only_count = patched.count(_EXPERIMENT_MASTER_ADDR_ONLY_ANCHOR)
        if master_count == 1:
            patched = patched.replace(
                _EXPERIMENT_MASTER_ADDR_ANCHOR,
                _EXPERIMENT_MASTER_ADDR_PATCH,
                1,
            )
        elif master_only_count == 1:
            patched = patched.replace(
                _EXPERIMENT_MASTER_ADDR_ONLY_ANCHOR,
                _EXPERIMENT_MASTER_ADDR_PATCH,
                1,
            )
        elif master_count > 1 or master_only_count > 1:
            raise RuntimeError("experiment MASTER_ADDR/NCCL anchor is not unique")
    if _EXPERIMENT_HOSTNAME_MARKER not in patched:
        host_count = patched.count(_EXPERIMENT_HOSTNAME_ANCHOR)
        if host_count == 1:
            patched = patched.replace(
                _EXPERIMENT_HOSTNAME_ANCHOR,
                _EXPERIMENT_HOSTNAME_PATCH,
                1,
            )
        elif host_count > 1:
            raise RuntimeError("experiment hostname-resolve anchor is not unique")
    if BUILD_TASK_ID != SEALED_TEMPLATE_BUILD_TASK_ID:
        sealed_assign = (
            f'{_EXPERIMENT_BUILD_TASK_MARKER_PREFIX}{SEALED_TEMPLATE_BUILD_TASK_ID}"'
        )
        active_assign = f'{_EXPERIMENT_BUILD_TASK_MARKER_PREFIX}{BUILD_TASK_ID}"'
        if active_assign not in patched:
            assign_count = patched.count(sealed_assign)
            if assign_count == 1:
                patched = patched.replace(sealed_assign, active_assign, 1)
            elif assign_count > 1:
                raise RuntimeError("experiment BUILD_TASK_ID assignment is not unique")
            elif SEALED_TEMPLATE_BUILD_TASK_ID in patched:
                # Fallback for string literals that are not the assignment form.
                if patched.count(SEALED_TEMPLATE_BUILD_TASK_ID) < 1:
                    raise RuntimeError("sealed BUILD_TASK_ID marker missing from script")
                patched = patched.replace(SEALED_TEMPLATE_BUILD_TASK_ID, BUILD_TASK_ID)
    return patched


def _script_sha256(script: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(script).encode("utf-8")).hexdigest()


def _ensure_clone_experiment_syspath_fix(task: object) -> None:
    """Patch a freshly cloned suite task before identity/enqueue checks."""

    script = _task_script(task)
    diff = script.get("diff")
    patched = _apply_experiment_syspath_patch(str(diff or ""))
    if patched == diff:
        return
    task_id = str(getattr(task, "id", "") or "")
    if not task_id:
        raise RuntimeError("cloned task has no ID for sys.path patch")
    from clearml.backend_api.session.client import APIClient

    APIClient().tasks.edit(task=task_id, script={"diff": patched})
    _reload(task)


def _template_identity(task: object, *, expected_task_id: str) -> dict[str, object]:
    if _normalized_task_status(task) != "completed":
        raise RuntimeError("bootstrap template task must be completed")
    if str(getattr(task, "id", "") or "") != expected_task_id:
        raise RuntimeError("bootstrap template task ID mismatch")
    script = _task_script(task)
    if script.get("entry_point") != EXPECTED_ENTRYPOINT:
        raise RuntimeError(
            f"template entrypoint must be exactly {EXPECTED_ENTRYPOINT!r}"
        )
    diff = script.get("diff")
    if type(diff) is not str:
        raise RuntimeError("template standalone script diff is missing")
    markers = (
        BASE_IMAGE_AMD64_MANIFEST_DIGEST,
        BASE_IMAGE_CONFIG_DIGEST,
        SEALED_TEMPLATE_BUILD_TASK_ID,
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    )
    for marker in markers:
        if marker not in diff:
            raise RuntimeError(f"template standalone script marker missing: {marker}")
    docker = _task_docker(task)
    _validate_docker_command(docker)
    parameters = _task_parameters(task)
    source_parameters = _validate_source_parameters(parameters)
    patched_script = dict(script)
    patched_script["diff"] = _apply_experiment_syspath_patch(diff)
    return {
        "task_id": expected_task_id,
        "entry_point": EXPECTED_ENTRYPOINT,
        "script_sha256": _script_sha256(patched_script),
        "docker_command": docker,
        "docker_image": EXPECTED_DOCKER_IMAGE,
        "base_image_manifest_digest": BASE_IMAGE_AMD64_MANIFEST_DIGEST,
        "base_image_config_digest": BASE_IMAGE_CONFIG_DIGEST,
        "native_build_task_id": BUILD_TASK_ID,
        "source_parameters": source_parameters,
    }


def _validate_clone_identity(
    task: object,
    *,
    identity: Mapping[str, object],
) -> None:
    script = _task_script(task)
    observed_script_sha = _script_sha256(script)
    if observed_script_sha != identity.get("script_sha256"):
        raise RuntimeError("cloned task script drifted from the verified template")
    if script.get("entry_point") != EXPECTED_ENTRYPOINT:
        raise RuntimeError("cloned task entrypoint drifted")
    docker = _task_docker(task)
    if docker != identity.get("docker_command"):
        raise RuntimeError("cloned task Docker command drifted from template")
    _validate_docker_command(docker)
    parameters = _task_parameters(task)
    source = identity.get("source_parameters")
    if not isinstance(source, Mapping):
        raise RuntimeError("template identity has invalid source parameters")
    for key, expected in source.items():
        if not _parameter_matches(parameters.get(str(key)), expected):
            raise RuntimeError(f"cloned task sealed parameter drifted: {key}")


def _experiment_parameters(
    template_parameters: Mapping[str, object],
    *,
    experiment: str,
    predecessor_task_id: str,
    teacher_task_id: str,
    teacher_model_id: str,
    teacher_checkpoint_sha256: str,
    allow_failed_teacher_task: bool,
) -> dict[str, object]:
    spec = EXPERIMENT_BY_NAME[experiment]
    result = {
        key: value
        for key, value in template_parameters.items()
        if key not in DYNAMIC_PARAMETER_KEYS
    }
    result.update(
        {
            "Args/experiment_from_task": experiment,
            "Args/predecessor_task_id": predecessor_task_id,
            "Args/gpus": EXPECTED_GPU_COUNT,
            "Args/stage": "all",
            "Args/max_epochs": EXPERIMENT_MAX_EPOCHS,
            "Args/amp": False,
        }
    )
    if spec.requires_teacher:
        result.update(
            {
                "Args/teacher_task_id": teacher_task_id,
                "Args/teacher_model_id": teacher_model_id,
                "Args/teacher_checkpoint_sha256": teacher_checkpoint_sha256,
                "Args/allow_failed_teacher_task": allow_failed_teacher_task,
            }
        )
    return result


def _execution_parameters_match(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    """Compare sealed execution params, tolerating empty inherited dynamic keys."""

    missing = set(expected) - set(observed)
    if missing:
        return False
    extras = set(observed) - set(expected)
    for key in extras:
        if key not in DYNAMIC_PARAMETER_KEYS:
            return False
        if observed.get(key) not in (None, "", False, "False"):
            return False
    return all(
        _parameter_matches(observed.get(key), value) for key, value in expected.items()
    )


def _set_and_validate_parameters(
    task: object,
    *,
    expected: Mapping[str, object],
    experiment: str,
) -> None:
    setter = getattr(task, "set_parameters", None)
    if not callable(setter):
        raise RuntimeError("cloned task cannot replace its parameters")
    setter(dict(expected))
    observed = _task_parameters(task)
    if not _execution_parameters_match(observed, expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(
            key
            for key in set(observed) - set(expected)
            if key not in DYNAMIC_PARAMETER_KEYS
            or observed.get(key) not in (None, "", False, "False")
        )
        raise RuntimeError(
            f"cloned task parameter keys mismatch; missing={missing}, extra={extra}"
        )
    leaked = [
        key for key in STUDENT_HANDOFF_KEYS if observed.get(key) not in (None, "")
    ]
    if leaked:
        raise RuntimeError(f"cloned task leaked student handoff parameters: {leaked}")
    teacher_keys = (
        "Args/teacher_task_id",
        "Args/teacher_model_id",
        "Args/teacher_checkpoint_sha256",
        "Args/allow_failed_teacher_task",
    )
    if not EXPERIMENT_BY_NAME[experiment].requires_teacher and any(
        observed.get(key) not in (None, "", False, "False") for key in teacher_keys
    ):
        raise RuntimeError(
            f"teacher-free experiment {experiment!r} leaked teacher data"
        )


def _artifact_payload(task: object, name: str) -> dict[str, object]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"task is missing required artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise RuntimeError(f"artifact {name!r} cannot be downloaded")
    value = getter()
    if isinstance(value, Mapping):
        return dict(value)
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise RuntimeError(f"artifact {name!r} is not a JSON object") from error
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"artifact {name!r} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"artifact {name!r} is not a JSON object")
    return dict(payload)


def _validate_teacher_reference(task_class: object, args: argparse.Namespace) -> None:
    teacher = task_class.get_task(task_id=args.teacher_task_id)
    status = _normalized_task_status(teacher)
    allowed = status == "completed" or (
        args.allow_failed_teacher_task and status == "failed"
    )
    if not allowed:
        raise RuntimeError(f"teacher task has unusable status: {status!r}")
    getter = getattr(teacher, "get_models", None)
    models = getter() if callable(getter) else None
    if not isinstance(models, Mapping):
        raise RuntimeError("teacher task returned an invalid model mapping")
    outputs = models.get("output")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise RuntimeError("teacher task has no output model sequence")
    candidates = [
        model
        for model in outputs
        if getattr(model, "name", None) == "ResilientV2X clean teacher"
        and str(getattr(model, "id", "") or "") == args.teacher_model_id
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "teacher handoff must resolve exactly one pinned OutputModel"
        )
    model = candidates[0]
    if str(getattr(model, "task", "") or "") != args.teacher_task_id:
        raise RuntimeError("teacher OutputModel ownership mismatch")
    _require_files_server_url(getattr(model, "url", ""), context="teacher model URL")


def _wait_for_completed(
    task: object,
    *,
    context: str,
    poll_seconds: float,
    sleeper: Callable[[float], None],
    on_status: Callable[[str], None] | None = None,
) -> object:
    while True:
        status = _normalized_task_status(task)
        if on_status is not None:
            on_status(status)
        if status == "completed":
            return task
        if status in FAILED_STATUSES:
            raise RuntimeError(f"{context} ended without completion: {status!r}")
        if status not in WAITABLE_STATUSES:
            raise RuntimeError(f"{context} has unexpected status: {status!r}")
        sleeper(poll_seconds)
        _reload(task)


def _extract_validation_task_id(payload: Mapping[str, object]) -> str | None:
    for key in ("validation_task_id", "gate_task_id"):
        value = payload.get(key)
        if type(value) is str and CLEARML_ID_PATTERN.fullmatch(value):
            return value
    for key in ("validation", "validation_task", "result"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            task_id = nested.get("task_id")
            if type(task_id) is str and CLEARML_ID_PATTERN.fullmatch(task_id):
                return task_id
            result = _extract_validation_task_id(nested)
            if result is not None:
                return result
    return None


def _resolve_gate_task_id(
    args: argparse.Namespace,
    *,
    task_class: object,
    sleeper: Callable[[float], None],
) -> str:
    if args.gate_task_id:
        return _clearml_id(args.gate_task_id, "gate task")
    controller_id = _clearml_id(
        args.paper_controller_task_id,
        "paper controller task",
    )
    paper_controller = task_class.get_task(task_id=controller_id)
    _wait_for_completed(
        paper_controller,
        context="paper experiment controller",
        poll_seconds=args.poll_seconds,
        sleeper=sleeper,
    )
    _reload(paper_controller)
    try:
        payload = _artifact_payload(
            paper_controller,
            args.paper_controller_summary_artifact,
        )
    except RuntimeError:
        payload = {}
    resolved = _extract_validation_task_id(payload)
    if resolved is None:
        parameters = _task_parameters(paper_controller)
        for key in (
            "Controller/validation_task_id",
            "General/validation_task_id",
            "Args/validation_task_id",
        ):
            value = parameters.get(key)
            if type(value) is str and CLEARML_ID_PATTERN.fullmatch(value):
                resolved = value
                break
    if resolved is None:
        raise RuntimeError("paper controller did not publish a validation task ID")
    return resolved


def _task_name(index: int, experiment: str, controller_task_id: str) -> str:
    return f"ResilientV2X post-main {index:02d} {experiment} [{controller_task_id}]"


def _task_parent(task: object) -> str:
    return str(getattr(task, "parent", "") or "")


def _find_recoverable_clone(
    task_class: object,
    *,
    project: str,
    name: str,
    controller_task_id: str,
) -> object | None:
    getter = getattr(task_class, "get_tasks", None)
    if not callable(getter):
        raise RuntimeError("Task class cannot search for recoverable clones")
    candidates = []
    try:
        candidates = list(
            getter(
                project_name=project,
                task_name=f"^{re.escape(name)}$",
                allow_archived=True,
                task_filter={"parent": controller_task_id},
            )
            or []
        )
    except Exception:
        candidates = []
    if not candidates:
        try:
            candidates = list(
                getter(
                    task_name=f"^{re.escape(name)}$",
                    allow_archived=True,
                    task_filter={"parent": controller_task_id},
                )
                or []
            )
        except TypeError:
            try:
                candidates = list(getter(task_name=f"^{re.escape(name)}$") or [])
            except Exception:
                candidates = []
        except Exception:
            candidates = []
    exact = [
        task
        for task in candidates
        if str(getattr(task, "name", "") or "") == name
        and _task_parent(task) == controller_task_id
        and _normalized_task_status(task) not in FAILED_STATUSES
    ]
    if len(exact) > 1:
        raise RuntimeError(f"multiple recoverable clones found for {name!r}")
    return exact[0] if exact else None


def _require_unique_final_model(task: object, *, experiment: str) -> object:
    getter = getattr(task, "get_models", None)
    models = getter() if callable(getter) else None
    if not isinstance(models, Mapping):
        raise RuntimeError("completed experiment returned an invalid model mapping")
    outputs = models.get("output")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise RuntimeError("completed experiment has no output model sequence")
    expected_name = f"ResilientV2X {experiment} final checkpoint"
    candidates = [
        model for model in outputs if getattr(model, "name", None) == expected_name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"completed experiment {experiment!r} must expose exactly one "
            f"final OutputModel; found {len(candidates)}"
        )
    model = candidates[0]
    task_id = str(getattr(task, "id", "") or "")
    if str(getattr(model, "task", "") or "") != task_id:
        raise RuntimeError("final OutputModel ownership mismatch")
    _clearml_id(getattr(model, "id", ""), "final model")
    url = _require_files_server_url(
        getattr(model, "url", ""),
        context="final OutputModel URL",
    )
    if Path(unquote(urlsplit(url).path)).name != f"{experiment}_epoch_50.pth":
        raise RuntimeError("final OutputModel filename mismatch")
    return model


def _require_contract_value(
    contract: Mapping[str, object],
    key: str,
    expected: object,
) -> None:
    if not _parameter_matches(contract.get(key), expected):
        raise RuntimeError(
            f"experiment run contract {key} mismatch: "
            f"expected {expected!r}, got {contract.get(key)!r}"
        )


def _validate_completed_experiment(
    task: object,
    *,
    experiment: str,
    predecessor_task_id: str,
    expected_parameters: Mapping[str, object],
    identity: Mapping[str, object],
    teacher_task_id: str,
    teacher_model_id: str,
    teacher_checkpoint_sha256: str,
    require_clone_identity: bool = True,
) -> dict[str, object]:
    if _normalized_task_status(task) != "completed":
        raise RuntimeError(f"experiment {experiment!r} is not completed")
    if require_clone_identity:
        _validate_clone_identity(task, identity=identity)
    observed_parameters = _task_parameters(task)
    if not _execution_parameters_match(observed_parameters, expected_parameters):
        raise RuntimeError("completed experiment parameter keys drifted")

    contract = _artifact_payload(task, RUN_CONTRACT_ARTIFACT)
    task_id = str(getattr(task, "id", "") or "")
    expected_contract = {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": experiment,
        "source_dataset_id": expected_parameters["Args/source_dataset_id"],
        "training_dataset_id": expected_parameters["Args/training_dataset_id"],
        "native_build_task_id": BUILD_TASK_ID,
        "native_bundle_sha256": expected_parameters["Args/native_bundle_sha256"],
        "build_manifest_sha256": expected_parameters["Args/build_manifest_sha256"],
        "base_image_manifest_digest": BASE_IMAGE_AMD64_MANIFEST_DIGEST,
        "predecessor_task_id": predecessor_task_id,
        "gpus": EXPECTED_GPU_COUNT,
        "ddp_processes": EXPECTED_GPU_COUNT,
        "max_epochs": EXPERIMENT_MAX_EPOCHS,
        "amp": False,
        "precision": "FP32",
        "runtime_profile": "rtx5090",
        "val_interval": 10,
        "per_epoch_validation": False,
        "condition_evaluation": False,
    }
    for key, expected in expected_contract.items():
        _require_contract_value(contract, key, expected)
    source_archive = contract.get("source_archive")
    if not isinstance(source_archive, Mapping):
        raise RuntimeError("experiment run contract has invalid source_archive")
    archive_expected = {
        "name": expected_parameters["Args/source_archive_name"],
        "size_bytes": expected_parameters["Args/source_archive_bytes"],
        "sha256": expected_parameters["Args/source_archive_sha256"],
    }
    for key, expected in archive_expected.items():
        if not _parameter_matches(source_archive.get(key), expected):
            raise RuntimeError(f"experiment source archive contract drifted: {key}")
    teacher = contract.get("teacher")
    if EXPERIMENT_BY_NAME[experiment].requires_teacher:
        if not isinstance(teacher, Mapping):
            raise RuntimeError("teacher-dependent experiment has no teacher contract")
        teacher_expected = {
            "task_id": teacher_task_id,
            "model_id": teacher_model_id,
            "sha256": teacher_checkpoint_sha256,
        }
        for key, expected in teacher_expected.items():
            if teacher.get(key) != expected:
                raise RuntimeError(f"experiment teacher contract drifted: {key}")
    elif teacher is not None:
        raise RuntimeError("teacher-free experiment emitted a teacher contract")

    model = _require_unique_final_model(task, experiment=experiment)
    checkpoint = _artifact_payload(task, FINAL_CHECKPOINT_ARTIFACT)
    model_id = str(getattr(model, "id", "") or "")
    model_url = str(getattr(model, "url", "") or "")
    checkpoint_expected = {
        "model_id": model_id,
        "name": f"ResilientV2X {experiment} final checkpoint",
        "url": model_url,
        "filename": "epoch_50.pth",
    }
    for key, expected in checkpoint_expected.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"final checkpoint contract drifted: {key}")
    checkpoint_sha = _sha256(checkpoint.get("sha256"), "final checkpoint")
    try:
        size_bytes = int(str(checkpoint.get("size_bytes")))
    except ValueError as error:
        raise RuntimeError("final checkpoint size is invalid") from error
    if size_bytes <= 0:
        raise RuntimeError("final checkpoint is empty")
    return {
        "task_id": task_id,
        "model_id": model_id,
        "model_name": checkpoint_expected["name"],
        "model_url": model_url,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": size_bytes,
        "run_contract_artifact": RUN_CONTRACT_ARTIFACT,
        "final_checkpoint_artifact": FINAL_CHECKPOINT_ARTIFACT,
    }


def _new_progress(
    *,
    controller_task_id: str,
    gate_task_id: str,
    template_identity: Mapping[str, object],
    teacher_task_id: str,
    teacher_model_id: str,
    teacher_checkpoint_sha256: str,
    allow_failed_teacher_task: bool,
    worker_queues: Sequence[str],
    max_parallel_training_tasks: int,
) -> dict[str, object]:
    queues = [str(item) for item in worker_queues]
    return {
        "schema_version": 1,
        "controller_type": "resilient_v2x_post_main_sequential_training",
        "controller_task_id": controller_task_id,
        "gate_task_id": gate_task_id,
        "gate_policy": "exact_task_must_be_completed_before_any_clone",
        "template": dict(template_identity),
        "teacher": {
            "task_id": teacher_task_id,
            "model_id": teacher_model_id,
            "checkpoint_sha256": teacher_checkpoint_sha256,
            "allow_failed_task": allow_failed_teacher_task,
        },
        "worker_queue": queues[0],
        "worker_queues": queues,
        "experiment_order": list(EXPERIMENT_ORDER),
        "max_parallel_training_tasks": int(max_parallel_training_tasks),
        "revision": 0,
        "created_at": _now(),
        "updated_at": _now(),
        "steps": [
            {
                "index": index,
                "experiment": experiment,
                "state": "pending",
                "task_name": _task_name(index, experiment, controller_task_id),
                "task_id": None,
                "predecessor_task_id": None,
                "result": None,
                "worker_queue": None,
                "adopted": False,
            }
            for index, experiment in enumerate(EXPERIMENT_ORDER, start=1)
        ],
    }


def _upload_mapping(task: object, name: str, payload: Mapping[str, object]) -> None:
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader):
        raise RuntimeError("controller task cannot upload artifacts")
    if not uploader(name, artifact_object=dict(payload), wait_on_upload=True):
        raise RuntimeError(f"failed to upload controller artifact {name!r}")
    flusher = getattr(task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)


def _save_progress(task: object, progress: MutableMapping[str, object]) -> None:
    progress["revision"] = int(progress.get("revision", 0)) + 1
    progress["updated_at"] = _now()
    sealed = _sealed(progress)
    _upload_mapping(task, PROGRESS_ARTIFACT, sealed)
    progress.clear()
    progress.update(sealed)


def _validate_progress(
    progress: Mapping[str, object],
    *,
    controller_task_id: str,
    gate_task_id: str,
    template_identity: Mapping[str, object],
    teacher_task_id: str,
    teacher_model_id: str,
    teacher_checkpoint_sha256: str,
    allow_failed_teacher_task: bool,
    worker_queues: Sequence[str],
    max_parallel_training_tasks: int,
) -> None:
    _require_valid_seal(progress, context="training progress")
    queues = [str(item) for item in worker_queues]
    expected = {
        "schema_version": 1,
        "controller_type": "resilient_v2x_post_main_sequential_training",
        "controller_task_id": controller_task_id,
        "gate_task_id": gate_task_id,
        "worker_queue": queues[0],
        "worker_queues": queues,
        "max_parallel_training_tasks": int(max_parallel_training_tasks),
    }
    for key, value in expected.items():
        if progress.get(key) != value:
            raise RuntimeError(f"training progress {key} mismatch")
    if progress.get("template") != dict(template_identity):
        raise RuntimeError("training progress template identity mismatch")
    teacher = progress.get("teacher")
    expected_teacher = {
        "task_id": teacher_task_id,
        "model_id": teacher_model_id,
        "checkpoint_sha256": teacher_checkpoint_sha256,
        "allow_failed_task": allow_failed_teacher_task,
    }
    if teacher != expected_teacher:
        raise RuntimeError("training progress teacher identity mismatch")
    if progress.get("experiment_order") != list(EXPERIMENT_ORDER):
        raise RuntimeError("training progress experiment order mismatch")
    steps = progress.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise RuntimeError("training progress steps are invalid")
    if len(steps) != len(EXPERIMENT_ORDER):
        raise RuntimeError("training progress step count mismatch")
    task_ids: list[str] = []
    active = 0
    seen_pending = False
    for index, (step, experiment) in enumerate(
        zip(steps, EXPERIMENT_ORDER, strict=True),
        start=1,
    ):
        if not isinstance(step, Mapping):
            raise RuntimeError("training progress step is not an object")
        if step.get("index") != index or step.get("experiment") != experiment:
            raise RuntimeError("training progress step identity mismatch")
        state = step.get("state")
        if state not in PROGRESS_STATES:
            raise RuntimeError("training progress contains an invalid state")
        task_id = step.get("task_id")
        if task_id is not None:
            task_ids.append(_clearml_id(task_id, f"progress task {experiment}"))
        if state == "pending":
            seen_pending = True
            if task_id is not None:
                raise RuntimeError("pending training progress step already has a task ID")
        elif seen_pending:
            # Pending steps must form a suffix; earlier slots may still be active.
            raise RuntimeError("training progress pending steps are not a suffix")
        if state == "completed":
            if task_id is None or not isinstance(step.get("result"), Mapping):
                raise RuntimeError("completed training progress step is incomplete")
        if state in {"created", "queued", "running"}:
            if task_id is None:
                raise RuntimeError("active training progress step has no task ID")
            # Adopted out-of-pool jobs (e.g. still finishing on 5090) do not
            # consume dual-queue parallel slots.
            if step.get("worker_queue") != "adopted-external" and not step.get(
                "adopted"
            ):
                active += 1
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("training progress reuses a task ID")
    if active > int(max_parallel_training_tasks):
        raise RuntimeError("training progress contains too many active tasks")


def _load_or_create_progress(
    controller_task: object,
    **identity: object,
) -> dict[str, object]:
    artifacts = getattr(controller_task, "artifacts", None)
    if isinstance(artifacts, Mapping) and PROGRESS_ARTIFACT in artifacts:
        progress = _artifact_payload(controller_task, PROGRESS_ARTIFACT)
        _validate_progress(progress, **identity)
        return progress
    progress = _new_progress(**identity)
    _save_progress(controller_task, progress)
    return progress


def _state_from_status(status: str) -> str:
    if status == "in_progress":
        return "running"
    if status in {"created", "queued", "completed"}:
        return status
    if status in FAILED_STATUSES:
        return "failed"
    raise RuntimeError(f"cannot map task status {status!r} into progress")


def _enqueue(task_class: object, task: object, *, worker_queue: str) -> None:
    response = task_class.enqueue(task=task, queue_name=worker_queue)
    if response is None or response is False:
        raise RuntimeError("ClearML did not confirm task enqueue")
    if isinstance(response, Mapping) and response.get("queued") == 0:
        raise RuntimeError("ClearML reported zero enqueued tasks")


def _latest_completed_predecessor(
    steps: Sequence[Mapping[str, object]],
    *,
    gate_task_id: str,
    before_index: int | None = None,
) -> str:
    predecessor = gate_task_id
    for step in steps:
        if before_index is not None and int(step["index"]) >= before_index:
            break
        if step.get("state") == "completed" and step.get("task_id"):
            predecessor = str(step["task_id"])
    return predecessor


def _active_worker_queues(steps: Sequence[Mapping[str, object]]) -> set[str]:
    active: set[str] = set()
    for step in steps:
        if step.get("state") in {"created", "queued", "running"}:
            queue = step.get("worker_queue")
            if type(queue) is str and queue.strip():
                active.add(queue.strip())
    return active


def _task_reports_training_iteration(task: object) -> bool:
    """True once ClearML has recorded at least one training iteration.

    Some sealed mmengine runs log train steps to the console without publishing
    ClearML scalar iterations. Treat ``grad_norm`` + ``loss`` console lines as
    proof the canary passed DDP init and entered the train loop.
    """

    getter = getattr(task, "get_last_iteration", None)
    if callable(getter):
        try:
            last_iteration = getter()
        except Exception:
            last_iteration = None
        if isinstance(last_iteration, int) and last_iteration > 0:
            return True
    console_getter = getattr(task, "get_reported_console_output", None)
    if not callable(console_getter):
        return False
    try:
        lines = console_getter(120) or []
    except Exception:
        return False
    text = "\n".join(str(line) for line in lines)
    if "Epoch(train)" in text:
        return True
    return "grad_norm:" in text and "loss:" in text


def _canary_blocks_extra_slots(
    steps: Sequence[Mapping[str, object]],
    *,
    canary_first: bool,
    task_class: object | None = None,
) -> bool:
    if not canary_first:
        return False
    owned_completed = any(
        step.get("state") == "completed" and not step.get("adopted") for step in steps
    )
    if owned_completed:
        return False
    if task_class is None:
        return True
    for step in steps:
        if step.get("adopted") or step.get("state") not in {"queued", "running"}:
            continue
        task_id = step.get("task_id")
        if task_id is None:
            continue
        try:
            task = task_class.get_task(task_id=str(task_id))
        except Exception:
            continue
        if _task_reports_training_iteration(task):
            return False
    return True


def _ensure_experiment_task(
    *,
    task_class: object,
    template_task: object,
    template_identity: Mapping[str, object],
    template_parameters: Mapping[str, object],
    controller_task: object,
    controller_task_id: str,
    progress: MutableMapping[str, object],
    step: MutableMapping[str, object],
    index: int,
    experiment: str,
    predecessor_task_id: str,
    teacher_task_id: str,
    teacher_model_id: str,
    teacher_sha: str,
    allow_failed_teacher_task: bool,
    project: str,
    adopted: bool,
) -> tuple[object, dict[str, object]]:
    expected_parameters = _experiment_parameters(
        template_parameters,
        experiment=experiment,
        predecessor_task_id=predecessor_task_id,
        teacher_task_id=teacher_task_id,
        teacher_model_id=teacher_model_id,
        teacher_checkpoint_sha256=teacher_sha,
        allow_failed_teacher_task=allow_failed_teacher_task,
    )
    task_name = _task_name(index, experiment, controller_task_id)
    task_id_value = step.get("task_id")
    if task_id_value is not None:
        task = task_class.get_task(
            task_id=_clearml_id(task_id_value, f"{experiment} task")
        )
    else:
        task = _find_recoverable_clone(
            task_class,
            project=project,
            name=task_name,
            controller_task_id=controller_task_id,
        )
        if task is None:
            task = task_class.clone(
                source_task=template_task,
                name=task_name,
                parent=controller_task_id,
            )
            status = _normalized_task_status(task)
            if status != "created":
                raise RuntimeError(
                    f"new clone for {experiment!r} is not created: {status!r}"
                )
            _ensure_clone_experiment_syspath_fix(task)
            _set_and_validate_parameters(
                task,
                expected=expected_parameters,
                experiment=experiment,
            )
        elif _normalized_task_status(task) == "created":
            _ensure_clone_experiment_syspath_fix(task)
            _validate_clone_identity(task, identity=template_identity)
            _set_and_validate_parameters(
                task,
                expected=expected_parameters,
                experiment=experiment,
            )
        task_id = _clearml_id(getattr(task, "id", ""), f"{experiment} task")
        recovered_status = _normalized_task_status(task)
        recovered_state = (
            "running"
            if recovered_status == "completed"
            else _state_from_status(recovered_status)
        )
        step.update(
            {
                "state": recovered_state,
                "task_id": task_id,
                "task_name": task_name,
                "predecessor_task_id": predecessor_task_id,
                "adopted": bool(adopted),
            }
        )
        _save_progress(controller_task, progress)

    if not adopted:
        if str(getattr(task, "name", "") or "") != task_name:
            raise RuntimeError(f"recovered task name mismatch for {experiment!r}")
        if _task_parent(task) != controller_task_id:
            raise RuntimeError(f"recovered task parent mismatch for {experiment!r}")
    if step.get("predecessor_task_id") is None:
        step["predecessor_task_id"] = predecessor_task_id
    if adopted:
        return task, expected_parameters
    if _normalized_task_status(task) == "created":
        _ensure_clone_experiment_syspath_fix(task)
    _validate_clone_identity(task, identity=template_identity)
    observed = _task_parameters(task)
    if not _execution_parameters_match(observed, expected_parameters):
        raise RuntimeError(f"execution parameters drifted for {experiment!r}")
    return task, expected_parameters


def _seal_completed_step(
    *,
    task: object,
    step: MutableMapping[str, object],
    progress: MutableMapping[str, object],
    controller_task: object,
    experiment: str,
    predecessor_task_id: str,
    expected_parameters: Mapping[str, object],
    template_identity: Mapping[str, object],
    teacher_task_id: str,
    teacher_model_id: str,
    teacher_sha: str,
) -> dict[str, object]:
    _reload(task)
    try:
        result = _validate_completed_experiment(
            task,
            experiment=experiment,
            predecessor_task_id=predecessor_task_id,
            expected_parameters=expected_parameters,
            identity=template_identity,
            teacher_task_id=teacher_task_id,
            teacher_model_id=teacher_model_id,
            teacher_checkpoint_sha256=teacher_sha,
            require_clone_identity=not bool(step.get("adopted")),
        )
    except (RuntimeError, ValueError) as error:
        step["state"] = "failed"
        step["failure_status"] = "completion_validation_failed"
        step["failure_message"] = str(error)
        _save_progress(controller_task, progress)
        raise
    step["state"] = "completed"
    step["result"] = result
    step["completed_at"] = _now()
    _save_progress(controller_task, progress)
    return result


def run_training_suite(
    args: argparse.Namespace,
    *,
    task_class: object,
    controller_task: object,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Execute or resume the suite without importing PipelineController."""

    controller_task_id = _clearml_id(
        getattr(controller_task, "id", ""),
        "controller task",
    )
    template_task_id = _clearml_id(args.template_task_id, "template task")
    teacher_task_id = _clearml_id(args.teacher_task_id, "teacher task")
    teacher_model_id = _clearml_id(args.teacher_model_id, "teacher model")
    teacher_sha = _sha256(
        args.teacher_checkpoint_sha256,
        "teacher checkpoint",
    )
    worker_queues = _resolve_worker_queues(args)
    max_parallel = int(getattr(args, "max_parallel", 1) or 1)
    if max_parallel < 1:
        raise ValueError("max parallel must be positive")
    if max_parallel > len(worker_queues):
        raise ValueError("max parallel cannot exceed worker queue count")
    canary_first = bool(getattr(args, "canary_first", False))
    adopt_map = _parse_adopt_experiments(getattr(args, "adopt_experiment", []) or [])
    if type(args.project) is not str or not args.project.strip():
        raise ValueError("project must be non-empty")
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")

    gate_task_id = _resolve_gate_task_id(
        args,
        task_class=task_class,
        sleeper=sleeper,
    )
    gate_task = task_class.get_task(task_id=gate_task_id)
    _wait_for_completed(
        gate_task,
        context="post-main 12-condition validation gate",
        poll_seconds=args.poll_seconds,
        sleeper=sleeper,
    )
    _validate_teacher_reference(task_class, args)

    global BUILD_TASK_ID
    build_override = str(getattr(args, "build_task_id", "") or "").strip()
    if build_override:
        BUILD_TASK_ID = _clearml_id(build_override, "build task")

    template_task = task_class.get_task(task_id=template_task_id)
    template_identity = _template_identity(
        template_task,
        expected_task_id=template_task_id,
    )
    template_parameters = dict(_task_parameters(template_task))
    bundle_bytes = int(getattr(args, "native_bundle_bytes", 0) or 0)
    bundle_sha = str(getattr(args, "native_bundle_sha256", "") or "").strip()
    manifest_sha = str(getattr(args, "build_manifest_sha256", "") or "").strip()
    if bundle_bytes or bundle_sha or manifest_sha:
        if not (bundle_bytes > 0 and bundle_sha and manifest_sha):
            raise ValueError(
                "native bundle overrides require --native-bundle-bytes, "
                "--native-bundle-sha256, and --build-manifest-sha256 together"
            )
        template_parameters["Args/native_bundle_bytes"] = bundle_bytes
        template_parameters["Args/native_bundle_sha256"] = _sha256(
            bundle_sha, "native bundle"
        )
        template_parameters["Args/build_manifest_sha256"] = _sha256(
            manifest_sha, "build manifest"
        )
    template_identity = dict(template_identity)
    template_identity["source_parameters"] = _validate_source_parameters(
        template_parameters
    )
    template_identity["native_build_task_id"] = BUILD_TASK_ID
    identity = {
        "controller_task_id": controller_task_id,
        "gate_task_id": gate_task_id,
        "template_identity": template_identity,
        "teacher_task_id": teacher_task_id,
        "teacher_model_id": teacher_model_id,
        "teacher_checkpoint_sha256": teacher_sha,
        "allow_failed_teacher_task": bool(args.allow_failed_teacher_task),
        "worker_queues": worker_queues,
        "max_parallel_training_tasks": max_parallel,
    }
    progress = _load_or_create_progress(controller_task, **identity)
    steps = progress["steps"]
    if not isinstance(steps, list):
        raise RuntimeError("training progress steps must be a mutable list")

    # Seed adopted tasks into pending slots before the main scheduler loop.
    for experiment, task_id in adopt_map.items():
        index = EXPERIMENT_ORDER.index(experiment) + 1
        step = steps[index - 1]
        if not isinstance(step, MutableMapping):
            raise RuntimeError("training progress step must be mutable")
        if step.get("task_id") not in {None, task_id}:
            raise RuntimeError(f"adopt conflict for {experiment!r}")
        if step.get("state") == "completed" and step.get("task_id") == task_id:
            continue
        predecessor_task_id = _latest_completed_predecessor(
            steps,
            gate_task_id=gate_task_id,
            before_index=index,
        )
        step["task_id"] = task_id
        step["adopted"] = True
        step["predecessor_task_id"] = predecessor_task_id
        task, expected_parameters = _ensure_experiment_task(
            task_class=task_class,
            template_task=template_task,
            template_identity=template_identity,
            template_parameters=template_parameters,
            controller_task=controller_task,
            controller_task_id=controller_task_id,
            progress=progress,
            step=step,
            index=index,
            experiment=experiment,
            predecessor_task_id=predecessor_task_id,
            teacher_task_id=teacher_task_id,
            teacher_model_id=teacher_model_id,
            teacher_sha=teacher_sha,
            allow_failed_teacher_task=bool(args.allow_failed_teacher_task),
            project=args.project,
            adopted=True,
        )
        step["task_name"] = str(getattr(task, "name", "") or step.get("task_name"))
        status = _normalized_task_status(task)
        if status == "completed":
            # Build expected params from the adopted task's own predecessor pin.
            observed = _task_parameters(task)
            pred = str(
                observed.get("Args/predecessor_task_id") or predecessor_task_id
            )
            step["predecessor_task_id"] = pred
            expected_parameters = _experiment_parameters(
                template_parameters,
                experiment=experiment,
                predecessor_task_id=pred,
                teacher_task_id=teacher_task_id,
                teacher_model_id=teacher_model_id,
                teacher_checkpoint_sha256=teacher_sha,
                allow_failed_teacher_task=bool(args.allow_failed_teacher_task),
            )
            _seal_completed_step(
                task=task,
                step=step,
                progress=progress,
                controller_task=controller_task,
                experiment=experiment,
                predecessor_task_id=pred,
                expected_parameters=expected_parameters,
                template_identity=template_identity,
                teacher_task_id=teacher_task_id,
                teacher_model_id=teacher_model_id,
                teacher_sha=teacher_sha,
            )
        elif status in FAILED_STATUSES:
            step["state"] = "failed"
            step["failure_status"] = status
            _save_progress(controller_task, progress)
            raise RuntimeError(f"adopted experiment {experiment!r} failed: {status!r}")
        else:
            step["state"] = _state_from_status(status)
            worker = str(getattr(getattr(task, "data", None), "last_worker", "") or "")
            if "A100" in worker:
                step["worker_queue"] = "GPU4-A100"
            elif "V100" in worker and "gpu0,1,2,3" in worker:
                step["worker_queue"] = "GPU4-V100"
            elif "V100" in worker:
                step["worker_queue"] = "GPU4-V100"
            elif any(q in worker for q in worker_queues):
                matched = next(q for q in worker_queues if q.replace("GPU4-", "") in worker or q in worker)
                step["worker_queue"] = matched
            else:
                # Keep out-of-pool jobs (e.g. still finishing on 5090) outside slot accounting.
                step["worker_queue"] = "adopted-external"
            _save_progress(controller_task, progress)

    results: list[dict[str, object]] = []

    while True:
        for step in steps:
            if not isinstance(step, MutableMapping):
                raise RuntimeError("training progress step must be mutable")
            if step.get("state") == "failed":
                raise RuntimeError(
                    f"experiment {step.get('experiment')!r} previously failed"
                )

        # Refresh active tasks and seal completions.
        progressed = False
        for step in steps:
            if step.get("state") not in {"created", "queued", "running"}:
                continue
            experiment = str(step["experiment"])
            index = int(step["index"])
            task = task_class.get_task(task_id=str(step["task_id"]))
            _reload(task)
            status = _normalized_task_status(task)
            if status in FAILED_STATUSES:
                step["state"] = "failed"
                step["failure_status"] = status
                _save_progress(controller_task, progress)
                raise RuntimeError(f"experiment {experiment!r} failed: {status!r}")
            if status == "completed":
                predecessor_task_id = str(
                    step.get("predecessor_task_id")
                    or _latest_completed_predecessor(
                        steps,
                        gate_task_id=gate_task_id,
                        before_index=index,
                    )
                )
                expected_parameters = _experiment_parameters(
                    template_parameters,
                    experiment=experiment,
                    predecessor_task_id=predecessor_task_id,
                    teacher_task_id=teacher_task_id,
                    teacher_model_id=teacher_model_id,
                    teacher_checkpoint_sha256=teacher_sha,
                    allow_failed_teacher_task=bool(args.allow_failed_teacher_task),
                )
                result = _seal_completed_step(
                    task=task,
                    step=step,
                    progress=progress,
                    controller_task=controller_task,
                    experiment=experiment,
                    predecessor_task_id=predecessor_task_id,
                    expected_parameters=expected_parameters,
                    template_identity=template_identity,
                    teacher_task_id=teacher_task_id,
                    teacher_model_id=teacher_model_id,
                    teacher_sha=teacher_sha,
                )
                results.append({"index": index, "experiment": experiment, **result})
                progressed = True
                continue
            mapped = _state_from_status(status)
            if step.get("state") != mapped:
                step["state"] = mapped
                _save_progress(controller_task, progress)
                progressed = True

        active_steps = [
            step
            for step in steps
            if step.get("state") in {"created", "queued", "running"}
        ]
        pending_steps = [step for step in steps if step.get("state") == "pending"]
        if not active_steps and not pending_steps:
            break

        effective_max = 1 if _canary_blocks_extra_slots(
            steps, canary_first=canary_first, task_class=task_class
        ) else max_parallel
        free_queues = [
            queue
            for queue in worker_queues
            if queue not in _active_worker_queues(active_steps)
            and queue != "adopted-external"
        ]
        # Created-but-not-enqueued tasks do not occupy a worker queue yet.
        pool_active = sum(
            1
            for step in active_steps
            if step.get("worker_queue") not in {None, "", "adopted-external"}
        )

        # Resume clones that were created before an enqueue interruption.
        for step in list(active_steps):
            if step.get("state") != "created" or step.get("worker_queue"):
                continue
            if pool_active >= effective_max or not free_queues:
                break
            experiment = str(step["experiment"])
            task = task_class.get_task(task_id=str(step["task_id"]))
            if _normalized_task_status(task) != "created":
                continue
            queue = free_queues.pop(0)
            _enqueue(task_class, task, worker_queue=queue)
            step["state"] = "queued"
            step["worker_queue"] = queue
            _save_progress(controller_task, progress)
            pool_active += 1
            progressed = True
            if canary_first:
                break

        while (
            pending_steps
            and pool_active < effective_max
            and free_queues
        ):
            step = pending_steps[0]
            experiment = str(step["experiment"])
            index = int(step["index"])
            predecessor_task_id = _latest_completed_predecessor(
                steps,
                gate_task_id=gate_task_id,
                before_index=index,
            )
            task, expected_parameters = _ensure_experiment_task(
                task_class=task_class,
                template_task=template_task,
                template_identity=template_identity,
                template_parameters=template_parameters,
                controller_task=controller_task,
                controller_task_id=controller_task_id,
                progress=progress,
                step=step,
                index=index,
                experiment=experiment,
                predecessor_task_id=predecessor_task_id,
                teacher_task_id=teacher_task_id,
                teacher_model_id=teacher_model_id,
                teacher_sha=teacher_sha,
                allow_failed_teacher_task=bool(args.allow_failed_teacher_task),
                project=args.project,
                adopted=False,
            )
            status = _normalized_task_status(task)
            if status == "completed":
                result = _seal_completed_step(
                    task=task,
                    step=step,
                    progress=progress,
                    controller_task=controller_task,
                    experiment=experiment,
                    predecessor_task_id=predecessor_task_id,
                    expected_parameters=expected_parameters,
                    template_identity=template_identity,
                    teacher_task_id=teacher_task_id,
                    teacher_model_id=teacher_model_id,
                    teacher_sha=teacher_sha,
                )
                results.append({"index": index, "experiment": experiment, **result})
                pending_steps = [s for s in steps if s.get("state") == "pending"]
                progressed = True
                continue
            if status in FAILED_STATUSES:
                step["state"] = "failed"
                step["failure_status"] = status
                _save_progress(controller_task, progress)
                raise RuntimeError(f"experiment {experiment!r} failed: {status!r}")
            queue = free_queues.pop(0)
            if status == "created":
                _enqueue(task_class, task, worker_queue=queue)
                step["state"] = "queued"
            else:
                step["state"] = _state_from_status(status)
            step["worker_queue"] = queue
            step["predecessor_task_id"] = predecessor_task_id
            _save_progress(controller_task, progress)
            pool_active += 1
            pending_steps = [s for s in steps if s.get("state") == "pending"]
            progressed = True
            if canary_first:
                break

        if not pending_steps and not [
            step
            for step in steps
            if step.get("state") in {"created", "queued", "running"}
        ]:
            break
        if not progressed:
            sleeper(float(args.poll_seconds))

    # Rebuild ordered results for the summary artifact.
    ordered_results: list[dict[str, object]] = []
    for step in steps:
        if step.get("state") != "completed":
            raise RuntimeError("training suite finished with incomplete steps")
        result = step.get("result")
        if not isinstance(result, Mapping):
            raise RuntimeError("completed step is missing sealed result")
        ordered_results.append(
            {
                "index": int(step["index"]),
                "experiment": str(step["experiment"]),
                **dict(result),
            }
        )

    summary = _sealed(
        {
            "schema_version": 1,
            "summary_type": "resilient_v2x_post_main_sequential_training",
            "status": "completed",
            "controller_task_id": controller_task_id,
            "gate_task_id": gate_task_id,
            "template": template_identity,
            "teacher": {
                "task_id": teacher_task_id,
                "model_id": teacher_model_id,
                "checkpoint_sha256": teacher_sha,
                "allow_failed_task": bool(args.allow_failed_teacher_task),
            },
            "worker_queue": worker_queues[0],
            "worker_queues": list(worker_queues),
            "experiment_order": list(EXPERIMENT_ORDER),
            "task_count": len(ordered_results),
            "max_parallel_training_tasks": max_parallel,
            "results": ordered_results,
            "completed_at": _now(),
            "progress_artifact": PROGRESS_ARTIFACT,
        }
    )
    _upload_mapping(controller_task, SUMMARY_ARTIFACT, summary)
    return summary


def _current_controller_task(
    task_class: object,
    *,
    auto_connect_arg_parser: bool = False,
) -> object:
    getter = getattr(task_class, "current_task", None)
    task = getter() if callable(getter) else None
    if task is not None:
        return task
    initializer = getattr(task_class, "init", None)
    if not callable(initializer):
        raise RuntimeError("ClearML Task class cannot initialize a controller task")
    return initializer(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X post-main sequential training controller",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
        auto_connect_arg_parser=auto_connect_arg_parser,
    )


def main(argv: Sequence[str] | None = None) -> int:
    from clearml import Task

    controller_task = None
    if argv is None:
        controller_task = _current_controller_task(
            Task,
            auto_connect_arg_parser=True,
        )
    args = _parser().parse_args(argv)
    if controller_task is None:
        controller_task = _current_controller_task(Task)
    controller_task.output_uri = FILES_SERVER_URI
    run_training_suite(
        args,
        task_class=Task,
        controller_task=controller_task,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "EXPERIMENT_ORDER",
    "PROGRESS_ARTIFACT",
    "SUMMARY_ARTIFACT",
    "main",
    "run_training_suite",
)
