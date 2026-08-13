#!/usr/bin/env python3
"""Run the sealed post-main RTX5090 training suite one task at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import time
from collections import Counter
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
FORMAL_1337_MANIFEST_ARTIFACT = "formal_1337_training_manifest"
RUN_CONTRACT_ARTIFACT = "run_contract"
FINAL_CHECKPOINT_ARTIFACT = "final_checkpoint_contract"
TEACHER_CHECKPOINT_ARTIFACT = "teacher_checkpoint_contract"
EXPERIMENT_RUN_CONTRACT_SCHEMA_VERSION = 1
EXPERIMENT_RUN_CONTRACT_SEED_FIELDS = frozenset(
    {"seed", "training_seed", "training_overlay_protocol_seed"}
)
RECOVERY_CONTRACT_SCHEMA_VERSION = 4
SUPPORTED_SOURCE_RECOVERY_SCHEMA_VERSIONS = frozenset({2, 3, 4})
SOURCE_REVISION_TRANSITION_SCHEMA_VERSION = 1
SOURCE_REVISION_TRANSITION_ID = (
    "resilient-v2x-source-5c984ad49b52-to-ad511d88b731-custom-imports-list-v1"
)
SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID = "f8c36e508c7d453dadc766207a5b25b2"
SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID = "d377543f6a574449a5d4b28cb9275dbc"
SOURCE_REVISION_NATIVE_BUILD_TASK_ID = "9055c0d3c4dd450c8a75dddfb21a56bd"
SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256 = (
    "f07079970f131abca46f8aebbabebc20a1910b0c4dcd9721b3fdc31da29058ad"
)
SOURCE_REVISION_SUPPORT_RESIDUAL_FAILURE_MESSAGE_SHA256 = (
    "bdfe874e51ff542d8162ea655753b2f7c3c0d5d850d4df6026512d0c888c2751"
)
SOURCE_REVISION_LEGACY_TASK_ADOPTIONS = {
    "support_residual": {
        "task_id": "95e72da24d464ab08d117dedabd6652e",
        "parent_controller_task_id": "6525107e60ae4104a2800731d74ecd4e",
        "sealed_provenance_type": "completion_validation_retry",
    },
    "no_distillation": {
        "task_id": "efe6522d87a44c55b1de7f9c144e5393",
        "parent_controller_task_id": "d4b83d9b68704050aeb24a2e34540d8a",
        "sealed_provenance_type": "carried_recovery_target",
    },
}
SOURCE_REVISION_LEGACY_NESTED_TEACHER_EXPERIMENTS = frozenset(
    {"resilient_v2x", "support_residual"}
)
SOURCE_REVISION_INVENTORY_FILE_COUNT = 631
SOURCE_REVISION_UNCHANGED_FILE_COUNT = 627
SOURCE_REVISION_SOURCE_TREE_SHA256 = (
    "5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6"
)
SOURCE_REVISION_TARGET_TREE_SHA256 = (
    "ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"
)
SOURCE_REVISION_TARGET_INVENTORY_SHA256 = (
    "39b1a42af65ad5df935945bd0a4eeac6e7f6e1cfdd1cc608f3dd8a708e9c5ca0"
)
SOURCE_REVISION_TARGET_INVENTORY_BYTES = 101_195
SOURCE_REVISION_SOURCE_INVENTORY_SHA256 = (
    "bed72cd86438f2ba932edda14052e3cd3d9589ec201d09d18a315e18a2f7cff2"
)
SOURCE_REVISION_SOURCE_INVENTORY_BYTES = 101_195
SOURCE_REVISION_SOURCE_BYTES = 8_926_106
SOURCE_REVISION_TARGET_BYTES = 8_926_102
SOURCE_REVISION_SOURCE_PARAMETERS = {
    "Args/source_dataset_id": "4f7fac0078a4419a907fec6ff9e306c8",
    "Args/source_archive_name": "resilient-v2x-source-5c984ad49b52.tar.zst",
    "Args/source_archive_bytes": "1222481",
    "Args/source_archive_sha256": (
        "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
    ),
    "Args/training_dataset_id": "7c59fabb9da949e6b3c94c732f000975",
    "Args/native_bundle_bytes": "753382966",
    "Args/native_bundle_sha256": (
        "19b8e7f5edc8216d4b43cb17854dccafe4fc9fe46995a88803e6342eeaa22b21"
    ),
    "Args/build_manifest_sha256": (
        "21c6ab7a6e9a2823ba111289a42e7f882c5f4ba02c73175106ff251fe5864a43"
    ),
}
SOURCE_REVISION_TARGET_PARAMETERS = {
    **SOURCE_REVISION_SOURCE_PARAMETERS,
    "Args/source_dataset_id": "351feedbbe81481fa31f1e9ae11a3f4e",
    "Args/source_archive_name": "resilient-v2x-source-ad511d88b731.tar.zst",
    "Args/source_archive_bytes": "1222492",
    "Args/source_archive_sha256": (
        "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
    ),
}
SOURCE_REVISION_ALLOWED_FILE_CHANGES = (
    (
        "configs/resilient_v2x/baselines/disconet.py",
        "a91621aa341cde33573e0bc95f42729f53052eaefd19b6eddaa80e1885a9a304",
        "82818e7157fd9c21fca04783094fc8dd9b3d6d71b0bb5898edf69496dc81c0bf",
    ),
    (
        "configs/resilient_v2x/baselines/how2comm.py",
        "fb7a40e9699945d496b7125226eef1aa9ab69b065fe91a89ed9c87b3d5b1ebf4",
        "451532356eaa249cc4d41d04d5d3df3c10a670b365521f8b85077238b5c6181f",
    ),
    (
        "configs/resilient_v2x/baselines/late_fusion.py",
        "f7e0224f2c01497ee55008a5df14076eb312e00f2a74a8699082e8bbe391e216",
        "e510b227499533e62d94d3b2b7768ac7342ad5d82e0aa837989f3a446e95f324",
    ),
    (
        "configs/resilient_v2x/baselines/where2comm.py",
        "8e60be68edd79d33c2b648e383fd027f4d4b8b50ed59eee69e007d83b63ad687",
        "9e90f2542fb3998550b67344ade1e4571d8e4bf1b712c8727c4e0c2c28e7c61e",
    ),
)
CLEAN_TEACHER_MODEL_NAME = "ResilientV2X clean teacher"
CLEAN_TEACHER_SELECTION_PROTOCOL = "DAIR-CLEAN-PAIR1789-v1"
CLEAN_TEACHER_SELECTION_METRIC = "resilient_v2x/car_bev_ap_r40_0.70"
CLEAN_TEACHER_SELECTION_RULE = "greater"
CLEAN_TEACHER_CHECKPOINT_POLICY = "clean_validation_best_for_teacher_handoff"
COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT = "common_teacher_initialization_audit"
COMMON_TEACHER_INITIALIZATION_CONTRACT = "shared-only-clean-teacher-initialization-v1"
COMMON_TEACHER_INITIALIZATION_PREFIXES = (
    "lidar_encoder.",
    "camera_encoder.",
    "bbox_head.",
    "detection_projection.",
)
COMMON_TEACHER_SOURCE_KEYS = 617
COMMON_TEACHER_SOURCE_NUMEL = 35_811_485
COMMON_TEACHER_SOURCE_BYTES = 143_246_244
COMMON_TEACHER_SHARED_KEYS = 468
COMMON_TEACHER_SHARED_NUMEL = 31_506_934
COMMON_TEACHER_SHARED_BYTES = 126_028_040
COMMON_TEACHER_FUSION_KEYS = 149
EXPERIMENT_MAX_EPOCHS = 50
EXPECTED_GPU_COUNT = 4
DEFAULT_TRAINING_SEED = 20250218
TRAINING_OVERLAY_PROTOCOL_SEED = 20250218
CLEARML_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FAILED_STATUSES = frozenset({"failed", "stopped", "closed"})
WAITABLE_STATUSES = frozenset({"created", "queued", "in_progress"})
PROGRESS_STATES = frozenset(
    {"pending", "created", "queued", "running", "completed", "failed"}
)
_EXPERIMENT_NESTED_TEACHER_LEGACY = (
    'NESTED_TEACHER_EXPERIMENTS = frozenset({"resilient_v2x", "support_residual"})\n'
)
_EXPERIMENT_NESTED_TEACHER_PATCH = """NESTED_TEACHER_EXPERIMENTS = frozenset(
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
"""
_EXPERIMENT_SYSPATH_ANCHOR = """    env.update(
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
"""
_EXPERIMENT_SYSPATH_PATCH = """    env.update(
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
"""
_EXPERIMENT_SYSPATH_MARKER = "Overlay helpers import sealed package modules in-process"
_EXPERIMENT_DDP_UNUSED_ANCHOR = """        f"test_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        *RTX5090_HEADLESS_CFG_OPTIONS,
    ]
"""
_EXPERIMENT_DDP_UNUSED_PATCH = """        f"test_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        "find_unused_parameters=True",
        *RTX5090_HEADLESS_CFG_OPTIONS,
    ]
"""
_EXPERIMENT_DDP_UNUSED_MARKER = "find_unused_parameters=True"
_EXPERIMENT_CAPABILITY_ANCHOR = """    capabilities = contract.get("capabilities")
    if capabilities != [list(EXPECTED_CAPABILITY)] * int(gpu_count):
        raise RuntimeError(f"RTX5090 GPU capability mismatch: {capabilities!r}")
"""
_EXPERIMENT_CAPABILITY_PATCH = """    capabilities = contract.get("capabilities")
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
"""
_EXPERIMENT_CAPABILITY_MARKER = "GPU capabilities must be homogeneous"
_EXPERIMENT_RUNNER_LOAD_ANCHOR = """    runner = _load_source_training_runner(source_root)
    dataset_root, env = _prepare_experiment_environment(
"""
_EXPERIMENT_RUNNER_LOAD_PATCH = """    runner = _load_source_training_runner(source_root)
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
"""
_EXPERIMENT_RUNNER_LOAD_MARKER = "_validate_rtx5090_runtime_contract_multi_gpu"
_EXPERIMENT_RUNNER_LOAD_TARGET_PREFIX = """    if args.predecessor_task_id == task_id:
        raise RuntimeError("an experiment task cannot be its own predecessor")

"""
_EXPERIMENT_RUNNER_LOAD_TARGET_ANCHOR = (
    _EXPERIMENT_RUNNER_LOAD_TARGET_PREFIX + _EXPERIMENT_RUNNER_LOAD_ANCHOR
)
_EXPERIMENT_RUNNER_LOAD_TARGET_PATCH = (
    _EXPERIMENT_RUNNER_LOAD_TARGET_PREFIX + _EXPERIMENT_RUNNER_LOAD_PATCH
)
_EXPERIMENT_RUNNER_LOAD_PATCH_MARKER_COUNT = _EXPERIMENT_RUNNER_LOAD_PATCH.count(
    _EXPERIMENT_RUNNER_LOAD_MARKER
)
_EXPERIMENT_SMOKE_CAPABILITY_ANCHOR = """if any(torch.cuda.get_device_capability(i) != (12, 0) for i in range(gpu_count)):
    raise RuntimeError("all GPUs must have compute capability 12.0")
"""
_EXPERIMENT_SMOKE_CAPABILITY_PATCH = """_allowed_caps = {(12, 0), (8, 0), (7, 0)}
_caps = [torch.cuda.get_device_capability(i) for i in range(gpu_count)]
if any(cap not in _allowed_caps for cap in _caps) or len(set(_caps)) != 1:
    raise RuntimeError(
        "all GPUs must share one allowed compute capability "
        f"from {sorted(_allowed_caps)}; got {_caps}"
    )
"""
_EXPERIMENT_SMOKE_CAPABILITY_MARKER = "share one allowed compute capability"
_EXPERIMENT_MASTER_ADDR_ANCHOR = """            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        }
"""
_EXPERIMENT_MASTER_ADDR_ONLY_ANCHOR = """            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "MASTER_ADDR": "127.0.0.1",
        }
"""
_EXPERIMENT_MASTER_ADDR_PATCH = """            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "MASTER_ADDR": "127.0.0.1",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
            "NCCL_P2P_DISABLE": "1",
            "GLOO_SOCKET_IFNAME": "lo",
        }
"""
_EXPERIMENT_MASTER_ADDR_MARKER = 'NCCL_SOCKET_IFNAME": "lo"'
_EXPERIMENT_HOSTNAME_ANCHOR = """    _assert_base_image()
    _validate_gpu_runtime(_capture_gpu_runtime())

    from clearml import Dataset, OutputModel, Task
"""
_EXPERIMENT_HOSTNAME_PATCH = """    _assert_base_image()
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
"""
_EXPERIMENT_HOSTNAME_MARKER = (
    "ClearML A100/V100 workers often lack a resolvable hostname"
)
_EXPERIMENT_BUILD_TASK_MARKER_PREFIX = 'BUILD_TASK_ID = "'


class ExperimentSpec(NamedTuple):
    name: str
    kind: str
    config: str | None
    requires_teacher: bool


# Evidence priority is intentional and is part of the sealed controller contract.
CORE_EXPERIMENT_SPECS = (
    ExperimentSpec(
        "ptf_none", "ablation", "configs/resilient_v2x/ablations/ptf_none.py", True
    ),
    ExperimentSpec(
        "ptf_linear",
        "ablation",
        "configs/resilient_v2x/ablations/ptf_linear.py",
        True,
    ),
    ExperimentSpec(
        "router_static",
        "ablation",
        "configs/resilient_v2x/ablations/router_static.py",
        True,
    ),
    ExperimentSpec(
        "no_distillation",
        "ablation",
        "configs/resilient_v2x/ablations/no_distillation.py",
        True,
    ),
    ExperimentSpec("coformernet", "baseline", None, True),
    ExperimentSpec(
        "router_uniform",
        "ablation",
        "configs/resilient_v2x/ablations/router_uniform.py",
        True,
    ),
    ExperimentSpec(
        "no_reliability",
        "ablation",
        "configs/resilient_v2x/ablations/no_reliability.py",
        True,
    ),
    ExperimentSpec(
        "no_delay_metadata",
        "ablation",
        "configs/resilient_v2x/ablations/no_delay_metadata.py",
        True,
    ),
    ExperimentSpec(
        "concat_capacity_matched",
        "ablation",
        "configs/resilient_v2x/ablations/concat_capacity_matched.py",
        True,
    ),
    ExperimentSpec("ffnet", "baseline", None, True),
    ExperimentSpec("bevfusion", "baseline", None, True),
    ExperimentSpec("v2x_vit", "baseline", None, True),
    ExperimentSpec("cobevt", "baseline", None, True),
)
ADDITIONAL_EXPERIMENT_SPECS = (
    ExperimentSpec(
        "support_residual",
        "improvement",
        "configs/resilient_v2x/improvements/support_residual.py",
        True,
    ),
    ExperimentSpec(
        "linear_no_distillation",
        "improvement",
        "configs/resilient_v2x/improvements/linear_no_distillation.py",
        True,
    ),
    ExperimentSpec(
        "no_distillation_peak_lr_3e4",
        "improvement",
        "configs/resilient_v2x/improvements/no_distillation_peak_lr_3e4.py",
        True,
    ),
    ExperimentSpec("ego_only", "baseline", None, True),
    ExperimentSpec("fcooper", "baseline", None, True),
    ExperimentSpec("attfuse", "baseline", None, True),
    ExperimentSpec("v2vnet", "baseline", None, True),
    ExperimentSpec("when2com", "baseline", None, True),
    ExperimentSpec("where2comm", "baseline", None, True),
    ExperimentSpec("late_fusion", "baseline", None, True),
    ExperimentSpec("disconet", "baseline", None, True),
    ExperimentSpec("how2comm", "baseline", None, True),
    ExperimentSpec(
        "resilient_v2x",
        "primary_method",
        "configs/resilient_v2x/dair_resilient_v2x.py",
        True,
    ),
)
EXPERIMENT_SPECS = (
    ADDITIONAL_EXPERIMENT_SPECS[:1]
    + CORE_EXPERIMENT_SPECS
    + ADDITIONAL_EXPERIMENT_SPECS[1:]
)
CORE_EXPERIMENT_ORDER = tuple(spec.name for spec in CORE_EXPERIMENT_SPECS)
EXPERIMENT_ORDER = tuple(spec.name for spec in EXPERIMENT_SPECS)
EXPERIMENT_BY_NAME = {spec.name: spec for spec in EXPERIMENT_SPECS}
SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS = (
    "where2comm",
    "late_fusion",
    "disconet",
    "how2comm",
    "resilient_v2x",
)
SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS = EXPERIMENT_ORDER[
    : -len(SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS)
]
if EXPERIMENT_ORDER != (
    SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS
    + SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
):
    raise RuntimeError("source-revision subject split is not an exact suffix")
TEACHER_DEPENDENT_EXPERIMENTS = frozenset(
    spec.name for spec in EXPERIMENT_SPECS if spec.requires_teacher
)
PRIMARY_METHOD_EXPERIMENTS = frozenset(
    spec.name for spec in EXPERIMENT_SPECS if spec.kind == "primary_method"
)
NESTED_TEACHER_EXPERIMENTS = frozenset(
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
ZERO_FUSION_ALLOWED_EXPERIMENTS = frozenset({"ego_only", "fcooper"})
FORMAL_1337_SUBJECT_ORDER = EXPERIMENT_ORDER
FORMAL_1337_PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
FORMAL_1337_SAMPLE_COUNT = 1337
FORMAL_1337_DELAYS_MS = (0, 100, 200, 300)
FORMAL_1337_CONDITIONS = ("Full", "L-Fail", "C-Fail")
FORMAL_1337_CHECKPOINT_POLICY = "epoch_50_final_only"
FORMAL_1337_RELEASE_SEMANTICS = "formal_manifest_after_full_training_suite_completion"

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
        "Args/training_seed",
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


def build_formal_1337_training_manifest(
    ordered_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Seal the exact epoch-50 training inputs for the formal 1337 matrix."""

    if len(ordered_results) != len(FORMAL_1337_SUBJECT_ORDER):
        raise RuntimeError("formal 1337 training result count mismatch")
    entries: list[dict[str, object]] = []
    training_task_ids: set[str] = set()
    model_ids: set[str] = set()
    training_seeds: set[int] = set()
    for index, (result, subject) in enumerate(
        zip(ordered_results, FORMAL_1337_SUBJECT_ORDER, strict=True),
        start=1,
    ):
        if not isinstance(result, Mapping):
            raise RuntimeError("formal 1337 training result is not an object")
        if result.get("index") != index or result.get("experiment") != subject:
            raise RuntimeError("formal 1337 training result order mismatch")
        task_id = _clearml_id(result.get("task_id"), f"{subject} training task")
        predecessor_task_id = _clearml_id(
            result.get("predecessor_task_id"),
            f"{subject} training predecessor",
        )
        model_id = _clearml_id(result.get("model_id"), f"{subject} final model")
        if task_id in training_task_ids:
            raise RuntimeError("formal 1337 training task IDs are not unique")
        if model_id in model_ids:
            raise RuntimeError("formal 1337 final model IDs are not unique")
        training_task_ids.add(task_id)
        model_ids.add(model_id)
        model_name = f"ResilientV2X {subject} final checkpoint"
        if result.get("model_name") != model_name:
            raise RuntimeError(f"formal 1337 {subject} model name mismatch")
        model_url = _require_files_server_url(
            result.get("model_url"),
            context=f"formal 1337 {subject} final model URL",
        )
        if Path(unquote(urlsplit(model_url).path)).name != (f"{subject}_epoch_50.pth"):
            raise RuntimeError(f"formal 1337 {subject} model filename mismatch")
        checkpoint_sha256 = _sha256(
            result.get("checkpoint_sha256"),
            f"formal 1337 {subject} final checkpoint",
        )
        try:
            checkpoint_size_bytes = int(str(result.get("checkpoint_size_bytes")))
        except ValueError as error:
            raise RuntimeError(
                f"formal 1337 {subject} checkpoint size is invalid"
            ) from error
        if checkpoint_size_bytes <= 0:
            raise RuntimeError(f"formal 1337 {subject} checkpoint is empty")
        if result.get("run_contract_artifact") != RUN_CONTRACT_ARTIFACT:
            raise RuntimeError(f"formal 1337 {subject} run artifact mismatch")
        if result.get("final_checkpoint_artifact") != FINAL_CHECKPOINT_ARTIFACT:
            raise RuntimeError(
                f"formal 1337 {subject} final checkpoint artifact mismatch"
            )
        if result.get("common_teacher_initialization_audit_artifact") != (
            COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT
        ):
            raise RuntimeError(
                f"formal 1337 {subject} initialization audit artifact mismatch"
            )
        initialization_audit_sha256 = _sha256(
            result.get("common_teacher_initialization_audit_sha256"),
            f"formal 1337 {subject} initialization audit",
        )
        training_seed = result.get("training_seed")
        if type(training_seed) is not int or training_seed < 0:
            raise RuntimeError(f"formal 1337 {subject} training seed is invalid")
        training_seeds.add(training_seed)
        if (
            result.get("training_overlay_protocol_seed")
            != TRAINING_OVERLAY_PROTOCOL_SEED
        ):
            raise RuntimeError(
                f"formal 1337 {subject} training overlay protocol seed mismatch"
            )
        entries.append(
            {
                "index": index,
                "subject": subject,
                "kind": EXPERIMENT_BY_NAME[subject].kind,
                "training_task_id": task_id,
                "training_predecessor_task_id": predecessor_task_id,
                "model_id": model_id,
                "model_name": model_name,
                "model_url": model_url,
                "checkpoint_filename": "epoch_50.pth",
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_size_bytes": checkpoint_size_bytes,
                "training_seed": training_seed,
                "training_overlay_protocol_seed": (TRAINING_OVERLAY_PROTOCOL_SEED),
                "common_teacher_initialization_audit_artifact": (
                    COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT
                ),
                "common_teacher_initialization_audit_sha256": (
                    initialization_audit_sha256
                ),
            }
        )
    if len(training_seeds) != 1:
        raise RuntimeError("formal 1337 subjects must share one training seed")
    training_seed = next(iter(training_seeds))
    return _sealed(
        {
            "schema_version": 1,
            "manifest_type": "resilient_v2x_formal_1337_training_inputs",
            "protocol_id": FORMAL_1337_PROTOCOL_ID,
            "sample_count": FORMAL_1337_SAMPLE_COUNT,
            "delays_ms": list(FORMAL_1337_DELAYS_MS),
            "conditions": list(FORMAL_1337_CONDITIONS),
            "run_count": len(FORMAL_1337_DELAYS_MS) * len(FORMAL_1337_CONDITIONS),
            "checkpoint_policy": FORMAL_1337_CHECKPOINT_POLICY,
            "training_seed": training_seed,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "evaluation_release_semantics": FORMAL_1337_RELEASE_SEMANTICS,
            "subject_order": list(FORMAL_1337_SUBJECT_ORDER),
            "subject_count": len(FORMAL_1337_SUBJECT_ORDER),
            "entries": entries,
        }
    )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
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
    parser.add_argument(
        "--resolve-teacher-reference",
        action="store_true",
        help=(
            "wait for the completed teacher task and resolve its exact model/SHA "
            "from teacher_checkpoint_contract"
        ),
    )
    parser.add_argument("--teacher-model-id")
    parser.add_argument("--teacher-checkpoint-sha256")
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
        "--recover-failed-controller-task-id",
        default="",
        help=(
            "fork a new immutable progress chain from one failed controller; "
            "the source controller and its artifacts are read only"
        ),
    )
    parser.add_argument(
        "--recovery-source-template-task-id",
        default="",
        help=(
            "explicit source template for a sealed source-revision transition; "
            "requires --recover-failed-controller-task-id and "
            "--recovery-source-transition"
        ),
    )
    parser.add_argument(
        "--recovery-source-transition",
        default="",
        help=(
            "sealed source-revision transition ID; arbitrary source drift is rejected"
        ),
    )
    parser.add_argument(
        "--rerun-failed-experiment",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "failed source-chain experiment to recreate under the new controller "
            "(repeatable; requires --recover-failed-controller-task-id)"
        ),
    )
    parser.add_argument(
        "--recovery-adopt-target-experiment",
        action="append",
        default=[],
        metavar="NAME=TASK_ID",
        help=(
            "explicitly adopt a task created under an earlier failed recovery "
            "target; only a source-snapshot pending slot is eligible (repeatable)"
        ),
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
    parser.add_argument(
        "--training-seed",
        type=_nonnegative_integer,
        default=DEFAULT_TRAINING_SEED,
        help=(
            "model, sampler, and dataset-augmentation seed; the sealed "
            "training-overlay protocol seed remains fixed independently"
        ),
    )
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


def _parse_rerun_experiments(values: Sequence[object]) -> tuple[str, ...]:
    rerun: list[str] = []
    for raw in values:
        if type(raw) is not str or not raw.strip():
            raise ValueError(f"invalid --rerun-failed-experiment value: {raw!r}")
        experiment = raw.strip()
        if experiment not in EXPERIMENT_BY_NAME:
            raise ValueError(f"unknown rerun experiment: {experiment!r}")
        if experiment in rerun:
            raise ValueError(f"duplicate rerun experiment: {experiment!r}")
        rerun.append(experiment)
    return tuple(rerun)


def _parse_recovery_target_adoptions(
    values: Sequence[object],
) -> dict[str, str]:
    adopted: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or "=" not in raw:
            raise ValueError(
                f"invalid --recovery-adopt-target-experiment value: {raw!r}"
            )
        name, task_id = raw.split("=", 1)
        name = name.strip()
        task_id = task_id.strip()
        if name not in EXPERIMENT_BY_NAME:
            raise ValueError(f"unknown recovery-target experiment: {name!r}")
        if name in adopted:
            raise ValueError(f"duplicate recovery-target experiment: {name!r}")
        adopted[name] = _clearml_id(
            task_id,
            f"recovery-target {name} task",
        )
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


def _utc_timestamp(value: object, context: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif type(value) is str and value.strip():
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as error:
            raise RuntimeError(
                f"{context} is not a valid ISO-8601 timestamp"
            ) from error
    else:
        raise RuntimeError(f"{context} is missing")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"{context} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _task_last_update(task: object, *, context: str) -> datetime:
    data = getattr(task, "data", None)
    value = getattr(data, "last_update", None)
    if value is None:
        value = getattr(task, "last_update", None)
    return _utc_timestamp(value, f"{context} last_update")


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
        or parsed.username is not None
        or parsed.password is not None
        or (parsed.path and not parsed.path.startswith("/"))
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


def _apply_experiment_runner_load_patch(diff: str) -> str:
    """Patch only the experiment runner load, with exact idempotent states."""

    marker_count = diff.count(_EXPERIMENT_RUNNER_LOAD_MARKER)
    target_anchor_count = diff.count(_EXPERIMENT_RUNNER_LOAD_TARGET_ANCHOR)
    target_patch_count = diff.count(_EXPERIMENT_RUNNER_LOAD_TARGET_PATCH)
    broad_anchor_count = diff.count(_EXPERIMENT_RUNNER_LOAD_ANCHOR)

    if marker_count == target_anchor_count == target_patch_count == 0:
        if broad_anchor_count:
            raise RuntimeError(
                "experiment runner-load target anchor is missing or ambiguous"
            )
        return diff

    if marker_count == 0 and target_patch_count == 0:
        if target_anchor_count != 1 or broad_anchor_count not in {1, 2}:
            raise RuntimeError("experiment runner-load anchor is not unique")
        patched = diff.replace(
            _EXPERIMENT_RUNNER_LOAD_TARGET_ANCHOR,
            _EXPERIMENT_RUNNER_LOAD_TARGET_PATCH,
            1,
        )
        if (
            patched.count(_EXPERIMENT_RUNNER_LOAD_MARKER)
            != _EXPERIMENT_RUNNER_LOAD_PATCH_MARKER_COUNT
            or patched.count(_EXPERIMENT_RUNNER_LOAD_TARGET_ANCHOR) != 0
            or patched.count(_EXPERIMENT_RUNNER_LOAD_TARGET_PATCH) != 1
            or patched.count(_EXPERIMENT_RUNNER_LOAD_ANCHOR) != broad_anchor_count - 1
        ):
            raise RuntimeError("experiment runner-load patch postcondition failed")
        return patched

    if (
        marker_count == _EXPERIMENT_RUNNER_LOAD_PATCH_MARKER_COUNT
        and target_anchor_count == 0
        and target_patch_count == 1
        and broad_anchor_count in {0, 1}
    ):
        return diff

    raise RuntimeError("experiment runner-load patch state is mixed or not unique")


def _apply_experiment_nested_teacher_patch(diff: str) -> str:
    """Upgrade exactly one legacy bootstrap expectation, idempotently."""

    legacy_count = diff.count(_EXPERIMENT_NESTED_TEACHER_LEGACY)
    patched_count = diff.count(_EXPERIMENT_NESTED_TEACHER_PATCH)
    if legacy_count == 0 and patched_count == 0:
        # Unit-test fixtures and unrelated scripts stay unchanged.
        return diff
    if legacy_count == 1 and patched_count == 0:
        patched = diff.replace(
            _EXPERIMENT_NESTED_TEACHER_LEGACY,
            _EXPERIMENT_NESTED_TEACHER_PATCH,
            1,
        )
        if (
            patched.count(_EXPERIMENT_NESTED_TEACHER_LEGACY) != 0
            or patched.count(_EXPERIMENT_NESTED_TEACHER_PATCH) != 1
        ):
            raise RuntimeError("nested-teacher patch postcondition failed")
        return patched
    if legacy_count == 0 and patched_count == 1:
        return diff
    raise RuntimeError("nested-teacher patch state is mixed or not unique")


def _apply_experiment_syspath_patch(
    diff: str,
    *,
    include_nested_teacher_fix: bool = True,
) -> str:
    """Ensure experiment bootstrap can import sealed package modules in-process."""

    if type(diff) is not str:
        raise RuntimeError("standalone script diff must be a string")
    if type(include_nested_teacher_fix) is not bool:
        raise TypeError("include_nested_teacher_fix must be a boolean")
    patched = (
        _apply_experiment_nested_teacher_patch(diff)
        if include_nested_teacher_fix
        else diff
    )
    if _EXPERIMENT_SYSPATH_MARKER in patched:
        pass
    else:
        anchor_count = patched.count(_EXPERIMENT_SYSPATH_ANCHOR)
        if anchor_count == 0:
            # Unit-test fixtures and unrelated scripts stay unchanged.
            pass
        elif anchor_count != 1:
            raise RuntimeError("experiment sys.path compatibility anchor is not unique")
        else:
            patched = patched.replace(
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
    patched = _apply_experiment_runner_load_patch(patched)
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
                    raise RuntimeError(
                        "sealed BUILD_TASK_ID marker missing from script"
                    )
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


def _template_identity(
    task: object,
    *,
    expected_task_id: str,
    include_nested_teacher_fix: bool = True,
) -> dict[str, object]:
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
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    )
    for marker in markers:
        if marker not in diff:
            raise RuntimeError(f"template standalone script marker missing: {marker}")
    if SEALED_TEMPLATE_BUILD_TASK_ID not in diff and BUILD_TASK_ID not in diff:
        raise RuntimeError(
            "template standalone script contains neither sealed nor active "
            "build-task marker"
        )
    docker = _task_docker(task)
    _validate_docker_command(docker)
    parameters = _task_parameters(task)
    source_parameters = _validate_source_parameters(parameters)
    patched_script = dict(script)
    patched_script["diff"] = _apply_experiment_syspath_patch(
        diff,
        include_nested_teacher_fix=include_nested_teacher_fix,
    )
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


def _experiment_config_path(experiment: str) -> str:
    spec = EXPERIMENT_BY_NAME[experiment]
    if spec.config is not None:
        return spec.config
    return f"configs/resilient_v2x/baselines/{experiment}.py"


def _source_parameter_contract(
    identity: Mapping[str, object],
    *,
    expected: Mapping[str, object],
    context: str,
) -> dict[str, str]:
    observed = identity.get("source_parameters")
    if not isinstance(observed, Mapping):
        raise RuntimeError(f"{context} has no source parameter identity")
    if set(observed) != set(SOURCE_PARAMETER_KEYS):
        raise RuntimeError(f"{context} source parameter keys drifted")
    for key in SOURCE_PARAMETER_KEYS:
        if not _parameter_matches(observed.get(key), expected[key]):
            raise RuntimeError(f"{context} source parameter drifted: {key}")
    return {key: str(expected[key]) for key in SOURCE_PARAMETER_KEYS}


def _template_identity_parameter_equivalence_receipt(
    observed_identity: object,
    expected_identity: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    """Prove exact identity equality modulo ClearML int/string parameter casts."""

    if not isinstance(observed_identity, Mapping):
        raise RuntimeError(f"{context} is not an object")
    observed = dict(observed_identity)
    expected = dict(expected_identity)
    if set(observed) != set(expected):
        raise RuntimeError(f"{context} identity keys drifted")
    observed_source = observed.get("source_parameters")
    expected_source = expected.get("source_parameters")
    if not isinstance(observed_source, Mapping) or not isinstance(
        expected_source, Mapping
    ):
        raise RuntimeError(f"{context} source parameters are invalid")
    if set(observed_source) != set(SOURCE_PARAMETER_KEYS) or set(
        expected_source
    ) != set(SOURCE_PARAMETER_KEYS):
        raise RuntimeError(f"{context} source parameter keys drifted")

    observed_common = dict(observed)
    expected_common = dict(expected)
    observed_common.pop("source_parameters")
    expected_common.pop("source_parameters")
    if observed_common != expected_common:
        raise RuntimeError(f"{context} non-parameter identity drifted")

    normalizations: list[dict[str, str]] = []
    for key in SOURCE_PARAMETER_KEYS:
        observed_value = observed_source[key]
        expected_value = expected_source[key]
        if not _parameter_matches(observed_value, expected_value):
            raise RuntimeError(f"{context} source parameter drifted: {key}")
        if type(observed_value) is not type(expected_value):
            normalizations.append(
                {
                    "key": key,
                    "observed_type": type(observed_value).__name__,
                    "expected_type": type(expected_value).__name__,
                    "canonical_value": str(expected_value),
                }
            )

    return _sealed(
        {
            "schema_version": 1,
            "contract_type": "template_identity_parameter_type_equivalence",
            "result": "pass",
            "context": context,
            "observed_identity": observed,
            "expected_identity": expected,
            "observed_identity_sha256": hashlib.sha256(
                _canonical_json(observed).encode("utf-8")
            ).hexdigest(),
            "expected_identity_sha256": hashlib.sha256(
                _canonical_json(expected).encode("utf-8")
            ).hexdigest(),
            "exact_identity_keys": sorted(expected),
            "exact_non_source_identity": True,
            "exact_source_parameter_keys": True,
            "parameter_values_equivalent": True,
            "normalized_type_only_keys": [item["key"] for item in normalizations],
            "type_normalizations": normalizations,
        }
    )


def _build_source_revision_transition_contract(
    *,
    transition_id: str,
    source_controller_task_id: str,
    source_template_identity: Mapping[str, object],
    source_legacy_template_identity: Mapping[str, object],
    target_template_identity: Mapping[str, object],
    source_template_parameters: Mapping[str, object],
    target_template_parameters: Mapping[str, object],
) -> dict[str, object]:
    """Seal the only permitted mixed-source recovery transition."""

    if transition_id != SOURCE_REVISION_TRANSITION_ID:
        raise ValueError("unsupported recovery source transition")
    if source_controller_task_id != SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID:
        raise ValueError("source-revision transition controller task ID mismatch")
    if source_template_identity.get("task_id") != (
        SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID
    ):
        raise RuntimeError("source-revision source template task ID mismatch")
    if target_template_identity.get("task_id") != (
        SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID
    ):
        raise RuntimeError("source-revision target template task ID mismatch")

    source_parameters = _source_parameter_contract(
        source_template_identity,
        expected=SOURCE_REVISION_SOURCE_PARAMETERS,
        context="source-revision source template",
    )
    target_parameters = _source_parameter_contract(
        target_template_identity,
        expected=SOURCE_REVISION_TARGET_PARAMETERS,
        context="source-revision target template",
    )

    source_identity_common = dict(source_template_identity)
    target_identity_common = dict(target_template_identity)
    source_identity_common.pop("task_id", None)
    target_identity_common.pop("task_id", None)
    source_identity_common.pop("source_parameters", None)
    target_identity_common.pop("source_parameters", None)
    if source_identity_common != target_identity_common:
        raise RuntimeError("source-revision template execution identity drifted")
    if source_identity_common.get("native_build_task_id") != (
        SOURCE_REVISION_NATIVE_BUILD_TASK_ID
    ):
        raise RuntimeError("source-revision native build task ID drifted")

    source_legacy_identity = dict(source_legacy_template_identity)
    if set(source_legacy_identity) != set(source_template_identity):
        raise RuntimeError("source-revision legacy template identity keys drifted")
    source_legacy_script_sha256 = _sha256(
        source_legacy_identity.get("script_sha256"),
        "source-revision legacy template script",
    )
    if source_legacy_script_sha256 != SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256:
        raise RuntimeError("source-revision legacy template script identity drifted")
    source_canonical_non_script = dict(source_template_identity)
    source_legacy_non_script = dict(source_legacy_identity)
    source_canonical_non_script.pop("script_sha256")
    source_legacy_non_script.pop("script_sha256")
    if source_legacy_non_script != source_canonical_non_script:
        raise RuntimeError(
            "source-revision legacy template differs outside script identity"
        )

    source_non_source_parameters = {
        str(key): value
        for key, value in source_template_parameters.items()
        if str(key) not in SOURCE_PARAMETER_KEYS
    }
    target_non_source_parameters = {
        str(key): value
        for key, value in target_template_parameters.items()
        if str(key) not in SOURCE_PARAMETER_KEYS
    }
    if source_non_source_parameters != target_non_source_parameters:
        raise RuntimeError("source-revision non-source template parameters drifted")

    changed_parameter_keys = [
        key
        for key in SOURCE_PARAMETER_KEYS
        if source_parameters[key] != target_parameters[key]
    ]
    expected_changed_parameter_keys = list(SOURCE_PARAMETER_KEYS[:4])
    if changed_parameter_keys != expected_changed_parameter_keys:
        raise RuntimeError("source-revision parameter delta is not exact")
    unchanged_parameter_keys = list(SOURCE_PARAMETER_KEYS[4:])
    changed_paths = {item[0] for item in SOURCE_REVISION_ALLOWED_FILE_CHANGES}
    source_subject_receipts = []
    for experiment in SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS:
        config_path = _experiment_config_path(experiment)
        if config_path in changed_paths:
            raise RuntimeError(
                f"source-revision old subject config changed: {experiment!r}"
            )
        source_subject_receipts.append(
            {
                "experiment": experiment,
                "config_path": config_path,
                "config_inventory_status": "byte_identical_across_revisions",
                "template_role": "source",
            }
        )
    for experiment in SOURCE_REVISION_LEGACY_TASK_ADOPTIONS:
        if experiment not in SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS:
            raise RuntimeError(
                "source-revision legacy exception is outside old subjects"
            )
        if (experiment in SOURCE_REVISION_LEGACY_NESTED_TEACHER_EXPERIMENTS) != (
            experiment in NESTED_TEACHER_EXPERIMENTS
        ):
            raise RuntimeError(
                f"source-revision legacy exception changes semantics for {experiment!r}"
            )

    shared_parameter_sha256 = hashlib.sha256(
        _canonical_json(source_non_source_parameters).encode("utf-8")
    ).hexdigest()
    return _sealed(
        {
            "schema_version": SOURCE_REVISION_TRANSITION_SCHEMA_VERSION,
            "contract_type": "resilient_v2x_source_revision_transition",
            "transition_id": SOURCE_REVISION_TRANSITION_ID,
            "source_controller_task_id": SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
            "source_template_identity": dict(source_template_identity),
            "target_template_identity": dict(target_template_identity),
            "source_revision": {
                "parameters": source_parameters,
                "tree_sha256": SOURCE_REVISION_SOURCE_TREE_SHA256,
                "inventory_sha256": SOURCE_REVISION_SOURCE_INVENTORY_SHA256,
                "inventory_bytes": SOURCE_REVISION_SOURCE_INVENTORY_BYTES,
                "inventory_file_count": SOURCE_REVISION_INVENTORY_FILE_COUNT,
                "source_bytes": SOURCE_REVISION_SOURCE_BYTES,
            },
            "target_revision": {
                "parameters": target_parameters,
                "tree_sha256": SOURCE_REVISION_TARGET_TREE_SHA256,
                "inventory_sha256": SOURCE_REVISION_TARGET_INVENTORY_SHA256,
                "inventory_bytes": SOURCE_REVISION_TARGET_INVENTORY_BYTES,
                "inventory_file_count": SOURCE_REVISION_INVENTORY_FILE_COUNT,
                "source_bytes": SOURCE_REVISION_TARGET_BYTES,
            },
            "parameter_equivalence": {
                "changed_keys": changed_parameter_keys,
                "unchanged_keys": unchanged_parameter_keys,
                "changes": {
                    key: {
                        "source": source_parameters[key],
                        "target": target_parameters[key],
                    }
                    for key in changed_parameter_keys
                },
                "shared_non_source_parameter_count": len(source_non_source_parameters),
                "shared_non_source_parameter_keys": sorted(
                    source_non_source_parameters
                ),
                "shared_non_source_parameters_sha256": shared_parameter_sha256,
                "template_execution_identity_equal": True,
            },
            "inventory_equivalence": {
                "source_file_count": SOURCE_REVISION_INVENTORY_FILE_COUNT,
                "target_file_count": SOURCE_REVISION_INVENTORY_FILE_COUNT,
                "unchanged_file_count": SOURCE_REVISION_UNCHANGED_FILE_COUNT,
                "changed_file_count": len(SOURCE_REVISION_ALLOWED_FILE_CHANGES),
                "added_paths": [],
                "removed_paths": [],
                "all_other_paths_byte_identical": True,
                "changed_files": [
                    {
                        "path": path,
                        "source_sha256": source_sha256,
                        "target_sha256": target_sha256,
                        "semantic_change": "python_tuple_to_list_only",
                    }
                    for path, source_sha256, target_sha256 in (
                        SOURCE_REVISION_ALLOWED_FILE_CHANGES
                    )
                ],
            },
            "subject_policy": {
                "source_template_adoptable_experiments": list(
                    SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS
                ),
                "target_template_required_experiments": list(
                    SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
                ),
                "source_template_adoptable_count": len(
                    SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS
                ),
                "target_template_required_count": len(
                    SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
                ),
                "source_subject_config_receipts": source_subject_receipts,
                "legacy_script_adoption_exceptions": [
                    {
                        "experiment": experiment,
                        "task_id": binding["task_id"],
                        "parent_controller_task_id": binding[
                            "parent_controller_task_id"
                        ],
                        "sealed_provenance_type": binding["sealed_provenance_type"],
                        "legacy_script_sha256": (
                            SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256
                        ),
                        "canonical_script_sha256": source_template_identity.get(
                            "script_sha256"
                        ),
                        "legacy_nested_teacher_membership": experiment
                        in SOURCE_REVISION_LEGACY_NESTED_TEACHER_EXPERIMENTS,
                        "canonical_nested_teacher_membership": experiment
                        in NESTED_TEACHER_EXPERIMENTS,
                        "subject_semantics_equal": (
                            (
                                experiment
                                in SOURCE_REVISION_LEGACY_NESTED_TEACHER_EXPERIMENTS
                            )
                            == (experiment in NESTED_TEACHER_EXPERIMENTS)
                        ),
                    }
                    for experiment, binding in (
                        SOURCE_REVISION_LEGACY_TASK_ADOPTIONS.items()
                    )
                ],
                "policy": (
                    "old tasks require exact source template and unchanged selected "
                    "config; five target subjects require the new template/source"
                ),
            },
        }
    )


def _experiment_parameters(
    template_parameters: Mapping[str, object],
    *,
    experiment: str,
    predecessor_task_id: str,
    teacher_task_id: str,
    teacher_model_id: str,
    teacher_checkpoint_sha256: str,
    allow_failed_teacher_task: bool,
    training_seed: int = DEFAULT_TRAINING_SEED,
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
            "Args/training_seed": training_seed,
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


def _validate_recovery_task_binding(
    *,
    task: object,
    task_id: str,
    experiment: str,
    predecessor_task_id: str,
    template_role: str,
    template_identity: Mapping[str, object],
    legacy_template_identity: Mapping[str, object] | None,
    template_parameters: Mapping[str, object],
    teacher_reference: Mapping[str, object],
    training_seed: int,
    sealed_recovery_provenance: Mapping[str, object] | None,
    source_controller_task_id: str,
    source_progress_seal_sha256: str,
    source_step: Mapping[str, object],
) -> dict[str, object]:
    if template_role not in {"source", "target"}:
        raise RuntimeError("recovery task has an invalid template role")
    if template_role == "source" and experiment not in (
        SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS
    ):
        raise RuntimeError(
            f"source-revision target-only experiment cannot adopt old source: "
            f"{experiment!r}"
        )
    if template_role == "target" and experiment not in (
        SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
    ):
        raise RuntimeError(
            f"source-revision old subject cannot adopt a new-source target: "
            f"{experiment!r}"
        )
    task_script_sha256 = _script_sha256(_task_script(task))
    canonical_script_sha256 = _sha256(
        template_identity.get("script_sha256"),
        "recovery canonical template script",
    )
    script_identity_policy = "exact_canonical_template"
    legacy_compatibility_receipt: dict[str, object] | None = None
    if task_script_sha256 == canonical_script_sha256:
        _validate_clone_identity(task, identity=template_identity)
    else:
        exception = SOURCE_REVISION_LEGACY_TASK_ADOPTIONS.get(experiment)
        if template_role != "source" or exception is None:
            _validate_clone_identity(task, identity=template_identity)
            raise AssertionError("unreachable clone-identity validation")
        if task_id != exception["task_id"]:
            raise RuntimeError(
                f"source-revision legacy task ID mismatch for {experiment!r}"
            )
        parent_controller_task_id = _clearml_id(
            _task_parent(task),
            f"source-revision legacy {experiment} parent controller",
        )
        if parent_controller_task_id != exception["parent_controller_task_id"]:
            raise RuntimeError(
                f"source-revision legacy parent mismatch for {experiment!r}"
            )
        if source_controller_task_id != SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID:
            raise RuntimeError("source-revision legacy source controller mismatch")
        if legacy_template_identity is None:
            raise RuntimeError("source-revision legacy template identity is missing")
        legacy_script_sha256 = _sha256(
            legacy_template_identity.get("script_sha256"),
            "source-revision legacy template script",
        )
        if (
            legacy_script_sha256 != SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256
            or task_script_sha256 != SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256
        ):
            raise RuntimeError(
                f"source-revision legacy script mismatch for {experiment!r}"
            )
        canonical_non_script = dict(template_identity)
        legacy_non_script = dict(legacy_template_identity)
        canonical_non_script.pop("script_sha256", None)
        legacy_non_script.pop("script_sha256", None)
        if legacy_non_script != canonical_non_script:
            raise RuntimeError(
                "source-revision legacy identity differs outside script hash"
            )
        if not isinstance(sealed_recovery_provenance, Mapping):
            raise RuntimeError(
                f"source-revision legacy task {experiment!r} lacks sealed carried "
                "recovery provenance"
            )
        provenance_type = exception["sealed_provenance_type"]
        if provenance_type == "carried_recovery_target":
            provenance_matches = (
                sealed_recovery_provenance.get("task_id") == task_id
                and sealed_recovery_provenance.get("parent_controller_task_id")
                == parent_controller_task_id
                and sealed_recovery_provenance.get("task_script_sha256")
                == task_script_sha256
                and sealed_recovery_provenance.get("provenance")
                == "explicit_failed_recovery_target"
                and sealed_recovery_provenance.get("source_snapshot_state") == "pending"
                and sealed_recovery_provenance.get("source_snapshot_task_id") is None
            )
            sealed_provenance = "explicit_failed_recovery_target"
        elif provenance_type == "completion_validation_retry":
            provenance_matches = (
                sealed_recovery_provenance.get("task_id") == task_id
                and sealed_recovery_provenance.get("source_snapshot_state") == "failed"
                and sealed_recovery_provenance.get("source_failure_status")
                == "completion_validation_failed"
                and sealed_recovery_provenance.get("live_task_status") == "completed"
                and sealed_recovery_provenance.get("provenance")
                == "full_completion_contract_revalidation_required"
                and sealed_recovery_provenance.get("source_progress_revision") == 2
                and sealed_recovery_provenance.get("source_failure_message_sha256")
                == SOURCE_REVISION_SUPPORT_RESIDUAL_FAILURE_MESSAGE_SHA256
            )
            sealed_provenance = "full_completion_contract_revalidation_required"
        else:
            raise RuntimeError(
                f"source-revision legacy task {experiment!r} has an invalid "
                "provenance policy"
            )
        if not provenance_matches:
            raise RuntimeError(
                f"source-revision legacy task {experiment!r} carried provenance "
                "mismatch"
            )
        if _normalized_task_status(task) != "completed":
            raise RuntimeError(
                f"source-revision legacy task {experiment!r} is not completed"
            )
        expected_legacy_name = _task_name(
            EXPERIMENT_ORDER.index(experiment) + 1,
            experiment,
            parent_controller_task_id,
        )
        step_result = source_step.get("result")
        if (
            source_step.get("state") != "completed"
            or source_step.get("adopted") is not True
            or source_step.get("task_id") != task_id
            or source_step.get("task_name") != expected_legacy_name
            or source_step.get("predecessor_task_id") != predecessor_task_id
            or not isinstance(step_result, Mapping)
            or step_result.get("task_id") != task_id
        ):
            raise RuntimeError(
                f"source-revision legacy task {experiment!r} sealed completed-step "
                "provenance mismatch"
            )
        if str(getattr(task, "name", "") or "") != expected_legacy_name:
            raise RuntimeError(
                f"source-revision legacy task {experiment!r} name mismatch"
            )
        legacy_nested = experiment in SOURCE_REVISION_LEGACY_NESTED_TEACHER_EXPERIMENTS
        canonical_nested = experiment in NESTED_TEACHER_EXPERIMENTS
        if legacy_nested != canonical_nested:
            raise RuntimeError(
                f"source-revision legacy task {experiment!r} changes subject semantics"
            )
        _validate_clone_identity(task, identity=legacy_template_identity)
        script_identity_policy = (
            "exact_allowlisted_legacy_nested_teacher_semantics_preserving"
        )
        carried_sha256 = hashlib.sha256(
            _canonical_json(dict(sealed_recovery_provenance)).encode("utf-8")
        ).hexdigest()
        legacy_compatibility_receipt = _sealed(
            {
                "schema_version": 1,
                "contract_type": ("source_revision_legacy_script_compatibility"),
                "result": "pass",
                "experiment": experiment,
                "task_id": task_id,
                "parent_controller_task_id": parent_controller_task_id,
                "source_controller_task_id": source_controller_task_id,
                "source_progress_seal_sha256": _sha256(
                    source_progress_seal_sha256,
                    "source-revision progress seal",
                ),
                "legacy_script_sha256": legacy_script_sha256,
                "canonical_script_sha256": canonical_script_sha256,
                "sealed_provenance_type": provenance_type,
                "sealed_recovery_provenance": sealed_provenance,
                "sealed_recovery_provenance_sha256": carried_sha256,
                "sealed_completed_step_sha256": hashlib.sha256(
                    _canonical_json(dict(source_step)).encode("utf-8")
                ).hexdigest(),
                "legacy_nested_teacher_membership": legacy_nested,
                "canonical_nested_teacher_membership": canonical_nested,
                "subject_semantics_equal": True,
                "exception_scope": "script_identity_only",
            }
        )
    expected = _experiment_parameters(
        template_parameters,
        experiment=experiment,
        predecessor_task_id=predecessor_task_id,
        teacher_task_id=_clearml_id(
            teacher_reference.get("task_id"), "recovery teacher task"
        ),
        teacher_model_id=_clearml_id(
            teacher_reference.get("model_id"), "recovery teacher model"
        ),
        teacher_checkpoint_sha256=_sha256(
            teacher_reference.get("checkpoint_sha256"), "recovery teacher checkpoint"
        ),
        allow_failed_teacher_task=bool(
            teacher_reference.get("allow_failed_task", False)
        ),
        training_seed=training_seed,
    )
    observed = _task_parameters(task)
    if not _execution_parameters_match(observed, expected):
        raise RuntimeError(
            f"recovery {template_role}-template task parameters drifted for "
            f"{experiment!r}"
        )
    normalized_expected = {key: str(value) for key, value in expected.items()}
    normalized_observed = {key: str(observed.get(key)) for key in expected}
    expected_sha256 = hashlib.sha256(
        _canonical_json(normalized_expected).encode("utf-8")
    ).hexdigest()
    observed_sha256 = hashlib.sha256(
        _canonical_json(normalized_observed).encode("utf-8")
    ).hexdigest()
    if expected_sha256 != observed_sha256:
        raise RuntimeError(
            f"recovery {template_role}-template normalized parameter evidence "
            f"drifted for {experiment!r}"
        )
    source_parameters = template_identity.get("source_parameters")
    if not isinstance(source_parameters, Mapping):
        raise RuntimeError("recovery task template has no source parameters")
    return {
        "task_id": task_id,
        "experiment": experiment,
        "template_role": template_role,
        "template_task_id": template_identity.get("task_id"),
        "template_script_sha256": template_identity.get("script_sha256"),
        "task_script_sha256": task_script_sha256,
        "script_identity_policy": script_identity_policy,
        "legacy_script_compatibility_receipt": legacy_compatibility_receipt,
        "source_parameters": dict(source_parameters),
        "config_path": _experiment_config_path(experiment),
        "config_inventory_status": (
            "byte_identical_across_revisions"
            if template_role == "source"
            else "target_revision_required"
        ),
        "expected_parameter_count": len(expected),
        "expected_parameters_sha256": expected_sha256,
        "observed_parameter_projection_sha256": observed_sha256,
        "exact_execution_parameter_match": True,
        "predecessor_task_id": predecessor_task_id,
    }


def _validate_transition_replaced_source_task_binding(
    *,
    task: object,
    task_id: str,
    source_controller_task_id: str,
    experiment: str,
    predecessor_task_id: str,
    template_identity: Mapping[str, object],
    template_parameters: Mapping[str, object],
    teacher_reference: Mapping[str, object],
    training_seed: int,
) -> dict[str, object]:
    """Bind a failed old-source task exactly before permitting its replacement."""

    if experiment not in SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS:
        raise RuntimeError("source-revision replaced task is not a target subject")
    if _task_parent(task) != source_controller_task_id:
        raise RuntimeError("source-revision replaced task parent drifted")
    expected_name = _task_name(
        EXPERIMENT_ORDER.index(experiment) + 1,
        experiment,
        source_controller_task_id,
    )
    if str(getattr(task, "name", "") or "") != expected_name:
        raise RuntimeError("source-revision replaced task name drifted")
    _validate_clone_identity(task, identity=template_identity)
    expected = _experiment_parameters(
        template_parameters,
        experiment=experiment,
        predecessor_task_id=predecessor_task_id,
        teacher_task_id=_clearml_id(
            teacher_reference.get("task_id"),
            "source-revision replaced task teacher",
        ),
        teacher_model_id=_clearml_id(
            teacher_reference.get("model_id"),
            "source-revision replaced task teacher model",
        ),
        teacher_checkpoint_sha256=_sha256(
            teacher_reference.get("checkpoint_sha256"),
            "source-revision replaced task teacher checkpoint",
        ),
        allow_failed_teacher_task=bool(
            teacher_reference.get("allow_failed_task", False)
        ),
        training_seed=training_seed,
    )
    observed = _task_parameters(task)
    if not _execution_parameters_match(observed, expected):
        raise RuntimeError(
            f"source-revision replaced task parameters drifted for {experiment!r}"
        )
    normalized_expected = {key: str(value) for key, value in expected.items()}
    normalized_observed = {key: str(observed.get(key)) for key in expected}
    return _sealed(
        {
            "schema_version": 1,
            "contract_type": "source_revision_replaced_source_task_binding",
            "result": "pass",
            "task_id": task_id,
            "experiment": experiment,
            "parent_controller_task_id": source_controller_task_id,
            "task_name": expected_name,
            "template_task_id": template_identity.get("task_id"),
            "template_script_sha256": template_identity.get("script_sha256"),
            "task_script_sha256": _script_sha256(_task_script(task)),
            "source_parameters": dict(template_identity.get("source_parameters", {})),
            "predecessor_task_id": predecessor_task_id,
            "expected_parameters_sha256": hashlib.sha256(
                _canonical_json(normalized_expected).encode("utf-8")
            ).hexdigest(),
            "observed_parameter_projection_sha256": hashlib.sha256(
                _canonical_json(normalized_observed).encode("utf-8")
            ).hexdigest(),
            "exact_execution_parameter_match": True,
        }
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


def _artifact_payload(
    task: object,
    name: str,
    *,
    force_download: bool = False,
) -> dict[str, object]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"task is missing required artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise RuntimeError(f"artifact {name!r} cannot be downloaded")
    if force_download:
        try:
            value = getter(force_download=True)
        except TypeError as error:
            raise RuntimeError(
                f"artifact {name!r} cannot guarantee a fresh download"
            ) from error
    else:
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


def _fresh_sealed_artifact_payload(
    task: object,
    name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Force two stable server readbacks so an overwritten artifact URL cannot be stale."""

    def artifact_url() -> str:
        artifacts = getattr(task, "artifacts", None)
        if not isinstance(artifacts, Mapping) or name not in artifacts:
            raise RuntimeError(f"task is missing required artifact {name!r}")
        value = str(getattr(artifacts[name], "url", "") or "").strip()
        if not value:
            raise RuntimeError(f"artifact {name!r} has no stable readback URL")
        return value

    _reload(task)
    first_url = artifact_url()
    first = _artifact_payload(task, name, force_download=True)
    _require_valid_seal(first, context=f"fresh {name} readback")
    _reload(task)
    second_url = artifact_url()
    if second_url != first_url:
        raise RuntimeError(f"artifact {name!r} URL changed during fresh readback")
    second = _artifact_payload(task, name, force_download=True)
    _require_valid_seal(second, context=f"fresh {name} confirmation")
    if first != second:
        raise RuntimeError(f"artifact {name!r} changed during fresh readback")
    revision = second.get("revision")
    if type(revision) is not int or revision < 1:
        raise RuntimeError(f"artifact {name!r} has an invalid revision")
    return second, {
        "url": second_url,
        "revision": revision,
        "seal_sha256": _sha256(
            second.get("seal_sha256"),
            f"{name} fresh readback seal",
        ),
        "force_download": True,
        "stable_readbacks": 2,
    }


def _teacher_reference_request(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None]:
    resolve = bool(getattr(args, "resolve_teacher_reference", False))
    model_value = str(getattr(args, "teacher_model_id", "") or "").strip()
    sha_value = str(getattr(args, "teacher_checkpoint_sha256", "") or "").strip()
    allow_failed = bool(getattr(args, "allow_failed_teacher_task", False))
    if resolve:
        if model_value or sha_value:
            raise ValueError(
                "--resolve-teacher-reference is mutually exclusive with an "
                "explicit teacher model/SHA pin"
            )
        if allow_failed:
            raise ValueError(
                "automatic teacher resolution requires a completed teacher task"
            )
        return "teacher_checkpoint_contract", None, None
    if bool(model_value) != bool(sha_value):
        raise ValueError(
            "--teacher-model-id and --teacher-checkpoint-sha256 must be supplied "
            "together"
        )
    if not model_value:
        raise ValueError(
            "select --resolve-teacher-reference or provide an explicit teacher "
            "model/SHA pin"
        )
    return (
        "explicit_pin",
        _clearml_id(model_value, "teacher model"),
        _sha256(sha_value, "teacher checkpoint"),
    )


def _teacher_output_models(teacher: object) -> list[object]:
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
        if getattr(model, "name", None) == CLEAN_TEACHER_MODEL_NAME
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "teacher handoff must expose exactly one clean-teacher OutputModel"
        )
    return candidates


def _validate_teacher_reference(
    task_class: object,
    *,
    teacher_task_id: str,
    teacher_model_id: str,
    allow_failed_teacher_task: bool,
) -> dict[str, object]:
    teacher = task_class.get_task(task_id=teacher_task_id)
    status = _normalized_task_status(teacher)
    allowed = status == "completed" or (
        allow_failed_teacher_task and status == "failed"
    )
    if not allowed:
        raise RuntimeError(f"teacher task has unusable status: {status!r}")
    model = _teacher_output_models(teacher)[0]
    if str(getattr(model, "id", "") or "") != teacher_model_id:
        raise RuntimeError("teacher OutputModel ID mismatch")
    if str(getattr(model, "task", "") or "") != teacher_task_id:
        raise RuntimeError("teacher OutputModel ownership mismatch")
    model_url = _require_files_server_url(
        getattr(model, "url", ""), context="teacher model URL"
    )
    return {
        "reference_mode": "explicit_pin",
        "task_id": teacher_task_id,
        "model_id": teacher_model_id,
        "model_name": CLEAN_TEACHER_MODEL_NAME,
        "model_url": model_url,
        "allow_failed_task": allow_failed_teacher_task,
    }


def _positive_contract_integer(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{context} must be a positive integer")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_and_verify_teacher_model(
    model: object,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> None:
    getter = getattr(model, "get_local_copy", None)
    if not callable(getter):
        raise RuntimeError("selected teacher OutputModel cannot be downloaded")
    value = getter(
        extract_archive=False,
        raise_on_error=True,
        force_download=True,
    )
    if not value:
        raise RuntimeError("selected teacher OutputModel returned no local copy")
    path = Path(value)
    if path.is_symlink():
        raise RuntimeError("selected teacher checkpoint must not be a symlink")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("selected teacher checkpoint does not exist") from error
    if not path.is_file():
        raise RuntimeError("selected teacher checkpoint is not a regular file")
    if path.stat().st_size != expected_size_bytes:
        raise RuntimeError("selected teacher checkpoint size mismatch")
    observed_sha256 = _sha256_path(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError("selected teacher checkpoint SHA-256 mismatch")


def _resolve_completed_teacher_reference(
    teacher: object,
    *,
    teacher_task_id: str,
) -> dict[str, object]:
    if _normalized_task_status(teacher) != "completed":
        raise RuntimeError("automatic teacher reference task is not completed")
    parameters = _task_parameters(teacher)
    expected_parameters = {
        "Args/stage": "teacher",
        "Args/max_epochs": 50,
        "Args/gpus": 4,
        "Args/amp": False,
    }
    for key, expected in expected_parameters.items():
        if not _parameter_matches(parameters.get(key), expected):
            raise RuntimeError(f"teacher task parameter {key} mismatch")

    run_contract = _artifact_payload(teacher, RUN_CONTRACT_ARTIFACT)
    expected_run_contract = {
        "stage": "teacher",
        "max_epochs": 50,
        "val_interval": 10,
        "gpus": 4,
        "global_batch_size": 8,
        "amp": False,
        "checkpoint_policy": CLEAN_TEACHER_CHECKPOINT_POLICY,
    }
    for key, expected in expected_run_contract.items():
        if not _parameter_matches(run_contract.get(key), expected):
            raise RuntimeError(f"teacher run contract {key} mismatch")

    contract = _artifact_payload(teacher, TEACHER_CHECKPOINT_ARTIFACT)
    expected_contract_keys = {
        "schema_version",
        "selection_protocol",
        "selection_metric",
        "selection_rule",
        "selected_epoch",
        "selected_checkpoint",
        "trained_epochs",
        "final_epoch",
        "final_checkpoint",
        "downstream_role",
    }
    missing = sorted(expected_contract_keys - set(contract))
    extra = sorted(set(contract) - expected_contract_keys)
    if missing or extra:
        raise RuntimeError(
            "teacher checkpoint contract schema mismatch: "
            f"missing={missing}, extra={extra}"
        )
    literals = {
        "schema_version": 1,
        "selection_protocol": CLEAN_TEACHER_SELECTION_PROTOCOL,
        "selection_metric": CLEAN_TEACHER_SELECTION_METRIC,
        "selection_rule": CLEAN_TEACHER_SELECTION_RULE,
        "trained_epochs": 50,
        "final_epoch": 50,
        "downstream_role": "frozen teacher and trainable student initialization",
    }
    for key, expected in literals.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"teacher checkpoint contract {key} mismatch")
    selected_epoch = _positive_contract_integer(
        contract.get("selected_epoch"), "teacher selected_epoch"
    )
    if selected_epoch > 50:
        raise RuntimeError("teacher selected_epoch exceeds trained epochs")

    selected = contract.get("selected_checkpoint")
    if not isinstance(selected, Mapping):
        raise RuntimeError("teacher selected_checkpoint is not an object")
    selected_keys = {"model_id", "name", "url", "filename", "size_bytes", "sha256"}
    if set(selected) != selected_keys:
        raise RuntimeError("teacher selected_checkpoint schema mismatch")
    model_id = _clearml_id(selected.get("model_id"), "selected teacher model")
    if selected.get("name") != CLEAN_TEACHER_MODEL_NAME:
        raise RuntimeError("teacher selected checkpoint name mismatch")
    filename = str(selected.get("filename") or "")
    filename_pattern = re.compile(
        r"best_resilient_v2x_car_bev_ap_r40_0\.70_teacher_epoch_(\d+)\.pth"
    )
    match = filename_pattern.fullmatch(filename)
    if match is None or int(match.group(1)) != selected_epoch:
        raise RuntimeError("teacher selected checkpoint filename mismatch")
    model_url = _require_files_server_url(
        selected.get("url"), context="selected teacher model URL"
    )
    if Path(unquote(urlsplit(model_url).path)).name != filename:
        raise RuntimeError("selected teacher model URL filename mismatch")
    size_bytes = _positive_contract_integer(
        selected.get("size_bytes"), "teacher selected checkpoint size"
    )
    checkpoint_sha256 = _sha256(selected.get("sha256"), "selected teacher checkpoint")

    final_checkpoint = contract.get("final_checkpoint")
    if not isinstance(final_checkpoint, Mapping):
        raise RuntimeError("teacher final_checkpoint is not an object")
    if set(final_checkpoint) != {"filename", "size_bytes", "sha256"}:
        raise RuntimeError("teacher final_checkpoint schema mismatch")
    if final_checkpoint.get("filename") != "teacher_epoch_50.pth":
        raise RuntimeError("teacher final checkpoint filename mismatch")
    final_size_bytes = _positive_contract_integer(
        final_checkpoint.get("size_bytes"), "teacher final checkpoint size"
    )
    final_sha256 = _sha256(final_checkpoint.get("sha256"), "teacher final checkpoint")

    model = _teacher_output_models(teacher)[0]
    if str(getattr(model, "id", "") or "") != model_id:
        raise RuntimeError("selected teacher OutputModel ID mismatch")
    if str(getattr(model, "task", "") or "") != teacher_task_id:
        raise RuntimeError("selected teacher OutputModel ownership mismatch")
    actual_url = _require_files_server_url(
        getattr(model, "url", ""), context="selected teacher OutputModel URL"
    )
    if actual_url != model_url:
        raise RuntimeError("selected teacher OutputModel URL mismatch")
    _download_and_verify_teacher_model(
        model,
        expected_size_bytes=size_bytes,
        expected_sha256=checkpoint_sha256,
    )
    return {
        "reference_mode": "teacher_checkpoint_contract",
        "task_id": teacher_task_id,
        "model_id": model_id,
        "model_name": CLEAN_TEACHER_MODEL_NAME,
        "model_url": model_url,
        "checkpoint_filename": filename,
        "checkpoint_size_bytes": size_bytes,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_epoch": selected_epoch,
        "final_epoch": 50,
        "final_checkpoint_size_bytes": final_size_bytes,
        "final_checkpoint_sha256": final_sha256,
        "run_contract_artifact": RUN_CONTRACT_ARTIFACT,
        "checkpoint_contract_artifact": TEACHER_CHECKPOINT_ARTIFACT,
        "allow_failed_task": False,
    }


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


def _resolve_teacher_reference(
    args: argparse.Namespace,
    *,
    task_class: object,
    sleeper: Callable[[float], None],
) -> dict[str, object]:
    teacher_task_id = _clearml_id(args.teacher_task_id, "teacher task")
    mode, explicit_model_id, explicit_sha256 = _teacher_reference_request(args)
    if mode == "teacher_checkpoint_contract":
        teacher = task_class.get_task(task_id=teacher_task_id)
        _wait_for_completed(
            teacher,
            context="clean teacher task",
            poll_seconds=args.poll_seconds,
            sleeper=sleeper,
        )
        return _resolve_completed_teacher_reference(
            teacher,
            teacher_task_id=teacher_task_id,
        )
    if explicit_model_id is None or explicit_sha256 is None:
        raise AssertionError("explicit teacher reference was not resolved")
    result = _validate_teacher_reference(
        task_class,
        teacher_task_id=teacher_task_id,
        teacher_model_id=explicit_model_id,
        allow_failed_teacher_task=bool(args.allow_failed_teacher_task),
    )
    result["checkpoint_sha256"] = explicit_sha256
    return result


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


def _task_project_name(task: object) -> str:
    getter = getattr(task, "get_project_name", None)
    value = getter() if callable(getter) else getattr(task, "project", "")
    return str(value or "")


def _parameter_string_list(value: object, context: str) -> list[str]:
    parsed = value
    if type(value) is str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{context} is not a JSON list") from error
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise RuntimeError(f"{context} is not a list")
    result = []
    for item in parsed:
        if type(item) is not str or not item.strip():
            raise RuntimeError(f"{context} contains an invalid item")
        result.append(item.strip())
    return result


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


def _find_exact_recovery_child(
    task_class: object,
    *,
    project: str,
    name: str,
    parent_controller_task_id: str,
) -> object:
    getter = getattr(task_class, "get_tasks", None)
    if not callable(getter):
        raise RuntimeError("Task class cannot search for recovery children")
    candidates = []
    project_query_succeeded = False
    try:
        candidates = list(
            getter(
                project_name=project,
                task_name=f"^{re.escape(name)}$",
                allow_archived=True,
                task_filter={"parent": parent_controller_task_id},
            )
            or []
        )
        project_query_succeeded = True
    except Exception:
        candidates = []
    if not candidates:
        try:
            candidates = list(
                getter(
                    task_name=f"^{re.escape(name)}$",
                    allow_archived=True,
                    task_filter={"parent": parent_controller_task_id},
                )
                or []
            )
        except TypeError:
            try:
                candidates = list(getter(task_name=f"^{re.escape(name)}$") or [])
            except Exception as error:
                if project_query_succeeded:
                    candidates = []
                else:
                    raise RuntimeError(
                        f"cannot search for recovery child {name!r}"
                    ) from error
        except Exception as error:
            raise RuntimeError(f"cannot search for recovery child {name!r}") from error
    exact = [
        task
        for task in candidates
        if str(getattr(task, "name", "") or "") == name
        and _task_parent(task) == parent_controller_task_id
        and _task_project_name(task) == project
    ]
    if len(exact) != 1:
        raise RuntimeError(
            f"expected exactly one recovery child {name!r}; found {len(exact)}"
        )
    return exact[0]


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


def _validate_experiment_run_contract_seed(
    contract: Mapping[str, object],
    *,
    expected_seed: object,
) -> dict[str, object]:
    """Validate the exact seed representation of the deployed schema-v1 task."""

    schema_version = contract.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != EXPERIMENT_RUN_CONTRACT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "experiment run contract seed schema version mismatch: "
            f"expected {EXPERIMENT_RUN_CONTRACT_SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )
    observed_fields = set(contract) & EXPERIMENT_RUN_CONTRACT_SEED_FIELDS
    expected_fields = {"seed"}
    if observed_fields != expected_fields:
        raise RuntimeError(
            "experiment run contract seed field schema mismatch: "
            f"expected={sorted(expected_fields)}, observed={sorted(observed_fields)}"
        )
    if type(expected_seed) is not int or expected_seed < 0:
        raise RuntimeError("expected experiment training seed is invalid")
    observed_seed = contract["seed"]
    if type(observed_seed) is not int or observed_seed != expected_seed:
        raise RuntimeError(
            "experiment run contract seed mismatch: "
            f"expected {expected_seed!r}, got {observed_seed!r}"
        )
    return {
        "run_contract_schema_version": schema_version,
        "run_contract_training_seed_field": "seed",
    }


def _require_exact_mapping_keys(
    value: object,
    *,
    keys: set[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} is not an object")
    mapping = dict(value)
    missing = sorted(keys - set(mapping))
    extra = sorted(set(mapping) - keys)
    if missing or extra:
        raise RuntimeError(
            f"{context} schema mismatch: missing={missing}, extra={extra}"
        )
    return mapping


def _audit_integer(
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
    allow_zero: bool = False,
) -> int:
    value = mapping.get(key)
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise RuntimeError(f"{context}.{key} must be a {qualifier} integer")
    return value


def _audit_sha256(
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str:
    try:
        return _sha256(mapping.get(key), f"{context}.{key}")
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _validate_common_teacher_initialization_audit(
    task: object,
    *,
    run_contract: Mapping[str, object],
    experiment: str,
    teacher_checkpoint_sha256: str,
) -> dict[str, object]:
    expect_nested_teacher = experiment in NESTED_TEACHER_EXPERIMENTS
    common_contract = _require_exact_mapping_keys(
        run_contract.get("common_teacher_initialization"),
        keys={
            "policy",
            "contract",
            "shared_prefixes",
            "expected_source_keys",
            "expected_source_numel",
            "expected_source_bytes",
            "expected_shared_keys",
            "expected_shared_numel",
            "expected_shared_bytes",
            "expected_teacher_fusion_keys",
            "teacher_checkpoint_sha256",
            "audit_artifact_name",
            "audit_filename",
            "expected_nested_teacher",
        },
        context="experiment common teacher initialization contract",
    )
    expected_common_contract = {
        "policy": "shared-only",
        "contract": COMMON_TEACHER_INITIALIZATION_CONTRACT,
        "shared_prefixes": list(COMMON_TEACHER_INITIALIZATION_PREFIXES),
        "expected_source_keys": COMMON_TEACHER_SOURCE_KEYS,
        "expected_source_numel": COMMON_TEACHER_SOURCE_NUMEL,
        "expected_source_bytes": COMMON_TEACHER_SOURCE_BYTES,
        "expected_shared_keys": COMMON_TEACHER_SHARED_KEYS,
        "expected_shared_numel": COMMON_TEACHER_SHARED_NUMEL,
        "expected_shared_bytes": COMMON_TEACHER_SHARED_BYTES,
        "expected_teacher_fusion_keys": COMMON_TEACHER_FUSION_KEYS,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "audit_artifact_name": COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT,
        "audit_filename": "common_teacher_initialization_audit.json",
        "expected_nested_teacher": expect_nested_teacher,
    }
    if common_contract != expected_common_contract:
        raise RuntimeError("experiment common teacher initialization contract drifted")

    try:
        audit = _artifact_payload(task, COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT)
    except RuntimeError as error:
        raise RuntimeError(
            "common teacher initialization audit is unavailable"
        ) from error
    audit = _require_exact_mapping_keys(
        audit,
        keys={
            "schema_version",
            "contract",
            "result",
            "checkpoint",
            "source",
            "shared_initialization",
            "method_specific_fusion",
            "target",
        },
        context="common teacher initialization audit",
    )
    expected_audit_literals = {
        "schema_version": 1,
        "contract": COMMON_TEACHER_INITIALIZATION_CONTRACT,
        "result": "pass",
    }
    for key, expected in expected_audit_literals.items():
        if type(audit.get(key)) is not type(expected) or audit.get(key) != expected:
            raise RuntimeError(f"common teacher initialization audit {key} mismatch")

    checkpoint = _require_exact_mapping_keys(
        audit.get("checkpoint"),
        keys={"path", "filename", "size_bytes", "sha256", "expected_sha256"},
        context="common teacher initialization audit.checkpoint",
    )
    checkpoint_path = checkpoint.get("path")
    checkpoint_filename = checkpoint.get("filename")
    if (
        type(checkpoint_path) is not str
        or not checkpoint_path
        or not Path(checkpoint_path).is_absolute()
        or type(checkpoint_filename) is not str
        or not checkpoint_filename
        or Path(checkpoint_path).name != checkpoint_filename
    ):
        raise RuntimeError(
            "common teacher initialization audit checkpoint path/filename mismatch"
        )
    _audit_integer(
        checkpoint,
        "size_bytes",
        context="common teacher initialization audit.checkpoint",
    )
    observed_checkpoint_sha256 = _audit_sha256(
        checkpoint,
        "sha256",
        context="common teacher initialization audit.checkpoint",
    )
    expected_checkpoint_sha256 = _audit_sha256(
        checkpoint,
        "expected_sha256",
        context="common teacher initialization audit.checkpoint",
    )
    if {
        observed_checkpoint_sha256,
        expected_checkpoint_sha256,
    } != {teacher_checkpoint_sha256}:
        raise RuntimeError(
            "common teacher initialization audit checkpoint SHA-256 mismatch"
        )

    source = _require_exact_mapping_keys(
        audit.get("source"),
        keys={
            "keys",
            "numel",
            "bytes",
            "expected_keys",
            "common_keys",
            "expected_common_keys",
            "fusion_keys",
            "expected_fusion_keys",
            "state_sha256",
        },
        context="common teacher initialization audit.source",
    )
    expected_source = {
        "keys": COMMON_TEACHER_SOURCE_KEYS,
        "numel": COMMON_TEACHER_SOURCE_NUMEL,
        "bytes": COMMON_TEACHER_SOURCE_BYTES,
        "expected_keys": COMMON_TEACHER_SOURCE_KEYS,
        "common_keys": COMMON_TEACHER_SHARED_KEYS,
        "expected_common_keys": COMMON_TEACHER_SHARED_KEYS,
        "fusion_keys": COMMON_TEACHER_FUSION_KEYS,
        "expected_fusion_keys": COMMON_TEACHER_FUSION_KEYS,
    }
    for key, expected in expected_source.items():
        if type(source.get(key)) is not int or source.get(key) != expected:
            raise RuntimeError(
                f"common teacher initialization audit.source.{key} mismatch"
            )
    _audit_sha256(
        source,
        "state_sha256",
        context="common teacher initialization audit.source",
    )

    shared = _require_exact_mapping_keys(
        audit.get("shared_initialization"),
        keys={
            "prefixes",
            "keys",
            "numel",
            "bytes",
            "expected_keys",
            "state_sha256",
            "shape_dtype_verified",
            "exact_tensor_equality_verified",
        },
        context="common teacher initialization audit.shared_initialization",
    )
    expected_shared = {
        "prefixes": list(COMMON_TEACHER_INITIALIZATION_PREFIXES),
        "keys": COMMON_TEACHER_SHARED_KEYS,
        "numel": COMMON_TEACHER_SHARED_NUMEL,
        "bytes": COMMON_TEACHER_SHARED_BYTES,
        "expected_keys": COMMON_TEACHER_SHARED_KEYS,
        "shape_dtype_verified": True,
        "exact_tensor_equality_verified": True,
    }
    for key, expected in expected_shared.items():
        if type(shared.get(key)) is not type(expected) or shared.get(key) != expected:
            raise RuntimeError(
                "common teacher initialization audit.shared_initialization."
                f"{key} mismatch"
            )
    _audit_sha256(
        shared,
        "state_sha256",
        context="common teacher initialization audit.shared_initialization",
    )

    fusion = _require_exact_mapping_keys(
        audit.get("method_specific_fusion"),
        keys={"keys", "numel", "bytes", "sha256_before", "sha256_after", "unchanged"},
        context="common teacher initialization audit.method_specific_fusion",
    )
    zero_allowed = experiment in ZERO_FUSION_ALLOWED_EXPERIMENTS
    fusion_keys = _audit_integer(
        fusion,
        "keys",
        context="common teacher initialization audit.method_specific_fusion",
        allow_zero=zero_allowed,
    )
    fusion_numel = _audit_integer(
        fusion,
        "numel",
        context="common teacher initialization audit.method_specific_fusion",
        allow_zero=zero_allowed,
    )
    fusion_bytes = _audit_integer(
        fusion,
        "bytes",
        context="common teacher initialization audit.method_specific_fusion",
        allow_zero=zero_allowed,
    )
    if fusion_keys == 0 and (fusion_numel != 0 or fusion_bytes != 0):
        raise RuntimeError("zero-key method-specific fusion has nonzero tensor stats")
    if fusion_keys > 0 and (fusion_numel == 0 or fusion_bytes == 0):
        raise RuntimeError("nonempty method-specific fusion has empty tensor stats")
    before_sha256 = _audit_sha256(
        fusion,
        "sha256_before",
        context="common teacher initialization audit.method_specific_fusion",
    )
    after_sha256 = _audit_sha256(
        fusion,
        "sha256_after",
        context="common teacher initialization audit.method_specific_fusion",
    )
    if fusion.get("unchanged") is not True or before_sha256 != after_sha256:
        raise RuntimeError(
            "common teacher initialization audit method-specific fusion changed"
        )

    target = _require_exact_mapping_keys(
        audit.get("target"),
        keys={
            "model_type",
            "target_key_count",
            "target_common_key_count",
            "target_fusion_key_count",
            "nested_teacher_present",
            "nested_teacher_key_count",
            "nested_teacher_full_equality_verified",
        },
        context="common teacher initialization audit.target",
    )
    if type(target.get("model_type")) is not str or not target["model_type"]:
        raise RuntimeError("common teacher initialization audit target model invalid")
    expected_target = {
        "target_key_count": (
            COMMON_TEACHER_SHARED_KEYS
            + fusion_keys
            + (COMMON_TEACHER_SOURCE_KEYS if expect_nested_teacher else 0)
        ),
        "target_common_key_count": COMMON_TEACHER_SHARED_KEYS,
        "target_fusion_key_count": fusion_keys,
        "nested_teacher_present": expect_nested_teacher,
        "nested_teacher_key_count": (
            COMMON_TEACHER_SOURCE_KEYS if expect_nested_teacher else 0
        ),
        "nested_teacher_full_equality_verified": expect_nested_teacher,
    }
    for key, expected in expected_target.items():
        if type(target.get(key)) is not type(expected) or target.get(key) != expected:
            raise RuntimeError(
                f"common teacher initialization audit.target.{key} mismatch"
            )

    content_sha256 = hashlib.sha256(_canonical_json(audit).encode("utf-8")).hexdigest()
    return {
        "common_teacher_initialization_audit_artifact": (
            COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT
        ),
        "common_teacher_initialization_audit_sha256": content_sha256,
    }


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

    contract = _artifact_payload(
        task,
        RUN_CONTRACT_ARTIFACT,
        force_download=True,
    )
    task_id = str(getattr(task, "id", "") or "")
    spec = EXPERIMENT_BY_NAME[experiment]
    expected_contract = {
        "schema_version": EXPERIMENT_RUN_CONTRACT_SCHEMA_VERSION,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": experiment,
        "experiment_kind": spec.kind,
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
    seed_binding = _validate_experiment_run_contract_seed(
        contract,
        expected_seed=expected_parameters["Args/training_seed"],
    )
    config = contract.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("experiment run contract has invalid config")
    if config.get("declared") != spec.config:
        raise RuntimeError("experiment declared config drifted")
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
    if spec.requires_teacher:
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

    audit_binding = _validate_common_teacher_initialization_audit(
        task,
        run_contract=contract,
        experiment=experiment,
        teacher_checkpoint_sha256=teacher_checkpoint_sha256,
    )
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
        "training_seed": expected_parameters["Args/training_seed"],
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "model_id": model_id,
        "model_name": checkpoint_expected["name"],
        "model_url": model_url,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": size_bytes,
        "run_contract_artifact": RUN_CONTRACT_ARTIFACT,
        "final_checkpoint_artifact": FINAL_CHECKPOINT_ARTIFACT,
        **seed_binding,
        **audit_binding,
    }


def _new_progress(
    *,
    controller_task_id: str,
    gate_task_id: str,
    template_identity: Mapping[str, object],
    teacher_reference: Mapping[str, object],
    worker_queues: Sequence[str],
    max_parallel_training_tasks: int,
    training_seed: int,
    recovery: Mapping[str, object] | None,
) -> dict[str, object]:
    queues = [str(item) for item in worker_queues]
    return {
        "schema_version": 1,
        "controller_type": "resilient_v2x_post_main_sequential_training",
        "controller_task_id": controller_task_id,
        "gate_task_id": gate_task_id,
        "gate_policy": "exact_task_must_be_completed_before_any_clone",
        "template": dict(template_identity),
        "teacher": dict(teacher_reference),
        "worker_queue": queues[0],
        "worker_queues": queues,
        "experiment_order": list(EXPERIMENT_ORDER),
        "max_parallel_training_tasks": int(max_parallel_training_tasks),
        "training_seed": training_seed,
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "recovery": dict(recovery) if recovery is not None else None,
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
    teacher_reference: Mapping[str, object],
    worker_queues: Sequence[str],
    max_parallel_training_tasks: int,
    training_seed: int,
    recovery: Mapping[str, object] | None,
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
        "training_seed": training_seed,
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "recovery": dict(recovery) if recovery is not None else None,
    }
    for key, value in expected.items():
        if progress.get(key) != value:
            raise RuntimeError(f"training progress {key} mismatch")
    if progress.get("template") != dict(template_identity):
        raise RuntimeError("training progress template identity mismatch")
    teacher = progress.get("teacher")
    if teacher != dict(teacher_reference):
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
    active_queue_counts: Counter[str] = Counter()
    queue_capacities = Counter(queues)
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
                raise RuntimeError(
                    "pending training progress step already has a task ID"
                )
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
            if step.get("worker_queue") != "adopted-external":
                active += 1
                queue = step.get("worker_queue")
                if queue not in (None, ""):
                    if type(queue) is not str or queue not in queue_capacities:
                        raise RuntimeError(
                            "training progress uses an undeclared worker queue"
                        )
                    active_queue_counts[queue] += 1
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("training progress reuses a task ID")
    if active > int(max_parallel_training_tasks):
        raise RuntimeError("training progress contains too many active tasks")
    for queue, count in active_queue_counts.items():
        if count > queue_capacities[queue]:
            raise RuntimeError(
                f"training progress exceeds worker queue capacity: {queue}"
            )


def _load_or_create_progress(
    controller_task: object,
    **identity: object,
) -> dict[str, object]:
    artifacts = getattr(controller_task, "artifacts", None)
    if isinstance(artifacts, Mapping) and PROGRESS_ARTIFACT in artifacts:
        progress, _readback = _fresh_sealed_artifact_payload(
            controller_task,
            PROGRESS_ARTIFACT,
        )
        _validate_progress(progress, **identity)
        return progress
    progress = _new_progress(**identity)
    _save_progress(controller_task, progress)
    return progress


def _validate_carried_recovery_target_adoption(
    *,
    task_class: object,
    task: object,
    task_id: str,
    task_status: str,
    experiment: str,
    predecessor_task_id: str,
    observation: Mapping[str, object],
    gate_task_id: str,
    project: str,
) -> dict[str, object]:
    sealed = _require_exact_mapping_keys(
        observation,
        keys={
            "task_id",
            "task_name",
            "task_status",
            "task_last_update",
            "task_script_sha256",
            "parent_controller_task_id",
            "parent_controller_status",
            "parent_controller_last_update",
            "parent_controller_script_sha256",
            "source_snapshot_state",
            "source_snapshot_task_id",
            "predecessor_task_id",
            "provenance",
        },
        context=f"source recovery target-adoption {experiment}",
    )
    if sealed["task_id"] != task_id:
        raise RuntimeError("carried recovery-target task ID mismatch")
    parent_id = _clearml_id(
        sealed["parent_controller_task_id"],
        "carried recovery-target parent controller",
    )
    expected_name = _task_name(
        EXPERIMENT_ORDER.index(experiment) + 1,
        experiment,
        parent_id,
    )
    if (
        sealed["task_name"] != expected_name
        or str(getattr(task, "name", "") or "") != expected_name
    ):
        raise RuntimeError("carried recovery-target task name mismatch")
    if _task_parent(task) != parent_id:
        raise RuntimeError("carried recovery-target parent mismatch")
    provenance = sealed["provenance"]
    if provenance == "explicit_failed_recovery_target":
        if (
            sealed["source_snapshot_state"] != "pending"
            or sealed["source_snapshot_task_id"] is not None
        ):
            raise RuntimeError("carried recovery-target source snapshot mismatch")
    elif provenance == "explicit_failed_source_revision_target_replacement":
        if sealed["source_snapshot_state"] not in {
            "created",
            "queued",
            "running",
            "failed",
        }:
            raise RuntimeError(
                "carried source-revision replacement snapshot state mismatch"
            )
        replaced_task_id = _clearml_id(
            sealed["source_snapshot_task_id"],
            "carried source-revision replaced task",
        )
        if replaced_task_id == task_id:
            raise RuntimeError(
                "carried source-revision replacement reuses the old task"
            )
    else:
        raise RuntimeError("carried recovery-target provenance mismatch")
    if sealed["predecessor_task_id"] != predecessor_task_id:
        raise RuntimeError("carried recovery-target predecessor mismatch")
    if sealed["task_status"] not in {"queued", "in_progress", "completed"}:
        raise RuntimeError("carried recovery-target sealed task status is invalid")
    _utc_timestamp(
        sealed["task_last_update"],
        "carried recovery-target sealed task last_update",
    )
    if _script_sha256(_task_script(task)) != _sha256(
        sealed["task_script_sha256"],
        "carried recovery-target task script",
    ):
        raise RuntimeError("carried recovery-target task script mismatch")
    parameters = _task_parameters(task)
    if parameters.get("Args/experiment_from_task") != experiment:
        raise RuntimeError("carried recovery-target experiment parameter mismatch")
    if parameters.get("Args/predecessor_task_id") != predecessor_task_id:
        raise RuntimeError("carried recovery-target predecessor parameter mismatch")

    parent = task_class.get_task(task_id=parent_id)
    _reload(parent)
    if _normalized_task_status(parent) != "failed":
        raise RuntimeError("carried recovery-target parent is not failed")
    if sealed["parent_controller_status"] != "failed":
        raise RuntimeError("carried recovery-target sealed parent status mismatch")
    if _task_parent(parent) != gate_task_id:
        raise RuntimeError("carried recovery-target parent gate mismatch")
    if _task_project_name(parent) != project or _task_project_name(task) != project:
        raise RuntimeError("carried recovery-target project mismatch")
    _utc_timestamp(
        sealed["parent_controller_last_update"],
        "carried recovery-target sealed parent last_update",
    )
    if _script_sha256(_task_script(parent)) != _sha256(
        sealed["parent_controller_script_sha256"],
        "carried recovery-target parent script",
    ):
        raise RuntimeError("carried recovery-target parent script mismatch")

    carried = dict(sealed)
    carried["task_status"] = task_status
    carried["task_last_update"] = _task_last_update(
        task,
        context=f"carried recovery-target {experiment} task",
    ).isoformat()
    carried["parent_controller_last_update"] = _task_last_update(
        parent,
        context="carried recovery-target parent controller",
    ).isoformat()
    return carried


def _resolve_failed_controller_recovery(
    *,
    task_class: object,
    controller_task_id: str,
    source_controller_task_id: object,
    rerun_experiments: Sequence[str],
    recovery_target_adoptions: Mapping[str, str],
    gate_task_id: str,
    source_template_identity: Mapping[str, object],
    source_legacy_template_identity: Mapping[str, object] | None,
    target_template_identity: Mapping[str, object],
    source_template_parameters: Mapping[str, object],
    target_template_parameters: Mapping[str, object],
    source_revision_transition: Mapping[str, object] | None,
    teacher_reference: Mapping[str, object],
    worker_queues: Sequence[str],
    max_parallel_training_tasks: int,
    training_seed: int,
    project: str,
    sealed_recovery: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, str], dict[str, str]]:
    """Fork a failed sealed chain without editing it or reusing failed tasks."""

    source_id = _clearml_id(
        source_controller_task_id,
        "failed source controller task",
    )
    if source_id == controller_task_id:
        raise ValueError("recovery source and target controller tasks must differ")
    rerun = tuple(rerun_experiments)
    if len(rerun) != len(set(rerun)):
        raise ValueError("failed-controller recovery rerun list contains duplicates")
    if any(experiment not in EXPERIMENT_BY_NAME for experiment in rerun):
        raise ValueError("failed-controller recovery names an unknown experiment")

    source_task = task_class.get_task(task_id=source_id)
    _reload(source_task)
    if _normalized_task_status(source_task) != "failed":
        raise RuntimeError("recovery source controller must be failed")
    source_progress, source_progress_artifact = _fresh_sealed_artifact_payload(
        source_task,
        PROGRESS_ARTIFACT,
    )
    source_recovery = source_progress.get("recovery")
    if source_recovery is not None and not isinstance(source_recovery, Mapping):
        raise RuntimeError("source controller recovery contract is invalid")
    source_recovery_schema_version: int | None = None
    source_recovery_target_adoptions: dict[str, Mapping[str, object]] = {}
    source_recovery_adopted_task_ids: Mapping[str, object] = {}
    source_recovery_adopted_predecessors: Mapping[str, object] = {}
    source_recovery_rerun_experiments: tuple[str, ...] = ()
    source_recovery_rerun_source_task_ids: Mapping[str, object] = {}
    source_recovery_rerun_predecessors: Mapping[str, object] = {}
    source_recovery_rerun_observations: Mapping[str, object] = {}
    source_recovery_completion_validation_retries: dict[str, Mapping[str, object]] = {}
    if source_recovery is None:
        if not rerun and source_revision_transition is None:
            raise ValueError(
                "initial failed-controller recovery requires an explicit rerun list"
            )
    else:
        source_recovery_schema_version = source_recovery.get("schema_version")
        if (
            type(source_recovery_schema_version) is not int
            or source_recovery_schema_version
            not in SUPPORTED_SOURCE_RECOVERY_SCHEMA_VERSIONS
        ):
            raise RuntimeError("source controller recovery contract version mismatch")
        if source_recovery.get("mode") != "failed_controller_immutable_fork":
            raise RuntimeError("source controller recovery mode mismatch")
        raw_target_adoptions = source_recovery.get("recovery_target_adoptions")
        if not isinstance(raw_target_adoptions, Mapping):
            raise RuntimeError(
                "source controller recovery target-adoption contract is invalid"
            )
        for experiment, raw_observation in raw_target_adoptions.items():
            if experiment not in EXPERIMENT_BY_NAME or not isinstance(
                raw_observation,
                Mapping,
            ):
                raise RuntimeError(
                    "source controller recovery target-adoption observation is invalid"
                )
            source_recovery_target_adoptions[str(experiment)] = raw_observation
        raw_adopted_task_ids = source_recovery.get("adopted_task_ids")
        raw_adopted_predecessors = source_recovery.get("adopted_predecessor_task_ids")
        if not isinstance(raw_adopted_task_ids, Mapping) or not isinstance(
            raw_adopted_predecessors,
            Mapping,
        ):
            raise RuntimeError(
                "source controller recovery adoption inventory is invalid"
            )
        source_recovery_adopted_task_ids = raw_adopted_task_ids
        source_recovery_adopted_predecessors = raw_adopted_predecessors
        raw_source_rerun = source_recovery.get("rerun_experiments")
        if not isinstance(raw_source_rerun, Sequence) or isinstance(
            raw_source_rerun,
            (str, bytes),
        ):
            raise RuntimeError("source controller recovery rerun list is invalid")
        source_recovery_rerun_experiments = tuple(raw_source_rerun)
        if any(
            type(experiment) is not str or experiment not in EXPERIMENT_BY_NAME
            for experiment in source_recovery_rerun_experiments
        ) or len(source_recovery_rerun_experiments) != len(
            set(source_recovery_rerun_experiments)
        ):
            raise RuntimeError("source controller recovery rerun list is invalid")
        raw_rerun_source_task_ids = source_recovery.get("rerun_source_task_ids")
        raw_rerun_predecessors = source_recovery.get("rerun_predecessor_task_ids")
        raw_rerun_observations = source_recovery.get("rerun_task_observations")
        rerun_inventory = set(source_recovery_rerun_experiments)
        if (
            not isinstance(raw_rerun_source_task_ids, Mapping)
            or not isinstance(raw_rerun_predecessors, Mapping)
            or not isinstance(raw_rerun_observations, Mapping)
            or set(raw_rerun_source_task_ids) != rerun_inventory
            or set(raw_rerun_predecessors) != rerun_inventory
            or set(raw_rerun_observations) != rerun_inventory
        ):
            raise RuntimeError("source controller recovery rerun inventory is invalid")
        for experiment in source_recovery_rerun_experiments:
            original_task_id = _clearml_id(
                raw_rerun_source_task_ids[experiment],
                f"source recovery {experiment} original task",
            )
            _clearml_id(
                raw_rerun_predecessors[experiment],
                f"source recovery {experiment} predecessor",
            )
            observation = raw_rerun_observations[experiment]
            if (
                not isinstance(observation, Mapping)
                or observation.get("task_id") != original_task_id
                or observation.get("terminal_status") not in FAILED_STATUSES
            ):
                raise RuntimeError(
                    f"source controller recovery rerun observation is invalid for "
                    f"{experiment!r}"
                )
        source_recovery_rerun_source_task_ids = raw_rerun_source_task_ids
        source_recovery_rerun_predecessors = raw_rerun_predecessors
        source_recovery_rerun_observations = raw_rerun_observations
        raw_completion_retries = source_recovery.get(
            "completion_validation_retries", {}
        )
        if not isinstance(raw_completion_retries, Mapping):
            raise RuntimeError(
                "source controller completion-validation retry inventory is invalid"
            )
        for experiment, raw_observation in raw_completion_retries.items():
            if experiment not in EXPERIMENT_BY_NAME:
                raise RuntimeError(
                    "source controller completion-validation retry subject is invalid"
                )
            observation = _require_exact_mapping_keys(
                raw_observation,
                keys={
                    "task_id",
                    "source_progress_revision",
                    "source_snapshot_state",
                    "source_failure_status",
                    "source_failure_message_sha256",
                    "live_task_status",
                    "provenance",
                },
                context=(f"source controller completion-validation retry {experiment}"),
            )
            _clearml_id(
                observation["task_id"],
                f"source completion-validation retry {experiment} task",
            )
            if (
                type(observation["source_progress_revision"]) is not int
                or int(observation["source_progress_revision"]) < 1
                or observation["source_snapshot_state"] != "failed"
                or observation["source_failure_status"]
                != "completion_validation_failed"
                or observation["live_task_status"] != "completed"
                or observation["provenance"]
                != "full_completion_contract_revalidation_required"
            ):
                raise RuntimeError(
                    "source controller completion-validation retry provenance "
                    f"is invalid for {experiment!r}"
                )
            _sha256(
                observation["source_failure_message_sha256"],
                f"source completion-validation retry {experiment} failure message",
            )
            source_recovery_completion_validation_retries[str(experiment)] = observation
    observed_template = source_progress.get("template")
    source_progress_template_equivalence: dict[str, object] | None = None
    if source_revision_transition is not None:
        _require_valid_seal(
            source_revision_transition,
            context="source revision transition",
        )
        source_progress_template_equivalence = (
            _template_identity_parameter_equivalence_receipt(
                observed_template,
                source_template_identity,
                context="source-revision controller template identity",
            )
        )
        if not isinstance(observed_template, Mapping):
            raise AssertionError("validated source template is not an object")
        # Preserve the exact trusted representation from the already sealed source
        # progress when validating that progress.  The receipt above proves the
        # fresh template differs, if at all, only by ClearML int/string casting.
        validated_source_template = dict(observed_template)
    elif observed_template == dict(source_template_identity):
        validated_source_template = source_template_identity
    elif observed_template == dict(target_template_identity):
        # A failed recovery controller already records the fixed template identity.
        validated_source_template = target_template_identity
    else:
        raise RuntimeError(
            "source controller template is neither the exact legacy nor fixed identity"
        )
    _validate_progress(
        source_progress,
        controller_task_id=source_id,
        gate_task_id=gate_task_id,
        template_identity=validated_source_template,
        teacher_reference=teacher_reference,
        worker_queues=worker_queues,
        max_parallel_training_tasks=max_parallel_training_tasks,
        training_seed=training_seed,
        recovery=source_recovery,
    )
    steps = source_progress.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise RuntimeError("source controller progress steps are invalid")

    revision = source_progress.get("revision")
    if type(revision) is not int or revision < 1:
        raise RuntimeError("source controller progress revision is invalid")
    snapshot_updated_at = _utc_timestamp(
        source_progress.get("updated_at"),
        "source controller progress updated_at",
    )
    rerun_set = set(rerun)

    adopted: dict[str, str] = {}
    predecessors: dict[str, str] = {}
    rerun_task_ids: dict[str, str] = {}
    rerun_observations: dict[str, dict[str, object]] = {}
    completion_validation_retries: dict[str, dict[str, object]] = {}
    carried_recovery_target_observations: dict[str, dict[str, object]] = {}
    recovered_pending_target_children: dict[str, dict[str, object]] = {}
    transition_replaced_source_tasks: dict[str, dict[str, object]] = {}
    source_steps: dict[str, Mapping[str, object]] = {}
    for step in steps:
        if not isinstance(step, Mapping):
            raise RuntimeError("source controller progress step is invalid")
        experiment = str(step.get("experiment") or "")
        source_steps[experiment] = step
        state = str(step.get("state") or "")
        raw_task_id = step.get("task_id")
        if raw_task_id is None:
            if experiment in rerun_set:
                raise RuntimeError(
                    f"rerun experiment {experiment!r} has no source task"
                )
            if state != "pending":
                raise RuntimeError(f"source experiment {experiment!r} has no task ID")
            carried_observation = source_recovery_target_adoptions.get(experiment)
            if carried_observation is not None:
                task_id = _clearml_id(
                    source_recovery_adopted_task_ids.get(experiment),
                    f"source recovery target-adoption {experiment} task",
                )
                predecessor = _clearml_id(
                    source_recovery_adopted_predecessors.get(experiment),
                    f"source recovery target-adoption {experiment} predecessor",
                )
                task = task_class.get_task(task_id=task_id)
                _reload(task)
                task_status = _normalized_task_status(task)
                if task_status not in WAITABLE_STATUSES | {"completed"}:
                    raise RuntimeError(
                        f"carried recovery-target experiment {experiment!r} is "
                        f"not adoptable: {task_status!r}"
                    )
                carried_recovery_target_observations[experiment] = (
                    _validate_carried_recovery_target_adoption(
                        task_class=task_class,
                        task=task,
                        task_id=task_id,
                        task_status=task_status,
                        experiment=experiment,
                        predecessor_task_id=predecessor,
                        observation=carried_observation,
                        gate_task_id=gate_task_id,
                        project=project,
                    )
                )
                adopted[experiment] = task_id
                predecessors[experiment] = predecessor
                continue
            if experiment in source_recovery_rerun_experiments:
                predecessor = _clearml_id(
                    source_recovery_rerun_predecessors.get(experiment),
                    f"source recovery pending {experiment} predecessor",
                )
                expected_name = _task_name(
                    EXPERIMENT_ORDER.index(experiment) + 1,
                    experiment,
                    source_id,
                )
                if step.get("task_name") != expected_name:
                    raise RuntimeError(
                        f"source recovery pending {experiment!r} task name mismatch"
                    )
                task = _find_exact_recovery_child(
                    task_class,
                    project=project,
                    name=expected_name,
                    parent_controller_task_id=source_id,
                )
                _reload(task)
                task_status = _normalized_task_status(task)
                if task_status not in WAITABLE_STATUSES | {"completed"}:
                    raise RuntimeError(
                        f"source recovery pending child {experiment!r} is not "
                        f"adoptable: {task_status!r}"
                    )
                _validate_clone_identity(task, identity=target_template_identity)
                parameters = _task_parameters(task)
                if parameters.get("Args/experiment_from_task") != experiment:
                    raise RuntimeError(
                        f"source recovery pending child {experiment!r} parameter "
                        "mismatch"
                    )
                if parameters.get("Args/predecessor_task_id") != predecessor:
                    raise RuntimeError(
                        f"source recovery pending child {experiment!r} predecessor "
                        "mismatch"
                    )
                task_id = _clearml_id(
                    getattr(task, "id", ""),
                    f"source recovery pending {experiment} child",
                )
                original_source_task_id = _clearml_id(
                    source_recovery_rerun_source_task_ids.get(experiment),
                    f"source recovery {experiment} original task",
                )
                original_observation = source_recovery_rerun_observations[experiment]
                if not isinstance(original_observation, Mapping):
                    raise RuntimeError(
                        f"source recovery {experiment!r} observation is invalid"
                    )
                adopted[experiment] = task_id
                predecessors[experiment] = predecessor
                recovered_pending_target_children[experiment] = {
                    "task_id": task_id,
                    "task_name": expected_name,
                    "task_status": task_status,
                    "task_last_update": _task_last_update(
                        task,
                        context=f"source recovery pending {experiment} child",
                    ).isoformat(),
                    "task_script_sha256": _script_sha256(_task_script(task)),
                    "parent_controller_task_id": source_id,
                    "source_snapshot_state": "pending",
                    "source_snapshot_task_id": None,
                    "predecessor_task_id": predecessor,
                    "recovery_intent_source_task_id": original_source_task_id,
                    "recovery_intent_terminal_status": original_observation.get(
                        "terminal_status"
                    ),
                    "provenance": "canonical_child_of_failed_recovery_controller",
                }
            continue
        if state not in {"created", "queued", "running", "completed", "failed"}:
            raise RuntimeError(
                f"source experiment {experiment!r} has invalid snapshot state {state!r}"
            )
        task_id = _clearml_id(raw_task_id, f"source {experiment} task")
        source_experiment_task = task_class.get_task(task_id=task_id)
        _reload(source_experiment_task)
        actual_status = _normalized_task_status(source_experiment_task)
        predecessor = _clearml_id(
            step.get("predecessor_task_id"),
            f"source {experiment} predecessor",
        )
        carried_observation = source_recovery_target_adoptions.get(experiment)
        if carried_observation is not None:
            if source_recovery_adopted_task_ids.get(experiment) != task_id:
                raise RuntimeError(
                    f"source recovery target-adoption task mismatch for {experiment!r}"
                )
            if source_recovery_adopted_predecessors.get(experiment) != predecessor:
                raise RuntimeError(
                    "source recovery target-adoption predecessor mismatch for "
                    f"{experiment!r}"
                )
            carried_recovery_target_observations[experiment] = (
                _validate_carried_recovery_target_adoption(
                    task_class=task_class,
                    task=source_experiment_task,
                    task_id=task_id,
                    task_status=actual_status,
                    experiment=experiment,
                    predecessor_task_id=predecessor,
                    observation=carried_observation,
                    gate_task_id=gate_task_id,
                    project=project,
                )
            )
        transition_target_replacement = (
            source_revision_transition is not None
            and experiment in SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
            and experiment in recovery_target_adoptions
        )
        if actual_status in FAILED_STATUSES and transition_target_replacement:
            if experiment in rerun_set:
                raise RuntimeError(
                    f"source-revision target replacement {experiment!r} cannot "
                    "also be rerun"
                )
            if state not in {"created", "queued", "running", "failed"}:
                raise RuntimeError(
                    f"source-revision replaced experiment {experiment!r} terminal "
                    f"status conflicts with sealed snapshot state {state!r}"
                )
            task_last_update = _task_last_update(
                source_experiment_task,
                context=f"source-revision replaced {experiment} task",
            )
            provenance = "sealed_snapshot"
            if state != "failed":
                if task_last_update <= snapshot_updated_at:
                    raise RuntimeError(
                        f"source-revision replaced experiment {experiment!r} "
                        "terminal status cannot be proven to postdate the sealed "
                        "snapshot"
                    )
                provenance = "observed_after_snapshot"
            transition_replaced_source_tasks[experiment] = {
                "task_id": task_id,
                "snapshot_progress_revision": revision,
                "snapshot_progress_updated_at": snapshot_updated_at.isoformat(),
                "snapshot_state": state,
                "snapshot_status": step.get("failure_status"),
                "task_last_update": task_last_update.isoformat(),
                "terminal_status": actual_status,
                "replacement_template_role": "target",
                "source_task_binding_receipt": (
                    _validate_transition_replaced_source_task_binding(
                        task=source_experiment_task,
                        task_id=task_id,
                        source_controller_task_id=source_id,
                        experiment=experiment,
                        predecessor_task_id=predecessor,
                        template_identity=source_template_identity,
                        template_parameters=source_template_parameters,
                        teacher_reference=teacher_reference,
                        training_seed=training_seed,
                    )
                ),
                "provenance": provenance,
            }
            predecessors[experiment] = predecessor
            continue
        if actual_status in FAILED_STATUSES:
            if experiment not in rerun_set:
                raise RuntimeError(
                    f"source experiment {experiment!r} is terminal {actual_status!r} "
                    "but was not explicitly included in --rerun-failed-experiment"
                )
            if state not in {"created", "queued", "running", "failed"}:
                raise RuntimeError(
                    f"source experiment {experiment!r} terminal status conflicts "
                    f"with sealed snapshot state {state!r}"
                )
            task_last_update = _task_last_update(
                source_experiment_task,
                context=f"source {experiment} task",
            )
            provenance = "sealed_snapshot"
            if state != "failed":
                if task_last_update <= snapshot_updated_at:
                    raise RuntimeError(
                        f"source experiment {experiment!r} terminal status cannot be "
                        "proven to postdate the sealed snapshot"
                    )
                provenance = "observed_after_snapshot"
            snapshot_status = step.get("failure_status")
            if snapshot_status is not None:
                snapshot_status = str(snapshot_status)
            rerun_task_ids[experiment] = task_id
            predecessors[experiment] = predecessor
            rerun_observations[experiment] = {
                "task_id": task_id,
                "snapshot_progress_revision": revision,
                "snapshot_progress_updated_at": snapshot_updated_at.isoformat(),
                "snapshot_state": state,
                "snapshot_status": snapshot_status,
                "task_last_update": task_last_update.isoformat(),
                "terminal_status": actual_status,
                "provenance": provenance,
            }
            continue
        if experiment in rerun_set:
            raise RuntimeError(
                f"rerun experiment {experiment!r} source task is not terminal: "
                f"{actual_status!r}"
            )
        if state == "failed" and actual_status == "completed":
            if (
                source_recovery is None
                or step.get("failure_status") != "completion_validation_failed"
            ):
                raise RuntimeError(
                    f"source experiment {experiment!r} has an untrusted "
                    "completed-task failure transition"
                )
            failure_message = step.get("failure_message")
            if type(failure_message) is not str or not failure_message:
                raise RuntimeError(
                    f"source experiment {experiment!r} completion-validation "
                    "failure has no message"
                )
            adopted[experiment] = task_id
            predecessors[experiment] = predecessor
            completion_validation_retries[experiment] = {
                "task_id": task_id,
                "source_progress_revision": revision,
                "source_snapshot_state": "failed",
                "source_failure_status": "completion_validation_failed",
                "source_failure_message_sha256": hashlib.sha256(
                    failure_message.encode("utf-8")
                ).hexdigest(),
                "live_task_status": "completed",
                "provenance": "full_completion_contract_revalidation_required",
            }
            continue
        if state == "failed":
            raise RuntimeError(
                f"source experiment {experiment!r} is sealed failed but its task "
                f"status is {actual_status!r}"
            )
        if actual_status not in WAITABLE_STATUSES | {"completed"}:
            raise RuntimeError(
                f"source experiment {experiment!r} has unexpected status "
                f"{actual_status!r}"
            )
        adopted[experiment] = task_id
        predecessors[experiment] = predecessor

    if set(carried_recovery_target_observations) != set(
        source_recovery_target_adoptions
    ):
        raise RuntimeError(
            "source recovery target-adoption provenance is not closed by its steps"
        )

    recovery_target_observations = dict(carried_recovery_target_observations)
    source_task_ids = (
        set(rerun_task_ids.values())
        | set(adopted.values())
        | {
            str(observation["task_id"])
            for observation in transition_replaced_source_tasks.values()
        }
    )
    for experiment, task_id in recovery_target_adoptions.items():
        if experiment in rerun_set:
            raise RuntimeError(
                f"recovery-target experiment {experiment!r} is also listed for rerun"
            )
        step = source_steps.get(experiment)
        if not isinstance(step, Mapping):
            raise RuntimeError(
                f"recovery-target experiment {experiment!r} is missing from source"
            )
        replaced_source = transition_replaced_source_tasks.get(experiment)
        source_slot_is_pending = (
            step.get("state") == "pending" and step.get("task_id") is None
        )
        source_slot_is_replaced_failure = (
            isinstance(replaced_source, Mapping)
            and step.get("state") in {"created", "queued", "running", "failed"}
            and step.get("task_id") == replaced_source.get("task_id")
        )
        if not (source_slot_is_pending or source_slot_is_replaced_failure):
            raise RuntimeError(
                f"recovery-target experiment {experiment!r} source snapshot "
                "must be pending or the exact sealed failed transition source"
            )
        if experiment in adopted:
            raise RuntimeError(
                f"recovery-target experiment {experiment!r} already has a source task"
            )
        if task_id in source_task_ids:
            raise RuntimeError("recovery-target task reuses a source-chain task ID")
        task = task_class.get_task(task_id=task_id)
        _reload(task)
        task_status = _normalized_task_status(task)
        if task_status not in {"queued", "in_progress", "completed"}:
            raise RuntimeError(
                f"recovery-target experiment {experiment!r} task status is "
                f"not adoptable: {task_status!r}"
            )
        parent_id = _clearml_id(
            _task_parent(task),
            f"recovery-target {experiment} parent controller",
        )
        if parent_id in {source_id, controller_task_id}:
            raise RuntimeError(
                f"recovery-target experiment {experiment!r} has an invalid parent"
            )
        parent = task_class.get_task(task_id=parent_id)
        _reload(parent)
        parent_status = _normalized_task_status(parent)
        if parent_status != "failed":
            raise RuntimeError("recovery-target parent controller must be failed")
        if _task_parent(parent) != gate_task_id:
            raise RuntimeError("recovery-target parent controller gate parent mismatch")
        if _task_project_name(parent) != project or _task_project_name(task) != project:
            raise RuntimeError("recovery-target task project mismatch")
        parent_parameters = _task_parameters(parent)
        expected_parent_parameters = {
            "Args/recover_failed_controller_task_id": source_id,
            "Args/gate_task_id": gate_task_id,
            "Args/template_task_id": target_template_identity.get("task_id"),
            "Args/training_seed": str(training_seed),
            "Args/worker_queues": ",".join(worker_queues),
            "Args/max_parallel": str(max_parallel_training_tasks),
        }
        if source_revision_transition is not None:
            expected_parent_parameters.update(
                {
                    "Args/recovery_source_template_task_id": (
                        SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID
                    ),
                    "Args/recovery_source_transition": (SOURCE_REVISION_TRANSITION_ID),
                }
            )
        for key, expected in expected_parent_parameters.items():
            if str(parent_parameters.get(key) or "") != str(expected):
                raise RuntimeError(
                    f"recovery-target parent controller parameter {key!r} mismatch"
                )
        parent_rerun = _parameter_string_list(
            parent_parameters.get("Args/rerun_failed_experiment"),
            "recovery-target parent rerun list",
        )
        if parent_rerun != list(rerun):
            raise RuntimeError("recovery-target parent rerun list mismatch")
        parent_manual_adopt = _parameter_string_list(
            parent_parameters.get("Args/adopt_experiment", "[]"),
            "recovery-target parent manual-adopt list",
        )
        if parent_manual_adopt:
            raise RuntimeError("recovery-target parent used manual adoption")
        parent_script = _task_script(parent)
        if parent_script.get("entry_point") != "clearml_5090_training_controller.py":
            raise RuntimeError("recovery-target parent controller entrypoint mismatch")
        expected_name = _task_name(
            EXPERIMENT_ORDER.index(experiment) + 1,
            experiment,
            parent_id,
        )
        if str(getattr(task, "name", "") or "") != expected_name:
            raise RuntimeError(
                f"recovery-target experiment {experiment!r} task name mismatch"
            )
        parameters = _task_parameters(task)
        if parameters.get("Args/experiment_from_task") != experiment:
            raise RuntimeError(
                f"recovery-target experiment {experiment!r} parameter mismatch"
            )
        predecessor = _latest_completed_predecessor(
            steps,
            gate_task_id=gate_task_id,
            before_index=EXPERIMENT_ORDER.index(experiment) + 1,
        )
        if parameters.get("Args/predecessor_task_id") != predecessor:
            raise RuntimeError(
                f"recovery-target experiment {experiment!r} predecessor mismatch"
            )
        task_last_update = _task_last_update(
            task,
            context=f"recovery-target {experiment} task",
        )
        parent_last_update = _task_last_update(
            parent,
            context="recovery-target parent controller",
        )
        adopted[experiment] = task_id
        predecessors[experiment] = predecessor
        source_task_ids.add(task_id)
        recovery_target_observations[experiment] = {
            "task_id": task_id,
            "task_name": expected_name,
            "task_status": task_status,
            "task_last_update": task_last_update.isoformat(),
            "task_script_sha256": _script_sha256(_task_script(task)),
            "parent_controller_task_id": parent_id,
            "parent_controller_status": parent_status,
            "parent_controller_last_update": parent_last_update.isoformat(),
            "parent_controller_script_sha256": _script_sha256(parent_script),
            "source_snapshot_state": (
                str(step.get("state")) if source_slot_is_replaced_failure else "pending"
            ),
            "source_snapshot_task_id": (
                replaced_source.get("task_id")
                if source_slot_is_replaced_failure
                else None
            ),
            "predecessor_task_id": predecessor,
            "provenance": (
                "explicit_failed_source_revision_target_replacement"
                if source_slot_is_replaced_failure
                else "explicit_failed_recovery_target"
            ),
        }

    adopted_task_template_roles = {
        experiment: ("target" if experiment in recovery_target_adoptions else "source")
        for experiment in adopted
    }
    adopted_task_binding_receipts: dict[str, dict[str, object]] = {}
    target_template_predecessor_task_ids: dict[str, str] = {}
    if source_revision_transition is not None:
        source_subjects = set(SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS)
        target_subjects = set(SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS)
        if rerun_set & source_subjects:
            raise RuntimeError(
                "source-revision recovery cannot rerun any of the exact 21 old "
                "subjects under the new source"
            )
        if set(recovery_target_adoptions) - target_subjects:
            raise RuntimeError(
                "source-revision recovery target adoption is outside the exact "
                "five new subjects"
            )
        source_adopted = {
            experiment
            for experiment, role in adopted_task_template_roles.items()
            if role == "source"
        }
        if source_adopted != source_subjects:
            missing = sorted(source_subjects - source_adopted)
            extra = sorted(source_adopted - source_subjects)
            raise RuntimeError(
                "source-revision old-subject adoption split mismatch: "
                f"missing={missing}, extra={extra}"
            )
        for experiment in SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS:
            step = source_steps.get(experiment)
            if not isinstance(step, Mapping):
                raise RuntimeError(
                    f"source-revision target subject is missing: {experiment!r}"
                )
            has_target_adoption = experiment in recovery_target_adoptions
            has_target_rerun = experiment in rerun_set
            is_target_pending = (
                not has_target_adoption
                and not has_target_rerun
                and step.get("state") == "pending"
                and step.get("task_id") is None
            )
            if sum((has_target_adoption, has_target_rerun, is_target_pending)) != 1:
                raise RuntimeError(
                    f"source-revision target subject {experiment!r} does not have "
                    "exactly one new-template disposition"
                )
            frozen_predecessor = _latest_completed_predecessor(
                steps,
                gate_task_id=gate_task_id,
                before_index=EXPERIMENT_ORDER.index(experiment) + 1,
            )
            recorded_predecessor = predecessors.get(experiment)
            if (
                recorded_predecessor is not None
                and recorded_predecessor != frozen_predecessor
            ):
                raise RuntimeError(
                    f"source-revision target predecessor drifted for {experiment!r}"
                )
            predecessors[experiment] = frozen_predecessor
            target_template_predecessor_task_ids[experiment] = frozen_predecessor

        for experiment in EXPERIMENT_ORDER:
            if experiment not in adopted:
                continue
            task_id = adopted[experiment]
            task = task_class.get_task(task_id=task_id)
            _reload(task)
            role = adopted_task_template_roles[experiment]
            selected_identity = (
                source_template_identity
                if role == "source"
                else target_template_identity
            )
            selected_parameters = (
                source_template_parameters
                if role == "source"
                else target_template_parameters
            )
            legacy_exception = SOURCE_REVISION_LEGACY_TASK_ADOPTIONS.get(experiment)
            sealed_legacy_provenance: Mapping[str, object] | None = None
            if role == "source" and legacy_exception is not None:
                if (
                    legacy_exception["sealed_provenance_type"]
                    == "carried_recovery_target"
                ):
                    sealed_legacy_provenance = carried_recovery_target_observations.get(
                        experiment
                    )
                elif (
                    legacy_exception["sealed_provenance_type"]
                    == "completion_validation_retry"
                ):
                    sealed_legacy_provenance = (
                        source_recovery_completion_validation_retries.get(experiment)
                    )
            adopted_task_binding_receipts[experiment] = _validate_recovery_task_binding(
                task=task,
                task_id=task_id,
                experiment=experiment,
                predecessor_task_id=predecessors[experiment],
                template_role=role,
                template_identity=selected_identity,
                legacy_template_identity=(
                    source_legacy_template_identity if role == "source" else None
                ),
                template_parameters=selected_parameters,
                teacher_reference=teacher_reference,
                training_seed=training_seed,
                sealed_recovery_provenance=sealed_legacy_provenance,
                source_controller_task_id=source_id,
                source_progress_seal_sha256=_sha256(
                    source_progress.get("seal_sha256"),
                    "source controller progress seal",
                ),
                source_step=source_steps[experiment],
            )

    ordered_rerun = [
        experiment for experiment in EXPERIMENT_ORDER if experiment in rerun_set
    ]
    if set(rerun_task_ids) != rerun_set:
        raise RuntimeError("recovery rerun source-task inventory mismatch")
    source_recovery_chain = None
    if source_recovery is not None:
        source_recovery_chain = {
            "source_controller_task_id": source_id,
            "source_recovery_schema_version": source_recovery_schema_version,
            "source_recovery_sha256": hashlib.sha256(
                _canonical_json(source_recovery).encode("utf-8")
            ).hexdigest(),
            "source_progress_revision": revision,
            "source_progress_seal_sha256": _sha256(
                source_progress.get("seal_sha256"),
                "source controller progress seal",
            ),
            "carried_recovery_target_experiments": [
                experiment
                for experiment in EXPERIMENT_ORDER
                if experiment in carried_recovery_target_observations
            ],
            "recovered_pending_target_experiments": [
                experiment
                for experiment in EXPERIMENT_ORDER
                if experiment in recovered_pending_target_children
            ],
            "provenance": "sealed_failed_recovery_controller",
        }
    recovery = {
        "schema_version": RECOVERY_CONTRACT_SCHEMA_VERSION,
        "mode": "failed_controller_immutable_fork",
        "source_controller_task_id": source_id,
        "source_controller_status": "failed",
        "source_progress_revision": source_progress.get("revision"),
        "source_progress_seal_sha256": _sha256(
            source_progress.get("seal_sha256"),
            "source controller progress seal",
        ),
        "source_progress_artifact_readback": source_progress_artifact,
        "source_template_script_sha256": _sha256(
            validated_source_template.get("script_sha256"),
            "source template script",
        ),
        "target_template_script_sha256": _sha256(
            target_template_identity.get("script_sha256"),
            "target template script",
        ),
        "source_revision_transition": (
            dict(source_revision_transition)
            if source_revision_transition is not None
            else None
        ),
        "source_progress_template_equivalence": (source_progress_template_equivalence),
        "source_patch": "nested-teacher-config-consistency-v1",
        "source_recovery_chain": source_recovery_chain,
        "rerun_experiments": ordered_rerun,
        "rerun_source_task_ids": {
            experiment: rerun_task_ids[experiment] for experiment in ordered_rerun
        },
        "rerun_task_observations": {
            experiment: rerun_observations[experiment] for experiment in ordered_rerun
        },
        "rerun_predecessor_task_ids": {
            experiment: predecessors[experiment] for experiment in ordered_rerun
        },
        "completion_validation_retries": {
            experiment: completion_validation_retries[experiment]
            for experiment in EXPERIMENT_ORDER
            if experiment in completion_validation_retries
        },
        "recovered_pending_target_children": {
            experiment: recovered_pending_target_children[experiment]
            for experiment in EXPERIMENT_ORDER
            if experiment in recovered_pending_target_children
        },
        "transition_replaced_source_tasks": {
            experiment: transition_replaced_source_tasks[experiment]
            for experiment in SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
            if experiment in transition_replaced_source_tasks
        },
        "target_template_predecessor_task_ids": {
            experiment: target_template_predecessor_task_ids[experiment]
            for experiment in SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
            if experiment in target_template_predecessor_task_ids
        },
        "recovery_target_adoptions": {
            experiment: recovery_target_observations[experiment]
            for experiment in EXPERIMENT_ORDER
            if experiment in recovery_target_observations
        },
        "adopted_task_ids": {
            experiment: adopted[experiment]
            for experiment in EXPERIMENT_ORDER
            if experiment in adopted
        },
        "adopted_predecessor_task_ids": {
            experiment: predecessors[experiment]
            for experiment in EXPERIMENT_ORDER
            if experiment in adopted
        },
        "adopted_task_template_roles": {
            experiment: adopted_task_template_roles[experiment]
            for experiment in EXPERIMENT_ORDER
            if experiment in adopted_task_template_roles
        },
        "adopted_task_binding_receipts": {
            experiment: adopted_task_binding_receipts[experiment]
            for experiment in EXPERIMENT_ORDER
            if experiment in adopted_task_binding_receipts
        },
    }
    if sealed_recovery is not None:
        if sealed_recovery.get("schema_version") != RECOVERY_CONTRACT_SCHEMA_VERSION:
            raise RuntimeError("sealed target recovery contract version mismatch")

        def immutable_projection(
            value: Mapping[str, object],
        ) -> dict[str, object]:
            projected = dict(value)
            raw_adoptions = value.get("recovery_target_adoptions")
            if not isinstance(raw_adoptions, Mapping):
                raise RuntimeError(
                    "sealed target recovery adoption contract is invalid"
                )
            projected_adoptions: dict[str, object] = {}
            for experiment, raw_observation in raw_adoptions.items():
                if not isinstance(raw_observation, Mapping):
                    raise RuntimeError(
                        "sealed target recovery adoption observation is invalid"
                    )
                projected_adoptions[str(experiment)] = {
                    key: item
                    for key, item in raw_observation.items()
                    if key not in {"task_status", "task_last_update"}
                }
            projected["recovery_target_adoptions"] = projected_adoptions
            raw_pending_children = value.get("recovered_pending_target_children")
            if not isinstance(raw_pending_children, Mapping):
                raise RuntimeError(
                    "sealed target pending-child recovery contract is invalid"
                )
            projected_pending_children: dict[str, object] = {}
            for experiment, raw_observation in raw_pending_children.items():
                if not isinstance(raw_observation, Mapping):
                    raise RuntimeError(
                        "sealed target pending-child recovery observation is invalid"
                    )
                projected_pending_children[str(experiment)] = {
                    key: item
                    for key, item in raw_observation.items()
                    if key not in {"task_status", "task_last_update"}
                }
            projected["recovered_pending_target_children"] = projected_pending_children
            return projected

        if immutable_projection(sealed_recovery) != immutable_projection(recovery):
            raise RuntimeError("sealed target recovery contract drifted")
        recovery = dict(sealed_recovery)
    return recovery, adopted, predecessors


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


def _active_worker_queues(
    steps: Sequence[Mapping[str, object]],
) -> Counter[str]:
    active: Counter[str] = Counter()
    for step in steps:
        if step.get("state") in {"created", "queued", "running"}:
            queue = step.get("worker_queue")
            if type(queue) is str and queue.strip():
                active[queue.strip()] += 1
    return active


def _task_worker_queue(
    task: object,
    *,
    worker_queues: Sequence[str],
    allow_external: bool,
) -> tuple[str, dict[str, object]]:
    declared = list(dict.fromkeys(str(queue) for queue in worker_queues))
    data = getattr(task, "data", None)
    execution = getattr(data, "execution", None)
    queue_id = str(getattr(execution, "queue", "") or "")
    last_worker = str(getattr(data, "last_worker", "") or "")
    queue_name = ""
    getter = getattr(task, "get_executed_queue", None)
    if callable(getter):
        try:
            queue_name = str(getter(return_name=True) or "")
        except Exception as error:
            if not last_worker:
                raise RuntimeError(
                    "cannot resolve an active task execution queue"
                ) from error
    worker_matches = [
        queue
        for queue in declared
        if queue in last_worker or queue.replace("GPU4-", "") in last_worker
    ]
    if len(worker_matches) > 1:
        raise RuntimeError("active task last_worker matches multiple worker queues")
    worker_queue = worker_matches[0] if worker_matches else ""
    if queue_name in declared:
        if worker_queue and worker_queue != queue_name:
            raise RuntimeError("active task queue and last_worker disagree")
        resolved = queue_name
    elif queue_name:
        if worker_queue:
            raise RuntimeError(
                "active task queue is undeclared but last_worker is in-pool"
            )
        if not allow_external:
            raise RuntimeError(f"active task uses undeclared queue {queue_name!r}")
        resolved = "adopted-external"
    elif worker_queue:
        resolved = worker_queue
    elif allow_external:
        resolved = "adopted-external"
    else:
        raise RuntimeError("active task has no verifiable execution queue or worker")
    return resolved, {
        "execution_queue_id": queue_id or None,
        "execution_queue_name": queue_name or None,
        "last_worker": last_worker or None,
        "resolved_worker_queue": resolved,
    }


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
    training_seed: int,
    project: str,
    adopted: bool,
    worker_queues: Sequence[str],
) -> tuple[object, dict[str, object]]:
    expected_parameters = _experiment_parameters(
        template_parameters,
        experiment=experiment,
        predecessor_task_id=predecessor_task_id,
        teacher_task_id=teacher_task_id,
        teacher_model_id=teacher_model_id,
        teacher_checkpoint_sha256=teacher_sha,
        allow_failed_teacher_task=allow_failed_teacher_task,
        training_seed=training_seed,
    )
    task_name = _task_name(index, experiment, controller_task_id)
    task_id_value = step.get("task_id")
    if task_id_value is not None:
        task = task_class.get_task(
            task_id=_clearml_id(task_id_value, f"{experiment} task")
        )
        recovered_status = _normalized_task_status(task)
        if recovered_status in {"queued", "in_progress"}:
            recovered_queue, queue_observation = _task_worker_queue(
                task,
                worker_queues=worker_queues,
                allow_external=adopted,
            )
            if step.get("worker_queue") not in {None, "", recovered_queue}:
                raise RuntimeError(f"recovered task queue drifted for {experiment!r}")
            step["worker_queue"] = recovered_queue
            step["queue_observation"] = queue_observation
            _save_progress(controller_task, progress)
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
        recovered_queue = None
        queue_observation = None
        if recovered_status in {"queued", "in_progress"}:
            recovered_queue, queue_observation = _task_worker_queue(
                task,
                worker_queues=worker_queues,
                allow_external=adopted,
            )
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
                "worker_queue": recovered_queue,
                "queue_observation": queue_observation,
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
    _teacher_reference_request(args)
    worker_queues = _resolve_worker_queues(args)
    max_parallel = int(getattr(args, "max_parallel", 1) or 1)
    if max_parallel < 1:
        raise ValueError("max parallel must be positive")
    if max_parallel > len(worker_queues):
        raise ValueError("max parallel cannot exceed worker queue count")
    canary_first = bool(getattr(args, "canary_first", False))
    training_seed = getattr(args, "training_seed", DEFAULT_TRAINING_SEED)
    if type(training_seed) is not int or training_seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    adopt_map = _parse_adopt_experiments(getattr(args, "adopt_experiment", []) or [])
    rerun_experiments = _parse_rerun_experiments(
        getattr(args, "rerun_failed_experiment", []) or []
    )
    recovery_target_adoptions = _parse_recovery_target_adoptions(
        getattr(args, "recovery_adopt_target_experiment", []) or []
    )
    recovery_source = str(
        getattr(args, "recover_failed_controller_task_id", "") or ""
    ).strip()
    recovery_source_template_task_id = str(
        getattr(args, "recovery_source_template_task_id", "") or ""
    ).strip()
    source_transition_id = str(
        getattr(args, "recovery_source_transition", "") or ""
    ).strip()
    if bool(recovery_source_template_task_id) != bool(source_transition_id):
        raise ValueError(
            "--recovery-source-template-task-id and "
            "--recovery-source-transition must be provided together"
        )
    if recovery_source:
        if adopt_map:
            raise ValueError(
                "failed-controller recovery derives adoption from the sealed source; "
                "manual --adopt-experiment is forbidden"
            )
        _clearml_id(recovery_source, "failed source controller task")
    elif rerun_experiments:
        raise ValueError(
            "--rerun-failed-experiment requires --recover-failed-controller-task-id"
        )
    elif recovery_target_adoptions:
        raise ValueError(
            "--recovery-adopt-target-experiment requires "
            "--recover-failed-controller-task-id"
        )
    if recovery_source_template_task_id:
        if not recovery_source:
            raise ValueError(
                "source-revision transition requires "
                "--recover-failed-controller-task-id"
            )
        _clearml_id(
            recovery_source_template_task_id,
            "recovery source template task",
        )
        if source_transition_id != SOURCE_REVISION_TRANSITION_ID:
            raise ValueError("unsupported recovery source transition")
        if recovery_source != SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID:
            raise ValueError("source-revision transition controller task ID mismatch")
        if recovery_source_template_task_id != (
            SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID
        ):
            raise ValueError("source-revision source template task ID mismatch")
        if template_task_id != SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID:
            raise ValueError("source-revision target template task ID mismatch")
        if any(
            (
                str(getattr(args, "build_task_id", "") or "").strip(),
                int(getattr(args, "native_bundle_bytes", 0) or 0),
                str(getattr(args, "native_bundle_sha256", "") or "").strip(),
                str(getattr(args, "build_manifest_sha256", "") or "").strip(),
            )
        ):
            raise ValueError(
                "source-revision transition forbids native build/bundle overrides"
            )
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
    teacher_reference = _resolve_teacher_reference(
        args,
        task_class=task_class,
        sleeper=sleeper,
    )
    teacher_model_id = _clearml_id(teacher_reference.get("model_id"), "teacher model")
    teacher_sha = _sha256(
        teacher_reference.get("checkpoint_sha256"), "teacher checkpoint"
    )
    allow_failed_teacher_task = bool(teacher_reference.get("allow_failed_task", False))

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
    source_template_task: object | None = None
    source_template_identity: dict[str, object] | None = None
    source_legacy_template_identity: dict[str, object] | None = None
    source_template_parameters: dict[str, object] | None = None
    if recovery_source_template_task_id:
        source_template_task = task_class.get_task(
            task_id=recovery_source_template_task_id
        )
        source_template_identity = _template_identity(
            source_template_task,
            expected_task_id=recovery_source_template_task_id,
        )
        source_legacy_template_identity = _template_identity(
            source_template_task,
            expected_task_id=recovery_source_template_task_id,
            include_nested_teacher_fix=False,
        )
        source_template_parameters = dict(_task_parameters(source_template_task))
    elif recovery_source:
        source_template_task = template_task
        source_template_identity = _template_identity(
            template_task,
            expected_task_id=template_task_id,
            include_nested_teacher_fix=False,
        )
        source_template_parameters = dict(template_parameters)
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
    if recovery_source and not recovery_source_template_task_id:
        source_template_parameters = dict(template_parameters)
    template_identity = dict(template_identity)
    template_identity["source_parameters"] = _validate_source_parameters(
        template_parameters
    )
    template_identity["native_build_task_id"] = BUILD_TASK_ID
    if source_template_identity is not None:
        source_template_identity = dict(source_template_identity)
        if source_template_parameters is None:
            raise AssertionError("recovery source template parameters are missing")
        source_template_identity["source_parameters"] = _validate_source_parameters(
            source_template_parameters
        )
        source_template_identity["native_build_task_id"] = BUILD_TASK_ID
    if source_legacy_template_identity is not None:
        if source_template_parameters is None:
            raise AssertionError("recovery legacy template parameters are missing")
        source_legacy_template_identity = dict(source_legacy_template_identity)
        source_legacy_template_identity["source_parameters"] = (
            _validate_source_parameters(source_template_parameters)
        )
        source_legacy_template_identity["native_build_task_id"] = BUILD_TASK_ID
    source_revision_transition: dict[str, object] | None = None
    if recovery_source_template_task_id:
        if (
            source_template_identity is None
            or source_legacy_template_identity is None
            or source_template_parameters is None
        ):
            raise AssertionError("source-revision source template is missing")
        source_revision_transition = _build_source_revision_transition_contract(
            transition_id=source_transition_id,
            source_controller_task_id=recovery_source,
            source_template_identity=source_template_identity,
            source_legacy_template_identity=source_legacy_template_identity,
            target_template_identity=template_identity,
            source_template_parameters=source_template_parameters,
            target_template_parameters=template_parameters,
        )
    recovery: dict[str, object] | None = None
    recovery_predecessors: dict[str, str] = {}
    sealed_recovery: Mapping[str, object] | None = None
    controller_artifacts = getattr(controller_task, "artifacts", None)
    if (
        recovery_source
        and isinstance(controller_artifacts, Mapping)
        and PROGRESS_ARTIFACT in controller_artifacts
    ):
        existing_progress, _readback = _fresh_sealed_artifact_payload(
            controller_task,
            PROGRESS_ARTIFACT,
        )
        raw_recovery = existing_progress.get("recovery")
        if not isinstance(raw_recovery, Mapping):
            raise RuntimeError("existing recovery progress has no recovery contract")
        sealed_recovery = raw_recovery
    if recovery_source:
        if source_template_identity is None or source_template_parameters is None:
            raise AssertionError("recovery source template binding is missing")
        recovery, adopt_map, recovery_predecessors = (
            _resolve_failed_controller_recovery(
                task_class=task_class,
                controller_task_id=controller_task_id,
                source_controller_task_id=recovery_source,
                rerun_experiments=rerun_experiments,
                recovery_target_adoptions=recovery_target_adoptions,
                gate_task_id=gate_task_id,
                source_template_identity=source_template_identity,
                source_legacy_template_identity=source_legacy_template_identity,
                target_template_identity=template_identity,
                source_template_parameters=source_template_parameters,
                target_template_parameters=template_parameters,
                source_revision_transition=source_revision_transition,
                teacher_reference=teacher_reference,
                worker_queues=worker_queues,
                max_parallel_training_tasks=max_parallel,
                training_seed=training_seed,
                project=args.project,
                sealed_recovery=sealed_recovery,
            )
        )
    identity = {
        "controller_task_id": controller_task_id,
        "gate_task_id": gate_task_id,
        "template_identity": template_identity,
        "teacher_reference": teacher_reference,
        "worker_queues": worker_queues,
        "max_parallel_training_tasks": max_parallel,
        "training_seed": training_seed,
        "recovery": recovery,
    }
    progress = _load_or_create_progress(controller_task, **identity)
    steps = progress["steps"]
    if not isinstance(steps, list):
        raise RuntimeError("training progress steps must be a mutable list")

    adopted_template_roles: dict[str, str] = {}
    if recovery is not None:
        raw_roles = recovery.get("adopted_task_template_roles")
        if not isinstance(raw_roles, Mapping) or set(raw_roles) != set(adopt_map):
            raise RuntimeError("recovery adopted-task template roles are invalid")
        for experiment, raw_role in raw_roles.items():
            role = str(raw_role)
            if role not in {"source", "target"}:
                raise RuntimeError("recovery adopted-task template role is invalid")
            adopted_template_roles[str(experiment)] = role

    def template_binding(
        experiment: str,
        *,
        adopted: bool,
    ) -> tuple[object, Mapping[str, object], Mapping[str, object]]:
        role = adopted_template_roles.get(experiment, "target") if adopted else "target"
        if role == "source":
            if (
                source_template_task is None
                or source_template_identity is None
                or source_template_parameters is None
            ):
                raise RuntimeError("source-template recovery binding is unavailable")
            return (
                source_template_task,
                source_template_identity,
                source_template_parameters,
            )
        return template_task, template_identity, template_parameters

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
        predecessor_task_id = recovery_predecessors.get(
            experiment,
            _latest_completed_predecessor(
                steps,
                gate_task_id=gate_task_id,
                before_index=index,
            ),
        )
        step["task_id"] = task_id
        step["adopted"] = True
        step["predecessor_task_id"] = predecessor_task_id
        (
            adopted_template_task,
            adopted_template_identity,
            adopted_template_parameters,
        ) = template_binding(experiment, adopted=True)
        task, expected_parameters = _ensure_experiment_task(
            task_class=task_class,
            template_task=adopted_template_task,
            template_identity=adopted_template_identity,
            template_parameters=adopted_template_parameters,
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
            allow_failed_teacher_task=allow_failed_teacher_task,
            training_seed=training_seed,
            project=args.project,
            adopted=True,
            worker_queues=worker_queues,
        )
        step["task_name"] = str(getattr(task, "name", "") or step.get("task_name"))
        observed_parameters = _task_parameters(task)
        if not _execution_parameters_match(observed_parameters, expected_parameters):
            raise RuntimeError(
                f"adopted experiment {experiment!r} execution parameters drifted"
            )
        status = _normalized_task_status(task)
        if status == "completed":
            # Build expected params from the adopted task's own predecessor pin.
            pred = str(
                observed_parameters.get("Args/predecessor_task_id")
                or predecessor_task_id
            )
            step["predecessor_task_id"] = pred
            expected_parameters = _experiment_parameters(
                adopted_template_parameters,
                experiment=experiment,
                predecessor_task_id=pred,
                teacher_task_id=teacher_task_id,
                teacher_model_id=teacher_model_id,
                teacher_checkpoint_sha256=teacher_sha,
                allow_failed_teacher_task=bool(args.allow_failed_teacher_task),
                training_seed=training_seed,
            )
            _seal_completed_step(
                task=task,
                step=step,
                progress=progress,
                controller_task=controller_task,
                experiment=experiment,
                predecessor_task_id=pred,
                expected_parameters=expected_parameters,
                template_identity=adopted_template_identity,
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
            if status in {"queued", "in_progress"} and not step.get("worker_queue"):
                raise RuntimeError(
                    f"adopted experiment {experiment!r} has no worker queue evidence"
                )
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
                (
                    _completion_template_task,
                    completion_template_identity,
                    completion_template_parameters,
                ) = template_binding(
                    experiment,
                    adopted=bool(step.get("adopted")),
                )
                expected_parameters = _experiment_parameters(
                    completion_template_parameters,
                    experiment=experiment,
                    predecessor_task_id=predecessor_task_id,
                    teacher_task_id=teacher_task_id,
                    teacher_model_id=teacher_model_id,
                    teacher_checkpoint_sha256=teacher_sha,
                    allow_failed_teacher_task=allow_failed_teacher_task,
                    training_seed=training_seed,
                )
                result = _seal_completed_step(
                    task=task,
                    step=step,
                    progress=progress,
                    controller_task=controller_task,
                    experiment=experiment,
                    predecessor_task_id=predecessor_task_id,
                    expected_parameters=expected_parameters,
                    template_identity=completion_template_identity,
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

        canary_blocking = _canary_blocks_extra_slots(
            steps, canary_first=canary_first, task_class=task_class
        )
        effective_max = 1 if canary_blocking else max_parallel
        occupied_queues = _active_worker_queues(active_steps)
        free_queues: list[str] = []
        for queue in worker_queues:
            if occupied_queues[queue] > 0:
                occupied_queues[queue] -= 1
            elif queue != "adopted-external":
                free_queues.append(queue)
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
            if canary_blocking:
                break

        while pending_steps and pool_active < effective_max and free_queues:
            step = pending_steps[0]
            experiment = str(step["experiment"])
            index = int(step["index"])
            predecessor_task_id = recovery_predecessors.get(
                experiment,
                _latest_completed_predecessor(
                    steps,
                    gate_task_id=gate_task_id,
                    before_index=index,
                ),
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
                allow_failed_teacher_task=allow_failed_teacher_task,
                training_seed=training_seed,
                project=args.project,
                adopted=False,
                worker_queues=worker_queues,
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
            if status == "created":
                queue = free_queues.pop(0)
                _enqueue(task_class, task, worker_queue=queue)
                step["state"] = "queued"
            else:
                queue = str(step.get("worker_queue") or "")
                if queue not in free_queues:
                    raise RuntimeError(
                        f"recovered experiment {experiment!r} execution queue "
                        "exceeds declared capacity"
                    )
                free_queues.remove(queue)
                step["state"] = _state_from_status(status)
            step["worker_queue"] = queue
            step["predecessor_task_id"] = predecessor_task_id
            _save_progress(controller_task, progress)
            pool_active += 1
            pending_steps = [s for s in steps if s.get("state") == "pending"]
            progressed = True
            if canary_blocking:
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
                "predecessor_task_id": _clearml_id(
                    step.get("predecessor_task_id"),
                    f"{step['experiment']} training predecessor",
                ),
                **dict(result),
            }
        )

    formal_1337_manifest = build_formal_1337_training_manifest(ordered_results)
    summary = _sealed(
        {
            "schema_version": 1,
            "summary_type": "resilient_v2x_post_main_sequential_training",
            "status": "completed",
            "controller_task_id": controller_task_id,
            "gate_task_id": gate_task_id,
            "template": template_identity,
            "teacher": dict(teacher_reference),
            "worker_queue": worker_queues[0],
            "worker_queues": list(worker_queues),
            "experiment_order": list(EXPERIMENT_ORDER),
            "task_count": len(ordered_results),
            "max_parallel_training_tasks": max_parallel,
            "training_seed": training_seed,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "recovery": dict(recovery) if recovery is not None else None,
            "results": ordered_results,
            "formal_1337_evaluation_manifest": formal_1337_manifest,
            "formal_1337_manifest_artifact": FORMAL_1337_MANIFEST_ARTIFACT,
            "completed_at": _now(),
            "progress_artifact": PROGRESS_ARTIFACT,
        }
    )
    _upload_mapping(
        controller_task,
        FORMAL_1337_MANIFEST_ARTIFACT,
        formal_1337_manifest,
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
    "FORMAL_1337_MANIFEST_ARTIFACT",
    "FORMAL_1337_SUBJECT_ORDER",
    "PROGRESS_ARTIFACT",
    "SUMMARY_ARTIFACT",
    "build_formal_1337_training_manifest",
    "main",
    "run_training_suite",
)
