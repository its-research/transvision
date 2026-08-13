#!/usr/bin/env python3
"""Materialize and verify the sealed RTX 5090 runtime before training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Callable, Mapping, NamedTuple, Sequence
from urllib.parse import unquote, urlsplit


BASE_IMAGE_AMD64_MANIFEST_DIGEST = (
    "sha256:dbc586035fffb2bc030e807290d43e8d4edf44ee864fa5c832db44ed099fc415"
)
BASE_IMAGE_CONFIG_DIGEST = (
    "sha256:3812e520c0e86bb621878970370f52cbacaa32921bf0e4b2ae6a2028a5cf95fb"
)
BUILD_TASK_ID = "9055c0d3c4dd450c8a75dddfb21a56bd"
NATIVE_BUILD_SOURCE_DATASET_ID = "bcbd15ae7e454e9885bc4250a3de774e"
NATIVE_BUILD_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-cf61ef3c5432.tar.zst"
NATIVE_BUILD_SOURCE_ARCHIVE_BYTES = 1142675
NATIVE_BUILD_SOURCE_ARCHIVE_SHA256 = (
    "b20eccf308934eae82bf0bf032d0baa2c795d57efa6e55fe731501d50e5d7289"
)
NATIVE_COMPATIBLE_PYTHON_ONLY_CHANGES = (
    "configs/ffnet/config_basemodel_veh_only_complemented.py",
    "configs/ffnet/config_basemodel_official_3class_complemented.py",
    "configs/resilient_v2x/_base_/model.py",
    "configs/resilient_v2x/_base_/runtime.py",
    "configs/resilient_v2x/baselines/_base_.py",
    "configs/resilient_v2x/baselines/attfuse.py",
    "configs/resilient_v2x/baselines/ego_only.py",
    "configs/resilient_v2x/baselines/fcooper.py",
    "configs/resilient_v2x/baselines/disconet.py",
    "configs/resilient_v2x/baselines/how2comm.py",
    "configs/resilient_v2x/baselines/late_fusion.py",
    "configs/resilient_v2x/baselines/v2vnet.py",
    "configs/resilient_v2x/baselines/when2com.py",
    "configs/resilient_v2x/baselines/where2comm.py",
    "configs/resilient_v2x/dair_clean_teacher.py",
    "configs/resilient_v2x/dair_resilient_v2x.py",
    "configs/resilient_v2x/dair_vehicle_pretrain.py",
    "configs/resilient_v2x/improvements/support_residual.py",
    "configs/resilient_v2x/improvements/linear_no_distillation.py",
    "configs/resilient_v2x/improvements/linear_weak_feature_distillation.py",
    "configs/resilient_v2x/improvements/no_distillation_peak_lr_3e4.py",
    "configs/resilient_v2x/improvements/teacher_init_linear_no_distillation.py",
    "configs/resilient_v2x/improvements/weak_feature_distillation.py",
    "tests/resilient_v2x/test_clearml_1337_dependency_watcher.py",
    "tests/resilient_v2x/test_clearml_paper_controller.py",
    "tests/resilient_v2x/test_clearml_suite.py",
    "tests/resilient_v2x/test_clearml_training_controller.py",
    "tests/resilient_v2x/test_clearml_workflow.py",
    "tests/resilient_v2x/test_collect_ffnet_baseline.py",
    "tests/resilient_v2x/test_common_teacher_hook_synthetic.py",
    "tests/resilient_v2x/test_common_teacher_initialization.py",
    "tests/resilient_v2x/test_common_teacher_runtime.py",
    "tests/resilient_v2x/test_configs.py",
    "tests/resilient_v2x/test_controlled_baseline_detectors.py",
    "tests/resilient_v2x/test_controlled_baseline_matrix.py",
    "tests/resilient_v2x/test_controlled_baselines.py",
    "tests/resilient_v2x/test_detection_evaluation.py",
    "tests/resilient_v2x/test_disconet_controlled.py",
    "tests/resilient_v2x/test_feature_fusion.py",
    "tests/resilient_v2x/test_how2comm_controlled.py",
    "tests/resilient_v2x/test_late_fusion_controlled.py",
    "tests/resilient_v2x/test_overlay_cli.py",
    "tests/resilient_v2x/test_prepare_data.py",
    "tests/resilient_v2x/test_prepare_ffnet_baseline.py",
    "tests/resilient_v2x/test_repair_pipeline.py",
    "tests/resilient_v2x/test_reproduction_docs.py",
    "tests/resilient_v2x/test_resilient_v2x_metric.py",
    "tests/resilient_v2x/test_routing.py",
    "tests/resilient_v2x/test_schedule.py",
    "tests/resilient_v2x/test_v2vnet_controlled.py",
    "tests/resilient_v2x/test_when2com_controlled.py",
    "tests/resilient_v2x/test_where2comm_controlled.py",
    "tools/resilient_v2x/build_overlays.py",
    "tools/resilient_v2x/clearml_1337_dependency_watcher.py",
    "tools/resilient_v2x/clearml_5090_bootstrap.py",
    "tools/resilient_v2x/clearml_5090_build.py",
    "tools/resilient_v2x/clearml_5090_paper_controller.py",
    "tools/resilient_v2x/clearml_5090_training_controller.py",
    "tools/resilient_v2x/clearml_train.py",
    "tools/resilient_v2x/collect_ffnet_baseline.py",
    "tools/resilient_v2x/evaluate_controlled_baselines.py",
    "tools/resilient_v2x/prepare_ffnet_baseline.py",
    "tools/resilient_v2x/prepare_data.py",
    "transvision/dataset/__init__.py",
    "transvision/dataset/resilient_v2x_schedule.py",
    "transvision/dataset/v2x_dataset.py",
    "transvision/evaluation/metrics/resilient_v2x_metric.py",
    "transvision/evaluation/resilient_v2x_detection.py",
    "transvision/models/common_teacher_initialization.py",
    "transvision/models/data_preprocessors/data_preprocessor.py",
    "transvision/models/detectors/__init__.py",
    "transvision/models/detectors/controlled_v2x_baseline.py",
    "transvision/models/detectors/resilient_v2x.py",
    "transvision/models/detectors/v2x_voxelnet.py",
    "transvision/models/hooks/__init__.py",
    "transvision/models/hooks/common_teacher_initialization.py",
    "transvision/models/resilient_v2x/baseline_inputs.py",
    "transvision/models/resilient_v2x/baselines.py",
    "transvision/models/resilient_v2x/causal_repair.py",
    "transvision/models/resilient_v2x/disconet_baseline.py",
    "transvision/models/resilient_v2x/fusion.py",
    "transvision/models/resilient_v2x/how2comm_baseline.py",
    "transvision/models/resilient_v2x/late_fusion_baseline.py",
    "transvision/models/resilient_v2x/routing.py",
    "transvision/models/resilient_v2x/v2vnet_baseline.py",
    "transvision/models/resilient_v2x/when2com_baseline.py",
    "transvision/models/resilient_v2x/where2comm_baseline.py",
    "transvision/models/voxel_encoders/__init__.py",
    "transvision/models/voxel_encoders/ffnet_legacy_pillar_encoder.py",
    "transvision/register.py",
)
NATIVE_BUILD_INPUT_SHA256 = {
    "setup.py": "904623e7d97254aca735a78a6c200dc609187d1ad1505c4b972855aa6215f9b3",
    "transvision/models/bev_pool/__init__.py": "3be2a83be8ea38b65417ac35b4d377f914c3d2edc087758ab7dcfefe2c1e9a2e",
    "transvision/models/bev_pool/bev_pool.py": "70b229d501b8d991de030d9629590794257ef312a72a9750114a528cf018919f",
    "transvision/models/bev_pool/src/bev_pool.cpp": "4a7e86d4109017b6b2ce0481cabdfe8e987f8b4ff3293e71ebfcfbee51d8ba5c",
    "transvision/models/bev_pool/src/bev_pool_cuda.cu": "56983ddb7edf2077aca5887b63a68cf9b4828a7caae3ac9b0c14fe1b045f6cdb",
    "transvision/models/voxel/__init__.py": "daa16f1184d7368c0f5744f1c23e80cb7b0c25e25ec159d773b51c071a4e211a",
    "transvision/models/voxel/scatter_points.py": "284c0f55bd1deb79c1d35e0d4a74b5e02ca9f19615f8eb2f33c30fe25d4c8449",
    "transvision/models/voxel/src/scatter_points_cpu.cpp": "77943c3a33938fce32171cabafb4311a4dfbf71a7e19940b1113fb11e8769e40",
    "transvision/models/voxel/src/scatter_points_cuda.cu": "c9d8fcfc175adc223faa34017873934fbd8f7744a0242ee8104a91dbe22cf29b",
    "transvision/models/voxel/src/voxelization.cpp": "d66e0b9d3a86c2d19364144e0ac0ab799cbf62cc4f18959c9f2aaf7bd4a0d00c",
    "transvision/models/voxel/src/voxelization.h": "69762ef078fac20e33ab93ca292f93d3e239a452218c825c1e50fcca84bd4003",
    "transvision/models/voxel/src/voxelization_cpu.cpp": "5ea70fa45c47f9ca2c2fa4abf4efc2f7be03fd4b63473c898d07bb94fc266255",
    "transvision/models/voxel/src/voxelization_cuda.cu": "fef995a3b331a51ed70bf2fb797a0aaeae1be96cf8c1b164a2dc6e1e412e0cd6",
    "transvision/models/voxel/voxelize.py": "e817555e4bc1656192c3a2ad8258ca677e7568b40d80c294432afe18e9f32534",
}
NATIVE_BUNDLE_ARTIFACT = "rtx5090_native_bundle"
BUILD_MANIFEST_ARTIFACT = "rtx5090_build_manifest"
PIP_FREEZE_ARTIFACT = "rtx5090_pip_freeze"
EXPECTED_ARTIFACTS = frozenset(
    {
        NATIVE_BUNDLE_ARTIFACT,
        BUILD_MANIFEST_ARTIFACT,
        PIP_FREEZE_ARTIFACT,
    }
)
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_GPU_COUNT = 4
EXPECTED_CAPABILITY = (12, 0)
# Sealed 5090 plus capacity-matched A100/V100 ablation workers.
ALLOWED_GPU_CAPABILITIES = frozenset({(12, 0), (8, 0), (7, 0)})
GPU_MEMORY_PREFLIGHT_MAX_USED_MIB = 1024
EXPECTED_PACKAGES = {
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
    "mmdet": "3.2.0",
    "mmdet3d": "1.3.0",
}
CUSTOM_OP_MODULES = (
    "transvision.models.voxel.voxel_layer",
    "transvision.models.bev_pool.bev_pool_ext",
)
WORKSPACE = Path("/workspace/resilient-v2x-5090-runtime")
VENV_ROOT = Path("/opt/resilient-v2x-5090")
BUILD_ONLY_ENVIRONMENT_KEYS = frozenset(
    {
        "CC",
        "CXX",
        "CPATH",
        "CUDA_HOME",
        "CUDACXX",
        "FORCE_CUDA",
        "LIBRARY_PATH",
        "MAX_JOBS",
        "MMCV_WITH_OPS",
        "TORCH_CUDA_ARCH_LIST",
    }
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CLEARML_TASK_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
CLEAN_TEACHER_MODEL_NAME = "ResilientV2X clean teacher"
DISTILLED_STUDENT_MODEL_NAME = "ResilientV2X distilled student"
DURABLE_MODEL_SCHEMES = frozenset({"http", "https", "s3", "gs", "azure"})
EXPECTED_FILES_SERVER_HOST = "10.100.34.118"
EXPECTED_FILES_SERVER_PORT = 8081
FILES_SERVER_URI = f"http://{EXPECTED_FILES_SERVER_HOST}:{EXPECTED_FILES_SERVER_PORT}"
EXPERIMENT_MAX_EPOCHS = 50
RTX5090_TRAIN_BATCH_SIZE_PER_GPU = 2
RTX5090_EVAL_BATCH_SIZE_PER_GPU = 4
CONTROLLED_BASELINE_EVALUATOR_SHA256 = (
    "68b4f6d0282f364fe33bf6493308ca73b044c2977f6468b5c79ba4866c7547fd"
)
CONTROLLED_BASELINE_LEGACY_HEADLESS_EVALUATOR_SHA256 = (
    "3b12fa2039e4f1e6d00924eace9f640ef3a2eb8c88ddfe48fb62b9dd5a58bb90"
)
CONTROLLED_BASELINE_HEADLESS_EVALUATOR_SHA256 = (
    "720855b259bc489359fb9c2b77c1290570acf9d9fb9ede3f3e982e5fce68cd50"
)
CONTROLLED_BASELINE_LEGACY_PROTOCOL_EVALUATOR_SHA256 = (
    "ef2818ede1a4e5fc152fc41aea6d320999b11ecf8f4e3329c808b109d4ecd807"
)
CONTROLLED_BASELINE_PROTOCOL_EVALUATOR_SHA256 = (
    "d233e054f2b608bc441de25833fb89d117197d841752812a0054e0514995ad36"
)
CANONICAL_1337_PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
DEFAULT_TRAINING_SEED = 20250218
TRAINING_OVERLAY_PROTOCOL_SEED = 20250218
CANONICAL_1337_SAMPLE_COUNT = 1337
CANONICAL_1337_GROUND_TRUTH_COUNT = 11330
CANONICAL_1337_MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
CANONICAL_1337_OVERLAY_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
CANONICAL_1337_SAMPLE_IDS_SHA256 = (
    "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
)
CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES = 256 * 1024 * 1024
CONTROLLED_EVIDENCE_REQUIRED_FILENAMES = (
    "resolved_config.py",
    "checkpoint.sha256",
    "predictions.json",
)
CONTROLLED_EVIDENCE_FORBIDDEN_STALE_HASH_FIELDS = frozenset(
    {
        "resolved_config_sha256",
        "evaluation_plan_sha256",
        "evaluation_plan_content_sha256",
        "plan_sha256",
        "plan_content_sha256",
    }
)
CLEARML_TRAIN_BASELINE_SHA256 = (
    "6ea88906c9a9c8bd5122d23f78306edcbb92ada629e0189df6e6817824c2f631"
)
CLEARML_TRAIN_METRICS_COMPAT_SHA256 = (
    "52cb5c508000ab96333b6e7ce5ba490587f144f3b97ea148974be87716a629a5"
)
CLEARML_TRAIN_COMPLEMENTED_BASELINE_SHA256 = (
    "d3edcccbf5c05779e8382270bbb271448de982b1153be8d3bd68ef9f50c9eff7"
)
CLEARML_TRAIN_COMPLEMENTED_METRICS_COMPAT_SHA256 = (
    "8435208ec277f84e3403fb6991fa9f3be920dfac3c5ab4f0fd2963a31bd29314"
)
CLEARML_TRAIN_FFNET_STAGE_SHA256 = (
    "67e1f6701f5e4ebee7d27fceabc22e0ec1c52db97f0cb58a091df8a0dfd36cfb"
)
CLEARML_TRAIN_FFNET_OFFICIAL_BASELINE_SHA256 = (
    "c338ed1afd32b23e1b2dc8a4e270994217e3913f7185de00354890f548e65c07"
)
CLEARML_TRAIN_FFNET_OFFICIAL_METRICS_COMPAT_SHA256 = (
    "44b64c59cfd55279efc6a65524e0200ec8ac38522953d1bdb621b704ffd84723"
)
CLEARML_TRAIN_PROTOCOL_SCHEMA_BASELINE_SHA256 = (
    "9958b56108f9f44f226bda72e13e521417e22d8dbc9a1d41ebd4ef97c5b02bd6"
)
CLEARML_TRAIN_PROTOCOL_SCHEMA_SHA256 = (
    "c804413a9734d9a798e1b71b797d089431b5f6f64596bd6add0a4515df220fab"
)
CLEARML_TRAIN_METRICS_COMPATIBILITY_IDENTITIES = {
    CLEARML_TRAIN_BASELINE_SHA256: CLEARML_TRAIN_METRICS_COMPAT_SHA256,
    CLEARML_TRAIN_COMPLEMENTED_BASELINE_SHA256: (
        CLEARML_TRAIN_COMPLEMENTED_METRICS_COMPAT_SHA256
    ),
    CLEARML_TRAIN_FFNET_OFFICIAL_BASELINE_SHA256: (
        CLEARML_TRAIN_FFNET_OFFICIAL_METRICS_COMPAT_SHA256
    ),
    CLEARML_TRAIN_PROTOCOL_SCHEMA_BASELINE_SHA256: (
        CLEARML_TRAIN_PROTOCOL_SCHEMA_SHA256
    ),
}
CLEARML_TRAIN_METRICS_REPLACEMENTS = (
    (
        """    candidates = sorted(work_dir.rglob("scalars.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one scalars.json under {work_dir}, found {len(candidates)}"
        )
""",
        """    candidates = sorted(work_dir.rglob("scalars.json"))
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
""",
    ),
    (
        """    metric_rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        scalars.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
""",
        """    metric_rows: list[dict[str, object]] = []
    metric_content = scalars.read_text(encoding="utf-8")
    # MMEngine may emit JSONL, a single JSON object, or concatenated objects
    # without newlines; parse with raw_decode across the whole payload.
    decoder = json.JSONDecoder()
    offset = 0
    length = len(metric_content)
    while offset < length:
        while offset < length and metric_content[offset].isspace():
            offset += 1
        if offset >= length:
            break
        try:
            row, end = decoder.raw_decode(metric_content, offset)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in {scalars} ({metric_source}) at offset {offset}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSON payload in {scalars} at offset {offset}")
        metrics = {
            key: value for key, value in row.items() if key.startswith("resilient_v2x/")
        }
        if metrics:
            metric_rows.append(metrics)
        offset = end
""",
    ),
)


class ExperimentSpec(NamedTuple):
    """One immutable member of the post-main training suite."""

    name: str
    kind: str
    config: str | None
    requires_teacher: bool


class ControlledEvidenceMemberReceipt(NamedTuple):
    """Immutable identity and content receipt for one staged evidence file."""

    relative_path: str
    size_bytes: int
    sha256: str
    identity: tuple[int, ...]


class ControlledEvidenceStage(NamedTuple):
    """Private exact-inventory snapshot uploaded as formal evaluation evidence."""

    root: Path
    root_identity: tuple[int, ...]
    members: tuple[ControlledEvidenceMemberReceipt, ...]


CORE_EXPERIMENT_SPECS = (
    ExperimentSpec(
        "support_residual",
        "improvement",
        "configs/resilient_v2x/improvements/support_residual.py",
        True,
    ),
    ExperimentSpec(
        "ptf_linear",
        "ablation",
        "configs/resilient_v2x/ablations/ptf_linear.py",
        True,
    ),
    ExperimentSpec(
        "ptf_none",
        "ablation",
        "configs/resilient_v2x/ablations/ptf_none.py",
        True,
    ),
    ExperimentSpec(
        "router_static",
        "ablation",
        "configs/resilient_v2x/ablations/router_static.py",
        True,
    ),
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
        "no_distillation",
        "ablation",
        "configs/resilient_v2x/ablations/no_distillation.py",
        True,
    ),
    ExperimentSpec(
        "concat_capacity_matched",
        "ablation",
        "configs/resilient_v2x/ablations/concat_capacity_matched.py",
        True,
    ),
    ExperimentSpec("v2x_vit", "baseline", None, True),
    ExperimentSpec("cobevt", "baseline", None, True),
    ExperimentSpec("coformernet", "baseline", None, True),
    ExperimentSpec("bevfusion", "baseline", None, True),
    ExperimentSpec("ffnet", "baseline", None, True),
)
ADDITIONAL_EXPERIMENT_SPECS = (
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
EXPERIMENT_SPECS = CORE_EXPERIMENT_SPECS + ADDITIONAL_EXPERIMENT_SPECS
CORE_EXPERIMENT_ORDER = tuple(spec.name for spec in CORE_EXPERIMENT_SPECS)
EXPERIMENT_ORDER = tuple(spec.name for spec in EXPERIMENT_SPECS)
EXPERIMENT_BY_NAME = {spec.name: spec for spec in EXPERIMENT_SPECS}
TEACHER_DEPENDENT_EXPERIMENTS = frozenset(
    spec.name for spec in EXPERIMENT_SPECS if spec.requires_teacher
)
BASELINE_EXPERIMENTS = frozenset(
    spec.name for spec in EXPERIMENT_SPECS if spec.kind == "baseline"
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
RTX5090_VAL_INTERVAL = 10
RTX5090_HEADLESS_CFG_OPTIONS = (
    "visualizer._scope_=mmengine",
    "visualizer.type=Visualizer",
    "visualizer.vis_backends.0._scope_=mmengine",
)
EXPERIMENT_CHECKPOINT_CFG_OPTIONS = (
    "default_hooks.checkpoint.interval=10",
    "default_hooks.checkpoint.max_keep_ckpts=5",
)
COMMON_TEACHER_INITIALIZATION_CONTRACT = "shared-only-clean-teacher-initialization-v1"
COMMON_TEACHER_INITIALIZATION_POLICY = "shared-only"
COMMON_TEACHER_INITIALIZATION_PREFIXES = (
    "lidar_encoder.",
    "camera_encoder.",
    "bbox_head.",
    "detection_projection.",
)
COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME = (
    "common_teacher_initialization_audit.json"
)
COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT = "common_teacher_initialization_audit"
COMMON_TEACHER_SOURCE_KEYS = 617
COMMON_TEACHER_SOURCE_NUMEL = 35_811_485
COMMON_TEACHER_SOURCE_BYTES = 143_246_244
COMMON_TEACHER_SHARED_KEYS = 468
COMMON_TEACHER_SHARED_NUMEL = 31_506_934
COMMON_TEACHER_SHARED_BYTES = 126_028_040
COMMON_TEACHER_FUSION_KEYS = 149
EMPTY_TENSOR_MAPPING_SHA256 = hashlib.sha256(b"").hexdigest()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _sha256_argument(value: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-id", required=True)
    parser.add_argument("--source-archive-name", required=True)
    parser.add_argument("--source-archive-bytes", type=_positive_integer, required=True)
    parser.add_argument("--source-archive-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--training-dataset-id", required=True)
    parser.add_argument("--native-bundle-bytes", type=_positive_integer, required=True)
    parser.add_argument("--native-bundle-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--build-manifest-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--gpus", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "ffnet",
            "ffnet_official_eval",
            "ffnet_official_train",
            "vehicle",
            "vehicle_teacher",
            "teacher",
            "student",
            "validate",
            "baseline_validate",
        ),
        default="all",
    )
    parser.add_argument("--max-epochs", type=_positive_integer, default=50)
    parser.add_argument(
        "--training-seed",
        type=_nonnegative_integer,
        default=DEFAULT_TRAINING_SEED,
        help=(
            "model, sampler, and dataset-augmentation seed; does not alter "
            "the sealed training-overlay protocol seed"
        ),
    )
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument(
        "--experiment-from-task",
        choices=EXPERIMENT_ORDER,
        help=(
            "run exactly one sealed post-main experiment inside the current "
            "ClearML task; absent means use the original --stage workflow"
        ),
    )
    parser.add_argument(
        "--teacher-task-id",
        help=(
            "task owning the clean-teacher OutputModel; used by remote "
            "student/validation stages or teacher-dependent experiments"
        ),
    )
    parser.add_argument(
        "--teacher-model-id",
        help="exact clean-teacher OutputModel ID for a remote checkpoint handoff",
    )
    parser.add_argument(
        "--teacher-checkpoint-sha256",
        type=_sha256_argument,
        help="expected SHA-256 of the remotely handed-off teacher checkpoint",
    )
    parser.add_argument(
        "--allow-failed-teacher-task",
        action="store_true",
        help=(
            "allow a failed task only when its sealed teacher model, 50-epoch "
            "run contract, model ID, storage URL, and checkpoint SHA all verify"
        ),
    )
    parser.add_argument(
        "--student-task-id",
        help=("completed task owning the distilled-student OutputModel for validation"),
    )
    parser.add_argument(
        "--student-model-id",
        help="exact distilled-student OutputModel ID for validation handoff",
    )
    parser.add_argument(
        "--student-checkpoint-sha256",
        type=_sha256_argument,
        help="expected SHA-256 of the remotely handed-off student checkpoint",
    )
    parser.add_argument(
        "--predecessor-task-id",
        help="completed task that must precede this experiment in the linear suite",
    )
    parser.add_argument(
        "--controlled-baseline",
        choices=EXPERIMENT_ORDER,
        help="controlled baseline evaluated by the baseline_validate stage",
    )
    parser.add_argument(
        "--controlled-baseline-task-id",
        help=("completed task owning the controlled-baseline final OutputModel"),
    )
    parser.add_argument(
        "--controlled-baseline-model-id",
        help="exact final OutputModel ID owned by the controlled-baseline task",
    )
    parser.add_argument(
        "--controlled-baseline-checkpoint-sha256",
        type=_sha256_argument,
        help="expected SHA-256 of the controlled-baseline final checkpoint",
    )
    parser.add_argument("--amp", action="store_true")
    return parser


def _validate_remote_checkpoint_handoff(
    args: argparse.Namespace,
    *,
    kind: str,
) -> bool:
    task_id = getattr(args, f"{kind}_task_id")
    model_id = getattr(args, f"{kind}_model_id")
    checkpoint_sha256 = getattr(args, f"{kind}_checkpoint_sha256")
    values = (task_id, model_id, checkpoint_sha256)
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise ValueError(
            f"remote {kind} handoff requires task ID, model ID, and checkpoint SHA"
        )
    if task_id is None:
        return False
    if CLEARML_TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise ValueError(f"--{kind}-task-id must be a lowercase 32-hex ID")
    if CLEARML_TASK_ID_PATTERN.fullmatch(model_id) is None:
        raise ValueError(f"--{kind}-model-id must be a lowercase 32-hex ID")
    if getattr(args, f"{kind}_checkpoint") is not None:
        raise ValueError(
            f"--{kind}-checkpoint and --{kind}-task-id are mutually exclusive"
        )
    return True


def _validate_arguments(args: argparse.Namespace) -> None:
    if type(args.training_seed) is not int or args.training_seed < 0:
        raise ValueError("--training-seed must be a non-negative integer")
    for field in ("source_dataset_id", "training_dataset_id"):
        value = getattr(args, field)
        if type(value) is not str or not value.strip():
            raise ValueError(f"--{field.replace('_', '-')} must be non-empty")
    archive_name = Path(args.source_archive_name)
    if archive_name.name != args.source_archive_name or archive_name.is_absolute():
        raise ValueError("--source-archive-name must be a plain filename")
    if args.amp:
        raise ValueError("RTX5090 first-run profile requires FP32; --amp is forbidden")

    experiment_name = getattr(args, "experiment_from_task", None)
    if experiment_name is None:
        baseline = getattr(args, "controlled_baseline", None)
        baseline_task_id = getattr(args, "controlled_baseline_task_id", None)
        baseline_model_id = getattr(args, "controlled_baseline_model_id", None)
        baseline_checkpoint_sha256 = getattr(
            args,
            "controlled_baseline_checkpoint_sha256",
            None,
        )
        if args.stage == "baseline_validate":
            if any(
                value is None
                for value in (
                    baseline,
                    baseline_task_id,
                    baseline_model_id,
                    baseline_checkpoint_sha256,
                )
            ):
                raise ValueError(
                    "baseline_validate requires --controlled-baseline, "
                    "--controlled-baseline-task-id, "
                    "--controlled-baseline-model-id, and "
                    "--controlled-baseline-checkpoint-sha256"
                )
            if CLEARML_TASK_ID_PATTERN.fullmatch(baseline_task_id) is None:
                raise ValueError(
                    "--controlled-baseline-task-id must be a lowercase 32-hex ID"
                )
            if CLEARML_TASK_ID_PATTERN.fullmatch(baseline_model_id) is None:
                raise ValueError(
                    "--controlled-baseline-model-id must be a lowercase 32-hex ID"
                )
            predecessor_task_id = getattr(args, "predecessor_task_id", None)
            if (
                predecessor_task_id is None
                or CLEARML_TASK_ID_PATTERN.fullmatch(predecessor_task_id) is None
            ):
                raise ValueError("baseline_validate requires --predecessor-task-id")
            remote_teacher = _validate_remote_checkpoint_handoff(args, kind="teacher")
            remote_student = _validate_remote_checkpoint_handoff(args, kind="student")
            if (
                remote_teacher
                or remote_student
                or args.teacher_checkpoint is not None
                or args.student_checkpoint is not None
                or args.allow_failed_teacher_task
            ):
                raise ValueError(
                    "baseline_validate forbids teacher/student checkpoint handoffs"
                )
            return
        if any(
            value is not None
            for value in (
                baseline,
                baseline_task_id,
                baseline_model_id,
                baseline_checkpoint_sha256,
            )
        ):
            raise ValueError(
                "--controlled-baseline options require stage baseline_validate"
            )
        if getattr(args, "predecessor_task_id", None) is not None:
            raise ValueError(
                "--predecessor-task-id is valid only with --experiment-from-task"
            )
        remote_teacher = _validate_remote_checkpoint_handoff(args, kind="teacher")
        remote_student = _validate_remote_checkpoint_handoff(args, kind="student")
        has_teacher = remote_teacher or args.teacher_checkpoint is not None
        has_student = remote_student or args.student_checkpoint is not None

        if args.allow_failed_teacher_task and not remote_teacher:
            raise ValueError(
                "--allow-failed-teacher-task requires a complete remote teacher handoff"
            )
        if args.stage in (
            "all",
            "ffnet",
            "ffnet_official_eval",
            "ffnet_official_train",
            "vehicle",
            "vehicle_teacher",
            "teacher",
        ):
            if has_teacher or has_student or args.allow_failed_teacher_task:
                raise ValueError(f"stage {args.stage!r} forbids checkpoint handoff")
        elif args.stage == "student":
            if not has_teacher:
                raise ValueError("student stage requires a teacher checkpoint handoff")
            if has_student:
                raise ValueError("student stage forbids a student checkpoint handoff")
        elif args.stage == "validate":
            if not has_teacher or not has_student:
                raise ValueError(
                    "validate stage requires teacher and student checkpoint handoffs"
                )
        return

    if args.stage != "all":
        raise ValueError(
            "--experiment-from-task is isolated from the original --stage workflow"
        )
    if args.teacher_checkpoint is not None or args.student_checkpoint is not None:
        raise ValueError(
            "experiment tasks accept checkpoint handoff only through OutputModel"
        )
    if any(
        value is not None
        for value in (
            args.student_task_id,
            args.student_model_id,
            args.student_checkpoint_sha256,
        )
    ):
        raise ValueError("experiment tasks forbid student checkpoint handoff")
    if args.max_epochs != EXPERIMENT_MAX_EPOCHS:
        raise ValueError(
            f"the fixed experiment suite requires {EXPERIMENT_MAX_EPOCHS} epochs"
        )
    if (
        getattr(args, "controlled_baseline", None) is not None
        or getattr(args, "controlled_baseline_task_id", None) is not None
        or getattr(args, "controlled_baseline_model_id", None) is not None
        or getattr(args, "controlled_baseline_checkpoint_sha256", None) is not None
    ):
        raise ValueError(
            "controlled-baseline options cannot be combined with an experiment"
        )

    spec = EXPERIMENT_BY_NAME[experiment_name]
    teacher_task_id = getattr(args, "teacher_task_id", None)
    predecessor_task_id = getattr(args, "predecessor_task_id", None)
    if (
        type(predecessor_task_id) is not str
        or CLEARML_TASK_ID_PATTERN.fullmatch(predecessor_task_id) is None
    ):
        raise ValueError(
            "experiment tasks require a lowercase 32-hex --predecessor-task-id"
        )
    if spec.requires_teacher:
        if (
            type(teacher_task_id) is not str
            or CLEARML_TASK_ID_PATTERN.fullmatch(teacher_task_id) is None
        ):
            raise ValueError(
                "teacher-dependent experiments require a lowercase 32-hex "
                "--teacher-task-id"
            )
        optional_pin_values = (
            args.teacher_model_id,
            args.teacher_checkpoint_sha256,
        )
        if any(value is not None for value in optional_pin_values) and not all(
            value is not None for value in optional_pin_values
        ):
            raise ValueError(
                "teacher model ID and checkpoint SHA must be provided together"
            )
        if (
            args.teacher_model_id is not None
            and CLEARML_TASK_ID_PATTERN.fullmatch(args.teacher_model_id) is None
        ):
            raise ValueError("--teacher-model-id must be a lowercase 32-hex ID")
        if args.allow_failed_teacher_task and not all(optional_pin_values):
            raise ValueError(
                "failed teacher reuse requires a pinned model ID and checkpoint SHA"
            )
    elif (
        any(
            value is not None
            for value in (
                teacher_task_id,
                args.teacher_model_id,
                args.teacher_checkpoint_sha256,
            )
        )
        or args.allow_failed_teacher_task
    ):
        raise ValueError(
            f"experiment {experiment_name!r} forbids teacher checkpoint handoff"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
) -> Path:
    if path.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"file size mismatch: {path}")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"file SHA-256 mismatch: {path}")
    return path


def _validate_native_build_inputs(source_root: Path) -> None:
    for relative_path, expected_sha256 in NATIVE_BUILD_INPUT_SHA256.items():
        _verify_file(
            source_root.joinpath(*PurePosixPath(relative_path).parts),
            expected_sha256=expected_sha256,
        )


def _require_new_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing path: {path}")
    path.mkdir(parents=True)
    return path


def _validated_member_path(
    member: tarfile.TarInfo,
    destination: Path,
    observed: set[str],
) -> Path:
    name = PurePosixPath(member.name)
    if name.is_absolute() or not name.parts or ".." in name.parts:
        raise ValueError(f"unsafe archive member path: {member.name!r}")
    canonical_name = name.as_posix()
    if canonical_name in observed:
        raise ValueError(f"duplicate archive member: {member.name!r}")
    observed.add(canonical_name)
    if not (member.isfile() or member.isdir()):
        raise ValueError(
            f"archive links and special files are forbidden: {member.name!r}"
        )
    target = destination.joinpath(*name.parts).resolve(strict=False)
    try:
        target.relative_to(destination.resolve(strict=True))
    except ValueError as error:
        raise ValueError(
            f"archive member escapes destination: {member.name!r}"
        ) from error
    return target


def _safe_extract_tar(archive: Path, destination: Path) -> Path:
    destination = _require_new_directory(destination)
    observed: set[str] = set()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        validated = [
            (member, _validated_member_path(member, destination, observed))
            for member in members
        ]
        for member, target in validated:
            _extract_member(bundle, member, target)
    return destination


def _safe_extract_zstd(archive: Path, destination: Path) -> Path:
    import zstandard

    destination = _require_new_directory(destination)
    observed: set[str] = set()
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as bundle:
                for member in bundle:
                    target = _validated_member_path(member, destination, observed)
                    _extract_member(bundle, member, target)
    return destination


def _apply_source_runner_metrics_compatibility(source_root: Path) -> Path:
    """Patch the sealed runner only when its complete baseline identity matches."""

    source_root = source_root.resolve(strict=True)
    target = source_root / "tools/resilient_v2x/clearml_train.py"
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"source training runner is not a regular file: {target}")
    target = target.resolve(strict=True)
    try:
        target.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"source training runner escaped source root: {target}"
        ) from error

    original = target.read_bytes()
    actual_sha256 = hashlib.sha256(original).hexdigest()
    compatible_identities = frozenset(
        (
            *CLEARML_TRAIN_METRICS_COMPATIBILITY_IDENTITIES.values(),
            CLEARML_TRAIN_FFNET_STAGE_SHA256,
        )
    )
    if actual_sha256 in compatible_identities:
        return target
    expected_patched_sha256 = CLEARML_TRAIN_METRICS_COMPATIBILITY_IDENTITIES.get(
        actual_sha256
    )
    if expected_patched_sha256 is None:
        raise ValueError(
            "source training runner identity does not match the sealed baseline: "
            f"{actual_sha256}"
        )

    patched = original.decode("utf-8")
    for old, new in CLEARML_TRAIN_METRICS_REPLACEMENTS:
        if patched.count(old) != 1:
            raise ValueError(
                "source training runner compatibility anchor is not unique"
            )
        patched = patched.replace(old, new)
    encoded = patched.encode("utf-8")
    patched_sha256 = hashlib.sha256(encoded).hexdigest()
    if patched_sha256 != expected_patched_sha256:
        raise ValueError(
            "source training runner compatibility result has an invalid identity: "
            f"{patched_sha256}"
        )

    temporary = target.with_name(f"{target.name}.metrics-compat.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(
            f"compatibility temporary path already exists: {temporary}"
        )
    mode = target.stat().st_mode & 0o777
    try:
        with temporary.open("xb") as output:
            output.write(encoded)
        temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return target


def _apply_controlled_evaluator_headless_compatibility(source_root: Path) -> Path:
    """Patch the sealed evaluator to the audited deployment-evaluation version."""

    source_root = source_root.resolve(strict=True)
    target = source_root / "tools/resilient_v2x/evaluate_controlled_baselines.py"
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"controlled evaluator is not a regular file: {target}")
    target = target.resolve(strict=True)
    try:
        target.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"controlled evaluator escaped source root: {target}"
        ) from error

    original = target.read_bytes()
    actual_sha256 = hashlib.sha256(original).hexdigest()
    if actual_sha256 in {
        CONTROLLED_BASELINE_HEADLESS_EVALUATOR_SHA256,
        CONTROLLED_BASELINE_LEGACY_PROTOCOL_EVALUATOR_SHA256,
        CONTROLLED_BASELINE_PROTOCOL_EVALUATOR_SHA256,
    }:
        return target
    if actual_sha256 not in {
        CONTROLLED_BASELINE_EVALUATOR_SHA256,
        CONTROLLED_BASELINE_LEGACY_HEADLESS_EVALUATOR_SHA256,
    }:
        raise ValueError(
            "controlled evaluator identity does not match the sealed baseline: "
            f"{actual_sha256}"
        )
    patched = original.decode("utf-8")
    if actual_sha256 == CONTROLLED_BASELINE_EVALUATOR_SHA256:
        headless_anchor = '    resolved["work_dir"] = str(output_dir)\n'
        headless_replacement = (
            headless_anchor
            + """    resolved["visualizer"] = {
        "_scope_": "mmengine",
        "type": "Visualizer",
        "name": "visualizer",
        "vis_backends": [
            {"_scope_": "mmengine", "type": "LocalVisBackend"}
        ],
    }
"""
        )
        if patched.count(headless_anchor) != 1:
            raise ValueError("controlled evaluator headless anchor is not unique")
        patched = patched.replace(headless_anchor, headless_replacement)

    deployment_replacements = (
        (
            'BASELINES = ("v2x_vit", "cobevt", "coformernet", "bevfusion", "ffnet")\n',
            """BASELINES = ("v2x_vit", "cobevt", "coformernet", "bevfusion", "ffnet")
ABLATIONS = (
    "ptf_none",
    "ptf_linear",
    "router_static",
    "router_uniform",
    "no_reliability",
    "no_delay_metadata",
    "no_distillation",
    "concat_capacity_matched",
)
IMPROVEMENTS = (
    "support_residual",
    "linear_no_distillation",
    "no_distillation_peak_lr_3e4",
)
EVALUATION_SUBJECTS = BASELINES + ABLATIONS + IMPROVEMENTS
""",
            "evaluation subjects",
        ),
        (
            """BASELINE_CONFIGS = {
    name: ROOT / "configs" / "resilient_v2x" / "baselines" / f"{name}.py"
    for name in BASELINES
}
""",
            """BASELINE_CONFIGS = {
    name: ROOT / "configs" / "resilient_v2x" / "baselines" / f"{name}.py"
    for name in BASELINES
}
ABLATION_CONFIGS = {
    name: ROOT / "configs" / "resilient_v2x" / "ablations" / f"{name}.py"
    for name in ABLATIONS
}
IMPROVEMENT_CONFIGS = {
    name: ROOT / "configs" / "resilient_v2x" / "improvements" / f"{name}.py"
    for name in IMPROVEMENTS
}
EVALUATION_CONFIGS = {**BASELINE_CONFIGS, **ABLATION_CONFIGS, **IMPROVEMENT_CONFIGS}
""",
            "evaluation configs",
        ),
        (
            """    if baseline not in BASELINES:
        raise ControlledBaselineEvaluationError(f"unsupported baseline: {baseline}")
""",
            """    if baseline not in EVALUATION_SUBJECTS:
        raise ControlledBaselineEvaluationError(
            f"unsupported evaluation subject: {baseline}"
        )
""",
            "subject validation",
        ),
        (
            """    baseline_config_path = BASELINE_CONFIGS[baseline]
    baseline_config = _load_python_config(baseline_config_path)
    baseline_config_digest = _sha256_file(
        baseline_config_path,
        "baseline config",
    )
    baseline_model = _mapping(baseline_config.get("model"), "baseline model")
    if baseline_model.get("type") != "ControlledCooperativeBaselineNet":
        raise ControlledBaselineEvaluationError(
            "baseline config does not use ControlledCooperativeBaselineNet"
        )
    if baseline_model.get("baseline_name") != baseline:
        raise ControlledBaselineEvaluationError("baseline config name mismatch")
""",
            """    baseline_config_path = EVALUATION_CONFIGS[baseline]
    baseline_config = _load_python_config(baseline_config_path)
    baseline_config_digest = _sha256_file(
        baseline_config_path,
        "baseline config",
    )
    baseline_model = _mapping(baseline_config.get("model"), "baseline model")
    if baseline in BASELINES:
        if baseline_model.get("type") != "ControlledCooperativeBaselineNet":
            raise ControlledBaselineEvaluationError(
                "baseline config does not use ControlledCooperativeBaselineNet"
            )
        if baseline_model.get("baseline_name") != baseline:
            raise ControlledBaselineEvaluationError("baseline config name mismatch")
    else:
        if baseline_model.get("type") != "ResilientV2XNet":
            raise ControlledBaselineEvaluationError(
                "controlled-method config does not use ResilientV2XNet"
            )
        # The teacher is a training-only branch.  Deployment evaluation loads
        # only the student weights from the controlled-method checkpoint.
        baseline_model.pop("teacher", None)
        baseline_model.pop("teacher_checkpoint", None)
        baseline_model.pop("distillation", None)
""",
            "deployment model",
        ),
        (
            '        "baseline_config_sha256": baseline_config_digest,\n',
            """        "baseline_config_sha256": baseline_config_digest,
        "evaluation_subject_type": (
            "baseline"
            if baseline in BASELINES
            else ("ablation" if baseline in ABLATIONS else "improvement")
        ),
""",
            "subject provenance",
        ),
        (
            '    parser.add_argument("--baseline", required=True, choices=BASELINES)\n',
            (
                '    parser.add_argument("--baseline", required=True, '
                "choices=EVALUATION_SUBJECTS)\n"
            ),
            "subject CLI",
        ),
        (
            '    "DELAYS",\n    "ControlledBaselineEvaluationError",\n',
            """    "DELAYS",
    "ABLATIONS",
    "ABLATION_CONFIGS",
    "IMPROVEMENTS",
    "IMPROVEMENT_CONFIGS",
    "EVALUATION_SUBJECTS",
    "ControlledBaselineEvaluationError",
""",
            "public exports",
        ),
    )
    for old, new, context in deployment_replacements:
        if patched.count(old) != 1:
            raise ValueError(f"controlled evaluator {context} anchor is not unique")
        patched = patched.replace(old, new)
    encoded = patched.encode("utf-8")
    patched_sha256 = hashlib.sha256(encoded).hexdigest()
    if patched_sha256 != CONTROLLED_BASELINE_HEADLESS_EVALUATOR_SHA256:
        raise ValueError(
            "controlled evaluator deployment result has an invalid identity: "
            f"{patched_sha256}"
        )

    temporary = target.with_name(f"{target.name}.deployment-compat.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"deployment temporary path already exists: {temporary}")
    mode = target.stat().st_mode & 0o777
    try:
        with temporary.open("xb") as output:
            output.write(encoded)
        temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return target


def _extract_member(
    bundle: tarfile.TarFile,
    member: tarfile.TarInfo,
    target: Path,
) -> None:
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite archive target: {target}")
    source = bundle.extractfile(member)
    if source is None:
        raise ValueError(f"regular archive member has no payload: {member.name!r}")
    with source, target.open("xb") as output:
        shutil.copyfileobj(source, output)
    target.chmod(member.mode & 0o777)


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        check=True,
        text=True,
        capture_output=capture,
    )


def _capture_gpu_runtime() -> dict[str, object]:
    import torch

    return {
        "python": list(sys.version_info[:2]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "torch_arch_list": list(torch.cuda.get_arch_list()),
        "cuda_available": torch.cuda.is_available(),
    }


def _nvidia_csv_rows(text: str, *, columns: int, context: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        values = [value.strip() for value in raw_line.split(",")]
        if len(values) != columns or any(not value for value in values):
            raise RuntimeError(
                f"{context} row {line_number} has an invalid column inventory"
            )
        rows.append(values)
    return rows


def _visible_gpu_rows(
    rows: list[dict[str, object]],
    *,
    expected_gpu_count: int,
    environ: Mapping[str, str],
) -> list[dict[str, object]]:
    raw_visible = (
        environ.get("CUDA_VISIBLE_DEVICES")
        or environ.get("NVIDIA_VISIBLE_DEVICES")
        or ""
    ).strip()
    if not raw_visible or raw_visible.lower() == "all":
        if len(rows) != expected_gpu_count:
            raise RuntimeError(
                "nvidia-smi visible GPU count is ambiguous without an exact "
                "CUDA/NVIDIA_VISIBLE_DEVICES binding"
            )
        return rows
    tokens = [token.strip() for token in raw_visible.split(",")]
    if (
        len(tokens) != expected_gpu_count
        or any(not token for token in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        raise RuntimeError("visible GPU binding does not match the requested GPU count")
    by_index = {str(row["index"]): row for row in rows}
    by_uuid = {str(row["uuid"]): row for row in rows}
    selected: list[dict[str, object]] = []
    for token in tokens:
        row = by_index.get(token) or by_uuid.get(token)
        if row is None:
            raise RuntimeError("visible GPU binding is absent from nvidia-smi")
        selected.append(row)
    return selected


def _capture_gpu_memory_preflight(
    expected_gpu_count: int,
    *,
    environ: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
) -> dict[str, object]:
    """Fail before importing Torch if the assigned GPUs are not effectively idle."""

    if expected_gpu_count not in {4, 8}:
        raise RuntimeError("GPU memory preflight requires exactly 4 or 8 GPUs")
    gpu_result = runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    )
    parsed_rows = _nvidia_csv_rows(
        gpu_result.stdout,
        columns=5,
        context="nvidia-smi GPU inventory",
    )
    rows: list[dict[str, object]] = []
    for values in parsed_rows:
        try:
            index, total_mib, used_mib, free_mib = (
                int(values[0]),
                int(values[2]),
                int(values[3]),
                int(values[4]),
            )
        except ValueError as error:
            raise RuntimeError("nvidia-smi GPU memory values are invalid") from error
        if (
            index < 0
            or total_mib <= 0
            or used_mib < 0
            or free_mib < 0
            or used_mib > total_mib
            or free_mib > total_mib
        ):
            raise RuntimeError("nvidia-smi GPU memory values are out of range")
        rows.append(
            {
                "index": index,
                "uuid": values[1],
                "total_mib": total_mib,
                "used_mib": used_mib,
                "free_mib": free_mib,
            }
        )
    if not rows or len({row["index"] for row in rows}) != len(rows) or len(
        {row["uuid"] for row in rows}
    ) != len(rows):
        raise RuntimeError("nvidia-smi GPU inventory is empty or duplicated")
    selected = _visible_gpu_rows(
        rows,
        expected_gpu_count=expected_gpu_count,
        environ=os.environ if environ is None else environ,
    )
    selected_uuids = {str(row["uuid"]) for row in selected}
    process_result = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    )
    processes: list[dict[str, object]] = []
    for values in _nvidia_csv_rows(
        process_result.stdout,
        columns=3,
        context="nvidia-smi compute process inventory",
    ):
        try:
            pid = int(values[1])
            used_mib = int(values[2])
        except ValueError as error:
            raise RuntimeError("nvidia-smi compute process values are invalid") from error
        if pid <= 0 or used_mib < 0:
            raise RuntimeError("nvidia-smi compute process values are out of range")
        if values[0] in selected_uuids:
            processes.append(
                {"gpu_uuid": values[0], "pid": pid, "used_mib": used_mib}
            )
    if processes:
        raise RuntimeError("assigned GPUs already have compute processes")
    busy = [
        row
        for row in selected
        if int(row["used_mib"]) > GPU_MEMORY_PREFLIGHT_MAX_USED_MIB
        or int(row["free_mib"])
        < int(row["total_mib"]) - GPU_MEMORY_PREFLIGHT_MAX_USED_MIB
    ]
    if busy:
        raise RuntimeError("assigned GPUs exceed the idle-memory preflight threshold")
    return {
        "event": "gpu_memory_preflight_pass",
        "policy": "no_compute_process_and_at_most_1024_MiB_used_per_gpu",
        "gpu_count": len(selected),
        "max_used_mib": GPU_MEMORY_PREFLIGHT_MAX_USED_MIB,
        "gpus": selected,
        "compute_process_count": 0,
    }


def _validate_gpu_runtime(contract: Mapping[str, object]) -> None:
    gpu_count = contract.get("gpu_count")
    if gpu_count not in {4, 8}:
        raise RuntimeError(
            f"RTX5090 runtime gpu_count mismatch: expected 4 or 8, got {gpu_count!r}"
        )
    expected = {
        "python": [3, 12],
        "torch": EXPECTED_TORCH,
        "torch_cuda": EXPECTED_TORCH_CUDA,
        "cuda_available": True,
    }
    for field, expected_value in expected.items():
        if contract.get(field) != expected_value:
            raise RuntimeError(
                f"RTX5090 runtime {field} mismatch: "
                f"expected {expected_value!r}, got {contract.get(field)!r}"
            )
    capabilities = contract.get("capabilities")
    if not isinstance(capabilities, list) or len(capabilities) != int(gpu_count):
        raise RuntimeError(f"RTX5090 GPU capability mismatch: {capabilities!r}")
    normalized = []
    for item in capabilities:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError(f"RTX5090 GPU capability mismatch: {capabilities!r}")
        capability = (int(item[0]), int(item[1]))
        if capability not in ALLOWED_GPU_CAPABILITIES:
            raise RuntimeError(f"RTX5090 GPU capability mismatch: {capabilities!r}")
        normalized.append(capability)
    if len(set(normalized)) != 1:
        raise RuntimeError(
            f"RTX5090 GPU capabilities must be homogeneous: {capabilities!r}"
        )
    arch_list = contract.get("torch_arch_list")
    if not isinstance(arch_list, list) or "sm_120" not in arch_list:
        raise RuntimeError("PyTorch build does not contain sm_120")


def _assert_base_image() -> None:
    actual = os.environ.get("RESILIENT_V2X_CONTAINER_IMAGE_DIGEST")
    if actual != BASE_IMAGE_AMD64_MANIFEST_DIGEST:
        raise RuntimeError(
            "RTX5090 base image digest mismatch: "
            f"expected {BASE_IMAGE_AMD64_MANIFEST_DIGEST!r}, got {actual!r}"
        )


def _artifact_path(artifact: object, name: str) -> Path:
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        raise RuntimeError(f"ClearML artifact {name!r} cannot be downloaded")
    value = getter()
    if not value:
        raise RuntimeError(f"ClearML artifact {name!r} returned no local copy")
    return Path(value).resolve(strict=True)


def _require_completed_build_task(task: object) -> Mapping[str, object]:
    status = getattr(task, "status", None)
    if callable(status):
        status = status()
    if status is None:
        get_status = getattr(task, "get_status", None)
        if callable(get_status):
            status = get_status()
    status_value = getattr(status, "value", status)
    normalized_status = str(status_value).rsplit(".", 1)[-1].lower()
    if normalized_status != "completed":
        raise RuntimeError(f"native build task is not completed: {status_value!r}")
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("native build task has no artifact mapping")
    missing = EXPECTED_ARTIFACTS - set(artifacts)
    if missing:
        raise RuntimeError(f"native build task is missing artifacts: {sorted(missing)}")
    return artifacts


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _required_mapping(
    value: object,
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _require_exact_audit_object(
    value: object,
    *,
    keys: set[str],
    context: str,
) -> Mapping[str, object]:
    mapping = _required_mapping(value, context)
    observed = set(mapping)
    missing = sorted(keys - observed)
    extra = sorted(observed - keys)
    if missing or extra:
        raise ValueError(f"{context} schema mismatch: missing={missing}, extra={extra}")
    return mapping


def _require_audit_literals(
    mapping: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    context: str,
) -> None:
    for key, expected_value in expected.items():
        observed = mapping[key]
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise ValueError(
                f"{context}.{key} mismatch: "
                f"expected {expected_value!r}, got {observed!r}"
            )


def _require_positive_audit_integer(
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> int:
    value = mapping[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return value


def _require_nonnegative_audit_integer(
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> int:
    value = mapping[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"{context}.{key} must be a non-negative integer")
    return value


def _require_audit_sha256(
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str:
    value = mapping[key]
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context}.{key} must be a lowercase SHA-256")
    return value


def _validate_common_teacher_initialization_audit(
    audit_path: Path,
    *,
    expected_teacher_sha256: str,
    expect_nested_teacher: bool,
) -> Path:
    if (
        type(expected_teacher_sha256) is not str
        or SHA256_PATTERN.fullmatch(expected_teacher_sha256) is None
    ):
        raise ValueError("expected teacher checkpoint SHA-256 is invalid")
    if type(expect_nested_teacher) is not bool:
        raise TypeError("expect_nested_teacher must be a boolean")
    if audit_path.is_symlink():
        raise ValueError(
            f"common teacher initialization audit must not be a symlink: {audit_path}"
        )
    try:
        resolved_audit = audit_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"common teacher initialization audit is missing: {audit_path}"
        ) from error
    if not resolved_audit.is_file():
        raise ValueError(
            "common teacher initialization audit must be a regular file: "
            f"{resolved_audit}"
        )
    if resolved_audit.stat().st_size <= 0:
        raise ValueError(
            f"common teacher initialization audit is empty: {resolved_audit}"
        )

    audit = _read_json_object(resolved_audit)
    audit = _require_exact_audit_object(
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
    _require_audit_literals(
        audit,
        {
            "schema_version": 1,
            "contract": COMMON_TEACHER_INITIALIZATION_CONTRACT,
            "result": "pass",
        },
        context="common teacher initialization audit",
    )

    checkpoint = _require_exact_audit_object(
        audit["checkpoint"],
        keys={"path", "filename", "size_bytes", "sha256", "expected_sha256"},
        context="common teacher initialization audit.checkpoint",
    )
    checkpoint_path = checkpoint["path"]
    checkpoint_filename = checkpoint["filename"]
    if (
        type(checkpoint_path) is not str
        or not checkpoint_path
        or not Path(checkpoint_path).is_absolute()
    ):
        raise ValueError(
            "common teacher initialization audit.checkpoint.path "
            "must be a non-empty absolute path"
        )
    if (
        type(checkpoint_filename) is not str
        or not checkpoint_filename
        or Path(checkpoint_path).name != checkpoint_filename
    ):
        raise ValueError(
            "common teacher initialization audit.checkpoint.filename "
            "does not match checkpoint.path"
        )
    _require_positive_audit_integer(
        checkpoint,
        "size_bytes",
        context="common teacher initialization audit.checkpoint",
    )
    checkpoint_sha256 = _require_audit_sha256(
        checkpoint,
        "sha256",
        context="common teacher initialization audit.checkpoint",
    )
    expected_checkpoint_sha256 = _require_audit_sha256(
        checkpoint,
        "expected_sha256",
        context="common teacher initialization audit.checkpoint",
    )
    if (
        checkpoint_sha256 != expected_teacher_sha256
        or expected_checkpoint_sha256 != expected_teacher_sha256
    ):
        raise ValueError(
            "common teacher initialization audit checkpoint SHA-256 "
            "does not match the clean-teacher handoff"
        )

    source = _require_exact_audit_object(
        audit["source"],
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
    _require_audit_literals(
        source,
        {
            "keys": COMMON_TEACHER_SOURCE_KEYS,
            "numel": COMMON_TEACHER_SOURCE_NUMEL,
            "bytes": COMMON_TEACHER_SOURCE_BYTES,
            "expected_keys": COMMON_TEACHER_SOURCE_KEYS,
            "common_keys": COMMON_TEACHER_SHARED_KEYS,
            "expected_common_keys": COMMON_TEACHER_SHARED_KEYS,
            "fusion_keys": COMMON_TEACHER_FUSION_KEYS,
            "expected_fusion_keys": COMMON_TEACHER_FUSION_KEYS,
        },
        context="common teacher initialization audit.source",
    )
    _require_audit_sha256(
        source,
        "state_sha256",
        context="common teacher initialization audit.source",
    )

    shared = _require_exact_audit_object(
        audit["shared_initialization"],
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
    _require_audit_literals(
        shared,
        {
            "prefixes": list(COMMON_TEACHER_INITIALIZATION_PREFIXES),
            "keys": COMMON_TEACHER_SHARED_KEYS,
            "numel": COMMON_TEACHER_SHARED_NUMEL,
            "bytes": COMMON_TEACHER_SHARED_BYTES,
            "expected_keys": COMMON_TEACHER_SHARED_KEYS,
            "shape_dtype_verified": True,
            "exact_tensor_equality_verified": True,
        },
        context="common teacher initialization audit.shared_initialization",
    )
    _require_audit_sha256(
        shared,
        "state_sha256",
        context="common teacher initialization audit.shared_initialization",
    )

    fusion = _require_exact_audit_object(
        audit["method_specific_fusion"],
        keys={
            "keys",
            "numel",
            "bytes",
            "sha256_before",
            "sha256_after",
            "unchanged",
        },
        context="common teacher initialization audit.method_specific_fusion",
    )
    fusion_keys = _require_nonnegative_audit_integer(
        fusion,
        "keys",
        context="common teacher initialization audit.method_specific_fusion",
    )
    fusion_numel = _require_nonnegative_audit_integer(
        fusion,
        "numel",
        context="common teacher initialization audit.method_specific_fusion",
    )
    fusion_bytes = _require_nonnegative_audit_integer(
        fusion,
        "bytes",
        context="common teacher initialization audit.method_specific_fusion",
    )
    fusion_sha256_before = _require_audit_sha256(
        fusion,
        "sha256_before",
        context="common teacher initialization audit.method_specific_fusion",
    )
    fusion_sha256_after = _require_audit_sha256(
        fusion,
        "sha256_after",
        context="common teacher initialization audit.method_specific_fusion",
    )
    _require_audit_literals(
        fusion,
        {"unchanged": True},
        context="common teacher initialization audit.method_specific_fusion",
    )
    if fusion_sha256_before != fusion_sha256_after:
        raise ValueError(
            "common teacher initialization audit method-specific fusion changed"
        )
    if fusion_keys == 0:
        if fusion_numel != 0 or fusion_bytes != 0:
            raise ValueError(
                "zero-state method-specific fusion must have keys=numel=bytes=0"
            )
        if fusion_sha256_before != EMPTY_TENSOR_MAPPING_SHA256:
            raise ValueError(
                "zero-state method-specific fusion must use the canonical "
                "empty-mapping SHA-256"
            )
    elif fusion_numel == 0 or fusion_bytes == 0:
        raise ValueError(
            "non-empty method-specific fusion must have positive numel and bytes"
        )
    elif fusion_sha256_before == EMPTY_TENSOR_MAPPING_SHA256:
        raise ValueError(
            "non-empty method-specific fusion cannot use the empty-mapping SHA-256"
        )

    target = _require_exact_audit_object(
        audit["target"],
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
    if type(target["model_type"]) is not str or not target["model_type"]:
        raise ValueError(
            "common teacher initialization audit.target.model_type "
            "must be a non-empty string"
        )
    target_key_count = _require_positive_audit_integer(
        target,
        "target_key_count",
        context="common teacher initialization audit.target",
    )
    _require_audit_literals(
        target,
        {
            "target_common_key_count": COMMON_TEACHER_SHARED_KEYS,
            "target_fusion_key_count": fusion_keys,
            "nested_teacher_present": expect_nested_teacher,
            "nested_teacher_key_count": (
                COMMON_TEACHER_SOURCE_KEYS if expect_nested_teacher else 0
            ),
            "nested_teacher_full_equality_verified": expect_nested_teacher,
        },
        context="common teacher initialization audit.target",
    )
    expected_target_keys = (
        COMMON_TEACHER_SHARED_KEYS
        + fusion_keys
        + (COMMON_TEACHER_SOURCE_KEYS if expect_nested_teacher else 0)
    )
    if target_key_count != expected_target_keys:
        raise ValueError(
            "common teacher initialization audit.target.target_key_count "
            f"mismatch: expected {expected_target_keys}, got {target_key_count}"
        )
    return resolved_audit


def _validate_and_upload_common_teacher_initialization_audit(
    task: object,
    *,
    work_dir: Path,
    expected_teacher_sha256: str,
    expect_nested_teacher: bool,
) -> Path:
    audit_path = _validate_common_teacher_initialization_audit(
        work_dir / COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME,
        expected_teacher_sha256=expected_teacher_sha256,
        expect_nested_teacher=expect_nested_teacher,
    )
    upload_artifact = getattr(task, "upload_artifact", None)
    flush = getattr(task, "flush", None)
    if not callable(upload_artifact) or not callable(flush):
        raise RuntimeError("ClearML task cannot publish the initialization audit")
    if not upload_artifact(
        COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT,
        artifact_object=str(audit_path),
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to upload the common teacher initialization audit")
    flush(wait_for_uploads=True)
    return audit_path


def _validate_build_manifest(
    manifest: Mapping[str, object],
    args: argparse.Namespace,
) -> None:
    expected_base_image = {
        "platform": "linux/amd64",
        "manifest_digest": BASE_IMAGE_AMD64_MANIFEST_DIGEST,
        "config_digest": BASE_IMAGE_CONFIG_DIGEST,
    }
    if manifest.get("base_image") != expected_base_image:
        raise ValueError("native build manifest base image contract mismatch")
    expected_scalars = {
        "source_dataset_id": NATIVE_BUILD_SOURCE_DATASET_ID,
        "source_archive_name": NATIVE_BUILD_SOURCE_ARCHIVE_NAME,
        "source_archive_bytes": NATIVE_BUILD_SOURCE_ARCHIVE_BYTES,
        "source_archive_sha256": NATIVE_BUILD_SOURCE_ARCHIVE_SHA256,
        "amp": False,
    }
    for field, expected_value in expected_scalars.items():
        if manifest.get(field) != expected_value:
            raise ValueError(
                f"native build manifest {field} mismatch: "
                f"expected {expected_value!r}, got {manifest.get(field)!r}"
            )

    bundle = _required_mapping(manifest.get("native_bundle"), "native_bundle")
    if bundle.get("bytes") != args.native_bundle_bytes:
        raise ValueError("native bundle byte count does not match the task contract")
    if bundle.get("sha256") != args.native_bundle_sha256:
        raise ValueError("native bundle SHA-256 does not match the task contract")

    overlay = _required_mapping(manifest.get("python_overlay"), "python_overlay")
    if overlay.get("archive_root") != "site-packages":
        raise ValueError("native build overlay root must be site-packages")
    for field in ("file_count", "total_bytes"):
        value = overlay.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"python_overlay.{field} must be a positive integer")

    runtime = _required_mapping(manifest.get("runtime"), "runtime")
    runtime_devices = runtime.get("devices")
    _validate_gpu_runtime(
        {
            "python": [
                int(part) for part in str(runtime.get("python", "")).split(".")[:2]
            ],
            "torch": runtime.get("torch"),
            "torch_cuda": runtime.get("torch_cuda"),
            "gpu_count": len(runtime_devices)
            if isinstance(runtime_devices, list)
            else None,
            "capabilities": [device.get("capability") for device in runtime_devices]
            if isinstance(runtime_devices, list)
            and all(isinstance(device, Mapping) for device in runtime_devices)
            else None,
            "torch_arch_list": runtime.get("torch_arch_list"),
            "cuda_available": True,
        }
    )

    wheel = _required_mapping(manifest.get("mmcv_wheel"), "mmcv_wheel")
    if (
        type(wheel.get("name")) is not str
        or type(wheel.get("bytes")) is not int
        or wheel["bytes"] <= 0
        or type(wheel.get("sha256")) is not str
        or SHA256_PATTERN.fullmatch(wheel["sha256"]) is None
    ):
        raise ValueError("native build manifest has an invalid MMCV wheel record")

    extensions = manifest.get("extensions")
    if not isinstance(extensions, list) or len(extensions) != 2:
        raise ValueError("native build manifest must contain exactly two extensions")
    for record in extensions:
        if not isinstance(record, Mapping):
            raise ValueError("native extension record must be an object")
        path = PurePosixPath(str(record.get("path", "")))
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".so"
            or path.parts[:2] != ("transvision", "models")
            or type(record.get("bytes")) is not int
            or record["bytes"] <= 0
            or type(record.get("sha256")) is not str
            or SHA256_PATTERN.fullmatch(record["sha256"]) is None
        ):
            raise ValueError(f"invalid native extension record: {record!r}")


def _tree_stats(root: Path) -> tuple[int, int]:
    root = root.resolve(strict=True)
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"runtime overlay contains a symlink: {path}")
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
        elif not path.is_dir():
            raise ValueError(f"runtime overlay contains a special file: {path}")
    return file_count, total_bytes


def _verify_bundle_layout(
    bundle_root: Path,
    manifest: Mapping[str, object],
) -> tuple[Path, Path, list[tuple[Path, Mapping[str, object]]]]:
    overlay = _required_mapping(manifest["python_overlay"], "python_overlay")
    overlay_root = (bundle_root / "site-packages").resolve(strict=True)
    if not overlay_root.is_dir():
        raise ValueError("native bundle is missing site-packages")
    stats = _tree_stats(overlay_root)
    if stats != (overlay["file_count"], overlay["total_bytes"]):
        raise ValueError(
            "site-packages overlay inventory mismatch: "
            f"expected {(overlay['file_count'], overlay['total_bytes'])!r}, got {stats!r}"
        )

    wheel = _required_mapping(manifest["mmcv_wheel"], "mmcv_wheel")
    wheel_path = _verify_file(
        bundle_root / "wheels" / str(wheel["name"]),
        expected_bytes=int(wheel["bytes"]),
        expected_sha256=str(wheel["sha256"]),
    )
    wheel_files = sorted((bundle_root / "wheels").glob("*"))
    if wheel_files != [wheel_path]:
        raise ValueError(
            "native bundle wheels directory must contain only the MMCV wheel"
        )

    verified_extensions: list[tuple[Path, Mapping[str, object]]] = []
    for record in manifest["extensions"]:
        if not isinstance(record, Mapping):
            raise ValueError("native extension record must be an object")
        path = _verify_file(
            bundle_root.joinpath(*PurePosixPath(str(record["path"])).parts),
            expected_bytes=int(record["bytes"]),
            expected_sha256=str(record["sha256"]),
        )
        verified_extensions.append((path, record))
    observed_extensions = sorted(bundle_root.glob("transvision/models/**/*.so"))
    if observed_extensions != sorted(path for path, _ in verified_extensions):
        raise ValueError("native bundle extension layout does not match its manifest")
    return overlay_root, wheel_path, verified_extensions


def _create_runtime_venv(path: Path, env: Mapping[str, str]) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite runtime venv: {path}")
    _run(
        [
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            str(path),
        ],
        env=env,
    )
    python = path / "bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"runtime venv Python is missing: {python}")
    return python


def _pin_hostname_to_loopback() -> None:
    """Ensure gethostname() resolves inside ClearML workers (A100/V100)."""

    import socket

    host = socket.gethostname().strip()
    if not host:
        return
    try:
        socket.getaddrinfo(host, None)
        return
    except OSError:
        pass
    hosts_path = Path("/etc/hosts")
    existing = hosts_path.read_text(encoding="utf-8") if hosts_path.exists() else ""
    marker = f"127.0.0.1 {host}"
    if marker in existing:
        return
    with hosts_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{marker}\n")


def _runtime_environment(
    base_env: Mapping[str, str],
    *,
    venv_root: Path,
    source_root: Path,
) -> dict[str, str]:
    env = dict(base_env)
    force_weights_only = (
        str(env.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "")).strip().lower()
    )
    if force_weights_only in {"1", "y", "yes", "true"}:
        raise RuntimeError(
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD conflicts with trusted MMEngine "
            "checkpoint loading"
        )
    for key in BUILD_ONLY_ENVIRONMENT_KEYS:
        env.pop(key, None)
    env.pop("PYTHONHOME", None)
    env.update(
        {
            "VIRTUAL_ENV": str(venv_root),
            "PATH": f"{venv_root / 'bin'}:{base_env.get('PATH', '')}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(source_root),
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            # ClearML A100/V100 workers may have unresolved hostnames (c10d).
            "MASTER_ADDR": "127.0.0.1",
            # Single-node DDP under docker --network host: avoid wrong NIC / IB.
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
            "NCCL_P2P_DISABLE": "1",
            "GLOO_SOCKET_IFNAME": "lo",
        }
    )
    if any("resilient-v2x-cuda" in value for value in env.values()):
        raise RuntimeError("portable CUDA toolchain leaked into formal runtime")
    return env


def _venv_site_packages(python: Path, env: Mapping[str, str]) -> Path:
    result = _run(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        env=env,
        capture=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"unexpected venv site-packages output: {result.stdout!r}")
    path = Path(lines[0]).resolve(strict=True)
    try:
        path.relative_to(VENV_ROOT.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError(f"site-packages escaped runtime venv: {path}") from error
    return path


RUNTIME_NATIVE_SMOKE = r"""
import importlib
import importlib.metadata
import json
import sys

import torch
from mmcv.ops import nms_rotated
from transvision.models.bev_pool.bev_pool import bev_pool
from transvision.models.voxel import Voxelization

expected = {
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
    "mmdet": "3.2.0",
    "mmdet3d": "1.3.0",
}
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"unexpected Python: {sys.version}")
if torch.__version__ != "2.10.0+cu128" or torch.version.cuda != "12.8":
    raise RuntimeError(f"unexpected Torch runtime: {torch.__version__}/{torch.version.cuda}")
if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected 4 GPUs, got {torch.cuda.device_count()}")
_allowed_caps = {(12, 0), (8, 0), (7, 0)}
_caps = [torch.cuda.get_device_capability(i) for i in range(4)]
if any(cap not in _allowed_caps for cap in _caps) or len(set(_caps)) != 1:
    raise RuntimeError(
        "all GPUs must share one allowed compute capability "
        f"from {sorted(_allowed_caps)}; got {_caps}"
    )
if "sm_120" not in torch.cuda.get_arch_list():
    raise RuntimeError("PyTorch build lacks sm_120")
for name, version in expected.items():
    if importlib.metadata.version(name) != version:
        raise RuntimeError(f"unexpected {name} version")
for module_name in (
    "mmcv._ext",
    "transvision.models.voxel.voxel_layer",
    "transvision.models.bev_pool.bev_pool_ext",
):
    importlib.import_module(module_name)

boxes = torch.tensor(
    [[0.0, 0.0, 2.0, 1.0, 0.0], [0.1, 0.1, 2.0, 1.0, 0.0]],
    device="cuda",
)
scores = torch.tensor([0.9, 0.8], device="cuda")
_, indices = nms_rotated(boxes, scores, 0.5)
if not len(indices):
    raise RuntimeError("mmcv nms_rotated returned no indices")

features = torch.randn(4, 8, device="cuda", requires_grad=True)
coordinates = torch.zeros((4, 4), dtype=torch.int32, device="cuda")
pooled = bev_pool(features, coordinates, 1, 1, 1, 1)
pooled.sum().backward()
if features.grad is None or not torch.isfinite(features.grad).all():
    raise RuntimeError("BEV pool backward did not produce finite gradients")

voxelizer = Voxelization(
    voxel_size=[0.5, 0.5, 0.5],
    point_cloud_range=[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
    max_num_points=5,
    max_voxels=10,
)
points = torch.tensor(
    [[0.1, 0.1, 0.1, 1.0], [0.2, 0.2, 0.2, 2.0], [1.1, 1.1, 1.1, 3.0]],
    dtype=torch.float32,
    device="cuda",
)
voxels, voxel_coordinates, point_counts = voxelizer(points)
if not len(voxels) or not len(voxel_coordinates) or not len(point_counts):
    raise RuntimeError("voxelization returned an empty result")
torch.cuda.synchronize()
print(json.dumps({"event": "rtx5090_native_runtime_smoke_pass"}, sort_keys=True))
"""


MODEL_SMOKE = r"""
import json
import tempfile
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.visualization import LocalVisBackend, Visualizer
import mmdet3d.models  # noqa: F401
from mmdet3d.registry import MODELS, VISUALIZERS

from transvision import register_all_modules

register_all_modules()
config = Config.fromfile("configs/resilient_v2x/dair_clean_teacher.py")
config.merge_from_dict(
    {
        "visualizer._scope_": "mmengine",
        "visualizer.type": "Visualizer",
        "visualizer.vis_backends.0._scope_": "mmengine",
    }
)
init_default_scope(config.get("default_scope", "mmdet3d"))
if config.default_hooks.get("visualization") is None:
    raise RuntimeError("headless config removed the inherited visualization hook")
with tempfile.TemporaryDirectory(prefix="resilient-v2x-visualizer-") as save_dir:
    config.visualizer.save_dir = save_dir
    visualizer = VISUALIZERS.build(config.visualizer)
    if type(visualizer) is not Visualizer:
        raise RuntimeError("headless visualizer did not resolve to mmengine.Visualizer")
    backends = list(visualizer._vis_backends.values())
    if len(backends) != 1 or type(backends[0]) is not LocalVisBackend:
        raise RuntimeError("headless visualizer requires exactly one LocalVisBackend")
    visualizer.add_scalar("headless/smoke", 1.0, step=0)
    scalars_path = Path(save_dir) / "vis_data" / "scalars.json"
    scalar_rows = (
        scalars_path.read_text(encoding="utf-8").splitlines()
        if scalars_path.is_file()
        else []
    )
    if not scalar_rows or json.loads(scalar_rows[-1]).get("headless/smoke") != 1.0:
        raise RuntimeError("LocalVisBackend did not write a valid scalars.json row")
    visualizer.close()
config.model.camera_encoder.image_backbone.init_cfg = None
model = MODELS.build(config.model).cuda()
parameters = sum(parameter.numel() for parameter in model.parameters())
if parameters <= 0:
    raise RuntimeError("teacher model has no parameters")
vehicle_config = Config.fromfile(
    "configs/resilient_v2x/dair_vehicle_pretrain.py"
)
vehicle = MODELS.build(vehicle_config.model).cuda()
vehicle_parameters = sum(parameter.numel() for parameter in vehicle.parameters())
if vehicle_parameters <= 0:
    raise RuntimeError("vehicle pretraining model has no parameters")
pillar_modules = tuple(vehicle.lidar_encoder.voxel_encoder.modules())
pillar_bn1d = sum(
    isinstance(module, torch.nn.BatchNorm1d) for module in pillar_modules
)
pillar_bn2d = sum(
    isinstance(module, torch.nn.BatchNorm2d) for module in pillar_modules
)
if pillar_bn1d <= 0 or pillar_bn2d != 0:
    raise RuntimeError("PillarFeatureNet must use BN1d and must not contain BN2d")
vehicle_keys = set(vehicle.state_dict())
teacher_keys = set(model.state_dict())
transfer_keys = {
    key
    for key in vehicle_keys
    if key.startswith("lidar_encoder.") or key.startswith("bbox_head.")
}
if not transfer_keys or transfer_keys != vehicle_keys:
    raise RuntimeError("vehicle checkpoint contains non-transfer model parameters")
if not transfer_keys.issubset(teacher_keys):
    raise RuntimeError("vehicle checkpoint keys are not a teacher state subset")
vehicle_state = vehicle.state_dict()
teacher_state = model.state_dict()
for key in sorted(transfer_keys):
    if vehicle_state[key].shape != teacher_state[key].shape:
        raise RuntimeError(
            f"vehicle checkpoint tensor shape differs from teacher for {key}: "
            f"{tuple(vehicle_state[key].shape)} != {tuple(teacher_state[key].shape)}"
        )
torch.cuda.synchronize()
print(
    json.dumps(
        {
            "event": "rtx5090_teacher_model_smoke_pass",
            "model": type(model).__name__,
            "visualizer": type(visualizer).__name__,
            "visualizer_backend": type(backends[0]).__name__,
            "scalars_json": True,
            "parameters": parameters,
            "vehicle_model": type(vehicle).__name__,
            "vehicle_parameters": vehicle_parameters,
            "vehicle_transfer_keys": len(transfer_keys),
        },
        sort_keys=True,
    )
)
"""


def _compile_embedded_smoke_scripts() -> None:
    compile(RUNTIME_NATIVE_SMOKE, "<rtx5090-native-smoke>", "exec")
    compile(MODEL_SMOKE, "<rtx5090-model-smoke>", "exec")


def _run_smoke(
    python: Path,
    script: str,
    *,
    expected_event: str,
    source_root: Path,
    env: Mapping[str, str],
) -> None:
    try:
        result = _run(
            [str(python), "-c", script],
            cwd=source_root,
            env=env,
            capture=True,
        )
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, end="" if error.stdout.endswith("\n") else "\n")
        if error.stderr:
            print(
                error.stderr,
                end="" if error.stderr.endswith("\n") else "\n",
                file=sys.stderr,
            )
        raise
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{expected_event} produced no output")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict) or payload.get("event") != expected_event:
        raise RuntimeError(f"unexpected smoke result: {payload!r}")


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


def _require_completed_task(task: object, *, context: str) -> None:
    status = _normalized_task_status(task)
    if status != "completed":
        raise RuntimeError(f"{context} is not completed: {status!r}")


def _require_files_server_url(uri: object, *, context: str) -> str:
    value = str(uri or "")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != EXPECTED_FILES_SERVER_HOST
        or parsed.port != EXPECTED_FILES_SERVER_PORT
    ):
        raise RuntimeError(
            f"{context} must use {EXPECTED_FILES_SERVER_HOST}:"
            f"{EXPECTED_FILES_SERVER_PORT}: {value!r}"
        )
    return value


def _require_failed_teacher_run_contract(
    task: object,
    *,
    expected_task_id: str,
    expected_dataset_id: str | None,
) -> dict[str, object]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or "run_contract" not in artifacts:
        raise RuntimeError("failed teacher task has no sealed run_contract artifact")
    getter = getattr(artifacts["run_contract"], "get", None)
    if not callable(getter):
        raise RuntimeError("failed teacher run_contract cannot be downloaded")
    contract = getter()
    if not isinstance(contract, Mapping):
        raise RuntimeError("failed teacher run_contract is not an object")
    expected = {
        "task_id": expected_task_id,
        "runtime_profile": "rtx5090",
        "gpus": EXPECTED_GPU_COUNT,
        "max_epochs": EXPERIMENT_MAX_EPOCHS,
        "checkpoint_policy": "final_epoch",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"failed teacher run_contract {key} mismatch: "
                f"expected {value!r}, got {contract.get(key)!r}"
            )
    if contract.get("stage") not in {"all", "teacher"}:
        raise RuntimeError("failed teacher run_contract did not train a teacher")
    if (
        expected_dataset_id is not None
        and contract.get("dataset_id") != expected_dataset_id
    ):
        raise RuntimeError("failed teacher run_contract dataset mismatch")
    return dict(contract)


def _require_unique_output_model(
    task: object,
    *,
    model_name: str,
    context: str,
    expected_task_id: str,
    expected_model_id: str | None = None,
    allow_failed_teacher_task: bool = False,
    expected_dataset_id: str | None = None,
) -> object:
    status = _normalized_task_status(task)
    salvaged_failed_teacher = (
        allow_failed_teacher_task
        and model_name == CLEAN_TEACHER_MODEL_NAME
        and status == "failed"
    )
    if status != "completed" and not salvaged_failed_teacher:
        raise RuntimeError(f"{context} is not completed: {status!r}")
    if salvaged_failed_teacher:
        _require_failed_teacher_run_contract(
            task,
            expected_task_id=expected_task_id,
            expected_dataset_id=expected_dataset_id,
        )

    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot enumerate its models")
    models = getter()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"{context} returned an invalid model mapping")
    output_models = models.get("output")
    if not isinstance(output_models, Sequence) or isinstance(
        output_models, (str, bytes)
    ):
        raise RuntimeError(f"{context} has no output model sequence")
    candidates = [
        model for model in output_models if getattr(model, "name", None) == model_name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{context} must expose exactly one {model_name!r} OutputModel; "
            f"found {len(candidates)}"
        )
    model = candidates[0]
    model_url = _require_files_server_url(
        getattr(model, "url", ""),
        context=f"{model_name} OutputModel URL",
    )
    model_task_id = str(getattr(model, "task", "") or "")
    if model_task_id != expected_task_id:
        raise RuntimeError(
            f"{model_name} OutputModel task mismatch: "
            f"expected {expected_task_id!r}, got {model_task_id!r}"
        )
    model_id = str(getattr(model, "id", "") or "")
    if CLEARML_TASK_ID_PATTERN.fullmatch(model_id) is None:
        raise RuntimeError(f"{model_name} OutputModel has an invalid ID: {model_id!r}")
    if expected_model_id is not None and model_id != expected_model_id:
        raise RuntimeError(
            f"{model_name} OutputModel ID mismatch: "
            f"expected {expected_model_id!r}, got {model_id!r}"
        )
    if salvaged_failed_teacher:
        source_filename = Path(unquote(urlsplit(model_url).path)).name
        if source_filename != "teacher_epoch_50.pth":
            raise RuntimeError(
                f"failed teacher task checkpoint filename mismatch: {source_filename!r}"
            )
    return model


def _require_unique_teacher_output_model(
    task: object,
    *,
    expected_task_id: str | None = None,
    expected_model_id: str | None = None,
    allow_failed_task: bool = False,
    expected_dataset_id: str | None = None,
) -> object:
    """Resolve one exact, provenance-bound clean-teacher OutputModel."""

    if expected_task_id is None:
        expected_task_id = str(
            next(
                (
                    getattr(model, "task", "")
                    for model in task.get_models().get("output", ())
                    if getattr(model, "name", None) == CLEAN_TEACHER_MODEL_NAME
                ),
                "",
            )
        )
    return _require_unique_output_model(
        task,
        model_name=CLEAN_TEACHER_MODEL_NAME,
        context="main training task",
        expected_task_id=expected_task_id,
        expected_model_id=expected_model_id,
        allow_failed_teacher_task=allow_failed_task,
        expected_dataset_id=expected_dataset_id,
    )


def _require_unique_student_output_model(
    task: object,
    *,
    expected_task_id: str,
    expected_model_id: str,
) -> object:
    return _require_unique_output_model(
        task,
        model_name=DISTILLED_STUDENT_MODEL_NAME,
        context="student training task",
        expected_task_id=expected_task_id,
        expected_model_id=expected_model_id,
    )


def _download_model_checkpoint(
    model: object,
    *,
    model_name: str,
    label: str,
    expected_sha256: str | None,
) -> tuple[Path, dict[str, object]]:
    getter = getattr(model, "get_local_copy", None)
    if not callable(getter):
        raise RuntimeError(f"{model_name} OutputModel cannot be downloaded")
    value = getter(
        extract_archive=False,
        raise_on_error=True,
        force_download=True,
    )
    if not value:
        raise RuntimeError(f"{model_name} OutputModel returned no local checkpoint")
    path = Path(value)
    if path.is_symlink():
        raise ValueError(f"{label} checkpoint must not be a symlink: {path}")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} checkpoint must be a regular file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} checkpoint is empty: {path}")
    observed_sha256 = _sha256(path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    model_url = str(getattr(model, "url", "") or "")
    return path, {
        "task_id": str(getattr(model, "task", "") or ""),
        "model_id": str(getattr(model, "id", "") or ""),
        "name": model_name,
        "url": model_url,
        "source_filename": Path(unquote(urlsplit(model_url).path)).name,
        "local_filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "trusted_mmengine_pickle": True,
    }


def _download_teacher_checkpoint(
    model: object,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, object]]:
    return _download_model_checkpoint(
        model,
        model_name=CLEAN_TEACHER_MODEL_NAME,
        label="teacher",
        expected_sha256=expected_sha256,
    )


def _download_student_checkpoint(
    model: object,
    *,
    expected_sha256: str,
) -> tuple[Path, dict[str, object]]:
    return _download_model_checkpoint(
        model,
        model_name=DISTILLED_STUDENT_MODEL_NAME,
        label="student",
        expected_sha256=expected_sha256,
    )


def _load_source_training_runner(source_root: Path) -> ModuleType:
    runner_path = (source_root / "tools/resilient_v2x/clearml_train.py").resolve(
        strict=True
    )
    spec = importlib.util.spec_from_file_location(
        "_sealed_resilient_v2x_clearml_train",
        runner_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sealed training runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ddp_training_command(
    python: Path,
    *,
    gpus: int,
    config: Path,
    work_dir: Path,
    max_epochs: int,
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> list[str]:
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "--module",
        "tools.resilient_v2x.run_deterministic",
        "tools/train.py",
        str(config),
        "--work-dir",
        str(work_dir),
        "--launcher",
        "pytorch",
        "--cfg-options",
        f"train_cfg.max_epochs={max_epochs}",
        f"train_cfg.val_interval={RTX5090_VAL_INTERVAL}",
        f"train_dataloader.batch_size={RTX5090_TRAIN_BATCH_SIZE_PER_GPU}",
        f"val_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        f"test_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        f"randomness.seed={training_seed}",
        f"train_dataloader.sampler.seed={training_seed}",
        f"val_dataloader.sampler.seed={training_seed}",
        f"test_dataloader.sampler.seed={training_seed}",
        f"train_dataloader.dataset.seed={training_seed}",
        f"val_dataloader.dataset.seed={training_seed}",
        f"test_dataloader.dataset.seed={training_seed}",
        f"implementation_choices_dataset.global_seed={training_seed}",
        "find_unused_parameters=True",
        *EXPERIMENT_CHECKPOINT_CFG_OPTIONS,
        *RTX5090_HEADLESS_CFG_OPTIONS,
    ]


def _baseline_plan_command(
    python: Path,
    *,
    source_root: Path,
    baseline: str,
    training_index: Path,
    work_dir: Path,
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> list[str]:
    return [
        str(python),
        str(source_root / "tools/resilient_v2x/train_controlled_baseline.py"),
        "--baseline",
        baseline,
        "--training-index",
        str(training_index),
        "--work-dir",
        str(work_dir),
        "--seed",
        str(training_seed),
        "--dry-run",
    ]


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(
        json.dumps({"command": list(command), "cwd": str(cwd)}, sort_keys=True),
        flush=True,
    )
    return _run(command, cwd=cwd, env=env, capture=capture)


def _materialize_training_dataset(
    dataset: object,
    *,
    destination: Path,
    runner: ModuleType,
) -> Path:
    materializer = getattr(runner, "_materialize_clearml_dataset", None)
    if not callable(materializer):
        raise RuntimeError("source runner has no ClearML dataset materializer")
    return materializer(dataset, destination)


def _prepare_experiment_environment(
    args: argparse.Namespace,
    *,
    task_id: str,
    source_root: Path,
    base_env: Mapping[str, str],
    dataset_class: object,
    runner: ModuleType,
) -> tuple[Path, dict[str, str]]:
    getter = getattr(dataset_class, "get", None)
    if not callable(getter):
        raise RuntimeError("ClearML Dataset class has no get method")
    dataset = getter(dataset_id=args.training_dataset_id, only_completed=True)
    dataset_root = _materialize_training_dataset(
        dataset,
        destination=(
            source_root
            / "work_dirs/clearml_dataset_materialization"
            / task_id
            / "training"
        ),
        runner=runner,
    )
    data_root = dataset_root / "cooperative-vehicle-infrastructure"
    manifest_path = dataset_root / "manifests/temporal_manifest_v2.json"
    resnet_checkpoint = dataset_root / "models/resnet50-0676ba61.pth"
    manifest = runner._read_json(manifest_path)
    if manifest.get("content_sha256") != runner.EXPECTED_MANIFEST_CONTENT_SHA256:
        raise ValueError("ClearML dataset contains the wrong temporal manifest")
    if runner._sha256(resnet_checkpoint) != runner.EXPECTED_RESNET_SHA256:
        raise ValueError("ClearML dataset contains the wrong ResNet-50 checkpoint")

    env = runner._runtime_environment(base_env, "rtx5090")
    env.update(
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


def _training_overlay_protocol_seed(dataset_root: Path) -> int:
    index_path = dataset_root / "protocols/dair_v2/training_overlays.json"
    index = _read_json_object(index_path.resolve(strict=True))
    observed = index.get("protocol_seed")
    if observed != TRAINING_OVERLAY_PROTOCOL_SEED:
        raise ValueError(
            "training overlay protocol seed mismatch: "
            f"expected {TRAINING_OVERLAY_PROTOCOL_SEED}, got {observed!r}"
        )
    return TRAINING_OVERLAY_PROTOCOL_SEED


def _validate_baseline_dry_run(
    *,
    spec: ExperimentSpec,
    work_dir: Path,
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> tuple[Path, Path, dict[str, object]]:
    plan_path = (work_dir / "training_plan.json").resolve(strict=True)
    resolved_path = (work_dir / "resolved_config.py").resolve(strict=True)
    plan = _read_json_object(plan_path)
    expected = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_training",
        "baseline": spec.name,
        "work_dir": str(work_dir.resolve(strict=True)),
        "plan_path": str(plan_path),
        "resolved_config": str(resolved_path),
        "resume": False,
        "seed": training_seed,
        "training_index_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(
                f"baseline dry-run plan {field} mismatch: "
                f"expected {expected_value!r}, got {plan.get(field)!r}"
            )
    resolved_sha256 = plan.get("resolved_config_sha256")
    if (
        type(resolved_sha256) is not str
        or SHA256_PATTERN.fullmatch(resolved_sha256) is None
        or _sha256(resolved_path) != resolved_sha256
    ):
        raise ValueError("baseline dry-run resolved config SHA-256 mismatch")
    try:
        baseline_config = Path(str(plan["baseline_config"])).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise ValueError(
            f"baseline dry-run source config is invalid: {error}"
        ) from error
    baseline_sha256 = plan.get("baseline_config_sha256")
    if (
        type(baseline_sha256) is not str
        or SHA256_PATTERN.fullmatch(baseline_sha256) is None
        or _sha256(baseline_config) != baseline_sha256
    ):
        raise ValueError("baseline dry-run source config SHA-256 mismatch")
    return plan_path, resolved_path, plan


def _upload_experiment_checkpoint(
    *,
    task: object,
    output_model_class: object,
    spec: ExperimentSpec,
    checkpoint: Path,
) -> dict[str, object]:
    if checkpoint.stat().st_size <= 0:
        raise ValueError(f"final checkpoint is empty: {checkpoint}")
    model = output_model_class(
        task=task,
        name=f"ResilientV2X {spec.name} final checkpoint",
        framework="PyTorch",
        tags=[
            "ResilientV2X",
            spec.kind,
            spec.name,
            "DDP",
            "4GPU",
            "RTX5090",
            "sm120",
            "FP32",
        ],
    )
    uri = model.update_weights(
        weights_filename=str(checkpoint),
        target_filename=f"{spec.name}_epoch_{EXPERIMENT_MAX_EPOCHS}.pth",
        iteration=EXPERIMENT_MAX_EPOCHS,
        auto_delete_file=False,
        async_enable=False,
    )
    uploaded_uri = _require_files_server_url(
        uri,
        context="final checkpoint OutputModel URL",
    )
    return {
        "model_id": str(getattr(model, "id", "") or ""),
        "name": f"ResilientV2X {spec.name} final checkpoint",
        "url": uploaded_uri,
        "filename": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }


def _resolve_clean_val_best_checkpoint(work_dir: Path) -> tuple[Path, int]:
    pattern = re.compile(r"best_resilient_v2x_car_bev_ap_r40_0\.70_epoch_(\d+)\.pth")
    candidates: list[tuple[Path, int]] = []
    for candidate in work_dir.glob(
        "best_resilient_v2x_car_bev_ap_r40_0.70_epoch_*.pth"
    ):
        match = pattern.fullmatch(candidate.name)
        if match is not None:
            candidates.append((candidate, int(match.group(1))))
    if len(candidates) != 1:
        raise RuntimeError(
            "training must retain exactly one clean-val best checkpoint; "
            f"found {[path.name for path, _ in candidates]!r}"
        )
    checkpoint, epoch = candidates[0]
    if checkpoint.is_symlink():
        raise ValueError(f"best checkpoint must not be a symlink: {checkpoint}")
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError(f"best checkpoint is missing or empty: {checkpoint}")
    if not 1 <= epoch <= EXPERIMENT_MAX_EPOCHS:
        raise ValueError(f"best checkpoint epoch is invalid: {epoch}")
    return checkpoint, epoch


def _upload_experiment_best_checkpoint(
    *,
    task: object,
    output_model_class: object,
    spec: ExperimentSpec,
    checkpoint: Path,
    epoch: int,
) -> dict[str, object]:
    if checkpoint.stat().st_size <= 0:
        raise ValueError(f"best checkpoint is empty: {checkpoint}")
    name = f"ResilientV2X {spec.name} clean-val best checkpoint"
    model = output_model_class(
        task=task,
        name=name,
        framework="PyTorch",
        tags=[
            "ResilientV2X",
            spec.kind,
            spec.name,
            "clean-val-best",
            "diagnostic",
            "not-final-claim",
            "DDP",
            "4GPU",
            "FP32",
        ],
    )
    uri = model.update_weights(
        weights_filename=str(checkpoint),
        target_filename=f"{spec.name}_clean_val_best_epoch_{epoch}.pth",
        iteration=epoch,
        auto_delete_file=False,
        async_enable=False,
    )
    return {
        "model_id": str(getattr(model, "id", "") or ""),
        "name": name,
        "url": _require_files_server_url(
            uri,
            context="clean-val best checkpoint OutputModel URL",
        ),
        "filename": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "epoch": epoch,
        "selection_protocol": "DAIR-CLEAN-PAIR1789-v1",
        "selection_metric": "resilient_v2x/car_bev_ap_r40_0.70",
        "claim_role": "diagnostic checkpoint candidate; final remains canonical",
    }


def _experiment_run_contract(
    args: argparse.Namespace,
    *,
    task_id: str,
    spec: ExperimentSpec,
    dataset_root: Path,
    teacher_contract: Mapping[str, object] | None,
    predecessor_task_id: str,
    config_path: Path,
    training_command: Sequence[str],
    baseline_plan: Mapping[str, object] | None,
) -> dict[str, object]:
    if teacher_contract is None:
        raise ValueError("experiment run contract requires a clean-teacher handoff")
    teacher_checkpoint_sha256 = teacher_contract.get("sha256")
    if (
        type(teacher_checkpoint_sha256) is not str
        or SHA256_PATTERN.fullmatch(teacher_checkpoint_sha256) is None
    ):
        raise ValueError("clean-teacher handoff has an invalid checkpoint SHA-256")
    resolved_config_sha256 = _sha256(config_path)
    if baseline_plan is None:
        declared_config = str(config_path)
        declared_config_sha256 = resolved_config_sha256
    else:
        declared_config = str(baseline_plan["baseline_config"])
        declared_config_sha256 = str(baseline_plan["baseline_config_sha256"])
    overlay_protocol_seed = _training_overlay_protocol_seed(dataset_root)
    if baseline_plan is not None:
        if baseline_plan.get("seed") != args.training_seed:
            raise ValueError("baseline plan training seed mismatch")
        if baseline_plan.get("training_index_protocol_seed") != overlay_protocol_seed:
            raise ValueError("baseline plan training overlay protocol seed mismatch")
    return {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": spec.name,
        "experiment_kind": spec.kind,
        "config": {
            "declared": spec.config,
            "declared_resolved": declared_config,
            "config_sha256": declared_config_sha256,
            "resolved": str(config_path),
            "size_bytes": config_path.stat().st_size,
            "resolved_config_sha256": resolved_config_sha256,
        },
        "source_dataset_id": args.source_dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "native_build_task_id": BUILD_TASK_ID,
        "native_bundle_sha256": args.native_bundle_sha256,
        "build_manifest_sha256": args.build_manifest_sha256,
        "base_image_manifest_digest": BASE_IMAGE_AMD64_MANIFEST_DIGEST,
        "source_archive": {
            "name": args.source_archive_name,
            "size_bytes": args.source_archive_bytes,
            "sha256": args.source_archive_sha256,
        },
        "predecessor_task_id": predecessor_task_id,
        "gpus": 4,
        "global_batch_size": args.gpus * RTX5090_TRAIN_BATCH_SIZE_PER_GPU,
        "train_batch_size_per_gpu": RTX5090_TRAIN_BATCH_SIZE_PER_GPU,
        "eval_batch_size_per_gpu": RTX5090_EVAL_BATCH_SIZE_PER_GPU,
        "launcher": "pytorch",
        "ddp_processes": 4,
        "max_epochs": EXPERIMENT_MAX_EPOCHS,
        "training_seed": args.training_seed,
        "training_overlay_protocol_seed": overlay_protocol_seed,
        "seed": args.training_seed,
        "learning_rate": 0.0001,
        "auto_scale_lr": False,
        "amp": False,
        "precision": "FP32",
        "runtime_profile": "rtx5090",
        "val_interval": RTX5090_VAL_INTERVAL,
        "per_epoch_validation": False,
        "condition_evaluation": False,
        "condition_evaluation_reason": (
            "the 12-condition matrix remains outside individual training tasks"
        ),
        "teacher": dict(teacher_contract) if teacher_contract is not None else None,
        "common_teacher_initialization": {
            "policy": COMMON_TEACHER_INITIALIZATION_POLICY,
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
            "audit_artifact_name": (COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT),
            "audit_filename": COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME,
            "expected_nested_teacher": (spec.name in NESTED_TEACHER_EXPERIMENTS),
        },
        "baseline_dry_run_required": spec.kind == "baseline",
        "dataset_local_copy": str(dataset_root),
        "training_command": list(training_command),
    }


def _prepare_teacher_handoff(
    *,
    spec: ExperimentSpec,
    teacher_task_id: str | None,
    task: object,
    task_class: object,
    env: dict[str, str],
    expected_model_id: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    allow_failed_task: bool = False,
    expected_dataset_id: str | None = None,
) -> dict[str, object] | None:
    if not spec.requires_teacher:
        env.pop("RESILIENT_V2X_TEACHER_CHECKPOINT", None)
        env.pop("RESILIENT_V2X_COMMON_INIT_CHECKPOINT", None)
        env.pop("RESILIENT_V2X_COMMON_INIT_SHA256", None)
        return None

    if teacher_task_id is None:
        raise AssertionError("teacher-dependent experiment has no teacher task ID")
    main_task = task_class.get_task(task_id=teacher_task_id)
    teacher_model = _require_unique_teacher_output_model(
        main_task,
        expected_task_id=teacher_task_id,
        expected_model_id=expected_model_id,
        allow_failed_task=allow_failed_task,
        expected_dataset_id=expected_dataset_id,
    )
    teacher_model_id = str(getattr(teacher_model, "id", "") or "")
    task.set_input_model(
        model_id=teacher_model_id,
        name="clean_teacher",
        update_task_design=False,
        update_task_labels=False,
    )
    teacher_checkpoint, teacher_contract = _download_teacher_checkpoint(
        teacher_model,
        expected_sha256=expected_checkpoint_sha256,
    )
    env["RESILIENT_V2X_TEACHER_CHECKPOINT"] = str(teacher_checkpoint)
    env["RESILIENT_V2X_COMMON_INIT_CHECKPOINT"] = str(teacher_checkpoint)
    env["RESILIENT_V2X_COMMON_INIT_SHA256"] = str(teacher_contract["sha256"])
    return teacher_contract


def _prepare_remote_checkpoint_handoffs(
    args: argparse.Namespace,
    *,
    task: object,
    task_class: object,
) -> tuple[Path | None, Path | None, dict[str, object]]:
    teacher_checkpoint = args.teacher_checkpoint
    student_checkpoint = args.student_checkpoint
    contract: dict[str, object] = {
        "schema_version": 1,
        "trust_policy": (
            "exact task/model/files-server/SHA pin with controlled "
            "MMEngine legacy pickle loading"
        ),
        "torch_force_no_weights_only_load": True,
    }

    if args.teacher_task_id is not None:
        teacher_task = task_class.get_task(task_id=args.teacher_task_id)
        teacher_model = _require_unique_teacher_output_model(
            teacher_task,
            expected_task_id=args.teacher_task_id,
            expected_model_id=args.teacher_model_id,
            allow_failed_task=args.allow_failed_teacher_task,
            expected_dataset_id=args.training_dataset_id,
        )
        task.set_input_model(
            model_id=args.teacher_model_id,
            name="clean_teacher",
            update_task_design=False,
            update_task_labels=False,
        )
        teacher_checkpoint, teacher_contract = _download_teacher_checkpoint(
            teacher_model,
            expected_sha256=args.teacher_checkpoint_sha256,
        )
        contract["teacher"] = teacher_contract
        contract["failed_task_salvage"] = args.allow_failed_teacher_task

    if args.student_task_id is not None:
        student_task = task_class.get_task(task_id=args.student_task_id)
        student_model = _require_unique_student_output_model(
            student_task,
            expected_task_id=args.student_task_id,
            expected_model_id=args.student_model_id,
        )
        task.set_input_model(
            model_id=args.student_model_id,
            name="distilled_student",
            update_task_design=False,
            update_task_labels=False,
        )
        student_checkpoint, student_contract = _download_student_checkpoint(
            student_model,
            expected_sha256=args.student_checkpoint_sha256,
        )
        contract["student"] = student_contract

    return teacher_checkpoint, student_checkpoint, contract


def _current_or_init_experiment_task(
    task_class: object,
    spec: ExperimentSpec,
) -> object:
    current_task_getter = getattr(task_class, "current_task", None)
    task = current_task_getter() if callable(current_task_getter) else None
    if task is not None:
        return task
    return task_class.init(
        project_name="ResilientV2X/Training",
        task_name=f"ResilientV2X suite: {spec.name}",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
        auto_connect_arg_parser=False,
    )


def _wait_for_completed_task(
    task: object,
    *,
    context: str,
    poll_seconds: int = 30,
) -> None:
    """Wait for one queued dependency and fail closed on terminal errors."""

    terminal_failures = {
        "failed",
        "stopped",
        "closed",
        "published",
        "rejected",
        "unknown",
    }
    previous_status: str | None = None
    while True:
        reload_task = getattr(task, "reload", None)
        if callable(reload_task):
            reload_task()
        status = _normalized_task_status(task)
        if status == "completed":
            return
        if status in terminal_failures:
            raise RuntimeError(f"{context} cannot complete: {status!r}")
        if status != previous_status:
            print(
                json.dumps(
                    {
                        "event": "waiting_for_task_dependency",
                        "context": context,
                        "status": status,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            previous_status = status
        time.sleep(poll_seconds)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"value is outside the canonical JSON domain: {error}"
        ) from error


def _producer_content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("secure evidence directory open is unavailable")
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)


def _regular_file_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("secure evidence file open is unavailable")
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )


def _open_secure_evidence_root(path: Path, *, context: str) -> tuple[Path, int]:
    candidate = Path(path).expanduser()
    try:
        before = candidate.lstat()
    except OSError as error:
        raise RuntimeError(f"{context} is unavailable") from error
    if candidate.is_symlink() or not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"{context} must be a non-symlink directory")
    try:
        descriptor = os.open(candidate, _directory_open_flags())
    except OSError as error:
        raise RuntimeError(f"{context} cannot be securely opened") from error
    try:
        opened = os.fstat(descriptor)
        current = candidate.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _stat_identity(before) != _stat_identity(opened)
            or _stat_identity(current) != _stat_identity(opened)
        ):
            raise RuntimeError(f"{context} identity changed during secure open")
        resolved = candidate.resolve(strict=True)
    except BaseException:
        os.close(descriptor)
        raise
    return resolved, descriptor


def _validate_evidence_relative_path(
    relative: PurePosixPath,
    *,
    context: str,
) -> None:
    if (
        relative.is_absolute()
        or relative.as_posix() in {"", "."}
        or ".." in relative.parts
        or any(not part or "/" in part or "\\" in part for part in relative.parts)
    ):
        raise RuntimeError(f"{context} has an unsafe relative path")


def _open_evidence_member(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    context: str,
) -> tuple[int, list[int], os.stat_result]:
    _validate_evidence_relative_path(relative, context=context)
    directories = [os.dup(root_descriptor)]
    descriptor = -1
    try:
        for component in relative.parts[:-1]:
            parent = directories[-1]
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise RuntimeError(f"{context} parent is not a regular directory")
            child = os.open(component, _directory_open_flags(), dir_fd=parent)
            directories.append(child)
            opened = os.fstat(child)
            if _stat_identity(before) != _stat_identity(opened):
                raise RuntimeError(f"{context} parent changed during secure open")

        parent = directories[-1]
        filename = relative.name
        before = os.stat(filename, dir_fd=parent, follow_symlinks=False)
        descriptor = os.open(filename, _regular_file_open_flags(), dir_fd=parent)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size <= 0
            or opened.st_size > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES
            or _stat_identity(before) != _stat_identity(opened)
        ):
            raise RuntimeError(f"{context} is not a sealed regular file")
        return descriptor, directories, opened
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        for descriptor in reversed(directories):
            os.close(descriptor)
        raise


def _verify_open_evidence_member_unchanged(
    descriptor: int,
    directories: Sequence[int],
    relative: PurePosixPath,
    before: os.stat_result,
    *,
    context: str,
) -> os.stat_result:
    after = os.fstat(descriptor)
    current = os.stat(
        relative.name,
        dir_fd=directories[-1],
        follow_symlinks=False,
    )
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(
        after
    ) != _stat_identity(current):
        raise RuntimeError(f"{context} changed while being staged")
    for index, component in enumerate(relative.parts[:-1]):
        child = os.fstat(directories[index + 1])
        current_child = os.stat(
            component,
            dir_fd=directories[index],
            follow_symlinks=False,
        )
        if _stat_identity(child) != _stat_identity(current_child):
            raise RuntimeError(f"{context} parent changed while being staged")
    return after


def _close_evidence_member(descriptor: int, directories: Sequence[int]) -> None:
    os.close(descriptor)
    for directory in reversed(directories):
        os.close(directory)


def _read_evidence_member(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    context: str,
) -> bytes:
    descriptor, directories, before = _open_evidence_member(
        root_descriptor,
        relative,
        context=context,
    )
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
                raise RuntimeError(f"{context} exceeded its byte cap")
            chunks.append(chunk)
        _verify_open_evidence_member_unchanged(
            descriptor,
            directories,
            relative,
            before,
            context=context,
        )
        if total != before.st_size:
            raise RuntimeError(f"{context} size changed while being read")
        return b"".join(chunks)
    finally:
        _close_evidence_member(descriptor, directories)


def _copy_fd_payload(source: int, destination: int) -> tuple[int, str]:
    """Copy one already-securely-opened file; split out for TOCTOU tests."""

    digest = hashlib.sha256()
    copied = 0
    while True:
        chunk = os.read(source, 1024 * 1024)
        if not chunk:
            break
        copied += len(chunk)
        if copied > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
            raise RuntimeError("controlled evidence member exceeded its byte cap")
        digest.update(chunk)
        view = memoryview(chunk)
        while view:
            written = os.write(destination, view)
            if written <= 0:
                raise RuntimeError("controlled evidence staging write made no progress")
            view = view[written:]
    return copied, digest.hexdigest()


def _open_stage_parent(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    context: str,
) -> list[int]:
    _validate_evidence_relative_path(relative, context=context)
    directories = [os.dup(root_descriptor)]
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                _directory_open_flags(),
                dir_fd=directories[-1],
            )
            directories.append(child)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                raise RuntimeError(f"{context} staging parent is not a directory")
        return directories
    except BaseException:
        for descriptor in reversed(directories):
            os.close(descriptor)
        raise


def _stage_copy_evidence_member(
    source_root: int,
    stage_root: int,
    relative: PurePosixPath,
) -> ControlledEvidenceMemberReceipt:
    context = f"controlled evidence member {relative.as_posix()!r}"
    source, source_directories, source_before = _open_evidence_member(
        source_root,
        relative,
        context=context,
    )
    stage_directories: list[int] = []
    destination = -1
    try:
        stage_directories = _open_stage_parent(
            stage_root,
            relative,
            context=context,
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_BINARY", 0)
        )
        destination = os.open(
            relative.name,
            flags,
            0o400,
            dir_fd=stage_directories[-1],
        )
        copied, digest = _copy_fd_payload(source, destination)
        os.fsync(destination)
        _verify_open_evidence_member_unchanged(
            source,
            source_directories,
            relative,
            source_before,
            context=context,
        )
        if copied != source_before.st_size:
            raise RuntimeError(f"{context} size changed while being staged")
        os.fchmod(destination, 0o400)
        staged = os.fstat(destination)
        current = os.stat(
            relative.name,
            dir_fd=stage_directories[-1],
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged.st_nlink != 1
            or _stat_identity(staged) != _stat_identity(current)
            or staged.st_size != copied
        ):
            raise RuntimeError(f"{context} staging destination drifted")
        return ControlledEvidenceMemberReceipt(
            relative.as_posix(),
            copied,
            digest,
            _stat_identity(staged),
        )
    finally:
        if destination >= 0:
            os.close(destination)
        for descriptor in reversed(stage_directories):
            os.close(descriptor)
        _close_evidence_member(source, source_directories)


def _stage_write_evidence_member(
    stage_root: int,
    relative: PurePosixPath,
    raw: bytes,
) -> ControlledEvidenceMemberReceipt:
    if not raw or len(raw) > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
        raise RuntimeError(f"staged evidence member {relative} has an invalid size")
    directories = _open_stage_parent(
        stage_root,
        relative,
        context=f"staged evidence member {relative.as_posix()!r}",
    )
    descriptor = -1
    try:
        descriptor = os.open(
            relative.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_BINARY", 0),
            0o400,
            dir_fd=directories[-1],
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("controlled evidence staging write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        value = os.fstat(descriptor)
        current = os.stat(
            relative.name,
            dir_fd=directories[-1],
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or value.st_size != len(raw)
            or _stat_identity(value) != _stat_identity(current)
        ):
            raise RuntimeError(f"staged evidence member {relative} drifted")
        return ControlledEvidenceMemberReceipt(
            relative.as_posix(),
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _stat_identity(value),
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)


def _json_mapping_from_bytes(raw: bytes, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise RuntimeError(f"{context} must be a JSON object")
    return value


def _reject_stale_evidence_hash_fields(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        stale = CONTROLLED_EVIDENCE_FORBIDDEN_STALE_HASH_FIELDS.intersection(value)
        if stale:
            raise RuntimeError(
                f"{context} contains stale plan/config hash fields: {sorted(stale)}"
            )
        for key, item in value.items():
            _reject_stale_evidence_hash_fields(item, context=f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_stale_evidence_hash_fields(
                item,
                context=f"{context}[{index}]",
            )


def _controlled_condition_ids() -> tuple[str, ...]:
    return tuple(
        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        for delay in (0, 100, 200, 300)
        for condition in ("Full", "L-Fail", "C-Fail")
    )


def _validate_controlled_evidence_documents(
    plan: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    work_dir: Path,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    work_dir_text = str(work_dir)
    expected_paths = {
        "work_dir": work_dir_text,
        "plan_path": str(work_dir / "evaluation_plan.json"),
        "metrics_output": str(work_dir / "metrics.json"),
    }
    for field, expected in expected_paths.items():
        if plan.get(field) != expected:
            raise RuntimeError(f"controlled evidence plan {field} drifted")
    declared_hash = plan.get("content_sha256")
    if (
        type(declared_hash) is not str
        or SHA256_PATTERN.fullmatch(declared_hash) is None
        or declared_hash != _producer_content_sha256(plan)
    ):
        raise RuntimeError("controlled evidence source plan content hash mismatch")

    plan_runs = plan.get("runs")
    metric_runs = metrics.get("runs")
    condition_ids = _controlled_condition_ids()
    if (
        not isinstance(plan_runs, list)
        or not isinstance(metric_runs, list)
        or len(plan_runs) != len(condition_ids)
        or len(metric_runs) != len(condition_ids)
        or metrics.get("complete") is not True
        or metrics.get("planned_run_count") != len(condition_ids)
    ):
        raise RuntimeError("controlled evidence matrix is not exactly 12 complete runs")
    _reject_stale_evidence_hash_fields(metrics, context="controlled metrics")

    normalized_plan_runs: list[Mapping[str, object]] = []
    normalized_metric_runs: list[Mapping[str, object]] = []
    for index, condition_id in enumerate(condition_ids):
        plan_run = plan_runs[index]
        metric_run = metric_runs[index]
        if not isinstance(plan_run, Mapping) or not isinstance(metric_run, Mapping):
            raise RuntimeError(f"controlled evidence run {index} is not an object")
        delay = (0, 100, 200, 300)[index // 3]
        condition = ("Full", "L-Fail", "C-Fail")[index % 3]
        expected_run = {
            "condition_id": condition_id,
            "delay_ms": delay,
            "condition": condition,
            "resolved_config": str(work_dir / condition_id / "resolved_config.py"),
            "predictions": str(work_dir / condition_id / "predictions.json"),
            "checkpoint_sha256_file": str(
                work_dir / condition_id / "checkpoint.sha256"
            ),
        }
        for field, expected in expected_run.items():
            if plan_run.get(field) != expected:
                raise RuntimeError(f"controlled evidence run {index} {field} drifted")
        for field in ("condition_id", "delay_ms", "condition", "predictions"):
            if metric_run.get(field) != expected_run[field]:
                raise RuntimeError(
                    f"controlled metrics run {index} {field} is not cross-bound"
                )
        normalized_plan_runs.append(plan_run)
        normalized_metric_runs.append(metric_run)
    return normalized_plan_runs, normalized_metric_runs


def _create_private_evidence_stage(destination: Path) -> tuple[Path, int]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to reuse evidence staging directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(destination, 0o700)
    return _open_secure_evidence_root(
        destination,
        context="controlled baseline evidence staging directory",
    )


def _stage_controlled_baseline_evidence(
    *,
    work_dir: Path,
    staging_dir: Path | None = None,
) -> ControlledEvidenceStage:
    """Create one private, exact 38-member snapshot from a polluted Runner work dir."""

    source_path, source_descriptor = _open_secure_evidence_root(
        work_dir,
        context="controlled baseline evaluation work directory",
    )
    try:
        source_identity = _stat_identity(os.fstat(source_descriptor))
        destination = (
            source_path.with_name(f".{source_path.name}.controlled-baseline-evidence")
            if staging_dir is None
            else Path(staging_dir).expanduser()
        )
        stage_path, stage_descriptor = _create_private_evidence_stage(destination)
    except BaseException:
        os.close(source_descriptor)
        raise
    receipts: list[ControlledEvidenceMemberReceipt] = []
    try:
        source_plan = _json_mapping_from_bytes(
            _read_evidence_member(
                source_descriptor,
                PurePosixPath("evaluation_plan.json"),
                context="source evaluation plan",
            ),
            context="source evaluation plan",
        )
        source_metrics_raw = _read_evidence_member(
            source_descriptor,
            PurePosixPath("metrics.json"),
            context="source controlled metrics",
        )
        source_metrics = _json_mapping_from_bytes(
            source_metrics_raw,
            context="source controlled metrics",
        )
        plan_runs, metric_runs = _validate_controlled_evidence_documents(
            source_plan,
            source_metrics,
            work_dir=source_path,
        )

        condition_ids = _controlled_condition_ids()
        for condition_id in condition_ids:
            os.mkdir(condition_id, 0o700, dir_fd=stage_descriptor)

        metrics_receipt = _stage_copy_evidence_member(
            source_descriptor,
            stage_descriptor,
            PurePosixPath("metrics.json"),
        )
        if metrics_receipt.sha256 != hashlib.sha256(source_metrics_raw).hexdigest():
            raise RuntimeError("source controlled metrics changed before staging")
        receipts.append(metrics_receipt)
        resealed_runs: list[dict[str, object]] = []
        checkpoint_sha256 = source_plan.get("checkpoint_sha256")
        if (
            type(checkpoint_sha256) is not str
            or SHA256_PATTERN.fullmatch(checkpoint_sha256) is None
        ):
            raise RuntimeError("controlled evidence checkpoint SHA-256 is invalid")
        for index, condition_id in enumerate(condition_ids):
            run = dict(plan_runs[index])
            copied: dict[str, ControlledEvidenceMemberReceipt] = {}
            for filename in CONTROLLED_EVIDENCE_REQUIRED_FILENAMES:
                relative = PurePosixPath(condition_id, filename)
                receipt = _stage_copy_evidence_member(
                    source_descriptor,
                    stage_descriptor,
                    relative,
                )
                receipts.append(receipt)
                copied[filename] = receipt
            run["resolved_config_sha256"] = copied["resolved_config.py"].sha256
            metric_prediction_sha = metric_runs[index].get("prediction_sha256")
            if metric_prediction_sha != copied["predictions.json"].sha256:
                raise RuntimeError(
                    f"controlled evidence run {index} prediction SHA-256 drifted"
                )
            checkpoint_raw = _read_evidence_member(
                stage_descriptor,
                PurePosixPath(condition_id, "checkpoint.sha256"),
                context=f"staged checkpoint SHA file for run {index}",
            )
            if checkpoint_raw != f"{checkpoint_sha256}\n".encode("ascii"):
                raise RuntimeError(
                    f"controlled evidence run {index} checkpoint SHA file drifted"
                )
            resealed_runs.append(run)

        resealed_plan = dict(source_plan)
        resealed_plan["runs"] = resealed_runs
        resealed_plan["content_sha256"] = _producer_content_sha256(resealed_plan)
        plan_raw = (
            json.dumps(
                resealed_plan,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        receipts.append(
            _stage_write_evidence_member(
                stage_descriptor,
                PurePosixPath("evaluation_plan.json"),
                plan_raw,
            )
        )

        for condition_id in condition_ids:
            directory = os.open(
                condition_id,
                _directory_open_flags(),
                dir_fd=stage_descriptor,
            )
            try:
                os.fchmod(directory, 0o500)
                os.fsync(directory)
            finally:
                os.close(directory)
        os.fchmod(stage_descriptor, 0o500)
        os.fsync(stage_descriptor)
        current_source = source_path.lstat()
        if source_identity != _stat_identity(current_source):
            raise RuntimeError(
                "controlled baseline evaluation work directory changed during staging"
            )
        root_identity = _stat_identity(os.fstat(stage_descriptor))
    finally:
        os.close(stage_descriptor)
        os.close(source_descriptor)

    ordered = tuple(sorted(receipts, key=lambda item: item.relative_path))
    if len(ordered) != 38 or sum(item.size_bytes for item in ordered) > (
        CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES
    ):
        raise RuntimeError("controlled baseline evidence staging inventory is invalid")
    stage = ControlledEvidenceStage(stage_path, root_identity, ordered)
    _verify_controlled_evidence_stage(stage)
    return stage


def _walk_staged_evidence(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    stack: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath("."))]
    while stack:
        directory, relative_root = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = (
                    PurePosixPath(entry.name)
                    if relative_root.as_posix() == "."
                    else relative_root / entry.name
                )
                value = entry.stat(follow_symlinks=False)
                if entry.is_symlink():
                    raise RuntimeError(
                        f"controlled evidence staging contains symlink {relative}"
                    )
                if stat.S_ISDIR(value.st_mode):
                    if stat.S_IMODE(value.st_mode) != 0o500:
                        raise RuntimeError(
                            f"controlled evidence staging directory mode drifted: {relative}"
                        )
                    directories.add(relative.as_posix())
                    stack.append((Path(entry.path), relative))
                elif stat.S_ISREG(value.st_mode) and value.st_nlink == 1:
                    if stat.S_IMODE(value.st_mode) != 0o400:
                        raise RuntimeError(
                            f"controlled evidence staging file mode drifted: {relative}"
                        )
                    files.add(relative.as_posix())
                else:
                    raise RuntimeError(
                        f"controlled evidence staging has non-regular member {relative}"
                    )
    return files, directories


def _hash_evidence_member(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    context: str,
) -> tuple[int, str, tuple[int, ...]]:
    descriptor, directories, before = _open_evidence_member(
        root_descriptor,
        relative,
        context=context,
    )
    try:
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
                raise RuntimeError(f"{context} exceeded its byte cap")
            digest.update(chunk)
        after = _verify_open_evidence_member_unchanged(
            descriptor,
            directories,
            relative,
            before,
            context=context,
        )
        return total, digest.hexdigest(), _stat_identity(after)
    finally:
        _close_evidence_member(descriptor, directories)


def _verify_controlled_evidence_stage(
    stage: ControlledEvidenceStage,
) -> tuple[dict[str, object], dict[str, object]]:
    root, descriptor = _open_secure_evidence_root(
        stage.root,
        context="controlled baseline evidence staging directory",
    )
    try:
        if _stat_identity(os.fstat(descriptor)) != stage.root_identity:
            raise RuntimeError("controlled evidence staging root identity drifted")
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o500:
            raise RuntimeError("controlled evidence staging root mode drifted")
        expected_files = {receipt.relative_path for receipt in stage.members}
        expected_directories = {
            PurePosixPath(path).parent.as_posix()
            for path in expected_files
            if PurePosixPath(path).parent.as_posix() != "."
        }
        observed_files, observed_directories = _walk_staged_evidence(root)
        if (
            len(stage.members) != 38
            or len(expected_files) != 38
            or observed_files != expected_files
            or observed_directories != expected_directories
        ):
            raise RuntimeError("controlled evidence staging exact inventory drifted")
        for receipt in stage.members:
            size, digest, identity = _hash_evidence_member(
                descriptor,
                PurePosixPath(receipt.relative_path),
                context=f"staged evidence member {receipt.relative_path!r}",
            )
            if (
                size != receipt.size_bytes
                or digest != receipt.sha256
                or identity != receipt.identity
            ):
                raise RuntimeError(
                    f"controlled evidence staging member drifted: {receipt.relative_path}"
                )
        plan = _json_mapping_from_bytes(
            _read_evidence_member(
                descriptor,
                PurePosixPath("evaluation_plan.json"),
                context="staged evaluation plan",
            ),
            context="staged evaluation plan",
        )
        metrics = _json_mapping_from_bytes(
            _read_evidence_member(
                descriptor,
                PurePosixPath("metrics.json"),
                context="staged controlled metrics",
            ),
            context="staged controlled metrics",
        )
        plan_runs, metric_runs = _validate_controlled_evidence_documents(
            plan,
            metrics,
            work_dir=Path(str(plan["work_dir"])),
        )
        receipt_by_path = {item.relative_path: item for item in stage.members}
        checkpoint_sha256 = plan.get("checkpoint_sha256")
        for index, condition_id in enumerate(_controlled_condition_ids()):
            config_receipt = receipt_by_path[f"{condition_id}/resolved_config.py"]
            prediction_receipt = receipt_by_path[f"{condition_id}/predictions.json"]
            if plan_runs[index].get("resolved_config_sha256") != config_receipt.sha256:
                raise RuntimeError(
                    f"staged evaluation plan run {index} config SHA-256 drifted"
                )
            if metric_runs[index].get("prediction_sha256") != prediction_receipt.sha256:
                raise RuntimeError(
                    f"staged controlled metrics run {index} prediction SHA drifted"
                )
            checkpoint_raw = _read_evidence_member(
                descriptor,
                PurePosixPath(condition_id, "checkpoint.sha256"),
                context=f"staged checkpoint SHA file for run {index}",
            )
            if checkpoint_raw != f"{checkpoint_sha256}\n".encode("ascii"):
                raise RuntimeError(
                    f"staged checkpoint SHA file for run {index} drifted"
                )
            prediction_raw = _read_evidence_member(
                descriptor,
                PurePosixPath(condition_id, "predictions.json"),
                context=f"staged predictions for run {index}",
            )
            if hashlib.sha256(prediction_raw).hexdigest() != prediction_receipt.sha256:
                raise RuntimeError(
                    f"staged predictions for run {index} changed during verification"
                )
            prediction = _json_mapping_from_bytes(
                prediction_raw,
                context=f"staged predictions for run {index}",
            )
            _reject_stale_evidence_hash_fields(
                prediction,
                context=f"staged predictions for run {index}",
            )
        if _stat_identity(stage.root.lstat()) != stage.root_identity:
            raise RuntimeError(
                "controlled evidence staging root changed during verification"
            )
        return plan, metrics
    finally:
        os.close(descriptor)


def _read_uploaded_artifact_file(artifact: object, *, context: str) -> Path:
    _require_files_server_url(getattr(artifact, "url", None), context=context)
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot be downloaded")
    try:
        value = getter(
            extract_archive=False,
            raise_on_error=True,
            force_download=True,
        )
    except TypeError as error:
        raise RuntimeError(
            f"{context} downloader cannot disable automatic archive extraction"
        ) from error
    if not value:
        raise RuntimeError(f"{context} returned no local copy")
    path = Path(value)
    try:
        result = path.lstat()
    except OSError as error:
        raise RuntimeError(f"{context} local copy is unavailable") from error
    if path.is_symlink() or not stat.S_ISREG(result.st_mode):
        raise RuntimeError(f"{context} local copy is not a regular file")
    return path


def _read_uploaded_artifact_bytes(artifact: object, *, context: str) -> bytes:
    path = _read_uploaded_artifact_file(artifact, context=context)
    try:
        descriptor = os.open(path, _regular_file_open_flags())
    except OSError as error:
        raise RuntimeError(f"{context} local copy cannot be securely opened") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES
        ):
            raise RuntimeError(f"{context} local copy size/type drifted")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
                raise RuntimeError(f"{context} exceeded its byte cap")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = path.lstat()
        if (
            total != before.st_size
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
        ):
            raise RuntimeError(f"{context} changed during readback")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_uploaded_evidence_zip(
    archive_path: Path,
    stage: ControlledEvidenceStage,
) -> None:
    descriptor = os.open(archive_path, _regular_file_open_flags())
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES
        ):
            raise RuntimeError("uploaded evidence ZIP size/type drifted")
        expected = {item.relative_path: item for item in stage.members}
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            try:
                with zipfile.ZipFile(stream, "r", allowZip64=False) as archive:
                    members = archive.infolist()
                    if archive.comment:
                        raise RuntimeError("uploaded evidence ZIP comment is forbidden")
                    names = [member.filename for member in members]
                    if (
                        len(names) != 38
                        or len(set(names)) != 38
                        or set(names) != set(expected)
                    ):
                        raise RuntimeError(
                            "uploaded evidence ZIP exact inventory drifted"
                        )
                    total = 0
                    for member in members:
                        original_name = getattr(member, "orig_filename", None)
                        relative = PurePosixPath(member.filename)
                        unix_mode = member.external_attr >> 16
                        if (
                            type(member.filename) is not str
                            or type(original_name) is not str
                            or original_name != member.filename
                            or "\x00" in original_name
                            or member.create_system != 3
                            or relative.is_absolute()
                            or relative.as_posix() != member.filename
                            or ".." in relative.parts
                            or "\\" in member.filename
                            or member.is_dir()
                            or member.flag_bits & 1
                            or member.compress_type
                            not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                            or member.file_size <= 0
                            or member.file_size > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES
                            or member.compress_size <= 0
                            or member.file_size / member.compress_size > 128.0
                            or member.extra
                            or member.comment
                            or not stat.S_ISREG(unix_mode)
                        ):
                            raise RuntimeError(
                                f"uploaded evidence ZIP member is unsafe: {member.filename!r}"
                            )
                        receipt = expected[member.filename]
                        digest = hashlib.sha256()
                        copied = 0
                        with archive.open(member, "r") as source:
                            while True:
                                chunk = source.read(1024 * 1024)
                                if not chunk:
                                    break
                                copied += len(chunk)
                                total += len(chunk)
                                if (
                                    copied > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES
                                    or total > CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES
                                ):
                                    raise RuntimeError(
                                        "uploaded evidence ZIP exceeded its byte cap"
                                    )
                                digest.update(chunk)
                        if (
                            copied != receipt.size_bytes
                            or copied != member.file_size
                            or digest.hexdigest() != receipt.sha256
                        ):
                            raise RuntimeError(
                                f"uploaded evidence ZIP member drifted: {member.filename}"
                            )
            except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
                if isinstance(error, RuntimeError):
                    raise
                raise RuntimeError(
                    "uploaded controlled evidence ZIP is invalid"
                ) from error
        after = os.fstat(descriptor)
        current = archive_path.lstat()
        if _stat_identity(before) != _stat_identity(after) or _stat_identity(
            after
        ) != _stat_identity(current):
            raise RuntimeError("uploaded evidence ZIP changed during verification")
    finally:
        os.close(descriptor)


def _verify_uploaded_controlled_evidence(
    task: object,
    stage: ControlledEvidenceStage,
) -> None:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("uploaded controlled evidence artifacts are unavailable")
    required = {
        "run_contract",
        "evaluation_plan",
        "controlled_baseline_metrics",
        "controlled_baseline_evidence",
    }
    if set(artifacts) != required:
        raise RuntimeError(
            "uploaded controlled evidence exact artifact inventory drifted"
        )

    stage_root, stage_descriptor = _open_secure_evidence_root(
        stage.root,
        context="controlled baseline evidence staging directory",
    )
    del stage_root
    try:
        staged_plan_raw = _read_evidence_member(
            stage_descriptor,
            PurePosixPath("evaluation_plan.json"),
            context="staged evaluation plan",
        )
    finally:
        os.close(stage_descriptor)
    uploaded_plan_raw = _read_uploaded_artifact_bytes(
        artifacts["evaluation_plan"],
        context="uploaded evaluation_plan artifact",
    )
    if uploaded_plan_raw != staged_plan_raw:
        raise RuntimeError("uploaded evaluation_plan bytes differ from sealed staging")

    _staged_plan, staged_metrics = _verify_controlled_evidence_stage(stage)
    metrics_artifact = artifacts["controlled_baseline_metrics"]
    if getattr(metrics_artifact, "type", None) != "dict":
        raise RuntimeError(
            "uploaded controlled_baseline_metrics is not a dict artifact"
        )
    uploaded_metrics = _json_mapping_from_bytes(
        _read_uploaded_artifact_bytes(
            metrics_artifact,
            context="uploaded controlled_baseline_metrics artifact",
        ),
        context="uploaded controlled_baseline_metrics artifact",
    )
    if _canonical_json_bytes(uploaded_metrics) != _canonical_json_bytes(staged_metrics):
        raise RuntimeError(
            "uploaded controlled_baseline_metrics differs from sealed staging"
        )

    evidence = artifacts["controlled_baseline_evidence"]
    if getattr(evidence, "type", None) != "archive":
        raise RuntimeError("uploaded controlled evidence artifact is not an archive")
    evidence_url = _require_files_server_url(
        getattr(evidence, "url", None),
        context="uploaded controlled_baseline_evidence artifact",
    )
    if PurePosixPath(urlsplit(evidence_url).path).suffix.lower() != ".zip":
        raise RuntimeError("uploaded controlled evidence artifact URL is not a ZIP")
    archive_path = _read_uploaded_artifact_file(
        evidence,
        context="uploaded controlled_baseline_evidence artifact",
    )
    _verify_uploaded_evidence_zip(archive_path, stage)


def _upload_controlled_baseline_artifacts(
    task: object,
    *,
    evidence_stage: ControlledEvidenceStage,
) -> None:
    _plan, metrics = _verify_controlled_evidence_stage(evidence_stage)
    for artifact_name, artifact_object in (
        (
            "evaluation_plan",
            str(evidence_stage.root / "evaluation_plan.json"),
        ),
        ("controlled_baseline_metrics", dict(metrics)),
        ("controlled_baseline_evidence", str(evidence_stage.root)),
    ):
        _verify_controlled_evidence_stage(evidence_stage)
        if not task.upload_artifact(
            artifact_name,
            artifact_object=artifact_object,
            wait_on_upload=True,
        ):
            raise RuntimeError(f"failed to upload {artifact_name}")
    _verify_controlled_evidence_stage(evidence_stage)
    flusher = getattr(task, "flush", None)
    if not callable(flusher):
        raise RuntimeError("controlled baseline task cannot flush artifact uploads")
    flusher(wait_for_uploads=True)
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()
    _verify_uploaded_controlled_evidence(task, evidence_stage)


def _execute_controlled_baseline_validation(
    args: argparse.Namespace,
    *,
    source_root: Path,
    python: Path,
    runtime_env: Mapping[str, str],
    dataset_class: object,
    task_class: object,
) -> int:
    """Evaluate one trained baseline on the sealed 12-condition matrix."""

    baseline = str(args.controlled_baseline)
    baseline_task_id = str(args.controlled_baseline_task_id)
    baseline_model_id = str(args.controlled_baseline_model_id)
    baseline_checkpoint_sha256 = str(args.controlled_baseline_checkpoint_sha256)
    predecessor_task_id = str(args.predecessor_task_id)
    current_task_getter = getattr(task_class, "current_task", None)
    task = current_task_getter() if callable(current_task_getter) else None
    if task is None:
        raise RuntimeError("baseline validation requires a current ClearML task")
    task.output_uri = FILES_SERVER_URI
    task_id = str(getattr(task, "id", "") or "")
    if CLEARML_TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise RuntimeError(f"current ClearML task has an invalid ID: {task_id!r}")
    if task_id in {predecessor_task_id, baseline_task_id}:
        raise RuntimeError("a baseline validation task cannot depend on itself")

    existing_tags = list(task.get_tags())
    task.set_tags(
        list(
            dict.fromkeys(
                existing_tags
                + [
                    "ResilientV2X-suite",
                    "controlled-baseline-evaluation",
                    baseline,
                    "12-condition",
                    "4gpu",
                    "FP32",
                ]
            )
        )
    )

    predecessor_task = task_class.get_task(task_id=predecessor_task_id)
    _wait_for_completed_task(
        predecessor_task,
        context=f"predecessor task {predecessor_task_id}",
    )
    baseline_task = task_class.get_task(task_id=baseline_task_id)
    _wait_for_completed_task(
        baseline_task,
        context=f"controlled baseline task {baseline_task_id}",
    )
    parameters_getter = getattr(baseline_task, "get_parameters", None)
    if not callable(parameters_getter):
        raise RuntimeError("controlled baseline task cannot expose parameters")
    baseline_parameters = parameters_getter()
    if not isinstance(baseline_parameters, Mapping):
        raise RuntimeError("controlled baseline task returned invalid parameters")
    if baseline_parameters.get("Args/experiment_from_task") != baseline:
        raise RuntimeError(
            "controlled baseline task experiment mismatch: "
            f"expected {baseline!r}, got "
            f"{baseline_parameters.get('Args/experiment_from_task')!r}"
        )

    model_name = f"ResilientV2X {baseline} final checkpoint"
    model = _require_unique_output_model(
        baseline_task,
        model_name=model_name,
        context=f"controlled baseline task {baseline_task_id}",
        expected_task_id=baseline_task_id,
        expected_model_id=baseline_model_id,
    )
    checkpoint, checkpoint_contract = _download_model_checkpoint(
        model,
        model_name=model_name,
        label=f"{baseline} controlled baseline",
        expected_sha256=baseline_checkpoint_sha256,
    )
    task.set_input_model(
        model_id=str(checkpoint_contract["model_id"]),
        name=f"{baseline}_final_checkpoint",
        update_task_design=False,
        update_task_labels=False,
    )

    runner = _load_source_training_runner(source_root)
    dataset_root, env = _prepare_experiment_environment(
        args,
        task_id=task_id,
        source_root=source_root,
        base_env=runtime_env,
        dataset_class=dataset_class,
        runner=runner,
    )
    work_dir = (
        source_root / "work_dirs/controlled_baseline_evaluation" / task_id / baseline
    )
    if work_dir.exists() or work_dir.is_symlink():
        raise FileExistsError(
            f"refusing to reuse evaluation work directory: {work_dir}"
        )
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    overlay_index = (
        dataset_root / "protocols/dair_v2/evaluation_overlays.json"
    ).resolve(strict=True)
    evaluator = _apply_controlled_evaluator_headless_compatibility(source_root)
    command = [
        str(python),
        str(evaluator),
        "--baseline",
        baseline,
        "--checkpoint",
        str(checkpoint),
        "--overlay-index",
        str(overlay_index),
        "--work-dir",
        str(work_dir),
        "--protocol-id",
        CANONICAL_1337_PROTOCOL_ID,
        "--expected-ground-truth-count",
        str(CANONICAL_1337_GROUND_TRUTH_COUNT),
    ]
    contract = {
        "schema_version": 1,
        "mode": "baseline_validate",
        "task_id": task_id,
        "baseline": baseline,
        "baseline_task_id": baseline_task_id,
        "predecessor_task_id": predecessor_task_id,
        "training_dataset_id": args.training_dataset_id,
        "checkpoint": checkpoint_contract,
        "overlay_index": str(overlay_index),
        "overlay_index_sha256": _sha256(overlay_index),
        "expected_delays_ms": [0, 100, 200, 300],
        "expected_conditions": ["Full", "L-Fail", "C-Fail"],
        "expected_run_count": 12,
        "protocol_id": CANONICAL_1337_PROTOCOL_ID,
        "expected_sample_count": CANONICAL_1337_SAMPLE_COUNT,
        "expected_ground_truth_count": CANONICAL_1337_GROUND_TRUTH_COUNT,
        "expected_manifest_content_sha256": (CANONICAL_1337_MANIFEST_CONTENT_SHA256),
        "expected_overlay_index_content_sha256": (
            CANONICAL_1337_OVERLAY_INDEX_CONTENT_SHA256
        ),
        "expected_sample_ids_sha256": CANONICAL_1337_SAMPLE_IDS_SHA256,
        "command": command,
        "evaluator": str(evaluator),
        "evaluator_sha256": _sha256(evaluator),
        "dataset_root": str(dataset_root),
    }
    if not task.upload_artifact(
        "run_contract", artifact_object=contract, wait_on_upload=True
    ):
        raise RuntimeError("failed to upload baseline-validation run contract")
    _run_logged(command, cwd=source_root, env=env)

    evidence_stage = _stage_controlled_baseline_evidence(
        work_dir=work_dir,
    )
    evaluation_plan, metrics = _verify_controlled_evidence_stage(evidence_stage)
    expected_evidence = {
        "protocol_id": CANONICAL_1337_PROTOCOL_ID,
        "manifest_content_sha256": CANONICAL_1337_MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": (CANONICAL_1337_OVERLAY_INDEX_CONTENT_SHA256),
        "sample_ids_sha256": CANONICAL_1337_SAMPLE_IDS_SHA256,
        "expected_sample_count": CANONICAL_1337_SAMPLE_COUNT,
        "expected_ground_truth_count": CANONICAL_1337_GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": 0,
    }
    for document_name, document in (
        ("evaluation plan", evaluation_plan),
        ("metrics", metrics),
    ):
        for field, expected_value in expected_evidence.items():
            if document.get(field) != expected_value:
                raise RuntimeError(
                    f"controlled baseline {document_name} {field} mismatch: "
                    f"expected {expected_value!r}, got {document.get(field)!r}"
                )
    runs = metrics.get("runs")
    if metrics.get("complete") is not True or metrics.get("planned_run_count") != 12:
        raise RuntimeError("controlled baseline evaluation did not complete 12 runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise RuntimeError("controlled baseline metrics must contain exactly 12 runs")
    expected = {
        (delay, condition)
        for delay in (0, 100, 200, 300)
        for condition in ("Full", "L-Fail", "C-Fail")
    }
    observed = {(run.get("delay_ms"), run.get("condition")) for run in runs}
    if observed != expected:
        raise RuntimeError("controlled baseline metrics matrix is incomplete")
    for run in runs:
        run_evidence = {
            "sample_count": CANONICAL_1337_SAMPLE_COUNT,
            "ground_truth_count": CANONICAL_1337_GROUND_TRUTH_COUNT,
            "sample_ids_sha256": CANONICAL_1337_SAMPLE_IDS_SHA256,
            "unsupported_sample_count": 0,
        }
        for field, expected_value in run_evidence.items():
            if run.get(field) != expected_value:
                raise RuntimeError(
                    f"controlled baseline run {field} mismatch: "
                    f"expected {expected_value!r}, got {run.get(field)!r}"
                )
    _upload_controlled_baseline_artifacts(
        task,
        evidence_stage=evidence_stage,
    )
    return 0


def _execute_experiment_from_task(
    args: argparse.Namespace,
    *,
    source_root: Path,
    python: Path,
    runtime_env: Mapping[str, str],
    dataset_class: object,
    task_class: object,
    output_model_class: object,
) -> int:
    """Run one suite member without entering the legacy stage/evaluation runner."""

    spec = EXPERIMENT_BY_NAME[args.experiment_from_task]
    task = _current_or_init_experiment_task(task_class, spec)
    task.output_uri = FILES_SERVER_URI
    existing_tags = list(task.get_tags())
    task.set_tags(
        list(
            dict.fromkeys(
                existing_tags
                + [
                    "ResilientV2X-suite",
                    spec.kind,
                    spec.name,
                    "4gpu",
                    "RTX5090",
                    "sm120",
                    "FP32",
                    "DDP",
                ]
            )
        )
    )

    task_id = str(getattr(task, "id", "") or "")
    if CLEARML_TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise RuntimeError(f"current ClearML task has an invalid ID: {task_id!r}")
    predecessor_task = task_class.get_task(task_id=args.predecessor_task_id)
    _require_completed_task(
        predecessor_task,
        context=f"predecessor task {args.predecessor_task_id}",
    )
    if args.predecessor_task_id == task_id:
        raise RuntimeError("an experiment task cannot be its own predecessor")

    runner = _load_source_training_runner(source_root)
    dataset_root, env = _prepare_experiment_environment(
        args,
        task_id=task_id,
        source_root=source_root,
        base_env=runtime_env,
        dataset_class=dataset_class,
        runner=runner,
    )
    teacher_contract = _prepare_teacher_handoff(
        spec=spec,
        teacher_task_id=args.teacher_task_id,
        task=task,
        task_class=task_class,
        env=env,
        expected_model_id=args.teacher_model_id,
        expected_checkpoint_sha256=args.teacher_checkpoint_sha256,
        allow_failed_task=args.allow_failed_teacher_task,
        expected_dataset_id=args.training_dataset_id,
    )

    work_dir = source_root / "work_dirs/clearml_suite" / task_id / spec.name
    if work_dir.exists() or work_dir.is_symlink():
        raise FileExistsError(
            f"refusing to reuse experiment work directory: {work_dir}"
        )
    work_dir.parent.mkdir(parents=True, exist_ok=True)

    config_path = (
        (source_root / spec.config).resolve(strict=True)
        if spec.config is not None
        else None
    )
    baseline_plan: Mapping[str, object] | None = None
    if spec.kind == "baseline":
        training_index = (
            dataset_root / "protocols/dair_v2/training_overlays.json"
        ).resolve(strict=True)
        plan_command = _baseline_plan_command(
            python,
            source_root=source_root,
            baseline=spec.name,
            training_index=training_index,
            work_dir=work_dir,
            training_seed=args.training_seed,
        )
        _run_logged(
            plan_command,
            cwd=source_root,
            env=env,
            capture=True,
        )
        plan_path, config_path, baseline_plan = _validate_baseline_dry_run(
            spec=spec,
            work_dir=work_dir,
            training_seed=args.training_seed,
        )
        if not task.upload_artifact(
            "baseline_dry_run_plan",
            artifact_object=str(plan_path),
            wait_on_upload=True,
        ):
            raise RuntimeError("failed to upload the baseline dry-run plan")
        if not task.upload_artifact(
            "baseline_resolved_config",
            artifact_object=str(config_path),
            wait_on_upload=True,
        ):
            raise RuntimeError("failed to upload the baseline resolved config")
    else:
        work_dir.mkdir(parents=True)

    if config_path is None:
        raise AssertionError("experiment config was not resolved")
    training_command = _ddp_training_command(
        python,
        gpus=args.gpus,
        config=config_path,
        work_dir=work_dir,
        max_epochs=args.max_epochs,
        training_seed=args.training_seed,
    )
    contract = _experiment_run_contract(
        args,
        task_id=task_id,
        spec=spec,
        dataset_root=dataset_root,
        teacher_contract=teacher_contract,
        predecessor_task_id=args.predecessor_task_id,
        config_path=config_path,
        training_command=training_command,
        baseline_plan=baseline_plan,
    )
    if not task.upload_artifact(
        "run_contract",
        artifact_object=contract,
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to upload the experiment run contract")
    _run_logged(training_command, cwd=source_root, env=env)

    if teacher_contract is None:
        raise AssertionError("training completed without a clean-teacher handoff")
    teacher_checkpoint_sha256 = teacher_contract.get("sha256")
    if type(teacher_checkpoint_sha256) is not str:
        raise AssertionError("clean-teacher handoff lost its checkpoint SHA-256")
    _validate_and_upload_common_teacher_initialization_audit(
        task,
        work_dir=work_dir,
        expected_teacher_sha256=teacher_checkpoint_sha256,
        expect_nested_teacher=spec.name in NESTED_TEACHER_EXPERIMENTS,
    )

    checkpoint = work_dir / f"epoch_{args.max_epochs}.pth"
    if checkpoint.is_symlink():
        raise ValueError(f"final checkpoint must not be a symlink: {checkpoint}")
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"final epoch checkpoint is missing: {checkpoint}")
    if checkpoint.stat().st_size <= 0:
        raise ValueError(f"final epoch checkpoint is empty: {checkpoint}")
    checkpoint_contract = _upload_experiment_checkpoint(
        task=task,
        output_model_class=output_model_class,
        spec=spec,
        checkpoint=checkpoint,
    )
    if not task.upload_artifact(
        "final_checkpoint_contract",
        artifact_object=checkpoint_contract,
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to upload the final checkpoint contract")
    best_checkpoint, best_epoch = _resolve_clean_val_best_checkpoint(work_dir)
    best_contract = _upload_experiment_best_checkpoint(
        task=task,
        output_model_class=output_model_class,
        spec=spec,
        checkpoint=best_checkpoint,
        epoch=best_epoch,
    )
    if not task.upload_artifact(
        "best_checkpoint_contract",
        artifact_object=best_contract,
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to upload the clean-val best checkpoint contract")
    task.flush(wait_for_uploads=True)
    return 0


def _runner_command(
    args: argparse.Namespace,
    *,
    python: Path,
    runner: Path,
    teacher_checkpoint: Path | None = None,
    student_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        str(python),
        str(runner),
        "--runtime-profile",
        "rtx5090",
        "--dataset-id",
        args.training_dataset_id,
        "--gpus",
        str(args.gpus),
        "--stage",
        args.stage,
        "--max-epochs",
        str(args.max_epochs),
        "--training-seed",
        str(getattr(args, "training_seed", DEFAULT_TRAINING_SEED)),
    ]
    teacher_value = (
        teacher_checkpoint
        if teacher_checkpoint is not None
        else args.teacher_checkpoint
    )
    student_value = (
        student_checkpoint
        if student_checkpoint is not None
        else args.student_checkpoint
    )
    if teacher_value is not None:
        command.extend(["--teacher-checkpoint", str(teacher_value)])
    if student_value is not None:
        command.extend(["--student-checkpoint", str(student_value)])
    if args.amp:
        command.append("--amp")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_arguments(args)
    _compile_embedded_smoke_scripts()
    _assert_base_image()
    print(
        json.dumps(
            _capture_gpu_memory_preflight(args.gpus),
            sort_keys=True,
        ),
        flush=True,
    )
    _validate_gpu_runtime(_capture_gpu_runtime())
    _pin_hostname_to_loopback()

    from clearml import Dataset, OutputModel, Task

    build_task = Task.get_task(task_id=BUILD_TASK_ID)
    artifacts = _require_completed_build_task(build_task)
    manifest_path = _verify_file(
        _artifact_path(artifacts[BUILD_MANIFEST_ARTIFACT], BUILD_MANIFEST_ARTIFACT),
        expected_sha256=args.build_manifest_sha256,
    )
    manifest = _read_json_object(manifest_path)
    _validate_build_manifest(manifest, args)
    bundle_path = _verify_file(
        _artifact_path(artifacts[NATIVE_BUNDLE_ARTIFACT], NATIVE_BUNDLE_ARTIFACT),
        expected_bytes=args.native_bundle_bytes,
        expected_sha256=args.native_bundle_sha256,
    )

    source_dataset = Dataset.get(
        dataset_id=args.source_dataset_id,
        only_completed=True,
    )
    source_copy = Path(source_dataset.get_local_copy()).resolve(strict=True)
    source_archive = _verify_file(
        source_copy / "source" / args.source_archive_name,
        expected_bytes=args.source_archive_bytes,
        expected_sha256=args.source_archive_sha256,
    )

    source_root = _safe_extract_zstd(source_archive, WORKSPACE)
    _validate_native_build_inputs(source_root)
    print(
        json.dumps(
            {
                "event": "native_bundle_python_source_compatibility_pass",
                "native_build_source_dataset_id": NATIVE_BUILD_SOURCE_DATASET_ID,
                "runtime_source_dataset_id": args.source_dataset_id,
                "runtime_source_archive_sha256": args.source_archive_sha256,
                "native_build_inputs": len(NATIVE_BUILD_INPUT_SHA256),
                "python_only_changed_paths": list(
                    NATIVE_COMPATIBLE_PYTHON_ONLY_CHANGES
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _apply_source_runner_metrics_compatibility(source_root)
    bundle_root = _safe_extract_tar(
        bundle_path,
        Path("/tmp/resilient-v2x-5090-native-bundle"),
    )
    overlay_root, _, extensions = _verify_bundle_layout(bundle_root, manifest)

    base_env = dict(os.environ)
    venv_python = _create_runtime_venv(VENV_ROOT, base_env)
    runtime_env = _runtime_environment(
        base_env,
        venv_root=VENV_ROOT,
        source_root=source_root,
    )
    site_packages = _venv_site_packages(venv_python, runtime_env)
    shutil.copytree(overlay_root, site_packages, dirs_exist_ok=True)

    for extension, record in extensions:
        destination = source_root.joinpath(*PurePosixPath(str(record["path"])).parts)
        try:
            destination.resolve(strict=False).relative_to(
                source_root.resolve(strict=True)
            )
        except ValueError as error:
            raise ValueError(
                f"extension destination escaped source: {destination}"
            ) from error
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite source extension: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extension, destination)
        _verify_file(
            destination,
            expected_bytes=int(record["bytes"]),
            expected_sha256=str(record["sha256"]),
        )

    _run_smoke(
        venv_python,
        RUNTIME_NATIVE_SMOKE,
        expected_event="rtx5090_native_runtime_smoke_pass",
        source_root=source_root,
        env=runtime_env,
    )
    _run_smoke(
        venv_python,
        MODEL_SMOKE,
        expected_event="rtx5090_teacher_model_smoke_pass",
        source_root=source_root,
        env=runtime_env,
    )

    runner = (source_root / "tools/resilient_v2x/clearml_train.py").resolve(strict=True)
    help_result = _run(
        [str(venv_python), str(runner), "--help"],
        cwd=source_root,
        env=runtime_env,
        capture=True,
    )
    if "--runtime-profile" not in help_result.stdout:
        raise RuntimeError("source runner does not support the RTX5090 runtime profile")

    runtime_env.update(
        {
            "RESILIENT_V2X_5090_BUILD_TASK_ID": BUILD_TASK_ID,
            "RESILIENT_V2X_5090_NATIVE_BUNDLE_SHA256": args.native_bundle_sha256,
            "RESILIENT_V2X_5090_BUILD_MANIFEST_SHA256": args.build_manifest_sha256,
            "RESILIENT_V2X_5090_BASE_IMAGE_CONFIG_DIGEST": BASE_IMAGE_CONFIG_DIGEST,
        }
    )
    if args.experiment_from_task is not None:
        return _execute_experiment_from_task(
            args,
            source_root=source_root,
            python=venv_python,
            runtime_env=runtime_env,
            dataset_class=Dataset,
            task_class=Task,
            output_model_class=OutputModel,
        )
    if args.stage == "baseline_validate":
        return _execute_controlled_baseline_validation(
            args,
            source_root=source_root,
            python=venv_python,
            runtime_env=runtime_env,
            dataset_class=Dataset,
            task_class=Task,
        )

    teacher_checkpoint = args.teacher_checkpoint
    student_checkpoint = args.student_checkpoint
    if args.teacher_task_id is not None or args.student_task_id is not None:
        task = Task.current_task()
        if task is None:
            raise RuntimeError(
                "remote checkpoint handoff requires a current ClearML task"
            )
        task.output_uri = FILES_SERVER_URI
        (
            teacher_checkpoint,
            student_checkpoint,
            handoff_contract,
        ) = _prepare_remote_checkpoint_handoffs(
            args,
            task=task,
            task_class=Task,
        )
        if not task.upload_artifact(
            "checkpoint_handoff_contract",
            artifact_object=handoff_contract,
            wait_on_upload=True,
        ):
            raise RuntimeError("failed to upload checkpoint handoff contract")
        task.flush(wait_for_uploads=True)

    command = _runner_command(
        args,
        python=venv_python,
        runner=runner,
        teacher_checkpoint=teacher_checkpoint,
        student_checkpoint=student_checkpoint,
    )
    os.chdir(source_root)
    os.execve(str(venv_python), command, runtime_env)
    raise AssertionError("os.execve unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
