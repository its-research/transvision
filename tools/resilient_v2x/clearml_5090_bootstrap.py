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
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Mapping, NamedTuple, Sequence
from urllib.parse import unquote, urlsplit


BASE_IMAGE_AMD64_MANIFEST_DIGEST = (
    "sha256:dbc586035fffb2bc030e807290d43e8d4edf44ee864fa5c832db44ed099fc415"
)
BASE_IMAGE_CONFIG_DIGEST = (
    "sha256:3812e520c0e86bb621878970370f52cbacaa32921bf0e4b2ae6a2028a5cf95fb"
)
BUILD_TASK_ID = "86a3ee30dcc749408ba19ba2088adb4c"
NATIVE_BUILD_SOURCE_DATASET_ID = "bcbd15ae7e454e9885bc4250a3de774e"
NATIVE_BUILD_SOURCE_ARCHIVE_NAME = (
    "resilient-v2x-source-cf61ef3c5432.tar.zst"
)
NATIVE_BUILD_SOURCE_ARCHIVE_BYTES = 1142675
NATIVE_BUILD_SOURCE_ARCHIVE_SHA256 = (
    "b20eccf308934eae82bf0bf032d0baa2c795d57efa6e55fe731501d50e5d7289"
)
NATIVE_COMPATIBLE_PYTHON_ONLY_CHANGES = (
    "configs/resilient_v2x/dair_clean_teacher.py",
    "configs/resilient_v2x/dair_vehicle_pretrain.py",
    "tests/resilient_v2x/test_clearml_paper_controller.py",
    "tests/resilient_v2x/test_clearml_suite.py",
    "tests/resilient_v2x/test_clearml_workflow.py",
    "tests/resilient_v2x/test_configs.py",
    "tests/resilient_v2x/test_controlled_baseline_detectors.py",
    "tests/resilient_v2x/test_detection_evaluation.py",
    "tests/resilient_v2x/test_resilient_v2x_metric.py",
    "tools/resilient_v2x/clearml_5090_bootstrap.py",
    "tools/resilient_v2x/clearml_5090_paper_controller.py",
    "tools/resilient_v2x/clearml_train.py",
    "transvision/evaluation/metrics/resilient_v2x_metric.py",
    "transvision/evaluation/resilient_v2x_detection.py",
    "transvision/models/detectors/__init__.py",
    "transvision/models/detectors/resilient_v2x.py",

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
FILES_SERVER_URI = (
    f"http://{EXPECTED_FILES_SERVER_HOST}:{EXPECTED_FILES_SERVER_PORT}"
)
EXPERIMENT_MAX_EPOCHS = 50
RTX5090_TRAIN_BATCH_SIZE_PER_GPU = 2
RTX5090_EVAL_BATCH_SIZE_PER_GPU = 4
CLEARML_TRAIN_BASELINE_SHA256 = (
    "6ea88906c9a9c8bd5122d23f78306edcbb92ada629e0189df6e6817824c2f631"
)
CLEARML_TRAIN_METRICS_COMPAT_SHA256 = (
    "52cb5c508000ab96333b6e7ce5ba490587f144f3b97ea148974be87716a629a5"
)
CLEARML_TRAIN_METRICS_REPLACEMENTS = (
    (
        '''    candidates = sorted(work_dir.rglob("scalars.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one scalars.json under {work_dir}, found {len(candidates)}"
        )
''',
        '''    candidates = sorted(work_dir.rglob("scalars.json"))
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
''',
    ),
    (
        '''    metric_rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        scalars.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
''',
        '''    metric_rows: list[dict[str, object]] = []
    metric_content = scalars.read_text(encoding="utf-8")
    metric_lines = (
        metric_content.splitlines()
        if metric_source == "scalars.json"
        else [metric_content]
    )
    for line_number, line in enumerate(metric_lines, start=1):
''',
    ),
)


class ExperimentSpec(NamedTuple):
    """One immutable member of the post-main training suite."""

    name: str
    kind: str
    config: str | None
    requires_teacher: bool


EXPERIMENT_SPECS = (
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
        False,
    ),
    ExperimentSpec(
        "concat_capacity_matched",
        "ablation",
        "configs/resilient_v2x/ablations/concat_capacity_matched.py",
        True,
    ),
    ExperimentSpec("v2x_vit", "baseline", None, False),
    ExperimentSpec("cobevt", "baseline", None, False),
    ExperimentSpec("coformernet", "baseline", None, False),
    ExperimentSpec("bevfusion", "baseline", None, False),
    ExperimentSpec("ffnet", "baseline", None, False),
)
EXPERIMENT_ORDER = tuple(spec.name for spec in EXPERIMENT_SPECS)
EXPERIMENT_BY_NAME = {spec.name: spec for spec in EXPERIMENT_SPECS}
TEACHER_DEPENDENT_EXPERIMENTS = frozenset(
    spec.name for spec in EXPERIMENT_SPECS if spec.requires_teacher
)
BASELINE_EXPERIMENTS = frozenset(
    spec.name for spec in EXPERIMENT_SPECS if spec.kind == "baseline"
)
RTX5090_HEADLESS_CFG_OPTIONS = (
    "visualizer._scope_=mmengine",
    "visualizer.type=Visualizer",
    "visualizer.vis_backends.0._scope_=mmengine",
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
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
    parser.add_argument("--gpus", type=int, choices=(4,), default=4)
    parser.add_argument(
        "--stage",
        choices=("all", "vehicle", "vehicle_teacher", "teacher", "student", "validate"),
        default="all",
    )
    parser.add_argument("--max-epochs", type=_positive_integer, default=50)
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
        help=(
            "completed task owning the distilled-student OutputModel "
            "for validation"
        ),
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
                "--allow-failed-teacher-task requires a complete remote "
                "teacher handoff"
            )
        if args.stage in ("all", "vehicle", "vehicle_teacher", "teacher"):
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
    elif any(
        value is not None
        for value in (
            teacher_task_id,
            args.teacher_model_id,
            args.teacher_checkpoint_sha256,
        )
    ) or args.allow_failed_teacher_task:
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
        raise ValueError(f"source training runner escaped source root: {target}") from error

    original = target.read_bytes()
    actual_sha256 = hashlib.sha256(original).hexdigest()
    if actual_sha256 == CLEARML_TRAIN_METRICS_COMPAT_SHA256:
        return target
    if actual_sha256 != CLEARML_TRAIN_BASELINE_SHA256:
        raise ValueError(
            "source training runner identity does not match the sealed baseline: "
            f"{actual_sha256}"
        )

    patched = original.decode("utf-8")
    for old, new in CLEARML_TRAIN_METRICS_REPLACEMENTS:
        if patched.count(old) != 1:
            raise ValueError("source training runner compatibility anchor is not unique")
        patched = patched.replace(old, new)
    encoded = patched.encode("utf-8")
    patched_sha256 = hashlib.sha256(encoded).hexdigest()
    if patched_sha256 != CLEARML_TRAIN_METRICS_COMPAT_SHA256:
        raise ValueError(
            "source training runner compatibility result has an invalid identity: "
            f"{patched_sha256}"
        )

    temporary = target.with_name(f"{target.name}.metrics-compat.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"compatibility temporary path already exists: {temporary}")
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


def _validate_gpu_runtime(contract: Mapping[str, object]) -> None:
    expected = {
        "python": [3, 12],
        "torch": EXPECTED_TORCH,
        "torch_cuda": EXPECTED_TORCH_CUDA,
        "gpu_count": EXPECTED_GPU_COUNT,
        "cuda_available": True,
    }
    for field, expected_value in expected.items():
        if contract.get(field) != expected_value:
            raise RuntimeError(
                f"RTX5090 runtime {field} mismatch: "
                f"expected {expected_value!r}, got {contract.get(field)!r}"
            )
    capabilities = contract.get("capabilities")
    if capabilities != [list(EXPECTED_CAPABILITY)] * EXPECTED_GPU_COUNT:
        raise RuntimeError(f"RTX5090 GPU capability mismatch: {capabilities!r}")
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


def _runtime_environment(
    base_env: Mapping[str, str],
    *,
    venv_root: Path,
    source_root: Path,
) -> dict[str, str]:
    env = dict(base_env)
    force_weights_only = str(
        env.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "")
    ).strip().lower()
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
if any(torch.cuda.get_device_capability(i) != (12, 0) for i in range(4)):
    raise RuntimeError("all GPUs must have compute capability 12.0")
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
        model
        for model in output_models
        if getattr(model, "name", None) == model_name
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
                "failed teacher task checkpoint filename mismatch: "
                f"{source_filename!r}"
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
    runner_path = (
        source_root / "tools/resilient_v2x/clearml_train.py"
    ).resolve(strict=True)
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
        f"train_dataloader.batch_size={RTX5090_TRAIN_BATCH_SIZE_PER_GPU}",
        f"val_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        f"test_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        *RTX5090_HEADLESS_CFG_OPTIONS,
    ]


def _baseline_plan_command(
    python: Path,
    *,
    source_root: Path,
    baseline: str,
    training_index: Path,
    work_dir: Path,
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
        "20250218",
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


def _prepare_experiment_environment(
    args: argparse.Namespace,
    *,
    source_root: Path,
    base_env: Mapping[str, str],
    dataset_class: object,
    runner: ModuleType,
) -> tuple[Path, dict[str, str]]:
    getter = getattr(dataset_class, "get", None)
    if not callable(getter):
        raise RuntimeError("ClearML Dataset class has no get method")
    dataset = getter(dataset_id=args.training_dataset_id, only_completed=True)
    dataset_root = Path(dataset.get_local_copy()).resolve(strict=True)
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
    return dataset_root, runner._overlay_environment(dataset_root, env)


def _validate_baseline_dry_run(
    *,
    spec: ExperimentSpec,
    work_dir: Path,
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
        raise ValueError(f"baseline dry-run source config is invalid: {error}") from error
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
    resolved_config_sha256 = _sha256(config_path)
    if baseline_plan is None:
        declared_config = str(config_path)
        declared_config_sha256 = resolved_config_sha256
    else:
        declared_config = str(baseline_plan["baseline_config"])
        declared_config_sha256 = str(baseline_plan["baseline_config_sha256"])
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
        "seed": 20250218,
        "learning_rate": 0.0001,
        "auto_scale_lr": False,
        "amp": False,
        "precision": "FP32",
        "runtime_profile": "rtx5090",
        "per_epoch_validation": True,
        "condition_evaluation": False,
        "condition_evaluation_reason": (
            "the 12-condition matrix remains outside individual training tasks"
        ),
        "teacher": dict(teacher_contract) if teacher_contract is not None else None,
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
        raise FileExistsError(f"refusing to reuse experiment work directory: {work_dir}")
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
    _validate_gpu_runtime(_capture_gpu_runtime())

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
