#!/usr/bin/env python3
"""Build and exercise the native ResilientV2X stack for four RTX 5090 GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Sequence


TOOLCHAIN_DATASET_ID = "7c63226cec1a4acfa410b5abe5df4d69"
TOOLCHAIN_ARCHIVE = "resilient-v2x-cuda128-toolchain-12.8.61.tar.zst"
TOOLCHAIN_ARCHIVE_BYTES = 1_390_955_440
TOOLCHAIN_ARCHIVE_SHA256 = (
    "70a1e3bda2bf87e0a2a8230ec56daf55498bdfa938a36f7ce29b22c8610cf90b"
)
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_GPU_COUNT = 4
ALLOWED_BUILD_GPU_COUNTS = frozenset({4, 8})
EXPECTED_CAPABILITY = (12, 0)
ALLOWED_BUILD_CAPABILITIES = frozenset({(12, 0), (8, 0), (7, 0)})
DEFAULT_TORCH_CUDA_ARCH_LIST = "7.0;8.0;12.0"
BASE_IMAGE_MANIFEST_DIGEST = (
    "sha256:dbc586035fffb2bc030e807290d43e8d4edf44ee864fa5c832db44ed099fc415"
)
BASE_IMAGE_CONFIG_DIGEST = (
    "sha256:3812e520c0e86bb621878970370f52cbacaa32921bf0e4b2ae6a2028a5cf95fb"
)
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


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _plain_filename(value: str) -> str:
    path = Path(value)
    if not value or value in {".", ".."} or path.is_absolute() or path.name != value:
        raise argparse.ArgumentTypeError("value must be a plain filename")
    return value


def _sha256_argument(value: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256")
    return value


def _normalize_torch_cuda_arch_list(value: str) -> str:
    # Accept comma or semicolon separators so ClearML CLI argv cannot be split by ';'.
    normalized_input = str(value).replace(",", ";")
    parts = [part.strip() for part in normalized_input.split(";") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "torch CUDA arch list must contain at least one arch"
        )
    normalized: list[str] = []
    for part in parts:
        if not re.fullmatch(r"\d+\.\d+", part):
            raise argparse.ArgumentTypeError(
                f"invalid CUDA arch entry {part!r}; expected forms like 7.0"
            )
        if part not in normalized:
            normalized.append(part)
    return ";".join(normalized)


def _required_nvcc_compute_archs(arch_list: str) -> frozenset[str]:
    mapping = {
        "7.0": "compute_70",
        "8.0": "compute_80",
        "12.0": "compute_120",
    }
    required = {mapping[part] for part in arch_list.split(";") if part in mapping}
    if not required:
        raise ValueError(f"unsupported TORCH_CUDA_ARCH_LIST: {arch_list!r}")
    return frozenset(required)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-id", required=True)
    parser.add_argument("--source-archive-name", type=_plain_filename, required=True)
    parser.add_argument("--source-archive-bytes", type=_positive_integer, required=True)
    parser.add_argument("--source-archive-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--toolchain-dataset-id", default=TOOLCHAIN_DATASET_ID)
    parser.add_argument(
        "--torch-cuda-arch-list",
        type=_normalize_torch_cuda_arch_list,
        default=DEFAULT_TORCH_CUDA_ARCH_LIST,
        help=(
            "CUDA arches for MMCV/project extensions as comma- or semicolon-separated "
            f"values (default: {DEFAULT_TORCH_CUDA_ARCH_LIST.replace(';', ',')})"
        ),
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    for field in ("source_dataset_id", "toolchain_dataset_id"):
        value = getattr(args, field)
        if type(value) is not str or not value.strip():
            raise ValueError(f"--{field.replace('_', '-')} must be non-empty")
    _plain_filename(args.source_archive_name)
    if type(args.source_archive_bytes) is not int or args.source_archive_bytes <= 0:
        raise ValueError("--source-archive-bytes must be positive")
    if (
        type(args.source_archive_sha256) is not str
        or SHA256_PATTERN.fullmatch(args.source_archive_sha256) is None
    ):
        raise ValueError("--source-archive-sha256 must be a lowercase SHA-256")
    args.torch_cuda_arch_list = _normalize_torch_cuda_arch_list(
        args.torch_cuda_arch_list
    )


def _assert_base_image() -> None:
    actual = os.environ.get("RESILIENT_V2X_CONTAINER_IMAGE_DIGEST")
    if actual != BASE_IMAGE_MANIFEST_DIGEST:
        raise RuntimeError(
            "RTX5090 base image digest mismatch: "
            f"expected {BASE_IMAGE_MANIFEST_DIGEST!r}, got {actual!r}"
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
    expected_bytes: int,
    expected_sha256: str,
) -> Path:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"file size mismatch: {path}")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"file SHA-256 mismatch: {path}")
    return path


def _require_new_directory(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {path}")
    path.mkdir(parents=True)
    return path


def _extract_zstd(archive: Path, destination: Path) -> None:
    import zstandard

    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as bundle:
                bundle.extractall(destination, filter="data")


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {shlex.join(command)}", flush=True)
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def _pip_install(
    python: Path,
    packages: Sequence[str],
    env: dict[str, str],
) -> None:
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *packages,
        ],
        env=env,
    )


def _probe_gpu_runtime(
    python: Path,
    env: dict[str, str],
) -> dict[str, object]:
    script_path = Path(__file__).resolve(strict=True)
    code = f"""
import importlib.util
import json

spec = importlib.util.spec_from_file_location("clearml_5090_build_probe", {str(script_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(module._assert_gpu_runtime(), sort_keys=True))
"""
    output = _run(
        [str(python), "-c", code],
        env=env,
        capture=True,
    ).stdout
    payload = json.loads(output.strip().splitlines()[-1])
    if not isinstance(payload, dict):
        raise RuntimeError("runtime probe did not return a JSON object")
    return payload


def _overlay_stats(root: Path) -> dict[str, int]:
    root = root.resolve(strict=True)
    files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"site-packages overlay contains a symlink: {path}")
        if path.is_file():
            stat = path.stat(follow_symlinks=False)
            if stat.st_nlink != 1:
                raise ValueError(f"site-packages overlay contains a hardlink: {path}")
            files += 1
            total_bytes += stat.st_size
        elif not path.is_dir():
            raise ValueError(f"site-packages overlay contains a special file: {path}")
    if files == 0 or total_bytes == 0:
        raise RuntimeError("site-packages overlay is empty")
    return {"file_count": files, "bytes": total_bytes}


def _verify_bundle_member_types(archive: Path) -> None:
    observed: set[str] = set()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
    if not members:
        raise RuntimeError("native runtime bundle is empty")
    for member in members:
        if member.name in observed:
            raise ValueError(f"duplicate native bundle member: {member.name!r}")
        observed.add(member.name)
        if not (member.isfile() or member.isdir()):
            raise ValueError(
                f"native bundle contains a link or special file: {member.name!r}"
            )


def _assert_gpu_runtime(*, arch_list: str = DEFAULT_TORCH_CUDA_ARCH_LIST) -> dict[str, object]:
    import torch

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"expected Python 3.12, found {sys.version}")
    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(
            f"expected torch {EXPECTED_TORCH}, found {torch.__version__}"
        )
    if torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"expected torch CUDA {EXPECTED_TORCH_CUDA}, found {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu_count = torch.cuda.device_count()
    if gpu_count not in ALLOWED_BUILD_GPU_COUNTS:
        raise RuntimeError(
            f"expected {sorted(ALLOWED_BUILD_GPU_COUNTS)} GPUs, found {gpu_count}"
        )
    arch_to_capability = {
        "7.0": (7, 0),
        "8.0": (8, 0),
        "12.0": (12, 0),
    }
    allowed_host = {
        arch_to_capability[part]
        for part in arch_list.split(";")
        if part in arch_to_capability
    } & ALLOWED_BUILD_CAPABILITIES
    if not allowed_host:
        raise RuntimeError(f"no allowed host capability for arch list {arch_list!r}")
    devices = []
    for index in range(gpu_count):
        capability = torch.cuda.get_device_capability(index)
        if capability not in allowed_host:
            raise RuntimeError(
                f"GPU {index} capability {capability} not in {sorted(allowed_host)} "
                f"for TORCH_CUDA_ARCH_LIST={arch_list!r}"
            )
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(capability),
            }
        )
    if len({tuple(device["capability"]) for device in devices}) != 1:
        raise RuntimeError(f"build host GPUs must be homogeneous: {devices!r}")
    if "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError("PyTorch build does not contain sm_120")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "devices": devices,
    }


KERNEL_SMOKE = r"""
import json
import torch

from mmcv.ops import nms_rotated
from transvision.models.bev_pool.bev_pool import bev_pool
from transvision.models.voxel import Voxelization

boxes = torch.tensor(
    [[0.0, 0.0, 2.0, 1.0, 0.0], [0.1, 0.1, 2.0, 1.0, 0.0]],
    device="cuda",
)
scores = torch.tensor([0.9, 0.8], device="cuda")
detections, indices = nms_rotated(boxes, scores, 0.5)
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
    [
        [0.1, 0.1, 0.1, 1.0],
        [0.2, 0.2, 0.2, 2.0],
        [1.1, 1.1, 1.1, 3.0],
    ],
    dtype=torch.float32,
    device="cuda",
)
voxels, coordinates, point_counts = voxelizer(points)
if not len(voxels) or not len(coordinates) or not len(point_counts):
    raise RuntimeError("voxelization returned an empty result")
torch.cuda.synchronize()
print(
    json.dumps(
        {
            "event": "native_kernel_smoke_pass",
            "nms_indices": indices.tolist(),
            "bev_pool_shape": list(pooled.shape),
            "voxel_shape": list(voxels.shape),
        },
        sort_keys=True,
    )
)
"""


MODEL_SMOKE = r"""
import json
import os
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
    raise RuntimeError("model has no parameters")
torch.cuda.synchronize()
print(
    json.dumps(
        {
            "event": "teacher_model_build_pass",
            "model": type(model).__name__,
            "visualizer": type(visualizer).__name__,
            "visualizer_backend": type(backends[0]).__name__,
            "scalars_json": True,
            "parameters": parameters,
        },
        sort_keys=True,
    )
)
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_arguments(args)
    if args.toolchain_dataset_id != TOOLCHAIN_DATASET_ID:
        raise ValueError("unexpected toolchain Dataset ID")
    _assert_base_image()

    from clearml import Dataset, Task

    arch_list = args.torch_cuda_arch_list
    task_name = (
        "RTX5090 native dependency build"
        if arch_list == "12.0"
        else f"RTX5090 native dependency build multiarch {arch_list.replace(';', '+')}"
    )
    task = Task.init(
        project_name="ResilientV2X/Training",
        task_name=task_name,
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
    )
    runtime = _assert_gpu_runtime(arch_list=arch_list)
    print(json.dumps({"event": "runtime_verified", **runtime}, sort_keys=True))

    source_dataset = Dataset.get(
        dataset_id=args.source_dataset_id,
        only_completed=True,
    )
    toolchain_dataset = Dataset.get(
        dataset_id=args.toolchain_dataset_id,
        only_completed=True,
    )
    source_copy = Path(source_dataset.get_local_copy()).resolve(strict=True)
    toolchain_copy = Path(toolchain_dataset.get_local_copy()).resolve(strict=True)
    source_archive = _verify_file(
        source_copy / "source" / args.source_archive_name,
        expected_bytes=args.source_archive_bytes,
        expected_sha256=args.source_archive_sha256,
    )
    toolchain_archive = _verify_file(
        toolchain_copy / "toolchain" / TOOLCHAIN_ARCHIVE,
        expected_bytes=TOOLCHAIN_ARCHIVE_BYTES,
        expected_sha256=TOOLCHAIN_ARCHIVE_SHA256,
    )

    source_root = _require_new_directory(Path("/workspace/resilient-v2x-5090"))
    cuda_home = _require_new_directory(Path("/opt/resilient-v2x-cuda-12.8"))
    _extract_zstd(source_archive, source_root)
    _extract_zstd(toolchain_archive, cuda_home)

    cuda_target = cuda_home / "targets" / "x86_64-linux"
    if not (cuda_home / "lib64").exists():
        (cuda_home / "lib64").symlink_to(cuda_target / "lib")
    base_env = dict(os.environ)
    env = dict(base_env)
    env.update(
        {
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CUDA_HOME": str(cuda_home),
            "CUDACXX": str(cuda_home / "bin" / "nvcc"),
            "FORCE_CUDA": "1",
            "MAX_JOBS": "8",
            "MMCV_WITH_OPS": "1",
            "TORCH_CUDA_ARCH_LIST": arch_list,
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONPATH": str(source_root),
            "PATH": f"{cuda_home / 'bin'}:{env.get('PATH', '')}",
            "CPATH": (f"{cuda_target / 'include'}:{env.get('CPATH', '')}").rstrip(":"),
            "LIBRARY_PATH": (
                f"{cuda_target / 'lib'}:{env.get('LIBRARY_PATH', '')}"
            ).rstrip(":"),
            "LD_LIBRARY_PATH": (
                f"{cuda_target / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
            ).rstrip(":"),
        }
    )
    nvcc = _run(
        [str(cuda_home / "bin" / "nvcc"), "--version"],
        env=env,
        capture=True,
    ).stdout
    architectures = _run(
        [str(cuda_home / "bin" / "nvcc"), "--list-gpu-arch"],
        env=env,
        capture=True,
    ).stdout.splitlines()
    architecture_text = "\n".join(architectures)
    required_compute = _required_nvcc_compute_archs(arch_list)
    required_cuda_files = (
        cuda_target / "include/cusparse.h",
        cuda_target / "include/cublas_v2.h",
        cuda_target / "include/cublasLt.h",
        cuda_target / "include/cusolverDn.h",
        cuda_target / "lib/libcusparse.so.12",
        cuda_target / "lib/libcublas.so.12",
        cuda_target / "lib/libcublasLt.so.12",
        cuda_target / "lib/libcusolver.so.11",
    )
    missing_compute = sorted(
        name for name in required_compute if name not in architecture_text
    )
    if (
        "release 12.8" not in nvcc
        or missing_compute
        or any(not path.is_file() for path in required_cuda_files)
    ):
        raise RuntimeError(
            "portable CUDA toolchain contract failed: "
            f"missing_compute={missing_compute}"
        )
    print(
        json.dumps(
            {
                "event": "toolchain_verified",
                "nvcc": nvcc.strip().splitlines()[-1],
                "torch_cuda_arch_list": arch_list,
                "required_compute": sorted(required_compute),
            },
            sort_keys=True,
        )
    )

    venv_root = _require_new_directory(Path("/opt/resilient-v2x-5090-build"))
    _run(
        [
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            str(venv_root),
        ],
        env=env,
    )
    runtime_python = venv_root / "bin/python"
    if not runtime_python.is_file():
        raise FileNotFoundError(f"venv Python is missing: {runtime_python}")
    env["PATH"] = f"{venv_root / 'bin'}:{env['PATH']}"
    env["PYTHONNOUSERSITE"] = "1"

    _pip_install(
        runtime_python,
        [
            "setuptools<81",
            "wheel",
            "ninja",
            "packaging",
            "numpy==1.26.4",
            "mmengine==0.10.7",
        ],
        env,
    )
    wheel_dir = _require_new_directory(Path("/tmp/resilient-v2x-wheels"))
    _run(
        [
            str(runtime_python),
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            "mmcv==2.1.0",
        ],
        env=env,
    )
    mmcv_wheels = list(wheel_dir.glob("mmcv-2.1.0-*.whl"))
    if len(mmcv_wheels) != 1:
        raise RuntimeError(f"expected one MMCV wheel, found {mmcv_wheels}")
    mmcv_wheel = mmcv_wheels[0]
    _pip_install(runtime_python, [str(mmcv_wheel)], env)
    _pip_install(
        runtime_python,
        [
            "mmdet==3.2.0",
            "mmdet3d==1.3.0",
            "fvcore==0.1.5.post20221221",
            "zstandard==0.22.0",
            "pypcd4==1.4.3",
            "jsonschema==4.23.0",
            "PyYAML==6.0.2",
        ],
        env,
    )

    runtime_env = dict(base_env)
    for variable in BUILD_ONLY_ENVIRONMENT_KEYS:
        runtime_env.pop(variable, None)
    runtime_env.pop("PYTHONHOME", None)
    runtime_env.update(
        {
            "VIRTUAL_ENV": str(venv_root),
            "PATH": f"{venv_root / 'bin'}:{base_env.get('PATH', '')}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(source_root),
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    if any("resilient-v2x-cuda" in value for value in runtime_env.values()):
        raise RuntimeError("runtime environment still references the build toolchain")

    runtime_after_install = _probe_gpu_runtime(runtime_python, runtime_env)
    print(
        json.dumps(
            {"event": "runtime_reverified", **runtime_after_install},
            sort_keys=True,
        )
    )

    _run(
        [str(runtime_python), "setup.py", "build_ext", "--inplace"],
        cwd=source_root,
        env=env,
    )
    _run(
        [str(runtime_python), "-c", KERNEL_SMOKE],
        cwd=source_root,
        env=runtime_env,
    )
    _run(
        [str(runtime_python), "-c", MODEL_SMOKE],
        cwd=source_root,
        env=runtime_env,
    )

    extensions = sorted(source_root.glob("transvision/models/**/*.so"))
    if len(extensions) != 2:
        raise RuntimeError(f"expected two project extensions, found {extensions}")
    site_packages = venv_root / "lib/python3.12/site-packages"
    overlay = _overlay_stats(site_packages)
    artifact_bundle = Path("/tmp/resilient-v2x-5090-runtime-overlay.tar.gz")
    with tarfile.open(artifact_bundle, "w:gz", dereference=True) as bundle:
        bundle.add(site_packages, arcname="site-packages")
        bundle.add(mmcv_wheel, arcname=f"wheels/{mmcv_wheel.name}")
        for extension in extensions:
            bundle.add(
                extension,
                arcname=str(extension.relative_to(source_root)),
            )
    _verify_bundle_member_types(artifact_bundle)

    freeze = _run(
        [str(runtime_python), "-m", "pip", "freeze", "--all"],
        env=runtime_env,
        capture=True,
    ).stdout
    freeze_path = Path("/tmp/resilient-v2x-5090-pip-freeze.txt")
    freeze_path.write_text(freeze)
    manifest = {
        "source_dataset_id": args.source_dataset_id,
        "source_archive_name": args.source_archive_name,
        "source_archive_bytes": args.source_archive_bytes,
        "source_archive_sha256": args.source_archive_sha256,
        "toolchain_dataset_id": TOOLCHAIN_DATASET_ID,
        "toolchain_archive_bytes": TOOLCHAIN_ARCHIVE_BYTES,
        "toolchain_archive_sha256": TOOLCHAIN_ARCHIVE_SHA256,
        "base_image": {
            "manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
            "platform": "linux/amd64",
            "config_digest": BASE_IMAGE_CONFIG_DIGEST,
        },
        "native_bundle": {
            "name": artifact_bundle.name,
            "bytes": artifact_bundle.stat().st_size,
            "sha256": _sha256(artifact_bundle),
        },
        "python_overlay": {
            "archive_root": "site-packages",
            "file_count": overlay["file_count"],
            "total_bytes": overlay["bytes"],
        },
        "mmcv_wheel": {
            "name": mmcv_wheel.name,
            "bytes": mmcv_wheel.stat().st_size,
            "sha256": _sha256(mmcv_wheel),
        },
        "extensions": [
            {
                "path": str(extension.relative_to(source_root)),
                "bytes": extension.stat().st_size,
                "sha256": _sha256(extension),
            }
            for extension in extensions
        ],
        "runtime": runtime_after_install,
        "torch_cuda_arch_list": env["TORCH_CUDA_ARCH_LIST"],
        "amp": False,
    }
    manifest_path = Path("/tmp/resilient-v2x-5090-build-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if not task.upload_artifact(
        "rtx5090_native_bundle",
        artifact_object=str(artifact_bundle),
        wait_on_upload=True,
    ):
        raise RuntimeError("native bundle upload failed")
    if not task.upload_artifact(
        "rtx5090_build_manifest",
        artifact_object=str(manifest_path),
        wait_on_upload=True,
    ):
        raise RuntimeError("build manifest upload failed")
    if not task.upload_artifact(
        "rtx5090_pip_freeze",
        artifact_object=str(freeze_path),
        wait_on_upload=True,
    ):
        raise RuntimeError("pip freeze upload failed")
    print(
        json.dumps(
            {"event": "rtx5090_native_build_pass", **manifest},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
