#!/usr/bin/env python3
"""Capture, persist, and validate the ResilientV2X runtime contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_DIR = ROOT / "environments" / "resilient_v2x"
RUNTIME_LOCK = ENVIRONMENT_DIR / "environment-linux-64.lock.yml"
BOOTSTRAP_LOCK = ENVIRONMENT_DIR / "bootstrap-linux-64.explicit.txt"
CONSTRAINTS = ENVIRONMENT_DIR / "constraints.txt"
SCHEMA = ENVIRONMENT_DIR / "environment-manifest.schema.json"
WORLD_SIZES = (1, 2, 4)
CUSTOM_OP_MODULES = (
    "mmcv._ext",
    "transvision.models.voxel.voxel_layer",
    "transvision.models.bev_pool.bev_pool_ext",
)


class EnvironmentMismatch(RuntimeError):
    """Captured runtime does not satisfy the requested environment contract."""


@dataclass(frozen=True)
class HardwareFingerprint:
    platform_system: str
    platform_machine: str
    cpu_model: str
    logical_cpu_count: int
    total_ram_bytes: int
    gpu_devices: tuple[Mapping[str, object], ...]
    cuda_driver: str
    cuda_runtime: str
    cudnn_version: str
    world_size: Literal[1, 2, 4]
    fingerprint_sha256: str


@dataclass(frozen=True)
class EnvironmentManifest:
    schema_version: Literal[1]
    classification: Literal["development", "controlled"]
    platform: Mapping[str, object]
    python: Mapping[str, object]
    packages: Mapping[str, str]
    cuda: Mapping[str, object]
    custom_ops: Mapping[str, str]
    determinism: Mapping[str, object]
    runtime_lock_sha256: str
    bootstrap_lock_sha256: str
    container_image_digest: str | None
    hardware: HardwareFingerprint
    actual_hardware_fingerprint_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class EnvironmentValidation:
    accepted: bool
    classification: Literal["development", "controlled"]
    environment_manifest_sha256: str
    actual_hardware_fingerprint_sha256: str
    verified_fields: tuple[str, ...]
    blocking_mismatches: tuple[str, ...]
    not_executed_checks: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _thaw(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EnvironmentMismatch(f"{field} must be a mapping")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise EnvironmentMismatch(f"{field} must be a string")
    return value


def _require_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EnvironmentMismatch(f"{field} must be an integer")
    return value


def _validate_world_size(world_size: int) -> Literal[1, 2, 4]:
    if world_size not in WORLD_SIZES:
        raise ValueError(f"world_size must be one of {WORLD_SIZES}, got {world_size!r}")
    return world_size


def _capture_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _capture_total_ram_bytes() -> int:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0
    return page_size * page_count


def _capture_platform() -> dict[str, object]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": _capture_cpu_model(),
        "logical_cpu_count": os.cpu_count() or 0,
        "total_ram_bytes": _capture_total_ram_bytes(),
    }


def _capture_python() -> dict[str, object]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def _constraint_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in CONSTRAINTS.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, version = stripped.split("==", 1)
        versions[name] = version
    return versions


def _capture_packages() -> dict[str, str]:
    captured: dict[str, str] = {}
    for name in _constraint_versions():
        try:
            captured[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return captured


def _run_nvidia_smi(query: str) -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _format_cudnn_version(version: int | None) -> str:
    if not version:
        return ""
    return f"{version // 1000}.{(version % 1000) // 100}.{version % 100}"


def _capture_cuda_and_gpus() -> tuple[dict[str, object], list[dict[str, object]]]:
    gpu_devices: list[dict[str, object]] = []
    for row in _run_nvidia_smi("name,uuid,compute_cap,memory.total"):
        values = [value.strip() for value in row.split(",")]
        if len(values) != 4:
            continue
        name, uuid, compute_capability, memory_mib = values
        try:
            memory_bytes = int(memory_mib) * 1024 * 1024
        except ValueError:
            continue
        gpu_devices.append(
            {
                "name": name,
                "uuid": uuid,
                "compute_capability": compute_capability,
                "memory_bytes": memory_bytes,
            }
        )

    driver_rows = _run_nvidia_smi("driver_version")
    driver = driver_rows[0] if driver_rows else ""
    runtime = ""
    cudnn_version = ""
    try:
        import torch

        runtime = torch.version.cuda or ""
        cudnn_version = _format_cudnn_version(torch.backends.cudnn.version())
    except (ImportError, AttributeError, RuntimeError):
        pass
    return (
        {
            "driver": driver,
            "runtime": runtime,
            "cudnn_version": cudnn_version,
        },
        gpu_devices,
    )


def _capture_custom_ops() -> dict[str, str]:
    captured: dict[str, str] = {}
    for module_name in CUSTOM_OP_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:
            captured[module_name] = "not_available"
        else:
            captured[module_name] = "imported"
    return captured


def _capture_determinism() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {
            "cudnn_benchmark": False,
            "tf32": False,
        }
    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32": bool(
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
        ),
    }


def _lock_hashes() -> tuple[str, str]:
    runtime_hash = _sha256_file(RUNTIME_LOCK) if RUNTIME_LOCK.is_file() else ""
    bootstrap_hash = _sha256_file(BOOTSTRAP_LOCK) if BOOTSTRAP_LOCK.is_file() else ""
    return runtime_hash, bootstrap_hash


def capture_environment(world_size: Literal[1, 2, 4]) -> dict[str, object]:
    world_size = _validate_world_size(world_size)
    cuda, gpu_devices = _capture_cuda_and_gpus()
    runtime_lock_sha256, bootstrap_lock_sha256 = _lock_hashes()
    return {
        "schema_version": 1,
        "platform": _capture_platform(),
        "python": _capture_python(),
        "packages": _capture_packages(),
        "cuda": cuda,
        "gpu": gpu_devices,
        "custom_ops": _capture_custom_ops(),
        "determinism": _capture_determinism(),
        "runtime_lock_sha256": runtime_lock_sha256,
        "bootstrap_lock_sha256": bootstrap_lock_sha256,
        "container_image_digest": os.environ.get(
            "RESILIENT_V2X_CONTAINER_IMAGE_DIGEST"
        ),
        "world_size": world_size,
    }


def hardware_fingerprint_sha256(
    fingerprint_without_hash: Mapping[str, object],
) -> str:
    if "fingerprint_sha256" in fingerprint_without_hash:
        raise EnvironmentMismatch(
            "fingerprint_without_hash must not contain fingerprint_sha256"
        )
    return _sha256_bytes(_canonical_json_bytes(fingerprint_without_hash))


def canonical_hardware_fingerprint(
    actual: Mapping[str, object],
    world_size: Literal[1, 2, 4],
) -> HardwareFingerprint:
    world_size = _validate_world_size(world_size)
    platform_capture = _require_mapping(actual.get("platform"), "platform")
    cuda_capture = _require_mapping(actual.get("cuda"), "cuda")
    gpu_capture = actual.get("gpu")
    if not isinstance(gpu_capture, (list, tuple)):
        raise EnvironmentMismatch("gpu must be a sequence")

    gpu_devices: list[dict[str, object]] = []
    for index, raw_gpu in enumerate(gpu_capture):
        gpu = _require_mapping(raw_gpu, f"gpu[{index}]")
        gpu_devices.append(
            {
                "name": _require_string(gpu.get("name"), f"gpu[{index}].name"),
                "uuid": _require_string(gpu.get("uuid"), f"gpu[{index}].uuid"),
                "compute_capability": _require_string(
                    gpu.get("compute_capability"),
                    f"gpu[{index}].compute_capability",
                ),
                "memory_bytes": _require_integer(
                    gpu.get("memory_bytes"), f"gpu[{index}].memory_bytes"
                ),
            }
        )

    payload: dict[str, object] = {
        "platform_system": _require_string(
            platform_capture.get("system"), "platform.system"
        ),
        "platform_machine": _require_string(
            platform_capture.get("machine"), "platform.machine"
        ),
        "cpu_model": _require_string(
            platform_capture.get("cpu_model"), "platform.cpu_model"
        ),
        "logical_cpu_count": _require_integer(
            platform_capture.get("logical_cpu_count"),
            "platform.logical_cpu_count",
        ),
        "total_ram_bytes": _require_integer(
            platform_capture.get("total_ram_bytes"), "platform.total_ram_bytes"
        ),
        "gpu_devices": gpu_devices,
        "cuda_driver": _require_string(cuda_capture.get("driver"), "cuda.driver"),
        "cuda_runtime": _require_string(
            cuda_capture.get("runtime"), "cuda.runtime"
        ),
        "cudnn_version": _require_string(
            cuda_capture.get("cudnn_version"), "cuda.cudnn_version"
        ),
        "world_size": world_size,
    }
    return HardwareFingerprint(
        platform_system=payload["platform_system"],
        platform_machine=payload["platform_machine"],
        cpu_model=payload["cpu_model"],
        logical_cpu_count=payload["logical_cpu_count"],
        total_ram_bytes=payload["total_ram_bytes"],
        gpu_devices=tuple(_freeze(gpu) for gpu in gpu_devices),
        cuda_driver=payload["cuda_driver"],
        cuda_runtime=payload["cuda_runtime"],
        cudnn_version=payload["cudnn_version"],
        world_size=world_size,
        fingerprint_sha256=hardware_fingerprint_sha256(payload),
    )


def hardware_fingerprint_to_dict(
    hardware: HardwareFingerprint,
) -> dict[str, object]:
    return {
        "platform_system": hardware.platform_system,
        "platform_machine": hardware.platform_machine,
        "cpu_model": hardware.cpu_model,
        "logical_cpu_count": hardware.logical_cpu_count,
        "total_ram_bytes": hardware.total_ram_bytes,
        "gpu_devices": _thaw(hardware.gpu_devices),
        "cuda_driver": hardware.cuda_driver,
        "cuda_runtime": hardware.cuda_runtime,
        "cudnn_version": hardware.cudnn_version,
        "world_size": hardware.world_size,
        "fingerprint_sha256": hardware.fingerprint_sha256,
    }


def environment_manifest_to_dict(
    manifest: EnvironmentManifest,
) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "classification": manifest.classification,
        "platform": _thaw(manifest.platform),
        "python": _thaw(manifest.python),
        "packages": _thaw(manifest.packages),
        "cuda": _thaw(manifest.cuda),
        "custom_ops": _thaw(manifest.custom_ops),
        "determinism": _thaw(manifest.determinism),
        "runtime_lock_sha256": manifest.runtime_lock_sha256,
        "bootstrap_lock_sha256": manifest.bootstrap_lock_sha256,
        "container_image_digest": manifest.container_image_digest,
        "hardware": hardware_fingerprint_to_dict(manifest.hardware),
        "actual_hardware_fingerprint_sha256": (
            manifest.actual_hardware_fingerprint_sha256
        ),
        "manifest_sha256": manifest.manifest_sha256,
    }


def _string_mapping(value: object, field: str) -> Mapping[str, str]:
    mapping = _require_mapping(value, field)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise EnvironmentMismatch(f"{field} keys and values must be strings")
        result[key] = item
    return _freeze(result)


def build_environment_manifest(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    world_size: Literal[1, 2, 4],
    classification: Literal["development", "controlled"],
) -> EnvironmentManifest:
    world_size = _validate_world_size(world_size)
    if classification not in ("development", "controlled"):
        raise ValueError(f"unsupported classification: {classification!r}")
    if actual.get("schema_version") != 1:
        raise EnvironmentMismatch("schema_version must be 1")

    hardware = canonical_hardware_fingerprint(actual, world_size)
    runtime_lock_sha256 = _require_string(
        actual.get(
            "runtime_lock_sha256",
            expected.get("runtime_lock_sha256"),
        ),
        "runtime_lock_sha256",
    )
    bootstrap_lock_sha256 = _require_string(
        actual.get(
            "bootstrap_lock_sha256",
            expected.get("bootstrap_lock_sha256"),
        ),
        "bootstrap_lock_sha256",
    )
    container_image_digest = actual.get(
        "container_image_digest",
        expected.get("container_image_digest"),
    )
    if container_image_digest is not None and not isinstance(
        container_image_digest, str
    ):
        raise EnvironmentMismatch("container_image_digest must be a string or null")

    platform_mapping = _freeze(_require_mapping(actual.get("platform"), "platform"))
    python_mapping = _freeze(_require_mapping(actual.get("python"), "python"))
    packages = _string_mapping(actual.get("packages"), "packages")
    cuda = _freeze(_require_mapping(actual.get("cuda"), "cuda"))
    custom_ops = _string_mapping(actual.get("custom_ops"), "custom_ops")
    determinism = _freeze(
        _require_mapping(actual.get("determinism"), "determinism")
    )
    base_payload: dict[str, object] = {
        "schema_version": 1,
        "classification": classification,
        "platform": _thaw(platform_mapping),
        "python": _thaw(python_mapping),
        "packages": _thaw(packages),
        "cuda": _thaw(cuda),
        "custom_ops": _thaw(custom_ops),
        "determinism": _thaw(determinism),
        "runtime_lock_sha256": runtime_lock_sha256,
        "bootstrap_lock_sha256": bootstrap_lock_sha256,
        "container_image_digest": container_image_digest,
        "hardware": hardware_fingerprint_to_dict(hardware),
        "actual_hardware_fingerprint_sha256": hardware.fingerprint_sha256,
    }
    manifest_sha256 = _sha256_bytes(_canonical_json_bytes(base_payload))
    return EnvironmentManifest(
        schema_version=1,
        classification=classification,
        platform=platform_mapping,
        python=python_mapping,
        packages=packages,
        cuda=cuda,
        custom_ops=custom_ops,
        determinism=determinism,
        runtime_lock_sha256=runtime_lock_sha256,
        bootstrap_lock_sha256=bootstrap_lock_sha256,
        container_image_digest=container_image_digest,
        hardware=hardware,
        actual_hardware_fingerprint_sha256=hardware.fingerprint_sha256,
        manifest_sha256=manifest_sha256,
    )


def _manifest_from_payload(payload: Mapping[str, object]) -> EnvironmentManifest:
    raw_hardware = _require_mapping(payload["hardware"], "hardware")
    gpu_devices = raw_hardware["gpu_devices"]
    if not isinstance(gpu_devices, list):
        raise EnvironmentMismatch("hardware.gpu_devices must be a list")
    hardware = HardwareFingerprint(
        platform_system=raw_hardware["platform_system"],
        platform_machine=raw_hardware["platform_machine"],
        cpu_model=raw_hardware["cpu_model"],
        logical_cpu_count=raw_hardware["logical_cpu_count"],
        total_ram_bytes=raw_hardware["total_ram_bytes"],
        gpu_devices=tuple(_freeze(item) for item in gpu_devices),
        cuda_driver=raw_hardware["cuda_driver"],
        cuda_runtime=raw_hardware["cuda_runtime"],
        cudnn_version=raw_hardware["cudnn_version"],
        world_size=raw_hardware["world_size"],
        fingerprint_sha256=raw_hardware["fingerprint_sha256"],
    )
    return EnvironmentManifest(
        schema_version=payload["schema_version"],
        classification=payload["classification"],
        platform=_freeze(payload["platform"]),
        python=_freeze(payload["python"]),
        packages=_freeze(payload["packages"]),
        cuda=_freeze(payload["cuda"]),
        custom_ops=_freeze(payload["custom_ops"]),
        determinism=_freeze(payload["determinism"]),
        runtime_lock_sha256=payload["runtime_lock_sha256"],
        bootstrap_lock_sha256=payload["bootstrap_lock_sha256"],
        container_image_digest=payload["container_image_digest"],
        hardware=hardware,
        actual_hardware_fingerprint_sha256=payload[
            "actual_hardware_fingerprint_sha256"
        ],
        manifest_sha256=payload["manifest_sha256"],
    )


def _verify_manifest_payload(payload: Mapping[str, object]) -> None:
    hardware = dict(_require_mapping(payload["hardware"], "hardware"))
    nested_hardware_hash = hardware.pop("fingerprint_sha256")
    recomputed_hardware_hash = hardware_fingerprint_sha256(hardware)
    if nested_hardware_hash != recomputed_hardware_hash:
        raise EnvironmentMismatch(
            "hardware.fingerprint_sha256 does not match canonical hardware"
        )
    if payload["actual_hardware_fingerprint_sha256"] != nested_hardware_hash:
        raise EnvironmentMismatch(
            "actual_hardware_fingerprint_sha256 does not match nested hardware"
        )
    manifest_payload = dict(payload)
    recorded_manifest_hash = manifest_payload.pop("manifest_sha256")
    recomputed_manifest_hash = _sha256_bytes(
        _canonical_json_bytes(manifest_payload)
    )
    if recorded_manifest_hash != recomputed_manifest_hash:
        raise EnvironmentMismatch(
            "manifest_sha256 does not match canonical environment manifest"
        )


def read_environment_manifest(path: Path) -> EnvironmentManifest:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EnvironmentMismatch(f"invalid environment manifest JSON: {error}") from error
    try:
        schema = json.loads(SCHEMA.read_text())
        jsonschema.Draft202012Validator(schema).validate(payload)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as error:
        raise EnvironmentMismatch(
            f"environment manifest schema validation failed: {error}"
        ) from error
    _verify_manifest_payload(payload)

    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        expected_sidecar = _sha256_bytes(path.read_bytes()) + "\n"
        if sidecar.read_text() != expected_sidecar:
            raise EnvironmentMismatch("environment manifest detached sidecar mismatch")
    return _manifest_from_payload(payload)


def _append_unique(target: list[str], value: str) -> None:
    if value not in target:
        target.append(value)


def validate_environment(
    actual: EnvironmentManifest,
    expected: Mapping[str, object],
    require_cuda: bool,
) -> EnvironmentValidation:
    _verify_manifest_payload(environment_manifest_to_dict(actual))
    controlled = actual.classification == "controlled"
    verified: list[str] = [
        "environment_manifest_sha256",
        "actual_hardware_fingerprint_sha256",
    ]
    blocking: list[str] = []
    not_executed: list[str] = []

    expected_world_size = expected.get("world_size")
    if expected_world_size != actual.hardware.world_size:
        blocking.append(
            "hardware.world_size mismatch: "
            f"expected {expected_world_size!r}, got {actual.hardware.world_size!r}"
        )
    else:
        verified.append("world_size")

    for field in ("runtime_lock_sha256", "bootstrap_lock_sha256"):
        expected_value = expected.get(field)
        actual_value = getattr(actual, field)
        if expected_value != actual_value:
            blocking.append(
                f"{field} mismatch: expected {expected_value!r}, got {actual_value!r}"
            )
        else:
            verified.append(field)

    expected_hardware = expected.get("hardware")
    if expected_hardware is not None:
        expected_hardware_mapping = _require_mapping(
            expected_hardware, "expected.hardware"
        )
        actual_hardware = hardware_fingerprint_to_dict(actual.hardware)
        actual_hardware.pop("fingerprint_sha256")
        for field, expected_value in expected_hardware_mapping.items():
            if actual_hardware.get(field) != expected_value:
                blocking.append(
                    f"hardware.{field} mismatch: expected {expected_value!r}, "
                    f"got {actual_hardware.get(field)!r}"
                )
            else:
                verified.append(f"hardware.{field}")

    expected_platform = _require_mapping(expected.get("platform", {}), "platform")
    for field, expected_value in expected_platform.items():
        actual_value = actual.platform.get(field)
        if actual_value == expected_value:
            verified.append(f"platform.{field}")
        elif controlled:
            blocking.append(
                f"platform.{field} mismatch: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )
        else:
            _append_unique(not_executed, "controlled_linux_runtime")
    if controlled and actual.hardware.platform_system != "Linux":
        blocking.append("controlled environment requires Linux")
    if not controlled:
        _append_unique(not_executed, "controlled_linux_runtime")

    expected_python = _require_mapping(expected.get("python", {}), "python")
    for field, expected_value in expected_python.items():
        actual_value = actual.python.get(field)
        if actual_value == expected_value:
            verified.append(f"python.{field}")
        elif controlled:
            blocking.append(
                f"python.{field} mismatch: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )
        else:
            _append_unique(not_executed, "runtime_python")

    expected_packages = _require_mapping(expected.get("packages", {}), "packages")
    package_mismatches = [
        f"{name} expected {version!r}, got {actual.packages.get(name)!r}"
        for name, version in expected_packages.items()
        if actual.packages.get(name) != version
    ]
    if package_mismatches:
        if controlled:
            blocking.extend(f"package {message}" for message in package_mismatches)
        else:
            _append_unique(not_executed, "runtime_packages")
    else:
        verified.append("packages")

    expected_cuda = _require_mapping(expected.get("cuda", {}), "cuda")
    cuda_available = bool(
        actual.hardware.gpu_devices
        and actual.hardware.cuda_driver
        and actual.hardware.cuda_runtime
        and actual.hardware.cudnn_version
    )
    if require_cuda and not cuda_available:
        blocking.append("CUDA hardware/runtime/cuDNN is unavailable")
    for field, expected_value in expected_cuda.items():
        actual_value = actual.cuda.get(field)
        if actual_value == expected_value:
            verified.append(f"cuda.{field}")
        elif controlled or require_cuda:
            blocking.append(
                f"CUDA {field} mismatch: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )
        else:
            _append_unique(not_executed, f"cuda_{field}")
    if not cuda_available and not require_cuda:
        _append_unique(not_executed, "cuda_runtime")

    expected_custom_ops = _require_mapping(
        expected.get("custom_ops", {}), "custom_ops"
    )
    if dict(actual.custom_ops) == dict(expected_custom_ops):
        verified.append("custom_ops")
    elif controlled:
        blocking.append(
            "custom_ops mismatch: "
            f"expected {dict(expected_custom_ops)!r}, got {dict(actual.custom_ops)!r}"
        )
    else:
        _append_unique(not_executed, "custom_ops")

    expected_determinism = _require_mapping(
        expected.get("determinism", {}), "determinism"
    )
    if not controlled and "torch" not in actual.packages:
        _append_unique(not_executed, "determinism_runtime")
    else:
        for field, expected_value in expected_determinism.items():
            actual_value = actual.determinism.get(field)
            if actual_value == expected_value:
                verified.append(f"determinism.{field}")
            elif controlled:
                blocking.append(
                    f"determinism.{field} mismatch: expected {expected_value!r}, "
                    f"got {actual_value!r}"
                )
            else:
                _append_unique(not_executed, f"determinism_{field}")

    expected_container = expected.get("container_image_digest")
    if controlled:
        if actual.container_image_digest is None:
            blocking.append("container_image_digest is required for controlled mode")
        elif (
            expected_container is not None
            and actual.container_image_digest != expected_container
        ):
            blocking.append(
                "container_image_digest mismatch: "
                f"expected {expected_container!r}, "
                f"got {actual.container_image_digest!r}"
            )
        else:
            verified.append("container_image_digest")
    elif actual.container_image_digest is None:
        _append_unique(not_executed, "container_image_digest")

    if blocking:
        raise EnvironmentMismatch("; ".join(blocking))
    return EnvironmentValidation(
        accepted=controlled and not not_executed,
        classification=actual.classification,
        environment_manifest_sha256=actual.manifest_sha256,
        actual_hardware_fingerprint_sha256=(
            actual.actual_hardware_fingerprint_sha256
        ),
        verified_fields=tuple(verified),
        blocking_mismatches=(),
        not_executed_checks=tuple(not_executed),
    )


def _crash_after_write(stage: str) -> None:
    requested = os.environ.get("RESILIENT_V2X_CRASH_AFTER_WRITE")
    if requested == stage:
        raise RuntimeError(f"injected crash after {stage}")


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_environment_manifest(
    manifest: EnvironmentManifest,
    output: Path,
) -> EnvironmentManifest:
    if not output.parent.is_dir():
        raise EnvironmentMismatch(
            f"environment manifest parent does not exist: {output.parent}"
        )
    payload = _canonical_json_bytes(environment_manifest_to_dict(manifest))
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar_payload = (_sha256_bytes(payload) + "\n").encode("ascii")

    if not output.exists() and sidecar.exists():
        raise EnvironmentMismatch(f"orphan environment manifest sidecar: {sidecar}")

    if output.exists():
        if output.read_bytes() != payload:
            raise EnvironmentMismatch(
                f"environment manifest JSON conflict, refusing overwrite: {output}"
            )
    else:
        _write_exclusive(output, payload)
    _crash_after_write("json")

    if sidecar.exists():
        if sidecar.read_bytes() != sidecar_payload:
            raise EnvironmentMismatch(
                f"environment manifest sidecar conflict, refusing overwrite: {sidecar}"
            )
    else:
        _write_exclusive(sidecar, sidecar_payload)
    _crash_after_write("hash")

    verified = read_environment_manifest(output)
    if verified != manifest:
        raise EnvironmentMismatch("reloaded environment manifest differs from capture")
    return verified


def expected_environment_contract() -> dict[str, object]:
    runtime_lock_sha256, bootstrap_lock_sha256 = _lock_hashes()
    return {
        "schema_version": 1,
        "platform": {"system": "Linux", "machine": "x86_64"},
        "python": {"version": "3.10.14", "implementation": "CPython"},
        "packages": _constraint_versions(),
        "cuda": {"runtime": "11.8"},
        "custom_ops": {module: "imported" for module in CUSTOM_OP_MODULES},
        "determinism": {
            "cudnn_benchmark": False,
            "tf32": False,
        },
        "runtime_lock_sha256": runtime_lock_sha256,
        "bootstrap_lock_sha256": bootstrap_lock_sha256,
        "container_image_digest": os.environ.get(
            "RESILIENT_V2X_CONTAINER_IMAGE_DIGEST"
        ),
        "world_size": 1,
    }


def _inside_evidence_root(output: Path, evidence_root: Path) -> bool:
    try:
        output.resolve(strict=False).relative_to(evidence_root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, choices=WORLD_SIZES, required=True)
    parser.add_argument(
        "--classification",
        choices=("development", "controlled"),
        default="development",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.output.is_absolute():
        raise EnvironmentMismatch("--output must be an absolute path")
    if not args.evidence_root.is_absolute():
        raise EnvironmentMismatch("--evidence-root must be an absolute path")
    if not args.evidence_root.is_dir():
        raise EnvironmentMismatch(
            f"evidence root does not exist: {args.evidence_root}"
        )
    if not _inside_evidence_root(args.output, args.evidence_root):
        raise EnvironmentMismatch(
            f"output is outside the explicitly supplied evidence root: {args.output}"
        )

    actual = capture_environment(world_size=args.world_size)
    expected = expected_environment_contract()
    manifest = build_environment_manifest(
        actual,
        expected,
        world_size=args.world_size,
        classification=args.classification,
    )
    verified = write_environment_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "manifest_sha256": verified.manifest_sha256,
                "actual_hardware_fingerprint_sha256": (
                    verified.actual_hardware_fingerprint_sha256
                ),
                "classification": verified.classification,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
