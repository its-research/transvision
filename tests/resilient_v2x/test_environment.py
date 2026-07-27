import copy
import hashlib
import importlib
import json
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from subprocess import run as _STDLIB_SUBPROCESS_RUN

import pytest
import yaml

from tools.resilient_v2x import capture_environment as capture_module
from tools.resilient_v2x import check_environment as check_module
from tools.resilient_v2x.capture_environment import (
    EnvironmentMismatch,
    build_environment_manifest,
    canonical_hardware_fingerprint,
    capture_environment,
    environment_manifest_to_dict,
    hardware_fingerprint_sha256,
    main as capture_main,
    read_environment_manifest,
    validate_environment,
    write_environment_manifest,
)

# Import the runtime validator before conda-lock 2.5.7 prepends Poetry's
# vendored jsonschema 3.2.0, and undo Poetry's Python-3 subprocess.run backport
# so this mixed Task 1 + Task 2 test process keeps the stdlib interface.
from conda_lock.lockfile import parse_conda_lock_file, write_conda_lock_file
from conda_lock.lockfile.v1.models import HashModel
from conda_lock.lockfile.v2prelim.models import LockedDependency, Lockfile

subprocess.run = _STDLIB_SUBPROCESS_RUN


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_DIR = ROOT / "environments" / "resilient_v2x"
ENVIRONMENT = ENVIRONMENT_DIR / "environment.yml"
RUNTIME_LOCK = ENVIRONMENT_DIR / "environment-linux-64.lock.yml"
PIP_REPORT = ENVIRONMENT_DIR / "runtime-pip-report.json"
BOOTSTRAP_ENVIRONMENT = ENVIRONMENT_DIR / "bootstrap-environment.yml"
BOOTSTRAP_LOCK = ENVIRONMENT_DIR / "bootstrap-linux-64.explicit.txt"
DOCKERFILE = ENVIRONMENT_DIR / "Dockerfile"
SCHEMA = ENVIRONMENT_DIR / "environment-manifest.schema.json"
BUILD_OPS = ROOT / "scripts" / "build_resilient_v2x_ops.sh"
CONSTRAINTS = ENVIRONMENT_DIR / "constraints.txt"
CUDA_IMAGE = (
    "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"
    "@sha256:bd746eb3b9953805ebe644847a227e218b5da775f47007c69930569a75c9ad7d"
)
MICROMAMBA_IMAGE = (
    "mambaorg/micromamba:2.8.1"
    "@sha256:79284aa2949ac9555eca7e975ceb4ecefefc5964a2d4fdd4e181ca8eaccf347e"
)

PACKAGE_VERSIONS = {
    "torch": "2.0.1+cu118",
    "torchvision": "0.15.2+cu118",
    "numpy": "1.24.4",
    "mmengine": "0.10.7",
    "mmcv": "2.1.0",
    "mmdet": "3.2.0",
    "mmdet3d": "1.3.0",
    "fvcore": "0.1.5.post20221221",
    "zstandard": "0.22.0",
    "pypcd4": "1.4.3",
    "pytest": "7.4.4",
    "jsonschema": "4.23.0",
    "PyYAML": "6.0.2",
    "conda-lock": "2.5.7",
}


def _seed_module():
    return importlib.import_module("tools.resilient_v2x.seed_runtime_lock")


def _committed_pip_report() -> dict[str, object]:
    return json.loads(PIP_REPORT.read_text())


def _runtime_environment() -> dict[str, object]:
    return yaml.safe_load(ENVIRONMENT.read_text())


def _complete_seed_lock(seed_path: Path, *, conda_hash: str = "c" * 32) -> None:
    lock = parse_conda_lock_file(seed_path)
    for name, version in {
        "python": "3.10.14",
        "pip": "23.3.2",
        "setuptools": "68.2.2",
        "wheel": "0.41.3",
        "packaging": "23.2",
        "cuda": "11.8.0",
        "cuda-toolkit": "11.8.0",
    }.items():
        lock.package.append(
            LockedDependency(
                name=name,
                version=version,
                manager="conda",
                platform="linux-64",
                dependencies={},
                url=(
                    "https://conda.anaconda.org/conda-forge/linux-64/"
                    f"{name}-{version}-test.conda"
                ),
                hash=HashModel(md5=conda_hash or None),
                category="main",
            )
        )
    write_conda_lock_file(lock, seed_path, metadata_choices=frozenset())


def fixed_gpus() -> list[dict[str, object]]:
    return [
        {
            "name": "NVIDIA GeForce RTX 3090",
            "uuid": "GPU-0001",
            "compute_capability": "8.6",
            "memory_bytes": 25_769_803_776,
        },
        {
            "name": "NVIDIA GeForce RTX 3090",
            "uuid": "GPU-0002",
            "compute_capability": "8.6",
            "memory_bytes": 25_769_803_776,
        },
    ]


def fixed_hardware_without_hash() -> dict[str, object]:
    return {
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "cpu_model": "AMD EPYC 7543",
        "logical_cpu_count": 64,
        "total_ram_bytes": 274_877_906_944,
        "gpu_devices": fixed_gpus(),
        "cuda_driver": "535.104.05",
        "cuda_runtime": "11.8",
        "cudnn_version": "8.7.0",
        "world_size": 2,
    }


def expected_environment(*, include_hardware: bool = False, world_size: int = 2) -> dict[str, object]:
    expected: dict[str, object] = {
        "schema_version": 1,
        "platform": {"system": "Linux", "machine": "x86_64"},
        "python": {"version": "3.10.14", "implementation": "CPython"},
        "packages": copy.deepcopy(PACKAGE_VERSIONS),
        "cuda": {
            "driver": "535.104.05",
            "runtime": "11.8",
            "cudnn_version": "8.7.0",
        },
        "custom_ops": {
            "transvision.models.voxel.voxel_layer": "imported",
            "transvision.models.bev_pool.bev_pool_ext": "imported",
        },
        "determinism": {
            "cudnn_benchmark": False,
            "tf32": False,
        },
        "runtime_lock_sha256": "a" * 64,
        "bootstrap_lock_sha256": "b" * 64,
        "container_image_digest": "sha256:" + "c" * 64,
        "world_size": world_size,
    }
    if include_hardware:
        hardware = fixed_hardware_without_hash()
        hardware["world_size"] = world_size
        expected["hardware"] = hardware
    return expected


def expected_environment_capture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "platform": {
            "system": "Linux",
            "machine": "x86_64",
            "cpu_model": "AMD EPYC 7543",
            "logical_cpu_count": 64,
            "total_ram_bytes": 274_877_906_944,
        },
        "python": {"version": "3.10.14", "implementation": "CPython"},
        "packages": copy.deepcopy(PACKAGE_VERSIONS),
        "cuda": {
            "driver": "535.104.05",
            "runtime": "11.8",
            "cudnn_version": "8.7.0",
        },
        "gpu": fixed_gpus(),
        "custom_ops": {
            "transvision.models.voxel.voxel_layer": "imported",
            "transvision.models.bev_pool.bev_pool_ext": "imported",
        },
        "determinism": {
            "cudnn_benchmark": False,
            "tf32": False,
        },
        "runtime_lock_sha256": "a" * 64,
        "bootstrap_lock_sha256": "b" * 64,
        "container_image_digest": "sha256:" + "c" * 64,
        "world_size": 2,
    }


def development_capture() -> dict[str, object]:
    actual = expected_environment_capture()
    actual["platform"] = {
        "system": "Darwin",
        "machine": "arm64",
        "cpu_model": "Apple M4",
        "logical_cpu_count": 12,
        "total_ram_bytes": 34_359_738_368,
    }
    actual["packages"] = {}
    actual["cuda"] = {"driver": "", "runtime": "", "cudnn_version": ""}
    actual["gpu"] = []
    actual["custom_ops"] = {
        "transvision.models.voxel.voxel_layer": "not_available",
        "transvision.models.bev_pool.bev_pool_ext": "not_available",
    }
    actual["container_image_digest"] = None
    actual["world_size"] = 1
    return actual


def manifest_for_write():
    return build_environment_manifest(
        expected_environment_capture(),
        expected_environment(),
        world_size=2,
        classification="controlled",
    )


def sidecar_for(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".sha256")


def test_capture_has_required_hardware_and_determinism_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        capture_module,
        "_capture_platform",
        lambda: {
            "system": "Linux",
            "machine": "x86_64",
            "cpu_model": "AMD EPYC 7543",
            "logical_cpu_count": 64,
            "total_ram_bytes": 274_877_906_944,
        },
    )
    monkeypatch.setattr(
        capture_module,
        "_capture_python",
        lambda: {"version": "3.10.14", "implementation": "CPython"},
    )
    monkeypatch.setattr(capture_module, "_capture_packages", lambda: copy.deepcopy(PACKAGE_VERSIONS))
    monkeypatch.setattr(
        capture_module,
        "_capture_cuda_and_gpus",
        lambda: (
            {"driver": "535.104.05", "runtime": "11.8", "cudnn_version": "8.7.0"},
            fixed_gpus(),
        ),
    )
    monkeypatch.setattr(
        capture_module,
        "_capture_determinism",
        lambda: {
            "cudnn_benchmark": False,
            "tf32": False,
        },
    )
    monkeypatch.setattr(
        capture_module,
        "_capture_custom_ops",
        lambda: {
            "transvision.models.voxel.voxel_layer": "imported",
            "transvision.models.bev_pool.bev_pool_ext": "imported",
        },
    )
    monkeypatch.setattr(capture_module, "_lock_hashes", lambda: ("a" * 64, "b" * 64))
    monkeypatch.setenv("RESILIENT_V2X_CONTAINER_IMAGE_DIGEST", "sha256:" + "c" * 64)

    captured = capture_environment(world_size=1)

    assert captured["schema_version"] == 1
    assert captured["platform"]["system"] == "Linux"
    assert captured["python"]["version"] == "3.10.14"
    assert captured["cuda"]["runtime"] == "11.8"
    assert captured["gpu"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert captured["determinism"]["cudnn_benchmark"] is False
    assert captured["determinism"]["tf32"] is False
    assert captured["world_size"] == 1
    assert len(captured["gpu"]) == 2


@pytest.mark.parametrize("world_size", [0, 3, 8])
def test_capture_rejects_world_size_outside_contract(world_size: int) -> None:
    with pytest.raises(ValueError, match="world_size"):
        capture_environment(world_size=world_size)


def test_canonical_hardware_hash_matches_fixed_vector() -> None:
    assert (
        hardware_fingerprint_sha256(fixed_hardware_without_hash())
        == "a1631f25a4c2bbb79cb53eb618c69379d38537c10f06cd839055235df52de2ed"
    )


def _mutate_cpu_model(actual: dict[str, object]) -> None:
    actual["platform"]["cpu_model"] = "AMD EPYC 7763"


def _mutate_cpu_count(actual: dict[str, object]) -> None:
    actual["platform"]["logical_cpu_count"] = 63


def _mutate_ram(actual: dict[str, object]) -> None:
    actual["platform"]["total_ram_bytes"] -= 1


def _mutate_gpu_order(actual: dict[str, object]) -> None:
    actual["gpu"].reverse()


def _mutate_gpu_name(actual: dict[str, object]) -> None:
    actual["gpu"][0]["name"] = "NVIDIA A100"


def _mutate_gpu_uuid(actual: dict[str, object]) -> None:
    actual["gpu"][0]["uuid"] = "GPU-DIFFERENT"


def _mutate_gpu_compute_capability(actual: dict[str, object]) -> None:
    actual["gpu"][0]["compute_capability"] = "8.0"


def _mutate_gpu_memory(actual: dict[str, object]) -> None:
    actual["gpu"][0]["memory_bytes"] -= 1


def _mutate_cuda_driver(actual: dict[str, object]) -> None:
    actual["cuda"]["driver"] = "550.54.14"


def _mutate_cuda_runtime(actual: dict[str, object]) -> None:
    actual["cuda"]["runtime"] = "12.1"


def _mutate_cudnn(actual: dict[str, object]) -> None:
    actual["cuda"]["cudnn_version"] = "9.0.0"


def _mutate_machine(actual: dict[str, object]) -> None:
    actual["platform"]["machine"] = "aarch64"


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (_mutate_cpu_model, "cpu_model"),
        (_mutate_cpu_count, "logical_cpu_count"),
        (_mutate_ram, "total_ram_bytes"),
        (_mutate_gpu_order, "gpu_devices"),
        (_mutate_gpu_name, "gpu_devices"),
        (_mutate_gpu_uuid, "gpu_devices"),
        (_mutate_gpu_compute_capability, "gpu_devices"),
        (_mutate_gpu_memory, "gpu_devices"),
        (_mutate_cuda_driver, "cuda_driver"),
        (_mutate_cuda_runtime, "cuda_runtime"),
        (_mutate_cudnn, "cudnn_version"),
        (_mutate_machine, "platform_machine"),
    ],
)
def test_hardware_mutation_is_a_blocking_mismatch(mutator, match: str) -> None:
    actual = expected_environment_capture()
    mutator(actual)
    manifest = build_environment_manifest(
        actual,
        expected_environment(include_hardware=True),
        world_size=2,
        classification="controlled",
    )

    with pytest.raises(EnvironmentMismatch, match=match):
        validate_environment(
            manifest,
            expected_environment(include_hardware=True),
            require_cuda=True,
        )


def test_launcher_world_size_mismatch_is_blocking() -> None:
    manifest = build_environment_manifest(
        expected_environment_capture(),
        expected_environment(world_size=1),
        world_size=4,
        classification="controlled",
    )

    with pytest.raises(EnvironmentMismatch, match="world_size"):
        validate_environment(
            manifest,
            expected_environment(world_size=1),
            require_cuda=True,
        )


def test_actual_hardware_hash_equals_nested_fingerprint_hash() -> None:
    manifest = manifest_for_write()

    assert (
        manifest.actual_hardware_fingerprint_sha256
        == manifest.hardware.fingerprint_sha256
    )


def _manifest_hash_after(mutator) -> str:
    actual = expected_environment_capture()
    expected = expected_environment()
    mutator(actual, expected)
    return build_environment_manifest(
        actual,
        expected,
        world_size=2,
        classification="controlled",
    ).manifest_sha256


@pytest.mark.parametrize(
    "mutator",
    [
        lambda actual, expected: actual["packages"].__setitem__("mmdet3d", "1.2.0"),
        lambda actual, expected: actual.__setitem__("runtime_lock_sha256", "d" * 64),
        lambda actual, expected: actual.__setitem__("bootstrap_lock_sha256", "e" * 64),
        lambda actual, expected: actual["custom_ops"].__setitem__(
            "transvision.models.voxel.voxel_layer", "not_available"
        ),
        lambda actual, expected: actual["determinism"].__setitem__("tf32", True),
        lambda actual, expected: actual.__setitem__(
            "container_image_digest", "sha256:" + "f" * 64
        ),
        lambda actual, expected: actual["platform"].__setitem__("cpu_model", "different"),
    ],
)
def test_manifest_hash_covers_runtime_contract_fields(mutator) -> None:
    baseline = manifest_for_write().manifest_sha256
    assert _manifest_hash_after(mutator) != baseline


def test_manifest_hash_excludes_only_itself() -> None:
    manifest = manifest_for_write()
    payload = environment_manifest_to_dict(manifest)
    payload.pop("manifest_sha256")
    expected_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    assert manifest.manifest_sha256 == expected_hash


def test_manifest_contract_mappings_are_immutable() -> None:
    manifest = manifest_for_write()

    with pytest.raises(TypeError):
        manifest.packages["mmdet3d"] = "1.2.0"
    with pytest.raises(TypeError):
        manifest.hardware.gpu_devices[0]["name"] = "different"
    with pytest.raises(FrozenInstanceError):
        manifest.runtime_lock_sha256 = "d" * 64


def test_wrong_mmdet3d_version_fails() -> None:
    actual = expected_environment_capture()
    actual["packages"]["mmdet3d"] = "1.2.0"
    manifest = build_environment_manifest(
        actual,
        expected_environment(),
        world_size=2,
        classification="controlled",
    )

    with pytest.raises(EnvironmentMismatch, match="mmdet3d"):
        validate_environment(manifest, expected_environment(), require_cuda=True)


def test_missing_cuda_fails_controlled_validation() -> None:
    actual = expected_environment_capture()
    actual["cuda"] = {"driver": "", "runtime": "", "cudnn_version": ""}
    actual["gpu"] = []
    manifest = build_environment_manifest(
        actual,
        expected_environment(),
        world_size=2,
        classification="controlled",
    )

    with pytest.raises(EnvironmentMismatch, match="CUDA"):
        validate_environment(manifest, expected_environment(), require_cuda=True)


@pytest.mark.parametrize(
    ("field", "dirty_value"),
    [
        ("cudnn_benchmark", True),
        ("tf32", True),
    ],
)
def test_dirty_deterministic_flag_fails(field: str, dirty_value: bool) -> None:
    actual = expected_environment_capture()
    actual["determinism"][field] = dirty_value
    manifest = build_environment_manifest(
        actual,
        expected_environment(),
        world_size=2,
        classification="controlled",
    )

    with pytest.raises(EnvironmentMismatch, match=field):
        validate_environment(manifest, expected_environment(), require_cuda=True)


def test_macos_development_is_not_accepted_as_controlled() -> None:
    actual = development_capture()
    expected = expected_environment(world_size=1)
    expected["container_image_digest"] = None
    manifest = build_environment_manifest(
        actual,
        expected,
        world_size=1,
        classification="development",
    )

    validation = validate_environment(manifest, expected, require_cuda=False)

    assert validation.accepted is False
    assert validation.classification == "development"
    assert "cuda_runtime" in validation.not_executed_checks
    assert "custom_ops" in validation.not_executed_checks
    assert "determinism_runtime" in validation.not_executed_checks
    assert "controlled_linux_runtime" in validation.not_executed_checks
    assert "determinism.cudnn_benchmark" not in validation.verified_fields


def test_exact_controlled_environment_is_accepted() -> None:
    manifest = manifest_for_write()

    validation = validate_environment(
        manifest,
        expected_environment(),
        require_cuda=True,
    )

    assert validation.accepted is True
    assert validation.blocking_mismatches == ()
    assert validation.not_executed_checks == ()
    assert "runtime_lock_sha256" in validation.verified_fields
    assert "custom_ops" in validation.verified_fields


def test_strict_schema_rejects_unknown_manifest_field(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    write_environment_manifest(manifest_for_write(), output)
    payload = json.loads(output.read_text())
    payload["unreviewed"] = True
    output.write_text(json.dumps(payload))

    with pytest.raises(EnvironmentMismatch, match="schema"):
        read_environment_manifest(output)


def test_strict_schema_rejects_unknown_nested_hardware_field(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    write_environment_manifest(manifest_for_write(), output)
    payload = json.loads(output.read_text())
    payload["hardware"]["nickname"] = "runner"
    output.write_text(json.dumps(payload))

    with pytest.raises(EnvironmentMismatch, match="schema"):
        read_environment_manifest(output)


def test_strict_parser_rejects_tampered_manifest_hash(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    write_environment_manifest(manifest_for_write(), output)
    payload = json.loads(output.read_text())
    payload["packages"]["mmdet3d"] = "1.2.0"
    output.write_text(json.dumps(payload))

    with pytest.raises(EnvironmentMismatch, match="manifest_sha256"):
        read_environment_manifest(output)


def _patch_capture_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        capture_module,
        "capture_environment",
        lambda world_size: {
            **expected_environment_capture(),
            "world_size": world_size,
        },
    )
    monkeypatch.setattr(
        capture_module,
        "expected_environment_contract",
        lambda: expected_environment(),
    )


def test_capture_cli_requires_explicit_world_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def unexpected_capture(world_size: int):
        nonlocal called
        called = True
        return expected_environment_capture()

    monkeypatch.setattr(capture_module, "capture_environment", unexpected_capture)

    with pytest.raises(SystemExit) as error:
        capture_main(
            [
                "--classification",
                "development",
                "--evidence-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "manifest.json"),
            ]
        )

    assert error.value.code == 2
    assert called is False


def test_capture_cli_refuses_relative_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_capture_cli(monkeypatch)

    with pytest.raises(EnvironmentMismatch, match="absolute"):
        capture_main(
            [
                "--world-size",
                "1",
                "--classification",
                "development",
                "--evidence-root",
                str(tmp_path),
                "--output",
                "manifest.json",
            ]
        )


def test_capture_cli_refuses_output_outside_evidence_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_capture_cli(monkeypatch)
    outside = tmp_path.with_name(tmp_path.name + "-outside") / "manifest.json"

    with pytest.raises(EnvironmentMismatch, match="evidence root"):
        capture_main(
            [
                "--world-size",
                "1",
                "--classification",
                "development",
                "--evidence-root",
                str(tmp_path),
                "--output",
                str(outside),
            ]
        )

    assert not outside.exists()


def test_capture_cli_writes_canonical_json_hash_and_world_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_capture_cli(monkeypatch)
    output = tmp_path / "evidence" / "manifest.json"
    evidence_root = output.parent
    evidence_root.mkdir()

    assert (
        capture_main(
            [
                "--world-size",
                "4",
                "--classification",
                "development",
                "--evidence-root",
                str(evidence_root),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text())
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert output.read_bytes() == canonical
    assert payload["hardware"]["world_size"] == 4
    assert sidecar_for(output).read_text() == hashlib.sha256(canonical).hexdigest() + "\n"
    assert json.loads(capsys.readouterr().out)["manifest_sha256"] == payload["manifest_sha256"]


def test_retry_recovers_after_crash_after_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"

    def crash(stage: str) -> None:
        if stage == "json":
            raise RuntimeError("injected crash after JSON")

    monkeypatch.setattr(capture_module, "_crash_after_write", crash)
    with pytest.raises(RuntimeError, match="after JSON"):
        write_environment_manifest(manifest_for_write(), output)

    json_before = output.read_bytes()
    assert not sidecar_for(output).exists()
    monkeypatch.setattr(capture_module, "_crash_after_write", lambda stage: None)

    assert write_environment_manifest(manifest_for_write(), output) == manifest_for_write()
    assert output.read_bytes() == json_before
    assert sidecar_for(output).is_file()


def test_retry_is_idempotent_after_crash_after_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"

    def crash(stage: str) -> None:
        if stage == "hash":
            raise RuntimeError("injected crash after hash")

    monkeypatch.setattr(capture_module, "_crash_after_write", crash)
    with pytest.raises(RuntimeError, match="after hash"):
        write_environment_manifest(manifest_for_write(), output)

    json_before = output.read_bytes()
    sidecar_before = sidecar_for(output).read_bytes()
    monkeypatch.setattr(capture_module, "_crash_after_write", lambda stage: None)

    assert write_environment_manifest(manifest_for_write(), output) == manifest_for_write()
    assert output.read_bytes() == json_before
    assert sidecar_for(output).read_bytes() == sidecar_before


def test_existing_conflicting_json_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.write_bytes(b'{"different":true}')
    before = output.read_bytes()

    with pytest.raises(EnvironmentMismatch, match="conflict"):
        write_environment_manifest(manifest_for_write(), output)

    assert output.read_bytes() == before
    assert not sidecar_for(output).exists()


def test_orphan_sidecar_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    sidecar = sidecar_for(output)
    sidecar.write_text("f" * 64 + "\n")
    before = sidecar.read_bytes()

    with pytest.raises(EnvironmentMismatch, match="orphan"):
        write_environment_manifest(manifest_for_write(), output)

    assert not output.exists()
    assert sidecar.read_bytes() == before


def test_existing_conflicting_sidecar_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    write_environment_manifest(manifest_for_write(), output)
    sidecar = sidecar_for(output)
    sidecar.write_text("f" * 64 + "\n")
    before_json = output.read_bytes()
    before_sidecar = sidecar.read_bytes()

    with pytest.raises(EnvironmentMismatch, match="sidecar"):
        write_environment_manifest(manifest_for_write(), output)

    assert output.read_bytes() == before_json
    assert sidecar.read_bytes() == before_sidecar


def test_committed_pip_report_is_the_audited_native_linux_input() -> None:
    # Production break caught: replacing the audited report with a regenerated
    # or host-native report silently changes the runtime closure.
    assert hashlib.sha256(PIP_REPORT.read_bytes()).hexdigest() == (
        "7f42b38a7734de6f14e8985e84df8d719e97742b9795b9f7fcad99d21e08ad64"
    )

    report = _committed_pip_report()

    assert report["version"] == "1"
    assert report["pip_version"] == "23.3.2"
    assert report["environment"]["platform_system"] == "Linux"
    assert report["environment"]["platform_machine"] == "x86_64"
    assert report["environment"]["python_full_version"] == "3.10.14"
    assert len(report["install"]) == 179


def test_committed_pip_report_preserves_compatible_build_toolchain() -> None:
    report = _committed_pip_report()
    versions = {
        re.sub(r"[-_.]+", "-", item["metadata"]["name"]).lower(): item[
            "metadata"
        ]["version"]
        for item in report["install"]
    }

    assert versions["setuptools"] == "68.2.2"
    assert versions["wheel"] == "0.41.3"
    assert versions["packaging"] == "23.2"


def test_committed_pip_report_uses_official_cuda_11_8_torch_wheels() -> None:
    report = _committed_pip_report()
    artifacts = {
        re.sub(r"[-_.]+", "-", item["metadata"]["name"]).lower(): item
        for item in report["install"]
    }

    assert artifacts["torch"]["metadata"]["version"] == "2.0.1+cu118"
    assert artifacts["torchvision"]["metadata"]["version"] == "0.15.2+cu118"
    assert artifacts["torch"]["download_info"]["url"].startswith(
        "https://download-r2.pytorch.org/whl/cu118/"
    )
    assert artifacts["torchvision"]["download_info"]["url"].startswith(
        "https://download-r2.pytorch.org/whl/cu118/"
    )
    assert not any(
        name.startswith("nvidia-") and name.endswith("-cu11")
        for name in artifacts
    )


@pytest.mark.parametrize(
    ("production_break", "match"),
    [
        ("wrong_pip", "pip"),
        ("non_linux", "Linux"),
        ("wrong_python", "Python"),
        ("duplicate_name", "duplicate"),
        ("missing_hash", "SHA-256"),
        ("credentialed_url", "credential"),
        ("direct_version_drift", "torch"),
        ("cu118_source_drift", "official CUDA 11.8"),
    ],
)
def test_seed_rejects_unaudited_pip_report_mutations(
    production_break: str,
    match: str,
) -> None:
    # Production breaks caught: accepting the wrong resolver/runtime platform,
    # ambiguous records, unverifiable artifacts, leaked credentials, or drifted
    # approved pins would make the seed non-reproducible.
    module = _seed_module()
    report = copy.deepcopy(_committed_pip_report())
    if production_break == "wrong_pip":
        report["pip_version"] = "24.0"
    elif production_break == "non_linux":
        report["environment"]["platform_system"] = "Darwin"
    elif production_break == "wrong_python":
        report["environment"]["python_full_version"] = "3.11.9"
    elif production_break == "duplicate_name":
        report["install"].append(copy.deepcopy(report["install"][0]))
    elif production_break == "missing_hash":
        report["install"][0]["download_info"]["archive_info"] = {}
    elif production_break == "credentialed_url":
        report["install"][0]["download_info"]["url"] = (
            "https://user:secret@files.pythonhosted.org/artifact.whl"
        )
    elif production_break == "direct_version_drift":
        torch = next(
            item for item in report["install"] if item["metadata"]["name"] == "torch"
        )
        torch["metadata"]["version"] = "2.0.0"
    elif production_break == "cu118_source_drift":
        torch = next(
            item for item in report["install"] if item["metadata"]["name"] == "torch"
        )
        torch["download_info"]["url"] = (
            "https://files.pythonhosted.org/drifted/torch-2.0.1+cu118.whl"
        )
    else:  # pragma: no cover - the table above is exhaustive
        raise AssertionError(production_break)

    with pytest.raises(module.PipReportMismatch, match=match):
        module.build_seed_lock(
            report,
            PACKAGE_VERSIONS,
            _runtime_environment(),
        )


def test_seed_lock_is_deterministic_pip_only_and_preserves_report_artifacts() -> None:
    # Production breaks caught: unsorted model construction, conda records in
    # the temporary seed, or synthesized versions/URLs/hashes.
    module = _seed_module()
    report = _committed_pip_report()

    first = module.build_seed_lock(
        report,
        PACKAGE_VERSIONS,
        _runtime_environment(),
    )
    second = module.build_seed_lock(
        copy.deepcopy(report),
        dict(reversed(list(PACKAGE_VERSIONS.items()))),
        copy.deepcopy(_runtime_environment()),
    )

    assert isinstance(first, Lockfile)
    assert first.dict() == second.dict()
    assert len(first.package) == 179
    assert all(isinstance(package, LockedDependency) for package in first.package)
    assert all(package.manager == "pip" for package in first.package)
    assert [package.name for package in first.package] == sorted(
        package.name for package in first.package
    )
    report_artifacts = {
        re.sub(r"[-_.]+", "-", item["metadata"]["name"]).lower(): (
            item["metadata"]["version"],
            item["download_info"]["url"],
            item["download_info"]["archive_info"]["hashes"]["sha256"],
        )
        for item in report["install"]
    }
    seed_artifacts = {
        package.name: (package.version, package.url, package.hash.sha256)
        for package in first.package
    }
    assert seed_artifacts == report_artifacts


def test_seed_adapter_refuses_unpinned_conda_lock_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Production break caught: running the adapter under a different model or
    # resolver version invalidates the deterministic seed and handoff contract.
    module = _seed_module()
    monkeypatch.setattr(module.conda_lock, "__version__", "2.6.0")

    with pytest.raises(module.PipReportMismatch, match="2.5.7"):
        module.build_seed_lock(
            _committed_pip_report(),
            PACKAGE_VERSIONS,
            _runtime_environment(),
        )


def test_cu118_wheel_cache_is_optional_and_rejects_wrong_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _seed_module()
    monkeypatch.delenv("RESILIENT_V2X_CU118_WHEEL_CACHE", raising=False)
    assert module._validated_cu118_wheel_cache() is None

    wrong_wheel = tmp_path / "torch-cu118.whl"
    wrong_wheel.write_bytes(b"not the audited torch wheel")
    monkeypatch.setenv("RESILIENT_V2X_CU118_WHEEL_CACHE", str(wrong_wheel))
    with pytest.raises(module.PipReportMismatch, match="SHA-256"):
        module._validated_cu118_wheel_cache()


def test_old_poetry_normalizes_only_torchvision_cu118_torch_dependency() -> None:
    module = _seed_module()

    class Dependency:
        name = "torch"
        constraint = "2.0.1"
        _pretty_constraint = "2.0.1"

        def set_constraint(self, value: str) -> None:
            self.constraint = value

    dependency = Dependency()
    package = type(
        "Package",
        (),
        {
            "name": "torchvision",
            "version": "0.15.2+cu118",
            "requires": [dependency],
        },
    )()

    assert module._normalize_torchvision_cu118_dependency(package) is package
    assert dependency.constraint == "2.0.1+cu118"
    assert dependency._pretty_constraint == "2.0.1+cu118"


def test_standard_handoff_uses_pinned_conda_lock_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Production break caught: bypassing the normal conda-lock 2.5.7 command,
    # changing its committed source, platform, format, or existing-lock path.
    module = _seed_module()
    seed_path = tmp_path / "seed.lock.yml"
    seed = module.build_seed_lock(
        _committed_pip_report(),
        PACKAGE_VERSIONS,
        _runtime_environment(),
    )
    write_conda_lock_file(seed, seed_path, metadata_choices=frozenset())
    observed: list[list[str]] = []

    def capture_run(command, **kwargs):
        observed.append(command)
        assert kwargs["check"] is True
        child_environment = kwargs["env"]
        assert child_environment[
            "RESILIENT_V2X_CONDA_LOCK_MANYLINUX_TAGS"
        ] == "_2_31,_2_34"
        compatibility_root = Path(
            child_environment["PYTHONPATH"].split(module.os.pathsep, 1)[0]
        )
        assert (compatibility_root / "sitecustomize.py").is_file()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", capture_run)

    module._run_standard_conda_lock(ENVIRONMENT, seed_path, "linux-64")

    assert observed == [
        [
            sys.executable,
            "-m",
            "conda_lock",
            "lock",
            "--file",
            str(ENVIRONMENT),
            "--platform",
            "linux-64",
            "--lockfile",
            str(seed_path),
            "--micromamba",
            "--no-mamba",
            "--with-cuda",
            "11.8",
            "--log-level",
            "INFO",
        ]
    ]


def test_standard_handoff_chooser_accepts_audited_newer_manylinux_wheels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Production break caught: conda-lock 2.5.7's stale manylinux_2_28 ceiling
    # rejects three wheels already accepted by the native Linux pip 23.3.2
    # report, after the resolver has otherwise completed successfully.
    module = _seed_module()
    seed_path = tmp_path / "seed.lock.yml"
    seed = module.build_seed_lock(
        _committed_pip_report(),
        PACKAGE_VERSIONS,
        _runtime_environment(),
    )
    write_conda_lock_file(seed, seed_path, metadata_choices=frozenset())
    real_run = subprocess.run

    def probe_child_environment(command, **kwargs):
        probe = """
from conda_lock._vendor.poetry.installation.chooser import Wheel
from conda_lock.pypi_solver import PlatformEnv

environment = PlatformEnv("3.10.14", "linux-64")
filenames = (
    "open3d-0.19.0-cp310-cp310-manylinux_2_31_x86_64.whl",
    "tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl",
    "cryptography-49.0.0-cp39-abi3-manylinux_2_34_x86_64.whl",
)
assert all(
    Wheel(filename).is_supported_by_environment(environment)
    for filename in filenames
)
"""
        real_run(
            [sys.executable, "-c", probe],
            check=True,
            env=kwargs["env"],
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", probe_child_environment)

    module._run_standard_conda_lock(ENVIRONMENT, seed_path, "linux-64")


def test_standard_handoff_preserves_explicit_conda_pip_pin_and_content_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Production break caught: conda-lock 2.5.7 appends an implicit unpinned
    # conda-side pip requirement after the committed pip=23.3.2 requirement;
    # aggregation is last-wins and otherwise resolves a newer pip.
    module = _seed_module()
    seed_path = tmp_path / "seed.lock.yml"
    seed = module.build_seed_lock(
        _committed_pip_report(),
        PACKAGE_VERSIONS,
        _runtime_environment(),
    )
    write_conda_lock_file(seed, seed_path, metadata_choices=frozenset())
    real_run = subprocess.run

    def probe_child_environment(command, **kwargs):
        probe = """
import os
from pathlib import Path

import yaml
from conda_lock.models.lock_spec import VersionedDependency
import conda_lock.src_parser.environment_yaml as environment_yaml
from conda_lock.virtual_package import default_virtual_package_repodata
from tools.resilient_v2x import seed_runtime_lock

environment_path = Path(
    os.environ["RESILIENT_V2X_CONDA_LOCK_ENVIRONMENT_SNAPSHOT"]
)
assert not (environment_path.stat().st_mode & 0o222)
original_parse_python_requirement = environment_yaml.parse_python_requirement

def parse_without_network(requirement, *args, **kwargs):
    if requirement == "pip" and kwargs.get("manager") == "conda":
        return VersionedDependency(
            name="pip",
            manager="conda",
            category="main",
            version="*",
        )
    return original_parse_python_requirement(requirement, *args, **kwargs)

environment_yaml.parse_python_requirement = parse_without_network
lock_spec = environment_yaml.parse_environment_file(
    environment_path,
    ["linux-64"],
)
conda_pip = [
    dependency
    for dependency in lock_spec.dependencies["linux-64"]
    if dependency.manager == "conda" and dependency.name == "pip"
]
assert len(conda_pip) == 1
assert conda_pip[0].version == "23.3.2.*"
lock_spec.virtual_package_repo = default_virtual_package_repodata(
    cuda_version="11.8"
)
committed_environment = yaml.safe_load(environment_path.read_text())
assert lock_spec.content_hash_for_platform("linux-64") == (
    seed_runtime_lock._environment_content_hash(
        committed_environment,
        "linux-64",
    )
)
"""
        real_run(
            [sys.executable, "-c", probe],
            check=True,
            cwd=ROOT,
            env=kwargs["env"],
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", probe_child_environment)

    module._run_standard_conda_lock(ENVIRONMENT, seed_path, "linux-64")


def test_standard_handoff_preserves_only_missing_audited_seed_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Production break caught: Poetry drops locked transitive packages when its
    # current metadata omits Torch's Linux wheel dependency closure, and also
    # suppresses exact pip artifacts that the conda solve marks installed.
    module = _seed_module()
    seed_path = tmp_path / "seed.lock.yml"
    seed = module.build_seed_lock(
        _committed_pip_report(),
        PACKAGE_VERSIONS,
        _runtime_environment(),
    )
    write_conda_lock_file(seed, seed_path, metadata_choices=frozenset())
    real_run = subprocess.run

    def probe_child_environment(command, **kwargs):
        probe = """
import os
import re
from pathlib import Path

from conda_lock._vendor.poetry.core.packages.package import Package
from conda_lock._vendor.poetry.installation.operations.install import Install
from conda_lock.lockfile import parse_conda_lock_file
from conda_lock.pypi_solver import PlatformEnv, Pool, get_requirements

seed = parse_conda_lock_file(
    Path(os.environ["RESILIENT_V2X_CONDA_LOCK_SEED_SNAPSHOT"])
)
assert len(seed.package) == 179
assert not (
    Path(os.environ["RESILIENT_V2X_CONDA_LOCK_SEED_SNAPSHOT"]).stat().st_mode
    & 0o222
)
expected = {
    re.sub(r"[-_.]+", "-", package.name).lower(): (
        package.version,
        package.url,
        package.hash.sha256,
    )
    for package in seed.package
}
torch = next(package for package in seed.package if package.name == "torch")
operation = Install(
    Package(
        torch.name,
        torch.version,
        source_type="url",
        source_url=f"{torch.url}#sha256={torch.hash.sha256}",
    )
)
requirements = get_requirements(
    [operation],
    "linux-64",
    Pool(),
    PlatformEnv("3.10.14", "linux-64"),
)
actual = {
    re.sub(r"[-_.]+", "-", package.name).lower(): (
        package.version,
        package.url,
        package.hash.sha256,
    )
    for package in requirements
}
assert len(requirements) == 179
assert actual == expected
"""
        real_run(
            [sys.executable, "-c", probe],
            check=True,
            env=kwargs["env"],
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", probe_child_environment)

    module._run_standard_conda_lock(ENVIRONMENT, seed_path, "linux-64")


def test_standard_handoff_rejects_resolver_artifact_drift_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Production break caught: a resolver-selected record with a seed package's
    # name must fail closed when version, URL, or hash drifts; the shim may only
    # add records that are entirely absent from the resolver output.
    module = _seed_module()
    seed_path = tmp_path / "seed.lock.yml"
    seed = module.build_seed_lock(
        _committed_pip_report(),
        PACKAGE_VERSIONS,
        _runtime_environment(),
    )
    write_conda_lock_file(seed, seed_path, metadata_choices=frozenset())
    real_run = subprocess.run

    def probe_child_environment(command, **kwargs):
        probe = """
from conda_lock._vendor.poetry.core.packages.package import Package
from conda_lock._vendor.poetry.installation.operations.install import Install
from conda_lock.pypi_solver import PlatformEnv, Pool, get_requirements

drifted = Install(
    Package(
        "torch",
        "2.0.1",
        source_type="url",
        source_url=(
            "https://files.pythonhosted.org/drifted/torch-2.0.1.whl"
            + "#sha256="
            + "0" * 64
        ),
    )
)
try:
    get_requirements(
        [drifted],
        "linux-64",
        Pool(),
        PlatformEnv("3.10.14", "linux-64"),
    )
except RuntimeError as error:
    assert "resolver artifact drift for torch" in str(error)
else:
    raise AssertionError("resolver artifact drift was silently overwritten")
"""
        real_run(
            [sys.executable, "-c", probe],
            check=True,
            env=kwargs["env"],
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", probe_child_environment)

    module._run_standard_conda_lock(ENVIRONMENT, seed_path, "linux-64")


def test_runtime_lock_generation_hands_seed_to_standard_solver_then_audits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Production break caught: handing conda-lock an empty/non-seed lock or
    # publishing its output before the independent final audit.
    module = _seed_module()
    output = tmp_path / "runtime.lock.yml"

    def complete_standard_lock(
        environment_path: Path,
        seed_path: Path,
        platform: str,
    ) -> None:
        assert environment_path == ENVIRONMENT
        assert platform == "linux-64"
        assert not output.exists()
        seed = parse_conda_lock_file(seed_path)
        assert len(seed.package) == 179
        assert all(package.manager == "pip" for package in seed.package)
        _complete_seed_lock(seed_path)

    monkeypatch.setattr(module, "_run_standard_conda_lock", complete_standard_lock)

    assert module.generate_runtime_lock(
        PIP_REPORT,
        CONSTRAINTS,
        ENVIRONMENT,
        output,
    ) == output

    final = parse_conda_lock_file(output)
    assert len([package for package in final.package if package.manager == "pip"]) == 179
    assert len(
        [package for package in final.package if package.manager == "conda"]
    ) == len(module.EXPECTED_CONDA_PINS)
    module.verify_final_runtime_lock(
        output,
        _committed_pip_report(),
        PACKAGE_VERSIONS,
    )


@pytest.mark.parametrize(
    ("production_break", "match"),
    [
        ("pip_url", "pip artifact"),
        ("pip_hash", "pip artifact"),
        ("pip_version", "pip artifact"),
        ("conda_hash", "conda.*hash"),
        ("conda_version", "python"),
        ("content_hash", "content hash"),
    ],
)
def test_final_runtime_lock_audit_rejects_artifact_or_metadata_drift(
    production_break: str,
    match: str,
    tmp_path: Path,
) -> None:
    # Production breaks caught: trusting the solver without independently
    # matching the report, conda hashes, direct pins, and environment content.
    module = _seed_module()
    lock_path = tmp_path / "candidate.lock.yml"
    seed = module.build_seed_lock(
        _committed_pip_report(),
        PACKAGE_VERSIONS,
        _runtime_environment(),
    )
    write_conda_lock_file(seed, lock_path, metadata_choices=frozenset())
    _complete_seed_lock(lock_path)
    lock = parse_conda_lock_file(lock_path)
    pip_package = next(package for package in lock.package if package.manager == "pip")
    conda_package = next(
        package
        for package in lock.package
        if package.manager == "conda" and package.name == "python"
    )
    if production_break == "pip_url":
        pip_package.url = "https://example.invalid/drifted.whl"
    elif production_break == "pip_hash":
        pip_package.hash.sha256 = "d" * 64
    elif production_break == "pip_version":
        pip_package.version = "0.0.0"
    elif production_break == "conda_hash":
        pass
    elif production_break == "conda_version":
        conda_package.version = "3.11.9"
    elif production_break == "content_hash":
        lock.metadata.content_hash["linux-64"] = "e" * 64
    else:  # pragma: no cover - the table above is exhaustive
        raise AssertionError(production_break)
    write_conda_lock_file(lock, lock_path, metadata_choices=frozenset())
    if production_break == "conda_hash":
        payload = yaml.safe_load(lock_path.read_text())
        python_package = next(
            package
            for package in payload["package"]
            if package["manager"] == "conda" and package["name"] == "python"
        )
        python_package["hash"] = {}
        lock_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(module.PipReportMismatch, match=match):
        module.verify_final_runtime_lock(
            lock_path,
            _committed_pip_report(),
            PACKAGE_VERSIONS,
        )


def test_runtime_lock_generation_failure_leaves_output_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Production break caught: moving the temporary seed to the requested path
    # after conda-lock crashes or before its result can be audited.
    module = _seed_module()
    output = tmp_path / "runtime.lock.yml"

    def fail_after_seed(
        environment_path: Path,
        seed_path: Path,
        platform: str,
    ) -> None:
        assert parse_conda_lock_file(seed_path).package
        raise RuntimeError("injected conda-lock failure")

    monkeypatch.setattr(module, "_run_standard_conda_lock", fail_after_seed)

    with pytest.raises(RuntimeError, match="injected conda-lock failure"):
        module.generate_runtime_lock(
            PIP_REPORT,
            CONSTRAINTS,
            ENVIRONMENT,
            output,
        )

    assert not output.exists()


def test_runtime_lock_generation_rejects_conflicting_output_without_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Production break caught: silently replacing an existing lock whose bytes
    # are not a valid audited result for the committed inputs.
    module = _seed_module()
    output = tmp_path / "runtime.lock.yml"
    output.write_bytes(b"conflicting lock")
    before = output.read_bytes()

    def unexpected_solver(*args) -> None:
        raise AssertionError("solver must not run for a conflicting output")

    monkeypatch.setattr(module, "_run_standard_conda_lock", unexpected_solver)

    with pytest.raises(module.PipReportMismatch, match="conflict"):
        module.generate_runtime_lock(
            PIP_REPORT,
            CONSTRAINTS,
            ENVIRONMENT,
            output,
        )

    assert output.read_bytes() == before


def test_existing_audited_runtime_lock_is_reused_without_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Production break caught: treating an exact, independently audited output
    # as a conflict or regenerating it non-idempotently.
    module = _seed_module()
    output = tmp_path / "runtime.lock.yml"
    seed = module.build_seed_lock(
        _committed_pip_report(),
        PACKAGE_VERSIONS,
        _runtime_environment(),
    )
    write_conda_lock_file(seed, output, metadata_choices=frozenset())
    _complete_seed_lock(output)
    before = output.read_bytes()

    def unexpected_solver(*args) -> None:
        raise AssertionError("solver must not rerun for an audited output")

    monkeypatch.setattr(module, "_run_standard_conda_lock", unexpected_solver)

    assert module.generate_runtime_lock(
        PIP_REPORT,
        CONSTRAINTS,
        ENVIRONMENT,
        output,
    ) == output
    assert output.read_bytes() == before


def test_runtime_lock_rejects_entry_without_hash(tmp_path: Path) -> None:
    lock = tmp_path / "runtime.lock.yml"
    lock.write_text(
        yaml.safe_dump(
            {
                "package": [
                    {
                        "name": "mmdet3d",
                        "version": "1.3.0",
                        "manager": "pip",
                        "platform": "linux-64",
                        "url": "https://example.invalid/mmdet3d.whl",
                        "hash": {},
                    }
                ]
            }
        )
    )
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("mmdet3d==1.3.0\n")

    with pytest.raises(EnvironmentMismatch, match="hash"):
        check_module.audit_runtime_lock(lock, constraints)


def test_runtime_lock_rejects_constraint_version_drift(tmp_path: Path) -> None:
    lock = tmp_path / "runtime.lock.yml"
    lock.write_text(
        yaml.safe_dump(
            {
                "package": [
                    {
                        "name": "mmdet3d",
                        "version": "1.2.0",
                        "manager": "pip",
                        "platform": "linux-64",
                        "url": "https://example.invalid/mmdet3d.whl",
                        "hash": {"sha256": "a" * 64},
                    }
                ]
            }
        )
    )
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("mmdet3d==1.3.0\n")

    with pytest.raises(EnvironmentMismatch, match="mmdet3d"):
        check_module.audit_runtime_lock(lock, constraints)


def test_runtime_lock_has_hashes_and_matches_all_constraints() -> None:
    audited = check_module.audit_runtime_lock(RUNTIME_LOCK, CONSTRAINTS)
    constraints = {
        line.split("==", 1)[0]: line.split("==", 1)[1]
        for line in CONSTRAINTS.read_text().splitlines()
        if line and not line.startswith("#")
    }

    assert audited["constraint_versions"] == constraints
    assert audited["artifact_count"] > len(constraints)
    assert re.fullmatch(r"[0-9a-f]{64}", audited["lock_sha256"])


def test_runtime_environment_pins_cuda_channel_and_build_toolchain() -> None:
    environment = yaml.safe_load(ENVIRONMENT.read_text())

    assert environment["channels"] == [
        "pytorch",
        "nvidia/label/cuda-11.8.0",
        "conda-forge",
    ]
    assert environment["dependencies"][:7] == [
        "python=3.10.14",
        "pip=23.3.2",
        "setuptools=68.2.2",
        "wheel=0.41.3",
        "packaging=23.2",
        "cuda=11.8.0",
        "cuda-toolkit=11.8.0",
    ]
    assert environment["dependencies"][7]["pip"][:2] == [
        "torch @ https://download-r2.pytorch.org/whl/cu118/"
        "torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl"
        "#sha256=a7a49d459bf4862f64f7bc1a68beccf8881c2fa9f3e0569608e16ba6f85ebf7b",
        "torchvision @ https://download-r2.pytorch.org/whl/cu118/"
        "torchvision-0.15.2%2Bcu118-cp310-cp310-linux_x86_64.whl"
        "#sha256=19ca4ab5d6179bbe53cff79df1a855ee6533c2861ddc7389f68349d8b9f8302a",
    ]


def test_runtime_lock_keeps_nvidia_packages_on_cuda_11_8_label() -> None:
    lock = parse_conda_lock_file(RUNTIME_LOCK)
    conda_packages = {
        package.name: package
        for package in lock.package
        if package.manager == "conda" and package.platform == "linux-64"
    }
    nvidia_packages = [
        package
        for package in conda_packages.values()
        if "conda.anaconda.org/nvidia/" in package.url
    ]

    assert conda_packages["setuptools"].version == "68.2.2"
    assert conda_packages["wheel"].version == "0.41.3"
    assert conda_packages["packaging"].version == "23.2"
    assert conda_packages["cuda-runtime"].version == "11.8.0"
    assert nvidia_packages
    assert all(
        "conda.anaconda.org/nvidia/label/cuda-11.8.0/" in package.url
        for package in nvidia_packages
    )


def test_bootstrap_spec_is_minimal_and_exact() -> None:
    environment = yaml.safe_load(BOOTSTRAP_ENVIRONMENT.read_text())

    assert environment["channels"] == ["conda-forge"]
    assert environment["dependencies"] == [
        "python=3.10.14",
        "pip=23.3.2",
        "conda-lock=2.5.7",
    ]


def test_bootstrap_explicit_lock_has_hash_for_every_artifact() -> None:
    audited = check_module.audit_bootstrap_lock(BOOTSTRAP_LOCK)

    assert audited["artifact_count"] > 3
    assert audited["direct_versions"] == {
        "python": "3.10.14",
        "pip": "23.3.2",
        "conda-lock": "2.5.7",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", audited["lock_sha256"])


def test_dockerfile_uses_corrected_micromamba_digest_and_pinned_cuda_digest() -> None:
    # Production break caught: restoring the invalid micromamba digest or
    # allowing either base image to float makes Docker resolution non-reproducible.
    from_lines = [
        line.strip()
        for line in DOCKERFILE.read_text().splitlines()
        if line.strip().upper().startswith("FROM ")
    ]

    assert from_lines == [
        f"FROM --platform=linux/amd64 {MICROMAMBA_IMAGE} AS micromamba",
        f"FROM --platform=linux/amd64 {CUDA_IMAGE}",
    ]


def test_dockerfile_installs_only_from_generated_locks() -> None:
    dockerfile = DOCKERFILE.read_text()
    lower = dockerfile.lower()

    assert "curl" not in lower
    assert "wget" not in lower
    assert "apt-get install" not in lower
    assert "micromamba install" not in lower
    assert "micromamba create --yes --prefix /opt/bootstrap --file" in lower
    assert (
        "/opt/bootstrap/bin/conda-lock install --micromamba "
        "--prefix /opt/resilient-v2x "
        "/tmp/environment-linux-64.lock.yml"
    ) in lower
    assert "conda env create" not in lower
    assert "pip install -r" not in lower


def test_dockerfile_preserves_dependency_layer_and_non_root_runtime() -> None:
    dockerfile = DOCKERFILE.read_text()
    source_copy = dockerfile.index("COPY transvision /workspace/transvision/transvision")
    lock_install = dockerfile.index("/opt/bootstrap/bin/conda-lock install")
    build_ops = dockerfile.index("scripts/build_resilient_v2x_ops.sh")
    user_lines = [
        line.split(maxsplit=1)[1]
        for line in dockerfile.splitlines()
        if line.strip().upper().startswith("USER ")
    ]

    assert lock_install < source_copy < build_ops
    assert user_lines
    assert user_lines[-1] not in {"0", "root", "0:0"}
    assert "transvision.models.voxel.voxel_layer" in dockerfile
    assert "transvision.models.bev_pool.bev_pool_ext" in dockerfile


@pytest.mark.parametrize(
    "forbidden",
    ["DAIR-V2X", "checkpoint.pth", "id_rsa", ".ssh", "external_baseline"],
)
def test_dockerfile_does_not_copy_restricted_experiment_assets(forbidden: str) -> None:
    assert forbidden not in DOCKERFILE.read_text()


def test_custom_op_builder_is_strict_isolated_and_smokes_both_extensions() -> None:
    script = BUILD_OPS.read_text()

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "mktemp -d" in script
    assert "FORCE_CUDA=1" in script
    assert "transvision.models.voxel.voxel_layer" in script
    assert "transvision.models.bev_pool.bev_pool_ext" in script
    subprocess.run(["bash", "-n", str(BUILD_OPS)], check=True)


def test_development_checker_reports_unavailable_checks_not_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = expected_environment(world_size=1)
    expected["container_image_digest"] = None
    monkeypatch.setattr(check_module, "capture_environment", lambda world_size: development_capture())
    monkeypatch.setattr(check_module, "expected_environment_contract", lambda: expected)

    assert check_module.main(["--mode", "development"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["accepted"] is False
    assert report["not_executed_checks"]
    assert not set(report["not_executed_checks"]) & set(report["verified_fields"])


def test_environment_checker_is_directly_executable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "resilient_v2x" / "check_environment.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--mode" in result.stdout


def test_manifest_schema_itself_is_strict() -> None:
    schema = json.loads(SCHEMA.read_text())

    assert schema["additionalProperties"] is False
    assert schema["properties"]["hardware"]["additionalProperties"] is False
    assert (
        schema["properties"]["hardware"]["properties"]["gpu_devices"]["items"][
            "additionalProperties"
        ]
        is False
    )


def test_environment_test_process_uses_mandated_interpreter() -> None:
    assert Path(sys.executable).resolve() == (
        ROOT / ".venv" / "resilient-v2x-dev" / "bin" / "python"
    ).resolve()
