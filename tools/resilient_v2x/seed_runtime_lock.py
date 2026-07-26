#!/usr/bin/env python3
"""Generate the ResilientV2X runtime lock from an audited Linux pip report."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, Mapping, Sequence
from urllib.parse import urlsplit

import conda_lock
import yaml
from conda_lock.lockfile import parse_conda_lock_file, write_conda_lock_file
from conda_lock.lockfile.v1.models import HashModel, LockMeta
from conda_lock.lockfile.v2prelim.models import LockedDependency, Lockfile
from conda_lock.models.channel import Channel
from conda_lock.models.lock_spec import LockSpecification, VersionedDependency
from conda_lock.src_parser.environment_yaml import conda_spec_to_versioned_dep
from conda_lock.src_parser.pyproject_toml import parse_python_requirement
from conda_lock.virtual_package import default_virtual_package_repodata


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_DIR = ROOT / "environments" / "resilient_v2x"
DEFAULT_REPORT = ENVIRONMENT_DIR / "runtime-pip-report.json"
DEFAULT_CONSTRAINTS = ENVIRONMENT_DIR / "constraints.txt"
DEFAULT_ENVIRONMENT = ENVIRONMENT_DIR / "environment.yml"
DEFAULT_OUTPUT = ENVIRONMENT_DIR / "environment-linux-64.lock.yml"
SUPPORTED_PLATFORM = "linux-64"
EXPECTED_PIP_REPORT_COUNT = 190
EXPECTED_PYTHON = "3.10.14"
EXPECTED_PIP = "23.3.2"
EXPECTED_CONDA_LOCK = "2.5.7"
EXPECTED_CUDA = "11.8.0"
EXPECTED_CHANNELS = ("pytorch", "nvidia", "conda-forge")
AUDITED_MANYLINUX_TAGS = ("_2_31", "_2_34")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MD5 = re.compile(r"^[0-9a-f]{32}$")
MANYLINUX_X86_64 = re.compile(r"manylinux_2_(\d+)_x86_64")

APPROVED_CONSTRAINTS = {
    "torch": "2.0.1",
    "torchvision": "0.15.2",
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
    "pyyaml": "6.0.2",
    "conda-lock": "2.5.7",
}
EXPECTED_CONDA_PINS = {
    "python": EXPECTED_PYTHON,
    "pip": EXPECTED_PIP,
    "cuda": EXPECTED_CUDA,
    "cuda-toolkit": EXPECTED_CUDA,
}


class PipReportMismatch(RuntimeError):
    """A Linux pip report cannot safely seed the requested runtime lock."""


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _require_runtime() -> None:
    if sys.version_info[:2] != (3, 10):
        raise PipReportMismatch(
            "seed adapter requires Python 3.10; "
            f"running {sys.version_info.major}.{sys.version_info.minor}"
        )
    if conda_lock.__version__ != EXPECTED_CONDA_LOCK:
        raise PipReportMismatch(
            "seed adapter requires conda-lock==2.5.7; "
            f"running {conda_lock.__version__}"
        )


def _canonical_constraints(
    expected_constraints: Mapping[str, str],
) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for raw_name, raw_version in expected_constraints.items():
        if not isinstance(raw_name, str) or not isinstance(raw_version, str):
            raise PipReportMismatch("constraints must map package names to versions")
        name = _canonical_name(raw_name)
        if name in canonical:
            raise PipReportMismatch(f"duplicate direct constraint for {name}")
        canonical[name] = raw_version
    if canonical != APPROVED_CONSTRAINTS:
        missing = sorted(set(APPROVED_CONSTRAINTS) - set(canonical))
        unexpected = sorted(set(canonical) - set(APPROVED_CONSTRAINTS))
        drifted = sorted(
            name
            for name in set(canonical) & set(APPROVED_CONSTRAINTS)
            if canonical[name] != APPROVED_CONSTRAINTS[name]
        )
        raise PipReportMismatch(
            "approved direct constraints changed: "
            f"missing={missing}, unexpected={unexpected}, drifted={drifted}"
        )
    return canonical


def _validate_environment(environment: Mapping[str, object]) -> None:
    if environment.get("name") != "resilient-v2x":
        raise PipReportMismatch("environment name must be resilient-v2x")
    channels = environment.get("channels")
    if channels != list(EXPECTED_CHANNELS):
        raise PipReportMismatch(
            f"environment channels changed: expected {list(EXPECTED_CHANNELS)!r}"
        )
    dependencies = environment.get("dependencies")
    if not isinstance(dependencies, list):
        raise PipReportMismatch("environment dependencies must be a list")

    conda_pins: dict[str, str] = {}
    pip_sections: list[Mapping[str, object]] = []
    for dependency in dependencies:
        if isinstance(dependency, str):
            if dependency.count("=") != 1:
                raise PipReportMismatch(
                    f"conda dependency is not an exact pin: {dependency!r}"
                )
            name, version = dependency.split("=", 1)
            if not name or not version or name in conda_pins:
                raise PipReportMismatch(
                    f"invalid or duplicate conda dependency: {dependency!r}"
                )
            conda_pins[name] = version
        elif isinstance(dependency, Mapping):
            pip_sections.append(dependency)
        else:
            raise PipReportMismatch("environment dependency is malformed")
    if conda_pins != EXPECTED_CONDA_PINS:
        raise PipReportMismatch(
            "required conda pins changed: "
            f"expected={EXPECTED_CONDA_PINS!r}, actual={conda_pins!r}"
        )
    if len(pip_sections) != 1 or set(pip_sections[0]) != {"pip"}:
        raise PipReportMismatch("environment must contain exactly one pip section")
    pip_dependencies = pip_sections[0]["pip"]
    if not isinstance(pip_dependencies, list):
        raise PipReportMismatch("environment pip dependencies must be a list")
    parsed: dict[str, str] = {}
    for dependency in pip_dependencies:
        if not isinstance(dependency, str) or dependency.count("==") != 1:
            raise PipReportMismatch(
                f"pip dependency is not an exact pin: {dependency!r}"
            )
        name, version = dependency.split("==", 1)
        canonical = _canonical_name(name)
        if canonical in parsed:
            raise PipReportMismatch(f"duplicate environment pip pin for {canonical}")
        parsed[canonical] = version
    if parsed != APPROVED_CONSTRAINTS:
        raise PipReportMismatch("environment pip pins changed from approved constraints")


def _environment_lock_spec(
    environment: Mapping[str, object],
    platform: str,
) -> LockSpecification:
    """Reproduce conda-lock 2.5.7's environment.yml content-hash model locally."""
    _validate_environment(environment)
    dependencies: list[VersionedDependency] = []
    has_pip_section = False
    for dependency in environment["dependencies"]:
        if isinstance(dependency, str):
            dependencies.append(conda_spec_to_versioned_dep(dependency, "main"))
            continue
        has_pip_section = True
        for requirement in dependency["pip"]:
            dependencies.append(
                parse_python_requirement(
                    requirement,
                    manager="pip",
                    category="main",
                    normalize_name=False,
                )
            )

    # The child compatibility shim keeps an existing explicit conda-side pip
    # pin instead of allowing conda-lock 2.5.7's implicit "*" requirement to
    # replace it. Mirror that repaired specification for the content hash.
    if has_pip_section and not any(
        dependency.manager == "conda" and dependency.name == "pip"
        for dependency in dependencies
    ):
        dependencies.append(
            VersionedDependency(
                name="pip",
                manager="conda",
                category="main",
                version="*",
            )
        )
    unique = {
        (dependency.manager, dependency.name): dependency
        for dependency in dependencies
    }
    virtual_packages = default_virtual_package_repodata(cuda_version="11.8")
    return LockSpecification(
        dependencies={platform: list(unique.values())},
        channels=[
            Channel.from_string(channel) for channel in environment["channels"]
        ],
        sources=[Path("environment.yml")],
        virtual_package_repo=virtual_packages,
    )


def _environment_content_hash(
    environment: Mapping[str, object],
    platform: str,
) -> str:
    try:
        return _environment_lock_spec(environment, platform).content_hash_for_platform(
            platform
        )
    except Exception as error:
        if isinstance(error, PipReportMismatch):
            raise
        raise PipReportMismatch(
            f"cannot compute environment content hash: {error}"
        ) from error


def _report_artifacts(
    report: Mapping[str, object],
    expected_constraints: Mapping[str, str],
) -> dict[str, tuple[str, str, str]]:
    constraints = _canonical_constraints(expected_constraints)
    if report.get("version") != "1":
        raise PipReportMismatch("pip report version must be 1")
    if report.get("pip_version") != EXPECTED_PIP:
        raise PipReportMismatch(
            f"pip report must declare pip {EXPECTED_PIP}"
        )
    environment = report.get("environment")
    if not isinstance(environment, Mapping):
        raise PipReportMismatch("pip report environment is missing")
    if (
        environment.get("platform_system") != "Linux"
        or environment.get("sys_platform") != "linux"
        or environment.get("os_name") != "posix"
    ):
        raise PipReportMismatch("pip report must describe native Linux")
    if environment.get("platform_machine") != "x86_64":
        raise PipReportMismatch("pip report must describe Linux x86_64")
    if (
        environment.get("python_full_version") != EXPECTED_PYTHON
        or environment.get("python_version") != "3.10"
        or environment.get("implementation_name") != "cpython"
    ):
        raise PipReportMismatch(
            f"pip report must describe CPython {EXPECTED_PYTHON}"
        )
    install = report.get("install")
    if not isinstance(install, list):
        raise PipReportMismatch(
            "pip report must contain an artifact list"
        )

    artifacts: dict[str, tuple[str, str, str]] = {}
    requested: dict[str, str] = {}
    for index, item in enumerate(install):
        if not isinstance(item, Mapping):
            raise PipReportMismatch(f"pip report item[{index}] is malformed")
        metadata = item.get("metadata")
        download = item.get("download_info")
        if not isinstance(metadata, Mapping) or not isinstance(download, Mapping):
            raise PipReportMismatch(
                f"pip report item[{index}] lacks metadata or download information"
            )
        raw_name = metadata.get("name")
        version = metadata.get("version")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise PipReportMismatch(f"pip report item[{index}] has no package name")
        if not isinstance(version, str) or not version.strip():
            raise PipReportMismatch(
                f"pip report item[{index}] {raw_name!r} has no version"
            )
        name = _canonical_name(raw_name)
        if name in artifacts:
            raise PipReportMismatch(f"duplicate canonical pip package name: {name}")
        url = download.get("url")
        if not isinstance(url, str):
            raise PipReportMismatch(f"pip report item[{index}] {name} has no URL")
        parsed_url = urlsplit(url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise PipReportMismatch(
                f"pip report item[{index}] {name} URL is not HTTPS"
            )
        if parsed_url.username is not None or parsed_url.password is not None:
            raise PipReportMismatch(
                f"pip report item[{index}] {name} contains credentialed URL"
            )
        archive = download.get("archive_info")
        hashes = archive.get("hashes") if isinstance(archive, Mapping) else None
        digest = hashes.get("sha256") if isinstance(hashes, Mapping) else None
        if (
            not isinstance(hashes, Mapping)
            or set(hashes) != {"sha256"}
            or not isinstance(digest, str)
            or not SHA256.fullmatch(digest)
        ):
            raise PipReportMismatch(
                f"pip report item[{index}] {name} must have one 64-hex SHA-256"
            )
        declared_hash = archive.get("hash") if isinstance(archive, Mapping) else None
        if declared_hash != f"sha256={digest}":
            raise PipReportMismatch(
                f"pip report item[{index}] {name} has inconsistent SHA-256 fields"
            )
        artifacts[name] = (version, url, digest)
        if item.get("requested") is True:
            requested[name] = version

    if len(install) != EXPECTED_PIP_REPORT_COUNT:
        raise PipReportMismatch(
            f"pip report must contain exactly 190 artifacts; found {len(install)}"
        )
    if requested != constraints:
        missing = sorted(set(constraints) - set(requested))
        unexpected = sorted(set(requested) - set(constraints))
        drifted = sorted(
            name
            for name in set(requested) & set(constraints)
            if requested[name] != constraints[name]
        )
        details = f"missing={missing}, unexpected={unexpected}, drifted={drifted}"
        if drifted:
            details += f", first drift={drifted[0]}"
        raise PipReportMismatch(
            f"pip report requested pins do not match approved constraints: {details}"
        )
    return artifacts


def build_seed_lock(
    report: Mapping[str, object],
    expected_constraints: Mapping[str, str],
    environment: Mapping[str, object],
    platform: Literal["linux-64"] = "linux-64",
) -> Lockfile:
    _require_runtime()
    if platform != SUPPORTED_PLATFORM:
        raise PipReportMismatch(f"unsupported seed platform: {platform}")
    artifacts = _report_artifacts(report, expected_constraints)
    content_hash = _environment_content_hash(environment, platform)
    packages = [
        LockedDependency(
            name=name,
            version=version,
            manager="pip",
            platform=platform,
            dependencies={},
            url=url,
            hash=HashModel(sha256=digest),
            category="main",
        )
        for name, (version, url, digest) in sorted(artifacts.items())
    ]
    return Lockfile(
        package=packages,
        metadata=LockMeta(
            content_hash={platform: content_hash},
            channels=[
                Channel.from_string(channel) for channel in environment["channels"]
            ],
            platforms=[platform],
            sources=["environment.yml"],
        ),
    )


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PipReportMismatch(f"cannot read pip report {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise PipReportMismatch("pip report root must be an object")
    return value


def _read_constraints(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise PipReportMismatch(f"cannot read constraints {path}: {error}") from error
    constraints: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.count("==") != 1:
            raise PipReportMismatch(f"constraint is not exact: {stripped!r}")
        name, version = stripped.split("==", 1)
        canonical = _canonical_name(name)
        if canonical in constraints:
            raise PipReportMismatch(f"duplicate direct constraint for {canonical}")
        constraints[canonical] = version
    return _canonical_constraints(constraints)


def _read_environment(path: Path) -> Mapping[str, object]:
    try:
        environment = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise PipReportMismatch(f"cannot read environment {path}: {error}") from error
    if not isinstance(environment, Mapping):
        raise PipReportMismatch("environment root must be an object")
    _validate_environment(environment)
    return environment


def _valid_package_hash(package: LockedDependency) -> bool:
    return bool(
        (
            isinstance(package.hash.sha256, str)
            and SHA256.fullmatch(package.hash.sha256)
        )
        or (
            isinstance(package.hash.md5, str)
            and MD5.fullmatch(package.hash.md5)
        )
    )


def verify_final_runtime_lock(
    lock_path: Path,
    report: Mapping[str, object],
    expected_constraints: Mapping[str, str],
    platform: Literal["linux-64"] = "linux-64",
    *,
    environment_path: Path | None = None,
) -> None:
    _require_runtime()
    if platform != SUPPORTED_PLATFORM:
        raise PipReportMismatch(f"unsupported final-lock platform: {platform}")
    artifacts = _report_artifacts(report, expected_constraints)
    source_path = environment_path or DEFAULT_ENVIRONMENT
    environment = _read_environment(source_path)
    expected_content_hash = _environment_content_hash(environment, platform)
    try:
        lock = parse_conda_lock_file(lock_path)
    except Exception as error:
        raise PipReportMismatch(
            f"cannot parse final runtime lock: {error}"
        ) from error
    if lock.metadata.platforms != [platform]:
        raise PipReportMismatch(
            f"final lock platforms changed: {lock.metadata.platforms!r}"
        )
    if lock.metadata.content_hash.get(platform) != expected_content_hash:
        raise PipReportMismatch(
            "final lock content hash does not match committed environment.yml"
        )

    pip_artifacts: dict[str, tuple[str, str, str | None]] = {}
    conda_versions: dict[str, set[str]] = {}
    conda_count = 0
    for package in lock.package:
        if package.platform != platform:
            raise PipReportMismatch(
                f"final lock package {package.name} has platform {package.platform}"
            )
        parsed_url = urlsplit(package.url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise PipReportMismatch(
                f"final lock package {package.name} has unsafe artifact URL"
            )
        if package.manager == "pip":
            name = _canonical_name(package.name)
            if name in pip_artifacts:
                raise PipReportMismatch(
                    f"final lock has duplicate pip package {name}"
                )
            pip_artifacts[name] = (
                package.version,
                package.url,
                package.hash.sha256,
            )
        elif package.manager == "conda":
            conda_count += 1
            if not _valid_package_hash(package):
                raise PipReportMismatch(
                    f"conda package {package.name} has no cryptographic hash"
                )
            conda_versions.setdefault(_canonical_name(package.name), set()).add(
                package.version
            )
        else:
            raise PipReportMismatch(
                f"final lock package {package.name} has unknown manager"
            )
    if pip_artifacts != artifacts:
        missing = sorted(set(artifacts) - set(pip_artifacts))
        unexpected = sorted(set(pip_artifacts) - set(artifacts))
        drifted = sorted(
            name
            for name in set(pip_artifacts) & set(artifacts)
            if pip_artifacts[name] != artifacts[name]
        )
        raise PipReportMismatch(
            "final pip artifact set differs from audited report: "
            f"missing={missing}, unexpected={unexpected}, drifted={drifted}"
        )
    if conda_count == 0:
        raise PipReportMismatch("final runtime lock has no conda artifacts")
    for name, version in EXPECTED_CONDA_PINS.items():
        resolved = conda_versions.get(name, set())
        if resolved != {version}:
            raise PipReportMismatch(
                f"final conda pin mismatch for {name}: "
                f"expected {version!r}, resolved {sorted(resolved)!r}"
            )


def _run_standard_conda_lock(
    environment_path: Path,
    seed_path: Path,
    platform: Literal["linux-64"],
) -> None:
    required_manylinux_tags: set[str] = set()
    try:
        seed = parse_conda_lock_file(seed_path)
    except Exception as error:
        raise PipReportMismatch(
            f"cannot inspect temporary seed lock: {error}"
        ) from error
    for package in seed.package:
        if package.manager != "pip":
            raise PipReportMismatch(
                "temporary runtime seed must contain pip records only"
            )
        filename = Path(urlsplit(package.url).path).name
        for match in MANYLINUX_X86_64.finditer(filename):
            tag = f"_2_{match.group(1)}"
            if int(match.group(1)) <= 28:
                continue
            if tag not in AUDITED_MANYLINUX_TAGS:
                raise PipReportMismatch(
                    f"temporary seed requires unauthorized manylinux tag {tag}"
                )
            required_manylinux_tags.add(tag)
    if required_manylinux_tags != set(AUDITED_MANYLINUX_TAGS):
        raise PipReportMismatch(
            "temporary seed does not contain the complete audited newer "
            f"manylinux tag set: {sorted(required_manylinux_tags)!r}"
        )

    command = [
        sys.executable,
        "-m",
        "conda_lock",
        "lock",
        "--file",
        str(environment_path),
        "--platform",
        platform,
        "--lockfile",
        str(seed_path),
        "--micromamba",
        "--no-mamba",
        "--with-cuda",
        "11.8",
        "--log-level",
        "INFO",
    ]
    with tempfile.TemporaryDirectory(
        prefix="resilient-v2x-conda-lock-compat-"
    ) as compatibility_directory:
        compatibility_root = Path(compatibility_directory)
        seed_snapshot = compatibility_root / "audited-pip-seed.lock.yml"
        shutil.copyfile(seed_path, seed_snapshot)
        seed_snapshot.chmod(0o400)
        environment_snapshot = compatibility_root / "environment.yml"
        shutil.copyfile(environment_path, environment_snapshot)
        environment_snapshot.chmod(0o400)
        (compatibility_root / "sitecustomize.py").write_text(
            """\
import os as _os
import re as _re
from pathlib import Path as _Path

from conda_lock.lockfile import parse_conda_lock_file as _parse_lock
import conda_lock.pypi_solver as _solver
import conda_lock.src_parser.environment_yaml as _environment_yaml

_allowed = {"_2_31", "_2_34"}
_requested = tuple(
    tag
    for tag in _os.environ[
        "RESILIENT_V2X_CONDA_LOCK_MANYLINUX_TAGS"
    ].split(",")
    if tag
)
if set(_requested) != _allowed:
    raise RuntimeError(
        "unexpected ResilientV2X manylinux compatibility tags: "
        + repr(_requested)
    )
_original = _solver._compute_compatible_manylinux_tags

def _with_audited_manylinux_tags(platform_virtual_packages):
    original = _original(platform_virtual_packages)
    additions = [tag for tag in reversed(_requested) if tag not in original]
    return additions + original

_solver._compute_compatible_manylinux_tags = _with_audited_manylinux_tags

_environment_path = _Path(
    _os.environ["RESILIENT_V2X_CONDA_LOCK_ENVIRONMENT_SNAPSHOT"]
)
_environment_content = _environment_path.read_text()
_original_parse_environment = (
    _environment_yaml._parse_environment_file_for_platform
)

def _with_explicit_conda_pip_pin(content, category, platform):
    if content != _environment_content:
        raise RuntimeError(
            "conda-lock parsed an unexpected environment source"
        )
    dependencies = _original_parse_environment(
        content,
        category,
        platform,
    )
    explicit_pip = [
        dependency
        for dependency in dependencies
        if (
            dependency.manager == "conda"
            and dependency.name == "pip"
            and dependency.version != "*"
        )
    ]
    if len(explicit_pip) != 1:
        raise RuntimeError(
            "committed environment must contain one explicit conda pip pin"
        )
    return [
        dependency
        for dependency in dependencies
        if not (
            dependency.manager == "conda"
            and dependency.name == "pip"
        )
    ] + explicit_pip

_environment_yaml._parse_environment_file_for_platform = (
    _with_explicit_conda_pip_pin
)

_seed_path = _Path(
    _os.environ["RESILIENT_V2X_CONDA_LOCK_SEED_SNAPSHOT"]
)
_seed = _parse_lock(_seed_path)
if len(_seed.package) != 190:
    raise RuntimeError(
        "unexpected ResilientV2X audited seed size: "
        + repr(len(_seed.package))
    )
if any(package.manager != "pip" for package in _seed.package):
    raise RuntimeError(
        "ResilientV2X audited seed snapshot must be pip-only"
    )

def _canonical_name(name):
    return _re.sub(r"[-_.]+", "-", name).lower()

_seed_by_name = {}
for _package in _seed.package:
    _name = _canonical_name(_package.name)
    if _name in _seed_by_name:
        raise RuntimeError(
            "duplicate ResilientV2X audited seed package: " + _name
        )
    _seed_by_name[_name] = _package

def _artifact_identity(package):
    return (
        str(package.version),
        package.url,
        package.hash.sha256,
    )

_original_get_requirements = _solver.get_requirements

def _with_missing_audited_seed_records(*args, **kwargs):
    requirements = _original_get_requirements(*args, **kwargs)
    resolved_by_name = {}
    for package in requirements:
        name = _canonical_name(package.name)
        if name in resolved_by_name:
            raise RuntimeError(
                "duplicate resolver pip package: " + name
            )
        resolved_by_name[name] = package
        seed_package = _seed_by_name.get(name)
        if (
            seed_package is not None
            and _artifact_identity(package)
            != _artifact_identity(seed_package)
        ):
            raise RuntimeError(
                "resolver artifact drift for " + name
            )
    return requirements + [
        _seed_by_name[name]
        for name in sorted(set(_seed_by_name) - set(resolved_by_name))
    ]

_solver.get_requirements = _with_missing_audited_seed_records
"""
        )
        child_environment = os.environ.copy()
        child_environment[
            "RESILIENT_V2X_CONDA_LOCK_MANYLINUX_TAGS"
        ] = ",".join(sorted(required_manylinux_tags))
        child_environment[
            "RESILIENT_V2X_CONDA_LOCK_SEED_SNAPSHOT"
        ] = str(seed_snapshot)
        child_environment[
            "RESILIENT_V2X_CONDA_LOCK_ENVIRONMENT_SNAPSHOT"
        ] = str(environment_snapshot)
        existing_pythonpath = child_environment.get("PYTHONPATH")
        child_environment["PYTHONPATH"] = (
            str(compatibility_root)
            if not existing_pythonpath
            else str(compatibility_root) + os.pathsep + existing_pythonpath
        )
        subprocess.run(command, check=True, env=child_environment)


def _existing_output_is_reusable(
    output_path: Path,
    report: Mapping[str, object],
    expected_constraints: Mapping[str, str],
    platform: Literal["linux-64"],
    environment_path: Path,
) -> bool:
    if not output_path.exists():
        return False
    try:
        verify_final_runtime_lock(
            output_path,
            report,
            expected_constraints,
            platform,
            environment_path=environment_path,
        )
    except PipReportMismatch as error:
        raise PipReportMismatch(
            f"output conflict at {output_path}: {error}"
        ) from error
    return True


def generate_runtime_lock(
    report_path: Path,
    constraints_path: Path,
    environment_path: Path,
    output_path: Path,
) -> Path:
    _require_runtime()
    report_path = Path(report_path).resolve()
    constraints_path = Path(constraints_path).resolve()
    environment_path = Path(environment_path).resolve()
    output_path = Path(output_path).resolve()
    report = _read_json_mapping(report_path)
    expected_constraints = _read_constraints(constraints_path)
    environment = _read_environment(environment_path)
    if _existing_output_is_reusable(
        output_path,
        report,
        expected_constraints,
        SUPPORTED_PLATFORM,
        environment_path,
    ):
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".seed",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        seed = build_seed_lock(
            report,
            expected_constraints,
            environment,
            SUPPORTED_PLATFORM,
        )
        write_conda_lock_file(
            seed,
            temporary_path,
            metadata_choices=frozenset(),
        )
        _run_standard_conda_lock(
            environment_path,
            temporary_path,
            SUPPORTED_PLATFORM,
        )
        verify_final_runtime_lock(
            temporary_path,
            report,
            expected_constraints,
            SUPPORTED_PLATFORM,
            environment_path=environment_path,
        )
        try:
            os.link(temporary_path, output_path)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
            if output_path.read_bytes() != temporary_path.read_bytes():
                raise PipReportMismatch(
                    f"output conflict at {output_path}: bytes differ"
                ) from error
            verify_final_runtime_lock(
                output_path,
                report,
                expected_constraints,
                SUPPORTED_PLATFORM,
                environment_path=environment_path,
            )
        return output_path
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = generate_runtime_lock(
        args.report,
        args.constraints,
        args.environment,
        args.output,
    )
    payload = {
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
