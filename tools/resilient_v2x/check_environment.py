#!/usr/bin/env python3
"""Audit locks and classify the current ResilientV2X execution environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.resilient_v2x.capture_environment import (  # noqa: E402
    BOOTSTRAP_LOCK,
    CONSTRAINTS,
    RUNTIME_LOCK,
    WORLD_SIZES,
    EnvironmentMismatch,
    build_environment_manifest,
    capture_environment,
    expected_environment_contract,
    validate_environment,
)


DATA_ROOT = ROOT / "data" / "DAIR-V2X"
CRYPTOGRAPHIC_HASH = re.compile(r"^(?:[0-9a-f]{32}|[0-9a-f]{64})$")


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _constraints(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.count("==") != 1:
            raise EnvironmentMismatch(f"constraint is not exact: {stripped!r}")
        name, version = stripped.split("==", 1)
        pins[name] = version
    return pins


def _has_cryptographic_hash(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return any(
        isinstance(digest, str) and CRYPTOGRAPHIC_HASH.fullmatch(digest)
        for digest in value.values()
    )


def audit_runtime_lock(lock: Path, constraints: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(lock.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise EnvironmentMismatch(f"cannot parse runtime lock: {error}") from error
    packages = payload.get("package") if isinstance(payload, Mapping) else None
    if not isinstance(packages, list) or not packages:
        raise EnvironmentMismatch("runtime lock has no package artifacts")

    resolved: dict[str, set[str]] = {}
    for index, package in enumerate(packages):
        if not isinstance(package, Mapping):
            raise EnvironmentMismatch(f"runtime lock package[{index}] is malformed")
        if not package.get("url"):
            raise EnvironmentMismatch(f"runtime lock package[{index}] has no URL")
        if not _has_cryptographic_hash(package.get("hash")):
            raise EnvironmentMismatch(
                f"runtime lock package[{index}] {package.get('name')!r} has no cryptographic hash"
            )
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise EnvironmentMismatch(
                f"runtime lock package[{index}] has invalid name/version"
            )
        resolved.setdefault(_canonical_name(name), set()).add(version)

    pins = _constraints(constraints)
    for name, version in pins.items():
        resolved_versions = resolved.get(_canonical_name(name), set())
        if version not in resolved_versions:
            raise EnvironmentMismatch(
                f"runtime lock pin mismatch for {name}: "
                f"expected {version!r}, resolved {sorted(resolved_versions)!r}"
            )
    return {
        "artifact_count": len(packages),
        "constraint_versions": pins,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    }


def audit_bootstrap_lock(lock: Path) -> dict[str, object]:
    try:
        lines = lock.read_text().splitlines()
    except OSError as error:
        raise EnvironmentMismatch(f"cannot read bootstrap lock: {error}") from error
    artifacts = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#") and line.strip() != "@EXPLICIT"
    ]
    if not artifacts:
        raise EnvironmentMismatch("bootstrap explicit lock has no artifacts")
    for index, artifact in enumerate(artifacts):
        if "#" not in artifact:
            raise EnvironmentMismatch(
                f"bootstrap lock artifact[{index}] has no cryptographic hash"
            )
        digest = artifact.rsplit("#", 1)[1]
        if not CRYPTOGRAPHIC_HASH.fullmatch(digest):
            raise EnvironmentMismatch(
                f"bootstrap lock artifact[{index}] has invalid cryptographic hash"
            )
    required = {
        "python": "3.10.14",
        "pip": "23.3.2",
        "conda-lock": "2.5.7",
    }
    filenames = [Path(urlsplit(artifact.split("#", 1)[0]).path).name for artifact in artifacts]
    for name, version in required.items():
        prefix = f"{name}-{version}-"
        if not any(filename.startswith(prefix) for filename in filenames):
            raise EnvironmentMismatch(
                f"bootstrap lock pin mismatch for {name}: expected {version!r}"
            )
    return {
        "artifact_count": len(artifacts),
        "direct_versions": required,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("development", "controlled"), required=True
    )
    parser.add_argument("--world-size", type=int, choices=WORLD_SIZES, default=1)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-custom-ops", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        audit_runtime_lock(RUNTIME_LOCK, CONSTRAINTS)
        audit_bootstrap_lock(BOOTSTRAP_LOCK)
        expected = expected_environment_contract()
        expected["world_size"] = args.world_size
        actual = capture_environment(world_size=args.world_size)
        manifest = build_environment_manifest(
            actual,
            expected,
            world_size=args.world_size,
            classification=args.mode,
        )
        validation = validate_environment(
            manifest,
            expected,
            require_cuda=args.require_cuda or args.mode == "controlled",
        )
        report = {
            "accepted": validation.accepted,
            "classification": validation.classification,
            "environment_manifest_sha256": (
                validation.environment_manifest_sha256
            ),
            "actual_hardware_fingerprint_sha256": (
                validation.actual_hardware_fingerprint_sha256
            ),
            "verified_fields": list(validation.verified_fields),
            "blocking_mismatches": list(validation.blocking_mismatches),
            "not_executed_checks": list(validation.not_executed_checks),
        }
        if args.mode == "development" and not DATA_ROOT.is_dir():
            report["not_executed_checks"].append("dair_v2x_data")
        if args.require_custom_ops and "custom_ops" not in report["verified_fields"]:
            raise EnvironmentMismatch("custom ops were required but not verified")
    except EnvironmentMismatch as error:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "classification": args.mode,
                    "blocking_mismatches": [str(error)],
                    "not_executed_checks": [],
                    "verified_fields": [],
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
