#!/usr/bin/env python3
"""Queue a created or failed standalone ClearML task under a hashed contract.

The helper never reuses an active clone, accepts no arbitrary shell command or
package URL, and prints only allowlisted identifiers and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from clearml import Task


SHA256 = re.compile(r"[0-9a-f]{64}")
SAFE_NAME = re.compile(r"[A-Za-z0-9._-]{1,160}")
TASK_NAME = re.compile(r"rtpv2x__[a-z0-9][a-z0-9_-]*(?:__[a-z0-9][a-z0-9_-]*)+")
PACKAGE = re.compile(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.+-]*")
CUDA_IMAGE = re.compile(
    r"nvidia/cuda:[0-9]+\.[0-9]+\.[0-9]+-cudnn-runtime-ubuntu[0-9]+\.[0-9]+"
    r"@sha256:[0-9a-f]{64}"
)
ALLOWED_ARGUMENTS = {
    "clearml_mode",
    "max_latency",
    "packet_loss",
    "require_a100",
    "require_gpu",
    "seed",
    "steps",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, label: str) -> str:
    if not SHA256.fullmatch(value):
        raise SystemExit(f"{label} must be a lowercase SHA-256")
    return value


def load_contract(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise SystemExit("execution contract must be a small regular file")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != require_sha256(expected_sha256, "contract hash"):
        raise SystemExit("execution contract hash mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit("execution contract is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "argument_overrides",
        "base_docker",
        "install_packages",
        "schema_version",
    }:
        raise SystemExit("execution contract fields do not match schema")
    if value["schema_version"] != 1:
        raise SystemExit("unsupported execution contract schema")
    image = value["base_docker"]
    packages = value["install_packages"]
    arguments = value["argument_overrides"]
    if not isinstance(image, str) or not CUDA_IMAGE.fullmatch(image):
        raise SystemExit("base Docker image must be an allowlisted digest-pinned CUDA image")
    if (
        not isinstance(packages, list)
        or not packages
        or len(packages) > 8
        or any(not isinstance(item, str) or not PACKAGE.fullmatch(item) for item in packages)
    ):
        raise SystemExit("install_packages must contain only exact package versions")
    if len(set(packages)) != len(packages):
        raise SystemExit("execution contract contains duplicate packages")
    if not isinstance(arguments, dict) or set(arguments) != ALLOWED_ARGUMENTS:
        raise SystemExit("argument_overrides do not match the allowlist")
    if arguments["clearml_mode"] != "required":
        raise SystemExit("remote canary must require ClearML context")
    if arguments["require_gpu"] is not True or arguments["require_a100"] is not True:
        raise SystemExit("remote canary must require an A100 GPU")
    if (
        isinstance(arguments["seed"], bool)
        or not isinstance(arguments["seed"], int)
        or isinstance(arguments["steps"], bool)
        or not isinstance(arguments["steps"], int)
        or arguments["steps"] < 24
        or isinstance(arguments["max_latency"], bool)
        or not isinstance(arguments["max_latency"], int)
        or arguments["max_latency"] < 0
        or isinstance(arguments["packet_loss"], bool)
        or not isinstance(arguments["packet_loss"], (int, float))
        or not 0.0 <= float(arguments["packet_loss"]) < 0.8
    ):
        raise SystemExit("execution contract contains invalid canary arguments")
    return value, observed_sha256


def parameter_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        rendered = str(value)
        if SAFE_NAME.fullmatch(rendered):
            return rendered
    raise SystemExit("execution contract argument cannot be serialized safely")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_task_id")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--entry-point", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--expected-diff-sha256", required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{32}", args.source_task_id):
        raise SystemExit("source task ID has invalid format")
    if not SAFE_NAME.fullmatch(args.queue):
        raise SystemExit("queue has invalid format")
    if not TASK_NAME.fullmatch(args.name):
        raise SystemExit("task name does not match the RTP-V2X naming contract")
    expected_diff = require_sha256(args.expected_diff_sha256, "expected diff hash")
    expected_script = require_sha256(args.expected_script_sha256, "expected script hash")
    entry_point = PurePosixPath(args.entry_point)
    if entry_point.is_absolute() or entry_point.name != args.entry_point:
        raise SystemExit("entry point must be a single relative file name")

    contract, contract_sha256 = load_contract(
        args.execution_contract, args.expected_contract_sha256
    )
    source = Task.get_task(task_id=args.source_task_id)
    if str(source.status).lower() not in {"created", "failed"}:
        raise SystemExit("source task must be an unqueued draft or a failed task")
    script = source.data.script
    diff = getattr(script, "diff", None)
    repository = getattr(script, "repository", None)
    if not isinstance(diff, str) or not diff or repository not in {None, ""}:
        raise SystemExit("source task is not a standalone task with an uploaded diff")
    observed_diff = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    if observed_diff != expected_diff:
        raise SystemExit("source standalone diff hash mismatch")

    active = Task.get_tasks(
        project_name=source.get_project_name(),
        task_name=args.name,
        task_filter={"status": ["created", "queued", "in_progress"]},
    )
    if active:
        raise SystemExit("an active task already has the target name")

    clone = Task.clone(
        source_task=source,
        name=args.name,
        comment="Retry under a digest-pinned execution contract.",
    )
    clone.set_script(
        repository="",
        branch="",
        commit="",
        diff=diff,
        working_dir=".",
        entry_point=args.entry_point,
    )
    packages = contract["install_packages"]
    clone.set_packages(packages)
    clone.set_base_docker(
        docker_cmd=contract["base_docker"],
        docker_setup_bash_script=[
            "python3 -m pip install --no-input --disable-pip-version-check "
            + " ".join(packages)
        ],
    )
    parameters = {
        **contract["argument_overrides"],
        "expected_diff_sha256": observed_diff,
        "expected_script_sha256": expected_script,
    }
    for name, value in sorted(parameters.items()):
        clone.set_parameter("Args/" + name, parameter_value(value))

    Task.enqueue(clone, queue_name=args.queue)
    queued = Task.get_task(task_id=clone.id)
    if str(queued.status).lower() != "queued":
        raise SystemExit("task did not enter the requested queue")
    payload = {
        "contract_sha256": contract_sha256,
        "enqueue_succeeded": True,
        "source_diff_sha256": observed_diff,
        "source_task_id": source.id,
        "task_id": clone.id,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
