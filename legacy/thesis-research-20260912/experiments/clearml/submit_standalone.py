#!/usr/bin/env python3
"""Submit a standalone ClearML script under a verified execution contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from clearml import Task, TaskTypes

from requeue_standalone import (
    SAFE_NAME,
    TASK_NAME,
    load_contract,
    parameter_value,
    require_sha256,
)


PROJECT = "Thesis/RTP-V2X"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--entry-point", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    args = parser.parse_args()

    if not TASK_NAME.fullmatch(args.name):
        raise SystemExit("task name does not match the RTP-V2X naming contract")
    if not SAFE_NAME.fullmatch(args.queue):
        raise SystemExit("queue has invalid format")
    if args.entry_point != args.script.name or not SAFE_NAME.fullmatch(args.entry_point):
        raise SystemExit("entry point must equal the standalone script file name")
    if args.script.is_symlink() or not args.script.is_file() or args.script.stat().st_size > 1024 * 1024:
        raise SystemExit("standalone script must be a small regular file")
    expected_script = require_sha256(args.expected_script_sha256, "expected script hash")
    script_bytes = args.script.read_bytes()
    observed_script = hashlib.sha256(script_bytes).hexdigest()
    if observed_script != expected_script:
        raise SystemExit("standalone script hash mismatch")
    try:
        script_text = script_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit("standalone script is not UTF-8") from exc
    if not script_text.startswith("#!/usr/bin/env python3\n"):
        raise SystemExit("standalone script header is not allowlisted")

    contract, contract_sha256 = load_contract(
        args.execution_contract, args.expected_contract_sha256
    )
    active = Task.get_tasks(
        project_name=PROJECT,
        task_name=args.name,
        task_filter={"status": ["created", "queued", "in_progress"]},
    )
    if active:
        raise SystemExit("an active task already has the target name")

    task = Task.create(
        project_name=PROJECT,
        task_name=args.name,
        task_type=TaskTypes.testing,
    )
    task.add_tags(["diagnostic-only", "verified-execution-contract"])
    task.set_script(
        repository="",
        branch="",
        commit="",
        diff=script_text,
        working_dir=".",
        entry_point=args.entry_point,
    )
    packages = contract["install_packages"]
    task.set_packages(packages)
    task.set_base_docker(
        docker_cmd=contract["base_docker"],
        docker_setup_bash_script=[
            "python3 -m pip install --no-input --disable-pip-version-check "
            + " ".join(packages)
        ],
    )
    parameters = {
        **contract["argument_overrides"],
        "expected_diff_sha256": observed_script,
        "expected_script_sha256": observed_script,
    }
    for name, value in sorted(parameters.items()):
        task.set_parameter("Args/" + name, parameter_value(value))

    Task.enqueue(task, queue_name=args.queue)
    queued = Task.get_task(task_id=task.id)
    if str(queued.status).lower() != "queued":
        raise SystemExit("task did not enter the requested queue")
    print(
        json.dumps(
            {
                "contract_sha256": contract_sha256,
                "enqueue_succeeded": True,
                "source_diff_sha256": observed_script,
                "task_id": task.id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
