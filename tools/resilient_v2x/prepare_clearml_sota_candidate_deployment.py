#!/usr/bin/env python3
"""Prepare or explicitly execute the sealed E1/E2/E3 ClearML deployment.

The default ``prepare`` command is local-only.  Remote-writing subcommands require
their exact execution token so an accidental invocation cannot create a Dataset,
template, controller, or queue entry.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import py_compile
import subprocess
import sys
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from tools.resilient_v2x import clearml_5090_training_controller as training
    from tools.resilient_v2x import clearml_sota_candidate_controller as candidate
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.resilient_v2x import clearml_5090_training_controller as training
    from tools.resilient_v2x import clearml_sota_candidate_controller as candidate


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE_DIR = (
    ROOT / "artifacts/resilient_v2x/sota_candidate_source_e1_e2_e3"
)
SOURCE_ARCHIVE = SOURCE_PACKAGE_DIR / candidate.CANDIDATE_SOURCE_PACKAGE["archive_name"]
SOURCE_INVENTORY = SOURCE_PACKAGE_DIR / "source-inventory.json"
DEPLOYMENT_DIR = SOURCE_PACKAGE_DIR / "deployment"
STANDALONE_CONTROLLER = DEPLOYMENT_DIR / "clearml_sota_candidate_controller.py"
DEPLOYMENT_PLAN = DEPLOYMENT_DIR / "deployment-plan.json"
BASE_INVENTORY_DEFAULT = Path(
    "/tmp/transvision-clearml-cache/storage_manager/datasets/"
    "ds_351feedbbe81481fa31f1e9ae11a3f4e/source/source-inventory.json"
)

SOURCE_PROJECT = "ResilientV2X/Source"
SOURCE_DATASET_NAME = "ResilientV2X source 8a22d6d600a5 E1-E3"
SOURCE_DATASET_VERSION = "8a22d6d600a5-e1-e3-v1"
TRAINING_PROJECT = "ResilientV2X/Training"
BASE_TEMPLATE_TASK_ID = "d377543f6a574449a5d4b28cb9275dbc"
GATE_CONTROLLER_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
TEACHER_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
TEACHER_MODEL_ID = "d962f6bae8474260b54e170a7a5f0418"
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
WORKER_QUEUES = ("GPU4-5090", "GPU4-V100")
MAX_PARALLEL = 2
SERVICES_QUEUE = "services"
SERVICE_DOCKER_IMAGE = "ubuntu:24.04"
SERVICE_DOCKER_ARGS = (
    "--network host "
    "--add-host apiserver:10.100.34.118 "
    "--add-host fileserver:10.100.34.118 "
    "--add-host webserver:10.100.34.118 "
    "-e CLEARML_API_HOST=http://10.100.34.118:8008 "
    "-e CLEARML_WEB_HOST=http://10.100.34.118:8080 "
    "-e CLEARML_FILES_HOST=http://10.100.34.118:8081"
)

UPLOAD_TOKEN = "UPLOAD_EXACT_CANDIDATE_SOURCE"
TEMPLATE_TOKEN = "CREATE_EXACT_CANDIDATE_TEMPLATE"
CONTROLLER_TOKEN = "CREATE_AND_QUEUE_EXACT_CANDIDATE_CONTROLLER"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _verify_source_package(base_inventory_path: Path) -> dict[str, object]:
    if SOURCE_ARCHIVE.is_symlink() or not SOURCE_ARCHIVE.is_file():
        raise ValueError("candidate source archive is missing or not a regular file")
    if SOURCE_INVENTORY.is_symlink() or not SOURCE_INVENTORY.is_file():
        raise ValueError("candidate source inventory is missing or not a regular file")
    observed = {
        "archive_name": SOURCE_ARCHIVE.name,
        "archive_bytes": SOURCE_ARCHIVE.stat().st_size,
        "archive_sha256": _sha256(SOURCE_ARCHIVE),
        "inventory_sha256": _sha256(SOURCE_INVENTORY),
        "inventory_bytes": SOURCE_INVENTORY.stat().st_size,
    }
    for key, value in observed.items():
        if value != candidate.CANDIDATE_SOURCE_PACKAGE[key]:
            raise ValueError(f"local candidate source package drifted: {key}")
    inventory = _json(SOURCE_INVENTORY)
    for key in ("tree_sha256", "file_count", "source_bytes"):
        if inventory.get(key) != candidate.CANDIDATE_SOURCE_PACKAGE[key]:
            raise ValueError(f"local candidate source inventory drifted: {key}")

    base = _json(base_inventory_path.resolve(strict=True))
    old_files = {
        item["path"]: item for item in base.get("files", []) if isinstance(item, Mapping)
    }
    new_files = {
        item["path"]: item
        for item in inventory.get("files", [])
        if isinstance(item, Mapping)
    }
    added = sorted(set(new_files) - set(old_files))
    removed = sorted(set(old_files) - set(new_files))
    changed = sorted(
        path
        for path in set(old_files) & set(new_files)
        if old_files[path] != new_files[path]
    )
    expected_added = sorted(candidate.CANDIDATE_CONFIG_SHA256)
    if added != expected_added or removed or changed:
        raise ValueError(
            "candidate source inventory is not the exact three-file additive delta"
        )
    for path, expected_sha in candidate.CANDIDATE_CONFIG_SHA256.items():
        if new_files[path].get("sha256") != expected_sha:
            raise ValueError(f"candidate source config receipt drifted: {path}")
    return {
        **candidate.CANDIDATE_SOURCE_PACKAGE,
        "archive_path": str(SOURCE_ARCHIVE),
        "inventory_path": str(SOURCE_INVENTORY),
        "base_tree_sha256": base.get("tree_sha256"),
        "added_files": added,
        "changed_files": changed,
        "removed_files": removed,
    }


def _encoded_source(path: Path) -> str:
    return base64.b85encode(zlib.compress(path.read_bytes(), level=9)).decode("ascii")


def _standalone_source() -> str:
    training_payload = _encoded_source(
        ROOT / "tools/resilient_v2x/clearml_5090_training_controller.py"
    )
    candidate_payload = _encoded_source(
        ROOT / "tools/resilient_v2x/clearml_sota_candidate_controller.py"
    )
    return f'''#!/usr/bin/env python3
"""Generated standalone ResilientV2X E1/E2/E3 candidate controller."""
import base64
import sys
import types
import zlib

def _decode(value):
    return zlib.decompress(base64.b85decode(value.encode("ascii"))).decode("utf-8")

tools_package = types.ModuleType("tools")
tools_package.__path__ = []
resilient_package = types.ModuleType("tools.resilient_v2x")
resilient_package.__path__ = []
sys.modules["tools"] = tools_package
sys.modules["tools.resilient_v2x"] = resilient_package

training_name = "tools.resilient_v2x.clearml_5090_training_controller"
training_module = types.ModuleType(training_name)
training_module.__file__ = "clearml_5090_training_controller.py"
training_module.__package__ = "tools.resilient_v2x"
sys.modules[training_name] = training_module
exec(compile(_decode("{training_payload}"), training_module.__file__, "exec"), training_module.__dict__)

candidate_globals = {{
    "__name__": "__main__",
    "__file__": "clearml_sota_candidate_controller.py",
    "__package__": "tools.resilient_v2x",
}}
exec(compile(_decode("{candidate_payload}"), candidate_globals["__file__"], "exec"), candidate_globals)
'''


def _write_standalone(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / STANDALONE_CONTROLLER.name
    source = _standalone_source()
    destination.write_text(source, encoding="utf-8")
    destination.chmod(0o755)
    py_compile.compile(str(destination), doraise=True)
    help_run = subprocess.run(
        [sys.executable, str(destination), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if "--candidate-source-transition-sha256" not in help_run.stdout:
        raise RuntimeError("standalone candidate controller help contract is missing")
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _transition_args(dataset_id: str) -> argparse.Namespace:
    package = candidate.CANDIDATE_SOURCE_PACKAGE
    values = {
        "candidate_source_dataset_id": dataset_id,
        "candidate_source_archive_name": package["archive_name"],
        "candidate_source_archive_bytes": package["archive_bytes"],
        "candidate_source_archive_sha256": package["archive_sha256"],
        "candidate_source_tree_sha256": package["tree_sha256"],
        "candidate_source_inventory_sha256": package["inventory_sha256"],
        "candidate_source_inventory_bytes": package["inventory_bytes"],
        "candidate_source_file_count": package["file_count"],
        "candidate_source_bytes": package["source_bytes"],
        "candidate_source_transition_sha256": "0" * 64,
    }
    namespace = argparse.Namespace(**values)
    payload = candidate._candidate_source_transition_payload(namespace)
    namespace.candidate_source_transition_sha256 = hashlib.sha256(
        candidate._canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return namespace


def _transition(dataset_id: str) -> dict[str, object]:
    return candidate.candidate_source_transition(_transition_args(dataset_id))


def _controller_parameters(
    *,
    dataset_id: str,
    template_task_id: str,
    template_script_sha256: str,
    gate_summary_seal_sha256: str = "",
) -> dict[str, object]:
    transition = _transition(dataset_id)
    package = candidate.CANDIDATE_SOURCE_PACKAGE
    return {
        "Args/gate_task_id": GATE_CONTROLLER_TASK_ID,
        "Args/paper_controller_task_id": "",
        "Args/paper_controller_summary_artifact": "paper_controller_summary",
        "Args/gate_summary_artifact": candidate.GATE_SUMMARY_ARTIFACT,
        "Args/gate_summary_seal_sha256": gate_summary_seal_sha256,
        "Args/template_task_id": template_task_id,
        "Args/template_script_sha256": template_script_sha256,
        "Args/teacher_task_id": TEACHER_TASK_ID,
        "Args/resolve_teacher_reference": False,
        "Args/teacher_model_id": TEACHER_MODEL_ID,
        "Args/teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "Args/allow_failed_teacher_task": False,
        "Args/worker_queue": WORKER_QUEUES[0],
        "Args/worker_queues": ",".join(WORKER_QUEUES),
        "Args/max_parallel": MAX_PARALLEL,
        "Args/canary_first": False,
        "Args/adopt_experiment": [],
        "Args/recover_failed_controller_task_id": "",
        "Args/recovery_source_template_task_id": "",
        "Args/recovery_source_transition": "",
        "Args/rerun_failed_experiment": [],
        "Args/recovery_adopt_target_experiment": [],
        "Args/build_task_id": "",
        "Args/native_bundle_bytes": 0,
        "Args/native_bundle_sha256": "",
        "Args/build_manifest_sha256": "",
        "Args/project": TRAINING_PROJECT,
        "Args/training_seed": 20250218,
        "Args/poll_seconds": 30.0,
        "Args/candidate_source_dataset_id": dataset_id,
        "Args/candidate_source_archive_name": package["archive_name"],
        "Args/candidate_source_archive_bytes": package["archive_bytes"],
        "Args/candidate_source_archive_sha256": package["archive_sha256"],
        "Args/candidate_source_tree_sha256": package["tree_sha256"],
        "Args/candidate_source_inventory_sha256": package["inventory_sha256"],
        "Args/candidate_source_inventory_bytes": package["inventory_bytes"],
        "Args/candidate_source_file_count": package["file_count"],
        "Args/candidate_source_bytes": package["source_bytes"],
        "Args/candidate_source_transition_sha256": transition["seal_sha256"],
    }


def _require_token(observed: str, expected: str) -> None:
    if observed != expected:
        raise PermissionError(f"remote mutation requires --execute-token {expected}")


def _parameter_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, list):
        return str(actual) == json.dumps(expected)
    if isinstance(expected, float):
        return str(actual) == str(expected)
    return training._parameter_matches(actual, expected)


def _verify_downloaded_source_dataset(dataset_id: str) -> Path:
    from clearml import Dataset

    dataset = Dataset.get(dataset_id=dataset_id)
    root = Path(dataset.get_local_copy()).resolve(strict=True)
    members = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    expected = sorted(
        [
            f"source/{SOURCE_ARCHIVE.name}",
            f"source/{SOURCE_INVENTORY.name}",
        ]
    )
    if members != expected:
        raise RuntimeError(f"candidate source Dataset inventory mismatch: {members}")
    source_root = root / "source"
    if _sha256(source_root / SOURCE_ARCHIVE.name) != (
        candidate.CANDIDATE_SOURCE_PACKAGE["archive_sha256"]
    ):
        raise RuntimeError("candidate source Dataset archive SHA mismatch")
    if _sha256(source_root / SOURCE_INVENTORY.name) != (
        candidate.CANDIDATE_SOURCE_PACKAGE["inventory_sha256"]
    ):
        raise RuntimeError("candidate source Dataset inventory SHA mismatch")
    return root


def _upload_source(args: argparse.Namespace) -> int:
    _require_token(args.execute_token, UPLOAD_TOKEN)
    _verify_source_package(args.base_inventory)
    from clearml import Dataset

    dataset = Dataset.create(
        dataset_project=SOURCE_PROJECT,
        dataset_name=SOURCE_DATASET_NAME,
        dataset_version=SOURCE_DATASET_VERSION,
        dataset_tags=["ResilientV2X", "source", "sota-candidates", "sealed"],
        output_uri=training.FILES_SERVER_URI,
        description=json.dumps(
            {
                "base_source_dataset_id": candidate.CURRENT_SOURCE["dataset_id"],
                "transition": "exact three-config additive E1/E2/E3 source",
                "package": candidate.CANDIDATE_SOURCE_PACKAGE,
            },
            sort_keys=True,
        ),
    )
    dataset.add_files(
        SOURCE_PACKAGE_DIR,
        wildcard=[SOURCE_ARCHIVE.name, SOURCE_INVENTORY.name],
        local_base_folder=str(SOURCE_PACKAGE_DIR),
        dataset_path="source",
        recursive=False,
        verbose=True,
        max_workers=2,
    )
    dataset.upload(show_progress=True, verbose=True, preview=False, max_workers=2)
    if not dataset.finalize(verbose=True, raise_on_error=True, auto_upload=False):
        raise RuntimeError("candidate source Dataset did not finalize")
    _verify_downloaded_source_dataset(dataset.id)
    print(json.dumps({"candidate_source_dataset_id": dataset.id}, sort_keys=True))
    return 0


def _create_template(args: argparse.Namespace) -> int:
    _require_token(args.execute_token, TEMPLATE_TOKEN)
    dataset_id = training._clearml_id(
        args.candidate_source_dataset_id, "candidate source Dataset"
    )
    _verify_downloaded_source_dataset(dataset_id)
    transition = _transition(dataset_id)
    from clearml import Task

    base = Task.get_task(task_id=BASE_TEMPLATE_TASK_ID)
    if training._normalized_task_status(base) != "completed":
        raise RuntimeError("base bootstrap template is not completed")
    template = Task.clone(
        source_task=base,
        name=f"ResilientV2X candidate bootstrap template [{str(transition['seal_sha256'])[:12]}]",
        parent=GATE_CONTROLLER_TASK_ID,
        project=str(getattr(base, "project", "") or ""),
    )
    if training._normalized_task_status(template) != "created":
        raise RuntimeError("candidate template clone is not in created state")
    parameters = dict(training._task_parameters(base))
    package = candidate.CANDIDATE_SOURCE_PACKAGE
    parameters.update(
        {
            "Args/source_dataset_id": dataset_id,
            "Args/source_archive_name": package["archive_name"],
            "Args/source_archive_bytes": package["archive_bytes"],
            "Args/source_archive_sha256": package["archive_sha256"],
        }
    )
    template.set_parameters(parameters)
    template.output_uri = training.FILES_SERVER_URI
    if not template.upload_artifact(
        candidate.CANDIDATE_SOURCE_TRANSITION_ARTIFACT,
        artifact_object=transition,
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to upload candidate source transition artifact")
    template.flush(wait_for_uploads=True)
    template.mark_completed(
        status_message="sealed candidate-enabled bootstrap template", force=True
    )
    candidate._install_candidate_extension()
    identity = training._template_identity(template, expected_task_id=template.id)
    print(
        json.dumps(
            {
                "candidate_template_task_id": template.id,
                "candidate_template_script_sha256": identity["script_sha256"],
                "candidate_source_transition_sha256": transition["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _create_controller(args: argparse.Namespace) -> int:
    _require_token(args.execute_token, CONTROLLER_TOKEN)
    dataset_id = training._clearml_id(
        args.candidate_source_dataset_id, "candidate source Dataset"
    )
    template_id = training._clearml_id(args.template_task_id, "candidate template")
    template_sha = candidate._sha256(
        args.template_script_sha256, "candidate template script"
    )
    _verify_downloaded_source_dataset(dataset_id)
    standalone = _write_standalone(DEPLOYMENT_DIR)
    from clearml import Task

    template = Task.get_task(task_id=template_id)
    if training._normalized_task_status(template) != "completed":
        raise RuntimeError("candidate template is not completed")
    candidate._install_candidate_extension()
    identity = training._template_identity(template, expected_task_id=template_id)
    if identity["script_sha256"] != template_sha:
        raise RuntimeError("candidate template script SHA mismatch before controller")
    gate = Task.get_task(task_id=GATE_CONTROLLER_TASK_ID)
    if training._normalized_task_status(gate) in training.FAILED_STATUSES:
        raise RuntimeError("candidate predecessor controller has failed")
    teacher = Task.get_task(task_id=TEACHER_TASK_ID)
    if training._normalized_task_status(teacher) != "completed":
        raise RuntimeError("clean teacher task is not completed")
    models = list(teacher.get_models().get("output", []))
    if len(models) != 1 or str(models[0].id) != TEACHER_MODEL_ID:
        raise RuntimeError("clean teacher OutputModel binding drifted")

    controller = Task.create(
        project_name=TRAINING_PROJECT,
        task_name="ResilientV2X E1-E2-E3 sealed candidate controller",
        task_type=Task.TaskTypes.controller,
        script=standalone["path"],
        packages=["clearml==2.1.11"],
        docker=SERVICE_DOCKER_IMAGE,
        docker_args=SERVICE_DOCKER_ARGS,
        add_task_init_call=False,
        force_single_script_file=True,
        binary="python",
        detect_repository=False,
    )
    standalone_source = Path(str(standalone["path"])).read_text(encoding="utf-8")
    controller.set_script(
        diff=standalone_source,
        working_dir=".",
        entry_point=STANDALONE_CONTROLLER.name,
    )
    controller.reload()
    observed_script = training._task_script(controller)
    if (
        observed_script.get("entry_point") != STANDALONE_CONTROLLER.name
        or hashlib.sha256(
            str(observed_script.get("diff") or "").encode("utf-8")
        ).hexdigest()
        != standalone["sha256"]
    ):
        raise RuntimeError("candidate standalone controller script upload drifted")
    controller.output_uri = training.FILES_SERVER_URI
    controller.set_parent(GATE_CONTROLLER_TASK_ID)
    parameters = _controller_parameters(
        dataset_id=dataset_id,
        template_task_id=template_id,
        template_script_sha256=template_sha,
        gate_summary_seal_sha256=args.gate_summary_seal_sha256,
    )
    controller.set_parameters(parameters)
    observed = training._task_parameters(controller)
    for key, expected in parameters.items():
        if not _parameter_matches(observed.get(key), expected):
            raise RuntimeError(f"candidate controller parameter drifted: {key}")
    response = Task.enqueue(task=controller, queue_name=SERVICES_QUEUE)
    if response is None or response is False:
        raise RuntimeError("ClearML did not confirm candidate controller enqueue")
    print(
        json.dumps(
            {
                "candidate_controller_task_id": controller.id,
                "queue": SERVICES_QUEUE,
                "standalone_controller": standalone,
                "worker_queues_after_gate": list(WORKER_QUEUES),
                "max_parallel_after_gate": MAX_PARALLEL,
            },
            sort_keys=True,
        )
    )
    return 0


def _prepare(args: argparse.Namespace) -> int:
    package = _verify_source_package(args.base_inventory)
    standalone = _write_standalone(args.output_dir)
    commands = {
        "1_upload_source": (
            f"python {Path(__file__).relative_to(ROOT)} upload-source "
            f"--base-inventory {args.base_inventory} --execute-token {UPLOAD_TOKEN}"
        ),
        "2_create_template": (
            f"python {Path(__file__).relative_to(ROOT)} create-template "
            f"--candidate-source-dataset-id <DATASET_ID> --execute-token {TEMPLATE_TOKEN}"
        ),
        "3_create_and_queue_controller": (
            f"python {Path(__file__).relative_to(ROOT)} create-controller "
            "--candidate-source-dataset-id <DATASET_ID> "
            "--template-task-id <TEMPLATE_ID> "
            "--template-script-sha256 <TEMPLATE_SCRIPT_SHA256> "
            f"--execute-token {CONTROLLER_TOKEN}"
        ),
    }
    plan = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_sota_candidate_deployment",
        "remote_state_changed": False,
        "source_package": package,
        "standalone_controller": standalone,
        "fixed_bindings": {
            "base_template_task_id": BASE_TEMPLATE_TASK_ID,
            "gate_controller_task_id": GATE_CONTROLLER_TASK_ID,
            "teacher_task_id": TEACHER_TASK_ID,
            "teacher_model_id": TEACHER_MODEL_ID,
            "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "worker_queues": list(WORKER_QUEUES),
            "max_parallel": MAX_PARALLEL,
            "services_queue": SERVICES_QUEUE,
        },
        "dependency_policy": (
            "controller waits on services for exact gate completion; no GPU child "
            "exists before sealed gate binding; then E1/E2 fill one queue each and "
            "E3 uses the first freed declared queue"
        ),
        "commands": commands,
        "risk_checks": [
            "source Dataset must contain exactly archive+inventory with pinned hashes",
            "candidate source differs from ad511 by exactly three added configs",
            "template is cloned from d377 and binds exact source transition artifact",
            "teacher task/model/checkpoint are explicit and completed",
            "controller runs on services and creates no GPU task before gate completion",
            "GPU queues are queue-level; a specific 4-7 worker is not addressable unless agents use distinct queue names",
            "no adoption, recovery, build override, AMP, or alternate seed is accepted",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / DEPLOYMENT_PLAN.name
    plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"deployment_plan": str(plan_path), **plan}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--base-inventory", type=Path, default=BASE_INVENTORY_DEFAULT)
    prepare.add_argument("--output-dir", type=Path, default=DEPLOYMENT_DIR)
    prepare.set_defaults(handler=_prepare)

    upload = subparsers.add_parser("upload-source")
    upload.add_argument("--base-inventory", type=Path, default=BASE_INVENTORY_DEFAULT)
    upload.add_argument("--execute-token", default="")
    upload.set_defaults(handler=_upload_source)

    template = subparsers.add_parser("create-template")
    template.add_argument("--candidate-source-dataset-id", required=True)
    template.add_argument("--execute-token", default="")
    template.set_defaults(handler=_create_template)

    controller = subparsers.add_parser("create-controller")
    controller.add_argument("--candidate-source-dataset-id", required=True)
    controller.add_argument("--template-task-id", required=True)
    controller.add_argument("--template-script-sha256", required=True)
    controller.add_argument("--gate-summary-seal-sha256", default="")
    controller.add_argument("--execute-token", default="")
    controller.set_defaults(handler=_create_controller)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
