#!/usr/bin/env python3
"""Plan or execute one train-only EventTrack-V2X development canary.

The default mode is a network-free dry-run.  ClearML is imported only for an
explicit local or remote execution mode, and only ``--execute-remote`` queues a
worker task.  Every execution is development-only and cannot become formal
evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.development_split import (  # noqa: E402
    DevelopmentSplitError,
    decode_development_split_manifest_v1,
)
from transvision.models.event_track_v2x.learning_contracts import (  # noqa: E402
    AssociationLearningContractError,
    AssociationTrainingConfigV1,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes  # noqa: E402


PROJECT = "Thesis/EventTrack-V2X/Development"
QUEUE = "GPU4-A100"
DATASET_TASK_ID = "691397743f934284b9419582adcce0f6"
DATASET_REQUIRED_TAG = "scientific-claim-forbidden"
OFFICIAL_SPLIT_SHA256 = (
    "4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3"
)
ENTRY_POINT = "tools/event_track_v2x/run_clearml_development_canary.py"
REPOSITORY = "git@github.com:its-research/transvision.git"

# This image/argument shape was read back from a completed ordinary PyTorch
# training task on GPU4-A100 on 2026-09-11.  It is evidence for the task shape,
# but is Python 3.12 and therefore is *not* an executable default for this
# repository's >=3.10,<3.11 contract.  Remote execution requires an explicitly
# supplied Python 3.10 image instead of silently using this observation.
OBSERVED_A100_IMAGE = (
    "gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr:24.12-py312-vit5-V100"
)
RUNTIME_ARGUMENTS = (
    "-e CLEARML_AGENT_FORCE_TASK_INIT=1 --shm-size 96g --env NCCL_P2P_DISABLE=1"
)
# Torch and NumPy must come from the CUDA runtime.  Bootstrapping the ClearML
# client is explicit so a canary never downloads/replaces a large torch build.
BOOTSTRAP_PACKAGES = ("clearml==2.1.5",)

TASK_TAGS = (
    "EventTrack-V2X",
    "development",
    "development-only",
    "train-only",
    DATASET_REQUIRED_TAG,
    "ranking-ineligible",
    "formal-evidence-false",
)
FORBIDDEN_SPLITS = ("val", "test", "test_A")
RECEIPT_TYPE = "event_track_v2x_clearml_development_submission_receipt_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class DevelopmentCanarySubmissionError(RuntimeError):
    """The development-only submission contract is not satisfied."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    appearance = parser.add_mutually_exclusive_group()
    appearance.add_argument("--appearance-cache", type=Path)
    appearance.add_argument(
        "--appearance-checkpoint",
        type=Path,
        help="absolute local-only frozen appearance checkpoint",
    )
    parser.add_argument("--appearance-checkpoint-sha256")
    parser.add_argument("--max-frame-pairs-per-sequence", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output")
    parser.add_argument(
        "--bootstrap-dependencies",
        action="store_true",
        help="bootstrap only clearml==2.1.5; torch/numpy must be preinstalled",
    )
    parser.add_argument(
        "--runtime-image",
        default=os.environ.get("EVENTTRACK_V2X_PY310_RUNTIME_IMAGE", ""),
        help="Python 3.10 CUDA image; required only for --execute-remote",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute-remote",
        action="store_true",
        help="create and enqueue the task; omission is always a dry-run",
    )
    execution.add_argument(
        "--execute-local",
        action="store_true",
        help="Task.init then train on the current host; never enqueues",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training_config(path: Path) -> AssociationTrainingConfigV1:
    def unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DevelopmentCanarySubmissionError(
                    f"duplicate JSON key in config: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                DevelopmentCanarySubmissionError(
                    f"non-finite JSON constant in config: {item}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentCanarySubmissionError("config must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise DevelopmentCanarySubmissionError("config must be a JSON object")
    try:
        if "kind" in value or "schema_version" in value:
            return AssociationTrainingConfigV1.from_mapping(value)
        allowed = {
            field
            for field in AssociationTrainingConfigV1.__dataclass_fields__
            if not field.startswith("_")
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise DevelopmentCanarySubmissionError(
                f"config contains unknown fields: {unknown}"
            )
        return replace(AssociationTrainingConfigV1(), **dict(value))
    except (AssociationLearningContractError, TypeError) as exc:
        raise DevelopmentCanarySubmissionError(
            "config violates the association training contract"
        ) from exc


def _regular_file_inside_root(
    path: Path, *, label: str, root: Path
) -> tuple[Path, str]:
    if path.is_symlink() or not path.is_file():
        raise DevelopmentCanarySubmissionError(f"{label} must be a regular file")
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise DevelopmentCanarySubmissionError(
            f"{label} must be inside the repository"
        ) from exc
    if resolved.stat().st_size <= 0:
        raise DevelopmentCanarySubmissionError(f"{label} must not be empty")
    return resolved, relative.as_posix()


def _safe_output(value: str | None, *, seed: int, fold_id: int) -> str:
    raw = value or (
        f"work_dirs/event_track_v2x/development/train-only/seed-{seed}/fold-{fold_id}"
    )
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or path == PurePosixPath(".")
    ):
        raise DevelopmentCanarySubmissionError(
            "output must be a non-empty repository-relative path"
        )
    return path.as_posix()


def _git(*arguments: str, root: Path) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DevelopmentCanarySubmissionError(
            "git source identity is unavailable"
        ) from exc


def source_identity(root: Path = ROOT) -> dict[str, object]:
    commit = _git("rev-parse", "HEAD", root=root).decode("ascii").strip()
    if _COMMIT.fullmatch(commit) is None:
        raise DevelopmentCanarySubmissionError("git HEAD is not a full commit SHA")
    branch = (
        _git("rev-parse", "--abbrev-ref", "HEAD", root=root).decode("utf-8").strip()
    )
    if not branch or branch == "HEAD":
        raise DevelopmentCanarySubmissionError("a named git branch is required")
    status = _git("status", "--porcelain=v1", "--untracked-files=all", "-z", root=root)
    diff = _git("diff", "--binary", "HEAD", "--", root=root)
    untracked_raw = _git("ls-files", "--others", "--exclude-standard", "-z", root=root)
    untracked: list[dict[str, object]] = []
    for raw in sorted(item for item in untracked_raw.split(b"\0") if item):
        try:
            relative = raw.decode("utf-8")
            path = (root / relative).resolve(strict=True)
            path.relative_to(root.resolve(strict=True))
        except (UnicodeDecodeError, OSError, RuntimeError, ValueError) as exc:
            raise DevelopmentCanarySubmissionError(
                "untracked source identity is unsafe"
            ) from exc
        if path.is_symlink() or not path.is_file():
            raise DevelopmentCanarySubmissionError(
                "untracked source identity contains a non-regular file"
            )
        untracked.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    worktree_payload = {
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "untracked": untracked,
    }
    return {
        "branch": branch,
        "commit": commit,
        "dirty": bool(status),
        "worktree_state_sha256": hashlib.sha256(
            canonical_json_bytes(worktree_payload)
        ).hexdigest(),
    }


def _split_train_sequences(path: Path) -> tuple[str, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        sequences = value["batch_split"]["train"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DevelopmentCanarySubmissionError(
            "official split does not contain batch_split.train"
        ) from exc
    if not isinstance(sequences, list) or any(
        type(item) is not str for item in sequences
    ):
        raise DevelopmentCanarySubmissionError(
            "batch_split.train must be a string array"
        )
    normalized = tuple(sorted(sequences))
    if len(normalized) != len(set(normalized)):
        raise DevelopmentCanarySubmissionError("batch_split.train contains duplicates")
    return normalized


def _parameters(
    *,
    dataset_root: str,
    split: str,
    development_manifest: str,
    config: str,
    split_sha256: str,
    development_manifest_sha256: str,
    development_manifest_content_sha256: str,
    config_file_sha256: str,
    config_sha256: str,
    source: Mapping[str, object],
    fold_id: int,
    seed: int,
    output: str,
    appearance_cache: str,
    appearance_cache_sha256: str,
    appearance_checkpoint: str,
    appearance_checkpoint_sha256: str,
    max_frame_pairs_per_sequence: int,
    device: str,
    dependency_bootstrap: bool,
) -> dict[str, object]:
    return {
        "Args/appearance_cache": appearance_cache,
        "Args/appearance_cache_sha256": appearance_cache_sha256,
        "Args/appearance_checkpoint": appearance_checkpoint,
        "Args/appearance_checkpoint_sha256": appearance_checkpoint_sha256,
        "Args/bootstrap_dependencies": dependency_bootstrap,
        "Args/config": config,
        "Args/dataset_root": dataset_root,
        "Args/dataset_task_id": DATASET_TASK_ID,
        "Args/development_manifest": development_manifest,
        "Args/development_only": True,
        "Args/device": device,
        "Args/expected_development_manifest_content_sha256": (
            development_manifest_content_sha256
        ),
        "Args/expected_config_file_sha256": config_file_sha256,
        "Args/expected_config_sha256": config_sha256,
        "Args/expected_development_manifest_sha256": (development_manifest_sha256),
        "Args/expected_split_sha256": split_sha256,
        "Args/fold_id": fold_id,
        "Args/forbidden_splits": ",".join(FORBIDDEN_SPLITS),
        "Args/formal_evidence": False,
        "Args/git_commit": source["commit"],
        "Args/git_dirty": source["dirty"],
        "Args/max_frame_pairs_per_sequence": max_frame_pairs_per_sequence,
        "Args/output": output,
        "Args/ranking_eligible": False,
        "Args/seed": seed,
        "Args/split": split,
        "Args/split_name": "train",
        "Args/worktree_state_sha256": source["worktree_state_sha256"],
    }


def build_submission_plan(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
    source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if args.seed < 0:
        raise DevelopmentCanarySubmissionError("seed must be non-negative")
    if (
        args.max_frame_pairs_per_sequence is not None
        and args.max_frame_pairs_per_sequence <= 0
    ):
        raise DevelopmentCanarySubmissionError(
            "max-frame-pairs-per-sequence must be positive"
        )
    dataset_path = PurePosixPath(args.dataset_root)
    if not dataset_path.is_absolute():
        raise DevelopmentCanarySubmissionError(
            "dataset-root must be an absolute worker path"
        )
    if not any(
        part == "train-only" or part.endswith("-train-only")
        for part in dataset_path.parts
    ):
        raise DevelopmentCanarySubmissionError(
            "dataset-root must name an isolated train-only directory"
        )
    split_path, split_relative = _regular_file_inside_root(
        args.split, label="split", root=root
    )
    manifest_path, manifest_relative = _regular_file_inside_root(
        args.development_manifest, label="development manifest", root=root
    )
    config_path, config_relative = _regular_file_inside_root(
        args.config, label="config", root=root
    )
    split_sha256 = _sha256_file(split_path)
    if split_sha256 != OFFICIAL_SPLIT_SHA256:
        raise DevelopmentCanarySubmissionError(
            "split is not the pinned official V2X-Seq-SPD split"
        )
    try:
        manifest = decode_development_split_manifest_v1(manifest_path.read_bytes())
    except DevelopmentSplitError as exc:
        raise DevelopmentCanarySubmissionError(
            "development manifest is invalid"
        ) from exc
    if manifest.split_name != "train" or manifest.split_sha256 != split_sha256:
        raise DevelopmentCanarySubmissionError(
            "development manifest is not bound to the supplied train split"
        )
    if tuple(manifest.sequence_ids) != _split_train_sequences(split_path):
        raise DevelopmentCanarySubmissionError(
            "development cohort differs from batch_split.train"
        )
    resolved_config = _load_training_config(config_path)
    source_value = dict(source or source_identity(root))
    if set(source_value) != {
        "branch",
        "commit",
        "dirty",
        "worktree_state_sha256",
    }:
        raise DevelopmentCanarySubmissionError("source identity fields are invalid")
    if (
        _COMMIT.fullmatch(str(source_value["commit"])) is None
        or type(source_value["dirty"]) is not bool
        or _SHA256.fullmatch(str(source_value["worktree_state_sha256"])) is None
    ):
        raise DevelopmentCanarySubmissionError("source identity values are invalid")
    appearance_relative = ""
    appearance_cache_sha256 = ""
    if args.appearance_cache is not None:
        appearance_path, appearance_relative = _regular_file_inside_root(
            args.appearance_cache, label="appearance cache", root=root
        )
        appearance_cache_sha256 = _sha256_file(appearance_path)
    checkpoint_value = ""
    checkpoint_sha256 = ""
    if (
        args.appearance_checkpoint_sha256 is not None
        and args.appearance_checkpoint is None
    ):
        raise DevelopmentCanarySubmissionError(
            "appearance-checkpoint-sha256 requires appearance-checkpoint"
        )
    if args.appearance_checkpoint is not None:
        if args.execute_remote:
            raise DevelopmentCanarySubmissionError(
                "appearance-checkpoint is local-only; remote execution requires a "
                "repository-bound appearance cache"
            )
        if not args.appearance_checkpoint.is_absolute():
            raise DevelopmentCanarySubmissionError(
                "appearance-checkpoint must be an absolute local path"
            )
        if (
            type(args.appearance_checkpoint_sha256) is not str
            or _SHA256.fullmatch(args.appearance_checkpoint_sha256) is None
        ):
            raise DevelopmentCanarySubmissionError(
                "appearance-checkpoint requires its lowercase SHA-256"
            )
        checkpoint_value = str(args.appearance_checkpoint)
        checkpoint_sha256 = args.appearance_checkpoint_sha256
        if args.execute_local:
            checkpoint = args.appearance_checkpoint
            if checkpoint.is_symlink() or not checkpoint.is_file():
                raise DevelopmentCanarySubmissionError(
                    "local appearance checkpoint must be a regular file"
                )
            if _sha256_file(checkpoint) != checkpoint_sha256:
                raise DevelopmentCanarySubmissionError(
                    "local appearance checkpoint SHA-256 mismatch"
                )
    if (args.execute_remote or args.execute_local) and not (
        appearance_relative or checkpoint_value
    ):
        raise DevelopmentCanarySubmissionError(
            "execution requires an explicit appearance cache or checkpoint"
        )
    output = _safe_output(args.output, seed=args.seed, fold_id=args.fold_id)
    task_name = (
        "eventtrack-v2x__development__train-only__"
        f"seed-{args.seed}__fold-{args.fold_id}__"
        f"{str(source_value['commit'])[:8]}"
    )
    parameters = _parameters(
        dataset_root=args.dataset_root,
        split=split_relative,
        development_manifest=manifest_relative,
        config=config_relative,
        split_sha256=split_sha256,
        development_manifest_sha256=_sha256_file(manifest_path),
        development_manifest_content_sha256=manifest.content_sha256,
        config_file_sha256=_sha256_file(config_path),
        config_sha256=resolved_config.content_sha256,
        source=source_value,
        fold_id=args.fold_id,
        seed=args.seed,
        output=output,
        appearance_cache=appearance_relative,
        appearance_cache_sha256=appearance_cache_sha256,
        appearance_checkpoint=checkpoint_value,
        appearance_checkpoint_sha256=checkpoint_sha256,
        max_frame_pairs_per_sequence=args.max_frame_pairs_per_sequence or 0,
        device=args.device,
        dependency_bootstrap=args.bootstrap_dependencies,
    )
    command = [
        sys.executable,
        "tools/event_track_v2x/train_association.py",
        "--dataset-root",
        args.dataset_root,
        "--split",
        split_relative,
        "--development-manifest",
        manifest_relative,
        "--fold-id",
        str(args.fold_id),
        "--seed",
        str(args.seed),
        "--output",
        output,
        "--config",
        config_relative,
        "--device",
        args.device,
    ]
    if appearance_relative:
        command.extend(("--appearance-cache", appearance_relative))
    elif checkpoint_value:
        command.extend(
            (
                "--appearance-checkpoint",
                checkpoint_value,
                "--appearance-checkpoint-sha256",
                checkpoint_sha256,
            )
        )
    if args.max_frame_pairs_per_sequence is not None:
        command.extend(
            (
                "--max-frame-pairs-per-sequence",
                str(args.max_frame_pairs_per_sequence),
            )
        )
    return {
        "schema_version": 1,
        "document_type": RECEIPT_TYPE,
        "mode": "dry-run",
        "submission_performed": False,
        "task_created": False,
        "task_enqueued": False,
        "project": PROJECT,
        "queue": QUEUE,
        "task_name": task_name,
        "dataset": {
            "task_id": DATASET_TASK_ID,
            "required_tag": DATASET_REQUIRED_TAG,
        },
        "source": source_value,
        "bindings": {
            "split_sha256": split_sha256,
            "development_manifest_sha256": _sha256_file(manifest_path),
            "development_manifest_content_sha256": manifest.content_sha256,
            "config_file_sha256": _sha256_file(config_path),
            "config_sha256": resolved_config.content_sha256,
        },
        "scope": {
            "split": "train",
            "forbidden_splits": list(FORBIDDEN_SPLITS),
            "development_only": True,
            "ranking_eligible": False,
            "formal_evidence": False,
        },
        "entry_point": ENTRY_POINT,
        "training_command": command,
        "parameters": parameters,
        "tags": [*TASK_TAGS, f"seed-{args.seed}", f"fold-{args.fold_id}"],
        "runtime": {
            "arguments": RUNTIME_ARGUMENTS,
            "image": args.runtime_image,
            "python_requires": ">=3.10,<3.11",
            "observed_a100_image": OBSERVED_A100_IMAGE,
            "observed_a100_image_compatible": False,
            "dependency_bootstrap": (
                list(BOOTSTRAP_PACKAGES) if args.bootstrap_dependencies else []
            ),
            "preinstalled_dependencies": ["numpy", "torch"],
        },
        "appearance": {
            "mode": (
                "cache"
                if appearance_relative
                else "checkpoint"
                if checkpoint_value
                else "unconfigured"
            ),
            "path": appearance_relative or checkpoint_value,
            "sha256": appearance_cache_sha256 or checkpoint_sha256,
        },
        "clearml_config": {
            "environment_variable": "CLEARML_CONFIG_FILE",
            "configured": bool(os.environ.get("CLEARML_CONFIG_FILE")),
        },
    }


def _dataset_tags(dataset: object) -> tuple[str, ...]:
    raw = getattr(dataset, "tags", None)
    if raw is None:
        getter = getattr(dataset, "get_tags", None)
        raw = getter() if callable(getter) else None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise DevelopmentCanarySubmissionError("ClearML dataset tags are unavailable")
    return tuple(str(item) for item in raw)


def execute_submission(
    plan: Mapping[str, object],
    *,
    task_api: Any,
    dataset_api: Any,
) -> dict[str, object]:
    source = plan["source"]
    if not isinstance(source, Mapping) or source.get("dirty") is not False:
        raise DevelopmentCanarySubmissionError(
            "remote execution requires a clean source worktree"
        )
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping) or not runtime.get("image"):
        raise DevelopmentCanarySubmissionError(
            "remote execution requires an explicit Python 3.10 runtime image"
        )
    if runtime.get("image") == OBSERVED_A100_IMAGE:
        raise DevelopmentCanarySubmissionError(
            "the observed GPU4-A100 image is Python 3.12 and incompatible"
        )
    config_value = os.environ.get("CLEARML_CONFIG_FILE")
    if not config_value:
        raise DevelopmentCanarySubmissionError(
            "CLEARML_CONFIG_FILE is required for remote execution"
        )
    config = Path(config_value)
    if (
        config.is_symlink()
        or not config.is_file()
        or not 0 < config.stat().st_size <= 1024 * 1024
    ):
        raise DevelopmentCanarySubmissionError(
            "CLEARML_CONFIG_FILE must name a small regular file"
        )
    dataset = dataset_api.get(dataset_id=DATASET_TASK_ID, only_completed=True)
    if DATASET_REQUIRED_TAG not in _dataset_tags(dataset):
        raise DevelopmentCanarySubmissionError(
            "dataset is missing scientific-claim-forbidden"
        )
    active = task_api.get_tasks(
        project_name=PROJECT,
        task_name=plan["task_name"],
        task_filter={"status": ["created", "queued", "in_progress"]},
    )
    if active:
        raise DevelopmentCanarySubmissionError(
            "an active task already has the development canary name"
        )
    task_types = getattr(task_api, "TaskTypes", None)
    training_type = getattr(task_types, "training", "training")
    task = task_api.create(
        project_name=PROJECT,
        task_name=plan["task_name"],
        task_type=training_type,
        detect_repository=False,
    )
    task.set_script(
        repository=REPOSITORY,
        branch=source["branch"],
        commit=source["commit"],
        diff="",
        working_dir=".",
        entry_point=ENTRY_POINT,
    )
    packages = runtime["dependency_bootstrap"]
    task.set_packages(list(packages))
    task.set_base_docker(
        docker_image=str(runtime["image"]),
        docker_arguments=RUNTIME_ARGUMENTS,
    )
    task.set_parameters(dict(plan["parameters"]))
    if task.set_tags(list(plan["tags"])) is False:
        raise DevelopmentCanarySubmissionError("ClearML task tags were not accepted")
    flushed = task.flush(wait_for_uploads=True)
    if flushed is False:
        raise DevelopmentCanarySubmissionError("ClearML task flush failed")
    task_api.enqueue(task, queue_name=QUEUE)
    queued = task_api.get_task(task_id=task.id)
    status = str(getattr(queued, "status", "")).lower()
    if "queued" not in status:
        raise DevelopmentCanarySubmissionError("task did not enter GPU4-A100")
    receipt = dict(plan)
    receipt.update(
        {
            "mode": "execute-remote",
            "submission_performed": True,
            "task_created": True,
            "task_enqueued": True,
            "task_id": str(task.id),
            "status": "queued",
        }
    )
    receipt["clearml_config"] = {
        "environment_variable": "CLEARML_CONFIG_FILE",
        "configured": True,
    }
    return receipt


def execute_local(
    plan: Mapping[str, object],
    *,
    task_api: Any,
    dataset_api: Any,
) -> dict[str, object]:
    """Create a tracked local task and run it without queueing a worker."""
    from tools.event_track_v2x.run_clearml_development_canary import run_remote

    dataset = dataset_api.get(dataset_id=DATASET_TASK_ID, only_completed=True)
    if DATASET_REQUIRED_TAG not in _dataset_tags(dataset):
        raise DevelopmentCanarySubmissionError(
            "dataset is missing scientific-claim-forbidden"
        )
    task_types = getattr(task_api, "TaskTypes", None)
    training_type = getattr(task_types, "training", "training")
    task = task_api.init(
        project_name=PROJECT,
        task_name=plan["task_name"],
        task_type=training_type,
        reuse_last_task_id=False,
        output_uri=True,
    )
    task.set_parameters(dict(plan["parameters"]))
    if task.set_tags(list(plan["tags"])) is False:
        raise DevelopmentCanarySubmissionError("ClearML task tags were not accepted")
    parameters = {
        key.removeprefix("Args/"): value for key, value in plan["parameters"].items()
    }
    try:
        run_receipt = run_remote(
            parameters,
            task=task,
            dataset_api=dataset_api,
            materialize_dataset=False,
        )
    except BaseException as exc:
        mark_failed = getattr(task, "mark_failed", None)
        if callable(mark_failed):
            mark_failed(
                ignore_errors=True,
                status_reason=type(exc).__name__,
                status_message=str(exc)[:2048],
                force=True,
            )
        raise
    close = getattr(task, "close", None)
    if not callable(close):
        raise DevelopmentCanarySubmissionError(
            "local ClearML task cannot be closed explicitly"
        )
    close()
    receipt = dict(plan)
    receipt.update(
        {
            "mode": "execute-local",
            "submission_performed": True,
            "task_created": True,
            "task_enqueued": False,
            "task_id": str(task.id),
            "status": "local-completed",
            "run_receipt": run_receipt,
        }
    )
    return receipt


def main(
    argv: Sequence[str] | None = None,
    *,
    task_api: Any | None = None,
    dataset_api: Any | None = None,
    root: Path = ROOT,
    source: Mapping[str, object] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    plan = build_submission_plan(args, root=root, source=source)
    if not args.execute_remote and not args.execute_local:
        print(canonical_json_bytes(plan).decode("utf-8"))
        return 0
    if args.execute_local and task_api is None and sys.version_info[:2] != (3, 10):
        raise DevelopmentCanarySubmissionError(
            "local execution requires Python 3.10 to match repository metadata"
        )
    if task_api is None and not os.environ.get("CLEARML_CONFIG_FILE"):
        raise DevelopmentCanarySubmissionError(
            "CLEARML_CONFIG_FILE is required for execution"
        )
    if args.bootstrap_dependencies and task_api is None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-input",
                "--disable-pip-version-check",
                *BOOTSTRAP_PACKAGES,
            ],
            check=True,
        )
    if task_api is None or dataset_api is None:
        # Import happens only behind the explicit mutation flag and only after
        # the CLEARML_CONFIG_FILE gate is checked by execute_submission.
        config_value = os.environ.get("CLEARML_CONFIG_FILE")
        assert config_value is not None
        os.environ["CLEARML_CONFIG_FILE"] = config_value
        from clearml import Dataset, Task

        task_api = Task
        dataset_api = Dataset
    if args.execute_local:
        receipt = execute_local(plan, task_api=task_api, dataset_api=dataset_api)
    else:
        receipt = execute_submission(plan, task_api=task_api, dataset_api=dataset_api)
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
