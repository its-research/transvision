#!/usr/bin/env python3
"""Run and audit one ClearML EventTrack-V2X development association canary."""

from __future__ import annotations

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
    decode_association_training_manifest_v1,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes  # noqa: E402


TRAIN_ENTRY_POINT = ROOT / "tools/event_track_v2x/train_association.py"
PREPARE_ENTRY_POINT = ROOT / "tools/event_track_v2x/prepare_spd_train_only.py"
DATASET_TASK_ID = "691397743f934284b9419582adcce0f6"
DATASET_REQUIRED_TAG = "scientific-claim-forbidden"
OFFICIAL_SPLIT_SHA256 = (
    "4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3"
)
FORBIDDEN_SPLITS = ("val", "test", "test_A")
BOUNDARY = {
    "development_only": True,
    "ranking_eligible": False,
    "formal_evidence": False,
}
RECEIPT_TYPE = "event_track_v2x_clearml_development_run_receipt_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class DevelopmentCanaryRuntimeError(RuntimeError):
    """The remote development-only execution contract was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    def unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DevelopmentCanaryRuntimeError(
                    f"duplicate JSON key in {label}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                DevelopmentCanaryRuntimeError(
                    f"non-finite JSON constant in {label}: {item}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentCanaryRuntimeError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise DevelopmentCanaryRuntimeError(f"{label} must be an object")
    return value


def _resolved_training_config(path: Path) -> AssociationTrainingConfigV1:
    value = _json_object(path, "association config")
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
            raise DevelopmentCanaryRuntimeError(
                f"association config contains unknown fields: {unknown}"
            )
        return replace(AssociationTrainingConfigV1(), **dict(value))
    except (AssociationLearningContractError, TypeError) as exc:
        raise DevelopmentCanaryRuntimeError(
            "association config violates its training contract"
        ) from exc


def _expected_sha256(parameters: Mapping[str, object], key: str) -> str:
    value = parameters.get(key)
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DevelopmentCanaryRuntimeError(f"{key} must be a lowercase SHA-256")
    return value


def _bool(value: object, label: str) -> bool:
    if type(value) is bool:
        return value
    if type(value) is str and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise DevelopmentCanaryRuntimeError(f"{label} must be boolean")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is int:
        parsed = value
    elif type(value) is str and value.isdecimal():
        parsed = int(value)
    else:
        raise DevelopmentCanaryRuntimeError(f"{label} must be an integer")
    if parsed < minimum:
        raise DevelopmentCanaryRuntimeError(f"{label} is below its minimum")
    return parsed


def _repository_file(value: object, label: str) -> Path:
    if type(value) is not str:
        raise DevelopmentCanaryRuntimeError(f"{label} path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise DevelopmentCanaryRuntimeError(f"{label} path is unsafe")
    path = ROOT.joinpath(*relative.parts)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(ROOT.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise DevelopmentCanaryRuntimeError(f"{label} is unavailable") from exc
    if path.is_symlink() or not resolved.is_file():
        raise DevelopmentCanaryRuntimeError(f"{label} must be a regular file")
    return resolved


def _output_path(value: object) -> Path:
    if type(value) is not str:
        raise DevelopmentCanaryRuntimeError("output path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise DevelopmentCanaryRuntimeError("output path is unsafe")
    path = ROOT.joinpath(*relative.parts)
    if path.exists():
        raise DevelopmentCanaryRuntimeError("output path already exists")
    return path


def _dataset_tags(dataset: object) -> tuple[str, ...]:
    raw = getattr(dataset, "tags", None)
    if raw is None:
        getter = getattr(dataset, "get_tags", None)
        raw = getter() if callable(getter) else None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise DevelopmentCanaryRuntimeError("ClearML dataset tags are unavailable")
    return tuple(str(item) for item in raw)


def _verify_train_only_manifest(
    dataset_root: Path,
    *,
    expected_split_sha256: str,
) -> tuple[Path, str]:
    manifest = dataset_root.parent / "train-only-manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise DevelopmentCanaryRuntimeError(
            "train-only dataset is missing its sibling preparation manifest"
        )
    value = _json_object(manifest, "train-only preparation manifest")
    observed_content = value.get("content_sha256")
    payload = dict(value)
    payload.pop("content_sha256", None)
    expected_content = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    source_split = value.get("source_split")
    policy = value.get("policy")
    tree_digest = hashlib.sha256()
    for path in sorted(dataset_root.rglob("*")):
        if path.is_symlink():
            raise DevelopmentCanaryRuntimeError(
                "train-only dataset tree contains a symlink"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise DevelopmentCanaryRuntimeError(
                "train-only dataset tree contains a non-regular entry"
            )
        record = {
            "path": path.relative_to(dataset_root.parent).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        tree_digest.update(canonical_json_bytes(record))
        tree_digest.update(b"\n")
    observed_tree_sha256 = tree_digest.hexdigest()
    if (
        type(observed_content) is not str
        or observed_content != expected_content
        or value.get("document_type") != "event_track_v2x_spd_train_only_manifest"
        or value.get("dataset") != "V2X-Seq-SPD"
        or value.get("subset") != "official-train-camera-only"
        or value.get("status") != "materialized"
        or value.get("output_tree_sha256") != observed_tree_sha256
        or value.get("output_tree_hash_algorithm")
        != "sha256-canonical-json-lines-path-size-sha256-v1"
        or not isinstance(source_split, Mapping)
        or source_split.get("sha256") != expected_split_sha256
        or not isinstance(policy, Mapping)
        or policy.get("included_partition") != "train"
        or policy.get("excluded_partitions") != list(FORBIDDEN_SPLITS)
        or policy.get("point_cloud_payloads_included") is not False
        or policy.get("data_info_projection") != "train-records-only"
    ):
        raise DevelopmentCanaryRuntimeError(
            "train-only preparation manifest violates the sealed partition contract"
        )
    return manifest, expected_content


def _task_parameters(task: object) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise DevelopmentCanaryRuntimeError("ClearML task parameters are unavailable")
    raw = getter()
    if not isinstance(raw, Mapping):
        raise DevelopmentCanaryRuntimeError("ClearML task parameters are invalid")
    result: dict[str, object] = {}
    for key, value in raw.items():
        if type(key) is str and key.startswith("Args/"):
            result[key[5:]] = value
    return result


def _require_boundary(parameters: Mapping[str, object]) -> None:
    if parameters.get("split_name") != "train":
        raise DevelopmentCanaryRuntimeError("only the train split is permitted")
    if parameters.get("forbidden_splits") != ",".join(FORBIDDEN_SPLITS):
        raise DevelopmentCanaryRuntimeError("forbidden split contract drifted")
    for key, expected in BOUNDARY.items():
        if _bool(parameters.get(key), key) is not expected:
            raise DevelopmentCanaryRuntimeError(f"{key} boundary drifted")
    commit = parameters.get("git_commit")
    if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
        raise DevelopmentCanaryRuntimeError("git commit binding is invalid")
    _bool(parameters.get("git_dirty"), "git_dirty")
    _expected_sha256(parameters, "worktree_state_sha256")
    if _expected_sha256(parameters, "expected_split_sha256") != OFFICIAL_SPLIT_SHA256:
        raise DevelopmentCanaryRuntimeError("official split binding drifted")


def build_training_command(
    parameters: Mapping[str, object],
    *,
    allow_external_appearance_checkpoint: bool = False,
) -> tuple[list[str], Path]:
    _require_boundary(parameters)
    if parameters.get("dataset_task_id") != DATASET_TASK_ID:
        raise DevelopmentCanaryRuntimeError("dataset task binding drifted")
    dataset_root = Path(str(parameters.get("dataset_root", "")))
    if not dataset_root.is_absolute() or not any(
        part == "train-only" or part.endswith("-train-only")
        for part in dataset_root.parts
    ):
        raise DevelopmentCanaryRuntimeError(
            "dataset root must be an isolated absolute train-only path"
        )
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise DevelopmentCanaryRuntimeError("train-only dataset root is unavailable")
    split = _repository_file(parameters.get("split"), "split")
    development = _repository_file(
        parameters.get("development_manifest"), "development manifest"
    )
    config = _repository_file(parameters.get("config"), "config")
    _verify_train_only_manifest(
        dataset_root,
        expected_split_sha256=_expected_sha256(parameters, "expected_split_sha256"),
    )
    for path, key in (
        (split, "expected_split_sha256"),
        (development, "expected_development_manifest_sha256"),
        (config, "expected_config_file_sha256"),
    ):
        if _sha256_file(path) != _expected_sha256(parameters, key):
            raise DevelopmentCanaryRuntimeError(f"{key} mismatch")
    try:
        development_value = decode_development_split_manifest_v1(
            development.read_bytes()
        )
    except (OSError, DevelopmentSplitError) as exc:
        raise DevelopmentCanaryRuntimeError("development manifest is invalid") from exc
    if (
        development_value.split_name != "train"
        or development_value.split_sha256
        != _expected_sha256(parameters, "expected_split_sha256")
        or development_value.content_sha256
        != _expected_sha256(parameters, "expected_development_manifest_content_sha256")
    ):
        raise DevelopmentCanaryRuntimeError("development cohort binding mismatch")
    resolved_config = _resolved_training_config(config)
    if resolved_config.content_sha256 != _expected_sha256(
        parameters, "expected_config_sha256"
    ):
        raise DevelopmentCanaryRuntimeError("resolved training config binding mismatch")
    fold_id = _integer(parameters.get("fold_id"), "fold_id")
    if fold_id >= 5:
        raise DevelopmentCanaryRuntimeError("fold_id must be below five")
    seed = _integer(parameters.get("seed"), "seed")
    output = _output_path(parameters.get("output"))
    device = parameters.get("device")
    if device not in {"cpu", "cuda"}:
        raise DevelopmentCanaryRuntimeError("device must be cpu or cuda")
    command = [
        sys.executable,
        str(TRAIN_ENTRY_POINT),
        "--dataset-root",
        str(dataset_root),
        "--split",
        str(split),
        "--development-manifest",
        str(development),
        "--fold-id",
        str(fold_id),
        "--seed",
        str(seed),
        "--output",
        str(output),
        "--config",
        str(config),
        "--device",
        str(device),
    ]
    appearance = parameters.get("appearance_cache")
    checkpoint = parameters.get("appearance_checkpoint")
    if appearance and checkpoint:
        raise DevelopmentCanaryRuntimeError(
            "appearance cache and checkpoint are mutually exclusive"
        )
    if appearance:
        appearance_path = _repository_file(appearance, "appearance cache")
        if _sha256_file(appearance_path) != _expected_sha256(
            parameters, "appearance_cache_sha256"
        ):
            raise DevelopmentCanaryRuntimeError("appearance cache SHA-256 mismatch")
        command.extend(("--appearance-cache", str(appearance_path)))
    elif checkpoint:
        if not allow_external_appearance_checkpoint or type(checkpoint) is not str:
            raise DevelopmentCanaryRuntimeError(
                "external appearance checkpoint is permitted only in local mode"
            )
        checkpoint_path = Path(checkpoint)
        if (
            not checkpoint_path.is_absolute()
            or checkpoint_path.is_symlink()
            or not checkpoint_path.is_file()
        ):
            raise DevelopmentCanaryRuntimeError(
                "local appearance checkpoint must be a regular absolute file"
            )
        checkpoint_sha256 = _expected_sha256(parameters, "appearance_checkpoint_sha256")
        if _sha256_file(checkpoint_path) != checkpoint_sha256:
            raise DevelopmentCanaryRuntimeError(
                "local appearance checkpoint SHA-256 mismatch"
            )
        command.extend(
            (
                "--appearance-checkpoint",
                str(checkpoint_path),
                "--appearance-checkpoint-sha256",
                checkpoint_sha256,
            )
        )
    else:
        raise DevelopmentCanaryRuntimeError(
            "an explicit appearance cache or checkpoint is required"
        )
    maximum = _integer(
        parameters.get("max_frame_pairs_per_sequence", 0),
        "max_frame_pairs_per_sequence",
    )
    if maximum:
        command.extend(("--max-frame-pairs-per-sequence", str(maximum)))
    return command, output


def _verify_artifacts(
    output: Path,
    parameters: Mapping[str, object],
) -> tuple[Path, Path, dict[str, object]]:
    checkpoint = output / "association-checkpoint.pt"
    manifest = output / "association-training-manifest.json"
    for path in (checkpoint, manifest):
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise DevelopmentCanaryRuntimeError(
                f"training artifact is missing or unsafe: {path.name}"
            )
    try:
        value = decode_association_training_manifest_v1(manifest.read_bytes())
    except (OSError, AssociationLearningContractError) as exc:
        raise DevelopmentCanaryRuntimeError("training manifest is invalid") from exc
    if (
        value.gt_supervised_development_only is not True
        or value.ranking_eligible is not False
        or value.formal_evidence is not False
        or value.paper_registry_writable is not False
    ):
        raise DevelopmentCanaryRuntimeError(
            "training artifacts did not preserve the development-only boundary"
        )
    appearance_cache_sha256 = parameters.get("appearance_cache_sha256")
    appearance_checkpoint_sha256 = parameters.get("appearance_checkpoint_sha256")
    if appearance_cache_sha256:
        expected_appearance_source = (
            f"frozen_npz_{_expected_sha256(parameters, 'appearance_cache_sha256')}_v1"
        )
    elif appearance_checkpoint_sha256:
        expected_appearance_source = (
            "frozen_resnet50_"
            f"{_expected_sha256(parameters, 'appearance_checkpoint_sha256')}_128_v1"
        )
    else:
        raise DevelopmentCanaryRuntimeError("appearance source binding is missing")
    expected = {
        "development_manifest_sha256": _expected_sha256(
            parameters, "expected_development_manifest_sha256"
        ),
        "official_split_sha256": _expected_sha256(parameters, "expected_split_sha256"),
        "config_sha256": _expected_sha256(parameters, "expected_config_sha256"),
        "fold_id": _integer(parameters.get("fold_id"), "fold_id"),
        "training_seed": _integer(parameters.get("seed"), "seed"),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "appearance_source": expected_appearance_source,
    }
    observed = {
        "development_manifest_sha256": value.development_manifest_sha256,
        "official_split_sha256": value.official_split_sha256,
        "config_sha256": value.config_sha256,
        "fold_id": value.fold_id,
        "training_seed": value.training_seed,
        "checkpoint_sha256": value.checkpoint_sha256,
        "appearance_source": value.appearance_source,
    }
    if observed != expected:
        drifted = sorted(key for key in expected if observed[key] != expected[key])
        raise DevelopmentCanaryRuntimeError(
            f"training artifact provenance mismatch: {drifted}"
        )
    return (
        checkpoint,
        manifest,
        {
            **observed,
            "training_cohort_sha256": value.cohort_sha256,
            "content_sha256": value.content_sha256,
            "gt_supervised_development_only": True,
            "paper_registry_writable": False,
            "ranking_eligible": False,
            "formal_evidence": False,
        },
    )


def _materialize_train_only_dataset(
    dataset: object,
    *,
    split: Path,
    task_id: str,
    environment: Mapping[str, str],
    runner: Any,
) -> tuple[Path, list[str]]:
    getter = getattr(dataset, "get_local_copy", None)
    if not callable(getter):
        raise DevelopmentCanaryRuntimeError(
            "ClearML dataset local-copy API is unavailable"
        )
    local_copy = getter()
    if not local_copy:
        raise DevelopmentCanaryRuntimeError("ClearML dataset local copy is unavailable")
    archive_root = Path(str(local_copy))
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise DevelopmentCanaryRuntimeError(
            "ClearML dataset local copy must be a regular directory"
        )
    if len(task_id) != 32 or any(
        character not in "0123456789abcdef" for character in task_id
    ):
        raise DevelopmentCanaryRuntimeError("ClearML task ID is invalid")
    preparation_parent = ROOT / "work_dirs/event_track_v2x/datasets/train-only"
    preparation_parent.mkdir(parents=True, exist_ok=True)
    preparation_output = preparation_parent / task_id
    if preparation_output.exists():
        raise DevelopmentCanaryRuntimeError(
            "train-only preparation output already exists"
        )
    command = [
        sys.executable,
        str(PREPARE_ENTRY_POINT),
        "--archive-dir",
        str(archive_root.resolve(strict=True)),
        "--split",
        str(split),
        "--output",
        str(preparation_output),
    ]
    runner(command, cwd=ROOT, env=dict(environment), check=True)
    dataset_root = preparation_output / "V2X-Seq-SPD"
    manifest = preparation_output / "train-only-manifest.json"
    if (
        dataset_root.is_symlink()
        or not dataset_root.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
    ):
        raise DevelopmentCanaryRuntimeError(
            "train-only dataset preparation did not produce its sealed layout"
        )
    return dataset_root, command


def run_remote(
    parameters: Mapping[str, object],
    *,
    task: Any,
    dataset_api: Any,
    runner: Any = subprocess.run,
    materialize_dataset: bool = True,
) -> dict[str, object]:
    dataset = dataset_api.get(dataset_id=DATASET_TASK_ID, only_completed=True)
    if DATASET_REQUIRED_TAG not in _dataset_tags(dataset):
        raise DevelopmentCanaryRuntimeError(
            "dataset is missing scientific-claim-forbidden"
        )
    environment = dict(os.environ)
    environment["EVENTTRACK_V2X_ALLOWED_SPLIT"] = "train"
    environment["EVENTTRACK_V2X_FORBIDDEN_SPLITS"] = ",".join(FORBIDDEN_SPLITS)
    resolved_parameters = dict(parameters)
    preparation_command: list[str] | None = None
    if materialize_dataset:
        split = _repository_file(parameters.get("split"), "split")
        dataset_root, preparation_command = _materialize_train_only_dataset(
            dataset,
            split=split,
            task_id=str(task.id),
            environment=environment,
            runner=runner,
        )
        resolved_parameters["dataset_root"] = str(dataset_root)
    command, output = build_training_command(
        resolved_parameters,
        allow_external_appearance_checkpoint=not materialize_dataset,
    )
    preparation_manifest, preparation_content_sha256 = _verify_train_only_manifest(
        Path(str(resolved_parameters["dataset_root"])),
        expected_split_sha256=_expected_sha256(
            resolved_parameters, "expected_split_sha256"
        ),
    )
    runner(command, cwd=ROOT, env=environment, check=True)
    checkpoint, manifest, artifact_provenance = _verify_artifacts(
        output, resolved_parameters
    )
    metadata = dict(BOUNDARY)
    metadata.update(
        {
            "dataset_task_id": DATASET_TASK_ID,
            "dataset_required_tag": DATASET_REQUIRED_TAG,
            "appearance_source": artifact_provenance["appearance_source"],
            "development_manifest_sha256": artifact_provenance[
                "development_manifest_sha256"
            ],
            "git_commit": str(parameters["git_commit"]),
            "git_dirty": _bool(parameters["git_dirty"], "git_dirty"),
            "fold_id": artifact_provenance["fold_id"],
            "gt_supervised_development_only": True,
            "paper_registry_writable": False,
            "split_name": "train",
            "training_seed": artifact_provenance["training_seed"],
            "training_cohort_sha256": artifact_provenance["training_cohort_sha256"],
        }
    )
    for name, path in (
        ("association_checkpoint_development_only", checkpoint),
        ("association_training_manifest_development_only", manifest),
    ):
        uploaded = task.upload_artifact(
            name=name,
            artifact_object=str(path),
            metadata=metadata,
            wait_on_upload=True,
        )
        if uploaded is False:
            raise DevelopmentCanaryRuntimeError(f"artifact upload failed: {name}")
    flushed = task.flush(wait_for_uploads=True)
    if flushed is False:
        raise DevelopmentCanaryRuntimeError("artifact upload flush failed")
    return {
        "schema_version": 1,
        "document_type": RECEIPT_TYPE,
        "mode": "remote-completed",
        "task_id": str(task.id),
        "dataset_task_id": DATASET_TASK_ID,
        "preparation_command": preparation_command,
        "preparation_manifest": {
            "path": str(preparation_manifest),
            "sha256": _sha256_file(preparation_manifest),
            "content_sha256": preparation_content_sha256,
        },
        "training_command": command,
        "provenance": {
            **artifact_provenance,
            "dataset_task_id": DATASET_TASK_ID,
            "git_commit": str(parameters["git_commit"]),
            "git_dirty": _bool(parameters["git_dirty"], "git_dirty"),
            "worktree_state_sha256": _expected_sha256(
                parameters, "worktree_state_sha256"
            ),
        },
        "artifacts": {
            checkpoint.name: {
                "sha256": _sha256_file(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            },
            manifest.name: {
                "sha256": _sha256_file(manifest),
                "size_bytes": manifest.stat().st_size,
            },
        },
        **BOUNDARY,
    }


def main() -> int:
    if sys.version_info[:2] != (3, 10):
        raise DevelopmentCanaryRuntimeError(
            "ClearML worker runtime must use Python 3.10"
        )
    from clearml import Dataset, Task

    task = Task.current_task()
    if task is None:
        raise DevelopmentCanaryRuntimeError(
            "remote runner requires an existing ClearML task context"
        )
    parameters = _task_parameters(task)
    receipt = run_remote(parameters, task=task, dataset_api=Dataset)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
