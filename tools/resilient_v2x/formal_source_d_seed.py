#!/usr/bin/env python3
"""Build explicit-seed, fail-closed-evidence source D from sealed source C."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence


LIVE_SOURCE_C_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
EXPECTED_SOURCE_C_SHA256 = LIVE_SOURCE_C_SHA256
EXPECTED_SOURCE_D_SHA256 = (
    "e7a9ab0fb05339223cf2c18c52eb72652c311733bf96d1058aa7a769096cf8c3"
)
EXPECTED_EQUIVALENCE_ARTIFACT_SHA256 = (
    "1156fe53f2fe924f91c1c6b50b6b21090d98cd2d74840d6d6f5a358316420433"
)
EXPECTED_DECLARED_REPLACEMENT_COUNT = 22
EXPECTED_UNCHANGED_SEGMENT_COUNT = 23
TRANSFORMATION_ID = "source-c-to-source-d-explicit-seed-evidence-v2"
TRAINING_OVERLAY_PROTOCOL_SEED = 20_250_218
MIN_TRAINING_SEED = 0
# NumPy-backed augmentation seeding requires the unsigned 32-bit range.
MAX_TRAINING_SEED = 2**32 - 1
PORTABLE_RUNNER_LOAD_MARKER = "_validate_rtx5090_runtime_contract_multi_gpu"
EXPECTED_PORTABLE_RUNNER_LOAD_MARKER_COUNT = 2
LEGACY_RUNNER_LOAD_TARGET_ANCHOR = """    if args.predecessor_task_id == task_id:
        raise RuntimeError("an experiment task cannot be its own predecessor")

    runner = _load_source_training_runner(source_root)
    dataset_root, env = _prepare_experiment_environment(
"""


class SourceDSeedError(RuntimeError):
    """Raised when source C or a derived source-D artifact fails closed."""


class _Anchor(NamedTuple):
    name: str
    before: str
    after: str


@dataclass(frozen=True)
class SourceDBuild:
    """One reproducible source-D build and its sealed transformation evidence."""

    source_d_text: str
    source_d_sha256: str
    artifact: dict[str, object]


_TRAINING_CFG_BINDINGS = (
    "randomness.seed",
    "train_dataloader.sampler.seed",
    "val_dataloader.sampler.seed",
    "test_dataloader.sampler.seed",
    "train_dataloader.dataset.seed",
    "val_dataloader.dataset.seed",
    "test_dataloader.dataset.seed",
    "implementation_choices_dataset.global_seed",
)

_STAGING_ANCHOR_NAMES = (
    "seal_evidence_security_imports",
    "declare_controlled_evidence_contract",
    "declare_controlled_evidence_receipts",
    "stage_and_verify_controlled_evidence",
    "stage_after_controlled_runner",
    "upload_only_sealed_controlled_evidence",
)
_STAGING_REQUIRED_FRAGMENTS = (
    ("class ControlledEvidenceMemberReceipt(NamedTuple):", 1),
    ("class ControlledEvidenceStage(NamedTuple):", 1),
    ("CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024", 1),
    ("CONTROLLED_EVIDENCE_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024", 1),
    ("CONTROLLED_EVIDENCE_MAX_COMPRESSION_RATIO = 128.0", 1),
    ("CONTROLLED_EVIDENCE_REQUIRED_FILENAMES = (", 1),
    ("def _stage_controlled_baseline_evidence(", 1),
    ("def _verify_controlled_evidence_stage(", 1),
    ("def _read_uploaded_artifact_bytes(", 1),
    ("def _snapshot_uploaded_evidence_zip(", 1),
    ("def _preflight_uploaded_evidence_zip(", 1),
    ("def _verify_uploaded_evidence_zip(", 1),
    ("def _verify_uploaded_controlled_evidence(", 1),
    ("def _reload_controlled_evidence_task(", 1),
    ("def _upload_controlled_baseline_artifacts(", 1),
    ("evidence_stage = _stage_controlled_baseline_evidence(", 1),
    ("if len(ordered) != 38 or sum(item.size_bytes for item in ordered) > (", 1),
    ("len(stage.members) != 38\n", 1),
    ("len(names) != 38\n", 1),
    ("or entries_on_disk != 38\n", 1),
    ("or entry_count != 38\n", 1),
    ('archive.read(4) == b"PK\\x06\\x07"', 1),
    (
        "central_directory_size > CONTROLLED_EVIDENCE_MAX_CENTRAL_DIRECTORY_BYTES",
        1,
    ),
    ('run["resolved_config_sha256"] = copied["resolved_config.py"].sha256', 1),
    ('resealed_plan["content_sha256"] = _producer_content_sha256(resealed_plan)', 1),
    ('("controlled_baseline_evidence", str(evidence_stage.root))', 1),
    ("extract_archive=False", 2),
    ("force_download=True", 2),
    ('original_name = getattr(member, "orig_filename", None)', 1),
    ("with tempfile.TemporaryFile(", 1),
    ('reloader = getattr(task, "_reload", None)', 1),
    ('setattr(task, "_data", snapshot)', 1),
    ("failed to flush controlled baseline artifact uploads", 1),
    ("st_nlink != 1", 5),
    ("_stat_identity(", 39),
    ("changed while being staged", 3),
    ("changed during readback", 1),
    ("staging root changed during verification", 1),
    ("cannot disable automatic archive extraction", 1),
)
_STAGING_FORBIDDEN_FRAGMENTS = (
    '("controlled_baseline_evidence", str(work_dir))',
    "metrics_getter()",
    'reloader = getattr(task, "reload", None)',
    "os.fdopen(os.dup(descriptor)",
)

_ANCHORS = (
    _Anchor(
        "declare_seed_contract",
        """CANONICAL_1337_PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
CANONICAL_1337_SAMPLE_COUNT = 1337
""",
        """CANONICAL_1337_PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
TRAINING_OVERLAY_PROTOCOL_SEED = 20250218
DEFAULT_TRAINING_SEED = TRAINING_OVERLAY_PROTOCOL_SEED
MAX_TRAINING_SEED = 2**32 - 1
CANONICAL_1337_SAMPLE_COUNT = 1337
""",
    ),
    _Anchor(
        "add_training_seed_type",
        """def _sha256_argument(value: str) -> str:
""",
        """def _training_seed_argument(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= MAX_TRAINING_SEED:
        raise argparse.ArgumentTypeError(
            f"value must be in [0, {MAX_TRAINING_SEED}]"
        )
    return parsed


def _sha256_argument(value: str) -> str:
""",
    ),
    _Anchor(
        "add_training_seed_cli",
        """    parser.add_argument("--max-epochs", type=_positive_integer, default=50)
    parser.add_argument("--teacher-checkpoint", type=Path)
""",
        """    parser.add_argument("--max-epochs", type=_positive_integer, default=50)
    parser.add_argument(
        "--training-seed",
        type=_training_seed_argument,
        default=DEFAULT_TRAINING_SEED,
        help=(
            "model, sampler, and dataset-augmentation seed; does not alter "
            "the sealed training-overlay protocol seed"
        ),
    )
    parser.add_argument("--teacher-checkpoint", type=Path)
""",
    ),
    _Anchor(
        "validate_training_seed_argument",
        """def _validate_arguments(args: argparse.Namespace) -> None:
    for field in ("source_dataset_id", "training_dataset_id"):
""",
        """def _validate_arguments(args: argparse.Namespace) -> None:
    if (
        type(args.training_seed) is not int
        or not 0 <= args.training_seed <= MAX_TRAINING_SEED
    ):
        raise ValueError(
            f"--training-seed must be in [0, {MAX_TRAINING_SEED}]"
        )
    for field in ("source_dataset_id", "training_dataset_id"):
""",
    ),
    _Anchor(
        "bind_ddp_training_seed_parameter",
        """def _ddp_training_command(
    python: Path,
    *,
    gpus: int,
    config: Path,
    work_dir: Path,
    max_epochs: int,
) -> list[str]:
""",
        """def _ddp_training_command(
    python: Path,
    *,
    gpus: int,
    config: Path,
    work_dir: Path,
    max_epochs: int,
    training_seed: int,
) -> list[str]:
""",
    ),
    _Anchor(
        "bind_ddp_cfg_seed_overrides",
        """        f"test_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        "find_unused_parameters=True",
""",
        """        f"test_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
        f"randomness.seed={training_seed}",
        f"train_dataloader.sampler.seed={training_seed}",
        f"val_dataloader.sampler.seed={training_seed}",
        f"test_dataloader.sampler.seed={training_seed}",
        f"train_dataloader.dataset.seed={training_seed}",
        f"val_dataloader.dataset.seed={training_seed}",
        f"test_dataloader.dataset.seed={training_seed}",
        f"implementation_choices_dataset.global_seed={training_seed}",
        "find_unused_parameters=True",
""",
    ),
    _Anchor(
        "bind_baseline_plan_seed_parameter",
        """def _baseline_plan_command(
    python: Path,
    *,
    source_root: Path,
    baseline: str,
    training_index: Path,
    work_dir: Path,
) -> list[str]:
""",
        """def _baseline_plan_command(
    python: Path,
    *,
    source_root: Path,
    baseline: str,
    training_index: Path,
    work_dir: Path,
    training_seed: int,
) -> list[str]:
""",
    ),
    _Anchor(
        "replace_baseline_plan_seed_literal",
        """        "--seed",
        "20250218",
        "--dry-run",
""",
        """        "--seed",
        str(training_seed),
        "--dry-run",
""",
    ),
    _Anchor(
        "validate_training_overlay_protocol_seed",
        """def _validate_baseline_dry_run(
    *,
    spec: ExperimentSpec,
    work_dir: Path,
) -> tuple[Path, Path, dict[str, object]]:
""",
        """def _training_overlay_protocol_seed(dataset_root: Path) -> int:
    index_path = dataset_root / "protocols/dair_v2/training_overlays.json"
    index = _read_json_object(index_path.resolve(strict=True))
    observed = index.get("protocol_seed")
    if type(observed) is not int or observed != TRAINING_OVERLAY_PROTOCOL_SEED:
        raise ValueError(
            "training overlay protocol seed mismatch: "
            f"expected {TRAINING_OVERLAY_PROTOCOL_SEED}, got {observed!r}"
        )
    return TRAINING_OVERLAY_PROTOCOL_SEED


def _validate_baseline_dry_run(
    *,
    spec: ExperimentSpec,
    work_dir: Path,
    training_seed: int,
) -> tuple[Path, Path, dict[str, object]]:
""",
    ),
    _Anchor(
        "validate_baseline_plan_seed_contract",
        """        "resolved_config": str(resolved_path),
        "resume": False,
    }
""",
        """        "resolved_config": str(resolved_path),
        "resume": False,
        "seed": training_seed,
        "training_index_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
    }
""",
    ),
    _Anchor(
        "validate_contract_seed_sources",
        """    else:
        declared_config = str(baseline_plan["baseline_config"])
        declared_config_sha256 = str(baseline_plan["baseline_config_sha256"])
    return {
        "schema_version": 1,
""",
        """    else:
        declared_config = str(baseline_plan["baseline_config"])
        declared_config_sha256 = str(baseline_plan["baseline_config_sha256"])
    overlay_protocol_seed = _training_overlay_protocol_seed(dataset_root)
    if baseline_plan is not None:
        if (
            type(baseline_plan.get("seed")) is not int
            or baseline_plan.get("seed") != args.training_seed
        ):
            raise ValueError("baseline plan training seed mismatch")
        if (
            type(baseline_plan.get("training_index_protocol_seed")) is not int
            or baseline_plan.get("training_index_protocol_seed")
            != overlay_protocol_seed
        ):
            raise ValueError("baseline plan training overlay protocol seed mismatch")
    return {
        "schema_version": 1,
""",
    ),
    _Anchor(
        "record_explicit_seed_contract",
        """        "max_epochs": EXPERIMENT_MAX_EPOCHS,
        "seed": 20250218,
        "learning_rate": 0.0001,
""",
        """        "max_epochs": EXPERIMENT_MAX_EPOCHS,
        "training_seed": args.training_seed,
        "training_overlay_protocol_seed": overlay_protocol_seed,
        "seed": args.training_seed,
        "learning_rate": 0.0001,
""",
    ),
    _Anchor(
        "pass_seed_to_baseline_plan",
        """            baseline=spec.name,
            training_index=training_index,
            work_dir=work_dir,
        )
""",
        """            baseline=spec.name,
            training_index=training_index,
            work_dir=work_dir,
            training_seed=args.training_seed,
        )
""",
    ),
    _Anchor(
        "pass_seed_to_baseline_validation",
        """        plan_path, config_path, baseline_plan = _validate_baseline_dry_run(
            spec=spec,
            work_dir=work_dir,
        )
""",
        """        plan_path, config_path, baseline_plan = _validate_baseline_dry_run(
            spec=spec,
            work_dir=work_dir,
            training_seed=args.training_seed,
        )
""",
    ),
    _Anchor(
        "pass_seed_to_ddp_training",
        """        config=config_path,
        work_dir=work_dir,
        max_epochs=args.max_epochs,
    )
    contract = _experiment_run_contract(
""",
        """        config=config_path,
        work_dir=work_dir,
        max_epochs=args.max_epochs,
        training_seed=args.training_seed,
    )
    contract = _experiment_run_contract(
""",
    ),
    _Anchor(
        "forward_seed_to_stage_runner",
        """        "--max-epochs",
        str(args.max_epochs),
    ]
""",
        """        "--max-epochs",
        str(args.max_epochs),
        "--training-seed",
        str(args.training_seed),
    ]
""",
    ),
    _Anchor(
        "seal_evidence_security_imports",
        r"""import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
""",
        r"""import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
""",
    ),
    _Anchor(
        "declare_controlled_evidence_contract",
        r"""CANONICAL_1337_SAMPLE_IDS_SHA256 = (
    "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
)
CLEARML_TRAIN_BASELINE_SHA256 = (""",
        r"""CANONICAL_1337_SAMPLE_IDS_SHA256 = (
    "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
)
CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES = 256 * 1024 * 1024
CONTROLLED_EVIDENCE_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024
CONTROLLED_EVIDENCE_MAX_COMPRESSION_RATIO = 128.0
CONTROLLED_EVIDENCE_REQUIRED_FILENAMES = (
    "resolved_config.py",
    "checkpoint.sha256",
    "predictions.json",
)
CONTROLLED_EVIDENCE_FORBIDDEN_STALE_HASH_FIELDS = frozenset(
    {
        "resolved_config_sha256",
        "evaluation_plan_sha256",
        "evaluation_plan_content_sha256",
        "plan_sha256",
        "plan_content_sha256",
    }
)
CLEARML_TRAIN_BASELINE_SHA256 = (""",
    ),
    _Anchor(
        "declare_controlled_evidence_receipts",
        r'''class ExperimentSpec(NamedTuple):
    """One immutable member of the post-main training suite."""

    name: str
    kind: str
    config: str | None
    requires_teacher: bool


CORE_EXPERIMENT_SPECS = (''',
        r'''class ExperimentSpec(NamedTuple):
    """One immutable member of the post-main training suite."""

    name: str
    kind: str
    config: str | None
    requires_teacher: bool


class ControlledEvidenceMemberReceipt(NamedTuple):
    """Immutable identity and content receipt for one staged evidence file."""

    relative_path: str
    size_bytes: int
    sha256: str
    identity: tuple[int, ...]


class ControlledEvidenceStage(NamedTuple):
    """Private exact-inventory snapshot uploaded as formal evaluation evidence."""

    root: Path
    root_identity: tuple[int, ...]
    members: tuple[ControlledEvidenceMemberReceipt, ...]


CORE_EXPERIMENT_SPECS = (''',
    ),
    _Anchor(
        "stage_and_verify_controlled_evidence",
        r"""def _upload_controlled_baseline_artifacts(
    task: object,
    *,
    work_dir: Path,
    metrics: Mapping[str, object],
) -> None:
    for artifact_name, artifact_object in (
        ("evaluation_plan", str(work_dir / "evaluation_plan.json")),
        ("controlled_baseline_metrics", dict(metrics)),
        ("controlled_baseline_evidence", str(work_dir)),
    ):
        if not task.upload_artifact(
            artifact_name,
            artifact_object=artifact_object,
            wait_on_upload=True,
        ):
            raise RuntimeError(f"failed to upload {artifact_name}")


def _execute_controlled_baseline_validation(""",
        r'''def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"value is outside the canonical JSON domain: {error}"
        ) from error


def _producer_content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("secure evidence directory open is unavailable")
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)


def _regular_file_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("secure evidence file open is unavailable")
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )


def _open_secure_evidence_root(path: Path, *, context: str) -> tuple[Path, int]:
    candidate = Path(path).expanduser()
    try:
        before = candidate.lstat()
    except OSError as error:
        raise RuntimeError(f"{context} is unavailable") from error
    if candidate.is_symlink() or not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"{context} must be a non-symlink directory")
    try:
        descriptor = os.open(candidate, _directory_open_flags())
    except OSError as error:
        raise RuntimeError(f"{context} cannot be securely opened") from error
    try:
        opened = os.fstat(descriptor)
        current = candidate.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _stat_identity(before) != _stat_identity(opened)
            or _stat_identity(current) != _stat_identity(opened)
        ):
            raise RuntimeError(f"{context} identity changed during secure open")
        resolved = candidate.resolve(strict=True)
    except BaseException:
        os.close(descriptor)
        raise
    return resolved, descriptor


def _validate_evidence_relative_path(
    relative: PurePosixPath,
    *,
    context: str,
) -> None:
    if (
        relative.is_absolute()
        or relative.as_posix() in {"", "."}
        or ".." in relative.parts
        or any(not part or "/" in part or "\\" in part for part in relative.parts)
    ):
        raise RuntimeError(f"{context} has an unsafe relative path")


def _open_evidence_member(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    context: str,
) -> tuple[int, list[int], os.stat_result]:
    _validate_evidence_relative_path(relative, context=context)
    directories = [os.dup(root_descriptor)]
    descriptor = -1
    try:
        for component in relative.parts[:-1]:
            parent = directories[-1]
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise RuntimeError(f"{context} parent is not a regular directory")
            child = os.open(component, _directory_open_flags(), dir_fd=parent)
            directories.append(child)
            opened = os.fstat(child)
            if _stat_identity(before) != _stat_identity(opened):
                raise RuntimeError(f"{context} parent changed during secure open")

        parent = directories[-1]
        filename = relative.name
        before = os.stat(filename, dir_fd=parent, follow_symlinks=False)
        descriptor = os.open(filename, _regular_file_open_flags(), dir_fd=parent)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size <= 0
            or opened.st_size > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES
            or _stat_identity(before) != _stat_identity(opened)
        ):
            raise RuntimeError(f"{context} is not a sealed regular file")
        return descriptor, directories, opened
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        for descriptor in reversed(directories):
            os.close(descriptor)
        raise


def _verify_open_evidence_member_unchanged(
    descriptor: int,
    directories: Sequence[int],
    relative: PurePosixPath,
    before: os.stat_result,
    *,
    context: str,
) -> os.stat_result:
    after = os.fstat(descriptor)
    current = os.stat(
        relative.name,
        dir_fd=directories[-1],
        follow_symlinks=False,
    )
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(
        after
    ) != _stat_identity(current):
        raise RuntimeError(f"{context} changed while being staged")
    for index, component in enumerate(relative.parts[:-1]):
        child = os.fstat(directories[index + 1])
        current_child = os.stat(
            component,
            dir_fd=directories[index],
            follow_symlinks=False,
        )
        if _stat_identity(child) != _stat_identity(current_child):
            raise RuntimeError(f"{context} parent changed while being staged")
    return after


def _close_evidence_member(descriptor: int, directories: Sequence[int]) -> None:
    os.close(descriptor)
    for directory in reversed(directories):
        os.close(directory)


def _read_evidence_member(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    context: str,
) -> bytes:
    descriptor, directories, before = _open_evidence_member(
        root_descriptor,
        relative,
        context=context,
    )
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
                raise RuntimeError(f"{context} exceeded its byte cap")
            chunks.append(chunk)
        _verify_open_evidence_member_unchanged(
            descriptor,
            directories,
            relative,
            before,
            context=context,
        )
        if total != before.st_size:
            raise RuntimeError(f"{context} size changed while being read")
        return b"".join(chunks)
    finally:
        _close_evidence_member(descriptor, directories)


def _copy_fd_payload(source: int, destination: int) -> tuple[int, str]:
    """Copy one already-securely-opened file; split out for TOCTOU tests."""

    digest = hashlib.sha256()
    copied = 0
    while True:
        chunk = os.read(source, 1024 * 1024)
        if not chunk:
            break
        copied += len(chunk)
        if copied > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
            raise RuntimeError("controlled evidence member exceeded its byte cap")
        digest.update(chunk)
        view = memoryview(chunk)
        while view:
            written = os.write(destination, view)
            if written <= 0:
                raise RuntimeError("controlled evidence staging write made no progress")
            view = view[written:]
    return copied, digest.hexdigest()


def _open_stage_parent(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    context: str,
) -> list[int]:
    _validate_evidence_relative_path(relative, context=context)
    directories = [os.dup(root_descriptor)]
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                _directory_open_flags(),
                dir_fd=directories[-1],
            )
            directories.append(child)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                raise RuntimeError(f"{context} staging parent is not a directory")
        return directories
    except BaseException:
        for descriptor in reversed(directories):
            os.close(descriptor)
        raise


def _stage_copy_evidence_member(
    source_root: int,
    stage_root: int,
    relative: PurePosixPath,
) -> ControlledEvidenceMemberReceipt:
    context = f"controlled evidence member {relative.as_posix()!r}"
    source, source_directories, source_before = _open_evidence_member(
        source_root,
        relative,
        context=context,
    )
    stage_directories: list[int] = []
    destination = -1
    try:
        stage_directories = _open_stage_parent(
            stage_root,
            relative,
            context=context,
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_BINARY", 0)
        )
        destination = os.open(
            relative.name,
            flags,
            0o400,
            dir_fd=stage_directories[-1],
        )
        copied, digest = _copy_fd_payload(source, destination)
        os.fsync(destination)
        _verify_open_evidence_member_unchanged(
            source,
            source_directories,
            relative,
            source_before,
            context=context,
        )
        if copied != source_before.st_size:
            raise RuntimeError(f"{context} size changed while being staged")
        os.fchmod(destination, 0o400)
        staged = os.fstat(destination)
        current = os.stat(
            relative.name,
            dir_fd=stage_directories[-1],
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged.st_nlink != 1
            or _stat_identity(staged) != _stat_identity(current)
            or staged.st_size != copied
        ):
            raise RuntimeError(f"{context} staging destination drifted")
        return ControlledEvidenceMemberReceipt(
            relative.as_posix(),
            copied,
            digest,
            _stat_identity(staged),
        )
    finally:
        if destination >= 0:
            os.close(destination)
        for descriptor in reversed(stage_directories):
            os.close(descriptor)
        _close_evidence_member(source, source_directories)


def _stage_write_evidence_member(
    stage_root: int,
    relative: PurePosixPath,
    raw: bytes,
) -> ControlledEvidenceMemberReceipt:
    if not raw or len(raw) > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
        raise RuntimeError(f"staged evidence member {relative} has an invalid size")
    directories = _open_stage_parent(
        stage_root,
        relative,
        context=f"staged evidence member {relative.as_posix()!r}",
    )
    descriptor = -1
    try:
        descriptor = os.open(
            relative.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_BINARY", 0),
            0o400,
            dir_fd=directories[-1],
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("controlled evidence staging write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        value = os.fstat(descriptor)
        current = os.stat(
            relative.name,
            dir_fd=directories[-1],
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or value.st_size != len(raw)
            or _stat_identity(value) != _stat_identity(current)
        ):
            raise RuntimeError(f"staged evidence member {relative} drifted")
        return ControlledEvidenceMemberReceipt(
            relative.as_posix(),
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _stat_identity(value),
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)


def _json_mapping_from_bytes(raw: bytes, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise RuntimeError(f"{context} must be a JSON object")
    return value


def _reject_stale_evidence_hash_fields(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        stale = CONTROLLED_EVIDENCE_FORBIDDEN_STALE_HASH_FIELDS.intersection(value)
        if stale:
            raise RuntimeError(
                f"{context} contains stale plan/config hash fields: {sorted(stale)}"
            )
        for key, item in value.items():
            _reject_stale_evidence_hash_fields(item, context=f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_stale_evidence_hash_fields(
                item,
                context=f"{context}[{index}]",
            )


def _controlled_condition_ids() -> tuple[str, ...]:
    return tuple(
        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        for delay in (0, 100, 200, 300)
        for condition in ("Full", "L-Fail", "C-Fail")
    )


def _validate_controlled_evidence_documents(
    plan: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    work_dir: Path,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    work_dir_text = str(work_dir)
    expected_paths = {
        "work_dir": work_dir_text,
        "plan_path": str(work_dir / "evaluation_plan.json"),
        "metrics_output": str(work_dir / "metrics.json"),
    }
    for field, expected in expected_paths.items():
        if plan.get(field) != expected:
            raise RuntimeError(f"controlled evidence plan {field} drifted")
    declared_hash = plan.get("content_sha256")
    if (
        type(declared_hash) is not str
        or SHA256_PATTERN.fullmatch(declared_hash) is None
        or declared_hash != _producer_content_sha256(plan)
    ):
        raise RuntimeError("controlled evidence source plan content hash mismatch")

    plan_runs = plan.get("runs")
    metric_runs = metrics.get("runs")
    condition_ids = _controlled_condition_ids()
    if (
        not isinstance(plan_runs, list)
        or not isinstance(metric_runs, list)
        or len(plan_runs) != len(condition_ids)
        or len(metric_runs) != len(condition_ids)
        or metrics.get("complete") is not True
        or metrics.get("planned_run_count") != len(condition_ids)
    ):
        raise RuntimeError("controlled evidence matrix is not exactly 12 complete runs")
    _reject_stale_evidence_hash_fields(metrics, context="controlled metrics")

    normalized_plan_runs: list[Mapping[str, object]] = []
    normalized_metric_runs: list[Mapping[str, object]] = []
    for index, condition_id in enumerate(condition_ids):
        plan_run = plan_runs[index]
        metric_run = metric_runs[index]
        if not isinstance(plan_run, Mapping) or not isinstance(metric_run, Mapping):
            raise RuntimeError(f"controlled evidence run {index} is not an object")
        delay = (0, 100, 200, 300)[index // 3]
        condition = ("Full", "L-Fail", "C-Fail")[index % 3]
        expected_run = {
            "condition_id": condition_id,
            "delay_ms": delay,
            "condition": condition,
            "resolved_config": str(work_dir / condition_id / "resolved_config.py"),
            "predictions": str(work_dir / condition_id / "predictions.json"),
            "checkpoint_sha256_file": str(
                work_dir / condition_id / "checkpoint.sha256"
            ),
        }
        for field, expected in expected_run.items():
            if plan_run.get(field) != expected:
                raise RuntimeError(f"controlled evidence run {index} {field} drifted")
        for field in ("condition_id", "delay_ms", "condition", "predictions"):
            if metric_run.get(field) != expected_run[field]:
                raise RuntimeError(
                    f"controlled metrics run {index} {field} is not cross-bound"
                )
        normalized_plan_runs.append(plan_run)
        normalized_metric_runs.append(metric_run)
    return normalized_plan_runs, normalized_metric_runs


def _create_private_evidence_stage(destination: Path) -> tuple[Path, int]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to reuse evidence staging directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(destination, 0o700)
    return _open_secure_evidence_root(
        destination,
        context="controlled baseline evidence staging directory",
    )


def _stage_controlled_baseline_evidence(
    *,
    work_dir: Path,
    staging_dir: Path | None = None,
) -> ControlledEvidenceStage:
    """Create one private, exact 38-member snapshot from a polluted Runner work dir."""

    source_path, source_descriptor = _open_secure_evidence_root(
        work_dir,
        context="controlled baseline evaluation work directory",
    )
    try:
        source_identity = _stat_identity(os.fstat(source_descriptor))
        destination = (
            source_path.with_name(f".{source_path.name}.controlled-baseline-evidence")
            if staging_dir is None
            else Path(staging_dir).expanduser()
        )
        stage_path, stage_descriptor = _create_private_evidence_stage(destination)
    except BaseException:
        os.close(source_descriptor)
        raise
    receipts: list[ControlledEvidenceMemberReceipt] = []
    try:
        source_plan = _json_mapping_from_bytes(
            _read_evidence_member(
                source_descriptor,
                PurePosixPath("evaluation_plan.json"),
                context="source evaluation plan",
            ),
            context="source evaluation plan",
        )
        source_metrics_raw = _read_evidence_member(
            source_descriptor,
            PurePosixPath("metrics.json"),
            context="source controlled metrics",
        )
        source_metrics = _json_mapping_from_bytes(
            source_metrics_raw,
            context="source controlled metrics",
        )
        plan_runs, metric_runs = _validate_controlled_evidence_documents(
            source_plan,
            source_metrics,
            work_dir=source_path,
        )

        condition_ids = _controlled_condition_ids()
        for condition_id in condition_ids:
            os.mkdir(condition_id, 0o700, dir_fd=stage_descriptor)

        metrics_receipt = _stage_copy_evidence_member(
            source_descriptor,
            stage_descriptor,
            PurePosixPath("metrics.json"),
        )
        if metrics_receipt.sha256 != hashlib.sha256(source_metrics_raw).hexdigest():
            raise RuntimeError("source controlled metrics changed before staging")
        receipts.append(metrics_receipt)
        resealed_runs: list[dict[str, object]] = []
        checkpoint_sha256 = source_plan.get("checkpoint_sha256")
        if (
            type(checkpoint_sha256) is not str
            or SHA256_PATTERN.fullmatch(checkpoint_sha256) is None
        ):
            raise RuntimeError("controlled evidence checkpoint SHA-256 is invalid")
        for index, condition_id in enumerate(condition_ids):
            run = dict(plan_runs[index])
            copied: dict[str, ControlledEvidenceMemberReceipt] = {}
            for filename in CONTROLLED_EVIDENCE_REQUIRED_FILENAMES:
                relative = PurePosixPath(condition_id, filename)
                receipt = _stage_copy_evidence_member(
                    source_descriptor,
                    stage_descriptor,
                    relative,
                )
                receipts.append(receipt)
                copied[filename] = receipt
            run["resolved_config_sha256"] = copied["resolved_config.py"].sha256
            metric_prediction_sha = metric_runs[index].get("prediction_sha256")
            if metric_prediction_sha != copied["predictions.json"].sha256:
                raise RuntimeError(
                    f"controlled evidence run {index} prediction SHA-256 drifted"
                )
            checkpoint_raw = _read_evidence_member(
                stage_descriptor,
                PurePosixPath(condition_id, "checkpoint.sha256"),
                context=f"staged checkpoint SHA file for run {index}",
            )
            if checkpoint_raw != f"{checkpoint_sha256}\n".encode("ascii"):
                raise RuntimeError(
                    f"controlled evidence run {index} checkpoint SHA file drifted"
                )
            resealed_runs.append(run)

        resealed_plan = dict(source_plan)
        resealed_plan["runs"] = resealed_runs
        resealed_plan["content_sha256"] = _producer_content_sha256(resealed_plan)
        plan_raw = (
            json.dumps(
                resealed_plan,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        receipts.append(
            _stage_write_evidence_member(
                stage_descriptor,
                PurePosixPath("evaluation_plan.json"),
                plan_raw,
            )
        )

        for condition_id in condition_ids:
            directory = os.open(
                condition_id,
                _directory_open_flags(),
                dir_fd=stage_descriptor,
            )
            try:
                os.fchmod(directory, 0o500)
                os.fsync(directory)
            finally:
                os.close(directory)
        os.fchmod(stage_descriptor, 0o500)
        os.fsync(stage_descriptor)
        current_source = source_path.lstat()
        if source_identity != _stat_identity(current_source):
            raise RuntimeError(
                "controlled baseline evaluation work directory changed during staging"
            )
        root_identity = _stat_identity(os.fstat(stage_descriptor))
    finally:
        os.close(stage_descriptor)
        os.close(source_descriptor)

    ordered = tuple(sorted(receipts, key=lambda item: item.relative_path))
    if len(ordered) != 38 or sum(item.size_bytes for item in ordered) > (
        CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES
    ):
        raise RuntimeError("controlled baseline evidence staging inventory is invalid")
    stage = ControlledEvidenceStage(stage_path, root_identity, ordered)
    _verify_controlled_evidence_stage(stage)
    return stage


def _walk_staged_evidence(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    stack: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath("."))]
    while stack:
        directory, relative_root = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = (
                    PurePosixPath(entry.name)
                    if relative_root.as_posix() == "."
                    else relative_root / entry.name
                )
                value = entry.stat(follow_symlinks=False)
                if entry.is_symlink():
                    raise RuntimeError(
                        f"controlled evidence staging contains symlink {relative}"
                    )
                if stat.S_ISDIR(value.st_mode):
                    if stat.S_IMODE(value.st_mode) != 0o500:
                        raise RuntimeError(
                            f"controlled evidence staging directory mode drifted: {relative}"
                        )
                    directories.add(relative.as_posix())
                    stack.append((Path(entry.path), relative))
                elif stat.S_ISREG(value.st_mode) and value.st_nlink == 1:
                    if stat.S_IMODE(value.st_mode) != 0o400:
                        raise RuntimeError(
                            f"controlled evidence staging file mode drifted: {relative}"
                        )
                    files.add(relative.as_posix())
                else:
                    raise RuntimeError(
                        f"controlled evidence staging has non-regular member {relative}"
                    )
    return files, directories


def _hash_evidence_member(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    context: str,
) -> tuple[int, str, tuple[int, ...]]:
    descriptor, directories, before = _open_evidence_member(
        root_descriptor,
        relative,
        context=context,
    )
    try:
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
                raise RuntimeError(f"{context} exceeded its byte cap")
            digest.update(chunk)
        after = _verify_open_evidence_member_unchanged(
            descriptor,
            directories,
            relative,
            before,
            context=context,
        )
        return total, digest.hexdigest(), _stat_identity(after)
    finally:
        _close_evidence_member(descriptor, directories)


def _verify_controlled_evidence_stage(
    stage: ControlledEvidenceStage,
) -> tuple[dict[str, object], dict[str, object]]:
    root, descriptor = _open_secure_evidence_root(
        stage.root,
        context="controlled baseline evidence staging directory",
    )
    try:
        if _stat_identity(os.fstat(descriptor)) != stage.root_identity:
            raise RuntimeError("controlled evidence staging root identity drifted")
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o500:
            raise RuntimeError("controlled evidence staging root mode drifted")
        expected_files = {receipt.relative_path for receipt in stage.members}
        expected_directories = {
            PurePosixPath(path).parent.as_posix()
            for path in expected_files
            if PurePosixPath(path).parent.as_posix() != "."
        }
        observed_files, observed_directories = _walk_staged_evidence(root)
        if (
            len(stage.members) != 38
            or len(expected_files) != 38
            or observed_files != expected_files
            or observed_directories != expected_directories
        ):
            raise RuntimeError("controlled evidence staging exact inventory drifted")
        for receipt in stage.members:
            size, digest, identity = _hash_evidence_member(
                descriptor,
                PurePosixPath(receipt.relative_path),
                context=f"staged evidence member {receipt.relative_path!r}",
            )
            if (
                size != receipt.size_bytes
                or digest != receipt.sha256
                or identity != receipt.identity
            ):
                raise RuntimeError(
                    f"controlled evidence staging member drifted: {receipt.relative_path}"
                )
        plan = _json_mapping_from_bytes(
            _read_evidence_member(
                descriptor,
                PurePosixPath("evaluation_plan.json"),
                context="staged evaluation plan",
            ),
            context="staged evaluation plan",
        )
        metrics = _json_mapping_from_bytes(
            _read_evidence_member(
                descriptor,
                PurePosixPath("metrics.json"),
                context="staged controlled metrics",
            ),
            context="staged controlled metrics",
        )
        plan_runs, metric_runs = _validate_controlled_evidence_documents(
            plan,
            metrics,
            work_dir=Path(str(plan["work_dir"])),
        )
        receipt_by_path = {item.relative_path: item for item in stage.members}
        checkpoint_sha256 = plan.get("checkpoint_sha256")
        for index, condition_id in enumerate(_controlled_condition_ids()):
            config_receipt = receipt_by_path[f"{condition_id}/resolved_config.py"]
            prediction_receipt = receipt_by_path[f"{condition_id}/predictions.json"]
            if plan_runs[index].get("resolved_config_sha256") != config_receipt.sha256:
                raise RuntimeError(
                    f"staged evaluation plan run {index} config SHA-256 drifted"
                )
            if metric_runs[index].get("prediction_sha256") != prediction_receipt.sha256:
                raise RuntimeError(
                    f"staged controlled metrics run {index} prediction SHA drifted"
                )
            checkpoint_raw = _read_evidence_member(
                descriptor,
                PurePosixPath(condition_id, "checkpoint.sha256"),
                context=f"staged checkpoint SHA file for run {index}",
            )
            if checkpoint_raw != f"{checkpoint_sha256}\n".encode("ascii"):
                raise RuntimeError(
                    f"staged checkpoint SHA file for run {index} drifted"
                )
            prediction_raw = _read_evidence_member(
                descriptor,
                PurePosixPath(condition_id, "predictions.json"),
                context=f"staged predictions for run {index}",
            )
            if hashlib.sha256(prediction_raw).hexdigest() != prediction_receipt.sha256:
                raise RuntimeError(
                    f"staged predictions for run {index} changed during verification"
                )
            prediction = _json_mapping_from_bytes(
                prediction_raw,
                context=f"staged predictions for run {index}",
            )
            _reject_stale_evidence_hash_fields(
                prediction,
                context=f"staged predictions for run {index}",
            )
        if _stat_identity(stage.root.lstat()) != stage.root_identity:
            raise RuntimeError(
                "controlled evidence staging root changed during verification"
            )
        return plan, metrics
    finally:
        os.close(descriptor)


def _read_uploaded_artifact_file(artifact: object, *, context: str) -> Path:
    _require_files_server_url(getattr(artifact, "url", None), context=context)
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot be downloaded")
    try:
        value = getter(
            extract_archive=False,
            raise_on_error=True,
            force_download=True,
        )
    except TypeError as error:
        raise RuntimeError(
            f"{context} downloader cannot disable automatic archive extraction"
        ) from error
    if not value:
        raise RuntimeError(f"{context} returned no local copy")
    path = Path(value)
    try:
        result = path.lstat()
    except OSError as error:
        raise RuntimeError(f"{context} local copy is unavailable") from error
    if path.is_symlink() or not stat.S_ISREG(result.st_mode):
        raise RuntimeError(f"{context} local copy is not a regular file")
    return path


def _read_uploaded_artifact_bytes(artifact: object, *, context: str) -> bytes:
    path = _read_uploaded_artifact_file(artifact, context=context)
    try:
        descriptor = os.open(path, _regular_file_open_flags())
    except OSError as error:
        raise RuntimeError(f"{context} local copy cannot be securely opened") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES
        ):
            raise RuntimeError(f"{context} local copy size/type drifted")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES:
                raise RuntimeError(f"{context} exceeded its byte cap")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = path.lstat()
        if (
            total != before.st_size
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
        ):
            raise RuntimeError(f"{context} changed during readback")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _snapshot_uploaded_evidence_zip(
    descriptor: int,
    snapshot: object,
    *,
    before: os.stat_result,
) -> int:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        copied = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES:
                raise RuntimeError("uploaded evidence ZIP exceeded its snapshot cap")
            snapshot.write(chunk)
        after = os.fstat(descriptor)
        position = snapshot.tell()
        snapshot.flush()
        snapshot.seek(0)
    except OSError as error:
        raise RuntimeError("uploaded evidence ZIP snapshot failed") from error
    if (
        copied != before.st_size
        or position != copied
        or _stat_identity(before) != _stat_identity(after)
    ):
        raise RuntimeError("uploaded evidence ZIP changed during snapshot")
    return copied


def _preflight_uploaded_evidence_zip(
    archive: object,
    *,
    archive_size: int,
    expected_members: set[str],
) -> None:
    eocd_size = 22
    if len(expected_members) != 38 or archive_size < eocd_size:
        raise RuntimeError("uploaded evidence ZIP EOCD/inventory is invalid")
    try:
        if archive_size >= eocd_size + 20:
            archive.seek(archive_size - eocd_size - 20)
            if archive.read(4) == b"PK\x06\x07":
                raise RuntimeError("uploaded evidence ZIP64 locator is forbidden")
        archive.seek(archive_size - eocd_size)
        eocd = archive.read(eocd_size)
        if len(eocd) != eocd_size:
            raise RuntimeError("uploaded evidence ZIP EOCD is truncated")
        (
            signature,
            disk_number,
            central_directory_disk,
            entries_on_disk,
            entry_count,
            central_directory_size,
            central_directory_offset,
            comment_size,
        ) = struct.unpack("<4s4H2LH", eocd)
    except (OSError, struct.error) as error:
        raise RuntimeError("uploaded evidence ZIP EOCD is unreadable") from error
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or central_directory_disk != 0
        or entries_on_disk != 38
        or entry_count != 38
        or entries_on_disk == 0xFFFF
        or entry_count == 0xFFFF
        or comment_size != 0
        or central_directory_size <= 0
        or central_directory_size > CONTROLLED_EVIDENCE_MAX_CENTRAL_DIRECTORY_BYTES
        or central_directory_offset == 0xFFFFFFFF
        or central_directory_size == 0xFFFFFFFF
        or central_directory_offset + central_directory_size
        != archive_size - eocd_size
    ):
        raise RuntimeError("uploaded evidence ZIP EOCD/ZIP64 contract drifted")

    try:
        archive.seek(central_directory_offset)
        central_directory = archive.read(central_directory_size)
    except OSError as error:
        raise RuntimeError(
            "uploaded evidence ZIP central directory is unreadable"
        ) from error
    if len(central_directory) != central_directory_size:
        raise RuntimeError("uploaded evidence ZIP central directory is truncated")
    try:
        expected_names = {name.encode("ascii") for name in expected_members}
    except UnicodeEncodeError as error:
        raise RuntimeError("uploaded evidence ZIP expected names are not ASCII") from error

    observed_names: list[bytes] = []
    observed_offsets: set[int] = set()
    total_size = 0
    cursor = 0
    for index in range(entry_count):
        if cursor + 46 > len(central_directory):
            raise RuntimeError(
                f"uploaded evidence ZIP central entry {index} is truncated"
            )
        try:
            (
                entry_signature,
                version_made_by,
                version_needed,
                flag_bits,
                compression,
                _modified_time,
                _modified_date,
                _crc32,
                compressed_size,
                file_size,
                filename_size,
                extra_size,
                entry_comment_size,
                entry_disk,
                _internal_attributes,
                external_attributes,
                local_header_offset,
            ) = struct.unpack_from(
                "<4s6H3L5H2L",
                central_directory,
                cursor,
            )
        except struct.error as error:
            raise RuntimeError(
                f"uploaded evidence ZIP central entry {index} is invalid"
            ) from error
        entry_end = cursor + 46 + filename_size + extra_size + entry_comment_size
        if entry_end > len(central_directory):
            raise RuntimeError(
                f"uploaded evidence ZIP central entry {index} overflows"
            )
        raw_name = central_directory[cursor + 46 : cursor + 46 + filename_size]
        unix_mode = external_attributes >> 16
        if (
            entry_signature != b"PK\x01\x02"
            or version_made_by >> 8 != 3
            or version_needed >= 45
            or flag_bits not in {0, 0x800}
            or compression not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or compressed_size <= 0
            or file_size <= 0
            or file_size > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES
            or file_size / compressed_size
            > CONTROLLED_EVIDENCE_MAX_COMPRESSION_RATIO
            or not 0 < filename_size <= 255
            or extra_size != 0
            or entry_comment_size != 0
            or entry_disk != 0
            or not stat.S_ISREG(unix_mode)
            or local_header_offset >= central_directory_offset
            or local_header_offset == 0xFFFFFFFF
            or local_header_offset in observed_offsets
            or raw_name not in expected_names
            or b"\x00" in raw_name
            or b"\\" in raw_name
        ):
            raise RuntimeError(
                f"uploaded evidence ZIP central entry {index} is unsafe"
            )
        observed_names.append(raw_name)
        observed_offsets.add(local_header_offset)
        total_size += file_size
        cursor = entry_end

    if (
        cursor != len(central_directory)
        or len(set(observed_names)) != len(observed_names)
        or set(observed_names) != expected_names
        or total_size > CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES
    ):
        raise RuntimeError("uploaded evidence ZIP central inventory drifted")
    archive.seek(0)


def _verify_uploaded_evidence_zip(
    archive_path: Path,
    stage: ControlledEvidenceStage,
) -> None:
    descriptor = os.open(archive_path, _regular_file_open_flags())
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES
        ):
            raise RuntimeError("uploaded evidence ZIP size/type drifted")
        expected = {item.relative_path: item for item in stage.members}
        with tempfile.TemporaryFile(mode="w+b") as stream:
            archive_size = _snapshot_uploaded_evidence_zip(
                descriptor,
                stream,
                before=before,
            )
            current = archive_path.lstat()
            if _stat_identity(before) != _stat_identity(current):
                raise RuntimeError("uploaded evidence ZIP changed during snapshot")
            _preflight_uploaded_evidence_zip(
                stream,
                archive_size=archive_size,
                expected_members=set(expected),
            )
            try:
                with zipfile.ZipFile(stream, "r", allowZip64=False) as archive:
                    members = archive.infolist()
                    if archive.comment:
                        raise RuntimeError("uploaded evidence ZIP comment is forbidden")
                    names = [member.filename for member in members]
                    if (
                        len(names) != 38
                        or len(set(names)) != 38
                        or set(names) != set(expected)
                    ):
                        raise RuntimeError(
                            "uploaded evidence ZIP exact inventory drifted"
                        )
                    total = 0
                    for member in members:
                        original_name = getattr(member, "orig_filename", None)
                        relative = PurePosixPath(member.filename)
                        unix_mode = member.external_attr >> 16
                        if (
                            type(member.filename) is not str
                            or type(original_name) is not str
                            or original_name != member.filename
                            or "\x00" in original_name
                            or member.create_system != 3
                            or relative.is_absolute()
                            or relative.as_posix() != member.filename
                            or ".." in relative.parts
                            or "\\" in member.filename
                            or member.is_dir()
                            or member.extract_version >= 45
                            or member.flag_bits not in {0, 0x800}
                            or member.compress_type
                            not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                            or member.file_size <= 0
                            or member.file_size > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES
                            or member.compress_size <= 0
                            or member.file_size / member.compress_size
                            > CONTROLLED_EVIDENCE_MAX_COMPRESSION_RATIO
                            or member.extra
                            or member.comment
                            or not stat.S_ISREG(unix_mode)
                        ):
                            raise RuntimeError(
                                f"uploaded evidence ZIP member is unsafe: {member.filename!r}"
                            )
                        receipt = expected[member.filename]
                        digest = hashlib.sha256()
                        copied = 0
                        with archive.open(member, "r") as source:
                            while True:
                                chunk = source.read(1024 * 1024)
                                if not chunk:
                                    break
                                copied += len(chunk)
                                total += len(chunk)
                                if (
                                    copied > CONTROLLED_EVIDENCE_MAX_MEMBER_BYTES
                                    or total > CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES
                                ):
                                    raise RuntimeError(
                                        "uploaded evidence ZIP exceeded its byte cap"
                                    )
                                digest.update(chunk)
                        if (
                            copied != receipt.size_bytes
                            or copied != member.file_size
                            or digest.hexdigest() != receipt.sha256
                        ):
                            raise RuntimeError(
                                f"uploaded evidence ZIP member drifted: {member.filename}"
                            )
            except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
                if isinstance(error, RuntimeError):
                    raise
                raise RuntimeError(
                    "uploaded controlled evidence ZIP is invalid"
                ) from error
        after = os.fstat(descriptor)
        current = archive_path.lstat()
        if _stat_identity(before) != _stat_identity(after) or _stat_identity(
            after
        ) != _stat_identity(current):
            raise RuntimeError("uploaded evidence ZIP changed during verification")
    finally:
        os.close(descriptor)


def _verify_uploaded_controlled_evidence(
    task: object,
    stage: ControlledEvidenceStage,
) -> None:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("uploaded controlled evidence artifacts are unavailable")
    required = {
        "run_contract",
        "evaluation_plan",
        "controlled_baseline_metrics",
        "controlled_baseline_evidence",
    }
    if set(artifacts) != required:
        raise RuntimeError(
            "uploaded controlled evidence exact artifact inventory drifted"
        )

    stage_root, stage_descriptor = _open_secure_evidence_root(
        stage.root,
        context="controlled baseline evidence staging directory",
    )
    del stage_root
    try:
        staged_plan_raw = _read_evidence_member(
            stage_descriptor,
            PurePosixPath("evaluation_plan.json"),
            context="staged evaluation plan",
        )
    finally:
        os.close(stage_descriptor)
    uploaded_plan_raw = _read_uploaded_artifact_bytes(
        artifacts["evaluation_plan"],
        context="uploaded evaluation_plan artifact",
    )
    if uploaded_plan_raw != staged_plan_raw:
        raise RuntimeError("uploaded evaluation_plan bytes differ from sealed staging")

    _staged_plan, staged_metrics = _verify_controlled_evidence_stage(stage)
    metrics_artifact = artifacts["controlled_baseline_metrics"]
    if getattr(metrics_artifact, "type", None) != "dict":
        raise RuntimeError(
            "uploaded controlled_baseline_metrics is not a dict artifact"
        )
    uploaded_metrics = _json_mapping_from_bytes(
        _read_uploaded_artifact_bytes(
            metrics_artifact,
            context="uploaded controlled_baseline_metrics artifact",
        ),
        context="uploaded controlled_baseline_metrics artifact",
    )
    if _canonical_json_bytes(uploaded_metrics) != _canonical_json_bytes(staged_metrics):
        raise RuntimeError(
            "uploaded controlled_baseline_metrics differs from sealed staging"
        )

    evidence = artifacts["controlled_baseline_evidence"]
    if getattr(evidence, "type", None) != "archive":
        raise RuntimeError("uploaded controlled evidence artifact is not an archive")
    evidence_url = _require_files_server_url(
        getattr(evidence, "url", None),
        context="uploaded controlled_baseline_evidence artifact",
    )
    if PurePosixPath(urlsplit(evidence_url).path).suffix.lower() != ".zip":
        raise RuntimeError("uploaded controlled evidence artifact URL is not a ZIP")
    archive_path = _read_uploaded_artifact_file(
        evidence,
        context="uploaded controlled_baseline_evidence artifact",
    )
    _verify_uploaded_evidence_zip(archive_path, stage)


def _controlled_evidence_task_id(value: object, *, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{context} must be a lowercase 32-hex ClearML ID")
    return value


def _reload_controlled_evidence_task(task: object) -> None:
    expected_task_id = _controlled_evidence_task_id(
        getattr(task, "id", ""),
        context="controlled baseline local task",
    )
    if bool(getattr(task, "_offline_mode", False)):
        raise RuntimeError("controlled baseline task cannot use offline reload")
    reloader = getattr(task, "_reload", None)
    if not callable(reloader):
        raise RuntimeError("controlled baseline task cannot be server-reloaded")
    has_skip_flag = hasattr(task, "_reload_skip_flag")
    previous_skip_flag = getattr(task, "_reload_skip_flag", None)
    try:
        if has_skip_flag:
            setattr(task, "_reload_skip_flag", False)
        snapshot = reloader()
    except Exception as error:
        raise RuntimeError("controlled baseline task server reload failed") from error
    finally:
        if has_skip_flag:
            setattr(task, "_reload_skip_flag", previous_skip_flag)
    if snapshot is None or isinstance(
        snapshot,
        (bool, int, float, str, bytes, bytearray),
    ):
        raise RuntimeError("controlled baseline task server reload returned no snapshot")
    snapshot_task_id = _controlled_evidence_task_id(
        getattr(snapshot, "id", ""),
        context="controlled baseline server snapshot",
    )
    if snapshot_task_id != expected_task_id:
        raise RuntimeError("controlled baseline server snapshot identity mismatch")
    try:
        setattr(task, "_data", snapshot)
    except Exception as error:
        raise RuntimeError(
            "controlled baseline server snapshot cannot be installed"
        ) from error


def _upload_controlled_baseline_artifacts(
    task: object,
    *,
    evidence_stage: ControlledEvidenceStage,
) -> None:
    _plan, metrics = _verify_controlled_evidence_stage(evidence_stage)
    for artifact_name, artifact_object in (
        (
            "evaluation_plan",
            str(evidence_stage.root / "evaluation_plan.json"),
        ),
        ("controlled_baseline_metrics", dict(metrics)),
        ("controlled_baseline_evidence", str(evidence_stage.root)),
    ):
        _verify_controlled_evidence_stage(evidence_stage)
        if task.upload_artifact(
            artifact_name,
            artifact_object=artifact_object,
            wait_on_upload=True,
        ) is not True:
            raise RuntimeError(f"failed to upload {artifact_name}")
    _verify_controlled_evidence_stage(evidence_stage)
    flusher = getattr(task, "flush", None)
    if not callable(flusher):
        raise RuntimeError("controlled baseline task cannot flush artifact uploads")
    flushed = flusher(wait_for_uploads=True)
    if flushed is not None and flushed is not True:
        raise RuntimeError("failed to flush controlled baseline artifact uploads")
    _reload_controlled_evidence_task(task)
    _verify_uploaded_controlled_evidence(task, evidence_stage)


def _execute_controlled_baseline_validation(''',
    ),
    _Anchor(
        "stage_after_controlled_runner",
        r"""    _run_logged(command, cwd=source_root, env=env)

    metrics_path = (work_dir / "metrics.json").resolve(strict=True)
    metrics = _read_json_object(metrics_path)
    evaluation_plan = _read_json_object(
        (work_dir / "evaluation_plan.json").resolve(strict=True)
    )
    expected_evidence = {""",
        r"""    _run_logged(command, cwd=source_root, env=env)

    evidence_stage = _stage_controlled_baseline_evidence(
        work_dir=work_dir,
    )
    evaluation_plan, metrics = _verify_controlled_evidence_stage(evidence_stage)
    expected_evidence = {""",
    ),
    _Anchor(
        "upload_only_sealed_controlled_evidence",
        r"""    _upload_controlled_baseline_artifacts(
        task,
        work_dir=work_dir,
        metrics=metrics,
    )
    task.flush(wait_for_uploads=True)
    return 0
""",
        r"""    _upload_controlled_baseline_artifacts(
        task,
        evidence_stage=evidence_stage,
    )
    return 0
""",
    ),
)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _digest_text(_canonical_json(result))
    return result


def validate_training_seed(value: object) -> int:
    """Return a seed safe for all configured Python/NumPy/PyTorch consumers."""

    if type(value) is not int or not MIN_TRAINING_SEED <= value <= MAX_TRAINING_SEED:
        raise SourceDSeedError(
            f"training seed must be an integer in "
            f"[{MIN_TRAINING_SEED}, {MAX_TRAINING_SEED}]"
        )
    return value


def _source_line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _validate_anchor_set(source_c_text: str) -> list[tuple[int, int, _Anchor]]:
    located: list[tuple[int, int, _Anchor]] = []
    for anchor in _ANCHORS:
        observed = source_c_text.count(anchor.before)
        if observed != 1:
            raise SourceDSeedError(
                f"source-C anchor {anchor.name!r} count mismatch: "
                f"expected 1, got {observed}"
            )
        start = source_c_text.index(anchor.before)
        located.append((start, start + len(anchor.before), anchor))

    located.sort(key=lambda item: item[0])
    previous_end = 0
    previous_name = "<start>"
    for start, end, anchor in located:
        if start < previous_end:
            raise SourceDSeedError(
                f"source-C anchors overlap: {previous_name!r} and {anchor.name!r}"
            )
        previous_end = end
        previous_name = anchor.name
    return located


def _validate_portability_contract(source_text: str, *, context: str) -> None:
    marker_count = source_text.count(PORTABLE_RUNNER_LOAD_MARKER)
    if marker_count != EXPECTED_PORTABLE_RUNNER_LOAD_MARKER_COUNT:
        raise SourceDSeedError(
            f"{context} portable runner-load marker count mismatch: "
            f"expected {EXPECTED_PORTABLE_RUNNER_LOAD_MARKER_COUNT}, "
            f"got {marker_count}"
        )
    legacy_count = source_text.count(LEGACY_RUNNER_LOAD_TARGET_ANCHOR)
    if legacy_count != 0:
        raise SourceDSeedError(
            f"{context} retains {legacy_count} legacy runner-load target anchor(s)"
        )


def _validate_derived_source(source_d_text: str) -> None:
    if len(_ANCHORS) != EXPECTED_DECLARED_REPLACEMENT_COUNT:
        raise SourceDSeedError("source-D declared replacement count drifted")
    observed_staging_anchors = tuple(
        anchor.name for anchor in _ANCHORS if anchor.name in _STAGING_ANCHOR_NAMES
    )
    if observed_staging_anchors != _STAGING_ANCHOR_NAMES:
        raise SourceDSeedError("source-D staging anchor identity/order drifted")
    try:
        compile(source_d_text, "<formal-source-d>", "exec")
    except SyntaxError as error:
        raise SourceDSeedError(f"derived source D does not compile: {error}") from error

    required_fragments = (
        ("TRAINING_OVERLAY_PROTOCOL_SEED = 20250218", 1),
        ("DEFAULT_TRAINING_SEED = TRAINING_OVERLAY_PROTOCOL_SEED", 1),
        ('"--training-seed"', 2),
        ('"training_seed": args.training_seed', 1),
        ('"training_overlay_protocol_seed": overlay_protocol_seed', 1),
        ('"seed": args.training_seed', 1),
        ('"training_index_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED', 1),
        *((f'f"{binding}={{training_seed}}"', 1) for binding in _TRAINING_CFG_BINDINGS),
    )
    for fragment, expected_count in required_fragments:
        if source_d_text.count(fragment) != expected_count:
            raise SourceDSeedError(
                f"derived source-D invariant count mismatch for {fragment!r}"
            )
    for fragment, expected_count in _STAGING_REQUIRED_FRAGMENTS:
        if source_d_text.count(fragment) != expected_count:
            raise SourceDSeedError(
                f"derived source-D staging invariant count mismatch for {fragment!r}"
            )
    for fragment in _STAGING_FORBIDDEN_FRAGMENTS:
        if fragment in source_d_text:
            raise SourceDSeedError(
                f"derived source D retains unsafe evidence upload: {fragment!r}"
            )
    if source_d_text.count("20250218") != 1:
        raise SourceDSeedError(
            "derived source D must retain 20250218 only as the overlay protocol seed"
        )
    forbidden_fragments = (
        '"--seed",\n        "20250218"',
        '"seed": 20250218',
    )
    for fragment in forbidden_fragments:
        if fragment in source_d_text:
            raise SourceDSeedError(
                f"derived source D retains a hard-coded training seed: {fragment!r}"
            )
    _validate_portability_contract(source_d_text, context="derived source D")


def build_source_d(source_c_text: str) -> SourceDBuild:
    """Transform exactly the sealed live source C into explicit-seed source D."""

    if type(source_c_text) is not str:
        raise TypeError("source_c_text must be str")
    source_c_sha256 = _digest_text(source_c_text)
    if source_c_sha256 != EXPECTED_SOURCE_C_SHA256:
        raise SourceDSeedError(
            "source-C SHA-256 mismatch: "
            f"expected {EXPECTED_SOURCE_C_SHA256}, got {source_c_sha256}"
        )
    _validate_portability_contract(source_c_text, context="source C")

    located = _validate_anchor_set(source_c_text)
    output_parts: list[str] = []
    source_replay_parts: list[str] = []
    diff_records: list[dict[str, object]] = []
    unchanged_segments: list[dict[str, object]] = []
    source_cursor = 0
    output_cursor = 0

    for index, (start, end, anchor) in enumerate(located, start=1):
        unchanged = source_c_text[source_cursor:start]
        output_parts.append(unchanged)
        source_replay_parts.append(unchanged)
        if unchanged:
            unchanged_segments.append(
                {
                    "index": len(unchanged_segments) + 1,
                    "source_char_start": source_cursor,
                    "source_char_end": start,
                    "output_char_start": output_cursor,
                    "output_char_end": output_cursor + len(unchanged),
                    "size_bytes": len(unchanged.encode("utf-8")),
                    "sha256": _digest_text(unchanged),
                }
            )
        output_cursor += len(unchanged)

        output_parts.append(anchor.after)
        source_replay_parts.append(anchor.before)
        diff_records.append(
            {
                "index": index,
                "name": anchor.name,
                "expected_count": 1,
                "observed_count": 1,
                "source_char_start": start,
                "source_char_end": end,
                "source_line_start": _source_line(source_c_text, start),
                "source_line_end": _source_line(source_c_text, end),
                "output_char_start": output_cursor,
                "output_char_end": output_cursor + len(anchor.after),
                "before_size_bytes": len(anchor.before.encode("utf-8")),
                "after_size_bytes": len(anchor.after.encode("utf-8")),
                "before_sha256": _digest_text(anchor.before),
                "after_sha256": _digest_text(anchor.after),
            }
        )
        output_cursor += len(anchor.after)
        source_cursor = end

    unchanged = source_c_text[source_cursor:]
    output_parts.append(unchanged)
    source_replay_parts.append(unchanged)
    if unchanged:
        unchanged_segments.append(
            {
                "index": len(unchanged_segments) + 1,
                "source_char_start": source_cursor,
                "source_char_end": len(source_c_text),
                "output_char_start": output_cursor,
                "output_char_end": output_cursor + len(unchanged),
                "size_bytes": len(unchanged.encode("utf-8")),
                "sha256": _digest_text(unchanged),
            }
        )

    source_d_text = "".join(output_parts)
    if "".join(source_replay_parts) != source_c_text:
        raise SourceDSeedError("source-C equivalence replay mismatch")
    _validate_derived_source(source_d_text)
    source_d_sha256 = _digest_text(source_d_text)

    artifact = _sealed(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_d_seed_diff",
            "transformation_id": TRANSFORMATION_ID,
            "source_c": {
                "sha256": source_c_sha256,
                "size_bytes": len(source_c_text.encode("utf-8")),
                "line_count": len(source_c_text.splitlines()),
            },
            "source_d": {
                "sha256": source_d_sha256,
                "size_bytes": len(source_d_text.encode("utf-8")),
                "line_count": len(source_d_text.splitlines()),
            },
            "seed_contract": {
                "training_seed_cli": "--training-seed",
                "default_training_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
                "training_seed_min": MIN_TRAINING_SEED,
                "training_seed_max": MAX_TRAINING_SEED,
                "training_cfg_bindings": list(_TRAINING_CFG_BINDINGS),
                "controlled_baseline_cli": "--seed",
                "run_contract_training_seed_field": "training_seed",
                "training_overlay_protocol_seed": (TRAINING_OVERLAY_PROTOCOL_SEED),
                "run_contract_overlay_seed_field": ("training_overlay_protocol_seed"),
                "overlay_seed_source": (
                    "protocols/dair_v2/training_overlays.json:protocol_seed"
                ),
            },
            "diff": diff_records,
            "equivalence": {
                "only_declared_anchor_replacements": True,
                "declared_replacement_count": len(_ANCHORS),
                "unchanged_segment_count": len(unchanged_segments),
                "unchanged_size_bytes": sum(
                    int(segment["size_bytes"]) for segment in unchanged_segments
                ),
                "unchanged_segments": unchanged_segments,
                "source_c_replay_sha256": _digest_text("".join(source_replay_parts)),
                "source_d_replay_sha256": _digest_text("".join(output_parts)),
                "source_d_compiles": True,
            },
        }
    )
    if source_c_sha256 == LIVE_SOURCE_C_SHA256:
        if source_d_sha256 != EXPECTED_SOURCE_D_SHA256:
            raise SourceDSeedError("live source-D SHA-256 drifted")
        equivalence = artifact.get("equivalence")
        if (
            not isinstance(equivalence, Mapping)
            or equivalence.get("declared_replacement_count")
            != EXPECTED_DECLARED_REPLACEMENT_COUNT
            or equivalence.get("unchanged_segment_count")
            != EXPECTED_UNCHANGED_SEGMENT_COUNT
            or artifact.get("artifact_sha256") != EXPECTED_EQUIVALENCE_ARTIFACT_SHA256
        ):
            raise SourceDSeedError("live source-D equivalence invariants drifted")
    return SourceDBuild(
        source_d_text=source_d_text,
        source_d_sha256=source_d_sha256,
        artifact=artifact,
    )


def verify_source_d(
    source_c_text: str,
    source_d_text: str,
    artifact: Mapping[str, object],
) -> SourceDBuild:
    """Reject any source-D byte or evidence not reproduced by this builder."""

    expected = build_source_d(source_c_text)
    if source_d_text != expected.source_d_text:
        raise SourceDSeedError("source D contains an undeclared diff")
    observed_hash = artifact.get("artifact_sha256")
    if (
        type(observed_hash) is not str
        or len(observed_hash) != 64
        or any(character not in "0123456789abcdef" for character in observed_hash)
    ):
        raise SourceDSeedError("source-D artifact SHA-256 is invalid")
    unhashed = dict(artifact)
    unhashed.pop("artifact_sha256", None)
    try:
        recomputed_hash = _digest_text(_canonical_json(unhashed))
        observed_canonical = _canonical_json(artifact)
        expected_canonical = _canonical_json(expected.artifact)
    except (TypeError, ValueError) as error:
        raise SourceDSeedError("source-D artifact is not canonical JSON") from error
    if recomputed_hash != observed_hash:
        raise SourceDSeedError("source-D artifact SHA-256 mismatch")
    # Canonical JSON distinguishes evidence values that Python equality aliases,
    # notably integer 1 and boolean true.
    if observed_canonical != expected_canonical:
        raise SourceDSeedError("source-D diff/equivalence artifact mismatch")
    return expected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-c", type=Path, required=True)
    parser.add_argument("--source-d", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    return parser


def _write_new(path: Path, content: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_c = args.source_c.resolve(strict=True).read_text(encoding="utf-8")
    result = build_source_d(source_c)
    if args.source_d.resolve(strict=False) == args.artifact.resolve(strict=False):
        raise ValueError("--source-d and --artifact must be different paths")
    _write_new(args.source_d, result.source_d_text)
    _write_new(
        args.artifact,
        json.dumps(result.artifact, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "source_c_sha256": EXPECTED_SOURCE_C_SHA256,
                "source_d_sha256": result.source_d_sha256,
                "artifact_sha256": result.artifact["artifact_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
