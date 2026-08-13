"""MMEngine hook for fail-closed shared-only teacher initialization."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch.distributed as dist
from mmdet3d.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner.checkpoint import CheckpointLoader

from transvision.models.common_teacher_initialization import (
    COMMON_TEACHER_PREFIXES,
    CONTRACT_NAME,
    EXPECTED_COMMON_KEY_COUNT,
    EXPECTED_TEACHER_FUSION_KEY_COUNT,
    EXPECTED_TEACHER_KEY_COUNT,
    CommonTeacherInitializationError,
    assert_tensor_mapping_equal,
    file_sha256,
    prepare_common_teacher_state,
    select_target_common_state,
    select_target_fusion_state,
    tensor_mapping_sha256,
    tensor_mapping_stats,
    validate_common_target,
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_AUDIT_FILENAME = "common_teacher_initialization_audit.json"


def _distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _synchronize_error(local_error: str | None) -> str | None:
    if not (dist.is_available() and dist.is_initialized()):
        return local_error
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_error)
    errors = [
        f"rank {rank}: {error}"
        for rank, error in enumerate(gathered)
        if error is not None
    ]
    return "; ".join(errors) if errors else None


def _atomic_write_audit(path: Path, audit: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CommonTeacherInitializationError(
            f"refusing to overwrite existing initialization audit: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@HOOKS.register_module()
class CommonTeacherInitializationHook(Hook):
    """Load only four shared prefixes from one pinned clean teacher.

    The hook executes after MMEngine has initialized model weights and before
    the first training batch. Runner-level ``load_from`` and resume are
    forbidden so method-specific fusion cannot have been populated by an
    earlier non-strict checkpoint load.
    """

    priority = "HIGHEST"

    def __init__(
        self,
        checkpoint: str,
        expected_sha256: str,
        *,
        expected_teacher_keys: int = EXPECTED_TEACHER_KEY_COUNT,
        expected_common_keys: int = EXPECTED_COMMON_KEY_COUNT,
        expected_teacher_fusion_keys: int = EXPECTED_TEACHER_FUSION_KEY_COUNT,
    ) -> None:
        if type(checkpoint) is not str or not checkpoint:
            raise ValueError("checkpoint must be a non-empty local path")
        if (
            type(expected_sha256) is not str
            or _SHA256_PATTERN.fullmatch(expected_sha256) is None
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256")
        expected_counts = (
            expected_teacher_keys,
            expected_common_keys,
            expected_teacher_fusion_keys,
        )
        if any(type(value) is not int or value <= 0 for value in expected_counts):
            raise ValueError("expected checkpoint key counts must be positive integers")

        self.checkpoint = checkpoint
        self.expected_sha256 = expected_sha256
        self.expected_teacher_keys = expected_teacher_keys
        self.expected_common_keys = expected_common_keys
        self.expected_teacher_fusion_keys = expected_teacher_fusion_keys
        self._completed = False

    def _resolve_checkpoint(self) -> tuple[Path, str]:
        try:
            checkpoint = Path(self.checkpoint).expanduser().resolve(strict=True)
        except OSError as error:
            raise CommonTeacherInitializationError(
                f"unable to resolve common-init checkpoint: {error}"
            ) from error
        if not checkpoint.is_file():
            raise CommonTeacherInitializationError(
                f"common-init checkpoint is not a file: {checkpoint}"
            )
        if checkpoint.stat().st_size <= 0:
            raise CommonTeacherInitializationError(
                f"common-init checkpoint is empty: {checkpoint}"
            )
        observed_sha256 = file_sha256(checkpoint)
        if observed_sha256 != self.expected_sha256:
            raise CommonTeacherInitializationError(
                "common-init checkpoint SHA-256 mismatch: "
                f"expected {self.expected_sha256}, got {observed_sha256}"
            )
        return checkpoint, observed_sha256

    def _initialize(self, runner: Any) -> dict[str, Any]:
        if self._completed:
            raise CommonTeacherInitializationError(
                "common teacher initialization hook ran more than once"
            )
        if bool(getattr(runner, "_resume", False)):
            raise CommonTeacherInitializationError(
                "shared-only teacher initialization forbids resume"
            )
        load_from = getattr(runner, "_load_from", None)
        if load_from is not None:
            raise CommonTeacherInitializationError(
                "shared-only teacher initialization requires runner.load_from=None; "
                f"got {load_from!r}"
            )

        checkpoint_path, checkpoint_sha256 = self._resolve_checkpoint()
        checkpoint = CheckpointLoader.load_checkpoint(
            str(checkpoint_path),
            map_location="cpu",
            logger=runner.logger,
        )
        source, common = prepare_common_teacher_state(
            checkpoint,
            expected_teacher_keys=self.expected_teacher_keys,
            expected_common_keys=self.expected_common_keys,
            expected_teacher_fusion_keys=self.expected_teacher_fusion_keys,
        )

        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        target_before = model.state_dict()
        target_contract = validate_common_target(source, common, target_before)
        fusion_before = select_target_fusion_state(target_before)
        fusion_sha256_before = tensor_mapping_sha256(fusion_before)

        incompatible = model.load_state_dict(common, strict=False)
        if incompatible.unexpected_keys:
            raise CommonTeacherInitializationError(
                "filtered common initialization produced unexpected keys: "
                + ", ".join(incompatible.unexpected_keys[:5])
            )

        target_after = model.state_dict()
        target_contract_after = validate_common_target(source, common, target_after)
        assert_tensor_mapping_equal(
            common,
            select_target_common_state(target_after),
            context="loaded shared initialization",
        )
        fusion_after = select_target_fusion_state(target_after)
        fusion_sha256_after = tensor_mapping_sha256(fusion_after)
        if fusion_sha256_after != fusion_sha256_before:
            raise CommonTeacherInitializationError(
                "method-specific fusion changed during shared-only initialization"
            )
        if target_contract_after != target_contract:
            raise CommonTeacherInitializationError(
                "target state-dict contract changed during shared-only initialization"
            )

        self._completed = True
        return {
            "schema_version": 1,
            "contract": CONTRACT_NAME,
            "result": "pass",
            "checkpoint": {
                "path": str(checkpoint_path),
                "filename": checkpoint_path.name,
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": checkpoint_sha256,
                "expected_sha256": self.expected_sha256,
            },
            "source": {
                **tensor_mapping_stats(source),
                "expected_keys": self.expected_teacher_keys,
                "common_keys": len(common),
                "expected_common_keys": self.expected_common_keys,
                "fusion_keys": len(source) - len(common),
                "expected_fusion_keys": self.expected_teacher_fusion_keys,
                "state_sha256": tensor_mapping_sha256(source),
            },
            "shared_initialization": {
                "prefixes": list(COMMON_TEACHER_PREFIXES),
                **tensor_mapping_stats(common),
                "expected_keys": self.expected_common_keys,
                "state_sha256": tensor_mapping_sha256(common),
                "shape_dtype_verified": True,
                "exact_tensor_equality_verified": True,
            },
            "method_specific_fusion": {
                **tensor_mapping_stats(fusion_after),
                "sha256_before": fusion_sha256_before,
                "sha256_after": fusion_sha256_after,
                "unchanged": True,
            },
            "target": {
                "model_type": type(model).__name__,
                **target_contract_after,
            },
        }

    def before_train(self, runner: Any) -> None:
        audit: dict[str, Any] | None = None
        local_exception: Exception | None = None
        try:
            audit = self._initialize(runner)
            local_error = None
        except Exception as error:  # synchronize every fail-closed path
            local_exception = error
            local_error = f"{type(error).__name__}: {error}"

        synchronized_error = _synchronize_error(local_error)
        if synchronized_error is not None:
            raise CommonTeacherInitializationError(
                "shared-only teacher initialization failed: " + synchronized_error
            ) from local_exception

        write_error: str | None = None
        if _distributed_rank() == 0:
            try:
                if audit is None:
                    raise AssertionError("successful initialization produced no audit")
                work_dir = Path(runner.work_dir).expanduser().resolve()
                _atomic_write_audit(work_dir / _AUDIT_FILENAME, audit)
            except Exception as error:
                write_error = f"{type(error).__name__}: {error}"
        synchronized_write_error = _synchronize_error(write_error)
        if synchronized_write_error is not None:
            raise CommonTeacherInitializationError(
                "failed to publish common teacher initialization audit: "
                + synchronized_write_error
            )
        runner.logger.info(
            "Verified shared-only clean-teacher initialization: %d tensors",
            self.expected_common_keys,
        )


__all__ = ("CommonTeacherInitializationHook",)
