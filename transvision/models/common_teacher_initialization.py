"""Fail-closed helpers for shared-only clean-teacher initialization.

The clean cooperative teacher and every controlled method share exactly the
two encoders, the detection projection, and the detection head.  These helpers
define the state-dict contract without depending on MMEngine so that the
contract can be tested with synthetic tensors.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


COMMON_TEACHER_PREFIXES = (
    "lidar_encoder.",
    "camera_encoder.",
    "bbox_head.",
    "detection_projection.",
)
TEACHER_FUSION_PREFIX = "resilient_fusion."
TARGET_FUSION_PREFIXES = (TEACHER_FUSION_PREFIX, "fusion.")
NESTED_TEACHER_PREFIX = "teacher.teacher."

EXPECTED_TEACHER_KEY_COUNT = 617
EXPECTED_COMMON_KEY_COUNT = 468
EXPECTED_TEACHER_FUSION_KEY_COUNT = 149
CONTRACT_NAME = "shared-only-clean-teacher-initialization-v1"
EMPTY_TENSOR_MAPPING_SHA256 = hashlib.sha256(b"").hexdigest()


class CommonTeacherInitializationError(RuntimeError):
    """Raised when a checkpoint or target model violates the contract."""


def _matches_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return key.startswith(prefixes)


def _require_tensor_mapping(
    state_dict: Mapping[str, Any],
    *,
    context: str,
) -> OrderedDict[str, Tensor]:
    if not state_dict:
        raise CommonTeacherInitializationError(f"{context} is empty")

    tensors: OrderedDict[str, Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if type(key) is not str or not key:
            raise CommonTeacherInitializationError(
                f"{context} contains a non-string or empty key"
            )
        if not isinstance(value, Tensor):
            raise CommonTeacherInitializationError(
                f"{context}[{key!r}] is not a tensor"
            )
        tensors[key] = value
    return tensors


def normalize_checkpoint_state_dict(
    checkpoint: Mapping[str, Any],
) -> OrderedDict[str, Tensor]:
    """Extract and normalize a checkpoint state dict like MMEngine.

    Only a literal leading ``module.`` is accepted.  Normalization collisions
    are rejected rather than silently choosing one tensor.
    """

    if not isinstance(checkpoint, Mapping):
        raise CommonTeacherInitializationError("checkpoint must be a mapping")
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise CommonTeacherInitializationError(
            "checkpoint state_dict must be a mapping"
        )
    tensors = _require_tensor_mapping(raw_state, context="checkpoint state_dict")

    normalized: OrderedDict[str, Tensor] = OrderedDict()
    for raw_key, value in tensors.items():
        key = raw_key[len("module.") :] if raw_key.startswith("module.") else raw_key
        if not key:
            raise CommonTeacherInitializationError(
                "checkpoint contains an empty key after module. normalization"
            )
        if key in normalized:
            raise CommonTeacherInitializationError(
                f"checkpoint key collision after normalization: {key!r}"
            )
        normalized[key] = value
    return normalized


def _assert_finite(state_dict: Mapping[str, Tensor], *, context: str) -> None:
    for key, value in state_dict.items():
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise CommonTeacherInitializationError(
                f"{context}[{key!r}] contains a non-finite value"
            )


def prepare_common_teacher_state(
    checkpoint: Mapping[str, Any],
    *,
    expected_teacher_keys: int = EXPECTED_TEACHER_KEY_COUNT,
    expected_common_keys: int = EXPECTED_COMMON_KEY_COUNT,
    expected_teacher_fusion_keys: int = EXPECTED_TEACHER_FUSION_KEY_COUNT,
) -> tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]:
    """Validate a clean-teacher checkpoint and select only shared tensors."""

    source = normalize_checkpoint_state_dict(checkpoint)
    common = OrderedDict(
        (key, value)
        for key, value in source.items()
        if _matches_prefix(key, COMMON_TEACHER_PREFIXES)
    )
    fusion = OrderedDict(
        (key, value)
        for key, value in source.items()
        if key.startswith(TEACHER_FUSION_PREFIX)
    )
    allowed = set(common) | set(fusion)
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise CommonTeacherInitializationError(
            "clean-teacher checkpoint contains unsupported prefixes: "
            + ", ".join(unknown[:5])
        )
    observed_counts = (len(source), len(common), len(fusion))
    expected_counts = (
        expected_teacher_keys,
        expected_common_keys,
        expected_teacher_fusion_keys,
    )
    if observed_counts != expected_counts:
        raise CommonTeacherInitializationError(
            "clean-teacher checkpoint key-count mismatch: "
            f"expected total/common/fusion={expected_counts}, "
            f"got {observed_counts}"
        )
    _assert_finite(source, context="clean-teacher checkpoint")
    return source, common


def _assert_compatible_tensor(
    source: Tensor,
    target: Tensor,
    *,
    key: str,
) -> None:
    if tuple(source.shape) != tuple(target.shape):
        raise CommonTeacherInitializationError(
            f"shared tensor shape mismatch for {key!r}: "
            f"source={tuple(source.shape)}, target={tuple(target.shape)}"
        )
    if source.dtype != target.dtype:
        raise CommonTeacherInitializationError(
            f"shared tensor dtype mismatch for {key!r}: "
            f"source={source.dtype}, target={target.dtype}"
        )


def validate_common_target(
    source: Mapping[str, Tensor],
    common: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Validate target keys and nested-teacher state before shared loading.

    A method-specific fusion module may legitimately have no parameters or
    buffers. Such a target has zero fusion keys; it is still protected by the
    canonical empty-mapping digest before and after shared initialization.
    """

    target_tensors = _require_tensor_mapping(target, context="target state_dict")
    target_common = OrderedDict(
        (key, value)
        for key, value in target_tensors.items()
        if _matches_prefix(key, COMMON_TEACHER_PREFIXES)
    )
    target_fusion = OrderedDict(
        (key, value)
        for key, value in target_tensors.items()
        if _matches_prefix(key, TARGET_FUSION_PREFIXES)
    )
    nested = OrderedDict(
        (key[len(NESTED_TEACHER_PREFIX) :], value)
        for key, value in target_tensors.items()
        if key.startswith(NESTED_TEACHER_PREFIX)
    )

    classified = (
        set(target_common)
        | set(target_fusion)
        | {f"{NESTED_TEACHER_PREFIX}{key}" for key in nested}
    )
    unknown = sorted(set(target_tensors) - classified)
    if unknown:
        raise CommonTeacherInitializationError(
            "target state_dict contains unsupported prefixes: " + ", ".join(unknown[:5])
        )

    missing = sorted(set(common) - set(target_common))
    extra = sorted(set(target_common) - set(common))
    if missing or extra:
        raise CommonTeacherInitializationError(
            f"target shared-key mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    for key, source_value in common.items():
        _assert_compatible_tensor(source_value, target_common[key], key=key)

    nested_verified = False
    if nested:
        missing_nested = sorted(set(source) - set(nested))
        extra_nested = sorted(set(nested) - set(source))
        if missing_nested or extra_nested:
            raise CommonTeacherInitializationError(
                "nested frozen-teacher key mismatch: "
                f"missing={missing_nested[:5]}, extra={extra_nested[:5]}"
            )
        assert_tensor_mapping_equal(
            source,
            nested,
            context="nested frozen teacher",
        )
        nested_verified = True

    return {
        "target_key_count": len(target_tensors),
        "target_common_key_count": len(target_common),
        "target_fusion_key_count": len(target_fusion),
        "nested_teacher_present": bool(nested),
        "nested_teacher_key_count": len(nested),
        "nested_teacher_full_equality_verified": nested_verified,
    }


def assert_tensor_mapping_equal(
    expected: Mapping[str, Tensor],
    observed: Mapping[str, Tensor],
    *,
    context: str,
) -> None:
    """Require exact key, shape, dtype, and tensor-value equality."""

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise CommonTeacherInitializationError(
            f"{context} key mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    for key, expected_value in expected.items():
        observed_value = observed[key]
        _assert_compatible_tensor(expected_value, observed_value, key=key)
        if not torch.equal(
            expected_value.detach().cpu(),
            observed_value.detach().cpu(),
        ):
            raise CommonTeacherInitializationError(
                f"{context} tensor value mismatch for {key!r}"
            )


def select_target_common_state(
    target: Mapping[str, Tensor],
) -> OrderedDict[str, Tensor]:
    """Return the target's four shared-prefix tensors."""

    return OrderedDict(
        (key, value)
        for key, value in target.items()
        if _matches_prefix(key, COMMON_TEACHER_PREFIXES)
    )


def select_target_fusion_state(
    target: Mapping[str, Tensor],
) -> OrderedDict[str, Tensor]:
    """Return only outer method-specific fusion tensors."""

    return OrderedDict(
        (key, value)
        for key, value in target.items()
        if _matches_prefix(key, TARGET_FUSION_PREFIXES)
    )


def tensor_mapping_stats(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    """Return deterministic tensor counts used in the audit artifact."""

    return {
        "keys": len(state_dict),
        "numel": sum(value.numel() for value in state_dict.values()),
        "bytes": sum(
            value.numel() * value.element_size() for value in state_dict.values()
        ),
    }


def tensor_mapping_sha256(
    state_dict: Mapping[str, Tensor],
    *,
    keys: Iterable[str] | None = None,
) -> str:
    """Hash keys, tensor metadata, and logical tensor bytes deterministically."""

    selected = sorted(state_dict if keys is None else keys)
    if not selected:
        return EMPTY_TENSOR_MAPPING_SHA256
    digest = hashlib.sha256()
    for key in selected:
        if key not in state_dict:
            raise CommonTeacherInitializationError(
                f"cannot hash missing tensor key {key!r}"
            )
        value = state_dict[key]
        cpu = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(cpu.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(cpu.numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one local checkpoint without reading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "COMMON_TEACHER_PREFIXES",
    "CONTRACT_NAME",
    "EMPTY_TENSOR_MAPPING_SHA256",
    "EXPECTED_COMMON_KEY_COUNT",
    "EXPECTED_TEACHER_FUSION_KEY_COUNT",
    "EXPECTED_TEACHER_KEY_COUNT",
    "NESTED_TEACHER_PREFIX",
    "TARGET_FUSION_PREFIXES",
    "TEACHER_FUSION_PREFIX",
    "CommonTeacherInitializationError",
    "assert_tensor_mapping_equal",
    "file_sha256",
    "normalize_checkpoint_state_dict",
    "prepare_common_teacher_state",
    "select_target_common_state",
    "select_target_fusion_state",
    "tensor_mapping_sha256",
    "tensor_mapping_stats",
    "validate_common_target",
)
