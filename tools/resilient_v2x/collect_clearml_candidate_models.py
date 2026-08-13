#!/usr/bin/env python3
"""Audit and incrementally archive completed single-seed SOTA candidates.

The default CLI is read-only.  Model bytes are streamed directly from the
ClearML files server only after an explicit collection token is supplied; the
ClearML model cache is never used.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    from tools.resilient_v2x import collect_clearml_formal_models as formal
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.resilient_v2x import collect_clearml_formal_models as formal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/trained_models/completed-live"
DOCUMENT_TYPE = "resilient_v2x_completed_candidate_model_archive"
EXECUTE_TOKEN = "ARCHIVE_VERIFIED_COMPLETED_CANDIDATES"
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
TRAINING_SEED = 20_250_218
GLOBAL_BATCH_SIZE = 8
GPU_COUNT = 4
MAX_EPOCHS = 50
VAL_INTERVAL = 10
PRECISION = "FP32"
TEACHER_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
TEACHER_MODEL_ID = "d962f6bae8474260b54e170a7a5f0418"
TEACHER_MODEL_NAME = "ResilientV2X clean teacher"
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
TEACHER_CHECKPOINT_BYTES = 146_189_677
PREDECESSOR_TASK_ID = "21368e8260cc4e5392fe2dbdf116e36f"
FINAL_ARTIFACT = "final_checkpoint_contract"
BEST_ARTIFACT = "best_checkpoint_contract"
RUN_ARTIFACT = "run_contract"
TEACHER_AUDIT_ARTIFACT = "common_teacher_initialization_audit"
CLEAN_SELECTION_PROTOCOL = "DAIR-CLEAN-PAIR1789-v1"
CLEAN_SELECTION_METRIC = "resilient_v2x/car_bev_ap_r40_0.70"
MAX_JSON_ARTIFACT_BYTES = 1024 * 1024
_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_SUBJECT = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class SourceIdentity:
    dataset_id: str
    archive_name: str
    archive_size_bytes: int
    archive_sha256: str
    tree_sha256: str
    file_count: int
    source_bytes: int
    inventory_size_bytes: int
    inventory_sha256: str
    archive_path: Path
    inventory_path: Path


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    subject: str
    task_id: str
    task_name: str
    config_path: str
    config_sha256: str
    source: SourceIdentity
    parent_task_id: str = PREDECESSOR_TASK_ID
    require_diagnostic_best: bool = True


_E123_SOURCE = SourceIdentity(
    dataset_id="c9ca3075434e44e29951e4aca36e9046",
    archive_name="resilient-v2x-source-8a22d6d600a5.tar.zst",
    archive_size_bytes=1_222_568,
    archive_sha256=("a249546f16f236d42c8346a33b137dea27c037bf0df73c643da4132c8e589557"),
    tree_sha256=("8a22d6d600a52117d01cfde5fa5f20fa1de269ec9c9ba040228f5e8e3a5e8767"),
    file_count=634,
    source_bytes=8_927_627,
    inventory_size_bytes=101_714,
    inventory_sha256=(
        "5bd96277989838760f3a2f9b6f3d86a4ce8a0ef79d5a21d366540b4fa7c7604f"
    ),
    archive_path=(
        PROJECT_ROOT
        / "artifacts/resilient_v2x/sota_candidate_source_e1_e2_e3"
        / "resilient-v2x-source-8a22d6d600a5.tar.zst"
    ),
    inventory_path=(
        PROJECT_ROOT
        / "artifacts/resilient_v2x/sota_candidate_source_e1_e2_e3"
        / "source-inventory.json"
    ),
)
_P0_SOURCE = SourceIdentity(
    dataset_id="ca2ef9dd8a984df6b05693fb02e89f34",
    archive_name="resilient-v2x-source-bff6f84c0e49.tar.zst",
    archive_size_bytes=1_223_924,
    archive_sha256=("8a33214c68b956730da1ab6664a008a50f8a6163cd20f299ac21f2b9158536c3"),
    tree_sha256=("bff6f84c0e4989d07c4061b8b0e302a5705accd6688d3a175ed9e3e24b5d259d"),
    file_count=635,
    source_bytes=8_928_152,
    inventory_size_bytes=101_900,
    inventory_sha256=(
        "fe9216c59b7cadf39593abbe2234615dd4d9f0fc2881af0d2c647cc29394849f"
    ),
    archive_path=(
        PROJECT_ROOT
        / "artifacts/resilient_v2x/sota_round2_p0"
        / "resilient-v2x-source-bff6f84c0e49.tar.zst"
    ),
    inventory_path=(
        PROJECT_ROOT / "artifacts/resilient_v2x/sota_round2_p0/source-inventory.json"
    ),
)
_P2_SOURCE = SourceIdentity(
    dataset_id="858238049cad4d13918384bb8faec630",
    archive_name="resilient-v2x-source-25d1a9a1b67f.tar.zst",
    archive_size_bytes=1_223_663,
    archive_sha256=("54e0cf371a6c8b68b041b84f8568efa72c432f0d38c9dfce89f005b2605ee1b3"),
    tree_sha256=("25d1a9a1b67f525a02220c34739855a83d39706cc78e9f2bbaf882b47c438331"),
    file_count=636,
    source_bytes=8_928_652,
    inventory_size_bytes=102_093,
    inventory_sha256=(
        "5e05b1f659f145fb1c6db5d55da7738bb63f6c91d3dbca476ae8b19684513787"
    ),
    archive_path=(
        PROJECT_ROOT
        / "artifacts/resilient_v2x/sota_round2_p2"
        / "resilient-v2x-source-25d1a9a1b67f.tar.zst"
    ),
    inventory_path=(
        PROJECT_ROOT / "artifacts/resilient_v2x/sota_round2_p2/source-inventory.json"
    ),
)

CANDIDATE_SPECS = (
    CandidateSpec(
        label="E1",
        subject="support_residual_linear",
        task_id="f0c3082f3aa34a81805903e0ffdc8610",
        task_name=("ResilientV2X fastlane E1 support_residual_linear [d6df8cc0ce59]"),
        config_path="configs/resilient_v2x/improvements/support_residual_linear.py",
        config_sha256=(
            "a105124cf16a693a8c6fb176e07720abe0b05a6ff9d187b076b83d46c30bb4c0"
        ),
        source=_E123_SOURCE,
    ),
    CandidateSpec(
        label="E2",
        subject="no_reliability_linear",
        task_id="969c8fce6d24446299561772b3955274",
        task_name="ResilientV2X fastlane E2 no_reliability_linear [d6df8cc0ce59]",
        config_path="configs/resilient_v2x/improvements/no_reliability_linear.py",
        config_sha256=(
            "a42f6b28bf1678f663458ccc6f98a409e2483d6786423aa50b2c72e68b721e5d"
        ),
        source=_E123_SOURCE,
    ),
    CandidateSpec(
        label="E3",
        subject="support_residual_no_reliability",
        task_id="dc037315c0684c3d854a2fd7c19a2a2f",
        task_name=(
            "ResilientV2X fastlane E3 support_residual_no_reliability [d6df8cc0ce59]"
        ),
        config_path=(
            "configs/resilient_v2x/improvements/support_residual_no_reliability.py"
        ),
        config_sha256=(
            "86f124aa6ae542891575c26d68560f6b4835b9ebe96516fed40074dfd5af1efa"
        ),
        source=_E123_SOURCE,
    ),
    CandidateSpec(
        label="P0",
        subject="support_residual_no_reliability_linear",
        task_id="8883c51ced4f4951a45edbaefe6342d4",
        task_name="ResilientV2X round2 P0 [ebd8421be6fc]",
        config_path=(
            "configs/resilient_v2x/improvements/"
            "support_residual_no_reliability_linear.py"
        ),
        config_sha256=(
            "e8208673e2633231c5a60d3783e0735aa3fe0ccaede1a386f4e337597414d1fc"
        ),
        source=_P0_SOURCE,
    ),
    CandidateSpec(
        label="P2",
        subject="support_residual_no_reliability_linear_bbox25",
        task_id="f5d3820b4cdf416183c8f1fee566abe3",
        task_name="ResilientV2X round2 P2 bbox2.5 [4ced839db7fb]",
        config_path=(
            "configs/resilient_v2x/improvements/"
            "support_residual_no_reliability_linear_bbox25.py"
        ),
        config_sha256=(
            "eb9e47fde3709b8e04a2b827ea458834fc8db31584369c4c84e2c28403359a88"
        ),
        source=_P2_SOURCE,
    ),
)
SPEC_BY_LABEL = {spec.label: spec for spec in CANDIDATE_SPECS}


class CandidateNotReady(RuntimeError):
    """A pinned candidate exists but has not completed training."""

    def __init__(self, *, label: str, task_id: str, status: str) -> None:
        super().__init__(f"{label} task {task_id} is not completed: {status}")
        self.label = label
        self.task_id = task_id
        self.status = status


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if _SHA256.fullmatch(result) is None:
        raise RuntimeError(f"{context} must be a lowercase SHA-256")
    return result


def _clearml_id(value: object, context: str) -> str:
    result = str(value or "")
    if _ID.fullmatch(result) is None:
        raise RuntimeError(f"{context} must be a lowercase 32-hex ClearML ID")
    return result


def _positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{context} must be a positive integer")
    return value


def _status(task: object) -> str:
    value = getattr(task, "status", "")
    return str(getattr(value, "value", value) or "").strip().lower()


def _reload(task: object) -> None:
    method = getattr(task, "reload", None)
    if callable(method):
        method()


def _parameters(task: object) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError("candidate task cannot expose parameters")
    result = getter()
    if not isinstance(result, Mapping):
        raise RuntimeError("candidate task parameters are invalid")
    return dict(result)


def _task_parent(task: object) -> str:
    value = getattr(task, "parent", None)
    if value is None:
        data = getattr(task, "data", None)
        value = getattr(data, "parent", None)
    return str(value or "")


def _equivalent(value: object, expected: object) -> bool:
    if type(expected) is bool:
        if type(value) is bool:
            return value is expected
        return str(value).strip().casefold() == str(expected).casefold()
    if type(expected) is int:
        try:
            return int(str(value)) == expected
        except (TypeError, ValueError):
            return False
    return str(value or "") == str(expected)


def _strict_json(raw: bytes, *, context: str) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} is not a JSON object")
    return dict(value)


def _regular_file(path: Path, *, size: int, sha256: str, context: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is not a regular file")
    if path.stat().st_size != size or _sha256_path(path) != sha256:
        raise RuntimeError(f"{context} identity mismatch")


def _validate_local_source(source: SourceIdentity) -> dict[str, object]:
    _clearml_id(source.dataset_id, "source Dataset")
    _sha256(source.archive_sha256, "source archive")
    _sha256(source.tree_sha256, "source tree")
    _sha256(source.inventory_sha256, "source inventory")
    _regular_file(
        source.archive_path,
        size=source.archive_size_bytes,
        sha256=source.archive_sha256,
        context="sealed source archive",
    )
    _regular_file(
        source.inventory_path,
        size=source.inventory_size_bytes,
        sha256=source.inventory_sha256,
        context="sealed source inventory",
    )
    inventory = _strict_json(
        source.inventory_path.read_bytes(), context="sealed source inventory"
    )
    expected = {
        "schema_version": 1,
        "tree_sha256": source.tree_sha256,
        "file_count": source.file_count,
        "source_bytes": source.source_bytes,
    }
    for key, value in expected.items():
        if inventory.get(key) != value:
            raise RuntimeError(f"sealed source inventory drifted: {key}")
    files = inventory.get("files")
    if not isinstance(files, list) or len(files) != source.file_count:
        raise RuntimeError("sealed source inventory file count mismatch")
    paths: set[str] = set()
    observed_bytes = 0
    for item in files:
        if not isinstance(item, Mapping):
            raise RuntimeError("sealed source inventory has invalid file entry")
        path = str(item.get("path") or "")
        parts = Path(path).parts
        if not path or Path(path).is_absolute() or ".." in parts or path in paths:
            raise RuntimeError("sealed source inventory has unsafe/duplicate path")
        paths.add(path)
        _sha256(item.get("sha256"), f"source file {path}")
        size = item.get("size_bytes", item.get("size"))
        if type(size) is not int or size < 0:
            raise RuntimeError(f"source file {path} has invalid size")
        observed_bytes += size
    if observed_bytes != source.source_bytes:
        raise RuntimeError("sealed source inventory byte total mismatch")
    return {
        "dataset_id": source.dataset_id,
        "archive_name": source.archive_name,
        "archive_size_bytes": source.archive_size_bytes,
        "archive_sha256": source.archive_sha256,
        "tree_sha256": source.tree_sha256,
        "file_count": source.file_count,
        "source_bytes": source.source_bytes,
        "inventory_size_bytes": source.inventory_size_bytes,
        "inventory_sha256": source.inventory_sha256,
        "archive_path": str(source.archive_path),
        "inventory_path": str(source.inventory_path),
    }


def _artifact_reference(task: object, name: str) -> tuple[object, dict[str, object]]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"candidate task is missing required artifact {name!r}")
    artifact = artifacts[name]
    expected_filename = f"{name}.json"
    url = formal._fileserver_url(
        getattr(artifact, "url", None),
        expected_filename=expected_filename,
        context=f"{name} artifact",
    )
    size = _positive_int(getattr(artifact, "size", None), f"{name} artifact bytes")
    if size > MAX_JSON_ARTIFACT_BYTES:
        raise RuntimeError(f"{name} artifact exceeds the JSON safety limit")
    sha256 = _sha256(getattr(artifact, "hash", None), f"{name} artifact")
    return artifact, {
        "name": name,
        "url": url,
        "size_bytes": size,
        "sha256": sha256,
    }


ArtifactBytesLoader = Callable[
    [object, Mapping[str, object], Callable[[], Mapping[str, str]], Callable[[], None]],
    bytes,
]


def _load_artifact_bytes(
    artifact: object,
    reference: Mapping[str, object],
    auth_header_provider: Callable[[], Mapping[str, str]],
    auth_refresher: Callable[[], None],
) -> bytes:
    del artifact
    response = formal._authenticated_model_response(
        url=str(reference["url"]),
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    raw = bytearray()
    try:
        final_url = formal._fileserver_url(
            getattr(response, "url", None),
            expected_filename=f"{reference['name']}.json",
            context=f"{reference['name']} response",
        )
        if final_url != reference["url"]:
            raise RuntimeError("artifact response URL drifted")
        headers = getattr(response, "headers", {})
        declared = headers.get("Content-Length") if headers else None
        if declared is not None:
            try:
                declared_size = int(declared)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "artifact response Content-Length is invalid"
                ) from error
            if declared_size != reference["size_bytes"]:
                raise RuntimeError("artifact response Content-Length mismatch")
        iterator = getattr(response, "iter_content", None)
        if not callable(iterator):
            raise RuntimeError("artifact response cannot expose content")
        for block in iterator(chunk_size=64 * 1024):
            if not block:
                continue
            if not isinstance(block, bytes):
                raise RuntimeError("artifact response returned non-bytes")
            raw.extend(block)
            if len(raw) > int(reference["size_bytes"]):
                raise RuntimeError("artifact response exceeds contracted size")
    finally:
        closer = getattr(response, "close", None)
        if callable(closer):
            closer()
    return bytes(raw)


def _artifact_payload(
    task: object,
    name: str,
    *,
    auth_header_provider: Callable[[], Mapping[str, str]],
    auth_refresher: Callable[[], None],
    artifact_bytes_loader: ArtifactBytesLoader,
) -> tuple[dict[str, object], dict[str, object]]:
    artifact, reference = _artifact_reference(task, name)
    raw = artifact_bytes_loader(
        artifact, reference, auth_header_provider, auth_refresher
    )
    if not isinstance(raw, bytes):
        raise RuntimeError(f"{name} artifact loader returned non-bytes")
    if len(raw) != reference["size_bytes"]:
        raise RuntimeError(f"{name} artifact size mismatch")
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise RuntimeError(f"{name} artifact SHA-256 mismatch")
    return _strict_json(raw, context=f"{name} artifact"), reference


def _validate_task_identity(
    task: object, spec: CandidateSpec
) -> tuple[str, dict[str, object]]:
    if _clearml_id(getattr(task, "id", None), f"{spec.label} resolved task") != (
        spec.task_id
    ):
        raise RuntimeError(f"{spec.label} resolved task ID drifted")
    if str(getattr(task, "name", "") or "") != spec.task_name:
        raise RuntimeError(f"{spec.label} task name drifted")
    if _task_parent(task) != spec.parent_task_id:
        raise RuntimeError(f"{spec.label} task parent drifted")
    parameters = _parameters(task)
    expected = {
        "Args/experiment_from_task": spec.subject,
        "Args/source_dataset_id": spec.source.dataset_id,
        "Args/source_archive_name": spec.source.archive_name,
        "Args/source_archive_bytes": spec.source.archive_size_bytes,
        "Args/source_archive_sha256": spec.source.archive_sha256,
        "Args/training_dataset_id": TRAINING_DATASET_ID,
        "Args/training_seed": TRAINING_SEED,
        "Args/gpus": GPU_COUNT,
        "Args/max_epochs": MAX_EPOCHS,
        "Args/amp": False,
        "Args/teacher_task_id": TEACHER_TASK_ID,
        "Args/teacher_model_id": TEACHER_MODEL_ID,
        "Args/teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
    }
    for key, value in expected.items():
        if not _equivalent(parameters.get(key), value):
            raise RuntimeError(f"{spec.label} task parameter drifted: {key}")
    return _status(task), {key: parameters[key] for key in expected}


def _validate_run_contract(
    payload: Mapping[str, object], spec: CandidateSpec
) -> dict[str, object]:
    expected = {
        "schema_version": 1,
        "task_id": spec.task_id,
        "experiment": spec.subject,
        "experiment_kind": "sota_candidate",
        "source_dataset_id": spec.source.dataset_id,
        "training_dataset_id": TRAINING_DATASET_ID,
        "seed": TRAINING_SEED,
        "gpus": GPU_COUNT,
        "ddp_processes": GPU_COUNT,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "train_batch_size_per_gpu": 2,
        "max_epochs": MAX_EPOCHS,
        "val_interval": VAL_INTERVAL,
        "amp": False,
        "precision": PRECISION,
        "auto_scale_lr": False,
        "condition_evaluation": False,
        "predecessor_task_id": spec.parent_task_id,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{spec.label} run contract drifted: {key}")
    source = payload.get("source_archive")
    if source != {
        "name": spec.source.archive_name,
        "sha256": spec.source.archive_sha256,
        "size_bytes": spec.source.archive_size_bytes,
    }:
        raise RuntimeError(f"{spec.label} run contract source archive drifted")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{spec.label} run contract has no config identity")
    config_expected = {
        "declared": spec.config_path,
        "config_sha256": spec.config_sha256,
        "resolved_config_sha256": spec.config_sha256,
    }
    for key, value in config_expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"{spec.label} run contract config drifted: {key}")
    teacher = payload.get("teacher")
    if not isinstance(teacher, Mapping):
        raise RuntimeError(f"{spec.label} run contract has no teacher identity")
    teacher_expected = {
        "task_id": TEACHER_TASK_ID,
        "model_id": TEACHER_MODEL_ID,
        "name": TEACHER_MODEL_NAME,
        "expected_sha256": TEACHER_CHECKPOINT_SHA256,
        "sha256": TEACHER_CHECKPOINT_SHA256,
        "size_bytes": TEACHER_CHECKPOINT_BYTES,
    }
    for key, value in teacher_expected.items():
        if teacher.get(key) != value:
            raise RuntimeError(f"{spec.label} run contract teacher drifted: {key}")
    initialization = payload.get("common_teacher_initialization")
    if not isinstance(initialization, Mapping):
        raise RuntimeError(f"{spec.label} run contract has no teacher init contract")
    if (
        initialization.get("audit_artifact_name") != TEACHER_AUDIT_ARTIFACT
        or initialization.get("contract")
        != "shared-only-clean-teacher-initialization-v1"
        or initialization.get("policy") != "shared-only"
        or initialization.get("teacher_checkpoint_sha256") != TEACHER_CHECKPOINT_SHA256
    ):
        raise RuntimeError(f"{spec.label} teacher initialization contract drifted")
    return {
        "task_id": spec.task_id,
        "experiment": spec.subject,
        "config_path": spec.config_path,
        "config_sha256": spec.config_sha256,
        "training_dataset_id": TRAINING_DATASET_ID,
        "seed": TRAINING_SEED,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "gpus": GPU_COUNT,
        "max_epochs": MAX_EPOCHS,
        "val_interval": VAL_INTERVAL,
        "precision": PRECISION,
        "amp": False,
    }


def _validate_teacher_audit(
    payload: Mapping[str, object], *, label: str
) -> dict[str, object]:
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != "shared-only-clean-teacher-initialization-v1"
        or payload.get("result") != "pass"
    ):
        raise RuntimeError(f"{label} teacher initialization audit did not pass")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"{label} teacher audit has no checkpoint")
    expected_checkpoint = {
        "expected_sha256": TEACHER_CHECKPOINT_SHA256,
        "sha256": TEACHER_CHECKPOINT_SHA256,
        "size_bytes": TEACHER_CHECKPOINT_BYTES,
    }
    for key, value in expected_checkpoint.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"{label} teacher audit checkpoint drifted: {key}")
    shared = payload.get("shared_initialization")
    fusion = payload.get("method_specific_fusion")
    target = payload.get("target")
    if not all(isinstance(item, Mapping) for item in (shared, fusion, target)):
        raise RuntimeError(f"{label} teacher audit sections are incomplete")
    assert isinstance(shared, Mapping)
    assert isinstance(fusion, Mapping)
    assert isinstance(target, Mapping)
    if (
        shared.get("exact_tensor_equality_verified") is not True
        or shared.get("shape_dtype_verified") is not True
        or shared.get("keys") != shared.get("expected_keys")
        or fusion.get("unchanged") is not True
        or fusion.get("sha256_before") != fusion.get("sha256_after")
        or target.get("nested_teacher_present") is not True
        or target.get("nested_teacher_full_equality_verified") is not True
        or target.get("model_type") != "ResilientV2XNet"
    ):
        raise RuntimeError(f"{label} teacher initialization audit invariants failed")
    return {
        "contract": payload["contract"],
        "result": payload["result"],
        "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "shared_tensor_equality_verified": True,
        "method_specific_fusion_unchanged": True,
        "nested_teacher_full_equality_verified": True,
    }


def _checkpoint_reference(
    payload: Mapping[str, object], *, spec: CandidateSpec, role: str
) -> dict[str, object]:
    if role == "canonical_final":
        if payload.get("filename") != "epoch_50.pth":
            raise RuntimeError(f"{spec.label} final contract is not epoch_50.pth")
        epoch = 50
        name = f"ResilientV2X {spec.subject} final checkpoint"
        remote_filename = f"{spec.subject}_epoch_50.pth"
        local_filename = remote_filename
    elif role == "clean_val_best_diagnostic":
        epoch = payload.get("epoch")
        if type(epoch) is not int or epoch not in {10, 20, 30, 40, 50}:
            raise RuntimeError(f"{spec.label} best checkpoint epoch is invalid")
        if payload.get("selection_protocol") != CLEAN_SELECTION_PROTOCOL:
            raise RuntimeError(f"{spec.label} best checkpoint protocol drifted")
        if payload.get("selection_metric") != CLEAN_SELECTION_METRIC:
            raise RuntimeError(f"{spec.label} best checkpoint metric drifted")
        if payload.get("claim_role") != (
            "diagnostic checkpoint candidate; final remains canonical"
        ):
            raise RuntimeError(f"{spec.label} best checkpoint role drifted")
        expected_source_name = (
            f"best_resilient_v2x_car_bev_ap_r40_0.70_epoch_{epoch}.pth"
        )
        if payload.get("filename") != expected_source_name:
            raise RuntimeError(f"{spec.label} best checkpoint filename drifted")
        name = f"ResilientV2X {spec.subject} clean-val best checkpoint"
        remote_filename = f"{spec.subject}_clean_val_best_epoch_{epoch}.pth"
        local_filename = remote_filename
    else:
        raise RuntimeError("unsupported checkpoint role")
    model_id = _clearml_id(payload.get("model_id"), f"{spec.label} {role} model")
    sha256 = _sha256(payload.get("sha256"), f"{spec.label} {role} checkpoint")
    size = _positive_int(
        payload.get("size_bytes"), f"{spec.label} {role} checkpoint bytes"
    )
    if payload.get("name") != name:
        raise RuntimeError(f"{spec.label} {role} model name drifted")
    url = formal._fileserver_url(
        payload.get("url"),
        expected_filename=remote_filename,
        context=f"{spec.label} {role} checkpoint",
    )
    return {
        "role": role,
        "paper_eligible": role == "canonical_final",
        "epoch": epoch,
        "model_id": model_id,
        "model_name": name,
        "model_url": url,
        "remote_filename": remote_filename,
        "filename": local_filename,
        "size_bytes": size,
        "sha256": sha256,
        **(
            {}
            if role == "canonical_final"
            else {
                "selection_protocol": CLEAN_SELECTION_PROTOCOL,
                "selection_metric": CLEAN_SELECTION_METRIC,
                "claim_scope": "diagnostic_only_not_paper_final",
            }
        ),
    }


def _validate_output_models(
    task: object,
    *,
    spec: CandidateSpec,
    checkpoints: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError(f"{spec.label} task cannot expose OutputModels")
    models = getter()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"{spec.label} OutputModel collection is invalid")
    outputs = list(models.get("output") or ())
    expected_ids = {str(item["model_id"]) for item in checkpoints}
    if len(outputs) != len(checkpoints):
        raise RuntimeError(f"{spec.label} has unexpected OutputModel count")
    records: list[dict[str, str]] = []
    observed_ids: set[str] = set()
    for model in outputs:
        model_id = str(getattr(model, "id", "") or "")
        if model_id not in expected_ids or model_id in observed_ids:
            raise RuntimeError(f"{spec.label} has unknown/duplicate OutputModel")
        reference = next(
            item for item in checkpoints if str(item["model_id"]) == model_id
        )
        expected = {
            "id": model_id,
            "task": spec.task_id,
            "name": str(reference["model_name"]),
            "url": str(reference["model_url"]),
        }
        for attribute, value in expected.items():
            observed = str(getattr(model, attribute, "") or "")
            if attribute == "url":
                observed = formal._fileserver_url(
                    observed,
                    expected_filename=str(reference["remote_filename"]),
                    context=f"{spec.label} OutputModel",
                )
            if observed != value:
                raise RuntimeError(
                    f"{spec.label} OutputModel identity drifted: {attribute}"
                )
        observed_ids.add(model_id)
        records.append(expected)
    return sorted(records, key=lambda item: item["id"])


def inspect_candidate(
    *,
    task_class: object,
    spec: CandidateSpec,
    auth_header_provider: Callable[[], Mapping[str, str]] | None = None,
    auth_refresher: Callable[[], None] | None = None,
    artifact_bytes_loader: ArtifactBytesLoader = _load_artifact_bytes,
) -> dict[str, object]:
    if _SAFE_SUBJECT.fullmatch(spec.subject) is None:
        raise RuntimeError("candidate subject is unsafe")
    _clearml_id(spec.task_id, f"{spec.label} task")
    _sha256(spec.config_sha256, f"{spec.label} config")
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise RuntimeError("ClearML Task class cannot resolve candidates")
    auth_header_provider, auth_refresher = formal._resolve_download_authentication(
        task_class,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    source = _validate_local_source(spec.source)
    task = getter(task_id=spec.task_id)
    _reload(task)
    status, critical_parameters = _validate_task_identity(task, spec)
    if status != "completed":
        raise CandidateNotReady(label=spec.label, task_id=spec.task_id, status=status)
    payloads: dict[str, dict[str, object]] = {}
    artifact_records: dict[str, dict[str, object]] = {}
    required = (RUN_ARTIFACT, TEACHER_AUDIT_ARTIFACT, FINAL_ARTIFACT)
    for name in required:
        payload, reference = _artifact_payload(
            task,
            name,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
            artifact_bytes_loader=artifact_bytes_loader,
        )
        payloads[name] = payload
        artifact_records[name] = reference
    artifacts = getattr(task, "artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"{spec.label} artifact collection is invalid")
    if BEST_ARTIFACT in artifacts:
        payload, reference = _artifact_payload(
            task,
            BEST_ARTIFACT,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
            artifact_bytes_loader=artifact_bytes_loader,
        )
        payloads[BEST_ARTIFACT] = payload
        artifact_records[BEST_ARTIFACT] = reference
    elif spec.require_diagnostic_best:
        raise RuntimeError(f"{spec.label} is missing diagnostic best contract")
    run_summary = _validate_run_contract(payloads[RUN_ARTIFACT], spec)
    teacher_summary = _validate_teacher_audit(
        payloads[TEACHER_AUDIT_ARTIFACT], label=spec.label
    )
    checkpoints = [
        _checkpoint_reference(
            payloads[FINAL_ARTIFACT], spec=spec, role="canonical_final"
        )
    ]
    if BEST_ARTIFACT in payloads:
        checkpoints.append(
            _checkpoint_reference(
                payloads[BEST_ARTIFACT],
                spec=spec,
                role="clean_val_best_diagnostic",
            )
        )
    if len({str(item["model_id"]) for item in checkpoints}) != len(checkpoints):
        raise RuntimeError(f"{spec.label} final and diagnostic models overlap")
    output_models = _validate_output_models(task, spec=spec, checkpoints=checkpoints)
    stable = {
        "label": spec.label,
        "subject": spec.subject,
        "task_id": spec.task_id,
        "task_name": spec.task_name,
        "parent_task_id": spec.parent_task_id,
        "status": status,
        "critical_parameters": critical_parameters,
        "source": source,
        "run_contract": run_summary,
        "teacher_audit": teacher_summary,
        "artifacts": artifact_records,
        "checkpoints": checkpoints,
        "output_models": output_models,
    }
    stable["remote_snapshot_sha256"] = hashlib.sha256(
        _canonical_json(stable).encode("utf-8")
    ).hexdigest()
    return stable


def _manifest_content(payload: Mapping[str, object]) -> str:
    detached = dict(payload)
    detached.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json(detached).encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("candidate archive manifest is not a regular file")
    return _strict_json(path.read_bytes(), context="candidate archive manifest")


def _legacy_manifest_matches(
    manifest: Mapping[str, object], snapshot: Mapping[str, object]
) -> bool:
    if manifest.get("schema_version") != 1:
        return False
    if (
        manifest.get("subject") != snapshot["subject"]
        or manifest.get("source_task_id") != snapshot["task_id"]
    ):
        return False
    models = manifest.get("models")
    checkpoints = snapshot.get("checkpoints")
    if not isinstance(models, list) or not isinstance(checkpoints, list):
        return False
    expected = {
        str(item["role"]): {
            "role": item["role"],
            "filename": item["filename"],
            "model_id": item["model_id"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in checkpoints
        if isinstance(item, Mapping)
    }
    observed_roles: set[str] = set()
    for record in models:
        if not isinstance(record, Mapping):
            return False
        role = str(record.get("role") or "")
        if role not in expected or role in observed_roles:
            return False
        observed_roles.add(role)
        for key, value in expected[role].items():
            if record.get(key) != value:
                return False
    return observed_roles == set(expected)


def _validate_v2_manifest(
    manifest: Mapping[str, object], snapshot: Mapping[str, object]
) -> None:
    if (
        manifest.get("schema_version") != 2
        or manifest.get("document_type") != DOCUMENT_TYPE
        or manifest.get("status") != "verified"
        or manifest.get("label") != snapshot["label"]
        or manifest.get("subject") != snapshot["subject"]
        or manifest.get("source_task_id") != snapshot["task_id"]
        or manifest.get("source_task_name") != snapshot["task_name"]
        or manifest.get("parent_task_id") != snapshot["parent_task_id"]
        or manifest.get("remote_snapshot_sha256") != snapshot["remote_snapshot_sha256"]
        or manifest.get("content_sha256") != _manifest_content(manifest)
    ):
        raise RuntimeError("candidate archive manifest identity mismatch")
    if (
        manifest.get("official_checkpoint_role") != "canonical_final"
        or manifest.get("diagnostic_checkpoint_role") != "clean_val_best_diagnostic"
        or manifest.get("checkpoint_policy") != "epoch_50_final_only_for_paper"
        or manifest.get("run_contract") != snapshot["run_contract"]
        or manifest.get("source") != snapshot["source"]
        or manifest.get("evidence_artifacts") != snapshot["artifacts"]
    ):
        raise RuntimeError("candidate archive manifest evidence binding mismatch")
    expected_teacher = {
        "task_id": TEACHER_TASK_ID,
        "model_id": TEACHER_MODEL_ID,
        "model_name": TEACHER_MODEL_NAME,
        "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": TEACHER_CHECKPOINT_BYTES,
        "initialization_audit": snapshot["teacher_audit"],
    }
    if manifest.get("teacher") != expected_teacher:
        raise RuntimeError("candidate archive manifest teacher binding mismatch")
    models = manifest.get("models")
    checkpoints = snapshot.get("checkpoints")
    if not isinstance(models, list) or not isinstance(checkpoints, list):
        raise RuntimeError("candidate archive manifest model list is invalid")
    expected_by_role = {
        str(item["role"]): dict(item)
        for item in checkpoints
        if isinstance(item, Mapping)
    }
    if len(models) != len(expected_by_role):
        raise RuntimeError("candidate archive manifest model count mismatch")
    observed_roles: set[str] = set()
    for model in models:
        if not isinstance(model, Mapping):
            raise RuntimeError("candidate archive manifest model is invalid")
        role = str(model.get("role") or "")
        expected = expected_by_role.get(role)
        if expected is None or role in observed_roles:
            raise RuntimeError(
                "candidate archive manifest model role is unknown/duplicate"
            )
        observed_roles.add(role)
        for key, value in expected.items():
            if model.get(key) != value:
                raise RuntimeError(
                    f"candidate archive manifest model drifted: {role}/{key}"
                )
        if model.get("source_task_id") != snapshot["task_id"]:
            raise RuntimeError("candidate archive manifest model task drifted")
    if observed_roles != set(expected_by_role):
        raise RuntimeError("candidate archive manifest model roles are incomplete")
    if manifest.get("model_count") != len(models) or manifest.get(
        "total_size_bytes"
    ) != sum(int(item["size_bytes"]) for item in checkpoints):
        raise RuntimeError("candidate archive manifest model summary mismatch")


def _candidate_paths(
    directory: Path, snapshot: Mapping[str, object]
) -> dict[str, Path]:
    checkpoints = snapshot.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise RuntimeError("candidate snapshot has invalid checkpoints")
    result: dict[str, Path] = {}
    for item in checkpoints:
        if not isinstance(item, Mapping):
            raise RuntimeError("candidate checkpoint is invalid")
        filename = str(item["filename"])
        if Path(filename).name != filename:
            raise RuntimeError("candidate checkpoint filename is unsafe")
        result[str(item["role"])] = directory / filename
    return result


def _validate_existing_directory(
    directory: Path, snapshot: Mapping[str, object]
) -> dict[str, object] | None:
    if not directory.exists():
        return None
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("candidate archive destination is not a directory")
    paths = _candidate_paths(directory, snapshot)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("candidate archive directory has no manifest")
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema_version") == 1:
        if not _legacy_manifest_matches(manifest, snapshot):
            raise RuntimeError(
                "legacy candidate manifest does not match remote evidence"
            )
        legacy = True
    else:
        _validate_v2_manifest(manifest, snapshot)
        legacy = False
    checkpoints = snapshot["checkpoints"]
    assert isinstance(checkpoints, list)
    allowed_files = {manifest_path.resolve(strict=True)}
    for item in checkpoints:
        assert isinstance(item, Mapping)
        path = paths[str(item["role"])]
        _regular_file(
            path,
            size=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
            context=f"{item['role']} local checkpoint",
        )
        allowed_files.add(path.resolve(strict=True))
    empty_legacy_directories: list[str] = []
    for child in directory.rglob("*"):
        if child.is_symlink():
            raise RuntimeError("candidate archive contains a symlink")
        if child.is_file() and child.resolve(strict=True) not in allowed_files:
            raise RuntimeError("candidate archive contains an unexpected file")
        if child.is_dir() and not any(child.iterdir()):
            empty_legacy_directories.append(str(child.relative_to(directory)))
    return {
        "legacy_manifest": legacy,
        "manifest": manifest,
        "paths": paths,
        "empty_legacy_directories": sorted(empty_legacy_directories),
    }


def audit_candidate_local_state(
    *, output_root: Path, snapshot: Mapping[str, object]
) -> dict[str, object]:
    directory = output_root / str(snapshot["subject"])
    existing = _validate_existing_directory(directory, snapshot)
    checkpoints = snapshot["checkpoints"]
    assert isinstance(checkpoints, list)
    return {
        "archive_directory": str(directory),
        "local_state": "verified_existing" if existing else "not_archived",
        "legacy_manifest": bool(existing and existing["legacy_manifest"]),
        "would_download_bytes": (
            0 if existing else sum(int(item["size_bytes"]) for item in checkpoints)
        ),
        "empty_legacy_directories": (
            [] if existing is None else existing["empty_legacy_directories"]
        ),
    }


@contextmanager
def _archive_lock(root: Path):
    path = root / ".candidate-archive.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise RuntimeError("cannot open candidate archive lock") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("candidate archive lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("candidate archive is already locked") from error
        yield
    finally:
        os.close(descriptor)


def _matching_blob(
    root: Path,
    *,
    size: int,
    sha256: str,
    exclude: Path | None = None,
) -> Path | None:
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = _read_manifest(manifest_path)
        except RuntimeError:
            continue
        models = manifest.get("models")
        if not isinstance(models, list):
            continue
        for record in models:
            if (
                not isinstance(record, Mapping)
                or record.get("size_bytes") != size
                or record.get("sha256") != sha256
            ):
                continue
            filename = str(record.get("filename") or "")
            if Path(filename).name != filename:
                continue
            candidate = manifest_path.parent / filename
            if exclude is not None and candidate == exclude:
                continue
            try:
                _regular_file(
                    candidate,
                    size=size,
                    sha256=sha256,
                    context="deduplication source",
                )
            except RuntimeError:
                continue
            return candidate.resolve(strict=True)
    return None


def _build_manifest(
    snapshot: Mapping[str, object],
    *,
    reused: Mapping[str, bool],
    hardlinked: Mapping[str, bool],
    archived_at: str,
    verified_at: str,
    empty_legacy_directories: Sequence[str],
) -> dict[str, object]:
    checkpoints = snapshot["checkpoints"]
    artifacts = snapshot["artifacts"]
    assert isinstance(checkpoints, list)
    assert isinstance(artifacts, Mapping)
    models = []
    for item in checkpoints:
        assert isinstance(item, Mapping)
        role = str(item["role"])
        models.append(
            {
                **dict(item),
                "source_task_id": snapshot["task_id"],
                "reused_existing_file": bool(reused.get(role)),
                "storage_deduplicated_via_hardlink": bool(hardlinked.get(role)),
            }
        )
    result: dict[str, object] = {
        "schema_version": 2,
        "document_type": DOCUMENT_TYPE,
        "status": "verified",
        "label": snapshot["label"],
        "subject": snapshot["subject"],
        "source_task_id": snapshot["task_id"],
        "source_task_name": snapshot["task_name"],
        "parent_task_id": snapshot["parent_task_id"],
        "official_checkpoint_role": "canonical_final",
        "diagnostic_checkpoint_role": "clean_val_best_diagnostic",
        "checkpoint_policy": "epoch_50_final_only_for_paper",
        "run_contract": snapshot["run_contract"],
        "source": snapshot["source"],
        "teacher": {
            "task_id": TEACHER_TASK_ID,
            "model_id": TEACHER_MODEL_ID,
            "model_name": TEACHER_MODEL_NAME,
            "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "checkpoint_size_bytes": TEACHER_CHECKPOINT_BYTES,
            "initialization_audit": snapshot["teacher_audit"],
        },
        "evidence_artifacts": {
            name: dict(record) for name, record in sorted(artifacts.items())
        },
        "models": models,
        "model_count": len(models),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in models),
        "remote_snapshot_sha256": snapshot["remote_snapshot_sha256"],
        "archived_at": archived_at,
        "last_verified_at": verified_at,
        "empty_legacy_directories_ignored": list(empty_legacy_directories),
    }
    result["content_sha256"] = _manifest_content(result)
    return result


ModelDownloader = Callable[..., tuple[Path, bool]]


def collect_candidate(
    *,
    task_class: object,
    spec: CandidateSpec,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    auth_header_provider: Callable[[], Mapping[str, str]] | None = None,
    auth_refresher: Callable[[], None] | None = None,
    artifact_bytes_loader: ArtifactBytesLoader = _load_artifact_bytes,
    model_downloader: ModelDownloader = formal._download_verified,
) -> dict[str, object]:
    auth_header_provider, auth_refresher = formal._resolve_download_authentication(
        task_class,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    snapshot = inspect_candidate(
        task_class=task_class,
        spec=spec,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
        artifact_bytes_loader=artifact_bytes_loader,
    )
    root = formal._safe_output_root(output_root)
    directory = root / spec.subject
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    with _archive_lock(root):
        existing = _validate_existing_directory(directory, snapshot)
        if existing is not None:
            checkpoints = snapshot["checkpoints"]
            assert isinstance(checkpoints, list)
            archived_at = now
            previous = existing["manifest"]
            if isinstance(previous, Mapping) and isinstance(
                previous.get("archived_at"), str
            ):
                archived_at = str(previous["archived_at"])
            manifest = _build_manifest(
                snapshot,
                reused={str(item["role"]): True for item in checkpoints},
                hardlinked={str(item["role"]): False for item in checkpoints},
                archived_at=archived_at,
                verified_at=now,
                empty_legacy_directories=existing["empty_legacy_directories"],
            )
            repeated = inspect_candidate(
                task_class=task_class,
                spec=spec,
                auth_header_provider=auth_header_provider,
                auth_refresher=auth_refresher,
                artifact_bytes_loader=artifact_bytes_loader,
            )
            if repeated["remote_snapshot_sha256"] != snapshot["remote_snapshot_sha256"]:
                raise RuntimeError("candidate evidence changed during local reuse")
            formal._atomic_write_text(
                directory / "manifest.json",
                json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)
                + "\n",
            )
            _validate_existing_directory(directory, snapshot)
            return manifest
        if directory.exists():
            raise RuntimeError("candidate archive destination appeared unexpectedly")
        checkpoints = snapshot["checkpoints"]
        assert isinstance(checkpoints, list)
        dedup_sources: dict[str, Path] = {}
        missing_bytes = 0
        for item in checkpoints:
            assert isinstance(item, Mapping)
            role = str(item["role"])
            source = _matching_blob(
                root,
                size=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
            )
            if source is None:
                missing_bytes += int(item["size_bytes"])
            else:
                dedup_sources[role] = source
        if missing_bytes:
            formal._require_collection_space(root, missing_bytes)
        staging_root = formal._safe_output_root(root / ".candidate-staging")
        if staging_root.stat().st_dev != root.stat().st_dev:
            raise RuntimeError("candidate staging must share the archive filesystem")
        stage: Path | None = Path(
            tempfile.mkdtemp(prefix=f"{spec.subject}.{spec.task_id}.", dir=staging_root)
        ).resolve(strict=True)
        try:
            reused: dict[str, bool] = {}
            hardlinked: dict[str, bool] = {}
            for item in checkpoints:
                assert isinstance(item, Mapping)
                role = str(item["role"])
                assert stage is not None
                destination = stage / str(item["filename"])
                source = dedup_sources.get(role)
                if source is not None:
                    os.link(source, destination)
                    reused[role] = True
                    hardlinked[role] = True
                else:
                    path, reused_download = model_downloader(
                        model_url=str(item["model_url"]),
                        destination=destination,
                        expected_size_bytes=int(item["size_bytes"]),
                        expected_sha256=str(item["sha256"]),
                        auth_header_provider=auth_header_provider,
                        auth_refresher=auth_refresher,
                    )
                    if Path(path).resolve(strict=True) != destination.resolve(
                        strict=True
                    ):
                        raise RuntimeError("model downloader returned unexpected path")
                    reused[role] = bool(reused_download)
                    hardlinked[role] = False
                _regular_file(
                    destination,
                    size=int(item["size_bytes"]),
                    sha256=str(item["sha256"]),
                    context=f"{role} staged checkpoint",
                )
            manifest = _build_manifest(
                snapshot,
                reused=reused,
                hardlinked=hardlinked,
                archived_at=now,
                verified_at=now,
                empty_legacy_directories=(),
            )
            formal._atomic_write_text(
                stage / "manifest.json",
                json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)
                + "\n",
            )
            repeated = inspect_candidate(
                task_class=task_class,
                spec=spec,
                auth_header_provider=auth_header_provider,
                auth_refresher=auth_refresher,
                artifact_bytes_loader=artifact_bytes_loader,
            )
            if repeated["remote_snapshot_sha256"] != snapshot["remote_snapshot_sha256"]:
                raise RuntimeError("candidate evidence changed during download")
            _validate_existing_directory(stage, snapshot)
            if directory.exists():
                raise RuntimeError(
                    "candidate archive destination appeared during download"
                )
            os.replace(stage, directory)
            stage = None
            formal._fsync_directory(root)
            _validate_existing_directory(directory, snapshot)
            return manifest
        finally:
            if stage is not None and stage.exists():
                shutil.rmtree(stage)


def audit_candidate(
    *,
    task_class: object,
    spec: CandidateSpec,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    auth_header_provider: Callable[[], Mapping[str, str]] | None = None,
    auth_refresher: Callable[[], None] | None = None,
    artifact_bytes_loader: ArtifactBytesLoader = _load_artifact_bytes,
) -> dict[str, object]:
    snapshot = inspect_candidate(
        task_class=task_class,
        spec=spec,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
        artifact_bytes_loader=artifact_bytes_loader,
    )
    local = audit_candidate_local_state(output_root=output_root, snapshot=snapshot)
    return {
        "label": spec.label,
        "subject": spec.subject,
        "task_id": spec.task_id,
        "status": "ready",
        "remote_snapshot_sha256": snapshot["remote_snapshot_sha256"],
        "checkpoint_count": len(snapshot["checkpoints"]),
        "official_checkpoint_role": "canonical_final",
        "diagnostic_checkpoint_role": "clean_val_best_diagnostic",
        **local,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        choices=("all", *SPEC_BY_LABEL),
        default=[],
        help="Pinned candidate label; repeatable. Defaults to all.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Archive model bytes. Without this flag the command is read-only.",
    )
    parser.add_argument("--execute-token", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.collect and args.execute_token != EXECUTE_TOKEN:
        raise RuntimeError(f"--collect requires --execute-token {EXECUTE_TOKEN}")
    if not args.collect and args.execute_token:
        raise RuntimeError("--execute-token is only valid with --collect")
    labels = list(args.candidate)
    if not labels or "all" in labels:
        if len(labels) > 1:
            raise RuntimeError("--candidate all cannot be combined with labels")
        labels = list(SPEC_BY_LABEL)
    if len(labels) != len(set(labels)):
        raise RuntimeError("candidate labels must be unique")
    from clearml import Task

    reports: list[dict[str, object]] = []
    failed = False
    for label in labels:
        spec = SPEC_BY_LABEL[label]
        try:
            if args.collect:
                manifest = collect_candidate(
                    task_class=Task,
                    spec=spec,
                    output_root=args.output_root,
                )
                reports.append(
                    {
                        "label": label,
                        "subject": spec.subject,
                        "task_id": spec.task_id,
                        "status": "archived",
                        "manifest_content_sha256": manifest["content_sha256"],
                    }
                )
            else:
                reports.append(
                    audit_candidate(
                        task_class=Task,
                        spec=spec,
                        output_root=args.output_root,
                    )
                )
        except CandidateNotReady as error:
            if args.collect:
                raise
            reports.append(
                {
                    "label": error.label,
                    "task_id": error.task_id,
                    "status": "not_ready",
                    "remote_status": error.status,
                }
            )
        except RuntimeError as error:
            if args.collect:
                raise
            failed = True
            reports.append(
                {
                    "label": label,
                    "task_id": spec.task_id,
                    "status": "evidence_failed",
                    "error": str(error),
                }
            )
    print(
        json.dumps(
            {"mode": "collect" if args.collect else "audit", "candidates": reports},
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
