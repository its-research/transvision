#!/usr/bin/env python3
"""Evaluate P1 with P0's canonical epoch-50 final checkpoint.

The candidate in this module is an inference-only checkpoint-reuse diagnostic.
It is intentionally outside the sealed 26-method trained leaderboard.  The
configuration subject and checkpoint owner are separate fields at every join.

No remote write occurs merely by importing this module.  ``run-worker`` is the
only command that writes ClearML artifacts and is intended for a dedicated,
separately packaged A100 evaluation task.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from allegroai import Dataset, Task
except ImportError:
    try:
        from clearml import Dataset, Task
    except ImportError:  # Local contract/unit tests do not require ClearML.
        Dataset = None  # type: ignore[assignment]
        Task = None  # type: ignore[assignment]


P0_TASK_ID = "8883c51ced4f4951a45edbaefe6342d4"
P0_CONFIG_SUBJECT = "support_residual_no_reliability_linear"
P0_EXPERIMENT_IDENTITY = "dair_improvement_support_residual_no_reliability_linear"
P0_CONFIG = (
    "configs/resilient_v2x/improvements/support_residual_no_reliability_linear.py"
)
P0_CONFIG_SHA256 = "e8208673e2633231c5a60d3783e0735aa3fe0ccaede1a386f4e337597414d1fc"
P0_FINAL_MODEL_NAME = (
    "ResilientV2X support_residual_no_reliability_linear final checkpoint"
)
P0_FINAL_SOURCE_FILENAME = "epoch_50.pth"
P0_FINAL_REMOTE_FILENAME = "support_residual_no_reliability_linear_epoch_50.pth"

P1_CONFIG_SUBJECT = "reliability_gated_residual"
P1_EXPERIMENT_IDENTITY = "dair_improvement_reliability_gated_residual"
P1_CONFIG = "configs/resilient_v2x/improvements/reliability_gated_residual.py"
P1_CONFIG_SHA256 = "76f7164326c7de8f6c5e9148887fb02772c141e981ea42e579ff9bea010c0ef1"
P1_TRAINED_CANDIDATE_IDENTITY = (
    "dair_improvement_reliability_gated_residual_trained_50e"
)
ZERO_SHOT_CANDIDATE_IDENTITY = (
    "dair_improvement_reliability_gated_residual_zero_shot_p0_epoch50_final"
)
ZERO_SHOT_DISPLAY_SUBJECT_ALIAS = "reliability_gated_residual_p0_final_zero_shot"
ZERO_SHOT_EVIDENCE_CLASS = "inference_only_checkpoint_reuse"
OPTIMIZATION_ORIGIN = "P0"
WEIGHTS_RETRAINED = False

PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
SAMPLE_COUNT = 1_337
GROUND_TRUTH_COUNT = 11_330
DELAYS_MS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
RUN_COUNT = len(DELAYS_MS) * len(CONDITIONS)
TRAINING_SEED = 20_250_218
TRAINING_OVERLAY_PROTOCOL_SEED = 20_250_218
CHECKPOINT_POLICY = "epoch_50_final_only"
CHECKPOINT_ROLE = "canonical_epoch_50_final_output_model"
FORBIDDEN_CHECKPOINT_ROLES = (
    "clean_validation_best",
    "clean_validation_best_for_teacher_handoff",
)

FILES_SERVER_HOST = "10.100.34.118"
FILES_SERVER_PORT = 8081
FILES_SERVER_URI = f"http://{FILES_SERVER_HOST}:{FILES_SERVER_PORT}"
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
WORKER_QUEUE = "GPU4-A100"
WORKER_QUEUE_ID = "9350f33af13a448da8339eb7bea52fdf"
REQUIRED_GPU_CAPABILITIES = ((8, 0),) * 4

FINAL_CHECKPOINT_ARTIFACT = "final_checkpoint_contract"
BEST_CHECKPOINT_ARTIFACT = "best_checkpoint_contract"
RUN_CONTRACT_ARTIFACT = "run_contract"
INITIALIZATION_AUDIT_ARTIFACT = "common_teacher_initialization_audit"
IDENTITY_ALIAS_MAPPING_ARTIFACT = "identity_alias_mapping"
CHECKPOINT_REUSE_BINDING_ARTIFACT = "checkpoint_reuse_binding"
CHECKPOINT_REUSE_EQUIVALENCE_ARTIFACT = "checkpoint_reuse_equivalence"
CHECKPOINT_REUSE_EVALUATION_RECEIPT_ARTIFACT = "checkpoint_reuse_evaluation_receipt"
CONTROLLED_METRICS_ARTIFACT = "controlled_baseline_metrics"

CLEARML_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
WAITING_STATUSES = frozenset({"created", "queued", "in_progress"})
FAILED_STATUSES = frozenset(
    {
        "failed",
        "stopped",
        "closed",
        "published",
        "publishing",
        "rejected",
        "unknown",
    }
)


class P1CheckpointReuseError(RuntimeError):
    """Raised when checkpoint reuse cannot be proven from exact evidence."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise P1CheckpointReuseError(
            f"value is outside the canonical JSON domain: {error}"
        ) from error


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _sha256(value: object, *, context: str) -> str:
    result = str(value or "")
    if SHA256_PATTERN.fullmatch(result) is None:
        raise P1CheckpointReuseError(f"{context} must be a lowercase SHA-256")
    return result


def _task_id(value: object, *, context: str) -> str:
    result = str(value or "")
    if CLEARML_ID_PATTERN.fullmatch(result) is None:
        raise P1CheckpointReuseError(f"{context} must be a lowercase 32-hex ClearML ID")
    return result


def _positive_int(value: object, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise P1CheckpointReuseError(f"{context} must be a positive integer")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise P1CheckpointReuseError(
            f"{context} keys mismatch; missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = _sha256(value.get("seal_sha256"), context=f"{context} seal")
    if _sealed(value)["seal_sha256"] != observed:
        raise P1CheckpointReuseError(f"{context} seal SHA-256 mismatch")
    return observed


def identity_alias_mapping() -> dict[str, object]:
    """Return the sealed one-to-one display alias for the canonical identity."""

    return _sealed(
        {
            "schema_version": 1,
            "document_type": "resilient_v2x_candidate_identity_alias_mapping",
            "canonical_identity": ZERO_SHOT_CANDIDATE_IDENTITY,
            "display_subject_alias": ZERO_SHOT_DISPLAY_SUBJECT_ALIAS,
            "mapping_cardinality": "one_to_one",
            "canonical_identity_count": 1,
            "alias_count": 1,
            "alias_is_candidate_identity": False,
            "aliases": {
                ZERO_SHOT_DISPLAY_SUBJECT_ALIAS: ZERO_SHOT_CANDIDATE_IDENTITY,
            },
        }
    )


def validate_identity_alias_mapping(value: Mapping[str, object]) -> str:
    """Require the exact sealed alias mapping; aliases never become identities."""

    seal = _require_seal(value, context="identity alias mapping")
    expected = identity_alias_mapping()
    if _canonical_json(value) != _canonical_json(expected):
        raise P1CheckpointReuseError("identity alias mapping is not the canonical map")
    if value.get("canonical_identity") == value.get("display_subject_alias"):
        raise P1CheckpointReuseError(
            "display alias was conflated with canonical identity"
        )
    return seal


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise P1CheckpointReuseError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _trusted_model_url(value: object, *, context: str) -> str:
    if type(value) is not str:
        raise P1CheckpointReuseError(f"{context} URL must be a string")
    parsed = urlsplit(value)
    path = PurePosixPath(unquote(parsed.path))
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != FILES_SERVER_HOST
        or parsed.port != FILES_SERVER_PORT
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or not parsed.path.startswith("/")
        or ".." in path.parts
    ):
        raise P1CheckpointReuseError(f"{context} URL is not trusted")
    return value


def _reload(task: object, *, context: str) -> None:
    method = getattr(task, "reload", None)
    if callable(method):
        try:
            result = method()
        except Exception as error:
            raise P1CheckpointReuseError(f"cannot reload {context}") from error
        if result is False:
            raise P1CheckpointReuseError(f"reload of {context} was rejected")


def _status(task: object) -> str:
    return str(getattr(task, "status", "") or "").lower()


def _parameters(task: object, *, context: str) -> Mapping[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise P1CheckpointReuseError(f"{context} cannot expose parameters")
    try:
        value = getter(cast=False)
    except TypeError:
        value = getter()
    if not isinstance(value, Mapping):
        raise P1CheckpointReuseError(f"{context} parameters are invalid")
    return value


def _parameter_matches(observed: object, expected: object) -> bool:
    if type(expected) is bool:
        return observed is expected or str(observed) == str(expected)
    if type(expected) is int:
        return str(observed) == str(expected)
    return (
        type(observed) is type(expected)
        and observed == expected
        or (type(expected) is str and str(observed) == expected)
    )


def _require_parameters(
    parameters: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    context: str,
) -> None:
    for key, expected_value in expected.items():
        if key not in parameters or not _parameter_matches(
            parameters[key], expected_value
        ):
            raise P1CheckpointReuseError(
                f"{context} parameter {key!r} mismatch: "
                f"expected {expected_value!r}, got {parameters.get(key)!r}"
            )


def _artifact_mapping(task: object, name: str, *, context: str) -> Mapping[str, object]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise P1CheckpointReuseError(f"{context} lacks artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise P1CheckpointReuseError(f"{context} artifact {name!r} is unreadable")
    try:
        value = getter(force_download=True)
    except TypeError:
        value = getter()
    if isinstance(value, (str, Path)):
        try:
            path = Path(value)
            if path.is_symlink():
                raise P1CheckpointReuseError(
                    f"{context} artifact {name!r} path is a symlink"
                )
            path = path.resolve(strict=True)
            if not path.is_file():
                raise P1CheckpointReuseError(
                    f"{context} artifact {name!r} path is not a file"
                )
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise P1CheckpointReuseError(
                f"{context} artifact {name!r} path is not canonical JSON"
            ) from error
    if not isinstance(value, Mapping):
        raise P1CheckpointReuseError(
            f"{context} artifact {name!r} is not a JSON object"
        )
    # Freeze through canonical JSON so mutable/mock mappings cannot change later.
    try:
        frozen = json.loads(_canonical_json(value))
    except json.JSONDecodeError as error:  # pragma: no cover - canonical producer.
        raise P1CheckpointReuseError(
            f"{context} artifact {name!r} cannot be frozen"
        ) from error
    if not isinstance(frozen, Mapping):
        raise P1CheckpointReuseError(f"{context} artifact {name!r} is invalid")
    return frozen


def _models(task: object, *, context: str) -> Mapping[str, object]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise P1CheckpointReuseError(f"{context} cannot enumerate models")
    value = getter()
    if not isinstance(value, Mapping):
        raise P1CheckpointReuseError(f"{context} model mapping is invalid")
    return value


def _task_parent(task: object) -> str:
    return str(
        getattr(getattr(task, "data", None), "parent", None)
        or getattr(task, "parent", None)
        or ""
    )


def _download_and_verify_model(
    model: object,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> Path:
    getter = getattr(model, "get_local_copy", None)
    if not callable(getter):
        raise P1CheckpointReuseError("canonical P0 OutputModel cannot be downloaded")
    try:
        local = getter(
            extract_archive=False,
            raise_on_error=True,
            force_download=True,
        )
    except TypeError:
        local = getter()
    if type(local) is not str or not local:
        raise P1CheckpointReuseError("canonical P0 OutputModel returned no local copy")
    candidate = Path(local)
    if candidate.is_symlink():
        raise P1CheckpointReuseError("canonical P0 local checkpoint is a symlink")
    try:
        path = candidate.resolve(strict=True)
    except OSError as error:
        raise P1CheckpointReuseError(
            "canonical P0 local checkpoint is missing"
        ) from error
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise P1CheckpointReuseError("canonical P0 local checkpoint byte size mismatch")
    if _sha256_file(path) != expected_sha256:
        raise P1CheckpointReuseError("canonical P0 local checkpoint SHA-256 mismatch")
    return path


def _validate_p0_run_contract(
    value: Mapping[str, object],
    *,
    task_id: str,
) -> dict[str, object]:
    expected = {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": P0_CONFIG_SUBJECT,
        "experiment_kind": "sota_candidate",
        "training_dataset_id": TRAINING_DATASET_ID,
        "gpus": 4,
        "global_batch_size": 8,
        "train_batch_size_per_gpu": 2,
        "max_epochs": 50,
        "training_seed": TRAINING_SEED,
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "seed": TRAINING_SEED,
        "amp": False,
        "precision": "FP32",
        "val_interval": 10,
        "per_epoch_validation": False,
        "condition_evaluation": False,
    }
    for key, expected_value in expected.items():
        if (
            key not in value
            or type(value[key]) is not type(expected_value)
            or (value[key] != expected_value)
        ):
            raise P1CheckpointReuseError(f"P0 run contract {key!r} mismatch")
    config = value.get("config")
    if not isinstance(config, Mapping) or config.get("declared") != P0_CONFIG:
        raise P1CheckpointReuseError("P0 run contract config identity mismatch")
    initialization = value.get("common_teacher_initialization")
    if not isinstance(initialization, Mapping) or (
        initialization.get("audit_artifact_name") != INITIALIZATION_AUDIT_ARTIFACT
    ):
        raise P1CheckpointReuseError("P0 initialization contract mismatch")
    return dict(value)


def _validate_initialization_audit(value: Mapping[str, object]) -> dict[str, object]:
    if (
        value.get("schema_version") != 1
        or value.get("result") != "pass"
        or value.get("contract") != "shared-only-clean-teacher-initialization-v1"
    ):
        raise P1CheckpointReuseError("P0 common-teacher initialization audit failed")
    shared = value.get("shared_initialization")
    if not isinstance(shared, Mapping) or (
        shared.get("shape_dtype_verified") is not True
        or shared.get("exact_tensor_equality_verified") is not True
    ):
        raise P1CheckpointReuseError("P0 shared initialization evidence is incomplete")
    return dict(value)


def resolve_p0_canonical_final(
    task: object,
    *,
    expected_task_id: str = P0_TASK_ID,
    verify_checkpoint_bytes: bool = True,
) -> dict[str, object]:
    """Resolve and pin only P0's unique canonical epoch-50 final OutputModel."""

    task_id = _task_id(expected_task_id, context="expected P0 task")
    _reload(task, context="P0 training task")
    if _task_id(getattr(task, "id", None), context="P0 training task") != task_id:
        raise P1CheckpointReuseError("P0 training task identity mismatch")
    if _status(task) != "completed":
        raise P1CheckpointReuseError(
            f"P0 training task is not completed: {_status(task)!r}"
        )
    parameters = _parameters(task, context="P0 training task")
    _require_parameters(
        parameters,
        {
            "Args/stage": "all",
            "Args/experiment_from_task": P0_CONFIG_SUBJECT,
            "Args/training_dataset_id": TRAINING_DATASET_ID,
            "Args/training_seed": TRAINING_SEED,
            "Args/gpus": 4,
            "Args/max_epochs": 50,
            "Args/amp": False,
        },
        context="P0 training task",
    )

    final_contract = _artifact_mapping(
        task,
        FINAL_CHECKPOINT_ARTIFACT,
        context="P0 training task",
    )
    _require_exact_keys(
        final_contract,
        {"model_id", "name", "url", "filename", "size_bytes", "sha256"},
        context="P0 final checkpoint contract",
    )
    model_id = _task_id(final_contract.get("model_id"), context="P0 final model")
    checkpoint_sha256 = _sha256(
        final_contract.get("sha256"), context="P0 final checkpoint"
    )
    checkpoint_bytes = _positive_int(
        final_contract.get("size_bytes"), context="P0 final checkpoint bytes"
    )
    model_url = _trusted_model_url(
        final_contract.get("url"), context="P0 final checkpoint"
    )
    if (
        final_contract.get("name") != P0_FINAL_MODEL_NAME
        or final_contract.get("filename") != P0_FINAL_SOURCE_FILENAME
        or PurePosixPath(unquote(urlsplit(model_url).path)).name
        != P0_FINAL_REMOTE_FILENAME
    ):
        raise P1CheckpointReuseError("P0 final checkpoint role/name/filename mismatch")

    output_models = _models(task, context="P0 training task").get("output")
    if not isinstance(output_models, Sequence) or isinstance(
        output_models, (str, bytes)
    ):
        raise P1CheckpointReuseError("P0 training task output models are invalid")
    candidates = [
        model
        for model in output_models
        if getattr(model, "name", None) == P0_FINAL_MODEL_NAME
    ]
    if len(candidates) != 1:
        raise P1CheckpointReuseError(
            "P0 must expose exactly one canonical final OutputModel; "
            f"found {len(candidates)}"
        )
    model = candidates[0]
    if (
        _task_id(getattr(model, "id", None), context="P0 OutputModel") != model_id
        or _task_id(getattr(model, "task", None), context="P0 OutputModel owner")
        != task_id
        or _trusted_model_url(getattr(model, "url", None), context="P0 OutputModel")
        != model_url
    ):
        raise P1CheckpointReuseError("P0 OutputModel binding mismatch")
    iteration = getattr(model, "iteration", None)
    if iteration is not None and iteration != 50:
        raise P1CheckpointReuseError("P0 OutputModel is not epoch 50")

    best_contract = _artifact_mapping(
        task,
        BEST_CHECKPOINT_ARTIFACT,
        context="P0 training task",
    )
    if (
        best_contract.get("model_id") == model_id
        or best_contract.get("name") == P0_FINAL_MODEL_NAME
        or best_contract.get("claim_role")
        != "diagnostic checkpoint candidate; final remains canonical"
    ):
        raise P1CheckpointReuseError(
            "P0 clean-best/final checkpoint roles are conflated"
        )

    run_contract = _validate_p0_run_contract(
        _artifact_mapping(task, RUN_CONTRACT_ARTIFACT, context="P0 training task"),
        task_id=task_id,
    )
    initialization_audit = _validate_initialization_audit(
        _artifact_mapping(
            task,
            INITIALIZATION_AUDIT_ARTIFACT,
            context="P0 training task",
        )
    )
    if verify_checkpoint_bytes:
        _download_and_verify_model(
            model,
            expected_bytes=checkpoint_bytes,
            expected_sha256=checkpoint_sha256,
        )

    binding = _sealed(
        {
            "schema_version": 1,
            "binding_type": "resilient_v2x_p1_p0_epoch50_checkpoint_reuse",
            "candidate_identity": ZERO_SHOT_CANDIDATE_IDENTITY,
            "identity_alias_mapping_seal_sha256": identity_alias_mapping()[
                "seal_sha256"
            ],
            "evidence_class": ZERO_SHOT_EVIDENCE_CLASS,
            "config_subject": P1_CONFIG_SUBJECT,
            "config_experiment_identity": P1_EXPERIMENT_IDENTITY,
            "config_path": P1_CONFIG,
            "config_sha256": P1_CONFIG_SHA256,
            "checkpoint_subject": P0_CONFIG_SUBJECT,
            "checkpoint_experiment_identity": P0_EXPERIMENT_IDENTITY,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "checkpoint_role": CHECKPOINT_ROLE,
            "weights_retrained": WEIGHTS_RETRAINED,
            "optimization_origin": OPTIMIZATION_ORIGIN,
            "source_task": {
                "task_id": task_id,
                "required_status": "completed",
                "run_contract_sha256": _content_sha256(run_contract),
                "initialization_audit_sha256": _content_sha256(initialization_audit),
                "training_seed": TRAINING_SEED,
                "training_overlay_protocol_seed": (TRAINING_OVERLAY_PROTOCOL_SEED),
                "max_epochs": 50,
                "precision": "FP32",
            },
            "checkpoint": {
                "model_id": model_id,
                "model_name": P0_FINAL_MODEL_NAME,
                "model_url": model_url,
                "source_filename": P0_FINAL_SOURCE_FILENAME,
                "remote_filename": P0_FINAL_REMOTE_FILENAME,
                "size_bytes": checkpoint_bytes,
                "sha256": checkpoint_sha256,
                "bytes_recomputed": verify_checkpoint_bytes,
            },
            "forbidden_checkpoint_roles": list(FORBIDDEN_CHECKPOINT_ROLES),
            "formal_evaluation": {
                "protocol_id": PROTOCOL_ID,
                "sample_count": SAMPLE_COUNT,
                "training_seed": TRAINING_SEED,
                "overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
                "delays_ms": list(DELAYS_MS),
                "conditions": list(CONDITIONS),
                "run_count": RUN_COUNT,
            },
        }
    )
    return binding


def download_bound_p0_checkpoint(
    task: object,
    binding: Mapping[str, object],
) -> Path:
    """Download exactly the OutputModel pinned by a machine-independent binding."""

    _require_seal(binding, context="checkpoint reuse binding")
    if (
        binding.get("candidate_identity") != ZERO_SHOT_CANDIDATE_IDENTITY
        or binding.get("checkpoint_subject") != P0_CONFIG_SUBJECT
        or binding.get("identity_alias_mapping_seal_sha256")
        != identity_alias_mapping()["seal_sha256"]
    ):
        raise P1CheckpointReuseError("checkpoint reuse binding identity drifted")
    _reload(task, context="P0 training task")
    if (
        _task_id(getattr(task, "id", None), context="P0 training task") != P0_TASK_ID
        or _status(task) != "completed"
    ):
        raise P1CheckpointReuseError("bound P0 training task identity/status mismatch")
    checkpoint = binding.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise P1CheckpointReuseError("checkpoint reuse binding lacks checkpoint")
    model_id = _task_id(checkpoint.get("model_id"), context="bound P0 model")
    expected_sha = _sha256(checkpoint.get("sha256"), context="bound P0 checkpoint")
    expected_bytes = _positive_int(
        checkpoint.get("size_bytes"), context="bound P0 checkpoint bytes"
    )
    model_url = _trusted_model_url(
        checkpoint.get("model_url"), context="bound P0 checkpoint"
    )
    output = _models(task, context="P0 training task").get("output")
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise P1CheckpointReuseError("P0 training task output models are invalid")
    candidates = [
        model
        for model in output
        if getattr(model, "id", None) == model_id
        and getattr(model, "name", None) == P0_FINAL_MODEL_NAME
    ]
    if len(candidates) != 1:
        raise P1CheckpointReuseError(
            "bound P0 checkpoint must resolve to exactly one OutputModel"
        )
    model = candidates[0]
    if (
        _task_id(getattr(model, "task", None), context="bound P0 model owner")
        != P0_TASK_ID
        or _trusted_model_url(
            getattr(model, "url", None), context="bound P0 OutputModel"
        )
        != model_url
    ):
        raise P1CheckpointReuseError("bound P0 OutputModel provenance drifted")
    return _download_and_verify_model(
        model,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha,
    )


def _strip_training_only_model_config(value: Mapping[str, object]) -> dict[str, object]:
    result = copy.deepcopy(dict(value))
    result.pop("teacher", None)
    result.pop("teacher_checkpoint", None)
    result.pop("distillation", None)
    return result


def validate_config_subject_split(
    p0_model_config: Mapping[str, object],
    p1_model_config: Mapping[str, object],
) -> dict[str, object]:
    """Prove the deployment model config delta is only ``False -> True``."""

    p0 = _strip_training_only_model_config(p0_model_config)
    p1 = _strip_training_only_model_config(p1_model_config)
    gate = "support_residual_reliability_gate"
    p0_gate = p0.get(gate, False)
    p1_gate = p1.get(gate, False)
    if p0_gate is not False or p1_gate is not True:
        raise P1CheckpointReuseError(
            "P0/P1 reliability-gate values are not exactly False -> True"
        )
    normalized_p0 = copy.deepcopy(p0)
    normalized_p1 = copy.deepcopy(p1)
    normalized_p0[gate] = False
    normalized_p1[gate] = False
    if _canonical_json(normalized_p0) != _canonical_json(normalized_p1):
        raise P1CheckpointReuseError(
            "P0/P1 deployment model configs differ beyond the reliability gate"
        )
    return {
        "config_subject": P1_CONFIG_SUBJECT,
        "checkpoint_subject": P0_CONFIG_SUBJECT,
        "only_deployment_model_delta": gate,
        "p0_value": False,
        "p1_value": True,
        "training_only_fields_removed": [
            "teacher",
            "teacher_checkpoint",
            "distillation",
        ],
        "normalized_deployment_model_sha256": _content_sha256(normalized_p0),
    }


def _tensor_schema(state: Mapping[str, Any], *, context: str) -> dict[str, object]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency.
        raise P1CheckpointReuseError(
            "PyTorch is required for checkpoint audit"
        ) from error
    entries: list[dict[str, object]] = []
    total_numel = 0
    total_bytes = 0
    for index, (key, tensor) in enumerate(state.items()):
        if type(key) is not str or not key or not isinstance(tensor, torch.Tensor):
            raise P1CheckpointReuseError(f"{context} is not a tensor mapping")
        numel = int(tensor.numel())
        size_bytes = numel * int(tensor.element_size())
        total_numel += numel
        total_bytes += size_bytes
        entries.append(
            {
                "index": index,
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": numel,
                "bytes": size_bytes,
            }
        )
    if not entries:
        raise P1CheckpointReuseError(f"{context} is empty")
    return {
        "key_count": len(entries),
        "numel": total_numel,
        "bytes": total_bytes,
        "entries_sha256": _content_sha256(entries),
    }


def prepare_filtered_student_state(
    checkpoint: Mapping[str, Any],
    *,
    p0_student_state: Mapping[str, Any],
    p1_student_state: Mapping[str, Any],
    p0_full_state: Mapping[str, Any],
) -> tuple[OrderedDict[str, Any], dict[str, object]]:
    """Validate schema and filter only the exact nested frozen-teacher extras."""

    try:
        import torch
        from transvision.models.common_teacher_initialization import (
            normalize_checkpoint_state_dict,
            tensor_mapping_sha256,
        )
    except ImportError as error:  # pragma: no cover - runtime dependency.
        raise P1CheckpointReuseError(
            "checkpoint audit dependencies are missing"
        ) from error

    source = normalize_checkpoint_state_dict(checkpoint)
    p0_schema = _tensor_schema(p0_student_state, context="P0 student state")
    p1_schema = _tensor_schema(p1_student_state, context="P1 student state")
    if list(p0_student_state) != list(p1_student_state) or p0_schema != p1_schema:
        raise P1CheckpointReuseError("P0/P1 student state schemas differ")

    student_keys = list(p0_student_state)
    missing = [key for key in student_keys if key not in source]
    extra = [key for key in source if key not in p0_student_state]
    expected_nested = [
        key for key in p0_full_state if key.startswith("teacher.teacher.")
    ]
    if missing:
        raise P1CheckpointReuseError(
            f"checkpoint is missing student keys: {missing[:5]!r}"
        )
    if extra != expected_nested or not expected_nested:
        raise P1CheckpointReuseError(
            "checkpoint extras are not the exact sealed nested-teacher key list"
        )
    if any(key not in source for key in expected_nested):
        raise P1CheckpointReuseError("checkpoint nested-teacher state is incomplete")

    filtered: OrderedDict[str, Any] = OrderedDict()
    for key in student_keys:
        source_tensor = source[key]
        p0_tensor = p0_student_state[key]
        p1_tensor = p1_student_state[key]
        if not all(
            isinstance(item, torch.Tensor)
            for item in (source_tensor, p0_tensor, p1_tensor)
        ):
            raise P1CheckpointReuseError(f"checkpoint key {key!r} is not tensor-valued")
        if (
            tuple(source_tensor.shape) != tuple(p0_tensor.shape)
            or tuple(source_tensor.shape) != tuple(p1_tensor.shape)
            or source_tensor.dtype != p0_tensor.dtype
            or source_tensor.dtype != p1_tensor.dtype
        ):
            raise P1CheckpointReuseError(
                f"checkpoint tensor schema mismatch for {key!r}"
            )
        if (
            source_tensor.is_floating_point() or source_tensor.is_complex()
        ) and not bool(torch.isfinite(source_tensor).all().item()):
            raise P1CheckpointReuseError(f"checkpoint tensor {key!r} is non-finite")
        filtered[key] = source_tensor

    filtered_schema = _tensor_schema(filtered, context="filtered student state")
    if filtered_schema != p0_schema:
        raise P1CheckpointReuseError("filtered checkpoint student schema drifted")
    nested_schema = _tensor_schema(
        OrderedDict((key, source[key]) for key in expected_nested),
        context="nested teacher state",
    )
    return filtered, {
        "student_schema": p0_schema,
        "checkpoint_key_count": len(source),
        "missing_student_keys": [],
        "extra_checkpoint_keys": expected_nested,
        "extra_checkpoint_keys_sha256": _content_sha256(expected_nested),
        "nested_teacher_schema": nested_schema,
        "filtered_student_state_sha256": tensor_mapping_sha256(filtered),
    }


def strict_load_equivalent_students(
    p0_model: object,
    p1_model: object,
    filtered: Mapping[str, Any],
) -> dict[str, object]:
    """Strict-load both students and prove every loaded tensor is identical."""

    try:
        import torch
        from transvision.models.common_teacher_initialization import (
            tensor_mapping_sha256,
        )
    except ImportError as error:  # pragma: no cover - runtime dependency.
        raise P1CheckpointReuseError(
            "checkpoint audit dependencies are missing"
        ) from error
    for label, model in (("P0", p0_model), ("P1", p1_model)):
        loader = getattr(model, "load_state_dict", None)
        if not callable(loader):
            raise P1CheckpointReuseError(f"{label} model cannot load a state dict")
        result = loader(filtered, strict=True)
        missing = list(getattr(result, "missing_keys", ()))
        unexpected = list(getattr(result, "unexpected_keys", ()))
        if missing or unexpected:
            raise P1CheckpointReuseError(
                f"{label} strict load returned missing={missing!r}, "
                f"unexpected={unexpected!r}"
            )
    p0_state = p0_model.state_dict()
    p1_state = p1_model.state_dict()
    if list(p0_state) != list(p1_state):
        raise P1CheckpointReuseError("loaded P0/P1 key order differs")
    for key in p0_state:
        if not torch.equal(p0_state[key].detach().cpu(), p1_state[key].detach().cpu()):
            raise P1CheckpointReuseError(f"loaded P0/P1 tensor differs for {key!r}")
    p0_hash = tensor_mapping_sha256(p0_state)
    p1_hash = tensor_mapping_sha256(p1_state)
    if p0_hash != p1_hash:
        raise P1CheckpointReuseError("loaded P0/P1 tensor mapping hashes differ")
    return {
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "loaded_tensor_mapping_equal": True,
        "loaded_tensor_mapping_sha256": p0_hash,
    }


def _scrub_external_initialization(value: object) -> None:
    if isinstance(value, dict):
        if "init_cfg" in value:
            value["init_cfg"] = None
        for child in value.values():
            _scrub_external_initialization(child)
    elif isinstance(value, list):
        for child in value:
            _scrub_external_initialization(child)


def _build_models_for_audit() -> tuple[
    object, object, Mapping[str, Any], dict[str, object]
]:
    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        import mmdet3d.models  # noqa: F401
        from mmdet3d.registry import MODELS

        from transvision import register_all_modules
    except ImportError as error:  # pragma: no cover - A100 runtime dependency.
        raise P1CheckpointReuseError(
            "MMEngine/MMDetection3D runtime is unavailable"
        ) from error

    p0_path = (ROOT / P0_CONFIG).resolve(strict=True)
    p1_path = (ROOT / P1_CONFIG).resolve(strict=True)
    if _sha256_file(p0_path) != P0_CONFIG_SHA256:
        raise P1CheckpointReuseError("P0 config SHA-256 drifted")
    if _sha256_file(p1_path) != P1_CONFIG_SHA256:
        raise P1CheckpointReuseError("P1 config SHA-256 drifted")
    register_all_modules()
    p0_config = Config.fromfile(str(p0_path))
    p1_config = Config.fromfile(str(p1_path))
    init_default_scope(p0_config.get("default_scope", "mmdet3d"))
    config_contract = validate_config_subject_split(p0_config.model, p1_config.model)

    p0_full_config = copy.deepcopy(p0_config.model)
    _scrub_external_initialization(p0_full_config)
    p0_full = MODELS.build(p0_full_config)
    p0_full_state = p0_full.state_dict()
    nested_state = OrderedDict(
        (key, tensor.detach().cpu().clone())
        for key, tensor in p0_full_state.items()
        if key.startswith("teacher.teacher.")
    )
    if not nested_state:
        raise P1CheckpointReuseError("P0 full model has no nested frozen teacher")
    del p0_full

    def student(model_config: Mapping[str, object]) -> object:
        value = _strip_training_only_model_config(model_config)
        _scrub_external_initialization(value)
        return MODELS.build(value)

    return (
        student(p0_config.model),
        student(p1_config.model),
        nested_state,
        config_contract,
    )


def _target_a100_cuda_audit(p0_model: object, p1_model: object) -> dict[str, object]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - A100 runtime dependency.
        raise P1CheckpointReuseError("PyTorch is unavailable") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise P1CheckpointReuseError("P1 audit requires exactly four visible CUDA GPUs")
    capabilities = tuple(
        tuple(torch.cuda.get_device_capability(index))
        for index in range(torch.cuda.device_count())
    )
    if capabilities != REQUIRED_GPU_CAPABILITIES:
        raise P1CheckpointReuseError(
            f"P1 audit requires four A100-class sm80 GPUs: {capabilities!r}"
        )
    p0_model = p0_model.cuda().eval()
    p1_model = p1_model.cuda().eval()
    p0_router = p0_model.resilient_fusion.router
    p1_router = p1_model.resilient_fusion.router
    if p0_router.support_residual_reliability_gate is not False:
        raise P1CheckpointReuseError("loaded P0 router unexpectedly enables the gate")
    if p1_router.support_residual_reliability_gate is not True:
        raise P1CheckpointReuseError("loaded P1 router did not enable the gate")

    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    lidar = torch.randn(1, 256, 4, 4, device="cuda", requires_grad=True)
    camera = torch.randn(1, 256, 4, 4, device="cuda", requires_grad=True)
    support = torch.ones(1, 4, dtype=torch.bool, device="cuda")
    common = {
        "lidar_feature": lidar,
        "camera_feature": camera,
        "lidar_branch_support": support[:, :2],
        "camera_branch_support": support[:, 2:],
        "branch_observed": torch.tensor(
            [[True, False, True, False]], dtype=torch.bool, device="cuda"
        ),
        "branch_propagated": torch.tensor(
            [[False, True, False, True]], dtype=torch.bool, device="cuda"
        ),
        "branch_age_intervals": torch.tensor([[0.0, 1.0, 0.0, 2.0]], device="cuda"),
        "rsu_delay_intervals": torch.tensor([[1.0]], device="cuda"),
        "routing_mode": "uniform",
        "use_reliability": False,
        "use_delay_metadata": True,
    }
    clean = {
        **common,
        "branch_reliability": torch.ones(1, 4, device="cuda"),
    }
    with torch.no_grad():
        p0_clean = p0_router(**clean)
        p1_clean = p1_router(**clean)
    if not torch.equal(p0_clean.fused, p1_clean.fused):
        raise P1CheckpointReuseError("P0/P1 clean CUDA output is not bitwise equal")
    if not torch.equal(p0_clean.weights, p1_clean.weights):
        raise P1CheckpointReuseError("P0/P1 clean CUDA routing weights differ")

    fault = {
        **common,
        "branch_reliability": torch.tensor([[0.8, 0.4, 0.2, 0.0]], device="cuda"),
    }
    with torch.no_grad():
        p0_fault = p0_router(**fault)
    p1_fault = p1_router(**fault)
    if torch.equal(p0_fault.fused, p1_fault.fused):
        raise P1CheckpointReuseError("P1 reliability gate did not activate under fault")
    if not torch.isfinite(p1_fault.fused).all().item():
        raise P1CheckpointReuseError("P1 fault forward contains non-finite values")
    loss = p1_fault.fused.float().square().mean()
    loss.backward()
    for name, gradient in (("lidar", lidar.grad), ("camera", camera.grad)):
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise P1CheckpointReuseError(f"P1 fault {name} gradient is invalid")
    return {
        "visible_gpu_count": 4,
        "capabilities": [list(value) for value in capabilities],
        "clean_all_supported_reliability": 1.0,
        "clean_fused_bitwise_equal": True,
        "clean_routing_weights_bitwise_equal": True,
        "fault_reliability_vector": [0.8, 0.4, 0.2, 0.0],
        "fault_gate_activated": True,
        "fault_forward_finite": True,
        "fault_backward_finite": True,
    }


def _trusted_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - A100 runtime dependency.
        raise P1CheckpointReuseError("PyTorch is unavailable") from error
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older pinned PyTorch.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise P1CheckpointReuseError("P0 checkpoint is not an MMEngine mapping")
    return value


def run_checkpoint_reuse_equivalence_audit(
    checkpoint: Path,
    binding: Mapping[str, object],
) -> dict[str, object]:
    """Strict-load P0 weights into P0/P1 and execute the target A100 audit."""

    binding_seal = _require_seal(binding, context="checkpoint reuse binding")
    alias_mapping_seal = validate_identity_alias_mapping(identity_alias_mapping())
    if binding.get("identity_alias_mapping_seal_sha256") != alias_mapping_seal:
        raise P1CheckpointReuseError("checkpoint binding identity alias seal mismatch")
    checkpoint_contract = binding.get("checkpoint")
    if not isinstance(checkpoint_contract, Mapping):
        raise P1CheckpointReuseError("checkpoint reuse binding lacks checkpoint")
    path = Path(checkpoint)
    if path.is_symlink():
        raise P1CheckpointReuseError("checkpoint audit input is a symlink")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise P1CheckpointReuseError("checkpoint audit input is missing") from error
    expected_bytes = _positive_int(
        checkpoint_contract.get("size_bytes"), context="bound checkpoint bytes"
    )
    expected_sha = _sha256(
        checkpoint_contract.get("sha256"), context="bound checkpoint"
    )
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise P1CheckpointReuseError("checkpoint audit input byte size mismatch")
    if _sha256_file(path) != expected_sha:
        raise P1CheckpointReuseError("checkpoint audit input SHA-256 mismatch")

    p0_model, p1_model, nested_state, config_contract = _build_models_for_audit()
    checkpoint_value = _trusted_torch_load(path)
    full_state = OrderedDict(p0_model.state_dict())
    full_state.update(nested_state)
    filtered, schema_contract = prepare_filtered_student_state(
        checkpoint_value,
        p0_student_state=p0_model.state_dict(),
        p1_student_state=p1_model.state_dict(),
        p0_full_state=full_state,
    )
    load_contract = strict_load_equivalent_students(p0_model, p1_model, filtered)
    cuda_contract = _target_a100_cuda_audit(p0_model, p1_model)
    return _sealed(
        {
            "schema_version": 1,
            "document_type": "resilient_v2x_p1_checkpoint_reuse_equivalence",
            "passed": True,
            "candidate_identity": ZERO_SHOT_CANDIDATE_IDENTITY,
            "identity_alias_mapping_seal_sha256": alias_mapping_seal,
            "evidence_class": ZERO_SHOT_EVIDENCE_CLASS,
            "weights_retrained": False,
            "optimization_origin": OPTIMIZATION_ORIGIN,
            "config_subject": P1_CONFIG_SUBJECT,
            "checkpoint_subject": P0_CONFIG_SUBJECT,
            "checkpoint_reuse_binding_seal_sha256": binding_seal,
            "checkpoint": {
                "task_id": P0_TASK_ID,
                "model_id": checkpoint_contract["model_id"],
                "size_bytes": expected_bytes,
                "sha256": expected_sha,
                "bytes_recomputed": True,
            },
            "config_equivalence": config_contract,
            "state_schema": schema_contract,
            "strict_load": load_contract,
            "target_cuda": cuda_contract,
            "formal_protocol": {
                "protocol_id": PROTOCOL_ID,
                "sample_count": SAMPLE_COUNT,
                "training_seed": TRAINING_SEED,
                "overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
                "run_count": RUN_COUNT,
            },
        }
    )


def evaluation_parameters(
    binding: Mapping[str, object],
) -> dict[str, object]:
    """Build the split-subject parameters for a dedicated eval-only task."""

    binding_seal = _require_seal(binding, context="checkpoint reuse binding")
    alias_mapping_seal = validate_identity_alias_mapping(identity_alias_mapping())
    if binding.get("identity_alias_mapping_seal_sha256") != alias_mapping_seal:
        raise P1CheckpointReuseError("checkpoint binding identity alias seal mismatch")
    checkpoint = binding.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise P1CheckpointReuseError("checkpoint reuse binding lacks checkpoint")
    return {
        "Args/stage": "checkpoint_reuse_validate",
        "Args/controlled_baseline": ZERO_SHOT_CANDIDATE_IDENTITY,
        "Args/candidate_identity_alias_mapping_sha256": alias_mapping_seal,
        "Args/controlled_baseline_config_subject": P1_CONFIG_SUBJECT,
        "Args/controlled_baseline_checkpoint_subject": P0_CONFIG_SUBJECT,
        "Args/controlled_baseline_task_id": P0_TASK_ID,
        "Args/controlled_baseline_model_id": checkpoint["model_id"],
        "Args/controlled_baseline_checkpoint_sha256": checkpoint["sha256"],
        "Args/predecessor_task_id": P0_TASK_ID,
        "Args/protocol_id": PROTOCOL_ID,
        "Args/sample_count": SAMPLE_COUNT,
        "Args/training_seed": TRAINING_SEED,
        "Args/training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "Args/weights_retrained": False,
        "Args/optimization_origin": OPTIMIZATION_ORIGIN,
        "Args/checkpoint_reuse_plan_sha256": binding_seal,
        "Args/training_dataset_id": TRAINING_DATASET_ID,
        "Args/gpus": 4,
        "Args/max_epochs": 50,
        "Args/amp": False,
    }


def validate_evaluation_parameters(
    parameters: Mapping[str, object],
    binding: Mapping[str, object],
) -> None:
    _require_parameters(
        parameters,
        evaluation_parameters(binding),
        context="P1 zero-shot evaluation",
    )
    if parameters.get("Args/controlled_baseline_config_subject") == parameters.get(
        "Args/controlled_baseline_checkpoint_subject"
    ):
        raise P1CheckpointReuseError("config subject and checkpoint owner were merged")


def _upload_and_readback(
    task: object,
    name: str,
    value: Mapping[str, object],
) -> None:
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader) or not uploader(
        name,
        artifact_object=dict(value),
        wait_on_upload=True,
    ):
        raise P1CheckpointReuseError(f"failed to upload {name!r}")
    flusher = getattr(task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)
    _reload(task, context="P1 zero-shot evaluation task")
    observed = _artifact_mapping(
        task,
        name,
        context="P1 zero-shot evaluation task",
    )
    if _canonical_json(observed) != _canonical_json(value):
        raise P1CheckpointReuseError(f"uploaded artifact {name!r} readback drifted")


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("checkpoint_reuse_validate",), required=True
    )
    parser.add_argument("--controlled-baseline", required=True)
    parser.add_argument("--candidate-identity-alias-mapping-sha256", required=True)
    parser.add_argument("--controlled-baseline-config-subject", required=True)
    parser.add_argument("--controlled-baseline-checkpoint-subject", required=True)
    parser.add_argument("--controlled-baseline-task-id", required=True)
    parser.add_argument("--controlled-baseline-model-id", required=True)
    parser.add_argument("--controlled-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--predecessor-task-id", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--training-overlay-protocol-seed", type=int, required=True)
    parser.add_argument("--weights-retrained", action="store_true")
    parser.add_argument("--optimization-origin", required=True)
    parser.add_argument("--checkpoint-reuse-plan-sha256", required=True)
    parser.add_argument("--training-dataset-id", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--amp", action="store_true")
    return parser


def _namespace_parameters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "Args/stage": args.stage,
        "Args/controlled_baseline": args.controlled_baseline,
        "Args/candidate_identity_alias_mapping_sha256": (
            args.candidate_identity_alias_mapping_sha256
        ),
        "Args/controlled_baseline_config_subject": (
            args.controlled_baseline_config_subject
        ),
        "Args/controlled_baseline_checkpoint_subject": (
            args.controlled_baseline_checkpoint_subject
        ),
        "Args/controlled_baseline_task_id": args.controlled_baseline_task_id,
        "Args/controlled_baseline_model_id": args.controlled_baseline_model_id,
        "Args/controlled_baseline_checkpoint_sha256": (
            args.controlled_baseline_checkpoint_sha256
        ),
        "Args/predecessor_task_id": args.predecessor_task_id,
        "Args/protocol_id": args.protocol_id,
        "Args/sample_count": args.sample_count,
        "Args/training_seed": args.training_seed,
        "Args/training_overlay_protocol_seed": args.training_overlay_protocol_seed,
        "Args/weights_retrained": args.weights_retrained,
        "Args/optimization_origin": args.optimization_origin,
        "Args/checkpoint_reuse_plan_sha256": args.checkpoint_reuse_plan_sha256,
        "Args/training_dataset_id": args.training_dataset_id,
        "Args/gpus": args.gpus,
        "Args/max_epochs": args.max_epochs,
        "Args/amp": args.amp,
    }


def run_worker(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    dataset_class: object = Dataset,
) -> dict[str, object]:
    """Execute the audited P1/P0 checkpoint-reuse evaluation on A100."""

    if task_class is None or dataset_class is None:
        raise P1CheckpointReuseError("ClearML runtime is unavailable")
    current_getter = getattr(task_class, "current_task", None)
    current = current_getter() if callable(current_getter) else None
    if current is None:
        raise P1CheckpointReuseError("run-worker requires a current ClearML task")
    current_id = _task_id(getattr(current, "id", None), context="evaluation task")
    if _task_parent(current) != P0_TASK_ID:
        raise P1CheckpointReuseError(
            "P1 zero-shot evaluation task parent must be the P0 training task"
        )
    p0_task = task_class.get_task(task_id=P0_TASK_ID)
    alias_mapping = identity_alias_mapping()
    alias_mapping_seal = validate_identity_alias_mapping(alias_mapping)
    binding = resolve_p0_canonical_final(p0_task, verify_checkpoint_bytes=True)
    validate_evaluation_parameters(_namespace_parameters(args), binding)
    if args.checkpoint_reuse_plan_sha256 != binding["seal_sha256"]:
        raise P1CheckpointReuseError("checkpoint reuse plan pin mismatch")
    if (
        args.candidate_identity_alias_mapping_sha256 != alias_mapping_seal
        or binding.get("identity_alias_mapping_seal_sha256") != alias_mapping_seal
    ):
        raise P1CheckpointReuseError("worker identity alias mapping pin mismatch")
    checkpoint_contract = binding["checkpoint"]
    assert isinstance(checkpoint_contract, Mapping)
    if (
        args.controlled_baseline_model_id != checkpoint_contract["model_id"]
        or args.controlled_baseline_checkpoint_sha256 != checkpoint_contract["sha256"]
    ):
        raise P1CheckpointReuseError("worker checkpoint pin differs from resolver")
    _upload_and_readback(current, IDENTITY_ALIAS_MAPPING_ARTIFACT, alias_mapping)
    _upload_and_readback(current, CHECKPOINT_REUSE_BINDING_ARTIFACT, binding)

    local_checkpoint = download_bound_p0_checkpoint(p0_task, binding)
    local_path = str(local_checkpoint)
    equivalence = run_checkpoint_reuse_equivalence_audit(local_checkpoint, binding)
    _upload_and_readback(
        current,
        CHECKPOINT_REUSE_EQUIVALENCE_ARTIFACT,
        equivalence,
    )

    setter = getattr(current, "set_input_model", None)
    if not callable(setter):
        raise P1CheckpointReuseError("evaluation task cannot bind an input model")
    setter(
        model_id=str(checkpoint_contract["model_id"]),
        name="p0_canonical_epoch50_final_checkpoint",
        update_task_design=False,
        update_task_labels=False,
    )

    from tools.resilient_v2x import clearml_5090_bootstrap as bootstrap

    source_root = ROOT
    runner = bootstrap._load_source_training_runner(source_root)
    dataset_root, runtime_env = bootstrap._prepare_experiment_environment(
        args,
        task_id=current_id,
        source_root=source_root,
        base_env=os.environ.copy(),
        dataset_class=dataset_class,
        runner=runner,
    )
    work_dir = (
        source_root
        / "work_dirs/p1_checkpoint_reuse_evaluation"
        / current_id
        / P1_CONFIG_SUBJECT
    )
    if work_dir.exists() or work_dir.is_symlink():
        raise P1CheckpointReuseError("refusing to reuse P1 evaluation work directory")
    overlay_index = (
        dataset_root / "protocols/dair_v2/evaluation_overlays.json"
    ).resolve(strict=True)
    evaluator = (source_root / "tools/resilient_v2x/evaluate_p1_zero_shot.py").resolve(
        strict=True
    )
    python = Path(sys.executable).resolve(strict=True)
    command = [
        str(python),
        str(evaluator),
        "--baseline",
        P1_CONFIG_SUBJECT,
        "--checkpoint",
        local_path,
        "--overlay-index",
        str(overlay_index),
        "--work-dir",
        str(work_dir),
        "--protocol-id",
        PROTOCOL_ID,
        "--expected-ground-truth-count",
        str(GROUND_TRUTH_COUNT),
    ]
    run_contract = _sealed(
        {
            "schema_version": 1,
            "mode": "checkpoint_reuse_validate",
            "task_id": current_id,
            "candidate_identity": ZERO_SHOT_CANDIDATE_IDENTITY,
            "identity_alias_mapping_seal_sha256": alias_mapping_seal,
            "evidence_class": ZERO_SHOT_EVIDENCE_CLASS,
            "config_subject": P1_CONFIG_SUBJECT,
            "checkpoint_subject": P0_CONFIG_SUBJECT,
            "checkpoint_reuse_binding_seal_sha256": binding["seal_sha256"],
            "checkpoint_reuse_equivalence_seal_sha256": equivalence["seal_sha256"],
            "weights_retrained": False,
            "optimization_origin": OPTIMIZATION_ORIGIN,
            "predecessor_task_id": P0_TASK_ID,
            "training_dataset_id": TRAINING_DATASET_ID,
            "protocol_id": PROTOCOL_ID,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "training_seed": TRAINING_SEED,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "delays_ms": list(DELAYS_MS),
            "conditions": list(CONDITIONS),
            "run_count": RUN_COUNT,
            "checkpoint": {
                key: checkpoint_contract[key]
                for key in (
                    "model_id",
                    "model_name",
                    "model_url",
                    "source_filename",
                    "remote_filename",
                    "size_bytes",
                    "sha256",
                )
            },
            "evaluator": str(evaluator),
            "evaluator_sha256": _sha256_file(evaluator),
            "command": command,
        }
    )
    _upload_and_readback(current, RUN_CONTRACT_ARTIFACT, run_contract)
    bootstrap._run_logged(command, cwd=source_root, env=runtime_env)
    evidence_stage = bootstrap._stage_controlled_baseline_evidence(work_dir=work_dir)
    _, metrics = bootstrap._verify_controlled_evidence_stage(evidence_stage)
    from tools.resilient_v2x import clearml_formal_candidate_selector as formal

    formal._parse_metrics(
        metrics,
        subject=P1_CONFIG_SUBJECT,
        checkpoint_sha256=str(checkpoint_contract["sha256"]),
    )
    bootstrap._upload_controlled_baseline_artifacts(
        current,
        evidence_stage=evidence_stage,
    )
    receipt = _sealed(
        {
            "schema_version": 1,
            "document_type": "resilient_v2x_p1_checkpoint_reuse_evaluation_receipt",
            "task_id": current_id,
            "candidate_identity": ZERO_SHOT_CANDIDATE_IDENTITY,
            "identity_alias_mapping_seal_sha256": alias_mapping_seal,
            "config_subject": P1_CONFIG_SUBJECT,
            "checkpoint_subject": P0_CONFIG_SUBJECT,
            "checkpoint_reuse_binding_seal_sha256": binding["seal_sha256"],
            "checkpoint_reuse_equivalence_seal_sha256": equivalence["seal_sha256"],
            "run_contract_seal_sha256": run_contract["seal_sha256"],
            "metrics_sha256": _content_sha256(metrics),
            "complete": True,
            "run_count": RUN_COUNT,
            "sample_count_per_run": SAMPLE_COUNT,
            "weights_retrained": False,
            "optimization_origin": OPTIMIZATION_ORIGIN,
        }
    )
    _upload_and_readback(
        current,
        CHECKPOINT_REUSE_EVALUATION_RECEIPT_ARTIFACT,
        receipt,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "show-contract",
        help="Print the immutable local candidate/protocol contract.",
    )
    subparsers.add_parser(
        "resolve-p0",
        help=(
            "Read and verify completed P0 final checkpoint provenance, then print "
            "the machine-independent launch parameters without remote writes."
        ),
    )
    worker = subparsers.add_parser(
        "run-worker",
        help="Run the dedicated ClearML A100 evaluation worker.",
    )
    for action in _worker_parser()._actions:
        if not action.option_strings or action.dest == "help":
            continue
        kwargs: dict[str, object] = {
            "dest": action.dest,
            "required": action.required,
            "default": action.default,
            "help": action.help,
        }
        if action.const is True and action.nargs == 0:
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = action.type
            if action.choices is not None:
                kwargs["choices"] = action.choices
        worker.add_argument(*action.option_strings, **kwargs)
    return parser


def local_execution_contract() -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 1,
            "contract_type": "resilient_v2x_p1_zero_shot_candidate_extension",
            "candidate_identity": ZERO_SHOT_CANDIDATE_IDENTITY,
            "display_subject_alias": ZERO_SHOT_DISPLAY_SUBJECT_ALIAS,
            "identity_alias_mapping_seal_sha256": identity_alias_mapping()[
                "seal_sha256"
            ],
            "distinct_from_trained_candidate": P1_TRAINED_CANDIDATE_IDENTITY,
            "evidence_class": ZERO_SHOT_EVIDENCE_CLASS,
            "weights_retrained": False,
            "optimization_origin": OPTIMIZATION_ORIGIN,
            "config_subject": P1_CONFIG_SUBJECT,
            "checkpoint_subject": P0_CONFIG_SUBJECT,
            "checkpoint_owner_task_id": P0_TASK_ID,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "checkpoint_role": CHECKPOINT_ROLE,
            "worker_queue": WORKER_QUEUE,
            "worker_queue_id": WORKER_QUEUE_ID,
            "required_gpu_capabilities": [
                list(value) for value in REQUIRED_GPU_CAPABILITIES
            ],
            "protocol_id": PROTOCOL_ID,
            "sample_count": SAMPLE_COUNT,
            "training_seed": TRAINING_SEED,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "run_count": RUN_COUNT,
            "original_26_method_chain_mutated": False,
            "result_destination": "candidate_extension_leaderboard_only",
        }
    )


def resolved_launch_contract(task: object) -> dict[str, object]:
    """Build a sealed, machine-independent pre-launch contract read-only."""

    alias_mapping = identity_alias_mapping()
    binding = resolve_p0_canonical_final(task, verify_checkpoint_bytes=True)
    if (
        binding.get("identity_alias_mapping_seal_sha256")
        != alias_mapping["seal_sha256"]
    ):
        raise P1CheckpointReuseError("resolved binding identity alias seal mismatch")
    return _sealed(
        {
            "schema_version": 1,
            "contract_type": "resilient_v2x_p1_zero_shot_resolved_launch",
            "remote_state_changed": False,
            "identity_alias_mapping": alias_mapping,
            "checkpoint_reuse_binding": binding,
            "evaluation_parameters": evaluation_parameters(binding),
        }
    )


def main() -> int:
    args = _parser().parse_args()
    if args.command == "show-contract":
        print(_canonical_json(local_execution_contract()))
        return 0
    if args.command == "resolve-p0":
        if Task is None:
            raise P1CheckpointReuseError("ClearML runtime is unavailable")
        print(
            _canonical_json(resolved_launch_contract(Task.get_task(task_id=P0_TASK_ID)))
        )
        return 0
    if args.command == "run-worker":
        run_worker(args)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CHECKPOINT_REUSE_BINDING_ARTIFACT",
    "CHECKPOINT_REUSE_EQUIVALENCE_ARTIFACT",
    "CHECKPOINT_REUSE_EVALUATION_RECEIPT_ARTIFACT",
    "IDENTITY_ALIAS_MAPPING_ARTIFACT",
    "P0_CONFIG_SUBJECT",
    "P0_TASK_ID",
    "P1_CONFIG_SUBJECT",
    "ZERO_SHOT_CANDIDATE_IDENTITY",
    "ZERO_SHOT_DISPLAY_SUBJECT_ALIAS",
    "evaluation_parameters",
    "download_bound_p0_checkpoint",
    "identity_alias_mapping",
    "local_execution_contract",
    "prepare_filtered_student_state",
    "resolve_p0_canonical_final",
    "resolved_launch_contract",
    "run_checkpoint_reuse_equivalence_audit",
    "run_worker",
    "strict_load_equivalent_students",
    "validate_config_subject_split",
    "validate_evaluation_parameters",
    "validate_identity_alias_mapping",
)
