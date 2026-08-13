#!/usr/bin/env python3
"""Build a sealed, local-only dependency plan for post-winner paper experiments.

The plan is intentionally a planning artifact: it never creates ClearML tasks,
enqueues work, trains a model, or executes an evaluator.  It binds the selected
method identity and local final checkpoint, audits reusable single-factor
ablations, and emits the exact dependency graph needed for Tables V--VII plus
the E-only/R-only diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import runpy
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_TYPE = "resilient_v2x_post_winner_experiment_plan"
SELECTED_IDENTITY_TYPE = "resilient_v2x_sealed_selected_method_identity"
PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
TRAINING_SEED = 20250218
CHECKPOINT_POLICY = "epoch_50_final_only"
SAMPLE_COUNT = 1337
GROUND_TRUTH_COUNT = 11330
UNSUPPORTED_SAMPLE_COUNT = 0
SAMPLE_IDS_SHA256 = (
    "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
)
SPLIT_SHA256 = (
    "d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
)
MAIN_TRAINING_PROBABILITY = 0.2
SENSITIVITY_PROBABILITIES = (0.0, 0.1, 0.3, 0.5)
TRAINING_CONTRACT = {
    "training_seed": TRAINING_SEED,
    "precision": "FP32",
    "gpu_count": 4,
    "train_batch_size_per_gpu": 2,
    "global_batch_size": 8,
    "max_epochs": 50,
    "val_interval": 10,
    "checkpoint_policy": CHECKPOINT_POLICY,
}


class PostWinnerPlanError(RuntimeError):
    """Raised when a post-winner plan cannot be bound without guessing."""


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
        raise PostWinnerPlanError(f"value is not canonical JSON: {error}") from error


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = value.get("seal_sha256")
    if type(observed) is not str or len(observed) != 64:
        raise PostWinnerPlanError(f"{context} seal is invalid")
    if _seal(value)["seal_sha256"] != observed:
        raise PostWinnerPlanError(f"{context} seal mismatch")
    return observed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PostWinnerPlanError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _read_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PostWinnerPlanError(f"cannot read {context}: {error}") from error
    if not isinstance(value, dict):
        raise PostWinnerPlanError(f"{context} must be a JSON object")
    return value


def _verify_content_document(value: Mapping[str, object], *, context: str) -> str:
    observed = value.get("content_sha256")
    if type(observed) is not str or len(observed) != 64:
        raise PostWinnerPlanError(f"{context} content_sha256 is invalid")
    payload = dict(value)
    payload.pop("content_sha256", None)
    if _content_sha256(payload) != observed:
        raise PostWinnerPlanError(f"{context} content_sha256 mismatch")
    return observed


def _safe_repo_path(value: object, *, context: str) -> tuple[str, Path]:
    if type(value) is not str:
        raise PostWinnerPlanError(f"{context} must be a repository-relative path")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or not logical.parts
        or logical.parts[0] != "configs"
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise PostWinnerPlanError(f"{context} is not a safe config path")
    local = (ROOT / Path(*logical.parts)).resolve()
    try:
        local.relative_to(ROOT.resolve())
    except ValueError as error:
        raise PostWinnerPlanError(f"{context} escapes the repository") from error
    if not local.is_file():
        raise PostWinnerPlanError(f"{context} does not exist: {local}")
    return value, local


def _validate_selected_identity(
    value: Mapping[str, object], *, checkpoint: Path
) -> dict[str, object]:
    seal = _require_seal(value, context="selected-method identity")
    if (
        value.get("schema_version") != 2
        or value.get("document_type") != SELECTED_IDENTITY_TYPE
        or value.get("status") != "sealed"
        or value.get("training_seed") != TRAINING_SEED
        or value.get("checkpoint_policy") != CHECKPOINT_POLICY
    ):
        raise PostWinnerPlanError("selected-method identity contract drifted")
    selected_subject = value.get("selected_subject")
    if type(selected_subject) is not str or not selected_subject:
        raise PostWinnerPlanError("selected-method identity has no selected subject")
    gate = value.get("gate_result")
    if not isinstance(gate, Mapping) or gate.get("gate_passed") is not True:
        raise PostWinnerPlanError("selected method did not pass the controlled gate")
    binding = value.get("method_binding")
    if not isinstance(binding, Mapping):
        raise PostWinnerPlanError("selected-method binding is missing")
    config_path, local_config = _safe_repo_path(
        binding.get("config_path"), context="selected config_path"
    )
    config_sha = binding.get("config_sha256")
    if type(config_sha) is not str or len(config_sha) != 64:
        raise PostWinnerPlanError("selected config SHA-256 is invalid")
    if _sha256_file(local_config) != config_sha:
        raise PostWinnerPlanError(
            "local selected config differs from the sealed selected-method identity"
        )
    checkpoint_path = checkpoint.expanduser().resolve(strict=True)
    if not checkpoint_path.is_file():
        raise PostWinnerPlanError("selected checkpoint must be a file")
    checkpoint_sha = binding.get("checkpoint_sha256")
    checkpoint_size = binding.get("checkpoint_size_bytes")
    if type(checkpoint_sha) is not str or len(checkpoint_sha) != 64:
        raise PostWinnerPlanError("selected checkpoint SHA-256 is invalid")
    if type(checkpoint_size) is not int or checkpoint_size <= 0:
        raise PostWinnerPlanError("selected checkpoint size is invalid")
    if checkpoint_path.stat().st_size != checkpoint_size:
        raise PostWinnerPlanError("selected checkpoint size differs from identity")
    if _sha256_file(checkpoint_path) != checkpoint_sha:
        raise PostWinnerPlanError("selected checkpoint hash differs from identity")
    return {
        "selected_subject": selected_subject,
        "selected_candidate_label": value.get("selected_candidate_label"),
        "identity_seal_sha256": seal,
        "identity_content_sha256": _content_sha256(value),
        "config_path": config_path,
        "config_sha256": config_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": checkpoint_size,
        "training_task_id": binding.get("training_task_id"),
        "evaluation_task_id": binding.get("evaluation_task_id"),
        "model_id": binding.get("model_id"),
        "training_dataset_id": binding.get("training_dataset_id"),
        "teacher_task_id": binding.get("teacher_task_id"),
        "teacher_model_id": binding.get("teacher_model_id"),
        "teacher_checkpoint_sha256": binding.get("teacher_checkpoint_sha256"),
    }


def _merge(base: object, update: object) -> object:
    if isinstance(update, dict) and update.get("_delete_") is True:
        return {
            key: copy.deepcopy(item)
            for key, item in update.items()
            if key != "_delete_"
        }
    if not isinstance(base, dict) or not isinstance(update, dict):
        return copy.deepcopy(update)
    result = copy.deepcopy(base)
    for key, item in update.items():
        if key != "_delete_":
            result[key] = _merge(result.get(key), item)
    return result


def _load_config(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, object]:
    source = path.resolve()
    if source in stack:
        raise PostWinnerPlanError(f"cyclic config inheritance at {source}")
    try:
        namespace = runpy.run_path(str(source))
    except (OSError, RuntimeError, ValueError) as error:
        raise PostWinnerPlanError(f"cannot load config {source}: {error}") from error
    bases = namespace.get("_base_", ())
    if isinstance(bases, str):
        bases = (bases,)
    if not isinstance(bases, (tuple, list)) or any(type(item) is not str for item in bases):
        raise PostWinnerPlanError(f"invalid _base_ in {source}")
    result: dict[str, object] = {}
    for base in bases:
        result = _merge(
            result, _load_config(source.parent / str(base), (*stack, source))
        )  # type: ignore[assignment]
    own = {key: item for key, item in namespace.items() if not key.startswith("_")}
    return _merge(result, own)  # type: ignore[return-value]


def _semantic_config(value: Mapping[str, object]) -> dict[str, object]:
    result = copy.deepcopy(dict(value))
    for key in tuple(result):
        if key == "experiment" or key.startswith("implementation_choices_"):
            result.pop(key, None)
    return result


def _diff_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: set[str] = set()
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if _canonical_json(list(left)) == _canonical_json(list(right)):
            return set()
        return {prefix}
    return set() if left == right else {prefix}


def _model_value(config: Mapping[str, object], key: str, default: object) -> object:
    model = config.get("model")
    return model.get(key, default) if isinstance(model, Mapping) else default


def _ablation_specs(winner: Mapping[str, object]) -> list[dict[str, object]]:
    ptf_mode = _model_value(winner, "ptf_mode", "nonlinear")
    alternative = "linear" if ptf_mode == "nonlinear" else "nonlinear"
    specs: list[dict[str, object]] = [
        {
            "id": "no_ptf",
            "existing_subject": "ptf_none",
            "concept": "PTF",
            "overrides": {"model.ptf_mode": "none"},
            "expected_difference_paths": ["model.ptf_mode"],
        },
        {
            "id": "alternative_ptf",
            "existing_subject": "ptf_linear" if alternative == "linear" else None,
            "concept": "PTF formulation",
            "overrides": {"model.ptf_mode": alternative},
            "expected_difference_paths": ["model.ptf_mode"],
        },
        {
            "id": "router_static",
            "existing_subject": "router_static",
            "concept": "routing rule",
            "overrides": {"model.routing_mode": "static"},
            "expected_difference_paths": ["model.routing_mode"],
        },
        {
            "id": "router_uniform",
            "existing_subject": "router_uniform",
            "concept": "routing rule",
            "overrides": {"model.routing_mode": "uniform"},
            "expected_difference_paths": ["model.routing_mode"],
        },
        {
            "id": "no_reliability",
            "existing_subject": "no_reliability",
            "concept": "routing reliability descriptor",
            "overrides": {"model.use_reliability": False},
            "expected_difference_paths": ["model.use_reliability"],
        },
        {
            "id": "no_delay_metadata",
            "existing_subject": "no_delay_metadata",
            "concept": "delay and age metadata",
            "overrides": {"model.use_delay_metadata": False},
            "expected_difference_paths": ["model.use_delay_metadata"],
        },
        {
            "id": "no_distillation",
            "existing_subject": "no_distillation",
            "concept": "distillation",
            "overrides": {
                "model.teacher": None,
                "model.teacher_checkpoint": None,
                "model.distillation": None,
                "train_dataloader.dataset.include_clean_teacher": False,
            },
            "expected_difference_paths": [
                "model.distillation",
                "model.teacher",
                "model.teacher_checkpoint",
                "train_dataloader.dataset.include_clean_teacher",
            ],
        },
    ]
    residual_weight = _model_value(winner, "support_residual_weight", 0.0)
    gated = _model_value(winner, "support_residual_reliability_gate", False)
    if (
        type(residual_weight) in {int, float}
        and math.isfinite(float(residual_weight))
        and float(residual_weight) != 0.0
    ):
        removal_paths = ["model.support_residual_weight"]
        overrides: dict[str, object] = {"model.support_residual_weight": 0.0}
        if gated is True:
            removal_paths.append("model.support_residual_reliability_gate")
            overrides["model.support_residual_reliability_gate"] = False
        specs.append(
            {
                "id": "winner_residual_module_removal",
                "existing_subject": None,
                "concept": "winner-added residual module",
                "overrides": overrides,
                "expected_difference_paths": sorted(removal_paths),
            }
        )
    return specs


def _get_dotted(value: Mapping[str, object], path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _existing_ablation_audit(
    winner: Mapping[str, object], spec: Mapping[str, object]
) -> dict[str, object]:
    overrides = spec["overrides"]
    assert isinstance(overrides, Mapping)
    no_op = all(_get_dotted(winner, path) == desired for path, desired in overrides.items())
    existing = spec.get("existing_subject")
    if no_op:
        return {
            "status": "not_applicable_winner_already_has_setting",
            "reusable": False,
            "reason": "the declared ablation is a no-op for the selected winner",
            "observed_difference_paths": [],
        }
    if type(existing) is not str:
        return {
            "status": "winner_derived_training_required",
            "reusable": False,
            "reason": "no pre-trained exact winner-derived variant is declared",
            "observed_difference_paths": [],
        }
    candidate_path = ROOT / "configs" / "resilient_v2x" / "ablations" / f"{existing}.py"
    candidate = _semantic_config(_load_config(candidate_path))
    observed = sorted(_diff_paths(winner, candidate))
    expected = sorted(spec["expected_difference_paths"])
    if observed == expected:
        return {
            "status": "reuse_candidate_pending_checkpoint_provenance",
            "reusable": True,
            "reason": "existing config differs from the winner only in the declared concept",
            "observed_difference_paths": observed,
            "existing_config_path": str(candidate_path.relative_to(ROOT)),
            "existing_config_sha256": _sha256_file(candidate_path),
        }
    return {
        "status": "winner_derived_training_required",
        "reusable": False,
        "reason": "existing config differs from the winner outside the declared change",
        "observed_difference_paths": observed,
    }


def _validate_training_index(value: Mapping[str, object]) -> dict[str, object]:
    digest = _verify_content_document(value, context="training overlay index")
    expected = {
        "artifact_type": "resilient_v2x_training_overlays",
        "schema_version": 1,
        "split": "train",
        "protocol_seed": TRAINING_SEED,
        "p_lidar": MAIN_TRAINING_PROBABILITY,
        "p_camera": MAIN_TRAINING_PROBABILITY,
        "epochs": list(range(50)),
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise PostWinnerPlanError(f"training overlay index {key} drifted")
    sample_ids = value.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise PostWinnerPlanError("training overlay sample inventory is invalid")
    return {
        "content_sha256": digest,
        "temporal_manifest_sha256": value.get("temporal_manifest_sha256"),
        "sample_count": len(sample_ids),
        "sample_ids_sha256": value.get("sample_ids_sha256"),
        "p_lidar": MAIN_TRAINING_PROBABILITY,
        "p_camera": MAIN_TRAINING_PROBABILITY,
    }


def _validate_evaluation_index(value: Mapping[str, object]) -> dict[str, object]:
    digest = _verify_content_document(value, context="evaluation overlay index")
    expected = {
        "artifact_type": "resilient_v2x_evaluation_overlays",
        "schema_version": 1,
        "split": "val",
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "delays_ms": [0, 100, 200, 300],
        "conditions": ["Full", "L-Fail", "C-Fail"],
        "agent_scopes": ["E+R", "E-only", "R-only"],
        "requested_duration": 1,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise PostWinnerPlanError(f"evaluation overlay index {key} drifted")
    sample_ids = value.get("sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != SAMPLE_COUNT:
        raise PostWinnerPlanError("evaluation overlay index is not the 1337 cohort")
    faults = value.get("fault_overlays")
    if not isinstance(faults, list) or len(faults) != 36:
        raise PostWinnerPlanError("evaluation overlay index does not contain 36 faults")
    return {
        "content_sha256": digest,
        "temporal_manifest_sha256": value.get("temporal_manifest_sha256"),
        "cohort_content_sha256": value.get("cohort_content_sha256"),
        "sample_count": SAMPLE_COUNT,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
    }


def _checkpoint_archive(subject: str) -> dict[str, object] | None:
    manifest_path = ROOT / "artifacts" / "trained_models" / "completed-live" / subject / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path, context=f"{subject} checkpoint manifest")
    if manifest.get("schema_version") != 1 or manifest.get("subject") != subject:
        return None
    models = manifest.get("models")
    if not isinstance(models, list):
        return None
    final = next(
        (
            item
            for item in models
            if isinstance(item, Mapping) and item.get("role") == "canonical_final"
        ),
        None,
    )
    if not isinstance(final, Mapping):
        return None
    filename = final.get("filename")
    if type(filename) is not str or PurePosixPath(filename).name != filename:
        return None
    checkpoint = manifest_path.parent / filename
    if not checkpoint.is_file():
        return None
    if checkpoint.stat().st_size != final.get("size_bytes"):
        return None
    if _sha256_file(checkpoint) != final.get("sha256"):
        return None
    return {
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": final.get("sha256"),
        "checkpoint_size_bytes": final.get("size_bytes"),
        "training_task_id": manifest.get("source_task_id"),
        "model_id": final.get("model_id"),
        "local_integrity_verified": True,
        "paper_evidence_eligible": False,
        "paper_evidence_blocker": (
            "must also match sealed run contract, teacher audit, final checkpoint "
            "contract, sealed source, and evaluator evidence"
        ),
    }


def _paper_interface_audit(paper_root: Path) -> dict[str, object]:
    registry_path = paper_root / "results" / "registry.json"
    contract_path = paper_root / "results" / "submission_contract.json"
    if not registry_path.is_file() or not contract_path.is_file():
        return {
            "status": "missing",
            "paper_root": str(paper_root),
            "missing_files": [str(registry_path), str(contract_path)],
        }
    registry = _read_json(registry_path, context="paper result registry")
    records = registry.get("results")
    if not isinstance(records, list):
        raise PostWinnerPlanError("paper registry results are invalid")
    records_by_id = {
        str(item["id"]): item
        for item in records
        if isinstance(item, Mapping) and type(item.get("id")) is str
    }
    ids = set(records_by_id)
    required_existing = {
        *(
            f"ours_{prefix}.{suffix}"
            for prefix in (
                "capacity_matched_full",
                "no_ptf",
                "linear_ptf",
                "static_three_expert",
                "uniform_gate",
                "no_reliability",
                "no_delay_metadata",
                "no_distillation",
            )
            for suffix in (
                "full.bev_ap_07",
                "l_fail.bev_ap_07",
                "full.latency_300.bev_ap_07",
            )
        ),
        *(f"ours_p_{str(p).replace('.', '_')}.{suffix}" for p in (0.0, 0.1, 0.3, 0.5) for suffix in ("full.bev_ap_07", "l_fail.bev_ap_07", "full.latency_300.bev_ap_07")),
        *(f"ours.failure_duration_q_{q}.bev_ap_07" for q in (1, 2, 3)),
        *(f"{subject}.complexity.{metric}" for subject in ("ours", "concat_capacity_matched") for metric in ("params_million", "flops_giga", "peak_gpu_memory_gb", "inference_latency_ms")),
    }
    missing_existing = sorted(required_existing - ids)
    diagnostic_ids = {
        f"ours.diagnostic.delay_{delay:03d}_{fault}_{scope}.bev_ap_07"
        for delay in (0, 100, 200, 300)
        for fault in ("l_fail", "c_fail")
        for scope in ("e_only", "r_only")
    }
    duration_modality_ids = {
        f"ours.failure_duration_q_{q}.{modality}.bev_ap_07"
        for q in (1, 2, 3)
        for modality in ("lidar", "camera")
    }
    missing_diagnostics = sorted(diagnostic_ids - ids)
    missing_duration_modalities = sorted(duration_modality_ids - ids)
    sensitivity_ids = {
        f"ours_p_{str(p).replace('.', '_')}.{suffix}"
        for p in (0.0, 0.1, 0.3, 0.5)
        for suffix in (
            "full.bev_ap_07",
            "l_fail.bev_ap_07",
            "full.latency_300.bev_ap_07",
        )
    }
    misbound_sensitivity = sorted(
        result_id
        for result_id in sensitivity_ids & ids
        if (
            PROTOCOL_ID
            not in str(records_by_id[result_id].get("fault_protocol") or "")
            or records_by_id[result_id].get("source")
            != "pending fixed-seed controlled measurement"
            or not str(records_by_id[result_id].get("model") or "").startswith(
                "Selected method"
            )
        )
    )
    return {
        "status": (
            "ready"
            if not missing_existing
            and not missing_diagnostics
            and not missing_duration_modalities
            and not misbound_sensitivity
            else "incomplete"
        ),
        "paper_root": str(paper_root.resolve()),
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": _sha256_file(registry_path),
        "submission_contract_path": str(contract_path.resolve()),
        "submission_contract_sha256": _sha256_file(contract_path),
        "missing_existing_table_ids": missing_existing,
        "missing_agent_scope_diagnostic_ids": missing_diagnostics,
        "missing_duration_modality_ids": missing_duration_modalities,
        "sensitivity_ids_requiring_controlled_rebinding": misbound_sensitivity,
        "table_v_reference_binding": (
            "replace the generic full-nonlinear reference with the selected "
            "winner's controlled result identity before component interpretation"
        ),
        "duration_table_blocker": (
            "the protocol says LiDAR and Camera are measured separately, but the "
            "current table has only one scalar ID per q; add modality-specific "
            "columns/IDs or seal an aggregation rule before importing AP"
        ),
    }


def _job(
    job_id: str,
    *,
    stage: str,
    kind: str,
    depends_on: Sequence[str],
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "stage": stage,
        "kind": kind,
        "depends_on": list(depends_on),
        "payload": dict(payload),
    }


def _assert_dag(jobs: Sequence[Mapping[str, object]]) -> None:
    ids = [job.get("job_id") for job in jobs]
    if any(type(item) is not str or not item for item in ids) or len(ids) != len(set(ids)):
        raise PostWinnerPlanError("post-winner job IDs are invalid or duplicated")
    seen: set[str] = set()
    for job in jobs:
        dependencies = job.get("depends_on")
        if not isinstance(dependencies, list) or any(item not in seen for item in dependencies):
            raise PostWinnerPlanError(
                f"job {job.get('job_id')} has a missing or forward dependency"
            )
        seen.add(str(job["job_id"]))


def build_plan(
    *,
    selected_identity: Mapping[str, object],
    checkpoint: Path,
    training_index_path: Path,
    evaluation_index_path: Path,
    temporal_manifest_path: Path,
    validation_cohort_path: Path,
    paper_root: Path,
) -> dict[str, object]:
    winner_binding = _validate_selected_identity(selected_identity, checkpoint=checkpoint)
    winner_config = _semantic_config(
        _load_config(ROOT / str(winner_binding["config_path"]))
    )
    training_index = _read_json(training_index_path, context="training overlay index")
    training_binding = _validate_training_index(training_index)
    evaluation_index = _read_json(evaluation_index_path, context="evaluation overlay index")
    evaluation_binding = _validate_evaluation_index(evaluation_index)
    if training_binding["temporal_manifest_sha256"] != evaluation_binding["temporal_manifest_sha256"]:
        raise PostWinnerPlanError("training and evaluation manifest identities differ")
    manifest = _read_json(temporal_manifest_path, context="temporal manifest")
    manifest_sha = _verify_content_document(manifest, context="temporal manifest")
    if manifest_sha != evaluation_binding["temporal_manifest_sha256"]:
        raise PostWinnerPlanError("temporal manifest differs from formal overlays")
    if manifest.get("split_sha256") != SPLIT_SHA256:
        raise PostWinnerPlanError("temporal manifest split SHA-256 drifted")
    cohort = _read_json(validation_cohort_path, context="validation cohort")
    cohort_sha = _verify_content_document(cohort, context="validation cohort")
    if (
        cohort_sha != evaluation_binding["cohort_content_sha256"]
        or cohort.get("sample_ids_sha256") != SAMPLE_IDS_SHA256
    ):
        raise PostWinnerPlanError("validation cohort differs from formal evaluation")

    jobs: list[dict[str, object]] = []
    jobs.append(
        _job(
            "winner.bind",
            stage="common",
            kind="local_identity_validation",
            depends_on=(),
            payload={"binding": winner_binding},
        )
    )
    ablations: list[dict[str, object]] = []
    for spec in _ablation_specs(winner_config):
        audit = _existing_ablation_audit(winner_config, spec)
        existing_subject = spec.get("existing_subject")
        archive = _checkpoint_archive(existing_subject) if type(existing_subject) is str else None
        record = {
            **dict(spec),
            "reuse_audit": audit,
            "local_checkpoint_archive": archive,
            "single_factor_contract": (
                "all merged training/evaluation config leaves outside "
                "expected_difference_paths must equal the selected winner"
            ),
        }
        ablations.append(record)
        resolve_id = f"table_v.{spec['id']}.resolve"
        jobs.append(
            _job(
                resolve_id,
                stage="table_v",
                kind="reuse_or_winner_derived_config",
                depends_on=("winner.bind",),
                payload=record,
            )
        )
        if audit["status"] == "not_applicable_winner_already_has_setting":
            continue
        execute_id = f"table_v.{spec['id']}.train_or_reuse"
        jobs.append(
            _job(
                execute_id,
                stage="table_v",
                kind="conditional_training",
                depends_on=(resolve_id,),
                payload={
                    "reuse_only_if": [
                        "local final checkpoint size/SHA verified",
                        "sealed run contract matches winner dataset/teacher/seed/FP32/global-batch-8/50epoch/val10",
                        "sealed source and exact single-factor config comparison pass",
                        "final checkpoint contract and teacher audit pass",
                    ],
                    "otherwise_train": True,
                    "training_contract": TRAINING_CONTRACT,
                },
            )
        )
        jobs.append(
            _job(
                f"table_v.{spec['id']}.evaluate",
                stage="table_v",
                kind="controlled_1337_evaluation",
                depends_on=(execute_id,),
                payload={
                    "protocol_id": PROTOCOL_ID,
                    "agent_scope": "E+R",
                    "requested_cells": [
                        "delay_000_full",
                        "delay_000_l_fail",
                        "delay_300_full",
                    ],
                    "required_metrics": [
                        "car_bev_ap_05",
                        "car_bev_ap_07",
                        "car_3d_ap_05",
                        "car_3d_ap_07",
                    ],
                    "sample_count": SAMPLE_COUNT,
                    "ground_truth_count": GROUND_TRUTH_COUNT,
                    "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
                    "sample_ids_sha256": SAMPLE_IDS_SHA256,
                },
            )
        )

    for probability in SENSITIVITY_PROBABILITIES:
        label = str(probability).replace(".", "_")
        overlay_id = f"table_vi.p_{label}.overlay"
        jobs.append(
            _job(
                overlay_id,
                stage="table_vi",
                kind="training_overlay_generation",
                depends_on=("winner.bind",),
                payload={
                    "p_lidar": probability,
                    "p_camera": probability,
                    "protocol_seed": TRAINING_SEED,
                    "epochs": list(range(50)),
                    "command": [
                        "python",
                        "tools/resilient_v2x/build_overlays.py",
                        "train",
                        str(temporal_manifest_path.resolve()),
                        "--expected-split-sha256",
                        SPLIT_SHA256,
                        "--protocol-seed",
                        str(TRAINING_SEED),
                        "--epochs",
                        *(str(epoch) for epoch in range(50)),
                        "--p-lidar",
                        str(probability),
                        "--p-camera",
                        str(probability),
                        "--out-dir",
                        f"artifacts/resilient_v2x/post-winner/sensitivity/p_{label}",
                    ],
                },
            )
        )
        train_id = f"table_vi.p_{label}.train"
        jobs.append(
            _job(
                train_id,
                stage="table_vi",
                kind="winner_architecture_training",
                depends_on=(overlay_id,),
                payload={"training_contract": TRAINING_CONTRACT, "probability": probability},
            )
        )
        jobs.append(
            _job(
                f"table_vi.p_{label}.evaluate",
                stage="table_vi",
                kind="controlled_1337_evaluation",
                depends_on=(train_id,),
                payload={
                    "requested_cells": [
                        "delay_000_full",
                        "delay_000_l_fail",
                        "delay_300_full",
                    ],
                    "sample_ids_sha256": SAMPLE_IDS_SHA256,
                },
            )
        )
    jobs.append(
        _job(
            "table_vi.p_0_2.reuse",
            stage="table_vi",
            kind="selected_winner_result_reuse",
            depends_on=("winner.bind",),
            payload={
                "p_lidar": MAIN_TRAINING_PROBABILITY,
                "p_camera": MAIN_TRAINING_PROBABILITY,
                "training_overlay_binding": training_binding,
                "source_cells": [
                    "controlled_1337.selected_method.delay_000_full.bev_ap_07",
                    "controlled_1337.selected_method.delay_000_l_fail.bev_ap_07",
                    "controlled_1337.selected_method.delay_300_full.bev_ap_07",
                ],
                "new_training_forbidden": True,
            },
        )
    )

    duration_prepare = "table_vii.duration.prepare_overlays"
    jobs.append(
        _job(
            duration_prepare,
            stage="table_vii",
            kind="duration_overlay_generation",
            depends_on=("winner.bind",),
            payload={
                "status": "required_current_formal_manifest_has_no_duration_2_3_overlays",
                "temporal_manifest": str(temporal_manifest_path.resolve()),
                "duration_cohort_contract": {
                    "max_delay_ms": 0,
                    "max_duration": 4,
                    "split": "val",
                    "must_bind_temporal_manifest_sha256": manifest_sha,
                },
                "durations_to_generate": [1, 2, 3, 4],
                "conditions": ["L-Fail", "C-Fail"],
                "agent_scopes": ["E+R", "E-only", "R-only"],
                "legacy_duration_overlays_reusable": False,
            },
        )
    )
    for duration in (1, 2, 3):
        for modality in ("lidar", "camera"):
            jobs.append(
                _job(
                    f"table_vii.duration.q{duration}.{modality}.evaluate",
                    stage="table_vii",
                    kind="winner_checkpoint_diagnostic_evaluation",
                    depends_on=(duration_prepare,),
                    payload={
                        "delay_ms": 0,
                        "duration_ticks": duration,
                        "modality": modality,
                        "condition": "L-Fail" if modality == "lidar" else "C-Fail",
                        "agent_scope": "E+R",
                        "checkpoint_sha256": winner_binding["checkpoint_sha256"],
                        "paper_result_id": f"ours.failure_duration_q_{duration}.{modality}.bev_ap_07",
                        "paper_binding_status": "missing_modality_specific_result_id",
                    },
                )
            )
    jobs.append(
        _job(
            "table_vii.duration.q4_boundary",
            stage="table_vii",
            kind="unsupported_no_extrapolation_trace",
            depends_on=(duration_prepare,),
            payload={
                "duration_ticks": 4,
                "status": "unsupported",
                "ap_result_id": None,
                "required_trace": [
                    "neutral feature",
                    "raw availability zero",
                    "propagated availability zero",
                    "reliability zero",
                ],
            },
        )
    )

    for scope in ("E-only", "R-only"):
        jobs.append(
            _job(
                f"diagnostic.{scope.lower().replace('-', '_')}.evaluate",
                stage="agent_scope_diagnostics",
                kind="winner_checkpoint_diagnostic_matrix",
                depends_on=("winner.bind",),
                payload={
                    "delays_ms": [0, 100, 200, 300],
                    "conditions": ["L-Fail", "C-Fail"],
                    "agent_scope": scope,
                    "duration_ticks": 1,
                    "run_count": 8,
                    "main_table_membership": False,
                    "required_distinct_result_ids": True,
                    "sample_count": SAMPLE_COUNT,
                    "ground_truth_count": GROUND_TRUTH_COUNT,
                    "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
                    "sample_ids_sha256": SAMPLE_IDS_SHA256,
                },
            )
        )

    concat_id = "table_vii.complexity.concat_config"
    jobs.append(
        _job(
            concat_id,
            stage="table_vii",
            kind="winner_derived_capacity_matched_concat",
            depends_on=("winner.bind",),
            payload={
                "overrides": {"model.routing_mode": "concat"},
                "must_retain_all_winner_parameters": True,
                "teacher_excluded": True,
                "reuse_existing_concat_only_if_full_semantic_match": True,
            },
        )
    )
    for subject, dependency in (("winner", "winner.bind"), ("concat", concat_id)):
        jobs.append(
            _job(
                f"table_vii.complexity.{subject}.profile",
                stage="table_vii",
                kind="deployment_profile",
                depends_on=(dependency,),
                payload={
                    "profiler": "tools/resilient_v2x/profile.py",
                    "warmup_iterations": 10,
                    "measured_iterations": 100,
                    "batch_size": 1,
                    "exclude_module_paths": ["teacher"],
                    "teacher_must_be_detached_before_cuda": True,
                    "flops_must_be_measured": True,
                    "peak_memory_must_be_measured": True,
                    "latency_boundary": (
                        "model.test_step from collated sensing tensors through decoded "
                        "predictions and post-processing; dataloader disk I/O excluded"
                    ),
                },
            )
        )
    jobs.append(
        _job(
            "table_vii.complexity.validate_pair",
            stage="table_vii",
            kind="complexity_pair_validation",
            depends_on=(
                "table_vii.complexity.winner.profile",
                "table_vii.complexity.concat.profile",
            ),
            payload={
                "required_equal_fields": [
                    "device",
                    "batch_size",
                    "warmup_iterations",
                    "measured_iterations",
                    "latency_boundary",
                    "input artifact hashes",
                ],
                "parameter_count_must_match_exactly": True,
                "teacher_cost_must_be_zero_for_both": True,
                "required_metrics": [
                    "parameters",
                    "FLOPs",
                    "peak allocated GPU memory",
                    "median synchronized end-to-end latency",
                ],
            },
        )
    )

    _assert_dag(jobs)
    paper = _paper_interface_audit(paper_root)
    winner_specific_actions: list[dict[str, object]] = []
    for row in ablations:
        audit = row["reuse_audit"]
        if not isinstance(audit, Mapping):
            raise PostWinnerPlanError("ablation reuse audit is invalid")
        if audit.get("status") == "not_applicable_winner_already_has_setting":
            winner_specific_actions.append(
                {
                    "ablation_id": row["id"],
                    "action": "remove_or_replace_noop_row",
                    "reason": audit["reason"],
                }
            )
        if row["id"] == "winner_residual_module_removal":
            winner_specific_actions.append(
                {
                    "ablation_id": row["id"],
                    "action": "add_winner_module_removal_row_and_three_result_ids",
                    "suggested_result_ids": [
                        "ours_winner_residual_module_removal.full.bev_ap_07",
                        "ours_winner_residual_module_removal.l_fail.bev_ap_07",
                        (
                            "ours_winner_residual_module_removal."
                            "full.latency_300.bev_ap_07"
                        ),
                    ],
                }
            )
    if _model_value(winner_config, "ptf_mode", "nonlinear") != "nonlinear":
        winner_specific_actions.append(
            {
                "ablation_id": "alternative_ptf",
                "action": "relabel_reference_and_alternative_ptf_rows",
                "reason": (
                    "the selected winner is not nonlinear PTF, so the current "
                    "Full nonlinear/Linear PTF row semantics are reversed"
                ),
            }
        )
    if isinstance(paper, dict):
        paper["winner_specific_table_v_actions"] = winner_specific_actions
        if winner_specific_actions:
            paper["status"] = "incomplete"
    plan = _seal(
        {
            "schema_version": 1,
            "document_type": DOCUMENT_TYPE,
            "mode": "local_plan_only_no_remote_writes",
            "protocol_id": PROTOCOL_ID,
            "training_seed": TRAINING_SEED,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "training_contract": TRAINING_CONTRACT,
            "winner_binding": winner_binding,
            "protocol_bindings": {
                "training_overlay_index_path": str(training_index_path.resolve()),
                "training_overlay": training_binding,
                "evaluation_overlay_index_path": str(evaluation_index_path.resolve()),
                "evaluation_overlay": evaluation_binding,
                "temporal_manifest_path": str(temporal_manifest_path.resolve()),
                "temporal_manifest_content_sha256": manifest_sha,
                "validation_cohort_path": str(validation_cohort_path.resolve()),
                "validation_cohort_content_sha256": cohort_sha,
            },
            "table_v_ablation_resolution": ablations,
            "table_vi_probability_contract": {
                "train": list(SENSITIVITY_PROBABILITIES),
                "reuse": [MAIN_TRAINING_PROBABILITY],
                "p_lidar_equals_p_camera": True,
            },
            "table_vii_duration_contract": {
                "scored_durations": [1, 2, 3],
                "modalities_measured_separately": ["lidar", "camera"],
                "agent_scope": "E+R",
                "unsupported_from_duration": 4,
                "no_ap_for_unsupported": True,
            },
            "complexity_contract": {
                "subjects": ["winner", "winner-derived capacity-matched concat"],
                "teacher_excluded": True,
                "same_hardware_and_timing_protocol": True,
                "parameter_count_exact_match_required": True,
            },
            "paper_interface_audit": paper,
            "job_count": len(jobs),
            "jobs": jobs,
        }
    )
    validate_plan(plan)
    return plan


def validate_plan(value: Mapping[str, object]) -> None:
    _require_seal(value, context="post-winner plan")
    if (
        value.get("schema_version") != 1
        or value.get("document_type") != DOCUMENT_TYPE
        or value.get("mode") != "local_plan_only_no_remote_writes"
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("training_seed") != TRAINING_SEED
        or value.get("checkpoint_policy") != CHECKPOINT_POLICY
        or value.get("training_contract") != TRAINING_CONTRACT
    ):
        raise PostWinnerPlanError("post-winner plan contract drifted")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or value.get("job_count") != len(jobs):
        raise PostWinnerPlanError("post-winner job inventory is invalid")
    _assert_dag(jobs)
    ids = {str(job["job_id"]) for job in jobs}
    mandatory = {
        "winner.bind",
        "table_vi.p_0_2.reuse",
        "table_vii.duration.q4_boundary",
        "diagnostic.e_only.evaluate",
        "diagnostic.r_only.evaluate",
        "table_vii.complexity.winner.profile",
        "table_vii.complexity.concat.profile",
        "table_vii.complexity.validate_pair",
    }
    if not mandatory <= ids:
        raise PostWinnerPlanError("post-winner plan is missing mandatory jobs")


def _atomic_write_new(path: Path, value: Mapping[str, object]) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise PostWinnerPlanError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-method-identity", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--training-index",
        type=Path,
        default=ROOT / "artifacts/resilient_v2x/dair_v2/training_overlays.json",
    )
    parser.add_argument(
        "--evaluation-index",
        type=Path,
        default=ROOT / "artifacts/resilient_v2x/dair_v2/evaluation_overlays.json",
    )
    parser.add_argument(
        "--temporal-manifest",
        type=Path,
        default=ROOT / "artifacts/resilient_v2x/dair/temporal_manifest_v2.json",
    )
    parser.add_argument(
        "--validation-cohort",
        type=Path,
        default=ROOT / "artifacts/resilient_v2x/dair_v2/validation_cohort.json",
    )
    parser.add_argument(
        "--paper-root", type=Path, default=Path("/home/lbin/Desktop/ResilientV2X")
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    identity = _read_json(
        args.selected_method_identity, context="selected-method identity"
    )
    plan = build_plan(
        selected_identity=identity,
        checkpoint=args.checkpoint,
        training_index_path=args.training_index,
        evaluation_index_path=args.evaluation_index,
        temporal_manifest_path=args.temporal_manifest,
        validation_cohort_path=args.validation_cohort,
        paper_root=args.paper_root,
    )
    output = _atomic_write_new(args.out, plan)
    print(
        json.dumps(
            {
                "output": str(output),
                "seal_sha256": plan["seal_sha256"],
                "job_count": plan["job_count"],
                "paper_interface_status": plan["paper_interface_audit"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
