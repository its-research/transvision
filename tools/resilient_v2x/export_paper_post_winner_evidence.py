#!/usr/bin/env python3
"""Export sealed Table V--VII evidence from completed ClearML tasks.

The command is read-only unless ``--write`` and the exact write token are
provided.  It validates the fixed single-seed identities, evaluation counts,
prediction archives, and deployment-only profile pair before publishing one
local JSON document for the paper repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from clearml import Task

try:
    from tools.resilient_v2x import collect_clearml_candidate_models as candidates
    from tools.resilient_v2x import export_paper_controlled_1337_evidence as controlled
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.resilient_v2x import collect_clearml_candidate_models as candidates
    from tools.resilient_v2x import export_paper_controlled_1337_evidence as controlled


DOCUMENT_TYPE = "resilient_v2x_paper_post_winner_evidence"
WRITE_TOKEN = "WRITE_VERIFIED_PAPER_POST_WINNER_EVIDENCE"
DEFAULT_OUTPUT = Path(
    "artifacts/resilient_v2x/paper-post-winner/formal-evidence.json"
)
DEFAULT_PLAN = Path(
    "artifacts/resilient_v2x/final-single-seed-selection/"
    "post-winner-experiment-plan.json"
)
DEFAULT_IDENTITY = Path(
    "artifacts/resilient_v2x/final-single-seed-selection/"
    "selected-method-identity.json"
)
DEFAULT_MODEL_ROOT = Path("artifacts/trained_models/completed-live")

WATCHER_TASK_ID = "74bf35decc2340b496167c11ec50f54b"
LEADERBOARD_TASK_ID = "cc7b54e4dcca4b92a4247f92987add60"
AUDIT_TASK_ID = "dd287fe58e264bcca48fe222b6981048"
SELECTOR_TASK_ID = "d7ea54ce540d4b0486904c5910100885"
SELECTED_IDENTITY_SEAL = (
    "7e24831bc1420b202abedf9c748423223875687c8d19fc2eeadc4ad14f944aad"
)
WINNER_TRAINING_TASK_ID = "54d28bc513794051810fd383140ae96e"
WINNER_CHECKPOINT_SHA256 = (
    "f6a5683a1df7126a3069edce577177eefb80208d9548ba3b61a8907bbf148a67"
)

TABLE_V = (
    ("full", "resilient_v2x"),
    ("no_ptf", "ptf_none"),
    ("linear_ptf", "ptf_linear"),
    ("static_three_expert", "router_static"),
    ("uniform_gate", "router_uniform"),
    ("no_reliability", "no_reliability"),
    ("no_delay_metadata", "no_delay_metadata"),
    ("no_distillation", "no_distillation"),
)
TABLE_VI = (
    (
        "0.0",
        "4c4ec65836ad463da0041a1caf732aba",
        "63ab06ebdd1649f7ba8b77262a211cc5",
        "b4be197c1c24492b9d5e0dc263d4993f",
    ),
    (
        "0.1",
        "bdeee07140b245409e89db2e935dd02f",
        "087bc9f0f679425da29c2329aa239197",
        "a9f5abbda6c7418eabd3bcd8b4672d24",
    ),
    (
        "0.3",
        "e431d93a909f4888a0d79561b5ca749c",
        "a698dc196a144104996b13528ddbc95e",
        "fc2f995b69b34a759e9aa7287a1258f9",
    ),
    (
        "0.5",
        "59d1c2bbb91741e98f46564accc58477",
        "f0b0faeb2c2a46eab8038ce562453ae6",
        "f30b2ec6fffd413c9a602c6cd13e6097",
    ),
)
DURATION_TASKS = (
    (1, "lidar", "cb5eb3f9cb624018b504049bf9492d30"),
    (1, "camera", "d9d1706284614fef8775a80d17c0cf91"),
    (2, "lidar", "4417e476be21492d8e3dc817b3f1f932"),
    (2, "camera", "fa3423b90d5944db99552a2cbf9f66a6"),
    (3, "lidar", "66121b5326e2460880f0f91cb844197b"),
    (3, "camera", "759463bca4424072ba3d5e9679f94fe3"),
)
WINNER_PROFILE_TASK_ID = "f34913cdea634939ac96c787f603567f"
CONCAT_PROFILE_TASK_ID = "aa91489fb79d42758ef5083afbccb7ea"
PAIR_VALIDATOR_TASK_ID = "8a8ded5f23f74920be24ac8fd12b3fbf"
PROFILE_SUBJECTS = {
    "table_vii.complexity.winner.profile": {
        "subject": "resilient_v2x",
        "training_task_id": WINNER_TRAINING_TASK_ID,
        "model_id": "ea23e399bc554b07bc2ce07e6b68700e",
        "checkpoint_sha256": WINNER_CHECKPOINT_SHA256,
    },
    "table_vii.complexity.concat.profile": {
        "subject": "concat_capacity_matched",
        "training_task_id": "175e6087f66646e282a93439fecf292a",
        "model_id": "2a3378058b9a453fa47e15a98aa4ca15",
        "checkpoint_sha256": (
            "f47d962959e9f990b0af1020e26b232e7ed1d72b69365ca19c68358316adeb56"
        ),
    },
}
BEV_AP_07 = "resilient_v2x/car_bev_ap_r40_0.70"


class PostWinnerEvidenceError(RuntimeError):
    pass


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PostWinnerEvidenceError(f"{context} is not an object")
    return dict(value)


def _read_sealed(path: Path, context: str) -> dict[str, object]:
    value = controlled._strict_json(path.resolve(strict=True).read_bytes(), context)
    controlled._require_seal(value, context)
    return value


def _payload(task: object, name: str, context: str) -> tuple[dict[str, object], dict[str, object]]:
    return controlled._artifact_payload(task, name, context=context)


def _content_seal(value: Mapping[str, object], context: str) -> str:
    observed = controlled._sha256(value.get("content_sha256"), f"{context} content")
    detached = dict(value)
    detached.pop("content_sha256", None)
    if controlled._content_sha256(detached) != observed:
        raise PostWinnerEvidenceError(f"{context} content hash drifted")
    return observed


def _three_values(runs: Sequence[Mapping[str, object]]) -> dict[str, float]:
    by_id = {str(run["condition_id"]): run for run in runs}
    required = {
        "full": "delay_000_full",
        "l_fail": "delay_000_l_fail",
        "latency_300": "delay_300_full",
    }
    return {
        key: controlled._finite_ap(
            _mapping(by_id[condition]["metrics"], condition).get(BEV_AP_07),
            f"{condition} BEV AP@0.7",
        )
        for key, condition in required.items()
    }


def _load_chain() -> tuple[
    controlled.ChainEvidence,
    str,
    dict[str, Mapping[str, object]],
    dict[str, Mapping[str, object]],
]:
    watcher = controlled._resolve_task(Task, WATCHER_TASK_ID, "watcher task")
    leaderboard_task = controlled._resolve_task(
        Task, LEADERBOARD_TASK_ID, "leaderboard task"
    )
    audit_task = controlled._resolve_task(Task, AUDIT_TASK_ID, "audit task")
    selector_task = controlled._resolve_task(Task, SELECTOR_TASK_ID, "selector task")
    if (
        controlled._parent(leaderboard_task) != WATCHER_TASK_ID
        or controlled._parent(audit_task) != LEADERBOARD_TASK_ID
        or controlled._parent(selector_task) != AUDIT_TASK_ID
    ):
        raise PostWinnerEvidenceError("formal W/L/A/S parent chain drifted")
    plan, _ = _payload(watcher, "formal_1337_evaluation_plan", "watcher task")
    leaderboard, _ = _payload(
        leaderboard_task, "formal_1337_leaderboard", "leaderboard task"
    )
    audit, _ = _payload(
        audit_task, "formal_1337_comparability_audit", "audit task"
    )
    selector, _ = _payload(
        selector_task, "formal_candidate_selection", "selector task"
    )
    controller_id = controlled._task_id(
        plan.get("training_controller_task_id"), "training controller"
    )
    controller = controlled._resolve_task(Task, controller_id, "training controller")
    training_manifest, _ = _payload(
        controller, "formal_1337_training_manifest", "training controller"
    )
    chain = controlled.ChainEvidence(
        WATCHER_TASK_ID,
        LEADERBOARD_TASK_ID,
        AUDIT_TASK_ID,
        SELECTOR_TASK_ID,
        plan,
        leaderboard,
        audit,
        selector,
        training_manifest,
    )
    validated = controlled._validate_chain(chain)
    return chain, controller_id, validated[4], validated[5]


def _collect_table_v(
    *,
    experiment_plan: Mapping[str, object],
    model_root: Path,
) -> list[dict[str, object]]:
    chain, controller_id, training_entries, leaderboard_results = _load_chain()
    raw_reuse = experiment_plan.get("table_v_ablation_resolution")
    if not isinstance(raw_reuse, list) or len(raw_reuse) != 7:
        raise PostWinnerEvidenceError("Table V reuse audit inventory drifted")
    reuse_by_subject = {
        str(_mapping(row, "Table V reuse row").get("existing_subject")): _mapping(
            row, "Table V reuse row"
        )
        for row in raw_reuse
    }
    result: list[dict[str, object]] = []
    for variant, subject in TABLE_V:
        training = training_entries.get(subject)
        leaderboard_row = leaderboard_results.get(subject)
        if training is None or leaderboard_row is None:
            raise PostWinnerEvidenceError(f"Table V subject {subject} is unavailable")
        training_task = controlled._resolve_task(
            Task, str(training["training_task_id"]), f"{subject} training task"
        )
        training_identity = controlled._training_identity(
            training_task,
            subject=subject,
            training_entry=training,
            audit=chain.audit,
            model_root=model_root,
        )
        evaluation_task = controlled._resolve_task(
            Task,
            str(leaderboard_row["evaluation_task_id"]),
            f"{subject} evaluation task",
        )
        identity, runs = controlled._collect_evaluation(
            evaluation_task,
            subject=subject,
            training_identity=training_identity,
            expected_parent=controller_id,
        )
        if subject == "resilient_v2x":
            audit = {
                "status": "selected_winner_reference",
                "observed_difference_paths": [],
            }
        else:
            reuse = reuse_by_subject.get(subject)
            if reuse is None:
                raise PostWinnerEvidenceError(f"Table V reuse row {subject} is absent")
            reuse_audit = _mapping(reuse.get("reuse_audit"), f"{subject} reuse audit")
            if (
                reuse_audit.get("reusable") is not True
                or reuse_audit.get("observed_difference_paths")
                != reuse.get("expected_difference_paths")
            ):
                raise PostWinnerEvidenceError(f"{subject} is not a single-factor reuse")
            local = _mapping(
                reuse.get("local_checkpoint_archive"), f"{subject} local checkpoint"
            )
            if (
                local.get("checkpoint_sha256") != identity["checkpoint_sha256"]
                or local.get("checkpoint_size_bytes")
                != identity["checkpoint_size_bytes"]
                or local.get("training_task_id") != identity["training_task_id"]
            ):
                raise PostWinnerEvidenceError(f"{subject} reuse checkpoint drifted")
            audit = {
                "status": "verified_single_factor_reuse",
                "concept": reuse.get("concept"),
                "observed_difference_paths": reuse_audit.get(
                    "observed_difference_paths"
                ),
                "config_path": reuse_audit.get("existing_config_path"),
                "config_sha256": reuse_audit.get("existing_config_sha256"),
            }
        result.append(
            {
                "variant": variant,
                "subject": subject,
                "identity": identity,
                "reuse_audit": audit,
                "values": _three_values(runs),
            }
        )
    return result


def _validate_training_probability(
    *, p: str, task_id: str, dataset_id: str
) -> dict[str, object]:
    task = controlled._resolve_task(Task, task_id, f"p={p} training task")
    launch, launch_record = _payload(
        task, "post_winner_formal_v2_launch_contract", f"p={p} training task"
    )
    controlled._require_seal(launch, f"p={p} launch contract")
    probability = float(p)
    expected_launch = {
        "artifact_type": "resilient_v2x_table_vi_formal_v2_launch_contract",
        "checkpoint_policy": "epoch_50_final_only",
        "global_batch": 8,
        "gpus": 4,
        "max_epochs": 50,
        "p_camera": probability,
        "p_lidar": probability,
        "precision": "FP32",
        "selected_method_identity_seal": SELECTED_IDENTITY_SEAL,
        "table": "VI",
        "task_id": task_id,
        "training_dataset_id": dataset_id,
        "training_seed": controlled.TRAINING_SEED,
        "validation_interval_epochs": 10,
    }
    for key, value in expected_launch.items():
        if launch.get(key) != value:
            raise PostWinnerEvidenceError(f"p={p} launch {key} drifted")
    run, run_record = _payload(task, "run_contract", f"p={p} training task")
    for key, value in {
        "experiment": "resilient_v2x",
        "training_dataset_id": dataset_id,
        "gpus": 4,
        "global_batch_size": 8,
        "max_epochs": 50,
        "val_interval": 10,
        "precision": "FP32",
        "amp": False,
        "seed": controlled.TRAINING_SEED,
    }.items():
        if run.get(key) != value:
            raise PostWinnerEvidenceError(f"p={p} training {key} drifted")
    teacher = _mapping(run.get("teacher"), f"p={p} teacher")
    if {
        "task_id": teacher.get("task_id"),
        "model_id": teacher.get("model_id"),
        "sha256": teacher.get("sha256"),
    } != {
        "task_id": controlled.TEACHER_TASK_ID,
        "model_id": controlled.TEACHER_MODEL_ID,
        "sha256": controlled.TEACHER_CHECKPOINT_SHA256,
    }:
        raise PostWinnerEvidenceError(f"p={p} teacher drifted")
    teacher_audit, teacher_record = _payload(
        task, "common_teacher_initialization_audit", f"p={p} training task"
    )
    candidates._validate_teacher_audit(teacher_audit, label=f"p={p}")
    final, final_record = _payload(
        task, "final_checkpoint_contract", f"p={p} training task"
    )
    if final.get("filename") != "epoch_50.pth":
        raise PostWinnerEvidenceError(f"p={p} final checkpoint is not epoch 50")
    checkpoint = {
        "model_id": controlled._task_id(final.get("model_id"), f"p={p} model"),
        "sha256": controlled._sha256(final.get("sha256"), f"p={p} checkpoint"),
        "size_bytes": controlled._positive_int(
            final.get("size_bytes"), f"p={p} checkpoint size"
        ),
        "url": str(final.get("url") or ""),
    }
    return {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "checkpoint": checkpoint,
        "artifact_sha256": {
            "launch": launch_record["hash"],
            "run_contract": run_record["hash"],
            "teacher_audit": teacher_record["hash"],
            "final_checkpoint": final_record["hash"],
        },
    }


def _collect_probability_evaluation(
    *, p: str, task_id: str, training: Mapping[str, object]
) -> dict[str, object]:
    task = controlled._resolve_task(Task, task_id, f"p={p} evaluation task")
    run_contract, run_record = _payload(task, "run_contract", f"p={p} evaluation")
    checkpoint = _mapping(run_contract.get("checkpoint"), f"p={p} eval checkpoint")
    trained = _mapping(training.get("checkpoint"), f"p={p} training checkpoint")
    if (
        run_contract.get("task_id") != task_id
        or run_contract.get("baseline_task_id") != training["task_id"]
        or run_contract.get("training_dataset_id") != controlled.TRAINING_DATASET_ID
        or checkpoint.get("task_id") != training["task_id"]
        or checkpoint.get("model_id") != trained["model_id"]
        or checkpoint.get("sha256") != trained["sha256"]
        or checkpoint.get("size_bytes") != trained["size_bytes"]
    ):
        raise PostWinnerEvidenceError(f"p={p} evaluation binding drifted")
    if p == "0.5":
        binding, _ = _payload(
            task, "post_winner_auto_validation_binding", f"p={p} evaluation"
        )
        if (
            binding.get("job_id") != "table_vi.p_0_5.evaluate"
            or binding.get("training_task_id") != training["task_id"]
            or binding.get("selected_identity_seal") != SELECTED_IDENTITY_SEAL
            or binding.get("model_id") != trained["model_id"]
            or binding.get("checkpoint_sha256") != trained["sha256"]
        ):
            raise PostWinnerEvidenceError("p=0.5 auto-validation binding drifted")
    elif controlled._parent(task) != training["task_id"]:
        raise PostWinnerEvidenceError(f"p={p} evaluation parent drifted")
    plan, plan_record = _payload(task, "evaluation_plan", f"p={p} evaluation")
    metrics, metrics_record = _payload(
        task, "controlled_baseline_metrics", f"p={p} evaluation"
    )
    runs = controlled._normalize_runs(
        metrics, subject="resilient_v2x", checkpoint_sha256=str(trained["sha256"])
    )
    evidence_path, evidence_record, temporary = controlled._artifact_file(
        task, "controlled_baseline_evidence", context=f"p={p} evaluation"
    )
    try:
        archive_sha = controlled._validate_prediction_archive(
            evidence_path,
            subject="resilient_v2x",
            checkpoint_sha256=str(trained["sha256"]),
            plan=plan,
            metrics=metrics,
            runs=runs,
        )
    finally:
        if temporary:
            evidence_path.unlink(missing_ok=True)
    if archive_sha != evidence_record["hash"]:
        raise PostWinnerEvidenceError(f"p={p} prediction archive drifted")
    return {
        "task_id": task_id,
        "values": _three_values(runs),
        "artifact_sha256": {
            "run_contract": run_record["hash"],
            "evaluation_plan": plan_record["hash"],
            "metrics": metrics_record["hash"],
            "prediction_evidence": evidence_record["hash"],
        },
    }


def _collect_table_vi() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for p, training_id, evaluation_id, dataset_id in TABLE_VI:
        training = _validate_training_probability(
            p=p, task_id=training_id, dataset_id=dataset_id
        )
        evaluation = _collect_probability_evaluation(
            p=p, task_id=evaluation_id, training=training
        )
        rows.append({"p": p, "training": training, "evaluation": evaluation})
    return rows


def _validate_duration_archive(
    *,
    task: object,
    task_id: str,
    q: int,
    modality: str,
    plan: Mapping[str, object],
    metrics: Mapping[str, object],
    run: Mapping[str, object],
) -> str:
    path, record, temporary = controlled._artifact_file(
        task, "post_winner_evidence", context=f"q={q} {modality}"
    )
    condition_id = "delay_000_l_fail" if modality == "lidar" else "delay_000_c_fail"
    required = {
        "evaluation_plan.json",
        "metrics.json",
        f"{condition_id}/resolved_config.py",
        f"{condition_id}/predictions.json",
        f"{condition_id}/checkpoint.sha256",
    }
    try:
        import zipfile

        with zipfile.ZipFile(path, "r", allowZip64=False) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or not required.issubset(names):
                raise PostWinnerEvidenceError(f"q={q} {modality} ZIP inventory drifted")
            extras = set(names) - required
            if any(
                not controlled._is_allowed_prediction_archive_extra(name)
                for name in extras
            ):
                raise PostWinnerEvidenceError(f"q={q} {modality} ZIP has unknown files")
            archived_plan = controlled._strict_json(
                archive.read("evaluation_plan.json"), f"q={q} {modality} plan"
            )
            archived_metrics = controlled._strict_json(
                archive.read("metrics.json"), f"q={q} {modality} metrics"
            )
            if not controlled._numbers_equal(archived_plan, plan) or not controlled._numbers_equal(
                archived_metrics, metrics
            ):
                raise PostWinnerEvidenceError(f"q={q} {modality} ZIP documents drifted")
            checkpoint_raw = archive.read(f"{condition_id}/checkpoint.sha256")
            if checkpoint_raw != f"{WINNER_CHECKPOINT_SHA256}\n".encode():
                raise PostWinnerEvidenceError(f"q={q} {modality} checkpoint drifted")
            controlled._validate_prediction_document(
                archive.read(f"{condition_id}/predictions.json"),
                expected_sha256=str(run["prediction_sha256"]),
                expected_content_sha256=str(run["prediction_content_sha256"]),
                context=f"q={q} {modality} predictions",
            )
        return str(record["hash"])
    finally:
        if temporary:
            path.unlink(missing_ok=True)


def _collect_duration() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for q, modality, task_id in DURATION_TASKS:
        task = controlled._resolve_task(Task, task_id, f"q={q} {modality} task")
        contract, contract_record = _payload(
            task, "post_winner_run_contract", f"q={q} {modality}"
        )
        condition = "L-Fail" if modality == "lidar" else "C-Fail"
        if {
            "task_id": contract.get("task_id"),
            "job_id": contract.get("job_id"),
            "duration_ticks": contract.get("duration_ticks"),
            "conditions": contract.get("conditions"),
            "delays_ms": contract.get("delays_ms"),
            "agent_scope": contract.get("agent_scope"),
            "selected_identity_seal": contract.get("selected_identity_seal"),
            "subject_task_id": contract.get("subject_task_id"),
            "training_dataset_id": contract.get("training_dataset_id"),
            "sample_count": contract.get("sample_count"),
            "ground_truth_count": contract.get("ground_truth_count"),
        } != {
            "task_id": task_id,
            "job_id": f"table_vii.duration.q{q}.{modality}.evaluate",
            "duration_ticks": q,
            "conditions": [condition],
            "delays_ms": [0],
            "agent_scope": controlled.AGENT_SCOPE,
            "selected_identity_seal": SELECTED_IDENTITY_SEAL,
            "subject_task_id": WINNER_TRAINING_TASK_ID,
            "training_dataset_id": controlled.TRAINING_DATASET_ID,
            "sample_count": controlled.SAMPLE_COUNT,
            "ground_truth_count": controlled.GROUND_TRUTH_COUNT,
        }:
            raise PostWinnerEvidenceError(f"q={q} {modality} contract drifted")
        checkpoint = _mapping(contract.get("checkpoint"), "duration checkpoint")
        if checkpoint.get("sha256") != WINNER_CHECKPOINT_SHA256:
            raise PostWinnerEvidenceError(f"q={q} {modality} checkpoint drifted")
        plan, plan_record = _payload(
            task, "post_winner_evaluation_plan", f"q={q} {modality}"
        )
        metrics, metrics_record = _payload(
            task, "post_winner_metrics", f"q={q} {modality}"
        )
        runs = metrics.get("runs")
        if (
            metrics.get("complete") is not True
            or metrics.get("planned_run_count") != 1
            or not isinstance(runs, list)
            or len(runs) != 1
        ):
            raise PostWinnerEvidenceError(f"q={q} {modality} metrics are incomplete")
        run = _mapping(runs[0], f"q={q} {modality} run")
        if (
            run.get("condition") != condition
            or run.get("delay_ms") != 0
            or run.get("sample_count") != controlled.SAMPLE_COUNT
            or run.get("ground_truth_count") != controlled.GROUND_TRUTH_COUNT
            or run.get("unsupported_sample_count") != 0
            or run.get("sample_ids_sha256") != controlled.SAMPLE_IDS_SHA256
        ):
            raise PostWinnerEvidenceError(f"q={q} {modality} run counts drifted")
        raw_metrics = _mapping(run.get("metrics"), f"q={q} {modality} metrics")
        for key, count in controlled.COUNT_METRICS.items():
            observed = raw_metrics.get(key)
            if type(observed) not in {int, float} or int(observed) != count:
                raise PostWinnerEvidenceError(f"q={q} {modality} {key} drifted")
        archive_sha = _validate_duration_archive(
            task=task,
            task_id=task_id,
            q=q,
            modality=modality,
            plan=plan,
            metrics=metrics,
            run=run,
        )
        result.append(
            {
                "q": q,
                "modality": modality,
                "task_id": task_id,
                "condition": condition,
                "bev_ap_07": controlled._finite_ap(
                    raw_metrics.get(BEV_AP_07), f"q={q} {modality} BEV AP@0.7"
                ),
                "artifact_sha256": {
                    "run_contract": contract_record["hash"],
                    "evaluation_plan": plan_record["hash"],
                    "metrics": metrics_record["hash"],
                    "prediction_evidence": archive_sha,
                },
            }
        )
    return result


def _profile(task_id: str, job_id: str) -> tuple[dict[str, object], dict[str, object]]:
    task = controlled._resolve_task(Task, task_id, f"{job_id} task")
    contract, contract_record = _payload(
        task, "post_winner_profile_run_contract", f"{job_id} task"
    )
    expected_subject = PROFILE_SUBJECTS[job_id]
    checkpoint = _mapping(contract.get("checkpoint"), f"{job_id} checkpoint")
    if (
        contract.get("task_id") != task_id
        or contract.get("job_id") != job_id
        or contract.get("selected_identity_seal") != SELECTED_IDENTITY_SEAL
        or contract.get("subject") != expected_subject["subject"]
        or contract.get("training_dataset_id") != controlled.TRAINING_DATASET_ID
        or contract.get("teacher_excluded") is not True
        or contract.get("warmup_iterations") != 10
        or contract.get("measured_iterations") != 100
        or checkpoint.get("task_id") != expected_subject["training_task_id"]
        or checkpoint.get("model_id") != expected_subject["model_id"]
        or checkpoint.get("sha256") != expected_subject["checkpoint_sha256"]
        or type(checkpoint.get("size_bytes")) is not int
        or int(checkpoint["size_bytes"]) <= 0
    ):
        raise PostWinnerEvidenceError(f"{job_id} profile contract drifted")
    profile, profile_record = _payload(task, "deployment_profile", f"{job_id} task")
    _content_seal(profile, f"{job_id} profile")
    if (
        profile.get("document_type") != "resilient_v2x_complexity_profile"
        or profile.get("batch_size") != 1
    ):
        raise PostWinnerEvidenceError(f"{job_id} profile identity drifted")
    parameters = _mapping(profile.get("parameters"), f"{job_id} parameters")
    flops = _mapping(profile.get("flops"), f"{job_id} FLOPs")
    runtime = _mapping(profile.get("runtime"), f"{job_id} runtime")
    memory = _mapping(runtime.get("gpu_memory"), f"{job_id} memory")
    latency = _mapping(runtime.get("latency"), f"{job_id} latency")
    device = _mapping(profile.get("device"), f"{job_id} device")
    if (
        flops.get("status") != "measured"
        or memory.get("status") != "measured"
        or latency.get("warmup_iterations") != 10
        or latency.get("measured_iterations") != 100
        or latency.get("synchronized") is not True
        or device.get("detached_training_only_modules") != ["teacher"]
    ):
        raise PostWinnerEvidenceError(f"{job_id} profile is incomplete")
    values = {
        "params_million": int(parameters["parameter_count"]) / 1_000_000,
        "flops_giga": int(flops["flop_count"]) / 1_000_000_000,
        "peak_gpu_memory_gb": int(memory["peak_allocated_bytes"]) / 1_000_000_000,
        "inference_latency_ms": float(latency["median_ms"]),
    }
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise PostWinnerEvidenceError(f"{job_id} profile values are invalid")
    return (
        {
            "task_id": task_id,
            "job_id": job_id,
            "device": device.get("name"),
            "checkpoint": checkpoint,
            "values": values,
            "artifact_sha256": {
                "run_contract": contract_record["hash"],
                "profile": profile_record["hash"],
            },
        },
        profile,
    )


def _collect_complexity() -> dict[str, object]:
    winner, winner_raw = _profile(
        WINNER_PROFILE_TASK_ID, "table_vii.complexity.winner.profile"
    )
    concat, concat_raw = _profile(
        CONCAT_PROFILE_TASK_ID, "table_vii.complexity.concat.profile"
    )
    validator = controlled._resolve_task(
        Task, PAIR_VALIDATOR_TASK_ID, "complexity pair validator"
    )
    pair, pair_record = _payload(
        validator, "complexity_pair_validation", "complexity pair validator"
    )
    _content_seal(pair, "complexity pair validation")
    winner_params = int(_mapping(winner_raw["parameters"], "winner parameters")["parameter_count"])
    concat_params = int(_mapping(concat_raw["parameters"], "concat parameters")["parameter_count"])
    winner_artifacts = winner_raw.get("artifacts")
    concat_artifacts = concat_raw.get("artifacts")
    if not isinstance(winner_artifacts, list) or not isinstance(concat_artifacts, list):
        raise PostWinnerEvidenceError("complexity profile artifacts are unavailable")
    def overlay_sha(rows: Sequence[object]) -> str:
        matches = [
            _mapping(row, "profile artifact")
            for row in rows
            if isinstance(row, Mapping) and row.get("role") == "evaluation_overlay_index"
        ]
        if len(matches) != 1:
            raise PostWinnerEvidenceError("profile overlay identity drifted")
        return str(matches[0]["sha256"])
    expected = {
        "status": "pass",
        "winner_task_id": WINNER_PROFILE_TASK_ID,
        "concat_task_id": CONCAT_PROFILE_TASK_ID,
        "device": winner["device"],
        "parameter_count": winner_params,
        "overlay_sha256": overlay_sha(winner_artifacts),
    }
    if any(pair.get(key) != value for key, value in expected.items()):
        raise PostWinnerEvidenceError("complexity pair validation drifted")
    if (
        winner_params != concat_params
        or winner["device"] != concat["device"]
        or overlay_sha(winner_artifacts) != overlay_sha(concat_artifacts)
    ):
        raise PostWinnerEvidenceError("complexity pair is not capacity matched")
    return {
        "winner": winner,
        "concat_capacity_matched": concat,
        "pair_validator_task_id": PAIR_VALIDATOR_TASK_ID,
        "pair_validator_artifact_sha256": pair_record["hash"],
    }


def collect(*, plan_path: Path, identity_path: Path, model_root: Path) -> dict[str, object]:
    experiment_plan = _read_sealed(plan_path, "post-winner experiment plan")
    identity = _read_sealed(identity_path, "selected method identity")
    if identity.get("seal_sha256") != SELECTED_IDENTITY_SEAL:
        raise PostWinnerEvidenceError("selected method identity seal drifted")
    document = {
        "schema_version": 1,
        "document_type": DOCUMENT_TYPE,
        "protocol_id": controlled.PROTOCOL_ID,
        "training_seed": controlled.TRAINING_SEED,
        "sample_count": controlled.SAMPLE_COUNT,
        "ground_truth_count": controlled.GROUND_TRUTH_COUNT,
        "unsupported_sample_count": 0,
        "sample_ids_sha256": controlled.SAMPLE_IDS_SHA256,
        "checkpoint_policy": controlled.CHECKPOINT_POLICY,
        "selected_method_identity_seal": SELECTED_IDENTITY_SEAL,
        "post_winner_plan_seal": experiment_plan["seal_sha256"],
        "table_v": _collect_table_v(
            experiment_plan=experiment_plan, model_root=model_root
        ),
        "table_vi": _collect_table_vi(),
        "table_vii": {
            "duration": _collect_duration(),
            "q_ge_4": "unsupported/no-extrapolation",
            "complexity": _collect_complexity(),
        },
        "producer": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        },
    }
    return controlled._sealed(document)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--selected-identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--write-token", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = collect(
        plan_path=args.plan,
        identity_path=args.selected_identity,
        model_root=args.model_root,
    )
    if args.write:
        if args.write_token != WRITE_TOKEN:
            raise PostWinnerEvidenceError(
                f"--write requires --write-token {WRITE_TOKEN}"
            )
        controlled._atomic_create(args.output, evidence)
    print(
        json.dumps(
            {
                "status": "written" if args.write else "ready",
                "table_v_rows": len(evidence["table_v"]),
                "table_vi_rows": len(evidence["table_vi"]),
                "duration_rows": len(evidence["table_vii"]["duration"]),
                "complexity_profiles": 2,
                "seal_sha256": evidence["seal_sha256"],
                "output": str(args.output) if args.write else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
