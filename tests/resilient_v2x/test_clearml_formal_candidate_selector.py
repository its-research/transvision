from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_formal_candidate_selector.py"
LEADERBOARD_SOURCE = "# fixture formal leaderboard producer\n"
AUDIT_SOURCE = "# fixture formal comparability audit producer\n"
WATCHER_SOURCE = "# fixture formal dependency watcher producer\n"
EVALUATION_SOURCE = "# fixture formal bootstrap producer\n"
LEGACY_TRAINING_SOURCE = "# fixture legacy formal bootstrap producer\n"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "clearml_formal_candidate_selector", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ClearMLArtifact = lambda raw: raw.artifact
    return module


class _Artifact:
    def __init__(self, value: object) -> None:
        self.value = value
        self.force_download_calls: list[bool] = []
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        self.record_hash = hashlib.sha256(encoded).hexdigest()
        self.content_size = max(1, len(encoded))

    def get(self, *, force_download: bool = False) -> object:
        self.force_download_calls.append(force_download)
        assert force_download is True
        return self.value


class _Task:
    def __init__(
        self,
        task_id: str,
        *,
        status: str = "completed",
        artifacts: dict[str, _Artifact] | None = None,
        reload_statuses: list[str] | None = None,
        parent: str = "",
        entry_point: str = "",
        script_source: str = "",
        parameters: dict[str, object] | None = None,
        models: dict[str, list[object]] | None = None,
    ) -> None:
        self.id = task_id
        self.server_id = task_id
        self.server_status = status
        self.artifacts = artifacts or {}
        self.server_artifacts = self.artifacts
        self.reload_statuses = list(reload_statuses or [])
        self.reload_calls = 0
        self.public_reload_calls = 0
        self._offline_mode = False
        self._reload_skip_flag = True
        self.raw_reload_result: object | None = None
        self.raw_reload_error: Exception | None = None
        self.server_parent = parent
        self.server_script = SimpleNamespace(
            repository="",
            working_dir=".",
            entry_point=entry_point,
            diff=script_source,
        )
        self.parameters = dict(parameters or {})
        self.server_parameters = self.parameters
        self.models = dict(models or {})
        self.server_models = self.models
        self._data = SimpleNamespace(
            id=task_id,
            status=status,
            parent=parent,
            script=self.server_script,
            execution=SimpleNamespace(artifacts=[]),
        )

    @property
    def data(self) -> object:
        return self._data

    @property
    def status(self) -> str:
        return str(getattr(self._data, "status", self.server_status))

    @status.setter
    def status(self, value: str) -> None:
        self.server_status = value
        self._data.status = value

    @property
    def parent(self) -> str:
        return self.server_parent

    @parent.setter
    def parent(self, value: str) -> None:
        self.server_parent = value
        self._data.parent = value

    def _reload(self) -> object:
        assert self._reload_skip_flag is False
        self.reload_calls += 1
        if self.raw_reload_error is not None:
            raise self.raw_reload_error
        if self.raw_reload_result is not None:
            return self.raw_reload_result
        if self.reload_statuses:
            self.server_status = self.reload_statuses.pop(0)
        return SimpleNamespace(
            id=self.server_id,
            status=self.server_status,
            parent=self.server_parent,
            script=self.server_script,
            execution=SimpleNamespace(
                artifacts=[
                    SimpleNamespace(
                        key=name,
                        artifact=artifact,
                        hash=getattr(artifact, "record_hash", "a" * 64),
                        content_size=getattr(artifact, "content_size", 1),
                    )
                    for name, artifact in self.server_artifacts.items()
                ]
            ),
        )

    def reload(self) -> None:
        self.public_reload_calls += 1
        raise AssertionError("public Task.reload() must never be used")

    def get_parameters(self, *, cast: bool = False) -> dict[str, object]:
        assert cast is False
        return dict(self.server_parameters)

    def get_models(self) -> dict[str, list[object]]:
        return dict(self.server_models)


class _OutputTask(_Task):
    def __init__(
        self,
        task_id: str = "f" * 32,
        *,
        upload_ok: bool = True,
        flush_ok: bool | None = None,
        parent: str = "",
    ) -> None:
        super().__init__(
            task_id,
            status="in_progress",
            parent=parent,
            entry_point="clearml_formal_candidate_selector.py",
            script_source=MODULE_PATH.read_text(encoding="utf-8"),
        )
        self.upload_ok = upload_ok
        self.flush_ok = flush_ok
        self.uploads: list[tuple[str, object, bool]] = []
        self.tags: list[str] = []
        self.flush_calls = 0

    def set_tags(self, tags: list[str]) -> None:
        self.tags = list(tags)

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        self.uploads.append((name, artifact_object, wait_on_upload))
        if self.upload_ok:
            self.artifacts[name] = _Artifact(artifact_object)
        return self.upload_ok

    def flush(self, *, wait_for_uploads: bool) -> bool | None:
        assert wait_for_uploads is True
        self.flush_calls += 1
        return self.flush_ok


class _TaskClass:
    tasks: dict[str, _Task] = {}
    current: _Task | None = None

    @classmethod
    def get_task(cls, *, task_id: str) -> _Task:
        return cls.tasks[task_id]

    @classmethod
    def current_task(cls) -> _Task | None:
        return cls.current


def _baseline_value(module, subject: str, run_index: int) -> float:
    subject_offset = module.BASELINE_SUBJECTS.index(subject)
    delay_index, condition_index = divmod(run_index, len(module.CONDITIONS))
    return float(51 + subject_offset - delay_index - condition_index)


def _best_baseline_value(module, run_index: int) -> float:
    return max(
        _baseline_value(module, subject, run_index)
        for subject in module.BASELINE_SUBJECTS
    )


def _default_margins(module) -> dict[str, list[float]]:
    return {
        module.PRIMARY_SUBJECT: [1.0] * 12,
        "support_residual": [8.0] * 12,
        "linear_no_distillation": [4.0] * 12,
        "no_distillation_peak_lr_3e4": [2.0] * 12,
    }


def _bev70_value(
    module,
    subject: str,
    run_index: int,
    margins: dict[str, list[float]],
) -> float:
    if subject in module.BASELINE_SUBJECTS:
        return _baseline_value(module, subject, run_index)
    if subject in module.CANDIDATE_ORDER:
        return _best_baseline_value(module, run_index) + margins[subject][run_index]
    return 40.0 - float(run_index)


def _metrics(
    module,
    subject: str,
    checkpoint_sha256: str,
    margins: dict[str, list[float]],
) -> dict[str, object]:
    runs = []
    for run_index, (delay, condition) in enumerate(
        (
            pair
            for delay in module.DELAYS_MS
            for pair in ((delay, item) for item in module.CONDITIONS)
        )
    ):
        bev70 = _bev70_value(module, subject, run_index, margins)
        metrics = {
            "resilient_v2x/car_bev_ap_r40_0.50": bev70 + 5.0,
            "resilient_v2x/car_bev_ap_r40_0.70": bev70,
            "resilient_v2x/car_3d_ap_r40_0.50": bev70 - 2.0,
            "resilient_v2x/car_3d_ap_r40_0.70": bev70 - 8.0,
            "resilient_v2x/sample_count": module.SAMPLE_COUNT,
            "resilient_v2x/car_ground_truth_count": module.GROUND_TRUTH_COUNT,
            "resilient_v2x/unsupported_sample_count": (module.UNSUPPORTED_SAMPLE_COUNT),
        }
        runs.append(
            {
                "condition_id": (
                    f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
                ),
                "delay_ms": delay,
                "condition": condition,
                "metrics": metrics,
                "predictions": f"predictions/{subject}/{run_index}.pkl",
                "prediction_sha256": f"{run_index + 1:064x}",
                "prediction_content_sha256": f"{run_index + 101:064x}",
                "sample_count": module.SAMPLE_COUNT,
                "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
                "ground_truth_count": module.GROUND_TRUTH_COUNT,
                "unsupported_sample_count": module.UNSUPPORTED_SAMPLE_COUNT,
            }
        )
    return {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": 12,
        "baseline": subject,
        "protocol_id": module.PROTOCOL_ID,
        "checkpoint": f"models/{subject}_epoch_50.pth",
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_content_sha256": module.MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": module.OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
        "expected_sample_count": module.SAMPLE_COUNT,
        "expected_ground_truth_count": module.GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": module.UNSUPPORTED_SAMPLE_COUNT,
        "runs": runs,
    }


def _leaderboard(module, evaluation_tasks: dict[str, _Task]) -> dict[str, object]:
    results = []
    for index, subject in enumerate(module.SUBJECT_ORDER, start=1):
        training_id = f"{index + 1000:032x}"
        model_id = f"{index + 2000:032x}"
        evaluation_id = f"{index + 3000:032x}"
        checkpoint_sha = f"{index:064x}"
        raw_metrics = (
            evaluation_tasks[evaluation_id].artifacts[module.METRICS_ARTIFACT].value
        )
        runs = module._parse_metrics(
            raw_metrics,
            subject=subject,
            checkpoint_sha256=checkpoint_sha,
        )
        results.append(
            {
                "index": index,
                "subject": subject,
                "kind": module.SUBJECT_KIND[subject],
                "training_task_id": training_id,
                "training_model_id": model_id,
                "training_checkpoint_sha256": checkpoint_sha,
                "source_revision_tree_sha256": module._expected_source(subject)[
                    "tree_sha256"
                ],
                "evaluation_task_id": evaluation_id,
                "metrics": {
                    metric_key: module._metric_summary(runs, metric_key)
                    for metric_key in module.AP_METRIC_KEYS
                },
            }
        )
    source_equivalence = module._sealed(
        {
            "schema_version": 1,
            "contract": "fixture-source-equivalence",
            "source_revisions": module.SOURCE_REVISION_CERTIFICATE_BY_TREE,
        }
    )
    module.SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = source_equivalence["seal_sha256"]
    source_subject_map = module._sealed(
        {
            "schema_version": 1,
            "contract": "fixture-source-subject-map",
            "source_revision_by_subject": {
                subject: module._expected_source(subject)["tree_sha256"]
                for subject in module.SUBJECT_ORDER
            },
        }
    )
    module.SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = source_subject_map["seal_sha256"]
    return module._sealed(
        {
            "schema_version": 3,
            "leaderboard_type": "resilient_v2x_formal_1337_leaderboard",
            "protocol_id": module.PROTOCOL_ID,
            "sample_count": module.SAMPLE_COUNT,
            "ground_truth_count": module.GROUND_TRUTH_COUNT,
            "unsupported_sample_count": module.UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(module.DELAYS_MS),
            "conditions": list(module.CONDITIONS),
            "run_count_per_subject": 12,
            "training_controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
            "watcher_task_id": module.WATCHER_TASK_ID,
            "training_manifest_seal_sha256": "a" * 64,
            "evaluation_plan_seal_sha256": "b" * 64,
            "training_provenance_task_id": module.TRAINING_PROVENANCE_TASK_ID,
            "training_provenance_seal_sha256": "e" * 64,
            "source_revision_equivalence": source_equivalence,
            "source_revision_equivalence_seal_sha256": source_equivalence[
                "seal_sha256"
            ],
            "source_revision_subject_map": source_subject_map,
            "source_revision_subject_map_seal_sha256": source_subject_map[
                "seal_sha256"
            ],
            "evaluation_source_revision_tree_sha256": module.SOURCE_TREE_SHA256,
            "evaluation_source_revision": module.SOURCE_REVISION_CERTIFICATE_BY_TREE[
                module.SOURCE_TREE_SHA256
            ],
            "subject_order": list(module.SUBJECT_ORDER),
            "subject_count": len(module.SUBJECT_ORDER),
            "baseline_subjects": list(module.CANONICAL_LEADERBOARD_BASELINE_SUBJECTS),
            "baseline_count": len(module.CANONICAL_LEADERBOARD_BASELINE_SUBJECTS),
            "metric_keys": list(module.AP_METRIC_KEYS),
            "results": results,
            "leadership": {"source": "fixture; selector recomputes independently"},
        }
    )


def _audit(
    module,
    leaderboard: dict[str, object],
    evaluation_tasks: dict[str, _Task],
) -> dict[str, object]:
    training_records = []
    evaluation_records = []
    for result in leaderboard["results"]:
        subject = str(result["subject"])
        source = module._expected_source(subject)
        index = int(result["index"])
        evaluation_id = str(result["evaluation_task_id"])
        metrics = (
            evaluation_tasks[evaluation_id].artifacts[module.METRICS_ARTIFACT].value
        )
        training_records.append(
            {
                "index": index,
                "subject": subject,
                "training_task_id": result["training_task_id"],
                "model_id": result["training_model_id"],
                "checkpoint_sha256": result["training_checkpoint_sha256"],
                "training_seed": module.TRAINING_SEED,
                "source_revision_tree_sha256": source["tree_sha256"],
                "source_dataset_id": source["dataset_id"],
                "source_archive_name": source["archive_name"],
                "source_archive_bytes": source["archive_bytes"],
                "source_archive_sha256": source["archive_sha256"],
                "training_dataset_id": module.TRAINING_DATASET_ID,
                "parent_controller_task_id": (
                    module._expected_training_parent_task_id(subject)
                ),
                "script_sha256": module._expected_training_script_sha256(subject),
                "run_contract_sha256": "d" * 64,
                "final_checkpoint_contract_content_sha256": "e" * 64,
                "run_contract_seed_fields": ["seed"],
                "gpus": 4,
                "precision": "FP32",
                "max_epochs": 50,
                "val_interval": 10,
                "common_teacher_initialization_audit_sha256": "f" * 64,
                "final_model_verification_level": ("clearml_metadata_contract_only"),
                "checkpoint_bytes_sha256_recomputed": False,
                "checkpoint_bytes_verifier": module.MODEL_BYTES_VERIFIER,
                "raw_authority_sha256": "1" * 64,
                "execution_queue_id": "4" * 32,
                "last_worker": "fixture-worker",
            }
        )
        evaluation_records.append(
            {
                "index": index,
                "subject": subject,
                "evaluation_task_id": evaluation_id,
                "training_task_id": result["training_task_id"],
                "model_id": result["training_model_id"],
                "checkpoint_sha256": result["training_checkpoint_sha256"],
                "source_revision_tree_sha256": module.SOURCE_TREE_SHA256,
                "source_dataset_id": module.SOURCE_DATASET_ID,
                "source_archive_name": module.SOURCE_ARCHIVE_NAME,
                "source_archive_bytes": module.SOURCE_ARCHIVE_BYTES,
                "source_archive_sha256": module.SOURCE_ARCHIVE_SHA256,
                "script_sha256": module.EVALUATION_SCRIPT_SHA256,
                "planned_queue": "GPU4-A100",
                "execution_queue_id": "4" * 32,
                "execution_queue_name": "GPU4-A100",
                "last_worker": "fixture-worker",
                "metrics_sha256": module._content_sha256(metrics),
                "run_count": 12,
                "sample_count_per_run": module.SAMPLE_COUNT,
                "ground_truth_count_per_run": module.GROUND_TRUTH_COUNT,
                "unsupported_sample_count_per_run": (module.UNSUPPORTED_SAMPLE_COUNT),
                "metric_keys": list(module.AP_METRIC_KEYS),
            }
        )
    return module._sealed(
        {
            "schema_version": 3,
            "document_type": "resilient_v2x_formal_1337_comparability_audit",
            "passed": True,
            "audit_scope": "protocol_and_clearml_metadata_comparability",
            "audit_task_id": module.DEFAULT_AUDIT_TASK_ID,
            "training_controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
            "training_provenance_task_id": module.TRAINING_PROVENANCE_TASK_ID,
            "watcher_task_id": module.WATCHER_TASK_ID,
            "leaderboard_task_id": module.DEFAULT_LEADERBOARD_TASK_ID,
            "protocol_id": module.PROTOCOL_ID,
            "evaluation_source_revision_tree_sha256": leaderboard[
                "evaluation_source_revision_tree_sha256"
            ],
            "evaluation_source_revision": leaderboard["evaluation_source_revision"],
            "source_revision_equivalence": leaderboard["source_revision_equivalence"],
            "source_revision_equivalence_seal_sha256": leaderboard[
                "source_revision_equivalence_seal_sha256"
            ],
            "source_revision_subject_map": leaderboard["source_revision_subject_map"],
            "source_revision_subject_map_seal_sha256": leaderboard[
                "source_revision_subject_map_seal_sha256"
            ],
            "source_revision_counts": {
                module.SOURCE_TREE_SHA256: 21,
                module.NEW_SOURCE_TREE_SHA256: 5,
            },
            "training_dataset_id": module.TRAINING_DATASET_ID,
            "training_seed": module.TRAINING_SEED,
            "training_seed_evidence": "all_26_live_run_contracts",
            "legacy_unseeded_training_manifest": True,
            "legacy_source_c_seed_schema": "explicit_run_contract_seed_only",
            "training_script_equivalence": {
                "legacy_script_sha256": module.LEGACY_TRAINING_SCRIPT_SHA256,
                "canonical_script_sha256": module.CANONICAL_TRAINING_SCRIPT_SHA256,
                "legacy_script_subjects": list(module.LEGACY_SCRIPT_SUBJECTS),
                "only_difference": "NESTED_TEACHER_EXPERIMENTS membership",
                "runtime_usage_closure": [
                    "expect_nested_teacher_keyword",
                    "expected_nested_teacher_contract_field",
                ],
            },
            "evaluation_script_sha256": module.EVALUATION_SCRIPT_SHA256,
            "checkpoint_policy": module.CHECKPOINT_POLICY,
            "sample_count": module.SAMPLE_COUNT,
            "ground_truth_count": module.GROUND_TRUTH_COUNT,
            "unsupported_sample_count": module.UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(module.DELAYS_MS),
            "conditions": list(module.CONDITIONS),
            "run_count_per_subject": 12,
            "subject_order": list(module.SUBJECT_ORDER),
            "subject_count": len(module.SUBJECT_ORDER),
            "total_evaluation_run_count": len(module.SUBJECT_ORDER) * 12,
            "metric_keys": list(module.AP_METRIC_KEYS),
            "final_model_verification_level": "clearml_metadata_contract_only",
            "checkpoint_bytes_sha256_recomputed": False,
            "checkpoint_bytes_verifier": module.MODEL_BYTES_VERIFIER,
            "checkpoint_bytes_verification_dependency": (
                "separate_collect_clearml_formal_models_byte_audit"
            ),
            "training_manifest_seal_sha256": "a" * 64,
            "training_progress_seal_sha256": "d" * 64,
            "training_provenance_seal_sha256": "e" * 64,
            "training_summary_seal_sha256": "c" * 64,
            "evaluation_plan_seal_sha256": "b" * 64,
            "leaderboard_seal_sha256": leaderboard["seal_sha256"],
            "training_tasks": training_records,
            "evaluation_tasks": evaluation_records,
        }
    )


def _world(module, *, margins: dict[str, list[float]] | None = None):
    module.LEADERBOARD_SCRIPT_SHA256 = hashlib.sha256(
        LEADERBOARD_SOURCE.encode("utf-8")
    ).hexdigest()
    module.AUDIT_SCRIPT_SHA256 = hashlib.sha256(
        AUDIT_SOURCE.encode("utf-8")
    ).hexdigest()
    module.LEGACY_TRAINING_SCRIPT_SHA256 = hashlib.sha256(
        LEGACY_TRAINING_SOURCE.encode("utf-8")
    ).hexdigest()
    module.CANONICAL_TRAINING_SCRIPT_SHA256 = hashlib.sha256(
        EVALUATION_SOURCE.encode("utf-8")
    ).hexdigest()
    module.EVALUATION_SCRIPT_SHA256 = module.CANONICAL_TRAINING_SCRIPT_SHA256
    margins = margins or _default_margins(module)
    evaluation_tasks: dict[str, _Task] = {}
    for index, subject in enumerate(module.SUBJECT_ORDER, start=1):
        checkpoint_sha = f"{index:064x}"
        training_id = f"{index + 1000:032x}"
        model_id = f"{index + 2000:032x}"
        evaluation_id = f"{index + 3000:032x}"
        evaluation_tasks[evaluation_id] = _Task(
            evaluation_id,
            parent=module.TRAINING_CONTROLLER_TASK_ID,
            entry_point=module.TRAINING_ENTRY_POINT,
            script_source=EVALUATION_SOURCE,
            parameters={
                "Args/stage": "baseline_validate",
                "Args/controlled_baseline": subject,
                "Args/controlled_baseline_task_id": training_id,
                "Args/controlled_baseline_model_id": model_id,
                "Args/controlled_baseline_checkpoint_sha256": checkpoint_sha,
                "Args/predecessor_task_id": training_id,
                "Args/source_dataset_id": module.SOURCE_DATASET_ID,
                "Args/source_archive_name": module.SOURCE_ARCHIVE_NAME,
                "Args/source_archive_bytes": module.SOURCE_ARCHIVE_BYTES,
                "Args/source_archive_sha256": module.SOURCE_ARCHIVE_SHA256,
                "Args/training_dataset_id": module.TRAINING_DATASET_ID,
                "Args/gpus": 4,
                "Args/max_epochs": 50,
                "Args/amp": False,
            },
            models={"input": [SimpleNamespace(id=model_id)]},
            artifacts={
                module.METRICS_ARTIFACT: _Artifact(
                    _metrics(module, subject, checkpoint_sha, margins)
                )
            },
        )
    leaderboard = _leaderboard(module, evaluation_tasks)
    audit = _audit(module, leaderboard, evaluation_tasks)
    audit_task = _Task(
        module.DEFAULT_AUDIT_TASK_ID,
        parent=module.DEFAULT_LEADERBOARD_TASK_ID,
        entry_point=module.AUDIT_ENTRY_POINT,
        script_source=AUDIT_SOURCE,
        parameters={
            "Args/training_controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
            "Args/training_provenance_task_id": module.TRAINING_PROVENANCE_TASK_ID,
            "Args/watcher_task_id": module.WATCHER_TASK_ID,
            "Args/leaderboard_task_id": module.DEFAULT_LEADERBOARD_TASK_ID,
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
        },
        artifacts={module.AUDIT_ARTIFACT: _Artifact(audit)},
    )
    leaderboard_task = _Task(
        module.DEFAULT_LEADERBOARD_TASK_ID,
        parent=module.WATCHER_TASK_ID,
        entry_point=module.LEADERBOARD_ENTRY_POINT,
        script_source=LEADERBOARD_SOURCE,
        parameters={
            "Args/training_controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
            "Args/watcher_task_id": module.WATCHER_TASK_ID,
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
        },
        artifacts={module.LEADERBOARD_ARTIFACT: _Artifact(leaderboard)},
    )
    output = _OutputTask(parent=module.DEFAULT_AUDIT_TASK_ID)
    tasks = {
        module.DEFAULT_AUDIT_TASK_ID: audit_task,
        module.DEFAULT_LEADERBOARD_TASK_ID: leaderboard_task,
        **evaluation_tasks,
    }
    _TaskClass.tasks = tasks
    _TaskClass.current = output
    args = SimpleNamespace(
        audit_task_id=module.DEFAULT_AUDIT_TASK_ID,
        leaderboard_task_id=module.DEFAULT_LEADERBOARD_TASK_ID,
        poll_seconds=1.0,
        timeout_hours=1.0,
    )
    return SimpleNamespace(
        args=args,
        tasks=tasks,
        output=output,
        audit=audit,
        audit_task=audit_task,
        leaderboard=leaderboard,
        leaderboard_task=leaderboard_task,
        evaluation_tasks=evaluation_tasks,
    )


def _enable_formal_inputs(module, world) -> None:
    module.WATCHER_SCRIPT_SHA256 = hashlib.sha256(
        WATCHER_SOURCE.encode("utf-8")
    ).hexdigest()
    leaderboard = copy.deepcopy(world.leaderboard)
    audit = copy.deepcopy(world.audit)
    audit_training = {str(row["subject"]): row for row in audit["training_tasks"]}
    audit_evaluation = {str(row["subject"]): row for row in audit["evaluation_tasks"]}
    training_tasks: dict[str, _Task] = {}
    for subject in module.FORMAL_INPUT_SUBJECTS:
        result = next(
            row for row in leaderboard["results"] if row["subject"] == subject
        )
        source = module._expected_source(subject)
        training_id = str(result["training_task_id"])
        model_id = str(result["training_model_id"])
        checkpoint_sha = str(result["training_checkpoint_sha256"])
        checkpoint_size = 100_000 + int(result["index"])
        config_sha = f"{int(result['index']) + 400:064x}"
        run_contract = {
            "schema_version": 1,
            "mode": "experiment_from_task",
            "task_id": training_id,
            "experiment": subject,
            "source_dataset_id": source["dataset_id"],
            "training_dataset_id": module.TRAINING_DATASET_ID,
            "predecessor_task_id": f"{int(result['index']) + 500:032x}",
            "gpus": 4,
            "global_batch_size": 8,
            "max_epochs": 50,
            "val_interval": 10,
            "precision": "FP32",
            "amp": False,
            "condition_evaluation": False,
            "seed": module.TRAINING_SEED,
            "source_archive": {
                "name": source["archive_name"],
                "size_bytes": source["archive_bytes"],
                "sha256": source["archive_sha256"],
            },
            "config": {
                "declared": f"configs/resilient_v2x/formal/{subject}.py",
                "config_sha256": config_sha,
                "resolved_config_sha256": config_sha,
            },
            "teacher": {
                "task_id": module.TEACHER_TASK_ID,
                "model_id": module.TEACHER_MODEL_ID,
                "sha256": module.TEACHER_CHECKPOINT_SHA256,
            },
        }
        initialization = {
            "schema_version": 1,
            "subject": subject,
            "teacher_checkpoint_sha256": module.TEACHER_CHECKPOINT_SHA256,
        }
        final = {
            "model_id": model_id,
            "name": f"ResilientV2X {subject} final checkpoint",
            "url": f"http://fileserver/{subject}_epoch_50.pth",
            "filename": "epoch_50.pth",
            "size_bytes": checkpoint_size,
            "sha256": checkpoint_sha,
        }
        audit_training[subject]["run_contract_sha256"] = module._content_sha256(
            run_contract
        )
        audit_training[subject]["common_teacher_initialization_audit_sha256"] = (
            module._content_sha256(initialization)
        )
        audit_training[subject]["final_checkpoint_contract_content_sha256"] = (
            module._content_sha256(final)
        )
        training_tasks[training_id] = _Task(
            training_id,
            parent=module._expected_training_parent_task_id(subject),
            entry_point=module.TRAINING_ENTRY_POINT,
            script_source=EVALUATION_SOURCE,
            parameters={
                "Args/stage": "all",
                "Args/experiment_from_task": subject,
                "Args/source_dataset_id": source["dataset_id"],
                "Args/source_archive_name": source["archive_name"],
                "Args/source_archive_bytes": source["archive_bytes"],
                "Args/source_archive_sha256": source["archive_sha256"],
                "Args/training_dataset_id": module.TRAINING_DATASET_ID,
                "Args/gpus": 4,
                "Args/max_epochs": 50,
                "Args/amp": False,
            },
            artifacts={
                module.RUN_CONTRACT_ARTIFACT: _Artifact(run_contract),
                module.INITIALIZATION_AUDIT_ARTIFACT: _Artifact(initialization),
                module.FINAL_CHECKPOINT_CONTRACT_ARTIFACT: _Artifact(final),
            },
        )
        evaluation_id = str(result["evaluation_task_id"])
        evaluation = world.evaluation_tasks[evaluation_id]
        evaluation_plan = {
            "schema_version": 1,
            "plan_type": "resilient_v2x_controlled_baseline_evaluation",
            "protocol_id": module.PROTOCOL_ID,
            "baseline": subject,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_content_sha256": module.MANIFEST_CONTENT_SHA256,
            "overlay_index_content_sha256": module.OVERLAY_INDEX_CONTENT_SHA256,
            "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
            "expected_sample_count": module.SAMPLE_COUNT,
            "expected_ground_truth_count": module.GROUND_TRUTH_COUNT,
            "expected_unsupported_sample_count": module.UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(module.DELAYS_MS),
            "conditions": list(module.CONDITIONS),
            "runs": [
                {
                    "condition_id": (
                        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
                    ),
                    "delay_ms": delay,
                    "condition": condition,
                    "agent_scope": "E+R",
                    "duration_ticks": 1,
                }
                for delay in module.DELAYS_MS
                for condition in module.CONDITIONS
            ],
        }
        evaluation_run_contract = {
            "schema_version": 1,
            "mode": "baseline_validate",
            "task_id": evaluation_id,
            "baseline": subject,
            "baseline_task_id": training_id,
            "predecessor_task_id": training_id,
            "training_dataset_id": module.TRAINING_DATASET_ID,
            "protocol_id": module.PROTOCOL_ID,
            "expected_sample_count": module.SAMPLE_COUNT,
            "expected_ground_truth_count": module.GROUND_TRUTH_COUNT,
            "expected_run_count": 12,
            "expected_manifest_content_sha256": module.MANIFEST_CONTENT_SHA256,
            "expected_overlay_index_content_sha256": (
                module.OVERLAY_INDEX_CONTENT_SHA256
            ),
            "expected_sample_ids_sha256": module.SAMPLE_IDS_SHA256,
            "checkpoint": {
                "task_id": training_id,
                "model_id": model_id,
                "sha256": checkpoint_sha,
                "size_bytes": checkpoint_size,
            },
        }
        evaluation.server_artifacts.update(
            {
                module.RUN_CONTRACT_ARTIFACT: _Artifact(evaluation_run_contract),
                module.EVALUATION_PLAN_ARTIFACT: _Artifact(evaluation_plan),
                module.PREDICTION_EVIDENCE_ARTIFACT: _Artifact(
                    {"subject": subject, "archive": "fixture"}
                ),
            }
        )
        audit_evaluation[subject]["metrics_sha256"] = module._content_sha256(
            evaluation.server_artifacts[module.METRICS_ARTIFACT].value
        )
    formal_plan = module._sealed(
        {
            "schema_version": 2,
            "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
            "training_controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
            "training_provenance_task_id": module.TRAINING_PROVENANCE_TASK_ID,
            "training_provenance_seal_sha256": leaderboard[
                "training_provenance_seal_sha256"
            ],
            "source_revision_equivalence": leaderboard["source_revision_equivalence"],
            "source_revision_equivalence_seal_sha256": leaderboard[
                "source_revision_equivalence_seal_sha256"
            ],
            "source_revision_subject_map": leaderboard["source_revision_subject_map"],
            "source_revision_subject_map_seal_sha256": leaderboard[
                "source_revision_subject_map_seal_sha256"
            ],
            "evaluation_source_revision_tree_sha256": leaderboard[
                "evaluation_source_revision_tree_sha256"
            ],
            "evaluation_source_revision": leaderboard["evaluation_source_revision"],
            "evaluation_template_task_id": "8" * 32,
            "protocol_id": module.PROTOCOL_ID,
            "sample_count": module.SAMPLE_COUNT,
            "delays_ms": list(module.DELAYS_MS),
            "conditions": list(module.CONDITIONS),
            "run_count": 12,
            "subject_order": list(module.SUBJECT_ORDER),
            "entries": [
                {
                    "subject": subject,
                    "evaluation_task_id": next(
                        row["evaluation_task_id"]
                        for row in leaderboard["results"]
                        if row["subject"] == subject
                    ),
                    "queue": audit_evaluation[subject]["planned_queue"],
                }
                for subject in module.SUBJECT_ORDER
            ],
        }
    )
    leaderboard["evaluation_plan_seal_sha256"] = formal_plan["seal_sha256"]
    leaderboard = module._sealed(leaderboard)
    audit["evaluation_plan_seal_sha256"] = formal_plan["seal_sha256"]
    audit["leaderboard_seal_sha256"] = leaderboard["seal_sha256"]
    audit = module._sealed(audit)
    world.leaderboard = leaderboard
    world.audit = audit
    world.leaderboard_task.server_artifacts = {
        module.LEADERBOARD_ARTIFACT: _Artifact(leaderboard)
    }
    world.leaderboard_task.artifacts = world.leaderboard_task.server_artifacts
    world.audit_task.server_artifacts = {module.AUDIT_ARTIFACT: _Artifact(audit)}
    world.audit_task.artifacts = world.audit_task.server_artifacts
    watcher_task = _Task(
        module.WATCHER_TASK_ID,
        parent=module.TRAINING_PROVENANCE_TASK_ID,
        entry_point=module.WATCHER_ENTRY_POINT,
        script_source=WATCHER_SOURCE,
        parameters={
            "Args/training_controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
            "Args/training_provenance_task_id": module.TRAINING_PROVENANCE_TASK_ID,
            "Args/expected_eval_script_sha256": module.EVALUATION_SCRIPT_SHA256,
            "Args/expected_training_dataset_id": module.TRAINING_DATASET_ID,
            "Args/expected_source_dataset_id": module.SOURCE_DATASET_ID,
            "Args/expected_source_archive_sha256": module.SOURCE_ARCHIVE_SHA256,
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
        },
        artifacts={module.FORMAL_EVALUATION_PLAN_ARTIFACT: _Artifact(formal_plan)},
    )
    world.watcher_task = watcher_task
    world.training_tasks = training_tasks
    world.formal_plan = formal_plan
    world.tasks[module.WATCHER_TASK_ID] = watcher_task
    world.tasks.update(training_tasks)
    world.args.formal_inputs_artifact = module.FORMAL_INPUTS_ARTIFACT


def _run(module, world):
    return module.run(
        world.args,
        task_class=_TaskClass,
        output_task=world.output,
        monotonic_clock=lambda: 0.0,
        sleeper=lambda _: None,
    )


def test_parser_defaults_pin_the_active_post_formal_dependencies() -> None:
    module = _load_module()

    args = module._parser().parse_args([])

    assert args.audit_task_id == "e19bab92884248f4ac07167e7eb66170"
    assert args.leaderboard_task_id == "f502bdd329ad4ef4b4b6cf5c5f52aba0"
    assert module.TRAINING_CONTROLLER_TASK_ID == "1011e98e10f64c428880af1d4b1d542b"
    assert module.TRAINING_PROVENANCE_TASK_ID == "7274a1a5344a44c18aa7bcc6d1cb2e95"
    assert module.WATCHER_TASK_ID == "7734387ddfb74b11ba6d84f3fea0bb97"
    assert module.LEADERBOARD_SCRIPT_SHA256 == (
        "1da0cf5dd4435c6ae85d5a474b5a67ec8435f50e9878a3d8f0c1d2872356524b"
    )
    assert module.AUDIT_SCRIPT_SHA256 == (
        "4746ae18bb67757c974d96e851f19cb01872f460aaf3b334609a0de988aaabaa"
    )
    assert args.formal_inputs_artifact == module.FORMAL_INPUTS_ARTIFACT
    assert args.poll_seconds == 60.0
    assert args.timeout_hours == 720.0


def test_schema_v2_formal_inputs_are_published_and_consumed_fail_closed() -> None:
    from tools.resilient_v2x import build_final_single_seed_selection as final_selector

    module = _load_module()
    world = _world(module)
    _enable_formal_inputs(module, world)

    _run(module, world)

    assert [name for name, _, _ in world.output.uploads] == [
        module.FORMAL_INPUTS_ARTIFACT,
        module.SELECTION_ARTIFACT,
    ]
    formal_inputs = world.output.artifacts[module.FORMAL_INPUTS_ARTIFACT].value
    assert formal_inputs["schema_version"] == 2
    assert formal_inputs["document_type"] == module.FORMAL_INPUTS_DOCUMENT_TYPE
    assert formal_inputs["subject_order"] == list(module.FORMAL_INPUT_SUBJECTS)
    assert (
        formal_inputs["authority_bindings"]["watcher_plan_seal_sha256"]
        == (world.formal_plan["seal_sha256"])
    )
    runs, bindings, seal = final_selector._validate_formal_inputs(formal_inputs)
    assert tuple(runs) == module.FORMAL_INPUT_SUBJECTS
    assert tuple(bindings) == module.FORMAL_INPUT_SUBJECTS
    assert seal == formal_inputs["seal_sha256"]
    assert all(len(rows) == 12 for rows in runs.values())


def test_formal_inputs_reject_watcher_plan_task_drift_before_write() -> None:
    module = _load_module()
    world = _world(module)
    _enable_formal_inputs(module, world)
    changed = copy.deepcopy(world.formal_plan)
    changed["entries"][0]["evaluation_task_id"] = "9" * 32
    changed = module._sealed(changed)
    world.watcher_task.server_artifacts = {
        module.FORMAL_EVALUATION_PLAN_ARTIFACT: _Artifact(changed)
    }
    world.watcher_task.artifacts = world.watcher_task.server_artifacts

    with pytest.raises(ValueError, match="entry .* mismatch"):
        _run(module, world)

    assert world.output.uploads == []


def test_formal_inputs_reject_artifact_byte_hash_drift_before_write() -> None:
    module = _load_module()
    world = _world(module)
    _enable_formal_inputs(module, world)
    subject = module.FORMAL_INPUT_SUBJECTS[0]
    training = next(
        task
        for task in world.training_tasks.values()
        if task.parameters["Args/experiment_from_task"] == subject
    )
    training.server_artifacts[module.RUN_CONTRACT_ARTIFACT].record_hash = "not-a-sha"

    with pytest.raises(ValueError, match="byte hash must be a lowercase SHA-256"):
        _run(module, world)

    assert world.output.uploads == []


def test_primary_single_seed_result_is_final_when_sota_gate_passes() -> None:
    module = _load_module()
    world = _world(module)

    payload = _run(module, world)

    assert payload["status"] == "selected"
    assert payload["selected_candidate"] == module.PRIMARY_SUBJECT
    assert payload["selected_candidate_gate_passed"] is True
    assert payload["architecture_revision_required"] is False
    assert payload["selection_claim"] == ("single_seed_1337_bev70_sota_gate_winner")
    assert payload["selection_is_final"] is True
    assert payload["claim_scope"] == "metrics_and_clearml_metadata_only"
    assert payload["requires_checkpoint_byte_audit"] is True
    assert not any(str(key).endswith("seed_confirmation") for key in payload)
    assert payload["training_provenance_task_id"] == module.TRAINING_PROVENANCE_TASK_ID
    assert payload["training_progress_seal_sha256"] == "d" * 64
    assert payload["training_provenance_seal_sha256"] == "e" * 64
    assert payload["schema_version"] == 4
    assert (
        payload["source_revision_equivalence"]
        == world.leaderboard["source_revision_equivalence"]
    )
    assert (
        payload["source_revision_subject_map"]
        == world.leaderboard["source_revision_subject_map"]
    )
    assert payload["evaluation_source_revision_tree_sha256"] == (
        module.SOURCE_TREE_SHA256
    )
    assert (
        payload["training_script_equivalence"]
        == world.audit["training_script_equivalence"]
    )
    assert payload["evaluation_script_sha256"] == module.EVALUATION_SCRIPT_SHA256
    assert payload["ranked_candidates"][0] == module.PRIMARY_SUBJECT
    assert payload["performance_ranked_candidates"][0] == module.PRIMARY_SUBJECT
    assert payload["candidate_count"] == 1
    assert payload["baseline_count"] == 5
    assert module._sealed(payload) == payload
    assert world.output.uploads == [(module.SELECTION_ARTIFACT, payload, True)]
    assert world.output.flush_calls == 1
    assert "formal-candidate-selection" in world.output.tags

    primary = next(
        item
        for item in payload["candidate_results"]
        if item["subject"] == module.PRIMARY_SUBJECT
    )
    assert primary["gate_passed"] is True
    assert primary["performance_rank"] == 1
    assert primary["rank"] == 1
    assert primary["conditions_won_fraction"] == "12/12"
    assert primary["ranking_values"] == {
        "worst_12_margin": 1.0,
        "mean_12_margin": 1.0,
        "full_0ms_margin": 1.0,
        "fixed_candidate_order": 0,
    }
    assert all(
        comparison["margin"] == 1.0 for comparison in primary["condition_comparisons"]
    )


def test_primary_gate_failure_requires_revision() -> None:
    module = _load_module()
    margins = _default_margins(module)
    margins[module.PRIMARY_SUBJECT][-1] = -1.0
    world = _world(module, margins=margins)

    payload = _run(module, world)

    assert payload["status"] == "architecture_revision_required"
    assert payload["selected_candidate"] is None
    assert payload["ranked_candidates"] == [module.PRIMARY_SUBJECT]
    primary = payload["candidate_results"][0]
    assert primary["gate_passed"] is False
    assert primary["conditions_won_fraction"] == "11/12"
    assert primary["rank"] == 1


def test_worst_12_failure_cannot_form_a_final_selection() -> None:
    module = _load_module()
    margins = _default_margins(module)
    margins[module.PRIMARY_SUBJECT] = [20.0] * 11 + [-0.1]
    margins["support_residual"] = [0.1] * 12
    margins["linear_no_distillation"] = [-1.0] * 12
    margins["no_distillation_peak_lr_3e4"] = [-2.0] * 12
    world = _world(module, margins=margins)

    payload = _run(module, world)

    assert payload["selected_candidate"] is None
    assert payload["candidate_results"][0]["gate_passed"] is False


def test_full_0ms_exact_half_point_deficit_passes() -> None:
    module = _load_module()
    margins = _default_margins(module)
    margins[module.PRIMARY_SUBJECT][0] = -0.5
    world = _world(module, margins=margins)

    payload = _run(module, world)

    assert payload["selected_candidate"] == module.PRIMARY_SUBJECT
    aggregate = payload["candidate_results"][0]["aggregate_comparisons"]
    assert aggregate["full_0ms"]["margin"] == -0.5
    assert aggregate["full_0ms"]["passes_gate"] is True


def test_no_passing_candidate_requires_revision_and_makes_no_leadership_claim() -> None:
    module = _load_module()
    margins = _default_margins(module)
    for candidate in module.CANDIDATE_ORDER:
        margins[candidate] = [1.0] * 11 + [-1.0]
    world = _world(module, margins=margins)

    payload = _run(module, world)

    assert payload["status"] == "architecture_revision_required"
    assert payload["selected_candidate"] is None
    assert payload["selected_candidate_gate_passed"] is False
    assert payload["selection_claim"] is None
    assert payload["architecture_revision_required"] is True
    assert all(item["gate_passed"] is False for item in payload["candidate_results"])


def test_waits_for_both_dependencies_before_reading_artifacts() -> None:
    module = _load_module()
    world = _world(module)
    world.audit_task.status = "created"
    world.audit_task.reload_statuses = ["in_progress", "completed"]
    world.leaderboard_task.status = "queued"
    world.leaderboard_task.reload_statuses = ["in_progress", "completed"]
    sleeps: list[float] = []

    payload = module.run(
        world.args,
        task_class=_TaskClass,
        output_task=world.output,
        monotonic_clock=lambda: 0.0,
        sleeper=sleeps.append,
    )

    assert payload["status"] == "selected"
    assert sleeps == [1.0]
    assert world.audit_task.reload_calls >= 3
    assert world.leaderboard_task.reload_calls >= 3
    assert world.audit_task.public_reload_calls == 0
    assert world.leaderboard_task.public_reload_calls == 0
    assert world.audit_task._reload_skip_flag is True
    assert world.leaderboard_task._reload_skip_flag is True


@pytest.mark.parametrize("dependency", ["audit_task", "leaderboard_task"])
def test_dependency_failure_is_terminal(dependency: str) -> None:
    module = _load_module()
    world = _world(module)
    getattr(world, dependency).status = "failed"

    with pytest.raises(RuntimeError, match="ended as 'failed'"):
        _run(module, world)


def test_pending_dependencies_time_out() -> None:
    module = _load_module()
    world = _world(module)
    world.audit_task.status = "queued"
    times = iter([0.0, 4_000.0])

    with pytest.raises(TimeoutError, match="candidate dependencies"):
        module.run(
            world.args,
            task_class=_TaskClass,
            output_task=world.output,
            monotonic_clock=lambda: next(times),
            sleeper=lambda _: None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("poll_seconds", 0),
        ("poll_seconds", float("nan")),
        ("timeout_hours", -1),
        ("timeout_hours", True),
    ],
)
def test_rejects_invalid_wait_configuration(field: str, value: object) -> None:
    module = _load_module()
    world = _world(module)
    setattr(world.args, field, value)

    with pytest.raises(ValueError, match="finite and positive"):
        _run(module, world)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("passed", False, "passed mismatch"),
        ("protocol_id", "wrong", "protocol_id mismatch"),
        ("sample_count", 1336, "sample_count mismatch"),
        ("ground_truth_count", 11329, "ground_truth_count mismatch"),
        ("unsupported_sample_count", 1, "unsupported_sample_count mismatch"),
        ("leaderboard_task_id", "1" * 32, "leaderboard_task_id mismatch"),
        ("watcher_task_id", "2" * 32, "watcher_task_id mismatch"),
        (
            "final_model_verification_level",
            "checkpoint_bytes_verified",
            "final_model_verification_level mismatch",
        ),
        (
            "checkpoint_bytes_sha256_recomputed",
            True,
            "checkpoint byte verification mismatch",
        ),
        (
            "evaluation_script_sha256",
            "0" * 64,
            "evaluation_script_sha256 mismatch",
        ),
        (
            "training_seed_evidence",
            "wrong",
            "training_seed_evidence mismatch",
        ),
        (
            "legacy_unseeded_training_manifest",
            False,
            "legacy_unseeded_training_manifest mismatch",
        ),
        (
            "legacy_source_c_seed_schema",
            "wrong",
            "legacy_source_c_seed_schema mismatch",
        ),
    ],
)
def test_rejects_resealed_audit_contract_drift(
    field: str, value: object, message: str
) -> None:
    module = _load_module()
    world = _world(module)
    changed = copy.deepcopy(world.audit)
    changed[field] = value
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(changed)

    with pytest.raises(ValueError, match=message):
        _run(module, world)


@pytest.mark.parametrize(
    "mutation",
    [
        "legacy_sha",
        "canonical_sha",
        "legacy_subject_order",
        "support_uses_canonical",
        "ffnet_uses_legacy",
        "support_parent",
        "provenance_task",
        "progress_seal",
        "provenance_seal",
        "evaluation_queue",
    ],
)
def test_rejects_resealed_mixed_training_provenance_drift(mutation: str) -> None:
    module = _load_module()
    world = _world(module)
    changed = copy.deepcopy(world.audit)
    equivalence = changed["training_script_equivalence"]
    records = changed["training_tasks"]
    if mutation == "legacy_sha":
        equivalence["legacy_script_sha256"] = module.CANONICAL_TRAINING_SCRIPT_SHA256
    elif mutation == "canonical_sha":
        equivalence["canonical_script_sha256"] = module.LEGACY_TRAINING_SCRIPT_SHA256
    elif mutation == "legacy_subject_order":
        equivalence["legacy_script_subjects"].reverse()
    elif mutation == "support_uses_canonical":
        records[0]["script_sha256"] = module.CANONICAL_TRAINING_SCRIPT_SHA256
    elif mutation == "ffnet_uses_legacy":
        ffnet = next(record for record in records if record["subject"] == "ffnet")
        ffnet["script_sha256"] = module.LEGACY_TRAINING_SCRIPT_SHA256
    elif mutation == "support_parent":
        records[0]["parent_controller_task_id"] = module.TRAINING_CONTROLLER_TASK_ID
    elif mutation == "provenance_task":
        changed["training_provenance_task_id"] = "9" * 32
    elif mutation == "progress_seal":
        changed["training_progress_seal_sha256"] = "not-a-sha"
    elif mutation == "provenance_seal":
        changed["training_provenance_seal_sha256"] = "not-a-sha"
    else:
        changed["evaluation_tasks"][0]["execution_queue_name"] = "GPU4-V100"
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(changed)

    with pytest.raises(ValueError):
        _run(module, world)


@pytest.mark.parametrize("target", ["leaderboard_task", "audit_task"])
def test_rejects_extra_formal_producer_parameter(target: str) -> None:
    module = _load_module()
    world = _world(module)
    getattr(world, target).parameters["Args/attacker_extra"] = "1"

    with pytest.raises(RuntimeError, match="exact parameter inventory mismatch"):
        _run(module, world)


def test_audit_passed_requires_a_real_boolean() -> None:
    module = _load_module()
    world = _world(module)
    changed = copy.deepcopy(world.audit)
    changed["passed"] = 1
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(changed)

    with pytest.raises(ValueError, match="passed mismatch"):
        _run(module, world)


def test_rejects_unsealed_audit_and_leaderboard_tampering() -> None:
    module = _load_module()
    world = _world(module)
    world.audit["sample_count"] = 1

    with pytest.raises(ValueError, match="audit seal SHA-256 mismatch"):
        _run(module, world)

    world = _world(module)
    world.leaderboard["sample_count"] = 1
    with pytest.raises(ValueError, match="leaderboard seal SHA-256 mismatch"):
        _run(module, world)


@pytest.mark.parametrize(
    "field",
    ["training_manifest_seal_sha256", "evaluation_plan_seal_sha256"],
)
def test_audit_seals_must_join_the_leaderboard_seals(field: str) -> None:
    module = _load_module()
    world = _world(module)
    changed = copy.deepcopy(world.audit)
    changed[field] = "d" * 64
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(changed)

    with pytest.raises(ValueError, match=rf"{field} mismatch"):
        _run(module, world)


@pytest.mark.parametrize(
    ("target", "parent", "message"),
    [
        ("leaderboard_task", "1" * 32, "leaderboard parent mismatch"),
        ("audit_task", "2" * 32, "comparability audit parent mismatch"),
        ("output", "3" * 32, "selector task parent drifted"),
    ],
)
def test_rejects_formal_parent_chain_drift(
    target: str, parent: str, message: str
) -> None:
    module = _load_module()
    world = _world(module)
    getattr(world, target).parent = parent

    with pytest.raises(RuntimeError, match=message):
        _run(module, world)
    assert world.output.tags == []
    assert world.output.uploads == []


@pytest.mark.parametrize("target", ["leaderboard_task", "audit_task"])
def test_rejects_producer_script_drift(target: str) -> None:
    module = _load_module()
    world = _world(module)
    getattr(world, target).data.script.diff += "# drift\n"

    with pytest.raises(RuntimeError, match="script SHA-256 mismatch"):
        _run(module, world)


@pytest.mark.parametrize(
    ("target", "parameter"),
    [
        ("leaderboard_task", "Args/watcher_task_id"),
        ("audit_task", "Args/leaderboard_task_id"),
    ],
)
def test_rejects_producer_parameter_drift(target: str, parameter: str) -> None:
    module = _load_module()
    world = _world(module)
    getattr(world, target).parameters[parameter] = "9" * 32

    with pytest.raises(RuntimeError, match="parameter .* mismatch"):
        _run(module, world)


@pytest.mark.parametrize("mutation", ["drop", "reorder", "duplicate_eval"])
def test_rejects_invalid_26_result_leaderboard(mutation: str) -> None:
    module = _load_module()
    world = _world(module)
    changed = copy.deepcopy(world.leaderboard)
    if mutation == "drop":
        changed["results"].pop()
    elif mutation == "reorder":
        changed["results"][0], changed["results"][1] = (
            changed["results"][1],
            changed["results"][0],
        )
    else:
        changed["results"][1]["evaluation_task_id"] = changed["results"][0][
            "evaluation_task_id"
        ]
    changed = module._sealed(changed)
    world.leaderboard_task.artifacts[module.LEADERBOARD_ARTIFACT].value = changed
    audit = copy.deepcopy(world.audit)
    audit["leaderboard_seal_sha256"] = changed["seal_sha256"]
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(audit)

    with pytest.raises(ValueError):
        _run(module, world)


def test_actual_metrics_are_rehashed_and_not_replaced_by_leaderboard_summaries() -> (
    None
):
    module = _load_module()
    world = _world(module)
    primary_result = next(
        item
        for item in world.leaderboard["results"]
        if item["subject"] == module.PRIMARY_SUBJECT
    )
    evaluation_id = primary_result["evaluation_task_id"]
    raw = world.evaluation_tasks[evaluation_id].artifacts[module.METRICS_ARTIFACT].value
    raw["runs"][0]["metrics"][module.LEADERSHIP_METRIC] += 7.0

    with pytest.raises(RuntimeError, match="evaluation metrics SHA-256 mismatch"):
        _run(module, world)


def test_rejects_resealed_audit_when_live_metrics_disagree_with_leaderboard() -> None:
    module = _load_module()
    world = _world(module)
    primary_result = next(
        item
        for item in world.leaderboard["results"]
        if item["subject"] == module.PRIMARY_SUBJECT
    )
    evaluation_id = primary_result["evaluation_task_id"]
    raw = world.evaluation_tasks[evaluation_id].artifacts[module.METRICS_ARTIFACT].value
    raw["runs"][0]["metrics"][module.LEADERSHIP_METRIC] += 7.0
    audit = copy.deepcopy(world.audit)
    record = next(
        item
        for item in audit["evaluation_tasks"]
        if item["subject"] == module.PRIMARY_SUBJECT
    )
    record["metrics_sha256"] = module._content_sha256(raw)
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(audit)

    with pytest.raises(ValueError, match="metric summary drifted"):
        _run(module, world)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("complete",), False, "complete mismatch"),
        (("complete",), 1, "complete mismatch"),
        (("planned_run_count",), 11, "planned_run_count mismatch"),
        (("runs", 0, "sample_count"), 1336, "sample_count mismatch"),
        (
            ("runs", 0, "metrics", "resilient_v2x/unsupported_sample_count"),
            1,
            "unsupported_sample_count mismatch",
        ),
        (
            ("runs", 0, "metrics", "resilient_v2x/car_bev_ap_r40_0.70"),
            float("nan"),
            "finite and within",
        ),
    ],
)
def test_rejects_invalid_live_metric_contract(
    path: tuple[object, ...], value: object, message: str
) -> None:
    module = _load_module()
    world = _world(module)
    first_result = world.leaderboard["results"][0]
    evaluation_id = first_result["evaluation_task_id"]
    raw = world.evaluation_tasks[evaluation_id].artifacts[module.METRICS_ARTIFACT].value
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    audit = copy.deepcopy(world.audit)
    audit["evaluation_tasks"][0]["metrics_sha256"] = module._content_sha256(raw)
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(audit)

    with pytest.raises(ValueError, match=message):
        _run(module, world)


def test_evaluation_task_must_still_be_completed_and_match_its_id() -> None:
    module = _load_module()
    world = _world(module)
    first_result = world.leaderboard["results"][0]
    evaluation_id = first_result["evaluation_task_id"]
    world.evaluation_tasks[evaluation_id].status = "failed"

    with pytest.raises(RuntimeError, match="is not completed"):
        _run(module, world)

    world = _world(module)
    evaluation_id = world.leaderboard["results"][0]["evaluation_task_id"]
    world.evaluation_tasks[evaluation_id].id = "9" * 32
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _run(module, world)

    world = _world(module)
    evaluation_id = world.leaderboard["results"][0]["evaluation_task_id"]
    world.evaluation_tasks[evaluation_id].parent = "8" * 32
    with pytest.raises(RuntimeError, match="parent mismatch"):
        _run(module, world)


@pytest.mark.parametrize("mutation", ["script", "parameter", "input_model"])
def test_live_evaluation_provenance_is_rechecked(mutation: str) -> None:
    module = _load_module()
    world = _world(module)
    evaluation_id = world.leaderboard["results"][0]["evaluation_task_id"]
    task = world.evaluation_tasks[evaluation_id]
    if mutation == "script":
        task.data.script.diff += "# drift\n"
        message = "script SHA-256 mismatch"
    elif mutation == "parameter":
        task.parameters["Args/gpus"] = 1
        message = "parameter .* mismatch"
    else:
        task.models["input"] = [SimpleNamespace(id="9" * 32)]
        message = "input model mismatch"

    with pytest.raises(RuntimeError, match=message):
        _run(module, world)


def test_final_snapshot_recheck_rejects_dependency_status_change() -> None:
    module = _load_module()
    world = _world(module)
    world.leaderboard_task.reload_statuses = ["completed", "failed"]

    with pytest.raises(RuntimeError, match="changed status"):
        _run(module, world)


def test_final_snapshot_freezes_mutable_artifact_content() -> None:
    module = _load_module()
    world = _world(module)

    class _MutatingArtifact(_Artifact):
        def __init__(self, value: dict[str, object]) -> None:
            super().__init__(value)
            self.calls = 0

        def get(self, *, force_download: bool = False) -> object:
            self.force_download_calls.append(force_download)
            assert force_download is True
            self.calls += 1
            if self.calls == 2:
                self.value["training_seed_evidence"] = "forged-after-initial"
                self.value = module._sealed(self.value)
            return self.value

    world.audit_task.artifacts[module.AUDIT_ARTIFACT] = _MutatingArtifact(world.audit)

    with pytest.raises(RuntimeError, match="audit artifact changed"):
        _run(module, world)


def test_publish_is_idempotent_but_rejects_existing_artifact_drift() -> None:
    module = _load_module()
    world = _world(module)
    first = _run(module, world)

    second = _run(module, world)

    assert second == first
    assert len(world.output.uploads) == 1
    world.output.artifacts[module.SELECTION_ARTIFACT].value = {"drift": True}
    with pytest.raises(RuntimeError, match="selection artifact drifted"):
        _run(module, world)


def test_publish_failure_and_missing_current_task_are_terminal() -> None:
    module = _load_module()
    world = _world(module)
    world.output = _OutputTask(upload_ok=False, parent=module.DEFAULT_AUDIT_TASK_ID)

    with pytest.raises(RuntimeError, match="failed to publish"):
        _run(module, world)

    world = _world(module)
    _TaskClass.current = None
    with pytest.raises(RuntimeError, match="requires a current ClearML task"):
        module.run(
            world.args,
            task_class=_TaskClass,
            output_task=None,
            monotonic_clock=lambda: 0.0,
            sleeper=lambda _: None,
        )


def test_flush_false_is_terminal() -> None:
    module = _load_module()
    world = _world(module)
    world.output = _OutputTask(flush_ok=False, parent=module.DEFAULT_AUDIT_TASK_ID)

    with pytest.raises(RuntimeError, match="failed to flush"):
        _run(module, world)


def test_selector_task_cannot_alias_a_dependency() -> None:
    module = _load_module()
    world = _world(module)
    world.output.id = module.DEFAULT_AUDIT_TASK_ID

    with pytest.raises(RuntimeError, match="must be a separate task"):
        _run(module, world)


@pytest.mark.parametrize("kind", ["controller", "training", "evaluation"])
def test_selector_task_cannot_alias_any_formal_task(kind: str) -> None:
    module = _load_module()
    world = _world(module)
    first = world.leaderboard["results"][0]
    aliases = {
        "controller": module.TRAINING_CONTROLLER_TASK_ID,
        "training": first["training_task_id"],
        "evaluation": first["evaluation_task_id"],
    }
    world.output.id = aliases[kind]
    world.output.server_id = aliases[kind]

    with pytest.raises(RuntimeError, match="separate task|aliases a formal dependency"):
        _run(module, world)


def test_artifact_reader_accepts_a_json_file(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"answer": 42}), encoding="utf-8")
    task = _Task("1" * 32, artifacts={"data": _Artifact(str(path))})
    module._reload(task, context="fixture")

    assert module._artifact_mapping(task, "data", context="fixture") == {"answer": 42}


@pytest.mark.parametrize(
    "snapshot",
    [None, False, True, 0, 1, 1.5, "snapshot", b"snapshot", bytearray(b"x")],
)
def test_raw_reload_rejects_non_snapshots_and_restores_skip_flag(
    snapshot: object,
) -> None:
    module = _load_module()
    task = _Task("1" * 32)

    def raw_reload() -> object:
        assert task._reload_skip_flag is False
        return snapshot

    task._reload = raw_reload  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="returned no snapshot"):
        module._reload(task, context="fixture")
    assert task._reload_skip_flag is True
    assert task.public_reload_calls == 0


def test_raw_reload_rejects_offline_error_and_wrong_server_identity() -> None:
    module = _load_module()
    task = _Task("1" * 32)
    task._offline_mode = True
    with pytest.raises(RuntimeError, match="offline reload"):
        module._reload(task, context="fixture")

    task._offline_mode = False
    task.raw_reload_error = OSError("network down")
    with pytest.raises(RuntimeError, match="server reload failed"):
        module._reload(task, context="fixture")
    assert task._reload_skip_flag is True

    task.raw_reload_error = None
    task.server_id = "2" * 32
    with pytest.raises(RuntimeError, match="snapshot identity mismatch"):
        module._reload(task, context="fixture")
    assert task._reload_skip_flag is True


def test_server_failed_status_overrides_local_completed_cache() -> None:
    module = _load_module()
    world = _world(module)
    world.audit_task._data.status = "completed"
    world.audit_task.server_status = "failed"

    with pytest.raises(RuntimeError, match="ended as 'failed'"):
        _run(module, world)
    assert world.output.uploads == []
    assert world.audit_task.public_reload_calls == 0


def test_local_metrics_cache_cannot_hide_empty_server_inventory() -> None:
    module = _load_module()
    world = _world(module)
    evaluation_id = str(world.leaderboard["results"][0]["evaluation_task_id"])
    task = world.evaluation_tasks[evaluation_id]
    assert module.METRICS_ARTIFACT in task.artifacts
    task.server_artifacts = {}

    with pytest.raises(RuntimeError, match="lacks artifact"):
        _run(module, world)
    assert world.output.uploads == []


def test_local_output_cache_cannot_substitute_for_server_readback() -> None:
    module = _load_module()
    world = _world(module)
    world.output.artifacts = {
        module.SELECTION_ARTIFACT: _Artifact({"local-only": True})
    }
    world.output.server_artifacts = {}

    with pytest.raises(RuntimeError, match="absent after publication"):
        _run(module, world)
    assert len(world.output.uploads) == 1


def test_extra_server_output_artifact_is_terminal_and_zero_write() -> None:
    module = _load_module()
    world = _world(module)
    world.output.server_artifacts["unexpected"] = _Artifact({"x": 1})

    with pytest.raises(RuntimeError, match="artifact inventory drifted"):
        _run(module, world)
    assert world.output.uploads == []


@pytest.mark.parametrize("flush_result", [0, 1, "", "ok", [], {}])
def test_non_boolean_flush_results_are_terminal(flush_result: object) -> None:
    module = _load_module()
    world = _world(module)
    world.output.flush_ok = flush_result  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="failed to flush"):
        _run(module, world)


def test_integer_one_upload_result_is_not_accepted_as_true() -> None:
    module = _load_module()
    world = _world(module)
    world.output.upload_ok = 1  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="failed to publish"):
        _run(module, world)


def test_server_readback_rejects_post_flush_payload_drift() -> None:
    module = _load_module()
    world = _world(module)

    class _DriftingOutput(_OutputTask):
        def flush(self, *, wait_for_uploads: bool) -> bool | None:
            result = super().flush(wait_for_uploads=wait_for_uploads)
            self.server_artifacts[module.SELECTION_ARTIFACT].value = {"drift": True}
            return result

    world.output = _DriftingOutput(parent=module.DEFAULT_AUDIT_TASK_ID)
    with pytest.raises(
        RuntimeError, match="published formal candidate selection drifted"
    ):
        _run(module, world)


def test_committed_upload_exception_can_resume_without_duplicate_upload() -> None:
    module = _load_module()
    world = _world(module)

    class _CommitThenRaiseOutput(_OutputTask):
        def upload_artifact(
            self,
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            result = super().upload_artifact(
                name,
                artifact_object=artifact_object,
                wait_on_upload=wait_on_upload,
            )
            if len(self.uploads) == 1:
                raise RuntimeError("transport failed after commit")
            return result

    world.output = _CommitThenRaiseOutput(parent=module.DEFAULT_AUDIT_TASK_ID)
    with pytest.raises(RuntimeError, match="transport failed after commit"):
        _run(module, world)
    assert len(world.output.uploads) == 1
    resumed = _run(module, world)
    assert resumed["selected_candidate"] == module.PRIMARY_SUBJECT
    assert len(world.output.uploads) == 1


def test_output_parent_drift_after_flush_is_terminal() -> None:
    module = _load_module()
    world = _world(module)

    class _ParentDriftingOutput(_OutputTask):
        def flush(self, *, wait_for_uploads: bool) -> bool | None:
            result = super().flush(wait_for_uploads=wait_for_uploads)
            self.server_parent = "9" * 32
            return result

    world.output = _ParentDriftingOutput(parent=module.DEFAULT_AUDIT_TASK_ID)
    with pytest.raises(RuntimeError, match="publication readback parent drifted"):
        _run(module, world)


def test_happy_path_never_uses_public_reload_on_any_task() -> None:
    module = _load_module()
    world = _world(module)
    _run(module, world)

    assert world.output.public_reload_calls == 0
    assert all(task.public_reload_calls == 0 for task in world.tasks.values())
    assert world.output._reload_skip_flag is True
    assert all(task._reload_skip_flag is True for task in world.tasks.values())


def test_artifact_reads_force_fresh_download_instead_of_stale_cache() -> None:
    module = _load_module()

    class _CacheSensitiveArtifact:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def get(self, *, force_download: bool = False) -> object:
            self.calls.append(force_download)
            return {"fresh": True} if force_download else {"stale": True}

    artifact = _CacheSensitiveArtifact()
    task = _Task("1" * 32, artifacts={"data": artifact})  # type: ignore[dict-item]
    module._reload(task, context="fixture")

    assert module._artifact_mapping(task, "data", context="fixture") == {"fresh": True}
    assert artifact.calls == [True]


def test_artifact_reader_never_falls_back_when_force_download_is_unsupported() -> None:
    module = _load_module()

    class _StaleOnlyArtifact:
        def __init__(self) -> None:
            self.calls = 0

        def get(self) -> object:
            self.calls += 1
            return {"stale": True}

    artifact = _StaleOnlyArtifact()
    task = _Task("1" * 32, artifacts={"data": artifact})  # type: ignore[dict-item]
    module._reload(task, context="fixture")

    with pytest.raises(RuntimeError, match="cannot be freshly downloaded"):
        module._artifact_mapping(task, "data", context="fixture")
    assert artifact.calls == 0


@pytest.mark.parametrize("status", ["failed", "completed", "published", "stopped"])
def test_output_server_status_must_remain_writable_in_progress(status: str) -> None:
    module = _load_module()
    world = _world(module)
    world.output.status = status

    with pytest.raises(RuntimeError, match="not writable in_progress"):
        _run(module, world)
    assert world.output.uploads == []


def test_raw_reload_rejects_silently_discarded_snapshot_install() -> None:
    module = _load_module()

    class _DiscardingTask:
        def __init__(self) -> None:
            object.__setattr__(self, "id", "1" * 32)
            object.__setattr__(self, "_offline_mode", False)
            object.__setattr__(self, "_reload_skip_flag", True)
            local = SimpleNamespace(id="1" * 32)
            object.__setattr__(self, "_data", local)
            object.__setattr__(self, "snapshot", SimpleNamespace(id="1" * 32))

        @property
        def data(self) -> object:
            return self._data

        def __setattr__(self, name: str, value: object) -> None:
            if name == "_data":
                return
            object.__setattr__(self, name, value)

        def _reload(self) -> object:
            assert self._reload_skip_flag is False
            return self.snapshot

    task = _DiscardingTask()
    with pytest.raises(RuntimeError, match="not installed exactly"):
        module._reload(task, context="fixture")
    assert task._reload_skip_flag is True


def test_raw_reload_rejects_data_adapter_instead_of_exact_snapshot() -> None:
    module = _load_module()

    class _AdaptingTask(_Task):
        @property
        def data(self) -> object:
            return SimpleNamespace(**vars(self._data))

    task = _AdaptingTask("1" * 32)
    with pytest.raises(RuntimeError, match="not installed exactly"):
        module._reload(task, context="fixture")
    assert task._reload_skip_flag is True


@pytest.mark.parametrize("target", ["leaderboard_task", "audit_task"])
def test_dependency_extra_server_artifact_is_terminal(target: str) -> None:
    module = _load_module()
    world = _world(module)
    getattr(world, target).server_artifacts["attacker_extra"] = _Artifact({"x": 1})

    with pytest.raises(RuntimeError, match="artifact inventory drifted"):
        _run(module, world)
    assert world.output.uploads == []


@pytest.mark.parametrize("value", [True, False, 1, 2, 1.0, "3"])
def test_leaderboard_schema_version_requires_exact_integer(value: object) -> None:
    module = _load_module()
    world = _world(module)
    changed = copy.deepcopy(world.leaderboard)
    changed["schema_version"] = value
    changed = module._sealed(changed)
    world.leaderboard_task.artifacts[module.LEADERBOARD_ARTIFACT].value = changed
    audit = copy.deepcopy(world.audit)
    audit["leaderboard_seal_sha256"] = changed["seal_sha256"]
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(audit)

    with pytest.raises(ValueError, match="schema_version mismatch"):
        _run(module, world)
    assert world.output.uploads == []


SOURCE_BINDING_FIELDS = (
    "training_provenance_task_id",
    "training_provenance_seal_sha256",
    "source_revision_equivalence",
    "source_revision_equivalence_seal_sha256",
    "source_revision_subject_map",
    "source_revision_subject_map_seal_sha256",
    "evaluation_source_revision_tree_sha256",
    "evaluation_source_revision",
)


@pytest.mark.parametrize("field", SOURCE_BINDING_FIELDS)
def test_leaderboard_v3_requires_all_source_binding_fields(field: str) -> None:
    module = _load_module()
    world = _world(module)
    changed = copy.deepcopy(world.leaderboard)
    changed.pop(field)
    changed = module._sealed(changed)
    world.leaderboard_task.artifacts[module.LEADERBOARD_ARTIFACT].value = changed

    with pytest.raises(ValueError, match="keys mismatch"):
        _run(module, world)
    assert world.output.uploads == []


def test_leaderboard_v3_rejects_extra_source_binding_field() -> None:
    module = _load_module()
    world = _world(module)
    changed = copy.deepcopy(world.leaderboard)
    changed["source_revision_unreviewed"] = "attacker"
    changed = module._sealed(changed)
    world.leaderboard_task.artifacts[module.LEADERBOARD_ARTIFACT].value = changed

    with pytest.raises(ValueError, match="keys mismatch"):
        _run(module, world)
    assert world.output.uploads == []


@pytest.mark.parametrize(
    "mutation",
    [
        "provenance_task",
        "provenance_seal",
        "equivalence",
        "equivalence_root_seal",
        "subject_map",
        "subject_map_root_seal",
        "evaluation_tree",
        "evaluation_revision",
        "result_tree",
    ],
)
def test_leaderboard_v3_rejects_resealed_source_binding_drift(
    mutation: str,
) -> None:
    module = _load_module()
    world = _world(module)
    changed = copy.deepcopy(world.leaderboard)
    if mutation == "provenance_task":
        changed["training_provenance_task_id"] = "9" * 32
    elif mutation == "provenance_seal":
        changed["training_provenance_seal_sha256"] = "not-a-sha"
    elif mutation == "equivalence":
        nested = changed["source_revision_equivalence"]
        nested["contract"] = "attacker"
        changed["source_revision_equivalence"] = module._sealed(nested)
        changed["source_revision_equivalence_seal_sha256"] = changed[
            "source_revision_equivalence"
        ]["seal_sha256"]
    elif mutation == "equivalence_root_seal":
        changed["source_revision_equivalence_seal_sha256"] = "9" * 64
    elif mutation == "subject_map":
        nested = changed["source_revision_subject_map"]
        nested["source_revision_by_subject"][module.PRIMARY_SUBJECT] = (
            module.SOURCE_TREE_SHA256
        )
        changed["source_revision_subject_map"] = module._sealed(nested)
        changed["source_revision_subject_map_seal_sha256"] = changed[
            "source_revision_subject_map"
        ]["seal_sha256"]
    elif mutation == "subject_map_root_seal":
        changed["source_revision_subject_map_seal_sha256"] = "9" * 64
    elif mutation == "evaluation_tree":
        changed["evaluation_source_revision_tree_sha256"] = (
            module.NEW_SOURCE_TREE_SHA256
        )
    elif mutation == "evaluation_revision":
        changed["evaluation_source_revision"] = (
            module.SOURCE_REVISION_CERTIFICATE_BY_TREE[module.NEW_SOURCE_TREE_SHA256]
        )
    else:
        changed["results"][-1]["source_revision_tree_sha256"] = (
            module.SOURCE_TREE_SHA256
        )
    changed = module._sealed(changed)
    world.leaderboard_task.artifacts[module.LEADERBOARD_ARTIFACT].value = changed

    with pytest.raises(ValueError):
        _run(module, world)
    assert world.output.uploads == []


def test_audit_nested_integer_cannot_be_resealed_as_boolean() -> None:
    module = _load_module()
    world = _world(module)
    audit = copy.deepcopy(world.audit)
    audit["training_tasks"][0]["gpus"] = True
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(audit)

    with pytest.raises(ValueError, match="training support_residual gpus mismatch"):
        _run(module, world)
    assert world.output.uploads == []


def test_metrics_integer_contract_cannot_be_resealed_as_boolean() -> None:
    module = _load_module()
    world = _world(module)
    evaluation_id = str(world.leaderboard["results"][0]["evaluation_task_id"])
    artifact = world.evaluation_tasks[evaluation_id].artifacts[module.METRICS_ARTIFACT]
    changed = copy.deepcopy(artifact.value)
    changed["planned_run_count"] = True
    artifact.value = changed
    audit = copy.deepcopy(world.audit)
    audit["evaluation_tasks"][0]["metrics_sha256"] = module._content_sha256(changed)
    world.audit_task.artifacts[module.AUDIT_ARTIFACT].value = module._sealed(audit)

    with pytest.raises(ValueError, match="planned_run_count mismatch"):
        _run(module, world)
    assert world.output.uploads == []


def test_two_final_dependency_passes_catch_late_cross_task_mutation() -> None:
    module = _load_module()
    world = _world(module)
    first_id = str(world.leaderboard["results"][0]["evaluation_task_id"])
    last_id = str(world.leaderboard["results"][-1]["evaluation_task_id"])
    first_artifact = world.evaluation_tasks[first_id].artifacts[module.METRICS_ARTIFACT]

    class _MutatingLastArtifact(_Artifact):
        def __init__(self, value: object) -> None:
            super().__init__(value)
            self.mutated = False

        def get(self, *, force_download: bool = False) -> object:
            value = super().get(force_download=force_download)
            if world.output.uploads and not self.mutated:
                first_artifact.value["baseline"] = "forged-after-first-read"
                self.mutated = True
            return value

    last = world.evaluation_tasks[last_id].artifacts[module.METRICS_ARTIFACT]
    world.evaluation_tasks[last_id].artifacts[module.METRICS_ARTIFACT] = (
        _MutatingLastArtifact(last.value)
    )

    with pytest.raises(RuntimeError, match="metrics changed"):
        _run(module, world)
    assert len(world.output.uploads) == 1


def test_final_output_readback_follows_both_dependency_passes() -> None:
    module = _load_module()
    world = _world(module)
    last_id = str(world.leaderboard["results"][-1]["evaluation_task_id"])

    class _OutputMutatingLastArtifact(_Artifact):
        def __init__(self, value: object) -> None:
            super().__init__(value)
            self.mutated = False

        def get(self, *, force_download: bool = False) -> object:
            value = super().get(force_download=force_download)
            selection = world.output.server_artifacts.get(module.SELECTION_ARTIFACT)
            if (
                selection is not None
                and selection.force_download_calls
                and not self.mutated
            ):
                selection.value = {"drift": True}
                self.mutated = True
            return value

    last = world.evaluation_tasks[last_id].artifacts[module.METRICS_ARTIFACT]
    world.evaluation_tasks[last_id].artifacts[module.METRICS_ARTIFACT] = (
        _OutputMutatingLastArtifact(last.value)
    )

    with pytest.raises(
        RuntimeError, match="published formal candidate selection drifted"
    ):
        _run(module, world)
    assert len(world.output.uploads) == 1


def test_output_script_must_equal_running_selector_bytes() -> None:
    module = _load_module()
    world = _world(module)
    world.output.server_script.diff += "# attacker drift\n"

    with pytest.raises(RuntimeError, match="script drifted|script bytes drifted"):
        _run(module, world)
    assert world.output.uploads == []
