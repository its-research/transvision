from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_formal_multiseed_plan.py"
VALIDATOR_PATH = ROOT / "tools/resilient_v2x/formal_multiseed_plan.py"
SELECTOR_PATH = ROOT / "tools/resilient_v2x/clearml_formal_candidate_selector.py"
AUDIT_PRODUCER_PATH = ROOT / "tools/resilient_v2x/clearml_formal_comparability_audit.py"
EXECUTOR_PATH = ROOT / "tools/resilient_v2x/clearml_formal_multiseed_executor.py"
PROVENANCE_PRODUCER_PATH = (
    ROOT / "tools/resilient_v2x/clearml_formal_training_provenance.py"
)
SOURCE_D_PRODUCER_PATH = (
    ROOT / "tools/resilient_v2x/clearml_formal_source_d_evidence.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(value: object) -> str:
    return _digest_text(_canonical(value))


def _seal(value: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(value)
    value.pop("seal_sha256", None)
    value["seal_sha256"] = _digest(value)
    return value


def _source_revision_binding(module) -> dict[str, object]:
    producer = _load(
        "formal_training_provenance_for_multiseed_planner_fixture",
        PROVENANCE_PRODUCER_PATH,
    )
    equivalence = producer._source_revision_equivalence()
    subject_map = producer._source_revision_subject_map()
    evaluation_tree = producer.SOURCE_TREE_SHA256
    assert equivalence["seal_sha256"] == (
        module.SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
    )
    assert subject_map["seal_sha256"] == (
        module.SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
    )
    return {
        "source_revision_equivalence": equivalence,
        "source_revision_equivalence_seal_sha256": equivalence["seal_sha256"],
        "source_revision_subject_map": subject_map,
        "source_revision_subject_map_seal_sha256": subject_map["seal_sha256"],
        "evaluation_source_revision_tree_sha256": evaluation_tree,
        "evaluation_source_revision": equivalence["source_revisions"][evaluation_tree],
    }


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(value)
    value.pop("artifact_sha256", None)
    value["artifact_sha256"] = _digest(value)
    return value


class _RawArtifact:
    def __init__(self, key: str, value: object) -> None:
        self.key = key
        self.value = value


class _Artifact:
    def __init__(self, value: object) -> None:
        self.from_raw = isinstance(value, _RawArtifact)
        self.value = value.value if self.from_raw else value
        self.force_downloads: list[object] = []

    def get(self, *, force_download: object = None) -> object:
        self.force_downloads.append(force_download)
        if self.from_raw and force_download is not True:
            raise AssertionError("server artifact reads must force a fresh download")
        return copy.deepcopy(self.value)


def _script(entry_point: str, source: str) -> SimpleNamespace:
    return SimpleNamespace(
        repository="",
        working_dir=".",
        entry_point=entry_point,
        diff=source,
    )


class _ServerSnapshot(SimpleNamespace):
    @property
    def execution(self) -> SimpleNamespace:
        override = getattr(self, "raw_artifact_override", None)
        if override is not None:
            return SimpleNamespace(artifacts=override)
        return SimpleNamespace(
            artifacts=[
                _RawArtifact(name, artifact.value)
                for name, artifact in self.artifacts.items()
            ]
        )


def _server(
    *,
    task_id: str,
    parent: str,
    status: str,
    script: SimpleNamespace,
    artifacts: dict[str, _Artifact] | None = None,
    parameters: dict[str, object] | None = None,
) -> _ServerSnapshot:
    return _ServerSnapshot(
        id=task_id,
        parent=parent,
        status=status,
        script=script,
        artifacts=artifacts or {},
        parameters=parameters or {},
    )


class _Task:
    def __init__(self, server: SimpleNamespace) -> None:
        self.server = copy.deepcopy(server)
        self._data = copy.deepcopy(server)
        self._reload_skip_flag = True
        self._offline_mode = False
        self.reload_result: object = ...
        self.reload_exception: Exception | None = None
        self.statuses: list[str] = []
        self.raw_reload_calls = 0
        self.public_reload_calls = 0
        self.uploads: list[str] = []
        self.flushes: list[str] = []
        self.upload_result: object = True
        self.flush_result: object = None
        self.upload_raise_after_commit = False
        self.omit_upload: str | None = None
        self.drift_upload: str | None = None
        self.after_upload: object = None
        self.local_artifacts: dict[str, _Artifact] = {}

    @property
    def id(self) -> object:
        return self._data.id

    @property
    def parent(self) -> object:
        return self._data.parent

    @property
    def status(self) -> object:
        return self._data.status

    @property
    def data(self) -> SimpleNamespace:
        return self._data

    @property
    def artifacts(self) -> dict[str, _Artifact]:
        return {**self._data.artifacts, **self.local_artifacts}

    def get_parameters(self) -> dict[str, object]:
        return copy.deepcopy(self._data.parameters)

    def reload(self) -> None:
        self.public_reload_calls += 1
        raise AssertionError("public reload must never be called")

    def _reload(self) -> object:
        assert self._reload_skip_flag is False
        self.raw_reload_calls += 1
        if self.reload_exception is not None:
            raise self.reload_exception
        if self.statuses:
            self.server.status = self.statuses.pop(0)
        if self.reload_result is not ...:
            return self.reload_result
        return copy.deepcopy(self.server)

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> object:
        assert wait_on_upload is True
        self.uploads.append(name)
        if name != self.omit_upload:
            value = copy.deepcopy(artifact_object)
            if name == self.drift_upload:
                value["attacker"] = True
            self.server.artifacts[name] = _Artifact(value)
        callback = self.after_upload
        if callable(callback):
            callback(name)
        if self.upload_raise_after_commit:
            raise RuntimeError("ambiguous committed upload")
        return self.upload_result

    def flush(self, *, wait_for_uploads: bool) -> object:
        assert wait_for_uploads is True
        self.flushes.append(self.uploads[-1])
        return self.flush_result

    def set_parent(self, parent: str) -> object:
        self.server.parent = parent
        self._data.parent = parent
        return True


class _TaskApi:
    def __init__(
        self,
        controller: _Task,
        provenance: _Task,
        watcher: _Task,
        selector: _Task,
        audit: _Task,
        leaderboard: _Task,
        output: _Task,
    ) -> None:
        self.controller = controller
        self.source_d = controller
        self.provenance = provenance
        self.watcher = watcher
        self.selector = selector
        self.audit = audit
        self.leaderboard = leaderboard
        self.output = output
        self.requested: list[str] = []

    def get_task(self, *, task_id: str) -> _Task:
        self.requested.append(task_id)
        if task_id == self.controller.id:
            return self.controller
        if task_id == self.provenance.id:
            return self.provenance
        if task_id == self.watcher.id:
            return self.watcher
        if task_id == self.selector.id:
            return self.selector
        if task_id == self.audit.id:
            return self.audit
        if task_id == self.leaderboard.id:
            return self.leaderboard
        raise AssertionError(task_id)

    def current_task(self) -> _Task:
        return self.output


def _args(**changes: object) -> SimpleNamespace:
    values = {"poll_seconds": 1.0, "timeout_hours": 1.0}
    values.update(changes)
    return SimpleNamespace(**values)


def _production_planner_parameters(module) -> dict[str, object]:
    args = module._parser().parse_args([])
    return {
        "Args/poll_seconds": str(args.poll_seconds),
        "Args/timeout_hours": str(args.timeout_hours),
        "Args/emit_standalone": (
            "" if args.emit_standalone is None else str(args.emit_standalone)
        ),
    }


def _selector_artifact(
    module,
    *,
    audit_seal: str,
    leaderboard_seal: str,
    progress_seal: str,
    provenance_seal: str,
) -> dict[str, object]:
    selector = _load("formal_selector_for_planner_fixture", SELECTOR_PATH)
    bases = {
        "resilient_v2x": 90.0,
        "support_residual": 89.0,
        "linear_no_distillation": 88.0,
        "no_distillation_peak_lr_3e4": 87.0,
        **{
            baseline: 50.0 + index
            for index, baseline in enumerate(selector.BASELINE_SUBJECTS)
        },
    }
    runs = {
        subject: {
            (delay, condition): {
                selector.LEADERSHIP_METRIC: (
                    base
                    - delay / 100.0
                    - {"Full": 0.0, "L-Fail": 2.0, "C-Fail": 1.0}[condition]
                )
            }
            for delay in selector.DELAYS_MS
            for condition in selector.CONDITIONS
        }
        for subject, base in bases.items()
    }
    audit_chain = {
        "training_provenance_task_id": module.TRAINING_PROVENANCE_TASK_ID,
        "training_progress_seal_sha256": progress_seal,
        "training_provenance_seal_sha256": provenance_seal,
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
    }
    audit_chain.update(_source_revision_binding(module))
    return selector.build_selection(
        audit_task_id=module.SELECTOR_AUDIT_TASK_ID,
        leaderboard_task_id=module.SELECTOR_LEADERBOARD_TASK_ID,
        audit_seal=audit_seal,
        leaderboard_seal=leaderboard_seal,
        audit_chain=audit_chain,
        runs_by_subject=runs,
    )


def _source_d_documents(module, source_c_text: str, source_d_text: str):
    source_c = _seal(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_c_snapshot",
            "complete": True,
            "source_c_task_id": module.SOURCE_C_TASK_ID,
            "source_c_task_status": "completed",
            "script": {
                "repository": "",
                "working_dir": ".",
                "entry_point": module.SOURCE_C_ENTRY_POINT,
                "sha256": module.SOURCE_C_SCRIPT_SHA256,
                "size_bytes": len(source_c_text.encode("utf-8")),
                "line_count": len(source_c_text.splitlines()),
            },
            "script_diff": source_c_text,
        }
    )
    source_d = _seal(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_d_script",
            "complete": True,
            "transformation_id": module.SOURCE_D_TRANSFORMATION_ID,
            "source_c_sha256": module.SOURCE_C_SCRIPT_SHA256,
            "source_d_sha256": module.SOURCE_D_SCRIPT_SHA256,
            "size_bytes": len(source_d_text.encode("utf-8")),
            "line_count": len(source_d_text.splitlines()),
            "script": source_d_text,
        }
    )
    ordinary_names = [
        f"replacement_{index}"
        for index in range(
            1,
            module.DECLARED_REPLACEMENT_COUNT
            - len(module.SOURCE_D_STAGING_ANCHOR_NAMES)
            + 1,
        )
    ]
    equivalence = _rehash(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_d_seed_diff",
            "transformation_id": module.SOURCE_D_TRANSFORMATION_ID,
            "source_c": {"sha256": module.SOURCE_C_SCRIPT_SHA256},
            "source_d": {"sha256": module.SOURCE_D_SCRIPT_SHA256},
            "seed_contract": {
                "default_training_seed": module.TRAINING_OVERLAY_PROTOCOL_SEED,
                "training_overlay_protocol_seed": (
                    module.TRAINING_OVERLAY_PROTOCOL_SEED
                ),
            },
            "diff": [
                {
                    "index": index,
                    "name": name,
                    "expected_count": 1,
                    "observed_count": 1,
                }
                for index, name in enumerate(
                    ordinary_names + list(module.SOURCE_D_STAGING_ANCHOR_NAMES),
                    start=1,
                )
            ],
            "equivalence": {
                "only_declared_anchor_replacements": True,
                "declared_replacement_count": module.DECLARED_REPLACEMENT_COUNT,
                "unchanged_segment_count": module.UNCHANGED_SEGMENT_COUNT,
                "source_c_replay_sha256": module.SOURCE_C_SCRIPT_SHA256,
                "source_d_replay_sha256": module.SOURCE_D_SCRIPT_SHA256,
                "source_d_compiles": True,
            },
        }
    )
    receipt = _seal(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_d_evidence_receipt",
            "complete": True,
            "publication_order": list(module.SOURCE_D_ARTIFACTS),
            "provenance": {
                "source_c_task_id": module.SOURCE_C_TASK_ID,
                "source_c_task_status": "completed",
                "source_c_task_parent": module.SOURCE_C_PARENT_TASK_ID,
                "source_c_entry_point": module.SOURCE_C_ENTRY_POINT,
                "source_c_sha256": module.SOURCE_C_SCRIPT_SHA256,
                "output_task_id": module.SOURCE_D_TASK_ID,
                "output_parent_task_id": module.SOURCE_C_TASK_ID,
                "builder_source_sha256": module.SOURCE_D_BUILDER_SHA256,
                "transformation_id": module.SOURCE_D_TRANSFORMATION_ID,
                "producer_entry_point": module.SOURCE_D_ENTRY_POINT,
                "producer_script_sha256": module.SOURCE_D_PRODUCER_SHA256,
            },
            "transformation": {
                "source_d_sha256": module.SOURCE_D_SCRIPT_SHA256,
                "equivalence_artifact_sha256": equivalence["artifact_sha256"],
                "declared_replacement_count": module.DECLARED_REPLACEMENT_COUNT,
                "unchanged_segment_count": module.UNCHANGED_SEGMENT_COUNT,
                "only_declared_anchor_replacements": True,
                "training_overlay_protocol_seed": (
                    module.TRAINING_OVERLAY_PROTOCOL_SEED
                ),
                "training_seed_cli": "--training-seed",
                "portable_runner_load_marker": module.PORTABLE_RUNNER_LOAD_MARKER,
                "portable_runner_load_marker_count": (
                    module.PORTABLE_RUNNER_LOAD_MARKER_COUNT
                ),
                "legacy_runner_load_target_anchor_count": 0,
            },
            "artifact_hashes": {
                module.SOURCE_C_SNAPSHOT_ARTIFACT: source_c["seal_sha256"],
                module.SOURCE_D_SCRIPT_ARTIFACT: source_d["seal_sha256"],
                module.SOURCE_D_EQUIVALENCE_ARTIFACT: equivalence["artifact_sha256"],
            },
        }
    )
    return source_c, source_d, equivalence, receipt


def _formal_leaderboard_artifact(
    module, *, provenance_seal: str
) -> dict[str, object]:
    source_binding = _source_revision_binding(module)
    results = []
    for index, subject in enumerate(module.AUDIT_SUBJECT_ORDER, start=1):
        results.append(
            {
                "index": index,
                "subject": subject,
                "kind": module._audit_subject_kind(subject),
                "training_task_id": f"{1_000 + index:032x}",
                "training_model_id": f"{2_000 + index:032x}",
                "training_checkpoint_sha256": _digest_text(f"checkpoint:{subject}"),
                "source_revision_tree_sha256": module._expected_source_revision(
                    subject
                )["tree_sha256"],
                "evaluation_task_id": f"{3_000 + index:032x}",
                "metrics": {
                    metric: {"fixture": float(index)}
                    for metric in module.AUDIT_METRIC_KEYS
                },
            }
        )
    return _seal(
        {
            "schema_version": 3,
            "leaderboard_type": "resilient_v2x_formal_1337_leaderboard",
            "protocol_id": module.AUDIT_PROTOCOL_ID,
            "sample_count": module.AUDIT_SAMPLE_COUNT,
            "ground_truth_count": module.AUDIT_GROUND_TRUTH_COUNT,
            "unsupported_sample_count": module.AUDIT_UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(module.AUDIT_DELAYS_MS),
            "conditions": list(module.AUDIT_CONDITIONS),
            "run_count_per_subject": len(module.AUDIT_DELAYS_MS)
            * len(module.AUDIT_CONDITIONS),
            "training_controller_task_id": (
                module.SELECTOR_TRAINING_CONTROLLER_TASK_ID
            ),
            "watcher_task_id": module.SELECTOR_WATCHER_TASK_ID,
            "training_manifest_seal_sha256": "1" * 64,
            "evaluation_plan_seal_sha256": "3" * 64,
            "training_provenance_task_id": module.TRAINING_PROVENANCE_TASK_ID,
            "training_provenance_seal_sha256": provenance_seal,
            **source_binding,
            "subject_order": list(module.AUDIT_SUBJECT_ORDER),
            "subject_count": len(module.AUDIT_SUBJECT_ORDER),
            "baseline_subjects": list(module.AUDIT_BASELINE_SUBJECTS),
            "baseline_count": len(module.AUDIT_BASELINE_SUBJECTS),
            "metric_keys": list(module.AUDIT_METRIC_KEYS),
            "results": results,
            "leadership": {},
        }
    )


def _formal_audit_artifact(
    module,
    *,
    leaderboard_seal: str,
    progress_seal: str,
    provenance_seal: str,
) -> dict[str, object]:
    source_binding = _source_revision_binding(module)
    training_records: list[dict[str, object]] = []
    evaluation_records: list[dict[str, object]] = []
    queues = tuple(sorted(module.AUDIT_SUPPORTED_QUEUES))
    for index, subject in enumerate(module.AUDIT_SUBJECT_ORDER, start=1):
        source = module._expected_source_revision(subject)
        training_task_id = f"{1_000 + index:032x}"
        model_id = f"{2_000 + index:032x}"
        evaluation_task_id = f"{3_000 + index:032x}"
        checkpoint_sha256 = _digest_text(f"checkpoint:{subject}")
        training_records.append(
            {
                "index": index,
                "subject": subject,
                "training_task_id": training_task_id,
                "model_id": model_id,
                "checkpoint_sha256": checkpoint_sha256,
                "final_checkpoint_contract_content_sha256": _digest_text(
                    f"final-contract:{subject}"
                ),
                "parent_controller_task_id": (
                    module._expected_training_parent_task_id(subject)
                ),
                "script_sha256": module._expected_training_script_sha256(subject),
                "run_contract_sha256": _digest_text(f"run-contract:{subject}"),
                "run_contract_seed_fields": ["seed"],
                "training_seed": module.AUDIT_TRAINING_SEED,
                "source_revision_tree_sha256": source["tree_sha256"],
                "source_dataset_id": source["dataset_id"],
                "source_archive_name": source["archive_name"],
                "source_archive_bytes": source["archive_bytes"],
                "source_archive_sha256": source["archive_sha256"],
                "training_dataset_id": module.AUDIT_TRAINING_DATASET_ID,
                "gpus": 4,
                "precision": "FP32",
                "max_epochs": 50,
                "val_interval": 10,
                "common_teacher_initialization_audit_sha256": _digest_text(
                    f"teacher-audit:{subject}"
                ),
                "raw_authority_sha256": _digest_text(f"authority:{subject}"),
                "execution_queue_id": f"{5_000 + index:032x}",
                "last_worker": f"training-worker-{index}",
                "final_model_verification_level": ("clearml_metadata_contract_only"),
                "checkpoint_bytes_sha256_recomputed": False,
                "checkpoint_bytes_verifier": module.AUDIT_MODEL_BYTES_VERIFIER,
            }
        )
        queue = queues[(index - 1) % len(queues)]
        evaluation_records.append(
            {
                "index": index,
                "subject": subject,
                "evaluation_task_id": evaluation_task_id,
                "training_task_id": training_task_id,
                "model_id": model_id,
                "checkpoint_sha256": checkpoint_sha256,
                "source_revision_tree_sha256": module.AUDIT_SOURCE_TREE_SHA256,
                "source_dataset_id": module.AUDIT_SOURCE_DATASET_ID,
                "source_archive_name": module.AUDIT_SOURCE_ARCHIVE_NAME,
                "source_archive_bytes": module.AUDIT_SOURCE_ARCHIVE_BYTES,
                "source_archive_sha256": module.AUDIT_SOURCE_ARCHIVE_SHA256,
                "script_sha256": module.AUDIT_EVALUATION_SCRIPT_SHA256,
                "planned_queue": queue,
                "execution_queue_id": f"{4_000 + index:032x}",
                "execution_queue_name": queue,
                "last_worker": f"worker-{index}",
                "metrics_sha256": _digest_text(f"metrics:{subject}"),
                "run_count": len(module.AUDIT_DELAYS_MS) * len(module.AUDIT_CONDITIONS),
                "sample_count_per_run": module.AUDIT_SAMPLE_COUNT,
                "ground_truth_count_per_run": module.AUDIT_GROUND_TRUTH_COUNT,
                "unsupported_sample_count_per_run": (
                    module.AUDIT_UNSUPPORTED_SAMPLE_COUNT
                ),
                "metric_keys": list(module.AUDIT_METRIC_KEYS),
            }
        )
    return _seal(
        {
            "schema_version": 3,
            "document_type": "resilient_v2x_formal_1337_comparability_audit",
            "passed": True,
            "audit_scope": "protocol_and_clearml_metadata_comparability",
            "audit_task_id": module.SELECTOR_AUDIT_TASK_ID,
            "training_controller_task_id": (
                module.SELECTOR_TRAINING_CONTROLLER_TASK_ID
            ),
            "training_provenance_task_id": module.TRAINING_PROVENANCE_TASK_ID,
            "watcher_task_id": module.SELECTOR_WATCHER_TASK_ID,
            "leaderboard_task_id": module.SELECTOR_LEADERBOARD_TASK_ID,
            "protocol_id": module.AUDIT_PROTOCOL_ID,
            **source_binding,
            "source_revision_counts": {
                module.AUDIT_SOURCE_TREE_SHA256: 21,
                module.AUDIT_NEW_SOURCE_TREE_SHA256: 5,
            },
            "training_dataset_id": module.AUDIT_TRAINING_DATASET_ID,
            "training_seed": module.AUDIT_TRAINING_SEED,
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
            "checkpoint_policy": module.AUDIT_CHECKPOINT_POLICY,
            "sample_count": module.AUDIT_SAMPLE_COUNT,
            "ground_truth_count": module.AUDIT_GROUND_TRUTH_COUNT,
            "unsupported_sample_count": module.AUDIT_UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(module.AUDIT_DELAYS_MS),
            "conditions": list(module.AUDIT_CONDITIONS),
            "run_count_per_subject": len(module.AUDIT_DELAYS_MS)
            * len(module.AUDIT_CONDITIONS),
            "subject_order": list(module.AUDIT_SUBJECT_ORDER),
            "subject_count": len(module.AUDIT_SUBJECT_ORDER),
            "total_evaluation_run_count": len(module.AUDIT_SUBJECT_ORDER)
            * len(module.AUDIT_DELAYS_MS)
            * len(module.AUDIT_CONDITIONS),
            "metric_keys": list(module.AUDIT_METRIC_KEYS),
            "final_model_verification_level": "clearml_metadata_contract_only",
            "checkpoint_bytes_sha256_recomputed": False,
            "checkpoint_bytes_verifier": module.AUDIT_MODEL_BYTES_VERIFIER,
            "checkpoint_bytes_verification_dependency": (
                "separate_collect_clearml_formal_models_byte_audit"
            ),
            "training_manifest_seal_sha256": "1" * 64,
            "training_progress_seal_sha256": progress_seal,
            "training_provenance_seal_sha256": provenance_seal,
            "training_summary_seal_sha256": "2" * 64,
            "evaluation_plan_seal_sha256": "3" * 64,
            "leaderboard_seal_sha256": leaderboard_seal,
            "training_tasks": training_records,
            "evaluation_tasks": evaluation_records,
        }
    )


def _schema4_training_documents(
    module, monkeypatch
) -> tuple[str, dict[str, object], dict[str, object]]:
    source_controller_task_id = "e" * 32
    source_template_identity = {"script_sha256": "1" * 64}
    target_template_identity = {"script_sha256": "2" * 64}
    transition = _seal(
        {
            "source_controller_task_id": source_controller_task_id,
            "source_template_identity": source_template_identity,
            "target_template_identity": target_template_identity,
        }
    )
    template_equivalence = _seal(
        {"expected_identity": source_template_identity, "passed": True}
    )
    monkeypatch.setattr(
        module,
        "SOURCE_REVISION_TRANSITION_SEAL_SHA256",
        transition["seal_sha256"],
    )
    monkeypatch.setattr(
        module,
        "SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256",
        template_equivalence["seal_sha256"],
    )
    adopted_task_id = "6" * 32
    legacy_receipt = _seal(
        {
            "schema_version": 1,
            "contract": "nested-teacher-semantics-preserving-v1",
            "passed": True,
        }
    )
    binding_receipt = {
        "task_id": adopted_task_id,
        "experiment": "support_residual",
        "template_role": "legacy",
        "template_task_id": "7" * 32,
        "template_script_sha256": module.LEGACY_TRAINING_SCRIPT_SHA256,
        "task_script_sha256": module.LEGACY_TRAINING_SCRIPT_SHA256,
        "script_identity_policy": (
            "exact_allowlisted_legacy_nested_teacher_semantics_preserving"
        ),
        "legacy_script_compatibility_receipt": legacy_receipt,
        "source_parameters": {},
        "config_path": "configs/resilient_v2x/improvements/support_residual.py",
        "config_inventory_status": "verified",
        "expected_parameter_count": 0,
        "expected_parameters_sha256": "3" * 64,
        "observed_parameter_projection_sha256": "3" * 64,
        "exact_execution_parameter_match": True,
        "predecessor_task_id": None,
    }
    source_progress_revision = 7
    source_progress_seal = "4" * 64
    recovery = {
        "schema_version": 4,
        "mode": "failed_controller_immutable_fork",
        "source_controller_task_id": source_controller_task_id,
        "source_controller_status": "failed",
        "source_progress_revision": source_progress_revision,
        "source_progress_seal_sha256": source_progress_seal,
        "source_progress_artifact_readback": {
            "url": (
                f"clearml://{source_controller_task_id}/"
                f"{module.TRAINING_PROGRESS_ARTIFACT}"
            ),
            "revision": source_progress_revision,
            "seal_sha256": source_progress_seal,
            "force_download": True,
            "stable_readbacks": 2,
        },
        "source_template_script_sha256": source_template_identity["script_sha256"],
        "target_template_script_sha256": target_template_identity["script_sha256"],
        "source_revision_transition": transition,
        "source_progress_template_equivalence": template_equivalence,
        "source_patch": "nested-teacher-config-consistency-v1",
        "source_recovery_chain": [],
        "rerun_experiments": [],
        "rerun_source_task_ids": {},
        "rerun_task_observations": {},
        "rerun_predecessor_task_ids": {},
        "completion_validation_retries": 0,
        "recovered_pending_target_children": [],
        "transition_replaced_source_tasks": [],
        "target_template_predecessor_task_ids": {},
        "recovery_target_adoptions": {},
        "adopted_task_ids": {"support_residual": adopted_task_id},
        "adopted_predecessor_task_ids": {"support_residual": None},
        "adopted_task_template_roles": {"support_residual": "legacy"},
        "adopted_task_binding_receipts": {
            "support_residual": binding_receipt
        },
    }
    progress = _seal(
        {
            "schema_version": 1,
            "document_type": "training_progress",
            "recovery": recovery,
        }
    )
    source_binding = _source_revision_binding(module)
    provenance_artifact = _seal(
        {
            "schema_version": 2,
            "document_type": (
                "resilient_v2x_formal_training_provenance_equivalence"
            ),
            "protocol_id": module.AUDIT_PROTOCOL_ID,
            "passed": True,
            "controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
            "controller_status": "completed",
            "controller_raw_authority_sha256": "5" * 64,
            "subject_order": list(module.AUDIT_SUBJECT_ORDER),
            "subject_count": len(module.AUDIT_SUBJECT_ORDER),
            "all_training_tasks_completed": True,
            "authoritative_metadata_read": "single_batch_per_snapshot",
            "progress_artifact": module.TRAINING_PROGRESS_ARTIFACT,
            "progress_seal_sha256": progress["seal_sha256"],
            "progress_content_sha256": _digest(progress),
            "formal_training_manifest_artifact": module.TRAINING_MANIFEST_ARTIFACT,
            "formal_training_manifest_seal_sha256": "6" * 64,
            "formal_training_manifest_content_sha256": "7" * 64,
            "recovery_contract_sha256": _digest(recovery),
            "recursive_progress_chain": [],
            "bootstrap_equivalence": {},
            "source_revision_equivalence": source_binding[
                "source_revision_equivalence"
            ],
            "source_revision_equivalence_seal_sha256": source_binding[
                "source_revision_equivalence_seal_sha256"
            ],
            "source_revision_subject_map": source_binding[
                "source_revision_subject_map"
            ],
            "source_revision_subject_map_seal_sha256": source_binding[
                "source_revision_subject_map_seal_sha256"
            ],
            "run_contract_equivalence": {},
            "capacity_matched_hardware": {},
            "recovery_parent_controllers": [],
            "training_tasks": [],
        }
    )
    validator_source = "\n".join(
        (
            f"PROGRESS_ARTIFACT = {module.TRAINING_PROGRESS_ARTIFACT!r}",
            "SOURCE_REVISION_TRANSITION_SEAL_SHA256 = "
            f"{transition['seal_sha256']!r}",
            "SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256 = "
            f"{template_equivalence['seal_sha256']!r}",
            "def _validate_progress(progress, *, controller_task_id):",
            "    if progress.get('schema_version') != 1:",
            "        raise ValueError('schema')",
            "    return [], progress['recovery'], progress['seal_sha256']",
            "",
        )
    )
    return validator_source, progress, provenance_artifact


def _world(monkeypatch):
    module = _load("clearml_formal_multiseed_plan_fixture", MODULE_PATH)
    monkeypatch.setattr(module, "ClearMLArtifact", _Artifact)
    controller_producer = "#!/usr/bin/env python3\n# controller fixture\n"
    provenance_producer, progress, provenance_artifact = (
        _schema4_training_documents(module, monkeypatch)
    )
    watcher_producer = "#!/usr/bin/env python3\n# watcher fixture\n"
    selector_producer = "#!/usr/bin/env python3\n# selector producer fixture\n"
    audit_producer = "#!/usr/bin/env python3\n# audit producer fixture\n"
    leaderboard_producer = "#!/usr/bin/env python3\n# leaderboard producer fixture\n"
    monkeypatch.setattr(
        module,
        "TRAINING_CONTROLLER_PRODUCER_SHA256",
        _digest_text(controller_producer),
    )
    monkeypatch.setattr(
        module,
        "TRAINING_PROVENANCE_PRODUCER_SHA256",
        _digest_text(provenance_producer),
    )
    monkeypatch.setattr(
        module, "WATCHER_PRODUCER_SHA256", _digest_text(watcher_producer)
    )
    monkeypatch.setattr(
        module, "SELECTOR_PRODUCER_SHA256", _digest_text(selector_producer)
    )
    monkeypatch.setattr(module, "AUDIT_PRODUCER_SHA256", _digest_text(audit_producer))
    monkeypatch.setattr(
        module,
        "LEADERBOARD_PRODUCER_SHA256",
        _digest_text(leaderboard_producer),
    )
    controller_task = _Task(
        _server(
            task_id=module.TRAINING_CONTROLLER_TASK_ID,
            parent="",
            status="completed",
            script=_script(module.TRAINING_CONTROLLER_ENTRY_POINT, controller_producer),
            artifacts={
                module.TRAINING_MANIFEST_ARTIFACT: _Artifact(
                    _seal({"kind": "manifest"})
                ),
                module.TRAINING_PROGRESS_ARTIFACT: _Artifact(progress),
                module.TRAINING_SUMMARY_ARTIFACT: _Artifact(_seal({"kind": "summary"})),
            },
        )
    )
    provenance_task = _Task(
        _server(
            task_id=module.TRAINING_PROVENANCE_TASK_ID,
            parent=module.TRAINING_CONTROLLER_TASK_ID,
            status="completed",
            script=_script(module.TRAINING_PROVENANCE_ENTRY_POINT, provenance_producer),
            artifacts={
                module.TRAINING_PROVENANCE_ARTIFACT: _Artifact(provenance_artifact)
            },
            parameters={
                "Args/training_controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
                "Args/poll_seconds": module.SELECTOR_POLL_SECONDS,
                "Args/timeout_hours": module.SELECTOR_TIMEOUT_HOURS,
            },
        )
    )
    watcher_task = _Task(
        _server(
            task_id=module.WATCHER_TASK_ID,
            parent=module.TRAINING_PROVENANCE_TASK_ID,
            status="completed",
            script=_script(module.WATCHER_ENTRY_POINT, watcher_producer),
            artifacts={
                module.EVALUATION_PLAN_ARTIFACT: _Artifact(_seal({"kind": "plan"}))
            },
            parameters={
                "Args/training_controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
                "Args/training_provenance_task_id": module.TRAINING_PROVENANCE_TASK_ID,
                "Args/evaluation_template_task_id": "8b77a3674dfe405388aae39ef82d06ef",
                "Args/expected_training_provenance_script_sha256": _digest_text(
                    provenance_producer
                ),
                "Args/expected_eval_script_sha256": module.EVALUATION_SCRIPT_SHA256,
                "Args/expected_source_dataset_id": module.AUDIT_SOURCE_DATASET_ID,
                "Args/expected_source_archive_sha256": module.AUDIT_SOURCE_ARCHIVE_SHA256,
                "Args/expected_training_source_dataset_id": module.AUDIT_SOURCE_DATASET_ID,
                "Args/expected_training_source_archive_sha256": module.AUDIT_SOURCE_ARCHIVE_SHA256,
                "Args/expected_training_dataset_id": module.AUDIT_TRAINING_DATASET_ID,
                "Args/poll_seconds": module.SELECTOR_POLL_SECONDS,
                "Args/timeout_hours": module.SELECTOR_TIMEOUT_HOURS,
            },
        )
    )
    leaderboard_artifact = _formal_leaderboard_artifact(
        module,
        provenance_seal=str(provenance_artifact["seal_sha256"]),
    )
    monkeypatch.setattr(
        module,
        "LEADERBOARD_ARTIFACT_SEAL_SHA256",
        leaderboard_artifact["seal_sha256"],
    )
    monkeypatch.setattr(
        module,
        "LEADERBOARD_ARTIFACT_CANONICAL_SHA256",
        _digest(leaderboard_artifact),
    )
    leaderboard_task = _Task(
        _server(
            task_id=module.SELECTOR_LEADERBOARD_TASK_ID,
            parent=module.SELECTOR_WATCHER_TASK_ID,
            status="completed",
            script=_script(module.LEADERBOARD_ENTRY_POINT, leaderboard_producer),
            artifacts={module.LEADERBOARD_ARTIFACT: _Artifact(leaderboard_artifact)},
            parameters={
                "Args/watcher_task_id": module.SELECTOR_WATCHER_TASK_ID,
                "Args/training_controller_task_id": (
                    module.SELECTOR_TRAINING_CONTROLLER_TASK_ID
                ),
                "Args/poll_seconds": module.LEADERBOARD_POLL_SECONDS,
                "Args/timeout_hours": module.LEADERBOARD_TIMEOUT_HOURS,
            },
        )
    )
    audit_artifact = _formal_audit_artifact(
        module,
        leaderboard_seal=str(leaderboard_artifact["seal_sha256"]),
        progress_seal=str(progress["seal_sha256"]),
        provenance_seal=str(provenance_artifact["seal_sha256"]),
    )
    audit_task = _Task(
        _server(
            task_id=module.SELECTOR_AUDIT_TASK_ID,
            parent=module.SELECTOR_LEADERBOARD_TASK_ID,
            status="completed",
            script=_script(module.AUDIT_ENTRY_POINT, audit_producer),
            artifacts={module.AUDIT_ARTIFACT: _Artifact(audit_artifact)},
            parameters={
                "Args/training_controller_task_id": (
                    module.SELECTOR_TRAINING_CONTROLLER_TASK_ID
                ),
                "Args/training_provenance_task_id": (
                    module.TRAINING_PROVENANCE_TASK_ID
                ),
                "Args/watcher_task_id": module.SELECTOR_WATCHER_TASK_ID,
                "Args/leaderboard_task_id": module.SELECTOR_LEADERBOARD_TASK_ID,
                "Args/poll_seconds": module.SELECTOR_POLL_SECONDS,
                "Args/timeout_hours": module.SELECTOR_TIMEOUT_HOURS,
            },
        )
    )
    selection = _selector_artifact(
        module,
        audit_seal=str(audit_artifact["seal_sha256"]),
        leaderboard_seal=str(leaderboard_artifact["seal_sha256"]),
        progress_seal=str(progress["seal_sha256"]),
        provenance_seal=str(provenance_artifact["seal_sha256"]),
    )
    selector_task = _Task(
        _server(
            task_id=module.SELECTOR_TASK_ID,
            parent=module.SELECTOR_AUDIT_TASK_ID,
            status="completed",
            script=_script(module.SELECTOR_ENTRY_POINT, selector_producer),
            artifacts={module.SELECTOR_ARTIFACT: _Artifact(selection)},
            parameters={
                "Args/audit_task_id": module.SELECTOR_AUDIT_TASK_ID,
                "Args/leaderboard_task_id": module.SELECTOR_LEADERBOARD_TASK_ID,
                "Args/poll_seconds": module.SELECTOR_POLL_SECONDS,
                "Args/timeout_hours": module.SELECTOR_TIMEOUT_HOURS,
            },
        )
    )
    output_task = _Task(
        _server(
            task_id="f" * 32,
            parent=module.SELECTOR_TASK_ID,
            status="in_progress",
            script=_script(
                module.PRODUCER_ENTRY_POINT,
                MODULE_PATH.read_text(encoding="utf-8"),
            ),
            parameters=_production_planner_parameters(module),
        )
    )
    api = _TaskApi(
        controller_task,
        provenance_task,
        watcher_task,
        selector_task,
        audit_task,
        leaderboard_task,
        output_task,
    )
    return module, controller_task, selector_task, output_task, api


def test_fixed_pins_and_standalone_are_byte_reproducible() -> None:
    module = _load("planner_producer_pins", MODULE_PATH)
    assert _digest_text(VALIDATOR_PATH.read_text(encoding="utf-8")) == (
        module.PLAN_VALIDATOR_SHA256
    )
    assert module.TRAINING_CONTROLLER_TASK_ID == "1011e98e10f64c428880af1d4b1d542b"
    assert module.TRAINING_PROVENANCE_TASK_ID == "7274a1a5344a44c18aa7bcc6d1cb2e95"
    assert module.WATCHER_TASK_ID == "7734387ddfb74b11ba6d84f3fea0bb97"
    assert module.LEADERBOARD_TASK_ID == "f502bdd329ad4ef4b4b6cf5c5f52aba0"
    assert module.AUDIT_TASK_ID == "e19bab92884248f4ac07167e7eb66170"
    assert module.SELECTOR_TASK_ID == "b8876e88bb494985a900e3d627439776"
    assert module.SELECTOR_PRODUCER_SHA256 == (
        "4610802b0854fc3d6b58db9b2863bdc1ab1757c83dccbcd677dd577374f86ac1"
    )
    assert module.AUDIT_PRODUCER_SHA256 == (
        "4746ae18bb67757c974d96e851f19cb01872f460aaf3b334609a0de988aaabaa"
    )
    assert module.LEADERBOARD_ARTIFACT_SEAL_SHA256 == ""
    assert module.LEADERBOARD_ARTIFACT_CANONICAL_SHA256 == ""
    with pytest.raises(module.FormalMultiseedPlanError, match="unresolved"):
        module._require_deployment_pins()
    standalone = module.generate_standalone_source()
    compile(standalone, "<standalone>", "exec")
    tree = ast.parse(standalone)
    encoded = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if isinstance(node.targets[0], ast.Name) and (
            node.targets[0].id == "_EMBEDDED_PLAN_VALIDATOR_B64"
        ):
            encoded = ast.literal_eval(node.value)
            break
    assert isinstance(encoded, str) and encoded
    assert base64.b64decode(encoded, validate=True) == VALIDATOR_PATH.read_bytes()
    namespace = {
        "__name__": "standalone_planner_fixture",
        "__file__": "/definitely/missing/formal_multiseed_plan.py",
    }
    exec(compile(standalone, "<standalone>", "exec"), namespace)
    validator = namespace["_load_validator"]()
    assert callable(validator.build_plan)


def test_upstream_producer_bytes_match_declared_pins() -> None:
    module = _load("planner_upstream_pins", MODULE_PATH)
    assert _digest_text(PROVENANCE_PRODUCER_PATH.read_text(encoding="utf-8")) == (
        module.TRAINING_PROVENANCE_PRODUCER_SHA256
    )
    assert _digest_text(SELECTOR_PATH.read_text(encoding="utf-8")) == (
        module.SELECTOR_PRODUCER_SHA256
    )
    assert _digest_text(AUDIT_PRODUCER_PATH.read_text(encoding="utf-8")) == (
        module.AUDIT_PRODUCER_SHA256
    )
    selector = _load("planner_selector_leaderboard_pin", SELECTOR_PATH)
    assert module.LEADERBOARD_PRODUCER_SHA256 == selector.LEADERBOARD_SCRIPT_SHA256
    audit = _load("planner_audit_contract_pin", AUDIT_PRODUCER_PATH)
    assert module.AUDIT_PROTOCOL_ID == audit.PROTOCOL_ID
    assert module.AUDIT_SOURCE_DATASET_ID == audit.SOURCE_DATASET_ID
    assert module.AUDIT_SOURCE_ARCHIVE_SHA256 == audit.SOURCE_ARCHIVE_SHA256
    assert module.AUDIT_TRAINING_DATASET_ID == audit.TRAINING_DATASET_ID
    assert module.AUDIT_TRAINING_SEED == audit.TRAINING_SEED
    assert module.AUDIT_SAMPLE_COUNT == audit.SAMPLE_COUNT
    assert module.AUDIT_GROUND_TRUTH_COUNT == audit.GROUND_TRUTH_COUNT
    assert module.AUDIT_UNSUPPORTED_SAMPLE_COUNT == audit.UNSUPPORTED_SAMPLE_COUNT
    assert module.AUDIT_DELAYS_MS == audit.DELAYS_MS
    assert module.AUDIT_CONDITIONS == audit.CONDITIONS
    assert module.AUDIT_SUBJECT_ORDER == audit.SUBJECT_ORDER
    assert module.AUDIT_BASELINE_SUBJECTS == audit.BASELINE_SUBJECTS
    assert module.AUDIT_METRIC_KEYS == audit.AP_METRIC_KEYS
    assert module.AUDIT_SUPPORTED_QUEUES == audit.SUPPORTED_QUEUES
    assert module.AUDIT_EVALUATION_SCRIPT_SHA256 == audit.EVALUATION_SCRIPT_SHA256
    assert module.AUDIT_CHECKPOINT_POLICY == audit.CHECKPOINT_POLICY
    assert module.AUDIT_MODEL_BYTES_VERIFIER == audit.MODEL_BYTES_VERIFIER
    assert module.LEGACY_TRAINING_SCRIPT_SHA256 == audit.LEGACY_TRAINING_SCRIPT_SHA256
    assert (
        module.CANONICAL_TRAINING_SCRIPT_SHA256
        == audit.CANONICAL_TRAINING_SCRIPT_SHA256
    )


def test_forced_artifact_download_failure_has_no_fallback(monkeypatch) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)

    class RejectFreshDownload(_Artifact):
        def get(self, *, force_download: object = None) -> object:
            assert force_download is True
            raise TypeError("fresh download unsupported")

    monkeypatch.setattr(module, "ClearMLArtifact", RejectFreshDownload)
    with pytest.raises(module.FormalMultiseedPlanError, match="cannot be downloaded"):
        module.run(_args(), task_class=api)
    assert output.uploads == []


def test_generated_standalone_runs_end_to_end_without_checkout(
    monkeypatch,
) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    standalone = module.generate_standalone_source()
    namespace = {
        "__name__": "standalone_planner_end_to_end",
        "__file__": "/definitely/missing/formal_multiseed_plan.py",
    }
    exec(compile(standalone, "<standalone>", "exec"), namespace)
    for name in (
        "SOURCE_C_SCRIPT_SHA256",
        "SOURCE_D_SCRIPT_SHA256",
        "SOURCE_D_EQUIVALENCE_SHA256",
        "SOURCE_D_PRODUCER_SHA256",
        "TRAINING_CONTROLLER_PRODUCER_SHA256",
        "TRAINING_PROVENANCE_PRODUCER_SHA256",
        "WATCHER_PRODUCER_SHA256",
        "SELECTOR_PRODUCER_SHA256",
        "AUDIT_PRODUCER_SHA256",
        "LEADERBOARD_PRODUCER_SHA256",
            "LEADERBOARD_ARTIFACT_SEAL_SHA256",
            "LEADERBOARD_ARTIFACT_CANONICAL_SHA256",
            "SOURCE_REVISION_TRANSITION_SEAL_SHA256",
            "SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256",
        "SOURCE_C_SNAPSHOT_SEAL_SHA256",
        "SOURCE_C_SNAPSHOT_CANONICAL_SHA256",
        "SOURCE_D_ARTIFACT_SEAL_SHA256",
        "SOURCE_D_ARTIFACT_CANONICAL_SHA256",
        "SOURCE_D_EQUIVALENCE_CANONICAL_SHA256",
        "SOURCE_D_RECEIPT_SEAL_SHA256",
        "SOURCE_D_RECEIPT_CANONICAL_SHA256",
    ):
        namespace[name] = getattr(module, name)
    namespace["ClearMLArtifact"] = _Artifact
    namespace["_runtime_source"] = lambda: standalone
    output.server.script.diff = standalone
    output._data = copy.deepcopy(output.server)
    receipt = namespace["run"](_args(), task_class=api)
    assert receipt["complete"] is True
    assert receipt["producer_script_sha256"] == _digest_text(standalone)
    assert tuple(output.server.artifacts) == module.PUBLICATION_ORDER


def test_run_publishes_exact_plan_then_receipt(monkeypatch) -> None:
    module, source_d, selector, output, api = _world(monkeypatch)
    receipt = module.run(_args(), task_class=api)
    assert api.requested == [
        module.TRAINING_CONTROLLER_TASK_ID,
        module.TRAINING_PROVENANCE_TASK_ID,
        module.WATCHER_TASK_ID,
        module.LEADERBOARD_TASK_ID,
        module.AUDIT_TASK_ID,
        module.SELECTOR_TASK_ID,
    ]
    assert output.uploads == list(module.PUBLICATION_ORDER)
    assert output.flushes == list(module.PUBLICATION_ORDER)
    assert tuple(output.server.artifacts) == module.PUBLICATION_ORDER
    plan = output.server.artifacts[module.PLAN_ARTIFACT].get()
    assert plan["schema_version"] == 2
    assert plan["training_provenance_task_id"] == module.TRAINING_PROVENANCE_TASK_ID
    assert plan["evaluation_script_sha256"] == module.EVALUATION_SCRIPT_SHA256
    assert plan["training_task_count"] == 45
    assert plan["evaluation_task_count"] == 45
    assert plan["total_evaluation_condition_count"] == 540
    assert plan["formal_selector_provenance"]["task_id"] == module.SELECTOR_TASK_ID
    assert (
        plan["formal_audit_seal_sha256"]
        == (api.audit.server.artifacts[module.AUDIT_ARTIFACT].value["seal_sha256"])
    )
    assert receipt["planner_parent_task_id"] == module.SELECTOR_TASK_ID
    assert receipt["producer_script_sha256"] == _digest_text(
        MODULE_PATH.read_text(encoding="utf-8")
    )
    assert source_d.public_reload_calls == 0
    assert selector.public_reload_calls == 0
    assert api.leaderboard.public_reload_calls == 0
    assert output.public_reload_calls == 0
    assert source_d.raw_reload_calls > 2
    assert selector.raw_reload_calls > 2
    assert api.leaderboard.raw_reload_calls > 2
    assert output.raw_reload_calls > 2


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("legacy_schema", "schema-v4 immutable fork"),
        ("single_readback", "readback drifted"),
        ("malformed_template_identity", "template identities are invalid"),
        ("missing_legacy_receipt", "lacks the sealed legacy"),
    ),
)
def test_schema4_recovery_rejects_weakened_evidence(
    monkeypatch, attack: str, message: str
) -> None:
    module, controller, _selector, _output, _api = _world(monkeypatch)
    progress = controller.server.artifacts[module.TRAINING_PROGRESS_ARTIFACT].value
    recovery = copy.deepcopy(progress["recovery"])
    if attack == "legacy_schema":
        recovery["schema_version"] = 3
    elif attack == "single_readback":
        recovery["source_progress_artifact_readback"]["stable_readbacks"] = 1
    elif attack == "malformed_template_identity":
        transition = recovery["source_revision_transition"]
        transition["source_template_identity"] = []
        recovery["source_revision_transition"] = _seal(transition)
    else:
        receipt = recovery["adopted_task_binding_receipts"]["support_residual"]
        receipt["script_identity_policy"] = "exact_canonical_template"
        receipt["legacy_script_compatibility_receipt"] = None
    with pytest.raises(module.FormalMultiseedPlanError, match=message):
        module._validate_schema4_recovery_shape(recovery)


def test_executor_resolves_producer_output_without_schema_translation(
    monkeypatch,
) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    module.run(_args(), task_class=api)
    output.server.status = "completed"
    output._data = copy.deepcopy(output.server)
    executor = _load("executor_planner_compatibility", EXECUTOR_PATH)
    monkeypatch.setattr(executor, "ClearMLArtifact", _Artifact)
    monkeypatch.setattr(
        executor, "PINNED_PLAN_VALIDATOR_SHA256", module.PLAN_VALIDATOR_SHA256
    )
    plan, receipt = executor._resolve_plan(
        output,
        planner_task_id=output.id,
        producer_sha256=_digest_text(MODULE_PATH.read_text(encoding="utf-8")),
        validator_path=VALIDATOR_PATH,
    )
    assert plan["training_task_count"] == 45
    assert receipt["complete"] is True


def test_waits_for_both_dependencies(monkeypatch) -> None:
    module, source_d, selector, output, api = _world(monkeypatch)
    source_d.server.status = "in_progress"
    source_d._data.status = "in_progress"
    source_d.statuses = ["in_progress", "completed"]
    selector.server.status = "queued"
    selector._data.status = "queued"
    selector.statuses = ["queued", "completed"]
    sleeps: list[float] = []
    module.run(_args(), task_class=api, sleeper=sleeps.append)
    assert sleeps == [1.0]
    assert output.uploads == list(module.PUBLICATION_ORDER)


@pytest.mark.parametrize(
    ("dependency", "status"),
    (("source_d", "failed"), ("selector", "stopped"), ("selector", "published")),
)
def test_terminal_dependency_fails_before_upload(
    monkeypatch, dependency: str, status: str
) -> None:
    module, source_d, selector, output, api = _world(monkeypatch)
    target = source_d if dependency == "source_d" else selector
    target.server.status = target._data.status = status
    with pytest.raises(module.FormalMultiseedPlanError, match="ended as"):
        module.run(_args(), task_class=api)
    assert output.uploads == []


@pytest.mark.parametrize("status", ("attacker.completed", "COMPLETED", 1, True))
def test_dependency_status_spoof_fails_before_upload(
    monkeypatch, status: object
) -> None:
    module, source_d, _selector, output, api = _world(monkeypatch)
    source_d.server.status = source_d._data.status = status
    with pytest.raises(module.FormalMultiseedPlanError, match="status"):
        module.run(_args(), task_class=api)
    assert output.uploads == []


def test_wait_timeout_fails_before_upload(monkeypatch) -> None:
    module, source_d, _selector, output, api = _world(monkeypatch)
    source_d.server.status = source_d._data.status = "queued"
    clock = iter((0.0, 3601.0))
    with pytest.raises(TimeoutError, match="timed out"):
        module.run(
            _args(),
            task_class=api,
            monotonic_clock=lambda: next(clock),
            sleeper=lambda _seconds: None,
        )
    assert output.uploads == []


@pytest.mark.parametrize(
    "snapshot", (None, False, True, 0, 1, 0.0, "", "snapshot", b"bytes")
)
def test_raw_reload_rejects_non_snapshot_and_restores_flag(
    monkeypatch, snapshot: object
) -> None:
    module, source_d, _selector, output, _api = _world(monkeypatch)
    source_d.reload_result = snapshot
    with pytest.raises(module.FormalMultiseedPlanError, match="no snapshot"):
        module._reload(source_d, context="attack")
    assert source_d._reload_skip_flag is True
    assert output.uploads == []


def test_raw_reload_rejects_offline_error_and_wrong_identity(monkeypatch) -> None:
    module, source_d, _selector, _output, _api = _world(monkeypatch)
    source_d._offline_mode = True
    with pytest.raises(module.FormalMultiseedPlanError, match="offline"):
        module._reload(source_d, context="attack")
    source_d._offline_mode = False
    source_d.reload_exception = RuntimeError("transport")
    with pytest.raises(module.FormalMultiseedPlanError, match="server reload failed"):
        module._reload(source_d, context="attack")
    assert source_d._reload_skip_flag is True
    source_d.reload_exception = None
    wrong = copy.deepcopy(source_d.server)
    wrong.id = "e" * 32
    source_d.reload_result = wrong
    with pytest.raises(module.FormalMultiseedPlanError, match="identity mismatch"):
        module._reload(source_d, context="attack")


def test_raw_reload_rejects_silently_discarded_snapshot(monkeypatch) -> None:
    module, source_d, _selector, _output, _api = _world(monkeypatch)

    class SilentTask(_Task):
        discard = False

        def __setattr__(self, name: str, value: object) -> None:
            if name == "_data" and getattr(self, "discard", False):
                return
            super().__setattr__(name, value)

    attacked = SilentTask(source_d.server)
    attacked.discard = True
    with pytest.raises(
        module.FormalMultiseedPlanError, match="installation did not persist"
    ):
        module._reload(attacked, context="attack")
    assert attacked._reload_skip_flag is True


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    (
        ("controller", "parent", "training controller parent"),
        ("controller", "script", "script SHA-256"),
        ("controller", "extra_artifact", "artifact inventory"),
        ("provenance", "parent", "training provenance parent"),
        ("provenance", "script", "script SHA-256"),
        ("provenance", "parameter", "parameter"),
        ("provenance", "extra_parameter", "parameter"),
        ("provenance", "extra_artifact", "artifact inventory"),
        ("watcher", "parent", "evaluation watcher parent"),
        ("watcher", "script", "script SHA-256"),
        ("watcher", "parameter", "parameter"),
        ("watcher", "extra_artifact", "artifact inventory"),
        ("selector", "parent", "selector parent"),
        ("selector", "script", "script SHA-256"),
        ("selector", "parameter", "parameter"),
        ("selector", "extra_parameter", "parameter"),
        ("selector", "extra_artifact", "artifact inventory"),
        ("leaderboard", "parent", "leaderboard parent"),
        ("leaderboard", "script", "script SHA-256"),
        ("leaderboard", "parameter", "parameter"),
        ("leaderboard", "extra_parameter", "parameter"),
        ("leaderboard", "extra_artifact", "artifact inventory"),
    ),
)
def test_dependency_provenance_attacks_upload_nothing(
    monkeypatch, target: str, mutation: str, message: str
) -> None:
    module, source_d, selector, output, api = _world(monkeypatch)
    if target == "controller":
        task = source_d
    elif target == "provenance":
        task = api.provenance
    elif target == "watcher":
        task = api.watcher
    elif target == "selector":
        task = selector
    else:
        task = api.leaderboard
    if mutation == "parent":
        task.server.parent = task._data.parent = "1" * 32
    elif mutation == "script":
        task.server.script.diff = task._data.script.diff = "tampered"
    elif mutation == "parameter":
        key = {
            "selector": "Args/audit_task_id",
            "leaderboard": "Args/watcher_task_id",
            "provenance": "Args/training_controller_task_id",
            "watcher": "Args/training_provenance_task_id",
        }[target]
        task.server.parameters[key] = "2" * 32
        task._data.parameters[key] = "2" * 32
    elif mutation == "extra_parameter":
        task.server.parameters["Args/attacker_mode"] = "enabled"
        task._data.parameters["Args/attacker_mode"] = "enabled"
    else:
        task.server.artifacts["attacker"] = _Artifact({})
        task._data.artifacts["attacker"] = _Artifact({})
    with pytest.raises(module.FormalMultiseedPlanError, match=message):
        module.run(_args(), task_class=api)
    assert output.uploads == []


@pytest.mark.parametrize(
    "attack",
    ("unsealed", "ranking", "audit_id", "audit_seal"),
)
def test_selector_artifact_attacks_upload_nothing(monkeypatch, attack: str) -> None:
    module, _source_d, selector, output, api = _world(monkeypatch)
    artifact = selector.server.artifacts[module.SELECTOR_ARTIFACT].value
    assert isinstance(artifact, dict)
    if attack == "unsealed":
        artifact["selected_candidate"] = "support_residual"
    elif attack == "ranking":
        artifact["ranked_candidates"][0:2] = reversed(
            artifact["ranked_candidates"][0:2]
        )
        artifact = _seal(artifact)
        selector.server.artifacts[module.SELECTOR_ARTIFACT] = _Artifact(artifact)
    elif attack == "audit_id":
        artifact["audit_task_id"] = "3" * 32
        artifact = _seal(artifact)
        selector.server.artifacts[module.SELECTOR_ARTIFACT] = _Artifact(artifact)
    else:
        artifact["audit_seal_sha256"] = "c" * 64
        artifact = _seal(artifact)
        selector.server.artifacts[module.SELECTOR_ARTIFACT] = _Artifact(artifact)
    selector._data = copy.deepcopy(selector.server)
    with pytest.raises(module.FormalMultiseedPlanError):
        module.run(_args(), task_class=api)
    assert output.uploads == []


@pytest.mark.parametrize(
    "attack",
    ("status", "parent", "script", "parameters", "extra_artifact", "artifact"),
)
def test_audit_task_attacks_upload_nothing(monkeypatch, attack: str) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    audit = api.audit
    if attack == "status":
        audit.server.status = audit._data.status = "failed"
    elif attack == "parent":
        audit.server.parent = audit._data.parent = "1" * 32
    elif attack == "script":
        audit.server.script.diff = audit._data.script.diff = "tampered"
    elif attack == "parameters":
        audit.server.parameters["Args/attacker"] = "enabled"
        audit._data.parameters["Args/attacker"] = "enabled"
    elif attack == "extra_artifact":
        audit.server.artifacts["attacker"] = _Artifact({})
        audit._data.artifacts["attacker"] = _Artifact({})
    else:
        value = audit.server.artifacts[module.AUDIT_ARTIFACT].value
        assert isinstance(value, dict)
        value["passed"] = False
        audit.server.artifacts[module.AUDIT_ARTIFACT] = _Artifact(_seal(value))
        audit._data = copy.deepcopy(audit.server)
    with pytest.raises(module.FormalMultiseedPlanError):
        module.run(_args(), task_class=api)
    assert output.uploads == []


@pytest.mark.parametrize(
    "attack", ("status", "unsealed", "resealed", "extra_key", "canonical_pin")
)
def test_leaderboard_artifact_attacks_upload_nothing(monkeypatch, attack: str) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    leaderboard = api.leaderboard
    if attack == "status":
        leaderboard.server.status = leaderboard._data.status = "failed"
    elif attack == "canonical_pin":
        monkeypatch.setattr(module, "LEADERBOARD_ARTIFACT_CANONICAL_SHA256", "d" * 64)
    else:
        value = copy.deepcopy(
            leaderboard.server.artifacts[module.LEADERBOARD_ARTIFACT].value
        )
        assert isinstance(value, dict)
        if attack in {"unsealed", "resealed"}:
            value["sample_count"] = module.AUDIT_SAMPLE_COUNT - 1
        else:
            value["attacker"] = True
        if attack != "unsealed":
            value = _seal(value)
        leaderboard.server.artifacts[module.LEADERBOARD_ARTIFACT] = _Artifact(value)
        leaderboard._data = copy.deepcopy(leaderboard.server)
    with pytest.raises(module.FormalMultiseedPlanError):
        module.run(_args(), task_class=api)
    assert output.uploads == []


@pytest.mark.parametrize(
    "attack",
    ("sample_count", "training_record", "extra_key", "leaderboard_seal"),
)
def test_coordinated_audit_and_selector_reseal_uploads_nothing(
    monkeypatch, attack: str
) -> None:
    module, _source_d, selector, output, api = _world(monkeypatch)
    audit_value = copy.deepcopy(api.audit.server.artifacts[module.AUDIT_ARTIFACT].value)
    assert isinstance(audit_value, dict)
    if attack == "sample_count":
        audit_value["sample_count"] = 1
    elif attack == "training_record":
        audit_value["training_tasks"][0]["gpus"] = 8
    elif attack == "extra_key":
        audit_value["attacker"] = True
    else:
        audit_value["leaderboard_seal_sha256"] = "e" * 64
    attacked_audit = _seal(audit_value)
    api.audit.server.artifacts[module.AUDIT_ARTIFACT] = _Artifact(attacked_audit)
    api.audit._data = copy.deepcopy(api.audit.server)

    selector_value = copy.deepcopy(
        selector.server.artifacts[module.SELECTOR_ARTIFACT].value
    )
    assert isinstance(selector_value, dict)
    selector_value["audit_seal_sha256"] = attacked_audit["seal_sha256"]
    if attack == "leaderboard_seal":
        selector_value["leaderboard_seal_sha256"] = "e" * 64
    selector.server.artifacts[module.SELECTOR_ARTIFACT] = _Artifact(
        _seal(selector_value)
    )
    selector._data = copy.deepcopy(selector.server)

    with pytest.raises(module.FormalMultiseedPlanError):
        module.run(_args(), task_class=api)
    assert output.uploads == []


def test_coordinated_leaderboard_audit_selector_reseal_uploads_nothing(
    monkeypatch,
) -> None:
    module, _source_d, selector, output, api = _world(monkeypatch)
    leaderboard_value = copy.deepcopy(
        api.leaderboard.server.artifacts[module.LEADERBOARD_ARTIFACT].value
    )
    assert isinstance(leaderboard_value, dict)
    leaderboard_value["results"][0]["training_task_id"] = "9" * 32
    attacked_leaderboard = _seal(leaderboard_value)
    api.leaderboard.server.artifacts[module.LEADERBOARD_ARTIFACT] = _Artifact(
        attacked_leaderboard
    )
    api.leaderboard._data = copy.deepcopy(api.leaderboard.server)

    audit_value = copy.deepcopy(api.audit.server.artifacts[module.AUDIT_ARTIFACT].value)
    assert isinstance(audit_value, dict)
    audit_value["leaderboard_seal_sha256"] = attacked_leaderboard["seal_sha256"]
    audit_value["training_tasks"][0]["training_task_id"] = "9" * 32
    audit_value["evaluation_tasks"][0]["training_task_id"] = "9" * 32
    attacked_audit = _seal(audit_value)
    api.audit.server.artifacts[module.AUDIT_ARTIFACT] = _Artifact(attacked_audit)
    api.audit._data = copy.deepcopy(api.audit.server)

    selector_value = copy.deepcopy(
        selector.server.artifacts[module.SELECTOR_ARTIFACT].value
    )
    assert isinstance(selector_value, dict)
    selector_value["audit_seal_sha256"] = attacked_audit["seal_sha256"]
    selector_value["leaderboard_seal_sha256"] = attacked_leaderboard["seal_sha256"]
    selector.server.artifacts[module.SELECTOR_ARTIFACT] = _Artifact(
        _seal(selector_value)
    )
    selector._data = copy.deepcopy(selector.server)

    with pytest.raises(
        module.FormalMultiseedPlanError,
        match="leaderboard artifact seal pin mismatch",
    ):
        module.run(_args(), task_class=api)
    assert output.uploads == []


@pytest.mark.parametrize(
    ("target", "reseal"),
    (
        ("progress", False),
        ("progress", True),
        ("provenance", False),
        ("provenance", True),
    ),
)
def test_training_provenance_seal_attacks_upload_nothing(
    monkeypatch, target: str, reseal: bool
) -> None:
    module, controller, _selector, output, api = _world(monkeypatch)
    task, artifact_name = (
        (controller, module.TRAINING_PROGRESS_ARTIFACT)
        if target == "progress"
        else (api.provenance, module.TRAINING_PROVENANCE_ARTIFACT)
    )
    value = copy.deepcopy(task.server.artifacts[artifact_name].value)
    assert isinstance(value, dict)
    value["attacker"] = True
    if reseal:
        value = _seal(value)
    task.server.artifacts[artifact_name] = _Artifact(value)
    task._data = copy.deepcopy(task.server)
    with pytest.raises(module.FormalMultiseedPlanError):
        module.run(_args(), task_class=api)
    assert output.uploads == []


@pytest.mark.parametrize(
    "names",
    (
        ("formal_multiseed_plan_receipt",),
        ("formal_multiseed_plan_receipt", "formal_multiseed_plan"),
        ("attacker",),
    ),
)
def test_invalid_existing_output_inventory_is_rejected(
    monkeypatch, names: tuple[str, ...]
) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    output.server.artifacts = {name: _Artifact({}) for name in names}
    output._data = copy.deepcopy(output.server)
    with pytest.raises(module.FormalMultiseedPlanError, match="artifact"):
        module.run(_args(), task_class=api)
    assert output.uploads == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("upload_result", False),
        ("upload_result", 0),
        ("upload_result", 1),
        ("flush_result", False),
        ("flush_result", 0),
    ),
)
def test_publish_return_values_fail_closed(
    monkeypatch, field: str, value: object
) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    setattr(output, field, value)
    with pytest.raises(module.FormalMultiseedPlanError):
        module.run(_args(), task_class=api)
    assert output.uploads == [module.PLAN_ARTIFACT]


@pytest.mark.parametrize("mode", ("missing", "drift"))
def test_server_readback_must_contain_exact_uploaded_bytes(
    monkeypatch, mode: str
) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    if mode == "missing":
        output.omit_upload = module.PLAN_ARTIFACT
    else:
        output.drift_upload = module.PLAN_ARTIFACT
    with pytest.raises(module.FormalMultiseedPlanError):
        module.run(_args(), task_class=api)
    assert output.uploads == [module.PLAN_ARTIFACT]


@pytest.mark.parametrize(
    "target", ("source", "leaderboard", "audit", "selector", "output")
)
def test_drift_after_plan_upload_prevents_receipt(monkeypatch, target: str) -> None:
    module, source_d, selector, output, api = _world(monkeypatch)

    def drift(name: str) -> None:
        if name != module.PLAN_ARTIFACT:
            return
        if target == "source":
            source_d.server.parent = "1" * 32
        elif target == "leaderboard":
            api.leaderboard.server.parent = "5" * 32
        elif target == "audit":
            api.audit.server.parent = "4" * 32
        elif target == "selector":
            selector.server.parameters["Args/audit_task_id"] = "2" * 32
        else:
            output.server.parent = "3" * 32

    output.after_upload = drift
    with pytest.raises(module.FormalMultiseedPlanError):
        module.run(_args(), task_class=api)
    assert output.uploads == [module.PLAN_ARTIFACT]
    assert tuple(output.server.artifacts) == (module.PLAN_ARTIFACT,)


def test_exact_rerun_and_exact_prefix_are_idempotent(monkeypatch) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    module.run(_args(), task_class=api)
    first_plan = copy.deepcopy(output.server.artifacts[module.PLAN_ARTIFACT])
    first_receipt = copy.deepcopy(output.server.artifacts[module.PLAN_RECEIPT_ARTIFACT])
    output.uploads.clear()
    output.flushes.clear()
    module.run(_args(), task_class=api)
    assert output.uploads == []
    output.server.artifacts = {module.PLAN_ARTIFACT: first_plan}
    output._data = copy.deepcopy(output.server)
    module.run(_args(), task_class=api)
    assert output.uploads == [module.PLAN_RECEIPT_ARTIFACT]
    assert _canonical(
        output.server.artifacts[module.PLAN_RECEIPT_ARTIFACT].value
    ) == _canonical(first_receipt.value)


def test_ambiguous_committed_upload_is_not_retried_in_same_run(monkeypatch) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    output.upload_raise_after_commit = True
    with pytest.raises(RuntimeError, match="ambiguous"):
        module.run(_args(), task_class=api)
    assert output.uploads == [module.PLAN_ARTIFACT]
    assert tuple(output.server.artifacts) == (module.PLAN_ARTIFACT,)


def test_wrong_output_parent_and_parent_binding_fail_closed(monkeypatch) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    output.server.parent = output._data.parent = "1" * 32
    with pytest.raises(module.FormalMultiseedPlanError, match="parent"):
        module.run(_args(), task_class=api)
    output.server.parent = output._data.parent = ""
    module._bind_output_parent(output)
    assert output.parent == module.SELECTOR_TASK_ID
    output.server.parent = output._data.parent = "2" * 32
    with pytest.raises(module.FormalMultiseedPlanError, match="different parent"):
        module._bind_output_parent(output)


def test_parent_binding_rejects_stale_local_empty_parent(monkeypatch) -> None:
    module, _source_d, _selector, output, _api = _world(monkeypatch)
    output._data.parent = ""
    output.server.parent = "1" * 32
    with pytest.raises(module.FormalMultiseedPlanError, match="different parent"):
        module._bind_output_parent(output)
    assert output.parent == "1" * 32


def test_local_perfect_server_empty_cannot_satisfy_readback(monkeypatch) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)
    output.local_artifacts[module.PLAN_ARTIFACT] = _Artifact({"local": "perfect"})
    assert output.server.artifacts == {}
    module.run(_args(), task_class=api)
    assert output.uploads == list(module.PUBLICATION_ORDER)
    assert tuple(output.server.artifacts) == module.PUBLICATION_ORDER


def test_local_upload_manager_entries_cannot_satisfy_server_readback(
    monkeypatch,
) -> None:
    module, _source_d, _selector, output, api = _world(monkeypatch)

    def local_only_upload(
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        assert wait_on_upload is True
        output.uploads.append(name)
        output.local_artifacts[name] = _Artifact(copy.deepcopy(artifact_object))
        return True

    output.upload_artifact = local_only_upload
    with pytest.raises(module.FormalMultiseedPlanError, match="absent after upload"):
        module.run(_args(), task_class=api)
    assert output.uploads == [module.PLAN_ARTIFACT]
    assert output.server.artifacts == {}
    assert module.PLAN_ARTIFACT in output.artifacts


def test_duplicate_raw_server_artifact_descriptors_fail_closed(monkeypatch) -> None:
    module, source_d, _selector, output, api = _world(monkeypatch)
    first = next(iter(source_d.server.artifacts.items()))
    duplicate = _RawArtifact(first[0], first[1].value)
    source_d.server.raw_artifact_override = [
        duplicate,
        copy.deepcopy(duplicate),
    ]
    source_d._data = copy.deepcopy(source_d.server)
    with pytest.raises(
        module.FormalMultiseedPlanError, match="raw server artifact inventory"
    ):
        module.run(_args(), task_class=api)
    assert output.uploads == []


def test_mutation_during_final_artifact_read_is_detected(monkeypatch) -> None:
    module, _source_d, selector, output, api = _world(monkeypatch)
    original = module._artifact_mapping
    plan_reads = 0

    def attacked(task: object, name: str, *, context: str):
        nonlocal plan_reads
        value = original(task, name, context=context)
        if task is output and name == module.PLAN_ARTIFACT:
            plan_reads += 1
            if plan_reads == 2:
                selector.server.parent = "1" * 32
        return value

    monkeypatch.setattr(module, "_artifact_mapping", attacked)
    with pytest.raises(module.FormalMultiseedPlanError, match="selector parent"):
        module.run(_args(), task_class=api)
    assert output.uploads == list(module.PUBLICATION_ORDER)


def test_mutation_during_final_dependency_pass_is_detected(monkeypatch) -> None:
    module, _controller, _selector, output, api = _world(monkeypatch)
    original_artifact_mapping = module._artifact_mapping
    original_chain_snapshots = module._training_chain_snapshots
    plan_reads = 0
    mutated = False

    def counted_mapping(task: object, name: str, *, context: str):
        nonlocal plan_reads
        value = original_artifact_mapping(task, name, context=context)
        if task is output and name == module.PLAN_ARTIFACT:
            plan_reads += 1
        return value

    def attacked_chain_snapshots(
        controller_task: object, provenance_task: object, watcher_task: object
    ):
        nonlocal mutated
        value = original_chain_snapshots(controller_task, provenance_task, watcher_task)
        if plan_reads >= 2 and not mutated:
            output.server.artifacts[module.PLAN_ARTIFACT].value = {"drift": True}
            mutated = True
        return value

    monkeypatch.setattr(module, "_artifact_mapping", counted_mapping)
    monkeypatch.setattr(module, "_training_chain_snapshots", attacked_chain_snapshots)
    with pytest.raises(module.FormalMultiseedPlanError, match="canonical bytes"):
        module.run(_args(), task_class=api)
    assert mutated is True
    assert output.uploads == list(module.PUBLICATION_ORDER)


def test_main_initializes_output_and_binds_selector_parent(monkeypatch, capsys) -> None:
    module, source_d, selector, output, _api = _world(monkeypatch)
    output.server.parent = output._data.parent = ""

    class TaskApi:
        init_calls: list[dict[str, object]] = []

        @classmethod
        def init(cls, **kwargs: object) -> _Task:
            cls.init_calls.append(dict(kwargs))
            return output

        @classmethod
        def get_task(cls, *, task_id: str) -> _Task:
            if task_id == module.TRAINING_CONTROLLER_TASK_ID:
                return source_d
            if task_id == module.TRAINING_PROVENANCE_TASK_ID:
                return _api.provenance
            if task_id == module.WATCHER_TASK_ID:
                return _api.watcher
            if task_id == module.SELECTOR_TASK_ID:
                return selector
            if task_id == module.SELECTOR_AUDIT_TASK_ID:
                return _api.audit
            if task_id == module.SELECTOR_LEADERBOARD_TASK_ID:
                return _api.leaderboard
            raise AssertionError(task_id)

    monkeypatch.setattr(module, "Task", TaskApi)
    assert module.main(["--poll-seconds", "1", "--timeout-hours", "1"]) == 0
    assert output.parent == module.SELECTOR_TASK_ID
    assert TaskApi.init_calls == [
        {
            "project_name": module.DEFAULT_PROJECT,
            "task_name": "ResilientV2X formal multi-seed plan",
            "reuse_last_task_id": False,
            "output_uri": module.FILES_SERVER_URI,
        }
    ]
    assert json.loads(capsys.readouterr().out)["complete"] is True


def test_main_rejects_unresolved_pins_before_task_init(monkeypatch) -> None:
    module = _load("planner_main_unresolved_pins", MODULE_PATH)

    class TaskApi:
        init_calls = 0

        @classmethod
        def init(cls, **_kwargs: object) -> object:
            cls.init_calls += 1
            raise AssertionError("unresolved pins must block before Task.init")

    monkeypatch.setattr(module, "Task", TaskApi)
    with pytest.raises(module.FormalMultiseedPlanError, match="unresolved"):
        module.main([])
    assert TaskApi.init_calls == 0


def test_emit_standalone_refuses_overwrite(tmp_path: Path) -> None:
    module = _load("planner_emit_standalone", MODULE_PATH)
    target = tmp_path / "formal_multiseed_plan.py"
    assert module.main(["--emit-standalone", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == module.generate_standalone_source()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.main(["--emit-standalone", str(target)])


def test_json_path_loader_rejects_symlink(tmp_path: Path) -> None:
    module = _load("planner_secure_json_path", MODULE_PATH)
    target = tmp_path / "artifact.json"
    target.write_text('{"valid":true}', encoding="utf-8")
    link = tmp_path / "artifact-link.json"
    link.symlink_to(target)
    with pytest.raises(module.FormalMultiseedPlanError, match="cannot be read"):
        module._read_json_path(link, context="attack")
