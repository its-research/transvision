from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_formal_comparability_audit.py"
CONTROLLER_ID = "a" * 32
WATCHER_ID = "b" * 32
LEADERBOARD_ID = "c" * 32
TEMPLATE_ID = f"{6:032x}"
GATE_ID = "e" * 32
PROVENANCE_ID = "6" * 32
LEGACY_PARENT_ID = "4" * 32
NO_DIST_PARENT_ID = "5" * 32
RECOVERY_PARENT_ID = "0" * 32
QUEUE_IDS = {
    "GPU4-A100": "7" * 32,
    "GPU4-V100": "8" * 32,
    "GPU4-5090": "9" * 32,
}
QUEUE_NAMES = {queue_id: name for name, queue_id in QUEUE_IDS.items()}

CONTROLLER_SOURCE = "sealed source-C training controller"
WATCHER_SOURCE = "sealed source-C dependency watcher"
LEADERBOARD_SOURCE = "sealed source-C leaderboard"
PROVENANCE_SOURCE = "sealed formal training provenance producer"
_BOOTSTRAP_SUFFIX = """
def consume(**kwargs):
    return kwargs

def uses_nested(spec):
    contract = {
        "expected_nested_teacher": spec.name in NESTED_TEACHER_EXPERIMENTS,
    }
    consume(
        expect_nested_teacher=spec.name in NESTED_TEACHER_EXPERIMENTS,
    )
    return contract
"""
LEGACY_TRAINING_BOOTSTRAP_SOURCE = (
    "NESTED_TEACHER_EXPERIMENTS = frozenset({'resilient_v2x', 'support_residual'})"
    + _BOOTSTRAP_SUFFIX
)
CANONICAL_TRAINING_BOOTSTRAP_SOURCE = (
    "NESTED_TEACHER_EXPERIMENTS = frozenset({"
    "'support_residual', 'ptf_none', 'ptf_linear', 'router_static', "
    "'router_uniform', 'no_reliability', 'no_delay_metadata', "
    "'concat_capacity_matched', 'resilient_v2x'})" + _BOOTSTRAP_SUFFIX
)
EVALUATION_BOOTSTRAP_SOURCE = CANONICAL_TRAINING_BOOTSTRAP_SOURCE


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "clearml_formal_comparability_audit",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _raw_backend_value(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _raw_backend_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_raw_backend_value(item) for item in value]
    serializer = getattr(value, "to_dict", None)
    if callable(serializer):
        return serializer()
    return copy.deepcopy(value)


class _RawNamespace(SimpleNamespace):
    def to_dict(self) -> dict[str, object]:
        return {key: _raw_backend_value(value) for key, value in vars(self).items()}


class _Artifact:
    def __init__(self, value: object) -> None:
        self.value = value
        self.force_download_calls: list[bool] = []

    def get(self, *, force_download: bool = False) -> object:
        self.force_download_calls.append(force_download)
        assert force_download is True
        return self.value


class _RawArtifact:
    def __init__(self, key: str, artifact: _Artifact) -> None:
        self.key = key
        self.artifact = artifact

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "content": _raw_backend_value(self.artifact.value),
        }


class _Task:
    def __init__(
        self,
        task_id: str,
        *,
        status: str = "completed",
        parent: str = "",
        entry_point: str = "clearml_5090_bootstrap.py",
        script_diff: str = EVALUATION_BOOTSTRAP_SOURCE,
        parameters: dict[str, object] | None = None,
        artifacts: dict[str, _Artifact] | None = None,
        input_models: list[object] | None = None,
        output_models: list[object] | None = None,
        reload_statuses: list[str] | None = None,
        execution_queue_id: str = "",
        output_uri: str = "",
    ) -> None:
        self.id = task_id
        self.status = status
        self.parent = parent
        self.parameters = parameters or {}
        self.artifacts = artifacts or {}
        self.input_models = list(input_models or [])
        self.output_models = list(output_models or [])
        self.reload_statuses = list(reload_statuses or [])
        self.reload_calls = 0
        self._offline_mode = False
        self._reload_skip_flag = True
        self.tags: list[str] = []
        self.tag_calls: list[list[str]] = []
        self._data = _RawNamespace(
            id=task_id,
            status=status,
            parent=parent,
            last_worker=f"worker:{execution_queue_id}",
            script=_RawNamespace(
                repository="",
                working_dir=".",
                entry_point=entry_point,
                diff=script_diff,
            ),
            execution=_RawNamespace(
                queue=execution_queue_id,
                artifacts=[
                    _RawArtifact(name, artifact)
                    for name, artifact in self.artifacts.items()
                ],
            ),
            output=_RawNamespace(destination=output_uri),
            tags=self.tags,
            system_tags=[],
            hyperparams=self.parameters,
            configuration={},
        )

    @property
    def data(self) -> object:
        return self._data

    def _reload(self) -> object:
        self.reload_calls += 1
        if self.reload_statuses:
            self.status = self.reload_statuses.pop(0)
            self._data.status = self.status
        return self._data

    def get_parameters(self, cast: bool = False) -> dict[str, object]:
        assert cast is False
        return dict(self.parameters)

    def get_models(self) -> dict[str, list[object]]:
        return {
            "input": list(self.input_models),
            "output": list(self.output_models),
        }


class _OutputTask(_Task):
    def __init__(self, module) -> None:
        super().__init__(
            "f" * 32,
            status="in_progress",
            parent=LEADERBOARD_ID,
            entry_point=module.PRODUCER_ENTRY_POINT,
            script_diff=module._runtime_source(),
            parameters={
                "Args/training_controller_task_id": CONTROLLER_ID,
                "Args/training_provenance_task_id": PROVENANCE_ID,
                "Args/watcher_task_id": WATCHER_ID,
                "Args/leaderboard_task_id": LEADERBOARD_ID,
                "Args/poll_seconds": "0.25",
                "Args/timeout_hours": "1.0",
            },
            output_uri=module.FILES_SERVER_URI,
        )
        self.uploads: list[tuple[str, object, bool]] = []
        self.flush_calls = 0

    def set_tags(self, tags: list[str]) -> bool:
        self.tag_calls.append(list(tags))
        self.tags = list(tags)
        self._data.tags = list(tags)
        return True

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        self.uploads.append((name, artifact_object, wait_on_upload))
        artifact = _Artifact(artifact_object)
        self.artifacts[name] = artifact
        self._data.execution.artifacts.append(_RawArtifact(name, artifact))
        return True

    def flush(self, *, wait_for_uploads: bool) -> bool:
        assert wait_for_uploads is True
        self.flush_calls += 1
        return True


class _TaskClass:
    tasks: dict[str, _Task] = {}
    current: _Task | None = None
    batch_calls: list[tuple[str, ...]] = []

    @classmethod
    def get_task(cls, *, task_id: str) -> _Task:
        return cls.tasks[task_id]

    @classmethod
    def current_task(cls) -> _Task | None:
        return cls.current

    @classmethod
    def _query_tasks(
        cls,
        *,
        task_ids: list[str],
        fetch_only_first_page: bool,
        only_fields: list[str],
        search_hidden: bool,
    ) -> list[dict[str, object]]:
        assert fetch_only_first_page is True
        assert only_fields
        assert search_hidden is True
        ordered_ids = tuple(task_ids)
        cls.batch_calls.append(ordered_ids)
        available = dict(cls.tasks)
        if cls.current is not None:
            available[cls.current.id] = cls.current
        return [
            copy.deepcopy(available[task_id].data.to_dict()) for task_id in ordered_ids
        ]


def _pin_fixture_sources(module) -> None:
    module.ClearMLArtifact = lambda raw: raw.artifact
    module.CONTROLLER_SCRIPT_SHA256 = _digest(CONTROLLER_SOURCE)
    module.WATCHER_SCRIPT_SHA256 = _digest(WATCHER_SOURCE)
    module.LEADERBOARD_SCRIPT_SHA256 = _digest(LEADERBOARD_SOURCE)
    module.TRAINING_CONTROLLER_TASK_ID = CONTROLLER_ID
    module.EVALUATION_TEMPLATE_TASK_ID = TEMPLATE_ID
    module.LEGACY_PARENT_TASK_ID = LEGACY_PARENT_ID
    module.NO_DISTILLATION_PARENT_TASK_ID = NO_DIST_PARENT_ID
    module.RECOVERY_PARENT_TASK_ID = RECOVERY_PARENT_ID
    module.LEGACY_TRAINING_SCRIPT_SHA256 = _digest(LEGACY_TRAINING_BOOTSTRAP_SOURCE)
    module.CANONICAL_TRAINING_SCRIPT_SHA256 = _digest(
        CANONICAL_TRAINING_BOOTSTRAP_SOURCE
    )
    module.TRAINING_SCRIPT_SHA256 = module.CANONICAL_TRAINING_SCRIPT_SHA256
    module.EVALUATION_SCRIPT_SHA256 = _digest(EVALUATION_BOOTSTRAP_SOURCE)
    module.TRAINING_PROVENANCE_SCRIPT_SHA256 = _digest(PROVENANCE_SOURCE)


def test_active_script_pins_split_training_from_evaluation() -> None:
    module = _load_module()
    assert module.CONTROLLER_SCRIPT_SHA256 == (
        "22fe485f3eb999c85b464e260a4376bc81a577c15e6a429585e48df06ffd0678"
    )
    assert module.WATCHER_SCRIPT_SHA256 == (
        "be493eae42b0f50204dbbad3d2dd60671d8480226d1c0176bd912bf2a47f2624"
    )
    assert module.LEGACY_TRAINING_SCRIPT_SHA256 == (
        "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
    )
    assert module.EVALUATION_SCRIPT_SHA256 == (
        "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
    )
    assert module.TRAINING_PROVENANCE_SCRIPT_SHA256 == (
        "981abba51adf5f8c3b78960a14c2200a2fd68260e74273a52984de869825f77a"
    )


def test_retired_training_provenance_script_pin_is_rejected_without_fixture_patching() -> (
    None
):
    module = _load_module()

    with pytest.raises(RuntimeError, match="producer script SHA-256 mismatch"):
        module._require_training_provenance_script_sha256(
            "35e579e3b2740bcfc920d65a50f1c83550e0a14b159c236ee67041a68bb35995"
        )


def _initialization_audit(subject: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": "shared teacher initialization",
        "result": "pass",
        "subject": subject,
    }


def _manifest(module, *, seeded: bool = False) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for index, subject in enumerate(module.SUBJECT_ORDER, start=1):
        entry: dict[str, object] = {
            "index": index,
            "subject": subject,
            "kind": module.SUBJECT_KIND[subject],
            "training_task_id": f"{index:032x}",
            "training_predecessor_task_id": (
                module.SOURCE_REVISION_TARGET_PREDECESSOR_TASK_ID
                if subject in module.NEW_SOURCE_SUBJECTS
                else f"{index + 500:032x}"
            ),
            "model_id": f"{index + 100:032x}",
            "model_name": f"ResilientV2X {subject} final checkpoint",
            "model_url": (f"http://10.100.34.118:8081/models/{subject}_epoch_50.pth"),
            "checkpoint_filename": "epoch_50.pth",
            "checkpoint_sha256": f"{index:064x}",
            "checkpoint_size_bytes": 1000 + index,
            "common_teacher_initialization_audit_artifact": (
                "common_teacher_initialization_audit"
            ),
            "common_teacher_initialization_audit_sha256": (
                module._content_sha256(_initialization_audit(subject))
            ),
        }
        if seeded:
            entry.update(
                {
                    "training_seed": module.TRAINING_SEED,
                    "training_overlay_protocol_seed": module.TRAINING_SEED,
                }
            )
        entries.append(entry)
    payload: dict[str, object] = {
        "schema_version": 1,
        "manifest_type": "resilient_v2x_formal_1337_training_inputs",
        "protocol_id": module.PROTOCOL_ID,
        "sample_count": module.SAMPLE_COUNT,
        "delays_ms": list(module.DELAYS_MS),
        "conditions": list(module.CONDITIONS),
        "run_count": 12,
        "checkpoint_policy": module.CHECKPOINT_POLICY,
        "evaluation_release_semantics": module.RELEASE_SEMANTICS,
        "subject_order": list(module.SUBJECT_ORDER),
        "subject_count": len(module.SUBJECT_ORDER),
        "entries": entries,
    }
    if seeded:
        payload.update(
            {
                "training_seed": module.TRAINING_SEED,
                "training_overlay_protocol_seed": module.TRAINING_SEED,
            }
        )
    return module._sealed(payload)


def _summary(
    module,
    manifest: dict[str, object],
) -> dict[str, object]:
    results = [
        {
            "index": entry["index"],
            "experiment": entry["subject"],
            "task_id": entry["training_task_id"],
            "predecessor_task_id": entry["training_predecessor_task_id"],
            "model_id": entry["model_id"],
            "model_name": entry["model_name"],
            "model_url": entry["model_url"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "checkpoint_size_bytes": entry["checkpoint_size_bytes"],
        }
        for entry in manifest["entries"]
    ]
    return module._sealed(
        {
            "schema_version": 1,
            "summary_type": "resilient_v2x_post_main_sequential_training",
            "status": "completed",
            "controller_task_id": CONTROLLER_ID,
            "gate_task_id": GATE_ID,
            "template": {
                "task_id": module.SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID,
                "source_parameters": {
                    "Args/source_dataset_id": module.NEW_SOURCE_DATASET_ID,
                    "Args/source_archive_name": module.NEW_SOURCE_ARCHIVE_NAME,
                    "Args/source_archive_bytes": str(
                        module.NEW_SOURCE_ARCHIVE_BYTES
                    ),
                    "Args/source_archive_sha256": (
                        module.NEW_SOURCE_ARCHIVE_SHA256
                    ),
                    "Args/training_dataset_id": module.TRAINING_DATASET_ID,
                },
            },
            "experiment_order": list(module.SUBJECT_ORDER),
            "task_count": len(module.SUBJECT_ORDER),
            "results": results,
            "formal_1337_manifest_artifact": (module.TRAINING_MANIFEST_ARTIFACT),
            "formal_1337_evaluation_manifest": manifest,
        }
    )


def _progress(module, manifest: dict[str, object]) -> dict[str, object]:
    entries = {entry["subject"]: entry for entry in manifest["entries"]}
    source_identity = {
        "task_id": module.SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID,
        "script_sha256": "5" * 64,
    }
    target_identity = {
        "task_id": module.SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID,
        "script_sha256": "5" * 64,
    }
    transition = module._sealed(
        {
            "schema_version": 1,
            "source_controller_task_id": (
                module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
            ),
            "source_template_identity": source_identity,
            "target_template_identity": target_identity,
            "source_revision": {"tree_sha256": module.SOURCE_TREE_SHA256},
            "target_revision": {"tree_sha256": module.NEW_SOURCE_TREE_SHA256},
        }
    )
    module.SOURCE_REVISION_TRANSITION_SEAL_SHA256 = transition["seal_sha256"]
    recovery_observations = {
        subject: {
            "task_id": entries[subject]["training_task_id"],
            "predecessor_task_id": entries[subject]["training_predecessor_task_id"],
            "parent_controller_task_id": module._expected_training_parent(
                subject, CONTROLLER_ID
            ),
        }
        for subject in (
            "no_distillation",
            "ptf_none",
            "ptf_linear",
            "router_static",
        )
    }
    adopted_subjects = tuple(
        subject
        for subject in module.SUBJECT_ORDER
        if subject not in module.NEW_SOURCE_SUBJECTS
    )
    recovery = {
        "schema_version": 4,
        "mode": "failed_controller_immutable_fork",
        "source_controller_task_id": (
            module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
        ),
        "source_controller_status": "failed",
        "source_patch": "nested-teacher-config-consistency-v1",
        "source_progress_revision": 2,
        "source_progress_seal_sha256": "4" * 64,
        "source_progress_artifact_readback": {},
        "source_recovery_chain": {},
        "source_template_script_sha256": "5" * 64,
        "target_template_script_sha256": "5" * 64,
        "source_revision_transition": transition,
        "adopted_task_ids": {
            subject: entries[subject]["training_task_id"]
            for subject in adopted_subjects
        },
        "adopted_predecessor_task_ids": {
            subject: entries[subject]["training_predecessor_task_id"]
            for subject in adopted_subjects
        },
        "adopted_task_template_roles": {
            subject: "source" for subject in adopted_subjects
        },
        "adopted_task_binding_receipts": {
            subject: {"fixture": "sealed"} for subject in adopted_subjects
        },
        "recovery_target_adoptions": {
            "no_distillation": recovery_observations["no_distillation"]
        },
        "recovered_pending_target_children": {
            subject: recovery_observations[subject]
            for subject in ("ptf_none", "ptf_linear", "router_static")
        },
        "completion_validation_retries": {},
        "rerun_experiments": [],
        "rerun_predecessor_task_ids": {},
        "rerun_source_task_ids": {},
        "rerun_task_observations": {},
        "transition_replaced_source_tasks": {},
        "target_template_predecessor_task_ids": {
            subject: module.SOURCE_REVISION_TARGET_PREDECESSOR_TASK_ID
            for subject in module.NEW_SOURCE_SUBJECTS
        },
    }
    return module._sealed(
        {
            "schema_version": 1,
            "controller_task_id": CONTROLLER_ID,
            "controller_type": "resilient_v2x_post_main_sequential_training",
            "created_at": "fixture-created",
            "experiment_order": list(module.SUBJECT_ORDER),
            "gate_policy": "exact_task_must_be_completed_before_any_clone",
            "gate_task_id": GATE_ID,
            "max_parallel_training_tasks": 4,
            "recovery": recovery,
            "revision": 99,
            "teacher": {},
            "template": target_identity,
            "training_overlay_protocol_seed": module.TRAINING_SEED,
            "training_seed": module.TRAINING_SEED,
            "updated_at": "fixture-updated",
            "worker_queue": "GPU4-A100",
            "worker_queues": [
                "GPU4-A100",
                "GPU4-A100",
                "GPU4-V100",
                "GPU4-5090",
            ],
            "steps": [
                {
                    "index": index,
                    "experiment": subject,
                    "state": "completed",
                    "task_id": entries[subject]["training_task_id"],
                    "predecessor_task_id": entries[subject][
                        "training_predecessor_task_id"
                    ],
                    "result": {"task_id": entries[subject]["training_task_id"]},
                }
                for index, subject in enumerate(module.SUBJECT_ORDER, start=1)
            ],
        }
    )


def _training_provenance(
    module,
    manifest: dict[str, object],
    progress: dict[str, object],
    tasks: dict[str, _Task],
) -> dict[str, object]:
    source_by_sha = {
        module.LEGACY_TRAINING_SCRIPT_SHA256: LEGACY_TRAINING_BOOTSTRAP_SOURCE,
        module.CANONICAL_TRAINING_SCRIPT_SHA256: (CANONICAL_TRAINING_BOOTSTRAP_SOURCE),
    }
    script_by_subject = {
        subject: module._expected_training_script(subject)
        for subject in module.SUBJECT_ORDER
    }
    equivalence = module._verify_training_script_equivalence(
        source_by_sha, script_by_subject
    )
    progress_seals_by_controller = {
        CONTROLLER_ID: str(progress["seal_sha256"]),
        module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID: "1" * 64,
        RECOVERY_PARENT_ID: "7" * 64,
        LEGACY_PARENT_ID: "6" * 64,
    }
    progress_content_by_controller = {
        CONTROLLER_ID: module._content_sha256(progress),
        module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID: "2" * 64,
        RECOVERY_PARENT_ID: "8" * 64,
        LEGACY_PARENT_ID: "5" * 64,
    }
    bootstrap = {
        "equivalence_contract": "nested-teacher-membership-only-v1",
        "verified_from_actual_script_bytes": True,
        "legacy_script_sha256": equivalence["legacy_script_sha256"],
        "expanded_script_sha256": equivalence["canonical_script_sha256"],
        "legacy_nested_teacher_experiments": sorted(
            module.LEGACY_NESTED_TEACHER_EXPERIMENTS
        ),
        "expanded_nested_teacher_experiments": sorted(
            module.CANONICAL_NESTED_TEACHER_EXPERIMENTS
        ),
        "common_text_projection_sha256": "a" * 64,
        "common_ast_projection_sha256": "b" * 64,
        "only_text_difference": "NESTED_TEACHER_EXPERIMENTS assignment",
        "only_ast_difference": "NESTED_TEACHER_EXPERIMENTS frozenset members",
        "capacity_matched_hardware_contract": "capacity-matched-hardware-v1",
        "runtime_guard_evidence": "reviewed_completed_bootstrap_bytes",
        "tf32_override": "0",
        "homogeneous_gpu_count": 4,
        "allowed_compute_capabilities": [[7, 0], [8, 0], [12, 0]],
        "legacy_script_subjects": equivalence["legacy_script_subjects"],
        "expanded_script_subjects": [
            subject
            for subject in module.SUBJECT_ORDER
            if subject not in module.LEGACY_SCRIPT_ALLOWED_SUBJECTS
        ],
    }
    entries = {entry["subject"]: entry for entry in manifest["entries"]}
    records = []
    for index, subject in enumerate(module.SUBJECT_ORDER, start=1):
        entry = entries[subject]
        task = tasks[str(entry["training_task_id"])]
        contract = task.artifacts[module.RUN_CONTRACT_ARTIFACT].value
        final_contract = task.artifacts[module.FINAL_CHECKPOINT_ARTIFACT].value
        teacher_audit = task.artifacts[module.COMMON_TEACHER_AUDIT_ARTIFACT].value
        if subject == "support_residual":
            parent_binding = "recursive_source_controller_created"
            recovery_lineage = [
                {
                    "controller_task_id": CONTROLLER_ID,
                    "decision": "adopted_from_source_progress",
                    "progress_seal_sha256": progress_seals_by_controller[CONTROLLER_ID],
                    "source_controller_task_id": (
                        module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
                    ),
                },
                {
                    "controller_task_id": (
                        module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
                    ),
                    "decision": "adopted_from_source_progress",
                    "progress_seal_sha256": progress_seals_by_controller[
                        module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
                    ],
                    "source_controller_task_id": RECOVERY_PARENT_ID,
                },
                {
                    "controller_task_id": RECOVERY_PARENT_ID,
                    "decision": "adopted_from_source_progress",
                    "progress_seal_sha256": progress_seals_by_controller[
                        RECOVERY_PARENT_ID
                    ],
                    "source_controller_task_id": LEGACY_PARENT_ID,
                },
                {
                    "controller_task_id": LEGACY_PARENT_ID,
                    "decision": "controller_created",
                    "progress_seal_sha256": progress_seals_by_controller[
                        LEGACY_PARENT_ID
                    ],
                },
            ]
        elif subject in {"ptf_none", "ptf_linear", "router_static"}:
            parent_binding = "recovered_pending_target_children"
            recovery_lineage = [
                {
                    "controller_task_id": CONTROLLER_ID,
                    "decision": parent_binding,
                    "progress_seal_sha256": progress_seals_by_controller[CONTROLLER_ID],
                    "parent_controller_task_id": RECOVERY_PARENT_ID,
                }
            ]
        elif subject == "no_distillation":
            parent_binding = "recovery_target_adoptions"
            recovery_lineage = [
                {
                    "controller_task_id": CONTROLLER_ID,
                    "decision": parent_binding,
                    "progress_seal_sha256": progress_seals_by_controller[CONTROLLER_ID],
                    "parent_controller_task_id": NO_DIST_PARENT_ID,
                }
            ]
        elif subject not in module.NEW_SOURCE_SUBJECTS:
            parent_binding = "recursive_source_controller_created"
            recovery_lineage = [
                {
                    "controller_task_id": CONTROLLER_ID,
                    "decision": "adopted_from_source_progress",
                    "progress_seal_sha256": progress_seals_by_controller[CONTROLLER_ID],
                    "source_controller_task_id": (
                        module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
                    ),
                },
                {
                    "controller_task_id": (
                        module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
                    ),
                    "decision": "controller_created",
                    "progress_seal_sha256": progress_seals_by_controller[
                        module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
                    ],
                },
            ]
        else:
            parent_binding = "current_controller_created"
            recovery_lineage = [
                {
                    "controller_task_id": CONTROLLER_ID,
                    "decision": "controller_created",
                    "progress_seal_sha256": progress_seals_by_controller[CONTROLLER_ID],
                }
            ]
        records.append(
            {
                "index": index,
                "subject": subject,
                "training_task_id": entry["training_task_id"],
                "predecessor_task_id": entry["training_predecessor_task_id"],
                "parent_controller_task_id": module._expected_training_parent(
                    subject, CONTROLLER_ID
                ),
                "raw_bootstrap_script_sha256": module._expected_training_script(
                    subject
                ),
                "parent_binding": parent_binding,
                "recovery_lineage": recovery_lineage,
                "normalized_task_script_identity_sha256": "c" * 64,
                "sealed_progress_script_identity_sha256": "d" * 64,
                "script_equivalence_class": (
                    "legacy_nested_teacher_membership"
                    if subject in module.LEGACY_SCRIPT_ALLOWED_SUBJECTS
                    else "expanded_nested_teacher_membership"
                ),
                "manifest_model_id": entry["model_id"],
                "manifest_model_name": entry["model_name"],
                "manifest_model_url": entry["model_url"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "checkpoint_size_bytes": entry["checkpoint_size_bytes"],
                "source_revision_tree_sha256": module.SOURCE_BY_REVISION[
                    module.SOURCE_REVISION_BY_SUBJECT[subject]
                ]["tree_sha256"],
                "source_dataset_id": module.SOURCE_BY_REVISION[
                    module.SOURCE_REVISION_BY_SUBJECT[subject]
                ]["dataset_id"],
                "source_archive_name": module.SOURCE_BY_REVISION[
                    module.SOURCE_REVISION_BY_SUBJECT[subject]
                ]["archive_name"],
                "source_archive_bytes": module.SOURCE_BY_REVISION[
                    module.SOURCE_REVISION_BY_SUBJECT[subject]
                ]["archive_size_bytes"],
                "source_archive_sha256": module.SOURCE_BY_REVISION[
                    module.SOURCE_REVISION_BY_SUBJECT[subject]
                ]["archive_sha256"],
                "observed_parameter_keys": sorted(task.parameters),
                "observed_parameters_sha256": "e" * 64,
                "docker_command_sha256": module.DOCKER_COMMAND_SHA256,
                "run_contract_content_sha256": module._content_sha256(contract),
                "final_checkpoint_contract_content_sha256": (
                    module._content_sha256(final_contract)
                ),
                "common_teacher_initialization_audit_content_sha256": (
                    module._content_sha256(teacher_audit)
                ),
                "execution_queue_id": task.data.execution.queue,
                "last_worker": task.data.last_worker,
                "raw_authority_sha256": module._content_sha256(
                    module._raw_task_authority(
                        task.data, context=f"fixture training {subject}"
                    )
                ),
            }
        )
    recovery = progress["recovery"]
    source_equivalence = module._sealed(
        {
            "schema_version": 1,
            "contract": "fixture-source-revision-equivalence",
            "source_revisions": module.SOURCE_REVISION_CERTIFICATE_BY_TREE,
        }
    )
    module.SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = source_equivalence[
        "seal_sha256"
    ]
    source_subject_map = module._sealed(
        {
            "schema_version": 1,
            "contract": "fixture-source-revision-subject-map",
            "source_revision_by_subject": {
                subject: module.SOURCE_BY_REVISION[
                    module.SOURCE_REVISION_BY_SUBJECT[subject]
                ]["tree_sha256"]
                for subject in module.SUBJECT_ORDER
            },
        }
    )
    module.SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = source_subject_map[
        "seal_sha256"
    ]
    source_parent_subjects = [
        subject
        for subject in module.SUBJECT_ORDER
        if subject not in module.NEW_SOURCE_SUBJECTS
        and subject
        not in {
            "support_residual",
            "no_distillation",
            "ptf_none",
            "ptf_linear",
            "router_static",
        }
    ]
    return module._sealed(
        {
            "schema_version": 2,
            "document_type": ("resilient_v2x_formal_training_provenance_equivalence"),
            "protocol_id": module.PROTOCOL_ID,
            "passed": True,
            "controller_task_id": CONTROLLER_ID,
            "controller_status": "completed",
            "controller_raw_authority_sha256": "2" * 64,
            "subject_order": list(module.SUBJECT_ORDER),
            "subject_count": len(module.SUBJECT_ORDER),
            "all_training_tasks_completed": True,
            "authoritative_metadata_read": "single_batch_per_snapshot",
            "progress_artifact": module.TRAINING_PROGRESS_ARTIFACT,
            "progress_seal_sha256": progress["seal_sha256"],
            "progress_content_sha256": module._content_sha256(progress),
            "formal_training_manifest_artifact": (module.TRAINING_MANIFEST_ARTIFACT),
            "formal_training_manifest_seal_sha256": manifest["seal_sha256"],
            "formal_training_manifest_content_sha256": (
                module._content_sha256(manifest)
            ),
            "recovery_contract_sha256": module._content_sha256(recovery),
            "recursive_progress_chain": [
                {
                    "task_id": task_id,
                    "role": (
                        "final_controller"
                        if task_id == CONTROLLER_ID
                        else "recursive_recovery_source_controller"
                    ),
                    "progress_artifact": module.TRAINING_PROGRESS_ARTIFACT,
                    "progress_revision": 1,
                    "progress_seal_sha256": progress_seals_by_controller[task_id],
                    "progress_content_sha256": progress_content_by_controller[task_id],
                    "template_script_identity_sha256": "3" * 64,
                }
                for task_id in (
                    CONTROLLER_ID,
                    module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
                    RECOVERY_PARENT_ID,
                    LEGACY_PARENT_ID,
                )
            ],
            "bootstrap_equivalence": bootstrap,
            "source_revision_equivalence": source_equivalence,
            "source_revision_equivalence_seal_sha256": source_equivalence[
                "seal_sha256"
            ],
            "source_revision_subject_map": source_subject_map,
            "source_revision_subject_map_seal_sha256": source_subject_map[
                "seal_sha256"
            ],
            "run_contract_equivalence": {
                "source_fields_vary_only_by_sealed_subject_revision_map": True,
                "source_revision_equivalence_seal_sha256": source_equivalence[
                    "seal_sha256"
                ],
                "source_revision_subject_map_seal_sha256": source_subject_map[
                    "seal_sha256"
                ],
                "source_revision_counts": {
                    module.SOURCE_TREE_SHA256: 21,
                    module.NEW_SOURCE_TREE_SHA256: 5,
                },
                "training_dataset_id": module.TRAINING_DATASET_ID,
                "native_bundle_sha256": module.NATIVE_BUNDLE_SHA256,
                "build_manifest_sha256": module.BUILD_MANIFEST_SHA256,
                "teacher_checkpoint_sha256": module.TEACHER_CHECKPOINT_SHA256,
                "training_seed": module.TRAINING_SEED,
                "global_batch_size": 8,
                "precision": "FP32",
                "max_epochs": 50,
                "val_interval": 10,
                "amp": False,
            },
            "capacity_matched_hardware": {
                "contract": "capacity-matched-hardware-v1",
                "docker_command_sha256": module.DOCKER_COMMAND_SHA256,
                "docker_command_identical_across_26_tasks": True,
                "gpu_count": 4,
                "homogeneous_per_task": True,
                "allowed_compute_capabilities": [[7, 0], [8, 0], [12, 0]],
                "gpu_class_may_vary_across_tasks": True,
                "tf32_override": "0",
                "precision": "FP32",
                "global_batch_size": 8,
                "runtime_guard_evidence": "reviewed_completed_bootstrap_bytes",
            },
            "recovery_parent_controllers": [
                {
                    "task_id": task_id,
                    "terminal_status": "failed",
                    "roles": (
                        [
                            "recursive_recovery_source",
                            "actual_training_parent",
                        ]
                        if task_id
                        in {
                            module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
                            LEGACY_PARENT_ID,
                            RECOVERY_PARENT_ID,
                        }
                        else ["actual_training_parent"]
                    ),
                    "actual_parent_subjects": subjects,
                    "progress_artifact": (
                        module.TRAINING_PROGRESS_ARTIFACT
                        if task_id
                        in {
                            module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
                            LEGACY_PARENT_ID,
                            RECOVERY_PARENT_ID,
                        }
                        else None
                    ),
                    "progress_seal_sha256": (
                        progress_seals_by_controller[task_id]
                        if task_id
                        in {
                            module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
                            LEGACY_PARENT_ID,
                            RECOVERY_PARENT_ID,
                        }
                        else None
                    ),
                    "progress_content_sha256": (
                        progress_content_by_controller[task_id]
                        if task_id
                        in {
                            module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
                            LEGACY_PARENT_ID,
                            RECOVERY_PARENT_ID,
                        }
                        else None
                    ),
                    "raw_authority_sha256": "9" * 64,
                }
                for task_id, subjects in (
                    (LEGACY_PARENT_ID, ["support_residual"]),
                    (NO_DIST_PARENT_ID, ["no_distillation"]),
                    (
                        RECOVERY_PARENT_ID,
                        ["ptf_none", "ptf_linear", "router_static"],
                    ),
                    (
                        module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
                        source_parent_subjects,
                    ),
                )
            ],
            "training_tasks": records,
        }
    )


def _plan(module, provenance: dict[str, object]) -> dict[str, object]:
    queues = ("GPU4-A100", "GPU4-V100", "GPU4-5090")
    entries = [
        {
            "subject": subject,
            "evaluation_task_id": f"{index + 1000:032x}",
            "queue": queues[(index - 1) % len(queues)],
        }
        for index, subject in enumerate(module.SUBJECT_ORDER, start=1)
    ]
    return module._sealed(
        {
            "schema_version": 2,
            "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
            "training_controller_task_id": CONTROLLER_ID,
            "training_provenance_task_id": PROVENANCE_ID,
            "training_provenance_seal_sha256": provenance["seal_sha256"],
            "source_revision_equivalence": provenance[
                "source_revision_equivalence"
            ],
            "source_revision_equivalence_seal_sha256": provenance[
                "source_revision_equivalence_seal_sha256"
            ],
            "source_revision_subject_map": provenance[
                "source_revision_subject_map"
            ],
            "source_revision_subject_map_seal_sha256": provenance[
                "source_revision_subject_map_seal_sha256"
            ],
            "evaluation_source_revision_tree_sha256": module.SOURCE_TREE_SHA256,
            "evaluation_source_revision": (
                module.SOURCE_REVISION_CERTIFICATE_BY_TREE[
                    module.SOURCE_TREE_SHA256
                ]
            ),
            "evaluation_template_task_id": TEMPLATE_ID,
            "protocol_id": module.PROTOCOL_ID,
            "sample_count": module.SAMPLE_COUNT,
            "delays_ms": list(module.DELAYS_MS),
            "conditions": list(module.CONDITIONS),
            "run_count": 12,
            "subject_order": list(module.SUBJECT_ORDER),
            "entries": entries,
        }
    )


def _subject_base(module, subject: str) -> float:
    if subject == module.PRIMARY_SUBJECT:
        return 85.0
    if subject == "support_residual":
        return 90.0
    if subject in module.BASELINE_SUBJECTS:
        return 50.0 + module.BASELINE_SUBJECTS.index(subject)
    return 75.0


def _metrics(module, subject: str, checkpoint_sha256: str) -> dict[str, object]:
    metric_offsets = {
        "resilient_v2x/car_bev_ap_r40_0.50": 5.0,
        "resilient_v2x/car_bev_ap_r40_0.70": 0.0,
        "resilient_v2x/car_3d_ap_r40_0.50": -3.0,
        "resilient_v2x/car_3d_ap_r40_0.70": -10.0,
    }
    condition_penalties = {"Full": 0.0, "L-Fail": 10.0, "C-Fail": 5.0}
    base = _subject_base(module, subject)
    runs = []
    for delay in module.DELAYS_MS:
        for condition in module.CONDITIONS:
            values = {
                key: base + offset - delay / 100.0 - condition_penalties[condition]
                for key, offset in metric_offsets.items()
            }
            values.update(module.COUNT_METRICS)
            runs.append(
                {
                    "condition_id": (
                        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
                    ),
                    "delay_ms": delay,
                    "condition": condition,
                    "metrics": values,
                    "predictions": (f"/evidence/{subject}/{delay}/{condition}.json"),
                    "prediction_sha256": "2" * 64,
                    "prediction_content_sha256": "3" * 64,
                    "sample_count": module.SAMPLE_COUNT,
                    "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
                    "ground_truth_count": module.GROUND_TRUTH_COUNT,
                    "unsupported_sample_count": (module.UNSUPPORTED_SAMPLE_COUNT),
                }
            )
    return {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": 12,
        "baseline": subject,
        "protocol_id": module.PROTOCOL_ID,
        "checkpoint": f"/checkpoints/{subject}_epoch_50.pth",
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_content_sha256": module.MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": (module.OVERLAY_INDEX_CONTENT_SHA256),
        "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
        "expected_sample_count": module.SAMPLE_COUNT,
        "expected_ground_truth_count": module.GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": module.UNSUPPORTED_SAMPLE_COUNT,
        "runs": runs,
    }


def _training_task(
    module,
    entry: dict[str, object],
    *,
    manifest_seeded: bool,
) -> _Task:
    subject = str(entry["subject"])
    task_id = str(entry["training_task_id"])
    source = module.SOURCE_BY_REVISION[module.SOURCE_REVISION_BY_SUBJECT[subject]]
    run_contract = {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": subject,
        "source_dataset_id": source["dataset_id"],
        "training_dataset_id": module.TRAINING_DATASET_ID,
        "source_archive": {
            "name": source["archive_name"],
            "size_bytes": source["archive_size_bytes"],
            "sha256": source["archive_sha256"],
        },
        "predecessor_task_id": entry["training_predecessor_task_id"],
        "gpus": 4,
        "global_batch_size": 8,
        "ddp_processes": 4,
        "max_epochs": 50,
        "seed": module.TRAINING_SEED,
        "amp": False,
        "precision": "FP32",
        "val_interval": 10,
        "condition_evaluation": False,
    }
    if manifest_seeded:
        run_contract.update(
            {
                "training_seed": module.TRAINING_SEED,
                "training_overlay_protocol_seed": module.TRAINING_SEED,
            }
        )
    model = SimpleNamespace(
        id=entry["model_id"],
        task=task_id,
        name=entry["model_name"],
        url=entry["model_url"],
    )
    artifacts = {
        module.RUN_CONTRACT_ARTIFACT: _Artifact(run_contract),
        module.FINAL_CHECKPOINT_ARTIFACT: _Artifact(
            {
                "model_id": entry["model_id"],
                "name": entry["model_name"],
                "url": entry["model_url"],
                "filename": "epoch_50.pth",
                "size_bytes": entry["checkpoint_size_bytes"],
                "sha256": entry["checkpoint_sha256"],
            }
        ),
        module.BEST_CHECKPOINT_ARTIFACT: _Artifact({"subject": subject}),
        module.COMMON_TEACHER_AUDIT_ARTIFACT: _Artifact(_initialization_audit(subject)),
    }
    if subject in module.BASELINE_SUBJECTS:
        artifacts.update(
            {
                module.BASELINE_DRY_RUN_PLAN_ARTIFACT: _Artifact({"subject": subject}),
                module.BASELINE_RESOLVED_CONFIG_ARTIFACT: _Artifact(
                    {"subject": subject}
                ),
            }
        )
    return _Task(
        task_id,
        parent=module._expected_training_parent(subject, CONTROLLER_ID),
        script_diff=(
            LEGACY_TRAINING_BOOTSTRAP_SOURCE
            if subject in module.LEGACY_SCRIPT_ALLOWED_SUBJECTS
            else CANONICAL_TRAINING_BOOTSTRAP_SOURCE
        ),
        parameters={
            "Args/experiment_from_task": subject,
            "Args/source_dataset_id": source["dataset_id"],
            "Args/source_archive_name": source["archive_name"],
            "Args/source_archive_bytes": str(source["archive_size_bytes"]),
            "Args/source_archive_sha256": source["archive_sha256"],
            "Args/training_dataset_id": module.TRAINING_DATASET_ID,
            "Args/predecessor_task_id": entry["training_predecessor_task_id"],
            "Args/stage": "all",
            "Args/max_epochs": "50",
            "Args/gpus": "4",
            "Args/amp": "False",
        },
        artifacts=artifacts,
        output_models=[model],
        execution_queue_id=QUEUE_IDS["GPU4-A100"],
    )


def _evaluation_task(
    module,
    training: dict[str, object],
    evaluation: dict[str, object],
) -> _Task:
    subject = str(training["subject"])
    return _Task(
        str(evaluation["evaluation_task_id"]),
        parent=CONTROLLER_ID,
        script_diff=EVALUATION_BOOTSTRAP_SOURCE,
        parameters={
            "Args/stage": "baseline_validate",
            "Args/controlled_baseline": subject,
            "Args/controlled_baseline_task_id": training["training_task_id"],
            "Args/controlled_baseline_model_id": training["model_id"],
            "Args/controlled_baseline_checkpoint_sha256": training["checkpoint_sha256"],
            "Args/predecessor_task_id": training["training_task_id"],
            "Args/source_dataset_id": module.SOURCE_DATASET_ID,
            "Args/source_archive_name": module.SOURCE_ARCHIVE_NAME,
            "Args/source_archive_bytes": str(module.SOURCE_ARCHIVE_BYTES),
            "Args/source_archive_sha256": module.SOURCE_ARCHIVE_SHA256,
            "Args/training_dataset_id": module.TRAINING_DATASET_ID,
            "Args/gpus": "4",
            "Args/max_epochs": "50",
            "Args/amp": "False",
        },
        artifacts={
            module.RUN_CONTRACT_ARTIFACT: _Artifact({"subject": subject}),
            module.CONTROLLED_EVALUATION_PLAN_ARTIFACT: _Artifact({"subject": subject}),
            module.METRICS_ARTIFACT: _Artifact(
                _metrics(
                    module,
                    subject,
                    str(training["checkpoint_sha256"]),
                )
            ),
            module.CONTROLLED_EVIDENCE_ARTIFACT: _Artifact({"subject": subject}),
        },
        input_models=[SimpleNamespace(id=training["model_id"])],
        execution_queue_id=QUEUE_IDS[str(evaluation["queue"])],
    )


def _leaderboard(
    module,
    manifest: dict[str, object],
    plan: dict[str, object],
    tasks: dict[str, _Task],
) -> dict[str, object]:
    runs_by_subject = {}
    results = []
    summaries = {}
    for index, (training, evaluation, subject) in enumerate(
        zip(
            manifest["entries"],
            plan["entries"],
            module.SUBJECT_ORDER,
            strict=True,
        ),
        start=1,
    ):
        metrics = (
            tasks[str(evaluation["evaluation_task_id"])]
            .artifacts[module.METRICS_ARTIFACT]
            .value
        )
        runs = module._validate_metrics(
            metrics,
            subject=subject,
            checkpoint_sha256=str(training["checkpoint_sha256"]),
        )
        runs_by_subject[subject] = runs
        subject_summaries = {
            metric_key: module._metric_summary(runs, metric_key)
            for metric_key in module.AP_METRIC_KEYS
        }
        summaries[subject] = subject_summaries
        results.append(
            {
                "index": index,
                "subject": subject,
                "kind": module.SUBJECT_KIND[subject],
                "training_task_id": training["training_task_id"],
                "training_model_id": training["model_id"],
                "training_checkpoint_sha256": training["checkpoint_sha256"],
                "source_revision_tree_sha256": module.SOURCE_BY_REVISION[
                    module.SOURCE_REVISION_BY_SUBJECT[subject]
                ]["tree_sha256"],
                "evaluation_task_id": evaluation["evaluation_task_id"],
                "metrics": subject_summaries,
            }
        )
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
            "training_controller_task_id": CONTROLLER_ID,
            "watcher_task_id": WATCHER_ID,
            "training_manifest_seal_sha256": manifest["seal_sha256"],
            "evaluation_plan_seal_sha256": plan["seal_sha256"],
            "training_provenance_task_id": plan[
                "training_provenance_task_id"
            ],
            "training_provenance_seal_sha256": plan[
                "training_provenance_seal_sha256"
            ],
            "source_revision_equivalence": plan[
                "source_revision_equivalence"
            ],
            "source_revision_equivalence_seal_sha256": plan[
                "source_revision_equivalence_seal_sha256"
            ],
            "source_revision_subject_map": plan[
                "source_revision_subject_map"
            ],
            "source_revision_subject_map_seal_sha256": plan[
                "source_revision_subject_map_seal_sha256"
            ],
            "evaluation_source_revision_tree_sha256": plan[
                "evaluation_source_revision_tree_sha256"
            ],
            "evaluation_source_revision": plan["evaluation_source_revision"],
            "subject_order": list(module.SUBJECT_ORDER),
            "subject_count": len(module.SUBJECT_ORDER),
            "baseline_subjects": list(module.BASELINE_SUBJECTS),
            "baseline_count": len(module.BASELINE_SUBJECTS),
            "metric_keys": list(module.AP_METRIC_KEYS),
            "results": results,
            "leadership": module._expected_leadership(
                runs_by_subject,
                summaries,
            ),
        }
    )


def _world(
    module,
    *,
    controller_status: str = "completed",
    watcher_status: str = "completed",
    leaderboard_status: str = "completed",
    controller_reload_statuses: list[str] | None = None,
    watcher_reload_statuses: list[str] | None = None,
    leaderboard_reload_statuses: list[str] | None = None,
):
    _pin_fixture_sources(module)
    manifest = _manifest(module, seeded=False)
    summary = _summary(module, manifest)
    progress = _progress(module, manifest)
    tasks: dict[str, _Task] = {}
    for entry in manifest["entries"]:
        training = _training_task(
            module,
            entry,
            manifest_seeded=False,
        )
        tasks[training.id] = training
    provenance_payload = _training_provenance(module, manifest, progress, tasks)
    plan = _plan(module, provenance_payload)
    for training, evaluation in zip(
        manifest["entries"],
        plan["entries"],
        strict=True,
    ):
        task = _evaluation_task(module, training, evaluation)
        tasks[task.id] = task
    leaderboard = _leaderboard(module, manifest, plan, tasks)
    controller = _Task(
        CONTROLLER_ID,
        status=controller_status,
        entry_point="clearml_5090_training_controller.py",
        script_diff=CONTROLLER_SOURCE,
        parameters={
            "Args/gate_task_id": GATE_ID,
            "Args/template_task_id": (
                module.SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID
            ),
        },
        artifacts={
            module.TRAINING_PROGRESS_ARTIFACT: _Artifact(progress),
            module.TRAINING_MANIFEST_ARTIFACT: _Artifact(manifest),
            module.TRAINING_SUMMARY_ARTIFACT: _Artifact(summary),
        },
        reload_statuses=controller_reload_statuses,
    )
    provenance_task = _Task(
        PROVENANCE_ID,
        parent=CONTROLLER_ID,
        entry_point=module.PROVENANCE_PRODUCER_ENTRY_POINT,
        script_diff=PROVENANCE_SOURCE,
        parameters={
            "Args/training_controller_task_id": CONTROLLER_ID,
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
        },
        artifacts={module.TRAINING_PROVENANCE_ARTIFACT: _Artifact(provenance_payload)},
    )
    provenance_task.tags = list(module.PROVENANCE_FORMAL_TAGS)
    provenance_task.data.tags = list(module.PROVENANCE_FORMAL_TAGS)
    watcher = _Task(
        WATCHER_ID,
        status=watcher_status,
        parent=PROVENANCE_ID,
        entry_point="clearml_1337_dependency_watcher.py",
        script_diff=WATCHER_SOURCE,
        parameters={
            "Args/training_controller_task_id": CONTROLLER_ID,
            "Args/training_provenance_task_id": PROVENANCE_ID,
            "Args/evaluation_template_task_id": TEMPLATE_ID,
            "Args/expected_training_provenance_script_sha256": (
                module.TRAINING_PROVENANCE_SCRIPT_SHA256
            ),
            "Args/expected_eval_script_sha256": module.EVALUATION_SCRIPT_SHA256,
            "Args/expected_source_dataset_id": module.SOURCE_DATASET_ID,
            "Args/expected_source_archive_sha256": (module.SOURCE_ARCHIVE_SHA256),
            "Args/expected_training_dataset_id": (module.TRAINING_DATASET_ID),
        },
        artifacts={module.EVALUATION_PLAN_ARTIFACT: _Artifact(plan)},
        reload_statuses=watcher_reload_statuses,
    )
    leaderboard_task = _Task(
        LEADERBOARD_ID,
        status=leaderboard_status,
        parent=WATCHER_ID,
        entry_point="clearml_1337_leaderboard.py",
        script_diff=LEADERBOARD_SOURCE,
        parameters={
            "Args/training_controller_task_id": CONTROLLER_ID,
            "Args/watcher_task_id": WATCHER_ID,
        },
        artifacts={module.LEADERBOARD_ARTIFACT: _Artifact(leaderboard)},
        reload_statuses=leaderboard_reload_statuses,
    )
    tasks.update(
        {
            CONTROLLER_ID: controller,
            PROVENANCE_ID: provenance_task,
            WATCHER_ID: watcher,
            LEADERBOARD_ID: leaderboard_task,
        }
    )
    output = _OutputTask(module)
    _TaskClass.tasks = tasks
    _TaskClass.current = output
    _TaskClass.batch_calls = []
    return SimpleNamespace(
        manifest=manifest,
        summary=summary,
        progress=progress,
        provenance_payload=provenance_payload,
        plan=plan,
        leaderboard=leaderboard,
        controller=controller,
        provenance_task=provenance_task,
        watcher=watcher,
        leaderboard_task=leaderboard_task,
        output=output,
        tasks=tasks,
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        training_controller_task_id=CONTROLLER_ID,
        training_provenance_task_id=PROVENANCE_ID,
        watcher_task_id=WATCHER_ID,
        leaderboard_task_id=LEADERBOARD_ID,
        poll_seconds=0.25,
        timeout_hours=1.0,
    )


def _run(module, world):
    return module.run(
        _args(),
        task_class=_TaskClass,
        output_task=world.output,
        monotonic_clock=lambda: 0.0,
        sleeper=lambda _: None,
        queue_name_resolver=QUEUE_NAMES.__getitem__,
    )


def _server_artifact(task: _Task, name: str) -> _Artifact:
    matches = [raw.artifact for raw in task.data.execution.artifacts if raw.key == name]
    assert len(matches) == 1
    return matches[0]


def _install_server_artifact(task: _Task, name: str, artifact: _Artifact) -> None:
    task.artifacts[name] = artifact
    task.data.execution.artifacts.append(_RawArtifact(name, artifact))


def _install_server_tags(task: _Task, tags: list[str]) -> None:
    task.tags = list(tags)
    task.data.tags = list(tags)


def _reseal_in_place(module, payload: dict[str, object]) -> None:
    payload.pop("seal_sha256", None)
    payload.update(module._sealed(payload))


def _rebind_progress_dependency_chain(module, world) -> None:
    root_lineage = world.provenance_payload["recursive_progress_chain"][0]
    root_lineage["progress_seal_sha256"] = world.progress["seal_sha256"]
    root_lineage["progress_content_sha256"] = module._content_sha256(world.progress)
    for record in world.provenance_payload["training_tasks"]:
        record["recovery_lineage"][0]["progress_seal_sha256"] = world.progress[
            "seal_sha256"
        ]
    world.provenance_payload["progress_seal_sha256"] = world.progress[
        "seal_sha256"
    ]
    world.provenance_payload["progress_content_sha256"] = module._content_sha256(
        world.progress
    )
    world.provenance_payload["recovery_contract_sha256"] = module._content_sha256(
        world.progress["recovery"]
    )
    _reseal_in_place(module, world.provenance_payload)
    world.plan["training_provenance_seal_sha256"] = world.provenance_payload[
        "seal_sha256"
    ]
    _reseal_in_place(module, world.plan)
    world.leaderboard["training_provenance_seal_sha256"] = (
        world.provenance_payload["seal_sha256"]
    )
    world.leaderboard["evaluation_plan_seal_sha256"] = world.plan["seal_sha256"]
    _reseal_in_place(module, world.leaderboard)


@pytest.mark.parametrize(
    "mutation",
    ["parent_swap", "third_sha", "legacy_wrong_subject", "provenance_task_swap"],
)
def test_recovery_chain_rejects_parent_script_and_task_swaps(mutation: str) -> None:
    module = _load_module()
    world = _world(module)
    entries = {entry["subject"]: entry for entry in world.manifest["entries"]}
    support = world.tasks[entries["support_residual"]["training_task_id"]]
    coformer = world.tasks[entries["coformernet"]["training_task_id"]]
    if mutation == "parent_swap":
        support.data.parent = RECOVERY_PARENT_ID
    elif mutation == "third_sha":
        coformer.data.script.diff = CANONICAL_TRAINING_BOOTSTRAP_SOURCE + "\n# third\n"
    elif mutation == "legacy_wrong_subject":
        coformer.data.script.diff = LEGACY_TRAINING_BOOTSTRAP_SOURCE
    else:
        records = world.provenance_payload["training_tasks"]
        assert isinstance(records, list)
        first = records[0]
        second = records[1]
        assert isinstance(first, dict) and isinstance(second, dict)
        first["training_task_id"], second["training_task_id"] = (
            second["training_task_id"],
            first["training_task_id"],
        )
        _reseal_in_place(module, world.provenance_payload)

    with pytest.raises(RuntimeError):
        _run(module, world)


def test_recovery_progress_seal_and_parent_observation_are_authoritative() -> None:
    module = _load_module()
    world = _world(module)
    recovery = world.progress["recovery"]
    assert isinstance(recovery, dict)
    observations = recovery["recovered_pending_target_children"]
    assert isinstance(observations, dict)
    observation = observations["ptf_none"]
    assert isinstance(observation, dict)
    observation["parent_controller_task_id"] = LEGACY_PARENT_ID
    _reseal_in_place(module, world.progress)
    _rebind_progress_dependency_chain(module, world)

    with pytest.raises(RuntimeError, match="recovery observation ptf_none drifted"):
        _run(module, world)


def test_recovery_progress_rejects_removed_ptf_observation_lineage() -> None:
    module = _load_module()
    world = _world(module)
    recovery = world.progress["recovery"]
    observations = recovery["recovered_pending_target_children"]
    observations.pop("ptf_none")
    _reseal_in_place(module, world.progress)
    _rebind_progress_dependency_chain(module, world)

    with pytest.raises(RuntimeError, match="ptf_none root lineage role drifted"):
        _run(module, world)


def test_recovery_progress_rejects_target_predecessor_splice() -> None:
    module = _load_module()
    world = _world(module)
    recovery = world.progress["recovery"]
    target_predecessors = recovery["target_template_predecessor_task_ids"]
    target_predecessors["where2comm"] = "9" * 32
    _reseal_in_place(module, world.progress)

    with pytest.raises(
        RuntimeError,
        match="target-template predecessor where2comm drifted",
    ):
        _run(module, world)


@pytest.mark.parametrize(
    "mutation",
    [
        "parent",
        "entrypoint",
        "source",
        "args",
        "legacy_args_30_72",
        "tags",
        "artifact",
        "artifact_extra_key",
        "deep_lineage_seal",
        "recursive_chain_reorder",
        "legacy_subject_duplicate",
        "legacy_subject_missing",
        "inventory",
    ],
)
def test_training_provenance_producer_contract_is_fail_closed(mutation: str) -> None:
    module = _load_module()
    world = _world(module)
    task = world.provenance_task
    if mutation == "parent":
        task.data.parent = WATCHER_ID
    elif mutation == "entrypoint":
        task.data.script.entry_point = "attacker.py"
    elif mutation == "source":
        task.data.script.diff += "\n# drift"
    elif mutation == "args":
        task.parameters["Args/extra"] = "1"
        task.data.hyperparams = task.parameters
    elif mutation == "legacy_args_30_72":
        task.parameters["Args/poll_seconds"] = "30.0"
        task.parameters["Args/timeout_hours"] = "72.0"
        task.data.hyperparams = task.parameters
    elif mutation == "tags":
        task.data.tags = [*module.PROVENANCE_FORMAL_TAGS, "attacker"]
    elif mutation in {"artifact", "artifact_extra_key"}:
        world.provenance_payload["passed"] = False
        if mutation == "artifact_extra_key":
            world.provenance_payload["attacker_extra"] = True
        _reseal_in_place(module, world.provenance_payload)
    elif mutation == "deep_lineage_seal":
        support = next(
            record
            for record in world.provenance_payload["training_tasks"]
            if record["subject"] == "support_residual"
        )
        support["recovery_lineage"][1]["progress_seal_sha256"] = "f" * 64
        _reseal_in_place(module, world.provenance_payload)
    elif mutation == "recursive_chain_reorder":
        chain = world.provenance_payload["recursive_progress_chain"]
        chain[1], chain[2] = chain[2], chain[1]
        _reseal_in_place(module, world.provenance_payload)
    elif mutation == "legacy_subject_duplicate":
        bootstrap = world.provenance_payload["bootstrap_equivalence"]
        bootstrap["legacy_script_subjects"].append("support_residual")
        _reseal_in_place(module, world.provenance_payload)
    elif mutation == "legacy_subject_missing":
        bootstrap = world.provenance_payload["bootstrap_equivalence"]
        bootstrap["legacy_script_subjects"].pop()
        _reseal_in_place(module, world.provenance_payload)
    else:
        _install_server_artifact(task, "attacker_extra", _Artifact({"x": 1}))

    with pytest.raises((RuntimeError, ValueError)):
        _run(module, world)


def test_training_provenance_tags_are_an_unordered_unique_set() -> None:
    module = _load_module()
    world = _world(module)
    reordered = list(reversed(module.PROVENANCE_FORMAL_TAGS))
    world.provenance_task.data.tags = reordered

    payload = _run(module, world)

    assert payload["passed"] is True


@pytest.mark.parametrize(
    ("invalid_state", "message"),
    [
        ("duplicate", "duplicate tags"),
        ("unknown", "formal tags drifted"),
    ],
)
def test_training_provenance_tags_reject_duplicates_and_unknown_labels(
    invalid_state: str,
    message: str,
) -> None:
    module = _load_module()
    world = _world(module)
    tags = list(module.PROVENANCE_FORMAL_TAGS)
    if invalid_state == "duplicate":
        tags.append(tags[0])
    else:
        tags[-1] = "unknown-third-state"
    world.provenance_task.data.tags = tags

    with pytest.raises(RuntimeError, match=message):
        _run(module, world)


def test_nested_teacher_usage_closure_is_recomputed_from_actual_bytes() -> None:
    module = _load_module()
    legacy = LEGACY_TRAINING_BOOTSTRAP_SOURCE
    canonical = CANONICAL_TRAINING_BOOTSTRAP_SOURCE.replace(
        "spec.name in NESTED_TEACHER_EXPERIMENTS",
        "spec.name not in NESTED_TEACHER_EXPERIMENTS",
        1,
    )
    module.LEGACY_TRAINING_SCRIPT_SHA256 = _digest(legacy)
    module.CANONICAL_TRAINING_SCRIPT_SHA256 = _digest(canonical)
    with pytest.raises(RuntimeError, match="semantic drifted"):
        module._verify_training_script_equivalence(
            {
                module.LEGACY_TRAINING_SCRIPT_SHA256: legacy,
                module.CANONICAL_TRAINING_SCRIPT_SHA256: canonical,
            },
            {
                subject: (
                    module.LEGACY_TRAINING_SCRIPT_SHA256
                    if subject in module.LEGACY_SCRIPT_ALLOWED_SUBJECTS
                    else module.CANONICAL_TRAINING_SCRIPT_SHA256
                )
                for subject in module.SUBJECT_ORDER
            },
        )


def test_training_script_equivalence_uses_canonical_subject_order() -> None:
    module = _load_module()

    equivalence = module._verify_training_script_equivalence(
        {
            module.LEGACY_TRAINING_SCRIPT_SHA256: LEGACY_TRAINING_BOOTSTRAP_SOURCE,
            module.CANONICAL_TRAINING_SCRIPT_SHA256: (
                CANONICAL_TRAINING_BOOTSTRAP_SOURCE
            ),
        },
        {
            subject: module._expected_training_script(subject)
            for subject in reversed(module.SUBJECT_ORDER)
        },
    )

    assert equivalence["legacy_script_subjects"] == [
        subject
        for subject in module.SUBJECT_ORDER
        if subject in module.LEGACY_SCRIPT_ALLOWED_SUBJECTS
    ]


def test_provenance_dependency_is_frozen_through_publication() -> None:
    module = _load_module()
    world = _world(module)
    original_upload = world.output.upload_artifact

    def upload(name: str, *, artifact_object: object, wait_on_upload: bool) -> bool:
        result = original_upload(
            name,
            artifact_object=artifact_object,
            wait_on_upload=wait_on_upload,
        )
        world.provenance_task.data.status = "failed"
        return result

    world.output.upload_artifact = upload
    with pytest.raises(RuntimeError, match="training provenance changed"):
        _run(module, world)


def test_run_publishes_sealed_legacy_source_c_audit() -> None:
    module = _load_module()
    world = _world(module)

    payload = _run(module, world)

    assert payload["passed"] is True
    assert payload["legacy_unseeded_training_manifest"] is True
    assert payload["training_seed"] == module.TRAINING_SEED
    assert payload["training_seed_evidence"] == "all_26_live_run_contracts"
    assert payload["audit_task_id"] == world.output.id
    assert payload["legacy_source_c_seed_schema"] == ("explicit_run_contract_seed_only")
    assert payload["final_model_verification_level"] == (
        "clearml_metadata_contract_only"
    )
    assert payload["checkpoint_bytes_sha256_recomputed"] is False
    assert payload["subject_count"] == 26
    assert payload["run_count_per_subject"] == 12
    assert payload["total_evaluation_run_count"] == 312
    assert len(payload["training_tasks"]) == 26
    assert len(payload["evaluation_tasks"]) == 26
    assert all(
        record["run_contract_seed_fields"] == ["seed"]
        for record in payload["training_tasks"]
    )
    assert {
        record["execution_queue_name"] for record in payload["evaluation_tasks"]
    } == set(QUEUE_IDS)
    assert all(
        record["metric_keys"] == list(module.AP_METRIC_KEYS)
        for record in payload["evaluation_tasks"]
    )
    assert module._sealed(payload) == payload
    assert world.output.uploads == [(module.AUDIT_ARTIFACT, payload, True)]
    assert world.output.flush_calls == 1
    assert "formal-1337-comparability-audit" in world.output.tags
    expected_batch_ids = {world.output.id, *world.tasks}
    assert len(_TaskClass.batch_calls) == 3
    assert all(len(call) == len(expected_batch_ids) for call in _TaskClass.batch_calls)
    assert all(set(call) == expected_batch_ids for call in _TaskClass.batch_calls)


def test_run_waits_for_all_three_completed_dependencies() -> None:
    module = _load_module()
    world = _world(
        module,
        controller_status="created",
        watcher_status="created",
        leaderboard_status="created",
        controller_reload_statuses=["in_progress", "completed"],
        watcher_reload_statuses=["in_progress", "completed"],
        leaderboard_reload_statuses=["in_progress", "completed"],
    )
    sleeps: list[float] = []

    module.run(
        _args(),
        task_class=_TaskClass,
        output_task=world.output,
        monotonic_clock=lambda: 0.0,
        sleeper=sleeps.append,
        queue_name_resolver=QUEUE_NAMES.__getitem__,
    )

    assert sleeps == [0.25]
    assert world.controller.reload_calls >= 2
    assert world.watcher.reload_calls >= 2
    assert world.leaderboard_task.reload_calls >= 2
    assert world.controller.reload_statuses == []
    assert world.watcher.reload_statuses == []
    assert world.leaderboard_task.reload_statuses == []


@pytest.mark.parametrize(
    ("target", "bad_parent", "message"),
    [
        ("provenance_task", WATCHER_ID, "training provenance parent mismatch"),
        ("watcher", CONTROLLER_ID, "evaluation watcher parent mismatch"),
        ("leaderboard_task", CONTROLLER_ID, "leaderboard parent mismatch"),
    ],
)
def test_formal_root_parent_chain_is_exact(
    target: str, bad_parent: str, message: str
) -> None:
    module = _load_module()
    world = _world(module)
    getattr(world, target).data.parent = bad_parent

    with pytest.raises(RuntimeError, match=message):
        _run(module, world)

    assert world.output.uploads == []
    assert world.output.tags == []


@pytest.mark.parametrize("stage", ["upload", "authoritative_batch"])
def test_watcher_parent_is_frozen_across_fresh_readbacks(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    world = _world(module)
    if stage == "upload":
        original_upload = world.output.upload_artifact

        def upload(
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            result = original_upload(
                name,
                artifact_object=artifact_object,
                wait_on_upload=wait_on_upload,
            )
            world.watcher.data.parent = CONTROLLER_ID
            return result

        world.output.upload_artifact = upload
    else:
        original_query = _TaskClass._query_tasks.__func__

        def query(cls, **kwargs) -> list[dict[str, object]]:
            world.watcher.data.parent = CONTROLLER_ID
            return original_query(cls, **kwargs)

        monkeypatch.setattr(_TaskClass, "_query_tasks", classmethod(query))

    with pytest.raises(RuntimeError, match="evaluation watcher changed"):
        _run(module, world)

    assert world.output.tags == []


def test_run_uses_private_reload_and_installed_raw_snapshot_only() -> None:
    module = _load_module()
    world = _world(module)
    world.controller.status = "failed"
    world.controller.parent = "0" * 32
    world.controller.artifacts = {}
    world.controller.reload = lambda: pytest.fail("public reload must not be called")

    payload = _run(module, world)

    assert payload["passed"] is True
    assert world.controller.reload_calls > 0


def test_run_rejects_missing_private_reload_and_raw_failed_status() -> None:
    module = _load_module()
    world = _world(module)
    world.controller._reload = None
    with pytest.raises(RuntimeError, match="cannot be server-reloaded"):
        _run(module, world)

    world = _world(module)
    world.controller.status = "completed"
    world.controller.data.status = "failed"
    with pytest.raises(RuntimeError, match="ended as 'failed'"):
        _run(module, world)


def test_run_reads_exact_raw_artifact_inventory_with_forced_downloads() -> None:
    module = _load_module()
    world = _world(module)
    public_only = _Artifact({"attacker": True})
    world.controller.artifacts["attacker_public_only"] = public_only

    _run(module, world)

    server_artifacts = [
        raw.artifact
        for task in world.tasks.values()
        for raw in task.data.execution.artifacts
    ]
    server_artifacts.extend(
        raw.artifact for raw in world.output.data.execution.artifacts
    )
    assert server_artifacts
    assert all(artifact.force_download_calls for artifact in server_artifacts)
    assert all(
        call is True
        for artifact in server_artifacts
        for call in artifact.force_download_calls
    )
    assert public_only.force_download_calls == []


@pytest.mark.parametrize(
    ("owner", "artifact_name"),
    [
        ("controller", "post_main_training_progress"),
        ("controller", "formal_1337_training_manifest"),
        ("controller", "post_main_training_summary"),
        ("provenance", "formal_1337_training_provenance_equivalence"),
        ("watcher", "formal_1337_evaluation_plan"),
        ("leaderboard", "formal_1337_leaderboard"),
        ("training", "run_contract"),
        ("training", "final_checkpoint_contract"),
        ("training", "best_checkpoint_contract"),
        ("training", "common_teacher_initialization_audit"),
        ("training", "baseline_dry_run_plan"),
        ("training", "baseline_resolved_config"),
        ("evaluation", "run_contract"),
        ("evaluation", "evaluation_plan"),
        ("evaluation", "controlled_baseline_metrics"),
        ("evaluation", "controlled_baseline_evidence"),
    ],
)
def test_publication_freezes_every_exact_artifact_content(
    owner: str,
    artifact_name: str,
) -> None:
    module = _load_module()
    world = _world(module)
    if owner == "controller":
        task = world.controller
    elif owner == "provenance":
        task = world.provenance_task
    elif owner == "watcher":
        task = world.watcher
    elif owner == "leaderboard":
        task = world.leaderboard_task
    else:
        training_entry = next(
            entry for entry in world.manifest["entries"] if entry["subject"] == "ffnet"
        )
        if owner == "training":
            task = world.tasks[training_entry["training_task_id"]]
        else:
            evaluation_entry = next(
                entry for entry in world.plan["entries"] if entry["subject"] == "ffnet"
            )
            task = world.tasks[evaluation_entry["evaluation_task_id"]]
    artifact = _server_artifact(task, artifact_name)
    original_upload = world.output.upload_artifact

    def upload(
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        result = original_upload(
            name,
            artifact_object=artifact_object,
            wait_on_upload=wait_on_upload,
        )
        artifact.value = {"tampered_artifact": artifact_name}
        return result

    world.output.upload_artifact = upload
    with pytest.raises(RuntimeError, match="changed during publication"):
        _run(module, world)

    assert artifact.force_download_calls
    assert all(call is True for call in artifact.force_download_calls)
    assert world.output.tags == []


@pytest.mark.parametrize("target", ["controller", "provenance", "training", "output"])
def test_run_rejects_extra_raw_server_artifacts(target: str) -> None:
    module = _load_module()
    world = _world(module)
    if target == "controller":
        task = world.controller
    elif target == "provenance":
        task = world.provenance_task
    elif target == "training":
        first = world.manifest["entries"][0]
        task = world.tasks[first["training_task_id"]]
    else:
        task = world.output
    _install_server_artifact(task, "attacker_extra", _Artifact({"extra": True}))

    with pytest.raises(RuntimeError, match="artifact inventory"):
        _run(module, world)


def test_run_rejects_duplicate_raw_server_artifact_names() -> None:
    module = _load_module()
    world = _world(module)
    raw = next(
        artifact
        for artifact in world.controller.data.execution.artifacts
        if artifact.key == module.TRAINING_MANIFEST_ARTIFACT
    )
    world.controller.data.execution.artifacts.append(
        _RawArtifact(raw.key, _Artifact(raw.artifact.value))
    )

    with pytest.raises(RuntimeError, match="raw server artifact inventory"):
        _run(module, world)


def test_artifact_reader_never_falls_back_without_force_download() -> None:
    module = _load_module()
    world = _world(module)

    class RejectForceDownload:
        def get(self) -> object:
            return world.manifest

    raw = next(
        artifact
        for artifact in world.controller.data.execution.artifacts
        if artifact.key == module.TRAINING_MANIFEST_ARTIFACT
    )
    raw.artifact = RejectForceDownload()
    module._reload(world.controller, context="training controller")

    with pytest.raises(RuntimeError, match="cannot be force-downloaded"):
        module._artifact_mapping(
            world.controller,
            module.TRAINING_MANIFEST_ARTIFACT,
            context="training controller",
        )


def test_secure_json_reader_rejects_symlink_hardlink_and_oversize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    target = tmp_path / "target.json"
    target.write_text('{"ok":true}', encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)

    with pytest.raises(RuntimeError, match="cannot be read|unsafe"):
        module._read_json_path(symlink, context="symlink")
    with pytest.raises(RuntimeError, match="unsafe"):
        module._read_json_path(hardlink, context="hardlink")

    standalone = tmp_path / "standalone.json"
    standalone.write_text('{"ok":true}', encoding="utf-8")
    monkeypatch.setattr(module, "MAX_JSON_ARTIFACT_BYTES", 1)
    with pytest.raises(RuntimeError, match="unsafe|too large"):
        module._read_json_path(standalone, context="oversize")


def test_artifact_snapshot_hashes_regular_files_and_directory_trees(
    tmp_path: Path,
) -> None:
    module = _load_module()
    config = tmp_path / "resolved.py"
    config.write_text("model = dict(type='FFNet')\n", encoding="utf-8")
    config_snapshot = module._snapshot_artifact_value(
        config,
        context="resolved config",
    )
    assert config_snapshot == {
        "kind": "file",
        "size_bytes": config.stat().st_size,
        "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }

    evidence = tmp_path / "evidence"
    nested = evidence / "nested"
    nested.mkdir(parents=True)
    (evidence / "prediction.json").write_text('{"score":1}', encoding="utf-8")
    (nested / "metrics.json").write_text('{"ap":2}', encoding="utf-8")
    before = module._snapshot_artifact_value(evidence, context="evidence")
    assert before["kind"] == "directory"
    assert [entry["path"] for entry in before["entries"]] == [
        "nested",
        "nested/metrics.json",
        "prediction.json",
    ]

    (nested / "metrics.json").write_text('{"ap":3}', encoding="utf-8")
    after = module._snapshot_artifact_value(evidence, context="evidence")
    assert after != before

    (evidence / "unsafe-link").symlink_to(config)
    with pytest.raises(RuntimeError, match="non-regular entry"):
        module._snapshot_artifact_value(evidence, context="evidence")


def test_canonical_json_rejects_nonfinite_nonstring_keys_and_cycles() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="non-finite"):
        module._canonical_json({"value": float("nan")})
    with pytest.raises(ValueError, match="non-string"):
        module._canonical_json({1: "ambiguous"})
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cycle"):
        module._canonical_json({"value": cyclic})


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("seed", 7, "seed mismatch"),
        ("precision", "FP16", "precision mismatch"),
        ("ddp_processes", 1, "ddp_processes mismatch"),
        ("val_interval", 1, "val_interval mismatch"),
    ],
)
def test_run_rejects_training_run_contract_drift(
    field: str,
    replacement: object,
    message: str,
) -> None:
    module = _load_module()
    world = _world(module)
    first = world.manifest["entries"][0]
    task = world.tasks[first["training_task_id"]]
    task.artifacts[module.RUN_CONTRACT_ARTIFACT].value[field] = replacement

    with pytest.raises(RuntimeError, match=message):
        _run(module, world)


def test_run_rejects_training_source_archive_or_script_drift() -> None:
    module = _load_module()
    world = _world(module)
    first = world.manifest["entries"][0]
    task = world.tasks[first["training_task_id"]]
    contract = task.artifacts[module.RUN_CONTRACT_ARTIFACT].value
    contract["source_archive"]["sha256"] = "9" * 64

    with pytest.raises(
        RuntimeError,
        match="run contract source archive identity mismatch",
    ):
        _run(module, world)

    world = _world(module)
    first = world.manifest["entries"][0]
    world.tasks[first["training_task_id"]].data.script.diff = "drifted source"
    with pytest.raises(RuntimeError, match="script SHA-256 mismatch"):
        _run(module, world)


def test_seed_contract_is_explicit_for_legacy_and_seeded_schemas() -> None:
    module = _load_module()
    assert module._run_contract_seed(
        {"seed": module.TRAINING_SEED},
        subject="ffnet",
        manifest_seeded=False,
    ) == (module.TRAINING_SEED, ["seed"])
    with pytest.raises(RuntimeError, match="seed schema mismatch"):
        module._run_contract_seed(
            {},
            subject="ffnet",
            manifest_seeded=False,
        )
    seeded = {
        "seed": module.TRAINING_SEED,
        "training_seed": module.TRAINING_SEED,
        "training_overlay_protocol_seed": module.TRAINING_SEED,
    }
    assert module._run_contract_seed(
        seeded,
        subject="ffnet",
        manifest_seeded=True,
    ) == (module.TRAINING_SEED, sorted(seeded))
    del seeded["training_overlay_protocol_seed"]
    with pytest.raises(RuntimeError, match="seed schema mismatch"):
        module._run_contract_seed(
            seeded,
            subject="ffnet",
            manifest_seeded=True,
        )


def test_run_never_uses_a_c2_w2_or_l2_task_as_output() -> None:
    module = _load_module()
    world = _world(module)
    world.output.id = CONTROLLER_ID

    with pytest.raises(RuntimeError, match="must be outside C2/W2/L2"):
        _run(module, world)

    assert world.output.tags == []
    assert world.output.uploads == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("parent", "output parent drifted"),
        ("status", "output status is not writable"),
        ("entry", "entry point drifted"),
        ("source", "producer SHA-256 drifted"),
        ("args", "Args contract drifted"),
        ("uri", "output destination drifted"),
    ],
)
def test_run_rejects_output_provenance_drift(
    mutation: str,
    message: str,
) -> None:
    module = _load_module()
    world = _world(module)
    if mutation == "parent":
        world.output.data.parent = ""
    elif mutation == "status":
        world.output.data.status = "completed"
    elif mutation == "entry":
        world.output.data.script.entry_point = "clearml_5090_bootstrap.py"
    elif mutation == "source":
        world.output.data.script.diff = "attacker source"
    elif mutation == "args":
        world.output.parameters["Args/attacker_extra"] = "1"
    else:
        world.output.data.output.destination = "http://attacker.invalid"

    with pytest.raises(RuntimeError, match=message):
        _run(module, world)

    assert world.output.tags == []
    assert world.output.uploads == []


def test_output_accepts_exact_original_queue_pins_and_known_empty_defaults() -> None:
    module = _load_module()
    world = _world(module)
    world.output.parameters.update(module.ORIGINAL_QUEUE_RUNTIME_PINS)
    world.output.parameters.update(module.OUTPUT_RUNTIME_EMPTY_DEFAULT_PARAMETERS)

    snapshot = module._output_snapshot(
        world.output,
        task_id=world.output.id,
        leaderboard_task_id=LEADERBOARD_ID,
        expected_source=module._runtime_source(),
        expected_parameters={
            "Args/training_controller_task_id": CONTROLLER_ID,
            "Args/training_provenance_task_id": PROVENANCE_ID,
            "Args/watcher_task_id": WATCHER_ID,
            "Args/leaderboard_task_id": LEADERBOARD_ID,
            "Args/poll_seconds": "0.25",
            "Args/timeout_hours": "1.0",
        },
    )

    assert len(module.ORIGINAL_QUEUE_RUNTIME_PINS) == 9
    assert all(
        snapshot["parameters"][key] == value
        for key, value in module.ORIGINAL_QUEUE_RUNTIME_PINS.items()
    )
    assert not (
        set(snapshot["parameters"])
        & set(module.OUTPUT_RUNTIME_EMPTY_DEFAULT_PARAMETERS)
    )


def test_output_preserves_declared_nonempty_amended_pins() -> None:
    module = _load_module()
    world = _world(module)
    amended = {
        f"Args/{name}": ("1" * 32 if kind == "task_id" else "a" * 64)
        for name, kind in module.SEALED_SUCCESSOR_PIN_ARGS
    }
    base = dict(world.output.parameters)
    world.output.parameters.update(amended)

    snapshot = module._output_snapshot(
        world.output,
        task_id=world.output.id,
        leaderboard_task_id=LEADERBOARD_ID,
        expected_source=module._runtime_source(),
        expected_parameters={**base, **amended},
    )

    assert all(snapshot["parameters"][key] == value for key, value in amended.items())


@pytest.mark.parametrize("mutation", ("missing_pin", "wrong_pin", "nonempty_default"))
def test_output_rejects_original_queue_or_runtime_default_drift(mutation: str) -> None:
    module = _load_module()
    world = _world(module)
    world.output.parameters.update(module.ORIGINAL_QUEUE_RUNTIME_PINS)
    world.output.parameters.update(module.OUTPUT_RUNTIME_EMPTY_DEFAULT_PARAMETERS)
    if mutation == "missing_pin":
        world.output.parameters.pop(next(iter(module.ORIGINAL_QUEUE_RUNTIME_PINS)))
        message = "original-queue authority is incomplete"
    elif mutation == "wrong_pin":
        key = next(iter(module.ORIGINAL_QUEUE_RUNTIME_PINS))
        world.output.parameters[key] = "attacker"
        message = "original-queue authority drifted"
    else:
        key = next(iter(module.OUTPUT_RUNTIME_EMPTY_DEFAULT_PARAMETERS))
        world.output.parameters[key] = "attacker"
        message = "runtime default"

    with pytest.raises(RuntimeError, match=message):
        _run(module, world)

    assert world.output.tags == []
    assert world.output.uploads == []


def test_run_ignores_public_output_facades() -> None:
    module = _load_module()
    world = _world(module)
    world.output.status = "completed"
    world.output.parent = ""
    world.output.artifacts["attacker_public_only"] = _Artifact({"extra": True})

    payload = _run(module, world)

    assert payload["passed"] is True


def test_run_rejects_child_object_alias_and_raw_identity_drift() -> None:
    module = _load_module()
    world = _world(module)
    first_id = world.manifest["entries"][0]["training_task_id"]
    world.tasks[first_id] = world.controller
    with pytest.raises(RuntimeError, match="aliases another formal task"):
        _run(module, world)

    world = _world(module)
    world.tasks[TEMPLATE_ID].data.id = "0" * 32
    with pytest.raises(RuntimeError, match="evaluation template identity mismatch"):
        _run(module, world)


def test_run_binds_initialization_audit_and_actual_execution_queue() -> None:
    module = _load_module()
    world = _world(module)
    training = world.manifest["entries"][0]
    task = world.tasks[training["training_task_id"]]
    task.artifacts[module.COMMON_TEACHER_AUDIT_ARTIFACT].value["result"] = "drift"

    with pytest.raises(RuntimeError, match="initialization audit SHA-256 mismatch"):
        _run(module, world)

    world = _world(module)
    evaluation = world.plan["entries"][0]
    task = world.tasks[evaluation["evaluation_task_id"]]
    task.data.execution.queue = QUEUE_IDS["GPU4-V100"]
    with pytest.raises(RuntimeError, match="execution queue mismatch"):
        _run(module, world)


def test_run_rejects_resealed_nested_manifest_or_template_drift() -> None:
    module = _load_module()
    world = _world(module)
    nested = copy.deepcopy(world.summary["formal_1337_evaluation_manifest"])
    nested["entries"][0]["checkpoint_size_bytes"] += 1
    nested = module._sealed(nested)
    world.summary["formal_1337_evaluation_manifest"] = nested
    world.controller.artifacts[module.TRAINING_SUMMARY_ARTIFACT].value = module._sealed(
        world.summary
    )

    with pytest.raises(ValueError, match="nested formal training manifests"):
        _run(module, world)

    world = _world(module)
    world.plan["evaluation_template_task_id"] = "9" * 32
    world.watcher.artifacts[module.EVALUATION_PLAN_ARTIFACT].value = module._sealed(
        world.plan
    )
    with pytest.raises(
        ValueError,
        match="evaluation_template_task_id mismatch",
    ):
        _run(module, world)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_ap", "must be numeric"),
        ("sample_count", "sample_count mismatch"),
        ("metric_count", "car_ground_truth_count mismatch"),
    ],
)
def test_run_rejects_incomparable_1337_metrics(
    mutation: str,
    message: str,
) -> None:
    module = _load_module()
    world = _world(module)
    evaluation = world.plan["entries"][0]
    task = world.tasks[evaluation["evaluation_task_id"]]
    run = task.artifacts[module.METRICS_ARTIFACT].value["runs"][0]
    if mutation == "missing_ap":
        del run["metrics"][module.AP_METRIC_KEYS[0]]
    elif mutation == "sample_count":
        run["sample_count"] = module.SAMPLE_COUNT + 1
    else:
        run["metrics"]["resilient_v2x/car_ground_truth_count"] = (
            module.GROUND_TRUTH_COUNT - 1
        )

    with pytest.raises(ValueError, match=message):
        _run(module, world)


def test_run_recomputes_and_rejects_resealed_leaderboard_drift() -> None:
    module = _load_module()
    world = _world(module)
    result = world.leaderboard["results"][0]
    result["metrics"][module.AP_METRIC_KEYS[0]]["mean_12"] += 1.0
    world.leaderboard_task.artifacts[
        module.LEADERBOARD_ARTIFACT
    ].value = module._sealed(world.leaderboard)

    with pytest.raises(ValueError, match="metric summary mismatch"):
        _run(module, world)


def test_publish_requires_exact_upload_flush_and_fresh_server_commit() -> None:
    module = _load_module()
    world = _world(module)
    world.output.upload_artifact = lambda *args, **kwargs: 1
    with pytest.raises(RuntimeError, match="failed to upload"):
        _run(module, world)
    assert world.output.tags == []

    world = _world(module)
    world.output.flush = lambda **kwargs: False
    with pytest.raises(RuntimeError, match="failed to flush"):
        _run(module, world)
    assert world.output.tags == []

    world = _world(module)
    world.output.upload_artifact = lambda *args, **kwargs: True
    with pytest.raises(RuntimeError, match="lacks artifact"):
        _run(module, world)
    assert world.output.tags == []


def test_publish_rejects_fresh_server_readback_drift_and_is_idempotent() -> None:
    module = _load_module()
    world = _world(module)

    def tampered_upload(
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        assert isinstance(artifact_object, dict)
        tampered = dict(artifact_object)
        tampered["passed"] = False
        _install_server_artifact(world.output, name, _Artifact(tampered))
        return wait_on_upload

    world.output.upload_artifact = tampered_upload
    with pytest.raises(RuntimeError, match="bytes drifted"):
        _run(module, world)

    world = _world(module)
    first = _run(module, world)
    second = _run(module, world)
    assert second == first
    assert len(world.output.uploads) == 1
    assert world.output.flush_calls == 1


@pytest.mark.parametrize("stage", ["upload", "flush", "readback"])
def test_publication_callback_rejects_dependency_toctou(stage: str) -> None:
    module = _load_module()
    world = _world(module)

    if stage == "upload":
        original_upload = world.output.upload_artifact

        def upload(
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            result = original_upload(
                name,
                artifact_object=artifact_object,
                wait_on_upload=wait_on_upload,
            )
            world.controller.data.status = "failed"
            return result

        world.output.upload_artifact = upload
    elif stage == "flush":
        original_flush = world.output.flush

        def flush(*, wait_for_uploads: bool) -> bool:
            result = original_flush(wait_for_uploads=wait_for_uploads)
            world.controller.data.status = "failed"
            return result

        world.output.flush = flush
    else:

        class MutatingReadback(_Artifact):
            def get(self, *, force_download: bool = False) -> object:
                world.controller.data.status = "failed"
                return super().get(force_download=force_download)

        def upload_for_readback(
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            _install_server_artifact(
                world.output,
                name,
                MutatingReadback(artifact_object),
            )
            return wait_on_upload

        world.output.upload_artifact = upload_for_readback

    with pytest.raises(RuntimeError, match="training controller changed"):
        _run(module, world)


def test_terminal_output_validation_cannot_mutate_an_already_checked_dependency() -> (
    None
):
    module = _load_module()
    world = _world(module)
    state = {"armed": False}
    original_output_reload = world.output._reload

    def output_reload() -> object:
        snapshot = original_output_reload()
        if state["armed"]:
            world.controller.data.status = "failed"
            state["armed"] = False
        return snapshot

    world.output._reload = output_reload

    class ArmOnReadback(_Artifact):
        def get(self, *, force_download: bool = False) -> object:
            state["armed"] = True
            return super().get(force_download=force_download)

    def upload(
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        _install_server_artifact(
            world.output,
            name,
            ArmOnReadback(artifact_object),
        )
        return wait_on_upload

    world.output.upload_artifact = upload
    with pytest.raises(RuntimeError, match="training controller changed"):
        _run(module, world)

    assert world.output.tags == []


def test_one_batch_snapshot_closes_last_dependency_reload_mutation_window() -> None:
    module = _load_module()
    world = _world(module)
    last_evaluation_id = world.plan["entries"][-1]["evaluation_task_id"]
    last_evaluation = world.tasks[last_evaluation_id]
    state = {"output_reloads": 0, "mutated": False}
    original_output_reload = world.output._reload
    original_evaluation_reload = last_evaluation._reload

    def output_reload() -> object:
        snapshot = original_output_reload()
        state["output_reloads"] += 1
        return snapshot

    def evaluation_reload() -> object:
        snapshot = original_evaluation_reload()
        # The initial output inspection is reload 1. Reload 2 occurs between
        # the two forward dependency sweeps; this evaluation is last, so the
        # old sequential algorithm had no later controller read to catch it.
        if state["output_reloads"] >= 2 and not state["mutated"]:
            world.controller.data.status = "failed"
            state["mutated"] = True
        return snapshot

    world.output._reload = output_reload
    last_evaluation._reload = evaluation_reload

    with pytest.raises(
        RuntimeError, match="training controller changed in authoritative batch"
    ):
        _run(module, world)

    assert state["mutated"] is True
    assert _TaskClass.batch_calls
    assert set(_TaskClass.batch_calls[-1]) == {world.output.id, *world.tasks}
    assert world.output.tags == []


def test_publication_has_no_sequential_fallback_without_batch_raw_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    world = _world(module)
    monkeypatch.setattr(_TaskClass, "_query_tasks", None)

    with pytest.raises(RuntimeError, match="cannot issue one authoritative batch read"):
        _run(module, world)

    assert world.output.uploads == []
    assert world.output.tags == []


def test_terminal_batch_binds_the_force_downloaded_output_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    world = _world(module)
    original_query = _TaskClass._query_tasks

    def query(cls, **kwargs) -> list[dict[str, object]]:
        records = original_query(**kwargs)
        if len(cls.batch_calls) == 2:
            output_record = next(
                record for record in records if record["id"] == world.output.id
            )
            output_record["execution"]["artifacts"][0]["content"] = {
                "tampered_after_download": True
            }
        return records

    monkeypatch.setattr(_TaskClass, "_query_tasks", classmethod(query))

    with pytest.raises(
        RuntimeError,
        match="audit output changed after its last verified committed reload",
    ):
        _run(module, world)

    assert world.output.tags == []


@pytest.mark.parametrize("behavior", ["false_after_commit", "raise_after_commit"])
def test_tag_callback_result_is_decided_by_fresh_batch_readback(
    behavior: str,
) -> None:
    module = _load_module()
    world = _world(module)
    original_set_tags = world.output.set_tags

    def set_tags(tags: list[str]) -> bool:
        original_set_tags(tags)
        if behavior == "raise_after_commit":
            raise RuntimeError("transport lost after server commit")
        return False

    world.output.set_tags = set_tags

    payload = _run(module, world)

    assert payload["passed"] is True
    assert world.output.tags == list(module.FORMAL_TAGS)
    assert world.output.data.tags == list(module.FORMAL_TAGS)
    assert world.output.tag_calls == [list(module.FORMAL_TAGS)]


def test_existing_formal_tags_are_idempotent_without_a_tag_callback() -> None:
    module = _load_module()
    world = _world(module)
    _install_server_tags(world.output, list(module.FORMAL_TAGS))
    world.output.set_tags = None

    payload = _run(module, world)

    assert payload["passed"] is True
    assert world.output.tags == list(module.FORMAL_TAGS)
    assert world.output.tag_calls == []


def test_tag_true_without_server_commit_is_rejected_and_keeps_original_tags() -> None:
    module = _load_module()
    world = _world(module)
    world.output.set_tags = lambda tags: True

    with pytest.raises(
        RuntimeError, match="formal tags were not committed by the server"
    ):
        _run(module, world)

    assert world.output.tags == []
    assert world.output.data.tags == []


@pytest.mark.parametrize("raise_after_commit", [False, True])
def test_tag_binding_drift_is_rejected_and_restores_original_tags(
    raise_after_commit: bool,
) -> None:
    module = _load_module()
    world = _world(module)
    original_tags = ["preexisting-review-tag"]
    _install_server_tags(world.output, original_tags)
    original_set_tags = world.output.set_tags

    def set_tags(tags: list[str]) -> bool:
        result = original_set_tags(tags)
        if tags == list(module.FORMAL_TAGS):
            world.controller.data.status = "failed"
        if raise_after_commit:
            raise RuntimeError("transport lost after server commit")
        return result

    world.output.set_tags = set_tags

    with pytest.raises(
        RuntimeError, match="formal task bindings changed during tag commit"
    ):
        _run(module, world)

    assert world.output.tags == original_tags
    assert world.output.data.tags == original_tags
    assert world.output.tag_calls == [list(module.FORMAL_TAGS), original_tags]


def test_set_tags_is_delayed_until_all_evidence_validates() -> None:
    module = _load_module()
    world = _world(module)
    evaluation = world.plan["entries"][0]
    metrics = _server_artifact(
        world.tasks[evaluation["evaluation_task_id"]],
        module.METRICS_ARTIFACT,
    ).value
    metrics["runs"][0]["sample_count"] += 1

    with pytest.raises(ValueError, match="sample_count mismatch"):
        _run(module, world)

    assert world.output.tags == []
    assert world.output.uploads == []


def test_leaderboard_values_have_independent_hand_calculated_expectations() -> None:
    module = _load_module()
    world = _world(module)
    primary = next(
        result
        for result in world.leaderboard["results"]
        if result["subject"] == module.PRIMARY_SUBJECT
    )
    summary = primary["metrics"][module.LEADERSHIP_METRIC]
    assert summary == {
        "clean_0ms": 85.0,
        "mean_12": pytest.approx(78.5),
        "worst_12": {
            "value": 72.0,
            "delay_ms": 300,
            "condition": "L-Fail",
        },
        "full_mean": pytest.approx(83.5),
        "l_fail_mean": pytest.approx(73.5),
        "c_fail_mean": pytest.approx(78.5),
        "full_300": 82.0,
        "pdr": pytest.approx(100.0 * 3.0 / 85.0),
    }
    leadership = world.leaderboard["leadership"]
    assert leadership["aggregate_dimensions"]["clean_0ms"]["margin"] == 22.0
    assert leadership["conditions_won"] == 12
    assert leadership["conditions_won_fraction"] == "12/12"

    _run(module, world)
