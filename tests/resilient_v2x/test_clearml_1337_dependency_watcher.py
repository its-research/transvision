from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections import UserList
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_1337_dependency_watcher.py"
CONTROLLER_PATH = ROOT / "tools/resilient_v2x/clearml_5090_training_controller.py"
PRODUCTION_TRAINING_SCRIPT_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
PRODUCTION_EVALUATION_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
TRAINING_PROJECT_ID = "6e43f972e5ea4cee901a7c8855fce8cd"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "clearml_1337_dependency_watcher", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "clearml_5090_training_controller_for_1337", CONTROLLER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _formal_training_manifest() -> dict[str, object]:
    controller = _load_controller()
    results = []
    predecessor = "f" * 32
    for index, subject in enumerate(controller.EXPERIMENT_ORDER, start=1):
        task_id = f"{index:032x}"
        results.append(
            {
                "index": index,
                "experiment": subject,
                "predecessor_task_id": predecessor,
                "task_id": task_id,
                "model_id": f"{index + 100:032x}",
                "model_name": f"ResilientV2X {subject} final checkpoint",
                "model_url": (
                    f"http://10.100.34.118:8081/models/{subject}_epoch_50.pth"
                ),
                "checkpoint_sha256": f"{index:064x}",
                "checkpoint_size_bytes": 1000 + index,
                "run_contract_artifact": controller.RUN_CONTRACT_ARTIFACT,
                "final_checkpoint_artifact": (controller.FINAL_CHECKPOINT_ARTIFACT),
                "common_teacher_initialization_audit_artifact": (
                    controller.COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT
                ),
                "common_teacher_initialization_audit_sha256": "a" * 64,
                "training_seed": controller.DEFAULT_TRAINING_SEED,
                "training_overlay_protocol_seed": (
                    controller.TRAINING_OVERLAY_PROTOCOL_SEED
                ),
            }
        )
        predecessor = task_id
    return controller.build_formal_1337_training_manifest(results)


def _evaluation_plan(module) -> list[dict[str, str]]:
    queues = tuple(module.EXPECTED_QUEUE_IDS)
    return [
        {
            "subject": subject,
            "evaluation_task_id": f"{index + 1000:032x}",
            "queue": queues[(index - 1) % len(queues)],
        }
        for index, subject in enumerate(module.FORMAL_SUBJECT_ORDER, start=1)
    ]


def _reseal(module, value: dict[str, object]) -> dict[str, object]:
    value.pop("seal_sha256", None)
    value["seal_sha256"] = hashlib.sha256(
        module._canonical_json(value).encode("utf-8")
    ).hexdigest()
    return value


def _formal_training_provenance_document(
    module,
    *,
    controller_task_id: str = "f" * 32,
    provenance_task_id: str = "e" * 32,
) -> tuple[dict[str, object], list[dict[str, str]], dict[str, object]]:
    manifest = _formal_training_manifest()
    training = module._parse_training_manifest(json.dumps(manifest))
    manifest_entries = manifest["entries"]
    assert isinstance(manifest_entries, list)
    records = []
    legacy_subjects = ["support_residual", "no_distillation"]
    expanded_subjects = [
        subject
        for subject in module.FORMAL_SUBJECT_ORDER
        if subject not in legacy_subjects
    ]
    observed_parameter_keys = sorted(
        {
            "Args/experiment_from_task",
            "Args/source_dataset_id",
            "Args/source_archive_name",
            "Args/source_archive_bytes",
            "Args/source_archive_sha256",
            "Args/training_dataset_id",
            "Args/native_bundle_sha256",
            "Args/build_manifest_sha256",
            "Args/predecessor_task_id",
            "Args/gpus",
            "Args/stage",
            "Args/max_epochs",
            "Args/amp",
            "Args/training_seed",
            "Args/teacher_checkpoint_sha256",
        }
    )
    for index, (subject, dependency) in enumerate(
        zip(module.FORMAL_SUBJECT_ORDER, training, strict=True), start=1
    ):
        legacy = subject in legacy_subjects
        source_revision = module.SOURCE_REVISION_BY_SUBJECT[subject]
        source = module.EXPECTED_SOURCE_BY_REVISION[source_revision]
        manifest_entry = manifest_entries[index - 1]
        assert isinstance(manifest_entry, dict)
        records.append(
            {
                "index": index,
                "subject": subject,
                "training_task_id": dependency["training_task_id"],
                "predecessor_task_id": dependency["training_predecessor_task_id"],
                "parent_controller_task_id": controller_task_id,
                "parent_binding": "current_controller_created",
                "recovery_lineage": [
                    {
                        "controller_task_id": controller_task_id,
                        "decision": "controller_created",
                        "progress_seal_sha256": "1" * 64,
                    }
                ],
                "raw_bootstrap_script_sha256": (
                    module.EXPECTED_LEGACY_TRAINING_SCRIPT_SHA256
                    if legacy
                    else module.EXPECTED_CANONICAL_TRAINING_SCRIPT_SHA256
                ),
                "normalized_task_script_identity_sha256": "6" * 64,
                "sealed_progress_script_identity_sha256": "6" * 64,
                "script_equivalence_class": (
                    "legacy_nested_teacher_membership"
                    if legacy
                    else "expanded_nested_teacher_membership"
                ),
                "manifest_model_id": dependency["training_model_id"],
                "manifest_model_name": manifest_entry["model_name"],
                "manifest_model_url": manifest_entry["model_url"],
                "checkpoint_sha256": manifest_entry["checkpoint_sha256"],
                "checkpoint_size_bytes": manifest_entry["checkpoint_size_bytes"],
                "source_revision_tree_sha256": source["tree_sha256"],
                "source_dataset_id": source["dataset_id"],
                "source_archive_name": source["archive_name"],
                "source_archive_bytes": source["archive_size_bytes"],
                "source_archive_sha256": source["archive_sha256"],
                "observed_parameter_keys": observed_parameter_keys,
                "observed_parameters_sha256": f"{index + 300:064x}",
                "docker_command_sha256": module.EXPECTED_DOCKER_COMMAND_SHA256,
                "execution_queue_id": next(iter(module.EXPECTED_QUEUE_IDS.values())),
                "last_worker": "sealed-worker",
                "raw_authority_sha256": "8" * 64,
                "run_contract_content_sha256": f"{index + 200:064x}",
                "final_checkpoint_contract_content_sha256": f"{index + 400:064x}",
                "common_teacher_initialization_audit_content_sha256": "a" * 64,
            }
        )
    source_equivalence = {
        "schema_version": 1,
        "contract": "fixture-source-revision-equivalence",
        "source_revisions": {
            source["tree_sha256"]: dict(source)
            for source in module.EXPECTED_SOURCE_BY_REVISION.values()
        },
    }
    _reseal(module, source_equivalence)
    module.EXPECTED_SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = source_equivalence[
        "seal_sha256"
    ]
    source_subject_map = {
        "schema_version": 1,
        "contract": "fixture-source-revision-subject-map",
        "source_revision_by_subject": {
            subject: module.EXPECTED_SOURCE_BY_REVISION[
                module.SOURCE_REVISION_BY_SUBJECT[subject]
            ]["tree_sha256"]
            for subject in module.FORMAL_SUBJECT_ORDER
        },
    }
    _reseal(module, source_subject_map)
    module.EXPECTED_SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = source_subject_map[
        "seal_sha256"
    ]
    document = {
        "schema_version": 2,
        "document_type": "resilient_v2x_formal_training_provenance_equivalence",
        "protocol_id": module.EXPECTED_PROTOCOL_ID,
        "passed": True,
        "controller_task_id": controller_task_id,
        "controller_status": "completed",
        "controller_raw_authority_sha256": "9" * 64,
        "subject_order": list(module.FORMAL_SUBJECT_ORDER),
        "subject_count": 26,
        "all_training_tasks_completed": True,
        "authoritative_metadata_read": "single_batch_per_snapshot",
        "progress_artifact": module.FORMAL_TRAINING_PROGRESS_ARTIFACT,
        "progress_seal_sha256": "1" * 64,
        "progress_content_sha256": "2" * 64,
        "formal_training_manifest_artifact": (module.FORMAL_TRAINING_MANIFEST_ARTIFACT),
        "formal_training_manifest_seal_sha256": manifest["seal_sha256"],
        "formal_training_manifest_content_sha256": hashlib.sha256(
            module._canonical_json(manifest).encode()
        ).hexdigest(),
        "recovery_contract_sha256": "3" * 64,
        "recursive_progress_chain": [
            {
                "task_id": controller_task_id,
                "role": "final_controller",
                "progress_artifact": module.FORMAL_TRAINING_PROGRESS_ARTIFACT,
                "progress_revision": 1,
                "progress_seal_sha256": "1" * 64,
                "progress_content_sha256": "2" * 64,
                "template_script_identity_sha256": "6" * 64,
            }
        ],
        "bootstrap_equivalence": {
            "equivalence_contract": "nested-teacher-membership-only-v1",
            "verified_from_actual_script_bytes": True,
            "legacy_script_sha256": (module.EXPECTED_LEGACY_TRAINING_SCRIPT_SHA256),
            "expanded_script_sha256": (
                module.EXPECTED_CANONICAL_TRAINING_SCRIPT_SHA256
            ),
            "legacy_nested_teacher_experiments": [
                "resilient_v2x",
                "support_residual",
            ],
            "expanded_nested_teacher_experiments": sorted(
                {
                    "support_residual",
                    "ptf_none",
                    "ptf_linear",
                    "router_static",
                    "router_uniform",
                    "no_reliability",
                    "no_delay_metadata",
                    "concat_capacity_matched",
                    "resilient_v2x",
                }
            ),
            "common_text_projection_sha256": "4" * 64,
            "common_ast_projection_sha256": "5" * 64,
            "only_text_difference": "NESTED_TEACHER_EXPERIMENTS assignment",
            "only_ast_difference": ("NESTED_TEACHER_EXPERIMENTS frozenset members"),
            "capacity_matched_hardware_contract": "capacity-matched-hardware-v1",
            "runtime_guard_evidence": "reviewed_completed_bootstrap_bytes",
            "tf32_override": "0",
            "homogeneous_gpu_count": 4,
            "allowed_compute_capabilities": [[7, 0], [8, 0], [12, 0]],
            "legacy_script_subjects": legacy_subjects,
            "expanded_script_subjects": expanded_subjects,
        },
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
                module.EXPECTED_SOURCE_TREE_SHA256: 21,
                module.EXPECTED_NEW_SOURCE_TREE_SHA256: 5,
            },
            "training_dataset_id": module.EXPECTED_TRAINING_DATASET_ID,
            "native_bundle_sha256": module.EXPECTED_NATIVE_BUNDLE_SHA256,
            "build_manifest_sha256": module.EXPECTED_BUILD_MANIFEST_SHA256,
            "teacher_checkpoint_sha256": (module.EXPECTED_TEACHER_CHECKPOINT_SHA256),
            "training_seed": module.EXPECTED_TRAINING_SEED,
            "global_batch_size": 8,
            "precision": "FP32",
            "max_epochs": 50,
            "val_interval": 10,
            "amp": False,
        },
        "capacity_matched_hardware": {
            "contract": "capacity-matched-hardware-v1",
            "docker_command_sha256": module.EXPECTED_DOCKER_COMMAND_SHA256,
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
        "recovery_parent_controllers": [],
        "training_tasks": records,
    }
    _reseal(module, document)
    assert provenance_task_id not in {item["training_task_id"] for item in training}
    return document, training, manifest


class _Artifact:
    def __init__(self, value: object, *, fail_on_get: bool = False) -> None:
        self.value = value
        self.fail_on_get = fail_on_get
        self.get_calls = 0
        self.url = "http://10.100.34.118:8081/artifacts/contract.json"
        self.size = 123
        self.hash = "b" * 64
        self.preview = (
            json.dumps(value)
            if isinstance(value, Mapping)
            else "artifact.json - downloaded JSON"
        )

    def get(self, *, force_download: bool = False) -> object:
        assert force_download is True
        self.get_calls += 1
        if self.fail_on_get:
            raise AssertionError("artifact content must not be downloaded")
        return self.value


def test_artifact_mapping_accepts_clearml_downloaded_json_path(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "common_teacher_initialization_audit.json"
    path.write_text('{"contract":"shared-only-clean-teacher-initialization-v1"}')
    artifact = _Artifact(path)

    assert module._artifact_mapping(
        artifact, "common_teacher_initialization_audit"
    ) == {"contract": "shared-only-clean-teacher-initialization-v1"}
    assert artifact.get_calls == 1


@pytest.mark.parametrize("attack", ("symlink", "hardlink", "fifo"))
def test_artifact_mapping_rejects_unsafe_downloaded_path(
    attack: str, tmp_path: Path
) -> None:
    module = _load_module()
    target = tmp_path / "target.json"
    target.write_text('{"safe":true}')
    path = tmp_path / "artifact.json"
    if attack == "symlink":
        path.symlink_to(target)
    elif attack == "hardlink":
        os.link(target, path)
    else:
        os.mkfifo(path)

    with pytest.raises(RuntimeError, match="unsafe|securely read"):
        module._artifact_mapping(_Artifact(path), "audit")


def _completed_training(
    module,
    *,
    filename: str = "epoch_50.pth",
    fail_on_get: bool = False,
):
    subject = "v2vnet"
    task_id = "1" * 32
    script_diff = "sealed bootstrap"
    model_id = "2" * 32
    model_url = "http://10.100.34.118:8081/models/v2vnet_epoch_50.pth"
    parameters = {
        "Args/experiment_from_task": subject,
        "Args/source_dataset_id": module.EXPECTED_SOURCE_DATASET_ID,
        "Args/source_archive_name": module.EXPECTED_SOURCE_ARCHIVE_NAME,
        "Args/source_archive_bytes": str(module.EXPECTED_SOURCE_ARCHIVE_BYTES),
        "Args/source_archive_sha256": module.EXPECTED_SOURCE_ARCHIVE_SHA256,
        "Args/training_dataset_id": module.EXPECTED_TRAINING_DATASET_ID,
        "Args/native_bundle_bytes": "654321",
        "Args/native_bundle_sha256": "c" * 64,
        "Args/build_manifest_sha256": "d" * 64,
        "Args/predecessor_task_id": module.EXPECTED_PREDECESSOR_TASK_ID,
        "Args/stage": "all",
        "Args/max_epochs": "50",
        "Args/gpus": "4",
        "Args/amp": "False",
    }
    run_contract = {
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": subject,
        "source_dataset_id": module.EXPECTED_SOURCE_DATASET_ID,
        "training_dataset_id": module.EXPECTED_TRAINING_DATASET_ID,
        "predecessor_task_id": module.EXPECTED_PREDECESSOR_TASK_ID,
        "gpus": 4,
        "global_batch_size": 8,
        "max_epochs": 50,
        "val_interval": 10,
        "precision": "FP32",
        "source_archive": {
            "name": module.EXPECTED_SOURCE_ARCHIVE_NAME,
            "size_bytes": module.EXPECTED_SOURCE_ARCHIVE_BYTES,
            "sha256": module.EXPECTED_SOURCE_ARCHIVE_SHA256,
        },
    }
    model_name = "ResilientV2X v2vnet final checkpoint"
    final_contract = {
        "model_id": model_id,
        "name": model_name,
        "filename": filename,
        "size_bytes": 123,
        "sha256": "a" * 64,
        "url": model_url,
    }
    model = SimpleNamespace(
        id=model_id,
        name=model_name,
        task=task_id,
        url=model_url,
    )
    run_artifact = _Artifact(run_contract, fail_on_get=fail_on_get)
    final_artifact = _Artifact(final_contract, fail_on_get=fail_on_get)
    artifact_records = [
        SimpleNamespace(
            key="run_contract",
            uri=run_artifact.url,
            content_size=run_artifact.size,
            hash=run_artifact.hash,
            type_data=SimpleNamespace(preview=run_artifact.preview),
        ),
        SimpleNamespace(
            key="final_checkpoint_contract",
            uri=final_artifact.url,
            content_size=final_artifact.size,
            hash=final_artifact.hash,
            type_data=SimpleNamespace(preview=final_artifact.preview),
        ),
    ]
    task = SimpleNamespace(
        status="completed",
        reload=lambda: None,
        get_parameters=lambda cast=False: parameters,
        data=SimpleNamespace(
            script=SimpleNamespace(
                repository="",
                working_dir=".",
                entry_point="clearml_5090_bootstrap.py",
                diff=script_diff,
            ),
            execution=SimpleNamespace(artifacts=artifact_records),
        ),
        artifacts={
            "run_contract": run_artifact,
            "final_checkpoint_contract": final_artifact,
        },
        get_models=lambda: {"input": [], "output": [model]},
    )
    return (
        task,
        task_id,
        subject,
        hashlib.sha256(script_diff.encode()).hexdigest(),
        run_artifact,
        final_artifact,
    )


def test_completed_training_accepts_local_checkpoint_filename() -> None:
    module = _load_module()
    task, task_id, subject, script_sha, run_artifact, final_artifact = (
        _completed_training(module)
    )

    assert module._require_completed_training(
        task,
        subject=subject,
        task_id=task_id,
        expected_script_sha256=script_sha,
        expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
        expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
        expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
        expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
        expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
        expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
    )
    assert run_artifact.get_calls == 1
    assert final_artifact.get_calls == 1


def test_completed_training_rejects_evaluation_script_pin() -> None:
    module = _load_module()
    task, task_id, subject, training_script_sha, _, _ = _completed_training(module)
    evaluation_script_sha = hashlib.sha256(b"sealed evaluation bootstrap").hexdigest()
    assert evaluation_script_sha != training_script_sha

    with pytest.raises(RuntimeError, match="training dependency .* script SHA-256"):
        module._require_completed_training(
            task,
            subject=subject,
            task_id=task_id,
            expected_script_sha256=evaluation_script_sha,
            expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
            expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
            expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
            expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
            expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
        )


def test_completed_training_metadata_only_does_not_download_artifacts() -> None:
    module = _load_module()
    task, task_id, subject, script_sha, run_artifact, final_artifact = (
        _completed_training(module, fail_on_get=True)
    )

    assert module._require_completed_training(
        task,
        subject=subject,
        task_id=task_id,
        expected_script_sha256=script_sha,
        expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
        expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
        expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
        expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
        expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
        expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
        metadata_only_artifact_gate=True,
    )
    assert run_artifact.get_calls == 0
    assert final_artifact.get_calls == 0


def test_completed_training_accepts_one_final_plus_clean_val_best() -> None:
    module = _load_module()
    task, task_id, subject, script_sha, _, _ = _completed_training(module)
    final_model = task.get_models()["output"][0]
    clean_val_best = SimpleNamespace(
        id="3" * 32,
        name=f"ResilientV2X {subject} clean-val best checkpoint",
        task=task_id,
        url=(f"http://10.100.34.118:8081/models/{subject}_clean_val_best.pth"),
    )
    task.get_models = lambda: {
        "input": [],
        "output": [final_model, clean_val_best],
    }

    assert module._require_completed_training(
        task,
        subject=subject,
        task_id=task_id,
        expected_script_sha256=script_sha,
        expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
        expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
        expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
        expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
        expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
        expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
    )


def test_completed_training_rejects_model_changed_after_manifest_seal() -> None:
    module = _load_module()
    task, task_id, subject, script_sha, _, _ = _completed_training(module)

    with pytest.raises(RuntimeError, match="final model ID mismatch"):
        module._require_completed_training(
            task,
            subject=subject,
            task_id=task_id,
            expected_script_sha256=script_sha,
            expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
            expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
            expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
            expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
            expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
            expected_final_model_id="9" * 32,
        )


def test_completed_training_rejects_two_matching_final_models() -> None:
    module = _load_module()
    task, task_id, subject, script_sha, _, _ = _completed_training(module)
    final_model = task.get_models()["output"][0]
    duplicate_final = SimpleNamespace(
        id="4" * 32,
        name=final_model.name,
        task=task_id,
        url=final_model.url,
    )
    task.get_models = lambda: {
        "input": [],
        "output": [final_model, duplicate_final],
    }

    with pytest.raises(RuntimeError, match="exactly one final model"):
        module._require_completed_training(
            task,
            subject=subject,
            task_id=task_id,
            expected_script_sha256=script_sha,
            expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
            expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
            expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
            expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
            expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
        )


def test_metadata_only_rejects_duplicate_contract_metadata() -> None:
    module = _load_module()
    task, task_id, subject, script_sha, _, _ = _completed_training(
        module, fail_on_get=True
    )
    task.data.execution.artifacts.append(
        SimpleNamespace(key="run_contract", uri="http://host:8081/duplicate.json")
    )

    with pytest.raises(RuntimeError, match="exactly one run_contract artifact"):
        module._require_completed_training(
            task,
            subject=subject,
            task_id=task_id,
            expected_script_sha256=script_sha,
            expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
            expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
            expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
            expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
            expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
            metadata_only_artifact_gate=True,
        )


def test_metadata_only_rejects_wrong_final_model_basename() -> None:
    module = _load_module()
    task, task_id, subject, script_sha, _, _ = _completed_training(
        module, fail_on_get=True
    )
    task.get_models()["output"][0].url = "http://10.100.34.118:8081/models/epoch_50.pth"

    with pytest.raises(RuntimeError, match="final model basename mismatch"):
        module._require_completed_training(
            task,
            subject=subject,
            task_id=task_id,
            expected_script_sha256=script_sha,
            expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
            expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
            expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
            expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
            expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
            metadata_only_artifact_gate=True,
        )


@pytest.mark.parametrize("preview", [None, "not-json"])
def test_metadata_only_rejects_missing_or_invalid_preview(preview: object) -> None:
    module = _load_module()
    task, task_id, subject, script_sha, run_artifact, _ = _completed_training(
        module, fail_on_get=True
    )
    run_artifact.preview = preview
    task.data.execution.artifacts[0].type_data.preview = preview

    with pytest.raises(RuntimeError, match="run_contract preview metadata"):
        module._require_completed_training(
            task,
            subject=subject,
            task_id=task_id,
            expected_script_sha256=script_sha,
            expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
            expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
            expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
            expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
            expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
            metadata_only_artifact_gate=True,
        )


def test_completed_training_rejects_uploaded_target_as_local_filename() -> None:
    module = _load_module()
    task, task_id, subject, script_sha, _, _ = _completed_training(
        module, filename="v2vnet_epoch_50.pth"
    )

    with pytest.raises(RuntimeError, match="final contract filename mismatch"):
        module._require_completed_training(
            task,
            subject=subject,
            task_id=task_id,
            expected_script_sha256=script_sha,
            expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
            expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
            expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
            expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
            expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
        )


def test_training_and_evaluation_script_pins_are_independent_by_default() -> None:
    module = _load_module()
    args = module._parser().parse_args(
        [
            "--dependencies-json",
            "[]",
            "--expected-eval-script-sha256",
            "a" * 64,
        ]
    )

    assert args.expected_training_script_sha256 == ""
    assert args.expected_training_source_dataset_id == ""
    assert args.expected_training_source_archive_sha256 == ""
    assert args.metadata_only_artifact_gate is False


def test_training_provenance_tags_are_an_unordered_unique_set() -> None:
    module = _load_module()
    reordered = list(reversed(module.EXPECTED_PROVENANCE_TAGS))

    observed = module._validate_provenance_tags(SimpleNamespace(tags=reordered))

    assert observed == frozenset(module.EXPECTED_PROVENANCE_TAGS)


@pytest.mark.parametrize(
    ("invalid_state", "message"),
    [
        ("duplicate", "duplicate tags"),
        ("unknown", "tags mismatch"),
    ],
)
def test_training_provenance_tags_reject_duplicates_and_unknown_labels(
    invalid_state: str,
    message: str,
) -> None:
    module = _load_module()
    tags = list(module.EXPECTED_PROVENANCE_TAGS)
    if invalid_state == "duplicate":
        tags.append(tags[0])
    else:
        tags[-1] = "unknown-third-state"

    with pytest.raises(RuntimeError, match=message):
        module._validate_provenance_tags(SimpleNamespace(tags=tags))


def _validate_provenance_document(module, document, training, manifest):
    return module._validate_training_provenance_document(
        document,
        provenance_task_id="e" * 32,
        controller_task_id="f" * 32,
        manifest=manifest,
        training=training,
    )


def test_training_provenance_accepts_only_sealed_per_subject_script_policy() -> None:
    module = _load_module()
    document, training, manifest = _formal_training_provenance_document(module)

    policies = _validate_provenance_document(module, document, training, manifest)

    assert list(policies) == list(module.FORMAL_SUBJECT_ORDER)
    assert policies["support_residual"]["script_sha256"] == (
        module.EXPECTED_LEGACY_TRAINING_SCRIPT_SHA256
    )
    assert policies["ffnet"]["script_sha256"] == (
        module.EXPECTED_CANONICAL_TRAINING_SCRIPT_SHA256
    )


def test_training_provenance_rejects_third_script_sha() -> None:
    module = _load_module()
    document, training, manifest = _formal_training_provenance_document(module)
    records = document["training_tasks"]
    assert isinstance(records, list) and isinstance(records[10], dict)
    records[10]["raw_bootstrap_script_sha256"] = "9" * 64
    _reseal(module, document)

    with pytest.raises(RuntimeError, match="third script"):
        _validate_provenance_document(module, document, training, manifest)


def test_training_provenance_rejects_legacy_script_for_wrong_subject() -> None:
    module = _load_module()
    document, training, manifest = _formal_training_provenance_document(module)
    records = document["training_tasks"]
    bootstrap = document["bootstrap_equivalence"]
    assert isinstance(records, list) and isinstance(records[10], dict)
    assert isinstance(bootstrap, dict)
    records[10]["raw_bootstrap_script_sha256"] = (
        module.EXPECTED_LEGACY_TRAINING_SCRIPT_SHA256
    )
    records[10]["script_equivalence_class"] = "legacy_nested_teacher_membership"
    bootstrap["legacy_script_subjects"].append("ffnet")
    bootstrap["expanded_script_subjects"].remove("ffnet")
    _reseal(module, document)

    with pytest.raises(RuntimeError, match="legacy training bootstrap.*ffnet"):
        _validate_provenance_document(module, document, training, manifest)


@pytest.mark.parametrize("swap", ["task", "parent"])
def test_training_provenance_rejects_task_or_parent_swap(swap: str) -> None:
    module = _load_module()
    document, training, manifest = _formal_training_provenance_document(module)
    records = document["training_tasks"]
    assert isinstance(records, list)
    assert isinstance(records[0], dict) and isinstance(records[1], dict)
    if swap == "task":
        records[0]["training_task_id"], records[1]["training_task_id"] = (
            records[1]["training_task_id"],
            records[0]["training_task_id"],
        )
    else:
        records[0]["parent_controller_task_id"] = "a" * 32
    _reseal(module, document)

    with pytest.raises(
        RuntimeError,
        match="manifest binding|parent binding|creator lineage mismatch",
    ):
        _validate_provenance_document(module, document, training, manifest)


def test_training_provenance_rejects_seal_drift() -> None:
    module = _load_module()
    document, training, manifest = _formal_training_provenance_document(module)
    document["passed"] = False

    with pytest.raises(ValueError, match="seal SHA-256 mismatch"):
        _validate_provenance_document(module, document, training, manifest)


def test_training_provenance_validates_recursive_recovery_lineage() -> None:
    module = _load_module()
    root = "f" * 32
    middle = "b" * 32
    origin = "6" * 32
    lineage = [
        {
            "controller_task_id": root,
            "decision": "adopted_from_source_progress",
            "progress_seal_sha256": "1" * 64,
            "source_controller_task_id": middle,
        },
        {
            "controller_task_id": middle,
            "decision": "adopted_from_source_progress",
            "progress_seal_sha256": "2" * 64,
            "source_controller_task_id": origin,
        },
        {
            "controller_task_id": origin,
            "decision": "controller_created",
            "progress_seal_sha256": "3" * 64,
        },
    ]

    assert (
        module._validate_training_recovery_lineage(
            lineage,
            subject="support_residual",
            controller_task_id=root,
            parent_controller_task_id=origin,
            parent_binding="recursive_source_controller_created",
            current_progress_seal_sha256="1" * 64,
        )
        == lineage
    )

    lineage[0]["source_controller_task_id"] = origin
    with pytest.raises(RuntimeError, match="lineage chain mismatch"):
        module._validate_training_recovery_lineage(
            lineage,
            subject="support_residual",
            controller_task_id=root,
            parent_controller_task_id=origin,
            parent_binding="recursive_source_controller_created",
            current_progress_seal_sha256="1" * 64,
        )


def test_training_provenance_task_must_be_completed() -> None:
    module = _load_module()
    provenance = SimpleNamespace(status="in_progress", reload=lambda: None)
    task_class = SimpleNamespace(get_task=lambda *, task_id: provenance)

    with pytest.raises(RuntimeError, match="provenance task is not completed"):
        module._consume_training_provenance(
            task_class=task_class,
            provenance_task_id="e" * 32,
            expected_provenance_script_sha256="a" * 64,
            controller=object(),
            controller_task_id="f" * 32,
            manifest={},
            training=[],
        )


def test_stable_artifact_read_rejects_toctou_content_change() -> None:
    module = _load_module()

    class ChangingArtifact(_Artifact):
        def __init__(self) -> None:
            super().__init__({"generation": 1})
            self.values = iter(({"generation": 1}, {"generation": 2}))

        def get(self, **_kwargs):
            return next(self.values)

    artifact = ChangingArtifact()
    record = SimpleNamespace(
        key="sealed",
        uri=artifact.url,
        content_size=artifact.size,
        hash=artifact.hash,
        type_data=SimpleNamespace(preview=artifact.preview),
    )
    task = SimpleNamespace(
        status="completed",
        reload=lambda: None,
        artifacts={"sealed": artifact},
        data=SimpleNamespace(execution=SimpleNamespace(artifacts=[record])),
    )

    with pytest.raises(RuntimeError, match="changed across fresh readbacks"):
        module._stable_artifact_mapping(task, "sealed", required_status="completed")


def test_run_routes_distinct_production_script_pins(monkeypatch) -> None:
    module = _load_module()
    dependency = {
        "subject": "ffnet",
        "training_task_id": "1" * 32,
        "training_predecessor_task_id": module.EXPECTED_PREDECESSOR_TASK_ID,
        "evaluation_task_id": "2" * 32,
        "queue": "GPU4-A100",
    }
    observed: dict[str, str] = {}
    current_task = SimpleNamespace(set_tags=lambda _tags: None)
    tasks = {
        dependency["training_task_id"]: object(),
        dependency["evaluation_task_id"]: object(),
    }
    task_class = SimpleNamespace(
        current_task=lambda: current_task,
        get_task=lambda *, task_id: tasks[task_id],
    )
    monkeypatch.setattr(
        module,
        "_parse_dependencies",
        lambda *_args, **_kwargs: [dependency],
    )
    monkeypatch.setattr(
        module,
        "_dependencies_in_release_order",
        lambda value: value,
    )
    monkeypatch.setattr(
        module,
        "_invariant_runtime_identity",
        lambda *_args, **_kwargs: {"sealed-runtime": "same"},
    )
    monkeypatch.setattr(
        module,
        "_source_runtime_identity",
        lambda *_args, **_kwargs: module._expected_source_runtime_identity("old"),
    )
    monkeypatch.setattr(module, "_status", lambda _task: "completed")

    def require_evaluation(_task, **kwargs):
        observed["evaluation"] = kwargs["expected_script_sha256"]
        return "completed"

    def require_training(_task, **kwargs):
        observed["training"] = kwargs["expected_script_sha256"]
        return True

    monkeypatch.setattr(module, "_require_evaluation_identity", require_evaluation)
    monkeypatch.setattr(module, "_require_completed_training", require_training)
    monkeypatch.setattr(
        module, "_require_released_input_model", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module, "_require_completed_evaluation_metrics", lambda *_args, **_kwargs: None
    )
    args = module._parser().parse_args(
        [
            "--dependencies-json",
            "sealed-plan",
            "--expected-training-script-sha256",
            PRODUCTION_TRAINING_SCRIPT_SHA256,
            "--expected-eval-script-sha256",
            PRODUCTION_EVALUATION_SCRIPT_SHA256,
        ]
    )

    module.run(args, task_class=task_class)

    assert observed == {
        "training": PRODUCTION_TRAINING_SCRIPT_SHA256,
        "evaluation": PRODUCTION_EVALUATION_SCRIPT_SHA256,
    }


def test_formal_subject_order_exactly_matches_training_registry() -> None:
    module = _load_module()
    controller = _load_controller()

    assert module.EXPECTED_SOURCE_DATASET_ID == ("4f7fac0078a4419a907fec6ff9e306c8")
    assert module.EXPECTED_SOURCE_ARCHIVE_SHA256 == (
        "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
    )
    assert module.EXPECTED_TRAINING_DATASET_ID == ("7c59fabb9da949e6b3c94c732f000975")
    assert module.EXPECTED_PREDECESSOR_TASK_ID == ("f041d43e48c14ba4a4562281860d13f6")
    assert len(module.FORMAL_SUBJECT_ORDER) == 26
    assert module.FORMAL_SUBJECT_ORDER == controller.EXPERIMENT_ORDER
    assert module.FORMAL_SUBJECT_ORDER == (
        "support_residual",
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
        "coformernet",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "concat_capacity_matched",
        "ffnet",
        "bevfusion",
        "v2x_vit",
        "cobevt",
        "linear_no_distillation",
        "no_distillation_peak_lr_3e4",
        "ego_only",
        "fcooper",
        "attfuse",
        "v2vnet",
        "when2com",
        "where2comm",
        "late_fusion",
        "disconet",
        "how2comm",
        "resilient_v2x",
    )


class _PlannerTask:
    def __init__(
        self,
        *,
        task_id: str,
        name: str,
        status: str,
        parameters: dict[str, object],
        script_diff: str,
        input_models: list[object] | None = None,
        parent: str = "",
        project: str = TRAINING_PROJECT_ID,
    ) -> None:
        self.id = task_id
        self.name = name
        self.status = status
        self.parameters = dict(parameters)
        self.parent = parent
        self.project = project
        self.input_models = list(input_models or [])
        self.removed_input_model_ids: list[str] = []
        self.data = SimpleNamespace(
            parent=parent,
            project=project,
            script=SimpleNamespace(
                repository="",
                working_dir=".",
                entry_point="clearml_5090_bootstrap.py",
                diff=script_diff,
            ),
            execution=SimpleNamespace(queue="", artifacts=[]),
        )

    def get_parameters(self, **_kwargs):
        return dict(self.parameters)

    def set_parameters(self, parameters):
        self.parameters = dict(parameters)

    def get_models(self):
        return {"input": list(self.input_models), "output": []}

    def remove_input_models(self, models_to_remove):
        model_ids = [
            model if isinstance(model, str) else model.id for model in models_to_remove
        ]
        self.removed_input_model_ids.extend(model_ids)
        self.input_models = [
            model for model in self.input_models if model.id not in set(model_ids)
        ]


class _PlannerTasks:
    def __init__(self, template: _PlannerTask) -> None:
        self.registry = {template.id: template}
        self.clone_count = 0
        self.next_id = 2000
        self.query_filters: list[dict[str, object]] = []
        self.legacy_filtered_lookup_count = 0

    def get_task(self, *, task_id: str):
        return self.registry[task_id]

    def get_tasks(self, **_kwargs):
        self.legacy_filtered_lookup_count += 1
        return []

    def query_tasks(self, *, task_filter):
        self.query_filters.append(dict(task_filter))
        project_ids = task_filter["project"]
        parent = task_filter["parent"]
        return [
            task.id
            for task in self.registry.values()
            if task.project in project_ids and task.parent == parent
        ]

    def clone(self, *, source_task, name, parent):
        self.clone_count += 1
        task = _PlannerTask(
            task_id=f"{self.next_id:032x}",
            name=name,
            status="created",
            parameters=source_task.parameters,
            script_diff=source_task.data.script.diff,
            input_models=source_task.input_models,
            parent=parent,
            project=source_task.project,
        )
        self.next_id += 1
        self.registry[task.id] = task
        return task


def _planned_task(
    *,
    task_id: str,
    name: str,
    parent: str,
    project: str = TRAINING_PROJECT_ID,
) -> _PlannerTask:
    return _PlannerTask(
        task_id=task_id,
        name=name,
        status="created",
        parameters={},
        script_diff="sealed planner template",
        parent=parent,
        project=project,
    )


def test_planned_lookup_bypasses_clearml_21_name_project_false_empty() -> None:
    module = _load_module()
    controller_id = "f" * 32
    template = _planned_task(task_id="e" * 32, name="template", parent="")
    tasks = _PlannerTasks(template)
    name = module._planned_evaluation_name(
        controller_task_id=controller_id,
        index=1,
        subject="support_residual",
    )
    existing = _planned_task(task_id="1" * 32, name=name, parent=controller_id)
    tasks.registry[existing.id] = existing

    assert (
        module._find_planned_evaluation(
            tasks,
            project_id=TRAINING_PROJECT_ID,
            name=name,
            controller_task_id=controller_id,
        )
        is existing
    )
    assert tasks.legacy_filtered_lookup_count == 0
    assert tasks.query_filters == [
        {
            "project": [TRAINING_PROJECT_ID],
            "parent": controller_id,
        }
    ]


def test_cross_controller_phase_release_truth_table() -> None:
    module = _load_module()
    assert module._formal_phase_allows_release(
        "ffnet", candidate_phase_ready=False
    )
    assert module._formal_phase_allows_release(
        "resilient_v2x", candidate_phase_ready=False
    )
    assert not module._formal_phase_allows_release(
        "ptf_none", candidate_phase_ready=False
    )
    assert module._formal_phase_allows_release(
        "ptf_none", candidate_phase_ready=True
    )


def test_formal_core_phase_status_ready_reads_authoritative_task_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "FORMAL_EVALUATION_RELEASE_PRIORITY", ("ffnet", "resilient_v2x")
    )
    template = _planned_task(task_id="e" * 32, name="template", parent="")
    tasks = _PlannerTasks(template)
    dependencies = []
    for index, subject in enumerate(("ffnet", "resilient_v2x"), start=1):
        evaluation = _planned_task(
            task_id=str(index) * 32,
            name=f"evaluation {subject}",
            parent="f" * 32,
        )
        evaluation.status = "completed" if subject == "ffnet" else "created"
        tasks.registry[evaluation.id] = evaluation
        dependencies.append(
            {"subject": subject, "evaluation_task_id": evaluation.id}
        )

    assert not module._formal_core_phase_status_ready(
        tasks, dependencies=dependencies
    )
    tasks.registry["2" * 32].status = "completed"
    assert module._formal_core_phase_status_ready(tasks, dependencies=dependencies)


def test_candidate_phase_created_e3_a100_queue_is_waiting_not_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    spec = {
        "label": "E3",
        "subject": "support_residual_no_reliability",
        "training_task_id": "d" * 32,
        "evaluation_task_id": "2" * 32,
        "queue": "GPU4-A100",
    }
    monkeypatch.setattr(module, "CANDIDATE_EVALUATION_PHASE", (spec,))
    template = _planned_task(task_id="e" * 32, name="template", parent="")
    tasks = _PlannerTasks(template)
    name = (
        "ResilientV2X formal1337 candidate eval E3 "
        f"support_residual_no_reliability [{str(spec['training_task_id'])[:12]}]"
    )
    candidate = _planned_task(
        task_id=str(spec["evaluation_task_id"]),
        name=name,
        parent=str(spec["training_task_id"]),
    )
    candidate.data.execution.queue = module.EXPECTED_QUEUE_IDS["GPU4-A100"]
    tasks.registry[candidate.id] = candidate

    snapshot, ready = module._candidate_evaluation_phase_snapshot(
        tasks,
        expected_project_id=TRAINING_PROJECT_ID,
        core_phase_ready=False,
    )

    assert ready is False
    assert snapshot["entries"] == [
        {
            "label": "E3",
            "subject": "support_residual_no_reliability",
            "training_task_id": "d" * 32,
            "evaluation_task_id": "2" * 32,
            "status": "created",
            "queue": "GPU4-A100",
            "execution_queue_id": module.EXPECTED_QUEUE_IDS["GPU4-A100"],
        }
    ]


@pytest.mark.parametrize("earlier_status", ("not_created", "created"))
@pytest.mark.parametrize("later_status", ("queued", "in_progress", "completed"))
def test_candidate_phase_allows_parallel_candidate_before_earlier_completed(
    monkeypatch: pytest.MonkeyPatch,
    earlier_status: str,
    later_status: str,
) -> None:
    module = _load_module()
    specs = (
        {
            "label": "E1",
            "subject": "support_residual_linear",
            "training_task_id": "a" * 32,
            "evaluation_task_id": "" if earlier_status == "not_created" else "1" * 32,
            "queue": "GPU4-5090",
        },
        {
            "label": "E2",
            "subject": "no_reliability_linear",
            "training_task_id": "b" * 32,
            "evaluation_task_id": "2" * 32,
            "queue": "GPU4-V100",
        },
    )
    monkeypatch.setattr(module, "CANDIDATE_EVALUATION_PHASE", specs)
    monkeypatch.setattr(
        module, "_require_completed_evaluation_metrics", lambda *_args, **_kwargs: None
    )
    template = _planned_task(task_id="e" * 32, name="template", parent="")
    tasks = _PlannerTasks(template)
    for spec, status in ((specs[0], earlier_status), (specs[1], later_status)):
        if status == "not_created":
            continue
        name = (
            f"ResilientV2X formal1337 candidate eval {spec['label']} "
            f"{spec['subject']} [{str(spec['training_task_id'])[:12]}]"
        )
        candidate = _planned_task(
            task_id=str(spec["evaluation_task_id"]),
            name=name,
            parent=str(spec["training_task_id"]),
        )
        candidate.status = status
        candidate.data.execution.queue = module.EXPECTED_QUEUE_IDS[str(spec["queue"])]
        tasks.registry[candidate.id] = candidate

    snapshot, ready = module._candidate_evaluation_phase_snapshot(
        tasks,
        expected_project_id=TRAINING_PROJECT_ID,
        core_phase_ready=True,
    )

    assert ready is False
    assert snapshot["max_active_candidate_evaluations"] == 3
    assert snapshot["active_candidate_evaluations"] == (
        0 if later_status == "completed" else 1
    )


def test_candidate_phase_rejects_wrong_execution_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    spec = {
        "label": "E3",
        "subject": "support_residual_no_reliability",
        "training_task_id": "d" * 32,
        "evaluation_task_id": "3" * 32,
        "queue": "GPU4-A100",
    }
    monkeypatch.setattr(module, "CANDIDATE_EVALUATION_PHASE", (spec,))
    template = _planned_task(task_id="e" * 32, name="template", parent="")
    tasks = _PlannerTasks(template)
    candidate = _planned_task(
        task_id="3" * 32,
        name=(
            "ResilientV2X formal1337 candidate eval E3 "
            "support_residual_no_reliability [dddddddddddd]"
        ),
        parent="d" * 32,
    )
    candidate.status = "in_progress"
    candidate.data.execution.queue = module.EXPECTED_QUEUE_IDS["GPU4-5090"]
    tasks.registry[candidate.id] = candidate

    with pytest.raises(RuntimeError, match="candidate E3 evaluation queue drifted"):
        module._candidate_evaluation_phase_snapshot(
            tasks,
            expected_project_id=TRAINING_PROJECT_ID,
            core_phase_ready=True,
        )


def test_candidate_phase_rejects_more_than_three_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    labels = ("E1", "E2", "E3", "P0")
    specs = tuple(
        {
            "label": label,
            "subject": f"subject_{label.lower()}",
            "training_task_id": str(index) * 32,
            "evaluation_task_id": chr(ord("a") + index) * 32,
            "queue": "GPU4-A100",
        }
        for index, label in enumerate(labels, start=1)
    )
    monkeypatch.setattr(module, "CANDIDATE_EVALUATION_PHASE", specs)
    template = _planned_task(task_id="e" * 32, name="template", parent="")
    tasks = _PlannerTasks(template)
    for spec in specs:
        candidate = _planned_task(
            task_id=str(spec["evaluation_task_id"]),
            name=(
                f"ResilientV2X formal1337 candidate eval {spec['label']} "
                f"{spec['subject']} [{str(spec['training_task_id'])[:12]}]"
            ),
            parent=str(spec["training_task_id"]),
        )
        candidate.status = "in_progress"
        candidate.data.execution.queue = module.EXPECTED_QUEUE_IDS["GPU4-A100"]
        tasks.registry[candidate.id] = candidate

    with pytest.raises(RuntimeError, match="active concurrency exceeds 3"):
        module._candidate_evaluation_phase_snapshot(
            tasks,
            expected_project_id=TRAINING_PROJECT_ID,
            core_phase_ready=True,
        )


@pytest.mark.parametrize("status", ("queued", "in_progress", "completed"))
def test_candidate_phase_rejects_active_candidate_before_formal_core(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    module = _load_module()
    spec = {
        "label": "E1",
        "subject": "support_residual_linear",
        "training_task_id": "a" * 32,
        "evaluation_task_id": "1" * 32,
        "queue": "GPU4-5090",
    }
    monkeypatch.setattr(module, "CANDIDATE_EVALUATION_PHASE", (spec,))
    template = _planned_task(task_id="e" * 32, name="template", parent="")
    tasks = _PlannerTasks(template)
    name = (
        "ResilientV2X formal1337 candidate eval E1 support_residual_linear "
        f"[{str(spec['training_task_id'])[:12]}]"
    )
    candidate = _planned_task(
        task_id=str(spec["evaluation_task_id"]),
        name=name,
        parent=str(spec["training_task_id"]),
    )
    candidate.status = status
    candidate.data.execution.queue = module.EXPECTED_QUEUE_IDS["GPU4-5090"]
    tasks.registry[candidate.id] = candidate

    with pytest.raises(RuntimeError, match="escaped the incomplete formal core phase"):
        module._candidate_evaluation_phase_snapshot(
            tasks,
            expected_project_id=TRAINING_PROJECT_ID,
            core_phase_ready=False,
        )


def test_planned_lookup_rejects_duplicate_exact_name_parent_project() -> None:
    module = _load_module()
    controller_id = "f" * 32
    template = _planned_task(task_id="e" * 32, name="template", parent="")
    tasks = _PlannerTasks(template)
    name = module._planned_evaluation_name(
        controller_task_id=controller_id,
        index=1,
        subject="support_residual",
    )
    for task_id in ("1" * 32, "2" * 32):
        task = _planned_task(task_id=task_id, name=name, parent=controller_id)
        tasks.registry[task.id] = task

    with pytest.raises(RuntimeError, match="duplicate planned evaluation task"):
        module._find_planned_evaluation(
            tasks,
            project_id=TRAINING_PROJECT_ID,
            name=name,
            controller_task_id=controller_id,
        )
    assert tasks.clone_count == 0


@pytest.mark.parametrize("scope_drift", ("parent", "project"))
def test_planned_lookup_rejects_server_scope_drift(scope_drift: str) -> None:
    module = _load_module()
    controller_id = "f" * 32
    template = _planned_task(task_id="e" * 32, name="template", parent="")
    tasks = _PlannerTasks(template)
    name = module._planned_evaluation_name(
        controller_task_id=controller_id,
        index=1,
        subject="support_residual",
    )
    leaked = _planned_task(
        task_id="1" * 32,
        name=name,
        parent="0" * 32 if scope_drift == "parent" else controller_id,
        project="0" * 32 if scope_drift == "project" else TRAINING_PROJECT_ID,
    )
    tasks.registry[leaked.id] = leaked
    tasks.query_tasks = lambda *, task_filter: [leaked.id]

    with pytest.raises(RuntimeError, match="escaped its project/parent scope"):
        module._find_planned_evaluation(
            tasks,
            project_id=TRAINING_PROJECT_ID,
            name=name,
            controller_task_id=controller_id,
        )


def test_task_project_id_rejects_conflicting_clearml_views() -> None:
    module = _load_module()
    task = _planned_task(task_id="1" * 32, name="task", parent="f" * 32)
    assert module._task_project_id(task, context="test task") == TRAINING_PROJECT_ID
    task.data.project = "0" * 32

    with pytest.raises(RuntimeError, match="project ID drifted"):
        module._task_project_id(task, context="test task")


def test_input_model_inventory_accepts_clearml_models_list_semantics() -> None:
    module = _load_module()
    teacher_input = SimpleNamespace(
        id="d962f6bae8474260b54e170a7a5f0418",
        name="ResilientV2X clean teacher",
        task="487dab2664a8485fa0cc7c4e2a0c3df8",
    )
    clearml_models_list = UserList([teacher_input])
    task = SimpleNamespace(
        get_models=lambda: {"input": clearml_models_list, "output": UserList()}
    )

    assert module._input_model_inventory(task, context="evaluation template") == [
        {
            "id": teacher_input.id,
            "name": teacher_input.name,
            "task": teacher_input.task,
        }
    ]

    task.get_models = lambda: {"input": teacher_input.id, "output": UserList()}
    with pytest.raises(RuntimeError, match="invalid input models"):
        module._input_model_inventory(task, context="evaluation template")


def _planned_runtime_contract_case(
    module: object,
    *,
    status: str,
    input_models: list[object],
    parameter_updates: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, str], dict[str, object]]:
    training_item = {
        "subject": "ffnet",
        "training_task_id": "1" * 32,
        "training_model_id": "2" * 32,
    }
    expected_parameters: dict[str, object] = {
        "Args/stage": "baseline_validate",
        "Args/controlled_baseline": "ffnet",
    }
    parameters = dict(expected_parameters)
    if status != "created":
        parameters.update(module.EVALUATION_BOOTSTRAP_RUNTIME_DEFAULT_PARAMETERS)
    parameters.update(parameter_updates or {})
    task = _PlannerTask(
        task_id="3" * 32,
        name="planned ffnet evaluation",
        status=status,
        parameters=parameters,
        script_diff="sealed evaluation bootstrap",
        input_models=input_models,
    )
    return task, training_item, expected_parameters


@pytest.mark.parametrize(
    ("status", "attach_expected_model"),
    (
        ("created", False),
        ("queued", False),
        ("in_progress", False),
        ("in_progress", True),
        ("completed", True),
    ),
)
def test_planned_evaluation_runtime_contract_accepts_restart_safe_states(
    status: str, attach_expected_model: bool
) -> None:
    module = _load_module()
    expected_model = SimpleNamespace(
        id="2" * 32,
        name="ResilientV2X ffnet final checkpoint",
        task="1" * 32,
    )
    task, training_item, expected_parameters = _planned_runtime_contract_case(
        module,
        status=status,
        input_models=[expected_model] if attach_expected_model else [],
    )

    module._require_planned_evaluation_runtime_contract(
        task,
        training_item=training_item,
        expected_parameters=expected_parameters,
        status=status,
    )


@pytest.mark.parametrize(
    ("status", "parameter_updates"),
    (
        ("created", {"Args/teacher_checkpoint": ""}),
        ("queued", {"Args/teacher_checkpoint": "unexpected"}),
        ("in_progress", {"Args/allow_failed_teacher_task": "True"}),
        ("completed", {"Args/unexpected_runtime_default": ""}),
    ),
)
def test_planned_evaluation_runtime_contract_rejects_parameter_drift(
    status: str, parameter_updates: Mapping[str, object]
) -> None:
    module = _load_module()
    expected_model = SimpleNamespace(
        id="2" * 32,
        name="ResilientV2X ffnet final checkpoint",
        task="1" * 32,
    )
    task, training_item, expected_parameters = _planned_runtime_contract_case(
        module,
        status=status,
        input_models=[expected_model] if status == "completed" else [],
        parameter_updates=parameter_updates,
    )

    with pytest.raises(RuntimeError, match="parameter"):
        module._require_planned_evaluation_runtime_contract(
            task,
            training_item=training_item,
            expected_parameters=expected_parameters,
            status=status,
        )


@pytest.mark.parametrize(
    ("status", "model_id", "model_name", "model_task", "extra_model", "empty"),
    (
        ("queued", "2" * 32, "ResilientV2X ffnet final checkpoint", "1" * 32, False, False),
        ("in_progress", "4" * 32, "ResilientV2X ffnet final checkpoint", "1" * 32, False, False),
        ("in_progress", "2" * 32, "wrong name", "1" * 32, False, False),
        ("in_progress", "2" * 32, "ResilientV2X ffnet final checkpoint", "4" * 32, False, False),
        ("in_progress", "2" * 32, "ResilientV2X ffnet final checkpoint", "1" * 32, True, False),
        ("completed", "2" * 32, "ResilientV2X ffnet final checkpoint", "1" * 32, False, True),
    ),
)
def test_planned_evaluation_runtime_contract_rejects_input_model_drift(
    status: str,
    model_id: str,
    model_name: str,
    model_task: str,
    extra_model: bool,
    empty: bool,
) -> None:
    module = _load_module()
    models = [] if empty else [SimpleNamespace(id=model_id, name=model_name, task=model_task)]
    if extra_model:
        models.append(SimpleNamespace(id="5" * 32, name="extra", task="6" * 32))
    task, training_item, expected_parameters = _planned_runtime_contract_case(
        module, status=status, input_models=models
    )

    with pytest.raises(RuntimeError, match="input model contract drifted"):
        module._require_planned_evaluation_runtime_contract(
            task,
            training_item=training_item,
            expected_parameters=expected_parameters,
            status=status,
        )


def test_planner_creates_26_pinned_tasks_idempotently_and_rejects_drift() -> None:
    module = _load_module()
    manifest = _formal_training_manifest()
    training = module._parse_training_manifest(json.dumps(manifest))
    script_diff = "sealed planner template"
    template_id = "e" * 32
    teacher_input = SimpleNamespace(
        id="d962f6bae8474260b54e170a7a5f0418",
        name="ResilientV2X clean teacher",
        task="487dab2664a8485fa0cc7c4e2a0c3df8",
    )
    template = _PlannerTask(
        task_id=template_id,
        name="formal eval template",
        status="completed",
        parameters={
            "Args/source_dataset_id": module.EXPECTED_SOURCE_DATASET_ID,
            "Args/source_archive_name": module.EXPECTED_SOURCE_ARCHIVE_NAME,
            "Args/source_archive_bytes": str(module.EXPECTED_SOURCE_ARCHIVE_BYTES),
            "Args/source_archive_sha256": module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            "Args/training_dataset_id": module.EXPECTED_TRAINING_DATASET_ID,
            "Args/native_bundle_bytes": "456",
            "Args/native_bundle_sha256": "c" * 64,
            "Args/build_manifest_sha256": "d" * 64,
            "Args/training_seed": "20250218",
            "Args/teacher_task_id": "1" * 32,
            "Args/teacher_model_id": "2" * 32,
            "Args/teacher_checkpoint_sha256": "a" * 64,
        },
        script_diff=script_diff,
        input_models=[teacher_input],
    )
    tasks = _PlannerTasks(template)
    queues = ["GPU4-A100", "GPU4-A100", "GPU4-V100", "GPU4-5090"]
    kwargs = {
        "task_class": tasks,
        "controller_task_id": "f" * 32,
        "template_task_id": template_id,
        "expected_project_id": TRAINING_PROJECT_ID,
        "training": training,
        "worker_queues": queues,
        "expected_script_sha256": hashlib.sha256(script_diff.encode()).hexdigest(),
    }
    with pytest.raises(RuntimeError, match="evaluation template script SHA-256"):
        module.create_or_validate_evaluation_plan(
            **{**kwargs, "expected_script_sha256": "a" * 64}
        )

    plan = module.create_or_validate_evaluation_plan(**kwargs)
    assert len(plan) == 26
    assert tasks.clone_count == 26
    assert all(
        task.removed_input_model_ids == [teacher_input.id]
        and task.get_models()["input"] == []
        for task_id, task in tasks.registry.items()
        if task_id != template_id
    )
    assert [item["queue"] for item in plan[:4]] == queues
    first = tasks.registry[plan[0]["evaluation_task_id"]]
    assert (
        first.parameters["Args/controlled_baseline_model_id"]
        == (training[0]["training_model_id"])
    )
    assert (
        first.parameters["Args/controlled_baseline_checkpoint_sha256"]
        == (training[0]["training_checkpoint_sha256"])
    )
    assert not any(key.startswith("Args/teacher_") for key in first.parameters)
    assert "Args/training_seed" not in first.parameters

    assert module.create_or_validate_evaluation_plan(**kwargs) == plan
    assert tasks.clone_count == 26
    assert tasks.legacy_filtered_lookup_count == 0
    assert tasks.query_filters
    assert all(
        task_filter
        == {
            "project": [TRAINING_PROJECT_ID],
            "parent": kwargs["controller_task_id"],
        }
        for task_filter in tasks.query_filters
    )
    first.data.script.diff = "drifted evaluation child"
    with pytest.raises(RuntimeError, match="evaluation .* script SHA mismatch"):
        module.create_or_validate_evaluation_plan(**kwargs)
    first.data.script.diff = script_diff
    first.input_models = [teacher_input]
    with pytest.raises(RuntimeError, match="planned evaluation .* input model"):
        module.create_or_validate_evaluation_plan(**kwargs)
    first.input_models = []
    first.parameters["Args/controlled_baseline_checkpoint_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="parameter drift"):
        module.create_or_validate_evaluation_plan(**kwargs)


def test_controller_mode_consumes_provenance_before_any_evaluation_clone(
    monkeypatch,
) -> None:
    module = _load_module()
    controller_id = "f" * 32
    provenance_id = "e" * 32
    template_id = "d" * 32
    template = _PlannerTask(
        task_id=template_id,
        name="formal eval template",
        status="completed",
        parameters={},
        script_diff="canonical evaluation template",
    )
    controller = SimpleNamespace(status="completed", reload=lambda: None)
    current = SimpleNamespace(set_tags=lambda _tags: None)
    task_class = SimpleNamespace(
        current_task=lambda: current,
        get_task=lambda *, task_id: {
            controller_id: controller,
            template_id: template,
        }[task_id],
    )
    manifest = _formal_training_manifest()
    observed = {"planner_called": False}
    monkeypatch.setattr(
        module,
        "_wait_for_controller_manifest",
        lambda *_args, **_kwargs: manifest,
    )

    def reject_provenance(**_kwargs):
        raise RuntimeError("sealed provenance rejected")

    def planner_must_not_run(**_kwargs):
        observed["planner_called"] = True
        raise AssertionError("evaluation planner ran before provenance")

    monkeypatch.setattr(module, "_consume_training_provenance", reject_provenance)
    monkeypatch.setattr(
        module, "create_or_validate_evaluation_plan", planner_must_not_run
    )
    args = module._parser().parse_args(
        [
            "--training-controller-task-id",
            controller_id,
            "--evaluation-template-task-id",
            template_id,
            "--training-provenance-task-id",
            provenance_id,
            "--expected-training-provenance-script-sha256",
            "a" * 64,
        ]
    )

    with pytest.raises(RuntimeError, match="sealed provenance rejected"):
        module.run(args, task_class=task_class)
    assert observed["planner_called"] is False


def test_dependency_plan_joins_all_26_training_and_evaluation_tasks() -> None:
    module = _load_module()
    manifest = _formal_training_manifest()
    evaluations = _evaluation_plan(module)
    assert manifest["evaluation_release_semantics"] == (
        "formal_manifest_after_full_training_suite_completion"
    )

    dependencies = module.build_dependency_plan(
        json.dumps(manifest), json.dumps(evaluations)
    )

    assert [item["subject"] for item in dependencies] == list(
        module.FORMAL_SUBJECT_ORDER
    )
    assert len({item["training_task_id"] for item in dependencies}) == 26
    assert len({item["evaluation_task_id"] for item in dependencies}) == 26
    assert all(item["training_predecessor_task_id"] for item in dependencies)
    assert module._parse_dependencies(json.dumps(dependencies)) == dependencies


def test_formal_evaluation_release_prioritizes_five_baselines_and_primary() -> None:
    module = _load_module()
    dependencies = module.build_dependency_plan(
        json.dumps(_formal_training_manifest()),
        json.dumps(_evaluation_plan(module)),
    )

    released = module._dependencies_in_release_order(dependencies)

    assert [item["subject"] for item in released[:6]] == [
        "ffnet",
        "coformernet",
        "v2x_vit",
        "cobevt",
        "bevfusion",
        "resilient_v2x",
    ]
    assert len(released) == 26
    assert {item["subject"] for item in released} == set(module.FORMAL_SUBJECT_ORDER)
    assert [item["subject"] for item in dependencies] == list(
        module.FORMAL_SUBJECT_ORDER
    )


def _gpu_worker(worker_id: str, queue_id: str) -> dict[str, object]:
    return {"id": worker_id, "queues": [{"id": queue_id}]}


def _gpu_telemetry(
    module: object,
    worker_id: str,
    *,
    memory_mib: float = 512.0,
    usage_percent: float = 0.0,
) -> dict[str, dict[str, object]]:
    return {
        worker_id: {
            "worker_id": worker_id,
            "window_from_unix": 1000.0,
            "window_to_unix": 1180.0,
            "interval_seconds": module.SCHEDULER_GPU_TELEMETRY_INTERVAL_SECONDS,
            "gpu_memory_used_mib": [memory_mib],
            "gpu_usage_percent": [usage_percent],
            "max_gpu_memory_used_mib": memory_mib,
            "max_gpu_usage_percent": usage_percent,
        }
    }


def test_formal_resource_gate_blocks_an_overlapping_physical_worker() -> None:
    module = _load_module()
    target = module.EXPECTED_QUEUE_IDS
    gpu8_a100_queue_id = "9" * 32
    external_task_id = "8" * 32
    external = SimpleNamespace(
        id=external_task_id,
        status="in_progress",
        data=SimpleNamespace(
            execution=SimpleNamespace(queue=gpu8_a100_queue_id),
            last_worker="10.0.0.1-A100:gpu0,1,2,3,4,5,6,7",
        ),
    )
    task_class = SimpleNamespace(
        query_tasks=lambda *, task_filter: [external_task_id]
        if task_filter == {"status": ["in_progress", "queued"]}
        else None,
        get_task=lambda *, task_id: external
        if task_id == external_task_id
        else None,
    )
    workers = [
        _gpu_worker("10.0.0.1-A100:gpu0,1,2,3", target["GPU4-A100"]),
        _gpu_worker("10.0.0.2-V100:gpu0,1,2,3", target["GPU4-V100"]),
        _gpu_worker("10.0.0.3-5090:gpu0,1,2,3", target["GPU4-5090"]),
        _gpu_worker(
            "10.0.0.1-A100:gpu0,1,2,3,4,5,6,7", gpu8_a100_queue_id
        ),
    ]

    snapshot = module._formal_resource_gate_snapshot(
        task_class,
        formal_evaluation_task_ids=set(),
        target_queue_id=target["GPU4-A100"],
        worker_rows=workers,
    )

    assert snapshot["ready"] is False
    assert snapshot["external_blockers"] == [
        {
            "task_id": external_task_id,
            "status": "in_progress",
            "queue_id": gpu8_a100_queue_id,
            "last_worker": "10.0.0.1-A100:gpu0,1,2,3,4,5,6,7",
            "reasons": ["overlapping_active_worker"],
        }
    ]


def test_formal_resource_gate_fails_closed_without_every_target_worker() -> None:
    module = _load_module()
    target = module.EXPECTED_QUEUE_IDS
    task_class = SimpleNamespace(query_tasks=lambda **_kwargs: [])

    with pytest.raises(RuntimeError, match="no auditable live GPU worker"):
        module._formal_resource_gate_snapshot(
            task_class,
            formal_evaluation_task_ids=set(),
            target_queue_id=target["GPU4-A100"],
            worker_rows=[
                _gpu_worker(
                    "10.0.0.2-V100:gpu0,1,2,3", target["GPU4-V100"]
                )
            ],
        )


def test_formal_same_queue_active_disjoint_gpu_does_not_block_idle_worker() -> None:
    module = _load_module()
    queue_id = module.EXPECTED_QUEUE_IDS["GPU4-A100"]
    external_task_id = "7" * 32
    external = SimpleNamespace(
        id=external_task_id,
        status="in_progress",
        data=SimpleNamespace(
            execution=SimpleNamespace(queue=queue_id),
            last_worker="10.0.0.1-A100:gpu0,1,2,3",
        ),
    )
    task_class = SimpleNamespace(
        query_tasks=lambda *, task_filter: [external_task_id]
        if task_filter == {"status": ["in_progress", "queued"]}
        else None,
        get_task=lambda *, task_id: external
        if task_id == external_task_id
        else None,
    )

    idle_worker = "10.0.0.1-A100:gpu4,5,6,7"
    snapshot = module._formal_resource_gate_snapshot(
        task_class,
        formal_evaluation_task_ids=set(),
        target_queue_id=queue_id,
        worker_rows=[
            _gpu_worker("10.0.0.1-A100:gpu0,1,2,3", queue_id),
            _gpu_worker("10.0.0.1-A100:gpu4,5,6,7", queue_id),
        ],
        worker_telemetry=_gpu_telemetry(module, idle_worker),
    )

    assert snapshot["ready"] is True
    assert snapshot["selected_idle_target_worker_id"] == (
        "10.0.0.1-A100:gpu4,5,6,7"
    )
    assert snapshot["external_blockers"] == []
    assert snapshot["occupied_target_worker_ids"] == [
        "10.0.0.1-A100:gpu0,1,2,3"
    ]
    assert snapshot["live_disjoint_worker_count"] == 2
    assert snapshot["active_target_worker_count"] == 1


def test_formal_unregistered_gpu_use_blocks_whole_unaddressable_queue() -> None:
    module = _load_module()
    queue_id = module.EXPECTED_QUEUE_IDS["GPU4-V100"]
    worker_0 = "10.0.0.2-V100:gpu0,1,2,3"
    worker_4 = "10.0.0.2-V100:gpu4,5,6,7"
    task_class = SimpleNamespace(query_tasks=lambda **_kwargs: [])
    telemetry = {
        **_gpu_telemetry(module, worker_0),
        **_gpu_telemetry(module, worker_4, memory_mib=2048.0),
    }

    snapshot = module._formal_resource_gate_snapshot(
        task_class,
        formal_evaluation_task_ids=set(),
        target_queue_id=queue_id,
        worker_rows=[
            _gpu_worker(worker_0, queue_id),
            _gpu_worker(worker_4, queue_id),
        ],
        worker_telemetry=telemetry,
    )

    assert snapshot["ready"] is False
    assert snapshot["selected_idle_target_worker_id"] is None
    assert snapshot["external_blockers"] == [
        {
            "task_id": None,
            "status": "unregistered_gpu_occupancy",
            "queue_id": queue_id,
            "last_worker": worker_4,
            "reasons": ["fresh_gpu_telemetry_not_idle"],
        }
    ]


def test_formal_active_worker_is_excluded_then_remaining_worker_is_gated() -> None:
    module = _load_module()
    queue_id = module.EXPECTED_QUEUE_IDS["GPU4-A100"]
    worker_0 = "10.0.0.1-A100:gpu0,1,2,3"
    worker_4 = "10.0.0.1-A100:gpu4,5,6,7"
    active_id = "5" * 32
    active = SimpleNamespace(
        id=active_id,
        status="in_progress",
        data=SimpleNamespace(
            execution=SimpleNamespace(queue=queue_id),
            last_worker=worker_0,
        ),
    )
    task_class = SimpleNamespace(
        query_tasks=lambda **_kwargs: [active_id],
        get_task=lambda *, task_id: active if task_id == active_id else None,
    )

    snapshot = module._formal_resource_gate_snapshot(
        task_class,
        formal_evaluation_task_ids={active_id},
        target_queue_id=queue_id,
        worker_rows=[
            _gpu_worker(worker_0, queue_id),
            _gpu_worker(worker_4, queue_id),
        ],
        worker_telemetry=_gpu_telemetry(
            module,
            worker_4,
            usage_percent=2.0,
        ),
    )

    assert snapshot["ready"] is False
    assert snapshot["occupied_target_worker_ids"] == [worker_0]
    assert snapshot["pullable_target_worker_ids"] == [worker_4]
    assert snapshot["managed_active_evaluation_count"] == 1


def test_formal_worker_capacity_and_single_release_claim_are_fail_closed() -> None:
    module = _load_module()
    queue_id = module.EXPECTED_QUEUE_IDS["GPU4-A100"]
    workers = [
        _gpu_worker("10.0.0.1-A100:gpu0,1,2,3", queue_id),
        _gpu_worker("10.0.0.1-A100:gpu4,5,6,7", queue_id),
    ]
    assert (
        module._live_disjoint_worker_capacity(
            workers,
            target_queue_id=queue_id,
        )
        == 2
    )
    claimed: set[str] = set()
    assert module._claim_single_queue_release(
        claimed, queue_name="GPU4-A100"
    )
    assert not module._claim_single_queue_release(
        claimed, queue_name="GPU4-A100"
    )
    assert module._claim_single_queue_release(
        claimed, queue_name="GPU4-V100"
    )

    with pytest.raises(RuntimeError, match="resources overlap"):
        module._live_disjoint_worker_capacity(
            [
                _gpu_worker("10.0.0.1-A100:gpu0,1,2,3", queue_id),
                _gpu_worker("10.0.0.1-A100:gpu0,1,2,3,4,5,6,7", queue_id),
            ],
            target_queue_id=queue_id,
        )


def test_formal_scheduler_queries_fresh_gpu_stats_and_accepts_idle_boundary() -> None:
    module = _load_module()
    worker_id = "10.0.0.3-5090:gpu0,1,2,3"
    observed: dict[str, object] = {}

    class Workers:
        @staticmethod
        def get_stats(**kwargs: object) -> dict[str, object]:
            observed.update(kwargs)
            return {
                "workers": [
                    {
                        "worker": worker_id,
                        "metrics": [
                            {
                                "metric": "gpu_memory_used",
                                "stats": [
                                    {
                                        "aggregation": "avg",
                                        "values": [1024.0],
                                    }
                                ],
                            },
                            {
                                "metric": "gpu_usage",
                                "stats": [
                                    {
                                        "aggregation": "avg",
                                        "values": [1.0],
                                    }
                                ],
                            },
                        ],
                    }
                ]
            }

    record = module._scheduler_worker_gpu_telemetry(
        SimpleNamespace(workers=Workers()),
        worker_id,
    )

    assert observed["worker_ids"] == [worker_id]
    assert observed["interval"] == module.SCHEDULER_GPU_TELEMETRY_INTERVAL_SECONDS
    assert observed["split_by_variant"] is True
    assert record["max_gpu_memory_used_mib"] == 1024.0
    assert record["max_gpu_usage_percent"] == 1.0

    queue_id = module.EXPECTED_QUEUE_IDS["GPU4-5090"]
    snapshot = module._formal_resource_gate_snapshot(
        SimpleNamespace(query_tasks=lambda **_kwargs: []),
        formal_evaluation_task_ids=set(),
        target_queue_id=queue_id,
        worker_rows=[_gpu_worker(worker_id, queue_id)],
        worker_telemetry={worker_id: record},
    )
    assert snapshot["ready"] is True


def test_formal_scheduler_telemetry_inventory_drift_fails_closed() -> None:
    module = _load_module()
    queue_id = module.EXPECTED_QUEUE_IDS["GPU4-A100"]
    worker_id = "10.0.0.1-A100:gpu4,5,6,7"

    with pytest.raises(RuntimeError, match="telemetry inventory drifted"):
        module._formal_resource_gate_snapshot(
            SimpleNamespace(query_tasks=lambda **_kwargs: []),
            formal_evaluation_task_ids=set(),
            target_queue_id=queue_id,
            worker_rows=[_gpu_worker(worker_id, queue_id)],
            worker_telemetry={},
        )


def test_formal_same_queue_queued_task_blocks_even_with_idle_worker() -> None:
    module = _load_module()
    queue_id = module.EXPECTED_QUEUE_IDS["GPU4-A100"]
    external_task_id = "6" * 32
    external = SimpleNamespace(
        id=external_task_id,
        status="queued",
        data=SimpleNamespace(
            execution=SimpleNamespace(queue=queue_id), last_worker=None
        ),
    )
    task_class = SimpleNamespace(
        query_tasks=lambda **_kwargs: [external_task_id],
        get_task=lambda *, task_id: external
        if task_id == external_task_id
        else None,
    )

    snapshot = module._formal_resource_gate_snapshot(
        task_class,
        formal_evaluation_task_ids=set(),
        target_queue_id=queue_id,
        worker_rows=[
            _gpu_worker("10.0.0.1-A100:gpu4,5,6,7", queue_id),
        ],
    )

    assert snapshot["ready"] is False
    assert snapshot["external_blockers"][0]["reasons"] == [
        "target_queue_waiting_task"
    ]


@pytest.mark.parametrize("raw_queue_id", [None, "transient-invalid-queue"])
def test_formal_active_task_with_transient_queue_uses_physical_worker(
    raw_queue_id: str | None,
) -> None:
    module = _load_module()
    queue_id = module.EXPECTED_QUEUE_IDS["GPU4-A100"]
    worker_0 = "10.0.0.1-A100:gpu0,1,2,3"
    worker_4 = "10.0.0.1-A100:gpu4,5,6,7"
    active_id = "4" * 32
    active = SimpleNamespace(
        id=active_id,
        status="in_progress",
        data=SimpleNamespace(
            execution=SimpleNamespace(queue=raw_queue_id),
            last_worker=worker_0,
        ),
    )
    task_class = SimpleNamespace(
        query_tasks=lambda **_kwargs: [active_id],
        get_task=lambda *, task_id: active if task_id == active_id else None,
    )

    snapshot = module._formal_resource_gate_snapshot(
        task_class,
        formal_evaluation_task_ids=set(),
        target_queue_id=queue_id,
        worker_rows=[
            _gpu_worker(worker_0, queue_id),
            _gpu_worker(worker_4, queue_id),
        ],
        worker_telemetry=_gpu_telemetry(module, worker_4),
    )

    assert snapshot["ready"] is True
    assert snapshot["occupied_target_worker_ids"] == [worker_0]
    assert snapshot["selected_idle_target_worker_id"] == worker_4
    assert snapshot["observed_overlapping_active_tasks"] == [
        {
            "task_id": active_id,
            "status": "in_progress",
            "queue_id": None,
            "last_worker": worker_0,
            "active_resource": worker_0,
            "overlapping_target_workers": [worker_0],
        }
    ]


@pytest.mark.parametrize("raw_queue_id", [None, "transient-invalid-queue"])
def test_formal_unassigned_queued_task_does_not_block_idle_worker(
    raw_queue_id: str | None,
) -> None:
    module = _load_module()
    queue_id = module.EXPECTED_QUEUE_IDS["GPU4-A100"]
    worker_id = "10.0.0.1-A100:gpu0,1,2,3"
    queued_id = "3" * 32
    queued = SimpleNamespace(
        id=queued_id,
        status="queued",
        data=SimpleNamespace(
            execution=SimpleNamespace(queue=raw_queue_id),
            last_worker=None,
        ),
    )
    task_class = SimpleNamespace(
        query_tasks=lambda **_kwargs: [queued_id],
        get_task=lambda *, task_id: queued if task_id == queued_id else None,
    )

    snapshot = module._formal_resource_gate_snapshot(
        task_class,
        formal_evaluation_task_ids=set(),
        target_queue_id=queue_id,
        worker_rows=[_gpu_worker(worker_id, queue_id)],
        worker_telemetry=_gpu_telemetry(module, worker_id),
    )

    assert snapshot["ready"] is True
    assert snapshot["selected_idle_target_worker_id"] == worker_id
    assert snapshot["external_blockers"] == []


def test_dependency_plan_accepts_only_completed_controller_summary() -> None:
    module = _load_module()
    summary = {
        "summary_type": "resilient_v2x_post_main_sequential_training",
        "status": "completed",
        "experiment_order": list(module.FORMAL_SUBJECT_ORDER),
        "task_count": 26,
        "formal_1337_manifest_artifact": "formal_1337_training_manifest",
        "formal_1337_evaluation_manifest": _formal_training_manifest(),
    }
    _reseal(module, summary)
    assert (
        len(
            module.build_dependency_plan(
                json.dumps(summary), json.dumps(_evaluation_plan(module))
            )
        )
        == 26
    )

    summary["status"] = "running"
    _reseal(module, summary)
    with pytest.raises(ValueError, match="summary status mismatch"):
        module.build_dependency_plan(
            json.dumps(summary), json.dumps(_evaluation_plan(module))
        )


@pytest.mark.parametrize("missing_from", ["training", "evaluation"])
def test_dependency_plan_rejects_any_missing_formal_subject(
    missing_from: str,
) -> None:
    module = _load_module()
    manifest = _formal_training_manifest()
    evaluations = _evaluation_plan(module)
    if missing_from == "training":
        entries = manifest["entries"]
        assert isinstance(entries, list)
        entries.pop()
        _reseal(module, manifest)
    else:
        evaluations.pop()

    with pytest.raises(ValueError, match="entry count|exactly 26"):
        module.build_dependency_plan(json.dumps(manifest), json.dumps(evaluations))


def test_dependency_plan_rejects_duplicate_training_or_evaluation_task() -> None:
    module = _load_module()
    manifest = _formal_training_manifest()
    entries = manifest["entries"]
    assert isinstance(entries, list)
    assert isinstance(entries[0], dict) and isinstance(entries[1], dict)
    entries[1]["training_task_id"] = entries[0]["training_task_id"]
    _reseal(module, manifest)
    evaluations = _evaluation_plan(module)

    with pytest.raises(ValueError, match="training task IDs must be unique"):
        module.build_dependency_plan(json.dumps(manifest), json.dumps(evaluations))

    manifest = _formal_training_manifest()
    evaluations[1]["evaluation_task_id"] = evaluations[0]["evaluation_task_id"]
    with pytest.raises(ValueError, match="evaluation task IDs must be unique"):
        module.build_dependency_plan(json.dumps(manifest), json.dumps(evaluations))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_id", "DAIR-CAUSAL-OTHER", "protocol_id mismatch"),
        ("sample_count", 1336, "sample_count mismatch"),
        ("run_count", 11, "run_count mismatch"),
        ("checkpoint_policy", "best", "checkpoint_policy mismatch"),
    ],
)
def test_dependency_plan_rejects_noncanonical_1337_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    module = _load_module()
    manifest = _formal_training_manifest()
    manifest[field] = value
    _reseal(module, manifest)

    with pytest.raises(ValueError, match=message):
        module.build_dependency_plan(
            json.dumps(manifest), json.dumps(_evaluation_plan(module))
        )


def test_dependency_plan_rejects_nonfinal_checkpoint() -> None:
    module = _load_module()
    manifest = _formal_training_manifest()
    entries = manifest["entries"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    entries[0]["checkpoint_filename"] = "best_epoch_40.pth"
    _reseal(module, manifest)

    with pytest.raises(ValueError, match="checkpoint policy mismatch"):
        module.build_dependency_plan(
            json.dumps(manifest), json.dumps(_evaluation_plan(module))
        )


def test_dependency_plan_rejects_unsupported_or_reordered_queue_entry() -> None:
    module = _load_module()
    evaluations = _evaluation_plan(module)
    evaluations[0]["queue"] = "GPU8-unsealed"
    with pytest.raises(ValueError, match="unsupported queue"):
        module.build_dependency_plan(
            json.dumps(_formal_training_manifest()), json.dumps(evaluations)
        )

    evaluations = _evaluation_plan(module)
    evaluations[0], evaluations[1] = evaluations[1], evaluations[0]
    with pytest.raises(ValueError, match="subject mismatch"):
        module.build_dependency_plan(
            json.dumps(_formal_training_manifest()), json.dumps(evaluations)
        )


def _evaluation_task(
    module,
    *,
    dependency: dict[str, str],
    status: str,
    script_diff: str = "sealed evaluation bootstrap",
    queue_id: str = "",
):
    parameters = {
        "Args/stage": "baseline_validate",
        "Args/controlled_baseline": dependency["subject"],
        "Args/controlled_baseline_task_id": dependency["training_task_id"],
        "Args/controlled_baseline_model_id": dependency["training_model_id"],
        "Args/controlled_baseline_checkpoint_sha256": dependency[
            "training_checkpoint_sha256"
        ],
        "Args/predecessor_task_id": dependency["training_task_id"],
        "Args/source_dataset_id": module.EXPECTED_SOURCE_DATASET_ID,
        "Args/source_archive_name": module.EXPECTED_SOURCE_ARCHIVE_NAME,
        "Args/source_archive_bytes": str(module.EXPECTED_SOURCE_ARCHIVE_BYTES),
        "Args/source_archive_sha256": module.EXPECTED_SOURCE_ARCHIVE_SHA256,
        "Args/training_dataset_id": module.EXPECTED_TRAINING_DATASET_ID,
        "Args/native_bundle_bytes": "654321",
        "Args/native_bundle_sha256": "c" * 64,
        "Args/build_manifest_sha256": "d" * 64,
        "Args/gpus": "4",
        "Args/max_epochs": "50",
        "Args/amp": "False",
    }
    return SimpleNamespace(
        status=status,
        reload=lambda: None,
        get_parameters=lambda cast=False: parameters,
        get_models=lambda: {"input": [], "output": []},
        data=SimpleNamespace(
            script=SimpleNamespace(
                repository="",
                working_dir=".",
                entry_point="clearml_5090_bootstrap.py",
                diff=script_diff,
            ),
            execution=SimpleNamespace(queue=queue_id),
        ),
    )


def test_evaluation_identity_rejects_script_or_queue_drift() -> None:
    module = _load_module()
    dependency = module.build_dependency_plan(
        json.dumps(_formal_training_manifest()),
        json.dumps(_evaluation_plan(module)),
    )[0]
    script_diff = "sealed evaluation bootstrap"
    script_sha = hashlib.sha256(script_diff.encode()).hexdigest()
    evaluation = _evaluation_task(
        module,
        dependency=dependency,
        status="created",
        script_diff=script_diff,
    )
    with pytest.raises(RuntimeError, match="script SHA-256 mismatch"):
        module._require_evaluation_identity(
            evaluation,
            dependency=dependency,
            expected_script_sha256="a" * 64,
            expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
            expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
        )

    evaluation = _evaluation_task(
        module,
        dependency=dependency,
        status="queued",
        script_diff=script_diff,
        queue_id="0" * 32,
    )
    with pytest.raises(RuntimeError, match="queue mismatch"):
        module._require_evaluation_identity(
            evaluation,
            dependency=dependency,
            expected_script_sha256=script_sha,
            expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
            expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
            expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
        )


def test_incomplete_training_dependency_is_never_ready_for_release() -> None:
    module = _load_module()
    task, task_id, subject, script_sha, _, _ = _completed_training(module)
    task.status = "in_progress"

    assert not module._require_completed_training(
        task,
        subject=subject,
        task_id=task_id,
        expected_script_sha256=script_sha,
        expected_source_dataset_id=module.EXPECTED_SOURCE_DATASET_ID,
        expected_source_archive_name=module.EXPECTED_SOURCE_ARCHIVE_NAME,
        expected_source_archive_bytes=module.EXPECTED_SOURCE_ARCHIVE_BYTES,
        expected_source_archive_sha256=module.EXPECTED_SOURCE_ARCHIVE_SHA256,
        expected_training_dataset_id=module.EXPECTED_TRAINING_DATASET_ID,
        expected_predecessor_task_id=module.EXPECTED_PREDECESSOR_TASK_ID,
    )


def test_training_and_evaluation_runtime_identity_must_match_exactly() -> None:
    module = _load_module()
    training, _, _, _, _, _ = _completed_training(module)
    dependency = module.build_dependency_plan(
        json.dumps(_formal_training_manifest()),
        json.dumps(_evaluation_plan(module)),
    )[0]
    evaluation = _evaluation_task(
        module,
        dependency=dependency,
        status="created",
    )
    training_identity = module._runtime_identity(training, context="training")
    evaluation_identity = module._runtime_identity(evaluation, context="evaluation")
    assert evaluation_identity == training_identity

    evaluation.get_parameters()["Args/native_bundle_sha256"] = "e" * 64
    assert (
        module._runtime_identity(evaluation, context="evaluation") != training_identity
    )
