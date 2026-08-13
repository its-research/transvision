from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_formal_training_provenance.py"
CONTROLLER_PATH = ROOT / "tools/resilient_v2x/clearml_5090_training_controller.py"
PARENT_CONTROLLER_ID = "e" * 32
RECOVERY_CONTROLLER_ID = "b" * 32
SOURCE_CONTROLLER_ID = "c" * 32
OUTPUT_TASK_ID = "f" * 32
GATE_TASK_ID = "d" * 32
FAKE_DOCKER_COMMAND = "sealed-training-image --network=host"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "clearml_formal_training_provenance",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "clearml_5090_training_controller_for_provenance", CONTROLLER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_controller_waits_for_required_artifact_visibility() -> None:
    module = _load_module()

    class _EventuallyConsistentController:
        status = "completed"

        def __init__(self) -> None:
            self.artifacts: dict[str, object] = {}
            self.reload_calls = 0

        def _reload(self) -> object:
            self.reload_calls += 1
            self.artifacts = {
                module.PROGRESS_ARTIFACT: object(),
                module.TRAINING_MANIFEST_ARTIFACT: object(),
            }
            return self

    controller = _EventuallyConsistentController()
    sleeps: list[float] = []
    module._wait_for_completed(
        controller,
        deadline=10.0,
        poll_seconds=1.0,
        monotonic_clock=lambda: 0.0,
        sleeper=sleeps.append,
        required_artifacts=(
            module.PROGRESS_ARTIFACT,
            module.TRAINING_MANIFEST_ARTIFACT,
        ),
    )

    assert sleeps == [1.0]
    assert controller.reload_calls == 1


class _Pathlib2Like:
    def __init__(self, value: str | bytes) -> None:
        self.value = value

    def __fspath__(self) -> str | bytes:
        return self.value


@pytest.mark.parametrize("as_bytes", [False, True])
def test_fresh_artifact_mapping_accepts_generic_pathlike(
    tmp_path: Path,
    as_bytes: bool,
) -> None:
    module = _load_module()
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text('{"result": "pass"}', encoding="utf-8")
    path_value: str | bytes = os.fspath(artifact_path)
    if as_bytes:
        path_value = os.fsencode(path_value)
    task = SimpleNamespace(
        artifacts={"audit": _Artifact(_Pathlib2Like(path_value))}
    )

    assert module._fresh_artifact_mapping(
        task,
        "audit",
        context="test",
    ) == {"result": "pass"}


@pytest.mark.parametrize("failure", ["invalid_return", "raised_error"])
def test_fresh_artifact_mapping_rejects_invalid_pathlike(failure: str) -> None:
    module = _load_module()

    class _InvalidPathLike:
        def __fspath__(self) -> object:
            if failure == "raised_error":
                raise RuntimeError("path conversion failed")
            return 7

    task = SimpleNamespace(artifacts={"audit": _Artifact(_InvalidPathLike())})

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="not a JSON object or valid local path",
    ):
        module._fresh_artifact_mapping(task, "audit", context="test")


def test_fresh_artifact_mapping_pathlike_cannot_bypass_symlink_guard(
    tmp_path: Path,
) -> None:
    module = _load_module()
    target = tmp_path / "target.json"
    target.write_text('{"result": "pass"}', encoding="utf-8")
    symlink = tmp_path / "artifact.json"
    symlink.symlink_to(target)
    task = SimpleNamespace(
        artifacts={"audit": _Artifact(_Pathlib2Like(os.fspath(symlink)))}
    )

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="cannot be read|path is unsafe",
    ):
        module._fresh_artifact_mapping(task, "audit", context="test")


def _source_revision_transition(module, monkeypatch) -> dict[str, object]:
    controller = _load_controller()
    source_parameters = module._transition_source_parameters("old")
    target_parameters = module._transition_source_parameters("new")
    common_identity = {
        "entry_point": "clearml_5090_bootstrap.py",
        "script_sha256": module.CANONICAL_NORMALIZED_SCRIPT_IDENTITY_SHA256,
        "docker_command": "sealed/image --network host",
        "docker_image": "sealed/image@sha256:" + "2" * 64,
        "base_image_manifest_digest": "sha256:" + "3" * 64,
        "base_image_config_digest": "sha256:" + "4" * 64,
        "native_build_task_id": module.SOURCE_REVISION_NATIVE_BUILD_TASK_ID,
    }
    source_identity = {
        "task_id": module.SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID,
        **common_identity,
        "source_parameters": source_parameters,
    }
    target_identity = {
        "task_id": module.SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID,
        **common_identity,
        "source_parameters": target_parameters,
    }
    source_legacy_identity = copy.deepcopy(source_identity)
    source_legacy_identity["script_sha256"] = (
        module.SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256
    )
    shared_parameters = {"General/retained_template_parameter": "sealed"}
    transition = controller._build_source_revision_transition_contract(
        transition_id=module.SOURCE_REVISION_TRANSITION_ID,
        source_controller_task_id=(module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID),
        source_template_identity=source_identity,
        source_legacy_template_identity=source_legacy_identity,
        target_template_identity=target_identity,
        source_template_parameters={**source_parameters, **shared_parameters},
        target_template_parameters={**target_parameters, **shared_parameters},
    )
    monkeypatch.setattr(
        module,
        "SOURCE_REVISION_TRANSITION_SEAL_SHA256",
        transition["seal_sha256"],
    )
    return transition


def _schema4_recovery_fixture(module, monkeypatch):
    transition = _source_revision_transition(module, monkeypatch)
    controller = _load_controller()
    source_identity = transition["source_template_identity"]
    target_identity = transition["target_template_identity"]
    assert isinstance(source_identity, dict) and isinstance(target_identity, dict)
    observed_source_identity = copy.deepcopy(source_identity)
    observed_source_parameters = observed_source_identity["source_parameters"]
    assert isinstance(observed_source_parameters, dict)
    observed_source_parameters["Args/native_bundle_bytes"] = int(
        observed_source_parameters["Args/native_bundle_bytes"]
    )
    template_equivalence = controller._template_identity_parameter_equivalence_receipt(
        observed_source_identity,
        source_identity,
        context="source-revision controller template identity",
    )
    monkeypatch.setattr(
        module,
        "SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256",
        template_equivalence["seal_sha256"],
    )
    adopted_subjects = [
        *module.SOURCE_REVISION_SUBJECTS["old"],
        *module.SOURCE_REVISION_SUBJECTS["new"][:3],
    ]
    steps = []
    predecessor = GATE_TASK_ID
    for index, subject in enumerate(module.SUBJECT_ORDER, start=1):
        task_id = f"{index:032x}"
        recorded_predecessor = (
            module.SOURCE_REVISION_TARGET_PREDECESSOR_TASK_ID
            if subject in module.SOURCE_REVISION_SUBJECTS["new"]
            else predecessor
        )
        steps.append(
            {
                "experiment": subject,
                "task_id": task_id,
                "predecessor_task_id": recorded_predecessor,
            }
        )
        predecessor = task_id
    steps_by_subject = {step["experiment"]: step for step in steps}
    adopted = {
        subject: steps_by_subject[subject]["task_id"] for subject in adopted_subjects
    }
    predecessors = {
        subject: steps_by_subject[subject]["predecessor_task_id"]
        for subject in adopted_subjects
    }
    roles = {
        subject: (
            "source" if subject in module.SOURCE_REVISION_SUBJECTS["old"] else "target"
        )
        for subject in adopted_subjects
    }
    receipts = {}
    for subject in adopted_subjects:
        role = roles[subject]
        identity = source_identity if role == "source" else target_identity
        parameters = module._transition_source_parameters(
            "old" if role == "source" else "new"
        )
        parameter_sha256 = module._content_sha256(parameters)
        receipts[subject] = {
            "task_id": adopted[subject],
            "experiment": subject,
            "template_role": role,
            "template_task_id": identity["task_id"],
            "template_script_sha256": identity["script_sha256"],
            "task_script_sha256": identity["script_sha256"],
            "script_identity_policy": "exact_canonical_template",
            "legacy_script_compatibility_receipt": None,
            "source_parameters": parameters,
            "config_path": module._transition_config_path(subject),
            "config_inventory_status": (
                "byte_identical_across_revisions"
                if role == "source"
                else "target_revision_required"
            ),
            "expected_parameter_count": len(parameters),
            "expected_parameters_sha256": parameter_sha256,
            "observed_parameter_projection_sha256": parameter_sha256,
            "exact_execution_parameter_match": True,
            "predecessor_task_id": predecessors[subject],
        }
    recovery = {
        "schema_version": 4,
        "mode": "failed_controller_immutable_fork",
        "source_controller_task_id": (module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID),
        "source_controller_status": "failed",
        "source_progress_revision": 1,
        "source_progress_seal_sha256": "5" * 64,
        "source_progress_artifact_readback": {
            "url": (
                "http://10.100.34.118:8081/project/source."
                f"{module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID}/artifacts/"
                f"{module.PROGRESS_ARTIFACT}/{module.PROGRESS_ARTIFACT}.json"
            ),
            "revision": 1,
            "seal_sha256": "5" * 64,
            "force_download": True,
            "stable_readbacks": 2,
        },
        "source_template_script_sha256": source_identity["script_sha256"],
        "target_template_script_sha256": target_identity["script_sha256"],
        "source_revision_transition": transition,
        "source_progress_template_equivalence": template_equivalence,
        "source_patch": "nested-teacher-config-consistency-v1",
        "source_recovery_chain": None,
        "rerun_experiments": [],
        "rerun_source_task_ids": {},
        "rerun_task_observations": {},
        "rerun_predecessor_task_ids": {},
        "completion_validation_retries": {},
        "recovered_pending_target_children": {},
        "transition_replaced_source_tasks": {},
        "target_template_predecessor_task_ids": {
            subject: module.SOURCE_REVISION_TARGET_PREDECESSOR_TASK_ID
            for subject in module.SOURCE_REVISION_SUBJECTS["new"]
        },
        "recovery_target_adoptions": {},
        "adopted_task_ids": adopted,
        "adopted_predecessor_task_ids": predecessors,
        "adopted_task_template_roles": roles,
        "adopted_task_binding_receipts": receipts,
    }
    return recovery, target_identity, steps


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bootstrap_sources() -> tuple[str, str]:
    prefix = (
        "from __future__ import annotations\n\n"
        "BOOTSTRAP_ID = 1\n"
        'GPU_ENV = {"NVIDIA_TF32_OVERRIDE": "0"}\n'
        "_allowed_caps = frozenset({(12, 0), (8, 0), (7, 0)})\n"
    )
    suffix = (
        "\ndef validate_capabilities(capabilities):\n"
        "    if len(capabilities) != 4:\n"
        '        raise RuntimeError("exactly four GPUs are required")\n'
        "    if len(set(capabilities)) != 1:\n"
        '        raise RuntimeError("GPU capabilities must be homogeneous")\n'
        "    return capabilities[0] in _allowed_caps\n\n"
        "def sink(**kwargs):\n"
        "    return kwargs\n\n"
        "def contract(spec):\n"
        '    return {"expected_nested_teacher": '
        "spec.name in NESTED_TEACHER_EXPERIMENTS}\n\n"
        "def validate(spec):\n"
        "    return sink(expect_nested_teacher="
        "spec.name in NESTED_TEACHER_EXPERIMENTS)\n\n"
        "def bootstrap():\n"
        "    return BOOTSTRAP_ID\n"
    )
    legacy = (
        prefix + 'NESTED_TEACHER_EXPERIMENTS = frozenset({"resilient_v2x", '
        '"support_residual"})\n' + suffix
    )
    expanded = (
        prefix
        + "NESTED_TEACHER_EXPERIMENTS = frozenset(\n"
        + "    {\n"
        + '        "support_residual",\n'
        + '        "ptf_none",\n'
        + '        "ptf_linear",\n'
        + '        "router_static",\n'
        + '        "router_uniform",\n'
        + '        "no_reliability",\n'
        + '        "no_delay_metadata",\n'
        + '        "concat_capacity_matched",\n'
        + '        "resilient_v2x",\n'
        + "    }\n"
        + ")\n"
        + suffix
    )
    return legacy, expanded


def _install_script_hashes(module, monkeypatch) -> tuple[str, str]:
    legacy, expanded = _bootstrap_sources()
    legacy_sha = _digest(legacy)
    expanded_sha = _digest(expanded)
    monkeypatch.setattr(module, "LEGACY_BOOTSTRAP_SHA256", legacy_sha)
    monkeypatch.setattr(module, "EXPANDED_BOOTSTRAP_SHA256", expanded_sha)
    monkeypatch.setattr(
        module,
        "ALLOWED_BOOTSTRAP_SHA256",
        frozenset({legacy_sha, expanded_sha}),
    )
    identities = {}
    for script_sha, source in ((legacy_sha, legacy), (expanded_sha, expanded)):
        script = {
            "repository": "",
            "working_dir": ".",
            "entry_point": "clearml_5090_bootstrap.py",
            "diff": source,
        }
        identity = {key: script.get(key) for key in module.SCRIPT_IDENTITY_KEYS}
        identities[script_sha] = module._content_sha256(identity)
    monkeypatch.setattr(
        module,
        "LEGACY_NORMALIZED_SCRIPT_IDENTITY_SHA256",
        identities[legacy_sha],
    )
    monkeypatch.setattr(
        module,
        "CANONICAL_NORMALIZED_SCRIPT_IDENTITY_SHA256",
        identities[expanded_sha],
    )
    monkeypatch.setattr(
        module,
        "NORMALIZED_SCRIPT_IDENTITY_BY_RAW_SHA256",
        identities,
    )
    return legacy, expanded


def _raw_value(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, dict):
        return {key: _raw_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_raw_value(item) for item in value]
    serializer = getattr(value, "to_dict", None)
    if callable(serializer):
        return serializer()
    return copy.deepcopy(value)


class _Raw(SimpleNamespace):
    def to_dict(self) -> dict[str, object]:
        return {key: _raw_value(value) for key, value in vars(self).items()}


class _Artifact:
    def __init__(self, value: object) -> None:
        self.value = copy.deepcopy(value)
        self.force_download_calls: list[bool] = []
        self.on_get = None

    def get(self, *, force_download: bool = False) -> object:
        self.force_download_calls.append(force_download)
        assert force_download is True
        if callable(self.on_get):
            self.on_get(self)
        return copy.deepcopy(self.value)


class _RawArtifact:
    def __init__(self, key: str, artifact: _Artifact) -> None:
        self.key = key
        self.artifact = artifact

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "type": "dict",
            "mode": "output",
            "uri": f"http://10.100.34.118:8081/artifacts/{self.key}.json",
            "content_size": len(repr(self.artifact.value)),
            "hash": hashlib.sha256(repr(self.artifact.value).encode()).hexdigest(),
        }


class _Model:
    def __init__(self, model_id: str, *, task: str, name: str, url: str) -> None:
        self.id = model_id
        self.task = task
        self.name = name
        self.url = url


class _Task:
    def __init__(
        self,
        task_id: str,
        *,
        status: str,
        parent: str,
        entry_point: str,
        source: str,
        artifacts: dict[str, _Artifact] | None = None,
        parameters: dict[str, object] | None = None,
        output_uri: str = "",
        name: str = "task",
        docker: str = FAKE_DOCKER_COMMAND,
        output_models: list[_Model] | None = None,
    ) -> None:
        self.id = task_id
        self.status = status
        self.parent = parent
        self.artifacts = artifacts or {}
        self.parameters = parameters or {}
        self.tags: list[str] = []
        self.uploads: list[tuple[str, object, bool]] = []
        self.flush_calls = 0
        self.tag_calls: list[list[str]] = []
        self.reload_calls = 0
        self.docker = docker
        self.output_models = list(output_models or [])
        self._data = _Raw(
            id=task_id,
            name=name,
            user="1" * 32,
            company="2" * 32,
            type="training",
            status=status,
            comment="",
            parent=parent,
            project="3" * 32,
            output=_Raw(destination=output_uri),
            execution=_Raw(
                queue="4" * 32,
                artifacts=[
                    _RawArtifact(key, artifact)
                    for key, artifact in self.artifacts.items()
                ],
            ),
            script=_Raw(
                repository="",
                working_dir=".",
                entry_point=entry_point,
                diff=source,
            ),
            tags=self.tags,
            system_tags=[],
            status_message="",
            status_reason="",
            last_worker="worker",
            hyperparams=self.parameters,
            configuration={},
        )

    @property
    def data(self) -> object:
        return self._data

    def _reload(self) -> object:
        self.reload_calls += 1
        return self._data

    def get_parameters(self, cast: bool = False) -> dict[str, object]:
        assert cast is False
        return copy.deepcopy(self.parameters)

    def get_base_docker(self) -> str:
        return self.docker

    def get_models(self) -> dict[str, list[_Model]]:
        return {"output": copy.deepcopy(self.output_models)}

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        self.uploads.append((name, copy.deepcopy(artifact_object), wait_on_upload))
        artifact = _Artifact(artifact_object)
        self.artifacts[name] = artifact
        self._data.execution.artifacts.append(_RawArtifact(name, artifact))
        return True

    def flush(self, *, wait_for_uploads: bool) -> bool:
        assert wait_for_uploads is True
        self.flush_calls += 1
        return True

    def set_tags(self, tags: list[str]) -> bool:
        self.tag_calls.append(list(tags))
        self.tags = list(tags)
        self._data.tags = list(tags)
        return True


def _task_id(index: int) -> str:
    return f"{index:032x}"


def _model_id(index: int) -> str:
    return f"{index + 1000:032x}"


def _run_contract(
    module, subject: str, task_id: str, predecessor: str
) -> dict[str, object]:
    source = module._source_revision_for_subject(subject)
    return {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": subject,
        "source_dataset_id": source["dataset_id"],
        "training_dataset_id": module.TRAINING_DATASET_ID,
        "native_bundle_sha256": module.NATIVE_BUNDLE_SHA256,
        "build_manifest_sha256": module.BUILD_MANIFEST_SHA256,
        "source_archive": {
            "name": source["archive_name"],
            "size_bytes": source["archive_size_bytes"],
            "sha256": source["archive_sha256"],
        },
        "predecessor_task_id": predecessor,
        "global_batch_size": 8,
        "max_epochs": 50,
        "seed": module.TRAINING_SEED,
        "amp": False,
        "precision": "FP32",
        "val_interval": 10,
        "teacher": {"sha256": module.TEACHER_CHECKPOINT_SHA256},
        "common_teacher_initialization": {
            "policy": "shared-only",
            "contract": module.COMMON_TEACHER_INITIALIZATION_CONTRACT,
            "shared_prefixes": list(module.COMMON_TEACHER_INITIALIZATION_PREFIXES),
            "expected_source_keys": module.COMMON_TEACHER_SOURCE_KEYS,
            "expected_source_numel": module.COMMON_TEACHER_SOURCE_NUMEL,
            "expected_source_bytes": module.COMMON_TEACHER_SOURCE_BYTES,
            "expected_shared_keys": module.COMMON_TEACHER_SHARED_KEYS,
            "expected_shared_numel": module.COMMON_TEACHER_SHARED_NUMEL,
            "expected_shared_bytes": module.COMMON_TEACHER_SHARED_BYTES,
            "expected_teacher_fusion_keys": module.COMMON_TEACHER_FUSION_KEYS,
            "teacher_checkpoint_sha256": module.TEACHER_CHECKPOINT_SHA256,
            "audit_artifact_name": module.COMMON_TEACHER_AUDIT_ARTIFACT,
            "audit_filename": "common_teacher_initialization_audit.json",
            "expected_nested_teacher": (
                subject in module.EXPANDED_NESTED_TEACHER_EXPERIMENTS
            ),
        },
    }


class _World:
    pass


def _world(module, monkeypatch) -> _World:
    legacy, expanded = _install_script_hashes(module, monkeypatch)
    monkeypatch.setattr(module, "DOCKER_COMMAND_SHA256", _digest(FAKE_DOCKER_COMMAND))
    monkeypatch.setattr(
        module,
        "_validate_source_revision_recovery",
        lambda *_args, **_kwargs: None,
    )
    world = _World()
    task_ids = [_task_id(index) for index in range(1, 27)]
    predecessors = [GATE_TASK_ID, *task_ids[:-1]]
    final_adopted_subjects = {
        "support_residual",
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
    }
    recursive_adopted_subjects = {"support_residual", "no_distillation"}
    recursive_created_subjects = {"ptf_none", "ptf_linear", "router_static"}
    legacy_subjects = {"support_residual", "no_distillation"}
    legacy_identity = module.LEGACY_NORMALIZED_SCRIPT_IDENTITY_SHA256
    canonical_identity = module.CANONICAL_NORMALIZED_SCRIPT_IDENTITY_SHA256
    entries = []
    tasks: dict[str, _Task] = {}
    contracts: dict[str, _Artifact] = {}
    final_contracts: dict[str, _Artifact] = {}
    teacher_audits: dict[str, _Artifact] = {}
    for index, (subject, task_id, predecessor) in enumerate(
        zip(module.SUBJECT_ORDER, task_ids, predecessors, strict=True), start=1
    ):
        model_id = _model_id(index)
        model_name = f"ResilientV2X {subject} final checkpoint"
        model_url = f"{module.FILES_SERVER_URI}/models/{subject}_epoch_50.pth"
        checkpoint_sha256 = f"{index + 2000:064x}"
        checkpoint_size_bytes = 1_000_000 + index
        nested_teacher = subject in module.EXPANDED_NESTED_TEACHER_EXPERIMENTS
        zero_fusion = subject in module.ZERO_FUSION_ALLOWED_SUBJECTS
        fusion_keys = 0 if zero_fusion else index + 10
        fusion_numel = 0 if zero_fusion else 10_000 + index
        fusion_bytes = 0 if zero_fusion else 40_000 + index
        fusion_sha256 = f"{index + 3000:064x}"
        teacher_audit = {
            "schema_version": 1,
            "contract": module.COMMON_TEACHER_INITIALIZATION_CONTRACT,
            "result": "pass",
            "checkpoint": {
                "path": "/sealed/teacher_checkpoint.pth",
                "filename": "teacher_checkpoint.pth",
                "size_bytes": 2_000_000,
                "sha256": module.TEACHER_CHECKPOINT_SHA256,
                "expected_sha256": module.TEACHER_CHECKPOINT_SHA256,
            },
            "source": {
                "keys": module.COMMON_TEACHER_SOURCE_KEYS,
                "numel": module.COMMON_TEACHER_SOURCE_NUMEL,
                "bytes": module.COMMON_TEACHER_SOURCE_BYTES,
                "expected_keys": module.COMMON_TEACHER_SOURCE_KEYS,
                "common_keys": module.COMMON_TEACHER_SHARED_KEYS,
                "expected_common_keys": module.COMMON_TEACHER_SHARED_KEYS,
                "fusion_keys": module.COMMON_TEACHER_FUSION_KEYS,
                "expected_fusion_keys": module.COMMON_TEACHER_FUSION_KEYS,
                "state_sha256": f"{index + 4000:064x}",
            },
            "shared_initialization": {
                "prefixes": list(module.COMMON_TEACHER_INITIALIZATION_PREFIXES),
                "keys": module.COMMON_TEACHER_SHARED_KEYS,
                "numel": module.COMMON_TEACHER_SHARED_NUMEL,
                "bytes": module.COMMON_TEACHER_SHARED_BYTES,
                "expected_keys": module.COMMON_TEACHER_SHARED_KEYS,
                "state_sha256": f"{index + 5000:064x}",
                "shape_dtype_verified": True,
                "exact_tensor_equality_verified": True,
            },
            "method_specific_fusion": {
                "keys": fusion_keys,
                "numel": fusion_numel,
                "bytes": fusion_bytes,
                "sha256_before": fusion_sha256,
                "sha256_after": fusion_sha256,
                "unchanged": True,
            },
            "target": {
                "model_type": f"Fixture{subject}",
                "target_key_count": (
                    module.COMMON_TEACHER_SHARED_KEYS
                    + fusion_keys
                    + (module.COMMON_TEACHER_SOURCE_KEYS if nested_teacher else 0)
                ),
                "target_common_key_count": module.COMMON_TEACHER_SHARED_KEYS,
                "target_fusion_key_count": fusion_keys,
                "nested_teacher_present": nested_teacher,
                "nested_teacher_key_count": (
                    module.COMMON_TEACHER_SOURCE_KEYS if nested_teacher else 0
                ),
                "nested_teacher_full_equality_verified": nested_teacher,
            },
        }
        entry = {
            "index": index,
            "subject": subject,
            "kind": module.SUBJECT_KIND[subject],
            "training_task_id": task_id,
            "training_predecessor_task_id": predecessor,
            "model_id": model_id,
            "model_name": model_name,
            "model_url": model_url,
            "checkpoint_filename": "epoch_50.pth",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "training_seed": module.TRAINING_SEED,
            "training_overlay_protocol_seed": module.TRAINING_SEED,
            "common_teacher_initialization_audit_artifact": (
                module.COMMON_TEACHER_AUDIT_ARTIFACT
            ),
            "common_teacher_initialization_audit_sha256": (
                module._content_sha256(teacher_audit)
            ),
        }
        entries.append(entry)
        contract_artifact = _Artifact(
            _run_contract(module, subject, task_id, predecessor)
        )
        final_contract_artifact = _Artifact(
            {
                "model_id": model_id,
                "name": model_name,
                "url": model_url,
                "filename": "epoch_50.pth",
                "size_bytes": checkpoint_size_bytes,
                "sha256": checkpoint_sha256,
            }
        )
        teacher_audit_artifact = _Artifact(teacher_audit)
        contracts[subject] = contract_artifact
        final_contracts[subject] = final_contract_artifact
        teacher_audits[subject] = teacher_audit_artifact
        source = legacy if subject in legacy_subjects else expanded
        source_revision = module._source_revision_for_subject(subject)
        if subject == "support_residual":
            parent = SOURCE_CONTROLLER_ID
        elif subject == "no_distillation":
            parent = PARENT_CONTROLLER_ID
        elif subject in recursive_created_subjects:
            parent = RECOVERY_CONTROLLER_ID
        else:
            parent = module.TRAINING_CONTROLLER_TASK_ID
        parameters = {
            "Args/experiment_from_task": subject,
            "Args/source_dataset_id": source_revision["dataset_id"],
            "Args/source_archive_name": source_revision["archive_name"],
            "Args/source_archive_bytes": str(source_revision["archive_size_bytes"]),
            "Args/source_archive_sha256": source_revision["archive_sha256"],
            "Args/training_dataset_id": module.TRAINING_DATASET_ID,
            "Args/native_bundle_sha256": module.NATIVE_BUNDLE_SHA256,
            "Args/build_manifest_sha256": module.BUILD_MANIFEST_SHA256,
            "Args/predecessor_task_id": predecessor,
            "Args/gpus": "4",
            "Args/stage": "all",
            "Args/max_epochs": "50",
            "Args/amp": "False",
            "Args/training_seed": str(module.TRAINING_SEED),
            "Args/teacher_checkpoint_sha256": module.TEACHER_CHECKPOINT_SHA256,
            "Args/fixture_extra": "sealed too",
        }
        tasks[task_id] = _Task(
            task_id,
            status="completed",
            parent=parent,
            entry_point="clearml_5090_bootstrap.py",
            source=source,
            artifacts={
                module.RUN_CONTRACT_ARTIFACT: contract_artifact,
                module.FINAL_CHECKPOINT_ARTIFACT: final_contract_artifact,
                module.COMMON_TEACHER_AUDIT_ARTIFACT: teacher_audit_artifact,
            },
            parameters=parameters,
            name=f"training {subject}",
            output_models=[
                _Model(
                    model_id,
                    task=task_id,
                    name=model_name,
                    url=model_url,
                )
            ],
        )

    def _step(
        index: int,
        subject: str,
        *,
        present: bool,
        state: str,
        adopted: bool,
        final: bool = False,
    ) -> dict[str, object]:
        task_id = task_ids[index - 1] if present else None
        predecessor = predecessors[index - 1] if present else None
        result = (
            {
                "task_id": task_id,
                "training_seed": module.TRAINING_SEED,
                "training_overlay_protocol_seed": module.TRAINING_SEED,
                "run_contract_artifact": module.RUN_CONTRACT_ARTIFACT,
            }
            if final
            else None
        )
        return {
            "index": index,
            "experiment": subject,
            "state": state,
            "task_name": f"task {subject}",
            "task_id": task_id,
            "predecessor_task_id": predecessor,
            "result": result,
            "worker_queue": "GPU4-A100",
            "adopted": adopted,
        }

    source_steps = [
        _step(
            index,
            subject,
            present=subject == "support_residual",
            state="running" if subject == "support_residual" else "pending",
            adopted=False,
        )
        for index, subject in enumerate(module.SUBJECT_ORDER, start=1)
    ]
    source_progress = module._sealed(
        {
            "schema_version": 1,
            "controller_type": "resilient_v2x_post_main_sequential_training",
            "controller_task_id": SOURCE_CONTROLLER_ID,
            "experiment_order": list(module.SUBJECT_ORDER),
            "training_seed": module.TRAINING_SEED,
            "training_overlay_protocol_seed": module.TRAINING_SEED,
            "revision": 14,
            "template": {"script_sha256": legacy_identity},
            "recovery": None,
            "steps": source_steps,
        }
    )
    recursive_recovery = {
        "schema_version": 2,
        "mode": "failed_controller_immutable_fork",
        "source_controller_task_id": SOURCE_CONTROLLER_ID,
        "source_controller_status": "failed",
        "source_progress_revision": source_progress["revision"],
        "source_progress_seal_sha256": source_progress["seal_sha256"],
        "source_recovery_chain": None,
        "adopted_task_ids": {
            subject: task_ids[module.SUBJECT_ORDER.index(subject)]
            for subject in recursive_adopted_subjects
        },
        "adopted_predecessor_task_ids": {
            subject: predecessors[module.SUBJECT_ORDER.index(subject)]
            for subject in recursive_adopted_subjects
        },
        "recovery_target_adoptions": {
            "no_distillation": {
                "task_id": task_ids[module.SUBJECT_ORDER.index("no_distillation")],
                "predecessor_task_id": predecessors[
                    module.SUBJECT_ORDER.index("no_distillation")
                ],
                "parent_controller_status": "failed",
                "parent_controller_task_id": PARENT_CONTROLLER_ID,
                "task_script_sha256": legacy_identity,
            }
        },
        "recovered_pending_target_children": None,
    }
    recursive_present = recursive_adopted_subjects | recursive_created_subjects
    recursive_steps = [
        _step(
            index,
            subject,
            present=subject in recursive_present,
            state=(
                "failed"
                if subject in recursive_adopted_subjects
                else "completed"
                if subject in recursive_created_subjects
                else "pending"
            ),
            adopted=subject in recursive_adopted_subjects,
        )
        for index, subject in enumerate(module.SUBJECT_ORDER, start=1)
    ]
    recursive_progress = module._sealed(
        {
            "schema_version": 1,
            "controller_type": "resilient_v2x_post_main_sequential_training",
            "controller_task_id": RECOVERY_CONTROLLER_ID,
            "experiment_order": list(module.SUBJECT_ORDER),
            "training_seed": module.TRAINING_SEED,
            "training_overlay_protocol_seed": module.TRAINING_SEED,
            "revision": 2,
            "template": {"script_sha256": canonical_identity},
            "recovery": recursive_recovery,
            "steps": recursive_steps,
        }
    )
    recovery = {
        "schema_version": 4,
        "mode": "failed_controller_immutable_fork",
        "source_controller_task_id": RECOVERY_CONTROLLER_ID,
        "source_controller_status": "failed",
        "source_progress_revision": recursive_progress["revision"],
        "source_progress_seal_sha256": recursive_progress["seal_sha256"],
        "source_recovery_chain": {
            "source_controller_task_id": RECOVERY_CONTROLLER_ID,
            "source_progress_revision": recursive_progress["revision"],
            "source_progress_seal_sha256": recursive_progress["seal_sha256"],
            "source_recovery_sha256": module._content_sha256(recursive_recovery),
        },
        "adopted_task_ids": {
            subject: task_ids[module.SUBJECT_ORDER.index(subject)]
            for subject in final_adopted_subjects
        },
        "adopted_predecessor_task_ids": {
            subject: predecessors[module.SUBJECT_ORDER.index(subject)]
            for subject in final_adopted_subjects
        },
        "recovery_target_adoptions": {},
        "recovered_pending_target_children": {},
    }
    steps = [
        _step(
            index,
            subject,
            present=True,
            state="completed",
            adopted=subject in final_adopted_subjects,
            final=True,
        )
        for index, subject in enumerate(module.SUBJECT_ORDER, start=1)
    ]
    progress = module._sealed(
        {
            "schema_version": 1,
            "controller_type": "resilient_v2x_post_main_sequential_training",
            "controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
            "experiment_order": list(module.SUBJECT_ORDER),
            "training_seed": module.TRAINING_SEED,
            "training_overlay_protocol_seed": module.TRAINING_SEED,
            "revision": 99,
            "template": {"script_sha256": canonical_identity},
            "recovery": recovery,
            "steps": steps,
        }
    )
    manifest = module._sealed(
        {
            "schema_version": 1,
            "manifest_type": "resilient_v2x_formal_1337_training_inputs",
            "protocol_id": "DAIR-CAUSAL-1337-v1",
            "sample_count": 1337,
            "delays_ms": [0, 100, 200, 300],
            "conditions": ["Full", "L-Fail", "C-Fail"],
            "run_count": 12,
            "checkpoint_policy": "epoch_50_final_only",
            "training_seed": module.TRAINING_SEED,
            "training_overlay_protocol_seed": module.TRAINING_SEED,
            "evaluation_release_semantics": (
                "formal_manifest_after_full_training_suite_completion"
            ),
            "subject_order": list(module.SUBJECT_ORDER),
            "subject_count": len(module.SUBJECT_ORDER),
            "entries": entries,
        }
    )
    progress_artifact = _Artifact(progress)
    manifest_artifact = _Artifact(manifest)
    controller = _Task(
        module.TRAINING_CONTROLLER_TASK_ID,
        status="completed",
        parent=GATE_TASK_ID,
        entry_point="clearml_5090_training_controller.py",
        source="controller source",
        artifacts={
            module.PROGRESS_ARTIFACT: progress_artifact,
            module.TRAINING_MANIFEST_ARTIFACT: manifest_artifact,
        },
        name="controller",
    )
    parent_controller = _Task(
        PARENT_CONTROLLER_ID,
        status="failed",
        parent=GATE_TASK_ID,
        entry_point="clearml_5090_training_controller.py",
        source="failed parent controller source",
        name="failed parent controller",
    )
    recursive_controller = _Task(
        RECOVERY_CONTROLLER_ID,
        status="failed",
        parent=GATE_TASK_ID,
        entry_point="clearml_5090_training_controller.py",
        source="recursive recovery controller source",
        artifacts={module.PROGRESS_ARTIFACT: _Artifact(recursive_progress)},
        name="recursive recovery controller",
    )
    source_controller = _Task(
        SOURCE_CONTROLLER_ID,
        status="failed",
        parent=GATE_TASK_ID,
        entry_point="clearml_5090_training_controller.py",
        source="source controller source",
        artifacts={module.PROGRESS_ARTIFACT: _Artifact(source_progress)},
        name="source controller",
    )
    args = Namespace(
        training_controller_task_id=module.TRAINING_CONTROLLER_TASK_ID,
        poll_seconds=0.25,
        timeout_hours=1.0,
    )
    output = _Task(
        OUTPUT_TASK_ID,
        status="in_progress",
        parent=module.TRAINING_CONTROLLER_TASK_ID,
        entry_point=module.PRODUCER_ENTRY_POINT,
        source=module._runtime_source(),
        parameters={
            "Args/training_controller_task_id": module.TRAINING_CONTROLLER_TASK_ID,
            "Args/poll_seconds": "0.25",
            "Args/timeout_hours": "1.0",
        },
        output_uri=module.FILES_SERVER_URI,
        name="provenance output",
    )
    tasks[module.TRAINING_CONTROLLER_TASK_ID] = controller
    tasks[PARENT_CONTROLLER_ID] = parent_controller
    tasks[RECOVERY_CONTROLLER_ID] = recursive_controller
    tasks[SOURCE_CONTROLLER_ID] = source_controller

    class _TaskClass:
        current = output
        batch_calls: list[tuple[str, ...]] = []
        batch_hook = None

        @classmethod
        def get_task(cls, *, task_id: str) -> _Task:
            return tasks[task_id]

        @classmethod
        def current_task(cls) -> _Task:
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
            cls.batch_calls.append(tuple(task_ids))
            if callable(cls.batch_hook):
                cls.batch_hook(len(cls.batch_calls), tasks, output)
            available = {**tasks, output.id: output}
            return [
                copy.deepcopy(available[task_id].data.to_dict()) for task_id in task_ids
            ]

    world.module = module
    world.args = args
    world.task_class = _TaskClass
    world.output = output
    world.controller = controller
    world.parent_controller = parent_controller
    world.recursive_controller = recursive_controller
    world.source_controller = source_controller
    world.tasks = tasks
    world.task_ids = task_ids
    world.contracts = contracts
    world.final_contracts = final_contracts
    world.teacher_audits = teacher_audits
    world.progress_artifact = progress_artifact
    world.manifest_artifact = manifest_artifact
    world.recursive_progress = recursive_progress
    world.source_progress = source_progress
    world.legacy = legacy
    world.expanded = expanded
    return world


def _run(world: _World) -> dict[str, object]:
    return world.module.run(
        world.args,
        task_class=world.task_class,
        output_task=world.output,
    )


def _reseal_artifact(module, artifact: _Artifact) -> None:
    artifact.value = module._sealed(artifact.value)


def _rebind_recursive_progress(module, world: _World) -> None:
    recursive_artifact = world.recursive_controller.artifacts[
        module.PROGRESS_ARTIFACT
    ]
    recursive_artifact.value = module._sealed(recursive_artifact.value)
    world.recursive_progress = copy.deepcopy(recursive_artifact.value)

    progress = world.progress_artifact.value
    recovery = progress["recovery"]
    recovery["source_progress_revision"] = recursive_artifact.value["revision"]
    recovery["source_progress_seal_sha256"] = recursive_artifact.value[
        "seal_sha256"
    ]
    chain = recovery["source_recovery_chain"]
    chain["source_progress_revision"] = recursive_artifact.value["revision"]
    chain["source_progress_seal_sha256"] = recursive_artifact.value["seal_sha256"]
    chain["source_recovery_sha256"] = module._content_sha256(
        recursive_artifact.value["recovery"]
    )
    world.progress_artifact.value = module._sealed(progress)


def _use_schema3_pending_child_lineage(module, world: _World) -> dict[str, object]:
    subjects = ("ptf_none", "ptf_linear", "router_static")
    recursive_artifact = world.recursive_controller.artifacts[
        module.PROGRESS_ARTIFACT
    ]
    recovery = recursive_artifact.value["recovery"]
    recovery["schema_version"] = 3
    observations: dict[str, object] = {}
    for offset, subject in enumerate(subjects, start=1):
        index = module.SUBJECT_ORDER.index(subject)
        task_id = world.task_ids[index]
        predecessor_task_id = (
            world.progress_artifact.value["steps"][index]["predecessor_task_id"]
        )
        recovery["adopted_task_ids"][subject] = task_id
        recovery["adopted_predecessor_task_ids"][subject] = predecessor_task_id
        observation = {
            "task_id": task_id,
            "task_name": f"training {subject}",
            "task_status": "completed",
            "task_last_update": f"2026-08-12T00:00:0{offset}+00:00",
            "task_script_sha256": (
                module.CANONICAL_NORMALIZED_SCRIPT_IDENTITY_SHA256
            ),
            "parent_controller_task_id": SOURCE_CONTROLLER_ID,
            "source_snapshot_state": "pending",
            "source_snapshot_task_id": None,
            "predecessor_task_id": predecessor_task_id,
            "recovery_intent_source_task_id": f"{100 + offset:032x}",
            "recovery_intent_terminal_status": (
                "failed" if subject == "router_static" else "stopped"
            ),
            "provenance": "canonical_child_of_failed_recovery_controller",
        }
        observations[subject] = observation
        recursive_step = recursive_artifact.value["steps"][index]
        recursive_step["adopted"] = True
        task = world.tasks[task_id]
        task.parent = SOURCE_CONTROLLER_ID
        task.data.parent = SOURCE_CONTROLLER_ID
    recovery["recovered_pending_target_children"] = observations
    _rebind_recursive_progress(module, world)
    return observations


def test_bootstrap_equivalence_proves_text_and_ast_difference_only(monkeypatch) -> None:
    module = _load_module()
    legacy, expanded = _install_script_hashes(module, monkeypatch)

    result = module._verify_bootstrap_equivalence(
        {_digest(legacy): legacy, _digest(expanded): expanded}
    )

    assert result["verified_from_actual_script_bytes"] is True
    assert result["only_text_difference"] == ("NESTED_TEACHER_EXPERIMENTS assignment")
    assert result["legacy_nested_teacher_experiments"] == [
        "resilient_v2x",
        "support_residual",
    ]
    assert len(result["expanded_nested_teacher_experiments"]) == 9


def test_bootstrap_equivalence_rejects_difference_outside_assignment(
    monkeypatch,
) -> None:
    module = _load_module()
    legacy, expanded = _install_script_hashes(module, monkeypatch)
    expanded = expanded.replace("BOOTSTRAP_ID = 1", "BOOTSTRAP_ID = 2")
    expanded_sha = _digest(expanded)
    monkeypatch.setattr(module, "EXPANDED_BOOTSTRAP_SHA256", expanded_sha)
    monkeypatch.setattr(
        module,
        "ALLOWED_BOOTSTRAP_SHA256",
        frozenset({_digest(legacy), expanded_sha}),
    )

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="differ outside",
    ):
        module._verify_bootstrap_equivalence(
            {_digest(legacy): legacy, expanded_sha: expanded}
        )


def test_bootstrap_equivalence_rejects_wrong_nested_membership(monkeypatch) -> None:
    module = _load_module()
    legacy, expanded = _install_script_hashes(module, monkeypatch)
    expanded = expanded.replace('        "ptf_none",\n', "")
    expanded_sha = _digest(expanded)
    monkeypatch.setattr(module, "EXPANDED_BOOTSTRAP_SHA256", expanded_sha)
    monkeypatch.setattr(
        module,
        "ALLOWED_BOOTSTRAP_SHA256",
        frozenset({_digest(legacy), expanded_sha}),
    )

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="membership mismatch",
    ):
        module._verify_bootstrap_equivalence(
            {_digest(legacy): legacy, expanded_sha: expanded}
        )


def test_bootstrap_equivalence_requires_both_reviewed_actual_sources(
    monkeypatch,
) -> None:
    module = _load_module()
    legacy, _expanded = _install_script_hashes(module, monkeypatch)

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="exactly the reviewed",
    ):
        module._verify_bootstrap_equivalence({_digest(legacy): legacy})


def test_bootstrap_equivalence_rejects_any_extra_nested_teacher_usage(
    monkeypatch,
) -> None:
    module = _load_module()
    legacy, expanded = _install_script_hashes(module, monkeypatch)
    expanded = expanded.replace(
        "def bootstrap():\n",
        "def unreviewed_usage(spec):\n"
        "    return spec.name in NESTED_TEACHER_EXPERIMENTS\n\n"
        "def bootstrap():\n",
    )
    expanded_sha = _digest(expanded)
    monkeypatch.setattr(module, "EXPANDED_BOOTSTRAP_SHA256", expanded_sha)
    monkeypatch.setattr(
        module,
        "ALLOWED_BOOTSTRAP_SHA256",
        frozenset({_digest(legacy), expanded_sha}),
    )

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="one Store and two Load",
    ):
        module._verify_bootstrap_equivalence(
            {_digest(legacy): legacy, expanded_sha: expanded}
        )


def test_source_revision_transition_accepts_the_controller_contract(
    monkeypatch,
) -> None:
    module = _load_module()
    transition = _source_revision_transition(module, monkeypatch)

    validated = module._validate_source_revision_transition(transition)

    assert validated["subject_policy"]["source_template_adoptable_count"] == 21
    assert validated["subject_policy"]["target_template_required_count"] == 5
    assert [
        item["experiment"]
        for item in validated["subject_policy"]["legacy_script_adoption_exceptions"]
    ] == ["support_residual", "no_distillation"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("cross_splice", "cross-spliced"),
        ("parameter_delta", "unknown parameter difference"),
        ("inventory_delta", "sealed four-file delta"),
        ("subject_split", "exact 21:5 split"),
    ),
)
def test_source_revision_transition_rejects_unsealed_differences(
    monkeypatch,
    mutation: str,
    message: str,
) -> None:
    module = _load_module()
    transition = _source_revision_transition(module, monkeypatch)
    if mutation == "cross_splice":
        transition["target_template_identity"]["docker_image"] = "drifted/image"
    elif mutation == "parameter_delta":
        transition["parameter_equivalence"]["changed_keys"].append(
            "Args/training_dataset_id"
        )
    elif mutation == "inventory_delta":
        transition["inventory_equivalence"]["changed_files"][0]["path"] = (
            "configs/resilient_v2x/unsealed.py"
        )
    else:
        transition["subject_policy"]["source_template_adoptable_count"] = 20
    transition = module._sealed(transition)
    monkeypatch.setattr(
        module,
        "SOURCE_REVISION_TRANSITION_SEAL_SHA256",
        transition["seal_sha256"],
    )

    with pytest.raises(module.FormalTrainingProvenanceError, match=message):
        module._validate_source_revision_transition(transition)


def test_schema4_recovery_accepts_all_old_and_partial_new_adoptions(
    monkeypatch,
) -> None:
    module = _load_module()
    recovery, target_identity, steps = _schema4_recovery_fixture(module, monkeypatch)

    module._validate_source_revision_recovery(
        recovery,
        controller_task_id="a" * 32,
        progress_template=target_identity,
        steps=steps,
    )

    assert len(recovery["adopted_task_ids"]) == 24


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "field inventory mismatch"),
        ("stale_revision", "contract drifted"),
        ("seal_splice", "contract drifted"),
        ("unstable", "contract drifted"),
        ("wrong_controller", "not bound to the source controller"),
    ),
)
def test_schema4_recovery_rejects_progress_artifact_readback_drift(
    monkeypatch,
    mutation: str,
    message: str,
) -> None:
    module = _load_module()
    recovery, target_identity, steps = _schema4_recovery_fixture(module, monkeypatch)
    readback = recovery["source_progress_artifact_readback"]
    if mutation == "missing":
        readback.pop("stable_readbacks")
    elif mutation == "stale_revision":
        readback["revision"] = 2
    elif mutation == "seal_splice":
        readback["seal_sha256"] = "6" * 64
    elif mutation == "unstable":
        readback["stable_readbacks"] = 1
    else:
        readback["url"] = readback["url"].replace(
            module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
            "9" * 32,
        )

    with pytest.raises(module.FormalTrainingProvenanceError, match=message):
        module._validate_source_revision_recovery(
            recovery,
            controller_task_id="a" * 32,
            progress_template=target_identity,
            steps=steps,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "field inventory mismatch"),
        ("tampered", "seal SHA-256 mismatch"),
        ("cross_spliced", "seal SHA-256 mismatch"),
    ),
)
def test_schema4_recovery_rejects_template_equivalence_drift(
    monkeypatch,
    mutation: str,
    message: str,
) -> None:
    module = _load_module()
    recovery, target_identity, steps = _schema4_recovery_fixture(module, monkeypatch)
    if mutation == "missing":
        recovery.pop("source_progress_template_equivalence")
    elif mutation == "tampered":
        recovery["source_progress_template_equivalence"]["result"] = "fail"
    else:
        recovery["source_progress_template_equivalence"]["expected_identity"][
            "task_id"
        ] = module.SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID

    with pytest.raises(module.FormalTrainingProvenanceError, match=message):
        module._validate_source_revision_recovery(
            recovery,
            controller_task_id="a" * 32,
            progress_template=target_identity,
            steps=steps,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_old", "exact 21:5 split"),
        ("role_splice", "template role"),
        ("receipt_splice", "binding receipt"),
        ("rerun_old", "rerun an old-source subject"),
        ("replace_old", "outside the five targets"),
        ("missing_target_predecessor", "exact five targets"),
        ("target_predecessor_splice", "cross-spliced"),
        ("target_step_splice", "cross-spliced"),
    ),
)
def test_schema4_recovery_rejects_cross_revision_splicing(
    monkeypatch,
    mutation: str,
    message: str,
) -> None:
    module = _load_module()
    recovery, target_identity, steps = _schema4_recovery_fixture(module, monkeypatch)
    subject = module.SOURCE_REVISION_SUBJECTS["old"][0]
    if mutation == "missing_old":
        for key in (
            "adopted_task_ids",
            "adopted_predecessor_task_ids",
            "adopted_task_template_roles",
            "adopted_task_binding_receipts",
        ):
            recovery[key].pop(subject)
    elif mutation == "role_splice":
        recovery["adopted_task_template_roles"][subject] = "target"
    elif mutation == "receipt_splice":
        recovery["adopted_task_binding_receipts"][subject]["template_task_id"] = (
            module.SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID
        )
    elif mutation == "rerun_old":
        recovery["rerun_experiments"] = [subject]
    elif mutation == "replace_old":
        recovery["transition_replaced_source_tasks"] = {subject: {}}
    elif mutation == "missing_target_predecessor":
        recovery["target_template_predecessor_task_ids"].pop(
            module.SOURCE_REVISION_SUBJECTS["new"][0]
        )
    elif mutation == "target_predecessor_splice":
        recovery["target_template_predecessor_task_ids"][
            module.SOURCE_REVISION_SUBJECTS["new"][0]
        ] = "9" * 32
    else:
        target = module.SOURCE_REVISION_SUBJECTS["new"][0]
        next(step for step in steps if step["experiment"] == target)[
            "predecessor_task_id"
        ] = "9" * 32

    with pytest.raises(module.FormalTrainingProvenanceError, match=message):
        module._validate_source_revision_recovery(
            recovery,
            controller_task_id="a" * 32,
            progress_template=target_identity,
            steps=steps,
        )


def test_run_contract_accepts_canonical_seed_without_aliases() -> None:
    module = _load_module()
    contract = _run_contract(module, "ptf_none", _task_id(2), _task_id(1))

    assert "training_seed" not in contract
    assert "training_overlay_protocol_seed" not in contract
    assert (
        module._validate_run_contract(
            contract,
            subject="ptf_none",
            task_id=_task_id(2),
            predecessor_task_id=_task_id(1),
        )
        == module._content_sha256(contract)
    )


def test_run_publishes_sealed_26_task_provenance(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)

    payload = _run(world)

    module._require_seal(payload, context="test provenance")
    assert payload["passed"] is True
    assert payload["controller_task_id"] == module.TRAINING_CONTROLLER_TASK_ID
    assert (
        payload["progress_seal_sha256"]
        == (world.progress_artifact.value["seal_sha256"])
    )
    assert (
        payload["formal_training_manifest_seal_sha256"]
        == (world.manifest_artifact.value["seal_sha256"])
    )
    records = payload["training_tasks"]
    assert len(records) == 26
    assert [record["subject"] for record in records] == list(module.SUBJECT_ORDER)
    records_by_subject = {record["subject"]: record for record in records}
    for subject, task_id in zip(module.SUBJECT_ORDER, world.task_ids, strict=True):
        record = records_by_subject[subject]
        assert record["observed_parameter_keys"] == sorted(
            world.tasks[task_id].parameters
        )
        assert record["observed_parameters_sha256"] == module._content_sha256(
            world.tasks[task_id].parameters
        )
        assert record["docker_command_sha256"] == module.DOCKER_COMMAND_SHA256
        assert (
            record["normalized_task_script_identity_sha256"]
            == record["sealed_progress_script_identity_sha256"]
        )
    legacy_records = [
        record
        for record in records
        if record["script_equivalence_class"] == "legacy_nested_teacher_membership"
    ]
    assert [record["subject"] for record in legacy_records] == [
        "support_residual",
        "no_distillation",
    ]
    assert all(record["run_contract_content_sha256"] for record in records)
    assert (
        records_by_subject["support_residual"]["parent_controller_task_id"]
        == SOURCE_CONTROLLER_ID
    )
    assert (
        records_by_subject["no_distillation"]["parent_controller_task_id"]
        == PARENT_CONTROLLER_ID
    )
    assert all(
        records_by_subject[subject]["parent_controller_task_id"]
        == RECOVERY_CONTROLLER_ID
        for subject in ("ptf_none", "ptf_linear", "router_static")
    )
    assert payload["protocol_id"] == "DAIR-CAUSAL-1337-v1"
    assert payload["capacity_matched_hardware"] == {
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
    }
    assert [record["task_id"] for record in payload["recursive_progress_chain"]] == [
        module.TRAINING_CONTROLLER_TASK_ID,
        RECOVERY_CONTROLLER_ID,
        SOURCE_CONTROLLER_ID,
    ]
    parent_records = {
        record["task_id"]: record for record in payload["recovery_parent_controllers"]
    }
    assert set(parent_records) == {
        RECOVERY_CONTROLLER_ID,
        SOURCE_CONTROLLER_ID,
        PARENT_CONTROLLER_ID,
    }
    assert parent_records[RECOVERY_CONTROLLER_ID]["actual_parent_subjects"] == [
        "ptf_none",
        "ptf_linear",
        "router_static",
    ]
    assert parent_records[SOURCE_CONTROLLER_ID]["actual_parent_subjects"] == [
        "support_residual"
    ]
    assert parent_records[PARENT_CONTROLLER_ID]["actual_parent_subjects"] == [
        "no_distillation"
    ]
    assert all(
        record["terminal_status"] == "failed" for record in parent_records.values()
    )
    assert len(world.output.uploads) == 1
    assert world.output.uploads[0][0] == module.PROVENANCE_ARTIFACT
    assert world.output.flush_calls == 1
    assert world.output.tags == list(module.FORMAL_TAGS)
    assert world.output.artifacts[module.PROVENANCE_ARTIFACT].force_download_calls == [
        True,
        True,
    ]
    expected_ids = {
        module.TRAINING_CONTROLLER_TASK_ID,
        PARENT_CONTROLLER_ID,
        RECOVERY_CONTROLLER_ID,
        SOURCE_CONTROLLER_ID,
        OUTPUT_TASK_ID,
        *world.task_ids,
    }
    assert len(world.task_class.batch_calls) == 4
    assert all(set(call) == expected_ids for call in world.task_class.batch_calls)
    assert all(len(call) == len(expected_ids) for call in world.task_class.batch_calls)
    assert all(
        len(artifact.force_download_calls) >= 7
        and set(artifact.force_download_calls) == {True}
        for artifact in world.contracts.values()
    )
    assert all(
        len(artifact.force_download_calls) >= 7
        and set(artifact.force_download_calls) == {True}
        for artifacts in (world.final_contracts, world.teacher_audits)
        for artifact in artifacts.values()
    )


def test_schema3_pending_children_accept_exact_legacy_contract(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    observations = _use_schema3_pending_child_lineage(module, world)

    payload = _run(world)

    subjects = ("ptf_none", "ptf_linear", "router_static")
    records = {record["subject"]: record for record in payload["training_tasks"]}
    assert set(observations) == set(subjects)
    assert all("parent_controller_status" not in observations[item] for item in subjects)
    assert all(
        records[subject]["parent_controller_task_id"] == SOURCE_CONTROLLER_ID
        and records[subject]["parent_binding"]
        == "recovered_pending_target_children"
        and records[subject]["recovery_lineage"][-1]["decision"]
        == "recovered_pending_target_children"
        for subject in subjects
    )
    parent_records = {
        record["task_id"]: record for record in payload["recovery_parent_controllers"]
    }
    assert parent_records[SOURCE_CONTROLLER_ID]["actual_parent_subjects"] == [
        "support_residual",
        *subjects,
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("task_id", "parent observation for ptf_none drifted"),
        ("predecessor", "parent observation for ptf_none drifted"),
        ("parent", "parent observation for ptf_none drifted"),
        ("provenance", "parent observation for ptf_none drifted"),
        ("snapshot_state", "parent observation for ptf_none drifted"),
        ("snapshot_task", "parent observation for ptf_none drifted"),
        ("schema4", "parent observation for ptf_none drifted"),
        ("inline_parent_status", "parent observation for ptf_none drifted"),
        ("script_identity", "progress/raw script identity mismatch"),
    ),
)
def test_schema3_pending_child_contract_rejects_key_drift(
    monkeypatch,
    mutation: str,
    message: str,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    observations = _use_schema3_pending_child_lineage(module, world)
    observation = observations["ptf_none"]
    assert isinstance(observation, dict)
    recursive_artifact = world.recursive_controller.artifacts[
        module.PROGRESS_ARTIFACT
    ]
    if mutation == "task_id":
        observation["task_id"] = "9" * 32
    elif mutation == "predecessor":
        observation["predecessor_task_id"] = "9" * 32
    elif mutation == "parent":
        observation["parent_controller_task_id"] = PARENT_CONTROLLER_ID
    elif mutation == "provenance":
        observation["provenance"] = "unsealed_pending_child"
    elif mutation == "snapshot_state":
        observation["source_snapshot_state"] = "created"
    elif mutation == "snapshot_task":
        observation["source_snapshot_task_id"] = "9" * 32
    elif mutation == "schema4":
        recursive_artifact.value["recovery"]["schema_version"] = 4
    elif mutation == "inline_parent_status":
        observation["parent_controller_status"] = "failed"
    else:
        observation["task_script_sha256"] = "0" * 64
    _rebind_recursive_progress(module, world)

    with pytest.raises(module.FormalTrainingProvenanceError, match=message):
        _run(world)

    assert world.output.uploads == []


def test_recovery_target_adoption_still_requires_failed_parent_status(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    recursive_artifact = world.recursive_controller.artifacts[
        module.PROGRESS_ARTIFACT
    ]
    adoption = recursive_artifact.value["recovery"]["recovery_target_adoptions"][
        "no_distillation"
    ]
    adoption.pop("parent_controller_status")
    _rebind_recursive_progress(module, world)

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="parent observation for no_distillation drifted",
    ):
        _run(world)

    assert world.output.uploads == []


def test_schema3_pending_child_still_requires_authoritative_failed_parent(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    _use_schema3_pending_child_lineage(module, world)
    world.source_controller.status = "in_progress"
    world.source_controller.data.status = "in_progress"

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match=f"recovery parent {SOURCE_CONTROLLER_ID} is not terminal failed",
    ):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_missing_schema3_recovery_observation_map(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    world.progress_artifact.value["recovery"].pop("recovered_pending_target_children")
    _reseal_artifact(module, world.progress_artifact)

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="recovered_pending_target_children",
    ):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_recursive_source_progress_drift(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    artifact = world.source_controller.artifacts[module.PROGRESS_ARTIFACT]
    artifact.value["revision"] += 1
    _reseal_artifact(module, artifact)

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="does not bind source progress",
    ):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_normalized_script_identity_drift(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    world.tasks[world.task_ids[-1]].data.script.binary = "/drifted/python"

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="normalized script identity mismatch",
    ):
        _run(world)

    assert world.output.uploads == []


def test_run_requires_exact_legacy_bootstrap_subject_inventory(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    subject = "no_distillation"
    task_id = world.task_ids[module.SUBJECT_ORDER.index(subject)]
    world.tasks[task_id].data.script.diff = world.expanded
    recursive_artifact = world.recursive_controller.artifacts[module.PROGRESS_ARTIFACT]
    recursive_recovery = recursive_artifact.value["recovery"]
    recursive_recovery["recovery_target_adoptions"][subject]["task_script_sha256"] = (
        module.CANONICAL_NORMALIZED_SCRIPT_IDENTITY_SHA256
    )
    _reseal_artifact(module, recursive_artifact)
    final_recovery = world.progress_artifact.value["recovery"]
    final_recovery["source_progress_seal_sha256"] = recursive_artifact.value[
        "seal_sha256"
    ]
    final_recovery["source_recovery_chain"]["source_progress_seal_sha256"] = (
        recursive_artifact.value["seal_sha256"]
    )
    final_recovery["source_recovery_chain"]["source_recovery_sha256"] = (
        module._content_sha256(recursive_recovery)
    )
    _reseal_artifact(module, world.progress_artifact)

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="subject-to-version inventory mismatch",
    ):
        _run(world)

    assert world.output.uploads == []


@pytest.mark.parametrize("mutation", ["args", "docker", "model", "checkpoint"])
def test_run_rejects_static_training_binding_drift(
    monkeypatch,
    mutation: str,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    subject = "router_uniform"
    task_id = world.task_ids[module.SUBJECT_ORDER.index(subject)]
    task = world.tasks[task_id]
    if mutation == "args":
        task.parameters["Args/gpus"] = "2"
    elif mutation == "docker":
        task.docker += " --drift"
    elif mutation == "model":
        task.output_models[0].url += ".drift"
    else:
        world.final_contracts[subject].value["sha256"] = "0" * 64

    with pytest.raises(module.FormalTrainingProvenanceError):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_semantically_false_teacher_audit_even_when_manifest_hash_matches(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    subject = "router_static"
    audit = world.teacher_audits[subject]
    audit.value["shared_initialization"]["exact_tensor_equality_verified"] = False
    entry = world.manifest_artifact.value["entries"][
        module.SUBJECT_ORDER.index(subject)
    ]
    entry["common_teacher_initialization_audit_sha256"] = module._content_sha256(
        audit.value
    )
    _reseal_artifact(module, world.manifest_artifact)

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="shared initialization mismatch",
    ):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_nested_teacher_run_contract_drift(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    initialization = world.contracts["ptf_none"].value["common_teacher_initialization"]
    initialization["expected_nested_teacher"] = False

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="common teacher initialization contract mismatch",
    ):
        _run(world)

    assert world.output.uploads == []


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("source_dataset_id", "0" * 32),
        ("training_dataset_id", "0" * 32),
        ("native_bundle_sha256", "0" * 64),
        ("build_manifest_sha256", "0" * 64),
        ("global_batch_size", 4),
        ("max_epochs", 49),
        ("seed", 1),
        ("training_seed", 1),
        ("training_overlay_protocol_seed", 1),
        ("training_seed", "20250218"),
        ("training_overlay_protocol_seed", "20250218"),
        ("training_seed", True),
        ("training_overlay_protocol_seed", True),
        ("precision", "TF32"),
        ("val_interval", 1),
        ("amp", True),
    ],
)
def test_run_rejects_each_fixed_run_contract_field_before_publish(
    monkeypatch,
    field: str,
    bad_value: object,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    world.contracts["ptf_none"].value[field] = bad_value

    with pytest.raises(module.FormalTrainingProvenanceError, match="run_contract"):
        _run(world)

    assert world.output.uploads == []
    assert world.output.tag_calls == []


def test_run_rejects_missing_canonical_run_contract_seed(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    world.contracts["ptf_none"].value.pop("seed")

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="run_contract seed mismatch",
    ):
        _run(world)

    assert world.output.uploads == []
    assert world.output.tag_calls == []


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("source_archive", "sha256"), "0" * 64),
        (("teacher", "sha256"), "0" * 64),
        (("common_teacher_initialization", "teacher_checkpoint_sha256"), "0" * 64),
    ],
)
def test_run_rejects_source_or_teacher_hash_before_publish(
    monkeypatch,
    path: tuple[str, str],
    bad_value: str,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    world.contracts["router_static"].value[path[0]][path[1]] = bad_value

    with pytest.raises(module.FormalTrainingProvenanceError):
        _run(world)

    assert world.output.uploads == []
    assert world.output.tag_calls == []


def test_run_rejects_incomplete_progress_before_authoritative_claim(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    world.progress_artifact.value["steps"][-1]["state"] = "running"
    _reseal_artifact(module, world.progress_artifact)

    with pytest.raises(module.FormalTrainingProvenanceError, match="not the completed"):
        _run(world)

    assert world.output.uploads == []
    assert world.task_class.batch_calls == []


def test_run_rejects_manifest_task_order_drift(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    entries = world.manifest_artifact.value["entries"]
    entries[0]["training_task_id"], entries[1]["training_task_id"] = (
        entries[1]["training_task_id"],
        entries[0]["training_task_id"],
    )
    _reseal_artifact(module, world.manifest_artifact)

    with pytest.raises(module.FormalTrainingProvenanceError, match="identity mismatch"):
        _run(world)

    assert world.output.uploads == []


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_initialization_audit_artifact",
        "extra_entry_field",
        "extra_manifest_field",
    ],
)
def test_run_rejects_nonexact_manifest_field_inventory(
    monkeypatch, mutation: str
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    entry = world.manifest_artifact.value["entries"][0]
    if mutation == "missing_initialization_audit_artifact":
        entry.pop("common_teacher_initialization_audit_artifact")
    elif mutation == "extra_entry_field":
        entry["unsealed_extension"] = True
    else:
        world.manifest_artifact.value["unsealed_extension"] = True
    _reseal_artifact(module, world.manifest_artifact)

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="field inventory mismatch",
    ):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_wrong_manifest_initialization_audit_artifact(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    entry = world.manifest_artifact.value["entries"][0]
    entry["common_teacher_initialization_audit_artifact"] = "wrong_artifact"
    _reseal_artifact(module, world.manifest_artifact)

    with pytest.raises(module.FormalTrainingProvenanceError, match="identity mismatch"):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_unsealed_progress(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    world.progress_artifact.value["revision"] += 1

    with pytest.raises(module.FormalTrainingProvenanceError, match="seal SHA-256"):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_authoritative_incomplete_task(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    task = world.tasks[world.task_ids[3]]
    task.status = "in_progress"
    task.data.status = "in_progress"

    with pytest.raises(module.FormalTrainingProvenanceError, match="not completed"):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_legacy_script_on_any_other_subject(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    task = world.tasks[world.task_ids[1]]
    task.data.script.diff = world.legacy

    with pytest.raises(module.FormalTrainingProvenanceError, match="not allowed"):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_unreviewed_bootstrap(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    world.tasks[world.task_ids[-1]].data.script.diff += "\nUNREVIEWED = True\n"

    with pytest.raises(module.FormalTrainingProvenanceError, match="unreviewed"):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_current_controller_task_with_external_parent(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    task = world.tasks[world.task_ids[2]]
    task.parent = PARENT_CONTROLLER_ID
    task.data.parent = PARENT_CONTROLLER_ID

    with pytest.raises(module.FormalTrainingProvenanceError, match="not sealed"):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_adopted_task_parent_not_declared_by_recovery(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    undeclared_parent = "a" * 32
    task = world.tasks[world.task_ids[0]]
    task.parent = undeclared_parent
    task.data.parent = undeclared_parent

    with pytest.raises(module.FormalTrainingProvenanceError, match="not sealed"):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_nonterminal_recovery_parent(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    world.parent_controller.status = "in_progress"
    world.parent_controller.data.status = "in_progress"

    with pytest.raises(module.FormalTrainingProvenanceError, match="not terminal"):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_recovery_adoption_task_mismatch(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    recovery = world.progress_artifact.value["recovery"]
    recovery["adopted_task_ids"]["support_residual"] = world.task_ids[1]
    _reseal_artifact(module, world.progress_artifact)

    with pytest.raises(module.FormalTrainingProvenanceError, match="does not match"):
        _run(world)

    assert world.output.uploads == []


def test_run_rejects_fresh_run_contract_content_change(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    artifact = world.contracts["ptf_linear"]
    calls = 0

    def mutate_after_first_read(current: _Artifact) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            current.value["extra_drift"] = True

    artifact.on_get = mutate_after_first_read

    with pytest.raises(module.FormalTrainingProvenanceError, match="fresh readbacks"):
        _run(world)

    assert world.output.uploads == []
    assert world.output.tag_calls == []


def test_run_detects_dependency_raw_toctou_after_upload_and_does_not_tag(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)

    def mutate_on_committed_batch(call: int, tasks, _output) -> None:
        if call == 3:
            tasks[world.task_ids[4]].data.comment = "terminal drift"

    world.task_class.batch_hook = mutate_on_committed_batch

    with pytest.raises(module.FormalTrainingProvenanceError, match="dependency"):
        _run(world)

    assert len(world.output.uploads) == 1
    assert world.output.tags == []
    assert world.output.tag_calls == []


def test_run_allows_autonomous_output_lifecycle_metadata_updates(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)

    def update_output_lifecycle(call: int, _tasks, output) -> None:
        if call == 2:
            output.data.status_message = "worker heartbeat updated"
            output.data.status_reason = "agent lifecycle update"
            output.data.last_worker = "clearml-agent-runtime-worker"

    world.task_class.batch_hook = update_output_lifecycle

    payload = _run(world)

    assert payload["passed"] is True
    assert world.output.tags == list(module.FORMAL_TAGS)
    assert world.output.data.status_message == "worker heartbeat updated"
    assert world.output.data.status_reason == "agent lifecycle update"
    assert world.output.data.last_worker == "clearml-agent-runtime-worker"


def test_run_accepts_reordered_formal_tags_as_idempotent(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    first = _run(world)
    reordered = list(reversed(module.FORMAL_TAGS))
    world.output.tags = list(reordered)
    world.output.data.tags = list(reordered)
    world.output.uploads.clear()
    world.output.tag_calls.clear()
    world.task_class.batch_calls.clear()

    second = _run(world)

    assert second == first
    assert world.output.uploads == []
    assert world.output.tag_calls == []
    assert world.output.tags == reordered
    assert len(world.task_class.batch_calls) == 4


def test_run_accepts_authoritative_tags_when_set_tags_callback_raises(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    desired = list(module.FORMAL_TAGS)
    authoritative = list(reversed(desired))

    def commit_then_raise(tags: list[str]) -> bool:
        assert tags == desired
        world.output.tag_calls.append(list(tags))
        world.output.tags = list(authoritative)
        world.output.data.tags = list(authoritative)
        raise RuntimeError("response lost after commit")

    world.output.set_tags = commit_then_raise

    payload = _run(world)

    assert payload["passed"] is True
    assert world.output.tag_calls == [desired]
    assert world.output.tags == authoritative
    assert len(world.task_class.batch_calls) == 4


def test_run_accepts_formal_tags_visible_on_fifth_authoritative_readback(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    desired = list(module.FORMAL_TAGS)
    authoritative = list(reversed(desired))

    def record_without_immediate_commit(tags: list[str]) -> bool:
        assert tags == desired
        world.output.tag_calls.append(list(tags))
        return True

    def commit_on_fifth_readback(call: int, _tasks, output) -> None:
        if call == 3 + module.TAG_AUTHORITY_READBACK_ATTEMPTS:
            output.tags = list(authoritative)
            output.data.tags = list(authoritative)

    world.output.set_tags = record_without_immediate_commit
    world.task_class.batch_hook = commit_on_fifth_readback

    payload = _run(world)

    assert payload["passed"] is True
    assert world.output.tags == authoritative
    assert len(world.task_class.batch_calls) == (
        3 + module.TAG_AUTHORITY_READBACK_ATTEMPTS
    )


def test_run_times_out_tag_commit_and_confirms_compensation_authoritatively(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    desired = list(module.FORMAL_TAGS)

    def lose_both_callbacks(tags: list[str]) -> bool:
        world.output.tag_calls.append(list(tags))
        if tags == desired:
            return True
        assert tags == []
        raise RuntimeError("compensation response lost")

    world.output.set_tags = lose_both_callbacks

    with pytest.raises(module.FormalTrainingProvenanceError, match="not committed"):
        _run(world)

    assert world.output.tag_calls == [desired, []]
    assert world.output.tags == []
    assert len(world.task_class.batch_calls) == (
        4 + module.TAG_AUTHORITY_READBACK_ATTEMPTS
    )


@pytest.mark.parametrize(
    ("terminal_state", "cause"),
    [
        ("unknown", "unknown state"),
        ("duplicate", "duplicate tags"),
    ],
)
def test_run_rejects_unknown_or_duplicate_terminal_tag_state_without_retry(
    monkeypatch,
    terminal_state: str,
    cause: str,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)

    def inject_invalid_state(call: int, _tasks, output) -> None:
        if call != 4:
            return
        if terminal_state == "unknown":
            tags = ["attacker"]
        else:
            tags = [*module.FORMAL_TAGS, module.FORMAL_TAGS[0]]
        output.tags = list(tags)
        output.data.tags = list(tags)

    world.task_class.batch_hook = inject_invalid_state

    with pytest.raises(
        module.FormalTrainingProvenanceError, match="not committed"
    ) as captured:
        _run(world)

    assert cause in str(captured.value.__cause__)
    assert world.output.tag_calls == [list(module.FORMAL_TAGS), []]
    assert world.output.tags == []
    assert len(world.task_class.batch_calls) == 5


def test_run_confirms_delayed_tag_compensation_on_fifth_readback(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    desired = list(module.FORMAL_TAGS)

    def commit_but_delay_restore(tags: list[str]) -> bool:
        world.output.tag_calls.append(list(tags))
        if tags == desired:
            world.output.tags = list(desired)
            world.output.data.tags = list(desired)
        else:
            assert tags == []
        return True

    def force_terminal_failure_then_restore_on_fifth(call: int, tasks, output) -> None:
        if call == 4:
            tasks[world.task_ids[5]].data.comment = "terminal drift"
        elif call == 5:
            tasks[world.task_ids[5]].data.comment = ""
        if call == 4 + module.TAG_AUTHORITY_READBACK_ATTEMPTS:
            output.tags = []
            output.data.tags = []

    world.output.set_tags = commit_but_delay_restore
    world.task_class.batch_hook = force_terminal_failure_then_restore_on_fifth

    with pytest.raises(module.FormalTrainingProvenanceError, match="not committed"):
        _run(world)

    assert world.output.tag_calls == [desired, []]
    assert world.output.tags == []
    assert len(world.task_class.batch_calls) == (
        4 + module.TAG_AUTHORITY_READBACK_ATTEMPTS
    )


def test_run_rejects_compensation_not_visible_after_five_readbacks(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    desired = list(module.FORMAL_TAGS)

    def commit_but_never_restore(tags: list[str]) -> bool:
        world.output.tag_calls.append(list(tags))
        if tags == desired:
            world.output.tags = list(desired)
            world.output.data.tags = list(desired)
        else:
            assert tags == []
        return True

    def force_terminal_failure(call: int, tasks, _output) -> None:
        if call == 4:
            tasks[world.task_ids[5]].data.comment = "terminal drift"
        elif call == 5:
            tasks[world.task_ids[5]].data.comment = ""

    world.output.set_tags = commit_but_never_restore
    world.task_class.batch_hook = force_terminal_failure

    with pytest.raises(
        module.FormalTrainingProvenanceError, match="compensation failed"
    ):
        _run(world)

    assert world.output.tag_calls == [desired, []]
    assert world.output.tags == desired
    assert len(world.task_class.batch_calls) == (
        4 + module.TAG_AUTHORITY_READBACK_ATTEMPTS
    )


def test_run_rejects_duplicate_output_tags_before_publish(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    duplicate_tags = [*module.FORMAL_TAGS, module.FORMAL_TAGS[0]]
    world.output.tags = list(duplicate_tags)
    world.output.data.tags = list(duplicate_tags)

    with pytest.raises(module.FormalTrainingProvenanceError, match="duplicate tags"):
        _run(world)

    assert world.output.uploads == []
    assert world.output.tag_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status_message", "dependency heartbeat drift"),
        ("status_reason", "dependency reason drift"),
        ("last_worker", "dependency-worker-drift"),
    ],
)
def test_run_rejects_dependency_lifecycle_metadata_drift(
    monkeypatch,
    field: str,
    value: str,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)

    def mutate_dependency_lifecycle(call: int, tasks, _output) -> None:
        if call == 2:
            setattr(tasks[world.task_ids[4]].data, field, value)

    world.task_class.batch_hook = mutate_dependency_lifecycle

    with pytest.raises(module.FormalTrainingProvenanceError, match="dependency"):
        _run(world)

    assert world.output.uploads == []
    assert world.output.tag_calls == []


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "status",
        "parent",
        "project",
        "script",
        "output",
        "execution",
        "hyperparams",
        "configuration",
    ],
)
def test_run_rejects_substantive_output_metadata_drift_between_snapshots(
    monkeypatch,
    field: str,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)

    def mutate_output_metadata(call: int, _tasks, output) -> None:
        if call != 2:
            return
        if field == "name":
            output.data.name = "drifted output"
        elif field == "status":
            output.data.status = "completed"
        elif field == "parent":
            output.data.parent = PARENT_CONTROLLER_ID
        elif field == "project":
            output.data.project = "9" * 32
        elif field == "script":
            output.data.script.diff += "\nDRIFT = True\n"
        elif field == "output":
            output.data.output.destination = "http://wrong.invalid"
        elif field == "execution":
            output.data.execution.queue = "9" * 32
        elif field == "hyperparams":
            output.data.hyperparams["Args/timeout_hours"] = "2.0"
        else:
            output.data.configuration["unsealed"] = "drift"

    world.task_class.batch_hook = mutate_output_metadata

    with pytest.raises(
        module.FormalTrainingProvenanceError,
        match="provenance output metadata changed",
    ):
        _run(world)

    assert world.output.uploads == []
    assert world.output.tag_calls == []


def test_run_compensates_tags_when_terminal_batch_changes_dependency(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)

    def mutate_during_terminal_then_restore(call: int, tasks, _output) -> None:
        if call == 4:
            tasks[world.task_ids[5]].data.comment = "terminal drift"
        elif call == 5:
            tasks[world.task_ids[5]].data.comment = ""

    world.task_class.batch_hook = mutate_during_terminal_then_restore

    with pytest.raises(module.FormalTrainingProvenanceError, match="not committed"):
        _run(world)

    assert world.output.tag_calls == [list(module.FORMAL_TAGS), []]
    assert world.output.tags == []


def test_run_rejects_output_artifact_fresh_readback_drift_without_tags(
    monkeypatch,
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    original_upload = world.output.upload_artifact

    def upload_with_drifting_readback(*args, **kwargs):
        result = original_upload(*args, **kwargs)
        artifact = world.output.artifacts[module.PROVENANCE_ARTIFACT]

        def mutate(current: _Artifact) -> None:
            current.value["passed"] = False

        artifact.on_get = mutate
        return result

    world.output.upload_artifact = upload_with_drifting_readback

    with pytest.raises(module.FormalTrainingProvenanceError, match="readback mismatch"):
        _run(world)

    assert world.output.tags == []
    assert world.output.tag_calls == []


@pytest.mark.parametrize(
    "mutation",
    ["parent", "source", "args", "output_uri", "tags"],
)
def test_run_rejects_output_contract_drift_before_publish(
    monkeypatch, mutation: str
) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    if mutation == "parent":
        world.output.parent = PARENT_CONTROLLER_ID
        world.output.data.parent = PARENT_CONTROLLER_ID
    elif mutation == "source":
        world.output.data.script.diff += "\nDRIFT = True\n"
    elif mutation == "args":
        world.output.parameters["Args/timeout_hours"] = "2.0"
    elif mutation == "output_uri":
        world.output.data.output.destination = "http://wrong.invalid"
    else:
        world.output.tags = ["premature"]
        world.output.data.tags = ["premature"]

    with pytest.raises(module.FormalTrainingProvenanceError):
        _run(world)

    assert world.output.uploads == []


def test_run_is_idempotent_for_exact_artifact_and_formal_tags(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    first = _run(world)
    world.output.uploads.clear()
    world.output.tag_calls.clear()
    world.task_class.batch_calls.clear()

    second = _run(world)

    assert second == first
    assert world.output.uploads == []
    assert world.output.tag_calls == []
    assert world.output.tags == list(module.FORMAL_TAGS)
    assert len(world.task_class.batch_calls) == 4


def test_run_accepts_a_sealed_successor_controller(monkeypatch) -> None:
    module = _load_module()
    world = _world(module, monkeypatch)
    old_controller_id = module.TRAINING_CONTROLLER_TASK_ID
    successor_id = "a" * 32
    world.args.training_controller_task_id = successor_id
    world.controller.id = successor_id
    world.controller.data.id = successor_id
    world.progress_artifact.value["controller_task_id"] = successor_id
    _reseal_artifact(module, world.progress_artifact)
    world.output.parent = successor_id
    world.output.data.parent = successor_id
    world.output.parameters["Args/training_controller_task_id"] = successor_id
    for task in world.tasks.values():
        if task.parent == old_controller_id:
            task.parent = successor_id
            task.data.parent = successor_id
    world.tasks[successor_id] = world.tasks.pop(old_controller_id)

    payload = _run(world)

    assert payload["controller_task_id"] == successor_id
