from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_ID = "1" * 32
GATE_ID = "2" * 32
TEMPLATE_ID = "3" * 32
TEACHER_TASK_ID = "4" * 32
TEACHER_MODEL_ID = "5" * 32
TEACHER_SHA = "a" * 64
SOURCE_DATASET_ID = "6" * 32
TRAINING_DATASET_ID = "7" * 32
SOURCE_SHA = "b" * 64
NATIVE_SHA = "c" * 64
BUILD_SHA = "d" * 64


def _load_controller():
    path = ROOT / "tools/resilient_v2x/clearml_5090_training_controller.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Artifact:
    def __init__(
        self,
        payload: object,
        *,
        url: str = "memory://artifact",
    ) -> None:
        self.payload = copy.deepcopy(payload)
        self.url = url
        self.force_download_calls = 0

    def get(self, *, force_download: bool = False):
        if force_download:
            self.force_download_calls += 1
        return copy.deepcopy(self.payload)


class _StaleUnlessForcedArtifact(_Artifact):
    def __init__(
        self,
        *,
        cached_payload: object,
        fresh_payload: object,
        url: str,
    ) -> None:
        super().__init__(fresh_payload, url=url)
        self.cached_payload = copy.deepcopy(cached_payload)

    def get(self, *, force_download: bool = False):
        if force_download:
            self.force_download_calls += 1
            return copy.deepcopy(self.payload)
        return copy.deepcopy(self.cached_payload)


class _Model:
    def __init__(
        self,
        *,
        name: str,
        task: str,
        model_id: str,
        url: str,
        local_copy: Path | None = None,
    ) -> None:
        self.name = name
        self.task = task
        self.id = model_id
        self.url = url
        self.local_copy = local_copy
        self.download_count = 0

    def get_local_copy(
        self,
        *,
        extract_archive: bool,
        raise_on_error: bool,
        force_download: bool,
    ) -> str | None:
        assert extract_archive is False
        assert raise_on_error is True
        assert force_download is True
        self.download_count += 1
        return str(self.local_copy) if self.local_copy is not None else None


def _initialization_contract(
    module, experiment: str, teacher_sha256: str = TEACHER_SHA
) -> dict[str, object]:
    return {
        "policy": "shared-only",
        "contract": module.COMMON_TEACHER_INITIALIZATION_CONTRACT,
        "shared_prefixes": list(module.COMMON_TEACHER_INITIALIZATION_PREFIXES),
        "expected_source_keys": 617,
        "expected_source_numel": 35_811_485,
        "expected_source_bytes": 143_246_244,
        "expected_shared_keys": 468,
        "expected_shared_numel": 31_506_934,
        "expected_shared_bytes": 126_028_040,
        "expected_teacher_fusion_keys": 149,
        "teacher_checkpoint_sha256": teacher_sha256,
        "audit_artifact_name": module.COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT,
        "audit_filename": "common_teacher_initialization_audit.json",
        "expected_nested_teacher": experiment in module.NESTED_TEACHER_EXPERIMENTS,
    }


def _initialization_audit(
    module, experiment: str, teacher_sha256: str = TEACHER_SHA
) -> dict[str, object]:
    nested = experiment in module.NESTED_TEACHER_EXPERIMENTS
    fusion_keys = 0 if experiment in {"ego_only", "fcooper"} else 7
    fusion_numel = 0 if fusion_keys == 0 else 1024
    fusion_bytes = 0 if fusion_keys == 0 else 4096
    return {
        "schema_version": 1,
        "contract": module.COMMON_TEACHER_INITIALIZATION_CONTRACT,
        "result": "pass",
        "checkpoint": {
            "path": "/tmp/teacher.pth",
            "filename": "teacher.pth",
            "size_bytes": 143_246_244,
            "sha256": teacher_sha256,
            "expected_sha256": teacher_sha256,
        },
        "source": {
            "keys": 617,
            "numel": 35_811_485,
            "bytes": 143_246_244,
            "expected_keys": 617,
            "common_keys": 468,
            "expected_common_keys": 468,
            "fusion_keys": 149,
            "expected_fusion_keys": 149,
            "state_sha256": "b" * 64,
        },
        "shared_initialization": {
            "prefixes": list(module.COMMON_TEACHER_INITIALIZATION_PREFIXES),
            "keys": 468,
            "numel": 31_506_934,
            "bytes": 126_028_040,
            "expected_keys": 468,
            "state_sha256": "c" * 64,
            "shape_dtype_verified": True,
            "exact_tensor_equality_verified": True,
        },
        "method_specific_fusion": {
            "keys": fusion_keys,
            "numel": fusion_numel,
            "bytes": fusion_bytes,
            "sha256_before": "d" * 64,
            "sha256_after": "d" * 64,
            "unchanged": True,
        },
        "target": {
            "model_type": "ControlledModel",
            "target_key_count": 468 + fusion_keys + (617 if nested else 0),
            "target_common_key_count": 468,
            "target_fusion_key_count": fusion_keys,
            "nested_teacher_present": nested,
            "nested_teacher_key_count": 617 if nested else 0,
            "nested_teacher_full_equality_verified": nested,
        },
    }


class _FakeTask:
    def __init__(
        self,
        owner: "_FakeTasks",
        *,
        task_id: str,
        name: str,
        status: str,
        parameters: dict[str, object] | None = None,
        script: dict[str, object] | None = None,
        docker: str = "",
        parent: str = "",
        project: str = "ResilientV2X/Training",
        models: list[_Model] | None = None,
    ) -> None:
        self.owner = owner
        self.id = task_id
        self.name = name
        self.status = status
        self.parameters = dict(parameters or {})
        self.script = dict(script or {})
        self.docker = docker
        self.parent = parent
        self.project = project
        self.artifacts: dict[str, _Artifact] = {}
        self.models = list(models or [])
        self.data = SimpleNamespace(
            script=SimpleNamespace(to_dict=lambda: dict(self.script)),
            last_update="2026-08-10T00:00:00+00:00",
            last_worker=None,
            execution=SimpleNamespace(queue=None),
        )
        self.output_uri = None

    def get_parameters(self, **_kwargs):
        return dict(self.parameters)

    def get_project_name(self) -> str:
        return self.project

    def get_executed_queue(self, *, return_name: bool = False) -> str | None:
        del return_name
        return self.data.execution.queue

    def set_parameters(self, parameters: dict[str, object]) -> None:
        self.parameters = dict(parameters)
        experiment = self.parameters["Args/experiment_from_task"]
        self.owner.events.append(f"override:{experiment}")

    def get_script(self):
        return {
            key: self.script.get(key)
            for key in ("working_dir", "entry_point", "branch", "repository")
        }

    def get_base_docker(self) -> str:
        return self.docker

    def get_models(self):
        return {"output": list(self.models)}

    def reload(self) -> None:
        experiment = self.parameters.get("Args/experiment_from_task")
        if not experiment:
            return
        if self.status == "queued":
            self.status = "in_progress"
            self.owner.events.append(f"running:{experiment}")
        elif self.status == "in_progress":
            if experiment in self.owner.hold_experiments:
                return
            if experiment == self.owner.fail_experiment:
                self.status = "failed"
                self.owner.active.discard(self.id)
                self.owner.events.append(f"failed:{experiment}")
            else:
                self.status = "completed"
                self.owner.active.discard(self.id)
                self.owner.events.append(f"complete:{experiment}")
                self._publish_experiment_outputs(experiment)

    def _publish_experiment_outputs(self, experiment: str) -> None:
        params = self.parameters
        spec = self.owner.module.EXPERIMENT_BY_NAME[experiment]
        requires_teacher = spec.requires_teacher
        teacher = None
        if requires_teacher:
            teacher = {
                "task_id": params["Args/teacher_task_id"],
                "model_id": params["Args/teacher_model_id"],
                "sha256": params["Args/teacher_checkpoint_sha256"],
            }
        run_contract = {
            "schema_version": 1,
            "mode": "experiment_from_task",
            "task_id": self.id,
            "experiment": experiment,
            "experiment_kind": spec.kind,
            "config": {"declared": spec.config},
            "source_dataset_id": params["Args/source_dataset_id"],
            "training_dataset_id": params["Args/training_dataset_id"],
            "native_build_task_id": self.owner.module.BUILD_TASK_ID,
            "native_bundle_sha256": params["Args/native_bundle_sha256"],
            "build_manifest_sha256": params["Args/build_manifest_sha256"],
            "base_image_manifest_digest": (
                self.owner.module.BASE_IMAGE_AMD64_MANIFEST_DIGEST
            ),
            "source_archive": {
                "name": params["Args/source_archive_name"],
                "size_bytes": int(str(params["Args/source_archive_bytes"])),
                "sha256": params["Args/source_archive_sha256"],
            },
            "predecessor_task_id": params["Args/predecessor_task_id"],
            "gpus": 4,
            "ddp_processes": 4,
            "max_epochs": 50,
            "seed": params["Args/training_seed"],
            "amp": False,
            "precision": "FP32",
            "runtime_profile": "rtx5090",
            "val_interval": 10,
            "per_epoch_validation": False,
            "condition_evaluation": False,
            "teacher": teacher,
            "common_teacher_initialization": _initialization_contract(
                self.owner.module,
                experiment,
                str(params["Args/teacher_checkpoint_sha256"]),
            ),
        }
        model_id = f"{int(self.id, 16) + 1000:032x}"
        model_name = f"ResilientV2X {experiment} final checkpoint"
        if experiment == self.owner.invalid_model_experiment:
            model_name = "wrong model"
        model_url = f"http://10.100.34.118:8081/models/{experiment}_epoch_50.pth"
        model = _Model(
            name=model_name,
            task=self.id,
            model_id=model_id,
            url=model_url,
        )
        self.models = [model]
        initialization_audit = _initialization_audit(
            self.owner.module,
            experiment,
            str(params["Args/teacher_checkpoint_sha256"]),
        )
        if experiment == self.owner.audit_drift_experiment:
            initialization_audit["result"] = "fail"
        self.artifacts = {
            "run_contract": _Artifact(run_contract),
            "final_checkpoint_contract": _Artifact(
                {
                    "model_id": model_id,
                    "name": f"ResilientV2X {experiment} final checkpoint",
                    "url": model_url,
                    "filename": "epoch_50.pth",
                    "size_bytes": 123456,
                    "sha256": "e" * 64,
                }
            ),
            self.owner.module.COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT: (
                _Artifact(initialization_audit)
            ),
        }
        if experiment == self.owner.omit_audit_experiment:
            del self.artifacts[
                self.owner.module.COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT
            ]

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        assert wait_on_upload is True
        self.artifacts[name] = _Artifact(
            artifact_object,
            url=f"memory://{self.id}/{name}",
        )
        self.owner.events.append(f"artifact:{name}")
        return True

    def flush(self, *, wait_for_uploads: bool) -> None:
        assert wait_for_uploads is True


class _FakeTasks:
    def __init__(self, module, *, gate_status: str = "completed") -> None:
        self.module = module
        self.registry: dict[str, _FakeTask] = {}
        self.events: list[str] = []
        self.active: set[str] = set()
        self.clone_counts: dict[str, int] = {}
        self.fail_experiment: str | None = None
        self.invalid_model_experiment: str | None = None
        self.audit_drift_experiment: str | None = None
        self.omit_audit_experiment: str | None = None
        self.hold_experiments: set[str] = set()
        self.enqueue_failures = 0
        self.enqueue_records: list[tuple[str, str]] = []
        self.next_id = 100
        self.controller = self._add(
            task_id=CONTROLLER_ID,
            name="controller",
            status="in_progress",
        )
        self.gate = self._add(
            task_id=GATE_ID,
            name="validation gate",
            status=gate_status,
        )
        teacher_model = _Model(
            name="ResilientV2X clean teacher",
            task=TEACHER_TASK_ID,
            model_id=TEACHER_MODEL_ID,
            url="http://10.100.34.118:8081/models/teacher_epoch_50.pth",
        )
        self.teacher = self._add(
            task_id=TEACHER_TASK_ID,
            name="teacher",
            status="failed",
            models=[teacher_model],
        )
        self.template = self._add(
            task_id=TEMPLATE_ID,
            name="verified bootstrap template",
            status="completed",
            parameters=_template_parameters(),
            script=_template_script(module),
            docker=_template_docker(module),
        )

    def _add(self, **kwargs) -> _FakeTask:
        task = _FakeTask(self, **kwargs)
        self.registry[task.id] = task
        return task

    def get_task(self, *, task_id: str):
        return self.registry[task_id]

    def get_tasks(
        self,
        *,
        project_name: str,
        task_name: str,
        allow_archived: bool = True,
        task_filter: dict[str, object] | None = None,
    ):
        assert allow_archived is True
        parent = (task_filter or {}).get("parent")
        return [
            task
            for task in self.registry.values()
            if task.project == project_name
            and re.fullmatch(task_name, task.name)
            and (parent is None or task.parent == parent)
        ]

    def clone(
        self,
        *,
        source_task: _FakeTask,
        name: str,
        parent: str,
    ):
        experiment = name.split()[3]
        self.events.append(f"clone:{experiment}")
        self.clone_counts[experiment] = self.clone_counts.get(experiment, 0) + 1
        task_id = f"{self.next_id:032x}"
        self.next_id += 1
        docker = source_task.docker
        if experiment == getattr(self, "drift_clone_experiment", None):
            docker = docker.replace("--network host", "--network bridge")
        return self._add(
            task_id=task_id,
            name=name,
            status="created",
            parameters=source_task.parameters,
            script=source_task.script,
            docker=docker,
            parent=parent,
            project=source_task.project,
        )

    def enqueue(self, *, task: _FakeTask, queue_name: str):
        assert queue_name in {"GPU4-5090", "GPU4-A100", "GPU4-V100"}
        if self.enqueue_failures:
            self.enqueue_failures -= 1
            self.events.append("enqueue-error")
            raise RuntimeError("simulated enqueue interruption")
        if getattr(self, "allow_parallel", False):
            pass
        else:
            assert not self.active, "the controller attempted parallel training"
        experiment = task.parameters["Args/experiment_from_task"]
        self.events.append(f"enqueue:{experiment}")
        self.enqueue_records.append((experiment, queue_name))
        self.active.add(task.id)
        task.data.last_worker = {
            "GPU4-A100": "10.100.34.18-A100:gpu0,1,2,3",
            "GPU4-V100": "10.100.34.26-V100:gpu0,1,2,3",
            "GPU4-5090": "10.100.34.130-5090:gpu4,5,6,7",
        }[queue_name]
        task.data.execution.queue = queue_name
        task.status = "queued"
        return {"queued": 1, "updated": 1}


def _template_parameters(**overrides: object) -> dict[str, object]:
    parameters = {
        "Args/source_dataset_id": SOURCE_DATASET_ID,
        "Args/source_archive_name": "source.tar.zst",
        "Args/source_archive_bytes": "123",
        "Args/source_archive_sha256": SOURCE_SHA,
        "Args/training_dataset_id": TRAINING_DATASET_ID,
        "Args/native_bundle_bytes": "456",
        "Args/native_bundle_sha256": NATIVE_SHA,
        "Args/build_manifest_sha256": BUILD_SHA,
        "Args/stage": "validate",
        "Args/gpus": 4,
        "Args/max_epochs": 50,
        "Args/student_task_id": "8" * 32,
        "Args/student_model_id": "9" * 32,
        "Args/student_checkpoint_sha256": "f" * 64,
        "Args/teacher_task_id": TEACHER_TASK_ID,
        "Args/teacher_model_id": TEACHER_MODEL_ID,
        "Args/teacher_checkpoint_sha256": TEACHER_SHA,
        "General/retained_template_parameter": "sealed",
    }
    parameters.update(overrides)
    return parameters


def _template_script(module) -> dict[str, object]:
    markers = "\n".join(
        (
            module.BASE_IMAGE_AMD64_MANIFEST_DIGEST,
            module.BASE_IMAGE_CONFIG_DIGEST,
            module.BUILD_TASK_ID,
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        )
    )
    return {
        "repository": "",
        "branch": "",
        "version_num": "",
        "working_dir": ".",
        "entry_point": "clearml_5090_bootstrap.py",
        "diff": markers,
    }


def _template_docker(module) -> str:
    return " ".join(
        (
            module.EXPECTED_DOCKER_IMAGE,
            "--network host",
            "--env PYTHONSAFEPATH=1",
        )
    )


def _args(**overrides):
    values = {
        "gate_task_id": GATE_ID,
        "paper_controller_task_id": None,
        "paper_controller_summary_artifact": "paper_controller_summary",
        "template_task_id": TEMPLATE_ID,
        "teacher_task_id": TEACHER_TASK_ID,
        "resolve_teacher_reference": False,
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "allow_failed_teacher_task": True,
        "worker_queue": "GPU4-5090",
        "worker_queues": "",
        "max_parallel": 1,
        "training_seed": 20250218,
        "canary_first": False,
        "adopt_experiment": [],
        "recover_failed_controller_task_id": "",
        "recovery_source_template_task_id": "",
        "recovery_source_transition": "",
        "rerun_failed_experiment": [],
        "recovery_adopt_target_experiment": [],
        "build_task_id": "",
        "native_bundle_bytes": 0,
        "native_bundle_sha256": "",
        "build_manifest_sha256": "",
        "project": "ResilientV2X/Training",
        "poll_seconds": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _source_revision_recovery_case(
    module,
    *,
    adopted_target_experiments: tuple[str, ...] | None = None,
    target_statuses: dict[str, str] | None = None,
    worker_queues: tuple[str, ...] = ("GPU4-5090",),
    max_parallel: int = 1,
):
    tasks = _FakeTasks(module)
    tasks.allow_parallel = max_parallel > 1
    target_statuses = dict(target_statuses or {})
    if adopted_target_experiments is None:
        adopted_target_experiments = module.SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
    if not set(adopted_target_experiments) <= set(
        module.SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
    ):
        raise AssertionError("fixture target adoption is outside transition suffix")

    def patch_fake_clone(task):
        script = dict(task.script)
        script["diff"] = module._apply_experiment_syspath_patch(
            str(script.get("diff") or "")
        )
        task.script = script

    module._ensure_clone_experiment_syspath_fix = patch_fake_clone
    old_parameters = _template_parameters()
    old_parameters.update(module.SOURCE_REVISION_SOURCE_PARAMETERS)
    new_parameters = _template_parameters()
    new_parameters.update(module.SOURCE_REVISION_TARGET_PARAMETERS)
    transition_template_script = _template_script(module)
    transition_template_script["diff"] = (
        f"{transition_template_script['diff']}\n"
        f"{module._EXPERIMENT_NESTED_TEACHER_LEGACY}"
    )
    old_template = tasks._add(
        task_id=module.SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID,
        name="old source bootstrap template",
        status="completed",
        parameters=old_parameters,
        script=transition_template_script,
        docker=_template_docker(module),
    )
    new_template = tasks._add(
        task_id=module.SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID,
        name="new source bootstrap template",
        status="completed",
        parameters=new_parameters,
        script=transition_template_script,
        docker=_template_docker(module),
    )
    teacher_reference = module._resolve_teacher_reference(
        _args(),
        task_class=tasks,
        sleeper=lambda _seconds: None,
    )
    source_controller = tasks._add(
        task_id=module.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
        name="sealed old-source controller",
        status="failed",
        parent=GATE_ID,
    )
    old_identity = module._template_identity(
        old_template,
        expected_task_id=old_template.id,
    )
    old_legacy_identity = module._template_identity(
        old_template,
        expected_task_id=old_template.id,
        include_nested_teacher_fix=False,
    )
    module.SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256 = old_legacy_identity[
        "script_sha256"
    ]
    canonical_old_script = dict(old_template.script)
    canonical_old_script["diff"] = module._apply_experiment_syspath_patch(
        str(old_template.script["diff"])
    )
    canonical_target_script = dict(new_template.script)
    canonical_target_script["diff"] = module._apply_experiment_syspath_patch(
        str(new_template.script["diff"])
    )
    source_progress = module._new_progress(
        controller_task_id=source_controller.id,
        gate_task_id=GATE_ID,
        template_identity=old_identity,
        teacher_reference=teacher_reference,
        worker_queues=list(worker_queues),
        max_parallel_training_tasks=max_parallel,
        training_seed=20250218,
        recovery=None,
    )
    source_steps = {str(step["experiment"]): step for step in source_progress["steps"]}
    predecessor = GATE_ID
    old_task_ids: dict[str, str] = {}
    old_predecessors: dict[str, str] = {}
    legacy_parents = {
        experiment: tasks._add(
            task_id=binding["parent_controller_task_id"],
            name=f"failed legacy parent for {experiment}",
            status="failed",
            parent=GATE_ID,
            script={
                "repository": "",
                "branch": "",
                "version_num": "",
                "working_dir": ".",
                "entry_point": "clearml_5090_training_controller.py",
                "diff": f"sealed legacy parent {experiment}",
            },
        )
        for experiment, binding in (
            module.SOURCE_REVISION_LEGACY_TASK_ADOPTIONS.items()
        )
    }
    for index, experiment in enumerate(
        module.SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS,
        start=1,
    ):
        legacy_binding = module.SOURCE_REVISION_LEGACY_TASK_ADOPTIONS.get(experiment)
        task_id = (
            legacy_binding["task_id"]
            if legacy_binding is not None
            else f"{3000 + index:032x}"
        )
        parent_id = (
            legacy_binding["parent_controller_task_id"]
            if legacy_binding is not None
            else source_controller.id
        )
        parameters = module._experiment_parameters(
            old_parameters,
            experiment=experiment,
            predecessor_task_id=predecessor,
            teacher_task_id=TEACHER_TASK_ID,
            teacher_model_id=TEACHER_MODEL_ID,
            teacher_checkpoint_sha256=TEACHER_SHA,
            allow_failed_teacher_task=True,
            training_seed=20250218,
        )
        task = tasks._add(
            task_id=task_id,
            name=module._task_name(index, experiment, parent_id),
            status="completed",
            parent=parent_id,
            parameters=parameters,
            script=(
                old_template.script
                if legacy_binding is not None
                else canonical_old_script
            ),
            docker=old_template.docker,
        )
        task._publish_experiment_outputs(experiment)
        source_steps[experiment].update(
            {
                "state": "completed",
                "task_id": task_id,
                "task_name": task.name,
                "predecessor_task_id": predecessor,
                "adopted": legacy_binding is not None,
                "worker_queue": (
                    None if experiment == "support_residual" else "GPU4-5090"
                ),
                "result": {"task_id": task_id},
            }
        )
        old_task_ids[experiment] = task_id
        old_predecessors[experiment] = predecessor
        predecessor = task_id

    where2comm_index = module.EXPERIMENT_ORDER.index("where2comm") + 1
    old_where2comm_id = f"{4000:032x}"
    old_where2comm = tasks._add(
        task_id=old_where2comm_id,
        name=module._task_name(
            where2comm_index,
            "where2comm",
            source_controller.id,
        ),
        status="failed",
        parent=source_controller.id,
        parameters=module._experiment_parameters(
            old_parameters,
            experiment="where2comm",
            predecessor_task_id=predecessor,
            teacher_task_id=TEACHER_TASK_ID,
            teacher_model_id=TEACHER_MODEL_ID,
            teacher_checkpoint_sha256=TEACHER_SHA,
            allow_failed_teacher_task=True,
            training_seed=20250218,
        ),
        script=canonical_old_script,
        docker=old_template.docker,
    )
    source_steps["where2comm"].update(
        {
            "state": "failed",
            "task_id": old_where2comm.id,
            "task_name": old_where2comm.name,
            "predecessor_task_id": predecessor,
            "failure_status": "failed",
        }
    )
    no_distillation = tasks.get_task(task_id=old_task_ids["no_distillation"])
    no_distillation_parent = legacy_parents["no_distillation"]
    no_distillation_observation = {
        "task_id": no_distillation.id,
        "task_name": no_distillation.name,
        "task_status": "completed",
        "task_last_update": "2026-08-10T00:00:00+00:00",
        "task_script_sha256": module._script_sha256(
            module._task_script(no_distillation)
        ),
        "parent_controller_task_id": no_distillation_parent.id,
        "parent_controller_status": "failed",
        "parent_controller_last_update": "2026-08-10T00:00:00+00:00",
        "parent_controller_script_sha256": module._script_sha256(
            module._task_script(no_distillation_parent)
        ),
        "source_snapshot_state": "pending",
        "source_snapshot_task_id": None,
        "predecessor_task_id": old_predecessors["no_distillation"],
        "provenance": "explicit_failed_recovery_target",
    }
    support_retry = {
        "task_id": old_task_ids["support_residual"],
        "source_progress_revision": 2,
        "source_snapshot_state": "failed",
        "source_failure_status": "completion_validation_failed",
        "source_failure_message_sha256": (
            module.SOURCE_REVISION_SUPPORT_RESIDUAL_FAILURE_MESSAGE_SHA256
        ),
        "live_task_status": "completed",
        "provenance": "full_completion_contract_revalidation_required",
    }
    source_recovery = {
        "schema_version": 3,
        "mode": "failed_controller_immutable_fork",
        "recovery_target_adoptions": {"no_distillation": no_distillation_observation},
        "adopted_task_ids": {"no_distillation": old_task_ids["no_distillation"]},
        "adopted_predecessor_task_ids": {
            "no_distillation": old_predecessors["no_distillation"]
        },
        "completion_validation_retries": {"support_residual": support_retry},
        "rerun_experiments": [],
        "rerun_source_task_ids": {},
        "rerun_predecessor_task_ids": {},
        "rerun_task_observations": {},
    }
    source_progress["recovery"] = source_recovery
    source_progress["template"]["source_parameters"]["Args/native_bundle_bytes"] = int(
        module.SOURCE_REVISION_SOURCE_PARAMETERS["Args/native_bundle_bytes"]
    )
    module._save_progress(source_controller, source_progress)

    failed_parent_id = "a1" * 16
    failed_parent = tasks._add(
        task_id=failed_parent_id,
        name="failed new-source launch parent",
        status="failed",
        parent=GATE_ID,
        parameters={
            "Args/recover_failed_controller_task_id": source_controller.id,
            "Args/gate_task_id": GATE_ID,
            "Args/template_task_id": new_template.id,
            "Args/recovery_source_template_task_id": old_template.id,
            "Args/recovery_source_transition": (module.SOURCE_REVISION_TRANSITION_ID),
            "Args/training_seed": "20250218",
            "Args/worker_queues": ",".join(worker_queues),
            "Args/max_parallel": str(max_parallel),
            "Args/rerun_failed_experiment": "[]",
            "Args/adopt_experiment": "[]",
        },
        script={
            "repository": "",
            "branch": "",
            "version_num": "",
            "working_dir": ".",
            "entry_point": "clearml_5090_training_controller.py",
            "diff": "sealed source-revision recovery launcher",
        },
    )
    target_task_ids: dict[str, str] = {}
    target_adoptions = []
    for offset, experiment in enumerate(
        adopted_target_experiments,
        start=1,
    ):
        index = module.EXPERIMENT_ORDER.index(experiment) + 1
        task_id = f"{5000 + offset:032x}"
        status = target_statuses.get(experiment, "completed")
        task = tasks._add(
            task_id=task_id,
            name=module._task_name(index, experiment, failed_parent.id),
            status=status,
            parent=failed_parent.id,
            parameters=module._experiment_parameters(
                new_parameters,
                experiment=experiment,
                predecessor_task_id=predecessor,
                teacher_task_id=TEACHER_TASK_ID,
                teacher_model_id=TEACHER_MODEL_ID,
                teacher_checkpoint_sha256=TEACHER_SHA,
                allow_failed_teacher_task=True,
                training_seed=20250218,
            ),
            script=canonical_target_script,
            docker=new_template.docker,
        )
        if status == "completed":
            task._publish_experiment_outputs(experiment)
        elif status in {"queued", "in_progress"}:
            queue = "GPU4-A100" if "GPU4-A100" in worker_queues else worker_queues[0]
            task.data.last_worker = {
                "GPU4-A100": "10.100.34.18-A100:gpu0,1,2,3",
                "GPU4-V100": "10.100.34.26-V100:gpu0,1,2,3",
                "GPU4-5090": "10.100.34.130-5090:gpu4,5,6,7",
            }[queue]
            task.data.execution.queue = queue
        target_task_ids[experiment] = task_id
        target_adoptions.append(f"{experiment}={task_id}")

    target_controller = tasks._add(
        task_id="a2" * 16,
        name="source-revision recovery controller",
        status="in_progress",
        parent=GATE_ID,
    )
    args = _args(
        template_task_id=new_template.id,
        recover_failed_controller_task_id=source_controller.id,
        recovery_source_template_task_id=old_template.id,
        recovery_source_transition=module.SOURCE_REVISION_TRANSITION_ID,
        recovery_adopt_target_experiment=target_adoptions,
        worker_queues=",".join(worker_queues),
        max_parallel=max_parallel,
    )
    return SimpleNamespace(
        tasks=tasks,
        args=args,
        source_controller=source_controller,
        target_controller=target_controller,
        failed_parent=failed_parent,
        old_template=old_template,
        new_template=new_template,
        old_task_ids=old_task_ids,
        old_where2comm_id=old_where2comm_id,
        target_task_ids=target_task_ids,
        predecessor_task_id=predecessor,
        source_recovery=source_recovery,
    )


def test_evidence_priority_order_is_exact_and_fixed() -> None:
    module = _load_controller()
    assert module.CORE_EXPERIMENT_ORDER == (
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
    )
    assert module.EXPERIMENT_ORDER == (
        "support_residual",
    ) + module.CORE_EXPERIMENT_ORDER + (
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
    assert module.PRIMARY_METHOD_EXPERIMENTS == {"resilient_v2x"}
    assert module.NESTED_TEACHER_EXPERIMENTS == {
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
    primary = module.EXPERIMENT_BY_NAME["resilient_v2x"]
    assert primary.kind == "primary_method"
    assert primary.config == "configs/resilient_v2x/dair_resilient_v2x.py"
    assert primary.requires_teacher is True


def test_parameter_override_is_complete_and_removes_student_handoff() -> None:
    module = _load_controller()
    template = _template_parameters()
    teacher = module._experiment_parameters(
        template,
        experiment="ptf_none",
        predecessor_task_id=GATE_ID,
        teacher_task_id=TEACHER_TASK_ID,
        teacher_model_id=TEACHER_MODEL_ID,
        teacher_checkpoint_sha256=TEACHER_SHA,
        allow_failed_teacher_task=True,
    )
    assert teacher["Args/experiment_from_task"] == "ptf_none"
    assert teacher["Args/predecessor_task_id"] == GATE_ID
    assert teacher["Args/gpus"] == 4
    assert teacher["Args/stage"] == "all"
    assert teacher["Args/max_epochs"] == 50
    assert teacher["Args/amp"] is False
    assert teacher["Args/training_seed"] == 20250218
    assert teacher["Args/teacher_task_id"] == TEACHER_TASK_ID
    assert teacher["Args/teacher_model_id"] == TEACHER_MODEL_ID
    assert teacher["Args/teacher_checkpoint_sha256"] == TEACHER_SHA
    assert teacher["Args/allow_failed_teacher_task"] is True
    assert not module.STUDENT_HANDOFF_KEYS & set(teacher)

    primary = module._experiment_parameters(
        template,
        experiment="resilient_v2x",
        predecessor_task_id=GATE_ID,
        teacher_task_id=TEACHER_TASK_ID,
        teacher_model_id=TEACHER_MODEL_ID,
        teacher_checkpoint_sha256=TEACHER_SHA,
        allow_failed_teacher_task=True,
    )
    assert primary["Args/experiment_from_task"] == "resilient_v2x"
    assert primary["Args/teacher_task_id"] == TEACHER_TASK_ID
    assert primary["Args/teacher_model_id"] == TEACHER_MODEL_ID
    assert primary["Args/teacher_checkpoint_sha256"] == TEACHER_SHA

    no_distillation = module._experiment_parameters(
        template,
        experiment="no_distillation",
        predecessor_task_id="a" * 32,
        teacher_task_id=TEACHER_TASK_ID,
        teacher_model_id=TEACHER_MODEL_ID,
        teacher_checkpoint_sha256=TEACHER_SHA,
        allow_failed_teacher_task=True,
    )
    assert no_distillation["Args/teacher_task_id"] == TEACHER_TASK_ID
    assert no_distillation["Args/teacher_model_id"] == TEACHER_MODEL_ID
    assert no_distillation["Args/teacher_checkpoint_sha256"] == TEACHER_SHA
    assert no_distillation["Args/allow_failed_teacher_task"] is True
    assert not module.STUDENT_HANDOFF_KEYS & set(no_distillation)


def test_full_suite_clones_queues_and_validates_strictly_one_at_a_time() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    summary = module.run_training_suite(
        _args(),
        task_class=tasks,
        controller_task=tasks.controller,
        sleeper=lambda _seconds: None,
    )
    assert summary["status"] == "completed"
    assert summary["task_count"] == len(module.EXPERIMENT_ORDER)
    assert [item["experiment"] for item in summary["results"]] == list(
        module.EXPERIMENT_ORDER
    )
    module._require_valid_seal(summary, context="summary")
    manifest = tasks.controller.artifacts[module.FORMAL_1337_MANIFEST_ARTIFACT].get()
    module._require_valid_seal(manifest, context="formal 1337 manifest")
    assert summary["formal_1337_evaluation_manifest"] == manifest
    assert manifest["protocol_id"] == "DAIR-CAUSAL-1337-v1"
    assert manifest["sample_count"] == 1337
    assert manifest["delays_ms"] == [0, 100, 200, 300]
    assert manifest["conditions"] == ["Full", "L-Fail", "C-Fail"]
    assert manifest["run_count"] == 12
    assert manifest["checkpoint_policy"] == "epoch_50_final_only"
    assert manifest["training_seed"] == 20250218
    assert manifest["training_overlay_protocol_seed"] == 20250218
    assert manifest["evaluation_release_semantics"] == (
        "formal_manifest_after_full_training_suite_completion"
    )
    assert manifest["subject_order"] == list(module.EXPERIMENT_ORDER)
    assert [item["subject"] for item in manifest["entries"]] == list(
        module.EXPERIMENT_ORDER
    )
    assert len({item["training_task_id"] for item in manifest["entries"]}) == 26
    assert all(
        item["checkpoint_filename"] == "epoch_50.pth" for item in manifest["entries"]
    )
    progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    module._require_valid_seal(progress, context="progress")
    assert [step["state"] for step in progress["steps"]] == ["completed"] * len(
        module.EXPERIMENT_ORDER
    )
    assert tasks.active == set()

    previous_complete = -1
    for experiment in module.EXPERIMENT_ORDER:
        clone_index = tasks.events.index(f"clone:{experiment}")
        enqueue_index = tasks.events.index(f"enqueue:{experiment}")
        complete_index = tasks.events.index(f"complete:{experiment}")
        assert previous_complete < clone_index < enqueue_index < complete_index
        previous_complete = complete_index
    clones = [task for task in tasks.registry.values() if task.parent == CONTROLLER_ID]
    assert len(clones) == len(module.EXPERIMENT_ORDER)
    predecessor = GATE_ID
    for experiment, task in zip(module.EXPERIMENT_ORDER, clones, strict=True):
        assert task.parameters["Args/predecessor_task_id"] == predecessor
        assert task.parameters["Args/experiment_from_task"] == experiment
        assert task.parameters["Args/training_seed"] == 20250218
        assert not module.STUDENT_HANDOFF_KEYS & set(task.parameters)
        if experiment in module.TEACHER_DEPENDENT_EXPERIMENTS:
            assert task.parameters["Args/teacher_model_id"] == TEACHER_MODEL_ID
        else:
            assert "Args/teacher_task_id" not in task.parameters
        predecessor = task.id


def test_failed_experiment_stops_before_any_later_clone() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.fail_experiment = "support_residual"
    with pytest.raises(RuntimeError, match="ended without completion|failed"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {"support_residual": 1}
    progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    assert progress["steps"][0]["state"] == "failed"
    assert all(step["state"] == "pending" for step in progress["steps"][1:])
    assert module.SUMMARY_ARTIFACT not in tasks.controller.artifacts


def test_enqueue_interruption_recovers_created_clone_without_duplication() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.enqueue_failures = 1
    with pytest.raises(RuntimeError, match="simulated enqueue interruption"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    assert progress["steps"][0]["state"] == "created"
    first_task_id = progress["steps"][0]["task_id"]

    summary = module.run_training_suite(
        _args(),
        task_class=tasks,
        controller_task=tasks.controller,
        sleeper=lambda _seconds: None,
    )
    assert summary["status"] == "completed"
    assert tasks.clone_counts["ptf_none"] == 1
    assert summary["results"][0]["task_id"] == first_task_id


def test_crash_window_clone_is_discovered_by_exact_name_and_not_repeated() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    recovered_id = "a1" * 16
    tasks._add(
        task_id=recovered_id,
        name=module._task_name(2, "ptf_none", CONTROLLER_ID),
        status="created",
        # Simulate a process death immediately after Task.clone(), before the
        # inherited student/validation parameters were replaced.
        parameters=tasks.template.parameters,
        script=tasks.template.script,
        docker=tasks.template.docker,
        parent=CONTROLLER_ID,
    )
    summary = module.run_training_suite(
        _args(),
        task_class=tasks,
        controller_task=tasks.controller,
        sleeper=lambda _seconds: None,
    )
    assert summary["results"][1]["task_id"] == recovered_id
    assert tasks.clone_counts.get("ptf_none", 0) == 0
    assert tasks.events.count("enqueue:ptf_none") == 1


@pytest.mark.parametrize("drift", ["entrypoint", "docker", "source"])
def test_template_identity_drift_fails_before_cloning(drift: str) -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    if drift == "entrypoint":
        tasks.template.script["entry_point"] = "/tmp/wrong.py"
    elif drift == "docker":
        tasks.template.docker += " ".join(
            (
                " --env CLEARML_API_HOST=http://10.100.35.118:8008",
                "--env CLEARML_WEB_HOST=http://10.100.34.118:8080",
                "--env CLEARML_FILES_HOST=http://10.100.34.118:8081",
            )
        )
    else:
        tasks.template.parameters["Args/source_archive_sha256"] = "invalid"
    with pytest.raises((RuntimeError, ValueError)):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {}


def test_clone_identity_drift_is_rejected_before_enqueue() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.drift_clone_experiment = "support_residual"
    with pytest.raises(RuntimeError, match="Docker command drifted"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {"support_residual": 1}
    assert not any(event.startswith("enqueue:") for event in tasks.events)


def test_invalid_output_model_blocks_the_next_experiment() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.invalid_model_experiment = "support_residual"
    with pytest.raises(RuntimeError, match="exactly one final OutputModel"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {"support_residual": 1}
    assert "clone:ptf_none" not in tasks.events
    progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    assert progress["steps"][0]["state"] == "failed"
    assert progress["steps"][0]["failure_status"] == ("completion_validation_failed")


def test_failed_gate_prevents_all_cloning() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module, gate_status="failed")
    with pytest.raises(RuntimeError, match="validation gate.*failed"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {}


def test_gate_can_be_resolved_from_paper_controller_summary() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    paper_id = "f" * 32
    paper = tasks._add(
        task_id=paper_id,
        name="paper controller",
        status="completed",
    )
    paper.artifacts["paper_controller_summary"] = _Artifact(
        {"validation": {"task_id": GATE_ID}}
    )
    args = _args(gate_task_id=None, paper_controller_task_id=paper_id)
    assert (
        module._resolve_gate_task_id(
            args,
            task_class=tasks,
            sleeper=lambda _seconds: None,
        )
        == GATE_ID
    )


def test_controller_source_never_uses_pipeline_controller() -> None:
    source = (
        ROOT / "tools/resilient_v2x/clearml_5090_training_controller.py"
    ).read_text(encoding="utf-8")
    assert "from clearml import PipelineController" not in source
    assert "PipelineController(" not in source


def test_experiment_syspath_patch_allows_a100_v100_smoke_capability() -> None:
    module = _load_controller()
    smoke = (
        "if any(torch.cuda.get_device_capability(i) != (12, 0) for i in range(gpu_count)):\n"
        '    raise RuntimeError("all GPUs must have compute capability 12.0")\n'
    )
    patched = module._apply_experiment_syspath_patch(smoke)
    assert module._EXPERIMENT_SMOKE_CAPABILITY_MARKER in patched
    assert "all GPUs must have compute capability 12.0" not in patched
    assert "(8, 0)" in patched and "(7, 0)" in patched


def test_nested_teacher_patch_upgrades_legacy_template_once_and_is_idempotent() -> None:
    module = _load_controller()
    legacy = f"prefix\n{module._EXPERIMENT_NESTED_TEACHER_LEGACY}suffix\n"
    patched = module._apply_experiment_syspath_patch(legacy)
    assert patched == (f"prefix\n{module._EXPERIMENT_NESTED_TEACHER_PATCH}suffix\n")
    assert module._apply_experiment_syspath_patch(patched) == patched
    assert (
        module._apply_experiment_syspath_patch(
            legacy,
            include_nested_teacher_fix=False,
        )
        == legacy
    )


@pytest.mark.parametrize(
    "invalid",
    (
        lambda module: module._EXPERIMENT_NESTED_TEACHER_LEGACY * 2,
        lambda module: (
            module._EXPERIMENT_NESTED_TEACHER_LEGACY
            + module._EXPERIMENT_NESTED_TEACHER_PATCH
        ),
        lambda module: module._EXPERIMENT_NESTED_TEACHER_PATCH * 2,
    ),
)
def test_nested_teacher_patch_rejects_mixed_or_ambiguous_states(invalid) -> None:
    module = _load_controller()
    with pytest.raises(RuntimeError, match="nested-teacher"):
        module._apply_experiment_syspath_patch(invalid(module))


def _live_teacher_runner_load_excerpt() -> str:
    """Exact excerpts from 487dab26 diff 4dfe2e9d...71485b67."""

    return """    # Overlay helpers import sealed package modules in-process.
    task.set_input_model(
        model_id=str(checkpoint_contract["model_id"]),
        name=f"{baseline}_final_checkpoint",
        update_task_design=False,
        update_task_labels=False,
    )

    runner = _load_source_training_runner(source_root)
    dataset_root, env = _prepare_experiment_environment(
        args,
        task_id=task_id,
        source_root=source_root,
        base_env=runtime_env,
        dataset_class=dataset_class,
        runner=runner,
    )

    if args.predecessor_task_id == task_id:
        raise RuntimeError("an experiment task cannot be its own predecessor")

    runner = _load_source_training_runner(source_root)
    dataset_root, env = _prepare_experiment_environment(
        args,
        task_id=task_id,
        source_root=source_root,
        base_env=runtime_env,
        dataset_class=dataset_class,
        runner=runner,
    )
"""


def test_experiment_runner_load_patch_replays_live_teacher_and_is_idempotent() -> None:
    module = _load_controller()
    live_excerpt = _live_teacher_runner_load_excerpt()
    assert hashlib.sha256(live_excerpt.encode("utf-8")).hexdigest() == (
        "25ec6b51427a612f0026e7926d2756950f19112c989f82d491c9bda2daa65515"
    )
    assert live_excerpt.count(module._EXPERIMENT_SYSPATH_MARKER) == 1
    assert live_excerpt.count(module._EXPERIMENT_RUNNER_LOAD_ANCHOR) == 2
    assert live_excerpt.count(module._EXPERIMENT_RUNNER_LOAD_TARGET_ANCHOR) == 1
    assert live_excerpt.count(module._EXPERIMENT_RUNNER_LOAD_MARKER) == 0

    patched = module._apply_experiment_syspath_patch(live_excerpt)
    assert patched != live_excerpt
    assert hashlib.sha256(patched.encode("utf-8")).hexdigest() == (
        "46e3e13c150688ebfc658ffa4406b6c6d8c52fc00ebf8b073bf0fb0a02e7900e"
    )
    assert patched.count(module._EXPERIMENT_RUNNER_LOAD_ANCHOR) == 1
    assert patched.count(module._EXPERIMENT_RUNNER_LOAD_TARGET_ANCHOR) == 0
    assert patched.count(module._EXPERIMENT_RUNNER_LOAD_TARGET_PATCH) == 1
    assert patched.count(module._EXPERIMENT_RUNNER_LOAD_MARKER) == (
        module._EXPERIMENT_RUNNER_LOAD_PATCH_MARKER_COUNT
    )
    assert module._apply_experiment_syspath_patch(patched) == patched


def test_experiment_runner_load_patch_rejects_mixed_and_ambiguous_states() -> None:
    module = _load_controller()
    live_excerpt = _live_teacher_runner_load_excerpt()
    patched = module._apply_experiment_syspath_patch(live_excerpt)
    invalid_states = (
        live_excerpt + module._EXPERIMENT_RUNNER_LOAD_TARGET_ANCHOR,
        patched + module._EXPERIMENT_RUNNER_LOAD_TARGET_ANCHOR,
        live_excerpt + module._EXPERIMENT_RUNNER_LOAD_MARKER,
        patched + module._EXPERIMENT_RUNNER_LOAD_TARGET_PATCH,
        module._EXPERIMENT_RUNNER_LOAD_ANCHOR * 2,
    )
    for invalid in invalid_states:
        with pytest.raises(RuntimeError, match="runner-load"):
            module._apply_experiment_syspath_patch(invalid)


def test_find_recoverable_clone_skips_failed_tasks() -> None:
    module = _load_controller()

    class Tasks:
        @staticmethod
        def get_tasks(**_kwargs):
            return [
                SimpleNamespace(
                    name="ResilientV2X post-main 03 router_static [ctrl]",
                    parent="ctrl",
                    status="failed",
                )
            ]

    found = module._find_recoverable_clone(
        Tasks,
        project="ResilientV2X/Training",
        name="ResilientV2X post-main 03 router_static [ctrl]",
        controller_task_id="ctrl",
    )
    assert found is None


def test_canary_remains_single_slot_until_completion() -> None:
    module = _load_controller()

    class Task:
        def __init__(self, last_iteration):
            self._last = last_iteration

        def get_last_iteration(self):
            return self._last

    class Tasks:
        @staticmethod
        def get_task(task_id):
            return Task(1 if task_id == "canary" else 0)

    steps = [
        {"state": "completed", "adopted": True, "task_id": "old"},
        {"state": "running", "adopted": False, "task_id": "canary"},
    ]
    assert (
        module._canary_blocks_extra_slots(steps, canary_first=True, task_class=Tasks)
        is True
    )
    assert (
        module._canary_blocks_extra_slots(
            [{"state": "running", "adopted": False, "task_id": "x"}],
            canary_first=True,
            task_class=type(
                "T",
                (),
                {"get_task": staticmethod(lambda task_id: Task(0))},
            ),
        )
        is True
    )


def test_remote_controller_initialization_enables_argparse_connection() -> None:
    module = _load_controller()
    current = object()
    calls: list[dict[str, object]] = []

    class Tasks:
        @staticmethod
        def current_task():
            return None

        @staticmethod
        def init(**kwargs):
            calls.append(dict(kwargs))
            return current

    assert (
        module._current_controller_task(
            Tasks,
            auto_connect_arg_parser=True,
        )
        is current
    )
    assert calls == [
        {
            "project_name": module.DEFAULT_PROJECT,
            "task_name": "ResilientV2X post-main sequential training controller",
            "reuse_last_task_id": False,
            "output_uri": module.FILES_SERVER_URI,
            "auto_connect_arg_parser": True,
        }
    ]


def _configure_auto_teacher(tasks: _FakeTasks, tmp_path: Path) -> tuple[_Model, str]:
    selected_epoch = 20
    filename = (
        f"best_resilient_v2x_car_bev_ap_r40_0.70_teacher_epoch_{selected_epoch}.pth"
    )
    checkpoint = tmp_path / filename
    checkpoint.write_bytes(b"selected-clean-teacher")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_url = f"http://10.100.34.118:8081/models/{filename}"
    model = _Model(
        name=tasks.module.CLEAN_TEACHER_MODEL_NAME,
        task=TEACHER_TASK_ID,
        model_id=TEACHER_MODEL_ID,
        url=model_url,
        local_copy=checkpoint,
    )
    tasks.teacher.status = "completed"
    tasks.teacher.parameters = {
        "Args/stage": "teacher",
        "Args/max_epochs": 50,
        "Args/gpus": 4,
        "Args/amp": False,
    }
    tasks.teacher.models = [model]
    tasks.teacher.artifacts = {
        tasks.module.RUN_CONTRACT_ARTIFACT: _Artifact(
            {
                "stage": "teacher",
                "max_epochs": 50,
                "val_interval": 10,
                "gpus": 4,
                "global_batch_size": 8,
                "amp": False,
                "checkpoint_policy": tasks.module.CLEAN_TEACHER_CHECKPOINT_POLICY,
            }
        ),
        tasks.module.TEACHER_CHECKPOINT_ARTIFACT: _Artifact(
            {
                "schema_version": 1,
                "selection_protocol": tasks.module.CLEAN_TEACHER_SELECTION_PROTOCOL,
                "selection_metric": tasks.module.CLEAN_TEACHER_SELECTION_METRIC,
                "selection_rule": tasks.module.CLEAN_TEACHER_SELECTION_RULE,
                "selected_epoch": selected_epoch,
                "selected_checkpoint": {
                    "model_id": TEACHER_MODEL_ID,
                    "name": tasks.module.CLEAN_TEACHER_MODEL_NAME,
                    "url": model_url,
                    "filename": filename,
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": checkpoint_sha256,
                },
                "trained_epochs": 50,
                "final_epoch": 50,
                "final_checkpoint": {
                    "filename": "teacher_epoch_50.pth",
                    "size_bytes": 123,
                    "sha256": "f" * 64,
                },
                "downstream_role": (
                    "frozen teacher and trainable student initialization"
                ),
            }
        ),
    }
    return model, checkpoint_sha256


@pytest.mark.parametrize(
    "overrides",
    (
        {"teacher_model_id": None},
        {"teacher_checkpoint_sha256": None},
        {"teacher_model_id": None, "teacher_checkpoint_sha256": None},
        {"resolve_teacher_reference": True},
    ),
)
def test_teacher_reference_request_rejects_ambiguous_modes(overrides) -> None:
    module = _load_controller()
    if overrides == {"resolve_teacher_reference": True}:
        overrides = {
            **overrides,
            "teacher_model_id": TEACHER_MODEL_ID,
            "teacher_checkpoint_sha256": TEACHER_SHA,
        }
    with pytest.raises(ValueError):
        module._teacher_reference_request(_args(**overrides))


def test_auto_teacher_reference_is_downloaded_and_persisted(tmp_path: Path) -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    model, checkpoint_sha256 = _configure_auto_teacher(tasks, tmp_path)
    args = _args(
        resolve_teacher_reference=True,
        teacher_model_id=None,
        teacher_checkpoint_sha256=None,
        allow_failed_teacher_task=False,
    )
    summary = module.run_training_suite(
        args,
        task_class=tasks,
        controller_task=tasks.controller,
        sleeper=lambda _seconds: None,
    )
    assert model.download_count == 1
    assert summary["teacher"]["reference_mode"] == "teacher_checkpoint_contract"
    assert summary["teacher"]["checkpoint_sha256"] == checkpoint_sha256
    progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    assert progress["teacher"] == summary["teacher"]
    first_clone = next(
        task for task in tasks.registry.values() if task.parent == CONTROLLER_ID
    )
    assert first_clone.parameters["Args/teacher_checkpoint_sha256"] == checkpoint_sha256


@pytest.mark.parametrize("mode", ("missing", "drift"))
def test_initialization_audit_is_a_mandatory_completion_gate(mode: str) -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    if mode == "missing":
        tasks.omit_audit_experiment = "support_residual"
    else:
        tasks.audit_drift_experiment = "support_residual"
    with pytest.raises(RuntimeError, match="initialization audit"):
        module.run_training_suite(
            _args(),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert tasks.clone_counts == {"support_residual": 1}


def test_experiment_run_contract_seed_schema_is_exact() -> None:
    module = _load_controller()
    assert module._validate_experiment_run_contract_seed(
        {"schema_version": 1, "seed": 20250218},
        expected_seed=20250218,
    ) == {
        "run_contract_schema_version": 1,
        "run_contract_training_seed_field": "seed",
    }


@pytest.mark.parametrize(
    ("contract", "message"),
    (
        ({"schema_version": 1}, "seed field schema mismatch"),
        (
            {
                "schema_version": 1,
                "seed": 20250218,
                "training_seed": 20250218,
            },
            "seed field schema mismatch",
        ),
        (
            {
                "schema_version": 1,
                "seed": 20250218,
                "training_overlay_protocol_seed": 20250218,
            },
            "seed field schema mismatch",
        ),
        ({"schema_version": 1, "seed": 7}, "seed mismatch"),
        ({"schema_version": 1, "seed": "20250218"}, "seed mismatch"),
        ({"schema_version": 2, "seed": 20250218}, "schema version mismatch"),
    ),
)
def test_experiment_run_contract_seed_schema_attacks_fail_closed(
    contract: dict[str, object],
    message: str,
) -> None:
    module = _load_controller()
    with pytest.raises(RuntimeError, match=message):
        module._validate_experiment_run_contract_seed(
            contract,
            expected_seed=20250218,
        )


def test_duplicate_a100_queue_slots_fill_exact_declared_capacity() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.allow_parallel = True
    queues = "GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090"
    summary = module.run_training_suite(
        _args(worker_queues=queues, max_parallel=4),
        task_class=tasks,
        controller_task=tasks.controller,
        sleeper=lambda _seconds: None,
    )
    assert summary["worker_queues"] == queues.split(",")
    assert [queue for _, queue in tasks.enqueue_records[:4]] == queues.split(",")
    assert sum(queue == "GPU4-A100" for _, queue in tasks.enqueue_records[:4]) == 2


def test_canary_first_runs_support_residual_then_fills_four_slots() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.allow_parallel = True
    queues = "GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090"
    module.run_training_suite(
        _args(worker_queues=queues, max_parallel=4, canary_first=True),
        task_class=tasks,
        controller_task=tasks.controller,
        sleeper=lambda _seconds: None,
    )
    assert tasks.enqueue_records[0] == ("support_residual", "GPU4-A100")
    assert tasks.events.index("complete:support_residual") < tasks.events.index(
        "enqueue:ptf_none"
    )
    assert [queue for _, queue in tasks.enqueue_records[1:5]] == queues.split(",")


def test_failed_controller_recovery_forks_progress_adopts_three_and_reruns_router() -> (
    None
):
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.allow_parallel = True
    tasks.hold_experiments = {"support_residual", "ptf_none", "ptf_linear"}
    tasks.fail_experiment = "router_static"
    queues = "GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090"
    source_args = _args(worker_queues=queues, max_parallel=4)

    with pytest.raises(RuntimeError, match="router_static.*failed"):
        module.run_training_suite(
            source_args,
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )

    source_progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    source_progress_snapshot = copy.deepcopy(source_progress)
    source_task_ids = {
        str(step["experiment"]): str(step["task_id"])
        for step in source_progress["steps"][:4]
    }
    assert [step["state"] for step in source_progress["steps"][:4]] == [
        "running",
        "running",
        "running",
        "failed",
    ]
    assert source_progress["recovery"] is None
    tasks.controller.status = "failed"
    tasks.fail_experiment = None

    target_controller_id = "9" * 32
    target_controller = tasks._add(
        task_id=target_controller_id,
        name="recovery controller",
        status="in_progress",
    )
    enqueue_count_before_recovery = len(tasks.enqueue_records)

    def release_adopted_tasks(_seconds: float) -> None:
        tasks.hold_experiments.clear()

    summary = module.run_training_suite(
        _args(
            worker_queues=queues,
            max_parallel=4,
            recover_failed_controller_task_id=CONTROLLER_ID,
            rerun_failed_experiment=["router_static"],
        ),
        task_class=tasks,
        controller_task=target_controller,
        sleeper=release_adopted_tasks,
    )

    assert summary["status"] == "completed"
    recovery = summary["recovery"]
    assert recovery["mode"] == "failed_controller_immutable_fork"
    assert recovery["source_controller_task_id"] == CONTROLLER_ID
    assert (
        recovery["source_progress_seal_sha256"]
        == (source_progress_snapshot["seal_sha256"])
    )
    assert recovery["rerun_experiments"] == ["router_static"]
    assert recovery["rerun_source_task_ids"] == {
        "router_static": source_task_ids["router_static"]
    }
    assert recovery["adopted_task_ids"] == {
        experiment: source_task_ids[experiment]
        for experiment in ("support_residual", "ptf_none", "ptf_linear")
    }
    assert tasks.enqueue_records[enqueue_count_before_recovery] == (
        "router_static",
        "GPU4-5090",
    )

    result_by_experiment = {
        str(item["experiment"]): item for item in summary["results"]
    }
    for experiment in ("support_residual", "ptf_none", "ptf_linear"):
        assert (
            result_by_experiment[experiment]["task_id"] == (source_task_ids[experiment])
        )
    assert (
        result_by_experiment["router_static"]["task_id"]
        != (source_task_ids["router_static"])
    )
    assert tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get() == (
        source_progress_snapshot
    )
    assert module.SUMMARY_ARTIFACT not in tasks.controller.artifacts
    target_progress = target_controller.artifacts[module.PROGRESS_ARTIFACT].get()
    assert target_progress["recovery"] == recovery
    module._require_valid_seal(target_progress, context="recovery progress")


def test_failed_controller_recovery_forces_fresh_snapshot_and_seals_late_terminals() -> (
    None
):
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.allow_parallel = True
    tasks.hold_experiments = {"support_residual", "ptf_none", "ptf_linear"}
    tasks.fail_experiment = "router_static"
    queues = "GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090"

    with pytest.raises(RuntimeError, match="router_static.*failed"):
        module.run_training_suite(
            _args(worker_queues=queues, max_parallel=4),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )

    fresh = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    source_steps = {
        str(step["experiment"]): step
        for step in fresh["steps"]
        if step["experiment"]
        in {"support_residual", "ptf_none", "ptf_linear", "router_static"}
    }
    assert source_steps["router_static"]["state"] == "failed"
    stale = copy.deepcopy(fresh)
    stale_router = next(
        step for step in stale["steps"] if step["experiment"] == "router_static"
    )
    stale_router["state"] = "running"
    stale_router.pop("failure_status", None)
    stale_router.pop("failure_message", None)
    stale["revision"] = int(fresh["revision"]) - 1
    stale["updated_at"] = "2026-08-10T00:00:00+00:00"
    stale = module._sealed(stale)
    source_artifact = _StaleUnlessForcedArtifact(
        cached_payload=stale,
        fresh_payload=fresh,
        url=f"memory://{CONTROLLER_ID}/{module.PROGRESS_ARTIFACT}",
    )
    tasks.controller.artifacts[module.PROGRESS_ARTIFACT] = source_artifact

    source_task_ids = {
        experiment: str(step["task_id"]) for experiment, step in source_steps.items()
    }
    for experiment, terminal_status in {
        "ptf_none": "stopped",
        "ptf_linear": "stopped",
    }.items():
        task = tasks.get_task(task_id=source_task_ids[experiment])
        task.status = terminal_status
        task.data.last_update = "2099-01-01T00:00:00+00:00"
        tasks.active.discard(task.id)
    tasks.controller.status = "failed"
    tasks.fail_experiment = None

    target_controller = tasks._add(
        task_id="9" * 32,
        name="fresh recovery controller",
        status="in_progress",
    )
    with pytest.raises(
        RuntimeError,
        match="ptf_none.*not explicitly included",
    ):
        module.run_training_suite(
            _args(
                worker_queues=queues,
                max_parallel=4,
                recover_failed_controller_task_id=CONTROLLER_ID,
                rerun_failed_experiment=["router_static"],
            ),
            task_class=tasks,
            controller_task=target_controller,
            sleeper=lambda _seconds: None,
        )
    assert module.PROGRESS_ARTIFACT not in target_controller.artifacts

    def release_adopted_task(_seconds: float) -> None:
        tasks.hold_experiments.clear()

    summary = module.run_training_suite(
        _args(
            worker_queues=queues,
            max_parallel=4,
            recover_failed_controller_task_id=CONTROLLER_ID,
            rerun_failed_experiment=[
                "ptf_none",
                "ptf_linear",
                "router_static",
            ],
        ),
        task_class=tasks,
        controller_task=target_controller,
        sleeper=release_adopted_task,
    )

    recovery = summary["recovery"]
    assert recovery["schema_version"] == 4
    assert recovery["source_progress_revision"] == fresh["revision"]
    assert recovery["source_progress_artifact_readback"] == {
        "url": f"memory://{CONTROLLER_ID}/{module.PROGRESS_ARTIFACT}",
        "revision": fresh["revision"],
        "seal_sha256": fresh["seal_sha256"],
        "force_download": True,
        "stable_readbacks": 2,
    }
    observations = recovery["rerun_task_observations"]
    assert observations["router_static"]["snapshot_state"] == "failed"
    assert observations["router_static"]["terminal_status"] == "failed"
    assert observations["router_static"]["provenance"] == "sealed_snapshot"
    for experiment in ("ptf_none", "ptf_linear"):
        observation = observations[experiment]
        assert observation["snapshot_state"] == "running"
        assert observation["snapshot_status"] is None
        assert observation["task_last_update"] == "2099-01-01T00:00:00+00:00"
        assert observation["terminal_status"] == "stopped"
        assert observation["provenance"] == "observed_after_snapshot"
    assert source_artifact.force_download_calls == 4
    assert source_artifact.get() == stale


def test_recovery_adopts_strict_failed_target_task_and_recovers_actual_queues() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.allow_parallel = True
    tasks.hold_experiments = {"support_residual", "ptf_none", "ptf_linear"}
    tasks.fail_experiment = "router_static"
    queues = "GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090"
    rerun = ["ptf_none", "ptf_linear", "router_static"]

    with pytest.raises(RuntimeError, match="router_static.*failed"):
        module.run_training_suite(
            _args(worker_queues=queues, max_parallel=4),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )

    source_progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    source_steps = {str(step["experiment"]): step for step in source_progress["steps"]}
    source_task_ids = {
        experiment: str(source_steps[experiment]["task_id"])
        for experiment in (
            "support_residual",
            "ptf_none",
            "ptf_linear",
            "router_static",
        )
    }
    support = tasks.get_task(task_id=source_task_ids["support_residual"])
    support.status = "completed"
    support._publish_experiment_outputs("support_residual")
    tasks.active.discard(support.id)
    for experiment in ("ptf_none", "ptf_linear"):
        source_task = tasks.get_task(task_id=source_task_ids[experiment])
        source_task.status = "stopped"
        source_task.data.last_update = "2099-01-01T00:00:00+00:00"
        tasks.active.discard(source_task.id)
    tasks.controller.status = "failed"
    tasks.fail_experiment = None

    invalid_target = tasks._add(
        task_id="e" * 32,
        name="invalid recovery target",
        status="in_progress",
    )
    with pytest.raises(RuntimeError, match="source snapshot must be pending"):
        module.run_training_suite(
            _args(
                worker_queues=queues,
                max_parallel=4,
                recover_failed_controller_task_id=CONTROLLER_ID,
                rerun_failed_experiment=rerun,
                recovery_adopt_target_experiment=[f"support_residual={'f' * 32}"],
            ),
            task_class=tasks,
            controller_task=invalid_target,
            sleeper=lambda _seconds: None,
        )
    assert module.PROGRESS_ARTIFACT not in invalid_target.artifacts

    failed_parent_id = "8" * 32
    failed_parent = tasks._add(
        task_id=failed_parent_id,
        name="failed prior recovery target",
        status="failed",
        parent=GATE_ID,
        parameters={
            "Args/recover_failed_controller_task_id": CONTROLLER_ID,
            "Args/gate_task_id": GATE_ID,
            "Args/template_task_id": TEMPLATE_ID,
            "Args/training_seed": "20250218",
            "Args/worker_queues": queues,
            "Args/max_parallel": "4",
            "Args/rerun_failed_experiment": json.dumps(rerun),
            "Args/adopt_experiment": "[]",
        },
        script={
            "repository": "",
            "branch": "",
            "version_num": "",
            "working_dir": ".",
            "entry_point": "clearml_5090_training_controller.py",
            "diff": "verified failed recovery controller",
        },
    )
    assert failed_parent.status == "failed"

    def expected_parameters(experiment: str) -> dict[str, object]:
        return module._experiment_parameters(
            tasks.template.parameters,
            experiment=experiment,
            predecessor_task_id=GATE_ID,
            teacher_task_id=TEACHER_TASK_ID,
            teacher_model_id=TEACHER_MODEL_ID,
            teacher_checkpoint_sha256=TEACHER_SHA,
            allow_failed_teacher_task=True,
            training_seed=20250218,
        )

    cross_task_id = "d" * 32
    cross_task = tasks._add(
        task_id=cross_task_id,
        name=module._task_name(5, "no_distillation", failed_parent_id),
        status="in_progress",
        parent=failed_parent_id,
        parameters=expected_parameters("no_distillation"),
        script=tasks.template.script,
        docker=tasks.template.docker,
    )
    cross_task.data.execution.queue = "GPU4-5090"
    cross_task.data.last_worker = "10.100.34.130-5090:gpu4,5,6,7"
    tasks.active.add(cross_task.id)

    target_controller_id = "9" * 32
    target_controller = tasks._add(
        task_id=target_controller_id,
        name="strict recovery controller",
        status="in_progress",
    )
    existing_specs = {
        "ptf_none": (
            "a" * 32,
            "in_progress",
            "GPU4-A100",
            "10.100.34.18-A100:gpu4,5,6,7",
        ),
        "ptf_linear": (
            "b" * 32,
            "in_progress",
            "GPU4-V100",
            "10.100.34.26-V100:gpu0,1,2,3",
        ),
        "router_static": ("c" * 32, "queued", "GPU4-A100", None),
    }
    existing_target_tasks = {}
    for experiment, (
        task_id,
        status,
        queue,
        last_worker,
    ) in existing_specs.items():
        index = module.EXPERIMENT_ORDER.index(experiment) + 1
        task = tasks._add(
            task_id=task_id,
            name=module._task_name(index, experiment, target_controller_id),
            status=status,
            parent=target_controller_id,
            parameters=expected_parameters(experiment),
            script=tasks.template.script,
            docker=tasks.template.docker,
        )
        task.data.execution.queue = queue
        task.data.last_worker = last_worker
        tasks.active.add(task.id)
        existing_target_tasks[experiment] = task

    tasks.hold_experiments = {
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
    }
    enqueue_count = len(tasks.enqueue_records)
    clone_counts = copy.deepcopy(tasks.clone_counts)

    def release_existing_tasks(_seconds: float) -> None:
        tasks.hold_experiments.clear()

    summary = module.run_training_suite(
        _args(
            worker_queues=queues,
            max_parallel=4,
            recover_failed_controller_task_id=CONTROLLER_ID,
            rerun_failed_experiment=rerun,
            recovery_adopt_target_experiment=[f"no_distillation={cross_task_id}"],
        ),
        task_class=tasks,
        controller_task=target_controller,
        sleeper=release_existing_tasks,
    )

    for experiment in (*rerun, "no_distillation"):
        assert tasks.clone_counts.get(experiment, 0) == clone_counts.get(
            experiment,
            0,
        )
    assert all(
        experiment not in {*rerun, "no_distillation"}
        for experiment, _queue in tasks.enqueue_records[enqueue_count:]
    )
    result_by_experiment = {
        str(result["experiment"]): result for result in summary["results"]
    }
    for experiment, task in existing_target_tasks.items():
        assert result_by_experiment[experiment]["task_id"] == task.id
    assert result_by_experiment["no_distillation"]["task_id"] == cross_task_id

    adoption = summary["recovery"]["recovery_target_adoptions"]["no_distillation"]
    assert adoption["task_id"] == cross_task_id
    assert adoption["parent_controller_task_id"] == failed_parent_id
    assert adoption["source_snapshot_state"] == "pending"
    assert adoption["provenance"] == "explicit_failed_recovery_target"

    progress = target_controller.artifacts[module.PROGRESS_ARTIFACT].get()
    step_by_experiment = {str(step["experiment"]): step for step in progress["steps"]}
    expected_queues = {
        "ptf_none": "GPU4-A100",
        "ptf_linear": "GPU4-V100",
        "router_static": "GPU4-A100",
        "no_distillation": "GPU4-5090",
    }
    for experiment, queue in expected_queues.items():
        step = step_by_experiment[experiment]
        assert step["worker_queue"] == queue
        assert step["queue_observation"]["execution_queue_name"] == queue
        assert step["queue_observation"]["resolved_worker_queue"] == queue

    recovery_snapshot = copy.deepcopy(summary["recovery"])
    clone_counts_after_completion = copy.deepcopy(tasks.clone_counts)
    enqueue_count_after_completion = len(tasks.enqueue_records)
    resumed = module.run_training_suite(
        _args(
            worker_queues=queues,
            max_parallel=4,
            recover_failed_controller_task_id=CONTROLLER_ID,
            rerun_failed_experiment=rerun,
            recovery_adopt_target_experiment=[f"no_distillation={cross_task_id}"],
        ),
        task_class=tasks,
        controller_task=target_controller,
        sleeper=lambda _seconds: pytest.fail("completed recovery should not sleep"),
    )
    assert resumed["recovery"] == recovery_snapshot
    assert tasks.clone_counts == clone_counts_after_completion
    assert len(tasks.enqueue_records) == enqueue_count_after_completion


def test_failed_recovery_forks_without_reruns_and_carries_cross_parent_provenance() -> (
    None
):
    module = _load_controller()
    tasks = _FakeTasks(module)
    tasks.allow_parallel = True
    tasks.hold_experiments = {"support_residual", "ptf_none", "ptf_linear"}
    tasks.fail_experiment = "router_static"
    queue_text = "GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090"
    queues = queue_text.split(",")
    rerun = ["ptf_none", "ptf_linear", "router_static"]

    with pytest.raises(RuntimeError, match="router_static.*failed"):
        module.run_training_suite(
            _args(worker_queues=queue_text, max_parallel=4),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    source_progress = tasks.controller.artifacts[module.PROGRESS_ARTIFACT].get()
    source_steps = {str(step["experiment"]): step for step in source_progress["steps"]}
    source_task_ids = {
        experiment: str(source_steps[experiment]["task_id"])
        for experiment in (
            "support_residual",
            "ptf_none",
            "ptf_linear",
            "router_static",
        )
    }
    support = tasks.get_task(task_id=source_task_ids["support_residual"])
    support.status = "completed"
    support._publish_experiment_outputs("support_residual")
    tasks.active.discard(support.id)
    for experiment in ("ptf_none", "ptf_linear"):
        task = tasks.get_task(task_id=source_task_ids[experiment])
        task.status = "stopped"
        task.data.last_update = "2099-01-01T00:00:00+00:00"
        tasks.active.discard(task.id)
    tasks.controller.status = "failed"
    tasks.fail_experiment = None

    failed_parent_id = "8" * 32
    failed_parent = tasks._add(
        task_id=failed_parent_id,
        name="failed first recovery target",
        status="failed",
        parent=GATE_ID,
        parameters={
            "Args/recover_failed_controller_task_id": CONTROLLER_ID,
            "Args/gate_task_id": GATE_ID,
            "Args/template_task_id": TEMPLATE_ID,
            "Args/training_seed": "20250218",
            "Args/worker_queues": queue_text,
            "Args/max_parallel": "4",
            "Args/rerun_failed_experiment": json.dumps(rerun),
            "Args/adopt_experiment": "[]",
        },
        script={
            "repository": "",
            "branch": "",
            "version_num": "",
            "working_dir": ".",
            "entry_point": "clearml_5090_training_controller.py",
            "diff": "verified failed recovery controller",
        },
    )
    assert failed_parent.status == "failed"

    def expected_parameters(experiment: str) -> dict[str, object]:
        return module._experiment_parameters(
            tasks.template.parameters,
            experiment=experiment,
            predecessor_task_id=GATE_ID,
            teacher_task_id=TEACHER_TASK_ID,
            teacher_model_id=TEACHER_MODEL_ID,
            teacher_checkpoint_sha256=TEACHER_SHA,
            allow_failed_teacher_task=True,
            training_seed=20250218,
        )

    cross_task_id = "d" * 32
    cross_task = tasks._add(
        task_id=cross_task_id,
        name=module._task_name(5, "no_distillation", failed_parent_id),
        status="in_progress",
        parent=failed_parent_id,
        parameters=expected_parameters("no_distillation"),
        script=tasks.template.script,
        docker=tasks.template.docker,
    )
    cross_task.data.execution.queue = "GPU4-5090"
    cross_task.data.last_worker = "10.100.34.130-5090:gpu4,5,6,7"
    tasks.active.add(cross_task.id)

    rec2_id = "9" * 32
    rec2 = tasks._add(
        task_id=rec2_id,
        name="failed second recovery controller",
        status="failed",
        parent=GATE_ID,
    )
    template_parameters = dict(tasks.template.parameters)
    target_template_identity = dict(
        module._template_identity(
            tasks.template,
            expected_task_id=TEMPLATE_ID,
        )
    )
    source_template_identity = dict(
        module._template_identity(
            tasks.template,
            expected_task_id=TEMPLATE_ID,
            include_nested_teacher_fix=False,
        )
    )
    for identity in (target_template_identity, source_template_identity):
        identity["source_parameters"] = module._validate_source_parameters(
            template_parameters
        )
        identity["native_build_task_id"] = module.BUILD_TASK_ID

    first_recovery, first_adopted, first_predecessors = (
        module._resolve_failed_controller_recovery(
            task_class=tasks,
            controller_task_id=rec2_id,
            source_controller_task_id=CONTROLLER_ID,
            rerun_experiments=rerun,
            recovery_target_adoptions={"no_distillation": cross_task_id},
            gate_task_id=GATE_ID,
            source_template_identity=source_template_identity,
            source_legacy_template_identity=None,
            target_template_identity=target_template_identity,
            source_template_parameters=template_parameters,
            target_template_parameters=template_parameters,
            source_revision_transition=None,
            teacher_reference=source_progress["teacher"],
            worker_queues=queues,
            max_parallel_training_tasks=4,
            training_seed=20250218,
            project="ResilientV2X/Training",
            sealed_recovery=None,
        )
    )
    assert first_adopted == {
        "support_residual": support.id,
        "no_distillation": cross_task_id,
    }
    assert set(first_predecessors) == {
        "support_residual",
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
    }

    # Mirror the schema-v2 recovery contract already sealed by the real rec2.
    sealed_rec2_recovery = copy.deepcopy(first_recovery)
    sealed_rec2_recovery["schema_version"] = 2
    sealed_rec2_recovery.pop("source_recovery_chain")
    sealed_rec2_recovery.pop("completion_validation_retries")
    sealed_rec2_recovery.pop("recovered_pending_target_children")

    rec2_children: dict[str, _FakeTask] = {}
    child_specs = {
        "ptf_none": ("a" * 32, "in_progress", "GPU4-A100"),
        "ptf_linear": ("b" * 32, "in_progress", "GPU4-V100"),
        "router_static": ("c" * 32, "queued", "GPU4-A100"),
    }
    for experiment, (task_id, status, queue) in child_specs.items():
        task = tasks._add(
            task_id=task_id,
            name=module._task_name(
                module.EXPERIMENT_ORDER.index(experiment) + 1,
                experiment,
                rec2_id,
            ),
            status=status,
            parent=rec2_id,
            parameters=expected_parameters(experiment),
            script=tasks.template.script,
            docker=tasks.template.docker,
        )
        task.data.execution.queue = queue
        if status == "in_progress":
            task.data.last_worker = {
                "GPU4-A100": "10.100.34.18-A100:gpu4,5,6,7",
                "GPU4-V100": "10.100.34.26-V100:gpu0,1,2,3",
            }[queue]
        tasks.active.add(task.id)
        rec2_children[experiment] = task

    rec2_progress = module._new_progress(
        controller_task_id=rec2_id,
        gate_task_id=GATE_ID,
        template_identity=target_template_identity,
        teacher_reference=source_progress["teacher"],
        worker_queues=queues,
        max_parallel_training_tasks=4,
        training_seed=20250218,
        recovery=sealed_rec2_recovery,
    )
    rec2_step_by_experiment = {
        str(step["experiment"]): step for step in rec2_progress["steps"]
    }
    support_step = rec2_step_by_experiment["support_residual"]
    support_step.update(
        {
            "state": "failed",
            "task_id": support.id,
            "task_name": support.name,
            "predecessor_task_id": GATE_ID,
            "adopted": True,
            "worker_queue": "GPU4-A100",
            "failure_status": "completion_validation_failed",
            "failure_message": (
                "experiment run contract training_seed mismatch: "
                "expected 20250218, got None"
            ),
        }
    )
    assert all(
        rec2_step_by_experiment[experiment]["state"] == "pending"
        and rec2_step_by_experiment[experiment]["task_id"] is None
        for experiment in (
            "ptf_none",
            "ptf_linear",
            "router_static",
            "no_distillation",
        )
    )
    module._save_progress(rec2, rec2_progress)
    module._validate_progress(
        rec2_progress,
        controller_task_id=rec2_id,
        gate_task_id=GATE_ID,
        template_identity=target_template_identity,
        teacher_reference=source_progress["teacher"],
        worker_queues=queues,
        max_parallel_training_tasks=4,
        training_seed=20250218,
        recovery=sealed_rec2_recovery,
    )
    rec2_artifact = rec2.artifacts[module.PROGRESS_ARTIFACT]

    rec3_id = "f" * 32
    rec3 = tasks._add(
        task_id=rec3_id,
        name="third recovery controller",
        status="in_progress",
        parent=GATE_ID,
    )
    tasks.hold_experiments = {
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
    }
    clone_counts = copy.deepcopy(tasks.clone_counts)
    enqueue_count = len(tasks.enqueue_records)

    def release_live_tasks(_seconds: float) -> None:
        tasks.hold_experiments.clear()

    summary = module.run_training_suite(
        _args(
            worker_queues=queue_text,
            max_parallel=4,
            recover_failed_controller_task_id=rec2_id,
            rerun_failed_experiment=[],
            recovery_adopt_target_experiment=[],
        ),
        task_class=tasks,
        controller_task=rec3,
        sleeper=release_live_tasks,
    )

    assert rec2_artifact.force_download_calls == 2
    for experiment in (
        "support_residual",
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
    ):
        assert tasks.clone_counts.get(experiment, 0) == clone_counts.get(
            experiment,
            0,
        )
        assert all(
            enqueued_experiment != experiment
            for enqueued_experiment, _queue in tasks.enqueue_records[enqueue_count:]
        )

    recovery = summary["recovery"]
    assert recovery["schema_version"] == 4
    assert recovery["rerun_experiments"] == []
    assert recovery["rerun_source_task_ids"] == {}
    assert recovery["source_recovery_chain"]["source_controller_task_id"] == rec2_id
    assert recovery["source_recovery_chain"]["source_recovery_schema_version"] == 2
    assert recovery["source_recovery_chain"]["carried_recovery_target_experiments"] == [
        "no_distillation"
    ]
    assert recovery["source_recovery_chain"][
        "recovered_pending_target_experiments"
    ] == ["ptf_none", "ptf_linear", "router_static"]
    assert (
        recovery["completion_validation_retries"]["support_residual"]["provenance"]
        == "full_completion_contract_revalidation_required"
    )
    carried = recovery["recovery_target_adoptions"]["no_distillation"]
    assert carried["task_id"] == cross_task_id
    assert carried["parent_controller_task_id"] == failed_parent_id
    assert carried["provenance"] == "explicit_failed_recovery_target"
    assert {
        experiment: observation["task_id"]
        for experiment, observation in recovery[
            "recovered_pending_target_children"
        ].items()
    } == {experiment: task.id for experiment, task in rec2_children.items()}
    assert all(
        observation["provenance"] == "canonical_child_of_failed_recovery_controller"
        for observation in recovery["recovered_pending_target_children"].values()
    )
    result_by_experiment = {
        str(result["experiment"]): result for result in summary["results"]
    }
    assert (
        result_by_experiment["support_residual"]["run_contract_training_seed_field"]
        == "seed"
    )
    assert result_by_experiment["support_residual"]["training_seed"] == 20250218

    recovery_snapshot = copy.deepcopy(recovery)
    clone_counts_after_completion = copy.deepcopy(tasks.clone_counts)
    enqueue_count_after_completion = len(tasks.enqueue_records)
    for task in (*rec2_children.values(), cross_task):
        task.data.last_update = "2099-01-01T00:00:00+00:00"
    resumed = module.run_training_suite(
        _args(
            worker_queues=queue_text,
            max_parallel=4,
            recover_failed_controller_task_id=rec2_id,
            rerun_failed_experiment=[],
            recovery_adopt_target_experiment=[],
        ),
        task_class=tasks,
        controller_task=rec3,
        sleeper=lambda _seconds: pytest.fail(
            "completed chained recovery should not sleep"
        ),
    )
    assert resumed["recovery"] == recovery_snapshot
    assert tasks.clone_counts == clone_counts_after_completion
    assert len(tasks.enqueue_records) == enqueue_count_after_completion


@pytest.mark.parametrize("candidate_count", (0, 2))
def test_exact_recovery_child_discovery_fails_closed(candidate_count: int) -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    parent_id = "8" * 32
    name = module._task_name(2, "ptf_none", parent_id)
    for offset in range(candidate_count):
        tasks._add(
            task_id=f"{900 + offset:032x}",
            name=name,
            status="in_progress",
            parent=parent_id,
        )
    with pytest.raises(
        RuntimeError,
        match=f"expected exactly one recovery child.*found {candidate_count}",
    ):
        module._find_exact_recovery_child(
            tasks,
            project="ResilientV2X/Training",
            name=name,
            parent_controller_task_id=parent_id,
        )


def test_failed_controller_recovery_rejects_manual_adoption() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    with pytest.raises(ValueError, match="manual --adopt-experiment is forbidden"):
        module.run_training_suite(
            _args(
                recover_failed_controller_task_id="9" * 32,
                rerun_failed_experiment=["router_static"],
                adopt_experiment=[f"support_residual={'8' * 32}"],
            ),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )


def test_source_revision_transition_adopts_exact_21_old_and_5_new() -> None:
    module = _load_controller()
    case = _source_revision_recovery_case(module)
    source_snapshot = copy.deepcopy(
        case.source_controller.artifacts[module.PROGRESS_ARTIFACT].get()
    )

    summary = module.run_training_suite(
        case.args,
        task_class=case.tasks,
        controller_task=case.target_controller,
        sleeper=lambda _seconds: pytest.fail("completed transition should not sleep"),
    )

    assert summary["status"] == "completed"
    recovery = summary["recovery"]
    assert recovery["schema_version"] == 4
    transition = recovery["source_revision_transition"]
    module._require_valid_seal(transition, context="test source transition")
    assert transition["transition_id"] == module.SOURCE_REVISION_TRANSITION_ID
    assert transition["source_revision"] == {
        "parameters": {
            key: str(value)
            for key, value in module.SOURCE_REVISION_SOURCE_PARAMETERS.items()
        },
        "tree_sha256": module.SOURCE_REVISION_SOURCE_TREE_SHA256,
        "inventory_sha256": module.SOURCE_REVISION_SOURCE_INVENTORY_SHA256,
        "inventory_bytes": 101_195,
        "inventory_file_count": 631,
        "source_bytes": 8_926_106,
    }
    assert transition["target_revision"]["inventory_sha256"] == (
        module.SOURCE_REVISION_TARGET_INVENTORY_SHA256
    )
    assert transition["target_revision"]["source_bytes"] == 8_926_102
    equivalence = transition["inventory_equivalence"]
    assert equivalence["unchanged_file_count"] == 627
    assert equivalence["changed_file_count"] == 4
    assert equivalence["added_paths"] == []
    assert equivalence["removed_paths"] == []
    assert {item["path"] for item in equivalence["changed_files"]} == {
        item[0] for item in module.SOURCE_REVISION_ALLOWED_FILE_CHANGES
    }

    roles = recovery["adopted_task_template_roles"]
    assert {
        experiment for experiment, role in roles.items() if role == "source"
    } == set(module.SOURCE_REVISION_SOURCE_ADOPTABLE_EXPERIMENTS)
    assert {
        experiment for experiment, role in roles.items() if role == "target"
    } == set(module.SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS)
    assert (
        recovery["transition_replaced_source_tasks"]["where2comm"]["task_id"]
        == case.old_where2comm_id
    )
    assert (
        recovery["recovery_target_adoptions"]["where2comm"]["source_snapshot_task_id"]
        == case.old_where2comm_id
    )
    template_equivalence = recovery["source_progress_template_equivalence"]
    module._require_valid_seal(
        template_equivalence,
        context="test source-progress template equivalence",
    )
    assert template_equivalence["normalized_type_only_keys"] == [
        "Args/native_bundle_bytes"
    ]
    assert (
        template_equivalence["observed_identity"]["source_parameters"][
            "Args/native_bundle_bytes"
        ]
        == 753_382_966
    )
    assert (
        template_equivalence["expected_identity"]["source_parameters"][
            "Args/native_bundle_bytes"
        ]
        == "753382966"
    )
    frozen_predecessors = recovery["target_template_predecessor_task_ids"]
    assert frozen_predecessors == {
        experiment: case.predecessor_task_id
        for experiment in module.SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
    }
    replaced_binding = recovery["transition_replaced_source_tasks"]["where2comm"][
        "source_task_binding_receipt"
    ]
    module._require_valid_seal(
        replaced_binding,
        context="test replaced source task binding",
    )
    assert replaced_binding["exact_execution_parameter_match"] is True
    receipts = recovery["adopted_task_binding_receipts"]
    assert set(receipts) == set(module.EXPERIMENT_ORDER)
    for experiment, provenance_type in (
        ("support_residual", "completion_validation_retry"),
        ("no_distillation", "carried_recovery_target"),
    ):
        receipt = receipts[experiment]
        assert receipt["script_identity_policy"] == (
            "exact_allowlisted_legacy_nested_teacher_semantics_preserving"
        )
        compatibility = receipt["legacy_script_compatibility_receipt"]
        module._require_valid_seal(
            compatibility,
            context=f"test {experiment} legacy compatibility",
        )
        assert compatibility["sealed_provenance_type"] == provenance_type
        assert compatibility["subject_semantics_equal"] is True
        assert compatibility["exception_scope"] == "script_identity_only"
    assert receipts["v2vnet"]["script_identity_policy"] == ("exact_canonical_template")
    assert receipts["v2vnet"]["legacy_script_compatibility_receipt"] is None
    assert (
        receipts["v2vnet"]["source_parameters"]["Args/source_dataset_id"]
        == module.SOURCE_REVISION_SOURCE_PARAMETERS["Args/source_dataset_id"]
    )
    assert (
        receipts["where2comm"]["source_parameters"]["Args/source_dataset_id"]
        == module.SOURCE_REVISION_TARGET_PARAMETERS["Args/source_dataset_id"]
    )
    assert all(
        case.tasks.get_task(task_id=task_id).parameters["Args/source_dataset_id"]
        == module.SOURCE_REVISION_SOURCE_PARAMETERS["Args/source_dataset_id"]
        for task_id in case.old_task_ids.values()
    )
    assert all(
        case.tasks.get_task(task_id=task_id).parameters["Args/source_dataset_id"]
        == module.SOURCE_REVISION_TARGET_PARAMETERS["Args/source_dataset_id"]
        for task_id in case.target_task_ids.values()
    )
    assert (
        case.source_controller.artifacts[module.PROGRESS_ARTIFACT].get()
        == source_snapshot
    )


def test_source_revision_transition_runs_live_four_adopted_one_pending_shape() -> None:
    module = _load_controller()
    queues = ("GPU4-A100", "GPU4-A100", "GPU4-V100", "GPU4-5090")
    adopted_targets = module.SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS[:4]
    case = _source_revision_recovery_case(
        module,
        adopted_target_experiments=adopted_targets,
        target_statuses={"how2comm": "in_progress"},
        worker_queues=queues,
        max_parallel=4,
    )
    case.tasks.hold_experiments.add("how2comm")
    sleeper_calls = 0

    def release_how2comm(_seconds: float) -> None:
        nonlocal sleeper_calls
        sleeper_calls += 1
        case.tasks.hold_experiments.discard("how2comm")

    summary = module.run_training_suite(
        case.args,
        task_class=case.tasks,
        controller_task=case.target_controller,
        sleeper=release_how2comm,
    )

    assert summary["status"] == "completed"
    assert sleeper_calls == 1
    recovery = summary["recovery"]
    assert (
        set(recovery["recovery_target_adoptions"])
        & set(module.SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS)
    ) == set(adopted_targets)
    assert recovery["target_template_predecessor_task_ids"] == {
        experiment: case.predecessor_task_id
        for experiment in module.SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
    }
    target_tasks = {
        str(task.parameters.get("Args/experiment_from_task")): task
        for task in case.tasks.registry.values()
        if task.parameters.get("Args/experiment_from_task")
        in module.SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS
        and task.id != case.old_where2comm_id
    }
    assert set(target_tasks) == set(module.SOURCE_REVISION_TARGET_REQUIRED_EXPERIMENTS)
    assert all(
        task.parameters["Args/predecessor_task_id"] == case.predecessor_task_id
        for task in target_tasks.values()
    )
    assert all(
        task.parameters["Args/source_dataset_id"]
        == module.SOURCE_REVISION_TARGET_PARAMETERS["Args/source_dataset_id"]
        for task in target_tasks.values()
    )
    assert case.tasks.clone_counts == {"resilient_v2x": 1}
    assert case.tasks.enqueue_records == [("resilient_v2x", "GPU4-A100")]


def test_source_revision_transition_rejects_historical_parameter_content_drift() -> (
    None
):
    module = _load_controller()
    case = _source_revision_recovery_case(module)
    progress = case.source_controller.artifacts[module.PROGRESS_ARTIFACT].get()
    progress["template"]["source_parameters"]["Args/native_bundle_bytes"] = 753_382_965
    case.source_controller.artifacts[module.PROGRESS_ARTIFACT] = _Artifact(
        module._sealed(progress),
        url=f"memory://{case.source_controller.id}/{module.PROGRESS_ARTIFACT}",
    )

    with pytest.raises(
        RuntimeError,
        match="controller template identity source parameter drifted",
    ):
        module.run_training_suite(
            case.args,
            task_class=case.tasks,
            controller_task=case.target_controller,
            sleeper=lambda _seconds: None,
        )
    assert module.PROGRESS_ARTIFACT not in case.target_controller.artifacts


def test_source_revision_transition_rejects_legacy_script_outside_allowlist() -> None:
    module = _load_controller()
    case = _source_revision_recovery_case(module)
    v2vnet = case.tasks.get_task(task_id=case.old_task_ids["v2vnet"])
    v2vnet.script = dict(case.old_template.script)

    with pytest.raises(RuntimeError, match="cloned task script drifted"):
        module.run_training_suite(
            case.args,
            task_class=case.tasks,
            controller_task=case.target_controller,
            sleeper=lambda _seconds: None,
        )
    assert module.PROGRESS_ARTIFACT not in case.target_controller.artifacts


def test_source_revision_transition_rejects_replaced_old_task_binding_drift() -> None:
    module = _load_controller()
    case = _source_revision_recovery_case(module)
    replaced = case.tasks.get_task(task_id=case.old_where2comm_id)
    replaced.parameters["Args/source_dataset_id"] = (
        module.SOURCE_REVISION_TARGET_PARAMETERS["Args/source_dataset_id"]
    )

    with pytest.raises(RuntimeError, match="cloned task sealed parameter drifted"):
        module.run_training_suite(
            case.args,
            task_class=case.tasks,
            controller_task=case.target_controller,
            sleeper=lambda _seconds: None,
        )
    assert module.PROGRESS_ARTIFACT not in case.target_controller.artifacts


@pytest.mark.parametrize(
    ("template_role", "key", "value", "message"),
    (
        (
            "source",
            "Args/source_archive_sha256",
            "0" * 64,
            "source template source parameter drifted",
        ),
        (
            "target",
            "Args/source_dataset_id",
            "0" * 32,
            "target template source parameter drifted",
        ),
        (
            "target",
            "General/retained_template_parameter",
            "drifted",
            "non-source template parameters drifted",
        ),
    ),
)
def test_source_revision_transition_rejects_template_parameter_drift(
    template_role: str,
    key: str,
    value: str,
    message: str,
) -> None:
    module = _load_controller()
    case = _source_revision_recovery_case(module)
    template = case.old_template if template_role == "source" else case.new_template
    template.parameters[key] = value

    with pytest.raises(RuntimeError, match=message):
        module.run_training_suite(
            case.args,
            task_class=case.tasks,
            controller_task=case.target_controller,
            sleeper=lambda _seconds: None,
        )
    assert module.PROGRESS_ARTIFACT not in case.target_controller.artifacts


@pytest.mark.parametrize("template_role", ("source", "target"))
def test_source_revision_transition_rejects_adopted_task_source_drift(
    template_role: str,
) -> None:
    module = _load_controller()
    case = _source_revision_recovery_case(module)
    if template_role == "source":
        experiment = "v2vnet"
        task_id = case.old_task_ids[experiment]
        drifted_dataset_id = module.SOURCE_REVISION_TARGET_PARAMETERS[
            "Args/source_dataset_id"
        ]
    else:
        experiment = "where2comm"
        task_id = case.target_task_ids[experiment]
        drifted_dataset_id = module.SOURCE_REVISION_SOURCE_PARAMETERS[
            "Args/source_dataset_id"
        ]
    case.tasks.get_task(task_id=task_id).parameters["Args/source_dataset_id"] = (
        drifted_dataset_id
    )

    with pytest.raises(RuntimeError, match="cloned task sealed parameter drifted"):
        module.run_training_suite(
            case.args,
            task_class=case.tasks,
            controller_task=case.target_controller,
            sleeper=lambda _seconds: None,
        )
    assert module.PROGRESS_ARTIFACT not in case.target_controller.artifacts


def test_source_revision_transition_rejects_sealed_evidence_tamper_on_resume() -> None:
    module = _load_controller()
    case = _source_revision_recovery_case(module)
    module.run_training_suite(
        case.args,
        task_class=case.tasks,
        controller_task=case.target_controller,
        sleeper=lambda _seconds: None,
    )
    progress = case.target_controller.artifacts[module.PROGRESS_ARTIFACT].get()
    recovery = copy.deepcopy(progress["recovery"])
    transition = copy.deepcopy(recovery["source_revision_transition"])
    transition["target_revision"]["source_bytes"] = 8_926_103
    recovery["source_revision_transition"] = module._sealed(transition)
    progress["recovery"] = recovery
    tampered_progress = module._sealed(progress)
    case.target_controller.artifacts[module.PROGRESS_ARTIFACT] = _Artifact(
        tampered_progress,
        url=f"memory://{case.target_controller.id}/{module.PROGRESS_ARTIFACT}",
    )

    with pytest.raises(RuntimeError, match="sealed target recovery contract drifted"):
        module.run_training_suite(
            case.args,
            task_class=case.tasks,
            controller_task=case.target_controller,
            sleeper=lambda _seconds: None,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {
            "recover_failed_controller_task_id": ("f8c36e508c7d453dadc766207a5b25b2"),
            "recovery_source_template_task_id": ("487dab2664a8485fa0cc7c4e2a0c3df8"),
        },
        {
            "recover_failed_controller_task_id": ("f8c36e508c7d453dadc766207a5b25b2"),
            "recovery_source_transition": (
                "resilient-v2x-source-5c984ad49b52-to-ad511d88b731-"
                "custom-imports-list-v1"
            ),
        },
    ),
)
def test_source_revision_transition_requires_both_explicit_pins(
    overrides: dict[str, object],
) -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    with pytest.raises(ValueError, match="must be provided together"):
        module.run_training_suite(
            _args(**overrides),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert module.PROGRESS_ARTIFACT not in tasks.controller.artifacts


def test_source_revision_transition_pins_require_recovery_mode() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    with pytest.raises(ValueError, match="requires.*recover-failed-controller"):
        module.run_training_suite(
            _args(
                recovery_source_template_task_id=(
                    module.SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID
                ),
                recovery_source_transition=module.SOURCE_REVISION_TRANSITION_ID,
            ),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
    assert module.PROGRESS_ARTIFACT not in tasks.controller.artifacts


def test_recovery_target_adoption_requires_recovery_mode() -> None:
    module = _load_controller()
    tasks = _FakeTasks(module)
    with pytest.raises(
        ValueError,
        match="recovery-adopt-target-experiment requires",
    ):
        module.run_training_suite(
            _args(
                recovery_adopt_target_experiment=[f"no_distillation={'8' * 32}"],
            ),
            task_class=tasks,
            controller_task=tasks.controller,
            sleeper=lambda _seconds: None,
        )
