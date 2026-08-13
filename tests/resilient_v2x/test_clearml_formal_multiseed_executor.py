from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import runpy
import shutil
import stat
import sys
import zipfile
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_formal_multiseed_executor.py"
DETECTION_PATH = ROOT / "transvision/evaluation/resilient_v2x_detection.py"
PLAN_PATH = ROOT / "tools/resilient_v2x/formal_multiseed_plan.py"
PLANNER_PRODUCER_PATH = ROOT / "tools/resilient_v2x/clearml_formal_multiseed_plan.py"
PROVENANCE_PRODUCER_PATH = (
    ROOT / "tools/resilient_v2x/clearml_formal_training_provenance.py"
)
SOURCE_D_PRODUCER_PATH = (
    ROOT / "tools/resilient_v2x/clearml_formal_source_d_evidence.py"
)
SOURCE_D_BUILDER_PATH = ROOT / "tools/resilient_v2x/formal_source_d_seed.py"
EXECUTOR_ID = "e" * 32
PARENT_ID = "d" * 32


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "clearml_formal_multiseed_executor_fixture", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ClearMLArtifact = lambda raw: raw.artifact
    return module


def _load_detection_module():
    name = "resilient_v2x_detection_contract_fixture"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, DETECTION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_source_d_producer_module():
    name = "clearml_formal_source_d_evidence_executor_fixture"
    spec = importlib.util.spec_from_file_location(name, SOURCE_D_PRODUCER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_revision_binding() -> dict[str, object]:
    name = "clearml_formal_training_provenance_executor_fixture"
    spec = importlib.util.spec_from_file_location(name, PROVENANCE_PRODUCER_PATH)
    assert spec is not None and spec.loader is not None
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    equivalence = producer._source_revision_equivalence()
    subject_map = producer._source_revision_subject_map()
    evaluation_tree = producer.SOURCE_TREE_SHA256
    return {
        "source_revision_equivalence": equivalence,
        "source_revision_equivalence_seal_sha256": equivalence["seal_sha256"],
        "source_revision_subject_map": subject_map,
        "source_revision_subject_map_seal_sha256": subject_map["seal_sha256"],
        "evaluation_source_revision_tree_sha256": evaluation_tree,
        "evaluation_source_revision": equivalence["source_revisions"][evaluation_tree],
    }


REAL_EXPERIMENT_CONFIGS = (
    (
        "support_residual",
        "configs/resilient_v2x/improvements/support_residual.py",
        True,
    ),
    ("ptf_linear", "configs/resilient_v2x/ablations/ptf_linear.py", True),
    ("ptf_none", "configs/resilient_v2x/ablations/ptf_none.py", True),
    ("router_static", "configs/resilient_v2x/ablations/router_static.py", True),
    ("router_uniform", "configs/resilient_v2x/ablations/router_uniform.py", True),
    ("no_reliability", "configs/resilient_v2x/ablations/no_reliability.py", True),
    (
        "no_delay_metadata",
        "configs/resilient_v2x/ablations/no_delay_metadata.py",
        True,
    ),
    (
        "no_distillation",
        "configs/resilient_v2x/ablations/no_distillation.py",
        False,
    ),
    (
        "concat_capacity_matched",
        "configs/resilient_v2x/ablations/concat_capacity_matched.py",
        True,
    ),
    ("v2x_vit", "configs/resilient_v2x/baselines/v2x_vit.py", False),
    ("cobevt", "configs/resilient_v2x/baselines/cobevt.py", False),
    ("coformernet", "configs/resilient_v2x/baselines/coformernet.py", False),
    ("bevfusion", "configs/resilient_v2x/baselines/bevfusion.py", False),
    ("ffnet", "configs/resilient_v2x/baselines/ffnet.py", False),
    (
        "linear_no_distillation",
        "configs/resilient_v2x/improvements/linear_no_distillation.py",
        False,
    ),
    (
        "no_distillation_peak_lr_3e4",
        "configs/resilient_v2x/improvements/no_distillation_peak_lr_3e4.py",
        False,
    ),
    ("ego_only", "configs/resilient_v2x/baselines/ego_only.py", False),
    ("fcooper", "configs/resilient_v2x/baselines/fcooper.py", False),
    ("attfuse", "configs/resilient_v2x/baselines/attfuse.py", False),
    ("v2vnet", "configs/resilient_v2x/baselines/v2vnet.py", False),
    ("when2com", "configs/resilient_v2x/baselines/when2com.py", False),
    ("where2comm", "configs/resilient_v2x/baselines/where2comm.py", False),
    ("late_fusion", "configs/resilient_v2x/baselines/late_fusion.py", False),
    ("disconet", "configs/resilient_v2x/baselines/disconet.py", False),
    ("how2comm", "configs/resilient_v2x/baselines/how2comm.py", False),
    (
        "resilient_v2x",
        "configs/resilient_v2x/dair_resilient_v2x.py",
        True,
    ),
)


def _merge_real_config(base: object, update: object) -> object:
    if not isinstance(base, dict) or not isinstance(update, dict):
        return copy.deepcopy(update)
    if update.get("_delete_") is True:
        return {
            key: copy.deepcopy(value)
            for key, value in update.items()
            if key != "_delete_"
        }
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if key != "_delete_":
            merged[key] = _merge_real_config(merged.get(key), value)
    return merged


def _load_real_config(
    path: Path,
    stack: tuple[Path, ...] = (),
) -> dict[str, object]:
    path = path.resolve(strict=True)
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    assert path not in stack, f"cyclic config inheritance: {path}"
    namespace = runpy.run_path(str(path))
    bases = namespace.get("_base_", ())
    if isinstance(bases, str):
        bases = (bases,)
    assert isinstance(bases, (list, tuple))
    merged: dict[str, object] = {}
    for base in bases:
        assert isinstance(base, str)
        merged = _merge_real_config(
            merged,
            _load_real_config((path.parent / base), (*stack, path)),
        )  # type: ignore[assignment]
    own = {key: value for key, value in namespace.items() if not key.startswith("_")}
    return _merge_real_config(merged, own)  # type: ignore[return-value]


class _ExecutionSnapshot:
    def __init__(self, owner: "_Task", queue: str) -> None:
        self.owner = owner
        self.queue = queue

    @property
    def artifacts(self) -> list[object]:
        return [
            SimpleNamespace(key=name, artifact=artifact)
            for name, artifact in self.owner.artifacts.items()
        ]


class _Task:
    def __init__(
        self,
        *,
        task_id: str,
        name: str,
        parent: str,
        script: dict[str, object],
        parameters: dict[str, object],
        status: str = "created",
        queue: str = "",
        project: str = "e" * 32,
        script_drift_after_set: bool = False,
        args_drift_after_set: bool = False,
        on_get_models: object | None = None,
    ) -> None:
        self.id = task_id
        self.name = name
        self.output_uri = "http://10.100.34.118:8081/tasks/output"
        self.parameters = copy.deepcopy(parameters)
        self.artifacts: dict[str, object] = {}
        self.output_models: list[object] = []
        self._offline_mode = False
        self._reload_skip_flag = True
        self.reload_calls = 0
        self.public_reload_calls = 0
        self.script_drift_after_set = script_drift_after_set
        self.args_drift_after_set = args_drift_after_set
        self.on_get_models = on_get_models
        self.get_models_calls = 0
        self._data = SimpleNamespace(
            id=task_id,
            status=status,
            parent=parent,
            name=name,
            project=project,
            script=SimpleNamespace(**copy.deepcopy(script)),
            execution=_ExecutionSnapshot(self, queue),
            last_worker="worker-a100",
            output=SimpleNamespace(destination=self.output_uri),
        )

    @property
    def data(self) -> object:
        return self._data

    @property
    def parent(self) -> str:
        return str(self._data.parent)

    @parent.setter
    def parent(self, value: str) -> None:
        self._data.parent = value

    @property
    def status(self) -> str:
        return str(self._data.status)

    @status.setter
    def status(self, value: str) -> None:
        self._data.status = value

    def _reload(self) -> object:
        assert self._reload_skip_flag is False
        self.reload_calls += 1
        return self._data

    def reload(self) -> None:
        self.public_reload_calls += 1
        raise AssertionError("public Task.reload() must never be used")

    def get_parameters(self, **_kwargs: object) -> dict[str, object]:
        return copy.deepcopy(self.parameters)

    def set_parameters(self, values: dict[str, object], **_kwargs: object) -> bool:
        self.parameters = copy.deepcopy(values)
        if self.args_drift_after_set:
            self.parameters["Args/gpus"] = 8
        return True

    def set_script(self, *args: object, **kwargs: object) -> bool:
        values = dict(args[0]) if args else dict(kwargs)
        if self.script_drift_after_set:
            values["diff"] = str(values["diff"]) + "# drift"
        self.data.script = SimpleNamespace(**values)
        return True

    def get_models(self) -> dict[str, list[object]]:
        self.get_models_calls += 1
        result = {"output": list(self.output_models)}
        if callable(self.on_get_models):
            self.on_get_models(self)
        return result


class _Backend:
    def __init__(
        self,
        tasks: list[_Task],
        *,
        enqueue_raises_after_commit: bool = False,
    ) -> None:
        self.tasks = tasks
        self.clone_calls = 0
        self.enqueue_calls = 0
        self.last_enqueue_queue_id: str | None = None
        self.enqueue_raises_after_commit = enqueue_raises_after_commit

    def get_tasks(self, **kwargs: object) -> list[_Task]:
        pattern = kwargs["task_name"]
        return [task for task in self.tasks if re.fullmatch(pattern, task.name)]

    def get_project_id(self, **_kwargs: object) -> str:
        return "a" * 32

    def clone(self, **_kwargs: object) -> _Task:
        self.clone_calls += 1
        raise AssertionError("unexpected clone")

    def enqueue(self, *, task: _Task, queue_id: str, force: bool) -> dict[str, int]:
        assert force is False
        self.enqueue_calls += 1
        self.last_enqueue_queue_id = queue_id
        task.status = "queued"
        task.data.execution.queue = queue_id
        if self.enqueue_raises_after_commit:
            raise RuntimeError("transport lost after server commit")
        return {"queued": 1, "updated": 1}


def _script(entry: str, source: str) -> dict[str, object]:
    return {
        "repository": "",
        "working_dir": ".",
        "entry_point": entry,
        "diff": source,
    }


def _v2_plan(
    module: object,
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    audit_seal = "3" * 64
    progress_seal = "4" * 64
    provenance_seal = "5" * 64
    equivalence = {
        "legacy_script_sha256": module.LEGACY_TRAINING_SCRIPT_SHA256,
        "canonical_script_sha256": module.CANONICAL_TRAINING_SCRIPT_SHA256,
        "legacy_script_subjects": list(module.LEGACY_SCRIPT_SUBJECTS),
        "only_difference": module.TRAINING_SCRIPT_ONLY_DIFFERENCE,
        "runtime_usage_closure": list(module.TRAINING_SCRIPT_RUNTIME_USAGE_CLOSURE),
    }
    selector_payload = {
            "schema_version": 3,
            "audit_task_id": "a" * 32,
            "leaderboard_task_id": "b" * 32,
            "training_controller_task_id": "c" * 32,
            "training_provenance_task_id": "d" * 32,
            "watcher_task_id": "f" * 32,
            "training_progress_seal_sha256": progress_seal,
            "training_provenance_seal_sha256": provenance_seal,
            "training_script_equivalence": equivalence,
            "evaluation_script_sha256": module.EVALUATION_SCRIPT_SHA256,
            "audit_seal_sha256": audit_seal,
        }
    selector_payload.update(_source_revision_binding())
    selector_artifact = module._sealed(selector_payload)
    payload: dict[str, object] = {
        "schema_version": 2,
        "formal_audit_seal_sha256": audit_seal,
        "training_provenance_task_id": "d" * 32,
        "training_progress_artifact_seal_sha256": progress_seal,
        "training_provenance_artifact_seal_sha256": provenance_seal,
        "training_script_equivalence": equivalence,
        "evaluation_script_sha256": module.EVALUATION_SCRIPT_SHA256,
        "initial_formal_selector_seal_sha256": selector_artifact["seal_sha256"],
        "formal_selector_provenance": {
            "task_id": "9" * 32,
            "artifact_seal_sha256": selector_artifact["seal_sha256"],
            "artifact": selector_artifact,
        },
    }
    if extra is not None:
        payload.update(copy.deepcopy(extra))
    return module._sealed(payload)


def _reseal_selector_in_plan(
    module: object,
    plan: dict[str, object],
) -> dict[str, object]:
    updated = copy.deepcopy(plan)
    selector = updated["formal_selector_provenance"]
    artifact = module._sealed(selector["artifact"])
    selector["artifact"] = artifact
    selector["artifact_seal_sha256"] = artifact["seal_sha256"]
    updated["initial_formal_selector_seal_sha256"] = artifact["seal_sha256"]
    return module._sealed(updated)


def _record() -> dict[str, object]:
    return {
        "task_key": "train-r01-s01-resilient_v2x",
        "worker_queue": "GPU4-A100",
    }


def test_standalone_embeds_the_exact_pinned_plan_validator() -> None:
    module = _load_module()
    source = module.generate_standalone_source()
    compile(source, "<standalone-executor>", "exec")
    assert hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == (
        module.PINNED_PLAN_VALIDATOR_SHA256
    )
    assert module.EXPECTED_SOURCE_D_PRODUCER_SHA256 == (
        "d5b759f38d39a9f349ab6e716c07fda53eb3e1635687ce4a077e0405920ffeec"
    )


def test_planner_producer_pin_matches_provisional_generated_standalone() -> None:
    module = _load_module()
    spec = importlib.util.spec_from_file_location(
        "clearml_formal_multiseed_plan_pin_fixture",
        PLANNER_PRODUCER_PATH,
    )
    assert spec is not None and spec.loader is not None
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)

    assert hashlib.sha256(PLANNER_PRODUCER_PATH.read_bytes()).hexdigest() == (
        "4ae93288665cf998df805ed0ac7b013af3df95454ffda62532aac07edcfc296e"
    )
    standalone = producer.generate_standalone_source()
    assert hashlib.sha256(standalone.encode("utf-8")).hexdigest() == (
        module.EXPECTED_PLANNER_PRODUCER_SHA256
    )
    assert module.EXPECTED_PLANNER_PRODUCER_SHA256 == (
        "0b5f3b60204ec876f01f417bf9b7c7e8cabd8a219d92aa453f7249ca2ad19047"
    )
    assert producer.PLAN_VALIDATOR_SHA256 == module.PINNED_PLAN_VALIDATOR_SHA256
    assert module.PLANNER_PRODUCER_PIN_FINALIZED is False
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="deployment pin is not final",
    ):
        module._require_deployment_pins()


def test_portable_source_d_chain_pins_match_reviewed_producer() -> None:
    module = _load_module()
    producer = _load_source_d_producer_module()
    wrapper_sha = hashlib.sha256(SOURCE_D_PRODUCER_PATH.read_bytes()).hexdigest()
    assert wrapper_sha == (
        "e94d5c1702bcd50d1ac79d8b2ad6a314f8986a49d5ac45f07a3d5e62f06384df"
    )
    standalone = producer.generate_standalone_source()
    standalone_sha = hashlib.sha256(standalone.encode("utf-8")).hexdigest()
    assert standalone_sha == module.EXPECTED_SOURCE_D_PRODUCER_SHA256
    assert (
        standalone_sha
        == "d5b759f38d39a9f349ab6e716c07fda53eb3e1635687ce4a077e0405920ffeec"
    )
    assert producer.SOURCE_C_TASK_ID == module.SOURCE_C_TASK_ID
    assert producer.SOURCE_C_PARENT_TASK_ID == module.SOURCE_C_PARENT_TASK_ID
    assert producer.SOURCE_C_ENTRY_POINT == module.SOURCE_C_ENTRY_POINT
    assert producer.SOURCE_C_SHA256 == module.EXPECTED_SOURCE_C_SCRIPT_SHA256
    assert producer.SOURCE_D_SHA256 == module.EXPECTED_SOURCE_D_SCRIPT_SHA256
    assert (
        producer.EQUIVALENCE_ARTIFACT_SHA256
        == module.EXPECTED_SOURCE_D_EQUIVALENCE_SHA256
    )
    builder_sha = hashlib.sha256(SOURCE_D_BUILDER_PATH.read_bytes()).hexdigest()
    assert builder_sha == module.EXPECTED_SOURCE_D_BUILDER_SHA256
    assert builder_sha == producer.BUILDER_SOURCE_SHA256
    assert producer.TRANSFORMATION_ID == module.SOURCE_D_TRANSFORMATION_ID
    assert producer.DECLARED_REPLACEMENT_COUNT == (
        module.SOURCE_D_DECLARED_REPLACEMENT_COUNT
    )
    assert producer.UNCHANGED_SEGMENT_COUNT == module.SOURCE_D_UNCHANGED_SEGMENT_COUNT
    assert producer.STAGING_ANCHOR_NAMES == module.SOURCE_D_STAGING_ANCHOR_NAMES
    assert len(producer.STAGING_ANCHOR_NAMES) == 6

    assert producer.PORTABLE_RUNNER_LOAD_MARKER == module.PORTABLE_RUNNER_LOAD_MARKER
    assert producer.EXPECTED_PORTABLE_RUNNER_LOAD_MARKER_COUNT == (
        module.PORTABLE_RUNNER_LOAD_MARKER_COUNT
    )
    assert module.SOURCE_D_RUNTIME_PROFILE == "rtx5090"
    assert module.EXPECTED_TEACHER_SCRIPT_SHA256 != (
        module.EXPECTED_SOURCE_C_SCRIPT_SHA256
    )


@pytest.mark.parametrize(
    ("subject", "relative_path", "expected_nested_teacher"),
    REAL_EXPERIMENT_CONFIGS,
    ids=[subject for subject, _path, _nested in REAL_EXPERIMENT_CONFIGS],
)
def test_nested_teacher_subjects_match_26_real_resolved_configs(
    subject: str,
    relative_path: str,
    expected_nested_teacher: bool,
) -> None:
    module = _load_module()
    assert len(REAL_EXPERIMENT_CONFIGS) == 26
    assert len({item[0] for item in REAL_EXPERIMENT_CONFIGS}) == 26
    assert module.NESTED_TEACHER_SUBJECTS <= {
        item[0] for item in REAL_EXPERIMENT_CONFIGS
    }
    config = _load_real_config(ROOT / relative_path)
    model = config.get("model")
    assert isinstance(model, dict)
    actual_nested_teacher = isinstance(model.get("teacher"), dict)
    assert actual_nested_teacher is expected_nested_teacher
    assert (subject in module.NESTED_TEACHER_SUBJECTS) is expected_nested_teacher


def test_task_name_binds_full_executor_id_and_task_key() -> None:
    module = _load_module()
    assert module._task_run_name(_record(), executor_task_id=EXECUTOR_ID) == (
        f"ResilientV2X formal 1337 {EXECUTOR_ID} train-r01-s01-resilient_v2x"
    )


@pytest.mark.parametrize("stage", ("teacher", "source_d", "final"))
def test_created_task_recovers_only_from_three_exact_stages(stage: str) -> None:
    module = _load_module()
    teacher_source = "# teacher\n"
    source_d = "# source d\n"
    teacher_parameters = {"Args/stage": "teacher"}
    expected = {"Args/stage": "all", "Args/gpus": 4}
    script = (
        _script(module.TRAINING_ENTRY_POINT, source_d)
        if stage != "teacher"
        else _script(module.TRAINING_ENTRY_POINT, teacher_source)
    )
    parameters = expected if stage == "final" else teacher_parameters
    name = module._task_run_name(_record(), executor_task_id=EXECUTOR_ID)
    task = _Task(
        task_id="1" * 32,
        name=name,
        parent=PARENT_ID,
        script=script,
        parameters=parameters,
    )
    backend = _Backend([task])
    resolved = module._clone_or_resume_created(
        backend,
        teacher_task=object(),
        teacher_script=_script(module.TRAINING_ENTRY_POINT, teacher_source),
        teacher_parameters=teacher_parameters,
        source_d=source_d,
        record=_record(),
        parent_task_id=PARENT_ID,
        executor_task_id=EXECUTOR_ID,
        expected_parameters=expected,
        project=module.DEFAULT_PROJECT,
        used_ids=set(),
    )
    assert resolved is task
    assert task.parameters == expected
    assert task.data.script.diff == source_d
    assert backend.clone_calls == 0


@pytest.mark.parametrize(
    ("script_drift", "args_drift", "message"),
    (
        (True, False, "script setter"),
        (False, True, "Args setter"),
    ),
)
def test_partial_setter_roundtrip_drift_fails_closed(
    script_drift: bool, args_drift: bool, message: str
) -> None:
    module = _load_module()
    teacher_source = "# teacher\n"
    name = module._task_run_name(_record(), executor_task_id=EXECUTOR_ID)
    task = _Task(
        task_id="2" * 32,
        name=name,
        parent=PARENT_ID,
        script=_script(module.TRAINING_ENTRY_POINT, teacher_source),
        parameters={"Args/stage": "teacher"},
        script_drift_after_set=script_drift,
        args_drift_after_set=args_drift,
    )
    with pytest.raises(module.FormalMultiseedExecutionError, match=message):
        module._clone_or_resume_created(
            _Backend([task]),
            teacher_task=object(),
            teacher_script=_script(module.TRAINING_ENTRY_POINT, teacher_source),
            teacher_parameters={"Args/stage": "teacher"},
            source_d="# source d\n",
            record=_record(),
            parent_task_id=PARENT_ID,
            executor_task_id=EXECUTOR_ID,
            expected_parameters={"Args/stage": "all", "Args/gpus": 4},
            project=module.DEFAULT_PROJECT,
            used_ids=set(),
        )


def test_drifted_or_failed_existing_name_never_clones() -> None:
    module = _load_module()
    name = module._task_run_name(_record(), executor_task_id=EXECUTOR_ID)
    for status in ("created", "failed"):
        task = _Task(
            task_id=("3" if status == "created" else "4") * 32,
            name=name,
            parent=PARENT_ID,
            script=_script(module.TRAINING_ENTRY_POINT, "# attacker\n"),
            parameters={"Args/stage": "attacker"},
            status=status,
        )
        backend = _Backend([task])
        with pytest.raises(module.FormalMultiseedExecutionError):
            module._clone_or_resume_created(
                backend,
                teacher_task=object(),
                teacher_script=_script(module.TRAINING_ENTRY_POINT, "# teacher\n"),
                teacher_parameters={"Args/stage": "teacher"},
                source_d="# source d\n",
                record=_record(),
                parent_task_id=PARENT_ID,
                executor_task_id=EXECUTOR_ID,
                expected_parameters={"Args/stage": "all"},
                project=module.DEFAULT_PROJECT,
                used_ids=set(),
            )
        assert backend.clone_calls == 0


def test_enqueue_exception_after_verified_commit_is_not_retried() -> None:
    module = _load_module()
    task = _Task(
        task_id="5" * 32,
        name="task",
        parent=PARENT_ID,
        script=_script(module.TRAINING_ENTRY_POINT, "# source d\n"),
        parameters={},
    )
    backend = _Backend([task], enqueue_raises_after_commit=True)
    module._enqueue_once(backend, task, queue_name="GPU4-A100")
    module._enqueue_once(backend, task, queue_name="GPU4-A100")
    assert backend.enqueue_calls == 1


def _receipts() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    queues = ("GPU4-A100", "GPU4-V100", "GPU4-5090")
    training = []
    evaluation = []
    for queue_index, queue in enumerate(queues):
        for index in range(15):
            offset = queue_index * 15 + index + 1
            training.append(
                {
                    "task_key": f"train-{offset:02d}",
                    "task_id": f"{offset:032x}",
                    "worker_queue": queue,
                    "result": {"model_id": f"{1000 + offset:032x}"},
                }
            )
            evaluation.append(
                {
                    "task_key": f"eval-{offset:02d}",
                    "task_id": f"{100 + offset:032x}",
                    "worker_queue": queue,
                    "result": {},
                }
            )
    return training, evaluation


def test_manifest_is_initial_only_and_model_ids_are_unique() -> None:
    module = _load_module()
    training, evaluation = _receipts()
    plan = _v2_plan(
        module,
        extra={
            "training_tasks": [
                {"task_key": receipt["task_key"]} for receipt in training
            ],
            "evaluation_tasks": [
                {"task_key": receipt["task_key"]} for receipt in evaluation
            ],
        },
    )
    manifest = module._manifest_payload(
        output_task_id=EXECUTOR_ID,
        executor_script_sha256="b" * 64,
        executor_parameters_sha256="c" * 64,
        plan=plan,
        dependency_fingerprints={},
        source_d={},
        teacher={},
        training_receipts=training,
        evaluation_receipts=evaluation,
    )
    expected_provenance = module._formal_plan_provenance(plan)
    assert manifest["schema_version"] == 2
    assert manifest["formal_plan_provenance"] == expected_provenance
    assert set(
        manifest["formal_plan_provenance"]["formal_selector_dependency_task_ids"]
    ) == set(module.SELECTOR_DEPENDENCY_TASK_ID_FIELDS)
    assert manifest["training_task_count"] == 45
    assert manifest["evaluation_task_count"] == 45
    assert manifest["evaluation_condition_count"] == 540
    assert manifest["gate_result_produced"] is False
    assert manifest["fallback_authorized"] is False
    assert manifest["fallback_execution_enabled"] is False
    assert len(set(manifest["all_execution_task_ids"])) == 90
    assert (
        module._validate_manifest_shape(
            manifest,
            output_task_id=EXECUTOR_ID,
            executor_script_sha256="b" * 64,
            executor_parameters_sha256="c" * 64,
            plan=plan,
            dependency_fingerprints={},
            source_d={},
            teacher={},
        )
        == manifest
    )

    old_manifest = copy.deepcopy(manifest)
    old_manifest["schema_version"] = 1
    old_manifest = module._sealed(old_manifest)
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="execution manifest schema must be exactly integer 2",
    ):
        module._validate_manifest_shape(
            old_manifest,
            output_task_id=EXECUTOR_ID,
            executor_script_sha256="b" * 64,
            executor_parameters_sha256="c" * 64,
            plan=plan,
            dependency_fingerprints={},
            source_d={},
            teacher={},
        )

    drifted_provenance = copy.deepcopy(manifest)
    (
        drifted_provenance["formal_plan_provenance"][
            "training_progress_artifact_seal_sha256"
        ],
        drifted_provenance["formal_plan_provenance"][
            "training_provenance_artifact_seal_sha256"
        ],
    ) = (
        drifted_provenance["formal_plan_provenance"][
            "training_provenance_artifact_seal_sha256"
        ],
        drifted_provenance["formal_plan_provenance"][
            "training_progress_artifact_seal_sha256"
        ],
    )
    drifted_provenance = module._sealed(drifted_provenance)
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="formal_plan_provenance drifted",
    ):
        module._validate_manifest_shape(
            drifted_provenance,
            output_task_id=EXECUTOR_ID,
            executor_script_sha256="b" * 64,
            executor_parameters_sha256="c" * 64,
            plan=plan,
            dependency_fingerprints={},
            source_d={},
            teacher={},
        )

    pristine_plan = copy.deepcopy(plan)
    snapshot = copy.deepcopy(manifest)
    plan["training_script_equivalence"]["only_difference"] = "mutated"
    plan["formal_selector_provenance"]["artifact"]["training_progress_seal_sha256"] = (
        "0" * 64
    )
    assert manifest == snapshot

    duplicated = copy.deepcopy(training)
    duplicated[1]["result"]["model_id"] = duplicated[0]["result"]["model_id"]
    with pytest.raises(module.FormalMultiseedExecutionError, match="OutputModel IDs"):
        module._manifest_payload(
            output_task_id=EXECUTOR_ID,
            executor_script_sha256="b" * 64,
            executor_parameters_sha256="c" * 64,
            plan=pristine_plan,
            dependency_fingerprints={},
            source_d={},
            teacher={},
            training_receipts=duplicated,
            evaluation_receipts=evaluation,
        )


@pytest.mark.parametrize(
    "url",
    (
        "http://10.100.34.118:8081/a?download=1",
        "http://10.100.34.118:8081/a/%2e%2e/secret",
        "http://user@10.100.34.118:8081/a",
    ),
)
def test_fileserver_url_rejects_query_credentials_and_encoded_traversal(
    url: str,
) -> None:
    module = _load_module()
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._require_fileserver_url(url, context="model")


class _Artifact:
    def __init__(
        self,
        value: object,
        *,
        local_copy: Path | None = None,
        artifact_type: str = "dict",
        url: str = "http://10.100.34.118:8081/artifacts/artifact.json",
        artifact_hash: str = "a" * 64,
        size: int = 1,
        mode: object = "output",
        on_get: object | None = None,
    ) -> None:
        self.value = copy.deepcopy(value)
        self.local_copy = local_copy
        self.type = artifact_type
        self.url = url
        self.hash = artifact_hash
        self.size = size
        self.mode = mode
        self.on_get = on_get
        self.name = "artifact"
        self.force_download_calls: list[bool] = []

    def get(self, *, force_download: bool = False) -> object:
        self.force_download_calls.append(force_download)
        assert force_download is True
        result = copy.deepcopy(self.value)
        if callable(self.on_get):
            self.on_get(self)
        return result

    def get_local_copy(
        self,
        *,
        extract_archive: bool,
        raise_on_error: bool,
        force_download: bool,
    ) -> str:
        assert extract_archive is False
        assert raise_on_error is True
        assert force_download is True
        assert self.local_copy is not None
        return str(self.local_copy)


def _server_artifact_task(artifacts: dict[str, _Artifact]) -> object:
    return SimpleNamespace(
        data=SimpleNamespace(
            execution=SimpleNamespace(
                artifacts=[
                    SimpleNamespace(key=name, artifact=artifact)
                    for name, artifact in artifacts.items()
                ]
            )
        )
    )


def _source_d_evidence_fixture(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_Task, dict[str, object], str]:
    task_id = "4" * 32
    producer_source = "#!/usr/bin/env python3\n# synthetic portable Source-D producer\n"
    source_c = "#!/usr/bin/env python3\n# independent Source-C fixture\n"
    source_d = (
        "#!/usr/bin/env python3\n"
        f"def {module.PORTABLE_RUNNER_LOAD_MARKER}(contract):\n"
        "    return contract\n"
        f"_portable_runtime_validator = {module.PORTABLE_RUNNER_LOAD_MARKER}\n"
        'RUNTIME_PROFILE = "rtx5090"\n'
    )
    producer_sha = hashlib.sha256(producer_source.encode("utf-8")).hexdigest()
    source_c_sha = hashlib.sha256(source_c.encode("utf-8")).hexdigest()
    source_d_sha = hashlib.sha256(source_d.encode("utf-8")).hexdigest()
    replacement_names = (
        tuple(f"seed_replacement_{index}" for index in range(1, 17))
        + module.SOURCE_D_STAGING_ANCHOR_NAMES
    )
    assert len(replacement_names) == module.SOURCE_D_DECLARED_REPLACEMENT_COUNT
    equivalence = {
        "schema_version": 1,
        "artifact_type": "resilient_v2x_formal_source_d_seed_diff",
        "transformation_id": module.SOURCE_D_TRANSFORMATION_ID,
        "seed_contract": {
            "training_overlay_protocol_seed": (module.TRAINING_OVERLAY_PROTOCOL_SEED)
        },
        "diff": [
            {
                "index": index,
                "name": name,
                "expected_count": 1,
                "observed_count": 1,
            }
            for index, name in enumerate(replacement_names, start=1)
        ],
        "equivalence": {
            "only_declared_anchor_replacements": True,
            "declared_replacement_count": (module.SOURCE_D_DECLARED_REPLACEMENT_COUNT),
            "unchanged_segment_count": module.SOURCE_D_UNCHANGED_SEGMENT_COUNT,
        },
    }
    equivalence["artifact_sha256"] = module._content_sha256(equivalence)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_D_PRODUCER_SHA256", producer_sha)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_C_SCRIPT_SHA256", source_c_sha)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_D_SCRIPT_SHA256", source_d_sha)
    monkeypatch.setattr(
        module,
        "EXPECTED_SOURCE_D_EQUIVALENCE_SHA256",
        equivalence["artifact_sha256"],
    )

    source_c_document = module._sealed(
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
                "sha256": source_c_sha,
                "size_bytes": len(source_c.encode("utf-8")),
                "line_count": len(source_c.splitlines()),
            },
            "script_diff": source_c,
        }
    )
    source_d_document = module._sealed(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_d_script",
            "complete": True,
            "transformation_id": module.SOURCE_D_TRANSFORMATION_ID,
            "source_c_sha256": source_c_sha,
            "source_d_sha256": source_d_sha,
            "size_bytes": len(source_d.encode("utf-8")),
            "line_count": len(source_d.splitlines()),
            "script": source_d,
        }
    )
    receipt = module._sealed(
        {
            "schema_version": 1,
            "artifact_type": ("resilient_v2x_formal_source_d_evidence_receipt"),
            "complete": True,
            "publication_order": list(module.SOURCE_D_ARTIFACT_ORDER),
            "provenance": {
                "source_c_task_id": module.SOURCE_C_TASK_ID,
                "source_c_task_status": "completed",
                "source_c_task_parent": module.SOURCE_C_PARENT_TASK_ID,
                "source_c_entry_point": module.SOURCE_C_ENTRY_POINT,
                "source_c_sha256": source_c_sha,
                "output_task_id": task_id,
                "output_parent_task_id": module.SOURCE_C_TASK_ID,
                "builder_source_sha256": (module.EXPECTED_SOURCE_D_BUILDER_SHA256),
                "transformation_id": module.SOURCE_D_TRANSFORMATION_ID,
                "producer_entry_point": module.SOURCE_D_PRODUCER_ENTRY_POINT,
                "producer_script_sha256": producer_sha,
            },
            "transformation": {
                "source_d_sha256": source_d_sha,
                "equivalence_artifact_sha256": equivalence["artifact_sha256"],
                "declared_replacement_count": module.SOURCE_D_DECLARED_REPLACEMENT_COUNT,
                "unchanged_segment_count": module.SOURCE_D_UNCHANGED_SEGMENT_COUNT,
                "only_declared_anchor_replacements": True,
                "training_overlay_protocol_seed": (
                    module.TRAINING_OVERLAY_PROTOCOL_SEED
                ),
                "training_seed_cli": "--training-seed",
                "portable_runner_load_marker": (module.PORTABLE_RUNNER_LOAD_MARKER),
                "portable_runner_load_marker_count": (
                    module.PORTABLE_RUNNER_LOAD_MARKER_COUNT
                ),
                "legacy_runner_load_target_anchor_count": (
                    module.LEGACY_RUNNER_LOAD_TARGET_ANCHOR_COUNT
                ),
            },
            "artifact_hashes": {
                module.SOURCE_C_SNAPSHOT_ARTIFACT: (source_c_document["seal_sha256"]),
                module.SOURCE_D_SCRIPT_ARTIFACT: (source_d_document["seal_sha256"]),
                module.SOURCE_D_EQUIVALENCE_ARTIFACT: (equivalence["artifact_sha256"]),
            },
        }
    )
    task = _Task(
        task_id=task_id,
        name="portable Source-D evidence",
        parent=module.SOURCE_C_TASK_ID,
        script=_script(module.SOURCE_D_PRODUCER_ENTRY_POINT, producer_source),
        parameters={},
        status="completed",
    )
    task.artifacts = {
        module.SOURCE_C_SNAPSHOT_ARTIFACT: _Artifact(source_c_document),
        module.SOURCE_D_SCRIPT_ARTIFACT: _Artifact(source_d_document),
        module.SOURCE_D_EQUIVALENCE_ARTIFACT: _Artifact(equivalence),
        module.SOURCE_D_RECEIPT_ARTIFACT: _Artifact(receipt),
    }
    plan = {
        "source_d_identity": {
            "script_sha256": source_d_sha,
            "equivalence_artifact_sha256": equivalence["artifact_sha256"],
        }
    }
    return task, plan, source_d


def test_portable_source_d_evidence_binds_source_c_parent_and_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    task, plan, source_d = _source_d_evidence_fixture(module, monkeypatch)
    validated = module._validate_source_d_evidence(
        task,
        task_id=task.id,
        plan=plan,
    )
    assert validated["source_c_task_id"] == module.SOURCE_C_TASK_ID
    assert validated["source_c_parent_task_id"] == module.SOURCE_C_PARENT_TASK_ID
    assert validated["output_parent_task_id"] == module.SOURCE_C_TASK_ID
    assert validated["script"] == source_d
    fingerprint = module._task_fingerprint(
        task,
        context="portable Source-D fixture",
        expected_id=task.id,
        entry_point=module.SOURCE_D_PRODUCER_ENTRY_POINT,
        script_sha256=module.EXPECTED_SOURCE_D_PRODUCER_SHA256,
        artifact_names=module.SOURCE_D_ARTIFACT_ORDER,
    )
    assert fingerprint["parent_task_id"] == module.SOURCE_C_TASK_ID


def test_reload_uses_only_raw_server_snapshot_and_restores_skip_flag() -> None:
    module = _load_module()
    task = _Task(
        task_id="1" * 32,
        name="raw reload",
        parent="",
        script=_script("producer.py", "# producer\n"),
        parameters={},
        status="completed",
    )

    module._reload(task, context="raw reload fixture")

    assert task.reload_calls == 1
    assert task.public_reload_calls == 0
    assert task._reload_skip_flag is True
    assert task.data is task._data


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("offline", "offline"),
        ("missing", "server-reloaded"),
        ("none", "returned no snapshot"),
        ("primitive", "returned no snapshot"),
        ("identity", "identity mismatch"),
    ),
)
def test_reload_rejects_untrusted_snapshot_attacks(
    attack: str,
    message: str,
) -> None:
    module = _load_module()
    task = _Task(
        task_id="1" * 32,
        name="raw reload",
        parent="",
        script=_script("producer.py", "# producer\n"),
        parameters={},
        status="completed",
    )
    if attack == "offline":
        task._offline_mode = True
    elif attack == "missing":
        task._reload = None  # type: ignore[method-assign]
    elif attack == "none":
        task._reload = lambda: None  # type: ignore[method-assign]
    elif attack == "primitive":
        task._reload = lambda: True  # type: ignore[method-assign]
    else:
        task._reload = lambda: SimpleNamespace(id="2" * 32)  # type: ignore[method-assign]

    with pytest.raises(module.FormalMultiseedExecutionError, match=message):
        module._reload(task, context="attacked raw reload")

    assert task.public_reload_calls == 0
    assert task._reload_skip_flag is True


def test_status_reads_only_the_installed_raw_snapshot() -> None:
    module = _load_module()

    class PublicStatusSpoof:
        data = SimpleNamespace(status="failed")

        @property
        def status(self) -> str:
            raise AssertionError("public status must not be read")

        def get_status(self) -> str:
            raise AssertionError("public get_status must not be called")

    assert module._status(PublicStatusSpoof(), context="status fixture") == "failed"


def test_status_rejects_missing_raw_status_without_public_fallback() -> None:
    module = _load_module()

    class PublicStatusOnly:
        data = SimpleNamespace(status=None)
        status = "completed"

        def get_status(self) -> str:
            return "completed"

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="status is not a string",
    ):
        module._status(PublicStatusOnly(), context="status fixture")


def test_script_rejects_missing_raw_metadata_without_public_fallback() -> None:
    module = _load_module()
    public_getter_called = False

    def public_getter() -> dict[str, object]:
        nonlocal public_getter_called
        public_getter_called = True
        return _script("attacker.py", "# attacker\n")

    task = SimpleNamespace(
        data=SimpleNamespace(script=None),
        get_script=public_getter,
    )
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="installed server snapshot has no script metadata",
    ):
        module._script(task, context="script fixture")
    assert public_getter_called is False


def test_task_name_reads_only_the_installed_raw_snapshot() -> None:
    module = _load_module()
    task = SimpleNamespace(
        name="attacker name",
        data=SimpleNamespace(name="sealed name"),
    )

    assert module._task_name(task, context="name fixture") == "sealed name"

    task.data.name = None
    with pytest.raises(module.FormalMultiseedExecutionError, match="unavailable"):
        module._task_name(task, context="name fixture")


def test_output_uri_reads_only_raw_taskdata_destination() -> None:
    module = _load_module()
    expected = "http://10.100.34.118:8081/tasks/output"
    task = SimpleNamespace(
        output_uri="http://attacker.invalid/output",
        data=SimpleNamespace(output=SimpleNamespace(destination=expected)),
    )

    assert module._output_uri(task, context="output fixture") == expected

    task.data.output.destination = None
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._output_uri(task, context="output fixture")


def test_artifact_mapping_uses_server_snapshot_and_force_download() -> None:
    module = _load_module()
    artifact = _Artifact({"fresh": True})
    task = _server_artifact_task({"expected": artifact})
    task.artifacts = {"expected": _Artifact({"fresh": False})}

    assert module._artifact_mapping(
        task,
        "expected",
        context="server artifact fixture",
    ) == {"fresh": True}
    assert artifact.force_download_calls == [True]


def test_artifact_mapping_securely_reads_regular_json_path(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "artifact.json"
    path.write_text('{"fresh":true}', encoding="utf-8")
    task = _server_artifact_task({"expected": _Artifact(path)})

    assert module._artifact_mapping(
        task,
        "expected",
        context="path artifact fixture",
    ) == {"fresh": True}


@pytest.mark.parametrize("attack", ("symlink", "hardlink", "fifo"))
def test_artifact_mapping_rejects_unsafe_path_types_and_links(
    attack: str,
    tmp_path: Path,
) -> None:
    module = _load_module()
    source = tmp_path / "source.json"
    source.write_text('{"fresh":true}', encoding="utf-8")
    path = tmp_path / "artifact.json"
    if attack == "symlink":
        path.symlink_to(source)
    elif attack == "hardlink":
        os.link(source, path)
    else:
        os.mkfifo(path)
    task = _server_artifact_task({"expected": _Artifact(path)})

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="unsafe|securely read",
    ):
        module._artifact_mapping(
            task,
            "expected",
            context=f"{attack} artifact fixture",
        )


def test_artifact_mapping_rejects_path_over_byte_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    path = tmp_path / "artifact.json"
    path.write_text('{"value":"too large"}', encoding="utf-8")
    monkeypatch.setattr(module, "JSON_ARTIFACT_MAX_BYTES", 8)
    task = _server_artifact_task({"expected": _Artifact(path)})

    with pytest.raises(module.FormalMultiseedExecutionError, match="unsafe"):
        module._artifact_mapping(
            task,
            "expected",
            context="large artifact fixture",
        )


def test_artifact_mapping_rejects_malformed_local_path() -> None:
    module = _load_module()
    task = _server_artifact_task({"expected": _Artifact("artifact\x00.json")})

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="cannot be securely read",
    ):
        module._artifact_mapping(
            task,
            "expected",
            context="malformed path artifact fixture",
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b"\xff", "not UTF-8 JSON"),
        (b"[]", "must be a JSON object"),
    ),
)
def test_artifact_mapping_rejects_non_utf8_or_non_mapping_path_payload(
    payload: bytes,
    message: str,
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "artifact.json"
    path.write_bytes(payload)
    task = _server_artifact_task({"expected": _Artifact(path)})

    with pytest.raises(module.FormalMultiseedExecutionError, match=message):
        module._artifact_mapping(
            task,
            "expected",
            context="invalid artifact fixture",
        )


def test_artifact_mapping_rejects_fstat_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    path = tmp_path / "artifact.json"
    path.write_text('{"fresh":true}', encoding="utf-8")
    task = _server_artifact_task({"expected": _Artifact(path)})
    real_fstat = module.os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        values = {
            name: getattr(observed, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        if calls == 2:
            values["st_mtime_ns"] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(module.os, "fstat", drifting_fstat)
    with pytest.raises(
        module.FormalMultiseedExecutionError, match="changed during read"
    ):
        module._artifact_mapping(
            task,
            "expected",
            context="drifting artifact fixture",
        )


def test_artifact_mapping_never_falls_back_without_force_download() -> None:
    module = _load_module()

    class LegacyArtifact:
        def get(self) -> object:
            return {"stale": True}

    task = _server_artifact_task({"expected": LegacyArtifact()})
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="freshly downloaded",
    ):
        module._artifact_mapping(
            task,
            "expected",
            context="legacy artifact fixture",
        )


def test_task_fingerprint_rejects_extra_server_artifact() -> None:
    module = _load_module()
    source = "# producer\n"
    task = _Task(
        task_id="1" * 32,
        name="fingerprint",
        parent="",
        script=_script("producer.py", source),
        parameters={},
        status="completed",
    )
    task.artifacts = {
        "expected": _Artifact({"value": 1}),
        "attacker": _Artifact({"value": 2}),
    }

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="exact artifact inventory",
    ):
        module._task_fingerprint(
            task,
            context="fingerprint fixture",
            expected_id=task.id,
            entry_point="producer.py",
            script_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            artifact_names=("expected",),
        )


def test_resolve_plan_rejects_parent_not_matching_sealed_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    source = "# planner\n"
    plan = _v2_plan(
        module,
        extra={
            "candidate_count": 1,
            "training_task_count": 45,
            "evaluation_task_count": 45,
            "total_evaluation_condition_count": 540,
        },
    )
    planner = _Task(
        task_id="1" * 32,
        name="planner",
        parent="8" * 32,
        script=_script(module.PLANNER_ENTRY_POINT, source),
        parameters=dict(module.PLANNER_PARAMETER_CONTRACT),
        status="completed",
    )
    planner.artifacts = {
        module.PLAN_ARTIFACT: _Artifact(plan),
        module.PLANNER_RECEIPT_ARTIFACT: _Artifact({}),
    }
    monkeypatch.setattr(
        module,
        "_load_plan_validator",
        lambda _path: SimpleNamespace(validate_plan=lambda value: value),
    )

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="parent does not match the sealed selector",
    ):
        module._resolve_plan(
            planner,
            planner_task_id=planner.id,
            producer_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )


def test_formal_plan_provenance_projects_all_five_dependencies() -> None:
    module = _load_module()
    plan = _v2_plan(module)
    projected = module._formal_plan_provenance(plan)
    selector = plan["formal_selector_provenance"]
    artifact = selector["artifact"]

    assert projected == {
        "plan_schema_version": 2,
        "formal_selector_task_id": selector["task_id"],
        "training_provenance_task_id": plan["training_provenance_task_id"],
        "training_progress_artifact_seal_sha256": plan[
            "training_progress_artifact_seal_sha256"
        ],
        "training_provenance_artifact_seal_sha256": plan[
            "training_provenance_artifact_seal_sha256"
        ],
        "source_revision_equivalence_seal_sha256": artifact[
            "source_revision_equivalence_seal_sha256"
        ],
        "source_revision_subject_map_seal_sha256": artifact[
            "source_revision_subject_map_seal_sha256"
        ],
        "evaluation_source_revision_tree_sha256": artifact[
            "evaluation_source_revision_tree_sha256"
        ],
        "training_script_equivalence": plan["training_script_equivalence"],
        "evaluation_script_sha256": plan["evaluation_script_sha256"],
        "formal_selector_dependency_task_ids": {
            field: artifact[field]
            for field in module.SELECTOR_DEPENDENCY_TASK_ID_FIELDS
        },
    }


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("plan_v1", "formal plan schema"),
        ("selector_v1", "selector artifact schema"),
        ("missing_training_provenance", "exactly five task ID fields"),
        ("extra_task_id", "exactly five task ID fields"),
        ("duplicate_dependency", "five unique non-selector tasks"),
        ("selector_alias", "five unique non-selector tasks"),
    ),
)
def test_formal_plan_provenance_rejects_legacy_and_inventory_attacks(
    attack: str,
    message: str,
) -> None:
    module = _load_module()
    plan = _v2_plan(module)
    if attack == "plan_v1":
        plan["schema_version"] = 1
        plan = module._sealed(plan)
    else:
        artifact = plan["formal_selector_provenance"]["artifact"]
        if attack == "selector_v1":
            artifact["schema_version"] = 1
        elif attack == "missing_training_provenance":
            artifact.pop("training_provenance_task_id")
        elif attack == "extra_task_id":
            artifact["attacker_task_id"] = "8" * 32
        elif attack == "duplicate_dependency":
            artifact["training_provenance_task_id"] = artifact["watcher_task_id"]
            plan["training_provenance_task_id"] = artifact["watcher_task_id"]
        else:
            artifact["watcher_task_id"] = plan["formal_selector_provenance"]["task_id"]
        plan = _reseal_selector_in_plan(module, plan)

    with pytest.raises(module.FormalMultiseedExecutionError, match=message):
        module._formal_plan_provenance(plan)


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("training_provenance", "task cross-binding mismatch"),
        ("swap_seals", "progress seal cross-binding mismatch"),
        ("one_sided_equivalence", "semantic contract mismatch"),
        ("two_sided_equivalence", "semantic contract mismatch"),
        ("two_sided_legacy_eval", "evaluation script semantic contract"),
    ),
)
def test_formal_plan_provenance_rejects_cross_binding_and_semantic_attacks(
    attack: str,
    message: str,
) -> None:
    module = _load_module()
    plan = _v2_plan(module)
    artifact = plan["formal_selector_provenance"]["artifact"]
    if attack == "training_provenance":
        artifact["training_provenance_task_id"] = "8" * 32
    elif attack == "swap_seals":
        (
            artifact["training_progress_seal_sha256"],
            artifact["training_provenance_seal_sha256"],
        ) = (
            artifact["training_provenance_seal_sha256"],
            artifact["training_progress_seal_sha256"],
        )
    elif attack == "one_sided_equivalence":
        artifact["training_script_equivalence"]["only_difference"] = "attacker"
    elif attack == "two_sided_equivalence":
        artifact["training_script_equivalence"]["runtime_usage_closure"].reverse()
        plan["training_script_equivalence"]["runtime_usage_closure"].reverse()
    else:
        artifact["evaluation_script_sha256"] = module.LEGACY_TRAINING_SCRIPT_SHA256
        plan["evaluation_script_sha256"] = module.LEGACY_TRAINING_SCRIPT_SHA256
    plan = _reseal_selector_in_plan(module, plan)

    with pytest.raises(module.FormalMultiseedExecutionError, match=message):
        module._formal_plan_provenance(plan)


@pytest.mark.parametrize("attack", ("plan_v1", "four_dependencies"))
def test_resolve_plan_independently_rejects_legacy_identity_validator_output(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    module = _load_module()
    source = "# planner\n"
    plan = _v2_plan(
        module,
        extra={
            "candidate_count": 1,
            "training_task_count": 45,
            "evaluation_task_count": 45,
            "total_evaluation_condition_count": 540,
        },
    )
    if attack == "plan_v1":
        plan["schema_version"] = 1
        plan = module._sealed(plan)
    else:
        plan["formal_selector_provenance"]["artifact"].pop(
            "training_provenance_task_id"
        )
        plan = _reseal_selector_in_plan(module, plan)
    planner = _Task(
        task_id="1" * 32,
        name="planner",
        parent="9" * 32,
        script=_script(module.PLANNER_ENTRY_POINT, source),
        parameters=dict(module.PLANNER_PARAMETER_CONTRACT),
        status="completed",
    )
    planner.artifacts = {
        module.PLAN_ARTIFACT: _Artifact(plan),
        module.PLANNER_RECEIPT_ARTIFACT: _Artifact({}),
    }
    monkeypatch.setattr(
        module,
        "_load_plan_validator",
        lambda _path: SimpleNamespace(validate_plan=lambda value: value),
    )

    with pytest.raises(module.FormalMultiseedExecutionError):
        module._resolve_plan(
            planner,
            planner_task_id=planner.id,
            producer_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )


def test_planner_parameter_contract_matches_fixed_producer_defaults() -> None:
    module = _load_module()
    spec = importlib.util.spec_from_file_location(
        "clearml_formal_multiseed_plan_args_fixture",
        PLANNER_PRODUCER_PATH,
    )
    assert spec is not None and spec.loader is not None
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    producer_args = producer._parser().parse_args([])
    assert producer_args.poll_seconds == 60.0
    assert producer_args.timeout_hours == 720.0
    assert producer_args.emit_standalone is None
    planner = _Task(
        task_id="1" * 32,
        name="planner",
        parent="9" * 32,
        script=_script(module.PLANNER_ENTRY_POINT, "# planner\n"),
        parameters=dict(module.PLANNER_PARAMETER_CONTRACT),
        status="completed",
    )

    assert module._validate_planner_parameters(planner) == {
        "Args/poll_seconds": "60.0",
        "Args/timeout_hours": "720.0",
        "Args/emit_standalone": "",
    }


@pytest.mark.parametrize("attack", ("missing", "extra", "value", "type"))
def test_planner_parameter_contract_rejects_exact_args_drift(attack: str) -> None:
    module = _load_module()
    parameters = dict(module.PLANNER_PARAMETER_CONTRACT)
    if attack == "missing":
        parameters.pop("Args/emit_standalone")
    elif attack == "extra":
        parameters["Args/attacker"] = "enabled"
    elif attack == "value":
        parameters["Args/poll_seconds"] = "30.0"
    else:
        parameters["Args/poll_seconds"] = 60.0
    planner = _Task(
        task_id="1" * 32,
        name="planner",
        parent="9" * 32,
        script=_script(module.PLANNER_ENTRY_POINT, "# planner\n"),
        parameters=parameters,
        status="completed",
    )

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="planner parameter contract mismatch",
    ):
        module._validate_planner_parameters(planner)


def test_manifest_task_revalidation_raw_reloads_every_get_task_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    training_plan: list[dict[str, object]] = []
    evaluation_plan: list[dict[str, object]] = []
    training_receipts: list[dict[str, object]] = []
    evaluation_receipts: list[dict[str, object]] = []
    tasks_by_id: dict[str, _Task] = {}
    tasks_by_name: dict[str, _Task] = {}

    for index in range(45):
        training_key = f"train-{index:02d}"
        evaluation_key = f"eval-{index:02d}"
        training_record = {
            "task_key": training_key,
            "worker_queue": "GPU4-A100",
        }
        evaluation_record = {
            "task_key": evaluation_key,
            "training_task_key": training_key,
            "worker_queue": "GPU4-A100",
        }
        training_plan.append(training_record)
        evaluation_plan.append(evaluation_record)
        for offset, record, receipts in (
            (1, training_record, training_receipts),
            (46, evaluation_record, evaluation_receipts),
        ):
            task_id = f"{index + offset:032x}"
            name = module._task_run_name(record, executor_task_id=EXECUTOR_ID)
            task = _Task(
                task_id=task_id,
                name=name,
                parent=EXECUTOR_ID,
                script=_script(module.TRAINING_ENTRY_POINT, "# source d\n"),
                parameters={},
                status="completed",
            )
            tasks_by_id[task_id] = task
            tasks_by_name[name] = task
            receipts.append({"task_key": record["task_key"], "task_id": task_id})

    class TaskApi:
        @staticmethod
        def get_task(*, task_id: str) -> _Task:
            return tasks_by_id[task_id]

    monkeypatch.setattr(
        module,
        "_query_named_tasks",
        lambda _task_class, *, project, name: [tasks_by_name[name]],
    )
    monkeypatch.setattr(module, "_training_parameters", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_evaluation_parameters", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_require_bound_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_validate_training_completion",
        lambda task, *args, **kwargs: {"task_id": task.id},
    )
    monkeypatch.setattr(
        module,
        "_validate_evaluation_completion",
        lambda task, *args, **kwargs: {"task_id": task.id},
    )
    monkeypatch.setattr(
        module,
        "_task_receipt",
        lambda record, result, **kwargs: {
            "task_key": record["task_key"],
            "task_id": result["task_id"],
        },
    )

    module._revalidate_manifest_tasks(
        TaskApi,
        {
            "training_tasks": training_receipts,
            "evaluation_tasks": evaluation_receipts,
        },
        plan={
            "training_tasks": training_plan,
            "evaluation_tasks": evaluation_plan,
        },
        project=module.DEFAULT_PROJECT,
        output_task_id=EXECUTOR_ID,
        source_d_task_id=PARENT_ID,
        source_d="# source d\n",
        teacher_task=object(),
        teacher={},
    )

    assert len(tasks_by_id) == 90
    assert all(task.reload_calls == 1 for task in tasks_by_id.values())
    assert all(task.public_reload_calls == 0 for task in tasks_by_id.values())


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("task_parent_clean_teacher", "parent mismatch"),
        ("task_identity", "task identity mismatch"),
        ("snapshot_clean_teacher", "snapshot identity"),
        ("snapshot_entry_point", "metadata drifted"),
        ("snapshot_script_sha", "script SHA-256 drifted"),
        ("provenance_source_c_parent", "provenance drifted"),
        ("provenance_output_parent", "provenance drifted"),
        ("provenance_builder", "provenance drifted"),
        ("portable_marker", "transformation drifted"),
        ("portable_marker_count", "transformation drifted"),
        ("legacy_anchor_count", "transformation drifted"),
        ("plan_source_d", "plan and Source-D evidence disagree"),
        ("plan_equivalence", "plan and Source-D evidence disagree"),
    ),
)
def test_portable_source_d_evidence_rejects_resealed_role_and_receipt_attacks(
    attack: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    task, plan, _source_d = _source_d_evidence_fixture(module, monkeypatch)
    clean_teacher_task_id = "3" * 32

    if attack == "task_parent_clean_teacher":
        task.parent = clean_teacher_task_id
        task.data.parent = clean_teacher_task_id
    elif attack == "task_identity":
        task.id = "5" * 32
    elif attack in {
        "snapshot_clean_teacher",
        "snapshot_entry_point",
        "snapshot_script_sha",
    }:
        document = task.artifacts[module.SOURCE_C_SNAPSHOT_ARTIFACT].value
        if attack == "snapshot_clean_teacher":
            document["source_c_task_id"] = clean_teacher_task_id
        elif attack == "snapshot_entry_point":
            document["script"]["entry_point"] = module.TRAINING_ENTRY_POINT + ".old"
        else:
            script = document["script_diff"] + "# attacker\n"
            script_sha = hashlib.sha256(script.encode("utf-8")).hexdigest()
            document["script_diff"] = script
            document["script"]["sha256"] = script_sha
            document["script"]["size_bytes"] = len(script.encode("utf-8"))
            document["script"]["line_count"] = len(script.splitlines())
        task.artifacts[module.SOURCE_C_SNAPSHOT_ARTIFACT] = _Artifact(
            module._sealed(document)
        )
    elif attack in {
        "provenance_source_c_parent",
        "provenance_output_parent",
        "provenance_builder",
        "portable_marker",
        "portable_marker_count",
        "legacy_anchor_count",
    }:
        receipt = task.artifacts[module.SOURCE_D_RECEIPT_ARTIFACT].value
        if attack == "provenance_source_c_parent":
            receipt["provenance"]["source_c_task_parent"] = clean_teacher_task_id
        elif attack == "provenance_output_parent":
            receipt["provenance"]["output_parent_task_id"] = clean_teacher_task_id
        elif attack == "provenance_builder":
            receipt["provenance"]["builder_source_sha256"] = "0" * 64
        elif attack == "portable_marker":
            receipt["transformation"]["portable_runner_load_marker"] = (
                "_load_source_training_runner"
            )
        elif attack == "portable_marker_count":
            receipt["transformation"]["portable_runner_load_marker_count"] = 2.0
        else:
            receipt["transformation"]["legacy_runner_load_target_anchor_count"] = True
        task.artifacts[module.SOURCE_D_RECEIPT_ARTIFACT] = _Artifact(
            module._sealed(receipt)
        )
    elif attack == "plan_source_d":
        plan["source_d_identity"]["script_sha256"] = "0" * 64
    else:
        plan["source_d_identity"]["equivalence_artifact_sha256"] = "0" * 64

    with pytest.raises(module.FormalMultiseedExecutionError, match=message):
        module._validate_source_d_evidence(
            task,
            task_id="4" * 32,
            plan=plan,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("expected_count", True),
        ("observed_count", 0),
    ),
)
@pytest.mark.parametrize(
    "anchor_name",
    (
        "seal_evidence_security_imports",
        "declare_controlled_evidence_contract",
        "declare_controlled_evidence_receipts",
        "stage_and_verify_controlled_evidence",
        "stage_after_controlled_runner",
        "upload_only_sealed_controlled_evidence",
    ),
)
def test_portable_source_d_evidence_rejects_resealed_staging_receipt_counts(
    field: str,
    bad_value: object,
    anchor_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    task, plan, _source_d = _source_d_evidence_fixture(module, monkeypatch)
    equivalence = task.artifacts[module.SOURCE_D_EQUIVALENCE_ARTIFACT].value
    receipt = next(item for item in equivalence["diff"] if item["name"] == anchor_name)
    receipt[field] = bad_value
    unhashed = copy.deepcopy(equivalence)
    unhashed.pop("artifact_sha256")
    equivalence["artifact_sha256"] = module._content_sha256(unhashed)
    task.artifacts[module.SOURCE_D_EQUIVALENCE_ARTIFACT] = _Artifact(equivalence)

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="replacement receipts drifted",
    ):
        module._validate_source_d_evidence(
            task,
            task_id=task.id,
            plan=plan,
        )


@pytest.mark.parametrize("attack", ("missing", "order"))
def test_portable_source_d_evidence_rejects_resealed_staging_receipt_identity(
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    task, plan, _source_d = _source_d_evidence_fixture(module, monkeypatch)
    equivalence = task.artifacts[module.SOURCE_D_EQUIVALENCE_ARTIFACT].value
    staging = [
        item
        for item in equivalence["diff"]
        if item["name"] in module.SOURCE_D_STAGING_ANCHOR_NAMES
    ]
    if attack == "missing":
        staging[0]["name"] = "drifted_staging_anchor"
    else:
        staging[0]["name"], staging[1]["name"] = (
            staging[1]["name"],
            staging[0]["name"],
        )
    unhashed = copy.deepcopy(equivalence)
    unhashed.pop("artifact_sha256")
    equivalence["artifact_sha256"] = module._content_sha256(unhashed)
    task.artifacts[module.SOURCE_D_EQUIVALENCE_ARTIFACT] = _Artifact(equivalence)

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="staging receipts drifted",
    ):
        module._validate_source_d_evidence(
            task,
            task_id=task.id,
            plan=plan,
        )


@pytest.mark.parametrize("argument_index", (1, 3, 5, 7))
def test_pinned_source_c_id_cannot_alias_any_cli_dependency(
    argument_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PLANNER_PRODUCER_PIN_FINALIZED", True)
    cli = _valid_cli(module)
    cli[argument_index] = module.SOURCE_C_TASK_ID
    args = module._parser().parse_args(cli)
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="dependency task IDs must be globally unique",
    ):
        module._validated_executor_arguments(args)


def _write_evidence_zip(root: Path) -> Path:
    archive = root.with_name(f"{root.name}.zip")
    if archive.exists():
        return archive
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as destination:
        for source in sorted(root.rglob("*")):
            if source.is_file():
                destination.write(
                    source,
                    arcname=source.relative_to(root).as_posix(),
                )
    return archive


class _PublishOutput(_Task):
    def __init__(
        self,
        *,
        existing: dict[str, object] | None = None,
        fail_after_commit: bool = False,
        upload_result: object = True,
        flush_result: object = True,
        on_upload: object | None = None,
        on_flush: object | None = None,
        artifact_on_get: object | None = None,
    ) -> None:
        super().__init__(
            task_id="f" * 32,
            name="manifest publisher",
            parent="",
            script=_script("clearml_formal_multiseed_executor.py", "# executor\n"),
            parameters={},
            status="in_progress",
        )
        self.artifacts = {
            key: _Artifact(value, on_get=artifact_on_get)
            for key, value in (existing or {}).items()
        }
        self.fail_after_commit = fail_after_commit
        self.upload_result = upload_result
        self.flush_result = flush_result
        self.on_upload = on_upload
        self.on_flush = on_flush
        self.artifact_on_get = artifact_on_get
        self.upload_calls = 0
        self.flush_calls = 0

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> object:
        assert wait_on_upload is True
        self.upload_calls += 1
        self.artifacts[name] = _Artifact(
            artifact_object,
            on_get=self.artifact_on_get,
        )
        if callable(self.on_upload):
            self.on_upload()
        if self.fail_after_commit:
            raise RuntimeError("transport lost after commit")
        return self.upload_result

    def flush(self, *, wait_for_uploads: bool) -> object:
        assert wait_for_uploads is True
        self.flush_calls += 1
        if callable(self.on_flush):
            self.on_flush()
        return self.flush_result


class _MainOutput(_Task):
    def __init__(self) -> None:
        super().__init__(
            task_id=EXECUTOR_ID,
            name="executor",
            parent="",
            script=_script("clearml_formal_multiseed_executor.py", "# executor\n"),
            parameters={},
            status="in_progress",
        )
        self.parent_calls: list[str] = []

    def set_parent(self, parent_task_id: str) -> bool:
        self.parent_calls.append(parent_task_id)
        self.parent = parent_task_id
        self.data.parent = parent_task_id
        return True


def _valid_cli(module: object) -> list[str]:
    return [
        "--planner-task-id",
        PARENT_ID,
        "--source-d-task-id",
        "1" * 32,
        "--teacher-quality-gate-task-id",
        "2" * 32,
        "--teacher-task-id",
        "3" * 32,
        "--expected-planner-producer-sha256",
        module.EXPECTED_PLANNER_PRODUCER_SHA256,
        "--project",
        module.DEFAULT_PROJECT,
        "--poll-seconds",
        "1",
        "--timeout-hours",
        "2",
    ]


class _BindingModel:
    def __init__(
        self,
        *,
        model_id: str,
        task_id: str,
        name: str,
        url: str,
    ) -> None:
        self.id = model_id
        self.task = task_id
        self.name = name
        self.url = url
        self.framework = "PyTorch"
        self.published = True


def _batch_raw_task_snapshot(task: _Task) -> object:
    hyperparams: dict[str, dict[str, object]] = {}
    for full_name, value in task.parameters.items():
        section, separator, name = full_name.partition("/")
        assert separator == "/" and section and name
        hyperparams.setdefault(section, {})[name] = SimpleNamespace(
            value=copy.deepcopy(value),
            type="str",
        )
    raw_artifacts = [
        SimpleNamespace(
            key=name,
            type=artifact.type,
            mode=artifact.mode,
            uri=artifact.url,
            content_size=artifact.size,
            hash=artifact.hash,
        )
        for name, artifact in task.artifacts.items()
    ]
    model_ids = [model.id for model in task.output_models]
    assert len(model_ids) <= 1
    return SimpleNamespace(
        id=task.id,
        project=task.data.project,
        parent=task.data.parent,
        name=task.data.name,
        status=task.data.status,
        script=SimpleNamespace(
            repository=task.data.script.repository,
            working_dir=task.data.script.working_dir,
            entry_point=task.data.script.entry_point,
            diff=task.data.script.diff,
        ),
        hyperparams=hyperparams,
        output=SimpleNamespace(
            destination=task.data.output.destination,
            model=model_ids[0] if model_ids else None,
        ),
        execution=SimpleNamespace(
            queue=task.data.execution.queue,
            artifacts=raw_artifacts,
        ),
        last_worker=task.data.last_worker,
    )


def _publication_matrix_fixture(
    module: object,
    *,
    output: _PublishOutput | None = None,
) -> dict[str, object]:
    output = output or _PublishOutput()
    root_ids = {
        "planner": "a" * 32,
        "source_d": "b" * 32,
        "teacher_gate": "c" * 32,
        "teacher": "d" * 32,
    }
    dependencies: dict[str, _Task] = {}
    for index, (name, task_id) in enumerate(root_ids.items()):
        task = _Task(
            task_id=task_id,
            name=f"root {name}",
            parent="",
            script=_script(f"{name}.py", f"# {name}\n"),
            parameters={"Args/root": name},
            status="completed",
        )
        task.artifacts = {
            "contract": _Artifact(
                {"root": name},
                artifact_hash=f"{index + 1:x}" * 64,
                size=index + 1,
                url=(f"http://10.100.34.118:8081/artifacts/root-{name}.json"),
            )
        }
        dependencies[name] = task

    plan: dict[str, object] = {
        "training_tasks": [],
        "evaluation_tasks": [],
    }
    manifest: dict[str, object] = {
        "training_tasks": [],
        "evaluation_tasks": [],
    }
    tasks_by_id: dict[str, _Task] = {}
    tasks_by_name: dict[str, _Task] = {}
    for index in range(45):
        training_id = f"{1000 + index:032x}"
        evaluation_id = f"{2000 + index:032x}"
        model_id = f"{3000 + index:032x}"
        training_key = f"train-{index:02d}"
        evaluation_key = f"eval-{index:02d}"
        subject = f"method_{index:02d}"
        training_record = {
            "task_key": training_key,
            "worker_queue": "GPU4-A100",
        }
        evaluation_record = {
            "task_key": evaluation_key,
            "training_task_key": training_key,
            "worker_queue": "GPU4-A100",
        }
        plan["training_tasks"].append(training_record)
        plan["evaluation_tasks"].append(evaluation_record)
        model_url = f"http://10.100.34.118:8081/models/{subject}-epoch-50.pth"
        training = _Task(
            task_id=training_id,
            name=module._task_run_name(
                training_record,
                executor_task_id=output.id,
            ),
            parent=output.id,
            script=_script(module.TRAINING_ENTRY_POINT, "# source d\n"),
            parameters={"Args/stage": "all"},
            status="completed",
            queue=module.QUEUE_IDS["GPU4-A100"],
        )
        training.artifacts = {
            "contract": _Artifact(
                {"task": training_key},
                artifact_hash=f"{4000 + index:064x}",
                size=4000 + index,
                url=(f"http://10.100.34.118:8081/artifacts/{training_key}.json"),
            )
        }
        training.output_models = [
            _BindingModel(
                model_id=model_id,
                task_id=training_id,
                name=f"ResilientV2X {subject} final checkpoint",
                url=model_url,
            )
        ]
        evaluation = _Task(
            task_id=evaluation_id,
            name=module._task_run_name(
                evaluation_record,
                executor_task_id=output.id,
            ),
            parent=training_id,
            script=_script(module.TRAINING_ENTRY_POINT, "# source d\n"),
            parameters={"Args/stage": "baseline_validate"},
            status="completed",
            queue=module.QUEUE_IDS["GPU4-A100"],
        )
        evaluation.artifacts = {
            "evidence": _Artifact(
                {"task": evaluation_key},
                artifact_hash=f"{5000 + index:064x}",
                size=5000 + index,
                url=(f"http://10.100.34.118:8081/artifacts/{evaluation_key}.zip"),
                artifact_type="archive",
            )
        }
        tasks_by_id[training_id] = training
        tasks_by_id[evaluation_id] = evaluation
        tasks_by_name[training.name] = training
        tasks_by_name[evaluation.name] = evaluation
        manifest["training_tasks"].append(
            {
                "task_key": training_key,
                "task_id": training_id,
                "subject": subject,
                "result": {
                    "model_id": model_id,
                    "model_url": model_url,
                },
            }
        )
        manifest["evaluation_tasks"].append(
            {
                "task_key": evaluation_key,
                "task_id": evaluation_id,
                "subject": subject,
                "result": {},
            }
        )

    batch_tasks_by_id = {
        output.id: output,
        **{task.id: task for task in dependencies.values()},
        **tasks_by_id,
    }
    batch_calls: list[dict[str, object]] = []

    class TaskApi:
        @staticmethod
        def get_task(*, task_id: str) -> _Task:
            return tasks_by_id[task_id]

        @staticmethod
        def get_tasks(**kwargs: object) -> list[_Task]:
            pattern = str(kwargs["task_name"])
            return [
                task
                for name, task in tasks_by_name.items()
                if re.fullmatch(pattern, name)
            ]

        @staticmethod
        def _query_tasks(**kwargs: object) -> list[object]:
            task_ids = kwargs.get("task_ids")
            only_fields = kwargs.get("only_fields")
            assert type(task_ids) is list
            assert len(task_ids) == 95
            assert kwargs.get("fetch_only_first_page") is True
            assert type(only_fields) is list
            assert set(only_fields) == set(module.PUBLICATION_BATCH_ONLY_FIELDS)
            batch_calls.append(copy.deepcopy(kwargs))
            return [
                _batch_raw_task_snapshot(batch_tasks_by_id[task_id])
                for task_id in reversed(task_ids)
            ]

    source = "# executor\n"
    executor_contract = {
        "task_id": output.id,
        "parent_task_id": "",
        "name": output.data.name,
        "script_source": source,
        "script_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "parameters": {},
        "parameters_sha256": module._content_sha256({}),
        "output_uri": output.output_uri,
        "project_id": output.data.project,
    }
    expected_roots = {
        "fingerprints": {},
        "plan": plan,
        "source_d": {"script": "# source d\n"},
        "teacher": {},
    }
    return {
        "output": output,
        "dependencies": dependencies,
        "dependency_ids": root_ids,
        "manifest": manifest,
        "plan": plan,
        "task_class": TaskApi,
        "tasks_by_id": tasks_by_id,
        "expected_roots": expected_roots,
        "executor_contract": executor_contract,
        "batch_calls": batch_calls,
        "batch_tasks_by_id": batch_tasks_by_id,
    }


def _mutate_publication_fixture(fixture: dict[str, object], target: str) -> None:
    if target == "output":
        fixture["output"].data.parent = "9" * 32
        return
    if target == "root":
        task = fixture["dependencies"]["planner"]
    else:
        phase, position = target.split("_", 1)
        field = "training_tasks" if phase == "train" else "evaluation_tasks"
        receipt = fixture["manifest"][field][0 if position == "first" else -1]
        task = fixture["tasks_by_id"][receipt["task_id"]]
    artifact = next(iter(task.artifacts.values()))
    artifact.hash = "f" * 64 if artifact.hash != "f" * 64 else "e" * 64


def _validate_publication_fixture(
    module: object,
    fixture: dict[str, object],
    mode: str,
) -> dict[str, object]:
    return module._validate_publication_bindings(
        mode,
        task_class=fixture["task_class"],
        output_task=fixture["output"],
        dependencies=fixture["dependencies"],
        dependency_ids=fixture["dependency_ids"],
        manifest=fixture["manifest"],
        plan=fixture["plan"],
        expected_root_snapshot=fixture["expected_roots"],
        executor_contract=fixture["executor_contract"],
        project=module.DEFAULT_PROJECT,
        planner_producer_sha256=module.EXPECTED_PLANNER_PRODUCER_SHA256,
        plan_validator_path=None,
    )


def _run_publication_harness(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing: bool = False,
    attack_stage: str | None = None,
) -> dict[str, object]:
    monkeypatch.setattr(module, "PLANNER_PRODUCER_PIN_FINALIZED", True)
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    binding_state = {"revision": 0}

    def mutate_binding() -> None:
        binding_state["revision"] += 1

    output = _PublishOutput(
        existing=({module.EXECUTION_MANIFEST_ARTIFACT: manifest} if existing else None),
        fail_after_commit=attack_stage == "transport_loss",
        on_upload=(
            mutate_binding if attack_stage in {"upload", "transport_loss"} else None
        ),
        on_flush=mutate_binding if attack_stage == "flush" else None,
    )
    args = module._parser().parse_args(_valid_cli(module))
    output.parent = PARENT_ID
    output.name = module.EXECUTOR_TASK_NAME
    output.data.name = module.EXECUTOR_TASK_NAME
    output.data.script = SimpleNamespace(
        **_script(module.EXECUTOR_ENTRY_POINT, "# executor\n")
    )
    output.parameters = module._validated_executor_arguments(args)

    dependency_ids = {
        "planner": PARENT_ID,
        "source_d": "1" * 32,
        "teacher_gate": "2" * 32,
        "teacher": "3" * 32,
    }
    dependencies = {
        name: _Task(
            task_id=task_id,
            name=f"root {name}",
            parent=(
                module.SOURCE_C_TASK_ID
                if name == "source_d"
                else dependency_ids["teacher"]
                if name == "teacher_gate"
                else ""
            ),
            script=_script(f"{name}.py", f"# {name}\n"),
            parameters={},
            status="completed",
        )
        for name, task_id in dependency_ids.items()
    }

    class TaskApi:
        @staticmethod
        def get_project_id(**_kwargs: object) -> str:
            return "e" * 32

        @staticmethod
        def get_task(*, task_id: str) -> _Task:
            return next(task for task in dependencies.values() if task.id == task_id)

    plan = _v2_plan(
        module,
        extra={"training_tasks": [], "evaluation_tasks": []},
    )
    source_d = {"script": "# source d\n"}
    teacher = {"checkpoint": "teacher.pth"}
    modes: list[str] = []

    monkeypatch.setattr(module, "_runtime_source", lambda: "# executor\n")
    monkeypatch.setattr(
        module,
        "_wait_for_completed",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "_resolve_plan",
        lambda *_args, **_kwargs: (plan, {"receipt": "planner"}),
    )
    monkeypatch.setattr(
        module,
        "_validate_source_d_evidence",
        lambda *_args, **_kwargs: source_d,
    )
    monkeypatch.setattr(
        module,
        "_validate_teacher_gate",
        lambda *_args, **_kwargs: teacher,
    )
    monkeypatch.setattr(
        module,
        "_task_fingerprint",
        lambda *_args, **kwargs: {"task_id": kwargs["expected_id"]},
    )
    monkeypatch.setattr(
        module,
        "_execute_matrix",
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        module,
        "_manifest_payload",
        lambda **_kwargs: copy.deepcopy(manifest),
    )
    monkeypatch.setattr(
        module,
        "_validate_manifest_shape",
        lambda value, **_kwargs: copy.deepcopy(value),
    )

    def validate_bindings(mode: str, **kwargs: object) -> dict[str, int]:
        assert kwargs["output_task"] is output
        observed_dependencies = kwargs["dependencies"]
        assert set(observed_dependencies) == set(dependencies)
        assert all(
            observed_dependencies[name] is task for name, task in dependencies.items()
        )
        assert kwargs["executor_contract"]["output_uri"] == output.output_uri
        modes.append(mode)
        return copy.deepcopy(binding_state)

    monkeypatch.setattr(
        module,
        "_validate_publication_bindings",
        validate_bindings,
    )
    return {
        "args": args,
        "task_class": TaskApi,
        "output": output,
        "manifest": manifest,
        "modes": modes,
    }


def test_duplicate_exact_task_names_fail_without_clone() -> None:
    module = _load_module()
    name = module._task_run_name(_record(), executor_task_id=EXECUTOR_ID)
    tasks = [
        _Task(
            task_id=character * 32,
            name=name,
            parent=PARENT_ID,
            script=_script(module.TRAINING_ENTRY_POINT, "# source d\n"),
            parameters={},
        )
        for character in ("6", "7")
    ]
    backend = _Backend(tasks)
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="duplicate ClearML tasks",
    ):
        module._clone_or_resume_created(
            backend,
            teacher_task=object(),
            teacher_script=_script(module.TRAINING_ENTRY_POINT, "# teacher\n"),
            teacher_parameters={},
            source_d="# source d\n",
            record=_record(),
            parent_task_id=PARENT_ID,
            executor_task_id=EXECUTOR_ID,
            expected_parameters={},
            project=module.DEFAULT_PROJECT,
            used_ids=set(),
        )
    assert backend.clone_calls == 0


def test_enqueue_uses_immutable_queue_id() -> None:
    module = _load_module()
    task = _Task(
        task_id="a" * 32,
        name="task",
        parent=PARENT_ID,
        script=_script(module.TRAINING_ENTRY_POINT, "# source d\n"),
        parameters={},
    )
    backend = _Backend([task])
    module._enqueue_once(backend, task, queue_name="GPU4-A100")
    assert backend.last_enqueue_queue_id == module.QUEUE_IDS["GPU4-A100"]


def test_enqueue_exception_without_commit_fails_closed() -> None:
    module = _load_module()
    task = _Task(
        task_id="b" * 32,
        name="task",
        parent=PARENT_ID,
        script=_script(module.TRAINING_ENTRY_POINT, "# source d\n"),
        parameters={},
    )

    class NoCommitBackend(_Backend):
        def enqueue(self, *, task: _Task, queue_id: str, force: bool) -> dict[str, int]:
            assert force is False
            self.enqueue_calls += 1
            self.last_enqueue_queue_id = queue_id
            raise RuntimeError("transport failed before commit")

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="failed before an auditable server commit",
    ):
        module._enqueue_once(NoCommitBackend([task]), task, queue_name="GPU4-A100")


def test_enqueue_ack_on_wrong_queue_id_fails_closed() -> None:
    module = _load_module()
    task = _Task(
        task_id="c" * 32,
        name="task",
        parent=PARENT_ID,
        script=_script(module.TRAINING_ENTRY_POINT, "# source d\n"),
        parameters={},
    )

    class WrongQueueBackend(_Backend):
        def enqueue(self, *, task: _Task, queue_id: str, force: bool) -> dict[str, int]:
            assert force is False
            task.status = "queued"
            task.data.execution.queue = "0" * 32
            return {"queued": 1, "updated": 1}

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="enqueue commit cannot be verified",
    ):
        module._enqueue_once(WrongQueueBackend([task]), task, queue_name="GPU4-A100")


def test_queued_task_is_never_enqueued_again() -> None:
    module = _load_module()
    task = _Task(
        task_id="d" * 32,
        name="task",
        parent=PARENT_ID,
        script=_script(module.TRAINING_ENTRY_POINT, "# source d\n"),
        parameters={},
        status="queued",
        queue=module.QUEUE_IDS["GPU4-A100"],
    )
    backend = _Backend([task])
    module._enqueue_once(backend, task, queue_name="GPU4-A100")
    assert backend.enqueue_calls == 0


def test_provisional_planner_pin_blocks_before_task_init_even_with_cli_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    class API:
        init_calls = 0

        @classmethod
        def init(cls, **_kwargs: object) -> object:
            cls.init_calls += 1
            raise AssertionError("provisional pin must block before Task.init")

    cli = _valid_cli(module)
    pin_index = cli.index("--expected-planner-producer-sha256") + 1
    cli[pin_index] = "f" * 64
    monkeypatch.setattr(module, "Task", API)
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="deployment pin is not final",
    ):
        module.main(cli)
    assert API.init_calls == 0

    args = module._parser().parse_args(cli)
    monkeypatch.setattr(module, "PLANNER_PRODUCER_PIN_FINALIZED", True)
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="must match the compiled deployment pin",
    ):
        module._validated_executor_arguments(args)


def test_main_explicit_argv_validates_before_task_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PLANNER_PRODUCER_PIN_FINALIZED", True)

    class API:
        init_calls = 0

        @classmethod
        def init(cls, **_kwargs: object) -> object:
            cls.init_calls += 1
            raise AssertionError("invalid CLI must not create a task")

    cli = _valid_cli(module)
    cli[3] = cli[1]
    monkeypatch.setattr(module, "Task", API)
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="dependency task IDs must be globally unique",
    ):
        module.main(cli)
    assert API.init_calls == 0


def test_main_binds_parent_and_exact_args_before_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PLANNER_PRODUCER_PIN_FINALIZED", True)
    output = _MainOutput()
    init_kwargs: list[dict[str, object]] = []
    run_calls: list[object] = []

    class API:
        @classmethod
        def init(cls, **kwargs: object) -> _MainOutput:
            init_kwargs.append(kwargs)
            return output

    def fake_run(
        args: object,
        *,
        task_class: object,
        output_task: object,
    ) -> dict[str, object]:
        assert task_class is API
        assert output_task is output
        assert output.parent == PARENT_ID
        run_calls.append(args)
        return {"complete": True}

    cli = _valid_cli(module)
    monkeypatch.setattr(module, "Task", API)
    monkeypatch.setattr(module, "run", fake_run)
    assert module.main(cli) == 0
    expected = module._validated_executor_arguments(module._parser().parse_args(cli))
    assert output.parameters == expected
    assert output.parent_calls == [PARENT_ID]
    assert init_kwargs[0]["auto_connect_arg_parser"] is False
    assert len(run_calls) == 1


def test_manifest_publish_is_idempotent_after_transport_loss() -> None:
    module = _load_module()
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    output = _PublishOutput(fail_after_commit=True)

    def validate(_mode: str) -> dict[str, str]:
        return {"binding": "stable"}

    module._publish_manifest(output, manifest, validate_bindings=validate)
    module._publish_manifest(output, manifest, validate_bindings=validate)
    assert output.upload_calls == 1
    assert output.flush_calls == 0


@pytest.mark.parametrize("existing", (False, True))
def test_run_routes_fresh_and_existing_manifests_through_same_binding_callback(
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    module = _load_module()
    harness = _run_publication_harness(module, monkeypatch, existing=existing)

    observed = module.run(
        harness["args"],
        task_class=harness["task_class"],
        output_task=harness["output"],
    )

    assert observed == harness["manifest"]
    assert harness["modes"] == ["deep", "snapshot", "snapshot"]
    assert harness["output"].upload_calls == (0 if existing else 1)
    assert harness["output"].flush_calls == (0 if existing else 1)


@pytest.mark.parametrize("attack_stage", ("upload", "flush", "transport_loss"))
def test_run_fails_closed_on_binding_drift_during_manifest_commit(
    monkeypatch: pytest.MonkeyPatch,
    attack_stage: str,
) -> None:
    module = _load_module()
    harness = _run_publication_harness(
        module,
        monkeypatch,
        attack_stage=attack_stage,
    )

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="bindings drifted",
    ):
        module.run(
            harness["args"],
            task_class=harness["task_class"],
            output_task=harness["output"],
        )

    assert harness["modes"] == ["deep", "snapshot"]
    assert harness["output"].upload_calls == 1
    assert harness["output"].flush_calls == (
        1 if attack_stage in {"upload", "flush"} else 0
    )


def test_publication_metadata_sweep_captures_raw_artifacts_and_models_without_download() -> (
    None
):
    module = _load_module()
    fixture = _publication_matrix_fixture(module)

    snapshot, resolved = module._publication_metadata_sweep(
        fixture["task_class"],
        fixture["output"],
        fixture["dependencies"],
        fixture["dependency_ids"],
        fixture["manifest"],
        plan=fixture["plan"],
        project=module.DEFAULT_PROJECT,
        output_task_id=fixture["output"].id,
    )

    assert len(resolved["training"]) == 45
    assert len(resolved["evaluation"]) == 45
    assert len(snapshot["training"]) == 45
    assert len(snapshot["evaluation"]) == 45
    planner_artifact = snapshot["roots"]["planner"]["artifacts"]["contract"]
    assert planner_artifact == {
        "type": "dict",
        "mode": "output",
        "url": "http://10.100.34.118:8081/artifacts/root-planner.json",
        "size_bytes": 1,
        "sha256": "1" * 64,
    }
    assert snapshot["output"]["project_id"] == "e" * 32
    first_training = snapshot["training"][0]["binding"]
    first_receipt = fixture["manifest"]["training_tasks"][0]
    assert first_training["output_models"] == [
        {
            "model_id": first_receipt["result"]["model_id"],
            "task_id": first_receipt["task_id"],
            "name": (f"ResilientV2X {first_receipt['subject']} final checkpoint"),
            "url": first_receipt["result"]["model_url"],
            "framework": "PyTorch",
            "published": True,
        }
    ]
    all_tasks = [
        fixture["output"],
        *fixture["dependencies"].values(),
        *fixture["tasks_by_id"].values(),
    ]
    assert len(all_tasks) == 95
    assert all(task.reload_calls == 1 for task in all_tasks)
    assert all(
        artifact.force_download_calls == []
        for task in all_tasks
        for artifact in task.artifacts.values()
    )


def test_publication_deep_validation_detects_drift_between_metadata_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)

    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )

    def mutate_during_semantic_validation(*_args: object, **_kwargs: object) -> None:
        _mutate_publication_fixture(fixture, "eval_last")

    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        mutate_during_semantic_validation,
    )

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="drifted during deep validation",
    ):
        _validate_publication_fixture(module, fixture, "deep")


def test_publication_snapshot_validation_detects_drift_between_metadata_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)
    original = module._validate_executor_publication_provenance
    calls = 0

    def mutate_between_sweeps(task: object, contract: object) -> None:
        nonlocal calls
        calls += 1
        original(task, contract)
        if calls == 1:
            _mutate_publication_fixture(fixture, "train_last")

    monkeypatch.setattr(
        module,
        "_validate_executor_publication_provenance",
        mutate_between_sweeps,
    )
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="drifted during snapshot validation",
    ):
        _validate_publication_fixture(module, fixture, "snapshot")


def test_publication_validation_uses_one_complete_authoritative_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)
    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )
    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        lambda *_args, **_kwargs: None,
    )

    snapshot = _validate_publication_fixture(module, fixture, "deep")

    assert snapshot["publication_batch_schema_version"] == 1
    assert len(fixture["batch_calls"]) == 1
    call = fixture["batch_calls"][0]
    assert len(call["task_ids"]) == 95
    assert set(call["task_ids"]) == set(fixture["batch_tasks_by_id"])
    assert call["fetch_only_first_page"] is True
    assert set(call["only_fields"]) == set(module.PUBLICATION_BATCH_ONLY_FIELDS)


def test_publication_provenance_rejects_trusted_host_output_path_drift() -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)
    fixture[
        "output"
    ].data.output.destination = "http://10.100.34.118:8081/tasks/attacker-output"

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="output URI drifted",
    ):
        _validate_publication_fixture(module, fixture, "snapshot")


def test_publication_artifact_mode_accepts_clearml_enum_value() -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)

    class ArtifactModeEnum(str, Enum):
        output = "output"

    planner = fixture["dependencies"]["planner"]
    planner.artifacts["contract"].mode = ArtifactModeEnum.output
    snapshot, _resolved = module._publication_metadata_sweep(
        fixture["task_class"],
        fixture["output"],
        fixture["dependencies"],
        fixture["dependency_ids"],
        fixture["manifest"],
        plan=fixture["plan"],
        project=module.DEFAULT_PROJECT,
        output_task_id=fixture["output"].id,
    )
    assert snapshot["roots"]["planner"]["artifacts"]["contract"]["mode"] == ("output")


def test_publication_provenance_rejects_executor_project_migration() -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)
    fixture["output"].data.project = "9" * 32

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="project drifted",
    ):
        _validate_publication_fixture(module, fixture, "snapshot")


@pytest.mark.parametrize(
    "target",
    (
        "output",
        "root",
        "train_first",
        "train_last",
        "eval_first",
        "eval_last",
    ),
)
@pytest.mark.parametrize("stage", ("upload", "flush"))
def test_manifest_publish_detects_binding_drift_at_upload_or_flush(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    stage: str,
) -> None:
    module = _load_module()
    holder: dict[str, dict[str, object]] = {}

    def mutate() -> None:
        _mutate_publication_fixture(holder["fixture"], target)

    output = _PublishOutput(
        on_upload=mutate if stage == "upload" else None,
        on_flush=mutate if stage == "flush" else None,
    )
    fixture = _publication_matrix_fixture(module, output=output)
    holder["fixture"] = fixture
    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )
    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        lambda *_args, **_kwargs: None,
    )

    def validate(mode: str) -> dict[str, object]:
        return _validate_publication_fixture(module, fixture, mode)

    with pytest.raises(module.FormalMultiseedExecutionError, match="drifted"):
        module._publish_manifest(
            output,
            {"schema_version": 1, "seal_sha256": "a" * 64},
            validate_bindings=validate,
        )
    assert output.upload_calls == 1
    assert output.flush_calls == 1


def test_manifest_publish_existing_path_uses_same_deep_and_snapshot_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    output = _PublishOutput(existing={module.EXECUTION_MANIFEST_ARTIFACT: manifest})
    fixture = _publication_matrix_fixture(module, output=output)
    modes: list[str] = []
    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )
    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        lambda *_args, **_kwargs: None,
    )

    def validate(mode: str) -> dict[str, object]:
        modes.append(mode)
        return _validate_publication_fixture(module, fixture, mode)

    module._publish_manifest(output, manifest, validate_bindings=validate)
    assert modes == ["deep", "snapshot", "snapshot"]
    assert output.upload_calls == 0
    assert output.flush_calls == 0


def test_manifest_publish_existing_path_detects_post_deep_root_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    output = _PublishOutput(existing={module.EXECUTION_MANIFEST_ARTIFACT: manifest})
    fixture = _publication_matrix_fixture(module, output=output)
    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )
    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        lambda *_args, **_kwargs: None,
    )

    def validate(mode: str) -> dict[str, object]:
        snapshot = _validate_publication_fixture(module, fixture, mode)
        if mode == "deep":
            _mutate_publication_fixture(fixture, "root")
        return snapshot

    with pytest.raises(module.FormalMultiseedExecutionError, match="drifted"):
        module._publish_manifest(output, manifest, validate_bindings=validate)
    assert output.upload_calls == 0
    assert output.flush_calls == 0


def test_manifest_publish_transport_loss_validates_committed_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    holder: dict[str, dict[str, object]] = {}

    def mutate() -> None:
        _mutate_publication_fixture(holder["fixture"], "train_first")

    output = _PublishOutput(fail_after_commit=True, on_upload=mutate)
    fixture = _publication_matrix_fixture(module, output=output)
    holder["fixture"] = fixture
    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )
    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        lambda *_args, **_kwargs: None,
    )

    def validate(mode: str) -> dict[str, object]:
        return _validate_publication_fixture(module, fixture, mode)

    with pytest.raises(module.FormalMultiseedExecutionError, match="drifted"):
        module._publish_manifest(
            output,
            {"schema_version": 1, "seal_sha256": "a" * 64},
            validate_bindings=validate,
        )
    assert output.upload_calls == 1
    assert output.flush_calls == 0


@pytest.mark.parametrize("existing", (False, True), ids=("fresh", "existing"))
@pytest.mark.parametrize(
    "target",
    (
        "output",
        "root",
        "train_first",
        "train_last",
        "eval_first",
        "eval_last",
    ),
)
def test_manifest_terminal_readback_is_followed_by_full_binding_sweep(
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    target: str,
) -> None:
    module = _load_module()
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    holder: dict[str, object] = {}
    terminal_read = 4 if existing else 2

    def mutate_after_terminal_read(artifact: _Artifact) -> None:
        if len(artifact.force_download_calls) == terminal_read and not holder.get(
            "mutated", False
        ):
            holder["mutated"] = True
            _mutate_publication_fixture(holder["fixture"], target)

    output = _PublishOutput(
        existing=({module.EXECUTION_MANIFEST_ARTIFACT: manifest} if existing else None),
        artifact_on_get=mutate_after_terminal_read,
    )
    fixture = _publication_matrix_fixture(module, output=output)
    holder["fixture"] = fixture
    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )
    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        lambda *_args, **_kwargs: None,
    )
    modes: list[str] = []

    def validate(mode: str) -> dict[str, object]:
        modes.append(mode)
        return _validate_publication_fixture(module, fixture, mode)

    with pytest.raises(module.FormalMultiseedExecutionError, match="drifted"):
        module._publish_manifest(output, manifest, validate_bindings=validate)
    assert holder["mutated"] is True
    assert modes == ["deep", "snapshot", "snapshot"]
    assert output.upload_calls == (0 if existing else 1)


@pytest.mark.parametrize("existing", (False, True), ids=("fresh", "existing"))
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mode", "input"),
        ("type", "archive"),
        ("url", "http://attacker.invalid/manifest.json"),
        ("hash", "bad"),
        ("size", -1),
    ),
)
def test_manifest_terminal_readback_rejects_descriptor_drift(
    existing: bool,
    field: str,
    value: object,
) -> None:
    module = _load_module()
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    terminal_read = 4 if existing else 2
    mutated = False

    def mutate_descriptor(artifact: _Artifact) -> None:
        nonlocal mutated
        if len(artifact.force_download_calls) == terminal_read and not mutated:
            mutated = True
            setattr(artifact, field, value)

    output = _PublishOutput(
        existing=({module.EXECUTION_MANIFEST_ARTIFACT: manifest} if existing else None),
        artifact_on_get=mutate_descriptor,
    )

    with pytest.raises(module.FormalMultiseedExecutionError):
        module._publish_manifest(
            output,
            manifest,
            validate_bindings=lambda _mode: {"binding": "stable"},
        )
    assert mutated is True


@pytest.mark.parametrize("existing", (False, True), ids=("fresh", "existing"))
@pytest.mark.parametrize(
    "target",
    (
        "output",
        "root",
        "train_first",
        "train_last",
        "eval_first",
        "eval_last",
    ),
)
def test_authoritative_batch_rejects_drift_after_terminal_get_models(
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    target: str,
) -> None:
    module = _load_module()
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    output = _PublishOutput(
        existing=({module.EXECUTION_MANIFEST_ARTIFACT: manifest} if existing else None)
    )
    fixture = _publication_matrix_fixture(module, output=output)
    last_evaluation_id = fixture["manifest"]["evaluation_tasks"][-1]["task_id"]
    last_evaluation = fixture["tasks_by_id"][last_evaluation_id]
    mutated = False

    def mutate_after_models_return(_task: _Task) -> None:
        nonlocal mutated
        if last_evaluation.get_models_calls == 6 and not mutated:
            mutated = True
            _mutate_publication_fixture(fixture, target)

    last_evaluation.on_get_models = mutate_after_models_return
    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )
    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        lambda *_args, **_kwargs: None,
    )
    modes: list[str] = []

    def validate(mode: str) -> dict[str, object]:
        modes.append(mode)
        return _validate_publication_fixture(module, fixture, mode)

    with pytest.raises(module.FormalMultiseedExecutionError, match="drifted"):
        module._publish_manifest(output, manifest, validate_bindings=validate)
    assert mutated is True
    assert modes == ["deep", "snapshot", "snapshot"]
    assert len(fixture["batch_calls"]) == 3
    assert all(
        set(call["task_ids"]) == set(fixture["batch_tasks_by_id"])
        for call in fixture["batch_calls"]
    )


@pytest.mark.parametrize("existing", (False, True), ids=("fresh", "existing"))
def test_authoritative_batch_anchors_raw_output_model_ids_after_terminal_get_models(
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    module = _load_module()
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    output = _PublishOutput(
        existing=({module.EXECUTION_MANIFEST_ARTIFACT: manifest} if existing else None)
    )
    fixture = _publication_matrix_fixture(module, output=output)
    first_training_id = fixture["manifest"]["training_tasks"][0]["task_id"]
    first_training = fixture["tasks_by_id"][first_training_id]
    last_evaluation_id = fixture["manifest"]["evaluation_tasks"][-1]["task_id"]
    last_evaluation = fixture["tasks_by_id"][last_evaluation_id]
    mutated = False

    def mutate_model_id_after_models_return(_task: _Task) -> None:
        nonlocal mutated
        if last_evaluation.get_models_calls == 6 and not mutated:
            mutated = True
            first_training.output_models[0].id = "f" * 32

    last_evaluation.on_get_models = mutate_model_id_after_models_return
    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )
    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        lambda *_args, **_kwargs: None,
    )
    modes: list[str] = []

    def validate(mode: str) -> dict[str, object]:
        modes.append(mode)
        return _validate_publication_fixture(module, fixture, mode)

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="batch output model IDs drifted",
    ):
        module._publish_manifest(output, manifest, validate_bindings=validate)
    assert mutated is True
    assert modes == ["deep", "snapshot", "snapshot"]
    assert len(fixture["batch_calls"]) == 3


@pytest.mark.parametrize("existing", (False, True), ids=("fresh", "existing"))
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mode", "input"),
        ("type", "archive"),
        ("url", "http://attacker.invalid/manifest.json"),
        ("hash", "bad"),
        ("size", -1),
    ),
)
def test_authoritative_batch_rejects_manifest_descriptor_drift_after_terminal_get_models(
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    field: str,
    value: object,
) -> None:
    module = _load_module()
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    output = _PublishOutput(
        existing=({module.EXECUTION_MANIFEST_ARTIFACT: manifest} if existing else None)
    )
    fixture = _publication_matrix_fixture(module, output=output)
    last_evaluation_id = fixture["manifest"]["evaluation_tasks"][-1]["task_id"]
    last_evaluation = fixture["tasks_by_id"][last_evaluation_id]
    mutated = False

    def mutate_descriptor_after_models_return(_task: _Task) -> None:
        nonlocal mutated
        if last_evaluation.get_models_calls == 6 and not mutated:
            mutated = True
            artifact = output.artifacts[module.EXECUTION_MANIFEST_ARTIFACT]
            setattr(artifact, field, value)

    last_evaluation.on_get_models = mutate_descriptor_after_models_return
    monkeypatch.setattr(
        module,
        "_dependency_snapshot",
        lambda *_args, **_kwargs: fixture["expected_roots"],
    )
    monkeypatch.setattr(
        module,
        "_revalidate_manifest_tasks",
        lambda *_args, **_kwargs: None,
    )
    modes: list[str] = []

    def validate(mode: str) -> dict[str, object]:
        modes.append(mode)
        return _validate_publication_fixture(module, fixture, mode)

    with pytest.raises(module.FormalMultiseedExecutionError):
        module._publish_manifest(output, manifest, validate_bindings=validate)
    assert mutated is True
    assert modes == ["deep", "snapshot", "snapshot"]
    assert len(fixture["batch_calls"]) == 3


def test_manifest_publish_final_readback_detects_manifest_mutation() -> None:
    module = _load_module()
    manifest = {"schema_version": 1, "seal_sha256": "a" * 64}
    output = _PublishOutput()

    def validate(mode: str) -> dict[str, str]:
        if mode == "snapshot":
            output.artifacts[module.EXECUTION_MANIFEST_ARTIFACT].value[
                "seal_sha256"
            ] = "b" * 64
        return {"binding": "stable"}

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="server readback drifted",
    ):
        module._publish_manifest(output, manifest, validate_bindings=validate)


def test_manifest_publish_requires_keyword_only_binding_callback() -> None:
    module = _load_module()
    with pytest.raises(TypeError, match="validate_bindings"):
        module._publish_manifest(
            _PublishOutput(),
            {"schema_version": 1, "seal_sha256": "a" * 64},
        )


@pytest.mark.parametrize(
    "value",
    (None, [], True, {"bad": object()}),
)
def test_manifest_publish_rejects_nonmapping_binding_callback(value: object) -> None:
    module = _load_module()
    output = _PublishOutput()

    def validate(_mode: str) -> object:
        return value

    with pytest.raises(module.FormalMultiseedExecutionError):
        module._publish_manifest(
            output,
            {"schema_version": 1, "seal_sha256": "a" * 64},
            validate_bindings=validate,
        )
    assert output.upload_calls == 0


def test_manifest_publish_rejects_mapping_subclass_binding_callback() -> None:
    module = _load_module()
    output = _PublishOutput()

    class DictSubclass(dict):
        pass

    def validate(_mode: str) -> object:
        return DictSubclass(binding="stable")

    with pytest.raises(module.FormalMultiseedExecutionError):
        module._publish_manifest(
            output,
            {"schema_version": 1, "seal_sha256": "a" * 64},
            validate_bindings=validate,
        )
    assert output.upload_calls == 0


def test_publication_sweep_rejects_output_root_object_alias() -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)
    fixture["dependencies"]["planner"] = fixture["output"]

    with pytest.raises(module.FormalMultiseedExecutionError, match="alias"):
        module._publication_metadata_sweep(
            fixture["task_class"],
            fixture["output"],
            fixture["dependencies"],
            fixture["dependency_ids"],
            fixture["manifest"],
            plan=fixture["plan"],
            project=module.DEFAULT_PROJECT,
            output_task_id=fixture["output"].id,
        )


def test_publication_sweep_rejects_execution_task_id_alias() -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)
    fixture["manifest"]["evaluation_tasks"][-1]["task_id"] = fixture["manifest"][
        "training_tasks"
    ][0]["task_id"]

    with pytest.raises(module.FormalMultiseedExecutionError, match="alias"):
        module._publication_metadata_sweep(
            fixture["task_class"],
            fixture["output"],
            fixture["dependencies"],
            fixture["dependency_ids"],
            fixture["manifest"],
            plan=fixture["plan"],
            project=module.DEFAULT_PROJECT,
            output_task_id=fixture["output"].id,
        )


def test_publication_sweep_rejects_execution_task_object_alias() -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)
    original_api = fixture["task_class"]
    first_id = fixture["manifest"]["training_tasks"][0]["task_id"]
    last_id = fixture["manifest"]["evaluation_tasks"][-1]["task_id"]
    shared = fixture["tasks_by_id"][first_id]

    class AliasApi:
        get_tasks = staticmethod(original_api.get_tasks)

        @staticmethod
        def get_task(*, task_id: str) -> _Task:
            if task_id == last_id:
                shared.id = last_id
                shared.data.id = last_id
                return shared
            if task_id == first_id:
                shared.id = first_id
                shared.data.id = first_id
            return original_api.get_task(task_id=task_id)

    with pytest.raises(module.FormalMultiseedExecutionError, match="objects.*alias"):
        module._publication_metadata_sweep(
            AliasApi,
            fixture["output"],
            fixture["dependencies"],
            fixture["dependency_ids"],
            fixture["manifest"],
            plan=fixture["plan"],
            project=module.DEFAULT_PROJECT,
            output_task_id=fixture["output"].id,
        )


def test_publication_sweep_rejects_training_model_binding_drift() -> None:
    module = _load_module()
    fixture = _publication_matrix_fixture(module)
    first_id = fixture["manifest"]["training_tasks"][0]["task_id"]
    fixture["tasks_by_id"][first_id].output_models[
        0
    ].url = "http://10.100.34.118:8081/models/attacker.pth"

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="model binding drifted",
    ):
        module._publication_metadata_sweep(
            fixture["task_class"],
            fixture["output"],
            fixture["dependencies"],
            fixture["dependency_ids"],
            fixture["manifest"],
            plan=fixture["plan"],
            project=module.DEFAULT_PROJECT,
            output_task_id=fixture["output"].id,
        )


def test_manifest_publish_rejects_existing_drift_without_write() -> None:
    module = _load_module()
    output = _PublishOutput(
        existing={
            module.EXECUTION_MANIFEST_ARTIFACT: {
                "schema_version": 1,
                "seal_sha256": "a" * 64,
            }
        }
    )
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="server readback drifted",
    ):
        module._publish_manifest(
            output,
            {"schema_version": 1, "seal_sha256": "b" * 64},
            validate_bindings=lambda _mode: {"binding": "stable"},
        )
    assert output.upload_calls == 0
    assert output.flush_calls == 0


@pytest.mark.parametrize("upload_result", (1, "true", object()))
def test_manifest_publish_requires_exact_true_upload_result(
    upload_result: object,
) -> None:
    module = _load_module()
    output = _PublishOutput(upload_result=upload_result)

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="failed to upload",
    ):
        module._publish_manifest(
            output,
            {"schema_version": 1, "seal_sha256": "a" * 64},
            validate_bindings=lambda _mode: {"binding": "stable"},
        )

    assert output.upload_calls == 1
    assert output.flush_calls == 0


@pytest.mark.parametrize("flush_result", (False, 0, 1, "true", object()))
def test_manifest_publish_requires_none_or_exact_true_flush_result(
    flush_result: object,
) -> None:
    module = _load_module()
    output = _PublishOutput(flush_result=flush_result)

    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="failed to flush",
    ):
        module._publish_manifest(
            output,
            {"schema_version": 1, "seal_sha256": "a" * 64},
            validate_bindings=lambda _mode: {"binding": "stable"},
        )

    assert output.upload_calls == 1
    assert output.flush_calls == 1


def test_manifest_publish_accepts_none_flush_result() -> None:
    module = _load_module()
    output = _PublishOutput(flush_result=None)

    module._publish_manifest(
        output,
        {"schema_version": 1, "seal_sha256": "a" * 64},
        validate_bindings=lambda _mode: {"binding": "stable"},
    )

    assert output.upload_calls == 1
    assert output.flush_calls == 1


def test_type_confusion_and_nonfinite_values_fail_closed() -> None:
    module = _load_module()

    class DictSubclass(dict):
        pass

    callbacks = (
        lambda: module._clearml_id("A" * 32, "task"),
        lambda: module._positive_int(True, context="count"),
        lambda: module._finite_ap(float("nan"), context="AP"),
        lambda: module._validate_json_domain((1, 2), context="tuple"),
        lambda: module._validate_json_domain(
            DictSubclass({"x": 1}), context="mapping subclass"
        ),
    )
    for callback in callbacks:
        with pytest.raises(module.FormalMultiseedExecutionError):
            callback()


def test_worker_id_is_opaque_and_queue_drift_fails_closed() -> None:
    module = _load_module()
    task = _Task(
        task_id="f" * 32,
        name="task",
        parent=PARENT_ID,
        script=_script(module.TRAINING_ENTRY_POINT, "# source\n"),
        parameters={"Args/stage": "all"},
        status="queued",
        queue=module.QUEUE_IDS["GPU4-A100"],
    )
    task.data.last_worker = "clearml-agent-17"
    assert module._require_worker(task, "V100", context="task") == "clearml-agent-17"
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="GPU model is invalid",
    ):
        module._require_worker(task, "H100", context="task")
    with pytest.raises(
        module.FormalMultiseedExecutionError,
        match="queue drifted",
    ):
        module._require_bound_task(
            task,
            expected_id=task.id,
            expected_parent=PARENT_ID,
            expected_name="task",
            expected_parameters={"Args/stage": "all"},
            source_d="# source\n",
            queue_name="GPU4-V100",
            context="task",
        )


@pytest.mark.parametrize("gpu_model", ("A100", "V100", "RTX5090"))
def test_fixed_source_d_runtime_profile_matches_real_producer(
    gpu_model: str,
) -> None:
    module = _load_module()
    assert (
        module._source_d_runtime_profile(gpu_model, context="training")
        == module.SOURCE_D_RUNTIME_PROFILE
        == "rtx5090"
    )


def test_metric_schema_matches_real_evaluator_producer() -> None:
    module = _load_module()
    detection = _load_detection_module()
    box = np.array(
        [0.0, 0.0, -1.0, 4.0, 2.0, 1.6, 0.0],
        dtype=np.float64,
    )
    sample = detection.DetectionSample(
        sample_id="schema-contract",
        predicted_boxes=box.reshape(1, 7),
        predicted_scores=np.array([0.9], dtype=np.float64),
        predicted_labels=np.zeros(1, dtype=np.int64),
        ground_truth_boxes=box.reshape(1, 7),
        ground_truth_labels=np.zeros(1, dtype=np.int64),
    )
    metrics = detection.evaluate_car_ap([sample])
    metrics.update(detection.detection_geometry_diagnostics([sample]))
    metrics["unsupported_sample_count"] = 0.0
    observed = {f"resilient_v2x/{key}" for key in metrics}

    assert observed == set(module.EVALUATION_METRIC_KEYS)
    assert len(metrics) == 16
    assert all(type(value) is float for value in metrics.values())


def test_evaluation_producer_pins_match_real_files() -> None:
    module = _load_module()
    evaluator = ROOT / "tools/resilient_v2x/evaluate_controlled_baselines.py"
    overlay = (
        ROOT / "artifacts/resilient_v2x/dair_v2_complemented/evaluation_overlays.json"
    )
    assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == (
        module.CONTROLLED_EVALUATOR_SHA256
    )
    assert hashlib.sha256(overlay.read_bytes()).hexdigest() == (
        module.OVERLAY_INDEX_FILE_SHA256
    )
    document = json.loads(overlay.read_text(encoding="utf-8"))
    assert [entry["overlay"]["path"] for entry in document["transport_overlays"]] == [
        f"val_transport_delay_{delay:03d}.jsonl.zst" for delay in module.DELAYS_MS
    ]
    assert [
        entry["overlay"]["path"]
        for entry in document["fault_overlays"]
        if entry["agent_scope"] == "E+R"
        and entry["duration"] == 1
        and entry["condition"] in {"L-Fail", "C-Fail"}
    ] == [
        (
            f"val_causal_delay_{delay:03d}_"
            f"{condition.lower().replace('-', '_')}.jsonl.zst"
        )
        for delay in module.DELAYS_MS
        for condition in ("L-Fail", "C-Fail")
    ]

    selected = {
        "resilient_v2x",
        "support_residual",
        "linear_no_distillation",
        "no_distillation_peak_lr_3e4",
    }
    config_paths = {
        subject: ROOT / "configs/resilient_v2x/baselines" / f"{subject}.py"
        for subject in set(module.SUBJECT_CONFIG_SHA256) - selected
    }
    config_paths.update(
        {
            "resilient_v2x": ROOT / "configs/resilient_v2x/dair_resilient_v2x.py",
            **{
                subject: ROOT / "configs/resilient_v2x/improvements" / f"{subject}.py"
                for subject in selected - {"resilient_v2x"}
            },
        }
    )
    assert {
        subject: hashlib.sha256(path.read_bytes()).hexdigest()
        for subject, path in config_paths.items()
    } == module.SUBJECT_CONFIG_SHA256

    transport_entries = {
        entry["delay_ms"]: entry["overlay"] for entry in document["transport_overlays"]
    }
    fault_entries = {
        (entry["delay_ms"], entry["condition"]): entry["overlay"]
        for entry in document["fault_overlays"]
        if entry["agent_scope"] == "E+R"
        and entry["duration"] == 1
        and entry["condition"] in {"L-Fail", "C-Fail"}
    }
    assert {
        delay: entry["compressed_sha256"] for delay, entry in transport_entries.items()
    } == module.TRANSPORT_OVERLAY_SHA256
    assert {
        identity: entry["compressed_sha256"]
        for identity, entry in fault_entries.items()
    } == module.FAULT_OVERLAY_SHA256
    assert {
        delay: hashlib.sha256((overlay.parent / entry["path"]).read_bytes()).hexdigest()
        for delay, entry in transport_entries.items()
    } == module.TRANSPORT_OVERLAY_SHA256
    assert {
        identity: hashlib.sha256(
            (overlay.parent / entry["path"]).read_bytes()
        ).hexdigest()
        for identity, entry in fault_entries.items()
    } == module.FAULT_OVERLAY_SHA256


def _controlled_prediction_diagnostic(
    *,
    sample_id: str = "sample-1",
    method: str = "ffnet",
) -> dict[str, object]:
    branch_order = ["lidar_ego", "lidar_rsu", "camera_ego", "camera_rsu"]
    return {
        "diagnostic_type": "controlled_baseline",
        "schema_version": 1,
        "sample_id": sample_id,
        "method": method,
        "overall_supported": True,
        "branch_order": branch_order,
        "support": {branch: True for branch in branch_order},
        "age_intervals": {branch: 0.0 for branch in branch_order},
    }


def _resilient_prediction_diagnostic(
    *,
    method: str = (
        "ResilientV2X(ptf=nonlinear,routing=dynamic,reliability=1,delay_metadata=1)"
    ),
) -> dict[str, object]:
    branch_order = (
        ("ego", "lidar"),
        ("rsu", "lidar"),
        ("ego", "camera"),
        ("rsu", "camera"),
    )
    branches = [
        {
            "agent": agent,
            "modality": modality,
            "supported": True,
            "source_tick": 3,
            "source_tau_ms": 300,
            "horizon": 0,
            "observed": True,
            "propagated": False,
            "gamma": 1.0,
            "reliability": 1.0,
            "ptf_queried": True,
            "reason": None,
        }
        for agent, modality in branch_order
    ]
    return {
        "method": method,
        "overall_supported": True,
        "routing": {
            "expert_support": [True, True, True],
            "weights": [0.25, 0.25, 0.5],
            "not_applicable_reason": None,
        },
        "branches": branches,
    }


@pytest.mark.parametrize(
    "method",
    (
        "ResilientV2X(ptf=nonlinear,routing=dynamic,reliability=1,delay_metadata=1)",
        "ResilientV2X(ptf=linear,routing=dynamic,reliability=1,delay_metadata=1)",
    ),
)
def test_prediction_diagnostic_accepts_real_producer_schemas(method: str) -> None:
    module = _load_module()
    module._validate_prediction_diagnostic(
        _controlled_prediction_diagnostic(),
        sample_id="sample-1",
        controlled_baseline=True,
        expected_method="ffnet",
        context="controlled",
    )
    module._validate_prediction_diagnostic(
        _resilient_prediction_diagnostic(method=method),
        sample_id="sample-1",
        controlled_baseline=False,
        expected_method=method,
        context="primary",
    )


@pytest.mark.parametrize(
    ("subject", "kind", "expected"),
    (
        ("ffnet", "external_controlled_baseline", "ffnet"),
        (
            "resilient_v2x",
            "selected_candidate",
            "ResilientV2X(ptf=nonlinear,routing=dynamic,reliability=1,delay_metadata=1)",
        ),
        (
            "support_residual",
            "selected_candidate",
            "ResilientV2X(ptf=nonlinear,routing=dynamic,reliability=1,delay_metadata=1)",
        ),
        (
            "linear_no_distillation",
            "selected_candidate",
            "ResilientV2X(ptf=linear,routing=dynamic,reliability=1,delay_metadata=1)",
        ),
        (
            "no_distillation_peak_lr_3e4",
            "selected_candidate",
            "ResilientV2X(ptf=nonlinear,routing=dynamic,reliability=1,delay_metadata=1)",
        ),
    ),
)
def test_prediction_diagnostic_method_is_exactly_derived(
    subject: str,
    kind: str,
    expected: str,
) -> None:
    module = _load_module()
    assert (
        module._expected_prediction_diagnostic_method(
            {"subject": subject, "kind": kind},
            context="method",
        )
        == expected
    )


@pytest.mark.parametrize(
    "row",
    (
        [0, 0, -1, 4, 2, 1, 0],
        [0.0, 0.0, -1.0, 0.0, 2.0, 1.0, 0.0],
        [0.0, 0.0, float("nan"), 4.0, 2.0, 1.0, 0.0],
    ),
)
def test_prediction_boxes_reject_nonproducer_numeric_domains(
    row: list[object],
) -> None:
    module = _load_module()
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._validate_box_rows([row], context="box")


@pytest.mark.parametrize(
    "attack",
    (
        "controlled_method",
        "controlled_sample",
        "controlled_uses_resilient_schema",
        "primary_uses_controlled_schema",
        "primary_empty_support",
        "controlled_integer_age",
        "primary_method",
        "primary_routing_stripped",
        "primary_branch_stripped",
        "primary_integer_reliability",
    ),
)
def test_prediction_diagnostic_rejects_cross_schema_and_identity_attacks(
    attack: str,
) -> None:
    module = _load_module()
    primary_method = (
        "ResilientV2X(ptf=nonlinear,routing=dynamic,reliability=1,delay_metadata=1)"
    )
    controlled = attack.startswith("controlled_")
    if attack == "primary_uses_controlled_schema":
        value = _controlled_prediction_diagnostic()
    elif attack == "controlled_uses_resilient_schema":
        value = _resilient_prediction_diagnostic()
    elif attack == "controlled_method":
        value = _controlled_prediction_diagnostic(method="attacker")
    elif attack == "controlled_sample":
        value = _controlled_prediction_diagnostic(sample_id="attacker")
    elif attack == "controlled_integer_age":
        value = _controlled_prediction_diagnostic()
        value["age_intervals"]["lidar_ego"] = 0
    else:
        value = _resilient_prediction_diagnostic()
        if attack == "primary_method":
            value["method"] = "attacker"
        elif attack == "primary_routing_stripped":
            value["routing"].pop("weights")
        elif attack == "primary_branch_stripped":
            value["branches"][0].pop("source_tick")
        elif attack == "primary_integer_reliability":
            value["branches"][0]["reliability"] = 1
        elif attack == "primary_empty_support":
            for branch in value["branches"]:
                branch["supported"] = False
    expected_method = "ffnet" if controlled else primary_method
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._validate_prediction_diagnostic(
            value,
            sample_id="sample-1",
            controlled_baseline=controlled,
            expected_method=expected_method,
            context="attacked diagnostic",
        )


def _write_evaluation_json(path: Path, value: object) -> str:
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _build_evaluation_evidence(
    module: object,
    root: Path,
    *,
    subject: str = "ffnet",
    kind: str = "external_controlled_baseline",
) -> Path:
    overlay = json.loads(
        (
            ROOT
            / "artifacts/resilient_v2x/dair_v2_complemented/evaluation_overlays.json"
        ).read_text(encoding="utf-8")
    )
    sample_ids = overlay["sample_ids"]
    assert len(sample_ids) == 1337
    assert module._producer_content_sha256(sample_ids) == module.SAMPLE_IDS_SHA256

    task_id = "6" * 32
    record = {"subject": subject, "kind": kind}
    controlled = kind == "external_controlled_baseline"
    expected_method = module._expected_prediction_diagnostic_method(
        record,
        context="fixture",
    )
    source_root = "/opt/formal-source"
    dataset_root = (
        f"{source_root}/work_dirs/clearml_dataset_materialization/{task_id}/training"
    )
    work_dir = (
        f"{source_root}/work_dirs/controlled_baseline_evaluation/{task_id}/{subject}"
    )
    overlay_dir = f"{dataset_root}/protocols/dair_v2"
    checkpoint = "/clearml/cache/final.pth"
    checkpoint_sha256 = "9" * 64

    box = [0.0, 0.0, -1.0, 4.0, 2.0, 1.6, 0.0]
    samples = []
    for index, sample_id in enumerate(sample_ids):
        has_prediction = index < 250
        diagnostic = (
            _controlled_prediction_diagnostic(
                sample_id=sample_id,
                method=subject,
            )
            if controlled
            else _resilient_prediction_diagnostic(method=expected_method)
        )
        samples.append(
            {
                "sample_id": sample_id,
                "predicted_boxes_lidar_bottom_center": [box] if has_prediction else [],
                "predicted_scores": [0.9] if has_prediction else [],
                "predicted_labels": [0] if has_prediction else [],
                "ground_truth_boxes_lidar_bottom_center": (
                    [box] * 11330 if index == 0 else []
                ),
                "ground_truth_labels": [0] * 11330 if index == 0 else [],
                "diagnostic": diagnostic,
            }
        )
    prediction_document = {
        "schema_version": 1,
        "document_type": "resilient_v2x_predictions",
        "coordinate_convention": (
            "[x,y,z_bottom,length,width,height,yaw] in current ego LiDAR"
        ),
        "point_cloud_range": [0.0, -40.0, -3.0, 80.0, 40.0, 1.0],
        "iou_thresholds": [0.5, 0.7],
        "max_detections": 100,
        "sample_count": 1337,
        "samples": samples,
    }
    prediction_document["content_sha256"] = module._producer_content_sha256(
        prediction_document
    )
    prediction_raw = (
        json.dumps(
            prediction_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    prediction_sha256 = hashlib.sha256(prediction_raw).hexdigest()

    condition_ids = [
        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        for delay in module.DELAYS_MS
        for condition in module.CONDITIONS
    ]
    plan_runs = []
    metric_runs = []
    root.mkdir(parents=True, exist_ok=True)
    for index, condition_id in enumerate(condition_ids):
        delay = module.DELAYS_MS[index // len(module.CONDITIONS)]
        condition = module.CONDITIONS[index % len(module.CONDITIONS)]
        condition_dir = root / condition_id
        condition_dir.mkdir()
        resolved_config = condition_dir / "resolved_config.py"
        resolved_raw = f"# sealed {condition_id}\n".encode()
        resolved_config.write_bytes(resolved_raw)
        (condition_dir / "checkpoint.sha256").write_text(
            f"{checkpoint_sha256}\n",
            encoding="ascii",
        )
        (condition_dir / "predictions.json").write_bytes(prediction_raw)
        condition_filename = (
            f"global_delay_{delay:03d}_full.py"
            if condition == "Full"
            else (f"causal_delay_{delay:03d}_{condition.lower().replace('-', '_')}.py")
        )
        transport_sha256 = module.TRANSPORT_OVERLAY_SHA256[delay]
        fault_slug = condition.lower().replace("-", "_")
        fault_path = (
            None
            if condition == "Full"
            else (f"{overlay_dir}/val_causal_delay_{delay:03d}_{fault_slug}.jsonl.zst")
        )
        fault_sha256 = (
            None
            if condition == "Full"
            else module.FAULT_OVERLAY_SHA256[(delay, condition)]
        )
        predictions = f"{work_dir}/{condition_id}/predictions.json"
        plan_runs.append(
            {
                "condition_id": condition_id,
                "delay_ms": delay,
                "condition": condition,
                "agent_scope": "E+R",
                "duration_ticks": 1,
                "condition_config": (
                    f"{source_root}/configs/resilient_v2x/conditions/"
                    f"{condition_filename}"
                ),
                "resolved_config": (f"{work_dir}/{condition_id}/resolved_config.py"),
                "resolved_config_sha256": hashlib.sha256(resolved_raw).hexdigest(),
                "predictions": predictions,
                "checkpoint_sha256_file": (
                    f"{work_dir}/{condition_id}/checkpoint.sha256"
                ),
                "transport_overlay": (
                    f"{overlay_dir}/val_transport_delay_{delay:03d}.jsonl.zst"
                ),
                "transport_overlay_sha256": transport_sha256,
                "fault_overlay": fault_path,
                "fault_overlay_sha256": fault_sha256,
            }
        )
        metric_runs.append(
            {
                "condition_id": condition_id,
                "delay_ms": delay,
                "condition": condition,
                "metrics": {
                    "resilient_v2x/sample_count": 1337.0,
                    "resilient_v2x/car_ground_truth_count": 11330.0,
                    "resilient_v2x/car_prediction_count": 250.0,
                    "resilient_v2x/unsupported_sample_count": 0.0,
                    **{key: 50.0 for key in module.AP_METRIC_KEYS},
                    "resilient_v2x/diagnostic_pred_z_bottom_p50": -0.5,
                    "resilient_v2x/diagnostic_gt_z_bottom_p50": -0.4,
                    "resilient_v2x/diagnostic_pred_height_p50": 1.6,
                    "resilient_v2x/diagnostic_gt_height_p50": 1.7,
                    "resilient_v2x/diagnostic_bev_match_050_count": 100.0,
                    "resilient_v2x/diagnostic_bev_match_050_abs_z_error_p50": 0.1,
                    "resilient_v2x/diagnostic_bev_match_050_vertical_iou_p50": 0.8,
                    "resilient_v2x/diagnostic_bev_match_050_3d_iou_p50": 0.6,
                },
                "predictions": predictions,
                "prediction_sha256": prediction_sha256,
                "prediction_content_sha256": prediction_document["content_sha256"],
                "sample_count": 1337,
                "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
                "ground_truth_count": 11330,
                "unsupported_sample_count": 0,
            }
        )
    if controlled:
        subject_type = "baseline"
        subject_config = f"{source_root}/configs/resilient_v2x/baselines/{subject}.py"
    elif subject == "resilient_v2x":
        subject_type = "primary_method"
        subject_config = f"{source_root}/configs/resilient_v2x/dair_resilient_v2x.py"
    else:
        subject_type = "improvement"
        subject_config = (
            f"{source_root}/configs/resilient_v2x/improvements/{subject}.py"
        )
    plan = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_evaluation",
        "protocol_id": module.PROTOCOL_ID,
        "baseline": subject,
        "baseline_config": subject_config,
        "baseline_config_sha256": module.SUBJECT_CONFIG_SHA256[subject],
        "evaluation_subject_type": subject_type,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "data_root": f"{dataset_root}/cooperative-vehicle-infrastructure",
        "split_sha256": module.OFFICIAL_SPLIT_SHA256,
        "manifest": f"{dataset_root}/manifests/temporal_manifest_v2.json",
        "manifest_content_sha256": module.MANIFEST_CONTENT_SHA256,
        "manifest_file_sha256": module.MANIFEST_FILE_SHA256,
        "overlay_index": f"{overlay_dir}/evaluation_overlays.json",
        "overlay_index_content_sha256": module.OVERLAY_INDEX_CONTENT_SHA256,
        "overlay_index_file_sha256": module.OVERLAY_INDEX_FILE_SHA256,
        "sample_ids": sample_ids,
        "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
        "expected_sample_count": 1337,
        "expected_ground_truth_count": 11330,
        "expected_unsupported_sample_count": 0,
        "work_dir": work_dir,
        "delays_ms": list(module.DELAYS_MS),
        "conditions": list(module.CONDITIONS),
        "runs": plan_runs,
        "metrics_output": f"{work_dir}/metrics.json",
        "plan_path": f"{work_dir}/evaluation_plan.json",
    }
    plan["content_sha256"] = module._producer_content_sha256(plan)
    metrics = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": 12,
        "baseline": subject,
        "protocol_id": module.PROTOCOL_ID,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_content_sha256": module.MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": module.OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
        "expected_sample_count": 1337,
        "expected_ground_truth_count": 11330,
        "expected_unsupported_sample_count": 0,
        "runs": metric_runs,
    }
    _write_evaluation_json(root / "evaluation_plan.json", plan)
    _write_evaluation_json(root / "metrics.json", metrics)
    return root


@pytest.fixture(scope="module")
def evaluation_evidence_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    module = _load_module()
    return _build_evaluation_evidence(
        module,
        tmp_path_factory.mktemp("formal-evaluation-evidence"),
    )


def _evaluation_fixture(
    module: object,
    evidence_root: Path,
    *,
    subject: str = "ffnet",
    kind: str = "external_controlled_baseline",
) -> tuple[_Task, dict[str, object], dict[str, object]]:
    task_id = "6" * 32
    training_task_id = "7" * 32
    source_root = "/opt/formal-source"
    dataset_root = (
        f"{source_root}/work_dirs/clearml_dataset_materialization/{task_id}/training"
    )
    work_dir = (
        f"{source_root}/work_dirs/controlled_baseline_evaluation/{task_id}/{subject}"
    )
    evaluator = f"{source_root}/tools/resilient_v2x/evaluate_controlled_baselines.py"
    overlay_index = f"{dataset_root}/protocols/dair_v2/evaluation_overlays.json"
    checkpoint = "/clearml/cache/final.pth"
    condition_ids = [
        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        for delay in module.DELAYS_MS
        for condition in module.CONDITIONS
    ]
    model = {
        "model_id": "8" * 32,
        "model_url": f"{module.FILES_SERVER_URI}/models/final.pth",
        "checkpoint_size_bytes": 123,
        "checkpoint_sha256": "9" * 64,
    }
    run_contract = {
        "schema_version": 1,
        "mode": "baseline_validate",
        "task_id": task_id,
        "baseline": subject,
        "baseline_task_id": training_task_id,
        "predecessor_task_id": training_task_id,
        "training_dataset_id": module.TRAINING_DATASET_ID,
        "checkpoint": {
            "task_id": training_task_id,
            "model_id": model["model_id"],
            "name": f"ResilientV2X {subject} final checkpoint",
            "url": model["model_url"],
            "source_filename": "final.pth",
            "local_filename": "final.pth",
            "size_bytes": model["checkpoint_size_bytes"],
            "sha256": model["checkpoint_sha256"],
            "expected_sha256": model["checkpoint_sha256"],
            "trusted_mmengine_pickle": True,
        },
        "overlay_index": overlay_index,
        "overlay_index_sha256": module.OVERLAY_INDEX_FILE_SHA256,
        "expected_delays_ms": list(module.DELAYS_MS),
        "expected_conditions": list(module.CONDITIONS),
        "expected_run_count": 12,
        "protocol_id": module.PROTOCOL_ID,
        "expected_sample_count": 1337,
        "expected_ground_truth_count": 11330,
        "expected_manifest_content_sha256": module.MANIFEST_CONTENT_SHA256,
        "expected_overlay_index_content_sha256": module.OVERLAY_INDEX_CONTENT_SHA256,
        "expected_sample_ids_sha256": module.SAMPLE_IDS_SHA256,
        "command": [
            module.RUNTIME_PYTHON,
            evaluator,
            "--baseline",
            subject,
            "--checkpoint",
            checkpoint,
            "--overlay-index",
            overlay_index,
            "--work-dir",
            work_dir,
            "--protocol-id",
            module.PROTOCOL_ID,
            "--expected-ground-truth-count",
            "11330",
        ],
        "evaluator": evaluator,
        "evaluator_sha256": module.CONTROLLED_EVALUATOR_SHA256,
        "dataset_root": dataset_root,
    }
    plan = json.loads(
        (evidence_root / "evaluation_plan.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((evidence_root / "metrics.json").read_text(encoding="utf-8"))
    box = [0.0, 0.0, -1.0, 4.0, 2.0, 1.6, 0.0]
    module.GROUND_TRUTH_CONTENT_SHA256 = module._producer_content_sha256(
        [
            {
                "sample_id": sample_id,
                "ground_truth_boxes_lidar_bottom_center": (
                    [box] * 11330 if index == 0 else []
                ),
                "ground_truth_labels": [0] * 11330 if index == 0 else [],
            }
            for index, sample_id in enumerate(plan["sample_ids"])
        ]
    )
    task = _Task(
        task_id=task_id,
        name="evaluation",
        parent=training_task_id,
        script=_script(module.TRAINING_ENTRY_POINT, "# source\n"),
        parameters={},
        status="completed",
    )
    task.artifacts = {
        module.RUN_CONTRACT_ARTIFACT: _Artifact(run_contract),
        module.EVALUATION_PLAN_ARTIFACT: _Artifact(plan),
        module.METRICS_ARTIFACT: _Artifact(metrics),
        module.EVALUATION_EVIDENCE_ARTIFACT: _Artifact(
            None,
            local_copy=_write_evidence_zip(evidence_root),
            artifact_type="archive",
            url=f"{module.FILES_SERVER_URI}/artifacts/{subject}-evidence.zip",
        ),
    }
    record = {
        "task_key": f"eval-r01-s01-{subject}",
        "subject": subject,
        "kind": kind,
        "gpu_model": "A100",
        "condition_ids": condition_ids,
    }
    return task, record, model


def test_evaluation_accepts_exact_1337_11330_zero_and_full_metrics(
    evaluation_evidence_root: Path,
) -> None:
    module = _load_module()
    task, record, model = _evaluation_fixture(module, evaluation_evidence_root)
    input_metrics = task.artifacts[module.METRICS_ARTIFACT].value
    assert set(input_metrics["runs"][0]["metrics"]) == set(
        module.EVALUATION_METRIC_KEYS
    )
    result = module._validate_evaluation_completion(
        task,
        record,
        training_task_id="7" * 32,
        training_model=model,
    )
    assert len(result["runs"]) == 12
    assert set(result["runs"][0]["metrics"]) == set(module.AP_METRIC_KEYS)


def test_fixed_ground_truth_pin_matches_producer_corpus() -> None:
    module = _load_module()
    assert module.GROUND_TRUTH_CONTENT_SHA256 == (
        "44f08b34a76bdd9dcf0f5b179a24eb229f2c73358635ad29b101ac5651675edc"
    )


@pytest.mark.parametrize(
    "subject",
    ("resilient_v2x", "linear_no_distillation"),
)
def test_evaluation_accepts_full_primary_and_linear_evidence(
    subject: str,
    tmp_path: Path,
) -> None:
    module = _load_module()
    evidence = _build_evaluation_evidence(
        module,
        tmp_path / subject,
        subject=subject,
        kind="selected_candidate",
    )
    task, record, model = _evaluation_fixture(
        module,
        evidence,
        subject=subject,
        kind="selected_candidate",
    )
    result = module._validate_evaluation_completion(
        task,
        record,
        training_task_id="7" * 32,
        training_model=model,
    )
    assert len(result["runs"]) == 12
    assert len(result["prediction_bundle_sha256"]) == 64


@pytest.mark.parametrize(
    "attack",
    (
        "artifact_type",
        "url_suffix",
        "path_traversal",
        "symlink_member",
        "unexpected_inventory",
    ),
)
def test_evaluation_evidence_rejects_unsafe_archives(
    attack: str,
    evaluation_evidence_root: Path,
    tmp_path: Path,
) -> None:
    module = _load_module()
    archive = tmp_path / "evidence.zip"
    shutil.copy2(_write_evidence_zip(evaluation_evidence_root), archive)
    artifact_type = "dict" if attack == "artifact_type" else "archive"
    url = (
        f"{module.FILES_SERVER_URI}/artifacts/evidence.tar.gz"
        if attack == "url_suffix"
        else f"{module.FILES_SERVER_URI}/artifacts/evidence.zip"
    )
    if attack not in {"artifact_type", "url_suffix"}:
        if attack == "path_traversal":
            name = "../escape"
            mode = stat.S_IFREG | 0o600
            payload = b"escaped"
        elif attack == "symlink_member":
            name = "delay_000_full/symlink"
            mode = stat.S_IFLNK | 0o777
            payload = b"../escape"
        else:
            name = "attacker.txt"
            mode = stat.S_IFREG | 0o600
            payload = b"unexpected"
        member = zipfile.ZipInfo(name)
        member.create_system = 3
        member.compress_type = zipfile.ZIP_DEFLATED
        member.external_attr = mode << 16
        with zipfile.ZipFile(archive, "a") as destination:
            destination.writestr(member, payload)
    task = _server_artifact_task(
        {
            module.EVALUATION_EVIDENCE_ARTIFACT: _Artifact(
                None,
                local_copy=archive,
                artifact_type=artifact_type,
                url=url,
            )
        }
    )
    with pytest.raises(module.FormalMultiseedExecutionError):
        with module._evaluation_evidence_root(task, context="unsafe archive"):
            pass
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "attack",
    ("entry_count", "compression_ratio", "nul_alias"),
)
def test_evidence_zip_preflight_rejects_metadata_attacks_before_zipfile(
    attack: str,
    evaluation_evidence_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    archive = tmp_path / "preflight.zip"
    if attack == "compression_ratio":
        with zipfile.ZipFile(
            archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as destination:
            for source in sorted(evaluation_evidence_root.rglob("*")):
                if not source.is_file():
                    continue
                relative = source.relative_to(evaluation_evidence_root).as_posix()
                if relative == "evaluation_plan.json":
                    member = zipfile.ZipInfo(relative)
                    member.create_system = 3
                    member.compress_type = zipfile.ZIP_DEFLATED
                    member.external_attr = (stat.S_IFREG | 0o600) << 16
                    destination.writestr(member, b"0" * (2 * 1024 * 1024))
                else:
                    destination.write(source, arcname=relative)
    else:
        shutil.copy2(_write_evidence_zip(evaluation_evidence_root), archive)
        if attack == "entry_count":
            member = zipfile.ZipInfo("attacker.txt")
            member.create_system = 3
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = (stat.S_IFREG | 0o600) << 16
            with zipfile.ZipFile(archive, "a") as destination:
                destination.writestr(member, b"attacker")
        else:
            raw = bytearray(archive.read_bytes())
            target = b"delay_000_full/resolved_config.py"
            prefix = b"evaluation_plan.json\x00"
            replacement = prefix + b"x" * (len(target) - len(prefix))
            position = raw.rfind(target)
            assert position >= 0 and len(replacement) == len(target)
            raw[position : position + len(target)] = replacement
            archive.write_bytes(raw)

    task = _server_artifact_task(
        {
            module.EVALUATION_EVIDENCE_ARTIFACT: _Artifact(
                None,
                local_copy=archive,
                artifact_type="archive",
                url=f"{module.FILES_SERVER_URI}/artifacts/preflight.zip",
            )
        }
    )

    def unexpected_zipfile(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ZipFile materialized before raw metadata rejection")

    monkeypatch.setattr(module.zipfile, "ZipFile", unexpected_zipfile)
    with pytest.raises(module.FormalMultiseedExecutionError):
        with module._evaluation_evidence_root(task, context="ZIP preflight"):
            pass


def test_evidence_archive_open_fstats_the_replaced_inode(
    evaluation_evidence_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    archive = tmp_path / "replaceable.zip"
    shutil.copy2(_write_evidence_zip(evaluation_evidence_root), archive)
    replacement = tmp_path / "oversized.zip"
    with replacement.open("wb") as output:
        output.truncate(module.EVALUATION_EVIDENCE_MAX_ARCHIVE_BYTES + 1)
    task = _server_artifact_task(
        {
            module.EVALUATION_EVIDENCE_ARTIFACT: _Artifact(
                None,
                local_copy=archive,
                artifact_type="archive",
                url=f"{module.FILES_SERVER_URI}/artifacts/replaceable.zip",
            )
        }
    )

    real_open = module.os.open
    swapped = False

    def swap_before_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if dir_fd is None and Path(path) == archive and not swapped:
            module.os.replace(replacement, archive)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "open", swap_before_open)
    with pytest.raises(module.FormalMultiseedExecutionError, match="size/type"):
        with module._evaluation_evidence_root(task, context="replaced archive"):
            pass
    assert swapped


@pytest.mark.parametrize(
    "attack",
    (
        "1789",
        "extra_metric",
        "extra_run_key",
        "missing_metric",
        "nonfinite_diagnostic",
        "fractional_prediction",
        "match_over_prediction",
        "integer_typed_count",
        "out_of_range_iou",
    ),
)
def test_evaluation_rejects_cohort_schema_and_metric_attacks(
    attack: str,
    evaluation_evidence_root: Path,
) -> None:
    module = _load_module()
    task, record, model = _evaluation_fixture(module, evaluation_evidence_root)
    metrics = task.artifacts[module.METRICS_ARTIFACT].value
    run = metrics["runs"][0]
    values = run["metrics"]
    if attack == "1789":
        run["sample_count"] = 1789
    elif attack == "extra_metric":
        values["attacker/metric"] = 1.0
    elif attack == "extra_run_key":
        run["attacker"] = True
    elif attack == "missing_metric":
        values.pop("resilient_v2x/car_prediction_count")
    elif attack == "nonfinite_diagnostic":
        values["resilient_v2x/diagnostic_pred_z_bottom_p50"] = float("inf")
    elif attack == "fractional_prediction":
        values["resilient_v2x/car_prediction_count"] = 1.5
    elif attack == "match_over_prediction":
        values["resilient_v2x/car_prediction_count"] = 1.0
        values["resilient_v2x/diagnostic_bev_match_050_count"] = 2.0
    elif attack == "integer_typed_count":
        values["resilient_v2x/sample_count"] = 1337
    else:
        values["resilient_v2x/diagnostic_bev_match_050_vertical_iou_p50"] = 1.1
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._validate_evaluation_completion(
            task,
            record,
            training_task_id="7" * 32,
            training_model=model,
        )


@pytest.mark.parametrize(
    "attack",
    (
        "extra_artifact",
        "missing_evidence",
        "evaluator_sha256",
        "overlay_file_sha256",
        "command",
        "dataset_root",
        "checkpoint_schema",
        "metrics_checkpoint",
        "plan_overlay_file_sha256",
        "plan_extra_key",
    ),
)
def test_evaluation_rejects_producer_provenance_attacks(
    attack: str,
    evaluation_evidence_root: Path,
) -> None:
    module = _load_module()
    task, record, model = _evaluation_fixture(module, evaluation_evidence_root)
    run_contract = task.artifacts[module.RUN_CONTRACT_ARTIFACT].value
    metrics = task.artifacts[module.METRICS_ARTIFACT].value
    plan = task.artifacts[module.EVALUATION_PLAN_ARTIFACT].value
    if attack == "extra_artifact":
        task.artifacts["attacker"] = _Artifact({"accepted": True})
    elif attack == "missing_evidence":
        task.artifacts.pop(module.EVALUATION_EVIDENCE_ARTIFACT)
    elif attack == "evaluator_sha256":
        run_contract["evaluator_sha256"] = "0" * 64
    elif attack == "overlay_file_sha256":
        run_contract["overlay_index_sha256"] = "0" * 64
    elif attack == "command":
        run_contract["command"][11] = "attacker-protocol"
    elif attack == "dataset_root":
        run_contract["dataset_root"] = "/tmp/attacker-dataset"
    elif attack == "checkpoint_schema":
        run_contract["checkpoint"]["attacker"] = True
    elif attack == "metrics_checkpoint":
        metrics["checkpoint"] = "/tmp/attacker.pth"
    elif attack == "plan_overlay_file_sha256":
        plan["overlay_index_file_sha256"] = "0" * 64
    else:
        plan["attacker"] = True
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._validate_evaluation_completion(
            task,
            record,
            training_task_id="7" * 32,
            training_model=model,
        )


@pytest.mark.parametrize(
    "attack",
    (
        "baseline_config_sha256",
        "transport_overlay_sha256",
        "fault_overlay_sha256",
    ),
)
def test_evaluation_rejects_resealed_config_and_overlay_pin_attacks(
    attack: str,
    evaluation_evidence_root: Path,
    tmp_path: Path,
) -> None:
    module = _load_module()
    evidence = tmp_path / "evidence"
    shutil.copytree(evaluation_evidence_root, evidence)
    plan_path = evidence / "evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if attack == "baseline_config_sha256":
        plan["baseline_config_sha256"] = "0" * 64
    elif attack == "transport_overlay_sha256":
        plan["runs"][0]["transport_overlay_sha256"] = "0" * 64
    else:
        plan["runs"][1]["fault_overlay_sha256"] = "0" * 64
    plan.pop("content_sha256")
    plan["content_sha256"] = module._producer_content_sha256(plan)
    _write_evaluation_json(plan_path, plan)

    task, record, model = _evaluation_fixture(module, evidence)
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._validate_evaluation_completion(
            task,
            record,
            training_task_id="7" * 32,
            training_model=model,
        )


@pytest.mark.parametrize(
    "attack",
    ("file_bytes", "resealed_sample_cohort", "resealed_ground_truth"),
)
def test_evaluation_rejects_prediction_archive_tampering(
    attack: str,
    evaluation_evidence_root: Path,
    tmp_path: Path,
) -> None:
    module = _load_module()
    evidence = tmp_path / "evidence"
    shutil.copytree(evaluation_evidence_root, evidence)
    prediction_path = evidence / "delay_000_full/predictions.json"
    if attack == "file_bytes":
        prediction_path.write_bytes(prediction_path.read_bytes() + b" ")
    else:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        if attack == "resealed_sample_cohort":
            prediction["samples"][0]["sample_id"] = "attacker-sample"
        else:
            prediction["samples"][0]["ground_truth_boxes_lidar_bottom_center"][0][
                0
            ] += 0.25
        prediction.pop("content_sha256")
        prediction["content_sha256"] = module._producer_content_sha256(prediction)
        prediction_file_sha256 = _write_evaluation_json(
            prediction_path,
            prediction,
        )
        metrics_path = evidence / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["runs"][0]["prediction_sha256"] = prediction_file_sha256
        metrics["runs"][0]["prediction_content_sha256"] = prediction["content_sha256"]
        _write_evaluation_json(metrics_path, metrics)

    task, record, model = _evaluation_fixture(module, evidence)
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._validate_evaluation_completion(
            task,
            record,
            training_task_id="7" * 32,
            training_model=model,
        )


def _initialization_audit_fixture(
    module: object,
    *,
    subject: str,
) -> tuple[dict[str, object], dict[str, object]]:
    teacher = {
        "teacher_checkpoint_size_bytes": 123,
        "teacher_checkpoint_sha256": "c" * 64,
    }
    zero_fusion = subject in module.ZERO_FUSION_SUBJECTS
    fusion_keys = 0 if zero_fusion else 1
    fusion_numel = 0 if zero_fusion else 1
    fusion_bytes = 0 if zero_fusion else 4
    fusion_sha = module.EMPTY_TENSOR_MAPPING_SHA256 if zero_fusion else "d" * 64
    nested = subject in module.NESTED_TEACHER_SUBJECTS
    audit = {
        "schema_version": 1,
        "contract": module.COMMON_TEACHER_INITIALIZATION_CONTRACT,
        "result": "pass",
        "checkpoint": {
            "path": "/tmp/teacher.pth",
            "filename": "teacher.pth",
            "size_bytes": teacher["teacher_checkpoint_size_bytes"],
            "sha256": teacher["teacher_checkpoint_sha256"],
            "expected_sha256": teacher["teacher_checkpoint_sha256"],
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
            "state_sha256": "a" * 64,
        },
        "shared_initialization": {
            "prefixes": list(module.COMMON_TEACHER_INITIALIZATION_PREFIXES),
            "keys": module.COMMON_TEACHER_SHARED_KEYS,
            "numel": module.COMMON_TEACHER_SHARED_NUMEL,
            "bytes": module.COMMON_TEACHER_SHARED_BYTES,
            "expected_keys": module.COMMON_TEACHER_SHARED_KEYS,
            "state_sha256": "b" * 64,
            "shape_dtype_verified": True,
            "exact_tensor_equality_verified": True,
        },
        "method_specific_fusion": {
            "keys": fusion_keys,
            "numel": fusion_numel,
            "bytes": fusion_bytes,
            "sha256_before": fusion_sha,
            "sha256_after": fusion_sha,
            "unchanged": True,
        },
        "target": {
            "model_type": "Detector",
            "target_key_count": (
                module.COMMON_TEACHER_SHARED_KEYS
                + fusion_keys
                + (module.COMMON_TEACHER_SOURCE_KEYS if nested else 0)
            ),
            "target_common_key_count": module.COMMON_TEACHER_SHARED_KEYS,
            "target_fusion_key_count": fusion_keys,
            "nested_teacher_present": nested,
            "nested_teacher_key_count": (
                module.COMMON_TEACHER_SOURCE_KEYS if nested else 0
            ),
            "nested_teacher_full_equality_verified": nested,
        },
    }
    return audit, teacher


@pytest.mark.parametrize("subject", ("ffnet", "ego_only", "resilient_v2x"))
def test_initialization_audit_accepts_exact_subject_semantics(
    subject: str,
) -> None:
    module = _load_module()
    audit, teacher = _initialization_audit_fixture(module, subject=subject)
    module._validate_initialization_audit(
        audit,
        subject=subject,
        teacher=teacher,
        context="training",
    )


@pytest.mark.parametrize(
    "attack",
    (
        "extra_key",
        "source_count",
        "shared_equality",
        "fusion_changed",
        "zero_nonzero",
        "target_count",
    ),
)
def test_initialization_audit_rejects_tensor_evidence_attacks(
    attack: str,
) -> None:
    module = _load_module()
    audit, teacher = _initialization_audit_fixture(module, subject="ffnet")
    if attack == "extra_key":
        audit["attacker"] = True
    elif attack == "source_count":
        audit["source"]["keys"] -= 1
    elif attack == "shared_equality":
        audit["shared_initialization"]["exact_tensor_equality_verified"] = False
    elif attack == "fusion_changed":
        audit["method_specific_fusion"]["sha256_after"] = "e" * 64
    elif attack == "zero_nonzero":
        fusion = audit["method_specific_fusion"]
        fusion.update(
            {
                "keys": 0,
                "numel": 0,
                "bytes": 0,
                "sha256_before": module.EMPTY_TENSOR_MAPPING_SHA256,
                "sha256_after": module.EMPTY_TENSOR_MAPPING_SHA256,
            }
        )
    else:
        audit["target"]["target_key_count"] += 1
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._validate_initialization_audit(
            audit,
            subject="ffnet",
            teacher=teacher,
            context="training",
        )


def _teacher_contract_fixture(
    module: object,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    str,
]:
    teacher_task_id = "1" * 32
    reference = {
        "task_id": teacher_task_id,
        "model_id": "2" * 32,
        "model_name": module.CLEAN_TEACHER_MODEL_NAME,
        "model_url": f"{module.FILES_SERVER_URI}/models/teacher-best.pth",
        "checkpoint_filename": (
            "best_resilient_v2x_car_bev_ap_r40_0.70_teacher_epoch_20.pth"
        ),
        "checkpoint_size_bytes": 123,
        "checkpoint_sha256": "a" * 64,
        "selected_epoch": 20,
        "final_checkpoint_filename": "teacher_epoch_50.pth",
        "final_checkpoint_size_bytes": 456,
        "final_checkpoint_sha256": "b" * 64,
    }
    run_contract = {
        "task_id": teacher_task_id,
        "dataset_id": module.TRAINING_DATASET_ID,
        "dataset_local_copy": "/data/dair",
        "gpus": 4,
        "training_world_size": 4,
        "global_batch_size": 8,
        "train_batch_size_per_gpu": 2,
        "eval_batch_size_per_gpu": 4,
        "vehicle_global_batch_size": 8,
        "vehicle_train_batch_size_per_gpu": 2,
        "vehicle_eval_batch_size_per_gpu": 4,
        "learning_rate": 0.0001,
        "auto_scale_lr": False,
        "expected_optimizer_steps": None,
        "max_epochs": 50,
        "val_interval": 10,
        "amp": False,
        "runtime_profile": "rtx5090",
        "python_safe_path": "1",
        "checkpoint_policy": "clean_validation_best_for_teacher_handoff",
        "manifest_content_sha256": module.MANIFEST_CONTENT_SHA256,
        "split_sha256": (
            "d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
        ),
        "evaluation_index_content_sha256": (module.OVERLAY_INDEX_CONTENT_SHA256),
        "evaluation_sample_ids_sha256": module.SAMPLE_IDS_SHA256,
        "evaluation_sample_count": 1337,
        "evaluation_condition_count": 12,
        "output_uri_scheme": "http",
        "stage": "teacher",
        "training_seed": module.TRAINING_OVERLAY_PROTOCOL_SEED,
        "training_overlay_protocol_seed": (module.TRAINING_OVERLAY_PROTOCOL_SEED),
        "seed": module.TRAINING_OVERLAY_PROTOCOL_SEED,
    }
    checkpoint_contract = {
        "schema_version": 1,
        "selection_protocol": "DAIR-CLEAN-PAIR1789-v1",
        "selection_metric": "resilient_v2x/car_bev_ap_r40_0.70",
        "selection_rule": "greater",
        "selected_epoch": reference["selected_epoch"],
        "selected_checkpoint": {
            "model_id": reference["model_id"],
            "name": module.CLEAN_TEACHER_MODEL_NAME,
            "url": reference["model_url"],
            "filename": reference["checkpoint_filename"],
            "size_bytes": reference["checkpoint_size_bytes"],
            "sha256": reference["checkpoint_sha256"],
        },
        "trained_epochs": 50,
        "final_epoch": 50,
        "final_checkpoint": {
            "filename": reference["final_checkpoint_filename"],
            "size_bytes": reference["final_checkpoint_size_bytes"],
            "sha256": reference["final_checkpoint_sha256"],
        },
        "downstream_role": ("frozen teacher and trainable student initialization"),
    }
    return (
        {"run_contract_schema": "seeded"},
        run_contract,
        checkpoint_contract,
        reference,
        teacher_task_id,
    )


def test_teacher_contract_documents_accept_exact_seeded_schema() -> None:
    module = _load_module()
    gate, run_contract, checkpoint, reference, task_id = _teacher_contract_fixture(
        module
    )
    module._validate_teacher_contract_documents(
        run_contract,
        checkpoint,
        gate=gate,
        reference=reference,
        teacher_task_id=task_id,
    )


@pytest.mark.parametrize(
    "attack",
    ("seed", "extra_run_key", "checkpoint_sha", "schema"),
)
def test_teacher_contract_documents_reject_drift(attack: str) -> None:
    module = _load_module()
    gate, run_contract, checkpoint, reference, task_id = _teacher_contract_fixture(
        module
    )
    if attack == "seed":
        run_contract["training_seed"] += 1
    elif attack == "extra_run_key":
        run_contract["attacker"] = True
    elif attack == "checkpoint_sha":
        checkpoint["selected_checkpoint"]["sha256"] = "f" * 64
    else:
        gate["run_contract_schema"] = "legacy_source_c"
    with pytest.raises(module.FormalMultiseedExecutionError):
        module._validate_teacher_contract_documents(
            run_contract,
            checkpoint,
            gate=gate,
            reference=reference,
            teacher_task_id=task_id,
        )
