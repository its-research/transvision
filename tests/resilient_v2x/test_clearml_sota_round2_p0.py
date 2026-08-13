from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as round2


class _Record:
    def __init__(
        self,
        value: dict[str, object],
        *,
        events: list[tuple[object, ...]] | None = None,
    ) -> None:
        self.value = value
        self.events = events

    def to_dict(self) -> dict[str, object]:
        if self.events is not None:
            self.events.append(("artifact_record_readback", self.value.get("key")))
        return dict(self.value)


class _MockTask:
    def __init__(
        self,
        task_id: str,
        name: str,
        status: str,
        *,
        parent: str = "",
        parameters: dict[str, object] | None = None,
        diff: str = "",
        entry_point: str = round2.fastlane.TEMPLATE_ENTRY_POINT,
        events: list[tuple[object, ...]] | None = None,
        artifact_drift: str | None = None,
    ) -> None:
        self.id = task_id
        self.name = name
        self.status = status
        self.parent = parent
        self.project = round2.TRAINING_PROJECT
        self.parameters = dict(parameters or {})
        self.output_uri = ""
        self.events = events if events is not None else []
        self.artifact_drift = artifact_drift
        self.data = SimpleNamespace(
            project=round2.TRAINING_PROJECT_ID,
            script=_Record(
                {
                    "entry_point": entry_point,
                    "diff": diff,
                    "binary": "python",
                    "working_dir": ".",
                }
            ),
            execution=SimpleNamespace(artifacts=[], queue=None),
        )

    def reload(self) -> None:
        self.events.append(("reload", self.id))

    def get_parameters(self, **_kwargs: object) -> dict[str, object]:
        return dict(self.parameters)

    def set_parameters(self, parameters: dict[str, object]) -> None:
        self.parameters = dict(parameters)
        self.events.append(("parameters", self.id))

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: dict[str, object],
        wait_on_upload: bool,
    ) -> bool:
        assert wait_on_upload is True
        self.events.append(("upload", self.id, name))
        serialized = json.dumps(artifact_object, sort_keys=True, indent=4).encode()
        preview = serialized.decode()
        digest = hashlib.sha256(serialized).hexdigest()
        size = len(serialized)
        uri = (
            "http://10.100.34.118:8081/ResilientV2X/Training/"
            f"task.{self.id}/artifacts/{name}/{name}.json"
        )
        if self.artifact_drift == "hash":
            digest = "0" * 64
        elif self.artifact_drift == "size":
            size += 1
        elif self.artifact_drift == "uri":
            uri = f"http://example.invalid/{name}.json"
        elif self.artifact_drift == "preview":
            observed = dict(artifact_object)
            observed["receipt_type"] = "tampered"
            preview = json.dumps(observed, sort_keys=True, indent=4)
        record = {
            "key": name,
            "hash": digest,
            "content_size": size,
            "uri": uri,
            "type_data": {
                "content_type": "application/json",
                "preview": preview,
            },
        }
        self.data.execution.artifacts = [_Record(record, events=self.events)]
        return True

    def flush(self, *, wait_for_uploads: bool) -> None:
        assert wait_for_uploads is True
        self.events.append(("flush", self.id))


class _MockTaskBackend:
    def __init__(
        self,
        template: _MockTask,
        events: list[tuple[object, ...]],
        *,
        duplicate_name: str | None = None,
        post_guard_drift: str | None = None,
        artifact_drift: str | None = None,
        queue_id: str = round2.WORKER_QUEUE_ID,
    ) -> None:
        self.template = template
        self.events = events
        self.duplicate_name = duplicate_name
        self.post_guard_drift = post_guard_drift
        self.artifact_drift = artifact_drift
        self.queue_id = queue_id
        self.tasks = {template.id: template}
        self.clone_count = 0
        self.enqueue_count = 0

    def get_task(self, *, task_id: str) -> _MockTask:
        self.events.append(("get", task_id))
        return self.tasks[task_id]

    def get_tasks(
        self, *, task_name: str, task_filter: dict[str, object]
    ) -> list[object]:
        assert task_filter == {"parent": round2.fastlane.PREDECESSOR_TASK_ID}
        self.events.append(("duplicate_query", task_name))
        matches = [
            task
            for task in self.tasks.values()
            if re.fullmatch(task_name, task.name)
            and task.parent == task_filter["parent"]
        ]
        if matches:
            current = matches[0]
            if self.post_guard_drift == "zero":
                return []
            if self.post_guard_drift == "multiple":
                return [
                    current,
                    SimpleNamespace(
                        id="d" * 32,
                        name=current.name,
                        parent=current.parent,
                    ),
                ]
            if self.post_guard_drift == "wrong_id":
                return [
                    SimpleNamespace(
                        id="d" * 32,
                        name=current.name,
                        parent=current.parent,
                    )
                ]
            return matches
        if self.duplicate_name is not None:
            return [
                SimpleNamespace(
                    id="d" * 32,
                    name=self.duplicate_name,
                    parent=task_filter["parent"],
                )
            ]
        return []

    def clone(
        self,
        *,
        source_task: _MockTask,
        name: str,
        parent: str,
        project: str,
    ) -> _MockTask:
        assert source_task is self.template
        assert parent == round2.fastlane.PREDECESSOR_TASK_ID
        assert project == round2.TRAINING_PROJECT_ID
        self.clone_count += 1
        task_id = f"{self.clone_count:032x}"
        clone = _MockTask(
            task_id,
            name,
            "created",
            parent=parent,
            parameters=source_task.parameters,
            diff=str(source_task.data.script.value["diff"]),
            events=self.events,
            artifact_drift=self.artifact_drift,
        )
        self.tasks[task_id] = clone
        self.events.append(("clone", task_id))
        return clone

    def enqueue(self, *, task: _MockTask, queue_name: str) -> bool:
        assert queue_name == round2.WORKER_QUEUE
        self.enqueue_count += 1
        self.events.append(("enqueue", task.id, queue_name))
        task.status = "in_progress"
        task.data.execution.queue = self.queue_id
        return True


def _raw_bootstrap_fixture(monkeypatch: pytest.MonkeyPatch) -> str:
    raw = """ADDITIONAL_EXPERIMENT_SPECS = (
    ExperimentSpec(
        "linear_no_distillation",
some_other_text = (
        "concat_capacity_matched",
        "resilient_v2x",
)\n"""
    e1_e2_e3 = round2.candidate._apply_candidate_experiment_patch(raw)
    monkeypatch.setattr(
        round2.fastlane,
        "TEMPLATE_SCRIPT_SHA256",
        hashlib.sha256(raw.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        round2.fastlane,
        "PATCHED_TEMPLATE_SCRIPT_SHA256",
        hashlib.sha256(e1_e2_e3.encode()).hexdigest(),
    )
    return raw


def _launch_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duplicate: bool = False,
    post_guard_drift: str | None = None,
    artifact_drift: str | None = None,
    queue_id: str = round2.WORKER_QUEUE_ID,
) -> tuple[SimpleNamespace, _MockTaskBackend, list[tuple[object, ...]]]:
    raw = _raw_bootstrap_fixture(monkeypatch)
    manifest = round2._verify_package(round2.OUTPUT_DIR)
    dataset_id = "a" * 32
    template_id = "b" * 32
    transition = round2._transition(manifest, dataset_id=dataset_id)
    plan = round2._static_plan(manifest)
    expected_template_name = (
        f"{round2.TEMPLATE_PREFIX} [{str(transition['seal_sha256'])[:12]}]"
    )
    expected_task_name = f"{round2.TASK_PREFIX} [{str(plan['seal_sha256'])[:12]}]"
    events: list[tuple[object, ...]] = []
    template = _MockTask(
        template_id,
        expected_template_name,
        "completed",
        parent=round2.fastlane.MAIN_CONTROLLER_TASK_ID,
        parameters=round2._expected_template_parameters(
            manifest, dataset_id=dataset_id
        ),
        diff=raw,
        events=events,
    )
    backend = _MockTaskBackend(
        template,
        events,
        duplicate_name=expected_task_name if duplicate else None,
        post_guard_drift=post_guard_drift,
        artifact_drift=artifact_drift,
        queue_id=queue_id,
    )
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=backend))
    monkeypatch.setattr(
        round2,
        "_downloaded_source_root",
        lambda _manifest, *, dataset_id: round2.OUTPUT_DIR,
    )
    e1_e2_e3 = round2.candidate._apply_candidate_experiment_patch(raw)
    monkeypatch.setattr(
        round2,
        "_validated_fixed_base",
        lambda _task_class: (template, raw, e1_e2_e3),
    )
    monkeypatch.setattr(
        round2, "_require_transition_artifact", lambda _task, _transition: None
    )

    def edit_script(task_id: str, diff: str) -> None:
        backend.tasks[task_id].data.script.value["diff"] = diff
        events.append(("script_edit", task_id))

    monkeypatch.setattr(round2.fastlane, "_edit_script", edit_script)
    args = SimpleNamespace(
        execute_token=round2.LAUNCH_TOKEN,
        queue=round2.WORKER_QUEUE,
        output_dir=round2.OUTPUT_DIR,
        source_dataset_id=dataset_id,
        template_task_id=template_id,
    )
    return args, backend, events


def test_sealed_round2_package_is_exactly_one_file_additive() -> None:
    manifest = round2._verify_package(round2.OUTPUT_DIR)
    base = manifest["base_source_package"]
    target = manifest["target_source_package"]
    delta = manifest["delta"]

    assert manifest["base_source_dataset_id"] == round2.fastlane.SOURCE_DATASET_ID
    assert target["file_count"] == base["file_count"] + 1
    assert (
        target["source_bytes"] == base["source_bytes"] + delta["added_files"][0]["size"]
    )
    assert delta["modified_files"] == []
    assert delta["removed_files"] == []
    assert [item["path"] for item in delta["added_files"]] == [round2.P0_CONFIG]
    round2._require_seal(manifest, context="test round2 package")


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("base_entry", "inventory is not exact base plus P0"),
        ("pseudo_delta", "not an exact one-file additive delta"),
        ("non_object", "inventory is not exact base plus P0"),
    ),
)
def test_add_only_verifier_rejects_inventory_or_delta_forgery(
    monkeypatch: pytest.MonkeyPatch, drift: str, message: str
) -> None:
    original_load = round2._load_json
    manifest_path = round2.OUTPUT_DIR / round2.PACKAGE_MANIFEST_NAME
    inventory_path = round2.OUTPUT_DIR / "source-inventory.json"
    forged_manifest = original_load(manifest_path)
    forged_inventory = original_load(inventory_path)
    if drift == "base_entry":
        forged_inventory["files"][0]["sha256"] = "0" * 64
    elif drift == "non_object":
        forged_inventory["files"][0] = "not-an-object"
    elif drift == "pseudo_delta":
        forged_manifest["delta"]["added_files"][0]["size"] += 1
        forged_manifest = round2._sealed(forged_manifest)

    def load(path):
        if path == manifest_path:
            return forged_manifest
        if path == inventory_path:
            return forged_inventory
        return original_load(path)

    monkeypatch.setattr(round2, "_load_json", load)
    with pytest.raises(RuntimeError, match=message):
        round2._verify_package(round2.OUTPUT_DIR)


def test_p0_bootstrap_patch_extends_e1_e2_e3_without_replacing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_bootstrap_fixture(monkeypatch)
    patched = round2._patch_template(raw)

    for experiment in (
        "support_residual_linear",
        "no_reliability_linear",
        "support_residual_no_reliability",
        round2.P0_EXPERIMENT,
    ):
        assert patched.count(f'"{experiment}"') >= 2
    assert patched.count(f'"{round2.P0_CONFIG}"') == 1


def test_round2_transition_and_task_parameters_pin_fixed_protocol() -> None:
    manifest = round2._verify_package(round2.OUTPUT_DIR)
    dataset_id = "a" * 32
    transition = round2._transition(manifest, dataset_id=dataset_id)
    parameters = round2._task_parameters(manifest, dataset_id=dataset_id)

    assert transition["base_source"]["dataset_id"] == round2.fastlane.SOURCE_DATASET_ID
    assert transition["target_source"]["dataset_id"] == dataset_id
    assert transition["inventory_delta"]["added_file_count"] == 1
    assert transition["inventory_delta"]["modified_file_count"] == 0
    assert transition["inventory_delta"]["removed_file_count"] == 0
    assert transition["protocol"] == {
        "global_batch_size": 8,
        "gpu_count": 4,
        "batch_size_per_gpu": 2,
        "max_epochs": 50,
        "val_interval": 10,
        "training_seed": 20250218,
        "training_overlay_protocol_seed": 20250218,
        "precision": "FP32",
    }
    round2._require_seal(transition, context="test transition")

    assert parameters["Args/experiment_from_task"] == round2.P0_EXPERIMENT
    assert parameters["Args/gpus"] == 4
    assert parameters["Args/max_epochs"] == 50
    assert parameters["Args/training_seed"] == 20250218
    assert parameters["Args/amp"] is False
    assert parameters["Args/teacher_task_id"] == round2.fastlane.TEACHER_TASK_ID
    assert parameters["Args/teacher_model_id"] == round2.fastlane.TEACHER_MODEL_ID
    assert parameters["Args/teacher_checkpoint_sha256"] == (
        round2.fastlane.TEACHER_CHECKPOINT_SHA256
    )


def test_transition_artifact_is_pinned_by_hash_size_uri_preview_and_seal() -> None:
    manifest = round2._verify_package(round2.OUTPUT_DIR)
    transition = round2._transition(manifest, dataset_id="a" * 32)
    serialized = json.dumps(transition, sort_keys=True, indent=4).encode()
    record = {
        "key": round2.SOURCE_TRANSITION_ARTIFACT,
        "hash": hashlib.sha256(serialized).hexdigest(),
        "content_size": len(serialized),
        "uri": (
            "http://10.100.34.118:8081/ResilientV2X/Training/"
            f"task.{'b' * 32}/artifacts/{round2.SOURCE_TRANSITION_ARTIFACT}/"
            f"{round2.SOURCE_TRANSITION_ARTIFACT}.json"
        ),
        "type_data": {
            "content_type": "application/json",
            "preview": serialized.decode(),
        },
    }
    task = SimpleNamespace(
        data=SimpleNamespace(execution=SimpleNamespace(artifacts=[_Record(record)]))
    )

    round2._require_transition_artifact(task, transition)
    record["hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256 drifted"):
        round2._require_transition_artifact(task, transition)


def test_json_artifact_receipt_matches_existing_pinned_clearml_convention() -> None:
    assert round2._clearml_json_artifact_receipt(
        round2.fastlane._expected_source_transition()
    ) == (
        round2.fastlane.SOURCE_TRANSITION_ARTIFACT_SHA256,
        round2.fastlane.SOURCE_TRANSITION_ARTIFACT_BYTES,
    )


@pytest.mark.parametrize(
    ("handler", "argv"),
    (
        (
            round2._upload_source,
            ["upload-source"],
        ),
        (
            round2._create_template,
            ["create-template", "--source-dataset-id", "a" * 32],
        ),
        (
            round2._launch,
            [
                "launch",
                "--source-dataset-id",
                "a" * 32,
                "--template-task-id",
                "b" * 32,
            ],
        ),
    ),
)
def test_all_remote_writes_fail_closed_before_importing_clearml(
    handler: object, argv: list[str]
) -> None:
    args = round2._parser().parse_args(argv)
    args.output_dir = round2.OUTPUT_DIR
    with pytest.raises(PermissionError, match="exact execute token required"):
        handler(args)


def test_round2_plan_targets_only_gpu4_a100() -> None:
    manifest = round2._verify_package(round2.OUTPUT_DIR)
    plan = round2._static_plan(manifest)

    assert plan["protocol"]["worker_queue"] == "GPU4-A100"
    assert plan["protocol"]["worker_queue_id"] == round2.WORKER_QUEUE_ID
    assert plan["protocol"]["global_batch_size"] == 8
    assert plan["protocol"]["max_epochs"] == 50
    assert plan["protocol"]["val_interval"] == 10
    assert plan["execution_policy"]["executor_count"] == 1
    assert plan["execution_policy"]["concurrent_launchers_supported"] is False
    round2._require_seal(plan, context="test plan")


def test_prepare_reseals_the_complete_persisted_deployment_plan(tmp_path) -> None:
    manifest = round2._verify_package(round2.OUTPUT_DIR)
    package = manifest["target_source_package"]
    for name in (
        round2.PACKAGE_MANIFEST_NAME,
        "source-inventory.json",
        package["archive_name"],
    ):
        shutil.copy2(round2.OUTPUT_DIR / name, tmp_path / name)

    assert round2._prepare(SimpleNamespace(output_dir=tmp_path)) == 0
    persisted = round2._load_json(tmp_path / round2.DEPLOYMENT_PLAN_NAME)
    round2._require_seal(persisted, context="test persisted deployment plan")
    assert persisted["remote_state_changed"] is False
    assert set(persisted["commands"]) == {
        "1_upload_source",
        "2_create_template",
        "3_launch_a100",
    }
    assert persisted["invariants"]
    expected_static = round2._static_plan(manifest)
    assert persisted["static_plan_seal_sha256"] == expected_static["seal_sha256"]
    observed, static = round2._verify_deployment_plan(tmp_path, manifest)
    assert observed == persisted
    assert static == expected_static


def test_deployment_plan_static_or_outer_seal_drift_fails_closed(tmp_path) -> None:
    manifest = round2._verify_package(round2.OUTPUT_DIR)
    plan = round2._deployment_plan(manifest, output_dir=tmp_path)
    path = tmp_path / round2.DEPLOYMENT_PLAN_NAME
    path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")

    plan["static_plan_seal_sha256"] = "0" * 64
    plan = round2._sealed(plan)
    path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="deployment plan drifted"):
        round2._verify_deployment_plan(tmp_path, manifest)

    plan = round2._deployment_plan(manifest, output_dir=tmp_path)
    plan["seal_sha256"] = "0" * 64
    path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="seal mismatch"):
        round2._verify_deployment_plan(tmp_path, manifest)


@pytest.mark.parametrize(
    ("command", "token", "extra"),
    (
        ("upload-source", round2.UPLOAD_TOKEN, []),
        (
            "create-template",
            round2.TEMPLATE_TOKEN,
            ["--source-dataset-id", "a" * 32],
        ),
        (
            "launch",
            round2.LAUNCH_TOKEN,
            [
                "--source-dataset-id",
                "a" * 32,
                "--template-task-id",
                "b" * 32,
            ],
        ),
    ),
)
def test_remote_handlers_reject_missing_plan_before_clearml_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    command: str,
    token: str,
    extra: list[str],
) -> None:
    manifest = round2._verify_package(round2.OUTPUT_DIR)
    monkeypatch.setattr(round2, "_verify_package", lambda _output_dir: manifest)
    touched = []
    monkeypatch.setattr(
        round2,
        "_downloaded_source_root",
        lambda *_args, **_kwargs: touched.append("dataset"),
    )
    args = round2._parser().parse_args(
        [command, "--output-dir", str(tmp_path), "--execute-token", token, *extra]
    )
    args.output_dir = tmp_path

    with pytest.raises(RuntimeError, match="deployment plan is missing"):
        args.handler(args)
    assert touched == []


def test_bootstrap_script_rejects_entry_point_drift() -> None:
    diff = "bootstrap"
    task = _MockTask(
        "a" * 32,
        "template",
        "completed",
        diff=diff,
        entry_point="wrong.py",
    )
    with pytest.raises(RuntimeError, match="entry point drifted"):
        round2._require_bootstrap_script(
            task,
            expected_sha256=hashlib.sha256(diff.encode()).hexdigest(),
            context="test template",
        )


@pytest.mark.parametrize(
    "parent",
    (
        round2.fastlane.MAIN_CONTROLLER_TASK_ID,
        round2.fastlane.PREDECESSOR_TASK_ID,
    ),
)
def test_clone_api_always_receives_training_project_id(parent: str) -> None:
    calls = []

    class Backend:
        @staticmethod
        def clone(**kwargs):
            calls.append(kwargs)
            return object()

    source = object()
    round2._clone_task(
        Backend,
        source_task=source,
        name="P0",
        parent=parent,
    )
    assert calls == [
        {
            "source_task": source,
            "name": "P0",
            "parent": parent,
            "project": round2.TRAINING_PROJECT_ID,
        }
    ]


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("status", "template identity drifted"),
        ("name", "template identity drifted"),
        ("parent", "template identity drifted"),
        ("project", "template identity drifted"),
        ("project_id", "template identity drifted"),
        ("entry_point", "entry point drifted"),
        ("parameters", "parameter keys drifted"),
    ),
)
def test_round2_template_identity_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch, drift: str, message: str
) -> None:
    raw = _raw_bootstrap_fixture(monkeypatch)
    parameters = {"Args/source_dataset_id": "a" * 32}
    task = _MockTask(
        "b" * 32,
        "expected template",
        "completed",
        parent=round2.fastlane.MAIN_CONTROLLER_TASK_ID,
        parameters=parameters,
        diff=raw,
    )
    if drift == "status":
        task.status = "created"
    elif drift == "name":
        task.name = "wrong"
    elif drift == "parent":
        task.parent = "d" * 32
    elif drift == "project":
        task.project = "wrong/project"
    elif drift == "project_id":
        task.data.project = "d" * 32
    elif drift == "entry_point":
        task.data.script.value["entry_point"] = "wrong.py"
    elif drift == "parameters":
        task.parameters["unexpected"] = True

    with pytest.raises(RuntimeError, match=message):
        round2._validate_round2_template(
            task,
            expected_name="expected template",
            expected_status="completed",
            expected_parameters=parameters,
        )


def test_fixed_base_validation_is_delegated_to_full_fastlane_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_bootstrap_fixture(monkeypatch)
    patched = round2.candidate._apply_candidate_experiment_patch(raw)
    base = _MockTask(
        round2.fastlane.TEMPLATE_TASK_ID,
        round2.fastlane.TEMPLATE_NAME,
        "completed",
        diff=raw,
    )
    calls = []

    def validate(task_class: object) -> tuple[object, str]:
        calls.append(task_class)
        return base, patched

    monkeypatch.setattr(round2.fastlane, "_validate_fixed_inputs", validate)
    assert round2._validated_fixed_base(object()) == (base, raw, patched)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "message",
    (
        "candidate template identity drifted",
        "candidate template source transition payload drifted",
        "sealed teacher OutputModel URL drifted",
        "teacher checkpoint contract drifted: size_bytes",
        "fastlane predecessor parameter drifted: Args/training_seed",
    ),
)
def test_fixed_base_identity_or_artifact_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    def reject(_task_class: object) -> tuple[object, str]:
        raise RuntimeError(message)

    monkeypatch.setattr(round2.fastlane, "_validate_fixed_inputs", reject)
    with pytest.raises(RuntimeError, match=message):
        round2._validated_fixed_base(object())


def test_mock_launch_duplicate_guard_prevents_clone_and_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, backend, _events = _launch_case(monkeypatch, duplicate=True)
    with pytest.raises(RuntimeError, match="refusing duplicate"):
        round2._launch(args)
    assert backend.clone_count == 0
    assert backend.enqueue_count == 0


def test_mock_launch_verifies_receipt_before_fixed_a100_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, backend, events = _launch_case(monkeypatch)
    assert round2._launch(args) == 0
    assert backend.clone_count == 1
    assert backend.enqueue_count == 1
    task = next(task for task_id, task in backend.tasks.items() if task_id != "b" * 32)
    assert task.data.execution.queue == round2.WORKER_QUEUE_ID
    receipt_record = next(
        record.value
        for record in task.data.execution.artifacts
        if record.value["key"] == round2.LAUNCH_RECEIPT_ARTIFACT
    )
    receipt = json.loads(receipt_record["type_data"]["preview"])
    manifest = round2._verify_package(round2.OUTPUT_DIR)
    deployment, static = round2._verify_deployment_plan(round2.OUTPUT_DIR, manifest)
    assert receipt["deployment_plan_seal_sha256"] == deployment["seal_sha256"]
    assert receipt["plan_seal_sha256"] == static["seal_sha256"]
    assert receipt["plan_seal_sha256"] == deployment["static_plan_seal_sha256"]
    assert receipt["task_id"] == task.id
    assert receipt["task_name"] == task.name
    assert receipt["parent_task_id"] == task.parent
    assert receipt["project"] == task.project == round2.TRAINING_PROJECT
    assert receipt["project_id"] == round2.TRAINING_PROJECT_ID
    assert receipt["entry_point"] == round2.fastlane.TEMPLATE_ENTRY_POINT
    assert receipt["pre_enqueue_queue_id"] is None
    assert receipt["planned_queue"] == round2.WORKER_QUEUE
    assert receipt["planned_queue_id"] == round2.WORKER_QUEUE_ID
    readback_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "artifact_record_readback"
        and event[1] == round2.LAUNCH_RECEIPT_ARTIFACT
    )
    enqueue_index = next(
        index for index, event in enumerate(events) if event[0] == "enqueue"
    )
    duplicate_query_indices = [
        index for index, event in enumerate(events) if event[0] == "duplicate_query"
    ]
    assert len(duplicate_query_indices) == 2
    assert duplicate_query_indices[0] < readback_index
    assert readback_index < duplicate_query_indices[1] < enqueue_index


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("zero", "expected exactly one exact name\\+parent match, got 0"),
        ("multiple", "expected exactly one exact name\\+parent match, got 2"),
        ("wrong_id", "matched a different task ID"),
    ),
)
def test_post_receipt_duplicate_guard_fails_closed_before_enqueue(
    monkeypatch: pytest.MonkeyPatch, drift: str, message: str
) -> None:
    args, backend, events = _launch_case(monkeypatch, post_guard_drift=drift)

    with pytest.raises(RuntimeError, match=message):
        round2._launch(args)

    assert backend.clone_count == 1
    assert backend.enqueue_count == 0
    readback_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "artifact_record_readback"
        and event[1] == round2.LAUNCH_RECEIPT_ARTIFACT
    )
    post_guard_index = max(
        index for index, event in enumerate(events) if event[0] == "duplicate_query"
    )
    assert readback_index < post_guard_index
    assert not any(event[0] == "enqueue" for event in events)


def _valid_clone() -> tuple[_MockTask, str, dict[str, object]]:
    patched = "exact patched bootstrap"
    parameters = {"Args/experiment_from_task": round2.P0_EXPERIMENT}
    task = _MockTask(
        "c" * 32,
        "expected P0",
        "created",
        parent=round2.fastlane.PREDECESSOR_TASK_ID,
        parameters=parameters,
        diff=patched,
    )
    return task, patched, parameters


def test_clone_full_observed_identity_is_returned_after_validation() -> None:
    task, patched, parameters = _valid_clone()
    observed = round2._validate_training_clone(
        task,
        expected_name=task.name,
        patched_diff=patched,
        expected_parameters=parameters,
    )
    assert observed == {
        "task_id": task.id,
        "task_name": task.name,
        "parent_task_id": task.parent,
        "project": round2.TRAINING_PROJECT,
        "project_id": round2.TRAINING_PROJECT_ID,
        "entry_point": round2.fastlane.TEMPLATE_ENTRY_POINT,
        "script_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "pre_enqueue_queue_id": None,
        "planned_queue": round2.WORKER_QUEUE,
        "planned_queue_id": round2.WORKER_QUEUE_ID,
    }


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("status", "clone identity drifted"),
        ("name", "clone identity drifted"),
        ("parent", "clone identity drifted"),
        ("project", "clone identity drifted"),
        ("project_id", "clone identity drifted"),
        ("queue", "clone identity drifted"),
        ("entry_point", "entry point drifted"),
        ("diff", "script SHA-256 drifted"),
        ("parameters", "parameter keys drifted"),
    ),
)
def test_clone_full_identity_drift_fails_closed(drift: str, message: str) -> None:
    task, patched, parameters = _valid_clone()
    if drift == "status":
        task.status = "queued"
    elif drift == "name":
        task.name = "wrong"
    elif drift == "parent":
        task.parent = "d" * 32
    elif drift == "project":
        task.project = "wrong/project"
    elif drift == "project_id":
        task.data.project = "d" * 32
    elif drift == "queue":
        task.data.execution.queue = round2.WORKER_QUEUE_ID
    elif drift == "entry_point":
        task.data.script.value["entry_point"] = "wrong.py"
    elif drift == "diff":
        task.data.script.value["diff"] = "wrong"
    elif drift == "parameters":
        task.parameters["unexpected"] = True

    with pytest.raises(RuntimeError, match=message):
        round2._validate_training_clone(
            task,
            expected_name="expected P0",
            patched_diff=patched,
            expected_parameters=parameters,
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("hash", "SHA-256 drifted"),
        ("size", "size drifted"),
        ("uri", "invalid fileserver URI"),
        ("preview", "launch receipt artifact drifted"),
    ),
)
def test_launch_receipt_server_drift_blocks_enqueue(drift: str, message: str) -> None:
    events: list[tuple[object, ...]] = []
    task = _MockTask(
        "c" * 32,
        "P0",
        "created",
        events=events,
        artifact_drift=drift,
    )
    backend = _MockTaskBackend(task, events)
    receipt = round2._sealed({"schema_version": 1, "receipt_type": "round2-test"})

    with pytest.raises(RuntimeError, match=message):
        round2._upload_verify_and_enqueue(
            backend, task, receipt, queue=round2.WORKER_QUEUE
        )
    assert backend.enqueue_count == 0


def test_launch_receipt_seal_drift_fails_before_upload_or_enqueue() -> None:
    events: list[tuple[object, ...]] = []
    task = _MockTask("c" * 32, "P0", "created", events=events)
    backend = _MockTaskBackend(task, events)
    receipt = round2._sealed({"schema_version": 1, "receipt_type": "round2-test"})
    receipt["seal_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="seal mismatch"):
        round2._upload_verify_and_enqueue(
            backend, task, receipt, queue=round2.WORKER_QUEUE
        )
    assert events == []
    assert backend.enqueue_count == 0


def test_enqueue_queue_id_drift_is_detected() -> None:
    events: list[tuple[object, ...]] = []
    task = _MockTask(
        "c" * 32,
        "P0",
        "created",
        parent=round2.fastlane.PREDECESSOR_TASK_ID,
        events=events,
    )
    backend = _MockTaskBackend(task, events, queue_id="d" * 32)
    receipt = round2._sealed({"schema_version": 1, "receipt_type": "round2-test"})

    with pytest.raises(RuntimeError, match="queue ID drifted"):
        round2._upload_verify_and_enqueue(
            backend, task, receipt, queue=round2.WORKER_QUEUE
        )
    assert backend.enqueue_count == 1
