from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import clearml_sota_fastlane_launcher as launcher


class _Record:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def to_dict(self) -> dict[str, object]:
        return dict(self.value)


class _ArtifactProxy:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        task_id: str,
        events: list[tuple[object, ...]],
    ) -> None:
        self.payload = dict(payload)
        self.task_id = task_id
        self.events = events

    def get(self, *, force_download: bool = False) -> dict[str, object]:
        self.events.append(("artifact_get", self.task_id, force_download))
        return dict(self.payload)


def _record(
    task_id: str,
    key: str,
    digest: str,
    size: int,
    *,
    payload: dict[str, object] | None = None,
) -> _Record:
    type_data: dict[str, object] = {"content_type": "application/octet-stream"}
    if payload is not None:
        type_data = {
            "content_type": "application/json",
            "preview": json.dumps(payload, sort_keys=True, indent=4),
        }
    return _Record(
        {
            "key": key,
            "hash": digest,
            "content_size": size,
            "uri": (
                "http://10.100.34.118:8081/ResilientV2X/Training/"
                f"task.{task_id}/artifacts/{key}/{key}.json"
            ),
            "type_data": type_data,
        }
    )


@dataclass
class _Model:
    id: str
    task: str
    name: str
    url: str


class _Task:
    def __init__(
        self,
        task_id: str,
        name: str,
        status: str,
        *,
        parent: str = "",
        parameters: dict[str, object] | None = None,
        diff: str = "",
        entry_point: str = launcher.TEMPLATE_ENTRY_POINT,
        artifacts: list[_Record] | None = None,
        models: list[_Model] | None = None,
        events: list[tuple[object, ...]] | None = None,
        queue: str | None = None,
    ) -> None:
        self.id = task_id
        self.name = name
        self.status = status
        self.parent = parent
        self.parameters = dict(parameters or {})
        self.output_uri = ""
        self.models = list(models or [])
        self.events = events if events is not None else []
        self.uploads: dict[str, dict[str, object]] = {}
        self.artifacts: dict[str, _ArtifactProxy] = {}
        self.data = SimpleNamespace(
            script=_Record(
                {
                    "entry_point": entry_point,
                    "diff": diff,
                    "binary": "python",
                    "working_dir": ".",
                }
            ),
            execution=SimpleNamespace(artifacts=list(artifacts or []), queue=queue),
        )

    def reload(self) -> None:
        self.events.append(("reload", self.id))

    def get_parameters(self, **_kwargs: object) -> dict[str, object]:
        return dict(self.parameters)

    def set_parameters(self, parameters: dict[str, object]) -> None:
        self.parameters = dict(parameters)
        self.events.append(("parameters", self.id))

    def get_models(self) -> dict[str, list[_Model]]:
        return {"output": list(self.models)}

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: dict[str, object],
        wait_on_upload: bool,
    ) -> bool:
        assert wait_on_upload is True
        payload = dict(artifact_object)
        self.uploads[name] = payload
        self.artifacts[name] = _ArtifactProxy(
            payload,
            task_id=self.id,
            events=self.events,
        )
        serialized = json.dumps(payload, sort_keys=True, indent=4).encode()
        record = _record(
            self.id,
            name,
            hashlib.sha256(serialized).hexdigest(),
            len(serialized),
            payload=payload,
        )
        self.data.execution.artifacts = [
            existing
            for existing in self.data.execution.artifacts
            if existing.value.get("key") != name
        ] + [record]
        self.events.append(("receipt", self.id, name))
        return True

    def flush(self, *, wait_for_uploads: bool) -> None:
        assert wait_for_uploads is True
        self.events.append(("flush", self.id))


class _TaskClass:
    def __init__(
        self, tasks: dict[str, _Task], events: list[tuple[object, ...]]
    ) -> None:
        self.tasks = tasks
        self.events = events
        self.clones: list[_Task] = []
        self.duplicates: dict[str, list[_Task]] = {}

    def get_task(self, *, task_id: str) -> _Task:
        self.events.append(("get", task_id))
        return self.tasks[task_id]

    def get_tasks(
        self, *, task_name: str, task_filter: dict[str, object]
    ) -> list[_Task]:
        assert task_filter == {"parent": launcher.PREDECESSOR_TASK_ID}
        self.events.append(("duplicate_preflight", task_name))
        return [
            task
            for name, tasks in self.duplicates.items()
            if re.fullmatch(task_name, name)
            for task in tasks
        ]

    def clone(
        self,
        *,
        source_task: _Task,
        name: str,
        parent: str,
    ) -> _Task:
        assert source_task.id == launcher.TEMPLATE_TASK_ID
        assert parent == launcher.PREDECESSOR_TASK_ID
        task_id = f"{len(self.clones) + 1:032x}"
        task = _Task(
            task_id,
            name,
            "created",
            parent=parent,
            parameters=source_task.parameters,
            diff=str(source_task.data.script.value["diff"]),
            events=self.events,
        )
        self.clones.append(task)
        self.tasks[task_id] = task
        self.events.append(("clone", task_id))
        return task

    def enqueue(self, *, task: _Task, queue_name: str) -> bool:
        task.status = "in_progress"
        task.data.execution.queue = queue_name
        self.events.append(("enqueue", task.id, queue_name))
        return True


def _script_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    raw = """ADDITIONAL_EXPERIMENT_SPECS = (
    ExperimentSpec(
        "linear_no_distillation",
some_other_text = (
        "concat_capacity_matched",
        "resilient_v2x",
)\n"""
    patched = launcher.candidate._apply_candidate_experiment_patch(raw)
    monkeypatch.setattr(
        launcher,
        "TEMPLATE_SCRIPT_SHA256",
        hashlib.sha256(raw.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        launcher,
        "PATCHED_TEMPLATE_SCRIPT_SHA256",
        hashlib.sha256(patched.encode()).hexdigest(),
    )
    return raw, patched


def _teacher_contract() -> dict[str, object]:
    return {
        "selected_checkpoint": {
            "model_id": launcher.TEACHER_MODEL_ID,
            "name": launcher.TEACHER_MODEL_NAME,
            "filename": launcher.TEACHER_CHECKPOINT_FILENAME,
            "size_bytes": launcher.TEACHER_CHECKPOINT_BYTES,
            "sha256": launcher.TEACHER_CHECKPOINT_SHA256,
            "url": (
                "http://10.100.34.118:8081/task/models/"
                + launcher.TEACHER_CHECKPOINT_FILENAME
            ),
        },
        "selected_epoch": 30,
        "trained_epochs": 50,
    }


def _predecessor_contract() -> dict[str, object]:
    return {
        "task_id": launcher.PREDECESSOR_TASK_ID,
        "experiment": "fcooper",
        "global_batch_size": launcher.GLOBAL_BATCH_SIZE,
        "gpus": launcher.GPU_COUNT,
        "max_epochs": launcher.MAX_EPOCHS,
        "val_interval": launcher.VAL_INTERVAL,
        "seed": launcher.TRAINING_SEED,
        "precision": launcher.PRECISION,
        "training_dataset_id": launcher.TRAINING_DATASET_ID,
    }


def _environment(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_TaskClass, list[tuple[object, ...]], str]:
    raw, patched = _script_fixture(monkeypatch)
    events: list[tuple[object, ...]] = []
    main = _Task(
        launcher.MAIN_CONTROLLER_TASK_ID,
        "main controller",
        "in_progress",
        events=events,
    )
    source = _Task(
        launcher.SOURCE_DATASET_ID,
        launcher.SOURCE_DATASET_NAME,
        "completed",
        parameters={"properties/version": launcher.SOURCE_DATASET_VERSION},
        artifacts=[
            _record(
                launcher.SOURCE_DATASET_ID,
                "data",
                launcher.SOURCE_DATA_ARTIFACT_SHA256,
                launcher.SOURCE_DATA_ARTIFACT_BYTES,
            ),
            _record(
                launcher.SOURCE_DATASET_ID,
                "state",
                launcher.SOURCE_STATE_ARTIFACT_SHA256,
                launcher.SOURCE_STATE_ARTIFACT_BYTES,
            ),
        ],
        events=events,
    )
    teacher = _Task(
        launcher.TEACHER_TASK_ID,
        launcher.TEACHER_TASK_NAME,
        "completed",
        models=[
            _Model(
                launcher.TEACHER_MODEL_ID,
                launcher.TEACHER_TASK_ID,
                launcher.TEACHER_MODEL_NAME,
                (
                    "http://10.100.34.118:8081/task/models/"
                    + launcher.TEACHER_CHECKPOINT_FILENAME
                ),
            )
        ],
        artifacts=[
            _record(
                launcher.TEACHER_TASK_ID,
                launcher.TEACHER_CONTRACT_ARTIFACT,
                launcher.TEACHER_CONTRACT_ARTIFACT_SHA256,
                launcher.TEACHER_CONTRACT_ARTIFACT_BYTES,
                payload=_teacher_contract(),
            )
        ],
        events=events,
    )
    predecessor = _Task(
        launcher.PREDECESSOR_TASK_ID,
        launcher.PREDECESSOR_TASK_NAME,
        "completed",
        parameters={
            "Args/experiment_from_task": "fcooper",
            "Args/gpus": "4",
            "Args/max_epochs": "50",
            "Args/training_seed": str(launcher.TRAINING_SEED),
            "Args/amp": "False",
            "Args/teacher_task_id": launcher.TEACHER_TASK_ID,
            "Args/teacher_model_id": launcher.TEACHER_MODEL_ID,
            "Args/teacher_checkpoint_sha256": launcher.TEACHER_CHECKPOINT_SHA256,
            "Args/training_dataset_id": launcher.TRAINING_DATASET_ID,
        },
        artifacts=[
            _record(
                launcher.PREDECESSOR_TASK_ID,
                "run_contract",
                launcher.PREDECESSOR_RUN_CONTRACT_SHA256,
                launcher.PREDECESSOR_RUN_CONTRACT_BYTES,
                payload=_predecessor_contract(),
            )
        ],
        events=events,
    )
    template = _Task(
        launcher.TEMPLATE_TASK_ID,
        launcher.TEMPLATE_NAME,
        "completed",
        parent=launcher.MAIN_CONTROLLER_TASK_ID,
        parameters=launcher.EXPECTED_TEMPLATE_PARAMETERS,
        diff=raw,
        artifacts=[
            _record(
                launcher.TEMPLATE_TASK_ID,
                launcher.SOURCE_TRANSITION_ARTIFACT,
                launcher.SOURCE_TRANSITION_ARTIFACT_SHA256,
                launcher.SOURCE_TRANSITION_ARTIFACT_BYTES,
                payload=launcher._expected_source_transition(),
            )
        ],
        events=events,
    )
    tasks = {task.id: task for task in (main, source, teacher, predecessor, template)}
    return _TaskClass(tasks, events), events, patched


def _editor(task_class: _TaskClass, events: list[tuple[object, ...]]):
    def edit(task_id: str, diff: str) -> None:
        task_class.tasks[task_id].data.script.value["diff"] = diff
        events.append(("edit", task_id))

    return edit


def _recovery_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_TaskClass, list[tuple[object, ...]], list[_Task], str]:
    task_class, events, patched = _environment(monkeypatch)
    plan = launcher._static_plan_payload()
    partials: list[_Task] = []
    for item, task_id in zip(
        launcher.CANDIDATES, launcher.RECOVERY_TASK_IDS, strict=True
    ):
        name = launcher._task_name(item, plan["seal_sha256"])
        task = _Task(
            task_id,
            name,
            "created",
            parent=launcher.PREDECESSOR_TASK_ID,
            parameters=launcher._task_parameters(str(item["experiment"])),
            diff=patched,
            events=events,
        )
        task.output_uri = launcher.FILES_SERVER_URI
        task_class.tasks[task_id] = task
        task_class.duplicates[name] = [task]
        partials.append(task)

    receipt = launcher._launch_receipt(
        plan=plan,
        tasks=partials,
        created_at=launcher.RECOVERY_RECEIPT_CREATED_AT,
    )
    receipt.pop("seal_sha256")
    receipt["recovery"] = {
        "mode": "adopt_exact_created_pre_enqueue_partial_v1",
        "incident": "fileserver_401_before_first_receipt_upload",
        "observed_task_ids": list(launcher.RECOVERY_TASK_IDS),
        "observed_statuses": ["created"] * 3,
        "observed_execution_queues": [None] * 3,
        "observed_artifact_names": [[], [], []],
        "prior_enqueue_count": 0,
        "server_time_anchor": "latest_clone_created_at",
        "all_clone_identity_checks": "pass",
    }
    monkeypatch.setattr(
        launcher,
        "RECOVERY_RECEIPT_SEAL_SHA256",
        launcher._sealed(receipt)["seal_sha256"],
    )
    return task_class, events, partials, patched


def _add_recovery_training_artifacts(
    task: _Task,
    names: tuple[str, ...] = launcher._RECOVERY_TRAINING_ARTIFACT_SEQUENCE,
) -> None:
    for name in names:
        task.data.execution.artifacts.append(
            _record(
                task.id,
                name,
                hashlib.sha256(f"{task.id}:{name}".encode()).hexdigest(),
                len(name),
            )
        )


def _upload_recovery_receipts(partials: list[_Task]) -> dict[str, object]:
    receipt = launcher._recovery_receipt(
        plan=launcher._static_plan_payload(),
        tasks=partials,
    )
    for task in partials:
        launcher._upload_receipt(task, receipt)
    return receipt


def test_default_invocation_is_a_sealed_local_only_dry_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert launcher.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    launcher._require_valid_seal(plan, context="test plan")
    assert plan["remote_state_changed"] is False
    assert plan["max_parallel"] == 2
    assert plan["worker_queues"] == ["GPU4-5090", "GPU4-V100"]
    assert [item["label"] for item in plan["candidates"]] == ["E1", "E2", "E3"]
    assert plan["protocol"] == {
        "global_batch_size": 8,
        "gpu_count": 4,
        "batch_size_per_gpu": 2,
        "max_epochs": 50,
        "val_interval": 10,
        "training_seed": 20250218,
        "training_overlay_protocol_seed": 20250218,
        "precision": "FP32",
        "amp": False,
    }


def test_source_transition_is_exact_and_recomputes_to_fixed_seal() -> None:
    transition = launcher._expected_source_transition()
    assert transition["seal_sha256"] == launcher.SOURCE_TRANSITION_SEAL_SHA256
    assert transition["target_source"]["dataset_id"] == launcher.SOURCE_DATASET_ID
    assert transition["candidate_order"] == [
        "support_residual_linear",
        "no_reliability_linear",
        "support_residual_no_reliability",
    ]
    assert transition["protocol"]["global_batch_size"] == 8
    assert transition["protocol"]["max_epochs"] == 50
    assert transition["protocol"]["val_interval"] == 10
    assert transition["protocol"]["training_seed"] == 20250218
    assert transition["protocol"]["precision"] == "FP32"


def test_wrong_execution_token_fails_before_any_remote_read() -> None:
    class _Forbidden:
        def get_task(self, **_kwargs: object) -> object:
            raise AssertionError("remote read occurred before token validation")

    with pytest.raises(PermissionError, match="remote mutation requires"):
        launcher.execute_fastlane(
            execute_token="wrong",
            poll_seconds=1,
            task_class=_Forbidden(),
        )


def test_token_is_rejected_without_explicit_execute_flag() -> None:
    with pytest.raises(ValueError, match="invalid without an execution mode"):
        launcher.main(["--execute-token", launcher.EXECUTE_TOKEN])


def test_execution_creates_all_receipts_before_enqueue_and_releases_only_after_e1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, events, patched = _environment(monkeypatch)
    slept = 0

    def sleeper(seconds: float) -> None:
        nonlocal slept
        assert seconds == 1
        slept += 1
        assert len([event for event in events if event[0] == "enqueue"]) == 2
        assert task_class.clones[2].status == "created"
        task_class.clones[0].status = "completed"
        events.append(("e1_completed", task_class.clones[0].id))

    result = launcher.execute_fastlane(
        execute_token=launcher.EXECUTE_TOKEN,
        poll_seconds=1,
        task_class=task_class,
        script_editor=_editor(task_class, events),
        sleeper=sleeper,
        now=lambda: "2026-08-12T04:00:00+00:00",
    )

    assert slept == 1
    assert [event[2] for event in events if event[0] == "enqueue"] == [
        "GPU4-5090",
        "GPU4-V100",
        "GPU4-5090",
    ]
    e3_enqueue_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "enqueue" and event[1] == task_class.clones[2].id
    )
    e1_completed_index = next(
        index for index, event in enumerate(events) if event[0] == "e1_completed"
    )
    assert e3_enqueue_index > e1_completed_index
    first_enqueue_index = next(
        index for index, event in enumerate(events) if event[0] == "enqueue"
    )
    receipt_indices = [
        index for index, event in enumerate(events) if event[0] == "receipt"
    ]
    assert len(receipt_indices) == 3
    assert max(receipt_indices) < first_enqueue_index

    receipts = [
        task.uploads[launcher.LAUNCH_RECEIPT_ARTIFACT] for task in task_class.clones
    ]
    assert receipts[0] == receipts[1] == receipts[2]
    launcher._require_valid_seal(receipts[0], context="test launch receipt")
    assert receipts[0]["created_at"] == "2026-08-12T04:00:00+00:00"
    assert receipts[0]["max_parallel"] == 2
    assert [item["task_id"] for item in receipts[0]["tasks"]] == [
        task.id for task in task_class.clones
    ]

    for item, task in zip(launcher.CANDIDATES, task_class.clones, strict=True):
        assert task.parent == launcher.PREDECESSOR_TASK_ID
        assert task.output_uri == launcher.FILES_SERVER_URI
        assert task.data.script.value["diff"] == patched
        assert task.parameters["Args/experiment_from_task"] == item["experiment"]
        assert task.parameters["Args/predecessor_task_id"] == (
            launcher.PREDECESSOR_TASK_ID
        )
        assert task.parameters["Args/teacher_task_id"] == launcher.TEACHER_TASK_ID
        assert task.parameters["Args/teacher_model_id"] == launcher.TEACHER_MODEL_ID
        assert task.parameters["Args/teacher_checkpoint_sha256"] == (
            launcher.TEACHER_CHECKPOINT_SHA256
        )
        assert task.parameters["Args/gpus"] == 4
        assert task.parameters["Args/max_epochs"] == 50
        assert task.parameters["Args/training_seed"] == 20250218
        assert task.parameters["Args/amp"] is False

    launcher._require_valid_seal(result, context="test release result")
    assert result["remote_state_changed"] is True
    assert [item["status"] for item in result["tasks"]] == [
        "completed",
        "in_progress",
        "in_progress",
    ]


def test_e1_failure_keeps_e3_created_and_unqueued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, events, _patched = _environment(monkeypatch)

    def sleeper(_seconds: float) -> None:
        task_class.clones[0].status = "failed"

    with pytest.raises(RuntimeError, match="E1 ended before E3 release"):
        launcher.execute_fastlane(
            execute_token=launcher.EXECUTE_TOKEN,
            poll_seconds=1,
            task_class=task_class,
            script_editor=_editor(task_class, events),
            sleeper=sleeper,
        )
    assert len([event for event in events if event[0] == "enqueue"]) == 2
    assert task_class.clones[2].status == "created"
    assert launcher.LAUNCH_RECEIPT_ARTIFACT in task_class.clones[2].uploads


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("main_status", "only while the main controller is in progress"),
        ("source_hash", "artifact 'data' SHA-256 drifted"),
        ("teacher_model", "teacher OutputModel binding drifted"),
        ("predecessor_status", "not the exact completed gate"),
        ("template_parent", "candidate template identity drifted"),
        ("template_parameter", "candidate template parameter drifted"),
        ("transition_payload", "source transition payload drifted"),
    ),
)
def test_remote_input_drift_fails_before_clone(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    task_class, events, _patched = _environment(monkeypatch)
    if tamper == "main_status":
        task_class.tasks[launcher.MAIN_CONTROLLER_TASK_ID].status = "completed"
    elif tamper == "source_hash":
        record = task_class.tasks[launcher.SOURCE_DATASET_ID].data.execution.artifacts[
            0
        ]
        record.value["hash"] = "0" * 64
    elif tamper == "teacher_model":
        task_class.tasks[launcher.TEACHER_TASK_ID].models[0].id = "0" * 32
    elif tamper == "predecessor_status":
        task_class.tasks[launcher.PREDECESSOR_TASK_ID].status = "failed"
    elif tamper == "template_parent":
        task_class.tasks[launcher.TEMPLATE_TASK_ID].parent = "0" * 32
    elif tamper == "template_parameter":
        task_class.tasks[launcher.TEMPLATE_TASK_ID].parameters["Args/max_epochs"] = 49
    elif tamper == "transition_payload":
        record = task_class.tasks[launcher.TEMPLATE_TASK_ID].data.execution.artifacts[0]
        payload = launcher._expected_source_transition()
        payload["protocol"]["max_epochs"] = 49
        record.value["type_data"]["preview"] = json.dumps(payload)
    else:
        raise AssertionError(tamper)

    with pytest.raises(RuntimeError, match=message):
        launcher.execute_fastlane(
            execute_token=launcher.EXECUTE_TOKEN,
            poll_seconds=1,
            task_class=task_class,
            script_editor=_editor(task_class, events),
            sleeper=lambda _seconds: None,
        )
    assert task_class.clones == []
    assert not any(event[0] == "enqueue" for event in events)


def test_duplicate_task_name_fails_before_any_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, events, _patched = _environment(monkeypatch)
    plan = launcher._static_plan_payload()
    name = launcher._task_name(launcher.CANDIDATES[0], plan["seal_sha256"])
    task_class.duplicates[name] = [_Task("f" * 32, name, "created", events=events)]
    with pytest.raises(RuntimeError, match="refusing duplicate fastlane task name"):
        launcher.execute_fastlane(
            execute_token=launcher.EXECUTE_TOKEN,
            poll_seconds=1,
            task_class=task_class,
            script_editor=_editor(task_class, events),
        )
    assert task_class.clones == []


def test_launch_receipt_tampering_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, _events, _patched = _environment(monkeypatch)
    plan = launcher._static_plan_payload()
    fake_tasks = [
        _Task(
            f"{index:032x}",
            launcher._task_name(item, plan["seal_sha256"]),
            "created",
            parent=launcher.PREDECESSOR_TASK_ID,
        )
        for index, item in enumerate(launcher.CANDIDATES, start=1)
    ]
    receipt = launcher._launch_receipt(
        plan=plan,
        tasks=fake_tasks,
        created_at="2026-08-12T04:00:00+00:00",
    )
    receipt["max_parallel"] = 3
    with pytest.raises(RuntimeError, match="seal_sha256 mismatch"):
        launcher._require_valid_seal(receipt, context="tampered receipt")


def test_production_recovery_receipt_structure_and_seal_are_exact() -> None:
    plan = launcher._static_plan_payload()
    tasks = [
        _Task(
            task_id,
            launcher._task_name(item, plan["seal_sha256"]),
            "created",
            parent=launcher.PREDECESSOR_TASK_ID,
        )
        for item, task_id in zip(
            launcher.CANDIDATES, launcher.RECOVERY_TASK_IDS, strict=True
        )
    ]
    receipt = launcher._recovery_receipt(plan=plan, tasks=tasks)
    assert receipt["seal_sha256"] == (
        "16d119a4260824b1f870d56722e25c4de14c1b3b2ff1f41d132401e111b13492"
    )
    assert receipt["created_at"] == "2026-08-11T20:14:57.817000+00:00"
    assert receipt["recovery"] == {
        "mode": "adopt_exact_created_pre_enqueue_partial_v1",
        "incident": "fileserver_401_before_first_receipt_upload",
        "observed_task_ids": list(launcher.RECOVERY_TASK_IDS),
        "observed_statuses": ["created", "created", "created"],
        "observed_execution_queues": [None, None, None],
        "observed_artifact_names": [[], [], []],
        "prior_enqueue_count": 0,
        "server_time_anchor": "latest_clone_created_at",
        "all_clone_identity_checks": "pass",
    }


def test_recovery_wrong_token_fails_before_any_remote_read() -> None:
    class _Forbidden:
        def get_task(self, **_kwargs: object) -> object:
            raise AssertionError(
                "remote read occurred before recovery token validation"
            )

    with pytest.raises(PermissionError, match="recovery mutation requires"):
        launcher.recover_exact_partials(
            execute_token="wrong",
            poll_seconds=1,
            task_class=_Forbidden(),
        )


def test_recovery_cli_requires_its_distinct_token() -> None:
    with pytest.raises(PermissionError, match=launcher.RECOVERY_EXECUTE_TOKEN):
        launcher.main(
            [
                "--recover-exact-partials",
                "--execute-token",
                launcher.EXECUTE_TOKEN,
            ]
        )


def test_recovery_reuses_exact_tasks_reads_receipts_and_releases_e3_after_e1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, events, partials, _patched = _recovery_environment(monkeypatch)
    slept = 0

    def sleeper(seconds: float) -> None:
        nonlocal slept
        assert seconds == 1
        slept += 1
        assert partials[2].status == "created"
        assert partials[2].data.execution.queue is None
        _add_recovery_training_artifacts(partials[0])
        _add_recovery_training_artifacts(
            partials[1],
            launcher._RECOVERY_TRAINING_ARTIFACT_SEQUENCE[:1],
        )
        partials[0].status = "completed"
        events.append(("recovery_e1_completed", partials[0].id))

    result = launcher.recover_exact_partials(
        execute_token=launcher.RECOVERY_EXECUTE_TOKEN,
        poll_seconds=1,
        task_class=task_class,
        sleeper=sleeper,
    )
    assert slept == 1
    assert task_class.clones == []
    assert [event[2] for event in events if event[0] == "enqueue"] == [
        "GPU4-5090",
        "GPU4-V100",
        "GPU4-5090",
    ]
    first_enqueue = next(
        index for index, event in enumerate(events) if event[0] == "enqueue"
    )
    assert (
        max(index for index, event in enumerate(events) if event[0] == "receipt")
        < first_enqueue
    )
    e1_completed = next(
        index
        for index, event in enumerate(events)
        if event[0] == "recovery_e1_completed"
    )
    e3_enqueue = next(
        index
        for index, event in enumerate(events)
        if event[0] == "enqueue" and event[1] == partials[2].id
    )
    assert e3_enqueue > e1_completed

    receipts = [task.uploads[launcher.LAUNCH_RECEIPT_ARTIFACT] for task in partials]
    assert receipts[0] == receipts[1] == receipts[2]
    assert receipts[0]["seal_sha256"] == launcher.RECOVERY_RECEIPT_SEAL_SHA256
    assert receipts[0]["recovery"]["prior_enqueue_count"] == 0
    forced_gets = [event for event in events if event[0] == "artifact_get"]
    assert forced_gets
    assert all(event[2] is True for event in forced_gets)
    assert {event[1] for event in forced_gets} == set(launcher.RECOVERY_TASK_IDS)
    launcher._require_valid_seal(result, context="test recovery result")
    assert result["launch_receipt_seal_sha256"] == (
        launcher.RECOVERY_RECEIPT_SEAL_SHA256
    )
    assert [item["task_id"] for item in result["tasks"]] == list(
        launcher.RECOVERY_TASK_IDS
    )


def test_recovery_completed_task_accepts_only_full_training_contract_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _task_class, _events, partials, _patched = _recovery_environment(monkeypatch)
    receipt = _upload_recovery_receipts(partials)
    task = partials[0]
    _add_recovery_training_artifacts(task)
    task.status = "completed"

    observation = launcher._recovery_receipt_readback(task, receipt=receipt)

    assert re.fullmatch(r"[0-9a-f]{64}", str(observation["hash"]))
    assert observation["content_size"] > 0


@pytest.mark.parametrize("status", ("created", "queued"))
def test_recovery_created_or_queued_task_remains_strictly_receipt_only(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    _task_class, _events, partials, _patched = _recovery_environment(monkeypatch)
    receipt = _upload_recovery_receipts(partials)
    task = partials[2]
    task.status = status
    _add_recovery_training_artifacts(
        task,
        launcher._RECOVERY_TRAINING_ARTIFACT_SEQUENCE[:1],
    )

    with pytest.raises(RuntimeError, match="is not receipt-only"):
        launcher._recovery_receipt_readback(task, receipt=receipt)


@pytest.mark.parametrize(
    "prefix_length",
    range(len(launcher._RECOVERY_TRAINING_ARTIFACT_SEQUENCE) + 1),
)
def test_recovery_in_progress_task_accepts_only_ordered_training_prefixes(
    monkeypatch: pytest.MonkeyPatch,
    prefix_length: int,
) -> None:
    _task_class, _events, partials, _patched = _recovery_environment(monkeypatch)
    receipt = _upload_recovery_receipts(partials)
    task = partials[1]
    task.status = "in_progress"
    _add_recovery_training_artifacts(
        task,
        launcher._RECOVERY_TRAINING_ARTIFACT_SEQUENCE[:prefix_length],
    )

    launcher._recovery_receipt_readback(task, receipt=receipt)


@pytest.mark.parametrize(
    "names",
    (
        ("final_checkpoint_contract",),
        launcher._RECOVERY_TRAINING_ARTIFACT_SEQUENCE + ("unexpected",),
    ),
)
def test_recovery_in_progress_task_rejects_nonprefix_or_extra_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
) -> None:
    _task_class, _events, partials, _patched = _recovery_environment(monkeypatch)
    receipt = _upload_recovery_receipts(partials)
    task = partials[1]
    task.status = "in_progress"
    _add_recovery_training_artifacts(task, names)

    with pytest.raises(RuntimeError, match="training artifact inventory drifted"):
        launcher._recovery_receipt_readback(task, receipt=receipt)


@pytest.mark.parametrize("drift", ("missing", "extra", "duplicate"))
def test_recovery_completed_task_rejects_contract_inventory_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    _task_class, _events, partials, _patched = _recovery_environment(monkeypatch)
    receipt = _upload_recovery_receipts(partials)
    task = partials[0]
    names = launcher._RECOVERY_TRAINING_ARTIFACT_SEQUENCE
    _add_recovery_training_artifacts(
        task,
        names[:-1] if drift == "missing" else names,
    )
    if drift == "extra":
        _add_recovery_training_artifacts(task, ("unexpected",))
    elif drift == "duplicate":
        _add_recovery_training_artifacts(task, names[:1])
    task.status = "completed"

    message = (
        "duplicate or unnamed"
        if drift == "duplicate"
        else "checkpoint-contract artifact inventory drifted"
    )
    with pytest.raises(RuntimeError, match=message):
        launcher._recovery_receipt_readback(task, receipt=receipt)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("id", "task ID is not allowlisted"),
        ("status", "partial is not created"),
        ("queue", "partial already has a queue"),
        ("artifact", "partial artifacts are not empty"),
        ("parent", "clone identity drifted"),
        ("script", "clone script drifted"),
        ("parameter", "clone parameter drifted"),
        ("output_uri", "partial output URI drifted"),
    ),
)
def test_recovery_partial_drift_fails_before_receipt_or_enqueue(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    task_class, events, partials, _patched = _recovery_environment(monkeypatch)
    if tamper == "id":
        name = partials[0].name
        task_class.duplicates[name] = [_Task("e" * 32, name, "created", events=events)]
    elif tamper == "status":
        partials[0].status = "queued"
    elif tamper == "queue":
        partials[0].data.execution.queue = "queue-id"
    elif tamper == "artifact":
        partials[0].data.execution.artifacts = [
            _record(partials[0].id, "unexpected", "0" * 64, 1)
        ]
    elif tamper == "parent":
        partials[0].parent = "0" * 32
    elif tamper == "script":
        partials[0].data.script.value["diff"] += "\n# drift"
    elif tamper == "parameter":
        partials[0].parameters["Args/max_epochs"] = 49
    elif tamper == "output_uri":
        partials[0].output_uri = ""
    else:
        raise AssertionError(tamper)

    with pytest.raises(RuntimeError, match=message):
        launcher.recover_exact_partials(
            execute_token=launcher.RECOVERY_EXECUTE_TOKEN,
            poll_seconds=1,
            task_class=task_class,
            sleeper=lambda _seconds: None,
        )
    assert not any(event[0] in {"receipt", "enqueue"} for event in events)
    assert task_class.clones == []


def test_recovery_forced_payload_drift_prevents_every_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, events, partials, _patched = _recovery_environment(monkeypatch)
    original = partials[1].upload_artifact

    def upload_tampered(
        name: str,
        *,
        artifact_object: dict[str, object],
        wait_on_upload: bool,
    ) -> bool:
        result = original(
            name,
            artifact_object=artifact_object,
            wait_on_upload=wait_on_upload,
        )
        partials[1].artifacts[name].payload["max_parallel"] = 3
        return result

    partials[1].upload_artifact = upload_tampered  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="forced server readback drifted"):
        launcher.recover_exact_partials(
            execute_token=launcher.RECOVERY_EXECUTE_TOKEN,
            poll_seconds=1,
            task_class=task_class,
            sleeper=lambda _seconds: None,
        )
    assert not any(event[0] == "enqueue" for event in events)


def test_recovery_cross_task_artifact_hash_drift_prevents_every_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, events, partials, _patched = _recovery_environment(monkeypatch)
    original = partials[1].upload_artifact

    def upload_wrong_hash(
        name: str,
        *,
        artifact_object: dict[str, object],
        wait_on_upload: bool,
    ) -> bool:
        result = original(
            name,
            artifact_object=artifact_object,
            wait_on_upload=wait_on_upload,
        )
        partials[1].data.execution.artifacts[0].value["hash"] = "0" * 64
        return result

    partials[1].upload_artifact = upload_wrong_hash  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="hashes differ across tasks"):
        launcher.recover_exact_partials(
            execute_token=launcher.RECOVERY_EXECUTE_TOKEN,
            poll_seconds=1,
            task_class=task_class,
            sleeper=lambda _seconds: None,
        )
    assert not any(event[0] == "enqueue" for event in events)


def test_recovery_e1_failure_keeps_e3_created_and_unqueued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_class, events, partials, _patched = _recovery_environment(monkeypatch)

    def sleeper(_seconds: float) -> None:
        partials[0].status = "failed"

    with pytest.raises(RuntimeError, match="recovered E1 ended before E3 release"):
        launcher.recover_exact_partials(
            execute_token=launcher.RECOVERY_EXECUTE_TOKEN,
            poll_seconds=1,
            task_class=task_class,
            sleeper=sleeper,
        )
    assert len([event for event in events if event[0] == "enqueue"]) == 2
    assert partials[2].status == "created"
    assert partials[2].data.execution.queue is None
