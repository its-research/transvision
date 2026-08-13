from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/pause_clearml_formal_evaluations.py"
SPEC = importlib.util.spec_from_file_location(
    "pause_clearml_formal_evaluations_test_subject", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
pause = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pause)


SOURCE = "print('fixed')\n"
SOURCE_SHA = hashlib.sha256(SOURCE.encode()).hexdigest()


class _Script:
    def __init__(self, entry_point: str) -> None:
        self.entry_point = entry_point

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_point": self.entry_point,
            "repository": "",
            "working_dir": ".",
            "diff": SOURCE,
        }


class _Task:
    def __init__(
        self,
        owner: type["_Tasks"],
        *,
        task_id: str,
        name: str,
        parent: str,
        status: str,
        subject: str | None,
        entry_point: str = pause.ENTRY_POINT,
    ) -> None:
        self.owner = owner
        self.id = task_id
        self.name = name
        self.parent = parent
        self.project = pause.PROJECT_ID
        self.status = status
        self.queue = "queue-id"
        self.callback_mode = "success"
        parameters: dict[str, object] = {}
        if subject is not None:
            parameters = {
                "Args/stage": "baseline_validate",
                "Args/controlled_baseline": subject,
                "Args/controlled_baseline_task_id": parent,
                "Args/predecessor_task_id": parent,
                "Args/controlled_baseline_model_id": "a" * 32,
                "Args/controlled_baseline_checkpoint_sha256": "b" * 64,
            }
        self.parameters = parameters
        self.data = SimpleNamespace(
            parent=parent,
            project=self.project,
            script=_Script(entry_point),
        )

    def _reload(self) -> None:
        self.owner.events.append(("authority", self.id))

    def get_parameters(self, **_: object) -> dict[str, object]:
        return dict(self.parameters)

    def get_executed_queue(self, *, return_name: bool) -> str:
        assert return_name is False
        return self.queue

    def mark_stopped(self, **_: object) -> bool | None:
        self.owner.events.append(("mark_stopped", self.id))
        if self.callback_mode != "no_effect":
            self.status = "stopped"
        if self.callback_mode == "raise_after_accept":
            raise RuntimeError("callback")
        return None


class _Tasks:
    registry: dict[str, _Task] = {}
    events: list[tuple[str, str]] = []

    @classmethod
    def get_task(cls, *, task_id: str) -> _Task:
        cls.events.append(("get_task", task_id))
        return cls.registry[task_id]

    @classmethod
    def dequeue(cls, *, task: _Task) -> bool | None:
        cls.events.append(("dequeue", task.id))
        if task.callback_mode != "no_effect":
            task.status = "created"
        if task.callback_mode == "raise_after_accept":
            raise RuntimeError("callback")
        return None


def _record(task: _Task, action: str) -> dict[str, object]:
    snapshot = pause._snapshot(
        task,
        expected_name=task.name,
        expected_parent=task.parent,
        expected_source_sha256=SOURCE_SHA,
        expected_subject=task.parameters.get("Args/controlled_baseline"),
        context="candidate test",
    )
    return {"action": action, "pre_authority": snapshot}


@pytest.fixture(autouse=True)
def _reset() -> None:
    _Tasks.registry = {}
    _Tasks.events = []


def _task(status: str = "queued") -> _Task:
    task = _Task(
        _Tasks,
        task_id="1" * 32,
        name="candidate",
        parent="2" * 32,
        status=status,
        subject="subject",
    )
    _Tasks.registry[task.id] = task
    return task


def test_dequeue_is_confirmed_by_authority() -> None:
    task = _task("queued")
    result = pause._mutate_one(
        _Tasks,
        record=_record(task, "dequeue_to_created"),
        sleeper=lambda _: None,
    )
    assert result["post_authority"]["status"] == "created"
    assert result["disposition"] == "dequeue_to_created_confirmed"
    assert ("dequeue", task.id) in _Tasks.events


def test_running_w3_task_is_stopped_and_confirmed() -> None:
    task = _task("in_progress")
    result = pause._mutate_one(
        _Tasks,
        record=_record(task, "mark_stopped"),
        sleeper=lambda _: None,
    )
    assert result["post_authority"]["status"] == "stopped"
    assert ("mark_stopped", task.id) in _Tasks.events


@pytest.mark.parametrize("action", ["dequeue_to_created", "mark_stopped"])
def test_callback_error_is_accepted_only_after_authority(action: str) -> None:
    task = _task("queued" if action.startswith("dequeue") else "in_progress")
    task.callback_mode = "raise_after_accept"
    result = pause._mutate_one(
        _Tasks, record=_record(task, action), sleeper=lambda _: None
    )
    assert result["disposition"].endswith("confirmed_after_callback_error")


def test_missing_authoritative_transition_fails_closed() -> None:
    task = _task("queued")
    task.callback_mode = "no_effect"
    with pytest.raises(pause.PauseError, match="readback failed"):
        pause._mutate_one(
            _Tasks,
            record=_record(task, "dequeue_to_created"),
            sleeper=lambda _: None,
        )


def test_preserved_task_performs_no_write() -> None:
    task = _task("failed")
    result = pause._mutate_one(
        _Tasks,
        record=_record(task, "preserve_terminal_or_paused"),
        sleeper=lambda _: None,
    )
    assert result["disposition"] == "preserved"
    assert not any(event[0] in {"dequeue", "mark_stopped"} for event in _Tasks.events)


def test_source_drift_fails_before_mutation() -> None:
    task = _task()
    with pytest.raises(pause.PauseError, match="source identity drifted"):
        pause._snapshot(
            task,
            expected_name=task.name,
            expected_parent=task.parent,
            expected_source_sha256="0" * 64,
            expected_subject="subject",
            context="candidate test",
        )


def test_training_parent_binding_drift_fails() -> None:
    task = _task()
    task.parameters["Args/predecessor_task_id"] = "3" * 32
    with pytest.raises(pause.PauseError, match="predecessor binding drifted"):
        _record(task, "dequeue_to_created")


def test_receipt_is_sealed(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task("created")
    controller = {"task_id": pause.CONTROLLER_TASK_ID, "status": "stopped"}
    inventory = {
        "controller": controller,
        "w3_plan": {"plan_seal_sha256": "a" * 64},
        "records": [
            {
                "scope": "w3_sealed_keep_set",
                "subject": "subject",
                "action": "preserve_terminal_or_paused",
                "pre_authority": _record(task, "preserve_terminal_or_paused")[
                    "pre_authority"
                ],
            }
        ],
    }
    monkeypatch.setattr(pause, "_inventory", lambda _: inventory)
    receipt = pause._receipt(_Tasks, execute=False)
    seal = receipt["seal_sha256"]
    unsigned = dict(receipt)
    unsigned.pop("seal_sha256")
    assert seal == pause._sha(unsigned)
    assert receipt["remote_state_changed"] is False


def test_candidate_running_status_is_rejected_by_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = "subject"
    keep = _Task(
        _Tasks,
        task_id="1" * 32,
        name="formal",
        parent="4" * 32,
        status="created",
        subject=subject,
    )
    candidate = _Task(
        _Tasks,
        task_id="2" * 32,
        name="candidate",
        parent="3" * 32,
        status="in_progress",
        subject="candidate-subject",
    )
    controller = _Task(
        _Tasks,
        task_id=pause.CONTROLLER_TASK_ID,
        name=pause.CONTROLLER_NAME,
        parent=pause.CONTROLLER_PARENT_ID,
        status="stopped",
        subject=None,
        entry_point=pause.CONTROLLER_ENTRY_POINT,
    )
    controller.parameters.update(
        {
            "Deployment/source_sha256": SOURCE_SHA,
            "Deployment/parent_task_id": pause.CONTROLLER_PARENT_ID,
        }
    )
    _Tasks.registry = {keep.id: keep, candidate.id: candidate, controller.id: controller}
    fake = SimpleNamespace(
        FORMAL_SUBJECT_ORDER=(subject,),
        TRAINING_CONTROLLER_TASK_ID=keep.parent,
        _planned_name=lambda _index, _subject: keep.name,
        _load_plan=lambda _task_class: (
            {"seal_sha256": "a" * 64},
            {subject: keep.id},
            "failed",
            {"acquisition": "test"},
        ),
    )
    monkeypatch.setattr(pause, "_load_reconciler", lambda: fake)
    monkeypatch.setattr(pause, "W3_EVALUATION_SOURCE_SHA256", SOURCE_SHA)
    monkeypatch.setattr(pause, "CONTROLLER_SOURCE_SHA256", SOURCE_SHA)
    monkeypatch.setattr(
        pause,
        "CANDIDATES",
        (
            {
                "label": "E",
                "subject": "candidate-subject",
                "task_id": candidate.id,
                "task_name": candidate.name,
                "parent_task_id": candidate.parent,
                "source_sha256": SOURCE_SHA,
            },
        ),
    )
    with pytest.raises(pause.PauseError, match="not safely"):
        pause._inventory(_Tasks)


def test_wrong_execute_token_fails_before_clearml_read() -> None:
    with pytest.raises(pause.PauseError, match="exact execute token"):
        pause.main(["--execute", "--execute-token", "wrong"])
