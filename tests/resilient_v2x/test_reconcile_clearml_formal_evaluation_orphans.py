from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "tools/resilient_v2x/reconcile_clearml_formal_evaluation_orphans.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reconcile_clearml_formal_evaluation_orphans_test_subject", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
reconcile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reconcile)


class _Artifact:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def get(self, *, force_download: bool) -> dict[str, object]:
        assert force_download is True
        return self.value


class _FakeTask:
    def __init__(
        self,
        owner: type["_FakeTaskClass"],
        *,
        task_id: str,
        name: str,
        parent: str,
        status: str,
    ) -> None:
        self.owner = owner
        self.id = task_id
        self.name = name
        self.project = reconcile.PROJECT_ID
        self.parent = parent
        self.status = status
        self.artifacts: dict[str, object] = {}
        self.tags: list[str] = []
        self.system_tags: list[str] = []
        self.models: dict[str, list[object]] = {"input": [], "output": []}
        self.reported_console: list[object] = []
        self.reported_scalars: dict[str, object] = {}
        self.reported_plots: list[object] = []
        self.reported_single_values: dict[str, object] = {}
        self.archive_mode = "success"
        self.data = SimpleNamespace(
            project=self.project,
            parent=parent,
            status=status,
            execution=SimpleNamespace(
                queue=None,
                parameters={},
                model_desc={},
                model_labels={},
                artifacts=[],
            ),
            output=SimpleNamespace(destination="http://fileserver"),
            runtime={},
            started=None,
            completed=None,
            published=None,
            last_worker=None,
        )

    def _reload(self) -> None:
        self.owner.events.append(("authority", self.id))

    def get_tags(self) -> list[str]:
        return list(self.tags)

    def get_system_tags(self) -> list[str]:
        return list(self.system_tags)

    def get_archived(self) -> bool:
        return reconcile.ARCHIVED_TAG in self.system_tags

    def get_models(self) -> dict[str, list[object]]:
        return {key: list(value) for key, value in self.models.items()}

    def get_executed_queue(self, *, return_name: bool) -> str | None:
        assert return_name is False
        return self.data.execution.queue

    def get_reported_console_output(self, *, number_of_reports: int) -> list[object]:
        assert number_of_reports == 1
        return list(self.reported_console)

    def get_reported_scalars(self, *, max_samples: int) -> dict[str, object]:
        assert max_samples == 1
        return dict(self.reported_scalars)

    def get_reported_plots(self, *, max_iterations: int) -> list[object]:
        assert max_iterations == 1
        return list(self.reported_plots)

    def get_reported_single_values(self) -> dict[str, object]:
        return dict(self.reported_single_values)

    def set_archived(self, archive: bool) -> bool | None:
        assert archive is True
        self.owner.events.append(("set_archived", self.id))
        if self.archive_mode == "no_effect":
            return True
        if reconcile.ARCHIVED_TAG not in self.system_tags:
            self.system_tags.append(reconcile.ARCHIVED_TAG)
        if self.archive_mode == "raise_after_accept":
            raise RuntimeError("transport callback failed")
        if self.archive_mode == "false_after_accept":
            return False
        return None


class _FakeTaskClass:
    registry: dict[str, _FakeTask] = {}
    events: list[tuple[str, str]] = []
    keep_ids: dict[str, str] = {}
    orphan_ids: dict[str, str] = {}

    @classmethod
    def reset(cls) -> None:
        cls.registry = {}
        cls.events = []
        cls.keep_ids = {}
        cls.orphan_ids = {}
        plan_entries = []
        for index, subject in enumerate(reconcile.FORMAL_SUBJECT_ORDER, start=1):
            name = reconcile._planned_name(index, subject)
            keep_id = f"{0x1000 + index:032x}"
            orphan_id = f"{0x2000 + index:032x}"
            cls.keep_ids[subject] = keep_id
            cls.orphan_ids[subject] = orphan_id
            cls.registry[keep_id] = _FakeTask(
                cls,
                task_id=keep_id,
                name=name,
                parent=reconcile.TRAINING_CONTROLLER_TASK_ID,
                status="queued",
            )
            cls.registry[orphan_id] = _FakeTask(
                cls,
                task_id=orphan_id,
                name=name,
                parent=reconcile.TRAINING_CONTROLLER_TASK_ID,
                status="created",
            )
            plan_entries.append(
                {
                    "subject": subject,
                    "evaluation_task_id": keep_id,
                    "queue": "GPU4-A100",
                }
            )
        plan: dict[str, object] = {
            "schema_version": 2,
            "plan_type": reconcile.PLAN_TYPE,
            "training_controller_task_id": reconcile.TRAINING_CONTROLLER_TASK_ID,
            "training_provenance_task_id": reconcile.PROVENANCE_TASK_ID,
            "protocol_id": reconcile.PROTOCOL_ID,
            "sample_count": 1337,
            "run_count": 12,
            "subject_order": list(reconcile.FORMAL_SUBJECT_ORDER),
            "entries": plan_entries,
        }
        plan["seal_sha256"] = reconcile._content_sha256(plan)
        w3 = _FakeTask(
            cls,
            task_id=reconcile.W3_TASK_ID,
            name=reconcile.W3_TASK_NAME,
            parent=reconcile.PROVENANCE_TASK_ID,
            status="in_progress",
        )
        w3.artifacts[reconcile.PLAN_ARTIFACT] = _Artifact(plan)
        cls.registry[w3.id] = w3

    @classmethod
    def get_task(cls, *, task_id: str) -> _FakeTask:
        cls.events.append(("get_task", task_id))
        return cls.registry[task_id]

    @classmethod
    def get_tasks(
        cls, *, allow_archived: bool, task_filter: dict[str, str]
    ) -> list[_FakeTask]:
        assert allow_archived is True
        assert task_filter == {"parent": reconcile.TRAINING_CONTROLLER_TASK_ID}
        cls.events.append(("get_tasks", "inventory"))
        return [
            task
            for task in cls.registry.values()
            if task.parent == reconcile.TRAINING_CONTROLLER_TASK_ID
        ]


@pytest.fixture(autouse=True)
def _reset() -> None:
    _FakeTaskClass.reset()


def _assert_sealed(receipt: dict[str, object]) -> None:
    seal = receipt["seal_sha256"]
    unsigned = dict(receipt)
    unsigned.pop("seal_sha256")
    assert seal == reconcile._content_sha256(unsigned)


def _orphan(subject: str = "support_residual") -> _FakeTask:
    return _FakeTaskClass.registry[_FakeTaskClass.orphan_ids[subject]]


def _keep(subject: str = "support_residual") -> _FakeTask:
    return _FakeTaskClass.registry[_FakeTaskClass.keep_ids[subject]]


def test_dry_run_binds_exact_w3_keep_set_without_mutation() -> None:
    receipt = reconcile._dry_run_receipt(_FakeTaskClass)

    _assert_sealed(receipt)
    assert receipt["mode"] == "dry_run"
    assert receipt["status"] == "planned"
    assert receipt["remote_state_changed"] is False
    assert receipt["summary"] == {
        "total": 52,
        "keep": 26,
        "orphan": 26,
        "would_archive": 26,
        "already_archived": 0,
    }
    assert receipt["plan_binding"]["keep_task_ids"] == list(
        _FakeTaskClass.keep_ids.values()
    )
    assert not any(event[0] == "set_archived" for event in _FakeTaskClass.events)
    assert sum(item["plan_binding"] == "keep" for item in receipt["tasks"]) == 26
    assert all("authority_sha256" in item["pre_authority"] for item in receipt["tasks"])


def test_failed_w3_producer_is_recorded_when_its_sealed_plan_is_valid() -> None:
    w3 = _FakeTaskClass.registry[reconcile.W3_TASK_ID]
    w3.status = "failed"
    w3.data.status = "failed"

    receipt = reconcile._dry_run_receipt(_FakeTaskClass)

    assert receipt["plan_binding"]["plan_producer_status"] == "failed"


def test_execute_archives_only_orphans_and_confirms_final_authority() -> None:
    receipt = reconcile._execute_receipt(_FakeTaskClass, sleeper=lambda _: None)

    _assert_sealed(receipt)
    assert receipt["status"] == "reconciled"
    assert receipt["summary"]["mutated_now"] == 26
    assert all(task.get_archived() for task in map(_orphan, reconcile.FORMAL_SUBJECT_ORDER))
    assert not any(task.get_archived() for task in map(_keep, reconcile.FORMAL_SUBJECT_ORDER))
    mutated = [task_id for event, task_id in _FakeTaskClass.events if event == "set_archived"]
    assert set(mutated) == set(_FakeTaskClass.orphan_ids.values())
    assert len(mutated) == 26
    assert all(
        item["final_authority"]["archived"]
        for item in receipt["tasks"]
        if item["plan_binding"] == "orphan"
    )


@pytest.mark.parametrize("mode", ["raise_after_accept", "false_after_accept"])
def test_archive_callback_failure_is_success_only_after_authority_accepts(mode: str) -> None:
    task = _orphan()
    task.archive_mode = mode

    before, after, disposition = reconcile._archive_with_readback(
        _FakeTaskClass,
        task_id=task.id,
        name=task.name,
        sleeper=lambda _: None,
    )

    assert before["archived"] is False
    assert after["archived"] is True
    assert disposition == "archived_authority_confirmed_after_callback_error"


def test_archive_readback_failure_fails_closed() -> None:
    task = _orphan()
    task.archive_mode = "no_effect"

    with pytest.raises(reconcile.ReconciliationError, match="readback failed"):
        reconcile._archive_with_readback(
            _FakeTaskClass,
            task_id=task.id,
            name=task.name,
            sleeper=lambda _: None,
        )


@pytest.mark.parametrize("unsafe_status", ["queued", "in_progress", "completed"])
def test_unsafe_orphan_status_rejects_all_mutation(unsafe_status: str) -> None:
    task = _orphan()
    task.status = unsafe_status
    task.data.status = unsafe_status

    with pytest.raises(reconcile.ReconciliationError, match="not created"):
        reconcile._execute_receipt(_FakeTaskClass, sleeper=lambda _: None)

    assert not any(event[0] == "set_archived" for event in _FakeTaskClass.events)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda task: setattr(task.data.execution, "queue", "queue-id"), "output evidence"),
        (lambda task: task.artifacts.update({"result": object()}), "output evidence"),
        (
            lambda task: task.models["output"].append(SimpleNamespace(id="a" * 32)),
            "output evidence",
        ),
        (lambda task: task.reported_console.append("ran"), "reported output"),
    ],
)
def test_execution_or_output_evidence_rejects_orphan(
    mutation: object, message: str
) -> None:
    mutation(_orphan())

    with pytest.raises(reconcile.ReconciliationError, match=message):
        reconcile._dry_run_receipt(_FakeTaskClass)


def test_clearml_default_zero_test_split_is_not_execution_evidence() -> None:
    task = _orphan()
    task.data.execution.test_split = 0

    receipt = reconcile._dry_run_receipt(_FakeTaskClass)

    record = next(
        item for item in receipt["tasks"] if item["task_id"] == task.id
    )
    assert record["pre_authority"]["execution_without_queue"] == {}


def test_invalid_plan_seal_rejected() -> None:
    artifact = _FakeTaskClass.registry[reconcile.W3_TASK_ID].artifacts[
        reconcile.PLAN_ARTIFACT
    ]
    artifact.value["protocol_id"] = "drift"

    with pytest.raises(reconcile.ReconciliationError, match="seal is invalid"):
        reconcile._preflight(_FakeTaskClass)


def test_inventory_must_be_exactly_two_per_subject_and_52_total() -> None:
    _FakeTaskClass.registry.pop(_FakeTaskClass.orphan_ids["ffnet"])

    with pytest.raises(reconcile.ReconciliationError, match="exactly 52"):
        reconcile._preflight(_FakeTaskClass)


def test_archived_orphan_is_idempotent_but_keep_may_never_be_archived() -> None:
    _orphan().system_tags.append(reconcile.ARCHIVED_TAG)
    receipt = reconcile._dry_run_receipt(_FakeTaskClass)
    assert receipt["summary"]["already_archived"] == 1
    _keep().system_tags.append(reconcile.ARCHIVED_TAG)

    with pytest.raises(reconcile.ReconciliationError, match="unexpectedly archived"):
        reconcile._preflight(_FakeTaskClass)


def test_non_archive_state_change_during_callback_is_rejected() -> None:
    task = _orphan()

    def unsafe_archive(archive: bool) -> bool:
        assert archive is True
        task.system_tags.append(reconcile.ARCHIVED_TAG)
        task.tags.append("drift")
        return True

    task.set_archived = unsafe_archive  # type: ignore[method-assign]
    with pytest.raises(reconcile.ReconciliationError, match="readback failed"):
        reconcile._archive_with_readback(
            _FakeTaskClass,
            task_id=task.id,
            name=task.name,
            sleeper=lambda _: None,
        )


def test_write_new_receipt_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    receipt = {"sealed": True}
    reconcile._write_new_receipt(path, receipt)
    assert json.loads(path.read_text(encoding="utf-8")) == receipt

    with pytest.raises(FileExistsError):
        reconcile._write_new_receipt(path, receipt)


def test_wrong_execute_token_fails_before_clearml_import() -> None:
    with pytest.raises(reconcile.ReconciliationError, match="exact execute token"):
        reconcile.main(["--execute", "--execute-token", "wrong"])


def test_execute_failure_exposes_sealed_partial_receipt() -> None:
    _orphan().status = "completed"
    with pytest.raises(reconcile.ReconciliationError) as raised:
        reconcile._execute_receipt(_FakeTaskClass, sleeper=lambda _: None)
    receipt = getattr(raised.value, "sealed_receipt")
    _assert_sealed(receipt)
    assert receipt["status"] == "failed_closed"
    assert receipt["remote_state_may_have_changed"] is False
