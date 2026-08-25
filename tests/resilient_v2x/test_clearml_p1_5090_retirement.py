from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import io
import inspect
import json
import re
import unittest
from types import SimpleNamespace
from unittest import mock

from tools.resilient_v2x import clearml_p1_5090_retirement as retirement
from tools.resilient_v2x import clearml_p1_multiseed_executor as legacy_executor


class _Artifact:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = copy.deepcopy(value)

    def get(self) -> dict[str, object]:
        return copy.deepcopy(self.value)


class _Task:
    def __init__(
        self,
        *,
        task_id: str,
        name: str,
        parent: str,
        tags: list[str],
        status: str,
        queue: str,
        events: list[str],
    ) -> None:
        self.id = task_id
        self.name = name
        self.parent = parent
        self.project = legacy_executor.PROJECT_ID
        self.tags = list(tags)
        self.status = status
        self.archived = False
        self.artifacts: dict[str, _Artifact] = {}
        self.output_models: list[object] = []
        self.data = SimpleNamespace(
            parent=parent,
            project=self.project,
            last_worker="",
            execution=SimpleNamespace(queue=queue),
        )
        self.events = events
        self.mark_failed_calls = 0
        self.claim_on_add_tags = False

    def get_tags(self) -> list[str]:
        return list(self.tags)

    def add_tags(self, tags: list[str]) -> None:
        self.events.append(f"tags:{self.id}")
        if self.claim_on_add_tags:
            self.claim_on_add_tags = False
            self.status = "in_progress"
            self.data.execution.queue = legacy_executor.QUEUE_ID
            self.data.last_worker = "worker-5090:gpu0,1,2,3"
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

    def get_archived(self) -> bool:
        return self.archived

    def get_models(self) -> dict[str, list[object]]:
        return {"output": list(self.output_models)}

    def upload_artifact(
        self,
        *,
        name: str,
        artifact_object: dict[str, object],
        wait_on_upload: bool,
    ) -> bool:
        if not wait_on_upload:
            raise AssertionError("retirement artifacts must be synchronous")
        self.events.append(f"artifact:{name}")
        self.artifacts[name] = _Artifact(artifact_object)
        return True

    def mark_stopped(self, *, force: bool, status_message: str) -> None:
        del force, status_message
        self.events.append(f"stop:{self.id}")
        raise AssertionError("quarantined-created retirement must never stop tasks")

    def mark_failed(self, **kwargs: object) -> None:
        del kwargs
        self.mark_failed_calls += 1
        raise AssertionError("expected retirement must not mark tasks failed")

    def set_archived(self, value: bool) -> None:
        del value
        self.events.append(f"archive:{self.id}")
        raise AssertionError("quarantined-created retirement must never archive tasks")


class _TaskClass:
    tasks: dict[str, _Task] = {}
    events: list[str] = []
    queue_entries: list[str] = []
    dequeue_mode = "success"

    @classmethod
    def get_task(cls, *, task_id: str) -> _Task:
        return cls.tasks[task_id]

    @classmethod
    def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
        rows = list(cls.tasks.values())
        task_name = kwargs.get("task_name")
        if task_name is not None:
            rows = [task for task in rows if re.match(str(task_name), task.name)]
        tags = kwargs.get("tags")
        if tags is not None:
            required = [tag for tag in tags if tag != "__$all"]
            rows = [task for task in rows if all(tag in task.tags for tag in required)]
        task_filter = kwargs.get("task_filter")
        if isinstance(task_filter, dict) and "parent" in task_filter:
            rows = [task for task in rows if task.parent == task_filter["parent"]]
        return [
            {
                "id": task.id,
                "name": task.name,
                "parent": task.parent,
                "project": task.project,
                "status": task.status,
                "tags": list(task.tags),
            }
            for task in rows
        ]

    @classmethod
    def dequeue(cls, *, task: _Task) -> dict[str, int]:
        cls.events.append("dequeue")
        if cls.dequeue_mode == "race_to_running":
            task.status = "in_progress"
            task.data.last_worker = "worker-5090:gpu0,1,2,3"
            cls.queue_entries = [item for item in cls.queue_entries if item != task.id]
            raise RuntimeError("worker claimed task before dequeue committed")
        task.status = "created"
        cls.queue_entries = [item for item in cls.queue_entries if item != task.id]
        if cls.dequeue_mode == "lost_response":
            raise RuntimeError("response transport failed after commit")
        return {"dequeued": 1, "updated": 1}

    @classmethod
    def queue_snapshot(cls) -> dict[str, object]:
        entries = sorted(cls.queue_entries)
        return {
            "queue_id": legacy_executor.QUEUE_ID,
            "queue_name": legacy_executor.QUEUE_NAME,
            "entry_task_ids": entries,
            "entry_count": len(entries),
        }


def _journal() -> dict[str, object]:
    plan = legacy_executor.build_plan()
    journal = legacy_executor._new_journal(
        plan=plan,
        execution_key=retirement.EXECUTION_KEY,
        controller_id=retirement.CONTROLLER_ID,
    )
    row = next(
        item
        for item in journal["tasks"]
        if item["task_key"] == retirement.CHILD_TASK_KEY
    )
    row.update(
        {
            "task_id": retirement.CHILD_ID,
            "state": "active",
            "server_status": "queued",
        }
    )
    journal["seal_sha256"] = legacy_executor._seal(journal)
    return journal


def _state() -> tuple[_Task, _Task, list[str]]:
    events: list[str] = []
    execution_tag = legacy_executor._execution_tag(retirement.EXECUTION_KEY)
    controller = _Task(
        task_id=retirement.CONTROLLER_ID,
        name=legacy_executor._controller_name(retirement.EXECUTION_KEY),
        parent=legacy_executor.SELECTOR_TASK_ID,
        tags=[execution_tag, legacy_executor.CONTROLLER_TAG],
        status="created",
        queue="",
        events=events,
    )
    controller.artifacts = {
        "p1_multiseed_plan": _Artifact(legacy_executor.build_plan()),
        "p1_multiseed_pinset": _Artifact(legacy_executor.build_pinset()),
        legacy_executor.JOURNAL_ARTIFACT: _Artifact(_journal()),
    }
    child = _Task(
        task_id=retirement.CHILD_ID,
        name=legacy_executor._task_name(
            retirement.EXECUTION_KEY, retirement.CHILD_TASK_KEY
        ),
        parent=retirement.CONTROLLER_ID,
        tags=[execution_tag, f"p1-task-key:{retirement.CHILD_TASK_KEY}"],
        status="queued",
        queue=legacy_executor.QUEUE_ID,
        events=events,
    )
    _TaskClass.tasks = {controller.id: controller, child.id: child}
    _TaskClass.events = events
    _TaskClass.queue_entries = [child.id]
    _TaskClass.dequeue_mode = "success"
    return controller, child, events


class _Harness:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.mutex_active = True

    def lease(self, execution_key: str, *, api_client: object) -> dict[str, object]:
        del api_client
        if execution_key != retirement.EXECUTION_KEY:
            raise AssertionError("unexpected execution key")
        return {
            "name": legacy_executor._lease_queue_name(execution_key),
            "status": "held" if self.mutex_active else "available",
            "queue_id": (
                retirement.LEGACY_MUTEX_QUEUE_ID if self.mutex_active else None
            ),
            "owner_tag": "owner:" + "a" * 32 if self.mutex_active else None,
        }

    @contextlib.contextmanager
    def mutex(self, execution_key: str, *, api_client: object):
        del api_client
        if execution_key != retirement.EXECUTION_KEY:
            raise AssertionError("unexpected execution key")
        self.events.append("mutex:enter")
        self.mutex_active = True
        try:
            yield {
                "queue_id": retirement.LEGACY_MUTEX_QUEUE_ID,
                "name": legacy_executor._lease_queue_name(execution_key),
                "owner_tag": "owner:" + "a" * 32,
            }
        finally:
            self.events.append("mutex:release")
            self.mutex_active = False

    def journal(self, _legacy: object) -> dict[str, object]:
        if self.mutex_active:
            return {
                "path": "/tmp/sealed-journal",
                "status": "present",
                "queue_id": retirement.LEGACY_MUTEX_QUEUE_ID,
            }
        return {"path": "/tmp/sealed-journal", "status": "absent"}


class ClearMLP15090RetirementTests(unittest.TestCase):
    def test_retirement_source_has_no_stop_or_archive_api_call(self) -> None:
        tree = ast.parse(inspect.getsource(retirement))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        forbidden = {"mark_stopped", "set_archived"}
        self.assertTrue(
            forbidden.isdisjoint(called_attributes),
            f"forbidden terminal task API call found: {forbidden & called_attributes}",
        )

    def test_default_mode_is_local_and_does_not_import_clearml(self) -> None:
        real_import = builtins.__import__

        def guarded(name: str, *args: object, **kwargs: object) -> object:
            if name == "clearml" or name.startswith("clearml."):
                raise AssertionError("dry-run imported ClearML")
            return real_import(name, *args, **kwargs)

        output = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=guarded):
            with contextlib.redirect_stdout(output):
                self.assertEqual(retirement.main([]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["remote_mutation_count"], 0)
        self.assertEqual(plan["seal_sha256"], retirement._seal(plan))
        self.assertEqual(
            plan["bindings"]["controller_task_id"], retirement.CONTROLLER_ID
        )
        self.assertIn(
            "upload sealed intent before any controller tag or dequeue mutation",
            plan["fail_closed_guarantees"],
        )
        self.assertIn(
            "never invoke task stop or archive APIs",
            plan["fail_closed_guarantees"],
        )

    def test_invalid_token_fails_before_loading_legacy_or_remote(self) -> None:
        with mock.patch.object(
            retirement, "_load_legacy", side_effect=AssertionError("loaded")
        ):
            with self.assertRaisesRegex(
                retirement.P15090RetirementError, "token mismatch"
            ):
                retirement.retire(authorization_token="wrong")

    def test_a100_script_hash_and_derived_seals_are_exactly_pinned(self) -> None:
        contract = retirement._load_a100_target_contract()
        self.assertEqual(
            contract,
            {
                "plan_seal_sha256": retirement.A100_PLAN_SEAL_SHA256,
                "pinset_seal_sha256": retirement.A100_PINSET_SEAL_SHA256,
                "execution_key": retirement.A100_EXECUTION_KEY,
                "executor_sha256": retirement.A100_EXECUTOR_SHA256,
            },
        )
        with mock.patch.object(retirement, "A100_EXECUTOR_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                retirement.P15090RetirementError, "A100 executor SHA-256"
            ):
                retirement._load_a100_target_contract()

    def test_intent_and_completion_are_distinct_sealed_artifacts(self) -> None:
        original = [
            "p1_multiseed_execution_journal",
            "p1_multiseed_pinset",
            "p1_multiseed_plan",
        ]
        intent = retirement._retirement_intent(
            original_controller_artifacts=original,
        )
        completed = retirement._retirement_manifest(
            original_controller_artifacts=original,
        )
        self.assertEqual(intent["phase"], "retiring")
        self.assertEqual(completed["phase"], "retired")
        self.assertEqual(intent["status"], "retiring")
        self.assertEqual(intent["seal_sha256"], retirement._seal(intent))
        self.assertEqual(completed["seal_sha256"], retirement._seal(completed))
        self.assertEqual(
            intent["target_a100_execution_key"],
            retirement.A100_EXECUTION_KEY,
        )
        self.assertEqual(
            completed["retirement_intent_seal_sha256"],
            intent["seal_sha256"],
        )
        self.assertEqual(
            completed["intent_target_a100_executor_sha256"],
            retirement.INTENT_TARGET_A100_EXECUTOR_SHA256,
        )
        self.assertEqual(
            completed["final_target_a100_executor_sha256"],
            retirement.A100_EXECUTOR_SHA256,
        )
        self.assertEqual(
            completed["target_a100_executor_amendment_reason"],
            "clearml_dequeue_retains_historical_execution_queue",
        )
        self.assertEqual(
            completed["retirement_proofs"]["queue_residency_authority"],
            "clearml_queue_entries",
        )
        self.assertEqual(
            completed["retirement_proofs"]["child_historical_execution_queue_id"],
            legacy_executor.QUEUE_ID,
        )
        retirement._validate_intent(intent)
        retirement._validate_manifest(completed)

    def test_full_retirement_orders_intent_before_barrier_and_dequeue(self) -> None:
        controller, child, events = _state()
        harness = _Harness(events)

        def full_preflight(**kwargs: object) -> dict[str, object]:
            del kwargs
            events.append("legacy:preflight")
            return {"status": "passed"}

        with (
            mock.patch.object(legacy_executor, "preflight", side_effect=full_preflight),
            mock.patch.object(
                legacy_executor, "_read_lease_snapshot", side_effect=harness.lease
            ),
            mock.patch.object(
                retirement, "_mutex_journal_snapshot", side_effect=harness.journal
            ),
            mock.patch.object(
                retirement,
                "_legacy_local_lock",
                return_value=contextlib.nullcontext(),
            ),
        ):
            receipt = retirement.retire(
                authorization_token=retirement.RETIREMENT_TOKEN,
                legacy=legacy_executor,
                task_class=_TaskClass,
                api_client=object(),
                server_mutex=harness.mutex,
                a100_validator=lambda: {},
                queue_reader=_TaskClass.queue_snapshot,
            )

        self.assertEqual(receipt["status"], "retired")
        self.assertEqual(controller.status, "created")
        self.assertEqual(child.status, "created")
        self.assertEqual(child.data.execution.queue, legacy_executor.QUEUE_ID)
        self.assertEqual(receipt["legacy_queue"]["entry_task_ids"], [])
        self.assertEqual(
            receipt["child"]["historical_execution_queue_id"],
            legacy_executor.QUEUE_ID,
        )
        self.assertFalse(receipt["child"]["queue_resident"])
        self.assertFalse(controller.archived)
        self.assertFalse(child.archived)
        self.assertTrue(
            {retirement.RETIREMENT_TAG, legacy_executor.ORPHAN_TAG}.issubset(
                controller.tags
            )
        )
        self.assertTrue(
            {retirement.RETIREMENT_TAG, legacy_executor.ORPHAN_TAG}.issubset(child.tags)
        )
        self.assertEqual(controller.mark_failed_calls + child.mark_failed_calls, 0)
        self.assertLess(
            events.index(f"artifact:{retirement.RETIREMENT_INTENT_ARTIFACT}"),
            events.index(f"tags:{retirement.CONTROLLER_ID}"),
        )
        self.assertLess(
            events.index(f"tags:{retirement.CONTROLLER_ID}"),
            events.index("dequeue"),
        )
        self.assertLess(
            events.index(f"tags:{retirement.CHILD_ID}"),
            events.index(f"artifact:{retirement.RETIREMENT_ARTIFACT}"),
        )
        self.assertFalse(any(event.startswith("stop:") for event in events))
        self.assertFalse(any(event.startswith("archive:") for event in events))
        self.assertEqual(events[-1], "mutex:release")

    def test_lost_dequeue_response_is_accepted_only_after_exact_fresh_state(
        self,
    ) -> None:
        _, child, _ = _state()
        _TaskClass.dequeue_mode = "lost_response"
        result = retirement._dequeue_child(
            _TaskClass,
            legacy_executor,
            child,
            queue_reader=_TaskClass.queue_snapshot,
        )
        self.assertEqual(result.status, "created")
        self.assertEqual(result.data.execution.queue, legacy_executor.QUEUE_ID)
        self.assertEqual(result.data.last_worker, "")

    def test_created_child_with_historical_queue_id_and_no_entry_skips_dequeue(
        self,
    ) -> None:
        _, child, _ = _state()
        child.status = "created"
        _TaskClass.queue_entries = []
        with mock.patch.object(
            _TaskClass,
            "dequeue",
            side_effect=AssertionError(
                "created nonresident child must not dequeue again"
            ),
        ):
            result = retirement._dequeue_child(
                _TaskClass,
                legacy_executor,
                child,
                queue_reader=_TaskClass.queue_snapshot,
            )
        self.assertIs(result, child)
        self.assertEqual(result.status, "created")
        self.assertEqual(result.data.execution.queue, legacy_executor.QUEUE_ID)
        self.assertNotIn(result.id, _TaskClass.queue_entries)

    def test_queue_entries_are_authoritative_over_task_status_and_history(
        self,
    ) -> None:
        for status, entries, historical_queue in (
            ("created", [retirement.CHILD_ID], legacy_executor.QUEUE_ID),
            ("queued", [], legacy_executor.QUEUE_ID),
            ("created", [], ""),
        ):
            with self.subTest(
                status=status,
                entries=entries,
                historical_queue=historical_queue,
            ):
                _, child, _ = _state()
                child.status = status
                child.data.execution.queue = historical_queue
                _TaskClass.queue_entries = list(entries)
                expected = (
                    "historical execution.queue"
                    if historical_queue != legacy_executor.QUEUE_ID
                    else "authoritative queue-entry residency"
                )
                with self.assertRaisesRegex(retirement.P15090RetirementError, expected):
                    retirement._inventory(
                        legacy_executor,
                        _TaskClass,
                        queue_reader=_TaskClass.queue_snapshot,
                    )

    def test_worker_claim_race_fails_without_stopping_or_archiving_child(self) -> None:
        controller, child, events = _state()
        harness = _Harness(events)
        _TaskClass.dequeue_mode = "race_to_running"
        with (
            mock.patch.object(
                legacy_executor,
                "preflight",
                return_value={"status": "passed"},
            ),
            mock.patch.object(
                legacy_executor, "_read_lease_snapshot", side_effect=harness.lease
            ),
            mock.patch.object(
                retirement, "_mutex_journal_snapshot", side_effect=harness.journal
            ),
            mock.patch.object(
                retirement,
                "_legacy_local_lock",
                return_value=contextlib.nullcontext(),
            ),
        ):
            with self.assertRaisesRegex(
                retirement.P15090RetirementError, "dequeue did not produce"
            ):
                retirement.retire(
                    authorization_token=retirement.RETIREMENT_TOKEN,
                    legacy=legacy_executor,
                    task_class=_TaskClass,
                    api_client=object(),
                    server_mutex=harness.mutex,
                    a100_validator=lambda: {},
                    queue_reader=_TaskClass.queue_snapshot,
                )
        self.assertEqual(child.status, "in_progress")
        self.assertNotIn(f"stop:{retirement.CHILD_ID}", events)
        self.assertNotIn(f"archive:{retirement.CHILD_ID}", events)
        self.assertNotIn(f"stop:{retirement.CONTROLLER_ID}", events)
        self.assertNotIn(retirement.RETIREMENT_ARTIFACT, controller.artifacts)
        self.assertIn(retirement.RETIREMENT_INTENT_ARTIFACT, controller.artifacts)

    def test_post_dequeue_worker_claim_race_never_stops_or_archives_tasks(self) -> None:
        controller, child, events = _state()
        harness = _Harness(events)
        child.claim_on_add_tags = True
        with (
            mock.patch.object(
                legacy_executor,
                "preflight",
                return_value={"status": "passed"},
            ),
            mock.patch.object(
                legacy_executor, "_read_lease_snapshot", side_effect=harness.lease
            ),
            mock.patch.object(
                retirement, "_mutex_journal_snapshot", side_effect=harness.journal
            ),
            mock.patch.object(
                retirement,
                "_legacy_local_lock",
                return_value=contextlib.nullcontext(),
            ),
        ):
            with self.assertRaisesRegex(
                retirement.P15090RetirementError, "acquired a worker"
            ):
                retirement.retire(
                    authorization_token=retirement.RETIREMENT_TOKEN,
                    legacy=legacy_executor,
                    task_class=_TaskClass,
                    api_client=object(),
                    server_mutex=harness.mutex,
                    a100_validator=lambda: {},
                    queue_reader=_TaskClass.queue_snapshot,
                )
        self.assertIn("dequeue", events)
        self.assertEqual(child.status, "in_progress")
        self.assertEqual(child.data.last_worker, "worker-5090:gpu0,1,2,3")
        self.assertFalse(child.archived)
        self.assertEqual(controller.status, "created")
        self.assertFalse(controller.archived)
        self.assertFalse(any(event.startswith("stop:") for event in events))
        self.assertFalse(any(event.startswith("archive:") for event in events))
        self.assertNotIn(retirement.RETIREMENT_ARTIFACT, controller.artifacts)
        self.assertIn(retirement.RETIREMENT_INTENT_ARTIFACT, controller.artifacts)

    def test_resume_from_durable_intent_skips_legacy_preflight(self) -> None:
        controller, _, events = _state()
        harness = _Harness(events)
        intent = retirement._retirement_intent(
            original_controller_artifacts=sorted(controller.artifacts),
        )
        controller.artifacts[retirement.RETIREMENT_INTENT_ARTIFACT] = _Artifact(intent)
        with (
            mock.patch.object(
                legacy_executor,
                "preflight",
                side_effect=AssertionError("legacy preflight rejects intent artifact"),
            ),
            mock.patch.object(
                legacy_executor, "_read_lease_snapshot", side_effect=harness.lease
            ),
            mock.patch.object(
                retirement, "_mutex_journal_snapshot", side_effect=harness.journal
            ),
            mock.patch.object(
                retirement,
                "_legacy_local_lock",
                return_value=contextlib.nullcontext(),
            ),
        ):
            receipt = retirement.retire(
                authorization_token=retirement.RETIREMENT_TOKEN,
                legacy=legacy_executor,
                task_class=_TaskClass,
                api_client=object(),
                server_mutex=harness.mutex,
                a100_validator=lambda: {},
                queue_reader=_TaskClass.queue_snapshot,
            )
        self.assertEqual(receipt["status"], "retired")
        self.assertEqual(receipt["controller"]["status"], "created")
        self.assertEqual(receipt["child"]["status"], "created")
        self.assertFalse(receipt["controller"]["archived"])
        self.assertFalse(receipt["child"]["archived"])
        self.assertEqual(
            events.count(f"artifact:{retirement.RETIREMENT_INTENT_ARTIFACT}"), 0
        )
        self.assertFalse(any(event.startswith("stop:") for event in events))
        self.assertFalse(any(event.startswith("archive:") for event in events))

    def test_resume_after_dequeue_before_child_tags_completes_safely(
        self,
    ) -> None:
        controller, child, events = _state()
        harness = _Harness(events)
        harness.mutex_active = False
        intent = retirement._retirement_intent(
            original_controller_artifacts=sorted(controller.artifacts),
        )
        controller.artifacts[retirement.RETIREMENT_INTENT_ARTIFACT] = _Artifact(intent)
        controller.tags.extend([retirement.RETIREMENT_TAG, legacy_executor.ORPHAN_TAG])
        child.status = "created"
        _TaskClass.queue_entries = []
        with (
            mock.patch.object(
                legacy_executor,
                "preflight",
                side_effect=AssertionError("resume must not call legacy preflight"),
            ),
            mock.patch.object(
                legacy_executor, "_read_lease_snapshot", side_effect=harness.lease
            ),
            mock.patch.object(
                retirement, "_mutex_journal_snapshot", side_effect=harness.journal
            ),
            mock.patch.object(
                retirement,
                "_legacy_local_lock",
                return_value=contextlib.nullcontext(),
            ),
        ):
            receipt = retirement.retire(
                authorization_token=retirement.RETIREMENT_TOKEN,
                legacy=legacy_executor,
                task_class=_TaskClass,
                api_client=object(),
                server_mutex=harness.mutex,
                a100_validator=lambda: {},
                queue_reader=_TaskClass.queue_snapshot,
            )
        self.assertEqual(receipt["status"], "retired")
        self.assertNotIn("dequeue", events)
        self.assertEqual(events.count(f"tags:{retirement.CHILD_ID}"), 1)
        self.assertNotIn(f"tags:{retirement.CONTROLLER_ID}", events)
        self.assertFalse(any(event.startswith("stop:") for event in events))
        self.assertFalse(any(event.startswith("archive:") for event in events))
        self.assertTrue(
            {retirement.RETIREMENT_TAG, legacy_executor.ORPHAN_TAG}.issubset(child.tags)
        )
        self.assertIn(retirement.RETIREMENT_ARTIFACT, controller.artifacts)

    def test_preflight_accepts_sealed_post_dequeue_intermediate_state_read_only(
        self,
    ) -> None:
        controller, child, events = _state()
        intent = retirement._retirement_intent(
            original_controller_artifacts=sorted(controller.artifacts),
        )
        controller.artifacts[retirement.RETIREMENT_INTENT_ARTIFACT] = _Artifact(intent)
        controller.tags.extend([retirement.RETIREMENT_TAG, legacy_executor.ORPHAN_TAG])
        child.status = "created"
        _TaskClass.queue_entries = []
        lease = {
            "name": legacy_executor._lease_queue_name(retirement.EXECUTION_KEY),
            "status": "available",
            "queue_id": None,
            "owner_tag": None,
        }
        with (
            mock.patch.object(
                retirement,
                "_local_lock_snapshot",
                return_value={
                    "path": "/tmp/legacy.lock",
                    "status": "available",
                    "acquirable": True,
                    "reason": None,
                },
            ),
            mock.patch.object(
                retirement,
                "_mutex_journal_snapshot",
                return_value={"path": "/tmp/journal", "status": "absent"},
            ),
            mock.patch.object(
                legacy_executor,
                "_read_lease_snapshot",
                return_value=lease,
            ),
            mock.patch.object(
                legacy_executor,
                "preflight",
                side_effect=AssertionError(
                    "sealed intermediate skips legacy preflight"
                ),
            ),
        ):
            receipt = retirement.preflight(
                legacy=legacy_executor,
                task_class=_TaskClass,
                api_client=object(),
                a100_validator=lambda: {},
                queue_reader=_TaskClass.queue_snapshot,
            )
        self.assertEqual(receipt["status"], "validated")
        self.assertTrue(receipt["ready_for_retirement"])
        self.assertTrue(receipt["controller"]["barrier_installed"])
        self.assertFalse(receipt["child"]["retirement_tagged"])
        self.assertEqual(receipt["child"]["status"], "created")
        self.assertEqual(
            receipt["child"]["historical_execution_queue_id"],
            legacy_executor.QUEUE_ID,
        )
        self.assertFalse(receipt["child"]["queue_resident"])
        self.assertEqual(
            receipt["legacy_queue"],
            {
                "queue_id": legacy_executor.QUEUE_ID,
                "queue_name": legacy_executor.QUEUE_NAME,
                "entry_task_ids": [],
                "entry_count": 0,
            },
        )
        self.assertEqual(receipt["remote_mutation_count"], 0)
        self.assertEqual(events, [])

    def test_preflight_is_read_only_and_reports_held_legacy_lock(self) -> None:
        _, _, events = _state()
        lease = {
            "name": legacy_executor._lease_queue_name(retirement.EXECUTION_KEY),
            "status": "held",
            "queue_id": retirement.LEGACY_MUTEX_QUEUE_ID,
            "owner_tag": "owner:" + "a" * 32,
        }
        with (
            mock.patch.object(
                retirement,
                "_local_lock_snapshot",
                return_value={
                    "path": "/tmp/legacy.lock",
                    "status": "held",
                    "acquirable": False,
                    "reason": "legacy executor still owns the local lock",
                },
            ),
            mock.patch.object(
                retirement,
                "_mutex_journal_snapshot",
                return_value={
                    "path": "/tmp/journal",
                    "status": "present",
                    "queue_id": retirement.LEGACY_MUTEX_QUEUE_ID,
                },
            ),
            mock.patch.object(
                legacy_executor,
                "_read_lease_snapshot",
                return_value=lease,
            ),
            mock.patch.object(
                legacy_executor,
                "preflight",
                return_value={"status": "passed"},
            ),
        ):
            receipt = retirement.preflight(
                legacy=legacy_executor,
                task_class=_TaskClass,
                api_client=object(),
                a100_validator=lambda: {},
                queue_reader=_TaskClass.queue_snapshot,
            )
        self.assertTrue(receipt["readonly"])
        self.assertEqual(receipt["remote_mutation_count"], 0)
        self.assertFalse(receipt["ready_for_retirement"])
        self.assertIn("legacy executor", receipt["blockers"][0])
        self.assertEqual(events, [])

    def test_delete_committed_stale_local_mutex_journal_is_recoverable(self) -> None:
        lease = {
            "name": retirement.LEGACY_MUTEX_NAME,
            "status": "available",
            "queue_id": None,
            "owner_tag": None,
        }
        journal = {
            "path": "/tmp/journal",
            "status": "present",
            "queue_id": retirement.LEGACY_MUTEX_QUEUE_ID,
        }
        with (
            mock.patch.object(legacy_executor, "_remove_mutex_receipt") as remove,
            mock.patch.object(
                retirement,
                "_mutex_journal_snapshot",
                return_value={"path": "/tmp/journal", "status": "absent"},
            ),
        ):
            observed = retirement._reconcile_local_mutex_journal(
                legacy_executor, lease=lease, journal=journal
            )
        remove.assert_called_once_with()
        self.assertEqual(observed["status"], "absent")

    def test_held_mutex_without_matching_journal_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            retirement.P15090RetirementError, "no matching sealed local journal"
        ):
            retirement._reconcile_local_mutex_journal(
                legacy_executor,
                lease={
                    "status": "held",
                    "queue_id": retirement.LEGACY_MUTEX_QUEUE_ID,
                },
                journal={"status": "absent"},
            )

    def test_duplicate_stable_child_fails_closed(self) -> None:
        _, child, _ = _state()
        duplicate = copy.copy(child)
        duplicate.id = "f" * 32
        _TaskClass.tasks[duplicate.id] = duplicate
        with self.assertRaisesRegex(
            retirement.P15090RetirementError, "stable child inventory"
        ):
            retirement._inventory(
                legacy_executor,
                _TaskClass,
                queue_reader=_TaskClass.queue_snapshot,
            )


if __name__ == "__main__":
    unittest.main()
