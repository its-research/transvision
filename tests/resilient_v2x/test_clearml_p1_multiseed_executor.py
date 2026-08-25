from __future__ import annotations

import builtins
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.resilient_v2x import clearml_p1_multiseed_executor as executor


class _FakeTask:
    def __init__(self) -> None:
        self.id = "a" * 32
        self.status = "created"
        self.data = SimpleNamespace(
            execution=SimpleNamespace(queue=""),
        )


class _FakeTaskClass:
    enqueue_calls: list[dict[str, object]] = []
    task = _FakeTask()

    @classmethod
    def get_task(cls, *, task_id: str) -> _FakeTask:
        if task_id != cls.task.id:
            raise AssertionError("unexpected task ID")
        return cls.task

    @classmethod
    def enqueue(cls, *, task: _FakeTask, queue_id: str, force: bool) -> dict[str, int]:
        cls.enqueue_calls.append({"task": task, "queue_id": queue_id, "force": force})
        task.status = "queued"
        task.data.execution.queue = queue_id
        return {"queued": 1, "updated": 1}


class _QueueService:
    def __init__(self) -> None:
        self.rows: dict[str, SimpleNamespace] = {}
        self.create_calls: list[dict[str, object]] = []
        self.deleted: list[str] = []

    def create(self, *, name: str, tags: list[str]) -> SimpleNamespace:
        self.create_calls.append({"name": name, "tags": list(tags)})
        if name in self.rows:
            raise RuntimeError("ExpectedUniqueData")
        row = SimpleNamespace(
            id=(f"{len(self.rows) + 1:032x}"),
            name=name,
            tags=list(tags),
            entries=[],
        )
        self.rows[name] = row
        return row

    def get_all(self, **kwargs: object) -> list[SimpleNamespace]:
        pattern = str(kwargs["name"])
        return [
            row
            for row in self.rows.values()
            if __import__("re").match(pattern, row.name)
        ]

    def get_by_id(self, *, queue: str, max_task_entries: int) -> SimpleNamespace:
        del max_task_entries
        return next(row for row in self.rows.values() if row.id == queue)

    def get_num_entries(self, *, queue: str) -> SimpleNamespace:
        row = self.get_by_id(queue=queue, max_task_entries=1)
        return SimpleNamespace(num=len(row.entries))

    def delete(self, *, queue: str, force: bool) -> SimpleNamespace:
        if force:
            raise AssertionError("mutex deletion must use force=False")
        name = next(name for name, row in self.rows.items() if row.id == queue)
        del self.rows[name]
        self.deleted.append(queue)
        return SimpleNamespace(deleted=1)


class _APIClient:
    def __init__(self) -> None:
        self.queues = _QueueService()


class _OrphanTask:
    def __init__(self) -> None:
        self.id = "b" * 32
        self.status = "created"
        self.tags: list[str] = []
        self.archived = False
        self.mark_failed_calls = 0
        self.archive_calls = 0
        self.data = SimpleNamespace(execution=SimpleNamespace(queue=""))

    def get_tags(self) -> list[str]:
        return list(self.tags)

    def add_tags(self, tags: list[str]) -> None:
        self.tags.extend(tags)

    def mark_failed(self, **kwargs: object) -> None:
        self.mark_failed_calls += 1
        self.status = "failed"

    def set_archived(self, value: bool) -> None:
        self.archive_calls += 1
        self.archived = value

    def get_archived(self) -> bool:
        return self.archived


class _OrphanTaskClass:
    task = _OrphanTask()

    @classmethod
    def get_task(cls, *, task_id: str) -> _OrphanTask:
        if task_id != cls.task.id:
            raise AssertionError("unexpected orphan task ID")
        return cls.task


class ClearMLP1MultiseedExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeTaskClass.enqueue_calls = []
        _FakeTaskClass.task = _FakeTask()

    def test_default_mode_is_read_only_and_does_not_import_clearml(self) -> None:
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "clearml" or name.startswith("clearml."):
                raise AssertionError("dry-run imported ClearML")
            return real_import(name, *args, **kwargs)

        output = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=guarded_import):
            with contextlib.redirect_stdout(output):
                self.assertEqual(executor.main([]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["default_mode"], "dry_run")
        self.assertEqual(plan["scope"]["new_training_task_count"], 4)
        self.assertEqual(plan["scope"]["new_evaluation_task_count"], 4)

    def test_plan_is_exactly_two_subjects_by_two_new_seeds(self) -> None:
        plan = executor.build_plan()
        self.assertEqual(plan["seal_sha256"], executor._seal(plan))
        self.assertEqual(plan["scope"]["subjects"], list(executor.SUBJECTS))
        self.assertEqual(
            plan["scope"]["new_training_seeds"], list(executor.TRAINING_SEEDS)
        )
        self.assertEqual(len(plan["pairs"]), 4)
        training = [pair["training"] for pair in plan["pairs"]]
        evaluation = [pair["evaluation"] for pair in plan["pairs"]]
        self.assertEqual(
            {(row["subject"], row["training_seed"]) for row in training},
            {
                (subject, seed)
                for subject in executor.SUBJECTS
                for seed in executor.TRAINING_SEEDS
            },
        )
        self.assertTrue(all(row["run_count"] == 12 for row in evaluation))
        self.assertTrue(
            all(
                row["condition_ids"] == list(executor.CONDITION_IDS)
                for row in evaluation
            )
        )

    def test_scope_tampering_fails_even_after_resealing(self) -> None:
        forged = copy.deepcopy(executor.build_plan())
        forged["scope"]["subjects"].append("cobevt")
        forged["seal_sha256"] = executor._seal(forged)
        with self.assertRaisesRegex(executor.P1ExecutorError, "scope drifted"):
            executor.validate_plan(forged)

    def test_seed_contract_is_exact(self) -> None:
        executor._validate_seed_contract(executor.SOURCE_D_SEED_CONTRACT)
        forged = copy.deepcopy(executor.SOURCE_D_SEED_CONTRACT)
        forged["training_overlay_protocol_seed"] = executor.TRAINING_SEEDS[0]
        with self.assertRaisesRegex(executor.P1ExecutorError, "seed contract"):
            executor._validate_seed_contract(forged)

    def test_invalid_execute_token_stops_before_clearml_loading(self) -> None:
        with mock.patch.object(
            executor,
            "_load_clearml",
            side_effect=AssertionError("ClearML loader must not run"),
        ) as loader:
            with self.assertRaisesRegex(executor.P1ExecutorError, "exact"):
                executor.main(["--execute", "--enqueue-token", "wrong"])
        loader.assert_not_called()

    def test_enqueue_rechecks_token_and_uses_force_false(self) -> None:
        with self.assertRaisesRegex(executor.P1ExecutorError, "token mismatch"):
            executor._enqueue_once(
                _FakeTaskClass,
                _FakeTaskClass.task,
                authorization_token="wrong",
            )
        self.assertEqual(_FakeTaskClass.enqueue_calls, [])

        executor._enqueue_once(
            _FakeTaskClass,
            _FakeTaskClass.task,
            authorization_token=executor.EXECUTION_TOKEN,
        )
        self.assertEqual(len(_FakeTaskClass.enqueue_calls), 1)
        self.assertEqual(_FakeTaskClass.enqueue_calls[0]["queue_id"], executor.QUEUE_ID)
        self.assertIs(_FakeTaskClass.enqueue_calls[0]["force"], False)

    def test_script_has_no_dependency_on_the_full_matrix_executor(self) -> None:
        root = Path(__file__).resolve().parents[2]
        source = (
            root / "tools/resilient_v2x/clearml_p1_multiseed_executor.py"
        ).read_text(
            encoding="utf-8"
        )
        forbidden = "clearml_formal_" + "multiseed_executor"
        self.assertNotIn(forbidden, source)

    def test_preflight_is_machine_readable_and_calls_no_mutation_method(self) -> None:
        query_calls: list[dict[str, object]] = []

        class ReadOnlyTaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> SimpleNamespace:
                return SimpleNamespace(id=task_id)

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[object]:
                query_calls.append(dict(kwargs))
                return []

            @classmethod
            def init(cls, **kwargs: object) -> object:
                raise AssertionError("preflight called Task.init")

            @classmethod
            def clone(cls, **kwargs: object) -> object:
                raise AssertionError("preflight called Task.clone")

            @classmethod
            def enqueue(cls, **kwargs: object) -> object:
                raise AssertionError("preflight called Task.enqueue")

        queue = {
            "queue_id": executor.QUEUE_ID,
            "queue_name": executor.QUEUE_NAME,
            "queued_entry_count": 0,
            "queued_task_ids": [],
            "registered_worker_count": 1,
            "workers": [
                {
                    "worker_id": "host-5090:gpu0,1,2,3",
                    "running_task_id": None,
                    "reported_queue_task_count": 0,
                }
            ],
        }
        lease = {
            "name": executor._lease_queue_name(
                executor._execution_key(executor.build_plan(), executor.build_pinset())
            ),
            "status": "available",
            "queue_id": None,
            "owner_tag": None,
        }
        with (
            mock.patch.object(executor, "_validate_selector"),
            mock.patch.object(
                executor, "_validate_source_d", return_value="sealed-source-d"
            ),
            mock.patch.object(executor, "_validate_teacher_gate"),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(executor, "_validate_anchor_tasks"),
        ):
            receipt = executor.preflight(
                task_class=ReadOnlyTaskClass,
                queue_reader=lambda _: queue,
                lease_reader=lambda _: lease,
            )
        self.assertTrue(receipt["readonly"])
        self.assertEqual(receipt["remote_mutation_count"], 0)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["seal_sha256"], executor._seal(receipt))
        self.assertEqual(len(receipt["existing_execution"]["children"]), 8)
        self.assertEqual(len([call for call in query_calls if "task_name" in call]), 9)

    def test_named_task_query_includes_hidden_and_archived_results(self) -> None:
        calls: list[dict[str, object]] = []

        class QueryTaskClass:
            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[object]:
                calls.append(dict(kwargs))
                return []

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                raise AssertionError(f"unexpected task load {task_id}")

        self.assertEqual(
            executor._query_named_tasks(QueryTaskClass, name="stable-name"), []
        )
        self.assertTrue(calls[0]["task_filter"]["search_hidden"])
        self.assertNotIn("project", calls[0]["task_filter"])

    def test_named_task_query_rejects_fresh_name_drift(self) -> None:
        task_id = "1" * 32

        class QueryTaskClass:
            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                del kwargs
                return [{"id": task_id, "name": "stable-name"}]

            @classmethod
            def get_task(cls, *, task_id: str) -> SimpleNamespace:
                return SimpleNamespace(id=task_id, name="renamed-after-query")

        with self.assertRaisesRegex(executor.P1ExecutorError, "round-trip"):
            executor._query_named_tasks(QueryTaskClass, name="stable-name")

    def test_descendant_query_uses_scalar_parent_without_scope_filters(self) -> None:
        calls: list[dict[str, object]] = []
        controller_id = "2" * 32

        class TaskClass:
            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[object]:
                calls.append(dict(kwargs))
                return []

        self.assertEqual(executor._query_direct_children(TaskClass, controller_id), [])
        task_filter = calls[0]["task_filter"]
        self.assertEqual(task_filter["parent"], controller_id)
        self.assertIsInstance(task_filter["parent"], str)
        self.assertTrue(task_filter["search_hidden"])
        self.assertNotIn("project", task_filter)
        self.assertNotIn("status", task_filter)

    def test_preflight_rejects_stable_child_when_controller_is_missing(self) -> None:
        plan = executor.build_plan()
        execution_key = executor._execution_key(plan, executor.build_pinset())
        task_key = str(plan["pairs"][0]["training"]["task_key"])
        residual = SimpleNamespace(
            id="3" * 32,
            name=executor._task_name(execution_key, task_key),
            project=executor.PROJECT_ID,
            parent="4" * 32,
            status="created",
            tags=[
                executor._execution_tag(execution_key),
                f"p1-task-key:{task_key}",
            ],
            artifacts={},
            get_models=lambda: {},
            data=SimpleNamespace(execution=SimpleNamespace(queue="")),
        )

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == residual.id:
                    return residual
                return SimpleNamespace(id=task_id)

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                pattern = str(kwargs.get("task_name", ""))
                if __import__("re").match(pattern, residual.name) is None:
                    return []
                return [
                    {
                        "id": residual.id,
                        "name": residual.name,
                        "project": residual.project,
                        "status": residual.status,
                        "tags": residual.tags,
                        "parent": residual.parent,
                    }
                ]

        queue = {
            "queue_id": executor.QUEUE_ID,
            "queue_name": executor.QUEUE_NAME,
            "registered_worker_count": 1,
            "workers": [{}],
        }
        lease = {
            "name": executor._lease_queue_name(execution_key),
            "status": "available",
        }
        with (
            mock.patch.object(executor, "_validate_selector"),
            mock.patch.object(
                executor, "_validate_source_d", return_value="sealed-source-d"
            ),
            mock.patch.object(executor, "_validate_teacher_gate"),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(executor, "_validate_anchor_tasks"),
        ):
            with self.assertRaisesRegex(
                executor.P1ExecutorError, "without a controller"
            ):
                executor.preflight(
                    task_class=TaskClass,
                    queue_reader=lambda _: queue,
                    lease_reader=lambda _: lease,
                )

    def test_preflight_rejects_unknown_direct_descendant_before_clone(self) -> None:
        plan = executor.build_plan()
        pinset = executor.build_pinset()
        execution_key = executor._execution_key(plan, pinset)
        controller_id = "5" * 32
        unknown_id = "6" * 32
        controller = SimpleNamespace(
            id=controller_id,
            name=executor._controller_name(execution_key),
            project=executor.PROJECT_ID,
            parent=executor.SELECTOR_TASK_ID,
            status="created",
            tags=[
                executor._execution_tag(execution_key),
                executor.CONTROLLER_TAG,
            ],
            parameters=executor._controller_parameters(
                plan=plan,
                pinset=pinset,
                execution_key=execution_key,
            ),
            artifacts={},
            data=SimpleNamespace(execution=SimpleNamespace(queue="")),
        )
        unknown = SimpleNamespace(
            id=unknown_id,
            name=f"legacy-controller-id-child-{controller_id}",
            project=executor.PROJECT_ID,
            parent=controller_id,
            status="in_progress",
            tags=[executor._execution_tag(execution_key)],
            data=SimpleNamespace(execution=SimpleNamespace(queue=executor.QUEUE_ID)),
        )

        class TaskClass:
            mutation_calls = 0

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == controller_id:
                    return controller
                if task_id == unknown_id:
                    return unknown
                return SimpleNamespace(id=task_id)

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                task_filter = kwargs.get("task_filter", {})
                if isinstance(task_filter, dict) and "parent" in task_filter:
                    return [
                        {
                            "id": unknown.id,
                            "name": unknown.name,
                            "project": unknown.project,
                            "parent": unknown.parent,
                            "status": unknown.status,
                            "tags": unknown.tags,
                        }
                    ]
                pattern = str(kwargs.get("task_name", ""))
                if __import__("re").match(pattern, controller.name):
                    return [
                        {
                            "id": controller.id,
                            "name": controller.name,
                            "project": controller.project,
                            "parent": controller.parent,
                            "status": controller.status,
                            "tags": controller.tags,
                        }
                    ]
                return []

            @classmethod
            def create(cls, **kwargs: object) -> object:
                del kwargs
                cls.mutation_calls += 1
                raise AssertionError("preflight created a replacement controller")

            @classmethod
            def clone(cls, **kwargs: object) -> object:
                del kwargs
                cls.mutation_calls += 1
                raise AssertionError("preflight cloned a replacement child")

        queue = {
            "queue_id": executor.QUEUE_ID,
            "queue_name": executor.QUEUE_NAME,
            "registered_worker_count": 1,
            "workers": [{}],
        }
        lease = {
            "name": executor._lease_queue_name(execution_key),
            "status": "available",
        }
        with (
            mock.patch.object(executor, "_validate_selector"),
            mock.patch.object(
                executor, "_validate_source_d", return_value="sealed-source-d"
            ),
            mock.patch.object(executor, "_validate_teacher_gate"),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(executor, "_validate_anchor_tasks"),
        ):
            with self.assertRaisesRegex(
                executor.P1ExecutorError, "descendant inventory"
            ):
                executor.preflight(
                    task_class=TaskClass,
                    queue_reader=lambda _: queue,
                    lease_reader=lambda _: lease,
                )
        self.assertEqual(TaskClass.mutation_calls, 0)

    def test_preflight_rejects_unknown_execution_tagged_task(self) -> None:
        plan = executor.build_plan()
        execution_key = executor._execution_key(plan, executor.build_pinset())
        unknown = SimpleNamespace(
            id="7" * 32,
            name="legacy-p1-task-outside-stable-inventory",
            project="legacy-project",
            parent="8" * 32,
            status="in_progress",
            tags=[executor._execution_tag(execution_key)],
            data=SimpleNamespace(execution=SimpleNamespace(queue=executor.QUEUE_ID)),
        )

        class TaskClass:
            mutation_calls = 0

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == unknown.id:
                    return unknown
                return SimpleNamespace(id=task_id)

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                if "tags" not in kwargs:
                    return []
                return [
                    {
                        "id": unknown.id,
                        "name": unknown.name,
                        "project": unknown.project,
                        "parent": unknown.parent,
                        "status": unknown.status,
                        "tags": unknown.tags,
                    }
                ]

            @classmethod
            def create(cls, **kwargs: object) -> object:
                del kwargs
                cls.mutation_calls += 1
                raise AssertionError("preflight created a replacement controller")

            @classmethod
            def clone(cls, **kwargs: object) -> object:
                del kwargs
                cls.mutation_calls += 1
                raise AssertionError("preflight cloned a replacement child")

        queue = {
            "queue_id": executor.QUEUE_ID,
            "queue_name": executor.QUEUE_NAME,
            "registered_worker_count": 1,
            "workers": [{}],
        }
        lease = {
            "name": executor._lease_queue_name(execution_key),
            "status": "available",
        }
        with (
            mock.patch.object(executor, "_validate_selector"),
            mock.patch.object(
                executor, "_validate_source_d", return_value="sealed-source-d"
            ),
            mock.patch.object(executor, "_validate_teacher_gate"),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(executor, "_validate_anchor_tasks"),
        ):
            with self.assertRaisesRegex(
                executor.P1ExecutorError,
                "execution-tag query differs from stable-name task inventory",
            ):
                executor.preflight(
                    task_class=TaskClass,
                    queue_reader=lambda _: queue,
                    lease_reader=lambda _: lease,
                )
        self.assertEqual(TaskClass.mutation_calls, 0)

    def test_preflight_allows_historical_orphan_child_with_replacement_controller(
        self,
    ) -> None:
        plan = executor.build_plan()
        pinset = executor.build_pinset()
        execution_key = executor._execution_key(plan, pinset)
        task_key = str(plan["pairs"][0]["training"]["task_key"])
        expected_execution_tag = executor._execution_tag(execution_key)

        for orphan_keeps_execution_tag in (False, True):
            with self.subTest(orphan_keeps_execution_tag=orphan_keeps_execution_tag):
                controller = SimpleNamespace(
                    id="9" * 32,
                    name=executor._controller_name(execution_key),
                    project=executor.PROJECT_ID,
                    parent=executor.SELECTOR_TASK_ID,
                    status="created",
                    tags=[expected_execution_tag, executor.CONTROLLER_TAG],
                    artifacts={},
                    get_parameters=lambda **_: executor._controller_parameters(
                        plan=plan,
                        pinset=pinset,
                        execution_key=execution_key,
                    ),
                    data=SimpleNamespace(execution=SimpleNamespace(queue="")),
                )
                old_controller = SimpleNamespace(
                    id="b" * 32,
                    name=executor._controller_name(execution_key),
                    project=executor.PROJECT_ID,
                    parent=executor.SELECTOR_TASK_ID,
                    status="failed",
                    tags=[
                        executor.ORPHAN_TAG,
                        expected_execution_tag,
                        executor.CONTROLLER_TAG,
                    ],
                    artifacts={},
                    get_parameters=lambda **_: {},
                    data=SimpleNamespace(execution=SimpleNamespace(queue="")),
                )
                orphan_tags = [executor.ORPHAN_TAG]
                if orphan_keeps_execution_tag:
                    orphan_tags.extend(
                        [expected_execution_tag, f"p1-task-key:{task_key}"]
                    )
                orphan = SimpleNamespace(
                    id="a" * 32,
                    name=executor._task_name(execution_key, task_key),
                    project=executor.PROJECT_ID,
                    parent=old_controller.id,
                    status="failed",
                    tags=orphan_tags,
                    data=SimpleNamespace(execution=SimpleNamespace(queue="")),
                )

                class TaskClass:
                    @classmethod
                    def get_task(cls, *, task_id: str) -> object:
                        if task_id == controller.id:
                            return controller
                        if task_id == old_controller.id:
                            return old_controller
                        if task_id == orphan.id:
                            return orphan
                        return SimpleNamespace(id=task_id)

                    @classmethod
                    def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                        if "tags" in kwargs:
                            tasks = [controller, old_controller]
                            if orphan_keeps_execution_tag:
                                tasks.append(orphan)
                        elif "parent" in kwargs.get("task_filter", {}):
                            parent = str(kwargs["task_filter"]["parent"])
                            tasks = [orphan] if parent == old_controller.id else []
                        else:
                            pattern = str(kwargs.get("task_name", ""))
                            tasks = [
                                task
                                for task in (controller, old_controller, orphan)
                                if __import__("re").match(pattern, task.name)
                            ]
                        return [
                            {
                                "id": task.id,
                                "name": task.name,
                                "project": task.project,
                                "parent": task.parent,
                                "status": task.status,
                                "tags": task.tags,
                            }
                            for task in tasks
                        ]

                queue = {
                    "queue_id": executor.QUEUE_ID,
                    "queue_name": executor.QUEUE_NAME,
                    "registered_worker_count": 1,
                    "workers": [{}],
                }
                lease = {
                    "name": executor._lease_queue_name(execution_key),
                    "status": "available",
                }
                with (
                    mock.patch.object(executor, "_validate_selector"),
                    mock.patch.object(
                        executor,
                        "_validate_source_d",
                        return_value="sealed-source-d",
                    ),
                    mock.patch.object(executor, "_validate_teacher_gate"),
                    mock.patch.object(executor, "_validate_teacher"),
                    mock.patch.object(executor, "_validate_anchor_tasks"),
                ):
                    receipt = executor.preflight(
                        task_class=TaskClass,
                        queue_reader=lambda _: queue,
                        lease_reader=lambda _: lease,
                    )
                self.assertEqual(receipt["status"], "passed")
                recorded = next(
                    item
                    for item in receipt["existing_execution"]["children"]
                    if item["task_key"] == task_key
                )
                self.assertEqual(recorded["quarantined_task_ids"], [orphan.id])
                self.assertEqual(
                    receipt["existing_execution"]["quarantined_controller_ids"],
                    [old_controller.id],
                )

    def test_preflight_allows_tagged_quarantined_controller_for_replacement(
        self,
    ) -> None:
        plan = executor.build_plan()
        pinset = executor.build_pinset()
        execution_key = executor._execution_key(plan, pinset)
        controller = SimpleNamespace(
            id="c" * 32,
            name=executor._controller_name(execution_key),
            project=executor.PROJECT_ID,
            parent=executor.SELECTOR_TASK_ID,
            status="failed",
            tags=[
                executor.ORPHAN_TAG,
                executor._execution_tag(execution_key),
                executor.CONTROLLER_TAG,
            ],
            artifacts={},
            get_parameters=lambda **_: {},
            data=SimpleNamespace(execution=SimpleNamespace(queue="")),
        )

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == controller.id:
                    return controller
                return SimpleNamespace(id=task_id)

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                if "tags" in kwargs:
                    tasks = [controller]
                elif "parent" in kwargs.get("task_filter", {}):
                    tasks = []
                else:
                    pattern = str(kwargs.get("task_name", ""))
                    tasks = (
                        [controller]
                        if __import__("re").match(pattern, controller.name)
                        else []
                    )
                return [
                    {
                        "id": task.id,
                        "name": task.name,
                        "project": task.project,
                        "parent": task.parent,
                        "status": task.status,
                        "tags": task.tags,
                    }
                    for task in tasks
                ]

        queue = {
            "queue_id": executor.QUEUE_ID,
            "queue_name": executor.QUEUE_NAME,
            "registered_worker_count": 1,
            "workers": [{}],
        }
        lease = {
            "name": executor._lease_queue_name(execution_key),
            "status": "available",
        }
        with (
            mock.patch.object(executor, "_validate_selector"),
            mock.patch.object(
                executor, "_validate_source_d", return_value="sealed-source-d"
            ),
            mock.patch.object(executor, "_validate_teacher_gate"),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(executor, "_validate_anchor_tasks"),
        ):
            receipt = executor.preflight(
                task_class=TaskClass,
                queue_reader=lambda _: queue,
                lease_reader=lambda _: lease,
            )
        self.assertEqual(receipt["status"], "passed")
        self.assertIsNone(receipt["existing_execution"]["controller_id"])
        self.assertEqual(
            receipt["existing_execution"]["quarantined_controller_ids"],
            [controller.id],
        )

    def test_preflight_rejects_unknown_descendant_of_quarantined_controller(
        self,
    ) -> None:
        plan = executor.build_plan()
        pinset = executor.build_pinset()
        execution_key = executor._execution_key(plan, pinset)
        controller = SimpleNamespace(
            id="d" * 32,
            name=executor._controller_name(execution_key),
            project=executor.PROJECT_ID,
            parent=executor.SELECTOR_TASK_ID,
            status="failed",
            tags=[
                executor.ORPHAN_TAG,
                executor._execution_tag(execution_key),
                executor.CONTROLLER_TAG,
            ],
            artifacts={},
            get_parameters=lambda **_: {},
            data=SimpleNamespace(execution=SimpleNamespace(queue="")),
        )
        unknown = SimpleNamespace(
            id="e" * 32,
            name="externally-attached-unknown-child",
            project=executor.PROJECT_ID,
            parent=controller.id,
            status="created",
            tags=[],
            data=SimpleNamespace(execution=SimpleNamespace(queue="")),
        )

        class TaskClass:
            mutation_calls = 0

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == controller.id:
                    return controller
                if task_id == unknown.id:
                    return unknown
                return SimpleNamespace(id=task_id)

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                if "tags" in kwargs:
                    tasks = [controller]
                elif "parent" in kwargs.get("task_filter", {}):
                    parent = str(kwargs["task_filter"]["parent"])
                    tasks = [unknown] if parent == controller.id else []
                else:
                    pattern = str(kwargs.get("task_name", ""))
                    tasks = (
                        [controller]
                        if __import__("re").match(pattern, controller.name)
                        else []
                    )
                return [
                    {
                        "id": task.id,
                        "name": task.name,
                        "project": task.project,
                        "parent": task.parent,
                        "status": task.status,
                        "tags": task.tags,
                    }
                    for task in tasks
                ]

            @classmethod
            def create(cls, **kwargs: object) -> object:
                del kwargs
                cls.mutation_calls += 1
                raise AssertionError("preflight created a replacement controller")

            @classmethod
            def clone(cls, **kwargs: object) -> object:
                del kwargs
                cls.mutation_calls += 1
                raise AssertionError("preflight cloned a replacement child")

        queue = {
            "queue_id": executor.QUEUE_ID,
            "queue_name": executor.QUEUE_NAME,
            "registered_worker_count": 1,
            "workers": [{}],
        }
        lease = {
            "name": executor._lease_queue_name(execution_key),
            "status": "available",
        }
        with (
            mock.patch.object(executor, "_validate_selector"),
            mock.patch.object(
                executor, "_validate_source_d", return_value="sealed-source-d"
            ),
            mock.patch.object(executor, "_validate_teacher_gate"),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(executor, "_validate_anchor_tasks"),
        ):
            with self.assertRaisesRegex(
                executor.P1ExecutorError,
                "quarantined controller descendant inventory",
            ):
                executor.preflight(
                    task_class=TaskClass,
                    queue_reader=lambda _: queue,
                    lease_reader=lambda _: lease,
                )
        self.assertEqual(TaskClass.mutation_calls, 0)

    def test_global_mutex_rejects_a_concurrent_holder(self) -> None:
        client = _APIClient()
        execution_key = "c" * 64
        with (
            mock.patch.object(
                executor,
                "_recover_owned_mutex_from_local_journal",
                return_value=None,
            ),
            mock.patch.object(executor, "_write_mutex_receipt"),
            mock.patch.object(executor, "_remove_mutex_receipt"),
        ):
            with executor._server_execution_mutex(
                execution_key,
                api_client=client,
                owner_token="1" * 32,
            ):
                with self.assertRaisesRegex(executor.P1ExecutorError, "already held"):
                    with executor._server_execution_mutex(
                        execution_key,
                        api_client=client,
                        owner_token="2" * 32,
                    ):
                        self.fail("second holder acquired the mutex")
        self.assertEqual(client.queues.rows, {})
        self.assertEqual(len(client.queues.deleted), 1)

    def test_crash_mutex_recovery_adopts_without_delete_recreate_window(self) -> None:
        client = _APIClient()
        execution_key = "d" * 64
        name = executor._lease_queue_name(execution_key)
        owner_tag = "owner:" + "3" * 32
        holder = client.queues.create(
            name=name,
            tags=["p1-scoped-mutex", executor._execution_tag(execution_key), owner_tag],
        )
        receipt = {
            "schema_version": 1,
            "document_type": "resilient_v2x_p1_local_mutex_journal",
            "execution_key": execution_key,
            "mutex_name": name,
            "queue_id": holder.id,
            "owner_tag": owner_tag,
        }
        receipt["seal_sha256"] = executor._seal(receipt)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"CLEARML_CACHE_DIR": directory}):
                executor._write_mutex_receipt(receipt)
                create_count = len(client.queues.create_calls)
                with executor._server_execution_mutex(
                    execution_key,
                    api_client=client,
                ) as adopted:
                    self.assertEqual(adopted["queue_id"], holder.id)
                    self.assertEqual(len(client.queues.create_calls), create_count)
                    self.assertEqual(client.queues.deleted, [])
                    self.assertTrue(executor._mutex_receipt_path().exists())
                    with mock.patch.object(
                        executor,
                        "_recover_owned_mutex_from_local_journal",
                        return_value=None,
                    ):
                        with self.assertRaisesRegex(
                            executor.P1ExecutorError, "already held"
                        ):
                            with executor._server_execution_mutex(
                                execution_key,
                                api_client=client,
                                owner_token="4" * 32,
                            ):
                                self.fail("competing host acquired the adopted mutex")
                    self.assertIn(name, client.queues.rows)
                    self.assertEqual(client.queues.deleted, [])
                self.assertFalse(executor._mutex_receipt_path().exists())
        self.assertEqual(client.queues.rows, {})
        self.assertEqual(client.queues.deleted, [holder.id])

    def test_crash_mutex_recovery_refuses_owner_mismatch_without_delete(self) -> None:
        client = _APIClient()
        execution_key = "5" * 64
        name = executor._lease_queue_name(execution_key)
        holder = client.queues.create(
            name=name,
            tags=[
                "p1-scoped-mutex",
                executor._execution_tag(execution_key),
                "owner:" + "6" * 32,
            ],
        )
        receipt = {
            "schema_version": 1,
            "document_type": "resilient_v2x_p1_local_mutex_journal",
            "execution_key": execution_key,
            "mutex_name": name,
            "queue_id": holder.id,
            "owner_tag": "owner:" + "7" * 32,
        }
        receipt["seal_sha256"] = executor._seal(receipt)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"CLEARML_CACHE_DIR": directory}):
                executor._write_mutex_receipt(receipt)
                with self.assertRaisesRegex(
                    executor.P1ExecutorError, "ownership or emptiness"
                ):
                    executor._recover_owned_mutex_from_local_journal(
                        execution_key, api_client=client
                    )
        self.assertIn(name, client.queues.rows)
        self.assertEqual(client.queues.deleted, [])

    def test_controller_create_and_parent_recovery_use_clearml_21_signature(
        self,
    ) -> None:
        class Artifact:
            def __init__(self, value: dict[str, object]) -> None:
                self.value = copy.deepcopy(value)

            def get(self) -> dict[str, object]:
                return copy.deepcopy(self.value)

        class Controller:
            def __init__(self, name: str) -> None:
                self.id = "9" * 32
                self.name = name
                self.status = "created"
                self.parent = ""
                self.project = executor.PROJECT_ID
                self.tags: list[str] = []
                self.parameters: dict[str, object] = {}
                self.artifacts: dict[str, Artifact] = {}
                self.parent_calls: list[str] = []
                self.output_uri = ""
                self.data = SimpleNamespace(
                    created="2026-08-18T00:00:00Z",
                    execution=SimpleNamespace(queue=""),
                    parent="",
                    project=executor.PROJECT_ID,
                )

            def set_parent(self, parent: str) -> bool:
                self.parent_calls.append(parent)
                self.parent = parent
                self.data.parent = parent
                return True

            def add_tags(self, tags: list[str]) -> bool:
                self.tags.extend(item for item in tags if item not in self.tags)
                return True

            def get_tags(self) -> list[str]:
                return list(self.tags)

            def set_parameters(
                self, parameters: dict[str, object], **kwargs: object
            ) -> bool:
                self.parameters = dict(parameters)
                return True

            def get_parameters(self, **kwargs: object) -> dict[str, object]:
                return dict(self.parameters)

            def upload_artifact(
                self,
                *,
                name: str,
                artifact_object: dict[str, object],
                wait_on_upload: bool,
            ) -> bool:
                self.assert_wait(wait_on_upload)
                self.artifacts[name] = Artifact(artifact_object)
                return True

            @staticmethod
            def assert_wait(wait_on_upload: bool) -> None:
                if wait_on_upload is not True:
                    raise AssertionError("controller artifact upload did not wait")

        class ControllerTaskClass:
            TaskTypes = SimpleNamespace(controller="controller")
            task: Controller | None = None
            create_calls = 0

            @classmethod
            def create(cls, **kwargs: object) -> Controller:
                cls.create_calls += 1
                cls.task = Controller(str(kwargs["task_name"]))
                return cls.task

            @classmethod
            def get_task(cls, *, task_id: str) -> Controller:
                if cls.task is None or task_id != cls.task.id:
                    raise AssertionError("unexpected controller task ID")
                return cls.task

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                if cls.task is None:
                    return []
                pattern = str(kwargs["task_name"])
                if __import__("re").match(pattern, cls.task.name) is None:
                    return []
                return [
                    {
                        "id": cls.task.id,
                        "name": cls.task.name,
                        "project": cls.task.project,
                        "status": cls.task.status,
                        "tags": cls.task.get_tags(),
                        "parent": cls.task.parent,
                    }
                ]

        plan = executor.build_plan()
        pinset = executor.build_pinset()
        execution_key = executor._execution_key(plan, pinset)
        first, first_created = executor._find_or_create_controller(
            ControllerTaskClass,
            plan=plan,
            pinset=pinset,
            execution_key=execution_key,
        )
        self.assertTrue(first_created)
        self.assertEqual(first.parent_calls, [executor.SELECTOR_TASK_ID])

        first.parent = ""
        first.data.parent = ""
        recovered, second_created = executor._find_or_create_controller(
            ControllerTaskClass,
            plan=plan,
            pinset=pinset,
            execution_key=execution_key,
        )
        self.assertFalse(second_created)
        self.assertIs(recovered, first)
        self.assertEqual(ControllerTaskClass.create_calls, 1)
        self.assertEqual(
            recovered.parent_calls,
            [executor.SELECTOR_TASK_ID, executor.SELECTOR_TASK_ID],
        )

    def test_controller_completion_round_trips_without_relying_on_close(self) -> None:
        class Controller:
            def __init__(self) -> None:
                self.id = "7" * 32
                self.status = "created"
                self.mark_completed_calls: list[dict[str, bool]] = []
                self.close_calls = 0

            def mark_completed(self, ignore_errors: bool, force: bool) -> bool:
                self.mark_completed_calls.append(
                    {"ignore_errors": ignore_errors, "force": force}
                )
                self.status = "completed"
                return True

            def close(self) -> None:
                raise AssertionError("controller lifecycle relied on close()")

        controller = Controller()

        class ControllerTaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> Controller:
                self.assertEqual(task_id, controller.id)
                return controller

        completed = executor._complete_controller(ControllerTaskClass, controller)
        self.assertIs(completed, controller)
        self.assertEqual(
            controller.mark_completed_calls,
            [{"ignore_errors": False, "force": True}],
        )
        self.assertEqual(controller.close_calls, 0)
        self.assertEqual(controller.status, "completed")

    def test_controller_recovery_failure_preserves_timeout_controller_and_child(
        self,
    ) -> None:
        plan = executor.build_plan()
        pinset = executor.build_pinset()
        execution_key = executor._execution_key(plan, pinset)
        controller_id = "d" * 32
        child_id = "e" * 32
        task_key = "train-r01-s02-resilient_v2x"

        class Controller:
            def __init__(self) -> None:
                self.id = controller_id
                self.name = executor._controller_name(execution_key)
                self.status = "created"
                self.parent = ""
                self.project = executor.PROJECT_ID
                self.tags = [
                    executor._execution_tag(execution_key),
                    executor.CONTROLLER_TAG,
                ]
                self.parameters = executor._controller_parameters(
                    plan=plan,
                    pinset=pinset,
                    execution_key=execution_key,
                )
                self.artifacts = {executor.JOURNAL_ARTIFACT: object()}
                self.archived = False
                self.data = SimpleNamespace(
                    created="2026-08-18T00:00:00Z",
                    execution=SimpleNamespace(queue=""),
                )

            def get_tags(self) -> list[str]:
                return list(self.tags)

            def set_parent(self, parent: str) -> None:
                raise RuntimeError(f"controller recovery mutation failed for {parent}")

            def mark_failed(self, **kwargs: object) -> None:
                raise AssertionError("resumed controller was quarantined")

            def set_archived(self, value: bool) -> None:
                raise AssertionError("resumed controller was archived")

        controller = Controller()
        child = SimpleNamespace(
            id=child_id,
            name=executor._task_name(execution_key, task_key),
            status="in_progress",
            parent=controller_id,
            project=executor.PROJECT_ID,
            tags=[
                executor._execution_tag(execution_key),
                f"p1-task-key:{task_key}",
            ],
            data=SimpleNamespace(
                created="2026-08-18T00:01:00Z",
                execution=SimpleNamespace(queue=executor.QUEUE_ID),
            ),
        )

        class TaskClass:
            create_calls = 0

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                pattern = str(kwargs.get("task_name", ""))
                tasks = [controller, child]
                return [
                    {
                        "id": task.id,
                        "name": task.name,
                        "project": task.project,
                        "status": task.status,
                        "tags": task.tags,
                        "parent": task.parent,
                    }
                    for task in tasks
                    if __import__("re").match(pattern, task.name)
                ]

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == controller.id:
                    return controller
                if task_id == child.id:
                    return child
                raise AssertionError("unexpected task ID")

            @classmethod
            def create(cls, **kwargs: object) -> object:
                cls.create_calls += 1
                raise AssertionError("replacement controller was created")

        with self.assertRaisesRegex(
            RuntimeError, "controller recovery mutation failed"
        ):
            executor._find_or_create_controller(
                TaskClass,
                plan=plan,
                pinset=pinset,
                execution_key=execution_key,
            )
        self.assertEqual(TaskClass.create_calls, 0)
        self.assertEqual(controller.status, "created")
        self.assertNotIn(executor.ORPHAN_TAG, controller.tags)
        self.assertFalse(controller.archived)
        self.assertEqual(child.status, "in_progress")

    def test_execute_plan_recovers_staged_controller_without_replacement(self) -> None:
        class StopExecution(RuntimeError):
            pass

        class Artifact:
            def __init__(self, value: dict[str, object]) -> None:
                self.value = copy.deepcopy(value)

            def get(self) -> dict[str, object]:
                return copy.deepcopy(self.value)

        plan = executor.build_plan()
        pinset = executor.build_pinset()
        execution_key = executor._execution_key(plan, pinset)

        class Controller:
            def __init__(self) -> None:
                self.id = "9" * 32
                self.name = executor._controller_name(execution_key)
                self.status = "created"
                self.parent = ""
                self.project = executor.PROJECT_ID
                self.tags: list[str] = []
                self.parameters: dict[str, object] = {}
                self.artifacts: dict[str, Artifact] = {}
                self.output_uri = ""
                self.data = SimpleNamespace(
                    created="2026-08-18T00:00:00Z",
                    execution=SimpleNamespace(queue=""),
                )

            def set_parent(self, parent: str) -> None:
                self.parent = parent

            def get_tags(self) -> list[str]:
                return list(self.tags)

            def add_tags(self, tags: list[str]) -> None:
                self.tags.extend(item for item in tags if item not in self.tags)

            def get_parameters(self, **kwargs: object) -> dict[str, object]:
                return dict(self.parameters)

            def set_parameters(
                self, parameters: dict[str, object], **kwargs: object
            ) -> None:
                self.parameters = dict(parameters)

            def upload_artifact(
                self,
                *,
                name: str,
                artifact_object: dict[str, object],
                wait_on_upload: bool,
            ) -> bool:
                if wait_on_upload is not True:
                    raise AssertionError("controller upload did not wait")
                self.artifacts[name] = Artifact(artifact_object)
                return True

        controller = Controller()

        class TaskClass:
            create_calls = 0

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                if "tags" in kwargs:
                    return []
                if "parent" in kwargs.get("task_filter", {}):
                    return []
                pattern = str(kwargs.get("task_name", ""))
                if __import__("re").match(pattern, controller.name) is None:
                    return []
                return [
                    {
                        "id": controller.id,
                        "name": controller.name,
                        "project": controller.project,
                        "status": controller.status,
                        "tags": controller.tags,
                        "parent": controller.parent,
                    }
                ]

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == controller.id:
                    return controller
                return SimpleNamespace(id=task_id)

            @classmethod
            def create(cls, **kwargs: object) -> object:
                cls.create_calls += 1
                raise AssertionError("staged controller was replaced")

        queue = {
            "queue_id": executor.QUEUE_ID,
            "queue_name": executor.QUEUE_NAME,
            "registered_worker_count": 1,
            "workers": [{}],
        }
        lease = {
            "name": executor._lease_queue_name(execution_key),
            "status": "available",
        }
        with (
            mock.patch.object(
                executor, "_execution_lock", return_value=contextlib.nullcontext()
            ),
            mock.patch.object(
                executor,
                "_server_execution_mutex",
                return_value=contextlib.nullcontext({}),
            ),
            mock.patch.object(executor, "_read_lease_snapshot", return_value=lease),
            mock.patch.object(executor, "_validate_selector"),
            mock.patch.object(
                executor, "_validate_source_d", return_value="sealed-source-d"
            ),
            mock.patch.object(executor, "_validate_teacher_gate"),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(executor, "_validate_anchor_tasks"),
            mock.patch.object(
                executor,
                "_create_task",
                side_effect=StopExecution("stop after controller recovery"),
            ),
        ):
            with self.assertRaisesRegex(
                StopExecution, "stop after controller recovery"
            ):
                executor.execute_plan(
                    authorization_token=executor.EXECUTION_TOKEN,
                    poll_seconds=1.0,
                    timeout_hours=1.0,
                    task_class=TaskClass,
                    queue_reader=lambda _: queue,
                    api_client=SimpleNamespace(),
                )
        self.assertEqual(TaskClass.create_calls, 0)
        self.assertEqual(controller.parent, executor.SELECTOR_TASK_ID)
        self.assertIn(executor._execution_tag(execution_key), controller.tags)
        self.assertIn(executor.CONTROLLER_TAG, controller.tags)
        self.assertEqual(
            controller.parameters,
            executor._controller_parameters(
                plan=plan,
                pinset=pinset,
                execution_key=execution_key,
            ),
        )
        self.assertIn("p1_multiseed_plan", controller.artifacts)
        self.assertIn("p1_multiseed_pinset", controller.artifacts)

    def test_execute_plan_recovers_staged_child_without_clone_or_enqueue(self) -> None:
        class StopExecution(RuntimeError):
            pass

        class Artifact:
            def __init__(self, value: dict[str, object]) -> None:
                self.value = copy.deepcopy(value)

            def get(self) -> dict[str, object]:
                return copy.deepcopy(self.value)

        plan = executor.build_plan()
        pinset = executor.build_pinset()
        execution_key = executor._execution_key(plan, pinset)
        training = plan["pairs"][0]["training"]
        task_key = str(training["task_key"])
        subject = str(training["subject"])
        seed = int(training["training_seed"])
        controller_id = "a" * 32
        source_d = "sealed-source-d"
        teacher_parameters = {"Args/template": "teacher"}

        class Controller:
            def __init__(self) -> None:
                self.id = controller_id
                self.name = executor._controller_name(execution_key)
                self.status = "created"
                self.parent = executor.SELECTOR_TASK_ID
                self.project = executor.PROJECT_ID
                self.tags = [
                    executor._execution_tag(execution_key),
                    executor.CONTROLLER_TAG,
                ]
                self.parameters = executor._controller_parameters(
                    plan=plan,
                    pinset=pinset,
                    execution_key=execution_key,
                )
                self.artifacts: dict[str, Artifact] = {
                    "p1_multiseed_plan": Artifact(plan),
                    "p1_multiseed_pinset": Artifact(pinset),
                }
                self.output_uri = executor.FILES_SERVER_URI
                self.data = SimpleNamespace(
                    created="2026-08-18T00:00:00Z",
                    execution=SimpleNamespace(queue=""),
                )

            def get_tags(self) -> list[str]:
                return list(self.tags)

            def add_tags(self, tags: list[str]) -> None:
                self.tags.extend(item for item in tags if item not in self.tags)

            def get_parameters(self, **kwargs: object) -> dict[str, object]:
                return dict(self.parameters)

            def set_parameters(
                self, parameters: dict[str, object], **kwargs: object
            ) -> None:
                self.parameters = dict(parameters)

            def upload_artifact(
                self,
                *,
                name: str,
                artifact_object: dict[str, object],
                wait_on_upload: bool,
            ) -> bool:
                if wait_on_upload is not True:
                    raise AssertionError("controller upload did not wait")
                self.artifacts[name] = Artifact(artifact_object)
                return True

        class Child:
            def __init__(self) -> None:
                self.id = "b" * 32
                self.name = executor._task_name(execution_key, task_key)
                self.status = "created"
                self.parent = controller_id
                self.project = executor.PROJECT_ID
                self.tags: list[str] = []
                self.parameters = dict(teacher_parameters)
                self.artifacts: dict[str, object] = {}
                self.output_uri = ""
                self.data = SimpleNamespace(
                    created="2026-08-18T00:01:00Z",
                    execution=SimpleNamespace(queue=""),
                    script=SimpleNamespace(
                        repository="",
                        working_dir=".",
                        entry_point=executor.TASK_ENTRY_POINT,
                        diff=source_d,
                    ),
                    container={
                        "image": "test-image",
                        "arguments": "--test",
                        "setup_shell_script": "",
                    },
                )

            def get_tags(self) -> list[str]:
                return list(self.tags)

            def add_tags(self, tags: list[str]) -> None:
                self.tags.extend(item for item in tags if item not in self.tags)

            def get_models(self) -> dict[str, list[object]]:
                return {}

            def get_parameters(self, **kwargs: object) -> dict[str, object]:
                return dict(self.parameters)

            def set_parameters(
                self, parameters: dict[str, object], **kwargs: object
            ) -> None:
                self.parameters = dict(parameters)

        controller = Controller()
        child = Child()
        teacher = SimpleNamespace(
            id=executor.TEACHER_TASK_ID,
            get_parameters=lambda **kwargs: dict(teacher_parameters),
            data=SimpleNamespace(
                script=SimpleNamespace(
                    repository="",
                    working_dir=".",
                    entry_point=executor.TASK_ENTRY_POINT,
                    diff="teacher-template",
                ),
                container=copy.deepcopy(child.data.container),
            ),
        )

        class TaskClass:
            clone_calls = 0
            enqueue_calls = 0

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                task_filter = kwargs.get("task_filter", {})
                if "tags" in kwargs:
                    candidates = [controller]
                elif "parent" in task_filter:
                    candidates = (
                        [child] if task_filter["parent"] == controller_id else []
                    )
                else:
                    pattern = str(kwargs.get("task_name", ""))
                    candidates = [
                        task
                        for task in (controller, child)
                        if __import__("re").match(pattern, task.name)
                    ]
                return [
                    {
                        "id": task.id,
                        "name": task.name,
                        "project": task.project,
                        "status": task.status,
                        "tags": task.tags,
                        "parent": task.parent,
                    }
                    for task in candidates
                ]

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == controller.id:
                    return controller
                if task_id == child.id:
                    return child
                if task_id == teacher.id:
                    return teacher
                return SimpleNamespace(id=task_id)

            @classmethod
            def clone(cls, **kwargs: object) -> object:
                cls.clone_calls += 1
                raise AssertionError("staged child was cloned")

            @classmethod
            def enqueue(cls, **kwargs: object) -> object:
                cls.enqueue_calls += 1
                raise AssertionError("staged child was enqueued")

        queue = {
            "queue_id": executor.QUEUE_ID,
            "queue_name": executor.QUEUE_NAME,
            "registered_worker_count": 1,
            "workers": [{}],
        }
        lease = {
            "name": executor._lease_queue_name(execution_key),
            "status": "available",
        }
        recovered: list[object] = []
        real_create_task = executor._create_task

        def recover_child_then_stop(*args: object, **kwargs: object) -> object:
            task = real_create_task(*args, **kwargs)
            recovered.append(task)
            raise StopExecution("stop after staged child recovery")

        with (
            mock.patch.object(
                executor, "_execution_lock", return_value=contextlib.nullcontext()
            ),
            mock.patch.object(
                executor,
                "_server_execution_mutex",
                return_value=contextlib.nullcontext({}),
            ),
            mock.patch.object(executor, "_read_lease_snapshot", return_value=lease),
            mock.patch.object(executor, "_validate_selector"),
            mock.patch.object(executor, "_validate_source_d", return_value=source_d),
            mock.patch.object(executor, "_validate_teacher_gate"),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(executor, "_validate_anchor_tasks"),
            mock.patch.object(executor, "_require_script"),
            mock.patch.object(
                executor, "_create_task", side_effect=recover_child_then_stop
            ),
        ):
            with self.assertRaisesRegex(
                StopExecution, "stop after staged child recovery"
            ):
                executor.execute_plan(
                    authorization_token=executor.EXECUTION_TOKEN,
                    poll_seconds=1.0,
                    timeout_hours=1.0,
                    task_class=TaskClass,
                    queue_reader=lambda _: queue,
                    api_client=SimpleNamespace(),
                )
        self.assertEqual(recovered, [child])
        self.assertEqual(TaskClass.clone_calls, 0)
        self.assertEqual(TaskClass.enqueue_calls, 0)
        self.assertIn(executor._execution_tag(execution_key), child.tags)
        self.assertIn(f"p1-task-key:{task_key}", child.tags)
        self.assertEqual(
            child.parameters,
            executor._training_parameters(subject, seed),
        )
        journal = controller.artifacts[executor.JOURNAL_ARTIFACT].get()
        self.assertEqual(executor._journal_row(journal, task_key)["task_id"], child.id)

    def test_completed_controller_manifest_returns_without_child_mutation(self) -> None:
        class Artifact:
            def __init__(self, value: dict[str, object]) -> None:
                self.value = copy.deepcopy(value)

            def get(self) -> dict[str, object]:
                return copy.deepcopy(self.value)

        plan = executor.build_plan()
        execution_key = executor._execution_key(plan, executor.build_pinset())
        controller_id = "8" * 32
        results: list[dict[str, object]] = []
        for index, pair in enumerate(plan["pairs"]):
            training = pair["training"]
            results.append(
                {
                    "subject": training["subject"],
                    "seed_index": training["seed_index"],
                    "training_seed": training["training_seed"],
                    "training_task_id": f"{index + 1:032x}",
                    "model_id": f"{index + 11:032x}",
                    "checkpoint_sha256": f"{index + 21:064x}",
                    "evaluation": {"task_id": f"{index + 31:032x}"},
                }
            )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "document_type": "resilient_v2x_p1_scoped_multiseed_execution",
            "execution_key": execution_key,
            "plan_seal_sha256": plan["seal_sha256"],
            "controller_task_id": controller_id,
            "status": "completed",
            "results": results,
        }
        manifest["seal_sha256"] = executor._seal(manifest)
        controller = SimpleNamespace(
            id=controller_id,
            status="completed",
            artifacts={executor.MANIFEST_ARTIFACT: Artifact(manifest)},
        )

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> SimpleNamespace:
                return SimpleNamespace(id=task_id)

        with (
            mock.patch.object(
                executor,
                "_execution_lock",
                return_value=contextlib.nullcontext(),
            ),
            mock.patch.object(
                executor,
                "_server_execution_mutex",
                return_value=contextlib.nullcontext({}),
            ),
            mock.patch.object(executor, "preflight") as preflight,
            mock.patch.object(
                executor, "_validate_source_d", return_value="sealed-source-d"
            ),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(
                executor,
                "_find_or_create_controller",
                return_value=(controller, False),
            ),
            mock.patch.object(executor, "_create_task") as create_task,
            mock.patch.object(executor, "_enqueue_once") as enqueue,
        ):
            observed = executor.execute_plan(
                authorization_token=executor.EXECUTION_TOKEN,
                poll_seconds=1.0,
                timeout_hours=1.0,
                task_class=TaskClass,
                api_client=SimpleNamespace(),
            )
        self.assertEqual(observed, manifest)
        self.assertEqual(preflight.call_count, 2)
        create_task.assert_not_called()
        enqueue.assert_not_called()

    def test_journal_is_sealed_and_allows_only_quarantined_replacement(self) -> None:
        plan = executor.build_plan()
        pinset = executor.build_pinset()
        execution_key = executor._execution_key(plan, pinset)
        controller_id = "e" * 32
        journal = executor._new_journal(
            plan=plan,
            execution_key=execution_key,
            controller_id=controller_id,
        )
        row = journal["tasks"][0]
        row.update(
            {
                "task_id": "f" * 32,
                "state": "quarantined",
                "server_status": "failed",
                "result": {"quarantined_task_ids": ["f" * 32]},
            }
        )
        journal["seal_sha256"] = executor._seal(journal)
        validated = executor._validate_journal(
            journal,
            plan=plan,
            execution_key=execution_key,
            controller_id=controller_id,
        )
        self.assertEqual(validated["tasks"][0]["state"], "quarantined")
        forged = copy.deepcopy(validated)
        forged["tasks"][0]["task_id"] = "0" * 32
        with self.assertRaisesRegex(executor.P1ExecutorError, "seal mismatch"):
            executor._validate_journal(
                forged,
                plan=plan,
                execution_key=execution_key,
                controller_id=controller_id,
            )

    def test_timeout_preserves_active_child_for_resume(self) -> None:
        task = _FakeTask()
        task.id = "1" * 32
        task.status = "queued"

        class ActiveTaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> _FakeTask:
                self.assertEqual(task_id, task.id)
                return task

        with mock.patch.object(executor.time, "monotonic", side_effect=[0.0, 1.0]):
            with self.assertRaises(executor.P1RecoverableTimeout):
                executor._wait_for_task(
                    ActiveTaskClass,
                    task.id,
                    timeout_hours=0.0001,
                    poll_seconds=1.0,
                )
        self.assertEqual(task.status, "queued")

    def test_created_orphan_is_failed_and_archived_without_delete(self) -> None:
        _OrphanTaskClass.task = _OrphanTask()
        executor._quarantine_created(
            _OrphanTaskClass,
            _OrphanTaskClass.task,
            reason="configuration round-trip failed",
        )
        self.assertEqual(_OrphanTaskClass.task.status, "failed")
        self.assertTrue(_OrphanTaskClass.task.archived)
        self.assertIn(executor.ORPHAN_TAG, _OrphanTaskClass.task.tags)

    def test_quarantine_is_irreversible_and_resumes_from_every_crash_point(
        self,
    ) -> None:
        crash_states = (
            ("created", False, 1, 1),
            ("failed", False, 0, 1),
            ("failed", True, 0, 0),
        )
        for (
            status,
            archived,
            expected_mark_calls,
            expected_archive_calls,
        ) in crash_states:
            with self.subTest(status=status, archived=archived):
                task = _OrphanTask()
                task.status = status
                task.archived = archived
                task.tags = [executor.ORPHAN_TAG]
                _OrphanTaskClass.task = task

                self.assertTrue(executor._is_quarantined(task))
                executor._quarantine_created(
                    _OrphanTaskClass,
                    task,
                    reason="resume crash-point quarantine",
                )
                self.assertEqual(task.status, "failed")
                self.assertTrue(task.archived)
                self.assertTrue(executor._is_quarantined(task))
                self.assertEqual(task.mark_failed_calls, expected_mark_calls)
                self.assertEqual(task.archive_calls, expected_archive_calls)

    def test_tagged_created_orphan_is_never_selected_as_a_resume_candidate(
        self,
    ) -> None:
        task = _OrphanTask()
        task.name = "stable-child-name"
        task.project = executor.PROJECT_ID
        task.tags = [executor.ORPHAN_TAG]
        task.data.created = "2026-08-18T00:00:00Z"
        _OrphanTaskClass.task = task

        class RestartTaskClass(_OrphanTaskClass):
            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                return [
                    {
                        "id": cls.task.id,
                        "name": cls.task.name,
                        "project": cls.task.project,
                        "status": cls.task.status,
                        "tags": cls.task.tags,
                        "parent": "",
                    }
                ]

        selected = executor._select_unique_created_candidate(
            RestartTaskClass,
            name=task.name,
            context="crash-point child",
        )
        self.assertIsNone(selected)
        self.assertEqual(task.status, "failed")
        self.assertTrue(task.archived)

    def test_duplicate_active_stable_name_fails_without_auto_quarantine(self) -> None:
        name = "stable-child-name"
        tasks = []
        for index in (1, 2):
            tasks.append(
                SimpleNamespace(
                    id=f"{index:032x}",
                    name=name,
                    project=executor.PROJECT_ID,
                    parent="f" * 32,
                    status="created",
                    tags=[],
                    data=SimpleNamespace(
                        created=f"2026-08-18T00:00:0{index}Z",
                        execution=SimpleNamespace(queue=""),
                    ),
                )
            )

        class TaskClass:
            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                return [
                    {
                        "id": task.id,
                        "name": task.name,
                        "project": task.project,
                        "status": task.status,
                        "tags": task.tags,
                        "parent": task.parent,
                    }
                    for task in tasks
                ]

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                return next(task for task in tasks if task.id == task_id)

        with self.assertRaisesRegex(executor.P1ExecutorError, "duplicate active"):
            executor._select_unique_created_candidate(
                TaskClass,
                name=name,
                context="stable child",
            )
        self.assertTrue(all(executor.ORPHAN_TAG not in task.tags for task in tasks))

    def test_missing_tag_orphan_passes_preflight_and_allows_replacement_clone(
        self,
    ) -> None:
        plan = executor.build_plan()
        execution_key = executor._execution_key(plan, executor.build_pinset())
        task_key = "train-r01-s02-resilient_v2x"
        controller_id = "6" * 32
        old = _OrphanTask()
        old.id = "7" * 32
        old.name = executor._task_name(execution_key, task_key)
        old.parent = controller_id
        old.project = executor.PROJECT_ID
        old.status = "failed"
        old.archived = True
        old.tags = [executor.ORPHAN_TAG]
        replacement: _OrphanTask | None = None

        class TaskClass:
            clone_calls = 0

            @classmethod
            def query_tasks(cls, **kwargs: object) -> list[dict[str, object]]:
                if "tags" in kwargs:
                    return []
                pattern = str(kwargs.get("task_name", ""))
                candidates = [old] + ([replacement] if replacement is not None else [])
                return [
                    {
                        "id": task.id,
                        "name": task.name,
                        "project": task.project,
                        "status": task.status,
                        "tags": task.tags,
                        "parent": task.parent,
                    }
                    for task in candidates
                    if __import__("re").match(pattern, task.name)
                ]

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == old.id:
                    return old
                if replacement is not None and task_id == replacement.id:
                    return replacement
                return SimpleNamespace(id=task_id)

            @classmethod
            def clone(cls, **kwargs: object) -> _OrphanTask:
                nonlocal replacement
                cls.clone_calls += 1
                replacement = _OrphanTask()
                replacement.id = "8" * 32
                replacement.name = str(kwargs["name"])
                replacement.parent = str(kwargs["parent"])
                replacement.project = str(kwargs["project"])
                return replacement

        queue = {
            "queue_id": executor.QUEUE_ID,
            "queue_name": executor.QUEUE_NAME,
            "registered_worker_count": 1,
            "workers": [{}],
        }
        lease = {
            "name": executor._lease_queue_name(execution_key),
            "status": "available",
        }
        with (
            mock.patch.object(executor, "_validate_selector"),
            mock.patch.object(
                executor, "_validate_source_d", return_value="sealed-source-d"
            ),
            mock.patch.object(executor, "_validate_teacher_gate"),
            mock.patch.object(executor, "_validate_teacher"),
            mock.patch.object(executor, "_validate_anchor_tasks"),
        ):
            receipt = executor.preflight(
                task_class=TaskClass,
                queue_reader=lambda _: queue,
                lease_reader=lambda _: lease,
            )
        inventory = receipt["existing_execution"]["children"]
        recorded = next(item for item in inventory if item["task_key"] == task_key)
        self.assertEqual(recorded["quarantined_task_ids"], [old.id])

        callback_origins: list[str] = []

        def fail_first_discovery(task_id: str, origin: str) -> None:
            callback_origins.append(origin)
            if origin == "cloned":
                raise RuntimeError(f"stop after replacement clone {task_id}")

        with self.assertRaisesRegex(RuntimeError, "stop after replacement clone"):
            executor._create_task(
                TaskClass,
                teacher_task=object(),
                controller_id=controller_id,
                task_key=task_key,
                source_d="source",
                parameters={},
                execution_key=execution_key,
                on_discovered=fail_first_discovery,
            )
        self.assertEqual(TaskClass.clone_calls, 1)
        self.assertIsNotNone(replacement)
        self.assertNotEqual(replacement.id, old.id)
        self.assertEqual(callback_origins, ["cloned", "quarantined"])

    def test_clone_is_quarantined_if_immediate_journal_persistence_fails(self) -> None:
        controller_id = "8" * 32
        execution_key = "9" * 64
        task_key = "train-r01-s02-resilient_v2x"
        task = _OrphanTask()
        task.name = executor._task_name(execution_key, task_key)
        task.parent = controller_id
        task.project = executor.PROJECT_ID

        class CloneTaskClass:
            @classmethod
            def clone(cls, **kwargs: object) -> _OrphanTask:
                return task

            @classmethod
            def get_task(cls, *, task_id: str) -> _OrphanTask:
                self.assertEqual(task_id, task.id)
                return task

        journal_origins: list[str] = []

        def journal_failure(task_id: str, origin: str) -> None:
            journal_origins.append(origin)
            if origin != "quarantined":
                raise RuntimeError(f"journal unavailable for {task_id} {origin}")

        with mock.patch.object(
            executor, "_select_unique_created_candidate", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "journal unavailable"):
                executor._create_task(
                    CloneTaskClass,
                    teacher_task=object(),
                    controller_id=controller_id,
                    task_key=task_key,
                    source_d="source",
                    parameters={},
                    execution_key=execution_key,
                    on_discovered=journal_failure,
                )
        self.assertEqual(task.status, "failed")
        self.assertTrue(task.archived)
        self.assertEqual(journal_origins, ["cloned", "quarantined"])

    def test_journal_failed_discovery_upload_is_copy_on_write_and_replaceable(
        self,
    ) -> None:
        class Artifact:
            def __init__(self, value: dict[str, object]) -> None:
                self.value = copy.deepcopy(value)

            def get(self) -> dict[str, object]:
                return copy.deepcopy(self.value)

        class Controller:
            def __init__(self) -> None:
                self.artifacts: dict[str, Artifact] = {}
                self.upload_calls = 0

            def upload_artifact(
                self,
                *,
                name: str,
                artifact_object: dict[str, object],
                wait_on_upload: bool,
            ) -> bool:
                if wait_on_upload is not True:
                    raise AssertionError("journal upload did not wait")
                self.upload_calls += 1
                if self.upload_calls == 1:
                    raise RuntimeError("first journal upload failed")
                self.artifacts[name] = Artifact(artifact_object)
                return True

        plan = executor.build_plan()
        execution_key = executor._execution_key(plan, executor.build_pinset())
        controller_id = "4" * 32
        task_key = "train-r01-s02-resilient_v2x"
        failed_task_id = "5" * 32
        replacement_task_id = "6" * 32
        controller = Controller()
        journal = executor._new_journal(
            plan=plan,
            execution_key=execution_key,
            controller_id=controller_id,
        )
        pristine = copy.deepcopy(journal)

        with self.assertRaisesRegex(RuntimeError, "first journal upload failed"):
            executor._record_journal_task(
                controller,
                journal,
                task_key=task_key,
                task_id=failed_task_id,
                state="discovered",
                server_status="created",
            )
        self.assertEqual(journal, pristine)

        executor._record_journal_task(
            controller,
            journal,
            task_key=task_key,
            task_id=failed_task_id,
            state="quarantined",
            server_status="failed",
            result={"quarantined_task_ids": [failed_task_id]},
        )
        journal["status"] = "paused_error"
        executor._persist_journal(controller, journal)

        restarted = executor._load_journal(
            controller,
            plan=plan,
            execution_key=execution_key,
            controller_id=controller_id,
        )
        self.assertEqual(
            executor._journal_row(restarted, task_key)["state"], "quarantined"
        )
        executor._record_journal_task(
            controller,
            restarted,
            task_key=task_key,
            task_id=replacement_task_id,
            state="discovered",
            server_status="created",
        )
        replacement = executor._journal_row(restarted, task_key)
        self.assertEqual(replacement["task_id"], replacement_task_id)
        self.assertEqual(replacement["state"], "discovered")
        self.assertEqual(
            replacement["result"], {"quarantined_task_ids": [failed_task_id]}
        )

    def test_restart_reconciles_server_quarantine_after_journal_upload_failure(
        self,
    ) -> None:
        class Artifact:
            def __init__(self, value: dict[str, object]) -> None:
                self.value = copy.deepcopy(value)

            def get(self) -> dict[str, object]:
                return copy.deepcopy(self.value)

        class Controller:
            def __init__(self) -> None:
                self.artifacts: dict[str, Artifact] = {}
                self.fail_next_upload = False

            def upload_artifact(
                self,
                *,
                name: str,
                artifact_object: dict[str, object],
                wait_on_upload: bool,
            ) -> bool:
                if wait_on_upload is not True:
                    raise AssertionError("journal upload did not wait")
                if self.fail_next_upload:
                    self.fail_next_upload = False
                    raise RuntimeError("quarantined journal upload failed")
                self.artifacts[name] = Artifact(artifact_object)
                return True

        plan = executor.build_plan()
        execution_key = executor._execution_key(plan, executor.build_pinset())
        controller_id = "a" * 32
        task_key = "train-r01-s02-resilient_v2x"
        old_task_id = "b" * 32
        replacement_task_id = "c" * 32
        controller = Controller()
        journal = executor._new_journal(
            plan=plan,
            execution_key=execution_key,
            controller_id=controller_id,
        )
        row = executor._journal_row(journal, task_key)
        row.update(
            {
                "task_id": old_task_id,
                "state": "discovered",
                "server_status": "created",
            }
        )
        journal["revision"] = 1
        journal["seal_sha256"] = executor._seal(journal)
        controller.artifacts[executor.JOURNAL_ARTIFACT] = Artifact(journal)

        controller.fail_next_upload = True
        with self.assertRaisesRegex(RuntimeError, "quarantined journal upload failed"):
            executor._record_journal_task(
                controller,
                journal,
                task_key=task_key,
                task_id=old_task_id,
                state="quarantined",
                server_status="failed",
                result={"quarantined_task_ids": [old_task_id]},
            )
        self.assertEqual(
            executor._journal_row(journal, task_key)["state"], "discovered"
        )
        persisted = controller.artifacts[executor.JOURNAL_ARTIFACT].get()
        self.assertEqual(
            executor._journal_row(persisted, task_key)["state"], "discovered"
        )

        orphan = _OrphanTask()
        orphan.id = old_task_id
        orphan.name = executor._task_name(execution_key, task_key)
        orphan.parent = controller_id
        orphan.project = executor.PROJECT_ID
        orphan.status = "failed"
        orphan.archived = True
        orphan.tags = [executor.ORPHAN_TAG]

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> _OrphanTask:
                self.assertEqual(task_id, old_task_id)
                return orphan

        restarted = executor._load_journal(
            controller,
            plan=plan,
            execution_key=execution_key,
            controller_id=controller_id,
        )
        executor._reconcile_quarantined_journal_tasks(
            TaskClass,
            controller,
            restarted,
            controller_id=controller_id,
            execution_key=execution_key,
        )
        self.assertEqual(
            executor._journal_row(restarted, task_key)["state"], "quarantined"
        )
        executor._record_journal_task(
            controller,
            restarted,
            task_key=task_key,
            task_id=replacement_task_id,
            state="discovered",
            server_status="created",
        )
        replacement = executor._journal_row(restarted, task_key)
        self.assertEqual(replacement["task_id"], replacement_task_id)
        self.assertEqual(replacement["result"], {"quarantined_task_ids": [old_task_id]})

    def test_metrics_require_exact_inventory_and_ranges(self) -> None:
        checkpoint = "2" * 64
        runs: list[dict[str, object]] = []
        condition_names = {
            "full": "Full",
            "l_fail": "L-Fail",
            "c_fail": "C-Fail",
        }
        for index, condition_id in enumerate(executor.CONDITION_IDS):
            values = {key: 1.0 for key in executor.METRIC_KEYS}
            values["resilient_v2x/sample_count"] = float(executor.SAMPLE_COUNT)
            values["resilient_v2x/car_ground_truth_count"] = float(
                executor.GROUND_TRUTH_COUNT
            )
            values["resilient_v2x/unsupported_sample_count"] = 0.0
            values["resilient_v2x/car_prediction_count"] = 10.0
            values["resilient_v2x/diagnostic_bev_match_050_count"] = 5.0
            suffix = condition_id.split("_", 2)[2]
            runs.append(
                {
                    "condition": condition_names[suffix],
                    "condition_id": condition_id,
                    "delay_ms": (0, 100, 200, 300)[index // 3],
                    "ground_truth_count": executor.GROUND_TRUTH_COUNT,
                    "metrics": values,
                    "prediction_content_sha256": "3" * 64,
                    "prediction_sha256": "4" * 64,
                    "predictions": f"predictions/{condition_id}.json",
                    "sample_count": executor.SAMPLE_COUNT,
                    "sample_ids_sha256": executor.SAMPLE_IDS_SHA256,
                    "unsupported_sample_count": 0,
                }
            )
        metrics = {
            "baseline": "resilient_v2x",
            "checkpoint": "/tmp/epoch_50.pth",
            "checkpoint_sha256": checkpoint,
            "complete": True,
            "expected_ground_truth_count": executor.GROUND_TRUTH_COUNT,
            "expected_sample_count": executor.SAMPLE_COUNT,
            "expected_unsupported_sample_count": 0,
            "manifest_content_sha256": executor.MANIFEST_CONTENT_SHA256,
            "overlay_index_content_sha256": executor.OVERLAY_INDEX_CONTENT_SHA256,
            "planned_run_count": 12,
            "protocol_id": executor.PROTOCOL_ID,
            "result_type": "resilient_v2x_controlled_baseline_metrics",
            "runs": runs,
            "sample_ids_sha256": executor.SAMPLE_IDS_SHA256,
            "schema_version": 1,
        }
        executor._validate_metrics(
            metrics, subject="resilient_v2x", checkpoint_sha256=checkpoint
        )
        forged = copy.deepcopy(metrics)
        forged["runs"][0]["metrics"]["resilient_v2x/car_bev_ap_r40_0.70"] = 101.0
        with self.assertRaisesRegex(executor.P1ExecutorError, "outside"):
            executor._validate_metrics(
                forged, subject="resilient_v2x", checkpoint_sha256=checkpoint
            )


if __name__ == "__main__":
    unittest.main()
