from __future__ import annotations

import copy
import contextlib
import builtins
import hashlib
import io
import json
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.resilient_v2x import clearml_p1_a100_multiseed_executor as base
from tools.resilient_v2x import clearml_p1_a100_parallel_v2_supervisor as supervisor


FROZEN_BASE_SHA256 = "c6dc31dd324f96b688b7dd82a2d3e50fb8a48cd14658dbf5ff96daed072b1298"
SEALED_PLAN_SHA256 = "43c1076057999ecb48dfa979f4ec6d952a62c4e4c02307cc350e8489a3728446"
SEALED_PINSET_SHA256 = (
    "3d44c31b0fd1f1c2ba4714a59828cd3f25f7ed2e1b64425618fb5e2e5d411eb7"
)
SEALED_EXECUTION_KEY = (
    "c4dfc01775ce835819685112d5bf9f3713ef15e94da5d38717a5f1d443bdb793"
)
CONTROLLER_ID = "c" * 32


def _new_journal() -> dict[str, object]:
    return base._new_journal(
        plan=base.build_plan(),
        execution_key=SEALED_EXECUTION_KEY,
        controller_id=CONTROLLER_ID,
    )


def _row(journal: dict[str, object], task_key: str) -> dict[str, object]:
    return base._journal_row(journal, task_key)


def _reseal(journal: dict[str, object]) -> None:
    journal["seal_sha256"] = base._seal(journal)


def _journal_candidate(
    prior: dict[str, object],
    *,
    task_key: str,
    task_id: str,
    state: str = "discovered",
    server_status: str = "created",
) -> dict[str, object]:
    candidate = copy.deepcopy(prior)
    _row(candidate, task_key).update(
        {
            "task_id": task_id,
            "state": state,
            "server_status": server_status,
            "result": None,
        }
    )
    candidate["status"] = "running"
    candidate["revision"] = int(prior["revision"]) + 1
    _reseal(candidate)
    return candidate


def _set_active(journal: dict[str, object], task_key: str, *, task_id: str) -> None:
    _row(journal, task_key).update(
        {
            "task_id": task_id,
            "state": "active",
            "server_status": "in_progress",
            "result": None,
        }
    )
    _reseal(journal)


def _set_training_completed(
    journal: dict[str, object], task_key: str, *, ordinal: int
) -> None:
    _row(journal, task_key).update(
        {
            "task_id": f"{ordinal + 1:032x}",
            "state": "completed",
            "server_status": "completed",
            "result": {
                "model_id": f"{ordinal + 101:032x}",
                "checkpoint_sha256": f"{ordinal + 201:064x}",
                "runtime": {},
            },
        }
    )
    _reseal(journal)


def _runtime_receipt(
    task_id: str, *, worker_id: str | None = None
) -> dict[str, object]:
    worker_id = worker_id or sorted(base.ALLOWED_WORKER_IDS)[0]
    contract = base.A100_RUNTIME_CONTRACT
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_a100_child_runtime",
        "task_id": task_id,
        "worker_id": worker_id,
        "queue_id": base.QUEUE_ID,
        "actual": {
            "hostname": contract["hostname"],
            "gpu_count": contract["gpu_count"],
            "gpu_type": list(contract["gpu_type_normalized"]),
            "gpu_memory": list(contract["gpu_memory_normalized"]),
            "gpu_driver_version": contract["gpu_driver_version"],
            "gpu_driver_cuda_version": contract["gpu_driver_cuda_version"],
            "python_version": contract["python_version"],
            "python_exec": contract["python_exec_allowlist"][0],
            "os": contract["os"],
        },
        "indirect_build_contract": {
            "native_build_task_id": base.NATIVE_BUILD_TASK_ID,
            "native_bundle_sha256": base.NATIVE_BUNDLE_SHA256,
            "build_manifest_sha256": base.BUILD_MANIFEST_SHA256,
            "gpu_compute_capability": contract["gpu_compute_capability"],
            "torch": contract["torch"],
            "torch_cuda": contract["torch_cuda"],
            "mmcv": contract["mmcv"],
            "mmengine": contract["mmengine"],
            "mmdet": contract["mmdet"],
            "mmdet3d": contract["mmdet3d"],
        },
    }
    receipt["seal_sha256"] = base._seal(receipt)
    return receipt


def _complete_journal(journal: dict[str, object]) -> None:
    training_rows = [row for row in journal["tasks"] if row["kind"] == "training"]
    evaluation_rows = [row for row in journal["tasks"] if row["kind"] == "evaluation"]
    for ordinal, row in enumerate(training_rows):
        task_id = f"{ordinal + 1:032x}"
        row.update(
            {
                "task_id": task_id,
                "state": "completed",
                "server_status": "completed",
                "result": {
                    "model_id": f"{ordinal + 101:032x}",
                    "checkpoint_sha256": f"{ordinal + 201:064x}",
                    "runtime": _runtime_receipt(task_id),
                },
            }
        )
    for ordinal, row in enumerate(evaluation_rows):
        task_id = f"{ordinal + 11:032x}"
        row.update(
            {
                "task_id": task_id,
                "state": "completed",
                "server_status": "completed",
                "result": {
                    "task_id": task_id,
                    "worker": sorted(base.ALLOWED_WORKER_IDS)[ordinal % 2],
                    "runtime": _runtime_receipt(
                        task_id,
                        worker_id=sorted(base.ALLOWED_WORKER_IDS)[ordinal % 2],
                    ),
                    "metrics_artifact_sha256": f"{ordinal + 301:064x}",
                    "prediction_evidence_artifact_sha256": f"{ordinal + 401:064x}",
                },
            }
        )
    journal["status"] = "completed"
    _reseal(journal)


def _completed_server_tasks(journal: dict[str, object]) -> dict[str, object]:
    return {
        str(row["task_id"]): SimpleNamespace(
            id=row["task_id"],
            status="completed",
            parent=CONTROLLER_ID,
            project=base.PROJECT_ID,
            name=base._task_name(SEALED_EXECUTION_KEY, str(row["task_key"])),
            data=SimpleNamespace(execution=SimpleNamespace(queue=base.QUEUE_ID)),
        )
        for row in journal["tasks"]
        if row["state"] == "completed"
    }


def _training_keys() -> list[str]:
    return [
        str(row["task_key"])
        for row in _new_journal()["tasks"]
        if row["kind"] == "training"
    ]


def _evaluation_keys() -> list[str]:
    return [
        str(row["task_key"])
        for row in _new_journal()["tasks"]
        if row["kind"] == "evaluation"
    ]


def _queue_receipt(
    *,
    running: dict[str, str | None] | None = None,
    queued_task_ids: list[str] | None = None,
) -> dict[str, object]:
    running = running or {}
    queued_task_ids = queued_task_ids or []
    workers = [
        {
            "worker_id": worker_id,
            "running_task_id": running.get(worker_id),
            "reported_queue_task_count": len(queued_task_ids),
        }
        for worker_id in sorted(base.ALLOWED_WORKER_IDS)
    ]
    return {
        "queue_id": base.QUEUE_ID,
        "queue_name": base.QUEUE_NAME,
        "queued_entry_count": len(queued_task_ids),
        "queued_task_ids": list(queued_task_ids),
        "registered_worker_count": len(workers),
        "workers": workers,
        "overlapping_gpu8_queue": {
            "queue_id": base.OVERLAPPING_GPU8_QUEUE_ID,
            "queue_name": base.OVERLAPPING_GPU8_QUEUE_NAME,
            "entry_count": 0,
            "entry_task_ids": [],
            "tags": [base.GPU8_FREEZE_TAG],
        },
        "overlapping_gpu8_worker": {
            "worker_id": base.OVERLAPPING_GPU8_WORKER_ID,
            "status": "idle",
            "queue_ids": [base.OVERLAPPING_GPU8_QUEUE_ID],
            "running_task_id": None,
        },
    }


def _exercise_execute_loop(
    *,
    journal: dict[str, object],
    queue_receipt: dict[str, object],
    monotonic_values: list[float],
    dispatch_error: Exception,
    invoke_pre_enqueue_recheck: bool,
) -> tuple[Exception, list[str], mock.Mock]:
    controller = SimpleNamespace(id=CONTROLLER_ID, status="created", artifacts={})
    source_d = SimpleNamespace(id=base.SOURCE_D_TASK_ID)
    teacher = SimpleNamespace(id=base.TEACHER_TASK_ID)
    events: list[str] = []

    class TaskClass:
        @classmethod
        def get_task(cls, *, task_id: str) -> object:
            if task_id == base.SOURCE_D_TASK_ID:
                return source_d
            if task_id == base.TEACHER_TASK_ID:
                return teacher
            raise AssertionError(f"unexpected task lookup: {task_id}")

    @contextlib.contextmanager
    def held(name: str) -> Iterator[object]:
        events.append(f"{name}:enter")
        try:
            yield {"name": name}
        finally:
            events.append(f"{name}:exit")

    def install(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        events.append("install-amendment")
        return {"seal_sha256": "a" * 64}

    def ensure(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        events.append("ensure-amendment")
        return {"seal_sha256": "a" * 64}

    def reconcile(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        if not callable(kwargs.get("completion_guard")):
            raise AssertionError("execute did not pass a completion guard")
        events.append("reconcile")
        return journal

    def dispatch(*args: object, **kwargs: object) -> object:
        del args
        events.append(f"dispatch:{kwargs['task_key']}")
        if invoke_pre_enqueue_recheck:
            kwargs["pre_enqueue_check"]()
        raise dispatch_error

    dispatch_mock = mock.Mock(side_effect=dispatch)
    base_receipt = {"existing_execution": {"controller_id": CONTROLLER_ID}}
    captured: Exception | None = None
    with (
        mock.patch.object(
            supervisor.base, "_execution_lock", return_value=held("local-lock")
        ),
        mock.patch.object(
            supervisor.base,
            "_server_execution_mutex",
            return_value=held("server-lock"),
        ),
        mock.patch.object(
            supervisor,
            "_read_base_preflight",
            return_value=(TaskClass, base_receipt),
        ),
        mock.patch.object(
            supervisor,
            "_load_existing_execution",
            return_value=(controller, journal),
        ),
        mock.patch.object(
            supervisor,
            "_validate_control_host",
            return_value=copy.deepcopy(supervisor.CONTROL_HOST_CONTRACT),
        ),
        mock.patch.object(supervisor, "_adopt_or_block_pending_journal_write"),
        mock.patch.object(supervisor, "_install_amendment", side_effect=install),
        mock.patch.object(supervisor.base, "_fresh", return_value=controller),
        mock.patch.object(supervisor.base, "_load_journal", return_value=journal),
        mock.patch.object(
            supervisor.base, "_validate_source_d", return_value="sealed-source-d"
        ),
        mock.patch.object(supervisor.base, "_validate_teacher"),
        mock.patch.object(supervisor, "_reconcile_quarantined_journal_tasks_safe"),
        mock.patch.object(
            supervisor, "_ensure_installed_amendment", side_effect=ensure
        ),
        mock.patch.object(supervisor, "_reconcile_journal", side_effect=reconcile),
        mock.patch.object(supervisor, "_dispatch_candidate", dispatch_mock),
        mock.patch.object(supervisor.time, "monotonic", side_effect=monotonic_values),
        mock.patch.object(supervisor.time, "sleep"),
    ):
        try:
            supervisor.execute_plan(
                authorization_token=supervisor.EXECUTION_TOKEN,
                poll_seconds=1.0,
                timeout_hours=1.0,
                task_class=TaskClass,
                queue_reader=lambda _task_class: queue_receipt,
                api_client=SimpleNamespace(),
            )
        except Exception as error:  # The harness deliberately stops the loop.
            captured = error
    if captured is None:
        raise AssertionError("execute harness unexpectedly returned")
    return captured, events, dispatch_mock


class ClearMLP1A100ParallelV2SupervisorTests(unittest.TestCase):
    def _exercise_final_revalidation(self, *, drift: bool) -> dict[str, object]:
        journal = _new_journal()
        _complete_journal(journal)
        journal["status"] = "running"
        _reseal(journal)
        before = copy.deepcopy(journal)
        children = _completed_server_tasks(journal)
        expected_ids = set(children)
        results_by_id = {
            str(row["task_id"]): copy.deepcopy(row["result"])
            for row in journal["tasks"]
        }
        controller = SimpleNamespace(id=CONTROLLER_ID, status="created", artifacts={})
        source_d_task = SimpleNamespace(id=base.SOURCE_D_TASK_ID)
        teacher = SimpleNamespace(id=base.TEACHER_TASK_ID)

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == base.SOURCE_D_TASK_ID:
                    return source_d_task
                if task_id == base.TEACHER_TASK_ID:
                    return teacher
                raise AssertionError(f"unexpected direct task lookup: {task_id}")

        def fresh(_task_class: object, task_id: str) -> object:
            if task_id == CONTROLLER_ID:
                return controller
            return children[task_id]

        normal_cache: set[str] | None = None
        fresh_cache_initial: list[set[str]] = []
        fresh_cache_final: list[set[str]] = []
        reconcile_calls = 0
        real_reconcile = supervisor._reconcile_journal

        def reconcile(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal normal_cache, reconcile_calls
            validated = kwargs["validated_completed_task_ids"]
            if not isinstance(validated, set):
                raise AssertionError("reconcile cache is not a set")
            reconcile_calls += 1
            if reconcile_calls == 1:
                normal_cache = validated
                validated.update(expected_ids)
                return journal
            fresh_cache_initial.append(set(validated))
            try:
                return real_reconcile(*args, **kwargs)
            finally:
                fresh_cache_final.append(set(validated))

        drifted = False

        def validate_training(task: object, **kwargs: object) -> dict[str, object]:
            nonlocal drifted
            del kwargs
            result = copy.deepcopy(results_by_id[str(task.id)])
            if drift and not drifted:
                if not isinstance(result, dict):
                    raise AssertionError("training result fixture is not a mapping")
                result["checkpoint_sha256"] = "f" * 64
                drifted = True
            return result

        def validate_evaluation(task: object, **kwargs: object) -> dict[str, object]:
            del kwargs
            result = results_by_id[str(task.id)]
            if not isinstance(result, dict):
                raise AssertionError("evaluation result fixture is not a mapping")
            return copy.deepcopy(result)

        upload_manifest = mock.Mock()
        persist_status = mock.Mock()
        complete_controller = mock.Mock()
        safe_write = mock.Mock()
        outcome: object | None = None
        error: Exception | None = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_execution_lock",
                    return_value=contextlib.nullcontext({}),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_server_execution_mutex",
                    return_value=contextlib.nullcontext({}),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor,
                    "_read_base_preflight",
                    return_value=(
                        TaskClass,
                        {"existing_execution": {"controller_id": CONTROLLER_ID}},
                    ),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor,
                    "_load_existing_execution",
                    return_value=(controller, journal),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor,
                    "_validate_control_host",
                    return_value=copy.deepcopy(supervisor.CONTROL_HOST_CONTRACT),
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor, "_adopt_or_block_pending_journal_write")
            )
            stack.enter_context(mock.patch.object(supervisor, "_install_amendment"))
            stack.enter_context(
                mock.patch.object(supervisor.base, "_fresh", side_effect=fresh)
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor.base, "_load_journal", return_value=journal
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_validate_source_d",
                    return_value="sealed-source-d",
                )
            )
            stack.enter_context(mock.patch.object(supervisor.base, "_validate_teacher"))
            stack.enter_context(
                mock.patch.object(
                    supervisor, "_reconcile_quarantined_journal_tasks_safe"
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor, "_ensure_installed_amendment")
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor, "_reconcile_journal", side_effect=reconcile
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor.base, "_require_server_unique_child")
            )
            training_validator = stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_validate_training_result",
                    side_effect=validate_training,
                )
            )
            evaluation_validator = stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_validate_evaluation_result",
                    side_effect=validate_evaluation,
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor, "_record_journal_task_safe", safe_write)
            )
            stack.enter_context(
                mock.patch.object(supervisor, "_upload_manifest_once", upload_manifest)
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor, "_persist_journal_status_safe", persist_status
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor.base, "_complete_controller", complete_controller
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor.time, "monotonic", return_value=0.0)
            )
            try:
                outcome = supervisor.execute_plan(
                    authorization_token=supervisor.EXECUTION_TOKEN,
                    poll_seconds=1.0,
                    timeout_hours=1.0,
                    task_class=TaskClass,
                    queue_reader=lambda _task_class: _queue_receipt(),
                    api_client=SimpleNamespace(),
                )
            except Exception as caught:
                error = caught
        return {
            "before": before,
            "journal": journal,
            "expected_ids": expected_ids,
            "normal_cache": normal_cache,
            "fresh_cache_initial": fresh_cache_initial,
            "fresh_cache_final": fresh_cache_final,
            "outcome": outcome,
            "error": error,
            "training_validator": training_validator,
            "evaluation_validator": evaluation_validator,
            "safe_write": safe_write,
            "upload_manifest": upload_manifest,
            "persist_status": persist_status,
            "complete_controller": complete_controller,
        }

    def _exercise_evaluation_candidate_revalidation(
        self, *, drift_round: int
    ) -> dict[str, object]:
        journal = _new_journal()
        for ordinal, task_key in enumerate(_training_keys()):
            _set_training_completed(journal, task_key, ordinal=ordinal)
        journal["status"] = "running"
        _reseal(journal)
        before = copy.deepcopy(journal)
        children = _completed_server_tasks(journal)
        training_ids = set(children)
        results_by_id = {
            str(row["task_id"]): copy.deepcopy(row["result"])
            for row in journal["tasks"]
            if row["kind"] == "training"
        }
        target_training_id = str(_row(journal, _training_keys()[0])["task_id"])
        candidate_key = _evaluation_keys()[0]
        candidate_task_id = "e" * 32
        candidate_task = SimpleNamespace(
            id=candidate_task_id,
            status="created",
            data=SimpleNamespace(execution=SimpleNamespace(queue="")),
        )
        controller = SimpleNamespace(id=CONTROLLER_ID, status="created", artifacts={})
        source_d_task = SimpleNamespace(id=base.SOURCE_D_TASK_ID)
        teacher = SimpleNamespace(id=base.TEACHER_TASK_ID)

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                if task_id == base.SOURCE_D_TASK_ID:
                    return source_d_task
                if task_id == base.TEACHER_TASK_ID:
                    return teacher
                raise AssertionError(f"unexpected direct task lookup: {task_id}")

        def fresh(_task_class: object, task_id: str) -> object:
            if task_id == CONTROLLER_ID:
                return controller
            if task_id == candidate_task_id:
                return candidate_task
            return children[task_id]

        real_reconcile = supervisor._reconcile_journal
        reconcile_calls = 0
        normal_cache: set[str] | None = None
        fresh_cache_initial: list[set[str]] = []
        fresh_cache_final: list[set[str]] = []

        def reconcile(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal normal_cache, reconcile_calls
            validated = kwargs["validated_completed_task_ids"]
            if not isinstance(validated, set):
                raise AssertionError("reconcile cache is not a set")
            reconcile_calls += 1
            if reconcile_calls == 1:
                normal_cache = validated
                validated.update(training_ids)
                return journal
            fresh_cache_initial.append(set(validated))
            try:
                return real_reconcile(*args, **kwargs)
            finally:
                fresh_cache_final.append(set(validated))

        validation_counts: dict[str, int] = {}

        def validate_training(task: object, **kwargs: object) -> dict[str, object]:
            del kwargs
            task_id = str(task.id)
            validation_counts[task_id] = validation_counts.get(task_id, 0) + 1
            result = copy.deepcopy(results_by_id[task_id])
            if (
                task_id == target_training_id
                and validation_counts[task_id] == drift_round
            ):
                if not isinstance(result, dict):
                    raise AssertionError("training result fixture is not a mapping")
                result["checkpoint_sha256"] = "f" * 64
            return result

        def create_task(*args: object, **kwargs: object) -> object:
            del args
            kwargs["on_discovered"](candidate_task_id, "cloned")
            return candidate_task

        def record_task(**kwargs: object) -> None:
            target = kwargs["journal"]
            if not isinstance(target, dict):
                raise AssertionError("safe writer received a non-dict journal")
            row = _row(target, str(kwargs["task_key"]))
            row.update(
                {
                    "task_id": kwargs["task_id"],
                    "state": kwargs["state"],
                    "server_status": kwargs["server_status"],
                    "result": kwargs.get("result", row.get("result")),
                }
            )
            target["revision"] = int(target["revision"]) + 1
            _reseal(target)

        create = mock.Mock(side_effect=create_task)
        safe_write = mock.Mock(side_effect=record_task)
        enqueue = mock.Mock()
        upload_manifest = mock.Mock()
        persist_status = mock.Mock()
        complete_controller = mock.Mock()
        error: Exception | None = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_execution_lock",
                    return_value=contextlib.nullcontext({}),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_server_execution_mutex",
                    return_value=contextlib.nullcontext({}),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor,
                    "_read_base_preflight",
                    return_value=(
                        TaskClass,
                        {"existing_execution": {"controller_id": CONTROLLER_ID}},
                    ),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor,
                    "_load_existing_execution",
                    return_value=(controller, journal),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor,
                    "_validate_control_host",
                    return_value=copy.deepcopy(supervisor.CONTROL_HOST_CONTRACT),
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor, "_adopt_or_block_pending_journal_write")
            )
            stack.enter_context(mock.patch.object(supervisor, "_install_amendment"))
            stack.enter_context(
                mock.patch.object(supervisor.base, "_fresh", side_effect=fresh)
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor.base, "_load_journal", return_value=journal
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_validate_source_d",
                    return_value="sealed-source-d",
                )
            )
            stack.enter_context(mock.patch.object(supervisor.base, "_validate_teacher"))
            stack.enter_context(
                mock.patch.object(
                    supervisor, "_reconcile_quarantined_journal_tasks_safe"
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor, "_ensure_installed_amendment")
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor, "_reconcile_journal", side_effect=reconcile
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor.base, "_require_server_unique_child")
            )
            training_validator = stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_validate_training_result",
                    side_effect=validate_training,
                )
            )
            evaluation_validator = stack.enter_context(
                mock.patch.object(
                    supervisor.base,
                    "_validate_evaluation_result",
                    side_effect=AssertionError(
                        "evaluation validator ran before enqueue"
                    ),
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor.base, "_create_task", create)
            )
            stack.enter_context(
                mock.patch.object(supervisor, "_record_journal_task_safe", safe_write)
            )
            stack.enter_context(
                mock.patch.object(supervisor.base, "_enqueue_once", enqueue)
            )
            stack.enter_context(
                mock.patch.object(supervisor, "_upload_manifest_once", upload_manifest)
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor, "_persist_journal_status_safe", persist_status
                )
            )
            stack.enter_context(
                mock.patch.object(
                    supervisor.base, "_complete_controller", complete_controller
                )
            )
            stack.enter_context(
                mock.patch.object(supervisor.time, "monotonic", return_value=0.0)
            )
            try:
                supervisor.execute_plan(
                    authorization_token=supervisor.EXECUTION_TOKEN,
                    poll_seconds=1.0,
                    timeout_hours=1.0,
                    task_class=TaskClass,
                    queue_reader=lambda _task_class: _queue_receipt(),
                    api_client=SimpleNamespace(),
                )
            except Exception as caught:
                error = caught
        return {
            "before": before,
            "journal": journal,
            "training_ids": training_ids,
            "normal_cache": normal_cache,
            "fresh_cache_initial": fresh_cache_initial,
            "fresh_cache_final": fresh_cache_final,
            "candidate_key": candidate_key,
            "error": error,
            "validation_counts": validation_counts,
            "training_validator": training_validator,
            "evaluation_validator": evaluation_validator,
            "create": create,
            "safe_write": safe_write,
            "enqueue": enqueue,
            "upload_manifest": upload_manifest,
            "persist_status": persist_status,
            "complete_controller": complete_controller,
        }

    def test_frozen_base_and_original_seals_are_exact(self) -> None:
        base_path = __import__(base.__name__, fromlist=["__file__"]).__file__
        self.assertIsNotNone(base_path)
        self.assertEqual(
            hashlib.sha256(Path(base_path).read_bytes()).hexdigest(),
            FROZEN_BASE_SHA256,
        )
        self.assertEqual(supervisor.FROZEN_BASE_SHA256, FROZEN_BASE_SHA256)
        self.assertEqual(supervisor.SEALED_PLAN_SHA256, SEALED_PLAN_SHA256)
        self.assertEqual(supervisor.SEALED_PINSET_SHA256, SEALED_PINSET_SHA256)
        self.assertEqual(supervisor.SEALED_EXECUTION_KEY, SEALED_EXECUTION_KEY)
        plan = base.build_plan()
        pinset = base.build_pinset()
        self.assertEqual(plan["seal_sha256"], SEALED_PLAN_SHA256)
        self.assertEqual(pinset["seal_sha256"], SEALED_PINSET_SHA256)
        self.assertEqual(base._execution_key(plan, pinset), SEALED_EXECUTION_KEY)

    def test_migration_intent_is_sealed_and_binds_existing_journal(self) -> None:
        journal = _new_journal()
        task_id = "1" * 32
        _set_active(journal, _training_keys()[0], task_id=task_id)
        intent = supervisor._build_migration_intent(
            controller_id=CONTROLLER_ID,
            journal=journal,
            active_task_ids=[task_id],
        )
        validated = supervisor._validate_migration_intent(
            intent,
            controller_id=CONTROLLER_ID,
            journal=journal,
            active_task_ids=[task_id],
        )
        self.assertEqual(validated["source"]["execution_key"], SEALED_EXECUTION_KEY)
        self.assertEqual(
            validated["source"]["journal_seal_sha256"], journal["seal_sha256"]
        )
        self.assertEqual(validated["source"]["active_task_ids"], [task_id])
        forged = copy.deepcopy(intent)
        forged["source"]["active_task_ids"] = []
        with self.assertRaises(supervisor.P1ParallelV2Error):
            supervisor._validate_migration_intent(
                forged,
                controller_id=CONTROLLER_ID,
                journal=journal,
                active_task_ids=[task_id],
            )

    def test_training_scheduler_caps_two_active_and_enqueues_at_most_one(self) -> None:
        journal = _new_journal()
        first, second = _training_keys()[:2]
        worker_ids = sorted(base.ALLOWED_WORKER_IDS)
        first_id, second_id = "1" * 32, "2" * 32
        _set_active(journal, first, task_id=first_id)
        decision = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(running={worker_ids[0]: first_id}),
        )
        self.assertEqual(decision["phase"], "training")
        self.assertLessEqual(len(decision["active_task_keys"]), 2)
        self.assertLessEqual(len(decision["enqueue_task_keys"]), 1)
        self.assertIn(first, decision["active_task_keys"])
        self.assertNotIn(first, decision["enqueue_task_keys"])
        self.assertEqual(decision["enqueue_task_keys"], [second])

        _set_active(journal, second, task_id=second_id)
        decision = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(
                running={worker_ids[0]: first_id, worker_ids[1]: second_id}
            ),
        )
        self.assertEqual(len(decision["active_task_keys"]), 2)
        self.assertEqual(decision["enqueue_task_keys"], [])

    def test_more_than_two_scoped_live_statuses_fails_closed(self) -> None:
        journal = _new_journal()
        worker_ids = sorted(base.ALLOWED_WORKER_IDS)
        first_id, second_id, third_id = "1" * 32, "2" * 32, "3" * 32
        for task_key, task_id in zip(
            _training_keys()[:3], [first_id, second_id, third_id], strict=True
        ):
            _set_active(journal, task_key, task_id=task_id)
        with self.assertRaisesRegex(
            supervisor.P1ParallelV2Error, "two|2|parallel|live"
        ):
            supervisor._scheduler_decision(
                journal=journal,
                queue_receipt=_queue_receipt(
                    running={worker_ids[0]: first_id, worker_ids[1]: second_id},
                    queued_task_ids=[third_id],
                ),
            )

    def test_initial_second_training_requires_exact_live_cutover_gate(self) -> None:
        journal = _new_journal()
        task_id = "1" * 32
        _set_active(journal, _training_keys()[0], task_id=task_id)
        blocked = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(
                running={supervisor.INITIAL_IDLE_WORKER_ID: task_id}
            ),
            require_initial_second_gate=True,
        )
        self.assertEqual(blocked["enqueue_task_keys"], [])
        self.assertEqual(
            blocked["blocked_reason"], "initial_second_training_gate_waiting"
        )

        allowed = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(
                running={supervisor.INITIAL_RUNNING_WORKER_ID: task_id}
            ),
            require_initial_second_gate=True,
        )
        self.assertEqual(allowed["enqueue_task_keys"], [_training_keys()[1]])

    def test_evaluation_waits_for_all_six_trainings_then_caps_one(self) -> None:
        journal = _new_journal()
        training_keys = _training_keys()
        for ordinal, task_key in enumerate(training_keys[:-1]):
            _set_training_completed(journal, task_key, ordinal=ordinal)
        decision = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(),
        )
        self.assertEqual(decision["phase"], "training")
        self.assertEqual(decision["enqueue_task_keys"], [training_keys[-1]])
        self.assertFalse(
            set(decision["enqueue_task_keys"]).intersection(_evaluation_keys())
        )

        _set_training_completed(
            journal, training_keys[-1], ordinal=len(training_keys) - 1
        )
        decision = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(),
        )
        self.assertEqual(decision["phase"], "evaluation")
        self.assertEqual(decision["enqueue_task_keys"], [_evaluation_keys()[0]])

        evaluation_id = "e" * 32
        _set_active(journal, _evaluation_keys()[0], task_id=evaluation_id)
        decision = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(
                running={sorted(base.ALLOWED_WORKER_IDS)[0]: evaluation_id}
            ),
        )
        self.assertEqual(decision["active_task_keys"], [_evaluation_keys()[0]])
        self.assertEqual(decision["enqueue_task_keys"], [])

    def test_foreign_target_worker_or_queue_residency_blocks_new_enqueue(self) -> None:
        journal = _new_journal()
        worker_ids = sorted(base.ALLOWED_WORKER_IDS)
        scoped_id = "1" * 32
        _set_active(journal, _training_keys()[0], task_id=scoped_id)
        receipt = _queue_receipt(
            running={worker_ids[0]: "f" * 32, worker_ids[1]: scoped_id}
        )
        decision = supervisor._scheduler_decision(
            journal=journal, queue_receipt=receipt
        )
        self.assertEqual(decision["enqueue_task_keys"], [])
        self.assertRegex(decision["blocked_reason"], "foreign|worker|capacity")

        journal = _new_journal()
        decision = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(running={worker_ids[0]: "f" * 32}),
        )
        self.assertEqual(decision["enqueue_task_keys"], [_training_keys()[0]])

        receipt = _queue_receipt(queued_task_ids=["f" * 32])
        decision = supervisor._scheduler_decision(
            journal=journal, queue_receipt=receipt
        )
        self.assertEqual(decision["enqueue_task_keys"], [])
        self.assertRegex(decision["blocked_reason"], "foreign|scoped")

    def test_gpu8_freeze_drift_is_rejected(self) -> None:
        for mutate in (
            lambda receipt: receipt["overlapping_gpu8_queue"].update(
                {"tags": [base.GPU8_ENABLE_TAG]}
            ),
            lambda receipt: receipt["overlapping_gpu8_queue"].update(
                {"entry_count": 1, "entry_task_ids": ["f" * 32]}
            ),
            lambda receipt: receipt["overlapping_gpu8_worker"].update(
                {"status": "busy", "running_task_id": "f" * 32}
            ),
        ):
            with self.subTest(mutate=mutate):
                receipt = _queue_receipt()
                mutate(receipt)
                with self.assertRaisesRegex(
                    supervisor.P1ParallelV2Error, "GPU8|overlapping"
                ):
                    supervisor._scheduler_decision(
                        journal=_new_journal(), queue_receipt=receipt
                    )

    def test_unreadable_target_identity_or_queue_inventory_fails_closed(self) -> None:
        for mutate in (
            lambda receipt: receipt["workers"][0].update(
                {"worker_id": "unexpected-worker"}
            ),
            lambda receipt: receipt.update(
                {"queued_entry_count": 1, "queued_task_ids": []}
            ),
        ):
            with self.subTest(mutate=mutate):
                receipt = _queue_receipt()
                mutate(receipt)
                with self.assertRaises(supervisor.P1ParallelV2Error):
                    supervisor._scheduler_decision(
                        journal=_new_journal(), queue_receipt=receipt
                    )

    def test_queue_receipt_lag_blocks_expansion_until_active_is_observed(self) -> None:
        journal = _new_journal()
        task_id = "1" * 32
        _set_active(journal, _training_keys()[0], task_id=task_id)
        decision = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(),
        )
        self.assertEqual(decision["enqueue_task_keys"], [])
        self.assertRegex(
            decision["blocked_reason"], "active|residency|receipt|snapshot|lag"
        )

    def test_execute_holds_both_locks_and_waits_when_foreign_work_uses_capacity(
        self,
    ) -> None:
        journal = _new_journal()
        task_id = "1" * 32
        _set_active(journal, _training_keys()[0], task_id=task_id)
        receipt = _queue_receipt(
            running={
                supervisor.INITIAL_RUNNING_WORKER_ID: task_id,
                supervisor.INITIAL_IDLE_WORKER_ID: "f" * 32,
            }
        )
        error, events, dispatch = _exercise_execute_loop(
            journal=journal,
            queue_receipt=receipt,
            monotonic_values=[0.0, 1.0, 2.0, 3601.0],
            dispatch_error=AssertionError("foreign occupancy must not dispatch"),
            invoke_pre_enqueue_recheck=False,
        )
        self.assertIsInstance(error, supervisor.P1RecoverableTimeout)
        dispatch.assert_not_called()
        self.assertLess(
            events.index("local-lock:enter"), events.index("server-lock:enter")
        )
        self.assertLess(
            events.index("server-lock:enter"), events.index("install-amendment")
        )
        self.assertLess(events.index("install-amendment"), events.index("reconcile"))
        self.assertEqual(events[-2:], ["server-lock:exit", "local-lock:exit"])

    def test_execute_initial_gate_dispatches_only_the_second_training(self) -> None:
        class LoopStopped(RuntimeError):
            pass

        journal = _new_journal()
        first, second = _training_keys()[:2]
        task_id = "1" * 32
        _set_active(journal, first, task_id=task_id)
        error, events, dispatch = _exercise_execute_loop(
            journal=journal,
            queue_receipt=_queue_receipt(
                running={supervisor.INITIAL_RUNNING_WORKER_ID: task_id}
            ),
            monotonic_values=[0.0, 1.0],
            dispatch_error=LoopStopped("stop after the first dispatch decision"),
            invoke_pre_enqueue_recheck=True,
        )
        self.assertIsInstance(error, LoopStopped)
        dispatch.assert_called_once()
        self.assertEqual(dispatch.call_args.kwargs["task_key"], second)
        self.assertNotEqual(dispatch.call_args.kwargs["task_key"], first)
        self.assertLess(
            events.index("install-amendment"), events.index(f"dispatch:{second}")
        )

    def test_final_branch_ignores_poll_cache_and_blocks_fresh_evidence_drift(
        self,
    ) -> None:
        observed = self._exercise_final_revalidation(drift=True)
        self.assertIsInstance(observed["error"], supervisor.P1ParallelV2Error)
        self.assertRegex(str(observed["error"]), "result evidence drifted")
        self.assertEqual(observed["normal_cache"], observed["expected_ids"])
        self.assertEqual(observed["fresh_cache_initial"], [set()])
        self.assertEqual(observed["fresh_cache_final"], [set()])
        observed["training_validator"].assert_called_once()
        observed["evaluation_validator"].assert_not_called()
        observed["safe_write"].assert_not_called()
        observed["upload_manifest"].assert_not_called()
        observed["persist_status"].assert_not_called()
        observed["complete_controller"].assert_not_called()
        self.assertEqual(observed["journal"], observed["before"])

    def test_normal_final_fresh_revalidation_covers_all_twelve_before_completion(
        self,
    ) -> None:
        observed = self._exercise_final_revalidation(drift=False)
        self.assertIsNone(observed["error"])
        manifest = observed["outcome"]
        self.assertIsInstance(manifest, dict)
        self.assertEqual(observed["normal_cache"], observed["expected_ids"])
        self.assertEqual(observed["fresh_cache_initial"], [set()])
        self.assertEqual(observed["fresh_cache_final"], [observed["expected_ids"]])
        self.assertEqual(observed["training_validator"].call_count, 6)
        self.assertEqual(observed["evaluation_validator"].call_count, 6)
        observed["safe_write"].assert_not_called()
        observed["upload_manifest"].assert_called_once()
        observed["persist_status"].assert_called_once()
        observed["complete_controller"].assert_called_once()
        validated = base._validate_execution_manifest(
            manifest,
            plan=base.build_plan(),
            execution_key=SEALED_EXECUTION_KEY,
            controller_id=CONTROLLER_ID,
        )
        self.assertEqual(validated, manifest)

    def test_evaluation_candidate_revalidates_six_trainings_before_clone_and_enqueue(
        self,
    ) -> None:
        before_clone = self._exercise_evaluation_candidate_revalidation(drift_round=1)
        self.assertIsInstance(before_clone["error"], supervisor.P1ParallelV2Error)
        self.assertRegex(str(before_clone["error"]), "result evidence drifted")
        self.assertEqual(before_clone["normal_cache"], before_clone["training_ids"])
        self.assertEqual(before_clone["fresh_cache_initial"], [set()])
        self.assertEqual(before_clone["fresh_cache_final"], [set()])
        before_clone["training_validator"].assert_called_once()
        before_clone["evaluation_validator"].assert_not_called()
        before_clone["create"].assert_not_called()
        before_clone["safe_write"].assert_not_called()
        before_clone["enqueue"].assert_not_called()
        before_clone["upload_manifest"].assert_not_called()
        before_clone["persist_status"].assert_not_called()
        before_clone["complete_controller"].assert_not_called()
        self.assertEqual(before_clone["journal"], before_clone["before"])

        pre_enqueue = self._exercise_evaluation_candidate_revalidation(drift_round=2)
        self.assertIsInstance(pre_enqueue["error"], supervisor.P1ParallelV2Error)
        self.assertRegex(str(pre_enqueue["error"]), "result evidence drifted")
        self.assertEqual(pre_enqueue["normal_cache"], pre_enqueue["training_ids"])
        self.assertEqual(pre_enqueue["fresh_cache_initial"], [set(), set()])
        self.assertEqual(
            pre_enqueue["fresh_cache_final"],
            [pre_enqueue["training_ids"], set()],
        )
        self.assertEqual(pre_enqueue["training_validator"].call_count, 7)
        pre_enqueue["evaluation_validator"].assert_not_called()
        pre_enqueue["create"].assert_called_once()
        self.assertEqual(pre_enqueue["safe_write"].call_count, 2)
        pre_enqueue["enqueue"].assert_not_called()
        pre_enqueue["upload_manifest"].assert_not_called()
        pre_enqueue["persist_status"].assert_not_called()
        pre_enqueue["complete_controller"].assert_not_called()
        candidate_row = _row(pre_enqueue["journal"], str(pre_enqueue["candidate_key"]))
        self.assertEqual(candidate_row["state"], "prepared")
        self.assertEqual(candidate_row["server_status"], "created")

    def test_frozen_base_controller_recovery_fails_closed_on_amendment_artifact(
        self,
    ) -> None:
        plan = base.build_plan()
        pinset = base.build_pinset()
        execution_key = base._execution_key(plan, pinset)
        parameters = base._controller_parameters(
            plan=plan, pinset=pinset, execution_key=execution_key
        )

        class Controller:
            id = CONTROLLER_ID
            status = "created"
            parent = base.SELECTOR_TASK_ID
            project = base.PROJECT_ID
            name = base._controller_name(execution_key)
            tags = [base._execution_tag(execution_key), base.CONTROLLER_TAG]
            artifacts = {
                base.PLAN_ARTIFACT: SimpleNamespace(),
                base.PINSET_ARTIFACT: SimpleNamespace(),
                base.JOURNAL_ARTIFACT: SimpleNamespace(),
                supervisor.AMENDMENT_ARTIFACT: SimpleNamespace(),
            }
            data = SimpleNamespace(execution=SimpleNamespace(queue=""))
            output_uri = base.FILES_SERVER_URI

            @staticmethod
            def get_parameters(**kwargs: object) -> dict[str, object]:
                del kwargs
                return dict(parameters)

        controller = Controller()
        with (
            mock.patch.object(
                base, "_select_unique_created_candidate", return_value=controller
            ),
            mock.patch.object(base, "_add_tags") as add_tags,
            mock.patch.object(base, "_set_parameters") as set_parameters,
            mock.patch.object(base, "_upload_artifact") as upload,
        ):
            with self.assertRaisesRegex(
                base.P1ExecutorError, "artifact inventory contains unknown"
            ):
                base._find_or_create_controller(
                    SimpleNamespace(),
                    plan=plan,
                    pinset=pinset,
                    execution_key=execution_key,
                )
        add_tags.assert_not_called()
        set_parameters.assert_not_called()
        upload.assert_not_called()

    def test_installed_amendment_allows_monotonic_progress_but_not_rebinding(
        self,
    ) -> None:
        source_journal = _new_journal()
        training_key = _training_keys()[0]
        source_task_id = "1" * 32
        _set_active(source_journal, training_key, task_id=source_task_id)
        source_journal["revision"] = 3
        _reseal(source_journal)
        amendment = supervisor._build_migration_intent(
            controller_id=CONTROLLER_ID,
            journal=source_journal,
            active_task_ids=[source_task_id],
        )

        class Artifact:
            def __init__(self, value: dict[str, object]) -> None:
                self.value = value

            def get(self) -> dict[str, object]:
                return copy.deepcopy(self.value)

        controller = SimpleNamespace(
            id=CONTROLLER_ID,
            artifacts={supervisor.AMENDMENT_ARTIFACT: Artifact(amendment)},
        )

        class TaskClass:
            calls = 0

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                self.assertEqual(task_id, CONTROLLER_ID)
                cls.calls += 1
                return controller

        observed = supervisor._ensure_installed_amendment(
            TaskClass, CONTROLLER_ID, source_journal, [source_task_id]
        )
        self.assertEqual(observed, amendment)
        self.assertEqual(TaskClass.calls, 1)

        advanced = copy.deepcopy(source_journal)
        _row(advanced, training_key).update(
            {
                "state": "completed",
                "server_status": "completed",
                "result": {
                    "model_id": "2" * 32,
                    "checkpoint_sha256": "3" * 64,
                    "runtime": {},
                },
            }
        )
        advanced["revision"] += 1
        _reseal(advanced)
        self.assertEqual(
            supervisor._ensure_installed_amendment(
                TaskClass, CONTROLLER_ID, advanced, []
            ),
            amendment,
        )

        rebound = copy.deepcopy(advanced)
        _row(rebound, training_key)["task_id"] = "4" * 32
        rebound["revision"] += 1
        _reseal(rebound)
        with self.assertRaisesRegex(
            supervisor.P1ParallelV2Error, "identity|task|binding|source"
        ):
            supervisor._ensure_installed_amendment(
                TaskClass, CONTROLLER_ID, rebound, []
            )

        rolled_back = copy.deepcopy(advanced)
        _row(rolled_back, training_key).update(
            {
                "task_id": source_task_id,
                "state": "discovered",
                "server_status": "created",
                "result": None,
            }
        )
        rolled_back["revision"] += 1
        _reseal(rolled_back)
        with self.assertRaisesRegex(
            supervisor.P1ParallelV2Error, "rollback|identity|task|source"
        ):
            supervisor._ensure_installed_amendment(
                TaskClass, CONTROLLER_ID, rolled_back, []
            )

        predating = copy.deepcopy(source_journal)
        predating["revision"] -= 1
        _reseal(predating)
        with self.assertRaisesRegex(
            supervisor.P1ParallelV2Error, "predates|revision|source"
        ):
            supervisor._ensure_installed_amendment(
                TaskClass, CONTROLLER_ID, predating, [source_task_id]
            )

        controller.artifacts = {}
        with self.assertRaisesRegex(supervisor.P1ParallelV2Error, "missing|amendment"):
            supervisor._ensure_installed_amendment(
                TaskClass, CONTROLLER_ID, advanced, []
            )

    def test_manifest_is_rebuilt_in_sealed_plan_order(self) -> None:
        journal = _new_journal()
        _complete_journal(journal)
        journal["tasks"] = list(reversed(journal["tasks"]))
        _reseal(journal)
        manifest = supervisor._build_execution_manifest(
            journal=journal, controller_id=CONTROLLER_ID
        )
        expected = [
            (
                pair["training"]["subject"],
                pair["training"]["seed_index"],
                pair["training"]["training_seed"],
            )
            for pair in base.build_plan()["pairs"]
        ]
        self.assertEqual(
            [
                (row["subject"], row["seed_index"], row["training_seed"])
                for row in manifest["results"]
            ],
            expected,
        )
        validated = base._validate_execution_manifest(
            manifest,
            plan=base.build_plan(),
            execution_key=SEALED_EXECUTION_KEY,
            controller_id=CONTROLLER_ID,
        )
        self.assertEqual(validated, manifest)

    def test_amendment_install_recovers_transport_error_after_fresh_commit(
        self,
    ) -> None:
        journal = _new_journal()
        controller = SimpleNamespace(id=CONTROLLER_ID, artifacts={})

        class Artifact:
            def __init__(self, value: dict[str, object]) -> None:
                self.value = value

            def get(self) -> dict[str, object]:
                return copy.deepcopy(self.value)

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                self.assertEqual(task_id, CONTROLLER_ID)
                return controller

        def committed_upload(
            _controller: object, name: str, value: dict[str, object]
        ) -> None:
            controller.artifacts[name] = Artifact(value)
            raise RuntimeError("reply lost after committed artifact upload")

        with mock.patch.object(
            supervisor.base, "_upload_artifact", side_effect=committed_upload
        ) as upload:
            amendment = supervisor._install_amendment(TaskClass, controller, journal)
        self.assertEqual(
            amendment["source"]["journal_seal_sha256"], journal["seal_sha256"]
        )
        self.assertIn(supervisor.AMENDMENT_ARTIFACT, controller.artifacts)
        upload.assert_called_once()

    def test_journal_wal_is_fsynced_before_remote_upload(self) -> None:
        prior = _new_journal()
        candidate = _journal_candidate(
            prior, task_key=_training_keys()[0], task_id="1" * 32
        )
        local = copy.deepcopy(prior)
        controller = SimpleNamespace(id=CONTROLLER_ID)
        remote = {"journal": copy.deepcopy(prior)}
        events: list[str] = []
        real_fsync = supervisor.os.fsync

        def fsync(descriptor: int) -> None:
            events.append("fsync")
            real_fsync(descriptor)

        def read_remote(
            *args: object, **kwargs: object
        ) -> tuple[object, dict[str, object]]:
            del args, kwargs
            return controller, copy.deepcopy(remote["journal"])

        def upload(_controller: object, name: str, value: dict[str, object]) -> None:
            self.assertEqual(name, base.JOURNAL_ARTIFACT)
            self.assertTrue(pending_path.exists())
            self.assertGreaterEqual(events.count("fsync"), 2)
            events.append("upload")
            remote["journal"] = copy.deepcopy(value)

        with tempfile.TemporaryDirectory() as directory:
            pending_path = Path(directory) / supervisor.PENDING_JOURNAL_FILENAME
            with (
                mock.patch.object(
                    supervisor, "_pending_journal_path", return_value=pending_path
                ),
                mock.patch.object(
                    supervisor, "_read_remote_journal", side_effect=read_remote
                ),
                mock.patch.object(
                    supervisor.base, "_upload_artifact", side_effect=upload
                ),
                mock.patch.object(supervisor.os, "fsync", side_effect=fsync),
            ):
                supervisor._commit_journal_candidate(
                    task_class=SimpleNamespace(),
                    controller=controller,
                    journal=local,
                    candidate=candidate,
                )
            self.assertEqual(local, candidate)
            self.assertEqual(remote["journal"], candidate)
            self.assertFalse(pending_path.exists())
            self.assertLess(events.index("fsync"), events.index("upload"))

    def test_wal_before_upload_crash_then_remote_prior_blocks_without_rewrite(
        self,
    ) -> None:
        prior = _new_journal()
        candidate = _journal_candidate(
            prior, task_key=_training_keys()[0], task_id="1" * 32
        )
        local = copy.deepcopy(prior)
        controller = SimpleNamespace(id=CONTROLLER_ID)
        real_write = supervisor._write_pending_journal_receipt

        class SimulatedCrash(RuntimeError):
            pass

        def write_then_crash(receipt: dict[str, object]) -> None:
            real_write(receipt)
            raise SimulatedCrash("crash after durable WAL and before upload")

        with tempfile.TemporaryDirectory() as directory:
            pending_path = Path(directory) / supervisor.PENDING_JOURNAL_FILENAME
            with (
                mock.patch.object(
                    supervisor, "_pending_journal_path", return_value=pending_path
                ),
                mock.patch.object(
                    supervisor,
                    "_read_remote_journal",
                    return_value=(controller, copy.deepcopy(prior)),
                ),
                mock.patch.object(
                    supervisor,
                    "_write_pending_journal_receipt",
                    side_effect=write_then_crash,
                ),
                mock.patch.object(supervisor.base, "_upload_artifact") as upload,
            ):
                with self.assertRaises(SimulatedCrash):
                    supervisor._commit_journal_candidate(
                        task_class=SimpleNamespace(),
                        controller=controller,
                        journal=local,
                        candidate=candidate,
                    )
            self.assertTrue(pending_path.exists())
            upload.assert_not_called()

            with (
                mock.patch.object(
                    supervisor, "_pending_journal_path", return_value=pending_path
                ),
                mock.patch.object(
                    supervisor,
                    "_read_remote_journal",
                    return_value=(controller, copy.deepcopy(prior)),
                ),
                mock.patch.object(
                    supervisor, "_write_pending_journal_receipt"
                ) as rewrite,
                mock.patch.object(supervisor.base, "_upload_artifact") as upload,
            ):
                with self.assertRaisesRegex(
                    supervisor.P1ParallelV2Error, "unresolved|different write"
                ):
                    supervisor._commit_journal_candidate(
                        task_class=SimpleNamespace(),
                        controller=controller,
                        journal=local,
                        candidate=candidate,
                    )
            rewrite.assert_not_called()
            upload.assert_not_called()
            self.assertTrue(pending_path.exists())

    def test_remote_commit_then_confirmation_crash_is_adopted_on_restart(self) -> None:
        prior = _new_journal()
        candidate = _journal_candidate(
            prior, task_key=_training_keys()[0], task_id="1" * 32
        )
        local = copy.deepcopy(prior)
        controller = SimpleNamespace(id=CONTROLLER_ID)
        remote = {"journal": copy.deepcopy(prior)}
        calls = 0

        def read_remote(
            *args: object, **kwargs: object
        ) -> tuple[object, dict[str, object]]:
            nonlocal calls
            del args, kwargs
            calls += 1
            if calls == 1:
                return controller, copy.deepcopy(remote["journal"])
            if calls == 2:
                raise RuntimeError("crash before remote confirmation")
            return controller, copy.deepcopy(remote["journal"])

        def committed_upload(
            _controller: object, name: str, value: dict[str, object]
        ) -> None:
            self.assertEqual(name, base.JOURNAL_ARTIFACT)
            remote["journal"] = copy.deepcopy(value)

        with tempfile.TemporaryDirectory() as directory:
            pending_path = Path(directory) / supervisor.PENDING_JOURNAL_FILENAME
            with (
                mock.patch.object(
                    supervisor, "_pending_journal_path", return_value=pending_path
                ),
                mock.patch.object(
                    supervisor, "_read_remote_journal", side_effect=read_remote
                ),
                mock.patch.object(
                    supervisor.base,
                    "_upload_artifact",
                    side_effect=committed_upload,
                ) as upload,
            ):
                with self.assertRaisesRegex(
                    supervisor.P1ParallelV2Error, "could not be confirmed|uncertain"
                ):
                    supervisor._commit_journal_candidate(
                        task_class=SimpleNamespace(),
                        controller=controller,
                        journal=local,
                        candidate=candidate,
                    )
                self.assertTrue(pending_path.exists())
                supervisor._adopt_or_block_pending_journal_write(
                    SimpleNamespace(), CONTROLLER_ID, local
                )
                upload.assert_called_once()
            self.assertEqual(local, candidate)
            self.assertEqual(remote["journal"], candidate)
            self.assertFalse(pending_path.exists())

    def test_third_valid_remote_journal_creates_poison_for_manual_audit(self) -> None:
        prior = _new_journal()
        candidate = _journal_candidate(
            prior, task_key=_training_keys()[0], task_id="1" * 32
        )
        third = _journal_candidate(
            prior, task_key=_training_keys()[0], task_id="2" * 32
        )
        controller = SimpleNamespace(id=CONTROLLER_ID)
        with tempfile.TemporaryDirectory() as directory:
            pending_path = Path(directory) / supervisor.PENDING_JOURNAL_FILENAME
            with (
                mock.patch.object(
                    supervisor, "_pending_journal_path", return_value=pending_path
                ),
                mock.patch.object(
                    supervisor,
                    "_read_remote_journal",
                    return_value=(controller, copy.deepcopy(third)),
                ),
                mock.patch.object(supervisor.base, "_upload_artifact") as upload,
            ):
                with self.assertRaisesRegex(
                    supervisor.P1ParallelV2Error, "stale|manual audit|overwrite"
                ):
                    supervisor._commit_journal_candidate(
                        task_class=SimpleNamespace(),
                        controller=controller,
                        journal=copy.deepcopy(prior),
                        candidate=candidate,
                    )
                poison = supervisor._read_pending_journal_receipt(
                    controller_id=CONTROLLER_ID
                )
            self.assertIsNotNone(poison)
            self.assertEqual(poison["status"], "poisoned")
            self.assertEqual(poison["observed_journal"], third)
            upload.assert_not_called()

    def test_discovery_upload_committed_then_raised_is_adopted_not_quarantined(
        self,
    ) -> None:
        journal = _new_journal()
        task_key = _training_keys()[0]
        task_id = "1" * 32
        controller = SimpleNamespace(id=CONTROLLER_ID)
        remote = {"journal": copy.deepcopy(journal)}

        def read_remote(
            *args: object, **kwargs: object
        ) -> tuple[object, dict[str, object]]:
            del args, kwargs
            return controller, copy.deepcopy(remote["journal"])

        def committed_upload(
            _controller: object, name: str, value: dict[str, object]
        ) -> None:
            self.assertEqual(name, base.JOURNAL_ARTIFACT)
            remote["journal"] = copy.deepcopy(value)
            raise RuntimeError("transport failed after commit")

        with tempfile.TemporaryDirectory() as directory:
            pending_path = Path(directory) / supervisor.PENDING_JOURNAL_FILENAME
            with (
                mock.patch.object(
                    supervisor, "_pending_journal_path", return_value=pending_path
                ),
                mock.patch.object(
                    supervisor, "_read_remote_journal", side_effect=read_remote
                ),
                mock.patch.object(
                    supervisor.base, "_upload_artifact", side_effect=committed_upload
                ),
            ):
                supervisor._record_discovery(
                    task_class=SimpleNamespace(),
                    controller=controller,
                    journal=journal,
                    task_key=task_key,
                    task_id=task_id,
                    origin="cloned",
                )
            self.assertFalse(pending_path.exists())
        self.assertEqual(_row(journal, task_key)["state"], "discovered")
        self.assertEqual(_row(journal, task_key)["task_id"], task_id)
        self.assertNotEqual(_row(journal, task_key)["state"], "quarantined")

    def test_unresolved_discovery_blocks_quarantine_same_revision_callback(
        self,
    ) -> None:
        journal = _new_journal()
        prior = copy.deepcopy(journal)
        task_key = _training_keys()[0]
        task_id = "1" * 32
        controller = SimpleNamespace(id=CONTROLLER_ID)
        upload_calls = 0

        def failed_upload(*args: object, **kwargs: object) -> None:
            nonlocal upload_calls
            del args, kwargs
            upload_calls += 1
            raise RuntimeError("upload did not commit")

        with tempfile.TemporaryDirectory() as directory:
            pending_path = Path(directory) / supervisor.PENDING_JOURNAL_FILENAME
            with (
                mock.patch.object(
                    supervisor, "_pending_journal_path", return_value=pending_path
                ),
                mock.patch.object(
                    supervisor,
                    "_read_remote_journal",
                    return_value=(controller, copy.deepcopy(prior)),
                ),
                mock.patch.object(
                    supervisor.base, "_upload_artifact", side_effect=failed_upload
                ),
            ):
                with self.assertRaisesRegex(
                    supervisor.P1ParallelV2Error, "unresolved|uncertain"
                ):
                    supervisor._record_discovery(
                        task_class=SimpleNamespace(),
                        controller=controller,
                        journal=journal,
                        task_key=task_key,
                        task_id=task_id,
                        origin="cloned",
                    )
                pending = supervisor._read_pending_journal_receipt(
                    controller_id=CONTROLLER_ID
                )
                with self.assertRaisesRegex(
                    supervisor.P1ParallelV2Error, "unresolved|different write"
                ):
                    supervisor._record_discovery(
                        task_class=SimpleNamespace(),
                        controller=controller,
                        journal=journal,
                        task_key=task_key,
                        task_id=task_id,
                        origin="quarantined",
                    )
            self.assertTrue(pending_path.exists())
        self.assertEqual(upload_calls, 1)
        self.assertEqual(journal, prior)
        self.assertEqual(
            _row(pending["candidate_journal"], task_key)["state"], "discovered"
        )

    def test_restart_adopts_discovery_then_orphan_reconciles_next_revision(
        self,
    ) -> None:
        prior = _new_journal()
        task_key = _training_keys()[0]
        task_id = "1" * 32
        discovered = _journal_candidate(prior, task_key=task_key, task_id=task_id)
        local = copy.deepcopy(prior)
        controller = SimpleNamespace(id=CONTROLLER_ID)
        remote = {"journal": copy.deepcopy(discovered)}
        orphan = SimpleNamespace(
            id=task_id,
            status="failed",
            parent=CONTROLLER_ID,
            project=base.PROJECT_ID,
            name=base._task_name(SEALED_EXECUTION_KEY, task_key),
            tags=[
                base.ORPHAN_TAG,
                base._execution_tag(SEALED_EXECUTION_KEY),
                f"p1-task-key:{task_key}",
            ],
            data=SimpleNamespace(execution=SimpleNamespace(queue="")),
        )

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                self.assertEqual(task_id, orphan.id)
                return orphan

        def read_remote(
            *args: object, **kwargs: object
        ) -> tuple[object, dict[str, object]]:
            del args, kwargs
            return controller, copy.deepcopy(remote["journal"])

        def upload(_controller: object, name: str, value: dict[str, object]) -> None:
            self.assertEqual(name, base.JOURNAL_ARTIFACT)
            remote["journal"] = copy.deepcopy(value)

        pending = supervisor._build_pending_journal_receipt(
            controller_id=CONTROLLER_ID,
            prior=prior,
            candidate=discovered,
            status="pending",
            observed=prior,
        )
        with tempfile.TemporaryDirectory() as directory:
            pending_path = Path(directory) / supervisor.PENDING_JOURNAL_FILENAME
            with (
                mock.patch.object(
                    supervisor, "_pending_journal_path", return_value=pending_path
                ),
                mock.patch.object(
                    supervisor, "_read_remote_journal", side_effect=read_remote
                ),
                mock.patch.object(
                    supervisor.base, "_upload_artifact", side_effect=upload
                ),
            ):
                supervisor._write_pending_journal_receipt(pending)
                supervisor._adopt_or_block_pending_journal_write(
                    TaskClass, CONTROLLER_ID, local
                )
                self.assertEqual(local, discovered)
                self.assertFalse(pending_path.exists())
                supervisor._reconcile_quarantined_journal_tasks_safe(
                    task_class=TaskClass,
                    controller=controller,
                    journal=local,
                    controller_id=CONTROLLER_ID,
                )
            self.assertFalse(pending_path.exists())
        self.assertEqual(local["revision"], discovered["revision"] + 1)
        self.assertEqual(_row(local, task_key)["state"], "quarantined")
        self.assertEqual(remote["journal"], local)

    def test_quarantined_failed_child_is_rescheduled_and_rebound_with_history(
        self,
    ) -> None:
        journal = _new_journal()
        task_key = _training_keys()[0]
        old_task_id = "1" * 32
        new_task_id = "2" * 32
        _row(journal, task_key).update(
            {
                "task_id": old_task_id,
                "state": "quarantined",
                "server_status": "failed",
                "result": {"quarantined_task_ids": [old_task_id]},
            }
        )
        journal["revision"] = 1
        _reseal(journal)

        decision = supervisor._scheduler_decision(
            journal=journal,
            queue_receipt=_queue_receipt(),
        )
        self.assertEqual(decision["phase"], "training")
        self.assertEqual(decision["enqueue_task_keys"], [task_key])
        self.assertIsNone(decision["blocked_reason"])

        task = SimpleNamespace(
            id=new_task_id,
            status="created",
            data=SimpleNamespace(execution=SimpleNamespace(queue="")),
        )
        controller = SimpleNamespace(id=CONTROLLER_ID)

        class TaskClass:
            enqueue_calls = 0

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                self.assertEqual(task_id, new_task_id)
                return task

            @classmethod
            def enqueue(cls, *, task: object, queue_id: str, force: bool) -> object:
                cls.enqueue_calls += 1
                self.assertEqual(queue_id, base.QUEUE_ID)
                self.assertFalse(force)
                task.status = "queued"
                task.data.execution.queue = queue_id
                return True

        def create_task(*args: object, **kwargs: object) -> object:
            del args
            kwargs["on_discovered"](new_task_id, "cloned")
            return task

        committed_candidates: list[dict[str, object]] = []

        def commit_candidate(**kwargs: object) -> None:
            target = kwargs["journal"]
            candidate = kwargs["candidate"]
            if not isinstance(target, dict) or not isinstance(candidate, dict):
                raise AssertionError("journal commit received invalid objects")
            base._validate_journal(
                candidate,
                plan=base.build_plan(),
                execution_key=SEALED_EXECUTION_KEY,
                controller_id=CONTROLLER_ID,
            )
            committed_candidates.append(copy.deepcopy(candidate))
            target.clear()
            target.update(copy.deepcopy(candidate))

        with (
            mock.patch.object(supervisor.base, "_create_task", side_effect=create_task),
            mock.patch.object(supervisor.base, "_require_server_unique_child"),
            mock.patch.object(supervisor, "_adopt_or_block_pending_journal_write"),
            mock.patch.object(
                supervisor, "_commit_journal_candidate", side_effect=commit_candidate
            ),
        ):
            observed = supervisor._dispatch_candidate(
                task_class=TaskClass,
                controller=controller,
                journal=journal,
                task_key=task_key,
                teacher=SimpleNamespace(),
                source_d="sealed-source-d",
                authorization_token=supervisor.EXECUTION_TOKEN,
            )

        self.assertIs(observed, task)
        self.assertEqual(TaskClass.enqueue_calls, 1)
        self.assertEqual(len(committed_candidates), 3)
        rebound = _row(journal, task_key)
        self.assertEqual(rebound["task_id"], new_task_id)
        self.assertEqual(rebound["state"], "active")
        self.assertEqual(rebound["server_status"], "queued")
        self.assertEqual(rebound["result"], {"quarantined_task_ids": [old_task_id]})

    def test_dispatch_is_journal_first_and_recovers_committed_enqueue_error(
        self,
    ) -> None:
        journal = _new_journal()
        task_key = _training_keys()[0]
        task = SimpleNamespace(
            id="1" * 32,
            status="created",
            data=SimpleNamespace(execution=SimpleNamespace(queue="")),
        )
        controller = SimpleNamespace(id=CONTROLLER_ID)
        events: list[str] = []

        class TaskClass:
            enqueue_calls = 0

            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                self.assertEqual(task_id, task.id)
                return task

            @classmethod
            def enqueue(cls, *, task: object, queue_id: str, force: bool) -> object:
                cls.enqueue_calls += 1
                events.append("enqueue")
                self.assertEqual(queue_id, base.QUEUE_ID)
                self.assertFalse(force)
                task.status = "queued"
                task.data.execution.queue = queue_id
                raise RuntimeError("reply lost after committed enqueue")

        def create_task(*args: object, **kwargs: object) -> object:
            del args
            callback = kwargs["on_discovered"]
            callback(task.id, "cloned")
            return task

        def record_task(**kwargs: object) -> None:
            target = kwargs["journal"]
            if not isinstance(target, dict):
                raise AssertionError("safe writer received a non-dict journal")
            state = str(kwargs["state"])
            events.append(f"journal:{state}")
            _row(target, str(kwargs["task_key"])).update(
                {
                    "task_id": kwargs["task_id"],
                    "state": state,
                    "server_status": kwargs["server_status"],
                    "result": kwargs.get("result"),
                }
            )
            target["revision"] += 1
            _reseal(target)

        with (
            mock.patch.object(supervisor.base, "_create_task", side_effect=create_task),
            mock.patch.object(
                supervisor, "_record_journal_task_safe", side_effect=record_task
            ),
            mock.patch.object(supervisor.base, "_require_server_unique_child"),
        ):
            observed = supervisor._dispatch_candidate(
                task_class=TaskClass,
                controller=controller,
                journal=journal,
                task_key=task_key,
                teacher=SimpleNamespace(),
                source_d="sealed-source-d",
                authorization_token=supervisor.EXECUTION_TOKEN,
            )
        self.assertEqual(observed.id, task.id)
        self.assertEqual(TaskClass.enqueue_calls, 1)
        self.assertLess(events.index("journal:discovered"), events.index("enqueue"))
        self.assertLess(events.index("journal:prepared"), events.index("enqueue"))
        self.assertEqual(_row(journal, task_key)["state"], "active")
        self.assertEqual(_row(journal, task_key)["server_status"], "queued")

    def test_dispatch_propagates_ambiguous_clone_without_enqueue_or_journal_drift(
        self,
    ) -> None:
        journal = _new_journal()
        before = copy.deepcopy(journal)
        controller = SimpleNamespace(id=CONTROLLER_ID)
        with (
            mock.patch.object(
                supervisor.base,
                "_create_task",
                side_effect=supervisor.base.P1ExecutorError("duplicate active tasks"),
            ),
            mock.patch.object(supervisor, "_record_journal_task_safe") as record,
            mock.patch.object(supervisor.base, "_enqueue_once") as enqueue,
            mock.patch.object(
                supervisor.base,
                "_active_named_tasks",
                return_value=[SimpleNamespace(), SimpleNamespace()],
            ),
        ):
            with self.assertRaisesRegex(
                supervisor.P1ExecutorError, "duplicate|ambiguous|active"
            ):
                supervisor._dispatch_candidate(
                    task_class=SimpleNamespace(),
                    controller=controller,
                    journal=journal,
                    task_key=_training_keys()[0],
                    teacher=SimpleNamespace(),
                    source_d="sealed-source-d",
                    authorization_token=supervisor.EXECUTION_TOKEN,
                )
        self.assertEqual(journal, before)
        record.assert_not_called()
        enqueue.assert_not_called()

    def test_reconcile_adopts_active_and_harvests_completed_without_enqueue(
        self,
    ) -> None:
        journal = _new_journal()
        active_key, completed_key = _training_keys()[:2]
        active_id, completed_id = "1" * 32, "2" * 32
        _set_active(journal, active_key, task_id=active_id)
        _set_active(journal, completed_key, task_id=completed_id)
        tasks = {
            active_id: SimpleNamespace(
                id=active_id,
                status="in_progress",
                parent=CONTROLLER_ID,
                project=base.PROJECT_ID,
                name=base._task_name(SEALED_EXECUTION_KEY, active_key),
                data=SimpleNamespace(execution=SimpleNamespace(queue=base.QUEUE_ID)),
            ),
            completed_id: SimpleNamespace(
                id=completed_id,
                status="completed",
                parent=CONTROLLER_ID,
                project=base.PROJECT_ID,
                name=base._task_name(SEALED_EXECUTION_KEY, completed_key),
                data=SimpleNamespace(execution=SimpleNamespace(queue=base.QUEUE_ID)),
            ),
        }
        controller = SimpleNamespace(id=CONTROLLER_ID)
        model = {
            "model_id": "3" * 32,
            "checkpoint_sha256": "4" * 64,
            "runtime": {},
        }

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                return tasks[task_id]

        def record_task(**kwargs: object) -> None:
            target = kwargs["journal"]
            if not isinstance(target, dict):
                raise AssertionError("safe writer received a non-dict journal")
            _row(target, str(kwargs["task_key"])).update(
                {
                    "task_id": kwargs["task_id"],
                    "state": kwargs["state"],
                    "server_status": kwargs["server_status"],
                    "result": kwargs.get("result"),
                }
            )
            target["revision"] += 1
            _reseal(target)

        with (
            mock.patch.object(
                supervisor, "_record_journal_task_safe", side_effect=record_task
            ),
            mock.patch.object(
                supervisor.base, "_validate_training_result", return_value=model
            ) as validate_training,
            mock.patch.object(
                supervisor.base,
                "_enqueue_once",
                side_effect=AssertionError("reconcile attempted enqueue"),
            ) as enqueue,
            mock.patch.object(supervisor.base, "_require_server_unique_child"),
        ):
            supervisor._reconcile_journal(
                task_class=TaskClass,
                controller=controller,
                journal=journal,
                source_d="sealed-source-d",
            )
        self.assertEqual(_row(journal, active_key)["state"], "active")
        self.assertEqual(_row(journal, completed_key)["state"], "completed")
        self.assertEqual(_row(journal, completed_key)["result"], model)
        validate_training.assert_called_once()
        enqueue.assert_not_called()

    def test_reconcile_rejects_completed_child_regressing_to_live_status(self) -> None:
        task_key = _training_keys()[0]
        for status in ("queued", "in_progress"):
            with self.subTest(status=status):
                journal = _new_journal()
                _set_training_completed(journal, task_key, ordinal=0)
                before = copy.deepcopy(journal)
                task_id = str(_row(journal, task_key)["task_id"])
                live = SimpleNamespace(
                    id=task_id,
                    status=status,
                    parent=CONTROLLER_ID,
                    project=base.PROJECT_ID,
                    name=base._task_name(SEALED_EXECUTION_KEY, task_key),
                    data=SimpleNamespace(
                        execution=SimpleNamespace(queue=base.QUEUE_ID)
                    ),
                )
                task_class = SimpleNamespace(get_task=mock.Mock(return_value=live))

                with (
                    mock.patch.object(supervisor.base, "_require_server_unique_child"),
                    mock.patch.object(
                        supervisor.base, "_validate_training_result"
                    ) as validate_training,
                    mock.patch.object(
                        supervisor, "_record_journal_task_safe"
                    ) as safe_write,
                    mock.patch.object(supervisor.base, "_enqueue_once") as enqueue,
                ):
                    with self.assertRaisesRegex(
                        supervisor.P1ParallelV2Error,
                        rf"completed child .* regressed to {status}",
                    ):
                        supervisor._reconcile_journal(
                            task_class=task_class,
                            controller=SimpleNamespace(id=CONTROLLER_ID),
                            journal=journal,
                            source_d="sealed-source-d",
                        )
                task_class.get_task.assert_called_once_with(task_id=task_id)
                validate_training.assert_not_called()
                safe_write.assert_not_called()
                enqueue.assert_not_called()
                self.assertEqual(journal, before)

    def test_reconcile_rejects_drifted_completed_result_without_wal_or_write(
        self,
    ) -> None:
        journal = _new_journal()
        task_key = _training_keys()[0]
        _set_training_completed(journal, task_key, ordinal=0)
        row = _row(journal, task_key)
        task_id = str(row["task_id"])
        recorded_result = copy.deepcopy(row["result"])
        before = copy.deepcopy(journal)
        fresh_result = copy.deepcopy(recorded_result)
        if not isinstance(fresh_result, dict):
            raise AssertionError("completed training fixture result is not a mapping")
        fresh_result["checkpoint_sha256"] = "f" * 64
        completed = SimpleNamespace(
            id=task_id,
            status="completed",
            parent=CONTROLLER_ID,
            project=base.PROJECT_ID,
            name=base._task_name(SEALED_EXECUTION_KEY, task_key),
            data=SimpleNamespace(execution=SimpleNamespace(queue=base.QUEUE_ID)),
        )
        task_class = SimpleNamespace(get_task=mock.Mock(return_value=completed))

        with (
            mock.patch.object(supervisor.base, "_require_server_unique_child"),
            mock.patch.object(
                supervisor.base,
                "_validate_training_result",
                return_value=fresh_result,
            ) as validate_training,
            mock.patch.object(supervisor, "_record_journal_task_safe") as safe_write,
            mock.patch.object(
                supervisor, "_write_pending_journal_receipt"
            ) as wal_write,
            mock.patch.object(supervisor.base, "_enqueue_once") as enqueue,
        ):
            with self.assertRaisesRegex(
                supervisor.P1ParallelV2Error, "result evidence drifted"
            ):
                supervisor._reconcile_journal(
                    task_class=task_class,
                    controller=SimpleNamespace(id=CONTROLLER_ID),
                    journal=journal,
                    source_d="sealed-source-d",
                )
        validate_training.assert_called_once()
        safe_write.assert_not_called()
        wal_write.assert_not_called()
        enqueue.assert_not_called()
        self.assertEqual(journal, before)

    def test_reconcile_can_fill_missing_result_for_completed_child(self) -> None:
        journal = _new_journal()
        task_key = _training_keys()[0]
        _set_training_completed(journal, task_key, ordinal=0)
        row = _row(journal, task_key)
        task_id = str(row["task_id"])
        row["result"] = None
        _reseal(journal)
        fresh_result = {
            "model_id": "2" * 32,
            "checkpoint_sha256": "3" * 64,
            "runtime": {},
        }
        completed = SimpleNamespace(
            id=task_id,
            status="completed",
            parent=CONTROLLER_ID,
            project=base.PROJECT_ID,
            name=base._task_name(SEALED_EXECUTION_KEY, task_key),
            data=SimpleNamespace(execution=SimpleNamespace(queue=base.QUEUE_ID)),
        )
        task_class = SimpleNamespace(get_task=mock.Mock(return_value=completed))

        def record_task(**kwargs: object) -> None:
            target = kwargs["journal"]
            if not isinstance(target, dict):
                raise AssertionError("safe writer received a non-dict journal")
            _row(target, str(kwargs["task_key"]))["result"] = kwargs["result"]
            target["revision"] = int(target["revision"]) + 1
            _reseal(target)

        with (
            mock.patch.object(supervisor.base, "_require_server_unique_child"),
            mock.patch.object(
                supervisor.base,
                "_validate_training_result",
                return_value=fresh_result,
            ) as validate_training,
            mock.patch.object(
                supervisor, "_record_journal_task_safe", side_effect=record_task
            ) as safe_write,
            mock.patch.object(supervisor.base, "_enqueue_once") as enqueue,
        ):
            supervisor._reconcile_journal(
                task_class=task_class,
                controller=SimpleNamespace(id=CONTROLLER_ID),
                journal=journal,
                source_d="sealed-source-d",
            )
        validate_training.assert_called_once()
        safe_write.assert_called_once()
        self.assertEqual(safe_write.call_args.kwargs["result"], fresh_result)
        enqueue.assert_not_called()
        self.assertEqual(_row(journal, task_key)["result"], fresh_result)

    def test_reconcile_gpu8_drift_before_validation_never_validates_or_writes(
        self,
    ) -> None:
        journal = _new_journal()
        task_key = _training_keys()[0]
        task_id = "1" * 32
        _set_active(journal, task_key, task_id=task_id)
        before = copy.deepcopy(journal)
        completed = SimpleNamespace(
            id=task_id,
            status="completed",
            parent=CONTROLLER_ID,
            project=base.PROJECT_ID,
            name=base._task_name(SEALED_EXECUTION_KEY, task_key),
            data=SimpleNamespace(execution=SimpleNamespace(queue=base.QUEUE_ID)),
        )

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                self.assertEqual(task_id, completed.id)
                return completed

        guard = mock.Mock(
            side_effect=supervisor.P1ParallelV2Error(
                "overlapping GPU8 worker binding drifted"
            )
        )
        with (
            mock.patch.object(supervisor.base, "_require_server_unique_child"),
            mock.patch.object(
                supervisor.base, "_validate_training_result"
            ) as validate_training,
            mock.patch.object(supervisor, "_record_journal_task_safe") as safe_write,
        ):
            with self.assertRaisesRegex(supervisor.P1ParallelV2Error, "GPU8.*drifted"):
                supervisor._reconcile_journal(
                    task_class=TaskClass,
                    controller=SimpleNamespace(id=CONTROLLER_ID),
                    journal=journal,
                    source_d="sealed-source-d",
                    completion_guard=guard,
                )
        guard.assert_called_once_with()
        validate_training.assert_not_called()
        safe_write.assert_not_called()
        self.assertEqual(journal, before)

    def test_reconcile_gpu8_drift_after_validation_discards_result_without_write(
        self,
    ) -> None:
        journal = _new_journal()
        task_key = _training_keys()[0]
        task_id = "1" * 32
        _set_active(journal, task_key, task_id=task_id)
        before = copy.deepcopy(journal)
        completed = SimpleNamespace(
            id=task_id,
            status="completed",
            parent=CONTROLLER_ID,
            project=base.PROJECT_ID,
            name=base._task_name(SEALED_EXECUTION_KEY, task_key),
            data=SimpleNamespace(execution=SimpleNamespace(queue=base.QUEUE_ID)),
        )
        result = {
            "model_id": "2" * 32,
            "checkpoint_sha256": "3" * 64,
            "runtime": {},
        }

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                self.assertEqual(task_id, completed.id)
                return completed

        guard = mock.Mock(
            side_effect=[
                None,
                supervisor.P1ParallelV2Error("overlapping GPU8 worker binding drifted"),
            ]
        )
        with (
            mock.patch.object(supervisor.base, "_require_server_unique_child"),
            mock.patch.object(
                supervisor.base, "_validate_training_result", return_value=result
            ) as validate_training,
            mock.patch.object(supervisor, "_record_journal_task_safe") as safe_write,
        ):
            with self.assertRaisesRegex(supervisor.P1ParallelV2Error, "GPU8.*drifted"):
                supervisor._reconcile_journal(
                    task_class=TaskClass,
                    controller=SimpleNamespace(id=CONTROLLER_ID),
                    journal=journal,
                    source_d="sealed-source-d",
                    completion_guard=guard,
                )
        self.assertEqual(guard.call_count, 2)
        validate_training.assert_called_once()
        safe_write.assert_not_called()
        self.assertEqual(journal, before)

    def test_default_mode_is_local_dry_run_without_clearml_import(self) -> None:
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "clearml" or name.startswith("clearml."):
                raise AssertionError("dry-run imported ClearML")
            return real_import(name, *args, **kwargs)

        output = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=guarded_import):
            with contextlib.redirect_stdout(output):
                self.assertEqual(supervisor.main([]), 0)
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt["default_mode"], "dry_run")
        self.assertEqual(receipt["execution_key"], SEALED_EXECUTION_KEY)
        self.assertFalse(receipt["remote_mutation_authorized"])

    def test_preflight_reports_pending_wal_without_deleting_it(self) -> None:
        journal = _new_journal()
        candidate = _journal_candidate(
            journal, task_key=_training_keys()[0], task_id="1" * 32
        )
        controller = SimpleNamespace(id=CONTROLLER_ID, status="created", artifacts={})
        base_receipt = {
            "seal_sha256": "a" * 64,
            "global_mutex": {"status": "available"},
            "queue": _queue_receipt(),
        }
        pending = supervisor._build_pending_journal_receipt(
            controller_id=CONTROLLER_ID,
            prior=journal,
            candidate=candidate,
            status="pending",
            observed=journal,
        )
        with tempfile.TemporaryDirectory() as directory:
            pending_path = Path(directory) / supervisor.PENDING_JOURNAL_FILENAME
            with mock.patch.object(
                supervisor, "_pending_journal_path", return_value=pending_path
            ):
                supervisor._write_pending_journal_receipt(pending)
            with (
                mock.patch.object(
                    supervisor, "_pending_journal_path", return_value=pending_path
                ),
                mock.patch.object(
                    supervisor,
                    "_validate_control_host",
                    return_value=copy.deepcopy(supervisor.CONTROL_HOST_CONTRACT),
                ),
                mock.patch.object(
                    supervisor,
                    "_read_base_preflight",
                    return_value=(SimpleNamespace(), base_receipt),
                ),
                mock.patch.object(
                    supervisor,
                    "_load_existing_execution",
                    return_value=(controller, journal),
                ),
                mock.patch.object(
                    supervisor,
                    "_remove_pending_journal_receipt",
                    side_effect=AssertionError("read-only preflight removed WAL"),
                ) as remove,
            ):
                receipt = supervisor.preflight(task_class=SimpleNamespace())
            self.assertTrue(pending_path.exists())
        self.assertEqual(receipt["remote_mutation_count"], 0)
        self.assertEqual(receipt["pending_journal_write"]["status"], "pending")
        self.assertEqual(
            receipt["scheduler_decision"]["blocked_reason"],
            "durable_pending_journal_write",
        )
        remove.assert_not_called()

    def test_control_host_mismatch_blocks_before_reads_or_locks(self) -> None:
        host_error = supervisor.P1ParallelV2Error("control host binding drifted")
        with (
            mock.patch.object(
                supervisor, "_validate_control_host", side_effect=host_error
            ) as validate_host,
            mock.patch.object(supervisor, "_read_base_preflight") as remote_read,
        ):
            with self.assertRaisesRegex(supervisor.P1ParallelV2Error, "control host"):
                supervisor.preflight(task_class=SimpleNamespace())
        validate_host.assert_called_once()
        remote_read.assert_not_called()

        with (
            mock.patch.object(
                supervisor,
                "_validate_control_host",
                side_effect=supervisor.P1ParallelV2Error(
                    "control host binding drifted"
                ),
            ) as validate_host,
            mock.patch.object(supervisor.base, "_execution_lock") as local_lock,
            mock.patch.object(
                supervisor.base, "_server_execution_mutex"
            ) as server_lock,
            mock.patch.object(supervisor, "_read_base_preflight") as remote_read,
        ):
            with self.assertRaisesRegex(supervisor.P1ParallelV2Error, "control host"):
                supervisor.execute_plan(
                    authorization_token=supervisor.EXECUTION_TOKEN,
                    poll_seconds=1.0,
                    timeout_hours=1.0,
                    task_class=SimpleNamespace(),
                    api_client=SimpleNamespace(),
                )
        validate_host.assert_called_once()
        local_lock.assert_not_called()
        server_lock.assert_not_called()
        remote_read.assert_not_called()

    def test_preflight_cli_never_dispatches_execution(self) -> None:
        receipt = {
            "schema_version": 1,
            "document_type": supervisor.PREFLIGHT_DOCUMENT_TYPE,
            "status": "validated",
            "remote_mutation_count": 0,
            "seal_sha256": "a" * 64,
        }
        output = io.StringIO()
        with (
            mock.patch.object(
                supervisor,
                "_validate_control_host",
                return_value=copy.deepcopy(supervisor.CONTROL_HOST_CONTRACT),
            ),
            mock.patch.object(supervisor, "preflight", return_value=receipt) as read,
            mock.patch.object(
                supervisor,
                "execute_plan",
                side_effect=AssertionError("preflight dispatched execution"),
            ) as execute,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(supervisor.main(["--preflight"]), 0)
        self.assertEqual(json.loads(output.getvalue()), receipt)
        read.assert_called_once()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
