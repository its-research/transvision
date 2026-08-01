#!/usr/bin/env python3
"""Schedule the fixed post-main RTX5090 training suite as a linear pipeline."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple
from urllib.parse import urlsplit


MAIN_TASK_ID = "6b0a279537a3419f83dca85864d7b4e1"
DEFAULT_WORKER_QUEUE = "GPU4-5090"
DEFAULT_PIPELINE_QUEUE = "services"
DEFAULT_DRAFT_TASK_IDS: Mapping[str, str] = {}
CLEARML_TASK_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
CLEAN_TEACHER_MODEL_NAME = "ResilientV2X clean teacher"
FILES_SERVER_URI = "http://10.100.34.118:8081"
EXPERIMENT_MAX_EPOCHS = 50


class ExperimentSpec(NamedTuple):
    name: str
    requires_teacher: bool


EXPERIMENT_SPECS = (
    ExperimentSpec("ptf_linear", True),
    ExperimentSpec("ptf_none", True),
    ExperimentSpec("router_static", True),
    ExperimentSpec("router_uniform", True),
    ExperimentSpec("no_reliability", True),
    ExperimentSpec("no_delay_metadata", True),
    ExperimentSpec("no_distillation", False),
    ExperimentSpec("concat_capacity_matched", True),
    ExperimentSpec("v2x_vit", False),
    ExperimentSpec("cobevt", False),
    ExperimentSpec("coformernet", False),
    ExperimentSpec("bevfusion", False),
    ExperimentSpec("ffnet", False),
)
EXPERIMENT_ORDER = tuple(spec.name for spec in EXPERIMENT_SPECS)
EXPERIMENT_BY_NAME = {spec.name: spec for spec in EXPERIMENT_SPECS}


def _normalized_task_status(task: object) -> str:
    status = getattr(task, "status", None)
    if callable(status):
        status = status()
    if status is None:
        getter = getattr(task, "get_status", None)
        if callable(getter):
            status = getter()
    value = getattr(status, "value", status)
    return str(value).rsplit(".", 1)[-1].lower()


def _require_unique_teacher_output_model(
    task: object,
    *,
    expected_task_id: str,
) -> object:
    if _normalized_task_status(task) != "completed":
        raise RuntimeError("main training task is not completed")
    models = task.get_models()
    if not isinstance(models, Mapping):
        raise RuntimeError("main training task returned an invalid model mapping")
    outputs = models.get("output")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise RuntimeError("main training task has no output model sequence")
    candidates = [
        model
        for model in outputs
        if getattr(model, "name", None) == CLEAN_TEACHER_MODEL_NAME
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "main training task must expose exactly one clean-teacher OutputModel"
        )
    model = candidates[0]
    if str(getattr(model, "task", "") or "") != expected_task_id:
        raise RuntimeError("clean-teacher OutputModel task mismatch")
    parsed = urlsplit(str(getattr(model, "url", "") or ""))
    if (
        parsed.hostname != "10.100.34.118"
        or parsed.port != 8081
        or parsed.scheme not in {"http", "https"}
    ):
        raise RuntimeError("clean-teacher OutputModel is outside the .34 files server")
    return model


def _task_id(value: str, context: str) -> str:
    if CLEARML_TASK_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase 32-hex ClearML task ID")
    return value


def _draft_binding(value: str) -> tuple[str, str]:
    experiment, separator, task_id = value.partition("=")
    if not separator or experiment not in EXPERIMENT_BY_NAME:
        raise argparse.ArgumentTypeError(
            "draft binding must be EXPERIMENT=32_HEX_TASK_ID"
        )
    try:
        return experiment, _task_id(task_id, f"draft {experiment}")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-task-id", default=MAIN_TASK_ID)
    parser.add_argument(
        "--draft-task-id",
        action="append",
        type=_draft_binding,
        default=[],
        metavar="EXPERIMENT=TASK_ID",
        help=(
            "bind one immutable suite member to its prepared draft task; "
            "provide all 13 unless DEFAULT_DRAFT_TASK_IDS is populated"
        ),
    )
    parser.add_argument("--worker-queue", default=DEFAULT_WORKER_QUEUE)
    parser.add_argument("--pipeline-queue", default=DEFAULT_PIPELINE_QUEUE)
    parser.add_argument("--project", default="ResilientV2X/Training")
    parser.add_argument(
        "--name",
        default="ResilientV2X RTX5090 sequential 13-experiment suite",
    )
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--main-poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the resolved DAG without importing or contacting ClearML",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="wait for the remote pipeline controller to finish",
    )
    return parser


def _resolve_draft_task_ids(
    bindings: Sequence[tuple[str, str]],
    *,
    defaults: Mapping[str, str] = DEFAULT_DRAFT_TASK_IDS,
) -> dict[str, str]:
    resolved = dict(defaults)
    for experiment, task_id in bindings:
        if experiment in resolved and resolved[experiment] != task_id:
            raise ValueError(f"conflicting draft task IDs for {experiment!r}")
        resolved[experiment] = task_id
    if set(resolved) != set(EXPERIMENT_ORDER):
        missing = sorted(set(EXPERIMENT_ORDER) - set(resolved))
        extra = sorted(set(resolved) - set(EXPERIMENT_ORDER))
        raise ValueError(
            f"draft bindings must match all 13 experiments; "
            f"missing={missing}, extra={extra}"
        )
    for experiment, task_id in resolved.items():
        _task_id(task_id, f"draft {experiment}")
    if len(set(resolved.values())) != len(EXPERIMENT_ORDER):
        raise ValueError("each experiment requires a distinct draft task ID")
    return {name: resolved[name] for name in EXPERIMENT_ORDER}


def _step_parameter_override(
    experiment: str,
    *,
    main_task_id: str,
    predecessor_task_id: str,
) -> dict[str, object]:
    spec = EXPERIMENT_BY_NAME[experiment]
    override: dict[str, object] = {
        "Args/experiment_from_task": experiment,
        "Args/gpus": 4,
        "Args/stage": "all",
        "Args/max_epochs": EXPERIMENT_MAX_EPOCHS,
        "Args/amp": False,
        "Args/predecessor_task_id": predecessor_task_id,
    }
    if spec.requires_teacher:
        override["Args/teacher_task_id"] = main_task_id
    return override


def build_step_plan(
    *,
    draft_task_ids: Mapping[str, str],
    main_task_id: str,
    worker_queue: str,
) -> list[dict[str, object]]:
    """Return the immutable, training-only linear DAG without using ClearML."""

    _task_id(main_task_id, "main task")
    if not worker_queue.strip():
        raise ValueError("worker queue must be non-empty")
    resolved = _resolve_draft_task_ids((), defaults=draft_task_ids)
    plan: list[dict[str, object]] = []
    parent: str | None = None
    previous_experiment: str | None = None
    for index, experiment in enumerate(EXPERIMENT_ORDER, start=1):
        step_name = f"train_{index:02d}_{experiment}"
        predecessor_task_id = (
            main_task_id
            if previous_experiment is None
            else resolved[previous_experiment]
        )
        plan.append(
            {
                "name": step_name,
                "experiment": experiment,
                "base_task_id": resolved[experiment],
                "parents": [] if parent is None else [parent],
                "parameter_override": _step_parameter_override(
                    experiment,
                    main_task_id=main_task_id,
                    predecessor_task_id=predecessor_task_id,
                ),
                "execution_queue": worker_queue,
            }
        )
        parent = step_name
        previous_experiment = experiment
    return plan


def _wait_for_completed_main(
    task_class: object,
    *,
    main_task_id: str,
    poll_seconds: float,
) -> object:
    if poll_seconds <= 0:
        raise ValueError("main poll interval must be positive")
    main_task = task_class.get_task(task_id=main_task_id)
    status = _normalized_task_status(main_task)
    if status != "completed":
        if status in {"failed", "stopped", "closed"}:
            raise RuntimeError(f"main task ended without completion: {status!r}")
        waiter = getattr(main_task, "wait_for_status", None)
        if not callable(waiter):
            raise RuntimeError("main task cannot wait for completion")
        waiter(
            status=("completed",),
            raise_on_status=("failed", "stopped", "closed"),
            check_interval_sec=poll_seconds,
        )
        reloader = getattr(main_task, "reload", None)
        if callable(reloader):
            reloader()
    _require_unique_teacher_output_model(
        main_task,
        expected_task_id=main_task_id,
    )
    return main_task


def _parameter_matches(actual: object, expected: object) -> bool:
    if type(expected) is bool:
        if type(actual) is bool:
            return actual is expected
        return str(actual).casefold() == str(expected).casefold()
    if type(expected) is int:
        return str(actual) == str(expected)
    return actual == expected


def _require_created_immutable_draft(
    task_class: object,
    *,
    experiment: str,
    draft_task_id: str,
    expected_parameters: Mapping[str, object],
) -> object:
    task = task_class.get_task(task_id=draft_task_id)
    status = _normalized_task_status(task)
    if status != "created":
        raise RuntimeError(
            f"draft task for {experiment!r} must be created, got {status!r}"
        )
    getter = getattr(task, "get_parameters", None)
    parameters = getter() if callable(getter) else {}
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise RuntimeError(f"draft task for {experiment!r} has invalid parameters")
    for key, expected in expected_parameters.items():
        if not _parameter_matches(parameters.get(key), expected):
            raise RuntimeError(
                f"draft task for {experiment!r} {key} mismatch: "
                f"expected {expected!r}, got {parameters.get(key)!r}"
            )
    for key in ("Args/teacher_checkpoint", "Args/student_checkpoint"):
        if parameters.get(key) not in (None, ""):
            raise RuntimeError(
                f"draft task for {experiment!r} contains forbidden {key}"
            )
    if (
        not EXPERIMENT_BY_NAME[experiment].requires_teacher
        and parameters.get("Args/teacher_task_id") not in (None, "")
    ):
        raise RuntimeError(
            f"teacher-free draft {experiment!r} contains a teacher task ID"
        )
    return task


def _step_release_guards(
    task_class: object,
    *,
    step_plan: Sequence[Mapping[str, object]],
    main_task_id: str,
    poll_seconds: float,
) -> dict[str, Callable[[object, object, dict[str, object]], bool]]:
    guards: dict[
        str,
        Callable[[object, object, dict[str, object]], bool],
    ] = {}
    for index, step in enumerate(step_plan):
        step_name = str(step["name"])
        experiment = str(step["experiment"])
        draft_task_id = str(step["base_task_id"])
        expected_parameters = dict(step["parameter_override"])

        def guard(
            _controller: object,
            _node: object,
            _parameters: dict[str, object],
            *,
            _index: int = index,
            _experiment: str = experiment,
            _draft_task_id: str = draft_task_id,
            _expected_parameters: Mapping[str, object] = expected_parameters,
        ) -> bool:
            if _index == 0:
                _wait_for_completed_main(
                    task_class,
                    main_task_id=main_task_id,
                    poll_seconds=poll_seconds,
                )
            _require_created_immutable_draft(
                task_class,
                experiment=_experiment,
                draft_task_id=_draft_task_id,
                expected_parameters=_expected_parameters,
            )
            return True

        guards[step_name] = guard
    return guards


def _validate_draft_tasks(
    task_class: object,
    draft_task_ids: Mapping[str, str],
    *,
    main_task_id: str,
) -> None:
    """Require genuine drafts and reject latent checkpoint handoffs."""

    previous_experiment: str | None = None
    for experiment in EXPERIMENT_ORDER:
        expected_predecessor = (
            main_task_id
            if previous_experiment is None
            else draft_task_ids[previous_experiment]
        )
        expected_parameters = _step_parameter_override(
            experiment,
            main_task_id=main_task_id,
            predecessor_task_id=expected_predecessor,
        )
        _require_created_immutable_draft(
            task_class,
            experiment=experiment,
            draft_task_id=draft_task_ids[experiment],
            expected_parameters=expected_parameters,
        )
        previous_experiment = experiment


def build_pipeline_controller(
    controller_class: object,
    *,
    step_plan: Sequence[Mapping[str, object]],
    project: str,
    name: str,
    version: str,
    worker_queue: str,
    step_guards: Mapping[
        str,
        Callable[[object, object, dict[str, object]], bool],
    ],
) -> object:
    controller = controller_class(
        name=name,
        project=project,
        version=version,
        add_pipeline_tags=True,
        target_project=True,
        abort_on_failure=True,
        output_uri=FILES_SERVER_URI,
    )
    controller.set_default_execution_queue(worker_queue)
    expected_step_names = {str(step["name"]) for step in step_plan}
    if set(step_guards) != expected_step_names:
        raise RuntimeError("every pipeline step requires exactly one release guard")
    for step in step_plan:
        step_name = str(step["name"])
        release_guard = step_guards[step_name]
        if not callable(release_guard):
            raise RuntimeError(f"release guard for {step_name!r} is not callable")
        added = controller.add_step(
            name=step_name,
            base_task_id=str(step["base_task_id"]),
            parents=list(step["parents"]),
            parameter_override=dict(step["parameter_override"]),
            execution_queue=str(step["execution_queue"]),
            clone_base_task=False,
            continue_on_fail=False,
            cache_executed_step=False,
            recursively_parse_parameters=True,
            output_uri=FILES_SERVER_URI,
            pre_execute_callback=release_guard,
        )
        if not added:
            raise RuntimeError(f"PipelineController rejected step {step['name']!r}")
    return controller


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    main_task_id = _task_id(args.main_task_id, "main task")
    draft_task_ids = _resolve_draft_task_ids(args.draft_task_id)
    step_plan = build_step_plan(
        draft_task_ids=draft_task_ids,
        main_task_id=main_task_id,
        worker_queue=args.worker_queue,
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "pipeline_type": "resilient_v2x_linear_training_suite",
                    "main_task_id": main_task_id,
                    "main_gate": "completed_with_exactly_one_clean_teacher_model",
                    "training_step_count": len(step_plan),
                    "condition_evaluation_steps": 0,
                    "steps": step_plan,
                },
                sort_keys=True,
            )
        )
        return 0

    from clearml import PipelineController, Task

    _validate_draft_tasks(
        Task,
        draft_task_ids,
        main_task_id=main_task_id,
    )
    guards = _step_release_guards(
        Task,
        step_plan=step_plan,
        main_task_id=main_task_id,
        poll_seconds=args.main_poll_seconds,
    )
    controller = build_pipeline_controller(
        PipelineController,
        step_plan=step_plan,
        project=args.project,
        name=args.name,
        version=args.version,
        worker_queue=args.worker_queue,
        step_guards=guards,
    )
    if not controller.start(queue=args.pipeline_queue, wait=args.wait):
        raise RuntimeError("failed to start the ClearML training suite controller")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_DRAFT_TASK_IDS",
    "DEFAULT_PIPELINE_QUEUE",
    "DEFAULT_WORKER_QUEUE",
    "MAIN_TASK_ID",
    "build_pipeline_controller",
    "build_step_plan",
    "main",
)
