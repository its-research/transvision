from __future__ import annotations

import builtins
import contextlib
import copy
import io
import json
import unittest
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from unittest import mock

from tools.resilient_v2x import clearml_p1_a100_parallel_v4_parameter_compat as compat


base = compat.base
v2 = compat.v2
v3 = compat.v3

CONTROLLER_ID = "c" * 32


class _ParameterTask:
    def __init__(self, parameters: Mapping[str, object]) -> None:
        self.parameters = dict(parameters)

    def get_parameters(self, **kwargs: object) -> dict[str, object]:
        self.last_get_kwargs = dict(kwargs)
        return dict(self.parameters)


def _logical_training() -> dict[str, object]:
    return base._training_parameters("resilient_v2x", 20250218)


def _logical_evaluation() -> dict[str, object]:
    return base._evaluation_parameters(
        "resilient_v2x",
        20250218,
        training_task_id="1" * 32,
        model_id="2" * 32,
        checkpoint_sha256="3" * 64,
    )


def _runtime_parameters(phase: str) -> dict[str, object]:
    logical = _logical_training() if phase == "training" else _logical_evaluation()
    return compat._completion_runtime_expected(logical, phase=phase)


def _source_journal() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for pair in base.build_plan()["pairs"]:
        for phase in ("training", "evaluation"):
            record = pair[phase]
            task_key = str(record["task_key"])
            binding = compat.LIVE_SOURCE_TASK_BINDINGS.get(task_key)
            rows.append(
                {
                    "task_key": task_key,
                    "task_id": binding["task_id"] if binding else None,
                    "state": binding["state"] if binding else "absent",
                    "server_status": (
                        binding["server_status"] if binding else "absent"
                    ),
                    "result": None,
                }
            )
    return {
        "revision": compat.LIVE_SOURCE_JOURNAL_REVISION,
        "seal_sha256": compat.LIVE_SOURCE_JOURNAL_SEAL_SHA256,
        "status": "running",
        "tasks": rows,
    }


def _amendment() -> dict[str, object]:
    return {"seal_sha256": compat.LIVE_V2_AMENDMENT_SEAL_SHA256}


def _v3_barrier() -> dict[str, object]:
    return {"seal_sha256": compat.LIVE_V3_BARRIER_SEAL_SHA256}


def _v4_barrier() -> dict[str, object]:
    return compat._build_v4_barrier(
        controller_id=CONTROLLER_ID,
        amendment=_amendment(),
        v3_barrier=_v3_barrier(),
        journal=_source_journal(),
    )


class ParallelV4ParameterCompatTests(unittest.TestCase):
    def test_frozen_v3_and_parameter_schemas_are_exact(self) -> None:
        self.assertEqual(
            compat.FROZEN_V3_SHA256,
            "baf914454f0b4152426c3dd7d3b36272db122e6734bde955ec7cd7d7c1755eea",
        )
        self.assertEqual(v3._v3_sha256(), compat.FROZEN_V3_SHA256)
        self.assertEqual(len(compat.TRAINING_EXPECTED_KEYS), 19)
        self.assertEqual(len(compat.TRAINING_RUNTIME_EXTRA_PARAMETERS), 9)
        self.assertEqual(len(compat.EVALUATION_EXPECTED_KEYS), 18)
        self.assertEqual(len(compat.EVALUATION_RUNTIME_EXTRA_PARAMETERS), 10)
        self.assertEqual(len(compat.FULL_RUNTIME_PARAMETER_KEYS), 28)
        self.assertEqual(
            set(compat.TRAINING_EXPECTED_KEYS)
            | set(compat.TRAINING_RUNTIME_EXTRA_PARAMETERS),
            set(compat.FULL_RUNTIME_PARAMETER_KEYS),
        )
        self.assertEqual(
            set(compat.EVALUATION_EXPECTED_KEYS)
            | set(compat.EVALUATION_RUNTIME_EXTRA_PARAMETERS),
            set(compat.FULL_RUNTIME_PARAMETER_KEYS),
        )
        self.assertFalse(
            set(compat.TRAINING_EXPECTED_KEYS)
            & set(compat.TRAINING_RUNTIME_EXTRA_PARAMETERS)
        )
        self.assertFalse(
            set(compat.EVALUATION_EXPECTED_KEYS)
            & set(compat.EVALUATION_RUNTIME_EXTRA_PARAMETERS)
        )

    def test_training_completion_accepts_only_the_exact_live_runtime_expansion(
        self,
    ) -> None:
        logical = _logical_training()
        observed = _runtime_parameters("training")
        task = _ParameterTask(observed)

        with compat._completion_parameter_guard("training"):
            base._require_parameters(task, logical, "completed training")

        self.assertEqual(
            task.last_get_kwargs,
            {"backwards_compatibility": False},
        )
        self.assertEqual(
            {key: observed[key] for key in compat.TRAINING_RUNTIME_EXTRA_PARAMETERS},
            compat.TRAINING_RUNTIME_EXTRA_PARAMETERS,
        )

    def test_evaluation_completion_accepts_clearml_string_false_normalization(
        self,
    ) -> None:
        logical = _logical_evaluation()
        observed = _runtime_parameters("evaluation")
        observed["Args/allow_failed_teacher_task"] = "False"
        task = _ParameterTask(observed)

        with compat._completion_parameter_guard("evaluation"):
            base._require_parameters(task, logical, "completed evaluation")

    def test_completion_guard_rejects_missing_unknown_or_nonempty_runtime_extra(
        self,
    ) -> None:
        cases: list[tuple[str, str, dict[str, object], str]] = []
        for phase in ("training", "evaluation"):
            exact = _runtime_parameters(phase)
            missing = dict(exact)
            missing.pop(next(iter(compat._runtime_extra_parameters(phase))))
            cases.append((phase, "missing", missing, "parameter keys drifted"))

            unknown = dict(exact)
            unknown["Args/unknown_runtime_option"] = ""
            cases.append((phase, "unknown", unknown, "parameter keys drifted"))

            wrong = dict(exact)
            string_extra = next(
                key
                for key, value in compat._runtime_extra_parameters(phase).items()
                if value == ""
            )
            wrong[string_extra] = "unexpected"
            cases.append(
                (phase, "nonempty", wrong, f"parameter {string_extra} drifted")
            )

        for phase, label, observed, message in cases:
            with self.subTest(phase=phase, label=label):
                logical = (
                    _logical_training()
                    if phase == "training"
                    else _logical_evaluation()
                )
                with self.assertRaisesRegex(base.P1ExecutorError, message):
                    with compat._completion_parameter_guard(phase):
                        base._require_parameters(
                            _ParameterTask(observed), logical, f"{phase} result"
                        )

    def test_completion_guard_rejects_incomplete_logical_expected_schema(self) -> None:
        for phase, logical in (
            ("training", _logical_training()),
            ("evaluation", _logical_evaluation()),
        ):
            with self.subTest(phase=phase):
                logical.pop(next(iter(logical)))
                with self.assertRaisesRegex(
                    compat.P1ParallelV4Error,
                    "logical parameter schema drifted",
                ):
                    compat._completion_runtime_expected(logical, phase=phase)

    def test_completion_guard_rejects_shared_expected_value_drift(self) -> None:
        cases = {
            "training": ("Args/training_seed", 20250219),
            "evaluation": ("Args/controlled_baseline", "bevfusion"),
        }
        for phase, (key, drifted_value) in cases.items():
            with self.subTest(phase=phase, key=key):
                observed = _runtime_parameters(phase)
                observed[key] = drifted_value
                logical = (
                    _logical_training()
                    if phase == "training"
                    else _logical_evaluation()
                )
                with self.assertRaisesRegex(
                    base.P1ExecutorError, f"parameter {key} drifted"
                ):
                    with compat._completion_parameter_guard(phase):
                        base._require_parameters(
                            _ParameterTask(observed), logical, f"{phase} result"
                        )

    def test_runtime_patches_only_completion_validators_and_restores_everything(
        self,
    ) -> None:
        original_require = base._require_parameters

        def training_validator(task: object, **kwargs: object) -> str:
            del kwargs
            base._require_parameters(task, _logical_training(), "training result")
            return "training-ok"

        def evaluation_validator(task: object, **kwargs: object) -> str:
            del kwargs
            base._require_parameters(task, _logical_evaluation(), "evaluation result")
            return "evaluation-ok"

        with (
            mock.patch.object(
                base, "_validate_training_result", side_effect=training_validator
            ) as original_training,
            mock.patch.object(
                base, "_validate_evaluation_result", side_effect=evaluation_validator
            ) as original_evaluation,
        ):
            with compat._completion_validation_runtime():
                self.assertIs(base._require_parameters, original_require)
                self.assertEqual(
                    base._validate_training_result(
                        _ParameterTask(_runtime_parameters("training"))
                    ),
                    "training-ok",
                )
                self.assertIs(base._require_parameters, original_require)
                self.assertEqual(
                    base._validate_evaluation_result(
                        _ParameterTask(_runtime_parameters("evaluation"))
                    ),
                    "evaluation-ok",
                )
                self.assertIs(base._require_parameters, original_require)

                with self.assertRaisesRegex(
                    base.P1ExecutorError, "parameter keys drifted"
                ):
                    base._require_parameters(
                        _ParameterTask(_runtime_parameters("training")),
                        _logical_training(),
                        "created scoped task",
                    )

            self.assertIs(base._require_parameters, original_require)
            self.assertIs(base._validate_training_result, original_training)
            self.assertIs(base._validate_evaluation_result, original_evaluation)

    def test_completion_validator_exception_restores_require_parameters(self) -> None:
        original_require = base._require_parameters

        def failing_validator(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("artifact validation failed first")

        with mock.patch.object(
            base, "_validate_training_result", side_effect=failing_validator
        ):
            with compat._completion_validation_runtime():
                with self.assertRaisesRegex(RuntimeError, "artifact validation failed"):
                    base._validate_training_result(object())
                self.assertIs(base._require_parameters, original_require)

    def test_v4_barrier_binds_live_source_chain_and_complete_schema(self) -> None:
        barrier = _v4_barrier()
        validated = compat._validate_v4_barrier(
            barrier,
            controller_id=CONTROLLER_ID,
            amendment=_amendment(),
            v3_barrier=_v3_barrier(),
        )

        self.assertEqual(validated, barrier)
        self.assertEqual(barrier["v4_supervisor_sha256"], compat._v4_sha256())
        self.assertEqual(barrier["v3_supervisor_sha256"], compat.FROZEN_V3_SHA256)
        self.assertEqual(barrier["execution_key"], v2.SEALED_EXECUTION_KEY)
        self.assertEqual(
            barrier["v2_amendment_seal_sha256"],
            compat.LIVE_V2_AMENDMENT_SEAL_SHA256,
        )
        self.assertEqual(
            barrier["v3_barrier_seal_sha256"],
            compat.LIVE_V3_BARRIER_SEAL_SHA256,
        )
        self.assertEqual(
            barrier["source_journal"], compat._expected_source_journal_binding()
        )
        contract = barrier["completion_parameter_contract"]
        self.assertEqual(
            contract["training"]["runtime_extra_parameters"],
            compat.TRAINING_RUNTIME_EXTRA_PARAMETERS,
        )
        self.assertEqual(
            contract["evaluation"]["runtime_extra_parameters"],
            compat.EVALUATION_RUNTIME_EXTRA_PARAMETERS,
        )

    def test_v4_barrier_rejects_resealed_binding_or_contract_drift(self) -> None:
        barrier = _v4_barrier()
        cases: list[tuple[str, object]] = [
            ("controller_task_id", "d" * 32),
            ("execution_key", "e" * 64),
            ("v3_supervisor_sha256", "f" * 64),
            ("v3_barrier_seal_sha256", "0" * 64),
        ]
        for field, replacement in cases:
            with self.subTest(field=field):
                drifted = copy.deepcopy(barrier)
                drifted[field] = replacement
                drifted["seal_sha256"] = base._seal(drifted)
                with self.assertRaisesRegex(
                    compat.P1ParallelV4Error, f"barrier {field} drifted"
                ):
                    compat._validate_v4_barrier(
                        drifted,
                        controller_id=CONTROLLER_ID,
                        amendment=_amendment(),
                        v3_barrier=_v3_barrier(),
                    )

        drifted = copy.deepcopy(barrier)
        drifted["completion_parameter_contract"]["unknown_keys_forbidden"] = False
        drifted["seal_sha256"] = base._seal(drifted)
        with self.assertRaisesRegex(
            compat.P1ParallelV4Error,
            "completion_parameter_contract drifted",
        ):
            compat._validate_v4_barrier(
                drifted,
                controller_id=CONTROLLER_ID,
                amendment=_amendment(),
                v3_barrier=_v3_barrier(),
            )

    def test_v4_barrier_install_is_authoritatively_confirmed_and_idempotent(
        self,
    ) -> None:
        barrier = _v4_barrier()
        with (
            mock.patch.object(
                compat,
                "_read_installed_v4_barrier",
                side_effect=[None, barrier],
            ) as read,
            mock.patch.object(compat, "_build_v4_barrier", return_value=barrier),
            mock.patch.object(v3, "_read_pending_wal", return_value=None),
            mock.patch.object(v3, "_upload_mapping_confirmed") as upload,
        ):
            installed = compat._install_v4_barrier(
                SimpleNamespace(),
                controller_id=CONTROLLER_ID,
                amendment=_amendment(),
                v3_barrier=_v3_barrier(),
                journal=_source_journal(),
            )

        self.assertEqual(installed, barrier)
        self.assertEqual(read.call_count, 2)
        upload.assert_called_once_with(
            mock.ANY,
            controller_id=CONTROLLER_ID,
            artifact_name=compat.V4_BARRIER_ARTIFACT,
            value=barrier,
        )

        with (
            mock.patch.object(
                compat, "_read_installed_v4_barrier", return_value=barrier
            ),
            mock.patch.object(
                v3,
                "_upload_mapping_confirmed",
                side_effect=AssertionError("existing barrier was re-uploaded"),
            ) as upload_again,
        ):
            self.assertEqual(
                compat._install_v4_barrier(
                    SimpleNamespace(),
                    controller_id=CONTROLLER_ID,
                    amendment=_amendment(),
                    v3_barrier=_v3_barrier(),
                    journal=_source_journal(),
                ),
                barrier,
            )
        upload_again.assert_not_called()

    def test_initial_v4_barrier_install_rejects_any_pending_wal(self) -> None:
        with (
            mock.patch.object(compat, "_read_installed_v4_barrier", return_value=None),
            mock.patch.object(
                v3, "_read_pending_wal", return_value={"status": "pending"}
            ),
            mock.patch.object(compat, "_build_v4_barrier") as build,
            mock.patch.object(v3, "_upload_mapping_confirmed") as upload,
        ):
            with self.assertRaisesRegex(
                compat.P1ParallelV4Error, "requires an absent pending WAL"
            ):
                compat._install_v4_barrier(
                    SimpleNamespace(),
                    controller_id=CONTROLLER_ID,
                    amendment=_amendment(),
                    v3_barrier=_v3_barrier(),
                    journal=_source_journal(),
                )
        build.assert_not_called()
        upload.assert_not_called()

    def test_old_v3_rejects_v4_barrier_before_any_mutation(self) -> None:
        controller = SimpleNamespace(
            id=CONTROLLER_ID,
            artifacts={compat.V4_BARRIER_ARTIFACT: object()},
        )
        base_receipt = {"existing_execution": {"controller_id": CONTROLLER_ID}}

        class TaskClass:
            enqueue = mock.Mock(side_effect=AssertionError("old v3 attempted enqueue"))

        with (
            mock.patch.object(v3, "_fresh_task", return_value=controller),
            mock.patch.object(base, "_validate_controller"),
            mock.patch.object(
                base,
                "_upload_artifact",
                side_effect=AssertionError("old v3 attempted upload"),
            ) as upload,
            mock.patch.object(
                base,
                "_create_task",
                side_effect=AssertionError("old v3 attempted clone"),
            ) as clone,
        ):
            with self.assertRaisesRegex(v3.P1ParallelV3Error, "unknown entries"):
                v3._load_existing_execution(TaskClass, base_receipt)

        upload.assert_not_called()
        clone.assert_not_called()
        TaskClass.enqueue.assert_not_called()

    def test_v4_load_existing_allows_only_v4_and_revalidates_both_barriers(
        self,
    ) -> None:
        controller = SimpleNamespace(
            id=CONTROLLER_ID,
            artifacts={
                base.PLAN_ARTIFACT: object(),
                base.PINSET_ARTIFACT: object(),
                base.JOURNAL_ARTIFACT: object(),
                v2.AMENDMENT_ARTIFACT: object(),
                v3.V3_BARRIER_ARTIFACT: object(),
                compat.V4_BARRIER_ARTIFACT: object(),
            },
        )
        journal = {"revision": 10, "seal_sha256": "a" * 64}
        base_receipt = {"existing_execution": {"controller_id": CONTROLLER_ID}}

        with (
            mock.patch.object(v3, "_fresh_task", return_value=controller),
            mock.patch.object(base, "_validate_controller"),
            mock.patch.object(v3, "_validate_authoritative_immutable_bindings"),
            mock.patch.object(
                v3,
                "_authoritative_remote_journal",
                return_value=(controller, journal),
            ),
            mock.patch.object(compat, "_validate_bound_barriers") as validate,
        ):
            observed_controller, observed_journal = compat._load_existing_execution(
                SimpleNamespace(), base_receipt
            )

        self.assertIs(observed_controller, controller)
        self.assertEqual(observed_journal, journal)
        validate.assert_called_once_with(
            mock.ANY,
            controller_id=CONTROLLER_ID,
            journal=journal,
        )

        controller.artifacts["unexpected-controller-artifact"] = object()
        with (
            mock.patch.object(v3, "_fresh_task", return_value=controller),
            mock.patch.object(base, "_validate_controller"),
        ):
            with self.assertRaisesRegex(compat.P1ParallelV4Error, "unknown entries"):
                compat._load_existing_execution(SimpleNamespace(), base_receipt)

    def test_every_authoritative_journal_load_revalidates_v3_and_v4_barriers(
        self,
    ) -> None:
        journal = {"revision": 11, "seal_sha256": "b" * 64}
        original = mock.Mock(return_value=journal)
        controller = SimpleNamespace(id=CONTROLLER_ID)
        task_class = SimpleNamespace(name="task-class")
        plan = base.build_plan()

        with mock.patch.object(compat, "_validate_bound_barriers") as validate:
            observed = compat._validated_authoritative_load_journal(
                original,
                controller,
                plan=plan,
                execution_key=v2.SEALED_EXECUTION_KEY,
                controller_id=CONTROLLER_ID,
                task_class=task_class,
            )

        self.assertEqual(observed, journal)
        validate.assert_called_once_with(
            task_class,
            controller_id=CONTROLLER_ID,
            journal=journal,
        )

    def test_v4_runtime_patches_v3_hooks_and_restores_them(self) -> None:
        original_existing = v3._load_existing_execution
        original_authoritative = v3._authoritative_load_journal
        original_authoritative_remote = v3._authoritative_remote_journal
        original_training = base._validate_training_result
        original_evaluation = base._validate_evaluation_result

        with compat._v4_runtime():
            self.assertIs(v3._load_existing_execution, compat._load_existing_execution)
            self.assertIsNot(v3._authoritative_load_journal, original_authoritative)
            self.assertIsNot(
                v3._authoritative_remote_journal, original_authoritative_remote
            )
            self.assertIsNot(base._validate_training_result, original_training)
            self.assertIsNot(base._validate_evaluation_result, original_evaluation)

        self.assertIs(v3._load_existing_execution, original_existing)
        self.assertIs(v3._authoritative_load_journal, original_authoritative)
        self.assertIs(v3._authoritative_remote_journal, original_authoritative_remote)
        self.assertIs(base._validate_training_result, original_training)
        self.assertIs(base._validate_evaluation_result, original_evaluation)

    def test_v2_remote_journal_commit_hook_revalidates_both_barriers(self) -> None:
        task_class = SimpleNamespace(name="task-class")
        controller = SimpleNamespace(id=CONTROLLER_ID)
        journal = {"revision": 12, "seal_sha256": "c" * 64}
        original_v2_remote = v2._read_remote_journal

        with (
            mock.patch.object(
                v3,
                "_authoritative_remote_journal",
                return_value=(controller, journal),
            ) as original_v3_remote,
            mock.patch.object(compat, "_validate_bound_barriers") as validate,
        ):
            with compat._v4_runtime():
                patched_v3_remote = v3._authoritative_remote_journal
                with v3._authoritative_v2_runtime(task_class=task_class):
                    self.assertIs(v2._read_remote_journal, patched_v3_remote)
                    observed_controller, observed_journal = v2._read_remote_journal(
                        task_class, CONTROLLER_ID
                    )
                self.assertIs(v2._read_remote_journal, original_v2_remote)
            self.assertIs(v3._authoritative_remote_journal, original_v3_remote)

        self.assertIs(observed_controller, controller)
        self.assertEqual(observed_journal, journal)
        original_v3_remote.assert_called_once_with(task_class, CONTROLLER_ID)
        validate.assert_called_once_with(
            task_class,
            controller_id=CONTROLLER_ID,
            journal=journal,
        )
        self.assertIs(v2._read_remote_journal, original_v2_remote)

    def test_preflight_is_read_only_and_never_installs_v4_barrier(self) -> None:
        task_class = SimpleNamespace(name="task-class")
        v3_receipt = {
            "seal_sha256": "1" * 64,
            "queue": {"status": "read-only"},
            "global_mutex": {"status": "available"},
        }
        base_receipt = {"existing_execution": {"controller_id": CONTROLLER_ID}}
        journal = {
            "revision": 10,
            "seal_sha256": compat.LIVE_SOURCE_JOURNAL_SEAL_SHA256,
        }
        barrier = _v4_barrier()
        mutation_error = AssertionError("preflight attempted mutation")

        with (
            mock.patch.object(v3, "preflight", return_value=v3_receipt),
            mock.patch.object(
                v3,
                "_read_base_preflight",
                return_value=(task_class, base_receipt),
            ),
            mock.patch.object(
                compat,
                "_read_current_bound_execution",
                return_value=(
                    CONTROLLER_ID,
                    journal,
                    _amendment(),
                    _v3_barrier(),
                ),
            ),
            mock.patch.object(
                compat, "_read_installed_v4_barrier", return_value=barrier
            ),
            mock.patch.object(v3, "_read_pending_wal", return_value=None),
            mock.patch.object(
                compat, "_install_v4_barrier", side_effect=mutation_error
            ) as install,
            mock.patch.object(
                v3, "_upload_mapping_confirmed", side_effect=mutation_error
            ) as upload,
        ):
            receipt = compat.preflight(task_class=task_class)

        self.assertTrue(receipt["readonly"])
        self.assertEqual(receipt["remote_mutation_count"], 0)
        self.assertEqual(receipt["v4_barrier_status"], "installed")
        self.assertEqual(
            receipt["source_binding"], compat._expected_source_journal_binding()
        )
        install.assert_not_called()
        upload.assert_not_called()

    def test_preflight_without_barrier_requires_exact_rev10_source(self) -> None:
        task_class = SimpleNamespace(name="task-class")
        v3_receipt = {
            "seal_sha256": "1" * 64,
            "queue": {},
            "global_mutex": {},
        }
        base_receipt = {"existing_execution": {"controller_id": CONTROLLER_ID}}
        drifted = _source_journal()
        drifted["revision"] = 11
        drifted["seal_sha256"] = "0" * 64

        with (
            mock.patch.object(v3, "preflight", return_value=v3_receipt),
            mock.patch.object(
                v3,
                "_read_base_preflight",
                return_value=(task_class, base_receipt),
            ),
            mock.patch.object(
                compat,
                "_read_current_bound_execution",
                return_value=(
                    CONTROLLER_ID,
                    drifted,
                    _amendment(),
                    _v3_barrier(),
                ),
            ),
            mock.patch.object(compat, "_read_installed_v4_barrier", return_value=None),
            mock.patch.object(v3, "_read_pending_wal", return_value=None),
        ):
            with self.assertRaisesRegex(
                compat.P1ParallelV4Error, "live source journal drifted"
            ):
                compat.preflight(task_class=task_class)

    def test_execute_installs_v4_fence_before_running_patched_v3(self) -> None:
        events: list[str] = []
        migration = {
            "controller_task_id": CONTROLLER_ID,
            "source_journal_revision": 10,
            "source_journal_seal_sha256": compat.LIVE_SOURCE_JOURNAL_SEAL_SHA256,
            "v3_barrier_seal_sha256": compat.LIVE_V3_BARRIER_SEAL_SHA256,
            "v4_barrier_seal_sha256": "4" * 64,
        }
        v3_receipt = {"status": "completed", "seal_sha256": "5" * 64}

        def install(**kwargs: object) -> dict[str, object]:
            del kwargs
            events.append("v4-barrier-installed")
            return migration

        @contextlib.contextmanager
        def runtime() -> Iterator[None]:
            self.assertEqual(events, ["v4-barrier-installed"])
            events.append("v4-runtime-entered")
            yield

        def execute(**kwargs: object) -> dict[str, object]:
            del kwargs
            self.assertEqual(events[-1], "v4-runtime-entered")
            events.append("v3-executed")
            return v3_receipt

        with (
            mock.patch.object(v3, "_validate_control_host"),
            mock.patch.object(
                compat,
                "_install_v4_barrier_under_mutex",
                side_effect=install,
            ),
            mock.patch.object(compat, "_v4_runtime", side_effect=runtime),
            mock.patch.object(v3, "execute_plan", side_effect=execute),
        ):
            receipt = compat.execute_plan(
                authorization_token=compat.V4_EXECUTION_TOKEN,
                poll_seconds=5.0,
                timeout_hours=6.0,
                task_class=SimpleNamespace(),
                api_client=SimpleNamespace(),
            )

        self.assertEqual(
            events,
            ["v4-barrier-installed", "v4-runtime-entered", "v3-executed"],
        )
        self.assertEqual(receipt["v4_barrier_seal_sha256"], "4" * 64)
        self.assertEqual(receipt["v3_execution_receipt"], v3_receipt)
        self.assertEqual(receipt["seal_sha256"], base._seal(receipt))

    def test_execute_requires_exact_v4_token_before_remote_checks(self) -> None:
        self.assertNotEqual(compat.V4_EXECUTION_TOKEN, v3.V3_EXECUTION_TOKEN)
        with mock.patch.object(v3, "_validate_control_host") as host:
            with self.assertRaisesRegex(
                compat.P1ParallelV4Error, "execution token mismatch"
            ):
                compat.execute_plan(
                    authorization_token=v3.V3_EXECUTION_TOKEN,
                    poll_seconds=30.0,
                    timeout_hours=72.0,
                    task_class=SimpleNamespace(),
                    api_client=SimpleNamespace(),
                )
        host.assert_not_called()

    def test_default_mode_is_local_dry_run_without_clearml_import(self) -> None:
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "clearml" or name.startswith("clearml."):
                raise AssertionError("dry-run imported ClearML")
            return real_import(name, *args, **kwargs)

        output = io.StringIO()
        with (
            mock.patch("builtins.__import__", side_effect=guarded_import),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(compat.main([]), 0)

        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt["document_type"], compat.V4_DRY_RUN_DOCUMENT_TYPE)
        self.assertEqual(receipt["default_mode"], "dry_run")
        self.assertFalse(receipt["remote_mutation_authorized"])
        self.assertEqual(receipt["v3_supervisor_sha256"], compat.FROZEN_V3_SHA256)

    def test_preflight_cli_never_dispatches_execute(self) -> None:
        receipt = {
            "schema_version": 1,
            "document_type": compat.V4_PREFLIGHT_DOCUMENT_TYPE,
            "readonly": True,
            "remote_mutation_count": 0,
            "seal_sha256": "a" * 64,
        }
        output = io.StringIO()
        with (
            mock.patch.object(compat, "preflight", return_value=receipt) as read,
            mock.patch.object(
                compat,
                "execute_plan",
                side_effect=AssertionError("preflight dispatched execute"),
            ) as execute,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(compat.main(["--preflight"]), 0)

        self.assertEqual(json.loads(output.getvalue()), receipt)
        read.assert_called_once()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
