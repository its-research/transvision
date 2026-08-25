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

from tools.resilient_v2x import (
    clearml_p1_a100_parallel_v5_evaluation_contract_compat as compat,
)


base = compat.base
v2 = compat.v2
v3 = compat.v3
v4 = compat.v4

CONTROLLER_ID = "c" * 32
TRAINING_TASK_ID = "1" * 32
MODEL_ID = "2" * 32
CHECKPOINT_SHA256 = "3" * 64
SUBJECT = "resilient_v2x"
SEED = 20250218


class _ParameterTask:
    def __init__(self, parameters: Mapping[str, object]) -> None:
        self.parameters = dict(parameters)

    def get_parameters(self, **kwargs: object) -> dict[str, object]:
        self.last_get_kwargs = dict(kwargs)
        return dict(self.parameters)


def _legacy_contract() -> dict[str, object]:
    return {
        key: {"task_id": TRAINING_TASK_ID} if key == "checkpoint" else f"value-{key}"
        for key in compat.LEGACY_EVALUATION_RUN_CONTRACT_KEYS
    }


def _artifact() -> SimpleNamespace:
    return SimpleNamespace(hash=compat.LIVE_EVALUATION_RUN_CONTRACT_SHA256)


def _runtime_parameters() -> dict[str, object]:
    logical = base._evaluation_parameters(
        SUBJECT,
        SEED,
        training_task_id=TRAINING_TASK_ID,
        model_id=MODEL_ID,
        checkpoint_sha256=CHECKPOINT_SHA256,
    )
    return v4._completion_runtime_expected(logical, phase="evaluation")


def _source_journal() -> dict[str, object]:
    rows = []
    for pair in base.build_plan()["pairs"]:
        for phase in ("training", "evaluation"):
            task_key = str(pair[phase]["task_key"])
            if task_key == compat.LIVE_COMPLETED_EVALUATION_TASK_KEY:
                row = {
                    "task_key": task_key,
                    "task_id": compat.LIVE_COMPLETED_EVALUATION_TASK_ID,
                    "state": "active",
                    "server_status": "in_progress",
                    "result": None,
                }
            else:
                row = {
                    "task_key": task_key,
                    "task_id": None,
                    "state": "absent",
                    "server_status": "absent",
                    "result": None,
                }
            rows.append(row)
    return {
        "revision": compat.LIVE_SOURCE_JOURNAL_REVISION,
        "seal_sha256": compat.LIVE_SOURCE_JOURNAL_SEAL_SHA256,
        "status": "running",
        "tasks": rows,
    }


def _amendment() -> dict[str, object]:
    return {"seal_sha256": "a" * 64}


def _v3_barrier() -> dict[str, object]:
    return {"seal_sha256": "b" * 64}


def _v4_barrier() -> dict[str, object]:
    return {"seal_sha256": "d" * 64}


def _v5_barrier() -> dict[str, object]:
    return compat._build_v5_barrier(
        controller_id=CONTROLLER_ID,
        journal=_source_journal(),
        amendment=_amendment(),
        v3_barrier=_v3_barrier(),
        v4_barrier=_v4_barrier(),
        live_evaluation_contract=(compat._expected_live_evaluation_contract_binding()),
    )


class ParallelV5EvaluationContractCompatTests(unittest.TestCase):
    def test_frozen_v4_and_live_failure_bindings_are_exact(self) -> None:
        self.assertEqual(
            compat.FROZEN_V4_SHA256,
            "e39dcde5187f99350f55e0210560890d505ca230da498fd1cb4d91c1d28a11c6",
        )
        self.assertEqual(v4._v4_sha256(), compat.FROZEN_V4_SHA256)
        self.assertEqual(compat.LIVE_SOURCE_JOURNAL_REVISION, 36)
        self.assertEqual(
            compat.LIVE_SOURCE_JOURNAL_SEAL_SHA256,
            "b09222eec6478fd87088a0c5f6af1566f709443c73150c07b72cfe41cee96d5c",
        )
        self.assertEqual(
            compat.LIVE_COMPLETED_EVALUATION_TASK_ID,
            "2fddc432826440389249f24fde8d3aa9",
        )
        self.assertEqual(len(compat.LEGACY_EVALUATION_RUN_CONTRACT_KEYS), 23)
        self.assertEqual(
            compat.LEGACY_MISSING_RUN_CONTRACT_FIELDS,
            {"training_seed", "training_overlay_protocol_seed"},
        )

    def test_source_journal_requires_exact_revision_and_task_binding(self) -> None:
        binding = compat._source_journal_binding(_source_journal())
        self.assertEqual(binding["revision"], 36)
        self.assertEqual(
            binding["evaluation_task_id"],
            compat.LIVE_COMPLETED_EVALUATION_TASK_ID,
        )

        cases = []
        drifted = _source_journal()
        drifted["revision"] = 37
        cases.append((drifted, "identity drifted"))
        drifted = _source_journal()
        drifted["tasks"] = "not-a-list"
        cases.append((drifted, "task list is invalid"))
        drifted = _source_journal()
        target = next(
            row
            for row in drifted["tasks"]
            if row["task_key"] == compat.LIVE_COMPLETED_EVALUATION_TASK_KEY
        )
        target["task_id"] = "f" * 32
        cases.append((drifted, "task_id drifted"))
        for journal, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(compat.P1ParallelV5Error, message):
                    compat._source_journal_binding(journal)

    def test_completed_evaluation_injects_only_two_fields_in_memory(self) -> None:
        observed = _legacy_contract()
        artifact = _artifact()
        task = _ParameterTask(_runtime_parameters())

        with mock.patch.object(
            base, "_artifact_mapping", return_value=observed
        ) as original:
            with compat._legacy_evaluation_run_contract_runtime(
                task,
                subject=SUBJECT,
                seed=SEED,
                training_task_id=TRAINING_TASK_ID,
            ):
                amended = base._artifact_mapping(
                    artifact, f"{SUBJECT} seed {SEED} evaluation run contract"
                )

        self.assertEqual(set(observed), compat.LEGACY_EVALUATION_RUN_CONTRACT_KEYS)
        self.assertFalse(set(observed) & compat.LEGACY_MISSING_RUN_CONTRACT_FIELDS)
        self.assertEqual(amended["training_seed"], SEED)
        self.assertEqual(
            amended["training_overlay_protocol_seed"],
            base.TRAINING_OVERLAY_PROTOCOL_SEED,
        )
        self.assertIsNot(amended, observed)
        original.assert_called_once_with(
            artifact, f"{SUBJECT} seed {SEED} evaluation run contract"
        )

    def test_non_target_artifact_mapping_is_unchanged(self) -> None:
        observed = {"unrelated": True}
        task = _ParameterTask(_runtime_parameters())
        with mock.patch.object(base, "_artifact_mapping", return_value=observed):
            with compat._legacy_evaluation_run_contract_runtime(
                task,
                subject=SUBJECT,
                seed=SEED,
                training_task_id=TRAINING_TASK_ID,
            ):
                self.assertIs(
                    base._artifact_mapping(object(), "unrelated artifact"), observed
                )

    def test_compatibility_rejects_hash_key_and_partial_materialization_drift(
        self,
    ) -> None:
        cases = []
        cases.append(
            (
                _legacy_contract(),
                SimpleNamespace(hash="0" * 64),
                "artifact SHA-256 drifted",
            )
        )
        unknown = _legacy_contract()
        unknown["unknown"] = True
        cases.append((unknown, _artifact(), "key inventory drifted"))
        missing = _legacy_contract()
        missing.pop("protocol_id")
        cases.append((missing, _artifact(), "key inventory drifted"))
        partial = _legacy_contract()
        partial["training_seed"] = SEED
        cases.append((partial, _artifact(), "partially materialized"))

        for mapping, artifact, message in cases:
            with self.subTest(message=message):
                task = _ParameterTask(_runtime_parameters())
                with mock.patch.object(base, "_artifact_mapping", return_value=mapping):
                    with compat._legacy_evaluation_run_contract_runtime(
                        task,
                        subject=SUBJECT,
                        seed=SEED,
                        training_task_id=TRAINING_TASK_ID,
                    ):
                        with self.assertRaisesRegex(compat.P1ParallelV5Error, message):
                            base._artifact_mapping(
                                artifact,
                                f"{SUBJECT} seed {SEED} evaluation run contract",
                            )

    def test_compatibility_rejects_authoritative_parameter_drift(self) -> None:
        for key, replacement in (
            ("Args/training_seed", SEED + 1),
            ("Args/predecessor_task_id", "4" * 32),
            ("Args/controlled_baseline_task_id", "5" * 32),
            ("Args/controlled_baseline", "bevfusion"),
            ("Args/stage", "all"),
        ):
            with self.subTest(key=key):
                parameters = _runtime_parameters()
                parameters[key] = replacement
                task = _ParameterTask(parameters)
                with mock.patch.object(
                    base, "_artifact_mapping", return_value=_legacy_contract()
                ):
                    with compat._legacy_evaluation_run_contract_runtime(
                        task,
                        subject=SUBJECT,
                        seed=SEED,
                        training_task_id=TRAINING_TASK_ID,
                    ):
                        with self.assertRaisesRegex(
                            compat.P1ParallelV5Error,
                            f"parameter {key} drifted",
                        ):
                            base._artifact_mapping(
                                _artifact(),
                                f"{SUBJECT} seed {SEED} evaluation run contract",
                            )

    def test_v4_parameter_guard_and_v5_contract_patch_compose_and_restore(self) -> None:
        task = _ParameterTask(_runtime_parameters())
        logical = base._evaluation_parameters(
            SUBJECT,
            SEED,
            training_task_id=TRAINING_TASK_ID,
            model_id=MODEL_ID,
            checkpoint_sha256=CHECKPOINT_SHA256,
        )
        artifact = _artifact()
        original_mapping = base._artifact_mapping
        original_validator = base._validate_evaluation_result

        def validator(current_task: object, **kwargs: object) -> dict[str, object]:
            del kwargs
            base._require_parameters(current_task, logical, "evaluation")
            run_contract = base._artifact_mapping(
                artifact, f"{SUBJECT} seed {SEED} evaluation run contract"
            )
            return {
                "training_seed": run_contract["training_seed"],
                "training_overlay_protocol_seed": run_contract[
                    "training_overlay_protocol_seed"
                ],
            }

        with (
            mock.patch.object(
                base, "_artifact_mapping", return_value=_legacy_contract()
            ),
            mock.patch.object(
                base, "_validate_evaluation_result", side_effect=validator
            ),
        ):
            with compat._v5_runtime():
                with v4._completion_validation_runtime():
                    result = base._validate_evaluation_result(
                        task,
                        subject=SUBJECT,
                        seed=SEED,
                        controller_id=CONTROLLER_ID,
                        training_task_id=TRAINING_TASK_ID,
                        model={
                            "model_id": MODEL_ID,
                            "checkpoint_sha256": CHECKPOINT_SHA256,
                        },
                        source_d="source",
                    )

        self.assertEqual(result["training_seed"], SEED)
        self.assertEqual(
            result["training_overlay_protocol_seed"],
            base.TRAINING_OVERLAY_PROTOCOL_SEED,
        )
        self.assertIs(base._artifact_mapping, original_mapping)
        self.assertIs(base._validate_evaluation_result, original_validator)

    def test_contract_patch_restores_mapper_after_validator_exception(self) -> None:
        original_mapping = base._artifact_mapping

        def failing(*args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            base._artifact_mapping(
                _artifact(), f"{SUBJECT} seed {SEED} evaluation run contract"
            )
            raise RuntimeError("later evidence validation failed")

        with mock.patch.object(
            base, "_artifact_mapping", return_value=_legacy_contract()
        ):
            patched_original = base._artifact_mapping
            with self.assertRaisesRegex(RuntimeError, "later evidence"):
                compat._validated_evaluation_result(
                    failing,
                    _ParameterTask(_runtime_parameters()),
                    subject=SUBJECT,
                    seed=SEED,
                    controller_id=CONTROLLER_ID,
                    training_task_id=TRAINING_TASK_ID,
                    model={},
                    source_d="source",
                )
            self.assertIs(base._artifact_mapping, patched_original)
        self.assertIs(base._artifact_mapping, original_mapping)

    def test_v5_barrier_binds_source_chain_and_compatibility_contract(self) -> None:
        barrier = _v5_barrier()
        validated = compat._validate_v5_barrier(
            barrier,
            controller_id=CONTROLLER_ID,
            amendment=_amendment(),
            v3_barrier=_v3_barrier(),
            v4_barrier=_v4_barrier(),
        )
        self.assertEqual(validated, barrier)
        self.assertEqual(barrier["v5_supervisor_sha256"], compat._v5_sha256())
        self.assertEqual(barrier["source_journal"]["revision"], 36)
        self.assertEqual(
            barrier["compatibility_contract"]["legacy_artifact_sha256"],
            compat.LIVE_EVALUATION_RUN_CONTRACT_SHA256,
        )
        self.assertEqual(barrier["v5_barrier_artifact"], compat.V5_BARRIER_ARTIFACT)
        self.assertEqual(
            barrier["source_evaluation_contract"],
            compat._expected_live_evaluation_contract_binding(),
        )

    def test_live_source_contract_is_authoritatively_revalidated(self) -> None:
        parameters = _runtime_parameters()
        parameters["Args/predecessor_task_id"] = (
            compat.LIVE_COMPLETED_EVALUATION_TRAINING_TASK_ID
        )
        parameters["Args/controlled_baseline_task_id"] = (
            compat.LIVE_COMPLETED_EVALUATION_TRAINING_TASK_ID
        )
        fresh_task = _ParameterTask(parameters)
        fresh_task.id = compat.LIVE_COMPLETED_EVALUATION_TASK_ID
        fresh_task.artifacts = {"run_contract": _artifact()}
        with (
            mock.patch.object(v3, "_fresh_task", return_value=fresh_task),
            mock.patch.object(base, "_task_status", return_value="completed"),
            mock.patch.object(base, "_task_parent", return_value=CONTROLLER_ID),
            mock.patch.object(
                v3,
                "_authoritative_artifact",
                return_value=(fresh_task, _legacy_contract(), {}, b"{}"),
            ),
        ):
            observed = compat._validate_live_evaluation_contract(
                SimpleNamespace(), controller_id=CONTROLLER_ID
            )
        self.assertEqual(observed, compat._expected_live_evaluation_contract_binding())

        fresh_task.artifacts["run_contract"] = SimpleNamespace(hash="0" * 64)
        with (
            mock.patch.object(v3, "_fresh_task", return_value=fresh_task),
            mock.patch.object(base, "_task_status", return_value="completed"),
            mock.patch.object(base, "_task_parent", return_value=CONTROLLER_ID),
            mock.patch.object(
                v3,
                "_authoritative_artifact",
                return_value=(fresh_task, _legacy_contract(), {}, b"{}"),
            ),
        ):
            with self.assertRaisesRegex(
                compat.P1ParallelV5Error, "run contract SHA-256 drifted"
            ):
                compat._validate_live_evaluation_contract(
                    SimpleNamespace(), controller_id=CONTROLLER_ID
                )

    def test_v5_barrier_rejects_resealed_binding_or_contract_drift(self) -> None:
        barrier = _v5_barrier()
        cases = [
            ("controller_task_id", "d" * 32),
            ("execution_key", "e" * 64),
            ("v4_barrier_seal_sha256", "f" * 64),
        ]
        for field, replacement in cases:
            with self.subTest(field=field):
                drifted = copy.deepcopy(barrier)
                drifted[field] = replacement
                drifted["seal_sha256"] = base._seal(drifted)
                with self.assertRaisesRegex(
                    compat.P1ParallelV5Error, f"barrier {field} drifted"
                ):
                    compat._validate_v5_barrier(
                        drifted,
                        controller_id=CONTROLLER_ID,
                        amendment=_amendment(),
                        v3_barrier=_v3_barrier(),
                        v4_barrier=_v4_barrier(),
                    )

        drifted = copy.deepcopy(barrier)
        drifted["compatibility_contract"]["artifact_rewrite_forbidden"] = False
        drifted["seal_sha256"] = base._seal(drifted)
        with self.assertRaisesRegex(
            compat.P1ParallelV5Error, "compatibility_contract drifted"
        ):
            compat._validate_v5_barrier(
                drifted,
                controller_id=CONTROLLER_ID,
                amendment=_amendment(),
                v3_barrier=_v3_barrier(),
                v4_barrier=_v4_barrier(),
            )

    def test_v5_barrier_install_is_idempotent_and_requires_absent_wal(self) -> None:
        barrier = _v5_barrier()
        with (
            mock.patch.object(
                compat, "_read_installed_v5_barrier", side_effect=[None, barrier]
            ),
            mock.patch.object(compat, "_build_v5_barrier", return_value=barrier),
            mock.patch.object(
                compat,
                "_validate_live_evaluation_contract",
                return_value=compat._expected_live_evaluation_contract_binding(),
            ),
            mock.patch.object(v3, "_read_pending_wal", return_value=None),
            mock.patch.object(v3, "_upload_mapping_confirmed") as upload,
        ):
            installed = compat._install_v5_barrier(
                SimpleNamespace(),
                controller_id=CONTROLLER_ID,
                journal=_source_journal(),
                amendment=_amendment(),
                v3_barrier=_v3_barrier(),
                v4_barrier=_v4_barrier(),
            )
        self.assertEqual(installed, barrier)
        upload.assert_called_once()

        with (
            mock.patch.object(compat, "_read_installed_v5_barrier", return_value=None),
            mock.patch.object(
                compat,
                "_validate_live_evaluation_contract",
                return_value=compat._expected_live_evaluation_contract_binding(),
            ),
            mock.patch.object(v3, "_read_pending_wal", return_value={"pending": True}),
            mock.patch.object(v3, "_upload_mapping_confirmed") as upload,
        ):
            with self.assertRaisesRegex(
                compat.P1ParallelV5Error, "requires an absent pending WAL"
            ):
                compat._install_v5_barrier(
                    SimpleNamespace(),
                    controller_id=CONTROLLER_ID,
                    journal=_source_journal(),
                    amendment=_amendment(),
                    v3_barrier=_v3_barrier(),
                    v4_barrier=_v4_barrier(),
                )
        upload.assert_not_called()

    def test_old_v4_rejects_v5_barrier_before_any_mutation(self) -> None:
        controller = SimpleNamespace(
            id=CONTROLLER_ID,
            artifacts={compat.V5_BARRIER_ARTIFACT: object()},
        )
        receipt = {"existing_execution": {"controller_id": CONTROLLER_ID}}
        task_class = SimpleNamespace(
            enqueue=mock.Mock(side_effect=AssertionError("old v4 enqueued"))
        )
        with (
            mock.patch.object(v3, "_fresh_task", return_value=controller),
            mock.patch.object(base, "_validate_controller"),
            mock.patch.object(base, "_upload_artifact") as upload,
            mock.patch.object(base, "_create_task") as create,
        ):
            with self.assertRaisesRegex(v4.P1ParallelV4Error, "unknown entries"):
                v4._load_existing_execution(task_class, receipt)
        upload.assert_not_called()
        create.assert_not_called()
        task_class.enqueue.assert_not_called()

    def test_v5_existing_execution_allows_only_v5_and_revalidates_barriers(
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
                v4.V4_BARRIER_ARTIFACT: object(),
                compat.V5_BARRIER_ARTIFACT: object(),
            },
        )
        journal = {"revision": 37, "seal_sha256": "7" * 64}
        receipt = {"existing_execution": {"controller_id": CONTROLLER_ID}}
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
            observed = compat._load_existing_execution(SimpleNamespace(), receipt)
        self.assertEqual(observed, (controller, journal))
        validate.assert_called_once_with(
            mock.ANY, controller_id=CONTROLLER_ID, journal=journal
        )

    def test_every_v2_commit_read_revalidates_v4_and_v5_barriers(self) -> None:
        task_class = SimpleNamespace(name="task-class")
        controller = SimpleNamespace(id=CONTROLLER_ID)
        journal = {"revision": 37, "seal_sha256": "8" * 64}
        original_v2_remote = v2._read_remote_journal
        with (
            mock.patch.object(
                v3,
                "_authoritative_remote_journal",
                return_value=(controller, journal),
            ),
            mock.patch.object(v4, "_validate_bound_barriers") as validate_v4,
            mock.patch.object(compat, "_validate_bound_barriers") as validate_v5,
        ):
            with compat._v5_runtime():
                with v4._v4_runtime():
                    with v3._authoritative_v2_runtime(task_class=task_class):
                        observed = v2._read_remote_journal(task_class, CONTROLLER_ID)
        self.assertEqual(observed, (controller, journal))
        validate_v4.assert_called_once_with(
            task_class, controller_id=CONTROLLER_ID, journal=journal
        )
        validate_v5.assert_called_once_with(
            task_class, controller_id=CONTROLLER_ID, journal=journal
        )
        self.assertIs(v2._read_remote_journal, original_v2_remote)

    def test_preflight_is_read_only_and_requires_exact_source_before_install(
        self,
    ) -> None:
        task_class = SimpleNamespace(name="task-class")
        v4_receipt = {
            "seal_sha256": "1" * 64,
            "queue": {"status": "readonly"},
            "global_mutex": {"status": "available"},
        }
        base_receipt = {"existing_execution": {"controller_id": CONTROLLER_ID}}
        with (
            mock.patch.object(v4, "preflight", return_value=v4_receipt),
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
                    _source_journal(),
                    _amendment(),
                    _v3_barrier(),
                    _v4_barrier(),
                ),
            ),
            mock.patch.object(compat, "_read_installed_v5_barrier", return_value=None),
            mock.patch.object(v3, "_read_pending_wal", return_value=None),
            mock.patch.object(
                compat,
                "_validate_live_evaluation_contract",
                return_value=compat._expected_live_evaluation_contract_binding(),
            ) as live_contract,
            mock.patch.object(v3, "_upload_mapping_confirmed") as upload,
        ):
            receipt = compat.preflight(task_class=task_class)
        self.assertTrue(receipt["readonly"])
        self.assertEqual(receipt["remote_mutation_count"], 0)
        self.assertEqual(receipt["v5_barrier_status"], "not_installed")
        self.assertEqual(receipt["source_binding"]["revision"], 36)
        self.assertEqual(
            receipt["source_evaluation_contract"],
            compat._expected_live_evaluation_contract_binding(),
        )
        live_contract.assert_called_once_with(task_class, controller_id=CONTROLLER_ID)
        upload.assert_not_called()

    def test_execute_installs_v5_barrier_before_running_patched_v4(self) -> None:
        events = []
        migration = {
            "controller_task_id": CONTROLLER_ID,
            "source_journal_revision": 36,
            "source_journal_seal_sha256": compat.LIVE_SOURCE_JOURNAL_SEAL_SHA256,
            "v4_barrier_seal_sha256": "4" * 64,
            "v5_barrier_seal_sha256": "5" * 64,
        }

        def install(**kwargs: object) -> dict[str, object]:
            del kwargs
            events.append("v5-installed")
            return migration

        @contextlib.contextmanager
        def runtime() -> Iterator[None]:
            self.assertEqual(events, ["v5-installed"])
            events.append("v5-runtime")
            yield

        def execute(**kwargs: object) -> dict[str, object]:
            self.assertEqual(events[-1], "v5-runtime")
            self.assertEqual(kwargs["authorization_token"], v4.V4_EXECUTION_TOKEN)
            events.append("v4-executed")
            return {"status": "completed", "seal_sha256": "6" * 64}

        with (
            mock.patch.object(v3, "_validate_control_host"),
            mock.patch.object(
                compat, "_install_v5_barrier_under_mutex", side_effect=install
            ),
            mock.patch.object(compat, "_v5_runtime", side_effect=runtime),
            mock.patch.object(v4, "execute_plan", side_effect=execute),
        ):
            receipt = compat.execute_plan(
                authorization_token=compat.V5_EXECUTION_TOKEN,
                poll_seconds=5.0,
                timeout_hours=6.0,
                task_class=SimpleNamespace(),
                api_client=SimpleNamespace(),
            )
        self.assertEqual(events, ["v5-installed", "v5-runtime", "v4-executed"])
        self.assertEqual(receipt["v5_barrier_seal_sha256"], "5" * 64)
        self.assertEqual(receipt["seal_sha256"], base._seal(receipt))

    def test_execute_requires_exact_v5_token_before_remote_checks(self) -> None:
        with mock.patch.object(v3, "_validate_control_host") as host:
            with self.assertRaisesRegex(
                compat.P1ParallelV5Error, "execution token mismatch"
            ):
                compat.execute_plan(
                    authorization_token=v4.V4_EXECUTION_TOKEN,
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
        self.assertEqual(receipt["document_type"], compat.V5_DRY_RUN_DOCUMENT_TYPE)
        self.assertFalse(receipt["remote_mutation_authorized"])
        self.assertEqual(receipt["v4_supervisor_sha256"], compat.FROZEN_V4_SHA256)

    def test_preflight_cli_never_dispatches_execute(self) -> None:
        receipt = {
            "schema_version": 1,
            "document_type": compat.V5_PREFLIGHT_DOCUMENT_TYPE,
            "readonly": True,
            "remote_mutation_count": 0,
            "seal_sha256": "9" * 64,
        }
        output = io.StringIO()
        with (
            mock.patch.object(compat, "preflight", return_value=receipt) as read,
            mock.patch.object(compat, "execute_plan") as execute,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(compat.main(["--preflight"]), 0)
        self.assertEqual(json.loads(output.getvalue()), receipt)
        read.assert_called_once()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
