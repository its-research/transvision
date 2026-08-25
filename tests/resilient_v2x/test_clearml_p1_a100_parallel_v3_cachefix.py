from __future__ import annotations

import builtins
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.resilient_v2x import clearml_p1_a100_parallel_v3_cachefix as cachefix


base = cachefix.base
legacy = cachefix.v2

CONTROLLER_ID = "c" * 32
FROZEN_V2_SHA256 = "124909587ef4866c615ca05427c9f5242d8230661b5b0ae98191e4589c477148"


def _new_journal(*, revision: int = 5) -> dict[str, object]:
    journal = base._new_journal(
        plan=base.build_plan(),
        execution_key=legacy.SEALED_EXECUTION_KEY,
        controller_id=CONTROLLER_ID,
    )
    journal["status"] = "running"
    journal["revision"] = revision
    journal["seal_sha256"] = base._seal(journal)
    return journal


def _candidate(
    prior: Mapping[str, object],
    *,
    task_id: str = "1" * 32,
    state: str = "discovered",
    server_status: str = "created",
) -> dict[str, object]:
    candidate = copy.deepcopy(dict(prior))
    rows = candidate["tasks"]
    if not isinstance(rows, list):
        raise AssertionError("journal task fixture is invalid")
    row = rows[0]
    if not isinstance(row, dict):
        raise AssertionError("journal row fixture is invalid")
    row.update(
        {
            "task_id": task_id,
            "state": state,
            "server_status": server_status,
            "result": None,
        }
    )
    candidate["status"] = "running"
    candidate["revision"] = int(prior["revision"]) + 1
    candidate["seal_sha256"] = base._seal(candidate)
    return candidate


def _next_candidate(prior: Mapping[str, object]) -> dict[str, object]:
    candidate = copy.deepcopy(dict(prior))
    rows = candidate["tasks"]
    if not isinstance(rows, list) or not isinstance(rows[0], dict):
        raise AssertionError("journal task fixture is invalid")
    rows[0].update(
        {
            "state": "prepared",
            "server_status": "created",
            "result": None,
        }
    )
    candidate["status"] = "running"
    candidate["revision"] = int(prior["revision"]) + 1
    candidate["seal_sha256"] = base._seal(candidate)
    return candidate


def _wal(
    prior: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    return legacy._build_pending_journal_receipt(
        controller_id=CONTROLLER_ID,
        prior=prior,
        candidate=candidate,
        status="pending",
        observed=prior,
    )


def _amendment(journal: Mapping[str, object]) -> dict[str, object]:
    return legacy._build_migration_intent(
        controller_id=CONTROLLER_ID,
        journal=journal,
        active_task_ids=legacy._active_task_ids_from_journal(journal),
    )


@contextlib.contextmanager
def _synthetic_live_pins(
    *,
    amendment: Mapping[str, object],
    wal: Mapping[str, object],
) -> Iterator[None]:
    prior = wal["prior_journal"]
    candidate = wal["candidate_journal"]
    if not isinstance(prior, Mapping) or not isinstance(candidate, Mapping):
        raise AssertionError("synthetic live WAL fixture is invalid")
    candidate_raw = cachefix._sdk_json_bytes(candidate)
    with (
        mock.patch.object(
            cachefix,
            "LIVE_V2_AMENDMENT_SEAL_SHA256",
            amendment["seal_sha256"],
        ),
        mock.patch.object(
            cachefix,
            "LIVE_INITIAL_WAL_SEAL_SHA256",
            wal["seal_sha256"],
        ),
        mock.patch.object(
            cachefix,
            "LIVE_INITIAL_PRIOR_JOURNAL_SEAL_SHA256",
            prior["seal_sha256"],
        ),
        mock.patch.object(
            cachefix,
            "LIVE_INITIAL_CANDIDATE_JOURNAL_SEAL_SHA256",
            candidate["seal_sha256"],
        ),
        mock.patch.object(
            cachefix,
            "LIVE_INITIAL_CANDIDATE_DESCRIPTOR_SHA256",
            hashlib.sha256(candidate_raw).hexdigest(),
        ),
        mock.patch.object(
            cachefix,
            "LIVE_INITIAL_CANDIDATE_DESCRIPTOR_SIZE",
            len(candidate_raw),
        ),
    ):
        yield


def _barrier_fixture(
    *,
    amendment: Mapping[str, object],
    wal: Mapping[str, object],
) -> dict[str, object]:
    with _synthetic_live_pins(amendment=amendment, wal=wal):
        return cachefix._build_v3_barrier(
            controller_id=CONTROLLER_ID,
            amendment=amendment,
            wal=wal,
        )


def _descriptor(
    value: Mapping[str, object], *, name: str = base.JOURNAL_ARTIFACT
) -> dict[str, object]:
    raw = cachefix._sdk_json_bytes(value)
    return {
        "key": name,
        "hash": hashlib.sha256(raw).hexdigest(),
        "content_size": len(raw),
        "uri": f"{base.FILES_SERVER_URI}/authoritative/{name}.json",
        "type": "dict",
        "mode": "output",
    }


def _journal_state(
    state: str, value: Mapping[str, object]
) -> tuple[str, object, dict[str, object], dict[str, object], bytes]:
    return (
        state,
        SimpleNamespace(id=CONTROLLER_ID),
        _descriptor(value),
        copy.deepcopy(dict(value)),
        cachefix._sdk_json_bytes(value),
    )


@contextlib.contextmanager
def _durable_pending_wal(
    receipt: Mapping[str, object],
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / legacy.PENDING_JOURNAL_FILENAME
        with mock.patch.object(legacy, "_pending_journal_path", return_value=path):
            legacy._write_pending_journal_receipt(receipt)
            yield path


class _ForcedArtifact:
    def __init__(
        self,
        path: Path,
        raw: bytes,
        *,
        artifact_hash: str | None = None,
        content_size: int | None = None,
        uri: str | None = None,
        name: str = "authoritative fixture",
    ) -> None:
        path.write_bytes(raw)
        self.path = path
        self.hash = artifact_hash or hashlib.sha256(raw).hexdigest()
        self.size = len(raw) if content_size is None else content_size
        self.url = uri or f"{base.FILES_SERVER_URI}/authoritative/artifact.json"
        self.type = "dict"
        self.mode = "output"
        self.name = name
        self.local_copy_calls: list[dict[str, object]] = []
        self.get_calls = 0

    def get_local_copy(self, **kwargs: object) -> str:
        self.local_copy_calls.append(dict(kwargs))
        return str(self.path)

    def get(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.get_calls += 1
        raise AssertionError("Artifact.get() cache is not authoritative")


class ParallelV3CacheFixTests(unittest.TestCase):
    def test_frozen_v2_and_source_wal_revision_are_exact(self) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        receipt = _wal(prior, candidate)

        self.assertEqual(cachefix.FROZEN_V2_SHA256, FROZEN_V2_SHA256)
        self.assertEqual(legacy._supervisor_sha256(), FROZEN_V2_SHA256)
        self.assertEqual(prior["revision"], 5)
        self.assertEqual(candidate["revision"], 6)
        self.assertEqual(receipt["prior_journal"], prior)
        self.assertEqual(receipt["candidate_journal"], candidate)
        self.assertEqual(receipt["observed_journal"], prior)
        self.assertEqual(
            cachefix._journal_transition_kind(prior, candidate), "task_row"
        )

    def test_sdk_json_bytes_match_the_pinned_clearml_dict_encoding(self) -> None:
        value = {"z": [1, 2], "a": {"unicode": "\u8f66"}}
        expected = json.dumps(value, sort_keys=True, indent=4).encode("utf-8")
        self.assertEqual(cachefix._sdk_json_bytes(value), expected)

    def test_forced_artifact_read_checks_hash_size_and_raw_json_exactly(self) -> None:
        value = {"schema_version": 1, "payload": {"revision": 6}}
        raw = cachefix._sdk_json_bytes(value)
        with tempfile.TemporaryDirectory() as directory:
            artifact = _ForcedArtifact(Path(directory) / "artifact.json", raw)
            observed, descriptor, observed_raw = cachefix._force_artifact_mapping(
                artifact,
                name="authoritative fixture",
            )

        self.assertEqual(observed, value)
        self.assertEqual(observed_raw, raw)
        self.assertEqual(descriptor["hash"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(descriptor["content_size"], len(raw))
        self.assertEqual(
            artifact.local_copy_calls,
            [
                {
                    "extract_archive": False,
                    "raise_on_error": True,
                    "force_download": True,
                }
            ],
        )
        self.assertEqual(artifact.get_calls, 0)

    def test_forced_artifact_read_rejects_descriptor_hash_or_size_drift(
        self,
    ) -> None:
        value = {"schema_version": 1, "payload": "candidate"}
        raw = cachefix._sdk_json_bytes(value)
        cases = {
            "hash": {"artifact_hash": "0" * 64},
            "content size": {"content_size": len(raw) + 1},
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    artifact = _ForcedArtifact(
                        Path(directory) / "artifact.json",
                        raw,
                        **kwargs,
                    )
                    with self.assertRaisesRegex(
                        cachefix.P1ParallelV3Error,
                        "descriptor and forced bytes differ",
                    ):
                        cachefix._force_artifact_mapping(
                            artifact,
                            name="authoritative fixture",
                        )
                self.assertEqual(artifact.get_calls, 0)

    def test_backend_descriptor_comes_from_direct_get_by_id_not_task_cache(
        self,
    ) -> None:
        descriptor = SimpleNamespace(
            key=base.JOURNAL_ARTIFACT,
            hash="a" * 64,
            content_size=123,
            uri=f"{base.FILES_SERVER_URI}/backend/journal.json",
            type="dict",
            mode="output",
        )
        task_data = SimpleNamespace(
            id=CONTROLLER_ID,
            execution=SimpleNamespace(artifacts=[descriptor]),
        )
        response = SimpleNamespace(response=SimpleNamespace(task=task_data))
        session = SimpleNamespace(send=mock.Mock(return_value=response))

        class TaskClass:
            get_task = mock.Mock(
                side_effect=AssertionError(
                    "Task.get_task cache is not descriptor authority"
                )
            )

            @classmethod
            def _get_default_session(cls) -> object:
                return session

        observed = cachefix._backend_artifact_descriptor(
            TaskClass,
            task_id=CONTROLLER_ID,
            artifact_name=base.JOURNAL_ARTIFACT,
        )

        self.assertIs(observed, descriptor)
        session.send.assert_called_once()
        TaskClass.get_task.assert_not_called()

    def test_authoritative_read_ignores_artifact_get_cache(self) -> None:
        candidate = _candidate(_new_journal())
        raw = cachefix._sdk_json_bytes(candidate)
        backend_descriptor = SimpleNamespace(
            key=base.JOURNAL_ARTIFACT,
            hash=hashlib.sha256(raw).hexdigest(),
            content_size=len(raw),
            uri=f"{base.FILES_SERVER_URI}/backend/journal.json",
            type="dict",
            mode="output",
        )
        poison = SimpleNamespace(
            get=mock.Mock(
                side_effect=AssertionError("cached Artifact.get() was consulted")
            )
        )
        controller = SimpleNamespace(
            id=CONTROLLER_ID,
            artifacts={base.JOURNAL_ARTIFACT: poison},
        )

        class TaskClass:
            @classmethod
            def get_task(cls, *, task_id: str) -> object:
                self.assertEqual(task_id, CONTROLLER_ID)
                return controller

        with tempfile.TemporaryDirectory() as directory:
            forced = _ForcedArtifact(
                Path(directory) / "journal.json",
                raw,
                name=base.JOURNAL_ARTIFACT,
            )
            with (
                mock.patch.object(
                    cachefix,
                    "_backend_artifact_descriptor",
                    return_value=backend_descriptor,
                ) as descriptor_read,
                mock.patch(
                    "clearml.binding.artifacts.Artifact",
                    return_value=forced,
                ) as artifact_type,
            ):
                fresh, observed, descriptor, observed_raw = (
                    cachefix._authoritative_artifact(
                        TaskClass,
                        task_id=CONTROLLER_ID,
                        artifact_name=base.JOURNAL_ARTIFACT,
                    )
                )

        self.assertIs(fresh, controller)
        self.assertEqual(observed, candidate)
        self.assertEqual(observed_raw, raw)
        self.assertEqual(descriptor["hash"], backend_descriptor.hash)
        descriptor_read.assert_called_once_with(
            TaskClass,
            task_id=CONTROLLER_ID,
            artifact_name=base.JOURNAL_ARTIFACT,
        )
        artifact_type.assert_called_once_with(backend_descriptor)
        poison.get.assert_not_called()
        self.assertEqual(forced.get_calls, 0)

    def test_plan_and_pinset_require_authoritative_exact_content(self) -> None:
        plan = base.build_plan()
        pinset = base.build_pinset()
        values = {
            base.PLAN_ARTIFACT: plan,
            base.PINSET_ARTIFACT: pinset,
        }

        def authoritative(
            task_class: object,
            *,
            task_id: str,
            artifact_name: str,
            **kwargs: object,
        ) -> tuple[object, dict[str, object], dict[str, object], bytes]:
            del kwargs
            self.assertIs(task_class, TaskClass)
            self.assertEqual(task_id, CONTROLLER_ID)
            value = values[artifact_name]
            return (
                SimpleNamespace(id=CONTROLLER_ID),
                copy.deepcopy(value),
                _descriptor(value, name=artifact_name),
                cachefix._sdk_json_bytes(value),
            )

        class TaskClass:
            pass

        with mock.patch.object(
            cachefix, "_authoritative_artifact", side_effect=authoritative
        ) as read:
            receipt = cachefix._validate_authoritative_immutable_bindings(
                TaskClass,
                controller_id=CONTROLLER_ID,
                plan=plan,
                pinset=pinset,
            )

        self.assertEqual(
            [call.kwargs["artifact_name"] for call in read.call_args_list],
            [base.PLAN_ARTIFACT, base.PINSET_ARTIFACT],
        )
        self.assertEqual(receipt[base.PLAN_ARTIFACT]["key"], base.PLAN_ARTIFACT)
        self.assertEqual(receipt[base.PINSET_ARTIFACT]["key"], base.PINSET_ARTIFACT)

        plan_raw = cachefix._sdk_json_bytes(plan)
        drifted_raw = plan_raw + b"\n"
        with mock.patch.object(
            cachefix,
            "_authoritative_artifact",
            return_value=(
                SimpleNamespace(id=CONTROLLER_ID),
                copy.deepcopy(plan),
                {
                    **_descriptor(plan, name=base.PLAN_ARTIFACT),
                    "hash": hashlib.sha256(drifted_raw).hexdigest(),
                    "content_size": len(drifted_raw),
                },
                drifted_raw,
            ),
        ):
            with self.assertRaisesRegex(
                cachefix.P1ParallelV3Error,
                f"controller {base.PLAN_ARTIFACT} drifted",
            ):
                cachefix._validate_authoritative_immutable_bindings(
                    TaskClass,
                    controller_id=CONTROLLER_ID,
                    plan=plan,
                    pinset=pinset,
                )

    def test_v3_barrier_binds_controller_amendment_key_and_rev5_to_rev6_wal(
        self,
    ) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)
        amendment = _amendment(prior)

        with _synthetic_live_pins(amendment=amendment, wal=wal):
            barrier = cachefix._build_v3_barrier(
                controller_id=CONTROLLER_ID,
                amendment=amendment,
                wal=wal,
            )
            validated = cachefix._validate_v3_barrier(
                barrier,
                controller_id=CONTROLLER_ID,
                amendment=amendment,
            )

        self.assertEqual(validated, barrier)
        self.assertEqual(barrier["v2_supervisor_sha256"], FROZEN_V2_SHA256)
        self.assertEqual(barrier["v3_supervisor_sha256"], cachefix._v3_sha256())
        self.assertEqual(barrier["controller_task_id"], CONTROLLER_ID)
        self.assertEqual(barrier["execution_key"], legacy.SEALED_EXECUTION_KEY)
        self.assertEqual(barrier["v2_amendment_artifact"], legacy.AMENDMENT_ARTIFACT)
        self.assertEqual(barrier["v2_amendment_seal_sha256"], amendment["seal_sha256"])
        source = barrier["source_pending_wal"]
        self.assertIsInstance(source, dict)
        self.assertEqual(source["wal_seal_sha256"], wal["seal_sha256"])
        self.assertEqual(source["prior_revision"], 5)
        self.assertEqual(source["prior_journal_seal_sha256"], prior["seal_sha256"])
        self.assertEqual(source["candidate_revision"], 6)
        self.assertEqual(
            source["candidate_journal_seal_sha256"], candidate["seal_sha256"]
        )
        self.assertEqual(source["transition_kind"], "task_row")
        self.assertEqual(
            barrier["cache_fix_contract"],
            {
                "backend_descriptor_required": True,
                "fresh_task_required": True,
                "force_download_required": True,
                "descriptor_hash_required": True,
                "descriptor_content_size_required": True,
                "raw_json_exact_required": True,
                "pending_candidate_retry_per_invocation_limit": 1,
                "pending_candidate_retry_must_be_byte_identical": True,
                "old_v2_execution_forbidden": True,
            },
        )
        self.assertEqual(
            barrier["journal_writer_attribution"],
            {
                "scheduler_core_sha256": FROZEN_V2_SHA256,
                "runtime_cachefix_sha256": cachefix._v3_sha256(),
                "upload_protocol": (
                    "frozen_v2_clearml_sdk_plus_authoritative_confirm_v1"
                ),
            },
        )
        cachefix._barrier_matches_wal(barrier, wal)

    def test_v3_barrier_rejects_a_different_valid_rev5_to_rev6_bundle(self) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)
        amendment = _amendment(prior)

        with self.assertRaisesRegex(
            cachefix.P1ParallelV3Error, "initial live migration pins drifted"
        ):
            cachefix._build_v3_barrier(
                controller_id=CONTROLLER_ID,
                amendment=amendment,
                wal=wal,
            )

    def test_v3_barrier_rejects_rebinding_even_when_resealed(self) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)
        amendment = _amendment(prior)
        with _synthetic_live_pins(amendment=amendment, wal=wal):
            barrier = cachefix._build_v3_barrier(
                controller_id=CONTROLLER_ID,
                amendment=amendment,
                wal=wal,
            )

            for field, replacement in (
                ("controller_task_id", "d" * 32),
                ("execution_key", "e" * 64),
                ("v2_supervisor_sha256", "f" * 64),
                ("v2_amendment_seal_sha256", "0" * 64),
            ):
                with self.subTest(field=field):
                    drifted = copy.deepcopy(barrier)
                    drifted[field] = replacement
                    drifted["seal_sha256"] = base._seal(drifted)
                    with self.assertRaisesRegex(
                        cachefix.P1ParallelV3Error, "barrier .* drifted"
                    ):
                        cachefix._validate_v3_barrier(
                            drifted,
                            controller_id=CONTROLLER_ID,
                            amendment=amendment,
                        )

            drifted = copy.deepcopy(barrier)
            drifted_source = drifted["source_pending_wal"]
            self.assertIsInstance(drifted_source, dict)
            drifted_source["wal_seal_sha256"] = "0" * 64
            drifted["seal_sha256"] = base._seal(drifted)
            with self.assertRaisesRegex(
                cachefix.P1ParallelV3Error, "source WAL drifted"
            ):
                cachefix._validate_v3_barrier(
                    drifted,
                    controller_id=CONTROLLER_ID,
                    amendment=amendment,
                )
            with self.assertRaisesRegex(
                cachefix.P1ParallelV3Error,
                "source WAL drifted|barrier and pending WAL differ",
            ):
                cachefix._barrier_matches_wal(drifted, wal)

    def test_legacy_1249_rejects_v3_barrier_before_any_mutation(self) -> None:
        controller = SimpleNamespace(
            id=CONTROLLER_ID,
            artifacts={cachefix.V3_BARRIER_ARTIFACT: object()},
        )
        base_receipt = {"existing_execution": {"controller_id": CONTROLLER_ID}}

        class TaskClass:
            enqueue = mock.Mock(
                side_effect=AssertionError("legacy supervisor attempted enqueue")
            )

        with (
            mock.patch.object(base, "_fresh", return_value=controller),
            mock.patch.object(base, "_validate_controller"),
            mock.patch.object(
                base,
                "_upload_artifact",
                side_effect=AssertionError("legacy supervisor attempted upload"),
            ) as upload,
            mock.patch.object(
                base,
                "_create_task",
                side_effect=AssertionError("legacy supervisor attempted clone"),
            ) as clone,
        ):
            with self.assertRaisesRegex(legacy.P1ParallelV2Error, "unknown entries"):
                legacy._load_existing_execution(TaskClass, base_receipt)

        upload.assert_not_called()
        clone.assert_not_called()
        TaskClass.enqueue.assert_not_called()

    def test_remote_prior_replays_exact_candidate_once_and_removes_wal(self) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)

        with _durable_pending_wal(wal) as path:
            with (
                mock.patch.object(
                    cachefix,
                    "_journal_descriptor_state",
                    side_effect=[
                        _journal_state("prior", prior),
                        _journal_state("candidate", candidate),
                    ],
                ) as authoritative_read,
                mock.patch.object(base, "_upload_artifact") as upload,
                mock.patch.object(
                    legacy,
                    "_read_remote_journal",
                    side_effect=AssertionError(
                        "ordinary v2 cache reader cannot confirm recovery"
                    ),
                ) as ordinary_read,
            ):
                result = cachefix._recover_pending_candidate_once(
                    SimpleNamespace(),
                    controller_id=CONTROLLER_ID,
                    wal=wal,
                )

            self.assertFalse(path.exists())

        self.assertEqual(result["before_state"], "prior")
        self.assertEqual(result["candidate_revision"], 6)
        self.assertEqual(result["upload_attempts"], 1)
        self.assertTrue(result["wal_removed"])
        upload.assert_called_once()
        uploaded_controller, uploaded_name, uploaded_value = upload.call_args.args
        self.assertEqual(uploaded_controller.id, CONTROLLER_ID)
        self.assertEqual(uploaded_name, base.JOURNAL_ARTIFACT)
        self.assertEqual(uploaded_value, candidate)
        self.assertEqual(authoritative_read.call_count, 2)
        ordinary_read.assert_not_called()

    def test_remote_candidate_is_adopted_without_upload_or_cache_warm_gate(
        self,
    ) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)

        with _durable_pending_wal(wal) as path:
            with (
                mock.patch.object(
                    cachefix,
                    "_journal_descriptor_state",
                    return_value=_journal_state("candidate", candidate),
                ) as authoritative_read,
                mock.patch.object(base, "_upload_artifact") as upload,
                mock.patch.object(
                    legacy,
                    "_read_remote_journal",
                    side_effect=AssertionError(
                        "ordinary v2 cache reader cannot gate authoritative adoption"
                    ),
                ) as ordinary_read,
            ):
                result = cachefix._recover_pending_candidate_once(
                    SimpleNamespace(),
                    controller_id=CONTROLLER_ID,
                    wal=wal,
                )

            self.assertFalse(path.exists())

        self.assertEqual(result["before_state"], "candidate")
        self.assertEqual(result["upload_attempts"], 0)
        upload.assert_not_called()
        self.assertGreaterEqual(authoritative_read.call_count, 1)
        ordinary_read.assert_not_called()

    def test_failed_exact_replay_is_not_retried_and_retains_wal(self) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)

        with _durable_pending_wal(wal) as path:
            with (
                mock.patch.object(
                    cachefix,
                    "_journal_descriptor_state",
                    side_effect=[
                        _journal_state("prior", prior),
                        _journal_state("prior", prior),
                    ],
                ),
                mock.patch.object(
                    base,
                    "_upload_artifact",
                    side_effect=RuntimeError("upload failed before commit"),
                ) as upload,
                mock.patch.object(legacy, "_remove_pending_journal_receipt") as remove,
            ):
                with self.assertRaisesRegex(
                    cachefix.P1ParallelV3Error,
                    "did not commit authoritatively|WAL was retained",
                ):
                    cachefix._recover_pending_candidate_once(
                        SimpleNamespace(),
                        controller_id=CONTROLLER_ID,
                        wal=wal,
                    )

            self.assertTrue(path.exists())

        upload.assert_called_once()
        remove.assert_not_called()

    def test_failed_authoritative_confirmation_retains_wal(self) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)

        with _durable_pending_wal(wal) as path:
            with (
                mock.patch.object(
                    cachefix,
                    "_journal_descriptor_state",
                    side_effect=[
                        _journal_state("prior", prior),
                        cachefix.P1ParallelV3Error("confirmation unavailable"),
                    ],
                ),
                mock.patch.object(base, "_upload_artifact") as upload,
                mock.patch.object(legacy, "_remove_pending_journal_receipt") as remove,
            ):
                with self.assertRaisesRegex(
                    cachefix.P1ParallelV3Error, "confirmation unavailable"
                ):
                    cachefix._recover_pending_candidate_once(
                        SimpleNamespace(),
                        controller_id=CONTROLLER_ID,
                        wal=wal,
                    )

            self.assertTrue(path.exists())

        upload.assert_called_once()
        remove.assert_not_called()

    def test_third_remote_state_is_rejected_without_upload_and_retains_wal(
        self,
    ) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)

        with _durable_pending_wal(wal) as path:
            with (
                mock.patch.object(
                    cachefix,
                    "_journal_descriptor_state",
                    side_effect=cachefix.P1ParallelV3Error(
                        "journal backend descriptor is neither prior nor candidate"
                    ),
                ),
                mock.patch.object(base, "_upload_artifact") as upload,
                mock.patch.object(legacy, "_remove_pending_journal_receipt") as remove,
            ):
                with self.assertRaisesRegex(
                    cachefix.P1ParallelV3Error, "neither prior nor candidate"
                ):
                    cachefix._recover_pending_candidate_once(
                        SimpleNamespace(),
                        controller_id=CONTROLLER_ID,
                        wal=wal,
                    )

            self.assertTrue(path.exists())

        upload.assert_not_called()
        remove.assert_not_called()

    def test_execute_commits_barrier_before_any_journal_force_read_or_wal_delete(
        self,
    ) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)
        amendment = _amendment(prior)
        controller = SimpleNamespace(id=CONTROLLER_ID)
        barrier = {"seal_sha256": "b" * 64}
        base_receipt = {
            "existing_execution": {"controller_id": CONTROLLER_ID},
        }
        events: list[str] = []

        def install(*args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            events.append("barrier-install")
            return barrier

        def journal_read(
            *args: object, **kwargs: object
        ) -> tuple[str, object, dict[str, object], dict[str, object], bytes]:
            del args, kwargs
            self.assertIn("barrier-install", events)
            events.append("journal-descriptor-force")
            return _journal_state("candidate", candidate)

        def remove(receipt: Mapping[str, object]) -> None:
            self.assertEqual(receipt, wal)
            self.assertIn("barrier-install", events)
            events.append("wal-delete")

        with (
            mock.patch.object(
                base, "_execution_lock", return_value=contextlib.nullcontext()
            ),
            mock.patch.object(
                base,
                "_server_execution_mutex",
                return_value=contextlib.nullcontext(),
            ),
            mock.patch.object(
                cachefix,
                "_read_base_preflight",
                return_value=(SimpleNamespace(), base_receipt),
            ),
            mock.patch.object(
                cachefix,
                "_controller_from_receipt",
                return_value=(controller, CONTROLLER_ID),
            ),
            mock.patch.object(cachefix, "_validate_authoritative_immutable_bindings"),
            mock.patch.object(cachefix, "_read_pending_wal", return_value=wal),
            mock.patch.object(
                cachefix, "_validate_v2_amendment", return_value=amendment
            ) as amendment_read,
            mock.patch.object(cachefix, "_install_barrier", side_effect=install),
            mock.patch.object(
                cachefix, "_journal_descriptor_state", side_effect=journal_read
            ) as authoritative_journal,
            mock.patch.object(
                legacy, "_remove_pending_journal_receipt", side_effect=remove
            ) as wal_delete,
            mock.patch.object(base, "_upload_artifact") as upload,
        ):
            result = cachefix._recover_and_install_barrier(
                task_class=SimpleNamespace(),
                api_client=SimpleNamespace(),
                queue_reader=None,
                legacy_retirement_reader=None,
            )

        self.assertEqual(result["barrier_seal_sha256"], barrier["seal_sha256"])
        self.assertEqual(result["recovery"]["upload_attempts"], 0)
        self.assertEqual(events[0], "barrier-install")
        self.assertLess(
            events.index("barrier-install"),
            events.index("journal-descriptor-force"),
        )
        self.assertLess(
            events.index("barrier-install"),
            events.index("wal-delete"),
        )
        amendment_read.assert_called_once_with(
            mock.ANY,
            CONTROLLER_ID,
            candidate,
        )
        self.assertEqual(authoritative_journal.call_count, 2)
        wal_delete.assert_called_once_with(wal)
        upload.assert_not_called()

    def test_restart_repairs_tag_after_barrier_upload_without_reupload(self) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)
        amendment = _amendment(prior)
        barrier = _barrier_fixture(amendment=amendment, wal=wal)

        class SimulatedCrash(RuntimeError):
            pass

        with (
            mock.patch.object(
                cachefix,
                "_read_installed_barrier",
                side_effect=[None, barrier],
            ),
            mock.patch.object(cachefix, "_build_v3_barrier", return_value=barrier),
            mock.patch.object(cachefix, "_upload_mapping_confirmed") as upload,
            mock.patch.object(
                cachefix,
                "_ensure_barrier_tag",
                side_effect=[SimulatedCrash("crash after barrier upload"), None],
            ) as ensure_tag,
        ):
            with self.assertRaises(SimulatedCrash):
                cachefix._install_barrier(
                    SimpleNamespace(),
                    controller_id=CONTROLLER_ID,
                    amendment=amendment,
                    wal=wal,
                )
            installed = cachefix._install_barrier(
                SimpleNamespace(),
                controller_id=CONTROLLER_ID,
                amendment=amendment,
                wal=wal,
            )

        self.assertEqual(installed, barrier)
        upload.assert_called_once()
        self.assertEqual(ensure_tag.call_count, 2)

    def test_installed_barrier_accepts_a_later_runtime_wal(self) -> None:
        prior = _new_journal(revision=5)
        migration_candidate = _candidate(prior)
        migration_wal = _wal(prior, migration_candidate)
        amendment = _amendment(prior)
        barrier = _barrier_fixture(amendment=amendment, wal=migration_wal)
        runtime_candidate = _next_candidate(migration_candidate)
        runtime_wal = _wal(migration_candidate, runtime_candidate)

        self.assertEqual(
            cachefix._wal_relation_to_barrier(barrier, runtime_wal),
            "post_barrier_runtime",
        )
        with (
            mock.patch.object(
                cachefix, "_read_installed_barrier", return_value=barrier
            ),
            mock.patch.object(cachefix, "_ensure_barrier_tag") as ensure_tag,
            mock.patch.object(
                cachefix,
                "_upload_mapping_confirmed",
                side_effect=AssertionError("existing barrier was re-uploaded"),
            ) as upload,
        ):
            installed = cachefix._install_barrier(
                SimpleNamespace(),
                controller_id=CONTROLLER_ID,
                amendment=amendment,
                wal=runtime_wal,
            )

        self.assertEqual(installed, barrier)
        ensure_tag.assert_called_once()
        upload.assert_not_called()

    def test_authoritative_runtime_replaces_every_v2_mutable_journal_hook(
        self,
    ) -> None:
        original_remote = legacy._read_remote_journal
        original_load = base._load_journal
        original_existing = legacy._load_existing_execution
        original_upload = base._upload_artifact
        task_class = SimpleNamespace(name="offline-task-class")
        controller = SimpleNamespace(
            id=CONTROLLER_ID,
            tags=[cachefix.V3_CONTROLLER_TAG],
            system_tags=[],
        )
        expected_journal = _candidate(_new_journal())
        plan = base.build_plan()

        with mock.patch.object(
            cachefix,
            "_authoritative_load_journal",
            return_value=expected_journal,
        ) as authoritative_load:
            with cachefix._authoritative_v2_runtime(task_class=task_class):
                self.assertIs(
                    legacy._read_remote_journal,
                    cachefix._authoritative_remote_journal,
                )
                self.assertIsNot(base._load_journal, original_load)
                self.assertIs(
                    legacy._load_existing_execution,
                    cachefix._load_existing_execution,
                )
                self.assertIs(base._upload_artifact, original_upload)
                observed = base._load_journal(
                    controller,
                    plan=plan,
                    execution_key=legacy.SEALED_EXECUTION_KEY,
                    controller_id=CONTROLLER_ID,
                )

        self.assertEqual(observed, expected_journal)
        authoritative_load.assert_called_once_with(
            controller,
            plan=plan,
            execution_key=legacy.SEALED_EXECUTION_KEY,
            controller_id=CONTROLLER_ID,
            task_class=task_class,
        )

        self.assertIs(legacy._read_remote_journal, original_remote)
        self.assertIs(base._load_journal, original_load)
        self.assertIs(legacy._load_existing_execution, original_existing)
        self.assertIs(base._upload_artifact, original_upload)

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
            self.assertEqual(cachefix.main([]), 0)

        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt["document_type"], cachefix.V3_DRY_RUN_DOCUMENT_TYPE)
        self.assertEqual(receipt["default_mode"], "dry_run")
        self.assertFalse(receipt["remote_mutation_authorized"])
        self.assertEqual(receipt["v2_supervisor_sha256"], FROZEN_V2_SHA256)

    def test_preflight_is_read_only_even_with_pending_wal_and_barrier(self) -> None:
        prior = _new_journal(revision=5)
        candidate = _candidate(prior)
        wal = _wal(prior, candidate)
        amendment = _amendment(prior)
        barrier = _barrier_fixture(amendment=amendment, wal=wal)
        controller = SimpleNamespace(id=CONTROLLER_ID)
        base_receipt = {
            "seal_sha256": "a" * 64,
            "existing_execution": {"controller_id": CONTROLLER_ID},
            "queue": {"status": "read-only-fixture"},
            "global_mutex": {"status": "available"},
        }

        mutation_error = AssertionError("preflight attempted remote mutation")
        with (
            mock.patch.object(
                cachefix,
                "_validate_control_host",
                return_value=copy.deepcopy(legacy.CONTROL_HOST_CONTRACT),
            ),
            mock.patch.object(
                cachefix,
                "_read_base_preflight",
                return_value=(SimpleNamespace(), base_receipt),
            ),
            mock.patch.object(
                cachefix,
                "_controller_from_receipt",
                return_value=(controller, CONTROLLER_ID),
            ),
            mock.patch.object(
                cachefix,
                "_validate_authoritative_immutable_bindings",
                return_value={
                    "plan": {"status": "exact"},
                    "pinset": {"status": "exact"},
                },
            ),
            mock.patch.object(cachefix, "_fresh_task", return_value=controller),
            mock.patch.object(cachefix, "_read_pending_wal", return_value=wal),
            mock.patch.object(
                cachefix,
                "_journal_descriptor_state",
                return_value=_journal_state("candidate", candidate),
            ),
            mock.patch.object(
                cachefix, "_validate_v2_amendment", return_value=amendment
            ),
            mock.patch.object(
                cachefix, "_read_installed_barrier", return_value=barrier
            ),
            mock.patch.object(
                cachefix, "_install_barrier", side_effect=mutation_error
            ) as install,
            mock.patch.object(
                cachefix,
                "_recover_pending_candidate_once",
                side_effect=mutation_error,
            ) as recover,
            mock.patch.object(
                cachefix, "_upload_mapping_confirmed", side_effect=mutation_error
            ) as upload,
            mock.patch.object(
                cachefix, "_ensure_barrier_tag", side_effect=mutation_error
            ) as tag,
        ):
            receipt = cachefix.preflight(task_class=SimpleNamespace())

        self.assertTrue(receipt["readonly"])
        self.assertEqual(receipt["remote_mutation_count"], 0)
        self.assertEqual(receipt["pending_wal_status"], "pending")
        self.assertEqual(receipt["barrier_status"], "installed")
        install.assert_not_called()
        recover.assert_not_called()
        upload.assert_not_called()
        tag.assert_not_called()

    def test_preflight_cli_never_dispatches_execute(self) -> None:
        receipt = {
            "schema_version": 1,
            "document_type": cachefix.V3_PREFLIGHT_DOCUMENT_TYPE,
            "readonly": True,
            "remote_mutation_count": 0,
            "seal_sha256": "a" * 64,
        }
        output = io.StringIO()
        with (
            mock.patch.object(cachefix, "preflight", return_value=receipt) as read,
            mock.patch.object(
                cachefix,
                "execute_plan",
                side_effect=AssertionError("preflight dispatched execute"),
            ) as execute,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(cachefix.main(["--preflight"]), 0)

        self.assertEqual(json.loads(output.getvalue()), receipt)
        read.assert_called_once()
        execute.assert_not_called()

    def test_execute_requires_the_exact_v3_token(self) -> None:
        self.assertNotEqual(cachefix.V3_EXECUTION_TOKEN, legacy.EXECUTION_TOKEN)
        with mock.patch.object(cachefix, "_validate_control_host") as host:
            with self.assertRaisesRegex(
                cachefix.P1ParallelV3Error, "execution token mismatch"
            ):
                cachefix.execute_plan(
                    authorization_token=legacy.EXECUTION_TOKEN,
                    poll_seconds=30.0,
                    timeout_hours=72.0,
                    task_class=SimpleNamespace(),
                    api_client=SimpleNamespace(),
                )
        host.assert_not_called()

        with mock.patch.object(cachefix, "execute_plan") as execute:
            with self.assertRaisesRegex(
                cachefix.P1ParallelV3Error, "exact --execution-token"
            ):
                cachefix.main(
                    [
                        "--execute",
                        "--execution-token",
                        legacy.EXECUTION_TOKEN,
                    ]
                )
        execute.assert_not_called()

    def test_execute_returns_a_sealed_outer_receipt_without_mutating_manifest(
        self,
    ) -> None:
        task_class = SimpleNamespace(name="offline-task-class")
        api_client = SimpleNamespace(name="offline-api-client")
        manifest = {
            "schema_version": 1,
            "document_type": "resilient_v2x_p1_a100_execution_manifest",
            "status": "completed",
            "results": [{"task_id": "1" * 32}],
            "seal_sha256": "a" * 64,
        }
        before = copy.deepcopy(manifest)
        migration = {
            "barrier_seal_sha256": "b" * 64,
            "recovery": {
                "candidate_revision": 6,
                "upload_attempts": 0,
                "wal_removed": True,
            },
        }

        with (
            mock.patch.object(cachefix, "_validate_control_host"),
            mock.patch.object(
                cachefix,
                "_recover_and_install_barrier",
                return_value=migration,
            ),
            mock.patch.object(
                cachefix,
                "_authoritative_v2_runtime",
                return_value=contextlib.nullcontext(),
            ) as runtime,
            mock.patch.object(
                legacy, "execute_plan", return_value=manifest
            ) as legacy_execute,
        ):
            receipt = cachefix.execute_plan(
                authorization_token=cachefix.V3_EXECUTION_TOKEN,
                poll_seconds=5.0,
                timeout_hours=6.0,
                task_class=task_class,
                api_client=api_client,
            )

        self.assertEqual(manifest, before)
        self.assertEqual(
            set(receipt),
            {
                "schema_version",
                "document_type",
                "status",
                "v3_supervisor_sha256",
                "v2_supervisor_sha256",
                "barrier_seal_sha256",
                "recovery",
                "execution_manifest",
                "seal_sha256",
            },
        )
        self.assertEqual(
            receipt["document_type"],
            cachefix.V3_EXECUTION_RECEIPT_DOCUMENT_TYPE,
        )
        self.assertEqual(receipt["execution_manifest"], before)
        self.assertIsNot(receipt["execution_manifest"], manifest)
        self.assertEqual(receipt["barrier_seal_sha256"], "b" * 64)
        self.assertEqual(receipt["recovery"], migration["recovery"])
        self.assertEqual(receipt["seal_sha256"], base._seal(receipt))
        runtime.assert_called_once_with(task_class=task_class)
        legacy_execute.assert_called_once_with(
            authorization_token=legacy.EXECUTION_TOKEN,
            poll_seconds=5.0,
            timeout_hours=6.0,
            task_class=task_class,
            queue_reader=None,
            api_client=api_client,
            legacy_retirement_reader=None,
        )

    def test_exact_execute_token_dispatches_once(self) -> None:
        manifest = {
            "schema_version": 1,
            "document_type": "test_manifest",
            "seal_sha256": "a" * 64,
        }
        output = io.StringIO()
        with (
            mock.patch.object(
                cachefix, "execute_plan", return_value=manifest
            ) as execute,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(
                cachefix.main(
                    [
                        "--execute",
                        "--execution-token",
                        cachefix.V3_EXECUTION_TOKEN,
                        "--poll-seconds",
                        "5",
                        "--timeout-hours",
                        "6",
                    ]
                ),
                0,
            )

        self.assertEqual(json.loads(output.getvalue()), manifest)
        execute.assert_called_once_with(
            authorization_token=cachefix.V3_EXECUTION_TOKEN,
            poll_seconds=5.0,
            timeout_hours=6.0,
        )


if __name__ == "__main__":
    unittest.main()
