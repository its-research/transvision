from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import amend_clearml_formal_evaluation_plan as amendment
from tools.resilient_v2x import (
    clearml_formal_candidate_evaluation_queue as candidate_queue,
)
from tools.resilient_v2x import (
    recover_clearml_exact_eval_chain_runtime_failure as recovery,
)


def _change(*, revised_seal: str) -> dict[str, object]:
    return {
        "base_plan_seal_sha256": recovery.EXPECTED_W_PLAN_SEAL,
        "revised_plan_seal_sha256": revised_seal,
        "changed_json_pointer": (
            f"/entries/{recovery.formal.FORMAL_SUBJECT_ORDER.index('ffnet')}/queue"
        ),
        "subject": "ffnet",
        "evaluation_task_id": recovery.FFNET_TASK_ID,
        "from_queue": recovery.AMENDMENT_SOURCE_QUEUE,
        "to_queue": recovery.AMENDMENT_TARGET_QUEUE,
        "evaluation_task_ids_sha256": recovery._content_sha256(
            [
                row[2]
                for row in recovery.evaluation_recovery.EXPECTED_RECOVERY_TASKS[:26]
            ]
        ),
        "fixed_baselines": list(amendment.FIXED_BASELINES),
        "protocol_id": recovery.formal.EXPECTED_PROTOCOL_ID,
        "sample_count": recovery.formal.EXPECTED_SAMPLE_COUNT,
        "changed_field_count": 1,
    }


def _amendment_receipt(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    producer_id = "a" * 32
    source_sha = "b" * 64
    revised_seal = "c" * 64
    change = _change(revised_seal=revised_seal)
    attempt_binding = {
        "receipt_seal_sha256": amendment.ATTEMPT_EVIDENCE_SEAL_SHA256,
        "task_id": amendment.FFNET_EVALUATION_TASK_ID,
        "last_worker": amendment.ORIGINAL_FFNET_WORKER_ID,
    }
    recovery_binding = {
        "receipt_seal_sha256": recovery.EXPECTED_EVALUATION_RECOVERY_SEAL,
        "all_tasks_created_unqueued": True,
        "replacement_tasks_created": False,
    }
    deployment_binding = {
        "receipt_seal_sha256": recovery.EXPECTED_SUCCESSOR_DEPLOYMENT_SEAL,
        "task_ids": {"P": amendment.BASE_PLAN_PRODUCER_TASK_ID},
    }

    def telemetry(worker_id: str, memory: float) -> dict[str, object]:
        return {
            "worker_id": worker_id,
            "window_from_unix": 1.0,
            "window_to_unix": 2.0,
            "interval_seconds": amendment.GPU_TELEMETRY_INTERVAL_SECONDS,
            "gpu_memory_used_mib": [memory],
            "gpu_usage_percent": [0.0],
            "max_gpu_memory_used_mib": memory,
            "max_gpu_usage_percent": 0.0,
        }

    worker_evidence = amendment._seal(
        {
            "captured_at_utc": "2026-08-12T00:00:00Z",
            "formal_evaluation_task_ids": sorted(
                row[2] for row in recovery.evaluation_recovery.EXPECTED_RECOVERY_TASKS
            ),
            "gpu4_v100": {
                "target_queue_id": recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"]
            },
            "gpu4_a100": {
                "target_queue_id": recovery.formal.EXPECTED_QUEUE_IDS["GPU4-A100"],
                "ready": True,
                "selected_idle_target_worker_id": amendment.A100_TARGET_WORKER_ID,
            },
            "gpu_telemetry": {
                amendment.V100_WORKER_IDS[0]: telemetry(
                    amendment.V100_WORKER_IDS[0], 2048.0
                ),
                amendment.V100_WORKER_IDS[1]: telemetry(
                    amendment.V100_WORKER_IDS[1], 0.0
                ),
                amendment.A100_TARGET_WORKER_ID: telemetry(
                    amendment.A100_TARGET_WORKER_ID, 0.0
                ),
            },
            "gpu_telemetry_policy": {
                "source_workers": list(amendment.V100_WORKER_IDS),
                "target_worker": amendment.A100_TARGET_WORKER_ID,
                "window_seconds": amendment.GPU_TELEMETRY_WINDOW_SECONDS,
                "interval_seconds": amendment.GPU_TELEMETRY_INTERVAL_SECONDS,
                "v100_busy_worker": amendment.ORIGINAL_FFNET_WORKER_ID,
                "v100_busy_min_memory_used_mib_exclusive": (
                    amendment.GPU_MEMORY_IDLE_LIMIT_MIB
                ),
                "a100_idle_max_memory_used_mib_inclusive": (
                    amendment.GPU_MEMORY_IDLE_LIMIT_MIB
                ),
                "a100_idle_max_usage_percent_inclusive": (
                    amendment.GPU_USAGE_IDLE_LIMIT_PERCENT
                ),
            },
        }
    )
    evidence = amendment.build_amendment_evidence(
        change=change,
        attempt_binding=attempt_binding,
        recovery_binding=recovery_binding,
        deployment_binding=deployment_binding,
        worker_evidence=worker_evidence,
        producer_task_id=producer_id,
    )
    producer_parameters = amendment._producer_parameters(
        revised_plan_seal=revised_seal,
        worker_evidence_seal=worker_evidence["seal_sha256"],
    )
    receipt = recovery._seal(
        {
            "schema_version": 1,
            "receipt_type": recovery.AMENDMENT_RECEIPT_TYPE,
            "generated_at_utc": "2026-08-12T00:00:00Z",
            "mode": "execute",
            "status": "completed",
            "remote_state_changed": True,
            "execute_token": amendment.EXECUTE_TOKEN,
            "base_plan_producer_task_id": amendment.BASE_PLAN_PRODUCER_TASK_ID,
            "base_plan_seal_sha256": recovery.EXPECTED_W_PLAN_SEAL,
            "revised_plan_seal_sha256": revised_seal,
            "change": change,
            "ffnet_attempt_evidence": attempt_binding,
            "recovery_receipt": recovery_binding,
            "deployment_receipt": deployment_binding,
            "worker_stats_evidence": worker_evidence,
            "staging_amendment_evidence": {"staging": True},
            "producer_source_sha256": source_sha,
            "remote_write_contract": {"producer_only": True},
            "attempt_receipt_seal_sha256": "d" * 64,
            "producer": {
                "task_id": producer_id,
                "status": "completed",
                "name": amendment.TASK_NAME,
                "project": amendment.PROJECT_NAME,
                "source_sha256": source_sha,
                "parent_task_id": amendment.BASE_PLAN_PRODUCER_TASK_ID,
                "parameters": producer_parameters,
                "tags": sorted(amendment.TASK_TAGS),
                "artifact_seals": {
                    amendment.REVISED_PLAN_ARTIFACT: revised_seal,
                    amendment.AMENDMENT_ARTIFACT: evidence["seal_sha256"],
                },
                "authoritative_readback_count": 2,
            },
            "amendment_evidence": evidence,
        }
    )
    path = tmp_path / "amendment.json"
    recovery._write_new(path, receipt)
    return path, receipt


def test_amendment_receipt_accepts_dynamic_producer_and_seal(tmp_path: Path) -> None:
    path, receipt = _amendment_receipt(tmp_path)
    binding, observed = recovery._amendment_binding(path)
    assert observed == receipt
    assert binding["receipt_seal_sha256"] == receipt["seal_sha256"]
    assert binding["producer_task_id"] == "a" * 32
    assert binding["producer_source_sha256"] == "b" * 64
    assert binding["revised_plan_seal_sha256"] == "c" * 64
    assert binding["worker_stats_evidence_seal_sha256"] == receipt[
        "worker_stats_evidence"
    ]["seal_sha256"]
    assert binding["change"]["to_queue"] == "GPU4-A100"


def test_amendment_receipt_rejects_any_second_change(tmp_path: Path) -> None:
    _, receipt = _amendment_receipt(tmp_path)
    receipt["change"]["changed_field_count"] = 2
    receipt = recovery._seal(receipt)
    path = tmp_path / "tampered.json"
    recovery._write_new(path, receipt)
    with pytest.raises(recovery.RuntimeRecoveryError, match="receipt is invalid"):
        recovery._amendment_binding(path)


def test_amendment_receipt_rejects_wrong_upstream_seal(tmp_path: Path) -> None:
    _, receipt = _amendment_receipt(tmp_path)
    receipt["recovery_receipt"]["receipt_seal_sha256"] = "0" * 64
    receipt = recovery._seal(receipt)
    path = tmp_path / "wrong-upstream.json"
    recovery._write_new(path, receipt)
    with pytest.raises(recovery.RuntimeRecoveryError, match="receipt is invalid"):
        recovery._amendment_binding(path)


def _binding(tmp_path: Path) -> dict[str, object]:
    path, _ = _amendment_receipt(tmp_path)
    binding, _ = recovery._amendment_binding(path)
    return binding


def test_candidate_source_patch_is_exact_compilable_and_amendment_aware(
    tmp_path: Path,
) -> None:
    source = Path(candidate_queue.__file__).read_text(encoding="utf-8")
    patched = recovery._candidate_amendment_source(
        source, amendment_binding=_binding(tmp_path)
    )
    compile(patched, "patched-candidate.py", "exec")
    old_ffnet = '''    {
        "subject": "ffnet",
        "task_id": "144397bfa9c242bc9a92a1279922558b",
        "index": 11,
        "queue": "GPU4-V100",
    },'''
    new_ffnet = old_ffnet.replace('"queue": "GPU4-V100"', '"queue": "GPU4-A100"')
    assert old_ffnet not in patched
    assert patched.count(new_ffnet) == 1
    assert '"queue": "GPU4-V100"' not in patched.split(
        '"subject": "ffnet"', 1
    )[1].split("},", 1)[0]
    assert "def _validate_formal_plan_amendment" in patched
    assert "formal_plan_amendment = _validate_formal_plan_amendment" in patched
    assert "FORMAL_PLAN_AMENDMENT_RECEIPT_SEAL_SHA256" in patched
    with pytest.raises(recovery.RuntimeRecoveryError, match="queue anchor drifted"):
        recovery._candidate_amendment_source(
            patched, amendment_binding=_binding(tmp_path / "second")
        )


def test_candidate_target_hashes_are_generated_not_old_hardcoded(
    tmp_path: Path,
) -> None:
    source = Path(candidate_queue.__file__).read_text(encoding="utf-8")
    pre = {
        "label": "C",
        "task_id": recovery.CANDIDATE_CONTROLLER_TASK_ID,
        "source": source,
        "source_sha256": recovery.EXPECTED_SOURCE_SHA256["C"],
        "source_bytes": len(source.encode()),
        "parameters": {
            "Args/execute": "True",
            "Deployment/source_sha256": recovery.EXPECTED_SOURCE_SHA256["C"],
        },
        "parameters_sha256": recovery.EXPECTED_PARAMETERS_SHA256["C"],
        "tags": sorted(recovery.candidate_deploy.TAGS),
        "queue": "services",
        "queue_id": recovery.candidate_deploy.SERVICES_QUEUE_ID,
    }
    binding = _binding(tmp_path)
    target = recovery._candidate_target_material(
        pre_reset=pre, amendment_binding=binding
    )
    assert target["source_sha256"] != recovery.EXPECTED_SOURCE_SHA256["C"]
    assert target["parameters_sha256"] != recovery.EXPECTED_PARAMETERS_SHA256["C"]
    assert target["pre_reset_source_sha256"] == recovery.EXPECTED_SOURCE_SHA256["C"]
    assert target["formal_plan_amendment"] == binding
    assert target["parameters"][
        "Args/formal_plan_amendment_producer_task_id"
    ] == binding["producer_task_id"]
    assert target["parameters"][
        "Args/revised_formal_plan_seal_sha256"
    ] == binding["revised_plan_seal_sha256"]
    assert target["parameters"]["Deployment/schema_version"] == "2"
    assert recovery.AMENDED_CANDIDATE_TAG in target["tags"]


class _ConsoleTask:
    status = "failed"

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.data = SimpleNamespace(status_reason="worker execution exit code 1")

    def get_reported_console_output(self, _limit: int) -> list[str]:
        return self.lines


def test_failure_reason_must_be_exact_and_unique() -> None:
    task = _ConsoleTask(["Traceback", "RuntimeError: assigned GPUs are not idle"])
    evidence = recovery._console_failure(task, label="FFNet")
    assert evidence["runtime_error"] == "assigned GPUs are not idle"
    task.lines.append("RuntimeError: cuda out of memory")
    with pytest.raises(recovery.RuntimeRecoveryError, match="failure reason"):
        recovery._console_failure(task, label="FFNet")


def test_write_once_receipt_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    recovery._write_new(path, {"value": 1})
    with pytest.raises(FileExistsError):
        recovery._write_new(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_attempt_key_order_survives_sorted_json_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt = recovery._seal(_orchestration_attempt())
    path = tmp_path / "attempt.json"
    recovery._write_new(path, attempt)
    observed, _ = recovery._read_sealed(path, context="attempt round trip")
    assert tuple(observed["target_material"]) == ("C", "FFNet")
    monkeypatch.setattr(recovery, "_validate_attempt_structure", lambda _v: None)
    assert observed == attempt


def test_execute_rejects_attempt_not_durably_frozen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt = recovery._seal(_orchestration_attempt())
    different = recovery._seal({**attempt, "generated_at_utc": "different"})
    path = tmp_path / "attempt.json"
    recovery._write_new(path, different)
    monkeypatch.setattr(recovery, "_validate_attempt_structure", lambda _v: None)
    with pytest.raises(recovery.RuntimeRecoveryError, match="durably frozen"):
        recovery.execute_recovery(
            object(),
            attempt_receipt=attempt,
            attempt_receipt_path=path,
            sleeper=lambda _seconds: None,
        )


def test_exact_name_inventory_rejects_unknown_duplicate_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = SimpleNamespace(id=recovery.FFNET_TASK_ID, name="same")
    duplicate = SimpleNamespace(id="f" * 32, name="same")

    class Tasks:
        @staticmethod
        def get_tasks(**_kwargs: object) -> list[object]:
            return [expected, duplicate]

    monkeypatch.setattr(
        recovery,
        "_authoritative_task",
        lambda _tc, task_id, **_kw: expected if task_id == expected.id else duplicate,
    )
    monkeypatch.setattr(
        recovery.successor_deploy,
        "_task_parent",
        lambda _task: "p" * 32,
    )
    ids = recovery._exact_name_inventory(
        Tasks,
        task_name="same",
        parent_task_id="p" * 32,
        expected_task_id=recovery.FFNET_TASK_ID,
    )
    assert ids == sorted([recovery.FFNET_TASK_ID, "f" * 32])


def test_wrong_execute_token_fails_before_receipt_or_clearml_read(tmp_path: Path) -> None:
    with pytest.raises(recovery.RuntimeRecoveryError, match="exact execute token"):
        recovery.main(
            [
                "--execute",
                "--execute-token",
                "wrong",
                "--evaluation-recovery-receipt",
                str(tmp_path / "evaluation.json"),
                "--successor-deployment-receipt",
                str(tmp_path / "successor.json"),
                "--controller-recovery-receipt",
                str(tmp_path / "controller.json"),
                "--candidate-deployment-receipt",
                str(tmp_path / "candidate.json"),
                "--amendment-receipt",
                str(tmp_path / "amendment.json"),
                "--attempt-receipt",
                str(tmp_path / "attempt.json"),
                "--receipt",
                str(tmp_path / "final.json"),
            ]
        )


class _ResetTask:
    def __init__(self, *, mode: str) -> None:
        self.status = "failed"
        self.mode = mode
        self.id = recovery.FFNET_TASK_ID

    def reset(self, *, force: bool) -> None:
        assert force is True
        if self.mode != "never":
            self.status = "created"
        if self.mode == "raise_after":
            raise RuntimeError("callback failed after acceptance")


def test_reset_callback_exception_uses_authoritative_created_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _ResetTask(mode="raise_after")
    monkeypatch.setattr(recovery, "_authoritative_task", lambda *_a, **_k: task)
    monkeypatch.setattr(
        recovery, "_validate_failed_target_against_material", lambda *_a, **_k: task
    )
    monkeypatch.setattr(
        recovery, "_created_unqueued_write_guard", lambda *_a, **_k: task
    )
    monkeypatch.setattr(
        recovery,
        "_created_shell_snapshot",
        lambda *_a, **_k: (task, {"frozen": True}),
    )
    monkeypatch.setattr(recovery, "_artifact_names", lambda _task: set())
    monkeypatch.setattr(
        recovery,
        "_model_inventory",
        lambda _task, *, context: {"input": [], "output": []},
    )
    monkeypatch.setattr(recovery, "_configure_target", lambda *_a, **_k: None)
    expected = {"task_id": task.id, "status": "created"}
    monkeypatch.setattr(
        recovery,
        "_validate_created_target",
        lambda *_a, **_k: expected,
    )
    observed = recovery._reset_and_install(
        object(),
        label="FFNet",
        material={"label": "FFNet"},
        sleeper=lambda _seconds: None,
    )
    assert observed == expected


def test_reset_callback_without_authoritative_acceptance_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _ResetTask(mode="never")
    monkeypatch.setattr(recovery, "_authoritative_task", lambda *_a, **_k: task)
    monkeypatch.setattr(
        recovery, "_validate_failed_target_against_material", lambda *_a, **_k: task
    )
    with pytest.raises(recovery.RuntimeRecoveryError, match="did not reach created"):
        recovery._reset_and_install(
            object(),
            label="FFNet",
            material={"label": "FFNet"},
            sleeper=lambda _seconds: None,
        )


def test_enqueue_callback_exception_uses_authoritative_queue_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = SimpleNamespace(id=recovery.CANDIDATE_CONTROLLER_TASK_ID)

    class Tasks:
        @staticmethod
        def enqueue(*, task: object, queue_name: str) -> None:
            assert queue_name == "services"
            task.accepted = True
            raise RuntimeError("callback failed after acceptance")

    monkeypatch.setattr(recovery, "_authoritative_task", lambda *_a, **_k: task)
    created = {"task_id": task.id, "status": "created"}
    monkeypatch.setattr(
        recovery, "_validate_created_target", lambda *_a, **_k: created
    )
    monkeypatch.setattr(
        recovery, "_created_unqueued_write_guard", lambda *_a, **_k: task
    )

    def validate(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert task.accepted is True
        return {"task_id": task.id, "status": "queued", "queue": "services"}

    monkeypatch.setattr(recovery, "_validate_queued_target", validate)
    result = recovery._enqueue_with_authority(
        Tasks,
        label="C",
        material={"queue": "services"},
        sleeper=lambda _seconds: None,
    )
    assert result["status"] == "queued"


def _orchestration_attempt() -> dict[str, object]:
    failure = {
        "content_stored": False,
        "tail_line_count": 1,
        "tail_bytes": 1,
        "tail_sha256": "1" * 64,
        "runtime_error": "frozen",
        "status_reason": "worker execution exit code 1",
    }
    ffnet = {
        "label": "FFNet",
        "task_id": recovery.FFNET_TASK_ID,
        "task_name": "ffnet-task",
        "parent_task_id": "f" * 32,
        "entry_point": "clearml_5090_bootstrap.py",
        "source": "ffnet-source",
        "source_sha256": recovery.hashlib.sha256(b"ffnet-source").hexdigest(),
        "parameters": {"p": "ffnet"},
        "parameters_sha256": recovery._content_sha256({"p": "ffnet"}),
        "tags": ["sealed"],
        "queue": "GPU4-A100",
        "queue_id": recovery.formal.EXPECTED_QUEUE_IDS["GPU4-A100"],
        "pre_reset_queue": "GPU4-V100",
        "pre_reset_queue_id": recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        "pre_reset_tags": ["dependency-released", "sealed"],
        "artifact_names": [],
        "failure_evidence": failure,
    }
    candidate = {
        "label": "C",
        "task_id": recovery.CANDIDATE_CONTROLLER_TASK_ID,
        "task_name": "candidate-task",
        "parent_task_id": "c" * 32,
        "entry_point": "clearml_formal_candidate_evaluation_queue.py",
        "source": "new-candidate-source",
        "source_sha256": recovery.hashlib.sha256(
            b"new-candidate-source"
        ).hexdigest(),
        "pre_reset_source_sha256": recovery.hashlib.sha256(
            b"old-candidate-source"
        ).hexdigest(),
        "parameters": {"p": "new"},
        "parameters_sha256": recovery._content_sha256({"p": "new"}),
        "pre_reset_parameters_sha256": recovery._content_sha256({"p": "old"}),
        "tags": ["amended"],
        "pre_reset_tags": ["old"],
        "queue": "services",
        "queue_id": recovery.successor_deploy.SERVICES_QUEUE_ID,
        "pre_reset_queue": "services",
        "pre_reset_queue_id": recovery.successor_deploy.SERVICES_QUEUE_ID,
        "artifact_names": [],
        "failure_evidence": failure,
    }
    return {
        "seal_sha256": "2" * 64,
        "bindings": {"formal_plan_amendment": {"receipt_seal_sha256": "3" * 64}},
        "amendment_producer_authority": {"task_id": "a" * 32},
        "old_successor_failure_evidence": {role: {} for role in ("W", "L", "A", "S")},
        "target_material": {"FFNet": ffnet, "C": candidate},
        "exact_name_inventories": {
            "FFNet": [recovery.FFNET_TASK_ID],
            "C": [recovery.CANDIDATE_CONTROLLER_TASK_ID],
        },
        "preexisting_same_name_tasks": {"FFNet": [], "C": []},
        "evaluation_inventory": {"records": []},
    }


def test_execute_mutates_only_two_exact_ids_and_enqueues_only_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = recovery._seal(_orchestration_attempt())
    tasks = {
        recovery.FFNET_TASK_ID: SimpleNamespace(
            id=recovery.FFNET_TASK_ID,
            status="failed",
            source="ffnet-source",
            parameters={"p": "ffnet"},
            tags=["dependency-released", "sealed"],
            queue="GPU4-V100",
            queue_id=recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        ),
        recovery.CANDIDATE_CONTROLLER_TASK_ID: SimpleNamespace(
            id=recovery.CANDIDATE_CONTROLLER_TASK_ID,
            status="failed",
            source="old-candidate-source",
            parameters={"p": "old"},
            tags=["old"],
            queue="services",
            queue_id=recovery.successor_deploy.SERVICES_QUEUE_ID,
        ),
    }
    monkeypatch.setattr(recovery, "_validate_attempt_structure", lambda _v: None)
    monkeypatch.setattr(
        recovery,
        "_exact_name_inventory",
        lambda _tc, *, expected_task_id, **_kw: [expected_task_id],
    )
    monkeypatch.setattr(
        recovery,
        "_authoritative_task",
        lambda _tc, task_id, **_kw: tasks.get(task_id, SimpleNamespace(id=task_id)),
    )
    monkeypatch.setattr(recovery, "_validate_w_plan", lambda _task: {"entries": []})
    monkeypatch.setattr(
        recovery,
        "_validate_remote_amendment_producer",
        lambda *_a, **_kw: {"task_id": "a" * 32},
    )
    frozen_old = {role: {"task_id": role} for role in ("W", "L", "A", "S")}
    monkeypatch.setattr(
        recovery,
        "_revalidate_old_successor_failures",
        lambda *_a, **_kw: frozen_old,
    )
    preserved = {
        "task_count": 27,
        "coformer": {"task_id": recovery.COFORMER_TASK_ID},
        "no_preserved_task_mutated_by_recovery": True,
    }
    monkeypatch.setattr(
        recovery,
        "_revalidate_preserved_evaluations",
        lambda *_a, **_kw: preserved,
    )
    monkeypatch.setattr(
        recovery,
        "_revalidate_write_dependency_barrier",
        lambda *_a, **_kw: {"phase": _kw["phase"]},
    )
    monkeypatch.setattr(recovery, "_status", lambda task: task.status)
    monkeypatch.setattr(recovery, "_artifact_names", lambda _task: set())
    monkeypatch.setattr(
        recovery,
        "_console_failure",
        lambda _task, *, label: attempt["target_material"][label][
            "failure_evidence"
        ],
    )
    monkeypatch.setattr(
        recovery,
        "_source",
        lambda task, **_kw: task.source,
    )
    monkeypatch.setattr(
        recovery,
        "_parameters",
        lambda task, **_kw: task.parameters,
    )
    monkeypatch.setattr(
        recovery,
        "_tags",
        lambda task, **_kw: list(task.tags),
    )
    monkeypatch.setattr(
        recovery,
        "_queue_identity",
        lambda task, **_kw: (task.queue, task.queue_id),
    )
    monkeypatch.setattr(
        recovery,
        "_model_inventory",
        lambda _task, **_kw: {"input": [], "output": []},
    )
    resets: list[str] = []
    enqueues: list[str] = []

    def reset(*_args: object, label: str, **_kwargs: object) -> dict[str, object]:
        resets.append(label)
        return {
            "label": label,
            "task_id": recovery.TARGET_TASK_IDS[label],
            "status": "created",
            "queue": None,
            "queue_id": None,
        }

    def enqueue(*_args: object, label: str, **_kwargs: object) -> dict[str, object]:
        enqueues.append(label)
        return {
            "label": label,
            "task_id": recovery.TARGET_TASK_IDS[label],
            "status": "queued",
            "queue": "services",
            "queue_id": recovery.successor_deploy.SERVICES_QUEUE_ID,
        }

    monkeypatch.setattr(recovery, "_reset_and_install", reset)
    monkeypatch.setattr(recovery, "_enqueue_with_authority", enqueue)
    monkeypatch.setattr(
        recovery,
        "_validate_created_target",
        lambda *_a, **_kw: {
            "label": "FFNet",
            "task_id": recovery.FFNET_TASK_ID,
            "status": "created",
            "queue": None,
            "queue_id": None,
        },
    )
    monkeypatch.setattr(
        recovery,
        "_validate_queued_target",
        lambda *_a, **_kw: {
            "label": "C",
            "task_id": recovery.CANDIDATE_CONTROLLER_TASK_ID,
            "status": "queued",
            "queue": "services",
            "queue_id": recovery.successor_deploy.SERVICES_QUEUE_ID,
        },
    )
    monkeypatch.setattr(
        recovery,
        "_candidate_authority_transition_valid",
        lambda *_a, **_kw: True,
    )
    attempt_path = tmp_path / "attempt.json"
    recovery._write_new(attempt_path, attempt)
    receipt = recovery.execute_recovery(
        object(),
        attempt_receipt=attempt,
        attempt_receipt_path=attempt_path,
        sleeper=lambda _seconds: None,
    )
    assert resets == ["FFNet", "C"]
    assert enqueues == ["C"]
    assert receipt["ffnet_final_authority"]["status"] == "created"
    assert receipt["queued_authority"]["C"]["status"] == "queued"
    assert receipt["old_successors_preserved"] == frozen_old
    assert receipt["coformer_mutated"] is False
    assert receipt["replacement_tasks_created"] is False



def test_service_requirements_reject_created_materialized_and_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = "\n".join(recovery.successor_deploy.SERVICE_REQUIREMENTS)
    task = SimpleNamespace(status="created")
    monkeypatch.setattr(
        recovery.successor_deploy,
        "_raw_requirements",
        lambda _task: {"pip": canonical},
    )
    observed = recovery._service_requirements_valid(task, context="service")
    assert observed["form"] == "canonical"
    assert observed["canonical_requirement"] == canonical

    monkeypatch.setattr(
        recovery.successor_deploy,
        "_raw_requirements",
        lambda _task: {"org_pip": canonical, "pip": [canonical]},
    )
    with pytest.raises(recovery.RuntimeRecoveryError, match="while created"):
        recovery._service_requirements_valid(task, context="service")

    task.status = "in_progress"
    monkeypatch.setattr(
        recovery.successor_deploy,
        "_raw_requirements",
        lambda _task: {
            "org_pip": canonical,
            "pip": [canonical, canonical],
        },
    )
    with pytest.raises(recovery.RuntimeRecoveryError, match="drifted"):
        recovery._service_requirements_valid(task, context="service")


def test_configure_race_fails_before_write(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _ResetTask(mode="raise_after")
    monkeypatch.setattr(
        recovery, "_validate_failed_target_against_material", lambda *_a, **_k: task
    )
    monkeypatch.setattr(recovery, "_wait_created", lambda *_a, **_k: task)
    snapshots = iter(
        [
            (task, {"source_sha256": "a" * 64}),
            (task, {"source_sha256": "b" * 64}),
        ]
    )
    monkeypatch.setattr(
        recovery, "_created_shell_snapshot", lambda *_a, **_k: next(snapshots)
    )
    writes: list[bool] = []
    monkeypatch.setattr(
        recovery, "_configure_target", lambda *_a, **_k: writes.append(True)
    )
    with pytest.raises(recovery.RuntimeRecoveryError, match="changed before configure"):
        recovery._reset_and_install(
            object(),
            label="FFNet",
            material={"label": "FFNet"},
            sleeper=lambda _seconds: None,
        )
    assert writes == []


def _completed_manifest_fixture() -> tuple[dict[str, object], dict[str, object]]:
    binding = {
        "producer_task_id": "a" * 32,
        "receipt_seal_sha256": "b" * 64,
        "amendment_evidence_seal_sha256": "c" * 64,
        "worker_stats_evidence_seal_sha256": "d" * 64,
        "evaluation_task_ids_sha256": "e" * 64,
        "revised_plan_seal_sha256": "f" * 64,
    }
    material: dict[str, object] = {"formal_plan_amendment": binding}
    contract = recovery._completed_candidate_manifest_contract(material)
    entries = [
        {
            "subject": spec.subject,
            "evidence_status": "validated",
            "training_status": "completed",
            "training_final": {"checkpoint": "final"},
            "formal_evidence": {"runs": [{}] * candidate_queue.RUN_COUNT},
        }
        for spec in candidate_queue.CANDIDATES
    ]
    core_entries = [
        {
            "subject": spec["subject"],
            "task_id": spec["task_id"],
            "status": "completed",
            "queue": (
                recovery.AMENDMENT_TARGET_QUEUE
                if spec["subject"] == "ffnet"
                else spec["queue"]
            ),
            "execution_queue_id": recovery.formal.EXPECTED_QUEUE_IDS[
                recovery.AMENDMENT_TARGET_QUEUE
                if spec["subject"] == "ffnet"
                else spec["queue"]
            ],
        }
        for spec in candidate_queue.FORMAL_CORE_EVALUATIONS
    ]
    manifest = candidate_queue._sealed(
        {
            **{key: value for key, value in contract.items() if key != "formal_plan_amendment"},
            "training_release_barrier": {
                "policy": "all_five_completed_and_final_contract_verified",
                "ready": True,
                "required_labels": [spec.label for spec in candidate_queue.CANDIDATES],
            },
            "formal_core_release_barrier": {
                "policy": "formal_core_six_completed_before_candidate_e1",
                "ready": True,
                "entries": core_entries,
                "formal_plan_amendment": contract["formal_plan_amendment"],
            },
            "entries": entries,
            "sota_selector_input": {},
        }
    )
    return material, manifest


def test_completed_candidate_manifest_validates_protocol_and_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material, manifest = _completed_manifest_fixture()
    monkeypatch.setattr(
        candidate_queue,
        "_artifact_record",
        lambda *_a, **_k: {"hash": "1" * 64, "content_size": 1234},
    )
    monkeypatch.setattr(
        candidate_queue, "_artifact_payload", lambda *_a, **_k: manifest
    )
    evidence = recovery._validate_completed_candidate_manifest(
        object(), material=material
    )
    assert evidence["all_candidates_validated"] is True
    assert evidence["sample_count"] == 1337
    assert evidence["ground_truth_count"] == 11330
    assert evidence["unsupported_sample_count"] == 0

    bad = candidate_queue._sealed({**manifest, "unsupported_sample_count": 1})
    monkeypatch.setattr(candidate_queue, "_artifact_payload", lambda *_a, **_k: bad)
    with pytest.raises(recovery.RuntimeRecoveryError, match="protocol identity"):
        recovery._validate_completed_candidate_manifest(object(), material=material)


def test_artifact_inventory_monotonicity_rejects_replacement() -> None:
    frozen = [{"name": "metrics", "hash": "a" * 64, "size": 10}]
    added = [
        *frozen,
        {"name": "prediction_evidence", "hash": "b" * 64, "size": 20},
    ]
    assert recovery._artifact_inventory_is_monotonic(
        frozen,
        added,
        allowed_names=frozenset({"metrics", "prediction_evidence"}),
    )
    replaced = [{"name": "metrics", "hash": "c" * 64, "size": 10}]
    assert not recovery._artifact_inventory_is_monotonic(
        frozen,
        replaced,
        allowed_names=frozenset({"metrics", "prediction_evidence"}),
    )


def test_exact_project_id_rejects_partial_or_wrong_identity() -> None:
    expected = recovery.evaluation_recovery.PROJECT_ID
    task = SimpleNamespace(
        project=expected,
        data=SimpleNamespace(project=expected),
        _data=SimpleNamespace(project=expected),
    )
    assert recovery._exact_project_id(task, context="task") == expected
    task.data.project = "0" * 32
    with pytest.raises(recovery.RuntimeRecoveryError, match="project ID"):
        recovery._exact_project_id(task, context="task")



def test_candidate_authority_allows_only_monotonic_status_transition() -> None:
    base = {
        "label": "C",
        "task_id": recovery.CANDIDATE_CONTROLLER_TASK_ID,
        "project_id": recovery.evaluation_recovery.PROJECT_ID,
        "task_name": "candidate",
        "task_type": "controller",
        "queue": "services",
        "queue_id": recovery.successor_deploy.SERVICES_QUEUE_ID,
        "source_sha256": "a" * 64,
        "parameters_sha256": "b" * 64,
        "parent_task_id": "c" * 32,
        "model_inventory": {"input": [], "output": []},
        "tags": ["sealed"],
        "system_tags": [],
        "service_environment": {"policy": recovery.SERVICE_ENVIRONMENT_POLICY},
        "completion_evidence": None,
    }
    queued = {**base, "status": "queued", "artifact_names": []}
    running = {
        **base,
        "status": "in_progress",
        "artifact_names": [candidate_queue.CONTROLLER_ARTIFACT],
    }
    assert recovery._candidate_authority_transition_valid(queued, running)
    assert not recovery._candidate_authority_transition_valid(running, queued)
    drifted = {**running, "parent_task_id": "d" * 32}
    assert not recovery._candidate_authority_transition_valid(queued, drifted)
