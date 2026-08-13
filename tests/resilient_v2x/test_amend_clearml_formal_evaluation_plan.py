from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import (
    amend_clearml_formal_evaluation_plan as amend,
)


def _base_plan(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    entries = []
    for index, subject in enumerate(amend.formal.FORMAL_SUBJECT_ORDER, start=1):
        task_id = f"{index:032x}"
        if subject == amend.FFNET_SUBJECT:
            task_id = amend.FFNET_EVALUATION_TASK_ID
        entries.append(
            {
                "subject": subject,
                "evaluation_task_id": task_id,
                "queue": (
                    amend.SOURCE_QUEUE
                    if subject == amend.FFNET_SUBJECT
                    else "GPU4-A100"
                ),
            }
        )
    plan = amend._seal(
        {
            "schema_version": 2,
            "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
            "training_controller_task_id": amend.TRAINING_CONTROLLER_TASK_ID,
            "training_provenance_task_id": amend.TRAINING_PROVENANCE_TASK_ID,
            "training_provenance_seal_sha256": "a" * 64,
            "source_revision_equivalence": {"fixture": True},
            "source_revision_equivalence_seal_sha256": "b" * 64,
            "source_revision_subject_map": {"fixture": True},
            "source_revision_subject_map_seal_sha256": "c" * 64,
            "evaluation_source_revision_tree_sha256": "d" * 64,
            "evaluation_source_revision": {"fixture": True},
            "evaluation_template_task_id": amend.EVALUATION_TEMPLATE_TASK_ID,
            "protocol_id": amend.formal.EXPECTED_PROTOCOL_ID,
            "sample_count": amend.formal.EXPECTED_SAMPLE_COUNT,
            "delays_ms": list(amend.formal.EXPECTED_DELAYS_MS),
            "conditions": list(amend.formal.EXPECTED_CONDITIONS),
            "run_count": 12,
            "subject_order": list(amend.formal.FORMAL_SUBJECT_ORDER),
            "entries": entries,
        }
    )
    monkeypatch.setattr(amend, "BASE_PLAN_SEAL_SHA256", plan["seal_sha256"])
    return plan


def _resource_snapshot(queue: str, *, ready: bool, worker: str) -> dict[str, object]:
    return {
        "ready": ready,
        "policy": "queued_target_queue_blocks_and_in_progress_occupies_physical_worker",
        "target_queue_id": amend.formal.EXPECTED_QUEUE_IDS[queue],
        "target_workers": [{"worker_id": worker, "resource": "host:gpu4,5,6,7"}],
        "idle_target_worker_ids": [worker] if ready else [],
        "selected_idle_target_worker_id": worker if ready else None,
        "occupied_target_worker_ids": [],
        "blocking_statuses": ["in_progress", "queued"],
        "observed_overlapping_active_tasks": [],
        "external_blockers": [],
    }


def _telemetry(worker_id: str, memory: float, usage: float) -> dict[str, object]:
    return {
        "worker_id": worker_id,
        "window_from_unix": 100.0,
        "window_to_unix": 280.0,
        "interval_seconds": amend.GPU_TELEMETRY_INTERVAL_SECONDS,
        "gpu_memory_used_mib": [memory, memory],
        "gpu_usage_percent": [usage, usage],
        "max_gpu_memory_used_mib": memory,
        "max_gpu_usage_percent": usage,
    }


def _worker_evidence() -> dict[str, object]:
    return amend._seal(
        {
            "captured_at_utc": "2026-08-12T08:00:00.000000Z",
            "formal_evaluation_task_ids": sorted(
                row[2] for row in amend.recovery.EXPECTED_RECOVERY_TASKS
            ),
            "gpu4_v100": _resource_snapshot(
                amend.SOURCE_QUEUE,
                ready=True,
                worker=amend.ORIGINAL_FFNET_WORKER_ID,
            ),
            "gpu4_a100": _resource_snapshot(
                amend.TARGET_QUEUE,
                ready=True,
                worker=amend.A100_TARGET_WORKER_ID,
            ),
            "gpu_telemetry": {
                amend.V100_WORKER_IDS[0]: _telemetry(
                    amend.V100_WORKER_IDS[0], 14_845.5, 38.3
                ),
                amend.V100_WORKER_IDS[1]: _telemetry(
                    amend.V100_WORKER_IDS[1], 282.0, 0.0
                ),
                amend.A100_TARGET_WORKER_ID: _telemetry(
                    amend.A100_TARGET_WORKER_ID, 518.0, 0.0
                ),
            },
            "gpu_telemetry_policy": {
                "source_workers": list(amend.V100_WORKER_IDS),
                "target_worker": amend.A100_TARGET_WORKER_ID,
                "window_seconds": amend.GPU_TELEMETRY_WINDOW_SECONDS,
                "interval_seconds": amend.GPU_TELEMETRY_INTERVAL_SECONDS,
                "v100_busy_worker": amend.ORIGINAL_FFNET_WORKER_ID,
                "v100_busy_min_memory_used_mib_exclusive": (
                    amend.GPU_MEMORY_IDLE_LIMIT_MIB
                ),
                "a100_idle_max_memory_used_mib_inclusive": (
                    amend.GPU_MEMORY_IDLE_LIMIT_MIB
                ),
                "a100_idle_max_usage_percent_inclusive": (
                    amend.GPU_USAGE_IDLE_LIMIT_PERCENT
                ),
            },
        }
    )


def _reseal(value: dict[str, object]) -> dict[str, object]:
    return amend._seal(value)


def test_amendment_changes_only_ffnet_queue_and_keeps_exact_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_plan(monkeypatch)
    revised = amend.amend_plan(base)
    change = amend.validate_one_field_amendment(base, revised)

    assert change["changed_field_count"] == 1
    assert change["changed_json_pointer"] == "/entries/10/queue"
    assert change["evaluation_task_id"] == amend.FFNET_EVALUATION_TASK_ID
    assert change["from_queue"] == "GPU4-V100"
    assert change["to_queue"] == "GPU4-A100"
    assert change["fixed_baselines"] == list(amend.FIXED_BASELINES)
    assert [row["evaluation_task_id"] for row in revised["entries"]] == [
        row["evaluation_task_id"] for row in base["entries"]
    ]


@pytest.mark.parametrize("drift", ["other_queue", "task_id", "protocol", "sample_count"])
def test_amendment_rejects_every_non_ffnet_queue_drift(
    monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    base = _base_plan(monkeypatch)
    revised = amend.amend_plan(base)
    malicious = copy.deepcopy(revised)
    if drift == "other_queue":
        malicious["entries"][0]["queue"] = "GPU4-V100"
    elif drift == "task_id":
        malicious["entries"][0]["evaluation_task_id"] = "f" * 32
    elif drift == "protocol":
        malicious["protocol_id"] = "DAIR-CLEAN-PAIR1789-v1"
    else:
        malicious["sample_count"] = 1789
    malicious = _reseal(malicious)

    with pytest.raises(amend.AmendmentError, match="more than FFNet"):
        amend.validate_one_field_amendment(base, malicious)


def test_base_plan_requires_fixed_five_and_exact_1337_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_plan(monkeypatch)
    bad = copy.deepcopy(base)
    bad["subject_order"][10] = "not_ffnet"
    bad = _reseal(bad)
    monkeypatch.setattr(amend, "BASE_PLAN_SEAL_SHA256", bad["seal_sha256"])
    with pytest.raises(amend.AmendmentError, match="subject_order"):
        amend.validate_base_plan(bad)


def test_worker_evidence_records_idle_v100_half_without_claiming_it_busy() -> None:
    evidence = amend.validate_worker_evidence(_worker_evidence())
    telemetry = evidence["gpu_telemetry"]
    assert telemetry[amend.V100_WORKER_IDS[0]]["max_gpu_memory_used_mib"] == 14_845.5
    assert telemetry[amend.V100_WORKER_IDS[1]]["max_gpu_memory_used_mib"] == 282.0
    assert telemetry[amend.A100_TARGET_WORKER_ID]["max_gpu_memory_used_mib"] == 518.0


def test_worker_evidence_fails_if_original_ffnet_worker_is_not_busy() -> None:
    evidence = _worker_evidence()
    evidence["gpu_telemetry"][amend.ORIGINAL_FFNET_WORKER_ID] = _telemetry(
        amend.ORIGINAL_FFNET_WORKER_ID, 282.0, 0.0
    )
    evidence = _reseal(evidence)
    with pytest.raises(amend.AmendmentError, match="original V100 worker"):
        amend.validate_worker_evidence(evidence)


@pytest.mark.parametrize(
    ("memory", "usage"),
    [(1024.1, 0.0), (518.0, 1.1)],
)
def test_worker_evidence_fails_if_a100_gpu4_7_is_not_idle(
    memory: float, usage: float
) -> None:
    evidence = _worker_evidence()
    evidence["gpu_telemetry"][amend.A100_TARGET_WORKER_ID] = _telemetry(
        amend.A100_TARGET_WORKER_ID, memory, usage
    )
    evidence = _reseal(evidence)
    with pytest.raises(amend.AmendmentError, match="A100 gpu4-7"):
        amend.validate_worker_evidence(evidence)


def test_existing_authority_receipts_and_ffnet_attempt_validate() -> None:
    recovery = amend.validate_recovery_receipt(amend.DEFAULT_RECOVERY_RECEIPT)
    deployment = amend.validate_deployment_receipt(amend.DEFAULT_DEPLOYMENT_RECEIPT)
    attempt = amend.validate_attempt_evidence(amend.DEFAULT_ATTEMPT_EVIDENCE)

    assert recovery["receipt_seal_sha256"] == amend.RECOVERY_RECEIPT_SEAL_SHA256
    assert deployment["receipt_seal_sha256"] == amend.DEPLOYMENT_RECEIPT_SEAL_SHA256
    assert attempt["last_worker"] == amend.ORIGINAL_FFNET_WORKER_ID
    assert attempt["task_id"] == amend.FFNET_EVALUATION_TASK_ID


def test_dry_run_is_local_only_and_has_exact_execute_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_plan(monkeypatch)
    revised = amend.amend_plan(base)
    assert amend.EXECUTE_TOKEN == "AMEND_FFNET_QUEUE_GPU4_V100_TO_GPU4_A100"
    assert revised["entries"][10] == {
        "subject": "ffnet",
        "evaluation_task_id": amend.FFNET_EVALUATION_TASK_ID,
        "queue": "GPU4-A100",
    }


class _Artifact:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def get(self, **_kwargs: object) -> dict[str, object]:
        return json.loads(json.dumps(self.value))


class _Producer:
    def __init__(
        self,
        *,
        source: str,
        plan: dict[str, object],
        evidence: dict[str, object],
        parameters: dict[str, str],
    ) -> None:
        self.id = "b" * 32
        self.name = amend.TASK_NAME
        self.status = "completed"
        self.tags = sorted(amend.TASK_TAGS)
        self.parent = amend.BASE_PLAN_PRODUCER_TASK_ID
        self.data = SimpleNamespace(
            parent=amend.BASE_PLAN_PRODUCER_TASK_ID,
            tags=sorted(amend.TASK_TAGS),
            script=SimpleNamespace(
                repository="",
                working_dir=".",
                entry_point=amend.ENTRY_POINT,
                diff=source,
            ),
        )
        self.artifacts = {
            amend.REVISED_PLAN_ARTIFACT: _Artifact(plan),
            amend.AMENDMENT_ARTIFACT: _Artifact(evidence),
        }
        self.parameters = parameters

    def reload(self) -> None:
        return None

    def get_project_name(self) -> str:
        return amend.PROJECT_NAME

    def get_parameters(self, **_kwargs: object) -> dict[str, str]:
        return dict(self.parameters)


def test_producer_authoritative_readback_checks_two_exact_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = amend.amend_plan(_base_plan(monkeypatch))
    evidence = amend._seal({"producer_task_id": "b" * 32})
    parameters = amend._producer_parameters(
        revised_plan_seal=str(plan["seal_sha256"]),
        worker_evidence_seal="c" * 64,
    )
    producer = _Producer(
        source="source", plan=plan, evidence=evidence, parameters=parameters
    )
    monkeypatch.setattr(amend.time, "sleep", lambda _seconds: None)
    observed = amend._producer_authoritative_readback(
        producer,
        source="source",
        revised_plan=plan,
        evidence=evidence,
        expected_parameters=parameters,
    )
    assert observed["authoritative_readback_count"] == 2
    assert observed["artifact_seals"] == {
        amend.REVISED_PLAN_ARTIFACT: plan["seal_sha256"],
        amend.AMENDMENT_ARTIFACT: evidence["seal_sha256"],
    }


def test_producer_authoritative_readback_rejects_unknown_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = amend.amend_plan(_base_plan(monkeypatch))
    evidence = amend._seal({"producer_task_id": "b" * 32})
    parameters = amend._producer_parameters(
        revised_plan_seal=str(plan["seal_sha256"]),
        worker_evidence_seal="c" * 64,
    )
    producer = _Producer(
        source="source", plan=plan, evidence=evidence, parameters=parameters
    )
    producer.tags.append("unknown-third-state")
    with pytest.raises(amend.AmendmentError, match="exact tags"):
        amend._producer_authoritative_readback(
            producer,
            source="source",
            revised_plan=plan,
            evidence=evidence,
            expected_parameters=parameters,
        )
