from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import recover_clearml_formal_evaluation_attempts as recovery


class Model:
    def __init__(self, model_id: str, name: str = "model") -> None:
        self.id = model_id
        self.name = name
        self.task = "a" * 32


class Task:
    def __init__(
        self,
        task_id: str,
        *,
        status: str,
        source: str,
        parameters: dict[str, object],
        queue_id: str,
        input_models: list[Model],
    ) -> None:
        self.id = task_id
        self.name = "evaluation"
        self.project = recovery.PROJECT_ID
        self.status = status
        self.tags = ["dependency-released"]
        self.parameters = dict(parameters)
        self.models = {"input": list(input_models), "output": []}
        self.artifacts = {}
        self.data = SimpleNamespace(
            id=task_id,
            project=recovery.PROJECT_ID,
            parent="b" * 32,
            tags=list(self.tags),
            execution=SimpleNamespace(queue=queue_id),
            last_worker="worker",
            status_reason="CUDA out of memory",
            status_message="failed",
            created="created",
            started="started",
            completed=None,
            published=None,
            script=SimpleNamespace(
                repository="",
                working_dir=".",
                entry_point="clearml_5090_bootstrap.py",
                diff=source,
            ),
        )
        self.actions: list[str] = []

    def reload(self) -> None:
        return None

    def get_parameters(self, **_kwargs: object) -> dict[str, object]:
        return dict(self.parameters)

    def get_models(self) -> dict[str, list[Model]]:
        return self.models

    def get_reported_console_output(self, _count: int) -> list[str]:
        return ["line 1", "CUDA out of memory"]

    def mark_stopped(self, **_kwargs: object) -> None:
        self.actions.append("stop")
        self.status = "stopped"

    def reset(self, *, force: bool) -> None:
        assert force is True
        self.actions.append("reset")
        self.status = "created"
        self.data.execution.queue = ""

    def remove_input_models(self, ids: list[str]) -> None:
        self.actions.append("remove_inputs")
        self.models["input"] = [model for model in self.models["input"] if model.id not in ids]

    def set_input_model(self, model_id: str, name: str, **_kwargs: object) -> None:
        self.actions.append("set_input")
        self.models["input"] = [Model(model_id, name)]

    def set_script(self, **kwargs: object) -> None:
        self.actions.append("set_script")
        self.data.script.repository = str(kwargs["repository"])
        self.data.script.working_dir = str(kwargs["working_dir"])
        self.data.script.entry_point = str(kwargs["entry_point"])
        self.data.script.diff = str(kwargs["diff"])

    def set_parameters(self, value: dict[str, object]) -> None:
        self.actions.append("set_parameters")
        self.parameters = dict(value)

    def set_tags(self, values: list[str]) -> bool:
        self.actions.append("set_tags")
        self.tags = list(values)
        self.data.tags = list(values)
        return True

    def flush(self, *, wait_for_uploads: bool) -> bool:
        assert wait_for_uploads is True
        return True


class Tasks:
    task: Task

    @classmethod
    def get_task(cls, *, task_id: str) -> Task:
        assert task_id == cls.task.id
        return cls.task

    @classmethod
    def dequeue(cls, task: Task) -> bool:
        task.actions.append("dequeue")
        task.status = "created"
        return True


class ArtifactResponse:
    def __init__(self, payload: bytes, url: str) -> None:
        self.status = 200
        self._stream = io.BytesIO(payload)
        self._url = url
        self.closed = False

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def read(self, size: int) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        self.closed = True


class ArtifactOpener:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls: list[str] = []
        self.authorization: list[str] = []

    def open(self, request: object, *, timeout: float) -> ArtifactResponse:
        assert timeout == 30.0
        url = str(request.full_url)
        self.urls.append(url)
        self.authorization.append(str(request.headers.get("Authorization") or ""))
        return ArtifactResponse(self.payload, url)


def artifact_task(payload: bytes, *, artifact_sha256: str | None = None) -> object:
    name = "sealed_provenance"
    digest = artifact_sha256 or hashlib.sha256(payload).hexdigest()
    url = f"http://{recovery.LEGACY_FILES_SERVER_HOST}:8081/path/payload.json"
    proxy = SimpleNamespace(uri=url, content_size=len(payload), hash=digest)
    record = SimpleNamespace(
        key=name,
        uri=url,
        content_size=len(payload),
        hash=digest,
        type_data=SimpleNamespace(preview=""),
    )
    return SimpleNamespace(
        id="e" * 32,
        artifacts={name: proxy},
        data=SimpleNamespace(execution=SimpleNamespace(artifacts=[record])),
    )


def recovery_receipt(path: object) -> dict[str, object]:
    rows = []
    for scope, subject, task_id in recovery.EXPECTED_RECOVERY_TASKS:
        rows.append(
            {
                "task_id": task_id,
                "scope": scope,
                "subject": subject,
                "status": "created",
                "execution_queue_id": "",
                "script_sha256": "1" * 64,
                "parameters_sha256": "2" * 64,
                "input_model_ids": ["3" * 32] if scope == "candidate" else [],
                "actions": [
                    "reset_force",
                    "install_gpu_preflight_source",
                    "restore_exact_parameters",
                ],
            }
        )
    value = recovery._seal(
        {
            "schema_version": 1,
            "receipt_type": recovery.RECOVERY_RECEIPT_TYPE,
            "generated_at_utc": "2026-08-12T00:00:00.000000Z",
            "status": "recovered",
            "remote_state_changed": True,
            "attempt_receipt_seal_sha256": "4" * 64,
            "authoritative_plan_producer_task_id": (
                recovery.formal.AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
            ),
            "authoritative_plan_seal_sha256": (
                recovery.formal.AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256
            ),
            "formal_task_count": 26,
            "candidate_task_ids": dict(recovery.CANDIDATE_EVALUATION_IDS),
            "all_tasks_created_unqueued": True,
            "replacement_tasks_created": False,
            "recovered_tasks": rows,
        }
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def record(task: Task, *, model_id: str = "") -> dict[str, object]:
    return {
        "scope": "candidate" if model_id else "formal_w3",
        "subject": "ffnet",
        "task_id": task.id,
        "task_name": task.name,
        "parent_task_id": task.data.parent,
        "queue": "GPU4-V100",
        "queue_id": recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        "parameters": {"Args/stage": "baseline_validate"},
        "input_model_id": model_id,
        "input_model_name": "ffnet_final_checkpoint",
        "old_source": "old source",
        "target_source": "new source",
        "task": task,
    }


def test_attempt_snapshot_freezes_failure_before_reset() -> None:
    task = Task(
        "1" * 32,
        status="failed",
        source="old source",
        parameters={"Args/stage": "baseline_validate"},
        queue_id=recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        input_models=[],
    )
    value = recovery._attempt_snapshot(task, record=record(task))
    assert value["status"] == "failed"
    assert value["status_reason"] == "CUDA out of memory"
    assert value["console_evidence"]["content_stored"] is False
    assert value["console_evidence"]["failure_markers"] == [
        "cuda_out_of_memory"
    ]
    assert value["console_evidence"]["tail_sha256"] == hashlib.sha256(
        b"line 1\nCUDA out of memory"
    ).hexdigest()
    assert value["attempt_evidence_sha256"] == recovery._content_sha256(
        {key: item for key, item in value.items() if key != "attempt_evidence_sha256"}
    )


def test_authenticated_artifact_download_rewrites_only_authority_and_caches() -> None:
    payload = b'{"sealed":true}'
    opener = ArtifactOpener(payload)
    downloader = recovery.AuthenticatedArtifactDownloader(
        files_server_url=(
            f"http://{recovery.EXPECTED_FILES_SERVER_HOST}:"
            f"{recovery.EXPECTED_FILES_SERVER_PORT}"
        ),
        add_auth_headers=lambda headers: headers.update(
            {"Authorization": "Bearer ephemeral-secret"}
        ),
        opener=opener,
    )
    task = artifact_task(payload)

    assert downloader(task, "sealed_provenance") == payload
    assert downloader(task, "sealed_provenance") == payload
    assert opener.urls == [
        f"http://{recovery.EXPECTED_FILES_SERVER_HOST}:8081/path/payload.json"
    ]
    assert opener.authorization == ["Bearer ephemeral-secret"]
    assert len(downloader.audit_records) == 1
    assert "ephemeral-secret" not in json.dumps(downloader.audit_records)


def test_authenticated_artifact_download_rejects_sha_mismatch() -> None:
    payload = b'{"sealed":true}'
    downloader = recovery.AuthenticatedArtifactDownloader(
        files_server_url=(
            f"http://{recovery.EXPECTED_FILES_SERVER_HOST}:"
            f"{recovery.EXPECTED_FILES_SERVER_PORT}"
        ),
        add_auth_headers=lambda headers: headers.update(
            {"Authorization": "Bearer ephemeral-secret"}
        ),
        opener=ArtifactOpener(payload),
    )

    with pytest.raises(recovery.RecoveryError, match="SHA-256 mismatch"):
        downloader(
            artifact_task(payload, artifact_sha256="0" * 64),
            "sealed_provenance",
        )


def test_recovery_receipt_consumer_requires_exact_created_unqueued_ids(
    tmp_path: object,
) -> None:
    path = tmp_path / "recovery.json"
    value = recovery_receipt(path)

    binding = recovery.validate_recovery_receipt(path)

    assert binding["receipt_seal_sha256"] == value["seal_sha256"]
    assert binding["task_count"] == 28
    assert binding["all_tasks_created_unqueued"] is True


def test_recovery_receipt_consumer_rejects_resealed_task_order_drift(
    tmp_path: object,
) -> None:
    path = tmp_path / "recovery.json"
    value = recovery_receipt(path)
    value["recovered_tasks"][0], value["recovered_tasks"][1] = (
        value["recovered_tasks"][1],
        value["recovered_tasks"][0],
    )
    value = recovery._seal(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(recovery.RecoveryError, match="identity/order drifted"):
        recovery.validate_recovery_receipt(path)


def test_failed_formal_task_is_force_reset_unqueued_and_patched() -> None:
    task = Task(
        "2" * 32,
        status="failed",
        source="old source",
        parameters={"Args/stage": "baseline_validate"},
        queue_id=recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        input_models=[],
    )
    Tasks.task = task
    result = recovery._restore_one(Tasks, record(task), sleeper=lambda _value: None)
    assert result["status"] == "created"
    assert result["execution_queue_id"] == ""
    assert result["input_model_ids"] == []
    assert task.data.script.diff == "new source"
    assert task.tags == []
    assert "reset" in task.actions


def test_created_formal_task_with_stale_queue_is_reset_and_unqueued() -> None:
    task = Task(
        "7" * 32,
        status="created",
        source="old source",
        parameters={"Args/stage": "baseline_validate"},
        queue_id=recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        input_models=[],
    )
    Tasks.task = task
    result = recovery._restore_one(
        Tasks, record(task), sleeper=lambda _value: None
    )
    assert result["execution_queue_id"] == ""
    assert task.actions[0] == "reset"


def test_created_candidate_e3_with_stale_queue_is_reset_and_rebound() -> None:
    model_id = "8" * 32
    task = Task(
        recovery.CANDIDATE_EVALUATION_IDS["E3"],
        status="created",
        source="old source",
        parameters={"Args/stage": "baseline_validate"},
        queue_id=recovery.candidates.QUEUE_IDS["GPU4-5090"],
        input_models=[Model(model_id)],
    )
    Tasks.task = task
    recovery_record = record(task, model_id=model_id)
    recovery_record["queue"] = "GPU4-5090"
    recovery_record["queue_id"] = recovery.candidates.QUEUE_IDS["GPU4-5090"]
    result = recovery._restore_one(
        Tasks, recovery_record, sleeper=lambda _value: None
    )
    assert result["execution_queue_id"] == ""
    assert result["input_model_ids"] == [model_id]
    assert task.actions.index("reset") < task.actions.index("set_input")


def test_candidate_task_restores_exact_final_model_binding() -> None:
    model_id = "3" * 32
    task = Task(
        "4" * 32,
        status="stopped",
        source="old source",
        parameters={"Args/stage": "baseline_validate"},
        queue_id=recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        input_models=[Model(model_id)],
    )
    Tasks.task = task
    result = recovery._restore_one(
        Tasks,
        record(task, model_id=model_id),
        sleeper=lambda _value: None,
    )
    assert result["input_model_ids"] == [model_id]
    assert task.actions.count("set_input") == 1


def test_active_task_is_explicitly_stopped_before_force_reset() -> None:
    task = Task(
        "5" * 32,
        status="in_progress",
        source="old source",
        parameters={"Args/stage": "baseline_validate"},
        queue_id=recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        input_models=[],
    )
    Tasks.task = task
    recovery._restore_one(Tasks, record(task), sleeper=lambda _value: None)
    assert task.actions.index("stop") < task.actions.index("reset")


def test_source_drift_fails_before_any_reset() -> None:
    task = Task(
        "6" * 32,
        status="failed",
        source="unknown source",
        parameters={"Args/stage": "baseline_validate"},
        queue_id=recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        input_models=[],
    )
    Tasks.task = task
    with pytest.raises(recovery.RecoveryError, match="source drifted"):
        recovery._restore_one(Tasks, record(task), sleeper=lambda _value: None)
    assert task.actions == []


def test_console_receipt_never_copies_credentials() -> None:
    task = Task(
        "9" * 32,
        status="failed",
        source="old source",
        parameters={"Args/stage": "baseline_validate"},
        queue_id=recovery.formal.EXPECTED_QUEUE_IDS["GPU4-V100"],
        input_models=[],
    )
    task.get_reported_console_output = lambda _count: [
        "api.credentials.access_key = AK-DO-NOT-COPY",
        "Authorization: Bearer SECRET-TOKEN",
        "CUDA out of memory",
    ]
    value = recovery._attempt_snapshot(task, record=record(task))
    serialized = json.dumps(value, sort_keys=True)
    assert "AK-DO-NOT-COPY" not in serialized
    assert "SECRET-TOKEN" not in serialized
    assert value["console_evidence"]["content_stored"] is False
    assert value["console_evidence"]["tail_line_count"] == 3
