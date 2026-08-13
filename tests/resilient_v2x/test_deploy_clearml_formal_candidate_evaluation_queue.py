from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "tools/resilient_v2x/deploy_clearml_formal_candidate_evaluation_queue.py"
)
FAILED_RECEIPT_PATH = (
    ROOT
    / "artifacts/resilient_v2x/formal-candidate-evaluation-deployment"
    / "execute-receipt-20260812T045021.280754Z.json"
)
FAILED_CREATED_TASK_ID = "b6f0fbab32a5478183a45b3ca833fc01"
FAILED_RECEIPT_SEAL = (
    "5847f916c236162a0c2f6ffcf53eded407d996d4eb34c3aa280b13391bd0d8a5"
)
SPEC = importlib.util.spec_from_file_location(
    "deploy_clearml_formal_candidate_evaluation_queue_test_subject", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


class _Script:
    def __init__(
        self,
        *,
        entry_point: str = "",
        diff: str = "",
        binary: str = "python",
        requirements: dict[str, object] | None = None,
    ) -> None:
        self.repository = ""
        self.branch = ""
        self.version_num = ""
        self.tag = ""
        self.working_dir = "."
        self.entry_point = entry_point
        self.diff = diff
        self.binary = binary
        self.requirements = requirements or {}

    def to_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "branch": self.branch,
            "version_num": self.version_num,
            "tag": self.tag,
            "working_dir": self.working_dir,
            "entry_point": self.entry_point,
            "diff": self.diff,
            "binary": self.binary,
            "requirements": dict(self.requirements),
        }


class _FakeTask:
    def __init__(
        self,
        owner: type["_FakeTaskClass"],
        *,
        task_id: str,
        name: str,
        project_id: str,
        project_name: str,
        status: str,
        task_type: str = "controller",
    ) -> None:
        self.owner = owner
        self.id = task_id
        self.name = name
        self.project = project_id
        self.project_name = project_name
        self.status = status
        self.task_type = task_type
        self._offline_mode = False
        self._reload_skip_flag = None
        self._parameters: dict[str, object] = {}
        self._docker = ""
        self._data = SimpleNamespace(
            id=task_id,
            project=project_id,
            parent=None,
            tags=[],
            script=_Script(),
            output=SimpleNamespace(destination=None),
        )

    @property
    def data(self) -> object:
        return self._data

    @property
    def output_uri(self) -> str | None:
        return self._data.output.destination

    @output_uri.setter
    def output_uri(self, value: str) -> None:
        self.owner.events.append(("output_uri", self.id))
        self._data.output.destination = value

    def _reload(self) -> object:
        self.owner.events.append(("authority", self.id))
        return self._data

    def get_project_name(self) -> str:
        return self.project_name

    def get_parameters(self, **_: object) -> dict[str, object]:
        return dict(self._parameters)

    def set_parameters(self, values: dict[str, object]) -> None:
        self.owner.events.append(("set_parameters", self.id))
        self._parameters = {key: str(value) for key, value in values.items()}

    def set_script(self, **values: object) -> None:
        self.owner.events.append(("set_script", self.id))
        self._data.script.repository = str(values.get("repository") or "")
        self._data.script.branch = str(values.get("branch") or "")
        self._data.script.version_num = str(values.get("commit") or "")
        self._data.script.working_dir = str(values.get("working_dir") or "")
        self._data.script.entry_point = str(values.get("entry_point") or "")
        self._data.script.diff = str(values.get("diff") or "")

    def set_packages(self, packages: list[str]) -> None:
        self.owner.events.append(("set_packages", self.id))
        self._data.script.requirements = {"pip": "\n".join(packages)}

    def set_base_docker(
        self, *, docker_image: str, docker_arguments: str
    ) -> None:
        self.owner.events.append(("set_docker", self.id))
        self._docker = f"{docker_image} {docker_arguments}"

    def get_base_docker(self) -> str:
        return self._docker

    def set_parent(self, parent: str) -> None:
        self.owner.events.append(("set_parent", self.id))
        self._data.parent = parent

    def set_tags(self, tags: list[str]) -> bool:
        self.owner.events.append(("set_tags", self.id))
        self._data.tags = list(tags)
        return True

    def get_executed_queue(self, *, return_name: bool = False) -> str | None:
        if self.status == "created":
            return None
        return self.owner.queue_name if return_name else self.owner.queue_id

    def flush(self, *, wait_for_uploads: bool) -> bool:
        assert wait_for_uploads is True
        self.owner.events.append(("flush", self.id))
        return True


class _FakeTaskClass:
    TaskTypes = SimpleNamespace(controller="controller")
    registry: dict[str, _FakeTask]
    events: list[tuple[str, str]]
    next_id: int
    enqueue_mode: str
    queue_name: str
    queue_id: str

    @classmethod
    def reset(cls) -> None:
        cls.registry = {}
        cls.events = []
        cls.next_id = 0x900
        cls.enqueue_mode = "success"
        cls.queue_name = deploy.SERVICES_QUEUE
        cls.queue_id = deploy.SERVICES_QUEUE_ID
        parent = _FakeTask(
            cls,
            task_id=deploy.PARENT_TASK_ID,
            name=deploy.PARENT_TASK_NAME,
            project_id=deploy.PROJECT_ID,
            project_name=deploy.PROJECT_NAME,
            status="completed",
        )
        cls.registry[parent.id] = parent

    @classmethod
    def create(cls, **values: object) -> _FakeTask:
        task_id = f"{cls.next_id:032x}"
        cls.next_id += 1
        cls.events.append(("create", task_id))
        task = _FakeTask(
            cls,
            task_id=task_id,
            name=str(values["task_name"]),
            project_id=deploy.PROJECT_ID,
            project_name=str(values["project_name"]),
            status="created",
            task_type=str(values["task_type"]),
        )
        cls.registry[task_id] = task
        return task

    @classmethod
    def get_task(cls, *, task_id: str) -> _FakeTask:
        cls.events.append(("get_task", task_id))
        return cls.registry[task_id]

    @classmethod
    def get_tasks(
        cls,
        *,
        task_name: str,
        task_filter: dict[str, object],
        allow_archived: bool,
    ) -> list[_FakeTask]:
        assert allow_archived is True
        assert task_filter == {"parent": deploy.PARENT_TASK_ID}
        cls.events.append(("get_tasks", task_name))
        return [
            task
            for task in cls.registry.values()
            if re.fullmatch(task_name, task.name)
            and task._data.parent == deploy.PARENT_TASK_ID
        ]

    @classmethod
    def enqueue(cls, *, task: _FakeTask, queue_name: str) -> bool:
        assert queue_name == deploy.SERVICES_QUEUE
        cls.events.append(("enqueue", task.id))
        if cls.enqueue_mode == "raise_before":
            raise RuntimeError("enqueue callback failed before acceptance")
        task.status = "queued"
        if cls.enqueue_mode == "raise_after":
            raise RuntimeError("enqueue callback failed after acceptance")
        return True


@pytest.fixture(scope="module")
def prepared() -> dict[str, object]:
    return deploy._prepare()


def _install_exact(
    prepared: dict[str, object], *, status: str = "created"
) -> _FakeTask:
    task = _FakeTaskClass.create(
        project_name=deploy.PROJECT_NAME,
        task_name=deploy.CONTROLLER_NAME,
        task_type="controller",
    )
    deploy._configure_shell(task, prepared=prepared)
    task.status = status
    _FakeTaskClass.events.clear()
    return task


def _install_exact_with_id(
    prepared: dict[str, object], *, task_id: str, status: str = "created"
) -> _FakeTask:
    task = _FakeTask(
        _FakeTaskClass,
        task_id=task_id,
        name=deploy.CONTROLLER_NAME,
        project_id=deploy.PROJECT_ID,
        project_name=deploy.PROJECT_NAME,
        status="created",
    )
    _FakeTaskClass.registry[task_id] = task
    deploy._configure_shell(task, prepared=prepared)
    task.status = status
    _FakeTaskClass.events.clear()
    return task


def _assert_sealed(value: dict[str, object]) -> None:
    observed = str(value["seal_sha256"])
    unsealed = dict(value)
    unsealed.pop("seal_sha256")
    assert deploy._content_sha256(unsealed) == observed


def test_default_main_is_local_only_and_writes_sealed_plan(
    tmp_path: Path,
    prepared: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deploy, "_prepare", lambda: prepared)
    destination = tmp_path / "plan.json"
    assert deploy.main(["--receipt", str(destination)]) == 0
    receipt = json.loads(destination.read_text(encoding="utf-8"))
    assert receipt["mode"] == "dry_run"
    assert receipt["status"] == "planned"
    assert receipt["remote_state_changed"] is False
    assert receipt["execution_contract"]["dry_run_performs_remote_calls"] is False
    assert receipt["execution_contract"]["controller_shell_count"] == 1
    assert receipt["fixed_bindings"]["services_queue_id"] == (
        "85707f0acbd04a4d95b49202474866b3"
    )
    assert receipt["parameters"]["Args/execute"] == "True"
    assert receipt["parameters"]["Args/execute_remotely"] == "False"
    assert receipt["parameters"]["Args/poll_seconds"] == "60.0"
    assert receipt["parameters"]["Args/timeout_hours"] == "168.0"
    _assert_sealed(receipt)


def test_source_is_exact_and_passes_isolated_python_i_smoke(
    prepared: dict[str, object],
) -> None:
    source = deploy.SOURCE_PATH.read_text(encoding="utf-8")
    assert prepared["source"] == source
    assert prepared["source_record"]["deployment_mode"] == "exact_standalone_diff"
    assert prepared["standalone_import_smoke"]["result"] == "pass"
    assert prepared["standalone_import_smoke"]["isolation"] == (
        "python_-I_without_repository_on_sys_path"
    )
    assert prepared["standalone_import_smoke"]["source_sha256"] == (
        prepared["source_record"]["sha256"]
    )


def test_fresh_execute_creates_exactly_one_shell_and_verifies_queue(
    prepared: dict[str, object],
) -> None:
    _FakeTaskClass.reset()
    result = deploy._deploy(
        _FakeTaskClass, prepared=prepared, sleeper=lambda _: None
    )
    assert result["created_shell"] is True
    assert result["disposition"] == "created_and_enqueued"
    assert result["parent"]["task_id"] == deploy.PARENT_TASK_ID
    assert result["queued_authority"]["status"] == "queued"
    assert result["queued_authority"]["queue"] == "services"
    assert result["queued_authority"]["queue_id"] == deploy.SERVICES_QUEUE_ID
    assert [event for event in _FakeTaskClass.events if event[0] == "create"] == [
        ("create", result["task_id"])
    ]
    assert len([event for event in _FakeTaskClass.events if event[0] == "enqueue"]) == 1
    task = _FakeTaskClass.registry[str(result["task_id"])]
    assert task._data.script.diff == prepared["source"]
    assert task._data.script.entry_point == deploy.ENTRY_POINT
    assert task._data.script.requirements == {"pip": "clearml==2.1.11"}
    assert task._parameters["Args/execute"] == "True"
    assert task._parameters["Args/execute_remotely"] == "False"
    assert task._parameters["Deployment/deployment_seal_sha256"] == (
        prepared["deployment_seal_sha256"]
    )


def test_repeated_execute_reuses_exact_queued_task_without_mutation(
    prepared: dict[str, object],
) -> None:
    _FakeTaskClass.reset()
    task = _install_exact(prepared, status="queued")
    result = deploy._deploy(
        _FakeTaskClass, prepared=prepared, sleeper=lambda _: None
    )
    assert result["task_id"] == task.id
    assert result["created_shell"] is False
    assert result["disposition"] == "reused_exact_deployment"
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]


def test_instant_start_accepts_only_auditable_agent_materialized_requirements(
    prepared: dict[str, object],
) -> None:
    _FakeTaskClass.reset()
    task = _install_exact(prepared, status="in_progress")
    task._data.script.requirements = {
        "org_pip": "clearml==2.1.11",
        "pip": ["attrs==26.1.0", "clearml==2.1.11"],
    }
    result = deploy._deploy(
        _FakeTaskClass, prepared=prepared, sleeper=lambda _: None
    )
    assert result["disposition"] == "reused_exact_deployment"
    assert result["queued_authority"]["requirements"]["form"] == (
        "agent_materialized"
    )
    assert result["queued_authority"]["requirements"]["resolved_count"] == 2

    task._data.script.requirements["pip"] = ["attrs==26.1.0"]
    with pytest.raises(deploy.DeploymentError, match="resolved package requirements"):
        deploy._deploy(_FakeTaskClass, prepared=prepared, sleeper=lambda _: None)


def test_repeated_execute_resumes_exact_created_shell_without_new_create(
    prepared: dict[str, object],
) -> None:
    _FakeTaskClass.reset()
    task = _install_exact(prepared, status="created")
    result = deploy._deploy(
        _FakeTaskClass, prepared=prepared, sleeper=lambda _: None
    )
    assert result["task_id"] == task.id
    assert result["created_shell"] is False
    assert result["disposition"] == "resumed_exact_created_shell"
    assert not [event for event in _FakeTaskClass.events if event[0] == "create"]
    assert [event for event in _FakeTaskClass.events if event[0] == "enqueue"] == [
        ("enqueue", task.id)
    ]


def test_clearml_21_failed_receipt_is_archived_after_source_revision(
    prepared: dict[str, object],
) -> None:
    failed = json.loads(FAILED_RECEIPT_PATH.read_text(encoding="utf-8"))
    assert failed["seal_sha256"] == FAILED_RECEIPT_SEAL
    _assert_sealed(failed)
    assert failed["partial_journal"]["created_task_id"] == FAILED_CREATED_TASK_ID
    assert failed["partial_journal"]["configured_task_id"] == FAILED_CREATED_TASK_ID
    assert failed["partial_journal"]["duplicate_post_create_task_ids"] == []
    assert failed["fixed_bindings"]["deployment_seal_sha256"] != (
        prepared["deployment_seal_sha256"]
    )
    assert failed["fixed_bindings"]["source_sha256"] != (
        prepared["source_record"]["sha256"]
    )


def test_project_id_drift_fails_closed_without_create_or_enqueue(
    prepared: dict[str, object],
) -> None:
    _FakeTaskClass.reset()
    task = _install_exact(prepared)
    task.project = "0" * 32
    task._data.project = "0" * 32
    with pytest.raises(deploy.DeploymentError, match="project ID drifted"):
        deploy._deploy(_FakeTaskClass, prepared=prepared, sleeper=lambda _: None)
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]


def test_same_name_source_or_seal_drift_fails_without_creating_duplicate(
    prepared: dict[str, object],
) -> None:
    _FakeTaskClass.reset()
    task = _install_exact(prepared)
    task._data.script.diff += "\n# drift\n"
    with pytest.raises(deploy.DeploymentError, match="source bytes drifted"):
        deploy._deploy(_FakeTaskClass, prepared=prepared, sleeper=lambda _: None)
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]

    task._data.script.diff = str(prepared["source"])
    task._parameters["Deployment/deployment_seal_sha256"] = "0" * 64
    with pytest.raises(deploy.DeploymentError, match="parameter contract drifted"):
        deploy._deploy(_FakeTaskClass, prepared=prepared, sleeper=lambda _: None)
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]


def test_callback_exception_after_authoritative_acceptance_is_success(
    prepared: dict[str, object],
) -> None:
    _FakeTaskClass.reset()
    _FakeTaskClass.enqueue_mode = "raise_after"
    result = deploy._deploy(
        _FakeTaskClass, prepared=prepared, sleeper=lambda _: None
    )
    assert result["disposition"] == "created_and_enqueued"
    assert result["queued_authority"]["status"] == "queued"


def test_failed_enqueue_receipt_is_sealed_and_retry_does_not_duplicate(
    prepared: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deploy, "_prepare", lambda: prepared)
    _FakeTaskClass.reset()
    _FakeTaskClass.enqueue_mode = "raise_before"
    with pytest.raises(deploy.DeploymentError) as caught:
        deploy._execute_receipt(_FakeTaskClass)
    receipt = caught.value.sealed_receipt
    assert receipt["status"] == "failed_closed"
    assert receipt["remote_state_changed"] is True
    assert receipt["partial_journal"]["created_task_id"]
    _assert_sealed(receipt)
    assert len([event for event in _FakeTaskClass.events if event[0] == "create"]) == 1

    _FakeTaskClass.enqueue_mode = "success"
    _FakeTaskClass.events.clear()
    recovered = deploy._execute_receipt(_FakeTaskClass)
    assert recovered["disposition"] == "resumed_exact_created_shell"
    assert recovered["created_shell"] is False
    assert not [event for event in _FakeTaskClass.events if event[0] == "create"]
    assert len([event for event in _FakeTaskClass.events if event[0] == "enqueue"]) == 1


def test_multiple_exact_names_fail_closed_before_create_or_enqueue(
    prepared: dict[str, object],
) -> None:
    _FakeTaskClass.reset()
    _install_exact(prepared)
    _install_exact(prepared)
    _FakeTaskClass.events.clear()
    with pytest.raises(deploy.DeploymentError, match="multiple exact-name"):
        deploy._deploy(_FakeTaskClass, prepared=prepared, sleeper=lambda _: None)
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]


def test_task_id_verifier_checks_source_parameters_docker_and_queue(
    prepared: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deploy, "_prepare", lambda: prepared)
    _FakeTaskClass.reset()
    task = _install_exact(prepared, status="queued")
    receipt = deploy._verify_receipt(_FakeTaskClass, task_id=task.id)
    assert receipt["mode"] == "verify"
    assert receipt["status"] == "verified"
    assert receipt["task_authority"]["source_sha256"] == (
        prepared["source_record"]["sha256"]
    )
    assert receipt["task_authority"]["queue_id"] == deploy.SERVICES_QUEUE_ID
    _assert_sealed(receipt)

    _FakeTaskClass.queue_id = "0" * 32
    with pytest.raises(deploy.DeploymentError, match="execution queue drifted"):
        deploy._verify_receipt(_FakeTaskClass, task_id=task.id)


def test_execute_cli_requires_exact_deployment_token() -> None:
    with pytest.raises(deploy.DeploymentError, match="exact deployment token"):
        deploy.main(["--execute", "--execute-token", "wrong-token"])


def test_execute_cli_requires_evaluation_recovery_receipt() -> None:
    with pytest.raises(
        deploy.DeploymentError, match="evaluation-recovery-receipt is required"
    ):
        deploy.main(
            [
                "--execute",
                "--execute-token",
                deploy.DEPLOY_EXECUTE_TOKEN,
            ]
        )


def test_execute_cli_requires_controller_recovery_receipt(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")
    with pytest.raises(
        deploy.DeploymentError, match="controller-recovery-receipt is required"
    ):
        deploy.main(
            [
                "--execute",
                "--execute-token",
                deploy.DEPLOY_EXECUTE_TOKEN,
                "--evaluation-recovery-receipt",
                str(evaluation),
            ]
        )


def test_dry_run_binds_validated_evaluation_recovery_receipt(
    tmp_path: Path,
    prepared: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovery_path = tmp_path / "recovery.json"
    recovery_path.write_text("{}", encoding="utf-8")
    binding = {
        "path": str(recovery_path.resolve()),
        "receipt_seal_sha256": "1" * 64,
        "attempt_receipt_seal_sha256": "2" * 64,
        "task_count": 28,
        "all_tasks_created_unqueued": True,
        "replacement_tasks_created": False,
    }
    monkeypatch.setattr(deploy, "_prepare", lambda: prepared)
    monkeypatch.setattr(
        deploy,
        "_validate_evaluation_recovery_receipt",
        lambda path: binding if path == recovery_path else None,
    )
    destination = tmp_path / "plan.json"

    assert (
        deploy.main(
            [
                "--evaluation-recovery-receipt",
                str(recovery_path),
                "--receipt",
                str(destination),
            ]
        )
        == 0
    )
    receipt = json.loads(destination.read_text(encoding="utf-8"))
    assert receipt["evaluation_recovery_receipt"] == binding
    _assert_sealed(receipt)


def test_dry_run_binds_both_recovery_receipts(
    tmp_path: Path,
    prepared: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    controller_path = tmp_path / "controller.json"
    evaluation_path.write_text("{}", encoding="utf-8")
    controller_path.write_text("{}", encoding="utf-8")
    evaluation_binding = {
        "path": str(evaluation_path.resolve()),
        "receipt_seal_sha256": "1" * 64,
        "task_count": 28,
        "task_ids_sha256": "3" * 64,
    }
    controller_binding = {
        "path": str(controller_path.resolve()),
        "receipt_seal_sha256": "2" * 64,
        "controller_task_id": FAILED_CREATED_TASK_ID,
        "created_unqueued": True,
        "evaluation_recovery_receipt_seal_sha256": "1" * 64,
        "evaluation_recovery_task_ids_sha256": "3" * 64,
    }
    monkeypatch.setattr(deploy, "_prepare", lambda: prepared)
    monkeypatch.setattr(
        deploy,
        "_validate_evaluation_recovery_receipt",
        lambda path: evaluation_binding if path == evaluation_path else None,
    )
    monkeypatch.setattr(
        deploy,
        "_validate_controller_recovery_receipt",
        lambda path: controller_binding if path == controller_path else None,
    )
    destination = tmp_path / "plan.json"
    assert (
        deploy.main(
            [
                "--evaluation-recovery-receipt",
                str(evaluation_path),
                "--controller-recovery-receipt",
                str(controller_path),
                "--receipt",
                str(destination),
            ]
        )
        == 0
    )
    receipt = json.loads(destination.read_text(encoding="utf-8"))
    assert receipt["evaluation_recovery_receipt"] == evaluation_binding
    assert receipt["candidate_controller_recovery_receipt"] == controller_binding
    _assert_sealed(receipt)
