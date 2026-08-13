from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/deploy_clearml_formal_successor_chain.py"
SPEC = importlib.util.spec_from_file_location(
    "deploy_clearml_formal_successor_chain_test_subject", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


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


class _ArtifactRecord:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def to_dict(self) -> dict[str, object]:
        return dict(self.value)


class _FakeTask:
    def __init__(
        self,
        owner: type["_FakeTaskClass"],
        *,
        task_id: str,
        name: str,
        project: str,
        status: str,
        task_type: str = "controller",
        entry_point: str = "",
        source: str = "",
    ) -> None:
        self.owner = owner
        self.id = task_id
        self.name = name
        self.project = project
        self.status = status
        self.task_type = task_type
        self._offline_mode = False
        self._reload_skip_flag = None
        self._parameters: dict[str, object] = {}
        self._configuration_objects: dict[str, object] = {}
        self._docker = ""
        self.artifacts: dict[str, object] = {}
        self._data = SimpleNamespace(
            id=task_id,
            parent=None,
            tags=[],
            system_tags=[],
            script=_Script(entry_point=entry_point, diff=source),
            output=SimpleNamespace(destination=None),
            execution=SimpleNamespace(artifacts=[]),
        )

    @property
    def data(self) -> object:
        return self._data

    @property
    def output_uri(self) -> str | None:
        return self._data.output.destination

    @output_uri.setter
    def output_uri(self, value: str) -> None:
        self.owner.events.append(("output_uri", self.name))
        self._data.output.destination = value

    def _reload(self) -> object:
        self.owner.events.append(("authority", self.name))
        if (
            self.owner.drift_parent_role
            and self.name
            == deploy.ROLE_DEFINITIONS[self.owner.drift_parent_role]["task_name"]
            and self._data.script.diff
        ):
            self._data.parent = "f" * 32
        return self._data

    def get_project_name(self) -> str:
        return self.project

    def get_parameters(self, **_: object) -> dict[str, object]:
        return dict(self._parameters)

    def get_configuration_objects(self) -> dict[str, object]:
        return dict(self._configuration_objects)

    def set_parameters(self, value: dict[str, object]) -> None:
        self.owner.events.append(("set_parameters", self.name))
        self._parameters = {key: str(item) for key, item in value.items()}

    def set_script(self, **values: object) -> None:
        self.owner.events.append(("set_script", self.name))
        self._data.script.repository = str(values.get("repository") or "")
        self._data.script.branch = str(values.get("branch") or "")
        self._data.script.version_num = str(values.get("commit") or "")
        self._data.script.working_dir = str(values.get("working_dir") or "")
        self._data.script.entry_point = str(values.get("entry_point") or "")
        self._data.script.diff = str(values.get("diff") or "")

    def set_packages(self, packages: list[str]) -> None:
        self.owner.events.append(("set_packages", self.name))
        self._data.script.requirements = {"pip": "\n".join(packages)}

    def set_base_docker(
        self,
        *,
        docker_image: str,
        docker_arguments: str,
    ) -> None:
        self.owner.events.append(("set_docker", self.name))
        self._docker = f"{docker_image} {docker_arguments}"

    def get_base_docker(self) -> str:
        return self._docker

    def get_executed_queue(self, *, return_name: bool = False) -> str | None:
        if self.status == "created":
            return None
        return deploy.SERVICES_QUEUE if return_name else deploy.SERVICES_QUEUE_ID

    def set_parent(self, parent: str) -> None:
        self.owner.events.append(("set_parent", self.name))
        self._data.parent = parent

    def flush(self, *, wait_for_uploads: bool) -> bool:
        assert wait_for_uploads is True
        self.owner.events.append(("flush", self.name))
        return True


class _FakeTaskClass:
    TaskTypes = SimpleNamespace(controller="controller")
    registry: dict[str, _FakeTask]
    events: list[tuple[str, str]]
    drift_parent_role: str | None
    enqueue_error: Exception | None
    next_id: int

    @classmethod
    def reset(cls, *, template_source: str = "template-source") -> None:
        cls.registry = {}
        cls.events = []
        cls.drift_parent_role = None
        cls.enqueue_error = None
        cls.next_id = 0x100
        controller_source = deploy._read_source(
            deploy.ROOT / "tools/resilient_v2x" / deploy.CONTROLLER_ENTRY_POINT,
            context="controller fixture",
        )
        cls.registry[deploy.TRAINING_CONTROLLER_TASK_ID] = _FakeTask(
            cls,
            task_id=deploy.TRAINING_CONTROLLER_TASK_ID,
            name="completed controller",
            project=deploy.PROJECT_NAME,
            status="completed",
            entry_point=deploy.CONTROLLER_ENTRY_POINT,
            source=controller_source,
        )
        cls.registry[deploy.EVALUATION_TEMPLATE_TASK_ID] = _FakeTask(
            cls,
            task_id=deploy.EVALUATION_TEMPLATE_TASK_ID,
            name="completed evaluation template",
            project=deploy.PROJECT_NAME,
            status="completed",
            entry_point="clearml_5090_bootstrap.py",
            source=template_source,
        )

    @classmethod
    def create(cls, **values: object) -> _FakeTask:
        task_id = f"{cls.next_id:032x}"
        cls.next_id += 1
        name = str(values["task_name"])
        cls.events.append(("create", name))
        task = _FakeTask(
            cls,
            task_id=task_id,
            name=name,
            project=str(values["project_name"]),
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
    def enqueue(cls, *, task: _FakeTask, queue_name: str) -> bool:
        assert queue_name == deploy.SERVICES_QUEUE
        cls.events.append(("enqueue", task.name))
        if cls.enqueue_error is not None:
            raise cls.enqueue_error
        task.status = "queued"
        return True


def _install_completed_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeTask:
    requirements: dict[str, object] = {
        "org_pip": deploy.SERVICE_REQUIREMENTS[0],
        "pip": [deploy.SERVICE_REQUIREMENTS[0]],
    }
    type_data = {
        "content_type": "application/json",
        "preview": "sealed provenance preview",
    }
    expected_artifact = dict(deploy.COMPLETED_PROVENANCE_ARTIFACT)
    expected_artifact["type_data_sha256"] = deploy._content_sha256(type_data)
    expected_artifact["type_data_preview_bytes"] = len(
        type_data["preview"].encode("utf-8")
    )
    monkeypatch.setattr(
        deploy,
        "COMPLETED_PROVENANCE_REQUIREMENTS_SHA256",
        deploy._content_sha256(requirements),
    )
    monkeypatch.setattr(deploy, "COMPLETED_PROVENANCE_ARTIFACT", expected_artifact)
    source = deploy._read_source(
        deploy.SOURCE_PATHS["P"], context="completed P fixture"
    )
    task = _FakeTask(
        _FakeTaskClass,
        task_id=deploy.COMPLETED_PROVENANCE_TASK_ID,
        name=str(deploy.ROLE_DEFINITIONS["P"]["task_name"]),
        project=deploy.PROJECT_NAME,
        status="completed",
        entry_point=str(deploy.ROLE_DEFINITIONS["P"]["entry_point"]),
        source=source,
    )
    task._data.parent = deploy.TRAINING_CONTROLLER_TASK_ID
    task._data.output.destination = deploy.FILES_SERVER_URI
    task._data.tags = sorted(deploy.COMPLETED_PROVENANCE_TAGS)
    task._data.system_tags = []
    task._data.script.requirements = requirements
    task._docker = f"{deploy.SERVICE_DOCKER_IMAGE} {deploy.SERVICE_DOCKER_ARGS}"
    task._parameters = {
        "Args/poll_seconds": "60.0",
        "Args/timeout_hours": "720.0",
        "Args/training_controller_task_id": deploy.TRAINING_CONTROLLER_TASK_ID,
    }
    artifact_value = {
        key: value
        for key, value in expected_artifact.items()
        if key
        not in {
            "type_data_sha256",
            "type_data_content_type",
            "type_data_preview_bytes",
        }
    }
    artifact_value["type_data"] = type_data
    task._data.execution.artifacts = [_ArtifactRecord(artifact_value)]
    task.artifacts = {str(expected_artifact["key"]): object()}
    _FakeTaskClass.registry[deploy.COMPLETED_PROVENANCE_TASK_ID] = task
    return task


def test_default_main_is_local_only_and_writes_a_valid_sealed_plan(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "plan.json"
    assert deploy.main(["--receipt", str(receipt_path)]) == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["mode"] == "dry_run"
    assert receipt["status"] == "planned"
    assert receipt["remote_state_changed"] is False
    assert receipt["fixed_bindings"]["services_queue"] == "services"
    assert receipt["fixed_bindings"]["services_queue_id"] == (
        "85707f0acbd04a4d95b49202474866b3"
    )
    assert receipt["execution_contract"]["dry_run_performs_remote_calls"] is False
    assert receipt["standalone_import_smoke"]["result"] == "pass"
    assert receipt["standalone_import_smoke"]["isolation"] == (
        "python_-I_without_repository_on_sys_path"
    )
    assert receipt["fixed_bindings"]["role_output_artifacts"]["S"] == [
        "final_selector_formal_inputs",
        "formal_candidate_selection",
    ]
    assert receipt["sources"]["S"]["output_artifacts"] == [
        "final_selector_formal_inputs",
        "formal_candidate_selection",
    ]
    assert (
        receipt["sources"]["S"]["constant_replacements"]["WATCHER_SCRIPT_SHA256"]
        == receipt["sources"]["W"]["deployed_sha256"]
    )
    for role in ("P", "W", "L", "A"):
        assert receipt["sources"][role]["standalone_bundle"] == {
            "embedded_module_count": 0,
            "embedded_modules": [],
        }
    selector_modules = receipt["sources"]["S"]["standalone_bundle"]
    assert selector_modules["embedded_module_count"] == 2
    expected_paths = {
        "tools.resilient_v2x.sota_gate": deploy.ROOT
        / "tools/resilient_v2x/sota_gate.py",
        "tools.resilient_v2x.sota_selector": deploy.ROOT
        / "tools/resilient_v2x/sota_selector.py",
    }
    observed_names = []
    for module in selector_modules["embedded_modules"]:
        name = module["module_name"]
        observed_names.append(name)
        source = expected_paths[name].read_bytes()
        assert module["base_bytes"] == len(source)
        assert module["base_sha256"] == hashlib.sha256(source).hexdigest()
        assert module["compressed_bytes"] > 0
        assert module["base85_payload_bytes"] > 0
    assert observed_names == list(expected_paths)
    assert receipt["standalone_import_smoke"]["roles"]["S"][
        "embedded_modules_verified"
    ] == list(expected_paths)
    seal = receipt.pop("seal_sha256")
    assert deploy._content_sha256(receipt) == seal


def test_render_sources_pins_dynamic_chain_ids_and_latest_producer_hashes() -> None:
    sources = deploy._read_base_sources()
    task_ids = {
        role: f"{index + 100:032x}" for index, role in enumerate(deploy.ROLE_ORDER)
    }
    rendered, _ = deploy._render_sources(sources, task_ids)
    assert deploy._assignment_value(
        rendered["A"], "TRAINING_PROVENANCE_SCRIPT_SHA256"
    ) == _sha(rendered["P"])
    assert deploy._assignment_value(rendered["A"], "WATCHER_SCRIPT_SHA256") == _sha(
        rendered["W"]
    )
    assert deploy._assignment_value(rendered["A"], "LEADERBOARD_SCRIPT_SHA256") == _sha(
        rendered["L"]
    )
    assert deploy._assignment_value(rendered["S"], "TRAINING_CONTROLLER_TASK_ID") == (
        deploy.TRAINING_CONTROLLER_TASK_ID
    )
    assert (
        deploy._assignment_value(rendered["S"], "TRAINING_PROVENANCE_TASK_ID")
        == (task_ids["P"])
    )
    assert deploy._assignment_value(rendered["S"], "WATCHER_TASK_ID") == task_ids["W"]
    assert (
        deploy._assignment_value(rendered["S"], "DEFAULT_LEADERBOARD_TASK_ID")
        == (task_ids["L"])
    )
    assert (
        deploy._assignment_value(rendered["S"], "DEFAULT_AUDIT_TASK_ID")
        == task_ids["A"]
    )
    assert deploy._assignment_value(rendered["S"], "AUDIT_SCRIPT_SHA256") == _sha(
        rendered["A"]
    )


def test_normal_chain_has_blockers_without_w3_recovery_pins() -> None:
    source_sha256 = {role: "a" * 64 for role in deploy.ROLE_ORDER}
    watcher = deploy._parameters(deploy.DRY_RUN_IDS, source_sha256)["W"]
    assert watcher["Args/candidate_release_blockers_json"] == (
        deploy.CANDIDATE_RELEASE_BLOCKERS_JSON
    )
    assert "Args/authoritative_evaluation_plan_producer_task_id" not in watcher
    assert "Args/authoritative_evaluation_plan_seal_sha256" not in watcher


def test_render_sources_rejects_duplicate_or_controller_alias_ids() -> None:
    sources = deploy._read_base_sources()
    duplicate = dict(deploy.DRY_RUN_IDS)
    duplicate["S"] = duplicate["A"]
    with pytest.raises(deploy.DeploymentError, match="must be unique"):
        deploy._render_sources(sources, duplicate)
    alias = dict(deploy.DRY_RUN_IDS)
    alias["P"] = deploy.TRAINING_CONTROLLER_TASK_ID
    with pytest.raises(deploy.DeploymentError, match="aliases the training controller"):
        deploy._render_sources(sources, alias)


def test_local_import_scanner_bundles_supported_modules_and_rejects_missing() -> None:
    assert deploy._repository_local_imports(
        "import sota_gate\nfrom tools.resilient_v2x import sota_selector\n"
    ) == (
        "tools.resilient_v2x.sota_gate",
        "tools.resilient_v2x.sota_selector",
    )
    with pytest.raises(deploy.DeploymentError, match="is not file-backed"):
        deploy._repository_local_imports(
            "from tools.resilient_v2x import definitely_missing_module\n"
        )


def test_isolated_import_smoke_rejects_tampered_embedded_payload() -> None:
    sources = deploy._read_base_sources()
    rendered, _, _, records = deploy._render_sources_with_bundle_evidence(
        sources, deploy.DRY_RUN_IDS
    )
    embedded_sha = str(records["S"][0]["base_sha256"])
    assert rendered["S"].count(embedded_sha) == 1
    rendered["S"] = rendered["S"].replace(embedded_sha, "0" * 64, 1)
    with pytest.raises(deploy.DeploymentError, match="isolated import smoke failed"):
        deploy._isolated_import_smoke(rendered, records)


def test_execute_creates_all_shells_before_sources_and_enqueues_in_dependency_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_source = "template-source"
    monkeypatch.setattr(deploy, "EVALUATION_SCRIPT_SHA256", _sha(template_source))
    _FakeTaskClass.reset(template_source=template_source)
    original_source_reader = deploy._read_base_sources

    def tracked_source_reader() -> dict[str, str]:
        _FakeTaskClass.events.append(("read_five_sources", "local"))
        return original_source_reader()

    monkeypatch.setattr(deploy, "_read_base_sources", tracked_source_reader)
    result = deploy._deploy_chain(_FakeTaskClass, sleeper=lambda _: None)

    creates = [event for event in _FakeTaskClass.events if event[0] == "create"]
    installs = [event for event in _FakeTaskClass.events if event[0] == "set_script"]
    assert len(creates) == len(installs) == 5
    last_create = max(_FakeTaskClass.events.index(event) for event in creates)
    first_source_read = _FakeTaskClass.events.index(("read_five_sources", "local"))
    first_install = min(_FakeTaskClass.events.index(event) for event in installs)
    assert last_create < first_source_read < first_install
    assert [event[1] for event in _FakeTaskClass.events if event[0] == "enqueue"] == [
        deploy.ROLE_DEFINITIONS[role]["task_name"] for role in deploy.ROLE_ORDER
    ]
    ids = result["task_ids"]
    assert result["parents"] == {
        "P": deploy.TRAINING_CONTROLLER_TASK_ID,
        "W": ids["P"],
        "L": ids["W"],
        "A": ids["L"],
        "S": ids["A"],
    }
    assert all(
        record["status"] == "created"
        for record in result["pre_enqueue_authority"].values()
    )
    assert all(
        record["status"] == "queued" and record["queue"] == "services"
        for record in result["queued_authority"].values()
    )
    assert result["standalone_import_smoke"]["result"] == "pass"
    assert result["sources"]["S"]["standalone_bundle"]["embedded_module_count"] == 2


def test_completed_provenance_successor_default_is_local_only(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "completed-p-successor-plan.json"
    assert (
        deploy.main(
            [
                "--from-completed-provenance-task-id",
                deploy.COMPLETED_PROVENANCE_TASK_ID,
                "--receipt",
                str(receipt_path),
            ]
        )
        == 0
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["mode"] == "completed_provenance_successor_dry_run"
    assert receipt["status"] == "planned"
    assert receipt["remote_state_changed"] is False
    assert receipt["dependency_task_ids"]["P"] == (deploy.COMPLETED_PROVENANCE_TASK_ID)
    assert set(receipt["placeholder_task_ids"]) == set(deploy.SUCCESSOR_ROLE_ORDER)
    assert receipt["created_roles"] == list(deploy.SUCCESSOR_ROLE_ORDER)
    assert receipt["fixed_bindings"]["enqueue_order"] == list(
        deploy.SUCCESSOR_ROLE_ORDER
    )
    assert receipt["fixed_bindings"]["candidate_release_blockers"] == [
        dict(item) for item in deploy.CANDIDATE_RELEASE_BLOCKERS
    ]
    assert (
        receipt["fixed_bindings"]["authoritative_evaluation_plan_producer_task_id"]
        == deploy.AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
    )
    watcher_parameters = receipt["parameters"]["W"]
    assert watcher_parameters["Args/candidate_release_blockers_json"] == (
        deploy.CANDIDATE_RELEASE_BLOCKERS_JSON
    )
    assert (
        watcher_parameters["Args/authoritative_evaluation_plan_producer_task_id"]
        == deploy.AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
    )
    assert (
        watcher_parameters["Args/authoritative_evaluation_plan_seal_sha256"]
        == deploy.AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256
    )
    contract = receipt["execution_contract"]
    assert contract["dry_run_performs_remote_calls"] is False
    assert contract["only_four_successor_shells_are_created"] is True
    assert contract["completed_provenance_authority_validation"] == (
        "deferred_until_explicit_execute"
    )
    seal = receipt.pop("seal_sha256")
    assert deploy._content_sha256(receipt) == seal


def test_completed_provenance_successor_creates_only_four_and_pins_latest_watcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_source = "template-source"
    monkeypatch.setattr(deploy, "EVALUATION_SCRIPT_SHA256", _sha(template_source))
    _FakeTaskClass.reset(template_source=template_source)
    _install_completed_provenance(monkeypatch)
    original_source_reader = deploy._read_base_sources
    latest_marker = "\n# latest watcher compatibility revision\n"

    def latest_source_reader() -> dict[str, str]:
        _FakeTaskClass.events.append(("read_five_sources", "local"))
        sources = original_source_reader()
        sources["W"] += latest_marker
        return sources

    monkeypatch.setattr(deploy, "_read_base_sources", latest_source_reader)
    result = deploy._deploy_successors_from_completed_provenance(
        _FakeTaskClass, sleeper=lambda _: None
    )

    creates = [event for event in _FakeTaskClass.events if event[0] == "create"]
    assert [event[1] for event in creates] == [
        deploy.ROLE_DEFINITIONS[role]["task_name"]
        for role in deploy.SUCCESSOR_ROLE_ORDER
    ]
    first_source_read = _FakeTaskClass.events.index(("read_five_sources", "local"))
    assert (
        max(_FakeTaskClass.events.index(event) for event in creates) < first_source_read
    )
    assert result["task_ids"]["P"] == deploy.COMPLETED_PROVENANCE_TASK_ID
    assert result["created_roles"] == list(deploy.SUCCESSOR_ROLE_ORDER)
    assert set(result["created_task_ids"]) == set(deploy.SUCCESSOR_ROLE_ORDER)
    assert result["enqueue_order"] == list(deploy.SUCCESSOR_ROLE_ORDER)
    assert [event[1] for event in _FakeTaskClass.events if event[0] == "enqueue"] == [
        deploy.ROLE_DEFINITIONS[role]["task_name"]
        for role in deploy.SUCCESSOR_ROLE_ORDER
    ]
    watcher_id = result["task_ids"]["W"]
    watcher_source = _FakeTaskClass.registry[watcher_id]._data.script.diff
    assert watcher_source.endswith(latest_marker)
    assert result["sources"]["W"]["deployed_sha256"] == _sha(watcher_source)
    audit_id = result["task_ids"]["A"]
    audit_source = _FakeTaskClass.registry[audit_id]._data.script.diff
    assert deploy._assignment_value(audit_source, "WATCHER_SCRIPT_SHA256") == _sha(
        watcher_source
    )
    selector_id = result["task_ids"]["S"]
    selector_source = _FakeTaskClass.registry[selector_id]._data.script.diff
    assert deploy._assignment_value(selector_source, "WATCHER_TASK_ID") == watcher_id
    assert (
        result["parameters"]["W"]["Args/expected_training_provenance_script_sha256"]
        == deploy.COMPLETED_PROVENANCE_SOURCE_SHA256
    )
    assert (
        result["completed_provenance"]["artifact"]["hash"]
        == (deploy.COMPLETED_PROVENANCE_ARTIFACT["hash"])
    )
    assert all(
        record["status"] == "queued" and record["queue"] == "services"
        for record in result["queued_authority"].values()
    )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("parent", "parent drifted"),
        ("source", "source bytes drifted"),
        ("unknown_tag", "formal tag set drifted"),
        ("duplicate_tag", "contains duplicate values"),
        ("configuration", "configuration objects drifted"),
        ("parameter", "exact parameter contract drifted"),
        ("artifact", "sealed artifact metadata drifted"),
        ("requirements", "package requirements drifted"),
    ],
)
def test_completed_provenance_drift_fails_before_creating_successors(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    template_source = "template-source"
    monkeypatch.setattr(deploy, "EVALUATION_SCRIPT_SHA256", _sha(template_source))
    _FakeTaskClass.reset(template_source=template_source)
    task = _install_completed_provenance(monkeypatch)
    if drift == "parent":
        task._data.parent = "f" * 32
    elif drift == "source":
        task._data.script.diff += "\n# remote drift\n"
    elif drift == "unknown_tag":
        task._data.tags.append("unknown-third-state")
    elif drift == "duplicate_tag":
        task._data.tags.append(task._data.tags[0])
    elif drift == "configuration":
        task._configuration_objects["unexpected"] = "drift"
    elif drift == "parameter":
        task._parameters["Args/poll_seconds"] = "61.0"
    elif drift == "artifact":
        task._data.execution.artifacts[0].value["hash"] = "0" * 64
    elif drift == "requirements":
        task._data.script.requirements["pip"] = [
            deploy.SERVICE_REQUIREMENTS[0],
            "unexpected==1",
        ]
    else:  # pragma: no cover - parametrization owns this inventory
        raise AssertionError(drift)
    with pytest.raises(deploy.DeploymentError, match=message):
        deploy._deploy_successors_from_completed_provenance(
            _FakeTaskClass, sleeper=lambda _: None
        )
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]


def test_completed_provenance_failure_receipt_is_sealed_without_remote_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_source = "template-source"
    monkeypatch.setattr(deploy, "EVALUATION_SCRIPT_SHA256", _sha(template_source))
    _FakeTaskClass.reset(template_source=template_source)
    task = _install_completed_provenance(monkeypatch)
    task._data.tags.append("unknown-third-state")
    with pytest.raises(deploy.DeploymentError) as caught:
        deploy._execute_completed_successor_receipt(_FakeTaskClass)
    receipt = caught.value.sealed_receipt
    assert receipt["mode"] == "completed_provenance_successor_execute"
    assert receipt["status"] == "failed_closed"
    assert receipt["remote_state_changed"] is False
    assert "created_task_ids" not in receipt["partial_journal"]
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]
    seal = receipt["seal_sha256"]
    unsealed = dict(receipt)
    unsealed.pop("seal_sha256")
    assert deploy._content_sha256(unsealed) == seal


def test_authority_drift_fails_before_any_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_source = "template-source"
    monkeypatch.setattr(deploy, "EVALUATION_SCRIPT_SHA256", _sha(template_source))
    _FakeTaskClass.reset(template_source=template_source)
    _FakeTaskClass.drift_parent_role = "W"
    with pytest.raises(deploy.DeploymentError, match="parent drifted"):
        deploy._deploy_chain(_FakeTaskClass, sleeper=lambda _: None)
    assert not [event for event in _FakeTaskClass.events if event[0] == "enqueue"]


def test_execute_failure_returns_a_sealed_partial_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_source = "template-source"
    monkeypatch.setattr(deploy, "EVALUATION_SCRIPT_SHA256", _sha(template_source))
    _FakeTaskClass.reset(template_source=template_source)
    _FakeTaskClass.drift_parent_role = "W"
    with pytest.raises(deploy.DeploymentError) as caught:
        deploy._execute_receipt(_FakeTaskClass)
    receipt = caught.value.sealed_receipt
    assert receipt["status"] == "failed_closed"
    assert receipt["remote_state_changed"] is True
    assert receipt["partial_journal"]["task_ids"]
    seal = receipt["seal_sha256"]
    unsealed = dict(receipt)
    unsealed.pop("seal_sha256")
    assert deploy._content_sha256(unsealed) == seal


def _failed_before_enqueue_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object]]:
    template_source = "template-source"
    monkeypatch.setattr(deploy, "EVALUATION_SCRIPT_SHA256", _sha(template_source))
    _FakeTaskClass.reset(template_source=template_source)
    _FakeTaskClass.enqueue_error = ValueError(
        'Could not find queue named "clearml-services"'
    )
    with pytest.raises(deploy.DeploymentError) as caught:
        deploy._execute_receipt(_FakeTaskClass)
    receipt = caught.value.sealed_receipt
    assert receipt["status"] == "failed_closed"
    assert "enqueued_roles" not in receipt["partial_journal"]
    path = tmp_path / "failed-deployment.json"
    _write_json(path, receipt)
    _FakeTaskClass.enqueue_error = None
    _FakeTaskClass.events.clear()
    return path, receipt


def test_resume_rejects_invalid_seal_without_create_or_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, receipt = _failed_before_enqueue_receipt(tmp_path, monkeypatch)
    tampered = dict(receipt)
    tampered["seal_sha256"] = "0" * 64
    path = tmp_path / "invalid-seal.json"
    _write_json(path, tampered)
    with pytest.raises(deploy.DeploymentError, match="seal mismatch"):
        deploy._resume_failed_deployment(
            _FakeTaskClass, failed_receipt_path=path, sleeper=lambda _: None
        )
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]


def test_resume_rejects_remote_drift_without_create_or_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, receipt = _failed_before_enqueue_receipt(tmp_path, monkeypatch)
    watcher_id = receipt["partial_journal"]["task_ids"]["W"]
    _FakeTaskClass.registry[watcher_id]._data.parent = "f" * 32
    with pytest.raises(deploy.DeploymentError, match="parent drifted"):
        deploy._resume_failed_deployment(
            _FakeTaskClass, failed_receipt_path=path, sleeper=lambda _: None
        )
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]


def test_resume_rejects_resealed_bundle_journal_drift_without_remote_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, receipt = _failed_before_enqueue_receipt(tmp_path, monkeypatch)
    tampered = json.loads(json.dumps(receipt))
    tampered["partial_journal"]["sources"]["S"]["standalone_bundle"][
        "embedded_modules"
    ][0]["base_sha256"] = "0" * 64
    tampered = deploy._seal(tampered)
    path = tmp_path / "drifted-journal.json"
    _write_json(path, tampered)
    with pytest.raises(deploy.DeploymentError, match="sources drifted"):
        deploy._resume_failed_deployment(
            _FakeTaskClass, failed_receipt_path=path, sleeper=lambda _: None
        )
    assert not [
        event for event in _FakeTaskClass.events if event[0] in {"create", "enqueue"}
    ]


def test_valid_resume_reuses_five_ids_and_enqueues_without_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, failed = _failed_before_enqueue_receipt(tmp_path, monkeypatch)
    # Match the first real failed receipt: it predates the corrected queue pin.
    failed = json.loads(json.dumps(failed))
    failed["fixed_bindings"]["services_queue"] = "clearml-services"
    failed["fixed_bindings"].pop("services_queue_id")
    failed = deploy._seal(failed)
    _write_json(path, failed)
    source_bytes = path.read_bytes()
    receipt = deploy._resume_failed_deployment(
        _FakeTaskClass, failed_receipt_path=path, sleeper=lambda _: None
    )
    task_ids = failed["partial_journal"]["task_ids"]
    assert receipt["mode"] == "recovery_execute"
    assert receipt["status"] == "deployed"
    assert receipt["task_ids"] == task_ids
    assert receipt["recovery"]["reused_task_ids"] == task_ids
    assert receipt["recovery"]["shell_creation_performed"] is False
    assert receipt["recovery"]["queue_name"] == "services"
    assert receipt["recovery"]["queue_id"] == deploy.SERVICES_QUEUE_ID
    assert not [event for event in _FakeTaskClass.events if event[0] == "create"]
    assert [event[1] for event in _FakeTaskClass.events if event[0] == "enqueue"] == [
        deploy.ROLE_DEFINITIONS[role]["task_name"] for role in deploy.ROLE_ORDER
    ]
    assert path.read_bytes() == source_bytes
    assert all(
        _FakeTaskClass.registry[task_id].status == "queued"
        for task_id in task_ids.values()
    )
    seal = receipt["seal_sha256"]
    unsealed = dict(receipt)
    unsealed.pop("seal_sha256")
    assert deploy._content_sha256(unsealed) == seal


def test_resume_cli_requires_explicit_execute(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        deploy.main(["--resume-failed-receipt", str(tmp_path / "missing.json")])
    assert caught.value.code == 2


def test_successor_execute_requires_evaluation_recovery_receipt() -> None:
    with pytest.raises(
        deploy.DeploymentError, match="evaluation-recovery-receipt is required"
    ):
        deploy.main(
            [
                "--execute",
                "--from-completed-provenance-task-id",
                deploy.COMPLETED_PROVENANCE_TASK_ID,
            ]
        )


def test_successor_dry_run_binds_validated_evaluation_recovery_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(
        deploy,
        "_validate_evaluation_recovery_receipt",
        lambda path: binding if path == recovery_path else None,
    )
    monkeypatch.setattr(
        deploy,
        "_completed_successor_dry_run_receipt",
        lambda **_kwargs: deploy._seal(
            {
                "schema_version": 1,
                "receipt_type": "test",
                "mode": "completed_provenance_successor_dry_run",
                "status": "planned",
                "remote_state_changed": False,
            }
        ),
    )
    destination = tmp_path / "plan.json"

    assert (
        deploy.main(
            [
                "--from-completed-provenance-task-id",
                deploy.COMPLETED_PROVENANCE_TASK_ID,
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
    seal = receipt.pop("seal_sha256")
    assert deploy._content_sha256(receipt) == seal


def _amendment_binding() -> dict[str, object]:
    return {
        "producer_task_id": "a" * 32,
        "receipt_seal_sha256": "1" * 64,
        "producer_source_sha256": "2" * 64,
        "revised_plan_seal_sha256": "3" * 64,
        "amendment_evidence_seal_sha256": "4" * 64,
        "worker_stats_evidence_seal_sha256": "5" * 64,
        "evaluation_task_ids_sha256": "6" * 64,
    }


def _runtime_recovery_binding() -> dict[str, object]:
    amendment = _amendment_binding()
    return {
        "receipt_seal_sha256": "7" * 64,
        "attempt_receipt_seal_sha256": "8" * 64,
        "formal_plan_amendment_receipt_seal_sha256": amendment[
            "receipt_seal_sha256"
        ],
        "formal_plan_amendment_producer_task_id": amendment["producer_task_id"],
        "revised_formal_plan_seal_sha256": amendment[
            "revised_plan_seal_sha256"
        ],
        "exact_evaluation_task_ids_sha256": amendment[
            "evaluation_task_ids_sha256"
        ],
        "ffnet_created_unqueued": True,
        "candidate_services": True,
        "same_ids_only": True,
        "replacement_tasks_created": False,
        "coformer_mutated": False,
        "exact_evaluation_task_count": 28,
        "target_task_ids": {
            "FFNet": deploy.plan_amendment.FFNET_EVALUATION_TASK_ID,
            "C": deploy.CANDIDATE_CONTROLLER_TASK_ID,
        },
        "ffnet_authority": {
            "source_sha256": "9" * 64,
            "parameters_sha256": "a" * 64,
        },
        "candidate_authority": {
            "source_sha256": "b" * 64,
            "parameters_sha256": "c" * 64,
        },
    }


def test_amended_parameters_pin_new_plan_and_runtime_into_every_successor() -> None:
    amendment = _amendment_binding()
    runtime = _runtime_recovery_binding()
    parameters = deploy._parameters(
        deploy.SUCCESSOR_DRY_RUN_IDS,
        {role: _sha(role) for role in deploy.ROLE_ORDER},
        authoritative_recovery=True,
        amendment_binding=amendment,
        runtime_recovery_binding=runtime,
    )
    for role in deploy.SUCCESSOR_ROLE_ORDER:
        assert parameters[role][
            "Args/evaluation_plan_amendment_producer_task_id"
        ] == amendment["producer_task_id"]
        assert parameters[role][
            "Args/evaluation_plan_amendment_revised_plan_seal_sha256"
        ] == amendment["revised_plan_seal_sha256"]
        assert parameters[role][
            "Args/exact_eval_runtime_recovery_receipt_seal_sha256"
        ] == runtime["receipt_seal_sha256"]
    assert parameters["W"][
        "Args/authoritative_evaluation_plan_producer_task_id"
    ] == amendment["producer_task_id"]
    assert parameters["W"][
        "Args/authoritative_evaluation_plan_seal_sha256"
    ] == amendment["revised_plan_seal_sha256"]


def test_amended_parameters_reject_runtime_plan_authority_drift() -> None:
    runtime = _runtime_recovery_binding()
    runtime["revised_formal_plan_seal_sha256"] = "f" * 64
    with pytest.raises(deploy.DeploymentError, match="authority drifted"):
        deploy._parameters(
            deploy.SUCCESSOR_DRY_RUN_IDS,
            {role: _sha(role) for role in deploy.ROLE_ORDER},
            authoritative_recovery=True,
            amendment_binding=_amendment_binding(),
            runtime_recovery_binding=runtime,
        )


def test_amended_runtime_pair_rejects_old_unknown_candidate_id() -> None:
    assert deploy.CANDIDATE_CONTROLLER_TASK_ID == (
        "b6f0fbab32a5478183a45b3ca833fc01"
    )
    runtime = _runtime_recovery_binding()
    runtime["target_task_ids"] = {
        "FFNet": deploy.plan_amendment.FFNET_EVALUATION_TASK_ID,
        "C": "b6f0a1ae538f48dfafba75ba080c2197",
    }
    with pytest.raises(deploy.DeploymentError, match="bindings drifted"):
        deploy._validate_amended_runtime_pair(_amendment_binding(), runtime)
