from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PRODUCER_PATH = ROOT / "tools/resilient_v2x/clearml_formal_training_provenance.py"
WATCHER_PATH = ROOT / "tools/resilient_v2x/clearml_1337_dependency_watcher.py"
AUDIT_PATH = ROOT / "tools/resilient_v2x/clearml_formal_comparability_audit.py"
PRODUCER_TEST_PATH = (
    ROOT / "tests/resilient_v2x/test_clearml_formal_training_provenance.py"
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chain(monkeypatch) -> SimpleNamespace:
    producer = _load(PRODUCER_PATH, "integration_training_provenance")
    producer_test = _load(PRODUCER_TEST_PATH, "integration_training_provenance_test")
    world = producer_test._world(producer, monkeypatch)
    source_controller_id = producer.TRAINING_CONTROLLER_TASK_ID
    successor_controller_id = "a" * 32
    source_controller = world.controller
    source_progress = copy.deepcopy(world.progress_artifact.value)
    recovery, target_identity, _steps = producer_test._schema4_recovery_fixture(
        producer,
        monkeypatch,
    )
    entries_by_subject = {
        entry["subject"]: entry for entry in world.manifest_artifact.value["entries"]
    }
    steps_by_subject = {
        step["experiment"]: step for step in source_progress["steps"]
    }
    for subject in producer.SOURCE_REVISION_SUBJECTS["new"]:
        predecessor = producer.SOURCE_REVISION_TARGET_PREDECESSOR_TASK_ID
        entries_by_subject[subject]["training_predecessor_task_id"] = predecessor
        steps_by_subject[subject]["predecessor_task_id"] = predecessor
        training_task = world.tasks[
            entries_by_subject[subject]["training_task_id"]
        ]
        training_task.parameters["Args/predecessor_task_id"] = predecessor
        training_task.data.hyperparams = training_task.parameters
        world.contracts[subject].value["predecessor_task_id"] = predecessor
    world.manifest_artifact.value.pop("seal_sha256", None)
    world.manifest_artifact.value = producer._sealed(
        world.manifest_artifact.value
    )
    source_progress.pop("seal_sha256", None)
    source_progress = producer._sealed(source_progress)
    world.progress_artifact.value = copy.deepcopy(source_progress)
    for subject in producer.SOURCE_REVISION_SUBJECTS["new"]:
        for key in (
            "adopted_task_ids",
            "adopted_predecessor_task_ids",
            "adopted_task_template_roles",
            "adopted_task_binding_receipts",
        ):
            recovery[key].pop(subject, None)
    recovery["source_progress_revision"] = source_progress["revision"]
    recovery["source_progress_seal_sha256"] = source_progress["seal_sha256"]
    recovery["source_recovery_chain"] = {
        "source_controller_task_id": source_controller_id,
        "source_progress_revision": source_progress["revision"],
        "source_progress_seal_sha256": source_progress["seal_sha256"],
        "source_recovery_sha256": producer._content_sha256(
            source_progress["recovery"]
        ),
    }
    successor_progress = copy.deepcopy(source_progress)
    successor_progress.update(
        {
            "controller_task_id": successor_controller_id,
            "revision": source_progress["revision"] + 1,
            "recovery": recovery,
            "template": target_identity,
        }
    )
    successor_progress.pop("seal_sha256", None)
    successor_progress = producer._sealed(successor_progress)
    successor_progress_artifact = producer_test._Artifact(successor_progress)
    source_controller.status = "failed"
    source_controller.data.status = "failed"
    successor_controller = producer_test._Task(
        successor_controller_id,
        status="completed",
        parent=producer_test.GATE_TASK_ID,
        entry_point="clearml_5090_training_controller.py",
        source="successor controller source",
        artifacts={
            producer.PROGRESS_ARTIFACT: successor_progress_artifact,
            producer.TRAINING_MANIFEST_ARTIFACT: world.manifest_artifact,
        },
        name="successor controller",
    )
    world.tasks[successor_controller_id] = successor_controller
    world.controller = successor_controller
    world.progress_artifact = successor_progress_artifact
    world.args.training_controller_task_id = successor_controller_id
    world.output.parent = successor_controller_id
    world.output.data.parent = successor_controller_id
    world.output.parameters["Args/training_controller_task_id"] = (
        successor_controller_id
    )
    world.output.data.hyperparams = world.output.parameters
    for subject in producer.SOURCE_REVISION_SUBJECTS["new"]:
        training_task = world.tasks[
            entries_by_subject[subject]["training_task_id"]
        ]
        training_task.parent = successor_controller_id
        training_task.data.parent = successor_controller_id
    world.args.poll_seconds = 60.0
    world.args.timeout_hours = 720.0
    world.output.parameters["Args/poll_seconds"] = "60.0"
    world.output.parameters["Args/timeout_hours"] = "720.0"
    payload = producer_test._run(world)
    world.output.status = "completed"
    world.output.data.status = "completed"

    class _TaskClass:
        @staticmethod
        def get_task(*, task_id: str):
            if task_id == world.output.id:
                return world.output
            return world.tasks[task_id]

    watcher = _load(WATCHER_PATH, "integration_dependency_watcher")
    monkeypatch.setattr(
        watcher,
        "EXPECTED_LEGACY_TRAINING_SCRIPT_SHA256",
        producer.LEGACY_BOOTSTRAP_SHA256,
    )
    monkeypatch.setattr(
        watcher,
        "EXPECTED_CANONICAL_TRAINING_SCRIPT_SHA256",
        producer.EXPANDED_BOOTSTRAP_SHA256,
    )
    monkeypatch.setattr(
        watcher,
        "EXPECTED_DOCKER_COMMAND_SHA256",
        producer.DOCKER_COMMAND_SHA256,
    )
    monkeypatch.setattr(
        watcher,
        "EXPECTED_QUEUE_IDS",
        {name: "4" * 32 for name in watcher.EXPECTED_QUEUE_IDS},
    )

    audit = _load(AUDIT_PATH, "integration_comparability_audit")
    monkeypatch.setattr(
        audit,
        "LEGACY_TRAINING_SCRIPT_SHA256",
        producer.LEGACY_BOOTSTRAP_SHA256,
    )
    monkeypatch.setattr(
        audit,
        "CANONICAL_TRAINING_SCRIPT_SHA256",
        producer.EXPANDED_BOOTSTRAP_SHA256,
    )
    monkeypatch.setattr(audit, "DOCKER_COMMAND_SHA256", producer.DOCKER_COMMAND_SHA256)
    monkeypatch.setattr(
        audit, "LEGACY_PARENT_TASK_ID", producer_test.SOURCE_CONTROLLER_ID
    )
    monkeypatch.setattr(
        audit, "RECOVERY_PARENT_TASK_ID", producer_test.RECOVERY_CONTROLLER_ID
    )
    monkeypatch.setattr(
        audit, "NO_DISTILLATION_PARENT_TASK_ID", producer_test.PARENT_CONTROLLER_ID
    )
    monkeypatch.setattr(
        audit,
        "SOURCE_REVISION_TRANSITION_SEAL_SHA256",
        producer.SOURCE_REVISION_TRANSITION_SEAL_SHA256,
    )

    manifest = world.manifest_artifact.value
    training = watcher._parse_training_manifest(json.dumps(manifest))
    entries, manifest_seal, _ = audit._validate_manifest(manifest)
    script_by_subject = {
        record["subject"]: record["raw_bootstrap_script_sha256"]
        for record in payload["training_tasks"]
    }
    script_equivalence = audit._verify_training_script_equivalence(
        {
            producer.LEGACY_BOOTSTRAP_SHA256: world.legacy,
            producer.EXPANDED_BOOTSTRAP_SHA256: world.expanded,
        },
        script_by_subject,
    )
    training_records = [
        {
            "subject": record["subject"],
            "script_sha256": record["raw_bootstrap_script_sha256"],
            "run_contract_sha256": record["run_contract_content_sha256"],
            "final_checkpoint_contract_content_sha256": record[
                "final_checkpoint_contract_content_sha256"
            ],
            "common_teacher_initialization_audit_sha256": record[
                "common_teacher_initialization_audit_content_sha256"
            ],
            "execution_queue_id": record["execution_queue_id"],
            "last_worker": record["last_worker"],
            "raw_authority_sha256": record["raw_authority_sha256"],
        }
        for record in payload["training_tasks"]
    ]
    return SimpleNamespace(
        producer=producer,
        watcher=watcher,
        audit=audit,
        world=world,
        payload=payload,
        manifest=manifest,
        training=training,
        entries=entries,
        manifest_seal=manifest_seal,
        script_equivalence=script_equivalence,
        training_records=training_records,
        task_class=_TaskClass,
        controller_task_id=successor_controller_id,
    )


def _reseal(producer: ModuleType, payload: dict[str, object]) -> None:
    payload.pop("seal_sha256", None)
    sealed = producer._sealed(payload)
    payload.clear()
    payload.update(sealed)


def _watcher_validate(chain: SimpleNamespace, payload: dict[str, object]):
    artifact = chain.world.output.artifacts[chain.producer.PROVENANCE_ARTIFACT]
    artifact.value = copy.deepcopy(payload)
    return chain.watcher._consume_training_provenance(
        task_class=chain.task_class,
        provenance_task_id=chain.world.output.id,
        expected_provenance_script_sha256=hashlib.sha256(
            chain.producer._runtime_source().encode("utf-8")
        ).hexdigest(),
        controller=chain.world.controller,
        controller_task_id=chain.controller_task_id,
        manifest=chain.manifest,
        training=chain.training,
    )


def _audit_validate(chain: SimpleNamespace, payload: dict[str, object]):
    return chain.audit._validate_training_provenance_payload(
        payload,
        controller_task_id=chain.controller_task_id,
        progress=chain.world.progress_artifact.value,
        progress_seal=chain.world.progress_artifact.value["seal_sha256"],
        recovery_sha256=chain.payload["recovery_contract_sha256"],
        manifest=chain.manifest,
        manifest_seal=chain.manifest_seal,
        training_entries=chain.entries,
        script_equivalence=chain.script_equivalence,
        training_records=chain.training_records,
    )


def _assert_both_reject(chain: SimpleNamespace, payload: dict[str, object]) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        _watcher_validate(chain, payload)
    with pytest.raises((RuntimeError, ValueError)):
        _audit_validate(chain, payload)


def test_real_producer_payload_is_accepted_by_watcher_and_audit(chain) -> None:
    policies = _watcher_validate(chain, chain.payload)
    audit_seal = _audit_validate(chain, chain.payload)

    assert list(policies) == list(chain.producer.SUBJECT_ORDER)
    assert len(policies) == len(chain.payload["training_tasks"]) == 26
    assert audit_seal == chain.payload["seal_sha256"]
    assert [
        record["training_task_id"] for record in chain.payload["training_tasks"]
    ] == [entry["training_task_id"] for entry in chain.manifest["entries"]]


def test_downstream_rejects_raw_and_normalized_script_sha_exchange(chain) -> None:
    payload = copy.deepcopy(chain.payload)
    record = payload["training_tasks"][10]
    (
        record["raw_bootstrap_script_sha256"],
        record["normalized_task_script_identity_sha256"],
    ) = (
        record["normalized_task_script_identity_sha256"],
        record["raw_bootstrap_script_sha256"],
    )
    _reseal(chain.producer, payload)

    _assert_both_reject(chain, payload)


def test_downstream_rejects_deep_recovery_seal_drift(chain) -> None:
    payload = copy.deepcopy(chain.payload)
    support = payload["training_tasks"][
        list(chain.producer.SUBJECT_ORDER).index("support_residual")
    ]
    support["recovery_lineage"][1]["progress_seal_sha256"] = "1" * 64
    _reseal(chain.producer, payload)

    _assert_both_reject(chain, payload)


def test_downstream_rejects_task_binding_swap(chain) -> None:
    payload = copy.deepcopy(chain.payload)
    left, right = payload["training_tasks"][10:12]
    left["training_task_id"], right["training_task_id"] = (
        right["training_task_id"],
        left["training_task_id"],
    )
    _reseal(chain.producer, payload)

    _assert_both_reject(chain, payload)


def test_downstream_rejects_parent_binding_swap(chain) -> None:
    payload = copy.deepcopy(chain.payload)
    records = payload["training_tasks"]
    support = records[list(chain.producer.SUBJECT_ORDER).index("support_residual")]
    ffnet = records[list(chain.producer.SUBJECT_ORDER).index("ffnet")]
    support["parent_controller_task_id"], ffnet["parent_controller_task_id"] = (
        ffnet["parent_controller_task_id"],
        support["parent_controller_task_id"],
    )
    _reseal(chain.producer, payload)

    _assert_both_reject(chain, payload)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "reorder"])
def test_downstream_rejects_recursive_progress_chain_drift(chain, mutation) -> None:
    payload = copy.deepcopy(chain.payload)
    recursive = payload["recursive_progress_chain"]
    if mutation == "duplicate":
        recursive[2]["task_id"] = recursive[1]["task_id"]
    elif mutation == "missing":
        recursive.pop()
    else:
        recursive[1], recursive[2] = recursive[2], recursive[1]
    _reseal(chain.producer, payload)

    _assert_both_reject(chain, payload)


def test_downstream_rejects_legacy_script_for_wrong_subject(chain) -> None:
    payload = copy.deepcopy(chain.payload)
    records = payload["training_tasks"]
    ffnet = records[list(chain.producer.SUBJECT_ORDER).index("ffnet")]
    bootstrap = payload["bootstrap_equivalence"]
    ffnet["raw_bootstrap_script_sha256"] = chain.producer.LEGACY_BOOTSTRAP_SHA256
    ffnet["script_equivalence_class"] = "legacy_nested_teacher_membership"
    bootstrap["legacy_script_subjects"].append("ffnet")
    bootstrap["expanded_script_subjects"].remove("ffnet")
    _reseal(chain.producer, payload)

    _assert_both_reject(chain, payload)


def test_downstream_rejects_docker_contract_drift(chain) -> None:
    payload = copy.deepcopy(chain.payload)
    payload["training_tasks"][10]["docker_command_sha256"] = "2" * 64
    _reseal(chain.producer, payload)

    _assert_both_reject(chain, payload)


def test_downstream_rejects_runtime_guard_drift(chain) -> None:
    payload = copy.deepcopy(chain.payload)
    payload["bootstrap_equivalence"]["tf32_override"] = "1"
    payload["capacity_matched_hardware"]["tf32_override"] = "1"
    _reseal(chain.producer, payload)

    _assert_both_reject(chain, payload)


def test_watcher_rejects_provenance_artifact_toctou(chain) -> None:
    artifact = chain.world.output.artifacts[chain.producer.PROVENANCE_ARTIFACT]
    reads = 0

    def mutate_on_terminal_read(current) -> None:
        nonlocal reads
        reads += 1
        if reads == 3:
            changed = copy.deepcopy(current.value)
            changed["passed"] = False
            _reseal(chain.producer, changed)
            current.value = changed

    artifact.on_get = mutate_on_terminal_read

    with pytest.raises(RuntimeError, match="changed before evaluation planning"):
        _watcher_validate(chain, chain.payload)
