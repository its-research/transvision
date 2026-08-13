from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared
from tools.resilient_v2x import prepare_clearml_sota_round2_p2 as p2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _p0_snapshot() -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(p2.P0_OUTPUT_DIR).as_posix(): (
            path.stat().st_size,
            _sha256(path),
        )
        for path in sorted(p2.P0_OUTPUT_DIR.iterdir())
        if path.is_file()
    }


def _prepare(output_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    assert p2.main(["prepare", "--output-dir", str(output_dir)]) == 0
    manifest = json.loads(
        (output_dir / p2.PACKAGE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    plan = json.loads(
        (output_dir / p2.DEPLOYMENT_PLAN_NAME).read_text(encoding="utf-8")
    )
    with shared.use_deployment_profile(p2.P2_PROFILE):
        assert shared._verify_package(output_dir) == manifest
        observed, static = shared._verify_deployment_plan(output_dir, manifest)
    assert observed == plan
    assert plan["static_plan_seal_sha256"] == static["seal_sha256"]
    return manifest, plan


def test_p2_profile_is_isolated_and_explicitly_derived_from_sealed_p0() -> None:
    p0_manifest = p2._verified_p0_manifest()
    profile = p2.P2_PROFILE

    assert shared._profile() is shared.P0_PROFILE
    assert profile.bootstrap_parent is shared.P0_PROFILE
    assert profile.output_dir != shared.P0_PROFILE.output_dir
    assert profile.package_manifest_name != shared.P0_PROFILE.package_manifest_name
    assert profile.deployment_plan_name != shared.P0_PROFILE.deployment_plan_name
    assert profile.source_transition_artifact != shared.SOURCE_TRANSITION_ARTIFACT
    assert profile.launch_receipt_artifact != shared.LAUNCH_RECEIPT_ARTIFACT
    assert profile.upload_token != shared.UPLOAD_TOKEN
    assert profile.template_token != shared.TEMPLATE_TOKEN
    assert profile.launch_token != shared.LAUNCH_TOKEN
    assert profile.base_source_dataset_id == p2.P0_SOURCE_DATASET_ID
    assert profile.base_source_package == p0_manifest["target_source_package"]
    assert profile.provenance == {
        "derivation": "P2 derived from P0",
        "parent_candidate": p0_manifest["candidate"],
        "parent_source_dataset_id": p2.P0_SOURCE_DATASET_ID,
        "parent_source_package_manifest_seal_sha256": p0_manifest["seal_sha256"],
        "parent_source_tree_sha256": p0_manifest["target_source_package"][
            "tree_sha256"
        ],
        "parent_template_task_id": p2.P0_TEMPLATE_TASK_ID,
        "parent_training_task_id": p2.P0_TRAINING_TASK_ID,
    }


def test_local_prepare_is_exactly_p0_plus_p2_and_never_mutates_p0(
    tmp_path: Path,
) -> None:
    before = _p0_snapshot()
    output_dir = tmp_path / "p2"
    manifest, plan = _prepare(output_dir)
    after = _p0_snapshot()
    p0_manifest = p2._verified_p0_manifest()
    p0_inventory = json.loads(
        (p2.P0_OUTPUT_DIR / "source-inventory.json").read_text(encoding="utf-8")
    )
    p2_inventory = json.loads(
        (output_dir / "source-inventory.json").read_text(encoding="utf-8")
    )

    assert before == after
    assert manifest["base_source_dataset_id"] == p2.P0_SOURCE_DATASET_ID
    assert manifest["base_source_package"] == p0_manifest["target_source_package"]
    assert manifest["provenance"]["derivation"] == "P2 derived from P0"
    assert manifest["delta"]["modified_files"] == []
    assert manifest["delta"]["removed_files"] == []
    assert [item["path"] for item in manifest["delta"]["added_files"]] == [p2.P2_CONFIG]
    assert p2.P2_CONFIG not in {item["path"] for item in p0_inventory["files"]}
    assert {item["path"] for item in p2_inventory["files"]} == {
        item["path"] for item in p0_inventory["files"]
    } | {p2.P2_CONFIG}
    assert p2_inventory["file_count"] == p0_inventory["file_count"] + 1
    assert p2_inventory["source_bytes"] == (
        p0_inventory["source_bytes"] + Path(shared.ROOT / p2.P2_CONFIG).stat().st_size
    )
    assert plan["remote_state_changed"] is False
    shared._require_seal(manifest, context="test P2 package")
    shared._require_seal(plan, context="test P2 deployment")


def test_p2_package_is_byte_deterministic_across_output_directories(
    tmp_path: Path,
) -> None:
    first_manifest, _ = _prepare(tmp_path / "first")
    second_manifest, _ = _prepare(tmp_path / "second")

    assert first_manifest == second_manifest
    package = first_manifest["target_source_package"]
    archive_name = package["archive_name"]
    for name in (archive_name, "source-inventory.json", p2.PACKAGE_MANIFEST_NAME):
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()


def test_p2_plan_transition_and_parameters_pin_the_p0_protocol(tmp_path: Path) -> None:
    manifest, plan = _prepare(tmp_path / "p2")
    target_dataset_id = "a" * 32
    with shared.use_deployment_profile(p2.P2_PROFILE):
        transition = shared._transition(manifest, dataset_id=target_dataset_id)
        parameters = shared._task_parameters(manifest, dataset_id=target_dataset_id)

    assert plan["candidate"] == {
        "experiment": p2.P2_EXPERIMENT,
        "config": p2.P2_CONFIG,
        "identity": p2.P2_IDENTITY,
    }
    assert plan["bindings"] == {
        "base_source_dataset_id": p2.P0_SOURCE_DATASET_ID,
        "base_template_task_id": shared.fastlane.TEMPLATE_TASK_ID,
        "predecessor_task_id": shared.fastlane.PREDECESSOR_TASK_ID,
        "teacher_task_id": shared.fastlane.TEACHER_TASK_ID,
        "teacher_model_id": shared.fastlane.TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": shared.fastlane.TEACHER_CHECKPOINT_SHA256,
        "training_dataset_id": shared.fastlane.TRAINING_DATASET_ID,
        "training_project": shared.TRAINING_PROJECT,
        "training_project_id": shared.TRAINING_PROJECT_ID,
    }
    assert plan["protocol"] == {
        "worker_queue": "GPU4-A100",
        "worker_queue_id": p2.WORKER_QUEUE_ID,
        "global_batch_size": 8,
        "gpu_count": 4,
        "batch_size_per_gpu": 2,
        "max_epochs": 50,
        "val_interval": 10,
        "training_seed": 20250218,
        "precision": "FP32",
    }
    assert plan["execution_policy"] == {
        "executor_count": 1,
        "duplicate_guard": (
            "exact task name plus predecessor parent before clone, followed by "
            "forced launch-receipt readback and a unique exact-name+parent "
            "current-clone ID guard immediately before enqueue"
        ),
        "concurrent_launchers_supported": False,
        "duplicate_guard_stages": [
            "pre_clone_requires_zero_exact_name_plus_parent_matches",
            (
                "post_receipt_readback_pre_enqueue_requires_one_exact_name_plus_"
                "parent_match_with_current_clone_id"
            ),
        ],
    }
    assert plan["provenance"] == manifest["provenance"]
    assert plan["early_gate_policy"] == p2.EARLY_GATE_POLICY
    shared._require_seal(
        plan["early_gate_policy"], context="test P2 E10 early-gate policy"
    )
    assert plan["early_gate_policy"]["declared_training_protocol"] == {
        "max_epochs": 50,
        "val_interval": 10,
        "protocol_unchanged_by_gate": True,
    }
    assert all(
        "prepare_clearml_sota_round2_p2.py" in command
        for command in plan["commands"].values()
    )
    assert p2.UPLOAD_TOKEN in plan["commands"]["1_upload_source"]
    assert p2.TEMPLATE_TOKEN in plan["commands"]["2_create_template"]
    assert p2.LAUNCH_TOKEN in plan["commands"]["3_launch_a100"]
    assert "--queue GPU4-A100" in plan["commands"]["3_launch_a100"]

    assert transition["base_source"]["dataset_id"] == p2.P0_SOURCE_DATASET_ID
    assert transition["target_source"]["dataset_id"] == target_dataset_id
    assert transition["inventory_delta"]["modified_file_count"] == 0
    assert transition["inventory_delta"]["added_file_count"] == 1
    assert transition["inventory_delta"]["removed_file_count"] == 0
    assert transition["provenance"]["derivation"] == "P2 derived from P0"
    assert transition["early_gate_policy"] == p2.EARLY_GATE_POLICY
    assert transition["protocol"] == {
        "global_batch_size": 8,
        "gpu_count": 4,
        "batch_size_per_gpu": 2,
        "max_epochs": 50,
        "val_interval": 10,
        "training_seed": 20250218,
        "training_overlay_protocol_seed": 20250218,
        "precision": "FP32",
    }
    shared._require_seal(transition, context="test P2 transition")
    assert parameters["Args/source_dataset_id"] == target_dataset_id
    assert parameters["Args/experiment_from_task"] == p2.P2_EXPERIMENT
    assert parameters["Args/predecessor_task_id"] == shared.fastlane.PREDECESSOR_TASK_ID
    assert parameters["Args/teacher_task_id"] == shared.fastlane.TEACHER_TASK_ID
    assert parameters["Args/teacher_model_id"] == shared.fastlane.TEACHER_MODEL_ID
    assert parameters["Args/teacher_checkpoint_sha256"] == (
        shared.fastlane.TEACHER_CHECKPOINT_SHA256
    )
    assert parameters["Args/training_seed"] == 20250218
    assert parameters["Args/gpus"] == 4
    assert parameters["Args/max_epochs"] == 50
    assert parameters["Args/amp"] is False


def test_e10_gate_receipt_defaults_to_continue_and_never_changes_protocol() -> None:
    receipt = p2.build_e10_early_gate_receipt(
        plan_seal_sha256="a" * 64,
        p0_task_id="b" * 32,
        p0_bev_ap70=61.0,
        p0_3d_ap70=32.0,
        p2_task_id="c" * 32,
        p2_bev_ap70=61.2,
        p2_3d_ap70=32.3,
        created_at="2026-08-12T00:00:00+00:00",
    )

    assert receipt["policy_seal_sha256"] == p2.EARLY_GATE_POLICY["seal_sha256"]
    assert receipt["decision"] == {
        "p0_significantly_below_resilient": False,
        "p2_significantly_below_resilient": False,
        "both_candidates_below_reference": False,
        "p2_regresses_from_p0": False,
        "runtime_regression": False,
        "stop_authorized": False,
        "action": "continue_to_epoch_50",
        "declared_training_protocol_changed": False,
    }
    shared._require_seal(receipt, context="test continue E10 gate receipt")


@pytest.mark.parametrize(
    ("metrics", "runtime_regression", "trigger"),
    (
        ((59.5, 30.5, 59.8, 30.8), "", "both_candidates_below_reference"),
        ((61.0, 33.0, 60.4, 32.4), "", "p2_regresses_from_p0"),
        ((61.0, 33.0, 61.1, 33.1), "non-finite loss at E10", "runtime_regression"),
    ),
)
def test_e10_gate_receipt_authorizes_only_an_explicit_stop_branch(
    metrics: tuple[float, float, float, float],
    runtime_regression: str,
    trigger: str,
) -> None:
    receipt = p2.build_e10_early_gate_receipt(
        plan_seal_sha256="a" * 64,
        p0_task_id="b" * 32,
        p0_bev_ap70=metrics[0],
        p0_3d_ap70=metrics[1],
        p2_task_id="c" * 32,
        p2_bev_ap70=metrics[2],
        p2_3d_ap70=metrics[3],
        runtime_regression=runtime_regression,
        created_at="2026-08-12T00:00:00+00:00",
    )

    assert receipt["decision"][trigger] is True
    assert receipt["decision"]["stop_authorized"] is True
    assert receipt["decision"]["action"] == "stop_after_epoch_10_validation"
    assert receipt["decision"]["declared_training_protocol_changed"] is False
    shared._require_seal(receipt, context=f"test {trigger} E10 gate receipt")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("plan_seal_sha256", "A" * 64, "lowercase SHA-256"),
        ("p0_task_id", "short", "P0 task"),
        ("p2_bev_ap70", float("nan"), "finite and in"),
    ),
)
def test_e10_gate_receipt_rejects_unverifiable_inputs(
    field: str, value: object, message: str
) -> None:
    kwargs: dict[str, object] = {
        "plan_seal_sha256": "a" * 64,
        "p0_task_id": "b" * 32,
        "p0_bev_ap70": 61.0,
        "p0_3d_ap70": 32.0,
        "p2_task_id": "c" * 32,
        "p2_bev_ap70": 61.0,
        "p2_3d_ap70": 32.0,
    }
    kwargs[field] = value
    with pytest.raises((ValueError, TypeError), match=message):
        p2.build_e10_early_gate_receipt(**kwargs)


def test_p2_bootstrap_patch_chains_p0_then_adds_only_p2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = """ADDITIONAL_EXPERIMENT_SPECS = (
    ExperimentSpec(
        "linear_no_distillation",
some_other_text = (
        "concat_capacity_matched",
        "resilient_v2x",
)\n"""
    e1_e2_e3 = shared.candidate._apply_candidate_experiment_patch(raw)
    monkeypatch.setattr(
        shared.fastlane,
        "TEMPLATE_SCRIPT_SHA256",
        hashlib.sha256(raw.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        shared.fastlane,
        "PATCHED_TEMPLATE_SCRIPT_SHA256",
        hashlib.sha256(e1_e2_e3.encode()).hexdigest(),
    )
    with shared.use_deployment_profile(p2.P2_PROFILE):
        patched = shared._patch_template(raw)

    assert patched.count(f'"{shared.P0_EXPERIMENT}"') >= 2
    assert patched.count(f'"{shared.P0_CONFIG}"') == 1
    assert patched.count(f'"{p2.P2_EXPERIMENT}"') >= 2
    assert patched.count(f'"{p2.P2_CONFIG}"') == 1
    assert patched.index(f'"{shared.P0_EXPERIMENT}"') < patched.index(
        f'"{p2.P2_EXPERIMENT}"'
    )


@pytest.mark.parametrize(
    "command",
    (
        ["upload-source"],
        ["create-template", "--source-dataset-id", "a" * 32],
        [
            "launch",
            "--source-dataset-id",
            "a" * 32,
            "--template-task-id",
            "b" * 32,
        ],
    ),
)
def test_remote_subcommands_reject_without_distinct_exact_p2_token(
    command: list[str],
) -> None:
    with pytest.raises(PermissionError, match="exact execute token required"):
        p2.main(command)
    assert shared._profile() is shared.P0_PROFILE


def test_p2_launch_duplicate_guard_runs_before_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "seal_sha256": "1" * 64,
        "candidate": {
            "experiment": p2.P2_EXPERIMENT,
            "config": p2.P2_CONFIG,
            "identity": p2.P2_IDENTITY,
        },
        "target_source_package": {
            "archive_name": "source.tar.zst",
            "archive_bytes": 1,
            "archive_sha256": "2" * 64,
        },
        "delta": {"modified_files": [], "removed_files": [], "added_files": []},
    }
    plan = {
        "seal_sha256": "3" * 64,
        "protocol": {"worker_queue": p2.WORKER_QUEUE},
        "execution_policy": {
            "executor_count": 1,
            "duplicate_guard": p2.P2_PROFILE.duplicate_guard_description,
            "concurrent_launchers_supported": False,
            "duplicate_guard_stages": list(p2.P2_PROFILE.duplicate_guard_stages or ()),
        },
    }
    transition = shared._sealed({"transition_type": "test"})
    expected_name = f"{p2.TASK_PREFIX} [{plan['seal_sha256'][:12]}]"
    calls: list[tuple[object, ...]] = []

    class FakeTask:
        @staticmethod
        def get_task(*, task_id: str) -> object:
            calls.append(("get_task", task_id))
            return object()

        @staticmethod
        def get_tasks(
            *, task_name: str, task_filter: dict[str, object]
        ) -> list[object]:
            calls.append(("duplicate_preflight", task_name, task_filter))
            return [SimpleNamespace(name=expected_name)]

        @staticmethod
        def clone(**_kwargs: object) -> object:
            calls.append(("clone",))
            raise AssertionError("duplicate guard must run before clone")

    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeTask))
    monkeypatch.setattr(shared, "_verify_package", lambda _output: manifest)
    monkeypatch.setattr(
        shared, "_verify_deployment_plan", lambda _output, _manifest: ({}, plan)
    )
    monkeypatch.setattr(
        shared, "_downloaded_source_root", lambda *_args, **_kwargs: Path(".")
    )
    monkeypatch.setattr(shared, "_transition", lambda *_args, **_kwargs: transition)
    monkeypatch.setattr(
        shared, "_validated_fixed_base", lambda _task: (object(), "", "")
    )
    monkeypatch.setattr(
        shared, "_validate_round2_template", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        shared, "_require_bootstrap_script", lambda *_args, **_kwargs: "raw"
    )
    monkeypatch.setattr(shared, "_patch_template", lambda _raw: "patched")
    monkeypatch.setattr(shared, "_require_transition_artifact", lambda *_args: None)

    with pytest.raises(RuntimeError, match="refusing duplicate round2 P2 task"):
        p2.main(
            [
                "launch",
                "--source-dataset-id",
                "a" * 32,
                "--template-task-id",
                "b" * 32,
                "--queue",
                p2.WORKER_QUEUE,
                "--execute-token",
                p2.LAUNCH_TOKEN,
            ]
        )

    assert any(call[0] == "duplicate_preflight" for call in calls)
    assert not any(call[0] == "clone" for call in calls)
    task_filter = next(call[2] for call in calls if call[0] == "duplicate_preflight")
    assert task_filter == {"parent": shared.fastlane.PREDECESSOR_TASK_ID}
    task_name = next(call[1] for call in calls if call[0] == "duplicate_preflight")
    assert re.fullmatch(task_name, expected_name)
    assert shared._profile() is shared.P0_PROFILE


def test_p2_success_path_builds_candidate_specific_launch_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "seal_sha256": "1" * 64,
        "candidate": {
            "experiment": p2.P2_EXPERIMENT,
            "config": p2.P2_CONFIG,
            "identity": p2.P2_IDENTITY,
        },
        "target_source_package": {
            "archive_name": "source.tar.zst",
            "archive_bytes": 1,
            "archive_sha256": "2" * 64,
        },
        "delta": {"modified_files": [], "removed_files": [], "added_files": []},
    }
    deployment = {"seal_sha256": "3" * 64}
    plan = {
        "seal_sha256": "4" * 64,
        "protocol": {
            "worker_queue": p2.WORKER_QUEUE,
            "worker_queue_id": p2.WORKER_QUEUE_ID,
        },
        "execution_policy": {
            "executor_count": 1,
            "duplicate_guard": p2.P2_PROFILE.duplicate_guard_description,
            "concurrent_launchers_supported": False,
            "duplicate_guard_stages": list(p2.P2_PROFILE.duplicate_guard_stages or ()),
        },
        "early_gate_policy": p2.EARLY_GATE_POLICY,
    }
    transition = shared._sealed({"transition_type": "test"})
    captured: dict[str, object] = {}

    class Clone:
        id = "c" * 32
        name = ""
        status = "created"
        output_uri = ""

        def set_parameters(self, _parameters: dict[str, object]) -> bool:
            return True

        def reload(self) -> None:
            return None

    clone = Clone()

    class FakeTask:
        @staticmethod
        def get_task(*, task_id: str) -> object:
            return SimpleNamespace(id=task_id)

        @staticmethod
        def get_tasks(**_kwargs: object) -> list[object]:
            return []

        @staticmethod
        def clone(**kwargs: object) -> Clone:
            clone.name = str(kwargs["name"])
            return clone

    def capture_enqueue(
        _task_class: object,
        task: Clone,
        receipt: dict[str, object],
        *,
        queue: str,
    ) -> None:
        captured["task"] = task
        captured["receipt"] = receipt
        captured["queue"] = queue
        task.status = "in_progress"

    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeTask))
    monkeypatch.setattr(shared, "_verify_package", lambda _output: manifest)
    monkeypatch.setattr(
        shared,
        "_verify_deployment_plan",
        lambda _output, _manifest: (deployment, plan),
    )
    monkeypatch.setattr(
        shared, "_downloaded_source_root", lambda *_args, **_kwargs: Path(".")
    )
    monkeypatch.setattr(shared, "_transition", lambda *_args, **_kwargs: transition)
    monkeypatch.setattr(
        shared, "_validated_fixed_base", lambda _task: (object(), "", "")
    )
    monkeypatch.setattr(
        shared, "_validate_round2_template", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        shared, "_require_bootstrap_script", lambda *_args, **_kwargs: "raw"
    )
    monkeypatch.setattr(shared, "_patch_template", lambda _raw: "patched")
    monkeypatch.setattr(shared, "_require_transition_artifact", lambda *_args: None)
    monkeypatch.setattr(shared.fastlane, "_edit_script", lambda *_args: None)
    monkeypatch.setattr(
        shared,
        "_validate_training_clone",
        lambda task, **_kwargs: {
            "task_id": task.id,
            "task_name": task.name,
            "parent_task_id": shared.fastlane.PREDECESSOR_TASK_ID,
            "project": shared.TRAINING_PROJECT,
            "project_id": shared.TRAINING_PROJECT_ID,
            "entry_point": shared.fastlane.TEMPLATE_ENTRY_POINT,
            "pre_enqueue_queue_id": None,
            "planned_queue": p2.WORKER_QUEUE,
            "planned_queue_id": p2.WORKER_QUEUE_ID,
        },
    )
    monkeypatch.setattr(shared, "_upload_verify_and_enqueue", capture_enqueue)

    assert (
        p2.main(
            [
                "launch",
                "--source-dataset-id",
                "a" * 32,
                "--template-task-id",
                "b" * 32,
                "--queue",
                p2.WORKER_QUEUE,
                "--execute-token",
                p2.LAUNCH_TOKEN,
            ]
        )
        == 0
    )
    receipt = captured["receipt"]
    assert isinstance(receipt, dict)
    assert captured["queue"] == p2.WORKER_QUEUE
    assert receipt["receipt_type"] == p2.P2_PROFILE.receipt_type
    assert receipt["plan_seal_sha256"] == plan["seal_sha256"]
    assert receipt["deployment_plan_seal_sha256"] == deployment["seal_sha256"]
    assert receipt["source_package_manifest_seal_sha256"] == manifest["seal_sha256"]
    assert receipt["source_transition_seal_sha256"] == transition["seal_sha256"]
    assert receipt["planned_queue"] == p2.WORKER_QUEUE
    assert receipt["planned_queue_id"] == p2.WORKER_QUEUE_ID
    assert receipt["teacher_task_id"] == shared.fastlane.TEACHER_TASK_ID
    assert receipt["teacher_model_id"] == shared.fastlane.TEACHER_MODEL_ID
    assert receipt["early_gate_policy"] == p2.EARLY_GATE_POLICY
    assert receipt["execution_policy"] == plan["execution_policy"]
    assert receipt["receipt_upload_state"] == "before_enqueue"
    shared._require_seal(receipt, context="test P2 launch receipt")
    assert shared._profile() is shared.P0_PROFILE
