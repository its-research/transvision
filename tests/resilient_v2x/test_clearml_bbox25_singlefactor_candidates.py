from __future__ import annotations

import copy
import hashlib
import json
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import (
    prepare_clearml_bbox25_singlefactor_candidates as bbox,
)
from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared
from tools.resilient_v2x import prepare_clearml_sota_round2_p2 as p2

from .test_configs import CONFIG_ROOT, _load_config


E2 = CONFIG_ROOT / "improvements" / "no_reliability_linear.py"
E2_BBOX25 = CONFIG_ROOT / "improvements" / "no_reliability_linear_bbox25.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _p0_p2_snapshot() -> dict[str, tuple[int, str]]:
    snapshot = {}
    for root in (bbox.P2_OUTPUT_DIR, p2.P0_OUTPUT_DIR):
        for path in sorted(root.iterdir()):
            if path.is_file():
                snapshot[f"{root.name}/{path.name}"] = (
                    path.stat().st_size,
                    _sha256(path),
                )
    return snapshot


def _prepare(
    tmp_path: Path, key: str
) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    output = tmp_path / key
    with bbox._candidate_engine(key):
        assert shared.main(["prepare", "--output-dir", str(output)]) == 0
        manifest = shared._verify_package(output)
        deployment, static = shared._verify_deployment_plan(output, manifest)
    return output, manifest, deployment, static


def _leaf_differences(
    left: object,
    right: object,
    path: tuple[str, ...] = (),
) -> dict[tuple[str, ...], tuple[object, object]]:
    missing = object()
    if isinstance(left, dict) and isinstance(right, dict):
        result: dict[tuple[str, ...], tuple[object, object]] = {}
        for key in left.keys() | right.keys():
            result.update(
                _leaf_differences(
                    left.get(key, missing),
                    right.get(key, missing),
                    (*path, key),
                )
            )
        return result
    return {} if left == right else {path: (left, right)}


def _bindings() -> dict[str, object]:
    return {
        "source_package_manifest_seal_sha256": "1" * 64,
        "source_tree_sha256": "2" * 64,
        "deployment_plan_seal_sha256": "3" * 64,
        "static_plan_seal_sha256": "4" * 64,
    }


def _signal_metrics() -> dict[str, float]:
    return {
        "p0_bev_ap70": 61.2,
        "p0_3d_ap70": 33.2,
        "p2_bev_ap70": 61.1,
        "p2_3d_ap70": 33.6,
    }


def _preferred_name() -> str:
    spec = bbox.CANDIDATE_SPECS[bbox.PRIMARY_KEY]
    return f"{spec.task_prefix} [{'a' * 12}]"


def test_e2_bbox25_is_exact_student_only_overlay_and_highest_priority() -> None:
    source = runpy.run_path(str(E2_BBOX25))
    base = _load_config(E2)
    candidate = _load_config(E2_BBOX25)

    assert source["_base_"] == ["./no_reliability_linear.py"]
    assert source["model"] == {"bbox_head": {"loss_bbox": {"loss_weight": 2.5}}}
    assert source["experiment"]["name"] == (
        "dair_improvement_no_reliability_linear_bbox25"
    )
    assert source["experiment"]["trigger"] == {
        "mode": "mutually_exclusive",
        "group": bbox.MUTUAL_EXCLUSION_GROUP,
        "priority": "preferred",
        "lower_priority_experiments": (
            "dair_improvement_no_reliability_bbox25",
            "dair_improvement_support_residual_bbox25",
        ),
    }
    assert source["experiment"]["protocol_lock"] == {
        "training_seed": 20250218,
        "precision": "FP32",
        "gpu_count": 4,
        "train_batch_size_per_gpu": 2,
        "global_batch_size": 8,
        "max_epochs": 50,
        "val_interval": 10,
    }

    operational = copy.deepcopy(candidate)
    operational["experiment"] = copy.deepcopy(base["experiment"])
    assert _leaf_differences(base, operational) == {
        ("model", "bbox_head", "loss_bbox", "loss_weight"): (2.0, 2.5)
    }
    model = candidate["model"]
    assert model["ptf_mode"] == "linear"
    assert model["use_reliability"] is False
    assert model.get("support_residual_weight", 0.0) == 0.0
    assert model["bbox_head"]["loss_bbox"]["loss_weight"] == 2.5
    assert model["teacher"]["bbox_head"]["loss_bbox"]["loss_weight"] == 2.0

    assert tuple(bbox.CANDIDATE_SPECS) == (
        bbox.PRIMARY_KEY,
        bbox.SECONDARY_KEY,
        bbox.BACKUP_KEY,
    )
    assert bbox.CANDIDATE_SPECS[bbox.PRIMARY_KEY].priority == "preferred"
    assert bbox.CANDIDATE_SPECS[bbox.SECONDARY_KEY].priority == "fallback"
    assert bbox.CANDIDATE_SPECS[bbox.BACKUP_KEY].priority == "fallback"


@pytest.mark.parametrize("key", tuple(bbox.CANDIDATE_SPECS))
def test_each_candidate_is_an_isolated_one_file_additive_p2_descendant(
    tmp_path: Path, key: str
) -> None:
    before = _p0_p2_snapshot()
    output, manifest, deployment, static = _prepare(tmp_path, key)
    after = _p0_p2_snapshot()
    spec = bbox.CANDIDATE_SPECS[key]
    p2_manifest = bbox._verified_p2_manifest()
    p2_inventory = json.loads(
        (bbox.P2_OUTPUT_DIR / bbox.SOURCE_INVENTORY_NAME).read_text()
    )
    inventory = json.loads((output / bbox.SOURCE_INVENTORY_NAME).read_text())

    assert before == after
    assert manifest["base_source_dataset_id"] == bbox.P2_SOURCE_DATASET_ID
    assert manifest["base_source_package"] == p2_manifest["target_source_package"]
    assert manifest["delta"]["modified_files"] == []
    assert manifest["delta"]["removed_files"] == []
    assert [entry["path"] for entry in manifest["delta"]["added_files"]] == [
        spec.config
    ]
    assert inventory["file_count"] == p2_inventory["file_count"] + 1
    assert {entry["path"] for entry in inventory["files"]} == {
        entry["path"] for entry in p2_inventory["files"]
    } | {spec.config}
    assert (
        manifest["provenance"]["parent_source_tree_sha256"]
        == (p2_manifest["target_source_package"]["tree_sha256"])
    )
    assert manifest["provenance"]["scientific_base_config"] == spec.base_config
    assert deployment["remote_state_changed"] is False
    assert static["protocol"] == {
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
    assert static["execution_contract"] == bbox.PROFILES[key].execution_contract
    shared._require_seal(
        static["execution_contract"], context="test execution contract"
    )
    commands = deployment["commands"]
    assert all(f"--candidate {key}" in command for command in commands.values())
    assert bbox.TRIGGER_RECEIPT_NAME in commands["5_launch_selected_a100"]
    assert bbox.MODEL_PREFLIGHT_RECEIPT_NAME in commands["5_launch_selected_a100"]
    assert spec.launch_token in commands["5_launch_selected_a100"]
    assert "GPU4-A100" in commands["5_launch_selected_a100"]
    shared._require_seal(manifest, context="test bbox25 package")
    shared._require_seal(deployment, context="test bbox25 deployment")
    assert shared._profile() is shared.P0_PROFILE
    assert shared._deployment_plan is bbox._ORIGINAL_DEPLOYMENT_PLAN


@pytest.mark.parametrize("key", tuple(bbox.CANDIDATE_SPECS))
def test_bootstrap_patch_installs_only_the_selected_bbox25_identity(
    key: str, monkeypatch: pytest.MonkeyPatch
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
    with bbox._candidate_engine(key):
        patched = shared._patch_template(raw)

    spec = bbox.CANDIDATE_SPECS[key]
    assert patched.count(f'"{p2.P2_EXPERIMENT}"') >= 2
    assert patched.count(f'"{spec.experiment}"') >= 2
    assert patched.count(f'"{spec.config}"') == 1
    for peer_key, peer in bbox.CANDIDATE_SPECS.items():
        if peer_key != key:
            assert f'"{peer.config}"' not in patched


def test_trigger_prefers_e2_bbox25_and_never_automatically_enqueues() -> None:
    receipt = bbox.build_trigger_receipt(
        key=bbox.PRIMARY_KEY,
        bindings=_bindings(),
        static_plan={"seal_sha256": "a" * 64},
        created_at="2026-08-12T00:00:00+00:00",
        **_signal_metrics(),
    )
    assert receipt["selected_candidate"]["key"] == bbox.PRIMARY_KEY
    assert receipt["selected_candidate"]["priority"] == "preferred"
    assert receipt["decision"] == {
        "bbox_signal": True,
        "authorized": True,
        "automatic_enqueue": False,
        "selected_count": 1,
    }
    assert receipt["fallback"] is None
    shared._require_seal(receipt, context="test preferred trigger")

    rejected = bbox.build_trigger_receipt(
        key=bbox.PRIMARY_KEY,
        bindings=_bindings(),
        static_plan={"seal_sha256": "a" * 64},
        p0_bev_ap70=61.2,
        p0_3d_ap70=33.2,
        p2_bev_ap70=60.9,
        p2_3d_ap70=33.3,
        created_at="2026-08-12T00:00:00+00:00",
    )
    assert rejected["decision"]["bbox_signal"] is False
    assert rejected["decision"]["authorized"] is False


@pytest.mark.parametrize("key", (bbox.SECONDARY_KEY, bbox.BACKUP_KEY))
def test_fallback_trigger_requires_terminal_preferred_task_and_explicit_reason(
    key: str,
) -> None:
    receipt = bbox.build_trigger_receipt(
        key=key,
        bindings=_bindings(),
        static_plan={"seal_sha256": "b" * 64},
        primary_task_id="c" * 32,
        primary_task_name=_preferred_name(),
        primary_status="completed",
        fallback_reason="preferred candidate did not satisfy the promotion threshold",
        created_at="2026-08-12T00:00:00+00:00",
        **_signal_metrics(),
    )
    assert receipt["decision"]["authorized"] is True
    assert receipt["selected_candidate"]["key"] == key
    assert receipt["fallback"] == {
        "primary_task_id": "c" * 32,
        "primary_task_name": _preferred_name(),
        "primary_terminal_status": "completed",
        "reason": "preferred candidate did not satisfy the promotion threshold",
    }
    shared._require_seal(receipt, context="test fallback trigger")

    with pytest.raises(ValueError, match="terminal preferred status"):
        bbox.build_trigger_receipt(
            key=key,
            bindings=_bindings(),
            static_plan={"seal_sha256": "b" * 64},
            primary_task_id="c" * 32,
            primary_task_name=_preferred_name(),
            primary_status="in_progress",
            fallback_reason="not terminal",
            **_signal_metrics(),
        )


def _valid_model_preflight(
    *,
    spec: bbox.CandidateSpec,
    manifest: Mapping[str, object],
    inventory: Mapping[str, object],
    bindings: Mapping[str, object],
) -> dict[str, object]:
    expected_runtime = bbox.MODEL_PREFLIGHT_POLICY["runtime"]
    assert isinstance(expected_runtime, dict)
    runtime = {
        **expected_runtime,
        "cuda_available": True,
        "capabilities": [[8, 0], [8, 0], [8, 0], [8, 0]],
        "torch_arch_list": ["sm_70", "sm_80", "sm_120"],
    }
    runtime.pop("capability")
    config_entry = bbox._inventory_entry(inventory, spec.config)
    base_entry = bbox._inventory_entry(inventory, spec.base_config)
    state_hash = "d" * 64
    return shared._sealed(
        {
            "schema_version": 1,
            "receipt_type": "resilient_v2x_bbox25_full_model_preflight_v1",
            "created_at": "2026-08-12T00:00:00+00:00",
            "policy": dict(bbox.MODEL_PREFLIGHT_POLICY),
            "bindings": dict(bindings),
            "candidate": dict(manifest["candidate"]),
            "config": {
                "candidate_path": spec.config,
                "candidate_sha256": config_entry["sha256"],
                "base_path": spec.base_config,
                "base_sha256": base_entry["sha256"],
            },
            "checkpoints": {
                "teacher_sha256": shared.fastlane.TEACHER_CHECKPOINT_SHA256,
                "resnet50_sha256": bbox.source_runner.EXPECTED_RESNET_SHA256,
            },
            "runtime": runtime,
            "source_audit": {
                "checked_file_count": len(inventory["files"]),
                "inventory_tree_sha256": inventory["tree_sha256"],
                "mismatches": [],
                "compatibility_modified_paths": [],
            },
            "model_audit": {
                "base_full_model_build_passed": True,
                "candidate_full_model_build_passed": True,
                "base_init_weights_passed": True,
                "candidate_init_weights_passed": True,
                "strict_load_passed": True,
                "strict_missing_keys": [],
                "strict_unexpected_keys": [],
                "state_key_count": 1085,
                "state_schema_sha256": "e" * 64,
                "source_state_sha256": state_hash,
                "target_state_sha256": state_hash,
                "complete_state_equal_after_strict_load": True,
                "base_student_bbox_loss_weight": 2.0,
                "candidate_student_bbox_loss_weight": 2.5,
                "candidate_nested_teacher_bbox_loss_weight": 2.0,
                "base_model_type": "ResilientV2XNet",
                "candidate_model_type": "ResilientV2XNet",
                "loaded_source_modules": {},
            },
            "authorized": True,
        }
    )


def test_model_preflight_receipt_requires_complete_build_init_and_strict_load(
    tmp_path: Path,
) -> None:
    output, manifest, deployment, static = _prepare(tmp_path, bbox.PRIMARY_KEY)
    inventory = json.loads((output / bbox.SOURCE_INVENTORY_NAME).read_text())
    bindings = bbox._candidate_bindings(manifest, deployment, static)
    spec = bbox.CANDIDATE_SPECS[bbox.PRIMARY_KEY]
    receipt = _valid_model_preflight(
        spec=spec,
        manifest=manifest,
        inventory=inventory,
        bindings=bindings,
    )
    bbox._require_model_preflight_receipt(
        receipt,
        spec=spec,
        manifest=manifest,
        bindings=bindings,
        inventory=inventory,
    )

    for field in (
        "candidate_full_model_build_passed",
        "candidate_init_weights_passed",
        "strict_load_passed",
        "complete_state_equal_after_strict_load",
    ):
        drifted = copy.deepcopy(receipt)
        drifted["model_audit"][field] = False
        drifted = shared._sealed(drifted)
        with pytest.raises(RuntimeError, match="build/strict-load gate did not pass"):
            bbox._require_model_preflight_receipt(
                drifted,
                spec=spec,
                manifest=manifest,
                bindings=bindings,
                inventory=inventory,
            )


class _Task:
    def __init__(self, task_id: str, name: str, status: str) -> None:
        self.id = task_id
        self.name = name
        self.parent = shared.fastlane.PREDECESSOR_TASK_ID
        self.status = status


class _TaskClass:
    tasks: list[_Task] = []

    @classmethod
    def get_tasks(cls, **_kwargs: object) -> list[_Task]:
        return list(cls.tasks)


def test_group_guard_allows_preferred_alone_and_only_one_terminal_fallback() -> None:
    preferred_spec = bbox.CANDIDATE_SPECS[bbox.PRIMARY_KEY]
    preferred_trigger = bbox.build_trigger_receipt(
        key=bbox.PRIMARY_KEY,
        bindings=_bindings(),
        static_plan={"seal_sha256": "a" * 64},
        **_signal_metrics(),
    )
    _TaskClass.tasks = []
    bbox._require_group_state(
        _TaskClass,
        spec=preferred_spec,
        trigger=preferred_trigger,
        current_task=None,
    )
    preferred_task = _Task(
        "c" * 32,
        preferred_trigger["selected_candidate"]["expected_task_name"],
        "completed",
    )
    _TaskClass.tasks = [preferred_task]
    with pytest.raises(RuntimeError, match="selected-task count"):
        bbox._require_group_state(
            _TaskClass,
            spec=preferred_spec,
            trigger=preferred_trigger,
            current_task=None,
        )

    fallback_spec = bbox.CANDIDATE_SPECS[bbox.SECONDARY_KEY]
    fallback_trigger = bbox.build_trigger_receipt(
        key=bbox.SECONDARY_KEY,
        bindings=_bindings(),
        static_plan={"seal_sha256": "b" * 64},
        primary_task_id=preferred_task.id,
        primary_task_name=preferred_task.name,
        primary_status="completed",
        fallback_reason="preferred did not promote",
        **_signal_metrics(),
    )
    bbox._require_group_state(
        _TaskClass,
        spec=fallback_spec,
        trigger=fallback_trigger,
        current_task=None,
    )
    fallback_task = _Task(
        "d" * 32,
        fallback_trigger["selected_candidate"]["expected_task_name"],
        "created",
    )
    _TaskClass.tasks = [preferred_task, fallback_task]
    bbox._require_group_state(
        _TaskClass,
        spec=fallback_spec,
        trigger=fallback_trigger,
        current_task=fallback_task,
    )
    peer = _Task(
        "e" * 32,
        f"{bbox.CANDIDATE_SPECS[bbox.BACKUP_KEY].task_prefix} [{'f' * 12}]",
        "completed",
    )
    _TaskClass.tasks.append(peer)
    with pytest.raises(RuntimeError, match="unauthorized task"):
        bbox._require_group_state(
            _TaskClass,
            spec=fallback_spec,
            trigger=fallback_trigger,
            current_task=fallback_task,
        )


def test_launch_hook_augments_receipt_and_runs_group_guard_after_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = bbox.CANDIDATE_SPECS[bbox.PRIMARY_KEY]
    task = _Task("c" * 32, f"{spec.task_prefix} [{'a' * 12}]", "created")
    trigger = shared._sealed({"receipt_type": "test-trigger"})
    preflight = shared._sealed({"receipt_type": "test-preflight"})
    events: list[str] = []
    captured: dict[str, object] = {}

    class FakeClearMLTask:
        pass

    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeClearMLTask))
    monkeypatch.setattr(
        bbox,
        "_require_group_state",
        lambda *_args, current_task, **_kwargs: events.append(
            "group_pre" if current_task is None else "group_post"
        ),
    )

    def original_unique(_task_class: object, _task: object) -> None:
        events.append("own_unique")

    def original_upload(
        task_class: object,
        observed_task: object,
        receipt: Mapping[str, object],
        *,
        queue: str,
    ) -> None:
        events.append("receipt_readback")
        captured["receipt"] = dict(receipt)
        captured["queue"] = queue
        shared._require_unique_current_clone(task_class, observed_task)

    def fake_launch(_args: argparse.Namespace) -> int:
        receipt = shared._sealed({"schema_version": 1, "receipt_type": "base"})
        shared._upload_verify_and_enqueue(
            FakeClearMLTask, task, receipt, queue=p2.WORKER_QUEUE
        )
        return 0

    import argparse

    monkeypatch.setattr(shared, "_require_unique_current_clone", original_unique)
    monkeypatch.setattr(shared, "_upload_verify_and_enqueue", original_upload)
    monkeypatch.setattr(shared, "_launch", fake_launch)
    args = argparse.Namespace()
    assert (
        bbox._launch_with_gates(args, spec=spec, trigger=trigger, preflight=preflight)
        == 0
    )
    assert events == ["group_pre", "receipt_readback", "own_unique", "group_post"]
    receipt = captured["receipt"]
    assert receipt["selection_trigger_receipt"] == trigger
    assert receipt["model_preflight_receipt"] == preflight
    assert receipt["mutual_exclusion_group"] == bbox.MUTUAL_EXCLUSION_GROUP
    assert receipt["candidate_priority"] == "preferred"
    shared._require_seal(receipt, context="test augmented launch receipt")
    assert captured["queue"] == p2.WORKER_QUEUE
    assert shared._upload_verify_and_enqueue is original_upload
    assert shared._require_unique_current_clone is original_unique


@pytest.mark.parametrize(
    "command",
    (
        ["upload-source", "--candidate", bbox.PRIMARY_KEY],
        [
            "create-template",
            "--candidate",
            bbox.PRIMARY_KEY,
            "--source-dataset-id",
            "a" * 32,
        ],
        [
            "launch",
            "--candidate",
            bbox.PRIMARY_KEY,
            "--source-dataset-id",
            "a" * 32,
            "--template-task-id",
            "b" * 32,
            "--trigger-receipt",
            "/nonexistent/trigger.json",
            "--model-preflight-receipt",
            "/nonexistent/preflight.json",
        ],
    ),
)
def test_remote_commands_fail_closed_on_exact_candidate_token(
    command: list[str],
) -> None:
    with pytest.raises(PermissionError, match="exact execute token required"):
        bbox.main(command)
    assert shared._profile() is shared.P0_PROFILE
    assert shared._deployment_plan is bbox._ORIGINAL_DEPLOYMENT_PLAN
