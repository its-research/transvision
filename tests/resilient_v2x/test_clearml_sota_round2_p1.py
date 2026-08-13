from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared
from tools.resilient_v2x import prepare_clearml_sota_round2_p1 as p1
from tools.resilient_v2x import prepare_clearml_sota_round2_p2 as p2


EXPECTED_P1_TREE_SHA256 = (
    "a55eec3ab94918e8cbbb63b76923ae95b22202f7eeb4bf7cc0d5ec1809c85378"
)
EXPECTED_P1_SOURCE_BYTES = 8_931_193


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(directory: Path) -> dict[str, tuple[int, str]]:
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(directory).as_posix(): (path.stat().st_size, _sha256(path))
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _prepare(output_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    assert p1.main(["prepare", "--output-dir", str(output_dir)]) == 0
    manifest = json.loads(
        (output_dir / p1.PACKAGE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    plan = json.loads(
        (output_dir / p1.DEPLOYMENT_PLAN_NAME).read_text(encoding="utf-8")
    )
    with shared.use_deployment_profile(p1.P1_PROFILE):
        assert shared._verify_package(output_dir) == manifest
        observed, static = shared._verify_deployment_plan(output_dir, manifest)
    assert observed == plan
    assert plan["static_plan_seal_sha256"] == static["seal_sha256"]
    return manifest, plan


def test_p1_profile_is_isolated_and_derived_from_exact_sealed_p0() -> None:
    p0_manifest, p0_deployment, p0_static = p1._verified_p0_parent()
    profile = p1.P1_PROFILE

    assert shared._profile() is shared.P0_PROFILE
    assert profile.bootstrap_parent is shared.P0_PROFILE
    assert profile.output_dir not in (
        shared.P0_PROFILE.output_dir,
        p2.P2_PROFILE.output_dir,
    )
    assert profile.package_manifest_name not in (
        shared.P0_PROFILE.package_manifest_name,
        p2.P2_PROFILE.package_manifest_name,
    )
    assert profile.source_transition_artifact not in (
        shared.P0_PROFILE.source_transition_artifact,
        p2.P2_PROFILE.source_transition_artifact,
    )
    assert profile.launch_receipt_artifact not in (
        shared.P0_PROFILE.launch_receipt_artifact,
        p2.P2_PROFILE.launch_receipt_artifact,
    )
    assert profile.upload_token not in (
        shared.P0_PROFILE.upload_token,
        p2.P2_PROFILE.upload_token,
    )
    assert profile.template_token not in (
        shared.P0_PROFILE.template_token,
        p2.P2_PROFILE.template_token,
    )
    assert profile.launch_token not in (
        shared.P0_PROFILE.launch_token,
        p2.P2_PROFILE.launch_token,
    )
    assert profile.base_source_dataset_id == p1.P0_SOURCE_DATASET_ID
    assert profile.base_source_package == p0_manifest["target_source_package"]
    assert profile.modified_source_paths == p1.P1_MODIFIED_SOURCE_PATHS
    assert profile.forbidden_source_paths == (p1.P2_CONFIG,)
    assert profile.training_launch_enabled is True
    assert profile.task_parameter_bindings == p1.P1_TASK_PARAMETER_BINDINGS
    assert profile.strict_enqueue_acknowledgement is True
    assert (
        profile.enqueue_acknowledgement_artifact == p1.ENQUEUE_ACKNOWLEDGEMENT_ARTIFACT
    )
    assert profile.execution_contract == p1.EXECUTION_CONTRACT
    assert (
        profile.provenance["parent_source_package_manifest_seal_sha256"]
        == (p0_manifest["seal_sha256"])
    )
    assert (
        profile.provenance["parent_static_plan_seal_sha256"]
        == (p0_static["seal_sha256"])
    )
    assert (
        profile.provenance["parent_deployment_plan_seal_sha256"]
        == (p0_deployment["seal_sha256"])
    )
    assert profile.provenance["explicitly_excluded_paths"] == [p1.P2_CONFIG]


def test_local_prepare_is_exact_p0_plus_three_modified_and_one_added(
    tmp_path: Path,
) -> None:
    p0_before = _snapshot(p1.P0_OUTPUT_DIR)
    p2_before = _snapshot(p2.OUTPUT_DIR)
    output_dir = tmp_path / "p1"
    manifest, plan = _prepare(output_dir)
    assert _snapshot(p1.P0_OUTPUT_DIR) == p0_before
    assert _snapshot(p2.OUTPUT_DIR) == p2_before

    p0_inventory = json.loads(
        (p1.P0_OUTPUT_DIR / "source-inventory.json").read_text(encoding="utf-8")
    )
    p1_inventory = json.loads(
        (output_dir / "source-inventory.json").read_text(encoding="utf-8")
    )
    base_by_path = {entry["path"]: entry for entry in p0_inventory["files"]}
    target_by_path = {entry["path"]: entry for entry in p1_inventory["files"]}
    modified = manifest["delta"]["modified_files"]
    added = manifest["delta"]["added_files"]

    assert manifest["base_source_dataset_id"] == p1.P0_SOURCE_DATASET_ID
    assert [entry["path"] for entry in modified] == sorted(p1.P1_MODIFIED_SOURCE_PATHS)
    assert [entry["path"] for entry in added] == [p1.P1_CONFIG]
    assert manifest["delta"]["removed_files"] == []
    assert manifest["excluded_paths"] == [p1.P2_CONFIG]
    assert p1.P2_CONFIG not in target_by_path
    assert set(target_by_path) == set(base_by_path) | {p1.P1_CONFIG}
    assert p1_inventory["file_count"] == p0_inventory["file_count"] + 1 == 636
    assert p1_inventory["source_bytes"] == EXPECTED_P1_SOURCE_BYTES
    assert p1_inventory["tree_sha256"] == EXPECTED_P1_TREE_SHA256
    assert manifest["target_source_package"]["tree_sha256"] == (EXPECTED_P1_TREE_SHA256)

    changed = {
        path for path in base_by_path if base_by_path[path] != target_by_path[path]
    }
    assert changed == set(p1.P1_MODIFIED_SOURCE_PATHS)
    for path in (*p1.P1_MODIFIED_SOURCE_PATHS, p1.P1_CONFIG):
        assert target_by_path[path]["sha256"] == p1.P1_TARGET_FILE_SHA256[path]
        assert target_by_path[path]["mode"] == 0o644

    assert plan["remote_state_changed"] is False
    assert set(plan["commands"]) == {
        "1_upload_source",
        "2_create_template",
        "3_launch_a100",
        "4_show_execution_contract",
    }
    assert f"--queue {p1.WORKER_QUEUE}" in plan["commands"]["3_launch_a100"]
    assert "show-execution-contract" in plan["commands"]["4_show_execution_contract"]
    assert plan["execution_policy"]["training_launch_enabled"] is True
    assert plan["execution_policy"]["executor_count"] == 1
    assert plan["execution_policy"]["concurrent_launchers_supported"] is False
    assert plan["execution_policy"]["queue_name_to_exact_id_preflight"] is True
    assert plan["execution_policy"]["post_enqueue_reload_required"] is True
    assert plan["execution_policy"]["post_enqueue_worker_assignment_validation"] is True
    assert plan["execution_policy"]["sealed_enqueue_acknowledgement_artifact"] == (
        p1.ENQUEUE_ACKNOWLEDGEMENT_ARTIFACT
    )
    assert plan["task_parameter_bindings"] == p1.P1_TASK_PARAMETER_BINDINGS
    assert plan["execution_contract"] == p1.EXECUTION_CONTRACT
    assert plan["execution_contract"]["default_candidate"] == (p1.P1_TRAINED_CANDIDATE)
    shared._require_seal(manifest, context="test P1 package")
    shared._require_seal(plan, context="test P1 deployment plan")


def test_p1_source_package_is_byte_deterministic(tmp_path: Path) -> None:
    first_manifest, _ = _prepare(tmp_path / "first")
    second_manifest, _ = _prepare(tmp_path / "second")
    assert first_manifest == second_manifest

    package = first_manifest["target_source_package"]
    for name in (
        package["archive_name"],
        "source-inventory.json",
        p1.PACKAGE_MANIFEST_NAME,
    ):
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()


def test_p1_transition_and_plan_bind_two_distinct_candidate_provenances(
    tmp_path: Path,
) -> None:
    manifest, plan = _prepare(tmp_path / "p1")
    with shared.use_deployment_profile(p1.P1_PROFILE):
        transition = shared._transition(manifest, dataset_id="a" * 32)
        parameters = shared._task_parameters(manifest, dataset_id="a" * 32)

    assert transition["inventory_delta"]["base_file_count"] == 635
    assert transition["inventory_delta"]["target_file_count"] == 636
    assert transition["inventory_delta"]["unchanged_file_count"] == 632
    assert transition["inventory_delta"]["modified_file_count"] == 3
    assert transition["inventory_delta"]["added_file_count"] == 1
    assert transition["inventory_delta"]["removed_file_count"] == 0
    assert (
        transition["inventory_delta"]["modified_files"]
        == (manifest["delta"]["modified_files"])
    )
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
    contract = plan["execution_contract"]
    assert contract["task_parameter_bindings"] == p1.P1_TASK_PARAMETER_BINDINGS
    assert contract["runtime_execution"] == {
        "runtime_profile": "a100_sm80",
        "runtime_hardware": "A100",
        "cuda_compute_capability": "sm80",
        "runtime_implementation_compatibility_layer": (
            p1.P1_RUNTIME_IMPLEMENTATION_COMPATIBILITY_LAYER
        ),
        "forbidden_result_tags": ["RTX5090", "sm120"],
    }
    variants = contract["candidate_variants"]
    trained = variants[p1.P1_TRAINED_CANDIDATE]
    zero_shot = variants[p1.P1_ZERO_SHOT_CANDIDATE]
    assert trained["candidate_identity"] == p1.P1_TRAINED_CANDIDATE_IDENTITY
    assert trained["candidate_variant"] == p1.P1_TRAINED_CANDIDATE
    assert zero_shot["candidate_identity"] == (p1.P1_ZERO_SHOT_CANDIDATE_IDENTITY)
    assert trained["candidate_identity"] != zero_shot["candidate_identity"]
    assert trained["mode"] == "train_then_formal_inference"
    assert zero_shot["mode"] == "inference_only"
    assert trained["training"] == {
        "max_epochs": 50,
        "val_interval": 10,
        "gpu_count": 4,
        "batch_size_per_gpu": 2,
        "global_batch_size": 8,
        "precision": "FP32",
        "training_seed": 20250218,
        "training_overlay": {
            "protocol_seed": 20250218,
            "delays_ms": [0, 100, 200, 300],
            "fault_conditions": ["Full", "L-Fail", "C-Fail"],
            "condition_count": 12,
            "uniformly_applied": True,
        },
    }
    assert trained["output_checkpoint"] == {
        "checkpoint_policy": "epoch_50_final_only",
        "checkpoint_filename": "epoch_50.pth",
        "output_model_role": "canonical_epoch_50_final_output_model",
        "model_id": None,
    }
    for candidate in (trained, zero_shot):
        evaluation = candidate["formal_evaluation"]
        assert evaluation["protocol_id"] == "DAIR-CAUSAL-1337-v1"
        assert evaluation["sample_count"] == 1337
        assert evaluation["overlay_seed"] == 20250218
        assert evaluation["condition_count"] == 12
        assert "seed" not in evaluation
    checkpoint = zero_shot["source_checkpoint"]
    assert checkpoint["source_task_id"] == p1.P0_TRAINING_TASK_ID
    assert checkpoint["checkpoint_policy"] == "epoch_50_final_only"
    assert checkpoint["checkpoint_filename"] == "epoch_50.pth"
    assert checkpoint["output_model_role"] == ("canonical_epoch_50_final_output_model")
    assert checkpoint["model_id"] is None
    assert "clean_validation_best" in checkpoint["forbidden_checkpoint_roles"]
    assert parameters["Args/experiment_from_task"] == p1.P1_EXPERIMENT
    assert parameters["Args/training_seed"] == 20250218
    assert parameters["Args/gpus"] == 4
    assert parameters["Args/max_epochs"] == 50
    assert parameters["Args/amp"] is False
    for key, value in p1.P1_TASK_PARAMETER_BINDINGS.items():
        assert parameters[key] == value
    shared._require_seal(transition, context="test P1 transition")


def _raw_bootstrap_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duplicate_first_extra_patch_anchor: bool = False,
) -> str:
    import zstandard

    target = "tools/resilient_v2x/clearml_5090_bootstrap.py"
    raw: str | None = None
    with p1.P1_PROFILE.base_source_archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    if member.name != target:
                        continue
                    payload = archive.extractfile(member)
                    assert payload is not None
                    raw = payload.read().decode("utf-8")
                    break
    assert raw is not None
    if duplicate_first_extra_patch_anchor:
        raw += p1._CANDIDATE_PARSER_ANCHOR
    parent_patched = shared.candidate._apply_candidate_experiment_patch(raw)
    monkeypatch.setattr(
        shared.fastlane,
        "TEMPLATE_SCRIPT_SHA256",
        hashlib.sha256(raw.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        shared.fastlane,
        "PATCHED_TEMPLATE_SCRIPT_SHA256",
        hashlib.sha256(parent_patched.encode()).hexdigest(),
    )
    return raw


def test_p1_bootstrap_chains_p0_and_installs_only_p1_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_bootstrap_fixture(monkeypatch)
    with shared.use_deployment_profile(p1.P1_PROFILE):
        patched = shared._patch_template(raw)
    compile(patched, "<p1-standalone-bootstrap>", "exec")

    assert patched.count(f'"{shared.P0_EXPERIMENT}"') >= 2
    assert patched.count(f'"{shared.P0_CONFIG}"') == 1
    assert patched.count(f'"{p1.P1_EXPERIMENT}"') >= 2
    assert patched.count(f'"{p1.P1_CONFIG}"') >= 1
    assert p1.P2_EXPERIMENT not in patched
    assert p1.P2_CONFIG not in patched
    assert "P1_TARGET_A100_CUDA_SMOKE" in patched
    assert p1.TARGET_A100_SMOKE_EVENT in patched
    assert p1._FINAL_TAGS_ANCHOR not in patched
    assert p1._TASK_TAGS_ANCHOR not in patched
    run_contract_start = patched.index("def _experiment_run_contract")
    run_contract_end = patched.index("def _prepare_teacher_handoff", run_contract_start)
    run_contract_source = patched[run_contract_start:run_contract_end]
    assert '"runtime_profile": "rtx5090"' not in run_contract_source
    for marker in (
        "P1_EXPECTED_CANDIDATE_PROVENANCE",
        "runtime_profile:a100_sm80",
        "runtime_hardware",
        "cuda_compute_capability",
        "runtime_implementation_compatibility_layer",
        "candidate_provenance=candidate_provenance",
        "final checkpoint OutputModel",
        "clean-val best OutputModel",
    ):
        assert marker in patched
    assert patched.index(f'"{shared.P0_EXPERIMENT}"') < patched.index(
        f'"{p1.P1_EXPERIMENT}"'
    )


def test_p1_standalone_bootstrap_runtime_contract_and_output_model_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_bootstrap_fixture(monkeypatch)
    with shared.use_deployment_profile(p1.P1_PROFILE):
        patched = shared._patch_template(raw)
    namespace: dict[str, object] = {"__name__": "p1_standalone_bootstrap_test"}
    exec(compile(patched, "<p1-standalone-bootstrap>", "exec"), namespace)

    spec = namespace["EXPERIMENT_BY_NAME"][p1.P1_EXPERIMENT]
    exact_args = SimpleNamespace(**p1.P1_CANDIDATE_PROVENANCE)
    provenance = namespace["_p1_candidate_provenance"](exact_args, spec)
    assert provenance == p1.P1_CANDIDATE_PROVENANCE

    for key in p1.P1_CANDIDATE_PROVENANCE:
        forged = dict(p1.P1_CANDIDATE_PROVENANCE)
        forged[key] = f"wrong-{key}"
        with pytest.raises(ValueError, match="P1 candidate provenance mismatch"):
            namespace["_p1_candidate_provenance"](
                SimpleNamespace(**forged),
                spec,
            )

    required_tags = set(namespace["_p1_runtime_tags"](spec, provenance))
    assert {"A100", "sm80", "runtime_profile:a100_sm80"} <= required_tags
    assert {"RTX5090", "sm120"}.isdisjoint(required_tags)
    for key, value in p1.P1_CANDIDATE_PROVENANCE.items():
        assert f"{key}={value}" in required_tags
    with pytest.raises(RuntimeError, match="forbidden result tags"):
        namespace["_p1_require_result_tags"](
            [*required_tags, "RTX5090"],
            spec,
            provenance,
            context="test result",
        )

    class FakeTrainingTask:
        def __init__(self) -> None:
            self.id = "9" * 32
            self.output_uri = ""
            self.tags = ["inherited", "RTX5090", "sm120"]

        def get_tags(self) -> list[str]:
            return list(self.tags)

        def set_tags(self, tags: list[str]) -> None:
            self.tags = list(tags)

    training_task = FakeTrainingTask()
    namespace["_current_or_init_experiment_task"] = lambda _task_class, _spec: (
        training_task
    )

    class StopAfterTaskTags:
        @staticmethod
        def get_task(*, task_id: str) -> object:
            assert task_id == "8" * 32
            raise RuntimeError("stop after task-tag validation")

    execute_args = SimpleNamespace(
        experiment_from_task=p1.P1_EXPERIMENT,
        predecessor_task_id="8" * 32,
        **p1.P1_CANDIDATE_PROVENANCE,
    )
    with pytest.raises(RuntimeError, match="stop after task-tag validation"):
        namespace["_execute_experiment_from_task"](
            execute_args,
            source_root=p1.ROOT,
            python=Path(sys.executable),
            runtime_env={},
            dataset_class=object(),
            task_class=StopAfterTaskTags,
            output_model_class=object(),
        )
    assert {"RTX5090", "sm120"}.isdisjoint(training_task.tags)
    assert required_tags <= set(training_task.tags)

    class FakeOutputModel:
        def __init__(self, *, tags: list[str], **_kwargs: object) -> None:
            self.id = "f" * 32
            self.tags = list(tags)

        def update_weights(self, **_kwargs: object) -> str:
            return "http://10.100.34.118:8081/ResilientV2X/Training/models/final.pth"

    checkpoint = tmp_path / "epoch_50.pth"
    checkpoint.write_bytes(b"sealed-p1-checkpoint")
    final_contract = namespace["_upload_experiment_checkpoint"](
        task=object(),
        output_model_class=FakeOutputModel,
        spec=spec,
        checkpoint=checkpoint,
        candidate_provenance=provenance,
    )
    best_contract = namespace["_upload_experiment_best_checkpoint"](
        task=object(),
        output_model_class=FakeOutputModel,
        spec=spec,
        checkpoint=checkpoint,
        epoch=50,
        candidate_provenance=provenance,
    )
    for contract in (final_contract, best_contract):
        assert contract | p1.P1_CANDIDATE_PROVENANCE == contract
        assert contract["runtime_profile"] == "a100_sm80"
        assert contract["runtime_hardware"] == "A100"
        assert contract["cuda_compute_capability"] == "sm80"
        assert contract["runtime_implementation_compatibility_layer"] == (
            p1.P1_RUNTIME_IMPLEMENTATION_COMPATIBILITY_LAYER
        )
        assert set(contract["required_result_tags"]) == required_tags

    namespace["_training_overlay_protocol_seed"] = lambda _root: 20250218
    run_args = SimpleNamespace(
        source_dataset_id="a" * 32,
        training_dataset_id="b" * 32,
        native_bundle_sha256="c" * 64,
        build_manifest_sha256="d" * 64,
        source_archive_name="source.tar.zst",
        source_archive_bytes=123,
        source_archive_sha256="e" * 64,
        gpus=4,
        training_seed=20250218,
    )
    run_contract = namespace["_experiment_run_contract"](
        run_args,
        task_id="f" * 32,
        spec=spec,
        dataset_root=tmp_path,
        teacher_contract={"sha256": "a" * 64},
        predecessor_task_id="1" * 32,
        config_path=p1.ROOT / p1.P1_CONFIG,
        training_command=["train"],
        baseline_plan=None,
        candidate_provenance=provenance,
    )
    assert run_contract | p1.P1_CANDIDATE_PROVENANCE == run_contract
    assert run_contract["runtime_profile"] == "a100_sm80"
    assert run_contract["runtime_hardware"] == "A100"
    assert run_contract["cuda_compute_capability"] == "sm80"
    assert run_contract["runtime_implementation_compatibility_layer"] == (
        p1.P1_RUNTIME_IMPLEMENTATION_COMPATIBILITY_LAYER
    )
    assert set(run_contract["required_result_tags"]) == required_tags


def test_p1_bootstrap_extra_patch_fails_closed_on_ambiguous_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_bootstrap_fixture(
        monkeypatch,
        duplicate_first_extra_patch_anchor=True,
    )
    with shared.use_deployment_profile(p1.P1_PROFILE):
        with pytest.raises(RuntimeError, match="extra patch 0 anchor is ambiguous"):
            shared._patch_template(raw)


def test_target_a100_smoke_contract_is_sealed_and_executable_source_is_pinned() -> None:
    compile(
        p1.TARGET_A100_CUDA_SMOKE,
        "<test-resilient-v2x-p1-target-a100-smoke>",
        "exec",
    )
    assert hashlib.sha256(p1.TARGET_A100_CUDA_SMOKE.encode()).hexdigest() == (
        p1.TARGET_A100_CUDA_SMOKE_SHA256
    )
    for path, expected_sha256 in p1.P1_TARGET_FILE_SHA256.items():
        assert _sha256(p1.ROOT / path) == expected_sha256

    contract = p1.TARGET_A100_SMOKE_CONTRACT
    shared._require_seal(contract, context="test P1 target A100 smoke")
    shared._require_seal(p1.EXECUTION_CONTRACT, context="test P1 execution")
    assert contract["required_capabilities"] == [[8, 0]] * 4
    assert contract["event"] == p1.TARGET_A100_SMOKE_EVENT
    assert contract["script_sha256"] == p1.TARGET_A100_CUDA_SMOKE_SHA256
    assert contract["bootstrap_patch_contract_sha256"] == (
        p1.BOOTSTRAP_PATCH_CONTRACT_SHA256
    )
    smoke = p1.TARGET_A100_CUDA_SMOKE
    for required in (
        "Config.fromfile",
        "MODELS.build",
        ".cuda()",
        "candidate_output = candidate(**common)",
        "loss.backward()",
        "torch.equal",
        "capabilities != [(8, 0)] * 4",
    ):
        assert required in smoke


def test_p1_verifier_rejects_exclusion_or_delta_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "p1"
    manifest, _ = _prepare(output_dir)
    manifest_path = output_dir / p1.PACKAGE_MANIFEST_NAME
    original_load = shared._load_json

    forged = dict(manifest)
    forged["excluded_paths"] = []
    forged = shared._sealed(forged)

    def load_exclusion(path: Path) -> dict[str, object]:
        if path == manifest_path:
            return forged
        return original_load(path)

    monkeypatch.setattr(shared, "_load_json", load_exclusion)
    with shared.use_deployment_profile(p1.P1_PROFILE):
        with pytest.raises(RuntimeError, match="excluded source paths drifted"):
            shared._verify_package(output_dir)

    monkeypatch.setattr(shared, "_load_json", original_load)
    forged = json.loads(json.dumps(manifest))
    forged["delta"]["added_files"].append(
        {
            "path": p1.P2_CONFIG,
            "mode": 0o644,
            "size": (p1.ROOT / p1.P2_CONFIG).stat().st_size,
            "sha256": _sha256(p1.ROOT / p1.P2_CONFIG),
        }
    )
    forged = shared._sealed(forged)

    def load_delta(path: Path) -> dict[str, object]:
        if path == manifest_path:
            return forged
        return original_load(path)

    monkeypatch.setattr(shared, "_load_json", load_delta)
    with shared.use_deployment_profile(p1.P1_PROFILE):
        with pytest.raises(RuntimeError, match="exact declared modified/additive"):
            shared._verify_package(output_dir)


class _MockRecord(dict[str, object]):
    def to_dict(self) -> dict[str, object]:
        return dict(self)


class _P1MockTask:
    def __init__(
        self,
        task_id: str,
        name: str,
        status: str,
        *,
        parent: str,
        parameters: Mapping[str, object],
        diff: str,
        events: list[tuple[object, ...]],
    ) -> None:
        self.id = task_id
        self.name = name
        self.status = status
        self.parent = parent
        self.project = shared.TRAINING_PROJECT
        self.parameters = dict(parameters)
        self.output_uri = ""
        self.events = events
        self.data = SimpleNamespace(
            project=shared.TRAINING_PROJECT_ID,
            last_worker=None,
            script=_MockRecord(
                entry_point=shared.fastlane.TEMPLATE_ENTRY_POINT,
                diff=diff,
                binary="python",
                working_dir=".",
            ),
            execution=SimpleNamespace(artifacts=[], queue=None),
        )

    def reload(self) -> None:
        self.events.append(("reload", self.id, self.status))

    def get_project_name(self) -> str:
        return self.project

    def get_parameters(self, **_kwargs: object) -> dict[str, object]:
        return dict(self.parameters)

    def set_parameters(self, parameters: Mapping[str, object]) -> None:
        self.parameters = dict(parameters)
        self.events.append(("parameters", self.id))

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: Mapping[str, object],
        wait_on_upload: bool,
    ) -> bool:
        assert wait_on_upload is True
        serialized = json.dumps(artifact_object, sort_keys=True, indent=4).encode()
        record = _MockRecord(
            key=name,
            hash=hashlib.sha256(serialized).hexdigest(),
            content_size=len(serialized),
            uri=(
                "http://10.100.34.118:8081/ResilientV2X/Training/"
                f"task.{self.id}/artifacts/{name}/{name}.json"
            ),
            type_data={
                "content_type": "application/json",
                "preview": serialized.decode(),
            },
        )
        self.data.execution.artifacts = [
            item for item in self.data.execution.artifacts if item.get("key") != name
        ] + [record]
        self.events.append(("upload", self.id, name))
        return True

    def flush(self, *, wait_for_uploads: bool) -> None:
        assert wait_for_uploads is True
        self.events.append(("flush", self.id))


class _P1MockBackend:
    def __init__(
        self,
        template: _P1MockTask,
        events: list[tuple[object, ...]],
        *,
        post_status: str = "in_progress",
        post_queue_id: str = p1.WORKER_QUEUE_ID,
        post_last_worker: str | None = "mock-a100-sm80-worker",
    ) -> None:
        self.template = template
        self.events = events
        self.post_status = post_status
        self.post_queue_id = post_queue_id
        self.post_last_worker = post_last_worker
        self.tasks = {template.id: template}
        self.clone_count = 0
        self.enqueue_count = 0

    def get_task(self, *, task_id: str) -> _P1MockTask:
        self.events.append(("get", task_id))
        return self.tasks[task_id]

    def get_tasks(
        self,
        *,
        task_name: str,
        task_filter: Mapping[str, object],
    ) -> list[_P1MockTask]:
        assert task_filter == {"parent": shared.fastlane.PREDECESSOR_TASK_ID}
        self.events.append(("duplicate_query", task_name))
        return [
            task
            for task in self.tasks.values()
            if re.fullmatch(task_name, task.name)
            and task.parent == task_filter["parent"]
        ]

    def clone(
        self,
        *,
        source_task: _P1MockTask,
        name: str,
        parent: str,
        project: str,
    ) -> _P1MockTask:
        assert source_task is self.template
        assert project == shared.TRAINING_PROJECT_ID
        self.clone_count += 1
        clone = _P1MockTask(
            f"{self.clone_count:032x}",
            name,
            "created",
            parent=parent,
            parameters=source_task.parameters,
            diff=str(source_task.data.script["diff"]),
            events=self.events,
        )
        self.tasks[clone.id] = clone
        self.events.append(("clone", clone.id))
        return clone

    def enqueue(
        self,
        *,
        task: _P1MockTask,
        queue_id: str,
        force: bool,
    ) -> dict[str, int]:
        assert queue_id == p1.WORKER_QUEUE_ID
        assert force is False
        self.enqueue_count += 1
        self.events.append(("enqueue", task.id, queue_id))
        task.status = self.post_status
        task.data.execution.queue = self.post_queue_id
        task.data.last_worker = self.post_last_worker
        return {"queued": 1, "updated": 1}


def _p1_mock_launch_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolved_queue_id: str = p1.WORKER_QUEUE_ID,
    post_status: str = "in_progress",
    post_queue_id: str = p1.WORKER_QUEUE_ID,
    post_last_worker: str | None = "mock-a100-sm80-worker",
) -> tuple[Path, _P1MockBackend, list[tuple[object, ...]]]:
    output_dir = tmp_path / "p1"
    manifest, _plan = _prepare(output_dir)
    dataset_id = "a" * 32
    template_id = "b" * 32
    with shared.use_deployment_profile(p1.P1_PROFILE):
        transition = shared._transition(manifest, dataset_id=dataset_id)
        template_parameters = shared._expected_template_parameters(
            manifest,
            dataset_id=dataset_id,
        )
    raw = _raw_bootstrap_fixture(monkeypatch)
    parent_patched = shared.candidate._apply_candidate_experiment_patch(raw)
    events: list[tuple[object, ...]] = []
    template = _P1MockTask(
        template_id,
        f"{p1.TEMPLATE_PREFIX} [{str(transition['seal_sha256'])[:12]}]",
        "completed",
        parent=shared.fastlane.MAIN_CONTROLLER_TASK_ID,
        parameters=template_parameters,
        diff=raw,
        events=events,
    )
    backend = _P1MockBackend(
        template,
        events,
        post_status=post_status,
        post_queue_id=post_queue_id,
        post_last_worker=post_last_worker,
    )
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=backend))
    monkeypatch.setattr(
        shared,
        "_downloaded_source_root",
        lambda _manifest, *, dataset_id: output_dir,
    )
    monkeypatch.setattr(
        shared,
        "_validated_fixed_base",
        lambda _task_class: (template, raw, parent_patched),
    )
    monkeypatch.setattr(
        shared,
        "_require_transition_artifact",
        lambda _task, _transition: None,
    )

    def edit_script(task_id: str, diff: str) -> None:
        backend.tasks[task_id].data.script["diff"] = diff
        events.append(("script_edit", task_id))

    monkeypatch.setattr(shared.fastlane, "_edit_script", edit_script)

    def resolve_queue(queue_name: str) -> str:
        assert queue_name == p1.WORKER_QUEUE
        events.append(("queue_resolve", queue_name, resolved_queue_id))
        return resolved_queue_id

    monkeypatch.setattr(shared, "_resolve_live_queue_id", resolve_queue)
    return output_dir, backend, events


def _p1_launch_argv(output_dir: Path) -> list[str]:
    return [
        "launch",
        "--output-dir",
        str(output_dir),
        "--source-dataset-id",
        "a" * 32,
        "--template-task-id",
        "b" * 32,
        "--queue",
        p1.WORKER_QUEUE,
        "--execute-token",
        p1.LAUNCH_TOKEN,
    ]


def test_p1_mock_launch_success_seals_queue_and_worker_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir, backend, events = _p1_mock_launch_case(tmp_path, monkeypatch)
    capsys.readouterr()
    assert p1.main(_p1_launch_argv(output_dir)) == 0
    launch_output = json.loads(capsys.readouterr().out)
    assert backend.clone_count == 1
    assert backend.enqueue_count == 1
    task = next(task for task_id, task in backend.tasks.items() if task_id != "b" * 32)
    assert task.status == "in_progress"
    assert task.data.execution.queue == p1.WORKER_QUEUE_ID
    assert task.data.last_worker == "mock-a100-sm80-worker"
    for key, value in p1.P1_TASK_PARAMETER_BINDINGS.items():
        assert task.parameters[key] == value

    artifacts = {
        str(record["key"]): json.loads(record["type_data"]["preview"])
        for record in task.data.execution.artifacts
    }
    assert set(artifacts) == {
        p1.LAUNCH_RECEIPT_ARTIFACT,
        p1.ENQUEUE_ACKNOWLEDGEMENT_ARTIFACT,
    }
    receipt = artifacts[p1.LAUNCH_RECEIPT_ARTIFACT]
    acknowledgement = artifacts[p1.ENQUEUE_ACKNOWLEDGEMENT_ARTIFACT]
    shared._require_seal(receipt, context="test P1 launch receipt")
    shared._require_seal(acknowledgement, context="test P1 enqueue acknowledgement")
    assert receipt["task_parameter_bindings"] == p1.P1_TASK_PARAMETER_BINDINGS
    assert acknowledgement["launch_receipt_seal_sha256"] == receipt["seal_sha256"]
    assert acknowledgement["resolved_queue_name"] == p1.WORKER_QUEUE
    assert acknowledgement["resolved_queue_id"] == p1.WORKER_QUEUE_ID
    assert acknowledgement["single_executor"] is True
    assert acknowledgement["post_enqueue_state"] == {
        "status": "in_progress",
        "execution_queue_id": p1.WORKER_QUEUE_ID,
        "execution_queue_name": p1.WORKER_QUEUE,
        "last_worker": "mock-a100-sm80-worker",
        "worker_assignment_state": "worker_assigned",
    }
    assert (
        launch_output["enqueue_acknowledgement_seal_sha256"]
        == (acknowledgement["seal_sha256"])
    )

    resolve_index = next(
        i for i, event in enumerate(events) if event[0] == "queue_resolve"
    )
    enqueue_index = next(i for i, event in enumerate(events) if event[0] == "enqueue")
    ack_upload_index = next(
        i
        for i, event in enumerate(events)
        if event[:3] == ("upload", task.id, p1.ENQUEUE_ACKNOWLEDGEMENT_ARTIFACT)
    )
    assert resolve_index < enqueue_index < ack_upload_index
    assert len([event for event in events if event[0] == "duplicate_query"]) == 2


def test_p1_wrong_live_queue_id_fails_before_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, backend, events = _p1_mock_launch_case(
        tmp_path,
        monkeypatch,
        resolved_queue_id="d" * 32,
    )
    with pytest.raises(RuntimeError, match="queue name-to-ID preflight drifted"):
        p1.main(_p1_launch_argv(output_dir))
    assert backend.clone_count == 1
    assert backend.enqueue_count == 0
    assert any(event[0] == "queue_resolve" for event in events)
    assert not any(event[0] == "enqueue" for event in events)
    assert not any(event[0] == "upload" for event in events)


@pytest.mark.parametrize(
    ("post_status", "post_queue_id", "post_last_worker", "message"),
    (
        ("in_progress", "d" * 32, "worker", "enqueue queue ID drifted"),
        ("in_progress", p1.WORKER_QUEUE_ID, None, "no auditable worker"),
        ("queued", p1.WORKER_QUEUE_ID, "worker", "premature worker"),
    ),
)
def test_p1_post_enqueue_state_drift_fails_before_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_status: str,
    post_queue_id: str,
    post_last_worker: str | None,
    message: str,
) -> None:
    output_dir, backend, _events = _p1_mock_launch_case(
        tmp_path,
        monkeypatch,
        post_status=post_status,
        post_queue_id=post_queue_id,
        post_last_worker=post_last_worker,
    )
    with pytest.raises(RuntimeError, match=message):
        p1.main(_p1_launch_argv(output_dir))
    assert backend.enqueue_count == 1
    task = next(task for task_id, task in backend.tasks.items() if task_id != "b" * 32)
    names = {record["key"] for record in task.data.execution.artifacts}
    assert p1.LAUNCH_RECEIPT_ARTIFACT in names
    assert p1.ENQUEUE_ACKNOWLEDGEMENT_ARTIFACT not in names


def test_p1_contract_and_task_parameter_binding_drift_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, _plan = _prepare(tmp_path / "p1")
    forged_bindings = dict(p1.P1_TASK_PARAMETER_BINDINGS)
    forged_bindings["Args/candidate_identity"] = "forged-candidate"
    forged_profile = replace(
        p1.P1_PROFILE,
        task_parameter_bindings=forged_bindings,
    )
    with shared.use_deployment_profile(forged_profile):
        with pytest.raises(
            RuntimeError,
            match="execution/task parameter bindings drifted",
        ):
            shared._static_plan(manifest)


@pytest.mark.parametrize("execute_token", (None, "WRONG_P1_LAUNCH_TOKEN"))
def test_default_training_launch_requires_exact_token_before_clearml_access(
    execute_token: str | None,
) -> None:
    arguments = [
        "launch",
        "--source-dataset-id",
        "a" * 32,
        "--template-task-id",
        "b" * 32,
        "--queue",
        p1.WORKER_QUEUE,
    ]
    if execute_token is not None:
        arguments.extend(("--execute-token", execute_token))
    with pytest.raises(PermissionError, match="exact execute token required"):
        p1.main(arguments)
    assert shared._profile() is shared.P0_PROFILE


@pytest.mark.parametrize(
    "command",
    (
        ["upload-source"],
        ["create-template", "--source-dataset-id", "a" * 32],
    ),
)
def test_p1_remote_writes_require_distinct_exact_tokens(
    command: list[str],
) -> None:
    with pytest.raises(PermissionError, match="exact execute token required"):
        p1.main(command)
    assert shared._profile() is shared.P0_PROFILE


def test_show_execution_contract_is_local_and_bound_to_persisted_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "p1"
    _prepare(output_dir)
    capsys.readouterr()
    assert (
        p1.main(
            [
                "show-execution-contract",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_tree_sha256"] == EXPECTED_P1_TREE_SHA256
    assert payload["execution_contract"] == p1.EXECUTION_CONTRACT
    assert payload["execution_contract"]["training_launch_enabled"] is True
    assert payload["execution_contract"]["default_candidate"] == (
        p1.P1_TRAINED_CANDIDATE
    )
