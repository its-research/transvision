from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from pathlib import Path
import pytest
import torch
from torch import nn

from tools.resilient_v2x import clearml_p1_zero_shot_candidate_extension as module
from tools.resilient_v2x import evaluate_controlled_baselines as base_evaluator
from tools.resilient_v2x import evaluate_p1_zero_shot as adapter


class _Artifact:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self, force_download: bool = False) -> object:
        del force_download
        return copy.deepcopy(self.value)


class _Model:
    def __init__(
        self,
        *,
        model_id: str,
        task_id: str,
        name: str,
        url: str,
        local: Path,
        iteration: int = 50,
    ) -> None:
        self.id = model_id
        self.task = task_id
        self.name = name
        self.url = url
        self.iteration = iteration
        self.local = local

    def get_local_copy(self, **_: object) -> str:
        return str(self.local)


class _Task:
    def __init__(
        self,
        *,
        parameters: dict[str, object],
        artifacts: dict[str, object],
        models: list[_Model],
    ) -> None:
        self.id = module.P0_TASK_ID
        self.status = "completed"
        self._parameters = parameters
        self.artifacts = {name: _Artifact(value) for name, value in artifacts.items()}
        self._models = models

    def reload(self) -> bool:
        return True

    def get_parameters(self, cast: bool = False) -> dict[str, object]:
        del cast
        return copy.deepcopy(self._parameters)

    def get_models(self) -> dict[str, list[_Model]]:
        return {"output": list(self._models)}


def _valid_p0_task(tmp_path: Path) -> tuple[_Task, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / module.P0_FINAL_REMOTE_FILENAME
    checkpoint.write_bytes(b"canonical-p0-epoch50")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_id = "1" * 32
    best_model_id = "2" * 32
    model_url = f"{module.FILES_SERVER_URI}/models/{module.P0_FINAL_REMOTE_FILENAME}"
    final = {
        "model_id": model_id,
        "name": module.P0_FINAL_MODEL_NAME,
        "url": model_url,
        "filename": module.P0_FINAL_SOURCE_FILENAME,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": checkpoint_sha,
    }
    best = {
        "model_id": best_model_id,
        "name": (
            "ResilientV2X support_residual_no_reliability_linear "
            "clean-val best checkpoint"
        ),
        "url": (
            f"{module.FILES_SERVER_URI}/models/"
            "support_residual_no_reliability_linear_clean_val_best_epoch_30.pth"
        ),
        "filename": "best_resilient_v2x_car_bev_ap_r40_0.70_epoch_30.pth",
        "size_bytes": 1,
        "sha256": "b" * 64,
        "epoch": 30,
        "selection_protocol": "DAIR-CLEAN-PAIR1789-v1",
        "selection_metric": "resilient_v2x/car_bev_ap_r40_0.70",
        "claim_role": "diagnostic checkpoint candidate; final remains canonical",
    }
    run_contract = {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": module.P0_TASK_ID,
        "experiment": module.P0_CONFIG_SUBJECT,
        "experiment_kind": "sota_candidate",
        "config": {
            "declared": module.P0_CONFIG,
            "declared_resolved": module.P0_CONFIG,
            "config_sha256": module.P0_CONFIG_SHA256,
            "resolved": module.P0_CONFIG,
            "size_bytes": 1,
            "resolved_config_sha256": module.P0_CONFIG_SHA256,
        },
        "training_dataset_id": module.TRAINING_DATASET_ID,
        "gpus": 4,
        "global_batch_size": 8,
        "train_batch_size_per_gpu": 2,
        "max_epochs": 50,
        "training_seed": module.TRAINING_SEED,
        "training_overlay_protocol_seed": module.TRAINING_OVERLAY_PROTOCOL_SEED,
        "seed": module.TRAINING_SEED,
        "amp": False,
        "precision": "FP32",
        "val_interval": 10,
        "per_epoch_validation": False,
        "condition_evaluation": False,
        "common_teacher_initialization": {
            "audit_artifact_name": module.INITIALIZATION_AUDIT_ARTIFACT,
        },
    }
    initialization = {
        "schema_version": 1,
        "result": "pass",
        "contract": "shared-only-clean-teacher-initialization-v1",
        "shared_initialization": {
            "shape_dtype_verified": True,
            "exact_tensor_equality_verified": True,
        },
    }
    parameters = {
        "Args/stage": "all",
        "Args/experiment_from_task": module.P0_CONFIG_SUBJECT,
        "Args/training_dataset_id": module.TRAINING_DATASET_ID,
        "Args/training_seed": str(module.TRAINING_SEED),
        "Args/gpus": "4",
        "Args/max_epochs": "50",
        "Args/amp": "False",
    }
    model = _Model(
        model_id=model_id,
        task_id=module.P0_TASK_ID,
        name=module.P0_FINAL_MODEL_NAME,
        url=model_url,
        local=checkpoint,
    )
    task = _Task(
        parameters=parameters,
        artifacts={
            module.FINAL_CHECKPOINT_ARTIFACT: final,
            module.BEST_CHECKPOINT_ARTIFACT: best,
            module.RUN_CONTRACT_ARTIFACT: run_contract,
            module.INITIALIZATION_AUDIT_ARTIFACT: initialization,
        },
        models=[model],
    )
    return task, checkpoint


def test_canonical_resolver_pins_only_p0_epoch50_final(tmp_path: Path) -> None:
    task, checkpoint = _valid_p0_task(tmp_path)
    binding = module.resolve_p0_canonical_final(task)
    module._require_seal(binding, context="test binding")

    assert binding["candidate_identity"] == module.ZERO_SHOT_CANDIDATE_IDENTITY
    assert binding["candidate_identity"] == (
        "dair_improvement_reliability_gated_residual_zero_shot_p0_epoch50_final"
    )
    assert (
        binding["identity_alias_mapping_seal_sha256"]
        == (module.identity_alias_mapping()["seal_sha256"])
    )
    assert binding["candidate_identity"] != module.P1_TRAINED_CANDIDATE_IDENTITY
    assert binding["config_subject"] == module.P1_CONFIG_SUBJECT
    assert binding["checkpoint_subject"] == module.P0_CONFIG_SUBJECT
    assert binding["weights_retrained"] is False
    assert binding["optimization_origin"] == "P0"
    assert binding["forbidden_checkpoint_roles"] == list(
        module.FORBIDDEN_CHECKPOINT_ROLES
    )
    checkpoint_contract = binding["checkpoint"]
    assert checkpoint_contract["source_filename"] == "epoch_50.pth"
    assert checkpoint_contract["remote_filename"] == checkpoint.name
    assert checkpoint_contract["bytes_recomputed"] is True
    assert "local_path" not in checkpoint_contract
    assert module.download_bound_p0_checkpoint(task, binding) == checkpoint.resolve()
    assert binding["formal_evaluation"] == {
        "protocol_id": "DAIR-CAUSAL-1337-v1",
        "sample_count": 1337,
        "training_seed": 20250218,
        "overlay_protocol_seed": 20250218,
        "delays_ms": [0, 100, 200, 300],
        "conditions": ["Full", "L-Fail", "C-Fail"],
        "run_count": 12,
    }


def test_identity_alias_mapping_is_sealed_one_to_one_and_never_an_identity() -> None:
    mapping = module.identity_alias_mapping()
    seal = module.validate_identity_alias_mapping(mapping)
    assert seal == mapping["seal_sha256"]
    assert mapping["canonical_identity"] == module.ZERO_SHOT_CANDIDATE_IDENTITY
    assert mapping["display_subject_alias"] == module.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS
    assert mapping["aliases"] == {
        module.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS: module.ZERO_SHOT_CANDIDATE_IDENTITY
    }
    assert mapping["mapping_cardinality"] == "one_to_one"
    assert mapping["alias_is_candidate_identity"] is False
    assert mapping["canonical_identity"] != mapping["display_subject_alias"]

    tampered = copy.deepcopy(mapping)
    tampered["canonical_identity"] = module.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS
    tampered = module._sealed(tampered)
    with pytest.raises(module.P1CheckpointReuseError, match="canonical map"):
        module.validate_identity_alias_mapping(tampered)


@pytest.mark.parametrize(
    "mutator,match",
    (
        (
            lambda task: setattr(task, "status", "in_progress"),
            "not completed",
        ),
        (
            lambda task: setattr(task._models[0], "task", "f" * 32),
            "binding mismatch",
        ),
        (
            lambda task: task._models.append(copy.copy(task._models[0])),
            "exactly one",
        ),
        (
            lambda task: task.artifacts[module.BEST_CHECKPOINT_ARTIFACT].value.update(
                {"model_id": "1" * 32}
            ),
            "roles are conflated",
        ),
        (
            lambda task: setattr(
                task._models[0],
                "url",
                "https://example.com/support_residual_no_reliability_linear_epoch_50.pth",
            ),
            "not trusted",
        ),
        (
            lambda task: setattr(task._models[0], "iteration", 40),
            "not epoch 50",
        ),
    ),
)
def test_canonical_resolver_fails_closed(
    tmp_path: Path,
    mutator: object,
    match: str,
) -> None:
    task, _ = _valid_p0_task(tmp_path)
    mutator(task)
    with pytest.raises(module.P1CheckpointReuseError, match=match):
        module.resolve_p0_canonical_final(task)


def test_canonical_resolver_recomputes_checkpoint_bytes_and_sha(tmp_path: Path) -> None:
    task, checkpoint = _valid_p0_task(tmp_path)
    checkpoint.write_bytes(b"tampered-after-contract")
    with pytest.raises(module.P1CheckpointReuseError, match="byte size mismatch|SHA"):
        module.resolve_p0_canonical_final(task)


def test_resolved_launch_contract_is_machine_path_independent(tmp_path: Path) -> None:
    first_task, first_checkpoint = _valid_p0_task(tmp_path / "first")
    second_task, second_checkpoint = _valid_p0_task(tmp_path / "second")
    assert first_checkpoint != second_checkpoint
    first = module.resolved_launch_contract(first_task)
    second = module.resolved_launch_contract(second_task)
    assert first == second
    assert first["remote_state_changed"] is False
    binding = first["checkpoint_reuse_binding"]
    assert "local_path" not in binding["checkpoint"]
    assert first["evaluation_parameters"]["Args/controlled_baseline"] == (
        module.ZERO_SHOT_CANDIDATE_IDENTITY
    )


def test_config_subject_split_accepts_only_false_to_true_gate() -> None:
    common = {
        "type": "ResilientV2XNet",
        "ptf_mode": "linear",
        "support_residual_weight": 0.5,
        "use_reliability": False,
        "teacher": {"type": "Teacher"},
        "distillation": {"weight": 1.0},
    }
    p1 = copy.deepcopy(common)
    p1["support_residual_reliability_gate"] = True
    contract = module.validate_config_subject_split(common, p1)
    assert contract["only_deployment_model_delta"] == (
        "support_residual_reliability_gate"
    )
    assert contract["p0_value"] is False
    assert contract["p1_value"] is True

    changed = copy.deepcopy(p1)
    changed["support_residual_weight"] = 0.6
    with pytest.raises(module.P1CheckpointReuseError, match="differ beyond"):
        module.validate_config_subject_split(common, changed)


def test_real_resolved_p0_p1_deployment_configs_differ_only_by_gate() -> None:
    p0 = base_evaluator._load_python_config(Path(module.P0_CONFIG))
    p1 = base_evaluator._load_python_config(Path(module.P1_CONFIG))
    contract = module.validate_config_subject_split(p0["model"], p1["model"])
    assert contract == {
        "config_subject": module.P1_CONFIG_SUBJECT,
        "checkpoint_subject": module.P0_CONFIG_SUBJECT,
        "only_deployment_model_delta": "support_residual_reliability_gate",
        "p0_value": False,
        "p1_value": True,
        "training_only_fields_removed": [
            "teacher",
            "teacher_checkpoint",
            "distillation",
        ],
        "normalized_deployment_model_sha256": (
            "a4fcd71a97cd5884ce6136940f50a984d5564d11108e045b109c98b11f2118ff"
        ),
    }


class _TinyStudent(nn.Module):
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.register_buffer("scale", torch.ones(width))


def _checkpoint_fixture() -> tuple[
    _TinyStudent,
    _TinyStudent,
    OrderedDict[str, torch.Tensor],
    dict[str, object],
]:
    p0 = _TinyStudent()
    p1 = _TinyStudent()
    student = OrderedDict(
        (key, torch.full_like(value, index + 1))
        for index, (key, value) in enumerate(p0.state_dict().items())
    )
    nested = OrderedDict(
        (
            ("teacher.teacher.encoder.weight", torch.ones(2, 2)),
            ("teacher.teacher.encoder.bias", torch.ones(2)),
        )
    )
    full = OrderedDict(p0.state_dict())
    full.update(nested)
    checkpoint_state = OrderedDict(student)
    checkpoint_state.update(nested)
    return p0, p1, full, {"state_dict": checkpoint_state}


def test_filtered_checkpoint_schema_and_strict_load_are_exact() -> None:
    p0, p1, full, checkpoint = _checkpoint_fixture()
    filtered, schema = module.prepare_filtered_student_state(
        checkpoint,
        p0_student_state=p0.state_dict(),
        p1_student_state=p1.state_dict(),
        p0_full_state=full,
    )
    assert list(filtered) == list(p0.state_dict())
    assert schema["missing_student_keys"] == []
    assert schema["extra_checkpoint_keys"] == [
        "teacher.teacher.encoder.weight",
        "teacher.teacher.encoder.bias",
    ]
    loaded = module.strict_load_equivalent_students(p0, p1, filtered)
    assert loaded["strict_load"] is True
    assert loaded["missing_keys"] == []
    assert loaded["unexpected_keys"] == []
    assert loaded["loaded_tensor_mapping_equal"] is True
    for key, value in filtered.items():
        torch.testing.assert_close(p0.state_dict()[key], value, rtol=0.0, atol=0.0)
        torch.testing.assert_close(p1.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_filtered_checkpoint_rejects_unsealed_extra_and_schema_drift() -> None:
    p0, p1, full, checkpoint = _checkpoint_fixture()
    checkpoint["state_dict"]["unexpected.weight"] = torch.ones(1)
    with pytest.raises(module.P1CheckpointReuseError, match="exact sealed"):
        module.prepare_filtered_student_state(
            checkpoint,
            p0_student_state=p0.state_dict(),
            p1_student_state=p1.state_dict(),
            p0_full_state=full,
        )

    _, _, full, checkpoint = _checkpoint_fixture()
    wrong_p1 = _TinyStudent(width=3)
    with pytest.raises(module.P1CheckpointReuseError, match="schemas differ"):
        module.prepare_filtered_student_state(
            checkpoint,
            p0_student_state=p0.state_dict(),
            p1_student_state=wrong_p1.state_dict(),
            p0_full_state=full,
        )


def test_evaluation_parameters_keep_config_and_checkpoint_subjects_separate(
    tmp_path: Path,
) -> None:
    task, _ = _valid_p0_task(tmp_path)
    binding = module.resolve_p0_canonical_final(task)
    parameters = module.evaluation_parameters(binding)
    assert parameters["Args/stage"] == "checkpoint_reuse_validate"
    assert parameters["Args/controlled_baseline"] == (
        module.ZERO_SHOT_CANDIDATE_IDENTITY
    )
    assert (
        parameters["Args/candidate_identity_alias_mapping_sha256"]
        == (module.identity_alias_mapping()["seal_sha256"])
    )
    assert parameters["Args/controlled_baseline_config_subject"] == (
        module.P1_CONFIG_SUBJECT
    )
    assert parameters["Args/controlled_baseline_checkpoint_subject"] == (
        module.P0_CONFIG_SUBJECT
    )
    assert parameters["Args/controlled_baseline_task_id"] == module.P0_TASK_ID
    assert parameters["Args/sample_count"] == 1337
    assert parameters["Args/training_seed"] == 20250218
    assert parameters["Args/weights_retrained"] is False
    module.validate_evaluation_parameters(parameters, binding)

    merged = dict(parameters)
    merged["Args/controlled_baseline_checkpoint_subject"] = module.P1_CONFIG_SUBJECT
    with pytest.raises(module.P1CheckpointReuseError, match="parameter|merged"):
        module.validate_evaluation_parameters(merged, binding)


def test_evaluator_adapter_is_idempotent_and_does_not_edit_canonical_subject_order() -> (
    None
):
    original_improvements = base_evaluator.IMPROVEMENTS
    original_subjects = base_evaluator.EVALUATION_SUBJECTS
    original_improvement_configs = dict(base_evaluator.IMPROVEMENT_CONFIGS)
    original_configs = dict(base_evaluator.EVALUATION_CONFIGS)
    try:
        assert adapter.CONFIG_SUBJECT not in base_evaluator.EVALUATION_SUBJECTS
        adapter.install_extension_subject()
        adapter.install_extension_subject()
        assert base_evaluator.EVALUATION_SUBJECTS.count(adapter.CONFIG_SUBJECT) == 1
        assert base_evaluator.IMPROVEMENTS.count(adapter.CONFIG_SUBJECT) == 1
        assert base_evaluator.EVALUATION_CONFIGS[adapter.CONFIG_SUBJECT] == (
            adapter.CONFIG_PATH
        )
    finally:
        base_evaluator.IMPROVEMENTS = original_improvements
        base_evaluator.EVALUATION_SUBJECTS = original_subjects
        base_evaluator.IMPROVEMENT_CONFIGS.clear()
        base_evaluator.IMPROVEMENT_CONFIGS.update(original_improvement_configs)
        base_evaluator.EVALUATION_CONFIGS.clear()
        base_evaluator.EVALUATION_CONFIGS.update(original_configs)


def test_local_contract_is_extension_only_and_seed_is_not_sample_count() -> None:
    contract = module.local_execution_contract()
    module._require_seal(contract, context="test local contract")
    assert contract["original_26_method_chain_mutated"] is False
    assert contract["result_destination"] == "candidate_extension_leaderboard_only"
    assert contract["sample_count"] == 1337
    assert contract["training_seed"] == 20250218
    assert contract["sample_count"] != contract["training_seed"]
    assert contract["weights_retrained"] is False
    assert contract["candidate_identity"] == module.ZERO_SHOT_CANDIDATE_IDENTITY
    assert contract["display_subject_alias"] == module.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS
    assert (
        contract["identity_alias_mapping_seal_sha256"]
        == (module.identity_alias_mapping()["seal_sha256"])
    )
    assert contract["required_gpu_capabilities"] == [[8, 0]] * 4


def test_original_26_chain_files_remain_at_their_preextension_hashes() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "tools/resilient_v2x/clearml_1337_leaderboard.py": (
            "1da0cf5dd4435c6ae85d5a474b5a67ec8435f50e9878a3d8f0c1d2872356524b"
        ),
        "tools/resilient_v2x/clearml_formal_candidate_selector.py": (
            "bc340f4a5165152022a70c4d9958560b911aa8f348055ca50a2583efa7045bf5"
        ),
        "tools/resilient_v2x/evaluate_controlled_baselines.py": (
            "d233e054f2b608bc441de25833fb89d117197d841752812a0054e0514995ad36"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
