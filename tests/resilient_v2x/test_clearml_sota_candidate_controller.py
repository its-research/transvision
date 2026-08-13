from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest

from tools.resilient_v2x import clearml_5090_training_controller as training
from tools.resilient_v2x import clearml_sota_candidate_controller as candidate


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "tools" / "resilient_v2x" / "clearml_5090_bootstrap.py"
GATE_ID = "a" * 32
TEACHER_TASK_ID = "b" * 32
TEACHER_MODEL_ID = "c" * 32
TEACHER_SHA256 = "d" * 64


def _source_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "candidate_source_dataset_id": "1" * 32,
        "candidate_source_archive_name": candidate.CANDIDATE_SOURCE_PACKAGE[
            "archive_name"
        ],
        "candidate_source_archive_bytes": candidate.CANDIDATE_SOURCE_PACKAGE[
            "archive_bytes"
        ],
        "candidate_source_archive_sha256": candidate.CANDIDATE_SOURCE_PACKAGE[
            "archive_sha256"
        ],
        "candidate_source_tree_sha256": candidate.CANDIDATE_SOURCE_PACKAGE[
            "tree_sha256"
        ],
        "candidate_source_inventory_sha256": candidate.CANDIDATE_SOURCE_PACKAGE[
            "inventory_sha256"
        ],
        "candidate_source_inventory_bytes": candidate.CANDIDATE_SOURCE_PACKAGE[
            "inventory_bytes"
        ],
        "candidate_source_file_count": candidate.CANDIDATE_SOURCE_PACKAGE["file_count"],
        "candidate_source_bytes": candidate.CANDIDATE_SOURCE_PACKAGE["source_bytes"],
        "candidate_source_transition_sha256": "0" * 64,
    }
    values.update(overrides)
    args = argparse.Namespace(**values)
    payload = candidate._candidate_source_transition_payload(args)
    args.candidate_source_transition_sha256 = hashlib.sha256(
        candidate._canonical_json(payload).encode()
    ).hexdigest()
    return args


def _controller_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "paper_controller_task_id": None,
        "training_seed": 20250218,
        "max_parallel": 3,
        "adopt_experiment": [],
        "recover_failed_controller_task_id": "",
        "recovery_source_template_task_id": "",
        "recovery_source_transition": "",
        "rerun_failed_experiment": [],
        "recovery_adopt_target_experiment": [],
        "build_task_id": "",
        "native_bundle_bytes": 0,
        "native_bundle_sha256": "",
        "build_manifest_sha256": "",
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": TEACHER_SHA256,
        "resolve_teacher_reference": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_candidate_identities_are_exact_distinct_and_teacher_dependent() -> None:
    assert candidate.CANDIDATE_ORDER == (
        "support_residual_linear",
        "no_reliability_linear",
        "support_residual_no_reliability",
    )
    assert len({spec.name for spec in candidate.CANDIDATE_SPECS}) == 3
    assert len({spec.config for spec in candidate.CANDIDATE_SPECS}) == 3
    assert all(spec.kind == "sota_candidate" for spec in candidate.CANDIDATE_SPECS)
    assert all(spec.requires_teacher for spec in candidate.CANDIDATE_SPECS)
    assert {spec.config for spec in candidate.CANDIDATE_SPECS} == set(
        candidate.CANDIDATE_CONFIG_SHA256
    )


def test_candidate_file_receipts_and_clone_only_bootstrap_patch_match() -> None:
    for relative_path, expected_sha256 in candidate.CANDIDATE_CONFIG_SHA256.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == (
            expected_sha256
        )
    source = BOOTSTRAP.read_text(encoding="utf-8")
    for spec in candidate.CANDIDATE_SPECS:
        assert f'"{spec.config}"' not in source
        assert f'"{spec.name}"' not in source
    patched = candidate._apply_candidate_bootstrap_patch(source)
    assert candidate._apply_candidate_bootstrap_patch(patched) == patched
    for spec in candidate.CANDIDATE_SPECS:
        assert patched.count(f'"{spec.config}"') == 1
        assert patched.count(f'"{spec.name}"') >= 2


def test_candidate_source_transition_is_exact_additive_and_sealed() -> None:
    args = _source_args()
    transition = candidate.candidate_source_transition(args)

    assert transition["seal_sha256"] == args.candidate_source_transition_sha256
    assert transition["base_source"] == candidate.CURRENT_SOURCE
    assert {
        key: transition["target_source"][key]
        for key in candidate.CANDIDATE_SOURCE_PACKAGE
    } == candidate.CANDIDATE_SOURCE_PACKAGE
    assert transition["candidate_order"] == list(candidate.CANDIDATE_ORDER)
    delta = transition["inventory_delta"]
    assert delta == {
        "base_file_count": 631,
        "target_file_count": 634,
        "unchanged_file_count": 631,
        "modified_file_count": 0,
        "added_file_count": 3,
        "removed_file_count": 0,
        "modified_files": [],
        "added_files": [
            {"path": path, "sha256": sha256}
            for path, sha256 in candidate.CANDIDATE_CONFIG_SHA256.items()
        ],
    }
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "candidate_source_dataset_id",
            candidate.CURRENT_SOURCE["dataset_id"],
            "must not alias",
        ),
        (
            "candidate_source_archive_sha256",
            candidate.CURRENT_SOURCE["archive_sha256"],
            "must not alias",
        ),
        (
            "candidate_source_tree_sha256",
            candidate.CURRENT_SOURCE["tree_sha256"],
            "must not alias",
        ),
        ("candidate_source_file_count", 635, "exact package mismatch: file_count"),
        ("candidate_source_bytes", 8_927_628, "exact package mismatch: source_bytes"),
        (
            "candidate_source_archive_name",
            "wrong.tar.zst",
            "exact package mismatch: archive_name",
        ),
    ),
)
def test_candidate_source_rejects_aliases_and_nonallowlisted_inventory(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _source_args(**{field: value})


def test_candidate_source_rejects_wrong_transition_seal() -> None:
    args = _source_args()
    args.candidate_source_transition_sha256 = "f" * 64
    with pytest.raises(ValueError, match="transition seal mismatch"):
        candidate.candidate_source_transition(args)


def test_extension_generates_exact_task_protocol_and_teacher_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "EXPERIMENT_SPECS",
        "CORE_EXPERIMENT_SPECS",
        "ADDITIONAL_EXPERIMENT_SPECS",
        "EXPERIMENT_ORDER",
        "CORE_EXPERIMENT_ORDER",
        "EXPERIMENT_BY_NAME",
        "TEACHER_DEPENDENT_EXPERIMENTS",
        "PRIMARY_METHOD_EXPERIMENTS",
        "NESTED_TEACHER_EXPERIMENTS",
        "FORMAL_1337_SUBJECT_ORDER",
        "PROGRESS_ARTIFACT",
        "SUMMARY_ARTIFACT",
        "FORMAL_1337_MANIFEST_ARTIFACT",
        "_apply_experiment_syspath_patch",
    )
    for name in names:
        monkeypatch.setattr(training, name, getattr(training, name))
    candidate._install_candidate_extension()

    template = {
        "Args/source_dataset_id": "1" * 32,
        "Args/source_archive_name": "resilient-v2x-source-333333333333.tar.zst",
        "Args/source_archive_bytes": 1_300_000,
        "Args/source_archive_sha256": "2" * 64,
        "Args/training_dataset_id": "5" * 32,
        "Args/native_bundle_bytes": 753_382_966,
        "Args/native_bundle_sha256": "6" * 64,
        "Args/build_manifest_sha256": "7" * 64,
    }
    for experiment in candidate.CANDIDATE_ORDER:
        params = training._experiment_parameters(
            template,
            experiment=experiment,
            predecessor_task_id=GATE_ID,
            teacher_task_id=TEACHER_TASK_ID,
            teacher_model_id=TEACHER_MODEL_ID,
            teacher_checkpoint_sha256=TEACHER_SHA256,
            allow_failed_teacher_task=False,
        )
        assert params["Args/experiment_from_task"] == experiment
        assert params["Args/predecessor_task_id"] == GATE_ID
        assert params["Args/teacher_task_id"] == TEACHER_TASK_ID
        assert params["Args/teacher_model_id"] == TEACHER_MODEL_ID
        assert params["Args/teacher_checkpoint_sha256"] == TEACHER_SHA256
        assert params["Args/source_dataset_id"] == "1" * 32
        assert params["Args/gpus"] == 4
        assert params["Args/max_epochs"] == 50
        assert params["Args/training_seed"] == 20250218
        assert params["Args/amp"] is False


def test_candidate_predecessor_policy_is_explicit_and_never_forward() -> None:
    steps = [
        {"index": index, "state": "pending", "task_id": None} for index in range(1, 4)
    ]
    assert training._latest_completed_predecessor(steps, gate_task_id=GATE_ID) == (
        GATE_ID
    )
    steps[0].update(state="completed", task_id="1" * 32)
    assert (
        training._latest_completed_predecessor(
            steps, gate_task_id=GATE_ID, before_index=2
        )
        == "1" * 32
    )
    steps[1].update(state="completed", task_id="2" * 32)
    assert (
        training._latest_completed_predecessor(
            steps, gate_task_id=GATE_ID, before_index=3
        )
        == "2" * 32
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"training_seed": 20250219}, "seed is sealed"),
        ({"max_parallel": 4}, "between 1 and 3"),
        ({"adopt_experiment": ["support_residual_linear=" + "1" * 32]}, "forbidden"),
        ({"resolve_teacher_reference": True}, "implicit teacher"),
        ({"teacher_model_id": ""}, "explicit sealed teacher"),
        ({"build_task_id": "1" * 32}, "build/source overrides"),
    ),
)
def test_candidate_arguments_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        candidate._validate_candidate_arguments(_controller_args(**overrides))


def test_valid_candidate_arguments_keep_parallelism_explicit() -> None:
    for max_parallel in (1, 2, 3):
        candidate._validate_candidate_arguments(
            _controller_args(max_parallel=max_parallel)
        )
