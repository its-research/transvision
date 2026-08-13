#!/usr/bin/env python3
"""Run the sealed E1/E2/E3 SOTA-candidate training queue.

This controller is deliberately separate from the active 26-task controller.  It
will not create a child until the exact predecessor controller, teacher, candidate
bootstrap template, and additive source-revision contract all verify.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from tools.resilient_v2x import clearml_5090_training_controller as training
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    _training_path = Path(__file__).with_name("clearml_5090_training_controller.py")
    _training_spec = importlib.util.spec_from_file_location(
        "clearml_5090_training_controller", _training_path
    )
    if _training_spec is None or _training_spec.loader is None:
        raise RuntimeError("cannot load the sibling training controller") from error
    training = importlib.util.module_from_spec(_training_spec)
    sys.modules[_training_spec.name] = training
    _training_spec.loader.exec_module(training)


CANDIDATE_CONTROLLER_TYPE = "resilient_v2x_sota_candidate_training_v1"
CANDIDATE_SOURCE_TRANSITION_ARTIFACT = "sota_candidate_source_transition"
CANDIDATE_PROGRESS_ARTIFACT = "sota_candidate_training_progress"
CANDIDATE_SUMMARY_ARTIFACT = "sota_candidate_training_summary"
CANDIDATE_TRAINING_MANIFEST_ARTIFACT = "sota_candidate_training_manifest"
CANDIDATE_GATE_BINDING_ARTIFACT = "sota_candidate_gate_binding"
GATE_SUMMARY_ARTIFACT = "post_main_training_summary"

CURRENT_SOURCE = {
    "dataset_id": "351feedbbe81481fa31f1e9ae11a3f4e",
    "archive_name": "resilient-v2x-source-ad511d88b731.tar.zst",
    "archive_bytes": 1_222_492,
    "archive_sha256": (
        "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
    ),
    "tree_sha256": ("ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"),
    "inventory_sha256": (
        "39b1a42af65ad5df935945bd0a4eeac6e7f6e1cfdd1cc608f3dd8a708e9c5ca0"
    ),
    "inventory_bytes": 101_195,
    "file_count": 631,
    "source_bytes": 8_926_102,
}
CANDIDATE_SOURCE_PACKAGE = {
    "archive_name": "resilient-v2x-source-8a22d6d600a5.tar.zst",
    "archive_bytes": 1_222_568,
    "archive_sha256": (
        "a249546f16f236d42c8346a33b137dea27c037bf0df73c643da4132c8e589557"
    ),
    "tree_sha256": ("8a22d6d600a52117d01cfde5fa5f20fa1de269ec9c9ba040228f5e8e3a5e8767"),
    "inventory_sha256": (
        "5bd96277989838760f3a2f9b6f3d86a4ce8a0ef79d5a21d366540b4fa7c7604f"
    ),
    "inventory_bytes": 101_714,
    "file_count": 634,
    "source_bytes": 8_927_627,
}
CANDIDATE_SPECS = (
    training.ExperimentSpec(
        "support_residual_linear",
        "sota_candidate",
        "configs/resilient_v2x/improvements/support_residual_linear.py",
        True,
    ),
    training.ExperimentSpec(
        "no_reliability_linear",
        "sota_candidate",
        "configs/resilient_v2x/improvements/no_reliability_linear.py",
        True,
    ),
    training.ExperimentSpec(
        "support_residual_no_reliability",
        "sota_candidate",
        "configs/resilient_v2x/improvements/support_residual_no_reliability.py",
        True,
    ),
)
CANDIDATE_ORDER = tuple(spec.name for spec in CANDIDATE_SPECS)
CANDIDATE_CONFIG_SHA256 = {
    "configs/resilient_v2x/improvements/support_residual_linear.py": (
        "a105124cf16a693a8c6fb176e07720abe0b05a6ff9d187b076b83d46c30bb4c0"
    ),
    "configs/resilient_v2x/improvements/no_reliability_linear.py": (
        "a42f6b28bf1678f663458ccc6f98a409e2483d6786423aa50b2c72e68b721e5d"
    ),
    "configs/resilient_v2x/improvements/support_residual_no_reliability.py": (
        "86f124aa6ae542891575c26d68560f6b4835b9ebe96516fed40074dfd5af1efa"
    ),
}
EXPECTED_GATE_ORDER = (
    "support_residual",
    "ptf_none",
    "ptf_linear",
    "router_static",
    "no_distillation",
    "coformernet",
    "router_uniform",
    "no_reliability",
    "no_delay_metadata",
    "concat_capacity_matched",
    "ffnet",
    "bevfusion",
    "v2x_vit",
    "cobevt",
    "linear_no_distillation",
    "no_distillation_peak_lr_3e4",
    "ego_only",
    "fcooper",
    "attfuse",
    "v2vnet",
    "when2com",
    "where2comm",
    "late_fusion",
    "disconet",
    "how2comm",
    "resilient_v2x",
)
_CANDIDATE_SPEC_ANCHOR = """ADDITIONAL_EXPERIMENT_SPECS = (
    ExperimentSpec(
        "linear_no_distillation",
"""
_CANDIDATE_SPEC_PATCH = """ADDITIONAL_EXPERIMENT_SPECS = (
    ExperimentSpec(
        "support_residual_linear",
        "sota_candidate",
        "configs/resilient_v2x/improvements/support_residual_linear.py",
        True,
    ),
    ExperimentSpec(
        "no_reliability_linear",
        "sota_candidate",
        "configs/resilient_v2x/improvements/no_reliability_linear.py",
        True,
    ),
    ExperimentSpec(
        "support_residual_no_reliability",
        "sota_candidate",
        "configs/resilient_v2x/improvements/support_residual_no_reliability.py",
        True,
    ),
    ExperimentSpec(
        "linear_no_distillation",
"""
_CANDIDATE_NESTED_ANCHOR = """        "concat_capacity_matched",
        "resilient_v2x",
"""
_CANDIDATE_NESTED_PATCH = """        "concat_capacity_matched",
        "support_residual_linear",
        "no_reliability_linear",
        "support_residual_no_reliability",
        "resilient_v2x",
"""
_ORIGINAL_EXPERIMENT_PATCH = training._apply_experiment_syspath_patch


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: object, context: str) -> str:
    text = str(value or "").strip()
    if training.SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return text


def _candidate_source_transition_payload(
    args: argparse.Namespace,
) -> dict[str, object]:
    target = {
        "dataset_id": training._clearml_id(
            args.candidate_source_dataset_id, "candidate source dataset"
        ),
        "archive_name": str(args.candidate_source_archive_name),
        "archive_bytes": int(args.candidate_source_archive_bytes),
        "archive_sha256": _sha256(
            args.candidate_source_archive_sha256, "candidate source archive"
        ),
        "tree_sha256": _sha256(
            args.candidate_source_tree_sha256, "candidate source tree"
        ),
        "inventory_sha256": _sha256(
            args.candidate_source_inventory_sha256, "candidate source inventory"
        ),
        "inventory_bytes": int(args.candidate_source_inventory_bytes),
        "file_count": int(args.candidate_source_file_count),
        "source_bytes": int(args.candidate_source_bytes),
    }
    if target["dataset_id"] == CURRENT_SOURCE["dataset_id"]:
        raise ValueError("candidate source must not alias the active ad511 source")
    if target["archive_sha256"] == CURRENT_SOURCE["archive_sha256"]:
        raise ValueError("candidate archive must not alias the active ad511 archive")
    if target["tree_sha256"] == CURRENT_SOURCE["tree_sha256"]:
        raise ValueError("candidate tree must not alias the active ad511 tree")
    if target["archive_bytes"] <= 0 or target["inventory_bytes"] <= 0:
        raise ValueError("candidate source byte counts must be positive")
    for key in (
        "archive_name",
        "archive_bytes",
        "archive_sha256",
        "tree_sha256",
        "inventory_sha256",
        "inventory_bytes",
        "file_count",
        "source_bytes",
    ):
        if target[key] != CANDIDATE_SOURCE_PACKAGE[key]:
            raise ValueError(f"candidate source exact package mismatch: {key}")

    additions = [
        {"path": path, "sha256": digest}
        for path, digest in CANDIDATE_CONFIG_SHA256.items()
    ]
    transition = {
        "schema_version": 1,
        "transition_type": "exact_additive_sota_candidate_source_revision",
        "controller_type": CANDIDATE_CONTROLLER_TYPE,
        "base_source": dict(CURRENT_SOURCE),
        "target_source": target,
        "inventory_delta": {
            "base_file_count": CURRENT_SOURCE["file_count"],
            "target_file_count": target["file_count"],
            "unchanged_file_count": CURRENT_SOURCE["file_count"],
            "modified_file_count": 0,
            "added_file_count": len(additions),
            "removed_file_count": 0,
            "modified_files": [],
            "added_files": additions,
        },
        "candidate_order": list(CANDIDATE_ORDER),
        "protocol": {
            "global_batch_size": 8,
            "gpu_count": 4,
            "batch_size_per_gpu": 2,
            "max_epochs": 50,
            "val_interval": 10,
            "training_seed": 20250218,
            "training_overlay_protocol_seed": 20250218,
            "precision": "FP32",
        },
    }
    return transition


def candidate_source_transition(args: argparse.Namespace) -> dict[str, object]:
    """Build the only accepted additive source transition and verify its seal."""

    transition = _candidate_source_transition_payload(args)
    observed = _sha256(args.candidate_source_transition_sha256, "source transition")
    expected = hashlib.sha256(_canonical_json(transition).encode()).hexdigest()
    if observed != expected:
        raise ValueError("candidate source transition seal mismatch")
    return {**transition, "seal_sha256": expected}


def _apply_candidate_bootstrap_patch(diff: str) -> str:
    """Add only the three candidate identities to a verified bootstrap clone."""

    if type(diff) is not str:
        raise TypeError("candidate standalone script diff must be a string")
    config_counts = {
        spec.config: diff.count(f'"{spec.config}"') for spec in CANDIDATE_SPECS
    }
    if all(count == 1 for count in config_counts.values()):
        if not all(diff.count(f'"{spec.name}"') >= 2 for spec in CANDIDATE_SPECS):
            raise RuntimeError("candidate bootstrap identity/config state is mixed")
        return diff
    if any(config_counts.values()):
        raise RuntimeError("candidate bootstrap config patch is partial")
    if diff.count(_CANDIDATE_SPEC_ANCHOR) != 1:
        raise RuntimeError("candidate bootstrap experiment-spec anchor is not unique")
    if diff.count(_CANDIDATE_NESTED_ANCHOR) != 1:
        raise RuntimeError("candidate bootstrap nested-teacher anchor is not unique")
    patched = diff.replace(_CANDIDATE_SPEC_ANCHOR, _CANDIDATE_SPEC_PATCH, 1)
    patched = patched.replace(_CANDIDATE_NESTED_ANCHOR, _CANDIDATE_NESTED_PATCH, 1)
    if not all(patched.count(f'"{spec.config}"') == 1 for spec in CANDIDATE_SPECS):
        raise RuntimeError("candidate bootstrap config patch postcondition failed")
    return patched


def _apply_candidate_experiment_patch(
    diff: str,
    *,
    include_nested_teacher_fix: bool = True,
) -> str:
    patched = _ORIGINAL_EXPERIMENT_PATCH(
        diff, include_nested_teacher_fix=include_nested_teacher_fix
    )
    return _apply_candidate_bootstrap_patch(patched)


def _install_candidate_extension() -> None:
    """Narrow the proven scheduler to the exact E1/E2/E3 identities."""

    training.EXPERIMENT_SPECS = CANDIDATE_SPECS
    training.CORE_EXPERIMENT_SPECS = CANDIDATE_SPECS
    training.ADDITIONAL_EXPERIMENT_SPECS = ()
    training.EXPERIMENT_ORDER = CANDIDATE_ORDER
    training.CORE_EXPERIMENT_ORDER = CANDIDATE_ORDER
    training.EXPERIMENT_BY_NAME = {spec.name: spec for spec in CANDIDATE_SPECS}
    training.TEACHER_DEPENDENT_EXPERIMENTS = frozenset(CANDIDATE_ORDER)
    training.PRIMARY_METHOD_EXPERIMENTS = frozenset()
    training.NESTED_TEACHER_EXPERIMENTS = frozenset(CANDIDATE_ORDER)
    training.FORMAL_1337_SUBJECT_ORDER = CANDIDATE_ORDER
    training.PROGRESS_ARTIFACT = CANDIDATE_PROGRESS_ARTIFACT
    training.SUMMARY_ARTIFACT = CANDIDATE_SUMMARY_ARTIFACT
    training.FORMAL_1337_MANIFEST_ARTIFACT = CANDIDATE_TRAINING_MANIFEST_ARTIFACT
    training._apply_experiment_syspath_patch = _apply_candidate_experiment_patch


def _validate_gate_and_template(
    args: argparse.Namespace,
    *,
    task_class: object,
    controller_task: object,
    sleeper=training.time.sleep,
) -> dict[str, object]:
    gate_id = training._clearml_id(args.gate_task_id, "candidate gate controller")
    gate = task_class.get_task(task_id=gate_id)
    training._wait_for_completed(
        gate,
        context="exact 26-task candidate predecessor controller",
        poll_seconds=args.poll_seconds,
        sleeper=sleeper,
    )
    training._reload(gate)
    summary = training._artifact_payload(gate, args.gate_summary_artifact)
    training._require_valid_seal(summary, context="candidate predecessor summary")
    observed_gate_seal = _sha256(summary.get("seal_sha256"), "gate summary seal")
    expected_gate_seal = str(args.gate_summary_seal_sha256 or "").strip()
    if expected_gate_seal and observed_gate_seal != _sha256(
        expected_gate_seal, "expected gate summary seal"
    ):
        raise RuntimeError("candidate predecessor summary seal mismatch")
    expected_gate = {
        "status": "completed",
        "controller_task_id": gate_id,
        "task_count": len(EXPECTED_GATE_ORDER),
        "experiment_order": list(EXPECTED_GATE_ORDER),
        "training_seed": 20250218,
        "training_overlay_protocol_seed": 20250218,
    }
    for key, expected in expected_gate.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"candidate predecessor summary {key} mismatch")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != len(EXPECTED_GATE_ORDER):
        raise RuntimeError("candidate predecessor result inventory mismatch")
    if [
        item.get("experiment") for item in results if isinstance(item, Mapping)
    ] != list(EXPECTED_GATE_ORDER):
        raise RuntimeError("candidate predecessor result order mismatch")

    teacher = summary.get("teacher")
    if not isinstance(teacher, Mapping):
        raise RuntimeError("candidate predecessor has no sealed teacher reference")
    expected_teacher = {
        "task_id": args.teacher_task_id,
        "model_id": args.teacher_model_id,
        "checkpoint_sha256": args.teacher_checkpoint_sha256,
        "allow_failed_task": bool(args.allow_failed_teacher_task),
    }
    for key, expected in expected_teacher.items():
        if teacher.get(key) != expected:
            raise RuntimeError(f"candidate teacher drifted from predecessor: {key}")

    transition = candidate_source_transition(args)
    template_id = training._clearml_id(args.template_task_id, "candidate template")
    template = task_class.get_task(task_id=template_id)
    identity = training._template_identity(template, expected_task_id=template_id)
    if identity.get("script_sha256") != _sha256(
        args.template_script_sha256, "candidate template script"
    ):
        raise RuntimeError("candidate template script SHA-256 mismatch")
    source = identity.get("source_parameters")
    target = transition["target_source"]
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise RuntimeError("candidate template source identity is invalid")
    expected_source = {
        "Args/source_dataset_id": target["dataset_id"],
        "Args/source_archive_name": target["archive_name"],
        "Args/source_archive_bytes": target["archive_bytes"],
        "Args/source_archive_sha256": target["archive_sha256"],
    }
    for key, expected in expected_source.items():
        if not training._parameter_matches(source.get(key), expected):
            raise RuntimeError(f"candidate template source parameter drifted: {key}")

    template_transition = training._artifact_payload(
        template, CANDIDATE_SOURCE_TRANSITION_ARTIFACT
    )
    training._require_valid_seal(
        template_transition, context="candidate template source transition"
    )
    if template_transition != transition:
        raise RuntimeError("candidate template source transition artifact drifted")
    script_diff = training._apply_experiment_syspath_patch(
        str(training._task_script(template).get("diff") or "")
    )
    for spec in CANDIDATE_SPECS:
        if script_diff.count(f'"{spec.name}"') < 1:
            raise RuntimeError(f"candidate template lacks identity {spec.name!r}")
        if script_diff.count(f'"{spec.config}"') != 1:
            raise RuntimeError(
                f"candidate template config is absent or aliased: {spec.name}"
            )
    binding = training._sealed(
        {
            "schema_version": 1,
            "binding_type": "resilient_v2x_sota_candidate_gate_binding",
            "controller_task_id": training._clearml_id(
                getattr(controller_task, "id", ""), "candidate controller"
            ),
            "gate_controller_task_id": gate_id,
            "gate_summary_artifact": args.gate_summary_artifact,
            "gate_summary_seal_sha256": observed_gate_seal,
            "teacher": expected_teacher,
            "template_task_id": template_id,
            "template_script_sha256": identity["script_sha256"],
            "candidate_source_transition_seal_sha256": transition["seal_sha256"],
            "candidate_order": list(CANDIDATE_ORDER),
            "worker_queues": training._resolve_worker_queues(args),
            "max_parallel_training_tasks": args.max_parallel,
        }
    )
    artifacts = getattr(controller_task, "artifacts", None)
    if isinstance(artifacts, Mapping) and CANDIDATE_GATE_BINDING_ARTIFACT in artifacts:
        existing = training._artifact_payload(
            controller_task, CANDIDATE_GATE_BINDING_ARTIFACT
        )
        if existing != binding:
            raise RuntimeError("candidate gate binding artifact drifted")
    else:
        training._upload_mapping(
            controller_task, CANDIDATE_GATE_BINDING_ARTIFACT, binding
        )
    return transition


def _parser() -> argparse.ArgumentParser:
    parser = training._parser()
    gate_action = next(
        action for action in parser._actions if action.dest == "gate_task_id"
    )
    gate_action.required = True
    parser.add_argument("--gate-summary-artifact", default=GATE_SUMMARY_ARTIFACT)
    parser.add_argument("--gate-summary-seal-sha256", default="")
    parser.add_argument("--template-script-sha256", required=True)
    parser.add_argument("--candidate-source-dataset-id", required=True)
    parser.add_argument("--candidate-source-archive-name", required=True)
    parser.add_argument("--candidate-source-archive-bytes", type=int, required=True)
    parser.add_argument("--candidate-source-archive-sha256", required=True)
    parser.add_argument("--candidate-source-tree-sha256", required=True)
    parser.add_argument("--candidate-source-inventory-sha256", required=True)
    parser.add_argument("--candidate-source-inventory-bytes", type=int, required=True)
    parser.add_argument("--candidate-source-file-count", type=int, required=True)
    parser.add_argument("--candidate-source-bytes", type=int, required=True)
    parser.add_argument("--candidate-source-transition-sha256", required=True)
    return parser


def _validate_candidate_arguments(args: argparse.Namespace) -> None:
    if args.paper_controller_task_id:
        raise ValueError("candidate queue requires the exact gate controller task ID")
    if args.training_seed != 20250218:
        raise ValueError("candidate training seed is sealed to 20250218")
    if not 1 <= args.max_parallel <= len(CANDIDATE_ORDER):
        raise ValueError("candidate max_parallel must be between 1 and 3")
    if args.adopt_experiment:
        raise ValueError("candidate task adoption is forbidden")
    forbidden = (
        args.recover_failed_controller_task_id,
        args.recovery_source_template_task_id,
        args.recovery_source_transition,
        args.rerun_failed_experiment,
        args.recovery_adopt_target_experiment,
        args.build_task_id,
        args.native_bundle_bytes,
        args.native_bundle_sha256,
        args.build_manifest_sha256,
    )
    if any(forbidden):
        raise ValueError("candidate queue forbids recovery and build/source overrides")
    if not args.teacher_model_id or not args.teacher_checkpoint_sha256:
        raise ValueError(
            "candidate queue requires an explicit sealed teacher model/SHA"
        )
    if args.resolve_teacher_reference:
        raise ValueError("candidate queue forbids implicit teacher resolution")


def run_candidate_suite(
    args: argparse.Namespace,
    *,
    task_class: object,
    controller_task: object,
    sleeper=training.time.sleep,
) -> dict[str, object]:
    _validate_candidate_arguments(args)
    _install_candidate_extension()
    _validate_gate_and_template(
        args,
        task_class=task_class,
        controller_task=controller_task,
        sleeper=sleeper,
    )
    return training.run_training_suite(
        args,
        task_class=task_class,
        controller_task=controller_task,
        sleeper=sleeper,
    )


def main(argv: Sequence[str] | None = None) -> int:
    from clearml import Task

    if argv is None and any(value in {"-h", "--help"} for value in sys.argv[1:]):
        _parser().parse_args(argv)
        raise AssertionError("argparse --help unexpectedly returned")
    controller_task = None
    if argv is None:
        controller_task = training._current_controller_task(
            Task, auto_connect_arg_parser=True
        )
    args = _parser().parse_args(argv)
    if controller_task is None:
        controller_task = training._current_controller_task(Task)
    controller_task.output_uri = training.FILES_SERVER_URI
    run_candidate_suite(args, task_class=Task, controller_task=controller_task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CANDIDATE_CONFIG_SHA256",
    "CANDIDATE_ORDER",
    "CANDIDATE_SOURCE_PACKAGE",
    "CANDIDATE_SPECS",
    "candidate_source_transition",
    "run_candidate_suite",
)
