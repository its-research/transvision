#!/usr/bin/env python3
"""Publish a sealed formal multi-seed plan after both pinned gates complete."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

try:
    from allegroai import Task
    from allegroai.binding.artifacts import Artifact as ClearMLArtifact
except ImportError:
    try:
        from clearml import Task
        from clearml.binding.artifacts import Artifact as ClearMLArtifact
    except ImportError:
        Task = None  # type: ignore[assignment]
        ClearMLArtifact = None  # type: ignore[assignment,misc]


DEFAULT_PROJECT = "ResilientV2X/Training"
FILES_SERVER_URI = "http://10.100.34.118:8081"

TRAINING_CONTROLLER_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
TRAINING_PROVENANCE_TASK_ID = "7274a1a5344a44c18aa7bcc6d1cb2e95"
WATCHER_TASK_ID = "7734387ddfb74b11ba6d84f3fea0bb97"
LEADERBOARD_TASK_ID = "f502bdd329ad4ef4b4b6cf5c5f52aba0"
AUDIT_TASK_ID = "e19bab92884248f4ac07167e7eb66170"
SELECTOR_TASK_ID = "b8876e88bb494985a900e3d627439776"
SELECTOR_AUDIT_TASK_ID = AUDIT_TASK_ID
SELECTOR_LEADERBOARD_TASK_ID = LEADERBOARD_TASK_ID
SELECTOR_TRAINING_CONTROLLER_TASK_ID = TRAINING_CONTROLLER_TASK_ID
SELECTOR_TRAINING_PROVENANCE_TASK_ID = TRAINING_PROVENANCE_TASK_ID
SELECTOR_WATCHER_TASK_ID = WATCHER_TASK_ID
SELECTOR_POLL_SECONDS = "60.0"
SELECTOR_TIMEOUT_HOURS = "720.0"
LEADERBOARD_POLL_SECONDS = "60.0"
LEADERBOARD_TIMEOUT_HOURS = "720.0"

SOURCE_D_ENTRY_POINT = "clearml_formal_source_d_evidence.py"
TRAINING_CONTROLLER_ENTRY_POINT = "clearml_5090_training_controller.py"
TRAINING_PROVENANCE_ENTRY_POINT = "clearml_formal_training_provenance.py"
WATCHER_ENTRY_POINT = "clearml_1337_dependency_watcher.py"
SELECTOR_ENTRY_POINT = "clearml_formal_candidate_selector.py"
AUDIT_ENTRY_POINT = "clearml_formal_comparability_audit.py"
LEADERBOARD_ENTRY_POINT = "clearml_1337_leaderboard.py"
PRODUCER_ENTRY_POINT = "formal_multiseed_plan.py"
PLAN_VALIDATOR_SHA256 = (
    "a7bc17bcd1afb1c836924f853f63acae8e26801557f784cd6f0bcd7148c35993"
)
SOURCE_D_PRODUCER_SHA256 = (
    "d5b759f38d39a9f349ab6e716c07fda53eb3e1635687ce4a077e0405920ffeec"
)
TRAINING_CONTROLLER_PRODUCER_SHA256 = (
    "22fe485f3eb999c85b464e260a4376bc81a577c15e6a429585e48df06ffd0678"
)
TRAINING_PROVENANCE_PRODUCER_SHA256 = (
    "981abba51adf5f8c3b78960a14c2200a2fd68260e74273a52984de869825f77a"
)
WATCHER_PRODUCER_SHA256 = (
    "be493eae42b0f50204dbbad3d2dd60671d8480226d1c0176bd912bf2a47f2624"
)
SELECTOR_PRODUCER_SHA256 = (
    "4610802b0854fc3d6b58db9b2863bdc1ab1757c83dccbcd677dd577374f86ac1"
)
AUDIT_PRODUCER_SHA256 = (
    "4746ae18bb67757c974d96e851f19cb01872f460aaf3b334609a0de988aaabaa"
)
LEADERBOARD_PRODUCER_SHA256 = (
    "1da0cf5dd4435c6ae85d5a474b5a67ec8435f50e9878a3d8f0c1d2872356524b"
)
# Intentionally unresolved until a real completed leaderboard task exists.
# Empty pins make every attempted publication fail closed; they must be replaced
# from one fresh server read of that immutable completed task before deployment.
LEADERBOARD_ARTIFACT_SEAL_SHA256 = ""  # REQUIRED_DEPLOYMENT_PIN
LEADERBOARD_ARTIFACT_CANONICAL_SHA256 = ""  # REQUIRED_DEPLOYMENT_PIN
LEGACY_TRAINING_SCRIPT_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
CANONICAL_TRAINING_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
EVALUATION_SCRIPT_SHA256 = CANONICAL_TRAINING_SCRIPT_SHA256
LEGACY_SCRIPT_SUBJECTS = ("support_residual", "no_distillation")
LEGACY_PARENT_TASK_ID = "6525107e60ae4104a2800731d74ecd4e"
NO_DISTILLATION_PARENT_TASK_ID = "d4b83d9b68704050aeb24a2e34540d8a"
RECOVERY_PARENT_TASK_ID = "bfcf18a3fd484776adcc5efd48a2c95e"
SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID = "f8c36e508c7d453dadc766207a5b25b2"
# Internal aliases used only by the dormant Source-D migration verifier below.
# The publication path never resolves a Source-D task; no retired task ID remains.
SOURCE_D_TASK_ID = TRAINING_CONTROLLER_TASK_ID
SOURCE_C_TASK_ID = TRAINING_CONTROLLER_TASK_ID
SOURCE_C_PARENT_TASK_ID = ""
SOURCE_C_SCRIPT_SHA256 = LEGACY_TRAINING_SCRIPT_SHA256
SOURCE_D_SCRIPT_SHA256 = (
    "e7a9ab0fb05339223cf2c18c52eb72652c311733bf96d1058aa7a769096cf8c3"
)
SOURCE_D_EQUIVALENCE_SHA256 = (
    "1156fe53f2fe924f91c1c6b50b6b21090d98cd2d74840d6d6f5a358316420433"
)
SOURCE_C_SNAPSHOT_SEAL_SHA256 = (
    "c6426dc3734db478934809217457bffaba4d571ac597132f014de6b0a6e08cb8"
)
SOURCE_C_SNAPSHOT_CANONICAL_SHA256 = (
    "d348264bc73d1895eb47856636b0707d4e539247ed8b6a63eac011c2518dd45d"
)
SOURCE_D_ARTIFACT_SEAL_SHA256 = (
    "49d3d20dcb358a6c483128e20f231767d3fe5c1eae33a2192cd1717d1a2d19b6"
)
SOURCE_D_ARTIFACT_CANONICAL_SHA256 = (
    "ceedef8add2e0c9d889e6ff4e9862cc951bbcb924d232ee896cd2d660301a1ae"
)
SOURCE_D_EQUIVALENCE_CANONICAL_SHA256 = (
    "607e261ae0e0b364d816eda33392a693a3810120575e7fa1e3662f9051d64bda"
)
SOURCE_D_RECEIPT_SEAL_SHA256 = (
    "1c4b3cfeb617eacc6aa1f7d194dee687c6cf51f47e6adf4987eaaed6d796c62d"
)
SOURCE_D_RECEIPT_CANONICAL_SHA256 = (
    "0bfc1c48719681ecd253df1e37c2e7c5e2edf43b56c1ae02b9cb15f316f2625b"
)
SOURCE_D_BUILDER_SHA256 = (
    "904fafd08d710b03b62bc57140121f2a546ada8fa2fb8763e6db8a44b9f8f7e7"
)
SOURCE_D_TRANSFORMATION_ID = "source-c-to-source-d-explicit-seed-evidence-v2"
TRAINING_OVERLAY_PROTOCOL_SEED = 20_250_218
DECLARED_REPLACEMENT_COUNT = 22
UNCHANGED_SEGMENT_COUNT = 23
PORTABLE_RUNNER_LOAD_MARKER = "_validate_rtx5090_runtime_contract_multi_gpu"
PORTABLE_RUNNER_LOAD_MARKER_COUNT = 2
SOURCE_C_ENTRY_POINT = "clearml_5090_bootstrap.py"
SOURCE_D_STAGING_ANCHOR_NAMES = (
    "seal_evidence_security_imports",
    "declare_controlled_evidence_contract",
    "declare_controlled_evidence_receipts",
    "stage_and_verify_controlled_evidence",
    "stage_after_controlled_runner",
    "upload_only_sealed_controlled_evidence",
)

AUDIT_PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
AUDIT_SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
AUDIT_SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
AUDIT_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
AUDIT_SOURCE_ARCHIVE_BYTES = 1_222_481
AUDIT_SOURCE_TREE_SHA256 = (
    "5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6"
)
AUDIT_SOURCE_INVENTORY_BYTES = 101_195
AUDIT_SOURCE_INVENTORY_SHA256 = (
    "bed72cd86438f2ba932edda14052e3cd3d9589ec201d09d18a315e18a2f7cff2"
)
AUDIT_SOURCE_FILE_COUNT = 631
AUDIT_SOURCE_BYTES = 8_926_106
AUDIT_NEW_SOURCE_DATASET_ID = "351feedbbe81481fa31f1e9ae11a3f4e"
AUDIT_NEW_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-ad511d88b731.tar.zst"
AUDIT_NEW_SOURCE_ARCHIVE_BYTES = 1_222_492
AUDIT_NEW_SOURCE_ARCHIVE_SHA256 = (
    "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
)
AUDIT_NEW_SOURCE_TREE_SHA256 = (
    "ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"
)
AUDIT_NEW_SOURCE_INVENTORY_BYTES = 101_195
AUDIT_NEW_SOURCE_INVENTORY_SHA256 = (
    "39b1a42af65ad5df935945bd0a4eeac6e7f6e1cfdd1cc608f3dd8a708e9c5ca0"
)
AUDIT_NEW_SOURCE_FILE_COUNT = 631
AUDIT_NEW_SOURCE_BYTES = 8_926_102
SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = (
    "29de9700cac66f9998be643e85a8ec646c04ec17fddbb6207bc1438e9e73941b"
)
SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = (
    "c3170f4a88b080f9cc7267053f354640dd260687cb2c35a6f7ffed73d69f4154"
)
SOURCE_REVISION_TRANSITION_SEAL_SHA256 = (
    "1c08edf7daeea7676ceb806d68dc51a4dedd5650d54665d675fe45ab750c6b78"
)
SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256 = (
    "4f06e07445e3297a8f9284ca20c78b27c3f36b4c7263ce5c7c3d9accc0a5a79c"
)
AUDIT_TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
AUDIT_TRAINING_SEED = 20_250_218
AUDIT_SAMPLE_COUNT = 1_337
AUDIT_GROUND_TRUTH_COUNT = 11_330
AUDIT_UNSUPPORTED_SAMPLE_COUNT = 0
AUDIT_DELAYS_MS = (0, 100, 200, 300)
AUDIT_CONDITIONS = ("Full", "L-Fail", "C-Fail")
AUDIT_SUBJECT_ORDER = (
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
AUDIT_NEW_SOURCE_SUBJECTS = frozenset(
    {"where2comm", "late_fusion", "disconet", "how2comm", "resilient_v2x"}
)
SOURCE_REVISION_CERTIFICATE_BY_TREE = {
    AUDIT_SOURCE_TREE_SHA256: {
        "dataset_id": AUDIT_SOURCE_DATASET_ID,
        "tree_sha256": AUDIT_SOURCE_TREE_SHA256,
        "file_count": AUDIT_SOURCE_FILE_COUNT,
        "source_bytes": AUDIT_SOURCE_BYTES,
        "archive": {
            "name": AUDIT_SOURCE_ARCHIVE_NAME,
            "size_bytes": AUDIT_SOURCE_ARCHIVE_BYTES,
            "sha256": AUDIT_SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": AUDIT_SOURCE_INVENTORY_BYTES,
            "sha256": AUDIT_SOURCE_INVENTORY_SHA256,
        },
    },
    AUDIT_NEW_SOURCE_TREE_SHA256: {
        "dataset_id": AUDIT_NEW_SOURCE_DATASET_ID,
        "tree_sha256": AUDIT_NEW_SOURCE_TREE_SHA256,
        "file_count": AUDIT_NEW_SOURCE_FILE_COUNT,
        "source_bytes": AUDIT_NEW_SOURCE_BYTES,
        "archive": {
            "name": AUDIT_NEW_SOURCE_ARCHIVE_NAME,
            "size_bytes": AUDIT_NEW_SOURCE_ARCHIVE_BYTES,
            "sha256": AUDIT_NEW_SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": AUDIT_NEW_SOURCE_INVENTORY_BYTES,
            "sha256": AUDIT_NEW_SOURCE_INVENTORY_SHA256,
        },
    },
}
AUDIT_BASELINE_SUBJECTS = (
    "coformernet",
    "ffnet",
    "bevfusion",
    "v2x_vit",
    "cobevt",
    "ego_only",
    "fcooper",
    "attfuse",
    "v2vnet",
    "when2com",
    "where2comm",
    "late_fusion",
    "disconet",
    "how2comm",
)
AUDIT_METRIC_KEYS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)
AUDIT_SUPPORTED_QUEUES = frozenset({"GPU4-A100", "GPU4-V100", "GPU4-5090"})
AUDIT_EVALUATION_SCRIPT_SHA256 = EVALUATION_SCRIPT_SHA256
AUDIT_CHECKPOINT_POLICY = "epoch_50_final_only"
AUDIT_MODEL_BYTES_VERIFIER = "tools/resilient_v2x/collect_clearml_formal_models.py"

SOURCE_C_SNAPSHOT_ARTIFACT = "formal_source_c_snapshot"
SOURCE_D_SCRIPT_ARTIFACT = "formal_source_d_script"
SOURCE_D_EQUIVALENCE_ARTIFACT = "formal_source_d_equivalence"
SOURCE_D_RECEIPT_ARTIFACT = "formal_source_d_evidence_receipt"
SOURCE_D_ARTIFACTS = (
    SOURCE_C_SNAPSHOT_ARTIFACT,
    SOURCE_D_SCRIPT_ARTIFACT,
    SOURCE_D_EQUIVALENCE_ARTIFACT,
    SOURCE_D_RECEIPT_ARTIFACT,
)
SELECTOR_ARTIFACT = "formal_candidate_selection"
AUDIT_ARTIFACT = "formal_1337_comparability_audit"
LEADERBOARD_ARTIFACT = "formal_1337_leaderboard"
TRAINING_MANIFEST_ARTIFACT = "formal_1337_training_manifest"
TRAINING_PROGRESS_ARTIFACT = "post_main_training_progress"
TRAINING_SUMMARY_ARTIFACT = "post_main_training_summary"
TRAINING_PROVENANCE_ARTIFACT = "formal_1337_training_provenance_equivalence"
EVALUATION_PLAN_ARTIFACT = "formal_1337_evaluation_plan"
PLAN_ARTIFACT = "formal_multiseed_plan"
PLAN_RECEIPT_ARTIFACT = "formal_multiseed_plan_receipt"
PUBLICATION_ORDER = (PLAN_ARTIFACT, PLAN_RECEIPT_ARTIFACT)

WAITING_STATUSES = frozenset({"created", "queued", "in_progress"})
FAILED_STATUSES = frozenset(
    {"failed", "stopped", "closed", "published", "publishing", "rejected", "unknown"}
)
MAX_JSON_ARTIFACT_BYTES = 64 * 1024 * 1024

# The standalone generator replaces this one exact anchor.
_EMBEDDED_PLAN_VALIDATOR_B64 = ""  # __FORMAL_MULTISEED_PLAN_BYTES_V1__


class FormalMultiseedPlanError(RuntimeError):
    """Raised when any planner provenance or publication invariant fails."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    parser.add_argument(
        "--emit-standalone",
        type=Path,
        help="write a new standalone producer with the pinned planner embedded",
    )
    return parser


def _validate_json_domain(value: object, *, context: str = "document") -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise FormalMultiseedPlanError(
                    f"{context} mapping keys must be exact strings"
                )
            _validate_json_domain(item, context=f"{context}.{key}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_domain(item, context=f"{context}[{index}]")
        return
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise FormalMultiseedPlanError(
        f"{context} is outside the exact canonical JSON domain"
    )


def _canonical_json(value: object) -> str:
    _validate_json_domain(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _content_sha256(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _frozen_mapping(value: object, *, context: str) -> tuple[dict[str, object], str]:
    if not isinstance(value, Mapping):
        raise FormalMultiseedPlanError(f"{context} must be a JSON object")
    try:
        canonical = _canonical_json(value)
        frozen = json.loads(canonical)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FormalMultiseedPlanError(f"{context} is not canonical JSON") from error
    if type(frozen) is not dict:
        raise FormalMultiseedPlanError(f"{context} must be a JSON object")
    return frozen, _sha256_text(canonical)


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    frozen, _ = _frozen_mapping(value, context="sealed payload")
    frozen.pop("seal_sha256", None)
    frozen["seal_sha256"] = _content_sha256(frozen)
    return frozen


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: object, *, context: str) -> str:
    if not _is_lower_hex(value, 64):
        raise FormalMultiseedPlanError(f"{context} must be a lowercase SHA-256")
    return value


def _clearml_id(value: object, *, context: str) -> str:
    if not _is_lower_hex(value, 32):
        raise FormalMultiseedPlanError(
            f"{context} must be a lowercase 32-hex ClearML ID"
        )
    return value


def _exact_bool(value: object, expected: bool, *, context: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise FormalMultiseedPlanError(f"{context} must be exactly {expected!r}")
    return value


def _exact_int(value: object, expected: int, *, context: str) -> int:
    if type(value) is not int or value != expected:
        raise FormalMultiseedPlanError(f"{context} must be exactly {expected}")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise FormalMultiseedPlanError(
            f"{context} keys mismatch; missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = _sha256(value.get("seal_sha256"), context=f"{context} seal")
    if _sealed(value)["seal_sha256"] != observed:
        raise FormalMultiseedPlanError(f"{context} seal SHA-256 mismatch")
    return observed


def _require_content_hash(value: Mapping[str, object], *, context: str) -> str:
    observed = _sha256(
        value.get("artifact_sha256"), context=f"{context} artifact SHA-256"
    )
    unhashed = dict(value)
    unhashed.pop("artifact_sha256", None)
    if _content_sha256(unhashed) != observed:
        raise FormalMultiseedPlanError(f"{context} artifact SHA-256 mismatch")
    return observed


def _task_parent(task: object, *, context: str) -> str:
    value = getattr(task, "parent", None)
    if value is None or value == "":
        value = getattr(getattr(task, "data", None), "parent", None)
    if value is None:
        return ""
    if type(value) is not str:
        raise FormalMultiseedPlanError(f"{context} parent must be a string")
    if value:
        _clearml_id(value, context=f"{context} parent")
    return value


def _status(task: object, *, context: str) -> str:
    value = getattr(getattr(task, "data", None), "status", None)
    value = getattr(value, "value", value)
    allowed = WAITING_STATUSES | FAILED_STATUSES | {"completed"}
    if type(value) is not str or value not in allowed:
        raise FormalMultiseedPlanError(f"{context} status is unavailable")
    return value


def _reload(task: object, *, context: str) -> None:
    expected_id = _clearml_id(getattr(task, "id", None), context=f"{context} local ID")
    if bool(getattr(task, "_offline_mode", False)):
        raise FormalMultiseedPlanError(f"{context} cannot use offline reload")
    reloader = getattr(task, "_reload", None)
    if not callable(reloader):
        raise FormalMultiseedPlanError(f"{context} cannot be server-reloaded")
    has_flag = hasattr(task, "_reload_skip_flag")
    old_flag = getattr(task, "_reload_skip_flag", None)
    try:
        if has_flag:
            setattr(task, "_reload_skip_flag", False)
        snapshot = reloader()
    except Exception as error:
        raise FormalMultiseedPlanError(f"{context} server reload failed") from error
    finally:
        if has_flag:
            setattr(task, "_reload_skip_flag", old_flag)
    if snapshot is None or isinstance(
        snapshot, (bool, int, float, str, bytes, bytearray)
    ):
        raise FormalMultiseedPlanError(f"{context} server reload returned no snapshot")
    snapshot_id = _clearml_id(
        getattr(snapshot, "id", None), context=f"{context} server snapshot ID"
    )
    if snapshot_id != expected_id:
        raise FormalMultiseedPlanError(f"{context} server snapshot identity mismatch")
    try:
        setattr(task, "_data", snapshot)
    except Exception as error:
        raise FormalMultiseedPlanError(
            f"{context} server snapshot cannot be installed"
        ) from error
    if (
        getattr(task, "_data", None) is not snapshot
        or getattr(task, "data", None) is not snapshot
    ):
        raise FormalMultiseedPlanError(
            f"{context} server snapshot installation did not persist"
        )


def _script_metadata(
    task: object,
    *,
    context: str,
    entry_point: str,
    expected_sha256: str | None = None,
    expected_source: str | None = None,
) -> tuple[str, dict[str, object]]:
    script = getattr(getattr(task, "data", None), "script", None)
    if script is None:
        raise FormalMultiseedPlanError(f"{context} has no script metadata")
    repository = getattr(script, "repository", None)
    working_dir = getattr(script, "working_dir", None)
    observed_entry = getattr(script, "entry_point", None)
    source = getattr(script, "diff", None)
    if repository != "" or working_dir != "." or observed_entry != entry_point:
        raise FormalMultiseedPlanError(f"{context} script provenance drifted")
    if type(source) is not str or not source:
        raise FormalMultiseedPlanError(f"{context} script diff is not raw text")
    observed_sha = _sha256_text(source)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise FormalMultiseedPlanError(f"{context} script SHA-256 mismatch")
    if expected_source is not None and source != expected_source:
        raise FormalMultiseedPlanError(f"{context} script bytes drifted")
    return source, {
        "repository": repository,
        "working_dir": working_dir,
        "entry_point": observed_entry,
        "sha256": observed_sha,
        "size_bytes": len(source.encode("utf-8")),
        "line_count": len(source.splitlines()),
    }


def _parameters(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise FormalMultiseedPlanError(f"{context} parameters are unavailable")
    try:
        values = getter()
    except Exception as error:
        raise FormalMultiseedPlanError(
            f"{context} parameters cannot be read"
        ) from error
    if not isinstance(values, Mapping):
        raise FormalMultiseedPlanError(f"{context} parameters are invalid")
    frozen, _ = _frozen_mapping(values, context=f"{context} parameters")
    return frozen


def _validate_selector_parameters(task: object) -> dict[str, object]:
    values = _parameters(task, context="formal selector")
    expected = {
        "Args/audit_task_id": SELECTOR_AUDIT_TASK_ID,
        "Args/leaderboard_task_id": SELECTOR_LEADERBOARD_TASK_ID,
        "Args/poll_seconds": SELECTOR_POLL_SECONDS,
        "Args/timeout_hours": SELECTOR_TIMEOUT_HOURS,
    }
    if _canonical_json(values) != _canonical_json(expected):
        raise FormalMultiseedPlanError("formal selector parameter contract mismatch")
    return values


def _validate_audit_parameters(task: object) -> dict[str, object]:
    values = _parameters(task, context="formal comparability audit")
    expected = {
        "Args/training_controller_task_id": SELECTOR_TRAINING_CONTROLLER_TASK_ID,
        "Args/training_provenance_task_id": SELECTOR_TRAINING_PROVENANCE_TASK_ID,
        "Args/watcher_task_id": SELECTOR_WATCHER_TASK_ID,
        "Args/leaderboard_task_id": SELECTOR_LEADERBOARD_TASK_ID,
        "Args/poll_seconds": SELECTOR_POLL_SECONDS,
        "Args/timeout_hours": SELECTOR_TIMEOUT_HOURS,
    }
    if _canonical_json(values) != _canonical_json(expected):
        raise FormalMultiseedPlanError(
            "formal comparability audit parameter contract mismatch"
        )
    return values


def _validate_training_provenance_parameters(task: object) -> dict[str, object]:
    values = _parameters(task, context="formal training provenance")
    expected = {
        "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
        "Args/poll_seconds": SELECTOR_POLL_SECONDS,
        "Args/timeout_hours": SELECTOR_TIMEOUT_HOURS,
    }
    if _canonical_json(values) != _canonical_json(expected):
        raise FormalMultiseedPlanError(
            "formal training provenance parameter contract mismatch"
        )
    return values


def _validate_watcher_parameters(task: object) -> dict[str, object]:
    values = _parameters(task, context="formal evaluation watcher")
    expected = {
        "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
        "Args/training_provenance_task_id": TRAINING_PROVENANCE_TASK_ID,
        "Args/evaluation_template_task_id": "8b77a3674dfe405388aae39ef82d06ef",
        "Args/expected_training_provenance_script_sha256": (
            TRAINING_PROVENANCE_PRODUCER_SHA256
        ),
        "Args/expected_eval_script_sha256": EVALUATION_SCRIPT_SHA256,
        "Args/expected_source_dataset_id": AUDIT_SOURCE_DATASET_ID,
        "Args/expected_source_archive_sha256": AUDIT_SOURCE_ARCHIVE_SHA256,
        "Args/expected_training_source_dataset_id": AUDIT_SOURCE_DATASET_ID,
        "Args/expected_training_source_archive_sha256": (AUDIT_SOURCE_ARCHIVE_SHA256),
        "Args/expected_training_dataset_id": AUDIT_TRAINING_DATASET_ID,
        "Args/poll_seconds": SELECTOR_POLL_SECONDS,
        "Args/timeout_hours": SELECTOR_TIMEOUT_HOURS,
    }
    if any(
        values.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise FormalMultiseedPlanError(
            "formal evaluation watcher parameter contract mismatch"
        )
    return values


def _expected_training_script_sha256(subject: str) -> str:
    if subject not in AUDIT_SUBJECT_ORDER:
        raise FormalMultiseedPlanError(f"unknown formal audit subject {subject!r}")
    if subject in LEGACY_SCRIPT_SUBJECTS:
        return LEGACY_TRAINING_SCRIPT_SHA256
    return CANONICAL_TRAINING_SCRIPT_SHA256


def _expected_training_parent_task_id(subject: str) -> str:
    if subject == "support_residual":
        return LEGACY_PARENT_TASK_ID
    if subject == "no_distillation":
        return NO_DISTILLATION_PARENT_TASK_ID
    if subject in {"ptf_none", "ptf_linear", "router_static"}:
        return RECOVERY_PARENT_TASK_ID
    if subject not in AUDIT_SUBJECT_ORDER:
        raise FormalMultiseedPlanError(f"unknown formal audit subject {subject!r}")
    if subject in AUDIT_NEW_SOURCE_SUBJECTS:
        return TRAINING_CONTROLLER_TASK_ID
    return SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID


def _require_deployment_pins() -> None:
    for value, context in (
        (PLAN_VALIDATOR_SHA256, "pure plan validator"),
        (LEADERBOARD_ARTIFACT_SEAL_SHA256, "leaderboard artifact seal"),
        (
            LEADERBOARD_ARTIFACT_CANONICAL_SHA256,
            "leaderboard artifact canonical",
        ),
    ):
        if not _is_lower_hex(value, 64):
            raise FormalMultiseedPlanError(f"{context} deployment pin is unresolved")


def _validate_leaderboard_parameters(task: object) -> dict[str, object]:
    values = _parameters(task, context="formal leaderboard")
    expected = {
        "Args/watcher_task_id": SELECTOR_WATCHER_TASK_ID,
        "Args/training_controller_task_id": (SELECTOR_TRAINING_CONTROLLER_TASK_ID),
        "Args/poll_seconds": LEADERBOARD_POLL_SECONDS,
        "Args/timeout_hours": LEADERBOARD_TIMEOUT_HOURS,
    }
    if _canonical_json(values) != _canonical_json(expected):
        raise FormalMultiseedPlanError("formal leaderboard parameter contract mismatch")
    return values


def _server_artifacts(task: object, *, context: str) -> dict[str, object]:
    """Materialize artifacts exclusively from the installed raw server snapshot."""

    data = getattr(task, "data", None)
    execution = getattr(data, "execution", None)
    raw_artifacts = getattr(execution, "artifacts", None)
    if raw_artifacts is None:
        raw_artifacts = ()
    if type(raw_artifacts) not in {list, tuple}:
        raise FormalMultiseedPlanError(f"{context} raw server artifacts are invalid")
    if ClearMLArtifact is None:
        raise FormalMultiseedPlanError(
            f"{context} ClearML artifact reader is unavailable"
        )
    result = {}
    for raw in raw_artifacts:
        name = getattr(raw, "key", None)
        if type(name) is not str or not name or name in result:
            raise FormalMultiseedPlanError(
                f"{context} raw server artifact inventory is invalid"
            )
        try:
            result[name] = ClearMLArtifact(raw)
        except Exception as error:
            raise FormalMultiseedPlanError(
                f"{context} raw server artifact {name!r} cannot be materialized"
            ) from error
    if any(type(name) is not str or not name for name in result):
        raise FormalMultiseedPlanError(
            f"{context} raw server artifact names are invalid"
        )
    return result


def _artifact_names(task: object, *, context: str) -> tuple[str, ...]:
    return tuple(_server_artifacts(task, context=context))


def _validate_output_inventory(names: Sequence[str]) -> None:
    observed = set(names)
    if len(observed) != len(names) or observed not in (
        set(),
        {PLAN_ARTIFACT},
        {PLAN_ARTIFACT, PLAN_RECEIPT_ARTIFACT},
    ):
        raise FormalMultiseedPlanError("planner output artifact prefix drifted")


def _read_json_path(value: str | Path, *, context: str) -> Mapping[str, object]:
    path = Path(value)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or before.st_nlink != 1
                or opened.st_nlink != 1
                or before.st_dev != opened.st_dev
                or before.st_ino != opened.st_ino
                or opened.st_size > MAX_JSON_ARTIFACT_BYTES
            ):
                raise FormalMultiseedPlanError(
                    f"{context} local artifact path is unsafe"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor, min(1024 * 1024, MAX_JSON_ARTIFACT_BYTES + 1 - total)
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_JSON_ARTIFACT_BYTES:
                    raise FormalMultiseedPlanError(
                        f"{context} local artifact is too large"
                    )
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final_path = path.lstat()
    except FormalMultiseedPlanError:
        raise
    except OSError as error:
        raise FormalMultiseedPlanError(
            f"{context} local artifact cannot be read"
        ) from error
    identities = (
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        (
            final_path.st_dev,
            final_path.st_ino,
            final_path.st_size,
            final_path.st_mtime_ns,
        ),
    )
    if identities[0] != identities[1] or identities[1] != identities[2]:
        raise FormalMultiseedPlanError(f"{context} local artifact changed during read")
    try:
        document = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FormalMultiseedPlanError(
            f"{context} local artifact is not JSON"
        ) from error
    if not isinstance(document, Mapping):
        raise FormalMultiseedPlanError(f"{context} artifact is not a JSON object")
    return document


def _artifact_mapping(task: object, name: str, *, context: str) -> Mapping[str, object]:
    artifacts = _server_artifacts(task, context=context)
    if name not in artifacts:
        raise FormalMultiseedPlanError(f"{context} lacks artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise FormalMultiseedPlanError(f"{context} artifact {name!r} cannot be read")
    try:
        value = getter(force_download=True)
    except Exception as error:
        raise FormalMultiseedPlanError(
            f"{context} artifact {name!r} cannot be downloaded"
        ) from error
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        return _read_json_path(value, context=f"{context} artifact {name!r}")
    raise FormalMultiseedPlanError(f"{context} artifact {name!r} is not JSON")


def _runtime_source() -> str:
    try:
        value = Path(__file__).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FormalMultiseedPlanError(
            "cannot read the running planner producer"
        ) from error
    if not value:
        raise FormalMultiseedPlanError("running planner producer is empty")
    return value


def _validator_source_bytes(validator_path: Path | None = None) -> bytes:
    if _EMBEDDED_PLAN_VALIDATOR_B64:
        try:
            source = base64.b64decode(
                _EMBEDDED_PLAN_VALIDATOR_B64.encode("ascii"), validate=True
            )
        except (ValueError, UnicodeError) as error:
            raise FormalMultiseedPlanError(
                "embedded planner is invalid base64"
            ) from error
    else:
        path = validator_path or Path(__file__).with_name("formal_multiseed_plan.py")
        try:
            source = path.read_bytes()
        except OSError as error:
            raise FormalMultiseedPlanError("cannot read the pinned planner") from error
    if _sha256_bytes(source) != PLAN_VALIDATOR_SHA256:
        raise FormalMultiseedPlanError("planner byte SHA-256 mismatch")
    try:
        source.decode("utf-8")
    except UnicodeError as error:
        raise FormalMultiseedPlanError("planner is not UTF-8") from error
    return source


def _load_validator(validator_path: Path | None = None) -> ModuleType:
    source = _validator_source_bytes(validator_path)
    name = "_resilient_v2x_formal_multiseed_plan_pinned"
    module = ModuleType(name)
    module.__file__ = "<pinned-formal-multiseed-plan.py>"
    sys.modules[name] = module
    try:
        exec(compile(source.decode("utf-8"), module.__file__, "exec"), module.__dict__)
    except Exception as error:
        sys.modules.pop(name, None)
        raise FormalMultiseedPlanError("cannot execute the pinned planner") from error
    for required in ("build_plan", "validate_plan"):
        if not callable(getattr(module, required, None)):
            raise FormalMultiseedPlanError(f"pinned planner lacks {required}")
    return module


def generate_standalone_source(
    *,
    validator_path: Path | None = None,
    wrapper_path: Path | None = None,
) -> str:
    validator_source = _validator_source_bytes(validator_path)
    path = wrapper_path or Path(__file__)
    try:
        wrapper = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FormalMultiseedPlanError("cannot read the planner wrapper") from error
    assignment = " ".join(("_EMBEDDED_PLAN_VALIDATOR_B64", "=", '""'))
    marker = " ".join(("#", "__FORMAL_MULTISEED_PLAN_BYTES_V1__"))
    anchor = assignment + "  " + marker
    if wrapper.count(anchor) != 1:
        raise FormalMultiseedPlanError("standalone planner anchor count mismatch")
    encoded = base64.b64encode(validator_source).decode("ascii")
    replacement = (
        f"_EMBEDDED_PLAN_VALIDATOR_B64 = {encoded!r}  "
        "# __FORMAL_MULTISEED_PLAN_BYTES_V1__"
    )
    standalone = wrapper.replace(anchor, replacement, 1)
    try:
        compile(standalone, "<clearml-formal-multiseed-plan>", "exec")
    except SyntaxError as error:
        raise FormalMultiseedPlanError(
            "generated standalone does not compile"
        ) from error
    if encoded not in standalone:
        raise FormalMultiseedPlanError("generated standalone lost planner bytes")
    return standalone


def _wait_for_dependencies(
    source_d: object,
    selector: object,
    *,
    deadline: float,
    poll_seconds: float,
    monotonic_clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    dependencies = (("Source-D evidence", source_d), ("formal selector", selector))
    while True:
        waiting = False
        for context, task in dependencies:
            _reload(task, context=context)
            status_value = _status(task, context=context)
            if status_value == "completed":
                continue
            if status_value in FAILED_STATUSES:
                raise FormalMultiseedPlanError(f"{context} ended as {status_value!r}")
            if status_value not in WAITING_STATUSES:
                raise FormalMultiseedPlanError(
                    f"{context} has unexpected status {status_value!r}"
                )
            waiting = True
        if not waiting:
            return
        if monotonic_clock() >= deadline:
            raise TimeoutError("timed out waiting for formal planner dependencies")
        sleeper(poll_seconds)


def _wait_for_chain(
    dependencies: Sequence[tuple[str, object]],
    *,
    deadline: float,
    poll_seconds: float,
    monotonic_clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    while True:
        waiting = False
        for context, task in dependencies:
            _reload(task, context=context)
            status_value = _status(task, context=context)
            if status_value == "completed":
                continue
            if status_value in FAILED_STATUSES:
                raise FormalMultiseedPlanError(f"{context} ended as {status_value!r}")
            waiting = True
        if not waiting:
            return
        if monotonic_clock() >= deadline:
            raise TimeoutError("timed out waiting for formal planner chain")
        sleeper(poll_seconds)


def _orchestrator_snapshot(
    task: object,
    *,
    task_id: str,
    parent_task_id: str,
    context: str,
    entry_point: str,
    script_sha256: str,
    artifact_names: set[str],
    parameter_validator: Callable[[object], dict[str, object]] | None = None,
) -> dict[str, object]:
    observed_id = _clearml_id(getattr(task, "id", None), context=f"{context} task")
    if observed_id != task_id:
        raise FormalMultiseedPlanError(f"{context} identity mismatch")
    if _status(task, context=context) != "completed":
        raise FormalMultiseedPlanError(f"{context} is not completed")
    if _task_parent(task, context=context) != parent_task_id:
        raise FormalMultiseedPlanError(f"{context} parent mismatch")
    _, script = _script_metadata(
        task,
        context=context,
        entry_point=entry_point,
        expected_sha256=script_sha256,
    )
    observed_artifacts = _artifact_names(task, context=context)
    if set(observed_artifacts) != artifact_names or len(observed_artifacts) != len(
        artifact_names
    ):
        raise FormalMultiseedPlanError(f"{context} artifact inventory drifted")
    parameters = parameter_validator(task) if parameter_validator is not None else None
    snapshot: dict[str, object] = {
        "task_id": observed_id,
        "parent_task_id": parent_task_id,
        "status": "completed",
        "script": script,
        "artifact_names": sorted(artifact_names),
    }
    if parameters is not None:
        snapshot["parameters"] = parameters
    return snapshot


def _load_training_provenance_validator(source: str) -> ModuleType:
    module = ModuleType("_resilient_v2x_pinned_training_provenance_validator")
    module.__file__ = f"<{TRAINING_PROVENANCE_ENTRY_POINT}>"
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    except Exception as error:
        raise FormalMultiseedPlanError(
            "formal training provenance validator cannot be loaded"
        ) from error
    validator = getattr(module, "_validate_progress", None)
    expected_constants = {
        "PROGRESS_ARTIFACT": TRAINING_PROGRESS_ARTIFACT,
        "SOURCE_REVISION_TRANSITION_SEAL_SHA256": (
            SOURCE_REVISION_TRANSITION_SEAL_SHA256
        ),
        "SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256": (
            SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256
        ),
    }
    if not callable(validator) or any(
        getattr(module, key, None) != value
        for key, value in expected_constants.items()
    ):
        raise FormalMultiseedPlanError(
            "formal training provenance validator contract drifted"
        )
    return module


def _validate_schema4_recovery_shape(recovery_value: object) -> dict[str, object]:
    if not isinstance(recovery_value, Mapping):
        raise FormalMultiseedPlanError(
            "formal training progress lacks schema-v4 recovery"
        )
    recovery, _ = _frozen_mapping(
        recovery_value, context="formal training schema-v4 recovery"
    )
    _exact_keys(
        recovery,
        {
            "schema_version",
            "mode",
            "source_controller_task_id",
            "source_controller_status",
            "source_progress_revision",
            "source_progress_seal_sha256",
            "source_progress_artifact_readback",
            "source_template_script_sha256",
            "target_template_script_sha256",
            "source_revision_transition",
            "source_progress_template_equivalence",
            "source_patch",
            "source_recovery_chain",
            "rerun_experiments",
            "rerun_source_task_ids",
            "rerun_task_observations",
            "rerun_predecessor_task_ids",
            "completion_validation_retries",
            "recovered_pending_target_children",
            "transition_replaced_source_tasks",
            "target_template_predecessor_task_ids",
            "recovery_target_adoptions",
            "adopted_task_ids",
            "adopted_predecessor_task_ids",
            "adopted_task_template_roles",
            "adopted_task_binding_receipts",
        },
        context="formal training schema-v4 recovery",
    )
    if (
        recovery.get("schema_version") != 4
        or recovery.get("mode") != "failed_controller_immutable_fork"
        or recovery.get("source_controller_status") != "failed"
        or recovery.get("source_patch") != "nested-teacher-config-consistency-v1"
    ):
        raise FormalMultiseedPlanError(
            "formal training recovery is not the schema-v4 immutable fork"
        )
    source_controller_task_id = _clearml_id(
        recovery.get("source_controller_task_id"),
        context="formal training recovery source controller",
    )
    if source_controller_task_id == TRAINING_CONTROLLER_TASK_ID:
        raise FormalMultiseedPlanError(
            "formal training recovery source aliases the target controller"
        )
    source_revision = recovery.get("source_progress_revision")
    if type(source_revision) is not int or source_revision < 1:
        raise FormalMultiseedPlanError(
            "formal training recovery source progress revision is invalid"
        )
    source_progress_seal = _sha256(
        recovery.get("source_progress_seal_sha256"),
        context="formal training recovery source progress seal",
    )
    readback = recovery.get("source_progress_artifact_readback")
    if not isinstance(readback, Mapping):
        raise FormalMultiseedPlanError(
            "formal training recovery source progress readback is invalid"
        )
    _exact_keys(
        readback,
        {"url", "revision", "seal_sha256", "force_download", "stable_readbacks"},
        context="formal training recovery source progress readback",
    )
    url = readback.get("url")
    if (
        type(url) is not str
        or source_controller_task_id not in url
        or TRAINING_PROGRESS_ARTIFACT not in url
        or readback.get("revision") != source_revision
        or readback.get("seal_sha256") != source_progress_seal
        or readback.get("force_download") is not True
        or readback.get("stable_readbacks") != 2
    ):
        raise FormalMultiseedPlanError(
            "formal training recovery source progress readback drifted"
        )
    transition = recovery.get("source_revision_transition")
    template_equivalence = recovery.get("source_progress_template_equivalence")
    if not isinstance(transition, Mapping) or not isinstance(
        template_equivalence, Mapping
    ):
        raise FormalMultiseedPlanError(
            "formal training recovery source transition evidence is invalid"
        )
    transition_seal = _require_seal(
        transition, context="formal training recovery source revision transition"
    )
    template_equivalence_seal = _require_seal(
        template_equivalence,
        context="formal training recovery source progress template equivalence",
    )
    source_template_identity = transition.get("source_template_identity")
    target_template_identity = transition.get("target_template_identity")
    if not isinstance(source_template_identity, Mapping) or not isinstance(
        target_template_identity, Mapping
    ):
        raise FormalMultiseedPlanError(
            "formal training recovery transition template identities are invalid"
        )
    if (
        transition_seal != SOURCE_REVISION_TRANSITION_SEAL_SHA256
        or template_equivalence_seal
        != SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256
        or transition.get("source_controller_task_id") != source_controller_task_id
        or template_equivalence.get("expected_identity")
        != source_template_identity
        or recovery.get("source_template_script_sha256")
        != source_template_identity.get("script_sha256")
        or recovery.get("target_template_script_sha256")
        != target_template_identity.get("script_sha256")
    ):
        raise FormalMultiseedPlanError(
            "formal training recovery transition/template binding drifted"
        )
    receipts = recovery.get("adopted_task_binding_receipts")
    adopted_ids = recovery.get("adopted_task_ids")
    if (
        not isinstance(receipts, Mapping)
        or not isinstance(adopted_ids, Mapping)
        or set(receipts) != set(adopted_ids)
        or not receipts
    ):
        raise FormalMultiseedPlanError(
            "formal training recovery adopted receipt inventory drifted"
        )
    legacy_receipt_seen = False
    for subject, raw_receipt in receipts.items():
        if not isinstance(raw_receipt, Mapping):
            raise FormalMultiseedPlanError(
                f"formal training recovery receipt {subject} is invalid"
            )
        _exact_keys(
            raw_receipt,
            {
                "task_id",
                "experiment",
                "template_role",
                "template_task_id",
                "template_script_sha256",
                "task_script_sha256",
                "script_identity_policy",
                "legacy_script_compatibility_receipt",
                "source_parameters",
                "config_path",
                "config_inventory_status",
                "expected_parameter_count",
                "expected_parameters_sha256",
                "observed_parameter_projection_sha256",
                "exact_execution_parameter_match",
                "predecessor_task_id",
            },
            context=f"formal training recovery receipt {subject}",
        )
        if (
            raw_receipt.get("experiment") != subject
            or raw_receipt.get("task_id") != adopted_ids[subject]
            or raw_receipt.get("exact_execution_parameter_match") is not True
        ):
            raise FormalMultiseedPlanError(
                f"formal training recovery receipt {subject} is cross-spliced"
            )
        policy = raw_receipt.get("script_identity_policy")
        legacy_receipt = raw_receipt.get("legacy_script_compatibility_receipt")
        if policy == "exact_canonical_template":
            if legacy_receipt is not None:
                raise FormalMultiseedPlanError(
                    f"formal training recovery receipt {subject} has a false legacy exception"
                )
        elif policy == "exact_allowlisted_legacy_nested_teacher_semantics_preserving":
            if not isinstance(legacy_receipt, Mapping):
                raise FormalMultiseedPlanError(
                    f"formal training recovery receipt {subject} lacks legacy evidence"
                )
            _require_seal(
                legacy_receipt,
                context=f"formal training recovery receipt {subject} legacy evidence",
            )
            legacy_receipt_seen = True
        else:
            raise FormalMultiseedPlanError(
                f"formal training recovery receipt {subject} script policy drifted"
            )
    if not legacy_receipt_seen:
        raise FormalMultiseedPlanError(
            "formal training recovery lacks the sealed legacy compatibility receipts"
        )
    return recovery


def _validate_training_provenance_document(
    document_value: Mapping[str, object],
    *,
    controller_task_id: str,
    progress: Mapping[str, object],
    progress_seal: str,
    recovery: Mapping[str, object],
) -> None:
    document, _ = _frozen_mapping(
        document_value, context="formal training provenance document"
    )
    _exact_keys(
        document,
        {
            "schema_version",
            "document_type",
            "protocol_id",
            "passed",
            "controller_task_id",
            "controller_status",
            "controller_raw_authority_sha256",
            "subject_order",
            "subject_count",
            "all_training_tasks_completed",
            "authoritative_metadata_read",
            "progress_artifact",
            "progress_seal_sha256",
            "progress_content_sha256",
            "formal_training_manifest_artifact",
            "formal_training_manifest_seal_sha256",
            "formal_training_manifest_content_sha256",
            "recovery_contract_sha256",
            "recursive_progress_chain",
            "bootstrap_equivalence",
            "source_revision_equivalence",
            "source_revision_equivalence_seal_sha256",
            "source_revision_subject_map",
            "source_revision_subject_map_seal_sha256",
            "run_contract_equivalence",
            "capacity_matched_hardware",
            "recovery_parent_controllers",
            "training_tasks",
            "seal_sha256",
        },
        context="formal training provenance document",
    )
    if (
        document.get("schema_version") != 2
        or document.get("document_type")
        != "resilient_v2x_formal_training_provenance_equivalence"
        or document.get("protocol_id") != AUDIT_PROTOCOL_ID
        or document.get("passed") is not True
        or document.get("controller_task_id") != controller_task_id
        or document.get("controller_status") != "completed"
        or document.get("all_training_tasks_completed") is not True
        or document.get("authoritative_metadata_read")
        != "single_batch_per_snapshot"
        or document.get("progress_artifact") != TRAINING_PROGRESS_ARTIFACT
        or document.get("progress_seal_sha256") != progress_seal
        or document.get("progress_content_sha256") != _content_sha256(progress)
        or document.get("recovery_contract_sha256") != _content_sha256(recovery)
    ):
        raise FormalMultiseedPlanError(
            "formal training provenance schema-v4 progress binding drifted"
        )
    _require_seal(document, context="formal training provenance document")
    for field, expected_seal in (
        ("source_revision_equivalence", SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256),
        ("source_revision_subject_map", SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256),
    ):
        value = document.get(field)
        if not isinstance(value, Mapping):
            raise FormalMultiseedPlanError(
                f"formal training provenance {field} is invalid"
            )
        observed = _require_seal(value, context=f"formal training provenance {field}")
        if (
            observed != expected_seal
            or document.get(f"{field}_seal_sha256") != observed
        ):
            raise FormalMultiseedPlanError(
                f"formal training provenance {field} seal drifted"
            )


def _training_chain_snapshots(
    controller: object,
    provenance: object,
    watcher: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str, str]:
    controller_snapshot = _orchestrator_snapshot(
        controller,
        task_id=TRAINING_CONTROLLER_TASK_ID,
        parent_task_id="",
        context="formal training controller",
        entry_point=TRAINING_CONTROLLER_ENTRY_POINT,
        script_sha256=TRAINING_CONTROLLER_PRODUCER_SHA256,
        artifact_names={
            TRAINING_MANIFEST_ARTIFACT,
            TRAINING_PROGRESS_ARTIFACT,
            TRAINING_SUMMARY_ARTIFACT,
        },
    )
    provenance_snapshot = _orchestrator_snapshot(
        provenance,
        task_id=TRAINING_PROVENANCE_TASK_ID,
        parent_task_id=TRAINING_CONTROLLER_TASK_ID,
        context="formal training provenance",
        entry_point=TRAINING_PROVENANCE_ENTRY_POINT,
        script_sha256=TRAINING_PROVENANCE_PRODUCER_SHA256,
        artifact_names={TRAINING_PROVENANCE_ARTIFACT},
        parameter_validator=_validate_training_provenance_parameters,
    )
    watcher_snapshot = _orchestrator_snapshot(
        watcher,
        task_id=WATCHER_TASK_ID,
        parent_task_id=TRAINING_PROVENANCE_TASK_ID,
        context="formal evaluation watcher",
        entry_point=WATCHER_ENTRY_POINT,
        script_sha256=WATCHER_PRODUCER_SHA256,
        artifact_names={EVALUATION_PLAN_ARTIFACT},
        parameter_validator=_validate_watcher_parameters,
    )
    progress, progress_canonical = _frozen_mapping(
        _artifact_mapping(
            controller,
            TRAINING_PROGRESS_ARTIFACT,
            context="formal training controller",
        ),
        context="formal training progress",
    )
    provenance_artifact, provenance_canonical = _frozen_mapping(
        _artifact_mapping(
            provenance,
            TRAINING_PROVENANCE_ARTIFACT,
            context="formal training provenance",
        ),
        context="formal training provenance artifact",
    )
    progress_seal = _require_seal(progress, context="formal training progress")
    recovery = _validate_schema4_recovery_shape(progress.get("recovery"))
    provenance_source, _ = _script_metadata(
        provenance,
        context="formal training provenance",
        entry_point=TRAINING_PROVENANCE_ENTRY_POINT,
        expected_sha256=TRAINING_PROVENANCE_PRODUCER_SHA256,
    )
    validator = _load_training_provenance_validator(provenance_source)
    try:
        validated_progress = validator._validate_progress(
            progress,
            controller_task_id=TRAINING_CONTROLLER_TASK_ID,
        )
    except Exception as error:
        raise FormalMultiseedPlanError(
            "formal training progress failed the pinned schema-v4 validator"
        ) from error
    if (
        not isinstance(validated_progress, tuple)
        or len(validated_progress) != 3
        or validated_progress[2] != progress_seal
        or _canonical_json(validated_progress[1]) != _canonical_json(recovery)
    ):
        raise FormalMultiseedPlanError(
            "formal training progress schema-v4 validator result drifted"
        )
    provenance_seal = _require_seal(
        provenance_artifact, context="formal training provenance artifact"
    )
    _validate_training_provenance_document(
        provenance_artifact,
        controller_task_id=TRAINING_CONTROLLER_TASK_ID,
        progress=progress,
        progress_seal=progress_seal,
        recovery=recovery,
    )
    controller_snapshot["training_progress_canonical_sha256"] = progress_canonical
    controller_snapshot["training_progress_seal_sha256"] = progress_seal
    provenance_snapshot["artifact_canonical_sha256"] = provenance_canonical
    provenance_snapshot["artifact_seal_sha256"] = provenance_seal
    return (
        controller_snapshot,
        provenance_snapshot,
        watcher_snapshot,
        progress_seal,
        provenance_seal,
    )


def _validate_source_d_documents(
    source_d: object,
) -> tuple[
    dict[str, object],
    dict[str, object],
    str,
    str,
    dict[str, str],
]:
    artifact_names = _artifact_names(source_d, context="Source-D evidence")
    if set(artifact_names) != set(SOURCE_D_ARTIFACTS) or any(
        type(name) is not str for name in artifact_names
    ):
        raise FormalMultiseedPlanError("Source-D artifact inventory drifted")
    source_c, source_c_canonical = _frozen_mapping(
        _artifact_mapping(
            source_d, SOURCE_C_SNAPSHOT_ARTIFACT, context="Source-D evidence"
        ),
        context="Source-C snapshot",
    )
    source_script, source_d_canonical = _frozen_mapping(
        _artifact_mapping(
            source_d, SOURCE_D_SCRIPT_ARTIFACT, context="Source-D evidence"
        ),
        context="Source-D script",
    )
    equivalence, equivalence_canonical = _frozen_mapping(
        _artifact_mapping(
            source_d, SOURCE_D_EQUIVALENCE_ARTIFACT, context="Source-D evidence"
        ),
        context="Source-D equivalence",
    )
    receipt, receipt_canonical = _frozen_mapping(
        _artifact_mapping(
            source_d, SOURCE_D_RECEIPT_ARTIFACT, context="Source-D evidence"
        ),
        context="Source-D receipt",
    )
    _exact_keys(
        source_c,
        {
            "schema_version",
            "artifact_type",
            "complete",
            "source_c_task_id",
            "source_c_task_status",
            "script",
            "script_diff",
            "seal_sha256",
        },
        context="Source-C snapshot",
    )
    _exact_keys(
        source_script,
        {
            "schema_version",
            "artifact_type",
            "complete",
            "transformation_id",
            "source_c_sha256",
            "source_d_sha256",
            "size_bytes",
            "line_count",
            "script",
            "seal_sha256",
        },
        context="Source-D script",
    )
    _exact_keys(
        receipt,
        {
            "schema_version",
            "artifact_type",
            "complete",
            "publication_order",
            "provenance",
            "transformation",
            "artifact_hashes",
            "seal_sha256",
        },
        context="Source-D receipt",
    )
    source_c_seal = _require_seal(source_c, context="Source-C snapshot")
    source_d_seal = _require_seal(source_script, context="Source-D script")
    equivalence_hash = _require_content_hash(
        equivalence, context="Source-D equivalence"
    )
    receipt_seal = _require_seal(receipt, context="Source-D receipt")
    _exact_bool(source_c.get("complete"), True, context="Source-C complete")
    _exact_bool(source_script.get("complete"), True, context="Source-D complete")
    _exact_bool(receipt.get("complete"), True, context="Source-D receipt complete")
    _exact_int(source_c.get("schema_version"), 1, context="Source-C schema")
    _exact_int(source_script.get("schema_version"), 1, context="Source-D schema")
    _exact_int(receipt.get("schema_version"), 1, context="Source-D receipt schema")
    source_c_text = source_c.get("script_diff")
    if type(source_c_text) is not str or not source_c_text:
        raise FormalMultiseedPlanError("Source-C snapshot script bytes are missing")
    source_c_hash = _sha256_text(source_c_text)
    expected_source_c_metadata = {
        "repository": "",
        "working_dir": ".",
        "entry_point": SOURCE_C_ENTRY_POINT,
        "sha256": source_c_hash,
        "size_bytes": len(source_c_text.encode("utf-8")),
        "line_count": len(source_c_text.splitlines()),
    }
    if (
        source_c.get("artifact_type") != "resilient_v2x_formal_source_c_snapshot"
        or source_c.get("source_c_task_id") != SOURCE_C_TASK_ID
        or source_c.get("source_c_task_status") != "completed"
        or source_c_hash != SOURCE_C_SCRIPT_SHA256
        or _canonical_json(source_c.get("script"))
        != _canonical_json(expected_source_c_metadata)
    ):
        raise FormalMultiseedPlanError("Source-C snapshot provenance drifted")
    source_d_text = source_script.get("script")
    if type(source_d_text) is not str or not source_d_text:
        raise FormalMultiseedPlanError("Source-D script bytes are missing")
    source_d_hash = _sha256_text(source_d_text)
    if (
        source_script.get("artifact_type") != "resilient_v2x_formal_source_d_script"
        or source_script.get("transformation_id") != SOURCE_D_TRANSFORMATION_ID
        or source_script.get("source_c_sha256") != source_c_hash
        or source_script.get("source_d_sha256") != source_d_hash
        or source_d_hash != SOURCE_D_SCRIPT_SHA256
        or type(source_script.get("size_bytes")) is not int
        or source_script.get("size_bytes") != len(source_d_text.encode("utf-8"))
        or type(source_script.get("line_count")) is not int
        or source_script.get("line_count") != len(source_d_text.splitlines())
    ):
        raise FormalMultiseedPlanError("Source-D script semantics drifted")
    diff = equivalence.get("diff")
    staging_names = (
        tuple(
            item["name"]
            for item in diff
            if isinstance(item, Mapping)
            and item.get("name") in SOURCE_D_STAGING_ANCHOR_NAMES
        )
        if isinstance(diff, list)
        else ()
    )
    if (
        equivalence_hash != SOURCE_D_EQUIVALENCE_SHA256
        or equivalence.get("transformation_id") != SOURCE_D_TRANSFORMATION_ID
        or not isinstance(diff, list)
        or len(diff) != DECLARED_REPLACEMENT_COUNT
        or any(
            not isinstance(item, Mapping)
            or type(item.get("index")) is not int
            or item.get("index") != index
            or type(item.get("name")) is not str
            or type(item.get("expected_count")) is not int
            or item.get("expected_count") != 1
            or type(item.get("observed_count")) is not int
            or item.get("observed_count") != 1
            for index, item in enumerate(diff or [], start=1)
        )
        or staging_names != SOURCE_D_STAGING_ANCHOR_NAMES
        or not isinstance(equivalence.get("equivalence"), Mapping)
        or equivalence["equivalence"].get("source_c_replay_sha256") != source_c_hash
        or equivalence["equivalence"].get("source_d_replay_sha256") != source_d_hash
        or equivalence["equivalence"].get("only_declared_anchor_replacements")
        is not True
        or type(equivalence["equivalence"].get("declared_replacement_count")) is not int
        or equivalence["equivalence"]["declared_replacement_count"]
        != DECLARED_REPLACEMENT_COUNT
        or type(equivalence["equivalence"].get("unchanged_segment_count")) is not int
        or equivalence["equivalence"]["unchanged_segment_count"]
        != UNCHANGED_SEGMENT_COUNT
        or equivalence["equivalence"].get("source_d_compiles") is not True
        or not isinstance(equivalence.get("seed_contract"), Mapping)
        or type(equivalence["seed_contract"].get("training_overlay_protocol_seed"))
        is not int
        or equivalence["seed_contract"]["training_overlay_protocol_seed"]
        != TRAINING_OVERLAY_PROTOCOL_SEED
    ):
        raise FormalMultiseedPlanError("Source-D equivalence semantics drifted")
    expected_hashes = {
        SOURCE_C_SNAPSHOT_ARTIFACT: source_c_seal,
        SOURCE_D_SCRIPT_ARTIFACT: source_d_seal,
        SOURCE_D_EQUIVALENCE_ARTIFACT: equivalence_hash,
    }
    expected_provenance = {
        "source_c_task_id": SOURCE_C_TASK_ID,
        "source_c_task_status": "completed",
        "source_c_task_parent": SOURCE_C_PARENT_TASK_ID,
        "source_c_entry_point": SOURCE_C_ENTRY_POINT,
        "source_c_sha256": source_c_hash,
        "output_task_id": SOURCE_D_TASK_ID,
        "output_parent_task_id": SOURCE_C_TASK_ID,
        "builder_source_sha256": SOURCE_D_BUILDER_SHA256,
        "transformation_id": SOURCE_D_TRANSFORMATION_ID,
        "producer_entry_point": SOURCE_D_ENTRY_POINT,
        "producer_script_sha256": SOURCE_D_PRODUCER_SHA256,
    }
    expected_transformation = {
        "source_d_sha256": source_d_hash,
        "equivalence_artifact_sha256": equivalence_hash,
        "declared_replacement_count": DECLARED_REPLACEMENT_COUNT,
        "unchanged_segment_count": UNCHANGED_SEGMENT_COUNT,
        "only_declared_anchor_replacements": True,
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "training_seed_cli": "--training-seed",
        "portable_runner_load_marker": PORTABLE_RUNNER_LOAD_MARKER,
        "portable_runner_load_marker_count": PORTABLE_RUNNER_LOAD_MARKER_COUNT,
        "legacy_runner_load_target_anchor_count": 0,
    }
    if (
        receipt.get("artifact_type") != "resilient_v2x_formal_source_d_evidence_receipt"
        or receipt.get("publication_order") != list(SOURCE_D_ARTIFACTS)
        or _canonical_json(receipt.get("artifact_hashes"))
        != _canonical_json(expected_hashes)
        or _canonical_json(receipt.get("provenance"))
        != _canonical_json(expected_provenance)
        or _canonical_json(receipt.get("transformation"))
        != _canonical_json(expected_transformation)
    ):
        raise FormalMultiseedPlanError("Source-D receipt provenance drifted")
    exact_pins = (
        (source_c_seal, SOURCE_C_SNAPSHOT_SEAL_SHA256, "Source-C snapshot seal"),
        (
            source_c_canonical,
            SOURCE_C_SNAPSHOT_CANONICAL_SHA256,
            "Source-C snapshot canonical",
        ),
        (source_d_seal, SOURCE_D_ARTIFACT_SEAL_SHA256, "Source-D artifact seal"),
        (
            source_d_canonical,
            SOURCE_D_ARTIFACT_CANONICAL_SHA256,
            "Source-D artifact canonical",
        ),
        (
            equivalence_canonical,
            SOURCE_D_EQUIVALENCE_CANONICAL_SHA256,
            "Source-D equivalence canonical",
        ),
        (receipt_seal, SOURCE_D_RECEIPT_SEAL_SHA256, "Source-D receipt seal"),
        (
            receipt_canonical,
            SOURCE_D_RECEIPT_CANONICAL_SHA256,
            "Source-D receipt canonical",
        ),
    )
    for observed, expected, context in exact_pins:
        if observed != expected:
            raise FormalMultiseedPlanError(f"{context} pin mismatch")
    return (
        source_script,
        equivalence,
        source_d_seal,
        equivalence_hash,
        {
            SOURCE_C_SNAPSHOT_ARTIFACT: source_c_canonical,
            SOURCE_D_SCRIPT_ARTIFACT: source_d_canonical,
            SOURCE_D_EQUIVALENCE_ARTIFACT: equivalence_canonical,
            SOURCE_D_RECEIPT_ARTIFACT: receipt_canonical,
        },
    )


def _source_d_snapshot(source_d: object) -> dict[str, object]:
    task_id = _clearml_id(getattr(source_d, "id", None), context="Source-D task")
    if task_id != SOURCE_D_TASK_ID:
        raise FormalMultiseedPlanError("Source-D identity mismatch")
    if _status(source_d, context="Source-D task") != "completed":
        raise FormalMultiseedPlanError("Source-D is not completed")
    if _task_parent(source_d, context="Source-D task") != SOURCE_C_TASK_ID:
        raise FormalMultiseedPlanError("Source-D parent mismatch")
    _, script = _script_metadata(
        source_d,
        context="Source-D task",
        entry_point=SOURCE_D_ENTRY_POINT,
        expected_sha256=SOURCE_D_PRODUCER_SHA256,
    )
    source_artifact, equivalence, source_seal, equivalence_hash, artifact_hashes = (
        _validate_source_d_documents(source_d)
    )
    return {
        "task_id": task_id,
        "parent_task_id": SOURCE_C_TASK_ID,
        "status": "completed",
        "script": script,
        "artifact_canonical_sha256": artifact_hashes,
        "source_d_script_sha256": source_artifact["source_d_sha256"],
        "source_d_script_seal_sha256": source_seal,
        "equivalence_sha256": equivalence_hash,
        "equivalence_transformation_id": equivalence["transformation_id"],
    }


def _audit_subject_kind(subject: str) -> str:
    if subject in AUDIT_BASELINE_SUBJECTS:
        return "baseline"
    if subject in {
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "concat_capacity_matched",
    }:
        return "ablation"
    if subject in {
        "support_residual",
        "linear_no_distillation",
        "no_distillation_peak_lr_3e4",
    }:
        return "improvement"
    if subject == "resilient_v2x":
        return "primary_method"
    raise FormalMultiseedPlanError(f"unknown formal audit subject {subject!r}")


def _expected_source_revision(subject: str) -> dict[str, object]:
    if subject not in AUDIT_SUBJECT_ORDER:
        raise FormalMultiseedPlanError(f"unknown formal audit subject {subject!r}")
    tree_sha256 = (
        AUDIT_NEW_SOURCE_TREE_SHA256
        if subject in AUDIT_NEW_SOURCE_SUBJECTS
        else AUDIT_SOURCE_TREE_SHA256
    )
    certificate = SOURCE_REVISION_CERTIFICATE_BY_TREE[tree_sha256]
    archive = certificate["archive"]
    if not isinstance(archive, Mapping):  # pragma: no cover - static constant
        raise AssertionError("source revision archive certificate is invalid")
    return {
        "tree_sha256": tree_sha256,
        "dataset_id": certificate["dataset_id"],
        "archive_name": archive["name"],
        "archive_bytes": archive["size_bytes"],
        "archive_sha256": archive["sha256"],
    }


def _validate_source_provenance_binding(
    value: Mapping[str, object], *, context: str
) -> dict[str, object]:
    _exact_keys(
        value,
        {
            "training_provenance_task_id",
            "training_provenance_seal_sha256",
            "source_revision_equivalence",
            "source_revision_equivalence_seal_sha256",
            "source_revision_subject_map",
            "source_revision_subject_map_seal_sha256",
            "evaluation_source_revision_tree_sha256",
            "evaluation_source_revision",
        },
        context=context,
    )
    provenance_task_id = _clearml_id(
        value.get("training_provenance_task_id"),
        context=f"{context} training provenance task",
    )
    if provenance_task_id != TRAINING_PROVENANCE_TASK_ID:
        raise FormalMultiseedPlanError(
            f"{context} training provenance task mismatch"
        )
    provenance_seal = _sha256(
        value.get("training_provenance_seal_sha256"),
        context=f"{context} training provenance seal",
    )
    equivalence = value.get("source_revision_equivalence")
    if not isinstance(equivalence, Mapping):
        raise FormalMultiseedPlanError(
            f"{context} source revision equivalence is invalid"
        )
    equivalence_frozen, _ = _frozen_mapping(
        equivalence, context=f"{context} source revision equivalence"
    )
    equivalence_seal = _require_seal(
        equivalence_frozen, context=f"{context} source revision equivalence"
    )
    if (
        equivalence_seal != SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        or value.get("source_revision_equivalence_seal_sha256")
        != equivalence_seal
        or equivalence_frozen.get("source_revisions")
        != SOURCE_REVISION_CERTIFICATE_BY_TREE
    ):
        raise FormalMultiseedPlanError(
            f"{context} source revision equivalence mismatch"
        )
    subject_map = value.get("source_revision_subject_map")
    if not isinstance(subject_map, Mapping):
        raise FormalMultiseedPlanError(
            f"{context} source revision subject map is invalid"
        )
    subject_map_frozen, _ = _frozen_mapping(
        subject_map, context=f"{context} source revision subject map"
    )
    subject_map_seal = _require_seal(
        subject_map_frozen, context=f"{context} source revision subject map"
    )
    expected_subject_map = {
        subject: _expected_source_revision(subject)["tree_sha256"]
        for subject in AUDIT_SUBJECT_ORDER
    }
    if (
        subject_map_seal != SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        or value.get("source_revision_subject_map_seal_sha256")
        != subject_map_seal
        or subject_map_frozen.get("subject_order") != list(AUDIT_SUBJECT_ORDER)
        or subject_map_frozen.get("source_revision_by_subject")
        != expected_subject_map
        or subject_map_frozen.get("revision_subject_counts")
        != {AUDIT_SOURCE_TREE_SHA256: 21, AUDIT_NEW_SOURCE_TREE_SHA256: 5}
    ):
        raise FormalMultiseedPlanError(
            f"{context} source revision subject map mismatch"
        )
    evaluation_tree = _sha256(
        value.get("evaluation_source_revision_tree_sha256"),
        context=f"{context} evaluation source revision tree",
    )
    if (
        evaluation_tree != AUDIT_SOURCE_TREE_SHA256
        or value.get("evaluation_source_revision")
        != SOURCE_REVISION_CERTIFICATE_BY_TREE[AUDIT_SOURCE_TREE_SHA256]
    ):
        raise FormalMultiseedPlanError(
            f"{context} evaluation source revision mismatch"
        )
    return {
        "training_provenance_task_id": provenance_task_id,
        "training_provenance_seal_sha256": provenance_seal,
        "source_revision_equivalence": equivalence_frozen,
        "source_revision_equivalence_seal_sha256": equivalence_seal,
        "source_revision_subject_map": subject_map_frozen,
        "source_revision_subject_map_seal_sha256": subject_map_seal,
        "evaluation_source_revision_tree_sha256": evaluation_tree,
        "evaluation_source_revision": dict(
            SOURCE_REVISION_CERTIFICATE_BY_TREE[AUDIT_SOURCE_TREE_SHA256]
        ),
    }


def _validate_leaderboard_document(artifact: Mapping[str, object]) -> str:
    seed_fields = {"training_seed", "training_overlay_protocol_seed"}
    present_seed_fields = set(artifact) & seed_fields
    if present_seed_fields not in (set(), seed_fields):
        raise FormalMultiseedPlanError("formal leaderboard seed metadata is partial")
    seeded = present_seed_fields == seed_fields
    _exact_keys(
        artifact,
        {
            "schema_version",
            "leaderboard_type",
            "protocol_id",
            "sample_count",
            "ground_truth_count",
            "unsupported_sample_count",
            "delays_ms",
            "conditions",
            "run_count_per_subject",
            "training_controller_task_id",
            "watcher_task_id",
            "training_manifest_seal_sha256",
            "evaluation_plan_seal_sha256",
            "training_provenance_task_id",
            "training_provenance_seal_sha256",
            "source_revision_equivalence",
            "source_revision_equivalence_seal_sha256",
            "source_revision_subject_map",
            "source_revision_subject_map_seal_sha256",
            "evaluation_source_revision_tree_sha256",
            "evaluation_source_revision",
            "subject_order",
            "subject_count",
            "baseline_subjects",
            "baseline_count",
            "metric_keys",
            "results",
            "leadership",
            "seal_sha256",
        }
        | (seed_fields if seeded else set()),
        context="formal leaderboard artifact",
    )
    seal = _require_seal(artifact, context="formal leaderboard artifact")
    fixed_values = {
        "leaderboard_type": "resilient_v2x_formal_1337_leaderboard",
        "protocol_id": AUDIT_PROTOCOL_ID,
        "conditions": list(AUDIT_CONDITIONS),
        "training_controller_task_id": SELECTOR_TRAINING_CONTROLLER_TASK_ID,
        "watcher_task_id": SELECTOR_WATCHER_TASK_ID,
        "subject_order": list(AUDIT_SUBJECT_ORDER),
        "baseline_subjects": list(AUDIT_BASELINE_SUBJECTS),
        "metric_keys": list(AUDIT_METRIC_KEYS),
    }
    if any(artifact.get(key) != expected for key, expected in fixed_values.items()):
        raise FormalMultiseedPlanError("formal leaderboard immutable protocol drifted")
    delays = artifact.get("delays_ms")
    if type(delays) is not list or len(delays) != len(AUDIT_DELAYS_MS):
        raise FormalMultiseedPlanError("formal leaderboard delays drifted")
    for index, (observed, expected) in enumerate(
        zip(delays, AUDIT_DELAYS_MS, strict=True), start=1
    ):
        _exact_int(
            observed,
            expected,
            context=f"formal leaderboard delay {index}",
        )
    for key, expected in (
        ("schema_version", 3),
        ("sample_count", AUDIT_SAMPLE_COUNT),
        ("ground_truth_count", AUDIT_GROUND_TRUTH_COUNT),
        ("unsupported_sample_count", AUDIT_UNSUPPORTED_SAMPLE_COUNT),
        ("run_count_per_subject", len(AUDIT_DELAYS_MS) * len(AUDIT_CONDITIONS)),
        ("subject_count", len(AUDIT_SUBJECT_ORDER)),
        ("baseline_count", len(AUDIT_BASELINE_SUBJECTS)),
    ):
        _exact_int(artifact.get(key), expected, context=f"formal leaderboard {key}")
    if seeded:
        _exact_int(
            artifact.get("training_seed"),
            AUDIT_TRAINING_SEED,
            context="formal leaderboard training seed",
        )
        _exact_int(
            artifact.get("training_overlay_protocol_seed"),
            AUDIT_TRAINING_SEED,
            context="formal leaderboard overlay seed",
        )
    for key in (
        "training_manifest_seal_sha256",
        "evaluation_plan_seal_sha256",
    ):
        _sha256(artifact.get(key), context=f"formal leaderboard {key}")
    _validate_source_provenance_binding(
        {
            key: artifact[key]
            for key in (
                "training_provenance_task_id",
                "training_provenance_seal_sha256",
                "source_revision_equivalence",
                "source_revision_equivalence_seal_sha256",
                "source_revision_subject_map",
                "source_revision_subject_map_seal_sha256",
                "evaluation_source_revision_tree_sha256",
                "evaluation_source_revision",
            )
        },
        context="formal leaderboard",
    )
    results = artifact.get("results")
    if type(results) is not list or len(results) != len(AUDIT_SUBJECT_ORDER):
        raise FormalMultiseedPlanError("formal leaderboard result count mismatch")
    for index, (record, subject) in enumerate(
        zip(results, AUDIT_SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(record, Mapping):
            raise FormalMultiseedPlanError(
                f"formal leaderboard result {index} is invalid"
            )
        frozen, _ = _frozen_mapping(
            record, context=f"formal leaderboard result {subject}"
        )
        _exact_keys(
            frozen,
            {
                "index",
                "subject",
                "kind",
                "training_task_id",
                "training_model_id",
                "training_checkpoint_sha256",
                "source_revision_tree_sha256",
                "evaluation_task_id",
                "metrics",
            },
            context=f"formal leaderboard result {subject}",
        )
        _exact_int(
            frozen.get("index"),
            index,
            context=f"formal leaderboard result {subject} index",
        )
        if (
            frozen.get("subject") != subject
            or frozen.get("kind") != _audit_subject_kind(subject)
            or not isinstance(frozen.get("metrics"), Mapping)
            or set(frozen["metrics"]) != set(AUDIT_METRIC_KEYS)
            or frozen.get("source_revision_tree_sha256")
            != _expected_source_revision(subject)["tree_sha256"]
        ):
            raise FormalMultiseedPlanError(
                f"formal leaderboard result {subject} contract drifted"
            )
        _clearml_id(
            frozen.get("training_task_id"),
            context=f"formal leaderboard training task {subject}",
        )
        _clearml_id(
            frozen.get("training_model_id"),
            context=f"formal leaderboard model {subject}",
        )
        _sha256(
            frozen.get("training_checkpoint_sha256"),
            context=f"formal leaderboard checkpoint {subject}",
        )
        _clearml_id(
            frozen.get("evaluation_task_id"),
            context=f"formal leaderboard evaluation task {subject}",
        )
    if not isinstance(artifact.get("leadership"), Mapping):
        raise FormalMultiseedPlanError("formal leaderboard leadership is invalid")
    return seal


def _leaderboard_snapshot(
    leaderboard: object,
) -> tuple[dict[str, object], dict[str, object]]:
    task_id = _clearml_id(
        getattr(leaderboard, "id", None), context="formal leaderboard"
    )
    if task_id != SELECTOR_LEADERBOARD_TASK_ID:
        raise FormalMultiseedPlanError("formal leaderboard identity mismatch")
    if _status(leaderboard, context="formal leaderboard") != "completed":
        raise FormalMultiseedPlanError("formal leaderboard is not completed")
    if (
        _task_parent(leaderboard, context="formal leaderboard")
        != SELECTOR_WATCHER_TASK_ID
    ):
        raise FormalMultiseedPlanError("formal leaderboard parent mismatch")
    _, script = _script_metadata(
        leaderboard,
        context="formal leaderboard",
        entry_point=LEADERBOARD_ENTRY_POINT,
        expected_sha256=LEADERBOARD_PRODUCER_SHA256,
    )
    parameters = _validate_leaderboard_parameters(leaderboard)
    if _artifact_names(leaderboard, context="formal leaderboard") != (
        LEADERBOARD_ARTIFACT,
    ):
        raise FormalMultiseedPlanError("formal leaderboard artifact inventory drifted")
    artifact, canonical_sha = _frozen_mapping(
        _artifact_mapping(
            leaderboard, LEADERBOARD_ARTIFACT, context="formal leaderboard"
        ),
        context="formal leaderboard artifact",
    )
    seal = _validate_leaderboard_document(artifact)
    if seal != LEADERBOARD_ARTIFACT_SEAL_SHA256:
        raise FormalMultiseedPlanError("formal leaderboard artifact seal pin mismatch")
    if canonical_sha != LEADERBOARD_ARTIFACT_CANONICAL_SHA256:
        raise FormalMultiseedPlanError(
            "formal leaderboard artifact canonical pin mismatch"
        )
    return artifact, {
        "task_id": task_id,
        "parent_task_id": SELECTOR_WATCHER_TASK_ID,
        "status": "completed",
        "script": script,
        "parameters": parameters,
        "artifact_canonical_sha256": canonical_sha,
        "artifact_seal_sha256": seal,
    }


def _cross_validate_leaderboard_audit(
    leaderboard: Mapping[str, object],
    audit: Mapping[str, object],
) -> None:
    leaderboard_seal = _sha256(
        leaderboard.get("seal_sha256"), context="formal leaderboard seal"
    )
    if (
        audit.get("leaderboard_seal_sha256") != leaderboard_seal
        or audit.get("training_manifest_seal_sha256")
        != leaderboard.get("training_manifest_seal_sha256")
        or audit.get("evaluation_plan_seal_sha256")
        != leaderboard.get("evaluation_plan_seal_sha256")
    ):
        raise FormalMultiseedPlanError(
            "formal leaderboard and audit seals are not cross-bound"
        )
    for key in (
        "training_provenance_task_id",
        "training_provenance_seal_sha256",
        "source_revision_equivalence",
        "source_revision_equivalence_seal_sha256",
        "source_revision_subject_map",
        "source_revision_subject_map_seal_sha256",
        "evaluation_source_revision_tree_sha256",
        "evaluation_source_revision",
    ):
        if audit.get(key) != leaderboard.get(key):
            raise FormalMultiseedPlanError(
                f"formal leaderboard and audit {key} are not cross-bound"
            )
    results = leaderboard.get("results")
    training = audit.get("training_tasks")
    evaluation = audit.get("evaluation_tasks")
    if (
        type(results) is not list
        or type(training) is not list
        or type(evaluation) is not list
    ):
        raise FormalMultiseedPlanError(
            "formal leaderboard and audit records are unavailable"
        )
    for result, training_record, evaluation_record, subject in zip(
        results,
        training,
        evaluation,
        AUDIT_SUBJECT_ORDER,
        strict=True,
    ):
        if (
            not isinstance(result, Mapping)
            or not isinstance(training_record, Mapping)
            or not isinstance(evaluation_record, Mapping)
            or result.get("subject") != subject
            or result.get("training_task_id") != training_record.get("training_task_id")
            or result.get("training_model_id") != training_record.get("model_id")
            or result.get("training_checkpoint_sha256")
            != training_record.get("checkpoint_sha256")
            or result.get("source_revision_tree_sha256")
            != training_record.get("source_revision_tree_sha256")
            or result.get("evaluation_task_id")
            != evaluation_record.get("evaluation_task_id")
            or evaluation_record.get("source_revision_tree_sha256")
            != AUDIT_SOURCE_TREE_SHA256
        ):
            raise FormalMultiseedPlanError(
                f"formal leaderboard and audit record {subject} drifted"
            )


def _validate_audit_training_records(
    value: object,
    *,
    legacy_unseeded_manifest: bool,
) -> dict[str, dict[str, object]]:
    if type(value) is not list or len(value) != len(AUDIT_SUBJECT_ORDER):
        raise FormalMultiseedPlanError("formal audit training record count mismatch")
    expected_keys = {
        "index",
        "subject",
        "training_task_id",
        "model_id",
        "checkpoint_sha256",
        "final_checkpoint_contract_content_sha256",
        "parent_controller_task_id",
        "script_sha256",
        "run_contract_sha256",
        "run_contract_seed_fields",
        "training_seed",
        "source_revision_tree_sha256",
        "source_dataset_id",
        "source_archive_name",
        "source_archive_bytes",
        "source_archive_sha256",
        "training_dataset_id",
        "gpus",
        "precision",
        "max_epochs",
        "val_interval",
        "common_teacher_initialization_audit_sha256",
        "raw_authority_sha256",
        "execution_queue_id",
        "last_worker",
        "final_model_verification_level",
        "checkpoint_bytes_sha256_recomputed",
        "checkpoint_bytes_verifier",
    }
    expected_seed_fields = (
        ["seed"]
        if legacy_unseeded_manifest
        else ["seed", "training_overlay_protocol_seed", "training_seed"]
    )
    result: dict[str, dict[str, object]] = {}
    task_ids: set[str] = set()
    model_ids: set[str] = set()
    for index, (record, subject) in enumerate(
        zip(value, AUDIT_SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(record, Mapping):
            raise FormalMultiseedPlanError(
                f"formal audit training record {index} is invalid"
            )
        frozen, _ = _frozen_mapping(
            record, context=f"formal audit training record {subject}"
        )
        _exact_keys(
            frozen,
            expected_keys,
            context=f"formal audit training record {subject}",
        )
        _exact_int(
            frozen.get("index"),
            index,
            context=f"formal audit training record {subject} index",
        )
        if frozen.get("subject") != subject:
            raise FormalMultiseedPlanError(
                f"formal audit training record {subject} identity mismatch"
            )
        task_id = _clearml_id(
            frozen.get("training_task_id"),
            context=f"formal audit training task {subject}",
        )
        model_id = _clearml_id(
            frozen.get("model_id"), context=f"formal audit model {subject}"
        )
        if task_id in task_ids or model_id in model_ids:
            raise FormalMultiseedPlanError(
                "formal audit training task/model IDs are not unique"
            )
        task_ids.add(task_id)
        model_ids.add(model_id)
        for key in (
            "checkpoint_sha256",
            "final_checkpoint_contract_content_sha256",
            "run_contract_sha256",
            "common_teacher_initialization_audit_sha256",
            "raw_authority_sha256",
        ):
            _sha256(
                frozen.get(key),
                context=f"formal audit training record {subject} {key}",
            )
        source = _expected_source_revision(subject)
        fixed_values = {
            "parent_controller_task_id": _expected_training_parent_task_id(subject),
            "script_sha256": _expected_training_script_sha256(subject),
            "run_contract_seed_fields": expected_seed_fields,
            "source_revision_tree_sha256": source["tree_sha256"],
            "source_dataset_id": source["dataset_id"],
            "source_archive_name": source["archive_name"],
            "source_archive_sha256": source["archive_sha256"],
            "training_dataset_id": AUDIT_TRAINING_DATASET_ID,
            "precision": "FP32",
            "final_model_verification_level": "clearml_metadata_contract_only",
            "checkpoint_bytes_verifier": AUDIT_MODEL_BYTES_VERIFIER,
        }
        if any(frozen.get(key) != expected for key, expected in fixed_values.items()):
            raise FormalMultiseedPlanError(
                f"formal audit training record {subject} contract drifted"
            )
        for key, expected in (
            ("training_seed", AUDIT_TRAINING_SEED),
            ("source_archive_bytes", source["archive_bytes"]),
            ("gpus", 4),
            ("max_epochs", 50),
            ("val_interval", 10),
        ):
            _exact_int(
                frozen.get(key),
                expected,
                context=f"formal audit training record {subject} {key}",
            )
        _exact_bool(
            frozen.get("checkpoint_bytes_sha256_recomputed"),
            False,
            context=(f"formal audit training record {subject} checkpoint byte flag"),
        )
        _clearml_id(
            frozen.get("execution_queue_id"),
            context=f"formal audit training record {subject} execution queue",
        )
        if type(frozen.get("last_worker")) is not str or not frozen.get("last_worker"):
            raise FormalMultiseedPlanError(
                f"formal audit training record {subject} last worker is invalid"
            )
        result[subject] = frozen
    reserved_ids = {
        TRAINING_PROVENANCE_TASK_ID,
        SELECTOR_TASK_ID,
        SELECTOR_AUDIT_TASK_ID,
        SELECTOR_LEADERBOARD_TASK_ID,
        SELECTOR_TRAINING_CONTROLLER_TASK_ID,
        SELECTOR_WATCHER_TASK_ID,
    }
    if task_ids & reserved_ids:
        raise FormalMultiseedPlanError(
            "formal audit training tasks alias controller dependencies"
        )
    return result


def _validate_audit_evaluation_records(
    value: object,
    *,
    training_records: Mapping[str, Mapping[str, object]],
) -> None:
    if type(value) is not list or len(value) != len(AUDIT_SUBJECT_ORDER):
        raise FormalMultiseedPlanError("formal audit evaluation record count mismatch")
    expected_keys = {
        "index",
        "subject",
        "evaluation_task_id",
        "training_task_id",
        "model_id",
        "checkpoint_sha256",
        "source_revision_tree_sha256",
        "source_dataset_id",
        "source_archive_name",
        "source_archive_bytes",
        "source_archive_sha256",
        "script_sha256",
        "planned_queue",
        "execution_queue_id",
        "execution_queue_name",
        "last_worker",
        "metrics_sha256",
        "run_count",
        "sample_count_per_run",
        "ground_truth_count_per_run",
        "unsupported_sample_count_per_run",
        "metric_keys",
    }
    evaluation_ids: set[str] = set()
    training_ids = {
        str(record["training_task_id"]) for record in training_records.values()
    }
    for index, (record, subject) in enumerate(
        zip(value, AUDIT_SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(record, Mapping):
            raise FormalMultiseedPlanError(
                f"formal audit evaluation record {index} is invalid"
            )
        frozen, _ = _frozen_mapping(
            record, context=f"formal audit evaluation record {subject}"
        )
        _exact_keys(
            frozen,
            expected_keys,
            context=f"formal audit evaluation record {subject}",
        )
        _exact_int(
            frozen.get("index"),
            index,
            context=f"formal audit evaluation record {subject} index",
        )
        if frozen.get("subject") != subject:
            raise FormalMultiseedPlanError(
                f"formal audit evaluation record {subject} identity mismatch"
            )
        task_id = _clearml_id(
            frozen.get("evaluation_task_id"),
            context=f"formal audit evaluation task {subject}",
        )
        if task_id in evaluation_ids:
            raise FormalMultiseedPlanError(
                "formal audit evaluation task IDs are not unique"
            )
        evaluation_ids.add(task_id)
        training = training_records[subject]
        for key in ("training_task_id", "model_id", "checkpoint_sha256"):
            if frozen.get(key) != training[key]:
                raise FormalMultiseedPlanError(
                    f"formal audit evaluation record {subject} {key} drifted"
                )
        _clearml_id(
            frozen.get("execution_queue_id"),
            context=f"formal audit evaluation queue {subject}",
        )
        _sha256(
            frozen.get("metrics_sha256"),
            context=f"formal audit evaluation metrics {subject}",
        )
        planned_queue = frozen.get("planned_queue")
        if (
            type(planned_queue) is not str
            or planned_queue not in AUDIT_SUPPORTED_QUEUES
            or frozen.get("execution_queue_name") != planned_queue
            or type(frozen.get("last_worker")) is not str
            or frozen.get("script_sha256") != AUDIT_EVALUATION_SCRIPT_SHA256
            or frozen.get("metric_keys") != list(AUDIT_METRIC_KEYS)
            or frozen.get("source_revision_tree_sha256")
            != AUDIT_SOURCE_TREE_SHA256
            or frozen.get("source_dataset_id") != AUDIT_SOURCE_DATASET_ID
            or frozen.get("source_archive_name") != AUDIT_SOURCE_ARCHIVE_NAME
            or frozen.get("source_archive_sha256") != AUDIT_SOURCE_ARCHIVE_SHA256
        ):
            raise FormalMultiseedPlanError(
                f"formal audit evaluation record {subject} contract drifted"
            )
        for key, expected in (
            ("run_count", len(AUDIT_DELAYS_MS) * len(AUDIT_CONDITIONS)),
            ("source_archive_bytes", AUDIT_SOURCE_ARCHIVE_BYTES),
            ("sample_count_per_run", AUDIT_SAMPLE_COUNT),
            ("ground_truth_count_per_run", AUDIT_GROUND_TRUTH_COUNT),
            (
                "unsupported_sample_count_per_run",
                AUDIT_UNSUPPORTED_SAMPLE_COUNT,
            ),
        ):
            _exact_int(
                frozen.get(key),
                expected,
                context=f"formal audit evaluation record {subject} {key}",
            )
    reserved_ids = {
        TRAINING_PROVENANCE_TASK_ID,
        SELECTOR_TASK_ID,
        SELECTOR_AUDIT_TASK_ID,
        SELECTOR_LEADERBOARD_TASK_ID,
        SELECTOR_TRAINING_CONTROLLER_TASK_ID,
        SELECTOR_WATCHER_TASK_ID,
    }
    if evaluation_ids & (training_ids | reserved_ids):
        raise FormalMultiseedPlanError(
            "formal audit evaluation tasks alias another formal task"
        )


def _validate_audit_document(artifact: Mapping[str, object]) -> None:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "document_type",
            "passed",
            "audit_scope",
            "audit_task_id",
            "training_controller_task_id",
            "training_provenance_task_id",
            "watcher_task_id",
            "leaderboard_task_id",
            "protocol_id",
            "evaluation_source_revision_tree_sha256",
            "evaluation_source_revision",
            "source_revision_equivalence",
            "source_revision_equivalence_seal_sha256",
            "source_revision_subject_map",
            "source_revision_subject_map_seal_sha256",
            "source_revision_counts",
            "training_dataset_id",
            "training_seed",
            "training_seed_evidence",
            "legacy_unseeded_training_manifest",
            "legacy_source_c_seed_schema",
            "training_script_equivalence",
            "evaluation_script_sha256",
            "checkpoint_policy",
            "sample_count",
            "ground_truth_count",
            "unsupported_sample_count",
            "delays_ms",
            "conditions",
            "run_count_per_subject",
            "subject_order",
            "subject_count",
            "total_evaluation_run_count",
            "metric_keys",
            "final_model_verification_level",
            "checkpoint_bytes_sha256_recomputed",
            "checkpoint_bytes_verifier",
            "checkpoint_bytes_verification_dependency",
            "training_manifest_seal_sha256",
            "training_progress_seal_sha256",
            "training_provenance_seal_sha256",
            "training_summary_seal_sha256",
            "evaluation_plan_seal_sha256",
            "leaderboard_seal_sha256",
            "training_tasks",
            "evaluation_tasks",
            "seal_sha256",
        },
        context="formal comparability audit artifact",
    )
    _exact_int(
        artifact.get("schema_version"), 3, context="formal comparability audit schema"
    )
    _exact_bool(
        artifact.get("passed"), True, context="formal comparability audit passed"
    )
    legacy = artifact.get("legacy_unseeded_training_manifest")
    if type(legacy) is not bool:
        raise FormalMultiseedPlanError(
            "formal comparability audit legacy manifest flag is invalid"
        )
    _exact_bool(
        artifact.get("checkpoint_bytes_sha256_recomputed"),
        False,
        context="formal comparability audit checkpoint byte flag",
    )
    fixed_values = {
        "document_type": "resilient_v2x_formal_1337_comparability_audit",
        "audit_scope": "protocol_and_clearml_metadata_comparability",
        "audit_task_id": SELECTOR_AUDIT_TASK_ID,
        "training_controller_task_id": SELECTOR_TRAINING_CONTROLLER_TASK_ID,
        "training_provenance_task_id": SELECTOR_TRAINING_PROVENANCE_TASK_ID,
        "watcher_task_id": SELECTOR_WATCHER_TASK_ID,
        "leaderboard_task_id": SELECTOR_LEADERBOARD_TASK_ID,
        "protocol_id": AUDIT_PROTOCOL_ID,
        "evaluation_source_revision_tree_sha256": AUDIT_SOURCE_TREE_SHA256,
        "evaluation_source_revision": SOURCE_REVISION_CERTIFICATE_BY_TREE[
            AUDIT_SOURCE_TREE_SHA256
        ],
        "source_revision_counts": {
            AUDIT_SOURCE_TREE_SHA256: 21,
            AUDIT_NEW_SOURCE_TREE_SHA256: 5,
        },
        "training_dataset_id": AUDIT_TRAINING_DATASET_ID,
        "training_seed_evidence": "all_26_live_run_contracts",
        "legacy_source_c_seed_schema": (
            "explicit_run_contract_seed_only" if legacy else None
        ),
        "training_script_equivalence": {
            "legacy_script_sha256": LEGACY_TRAINING_SCRIPT_SHA256,
            "canonical_script_sha256": CANONICAL_TRAINING_SCRIPT_SHA256,
            "legacy_script_subjects": list(LEGACY_SCRIPT_SUBJECTS),
            "only_difference": "NESTED_TEACHER_EXPERIMENTS membership",
            "runtime_usage_closure": [
                "expect_nested_teacher_keyword",
                "expected_nested_teacher_contract_field",
            ],
        },
        "evaluation_script_sha256": EVALUATION_SCRIPT_SHA256,
        "checkpoint_policy": AUDIT_CHECKPOINT_POLICY,
        "conditions": list(AUDIT_CONDITIONS),
        "subject_order": list(AUDIT_SUBJECT_ORDER),
        "metric_keys": list(AUDIT_METRIC_KEYS),
        "final_model_verification_level": "clearml_metadata_contract_only",
        "checkpoint_bytes_verifier": AUDIT_MODEL_BYTES_VERIFIER,
        "checkpoint_bytes_verification_dependency": (
            "separate_collect_clearml_formal_models_byte_audit"
        ),
    }
    if any(artifact.get(key) != expected for key, expected in fixed_values.items()):
        raise FormalMultiseedPlanError(
            "formal comparability audit immutable protocol drifted"
        )
    _validate_source_provenance_binding(
        {
            "training_provenance_task_id": artifact[
                "training_provenance_task_id"
            ],
            "training_provenance_seal_sha256": artifact[
                "training_provenance_seal_sha256"
            ],
            "source_revision_equivalence": artifact[
                "source_revision_equivalence"
            ],
            "source_revision_equivalence_seal_sha256": artifact[
                "source_revision_equivalence_seal_sha256"
            ],
            "source_revision_subject_map": artifact["source_revision_subject_map"],
            "source_revision_subject_map_seal_sha256": artifact[
                "source_revision_subject_map_seal_sha256"
            ],
            "evaluation_source_revision_tree_sha256": artifact[
                "evaluation_source_revision_tree_sha256"
            ],
            "evaluation_source_revision": artifact["evaluation_source_revision"],
        },
        context="formal comparability audit",
    )
    delays = artifact.get("delays_ms")
    if type(delays) is not list or len(delays) != len(AUDIT_DELAYS_MS):
        raise FormalMultiseedPlanError("formal comparability audit delays drifted")
    for index, (observed, expected) in enumerate(
        zip(delays, AUDIT_DELAYS_MS, strict=True), start=1
    ):
        _exact_int(
            observed,
            expected,
            context=f"formal comparability audit delay {index}",
        )
    for key, expected in (
        ("training_seed", AUDIT_TRAINING_SEED),
        ("sample_count", AUDIT_SAMPLE_COUNT),
        ("ground_truth_count", AUDIT_GROUND_TRUTH_COUNT),
        ("unsupported_sample_count", AUDIT_UNSUPPORTED_SAMPLE_COUNT),
        ("run_count_per_subject", len(AUDIT_DELAYS_MS) * len(AUDIT_CONDITIONS)),
        ("subject_count", len(AUDIT_SUBJECT_ORDER)),
        (
            "total_evaluation_run_count",
            len(AUDIT_SUBJECT_ORDER) * len(AUDIT_DELAYS_MS) * len(AUDIT_CONDITIONS),
        ),
    ):
        _exact_int(
            artifact.get(key), expected, context=f"formal comparability audit {key}"
        )
    for key in (
        "training_manifest_seal_sha256",
        "training_progress_seal_sha256",
        "training_provenance_seal_sha256",
        "training_summary_seal_sha256",
        "evaluation_plan_seal_sha256",
        "leaderboard_seal_sha256",
    ):
        _sha256(artifact.get(key), context=f"formal comparability audit artifact {key}")
    training = _validate_audit_training_records(
        artifact.get("training_tasks"),
        legacy_unseeded_manifest=legacy,
    )
    _validate_audit_evaluation_records(
        artifact.get("evaluation_tasks"), training_records=training
    )


def _audit_snapshot(audit: object) -> tuple[dict[str, object], dict[str, object]]:
    task_id = _clearml_id(
        getattr(audit, "id", None), context="formal comparability audit"
    )
    if task_id != SELECTOR_AUDIT_TASK_ID:
        raise FormalMultiseedPlanError("formal comparability audit identity mismatch")
    if _status(audit, context="formal comparability audit") != "completed":
        raise FormalMultiseedPlanError("formal comparability audit is not completed")
    if (
        _task_parent(audit, context="formal comparability audit")
        != SELECTOR_LEADERBOARD_TASK_ID
    ):
        raise FormalMultiseedPlanError("formal comparability audit parent mismatch")
    _, script = _script_metadata(
        audit,
        context="formal comparability audit",
        entry_point=AUDIT_ENTRY_POINT,
        expected_sha256=AUDIT_PRODUCER_SHA256,
    )
    parameters = _validate_audit_parameters(audit)
    if _artifact_names(audit, context="formal comparability audit") != (
        AUDIT_ARTIFACT,
    ):
        raise FormalMultiseedPlanError(
            "formal comparability audit artifact inventory drifted"
        )
    artifact, canonical_sha = _frozen_mapping(
        _artifact_mapping(audit, AUDIT_ARTIFACT, context="formal comparability audit"),
        context="formal comparability audit artifact",
    )
    seal = _require_seal(artifact, context="formal comparability audit artifact")
    _validate_audit_document(artifact)
    return artifact, {
        "task_id": task_id,
        "parent_task_id": SELECTOR_LEADERBOARD_TASK_ID,
        "status": "completed",
        "script": script,
        "parameters": parameters,
        "artifact_canonical_sha256": canonical_sha,
        "artifact_seal_sha256": seal,
    }


def _selector_snapshot(
    selector: object,
    *,
    expected_audit_seal: str,
    expected_leaderboard_seal: str,
    expected_progress_seal: str,
    expected_provenance_seal: str,
    expected_training_script_equivalence: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    task_id = _clearml_id(getattr(selector, "id", None), context="formal selector")
    if task_id != SELECTOR_TASK_ID:
        raise FormalMultiseedPlanError("formal selector identity mismatch")
    if _status(selector, context="formal selector") != "completed":
        raise FormalMultiseedPlanError("formal selector is not completed")
    if _task_parent(selector, context="formal selector") != SELECTOR_AUDIT_TASK_ID:
        raise FormalMultiseedPlanError("formal selector parent mismatch")
    _, script = _script_metadata(
        selector,
        context="formal selector",
        entry_point=SELECTOR_ENTRY_POINT,
        expected_sha256=SELECTOR_PRODUCER_SHA256,
    )
    parameters = _validate_selector_parameters(selector)
    if _artifact_names(selector, context="formal selector") != (SELECTOR_ARTIFACT,):
        raise FormalMultiseedPlanError("formal selector artifact inventory drifted")
    artifact, canonical_sha = _frozen_mapping(
        _artifact_mapping(selector, SELECTOR_ARTIFACT, context="formal selector"),
        context="formal candidate selection",
    )
    selector_seal = _require_seal(artifact, context="formal candidate selection")
    if (
        artifact.get("schema_version") != 3
        or artifact.get("document_type") != "resilient_v2x_formal_candidate_selection"
        or artifact.get("status") != "selected"
        or artifact.get("selection_stage") != "single_seed_candidate_screening"
        or artifact.get("selection_is_final") is not False
        or artifact.get("requires_multiseed_confirmation") is not True
        or artifact.get("audit_task_id") != SELECTOR_AUDIT_TASK_ID
        or artifact.get("leaderboard_task_id") != SELECTOR_LEADERBOARD_TASK_ID
        or artifact.get("training_controller_task_id")
        != SELECTOR_TRAINING_CONTROLLER_TASK_ID
        or artifact.get("training_provenance_task_id")
        != SELECTOR_TRAINING_PROVENANCE_TASK_ID
        or artifact.get("watcher_task_id") != SELECTOR_WATCHER_TASK_ID
        or artifact.get("training_progress_seal_sha256") != expected_progress_seal
        or artifact.get("training_provenance_seal_sha256") != expected_provenance_seal
        or _canonical_json(artifact.get("training_script_equivalence"))
        != _canonical_json(expected_training_script_equivalence)
        or artifact.get("evaluation_script_sha256") != EVALUATION_SCRIPT_SHA256
        or artifact.get("audit_seal_sha256") != expected_audit_seal
        or artifact.get("leaderboard_seal_sha256") != expected_leaderboard_seal
        or type(artifact.get("selected_candidate")) is not str
        or not artifact.get("selected_candidate")
    ):
        raise FormalMultiseedPlanError("formal candidate selection provenance drifted")
    _validate_source_provenance_binding(
        {
            key: artifact[key]
            for key in (
                "training_provenance_task_id",
                "training_provenance_seal_sha256",
                "source_revision_equivalence",
                "source_revision_equivalence_seal_sha256",
                "source_revision_subject_map",
                "source_revision_subject_map_seal_sha256",
                "evaluation_source_revision_tree_sha256",
                "evaluation_source_revision",
            )
        },
        context="formal selector",
    )
    snapshot = {
        "task_id": task_id,
        "parent_task_id": SELECTOR_AUDIT_TASK_ID,
        "status": "completed",
        "script": script,
        "parameters": parameters,
        "artifact_canonical_sha256": canonical_sha,
        "artifact_seal_sha256": selector_seal,
    }
    return artifact, snapshot


def _output_snapshot(
    output: object,
    *,
    output_task_id: str,
    expected_source: str,
) -> dict[str, object]:
    task_id = _clearml_id(getattr(output, "id", None), context="planner output")
    if task_id != output_task_id:
        raise FormalMultiseedPlanError("planner output identity drifted")
    if _task_parent(output, context="planner output") != SELECTOR_TASK_ID:
        raise FormalMultiseedPlanError("planner output parent drifted")
    status_value = _status(output, context="planner output")
    if status_value not in WAITING_STATUSES:
        raise FormalMultiseedPlanError("planner output status is not writable")
    _, script = _script_metadata(
        output,
        context="planner output",
        entry_point=PRODUCER_ENTRY_POINT,
        expected_source=expected_source,
    )
    names = _artifact_names(output, context="planner output")
    _validate_output_inventory(names)
    return {
        "task_id": task_id,
        "parent_task_id": SELECTOR_TASK_ID,
        "status": status_value,
        "script": script,
        "artifact_names": names,
    }


def _build_payloads(
    *,
    validator: ModuleType,
    output_task_id: str,
    output_parent_task_id: str,
    producer_script_sha256: str,
    selector_artifact: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    try:
        plan_value = validator.build_plan(
            source_d_script_sha256=SOURCE_D_SCRIPT_SHA256,
            source_d_equivalence_sha256=SOURCE_D_EQUIVALENCE_SHA256,
            formal_audit_seal_sha256=selector_artifact["audit_seal_sha256"],
            formal_selector_seal_sha256=selector_artifact["seal_sha256"],
            formal_selector_task_id=SELECTOR_TASK_ID,
            formal_selector_artifact=selector_artifact,
            selected_candidate=selector_artifact["selected_candidate"],
            training_provenance_task_id=TRAINING_PROVENANCE_TASK_ID,
            training_progress_artifact_seal_sha256=selector_artifact[
                "training_progress_seal_sha256"
            ],
            training_provenance_artifact_seal_sha256=selector_artifact[
                "training_provenance_seal_sha256"
            ],
            training_script_equivalence=selector_artifact[
                "training_script_equivalence"
            ],
            evaluation_script_sha256=selector_artifact["evaluation_script_sha256"],
        )
        plan_value = validator.validate_plan(plan_value)
    except Exception as error:
        raise FormalMultiseedPlanError(
            "formal multi-seed plan build or validation failed"
        ) from error
    plan, _ = _frozen_mapping(plan_value, context="formal multi-seed plan")
    receipt = _sealed(
        {
            "schema_version": 1,
            "receipt_type": "resilient_v2x_formal_multiseed_plan_receipt",
            "complete": True,
            "planner_task_id": output_task_id,
            "planner_parent_task_id": output_parent_task_id,
            "producer_entry_point": PRODUCER_ENTRY_POINT,
            "producer_script_sha256": producer_script_sha256,
            "plan_artifact_name": PLAN_ARTIFACT,
            "plan_seal_sha256": plan["seal_sha256"],
            "plan_canonical_sha256": _content_sha256(plan),
        }
    )
    return {PLAN_ARTIFACT: plan, PLAN_RECEIPT_ARTIFACT: receipt}


def _read_expected(
    task: object,
    name: str,
    *,
    expected: Mapping[str, object],
    validator: ModuleType,
) -> dict[str, object]:
    current, current_sha = _frozen_mapping(
        _artifact_mapping(task, name, context="planner output"),
        context=f"planner output artifact {name!r}",
    )
    expected_frozen, expected_sha = _frozen_mapping(
        expected, context=f"expected artifact {name!r}"
    )
    if current_sha != expected_sha or _canonical_json(current) != _canonical_json(
        expected_frozen
    ):
        raise FormalMultiseedPlanError(
            f"planner artifact {name!r} canonical bytes drifted"
        )
    if name == PLAN_ARTIFACT:
        try:
            validated = validator.validate_plan(current)
        except Exception as error:
            raise FormalMultiseedPlanError(
                "published plan failed validation"
            ) from error
        if _canonical_json(validated) != _canonical_json(current):
            raise FormalMultiseedPlanError("published plan validation changed bytes")
    else:
        _validate_receipt(current, plan=None)
    return current


def _validate_receipt(
    receipt_value: Mapping[str, object],
    *,
    plan: Mapping[str, object] | None,
) -> dict[str, object]:
    receipt, _ = _frozen_mapping(receipt_value, context="planner receipt")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "receipt_type",
            "complete",
            "planner_task_id",
            "planner_parent_task_id",
            "producer_entry_point",
            "producer_script_sha256",
            "plan_artifact_name",
            "plan_seal_sha256",
            "plan_canonical_sha256",
            "seal_sha256",
        },
        context="planner receipt",
    )
    _exact_int(receipt.get("schema_version"), 1, context="planner receipt schema")
    _exact_bool(receipt.get("complete"), True, context="planner receipt complete")
    if (
        receipt.get("receipt_type") != "resilient_v2x_formal_multiseed_plan_receipt"
        or receipt.get("producer_entry_point") != PRODUCER_ENTRY_POINT
        or receipt.get("plan_artifact_name") != PLAN_ARTIFACT
    ):
        raise FormalMultiseedPlanError("planner receipt contract drifted")
    _clearml_id(receipt.get("planner_task_id"), context="planner receipt task")
    _clearml_id(receipt.get("planner_parent_task_id"), context="planner receipt parent")
    _sha256(receipt.get("producer_script_sha256"), context="planner receipt producer")
    _sha256(receipt.get("plan_seal_sha256"), context="planner receipt plan seal")
    _sha256(receipt.get("plan_canonical_sha256"), context="planner receipt plan hash")
    _require_seal(receipt, context="planner receipt")
    if plan is not None and (
        receipt["plan_seal_sha256"] != plan.get("seal_sha256")
        or receipt["plan_canonical_sha256"] != _content_sha256(plan)
    ):
        raise FormalMultiseedPlanError("planner receipt plan binding drifted")
    return receipt


def _publish_sequence(
    output: object,
    payloads: Mapping[str, Mapping[str, object]],
    *,
    validator: ModuleType,
    validate_bindings: Callable[[], None],
) -> dict[str, object]:
    if tuple(payloads) != PUBLICATION_ORDER:
        raise FormalMultiseedPlanError("planner payload order drifted")
    frozen: dict[str, dict[str, object]] = {}
    for name in PUBLICATION_ORDER:
        frozen[name], _ = _frozen_mapping(
            payloads[name], context=f"expected artifact {name!r}"
        )
    _validate_receipt(frozen[PLAN_RECEIPT_ARTIFACT], plan=frozen[PLAN_ARTIFACT])
    validate_bindings()
    names = _artifact_names(output, context="planner output")
    _validate_output_inventory(names)
    uploader = getattr(output, "upload_artifact", None)
    flusher = getattr(output, "flush", None)
    if not callable(uploader) or not callable(flusher):
        raise FormalMultiseedPlanError("planner output cannot publish artifacts")
    for name in PUBLICATION_ORDER:
        validate_bindings()
        if name in _artifact_names(output, context="planner output"):
            _read_expected(output, name, expected=frozen[name], validator=validator)
            continue
        uploaded = uploader(name, artifact_object=frozen[name], wait_on_upload=True)
        if uploaded is not True:
            raise FormalMultiseedPlanError(f"failed to upload artifact {name!r}")
        flushed = flusher(wait_for_uploads=True)
        if flushed is not None and flushed is not True:
            raise FormalMultiseedPlanError(f"failed to flush artifact {name!r}")
        validate_bindings()
        if name not in _artifact_names(output, context="planner output"):
            raise FormalMultiseedPlanError(f"artifact {name!r} absent after upload")
        _read_expected(output, name, expected=frozen[name], validator=validator)
    validated_receipt: dict[str, object] | None = None
    # Require two consecutive, fresh, complete snapshots.  Each pass reloads
    # every dependency and the output before reading both exact output payloads.
    # The second pass detects a persistent mutation made after an earlier task
    # was checked during the first sequential snapshot.
    for _snapshot_pass in range(2):
        validate_bindings()
        plan = _read_expected(
            output,
            PLAN_ARTIFACT,
            expected=frozen[PLAN_ARTIFACT],
            validator=validator,
        )
        receipt = _read_expected(
            output,
            PLAN_RECEIPT_ARTIFACT,
            expected=frozen[PLAN_RECEIPT_ARTIFACT],
            validator=validator,
        )
        validated_receipt = _validate_receipt(receipt, plan=plan)
    if validated_receipt is None:  # pragma: no cover - fixed non-empty range
        raise FormalMultiseedPlanError("planner final readback did not run")
    return validated_receipt


def _bind_output_parent(output: object) -> None:
    _reload(output, context="planner output parent preflight")
    current = _task_parent(output, context="planner output")
    if current == SELECTOR_TASK_ID:
        return
    if current:
        raise FormalMultiseedPlanError("planner output already has a different parent")
    setter = getattr(output, "set_parent", None)
    if not callable(setter):
        raise FormalMultiseedPlanError("planner output cannot set selector parent")
    result = setter(SELECTOR_TASK_ID)
    if result is not None and result is not True:
        raise FormalMultiseedPlanError("planner output rejected selector parent")
    _reload(output, context="planner output parent binding")
    if _task_parent(output, context="planner output") != SELECTOR_TASK_ID:
        raise FormalMultiseedPlanError("planner output parent did not persist")


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    validator_path: Path | None = None,
) -> dict[str, object]:
    if (
        type(args.poll_seconds) not in {int, float}
        or not math.isfinite(float(args.poll_seconds))
        or args.poll_seconds <= 0
        or type(args.timeout_hours) not in {int, float}
        or not math.isfinite(float(args.timeout_hours))
        or args.timeout_hours <= 0
    ):
        raise ValueError("poll interval and timeout must be finite and positive")
    if task_class is None:
        raise FormalMultiseedPlanError("ClearML is unavailable")
    _require_deployment_pins()
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise FormalMultiseedPlanError("ClearML Task cannot resolve dependencies")
    controller = getter(task_id=TRAINING_CONTROLLER_TASK_ID)
    provenance = getter(task_id=TRAINING_PROVENANCE_TASK_ID)
    watcher = getter(task_id=WATCHER_TASK_ID)
    leaderboard = getter(task_id=LEADERBOARD_TASK_ID)
    audit = getter(task_id=AUDIT_TASK_ID)
    selector = getter(task_id=SELECTOR_TASK_ID)
    dependencies = (controller, provenance, watcher, leaderboard, audit, selector)
    if len({id(task) for task in dependencies}) != len(dependencies):
        raise FormalMultiseedPlanError("planner dependencies alias each other")
    if output_task is None:
        current = getattr(task_class, "current_task", None)
        output_task = current() if callable(current) else None
    if output_task is None:
        raise FormalMultiseedPlanError("planner requires a current output task")
    output_id = _clearml_id(getattr(output_task, "id", None), context="planner output")
    if (
        output_task is controller
        or output_task is provenance
        or output_task is watcher
        or output_task is selector
        or output_task is audit
        or output_task is leaderboard
        or output_id
        in (
            TRAINING_CONTROLLER_TASK_ID,
            TRAINING_PROVENANCE_TASK_ID,
            WATCHER_TASK_ID,
            SELECTOR_TASK_ID,
            SELECTOR_AUDIT_TASK_ID,
            SELECTOR_LEADERBOARD_TASK_ID,
        )
    ):
        raise FormalMultiseedPlanError("planner output aliases a dependency")
    runtime_source = _runtime_source()
    _output_snapshot(
        output_task, output_task_id=output_id, expected_source=runtime_source
    )
    deadline = monotonic_clock() + float(args.timeout_hours) * 3600.0
    _wait_for_chain(
        (
            ("formal training controller", controller),
            ("formal training provenance", provenance),
            ("formal evaluation watcher", watcher),
            ("formal leaderboard", leaderboard),
            ("formal comparability audit", audit),
            ("formal selector", selector),
        ),
        deadline=deadline,
        poll_seconds=float(args.poll_seconds),
        monotonic_clock=monotonic_clock,
        sleeper=sleeper,
    )
    (
        controller_snapshot,
        provenance_snapshot,
        watcher_snapshot,
        progress_seal,
        provenance_seal,
    ) = _training_chain_snapshots(controller, provenance, watcher)
    _reload(leaderboard, context="formal leaderboard")
    leaderboard_artifact, leaderboard_snapshot = _leaderboard_snapshot(leaderboard)
    _reload(audit, context="formal comparability audit")
    audit_artifact, audit_snapshot = _audit_snapshot(audit)
    if (
        audit_artifact.get("training_progress_seal_sha256") != progress_seal
        or audit_artifact.get("training_provenance_seal_sha256") != provenance_seal
    ):
        raise FormalMultiseedPlanError(
            "formal audit provenance artifact seals are not cross-bound"
        )
    _cross_validate_leaderboard_audit(leaderboard_artifact, audit_artifact)
    selector_artifact, selector_snapshot = _selector_snapshot(
        selector,
        expected_audit_seal=str(audit_snapshot["artifact_seal_sha256"]),
        expected_leaderboard_seal=str(leaderboard_snapshot["artifact_seal_sha256"]),
        expected_progress_seal=progress_seal,
        expected_provenance_seal=provenance_seal,
        expected_training_script_equivalence=dict(
            audit_artifact["training_script_equivalence"]
        ),
    )
    for key in (
        "training_provenance_task_id",
        "training_provenance_seal_sha256",
        "source_revision_equivalence",
        "source_revision_equivalence_seal_sha256",
        "source_revision_subject_map",
        "source_revision_subject_map_seal_sha256",
        "evaluation_source_revision_tree_sha256",
        "evaluation_source_revision",
    ):
        if selector_artifact.get(key) != audit_artifact.get(key):
            raise FormalMultiseedPlanError(
                f"formal audit and selector {key} are not cross-bound"
            )
    validator = _load_validator(validator_path)
    # This invokes the pure planner's complete selector-schema validator.
    payloads = _build_payloads(
        validator=validator,
        output_task_id=output_id,
        output_parent_task_id=SELECTOR_TASK_ID,
        producer_script_sha256=_sha256_text(runtime_source),
        selector_artifact=selector_artifact,
    )

    def validate_bindings() -> None:
        for task, context in (
            (controller, "controller publication binding"),
            (provenance, "provenance publication binding"),
            (watcher, "watcher publication binding"),
        ):
            _reload(task, context=context)
        current_chain = _training_chain_snapshots(controller, provenance, watcher)
        if current_chain[:3] != (
            controller_snapshot,
            provenance_snapshot,
            watcher_snapshot,
        ) or current_chain[3:] != (progress_seal, provenance_seal):
            raise FormalMultiseedPlanError(
                "formal training provenance chain changed during publication"
            )
        _reload(leaderboard, context="leaderboard publication binding")
        current_leaderboard, current_leaderboard_snapshot = _leaderboard_snapshot(
            leaderboard
        )
        if current_leaderboard_snapshot != leaderboard_snapshot or _canonical_json(
            current_leaderboard
        ) != _canonical_json(leaderboard_artifact):
            raise FormalMultiseedPlanError(
                "formal leaderboard changed during publication"
            )
        _reload(audit, context="audit publication binding")
        current_audit, current_audit_snapshot = _audit_snapshot(audit)
        if current_audit_snapshot != audit_snapshot or _canonical_json(
            current_audit
        ) != _canonical_json(audit_artifact):
            raise FormalMultiseedPlanError(
                "formal comparability audit changed during publication"
            )
        _cross_validate_leaderboard_audit(current_leaderboard, current_audit)
        if (
            current_audit.get("training_progress_seal_sha256") != progress_seal
            or current_audit.get("training_provenance_seal_sha256") != provenance_seal
        ):
            raise FormalMultiseedPlanError(
                "formal audit provenance artifact seals changed during publication"
            )
        _reload(selector, context="selector publication binding")
        current_selector, current_snapshot = _selector_snapshot(
            selector,
            expected_audit_seal=str(audit_snapshot["artifact_seal_sha256"]),
            expected_leaderboard_seal=str(leaderboard_snapshot["artifact_seal_sha256"]),
            expected_progress_seal=progress_seal,
            expected_provenance_seal=provenance_seal,
            expected_training_script_equivalence=dict(
                audit_artifact["training_script_equivalence"]
            ),
        )
        if current_snapshot != selector_snapshot or _canonical_json(
            current_selector
        ) != _canonical_json(selector_artifact):
            raise FormalMultiseedPlanError(
                "selector snapshot changed during publication"
            )
        _reload(output_task, context="planner output publication binding")
        _output_snapshot(
            output_task, output_task_id=output_id, expected_source=runtime_source
        )

    validate_bindings()
    return _publish_sequence(
        output_task,
        payloads,
        validator=validator,
        validate_bindings=validate_bindings,
    )


def _write_new(path: Path, content: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.emit_standalone is not None:
        standalone = generate_standalone_source()
        _write_new(args.emit_standalone, standalone)
        print(
            json.dumps(
                {
                    "standalone_sha256": _sha256_text(standalone),
                    "plan_validator_sha256": PLAN_VALIDATOR_SHA256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    _require_deployment_pins()
    if Task is None:
        raise FormalMultiseedPlanError("ClearML is unavailable")
    output = Task.init(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X formal multi-seed plan",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
    )
    _bind_output_parent(output)
    receipt = run(args, task_class=Task, output_task=output)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
