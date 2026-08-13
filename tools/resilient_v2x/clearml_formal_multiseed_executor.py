#!/usr/bin/env python3
"""Execute one sealed initial-only formal multi-seed plan on ClearML.

The executor is deliberately fail closed.  It consumes a completed planner
task, the four-artifact Source-D evidence chain, and a completed clean-teacher
quality gate.  It never computes a multi-seed gate and never enables fallback.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import BinaryIO, Literal
from urllib.parse import unquote, urlsplit

try:
    from allegroai import Task
    from allegroai.binding.artifacts import Artifact as ClearMLArtifact
except ImportError:
    try:
        from clearml import Task
        from clearml.binding.artifacts import Artifact as ClearMLArtifact
    except ImportError:  # pragma: no cover - only exercised without ClearML
        Task = None  # type: ignore[assignment]
        ClearMLArtifact = None  # type: ignore[assignment,misc]


DEFAULT_PROJECT = "ResilientV2X/Training"
FILES_SERVER_URI = "http://10.100.34.118:8081"
EXPECTED_FILES_SERVER_HOST = "10.100.34.118"
EXPECTED_FILES_SERVER_PORT = 8081

EXECUTOR_ENTRY_POINT = "clearml_formal_multiseed_executor.py"
EXECUTOR_TASK_NAME = "ResilientV2X formal initial multi-seed executor"
PLANNER_ENTRY_POINT = "formal_multiseed_plan.py"
SOURCE_D_PRODUCER_ENTRY_POINT = "clearml_formal_source_d_evidence.py"
TEACHER_GATE_ENTRY_POINT = "clearml_teacher_quality_gate.py"
SOURCE_C_ENTRY_POINT = "clearml_5090_bootstrap.py"
TRAINING_ENTRY_POINT = "clearml_5090_bootstrap.py"

# The pinned planner producer parses these defaults before Task.init().  The
# ClearML argparse bridge records all three entries and get_parameters(cast=False)
# returns their string representations, including the unused Path default.
PLANNER_PARAMETER_CONTRACT = {
    "Args/poll_seconds": "60.0",
    "Args/timeout_hours": "720.0",
    "Args/emit_standalone": "",
}

PINNED_PLAN_VALIDATOR_SHA256 = (
    "a7bc17bcd1afb1c836924f853f63acae8e26801557f784cd6f0bcd7148c35993"
)
# Public pin interface.  The planner task runs the independently reviewed
# standalone producer, which embeds the pinned pure plan validator.  The two
# hashes are intentionally distinct and neither is inferred from a task ID.
UNRESOLVED_PLANNER_PRODUCER_SHA256 = "0" * 64
PROVISIONAL_PLANNER_PRODUCER_SHA256 = (
    "0b5f3b60204ec876f01f417bf9b7c7e8cabd8a219d92aa453f7249ca2ad19047"
)
EXPECTED_PLANNER_PRODUCER_SHA256 = PROVISIONAL_PLANNER_PRODUCER_SHA256
PLANNER_PRODUCER_PIN_FINALIZED = False  # REQUIRED_FINAL_DEPLOYMENT_REPIN
EXPECTED_SOURCE_D_PRODUCER_SHA256 = (
    "d5b759f38d39a9f349ab6e716c07fda53eb3e1635687ce4a077e0405920ffeec"
)
EXPECTED_TEACHER_GATE_PRODUCER_SHA256 = (
    "432514f19f4b663d1023e1ad2faed28c8b2c983626660a9c3bfd011302edb557"
)
EXPECTED_TEACHER_SCRIPT_SHA256 = (
    "4dfe2e9d2ee3076df1679818211f40b7c2ebcbc73efb4cdae2f230c971485b67"
)
EXPECTED_SOURCE_C_SCRIPT_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
LEGACY_TRAINING_SCRIPT_SHA256 = EXPECTED_SOURCE_C_SCRIPT_SHA256
CANONICAL_TRAINING_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
EVALUATION_SCRIPT_SHA256 = CANONICAL_TRAINING_SCRIPT_SHA256
LEGACY_SCRIPT_SUBJECTS = ("support_residual", "no_distillation")
TRAINING_SCRIPT_ONLY_DIFFERENCE = "NESTED_TEACHER_EXPERIMENTS membership"
TRAINING_SCRIPT_RUNTIME_USAGE_CLOSURE = (
    "expect_nested_teacher_keyword",
    "expected_nested_teacher_contract_field",
)
SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = (
    "29de9700cac66f9998be643e85a8ec646c04ec17fddbb6207bc1438e9e73941b"
)
SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = (
    "c3170f4a88b080f9cc7267053f354640dd260687cb2c35a6f7ffed73d69f4154"
)
EVALUATION_SOURCE_REVISION_TREE_SHA256 = (
    "5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6"
)
SELECTOR_DEPENDENCY_TASK_ID_FIELDS = (
    "audit_task_id",
    "leaderboard_task_id",
    "training_controller_task_id",
    "training_provenance_task_id",
    "watcher_task_id",
)
EXPECTED_SOURCE_D_SCRIPT_SHA256 = (
    "e7a9ab0fb05339223cf2c18c52eb72652c311733bf96d1058aa7a769096cf8c3"
)
EXPECTED_SOURCE_D_EQUIVALENCE_SHA256 = (
    "1156fe53f2fe924f91c1c6b50b6b21090d98cd2d74840d6d6f5a358316420433"
)
EXPECTED_SOURCE_D_BUILDER_SHA256 = (
    "904fafd08d710b03b62bc57140121f2a546ada8fa2fb8763e6db8a44b9f8f7e7"
)
SOURCE_D_TRANSFORMATION_ID = "source-c-to-source-d-explicit-seed-evidence-v2"
SOURCE_D_DECLARED_REPLACEMENT_COUNT = 22
SOURCE_D_UNCHANGED_SEGMENT_COUNT = 23
SOURCE_D_STAGING_ANCHOR_NAMES = (
    "seal_evidence_security_imports",
    "declare_controlled_evidence_contract",
    "declare_controlled_evidence_receipts",
    "stage_and_verify_controlled_evidence",
    "stage_after_controlled_runner",
    "upload_only_sealed_controlled_evidence",
)

PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
TRAINING_OVERLAY_PROTOCOL_SEED = 20_250_218
SOURCE_C_TASK_ID = "95e72da24d464ab08d117dedabd6652e"
SOURCE_C_PARENT_TASK_ID = "6525107e60ae4104a2800731d74ecd4e"
SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
SOURCE_ARCHIVE_BYTES = 1_222_481
SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
NATIVE_BUNDLE_BYTES = 753_382_966
NATIVE_BUNDLE_SHA256 = (
    "19b8e7f5edc8216d4b43cb17854dccafe4fc9fe46995a88803e6342eeaa22b21"
)
NATIVE_BUILD_TASK_ID = "9055c0d3c4dd450c8a75dddfb21a56bd"
BASE_IMAGE_MANIFEST_DIGEST = (
    "sha256:dbc586035fffb2bc030e807290d43e8d4edf44ee864fa5c832db44ed099fc415"
)

BUILD_MANIFEST_SHA256 = (
    "21c6ab7a6e9a2823ba111289a42e7f882c5f4ba02c73175106ff251fe5864a43"
)
MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
MANIFEST_FILE_SHA256 = (
    "6d0a37698d39891d212a33ac042b5ac62ce47ca5db6fa0303af3b9b55ab891a2"
)
OVERLAY_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
OVERLAY_INDEX_FILE_SHA256 = (
    "3418a0aa7025eb2cae19a054baccbb0f1022cbc65aa4813ef0bc25d353914725"
)
SAMPLE_IDS_SHA256 = "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
OFFICIAL_SPLIT_SHA256 = (
    "d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
)
CONTROLLED_EVALUATOR_SHA256 = (
    "d233e054f2b608bc441de25833fb89d117197d841752812a0054e0514995ad36"
)
RUNTIME_PYTHON = "/opt/resilient-v2x-5090/bin/python"
GROUND_TRUTH_CONTENT_SHA256 = (
    "44f08b34a76bdd9dcf0f5b179a24eb229f2c73358635ad29b101ac5651675edc"
)
SUBJECT_CONFIG_SHA256 = {
    "coformernet": "1ef915ec50e0bbd2b6a5f9c9921faf3b5ccf42860b2c42610e79e4542703b675",
    "ffnet": "624c656e1c494e8d3e47f38462c8f6e2f22d330ba6ee06bdbdf0d36b092be8a7",
    "bevfusion": "b51cfd386185b91d319721917e9ac0de85bb3253d6abc48cf4e4d163312f3922",
    "v2x_vit": "b9806fe1824e6a9f12472888b1d1e22ff7a0b1851bd080c1777984b0af0bd4bd",
    "cobevt": "261e656f11199e17a030d65c0319101212aafd13cd5f0b57f5a0127cea869d94",
    "ego_only": "01274901ac38c958f37619660c6ae8fd1a03346947b0a7910367aed806f3fbdb",
    "fcooper": "86a9dfafbd6455c1dd4ba54e4446ab3cdf072e3658fe7ebc7304bdc416b610c0",
    "attfuse": "86057c73f9af227c85252b5038297794ad896500db5fcaf8b9f8a8bd7ed81955",
    "v2vnet": "8e8f5e33828f37b10d770a048aac183678f9da7ed346026f19b754d8c5a29da1",
    "when2com": "e1fe66d0b5069884d7c76c6e389571627b2e41348a4a4a64d6ce78ad0de015b7",
    "where2comm": "9e90f2542fb3998550b67344ade1e4571d8e4bf1b712c8727c4e0c2c28e7c61e",
    "late_fusion": "e510b227499533e62d94d3b2b7768ac7342ad5d82e0aa837989f3a446e95f324",
    "disconet": "82818e7157fd9c21fca04783094fc8dd9b3d6d71b0bb5898edf69496dc81c0bf",
    "how2comm": "451532356eaa249cc4d41d04d5d3df3c10a670b365521f8b85077238b5c6181f",
    "resilient_v2x": "19ae2d94eec8e8123293bd6f536710aa6c8b3efe31ebf651e28b6ee728fb8205",
    "support_residual": "00d746aaf4d1c67e04f4df8ae1b42358498b8503feb4be72ae7f6811488748b9",
    "linear_no_distillation": "c76c2256a50b9b2c688528340d4baa04a3be1b4449720a7ba4409877bd596960",
    "no_distillation_peak_lr_3e4": "2e31c210ecf639372a9a528199c05837eb998472b60a62e3b28717fdb6d23f91",
}
TRANSPORT_OVERLAY_SHA256 = {
    0: "d9bacedf17b3a6daaf8ea65b11b4706c64b4e079a85c0f0542f37dfd2c6ae9aa",
    100: "35ba0988616bfa2fe2dced9a91028ceecbf0fc0ba81699689c6966784cb023b7",
    200: "6fa38ba84bf512bb18e1ae8931cf32c1b098ce6aa262cef0b0a50495c2d004fc",
    300: "04f47695063e3b1107af94b10f5d4e63e453bc8086d6693bc9d415741ceb40ff",
}
FAULT_OVERLAY_SHA256 = {
    (0, "L-Fail"): "677becb4b8f11f2a42232ba5600304d6d3ff457b0391b52d31ca1363ba9162c1",
    (0, "C-Fail"): "f5365033039ba6422dcef8f93f8865da4364cadbe4b099b16551d067456c1fec",
    (100, "L-Fail"): "8f0b9263c49305fc9d529bfe10e7da9e78ed9d84982b58730d87ca00673fb213",
    (100, "C-Fail"): "768fa0d3c315948b02a86ebad3ce407e28ef108e33e30169573d78144522fe8b",
    (200, "L-Fail"): "1df6c4a713d3e12ba8fbfbf18e49045ef4f3c83948e702b9141aa8b935e18b14",
    (200, "C-Fail"): "fc928fc4fc0433460b185ae2bd829e6c062622c2c39d0821e97a4838290ce63b",
    (300, "L-Fail"): "948de1d9192439b6e120b8aacdaf189469978449be11cecd69aa4b94ba9193d8",
    (300, "C-Fail"): "aea9a8737edf9387ca111f4f9bd429732a0f3bc0c20da1fb7de7efd4f34d6e49",
}

PLAN_ARTIFACT = "formal_multiseed_plan"
PLANNER_RECEIPT_ARTIFACT = "formal_multiseed_plan_receipt"
EXECUTION_MANIFEST_ARTIFACT = "formal_multiseed_execution_manifest"
SOURCE_C_SNAPSHOT_ARTIFACT = "formal_source_c_snapshot"
SOURCE_D_SCRIPT_ARTIFACT = "formal_source_d_script"
SOURCE_D_EQUIVALENCE_ARTIFACT = "formal_source_d_equivalence"
SOURCE_D_RECEIPT_ARTIFACT = "formal_source_d_evidence_receipt"
SOURCE_D_ARTIFACT_ORDER = (
    SOURCE_C_SNAPSHOT_ARTIFACT,
    SOURCE_D_SCRIPT_ARTIFACT,
    SOURCE_D_EQUIVALENCE_ARTIFACT,
    SOURCE_D_RECEIPT_ARTIFACT,
)
TEACHER_QUALITY_GATE_ARTIFACT = "teacher_quality_gate"
TEACHER_RUN_CONTRACT_ARTIFACT = "run_contract"
TEACHER_CHECKPOINT_ARTIFACT = "teacher_checkpoint_contract"
RUN_CONTRACT_ARTIFACT = "run_contract"
FINAL_CHECKPOINT_ARTIFACT = "final_checkpoint_contract"
INITIALIZATION_AUDIT_ARTIFACT = "common_teacher_initialization_audit"
METRICS_ARTIFACT = "controlled_baseline_metrics"
EVALUATION_PLAN_ARTIFACT = "evaluation_plan"
EVALUATION_EVIDENCE_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
EVALUATION_EVIDENCE_MAX_MEMBER_BYTES = 256 * 1024 * 1024
EVALUATION_EVIDENCE_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024
EVALUATION_EVIDENCE_MAX_COMPRESSION_RATIO = 128.0
EVALUATION_EVIDENCE_ARTIFACT = "controlled_baseline_evidence"
JSON_ARTIFACT_MAX_BYTES = 64 * 1024 * 1024
EVALUATION_ARTIFACTS = frozenset(
    {
        RUN_CONTRACT_ARTIFACT,
        EVALUATION_PLAN_ARTIFACT,
        METRICS_ARTIFACT,
        EVALUATION_EVIDENCE_ARTIFACT,
    }
)

QUEUE_IDS = {
    "GPU4-A100": "9350f33af13a448da8339eb7bea52fdf",
    "GPU4-V100": "3925e906ce484620a941e6ccedc4bdbd",
    "GPU4-5090": "5a84454c072349069e7b61af38637c6d",
}
QUEUE_CAPACITY = {"GPU4-A100": 2, "GPU4-V100": 1, "GPU4-5090": 1}
PLANNED_GPU_MODELS = frozenset({"A100", "V100", "RTX5090"})
SOURCE_D_RUNTIME_PROFILE = "rtx5090"
PORTABLE_RUNNER_LOAD_MARKER = "_validate_rtx5090_runtime_contract_multi_gpu"
PORTABLE_RUNNER_LOAD_MARKER_COUNT = 2
LEGACY_RUNNER_LOAD_TARGET_ANCHOR_COUNT = 0

DELAYS_MS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
AP_METRIC_KEYS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)
COUNT_METRIC_KEYS = (
    "resilient_v2x/sample_count",
    "resilient_v2x/car_ground_truth_count",
    "resilient_v2x/car_prediction_count",
    "resilient_v2x/unsupported_sample_count",
)
DIAGNOSTIC_METRIC_KEYS = (
    "resilient_v2x/diagnostic_pred_z_bottom_p50",
    "resilient_v2x/diagnostic_gt_z_bottom_p50",
    "resilient_v2x/diagnostic_pred_height_p50",
    "resilient_v2x/diagnostic_gt_height_p50",
    "resilient_v2x/diagnostic_bev_match_050_count",
    "resilient_v2x/diagnostic_bev_match_050_abs_z_error_p50",
    "resilient_v2x/diagnostic_bev_match_050_vertical_iou_p50",
    "resilient_v2x/diagnostic_bev_match_050_3d_iou_p50",
)
EVALUATION_METRIC_KEYS = (
    *COUNT_METRIC_KEYS,
    *AP_METRIC_KEYS,
    *DIAGNOSTIC_METRIC_KEYS,
)

CLEAN_TEACHER_MODEL_NAME = "ResilientV2X clean teacher"
COMMON_TEACHER_INITIALIZATION_CONTRACT = "shared-only-clean-teacher-initialization-v1"
COMMON_TEACHER_INITIALIZATION_PREFIXES = (
    "lidar_encoder.",
    "camera_encoder.",
    "bbox_head.",
    "detection_projection.",
)
COMMON_TEACHER_SOURCE_KEYS = 617
COMMON_TEACHER_SOURCE_NUMEL = 35_811_485
COMMON_TEACHER_SOURCE_BYTES = 143_246_244
COMMON_TEACHER_SHARED_KEYS = 468
COMMON_TEACHER_SHARED_NUMEL = 31_506_934
COMMON_TEACHER_SHARED_BYTES = 126_028_040
COMMON_TEACHER_FUSION_KEYS = 149
NESTED_TEACHER_SUBJECTS = frozenset(
    {
        "concat_capacity_matched",
        "no_delay_metadata",
        "no_reliability",
        "ptf_linear",
        "ptf_none",
        "resilient_v2x",
        "router_static",
        "router_uniform",
        "support_residual",
    }
)
ZERO_FUSION_SUBJECTS = frozenset({"ego_only", "fcooper"})
COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME = (
    "common_teacher_initialization_audit.json"
)
EMPTY_TENSOR_MAPPING_SHA256 = hashlib.sha256(b"").hexdigest()

WAITING_STATUSES = frozenset({"created", "queued", "in_progress"})
ACTIVE_STATUSES = frozenset({"queued", "in_progress"})
FAILED_STATUSES = frozenset(
    {
        "failed",
        "stopped",
        "closed",
        "published",
        "publishing",
        "rejected",
        "unknown",
    }
)

_EMBEDDED_PLAN_VALIDATOR_B64 = ""  # __FORMAL_MULTISEED_PLAN_BYTES_V2__


class FormalMultiseedExecutionError(RuntimeError):
    """Raised when any execution or provenance invariant fails closed."""


def _validate_json_domain(value: object, *, context: str = "document") -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise FormalMultiseedExecutionError(
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
    raise FormalMultiseedExecutionError(
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


def _freeze(value: object, *, context: str) -> object:
    try:
        frozen = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FormalMultiseedExecutionError(
            f"{context} is not canonical JSON"
        ) from error
    return frozen


def _frozen_mapping(value: object, *, context: str) -> dict[str, object]:
    frozen = _freeze(value, context=context)
    if type(frozen) is not dict:
        raise FormalMultiseedExecutionError(f"{context} must be a JSON object")
    return frozen


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _producer_content_sha256(value: object) -> str:
    _validate_json_domain(value)
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_producer_content_hash(
    value: Mapping[str, object], *, field: str, context: str
) -> str:
    observed = _sha256(value.get(field), f"{context} {field}")
    unhashed = _frozen_mapping(value, context=context)
    unhashed.pop(field, None)
    if _producer_content_sha256(unhashed) != observed:
        raise FormalMultiseedExecutionError(f"{context} {field} mismatch")
    return observed


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = _frozen_mapping(value, context="sealed document")
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = _sha256(value.get("seal_sha256"), f"{context} seal")
    if _sealed(value)["seal_sha256"] != observed:
        raise FormalMultiseedExecutionError(f"{context} seal SHA-256 mismatch")
    return observed


def _require_content_hash(
    value: Mapping[str, object], *, field: str, context: str
) -> str:
    observed = _sha256(value.get(field), f"{context} {field}")
    unhashed = _frozen_mapping(value, context=context)
    unhashed.pop(field, None)
    if _content_sha256(unhashed) != observed:
        raise FormalMultiseedExecutionError(f"{context} {field} mismatch")
    return observed


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise FormalMultiseedExecutionError(
            f"{context} keys mismatch; "
            f"missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _clearml_id(value: object, context: str) -> str:
    if type(value) is not str or not _is_lower_hex(value, 32):
        raise FormalMultiseedExecutionError(
            f"{context} must be a lowercase 32-hex ClearML ID"
        )
    return value


def _sha256(value: object, context: str) -> str:
    if type(value) is not str or not _is_lower_hex(value, 64):
        raise FormalMultiseedExecutionError(f"{context} must be a lowercase SHA-256")
    return value


def _require_deployment_pins() -> str:
    producer_sha256 = _sha256(
        EXPECTED_PLANNER_PRODUCER_SHA256,
        "compiled planner producer pin",
    )
    if (
        producer_sha256 == UNRESOLVED_PLANNER_PRODUCER_SHA256
        or not PLANNER_PRODUCER_PIN_FINALIZED
    ):
        raise FormalMultiseedExecutionError(
            "planner producer deployment pin is not final"
        )
    return producer_sha256


def _exact_int(value: object, expected: int, *, context: str) -> int:
    if type(value) is not int or value != expected:
        raise FormalMultiseedExecutionError(
            f"{context} must be exactly integer {expected}"
        )
    return value


def _positive_int(value: object, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise FormalMultiseedExecutionError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, context: str) -> int:
    if type(value) is not int or value < 0:
        raise FormalMultiseedExecutionError(f"{context} must be a nonnegative integer")
    return value


def _exact_bool(value: object, expected: bool, *, context: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise FormalMultiseedExecutionError(f"{context} must be exactly {expected!r}")
    return value


def _validate_training_script_equivalence(
    value: object,
    *,
    context: str,
) -> dict[str, object]:
    equivalence = _frozen_mapping(value, context=context)
    _require_exact_keys(
        equivalence,
        {
            "legacy_script_sha256",
            "canonical_script_sha256",
            "legacy_script_subjects",
            "only_difference",
            "runtime_usage_closure",
        },
        context=context,
    )
    legacy_sha = _sha256(
        equivalence.get("legacy_script_sha256"),
        f"{context} legacy script",
    )
    canonical_sha = _sha256(
        equivalence.get("canonical_script_sha256"),
        f"{context} canonical script",
    )
    if (
        legacy_sha != LEGACY_TRAINING_SCRIPT_SHA256
        or canonical_sha != CANONICAL_TRAINING_SCRIPT_SHA256
        or legacy_sha == canonical_sha
        or equivalence.get("legacy_script_subjects") != list(LEGACY_SCRIPT_SUBJECTS)
        or equivalence.get("only_difference") != TRAINING_SCRIPT_ONLY_DIFFERENCE
        or equivalence.get("runtime_usage_closure")
        != list(TRAINING_SCRIPT_RUNTIME_USAGE_CLOSURE)
    ):
        raise FormalMultiseedExecutionError(f"{context} semantic contract mismatch")
    return equivalence


def _formal_plan_provenance(
    value: Mapping[str, object],
) -> dict[str, object]:
    plan = _frozen_mapping(value, context="formal plan provenance source")
    _exact_int(plan.get("schema_version"), 2, context="formal plan schema")
    _require_seal(plan, context="formal plan provenance source")
    formal_audit_seal = _sha256(
        plan.get("formal_audit_seal_sha256"),
        "formal plan audit seal",
    )
    training_provenance_task_id = _clearml_id(
        plan.get("training_provenance_task_id"),
        "formal plan training provenance task",
    )
    progress_seal = _sha256(
        plan.get("training_progress_artifact_seal_sha256"),
        "formal plan training progress artifact seal",
    )
    provenance_seal = _sha256(
        plan.get("training_provenance_artifact_seal_sha256"),
        "formal plan training provenance artifact seal",
    )
    training_script_equivalence = _validate_training_script_equivalence(
        plan.get("training_script_equivalence"),
        context="formal plan training script equivalence",
    )
    evaluation_script_sha256 = _sha256(
        plan.get("evaluation_script_sha256"),
        "formal plan evaluation script",
    )
    if evaluation_script_sha256 != EVALUATION_SCRIPT_SHA256:
        raise FormalMultiseedExecutionError(
            "formal plan evaluation script semantic contract mismatch"
        )

    selector_provenance = plan.get("formal_selector_provenance")
    if not isinstance(selector_provenance, Mapping):
        raise FormalMultiseedExecutionError(
            "formal plan selector provenance is invalid"
        )
    selector_task_id = _clearml_id(
        selector_provenance.get("task_id"),
        "formal plan selector task",
    )
    raw_selector_artifact = selector_provenance.get("artifact")
    if not isinstance(raw_selector_artifact, Mapping):
        raise FormalMultiseedExecutionError("formal plan selector artifact is invalid")
    selector_artifact = _frozen_mapping(
        raw_selector_artifact,
        context="formal plan selector artifact",
    )
    _exact_int(
        selector_artifact.get("schema_version"),
        3,
        context="formal plan selector artifact schema",
    )
    selector_artifact_seal = _require_seal(
        selector_artifact,
        context="formal plan selector artifact",
    )
    selector_provenance_seal = _sha256(
        selector_provenance.get("artifact_seal_sha256"),
        "formal plan selector provenance artifact seal",
    )
    initial_selector_seal = _sha256(
        plan.get("initial_formal_selector_seal_sha256"),
        "formal plan initial selector seal",
    )
    if not (
        selector_artifact_seal == selector_provenance_seal == initial_selector_seal
    ):
        raise FormalMultiseedExecutionError(
            "formal plan selector seal cross-binding mismatch"
        )
    if selector_artifact.get("audit_seal_sha256") != formal_audit_seal:
        raise FormalMultiseedExecutionError(
            "formal plan audit seal cross-binding mismatch"
        )
    expected_dependency_fields = set(SELECTOR_DEPENDENCY_TASK_ID_FIELDS)
    observed_dependency_fields = {
        field for field in selector_artifact if field.endswith("_task_id")
    }
    if observed_dependency_fields != expected_dependency_fields:
        raise FormalMultiseedExecutionError(
            "formal plan selector dependency inventory must contain exactly five "
            "task ID fields; "
            f"missing={sorted(expected_dependency_fields - observed_dependency_fields)!r}, "
            f"extra={sorted(observed_dependency_fields - expected_dependency_fields)!r}"
        )
    selector_dependency_task_ids = {
        field: _clearml_id(
            selector_artifact.get(field),
            f"formal plan selector dependency {field}",
        )
        for field in SELECTOR_DEPENDENCY_TASK_ID_FIELDS
    }
    dependency_ids = set(selector_dependency_task_ids.values())
    if len(dependency_ids) != 5 or selector_task_id in dependency_ids:
        raise FormalMultiseedExecutionError(
            "formal plan selector dependency inventory must contain five unique "
            "non-selector tasks"
        )
    if (
        selector_dependency_task_ids["training_provenance_task_id"]
        != training_provenance_task_id
    ):
        raise FormalMultiseedExecutionError(
            "formal plan training provenance task cross-binding mismatch"
        )
    if selector_artifact.get("training_progress_seal_sha256") != progress_seal:
        raise FormalMultiseedExecutionError(
            "formal plan training progress seal cross-binding mismatch"
        )
    if selector_artifact.get("training_provenance_seal_sha256") != provenance_seal:
        raise FormalMultiseedExecutionError(
            "formal plan training provenance seal cross-binding mismatch"
        )
    source_equivalence = _frozen_mapping(
        selector_artifact.get("source_revision_equivalence"),
        context="formal plan selector source revision equivalence",
    )
    source_equivalence_seal = _require_seal(
        source_equivalence,
        context="formal plan selector source revision equivalence",
    )
    if (
        source_equivalence_seal != SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        or selector_artifact.get("source_revision_equivalence_seal_sha256")
        != source_equivalence_seal
    ):
        raise FormalMultiseedExecutionError(
            "formal plan source revision equivalence cross-binding mismatch"
        )
    source_subject_map = _frozen_mapping(
        selector_artifact.get("source_revision_subject_map"),
        context="formal plan selector source revision subject map",
    )
    source_subject_map_seal = _require_seal(
        source_subject_map,
        context="formal plan selector source revision subject map",
    )
    evaluation_source_tree = _sha256(
        selector_artifact.get("evaluation_source_revision_tree_sha256"),
        "formal plan selector evaluation source revision tree",
    )
    if (
        source_subject_map_seal != SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        or selector_artifact.get("source_revision_subject_map_seal_sha256")
        != source_subject_map_seal
        or evaluation_source_tree != EVALUATION_SOURCE_REVISION_TREE_SHA256
        or not isinstance(
            selector_artifact.get("evaluation_source_revision"), Mapping
        )
        or selector_artifact["evaluation_source_revision"].get("tree_sha256")
        != evaluation_source_tree
    ):
        raise FormalMultiseedExecutionError(
            "formal plan source revision subject/evaluation binding mismatch"
        )
    selector_equivalence = _validate_training_script_equivalence(
        selector_artifact.get("training_script_equivalence"),
        context="formal plan selector training script equivalence",
    )
    if _canonical_json(selector_equivalence) != _canonical_json(
        training_script_equivalence
    ):
        raise FormalMultiseedExecutionError(
            "formal plan training script equivalence cross-binding mismatch"
        )
    selector_evaluation_script_sha256 = _sha256(
        selector_artifact.get("evaluation_script_sha256"),
        "formal plan selector evaluation script",
    )
    if selector_evaluation_script_sha256 != evaluation_script_sha256:
        raise FormalMultiseedExecutionError(
            "formal plan evaluation script cross-binding mismatch"
        )
    return {
        "plan_schema_version": 2,
        "formal_selector_task_id": selector_task_id,
        "training_provenance_task_id": training_provenance_task_id,
        "training_progress_artifact_seal_sha256": progress_seal,
        "training_provenance_artifact_seal_sha256": provenance_seal,
        "source_revision_equivalence_seal_sha256": source_equivalence_seal,
        "source_revision_subject_map_seal_sha256": source_subject_map_seal,
        "evaluation_source_revision_tree_sha256": evaluation_source_tree,
        "training_script_equivalence": training_script_equivalence,
        "evaluation_script_sha256": evaluation_script_sha256,
        "formal_selector_dependency_task_ids": selector_dependency_task_ids,
    }


def _finite_ap(value: object, *, context: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise FormalMultiseedExecutionError(f"{context} must be finite numeric AP")
    result = float(value)
    if not 0.0 <= result <= 100.0:
        raise FormalMultiseedExecutionError(f"{context} must be inside [0, 100]")
    return result


def _finite_metric_float(value: object, *, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise FormalMultiseedExecutionError(
            f"{context} must be a finite producer-native float"
        )
    return value


def _nonnegative_metric_float(value: object, *, context: str) -> float:
    result = _finite_metric_float(value, context=context)
    if result < 0.0:
        raise FormalMultiseedExecutionError(f"{context} must be nonnegative")
    return result


def _unit_metric_float(value: object, *, context: str) -> float:
    result = _finite_metric_float(value, context=context)
    if not 0.0 <= result <= 1.0:
        raise FormalMultiseedExecutionError(f"{context} must be inside [0, 1]")
    return result


def _count_metric_float(
    value: object,
    *,
    context: str,
    expected: int | None = None,
    maximum: int | None = None,
) -> int:
    result = _nonnegative_metric_float(value, context=context)
    if not result.is_integer():
        raise FormalMultiseedExecutionError(f"{context} must be integer-valued")
    integer = int(result)
    if expected is not None and integer != expected:
        raise FormalMultiseedExecutionError(
            f"{context} must be exactly {expected}, got {integer}"
        )
    if maximum is not None and integer > maximum:
        raise FormalMultiseedExecutionError(
            f"{context} must not exceed {maximum}, got {integer}"
        )
    return integer


def _task_parent(task: object, *, context: str) -> str:
    value = getattr(getattr(task, "data", None), "parent", None)
    if value is None:
        return ""
    if type(value) is not str:
        raise FormalMultiseedExecutionError(
            f"{context} parent must be a string, empty string, or None"
        )
    if value:
        _clearml_id(value, f"{context} parent")
    return value


def _status(task: object, *, context: str) -> str:
    value = getattr(getattr(task, "data", None), "status", None)
    value = getattr(value, "value", value)
    allowed = WAITING_STATUSES | FAILED_STATUSES | {"completed"}
    if type(value) is not str or value not in allowed:
        raise FormalMultiseedExecutionError(f"{context} status is not a string")
    return value


def _reload(task: object, *, context: str) -> None:
    expected_task_id = _clearml_id(getattr(task, "id", None), f"{context} local task")
    if bool(getattr(task, "_offline_mode", False)):
        raise FormalMultiseedExecutionError(f"{context} cannot use offline reload")
    reloader = getattr(task, "_reload", None)
    if not callable(reloader):
        raise FormalMultiseedExecutionError(f"{context} cannot be server-reloaded")
    has_skip_flag = hasattr(task, "_reload_skip_flag")
    previous_skip_flag = getattr(task, "_reload_skip_flag", None)
    try:
        if has_skip_flag:
            setattr(task, "_reload_skip_flag", False)
        snapshot = reloader()
    except Exception as error:
        raise FormalMultiseedExecutionError(
            f"{context} server reload failed"
        ) from error
    finally:
        if has_skip_flag:
            setattr(task, "_reload_skip_flag", previous_skip_flag)
    if snapshot is None or isinstance(
        snapshot, (bool, int, float, str, bytes, bytearray)
    ):
        raise FormalMultiseedExecutionError(
            f"{context} server reload returned no snapshot"
        )
    snapshot_task_id = _clearml_id(
        getattr(snapshot, "id", None), f"{context} server snapshot"
    )
    if snapshot_task_id != expected_task_id:
        raise FormalMultiseedExecutionError(
            f"{context} server snapshot identity mismatch"
        )
    try:
        setattr(task, "_data", snapshot)
    except Exception as error:
        raise FormalMultiseedExecutionError(
            f"{context} server snapshot cannot be installed"
        ) from error
    if (
        getattr(task, "_data", None) is not snapshot
        or getattr(task, "data", None) is not snapshot
    ):
        raise FormalMultiseedExecutionError(
            f"{context} server snapshot was not installed exactly"
        )


def _parameters(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise FormalMultiseedExecutionError(f"{context} cannot enumerate Args")
    try:
        value = getter(backwards_compatibility=False, cast=False)
    except TypeError:
        try:
            value = getter(cast=False)
        except TypeError:
            value = getter()
    return _frozen_mapping(value, context=f"{context} parameters")


def _script(task: object, *, context: str) -> dict[str, object]:
    script = getattr(getattr(task, "data", None), "script", None)
    if script is None:
        raise FormalMultiseedExecutionError(
            f"{context} installed server snapshot has no script metadata"
        )
    raw: object = {
        "repository": getattr(script, "repository", None),
        "working_dir": getattr(script, "working_dir", None),
        "entry_point": getattr(script, "entry_point", None),
        "diff": getattr(script, "diff", None),
    }
    result = _frozen_mapping(raw, context=f"{context} script")
    _require_exact_keys(
        result,
        {"repository", "working_dir", "entry_point", "diff"},
        context=f"{context} script",
    )
    return result


def _require_no_repo_script(
    task: object,
    *,
    entry_point: str,
    expected_sha256: str,
    context: str,
    expected_source: str | None = None,
) -> dict[str, object]:
    script = _script(task, context=context)
    if script["repository"] != "":
        raise FormalMultiseedExecutionError(f"{context} is not a no-repo task")
    if script["working_dir"] != ".":
        raise FormalMultiseedExecutionError(f"{context} working directory drifted")
    if script["entry_point"] != entry_point:
        raise FormalMultiseedExecutionError(f"{context} entry point drifted")
    source = script["diff"]
    if type(source) is not str or not source:
        raise FormalMultiseedExecutionError(f"{context} script diff is not text")
    observed = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if observed != _sha256(expected_sha256, f"{context} expected script"):
        raise FormalMultiseedExecutionError(f"{context} script SHA-256 mismatch")
    if expected_source is not None and source != expected_source:
        raise FormalMultiseedExecutionError(f"{context} script bytes drifted")
    return {
        "repository": "",
        "working_dir": ".",
        "entry_point": entry_point,
        "sha256": observed,
        "size_bytes": len(source.encode("utf-8")),
    }


def _queue_id(task: object, *, context: str) -> str:
    value = getattr(
        getattr(getattr(task, "data", None), "execution", None),
        "queue",
        None,
    )
    if value is None or value == "":
        return ""
    return _clearml_id(value, f"{context} execution queue")


def _last_worker(task: object, *, context: str) -> str:
    value = getattr(getattr(task, "data", None), "last_worker", None)
    if type(value) is not str or not value.strip():
        raise FormalMultiseedExecutionError(
            f"{context} has no auditable last_worker identity"
        )
    return value.strip()


def _server_artifacts(task: object, *, context: str) -> dict[str, object]:
    data = getattr(task, "data", None)
    execution = getattr(data, "execution", None)
    raw_artifacts = getattr(execution, "artifacts", None)
    if raw_artifacts is None:
        raw_artifacts = ()
    if type(raw_artifacts) not in {list, tuple}:
        raise FormalMultiseedExecutionError(
            f"{context} raw server artifacts are invalid"
        )
    if ClearMLArtifact is None:
        raise FormalMultiseedExecutionError(
            f"{context} ClearML artifact reader is unavailable"
        )
    result: dict[str, object] = {}
    for raw in raw_artifacts:
        name = getattr(raw, "key", None)
        if type(name) is not str or not name or name in result:
            raise FormalMultiseedExecutionError(
                f"{context} raw server artifact inventory is invalid"
            )
        try:
            result[name] = ClearMLArtifact(raw)
        except Exception as error:
            raise FormalMultiseedExecutionError(
                f"{context} raw server artifact {name!r} cannot be materialized"
            ) from error
    return result


def _artifact_names(task: object, *, context: str) -> tuple[str, ...]:
    return tuple(sorted(_server_artifacts(task, context=context)))


def _read_json_artifact_path(
    value: str | Path,
    *,
    context: str,
) -> dict[str, object]:
    path = Path(value)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise FormalMultiseedExecutionError(
            f"{context} secure local artifact open is unavailable"
        )
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        path_before = path.lstat()
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(path_before.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or path_before.st_nlink != 1
                or opened.st_nlink != 1
                or path_before.st_dev != opened.st_dev
                or path_before.st_ino != opened.st_ino
                or opened.st_size <= 0
                or opened.st_size > JSON_ARTIFACT_MAX_BYTES
            ):
                raise FormalMultiseedExecutionError(
                    f"{context} local artifact path is unsafe"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, JSON_ARTIFACT_MAX_BYTES + 1 - total),
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > JSON_ARTIFACT_MAX_BYTES:
                    raise FormalMultiseedExecutionError(
                        f"{context} local artifact exceeded its byte cap"
                    )
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = path.lstat()
    except FormalMultiseedExecutionError:
        raise
    except (OSError, ValueError) as error:
        raise FormalMultiseedExecutionError(
            f"{context} local artifact cannot be securely read"
        ) from error

    def identity(snapshot: os.stat_result) -> tuple[int, ...]:
        return (
            snapshot.st_dev,
            snapshot.st_ino,
            snapshot.st_mode,
            snapshot.st_nlink,
            snapshot.st_size,
            snapshot.st_mtime_ns,
            snapshot.st_ctime_ns,
        )

    if (
        identity(opened) != identity(after)
        or identity(path_before) != identity(opened)
        or identity(path_after) != identity(after)
        or total != opened.st_size
    ):
        raise FormalMultiseedExecutionError(
            f"{context} local artifact changed during read"
        )
    try:
        document = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FormalMultiseedExecutionError(
            f"{context} local artifact is not UTF-8 JSON"
        ) from error
    return _frozen_mapping(document, context=context)


def _artifact_mapping(task: object, name: str, *, context: str) -> dict[str, object]:
    artifacts = _server_artifacts(task, context=context)
    if name not in artifacts:
        raise FormalMultiseedExecutionError(
            f"{context} lacks required artifact {name!r}"
        )
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise FormalMultiseedExecutionError(
            f"{context} artifact {name!r} cannot be read"
        )
    try:
        value = getter(force_download=True)
    except Exception as error:
        raise FormalMultiseedExecutionError(
            f"{context} artifact {name!r} cannot be freshly downloaded"
        ) from error
    if isinstance(value, Mapping):
        return _frozen_mapping(value, context=f"{context} artifact {name!r}")
    if not isinstance(value, (str, Path)):
        raise FormalMultiseedExecutionError(
            f"{context} artifact {name!r} is not a JSON object"
        )
    return _read_json_artifact_path(
        value,
        context=f"{context} artifact {name!r}",
    )


def _file_sha256(path: Path, *, context: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise FormalMultiseedExecutionError(
            f"{context} must be a regular non-symlink file"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise FormalMultiseedExecutionError(f"{context} cannot be hashed") from error
    return digest.hexdigest()


def _read_json_mapping_file(path: Path, *, context: str) -> dict[str, object]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise FormalMultiseedExecutionError(f"{context} cannot be inspected") from error
    if size <= 0 or size > 512 * 1024 * 1024:
        raise FormalMultiseedExecutionError(f"{context} has an invalid size")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalMultiseedExecutionError(
            f"{context} is not readable JSON"
        ) from error
    return _frozen_mapping(value, context=context)


def _canonical_absolute_posix_path(value: object, *, context: str) -> PurePosixPath:
    if type(value) is not str or not value:
        raise FormalMultiseedExecutionError(f"{context} path is invalid")
    path = PurePosixPath(value)
    if not path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise FormalMultiseedExecutionError(
            f"{context} must be a canonical absolute POSIX path"
        )
    return path


def _safe_evidence_file(
    root: Path,
    relative: PurePosixPath,
    *,
    context: str,
) -> Path:
    if (
        relative.is_absolute()
        or relative.as_posix() in {"", "."}
        or ".." in relative.parts
    ):
        raise FormalMultiseedExecutionError(f"{context} relative path is invalid")
    cursor = root
    try:
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise FormalMultiseedExecutionError(
                    f"{context} traverses a symbolic link"
                )
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except FormalMultiseedExecutionError:
        raise
    except (OSError, ValueError) as error:
        raise FormalMultiseedExecutionError(
            f"{context} escaped or is missing from evidence"
        ) from error
    if not resolved.is_file():
        raise FormalMultiseedExecutionError(f"{context} is not a regular file")
    return resolved


def _preflight_evaluation_evidence_zip(
    archive: BinaryIO,
    *,
    archive_size: int,
    expected_members: set[str],
    context: str,
) -> None:
    eocd_size = 22
    if archive_size < eocd_size:
        raise FormalMultiseedExecutionError(f"{context} evidence ZIP EOCD is missing")
    try:
        archive.seek(archive_size - eocd_size)
        eocd = archive.read(eocd_size)
        (
            signature,
            disk_number,
            central_directory_disk,
            entries_on_disk,
            entry_count,
            central_directory_size,
            central_directory_offset,
            comment_size,
        ) = struct.unpack("<4s4H2LH", eocd)
    except (OSError, struct.error) as error:
        raise FormalMultiseedExecutionError(
            f"{context} evidence ZIP EOCD is unreadable"
        ) from error
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or central_directory_disk != 0
        or entries_on_disk != len(expected_members)
        or entry_count != len(expected_members)
        or comment_size != 0
        or central_directory_size <= 0
        or central_directory_size > EVALUATION_EVIDENCE_MAX_CENTRAL_DIRECTORY_BYTES
        or central_directory_offset in {0xFFFFFFFF}
        or central_directory_size in {0xFFFFFFFF}
        or entries_on_disk == 0xFFFF
        or entry_count == 0xFFFF
        or central_directory_offset + central_directory_size != archive_size - eocd_size
    ):
        raise FormalMultiseedExecutionError(
            f"{context} evidence ZIP EOCD/ZIP64 contract drifted"
        )

    try:
        archive.seek(central_directory_offset)
        central_directory = archive.read(central_directory_size)
    except OSError as error:
        raise FormalMultiseedExecutionError(
            f"{context} evidence ZIP central directory is unreadable"
        ) from error
    if len(central_directory) != central_directory_size:
        raise FormalMultiseedExecutionError(
            f"{context} evidence ZIP central directory is truncated"
        )

    expected_names = {name.encode("ascii") for name in expected_members}
    observed_names: list[bytes] = []
    observed_offsets: set[int] = set()
    total_size = 0
    cursor = 0
    for index in range(entry_count):
        if cursor + 46 > len(central_directory):
            raise FormalMultiseedExecutionError(
                f"{context} evidence ZIP central entry {index} is truncated"
            )
        try:
            (
                entry_signature,
                version_made_by,
                _version_needed,
                flag_bits,
                compression,
                _modified_time,
                _modified_date,
                _crc32,
                compressed_size,
                file_size,
                filename_size,
                extra_size,
                entry_comment_size,
                entry_disk,
                _internal_attributes,
                external_attributes,
                local_header_offset,
            ) = struct.unpack_from(
                "<4s6H3L5H2L",
                central_directory,
                cursor,
            )
        except struct.error as error:
            raise FormalMultiseedExecutionError(
                f"{context} evidence ZIP central entry {index} is invalid"
            ) from error
        entry_end = cursor + 46 + filename_size + extra_size + entry_comment_size
        if entry_end > len(central_directory):
            raise FormalMultiseedExecutionError(
                f"{context} evidence ZIP central entry {index} overflows"
            )
        raw_name = central_directory[cursor + 46 : cursor + 46 + filename_size]
        unix_mode = external_attributes >> 16
        if (
            entry_signature != b"PK\x01\x02"
            or version_made_by >> 8 != 3
            or flag_bits & 1
            or compression not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or compressed_size <= 0
            or file_size <= 0
            or file_size > EVALUATION_EVIDENCE_MAX_MEMBER_BYTES
            or file_size / compressed_size > EVALUATION_EVIDENCE_MAX_COMPRESSION_RATIO
            or not 0 < filename_size <= 255
            or extra_size != 0
            or entry_comment_size != 0
            or entry_disk != 0
            or not stat.S_ISREG(unix_mode)
            or local_header_offset >= central_directory_offset
            or local_header_offset in observed_offsets
            or raw_name not in expected_names
            or b"\x00" in raw_name
            or b"\\" in raw_name
        ):
            raise FormalMultiseedExecutionError(
                f"{context} evidence ZIP central entry {index} is unsafe"
            )
        observed_names.append(raw_name)
        observed_offsets.add(local_header_offset)
        total_size += file_size
        cursor = entry_end

    if (
        cursor != len(central_directory)
        or len(set(observed_names)) != len(observed_names)
        or set(observed_names) != expected_names
        or total_size > EVALUATION_EVIDENCE_MAX_ARCHIVE_BYTES
    ):
        raise FormalMultiseedExecutionError(
            f"{context} evidence ZIP central inventory drifted"
        )
    archive.seek(0)


def _snapshot_evaluation_evidence_archive(
    archive: Path,
    snapshot: BinaryIO,
    *,
    context: str,
) -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise FormalMultiseedExecutionError(
            f"{context} secure evidence archive open is unavailable"
        )
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(archive, flags)
    except OSError as error:
        raise FormalMultiseedExecutionError(
            f"{context} evidence archive cannot be securely opened"
        ) from error

    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            before = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size <= 0
                or before.st_size > EVALUATION_EVIDENCE_MAX_ARCHIVE_BYTES
            ):
                raise FormalMultiseedExecutionError(
                    f"{context} evidence archive size/type drifted"
                )
            copied = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > EVALUATION_EVIDENCE_MAX_ARCHIVE_BYTES:
                    raise FormalMultiseedExecutionError(
                        f"{context} evidence archive exceeded its byte cap"
                    )
                snapshot.write(chunk)
            after = os.fstat(source.fileno())
    except FormalMultiseedExecutionError:
        raise
    except OSError as error:
        raise FormalMultiseedExecutionError(
            f"{context} evidence archive snapshot failed"
        ) from error

    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        before_identity != after_identity
        or copied != before.st_size
        or snapshot.tell() != copied
    ):
        raise FormalMultiseedExecutionError(
            f"{context} evidence archive changed during snapshot"
        )
    snapshot.flush()
    snapshot.seek(0)
    return copied


@contextlib.contextmanager
def _evaluation_evidence_root(task: object, *, context: str) -> Iterator[Path]:
    artifacts = _server_artifacts(task, context=context)
    if EVALUATION_EVIDENCE_ARTIFACT not in artifacts:
        raise FormalMultiseedExecutionError(
            f"{context} lacks required artifact {EVALUATION_EVIDENCE_ARTIFACT!r}"
        )
    artifact = artifacts[EVALUATION_EVIDENCE_ARTIFACT]
    if getattr(artifact, "type", None) != "archive":
        raise FormalMultiseedExecutionError(
            f"{context} evidence artifact type must be archive"
        )
    artifact_url = _require_fileserver_url(
        getattr(artifact, "url", None),
        context=f"{context} evidence artifact",
    )
    artifact_path = PurePosixPath(unquote(urlsplit(artifact_url).path))
    if artifact_path.suffix.lower() != ".zip":
        raise FormalMultiseedExecutionError(
            f"{context} evidence artifact URL must identify a ZIP"
        )
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        raise FormalMultiseedExecutionError(
            f"{context} evidence artifact cannot be downloaded"
        )
    try:
        value = getter(
            extract_archive=False,
            raise_on_error=True,
            force_download=True,
        )
    except Exception as error:
        raise FormalMultiseedExecutionError(
            f"{context} evidence artifact download failed"
        ) from error
    if not isinstance(value, (str, Path)) or not str(value):
        raise FormalMultiseedExecutionError(
            f"{context} evidence artifact returned no archive"
        )
    archive = Path(value)

    expected_members = {"evaluation_plan.json", "metrics.json"}
    for delay in DELAYS_MS:
        for condition in CONDITIONS:
            condition_id = f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
            expected_members.update(
                {
                    f"{condition_id}/resolved_config.py",
                    f"{condition_id}/checkpoint.sha256",
                    f"{condition_id}/predictions.json",
                }
            )

    try:
        with tempfile.TemporaryDirectory(prefix="resilient-v2x-evidence-") as temporary:
            temporary_root = Path(temporary).resolve(strict=True)
            snapshot_path = temporary_root / "evidence.zip"
            extracted = temporary_root / "extracted"
            extracted.mkdir(mode=0o700)
            with snapshot_path.open("x+b") as snapshot:
                archive_size = _snapshot_evaluation_evidence_archive(
                    archive,
                    snapshot,
                    context=context,
                )
                _preflight_evaluation_evidence_zip(
                    snapshot,
                    archive_size=archive_size,
                    expected_members=expected_members,
                    context=context,
                )
                with zipfile.ZipFile(snapshot, "r") as source:
                    if source.comment != b"":
                        raise FormalMultiseedExecutionError(
                            f"{context} evidence ZIP comment is forbidden"
                        )
                    members = source.infolist()
                    observed_names: list[str] = []
                    observed_offsets: set[int] = set()
                    total_size = 0
                    for member in members:
                        name = member.filename
                        original_name = member.orig_filename
                        relative = PurePosixPath(name)
                        unix_mode = member.external_attr >> 16
                        if (
                            type(name) is not str
                            or type(original_name) is not str
                            or not name
                            or original_name != name
                            or "\x00" in original_name
                            or "\\" in original_name
                            or relative.is_absolute()
                            or relative.as_posix() != name
                            or ".." in relative.parts
                            or member.is_dir()
                            or member.flag_bits & 1
                            or member.compress_type
                            not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                            or not stat.S_ISREG(unix_mode)
                            or member.file_size <= 0
                            or member.file_size > EVALUATION_EVIDENCE_MAX_MEMBER_BYTES
                            or member.compress_size <= 0
                            or member.file_size / member.compress_size
                            > EVALUATION_EVIDENCE_MAX_COMPRESSION_RATIO
                            or member.extra
                            or member.comment
                            or member.header_offset in observed_offsets
                        ):
                            raise FormalMultiseedExecutionError(
                                f"{context} evidence ZIP member is unsafe"
                            )
                        observed_names.append(name)
                        observed_offsets.add(member.header_offset)
                        total_size += member.file_size
                    if (
                        len(observed_names) != len(expected_members)
                        or len(set(observed_names)) != len(observed_names)
                        or set(observed_names) != expected_members
                        or total_size > EVALUATION_EVIDENCE_MAX_ARCHIVE_BYTES
                    ):
                        raise FormalMultiseedExecutionError(
                            f"{context} evidence ZIP inventory drifted"
                        )

                    actual_total = 0
                    for member in members:
                        target = extracted.joinpath(
                            *PurePosixPath(member.filename).parts
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        written = 0
                        with (
                            source.open(member, "r") as reader,
                            target.open("xb") as writer,
                        ):
                            while True:
                                chunk = reader.read(1024 * 1024)
                                if not chunk:
                                    break
                                written += len(chunk)
                                actual_total += len(chunk)
                                if (
                                    written > member.file_size
                                    or actual_total
                                    > EVALUATION_EVIDENCE_MAX_ARCHIVE_BYTES
                                ):
                                    raise FormalMultiseedExecutionError(
                                        f"{context} evidence ZIP extraction "
                                        "exceeded its byte cap"
                                    )
                                writer.write(chunk)
                        target_stat = target.lstat()
                        if (
                            written != member.file_size
                            or not stat.S_ISREG(target_stat.st_mode)
                            or target_stat.st_size != member.file_size
                        ):
                            raise FormalMultiseedExecutionError(
                                f"{context} evidence ZIP extraction drifted"
                            )
            yield extracted
    except FormalMultiseedExecutionError:
        raise
    except (
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as error:
        raise FormalMultiseedExecutionError(
            f"{context} evidence ZIP cannot be safely extracted"
        ) from error


def _evidence_file(
    evidence_root: Path,
    original_root: object,
    original_path: object,
    *,
    context: str,
) -> Path:
    root = _canonical_absolute_posix_path(original_root, context=f"{context} root")
    path = _canonical_absolute_posix_path(original_path, context=context)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise FormalMultiseedExecutionError(
            f"{context} is outside the sealed work directory"
        ) from error
    return _safe_evidence_file(evidence_root, relative, context=context)


def _artifact_inventory(task: object, *, context: str) -> dict[str, str]:
    artifacts = _server_artifacts(task, context=context)
    result: dict[str, str] = {}
    for name in sorted(artifacts):
        if type(name) is not str:
            raise FormalMultiseedExecutionError(
                f"{context} artifact names must be strings"
            )
        result[name] = _content_sha256(_artifact_mapping(task, name, context=context))
    return result


def _wait_for_completed(
    task: object,
    *,
    context: str,
    deadline: float,
    poll_seconds: float,
    monotonic_clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    while True:
        _reload(task, context=context)
        status = _status(task, context=context)
        if status == "completed":
            return
        if status in FAILED_STATUSES:
            raise FormalMultiseedExecutionError(
                f"{context} ended without completion: {status!r}"
            )
        if status not in WAITING_STATUSES:
            raise FormalMultiseedExecutionError(
                f"{context} has unexpected status {status!r}"
            )
        if monotonic_clock() >= deadline:
            raise TimeoutError(f"timed out waiting for {context}")
        sleeper(poll_seconds)


def _plan_validator_bytes(path: Path | None = None) -> bytes:
    if _EMBEDDED_PLAN_VALIDATOR_B64:
        try:
            source = base64.b64decode(
                _EMBEDDED_PLAN_VALIDATOR_B64.encode("ascii"), validate=True
            )
        except (ValueError, UnicodeError) as error:
            raise FormalMultiseedExecutionError(
                "embedded plan validator is invalid base64"
            ) from error
    else:
        source_path = path or Path(__file__).with_name("formal_multiseed_plan.py")
        try:
            source = source_path.read_bytes()
        except OSError as error:
            raise FormalMultiseedExecutionError(
                "cannot read the pinned plan validator"
            ) from error
    if hashlib.sha256(source).hexdigest() != PINNED_PLAN_VALIDATOR_SHA256:
        raise FormalMultiseedExecutionError(
            "pinned plan validator byte SHA-256 mismatch"
        )
    try:
        source.decode("utf-8")
    except UnicodeError as error:
        raise FormalMultiseedExecutionError(
            "pinned plan validator is not UTF-8"
        ) from error
    return source


def _load_plan_validator(path: Path | None = None) -> ModuleType:
    source = _plan_validator_bytes(path)
    module_name = "_resilient_v2x_formal_multiseed_plan_pinned"
    module = ModuleType(module_name)
    module.__file__ = "<pinned-formal-multiseed-plan.py>"
    sys.modules[module_name] = module
    try:
        exec(compile(source.decode("utf-8"), module.__file__, "exec"), module.__dict__)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise FormalMultiseedExecutionError(
            "cannot execute the pinned plan validator"
        ) from error
    if not callable(getattr(module, "validate_plan", None)):
        raise FormalMultiseedExecutionError("pinned plan validator lacks validate_plan")
    return module


def generate_standalone_source(
    *,
    plan_validator_path: Path | None = None,
    wrapper_path: Path | None = None,
) -> str:
    """Return this executor with the exact reviewed plan validator embedded."""

    plan_source = _plan_validator_bytes(plan_validator_path)
    path = wrapper_path or Path(__file__)
    try:
        wrapper = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FormalMultiseedExecutionError(
            "cannot read the executor wrapper"
        ) from error
    anchor = "".join(
        (
            "_EMBEDDED_PLAN_VALIDATOR_B64",
            ' = ""  # __FORMAL_MULTISEED_PLAN_BYTES_V2__',
        )
    )
    if wrapper.count(anchor) != 1:
        raise FormalMultiseedExecutionError(
            "standalone plan-validator anchor count mismatch"
        )
    encoded = base64.b64encode(plan_source).decode("ascii")
    replacement = (
        f"_EMBEDDED_PLAN_VALIDATOR_B64 = {encoded!r} "
        "# __FORMAL_MULTISEED_PLAN_BYTES_V2__"
    )
    standalone = wrapper.replace(anchor, replacement, 1)
    try:
        compile(standalone, "<clearml-formal-multiseed-executor>", "exec")
    except SyntaxError as error:
        raise FormalMultiseedExecutionError(
            "generated standalone executor does not compile"
        ) from error
    return standalone


def _runtime_source() -> str:
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FormalMultiseedExecutionError(
            "cannot read the running executor script"
        ) from error
    if not source:
        raise FormalMultiseedExecutionError("running executor script is empty")
    return source


def _task_fingerprint(
    task: object,
    *,
    context: str,
    expected_id: str,
    entry_point: str,
    script_sha256: str,
    artifact_names: Sequence[str],
) -> dict[str, object]:
    task_id = _clearml_id(getattr(task, "id", None), context)
    if task_id != expected_id:
        raise FormalMultiseedExecutionError(f"{context} identity mismatch")
    status = _status(task, context=context)
    if status != "completed":
        raise FormalMultiseedExecutionError(f"{context} is not completed")
    script = _require_no_repo_script(
        task,
        entry_point=entry_point,
        expected_sha256=script_sha256,
        context=context,
    )
    expected_artifact_names = tuple(sorted(artifact_names))
    if _artifact_names(task, context=context) != expected_artifact_names:
        raise FormalMultiseedExecutionError(
            f"{context} exact artifact inventory drifted"
        )
    artifacts = {
        name: _content_sha256(_artifact_mapping(task, name, context=context))
        for name in expected_artifact_names
    }
    return {
        "task_id": task_id,
        "parent_task_id": _task_parent(task, context=context),
        "status": status,
        "script": script,
        "parameters": _parameters(task, context=context),
        "artifacts": artifacts,
    }


def _validate_planner_receipt(
    value: Mapping[str, object],
    *,
    planner_task_id: str,
    planner_parent_task_id: str,
    producer_sha256: str,
    plan: Mapping[str, object],
) -> dict[str, object]:
    receipt = _frozen_mapping(value, context="planner receipt")
    _require_exact_keys(
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
    expected = {
        "receipt_type": "resilient_v2x_formal_multiseed_plan_receipt",
        "planner_task_id": planner_task_id,
        "planner_parent_task_id": planner_parent_task_id,
        "producer_entry_point": PLANNER_ENTRY_POINT,
        "producer_script_sha256": producer_sha256,
        "plan_artifact_name": PLAN_ARTIFACT,
        "plan_seal_sha256": plan["seal_sha256"],
        "plan_canonical_sha256": _content_sha256(plan),
    }
    for key, expected_value in expected.items():
        if (
            type(receipt.get(key)) is not type(expected_value)
            or receipt.get(key) != expected_value
        ):
            raise FormalMultiseedExecutionError(f"planner receipt {key} mismatch")
    _require_seal(receipt, context="planner receipt")
    return receipt


def _validate_planner_parameters(planner: object) -> dict[str, object]:
    observed = _parameters(planner, context="formal multi-seed planner")
    expected = _frozen_mapping(
        PLANNER_PARAMETER_CONTRACT,
        context="expected formal multi-seed planner parameters",
    )
    if _canonical_json(observed) != _canonical_json(expected):
        raise FormalMultiseedExecutionError(
            "formal multi-seed planner parameter contract mismatch"
        )
    return observed


def _resolve_plan(
    planner: object,
    *,
    planner_task_id: str,
    producer_sha256: str,
    validator_path: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    _require_no_repo_script(
        planner,
        entry_point=PLANNER_ENTRY_POINT,
        expected_sha256=producer_sha256,
        context="formal multi-seed planner",
    )
    _validate_planner_parameters(planner)
    raw_plan = _artifact_mapping(planner, PLAN_ARTIFACT, context="planner")
    validator = _load_plan_validator(validator_path)
    try:
        plan = validator.validate_plan(raw_plan)
    except Exception as error:
        raise FormalMultiseedExecutionError(
            "planner formal_multiseed_plan failed pinned validation"
        ) from error
    plan = _frozen_mapping(plan, context="validated formal multi-seed plan")
    _formal_plan_provenance(plan)
    for key, expected in {
        "candidate_count": 1,
        "training_task_count": 45,
        "evaluation_task_count": 45,
        "total_evaluation_condition_count": 540,
    }.items():
        _exact_int(plan.get(key), expected, context=f"formal plan {key}")
    planner_parent = _task_parent(planner, context="formal multi-seed planner")
    selector_provenance = plan.get("formal_selector_provenance")
    if not isinstance(selector_provenance, Mapping):
        raise FormalMultiseedExecutionError(
            "formal plan selector provenance is invalid"
        )
    selector_task_id = _clearml_id(
        selector_provenance.get("task_id"),
        "formal plan selector task",
    )
    if planner_parent != selector_task_id:
        raise FormalMultiseedExecutionError(
            "formal multi-seed planner parent does not match the sealed selector"
        )
    receipt = _validate_planner_receipt(
        _artifact_mapping(
            planner, PLANNER_RECEIPT_ARTIFACT, context="formal multi-seed planner"
        ),
        planner_task_id=planner_task_id,
        planner_parent_task_id=planner_parent,
        producer_sha256=producer_sha256,
        plan=plan,
    )
    return plan, receipt


def _validate_source_d_evidence(
    task: object,
    *,
    task_id: str,
    plan: Mapping[str, object],
) -> dict[str, object]:
    context = "Source-D evidence task"
    expected_task_id = _clearml_id(task_id, "expected Source-D evidence task")
    if _clearml_id(getattr(task, "id", None), context) != expected_task_id:
        raise FormalMultiseedExecutionError("Source-D evidence task identity mismatch")
    _require_no_repo_script(
        task,
        entry_point=SOURCE_D_PRODUCER_ENTRY_POINT,
        expected_sha256=EXPECTED_SOURCE_D_PRODUCER_SHA256,
        context=context,
    )
    if _task_parent(task, context=context) != SOURCE_C_TASK_ID:
        raise FormalMultiseedExecutionError("Source-D evidence parent mismatch")
    if _artifact_names(task, context=context) != tuple(sorted(SOURCE_D_ARTIFACT_ORDER)):
        raise FormalMultiseedExecutionError(
            "Source-D evidence artifact inventory drifted"
        )
    source_c = _artifact_mapping(task, SOURCE_C_SNAPSHOT_ARTIFACT, context=context)
    source_d = _artifact_mapping(task, SOURCE_D_SCRIPT_ARTIFACT, context=context)
    equivalence = _artifact_mapping(
        task, SOURCE_D_EQUIVALENCE_ARTIFACT, context=context
    )
    receipt = _artifact_mapping(task, SOURCE_D_RECEIPT_ARTIFACT, context=context)
    _require_exact_keys(
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
    source_c_seal = _require_seal(source_c, context="Source-C snapshot")
    _require_exact_keys(
        source_d,
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
    source_d_seal = _require_seal(source_d, context="Source-D script")
    equivalence_sha = _require_content_hash(
        equivalence,
        field="artifact_sha256",
        context="Source-D equivalence",
    )
    if equivalence.get("transformation_id") != SOURCE_D_TRANSFORMATION_ID:
        raise FormalMultiseedExecutionError(
            "Source-D equivalence transformation identity drifted"
        )
    diff = equivalence.get("diff")
    if (
        not isinstance(diff, list)
        or len(diff) != SOURCE_D_DECLARED_REPLACEMENT_COUNT
        or any(
            not isinstance(item, Mapping)
            or type(item.get("index")) is not int
            or item.get("index") != index
            or type(item.get("name")) is not str
            or type(item.get("expected_count")) is not int
            or item.get("expected_count") != 1
            or type(item.get("observed_count")) is not int
            or item.get("observed_count") != 1
            for index, item in enumerate(diff, start=1)
        )
    ):
        raise FormalMultiseedExecutionError(
            "Source-D equivalence replacement receipts drifted"
        )
    staging_names = tuple(
        item["name"] for item in diff if item["name"] in SOURCE_D_STAGING_ANCHOR_NAMES
    )
    if staging_names != SOURCE_D_STAGING_ANCHOR_NAMES:
        raise FormalMultiseedExecutionError(
            "Source-D equivalence staging receipts drifted"
        )
    _require_exact_keys(
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
    receipt_seal = _require_seal(receipt, context="Source-D receipt")
    _exact_bool(source_c.get("complete"), True, context="Source-C complete")
    _exact_bool(source_d.get("complete"), True, context="Source-D complete")
    _exact_bool(receipt.get("complete"), True, context="Source-D receipt complete")
    _exact_int(
        source_c.get("schema_version"),
        1,
        context="Source-C snapshot schema",
    )
    if (
        source_c.get("artifact_type") != "resilient_v2x_formal_source_c_snapshot"
        or source_c.get("source_c_task_id") != SOURCE_C_TASK_ID
        or source_c.get("source_c_task_status") != "completed"
    ):
        raise FormalMultiseedExecutionError("Source-C snapshot identity drifted")
    source_c_script = source_c.get("script_diff")
    if type(source_c_script) is not str or not source_c_script:
        raise FormalMultiseedExecutionError(
            "Source-C snapshot script bytes are missing"
        )
    source_c_sha = hashlib.sha256(source_c_script.encode("utf-8")).hexdigest()
    if source_c_sha != EXPECTED_SOURCE_C_SCRIPT_SHA256:
        raise FormalMultiseedExecutionError("Source-C snapshot script SHA-256 drifted")
    metadata = source_c.get("script")
    expected_metadata = {
        "repository": "",
        "working_dir": ".",
        "entry_point": SOURCE_C_ENTRY_POINT,
        "sha256": source_c_sha,
        "size_bytes": len(source_c_script.encode("utf-8")),
        "line_count": len(source_c_script.splitlines()),
    }
    if not isinstance(metadata, Mapping) or (
        _canonical_json(metadata) != _canonical_json(expected_metadata)
    ):
        raise FormalMultiseedExecutionError("Source-C snapshot script metadata drifted")
    source_script = source_d.get("script")
    if type(source_script) is not str or not source_script:
        raise FormalMultiseedExecutionError("Source-D script artifact has no text")
    script_sha = hashlib.sha256(source_script.encode("utf-8")).hexdigest()
    _exact_int(
        source_d.get("schema_version"),
        1,
        context="Source-D script schema",
    )
    if (
        source_d.get("artifact_type") != "resilient_v2x_formal_source_d_script"
        or source_d.get("transformation_id") != SOURCE_D_TRANSFORMATION_ID
        or source_d.get("source_c_sha256") != source_c_sha
        or source_d.get("source_d_sha256") != script_sha
        or source_d.get("size_bytes") != len(source_script.encode("utf-8"))
        or source_d.get("line_count") != len(source_script.splitlines())
    ):
        raise FormalMultiseedExecutionError("Source-D script semantics drifted")
    source_identity = plan.get("source_d_identity")
    if not isinstance(source_identity, Mapping):
        raise FormalMultiseedExecutionError("plan Source-D identity is missing")
    if (
        script_sha != source_identity.get("script_sha256")
        or script_sha != EXPECTED_SOURCE_D_SCRIPT_SHA256
        or equivalence_sha != source_identity.get("equivalence_artifact_sha256")
        or equivalence_sha != EXPECTED_SOURCE_D_EQUIVALENCE_SHA256
    ):
        raise FormalMultiseedExecutionError("plan and Source-D evidence disagree")
    seed_contract = equivalence.get("seed_contract")
    if not isinstance(seed_contract, Mapping):
        raise FormalMultiseedExecutionError("Source-D seed contract is missing")
    _exact_int(
        seed_contract.get("training_overlay_protocol_seed"),
        TRAINING_OVERLAY_PROTOCOL_SEED,
        context="Source-D overlay seed",
    )
    _exact_int(
        receipt.get("schema_version"),
        1,
        context="Source-D receipt schema",
    )
    if receipt.get(
        "artifact_type"
    ) != "resilient_v2x_formal_source_d_evidence_receipt" or receipt.get(
        "publication_order"
    ) != list(SOURCE_D_ARTIFACT_ORDER):
        raise FormalMultiseedExecutionError(
            "Source-D receipt identity or publication order drifted"
        )
    expected_hashes = {
        SOURCE_C_SNAPSHOT_ARTIFACT: source_c_seal,
        SOURCE_D_SCRIPT_ARTIFACT: source_d_seal,
        SOURCE_D_EQUIVALENCE_ARTIFACT: equivalence_sha,
    }
    artifact_hashes = receipt.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping) or _canonical_json(
        artifact_hashes
    ) != _canonical_json(expected_hashes):
        raise FormalMultiseedExecutionError("Source-D receipt artifact hashes drifted")
    provenance = receipt.get("provenance")
    transformation = receipt.get("transformation")
    if not isinstance(provenance, Mapping) or not isinstance(transformation, Mapping):
        raise FormalMultiseedExecutionError("Source-D receipt provenance is missing")
    expected_provenance = {
        "source_c_task_id": SOURCE_C_TASK_ID,
        "source_c_task_status": "completed",
        "source_c_task_parent": SOURCE_C_PARENT_TASK_ID,
        "source_c_entry_point": SOURCE_C_ENTRY_POINT,
        "source_c_sha256": source_c_sha,
        "output_task_id": task_id,
        "output_parent_task_id": SOURCE_C_TASK_ID,
        "builder_source_sha256": EXPECTED_SOURCE_D_BUILDER_SHA256,
        "transformation_id": SOURCE_D_TRANSFORMATION_ID,
        "producer_entry_point": SOURCE_D_PRODUCER_ENTRY_POINT,
        "producer_script_sha256": EXPECTED_SOURCE_D_PRODUCER_SHA256,
    }
    if _canonical_json(provenance) != _canonical_json(expected_provenance):
        raise FormalMultiseedExecutionError("Source-D receipt provenance drifted")
    expected_transformation = {
        "source_d_sha256": script_sha,
        "equivalence_artifact_sha256": equivalence_sha,
        "declared_replacement_count": SOURCE_D_DECLARED_REPLACEMENT_COUNT,
        "unchanged_segment_count": SOURCE_D_UNCHANGED_SEGMENT_COUNT,
        "only_declared_anchor_replacements": True,
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "training_seed_cli": "--training-seed",
        "portable_runner_load_marker": PORTABLE_RUNNER_LOAD_MARKER,
        "portable_runner_load_marker_count": PORTABLE_RUNNER_LOAD_MARKER_COUNT,
        "legacy_runner_load_target_anchor_count": (
            LEGACY_RUNNER_LOAD_TARGET_ANCHOR_COUNT
        ),
    }
    if _canonical_json(transformation) != _canonical_json(expected_transformation):
        raise FormalMultiseedExecutionError("Source-D receipt transformation drifted")
    return {
        "task_id": task_id,
        "source_c_task_id": SOURCE_C_TASK_ID,
        "source_c_parent_task_id": SOURCE_C_PARENT_TASK_ID,
        "output_parent_task_id": SOURCE_C_TASK_ID,
        "source_c_script_sha256": source_c_sha,
        "builder_source_sha256": EXPECTED_SOURCE_D_BUILDER_SHA256,
        "source_c_snapshot_seal_sha256": source_c_seal,
        "source_d_script_seal_sha256": source_d_seal,
        "source_d_script_sha256": script_sha,
        "source_d_equivalence_sha256": equivalence_sha,
        "source_d_receipt_seal_sha256": receipt_seal,
        "script": source_script,
    }


def _require_fileserver_url(value: object, *, context: str) -> str:
    if type(value) is not str:
        raise FormalMultiseedExecutionError(f"{context} URL must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != EXPECTED_FILES_SERVER_HOST
        or parsed.port != EXPECTED_FILES_SERVER_PORT
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
        or (parsed.path and not parsed.path.startswith("/"))
        or ".." in PurePosixPath(unquote(parsed.path)).parts
    ):
        raise FormalMultiseedExecutionError(f"{context} URL is not trusted")
    return value


def _models(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise FormalMultiseedExecutionError(f"{context} cannot enumerate models")
    value = getter()
    if not isinstance(value, Mapping):
        raise FormalMultiseedExecutionError(f"{context} model mapping is invalid")
    result: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise FormalMultiseedExecutionError(
                f"{context} model mapping keys must be exact strings"
            )
        result[key] = item
    return result


def _download_and_hash_model(
    model: object,
    *,
    expected_bytes: int,
    expected_sha256: str,
    context: str,
) -> None:
    getter = getattr(model, "get_local_copy", None)
    if not callable(getter):
        raise FormalMultiseedExecutionError(f"{context} cannot be downloaded")
    try:
        value = getter(
            extract_archive=False,
            raise_on_error=True,
            force_download=True,
        )
    except TypeError:
        value = getter()
    if type(value) is not str or not value:
        raise FormalMultiseedExecutionError(f"{context} returned no local copy")
    path = Path(value)
    if path.is_symlink():
        raise FormalMultiseedExecutionError(f"{context} local copy is a symlink")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise FormalMultiseedExecutionError(
            f"{context} local copy does not exist"
        ) from error
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise FormalMultiseedExecutionError(f"{context} byte size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise FormalMultiseedExecutionError(f"{context} SHA-256 mismatch")


def _validate_teacher_gate_semantics(
    gate: Mapping[str, object],
    gate_task: object,
    *,
    teacher_task_id: str,
) -> None:
    _require_exact_keys(
        gate,
        {
            "schema_version",
            "document_type",
            "gate_type",
            "passed",
            "teacher_task_id",
            "quality_gate_task_id",
            "dataset_id",
            "selection_protocol",
            "selection_metric",
            "expected_validation_epochs",
            "validation_count",
            "validations",
            "best",
            "thresholds",
            "teacher",
            "run_contract_schema",
            "training_seed_evidence",
            "training_seed_binding",
            "source_identity",
            "contracts",
            "content_sha256",
        },
        context="teacher quality gate artifact",
    )
    expected_scalars = {
        "gate_type": ("controlled_ffnet_style_bev_noninferiority_and_geometry_safety"),
        "selection_protocol": "DAIR-CLEAN-PAIR1789-v1",
        "selection_metric": "resilient_v2x/car_bev_ap_r40_0.70",
        "expected_validation_epochs": [10, 20, 30, 40, 50],
        "validation_count": 5,
    }
    for key, expected in expected_scalars.items():
        if type(gate.get(key)) is not type(expected) or gate.get(key) != expected:
            raise FormalMultiseedExecutionError(f"teacher quality gate {key} drifted")
    thresholds = gate.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise FormalMultiseedExecutionError("teacher gate thresholds are missing")
    _require_exact_keys(
        thresholds,
        {
            "minimum_car_bev_ap_r40_0.70",
            "minimum_car_3d_ap_r40_0.70",
            "bev_gate_semantics",
            "three_d_gate_semantics",
            "reference",
        },
        context="teacher gate thresholds",
    )
    if (
        thresholds.get("minimum_car_bev_ap_r40_0.70") != 59.6257
        or thresholds.get("minimum_car_3d_ap_r40_0.70") != 30.0
        or thresholds.get("bev_gate_semantics")
        != "ffnet_style_reference_minus_noninferiority_margin"
        or thresholds.get("three_d_gate_semantics")
        != "independent_geometry_safety_floor"
    ):
        raise FormalMultiseedExecutionError(
            "teacher gate thresholds are not the pinned FFNet floors"
        )
    reference = thresholds.get("reference")
    expected_reference = {
        "task_id": "ddbeeec499fb4b55bcd13bc14823b9df",
        "sample_count": 1789,
        "car_ground_truth_count": 15337,
        "unsupported_sample_count": 0,
        "point_cloud_range": [0.0, -40.0, -3.0, 80.0, 40.0, 1.0],
        "car_bev_ap_r40_0.70": 60.6257,
        "car_3d_ap_r40_0.70": 34.9827,
        "bev_noninferiority_margin_ap": 1.0,
    }
    if not isinstance(reference, Mapping) or (
        _canonical_json(reference) != _canonical_json(expected_reference)
    ):
        raise FormalMultiseedExecutionError("teacher gate FFNet reference drifted")
    validations = gate.get("validations")
    if not isinstance(validations, list) or len(validations) != 5:
        raise FormalMultiseedExecutionError("teacher gate validations are incomplete")
    validation_keys = {
        "epoch",
        "validation_iterations",
        "sample_count",
        "car_ground_truth_count",
        "unsupported_sample_count",
        "car_bev_ap_r40_0.50",
        "car_bev_ap_r40_0.70",
        "car_3d_ap_r40_0.50",
        "car_3d_ap_r40_0.70",
    }
    by_epoch: dict[int, Mapping[str, object]] = {}
    for item in validations:
        if not isinstance(item, Mapping):
            raise FormalMultiseedExecutionError("teacher gate validation is invalid")
        _require_exact_keys(item, validation_keys, context="teacher gate validation")
        epoch = item.get("epoch")
        if type(epoch) is not int or epoch in by_epoch:
            raise FormalMultiseedExecutionError("teacher gate validation epoch drifted")
        _positive_int(
            item.get("validation_iterations"),
            context="teacher validation iterations",
        )
        for key, expected in {
            "sample_count": 1789,
            "car_ground_truth_count": 15337,
            "unsupported_sample_count": 0,
        }.items():
            if type(item.get(key)) is not int or item.get(key) != expected:
                raise FormalMultiseedExecutionError(
                    f"teacher gate validation {key} drifted"
                )
        for key in (
            "car_bev_ap_r40_0.50",
            "car_bev_ap_r40_0.70",
            "car_3d_ap_r40_0.50",
            "car_3d_ap_r40_0.70",
        ):
            _finite_ap(item.get(key), context=f"teacher validation {key}")
        by_epoch[epoch] = item
    if list(by_epoch) != [10, 20, 30, 40, 50]:
        raise FormalMultiseedExecutionError("teacher gate validation epoch set drifted")
    teacher = gate.get("teacher")
    best = gate.get("best")
    if not isinstance(teacher, Mapping) or not isinstance(best, Mapping):
        raise FormalMultiseedExecutionError(
            "teacher gate selection evidence is missing"
        )
    selected_epoch = teacher.get("selected_epoch")
    if type(selected_epoch) is not int or selected_epoch not in by_epoch:
        raise FormalMultiseedExecutionError("teacher gate selected epoch is invalid")
    selected = by_epoch[selected_epoch]
    if (
        float(selected["car_bev_ap_r40_0.70"]) < 59.6257
        or float(selected["car_3d_ap_r40_0.70"]) < 30.0
        or best.get("selected_epoch") != selected_epoch
    ):
        raise FormalMultiseedExecutionError(
            "teacher gate selected metrics fail the pinned thresholds"
        )
    _require_exact_keys(
        teacher,
        {
            "task_id",
            "model_id",
            "model_name",
            "model_url",
            "checkpoint_filename",
            "checkpoint_size_bytes",
            "checkpoint_sha256",
            "selected_epoch",
            "final_checkpoint_filename",
            "final_checkpoint_size_bytes",
            "final_checkpoint_sha256",
        },
        context="teacher gate selected teacher",
    )
    expected_checkpoint_filename = (
        f"best_resilient_v2x_car_bev_ap_r40_0.70_teacher_epoch_{selected_epoch}.pth"
    )
    if (
        teacher.get("checkpoint_filename") != expected_checkpoint_filename
        or teacher.get("final_checkpoint_filename") != "teacher_epoch_50.pth"
    ):
        raise FormalMultiseedExecutionError("teacher gate checkpoint filenames drifted")
    _positive_int(
        teacher.get("checkpoint_size_bytes"),
        context="teacher selected checkpoint bytes",
    )
    _positive_int(
        teacher.get("final_checkpoint_size_bytes"),
        context="teacher final checkpoint bytes",
    )
    _sha256(
        teacher.get("checkpoint_sha256"),
        "teacher selected checkpoint",
    )
    _sha256(
        teacher.get("final_checkpoint_sha256"),
        "teacher final checkpoint",
    )

    _require_exact_keys(
        best,
        {
            "selected_epoch",
            "maximum_bev_ap_r40_0.70",
            "rounding_equivalent_best_epochs",
            "log_rounding_unit",
            "selected_metrics",
        },
        context="teacher gate best evidence",
    )
    maximum_bev = max(float(item["car_bev_ap_r40_0.70"]) for item in validations)
    best_epochs = [
        int(item["epoch"])
        for item in validations
        if maximum_bev - float(item["car_bev_ap_r40_0.70"]) <= 0.0001 + 1e-12
    ]
    selected_metrics = {
        key: selected[key]
        for key in (
            "car_bev_ap_r40_0.50",
            "car_bev_ap_r40_0.70",
            "car_3d_ap_r40_0.50",
            "car_3d_ap_r40_0.70",
        )
    }
    if (
        type(best.get("maximum_bev_ap_r40_0.70")) not in {int, float}
        or not math.isfinite(float(best["maximum_bev_ap_r40_0.70"]))
        or float(best["maximum_bev_ap_r40_0.70"]) != maximum_bev
        or best.get("rounding_equivalent_best_epochs") != best_epochs
        or type(best.get("log_rounding_unit")) is not float
        or best.get("log_rounding_unit") != 0.0001
        or not isinstance(best.get("selected_metrics"), Mapping)
        or _canonical_json(best["selected_metrics"])
        != _canonical_json(selected_metrics)
    ):
        raise FormalMultiseedExecutionError(
            "teacher gate best-selection evidence drifted"
        )

    expected_source_identity = {
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_archive_name": SOURCE_ARCHIVE_NAME,
        "source_archive_bytes": SOURCE_ARCHIVE_BYTES,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "training_dataset_id": TRAINING_DATASET_ID,
        "teacher_script_diff_sha256": EXPECTED_TEACHER_SCRIPT_SHA256,
    }
    source_identity = gate.get("source_identity")
    if not isinstance(source_identity, Mapping) or (
        _canonical_json(source_identity) != _canonical_json(expected_source_identity)
    ):
        raise FormalMultiseedExecutionError("teacher gate source identity drifted")
    schema = gate.get("run_contract_schema")
    seed_evidence = gate.get("training_seed_evidence")
    seed_binding = gate.get("training_seed_binding")
    if schema == "legacy_source_c":
        expected_seed_evidence = "legacy_fixed_by_sealed_source"
        expected_seed_binding = {
            "teacher_script_diff_sha256": EXPECTED_TEACHER_SCRIPT_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        }
    elif schema == "seeded":
        expected_seed_evidence = "run_contract_explicit"
        expected_seed_binding = {
            "training_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        }
    else:
        raise FormalMultiseedExecutionError("teacher gate run-contract schema drifted")
    if (
        seed_evidence != expected_seed_evidence
        or not isinstance(seed_binding, Mapping)
        or _canonical_json(seed_binding) != _canonical_json(expected_seed_binding)
    ):
        raise FormalMultiseedExecutionError(
            "teacher gate training seed binding drifted"
        )
    parameters = _parameters(gate_task, context="teacher quality gate")
    for key, expected in {
        "Args/teacher_task_id": teacher_task_id,
        "Args/min_bev_ap70": 59.6257,
        "Args/min_3d_ap70": 30.0,
    }.items():
        if not _parameter_matches(parameters.get(key), expected):
            raise FormalMultiseedExecutionError(f"teacher quality gate {key} drifted")


def _validate_teacher_contract_documents(
    run_contract: Mapping[str, object],
    checkpoint_contract: Mapping[str, object],
    *,
    gate: Mapping[str, object],
    reference: Mapping[str, object],
    teacher_task_id: str,
) -> None:
    base_run_keys = {
        "task_id",
        "dataset_id",
        "dataset_local_copy",
        "gpus",
        "training_world_size",
        "global_batch_size",
        "train_batch_size_per_gpu",
        "eval_batch_size_per_gpu",
        "vehicle_global_batch_size",
        "vehicle_train_batch_size_per_gpu",
        "vehicle_eval_batch_size_per_gpu",
        "learning_rate",
        "auto_scale_lr",
        "expected_optimizer_steps",
        "max_epochs",
        "val_interval",
        "amp",
        "runtime_profile",
        "python_safe_path",
        "checkpoint_policy",
        "manifest_content_sha256",
        "split_sha256",
        "evaluation_index_content_sha256",
        "evaluation_sample_ids_sha256",
        "evaluation_sample_count",
        "evaluation_condition_count",
        "output_uri_scheme",
        "stage",
    }
    schema = gate.get("run_contract_schema")
    if schema == "legacy_source_c":
        expected_run_keys = base_run_keys
    elif schema == "seeded":
        expected_run_keys = base_run_keys | {
            "training_seed",
            "training_overlay_protocol_seed",
            "seed",
        }
    else:
        raise FormalMultiseedExecutionError("teacher run contract schema is invalid")
    _require_exact_keys(
        run_contract,
        expected_run_keys,
        context="teacher run contract",
    )
    expected_run = {
        "task_id": teacher_task_id,
        "dataset_id": TRAINING_DATASET_ID,
        "gpus": 4,
        "training_world_size": 4,
        "global_batch_size": 8,
        "train_batch_size_per_gpu": 2,
        "eval_batch_size_per_gpu": 4,
        "vehicle_global_batch_size": 8,
        "vehicle_train_batch_size_per_gpu": 2,
        "vehicle_eval_batch_size_per_gpu": 4,
        "learning_rate": 0.0001,
        "auto_scale_lr": False,
        "expected_optimizer_steps": None,
        "max_epochs": 50,
        "val_interval": 10,
        "amp": False,
        "runtime_profile": "rtx5090",
        "python_safe_path": "1",
        "checkpoint_policy": ("clean_validation_best_for_teacher_handoff"),
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "split_sha256": (
            "d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
        ),
        "evaluation_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "evaluation_sample_ids_sha256": SAMPLE_IDS_SHA256,
        "evaluation_sample_count": 1337,
        "evaluation_condition_count": 12,
        "stage": "teacher",
    }
    if schema == "seeded":
        expected_run.update(
            {
                "training_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
                "training_overlay_protocol_seed": (TRAINING_OVERLAY_PROTOCOL_SEED),
                "seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            }
        )
    for field, expected in expected_run.items():
        if (
            type(run_contract.get(field)) is not type(expected)
            or run_contract.get(field) != expected
        ):
            raise FormalMultiseedExecutionError(f"teacher run contract {field} drifted")
    dataset_local_copy = run_contract.get("dataset_local_copy")
    if (
        type(dataset_local_copy) is not str
        or not dataset_local_copy
        or not PurePosixPath(dataset_local_copy).is_absolute()
        or ".." in PurePosixPath(dataset_local_copy).parts
    ):
        raise FormalMultiseedExecutionError(
            "teacher run contract dataset path is invalid"
        )
    if run_contract.get("output_uri_scheme") not in {
        "http",
        "https",
        "clearml-default",
    }:
        raise FormalMultiseedExecutionError(
            "teacher run contract output URI scheme is invalid"
        )

    _require_exact_keys(
        checkpoint_contract,
        {
            "schema_version",
            "selection_protocol",
            "selection_metric",
            "selection_rule",
            "selected_epoch",
            "selected_checkpoint",
            "trained_epochs",
            "final_epoch",
            "final_checkpoint",
            "downstream_role",
        },
        context="teacher checkpoint contract",
    )
    expected_checkpoint_scalars = {
        "schema_version": 1,
        "selection_protocol": "DAIR-CLEAN-PAIR1789-v1",
        "selection_metric": "resilient_v2x/car_bev_ap_r40_0.70",
        "selection_rule": "greater",
        "selected_epoch": reference["selected_epoch"],
        "trained_epochs": 50,
        "final_epoch": 50,
        "downstream_role": ("frozen teacher and trainable student initialization"),
    }
    for field, expected in expected_checkpoint_scalars.items():
        if (
            type(checkpoint_contract.get(field)) is not type(expected)
            or checkpoint_contract.get(field) != expected
        ):
            raise FormalMultiseedExecutionError(
                f"teacher checkpoint contract {field} drifted"
            )
    selected_checkpoint = checkpoint_contract.get("selected_checkpoint")
    expected_selected_checkpoint = {
        "model_id": reference["model_id"],
        "name": CLEAN_TEACHER_MODEL_NAME,
        "url": reference["model_url"],
        "filename": reference["checkpoint_filename"],
        "size_bytes": reference["checkpoint_size_bytes"],
        "sha256": reference["checkpoint_sha256"],
    }
    if not isinstance(selected_checkpoint, Mapping) or (
        _canonical_json(selected_checkpoint)
        != _canonical_json(expected_selected_checkpoint)
    ):
        raise FormalMultiseedExecutionError(
            "teacher selected checkpoint contract drifted"
        )
    final_checkpoint = checkpoint_contract.get("final_checkpoint")
    expected_final_checkpoint = {
        "filename": reference["final_checkpoint_filename"],
        "size_bytes": reference["final_checkpoint_size_bytes"],
        "sha256": reference["final_checkpoint_sha256"],
    }
    if not isinstance(final_checkpoint, Mapping) or (
        _canonical_json(final_checkpoint) != _canonical_json(expected_final_checkpoint)
    ):
        raise FormalMultiseedExecutionError("teacher final checkpoint contract drifted")


def _validate_teacher_gate(
    gate_task: object,
    teacher: object,
    *,
    gate_task_id: str,
    teacher_task_id: str,
) -> dict[str, object]:
    _require_no_repo_script(
        gate_task,
        entry_point=TEACHER_GATE_ENTRY_POINT,
        expected_sha256=EXPECTED_TEACHER_GATE_PRODUCER_SHA256,
        context="teacher quality gate",
    )
    _require_no_repo_script(
        teacher,
        entry_point=TRAINING_ENTRY_POINT,
        expected_sha256=EXPECTED_TEACHER_SCRIPT_SHA256,
        context="clean teacher",
    )
    if _task_parent(teacher, context="clean teacher") != "":
        raise FormalMultiseedExecutionError("clean teacher must be a root task")
    gate = _artifact_mapping(
        gate_task,
        TEACHER_QUALITY_GATE_ARTIFACT,
        context="teacher quality gate",
    )
    gate_hash = _require_content_hash(
        gate,
        field="content_sha256",
        context="teacher quality gate artifact",
    )
    _exact_bool(gate.get("passed"), True, context="teacher quality gate passed")
    _exact_int(gate.get("schema_version"), 1, context="teacher gate schema")
    if gate.get("document_type") != "resilient_v2x_teacher_quality_gate":
        raise FormalMultiseedExecutionError("teacher gate document type mismatch")
    if (
        gate.get("teacher_task_id") != teacher_task_id
        or gate.get("quality_gate_task_id") != gate_task_id
        or gate.get("dataset_id") != TRAINING_DATASET_ID
    ):
        raise FormalMultiseedExecutionError("teacher gate identity mismatch")
    _validate_teacher_gate_semantics(
        gate,
        gate_task,
        teacher_task_id=teacher_task_id,
    )
    contracts = gate.get("contracts")
    reference = gate.get("teacher")
    if not isinstance(contracts, Mapping) or not isinstance(reference, Mapping):
        raise FormalMultiseedExecutionError("teacher gate contracts are missing")
    _require_exact_keys(
        contracts,
        {
            "run_contract_artifact",
            "run_contract_sha256",
            "checkpoint_contract_artifact",
            "checkpoint_contract_sha256",
        },
        context="teacher gate contract bindings",
    )
    run_contract = _artifact_mapping(
        teacher, TEACHER_RUN_CONTRACT_ARTIFACT, context="clean teacher"
    )
    checkpoint_contract = _artifact_mapping(
        teacher, TEACHER_CHECKPOINT_ARTIFACT, context="clean teacher"
    )
    _validate_teacher_contract_documents(
        run_contract,
        checkpoint_contract,
        gate=gate,
        reference=reference,
        teacher_task_id=teacher_task_id,
    )

    run_hash = _content_sha256(run_contract)
    checkpoint_contract_hash = _content_sha256(checkpoint_contract)
    if (
        contracts.get("run_contract_artifact") != TEACHER_RUN_CONTRACT_ARTIFACT
        or contracts.get("run_contract_sha256") != run_hash
        or contracts.get("checkpoint_contract_artifact") != TEACHER_CHECKPOINT_ARTIFACT
        or contracts.get("checkpoint_contract_sha256") != checkpoint_contract_hash
    ):
        raise FormalMultiseedExecutionError("teacher gate contract hashes mismatch")
    teacher_model_id = _clearml_id(reference.get("model_id"), "teacher model")
    checkpoint_sha = _sha256(reference.get("checkpoint_sha256"), "teacher checkpoint")
    checkpoint_bytes = _positive_int(
        reference.get("checkpoint_size_bytes"), context="teacher checkpoint bytes"
    )
    model_url = _require_fileserver_url(
        reference.get("model_url"), context="teacher model"
    )
    if (
        reference.get("task_id") != teacher_task_id
        or reference.get("model_name") != CLEAN_TEACHER_MODEL_NAME
    ):
        raise FormalMultiseedExecutionError("teacher reference identity mismatch")
    outputs = _models(teacher, context="clean teacher").get("output")
    if (
        not isinstance(outputs, Sequence)
        or isinstance(outputs, (str, bytes))
        or len(outputs) != 1
    ):
        raise FormalMultiseedExecutionError("teacher output models are invalid")
    matches = [
        model
        for model in outputs
        if getattr(model, "name", None) == CLEAN_TEACHER_MODEL_NAME
    ]
    if len(matches) != 1:
        raise FormalMultiseedExecutionError(
            "teacher must expose exactly one selected OutputModel"
        )
    model = matches[0]
    if (
        _clearml_id(getattr(model, "id", None), "teacher OutputModel")
        != teacher_model_id
        or getattr(model, "task", None) != teacher_task_id
        or _require_fileserver_url(
            getattr(model, "url", None), context="teacher OutputModel"
        )
        != model_url
    ):
        raise FormalMultiseedExecutionError("teacher OutputModel binding mismatch")
    _download_and_hash_model(
        model,
        expected_bytes=checkpoint_bytes,
        expected_sha256=checkpoint_sha,
        context="teacher OutputModel",
    )
    return {
        "quality_gate_task_id": gate_task_id,
        "quality_gate_content_sha256": gate_hash,
        "teacher_task_id": teacher_task_id,
        "teacher_model_id": teacher_model_id,
        "teacher_model_url": model_url,
        "teacher_checkpoint_size_bytes": checkpoint_bytes,
        "teacher_checkpoint_sha256": checkpoint_sha,
        "teacher_run_contract_sha256": run_hash,
        "teacher_checkpoint_contract_sha256": checkpoint_contract_hash,
    }


def _project_id(task_class: object, project: str) -> str:
    if type(project) is str and _is_lower_hex(project, 32):
        return project
    getter = getattr(task_class, "get_project_id", None)
    if not callable(getter):
        raise FormalMultiseedExecutionError("ClearML cannot resolve the project ID")
    try:
        value = getter(project_name=project, search_hidden=True)
    except TypeError:
        value = getter(project_name=project)
    return _clearml_id(value, "formal execution project")


def _task_name(task: object, *, context: str) -> str:
    value = getattr(getattr(task, "data", None), "name", None)
    if type(value) is not str or not value:
        raise FormalMultiseedExecutionError(f"{context} name is unavailable")
    return value


def _output_uri(task: object, *, context: str) -> str:
    value = getattr(
        getattr(getattr(task, "data", None), "output", None),
        "destination",
        None,
    )
    return _require_fileserver_url(value, context=f"{context} output")


def _set_parameters_exact(task: object, parameters: Mapping[str, object]) -> None:
    setter = getattr(task, "set_parameters", None)
    if not callable(setter):
        raise FormalMultiseedExecutionError("created task cannot replace Args")
    frozen = _frozen_mapping(parameters, context="expected task parameters")
    try:
        result = setter(frozen, __update=False)
    except TypeError:
        result = setter(frozen)
    if result is False:
        raise FormalMultiseedExecutionError("created task rejected exact Args")


def _set_source_d_script(task: object, source: str) -> None:
    setter = getattr(task, "set_script", None)
    if not callable(setter):
        raise FormalMultiseedExecutionError("created task cannot set Source-D script")
    values = {
        "repository": "",
        "working_dir": ".",
        "entry_point": TRAINING_ENTRY_POINT,
        "diff": source,
    }
    try:
        result = setter(**values)
    except TypeError:
        result = setter(values)
    if result is False:
        raise FormalMultiseedExecutionError("created task rejected Source-D script")


def _parameter_matches(observed: object, expected: object) -> bool:
    if type(expected) is bool:
        return (type(observed) is bool and observed is expected) or (
            type(observed) is str and observed == str(expected)
        )
    if type(expected) is int:
        return (type(observed) is int and observed == expected) or (
            type(observed) is str and observed == str(expected)
        )
    if type(expected) is float:
        return (
            type(observed) is float and math.isfinite(observed) and observed == expected
        ) or (type(observed) is str and observed == str(expected))
    return type(observed) is type(expected) and observed == expected


def _require_exact_parameters(
    task: object,
    expected: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    observed = _parameters(task, context=context)
    if set(observed) != set(expected):
        raise FormalMultiseedExecutionError(f"{context} Args key set drifted")
    normalized: dict[str, object] = {}
    for key, expected_value in expected.items():
        if not _parameter_matches(observed.get(key), expected_value):
            raise FormalMultiseedExecutionError(f"{context} {key} drifted")
        normalized[key] = expected_value
    return normalized


def _base_parameters(*, training_seed: int) -> dict[str, object]:
    return {
        "Args/source_dataset_id": SOURCE_DATASET_ID,
        "Args/source_archive_name": SOURCE_ARCHIVE_NAME,
        "Args/source_archive_bytes": SOURCE_ARCHIVE_BYTES,
        "Args/source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "Args/training_dataset_id": TRAINING_DATASET_ID,
        "Args/native_bundle_bytes": NATIVE_BUNDLE_BYTES,
        "Args/native_bundle_sha256": NATIVE_BUNDLE_SHA256,
        "Args/build_manifest_sha256": BUILD_MANIFEST_SHA256,
        "Args/gpus": 4,
        "Args/max_epochs": 50,
        "Args/training_seed": training_seed,
        "Args/amp": False,
    }


def _training_parameters(
    record: Mapping[str, object],
    *,
    source_d_task_id: str,
    teacher: Mapping[str, object],
) -> dict[str, object]:
    result = _base_parameters(training_seed=int(record["training_seed"]))
    result.update(
        {
            "Args/stage": "all",
            "Args/experiment_from_task": record["subject"],
            "Args/predecessor_task_id": source_d_task_id,
            "Args/teacher_task_id": teacher["teacher_task_id"],
            "Args/teacher_model_id": teacher["teacher_model_id"],
            "Args/teacher_checkpoint_sha256": teacher["teacher_checkpoint_sha256"],
            "Args/allow_failed_teacher_task": False,
        }
    )
    return result


def _evaluation_parameters(
    record: Mapping[str, object],
    *,
    training_task_id: str,
    model: Mapping[str, object],
) -> dict[str, object]:
    result = _base_parameters(training_seed=int(record["training_seed"]))
    result.update(
        {
            "Args/stage": "baseline_validate",
            "Args/predecessor_task_id": training_task_id,
            "Args/controlled_baseline": record["subject"],
            "Args/controlled_baseline_task_id": training_task_id,
            "Args/controlled_baseline_model_id": model["model_id"],
            "Args/controlled_baseline_checkpoint_sha256": model["checkpoint_sha256"],
        }
    )
    return result


def _task_run_name(record: Mapping[str, object], *, executor_task_id: str) -> str:
    key = record.get("task_key")
    if type(key) is not str or not re.fullmatch(r"[a-z0-9_-]+", key):
        raise FormalMultiseedExecutionError("formal task key is unsafe")
    owner = _clearml_id(executor_task_id, "executor task")
    return f"ResilientV2X formal 1337 {owner} {key}"


def _query_named_tasks(
    task_class: object,
    *,
    project: str,
    name: str,
) -> list[object]:
    getter = getattr(task_class, "get_tasks", None)
    if not callable(getter):
        raise FormalMultiseedExecutionError(
            "ClearML cannot discover prior formal execution tasks"
        )
    query: dict[str, object] = {
        "task_name": f"^{re.escape(name)}$",
    }
    if _is_lower_hex(project, 32):
        query["project_name"] = None
        query["task_filter"] = {"project": [project]}
    else:
        query["project_name"] = project
    raw = getter(**query)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise FormalMultiseedExecutionError("ClearML task discovery is invalid")
    return [task for task in raw if _task_name(task, context="discovered task") == name]


def _script_equals(
    task: object, expected: Mapping[str, object], *, context: str
) -> bool:
    return _canonical_json(_script(task, context=context)) == _canonical_json(expected)


def _parameters_equal(
    task: object, expected: Mapping[str, object], *, context: str
) -> bool:
    try:
        _require_exact_parameters(task, expected, context=context)
    except FormalMultiseedExecutionError:
        return False
    return True


def _empty_draft_outputs(task: object, *, context: str) -> None:
    if _artifact_names(task, context=context):
        raise FormalMultiseedExecutionError(f"{context} contains stale artifacts")
    outputs = _models(task, context=context).get("output", [])
    if (
        not isinstance(outputs, Sequence)
        or isinstance(outputs, (str, bytes))
        or outputs
    ):
        raise FormalMultiseedExecutionError(f"{context} contains stale output models")


def _require_bound_task(
    task: object,
    *,
    expected_id: str,
    expected_parent: str,
    expected_name: str,
    expected_parameters: Mapping[str, object],
    source_d: str,
    queue_name: str | None,
    context: str,
) -> None:
    if _clearml_id(getattr(task, "id", None), context) != expected_id:
        raise FormalMultiseedExecutionError(f"{context} identity drifted")
    if _task_parent(task, context=context) != expected_parent:
        raise FormalMultiseedExecutionError(f"{context} parent drifted")
    if _task_name(task, context=context) != expected_name:
        raise FormalMultiseedExecutionError(f"{context} name drifted")
    _require_no_repo_script(
        task,
        entry_point=TRAINING_ENTRY_POINT,
        expected_sha256=hashlib.sha256(source_d.encode("utf-8")).hexdigest(),
        expected_source=source_d,
        context=context,
    )
    _require_exact_parameters(task, expected_parameters, context=context)
    _output_uri(task, context=context)
    observed_queue = _queue_id(task, context=context)
    if queue_name is None:
        if observed_queue:
            raise FormalMultiseedExecutionError(f"{context} was queued prematurely")
    elif observed_queue != QUEUE_IDS[queue_name]:
        raise FormalMultiseedExecutionError(f"{context} queue drifted")


def _clone_or_resume_created(
    task_class: object,
    *,
    teacher_task: object,
    teacher_script: Mapping[str, object],
    teacher_parameters: Mapping[str, object],
    source_d: str,
    record: Mapping[str, object],
    parent_task_id: str,
    executor_task_id: str,
    expected_parameters: Mapping[str, object],
    project: str,
    used_ids: set[str],
) -> object:
    name = _task_run_name(record, executor_task_id=executor_task_id)
    matches = _query_named_tasks(task_class, project=project, name=name)
    if len(matches) > 1:
        raise FormalMultiseedExecutionError(f"duplicate ClearML tasks for {name!r}")
    if matches:
        task = matches[0]
    else:
        clone = getattr(task_class, "clone", None)
        if not callable(clone):
            raise FormalMultiseedExecutionError("ClearML cannot clone formal tasks")
        task = clone(
            source_task=teacher_task,
            name=name,
            comment="Sealed initial-only DAIR-CAUSAL-1337-v1 multi-seed task.",
            parent=parent_task_id,
            project=_project_id(task_class, project),
        )
        if task is None:
            raise FormalMultiseedExecutionError("ClearML clone returned no task")
    task_id = _clearml_id(getattr(task, "id", None), "formal task")
    if task_id in used_ids:
        raise FormalMultiseedExecutionError("ClearML task IDs are not globally unique")
    used_ids.add(task_id)
    _reload(task, context=f"formal task {task_id}")
    status = _status(task, context=f"formal task {task_id}")
    if _task_parent(task, context="formal task") != parent_task_id:
        raise FormalMultiseedExecutionError("formal task parent drifted")
    if _task_name(task, context="formal task") != name:
        raise FormalMultiseedExecutionError("formal task name drifted")
    if status == "created":
        _empty_draft_outputs(task, context="formal created task")
        source_d_script = {
            "repository": "",
            "working_dir": ".",
            "entry_point": TRAINING_ENTRY_POINT,
            "diff": source_d,
        }
        teacher_stage = _script_equals(
            task, teacher_script, context="formal created task"
        ) and _parameters_equal(task, teacher_parameters, context="formal created task")
        source_stage = _script_equals(
            task, source_d_script, context="formal created task"
        ) and _parameters_equal(task, teacher_parameters, context="formal created task")
        final_stage = _script_equals(
            task, source_d_script, context="formal created task"
        ) and _parameters_equal(
            task, expected_parameters, context="formal created task"
        )
        if teacher_stage:
            _set_source_d_script(task, source_d)
            _reload(task, context="formal created task")
            source_stage = _script_equals(
                task, source_d_script, context="formal created task"
            ) and _parameters_equal(
                task, teacher_parameters, context="formal created task"
            )
            if not source_stage:
                raise FormalMultiseedExecutionError(
                    "Source-D script setter did not round-trip exact bytes"
                )
        if source_stage:
            _set_parameters_exact(task, expected_parameters)
            _reload(task, context="formal created task")
            final_stage = _script_equals(
                task, source_d_script, context="formal created task"
            ) and _parameters_equal(
                task, expected_parameters, context="formal created task"
            )
            if not final_stage:
                raise FormalMultiseedExecutionError(
                    "formal Args setter did not round-trip exactly"
                )
        if not final_stage:
            raise FormalMultiseedExecutionError(
                "created task is outside the three recoverable exact stages"
            )
        task.output_uri = FILES_SERVER_URI
        _reload(task, context="formal created task")
        _require_bound_task(
            task,
            expected_id=task_id,
            expected_parent=parent_task_id,
            expected_name=name,
            expected_parameters=expected_parameters,
            source_d=source_d,
            queue_name=None,
            context="formal created task",
        )
    elif status in ACTIVE_STATUSES | {"completed"}:
        queue_name = str(record["worker_queue"])
        _require_bound_task(
            task,
            expected_id=task_id,
            expected_parent=parent_task_id,
            expected_name=name,
            expected_parameters=expected_parameters,
            source_d=source_d,
            queue_name=queue_name,
            context="formal resumed task",
        )
    else:
        raise FormalMultiseedExecutionError(
            f"formal task has unrecoverable status {status!r}"
        )
    return task


def _enqueue_once(task_class: object, task: object, *, queue_name: str) -> None:
    task_id = _clearml_id(getattr(task, "id", None), "formal task")
    if _status(task, context="formal task") != "created":
        return
    enqueue = getattr(task_class, "enqueue", None)
    if not callable(enqueue):
        raise FormalMultiseedExecutionError("ClearML cannot enqueue formal tasks")
    caught: Exception | None = None
    try:
        response = enqueue(task=task, queue_id=QUEUE_IDS[queue_name], force=False)
    except Exception as error:
        caught = error
        response = None
    _reload(task, context=f"formal task {task_id}")
    committed = (
        _status(task, context="formal task") in ACTIVE_STATUSES | {"completed"}
        and _queue_id(task, context="formal task") == QUEUE_IDS[queue_name]
    )
    if caught is not None and not committed:
        raise FormalMultiseedExecutionError(
            "ClearML enqueue failed before an auditable server commit"
        ) from caught
    if caught is None:
        if isinstance(response, Mapping):
            acknowledged = response.get("queued") == 1 and response.get("updated") == 1
        else:
            acknowledged = (
                getattr(response, "queued", None) == 1
                and getattr(response, "updated", None) == 1
            )
        if not acknowledged and not committed:
            raise FormalMultiseedExecutionError("ClearML did not acknowledge enqueue")
    if not committed:
        raise FormalMultiseedExecutionError("ClearML enqueue commit cannot be verified")


def _source_d_runtime_profile(gpu_model: object, *, context: str) -> str:
    if type(gpu_model) is not str or gpu_model not in PLANNED_GPU_MODELS:
        raise FormalMultiseedExecutionError(f"{context} GPU model is invalid")
    return SOURCE_D_RUNTIME_PROFILE


def _require_worker(task: object, gpu_model: object, *, context: str) -> str:
    _source_d_runtime_profile(
        gpu_model,
        context=f"{context} worker",
    )
    # ClearML last_worker is an opaque agent ID; hardware is bound by queue ID.
    return _last_worker(task, context=context)


def _final_model(task: object, *, subject: str, context: str) -> object:
    name = f"ResilientV2X {subject} final checkpoint"
    outputs = _models(task, context=context).get("output")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise FormalMultiseedExecutionError(f"{context} output models are invalid")
    matches = [item for item in outputs if getattr(item, "name", None) == name]
    if len(matches) != 1:
        raise FormalMultiseedExecutionError(
            f"{context} must expose exactly one final checkpoint model"
        )
    return matches[0]


def _validate_initialization_audit(
    audit: Mapping[str, object],
    *,
    subject: str,
    teacher: Mapping[str, object],
    context: str,
) -> None:
    _require_exact_keys(
        audit,
        {
            "schema_version",
            "contract",
            "result",
            "checkpoint",
            "source",
            "shared_initialization",
            "method_specific_fusion",
            "target",
        },
        context=f"{context} initialization audit",
    )
    _exact_int(
        audit.get("schema_version"),
        1,
        context=f"{context} initialization audit schema",
    )
    if (
        audit.get("contract") != COMMON_TEACHER_INITIALIZATION_CONTRACT
        or audit.get("result") != "pass"
    ):
        raise FormalMultiseedExecutionError(
            f"{context} initialization audit identity drifted"
        )

    checkpoint = audit.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise FormalMultiseedExecutionError(f"{context} audit checkpoint is missing")
    _require_exact_keys(
        checkpoint,
        {"path", "filename", "size_bytes", "sha256", "expected_sha256"},
        context=f"{context} audit checkpoint",
    )
    checkpoint_path = checkpoint.get("path")
    checkpoint_filename = checkpoint.get("filename")
    if (
        type(checkpoint_path) is not str
        or not checkpoint_path
        or not Path(checkpoint_path).is_absolute()
        or type(checkpoint_filename) is not str
        or not checkpoint_filename
        or Path(checkpoint_path).name != checkpoint_filename
    ):
        raise FormalMultiseedExecutionError(f"{context} audit checkpoint path drifted")
    _exact_int(
        checkpoint.get("size_bytes"),
        int(teacher["teacher_checkpoint_size_bytes"]),
        context=f"{context} audit checkpoint bytes",
    )
    for field in ("sha256", "expected_sha256"):
        if (
            _sha256(
                checkpoint.get(field),
                f"{context} audit checkpoint {field}",
            )
            != teacher["teacher_checkpoint_sha256"]
        ):
            raise FormalMultiseedExecutionError(f"{context} audit teacher SHA drifted")

    source = audit.get("source")
    if not isinstance(source, Mapping):
        raise FormalMultiseedExecutionError(
            f"{context} audit teacher source is missing"
        )
    _require_exact_keys(
        source,
        {
            "keys",
            "numel",
            "bytes",
            "expected_keys",
            "common_keys",
            "expected_common_keys",
            "fusion_keys",
            "expected_fusion_keys",
            "state_sha256",
        },
        context=f"{context} audit teacher source",
    )
    for field, expected in {
        "keys": COMMON_TEACHER_SOURCE_KEYS,
        "numel": COMMON_TEACHER_SOURCE_NUMEL,
        "bytes": COMMON_TEACHER_SOURCE_BYTES,
        "expected_keys": COMMON_TEACHER_SOURCE_KEYS,
        "common_keys": COMMON_TEACHER_SHARED_KEYS,
        "expected_common_keys": COMMON_TEACHER_SHARED_KEYS,
        "fusion_keys": COMMON_TEACHER_FUSION_KEYS,
        "expected_fusion_keys": COMMON_TEACHER_FUSION_KEYS,
    }.items():
        _exact_int(
            source.get(field),
            expected,
            context=f"{context} audit source {field}",
        )
    _sha256(source.get("state_sha256"), f"{context} audit source state")

    shared = audit.get("shared_initialization")
    if not isinstance(shared, Mapping):
        raise FormalMultiseedExecutionError(
            f"{context} shared initialization is missing"
        )
    _require_exact_keys(
        shared,
        {
            "prefixes",
            "keys",
            "numel",
            "bytes",
            "expected_keys",
            "state_sha256",
            "shape_dtype_verified",
            "exact_tensor_equality_verified",
        },
        context=f"{context} shared initialization",
    )
    if shared.get("prefixes") != list(COMMON_TEACHER_INITIALIZATION_PREFIXES):
        raise FormalMultiseedExecutionError(
            f"{context} shared initialization prefixes drifted"
        )
    for field, expected in {
        "keys": COMMON_TEACHER_SHARED_KEYS,
        "numel": COMMON_TEACHER_SHARED_NUMEL,
        "bytes": COMMON_TEACHER_SHARED_BYTES,
        "expected_keys": COMMON_TEACHER_SHARED_KEYS,
    }.items():
        _exact_int(
            shared.get(field),
            expected,
            context=f"{context} shared initialization {field}",
        )
    _sha256(
        shared.get("state_sha256"),
        f"{context} shared initialization state",
    )
    _exact_bool(
        shared.get("shape_dtype_verified"),
        True,
        context=f"{context} shared shape/dtype audit",
    )
    _exact_bool(
        shared.get("exact_tensor_equality_verified"),
        True,
        context=f"{context} shared tensor equality audit",
    )

    fusion = audit.get("method_specific_fusion")
    if not isinstance(fusion, Mapping):
        raise FormalMultiseedExecutionError(
            f"{context} method-specific fusion audit is missing"
        )
    _require_exact_keys(
        fusion,
        {
            "keys",
            "numel",
            "bytes",
            "sha256_before",
            "sha256_after",
            "unchanged",
        },
        context=f"{context} method-specific fusion audit",
    )
    fusion_keys = _nonnegative_int(
        fusion.get("keys"),
        context=f"{context} fusion keys",
    )
    fusion_numel = _nonnegative_int(
        fusion.get("numel"),
        context=f"{context} fusion numel",
    )
    fusion_bytes = _nonnegative_int(
        fusion.get("bytes"),
        context=f"{context} fusion bytes",
    )
    fusion_before = _sha256(
        fusion.get("sha256_before"),
        f"{context} fusion before",
    )
    fusion_after = _sha256(
        fusion.get("sha256_after"),
        f"{context} fusion after",
    )
    _exact_bool(
        fusion.get("unchanged"),
        True,
        context=f"{context} method-specific fusion unchanged",
    )
    if fusion_before != fusion_after:
        raise FormalMultiseedExecutionError(f"{context} method-specific fusion changed")
    if subject in ZERO_FUSION_SUBJECTS:
        if (
            fusion_keys != 0
            or fusion_numel != 0
            or fusion_bytes != 0
            or fusion_before != EMPTY_TENSOR_MAPPING_SHA256
        ):
            raise FormalMultiseedExecutionError(f"{context} zero-fusion audit drifted")
    elif (
        fusion_keys <= 0
        or fusion_numel <= 0
        or fusion_bytes <= 0
        or fusion_before == EMPTY_TENSOR_MAPPING_SHA256
    ):
        raise FormalMultiseedExecutionError(f"{context} non-empty fusion audit drifted")

    target = audit.get("target")
    if not isinstance(target, Mapping):
        raise FormalMultiseedExecutionError(
            f"{context} initialization target audit is missing"
        )
    _require_exact_keys(
        target,
        {
            "model_type",
            "target_key_count",
            "target_common_key_count",
            "target_fusion_key_count",
            "nested_teacher_present",
            "nested_teacher_key_count",
            "nested_teacher_full_equality_verified",
        },
        context=f"{context} initialization target audit",
    )
    if type(target.get("model_type")) is not str or not target["model_type"]:
        raise FormalMultiseedExecutionError(
            f"{context} initialization target type is invalid"
        )
    nested_teacher = subject in NESTED_TEACHER_SUBJECTS
    expected_target_keys = (
        COMMON_TEACHER_SHARED_KEYS
        + fusion_keys
        + (COMMON_TEACHER_SOURCE_KEYS if nested_teacher else 0)
    )
    for field, expected in {
        "target_key_count": expected_target_keys,
        "target_common_key_count": COMMON_TEACHER_SHARED_KEYS,
        "target_fusion_key_count": fusion_keys,
        "nested_teacher_key_count": (
            COMMON_TEACHER_SOURCE_KEYS if nested_teacher else 0
        ),
    }.items():
        _exact_int(
            target.get(field),
            expected,
            context=f"{context} initialization target {field}",
        )
    _exact_bool(
        target.get("nested_teacher_present"),
        nested_teacher,
        context=f"{context} nested teacher presence",
    )
    _exact_bool(
        target.get("nested_teacher_full_equality_verified"),
        nested_teacher,
        context=f"{context} nested teacher equality",
    )


def _validate_training_completion(
    task: object,
    record: Mapping[str, object],
    *,
    teacher: Mapping[str, object],
    source_d_task_id: str,
) -> dict[str, object]:
    context = f"training task {record['task_key']}"
    task_id = _clearml_id(getattr(task, "id", None), context)
    if _status(task, context=context) != "completed":
        raise FormalMultiseedExecutionError(f"{context} is not completed")
    worker = _require_worker(task, record.get("gpu_model"), context=context)
    run_contract = _artifact_mapping(task, RUN_CONTRACT_ARTIFACT, context=context)
    _require_exact_keys(
        run_contract,
        {
            "schema_version",
            "mode",
            "task_id",
            "experiment",
            "experiment_kind",
            "config",
            "source_dataset_id",
            "training_dataset_id",
            "native_build_task_id",
            "native_bundle_sha256",
            "build_manifest_sha256",
            "base_image_manifest_digest",
            "source_archive",
            "predecessor_task_id",
            "gpus",
            "global_batch_size",
            "train_batch_size_per_gpu",
            "eval_batch_size_per_gpu",
            "launcher",
            "ddp_processes",
            "max_epochs",
            "training_seed",
            "training_overlay_protocol_seed",
            "seed",
            "learning_rate",
            "auto_scale_lr",
            "amp",
            "precision",
            "runtime_profile",
            "val_interval",
            "per_epoch_validation",
            "condition_evaluation",
            "condition_evaluation_reason",
            "teacher",
            "common_teacher_initialization",
            "baseline_dry_run_required",
            "dataset_local_copy",
            "training_command",
        },
        context=f"{context} run contract",
    )
    subject = record.get("subject")
    if type(subject) is not str or not subject:
        raise FormalMultiseedExecutionError(f"{context} planned subject is invalid")
    if record.get("kind") == "external_controlled_baseline":
        experiment_kind = "baseline"
    elif subject == "resilient_v2x":
        experiment_kind = "primary_method"
    elif record.get("kind") == "selected_candidate":
        experiment_kind = "improvement"
    else:
        raise FormalMultiseedExecutionError(
            f"{context} planned experiment kind is invalid"
        )
    gpu_model = record.get("gpu_model")
    runtime_profile = _source_d_runtime_profile(
        gpu_model,
        context=f"{context} planned",
    )
    expected_run = {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": record["subject"],
        "experiment_kind": experiment_kind,
        "source_dataset_id": SOURCE_DATASET_ID,
        "training_dataset_id": TRAINING_DATASET_ID,
        "native_build_task_id": NATIVE_BUILD_TASK_ID,
        "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
        "build_manifest_sha256": BUILD_MANIFEST_SHA256,
        "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        "predecessor_task_id": source_d_task_id,
        "gpus": 4,
        "global_batch_size": 8,
        "train_batch_size_per_gpu": 2,
        "eval_batch_size_per_gpu": 4,
        "launcher": "pytorch",
        "ddp_processes": 4,
        "max_epochs": 50,
        "training_seed": record["training_seed"],
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "seed": record["training_seed"],
        "learning_rate": 0.0001,
        "auto_scale_lr": False,
        "amp": False,
        "precision": "FP32",
        "runtime_profile": runtime_profile,
        "val_interval": 10,
        "per_epoch_validation": False,
        "condition_evaluation": False,
        "condition_evaluation_reason": (
            "the 12-condition matrix remains outside individual training tasks"
        ),
        "baseline_dry_run_required": experiment_kind == "baseline",
    }
    for key, expected in expected_run.items():
        if (
            type(run_contract.get(key)) is not type(expected)
            or run_contract.get(key) != expected
        ):
            raise FormalMultiseedExecutionError(f"{context} run contract {key} drifted")
    source_archive = run_contract.get("source_archive")
    expected_source_archive = {
        "name": SOURCE_ARCHIVE_NAME,
        "size_bytes": SOURCE_ARCHIVE_BYTES,
        "sha256": SOURCE_ARCHIVE_SHA256,
    }
    if not isinstance(source_archive, Mapping) or (
        _canonical_json(source_archive) != _canonical_json(expected_source_archive)
    ):
        raise FormalMultiseedExecutionError(
            f"{context} source archive contract drifted"
        )

    config = run_contract.get("config")
    if not isinstance(config, Mapping):
        raise FormalMultiseedExecutionError(f"{context} config contract is missing")
    _require_exact_keys(
        config,
        {
            "declared",
            "declared_resolved",
            "config_sha256",
            "resolved",
            "size_bytes",
            "resolved_config_sha256",
        },
        context=f"{context} config contract",
    )
    candidate_configs = {
        "resilient_v2x": "configs/resilient_v2x/dair_resilient_v2x.py",
        "support_residual": ("configs/resilient_v2x/improvements/support_residual.py"),
        "linear_no_distillation": (
            "configs/resilient_v2x/improvements/linear_no_distillation.py"
        ),
        "no_distillation_peak_lr_3e4": (
            "configs/resilient_v2x/improvements/no_distillation_peak_lr_3e4.py"
        ),
    }
    if experiment_kind == "baseline":
        declared_config = None
    else:
        declared_config = candidate_configs.get(subject)
        if declared_config is None:
            raise FormalMultiseedExecutionError(
                f"{context} selected candidate config is not pinned"
            )
    if config.get("declared") != declared_config:
        raise FormalMultiseedExecutionError(f"{context} declared config drifted")
    declared_resolved = config.get("declared_resolved")
    resolved_config = config.get("resolved")
    if (
        type(declared_resolved) is not str
        or not declared_resolved
        or not Path(declared_resolved).is_absolute()
        or type(resolved_config) is not str
        or not resolved_config
        or not Path(resolved_config).is_absolute()
    ):
        raise FormalMultiseedExecutionError(f"{context} resolved config path drifted")
    declared_config_sha = _sha256(
        config.get("config_sha256"),
        f"{context} declared config",
    )
    resolved_config_sha = _sha256(
        config.get("resolved_config_sha256"),
        f"{context} resolved config",
    )
    _positive_int(
        config.get("size_bytes"),
        context=f"{context} resolved config bytes",
    )
    if experiment_kind != "baseline" and (
        declared_resolved != resolved_config
        or declared_config_sha != resolved_config_sha
    ):
        raise FormalMultiseedExecutionError(
            f"{context} candidate config identity drifted"
        )

    dataset_local_copy = run_contract.get("dataset_local_copy")
    if (
        type(dataset_local_copy) is not str
        or not dataset_local_copy
        or not Path(dataset_local_copy).is_absolute()
    ):
        raise FormalMultiseedExecutionError(f"{context} dataset local path is invalid")
    training_command = run_contract.get("training_command")
    if type(training_command) is not list or any(
        type(item) is not str or not item for item in training_command
    ):
        raise FormalMultiseedExecutionError(f"{context} training command is invalid")
    seed = int(record["training_seed"])
    expected_command_tail = [
        "train_cfg.max_epochs=50",
        "train_cfg.val_interval=10",
        "train_dataloader.batch_size=2",
        "val_dataloader.batch_size=4",
        "test_dataloader.batch_size=4",
        f"randomness.seed={seed}",
        f"train_dataloader.sampler.seed={seed}",
        f"val_dataloader.sampler.seed={seed}",
        f"test_dataloader.sampler.seed={seed}",
        f"train_dataloader.dataset.seed={seed}",
        f"val_dataloader.dataset.seed={seed}",
        f"test_dataloader.dataset.seed={seed}",
        f"implementation_choices_dataset.global_seed={seed}",
        "find_unused_parameters=True",
        "default_hooks.checkpoint.interval=10",
        "default_hooks.checkpoint.max_keep_ckpts=5",
        "visualizer._scope_=mmengine",
        "visualizer.type=Visualizer",
        "visualizer.vis_backends.0._scope_=mmengine",
    ]
    if (
        len(training_command) != 14 + len(expected_command_tail)
        or not Path(training_command[0]).is_absolute()
        or training_command[1:8]
        != [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "--module",
            "tools.resilient_v2x.run_deterministic",
            "tools/train.py",
        ]
        or training_command[8] != resolved_config
        or training_command[9] != "--work-dir"
        or not Path(training_command[10]).is_absolute()
        or training_command[11:14] != ["--launcher", "pytorch", "--cfg-options"]
        or training_command[14:] != expected_command_tail
    ):
        raise FormalMultiseedExecutionError(
            f"{context} deterministic training command drifted"
        )

    teacher_handoff = run_contract.get("teacher")
    if not isinstance(teacher_handoff, Mapping):
        raise FormalMultiseedExecutionError(f"{context} teacher handoff is missing")
    _require_exact_keys(
        teacher_handoff,
        {
            "task_id",
            "model_id",
            "name",
            "url",
            "source_filename",
            "local_filename",
            "size_bytes",
            "sha256",
            "expected_sha256",
            "trusted_mmengine_pickle",
        },
        context=f"{context} teacher handoff",
    )
    expected_teacher = {
        "task_id": teacher["teacher_task_id"],
        "model_id": teacher["teacher_model_id"],
        "name": CLEAN_TEACHER_MODEL_NAME,
        "url": teacher["teacher_model_url"],
        "size_bytes": teacher["teacher_checkpoint_size_bytes"],
        "sha256": teacher["teacher_checkpoint_sha256"],
        "expected_sha256": teacher["teacher_checkpoint_sha256"],
        "trusted_mmengine_pickle": True,
    }
    for field, expected in expected_teacher.items():
        if (
            type(teacher_handoff.get(field)) is not type(expected)
            or teacher_handoff.get(field) != expected
        ):
            raise FormalMultiseedExecutionError(
                f"{context} teacher handoff {field} drifted"
            )
    expected_source_filename = PurePosixPath(
        unquote(urlsplit(str(teacher["teacher_model_url"])).path)
    ).name
    local_filename = teacher_handoff.get("local_filename")
    if (
        teacher_handoff.get("source_filename") != expected_source_filename
        or type(local_filename) is not str
        or not local_filename
        or PurePosixPath(local_filename).name != local_filename
    ):
        raise FormalMultiseedExecutionError(
            f"{context} teacher handoff filename drifted"
        )
    common = run_contract.get("common_teacher_initialization")
    if not isinstance(common, Mapping) or (
        common.get("contract") != COMMON_TEACHER_INITIALIZATION_CONTRACT
        or common.get("shared_prefixes") != list(COMMON_TEACHER_INITIALIZATION_PREFIXES)
        or common.get("teacher_checkpoint_sha256")
        != teacher["teacher_checkpoint_sha256"]
    ):
        raise FormalMultiseedExecutionError(f"{context} common initialization drifted")
    expected_common = {
        "policy": "shared-only",
        "contract": COMMON_TEACHER_INITIALIZATION_CONTRACT,
        "shared_prefixes": list(COMMON_TEACHER_INITIALIZATION_PREFIXES),
        "expected_source_keys": COMMON_TEACHER_SOURCE_KEYS,
        "expected_source_numel": COMMON_TEACHER_SOURCE_NUMEL,
        "expected_source_bytes": COMMON_TEACHER_SOURCE_BYTES,
        "expected_shared_keys": COMMON_TEACHER_SHARED_KEYS,
        "expected_shared_numel": COMMON_TEACHER_SHARED_NUMEL,
        "expected_shared_bytes": COMMON_TEACHER_SHARED_BYTES,
        "expected_teacher_fusion_keys": COMMON_TEACHER_FUSION_KEYS,
        "teacher_checkpoint_sha256": teacher["teacher_checkpoint_sha256"],
        "audit_artifact_name": INITIALIZATION_AUDIT_ARTIFACT,
        "audit_filename": COMMON_TEACHER_INITIALIZATION_AUDIT_FILENAME,
        "expected_nested_teacher": subject in NESTED_TEACHER_SUBJECTS,
    }
    if not isinstance(common, Mapping) or (
        _canonical_json(common) != _canonical_json(expected_common)
    ):
        raise FormalMultiseedExecutionError(
            f"{context} exact common initialization contract drifted"
        )
    checkpoint = _artifact_mapping(task, FINAL_CHECKPOINT_ARTIFACT, context=context)
    _require_exact_keys(
        checkpoint,
        {"model_id", "name", "url", "filename", "size_bytes", "sha256"},
        context=f"{context} final checkpoint contract",
    )
    model_id = _clearml_id(checkpoint.get("model_id"), f"{context} final model")
    model_name = f"ResilientV2X {record['subject']} final checkpoint"
    checkpoint_sha = _sha256(checkpoint.get("sha256"), f"{context} checkpoint")
    checkpoint_bytes = _positive_int(
        checkpoint.get("size_bytes"),
        context=f"{context} checkpoint bytes",
    )
    model_url = _require_fileserver_url(checkpoint.get("url"), context=context)
    if (
        checkpoint.get("name") != model_name
        or checkpoint.get("filename") != "epoch_50.pth"
    ):
        raise FormalMultiseedExecutionError(f"{context} model name drifted")
    model = _final_model(task, subject=str(record["subject"]), context=context)
    if (
        getattr(model, "id", None) != model_id
        or getattr(model, "task", None) != task_id
        or getattr(model, "url", None) != model_url
    ):
        raise FormalMultiseedExecutionError(f"{context} OutputModel binding drifted")
    _download_and_hash_model(
        model,
        expected_bytes=checkpoint_bytes,
        expected_sha256=checkpoint_sha,
        context=f"{context} OutputModel",
    )
    audit = _artifact_mapping(task, INITIALIZATION_AUDIT_ARTIFACT, context=context)
    if (
        type(audit.get("schema_version")) is not int
        or audit.get("schema_version") != 1
        or audit.get("contract") != COMMON_TEACHER_INITIALIZATION_CONTRACT
        or audit.get("result") != "pass"
    ):
        raise FormalMultiseedExecutionError(f"{context} initialization audit drifted")
    shared = audit.get("shared_initialization")
    if not isinstance(shared, Mapping) or (
        shared.get("prefixes") != list(COMMON_TEACHER_INITIALIZATION_PREFIXES)
        or shared.get("keys") != COMMON_TEACHER_SHARED_KEYS
        or shared.get("numel") != COMMON_TEACHER_SHARED_NUMEL
        or shared.get("bytes") != COMMON_TEACHER_SHARED_BYTES
        or shared.get("shape_dtype_verified") is not True
        or shared.get("exact_tensor_equality_verified") is not True
    ):
        raise FormalMultiseedExecutionError(f"{context} shared initialization drifted")
    audit_checkpoint = audit.get("checkpoint")
    if not isinstance(audit_checkpoint, Mapping) or (
        audit_checkpoint.get("sha256") != teacher["teacher_checkpoint_sha256"]
        or audit_checkpoint.get("expected_sha256")
        != teacher["teacher_checkpoint_sha256"]
    ):
        raise FormalMultiseedExecutionError(f"{context} audit teacher SHA drifted")
    _validate_initialization_audit(
        audit,
        subject=subject,
        teacher=teacher,
        context=context,
    )
    return {
        "task_id": task_id,
        "worker": worker,
        "model_id": model_id,
        "model_url": model_url,
        "checkpoint_size_bytes": checkpoint_bytes,
        "checkpoint_sha256": checkpoint_sha,
        "run_contract_sha256": _content_sha256(run_contract),
        "initialization_audit_sha256": _content_sha256(audit),
    }


def _evaluation_subject_contract(
    record: Mapping[str, object],
    *,
    source_root: PurePosixPath,
    context: str,
) -> tuple[str, PurePosixPath]:
    subject = record.get("subject")
    kind = record.get("kind")
    if type(subject) is not str or not subject:
        raise FormalMultiseedExecutionError(f"{context} subject is invalid")
    if kind == "external_controlled_baseline":
        return (
            "baseline",
            source_root / "configs/resilient_v2x/baselines" / f"{subject}.py",
        )
    if kind != "selected_candidate":
        raise FormalMultiseedExecutionError(f"{context} subject kind is invalid")
    if subject == "resilient_v2x":
        return (
            "primary_method",
            source_root / "configs/resilient_v2x/dair_resilient_v2x.py",
        )
    if subject not in {
        "support_residual",
        "linear_no_distillation",
        "no_distillation_peak_lr_3e4",
    }:
        raise FormalMultiseedExecutionError(
            f"{context} selected candidate is not pinned"
        )
    return (
        "improvement",
        source_root / "configs/resilient_v2x/improvements" / f"{subject}.py",
    )


def _validate_evaluation_run_contract(
    run_contract: Mapping[str, object],
    record: Mapping[str, object],
    *,
    task_id: str,
    training_task_id: str,
    training_model: Mapping[str, object],
    context: str,
) -> dict[str, str]:
    _require_exact_keys(
        run_contract,
        {
            "schema_version",
            "mode",
            "task_id",
            "baseline",
            "baseline_task_id",
            "predecessor_task_id",
            "training_dataset_id",
            "checkpoint",
            "overlay_index",
            "overlay_index_sha256",
            "expected_delays_ms",
            "expected_conditions",
            "expected_run_count",
            "protocol_id",
            "expected_sample_count",
            "expected_ground_truth_count",
            "expected_manifest_content_sha256",
            "expected_overlay_index_content_sha256",
            "expected_sample_ids_sha256",
            "command",
            "evaluator",
            "evaluator_sha256",
            "dataset_root",
        },
        context=f"{context} run contract",
    )
    subject = record.get("subject")
    if type(subject) is not str or not subject:
        raise FormalMultiseedExecutionError(f"{context} subject is invalid")
    expected_run = {
        "schema_version": 1,
        "mode": "baseline_validate",
        "task_id": task_id,
        "baseline": subject,
        "baseline_task_id": training_task_id,
        "predecessor_task_id": training_task_id,
        "training_dataset_id": TRAINING_DATASET_ID,
        "overlay_index_sha256": OVERLAY_INDEX_FILE_SHA256,
        "expected_delays_ms": list(DELAYS_MS),
        "expected_conditions": list(CONDITIONS),
        "expected_run_count": len(DELAYS_MS) * len(CONDITIONS),
        "protocol_id": PROTOCOL_ID,
        "expected_sample_count": 1337,
        "expected_ground_truth_count": 11330,
        "expected_manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "expected_overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "expected_sample_ids_sha256": SAMPLE_IDS_SHA256,
        "evaluator_sha256": CONTROLLED_EVALUATOR_SHA256,
    }
    for key, expected in expected_run.items():
        if (
            type(run_contract.get(key)) is not type(expected)
            or run_contract.get(key) != expected
        ):
            raise FormalMultiseedExecutionError(f"{context} run contract {key} drifted")

    evaluator = _canonical_absolute_posix_path(
        run_contract.get("evaluator"),
        context=f"{context} evaluator",
    )
    if len(evaluator.parents) < 3:
        raise FormalMultiseedExecutionError(f"{context} evaluator path is too shallow")
    source_root = evaluator.parents[2]
    if evaluator != (
        source_root / "tools/resilient_v2x/evaluate_controlled_baselines.py"
    ):
        raise FormalMultiseedExecutionError(f"{context} evaluator path drifted")

    dataset_root = _canonical_absolute_posix_path(
        run_contract.get("dataset_root"),
        context=f"{context} dataset root",
    )
    expected_dataset_root = (
        source_root / "work_dirs/clearml_dataset_materialization" / task_id / "training"
    )
    if dataset_root != expected_dataset_root:
        raise FormalMultiseedExecutionError(f"{context} dataset root drifted")
    overlay_index = _canonical_absolute_posix_path(
        run_contract.get("overlay_index"),
        context=f"{context} overlay index",
    )
    if overlay_index != (dataset_root / "protocols/dair_v2/evaluation_overlays.json"):
        raise FormalMultiseedExecutionError(f"{context} overlay index path drifted")
    work_dir = (
        source_root / "work_dirs/controlled_baseline_evaluation" / task_id / subject
    )

    command = run_contract.get("command")
    if (
        type(command) is not list
        or len(command) != 14
        or any(type(item) is not str or not item for item in command)
    ):
        raise FormalMultiseedExecutionError(f"{context} evaluation command is invalid")
    checkpoint_path = _canonical_absolute_posix_path(
        command[5],
        context=f"{context} evaluation checkpoint",
    )
    expected_command = [
        RUNTIME_PYTHON,
        evaluator.as_posix(),
        "--baseline",
        subject,
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--overlay-index",
        overlay_index.as_posix(),
        "--work-dir",
        work_dir.as_posix(),
        "--protocol-id",
        PROTOCOL_ID,
        "--expected-ground-truth-count",
        "11330",
    ]
    if command != expected_command:
        raise FormalMultiseedExecutionError(f"{context} evaluation command drifted")

    checkpoint = run_contract.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise FormalMultiseedExecutionError(f"{context} checkpoint handoff is missing")
    _require_exact_keys(
        checkpoint,
        {
            "task_id",
            "model_id",
            "name",
            "url",
            "source_filename",
            "local_filename",
            "size_bytes",
            "sha256",
            "expected_sha256",
            "trusted_mmengine_pickle",
        },
        context=f"{context} checkpoint handoff",
    )
    model_url = _require_fileserver_url(
        training_model.get("model_url"),
        context=f"{context} training model",
    )
    source_filename = PurePosixPath(unquote(urlsplit(model_url).path)).name
    expected_checkpoint = {
        "task_id": training_task_id,
        "model_id": training_model["model_id"],
        "name": f"ResilientV2X {subject} final checkpoint",
        "url": model_url,
        "source_filename": source_filename,
        "local_filename": checkpoint_path.name,
        "size_bytes": training_model["checkpoint_size_bytes"],
        "sha256": training_model["checkpoint_sha256"],
        "expected_sha256": training_model["checkpoint_sha256"],
        "trusted_mmengine_pickle": True,
    }
    for key, expected in expected_checkpoint.items():
        if (
            type(checkpoint.get(key)) is not type(expected)
            or checkpoint.get(key) != expected
        ):
            raise FormalMultiseedExecutionError(
                f"{context} checkpoint handoff {key} drifted"
            )
    return {
        "source_root": source_root.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "overlay_index": overlay_index.as_posix(),
        "work_dir": work_dir.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "evaluator": evaluator.as_posix(),
    }


def _validate_box_rows(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise FormalMultiseedExecutionError(f"{context} must be a list")
    for index, row in enumerate(value):
        if (
            not isinstance(row, list)
            or len(row) != 7
            or any(
                type(coordinate) is not float or not math.isfinite(coordinate)
                for coordinate in row
            )
            or any(row[axis] <= 0.0 for axis in (3, 4, 5))
        ):
            raise FormalMultiseedExecutionError(
                f"{context}[{index}] must contain seven finite numbers"
            )
    return value


def _expected_prediction_diagnostic_method(
    record: Mapping[str, object],
    *,
    context: str,
) -> str:
    subject = record.get("subject")
    kind = record.get("kind")
    candidates = {
        "resilient_v2x",
        "support_residual",
        "linear_no_distillation",
        "no_distillation_peak_lr_3e4",
    }
    if type(subject) is not str or subject not in SUBJECT_CONFIG_SHA256:
        raise FormalMultiseedExecutionError(
            f"{context} diagnostic subject is not pinned"
        )
    if kind == "external_controlled_baseline":
        if subject in candidates:
            raise FormalMultiseedExecutionError(
                f"{context} controlled diagnostic subject drifted"
            )
        return subject
    if kind != "selected_candidate" or subject not in candidates:
        raise FormalMultiseedExecutionError(
            f"{context} ResilientV2X diagnostic subject drifted"
        )
    ptf_mode = "linear" if subject == "linear_no_distillation" else "nonlinear"
    return (
        f"ResilientV2X(ptf={ptf_mode},routing=dynamic,reliability=1,delay_metadata=1)"
    )


def _validate_resilient_prediction_branch(
    value: object,
    *,
    expected_agent: str,
    expected_modality: str,
    index: int,
    context: str,
) -> bool:
    fields = {
        "agent",
        "modality",
        "supported",
        "source_tick",
        "source_tau_ms",
        "horizon",
        "observed",
        "propagated",
        "gamma",
        "reliability",
        "ptf_queried",
        "reason",
    }
    reasons = {
        "empty_arrival_set",
        "empty_modality_history",
        "unsupported_horizon",
        "invalid_timestamp",
        "invalid_pose",
        "invalid_calibration",
        "missing_payload",
        "method_not_applicable",
    }
    if not isinstance(value, Mapping):
        raise FormalMultiseedExecutionError(
            f"{context} ResilientV2X branch {index} drifted"
        )
    _require_exact_keys(
        value,
        fields,
        context=f"{context} ResilientV2X branch {index}",
    )
    supported = value.get("supported")
    observed = value.get("observed")
    propagated = value.get("propagated")
    ptf_queried = value.get("ptf_queried")
    reliability = value.get("reliability")
    if (
        value.get("agent") != expected_agent
        or value.get("modality") != expected_modality
        or type(supported) is not bool
        or type(observed) is not bool
        or type(propagated) is not bool
        or type(ptf_queried) is not bool
        or type(reliability) is not float
        or not math.isfinite(reliability)
        or not 0.0 <= reliability <= 1.0
    ):
        raise FormalMultiseedExecutionError(
            f"{context} ResilientV2X branch {index} identity/type drifted"
        )
    if not supported:
        if (
            value.get("source_tick") is not None
            or value.get("source_tau_ms") is not None
            or value.get("horizon") is not None
            or observed
            or propagated
            or value.get("gamma") is not None
            or reliability != 0.0
            or ptf_queried
            or value.get("reason") not in reasons
        ):
            raise FormalMultiseedExecutionError(
                f"{context} unsupported ResilientV2X branch {index} drifted"
            )
        return False

    source_tick = value.get("source_tick")
    source_tau_ms = value.get("source_tau_ms")
    horizon = value.get("horizon")
    gamma = value.get("gamma")
    if (
        type(source_tick) is not int
        or source_tick < 0
        or type(source_tau_ms) is not int
        or type(horizon) is not int
        or not 0 <= horizon <= 3
        or observed == propagated
        or ptf_queried is not True
        or type(gamma) is not float
        or not math.isfinite(gamma)
        or not 0.0 <= gamma <= 1.0
        or value.get("reason") is not None
    ):
        raise FormalMultiseedExecutionError(
            f"{context} supported ResilientV2X branch {index} drifted"
        )
    if expected_agent == "ego":
        expected_observed = horizon == 0
        if observed is not expected_observed or propagated is expected_observed:
            raise FormalMultiseedExecutionError(
                f"{context} Ego branch {index} timing drifted"
            )
    elif horizon == 0 and (not observed or propagated):
        raise FormalMultiseedExecutionError(
            f"{context} RSU branch {index} timing drifted"
        )
    return True


def _validate_prediction_diagnostic(
    value: object,
    *,
    sample_id: str,
    controlled_baseline: bool,
    expected_method: str,
    context: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or type(value.get("method")) is not str
        or not value.get("method")
        or value.get("overall_supported") is not True
    ):
        raise FormalMultiseedExecutionError(f"{context} diagnostic drifted")
    if controlled_baseline:
        _require_exact_keys(
            value,
            {
                "diagnostic_type",
                "schema_version",
                "sample_id",
                "method",
                "overall_supported",
                "branch_order",
                "support",
                "age_intervals",
            },
            context=f"{context} controlled baseline diagnostic",
        )
        branch_order = ("lidar_ego", "lidar_rsu", "camera_ego", "camera_rsu")
        if (
            value.get("diagnostic_type") != "controlled_baseline"
            or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1
            or value.get("sample_id") != sample_id
            or value.get("method") != expected_method
            or value.get("branch_order") != list(branch_order)
        ):
            raise FormalMultiseedExecutionError(
                f"{context} controlled baseline diagnostic identity drifted"
            )
        support = value.get("support")
        ages = value.get("age_intervals")
        if (
            not isinstance(support, Mapping)
            or set(support) != set(branch_order)
            or not isinstance(ages, Mapping)
            or set(ages) != set(branch_order)
        ):
            raise FormalMultiseedExecutionError(
                f"{context} controlled baseline branch contract drifted"
            )
        for branch in branch_order:
            supported = support[branch]
            age = ages[branch]
            if type(supported) is not bool:
                raise FormalMultiseedExecutionError(
                    f"{context} controlled baseline support drifted"
                )
            if supported:
                if type(age) is not float or not math.isfinite(age) or age < 0.0:
                    raise FormalMultiseedExecutionError(
                        f"{context} controlled baseline age drifted"
                    )
            elif age is not None:
                raise FormalMultiseedExecutionError(
                    f"{context} unsupported controlled baseline age drifted"
                )
        if not any(support.values()):
            raise FormalMultiseedExecutionError(
                f"{context} controlled baseline support is empty"
            )
        return

    _require_exact_keys(
        value,
        {"method", "overall_supported", "branches", "routing"},
        context=f"{context} ResilientV2X diagnostic",
    )
    if value.get("method") != expected_method:
        raise FormalMultiseedExecutionError(
            f"{context} ResilientV2X diagnostic method drifted"
        )

    routing = value.get("routing")
    if not isinstance(routing, Mapping):
        raise FormalMultiseedExecutionError(f"{context} ResilientV2X routing drifted")
    _require_exact_keys(
        routing,
        {"expert_support", "weights", "not_applicable_reason"},
        context=f"{context} ResilientV2X routing",
    )
    expert_support = routing.get("expert_support")
    weights = routing.get("weights")
    if (
        not isinstance(expert_support, list)
        or len(expert_support) != 3
        or any(type(item) is not bool for item in expert_support)
        or not isinstance(weights, list)
        or len(weights) != 3
        or any(
            type(weight) is not float
            or not math.isfinite(weight)
            or not 0.0 <= weight <= 1.0
            for weight in weights
        )
        or routing.get("not_applicable_reason") is not None
        or not any(expert_support)
        or any(
            not supported and weight != 0.0
            for supported, weight in zip(
                expert_support,
                weights,
                strict=True,
            )
        )
        or not math.isclose(
            sum(weights),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    ):
        raise FormalMultiseedExecutionError(
            f"{context} ResilientV2X routing values drifted"
        )
    branch_order = (
        ("ego", "lidar"),
        ("rsu", "lidar"),
        ("ego", "camera"),
        ("rsu", "camera"),
    )
    branches = value.get("branches")
    if not isinstance(branches, list) or len(branches) != len(branch_order):
        raise FormalMultiseedExecutionError(f"{context} ResilientV2X branches drifted")
    supported_branches = []
    for index, (branch, (agent, modality)) in enumerate(
        zip(branches, branch_order, strict=True)
    ):
        supported_branches.append(
            _validate_resilient_prediction_branch(
                branch,
                expected_agent=agent,
                expected_modality=modality,
                index=index,
                context=context,
            )
        )
    if not any(supported_branches):
        raise FormalMultiseedExecutionError(
            f"{context} ResilientV2X branch support is empty"
        )


def _validate_prediction_document(
    path: Path,
    run: Mapping[str, object],
    *,
    expected_sample_ids: list[str],
    expected_prediction_count: int,
    controlled_baseline: bool,
    expected_method: str,
    context: str,
) -> dict[str, str]:
    observed_file_sha256 = _file_sha256(path, context=context)
    expected_file_sha256 = _sha256(
        run.get("prediction_sha256"),
        f"{context} file",
    )
    if observed_file_sha256 != expected_file_sha256:
        raise FormalMultiseedExecutionError(f"{context} file SHA-256 mismatch")
    document = _read_json_mapping_file(path, context=context)
    _require_exact_keys(
        document,
        {
            "schema_version",
            "document_type",
            "coordinate_convention",
            "point_cloud_range",
            "iou_thresholds",
            "max_detections",
            "sample_count",
            "samples",
            "content_sha256",
        },
        context=context,
    )
    observed_content_sha256 = _require_producer_content_hash(
        document,
        field="content_sha256",
        context=context,
    )
    expected_content_sha256 = _sha256(
        run.get("prediction_content_sha256"),
        f"{context} content",
    )
    if observed_content_sha256 != expected_content_sha256:
        raise FormalMultiseedExecutionError(f"{context} content SHA-256 drifted")
    expected_document = {
        "schema_version": 1,
        "document_type": "resilient_v2x_predictions",
        "coordinate_convention": (
            "[x,y,z_bottom,length,width,height,yaw] in current ego LiDAR"
        ),
        "point_cloud_range": [0.0, -40.0, -3.0, 80.0, 40.0, 1.0],
        "iou_thresholds": [0.5, 0.7],
        "max_detections": 100,
        "sample_count": 1337,
    }
    for key, expected in expected_document.items():
        if (
            type(document.get(key)) is not type(expected)
            or document.get(key) != expected
        ):
            raise FormalMultiseedExecutionError(f"{context} {key} drifted")
    samples = document.get("samples")
    if not isinstance(samples, list) or len(samples) != 1337:
        raise FormalMultiseedExecutionError(
            f"{context} must contain exactly 1337 samples"
        )
    sample_fields = {
        "sample_id",
        "predicted_boxes_lidar_bottom_center",
        "predicted_scores",
        "predicted_labels",
        "ground_truth_boxes_lidar_bottom_center",
        "ground_truth_labels",
        "diagnostic",
    }
    observed_ids: list[str] = []
    ground_truth_records: list[dict[str, object]] = []
    ground_truth_count = 0
    prediction_count = 0
    for index, (sample, expected_id) in enumerate(
        zip(samples, expected_sample_ids, strict=True)
    ):
        if not isinstance(sample, Mapping):
            raise FormalMultiseedExecutionError(
                f"{context} sample {index} is not an object"
            )
        _require_exact_keys(
            sample,
            sample_fields,
            context=f"{context} sample {index}",
        )
        sample_id = sample.get("sample_id")
        if type(sample_id) is not str or sample_id != expected_id:
            raise FormalMultiseedExecutionError(
                f"{context} sample {index} identity drifted"
            )
        observed_ids.append(sample_id)
        predicted_boxes = _validate_box_rows(
            sample.get("predicted_boxes_lidar_bottom_center"),
            context=f"{context} sample {index} predicted boxes",
        )
        ground_truth_boxes = _validate_box_rows(
            sample.get("ground_truth_boxes_lidar_bottom_center"),
            context=f"{context} sample {index} ground truth boxes",
        )
        predicted_scores = sample.get("predicted_scores")
        predicted_labels = sample.get("predicted_labels")
        ground_truth_labels = sample.get("ground_truth_labels")
        if (
            not isinstance(predicted_scores, list)
            or not isinstance(predicted_labels, list)
            or not isinstance(ground_truth_labels, list)
            or len(predicted_boxes) != len(predicted_scores)
            or len(predicted_boxes) != len(predicted_labels)
            or len(predicted_boxes) > 100
            or len(ground_truth_boxes) != len(ground_truth_labels)
            or any(
                type(score) is not float
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
                for score in predicted_scores
            )
            or any(
                type(label) is not int or label < 0
                for label in [*predicted_labels, *ground_truth_labels]
            )
        ):
            raise FormalMultiseedExecutionError(
                f"{context} sample {index} arrays drifted"
            )
        prediction_count += len(predicted_boxes)
        ground_truth_count += len(ground_truth_boxes)

        ground_truth_records.append(
            {
                "sample_id": sample_id,
                "ground_truth_boxes_lidar_bottom_center": ground_truth_boxes,
                "ground_truth_labels": ground_truth_labels,
            }
        )
        _validate_prediction_diagnostic(
            sample.get("diagnostic"),
            sample_id=sample_id,
            controlled_baseline=controlled_baseline,
            expected_method=expected_method,
            context=f"{context} sample {index}",
        )
    if (
        observed_ids != expected_sample_ids
        or _producer_content_sha256(observed_ids) != SAMPLE_IDS_SHA256
    ):
        raise FormalMultiseedExecutionError(f"{context} sample cohort drifted")
    if ground_truth_count != 11330:
        raise FormalMultiseedExecutionError(
            f"{context} ground-truth count must be exactly 11330"
        )
    ground_truth_content_sha256 = _producer_content_sha256(ground_truth_records)
    if ground_truth_content_sha256 != GROUND_TRUTH_CONTENT_SHA256:
        raise FormalMultiseedExecutionError(
            f"{context} fixed ground-truth content drifted"
        )
    if prediction_count != expected_prediction_count:
        raise FormalMultiseedExecutionError(
            f"{context} prediction count disagrees with metrics"
        )
    return {
        "file_sha256": observed_file_sha256,
        "content_sha256": observed_content_sha256,
        "ground_truth_content_sha256": ground_truth_content_sha256,
    }


def _validate_evaluation_evidence_at_root(
    record: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    evidence_root: Path,
    paths: Mapping[str, str],
    context: str,
) -> dict[str, object]:
    work_dir = paths["work_dir"]
    archived_plan_path = _evidence_file(
        evidence_root,
        work_dir,
        f"{work_dir}/evaluation_plan.json",
        context=f"{context} archived evaluation plan",
    )
    archived_plan = _read_json_mapping_file(
        archived_plan_path,
        context=f"{context} archived evaluation plan",
    )
    if _canonical_json(archived_plan) != _canonical_json(plan):
        raise FormalMultiseedExecutionError(
            f"{context} evaluation plan artifact disagrees with evidence"
        )
    archived_metrics_path = _evidence_file(
        evidence_root,
        work_dir,
        f"{work_dir}/metrics.json",
        context=f"{context} archived metrics",
    )
    archived_metrics = _read_json_mapping_file(
        archived_metrics_path,
        context=f"{context} archived metrics",
    )
    if _canonical_json(archived_metrics) != _canonical_json(metrics):
        raise FormalMultiseedExecutionError(
            f"{context} metrics artifact disagrees with evidence"
        )

    _require_exact_keys(
        plan,
        {
            "schema_version",
            "plan_type",
            "protocol_id",
            "baseline",
            "baseline_config",
            "baseline_config_sha256",
            "evaluation_subject_type",
            "checkpoint",
            "checkpoint_sha256",
            "data_root",
            "split_sha256",
            "manifest",
            "manifest_content_sha256",
            "manifest_file_sha256",
            "overlay_index",
            "overlay_index_content_sha256",
            "overlay_index_file_sha256",
            "sample_ids",
            "sample_ids_sha256",
            "expected_sample_count",
            "expected_ground_truth_count",
            "expected_unsupported_sample_count",
            "work_dir",
            "delays_ms",
            "conditions",
            "runs",
            "metrics_output",
            "plan_path",
            "content_sha256",
        },
        context=f"{context} evaluation plan",
    )
    plan_content_sha256 = _require_producer_content_hash(
        plan,
        field="content_sha256",
        context=f"{context} evaluation plan",
    )
    source_root = _canonical_absolute_posix_path(
        paths["source_root"],
        context=f"{context} source root",
    )
    subject_type, baseline_config = _evaluation_subject_contract(
        record,
        source_root=source_root,
        context=context,
    )
    dataset_root = PurePosixPath(paths["dataset_root"])
    expected_plan = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_evaluation",
        "protocol_id": PROTOCOL_ID,
        "baseline": record["subject"],
        "baseline_config": baseline_config.as_posix(),
        "evaluation_subject_type": subject_type,
        "checkpoint": paths["checkpoint"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "data_root": (dataset_root / "cooperative-vehicle-infrastructure").as_posix(),
        "split_sha256": OFFICIAL_SPLIT_SHA256,
        "manifest": (dataset_root / "manifests/temporal_manifest_v2.json").as_posix(),
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "overlay_index": paths["overlay_index"],
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "overlay_index_file_sha256": OVERLAY_INDEX_FILE_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "expected_sample_count": 1337,
        "expected_ground_truth_count": 11330,
        "expected_unsupported_sample_count": 0,
        "work_dir": work_dir,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "metrics_output": f"{work_dir}/metrics.json",
        "plan_path": f"{work_dir}/evaluation_plan.json",
    }
    for key, expected in expected_plan.items():
        if type(plan.get(key)) is not type(expected) or plan.get(key) != expected:
            raise FormalMultiseedExecutionError(
                f"{context} evaluation plan {key} drifted"
            )
    baseline_config_sha = _sha256(
        plan.get("baseline_config_sha256"),
        f"{context} baseline config",
    )
    expected_config_sha = SUBJECT_CONFIG_SHA256[str(record["subject"])]
    if baseline_config_sha != expected_config_sha:
        raise FormalMultiseedExecutionError(
            f"{context} baseline config SHA-256 drifted"
        )
    sample_ids = plan.get("sample_ids")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != 1337
        or len(set(sample_ids)) != 1337
        or any(type(sample_id) is not str or not sample_id for sample_id in sample_ids)
        or _producer_content_sha256(sample_ids) != SAMPLE_IDS_SHA256
    ):
        raise FormalMultiseedExecutionError(
            f"{context} evaluation plan sample cohort drifted"
        )

    expected_condition_ids = [
        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        for delay in DELAYS_MS
        for condition in CONDITIONS
    ]
    if record.get("condition_ids") != expected_condition_ids:
        raise FormalMultiseedExecutionError(f"{context} planned condition IDs drifted")
    plan_runs = plan.get("runs")
    metric_runs = metrics.get("runs")
    if (
        not isinstance(plan_runs, list)
        or not isinstance(metric_runs, list)
        or len(plan_runs) != 12
        or len(metric_runs) != 12
    ):
        raise FormalMultiseedExecutionError(
            f"{context} evaluation evidence matrix is incomplete"
        )
    transport_by_delay: dict[int, tuple[str, str]] = {}
    prediction_receipts: list[dict[str, str]] = []
    expected_diagnostic_method = _expected_prediction_diagnostic_method(
        record,
        context=context,
    )
    for index, (plan_run, metric_run, condition_id) in enumerate(
        zip(plan_runs, metric_runs, expected_condition_ids, strict=True)
    ):
        if not isinstance(plan_run, Mapping) or not isinstance(metric_run, Mapping):
            raise FormalMultiseedExecutionError(
                f"{context} evidence run {index} is invalid"
            )
        _require_exact_keys(
            plan_run,
            {
                "condition_id",
                "delay_ms",
                "condition",
                "agent_scope",
                "duration_ticks",
                "condition_config",
                "resolved_config",
                "resolved_config_sha256",
                "predictions",
                "checkpoint_sha256_file",
                "transport_overlay",
                "transport_overlay_sha256",
                "fault_overlay",
                "fault_overlay_sha256",
            },
            context=f"{context} evaluation plan run {index}",
        )
        delay = DELAYS_MS[index // len(CONDITIONS)]
        condition = CONDITIONS[index % len(CONDITIONS)]
        expected_condition_filename = (
            f"global_delay_{delay:03d}_full.py"
            if condition == "Full"
            else (f"causal_delay_{delay:03d}_{condition.lower().replace('-', '_')}.py")
        )
        expected_run = {
            "condition_id": condition_id,
            "delay_ms": delay,
            "condition": condition,
            "agent_scope": "E+R",
            "duration_ticks": 1,
            "condition_config": (
                source_root
                / "configs/resilient_v2x/conditions"
                / expected_condition_filename
            ).as_posix(),
            "resolved_config": f"{work_dir}/{condition_id}/resolved_config.py",
            "predictions": f"{work_dir}/{condition_id}/predictions.json",
            "checkpoint_sha256_file": (f"{work_dir}/{condition_id}/checkpoint.sha256"),
        }
        for key, expected in expected_run.items():
            if (
                type(plan_run.get(key)) is not type(expected)
                or plan_run.get(key) != expected
            ):
                raise FormalMultiseedExecutionError(
                    f"{context} evaluation plan run {index} {key} drifted"
                )
        if plan_run["predictions"] != metric_run.get("predictions"):
            raise FormalMultiseedExecutionError(
                f"{context} run {index} prediction path is not cross-bound"
            )

        resolved_config = _evidence_file(
            evidence_root,
            work_dir,
            plan_run["resolved_config"],
            context=f"{context} run {index} resolved config",
        )
        expected_resolved_sha256 = _sha256(
            plan_run.get("resolved_config_sha256"),
            f"{context} run {index} resolved config",
        )
        if (
            _file_sha256(
                resolved_config,
                context=f"{context} run {index} resolved config",
            )
            != expected_resolved_sha256
        ):
            raise FormalMultiseedExecutionError(
                f"{context} run {index} resolved config SHA-256 mismatch"
            )
        checkpoint_sha_file = _evidence_file(
            evidence_root,
            work_dir,
            plan_run["checkpoint_sha256_file"],
            context=f"{context} run {index} checkpoint SHA file",
        )
        try:
            checkpoint_sha_text = checkpoint_sha_file.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise FormalMultiseedExecutionError(
                f"{context} run {index} checkpoint SHA file is unreadable"
            ) from error
        if checkpoint_sha_text != f"{metrics['checkpoint_sha256']}\n":
            raise FormalMultiseedExecutionError(
                f"{context} run {index} checkpoint SHA file drifted"
            )

        overlay_dir = PurePosixPath(paths["overlay_index"]).parent
        expected_transport = overlay_dir / f"val_transport_delay_{delay:03d}.jsonl.zst"
        transport = _canonical_absolute_posix_path(
            plan_run.get("transport_overlay"),
            context=f"{context} run {index} transport overlay",
        )
        transport_sha = _sha256(
            plan_run.get("transport_overlay_sha256"),
            f"{context} run {index} transport overlay",
        )
        if (
            transport != expected_transport
            or transport_sha != TRANSPORT_OVERLAY_SHA256[delay]
        ):
            raise FormalMultiseedExecutionError(
                f"{context} run {index} transport overlay path drifted"
            )
        previous_transport = transport_by_delay.setdefault(
            delay,
            (transport.as_posix(), transport_sha),
        )
        if previous_transport != (transport.as_posix(), transport_sha):
            raise FormalMultiseedExecutionError(
                f"{context} delay {delay} transport identity drifted"
            )
        if condition == "Full":
            if (
                plan_run.get("fault_overlay") is not None
                or plan_run.get("fault_overlay_sha256") is not None
            ):
                raise FormalMultiseedExecutionError(
                    f"{context} run {index} Full condition has a fault overlay"
                )
        else:
            expected_fault = overlay_dir / (
                f"val_causal_delay_{delay:03d}_"
                f"{condition.lower().replace('-', '_')}.jsonl.zst"
            )
            fault = _canonical_absolute_posix_path(
                plan_run.get("fault_overlay"),
                context=f"{context} run {index} fault overlay",
            )
            fault_sha = _sha256(
                plan_run.get("fault_overlay_sha256"),
                f"{context} run {index} fault overlay",
            )
            if (
                fault != expected_fault
                or fault_sha != FAULT_OVERLAY_SHA256[(delay, condition)]
            ):
                raise FormalMultiseedExecutionError(
                    f"{context} run {index} fault overlay path drifted"
                )

        prediction_path = _evidence_file(
            evidence_root,
            work_dir,
            plan_run["predictions"],
            context=f"{context} run {index} predictions",
        )
        prediction_count = int(
            metric_run["metrics"]["resilient_v2x/car_prediction_count"]
        )
        receipt = _validate_prediction_document(
            prediction_path,
            metric_run,
            expected_sample_ids=sample_ids,
            controlled_baseline=(record.get("kind") == "external_controlled_baseline"),
            expected_method=expected_diagnostic_method,
            expected_prediction_count=prediction_count,
            context=f"{context} run {index} predictions",
        )
        prediction_receipts.append(
            {
                "condition_id": condition_id,
                **receipt,
            }
        )
    return {
        "evaluation_plan_content_sha256": plan_content_sha256,
        "evaluation_plan_file_sha256": _file_sha256(
            archived_plan_path,
            context=f"{context} archived evaluation plan",
        ),
        "metrics_file_sha256": _file_sha256(
            archived_metrics_path,
            context=f"{context} archived metrics",
        ),
        "prediction_bundle_sha256": _content_sha256(prediction_receipts),
    }


def _validate_evaluation_evidence(
    task: object,
    record: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    paths: Mapping[str, str],
    context: str,
) -> dict[str, object]:
    plan = _artifact_mapping(
        task,
        EVALUATION_PLAN_ARTIFACT,
        context=context,
    )
    with _evaluation_evidence_root(task, context=context) as evidence_root:
        return _validate_evaluation_evidence_at_root(
            record,
            metrics,
            plan=plan,
            evidence_root=evidence_root,
            paths=paths,
            context=context,
        )


def _validate_evaluation_completion(
    task: object,
    record: Mapping[str, object],
    *,
    training_task_id: str,
    training_model: Mapping[str, object],
) -> dict[str, object]:
    context = f"evaluation task {record['task_key']}"
    task_id = _clearml_id(getattr(task, "id", None), context)
    if _status(task, context=context) != "completed":
        raise FormalMultiseedExecutionError(f"{context} is not completed")
    worker = _require_worker(task, record.get("gpu_model"), context=context)
    outputs = _models(task, context=context).get("output", [])
    if (
        not isinstance(outputs, Sequence)
        or isinstance(outputs, (str, bytes))
        or outputs
    ):
        raise FormalMultiseedExecutionError(
            f"{context} unexpectedly produced output models"
        )
    if _artifact_names(task, context=context) != tuple(sorted(EVALUATION_ARTIFACTS)):
        raise FormalMultiseedExecutionError(
            f"{context} exact four-artifact inventory drifted"
        )
    run_contract = _artifact_mapping(task, RUN_CONTRACT_ARTIFACT, context=context)
    paths = _validate_evaluation_run_contract(
        run_contract,
        record,
        task_id=task_id,
        training_task_id=training_task_id,
        training_model=training_model,
        context=context,
    )
    metrics = _artifact_mapping(task, METRICS_ARTIFACT, context=context)
    _require_exact_keys(
        metrics,
        {
            "schema_version",
            "result_type",
            "complete",
            "planned_run_count",
            "baseline",
            "protocol_id",
            "checkpoint",
            "checkpoint_sha256",
            "manifest_content_sha256",
            "overlay_index_content_sha256",
            "sample_ids_sha256",
            "expected_sample_count",
            "expected_ground_truth_count",
            "expected_unsupported_sample_count",
            "runs",
        },
        context=f"{context} metrics",
    )
    if (
        type(metrics.get("checkpoint")) is not str
        or metrics["checkpoint"] != paths["checkpoint"]
    ):
        raise FormalMultiseedExecutionError(f"{context} metrics checkpoint drifted")
    expected_metrics = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": 12,
        "baseline": record["subject"],
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": training_model["checkpoint_sha256"],
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "expected_sample_count": 1337,
        "expected_ground_truth_count": 11330,
        "expected_unsupported_sample_count": 0,
    }
    for key, expected in expected_metrics.items():
        if type(metrics.get(key)) is not type(expected) or metrics.get(key) != expected:
            raise FormalMultiseedExecutionError(f"{context} metrics {key} drifted")
    runs = metrics.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise FormalMultiseedExecutionError(
            f"{context} metrics must contain exactly 12 runs"
        )
    expected_ids = record.get("condition_ids")
    if not isinstance(expected_ids, list) or len(expected_ids) != 12:
        raise FormalMultiseedExecutionError(f"{context} planned conditions are invalid")
    normalized_runs: list[dict[str, object]] = []
    for index, (run, condition_id) in enumerate(zip(runs, expected_ids)):
        if not isinstance(run, Mapping):
            raise FormalMultiseedExecutionError(
                f"{context} run {index} is not an object"
            )
        _require_exact_keys(
            run,
            {
                "condition_id",
                "delay_ms",
                "condition",
                "metrics",
                "predictions",
                "prediction_sha256",
                "prediction_content_sha256",
                "sample_count",
                "sample_ids_sha256",
                "ground_truth_count",
                "unsupported_sample_count",
            },
            context=f"{context} run {index}",
        )
        if type(run.get("predictions")) is not str or not run["predictions"]:
            raise FormalMultiseedExecutionError(
                f"{context} run {index} predictions path is invalid"
            )
        _sha256(
            run.get("prediction_sha256"),
            f"{context} run {index} prediction",
        )
        _sha256(
            run.get("prediction_content_sha256"),
            f"{context} run {index} prediction content",
        )
        delay = DELAYS_MS[index // len(CONDITIONS)]
        condition = CONDITIONS[index % len(CONDITIONS)]
        expected_evidence = {
            "condition_id": condition_id,
            "delay_ms": delay,
            "condition": condition,
            "sample_count": 1337,
            "ground_truth_count": 11330,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
            "unsupported_sample_count": 0,
        }
        for key, expected in expected_evidence.items():
            if type(run.get(key)) is not type(expected) or run.get(key) != expected:
                raise FormalMultiseedExecutionError(
                    f"{context} run {index} {key} drifted"
                )
        values = run.get("metrics")
        if not isinstance(values, Mapping):
            raise FormalMultiseedExecutionError(
                f"{context} run {index} metrics are missing"
            )
        _require_exact_keys(
            values,
            set(EVALUATION_METRIC_KEYS),
            context=f"{context} run {index} metrics",
        )

        metric_context = f"{context} run {index} metrics"
        for metric_key, expected in (
            ("resilient_v2x/sample_count", 1337),
            ("resilient_v2x/car_ground_truth_count", 11330),
            ("resilient_v2x/unsupported_sample_count", 0),
        ):
            _count_metric_float(
                values.get(metric_key),
                expected=expected,
                context=f"{metric_context} {metric_key}",
            )
        prediction_count = _count_metric_float(
            values.get("resilient_v2x/car_prediction_count"),
            maximum=1337 * 100,
            context=f"{metric_context} car_prediction_count",
        )
        for metric_key in DIAGNOSTIC_METRIC_KEYS[:2]:
            _finite_metric_float(
                values.get(metric_key),
                context=f"{metric_context} {metric_key}",
            )
        for metric_key in DIAGNOSTIC_METRIC_KEYS[2:4]:
            _nonnegative_metric_float(
                values.get(metric_key),
                context=f"{metric_context} {metric_key}",
            )
        _count_metric_float(
            values.get(DIAGNOSTIC_METRIC_KEYS[4]),
            maximum=prediction_count,
            context=f"{metric_context} {DIAGNOSTIC_METRIC_KEYS[4]}",
        )
        _nonnegative_metric_float(
            values.get(DIAGNOSTIC_METRIC_KEYS[5]),
            context=f"{metric_context} {DIAGNOSTIC_METRIC_KEYS[5]}",
        )
        for metric_key in DIAGNOSTIC_METRIC_KEYS[6:]:
            _unit_metric_float(
                values.get(metric_key),
                context=f"{metric_context} {metric_key}",
            )
        ap = {}
        for key in AP_METRIC_KEYS:
            value = _finite_metric_float(
                values.get(key),
                context=f"{metric_context} {key}",
            )
            ap[key] = _finite_ap(
                value,
                context=f"{metric_context} {key}",
            )
        normalized_runs.append(
            {
                "condition_id": condition_id,
                "delay_ms": delay,
                "condition": condition,
                "metrics": ap,
            }
        )
    evidence = _validate_evaluation_evidence(
        task,
        record,
        metrics,
        paths=paths,
        context=context,
    )
    return {
        "task_id": task_id,
        "worker": worker,
        "run_contract_sha256": _content_sha256(run_contract),
        "metrics_sha256": _content_sha256(metrics),
        **evidence,
        "runs": normalized_runs,
    }


def _task_receipt(
    record: Mapping[str, object],
    result: Mapping[str, object],
    *,
    parent_task_id: str,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    return {
        "task_key": record["task_key"],
        "task_id": result["task_id"],
        "parent_task_id": parent_task_id,
        "subject": record["subject"],
        "kind": record["kind"],
        "seed_index": record["seed_index"],
        "training_seed": record["training_seed"],
        "training_overlay_protocol_seed": record["training_overlay_protocol_seed"],
        "gpu_model": record["gpu_model"],
        "worker_queue": record["worker_queue"],
        "queue_id": QUEUE_IDS[str(record["worker_queue"])],
        "queue_pool_slot": record["queue_pool_slot"],
        "worker": result["worker"],
        "parameters": _frozen_mapping(parameters, context="receipt parameters"),
        "parameters_sha256": _content_sha256(parameters),
        "result": _frozen_mapping(result, context="task result"),
    }


def _start_formal_task(
    task_class: object,
    *,
    teacher_task: object,
    teacher_script: Mapping[str, object],
    teacher_parameters: Mapping[str, object],
    source_d: str,
    record: Mapping[str, object],
    parent_task_id: str,
    executor_task_id: str,
    expected_parameters: Mapping[str, object],
    project: str,
    used_ids: set[str],
) -> object:
    task = _clone_or_resume_created(
        task_class,
        teacher_task=teacher_task,
        teacher_script=teacher_script,
        teacher_parameters=teacher_parameters,
        source_d=source_d,
        record=record,
        parent_task_id=parent_task_id,
        executor_task_id=executor_task_id,
        expected_parameters=expected_parameters,
        project=project,
        used_ids=used_ids,
    )
    task_id = _clearml_id(getattr(task, "id", None), "formal task")
    expected_name = _task_run_name(record, executor_task_id=executor_task_id)
    matches = _query_named_tasks(task_class, project=project, name=expected_name)
    if (
        len(matches) != 1
        or _clearml_id(getattr(matches[0], "id", None), "discovered formal task")
        != task_id
    ):
        raise FormalMultiseedExecutionError(
            "formal task name is not globally unique before enqueue"
        )
    if _status(task, context="formal task") == "created":
        _enqueue_once(task_class, task, queue_name=str(record["worker_queue"]))
    _require_bound_task(
        task,
        expected_id=_clearml_id(getattr(task, "id", None), "formal task"),
        expected_parent=parent_task_id,
        expected_name=_task_run_name(record, executor_task_id=executor_task_id),
        expected_parameters=expected_parameters,
        source_d=source_d,
        queue_name=str(record["worker_queue"]),
        context="formal scheduled task",
    )
    return task


def _execute_matrix(
    task_class: object,
    *,
    plan: Mapping[str, object],
    output_task_id: str,
    source_d_task_id: str,
    source_d: str,
    teacher_task: object,
    teacher: Mapping[str, object],
    project: str,
    used_ids: set[str],
    deadline: float,
    poll_seconds: float,
    monotonic_clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_training = plan.get("training_tasks")
    raw_evaluation = plan.get("evaluation_tasks")
    if not isinstance(raw_training, list) or not isinstance(raw_evaluation, list):
        raise FormalMultiseedExecutionError("formal plan task matrix is invalid")
    training = [
        _frozen_mapping(item, context="training plan record") for item in raw_training
    ]
    evaluation = [
        _frozen_mapping(item, context="evaluation plan record")
        for item in raw_evaluation
    ]
    eval_by_train = {str(item["training_task_key"]): item for item in evaluation}
    if len(eval_by_train) != 45:
        raise FormalMultiseedExecutionError(
            "evaluation dependencies are not one-to-one"
        )
    teacher_script = _script(teacher_task, context="clean teacher")
    teacher_parameters = _parameters(teacher_task, context="clean teacher")
    pending: dict[str, list[dict[str, object]]] = {
        queue: [record for record in training if record.get("worker_queue") == queue]
        for queue in QUEUE_IDS
    }
    active: list[dict[str, object]] = []
    train_receipts: dict[str, dict[str, object]] = {}
    eval_receipts: dict[str, dict[str, object]] = {}

    def launch_training(queue: str) -> None:
        if not pending[queue]:
            return
        record = pending[queue].pop(0)
        parameters = _training_parameters(
            record,
            source_d_task_id=source_d_task_id,
            teacher=teacher,
        )
        task = _start_formal_task(
            task_class,
            teacher_task=teacher_task,
            teacher_script=teacher_script,
            teacher_parameters=teacher_parameters,
            source_d=source_d,
            record=record,
            parent_task_id=output_task_id,
            executor_task_id=output_task_id,
            expected_parameters=parameters,
            project=project,
            used_ids=used_ids,
        )
        active.append(
            {
                "phase": "training",
                "queue": queue,
                "task": task,
                "task_id": _clearml_id(getattr(task, "id", None), "training task"),
                "record": record,
                "parameters": parameters,
            }
        )

    for queue, capacity in QUEUE_CAPACITY.items():
        for _slot in range(capacity):
            launch_training(queue)

    while active:
        progressed = False
        for state in list(active):
            task = state["task"]
            record = state["record"]
            if not isinstance(record, Mapping):
                raise AssertionError("internal scheduler record is invalid")
            _reload(task, context="scheduled formal task")
            status = _status(task, context="scheduled formal task")
            if status in FAILED_STATUSES:
                raise FormalMultiseedExecutionError(
                    f"{state['phase']} task {record['task_key']} "
                    f"ended without completion: {status!r}"
                )
            if status in WAITING_STATUSES:
                continue
            if status != "completed":
                raise FormalMultiseedExecutionError(
                    f"scheduled task has unexpected status {status!r}"
                )
            expected_parent = (
                output_task_id
                if state["phase"] == "training"
                else str(state["training_task_id"])
            )
            _require_bound_task(
                task,
                expected_id=str(state["task_id"]),
                expected_parent=expected_parent,
                expected_name=_task_run_name(record, executor_task_id=output_task_id),
                expected_parameters=state["parameters"],
                source_d=source_d,
                queue_name=str(state["queue"]),
                context=f"completed {state['phase']} task",
            )
            active.remove(state)
            progressed = True
            if state["phase"] == "training":
                result = _validate_training_completion(
                    task,
                    record,
                    teacher=teacher,
                    source_d_task_id=source_d_task_id,
                )
                task_id = str(result["task_id"])
                receipt = _task_receipt(
                    record,
                    result,
                    parent_task_id=output_task_id,
                    parameters=state["parameters"],
                )
                train_receipts[str(record["task_key"])] = receipt
                eval_record = eval_by_train[str(record["task_key"])]
                eval_parameters = _evaluation_parameters(
                    eval_record,
                    training_task_id=task_id,
                    model=result,
                )
                eval_task = _start_formal_task(
                    task_class,
                    teacher_task=teacher_task,
                    teacher_script=teacher_script,
                    teacher_parameters=teacher_parameters,
                    source_d=source_d,
                    record=eval_record,
                    parent_task_id=task_id,
                    executor_task_id=output_task_id,
                    expected_parameters=eval_parameters,
                    project=project,
                    used_ids=used_ids,
                )
                active.append(
                    {
                        "phase": "evaluation",
                        "queue": state["queue"],
                        "task": eval_task,
                        "task_id": _clearml_id(
                            getattr(eval_task, "id", None), "evaluation task"
                        ),
                        "record": eval_record,
                        "parameters": eval_parameters,
                        "training_result": result,
                        "training_task_id": task_id,
                    }
                )
            else:
                result = _validate_evaluation_completion(
                    task,
                    record,
                    training_task_id=str(state["training_task_id"]),
                    training_model=state["training_result"],
                )
                eval_receipts[str(record["task_key"])] = _task_receipt(
                    record,
                    result,
                    parent_task_id=str(state["training_task_id"]),
                    parameters=state["parameters"],
                )
                launch_training(str(state["queue"]))
        if active and not progressed:
            if monotonic_clock() >= deadline:
                raise TimeoutError("timed out waiting for the formal multi-seed matrix")
            sleeper(poll_seconds)
    if len(train_receipts) != 45 or len(eval_receipts) != 45:
        raise FormalMultiseedExecutionError(
            "formal execution did not complete the exact 45+45 matrix"
        )
    return (
        [train_receipts[str(item["task_key"])] for item in training],
        [eval_receipts[str(item["task_key"])] for item in evaluation],
    )


def _manifest_payload(
    *,
    output_task_id: str,
    executor_script_sha256: str,
    executor_parameters_sha256: str,
    plan: Mapping[str, object],
    dependency_fingerprints: Mapping[str, object],
    source_d: Mapping[str, object],
    teacher: Mapping[str, object],
    training_receipts: Sequence[Mapping[str, object]],
    evaluation_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    formal_plan_provenance = _formal_plan_provenance(plan)
    task_ids = [
        str(receipt["task_id"])
        for receipt in [*training_receipts, *evaluation_receipts]
    ]
    if len(task_ids) != 90 or len(set(task_ids)) != 90:
        raise FormalMultiseedExecutionError(
            "execution task IDs are not globally unique"
        )
    queue_counts = Counter(
        str(receipt.get("worker_queue"))
        for receipt in [*training_receipts, *evaluation_receipts]
    )
    if queue_counts != Counter({queue: 30 for queue in QUEUE_IDS}):
        raise FormalMultiseedExecutionError(
            "execution receipts do not preserve the exact per-seed queues"
        )
    model_ids = []
    for receipt in training_receipts:
        result = receipt.get("result")
        if not isinstance(result, Mapping):
            raise FormalMultiseedExecutionError("training receipt result is invalid")
        model_ids.append(_clearml_id(result.get("model_id"), "training receipt model"))
    if len(model_ids) != 45 or len(set(model_ids)) != 45:
        raise FormalMultiseedExecutionError(
            "training OutputModel IDs are not globally unique"
        )
    return _sealed(
        {
            "schema_version": 2,
            "manifest_type": ("resilient_v2x_formal_initial_multiseed_execution"),
            "complete": True,
            "initial_only": True,
            "fallback_execution_enabled": False,
            "gate_result_produced": False,
            "fallback_authorized": False,
            "protocol_id": PROTOCOL_ID,
            "executor_task_id": output_task_id,
            "executor_entry_point": EXECUTOR_ENTRY_POINT,
            "executor_script_sha256": executor_script_sha256,
            "executor_parameters_sha256": executor_parameters_sha256,
            "plan_seal_sha256": plan["seal_sha256"],
            "plan_canonical_sha256": _content_sha256(plan),
            "formal_plan_provenance": formal_plan_provenance,
            "dependency_fingerprints": _frozen_mapping(
                dependency_fingerprints,
                context="dependency fingerprints",
            ),
            "source_d_evidence": _frozen_mapping(source_d, context="Source-D evidence"),
            "teacher_gate": _frozen_mapping(teacher, context="teacher gate"),
            "queue_contract": {
                queue: {
                    "queue_id": QUEUE_IDS[queue],
                    "capacity": QUEUE_CAPACITY[queue],
                }
                for queue in QUEUE_IDS
            },
            "release_policy": (
                "evaluation_released_only_after_its_own_training_completion_"
                "and_verification"
            ),
            "training_task_count": 45,
            "evaluation_task_count": 45,
            "evaluation_condition_count": 540,
            "training_tasks": [
                _frozen_mapping(item, context="training receipt")
                for item in training_receipts
            ],
            "evaluation_tasks": [
                _frozen_mapping(item, context="evaluation receipt")
                for item in evaluation_receipts
            ],
            "all_execution_task_ids": task_ids,
        }
    )


def _validate_manifest_shape(
    value: Mapping[str, object],
    *,
    output_task_id: str,
    executor_script_sha256: str,
    executor_parameters_sha256: str,
    plan: Mapping[str, object],
    dependency_fingerprints: Mapping[str, object],
    source_d: Mapping[str, object],
    teacher: Mapping[str, object],
) -> dict[str, object]:
    formal_plan_provenance = _formal_plan_provenance(plan)
    manifest = _frozen_mapping(value, context="execution manifest")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "manifest_type",
            "complete",
            "initial_only",
            "fallback_execution_enabled",
            "gate_result_produced",
            "fallback_authorized",
            "protocol_id",
            "executor_task_id",
            "executor_entry_point",
            "executor_script_sha256",
            "executor_parameters_sha256",
            "plan_seal_sha256",
            "plan_canonical_sha256",
            "formal_plan_provenance",
            "dependency_fingerprints",
            "source_d_evidence",
            "teacher_gate",
            "queue_contract",
            "release_policy",
            "training_task_count",
            "evaluation_task_count",
            "evaluation_condition_count",
            "training_tasks",
            "evaluation_tasks",
            "all_execution_task_ids",
            "seal_sha256",
        },
        context="execution manifest",
    )
    _require_seal(manifest, context="execution manifest")
    _exact_int(
        manifest.get("schema_version"),
        2,
        context="execution manifest schema",
    )
    _exact_bool(
        manifest.get("complete"),
        True,
        context="execution manifest complete",
    )
    _exact_bool(
        manifest.get("initial_only"),
        True,
        context="execution manifest initial-only",
    )
    _exact_bool(
        manifest.get("fallback_execution_enabled"),
        False,
        context="execution manifest fallback",
    )
    _exact_bool(
        manifest.get("gate_result_produced"),
        False,
        context="execution manifest gate result",
    )
    _exact_bool(
        manifest.get("fallback_authorized"),
        False,
        context="execution manifest fallback authorization",
    )
    scalars = {
        "manifest_type": ("resilient_v2x_formal_initial_multiseed_execution"),
        "protocol_id": PROTOCOL_ID,
        "executor_task_id": output_task_id,
        "executor_entry_point": EXECUTOR_ENTRY_POINT,
        "executor_script_sha256": executor_script_sha256,
        "executor_parameters_sha256": executor_parameters_sha256,
        "plan_seal_sha256": plan["seal_sha256"],
        "plan_canonical_sha256": _content_sha256(plan),
        "release_policy": (
            "evaluation_released_only_after_its_own_training_completion_"
            "and_verification"
        ),
    }
    for key, expected in scalars.items():
        if (
            type(manifest.get(key)) is not type(expected)
            or manifest.get(key) != expected
        ):
            raise FormalMultiseedExecutionError(f"execution manifest {key} drifted")
    for key, expected in {
        "training_task_count": 45,
        "evaluation_task_count": 45,
        "evaluation_condition_count": 540,
    }.items():
        _exact_int(
            manifest.get(key),
            expected,
            context=f"execution manifest {key}",
        )
    for field, expected in {
        "formal_plan_provenance": formal_plan_provenance,
        "dependency_fingerprints": dependency_fingerprints,
        "source_d_evidence": source_d,
        "teacher_gate": teacher,
        "queue_contract": {
            queue: {
                "queue_id": QUEUE_IDS[queue],
                "capacity": QUEUE_CAPACITY[queue],
            }
            for queue in QUEUE_IDS
        },
    }.items():
        observed = manifest.get(field)
        if not isinstance(observed, Mapping) or (
            _canonical_json(observed) != _canonical_json(expected)
        ):
            raise FormalMultiseedExecutionError(f"execution manifest {field} drifted")
    training = manifest.get("training_tasks")
    evaluation = manifest.get("evaluation_tasks")
    all_ids = manifest.get("all_execution_task_ids")
    if (
        not isinstance(training, list)
        or len(training) != 45
        or not isinstance(evaluation, list)
        or len(evaluation) != 45
        or not isinstance(all_ids, list)
        or len(all_ids) != 90
    ):
        raise FormalMultiseedExecutionError(
            "execution manifest does not contain exact 45+45 receipts"
        )
    expected_training_keys = [
        item["task_key"] for item in plan["training_tasks"] if type(item) is dict
    ]
    expected_evaluation_keys = [
        item["task_key"] for item in plan["evaluation_tasks"] if type(item) is dict
    ]
    for label, receipts, expected_keys in (
        ("training", training, expected_training_keys),
        ("evaluation", evaluation, expected_evaluation_keys),
    ):
        observed_keys = [
            item.get("task_key") if type(item) is dict else None for item in receipts
        ]
        if observed_keys != expected_keys or len(set(observed_keys)) != 45:
            raise FormalMultiseedExecutionError(
                f"execution manifest {label} task keys drifted"
            )
    ids = [_clearml_id(item, "execution manifest task ID") for item in all_ids]
    receipt_ids = []
    for receipt in [*training, *evaluation]:
        if not isinstance(receipt, Mapping):
            raise FormalMultiseedExecutionError("execution manifest receipt is invalid")
        receipt_ids.append(
            _clearml_id(
                receipt.get("task_id"),
                "execution manifest receipt task ID",
            )
        )
    if ids != receipt_ids or len(set(ids)) != 90:
        raise FormalMultiseedExecutionError(
            "execution manifest task IDs drifted or alias"
        )
    model_ids = []
    for receipt in training:
        result = receipt.get("result")
        if not isinstance(result, Mapping):
            raise FormalMultiseedExecutionError(
                "execution manifest training result is invalid"
            )
        model_ids.append(
            _clearml_id(
                result.get("model_id"),
                "execution manifest training model ID",
            )
        )
    if len(set(model_ids)) != 45:
        raise FormalMultiseedExecutionError(
            "execution manifest training model IDs drifted or alias"
        )
    return manifest


def _artifact_descriptor_metadata(
    artifact: object,
    *,
    context: str,
    expected_type: str | None = None,
) -> dict[str, object]:
    artifact_type = getattr(artifact, "type", None)
    raw_mode = getattr(artifact, "mode", None)
    mode = getattr(raw_mode, "value", raw_mode)
    if type(artifact_type) is not str or not artifact_type:
        raise FormalMultiseedExecutionError(f"{context} type is invalid")
    if expected_type is not None and artifact_type != expected_type:
        raise FormalMultiseedExecutionError(
            f"{context} type is not exact {expected_type!r}"
        )
    if type(mode) is not str or mode != "output":
        raise FormalMultiseedExecutionError(f"{context} mode is not exact output")
    return {
        "type": artifact_type,
        "mode": mode,
        "url": _require_fileserver_url(
            getattr(artifact, "url", None),
            context=context,
        ),
        "size_bytes": _nonnegative_int(
            getattr(artifact, "size", None),
            context=f"{context} bytes",
        ),
        "sha256": _sha256(
            getattr(artifact, "hash", None),
            context,
        ),
    }


def _artifact_binding_metadata(
    task: object,
    *,
    context: str,
    excluded_names: frozenset[str] = frozenset(),
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, artifact in sorted(_server_artifacts(task, context=context).items()):
        if name in excluded_names:
            continue
        result[name] = _artifact_descriptor_metadata(
            artifact,
            context=f"{context} artifact {name!r}",
        )
    return result


def _output_model_binding_metadata(
    task: object,
    *,
    context: str,
) -> list[dict[str, object]]:
    outputs = _models(task, context=context).get("output", [])
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise FormalMultiseedExecutionError(
            f"{context} output model inventory is invalid"
        )
    result: list[dict[str, object]] = []
    for model in outputs:
        model_id = _clearml_id(getattr(model, "id", None), f"{context} model")
        task_id = _clearml_id(
            getattr(model, "task", None),
            f"{context} model task",
        )
        name = getattr(model, "name", None)
        framework = getattr(model, "framework", None)
        published = getattr(model, "published", None)
        if type(name) is not str or not name:
            raise FormalMultiseedExecutionError(f"{context} model name is invalid")
        if framework is not None and (type(framework) is not str or not framework):
            raise FormalMultiseedExecutionError(f"{context} model framework is invalid")
        if type(published) is not bool:
            raise FormalMultiseedExecutionError(
                f"{context} model published state is invalid"
            )
        result.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "name": name,
                "url": _require_fileserver_url(
                    getattr(model, "url", None),
                    context=f"{context} model",
                ),
                "framework": framework,
                "published": published,
            }
        )
    result.sort(key=lambda item: str(item["model_id"]))
    if len({str(item["model_id"]) for item in result}) != len(result):
        raise FormalMultiseedExecutionError(
            f"{context} output model IDs drifted or alias"
        )
    return result


def _task_binding_metadata(
    task: object,
    *,
    context: str,
    excluded_artifacts: frozenset[str] = frozenset(),
) -> dict[str, object]:
    data = getattr(task, "data", None)
    task_id = _clearml_id(getattr(data, "id", None), f"{context} snapshot")
    worker = getattr(data, "last_worker", None)
    if worker is not None and (type(worker) is not str or not worker.strip()):
        raise FormalMultiseedExecutionError(
            f"{context} last_worker metadata is invalid"
        )
    script = _script(task, context=context)
    parameters = _parameters(task, context=context)
    return {
        "task_id": task_id,
        "project_id": _clearml_id(
            getattr(data, "project", None),
            f"{context} project",
        ),
        "parent_task_id": _task_parent(task, context=context),
        "name": _task_name(task, context=context),
        "status": _status(task, context=context),
        "script_sha256": _content_sha256(script),
        "parameters_sha256": _content_sha256(parameters),
        "output_uri": _output_uri(task, context=context),
        "queue_id": _queue_id(task, context=context),
        "last_worker": worker.strip() if type(worker) is str else None,
        "artifacts": _artifact_binding_metadata(
            task,
            context=context,
            excluded_names=excluded_artifacts,
        ),
        "output_models": _output_model_binding_metadata(task, context=context),
    }


PUBLICATION_BATCH_ONLY_FIELDS = (
    "id",
    "project",
    "parent",
    "name",
    "status",
    "script",
    "hyperparams",
    "output",
    "execution",
    "last_worker",
)


def _raw_publication_parameters(data: object, *, context: str) -> dict[str, object]:
    raw_sections = getattr(data, "hyperparams", None)
    if raw_sections is None:
        raw_sections = {}
    if not isinstance(raw_sections, Mapping):
        raise FormalMultiseedExecutionError(
            f"{context} raw hyperparameters are invalid"
        )
    result: dict[str, object] = {}
    missing = object()
    for section, raw_parameters in raw_sections.items():
        if type(section) is not str or not section:
            raise FormalMultiseedExecutionError(
                f"{context} raw hyperparameter section is invalid"
            )
        if not isinstance(raw_parameters, Mapping):
            raise FormalMultiseedExecutionError(
                f"{context} raw hyperparameter section is not a mapping"
            )
        for name, raw_parameter in raw_parameters.items():
            if type(name) is not str or not name:
                raise FormalMultiseedExecutionError(
                    f"{context} raw hyperparameter name is invalid"
                )
            value = getattr(raw_parameter, "value", missing)
            if value is missing and isinstance(raw_parameter, Mapping):
                value = raw_parameter.get("value", missing)
            if value is missing:
                raise FormalMultiseedExecutionError(
                    f"{context} raw hyperparameter value is unavailable"
                )
            key = f"{section}/{name}"
            if key in result:
                raise FormalMultiseedExecutionError(
                    f"{context} raw hyperparameter keys alias"
                )
            result[key] = value
    return _frozen_mapping(result, context=f"{context} raw parameters")


def _raw_publication_script(data: object, *, context: str) -> dict[str, object]:
    script = getattr(data, "script", None)
    if script is None:
        raise FormalMultiseedExecutionError(
            f"{context} raw script metadata is unavailable"
        )
    result = _frozen_mapping(
        {
            "repository": getattr(script, "repository", None),
            "working_dir": getattr(script, "working_dir", None),
            "entry_point": getattr(script, "entry_point", None),
            "diff": getattr(script, "diff", None),
        },
        context=f"{context} raw script",
    )
    _require_exact_keys(
        result,
        {"repository", "working_dir", "entry_point", "diff"},
        context=f"{context} raw script",
    )
    return result


def _raw_publication_artifact_inventory(
    data: object,
    *,
    context: str,
) -> dict[str, object]:
    execution = getattr(data, "execution", None)
    raw_artifacts = getattr(execution, "artifacts", None)
    if raw_artifacts is None:
        raw_artifacts = ()
    if type(raw_artifacts) not in {list, tuple}:
        raise FormalMultiseedExecutionError(
            f"{context} raw artifact inventory is invalid"
        )
    result: dict[str, object] = {}
    for raw in raw_artifacts:
        name = getattr(raw, "key", None)
        if type(name) is not str or not name or name in result:
            raise FormalMultiseedExecutionError(
                f"{context} raw artifact names drifted or alias"
            )
        artifact_type = getattr(raw, "type", None)
        raw_mode = getattr(raw, "mode", None)
        mode = getattr(raw_mode, "value", raw_mode)
        if type(artifact_type) is not str or not artifact_type:
            raise FormalMultiseedExecutionError(
                f"{context} raw artifact {name!r} type is invalid"
            )
        if type(mode) is not str or mode != "output":
            raise FormalMultiseedExecutionError(
                f"{context} raw artifact {name!r} mode is not exact output"
            )
        result[name] = {
            "type": artifact_type,
            "mode": mode,
            "url": _require_fileserver_url(
                getattr(raw, "uri", None),
                context=f"{context} raw artifact {name!r}",
            ),
            "size_bytes": _nonnegative_int(
                getattr(raw, "content_size", None),
                context=f"{context} raw artifact {name!r} bytes",
            ),
            "sha256": _sha256(
                getattr(raw, "hash", None),
                f"{context} raw artifact {name!r}",
            ),
        }
    return result


def _raw_publication_artifacts(
    data: object,
    *,
    context: str,
    excluded_names: frozenset[str] = frozenset(),
) -> dict[str, object]:
    inventory = _raw_publication_artifact_inventory(data, context=context)
    return {
        name: descriptor
        for name, descriptor in inventory.items()
        if name not in excluded_names
    }


def _raw_publication_output_model_ids(
    data: object,
    *,
    context: str,
) -> list[str]:
    raw_model = getattr(getattr(data, "output", None), "model", None)
    if raw_model is None or raw_model == "":
        return []
    return [_clearml_id(raw_model, f"{context} raw output model")]


def _raw_publication_task_binding(
    data: object,
    *,
    context: str,
    expected_output_models: object,
    excluded_artifacts: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if type(expected_output_models) is not list:
        raise FormalMultiseedExecutionError(
            f"{context} frozen output model descriptors are invalid"
        )
    model_ids: list[str] = []
    for descriptor in expected_output_models:
        if type(descriptor) is not dict:
            raise FormalMultiseedExecutionError(
                f"{context} frozen output model descriptor is invalid"
            )
        model_ids.append(
            _clearml_id(
                descriptor.get("model_id"),
                f"{context} frozen output model",
            )
        )
    if len(model_ids) != len(set(model_ids)):
        raise FormalMultiseedExecutionError(
            f"{context} frozen output model IDs drifted or alias"
        )
    if _raw_publication_output_model_ids(data, context=context) != model_ids:
        raise FormalMultiseedExecutionError(f"{context} batch output model IDs drifted")
    raw_status = getattr(data, "status", None)
    status = getattr(raw_status, "value", raw_status)
    if type(status) is not str or status not in (
        WAITING_STATUSES | FAILED_STATUSES | {"completed"}
    ):
        raise FormalMultiseedExecutionError(f"{context} raw status is invalid")
    parent = getattr(data, "parent", None)
    if parent is None:
        parent = ""
    if type(parent) is not str:
        raise FormalMultiseedExecutionError(f"{context} raw parent is invalid")
    if parent:
        _clearml_id(parent, f"{context} raw parent")
    name = getattr(data, "name", None)
    if type(name) is not str or not name:
        raise FormalMultiseedExecutionError(f"{context} raw name is invalid")
    worker = getattr(data, "last_worker", None)
    if worker is not None and (type(worker) is not str or not worker.strip()):
        raise FormalMultiseedExecutionError(f"{context} raw last_worker is invalid")
    execution = getattr(data, "execution", None)
    queue = getattr(execution, "queue", None)
    if queue is None or queue == "":
        queue_id = ""
    else:
        queue_id = _clearml_id(queue, f"{context} raw execution queue")
    output_uri = _require_fileserver_url(
        getattr(getattr(data, "output", None), "destination", None),
        context=f"{context} raw output",
    )
    script = _raw_publication_script(data, context=context)
    parameters = _raw_publication_parameters(data, context=context)
    return {
        "task_id": _clearml_id(getattr(data, "id", None), f"{context} raw task"),
        "project_id": _clearml_id(
            getattr(data, "project", None),
            f"{context} raw project",
        ),
        "parent_task_id": parent,
        "name": name,
        "status": status,
        "script_sha256": _content_sha256(script),
        "parameters_sha256": _content_sha256(parameters),
        "output_uri": output_uri,
        "queue_id": queue_id,
        "last_worker": worker.strip() if type(worker) is str else None,
        "artifacts": _raw_publication_artifacts(
            data,
            context=context,
            excluded_names=excluded_artifacts,
        ),
        "output_models": _freeze(
            expected_output_models,
            context=f"{context} frozen output model descriptors",
        ),
    }


def _publication_authoritative_batch_snapshot(
    task_class: object,
    snapshot: Mapping[str, object],
    dependency_ids: Mapping[str, str],
    manifest: Mapping[str, object],
    *,
    output_task_id: str,
) -> dict[str, object]:
    """Validate publication state at one terminal ClearML batch-read boundary."""
    frozen = _frozen_mapping(snapshot, context="pre-batch publication snapshot")
    root_order = ("planner", "source_d", "teacher_gate", "teacher")
    roots = frozen.get("roots")
    output_binding = frozen.get("output")
    if type(roots) is not dict or type(output_binding) is not dict:
        raise FormalMultiseedExecutionError(
            "pre-batch publication root snapshot is invalid"
        )
    entries: list[tuple[str, str, str, dict[str, object], frozenset[str]]] = [
        (
            "output",
            "",
            output_task_id,
            output_binding,
            frozenset({EXECUTION_MANIFEST_ARTIFACT}),
        )
    ]
    for name in root_order:
        binding = roots.get(name)
        if type(binding) is not dict:
            raise FormalMultiseedExecutionError(
                f"pre-batch publication root {name} is invalid"
            )
        entries.append(
            (
                "root",
                name,
                _clearml_id(
                    dependency_ids.get(name),
                    f"pre-batch publication root {name}",
                ),
                binding,
                frozenset(),
            )
        )
    phase_items: dict[str, list[tuple[str, str]]] = {
        "training": [],
        "evaluation": [],
    }
    for phase, field in (
        ("training", "training_tasks"),
        ("evaluation", "evaluation_tasks"),
    ):
        receipts = manifest.get(field)
        expected_items = frozen.get(phase)
        if (
            type(receipts) is not list
            or type(expected_items) is not list
            or len(receipts) != 45
            or len(expected_items) != 45
        ):
            raise FormalMultiseedExecutionError(
                f"pre-batch publication {phase} inventory is invalid"
            )
        for receipt, item in zip(receipts, expected_items):
            if type(receipt) is not dict or type(item) is not dict:
                raise FormalMultiseedExecutionError(
                    f"pre-batch publication {phase} item is invalid"
                )
            key = receipt.get("task_key")
            task_id = _clearml_id(
                receipt.get("task_id"),
                f"pre-batch publication {phase} task",
            )
            binding = item.get("binding")
            if (
                type(key) is not str
                or item.get("task_key") != key
                or type(binding) is not dict
            ):
                raise FormalMultiseedExecutionError(
                    f"pre-batch publication {phase} binding is invalid"
                )
            entries.append((phase, key, task_id, binding, frozenset()))
            phase_items[phase].append((key, task_id))
    task_ids = [entry[2] for entry in entries]
    if len(task_ids) != 95 or len(set(task_ids)) != 95:
        raise FormalMultiseedExecutionError(
            "pre-batch publication task IDs drifted or alias"
        )
    query = getattr(task_class, "_query_tasks", None)
    if not callable(query):
        raise FormalMultiseedExecutionError(
            "ClearML cannot issue one authoritative publication batch query"
        )
    try:
        # This is the sole terminal server read; validation below uses only records
        # returned by this one request, so this response defines the linearization point.
        raw_records = query(
            task_ids=list(task_ids),
            fetch_only_first_page=True,
            only_fields=list(PUBLICATION_BATCH_ONLY_FIELDS),
        )
    except Exception as error:
        raise FormalMultiseedExecutionError(
            "authoritative publication batch query failed"
        ) from error
    if type(raw_records) is not list or len(raw_records) != 95:
        raise FormalMultiseedExecutionError(
            "authoritative publication batch is incomplete"
        )
    records_by_id: dict[str, object] = {}
    object_ids: set[int] = set()
    for raw in raw_records:
        task_id = _clearml_id(
            getattr(raw, "id", None),
            "authoritative publication batch task",
        )
        if task_id in records_by_id or id(raw) in object_ids:
            raise FormalMultiseedExecutionError(
                "authoritative publication batch records drifted or alias"
            )
        records_by_id[task_id] = raw
        object_ids.add(id(raw))
    if set(records_by_id) != set(task_ids):
        raise FormalMultiseedExecutionError(
            "authoritative publication batch membership drifted"
        )
    output_artifacts = _raw_publication_artifact_inventory(
        records_by_id[output_task_id],
        context="authoritative publication output",
    )
    manifest_descriptor = output_artifacts.get(EXECUTION_MANIFEST_ARTIFACT)
    if manifest_descriptor is not None and (
        type(manifest_descriptor) is not dict
        or manifest_descriptor.get("type") != "dict"
    ):
        raise FormalMultiseedExecutionError(
            "authoritative execution manifest descriptor type drifted"
        )
    bindings_by_id: dict[str, dict[str, object]] = {}
    for phase, key, task_id, expected_binding, excluded in entries:
        context = (
            "authoritative publication output"
            if phase == "output"
            else f"authoritative publication {phase} {key}"
        )
        observed_binding = _raw_publication_task_binding(
            records_by_id[task_id],
            context=context,
            expected_output_models=expected_binding.get("output_models"),
            excluded_artifacts=excluded,
        )
        if _canonical_json(observed_binding) != _canonical_json(expected_binding):
            raise FormalMultiseedExecutionError(
                f"{context} drifted at batch linearization point"
            )
        bindings_by_id[task_id] = observed_binding
    result = {
        "schema_version": 1,
        "output": bindings_by_id[output_task_id],
        "roots": {name: bindings_by_id[dependency_ids[name]] for name in root_order},
        "training": [
            {"task_key": key, "binding": bindings_by_id[task_id]}
            for key, task_id in phase_items["training"]
        ],
        "evaluation": [
            {"task_key": key, "binding": bindings_by_id[task_id]}
            for key, task_id in phase_items["evaluation"]
        ],
    }
    authoritative = _frozen_mapping(
        result,
        context="authoritative publication batch snapshot",
    )
    if _canonical_json(authoritative) != _canonical_json(frozen):
        raise FormalMultiseedExecutionError(
            "authoritative publication batch snapshot drifted"
        )
    return _frozen_mapping(
        {
            "publication_batch_schema_version": 1,
            "bindings": authoritative,
            "execution_manifest_descriptor": manifest_descriptor,
        },
        context="authoritative publication batch boundary",
    )


def _resolve_manifest_tasks(
    task_class: object,
    manifest: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    project: str,
    output_task_id: str,
) -> dict[str, dict[str, object]]:
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise FormalMultiseedExecutionError(
            "ClearML cannot resolve publication task bindings"
        )
    resolved: dict[str, dict[str, object]] = {
        "training": {},
        "evaluation": {},
    }
    used_ids: set[str] = set()
    used_objects: set[int] = set()
    for phase, plan_field, manifest_field in (
        ("training", "training_tasks", "training_tasks"),
        ("evaluation", "evaluation_tasks", "evaluation_tasks"),
    ):
        raw_plan = plan.get(plan_field)
        raw_receipts = manifest.get(manifest_field)
        if type(raw_plan) is not list or type(raw_receipts) is not list:
            raise FormalMultiseedExecutionError(
                f"publication {phase} task inventory is invalid"
            )
        records = {
            str(item["task_key"]): item
            for item in raw_plan
            if type(item) is dict and type(item.get("task_key")) is str
        }
        if len(records) != 45 or len(raw_receipts) != 45:
            raise FormalMultiseedExecutionError(
                f"publication {phase} task inventory must contain exactly 45 tasks"
            )
        for receipt in raw_receipts:
            if type(receipt) is not dict:
                raise FormalMultiseedExecutionError(
                    f"publication {phase} receipt is invalid"
                )
            key = receipt.get("task_key")
            if type(key) is not str or key not in records or key in resolved[phase]:
                raise FormalMultiseedExecutionError(
                    f"publication {phase} task key drifted"
                )
            task_id = _clearml_id(
                receipt.get("task_id"),
                f"publication {phase} task",
            )
            if task_id in used_ids:
                raise FormalMultiseedExecutionError(
                    "publication execution task IDs drifted or alias"
                )
            expected_name = _task_run_name(
                records[key],
                executor_task_id=output_task_id,
            )
            matches = _query_named_tasks(
                task_class,
                project=project,
                name=expected_name,
            )
            if (
                len(matches) != 1
                or _clearml_id(
                    getattr(matches[0], "id", None),
                    f"publication discovered {phase} task",
                )
                != task_id
            ):
                raise FormalMultiseedExecutionError(
                    f"publication {phase} task name is not globally unique"
                )
            task = getter(task_id=task_id)
            if (
                task is None
                or _clearml_id(
                    getattr(task, "id", None),
                    f"publication resolved {phase} task",
                )
                != task_id
            ):
                raise FormalMultiseedExecutionError(
                    f"publication {phase} task identity mismatch"
                )
            if id(task) in used_objects:
                raise FormalMultiseedExecutionError(
                    "publication execution task objects drifted or alias"
                )
            used_ids.add(task_id)
            used_objects.add(id(task))
            resolved[phase][key] = task
        if set(resolved[phase]) != set(records):
            raise FormalMultiseedExecutionError(
                f"publication {phase} task keys are incomplete"
            )
    if len(used_ids) != 90 or len(used_objects) != 90:
        raise FormalMultiseedExecutionError(
            "publication does not resolve 90 unique execution tasks"
        )
    return resolved


def _publication_metadata_sweep(
    task_class: object,
    output_task: object,
    dependencies: Mapping[str, object],
    dependency_ids: Mapping[str, str],
    manifest: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    project: str,
    output_task_id: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    root_order = ("planner", "source_d", "teacher_gate", "teacher")
    if set(dependencies) != set(root_order) or set(dependency_ids) != set(root_order):
        raise FormalMultiseedExecutionError(
            "publication root dependency inventory drifted"
        )
    resolved = _resolve_manifest_tasks(
        task_class,
        manifest,
        plan=plan,
        project=project,
        output_task_id=output_task_id,
    )
    ordered_execution: list[tuple[str, str, object]] = []
    for phase, field in (
        ("training", "training_tasks"),
        ("evaluation", "evaluation_tasks"),
    ):
        receipts = manifest.get(field)
        if type(receipts) is not list:
            raise FormalMultiseedExecutionError(
                f"publication {phase} receipts are invalid"
            )
        for receipt in receipts:
            if type(receipt) is not dict or type(receipt.get("task_key")) is not str:
                raise FormalMultiseedExecutionError(
                    f"publication {phase} receipt is invalid"
                )
            key = str(receipt["task_key"])
            ordered_execution.append((phase, key, resolved[phase][key]))

    ordered_tasks = [
        ("executor", output_task),
        *((f"root {name}", dependencies[name]) for name in root_order),
        *((f"{phase} {key}", task) for phase, key, task in ordered_execution),
    ]
    expected_ids = [
        output_task_id,
        *(dependency_ids[name] for name in root_order),
        *(
            _clearml_id(
                receipt.get("task_id"),
                f"publication {phase} receipt",
            )
            for phase, field in (
                ("training", "training_tasks"),
                ("evaluation", "evaluation_tasks"),
            )
            for receipt in manifest[field]
            if type(receipt) is dict
        ),
    ]
    if (
        len(ordered_tasks) != 95
        or len(expected_ids) != 95
        or len(set(expected_ids)) != 95
        or len({id(task) for _context, task in ordered_tasks}) != 95
    ):
        raise FormalMultiseedExecutionError(
            "publication output, roots, and execution tasks drifted or alias"
        )
    for (context, task), expected_id in zip(ordered_tasks, expected_ids):
        if _clearml_id(getattr(task, "id", None), context) != expected_id:
            raise FormalMultiseedExecutionError(
                f"publication {context} identity mismatch"
            )
        _reload(task, context=f"publication {context}")

    output_metadata = _task_binding_metadata(
        output_task,
        context="publication executor",
        excluded_artifacts=frozenset({EXECUTION_MANIFEST_ARTIFACT}),
    )
    if output_metadata["output_models"]:
        raise FormalMultiseedExecutionError(
            "publication executor unexpectedly exposes output models"
        )
    roots = {
        name: _task_binding_metadata(
            dependencies[name],
            context=f"publication root {name}",
        )
        for name in root_order
    }
    execution: dict[str, list[dict[str, object]]] = {
        "training": [],
        "evaluation": [],
    }
    receipts_by_phase = {
        "training": manifest["training_tasks"],
        "evaluation": manifest["evaluation_tasks"],
    }
    for phase, key, task in ordered_execution:
        metadata = _task_binding_metadata(
            task,
            context=f"publication {phase} task {key}",
        )
        receipt = next(
            item
            for item in receipts_by_phase[phase]
            if type(item) is dict and item.get("task_key") == key
        )
        if phase == "training":
            result = receipt.get("result")
            if type(result) is not dict:
                raise FormalMultiseedExecutionError(
                    "publication training result is invalid"
                )
            models = metadata["output_models"]
            expected_model = {
                "model_id": _clearml_id(
                    result.get("model_id"),
                    "publication training model",
                ),
                "task_id": metadata["task_id"],
                "name": f"ResilientV2X {receipt['subject']} final checkpoint",
                "url": _require_fileserver_url(
                    result.get("model_url"),
                    context="publication training model",
                ),
            }
            if (
                type(models) is not list
                or len(models) != 1
                or any(
                    models[0].get(name) != value
                    for name, value in expected_model.items()
                )
            ):
                raise FormalMultiseedExecutionError(
                    "publication training model binding drifted"
                )
        elif metadata["output_models"]:
            raise FormalMultiseedExecutionError(
                "publication evaluation task unexpectedly exposes output models"
            )
        execution[phase].append({"task_key": key, "binding": metadata})
    return (
        _frozen_mapping(
            {
                "schema_version": 1,
                "output": output_metadata,
                "roots": roots,
                "training": execution["training"],
                "evaluation": execution["evaluation"],
            },
            context="publication metadata sweep",
        ),
        resolved,
    )


def _validate_executor_publication_provenance(
    output_task: object,
    contract: Mapping[str, object],
) -> None:
    context = "publication executor provenance"
    _require_exact_keys(
        contract,
        {
            "task_id",
            "parent_task_id",
            "name",
            "script_source",
            "script_sha256",
            "parameters",
            "parameters_sha256",
            "output_uri",
            "project_id",
        },
        context=context,
    )
    task_id = _clearml_id(contract.get("task_id"), context)
    if _clearml_id(getattr(output_task, "id", None), context) != task_id:
        raise FormalMultiseedExecutionError("executor identity drifted at publication")
    if _status(output_task, context=context) not in {"created", "in_progress"}:
        raise FormalMultiseedExecutionError("executor status drifted at publication")
    if _task_parent(output_task, context=context) != contract.get("parent_task_id"):
        raise FormalMultiseedExecutionError("executor parent drifted at publication")
    if _task_name(output_task, context=context) != contract.get("name"):
        raise FormalMultiseedExecutionError("executor name drifted at publication")
    expected_project_id = _clearml_id(
        contract.get("project_id"),
        "executor publication project",
    )
    observed_project_id = _clearml_id(
        getattr(getattr(output_task, "data", None), "project", None),
        "executor publication project",
    )
    if observed_project_id != expected_project_id:
        raise FormalMultiseedExecutionError("executor project drifted at publication")
    source = contract.get("script_source")
    if type(source) is not str or not source:
        raise FormalMultiseedExecutionError("executor source contract is invalid")
    _require_no_repo_script(
        output_task,
        entry_point=EXECUTOR_ENTRY_POINT,
        expected_sha256=_sha256(
            contract.get("script_sha256"),
            "executor publication script",
        ),
        expected_source=source,
        context=context,
    )
    observed_output_uri = _output_uri(output_task, context=context)
    expected_output_uri = contract.get("output_uri")
    if type(expected_output_uri) is not str or not expected_output_uri:
        raise FormalMultiseedExecutionError(
            "executor publication output URI contract is invalid"
        )
    if observed_output_uri != expected_output_uri:
        raise FormalMultiseedExecutionError(
            "executor output URI drifted at publication"
        )
    expected_parameters = contract.get("parameters")
    if type(expected_parameters) is not dict:
        raise FormalMultiseedExecutionError(
            "executor publication parameter contract is invalid"
        )
    observed_parameters = _freeze_executor_parameters(
        output_task,
        expected_parameters,
        context=context,
    )
    expected_parameters_sha = _sha256(
        contract.get("parameters_sha256"),
        "executor publication parameters",
    )
    if _content_sha256(observed_parameters) != expected_parameters_sha:
        raise FormalMultiseedExecutionError("executor Args drifted at publication")
    outputs = _models(output_task, context=context).get("output", [])
    if (
        not isinstance(outputs, Sequence)
        or isinstance(outputs, (str, bytes))
        or outputs
    ):
        raise FormalMultiseedExecutionError(
            "executor output model inventory drifted at publication"
        )


def _validate_publication_bindings(
    mode: Literal["deep", "snapshot"],
    *,
    task_class: object,
    output_task: object,
    dependencies: Mapping[str, object],
    dependency_ids: Mapping[str, str],
    manifest: Mapping[str, object],
    plan: Mapping[str, object],
    expected_root_snapshot: Mapping[str, object],
    executor_contract: Mapping[str, object],
    project: str,
    planner_producer_sha256: str,
    plan_validator_path: Path | None,
) -> dict[str, object]:
    if type(mode) is not str or mode not in {"deep", "snapshot"}:
        raise FormalMultiseedExecutionError("publication validation mode is invalid")
    before, resolved = _publication_metadata_sweep(
        task_class,
        output_task,
        dependencies,
        dependency_ids,
        manifest,
        plan=plan,
        project=project,
        output_task_id=str(executor_contract["task_id"]),
    )
    _validate_executor_publication_provenance(output_task, executor_contract)
    if mode == "deep":
        current_roots = _dependency_snapshot(
            dependencies,
            dependency_ids,
            producer_sha=planner_producer_sha256,
            plan_validator_path=plan_validator_path,
            context="publication deep freeze",
            reload_tasks=False,
        )
        expected_roots = _frozen_mapping(
            expected_root_snapshot,
            context="expected publication root snapshot",
        )
        if _canonical_json(current_roots) != _canonical_json(expected_roots):
            raise FormalMultiseedExecutionError(
                "root dependencies drifted at publication"
            )
        source_d = expected_roots.get("source_d")
        teacher = expected_roots.get("teacher")
        if type(source_d) is not dict or type(teacher) is not dict:
            raise FormalMultiseedExecutionError(
                "expected publication dependency semantics are invalid"
            )
        _revalidate_manifest_tasks(
            task_class,
            manifest,
            plan=plan,
            project=project,
            output_task_id=str(executor_contract["task_id"]),
            source_d_task_id=dependency_ids["source_d"],
            source_d=str(source_d["script"]),
            teacher_task=dependencies["teacher"],
            teacher=teacher,
            resolved_tasks=resolved,
            reload_tasks=False,
        )
    after, _ = _publication_metadata_sweep(
        task_class,
        output_task,
        dependencies,
        dependency_ids,
        manifest,
        plan=plan,
        project=project,
        output_task_id=str(executor_contract["task_id"]),
    )
    if _canonical_json(after) != _canonical_json(before):
        raise FormalMultiseedExecutionError(
            f"publication bindings drifted during {mode} validation"
        )
    return _publication_authoritative_batch_snapshot(
        task_class,
        after,
        dependency_ids,
        manifest,
        output_task_id=str(executor_contract["task_id"]),
    )


def _publish_manifest(
    output_task: object,
    manifest: Mapping[str, object],
    *,
    validate_bindings: Callable[
        [Literal["deep", "snapshot"]],
        Mapping[str, object],
    ],
) -> None:
    frozen_manifest = _frozen_mapping(
        manifest, context="execution manifest publication"
    )

    def committed_snapshot() -> dict[str, object] | None:
        _reload(output_task, context="executor task")
        artifacts_before = _server_artifacts(output_task, context="executor task")
        names = set(artifacts_before)
        if not names:
            return None
        if names != {EXECUTION_MANIFEST_ARTIFACT}:
            raise FormalMultiseedExecutionError(
                "executor contains unexpected manifest publication artifacts"
            )
        descriptor_before = _artifact_descriptor_metadata(
            artifacts_before[EXECUTION_MANIFEST_ARTIFACT],
            context="execution manifest artifact",
            expected_type="dict",
        )
        readback = _artifact_mapping(
            output_task,
            EXECUTION_MANIFEST_ARTIFACT,
            context="executor task",
        )
        artifacts_after = _server_artifacts(output_task, context="executor task")
        if set(artifacts_after) != {EXECUTION_MANIFEST_ARTIFACT}:
            raise FormalMultiseedExecutionError(
                "execution manifest artifact inventory drifted during readback"
            )
        descriptor_after = _artifact_descriptor_metadata(
            artifacts_after[EXECUTION_MANIFEST_ARTIFACT],
            context="execution manifest artifact",
            expected_type="dict",
        )
        if _canonical_json(descriptor_after) != _canonical_json(descriptor_before):
            raise FormalMultiseedExecutionError(
                "execution manifest artifact descriptor drifted during readback"
            )
        if _canonical_json(readback) != _canonical_json(frozen_manifest):
            raise FormalMultiseedExecutionError(
                "execution manifest server readback drifted"
            )
        return _frozen_mapping(
            {
                "schema_version": 1,
                "artifact_name": EXECUTION_MANIFEST_ARTIFACT,
                "descriptor": descriptor_after,
                "content_sha256": _content_sha256(readback),
            },
            context="execution manifest committed seal",
        )

    def binding_snapshot(
        mode: Literal["deep", "snapshot"],
    ) -> dict[str, object]:
        try:
            value = validate_bindings(mode)
        except FormalMultiseedExecutionError:
            raise
        except Exception as error:
            raise FormalMultiseedExecutionError(
                f"publication {mode} binding validation failed"
            ) from error
        return _frozen_mapping(
            value,
            context=f"publication {mode} binding snapshot",
        )

    def require_committed(
        expected_seal: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        observed = committed_snapshot()
        if observed is None:
            raise FormalMultiseedExecutionError("execution manifest was not committed")
        if expected_seal is not None and _canonical_json(observed) != _canonical_json(
            expected_seal
        ):
            raise FormalMultiseedExecutionError(
                "execution manifest committed seal drifted"
            )
        return observed

    def authoritative_components(
        snapshot: Mapping[str, object],
    ) -> tuple[Mapping[str, object], Mapping[str, object] | None, bool]:
        if "publication_batch_schema_version" not in snapshot:
            return snapshot, None, False
        _require_exact_keys(
            snapshot,
            {
                "publication_batch_schema_version",
                "bindings",
                "execution_manifest_descriptor",
            },
            context="authoritative publication batch boundary",
        )
        if snapshot.get("publication_batch_schema_version") != 1:
            raise FormalMultiseedExecutionError(
                "authoritative publication batch schema drifted"
            )
        bindings = snapshot.get("bindings")
        descriptor = snapshot.get("execution_manifest_descriptor")
        if type(bindings) is not dict or (
            descriptor is not None and type(descriptor) is not dict
        ):
            raise FormalMultiseedExecutionError(
                "authoritative publication batch boundary is invalid"
            )
        return bindings, descriptor, True

    def require_precommit_bindings(
        snapshot: Mapping[str, object],
        expected: Mapping[str, object],
        committed_seal: Mapping[str, object],
    ) -> None:
        observed_bindings, observed_descriptor, observed_authoritative = (
            authoritative_components(snapshot)
        )
        expected_bindings, _expected_descriptor, expected_authoritative = (
            authoritative_components(expected)
        )
        if observed_authoritative != expected_authoritative or _canonical_json(
            observed_bindings
        ) != _canonical_json(expected_bindings):
            raise FormalMultiseedExecutionError(
                "publication bindings drifted across manifest commit"
            )
        if observed_authoritative:
            seal_descriptor = committed_seal.get("descriptor")
            if (
                type(seal_descriptor) is not dict
                or type(observed_descriptor) is not dict
                or _canonical_json(observed_descriptor)
                != _canonical_json(seal_descriptor)
            ):
                raise FormalMultiseedExecutionError(
                    "execution manifest descriptor drifted after committed seal"
                )

    def verify_committed(
        precommit: Mapping[str, object],
        committed_seal: Mapping[str, object] | None = None,
    ) -> None:
        seal = require_committed(committed_seal)
        postcommit = binding_snapshot("snapshot")
        require_precommit_bindings(postcommit, precommit, seal)
        require_committed(seal)
        terminal = binding_snapshot("snapshot")
        require_precommit_bindings(terminal, precommit, seal)

    initially_committed = committed_snapshot()
    precommit = binding_snapshot("deep")
    committed_after_validation = committed_snapshot()
    if initially_committed is not None and committed_after_validation is None:
        raise FormalMultiseedExecutionError(
            "execution manifest disappeared during binding validation"
        )
    if committed_after_validation is not None:
        if initially_committed is not None and _canonical_json(
            committed_after_validation
        ) != _canonical_json(initially_committed):
            raise FormalMultiseedExecutionError(
                "execution manifest committed seal drifted during binding validation"
            )
        verify_committed(precommit, committed_after_validation)
        return
    uploader = getattr(output_task, "upload_artifact", None)
    flusher = getattr(output_task, "flush", None)
    if not callable(uploader) or not callable(flusher):
        raise FormalMultiseedExecutionError(
            "executor task cannot publish the execution manifest"
        )
    try:
        uploaded = uploader(
            EXECUTION_MANIFEST_ARTIFACT,
            artifact_object=frozen_manifest,
            wait_on_upload=True,
        )
    except Exception as error:
        committed = committed_snapshot()
        if committed is not None:
            verify_committed(precommit, committed)
            return
        raise FormalMultiseedExecutionError(
            "failed to upload execution manifest"
        ) from error
    if uploaded is not True:
        raise FormalMultiseedExecutionError("failed to upload execution manifest")
    try:
        flushed = flusher(wait_for_uploads=True)
    except Exception as error:
        committed = committed_snapshot()
        if committed is not None:
            verify_committed(precommit, committed)
            return
        raise FormalMultiseedExecutionError(
            "failed to flush execution manifest"
        ) from error
    if flushed is not None and flushed is not True:
        raise FormalMultiseedExecutionError("failed to flush execution manifest")
    verify_committed(precommit)


def _revalidate_manifest_tasks(
    task_class: object,
    manifest: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    project: str,
    output_task_id: str,
    source_d_task_id: str,
    source_d: str,
    teacher_task: object,
    teacher: Mapping[str, object],
    resolved_tasks: Mapping[str, Mapping[str, object]] | None = None,
    reload_tasks: bool = True,
) -> None:
    training_plan = {
        str(item["task_key"]): item
        for item in plan["training_tasks"]
        if isinstance(item, Mapping)
    }
    evaluation_plan = {
        str(item["task_key"]): item
        for item in plan["evaluation_tasks"]
        if isinstance(item, Mapping)
    }
    if resolved_tasks is None:
        resolved = _resolve_manifest_tasks(
            task_class,
            manifest,
            plan=plan,
            project=project,
            output_task_id=output_task_id,
        )
    else:
        if set(resolved_tasks) != {"training", "evaluation"}:
            raise FormalMultiseedExecutionError(
                "resolved manifest task inventory drifted"
            )
        training_tasks = resolved_tasks.get("training")
        evaluation_tasks = resolved_tasks.get("evaluation")
        if not isinstance(training_tasks, Mapping) or not isinstance(
            evaluation_tasks, Mapping
        ):
            raise FormalMultiseedExecutionError(
                "resolved manifest task inventory is invalid"
            )
        resolved = {
            "training": dict(training_tasks),
            "evaluation": dict(evaluation_tasks),
        }
    if set(resolved["training"]) != set(training_plan) or set(
        resolved["evaluation"]
    ) != set(evaluation_plan):
        raise FormalMultiseedExecutionError("resolved manifest task keys drifted")
    training_results: dict[str, dict[str, object]] = {}
    used: set[str] = set()
    for receipt in manifest["training_tasks"]:
        if not isinstance(receipt, Mapping):
            raise FormalMultiseedExecutionError("training receipt is not an object")
        key = str(receipt.get("task_key"))
        record = training_plan.get(key)
        if record is None:
            raise FormalMultiseedExecutionError("training receipt task key drifted")
        task_id = _clearml_id(receipt.get("task_id"), "training receipt")
        if task_id in used:
            raise FormalMultiseedExecutionError("manifest contains duplicate task IDs")
        used.add(task_id)
        task = resolved["training"][key]
        if _clearml_id(getattr(task, "id", None), "resumed training task") != task_id:
            raise FormalMultiseedExecutionError(
                "resumed training task identity drifted"
            )
        if reload_tasks:
            _reload(task, context="resumed training task")
        parameters = _training_parameters(
            record,
            source_d_task_id=source_d_task_id,
            teacher=teacher,
        )
        _require_bound_task(
            task,
            expected_id=task_id,
            expected_parent=output_task_id,
            expected_name=_task_run_name(record, executor_task_id=output_task_id),
            expected_parameters=parameters,
            source_d=source_d,
            queue_name=str(record["worker_queue"]),
            context="resumed training task",
        )
        result = _validate_training_completion(
            task,
            record,
            teacher=teacher,
            source_d_task_id=source_d_task_id,
        )
        expected_receipt = _task_receipt(
            record,
            result,
            parent_task_id=output_task_id,
            parameters=parameters,
        )
        if _canonical_json(expected_receipt) != _canonical_json(receipt):
            raise FormalMultiseedExecutionError("resumed training receipt drifted")
        training_results[key] = result
    for receipt in manifest["evaluation_tasks"]:
        if not isinstance(receipt, Mapping):
            raise FormalMultiseedExecutionError("evaluation receipt is not an object")
        key = str(receipt.get("task_key"))
        record = evaluation_plan.get(key)
        if record is None:
            raise FormalMultiseedExecutionError("evaluation receipt task key drifted")
        training_key = str(record["training_task_key"])
        training_result = training_results[training_key]
        training_task_id = str(training_result["task_id"])
        task_id = _clearml_id(receipt.get("task_id"), "evaluation receipt")
        if task_id in used:
            raise FormalMultiseedExecutionError("manifest contains duplicate task IDs")
        used.add(task_id)
        task = resolved["evaluation"][key]
        if _clearml_id(getattr(task, "id", None), "resumed evaluation task") != task_id:
            raise FormalMultiseedExecutionError(
                "resumed evaluation task identity drifted"
            )
        if reload_tasks:
            _reload(task, context="resumed evaluation task")
        parameters = _evaluation_parameters(
            record,
            training_task_id=training_task_id,
            model=training_result,
        )
        _require_bound_task(
            task,
            expected_id=task_id,
            expected_parent=training_task_id,
            expected_name=_task_run_name(record, executor_task_id=output_task_id),
            expected_parameters=parameters,
            source_d=source_d,
            queue_name=str(record["worker_queue"]),
            context="resumed evaluation task",
        )
        result = _validate_evaluation_completion(
            task,
            record,
            training_task_id=training_task_id,
            training_model=training_result,
        )
        expected_receipt = _task_receipt(
            record,
            result,
            parent_task_id=training_task_id,
            parameters=parameters,
        )
        if _canonical_json(expected_receipt) != _canonical_json(receipt):
            raise FormalMultiseedExecutionError("resumed evaluation receipt drifted")
    if len(used) != 90:
        raise FormalMultiseedExecutionError(
            "resumed manifest does not bind 90 unique tasks"
        )


def _dependency_snapshot(
    dependencies: Mapping[str, object],
    ids: Mapping[str, str],
    *,
    producer_sha: str,
    plan_validator_path: Path | None,
    context: str,
    reload_tasks: bool = True,
) -> dict[str, object]:
    expected_names = {"planner", "source_d", "teacher_gate", "teacher"}
    if set(dependencies) != expected_names or set(ids) != expected_names:
        raise FormalMultiseedExecutionError(f"{context} dependency inventory drifted")
    entries = {
        "planner": PLANNER_ENTRY_POINT,
        "source_d": SOURCE_D_PRODUCER_ENTRY_POINT,
        "teacher_gate": TEACHER_GATE_ENTRY_POINT,
        "teacher": TRAINING_ENTRY_POINT,
    }
    script_hashes = {
        "planner": producer_sha,
        "source_d": EXPECTED_SOURCE_D_PRODUCER_SHA256,
        "teacher_gate": EXPECTED_TEACHER_GATE_PRODUCER_SHA256,
        "teacher": EXPECTED_TEACHER_SCRIPT_SHA256,
    }
    artifact_names = {
        "planner": (PLAN_ARTIFACT, PLANNER_RECEIPT_ARTIFACT),
        "source_d": SOURCE_D_ARTIFACT_ORDER,
        "teacher_gate": (TEACHER_QUALITY_GATE_ARTIFACT,),
        "teacher": (
            TEACHER_RUN_CONTRACT_ARTIFACT,
            TEACHER_CHECKPOINT_ARTIFACT,
        ),
    }
    fingerprints: dict[str, object] = {}
    for name in ("planner", "source_d", "teacher_gate", "teacher"):
        task = dependencies[name]
        if reload_tasks:
            _reload(task, context=f"{context} {name} dependency")
        if _status(task, context=f"{context} {name} dependency") != "completed":
            raise FormalMultiseedExecutionError(
                f"{context} {name} dependency is no longer completed"
            )
        fingerprints[name] = _task_fingerprint(
            task,
            context=f"{context} {name} dependency",
            expected_id=ids[name],
            entry_point=entries[name],
            script_sha256=script_hashes[name],
            artifact_names=artifact_names[name],
        )
    plan, _ = _resolve_plan(
        dependencies["planner"],
        planner_task_id=ids["planner"],
        producer_sha256=producer_sha,
        validator_path=plan_validator_path,
    )
    source_d = _validate_source_d_evidence(
        dependencies["source_d"],
        task_id=ids["source_d"],
        plan=plan,
    )
    teacher = _validate_teacher_gate(
        dependencies["teacher_gate"],
        dependencies["teacher"],
        gate_task_id=ids["teacher_gate"],
        teacher_task_id=ids["teacher"],
    )
    return {
        "fingerprints": fingerprints,
        "plan": plan,
        "formal_plan_provenance": _formal_plan_provenance(plan),
        "source_d": source_d,
        "teacher": teacher,
    }


def _positive_finite(value: object, *, context: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{context} must be finite and positive")
    return float(value)


def _validated_executor_arguments(
    args: argparse.Namespace,
) -> dict[str, object]:
    _require_deployment_pins()
    planner_task_id = _clearml_id(
        getattr(args, "planner_task_id", None), "planner task"
    )
    source_d_task_id = _clearml_id(
        getattr(args, "source_d_task_id", None), "Source-D task"
    )
    teacher_gate_task_id = _clearml_id(
        getattr(args, "teacher_quality_gate_task_id", None),
        "teacher quality gate task",
    )
    teacher_task_id = _clearml_id(
        getattr(args, "teacher_task_id", None), "teacher task"
    )
    if (
        len(
            {
                planner_task_id,
                source_d_task_id,
                teacher_gate_task_id,
                teacher_task_id,
                SOURCE_C_TASK_ID,
            }
        )
        != 5
    ):
        raise FormalMultiseedExecutionError(
            "dependency task IDs must be globally unique"
        )
    producer_sha256 = _sha256(
        getattr(args, "expected_planner_producer_sha256", None),
        "expected planner producer",
    )
    compiled_producer_sha256 = _sha256(
        EXPECTED_PLANNER_PRODUCER_SHA256,
        "compiled planner producer pin",
    )
    if producer_sha256 != compiled_producer_sha256:
        raise FormalMultiseedExecutionError(
            "expected planner producer must match the compiled deployment pin"
        )
    project = getattr(args, "project", None)
    if type(project) is not str or not project:
        raise ValueError("project must be a non-empty string")
    poll_seconds = _positive_finite(
        getattr(args, "poll_seconds", None),
        context="poll_seconds",
    )
    timeout_hours = _positive_finite(
        getattr(args, "timeout_hours", None),
        context="timeout_hours",
    )
    return {
        "Args/planner_task_id": planner_task_id,
        "Args/source_d_task_id": source_d_task_id,
        "Args/teacher_quality_gate_task_id": teacher_gate_task_id,
        "Args/teacher_task_id": teacher_task_id,
        "Args/expected_planner_producer_sha256": producer_sha256,
        "Args/project": project,
        "Args/poll_seconds": poll_seconds,
        "Args/timeout_hours": timeout_hours,
    }


def _freeze_executor_parameters(
    output: object,
    expected: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    observed = _parameters(output, context=context)
    if set(observed) != set(expected):
        raise FormalMultiseedExecutionError(f"{context} Args key set drifted")
    for key, expected_value in expected.items():
        if not _parameter_matches(observed.get(key), expected_value):
            raise FormalMultiseedExecutionError(f"{context} {key} drifted")
    return observed


def _current_output_task(task_class: object) -> object:
    getter = getattr(task_class, "current_task", None)
    task = getter() if callable(getter) else None
    if task is None:
        raise FormalMultiseedExecutionError(
            "formal execution requires a current ClearML output task"
        )
    return task


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    plan_validator_path: Path | None = None,
) -> dict[str, object]:
    """Validate all trust roots, execute 45+45 tasks, and seal one manifest."""

    _require_deployment_pins()
    if task_class is None:
        raise FormalMultiseedExecutionError("ClearML SDK is unavailable")
    executor_expected_parameters = _validated_executor_arguments(args)
    poll_seconds = _positive_finite(
        getattr(args, "poll_seconds", None),
        context="poll_seconds",
    )
    timeout_hours = _positive_finite(
        getattr(args, "timeout_hours", None),
        context="timeout_hours",
    )
    ids = {
        "planner": _clearml_id(getattr(args, "planner_task_id", None), "planner task"),
        "source_d": _clearml_id(
            getattr(args, "source_d_task_id", None), "Source-D task"
        ),
        "teacher_gate": _clearml_id(
            getattr(args, "teacher_quality_gate_task_id", None),
            "teacher quality gate task",
        ),
        "teacher": _clearml_id(getattr(args, "teacher_task_id", None), "teacher task"),
    }
    producer_sha = _sha256(
        getattr(args, "expected_planner_producer_sha256", None),
        "expected planner producer",
    )
    project = getattr(args, "project", None)
    if type(project) is not str or not project:
        raise ValueError("project must be a non-empty string")
    execution_project_id = _project_id(task_class, project)
    output = output_task or _current_output_task(task_class)
    output_id = _clearml_id(getattr(output, "id", None), "executor task")
    if len(set([*ids.values(), output_id])) != 5:
        raise FormalMultiseedExecutionError(
            "executor and dependency task IDs must be globally unique"
        )
    _reload(output, context="executor task initial snapshot")
    if _status(output, context="executor task initial snapshot") not in {
        "created",
        "in_progress",
    }:
        raise FormalMultiseedExecutionError(
            "executor task initial status is not publishable"
        )
    runtime_source = _runtime_source()
    executor_sha = hashlib.sha256(runtime_source.encode("utf-8")).hexdigest()
    _require_no_repo_script(
        output,
        entry_point=EXECUTOR_ENTRY_POINT,
        expected_sha256=executor_sha,
        expected_source=runtime_source,
        context="executor task",
    )
    if _task_parent(output, context="executor task") != ids["planner"]:
        raise FormalMultiseedExecutionError(
            "executor task parent must be the pinned planner task"
        )
    if (
        _clearml_id(
            getattr(getattr(output, "data", None), "project", None),
            "executor task project",
        )
        != execution_project_id
    ):
        raise FormalMultiseedExecutionError(
            "executor task project must match the formal execution project"
        )
    executor_output_uri = _output_uri(output, context="executor task")
    executor_parameters = _freeze_executor_parameters(
        output, executor_expected_parameters, context="executor task"
    )
    executor_parameters_sha256 = _content_sha256(executor_parameters)
    dependencies = {
        name: task_class.get_task(task_id=task_id) for name, task_id in ids.items()
    }
    if len({id(task) for task in dependencies.values()}) != 4:
        raise FormalMultiseedExecutionError(
            "dependency lookups alias the same task object"
        )
    deadline = monotonic_clock() + timeout_hours * 3600.0
    for name, task in dependencies.items():
        if _clearml_id(getattr(task, "id", None), f"{name} dependency") != ids[name]:
            raise FormalMultiseedExecutionError(f"{name} dependency identity mismatch")
        _wait_for_completed(
            task,
            context=f"{name} dependency",
            deadline=deadline,
            poll_seconds=poll_seconds,
            monotonic_clock=monotonic_clock,
            sleeper=sleeper,
        )
    if (
        _task_parent(dependencies["source_d"], context="Source-D evidence task")
        != SOURCE_C_TASK_ID
    ):
        raise FormalMultiseedExecutionError("Source-D evidence parent mismatch")
    if (
        _task_parent(dependencies["teacher_gate"], context="teacher quality gate")
        != ids["teacher"]
    ):
        raise FormalMultiseedExecutionError("teacher quality gate parent mismatch")
    plan, _planner_receipt = _resolve_plan(
        dependencies["planner"],
        planner_task_id=ids["planner"],
        producer_sha256=producer_sha,
        validator_path=plan_validator_path,
    )
    formal_plan_provenance = _formal_plan_provenance(plan)
    source_d = _validate_source_d_evidence(
        dependencies["source_d"],
        task_id=ids["source_d"],
        plan=plan,
    )
    teacher = _validate_teacher_gate(
        dependencies["teacher_gate"],
        dependencies["teacher"],
        gate_task_id=ids["teacher_gate"],
        teacher_task_id=ids["teacher"],
    )
    fingerprints = {
        "planner": _task_fingerprint(
            dependencies["planner"],
            context="formal multi-seed planner",
            expected_id=ids["planner"],
            entry_point=PLANNER_ENTRY_POINT,
            script_sha256=producer_sha,
            artifact_names=(PLAN_ARTIFACT, PLANNER_RECEIPT_ARTIFACT),
        ),
        "source_d": _task_fingerprint(
            dependencies["source_d"],
            context="Source-D evidence task",
            expected_id=ids["source_d"],
            entry_point=SOURCE_D_PRODUCER_ENTRY_POINT,
            script_sha256=EXPECTED_SOURCE_D_PRODUCER_SHA256,
            artifact_names=SOURCE_D_ARTIFACT_ORDER,
        ),
        "teacher_gate": _task_fingerprint(
            dependencies["teacher_gate"],
            context="teacher quality gate",
            expected_id=ids["teacher_gate"],
            entry_point=TEACHER_GATE_ENTRY_POINT,
            script_sha256=EXPECTED_TEACHER_GATE_PRODUCER_SHA256,
            artifact_names=(TEACHER_QUALITY_GATE_ARTIFACT,),
        ),
        "teacher": _task_fingerprint(
            dependencies["teacher"],
            context="clean teacher",
            expected_id=ids["teacher"],
            entry_point=TRAINING_ENTRY_POINT,
            script_sha256=EXPECTED_TEACHER_SCRIPT_SHA256,
            artifact_names=(
                TEACHER_RUN_CONTRACT_ARTIFACT,
                TEACHER_CHECKPOINT_ARTIFACT,
            ),
        ),
    }
    existing_names = set(_artifact_names(output, context="executor task"))
    if existing_names:
        if existing_names != {EXECUTION_MANIFEST_ARTIFACT}:
            raise FormalMultiseedExecutionError(
                "executor contains unexpected pre-existing artifacts"
            )
        manifest = _validate_manifest_shape(
            _artifact_mapping(
                output,
                EXECUTION_MANIFEST_ARTIFACT,
                context="executor task",
            ),
            output_task_id=output_id,
            executor_script_sha256=executor_sha,
            executor_parameters_sha256=executor_parameters_sha256,
            plan=plan,
            dependency_fingerprints=fingerprints,
            source_d=source_d,
            teacher=teacher,
        )
    else:
        used_ids = {*ids.values(), output_id}
        training, evaluation = _execute_matrix(
            task_class,
            plan=plan,
            output_task_id=output_id,
            source_d_task_id=ids["source_d"],
            source_d=str(source_d["script"]),
            teacher_task=dependencies["teacher"],
            teacher=teacher,
            project=project,
            used_ids=used_ids,
            deadline=deadline,
            poll_seconds=poll_seconds,
            monotonic_clock=monotonic_clock,
            sleeper=sleeper,
        )
        manifest = _manifest_payload(
            output_task_id=output_id,
            executor_script_sha256=executor_sha,
            executor_parameters_sha256=executor_parameters_sha256,
            plan=plan,
            dependency_fingerprints=fingerprints,
            source_d=source_d,
            teacher=teacher,
            training_receipts=training,
            evaluation_receipts=evaluation,
        )
        manifest = _validate_manifest_shape(
            manifest,
            output_task_id=output_id,
            executor_script_sha256=executor_sha,
            executor_parameters_sha256=executor_parameters_sha256,
            plan=plan,
            dependency_fingerprints=fingerprints,
            source_d=source_d,
            teacher=teacher,
        )
    expected_root_snapshot = {
        "fingerprints": fingerprints,
        "plan": plan,
        "formal_plan_provenance": formal_plan_provenance,
        "source_d": source_d,
        "teacher": teacher,
    }
    executor_contract = {
        "task_id": output_id,
        "parent_task_id": ids["planner"],
        "name": EXECUTOR_TASK_NAME,
        "script_source": runtime_source,
        "script_sha256": executor_sha,
        "parameters": executor_parameters,
        "parameters_sha256": executor_parameters_sha256,
        "output_uri": executor_output_uri,
        "project_id": execution_project_id,
    }

    def validate_bindings(
        mode: Literal["deep", "snapshot"],
    ) -> Mapping[str, object]:
        return _validate_publication_bindings(
            mode,
            task_class=task_class,
            output_task=output,
            dependencies=dependencies,
            dependency_ids=ids,
            manifest=manifest,
            plan=plan,
            expected_root_snapshot=expected_root_snapshot,
            executor_contract=executor_contract,
            project=project,
            planner_producer_sha256=producer_sha,
            plan_validator_path=plan_validator_path,
        )

    _publish_manifest(
        output,
        manifest,
        validate_bindings=validate_bindings,
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-task-id", required=True)
    parser.add_argument("--source-d-task-id", required=True)
    parser.add_argument("--teacher-quality-gate-task-id", required=True)
    parser.add_argument("--teacher-task-id", required=True)
    parser.add_argument(
        "--expected-planner-producer-sha256",
        default=EXPECTED_PLANNER_PRODUCER_SHA256,
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _require_deployment_pins()
    if Task is None:
        raise FormalMultiseedExecutionError("ClearML SDK is unavailable")
    output = None
    if argv is None:
        output = Task.init(
            project_name=DEFAULT_PROJECT,
            task_name=EXECUTOR_TASK_NAME,
            reuse_last_task_id=False,
            output_uri=FILES_SERVER_URI,
            auto_connect_arg_parser=True,
        )
    args = _parser().parse_args(argv)
    expected_parameters = _validated_executor_arguments(args)
    if output is None:
        output = Task.init(
            project_name=args.project,
            task_name=EXECUTOR_TASK_NAME,
            reuse_last_task_id=False,
            output_uri=FILES_SERVER_URI,
            auto_connect_arg_parser=False,
        )
        _set_parameters_exact(output, expected_parameters)
        _reload(output, context="executor task")
    observed_parent = _task_parent(output, context="executor task")
    if observed_parent == "":
        setter = getattr(output, "set_parent", None)
        if not callable(setter):
            raise FormalMultiseedExecutionError(
                "executor task cannot bind its planner parent"
            )
        try:
            result = setter(parent_task_id=args.planner_task_id)
        except TypeError:
            result = setter(args.planner_task_id)
        if result is False:
            raise FormalMultiseedExecutionError(
                "executor task rejected its planner parent"
            )
        _reload(output, context="executor task")
    elif observed_parent != args.planner_task_id:
        raise FormalMultiseedExecutionError(
            "executor task already has a drifted non-empty parent"
        )
    if _task_parent(output, context="executor task") != args.planner_task_id:
        raise FormalMultiseedExecutionError(
            "executor planner parent did not round-trip"
        )
    manifest = run(args, task_class=Task, output_task=output)
    print(_canonical_json(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "EXECUTION_MANIFEST_ARTIFACT",
    "EXPECTED_PLANNER_PRODUCER_SHA256",
    "FormalMultiseedExecutionError",
    "generate_standalone_source",
    "main",
    "run",
)
