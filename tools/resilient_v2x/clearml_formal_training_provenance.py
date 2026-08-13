#!/usr/bin/env python3
"""Prove the recovered 26-task formal training chain is provenance-equivalent."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    from allegroai import Task
except ImportError:  # pragma: no cover - production package spelling varies
    from clearml import Task


DEFAULT_PROJECT = "ResilientV2X/Training"
FILES_SERVER_URI = "http://10.100.34.118:8081"
TRAINING_CONTROLLER_TASK_ID = "f8c36e508c7d453dadc766207a5b25b2"
SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID = TRAINING_CONTROLLER_TASK_ID
SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID = "d377543f6a574449a5d4b28cb9275dbc"
SOURCE_REVISION_NATIVE_BUILD_TASK_ID = "9055c0d3c4dd450c8a75dddfb21a56bd"
SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256 = (
    "f07079970f131abca46f8aebbabebc20a1910b0c4dcd9721b3fdc31da29058ad"
)
SOURCE_REVISION_LEGACY_TASK_ADOPTIONS = {
    "support_residual": {
        "task_id": "95e72da24d464ab08d117dedabd6652e",
        "parent_controller_task_id": "6525107e60ae4104a2800731d74ecd4e",
        "sealed_provenance_type": "completion_validation_retry",
    },
    "no_distillation": {
        "task_id": "efe6522d87a44c55b1de7f9c144e5393",
        "parent_controller_task_id": "d4b83d9b68704050aeb24a2e34540d8a",
        "sealed_provenance_type": "carried_recovery_target",
    },
}
SOURCE_REVISION_TRANSITION_ID = (
    "resilient-v2x-source-5c984ad49b52-to-ad511d88b731-custom-imports-list-v1"
)
SOURCE_REVISION_TRANSITION_SEAL_SHA256 = (
    "1c08edf7daeea7676ceb806d68dc51a4dedd5650d54665d675fe45ab750c6b78"
)
SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256 = (
    "4f06e07445e3297a8f9284ca20c78b27c3f36b4c7263ce5c7c3d9accc0a5a79c"
)
SOURCE_REVISION_TARGET_PREDECESSOR_TASK_ID = "21368e8260cc4e5392fe2dbdf116e36f"
PROGRESS_ARTIFACT = "post_main_training_progress"
TRAINING_MANIFEST_ARTIFACT = "formal_1337_training_manifest"
RUN_CONTRACT_ARTIFACT = "run_contract"
PROVENANCE_ARTIFACT = "formal_1337_training_provenance_equivalence"
PRODUCER_ENTRY_POINT = "clearml_formal_training_provenance.py"
FINAL_CHECKPOINT_ARTIFACT = "final_checkpoint_contract"
COMMON_TEACHER_AUDIT_ARTIFACT = "common_teacher_initialization_audit"

SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
SOURCE_ARCHIVE_BYTES = 1_222_481
SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
SOURCE_TREE_SHA256 = "5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6"
SOURCE_INVENTORY_BYTES = 101_195
SOURCE_INVENTORY_SHA256 = (
    "bed72cd86438f2ba932edda14052e3cd3d9589ec201d09d18a315e18a2f7cff2"
)
SOURCE_FILE_COUNT = 631
SOURCE_BYTES = 8_926_106
NEW_SOURCE_DATASET_ID = "351feedbbe81481fa31f1e9ae11a3f4e"
NEW_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-ad511d88b731.tar.zst"
NEW_SOURCE_ARCHIVE_BYTES = 1_222_492
NEW_SOURCE_ARCHIVE_SHA256 = (
    "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
)
NEW_SOURCE_TREE_SHA256 = (
    "ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"
)
NEW_SOURCE_INVENTORY_BYTES = 101_195
NEW_SOURCE_INVENTORY_SHA256 = (
    "39b1a42af65ad5df935945bd0a4eeac6e7f6e1cfdd1cc608f3dd8a708e9c5ca0"
)
NEW_SOURCE_FILE_COUNT = 631
NEW_SOURCE_BYTES = 8_926_102
NATIVE_BUNDLE_BYTES = 753_382_966
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
NATIVE_BUNDLE_SHA256 = (
    "19b8e7f5edc8216d4b43cb17854dccafe4fc9fe46995a88803e6342eeaa22b21"
)
BUILD_MANIFEST_SHA256 = (
    "21c6ab7a6e9a2823ba111289a42e7f882c5f4ba02c73175106ff251fe5864a43"
)
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
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
TRAINING_SEED = 20_250_218
GLOBAL_BATCH_SIZE = 8
MAX_EPOCHS = 50
VAL_INTERVAL = 10
PRECISION = "FP32"
DOCKER_COMMAND_SHA256 = (
    "f68e2426a223b700bc58e7bd60ca040773d355323ae32cd70fd61ee2e322569b"
)

LEGACY_BOOTSTRAP_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
EXPANDED_BOOTSTRAP_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
ALLOWED_BOOTSTRAP_SHA256 = frozenset(
    {LEGACY_BOOTSTRAP_SHA256, EXPANDED_BOOTSTRAP_SHA256}
)
CANONICAL_NORMALIZED_SCRIPT_IDENTITY_SHA256 = (
    "3493f702c9772ab89f8b7030fe01a086bf610342d678f8aee965c9c364bd6967"
)
LEGACY_NORMALIZED_SCRIPT_IDENTITY_SHA256 = (
    "f07079970f131abca46f8aebbabebc20a1910b0c4dcd9721b3fdc31da29058ad"
)
NORMALIZED_SCRIPT_IDENTITY_BY_RAW_SHA256 = {
    LEGACY_BOOTSTRAP_SHA256: LEGACY_NORMALIZED_SCRIPT_IDENTITY_SHA256,
    EXPANDED_BOOTSTRAP_SHA256: CANONICAL_NORMALIZED_SCRIPT_IDENTITY_SHA256,
}
LEGACY_SCRIPT_ALLOWED_SUBJECTS = frozenset({"support_residual", "no_distillation"})
LEGACY_NESTED_TEACHER_EXPERIMENTS = frozenset({"resilient_v2x", "support_residual"})
EXPANDED_NESTED_TEACHER_EXPERIMENTS = frozenset(
    {
        "support_residual",
        "ptf_none",
        "ptf_linear",
        "router_static",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "concat_capacity_matched",
        "resilient_v2x",
    }
)

SUBJECT_ORDER = (
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
NEW_SOURCE_SUBJECTS = frozenset(
    {"where2comm", "late_fusion", "disconet", "how2comm", "resilient_v2x"}
)
OLD_SOURCE_SUBJECTS = frozenset(SUBJECT_ORDER) - NEW_SOURCE_SUBJECTS
SOURCE_REVISION_BY_SUBJECT = {
    subject: ("new" if subject in NEW_SOURCE_SUBJECTS else "old")
    for subject in SUBJECT_ORDER
}
SOURCE_REVISION_SUBJECTS = {
    "old": [subject for subject in SUBJECT_ORDER if subject in OLD_SOURCE_SUBJECTS],
    "new": [subject for subject in SUBJECT_ORDER if subject in NEW_SOURCE_SUBJECTS],
}
SOURCE_REVISION_CHANGED_FILES = (
    {
        "path": "configs/resilient_v2x/baselines/disconet.py",
        "mode": 420,
        "module": "transvision.models.resilient_v2x.disconet_baseline",
        "old": {
            "sha256": (
                "a91621aa341cde33573e0bc95f42729f53052eaefd19b6eddaa80e1885a9a304"
            ),
            "size_bytes": 965,
            "imports_container": "tuple",
        },
        "new": {
            "sha256": (
                "82818e7157fd9c21fca04783094fc8dd9b3d6d71b0bb5898edf69496dc81c0bf"
            ),
            "size_bytes": 964,
            "imports_container": "list",
        },
    },
    {
        "path": "configs/resilient_v2x/baselines/how2comm.py",
        "mode": 420,
        "module": "transvision.models.resilient_v2x.how2comm_baseline",
        "old": {
            "sha256": (
                "fb7a40e9699945d496b7125226eef1aa9ab69b065fe91a89ed9c87b3d5b1ebf4"
            ),
            "size_bytes": 1115,
            "imports_container": "tuple",
        },
        "new": {
            "sha256": (
                "451532356eaa249cc4d41d04d5d3df3c10a670b365521f8b85077238b5c6181f"
            ),
            "size_bytes": 1114,
            "imports_container": "list",
        },
    },
    {
        "path": "configs/resilient_v2x/baselines/late_fusion.py",
        "mode": 420,
        "module": "transvision.models.resilient_v2x.late_fusion_baseline",
        "old": {
            "sha256": (
                "f7e0224f2c01497ee55008a5df14076eb312e00f2a74a8699082e8bbe391e216"
            ),
            "size_bytes": 990,
            "imports_container": "tuple",
        },
        "new": {
            "sha256": (
                "e510b227499533e62d94d3b2b7768ac7342ad5d82e0aa837989f3a446e95f324"
            ),
            "size_bytes": 989,
            "imports_container": "list",
        },
    },
    {
        "path": "configs/resilient_v2x/baselines/where2comm.py",
        "mode": 420,
        "module": "transvision.models.resilient_v2x.where2comm_baseline",
        "old": {
            "sha256": (
                "8e60be68edd79d33c2b648e383fd027f4d4b8b50ed59eee69e007d83b63ad687"
            ),
            "size_bytes": 1031,
            "imports_container": "tuple",
        },
        "new": {
            "sha256": (
                "9e90f2542fb3998550b67344ade1e4571d8e4bf1b712c8727c4e0c2c28e7c61e"
            ),
            "size_bytes": 1030,
            "imports_container": "list",
        },
    },
)
BASELINE_SUBJECTS = frozenset(
    {
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
    }
)
ABLATION_SUBJECTS = frozenset(
    {
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "concat_capacity_matched",
    }
)
IMPROVEMENT_SUBJECTS = frozenset(
    {
        "support_residual",
        "linear_no_distillation",
        "no_distillation_peak_lr_3e4",
    }
)
ZERO_FUSION_ALLOWED_SUBJECTS = frozenset({"ego_only", "fcooper"})
SUBJECT_KIND = {
    subject: (
        "baseline"
        if subject in BASELINE_SUBJECTS
        else "ablation"
        if subject in ABLATION_SUBJECTS
        else "improvement"
        if subject in IMPROVEMENT_SUBJECTS
        else "primary_method"
    )
    for subject in SUBJECT_ORDER
}

CLEARML_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
WAITING_STATUSES = frozenset({"created", "queued", "in_progress"})
TERMINAL_PARENT_STATUSES = frozenset({"failed"})
MAX_JSON_ARTIFACT_BYTES = 64 * 1024 * 1024
RAW_AUTHORITY_FIELDS = (
    "id",
    "name",
    "user",
    "company",
    "type",
    "status",
    "comment",
    "parent",
    "project",
    "output",
    "execution",
    "script",
    "tags",
    "system_tags",
    "status_message",
    "status_reason",
    "last_worker",
    "hyperparams",
    "configuration",
)
OUTPUT_VOLATILE_LIFECYCLE_FIELDS = (
    "status_message",
    "status_reason",
    "last_worker",
)
TAG_AUTHORITY_READBACK_ATTEMPTS = 5
SCRIPT_IDENTITY_KEYS = (
    "binary",
    "repository",
    "branch",
    "version_num",
    "tag",
    "working_dir",
    "entry_point",
    "diff",
    "requirements",
)
FORMAL_TAGS = (
    "ResilientV2X-suite",
    "formal-training-provenance-equivalence",
    "DAIR-CAUSAL-1337-v1",
    "cpu-controller",
)


class FormalTrainingProvenanceError(RuntimeError):
    """Raised when any provenance acceptance condition is unproven."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-controller-task-id",
        default=TRAINING_CONTROLLER_TASK_ID,
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=72.0)
    return parser


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as error:
        raise FormalTrainingProvenanceError(
            "formal provenance value is not canonical JSON"
        ) from error


def _json_copy(value: object, *, context: str) -> object:
    try:
        return json.loads(_canonical_json(value))
    except (json.JSONDecodeError, TypeError, ValueError) as error:  # pragma: no cover
        raise FormalTrainingProvenanceError(f"{context} is not JSON-safe") from error


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _source_revision_descriptor(revision: str) -> dict[str, object]:
    if revision == "old":
        values = {
            "dataset_id": SOURCE_DATASET_ID,
            "archive_name": SOURCE_ARCHIVE_NAME,
            "archive_size_bytes": SOURCE_ARCHIVE_BYTES,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "tree_sha256": SOURCE_TREE_SHA256,
            "inventory_size_bytes": SOURCE_INVENTORY_BYTES,
            "inventory_sha256": SOURCE_INVENTORY_SHA256,
            "file_count": SOURCE_FILE_COUNT,
            "source_bytes": SOURCE_BYTES,
        }
    elif revision == "new":
        values = {
            "dataset_id": NEW_SOURCE_DATASET_ID,
            "archive_name": NEW_SOURCE_ARCHIVE_NAME,
            "archive_size_bytes": NEW_SOURCE_ARCHIVE_BYTES,
            "archive_sha256": NEW_SOURCE_ARCHIVE_SHA256,
            "tree_sha256": NEW_SOURCE_TREE_SHA256,
            "inventory_size_bytes": NEW_SOURCE_INVENTORY_BYTES,
            "inventory_sha256": NEW_SOURCE_INVENTORY_SHA256,
            "file_count": NEW_SOURCE_FILE_COUNT,
            "source_bytes": NEW_SOURCE_BYTES,
        }
    else:
        raise FormalTrainingProvenanceError("unknown sealed source revision")
    return {"revision": revision, **values}


def _source_revision_certificate(revision: str) -> dict[str, object]:
    descriptor = _source_revision_descriptor(revision)
    return {
        "dataset_id": descriptor["dataset_id"],
        "tree_sha256": descriptor["tree_sha256"],
        "file_count": descriptor["file_count"],
        "source_bytes": descriptor["source_bytes"],
        "archive": {
            "name": descriptor["archive_name"],
            "size_bytes": descriptor["archive_size_bytes"],
            "sha256": descriptor["archive_sha256"],
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": descriptor["inventory_size_bytes"],
            "sha256": descriptor["inventory_sha256"],
        },
    }


def _source_revision_equivalence() -> dict[str, object]:
    old_revision = _source_revision_certificate("old")
    new_revision = _source_revision_certificate("new")
    changed_files = []
    for raw_change in SOURCE_REVISION_CHANGED_FILES:
        change = copy.deepcopy(raw_change)
        change["old"].pop("imports_container", None)
        change["new"].pop("imports_container", None)
        change["transformation"] = "custom_imports.imports singleton tuple-to-list"
        changed_files.append(change)
    return _sealed(
        {
            "schema_version": 1,
            "contract": "sealed-source-inventory-tuple-to-list-only-v1",
            "passed": True,
            "old_tree_sha256": SOURCE_TREE_SHA256,
            "new_tree_sha256": NEW_SOURCE_TREE_SHA256,
            "source_revisions": {
                SOURCE_TREE_SHA256: old_revision,
                NEW_SOURCE_TREE_SHA256: new_revision,
            },
            "path_set_equal": True,
            "mode_map_equal": True,
            "file_count": 631,
            "byte_identical_file_count": 627,
            "changed_file_count": 4,
            "changed_files": changed_files,
        }
    )


def _source_revision_subject_map() -> dict[str, object]:
    by_subject = {
        subject: (
            NEW_SOURCE_TREE_SHA256
            if SOURCE_REVISION_BY_SUBJECT[subject] == "new"
            else SOURCE_TREE_SHA256
        )
        for subject in SUBJECT_ORDER
    }
    return _sealed(
        {
            "schema_version": 1,
            "contract": "formal-26-source-revision-subject-map-v1",
            "subject_order": list(SUBJECT_ORDER),
            "source_revision_by_subject": by_subject,
            "revision_subjects": {
                SOURCE_TREE_SHA256: copy.deepcopy(SOURCE_REVISION_SUBJECTS["old"]),
                NEW_SOURCE_TREE_SHA256: copy.deepcopy(SOURCE_REVISION_SUBJECTS["new"]),
            },
            "revision_subject_counts": {
                SOURCE_TREE_SHA256: 21,
                NEW_SOURCE_TREE_SHA256: 5,
            },
            "evaluation_source_revision_tree_sha256": SOURCE_TREE_SHA256,
        }
    )


def _source_revision_for_subject(subject: str) -> dict[str, object]:
    revision = SOURCE_REVISION_BY_SUBJECT.get(subject)
    if revision is None:
        raise FormalTrainingProvenanceError(
            f"{subject!r} is not in the sealed source revision map"
        )
    return _source_revision_descriptor(revision)


def _transition_source_parameters(revision: str) -> dict[str, str]:
    source = _source_revision_descriptor(revision)
    return {
        "Args/source_dataset_id": str(source["dataset_id"]),
        "Args/source_archive_name": str(source["archive_name"]),
        "Args/source_archive_bytes": str(source["archive_size_bytes"]),
        "Args/source_archive_sha256": str(source["archive_sha256"]),
        "Args/training_dataset_id": TRAINING_DATASET_ID,
        "Args/native_bundle_bytes": str(NATIVE_BUNDLE_BYTES),
        "Args/native_bundle_sha256": NATIVE_BUNDLE_SHA256,
        "Args/build_manifest_sha256": BUILD_MANIFEST_SHA256,
    }


def _transition_config_path(subject: str) -> str:
    if subject in BASELINE_SUBJECTS:
        directory = "baselines"
    elif subject in ABLATION_SUBJECTS:
        directory = "ablations"
    elif subject in IMPROVEMENT_SUBJECTS:
        directory = "improvements"
    else:  # pragma: no cover - the primary method is target-only
        return "configs/resilient_v2x/dair_resilient_v2x.py"
    return f"configs/resilient_v2x/{directory}/{subject}.py"


def _validate_transition_template_identity(
    raw_identity: object,
    *,
    role: str,
) -> dict[str, object]:
    identity = _mapping(raw_identity, context=f"source transition {role} template")
    expected_keys = {
        "task_id",
        "entry_point",
        "script_sha256",
        "docker_command",
        "docker_image",
        "base_image_manifest_digest",
        "base_image_config_digest",
        "native_build_task_id",
        "source_parameters",
    }
    if set(identity) != expected_keys:
        raise FormalTrainingProvenanceError(
            f"source transition {role} template field inventory mismatch"
        )
    expected_task_id = (
        SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID
        if role == "source"
        else SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID
    )
    expected_revision = "old" if role == "source" else "new"
    if (
        identity.get("task_id") != expected_task_id
        or identity.get("entry_point") != "clearml_5090_bootstrap.py"
        or identity.get("native_build_task_id") != SOURCE_REVISION_NATIVE_BUILD_TASK_ID
        or type(identity.get("docker_command")) is not str
        or not str(identity["docker_command"]).strip()
        or type(identity.get("docker_image")) is not str
        or not identity["docker_image"]
    ):
        raise FormalTrainingProvenanceError(
            f"source transition {role} template identity mismatch"
        )
    _sha256(identity.get("script_sha256"), f"source transition {role} script")
    parameters = _mapping(
        identity.get("source_parameters"),
        context=f"source transition {role} template source parameters",
    )
    expected_parameters = _transition_source_parameters(expected_revision)
    if set(parameters) != set(expected_parameters) or any(
        str(parameters[key]) != expected
        for key, expected in expected_parameters.items()
    ):
        raise FormalTrainingProvenanceError(
            f"source transition {role} template source parameters mismatch"
        )
    identity["source_parameters"] = parameters
    return identity


def _validate_source_revision_transition(raw_transition: object) -> dict[str, object]:
    transition = _mapping(raw_transition, context="source revision transition")
    seal = _require_seal(transition, context="source revision transition")
    if seal != SOURCE_REVISION_TRANSITION_SEAL_SHA256:
        raise FormalTrainingProvenanceError(
            "source revision transition does not match the sealed real-template pins"
        )
    expected_keys = {
        "schema_version",
        "contract_type",
        "transition_id",
        "source_controller_task_id",
        "source_template_identity",
        "target_template_identity",
        "source_revision",
        "target_revision",
        "parameter_equivalence",
        "inventory_equivalence",
        "subject_policy",
        "seal_sha256",
    }
    if set(transition) != expected_keys:
        raise FormalTrainingProvenanceError(
            "source revision transition field inventory mismatch"
        )
    if (
        transition.get("schema_version") != 1
        or transition.get("contract_type") != "resilient_v2x_source_revision_transition"
        or transition.get("transition_id") != SOURCE_REVISION_TRANSITION_ID
        or transition.get("source_controller_task_id")
        != SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
    ):
        raise FormalTrainingProvenanceError("source revision transition root mismatch")
    source_identity = _validate_transition_template_identity(
        transition.get("source_template_identity"), role="source"
    )
    target_identity = _validate_transition_template_identity(
        transition.get("target_template_identity"), role="target"
    )
    source_common = dict(source_identity)
    target_common = dict(target_identity)
    for identity in (source_common, target_common):
        identity.pop("task_id")
        identity.pop("source_parameters")
    if source_common != target_common:
        raise FormalTrainingProvenanceError(
            "source transition template execution identities are cross-spliced"
        )
    source_parameters = _transition_source_parameters("old")
    target_parameters = _transition_source_parameters("new")
    expected_revisions = {
        "source_revision": {
            "parameters": source_parameters,
            "tree_sha256": SOURCE_TREE_SHA256,
            "inventory_sha256": SOURCE_INVENTORY_SHA256,
            "inventory_bytes": SOURCE_INVENTORY_BYTES,
            "inventory_file_count": SOURCE_FILE_COUNT,
            "source_bytes": SOURCE_BYTES,
        },
        "target_revision": {
            "parameters": target_parameters,
            "tree_sha256": NEW_SOURCE_TREE_SHA256,
            "inventory_sha256": NEW_SOURCE_INVENTORY_SHA256,
            "inventory_bytes": NEW_SOURCE_INVENTORY_BYTES,
            "inventory_file_count": NEW_SOURCE_FILE_COUNT,
            "source_bytes": NEW_SOURCE_BYTES,
        },
    }
    for key, expected in expected_revisions.items():
        if transition.get(key) != expected:
            raise FormalTrainingProvenanceError(
                f"source revision transition {key} mismatch"
            )
    changed_keys = list(source_parameters)[:4]
    unchanged_keys = list(source_parameters)[4:]
    parameter_equivalence = _mapping(
        transition.get("parameter_equivalence"),
        context="source transition parameter equivalence",
    )
    if set(parameter_equivalence) != {
        "changed_keys",
        "unchanged_keys",
        "changes",
        "shared_non_source_parameter_count",
        "shared_non_source_parameter_keys",
        "shared_non_source_parameters_sha256",
        "template_execution_identity_equal",
    }:
        raise FormalTrainingProvenanceError(
            "source transition parameter equivalence field inventory mismatch"
        )
    expected_changes = {
        key: {"source": source_parameters[key], "target": target_parameters[key]}
        for key in changed_keys
    }
    shared_keys = parameter_equivalence.get("shared_non_source_parameter_keys")
    shared_count = parameter_equivalence.get("shared_non_source_parameter_count")
    if (
        parameter_equivalence.get("changed_keys") != changed_keys
        or parameter_equivalence.get("unchanged_keys") != unchanged_keys
        or parameter_equivalence.get("changes") != expected_changes
        or parameter_equivalence.get("template_execution_identity_equal") is not True
        or not isinstance(shared_keys, list)
        or any(type(key) is not str for key in shared_keys)
        or shared_keys != sorted(set(shared_keys))
        or type(shared_count) is not int
        or shared_count != len(shared_keys)
    ):
        raise FormalTrainingProvenanceError(
            "source transition contains an unknown parameter difference"
        )
    _sha256(
        parameter_equivalence.get("shared_non_source_parameters_sha256"),
        "source transition shared non-source parameters",
    )
    expected_changed_files = [
        {
            "path": str(change["path"]),
            "source_sha256": str(change["old"]["sha256"]),
            "target_sha256": str(change["new"]["sha256"]),
            "semantic_change": "python_tuple_to_list_only",
        }
        for change in SOURCE_REVISION_CHANGED_FILES
    ]
    expected_inventory = {
        "source_file_count": SOURCE_FILE_COUNT,
        "target_file_count": NEW_SOURCE_FILE_COUNT,
        "unchanged_file_count": 627,
        "changed_file_count": len(expected_changed_files),
        "added_paths": [],
        "removed_paths": [],
        "all_other_paths_byte_identical": True,
        "changed_files": expected_changed_files,
    }
    if transition.get("inventory_equivalence") != expected_inventory:
        raise FormalTrainingProvenanceError(
            "source transition inventory difference is not the sealed four-file delta"
        )
    old_subjects = SOURCE_REVISION_SUBJECTS["old"]
    new_subjects = SOURCE_REVISION_SUBJECTS["new"]
    expected_receipts = [
        {
            "experiment": subject,
            "config_path": _transition_config_path(subject),
            "config_inventory_status": "byte_identical_across_revisions",
            "template_role": "source",
        }
        for subject in old_subjects
    ]
    expected_policy = {
        "source_template_adoptable_experiments": old_subjects,
        "target_template_required_experiments": new_subjects,
        "source_template_adoptable_count": 21,
        "target_template_required_count": 5,
        "source_subject_config_receipts": expected_receipts,
        "legacy_script_adoption_exceptions": [
            {
                "experiment": subject,
                "task_id": binding["task_id"],
                "parent_controller_task_id": binding["parent_controller_task_id"],
                "sealed_provenance_type": binding["sealed_provenance_type"],
                "legacy_script_sha256": (SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256),
                "canonical_script_sha256": source_identity["script_sha256"],
                "legacy_nested_teacher_membership": (
                    subject in LEGACY_NESTED_TEACHER_EXPERIMENTS
                ),
                "canonical_nested_teacher_membership": (
                    subject in EXPANDED_NESTED_TEACHER_EXPERIMENTS
                ),
                "subject_semantics_equal": (
                    (subject in LEGACY_NESTED_TEACHER_EXPERIMENTS)
                    == (subject in EXPANDED_NESTED_TEACHER_EXPERIMENTS)
                ),
            }
            for subject, binding in SOURCE_REVISION_LEGACY_TASK_ADOPTIONS.items()
        ],
        "policy": (
            "old tasks require exact source template and unchanged selected config; "
            "five target subjects require the new template/source"
        ),
    }
    if transition.get("subject_policy") != expected_policy:
        raise FormalTrainingProvenanceError(
            "source transition subject policy does not preserve the exact 21:5 split"
        )
    transition["source_template_identity"] = source_identity
    transition["target_template_identity"] = target_identity
    return transition


def _clearml_id(value: object, context: str) -> str:
    result = str(value or "")
    if CLEARML_ID_PATTERN.fullmatch(result) is None:
        raise FormalTrainingProvenanceError(
            f"{context} must be a lowercase 32-hex ClearML ID"
        )
    return result


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if SHA256_PATTERN.fullmatch(result) is None:
        raise FormalTrainingProvenanceError(f"{context} must be a lowercase SHA-256")
    return result


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = _sha256(value.get("seal_sha256"), f"{context} seal")
    expected = _sealed(value)["seal_sha256"]
    if observed != expected:
        raise FormalTrainingProvenanceError(f"{context} seal SHA-256 mismatch")
    return observed


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FormalTrainingProvenanceError(f"{context} is not an object")
    copied = _json_copy(value, context=context)
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise FormalTrainingProvenanceError(f"{context} is not an object")
    return copied


def _sequence(value: object, *, context: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise FormalTrainingProvenanceError(f"{context} is not a sequence")
    return list(value)


def _backend_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    serializer = getattr(value, "to_dict", None)
    if not callable(serializer):
        raise FormalTrainingProvenanceError(f"{context} is not a raw backend record")
    try:
        result = serializer()
    except Exception as error:
        raise FormalTrainingProvenanceError(
            f"{context} cannot be serialized"
        ) from error
    if not isinstance(result, Mapping):
        raise FormalTrainingProvenanceError(f"{context} serialization is invalid")
    return result


def _tag_list(value: object, *, context: str) -> list[str]:
    if value is None:
        return []
    items = _sequence(value, context=context)
    if any(type(item) is not str for item in items):
        raise FormalTrainingProvenanceError(f"{context} contains a non-string tag")
    return [str(item) for item in items]


def _unique_tag_set(value: object, *, context: str) -> frozenset[str]:
    tags = _tag_list(value, context=context)
    unique = frozenset(tags)
    if len(unique) != len(tags):
        raise FormalTrainingProvenanceError(f"{context} contains duplicate tags")
    return unique


def _raw_authority(value: object, *, context: str) -> dict[str, object]:
    raw = _backend_mapping(value, context=context)
    selected = {field: raw.get(field) for field in RAW_AUTHORITY_FIELDS}
    selected["tags"] = _tag_list(selected.get("tags"), context=f"{context} tags")
    selected["system_tags"] = _tag_list(
        selected.get("system_tags"), context=f"{context} system tags"
    )
    execution = selected.get("execution") or {}
    execution_mapping = _backend_mapping(execution, context=f"{context} execution")
    execution_copy = _mapping(execution_mapping, context=f"{context} execution")
    artifacts = execution_copy.pop("artifacts", None) or []
    records: list[dict[str, object]] = []
    names: set[str] = set()
    for index, artifact in enumerate(
        _sequence(artifacts, context=f"{context} artifacts")
    ):
        record = _mapping(
            _backend_mapping(
                artifact,
                context=f"{context} artifact record {index}",
            ),
            context=f"{context} artifact record {index}",
        )
        name = record.get("key")
        if type(name) is not str or not name:
            raise FormalTrainingProvenanceError(
                f"{context} artifact record {index} has no key"
            )
        if name in names:
            raise FormalTrainingProvenanceError(
                f"{context} has duplicate artifact {name!r}"
            )
        names.add(name)
        records.append(record)
    records.sort(key=lambda item: str(item["key"]))
    selected["execution"] = execution_copy
    selected["artifacts"] = records
    copied = _mapping(selected, context=f"{context} authority")
    _clearml_id(copied.get("id"), f"{context} ID")
    return copied


def _batch_authority_snapshot(
    task_class: object,
    task_ids: Sequence[str],
    *,
    context: str,
) -> dict[str, dict[str, object]]:
    ordered = list(task_ids)
    if not ordered or len(set(ordered)) != len(ordered):
        raise FormalTrainingProvenanceError(f"{context} task inventory is invalid")
    query = getattr(task_class, "_query_tasks", None)
    if not callable(query):
        raise FormalTrainingProvenanceError(
            f"{context} cannot issue an authoritative batch read"
        )
    try:
        records = query(
            task_ids=ordered,
            fetch_only_first_page=True,
            only_fields=list(RAW_AUTHORITY_FIELDS),
            search_hidden=True,
        )
    except Exception as error:
        raise FormalTrainingProvenanceError(
            f"{context} authoritative batch read failed"
        ) from error
    result: dict[str, dict[str, object]] = {}
    for index, record in enumerate(_sequence(records, context=f"{context} response")):
        snapshot = _raw_authority(record, context=f"{context} record {index}")
        task_id = str(snapshot["id"])
        if task_id in result:
            raise FormalTrainingProvenanceError(
                f"{context} returned duplicate task {task_id}"
            )
        result[task_id] = snapshot
    if set(result) != set(ordered):
        raise FormalTrainingProvenanceError(
            f"{context} task inventory mismatch; expected={sorted(ordered)!r}, "
            f"observed={sorted(result)!r}"
        )
    return {task_id: result[task_id] for task_id in ordered}


def _raw_status(authority: Mapping[str, object], *, context: str) -> str:
    raw = authority.get("status")
    value = getattr(raw, "value", raw)
    if type(value) is not str:
        raise FormalTrainingProvenanceError(f"{context} status is invalid")
    return value


def _raw_parent(authority: Mapping[str, object], *, context: str) -> str:
    value = authority.get("parent")
    if value in {None, ""}:
        return ""
    return _clearml_id(value, f"{context} parent")


def _raw_artifact_names(
    authority: Mapping[str, object], *, context: str
) -> tuple[str, ...]:
    records = _sequence(authority.get("artifacts"), context=f"{context} artifacts")
    names: list[str] = []
    for index, record in enumerate(records):
        mapping = _mapping(record, context=f"{context} artifact {index}")
        name = mapping.get("key")
        if type(name) is not str or not name:
            raise FormalTrainingProvenanceError(
                f"{context} artifact {index} has no key"
            )
        names.append(name)
    return tuple(names)


def _raw_script(authority: Mapping[str, object], *, context: str) -> tuple[str, str]:
    script = _raw_script_mapping(authority, context=context)
    if script.get("repository") != "" or script.get("working_dir") != ".":
        raise FormalTrainingProvenanceError(f"{context} is not a standalone script")
    entry_point = script.get("entry_point")
    source = script.get("diff")
    if type(entry_point) is not str or type(source) is not str or not source:
        raise FormalTrainingProvenanceError(f"{context} script metadata is incomplete")
    return entry_point, source


def _raw_script_mapping(
    authority: Mapping[str, object], *, context: str
) -> dict[str, object]:
    return _mapping(authority.get("script"), context=f"{context} script")


def _normalized_task_script_identity_sha256(
    authority: Mapping[str, object], *, context: str
) -> str:
    script = _raw_script_mapping(authority, context=context)
    identity = {key: script.get(key) for key in SCRIPT_IDENTITY_KEYS}
    return _content_sha256(identity)


def _wrapper_status(task: object, *, context: str) -> str:
    value = getattr(task, "status", None)
    if callable(value):
        value = value()
    value = getattr(value, "value", value)
    if type(value) is not str:
        raise FormalTrainingProvenanceError(f"{context} status is invalid")
    return value


def _reload_for_wait(task: object, *, context: str) -> None:
    method = getattr(task, "_reload", None)
    if not callable(method):
        method = getattr(task, "reload", None)
    if not callable(method):
        raise FormalTrainingProvenanceError(f"{context} cannot be refreshed")
    try:
        snapshot = method()
    except Exception as error:
        raise FormalTrainingProvenanceError(f"{context} refresh failed") from error
    if snapshot is None:
        raise FormalTrainingProvenanceError(f"{context} refresh returned no snapshot")


def _wait_for_completed(
    controller: object,
    *,
    deadline: float,
    poll_seconds: float,
    monotonic_clock: Callable[[], float],
    sleeper: Callable[[float], None],
    required_artifacts: Sequence[str] = (),
) -> None:
    required = tuple(required_artifacts)
    if any(type(name) is not str or not name for name in required):
        raise FormalTrainingProvenanceError(
            "required controller artifact names must be non-empty strings"
        )
    while True:
        status = _wrapper_status(controller, context="training controller")
        if status == "completed":
            artifacts = getattr(controller, "artifacts", None)
            missing = [
                name
                for name in required
                if not isinstance(artifacts, Mapping) or name not in artifacts
            ]
            if not missing:
                return
        elif status not in WAITING_STATUSES:
            raise FormalTrainingProvenanceError(
                f"training controller is terminal {status!r}, not completed"
            )
        if monotonic_clock() >= deadline:
            if status == "completed":
                raise TimeoutError(
                    "timed out waiting for completed training controller "
                    f"artifacts {missing!r}"
                )
            raise TimeoutError("timed out waiting for the formal training controller")
        sleeper(poll_seconds)
        _reload_for_wait(controller, context="training controller")


def _artifact_proxy(task: object, name: str, *, context: str) -> object:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise FormalTrainingProvenanceError(f"{context} lacks artifact {name!r}")
    return artifacts[name]


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_json_path(path_value: str | bytes, *, context: str) -> dict[str, object]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise FormalTrainingProvenanceError(
            f"{context} secure local-file flags are unavailable"
        )
    path = Path(os.fsdecode(path_value))
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or before.st_dev != opened.st_dev
                or before.st_ino != opened.st_ino
                or opened.st_size > MAX_JSON_ARTIFACT_BYTES
            ):
                raise FormalTrainingProvenanceError(f"{context} path is unsafe")
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise FormalTrainingProvenanceError(f"{context} was truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise FormalTrainingProvenanceError(f"{context} grew during read")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final = path.lstat()
    except FormalTrainingProvenanceError:
        raise
    except OSError as error:
        raise FormalTrainingProvenanceError(f"{context} cannot be read") from error
    if _stat_signature(before) != _stat_signature(final) or _stat_signature(
        opened
    ) != _stat_signature(after):
        raise FormalTrainingProvenanceError(f"{context} changed during read")
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FormalTrainingProvenanceError(f"{context} is not valid JSON") from error
    return _mapping(value, context=context)


def _fresh_artifact_mapping(
    task: object, name: str, *, context: str
) -> dict[str, object]:
    artifact = _artifact_proxy(task, name, context=context)
    getter = getattr(artifact, "get", None)
    if not callable(getter):
        raise FormalTrainingProvenanceError(
            f"{context} artifact {name!r} cannot be read"
        )
    try:
        value = getter(force_download=True)
    except Exception as error:
        raise FormalTrainingProvenanceError(
            f"{context} artifact {name!r} cannot be force-downloaded"
        ) from error
    if isinstance(value, Mapping):
        return _mapping(value, context=f"{context} artifact {name!r}")
    try:
        path_value = os.fspath(value)
    except Exception as error:
        raise FormalTrainingProvenanceError(
            f"{context} artifact {name!r} is not a JSON object or valid local path"
        ) from error
    return _read_json_path(path_value, context=f"{context} artifact {name!r}")


def _parameters(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise FormalTrainingProvenanceError(f"{context} cannot expose parameters")
    try:
        value = getter(backwards_compatibility=False, cast=False)
    except TypeError:
        try:
            value = getter(cast=False)
        except TypeError:
            value = getter()
    return _mapping(value, context=f"{context} parameters")


def _fileserver_url(value: object, *, expected_filename: str, context: str) -> str:
    result = str(value or "")
    try:
        parsed = urlsplit(result)
        port = parsed.port
    except ValueError as error:
        raise FormalTrainingProvenanceError(f"{context} URL is invalid") from error
    filename = unquote(parsed.path.rsplit("/", 1)[-1])
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != "10.100.34.118"
        or port != 8081
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or filename != expected_filename
    ):
        raise FormalTrainingProvenanceError(
            f"{context} is not the expected fileserver URL"
        )
    return result


def _validate_source_progress_artifact_readback(
    value: object,
    *,
    source_controller_task_id: str,
    expected_revision: object,
    expected_seal_sha256: object,
) -> dict[str, object]:
    readback = _mapping(value, context="source progress artifact readback")
    if set(readback) != {
        "force_download",
        "revision",
        "seal_sha256",
        "stable_readbacks",
        "url",
    }:
        raise FormalTrainingProvenanceError(
            "source progress artifact readback field inventory mismatch"
        )
    seal_sha256 = _sha256(expected_seal_sha256, "source progress seal")
    if (
        type(expected_revision) is not int
        or expected_revision < 1
        or readback.get("force_download") is not True
        or readback.get("stable_readbacks") != 2
        or type(readback.get("revision")) is not int
        or readback.get("revision") != expected_revision
        or readback.get("seal_sha256") != seal_sha256
    ):
        raise FormalTrainingProvenanceError(
            "source progress artifact readback contract drifted"
        )
    url = _fileserver_url(
        readback.get("url"),
        expected_filename=f"{PROGRESS_ARTIFACT}.json",
        context="source progress artifact readback",
    )
    path_parts = [part for part in unquote(urlsplit(url).path).split("/") if part]
    expected_tail = [
        "artifacts",
        PROGRESS_ARTIFACT,
        f"{PROGRESS_ARTIFACT}.json",
    ]
    if (
        len(path_parts) < 4
        or path_parts[-3:] != expected_tail
        or not path_parts[-4].endswith(f".{source_controller_task_id}")
    ):
        raise FormalTrainingProvenanceError(
            "source progress artifact readback is not bound to the source controller"
        )
    return readback


def _validate_legacy_script_compatibility_receipt(
    value: object,
    *,
    subject: str,
    task_id: str,
    task_script_sha256: str,
    canonical_script_sha256: str,
    source_progress_seal_sha256: object,
) -> None:
    receipt = _mapping(value, context=f"legacy script compatibility receipt {subject}")
    expected_keys = {
        "canonical_nested_teacher_membership",
        "canonical_script_sha256",
        "contract_type",
        "exception_scope",
        "experiment",
        "legacy_nested_teacher_membership",
        "legacy_script_sha256",
        "parent_controller_task_id",
        "result",
        "schema_version",
        "seal_sha256",
        "sealed_completed_step_sha256",
        "sealed_provenance_type",
        "sealed_recovery_provenance",
        "sealed_recovery_provenance_sha256",
        "source_controller_task_id",
        "source_progress_seal_sha256",
        "subject_semantics_equal",
        "task_id",
    }
    if set(receipt) != expected_keys:
        raise FormalTrainingProvenanceError(
            f"legacy script compatibility receipt {subject} field inventory mismatch"
        )
    _require_seal(receipt, context=f"legacy script compatibility receipt {subject}")
    exception = SOURCE_REVISION_LEGACY_TASK_ADOPTIONS.get(subject)
    expected_legacy_nested = subject in LEGACY_NESTED_TEACHER_EXPERIMENTS
    expected_canonical_nested = subject in EXPANDED_NESTED_TEACHER_EXPERIMENTS
    expected_provenance = (
        "explicit_failed_recovery_target"
        if isinstance(exception, Mapping)
        and exception.get("sealed_provenance_type") == "carried_recovery_target"
        else "full_completion_contract_revalidation_required"
    )
    if (
        not isinstance(exception, Mapping)
        or receipt.get("schema_version") != 1
        or receipt.get("contract_type") != "source_revision_legacy_script_compatibility"
        or receipt.get("result") != "pass"
        or receipt.get("experiment") != subject
        or receipt.get("task_id") != task_id
        or receipt.get("parent_controller_task_id")
        != exception.get("parent_controller_task_id")
        or receipt.get("source_controller_task_id")
        != SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
        or receipt.get("source_progress_seal_sha256") != source_progress_seal_sha256
        or receipt.get("legacy_script_sha256")
        != SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256
        or task_script_sha256 != SOURCE_REVISION_LEGACY_TASK_SCRIPT_SHA256
        or receipt.get("canonical_script_sha256") != canonical_script_sha256
        or receipt.get("sealed_provenance_type")
        != exception.get("sealed_provenance_type")
        or receipt.get("sealed_recovery_provenance") != expected_provenance
        or receipt.get("legacy_nested_teacher_membership") is not expected_legacy_nested
        or receipt.get("canonical_nested_teacher_membership")
        is not expected_canonical_nested
        or expected_legacy_nested is not expected_canonical_nested
        or receipt.get("subject_semantics_equal") is not True
        or receipt.get("exception_scope") != "script_identity_only"
    ):
        raise FormalTrainingProvenanceError(
            f"legacy script compatibility receipt {subject} is invalid"
        )
    for key in (
        "sealed_recovery_provenance_sha256",
        "sealed_completed_step_sha256",
    ):
        _sha256(
            receipt.get(key), f"legacy script compatibility receipt {subject} {key}"
        )


def _parameter_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return str(actual).casefold() == str(expected).casefold()
    return str(actual) == str(expected)


def _training_task_static_binding(
    task: object,
    *,
    subject: str,
    task_id: str,
    predecessor_task_id: str,
    entry: Mapping[str, object],
) -> dict[str, object]:
    parameters = _parameters(task, context=f"training task {subject}")
    source_revision = _source_revision_for_subject(subject)
    expected_parameters = {
        "Args/experiment_from_task": subject,
        "Args/source_dataset_id": source_revision["dataset_id"],
        "Args/source_archive_name": source_revision["archive_name"],
        "Args/source_archive_bytes": source_revision["archive_size_bytes"],
        "Args/source_archive_sha256": source_revision["archive_sha256"],
        "Args/training_dataset_id": TRAINING_DATASET_ID,
        "Args/native_bundle_sha256": NATIVE_BUNDLE_SHA256,
        "Args/build_manifest_sha256": BUILD_MANIFEST_SHA256,
        "Args/predecessor_task_id": predecessor_task_id,
        "Args/gpus": 4,
        "Args/stage": "all",
        "Args/max_epochs": MAX_EPOCHS,
        "Args/amp": False,
        "Args/training_seed": TRAINING_SEED,
        "Args/teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
    }
    for key, expected in expected_parameters.items():
        if key not in parameters or not _parameter_matches(parameters[key], expected):
            raise FormalTrainingProvenanceError(
                f"training task {subject} parameter {key} mismatch"
            )
    docker_getter = getattr(task, "get_base_docker", None)
    if not callable(docker_getter):
        raise FormalTrainingProvenanceError(
            f"training task {subject} cannot expose its Docker command"
        )
    docker = docker_getter()
    if type(docker) is not str or not docker.strip():
        raise FormalTrainingProvenanceError(
            f"training task {subject} Docker command is invalid"
        )
    docker_sha256 = hashlib.sha256(docker.strip().encode("utf-8")).hexdigest()
    if docker_sha256 != DOCKER_COMMAND_SHA256:
        raise FormalTrainingProvenanceError(
            f"training task {subject} Docker command SHA-256 mismatch"
        )
    model_getter = getattr(task, "get_models", None)
    if not callable(model_getter):
        raise FormalTrainingProvenanceError(
            f"training task {subject} cannot expose output models"
        )
    models = model_getter()
    if not isinstance(models, Mapping):
        raise FormalTrainingProvenanceError(
            f"training task {subject} output model mapping is invalid"
        )
    outputs = models.get("output")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise FormalTrainingProvenanceError(
            f"training task {subject} output models are invalid"
        )
    candidates = [
        model
        for model in outputs
        if str(getattr(model, "id", "") or "") == str(entry["model_id"])
    ]
    if len(candidates) != 1:
        raise FormalTrainingProvenanceError(
            f"training task {subject} must expose one manifest final model"
        )
    model = candidates[0]
    expected_model = {
        "id": entry["model_id"],
        "task": task_id,
        "name": entry["model_name"],
        "url": entry["model_url"],
    }
    for attribute, expected in expected_model.items():
        if str(getattr(model, attribute, "") or "") != str(expected):
            raise FormalTrainingProvenanceError(
                f"training task {subject} final OutputModel {attribute} mismatch"
            )
    return {
        "source_revision_tree_sha256": source_revision["tree_sha256"],
        "source_dataset_id": source_revision["dataset_id"],
        "source_archive_name": source_revision["archive_name"],
        "source_archive_bytes": source_revision["archive_size_bytes"],
        "source_archive_sha256": source_revision["archive_sha256"],
        "required_parameters": expected_parameters,
        "observed_parameter_keys": sorted(parameters),
        "observed_parameters_sha256": _content_sha256(parameters),
        "docker_command": docker.strip(),
        "docker_command_sha256": docker_sha256,
        "final_output_model": expected_model,
    }


def _validate_source_progress_template_equivalence(
    value: object,
    *,
    expected_identity: Mapping[str, object],
) -> dict[str, object]:
    receipt = _mapping(
        value,
        context="source progress template equivalence",
    )
    expected_keys = {
        "schema_version",
        "contract_type",
        "result",
        "context",
        "observed_identity",
        "expected_identity",
        "observed_identity_sha256",
        "expected_identity_sha256",
        "exact_identity_keys",
        "exact_non_source_identity",
        "exact_source_parameter_keys",
        "parameter_values_equivalent",
        "normalized_type_only_keys",
        "type_normalizations",
        "seal_sha256",
    }
    if set(receipt) != expected_keys:
        raise FormalTrainingProvenanceError(
            "source progress template equivalence field inventory mismatch"
        )
    seal = _require_seal(
        receipt,
        context="source progress template equivalence",
    )
    if seal != SOURCE_PROGRESS_TEMPLATE_EQUIVALENCE_SEAL_SHA256:
        raise FormalTrainingProvenanceError(
            "source progress template equivalence seal pin mismatch"
        )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("contract_type")
        != "template_identity_parameter_type_equivalence"
        or receipt.get("result") != "pass"
        or receipt.get("context") != "source-revision controller template identity"
        or receipt.get("exact_non_source_identity") is not True
        or receipt.get("exact_source_parameter_keys") is not True
        or receipt.get("parameter_values_equivalent") is not True
    ):
        raise FormalTrainingProvenanceError(
            "source progress template equivalence contract drifted"
        )

    expected = _mapping(
        receipt.get("expected_identity"),
        context="source progress expected template identity",
    )
    observed = _mapping(
        receipt.get("observed_identity"),
        context="source progress observed template identity",
    )
    canonical_expected = _json_copy(
        expected_identity,
        context="source progress canonical expected template identity",
    )
    if expected != canonical_expected:
        raise FormalTrainingProvenanceError(
            "source progress expected template identity is cross-spliced"
        )
    if set(observed) != set(expected):
        raise FormalTrainingProvenanceError(
            "source progress observed template identity keys drifted"
        )
    if receipt.get("exact_identity_keys") != sorted(expected):
        raise FormalTrainingProvenanceError(
            "source progress template identity key receipt drifted"
        )
    if receipt.get("expected_identity_sha256") != _content_sha256(expected):
        raise FormalTrainingProvenanceError(
            "source progress expected template identity digest mismatch"
        )
    if receipt.get("observed_identity_sha256") != _content_sha256(observed):
        raise FormalTrainingProvenanceError(
            "source progress observed template identity digest mismatch"
        )

    expected_source = _mapping(
        expected.get("source_parameters"),
        context="source progress expected template parameters",
    )
    observed_source = _mapping(
        observed.get("source_parameters"),
        context="source progress observed template parameters",
    )
    if set(observed_source) != set(expected_source):
        raise FormalTrainingProvenanceError(
            "source progress template parameter keys drifted"
        )
    expected_common = dict(expected)
    observed_common = dict(observed)
    expected_common.pop("source_parameters")
    observed_common.pop("source_parameters")
    if observed_common != expected_common:
        raise FormalTrainingProvenanceError(
            "source progress non-parameter template identity drifted"
        )

    normalizations: list[dict[str, str]] = []
    for key in expected_source:
        expected_value = expected_source[key]
        observed_value = observed_source[key]
        if not _parameter_matches(observed_value, expected_value):
            raise FormalTrainingProvenanceError(
                f"source progress template parameter drifted: {key}"
            )
        if type(observed_value) is not type(expected_value):
            normalizations.append(
                {
                    "key": key,
                    "observed_type": type(observed_value).__name__,
                    "expected_type": type(expected_value).__name__,
                    "canonical_value": str(expected_value),
                }
            )
    if (
        receipt.get("normalized_type_only_keys")
        != [item["key"] for item in normalizations]
        or receipt.get("type_normalizations") != normalizations
        or receipt.get("normalized_type_only_keys") != ["Args/native_bundle_bytes"]
    ):
        raise FormalTrainingProvenanceError(
            "source progress template type normalization drifted"
        )
    return receipt


def _validate_source_revision_recovery(
    recovery: Mapping[str, object],
    *,
    controller_task_id: str,
    progress_template: object,
    steps: Sequence[Mapping[str, object]],
) -> None:
    expected_recovery_keys = {
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
    }
    if set(recovery) != expected_recovery_keys:
        raise FormalTrainingProvenanceError(
            "training recovery schema-v4 field inventory mismatch"
        )
    if (
        recovery.get("schema_version") != 4
        or recovery.get("mode") != "failed_controller_immutable_fork"
        or recovery.get("source_controller_status") != "failed"
        or recovery.get("source_controller_task_id")
        != SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
        or controller_task_id == SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
        or recovery.get("source_patch") != "nested-teacher-config-consistency-v1"
    ):
        raise FormalTrainingProvenanceError(
            "training recovery is not the sealed mixed-source successor contract"
        )
    _validate_source_progress_artifact_readback(
        recovery.get("source_progress_artifact_readback"),
        source_controller_task_id=SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
        expected_revision=recovery.get("source_progress_revision"),
        expected_seal_sha256=recovery.get("source_progress_seal_sha256"),
    )
    transition = _validate_source_revision_transition(
        recovery.get("source_revision_transition")
    )
    source_identity = _mapping(
        transition.get("source_template_identity"),
        context="source transition source template",
    )
    target_identity = _mapping(
        transition.get("target_template_identity"),
        context="source transition target template",
    )
    _validate_source_progress_template_equivalence(
        recovery.get("source_progress_template_equivalence"),
        expected_identity=source_identity,
    )
    if progress_template != target_identity:
        raise FormalTrainingProvenanceError(
            "training progress template is not the sealed new-source target identity"
        )
    if (
        recovery.get("source_template_script_sha256")
        != source_identity["script_sha256"]
        or recovery.get("target_template_script_sha256")
        != target_identity["script_sha256"]
    ):
        raise FormalTrainingProvenanceError(
            "training recovery template script binding is cross-spliced"
        )
    adopted = _mapping(
        recovery.get("adopted_task_ids"), context="recovery adopted tasks"
    )
    predecessors = _mapping(
        recovery.get("adopted_predecessor_task_ids"),
        context="recovery adopted predecessors",
    )
    roles = _mapping(
        recovery.get("adopted_task_template_roles"),
        context="recovery adopted template roles",
    )
    receipts = _mapping(
        recovery.get("adopted_task_binding_receipts"),
        context="recovery adopted binding receipts",
    )
    adopted_subjects = set(adopted)
    old_subjects = set(SOURCE_REVISION_SUBJECTS["old"])
    new_subjects = set(SOURCE_REVISION_SUBJECTS["new"])
    if (
        set(predecessors) != adopted_subjects
        or set(roles) != adopted_subjects
        or set(receipts) != adopted_subjects
        or not old_subjects <= adopted_subjects
        or not adopted_subjects <= old_subjects | new_subjects
    ):
        raise FormalTrainingProvenanceError(
            "training recovery adoption inventory violates the exact 21:5 split"
        )
    step_by_subject = {
        str(step.get("experiment")): step for step in steps if isinstance(step, Mapping)
    }
    target_predecessors = _mapping(
        recovery.get("target_template_predecessor_task_ids"),
        context="recovery target-template predecessors",
    )
    if set(target_predecessors) != new_subjects:
        raise FormalTrainingProvenanceError(
            "training recovery target-template predecessor inventory does not "
            "match the exact five targets"
        )
    for subject in SOURCE_REVISION_SUBJECTS["new"]:
        target_predecessor = _clearml_id(
            target_predecessors[subject],
            f"recovery target-template predecessor {subject}",
        )
        step = step_by_subject.get(subject)
        if (
            target_predecessor != SOURCE_REVISION_TARGET_PREDECESSOR_TASK_ID
            or not isinstance(step, Mapping)
            or step.get("predecessor_task_id") != target_predecessor
            or (subject in predecessors and predecessors[subject] != target_predecessor)
        ):
            raise FormalTrainingProvenanceError(
                f"training recovery target-template predecessor for {subject} "
                "is cross-spliced"
            )
        rerun_predecessors = recovery.get("rerun_predecessor_task_ids")
        if (
            isinstance(rerun_predecessors, Mapping)
            and subject in rerun_predecessors
            and rerun_predecessors[subject] != target_predecessor
        ):
            raise FormalTrainingProvenanceError(
                f"training recovery target-template predecessor for {subject} "
                "does not match its rerun"
            )
        for observation_key in (
            "recovery_target_adoptions",
            "recovered_pending_target_children",
        ):
            observations = recovery.get(observation_key)
            if not isinstance(observations, Mapping):
                continue
            observation = observations.get(subject)
            if (
                isinstance(observation, Mapping)
                and observation.get("predecessor_task_id") != target_predecessor
            ):
                raise FormalTrainingProvenanceError(
                    f"training recovery target-template predecessor for {subject} "
                    "does not match its adoption"
                )
    source_parameters = _transition_source_parameters("old")
    target_parameters = _transition_source_parameters("new")
    receipt_keys = {
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
    }
    for subject in SUBJECT_ORDER:
        if subject not in adopted:
            if subject in old_subjects:
                raise FormalTrainingProvenanceError(
                    f"old-source subject {subject} was not adopted from the source"
                )
            continue
        expected_role = "source" if subject in old_subjects else "target"
        if roles.get(subject) != expected_role:
            raise FormalTrainingProvenanceError(
                f"recovery template role for {subject} violates the sealed split"
            )
        task_id = _clearml_id(adopted[subject], f"recovery adopted {subject} task")
        predecessor = _clearml_id(
            predecessors[subject], f"recovery adopted {subject} predecessor"
        )
        step = step_by_subject.get(subject)
        if (
            not isinstance(step, Mapping)
            or step.get("task_id") != task_id
            or step.get("predecessor_task_id") != predecessor
        ):
            raise FormalTrainingProvenanceError(
                f"recovery adoption for {subject} is cross-spliced with progress"
            )
        receipt = _mapping(
            receipts[subject], context=f"recovery binding receipt {subject}"
        )
        selected_identity = (
            source_identity if expected_role == "source" else target_identity
        )
        expected_parameters = (
            source_parameters if expected_role == "source" else target_parameters
        )
        receipt_parameters = receipt.get("source_parameters")
        task_script_sha256 = _sha256(
            receipt.get("task_script_sha256"),
            f"recovery binding receipt {subject} task script",
        )
        canonical_script_sha256 = _sha256(
            selected_identity.get("script_sha256"),
            f"recovery binding receipt {subject} canonical script",
        )
        script_identity_policy = receipt.get("script_identity_policy")
        legacy_receipt = receipt.get("legacy_script_compatibility_receipt")
        if task_script_sha256 == canonical_script_sha256:
            script_identity_valid = (
                script_identity_policy == "exact_canonical_template"
                and legacy_receipt is None
            )
        else:
            script_identity_valid = (
                script_identity_policy
                == "exact_allowlisted_legacy_nested_teacher_semantics_preserving"
                and subject in SOURCE_REVISION_LEGACY_TASK_ADOPTIONS
            )
            if script_identity_valid:
                _validate_legacy_script_compatibility_receipt(
                    legacy_receipt,
                    subject=subject,
                    task_id=task_id,
                    task_script_sha256=task_script_sha256,
                    canonical_script_sha256=canonical_script_sha256,
                    source_progress_seal_sha256=recovery.get(
                        "source_progress_seal_sha256"
                    ),
                )
        expected_config_status = (
            "byte_identical_across_revisions"
            if expected_role == "source"
            else "target_revision_required"
        )
        if (
            set(receipt) != receipt_keys
            or receipt.get("task_id") != task_id
            or receipt.get("experiment") != subject
            or receipt.get("template_role") != expected_role
            or receipt.get("template_task_id") != selected_identity["task_id"]
            or receipt.get("template_script_sha256")
            != selected_identity["script_sha256"]
            or not script_identity_valid
            or not isinstance(receipt_parameters, Mapping)
            or {key: str(value) for key, value in receipt_parameters.items()}
            != expected_parameters
            or receipt.get("config_path") != _transition_config_path(subject)
            or receipt.get("config_inventory_status") != expected_config_status
            or receipt.get("exact_execution_parameter_match") is not True
            or receipt.get("predecessor_task_id") != predecessor
        ):
            raise FormalTrainingProvenanceError(
                f"recovery binding receipt for {subject} is invalid"
            )
        for key in (
            "expected_parameters_sha256",
            "observed_parameter_projection_sha256",
        ):
            _sha256(receipt.get(key), f"recovery binding receipt {subject} {key}")
    rerun_subjects = recovery.get("rerun_experiments")
    if (
        not isinstance(rerun_subjects, list)
        or not set(rerun_subjects) <= new_subjects
        or len(rerun_subjects) != len(set(rerun_subjects))
    ):
        raise FormalTrainingProvenanceError(
            "source revision recovery attempts to rerun an old-source subject"
        )
    replaced = _mapping(
        recovery.get("transition_replaced_source_tasks"),
        context="training recovery transition_replaced_source_tasks",
    )
    if not set(replaced) <= new_subjects:
        raise FormalTrainingProvenanceError(
            "training recovery replaces a source task outside the five targets"
        )
    for key in ("recovery_target_adoptions", "recovered_pending_target_children"):
        values = _mapping(recovery.get(key), context=f"training recovery {key}")
        if not set(values) <= adopted_subjects:
            raise FormalTrainingProvenanceError(
                f"training recovery {key} is not closed over adopted tasks"
            )


def _validate_progress(
    progress: Mapping[str, object],
    *,
    controller_task_id: str,
) -> tuple[list[dict[str, object]], dict[str, object], str]:
    seal = _require_seal(progress, context="training progress")
    expected = {
        "schema_version": 1,
        "controller_type": "resilient_v2x_post_main_sequential_training",
        "controller_task_id": controller_task_id,
        "experiment_order": list(SUBJECT_ORDER),
        "training_seed": TRAINING_SEED,
        "training_overlay_protocol_seed": TRAINING_SEED,
    }
    for key, value in expected.items():
        if progress.get(key) != value:
            raise FormalTrainingProvenanceError(f"training progress {key} mismatch")
    revision = progress.get("revision")
    if type(revision) is not int or revision < 1:
        raise FormalTrainingProvenanceError("training progress revision is invalid")
    raw_recovery = progress.get("recovery")
    if not isinstance(raw_recovery, Mapping):
        raise FormalTrainingProvenanceError(
            "training progress lacks the sealed recovery contract"
        )
    recovery = _mapping(raw_recovery, context="training recovery")
    if (
        recovery.get("schema_version") != 4
        or recovery.get("mode") != "failed_controller_immutable_fork"
        or recovery.get("source_controller_status") != "failed"
    ):
        raise FormalTrainingProvenanceError("training recovery contract mismatch")
    raw_steps = _sequence(progress.get("steps"), context="training progress steps")
    if len(raw_steps) != len(SUBJECT_ORDER):
        raise FormalTrainingProvenanceError("training progress step count mismatch")
    steps: list[dict[str, object]] = []
    task_ids: set[str] = set()
    for index, (raw_step, subject) in enumerate(
        zip(raw_steps, SUBJECT_ORDER, strict=True), start=1
    ):
        step = _mapping(raw_step, context=f"training progress step {index}")
        if (
            step.get("index") != index
            or step.get("experiment") != subject
            or step.get("state") != "completed"
        ):
            raise FormalTrainingProvenanceError(
                f"training progress step {index} is not the completed {subject!r} step"
            )
        task_id = _clearml_id(step.get("task_id"), f"{subject} progress task")
        if task_id in task_ids:
            raise FormalTrainingProvenanceError("training progress reuses a task ID")
        task_ids.add(task_id)
        predecessor = _clearml_id(
            step.get("predecessor_task_id"), f"{subject} progress predecessor"
        )
        result = _mapping(step.get("result"), context=f"{subject} progress result")
        if result.get("task_id") != task_id:
            raise FormalTrainingProvenanceError(
                f"{subject} progress result task ID mismatch"
            )
        if (
            result.get("run_contract_artifact") != RUN_CONTRACT_ARTIFACT
            or result.get("training_seed") != TRAINING_SEED
            or result.get("training_overlay_protocol_seed") != TRAINING_SEED
        ):
            raise FormalTrainingProvenanceError(
                f"{subject} progress result training contract mismatch"
            )
        step["task_id"] = task_id
        step["predecessor_task_id"] = predecessor
        steps.append(step)
    _validate_source_revision_recovery(
        recovery,
        controller_task_id=controller_task_id,
        progress_template=progress.get("template"),
        steps=steps,
    )
    return steps, recovery, seal


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    steps: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], str]:
    seal = _require_seal(manifest, context="formal training manifest")
    expected = {
        "schema_version": 1,
        "manifest_type": "resilient_v2x_formal_1337_training_inputs",
        "protocol_id": "DAIR-CAUSAL-1337-v1",
        "sample_count": 1337,
        "delays_ms": [0, 100, 200, 300],
        "conditions": ["Full", "L-Fail", "C-Fail"],
        "run_count": 12,
        "checkpoint_policy": "epoch_50_final_only",
        "training_seed": TRAINING_SEED,
        "training_overlay_protocol_seed": TRAINING_SEED,
        "evaluation_release_semantics": (
            "formal_manifest_after_full_training_suite_completion"
        ),
        "subject_order": list(SUBJECT_ORDER),
        "subject_count": len(SUBJECT_ORDER),
    }
    expected_manifest_keys = set(expected) | {"entries", "seal_sha256"}
    if set(manifest) != expected_manifest_keys:
        raise FormalTrainingProvenanceError(
            "formal training manifest field inventory mismatch"
        )
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise FormalTrainingProvenanceError(
                f"formal training manifest {key} mismatch"
            )
    raw_entries = _sequence(
        manifest.get("entries"), context="formal training manifest entries"
    )
    if len(raw_entries) != len(SUBJECT_ORDER):
        raise FormalTrainingProvenanceError(
            "formal training manifest entry count mismatch"
        )
    entries: list[dict[str, object]] = []
    seen_task_ids: set[str] = set()
    expected_entry_keys = {
        "index",
        "subject",
        "kind",
        "training_task_id",
        "training_predecessor_task_id",
        "model_id",
        "model_name",
        "model_url",
        "checkpoint_filename",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "training_seed",
        "training_overlay_protocol_seed",
        "common_teacher_initialization_audit_artifact",
        "common_teacher_initialization_audit_sha256",
    }
    for index, (raw_entry, subject, step) in enumerate(
        zip(raw_entries, SUBJECT_ORDER, steps, strict=True), start=1
    ):
        entry = _mapping(raw_entry, context=f"formal training entry {subject}")
        if set(entry) != expected_entry_keys:
            raise FormalTrainingProvenanceError(
                f"formal training entry {subject} field inventory mismatch"
            )
        task_id = _clearml_id(entry.get("training_task_id"), f"{subject} manifest task")
        predecessor = _clearml_id(
            entry.get("training_predecessor_task_id"),
            f"{subject} manifest predecessor",
        )
        if (
            entry.get("index") != index
            or entry.get("subject") != subject
            or entry.get("kind") != SUBJECT_KIND[subject]
            or task_id != step["task_id"]
            or predecessor != step["predecessor_task_id"]
            or entry.get("training_seed") != TRAINING_SEED
            or entry.get("training_overlay_protocol_seed") != TRAINING_SEED
            or entry.get("common_teacher_initialization_audit_artifact")
            != COMMON_TEACHER_AUDIT_ARTIFACT
        ):
            raise FormalTrainingProvenanceError(
                f"formal training entry {subject} identity mismatch"
            )
        if task_id in seen_task_ids:
            raise FormalTrainingProvenanceError(
                "formal training manifest reuses a task ID"
            )
        seen_task_ids.add(task_id)
        model_id = _clearml_id(entry.get("model_id"), f"{subject} manifest model")
        model_name = f"ResilientV2X {subject} final checkpoint"
        if entry.get("model_name") != model_name:
            raise FormalTrainingProvenanceError(
                f"{subject} manifest final model name mismatch"
            )
        model_url = _fileserver_url(
            entry.get("model_url"),
            expected_filename=f"{subject}_epoch_50.pth",
            context=f"{subject} manifest final model",
        )
        if entry.get("checkpoint_filename") != "epoch_50.pth":
            raise FormalTrainingProvenanceError(
                f"{subject} manifest checkpoint filename mismatch"
            )
        checkpoint_sha256 = _sha256(
            entry.get("checkpoint_sha256"), f"{subject} checkpoint"
        )
        checkpoint_size = entry.get("checkpoint_size_bytes")
        if type(checkpoint_size) is not int or checkpoint_size <= 0:
            raise FormalTrainingProvenanceError(
                f"{subject} manifest checkpoint size is invalid"
            )
        entry["model_id"] = model_id
        entry["model_name"] = model_name
        entry["model_url"] = model_url
        entry["checkpoint_sha256"] = checkpoint_sha256
        _sha256(
            entry.get("common_teacher_initialization_audit_sha256"),
            f"{subject} initialization audit",
        )
        entries.append(entry)
    return entries, seal


def _validate_lineage_progress(
    progress: Mapping[str, object],
    *,
    controller_task_id: str,
) -> tuple[dict[str, dict[str, object]], dict[str, object] | None, str, str]:
    seal = _require_seal(progress, context=f"controller {controller_task_id} progress")
    expected = {
        "schema_version": 1,
        "controller_type": "resilient_v2x_post_main_sequential_training",
        "controller_task_id": controller_task_id,
        "experiment_order": list(SUBJECT_ORDER),
        "training_seed": TRAINING_SEED,
        "training_overlay_protocol_seed": TRAINING_SEED,
    }
    for key, value in expected.items():
        if progress.get(key) != value:
            raise FormalTrainingProvenanceError(
                f"controller {controller_task_id} progress {key} mismatch"
            )
    revision = progress.get("revision")
    if type(revision) is not int or revision < 1:
        raise FormalTrainingProvenanceError(
            f"controller {controller_task_id} progress revision is invalid"
        )
    template = _mapping(
        progress.get("template"),
        context=f"controller {controller_task_id} progress template",
    )
    template_script_identity = _sha256(
        template.get("script_sha256"),
        f"controller {controller_task_id} template script identity",
    )
    raw_steps = _sequence(
        progress.get("steps"),
        context=f"controller {controller_task_id} progress steps",
    )
    if len(raw_steps) != len(SUBJECT_ORDER):
        raise FormalTrainingProvenanceError(
            f"controller {controller_task_id} progress step count mismatch"
        )
    steps: dict[str, dict[str, object]] = {}
    for index, (raw_step, subject) in enumerate(
        zip(raw_steps, SUBJECT_ORDER, strict=True), start=1
    ):
        step = _mapping(
            raw_step,
            context=f"controller {controller_task_id} progress step {index}",
        )
        if step.get("index") != index or step.get("experiment") != subject:
            raise FormalTrainingProvenanceError(
                f"controller {controller_task_id} progress order mismatch"
            )
        task_id = step.get("task_id")
        if task_id is not None:
            _clearml_id(task_id, f"controller {controller_task_id} {subject} task")
        steps[subject] = step
    raw_recovery = progress.get("recovery")
    recovery: dict[str, object] | None
    if raw_recovery is None:
        recovery = None
    else:
        recovery = _mapping(
            raw_recovery,
            context=f"controller {controller_task_id} recovery",
        )
        if (
            recovery.get("schema_version") not in {2, 3, 4}
            or recovery.get("mode") != "failed_controller_immutable_fork"
            or recovery.get("source_controller_status") != "failed"
        ):
            raise FormalTrainingProvenanceError(
                f"controller {controller_task_id} recovery contract mismatch"
            )
    return steps, recovery, seal, template_script_identity


def _discover_progress_lineage(
    task_class: object,
    *,
    controller_task_id: str,
    controller_task: object,
    progress: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    states: dict[str, dict[str, object]] = {}
    current_id = controller_task_id
    current_task = controller_task
    current_progress = dict(progress)
    while True:
        if current_id in states:
            raise FormalTrainingProvenanceError(
                "recovery controller chain contains a cycle"
            )
        steps, recovery, seal, template_script_identity = _validate_lineage_progress(
            current_progress,
            controller_task_id=current_id,
        )
        states[current_id] = {
            "task": current_task,
            "progress": current_progress,
            "steps": steps,
            "recovery": recovery,
            "progress_seal_sha256": seal,
            "template_script_identity_sha256": template_script_identity,
        }
        if recovery is None:
            break
        source_id = _clearml_id(
            recovery.get("source_controller_task_id"),
            f"controller {current_id} recovery source",
        )
        if source_id in states or source_id == controller_task_id:
            raise FormalTrainingProvenanceError(
                "recovery controller chain contains a cycle"
            )
        source_task = task_class.get_task(task_id=source_id)
        source_progress = _fresh_artifact_mapping(
            source_task,
            PROGRESS_ARTIFACT,
            context=f"recovery source controller {source_id}",
        )
        source_seal = _require_seal(
            source_progress,
            context=f"recovery source controller {source_id} progress",
        )
        if (
            recovery.get("source_progress_revision") != source_progress.get("revision")
            or recovery.get("source_progress_seal_sha256") != source_seal
        ):
            raise FormalTrainingProvenanceError(
                f"controller {current_id} recovery does not bind source progress"
            )
        source_recovery = source_progress.get("recovery")
        source_chain = recovery.get("source_recovery_chain")
        if source_recovery is None:
            if source_chain is not None:
                raise FormalTrainingProvenanceError(
                    f"controller {current_id} invents a source recovery chain"
                )
        else:
            chain = _mapping(
                source_chain,
                context=f"controller {current_id} source recovery chain",
            )
            if (
                chain.get("source_controller_task_id") != source_id
                or chain.get("source_progress_revision")
                != source_progress.get("revision")
                or chain.get("source_progress_seal_sha256") != source_seal
                or chain.get("source_recovery_sha256")
                != _content_sha256(source_recovery)
            ):
                raise FormalTrainingProvenanceError(
                    f"controller {current_id} source recovery chain drifted"
                )
        current_id = source_id
        current_task = source_task
        current_progress = source_progress
    return states


def _trace_training_task_origin(
    *,
    subject: str,
    task_id: str,
    predecessor_task_id: str,
    controller_task_id: str,
    states: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    lineage: list[dict[str, object]] = []
    current_id = controller_task_id
    while True:
        state = states.get(current_id)
        if not isinstance(state, Mapping):
            raise FormalTrainingProvenanceError(
                f"{subject} recovery lineage lacks controller {current_id}"
            )
        steps = state.get("steps")
        if not isinstance(steps, Mapping):  # pragma: no cover - constructed helper
            raise FormalTrainingProvenanceError(f"{subject} lineage steps are invalid")
        step = steps.get(subject)
        if not isinstance(step, Mapping) or step.get("task_id") != task_id:
            raise FormalTrainingProvenanceError(
                f"{subject} task is not preserved in controller {current_id} progress"
            )
        if step.get("predecessor_task_id") != predecessor_task_id:
            raise FormalTrainingProvenanceError(
                f"{subject} predecessor drifted in controller {current_id} progress"
            )
        recovery = state.get("recovery")
        if recovery is None:
            identity = _sha256(
                state.get("template_script_identity_sha256"),
                f"{subject} origin template script identity",
            )
            lineage.append(
                {
                    "controller_task_id": current_id,
                    "decision": "controller_created",
                    "progress_seal_sha256": state["progress_seal_sha256"],
                }
            )
            return {
                "task_id": task_id,
                "expected_parent_task_id": current_id,
                "parent_binding": (
                    "current_controller_created"
                    if current_id == controller_task_id
                    else "recursive_source_controller_created"
                ),
                "sealed_progress_script_identity_sha256": identity,
                "recovery_lineage": lineage,
            }
        if not isinstance(recovery, Mapping):  # pragma: no cover - validated helper
            raise FormalTrainingProvenanceError(f"{subject} recovery is invalid")
        adopted = _mapping(
            recovery.get("adopted_task_ids"),
            context=f"controller {current_id} adopted tasks",
        )
        predecessors = _mapping(
            recovery.get("adopted_predecessor_task_ids"),
            context=f"controller {current_id} adopted predecessors",
        )
        if set(adopted) != set(predecessors) or not set(adopted) <= set(SUBJECT_ORDER):
            raise FormalTrainingProvenanceError(
                f"controller {current_id} adoption inventory is invalid"
            )
        if subject not in adopted:
            identity = _sha256(
                state.get("template_script_identity_sha256"),
                f"{subject} creator template script identity",
            )
            lineage.append(
                {
                    "controller_task_id": current_id,
                    "decision": "controller_created",
                    "progress_seal_sha256": state["progress_seal_sha256"],
                }
            )
            return {
                "task_id": task_id,
                "expected_parent_task_id": current_id,
                "parent_binding": (
                    "current_controller_created"
                    if current_id == controller_task_id
                    else "recursive_source_controller_created"
                ),
                "sealed_progress_script_identity_sha256": identity,
                "recovery_lineage": lineage,
            }
        if adopted[subject] != task_id or predecessors[subject] != predecessor_task_id:
            raise FormalTrainingProvenanceError(
                f"recovery adoption for {subject} does not match sealed progress"
            )
        observations: list[tuple[str, dict[str, object]]] = []
        for key in ("recovery_target_adoptions", "recovered_pending_target_children"):
            raw_values = recovery.get(key)
            if (
                raw_values is None
                and recovery.get("schema_version") == 2
                and key == "recovered_pending_target_children"
            ):
                # Revision-2 recovery contracts predate this observation map.
                # Its absence means no observations, while revision 3 must carry
                # the explicit map and remains fail-closed.
                values: dict[str, object] = {}
            else:
                values = _mapping(raw_values, context=f"controller {current_id} {key}")
            raw_observation = values.get(subject)
            if raw_observation is not None:
                observations.append(
                    (
                        key,
                        _mapping(
                            raw_observation,
                            context=f"controller {current_id} {key} {subject}",
                        ),
                    )
                )
        if len(observations) > 1:
            raise FormalTrainingProvenanceError(
                f"controller {current_id} has overlapping observations for {subject}"
            )
        if observations:
            key, observation = observations[0]
            if (
                observation.get("task_id") != task_id
                or observation.get("predecessor_task_id") != predecessor_task_id
            ):
                raise FormalTrainingProvenanceError(
                    f"controller {current_id} parent observation for {subject} drifted"
                )
            if key == "recovery_target_adoptions":
                if observation.get("parent_controller_status") != "failed":
                    raise FormalTrainingProvenanceError(
                        f"controller {current_id} parent observation for "
                        f"{subject} drifted"
                    )
            else:
                expected_pending_keys = {
                    "task_id",
                    "task_name",
                    "task_status",
                    "task_last_update",
                    "task_script_sha256",
                    "parent_controller_task_id",
                    "source_snapshot_state",
                    "source_snapshot_task_id",
                    "predecessor_task_id",
                    "recovery_intent_source_task_id",
                    "recovery_intent_terminal_status",
                    "provenance",
                }
                if (
                    recovery.get("schema_version") != 3
                    or set(observation) != expected_pending_keys
                    or observation.get("provenance")
                    != "canonical_child_of_failed_recovery_controller"
                    or observation.get("source_snapshot_state") != "pending"
                    or observation.get("source_snapshot_task_id") is not None
                    or observation.get("parent_controller_task_id")
                    != recovery.get("source_controller_task_id")
                    or recovery.get("source_controller_status") != "failed"
                    or observation.get("task_status")
                    not in WAITING_STATUSES | {"completed"}
                    or type(observation.get("task_name")) is not str
                    or not str(observation.get("task_name"))
                    or type(observation.get("task_last_update")) is not str
                    or not str(observation.get("task_last_update"))
                    or observation.get("recovery_intent_terminal_status")
                    not in {"failed", "stopped", "closed"}
                ):
                    raise FormalTrainingProvenanceError(
                        f"controller {current_id} parent observation for "
                        f"{subject} drifted"
                    )
                _clearml_id(
                    observation.get("recovery_intent_source_task_id"),
                    f"{subject} recovery-intent source task",
                )
            parent_id = _clearml_id(
                observation.get("parent_controller_task_id"),
                f"{subject} observed parent controller",
            )
            identity = _sha256(
                observation.get("task_script_sha256"),
                f"{subject} observed normalized task script identity",
            )
            lineage.append(
                {
                    "controller_task_id": current_id,
                    "decision": key,
                    "progress_seal_sha256": state["progress_seal_sha256"],
                    "parent_controller_task_id": parent_id,
                }
            )
            return {
                "task_id": task_id,
                "expected_parent_task_id": parent_id,
                "parent_binding": key,
                "sealed_progress_script_identity_sha256": identity,
                "recovery_lineage": lineage,
            }
        source_id = _clearml_id(
            recovery.get("source_controller_task_id"),
            f"controller {current_id} recovery source",
        )
        lineage.append(
            {
                "controller_task_id": current_id,
                "decision": "adopted_from_source_progress",
                "progress_seal_sha256": state["progress_seal_sha256"],
                "source_controller_task_id": source_id,
            }
        )
        current_id = source_id


def _recovery_parent_bindings(
    *,
    controller_task_id: str,
    steps: Sequence[Mapping[str, object]],
    states: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    bindings: dict[str, dict[str, object]] = {}
    parent_subjects: dict[str, list[str]] = {}
    for subject, step in zip(SUBJECT_ORDER, steps, strict=True):
        binding = _trace_training_task_origin(
            subject=subject,
            task_id=str(step["task_id"]),
            predecessor_task_id=str(step["predecessor_task_id"]),
            controller_task_id=controller_task_id,
            states=states,
        )
        bindings[subject] = binding
        parent_id = str(binding["expected_parent_task_id"])
        if parent_id != controller_task_id:
            parent_subjects.setdefault(parent_id, []).append(subject)
    return bindings, parent_subjects


def _nested_teacher_assignment(
    source: str,
    *,
    context: str,
) -> tuple[ast.Module, ast.Assign, frozenset[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise FormalTrainingProvenanceError(f"{context} is not valid Python") from error
    candidates: list[ast.Assign] = []
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "NESTED_TEACHER_EXPERIMENTS":
            candidates.append(node)
    if len(candidates) != 1:
        raise FormalTrainingProvenanceError(
            f"{context} must define NESTED_TEACHER_EXPERIMENTS exactly once"
        )
    assignment = candidates[0]
    value = assignment.value
    if (
        not isinstance(value, ast.Call)
        or not isinstance(value.func, ast.Name)
        or value.func.id != "frozenset"
        or len(value.args) != 1
        or value.keywords
        or not isinstance(value.args[0], ast.Set)
    ):
        raise FormalTrainingProvenanceError(
            f"{context} nested-teacher definition has an unexpected AST"
        )
    members: list[str] = []
    for element in value.args[0].elts:
        if not isinstance(element, ast.Constant) or type(element.value) is not str:
            raise FormalTrainingProvenanceError(
                f"{context} nested-teacher set contains a non-literal member"
            )
        members.append(element.value)
    if len(members) != len(set(members)):
        raise FormalTrainingProvenanceError(
            f"{context} nested-teacher set contains duplicate members"
        )
    usages = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "NESTED_TEACHER_EXPERIMENTS"
    ]
    stores = [node for node in usages if isinstance(node.ctx, ast.Store)]
    loads = [node for node in usages if isinstance(node.ctx, ast.Load)]
    if len(usages) != 3 or len(stores) != 1 or len(loads) != 2:
        raise FormalTrainingProvenanceError(
            f"{context} must have one Store and two Load usages of "
            "NESTED_TEACHER_EXPERIMENTS"
        )
    if stores[0] is not assignment.targets[0]:
        raise FormalTrainingProvenanceError(
            f"{context} nested-teacher Store is not the reviewed definition"
        )
    load_positions: set[str] = set()
    for load in loads:
        compare = parents.get(load)
        if (
            not isinstance(compare, ast.Compare)
            or len(compare.ops) != 1
            or not isinstance(compare.ops[0], ast.In)
            or compare.comparators != [load]
            or not isinstance(compare.left, ast.Attribute)
            or not isinstance(compare.left.value, ast.Name)
            or compare.left.value.id != "spec"
            or compare.left.attr != "name"
        ):
            raise FormalTrainingProvenanceError(
                f"{context} nested-teacher Load is outside reviewed spec.name membership"
            )
        container = parents.get(compare)
        if (
            isinstance(container, ast.keyword)
            and container.arg == "expect_nested_teacher"
        ):
            load_positions.add("expect_nested_teacher_keyword")
            continue
        if isinstance(container, ast.Dict):
            matches = [
                key
                for key, value in zip(container.keys, container.values, strict=True)
                if value is compare
            ]
            if (
                len(matches) == 1
                and isinstance(matches[0], ast.Constant)
                and matches[0].value == "expected_nested_teacher"
            ):
                load_positions.add("expected_nested_teacher_contract_field")
                continue
        raise FormalTrainingProvenanceError(
            f"{context} nested-teacher Load is outside a reviewed call site"
        )
    if load_positions != {
        "expect_nested_teacher_keyword",
        "expected_nested_teacher_contract_field",
    }:
        raise FormalTrainingProvenanceError(
            f"{context} nested-teacher Load call-site inventory mismatch"
        )
    return tree, assignment, frozenset(members)


def _source_without_assignment(
    source: str,
    assignment: ast.Assign,
    *,
    context: str,
) -> str:
    if (
        assignment.lineno is None
        or assignment.end_lineno is None
        or assignment.col_offset != 0
        or assignment.end_col_offset is None
    ):
        raise FormalTrainingProvenanceError(
            f"{context} nested-teacher assignment has unsafe source coordinates"
        )
    lines = source.splitlines(keepends=True)
    start_line = assignment.lineno - 1
    end_line = assignment.end_lineno - 1
    start = sum(len(line) for line in lines[:start_line]) + assignment.col_offset
    end = sum(len(line) for line in lines[:end_line]) + assignment.end_col_offset
    return source[:start] + "__NESTED_TEACHER_ASSIGNMENT__" + source[end:]


def _normalized_ast_dump(tree: ast.Module, assignment: ast.Assign) -> str:
    normalized = copy.deepcopy(tree)
    matches = [
        node
        for node in ast.walk(normalized)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "NESTED_TEACHER_EXPERIMENTS"
    ]
    if len(matches) != 1:  # pragma: no cover - checked before copying
        raise FormalTrainingProvenanceError("normalized bootstrap AST is ambiguous")
    matches[0].value = ast.Call(
        func=ast.Name(id="frozenset", ctx=ast.Load()),
        args=[ast.Set(elts=[ast.Constant(value="__NESTED_TEACHER_MEMBERS__")])],
        keywords=[],
    )
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def _verify_bootstrap_equivalence(
    source_by_sha: Mapping[str, str],
) -> dict[str, object]:
    if set(source_by_sha) != set(ALLOWED_BOOTSTRAP_SHA256):
        raise FormalTrainingProvenanceError(
            "the completed chain must expose exactly the reviewed legacy and expanded "
            "bootstrap sources"
        )
    parsed: dict[str, tuple[ast.Module, ast.Assign, frozenset[str]]] = {}
    for script_sha, source in source_by_sha.items():
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != script_sha:
            raise FormalTrainingProvenanceError(
                f"bootstrap source bytes do not match {script_sha}"
            )
        parsed[script_sha] = _nested_teacher_assignment(
            source, context=f"bootstrap {script_sha}"
        )
    legacy_tree, legacy_assignment, legacy_members = parsed[LEGACY_BOOTSTRAP_SHA256]
    expanded_tree, expanded_assignment, expanded_members = parsed[
        EXPANDED_BOOTSTRAP_SHA256
    ]
    if legacy_members != LEGACY_NESTED_TEACHER_EXPERIMENTS:
        raise FormalTrainingProvenanceError(
            "legacy bootstrap nested-teacher membership mismatch"
        )
    if expanded_members != EXPANDED_NESTED_TEACHER_EXPERIMENTS:
        raise FormalTrainingProvenanceError(
            "expanded bootstrap nested-teacher membership mismatch"
        )
    legacy_text = _source_without_assignment(
        source_by_sha[LEGACY_BOOTSTRAP_SHA256],
        legacy_assignment,
        context="legacy bootstrap",
    )
    expanded_text = _source_without_assignment(
        source_by_sha[EXPANDED_BOOTSTRAP_SHA256],
        expanded_assignment,
        context="expanded bootstrap",
    )
    if legacy_text != expanded_text:
        raise FormalTrainingProvenanceError(
            "bootstrap sources differ outside NESTED_TEACHER_EXPERIMENTS"
        )
    legacy_ast = _normalized_ast_dump(legacy_tree, legacy_assignment)
    expanded_ast = _normalized_ast_dump(expanded_tree, expanded_assignment)
    if legacy_ast != expanded_ast:
        raise FormalTrainingProvenanceError(
            "bootstrap ASTs differ outside NESTED_TEACHER_EXPERIMENTS membership"
        )
    runtime_guards = {
        "tf32_override_off": '"NVIDIA_TF32_OVERRIDE": "0"',
        "allowed_gpu_capabilities": (
            "_allowed_caps = frozenset({(12, 0), (8, 0), (7, 0)})"
        ),
        "homogeneous_gpu_guard": "GPU capabilities must be homogeneous",
        "four_gpu_guard": "len(capabilities) != 4",
    }
    for name, marker in runtime_guards.items():
        if marker not in legacy_text:
            raise FormalTrainingProvenanceError(
                f"reviewed bootstrap lacks {name} runtime guard"
            )
    return {
        "equivalence_contract": "nested-teacher-membership-only-v1",
        "verified_from_actual_script_bytes": True,
        "legacy_script_sha256": LEGACY_BOOTSTRAP_SHA256,
        "expanded_script_sha256": EXPANDED_BOOTSTRAP_SHA256,
        "legacy_nested_teacher_experiments": sorted(legacy_members),
        "expanded_nested_teacher_experiments": sorted(expanded_members),
        "common_text_projection_sha256": hashlib.sha256(
            legacy_text.encode("utf-8")
        ).hexdigest(),
        "common_ast_projection_sha256": hashlib.sha256(
            legacy_ast.encode("utf-8")
        ).hexdigest(),
        "only_text_difference": "NESTED_TEACHER_EXPERIMENTS assignment",
        "only_ast_difference": "NESTED_TEACHER_EXPERIMENTS frozenset members",
        "capacity_matched_hardware_contract": "capacity-matched-hardware-v1",
        "runtime_guard_evidence": "reviewed_completed_bootstrap_bytes",
        "tf32_override": "0",
        "homogeneous_gpu_count": 4,
        "allowed_compute_capabilities": [[7, 0], [8, 0], [12, 0]],
    }


def _validate_run_contract(
    contract: Mapping[str, object],
    *,
    subject: str,
    task_id: str,
    predecessor_task_id: str,
) -> str:
    source_revision = _source_revision_for_subject(subject)
    expected = {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": subject,
        "source_dataset_id": source_revision["dataset_id"],
        "training_dataset_id": TRAINING_DATASET_ID,
        "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
        "build_manifest_sha256": BUILD_MANIFEST_SHA256,
        "predecessor_task_id": predecessor_task_id,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "seed": TRAINING_SEED,
        "precision": PRECISION,
        "val_interval": VAL_INTERVAL,
    }
    for key, value in expected.items():
        if contract.get(key) != value or type(contract.get(key)) is not type(value):
            raise FormalTrainingProvenanceError(
                f"{subject} run_contract {key} mismatch"
            )
    for key in ("training_seed", "training_overlay_protocol_seed"):
        if key in contract and (
            type(contract[key]) is not int or contract[key] != TRAINING_SEED
        ):
            raise FormalTrainingProvenanceError(
                f"{subject} run_contract {key} mismatch"
            )
    if contract.get("amp") is not False:
        raise FormalTrainingProvenanceError(f"{subject} run_contract amp mismatch")
    source_archive = _mapping(
        contract.get("source_archive"), context=f"{subject} source archive"
    )
    expected_source_archive = {
        "name": source_revision["archive_name"],
        "size_bytes": source_revision["archive_size_bytes"],
        "sha256": source_revision["archive_sha256"],
    }
    if source_archive != expected_source_archive:
        raise FormalTrainingProvenanceError(
            f"{subject} run_contract source archive identity mismatch"
        )
    teacher = _mapping(contract.get("teacher"), context=f"{subject} teacher")
    if teacher.get("sha256") != TEACHER_CHECKPOINT_SHA256:
        raise FormalTrainingProvenanceError(
            f"{subject} run_contract teacher SHA-256 mismatch"
        )
    initialization = _mapping(
        contract.get("common_teacher_initialization"),
        context=f"{subject} common teacher initialization",
    )
    expected_initialization = {
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
        "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "audit_artifact_name": COMMON_TEACHER_AUDIT_ARTIFACT,
        "audit_filename": "common_teacher_initialization_audit.json",
        "expected_nested_teacher": (subject in EXPANDED_NESTED_TEACHER_EXPERIMENTS),
    }
    if initialization != expected_initialization:
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher initialization contract mismatch"
        )
    return _content_sha256(contract)


def _validate_final_checkpoint_contract(
    contract: Mapping[str, object],
    *,
    subject: str,
    entry: Mapping[str, object],
) -> str:
    expected = {
        "model_id": entry["model_id"],
        "name": entry["model_name"],
        "url": entry["model_url"],
        "filename": "epoch_50.pth",
        "size_bytes": entry["checkpoint_size_bytes"],
        "sha256": entry["checkpoint_sha256"],
    }
    for key, value in expected.items():
        if contract.get(key) != value or type(contract.get(key)) is not type(value):
            raise FormalTrainingProvenanceError(
                f"{subject} final checkpoint contract {key} mismatch"
            )
    return _content_sha256(contract)


def _validate_common_teacher_audit(
    audit: Mapping[str, object],
    *,
    subject: str,
    entry: Mapping[str, object],
) -> str:
    def exact_mapping(
        value: object,
        *,
        keys: set[str],
        context: str,
    ) -> dict[str, object]:
        result = _mapping(value, context=context)
        if set(result) != keys:
            raise FormalTrainingProvenanceError(f"{context} key inventory mismatch")
        return result

    if set(audit) != {
        "schema_version",
        "contract",
        "result",
        "checkpoint",
        "source",
        "shared_initialization",
        "method_specific_fusion",
        "target",
    }:
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher audit key inventory mismatch"
        )
    expected_literals = {
        "schema_version": 1,
        "contract": COMMON_TEACHER_INITIALIZATION_CONTRACT,
        "result": "pass",
    }
    for key, expected_value in expected_literals.items():
        if (
            type(audit.get(key)) is not type(expected_value)
            or audit.get(key) != expected_value
        ):
            raise FormalTrainingProvenanceError(
                f"{subject} common teacher audit {key} mismatch"
            )

    checkpoint = exact_mapping(
        audit.get("checkpoint"),
        keys={"path", "filename", "size_bytes", "sha256", "expected_sha256"},
        context=f"{subject} common teacher audit checkpoint",
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
        or type(checkpoint.get("size_bytes")) is not int
        or int(checkpoint["size_bytes"]) <= 0
        or _sha256(
            checkpoint.get("sha256"),
            f"{subject} common teacher audit checkpoint",
        )
        != TEACHER_CHECKPOINT_SHA256
        or _sha256(
            checkpoint.get("expected_sha256"),
            f"{subject} common teacher audit expected checkpoint",
        )
        != TEACHER_CHECKPOINT_SHA256
    ):
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher audit checkpoint mismatch"
        )

    source = exact_mapping(
        audit.get("source"),
        keys={
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
        context=f"{subject} common teacher audit source",
    )
    expected_source = {
        "keys": COMMON_TEACHER_SOURCE_KEYS,
        "numel": COMMON_TEACHER_SOURCE_NUMEL,
        "bytes": COMMON_TEACHER_SOURCE_BYTES,
        "expected_keys": COMMON_TEACHER_SOURCE_KEYS,
        "common_keys": COMMON_TEACHER_SHARED_KEYS,
        "expected_common_keys": COMMON_TEACHER_SHARED_KEYS,
        "fusion_keys": COMMON_TEACHER_FUSION_KEYS,
        "expected_fusion_keys": COMMON_TEACHER_FUSION_KEYS,
    }
    if any(
        type(source.get(key)) is not int or source.get(key) != expected_value
        for key, expected_value in expected_source.items()
    ):
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher audit source mismatch"
        )
    _sha256(source.get("state_sha256"), f"{subject} common teacher source state")

    shared = exact_mapping(
        audit.get("shared_initialization"),
        keys={
            "prefixes",
            "keys",
            "numel",
            "bytes",
            "expected_keys",
            "state_sha256",
            "shape_dtype_verified",
            "exact_tensor_equality_verified",
        },
        context=f"{subject} common teacher audit shared initialization",
    )
    expected_shared = {
        "prefixes": list(COMMON_TEACHER_INITIALIZATION_PREFIXES),
        "keys": COMMON_TEACHER_SHARED_KEYS,
        "numel": COMMON_TEACHER_SHARED_NUMEL,
        "bytes": COMMON_TEACHER_SHARED_BYTES,
        "expected_keys": COMMON_TEACHER_SHARED_KEYS,
        "shape_dtype_verified": True,
        "exact_tensor_equality_verified": True,
    }
    if any(
        type(shared.get(key)) is not type(expected_value)
        or shared.get(key) != expected_value
        for key, expected_value in expected_shared.items()
    ):
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher audit shared initialization mismatch"
        )
    _sha256(shared.get("state_sha256"), f"{subject} common teacher shared state")

    fusion = exact_mapping(
        audit.get("method_specific_fusion"),
        keys={"keys", "numel", "bytes", "sha256_before", "sha256_after", "unchanged"},
        context=f"{subject} common teacher audit method-specific fusion",
    )
    if any(
        type(fusion.get(key)) is not int or int(fusion[key]) < 0
        for key in ("keys", "numel", "bytes")
    ):
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher audit fusion statistics are invalid"
        )
    fusion_keys = int(fusion["keys"])
    fusion_numel = int(fusion["numel"])
    fusion_bytes = int(fusion["bytes"])
    zero_allowed = subject in ZERO_FUSION_ALLOWED_SUBJECTS
    if (
        (fusion_keys == 0 and not zero_allowed)
        or (fusion_keys == 0 and (fusion_numel != 0 or fusion_bytes != 0))
        or (fusion_keys > 0 and (fusion_numel <= 0 or fusion_bytes <= 0))
    ):
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher audit fusion statistics mismatch"
        )
    before = _sha256(fusion.get("sha256_before"), f"{subject} fusion state before")
    after = _sha256(fusion.get("sha256_after"), f"{subject} fusion state after")
    if fusion.get("unchanged") is not True or before != after:
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher audit method-specific fusion changed"
        )

    target = exact_mapping(
        audit.get("target"),
        keys={
            "model_type",
            "target_key_count",
            "target_common_key_count",
            "target_fusion_key_count",
            "nested_teacher_present",
            "nested_teacher_key_count",
            "nested_teacher_full_equality_verified",
        },
        context=f"{subject} common teacher audit target",
    )
    nested = subject in EXPANDED_NESTED_TEACHER_EXPERIMENTS
    expected_target = {
        "target_key_count": (
            COMMON_TEACHER_SHARED_KEYS
            + fusion_keys
            + (COMMON_TEACHER_SOURCE_KEYS if nested else 0)
        ),
        "target_common_key_count": COMMON_TEACHER_SHARED_KEYS,
        "target_fusion_key_count": fusion_keys,
        "nested_teacher_present": nested,
        "nested_teacher_key_count": COMMON_TEACHER_SOURCE_KEYS if nested else 0,
        "nested_teacher_full_equality_verified": nested,
    }
    if (
        type(target.get("model_type")) is not str
        or not target["model_type"]
        or any(
            type(target.get(key)) is not type(expected_value)
            or target.get(key) != expected_value
            for key, expected_value in expected_target.items()
        )
    ):
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher audit target mismatch"
        )

    observed = _content_sha256(audit)
    expected = _sha256(
        entry.get("common_teacher_initialization_audit_sha256"),
        f"{subject} manifest common teacher audit",
    )
    if observed != expected:
        raise FormalTrainingProvenanceError(
            f"{subject} common teacher audit content SHA-256 mismatch"
        )
    return observed


def _runtime_source() -> str:
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FormalTrainingProvenanceError(
            "cannot read the running formal provenance producer"
        ) from error
    if not source:
        raise FormalTrainingProvenanceError("formal provenance producer is empty")
    return source


def _validate_output_authority(
    authority: Mapping[str, object],
    *,
    output_task_id: str,
    controller_task_id: str,
    expected_source: str,
) -> None:
    if authority.get("id") != output_task_id:
        raise FormalTrainingProvenanceError("provenance output identity drifted")
    if _raw_status(authority, context="provenance output") not in WAITING_STATUSES:
        raise FormalTrainingProvenanceError("provenance output is not writable")
    if _raw_parent(authority, context="provenance output") != controller_task_id:
        raise FormalTrainingProvenanceError("provenance output parent drifted")
    entry_point, source = _raw_script(authority, context="provenance output")
    if entry_point != PRODUCER_ENTRY_POINT or source != expected_source:
        raise FormalTrainingProvenanceError("provenance output source bytes drifted")
    output = _mapping(authority.get("output"), context="provenance output destination")
    if output.get("destination") != FILES_SERVER_URI:
        raise FormalTrainingProvenanceError("provenance output destination drifted")
    names = _raw_artifact_names(authority, context="provenance output")
    if names not in ((), (PROVENANCE_ARTIFACT,)):
        raise FormalTrainingProvenanceError(
            "provenance output artifact inventory drifted"
        )
    tags = _unique_tag_set(authority.get("tags"), context="provenance output tags")
    if tags not in (frozenset(), frozenset(FORMAL_TAGS)):
        raise FormalTrainingProvenanceError("provenance output tags drifted")


def _output_without_commit_fields(authority: Mapping[str, object]) -> dict[str, object]:
    result = _mapping(authority, context="provenance output authority")
    result["tags"] = []
    result["artifacts"] = []
    for field in OUTPUT_VOLATILE_LIFECYCLE_FIELDS:
        result.pop(field, None)
    return result


def _assert_same_authority(
    observed: Mapping[str, Mapping[str, object]],
    expected: Mapping[str, Mapping[str, object]],
    *,
    dependency_ids: Sequence[str],
    output_task_id: str,
    initial_output: Mapping[str, object],
    require_output_artifact: bool,
    expected_output_tags: Sequence[str],
    context: str,
) -> None:
    for task_id in dependency_ids:
        if _canonical_json(observed[task_id]) != _canonical_json(expected[task_id]):
            raise FormalTrainingProvenanceError(
                f"{context}: dependency {task_id} changed"
            )
    output = observed[output_task_id]
    if _canonical_json(_output_without_commit_fields(output)) != _canonical_json(
        _output_without_commit_fields(initial_output)
    ):
        raise FormalTrainingProvenanceError(
            f"{context}: provenance output metadata changed"
        )
    names = _raw_artifact_names(output, context=f"{context} provenance output")
    expected_names = (PROVENANCE_ARTIFACT,) if require_output_artifact else ()
    if names != expected_names:
        raise FormalTrainingProvenanceError(
            f"{context}: provenance output artifact was not committed exactly"
        )
    if _unique_tag_set(
        output.get("tags"), context=f"{context} output tags"
    ) != _unique_tag_set(expected_output_tags, context=f"{context} expected tags"):
        raise FormalTrainingProvenanceError(
            f"{context}: provenance output tags drifted"
        )


def _await_authoritative_output_tags(
    task_class: object,
    all_task_ids: Sequence[str],
    *,
    initial_authority: Mapping[str, Mapping[str, object]],
    dependency_ids: Sequence[str],
    output_task_id: str,
    initial_output: Mapping[str, object],
    target_tags: Sequence[str],
    allowed_tag_states: Sequence[Sequence[str]],
    context: str,
) -> dict[str, dict[str, object]]:
    target = _unique_tag_set(target_tags, context=f"{context} target tags")
    allowed = {
        _unique_tag_set(tags, context=f"{context} allowed tags")
        for tags in allowed_tag_states
    }
    if target not in allowed:
        raise FormalTrainingProvenanceError(
            f"{context} target tags are not an allowed state"
        )
    for attempt in range(1, TAG_AUTHORITY_READBACK_ATTEMPTS + 1):
        attempt_context = f"{context} attempt {attempt}"
        observed = _batch_authority_snapshot(
            task_class,
            all_task_ids,
            context=attempt_context,
        )
        output_tags = _tag_list(
            observed[output_task_id].get("tags"),
            context=f"{attempt_context} output tags",
        )
        _assert_same_authority(
            observed,
            initial_authority,
            dependency_ids=dependency_ids,
            output_task_id=output_task_id,
            initial_output=initial_output,
            require_output_artifact=True,
            expected_output_tags=output_tags,
            context=attempt_context,
        )
        observed_state = _unique_tag_set(
            output_tags,
            context=f"{attempt_context} output tags",
        )
        if observed_state not in allowed:
            raise FormalTrainingProvenanceError(
                f"{attempt_context}: provenance output tags entered an unknown state"
            )
        if observed_state == target:
            return observed
    raise FormalTrainingProvenanceError(
        f"{context}: target tags were not authoritative after "
        f"{TAG_AUTHORITY_READBACK_ATTEMPTS} readbacks"
    )


def _publish_and_commit(
    output_task: object,
    payload: Mapping[str, object],
    *,
    output_task_id: str,
    initial_authority: Mapping[str, Mapping[str, object]],
    dependency_ids: Sequence[str],
    all_task_ids: Sequence[str],
    task_class: object,
    assert_dependency_contents: Callable[[], None],
) -> None:
    output_initial = initial_authority[output_task_id]
    original_tags = _tag_list(output_initial.get("tags"), context="initial output tags")
    original_tag_set = _unique_tag_set(original_tags, context="initial output tags")
    existing_names = _raw_artifact_names(
        output_initial, context="initial provenance output"
    )
    desired_tags = list(FORMAL_TAGS)
    desired_tag_set = _unique_tag_set(desired_tags, context="desired formal tags")
    if original_tag_set == desired_tag_set and existing_names != (PROVENANCE_ARTIFACT,):
        raise FormalTrainingProvenanceError(
            "formal provenance tags exist without the sealed artifact"
        )
    assert_dependency_contents()
    if existing_names == ():
        uploader = getattr(output_task, "upload_artifact", None)
        flusher = getattr(output_task, "flush", None)
        if not callable(uploader) or not callable(flusher):
            raise FormalTrainingProvenanceError(
                "provenance output cannot publish artifacts"
            )
        try:
            uploaded = uploader(
                PROVENANCE_ARTIFACT,
                artifact_object=dict(payload),
                wait_on_upload=True,
            )
        except Exception as error:
            raise FormalTrainingProvenanceError(
                "formal provenance artifact upload failed"
            ) from error
        if uploaded is not True:
            raise FormalTrainingProvenanceError(
                "formal provenance artifact upload was not acknowledged"
            )
        assert_dependency_contents()
        try:
            flushed = flusher(wait_for_uploads=True)
        except Exception as error:
            raise FormalTrainingProvenanceError(
                "formal provenance artifact flush failed"
            ) from error
        if flushed not in {None, True}:
            raise FormalTrainingProvenanceError(
                "formal provenance artifact flush was not acknowledged"
            )
    for _pass in range(2):
        observed = _fresh_artifact_mapping(
            output_task,
            PROVENANCE_ARTIFACT,
            context="provenance output",
        )
        if _canonical_json(observed) != _canonical_json(payload):
            raise FormalTrainingProvenanceError(
                "formal provenance artifact fresh readback mismatch"
            )
        assert_dependency_contents()
    committed = _batch_authority_snapshot(
        task_class,
        all_task_ids,
        context="committed authoritative batch",
    )
    _assert_same_authority(
        committed,
        initial_authority,
        dependency_ids=dependency_ids,
        output_task_id=output_task_id,
        initial_output=output_initial,
        require_output_artifact=True,
        expected_output_tags=original_tags,
        context="committed authoritative batch",
    )
    if original_tag_set != desired_tag_set:
        tagger = getattr(output_task, "set_tags", None)
        if not callable(tagger):
            raise FormalTrainingProvenanceError(
                "provenance output cannot commit formal tags"
            )
        callback_error: Exception | None = None
        try:
            tagger(desired_tags)
        except Exception as error:  # server readback remains authoritative
            callback_error = error
        terminal_error: Exception | None = None
        try:
            assert_dependency_contents()
            _await_authoritative_output_tags(
                task_class,
                all_task_ids,
                initial_authority=initial_authority,
                dependency_ids=dependency_ids,
                output_task_id=output_task_id,
                initial_output=output_initial,
                target_tags=desired_tags,
                allowed_tag_states=(original_tags, desired_tags),
                context="terminal authoritative batch",
            )
        except Exception as error:
            terminal_error = error
        if terminal_error is not None:
            compensation_callback_error: Exception | None = None
            compensation_error: Exception | None = None
            try:
                tagger(original_tags)
            except Exception as error:  # server readback remains authoritative
                compensation_callback_error = error
            try:
                _await_authoritative_output_tags(
                    task_class,
                    all_task_ids,
                    initial_authority=initial_authority,
                    dependency_ids=dependency_ids,
                    output_task_id=output_task_id,
                    initial_output=output_initial,
                    target_tags=original_tags,
                    allowed_tag_states=(original_tags, desired_tags),
                    context="tag compensation authoritative batch",
                )
            except Exception as error:
                compensation_error = error
            if compensation_error is not None:
                raise FormalTrainingProvenanceError(
                    "formal provenance tag compensation failed"
                ) from (compensation_error or compensation_callback_error)
            raise FormalTrainingProvenanceError(
                "formal provenance tags were not committed"
            ) from (terminal_error or callback_error)
        return
    assert_dependency_contents()
    terminal = _batch_authority_snapshot(
        task_class,
        all_task_ids,
        context="terminal authoritative batch",
    )
    _assert_same_authority(
        terminal,
        initial_authority,
        dependency_ids=dependency_ids,
        output_task_id=output_task_id,
        initial_output=output_initial,
        require_output_artifact=True,
        expected_output_tags=desired_tags,
        context="terminal authoritative batch",
    )


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    controller_task_id = _clearml_id(
        args.training_controller_task_id, "training controller"
    )
    for name, value in (
        ("poll_seconds", args.poll_seconds),
        ("timeout_hours", args.timeout_hours),
    ):
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise FormalTrainingProvenanceError(f"{name} must be finite and positive")
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise FormalTrainingProvenanceError("ClearML Task class cannot resolve tasks")
    controller = getter(task_id=controller_task_id)
    if output_task is None:
        current = getattr(task_class, "current_task", None)
        output_task = current() if callable(current) else None
    if output_task is None:
        raise FormalTrainingProvenanceError(
            "formal training provenance requires a current ClearML task"
        )
    output_task_id = _clearml_id(
        getattr(output_task, "id", None), "formal provenance output"
    )
    if output_task_id == controller_task_id or id(output_task) == id(controller):
        raise FormalTrainingProvenanceError(
            "formal provenance output aliases the training controller"
        )
    deadline = monotonic_clock() + float(args.timeout_hours) * 3600.0
    _wait_for_completed(
        controller,
        deadline=deadline,
        poll_seconds=float(args.poll_seconds),
        monotonic_clock=monotonic_clock,
        sleeper=sleeper,
        required_artifacts=(PROGRESS_ARTIFACT, TRAINING_MANIFEST_ARTIFACT),
    )

    # These first fresh reads discover the exact sealed task inventory. Every
    # acceptance read below is repeated and sandwiched by all-task raw batches.
    progress = _fresh_artifact_mapping(
        controller, PROGRESS_ARTIFACT, context="training controller"
    )
    steps, recovery, progress_seal = _validate_progress(
        progress, controller_task_id=controller_task_id
    )
    manifest = _fresh_artifact_mapping(
        controller, TRAINING_MANIFEST_ARTIFACT, context="training controller"
    )
    entries, manifest_seal = _validate_manifest(manifest, steps=steps)
    lineage_states = _discover_progress_lineage(
        task_class,
        controller_task_id=controller_task_id,
        controller_task=controller,
        progress=progress,
    )
    parent_bindings, parent_subjects = _recovery_parent_bindings(
        controller_task_id=controller_task_id,
        steps=steps,
        states=lineage_states,
    )

    training_tasks: dict[str, object] = {}
    seen_objects = {id(controller), id(output_task)}
    for subject, step in zip(SUBJECT_ORDER, steps, strict=True):
        task = getter(task_id=str(step["task_id"]))
        if id(task) in seen_objects:
            raise FormalTrainingProvenanceError(
                f"training task {subject} aliases another formal task object"
            )
        seen_objects.add(id(task))
        training_tasks[subject] = task
    parent_tasks: dict[str, object] = {}
    external_controller_ids = (set(lineage_states) - {controller_task_id}) | set(
        parent_subjects
    )
    for parent_id in sorted(external_controller_ids):
        state = lineage_states.get(parent_id)
        parent_task = (
            state.get("task")
            if isinstance(state, Mapping)
            else getter(task_id=parent_id)
        )
        if parent_task is None:  # pragma: no cover - guarded construction
            raise FormalTrainingProvenanceError(
                f"recovery parent {parent_id} cannot be resolved"
            )
        if id(parent_task) in seen_objects:
            raise FormalTrainingProvenanceError(
                "a recovery parent aliases another formal task object"
            )
        seen_objects.add(id(parent_task))
        parent_tasks[parent_id] = parent_task

    training_task_ids = [str(step["task_id"]) for step in steps]
    dependency_ids = [
        controller_task_id,
        *training_task_ids,
        *sorted(parent_tasks),
    ]
    if len(set(dependency_ids)) != len(dependency_ids):
        raise FormalTrainingProvenanceError(
            "formal provenance dependency IDs are not disjoint"
        )
    all_task_ids = [*dependency_ids, output_task_id]
    initial_authority = _batch_authority_snapshot(
        task_class,
        all_task_ids,
        context="initial authoritative batch",
    )
    controller_authority = initial_authority[controller_task_id]
    if _raw_status(controller_authority, context="training controller") != "completed":
        raise FormalTrainingProvenanceError(
            "authoritative training controller is not completed"
        )
    runtime_source = _runtime_source()
    _validate_output_authority(
        initial_authority[output_task_id],
        output_task_id=output_task_id,
        controller_task_id=controller_task_id,
        expected_source=runtime_source,
    )
    expected_output_parameters = {
        "Args/training_controller_task_id": controller_task_id,
        "Args/poll_seconds": str(float(args.poll_seconds)),
        "Args/timeout_hours": str(float(args.timeout_hours)),
    }
    if (
        _parameters(output_task, context="provenance output")
        != expected_output_parameters
    ):
        raise FormalTrainingProvenanceError("provenance output Args contract drifted")

    source_by_sha: dict[str, str] = {}
    script_subjects: dict[str, list[str]] = {
        LEGACY_BOOTSTRAP_SHA256: [],
        EXPANDED_BOOTSTRAP_SHA256: [],
    }
    static_bindings: dict[str, dict[str, object]] = {}
    docker_commands: set[str] = set()
    records: list[dict[str, object]] = []
    for index, (subject, step, entry) in enumerate(
        zip(SUBJECT_ORDER, steps, entries, strict=True), start=1
    ):
        task_id = str(step["task_id"])
        authority = initial_authority[task_id]
        if _raw_status(authority, context=f"training task {subject}") != "completed":
            raise FormalTrainingProvenanceError(
                f"training task {subject} is not completed"
            )
        binding = parent_bindings[subject]
        parent_id = _raw_parent(authority, context=f"training task {subject}")
        if parent_id != binding["expected_parent_task_id"]:
            raise FormalTrainingProvenanceError(
                f"training task {subject} parent is not sealed by recovery progress"
            )
        entry_point, source = _raw_script(authority, context=f"training task {subject}")
        if Path(entry_point).name != PRODUCER_ENTRY_POINT.replace(
            "formal_training_provenance", "5090_bootstrap"
        ):
            raise FormalTrainingProvenanceError(
                f"training task {subject} bootstrap entry point drifted"
            )
        script_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if script_sha not in ALLOWED_BOOTSTRAP_SHA256:
            raise FormalTrainingProvenanceError(
                f"training task {subject} uses an unreviewed bootstrap SHA-256"
            )
        if script_sha == LEGACY_BOOTSTRAP_SHA256 and (
            subject not in LEGACY_SCRIPT_ALLOWED_SUBJECTS
        ):
            raise FormalTrainingProvenanceError(
                f"legacy bootstrap is not allowed for {subject}"
            )
        normalized_script_identity = _normalized_task_script_identity_sha256(
            authority,
            context=f"training task {subject}",
        )
        expected_normalized_identity = NORMALIZED_SCRIPT_IDENTITY_BY_RAW_SHA256[
            script_sha
        ]
        if normalized_script_identity != expected_normalized_identity:
            raise FormalTrainingProvenanceError(
                f"training task {subject} normalized script identity mismatch"
            )
        sealed_progress_identity = _sha256(
            binding.get("sealed_progress_script_identity_sha256"),
            f"training task {subject} sealed progress script identity",
        )
        if sealed_progress_identity != normalized_script_identity:
            raise FormalTrainingProvenanceError(
                f"training task {subject} progress/raw script identity mismatch"
            )
        previous = source_by_sha.setdefault(script_sha, source)
        if previous != source:
            raise FormalTrainingProvenanceError(
                f"script SHA-256 {script_sha} maps to inconsistent source bytes"
            )
        script_subjects[script_sha].append(subject)
        artifact_names = set(
            _raw_artifact_names(authority, context=f"training task {subject}")
        )
        if (
            not {
                RUN_CONTRACT_ARTIFACT,
                FINAL_CHECKPOINT_ARTIFACT,
                COMMON_TEACHER_AUDIT_ARTIFACT,
            }
            <= artifact_names
        ):
            raise FormalTrainingProvenanceError(
                f"training task {subject} lacks required raw artifact metadata"
            )
        static_binding = _training_task_static_binding(
            training_tasks[subject],
            subject=subject,
            task_id=task_id,
            predecessor_task_id=str(step["predecessor_task_id"]),
            entry=entry,
        )
        static_bindings[subject] = static_binding
        docker_commands.add(str(static_binding["docker_command"]))
        execution = _mapping(
            authority.get("execution"), context=f"training task {subject} execution"
        )
        queue_id = execution.get("queue")
        if type(queue_id) is not str or not queue_id:
            raise FormalTrainingProvenanceError(
                f"training task {subject} execution queue is missing"
            )
        last_worker = authority.get("last_worker")
        if type(last_worker) is not str or not last_worker:
            raise FormalTrainingProvenanceError(
                f"training task {subject} last worker is missing"
            )
        records.append(
            {
                "index": index,
                "subject": subject,
                "training_task_id": task_id,
                "predecessor_task_id": str(step["predecessor_task_id"]),
                "parent_controller_task_id": parent_id,
                "parent_binding": binding["parent_binding"],
                "recovery_lineage": binding["recovery_lineage"],
                "raw_bootstrap_script_sha256": script_sha,
                "normalized_task_script_identity_sha256": (normalized_script_identity),
                "sealed_progress_script_identity_sha256": (sealed_progress_identity),
                "script_equivalence_class": (
                    "legacy_nested_teacher_membership"
                    if script_sha == LEGACY_BOOTSTRAP_SHA256
                    else "expanded_nested_teacher_membership"
                ),
                "manifest_model_id": entry["model_id"],
                "manifest_model_name": entry["model_name"],
                "manifest_model_url": entry["model_url"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "checkpoint_size_bytes": entry["checkpoint_size_bytes"],
                "source_revision_tree_sha256": static_binding[
                    "source_revision_tree_sha256"
                ],
                "source_dataset_id": static_binding["source_dataset_id"],
                "source_archive_name": static_binding["source_archive_name"],
                "source_archive_bytes": static_binding["source_archive_bytes"],
                "source_archive_sha256": static_binding["source_archive_sha256"],
                "observed_parameter_keys": static_binding["observed_parameter_keys"],
                "observed_parameters_sha256": static_binding[
                    "observed_parameters_sha256"
                ],
                "docker_command_sha256": static_binding["docker_command_sha256"],
                "execution_queue_id": queue_id,
                "last_worker": last_worker,
                "raw_authority_sha256": _content_sha256(authority),
            }
        )
    bootstrap_equivalence = _verify_bootstrap_equivalence(source_by_sha)
    bootstrap_equivalence["legacy_script_subjects"] = script_subjects[
        LEGACY_BOOTSTRAP_SHA256
    ]
    bootstrap_equivalence["expanded_script_subjects"] = script_subjects[
        EXPANDED_BOOTSTRAP_SHA256
    ]
    expected_legacy_subjects = [
        subject
        for subject in SUBJECT_ORDER
        if subject in LEGACY_SCRIPT_ALLOWED_SUBJECTS
    ]
    expected_expanded_subjects = [
        subject
        for subject in SUBJECT_ORDER
        if subject not in LEGACY_SCRIPT_ALLOWED_SUBJECTS
    ]
    if (
        script_subjects[LEGACY_BOOTSTRAP_SHA256] != expected_legacy_subjects
        or script_subjects[EXPANDED_BOOTSTRAP_SHA256] != expected_expanded_subjects
    ):
        raise FormalTrainingProvenanceError(
            "bootstrap subject-to-version inventory mismatch"
        )
    if len(docker_commands) != 1:
        raise FormalTrainingProvenanceError(
            "formal training tasks do not share one exact Docker command"
        )

    for parent_id in sorted(external_controller_ids):
        subjects = parent_subjects.get(parent_id, [])
        authority = initial_authority[parent_id]
        if _raw_status(authority, context=f"recovery parent {parent_id}") not in (
            TERMINAL_PARENT_STATUSES
        ):
            raise FormalTrainingProvenanceError(
                f"recovery parent {parent_id} is not terminal failed"
            )
        if parent_id == controller_task_id:
            raise FormalTrainingProvenanceError(
                "recovery parent aliases the current controller"
            )
        if parent_id not in lineage_states and not subjects:
            raise FormalTrainingProvenanceError(
                f"recovery parent {parent_id} has no sealed lineage role"
            )

    def capture_dependency_contents() -> dict[str, object]:
        current_progress = _fresh_artifact_mapping(
            controller, PROGRESS_ARTIFACT, context="training controller"
        )
        current_steps, current_recovery, current_progress_seal = _validate_progress(
            current_progress,
            controller_task_id=controller_task_id,
        )
        current_manifest = _fresh_artifact_mapping(
            controller,
            TRAINING_MANIFEST_ARTIFACT,
            context="training controller",
        )
        current_entries, current_manifest_seal = _validate_manifest(
            current_manifest, steps=current_steps
        )
        if (
            current_progress_seal != progress_seal
            or current_manifest_seal != manifest_seal
        ):
            raise FormalTrainingProvenanceError(
                "controller progress/manifest seals changed during provenance audit"
            )
        if _canonical_json(current_recovery) != _canonical_json(recovery):
            raise FormalTrainingProvenanceError(
                "controller recovery contract changed during provenance audit"
            )
        current_states = _discover_progress_lineage(
            task_class,
            controller_task_id=controller_task_id,
            controller_task=controller,
            progress=current_progress,
        )
        if set(current_states) != set(lineage_states):
            raise FormalTrainingProvenanceError(
                "recursive recovery controller inventory changed"
            )
        current_bindings, current_parent_subjects = _recovery_parent_bindings(
            controller_task_id=controller_task_id,
            steps=current_steps,
            states=current_states,
        )
        if _canonical_json(current_bindings) != _canonical_json(
            parent_bindings
        ) or _canonical_json(current_parent_subjects) != _canonical_json(
            parent_subjects
        ):
            raise FormalTrainingProvenanceError(
                "recursive recovery parent bindings changed"
            )
        progress_chain: dict[str, object] = {}
        for chain_controller_id, state in current_states.items():
            current_chain_progress = state.get("progress")
            if not isinstance(current_chain_progress, Mapping):
                raise FormalTrainingProvenanceError(
                    "recursive recovery progress is invalid"
                )
            expected_state = lineage_states[chain_controller_id]
            if state.get("progress_seal_sha256") != expected_state.get(
                "progress_seal_sha256"
            ):
                raise FormalTrainingProvenanceError(
                    f"controller {chain_controller_id} progress seal changed"
                )
            progress_chain[chain_controller_id] = dict(current_chain_progress)
        task_artifacts: dict[str, object] = {}
        current_static_bindings: dict[str, object] = {}
        current_docker_commands: set[str] = set()
        for subject, task_id, step, entry in zip(
            SUBJECT_ORDER,
            training_task_ids,
            current_steps,
            current_entries,
            strict=True,
        ):
            if task_id != step["task_id"] or task_id != entry["training_task_id"]:
                raise FormalTrainingProvenanceError(
                    f"{subject} task binding changed during artifact readback"
                )
            contract = _fresh_artifact_mapping(
                training_tasks[subject],
                RUN_CONTRACT_ARTIFACT,
                context=f"training task {subject}",
            )
            contract_sha = _validate_run_contract(
                contract,
                subject=subject,
                task_id=task_id,
                predecessor_task_id=str(step["predecessor_task_id"]),
            )
            final_contract = _fresh_artifact_mapping(
                training_tasks[subject],
                FINAL_CHECKPOINT_ARTIFACT,
                context=f"training task {subject}",
            )
            final_contract_sha = _validate_final_checkpoint_contract(
                final_contract,
                subject=subject,
                entry=entry,
            )
            teacher_audit = _fresh_artifact_mapping(
                training_tasks[subject],
                COMMON_TEACHER_AUDIT_ARTIFACT,
                context=f"training task {subject}",
            )
            teacher_audit_sha = _validate_common_teacher_audit(
                teacher_audit,
                subject=subject,
                entry=entry,
            )
            static_binding = _training_task_static_binding(
                training_tasks[subject],
                subject=subject,
                task_id=task_id,
                predecessor_task_id=str(step["predecessor_task_id"]),
                entry=entry,
            )
            if _canonical_json(static_binding) != _canonical_json(
                static_bindings[subject]
            ):
                raise FormalTrainingProvenanceError(
                    f"training task {subject} static binding changed"
                )
            current_static_bindings[subject] = static_binding
            current_docker_commands.add(str(static_binding["docker_command"]))
            task_artifacts[subject] = {
                "run_contract": contract,
                "run_contract_content_sha256": contract_sha,
                "final_checkpoint_contract": final_contract,
                "final_checkpoint_contract_content_sha256": final_contract_sha,
                "common_teacher_initialization_audit": teacher_audit,
                "common_teacher_initialization_audit_content_sha256": (
                    teacher_audit_sha
                ),
            }
        if (
            current_docker_commands != docker_commands
            or len(current_docker_commands) != 1
        ):
            raise FormalTrainingProvenanceError(
                "formal training Docker command changed during readback"
            )
        return {
            "controller_progress_chain": progress_chain,
            "manifest": current_manifest,
            "task_artifacts": task_artifacts,
            "task_static_bindings": current_static_bindings,
        }

    baseline_contents = capture_dependency_contents()
    second_contents = capture_dependency_contents()
    if _canonical_json(second_contents) != _canonical_json(baseline_contents):
        raise FormalTrainingProvenanceError(
            "formal training dependency artifacts changed across fresh readbacks"
        )
    stable_authority = _batch_authority_snapshot(
        task_class,
        all_task_ids,
        context="post-artifact authoritative batch",
    )
    _assert_same_authority(
        stable_authority,
        initial_authority,
        dependency_ids=dependency_ids,
        output_task_id=output_task_id,
        initial_output=initial_authority[output_task_id],
        require_output_artifact=(
            _raw_artifact_names(
                initial_authority[output_task_id], context="initial output"
            )
            == (PROVENANCE_ARTIFACT,)
        ),
        expected_output_tags=_tag_list(
            initial_authority[output_task_id].get("tags"),
            context="initial output tags",
        ),
        context="post-artifact authoritative batch",
    )

    task_artifacts = _mapping(
        baseline_contents["task_artifacts"], context="stable task artifacts"
    )
    for record in records:
        subject = str(record["subject"])
        artifact_record = _mapping(
            task_artifacts[subject], context=f"stable task artifacts {subject}"
        )
        record["run_contract_content_sha256"] = artifact_record[
            "run_contract_content_sha256"
        ]
        record["final_checkpoint_contract_content_sha256"] = artifact_record[
            "final_checkpoint_contract_content_sha256"
        ]
        record["common_teacher_initialization_audit_content_sha256"] = artifact_record[
            "common_teacher_initialization_audit_content_sha256"
        ]
    progress_chain_records = []
    for chain_controller_id, state in lineage_states.items():
        chain_progress = state.get("progress")
        if not isinstance(chain_progress, Mapping):  # pragma: no cover - helper
            raise FormalTrainingProvenanceError("lineage progress is invalid")
        progress_chain_records.append(
            {
                "task_id": chain_controller_id,
                "role": (
                    "final_controller"
                    if chain_controller_id == controller_task_id
                    else "recursive_recovery_source_controller"
                ),
                "progress_artifact": PROGRESS_ARTIFACT,
                "progress_revision": chain_progress["revision"],
                "progress_seal_sha256": state["progress_seal_sha256"],
                "progress_content_sha256": _content_sha256(chain_progress),
                "template_script_identity_sha256": state[
                    "template_script_identity_sha256"
                ],
            }
        )
    recovery_parent_records = []
    for parent_id in sorted(external_controller_ids):
        state = lineage_states.get(parent_id)
        has_progress = isinstance(state, Mapping)
        parent_progress = state.get("progress") if has_progress else None
        recovery_parent_records.append(
            {
                "task_id": parent_id,
                "terminal_status": "failed",
                "roles": [
                    role
                    for role, present in (
                        ("recursive_recovery_source", has_progress),
                        ("actual_training_parent", parent_id in parent_subjects),
                    )
                    if present
                ],
                "actual_parent_subjects": parent_subjects.get(parent_id, []),
                "progress_artifact": PROGRESS_ARTIFACT if has_progress else None,
                "progress_seal_sha256": (
                    state.get("progress_seal_sha256") if has_progress else None
                ),
                "progress_content_sha256": (
                    _content_sha256(parent_progress)
                    if isinstance(parent_progress, Mapping)
                    else None
                ),
                "raw_authority_sha256": _content_sha256(initial_authority[parent_id]),
            }
        )
    source_revision_equivalence = _source_revision_equivalence()
    source_revision_subject_map = _source_revision_subject_map()
    payload = _sealed(
        {
            "schema_version": 2,
            "document_type": ("resilient_v2x_formal_training_provenance_equivalence"),
            "protocol_id": "DAIR-CAUSAL-1337-v1",
            "passed": True,
            "controller_task_id": controller_task_id,
            "controller_status": "completed",
            "controller_raw_authority_sha256": _content_sha256(
                initial_authority[controller_task_id]
            ),
            "subject_order": list(SUBJECT_ORDER),
            "subject_count": len(SUBJECT_ORDER),
            "all_training_tasks_completed": True,
            "authoritative_metadata_read": "single_batch_per_snapshot",
            "progress_artifact": PROGRESS_ARTIFACT,
            "progress_seal_sha256": progress_seal,
            "progress_content_sha256": _content_sha256(progress),
            "formal_training_manifest_artifact": TRAINING_MANIFEST_ARTIFACT,
            "formal_training_manifest_seal_sha256": manifest_seal,
            "formal_training_manifest_content_sha256": _content_sha256(manifest),
            "recovery_contract_sha256": _content_sha256(recovery),
            "recursive_progress_chain": progress_chain_records,
            "bootstrap_equivalence": bootstrap_equivalence,
            "source_revision_equivalence": source_revision_equivalence,
            "source_revision_equivalence_seal_sha256": (
                source_revision_equivalence["seal_sha256"]
            ),
            "source_revision_subject_map": source_revision_subject_map,
            "source_revision_subject_map_seal_sha256": (
                source_revision_subject_map["seal_sha256"]
            ),
            "run_contract_equivalence": {
                "source_fields_vary_only_by_sealed_subject_revision_map": True,
                "source_revision_equivalence_seal_sha256": (
                    source_revision_equivalence["seal_sha256"]
                ),
                "source_revision_subject_map_seal_sha256": (
                    source_revision_subject_map["seal_sha256"]
                ),
                "source_revision_counts": {
                    SOURCE_TREE_SHA256: 21,
                    NEW_SOURCE_TREE_SHA256: 5,
                },
                "training_dataset_id": TRAINING_DATASET_ID,
                "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
                "build_manifest_sha256": BUILD_MANIFEST_SHA256,
                "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
                "training_seed": TRAINING_SEED,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "precision": PRECISION,
                "max_epochs": MAX_EPOCHS,
                "val_interval": VAL_INTERVAL,
                "amp": False,
            },
            "capacity_matched_hardware": {
                "contract": "capacity-matched-hardware-v1",
                "docker_command_sha256": DOCKER_COMMAND_SHA256,
                "docker_command_identical_across_26_tasks": True,
                "gpu_count": 4,
                "homogeneous_per_task": True,
                "allowed_compute_capabilities": [[7, 0], [8, 0], [12, 0]],
                "gpu_class_may_vary_across_tasks": True,
                "tf32_override": "0",
                "precision": PRECISION,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "runtime_guard_evidence": "reviewed_completed_bootstrap_bytes",
            },
            "recovery_parent_controllers": recovery_parent_records,
            "training_tasks": records,
        }
    )

    def assert_dependency_contents() -> None:
        current = capture_dependency_contents()
        if _canonical_json(current) != _canonical_json(baseline_contents):
            raise FormalTrainingProvenanceError(
                "formal training dependency artifacts changed during publication"
            )

    _publish_and_commit(
        output_task,
        payload,
        output_task_id=output_task_id,
        initial_authority=initial_authority,
        dependency_ids=dependency_ids,
        all_task_ids=all_task_ids,
        task_class=task_class,
        assert_dependency_contents=assert_dependency_contents,
    )
    return payload


def main() -> int:
    args = _parser().parse_args()
    task = Task.init(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X formal training provenance equivalence",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
    )
    run(args, output_task=task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "PROVENANCE_ARTIFACT",
    "SUBJECT_ORDER",
    "TRAINING_CONTROLLER_TASK_ID",
    "_verify_bootstrap_equivalence",
    "run",
)
