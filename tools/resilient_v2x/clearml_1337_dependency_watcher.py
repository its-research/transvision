#!/usr/bin/env python3
"""Release sealed 1337 evaluation tasks only after training dependencies finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

try:
    from allegroai import Task
except ImportError:
    from clearml import Task


CLEARML_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ARTIFACT_HASH_PATTERN = re.compile(r"[0-9a-fA-F]{32,128}")
MAX_JSON_ARTIFACT_BYTES = 64 * 1024 * 1024
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
EXPECTED_SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
EXPECTED_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
EXPECTED_SOURCE_ARCHIVE_BYTES = 1_222_481
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6"
)
EXPECTED_NEW_SOURCE_DATASET_ID = "351feedbbe81481fa31f1e9ae11a3f4e"
EXPECTED_NEW_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-ad511d88b731.tar.zst"
EXPECTED_NEW_SOURCE_ARCHIVE_BYTES = 1_222_492
EXPECTED_NEW_SOURCE_ARCHIVE_SHA256 = (
    "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
)
EXPECTED_NEW_SOURCE_TREE_SHA256 = (
    "ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"
)
EXPECTED_SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = (
    "29de9700cac66f9998be643e85a8ec646c04ec17fddbb6207bc1438e9e73941b"
)
EXPECTED_SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = (
    "c3170f4a88b080f9cc7267053f354640dd260687cb2c35a6f7ffed73d69f4154"
)
EXPECTED_TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
EXPECTED_PREDECESSOR_TASK_ID = "f041d43e48c14ba4a4562281860d13f6"
EXPECTED_PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
EXPECTED_SAMPLE_COUNT = 1337
EXPECTED_DELAYS_MS = (0, 100, 200, 300)
EXPECTED_CONDITIONS = ("Full", "L-Fail", "C-Fail")
EXPECTED_CHECKPOINT_POLICY = "epoch_50_final_only"
COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT = "common_teacher_initialization_audit"
FORMAL_TRAINING_MANIFEST_ARTIFACT = "formal_1337_training_manifest"
FORMAL_TRAINING_PROGRESS_ARTIFACT = "post_main_training_progress"
FORMAL_TRAINING_PROVENANCE_ARTIFACT = "formal_1337_training_provenance_equivalence"
FORMAL_EVALUATION_PLAN_ARTIFACT = "formal_1337_evaluation_plan"
EXPECTED_RELEASE_SEMANTICS = "formal_manifest_after_full_training_suite_completion"
EXPECTED_CANONICAL_TRAINING_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
EXPECTED_LEGACY_TRAINING_SCRIPT_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
LEGACY_TRAINING_SCRIPT_ALLOWED_SUBJECTS = frozenset(
    {"support_residual", "no_distillation"}
)
EXPECTED_NATIVE_BUNDLE_SHA256 = (
    "19b8e7f5edc8216d4b43cb17854dccafe4fc9fe46995a88803e6342eeaa22b21"
)
EXPECTED_BUILD_MANIFEST_SHA256 = (
    "21c6ab7a6e9a2823ba111289a42e7f882c5f4ba02c73175106ff251fe5864a43"
)
EXPECTED_TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
EXPECTED_TRAINING_SEED = 20_250_218
EXPECTED_DOCKER_COMMAND_SHA256 = (
    "f68e2426a223b700bc58e7bd60ca040773d355323ae32cd70fd61ee2e322569b"
)
EXPECTED_PROVENANCE_TAGS = (
    "ResilientV2X-suite",
    "formal-training-provenance-equivalence",
    EXPECTED_PROTOCOL_ID,
    "cpu-controller",
)
CANDIDATE_RELEASE_BLOCKER_PARENT_TASK_ID = "21368e8260cc4e5392fe2dbdf116e36f"
CANDIDATE_RELEASE_BLOCKERS = (
    {
        "label": "E1",
        "training_task_id": "f0c3082f3aa34a81805903e0ffdc8610",
        "task_name": (
            "ResilientV2X fastlane E1 support_residual_linear [d6df8cc0ce59]"
        ),
        "subject": "support_residual_linear",
    },
    {
        "label": "E2",
        "training_task_id": "969c8fce6d24446299561772b3955274",
        "task_name": (
            "ResilientV2X fastlane E2 no_reliability_linear [d6df8cc0ce59]"
        ),
        "subject": "no_reliability_linear",
    },
    {
        "label": "E3",
        "training_task_id": "dc037315c0684c3d854a2fd7c19a2a2f",
        "task_name": (
            "ResilientV2X fastlane E3 support_residual_no_reliability "
            "[d6df8cc0ce59]"
        ),
        "subject": "support_residual_no_reliability",
    },
    {
        "label": "P0",
        "training_task_id": "8883c51ced4f4951a45edbaefe6342d4",
        "task_name": "ResilientV2X round2 P0 [ebd8421be6fc]",
        "subject": "support_residual_no_reliability_linear",
    },
    {
        "label": "P2",
        "training_task_id": "f5d3820b4cdf416183c8f1fee566abe3",
        "task_name": "ResilientV2X round2 P2 bbox2.5 [4ced839db7fb]",
        "subject": "support_residual_no_reliability_linear_bbox25",
    },
)
CANDIDATE_RELEASE_BLOCKERS_JSON = json.dumps(
    CANDIDATE_RELEASE_BLOCKERS,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
)
AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID = (
    "dbc05b28fbd044bf89edc3872742843a"
)
AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_NAME = (
    "ResilientV2X formal 1337 dependency watcher successor"
)
AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_SCRIPT_SHA256 = (
    "80c57bc341b39caa0aa7d41f4e64b947eaa8831ab592784daaf11aebf4b179ff"
)
AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256 = (
    "c5d14aa8021d06609c7a7e9a5401f4b5a0a417601fa8c95163f083eeb397e0f3"
)
AUTHORITATIVE_EVALUATION_PLAN_ARTIFACT_SHA256 = (
    "66ee8180ee13a34ff1519b204ddb71437e729f168466a38dbefa26c7eb8a4f57"
)
AUTHORITATIVE_EVALUATION_PLAN_ARTIFACT_BYTES = 16_552
AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_STATUS = "failed"
AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TAGS = frozenset(
    {
        EXPECTED_PROTOCOL_ID,
        "ResilientV2X-suite",
        "cpu-controller",
        "dependency-watcher",
    }
)
AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_PARAMETERS = {
    "Args/dependencies_json": "",
    "Args/evaluation_plan_json": "",
    "Args/evaluation_template_task_id": "8b77a3674dfe405388aae39ef82d06ef",
    "Args/evaluation_worker_queues": "GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090",
    "Args/expected_eval_script_sha256": (
        "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
    ),
    "Args/expected_predecessor_task_id": EXPECTED_PREDECESSOR_TASK_ID,
    "Args/expected_source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
    "Args/expected_source_dataset_id": EXPECTED_SOURCE_DATASET_ID,
    "Args/expected_training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
    "Args/expected_training_provenance_script_sha256": (
        "0bcc0b9ce383e5f2cfa35e87959caff9e388a988423e74797c7127a560eb007a"
    ),
    "Args/expected_training_script_sha256": "",
    "Args/expected_training_source_archive_sha256": "",
    "Args/expected_training_source_dataset_id": "",
    "Args/metadata_only_artifact_gate": "False",
    "Args/poll_seconds": "60.0",
    "Args/timeout_hours": "720.0",
    "Args/training_controller_task_id": "1011e98e10f64c428880af1d4b1d542b",
    "Args/training_manifest_json": "",
    "Args/training_provenance_task_id": "7e244a711751469b8cfdb25d77b05269",
}
AMENDED_EVALUATION_PLAN_PRODUCER_NAME = (
    "ResilientV2X formal 1337 FFNet queue amendment producer"
)
AMENDED_EVALUATION_PLAN_PRODUCER_ENTRY_POINT = (
    "amend_clearml_formal_evaluation_plan.py"
)
AMENDED_EVALUATION_PLAN_ARTIFACT = "formal_1337_evaluation_plan_amendment"
AMENDED_EVALUATION_PLAN_PRODUCER_TAGS = frozenset(
    {
        EXPECTED_PROTOCOL_ID,
        "ResilientV2X-suite",
        "cpu-controller",
        "formal-evaluation-plan-amendment",
    }
)
AMENDED_PLAN_RECOVERY_RECEIPT_SEAL_SHA256 = (
    "d95e2bc2bb93ea6c8c69ea683d31ba09ec2bf219c7a0c09591c0ff12de60fcdf"
)
AMENDED_PLAN_DEPLOYMENT_RECEIPT_SEAL_SHA256 = (
    "f54d6a51c1fba1f5c776e236ed79087922da61b71d4b4997f76c0bb1c6a5bb6d"
)
AMENDED_PLAN_ATTEMPT_EVIDENCE_SEAL_SHA256 = (
    "3342f4e1bf2cba08e5b3e71afe2719752ae362c288964c2fee3589ee57a6b5bf"
)
AMENDED_PLAN_FFNET_TASK_ID = "144397bfa9c242bc9a92a1279922558b"
AMENDED_PLAN_FFNET_OLD_QUEUE = "GPU4-V100"
AMENDED_PLAN_FFNET_NEW_QUEUE = "GPU4-A100"
AMENDED_PLAN_FFNET_ORIGINAL_WORKER = "10.100.34.26-V100:gpu0,1,2,3"
AMENDED_PLAN_A100_TARGET_WORKER = "10.100.34.18-A100:gpu4,5,6,7"
AMENDED_PLAN_GPU_MEMORY_IDLE_LIMIT_MIB = 1024.0
AMENDED_PLAN_GPU_USAGE_IDLE_LIMIT_PERCENT = 1.0
AMENDED_PLAN_FIXED_BASELINES = (
    "ffnet",
    "coformernet",
    "v2x_vit",
    "cobevt",
    "bevfusion",
)
AMENDED_PLAN_PRIOR_SUCCESSOR_TASK_IDS = {
    "P": "7e244a711751469b8cfdb25d77b05269",
    "W": "40094c850391452da03df4f6b2499b90",
    "L": "70f1ceb80c1a4fceb4cf88343e360231",
    "A": "d985d344576b4718b59b4c207e2442c2",
    "S": "edb75f8a5c81413186eb866f9bd9855b",
}
EXPECTED_EVALUATION_TEMPLATE_INPUT_MODELS = (
    {
        "id": "d962f6bae8474260b54e170a7a5f0418",
        "name": "ResilientV2X clean teacher",
        "task": "487dab2664a8485fa0cc7c4e2a0c3df8",
    },
)
EVALUATION_BOOTSTRAP_RUNTIME_DEFAULT_PARAMETERS = {
    "Args/teacher_checkpoint": "",
    "Args/student_checkpoint": "",
    "Args/experiment_from_task": "",
    "Args/teacher_task_id": "",
    "Args/teacher_model_id": "",
    "Args/teacher_checkpoint_sha256": "",
    "Args/allow_failed_teacher_task": "False",
    "Args/student_task_id": "",
    "Args/student_model_id": "",
    "Args/student_checkpoint_sha256": "",
}
FORMAL_SUBJECT_ORDER = (
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
FORMAL_EVALUATION_RELEASE_PRIORITY = (
    "ffnet",
    "coformernet",
    "v2x_vit",
    "cobevt",
    "bevfusion",
    "resilient_v2x",
)
MAX_ACTIVE_CANDIDATE_EVALUATIONS = 3
CANDIDATE_EVALUATION_PHASE = (
    {
        "label": "E1",
        "subject": "support_residual_linear",
        "training_task_id": "f0c3082f3aa34a81805903e0ffdc8610",
        "evaluation_task_id": "c6f26cc7902142c090ec238856409ac6",
        "queue": "GPU4-5090",
    },
    {
        "label": "E2",
        "subject": "no_reliability_linear",
        "training_task_id": "969c8fce6d24446299561772b3955274",
        "evaluation_task_id": "",
        "queue": "GPU4-V100",
    },
    {
        "label": "E3",
        "subject": "support_residual_no_reliability",
        "training_task_id": "dc037315c0684c3d854a2fd7c19a2a2f",
        "evaluation_task_id": "27a82d39e6354697ad1532b6399bbde2",
        "queue": "GPU4-A100",
    },
    {
        "label": "P0",
        "subject": "support_residual_no_reliability_linear",
        "training_task_id": "8883c51ced4f4951a45edbaefe6342d4",
        "evaluation_task_id": "",
        "queue": "GPU4-A100",
    },
    {
        "label": "P2",
        "subject": "support_residual_no_reliability_linear_bbox25",
        "training_task_id": "f5d3820b4cdf416183c8f1fee566abe3",
        "evaluation_task_id": "",
        "queue": "GPU4-A100",
    },
)
NEW_SOURCE_SUBJECTS = frozenset(
    {"where2comm", "late_fusion", "disconet", "how2comm", "resilient_v2x"}
)
SOURCE_REVISION_BY_SUBJECT = {
    subject: ("new" if subject in NEW_SOURCE_SUBJECTS else "old")
    for subject in FORMAL_SUBJECT_ORDER
}
EXPECTED_SOURCE_BY_REVISION = {
    "old": {
        "dataset_id": EXPECTED_SOURCE_DATASET_ID,
        "archive_name": EXPECTED_SOURCE_ARCHIVE_NAME,
        "archive_size_bytes": EXPECTED_SOURCE_ARCHIVE_BYTES,
        "archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
    },
    "new": {
        "dataset_id": EXPECTED_NEW_SOURCE_DATASET_ID,
        "archive_name": EXPECTED_NEW_SOURCE_ARCHIVE_NAME,
        "archive_size_bytes": EXPECTED_NEW_SOURCE_ARCHIVE_BYTES,
        "archive_sha256": EXPECTED_NEW_SOURCE_ARCHIVE_SHA256,
        "tree_sha256": EXPECTED_NEW_SOURCE_TREE_SHA256,
    },
}
EXPECTED_QUEUE_IDS = {
    "GPU4-A100": "9350f33af13a448da8339eb7bea52fdf",
    "GPU4-V100": "3925e906ce484620a941e6ccedc4bdbd",
    "GPU4-5090": "5a84454c072349069e7b61af38637c6d",
}
RESOURCE_BLOCKING_STATUSES = frozenset({"queued", "in_progress"})
WORKER_GPU_ID_PATTERN = re.compile(
    r"(?P<host>[^:]+):gpu(?P<gpu>[0-9]+(?:,[0-9]+)*)$"
)
SCHEDULER_GPU_MEMORY_IDLE_LIMIT_MIB = 1024.0
SCHEDULER_GPU_USAGE_IDLE_LIMIT_PERCENT = 1.0
SCHEDULER_GPU_TELEMETRY_WINDOW_SECONDS = 180
SCHEDULER_GPU_TELEMETRY_INTERVAL_SECONDS = 10
GPU_PREFLIGHT_ANCHOR = (
    "def _capture_gpu_runtime() -> dict[str, object]:\n"
    "    import torch\n"
)
GPU_PREFLIGHT_REPLACEMENT = '''def _capture_gpu_runtime() -> dict[str, object]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [
        [value.strip() for value in line.split(",")]
        for line in gpu_result.stdout.splitlines()
        if line.strip()
    ]
    if len(rows) not in {4, 8} or any(len(row) != 5 for row in rows):
        raise RuntimeError("GPU memory preflight inventory is invalid")
    gpus = []
    for row in rows:
        try:
            index = int(row[0])
            total_mib, used_mib, free_mib = map(int, row[2:])
        except ValueError as error:
            raise RuntimeError("GPU memory preflight values are invalid") from error
        if (
            index < 0
            or total_mib <= 0
            or used_mib < 0
            or free_mib < 0
            or used_mib > 1024
            or free_mib < total_mib - 1024
        ):
            raise RuntimeError("assigned GPUs are not idle")
        gpus.append(
            {
                "index": index,
                "uuid": row[1],
                "total_mib": total_mib,
                "used_mib": used_mib,
                "free_mib": free_mib,
            }
        )
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    if any(line.strip() for line in process_result.stdout.splitlines()):
        raise RuntimeError("assigned GPUs already have compute processes")
    print(
        json.dumps(
            {
                "event": "gpu_memory_preflight_pass",
                "policy": "no_compute_process_and_at_most_1024_MiB_used_per_gpu",
                "gpu_count": len(gpus),
                "gpus": gpus,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    import torch
'''
SHARED_RUNTIME_PARAMETER_KEYS = (
    "Args/source_dataset_id",
    "Args/source_archive_name",
    "Args/source_archive_bytes",
    "Args/source_archive_sha256",
    "Args/training_dataset_id",
    "Args/native_bundle_bytes",
    "Args/native_bundle_sha256",
    "Args/build_manifest_sha256",
    "Args/gpus",
    "Args/amp",
)
SOURCE_RUNTIME_PARAMETER_KEYS = SHARED_RUNTIME_PARAMETER_KEYS[:4]
INVARIANT_RUNTIME_PARAMETER_KEYS = SHARED_RUNTIME_PARAMETER_KEYS[4:]
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


class TrainingProvenancePolicies(dict[str, dict[str, object]]):
    """Per-subject policies plus the exact upstream seals they came from."""

    project_id: str
    provenance_task_id: str
    provenance_seal_sha256: str
    source_revision_equivalence: dict[str, object]
    source_revision_equivalence_seal_sha256: str
    source_revision_subject_map: dict[str, object]
    source_revision_subject_map_seal_sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    dependency_source = parser.add_mutually_exclusive_group(required=True)
    dependency_source.add_argument(
        "--dependencies-json",
        help="Exact formal-subject dependency plan (legacy direct form).",
    )
    dependency_source.add_argument(
        "--training-manifest-json",
        help="Sealed formal_1337_training_manifest from the training controller.",
    )
    dependency_source.add_argument(
        "--training-controller-task-id",
        help=(
            "Early-queued service mode: wait for this controller, consume its "
            "sealed formal manifest, and idempotently create all evaluation tasks."
        ),
    )
    parser.add_argument(
        "--evaluation-plan-json",
        default="",
        help=(
            "Exact formal-subject evaluation task/queue list; required with "
            "--training-manifest-json."
        ),
    )
    parser.add_argument(
        "--evaluation-template-task-id",
        default="",
        help="Required with --training-controller-task-id.",
    )
    parser.add_argument(
        "--training-provenance-task-id",
        default="",
        help=(
            "Completed formal provenance attestor; required with "
            "--training-controller-task-id and consumed before any evaluation clone."
        ),
    )
    parser.add_argument(
        "--expected-training-provenance-script-sha256",
        default="",
        help="Independent pin for clearml_formal_training_provenance.py.",
    )
    parser.add_argument(
        "--evaluation-worker-queues",
        default="GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090",
        help="Comma-separated capacity slots used round-robin by planner mode.",
    )
    parser.add_argument(
        "--candidate-release-blockers-json",
        default="",
        help=(
            "Exact ordered E1/E2/E3/P0/P2 training blocker contract; required "
            "with --training-controller-task-id."
        ),
    )
    parser.add_argument(
        "--authoritative-evaluation-plan-producer-task-id",
        default="",
        help=(
            "Recovery-only producer of a sealed 26-task evaluation plan. When "
            "set, the planner must reuse those exact task IDs and cannot clone."
        ),
    )
    parser.add_argument(
        "--authoritative-evaluation-plan-seal-sha256",
        default="",
        help=(
            "Exact recovery plan seal; required with "
            "--authoritative-evaluation-plan-producer-task-id."
        ),
    )
    parser.add_argument(
        "--authoritative-evaluation-plan-amendment-receipt-seal-sha256",
        default="",
        help="Executed local amendment receipt seal; set with all amendment pins.",
    )
    parser.add_argument(
        "--authoritative-evaluation-plan-producer-source-sha256",
        default="",
        help="Standalone amended-plan producer source SHA-256.",
    )
    parser.add_argument(
        "--authoritative-evaluation-plan-amendment-evidence-seal-sha256",
        default="",
        help="Sealed formal_1337_evaluation_plan_amendment artifact identity.",
    )
    parser.add_argument(
        "--authoritative-evaluation-plan-worker-evidence-seal-sha256",
        default="",
        help="Fresh worker telemetry evidence seal bound by the amendment.",
    )
    parser.add_argument(
        "--authoritative-evaluation-plan-task-ids-sha256",
        default="",
        help="Ordered unchanged 26-task identity inventory SHA-256.",
    )
    parser.add_argument("--evaluation-plan-amendment-producer-task-id", default="")
    parser.add_argument(
        "--evaluation-plan-amendment-receipt-seal-sha256", default=""
    )
    parser.add_argument(
        "--evaluation-plan-amendment-revised-plan-seal-sha256", default=""
    )
    parser.add_argument(
        "--evaluation-plan-amendment-evidence-seal-sha256", default=""
    )
    parser.add_argument(
        "--evaluation-plan-amendment-worker-evidence-seal-sha256", default=""
    )
    parser.add_argument("--evaluation-plan-amendment-task-ids-sha256", default="")
    parser.add_argument(
        "--exact-eval-runtime-recovery-receipt-seal-sha256", default=""
    )
    parser.add_argument(
        "--exact-eval-runtime-recovery-attempt-seal-sha256", default=""
    )
    parser.add_argument("--exact-eval-runtime-ffnet-source-sha256", default="")
    parser.add_argument("--exact-eval-runtime-ffnet-parameters-sha256", default="")
    parser.add_argument("--exact-eval-runtime-candidate-source-sha256", default="")
    parser.add_argument(
        "--exact-eval-runtime-candidate-parameters-sha256", default=""
    )
    parser.add_argument(
        "--expected-eval-script-sha256",
        default="",
        help="Planner mode derives this from the sealed evaluation template.",
    )
    parser.add_argument(
        "--expected-training-script-sha256",
        default="",
        help=(
            "Independent single training-child bootstrap pin for non-controller "
            "legacy plans. Controller mode derives a per-subject policy only from "
            "the sealed provenance task."
        ),
    )
    parser.add_argument(
        "--expected-source-dataset-id",
        default=EXPECTED_SOURCE_DATASET_ID,
    )
    parser.add_argument(
        "--expected-source-archive-sha256",
        default=EXPECTED_SOURCE_ARCHIVE_SHA256,
    )
    parser.add_argument(
        "--expected-training-source-dataset-id",
        default="",
        help="Defaults to --expected-source-dataset-id.",
    )
    parser.add_argument(
        "--expected-training-source-archive-sha256",
        default="",
        help="Defaults to --expected-source-archive-sha256.",
    )
    parser.add_argument(
        "--expected-training-dataset-id",
        default=EXPECTED_TRAINING_DATASET_ID,
    )
    parser.add_argument(
        "--metadata-only-artifact-gate",
        action="store_true",
        help=(
            "Validate artifact metadata without downloading artifact contents. "
            "The default remains strict content validation."
        ),
    )
    parser.add_argument(
        "--expected-predecessor-task-id",
        default=EXPECTED_PREDECESSOR_TASK_ID,
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=72.0)
    return parser


def _task_id(value: object, context: str) -> str:
    result = str(value or "")
    if CLEARML_ID_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase 32-hex ClearML ID")
    return result


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return result


def _status(task: object) -> str:
    reload_method = getattr(task, "reload", None)
    if callable(reload_method):
        reload_method()
    return str(getattr(task, "status", "") or "").lower()


def _parameters(task: object) -> Mapping[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError("ClearML task does not expose parameters")
    result = getter(cast=False)
    if not isinstance(result, Mapping):
        raise RuntimeError("ClearML task returned invalid parameters")
    return result


def _parameter_identity(
    task: object,
    *,
    keys: tuple[str, ...],
    context: str,
) -> dict[str, str]:
    parameters = _parameters(task)
    identity: dict[str, str] = {}
    for key in keys:
        raw_value = parameters.get(key)
        if raw_value is None or raw_value == "":
            raise RuntimeError(f"{context} lacks shared runtime parameter {key}")
        identity[key] = str(raw_value)
    return identity


def _runtime_identity(task: object, *, context: str) -> dict[str, str]:
    return _parameter_identity(
        task,
        keys=SHARED_RUNTIME_PARAMETER_KEYS,
        context=context,
    )


def _invariant_runtime_identity(task: object, *, context: str) -> dict[str, str]:
    return _parameter_identity(
        task,
        keys=INVARIANT_RUNTIME_PARAMETER_KEYS,
        context=context,
    )


def _source_runtime_identity(task: object, *, context: str) -> dict[str, str]:
    return _parameter_identity(
        task,
        keys=SOURCE_RUNTIME_PARAMETER_KEYS,
        context=context,
    )


def _expected_source_runtime_identity(revision: str) -> dict[str, str]:
    source = EXPECTED_SOURCE_BY_REVISION.get(revision)
    if source is None:
        raise RuntimeError("unknown sealed source revision")
    return {
        "Args/source_dataset_id": str(source["dataset_id"]),
        "Args/source_archive_name": str(source["archive_name"]),
        "Args/source_archive_bytes": str(source["archive_size_bytes"]),
        "Args/source_archive_sha256": str(source["archive_sha256"]),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_valid_seal(value: Mapping[str, object], *, context: str) -> None:
    observed = _sha256(value.get("seal_sha256"), f"{context} seal")
    payload = dict(value)
    payload.pop("seal_sha256", None)
    expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if observed != expected:
        raise ValueError(f"{context} seal SHA-256 mismatch")


def _json_value(raw: str, *, context: str) -> object:
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} is not valid JSON") from error


def _validate_dependency_task_ids(dependencies: list[dict[str, str]]) -> None:
    training_ids = [item["training_task_id"] for item in dependencies]
    evaluation_ids = [item["evaluation_task_id"] for item in dependencies]
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("training task IDs must be unique")
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("evaluation task IDs must be unique")
    if set(training_ids) & set(evaluation_ids):
        raise ValueError("training and evaluation task IDs must be disjoint")


def _parse_dependencies(
    raw: str,
    *,
    default_training_predecessor_task_id: str = EXPECTED_PREDECESSOR_TASK_ID,
) -> list[dict[str, str]]:
    value = _json_value(raw, context="dependencies JSON")
    if not isinstance(value, list):
        raise ValueError("dependencies JSON must be a list")
    if len(value) != len(FORMAL_SUBJECT_ORDER):
        raise ValueError(
            "dependencies must contain exactly "
            f"{len(FORMAL_SUBJECT_ORDER)} formal subjects"
        )
    default_predecessor = _task_id(
        default_training_predecessor_task_id,
        "default training predecessor task",
    )
    result: list[dict[str, str]] = []
    for index, (item, expected_subject) in enumerate(
        zip(value, FORMAL_SUBJECT_ORDER, strict=True),
        start=1,
    ):
        if not isinstance(item, Mapping):
            raise ValueError(f"dependency {index} must be an object")
        subject = str(item.get("subject") or "")
        if subject != expected_subject:
            raise ValueError(
                f"dependency {index} subject mismatch: expected {expected_subject!r}"
            )
        training_task_id = _task_id(
            item.get("training_task_id"), f"dependency {subject} training task"
        )
        training_predecessor_task_id = _task_id(
            item.get("training_predecessor_task_id") or default_predecessor,
            f"dependency {subject} training predecessor",
        )
        evaluation_task_id = _task_id(
            item.get("evaluation_task_id"), f"dependency {subject} evaluation task"
        )
        expected_model_fields = (
            item.get("training_model_id"),
            item.get("training_checkpoint_sha256"),
            item.get("training_model_url"),
        )
        if any(value not in {None, ""} for value in expected_model_fields) and not all(
            value not in {None, ""} for value in expected_model_fields
        ):
            raise ValueError(f"dependency {subject} final model binding is incomplete")
        training_model_id = (
            _task_id(expected_model_fields[0], f"dependency {subject} final model")
            if expected_model_fields[0] not in {None, ""}
            else ""
        )
        training_checkpoint_sha256 = (
            _sha256(
                expected_model_fields[1],
                f"dependency {subject} final checkpoint",
            )
            if expected_model_fields[1] not in {None, ""}
            else ""
        )
        training_model_url = (
            _reasonable_remote_url(
                expected_model_fields[2],
                f"dependency {subject} final model",
            )
            if expected_model_fields[2] not in {None, ""}
            else ""
        )
        training_initialization_audit_sha256 = (
            _sha256(
                item.get("training_initialization_audit_sha256"),
                f"dependency {subject} initialization audit",
            )
            if item.get("training_initialization_audit_sha256") not in {None, ""}
            else ""
        )
        queue = str(item.get("queue") or "")
        if queue not in EXPECTED_QUEUE_IDS:
            raise ValueError(f"dependency {subject} has an unsupported queue")
        result.append(
            {
                "subject": subject,
                "training_task_id": training_task_id,
                "training_predecessor_task_id": training_predecessor_task_id,
                "training_model_id": training_model_id,
                "training_checkpoint_sha256": training_checkpoint_sha256,
                "training_model_url": training_model_url,
                "training_initialization_audit_sha256": (
                    training_initialization_audit_sha256
                ),
                "evaluation_task_id": evaluation_task_id,
                "queue": queue,
            }
        )
    _validate_dependency_task_ids(result)
    return result


def _parse_training_manifest(raw: str) -> list[dict[str, str]]:
    value = _json_value(raw, context="training manifest JSON")
    if not isinstance(value, Mapping):
        raise ValueError("training manifest JSON must be an object")
    if "formal_1337_evaluation_manifest" in value:
        _require_valid_seal(value, context="training controller summary")
        summary_expected = {
            "summary_type": "resilient_v2x_post_main_sequential_training",
            "status": "completed",
            "experiment_order": list(FORMAL_SUBJECT_ORDER),
            "task_count": len(FORMAL_SUBJECT_ORDER),
            "formal_1337_manifest_artifact": "formal_1337_training_manifest",
        }
        for key, expected in summary_expected.items():
            if value.get(key) != expected:
                raise ValueError(f"training controller summary {key} mismatch")
        value = value["formal_1337_evaluation_manifest"]
        if not isinstance(value, Mapping):
            raise ValueError("controller formal 1337 manifest is not an object")
    _require_valid_seal(value, context="formal 1337 training manifest")
    expected_fields = {
        "schema_version": 1,
        "manifest_type": "resilient_v2x_formal_1337_training_inputs",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "delays_ms": list(EXPECTED_DELAYS_MS),
        "conditions": list(EXPECTED_CONDITIONS),
        "run_count": len(EXPECTED_DELAYS_MS) * len(EXPECTED_CONDITIONS),
        "checkpoint_policy": EXPECTED_CHECKPOINT_POLICY,
        "evaluation_release_semantics": EXPECTED_RELEASE_SEMANTICS,
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "subject_count": len(FORMAL_SUBJECT_ORDER),
    }
    for key, expected in expected_fields.items():
        if value.get(key) != expected:
            raise ValueError(f"formal 1337 training manifest {key} mismatch")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != len(FORMAL_SUBJECT_ORDER):
        raise ValueError("formal 1337 training manifest entry count mismatch")
    result: list[dict[str, str]] = []
    model_ids: set[str] = set()
    for index, (entry, subject) in enumerate(
        zip(entries, FORMAL_SUBJECT_ORDER, strict=True),
        start=1,
    ):
        if not isinstance(entry, Mapping):
            raise ValueError(f"formal 1337 training entry {index} is invalid")
        if entry.get("index") != index or entry.get("subject") != subject:
            raise ValueError("formal 1337 training manifest entry order mismatch")
        training_task_id = _task_id(
            entry.get("training_task_id"), f"{subject} training task"
        )
        predecessor_task_id = _task_id(
            entry.get("training_predecessor_task_id"),
            f"{subject} training predecessor",
        )
        model_id = _task_id(entry.get("model_id"), f"{subject} final model")
        if model_id in model_ids:
            raise ValueError("formal 1337 final model IDs must be unique")
        model_ids.add(model_id)
        if entry.get("model_name") != (f"ResilientV2X {subject} final checkpoint"):
            raise ValueError(f"formal 1337 {subject} model name mismatch")
        if entry.get("checkpoint_filename") != "epoch_50.pth":
            raise ValueError(f"formal 1337 {subject} checkpoint policy mismatch")
        model_url = _reasonable_remote_url(
            entry.get("model_url"), f"formal 1337 {subject} final model"
        )
        if PurePosixPath(unquote(urlparse(model_url).path)).name != (
            f"{subject}_epoch_50.pth"
        ):
            raise ValueError(f"formal 1337 {subject} model filename mismatch")
        checkpoint_sha256 = _sha256(
            entry.get("checkpoint_sha256"), f"{subject} checkpoint"
        )
        if (
            type(entry.get("checkpoint_size_bytes")) is not int
            or entry["checkpoint_size_bytes"] <= 0
        ):
            raise ValueError(f"formal 1337 {subject} checkpoint size is invalid")
        if entry.get("common_teacher_initialization_audit_artifact") != (
            COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT
        ):
            raise ValueError(
                f"formal 1337 {subject} initialization audit artifact mismatch"
            )
        initialization_audit_sha256 = _sha256(
            entry.get("common_teacher_initialization_audit_sha256"),
            f"formal 1337 {subject} initialization audit",
        )
        result.append(
            {
                "subject": subject,
                "training_task_id": training_task_id,
                "training_predecessor_task_id": predecessor_task_id,
                "training_model_id": model_id,
                "training_checkpoint_sha256": checkpoint_sha256,
                "training_model_url": model_url,
                "training_initialization_audit_sha256": (initialization_audit_sha256),
            }
        )
    training_ids = [item["training_task_id"] for item in result]
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("formal 1337 training task IDs must be unique")
    return result


def _parse_evaluation_plan(raw: str) -> list[dict[str, str]]:
    value = _json_value(raw, context="evaluation plan JSON")
    if not isinstance(value, list) or len(value) != len(FORMAL_SUBJECT_ORDER):
        raise ValueError(
            "evaluation plan must contain exactly "
            f"{len(FORMAL_SUBJECT_ORDER)} formal subjects"
        )
    result: list[dict[str, str]] = []
    for index, (item, subject) in enumerate(
        zip(value, FORMAL_SUBJECT_ORDER, strict=True),
        start=1,
    ):
        if not isinstance(item, Mapping) or item.get("subject") != subject:
            raise ValueError(f"evaluation plan entry {index} subject mismatch")
        evaluation_task_id = _task_id(
            item.get("evaluation_task_id"), f"{subject} evaluation task"
        )
        queue = str(item.get("queue") or "")
        if queue not in EXPECTED_QUEUE_IDS:
            raise ValueError(f"evaluation plan {subject} has an unsupported queue")
        result.append(
            {
                "subject": subject,
                "evaluation_task_id": evaluation_task_id,
                "queue": queue,
            }
        )
    evaluation_ids = [item["evaluation_task_id"] for item in result]
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("evaluation task IDs must be unique")
    return result


def build_dependency_plan(
    training_manifest_json: str,
    evaluation_plan_json: str,
) -> list[dict[str, str]]:
    training = _parse_training_manifest(training_manifest_json)
    evaluations = _parse_evaluation_plan(evaluation_plan_json)
    dependencies = [
        {**training_item, **evaluation_item}
        for training_item, evaluation_item in zip(training, evaluations, strict=True)
    ]
    _validate_dependency_task_ids(dependencies)
    return dependencies


def _planner_worker_queues(raw: object) -> list[str]:
    if type(raw) is not str:
        raise ValueError("evaluation worker queues must be a comma-separated string")
    queues = [part.strip() for part in raw.split(",") if part.strip()]
    if not queues or any(queue not in EXPECTED_QUEUE_IDS for queue in queues):
        raise ValueError("evaluation worker queues contain an unsupported queue")
    return queues


def _dependencies_in_release_order(
    dependencies: list[dict[str, str]],
) -> list[dict[str, str]]:
    dependency_by_subject = {item["subject"]: item for item in dependencies}
    if (
        len(dependency_by_subject) != len(dependencies)
        or set(dependency_by_subject) != set(FORMAL_SUBJECT_ORDER)
    ):
        raise RuntimeError("formal evaluation dependency inventory drifted")
    priority = FORMAL_EVALUATION_RELEASE_PRIORITY
    if len(set(priority)) != len(priority) or any(
        subject not in FORMAL_SUBJECT_ORDER for subject in priority
    ):
        raise RuntimeError("formal evaluation release priority drifted")
    release_subject_order = (
        *priority,
        *(subject for subject in FORMAL_SUBJECT_ORDER if subject not in priority),
    )
    return [dependency_by_subject[subject] for subject in release_subject_order]


def _formal_phase_allows_release(
    subject: str, *, candidate_phase_ready: bool
) -> bool:
    return (
        subject in FORMAL_EVALUATION_RELEASE_PRIORITY
        or candidate_phase_ready
    )


def _formal_core_phase_status_ready(
    task_class: object, *, dependencies: Sequence[Mapping[str, str]]
) -> bool:
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise RuntimeError("ClearML task class cannot inspect formal core evaluations")
    dependency_by_subject = {
        str(dependency["subject"]): dependency for dependency in dependencies
    }
    if not set(FORMAL_EVALUATION_RELEASE_PRIORITY) <= set(dependency_by_subject):
        raise RuntimeError("formal core evaluation inventory drifted")
    return all(
        _status(
            getter(
                task_id=dependency_by_subject[subject]["evaluation_task_id"]
            )
        )
        == "completed"
        for subject in FORMAL_EVALUATION_RELEASE_PRIORITY
    )


def _claim_single_queue_release(
    claimed_queues: set[str], *, queue_name: str
) -> bool:
    if queue_name not in EXPECTED_QUEUE_IDS:
        raise RuntimeError("formal release queue is unsupported")
    if queue_name in claimed_queues:
        return False
    claimed_queues.add(queue_name)
    return True


def _candidate_evaluation_phase_snapshot(
    task_class: object,
    *,
    expected_project_id: str,
    core_phase_ready: bool,
) -> tuple[dict[str, object], bool]:
    query = getattr(task_class, "query_tasks", None)
    getter = getattr(task_class, "get_task", None)
    if not callable(query) or not callable(getter):
        raise RuntimeError(
            "ClearML task class cannot inspect candidate phase evaluations"
        )
    entries: list[dict[str, object]] = []
    observed_ids: set[str] = set()
    ready = core_phase_ready
    active_count = 0
    for spec in CANDIDATE_EVALUATION_PHASE:
        label = str(spec["label"])
        subject = str(spec["subject"])
        training_task_id = str(spec["training_task_id"])
        expected_name = (
            f"ResilientV2X formal1337 candidate eval {label} {subject} "
            f"[{training_task_id[:12]}]"
        )
        values = query(
            task_filter={
                "project": [expected_project_id],
                "parent": training_task_id,
            }
        )
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RuntimeError("candidate phase lookup returned an invalid result")
        matches: list[object] = []
        for raw in values:
            raw_id = raw.get("id") if isinstance(raw, Mapping) else raw
            task_id = _task_id(raw_id, f"candidate {label} evaluation")
            task = getter(task_id=task_id)
            if (
                str(getattr(task, "name", "") or "") == expected_name
                and _task_parent(task) == training_task_id
                and _task_project_id(
                    task, context=f"candidate {label} evaluation"
                )
                == expected_project_id
            ):
                matches.append(task)
        if len(matches) > 1:
            raise RuntimeError(
                f"candidate {label} evaluation lookup is ambiguous"
            )
        if not matches:
            if spec["evaluation_task_id"]:
                raise RuntimeError(
                    f"candidate {label} pinned evaluation is missing"
                )
            entries.append(
                {
                    "label": label,
                    "subject": subject,
                    "training_task_id": training_task_id,
                    "evaluation_task_id": None,
                    "status": "not_created",
                    "queue": spec["queue"],
                }
            )
            ready = False
            continue
        task = matches[0]
        task_id = _task_id(
            getattr(task, "id", ""), f"candidate {label} evaluation"
        )
        if spec["evaluation_task_id"] and task_id != spec["evaluation_task_id"]:
            raise RuntimeError(f"candidate {label} evaluation ID drifted")
        if task_id in observed_ids:
            raise RuntimeError("candidate phase evaluation IDs are not unique")
        observed_ids.add(task_id)
        status = _status(task)
        if status in FAILED_STATUSES:
            raise RuntimeError(
                f"candidate {label} evaluation ended as {status!r}"
            )
        if status not in {"created", "queued", "in_progress", "completed"}:
            raise RuntimeError(f"candidate {label} evaluation status is invalid")
        execution = getattr(getattr(task, "data", None), "execution", None)
        queue_id = str(getattr(execution, "queue", "") or "")
        if status in {"queued", "in_progress", "completed"} and queue_id != (
            EXPECTED_QUEUE_IDS[str(spec["queue"])]
        ):
            raise RuntimeError(f"candidate {label} evaluation queue drifted")
        if status in {"queued", "in_progress", "completed"}:
            if not core_phase_ready:
                raise RuntimeError(
                    f"candidate {label} evaluation escaped the incomplete "
                    "formal core phase"
                )
        if status in {"queued", "in_progress"}:
            active_count += 1
            if active_count > MAX_ACTIVE_CANDIDATE_EVALUATIONS:
                raise RuntimeError(
                    "candidate evaluation active concurrency exceeds "
                    f"{MAX_ACTIVE_CANDIDATE_EVALUATIONS}"
                )
        if status == "completed":
            _require_completed_evaluation_metrics(
                task,
                dependency={
                    "subject": subject,
                    "evaluation_task_id": task_id,
                },
            )
        else:
            ready = False
        entries.append(
            {
                "label": label,
                "subject": subject,
                "training_task_id": training_task_id,
                "evaluation_task_id": task_id,
                "status": status,
                "queue": spec["queue"],
                "execution_queue_id": queue_id,
            }
        )
    return (
        {
            "policy": (
                "formal_core_six_then_candidate_five_parallel_three_then_"
                "remaining_formal"
            ),
            "max_active_candidate_evaluations": (
                MAX_ACTIVE_CANDIDATE_EVALUATIONS
            ),
            "active_candidate_evaluations": active_count,
            "ready": ready,
            "entries": entries,
        },
        ready,
    )


def _record_value(record: object, key: str) -> object:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _worker_resource(worker_id: object, *, context: str) -> tuple[str, frozenset[int]]:
    if type(worker_id) is not str:
        raise RuntimeError(f"{context} worker ID is invalid")
    match = WORKER_GPU_ID_PATTERN.fullmatch(worker_id.strip())
    if match is None:
        raise RuntimeError(f"{context} worker ID has no GPU resource identity")
    gpu_ids = tuple(int(value) for value in match.group("gpu").split(","))
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise RuntimeError(f"{context} worker GPU inventory is invalid")
    return match.group("host"), frozenset(gpu_ids)


def _worker_queue_ids(worker: object, *, context: str) -> tuple[str, ...]:
    values = _record_value(worker, "queues")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError(f"{context} queue inventory is invalid")
    result: list[str] = []
    for index, value in enumerate(values):
        result.append(
            _task_id(_record_value(value, "id"), f"{context} queue {index}")
        )
    if len(set(result)) != len(result):
        raise RuntimeError(f"{context} queue inventory contains duplicates")
    return tuple(result)


def _live_worker_rows() -> list[object]:
    try:
        from clearml.backend_api.session.client import APIClient

        response = APIClient().workers.get_all()
        rows = list(response)
    except Exception as error:
        raise RuntimeError(
            "ClearML worker inventory is unavailable for formal resource gating"
        ) from error
    if not rows:
        raise RuntimeError("ClearML worker inventory is empty")
    return rows


def _finite_scheduler_telemetry_values(
    value: object, *, context: str
) -> list[float]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) not in {int, float} for item in value)
    ):
        raise RuntimeError(f"{context} values are invalid")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise RuntimeError(f"{context} values are not finite and non-negative")
    return result


def _scheduler_telemetry_metric_values(
    metric: object, *, context: str
) -> list[float]:
    value = metric.to_dict() if hasattr(metric, "to_dict") else metric
    if not isinstance(value, Mapping) or value.get("metric") != context:
        raise RuntimeError(f"worker {context} metric identity drifted")
    stats = value.get("stats")
    if not isinstance(stats, list) or len(stats) != 1:
        raise RuntimeError(f"worker {context} metric aggregation drifted")
    stat = stats[0]
    if not isinstance(stat, Mapping) or stat.get("aggregation") != "avg":
        raise RuntimeError(f"worker {context} metric aggregation drifted")
    return _finite_scheduler_telemetry_values(
        stat.get("values"), context=f"worker {context}"
    )


def _scheduler_worker_gpu_telemetry(
    api_client: object, worker_id: str
) -> dict[str, object]:
    end = time.time()
    start = end - SCHEDULER_GPU_TELEMETRY_WINDOW_SECONDS
    response = api_client.workers.get_stats(
        from_date=start,
        to_date=end,
        interval=SCHEDULER_GPU_TELEMETRY_INTERVAL_SECONDS,
        items=[
            {"key": "gpu_memory_used", "category": "max"},
            {"key": "gpu_usage", "category": "max"},
        ],
        worker_ids=[worker_id],
        split_by_variant=True,
    )
    payload = response.to_dict() if hasattr(response, "to_dict") else response
    workers = payload.get("workers") if isinstance(payload, Mapping) else None
    if not isinstance(workers, list) or len(workers) != 1:
        raise RuntimeError(f"worker {worker_id} telemetry response drifted")
    worker = workers[0]
    if not isinstance(worker, Mapping) or worker.get("worker") != worker_id:
        raise RuntimeError(f"worker {worker_id} telemetry identity drifted")
    metrics = worker.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 2:
        raise RuntimeError(f"worker {worker_id} telemetry metrics drifted")
    by_name: dict[str, object] = {}
    for item in metrics:
        normalized = item.to_dict() if hasattr(item, "to_dict") else item
        if isinstance(normalized, Mapping):
            by_name[str(normalized.get("metric") or "")] = item
    if set(by_name) != {"gpu_memory_used", "gpu_usage"}:
        raise RuntimeError(f"worker {worker_id} telemetry metric inventory drifted")
    memory = _scheduler_telemetry_metric_values(
        by_name["gpu_memory_used"], context="gpu_memory_used"
    )
    usage = _scheduler_telemetry_metric_values(
        by_name["gpu_usage"], context="gpu_usage"
    )
    if len(memory) != len(usage):
        raise RuntimeError(f"worker {worker_id} telemetry sample count drifted")
    return {
        "worker_id": worker_id,
        "window_from_unix": start,
        "window_to_unix": end,
        "interval_seconds": SCHEDULER_GPU_TELEMETRY_INTERVAL_SECONDS,
        "gpu_memory_used_mib": memory,
        "gpu_usage_percent": usage,
        "max_gpu_memory_used_mib": max(memory),
        "max_gpu_usage_percent": max(usage),
    }


def _fresh_scheduler_gpu_telemetry(
    worker_ids: Sequence[str],
) -> dict[str, dict[str, object]]:
    expected = tuple(sorted(worker_ids))
    if not expected or len(set(expected)) != len(expected):
        raise RuntimeError("scheduler telemetry worker inventory is invalid")
    try:
        from clearml.backend_api.session.client import APIClient

        api_client = APIClient()
        result = {
            worker_id: _scheduler_worker_gpu_telemetry(api_client, worker_id)
            for worker_id in expected
        }
    except Exception as error:
        raise RuntimeError(
            "fresh ClearML worker GPU telemetry is unavailable"
        ) from error
    return result


def _validate_scheduler_gpu_telemetry(
    value: object, *, expected_worker_ids: Sequence[str]
) -> dict[str, dict[str, object]]:
    expected = tuple(sorted(expected_worker_ids))
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise RuntimeError("scheduler GPU telemetry inventory drifted")
    result: dict[str, dict[str, object]] = {}
    expected_keys = {
        "worker_id",
        "window_from_unix",
        "window_to_unix",
        "interval_seconds",
        "gpu_memory_used_mib",
        "gpu_usage_percent",
        "max_gpu_memory_used_mib",
        "max_gpu_usage_percent",
    }
    for worker_id in expected:
        raw = value[worker_id]
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise RuntimeError(f"worker {worker_id} telemetry shape drifted")
        record = dict(raw)
        if record.get("worker_id") != worker_id:
            raise RuntimeError(f"worker {worker_id} telemetry identity drifted")
        if (
            type(record.get("window_from_unix")) not in {int, float}
            or type(record.get("window_to_unix")) not in {int, float}
            or float(record["window_to_unix"])
            <= float(record["window_from_unix"])
            or record.get("interval_seconds")
            != SCHEDULER_GPU_TELEMETRY_INTERVAL_SECONDS
        ):
            raise RuntimeError(f"worker {worker_id} telemetry window drifted")
        memory = _finite_scheduler_telemetry_values(
            record.get("gpu_memory_used_mib"),
            context=f"worker {worker_id} GPU memory",
        )
        usage = _finite_scheduler_telemetry_values(
            record.get("gpu_usage_percent"),
            context=f"worker {worker_id} GPU usage",
        )
        if len(memory) != len(usage):
            raise RuntimeError(f"worker {worker_id} telemetry sample count drifted")
        if record.get("max_gpu_memory_used_mib") != max(memory) or record.get(
            "max_gpu_usage_percent"
        ) != max(usage):
            raise RuntimeError(f"worker {worker_id} telemetry summary drifted")
        result[worker_id] = record
    return result


def _query_task_ids_by_status(
    task_class: object, statuses: frozenset[str]
) -> list[str]:
    query = getattr(task_class, "query_tasks", None)
    if not callable(query):
        raise RuntimeError(
            "ClearML task class cannot authoritatively inspect queue occupancy"
        )
    values = query(task_filter={"status": sorted(statuses)})
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError("ClearML returned an invalid queue occupancy result")
    result: list[str] = []
    for index, value in enumerate(values):
        raw_id = value.get("id") if isinstance(value, Mapping) else value
        result.append(_task_id(raw_id, f"queue occupancy task {index}"))
    if len(set(result)) != len(result):
        raise RuntimeError("ClearML returned duplicate queue occupancy tasks")
    return result


def _resources_overlap(
    left: tuple[str, frozenset[int]], right: tuple[str, frozenset[int]]
) -> bool:
    return left[0] == right[0] and not left[1].isdisjoint(right[1])


def _live_disjoint_worker_capacity(
    worker_rows: Sequence[object], *, target_queue_id: str
) -> int:
    selected_queue_id = _task_id(target_queue_id, "formal target queue")
    workers: list[tuple[str, tuple[str, frozenset[int]]]] = []
    for index, worker in enumerate(worker_rows):
        queues = _worker_queue_ids(worker, context=f"worker {index}")
        if selected_queue_id not in queues:
            continue
        worker_id = _record_value(worker, "id")
        resource = _worker_resource(worker_id, context=f"worker {index}")
        workers.append((str(worker_id).strip(), resource))
    if not workers:
        raise RuntimeError("formal target queue has no auditable live GPU worker")
    worker_ids = [worker_id for worker_id, _ in workers]
    if len(set(worker_ids)) != len(worker_ids):
        raise RuntimeError("ClearML worker inventory contains duplicate workers")
    for index, (worker_id, resource) in enumerate(workers):
        for other_worker_id, other_resource in workers[index + 1 :]:
            if _resources_overlap(resource, other_resource):
                raise RuntimeError(
                    "formal target queue live worker resources overlap: "
                    f"{worker_id}, {other_worker_id}"
                )
    return len(workers)


def _formal_resource_gate_snapshot(
    task_class: object,
    *,
    formal_evaluation_task_ids: set[str],
    target_queue_id: str,
    worker_rows: Sequence[object] | None = None,
    worker_telemetry: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prove one queue has a live, disjoint, telemetrically idle worker slot."""

    rows = list(worker_rows) if worker_rows is not None else _live_worker_rows()
    if not rows:
        raise RuntimeError("ClearML worker inventory is empty")
    target_queue_ids = frozenset(EXPECTED_QUEUE_IDS.values())
    selected_queue_id = _task_id(target_queue_id, "formal target queue")
    if selected_queue_id not in target_queue_ids:
        raise ValueError("formal target queue is unsupported")
    queue_resources: dict[str, set[tuple[str, frozenset[int]]]] = {}
    worker_resources: dict[str, tuple[str, frozenset[int]]] = {}
    queue_workers: dict[str, set[str]] = {}
    for index, worker in enumerate(rows):
        worker_id = _record_value(worker, "id")
        queues = _worker_queue_ids(worker, context=f"worker {index}")
        if not queues:
            continue
        if not WORKER_GPU_ID_PATTERN.fullmatch(str(worker_id or "").strip()):
            if target_queue_ids.intersection(queues):
                raise RuntimeError("formal target queue is served by an invalid worker")
            continue
        resource = _worker_resource(worker_id, context=f"worker {index}")
        normalized_worker_id = str(worker_id).strip()
        if normalized_worker_id in worker_resources:
            raise RuntimeError("ClearML worker inventory contains duplicate workers")
        worker_resources[normalized_worker_id] = resource
        for queue_id in queues:
            queue_resources.setdefault(queue_id, set()).add(resource)
            queue_workers.setdefault(queue_id, set()).add(normalized_worker_id)

    if selected_queue_id not in queue_resources:
        raise RuntimeError("formal target queue has no auditable live GPU worker")
    target_resources = set(queue_resources[selected_queue_id])
    target_worker_ids = set(queue_workers[selected_queue_id])
    ordered_target_workers = sorted(target_worker_ids)
    for index, worker_id in enumerate(ordered_target_workers):
        for other_worker_id in ordered_target_workers[index + 1 :]:
            if _resources_overlap(
                worker_resources[worker_id], worker_resources[other_worker_id]
            ):
                raise RuntimeError(
                    "formal target queue live worker resources overlap"
                )
    queued_blockers: list[dict[str, object]] = []
    overlapping_active: list[dict[str, object]] = []
    for task_id in _query_task_ids_by_status(
        task_class, RESOURCE_BLOCKING_STATUSES
    ):
        task = task_class.get_task(task_id=task_id)
        if _task_id(getattr(task, "id", ""), "queue occupancy task") != task_id:
            raise RuntimeError("queue occupancy task identity drifted")
        status = _status(task)
        if status not in RESOURCE_BLOCKING_STATUSES:
            continue
        execution = getattr(getattr(task, "data", None), "execution", None)
        raw_queue_id = str(getattr(execution, "queue", "") or "").strip()
        queue_id: str | None = None
        if raw_queue_id:
            try:
                queue_id = _task_id(
                    raw_queue_id, "queue occupancy execution queue"
                )
            except ValueError:
                # ClearML can expose a stale or transitional execution.queue
                # while status readback still says queued/in_progress. An
                # active task is accounted for from its physical last_worker;
                # target-worker telemetry remains the fail-closed release gate.
                queue_id = None
        last_worker_raw = getattr(getattr(task, "data", None), "last_worker", None)
        last_worker = str(last_worker_raw or "").strip()
        if status == "queued":
            if queue_id == selected_queue_id:
                queued_blockers.append(
                    {
                        "task_id": task_id,
                        "status": status,
                        "queue_id": queue_id,
                        "last_worker": last_worker or None,
                        "reasons": ["target_queue_waiting_task"],
                    }
                )
            continue
        queue_can_overlap = any(
            _resources_overlap(resource, target_resource)
            for resource in queue_resources.get(queue_id or "", set())
            for target_resource in target_resources
        )
        if not last_worker:
            if queue_id == selected_queue_id or queue_can_overlap:
                raise RuntimeError("active queue occupancy task has no last_worker")
            continue
        try:
            active_resource = worker_resources.get(last_worker) or _worker_resource(
                last_worker, context="active queue occupancy task"
            )
        except RuntimeError:
            if queue_id == selected_queue_id or queue_can_overlap:
                raise
            continue
        overlapping_workers = sorted(
            worker_id
            for worker_id in target_worker_ids
            if _resources_overlap(active_resource, worker_resources[worker_id])
        )
        if overlapping_workers:
            overlapping_active.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "queue_id": queue_id,
                    "last_worker": last_worker or None,
                    "active_resource": (
                        f"{active_resource[0]}:gpu"
                        + ",".join(str(value) for value in sorted(active_resource[1]))
                    ),
                    "overlapping_target_workers": overlapping_workers,
                }
            )
    occupied_target_worker_ids = {
        worker_id
        for item in overlapping_active
        for worker_id in item["overlapping_target_workers"]
    }
    active_task_ids_by_worker: dict[str, list[str]] = {}
    for item in overlapping_active:
        for worker_id in item["overlapping_target_workers"]:
            active_task_ids_by_worker.setdefault(worker_id, []).append(
                str(item["task_id"])
            )
    if any(len(task_ids) > 1 for task_ids in active_task_ids_by_worker.values()):
        raise RuntimeError(
            "multiple active tasks overlap one formal target worker"
        )
    idle_target_worker_ids = sorted(target_worker_ids - occupied_target_worker_ids)
    active_blockers = []
    if not idle_target_worker_ids:
        active_blockers = [
            {
                "task_id": item["task_id"],
                "status": item["status"],
                "queue_id": item["queue_id"],
                "last_worker": item["last_worker"],
                "reasons": ["overlapping_active_worker"],
            }
            for item in overlapping_active
        ]
    telemetry: dict[str, dict[str, object]] = {}
    telemetry_blockers: list[dict[str, object]] = []
    if not queued_blockers and idle_target_worker_ids:
        raw_telemetry = (
            worker_telemetry
            if worker_telemetry is not None
            else _fresh_scheduler_gpu_telemetry(idle_target_worker_ids)
        )
        telemetry = _validate_scheduler_gpu_telemetry(
            raw_telemetry,
            expected_worker_ids=idle_target_worker_ids,
        )
        for worker_id in idle_target_worker_ids:
            record = telemetry[worker_id]
            if (
                float(record["max_gpu_memory_used_mib"])
                > SCHEDULER_GPU_MEMORY_IDLE_LIMIT_MIB
                or float(record["max_gpu_usage_percent"])
                > SCHEDULER_GPU_USAGE_IDLE_LIMIT_PERCENT
            ):
                telemetry_blockers.append(
                    {
                        "task_id": None,
                        "status": "unregistered_gpu_occupancy",
                        "queue_id": selected_queue_id,
                        "last_worker": worker_id,
                        "reasons": ["fresh_gpu_telemetry_not_idle"],
                    }
                )
    blockers = [*queued_blockers, *active_blockers, *telemetry_blockers]
    blockers.sort(key=lambda item: (str(item["queue_id"]), str(item["task_id"])))
    selected_idle_worker = (
        idle_target_worker_ids[0]
        if idle_target_worker_ids and not telemetry_blockers and not queued_blockers
        else None
    )
    return {
        "ready": selected_idle_worker is not None,
        "policy": (
            "live_disjoint_worker_capacity_with_fresh_gpu_telemetry_fail_closed"
        ),
        "target_queue_id": selected_queue_id,
        "target_workers": [
            {
                "worker_id": worker_id,
                "resource": (
                    f"{worker_resources[worker_id][0]}:gpu"
                    + ",".join(
                        str(value)
                        for value in sorted(worker_resources[worker_id][1])
                    )
                ),
            }
            for worker_id in sorted(target_worker_ids)
        ],
        "idle_target_worker_ids": idle_target_worker_ids,
        "pullable_target_worker_ids": idle_target_worker_ids,
        "selected_idle_target_worker_id": selected_idle_worker,
        "occupied_target_worker_ids": sorted(occupied_target_worker_ids),
        "live_disjoint_worker_count": len(target_worker_ids),
        "active_target_worker_count": len(occupied_target_worker_ids),
        "managed_active_evaluation_count": sum(
            1
            for item in overlapping_active
            if item["task_id"] in formal_evaluation_task_ids
        ),
        "blocking_statuses": sorted(RESOURCE_BLOCKING_STATUSES),
        "observed_overlapping_active_tasks": overlapping_active,
        "fresh_gpu_telemetry": telemetry,
        "gpu_telemetry_policy": {
            "window_seconds": SCHEDULER_GPU_TELEMETRY_WINDOW_SECONDS,
            "interval_seconds": SCHEDULER_GPU_TELEMETRY_INTERVAL_SECONDS,
            "max_memory_used_mib_inclusive": (
                SCHEDULER_GPU_MEMORY_IDLE_LIMIT_MIB
            ),
            "max_usage_percent_inclusive": (
                SCHEDULER_GPU_USAGE_IDLE_LIMIT_PERCENT
            ),
            "all_pullable_workers_must_be_idle_when_worker_is_not_addressable": True,
        },
        "external_blockers": blockers,
    }


def _script_sha256(task: object, *, context: str) -> str:
    script = getattr(getattr(task, "data", None), "script", None)
    if (
        str(getattr(script, "repository", "") or "") != ""
        or str(getattr(script, "working_dir", "") or "") != "."
        or str(getattr(script, "entry_point", "") or "") != "clearml_5090_bootstrap.py"
    ):
        raise RuntimeError(f"{context} script identity mismatch")
    diff = str(getattr(script, "diff", "") or "")
    if not diff:
        raise RuntimeError(f"{context} has an empty standalone script")
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _inject_gpu_memory_preflight(source: str, *, context: str) -> str:
    if source.count(GPU_PREFLIGHT_ANCHOR) != 1:
        raise RuntimeError(f"{context} GPU preflight anchor drifted")
    return source.replace(GPU_PREFLIGHT_ANCHOR, GPU_PREFLIGHT_REPLACEMENT, 1)


def _authoritative_evaluation_script_sha256(template: object) -> str:
    script = getattr(getattr(template, "data", None), "script", None)
    source = str(getattr(script, "diff", "") or "")
    patched = _inject_gpu_memory_preflight(
        source, context="authoritative evaluation template"
    )
    return hashlib.sha256(patched.encode("utf-8")).hexdigest()


def _wait_for_controller_manifest(
    controller: object,
    *,
    deadline: float,
    poll_seconds: float,
    metadata_only_artifact_gate: bool = False,
    authenticated_artifact_downloader: (
        Callable[[object, str], bytes] | None
    ) = None,
) -> dict[str, object]:
    while True:
        status = _status(controller)
        if status == "completed":
            artifacts = getattr(controller, "artifacts", None)
            if (
                not isinstance(artifacts, Mapping)
                or FORMAL_TRAINING_MANIFEST_ARTIFACT not in artifacts
            ):
                raise RuntimeError(
                    "completed training controller lacks the formal 1337 manifest"
                )
            if authenticated_artifact_downloader is not None:
                return _stable_authenticated_artifact_mapping(
                    controller,
                    FORMAL_TRAINING_MANIFEST_ARTIFACT,
                    required_status="completed",
                    downloader=authenticated_artifact_downloader,
                )
            if metadata_only_artifact_gate:
                return _stable_artifact_preview_mapping(
                    controller,
                    FORMAL_TRAINING_MANIFEST_ARTIFACT,
                    required_status="completed",
                )
            return _artifact_mapping(
                artifacts[FORMAL_TRAINING_MANIFEST_ARTIFACT],
                FORMAL_TRAINING_MANIFEST_ARTIFACT,
            )
        if status in FAILED_STATUSES:
            raise RuntimeError(
                f"training controller ended without completion: {status!r}"
            )
        if status not in {"created", "queued", "in_progress"}:
            raise RuntimeError(f"training controller has unexpected status: {status!r}")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the training controller")
        time.sleep(poll_seconds)
        reloader = getattr(controller, "reload", None)
        if callable(reloader):
            reloader()


def _planned_evaluation_name(
    *, controller_task_id: str, index: int, subject: str
) -> str:
    return f"ResilientV2X formal1337 eval {index:02d} {subject} [{controller_task_id}]"


def _task_parent(task: object) -> str:
    parent = getattr(task, "parent", None)
    if parent in {None, ""}:
        parent = getattr(getattr(task, "data", None), "parent", None)
    return str(parent or "")


def _task_project_id(task: object, *, context: str) -> str:
    observed: list[str] = []
    for source in (task, getattr(task, "data", None), getattr(task, "_data", None)):
        if source is None:
            continue
        value = getattr(source, "project", None)
        if value is None or value == "":
            continue
        try:
            observed.append(_task_id(value, f"{context} project"))
        except ValueError as error:
            raise RuntimeError(f"{context} has an invalid project ID") from error
    if not observed:
        raise RuntimeError(f"{context} does not expose its project ID")
    if len(set(observed)) != 1:
        raise RuntimeError(f"{context} project ID drifted")
    return observed[0]


def _parse_candidate_release_blockers(raw: object) -> tuple[dict[str, str], ...]:
    if type(raw) is not str or not raw:
        raise ValueError("candidate release blocker contract is required")
    value = _json_value(raw, context="candidate release blockers JSON")
    expected = [dict(item) for item in CANDIDATE_RELEASE_BLOCKERS]
    if value != expected:
        raise ValueError("candidate release blocker contract mismatch")
    labels = [str(item["label"]) for item in expected]
    task_ids = [str(item["training_task_id"]) for item in expected]
    if len(set(labels)) != len(labels) or len(set(task_ids)) != len(task_ids):
        raise RuntimeError("sealed candidate release blocker contract is invalid")
    return tuple(expected)


def _candidate_release_blocker_snapshot(
    task_class: object,
    *,
    blockers: Sequence[Mapping[str, str]],
    expected_project_id: str,
) -> tuple[list[dict[str, str]], bool]:
    if [dict(item) for item in blockers] != [
        dict(item) for item in CANDIDATE_RELEASE_BLOCKERS
    ]:
        raise RuntimeError("candidate release blocker inventory drifted")
    project_id = _task_id(expected_project_id, "candidate blocker project")
    result: list[dict[str, str]] = []
    for blocker in blockers:
        label = str(blocker["label"])
        task_id = _task_id(
            blocker["training_task_id"], f"candidate blocker {label} task"
        )
        task = task_class.get_task(task_id=task_id)
        if _task_id(getattr(task, "id", ""), f"candidate blocker {label}") != task_id:
            raise RuntimeError(f"candidate blocker {label} task identity mismatch")
        first_status = _status(task)
        if (
            str(getattr(task, "name", "") or "") != blocker["task_name"]
            or _task_parent(task) != CANDIDATE_RELEASE_BLOCKER_PARENT_TASK_ID
            or _task_project_id(task, context=f"candidate blocker {label}")
            != project_id
        ):
            raise RuntimeError(f"candidate blocker {label} identity drifted")
        parameters = _parameters(task)
        expected_parameters = {
            "Args/experiment_from_task": blocker["subject"],
            "Args/training_seed": str(EXPECTED_TRAINING_SEED),
            "Args/gpus": "4",
            "Args/max_epochs": "50",
            "Args/amp": "False",
            "Args/training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
        }
        if any(parameters.get(key) != value for key, value in expected_parameters.items()):
            raise RuntimeError(f"candidate blocker {label} training identity drifted")
        second_status = _status(task)
        if first_status != second_status:
            raise RuntimeError(f"candidate blocker {label} status changed across readback")
        if second_status in FAILED_STATUSES:
            raise RuntimeError(
                f"candidate blocker {label} ended without completion: {second_status!r}"
            )
        if second_status not in {"created", "queued", "in_progress", "completed"}:
            raise RuntimeError(
                f"candidate blocker {label} has unexpected status: {second_status!r}"
            )
        result.append(
            {
                "label": label,
                "training_task_id": task_id,
                "status": second_status,
            }
        )
    return result, all(item["status"] == "completed" for item in result)


def _find_planned_evaluation(
    task_class: object,
    *,
    project_id: str,
    name: str,
    controller_task_id: str,
) -> object | None:
    query = getattr(task_class, "query_tasks", None)
    getter = getattr(task_class, "get_task", None)
    if not callable(query) or not callable(getter):
        raise RuntimeError(
            "ClearML Task class cannot perform authoritative planned evaluation lookup"
        )
    expected_project_id = _task_id(project_id, "formal evaluation project")
    expected_parent_id = _task_id(
        controller_task_id, "formal evaluation parent controller"
    )
    try:
        candidate_ids = query(
            task_filter={
                "project": [expected_project_id],
                "parent": expected_parent_id,
            }
        )
    except Exception as error:
        raise RuntimeError(
            "failed to query the authoritative planned evaluation task inventory"
        ) from error
    if not isinstance(candidate_ids, Sequence) or isinstance(
        candidate_ids, (str, bytes, bytearray)
    ):
        raise RuntimeError("planned evaluation task inventory is invalid")
    candidates: list[object] = []
    seen_ids: set[str] = set()
    for raw_task_id in candidate_ids:
        task_id = _task_id(raw_task_id, "planned evaluation lookup task")
        if task_id in seen_ids:
            raise RuntimeError("planned evaluation lookup returned a duplicate task ID")
        seen_ids.add(task_id)
        task = getter(task_id=task_id)
        if _task_id(getattr(task, "id", ""), "planned evaluation task") != task_id:
            raise RuntimeError("planned evaluation lookup task identity drifted")
        if (
            _task_parent(task) != expected_parent_id
            or _task_project_id(task, context=f"planned evaluation task {task_id}")
            != expected_project_id
        ):
            raise RuntimeError("planned evaluation lookup escaped its project/parent scope")
        candidates.append(task)
    matches = [
        task
        for task in candidates
        if str(getattr(task, "name", "") or "") == name
    ]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate planned evaluation task: {name}")
    return matches[0] if matches else None


def _evaluation_parameters(
    template: object,
    *,
    training_item: Mapping[str, str],
) -> dict[str, object]:
    subject = training_item["subject"]
    training_task_id = training_item["training_task_id"]
    parameters = dict(_parameters(template))
    forbidden_prefixes = (
        "Args/teacher_",
        "Args/student_",
    )
    for key in list(parameters):
        if key in {
            "Args/experiment_from_task",
            "Args/allow_failed_teacher_task",
            "Args/training_seed",
        } or key.startswith(forbidden_prefixes):
            parameters.pop(key, None)
    parameters.update(
        {
            "Args/stage": "baseline_validate",
            "Args/controlled_baseline": subject,
            "Args/controlled_baseline_task_id": training_task_id,
            "Args/controlled_baseline_model_id": training_item["training_model_id"],
            "Args/controlled_baseline_checkpoint_sha256": training_item[
                "training_checkpoint_sha256"
            ],
            "Args/predecessor_task_id": training_task_id,
            "Args/gpus": "4",
            "Args/max_epochs": "50",
            "Args/amp": "False",
        }
    )
    return parameters


def _set_exact_evaluation_parameters(
    task: object,
    *,
    expected: Mapping[str, object],
    context: str,
) -> None:
    setter = getattr(task, "set_parameters", None)
    if not callable(setter):
        raise RuntimeError(f"{context} cannot set parameters")
    setter(dict(expected))
    observed = _parameters(task)
    if observed != dict(expected):
        raise RuntimeError(f"{context} parameter replacement drifted")


def _require_planned_evaluation_runtime_contract(
    task: object,
    *,
    training_item: Mapping[str, str],
    expected_parameters: Mapping[str, object],
    status: str,
) -> None:
    subject = training_item["subject"]
    observed_parameters = dict(_parameters(task))
    if status != "created":
        for key, expected in EVALUATION_BOOTSTRAP_RUNTIME_DEFAULT_PARAMETERS.items():
            if key not in observed_parameters:
                continue
            if observed_parameters.pop(key) != expected:
                raise RuntimeError(
                    f"planned evaluation {subject} bootstrap parameter {key} drift"
                )
    if observed_parameters != dict(expected_parameters):
        raise RuntimeError(f"planned evaluation {subject} parameter drift")

    inventory = _input_model_inventory(
        task, context=f"planned evaluation {subject}"
    )
    expected_input = {
        "id": training_item["training_model_id"],
        "name": f"ResilientV2X {subject} final checkpoint",
        "task": training_item["training_task_id"],
    }
    if status in {"created", "queued"}:
        allowed = inventory == []
    elif status == "in_progress":
        allowed = inventory in ([], [expected_input])
    elif status == "completed":
        allowed = inventory == [expected_input]
    else:
        raise RuntimeError(
            f"planned evaluation {subject} has unusable status {status!r}"
        )
    if not allowed:
        raise RuntimeError(
            f"planned evaluation {subject} input model contract drifted for "
            f"status {status!r}"
        )


def _input_model_inventory(task: object, *, context: str) -> list[dict[str, str]]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot expose models")
    models = getter()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"{context} returned invalid input models")
    input_models = models.get("input")
    if isinstance(input_models, (str, bytes, bytearray)) or not isinstance(
        input_models, Sequence
    ):
        raise RuntimeError(f"{context} returned invalid input models")
    inventory: list[dict[str, str]] = []
    model_ids: set[str] = set()
    for model in input_models:
        model_id = _task_id(getattr(model, "id", ""), f"{context} input model")
        if model_id in model_ids:
            raise RuntimeError(f"{context} repeats an input model")
        model_ids.add(model_id)
        inventory.append(
            {
                "id": model_id,
                "name": str(getattr(model, "name", "") or ""),
                "task": _task_id(
                    getattr(model, "task", ""), f"{context} input model task"
                ),
            }
        )
    return inventory


def _remove_cloned_template_input_models(task: object, *, subject: str) -> None:
    context = f"new evaluation task {subject}"
    inventory = _input_model_inventory(task, context=context)
    if inventory != list(EXPECTED_EVALUATION_TEMPLATE_INPUT_MODELS):
        raise RuntimeError(f"{context} inherited an unexpected input model inventory")
    remover = getattr(task, "remove_input_models", None)
    if not callable(remover):
        raise RuntimeError(f"{context} cannot remove inherited input models")
    remover([item["id"] for item in inventory])
    if _status(task) != "created":
        raise RuntimeError(f"{context} changed status while removing input models")
    if _input_model_inventory(task, context=context):
        raise RuntimeError(f"{context} retained inherited input models")


def create_or_validate_evaluation_plan(
    *,
    task_class: object,
    controller_task_id: str,
    template_task_id: str,
    expected_project_id: str,
    training: list[dict[str, str]],
    worker_queues: list[str],
    expected_script_sha256: str,
    authoritative_entries: Sequence[Mapping[str, str]] | None = None,
) -> list[dict[str, str]]:
    template = task_class.get_task(task_id=template_task_id)
    project_id = _task_id(expected_project_id, "formal evaluation project")
    if _task_project_id(template, context="evaluation template") != project_id:
        raise RuntimeError("evaluation template project ID mismatch")
    if _status(template) != "completed":
        raise RuntimeError("evaluation template is not completed")
    if (
        _script_sha256(template, context="evaluation template")
        != expected_script_sha256
    ):
        raise RuntimeError("evaluation template script SHA-256 mismatch")
    if _input_model_inventory(template, context="evaluation template") != list(
        EXPECTED_EVALUATION_TEMPLATE_INPUT_MODELS
    ):
        raise RuntimeError("evaluation template input model inventory mismatch")
    authoritative_by_subject: dict[str, Mapping[str, str]] | None = None
    planned_script_sha256 = expected_script_sha256
    if authoritative_entries is not None:
        normalized = _parse_evaluation_plan(_canonical_json(authoritative_entries))
        authoritative_by_subject = {item["subject"]: item for item in normalized}
        planned_script_sha256 = _authoritative_evaluation_script_sha256(template)
    plan: list[dict[str, str]] = []
    for index, training_item in enumerate(training, start=1):
        subject = training_item["subject"]
        name = _planned_evaluation_name(
            controller_task_id=controller_task_id,
            index=index,
            subject=subject,
        )
        authoritative_entry = (
            authoritative_by_subject[subject]
            if authoritative_by_subject is not None
            else None
        )
        if authoritative_entry is not None:
            try:
                task = task_class.get_task(
                    task_id=authoritative_entry["evaluation_task_id"]
                )
            except Exception as error:
                raise RuntimeError(
                    f"authoritative planned evaluation {subject} cannot be read"
                ) from error
            if task is None:
                raise RuntimeError(
                    f"authoritative planned evaluation {subject} is missing"
                )
        else:
            task = _find_planned_evaluation(
                task_class,
                project_id=project_id,
                name=name,
                controller_task_id=controller_task_id,
            )
        expected_parameters = _evaluation_parameters(
            template,
            training_item=training_item,
        )
        if task is None:
            task = task_class.clone(
                source_task=template,
                name=name,
                parent=controller_task_id,
            )
            if _status(task) != "created":
                raise RuntimeError(f"new evaluation task {subject} is not created")
            _remove_cloned_template_input_models(task, subject=subject)
            _set_exact_evaluation_parameters(
                task,
                expected=expected_parameters,
                context=f"evaluation task {subject}",
            )
        else:
            if (
                authoritative_entry is not None
                and _task_id(
                    getattr(task, "id", ""),
                    f"authoritative {subject} evaluation task",
                )
                != authoritative_entry["evaluation_task_id"]
            ):
                raise RuntimeError(
                    f"authoritative planned evaluation {subject} identity drifted"
                )
            status = _status(task)
            if status in FAILED_STATUSES:
                raise RuntimeError(
                    f"planned evaluation {subject} has unusable status {status!r}"
                )
            if status not in {"created", "queued", "in_progress", "completed"}:
                raise RuntimeError(
                    f"planned evaluation {subject} has unusable status {status!r}"
                )
            _require_planned_evaluation_runtime_contract(
                task,
                training_item=training_item,
                expected_parameters=expected_parameters,
                status=status,
            )
        if _task_parent(task) != controller_task_id:
            raise RuntimeError(f"planned evaluation {subject} parent mismatch")
        if str(getattr(task, "name", "") or "") != name:
            raise RuntimeError(f"planned evaluation {subject} name mismatch")
        if _task_project_id(task, context=f"planned evaluation {subject}") != project_id:
            raise RuntimeError(f"planned evaluation {subject} project ID mismatch")
        if _script_sha256(task, context=f"evaluation task {subject}") != (
            planned_script_sha256
        ):
            raise RuntimeError(f"planned evaluation {subject} script SHA mismatch")
        plan.append(
            {
                "subject": subject,
                "evaluation_task_id": _task_id(
                    getattr(task, "id", ""), f"{subject} evaluation task"
                ),
                "queue": (
                    authoritative_entry["queue"]
                    if authoritative_entry is not None
                    else worker_queues[(index - 1) % len(worker_queues)]
                ),
            }
        )
    return plan


def _load_authoritative_evaluation_plan(
    *,
    task_class: object,
    producer_task_id: str,
    expected_seal_sha256: str,
    expected_project_id: str,
    controller_task_id: str,
    provenance_task_id: str,
    template_task_id: str,
) -> list[dict[str, str]]:
    producer_id = _task_id(producer_task_id, "authoritative plan producer")
    if producer_id != AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID:
        raise ValueError("authoritative evaluation plan producer mismatch")
    expected_seal = _sha256(
        expected_seal_sha256, "authoritative evaluation plan seal"
    )
    if expected_seal != AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256:
        raise ValueError("authoritative evaluation plan seal pin mismatch")
    producer = task_class.get_task(task_id=producer_id)
    if (
        _task_id(getattr(producer, "id", ""), "authoritative plan producer")
        != producer_id
        or str(getattr(producer, "name", "") or "")
        != AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_NAME
        or _task_project_id(producer, context="authoritative plan producer")
        != expected_project_id
        or _task_parent(producer) != provenance_task_id
        or _status(producer) != AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_STATUS
        or _standalone_script_sha256(
            producer,
            expected_entry_point="clearml_1337_dependency_watcher.py",
            context="authoritative plan producer",
        )
        != AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_SCRIPT_SHA256
        or _parameters(producer)
        != AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_PARAMETERS
    ):
        raise RuntimeError("authoritative evaluation plan producer identity drifted")
    tags = getattr(producer, "tags", None)
    if tags is None:
        tags = getattr(getattr(producer, "data", None), "tags", None)
    if (
        not isinstance(tags, (list, tuple))
        or len(tags) != len(set(tags))
        or frozenset(tags) != AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TAGS
    ):
        raise RuntimeError("authoritative evaluation plan producer tags drifted")
    artifacts = getattr(producer, "artifacts", None)
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        FORMAL_EVALUATION_PLAN_ARTIFACT
    }:
        raise RuntimeError("authoritative evaluation plan artifact inventory drifted")
    artifact, record, _ = _unique_artifact_metadata(
        producer, FORMAL_EVALUATION_PLAN_ARTIFACT
    )
    hashes = {
        str(value or "")
        for value in (
            _metadata_value(artifact, "hash", "sha256"),
            _metadata_value(record, "hash", "sha256"),
        )
        if value not in {None, ""}
    }
    sizes: set[int] = set()
    for value in (
        _metadata_value(artifact, "content_size", "size", "size_bytes"),
        _metadata_value(record, "content_size", "size", "size_bytes"),
    ):
        if value not in {None, ""}:
            try:
                sizes.add(int(value))
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "authoritative evaluation plan artifact size is invalid"
                ) from error
    if hashes != {AUTHORITATIVE_EVALUATION_PLAN_ARTIFACT_SHA256} or sizes != {
        AUTHORITATIVE_EVALUATION_PLAN_ARTIFACT_BYTES
    }:
        raise RuntimeError("authoritative evaluation plan artifact identity drifted")
    plan = _json_mapping_copy(
        _artifact_preview_mapping(
            artifact, record, FORMAL_EVALUATION_PLAN_ARTIFACT
        ),
        context="authoritative evaluation plan preview",
    )
    preview_bytes = json.dumps(plan, indent=4, sort_keys=True).encode("utf-8")
    if (
        len(preview_bytes) != AUTHORITATIVE_EVALUATION_PLAN_ARTIFACT_BYTES
        or hashlib.sha256(preview_bytes).hexdigest()
        != AUTHORITATIVE_EVALUATION_PLAN_ARTIFACT_SHA256
    ):
        raise RuntimeError("authoritative evaluation plan preview bytes drifted")
    _require_valid_seal(plan, context="authoritative evaluation plan")
    expected = {
        "schema_version": 2,
        "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
        "training_controller_task_id": controller_task_id,
        "training_provenance_task_id": provenance_task_id,
        "evaluation_template_task_id": template_task_id,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "delays_ms": list(EXPECTED_DELAYS_MS),
        "conditions": list(EXPECTED_CONDITIONS),
        "run_count": 12,
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "seal_sha256": expected_seal,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(f"authoritative evaluation plan {key} drifted")
    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("authoritative evaluation plan entries are invalid")
    return _parse_evaluation_plan(_canonical_json(entries))


def _amended_plan_producer_parameters(
    *, revised_plan_seal: str, worker_evidence_seal: str
) -> dict[str, str]:
    return {
        "Args/base_plan_producer_task_id": (
            AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
        ),
        "Args/base_plan_seal_sha256": AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256,
        "Args/evaluation_recovery_receipt_seal_sha256": (
            AMENDED_PLAN_RECOVERY_RECEIPT_SEAL_SHA256
        ),
        "Args/ffnet_attempt_evidence_seal_sha256": (
            AMENDED_PLAN_ATTEMPT_EVIDENCE_SEAL_SHA256
        ),
        "Args/ffnet_original_worker_id": AMENDED_PLAN_FFNET_ORIGINAL_WORKER,
        "Args/successor_deployment_receipt_seal_sha256": (
            AMENDED_PLAN_DEPLOYMENT_RECEIPT_SEAL_SHA256
        ),
        "Args/subject": "ffnet",
        "Args/evaluation_task_id": AMENDED_PLAN_FFNET_TASK_ID,
        "Args/from_queue": AMENDED_PLAN_FFNET_OLD_QUEUE,
        "Args/to_queue": AMENDED_PLAN_FFNET_NEW_QUEUE,
        "Args/revised_plan_seal_sha256": revised_plan_seal,
        "Args/worker_evidence_seal_sha256": worker_evidence_seal,
    }


def _validate_amended_worker_evidence(
    value: object, *, expected_seal_sha256: str
) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError("amended plan worker evidence is not an object")
    evidence = _json_mapping_copy(value, context="amended plan worker evidence")
    _require_valid_seal(evidence, context="amended plan worker evidence")
    if evidence.get("seal_sha256") != expected_seal_sha256:
        raise RuntimeError("amended plan worker evidence seal drifted")
    expected_keys = {
        "captured_at_utc",
        "formal_evaluation_task_ids",
        "gpu4_v100",
        "gpu4_a100",
        "gpu_telemetry",
        "gpu_telemetry_policy",
        "seal_sha256",
    }
    if set(evidence) != expected_keys:
        raise RuntimeError("amended plan worker evidence inventory drifted")
    task_ids = evidence.get("formal_evaluation_task_ids")
    if (
        not isinstance(task_ids, list)
        or len(task_ids) != 28
        or len(set(task_ids)) != 28
        or AMENDED_PLAN_FFNET_TASK_ID not in task_ids
    ):
        raise RuntimeError("amended plan worker task inventory drifted")
    a100 = evidence.get("gpu4_a100")
    if (
        not isinstance(a100, Mapping)
        or a100.get("target_queue_id")
        != EXPECTED_QUEUE_IDS[AMENDED_PLAN_FFNET_NEW_QUEUE]
        or a100.get("ready") is not True
        or a100.get("selected_idle_target_worker_id")
        != AMENDED_PLAN_A100_TARGET_WORKER
    ):
        raise RuntimeError("amended plan A100 resource snapshot drifted")
    telemetry = evidence.get("gpu_telemetry")
    if not isinstance(telemetry, Mapping):
        raise RuntimeError("amended plan GPU telemetry is invalid")

    def maximum(worker_id: str, key: str) -> float:
        record = telemetry.get(worker_id)
        if not isinstance(record, Mapping) or record.get("worker_id") != worker_id:
            raise RuntimeError("amended plan GPU telemetry worker drifted")
        raw = record.get(key)
        if type(raw) not in {int, float}:
            raise RuntimeError("amended plan GPU telemetry value is invalid")
        result = float(raw)
        if not math.isfinite(result) or result < 0.0:
            raise RuntimeError("amended plan GPU telemetry value is invalid")
        return result

    if maximum(
        AMENDED_PLAN_FFNET_ORIGINAL_WORKER, "max_gpu_memory_used_mib"
    ) <= AMENDED_PLAN_GPU_MEMORY_IDLE_LIMIT_MIB:
        raise RuntimeError("amended plan original FFNet V100 worker was not busy")
    if (
        maximum(AMENDED_PLAN_A100_TARGET_WORKER, "max_gpu_memory_used_mib")
        > AMENDED_PLAN_GPU_MEMORY_IDLE_LIMIT_MIB
        or maximum(AMENDED_PLAN_A100_TARGET_WORKER, "max_gpu_usage_percent")
        > AMENDED_PLAN_GPU_USAGE_IDLE_LIMIT_PERCENT
    ):
        raise RuntimeError("amended plan A100 target worker was not idle")


def _load_amended_evaluation_plan(
    *,
    task_class: object,
    producer_task_id: str,
    expected_plan_seal_sha256: str,
    expected_receipt_seal_sha256: str,
    expected_producer_source_sha256: str,
    expected_amendment_evidence_seal_sha256: str,
    expected_worker_evidence_seal_sha256: str,
    expected_task_ids_sha256: str,
    expected_project_id: str,
    controller_task_id: str,
    provenance_task_id: str,
    template_task_id: str,
) -> list[dict[str, str]]:
    producer_id = _task_id(producer_task_id, "amended plan producer")
    if producer_id == AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID:
        raise ValueError("amended plan producer must be independent from the base plan")
    plan_seal = _sha256(expected_plan_seal_sha256, "amended evaluation plan seal")
    receipt_seal = _sha256(
        expected_receipt_seal_sha256, "amended plan receipt seal"
    )
    producer_source_sha = _sha256(
        expected_producer_source_sha256, "amended plan producer source"
    )
    evidence_seal = _sha256(
        expected_amendment_evidence_seal_sha256,
        "amended plan evidence seal",
    )
    worker_seal = _sha256(
        expected_worker_evidence_seal_sha256, "amended worker evidence seal"
    )
    task_ids_seal = _sha256(
        expected_task_ids_sha256, "amended evaluation task inventory"
    )
    producer = task_class.get_task(task_id=producer_id)
    if (
        _task_id(getattr(producer, "id", ""), "amended plan producer")
        != producer_id
        or str(getattr(producer, "name", "") or "")
        != AMENDED_EVALUATION_PLAN_PRODUCER_NAME
        or _task_project_id(producer, context="amended plan producer")
        != expected_project_id
        or _task_parent(producer) != AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
        or _status(producer) != "completed"
        or _standalone_script_sha256(
            producer,
            expected_entry_point=AMENDED_EVALUATION_PLAN_PRODUCER_ENTRY_POINT,
            context="amended plan producer",
        )
        != producer_source_sha
        or _parameters(producer)
        != _amended_plan_producer_parameters(
            revised_plan_seal=plan_seal,
            worker_evidence_seal=worker_seal,
        )
    ):
        raise RuntimeError("amended evaluation plan producer identity drifted")
    tags = getattr(producer, "tags", None)
    if tags is None:
        tags = getattr(getattr(producer, "data", None), "tags", None)
    if (
        not isinstance(tags, (list, tuple))
        or len(tags) != len(set(tags))
        or frozenset(tags) != AMENDED_EVALUATION_PLAN_PRODUCER_TAGS
    ):
        raise RuntimeError("amended evaluation plan producer tags drifted")
    artifacts = getattr(producer, "artifacts", None)
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        FORMAL_EVALUATION_PLAN_ARTIFACT,
        AMENDED_EVALUATION_PLAN_ARTIFACT,
    }:
        raise RuntimeError("amended evaluation plan artifact inventory drifted")
    plan = _stable_artifact_mapping(
        producer, FORMAL_EVALUATION_PLAN_ARTIFACT, required_status="completed"
    )
    amendment = _stable_artifact_mapping(
        producer, AMENDED_EVALUATION_PLAN_ARTIFACT, required_status="completed"
    )
    _require_valid_seal(plan, context="amended formal evaluation plan")
    _require_valid_seal(amendment, context="formal evaluation plan amendment")
    if plan.get("seal_sha256") != plan_seal or amendment.get(
        "seal_sha256"
    ) != evidence_seal:
        raise RuntimeError("amended evaluation plan artifact seal drifted")
    expected_plan = {
        "schema_version": 2,
        "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
        "training_controller_task_id": controller_task_id,
        "training_provenance_task_id": provenance_task_id,
        "evaluation_template_task_id": template_task_id,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "delays_ms": list(EXPECTED_DELAYS_MS),
        "conditions": list(EXPECTED_CONDITIONS),
        "run_count": 12,
        "subject_order": list(FORMAL_SUBJECT_ORDER),
    }
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise RuntimeError(f"amended evaluation plan {key} drifted")
    entries_raw = plan.get("entries")
    if not isinstance(entries_raw, list):
        raise RuntimeError("amended evaluation plan entries are invalid")
    entries = _parse_evaluation_plan(_canonical_json(entries_raw))
    base_entries = _load_authoritative_evaluation_plan(
        task_class=task_class,
        producer_task_id=AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID,
        expected_seal_sha256=AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256,
        expected_project_id=expected_project_id,
        controller_task_id=controller_task_id,
        provenance_task_id=provenance_task_id,
        template_task_id=template_task_id,
    )
    for base, revised in zip(base_entries, entries, strict=True):
        expected = dict(base)
        if base["subject"] == "ffnet":
            if (
                base["evaluation_task_id"] != AMENDED_PLAN_FFNET_TASK_ID
                or base["queue"] != AMENDED_PLAN_FFNET_OLD_QUEUE
            ):
                raise RuntimeError("base FFNet evaluation identity drifted")
            expected["queue"] = AMENDED_PLAN_FFNET_NEW_QUEUE
        if revised != expected:
            raise RuntimeError(
                "amended plan changed more than the FFNet evaluation queue"
            )
    observed_task_ids_seal = hashlib.sha256(
        _canonical_json([row["evaluation_task_id"] for row in entries]).encode(
            "utf-8"
        )
    ).hexdigest()
    if observed_task_ids_seal != task_ids_seal:
        raise RuntimeError("amended evaluation task identity inventory drifted")
    expected_change = {
        "base_plan_seal_sha256": AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256,
        "revised_plan_seal_sha256": plan_seal,
        "changed_json_pointer": (
            f"/entries/{FORMAL_SUBJECT_ORDER.index('ffnet')}/queue"
        ),
        "subject": "ffnet",
        "evaluation_task_id": AMENDED_PLAN_FFNET_TASK_ID,
        "from_queue": AMENDED_PLAN_FFNET_OLD_QUEUE,
        "to_queue": AMENDED_PLAN_FFNET_NEW_QUEUE,
        "evaluation_task_ids_sha256": task_ids_seal,
        "fixed_baselines": list(AMENDED_PLAN_FIXED_BASELINES),
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "changed_field_count": 1,
    }
    if (
        amendment.get("schema_version") != 1
        or amendment.get("evidence_type")
        != "resilient_v2x_formal_1337_evaluation_plan_amendment"
        or amendment.get("producer_task_id") != producer_id
        or amendment.get("base_authority")
        != {
            "producer_task_id": AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID,
            "artifact_name": FORMAL_EVALUATION_PLAN_ARTIFACT,
            "plan_seal_sha256": AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256,
        }
        or amendment.get("amendment") != expected_change
    ):
        raise RuntimeError("formal evaluation plan amendment contract drifted")
    authorization = amendment.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("evaluation_recovery_receipt_seal_sha256")
        != AMENDED_PLAN_RECOVERY_RECEIPT_SEAL_SHA256
        or authorization.get("successor_deployment_receipt_seal_sha256")
        != AMENDED_PLAN_DEPLOYMENT_RECEIPT_SEAL_SHA256
        or authorization.get("successor_task_ids")
        != AMENDED_PLAN_PRIOR_SUCCESSOR_TASK_IDS
        or not isinstance(authorization.get("ffnet_attempt_evidence"), Mapping)
        or authorization["ffnet_attempt_evidence"].get("receipt_seal_sha256")
        != AMENDED_PLAN_ATTEMPT_EVIDENCE_SEAL_SHA256
        or authorization["ffnet_attempt_evidence"].get("task_id")
        != AMENDED_PLAN_FFNET_TASK_ID
        or authorization["ffnet_attempt_evidence"].get("last_worker")
        != AMENDED_PLAN_FFNET_ORIGINAL_WORKER
    ):
        raise RuntimeError("formal evaluation plan amendment authority drifted")
    _validate_amended_worker_evidence(
        amendment.get("worker_stats_evidence"),
        expected_seal_sha256=worker_seal,
    )
    invariants = amendment.get("invariants")
    if not isinstance(invariants, Mapping) or invariants != {
        "one_field_amendment": True,
        "evaluation_task_ids_unchanged": True,
        "evaluation_parameters_unchanged": True,
        "evaluation_sources_unchanged": True,
        "evaluation_models_unchanged": True,
        "protocol_and_sample_count_unchanged": True,
        "fixed_five_baselines_unchanged": True,
        "replacement_tasks_created": False,
    }:
        raise RuntimeError("formal evaluation plan amendment invariants drifted")
    # The receipt itself is a local write-once authority. Its content is bound
    # into every deployed successor parameter; W cannot read a local path, but
    # it refuses execution unless the independently supplied 64-hex pin exists.
    if not receipt_seal:
        raise RuntimeError("amended evaluation plan receipt seal is missing")
    return entries


def _publish_evaluation_plan(
    task: object,
    *,
    controller_task_id: str,
    template_task_id: str,
    entries: list[dict[str, str]],
    provenance_policies: TrainingProvenancePolicies,
) -> None:
    if list(provenance_policies) != list(FORMAL_SUBJECT_ORDER):
        raise RuntimeError("formal training provenance subject inventory drifted")
    source_equivalence = _json_mapping_copy(
        provenance_policies.source_revision_equivalence,
        context="evaluation plan source revision equivalence",
    )
    _require_valid_seal(
        source_equivalence,
        context="evaluation plan source revision equivalence",
    )
    if source_equivalence.get("seal_sha256") != (
        EXPECTED_SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
    ):
        raise RuntimeError("evaluation plan source revision equivalence mismatch")
    source_subject_map = _json_mapping_copy(
        provenance_policies.source_revision_subject_map,
        context="evaluation plan source revision subject map",
    )
    _require_valid_seal(
        source_subject_map,
        context="evaluation plan source revision subject map",
    )
    if source_subject_map.get("seal_sha256") != (
        EXPECTED_SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
    ):
        raise RuntimeError("evaluation plan source revision subject map mismatch")
    source_revisions = source_equivalence.get("source_revisions")
    if not isinstance(source_revisions, Mapping) or not isinstance(
        source_revisions.get(EXPECTED_SOURCE_TREE_SHA256), Mapping
    ):
        raise RuntimeError("evaluation plan old source certificate is missing")
    payload = {
        "schema_version": 2,
        "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
        "training_controller_task_id": controller_task_id,
        "training_provenance_task_id": provenance_policies.provenance_task_id,
        "training_provenance_seal_sha256": (
            provenance_policies.provenance_seal_sha256
        ),
        "source_revision_equivalence": source_equivalence,
        "source_revision_equivalence_seal_sha256": (
            provenance_policies.source_revision_equivalence_seal_sha256
        ),
        "source_revision_subject_map": source_subject_map,
        "source_revision_subject_map_seal_sha256": (
            provenance_policies.source_revision_subject_map_seal_sha256
        ),
        "evaluation_source_revision_tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
        "evaluation_source_revision": dict(
            source_revisions[EXPECTED_SOURCE_TREE_SHA256]
        ),
        "evaluation_template_task_id": template_task_id,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "delays_ms": list(EXPECTED_DELAYS_MS),
        "conditions": list(EXPECTED_CONDITIONS),
        "run_count": 12,
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "entries": entries,
    }
    payload["seal_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, Mapping) and FORMAL_EVALUATION_PLAN_ARTIFACT in artifacts:
        observed = _artifact_mapping(
            artifacts[FORMAL_EVALUATION_PLAN_ARTIFACT],
            FORMAL_EVALUATION_PLAN_ARTIFACT,
        )
        if observed != payload:
            raise RuntimeError("existing formal evaluation plan artifact drifted")
        return
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader) or not uploader(
        FORMAL_EVALUATION_PLAN_ARTIFACT,
        artifact_object=payload,
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to publish the formal evaluation plan")
    flusher = getattr(task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)


def _metadata_value(metadata: object, *names: str) -> object | None:
    for name in names:
        if isinstance(metadata, Mapping) and name in metadata:
            return metadata[name]
        value = getattr(metadata, name, None)
        if value is not None:
            return value
    return None


def _reasonable_remote_url(value: object, context: str) -> str:
    url = str(value or "")
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https", "s3", "gs", "azure"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path
    ):
        raise RuntimeError(f"{context} URL metadata is invalid")
    return url


def _metadata_origins(metadata: object, context: str) -> set[tuple[str, str]]:
    origins: set[tuple[str, str]] = set()
    url = _metadata_value(metadata, "uri", "url", "artifact_uri")
    if url not in {None, ""}:
        parsed = urlparse(_reasonable_remote_url(url, context))
        if parsed.scheme in {"http", "https"}:
            origins.add((parsed.scheme, parsed.netloc))
    size = _metadata_value(metadata, "content_size", "size", "size_bytes")
    if size not in {None, ""}:
        try:
            parsed_size = int(size)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{context} size metadata is invalid") from error
        if isinstance(size, bool) or parsed_size <= 0:
            raise RuntimeError(f"{context} size metadata is invalid")
    artifact_hash = _metadata_value(metadata, "hash", "sha256")
    if (
        artifact_hash not in {None, ""}
        and ARTIFACT_HASH_PATTERN.fullmatch(str(artifact_hash)) is None
    ):
        raise RuntimeError(f"{context} hash metadata is invalid")
    return origins


def _unique_artifact_metadata(
    task: object, name: str
) -> tuple[object, object, set[tuple[str, str]]]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("ClearML task does not expose artifact metadata")
    matching_proxies = [artifact for key, artifact in artifacts.items() if key == name]
    if len(matching_proxies) != 1:
        raise RuntimeError(f"ClearML task lacks {name}")
    execution = getattr(getattr(task, "data", None), "execution", None)
    artifact_records = getattr(execution, "artifacts", None)
    if not isinstance(artifact_records, (list, tuple)):
        raise RuntimeError("ClearML task does not expose execution artifact metadata")
    matching_records = [
        record
        for record in artifact_records
        if str(_metadata_value(record, "key", "name") or "") == name
    ]
    if len(matching_records) != 1:
        raise RuntimeError(f"ClearML task must expose exactly one {name} artifact")
    context = f"ClearML artifact {name}"
    origins = _metadata_origins(matching_proxies[0], context)
    origins.update(_metadata_origins(matching_records[0], context))
    return matching_proxies[0], matching_records[0], origins


def _artifact_preview_mapping(
    artifact: object,
    record: object,
    name: str,
) -> Mapping[str, object]:
    type_data = _metadata_value(record, "type_data")
    preview_values = [
        value
        for value in (
            _metadata_value(artifact, "preview"),
            _metadata_value(type_data, "preview") if type_data is not None else None,
        )
        if value not in {None, ""}
    ]
    if not preview_values:
        raise RuntimeError(f"ClearML artifact {name} preview metadata is missing")
    parsed_values: list[Mapping[str, object]] = []
    for value in preview_values:
        if not isinstance(value, str):
            raise RuntimeError(f"ClearML artifact {name} preview metadata is invalid")
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"ClearML artifact {name} preview metadata is invalid"
            ) from error
        if not isinstance(parsed, Mapping):
            raise RuntimeError(f"ClearML artifact {name} preview is not a JSON object")
        parsed_values.append(parsed)
    if any(value != parsed_values[0] for value in parsed_values[1:]):
        raise RuntimeError(f"ClearML artifact {name} preview metadata mismatch")
    return parsed_values[0]


def _artifact_path_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_json_artifact_path(value: object, *, name: str) -> dict[str, object]:
    try:
        path = Path(os.fsdecode(os.fspath(value)))
    except (TypeError, ValueError, OSError) as error:
        raise RuntimeError(
            f"ClearML artifact {name} is not a JSON object or valid local path"
        ) from error
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError(f"ClearML artifact {name} secure local open is unavailable")
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
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
                or opened.st_size <= 0
                or opened.st_size > MAX_JSON_ARTIFACT_BYTES
            ):
                raise RuntimeError(f"ClearML artifact {name} local path is unsafe")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, MAX_JSON_ARTIFACT_BYTES + 1 - total),
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_JSON_ARTIFACT_BYTES:
                    raise RuntimeError(
                        f"ClearML artifact {name} exceeded its byte cap"
                    )
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final = path.lstat()
    except RuntimeError:
        raise
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"ClearML artifact {name} local path cannot be securely read"
        ) from error
    if (
        _artifact_path_signature(before) != _artifact_path_signature(opened)
        or _artifact_path_signature(opened) != _artifact_path_signature(after)
        or _artifact_path_signature(after) != _artifact_path_signature(final)
        or total != opened.st_size
    ):
        raise RuntimeError(f"ClearML artifact {name} changed during local read")
    try:
        document = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"ClearML artifact {name} is not UTF-8 JSON") from error
    return _json_mapping_copy(document, context=f"ClearML artifact {name}")


def _artifact_mapping(artifact: object, name: str) -> Mapping[str, object]:
    getter = getattr(artifact, "get", None)
    if not callable(getter):
        raise RuntimeError(f"ClearML artifact {name} cannot be read")
    try:
        result = getter(force_download=True)
    except TypeError:
        result = getter()
    if isinstance(result, Mapping):
        return _json_mapping_copy(result, context=f"ClearML artifact {name}")
    return _read_json_artifact_path(result, name=name)


def _json_mapping_copy(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} is not a JSON object")
    try:
        copied = json.loads(_canonical_json(value))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} is not canonical JSON") from error
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise RuntimeError(f"{context} is not a JSON object")
    return copied


def _artifact_metadata_signature(task: object, name: str) -> str:
    proxy, record, _ = _unique_artifact_metadata(task, name)
    type_data = _metadata_value(record, "type_data")
    signature = {
        "proxy": {
            "uri": _metadata_value(proxy, "uri", "url", "artifact_uri"),
            "size": _metadata_value(proxy, "content_size", "size", "size_bytes"),
            "hash": _metadata_value(proxy, "hash", "sha256"),
            "preview": _metadata_value(proxy, "preview"),
        },
        "record": {
            "key": _metadata_value(record, "key", "name"),
            "uri": _metadata_value(record, "uri", "url", "artifact_uri"),
            "size": _metadata_value(record, "content_size", "size", "size_bytes"),
            "hash": _metadata_value(record, "hash", "sha256"),
            "preview": (
                _metadata_value(type_data, "preview") if type_data is not None else None
            ),
        },
    }
    return _canonical_json(signature)


def _fresh_artifact_mapping(task: object, name: str) -> dict[str, object]:
    artifact, _, _ = _unique_artifact_metadata(task, name)
    getter = getattr(artifact, "get", None)
    if not callable(getter):
        raise RuntimeError(f"ClearML artifact {name} cannot be read")
    try:
        value = getter(force_download=True)
    except TypeError:
        value = getter()
    return _json_mapping_copy(value, context=f"ClearML artifact {name}")


def _stable_artifact_mapping(
    task: object,
    name: str,
    *,
    required_status: str,
) -> dict[str, object]:
    if _status(task) != required_status:
        raise RuntimeError(f"artifact owner for {name} is not {required_status}")
    first_metadata = _artifact_metadata_signature(task, name)
    first = _fresh_artifact_mapping(task, name)
    if _status(task) != required_status:
        raise RuntimeError(f"artifact owner for {name} changed status")
    second_metadata = _artifact_metadata_signature(task, name)
    second = _fresh_artifact_mapping(task, name)
    if first_metadata != second_metadata or first != second:
        raise RuntimeError(f"ClearML artifact {name} changed across fresh readbacks")
    if _status(task) != required_status:
        raise RuntimeError(f"artifact owner for {name} changed status")
    return first


def _fresh_artifact_preview_mapping(
    task: object, name: str
) -> dict[str, object]:
    artifact, record, _ = _unique_artifact_metadata(task, name)
    raw_values = [
        value
        for value in (
            _metadata_value(artifact, "preview"),
            _metadata_value(_metadata_value(record, "type_data"), "preview"),
        )
        if value not in {None, ""}
    ]
    if not raw_values or any(not isinstance(value, str) for value in raw_values):
        raise RuntimeError(f"ClearML artifact {name} preview metadata is invalid")
    if any(value != raw_values[0] for value in raw_values[1:]):
        raise RuntimeError(f"ClearML artifact {name} preview metadata mismatch")
    raw = raw_values[0].encode("utf-8")
    sizes = {
        int(value)
        for value in (
            _metadata_value(artifact, "content_size", "size", "size_bytes"),
            _metadata_value(record, "content_size", "size", "size_bytes"),
        )
        if value not in {None, ""}
    }
    hashes = {
        str(value)
        for value in (
            _metadata_value(artifact, "hash", "sha256"),
            _metadata_value(record, "hash", "sha256"),
        )
        if value not in {None, ""}
    }
    if sizes != {len(raw)}:
        raise RuntimeError(f"ClearML artifact {name} preview size mismatch")
    if hashes != {hashlib.sha256(raw).hexdigest()}:
        raise RuntimeError(f"ClearML artifact {name} preview SHA-256 mismatch")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"ClearML artifact {name} preview is not UTF-8 JSON"
        ) from error
    return _json_mapping_copy(parsed, context=f"ClearML artifact {name} preview")


def _stable_artifact_preview_mapping(
    task: object,
    name: str,
    *,
    required_status: str,
) -> dict[str, object]:
    if _status(task) != required_status:
        raise RuntimeError(f"artifact owner for {name} is not {required_status}")
    first_metadata = _artifact_metadata_signature(task, name)
    first = _fresh_artifact_preview_mapping(task, name)
    if _status(task) != required_status:
        raise RuntimeError(f"artifact owner for {name} changed status")
    second_metadata = _artifact_metadata_signature(task, name)
    second = _fresh_artifact_preview_mapping(task, name)
    if first_metadata != second_metadata or first != second:
        raise RuntimeError(f"ClearML artifact {name} changed across fresh readbacks")
    if _status(task) != required_status:
        raise RuntimeError(f"artifact owner for {name} changed status")
    return first


def _authenticated_artifact_bytes_mapping(
    task: object,
    name: str,
    *,
    downloader: Callable[[object, str], bytes],
) -> dict[str, object]:
    artifact, record, _ = _unique_artifact_metadata(task, name)
    value = downloader(task, name)
    if not isinstance(value, bytes):
        raise RuntimeError(
            f"ClearML artifact {name} authenticated download is not bytes"
        )
    sizes = {
        int(item)
        for item in (
            _metadata_value(artifact, "content_size", "size", "size_bytes"),
            _metadata_value(record, "content_size", "size", "size_bytes"),
        )
        if item not in {None, ""}
    }
    hashes = {
        str(item)
        for item in (
            _metadata_value(artifact, "hash", "sha256"),
            _metadata_value(record, "hash", "sha256"),
        )
        if item not in {None, ""}
    }
    if sizes != {len(value)}:
        raise RuntimeError(f"ClearML artifact {name} authenticated size mismatch")
    if hashes != {hashlib.sha256(value).hexdigest()}:
        raise RuntimeError(f"ClearML artifact {name} authenticated SHA-256 mismatch")
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"ClearML artifact {name} authenticated bytes are not UTF-8 JSON"
        ) from error
    return _json_mapping_copy(
        parsed, context=f"ClearML artifact {name} authenticated bytes"
    )


def _stable_authenticated_artifact_mapping(
    task: object,
    name: str,
    *,
    required_status: str,
    downloader: Callable[[object, str], bytes],
) -> dict[str, object]:
    if _status(task) != required_status:
        raise RuntimeError(f"artifact owner for {name} is not {required_status}")
    first_metadata = _artifact_metadata_signature(task, name)
    value = _authenticated_artifact_bytes_mapping(
        task, name, downloader=downloader
    )
    if _status(task) != required_status:
        raise RuntimeError(f"artifact owner for {name} changed status")
    second_metadata = _artifact_metadata_signature(task, name)
    if first_metadata != second_metadata:
        raise RuntimeError(f"ClearML artifact {name} changed across fresh readbacks")
    if _status(task) != required_status:
        raise RuntimeError(f"artifact owner for {name} changed status")
    return value


def _standalone_script_sha256(
    task: object,
    *,
    expected_entry_point: str,
    context: str,
) -> str:
    script = getattr(getattr(task, "data", None), "script", None)
    if (
        str(getattr(script, "repository", "") or "") != ""
        or str(getattr(script, "working_dir", "") or "") != "."
        or str(getattr(script, "entry_point", "") or "") != expected_entry_point
    ):
        raise RuntimeError(f"{context} script identity mismatch")
    source = str(getattr(script, "diff", "") or "")
    if not source:
        raise RuntimeError(f"{context} has an empty standalone script")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _normalized_task_script_identity_sha256(task: object) -> str:
    script = getattr(getattr(task, "data", None), "script", None)
    if script is None:
        raise RuntimeError("training task does not expose script metadata")
    identity = {
        key: (
            script.get(key)
            if isinstance(script, Mapping)
            else getattr(script, key, None)
        )
        for key in SCRIPT_IDENTITY_KEYS
    }
    try:
        canonical = _canonical_json(identity)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "training task script identity is not canonical JSON"
        ) from error
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_provenance_tags(task: object) -> frozenset[str]:
    tags = getattr(task, "tags", None)
    if tags is None:
        tags = getattr(getattr(task, "data", None), "tags", None)
    if not isinstance(tags, (list, tuple)) or any(
        type(value) is not str for value in tags
    ):
        raise RuntimeError("formal training provenance task tags are invalid")
    copied = list(tags)
    unique = frozenset(copied)
    if len(unique) != len(copied):
        raise RuntimeError("formal training provenance task has duplicate tags")
    if unique != frozenset(EXPECTED_PROVENANCE_TAGS):
        raise RuntimeError("formal training provenance task tags mismatch")
    return unique


def _validate_training_recovery_lineage(
    value: object,
    *,
    subject: str,
    controller_task_id: str,
    parent_controller_task_id: str,
    parent_binding: str,
    current_progress_seal_sha256: str,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"formal training provenance {subject} lineage is invalid")
    lineage = [_json_mapping_copy(item, context=f"{subject} lineage") for item in value]
    expected_controller = controller_task_id
    seen_controllers: set[str] = set()
    for index, item in enumerate(lineage):
        controller_id = _task_id(
            item.get("controller_task_id"), f"{subject} lineage controller"
        )
        if controller_id != expected_controller or controller_id in seen_controllers:
            raise RuntimeError(
                f"formal training provenance {subject} lineage chain mismatch"
            )
        seen_controllers.add(controller_id)
        seal = _sha256(
            item.get("progress_seal_sha256"), f"{subject} lineage progress seal"
        )
        if index == 0 and seal != current_progress_seal_sha256:
            raise RuntimeError(
                f"formal training provenance {subject} root lineage seal mismatch"
            )
        decision = str(item.get("decision") or "")
        terminal = index == len(lineage) - 1
        if decision == "adopted_from_source_progress":
            if terminal or set(item) != {
                "controller_task_id",
                "decision",
                "progress_seal_sha256",
                "source_controller_task_id",
            }:
                raise RuntimeError(
                    f"formal training provenance {subject} lineage adoption invalid"
                )
            source_id = _task_id(
                item.get("source_controller_task_id"),
                f"{subject} lineage source controller",
            )
            if source_id == controller_id:
                raise RuntimeError(
                    f"formal training provenance {subject} lineage cycle"
                )
            expected_controller = source_id
            continue
        if not terminal:
            raise RuntimeError(
                f"formal training provenance {subject} lineage terminates early"
            )
        if decision == "controller_created":
            if (
                set(item)
                != {
                    "controller_task_id",
                    "decision",
                    "progress_seal_sha256",
                }
                or parent_controller_task_id != controller_id
            ):
                raise RuntimeError(
                    f"formal training provenance {subject} creator lineage mismatch"
                )
            expected_binding = (
                "current_controller_created"
                if controller_id == controller_task_id
                else "recursive_source_controller_created"
            )
            if parent_binding != expected_binding:
                raise RuntimeError(
                    f"formal training provenance {subject} creator binding mismatch"
                )
        elif decision in {
            "recovery_target_adoptions",
            "recovered_pending_target_children",
        }:
            if (
                set(item)
                != {
                    "controller_task_id",
                    "decision",
                    "progress_seal_sha256",
                    "parent_controller_task_id",
                }
                or _task_id(
                    item.get("parent_controller_task_id"),
                    f"{subject} lineage observed parent",
                )
                != parent_controller_task_id
            ):
                raise RuntimeError(
                    f"formal training provenance {subject} observed parent mismatch"
                )
            if parent_binding != decision:
                raise RuntimeError(
                    f"formal training provenance {subject} observed binding mismatch"
                )
        else:
            raise RuntimeError(
                f"formal training provenance {subject} lineage decision is unknown"
            )
    return lineage


def _validate_training_provenance_document(
    document: Mapping[str, object],
    *,
    provenance_task_id: str,
    controller_task_id: str,
    manifest: Mapping[str, object],
    training: list[dict[str, str]],
) -> dict[str, dict[str, object]]:
    _require_valid_seal(document, context="formal training provenance")
    exact_top_level_fields = {
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
    }
    if set(document) != exact_top_level_fields:
        raise RuntimeError("formal training provenance field inventory mismatch")
    expected = {
        "schema_version": 2,
        "document_type": "resilient_v2x_formal_training_provenance_equivalence",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "passed": True,
        "controller_task_id": controller_task_id,
        "controller_status": "completed",
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "subject_count": len(FORMAL_SUBJECT_ORDER),
        "all_training_tasks_completed": True,
        "authoritative_metadata_read": "single_batch_per_snapshot",
        "progress_artifact": FORMAL_TRAINING_PROGRESS_ARTIFACT,
        "formal_training_manifest_artifact": FORMAL_TRAINING_MANIFEST_ARTIFACT,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(f"formal training provenance {key} mismatch")
    manifest_seal = _sha256(
        manifest.get("seal_sha256"), "formal training manifest seal"
    )
    if document.get("formal_training_manifest_seal_sha256") != manifest_seal:
        raise RuntimeError("formal training provenance manifest seal mismatch")
    manifest_content_sha = hashlib.sha256(
        _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    if document.get("formal_training_manifest_content_sha256") != manifest_content_sha:
        raise RuntimeError("formal training provenance manifest content mismatch")
    for key in (
        "controller_raw_authority_sha256",
        "progress_seal_sha256",
        "progress_content_sha256",
        "recovery_contract_sha256",
    ):
        _sha256(document.get(key), f"formal training provenance {key}")

    source_equivalence = _json_mapping_copy(
        document.get("source_revision_equivalence"),
        context="formal training provenance source revision equivalence",
    )
    _require_valid_seal(
        source_equivalence,
        context="formal training provenance source revision equivalence",
    )
    source_equivalence_seal = _sha256(
        source_equivalence.get("seal_sha256"),
        "formal training provenance source revision equivalence seal",
    )
    if (
        source_equivalence_seal
        != EXPECTED_SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        or document.get("source_revision_equivalence_seal_sha256")
        != source_equivalence_seal
    ):
        raise RuntimeError(
            "formal training provenance source revision equivalence mismatch"
        )
    source_subject_map = _json_mapping_copy(
        document.get("source_revision_subject_map"),
        context="formal training provenance source revision subject map",
    )
    _require_valid_seal(
        source_subject_map,
        context="formal training provenance source revision subject map",
    )
    source_subject_map_seal = _sha256(
        source_subject_map.get("seal_sha256"),
        "formal training provenance source revision subject map seal",
    )
    if (
        source_subject_map_seal
        != EXPECTED_SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        or document.get("source_revision_subject_map_seal_sha256")
        != source_subject_map_seal
    ):
        raise RuntimeError(
            "formal training provenance source revision subject map mismatch"
        )

    progress_chain = document.get("recursive_progress_chain")
    if not isinstance(progress_chain, list) or not progress_chain:
        raise RuntimeError("formal training provenance progress chain is invalid")
    progress_chain_by_id: dict[str, dict[str, object]] = {}
    for index, raw_record in enumerate(progress_chain):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "task_id",
            "role",
            "progress_artifact",
            "progress_revision",
            "progress_seal_sha256",
            "progress_content_sha256",
            "template_script_identity_sha256",
        }:
            raise RuntimeError(
                "formal training provenance progress chain entry is invalid"
            )
        record = _json_mapping_copy(raw_record, context="recursive progress chain")
        task_id = _task_id(record.get("task_id"), "recursive progress controller")
        if task_id in progress_chain_by_id:
            raise RuntimeError(
                "formal training provenance progress chain repeats a controller"
            )
        expected_role = (
            "final_controller"
            if index == 0
            else ("recursive_recovery_source_controller")
        )
        if (
            record.get("role") != expected_role
            or record.get("progress_artifact") != FORMAL_TRAINING_PROGRESS_ARTIFACT
            or type(record.get("progress_revision")) is not int
            or record["progress_revision"] < 1
        ):
            raise RuntimeError(
                "formal training provenance progress chain contract mismatch"
            )
        for key in (
            "progress_seal_sha256",
            "progress_content_sha256",
            "template_script_identity_sha256",
        ):
            _sha256(record.get(key), f"recursive progress chain {key}")
        if index == 0 and (
            task_id != controller_task_id
            or record.get("progress_seal_sha256") != document["progress_seal_sha256"]
            or record.get("progress_content_sha256")
            != document["progress_content_sha256"]
        ):
            raise RuntimeError(
                "formal training provenance root progress chain mismatch"
            )
        progress_chain_by_id[task_id] = record

    bootstrap = document.get("bootstrap_equivalence")
    if not isinstance(bootstrap, Mapping):
        raise RuntimeError(
            "formal training provenance bootstrap equivalence is invalid"
        )
    expected_bootstrap_fields = {
        "equivalence_contract",
        "verified_from_actual_script_bytes",
        "legacy_script_sha256",
        "expanded_script_sha256",
        "legacy_nested_teacher_experiments",
        "expanded_nested_teacher_experiments",
        "common_text_projection_sha256",
        "common_ast_projection_sha256",
        "only_text_difference",
        "only_ast_difference",
        "capacity_matched_hardware_contract",
        "runtime_guard_evidence",
        "tf32_override",
        "homogeneous_gpu_count",
        "allowed_compute_capabilities",
        "legacy_script_subjects",
        "expanded_script_subjects",
    }
    if set(bootstrap) != expected_bootstrap_fields:
        raise RuntimeError(
            "formal training provenance bootstrap field inventory mismatch"
        )
    expected_bootstrap = {
        "equivalence_contract": "nested-teacher-membership-only-v1",
        "verified_from_actual_script_bytes": True,
        "legacy_script_sha256": EXPECTED_LEGACY_TRAINING_SCRIPT_SHA256,
        "expanded_script_sha256": EXPECTED_CANONICAL_TRAINING_SCRIPT_SHA256,
        "legacy_nested_teacher_experiments": ["resilient_v2x", "support_residual"],
        "expanded_nested_teacher_experiments": sorted(
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
        ),
        "only_text_difference": "NESTED_TEACHER_EXPERIMENTS assignment",
        "only_ast_difference": "NESTED_TEACHER_EXPERIMENTS frozenset members",
        "capacity_matched_hardware_contract": "capacity-matched-hardware-v1",
        "runtime_guard_evidence": "reviewed_completed_bootstrap_bytes",
        "tf32_override": "0",
        "homogeneous_gpu_count": 4,
        "allowed_compute_capabilities": [[7, 0], [8, 0], [12, 0]],
    }
    for key, value in expected_bootstrap.items():
        if bootstrap.get(key) != value:
            raise RuntimeError(f"formal training provenance bootstrap {key} mismatch")
    _sha256(
        bootstrap.get("common_text_projection_sha256"),
        "formal training provenance common text projection",
    )
    _sha256(
        bootstrap.get("common_ast_projection_sha256"),
        "formal training provenance common AST projection",
    )

    run_equivalence = document.get("run_contract_equivalence")
    expected_run_equivalence = {
        "source_fields_vary_only_by_sealed_subject_revision_map": True,
        "source_revision_equivalence_seal_sha256": source_equivalence_seal,
        "source_revision_subject_map_seal_sha256": source_subject_map_seal,
        "source_revision_counts": {
            EXPECTED_SOURCE_TREE_SHA256: 21,
            EXPECTED_NEW_SOURCE_TREE_SHA256: 5,
        },
        "training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
        "native_bundle_sha256": EXPECTED_NATIVE_BUNDLE_SHA256,
        "build_manifest_sha256": EXPECTED_BUILD_MANIFEST_SHA256,
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_CHECKPOINT_SHA256,
        "training_seed": EXPECTED_TRAINING_SEED,
        "global_batch_size": 8,
        "precision": "FP32",
        "max_epochs": 50,
        "val_interval": 10,
        "amp": False,
    }
    if run_equivalence != expected_run_equivalence:
        raise RuntimeError("formal training provenance run contract mismatch")

    expected_capacity = {
        "contract": "capacity-matched-hardware-v1",
        "docker_command_sha256": EXPECTED_DOCKER_COMMAND_SHA256,
        "docker_command_identical_across_26_tasks": True,
        "gpu_count": 4,
        "homogeneous_per_task": True,
        "allowed_compute_capabilities": [[7, 0], [8, 0], [12, 0]],
        "gpu_class_may_vary_across_tasks": True,
        "tf32_override": "0",
        "precision": "FP32",
        "global_batch_size": 8,
        "runtime_guard_evidence": "reviewed_completed_bootstrap_bytes",
    }
    if document.get("capacity_matched_hardware") != expected_capacity:
        raise RuntimeError("formal training provenance capacity contract mismatch")

    parent_records = document.get("recovery_parent_controllers")
    if not isinstance(parent_records, list):
        raise RuntimeError("formal training provenance recovery parents are invalid")
    recovery_subjects: dict[str, set[str]] = {}
    for record in parent_records:
        if not isinstance(record, Mapping) or set(record) != {
            "task_id",
            "terminal_status",
            "roles",
            "actual_parent_subjects",
            "progress_artifact",
            "progress_seal_sha256",
            "progress_content_sha256",
            "raw_authority_sha256",
        }:
            raise RuntimeError("formal training provenance recovery parent is invalid")
        parent_id = _task_id(record.get("task_id"), "recovery parent task")
        if parent_id in recovery_subjects or parent_id in {
            controller_task_id,
            provenance_task_id,
        }:
            raise RuntimeError("formal training provenance recovery parent aliases")
        if record.get("terminal_status") != "failed":
            raise RuntimeError(
                "formal training provenance recovery parent status mismatch"
            )
        subjects = record.get("actual_parent_subjects")
        if (
            not isinstance(subjects, list)
            or any(subject not in FORMAL_SUBJECT_ORDER for subject in subjects)
            or len(subjects) != len(set(subjects))
        ):
            raise RuntimeError(
                "formal training provenance adopted subjects are invalid"
            )
        roles = record.get("roles")
        expected_roles = [
            role
            for role, present in (
                ("recursive_recovery_source", parent_id in progress_chain_by_id),
                ("actual_training_parent", bool(subjects)),
            )
            if present
        ]
        if roles != expected_roles or not expected_roles:
            raise RuntimeError(
                "formal training provenance recovery parent roles mismatch"
            )
        if parent_id in progress_chain_by_id:
            chain_record = progress_chain_by_id[parent_id]
            if (
                record.get("progress_artifact") != FORMAL_TRAINING_PROGRESS_ARTIFACT
                or record.get("progress_seal_sha256")
                != chain_record["progress_seal_sha256"]
                or record.get("progress_content_sha256")
                != chain_record["progress_content_sha256"]
            ):
                raise RuntimeError(
                    "formal training provenance recovery progress mismatch"
                )
        elif any(
            record.get(key) is not None
            for key in (
                "progress_artifact",
                "progress_seal_sha256",
                "progress_content_sha256",
            )
        ):
            raise RuntimeError(
                "formal training provenance recovery parent invents progress"
            )
        _sha256(record.get("raw_authority_sha256"), "recovery parent raw authority")
        recovery_subjects[parent_id] = set(subjects)

    records = document.get("training_tasks")
    if not isinstance(records, list) or len(records) != len(FORMAL_SUBJECT_ORDER):
        raise RuntimeError("formal training provenance task count mismatch")
    if len(training) != len(FORMAL_SUBJECT_ORDER):
        raise RuntimeError("formal training dependency count mismatch")
    policies = TrainingProvenancePolicies()
    legacy_subjects: list[str] = []
    expanded_subjects: list[str] = []
    seen_task_ids: set[str] = set()
    covered_recovery_subjects: set[str] = set()
    expected_entry_fields = {
        "index",
        "subject",
        "training_task_id",
        "predecessor_task_id",
        "parent_controller_task_id",
        "parent_binding",
        "recovery_lineage",
        "raw_bootstrap_script_sha256",
        "normalized_task_script_identity_sha256",
        "sealed_progress_script_identity_sha256",
        "script_equivalence_class",
        "manifest_model_id",
        "manifest_model_name",
        "manifest_model_url",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "source_revision_tree_sha256",
        "source_dataset_id",
        "source_archive_name",
        "source_archive_bytes",
        "source_archive_sha256",
        "observed_parameter_keys",
        "observed_parameters_sha256",
        "docker_command_sha256",
        "execution_queue_id",
        "last_worker",
        "raw_authority_sha256",
        "run_contract_content_sha256",
        "final_checkpoint_contract_content_sha256",
        "common_teacher_initialization_audit_content_sha256",
    }
    required_training_parameter_keys = {
        "Args/experiment_from_task",
        "Args/source_dataset_id",
        "Args/source_archive_name",
        "Args/source_archive_bytes",
        "Args/source_archive_sha256",
        "Args/training_dataset_id",
        "Args/native_bundle_sha256",
        "Args/build_manifest_sha256",
        "Args/predecessor_task_id",
        "Args/gpus",
        "Args/stage",
        "Args/max_epochs",
        "Args/amp",
        "Args/training_seed",
        "Args/teacher_checkpoint_sha256",
    }
    manifest_entries = manifest.get("entries")
    if not isinstance(manifest_entries, list) or len(manifest_entries) != len(
        FORMAL_SUBJECT_ORDER
    ):
        raise RuntimeError("formal training provenance manifest entry count mismatch")
    for index, (record, dependency, subject) in enumerate(
        zip(records, training, FORMAL_SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(record, Mapping) or set(record) != expected_entry_fields:
            raise RuntimeError(f"formal training provenance {subject} entry invalid")
        if record.get("index") != index or record.get("subject") != subject:
            raise RuntimeError("formal training provenance task/subject order mismatch")
        task_id = _task_id(record.get("training_task_id"), f"{subject} provenance task")
        predecessor = _task_id(
            record.get("predecessor_task_id"), f"{subject} provenance predecessor"
        )
        parent_id = _task_id(
            record.get("parent_controller_task_id"),
            f"{subject} provenance parent controller",
        )
        if task_id in seen_task_ids or task_id in {
            controller_task_id,
            provenance_task_id,
        }:
            raise RuntimeError("formal training provenance task IDs alias")
        seen_task_ids.add(task_id)
        if (
            task_id != dependency["training_task_id"]
            or predecessor != dependency["training_predecessor_task_id"]
            or record.get("manifest_model_id") != dependency["training_model_id"]
        ):
            raise RuntimeError(
                f"formal training provenance {subject} manifest binding mismatch"
            )
        manifest_entry = manifest_entries[index - 1]
        if not isinstance(manifest_entry, Mapping) or any(
            record.get(record_key) != manifest_entry.get(manifest_key)
            for record_key, manifest_key in {
                "manifest_model_id": "model_id",
                "manifest_model_name": "model_name",
                "manifest_model_url": "model_url",
                "checkpoint_sha256": "checkpoint_sha256",
                "checkpoint_size_bytes": "checkpoint_size_bytes",
            }.items()
        ):
            raise RuntimeError(
                f"formal training provenance {subject} static manifest binding mismatch"
            )
        source_revision = SOURCE_REVISION_BY_SUBJECT[subject]
        expected_source = EXPECTED_SOURCE_BY_REVISION[source_revision]
        if (
            record.get("source_revision_tree_sha256")
            != expected_source["tree_sha256"]
            or record.get("source_dataset_id") != expected_source["dataset_id"]
            or record.get("source_archive_name")
            != expected_source["archive_name"]
            or record.get("source_archive_bytes")
            != expected_source["archive_size_bytes"]
            or record.get("source_archive_sha256")
            != expected_source["archive_sha256"]
        ):
            raise RuntimeError(
                f"formal training provenance {subject} source revision mismatch"
            )
        script_sha = _sha256(
            record.get("raw_bootstrap_script_sha256"),
            f"{subject} provenance raw bootstrap script",
        )
        if script_sha == EXPECTED_LEGACY_TRAINING_SCRIPT_SHA256:
            if subject not in LEGACY_TRAINING_SCRIPT_ALLOWED_SUBJECTS:
                raise RuntimeError(
                    f"legacy training bootstrap is not allowed for {subject}"
                )
            if record.get("script_equivalence_class") != (
                "legacy_nested_teacher_membership"
            ):
                raise RuntimeError(
                    f"formal training provenance {subject} script class mismatch"
                )
            legacy_subjects.append(subject)
        elif script_sha == EXPECTED_CANONICAL_TRAINING_SCRIPT_SHA256:
            if record.get("script_equivalence_class") != (
                "expanded_nested_teacher_membership"
            ):
                raise RuntimeError(
                    f"formal training provenance {subject} script class mismatch"
                )
            expanded_subjects.append(subject)
        else:
            raise RuntimeError(
                f"formal training provenance {subject} uses a third script"
            )
        binding = str(record.get("parent_binding") or "")
        lineage = _validate_training_recovery_lineage(
            record.get("recovery_lineage"),
            subject=subject,
            controller_task_id=controller_task_id,
            parent_controller_task_id=parent_id,
            parent_binding=binding,
            current_progress_seal_sha256=str(document["progress_seal_sha256"]),
        )
        if parent_id == controller_task_id:
            if binding != "current_controller_created" or any(
                subject in adopted for adopted in recovery_subjects.values()
            ):
                raise RuntimeError(
                    f"formal training provenance {subject} parent binding mismatch"
                )
        else:
            if (
                parent_id not in recovery_subjects
                or subject not in recovery_subjects[parent_id]
            ):
                raise RuntimeError(
                    f"formal training provenance {subject} recovery parent mismatch"
                )
            covered_recovery_subjects.add(subject)
        normalized_script_sha = _sha256(
            record.get("normalized_task_script_identity_sha256"),
            f"{subject} normalized task script identity",
        )
        if record.get("sealed_progress_script_identity_sha256") != (
            normalized_script_sha
        ):
            raise RuntimeError(
                f"formal training provenance {subject} progress script identity mismatch"
            )
        parameter_keys = record.get("observed_parameter_keys")
        if (
            not isinstance(parameter_keys, list)
            or not parameter_keys
            or any(type(key) is not str or not key for key in parameter_keys)
            or parameter_keys != sorted(set(parameter_keys))
            or not required_training_parameter_keys <= set(parameter_keys)
        ):
            raise RuntimeError(
                f"formal training provenance {subject} parameter inventory mismatch"
            )
        observed_parameters_sha = _sha256(
            record.get("observed_parameters_sha256"),
            f"{subject} provenance observed parameters",
        )
        docker_command_sha = _sha256(
            record.get("docker_command_sha256"),
            f"{subject} provenance Docker command",
        )
        if docker_command_sha != EXPECTED_DOCKER_COMMAND_SHA256:
            raise RuntimeError(
                f"formal training provenance {subject} Docker command mismatch"
            )
        _sha256(record.get("raw_authority_sha256"), f"{subject} raw authority")
        if record.get("execution_queue_id") not in set(EXPECTED_QUEUE_IDS.values()):
            raise RuntimeError(
                f"formal training provenance {subject} execution queue mismatch"
            )
        if type(record.get("last_worker")) is not str or not record["last_worker"]:
            raise RuntimeError(
                f"formal training provenance {subject} last worker is missing"
            )
        run_contract_sha = _sha256(
            record.get("run_contract_content_sha256"),
            f"{subject} provenance run contract",
        )
        final_contract_sha = _sha256(
            record.get("final_checkpoint_contract_content_sha256"),
            f"{subject} provenance final checkpoint contract",
        )
        teacher_audit_sha = _sha256(
            record.get("common_teacher_initialization_audit_content_sha256"),
            f"{subject} provenance common teacher initialization audit",
        )
        if teacher_audit_sha != dependency["training_initialization_audit_sha256"]:
            raise RuntimeError(
                f"formal training provenance {subject} initialization audit binding mismatch"
            )
        policies[subject] = {
            "training_task_id": task_id,
            "predecessor_task_id": predecessor,
            "parent_controller_task_id": parent_id,
            "script_sha256": script_sha,
            "normalized_task_script_identity_sha256": normalized_script_sha,
            "recovery_lineage": lineage,
            "observed_parameter_keys": parameter_keys,
            "observed_parameters_sha256": observed_parameters_sha,
            "docker_command_sha256": docker_command_sha,
            "execution_queue_id": record["execution_queue_id"],
            "last_worker": record["last_worker"],
            "source_revision": source_revision,
            "source_revision_tree_sha256": expected_source["tree_sha256"],
            "source_dataset_id": expected_source["dataset_id"],
            "source_archive_name": expected_source["archive_name"],
            "source_archive_bytes": expected_source["archive_size_bytes"],
            "source_archive_sha256": expected_source["archive_sha256"],
            "run_contract_content_sha256": run_contract_sha,
            "final_checkpoint_contract_content_sha256": final_contract_sha,
            "common_teacher_initialization_audit_content_sha256": (teacher_audit_sha),
        }
    all_recovery_subjects = (
        set().union(*recovery_subjects.values()) if recovery_subjects else set()
    )
    if covered_recovery_subjects != all_recovery_subjects:
        raise RuntimeError(
            "formal training provenance recovery inventory is not closed"
        )
    if (
        bootstrap.get("legacy_script_subjects") != legacy_subjects
        or bootstrap.get("expanded_script_subjects") != expanded_subjects
    ):
        raise RuntimeError(
            "formal training provenance script subject inventory mismatch"
        )
    policies.provenance_task_id = provenance_task_id
    policies.provenance_seal_sha256 = _sha256(
        document.get("seal_sha256"), "formal training provenance seal"
    )
    policies.source_revision_equivalence = source_equivalence
    policies.source_revision_equivalence_seal_sha256 = source_equivalence_seal
    policies.source_revision_subject_map = source_subject_map
    policies.source_revision_subject_map_seal_sha256 = source_subject_map_seal
    return policies


def _training_provenance_task_snapshot(
    task_class: object,
    *,
    training: list[dict[str, str]],
    policies: Mapping[str, Mapping[str, object]],
    expected_project_id: str,
) -> str:
    snapshots: list[dict[str, object]] = []
    for dependency in training:
        subject = dependency["subject"]
        policy = policies[subject]
        task = task_class.get_task(task_id=dependency["training_task_id"])
        observed_status = _status(task)
        observed_id = _task_id(getattr(task, "id", ""), f"{subject} live training task")
        parameters = _parameters(task)
        docker_getter = getattr(task, "get_base_docker", None)
        if not callable(docker_getter):
            raise RuntimeError(
                f"training task {subject} cannot expose its Docker command"
            )
        docker_command = docker_getter()
        if type(docker_command) is not str or not docker_command.strip():
            raise RuntimeError(f"training task {subject} Docker command is invalid")
        data = getattr(task, "data", None)
        execution = getattr(data, "execution", None)
        snapshot = {
            "subject": subject,
            "task_id": observed_id,
            "project_id": _task_project_id(
                task, context=f"training task {subject}"
            ),
            "status": observed_status,
            "parent_controller_task_id": _task_parent(task),
            "script_sha256": _script_sha256(task, context=f"training task {subject}"),
            "normalized_task_script_identity_sha256": (
                _normalized_task_script_identity_sha256(task)
            ),
            "experiment": parameters.get("Args/experiment_from_task"),
            "predecessor_task_id": parameters.get("Args/predecessor_task_id"),
            "observed_parameter_keys": sorted(parameters),
            "observed_parameters_sha256": hashlib.sha256(
                _canonical_json(parameters).encode("utf-8")
            ).hexdigest(),
            "docker_command_sha256": hashlib.sha256(
                docker_command.strip().encode("utf-8")
            ).hexdigest(),
            "execution_queue_id": str(getattr(execution, "queue", "") or ""),
            "last_worker": str(getattr(data, "last_worker", "") or ""),
        }
        expected = {
            "subject": subject,
            "task_id": policy["training_task_id"],
            "project_id": expected_project_id,
            "status": "completed",
            "parent_controller_task_id": policy["parent_controller_task_id"],
            "script_sha256": policy["script_sha256"],
            "normalized_task_script_identity_sha256": policy[
                "normalized_task_script_identity_sha256"
            ],
            "experiment": subject,
            "predecessor_task_id": policy["predecessor_task_id"],
            "observed_parameter_keys": policy["observed_parameter_keys"],
            "observed_parameters_sha256": policy["observed_parameters_sha256"],
            "docker_command_sha256": policy["docker_command_sha256"],
            "execution_queue_id": policy["execution_queue_id"],
            "last_worker": policy["last_worker"],
        }
        if snapshot != expected:
            raise RuntimeError(f"training task {subject} provenance preflight mismatch")
        snapshots.append(snapshot)
    return _canonical_json(snapshots)


def _consume_training_provenance(
    *,
    task_class: object,
    provenance_task_id: str,
    expected_provenance_script_sha256: str,
    controller: object,
    controller_task_id: str,
    manifest: Mapping[str, object],
    training: list[dict[str, str]],
    metadata_only_artifact_gate: bool = False,
    authenticated_artifact_downloader: (
        Callable[[object, str], bytes] | None
    ) = None,
) -> dict[str, dict[str, object]]:
    if authenticated_artifact_downloader is not None:
        def stable_reader(
            artifact_task: object,
            artifact_name: str,
            *,
            required_status: str,
        ) -> dict[str, object]:
            return _stable_authenticated_artifact_mapping(
                artifact_task,
                artifact_name,
                required_status=required_status,
                downloader=authenticated_artifact_downloader,
            )
    else:
        stable_reader = (
            _stable_artifact_preview_mapping
            if metadata_only_artifact_gate
            else _stable_artifact_mapping
        )
    provenance = task_class.get_task(task_id=provenance_task_id)
    if _status(provenance) != "completed":
        raise RuntimeError("formal training provenance task is not completed")
    if _task_id(getattr(provenance, "id", ""), "formal training provenance task") != (
        provenance_task_id
    ):
        raise RuntimeError("formal training provenance task identity mismatch")
    if _task_parent(provenance) != controller_task_id:
        raise RuntimeError("formal training provenance task parent mismatch")
    project_id = _task_project_id(
        provenance, context="formal training provenance task"
    )
    if _task_project_id(controller, context="formal training controller") != project_id:
        raise RuntimeError("formal training controller project ID mismatch")
    if (
        _standalone_script_sha256(
            provenance,
            expected_entry_point="clearml_formal_training_provenance.py",
            context="formal training provenance task",
        )
        != expected_provenance_script_sha256
    ):
        raise RuntimeError("formal training provenance task script SHA-256 mismatch")
    parameters = _parameters(provenance)
    if (
        set(parameters)
        != {
            "Args/training_controller_task_id",
            "Args/poll_seconds",
            "Args/timeout_hours",
        }
        or parameters.get("Args/training_controller_task_id") != controller_task_id
    ):
        raise RuntimeError("formal training provenance task Args contract mismatch")
    for key in ("Args/poll_seconds", "Args/timeout_hours"):
        try:
            value = float(str(parameters[key]))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "formal training provenance task Args contract mismatch"
            ) from error
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("formal training provenance task Args contract mismatch")
    _validate_provenance_tags(provenance)
    artifacts = getattr(provenance, "artifacts", None)
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        FORMAL_TRAINING_PROVENANCE_ARTIFACT
    }:
        raise RuntimeError("formal training provenance artifact inventory mismatch")
    document = stable_reader(
        provenance,
        FORMAL_TRAINING_PROVENANCE_ARTIFACT,
        required_status="completed",
    )
    policies = _validate_training_provenance_document(
        document,
        provenance_task_id=provenance_task_id,
        controller_task_id=controller_task_id,
        manifest=manifest,
        training=training,
    )
    progress = stable_reader(
        controller,
        FORMAL_TRAINING_PROGRESS_ARTIFACT,
        required_status="completed",
    )
    _require_valid_seal(progress, context="formal training progress")
    if progress.get("seal_sha256") != document["progress_seal_sha256"] or (
        hashlib.sha256(_canonical_json(progress).encode("utf-8")).hexdigest()
        != document["progress_content_sha256"]
    ):
        raise RuntimeError("formal training provenance progress binding mismatch")
    recovery = progress.get("recovery")
    if (
        not isinstance(recovery, Mapping)
        or hashlib.sha256(_canonical_json(recovery).encode("utf-8")).hexdigest()
        != document["recovery_contract_sha256"]
    ):
        raise RuntimeError("formal training provenance recovery contract mismatch")
    lineage_progress_seals: dict[str, str] = {}
    recovery_parent_ids: set[str] = set()
    for policy in policies.values():
        parent_id = str(policy["parent_controller_task_id"])
        if parent_id != controller_task_id:
            recovery_parent_ids.add(parent_id)
        lineage = policy["recovery_lineage"]
        if not isinstance(lineage, list):  # pragma: no cover - validator owns shape
            raise RuntimeError("formal training provenance lineage policy is invalid")
        for node in lineage:
            if not isinstance(node, Mapping):  # pragma: no cover - validator owns shape
                raise RuntimeError("formal training provenance lineage node is invalid")
            lineage_controller_id = str(node["controller_task_id"])
            lineage_seal = str(node["progress_seal_sha256"])
            previous = lineage_progress_seals.setdefault(
                lineage_controller_id, lineage_seal
            )
            if previous != lineage_seal:
                raise RuntimeError(
                    "formal training provenance lineage controller seal mismatch"
                )
    if lineage_progress_seals.get(controller_task_id) != progress.get("seal_sha256"):
        raise RuntimeError("formal training provenance root lineage seal mismatch")
    progress_chain = document["recursive_progress_chain"]
    if not isinstance(progress_chain, list):  # pragma: no cover - validator owns shape
        raise RuntimeError("formal training provenance progress chain is invalid")
    chain_by_id = {
        str(record["task_id"]): record
        for record in progress_chain
        if isinstance(record, Mapping)
    }
    if not set(lineage_progress_seals) <= set(chain_by_id):
        raise RuntimeError("formal training provenance lineage inventory mismatch")
    for chain_index, raw_expected_record in enumerate(progress_chain):
        if not isinstance(raw_expected_record, Mapping):  # pragma: no cover
            raise RuntimeError("formal training provenance progress chain is invalid")
        expected_record = raw_expected_record
        lineage_controller_id = str(expected_record["task_id"])
        if lineage_controller_id == controller_task_id:
            lineage_controller = controller
            lineage_progress = progress
        else:
            lineage_controller = task_class.get_task(task_id=lineage_controller_id)
            lineage_progress = stable_reader(
                lineage_controller,
                FORMAL_TRAINING_PROGRESS_ARTIFACT,
                required_status="failed",
            )
        if (
            _task_id(
                getattr(lineage_controller, "id", ""),
                "recursive progress controller",
            )
            != lineage_controller_id
        ):
            raise RuntimeError(
                "formal training provenance recursive progress task mismatch"
            )
        _require_valid_seal(
            lineage_progress,
            context=f"recovery controller {lineage_controller_id} progress",
        )
        template = lineage_progress.get("template")
        if (
            lineage_progress.get("controller_task_id") != lineage_controller_id
            or lineage_progress.get("seal_sha256")
            != expected_record["progress_seal_sha256"]
            or hashlib.sha256(
                _canonical_json(lineage_progress).encode("utf-8")
            ).hexdigest()
            != expected_record["progress_content_sha256"]
            or lineage_progress.get("revision") != expected_record["progress_revision"]
            or not isinstance(template, Mapping)
            or template.get("script_sha256")
            != expected_record["template_script_identity_sha256"]
            or lineage_progress_seals[lineage_controller_id]
            != expected_record["progress_seal_sha256"]
        ):
            raise RuntimeError(
                "formal training provenance recursive progress binding mismatch"
            )
        recovery_contract = lineage_progress.get("recovery")
        next_controller_id = (
            str(progress_chain[chain_index + 1]["task_id"])
            if chain_index + 1 < len(progress_chain)
            and isinstance(progress_chain[chain_index + 1], Mapping)
            else None
        )
        if next_controller_id is None:
            if recovery_contract is not None:
                raise RuntimeError(
                    "formal training provenance progress chain is not terminal"
                )
        elif (
            not isinstance(recovery_contract, Mapping)
            or recovery_contract.get("source_controller_task_id") != next_controller_id
            or recovery_contract.get("source_controller_status") != "failed"
        ):
            raise RuntimeError(
                "formal training provenance recovery source chain mismatch"
            )
    for parent_id in recovery_parent_ids - set(lineage_progress_seals):
        if _status(task_class.get_task(task_id=parent_id)) != "failed":
            raise RuntimeError(
                "formal training provenance external recovery parent is not failed"
            )
    stable_manifest = stable_reader(
        controller,
        FORMAL_TRAINING_MANIFEST_ARTIFACT,
        required_status="completed",
    )
    if stable_manifest != dict(manifest):
        raise RuntimeError(
            "formal training manifest changed before evaluation planning"
        )
    first_tasks = _training_provenance_task_snapshot(
        task_class,
        training=training,
        policies=policies,
        expected_project_id=project_id,
    )
    second_tasks = _training_provenance_task_snapshot(
        task_class,
        training=training,
        policies=policies,
        expected_project_id=project_id,
    )
    if first_tasks != second_tasks:
        raise RuntimeError("formal training task metadata changed across preflight")
    terminal_document = stable_reader(
        provenance,
        FORMAL_TRAINING_PROVENANCE_ARTIFACT,
        required_status="completed",
    )
    if terminal_document != document:
        raise RuntimeError(
            "formal training provenance changed before evaluation planning"
        )
    policies.project_id = project_id
    return policies


def _require_final_model_url(
    value: object,
    *,
    subject: str,
    artifact_origins: set[tuple[str, str]],
) -> str:
    context = f"training dependency {subject} final model"
    url = _reasonable_remote_url(value, context)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"{context} URL is not a fileserver URL")
    if artifact_origins and (parsed.scheme, parsed.netloc) not in artifact_origins:
        raise RuntimeError(f"{context} URL is not on the artifact fileserver")
    filename = PurePosixPath(unquote(parsed.path)).name
    if filename != f"{subject}_epoch_50.pth":
        raise RuntimeError(f"{context} basename mismatch")
    return url


def _require_completed_training(
    task: object,
    *,
    subject: str,
    task_id: str,
    expected_script_sha256: str,
    expected_source_dataset_id: str,
    expected_source_archive_name: str,
    expected_source_archive_bytes: int,
    expected_source_archive_sha256: str,
    expected_training_dataset_id: str,
    expected_predecessor_task_id: str,
    expected_parent_controller_task_id: str = "",
    expected_run_contract_content_sha256: str = "",
    expected_final_checkpoint_contract_content_sha256: str = "",
    expected_final_model_id: str = "",
    expected_final_checkpoint_sha256: str = "",
    expected_final_model_url: str = "",
    expected_initialization_audit_sha256: str = "",
    metadata_only_artifact_gate: bool = False,
) -> bool:
    status = _status(task)
    if status in FAILED_STATUSES:
        raise RuntimeError(
            f"training dependency {subject} ({task_id}) ended as {status!r}"
        )
    if status != "completed":
        return False
    if expected_parent_controller_task_id and _task_parent(task) != (
        expected_parent_controller_task_id
    ):
        raise RuntimeError(f"training dependency {subject} parent mismatch")
    parameters = _parameters(task)
    expected_parameters = {
        "Args/experiment_from_task": subject,
        "Args/source_dataset_id": expected_source_dataset_id,
        "Args/source_archive_name": expected_source_archive_name,
        "Args/source_archive_bytes": str(expected_source_archive_bytes),
        "Args/source_archive_sha256": expected_source_archive_sha256,
        "Args/training_dataset_id": expected_training_dataset_id,
        "Args/predecessor_task_id": expected_predecessor_task_id,
        "Args/stage": "all",
        "Args/max_epochs": "50",
        "Args/gpus": "4",
        "Args/amp": "False",
    }
    for key, expected in expected_parameters.items():
        if str(parameters.get(key)) != expected:
            raise RuntimeError(
                f"training dependency {subject} parameter {key} mismatch"
            )
    script = getattr(getattr(task, "data", None), "script", None)
    if (
        str(getattr(script, "repository", "") or "") != ""
        or str(getattr(script, "working_dir", "") or "") != "."
        or str(getattr(script, "entry_point", "") or "") != "clearml_5090_bootstrap.py"
    ):
        raise RuntimeError(f"training dependency {subject} script identity mismatch")
    training_script_sha256 = hashlib.sha256(
        str(getattr(script, "diff", "") or "").encode("utf-8")
    ).hexdigest()
    if training_script_sha256 != expected_script_sha256:
        raise RuntimeError(f"training dependency {subject} script SHA-256 mismatch")
    run_artifact, run_record, run_artifact_origins = _unique_artifact_metadata(
        task, "run_contract"
    )
    expected_run_contract = {
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": subject,
        "source_dataset_id": expected_source_dataset_id,
        "training_dataset_id": expected_training_dataset_id,
        "predecessor_task_id": expected_predecessor_task_id,
        "gpus": 4,
        "global_batch_size": 8,
        "max_epochs": 50,
        "val_interval": 10,
        "precision": "FP32",
    }
    run_contract = (
        _artifact_preview_mapping(run_artifact, run_record, "run_contract")
        if metadata_only_artifact_gate
        else _artifact_mapping(run_artifact, "run_contract")
    )
    if (
        expected_run_contract_content_sha256
        and hashlib.sha256(_canonical_json(run_contract).encode("utf-8")).hexdigest()
        != expected_run_contract_content_sha256
    ):
        raise RuntimeError(
            f"training dependency {subject} sealed run contract SHA-256 mismatch"
        )
    for key, expected in expected_run_contract.items():
        if run_contract.get(key) != expected:
            raise RuntimeError(
                f"training dependency {subject} run contract {key} mismatch"
            )
    source_archive = run_contract.get("source_archive")
    if not isinstance(source_archive, Mapping):
        raise RuntimeError(
            f"training dependency {subject} source archive contract is missing"
        )
    expected_source_archive = {
        "name": expected_source_archive_name,
        "size_bytes": expected_source_archive_bytes,
        "sha256": expected_source_archive_sha256,
    }
    if source_archive != expected_source_archive:
        raise RuntimeError(
            f"training dependency {subject} source archive identity mismatch"
        )
    if expected_initialization_audit_sha256:
        audit_artifact, _, _ = _unique_artifact_metadata(
            task, COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT
        )
        initialization_audit = _artifact_mapping(
            audit_artifact,
            COMMON_TEACHER_INITIALIZATION_AUDIT_ARTIFACT,
        )
        observed_audit_sha256 = hashlib.sha256(
            _canonical_json(initialization_audit).encode("utf-8")
        ).hexdigest()
        if observed_audit_sha256 != expected_initialization_audit_sha256:
            raise RuntimeError(
                f"training dependency {subject} initialization audit SHA-256 mismatch"
            )
    final_artifact, final_record, final_artifact_origins = _unique_artifact_metadata(
        task, "final_checkpoint_contract"
    )
    model_name = f"ResilientV2X {subject} final checkpoint"
    get_models = getattr(task, "get_models", None)
    if not callable(get_models):
        raise RuntimeError(f"training dependency {subject} does not expose models")
    models = get_models()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"training dependency {subject} returned invalid models")
    output_models = [
        model
        for model in models.get("output", [])
        if str(getattr(model, "name", "") or "") == model_name
        and str(getattr(model, "task", "") or "") == task_id
    ]
    if len(output_models) != 1:
        raise RuntimeError(
            f"training dependency {subject} must expose exactly one final model"
        )
    model = output_models[0]
    if expected_final_model_id and (
        str(getattr(model, "id", "") or "") != expected_final_model_id
    ):
        raise RuntimeError(f"training dependency {subject} final model ID mismatch")
    artifact_origins = run_artifact_origins | final_artifact_origins
    model_url = _require_final_model_url(
        getattr(model, "url", ""),
        subject=subject,
        artifact_origins=artifact_origins,
    )
    if expected_final_model_url and model_url != expected_final_model_url:
        raise RuntimeError(
            f"training dependency {subject} sealed final model URL mismatch"
        )
    final_contract = (
        _artifact_preview_mapping(
            final_artifact,
            final_record,
            "final_checkpoint_contract",
        )
        if metadata_only_artifact_gate
        else _artifact_mapping(final_artifact, "final_checkpoint_contract")
    )
    if (
        expected_final_checkpoint_contract_content_sha256
        and hashlib.sha256(_canonical_json(final_contract).encode("utf-8")).hexdigest()
        != expected_final_checkpoint_contract_content_sha256
    ):
        raise RuntimeError(
            f"training dependency {subject} sealed final contract SHA-256 mismatch"
        )
    expected_final_contract = {
        "model_id": str(getattr(model, "id", "") or ""),
        "name": model_name,
        "filename": "epoch_50.pth",
    }
    for key, expected in expected_final_contract.items():
        if final_contract.get(key) != expected:
            raise RuntimeError(
                f"training dependency {subject} final contract {key} mismatch"
            )
    if (
        type(final_contract.get("size_bytes")) is not int
        or final_contract["size_bytes"] <= 0
    ):
        raise RuntimeError(
            f"training dependency {subject} final checkpoint size is invalid"
        )
    checkpoint_sha256 = _sha256(
        final_contract.get("sha256"),
        f"training dependency {subject} final checkpoint",
    )
    if (
        expected_final_checkpoint_sha256
        and checkpoint_sha256 != expected_final_checkpoint_sha256
    ):
        raise RuntimeError(
            f"training dependency {subject} final checkpoint SHA-256 mismatch"
        )
    if final_contract.get("url") != model_url:
        raise RuntimeError(f"training dependency {subject} final model URL mismatch")
    return True


def _require_evaluation_identity(
    task: object,
    *,
    dependency: Mapping[str, str],
    expected_script_sha256: str,
    expected_source_dataset_id: str,
    expected_source_archive_sha256: str,
    expected_training_dataset_id: str,
) -> str:
    status = _status(task)
    if status in FAILED_STATUSES:
        raise RuntimeError(
            f"evaluation task {dependency['subject']} ended as {status!r}"
        )
    parameters = _parameters(task)
    expected_parameters = {
        "Args/stage": "baseline_validate",
        "Args/controlled_baseline": dependency["subject"],
        "Args/controlled_baseline_task_id": dependency["training_task_id"],
        "Args/predecessor_task_id": dependency["training_task_id"],
        "Args/source_dataset_id": expected_source_dataset_id,
        "Args/source_archive_sha256": expected_source_archive_sha256,
        "Args/training_dataset_id": expected_training_dataset_id,
        "Args/gpus": "4",
        "Args/max_epochs": "50",
        "Args/amp": "False",
    }
    if dependency.get("training_model_id"):
        expected_parameters.update(
            {
                "Args/controlled_baseline_model_id": dependency["training_model_id"],
                "Args/controlled_baseline_checkpoint_sha256": dependency[
                    "training_checkpoint_sha256"
                ],
            }
        )
    for key, expected in expected_parameters.items():
        if parameters.get(key) != expected:
            raise RuntimeError(
                f"evaluation {dependency['subject']} parameter {key} mismatch"
            )
    script = getattr(getattr(task, "data", None), "script", None)
    if (
        str(getattr(script, "repository", "") or "") != ""
        or str(getattr(script, "working_dir", "") or "") != "."
        or str(getattr(script, "entry_point", "") or "") != "clearml_5090_bootstrap.py"
    ):
        raise RuntimeError(
            f"evaluation {dependency['subject']} script identity mismatch"
        )
    diff = str(getattr(script, "diff", "") or "")
    observed_script_sha256 = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    if observed_script_sha256 != expected_script_sha256:
        raise RuntimeError(
            f"evaluation {dependency['subject']} script SHA-256 mismatch"
        )
    if status in {"created", "queued"}:
        input_models = getattr(task, "get_models")().get("input", [])
        if input_models:
            raise RuntimeError(
                f"evaluation {dependency['subject']} inherited input models"
            )
    execution = getattr(getattr(task, "data", None), "execution", None)
    observed_queue_id = str(getattr(execution, "queue", "") or "")
    if status == "created" and observed_queue_id:
        raise RuntimeError(
            f"evaluation {dependency['subject']} created task already has a queue"
        )
    if status in {"queued", "in_progress", "completed"}:
        expected_queue_id = EXPECTED_QUEUE_IDS[dependency["queue"]]
        if observed_queue_id != expected_queue_id:
            raise RuntimeError(f"evaluation {dependency['subject']} queue mismatch")
    return status


def _require_released_input_model(
    evaluation: object,
    training: object,
    *,
    subject: str,
    status: str,
) -> None:
    if status not in {"in_progress", "completed"}:
        return
    model_name = f"ResilientV2X {subject} final checkpoint"
    final_models = [
        model
        for model in getattr(training, "get_models")().get("output", [])
        if str(getattr(model, "name", "") or "") == model_name
    ]
    if len(final_models) != 1:
        raise RuntimeError(
            f"training dependency {subject} final model changed after release"
        )
    expected_model_id = str(getattr(final_models[0], "id", "") or "")
    input_model_ids = {
        str(getattr(model, "id", "") or "")
        for model in getattr(evaluation, "get_models")().get("input", [])
    }
    allowed = {frozenset(), frozenset({expected_model_id})}
    if frozenset(input_model_ids) not in allowed:
        raise RuntimeError(
            f"evaluation {subject} has unexpected input models after release"
        )
    if status == "completed" and input_model_ids != {expected_model_id}:
        raise RuntimeError(
            f"completed evaluation {subject} lacks the final input model"
        )


def _require_completed_evaluation_metrics(
    task: object,
    *,
    dependency: Mapping[str, str],
) -> None:
    artifacts = getattr(task, "artifacts", None)
    if (
        not isinstance(artifacts, Mapping)
        or "controlled_baseline_metrics" not in artifacts
    ):
        raise RuntimeError(
            f"completed evaluation {dependency['subject']} lacks metrics evidence"
        )
    metrics = _artifact_mapping(
        artifacts["controlled_baseline_metrics"],
        "controlled_baseline_metrics",
    )
    expected = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": 12,
        "baseline": dependency["subject"],
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "expected_sample_count": EXPECTED_SAMPLE_COUNT,
        "expected_ground_truth_count": 11_330,
        "expected_unsupported_sample_count": 0,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(
                f"completed evaluation {dependency['subject']} metrics {key} mismatch"
            )
    expected_checkpoint_sha256 = dependency.get("training_checkpoint_sha256", "")
    if expected_checkpoint_sha256 and metrics.get("checkpoint_sha256") != (
        expected_checkpoint_sha256
    ):
        raise RuntimeError(
            f"completed evaluation {dependency['subject']} checkpoint SHA mismatch"
        )
    runs = metrics.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise RuntimeError(
            f"completed evaluation {dependency['subject']} run count mismatch"
        )
    observed = {
        (run.get("delay_ms"), run.get("condition"))
        for run in runs
        if isinstance(run, Mapping)
    }
    expected_matrix = {
        (delay, condition)
        for delay in EXPECTED_DELAYS_MS
        for condition in EXPECTED_CONDITIONS
    }
    if observed != expected_matrix:
        raise RuntimeError(
            f"completed evaluation {dependency['subject']} matrix is incomplete"
        )
    for run in runs:
        if not isinstance(run, Mapping) or any(
            run.get(key) != value
            for key, value in {
                "sample_count": EXPECTED_SAMPLE_COUNT,
                "ground_truth_count": 11_330,
                "unsupported_sample_count": 0,
            }.items()
        ):
            raise RuntimeError(
                f"completed evaluation {dependency['subject']} run evidence drifted"
            )


def run(args: argparse.Namespace, *, task_class: object = Task) -> None:
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("poll interval and timeout must be positive")
    current_task = task_class.current_task()
    if current_task is None:
        raise RuntimeError("dependency watcher requires a current ClearML task")
    current_task.set_tags(
        [
            "ResilientV2X-suite",
            "dependency-watcher",
            EXPECTED_PROTOCOL_ID,
            "cpu-controller",
        ]
    )
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    controller_task_id_value = getattr(args, "training_controller_task_id", None)
    evaluation_template_task_id = getattr(args, "evaluation_template_task_id", "")
    provenance_task_id_value = getattr(args, "training_provenance_task_id", "")
    provenance_script_pin_value = getattr(
        args, "expected_training_provenance_script_sha256", ""
    )
    evaluation_plan_json = getattr(args, "evaluation_plan_json", "")
    controller_mode = controller_task_id_value is not None
    blocker_contract_raw = getattr(args, "candidate_release_blockers_json", "")
    authoritative_plan_producer_id_raw = getattr(
        args, "authoritative_evaluation_plan_producer_task_id", ""
    )
    authoritative_plan_seal_raw = getattr(
        args, "authoritative_evaluation_plan_seal_sha256", ""
    )
    amendment_pins = {
        "receipt": getattr(
            args,
            "authoritative_evaluation_plan_amendment_receipt_seal_sha256",
            "",
        ),
        "producer_source": getattr(
            args, "authoritative_evaluation_plan_producer_source_sha256", ""
        ),
        "amendment_evidence": getattr(
            args,
            "authoritative_evaluation_plan_amendment_evidence_seal_sha256",
            "",
        ),
        "worker_evidence": getattr(
            args,
            "authoritative_evaluation_plan_worker_evidence_seal_sha256",
            "",
        ),
        "task_ids": getattr(
            args, "authoritative_evaluation_plan_task_ids_sha256", ""
        ),
    }
    amendment_requested = any(bool(value) for value in amendment_pins.values())
    explicit_amendment_pins = {
        "producer": getattr(
            args, "evaluation_plan_amendment_producer_task_id", ""
        ),
        "receipt": getattr(
            args, "evaluation_plan_amendment_receipt_seal_sha256", ""
        ),
        "revised_plan": getattr(
            args, "evaluation_plan_amendment_revised_plan_seal_sha256", ""
        ),
        "evidence": getattr(
            args, "evaluation_plan_amendment_evidence_seal_sha256", ""
        ),
        "worker": getattr(
            args, "evaluation_plan_amendment_worker_evidence_seal_sha256", ""
        ),
        "task_ids": getattr(
            args, "evaluation_plan_amendment_task_ids_sha256", ""
        ),
    }
    explicit_amendment_requested = any(
        bool(value) for value in explicit_amendment_pins.values()
    )
    runtime_recovery_pins = {
        "receipt": getattr(
            args, "exact_eval_runtime_recovery_receipt_seal_sha256", ""
        ),
        "attempt": getattr(
            args, "exact_eval_runtime_recovery_attempt_seal_sha256", ""
        ),
        "ffnet_source": getattr(
            args, "exact_eval_runtime_ffnet_source_sha256", ""
        ),
        "ffnet_parameters": getattr(
            args, "exact_eval_runtime_ffnet_parameters_sha256", ""
        ),
        "candidate_source": getattr(
            args, "exact_eval_runtime_candidate_source_sha256", ""
        ),
        "candidate_parameters": getattr(
            args, "exact_eval_runtime_candidate_parameters_sha256", ""
        ),
    }
    runtime_recovery_requested = any(
        bool(value) for value in runtime_recovery_pins.values()
    )
    template_task_id = ""
    if controller_mode:
        template_task_id = _task_id(
            evaluation_template_task_id,
            "evaluation template task",
        )
        template = task_class.get_task(task_id=template_task_id)
        derived_script_sha256 = _script_sha256(template, context="evaluation template")
        expected_evaluation_script_sha256 = _sha256(
            args.expected_eval_script_sha256 or derived_script_sha256,
            "evaluation script",
        )
        if expected_evaluation_script_sha256 != derived_script_sha256:
            raise ValueError("evaluation template script SHA-256 does not match pin")
    else:
        if evaluation_template_task_id:
            raise ValueError(
                "--evaluation-template-task-id requires controller planner mode"
            )
        expected_evaluation_script_sha256 = _sha256(
            args.expected_eval_script_sha256, "evaluation script"
        )
    expected_training_script_sha256 = ""
    provenance_task_id = ""
    expected_provenance_script_sha256 = ""
    if controller_mode:
        if args.expected_training_script_sha256:
            raise ValueError(
                "controller mode forbids a single training script pin; use the "
                "sealed per-subject provenance task"
            )
        provenance_task_id = _task_id(
            provenance_task_id_value, "formal training provenance task"
        )
        expected_provenance_script_sha256 = _sha256(
            provenance_script_pin_value, "formal training provenance script"
        )
    else:
        if provenance_task_id_value or provenance_script_pin_value:
            raise ValueError(
                "training provenance task options require controller planner mode"
            )
        expected_training_script_sha256 = _sha256(
            args.expected_training_script_sha256,
            "training script",
        )
    expected_source_dataset_id = _task_id(
        args.expected_source_dataset_id,
        "source dataset",
    )
    expected_source_archive_sha256 = _sha256(
        args.expected_source_archive_sha256,
        "source archive",
    )
    expected_training_source_dataset_id = _task_id(
        args.expected_training_source_dataset_id or expected_source_dataset_id,
        "training source dataset",
    )
    expected_training_source_archive_sha256 = _sha256(
        args.expected_training_source_archive_sha256 or expected_source_archive_sha256,
        "training source archive",
    )
    expected_training_dataset_id = _task_id(
        args.expected_training_dataset_id,
        "training dataset",
    )
    expected_predecessor_task_id = _task_id(
        args.expected_predecessor_task_id,
        "predecessor task",
    )
    if (
        expected_source_dataset_id != EXPECTED_SOURCE_DATASET_ID
        or expected_source_archive_sha256 != EXPECTED_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError(
            "formal 1337 evaluation must use the sealed old source revision"
        )
    if controller_mode and (
        args.expected_training_source_dataset_id
        or args.expected_training_source_archive_sha256
    ):
        raise ValueError(
            "controller mode forbids a single training source pin; use the sealed "
            "per-subject source revision map"
        )
    if not controller_mode and (
        expected_training_source_dataset_id != expected_source_dataset_id
        or expected_training_source_archive_sha256
        != expected_source_archive_sha256
    ):
        raise ValueError(
            "legacy formal training and evaluation must use the same sealed source"
        )
    training_provenance_policies: TrainingProvenancePolicies | dict[
        str, dict[str, object]
    ] = {}
    if controller_mode:
        if bool(authoritative_plan_producer_id_raw) != bool(
            authoritative_plan_seal_raw
        ):
            raise ValueError(
                "authoritative evaluation plan producer and seal must be set together"
            )
        if amendment_requested and (
            not all(bool(value) for value in amendment_pins.values())
            or not authoritative_plan_producer_id_raw
            or not authoritative_plan_seal_raw
        ):
            raise ValueError(
                "all amended evaluation plan pins and producer/seal are required together"
            )
        if explicit_amendment_requested and not all(
            bool(value) for value in explicit_amendment_pins.values()
        ):
            raise ValueError("all explicit amended evaluation plan pins are required")
        if amendment_requested != explicit_amendment_requested:
            raise ValueError(
                "authoritative and explicit amended evaluation plan pins must be set together"
            )
        if runtime_recovery_requested and not all(
            bool(value) for value in runtime_recovery_pins.values()
        ):
            raise ValueError("all exact evaluation runtime recovery pins are required")
        if runtime_recovery_requested != amendment_requested:
            raise ValueError(
                "runtime recovery and amended evaluation plan pins must be set together"
            )
        if runtime_recovery_requested:
            for name, value in runtime_recovery_pins.items():
                _sha256(value, f"exact evaluation runtime recovery {name}")
        if amendment_requested and explicit_amendment_pins != {
            "producer": authoritative_plan_producer_id_raw,
            "receipt": amendment_pins["receipt"],
            "revised_plan": authoritative_plan_seal_raw,
            "evidence": amendment_pins["amendment_evidence"],
            "worker": amendment_pins["worker_evidence"],
            "task_ids": amendment_pins["task_ids"],
        }:
            raise ValueError("explicit amended evaluation plan pin mismatch")
        if evaluation_plan_json:
            raise ValueError(
                "controller planner mode creates its evaluation plan automatically"
            )
        controller_task_id = _task_id(
            controller_task_id_value,
            "training controller task",
        )
        controller = task_class.get_task(task_id=controller_task_id)
        manifest = _wait_for_controller_manifest(
            controller,
            deadline=deadline,
            poll_seconds=args.poll_seconds,
            metadata_only_artifact_gate=args.metadata_only_artifact_gate,
        )
        training = _parse_training_manifest(json.dumps(manifest))
        training_provenance_policies = _consume_training_provenance(
            task_class=task_class,
            provenance_task_id=provenance_task_id,
            expected_provenance_script_sha256=(expected_provenance_script_sha256),
            controller=controller,
            controller_task_id=controller_task_id,
            manifest=manifest,
            training=training,
            metadata_only_artifact_gate=args.metadata_only_artifact_gate,
        )
        # Keep the provenance authority check ahead of every planner/release
        # contract. A malformed successor argument must never mask a rejected
        # P artifact or permit evaluation task discovery/cloning first.
        candidate_release_blockers = _parse_candidate_release_blockers(
            blocker_contract_raw
        )
        authoritative_entries = None
        if authoritative_plan_producer_id_raw:
            if amendment_requested:
                authoritative_entries = _load_amended_evaluation_plan(
                    task_class=task_class,
                    producer_task_id=authoritative_plan_producer_id_raw,
                    expected_plan_seal_sha256=authoritative_plan_seal_raw,
                    expected_receipt_seal_sha256=amendment_pins["receipt"],
                    expected_producer_source_sha256=(
                        amendment_pins["producer_source"]
                    ),
                    expected_amendment_evidence_seal_sha256=(
                        amendment_pins["amendment_evidence"]
                    ),
                    expected_worker_evidence_seal_sha256=(
                        amendment_pins["worker_evidence"]
                    ),
                    expected_task_ids_sha256=amendment_pins["task_ids"],
                    expected_project_id=training_provenance_policies.project_id,
                    controller_task_id=controller_task_id,
                    provenance_task_id=provenance_task_id,
                    template_task_id=template_task_id,
                )
            else:
                authoritative_entries = _load_authoritative_evaluation_plan(
                    task_class=task_class,
                    producer_task_id=authoritative_plan_producer_id_raw,
                    expected_seal_sha256=authoritative_plan_seal_raw,
                    expected_project_id=training_provenance_policies.project_id,
                    controller_task_id=controller_task_id,
                    provenance_task_id=provenance_task_id,
                    template_task_id=template_task_id,
                )
        evaluation_plan = create_or_validate_evaluation_plan(
            task_class=task_class,
            controller_task_id=controller_task_id,
            template_task_id=template_task_id,
            expected_project_id=training_provenance_policies.project_id,
            training=training,
            worker_queues=_planner_worker_queues(
                getattr(
                    args,
                    "evaluation_worker_queues",
                    "GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090",
                )
            ),
            expected_script_sha256=expected_evaluation_script_sha256,
            authoritative_entries=authoritative_entries,
        )
        if authoritative_entries is not None:
            expected_evaluation_script_sha256 = (
                _authoritative_evaluation_script_sha256(template)
            )
        _publish_evaluation_plan(
            current_task,
            controller_task_id=controller_task_id,
            template_task_id=template_task_id,
            entries=evaluation_plan,
            provenance_policies=training_provenance_policies,
        )
        dependencies = build_dependency_plan(
            json.dumps(manifest),
            json.dumps(evaluation_plan),
        )
    elif args.training_manifest_json is not None:
        if (
            blocker_contract_raw
            or authoritative_plan_producer_id_raw
            or authoritative_plan_seal_raw
            or amendment_requested
            or explicit_amendment_requested
        ):
            raise ValueError(
                "candidate blockers and authoritative plan recovery require "
                "controller planner mode"
            )
        if not evaluation_plan_json:
            raise ValueError(
                "--evaluation-plan-json is required with --training-manifest-json"
            )
        dependencies = build_dependency_plan(
            args.training_manifest_json,
            evaluation_plan_json,
        )
    else:
        if (
            blocker_contract_raw
            or authoritative_plan_producer_id_raw
            or authoritative_plan_seal_raw
            or amendment_requested
            or explicit_amendment_requested
        ):
            raise ValueError(
                "candidate blockers and authoritative plan recovery require "
                "controller planner mode"
            )
        candidate_release_blockers = ()
        if evaluation_plan_json:
            raise ValueError("--evaluation-plan-json requires --training-manifest-json")
        dependencies = _parse_dependencies(
            str(args.dependencies_json),
            default_training_predecessor_task_id=(expected_predecessor_task_id),
        )
    dependencies_for_release = _dependencies_in_release_order(dependencies)
    formal_evaluation_task_ids = {
        dependency["evaluation_task_id"] for dependency in dependencies
    }
    completed: set[str] = set()
    suite_runtime_identity: dict[str, str] | None = None
    previous_snapshot = ""
    while len(completed) != len(dependencies):
        release_consumed_by_queue: set[str] = set()
        active_evaluation_count = {queue: 0 for queue in EXPECTED_QUEUE_IDS}
        blockers_ready = True
        core_phase_ready = set(FORMAL_EVALUATION_RELEASE_PRIORITY) <= completed
        candidate_phase_ready = not controller_mode
        candidate_phase_state: dict[str, object] = {
            "policy": "formal_core_six_then_candidate_five_then_remaining_formal",
            "ready": candidate_phase_ready,
            "state": (
                "legacy_dependency_mode"
                if not controller_mode
                else "waiting_for_formal_core"
            ),
            "entries": [],
        }
        if controller_mode:
            core_phase_ready = _formal_core_phase_status_ready(
                task_class, dependencies=dependencies_for_release
            )
            _blocker_snapshot, blockers_ready = (
                _candidate_release_blocker_snapshot(
                    task_class,
                    blockers=candidate_release_blockers,
                    expected_project_id=training_provenance_policies.project_id,
                )
            )
            candidate_phase_state, candidate_phase_ready = (
                _candidate_evaluation_phase_snapshot(
                    task_class,
                    expected_project_id=(
                        training_provenance_policies.project_id
                    ),
                    core_phase_ready=core_phase_ready,
                )
            )
        snapshot: list[dict[str, object]] = []
        for dependency in dependencies_for_release:
            subject = dependency["subject"]
            provenance_policy = training_provenance_policies.get(subject, {})
            training = task_class.get_task(task_id=dependency["training_task_id"])
            evaluation = task_class.get_task(task_id=dependency["evaluation_task_id"])
            training_runtime = _invariant_runtime_identity(
                training,
                context=f"training dependency {subject}",
            )
            evaluation_runtime = _invariant_runtime_identity(
                evaluation,
                context=f"evaluation task {subject}",
            )
            if evaluation_runtime != training_runtime:
                raise RuntimeError(
                    f"evaluation {subject} shared runtime parameters mismatch"
                )
            if suite_runtime_identity is None:
                suite_runtime_identity = training_runtime
            elif training_runtime != suite_runtime_identity:
                raise RuntimeError(
                    f"training dependency {subject} shared runtime parameters mismatch"
                )
            expected_training_revision = str(
                provenance_policy.get("source_revision") or "old"
            )
            expected_training_source = _expected_source_runtime_identity(
                expected_training_revision
            )
            if _source_runtime_identity(
                training, context=f"training dependency {subject}"
            ) != expected_training_source:
                raise RuntimeError(
                    f"training dependency {subject} source revision mismatch"
                )
            if _source_runtime_identity(
                evaluation, context=f"evaluation task {subject}"
            ) != _expected_source_runtime_identity("old"):
                raise RuntimeError(
                    f"evaluation {subject} did not use the sealed old source revision"
                )
            training_status = _status(training)
            worker_queue = dependency["queue"]
            evaluation_status = _require_evaluation_identity(
                evaluation,
                dependency=dependency,
                expected_script_sha256=expected_evaluation_script_sha256,
                expected_source_dataset_id=expected_source_dataset_id,
                expected_source_archive_sha256=expected_source_archive_sha256,
                expected_training_dataset_id=expected_training_dataset_id,
            )
            training_ready = _require_completed_training(
                training,
                subject=subject,
                task_id=dependency["training_task_id"],
                expected_script_sha256=(
                    provenance_policy.get("script_sha256")
                    or expected_training_script_sha256
                ),
                expected_source_dataset_id=str(
                    provenance_policy.get("source_dataset_id")
                    or expected_training_source_dataset_id
                ),
                expected_source_archive_name=str(
                    provenance_policy.get("source_archive_name")
                    or EXPECTED_SOURCE_ARCHIVE_NAME
                ),
                expected_source_archive_bytes=int(
                    provenance_policy.get("source_archive_bytes")
                    or EXPECTED_SOURCE_ARCHIVE_BYTES
                ),
                expected_source_archive_sha256=(
                    str(
                        provenance_policy.get("source_archive_sha256")
                        or expected_training_source_archive_sha256
                    )
                ),
                expected_training_dataset_id=expected_training_dataset_id,
                expected_predecessor_task_id=(
                    dependency["training_predecessor_task_id"]
                ),
                expected_parent_controller_task_id=provenance_policy.get(
                    "parent_controller_task_id", ""
                ),
                expected_run_contract_content_sha256=provenance_policy.get(
                    "run_contract_content_sha256", ""
                ),
                expected_final_checkpoint_contract_content_sha256=(
                    provenance_policy.get(
                        "final_checkpoint_contract_content_sha256", ""
                    )
                ),
                expected_final_model_id=dependency.get("training_model_id", ""),
                expected_final_checkpoint_sha256=dependency.get(
                    "training_checkpoint_sha256", ""
                ),
                expected_final_model_url=dependency.get("training_model_url", ""),
                expected_initialization_audit_sha256=(
                    provenance_policy.get(
                        "common_teacher_initialization_audit_content_sha256", ""
                    )
                    or dependency.get("training_initialization_audit_sha256", "")
                ),
                metadata_only_artifact_gate=args.metadata_only_artifact_gate,
            )
            if evaluation_status in {"queued", "in_progress", "completed"}:
                if not _formal_phase_allows_release(
                    subject, candidate_phase_ready=candidate_phase_ready
                ):
                    raise RuntimeError(
                        f"evaluation {subject} was released before the candidate "
                        "evaluation phase completed"
                    )
                if not blockers_ready:
                    raise RuntimeError(
                        f"evaluation {subject} was released before all candidate "
                        "training blockers completed"
                    )
                if not training_ready:
                    raise RuntimeError(
                        f"evaluation {subject} was released before training completed"
                    )
                _require_released_input_model(
                    evaluation,
                    training,
                    subject=subject,
                    status=evaluation_status,
                )
                if evaluation_status == "completed":
                    _require_completed_evaluation_metrics(
                        evaluation,
                        dependency=dependency,
                    )
                    completed.add(subject)
                else:
                    active_evaluation_count[worker_queue] += 1
            elif evaluation_status == "created":
                resource_gate: dict[str, object] | None = None
                if (
                    training_ready
                    and blockers_ready
                    and _formal_phase_allows_release(
                        subject,
                        candidate_phase_ready=candidate_phase_ready,
                    )
                    and _claim_single_queue_release(
                        release_consumed_by_queue,
                        queue_name=worker_queue,
                    )
                ):
                    # Claim the queue before probing it.  Even a blocked first
                    # task prevents a later task on the same queue from
                    # overtaking it during this reconciliation pass.
                    resource_gate = _formal_resource_gate_snapshot(
                        task_class,
                        formal_evaluation_task_ids=(formal_evaluation_task_ids),
                        target_queue_id=EXPECTED_QUEUE_IDS[worker_queue],
                    )
                    if resource_gate["ready"] is True:
                        evaluation.add_tags("dependency-released")
                        response = task_class.enqueue(
                            task=evaluation,
                            queue_name=dependency["queue"],
                        )
                        if not response:
                            raise RuntimeError(
                                f"failed to enqueue evaluation for {subject}"
                            )
                        evaluation_status = _status(evaluation)
                        for _ in range(10):
                            if evaluation_status in {
                                "queued",
                                "in_progress",
                                "completed",
                            }:
                                break
                            time.sleep(1.0)
                            evaluation_status = _status(evaluation)
                        if evaluation_status not in {
                            "queued",
                            "in_progress",
                            "completed",
                        }:
                            raise RuntimeError(
                                f"evaluation for {subject} was not released"
                            )
                        evaluation_status = _require_evaluation_identity(
                            evaluation,
                            dependency=dependency,
                            expected_script_sha256=(
                                expected_evaluation_script_sha256
                            ),
                            expected_source_dataset_id=(
                                expected_source_dataset_id
                            ),
                            expected_source_archive_sha256=(
                                expected_source_archive_sha256
                            ),
                            expected_training_dataset_id=(
                                expected_training_dataset_id
                            ),
                        )
                snapshot_row: dict[str, object] = {
                    "subject": subject,
                    "training": training_status,
                    "evaluation": evaluation_status,
                    "queue": dependency["queue"],
                    "phase_release_allowed": _formal_phase_allows_release(
                        subject,
                        candidate_phase_ready=candidate_phase_ready,
                    ),
                    "candidate_phase_ready": candidate_phase_state["ready"],
                }
                if resource_gate is not None:
                    snapshot_row["resource_gate"] = resource_gate
                snapshot.append(snapshot_row)
                continue
            snapshot.append(
                {
                    "subject": subject,
                    "training": training_status,
                    "evaluation": evaluation_status,
                    "queue": dependency["queue"],
                    "phase_release_allowed": _formal_phase_allows_release(
                        subject,
                        candidate_phase_ready=candidate_phase_ready,
                    ),
                    "candidate_phase_ready": candidate_phase_state["ready"],
                }
            )
        if any(active_evaluation_count.values()):
            live_workers = _live_worker_rows()
            for queue_name, active_count in active_evaluation_count.items():
                if active_count == 0:
                    continue
                live_capacity = _live_disjoint_worker_capacity(
                    live_workers,
                    target_queue_id=EXPECTED_QUEUE_IDS[queue_name],
                )
                if active_count > live_capacity:
                    raise RuntimeError(
                        f"formal evaluation concurrency on {queue_name} exceeds "
                        f"its {live_capacity} live disjoint workers"
                    )
        snapshot_json = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        if snapshot_json != previous_snapshot:
            print(snapshot_json, flush=True)
            previous_snapshot = snapshot_json
        if len(completed) == len(dependencies):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("dependency watcher timed out")
        time.sleep(args.poll_seconds)


def main() -> int:
    Task.init()
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
