#!/usr/bin/env python3
"""Plan or execute the sealed P1 two-method, three-seed A100 ClearML matrix.

The default invocation is local and read-only: it prints a sealed execution
plan and does not import ClearML.  Remote task creation and enqueueing require
both ``--execute`` and the exact acknowledgement token.  The executable scope
is fixed to ResilientV2X and BEVFusion at seed1, seed2, and seed3.  Seed1 is
always retrained on A100 and never reuses the historical RTX 5090 checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path


EXECUTION_TOKEN = "ENQUEUE_P1_A100_MULTISEED_6_TRAIN_6_EVAL"
PREFLIGHT_RECEIPT_TYPE = "resilient_v2x_p1_a100_multiseed_preflight"
JOURNAL_DOCUMENT_TYPE = "resilient_v2x_p1_a100_multiseed_journal"
MANIFEST_DOCUMENT_TYPE = "resilient_v2x_p1_a100_multiseed_execution"
LOCAL_MUTEX_DOCUMENT_TYPE = "resilient_v2x_p1_a100_local_mutex_journal"
MUTEX_TAG = "p1-a100-scoped-mutex"
EXECUTION_KEY_TAG_PREFIX = "resilient-v2x-p1-a100-execution:"
CONTROLLER_TAG = "resilient-v2x-p1-a100-controller-lease"
ORPHAN_TAG = "resilient-v2x-p1-a100-quarantined-orphan"
PLAN_ARTIFACT = "p1_a100_multiseed_plan"
PINSET_ARTIFACT = "p1_a100_multiseed_pinset"
JOURNAL_ARTIFACT = "p1_a100_multiseed_execution_journal"
MANIFEST_ARTIFACT = "p1_a100_multiseed_execution_manifest"
PROJECT_ID = "6e43f972e5ea4cee901a7c8855fce8cd"
PROJECT_NAME = "ResilientV2X/Training"
FILES_SERVER_URI = "http://10.100.34.118:8081"
QUEUE_NAME = "GPU4-A100"
QUEUE_ID = "9350f33af13a448da8339eb7bea52fdf"
ALLOWED_WORKER_IDS = frozenset(
    {
        "10.100.34.18-A100:gpu0,1,2,3",
        "10.100.34.18-A100:gpu4,5,6,7",
    }
)
OVERLAPPING_GPU8_WORKER_ID = "10.100.34.18-A100:gpu0,1,2,3,4,5,6,7"
OVERLAPPING_GPU8_QUEUE_NAME = "GPU8-A100"
OVERLAPPING_GPU8_QUEUE_ID = "b4848a934ecc4aa6b48a80e6fc8ec602"
GPU8_FREEZE_TAG = "force_workers:off"
GPU8_ENABLE_TAG = "force_workers:on"

# The A100 executor is allowed to start only after the exact sealed RTX 5090
# controller has reached the terminal retirement state produced by
# ``clearml_p1_5090_retirement.py``.  These bindings are repeated here instead
# of importing or invoking the legacy executor, so this executable remains an
# independent, narrowly scoped 6+6 controller.
LEGACY_EXECUTION_KEY = (
    "63edb447d0c7ce7d82071bffbac15ec111e37ff9a747453bec9b93fc744060cd"
)
LEGACY_EXECUTION_KEY_TAG_PREFIX = "resilient-v2x-p1-execution:"
LEGACY_CONTROLLER_TAG = "resilient-v2x-p1-controller-lease"
LEGACY_ORPHAN_TAG = "resilient-v2x-p1-quarantined-orphan"
LEGACY_RETIREMENT_TAG = "resilient-v2x-p1-retired-for-homogeneous-a100"
LEGACY_RETIREMENT_INTENT_ARTIFACT = "p1_5090_retirement_intent"
LEGACY_RETIREMENT_ARTIFACT = "p1_5090_retirement_manifest"
LEGACY_CONTROLLER_ARTIFACTS = frozenset(
    {
        "p1_multiseed_plan",
        "p1_multiseed_pinset",
        "p1_multiseed_execution_journal",
        LEGACY_RETIREMENT_INTENT_ARTIFACT,
        LEGACY_RETIREMENT_ARTIFACT,
    }
)
LEGACY_RETIREMENT_DOCUMENT_TYPE = "resilient_v2x_p1_5090_retirement"
INTENT_TARGET_A100_EXECUTOR_SHA256 = (
    "5ffe844378b99aaf460e059d29c4c6e9ff6e9ba614a5894956a81c59cf7388fe"
)
A100_EXECUTOR_AMENDMENT_REASON = "clearml_dequeue_retains_historical_execution_queue"
LEGACY_EXECUTOR_SHA256 = (
    "709d7b272d6c575f8d81f2d6eeda3a42f44b13fec3da6fb622d5a3e76b2a3b1b"
)
LEGACY_CONTROLLER_ID = "3dd5ba86053f49b59477c6a41219600b"
LEGACY_CHILD_ID = "4713942e362c40b9b093b00845a74511"
LEGACY_CHILD_TASK_KEY = "train-r01-s02-resilient_v2x"
LEGACY_TASK_KEYS = tuple(
    f"{phase}-r01-s{seed_index:02d}-{subject}"
    for seed_index in (2, 3)
    for subject in ("resilient_v2x", "bevfusion")
    for phase in ("train", "eval")
)
LEGACY_QUEUE_NAME = "GPU4-5090"
LEGACY_QUEUE_ID = "5a84454c072349069e7b61af38637c6d"
LEGACY_MUTEX_QUEUE_ID = "602fbb124f8d450491100c46959a6a6d"
LEGACY_MUTEX_NAME = (
    "__p1_resilient_v2x_mutex_"
    "63edb447d0c7ce7d82071bffbac15ec111e37ff9a747453bec9b93fc744060cd"
)
LEGACY_MUTEX_JOURNAL_NAME = ".p1-scoped-multiseed-mutex.json"

A100_RUNTIME_CONTRACT = {
    "gpu_count": 4,
    "gpu_type_normalized": ["NVIDIA A100-PCIE-40GB"] * 4,
    "gpu_memory_normalized": ["40GB"] * 4,
    "task_runtime_gpu_type_representation": "comma_delimited_string",
    "task_runtime_gpu_memory_representation": "comma_delimited_string",
    "gpu_compute_capability": [8, 0],
    "gpu_compute_capability_source": "native_build_manifest",
    "gpu_driver_version": "595.84",
    "gpu_driver_cuda_version": "13.2",
    "python_version": "3.12.12",
    "python_exec_allowlist": [
        "/opt/miniforge3/bin/python3.12",
        "/opt/resilient-v2x-5090/bin/python",
    ],
    "source_d_child_python_exec_preferred": "/opt/miniforge3/bin/python3.12",
    "os": "Linux-7.0.0-28-generic-x86_64-with-glibc2.31",
    "hostname": "ubuntu-a100",
    "torch": "2.10.0+cu128",
    "torch_cuda": "12.8",
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
    "mmdet": "3.2.0",
    "mmdet3d": "1.3.0",
}

SELECTOR_TASK_ID = "d7ea54ce540d4b0486904c5910100885"
SELECTOR_PARENT_TASK_ID = "dd287fe58e264bcca48fe222b6981048"
SELECTOR_SCRIPT_SHA256 = (
    "4aa9d7ed9b55594342c54be212cbc0d096b54cfacee60f4f5f4611318e58f2c9"
)
SELECTOR_ARTIFACT_HASHES = {
    "final_selector_formal_inputs": (
        "a85b88e7ee3dcbbf5f516c4442544ff55126958305fd8b68f7ed65527bc43005"
    ),
    "formal_candidate_selection": (
        "9f77afae4ac06ab054e76be02e7384f9cfdef7c55015634e3dce8aa73a9acb7e"
    ),
}

SOURCE_D_TASK_ID = "3938784889d74c1f861db2b20f10ec5b"
SOURCE_D_PARENT_TASK_ID = "95e72da24d464ab08d117dedabd6652e"
SOURCE_C_PARENT_TASK_ID = "6525107e60ae4104a2800731d74ecd4e"
SOURCE_D_PRODUCER_SHA256 = (
    "d5b759f38d39a9f349ab6e716c07fda53eb3e1635687ce4a077e0405920ffeec"
)
SOURCE_D_BUILDER_SHA256 = (
    "904fafd08d710b03b62bc57140121f2a546ada8fa2fb8763e6db8a44b9f8f7e7"
)
SOURCE_C_SCRIPT_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
SOURCE_D_SCRIPT_SHA256 = (
    "e7a9ab0fb05339223cf2c18c52eb72652c311733bf96d1058aa7a769096cf8c3"
)
SOURCE_D_EQUIVALENCE_SHA256 = (
    "1156fe53f2fe924f91c1c6b50b6b21090d98cd2d74840d6d6f5a358316420433"
)
SOURCE_C_SNAPSHOT_SEAL_SHA256 = (
    "c6426dc3734db478934809217457bffaba4d571ac597132f014de6b0a6e08cb8"
)
SOURCE_D_SCRIPT_SEAL_SHA256 = (
    "49d3d20dcb358a6c483128e20f231767d3fe5c1eae33a2192cd1717d1a2d19b6"
)
SOURCE_D_RECEIPT_SEAL_SHA256 = (
    "1c4b3cfeb617eacc6aa1f7d194dee687c6cf51f47e6adf4987eaaed6d796c62d"
)
SOURCE_D_TRANSFORMATION_ID = "source-c-to-source-d-explicit-seed-evidence-v2"
SOURCE_D_ARTIFACT_HASHES = {
    "formal_source_c_snapshot": (
        "ecf2cf2f50f5e63c1e568be2fa6bb2053060c2b61330306f231ade2f17327b00"
    ),
    "formal_source_d_script": (
        "935e37fa260212e07d3aa84823a276dbeada71bf1b5ae27bc617e7db3859d42a"
    ),
    "formal_source_d_equivalence": (
        "54998e80987a510d99624130e3bc5ca9b12f42707f41af743786bf8d1c9832be"
    ),
    "formal_source_d_evidence_receipt": (
        "94a70498e098b05fab1d68e2fe86b45f3e0ac1531e5fdd049c49ed2a68608da8"
    ),
}
SOURCE_D_REPLACEMENT_NAMES = (
    "seal_evidence_security_imports",
    "declare_seed_contract",
    "declare_controlled_evidence_contract",
    "declare_controlled_evidence_receipts",
    "add_training_seed_type",
    "add_training_seed_cli",
    "validate_training_seed_argument",
    "bind_ddp_training_seed_parameter",
    "bind_ddp_cfg_seed_overrides",
    "bind_baseline_plan_seed_parameter",
    "replace_baseline_plan_seed_literal",
    "validate_training_overlay_protocol_seed",
    "validate_baseline_plan_seed_contract",
    "validate_contract_seed_sources",
    "record_explicit_seed_contract",
    "stage_and_verify_controlled_evidence",
    "stage_after_controlled_runner",
    "upload_only_sealed_controlled_evidence",
    "pass_seed_to_baseline_plan",
    "pass_seed_to_baseline_validation",
    "pass_seed_to_ddp_training",
    "forward_seed_to_stage_runner",
)

TEACHER_GATE_TASK_ID = "f041d43e48c14ba4a4562281860d13f6"
TEACHER_GATE_SCRIPT_SHA256 = (
    "432514f19f4b663d1023e1ad2faed28c8b2c983626660a9c3bfd011302edb557"
)
TEACHER_GATE_ARTIFACT_HASH = (
    "d5aa618377d87b5cb2fe498b501ac731a984f583804d6c1876b864d64f38abd5"
)
TEACHER_GATE_CONTENT_SHA256 = (
    "25efde30998e94810905535c36f1cee3e3f828cd93f3ce3911c186bd07016543"
)
TEACHER_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
TEACHER_SCRIPT_SHA256 = (
    "4dfe2e9d2ee3076df1679818211f40b7c2ebcbc73efb4cdae2f230c971485b67"
)
TEACHER_MODEL_ID = "d962f6bae8474260b54e170a7a5f0418"
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
TEACHER_ARTIFACT_HASHES = {
    "run_contract": (
        "8fec7de88e5ebea34e1e3d7bf6226dd19d8be26be8fecfd64b2875f0507d4c8b"
    ),
    "teacher_checkpoint_contract": (
        "c017041480d23b94520dcbb31d5e55a0ea5eda04aaf6865cbf01bbac0fa1dd76"
    ),
}

PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
TRAINING_OVERLAY_PROTOCOL_SEED = 20_250_218
TRAINING_SEED_INDICES = (1, 2, 3)
TRAINING_SEEDS = (
    TRAINING_OVERLAY_PROTOCOL_SEED,
    2_856_275_757,
    2_517_382_461,
)
SUBJECTS = ("resilient_v2x", "bevfusion")
SUBJECT_CONFIG_SHA256 = {
    "resilient_v2x": (
        "19ae2d94eec8e8123293bd6f536710aa6c8b3efe31ebf651e28b6ee728fb8205"
    ),
    "bevfusion": ("b51cfd386185b91d319721917e9ac0de85bb3253d6abc48cf4e4d163312f3922"),
}
CONDITION_IDS = tuple(
    f"delay_{delay:03d}_{condition}"
    for delay in (0, 100, 200, 300)
    for condition in ("full", "l_fail", "c_fail")
)

SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
SOURCE_ARCHIVE_BYTES = 1_222_481
SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
NATIVE_BUILD_TASK_ID = "9055c0d3c4dd450c8a75dddfb21a56bd"
NATIVE_BUNDLE_BYTES = 753_382_966
NATIVE_BUNDLE_SHA256 = (
    "19b8e7f5edc8216d4b43cb17854dccafe4fc9fe46995a88803e6342eeaa22b21"
)
BUILD_MANIFEST_SHA256 = (
    "21c6ab7a6e9a2823ba111289a42e7f882c5f4ba02c73175106ff251fe5864a43"
)
NATIVE_BUILD_MANIFEST_ARTIFACT = "rtx5090_build_manifest"
NATIVE_BUILD_PIP_FREEZE_ARTIFACT = "rtx5090_pip_freeze"
NATIVE_BUILD_MANIFEST_BYTES = 2_501
MMCV_WHEEL_NAME = "mmcv-2.1.0-cp312-cp312-linux_x86_64.whl"
MMCV_WHEEL_BYTES = 12_933_197
MMCV_WHEEL_SHA256 = "388a6de5cdcdde67e6854a80e8f29f8f978d969ac7b1f9db1eedd738fe873735"
TORCH_ARCH_LIST = ("sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120")
PIP_FREEZE_REQUIRED_LINES = frozenset(
    {
        "clearml==2.1.4",
        "clearml-agent==3.0.3",
        (
            "mmcv @ file:///tmp/resilient-v2x-wheels/"
            f"{MMCV_WHEEL_NAME}#sha256={MMCV_WHEEL_SHA256}"
        ),
        "mmengine==0.10.7",
        "mmdet==3.2.0",
        "mmdet3d==1.3.0",
        "numpy==1.26.4",
        "nvidia-cuda-cupti-cu12==12.8.90",
        "nvidia-cuda-nvrtc-cu12==12.8.93",
        "nvidia-cuda-runtime-cu12==12.8.90",
        "opencv-python==4.11.0.86",
        "opencv-python-headless==4.11.0.86",
        "torch==2.10.0",
        "torchvision==0.25.0",
    }
)
BASE_IMAGE_MANIFEST_DIGEST = (
    "sha256:dbc586035fffb2bc030e807290d43e8d4edf44ee864fa5c832db44ed099fc415"
)
CONTAINER_IMAGE = (
    "gitlab.zhht.ai.com:5000/aitech/model_infer:"
    "py-3.12-cuda-12.6.2-torch-2.10-ultralytics-8.4.13-dvc-v2-onnx-clearml"
)
CONTAINER_ARGUMENTS = (
    "--user 0 --shm-size 16g --network host --add-host 509gpu:127.0.0.1 "
    "--env PYTHONSAFEPATH=1 --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 "
    f"--env RESILIENT_V2X_CONTAINER_IMAGE_DIGEST={BASE_IMAGE_MANIFEST_DIGEST} "
    "--env NVIDIA_TF32_OVERRIDE=0 --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "
    "-e CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1"
)
CONTROLLED_EVALUATOR_SHA256 = (
    "d233e054f2b608bc441de25833fb89d117197d841752812a0054e0514995ad36"
)
MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
OVERLAY_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
SAMPLE_IDS_SHA256 = "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
SAMPLE_COUNT = 1_337
GROUND_TRUTH_COUNT = 11_330

ANCHOR_TRAINING_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
ANCHOR_EVALUATION_SCRIPT_SHA256 = (
    "982f607cc7c7cc34872c68ce9c662bb7635d66d15bc7e0fc8cfc9ce70d7c3c3c"
)
ANCHORS = {
    "resilient_v2x": {
        "training_task_id": "54d28bc513794051810fd383140ae96e",
        "model_id": "ea23e399bc554b07bc2ce07e6b68700e",
        "checkpoint_sha256": (
            "f6a5683a1df7126a3069edce577177eefb80208d9548ba3b61a8907bbf148a67"
        ),
        "evaluation_task_id": "7deb18e532324850bee1fb4279a838b7",
    },
    "bevfusion": {
        "training_task_id": "48cf33da96114669b56a7646582c2835",
        "model_id": "293c6302e7b941ca95936b2e8ba18c2d",
        "checkpoint_sha256": (
            "bb35801fdf93f890f31f680c075aedfa9c225408e74a5ec904a0e93c0fc8bc9f"
        ),
        "evaluation_task_id": "c3b87760b78742cb8e7de3a506a08999",
    },
}

ANCHOR_TRAINING_ARTIFACT_HASHES = {
    "resilient_v2x": {
        "best_checkpoint_contract": (
            "fbd2abd4ebed1c3feecd0135e7bf23f22867c6214f485519934f06ec28f441a5"
        ),
        "common_teacher_initialization_audit": (
            "bf67dcdd74fad552dde56e371d183e9e7612a3b000a2df8b2f36d50724cd79d3"
        ),
        "final_checkpoint_contract": (
            "0c497a5f8b6a73c4f5cf50cdac7be9543f9851594b24664f5dd0b8ffe37a6d08"
        ),
        "run_contract": (
            "c296b999c348b9aac625e31c99b06ad8a36addba1abda23dc63be93796e0bea4"
        ),
    },
    "bevfusion": {
        "baseline_dry_run_plan": (
            "52ed4264c652afdb5e9d00dde1c94167546b30ec9aff838815e729dd350a9eed"
        ),
        "baseline_resolved_config": (
            "6b5272d783259c65e05463d0dfce2f7eb8e858c70ba44f1ced9d45d497580a6c"
        ),
        "best_checkpoint_contract": (
            "58f8f406323e955d332ddc944b4d04770b5b199e6f514f6ed439a10187d0b4a0"
        ),
        "common_teacher_initialization_audit": (
            "699d6be8aabefbb898b901a98822096bcd87da2e5840da47dee8860de3783531"
        ),
        "final_checkpoint_contract": (
            "f451ad01a6a5c6d566ec8d86d97cd93a0b2b1e452be02fd1463ce104a4e989f8"
        ),
        "run_contract": (
            "449c600ac1359b06e11f9f9fd17ed84a3e7617199078cd3edb383a47984c9eff"
        ),
    },
}

ANCHOR_EVALUATION_ARTIFACT_HASHES = {
    "resilient_v2x": {
        "controlled_baseline_evidence": (
            "932e6bed7a4749801f5212e434ac5a4cc712bab58203218cab70ae825e56c535"
        ),
        "controlled_baseline_metrics": (
            "5a5f111876544de2403ef18ca3b7cb616ba6047277119a4b3461ad5cc0a56a65"
        ),
        "evaluation_plan": (
            "1dfdb3a9ecbb40cd58df5c90e49776d7a2cfa350dfefdd68afc00e71f200403b"
        ),
        "run_contract": (
            "b921be133adaf64503206ffc99fb616e96595f5964d9354f91ec1af443df1a7e"
        ),
    },
    "bevfusion": {
        "controlled_baseline_evidence": (
            "6f956880f088f7befe311c0db124cbdf08c9a0c28fffeb62e07ec72e0efabc4a"
        ),
        "controlled_baseline_metrics": (
            "43977ebdae38e74882bb659c53d4c10e2cb20eba1a7a5c77b2514bccabd21c1f"
        ),
        "evaluation_plan": (
            "d88f64af49e1c4e098518f61f25a0908af1552ebee4622c64bb85058ccc41df7"
        ),
        "run_contract": (
            "f8dd1db1afaecb6716647b5021a1010981c427b4d804d9ddcff59c1fd836bf23"
        ),
    },
}

SOURCE_D_SEED_CONTRACT = {
    "training_seed_cli": "--training-seed",
    "default_training_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
    "training_seed_min": 0,
    "training_seed_max": 4_294_967_295,
    "training_cfg_bindings": [
        "randomness.seed",
        "train_dataloader.sampler.seed",
        "val_dataloader.sampler.seed",
        "test_dataloader.sampler.seed",
        "train_dataloader.dataset.seed",
        "val_dataloader.dataset.seed",
        "test_dataloader.dataset.seed",
        "implementation_choices_dataset.global_seed",
    ],
    "controlled_baseline_cli": "--seed",
    "run_contract_training_seed_field": "training_seed",
    "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
    "run_contract_overlay_seed_field": "training_overlay_protocol_seed",
    "overlay_seed_source": "protocols/dair_v2/training_overlays.json:protocol_seed",
}

METRIC_KEYS = frozenset(
    {
        "resilient_v2x/car_3d_ap_r40_0.50",
        "resilient_v2x/car_3d_ap_r40_0.70",
        "resilient_v2x/car_bev_ap_r40_0.50",
        "resilient_v2x/car_bev_ap_r40_0.70",
        "resilient_v2x/car_ground_truth_count",
        "resilient_v2x/car_prediction_count",
        "resilient_v2x/diagnostic_bev_match_050_3d_iou_p50",
        "resilient_v2x/diagnostic_bev_match_050_abs_z_error_p50",
        "resilient_v2x/diagnostic_bev_match_050_count",
        "resilient_v2x/diagnostic_bev_match_050_vertical_iou_p50",
        "resilient_v2x/diagnostic_gt_height_p50",
        "resilient_v2x/diagnostic_gt_z_bottom_p50",
        "resilient_v2x/diagnostic_pred_height_p50",
        "resilient_v2x/diagnostic_pred_z_bottom_p50",
        "resilient_v2x/sample_count",
        "resilient_v2x/unsupported_sample_count",
    }
)
AP_METRIC_KEYS = frozenset(key for key in METRIC_KEYS if "_ap_r40_" in key)
UNIT_INTERVAL_METRIC_KEYS = frozenset(
    {
        "resilient_v2x/diagnostic_bev_match_050_3d_iou_p50",
        "resilient_v2x/diagnostic_bev_match_050_vertical_iou_p50",
    }
)
NONNEGATIVE_METRIC_KEYS = frozenset(
    {
        "resilient_v2x/car_prediction_count",
        "resilient_v2x/diagnostic_bev_match_050_abs_z_error_p50",
        "resilient_v2x/diagnostic_bev_match_050_count",
        "resilient_v2x/diagnostic_gt_height_p50",
        "resilient_v2x/diagnostic_pred_height_p50",
    }
)
INTEGER_METRIC_KEYS = frozenset(
    {
        "resilient_v2x/car_ground_truth_count",
        "resilient_v2x/car_prediction_count",
        "resilient_v2x/diagnostic_bev_match_050_count",
        "resilient_v2x/sample_count",
        "resilient_v2x/unsupported_sample_count",
    }
)

EVALUATION_PLAN_KEYS = {
    "baseline",
    "baseline_config",
    "baseline_config_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "conditions",
    "content_sha256",
    "data_root",
    "delays_ms",
    "evaluation_subject_type",
    "expected_ground_truth_count",
    "expected_sample_count",
    "expected_unsupported_sample_count",
    "manifest",
    "manifest_content_sha256",
    "manifest_file_sha256",
    "metrics_output",
    "overlay_index",
    "overlay_index_content_sha256",
    "overlay_index_file_sha256",
    "plan_path",
    "plan_type",
    "protocol_id",
    "runs",
    "sample_ids",
    "sample_ids_sha256",
    "schema_version",
    "split_sha256",
    "work_dir",
}
EVALUATION_PLAN_RUN_KEYS = {
    "agent_scope",
    "checkpoint_sha256_file",
    "condition",
    "condition_config",
    "condition_id",
    "delay_ms",
    "duration_ticks",
    "fault_overlay",
    "fault_overlay_sha256",
    "predictions",
    "resolved_config",
    "resolved_config_sha256",
    "transport_overlay",
    "transport_overlay_sha256",
}

ACTIVE_STATUSES = frozenset({"created", "queued", "in_progress"})
FAILURE_STATUSES = frozenset(
    {"failed", "stopped", "closed", "published", "publishing", "rejected"}
)
MAX_JSON_BYTES = 64 * 1024 * 1024
# This filename is part of the sealed Source-D and historical anchor pins.  The
# A100 execution contract is enforced independently by the exact queue/worker
# checks below; renaming the entry point would require a new Source-D artifact.
TASK_ENTRY_POINT = "clearml_5090_bootstrap.py"


class P1ExecutorError(RuntimeError):
    """Raised when the scoped plan or a ClearML binding drifts."""


class P1RecoverableTimeout(P1ExecutorError):
    """Raised after preserving an active child and resumable journal."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise P1ExecutorError(f"value is outside canonical JSON: {error}") from None


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, object]) -> str:
    detached = dict(value)
    detached.pop("seal_sha256", None)
    return _content_sha256(detached)


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise P1ExecutorError(f"{context} must be a lowercase SHA-256")
    return result


def _clearml_id(value: object, context: str) -> str:
    result = str(value or "")
    if re.fullmatch(r"[0-9a-f]{32}", result) is None:
        raise P1ExecutorError(f"{context} must be a lowercase ClearML ID")
    return result


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise P1ExecutorError(f"{context} must be an object")
    result = dict(value)
    _canonical_json(result)
    return result


def _exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    observed = set(value)
    if observed != expected:
        raise P1ExecutorError(
            f"{context} keys drifted; missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _require_equal(observed: object, expected: object, context: str) -> None:
    if observed != expected or type(observed) is not type(expected):
        raise P1ExecutorError(
            f"{context} drifted: expected {expected!r}, got {observed!r}"
        )


def _executor_sha256() -> str:
    path = Path(__file__).resolve(strict=True)
    if not path.is_file():
        raise P1ExecutorError("A100 executor source is not a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_retirement_contract() -> dict[str, object]:
    return {
        "required": True,
        "execution_key": LEGACY_EXECUTION_KEY,
        "controller_task_id": LEGACY_CONTROLLER_ID,
        "child_task_id": LEGACY_CHILD_ID,
        "child_task_key": LEGACY_CHILD_TASK_KEY,
        "legacy_queue_id": LEGACY_QUEUE_ID,
        "retirement_document_type": LEGACY_RETIREMENT_DOCUMENT_TYPE,
        "retirement_artifact": LEGACY_RETIREMENT_ARTIFACT,
        "required_final_status": "retired",
        "required_controller_status": "created",
        "required_child_status": "created",
        "required_tasks_archived": False,
        "required_tasks_quarantined": True,
        "legacy_mutex_name": LEGACY_MUTEX_NAME,
        "legacy_mutex_required_status": "available",
        "legacy_local_mutex_journal_required_status": "absent",
    }


def _a100_runtime_contract() -> dict[str, object]:
    return {
        **A100_RUNTIME_CONTRACT,
        "allowed_worker_ids": sorted(ALLOWED_WORKER_IDS),
        "overlapping_gpu8_worker_id": OVERLAPPING_GPU8_WORKER_ID,
        "overlapping_gpu8_worker_required_state": "idle",
    }


def _native_build_runtime_contract() -> dict[str, object]:
    contract: dict[str, object] = {
        "native_build_task_id": NATIVE_BUILD_TASK_ID,
        "manifest_artifact": NATIVE_BUILD_MANIFEST_ARTIFACT,
        "pip_freeze_artifact": NATIVE_BUILD_PIP_FREEZE_ARTIFACT,
        "build_manifest_sha256": BUILD_MANIFEST_SHA256,
        "build_manifest_bytes": NATIVE_BUILD_MANIFEST_BYTES,
        "native_bundle_bytes": NATIVE_BUNDLE_BYTES,
        "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
        "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        "gpu_devices": [
            {
                "index": index,
                "name": "NVIDIA A100-PCIE-40GB",
                "capability": [8, 0],
            }
            for index in range(4)
        ],
        "python": "3.12.12",
        "torch": "2.10.0+cu128",
        "torch_cuda": "12.8",
        "torch_arch_list": list(TORCH_ARCH_LIST),
        "torch_cuda_arch_list": "7.0;8.0;12.0",
        "mmcv_wheel": {
            "name": MMCV_WHEEL_NAME,
            "bytes": MMCV_WHEEL_BYTES,
            "sha256": MMCV_WHEEL_SHA256,
        },
        "pip_freeze_required_lines": sorted(PIP_FREEZE_REQUIRED_LINES),
    }
    contract["seal_sha256"] = _seal(contract)
    return contract


def _task_record(subject: str, seed_index: int, seed: int) -> dict[str, object]:
    train_key = f"train-r01-s{seed_index:02d}-{subject}"
    return {
        "training": {
            "task_key": train_key,
            "kind": "training",
            "subject": subject,
            "seed_index": seed_index,
            "training_seed": seed,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "parent_role": "p1_a100_controller",
            "predecessor_task_id": SOURCE_D_TASK_ID,
            "queue_name": QUEUE_NAME,
            "queue_id": QUEUE_ID,
            "config_sha256": SUBJECT_CONFIG_SHA256[subject],
            "gpus": 4,
            "global_batch_size": 8,
            "max_epochs": 50,
            "precision": "FP32",
            "checkpoint_policy": "epoch_50_final",
        },
        "evaluation": {
            "task_key": f"eval-r01-s{seed_index:02d}-{subject}",
            "kind": "evaluation",
            "subject": subject,
            "seed_index": seed_index,
            "training_seed": seed,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "parent_role": "p1_a100_controller",
            "predecessor_task_key": train_key,
            "queue_name": QUEUE_NAME,
            "queue_id": QUEUE_ID,
            "protocol_id": PROTOCOL_ID,
            "condition_ids": list(CONDITION_IDS),
            "run_count": len(CONDITION_IDS),
        },
    }


def build_plan() -> dict[str, object]:
    """Return the one allowed P1 A100 seed1/2/3 plan."""

    pairs = [
        _task_record(subject, seed_index, seed)
        for seed_index, seed in zip(TRAINING_SEED_INDICES, TRAINING_SEEDS, strict=True)
        for subject in SUBJECTS
    ]
    plan: dict[str, object] = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_a100_multiseed_plan",
        "default_mode": "dry_run",
        "remote_mutation_authorization": {
            "execute_flag_required": True,
            "exact_token_required": True,
            "token_sha256": hashlib.sha256(EXECUTION_TOKEN.encode("utf-8")).hexdigest(),
        },
        "scope": {
            "subjects": list(SUBJECTS),
            "training_seeds": list(TRAINING_SEEDS),
            "historical_anchor_training_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "training_task_count": 6,
            "evaluation_task_count": 6,
            "evaluation_condition_count": 72,
            "max_parallel_tasks": 1,
            "retrain_seed1": True,
            "reuse_historical_seed1_checkpoint_allowed": False,
            "legacy_5090_executor_allowed": False,
            "old_full_matrix_executor_allowed": False,
        },
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "condition_ids": list(CONDITION_IDS),
        },
        "provenance": {
            "controller_parent_task_id": SELECTOR_TASK_ID,
            "source_d_task_id": SOURCE_D_TASK_ID,
            "source_d_script_sha256": SOURCE_D_SCRIPT_SHA256,
            "source_d_equivalence_sha256": SOURCE_D_EQUIVALENCE_SHA256,
            "teacher_quality_gate_task_id": TEACHER_GATE_TASK_ID,
            "teacher_task_id": TEACHER_TASK_ID,
            "teacher_model_id": TEACHER_MODEL_ID,
            "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        },
        "runtime": {
            "project_id": PROJECT_ID,
            "queue_name": QUEUE_NAME,
            "queue_id": QUEUE_ID,
            "accelerator": "A100",
            "allowed_worker_ids": sorted(ALLOWED_WORKER_IDS),
            "a100_runtime_contract": _a100_runtime_contract(),
            "native_build_runtime_contract": _native_build_runtime_contract(),
            "files_server_uri": FILES_SERVER_URI,
            "container_image": CONTAINER_IMAGE,
            "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "training_dataset_id": TRAINING_DATASET_ID,
            "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
            "build_manifest_sha256": BUILD_MANIFEST_SHA256,
        },
        "legacy_retirement_gate": _legacy_retirement_contract(),
        "execution_protocol": {
            "stable_execution_key": True,
            "server_side_exact_name_deduplication": True,
            "local_nonblocking_lock": True,
            "recoverable_journal": JOURNAL_ARTIFACT,
            "timeout_leaves_active_child_unchanged": True,
            "created_orphan_policy": "mark_failed_then_archive",
        },
        "anchors": ANCHORS,
        "pairs": pairs,
    }
    plan["seal_sha256"] = _seal(plan)
    validate_plan(plan)
    return plan


def validate_plan(value: Mapping[str, object]) -> None:
    """Fail closed if the serialized plan expands beyond 6 train + 6 eval."""

    plan = _mapping(value, "P1 plan")
    _exact_keys(
        plan,
        {
            "anchors",
            "default_mode",
            "document_type",
            "execution_protocol",
            "pairs",
            "protocol",
            "provenance",
            "remote_mutation_authorization",
            "runtime",
            "legacy_retirement_gate",
            "schema_version",
            "scope",
            "seal_sha256",
        },
        "P1 plan",
    )
    if plan.get("seal_sha256") != _seal(plan):
        raise P1ExecutorError("P1 plan seal mismatch")
    _require_equal(plan.get("schema_version"), 1, "P1 plan schema")
    _require_equal(
        plan.get("document_type"),
        "resilient_v2x_p1_a100_multiseed_plan",
        "P1 plan document type",
    )
    scope = _mapping(plan.get("scope"), "P1 plan scope")
    expected_scope = {
        "subjects": list(SUBJECTS),
        "training_seeds": list(TRAINING_SEEDS),
        "historical_anchor_training_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "training_task_count": 6,
        "evaluation_task_count": 6,
        "evaluation_condition_count": 72,
        "max_parallel_tasks": 1,
        "retrain_seed1": True,
        "reuse_historical_seed1_checkpoint_allowed": False,
        "legacy_5090_executor_allowed": False,
        "old_full_matrix_executor_allowed": False,
    }
    if scope != expected_scope:
        raise P1ExecutorError("P1 plan scope drifted")
    expected_sections = {
        "default_mode": "dry_run",
        "remote_mutation_authorization": {
            "execute_flag_required": True,
            "exact_token_required": True,
            "token_sha256": hashlib.sha256(EXECUTION_TOKEN.encode("utf-8")).hexdigest(),
        },
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "condition_ids": list(CONDITION_IDS),
        },
        "provenance": {
            "controller_parent_task_id": SELECTOR_TASK_ID,
            "source_d_task_id": SOURCE_D_TASK_ID,
            "source_d_script_sha256": SOURCE_D_SCRIPT_SHA256,
            "source_d_equivalence_sha256": SOURCE_D_EQUIVALENCE_SHA256,
            "teacher_quality_gate_task_id": TEACHER_GATE_TASK_ID,
            "teacher_task_id": TEACHER_TASK_ID,
            "teacher_model_id": TEACHER_MODEL_ID,
            "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        },
        "runtime": {
            "project_id": PROJECT_ID,
            "queue_name": QUEUE_NAME,
            "queue_id": QUEUE_ID,
            "accelerator": "A100",
            "allowed_worker_ids": sorted(ALLOWED_WORKER_IDS),
            "a100_runtime_contract": _a100_runtime_contract(),
            "native_build_runtime_contract": _native_build_runtime_contract(),
            "files_server_uri": FILES_SERVER_URI,
            "container_image": CONTAINER_IMAGE,
            "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "training_dataset_id": TRAINING_DATASET_ID,
            "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
            "build_manifest_sha256": BUILD_MANIFEST_SHA256,
        },
        "legacy_retirement_gate": _legacy_retirement_contract(),
        "execution_protocol": {
            "stable_execution_key": True,
            "server_side_exact_name_deduplication": True,
            "local_nonblocking_lock": True,
            "recoverable_journal": JOURNAL_ARTIFACT,
            "timeout_leaves_active_child_unchanged": True,
            "created_orphan_policy": "mark_failed_then_archive",
        },
        "anchors": ANCHORS,
    }
    for field, expected in expected_sections.items():
        _require_equal(plan.get(field), expected, f"P1 plan {field}")
    pairs = plan.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 6:
        raise P1ExecutorError("P1 plan must contain exactly six train/eval pairs")
    expected_pairs = [
        _task_record(subject, seed_index, seed)
        for seed_index, seed in zip(TRAINING_SEED_INDICES, TRAINING_SEEDS, strict=True)
        for subject in SUBJECTS
    ]
    if pairs != expected_pairs:
        raise P1ExecutorError("P1 train/eval matrix drifted")


def build_pinset() -> dict[str, object]:
    """Return every immutable remote or protocol pin checked by preflight."""

    pinset: dict[str, object] = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_a100_multiseed_pinset",
        "selector": {
            "task_id": SELECTOR_TASK_ID,
            "parent_task_id": SELECTOR_PARENT_TASK_ID,
            "script_sha256": SELECTOR_SCRIPT_SHA256,
            "artifact_hashes": SELECTOR_ARTIFACT_HASHES,
        },
        "source_d": {
            "task_id": SOURCE_D_TASK_ID,
            "parent_task_id": SOURCE_D_PARENT_TASK_ID,
            "source_c_parent_task_id": SOURCE_C_PARENT_TASK_ID,
            "producer_sha256": SOURCE_D_PRODUCER_SHA256,
            "builder_sha256": SOURCE_D_BUILDER_SHA256,
            "source_c_script_sha256": SOURCE_C_SCRIPT_SHA256,
            "source_d_script_sha256": SOURCE_D_SCRIPT_SHA256,
            "equivalence_sha256": SOURCE_D_EQUIVALENCE_SHA256,
            "source_c_snapshot_seal_sha256": SOURCE_C_SNAPSHOT_SEAL_SHA256,
            "source_d_script_seal_sha256": SOURCE_D_SCRIPT_SEAL_SHA256,
            "source_d_receipt_seal_sha256": SOURCE_D_RECEIPT_SEAL_SHA256,
            "transformation_id": SOURCE_D_TRANSFORMATION_ID,
            "artifact_hashes": SOURCE_D_ARTIFACT_HASHES,
            "replacement_names": list(SOURCE_D_REPLACEMENT_NAMES),
            "seed_contract": SOURCE_D_SEED_CONTRACT,
        },
        "teacher": {
            "gate_task_id": TEACHER_GATE_TASK_ID,
            "gate_script_sha256": TEACHER_GATE_SCRIPT_SHA256,
            "gate_artifact_hash": TEACHER_GATE_ARTIFACT_HASH,
            "gate_content_sha256": TEACHER_GATE_CONTENT_SHA256,
            "task_id": TEACHER_TASK_ID,
            "script_sha256": TEACHER_SCRIPT_SHA256,
            "model_id": TEACHER_MODEL_ID,
            "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "artifact_hashes": TEACHER_ARTIFACT_HASHES,
        },
        "anchors": {
            "tasks": ANCHORS,
            "training_script_sha256": ANCHOR_TRAINING_SCRIPT_SHA256,
            "evaluation_script_sha256": ANCHOR_EVALUATION_SCRIPT_SHA256,
            "training_artifact_hashes": ANCHOR_TRAINING_ARTIFACT_HASHES,
            "evaluation_artifact_hashes": ANCHOR_EVALUATION_ARTIFACT_HASHES,
        },
        "runtime": {
            "project_id": PROJECT_ID,
            "queue_id": QUEUE_ID,
            "queue_name": QUEUE_NAME,
            "accelerator": "A100",
            "allowed_worker_ids": sorted(ALLOWED_WORKER_IDS),
            "a100_runtime_contract": _a100_runtime_contract(),
            "native_build_runtime_contract": _native_build_runtime_contract(),
            "files_server_uri": FILES_SERVER_URI,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_archive_name": SOURCE_ARCHIVE_NAME,
            "source_archive_bytes": SOURCE_ARCHIVE_BYTES,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "training_dataset_id": TRAINING_DATASET_ID,
            "native_build_task_id": NATIVE_BUILD_TASK_ID,
            "native_bundle_bytes": NATIVE_BUNDLE_BYTES,
            "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
            "build_manifest_sha256": BUILD_MANIFEST_SHA256,
            "container_image": CONTAINER_IMAGE,
            "container_arguments": CONTAINER_ARGUMENTS,
            "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        },
        "legacy_retirement_gate": _legacy_retirement_contract(),
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "training_seeds": list(TRAINING_SEEDS),
            "historical_anchor_training_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "subjects": list(SUBJECTS),
            "condition_ids": list(CONDITION_IDS),
            "controlled_evaluator_sha256": CONTROLLED_EVALUATOR_SHA256,
            "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
            "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "subject_config_sha256": SUBJECT_CONFIG_SHA256,
        },
    }
    pinset["seal_sha256"] = _seal(pinset)
    return pinset


def _execution_key(plan: Mapping[str, object], pinset: Mapping[str, object]) -> str:
    validate_plan(plan)
    if pinset.get("seal_sha256") != _seal(pinset):
        raise P1ExecutorError("P1 pinset seal mismatch")
    return _content_sha256(
        {
            "plan_seal_sha256": plan["seal_sha256"],
            "pinset_seal_sha256": pinset["seal_sha256"],
        }
    )


def _execution_tag(execution_key: str) -> str:
    return EXECUTION_KEY_TAG_PREFIX + _sha256(execution_key, "execution key")


def _controller_name(execution_key: str) -> str:
    return f"ResilientV2X P1 A100 scoped 6-train 6-eval {execution_key} controller"


def _lease_queue_name(execution_key: str) -> str:
    return f"__p1_a100_resilient_v2x_mutex_{_sha256(execution_key, 'execution key')}"


def _task_status(task: object) -> str:
    value = getattr(task, "status", None)
    if callable(value):
        value = value()
    if value is None:
        getter = getattr(task, "get_status", None)
        if callable(getter):
            value = getter()
    value = getattr(value, "value", value)
    return str(value).rsplit(".", 1)[-1].lower()


def _task_parent(task: object) -> str:
    value = getattr(task, "parent", None)
    if value in (None, ""):
        value = getattr(getattr(task, "data", None), "parent", None)
    return "" if value is None else str(value)


def _task_project(task: object) -> str:
    value = getattr(task, "project", None)
    if value in (None, ""):
        value = getattr(getattr(task, "data", None), "project", None)
    return "" if value is None else str(value)


def _script(task: object) -> dict[str, str]:
    value = getattr(getattr(task, "data", None), "script", None)
    if value is None:
        raise P1ExecutorError("ClearML task has no script")
    return {
        "repository": str(getattr(value, "repository", "") or ""),
        "working_dir": str(getattr(value, "working_dir", "") or ""),
        "entry_point": str(getattr(value, "entry_point", "") or ""),
        "diff": str(getattr(value, "diff", "") or ""),
    }


def _require_script(
    task: object,
    *,
    entry_point: str,
    sha256: str,
    source: str | None = None,
    context: str,
) -> None:
    observed = _script(task)
    if (
        observed["repository"] != ""
        or observed["working_dir"] != "."
        or observed["entry_point"] != entry_point
    ):
        raise P1ExecutorError(f"{context} script location drifted")
    if hashlib.sha256(observed["diff"].encode("utf-8")).hexdigest() != sha256:
        raise P1ExecutorError(f"{context} script SHA-256 drifted")
    if source is not None and observed["diff"] != source:
        raise P1ExecutorError(f"{context} script bytes drifted")


def _artifact_inventory(task: object) -> dict[str, object]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        raise P1ExecutorError("ClearML task has no artifact mapping")
    return dict(artifacts)


def _require_artifact_hashes(
    task: object, expected: Mapping[str, str], context: str
) -> dict[str, object]:
    artifacts = _artifact_inventory(task)
    if set(artifacts) != set(expected):
        raise P1ExecutorError(f"{context} artifact inventory drifted")
    for name, expected_hash in expected.items():
        if str(getattr(artifacts[name], "hash", "") or "") != expected_hash:
            raise P1ExecutorError(f"{context} artifact {name} hash drifted")
    return artifacts


def _read_json_path(path: Path, context: str) -> dict[str, object]:
    path = path.absolute()
    if path.is_symlink():
        raise P1ExecutorError(f"{context} cannot be a symlink")
    path = path.resolve(strict=True)
    flags = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= int(getattr(os, name, 0))
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise P1ExecutorError(f"{context} must be a single-link regular file")
        if before.st_size <= 0 or before.st_size > MAX_JSON_BYTES:
            raise P1ExecutorError(f"{context} size is outside the safe limit")
        payload = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) != before.st_size or any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise P1ExecutorError(f"{context} changed while being read")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P1ExecutorError(f"{context} is invalid JSON: {error}") from None
    return _mapping(value, context)


def _artifact_mapping(artifact: object, context: str) -> dict[str, object]:
    getter = getattr(artifact, "get", None)
    if not callable(getter):
        raise P1ExecutorError(f"{context} cannot be downloaded")
    value = getter()
    if isinstance(value, Mapping):
        return _mapping(value, context)
    if isinstance(value, (str, os.PathLike)):
        return _read_json_path(Path(value), context)
    raise P1ExecutorError(f"{context} is not a JSON object artifact")


def _artifact_local_bytes(
    artifact: object, context: str, *, max_bytes: int = MAX_JSON_BYTES
) -> bytes:
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        raise P1ExecutorError(f"{context} has no local-copy interface")
    value = getter()
    if not isinstance(value, (str, os.PathLike)):
        raise P1ExecutorError(f"{context} local copy is not a path")
    path = Path(value).absolute()
    if path.is_symlink():
        raise P1ExecutorError(f"{context} local copy cannot be a symlink")
    path = path.resolve(strict=True)
    flags = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= int(getattr(os, name, 0))
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise P1ExecutorError(f"{context} local copy is outside the safe limit")
        payload = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) != before.st_size or any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise P1ExecutorError(f"{context} local copy changed while being read")
    return payload


def _validate_selector(task: object) -> None:
    if str(getattr(task, "id", "")) != SELECTOR_TASK_ID:
        raise P1ExecutorError("selector task identity drifted")
    if _task_status(task) != "completed":
        raise P1ExecutorError("selector task is not completed")
    if (
        _task_parent(task) != SELECTOR_PARENT_TASK_ID
        or _task_project(task) != PROJECT_ID
    ):
        raise P1ExecutorError("selector parent or project drifted")
    _require_script(
        task,
        entry_point="clearml_formal_candidate_selector.py",
        sha256=SELECTOR_SCRIPT_SHA256,
        context="selector task",
    )
    _require_artifact_hashes(task, SELECTOR_ARTIFACT_HASHES, "selector task")


def _validate_seed_contract(value: object) -> None:
    if _mapping(value, "Source-D seed contract") != SOURCE_D_SEED_CONTRACT:
        raise P1ExecutorError("Source-D seed contract drifted")


def _validate_source_d(task: object) -> str:
    if str(getattr(task, "id", "")) != SOURCE_D_TASK_ID:
        raise P1ExecutorError("Source-D task identity drifted")
    if _task_status(task) != "completed":
        raise P1ExecutorError("Source-D task is not completed")
    if (
        _task_parent(task) != SOURCE_D_PARENT_TASK_ID
        or _task_project(task) != PROJECT_ID
    ):
        raise P1ExecutorError("Source-D parent or project drifted")
    _require_script(
        task,
        entry_point="clearml_formal_source_d_evidence.py",
        sha256=SOURCE_D_PRODUCER_SHA256,
        context="Source-D evidence task",
    )
    artifacts = _require_artifact_hashes(
        task, SOURCE_D_ARTIFACT_HASHES, "Source-D evidence task"
    )
    source_c = _artifact_mapping(
        artifacts["formal_source_c_snapshot"], "Source-C snapshot"
    )
    _exact_keys(
        source_c,
        {
            "artifact_type",
            "complete",
            "schema_version",
            "script",
            "script_diff",
            "seal_sha256",
            "source_c_task_id",
            "source_c_task_status",
        },
        "Source-C snapshot",
    )
    if source_c.get("seal_sha256") != _seal(source_c):
        raise P1ExecutorError("Source-C snapshot seal mismatch")
    expected_source_c = {
        "artifact_type": "resilient_v2x_formal_source_c_snapshot",
        "complete": True,
        "schema_version": 1,
        "source_c_task_id": SOURCE_D_PARENT_TASK_ID,
        "source_c_task_status": "completed",
        "seal_sha256": SOURCE_C_SNAPSHOT_SEAL_SHA256,
    }
    for field, expected in expected_source_c.items():
        _require_equal(source_c.get(field), expected, f"Source-C {field}")
    source_c_text = str(source_c.get("script_diff") or "")
    if (
        hashlib.sha256(source_c_text.encode("utf-8")).hexdigest()
        != SOURCE_C_SCRIPT_SHA256
    ):
        raise P1ExecutorError("Source-C script bytes drifted")
    expected_source_c_metadata = {
        "repository": "",
        "working_dir": ".",
        "entry_point": TASK_ENTRY_POINT,
        "sha256": SOURCE_C_SCRIPT_SHA256,
        "size_bytes": 137_249,
        "line_count": 3_729,
    }
    if source_c.get("script") != expected_source_c_metadata:
        raise P1ExecutorError("Source-C script metadata drifted")

    source_d = _artifact_mapping(artifacts["formal_source_d_script"], "Source-D script")
    _exact_keys(
        source_d,
        {
            "artifact_type",
            "complete",
            "line_count",
            "schema_version",
            "script",
            "seal_sha256",
            "size_bytes",
            "source_c_sha256",
            "source_d_sha256",
            "transformation_id",
        },
        "Source-D script",
    )
    source_d_text = str(source_d.get("script") or "")
    expected_source_d_fields = {
        "artifact_type": "resilient_v2x_formal_source_d_script",
        "complete": True,
        "line_count": 5_167,
        "schema_version": 1,
        "seal_sha256": SOURCE_D_SCRIPT_SEAL_SHA256,
        "size_bytes": 192_238,
        "source_c_sha256": SOURCE_C_SCRIPT_SHA256,
        "source_d_sha256": SOURCE_D_SCRIPT_SHA256,
        "transformation_id": SOURCE_D_TRANSFORMATION_ID,
    }
    for field, expected in expected_source_d_fields.items():
        _require_equal(source_d.get(field), expected, f"Source-D {field}")
    if source_d.get("seal_sha256") != _seal(source_d):
        raise P1ExecutorError("Source-D script seal mismatch")
    if (
        hashlib.sha256(source_d_text.encode("utf-8")).hexdigest()
        != SOURCE_D_SCRIPT_SHA256
    ):
        raise P1ExecutorError("Source-D script bytes drifted")

    equivalence = _artifact_mapping(
        artifacts["formal_source_d_equivalence"], "Source-D equivalence"
    )
    unhashed = dict(equivalence)
    observed_equivalence_hash = _sha256(
        unhashed.pop("artifact_sha256", None), "Source-D equivalence hash"
    )
    if (
        observed_equivalence_hash != SOURCE_D_EQUIVALENCE_SHA256
        or _content_sha256(unhashed) != SOURCE_D_EQUIVALENCE_SHA256
    ):
        raise P1ExecutorError("Source-D equivalence hash mismatch")
    _require_equal(
        equivalence.get("transformation_id"),
        SOURCE_D_TRANSFORMATION_ID,
        "Source-D transformation",
    )
    _validate_seed_contract(equivalence.get("seed_contract"))
    diff = equivalence.get("diff")
    if not isinstance(diff, list) or len(diff) != len(SOURCE_D_REPLACEMENT_NAMES):
        raise P1ExecutorError("Source-D replacement inventory drifted")
    for index, (item, name) in enumerate(
        zip(diff, SOURCE_D_REPLACEMENT_NAMES, strict=True), start=1
    ):
        item = _mapping(item, f"Source-D replacement {index}")
        expected = {
            "index": index,
            "name": name,
            "expected_count": 1,
            "observed_count": 1,
        }
        if any(item.get(field) != value for field, value in expected.items()):
            raise P1ExecutorError(f"Source-D replacement {index} drifted")

    receipt = _artifact_mapping(
        artifacts["formal_source_d_evidence_receipt"], "Source-D receipt"
    )
    _exact_keys(
        receipt,
        {
            "artifact_hashes",
            "artifact_type",
            "complete",
            "provenance",
            "publication_order",
            "schema_version",
            "seal_sha256",
            "transformation",
        },
        "Source-D receipt",
    )
    if receipt.get("seal_sha256") != _seal(receipt):
        raise P1ExecutorError("Source-D receipt seal mismatch")
    expected_receipt_fields = {
        "artifact_type": "resilient_v2x_formal_source_d_evidence_receipt",
        "complete": True,
        "schema_version": 1,
        "seal_sha256": SOURCE_D_RECEIPT_SEAL_SHA256,
        "publication_order": list(SOURCE_D_ARTIFACT_HASHES),
        "artifact_hashes": {
            "formal_source_c_snapshot": SOURCE_C_SNAPSHOT_SEAL_SHA256,
            "formal_source_d_script": SOURCE_D_SCRIPT_SEAL_SHA256,
            "formal_source_d_equivalence": SOURCE_D_EQUIVALENCE_SHA256,
        },
    }
    for field, expected in expected_receipt_fields.items():
        if receipt.get(field) != expected:
            raise P1ExecutorError(f"Source-D receipt {field} drifted")
    provenance = _mapping(receipt.get("provenance"), "Source-D provenance")
    expected_provenance = {
        "source_c_task_id": SOURCE_D_PARENT_TASK_ID,
        "source_c_task_status": "completed",
        "source_c_task_parent": SOURCE_C_PARENT_TASK_ID,
        "source_c_entry_point": TASK_ENTRY_POINT,
        "source_c_sha256": SOURCE_C_SCRIPT_SHA256,
        "output_task_id": SOURCE_D_TASK_ID,
        "output_parent_task_id": SOURCE_D_PARENT_TASK_ID,
        "builder_source_sha256": SOURCE_D_BUILDER_SHA256,
        "transformation_id": SOURCE_D_TRANSFORMATION_ID,
        "producer_entry_point": "clearml_formal_source_d_evidence.py",
        "producer_script_sha256": SOURCE_D_PRODUCER_SHA256,
    }
    if provenance != expected_provenance:
        raise P1ExecutorError("Source-D provenance drifted")
    transformation = _mapping(receipt.get("transformation"), "Source-D transformation")
    expected_transformation = {
        "source_d_sha256": SOURCE_D_SCRIPT_SHA256,
        "equivalence_artifact_sha256": SOURCE_D_EQUIVALENCE_SHA256,
        "declared_replacement_count": 22,
        "unchanged_segment_count": 23,
        "only_declared_anchor_replacements": True,
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "training_seed_cli": "--training-seed",
        "portable_runner_load_marker": "_validate_rtx5090_runtime_contract_multi_gpu",
        "portable_runner_load_marker_count": 2,
        "legacy_runner_load_target_anchor_count": 0,
    }
    if transformation != expected_transformation:
        raise P1ExecutorError("Source-D receipt transformation drifted")
    return source_d_text


def _validate_teacher_gate(task: object) -> dict[str, object]:
    if str(getattr(task, "id", "")) != TEACHER_GATE_TASK_ID:
        raise P1ExecutorError("teacher gate identity drifted")
    if _task_status(task) != "completed":
        raise P1ExecutorError("teacher gate is not completed")
    if _task_parent(task) != TEACHER_TASK_ID or _task_project(task) != PROJECT_ID:
        raise P1ExecutorError("teacher gate parent or project drifted")
    _require_script(
        task,
        entry_point="clearml_teacher_quality_gate.py",
        sha256=TEACHER_GATE_SCRIPT_SHA256,
        context="teacher gate",
    )
    artifacts = _require_artifact_hashes(
        task,
        {"teacher_quality_gate": TEACHER_GATE_ARTIFACT_HASH},
        "teacher gate",
    )
    gate = _artifact_mapping(artifacts["teacher_quality_gate"], "teacher gate artifact")
    detached = dict(gate)
    observed_hash = _sha256(
        detached.pop("content_sha256", None), "teacher gate content"
    )
    if (
        observed_hash != TEACHER_GATE_CONTENT_SHA256
        or _content_sha256(detached) != TEACHER_GATE_CONTENT_SHA256
    ):
        raise P1ExecutorError("teacher gate content hash mismatch")
    expected = {
        "schema_version": 1,
        "document_type": "resilient_v2x_teacher_quality_gate",
        "passed": True,
        "quality_gate_task_id": TEACHER_GATE_TASK_ID,
        "teacher_task_id": TEACHER_TASK_ID,
        "dataset_id": TRAINING_DATASET_ID,
        "selection_protocol": "DAIR-CLEAN-PAIR1789-v1",
        "selection_metric": "resilient_v2x/car_bev_ap_r40_0.70",
        "training_seed_evidence": "legacy_fixed_by_sealed_source",
    }
    for field, value in expected.items():
        _require_equal(gate.get(field), value, f"teacher gate {field}")
    teacher = _mapping(gate.get("teacher"), "teacher gate teacher")
    expected_teacher = {
        "task_id": TEACHER_TASK_ID,
        "model_id": TEACHER_MODEL_ID,
        "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "selected_epoch": 30,
    }
    for field, value in expected_teacher.items():
        _require_equal(teacher.get(field), value, f"teacher gate teacher {field}")
    source = _mapping(gate.get("source_identity"), "teacher gate source")
    expected_source = {
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_archive_name": SOURCE_ARCHIVE_NAME,
        "source_archive_bytes": SOURCE_ARCHIVE_BYTES,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "training_dataset_id": TRAINING_DATASET_ID,
        "teacher_script_diff_sha256": TEACHER_SCRIPT_SHA256,
    }
    if source != expected_source:
        raise P1ExecutorError("teacher gate source identity drifted")
    return gate


def _container(task: object) -> dict[str, str]:
    value = getattr(getattr(task, "data", None), "container", None)
    if value is None:
        raise P1ExecutorError("ClearML task has no container")
    if isinstance(value, Mapping):
        return {str(key): str(item or "") for key, item in value.items()}
    return {
        "image": str(getattr(value, "image", "") or ""),
        "arguments": str(getattr(value, "arguments", "") or ""),
        "setup_shell_script": str(getattr(value, "setup_shell_script", "") or ""),
    }


def _validate_teacher(task: object) -> None:
    if str(getattr(task, "id", "")) != TEACHER_TASK_ID:
        raise P1ExecutorError("teacher task identity drifted")
    if _task_status(task) != "completed" or _task_parent(task) != "":
        raise P1ExecutorError("teacher status or parent drifted")
    if _task_project(task) != PROJECT_ID:
        raise P1ExecutorError("teacher project drifted")
    _require_script(
        task,
        entry_point=TASK_ENTRY_POINT,
        sha256=TEACHER_SCRIPT_SHA256,
        context="teacher task",
    )
    _require_artifact_hashes(task, TEACHER_ARTIFACT_HASHES, "teacher task")
    expected_container = {
        "image": CONTAINER_IMAGE,
        "arguments": CONTAINER_ARGUMENTS,
        "setup_shell_script": "",
    }
    if _container(task) != expected_container:
        raise P1ExecutorError("teacher container drifted")
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise P1ExecutorError("teacher task cannot expose models")
    models = getter()
    outputs = models.get("output", []) if isinstance(models, Mapping) else []
    if (
        len(
            [
                model
                for model in outputs
                if str(getattr(model, "id", "")) == TEACHER_MODEL_ID
            ]
        )
        != 1
    ):
        raise P1ExecutorError("teacher output model identity drifted")


def _normalize_parameter(observed: object, expected: object) -> bool:
    if type(expected) is bool:
        return observed is expected or observed == str(expected)
    if type(expected) is int:
        return (type(observed) is int and observed == expected) or observed == str(
            expected
        )
    return type(observed) is type(expected) and observed == expected


def _parameters(task: object) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise P1ExecutorError("ClearML task cannot expose parameters")
    try:
        value = getter(backwards_compatibility=False)
    except TypeError:
        value = getter()
    return _mapping(value, "ClearML task parameters")


def _require_parameters(
    task: object, expected: Mapping[str, object], context: str
) -> None:
    observed = _parameters(task)
    if set(observed) != set(expected):
        raise P1ExecutorError(f"{context} parameter keys drifted")
    for key, expected_value in expected.items():
        if not _normalize_parameter(observed.get(key), expected_value):
            raise P1ExecutorError(f"{context} parameter {key} drifted")


def _base_parameters(seed: int) -> dict[str, object]:
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
        "Args/training_seed": seed,
        "Args/amp": False,
    }


def _training_parameters(subject: str, seed: int) -> dict[str, object]:
    result = _base_parameters(seed)
    result.update(
        {
            "Args/stage": "all",
            "Args/experiment_from_task": subject,
            "Args/predecessor_task_id": SOURCE_D_TASK_ID,
            "Args/teacher_task_id": TEACHER_TASK_ID,
            "Args/teacher_model_id": TEACHER_MODEL_ID,
            "Args/teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "Args/allow_failed_teacher_task": False,
        }
    )
    return result


def _evaluation_parameters(
    subject: str,
    seed: int,
    *,
    training_task_id: str,
    model_id: str,
    checkpoint_sha256: str,
) -> dict[str, object]:
    result = _base_parameters(seed)
    result.update(
        {
            "Args/stage": "baseline_validate",
            "Args/predecessor_task_id": training_task_id,
            "Args/controlled_baseline": subject,
            "Args/controlled_baseline_task_id": training_task_id,
            "Args/controlled_baseline_model_id": model_id,
            "Args/controlled_baseline_checkpoint_sha256": checkpoint_sha256,
        }
    )
    return result


def _validate_anchor_tasks(task_class: object) -> None:
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise P1ExecutorError("ClearML cannot load anchor tasks")
    for subject, pins in ANCHORS.items():
        training = getter(task_id=pins["training_task_id"])
        if (
            _task_status(training) != "completed"
            or _task_project(training) != PROJECT_ID
        ):
            raise P1ExecutorError(f"{subject} anchor training task drifted")
        _require_script(
            training,
            entry_point=TASK_ENTRY_POINT,
            sha256=ANCHOR_TRAINING_SCRIPT_SHA256,
            context=f"{subject} anchor training",
        )
        training_artifacts = _require_artifact_hashes(
            training,
            ANCHOR_TRAINING_ARTIFACT_HASHES[subject],
            f"{subject} anchor training",
        )
        contract = _artifact_mapping(
            training_artifacts["final_checkpoint_contract"],
            f"{subject} anchor checkpoint contract",
        )
        for field, expected in {
            "model_id": pins["model_id"],
            "sha256": pins["checkpoint_sha256"],
            "filename": "epoch_50.pth",
        }.items():
            _require_equal(contract.get(field), expected, f"{subject} anchor {field}")
        run_contract = _artifact_mapping(
            training_artifacts["run_contract"],
            f"{subject} anchor run contract",
        )
        config = _mapping(run_contract.get("config"), f"{subject} anchor config")
        _require_equal(
            config.get("config_sha256"),
            SUBJECT_CONFIG_SHA256[subject],
            f"{subject} anchor config SHA",
        )
        _require_equal(
            run_contract.get("seed"),
            TRAINING_OVERLAY_PROTOCOL_SEED,
            f"{subject} anchor seed",
        )

        evaluation = getter(task_id=pins["evaluation_task_id"])
        if (
            _task_status(evaluation) != "completed"
            or _task_project(evaluation) != PROJECT_ID
        ):
            raise P1ExecutorError(f"{subject} anchor evaluation task drifted")
        _require_script(
            evaluation,
            entry_point=TASK_ENTRY_POINT,
            sha256=ANCHOR_EVALUATION_SCRIPT_SHA256,
            context=f"{subject} anchor evaluation",
        )
        evaluation_artifacts = _require_artifact_hashes(
            evaluation,
            ANCHOR_EVALUATION_ARTIFACT_HASHES[subject],
            f"{subject} anchor evaluation",
        )
        metrics = _artifact_mapping(
            evaluation_artifacts["controlled_baseline_metrics"],
            f"{subject} anchor metrics",
        )
        _validate_metrics(
            metrics,
            subject=subject,
            checkpoint_sha256=str(pins["checkpoint_sha256"]),
        )


def _validate_native_build_task(task: object) -> dict[str, object]:
    context = "native A100 build"
    if (
        str(getattr(task, "id", "") or "") != NATIVE_BUILD_TASK_ID
        or _task_status(task) != "completed"
        or _task_project(task) != PROJECT_ID
    ):
        raise P1ExecutorError("native A100 build identity or status drifted")
    runtime_receipt = _validate_task_runtime(task, context)
    artifacts = _artifact_inventory(task)
    if not {
        NATIVE_BUILD_MANIFEST_ARTIFACT,
        NATIVE_BUILD_PIP_FREEZE_ARTIFACT,
    }.issubset(artifacts):
        raise P1ExecutorError("native A100 build runtime artifacts are missing")
    manifest_bytes = _artifact_local_bytes(
        artifacts[NATIVE_BUILD_MANIFEST_ARTIFACT],
        "native A100 build manifest",
    )
    manifest_artifact = artifacts[NATIVE_BUILD_MANIFEST_ARTIFACT]
    if (
        len(manifest_bytes) != NATIVE_BUILD_MANIFEST_BYTES
        or str(getattr(manifest_artifact, "hash", "") or "") != BUILD_MANIFEST_SHA256
        or hashlib.sha256(manifest_bytes).hexdigest() != BUILD_MANIFEST_SHA256
    ):
        raise P1ExecutorError("native A100 build manifest SHA-256 drifted")
    try:
        manifest_value = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P1ExecutorError(
            f"native A100 build manifest is invalid JSON: {error}"
        ) from None
    manifest = _mapping(manifest_value, "native A100 build manifest")
    _require_equal(manifest.get("amp"), False, "native A100 build amp")
    base_image = _mapping(manifest.get("base_image"), "native A100 build base image")
    _require_equal(
        base_image.get("manifest_digest"),
        BASE_IMAGE_MANIFEST_DIGEST,
        "native A100 build base image manifest",
    )
    _require_equal(
        base_image.get("platform"), "linux/amd64", "native A100 build platform"
    )
    mmcv = _mapping(manifest.get("mmcv_wheel"), "native A100 build MMCV wheel")
    _require_equal(
        {field: mmcv.get(field) for field in ("bytes", "name", "sha256")},
        {
            "bytes": MMCV_WHEEL_BYTES,
            "name": MMCV_WHEEL_NAME,
            "sha256": MMCV_WHEEL_SHA256,
        },
        "native A100 build MMCV wheel",
    )
    bundle = _mapping(manifest.get("native_bundle"), "native A100 build native bundle")
    _require_equal(
        {field: bundle.get(field) for field in ("bytes", "sha256")},
        {"bytes": NATIVE_BUNDLE_BYTES, "sha256": NATIVE_BUNDLE_SHA256},
        "native A100 build native bundle",
    )
    build_runtime = _mapping(manifest.get("runtime"), "native A100 build runtime")
    expected_build_runtime = _native_build_runtime_contract()
    for field, expected in {
        "devices": expected_build_runtime["gpu_devices"],
        "python": expected_build_runtime["python"],
        "torch": expected_build_runtime["torch"],
        "torch_cuda": expected_build_runtime["torch_cuda"],
        "torch_arch_list": expected_build_runtime["torch_arch_list"],
    }.items():
        _require_equal(
            build_runtime.get(field), expected, f"native A100 build runtime {field}"
        )
    _require_equal(
        manifest.get("torch_cuda_arch_list"),
        expected_build_runtime["torch_cuda_arch_list"],
        "native A100 build torch CUDA arch list",
    )
    freeze_bytes = _artifact_local_bytes(
        artifacts[NATIVE_BUILD_PIP_FREEZE_ARTIFACT],
        "native A100 pip freeze",
        max_bytes=8 * 1024 * 1024,
    )
    try:
        freeze_lines = freeze_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise P1ExecutorError(f"native A100 pip freeze is not UTF-8: {error}") from None
    if len(freeze_lines) != len(set(freeze_lines)):
        raise P1ExecutorError("native A100 pip freeze contains duplicate lines")
    if not PIP_FREEZE_REQUIRED_LINES.issubset(freeze_lines):
        raise P1ExecutorError("native A100 pip freeze required pins drifted")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_a100_native_build_receipt",
        "task_id": NATIVE_BUILD_TASK_ID,
        "contract": expected_build_runtime,
        "task_runtime": runtime_receipt,
        "manifest_sha256": BUILD_MANIFEST_SHA256,
        "pip_freeze_required_line_count": len(PIP_FREEZE_REQUIRED_LINES),
    }
    receipt["seal_sha256"] = _seal(receipt)
    return receipt


def _task_name(execution_key: str, task_key: str) -> str:
    execution_key = _sha256(execution_key, "execution key")
    if (
        re.fullmatch(
            r"(?:train|eval)-r01-s0[123]-(?:resilient_v2x|bevfusion)", task_key
        )
        is None
    ):
        raise P1ExecutorError("P1 task key is outside the fixed scope")
    return f"ResilientV2X P1 A100 multiseed {execution_key} {task_key}"


def _set_script(task: object, source: str) -> None:
    setter = getattr(task, "set_script", None)
    if not callable(setter):
        raise P1ExecutorError("created task cannot set Source-D script")
    result = setter(
        repository="",
        working_dir=".",
        entry_point=TASK_ENTRY_POINT,
        diff=source,
    )
    if result is False:
        raise P1ExecutorError("created task rejected Source-D script")


def _set_parameters(task: object, parameters: Mapping[str, object]) -> None:
    setter = getattr(task, "set_parameters", None)
    if not callable(setter):
        raise P1ExecutorError("created task cannot set parameters")
    try:
        result = setter(dict(parameters), __update=False)
    except TypeError:
        result = setter(dict(parameters))
    if result is False:
        raise P1ExecutorError("created task rejected exact parameters")


def _fresh(task_class: object, task_id: str) -> object:
    return task_class.get_task(task_id=_clearml_id(task_id, "task"))


def _task_queue(task: object) -> str:
    return str(
        getattr(getattr(getattr(task, "data", None), "execution", None), "queue", "")
        or ""
    )


def _add_tags(task: object, tags: Sequence[str]) -> None:
    adder = getattr(task, "add_tags", None)
    if not callable(adder):
        raise P1ExecutorError("ClearML task cannot accept idempotency tags")
    result = adder(list(tags))
    if result is False:
        raise P1ExecutorError("ClearML task rejected idempotency tags")


def _quarantine_created(task_class: object, task: object, *, reason: str) -> None:
    task_id = _clearml_id(getattr(task, "id", None), "orphan task")
    task = _fresh(task_class, task_id)
    tags = _task_tags(task)
    status = _task_status(task)
    if ORPHAN_TAG not in tags:
        if status != "created" or _task_queue(task):
            raise P1ExecutorError(
                "only an unqueued created orphan may enter quarantine"
            )
        _add_tags(task, [ORPHAN_TAG])
        task = _fresh(task_class, task_id)
        tags = _task_tags(task)
        if ORPHAN_TAG not in tags:
            raise P1ExecutorError("orphan quarantine tag did not round-trip")
        status = _task_status(task)
    if _task_queue(task):
        raise P1ExecutorError("a quarantined orphan unexpectedly entered a queue")
    if _task_archived(task):
        return
    if status == "created":
        marker = getattr(task, "mark_failed", None)
        if not callable(marker):
            raise P1ExecutorError("created orphan cannot be marked failed")
        marker(
            force=True,
            status_reason="P1 scoped executor quarantined a pre-enqueue orphan",
            status_message=reason[:512],
        )
        task = _fresh(task_class, task_id)
        status = _task_status(task)
    if status != "failed":
        raise P1ExecutorError(
            "tagged orphan is outside the recoverable quarantine states"
        )
    archiver = getattr(task, "set_archived", None)
    if not callable(archiver):
        raise P1ExecutorError("failed orphan cannot be archived")
    archiver(True)
    task = _fresh(task_class, task_id)
    if (
        _task_status(task) != "failed"
        or not _task_archived(task)
        or ORPHAN_TAG not in _task_tags(task)
    ):
        raise P1ExecutorError("orphan quarantine did not round-trip")


def _select_unique_created_candidate(
    task_class: object, *, name: str, context: str
) -> object | None:
    queried = _query_named_tasks(task_class, name=name)
    for task in queried:
        if _is_quarantined(task):
            _quarantine_created(
                task_class,
                task,
                reason="resuming an interrupted orphan quarantine",
            )
    matches = [task for task in queried if not _is_quarantined(task)]
    if not matches:
        return None
    if len(matches) != 1:
        raise P1ExecutorError(f"{context} has duplicate active tasks")
    return matches[0]


def _create_task(
    task_class: object,
    *,
    teacher_task: object,
    controller_id: str,
    task_key: str,
    source_d: str,
    parameters: Mapping[str, object],
    execution_key: str,
    on_discovered: Callable[[str, str], None] | None = None,
) -> object:
    clone = getattr(task_class, "clone", None)
    if not callable(clone):
        raise P1ExecutorError("ClearML cannot clone scoped tasks")
    name = _task_name(execution_key, task_key)
    task = _select_unique_created_candidate(
        task_class, name=name, context=f"scoped task {task_key}"
    )
    origin = "resumed"
    if task is None:
        task = clone(
            source_task=teacher_task,
            name=name,
            comment="Sealed P1 A100 two-method/three-seed recoverable task.",
            parent=controller_id,
            project=PROJECT_ID,
        )
        if task is None:
            raise P1ExecutorError("ClearML clone returned no task")
        origin = "cloned"
    task_id = _clearml_id(getattr(task, "id", ""), "created task")
    if on_discovered is not None:
        try:
            on_discovered(task_id, origin)
        except Exception as error:
            latest = _fresh(task_class, task_id)
            if (
                origin == "cloned"
                and _task_status(latest) == "created"
                and not _task_queue(latest)
            ):
                _quarantine_created(
                    task_class,
                    latest,
                    reason="journal persistence failed immediately after clone",
                )
                try:
                    on_discovered(task_id, "quarantined")
                except Exception as recovery_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "best-effort quarantined journal persistence also failed: "
                            f"{recovery_error}"
                        )
            raise
    if origin == "cloned":
        try:
            _add_tags(task, [_execution_tag(execution_key), f"p1-task-key:{task_key}"])
            task = _select_unique_created_candidate(
                task_class, name=name, context=f"scoped task {task_key}"
            )
        except Exception:
            latest = _fresh(task_class, task_id)
            if _task_status(latest) == "created" and not _task_queue(latest):
                _quarantine_created(
                    task_class,
                    latest,
                    reason="idempotency tag or server-side deduplication failed",
                )
                if on_discovered is not None:
                    on_discovered(task_id, "quarantined")
            raise
        if task is None:
            raise P1ExecutorError("new scoped task disappeared during deduplication")
        task_id = _clearml_id(getattr(task, "id", ""), "deduplicated task")
        if on_discovered is not None:
            on_discovered(task_id, "deduplicated")
    task = _fresh(task_class, task_id)
    if _task_parent(task) != controller_id or _task_project(task) != PROJECT_ID:
        raise P1ExecutorError("new scoped task parent or project drifted")
    if str(getattr(task, "name", "") or "") != name:
        raise P1ExecutorError("new scoped task name drifted")
    status = _task_status(task)
    required_tags = {_execution_tag(execution_key), f"p1-task-key:{task_key}"}
    observed_tags = _task_tags(task)
    conflicting_execution_tags = {
        item
        for item in observed_tags
        if item.startswith(EXECUTION_KEY_TAG_PREFIX) and item not in required_tags
    }
    conflicting_task_key_tags = {
        item
        for item in observed_tags
        if item.startswith("p1-task-key:") and item not in required_tags
    }
    if conflicting_execution_tags or conflicting_task_key_tags:
        raise P1ExecutorError("scoped task has conflicting idempotency tags")
    if not required_tags.issubset(observed_tags):
        if status != "created" or _task_queue(task):
            raise P1ExecutorError("active scoped task is missing idempotency tags")
        _add_tags(task, sorted(required_tags - observed_tags))
        task = _fresh(task_class, task_id)
        if not required_tags.issubset(_task_tags(task)):
            raise P1ExecutorError("scoped task idempotency tags did not round-trip")
    if status == "created":
        try:
            if _artifact_inventory(task):
                raise P1ExecutorError("created scoped task inherited stale artifacts")
            models = task.get_models()
            if isinstance(models, Mapping) and models.get("output"):
                raise P1ExecutorError("created scoped task inherited output models")
            teacher_parameters = _parameters(teacher_task)
            observed_script = _script(task)
            teacher_script = _script(teacher_task)
            source_script = {
                "repository": "",
                "working_dir": ".",
                "entry_point": TASK_ENTRY_POINT,
                "diff": source_d,
            }
            observed_parameters = _parameters(task)
            if (
                observed_script == teacher_script
                and all(
                    _normalize_parameter(observed_parameters.get(key), item)
                    for key, item in teacher_parameters.items()
                )
                and set(observed_parameters) == set(teacher_parameters)
            ):
                _set_script(task, source_d)
                task = _fresh(task_class, task_id)
                observed_script = _script(task)
                observed_parameters = _parameters(task)
            if (
                observed_script == source_script
                and all(
                    _normalize_parameter(observed_parameters.get(key), item)
                    for key, item in teacher_parameters.items()
                )
                and set(observed_parameters) == set(teacher_parameters)
            ):
                _set_parameters(task, parameters)
                task = _fresh(task_class, task_id)
            task.output_uri = FILES_SERVER_URI
            task = _fresh(task_class, task_id)
        except Exception as error:
            latest = _fresh(task_class, task_id)
            if (
                origin == "cloned"
                and _task_status(latest) == "created"
                and not _task_queue(latest)
            ):
                _quarantine_created(task_class, latest, reason=str(error))
                if on_discovered is not None:
                    on_discovered(task_id, "quarantined")
            raise
    elif status not in ACTIVE_STATUSES | {"completed"}:
        raise P1ExecutorError(f"scoped task has unrecoverable status {status!r}")
    _require_script(
        task,
        entry_point=TASK_ENTRY_POINT,
        sha256=SOURCE_D_SCRIPT_SHA256,
        source=source_d,
        context="created scoped task",
    )
    _require_parameters(task, parameters, "created scoped task")
    if _container(task) != _container(teacher_task):
        raise P1ExecutorError("created scoped task container drifted")
    expected_queue = "" if status == "created" else QUEUE_ID
    if _task_queue(task) != expected_queue:
        raise P1ExecutorError("scoped task queue binding drifted during recovery")
    return task


def _enqueue_once(
    task_class: object,
    task: object,
    *,
    authorization_token: str,
) -> None:
    if authorization_token != EXECUTION_TOKEN:
        raise P1ExecutorError("remote enqueue token mismatch")
    status = _task_status(task)
    if status in ACTIVE_STATUSES - {"created"} | {"completed"}:
        if _task_queue(task) != QUEUE_ID:
            raise P1ExecutorError("resumed task queue binding drifted")
        return
    if status != "created":
        raise P1ExecutorError("only a verified or resumed task may be enqueued")
    enqueue = getattr(task_class, "enqueue", None)
    if not callable(enqueue):
        raise P1ExecutorError("ClearML cannot enqueue scoped tasks")
    caught: Exception | None = None
    try:
        response = enqueue(task=task, queue_id=QUEUE_ID, force=False)
    except Exception as error:
        caught = error
        response = None
    task = _fresh(task_class, str(getattr(task, "id", "")))
    committed = (
        _task_status(task) in ACTIVE_STATUSES | {"completed"}
        and _task_queue(task) == QUEUE_ID
    )
    if caught is not None and not committed:
        raise P1ExecutorError(
            "ClearML enqueue failed before an auditable server commit"
        ) from caught
    observed_queue = str(
        getattr(getattr(getattr(task, "data", None), "execution", None), "queue", "")
        or ""
    )
    if observed_queue != QUEUE_ID or not committed:
        raise P1ExecutorError("scoped task queue binding drifted")
    acknowledged = (
        response.get("queued") == 1 and response.get("updated") == 1
        if isinstance(response, Mapping)
        else getattr(response, "queued", None) == 1
        and getattr(response, "updated", None) == 1
    )
    if caught is None and not acknowledged and not committed:
        raise P1ExecutorError("ClearML did not acknowledge the scoped enqueue")


def _wait_for_task(
    task_class: object,
    task_id: str,
    *,
    timeout_hours: float,
    poll_seconds: float,
    on_poll: Callable[[object], None] | None = None,
) -> object:
    deadline = time.monotonic() + timeout_hours * 3600.0
    while True:
        task = _fresh(task_class, task_id)
        status = _task_status(task)
        if status == "completed":
            return task
        if status in FAILURE_STATUSES or status not in ACTIVE_STATUSES:
            raise P1ExecutorError(f"scoped task {task_id} ended with status {status!r}")
        if on_poll is not None:
            on_poll(task)
        if time.monotonic() >= deadline:
            raise P1RecoverableTimeout(
                f"scoped task {task_id} exceeded the timeout and remains {status!r}"
            )
        time.sleep(poll_seconds)


def _worker_id(task: object) -> str:
    value = getattr(getattr(task, "data", None), "last_worker", None)
    return str(value or "")


def _validate_worker_and_queue(task: object, context: str) -> None:
    worker = _worker_id(task)
    if worker not in ALLOWED_WORKER_IDS:
        raise P1ExecutorError(f"{context} did not run on an allowed 4xA100 worker")
    queue = str(
        getattr(getattr(getattr(task, "data", None), "execution", None), "queue", "")
        or ""
    )
    if queue != QUEUE_ID:
        raise P1ExecutorError(f"{context} queue drifted")


def _runtime_mapping(task: object, context: str) -> dict[str, object]:
    runtime = getattr(getattr(task, "data", None), "runtime", None)
    if isinstance(runtime, Mapping):
        return dict(runtime)
    if runtime is not None:
        try:
            value = vars(runtime)
        except TypeError:
            value = None
        if isinstance(value, Mapping):
            return dict(value)
    raise P1ExecutorError(f"{context} has no readable ClearML runtime receipt")


def _runtime_device_values(value: object, context: str) -> list[str]:
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [str(item).strip() for item in value]
    else:
        raise P1ExecutorError(f"{context} is not a device vector")
    if any(not item for item in values):
        raise P1ExecutorError(f"{context} contains an empty device value")
    return values


def _validate_task_runtime(task: object, context: str) -> dict[str, object]:
    """Bind a completed child to the observed A100 runtime plus build pins."""

    _validate_worker_and_queue(task, context)
    runtime = _runtime_mapping(task, context)
    gpu_types = _runtime_device_values(runtime.get("gpu_type"), f"{context} gpu_type")
    gpu_memory = _runtime_device_values(
        runtime.get("gpu_memory"), f"{context} gpu_memory"
    )
    if type(runtime.get("gpu_type")) is not str:
        raise P1ExecutorError(f"{context} runtime gpu_type representation drifted")
    if type(runtime.get("gpu_memory")) is not str:
        raise P1ExecutorError(f"{context} runtime gpu_memory representation drifted")
    expected_gpu_types = list(A100_RUNTIME_CONTRACT["gpu_type_normalized"])
    expected_gpu_memory = list(A100_RUNTIME_CONTRACT["gpu_memory_normalized"])
    actual = {
        "hostname": runtime.get("_exec_agent_hostname"),
        "gpu_count": runtime.get("gpu_count"),
        "gpu_type": gpu_types,
        "gpu_memory": gpu_memory,
        "gpu_driver_version": runtime.get("gpu_driver_version"),
        "gpu_driver_cuda_version": runtime.get("gpu_driver_cuda_version"),
        "python_version": runtime.get("python_version"),
        "python_exec": runtime.get("python_exec"),
        "os": runtime.get("OS"),
    }
    required_actual = {
        "hostname": A100_RUNTIME_CONTRACT["hostname"],
        "gpu_count": A100_RUNTIME_CONTRACT["gpu_count"],
        "gpu_type": expected_gpu_types,
        "gpu_memory": expected_gpu_memory,
        "gpu_driver_version": A100_RUNTIME_CONTRACT["gpu_driver_version"],
        "gpu_driver_cuda_version": A100_RUNTIME_CONTRACT["gpu_driver_cuda_version"],
        "python_version": A100_RUNTIME_CONTRACT["python_version"],
        "os": A100_RUNTIME_CONTRACT["os"],
    }
    for field, expected in required_actual.items():
        _require_equal(actual[field], expected, f"{context} runtime {field}")
    python_exec = actual["python_exec"]
    if (
        type(python_exec) is not str
        or python_exec not in A100_RUNTIME_CONTRACT["python_exec_allowlist"]
    ):
        raise P1ExecutorError(f"{context} runtime python_exec drifted")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_a100_child_runtime",
        "task_id": _clearml_id(getattr(task, "id", None), f"{context} task"),
        "worker_id": _worker_id(task),
        "queue_id": QUEUE_ID,
        "actual": actual,
        "indirect_build_contract": {
            "native_build_task_id": NATIVE_BUILD_TASK_ID,
            "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
            "build_manifest_sha256": BUILD_MANIFEST_SHA256,
            "gpu_compute_capability": A100_RUNTIME_CONTRACT["gpu_compute_capability"],
            "torch": A100_RUNTIME_CONTRACT["torch"],
            "torch_cuda": A100_RUNTIME_CONTRACT["torch_cuda"],
            "mmcv": A100_RUNTIME_CONTRACT["mmcv"],
            "mmengine": A100_RUNTIME_CONTRACT["mmengine"],
            "mmdet": A100_RUNTIME_CONTRACT["mmdet"],
            "mmdet3d": A100_RUNTIME_CONTRACT["mmdet3d"],
        },
    }
    receipt["seal_sha256"] = _seal(receipt)
    return receipt


def _validate_recorded_runtime_receipt(
    value: Mapping[str, object], *, task_id: str
) -> dict[str, object]:
    receipt = _mapping(value, "recorded A100 child runtime")
    if receipt.get("seal_sha256") != _seal(receipt):
        raise P1ExecutorError("recorded A100 child runtime seal mismatch")
    expected = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_a100_child_runtime",
        "task_id": task_id,
        "queue_id": QUEUE_ID,
        "indirect_build_contract": {
            "native_build_task_id": NATIVE_BUILD_TASK_ID,
            "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
            "build_manifest_sha256": BUILD_MANIFEST_SHA256,
            "gpu_compute_capability": A100_RUNTIME_CONTRACT["gpu_compute_capability"],
            "torch": A100_RUNTIME_CONTRACT["torch"],
            "torch_cuda": A100_RUNTIME_CONTRACT["torch_cuda"],
            "mmcv": A100_RUNTIME_CONTRACT["mmcv"],
            "mmengine": A100_RUNTIME_CONTRACT["mmengine"],
            "mmdet": A100_RUNTIME_CONTRACT["mmdet"],
            "mmdet3d": A100_RUNTIME_CONTRACT["mmdet3d"],
        },
    }
    for field, expected_value in expected.items():
        _require_equal(
            receipt.get(field), expected_value, f"recorded A100 runtime {field}"
        )
    worker = str(receipt.get("worker_id") or "")
    if worker not in ALLOWED_WORKER_IDS:
        raise P1ExecutorError("recorded A100 runtime worker drifted")
    actual = _mapping(receipt.get("actual"), "recorded A100 runtime actual")
    required_actual = {
        "hostname": A100_RUNTIME_CONTRACT["hostname"],
        "gpu_count": A100_RUNTIME_CONTRACT["gpu_count"],
        "gpu_type": A100_RUNTIME_CONTRACT["gpu_type_normalized"],
        "gpu_memory": A100_RUNTIME_CONTRACT["gpu_memory_normalized"],
        "gpu_driver_version": A100_RUNTIME_CONTRACT["gpu_driver_version"],
        "gpu_driver_cuda_version": A100_RUNTIME_CONTRACT["gpu_driver_cuda_version"],
        "python_version": A100_RUNTIME_CONTRACT["python_version"],
        "os": A100_RUNTIME_CONTRACT["os"],
    }
    for field, expected_value in required_actual.items():
        _require_equal(
            actual.get(field), expected_value, f"recorded A100 runtime {field}"
        )
    if actual.get("python_exec") not in A100_RUNTIME_CONTRACT["python_exec_allowlist"]:
        raise P1ExecutorError("recorded A100 runtime python_exec drifted")
    return receipt


def _validate_training_result(
    task: object,
    *,
    subject: str,
    seed: int,
    controller_id: str,
    source_d: str,
) -> dict[str, object]:
    context = f"{subject} seed {seed} training"
    if _task_status(task) != "completed" or _task_parent(task) != controller_id:
        raise P1ExecutorError(f"{context} status or parent drifted")
    runtime_receipt = _validate_task_runtime(task, context)
    _require_script(
        task,
        entry_point=TASK_ENTRY_POINT,
        sha256=SOURCE_D_SCRIPT_SHA256,
        source=source_d,
        context=context,
    )
    _require_parameters(task, _training_parameters(subject, seed), context)
    artifacts = _artifact_inventory(task)
    for name in (
        "run_contract",
        "final_checkpoint_contract",
        "common_teacher_initialization_audit",
    ):
        if name not in artifacts:
            raise P1ExecutorError(f"{context} is missing {name}")
    run_contract = _artifact_mapping(
        artifacts["run_contract"], f"{context} run contract"
    )
    expected_run = {
        "task_id": str(getattr(task, "id", "")),
        "experiment": subject,
        "predecessor_task_id": SOURCE_D_TASK_ID,
        "training_dataset_id": TRAINING_DATASET_ID,
        "native_build_task_id": NATIVE_BUILD_TASK_ID,
        "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
        "build_manifest_sha256": BUILD_MANIFEST_SHA256,
        "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        "gpus": 4,
        "global_batch_size": 8,
        "max_epochs": 50,
        "precision": "FP32",
        "amp": False,
        "training_seed": seed,
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
    }
    for field, expected in expected_run.items():
        _require_equal(run_contract.get(field), expected, f"{context} {field}")
    config = _mapping(run_contract.get("config"), f"{context} config")
    _require_equal(
        config.get("config_sha256"),
        SUBJECT_CONFIG_SHA256[subject],
        f"{context} config SHA",
    )
    teacher = _mapping(run_contract.get("teacher"), f"{context} teacher")
    for field, expected in {
        "task_id": TEACHER_TASK_ID,
        "model_id": TEACHER_MODEL_ID,
        "sha256": TEACHER_CHECKPOINT_SHA256,
        "expected_sha256": TEACHER_CHECKPOINT_SHA256,
    }.items():
        _require_equal(teacher.get(field), expected, f"{context} teacher {field}")
    checkpoint = _artifact_mapping(
        artifacts["final_checkpoint_contract"], f"{context} checkpoint contract"
    )
    if checkpoint.get("filename") != "epoch_50.pth":
        raise P1ExecutorError(
            f"{context} did not publish the epoch-50 final checkpoint"
        )
    model_id = _clearml_id(checkpoint.get("model_id"), f"{context} model")
    checkpoint_sha = _sha256(checkpoint.get("sha256"), f"{context} checkpoint")
    if checkpoint.get("name") != f"ResilientV2X {subject} final checkpoint":
        raise P1ExecutorError(f"{context} final model name drifted")
    outputs = task.get_models().get("output", [])
    if (
        len([model for model in outputs if str(getattr(model, "id", "")) == model_id])
        != 1
    ):
        raise P1ExecutorError(f"{context} output model binding drifted")
    return {
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha,
        "runtime": runtime_receipt,
    }


def _validate_metrics(
    value: Mapping[str, object], *, subject: str, checkpoint_sha256: str
) -> None:
    metrics = _mapping(value, f"{subject} metrics")
    _exact_keys(
        metrics,
        {
            "baseline",
            "checkpoint",
            "checkpoint_sha256",
            "complete",
            "expected_ground_truth_count",
            "expected_sample_count",
            "expected_unsupported_sample_count",
            "manifest_content_sha256",
            "overlay_index_content_sha256",
            "planned_run_count",
            "protocol_id",
            "result_type",
            "runs",
            "sample_ids_sha256",
            "schema_version",
        },
        f"{subject} metrics",
    )
    expected = {
        "schema_version": 1,
        "baseline": subject,
        "checkpoint_sha256": checkpoint_sha256,
        "complete": True,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_unsupported_sample_count": 0,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "planned_run_count": 12,
        "protocol_id": PROTOCOL_ID,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
    }
    for field, item in expected.items():
        _require_equal(metrics.get(field), item, f"{subject} metrics {field}")
    runs = metrics.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise P1ExecutorError(f"{subject} metrics must contain 12 runs")
    if type(metrics.get("checkpoint")) is not str or not metrics["checkpoint"]:
        raise P1ExecutorError(f"{subject} metrics checkpoint path is invalid")
    observed_ids: list[str] = []
    condition_names = {"full": "Full", "l_fail": "L-Fail", "c_fail": "C-Fail"}
    for index, run in enumerate(runs):
        run = _mapping(run, f"{subject} metric run")
        _exact_keys(
            run,
            {
                "condition",
                "condition_id",
                "delay_ms",
                "ground_truth_count",
                "metrics",
                "prediction_content_sha256",
                "prediction_sha256",
                "predictions",
                "sample_count",
                "sample_ids_sha256",
                "unsupported_sample_count",
            },
            f"{subject} metric run {index}",
        )
        observed_ids.append(str(run.get("condition_id") or ""))
        condition_id = CONDITION_IDS[index]
        suffix = condition_id.split("_", 2)[2]
        expected_run = {
            "condition_id": condition_id,
            "delay_ms": (0, 100, 200, 300)[index // 3],
            "condition": condition_names[suffix],
        }
        for field, expected_value in expected_run.items():
            _require_equal(
                run.get(field), expected_value, f"{subject} run {index} {field}"
            )
        for field, expected_value in {
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": 0,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
        }.items():
            _require_equal(run.get(field), expected_value, f"{subject} run {field}")
        if type(run.get("predictions")) is not str or not run["predictions"]:
            raise P1ExecutorError(f"{subject} run predictions path is invalid")
        _sha256(run.get("prediction_sha256"), f"{subject} run prediction")
        _sha256(
            run.get("prediction_content_sha256"),
            f"{subject} run prediction content",
        )
        values = _mapping(run.get("metrics"), f"{subject} run metrics")
        _exact_keys(values, set(METRIC_KEYS), f"{subject} run metric inventory")
        for key, number in values.items():
            if type(number) is not float or not math.isfinite(number):
                raise P1ExecutorError(f"{subject} run metric {key} is not finite float")
            if key in AP_METRIC_KEYS and not 0.0 <= number <= 100.0:
                raise P1ExecutorError(f"{subject} run metric {key} is outside [0, 100]")
            if key in UNIT_INTERVAL_METRIC_KEYS and not 0.0 <= number <= 1.0:
                raise P1ExecutorError(f"{subject} run metric {key} is outside [0, 1]")
            if key in NONNEGATIVE_METRIC_KEYS and number < 0.0:
                raise P1ExecutorError(f"{subject} run metric {key} is negative")
            if key in INTEGER_METRIC_KEYS and not number.is_integer():
                raise P1ExecutorError(f"{subject} run metric {key} is not integral")
        fixed_counts = {
            "resilient_v2x/sample_count": SAMPLE_COUNT,
            "resilient_v2x/car_ground_truth_count": GROUND_TRUTH_COUNT,
            "resilient_v2x/unsupported_sample_count": 0,
        }
        for key, expected_count in fixed_counts.items():
            if values[key] != float(expected_count):
                raise P1ExecutorError(f"{subject} run metric {key} count drifted")
        if values["resilient_v2x/car_prediction_count"] > SAMPLE_COUNT * 100:
            raise P1ExecutorError(f"{subject} prediction count exceeds safe bound")
        if (
            values["resilient_v2x/diagnostic_bev_match_050_count"]
            > values["resilient_v2x/car_prediction_count"]
        ):
            raise P1ExecutorError(f"{subject} diagnostic match count is impossible")
    if observed_ids != list(CONDITION_IDS):
        raise P1ExecutorError(f"{subject} metric condition order drifted")


def _validate_evaluation_plan(
    value: Mapping[str, object], *, subject: str, checkpoint_sha256: str
) -> None:
    plan = _mapping(value, f"{subject} evaluation plan")
    _exact_keys(plan, EVALUATION_PLAN_KEYS, f"{subject} evaluation plan")
    detached = dict(plan)
    content_sha = _sha256(
        detached.pop("content_sha256", None), f"{subject} evaluation plan content"
    )
    if content_sha != _content_sha256(detached):
        raise P1ExecutorError(f"{subject} evaluation plan content hash mismatch")
    expected = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_evaluation",
        "protocol_id": PROTOCOL_ID,
        "baseline": subject,
        "baseline_config_sha256": SUBJECT_CONFIG_SHA256[subject],
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": 0,
        "delays_ms": [0, 100, 200, 300],
        "conditions": ["Full", "L-Fail", "C-Fail"],
    }
    for field, expected_value in expected.items():
        _require_equal(
            plan.get(field), expected_value, f"{subject} evaluation plan {field}"
        )
    for field in (
        "baseline_config",
        "checkpoint",
        "data_root",
        "manifest",
        "metrics_output",
        "overlay_index",
        "plan_path",
        "work_dir",
    ):
        if type(plan.get(field)) is not str or not plan[field]:
            raise P1ExecutorError(f"{subject} evaluation plan {field} is invalid")
    for field in (
        "manifest_file_sha256",
        "overlay_index_file_sha256",
        "split_sha256",
    ):
        _sha256(plan.get(field), f"{subject} evaluation plan {field}")
    sample_ids = plan.get("sample_ids")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != SAMPLE_COUNT
        or len(set(sample_ids)) != SAMPLE_COUNT
        or any(type(item) is not str or not item for item in sample_ids)
        or _content_sha256(sample_ids) != SAMPLE_IDS_SHA256
    ):
        raise P1ExecutorError(f"{subject} evaluation plan sample cohort drifted")
    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != len(CONDITION_IDS):
        raise P1ExecutorError(f"{subject} evaluation plan must contain 12 runs")
    condition_names = {"full": "Full", "l_fail": "L-Fail", "c_fail": "C-Fail"}
    for index, (run, condition_id) in enumerate(zip(runs, CONDITION_IDS, strict=True)):
        run = _mapping(run, f"{subject} evaluation plan run {index}")
        _exact_keys(
            run,
            EVALUATION_PLAN_RUN_KEYS,
            f"{subject} evaluation plan run {index}",
        )
        suffix = condition_id.split("_", 2)[2]
        for field, expected_value in {
            "condition_id": condition_id,
            "delay_ms": (0, 100, 200, 300)[index // 3],
            "condition": condition_names[suffix],
        }.items():
            _require_equal(
                run.get(field),
                expected_value,
                f"{subject} evaluation plan run {index} {field}",
            )
        for field in (
            "fault_overlay_sha256",
            "resolved_config_sha256",
            "transport_overlay_sha256",
        ):
            _sha256(run.get(field), f"{subject} evaluation plan run {index} {field}")
        for field in (
            "checkpoint_sha256_file",
            "condition_config",
            "fault_overlay",
            "predictions",
            "resolved_config",
            "transport_overlay",
        ):
            if type(run.get(field)) is not str or not run[field]:
                raise P1ExecutorError(
                    f"{subject} evaluation plan run {index} {field} is invalid"
                )


def _validate_evidence_descriptor(artifact: object, *, subject: str) -> None:
    artifact_hash = _sha256(
        getattr(artifact, "hash", None), f"{subject} evaluation evidence artifact"
    )
    if not artifact_hash:
        raise P1ExecutorError(f"{subject} evaluation evidence hash is missing")
    if str(getattr(artifact, "type", "") or "") != "archive":
        raise P1ExecutorError(f"{subject} evaluation evidence is not an archive")
    url = str(getattr(artifact, "url", "") or "")
    if not url.startswith(FILES_SERVER_URI + "/") or not url.endswith(
        f"/{subject}.zip"
    ):
        raise P1ExecutorError(f"{subject} evaluation evidence URL drifted")


def _validate_evaluation_result(
    task: object,
    *,
    subject: str,
    seed: int,
    controller_id: str,
    training_task_id: str,
    model: Mapping[str, object],
    source_d: str,
) -> dict[str, object]:
    context = f"{subject} seed {seed} evaluation"
    if _task_status(task) != "completed" or _task_parent(task) != controller_id:
        raise P1ExecutorError(f"{context} status or parent drifted")
    runtime_receipt = _validate_task_runtime(task, context)
    expected_parameters = _evaluation_parameters(
        subject,
        seed,
        training_task_id=training_task_id,
        model_id=str(model["model_id"]),
        checkpoint_sha256=str(model["checkpoint_sha256"]),
    )
    _require_script(
        task,
        entry_point=TASK_ENTRY_POINT,
        sha256=SOURCE_D_SCRIPT_SHA256,
        source=source_d,
        context=context,
    )
    _require_parameters(task, expected_parameters, context)
    artifacts = _artifact_inventory(task)
    expected_artifacts = {
        "run_contract",
        "evaluation_plan",
        "controlled_baseline_metrics",
        "controlled_baseline_evidence",
    }
    if set(artifacts) != expected_artifacts:
        raise P1ExecutorError(f"{context} artifact inventory drifted")
    run_contract = _artifact_mapping(
        artifacts["run_contract"], f"{context} run contract"
    )
    expected_run = {
        "task_id": str(getattr(task, "id", "")),
        "mode": "baseline_validate",
        "baseline": subject,
        "baseline_task_id": training_task_id,
        "predecessor_task_id": training_task_id,
        "training_dataset_id": TRAINING_DATASET_ID,
        "protocol_id": PROTOCOL_ID,
        "expected_run_count": 12,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_sample_ids_sha256": SAMPLE_IDS_SHA256,
        "expected_manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "expected_overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "evaluator_sha256": CONTROLLED_EVALUATOR_SHA256,
        "training_seed": seed,
        "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
    }
    for field, expected in expected_run.items():
        _require_equal(run_contract.get(field), expected, f"{context} {field}")
    checkpoint = _mapping(run_contract.get("checkpoint"), f"{context} checkpoint")
    for field, expected in {
        "task_id": training_task_id,
        "model_id": str(model["model_id"]),
        "sha256": str(model["checkpoint_sha256"]),
        "expected_sha256": str(model["checkpoint_sha256"]),
    }.items():
        _require_equal(checkpoint.get(field), expected, f"{context} checkpoint {field}")
    metrics = _artifact_mapping(
        artifacts["controlled_baseline_metrics"], f"{context} metrics"
    )
    _validate_metrics(
        metrics, subject=subject, checkpoint_sha256=str(model["checkpoint_sha256"])
    )
    evaluation_plan = _artifact_mapping(
        artifacts["evaluation_plan"], f"{context} evaluation plan"
    )
    _validate_evaluation_plan(
        evaluation_plan,
        subject=subject,
        checkpoint_sha256=str(model["checkpoint_sha256"]),
    )
    if metrics["checkpoint"] != evaluation_plan["checkpoint"]:
        raise P1ExecutorError(f"{context} metrics and plan checkpoint drifted")
    _validate_evidence_descriptor(
        artifacts["controlled_baseline_evidence"], subject=subject
    )
    for name, artifact in artifacts.items():
        _sha256(getattr(artifact, "hash", None), f"{context} artifact {name}")
    return {
        "task_id": str(getattr(task, "id", "")),
        "worker": _worker_id(task),
        "runtime": runtime_receipt,
        "metrics_artifact_sha256": str(
            getattr(artifacts["controlled_baseline_metrics"], "hash", "") or ""
        ),
        "prediction_evidence_artifact_sha256": str(
            getattr(artifacts["controlled_baseline_evidence"], "hash", "") or ""
        ),
    }


def _validate_cache_directory() -> Path:
    raw = os.environ.get("CLEARML_CACHE_DIR", "")
    if not raw:
        raise P1ExecutorError(
            "CLEARML_CACHE_DIR must be set for preflight or execution"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise P1ExecutorError(
            "CLEARML_CACHE_DIR must be an existing absolute directory"
        )
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise P1ExecutorError("CLEARML_CACHE_DIR must be a directory")
    return path


def _load_clearml() -> object:
    _validate_cache_directory()
    try:
        from clearml import Task
        from clearml.storage.helper import _HttpDriver
    except ImportError as error:
        raise P1ExecutorError(f"ClearML SDK is unavailable: {error}") from None
    _HttpDriver._file_server_hosts = [FILES_SERVER_URI]
    return Task


def _response_data(response: object, context: str) -> dict[str, object]:
    value = getattr(response, "response_data", None)
    if not isinstance(value, Mapping):
        raise P1ExecutorError(f"{context} returned no response data")
    return dict(value)


def _read_queue_snapshot(task_class: object) -> dict[str, object]:
    """Read queue and worker state through GET-only ClearML service requests."""

    session_getter = getattr(task_class, "_get_default_session", None)
    if not callable(session_getter):
        raise P1ExecutorError("ClearML cannot expose a read-only backend session")
    try:
        from clearml.backend_api.services.v2_13 import queues, workers
    except ImportError as error:
        raise P1ExecutorError(f"ClearML queue API is unavailable: {error}") from None
    session = session_getter()
    sender = getattr(session, "send", None)
    if not callable(sender):
        raise P1ExecutorError("ClearML backend session cannot send read-only requests")
    queue_response = _response_data(
        sender(queues.GetAllRequest(id=[QUEUE_ID], page_size=2)), "queue lookup"
    )
    queue_rows = queue_response.get("queues")
    if not isinstance(queue_rows, list) or len(queue_rows) != 1:
        raise P1ExecutorError("target queue identity is missing or ambiguous")
    queue = _mapping(queue_rows[0], "target queue")
    if queue.get("id") != QUEUE_ID or queue.get("name") != QUEUE_NAME:
        raise P1ExecutorError("target queue ID or name drifted")
    entries = queue.get("entries")
    if not isinstance(entries, list):
        raise P1ExecutorError("target queue entries are not readable")

    overlap_queue_response = _response_data(
        sender(
            queues.GetAllRequest(
                id=[OVERLAPPING_GPU8_QUEUE_ID],
                page_size=2,
            )
        ),
        "overlapping GPU8 queue lookup",
    )
    overlap_queue_rows = overlap_queue_response.get("queues")
    if not isinstance(overlap_queue_rows, list) or len(overlap_queue_rows) != 1:
        raise P1ExecutorError("overlapping GPU8 queue is missing or ambiguous")
    overlap_queue = _mapping(
        overlap_queue_rows[0],
        "overlapping GPU8 queue",
    )
    overlap_entries = overlap_queue.get("entries")
    overlap_tags = overlap_queue.get("tags")
    if (
        overlap_queue.get("id") != OVERLAPPING_GPU8_QUEUE_ID
        or overlap_queue.get("name") != OVERLAPPING_GPU8_QUEUE_NAME
        or not isinstance(overlap_entries, list)
        or bool(overlap_entries)
        or not isinstance(overlap_tags, list)
        or any(type(item) is not str or not item for item in overlap_tags)
        or len(overlap_tags) != len(set(overlap_tags))
        or GPU8_FREEZE_TAG not in overlap_tags
        or GPU8_ENABLE_TAG in overlap_tags
    ):
        raise P1ExecutorError(
            "overlapping GPU8 queue is not empty and force-workers frozen"
        )

    worker_response = _response_data(
        sender(workers.GetAllRequest(last_seen=3600)), "worker lookup"
    )
    worker_rows = worker_response.get("workers")
    if not isinstance(worker_rows, list):
        raise P1ExecutorError("ClearML workers are not readable")
    attached: list[dict[str, object]] = []
    overlapping_gpu8: list[dict[str, object]] = []
    for raw_worker in worker_rows:
        worker = _mapping(raw_worker, "ClearML worker")
        worker_id = str(worker.get("id") or "")
        worker_queues = worker.get("queues")
        if not isinstance(worker_queues, list):
            if worker_id == OVERLAPPING_GPU8_WORKER_ID:
                raise P1ExecutorError(
                    "overlapping GPU8 worker queue state is not readable"
                )
            continue
        if worker_id == OVERLAPPING_GPU8_WORKER_ID:
            running = worker.get("task")
            running_id = ""
            if isinstance(running, Mapping):
                running_id = str(running.get("id") or "")
                if running_id:
                    _clearml_id(running_id, "overlapping GPU8 worker running task")
            elif running not in (None, ""):
                raise P1ExecutorError(
                    "overlapping GPU8 worker task state is not readable"
                )
            if running_id:
                raise P1ExecutorError(
                    "overlapping GPU8 worker must be idle before A100 execution"
                )
            overlapping_gpu8.append(
                {
                    "worker_id": worker_id,
                    "status": "idle",
                    "queue_ids": sorted(
                        str(item.get("id") or "")
                        for item in worker_queues
                        if isinstance(item, Mapping)
                    ),
                    "running_task_id": None,
                }
            )
        matching = [
            _mapping(item, "worker queue")
            for item in worker_queues
            if isinstance(item, Mapping) and item.get("id") == QUEUE_ID
        ]
        if not matching:
            continue
        if len(matching) != 1 or matching[0].get("name") != QUEUE_NAME:
            raise P1ExecutorError("worker target queue binding drifted")
        if worker_id not in ALLOWED_WORKER_IDS:
            raise P1ExecutorError("target queue has a non-pinned worker")
        running = worker.get("task")
        running_id = ""
        if isinstance(running, Mapping):
            running_id = str(running.get("id") or "")
            if running_id:
                _clearml_id(running_id, "worker running task")
        attached.append(
            {
                "worker_id": worker_id,
                "running_task_id": running_id or None,
                "reported_queue_task_count": matching[0].get("num_tasks", 0),
            }
        )
    attached.sort(key=lambda item: str(item["worker_id"]))
    attached_worker_ids = [str(item["worker_id"]) for item in attached]
    if (
        len(attached_worker_ids) != len(ALLOWED_WORKER_IDS)
        or set(attached_worker_ids) != ALLOWED_WORKER_IDS
    ):
        raise P1ExecutorError("target queue worker inventory drifted")
    if len(overlapping_gpu8) > 1:
        raise P1ExecutorError("overlapping GPU8 worker inventory is ambiguous")
    return {
        "queue_id": QUEUE_ID,
        "queue_name": QUEUE_NAME,
        "queued_entry_count": len(entries),
        "queued_task_ids": [
            str(item.get("task") or item.get("id") or "")
            if isinstance(item, Mapping)
            else str(item)
            for item in entries
        ],
        "registered_worker_count": len(attached),
        "workers": attached,
        "overlapping_gpu8_queue": {
            "queue_id": OVERLAPPING_GPU8_QUEUE_ID,
            "queue_name": OVERLAPPING_GPU8_QUEUE_NAME,
            "entry_count": 0,
            "entry_task_ids": [],
            "tags": sorted(overlap_tags),
        },
        "overlapping_gpu8_worker": (
            overlapping_gpu8[0]
            if overlapping_gpu8
            else {
                "worker_id": OVERLAPPING_GPU8_WORKER_ID,
                "status": "absent",
                "queue_ids": [],
                "running_task_id": None,
            }
        ),
    }


def _validate_a100_queue_receipt(
    value: Mapping[str, object], *, context: str
) -> dict[str, object]:
    queue = _mapping(value, context)
    for field, expected in {"queue_id": QUEUE_ID, "queue_name": QUEUE_NAME}.items():
        _require_equal(queue.get(field), expected, f"{context} {field}")
    workers = queue.get("workers")
    worker_ids = (
        [str(item.get("worker_id") or "") for item in workers]
        if isinstance(workers, list)
        and all(isinstance(item, Mapping) for item in workers)
        else []
    )
    if (
        type(queue.get("registered_worker_count")) is not int
        or int(queue["registered_worker_count"]) != len(ALLOWED_WORKER_IDS)
        or not isinstance(workers, list)
        or len(workers) != queue["registered_worker_count"]
        or len(worker_ids) != len(ALLOWED_WORKER_IDS)
        or set(worker_ids) != ALLOWED_WORKER_IDS
    ):
        raise P1ExecutorError(f"{context} has no exact worker receipt")
    overlap_queue = _mapping(
        queue.get("overlapping_gpu8_queue"),
        f"{context} overlapping GPU8 queue",
    )
    _exact_keys(
        overlap_queue,
        {"entry_count", "entry_task_ids", "queue_id", "queue_name", "tags"},
        f"{context} overlapping GPU8 queue",
    )
    overlap_queue_tags = overlap_queue.get("tags")
    if (
        overlap_queue.get("queue_id") != OVERLAPPING_GPU8_QUEUE_ID
        or overlap_queue.get("queue_name") != OVERLAPPING_GPU8_QUEUE_NAME
        or overlap_queue.get("entry_count") != 0
        or type(overlap_queue.get("entry_count")) is not int
        or overlap_queue.get("entry_task_ids") != []
        or not isinstance(overlap_queue_tags, list)
        or any(type(item) is not str or not item for item in overlap_queue_tags)
        or overlap_queue_tags != sorted(overlap_queue_tags)
        or len(overlap_queue_tags) != len(set(overlap_queue_tags))
        or GPU8_FREEZE_TAG not in overlap_queue_tags
        or GPU8_ENABLE_TAG in overlap_queue_tags
    ):
        raise P1ExecutorError(f"{context} overlapping GPU8 queue is not safely frozen")
    overlapping = _mapping(
        queue.get("overlapping_gpu8_worker"),
        f"{context} overlapping GPU8 worker",
    )
    _exact_keys(
        overlapping,
        {"queue_ids", "running_task_id", "status", "worker_id"},
        f"{context} overlapping GPU8 worker",
    )
    queue_ids = overlapping.get("queue_ids")
    if (
        overlapping.get("worker_id") != OVERLAPPING_GPU8_WORKER_ID
        or overlapping.get("status") not in {"absent", "idle"}
        or overlapping.get("running_task_id") is not None
        or not isinstance(queue_ids, list)
        or any(type(item) is not str or not item for item in queue_ids)
        or QUEUE_ID in queue_ids
    ):
        raise P1ExecutorError(f"{context} overlapping GPU8 worker is not safely idle")
    return queue


def _api_client() -> object:
    try:
        from clearml.backend_api.session.client import APIClient
    except ImportError as error:
        raise P1ExecutorError(f"ClearML API client is unavailable: {error}") from None
    client = APIClient()
    session = getattr(client, "session", None)
    checker = getattr(session, "check_min_api_version", None)
    if not callable(checker) or not checker("2.23"):
        raise P1ExecutorError("ClearML server does not satisfy API v2.23")
    return client


def _lease_rows(api_client: object, execution_key: str) -> list[object]:
    service = getattr(api_client, "queues", None)
    getter = getattr(service, "get_all", None)
    if not callable(getter):
        raise P1ExecutorError("ClearML cannot query the execution mutex")
    name = _lease_queue_name(execution_key)
    rows = getter(
        name=f"^{re.escape(name)}$",
        search_hidden=True,
        max_task_entries=1,
    )
    if isinstance(rows, (str, bytes, Mapping)):
        raise P1ExecutorError("ClearML mutex query returned an invalid result")
    try:
        values = list(rows)
    except TypeError:
        raise P1ExecutorError("ClearML mutex query is not iterable") from None
    return [row for row in values if str(getattr(row, "name", "") or "") == name]


def _read_lease_snapshot(
    execution_key: str, *, api_client: object | None = None
) -> dict[str, object]:
    client = api_client or _api_client()
    rows = _lease_rows(client, execution_key)
    if len(rows) > 1:
        raise P1ExecutorError("ClearML contains duplicate execution mutex queues")
    if not rows:
        return {
            "name": _lease_queue_name(execution_key),
            "status": "available",
            "queue_id": None,
            "owner_tag": None,
        }
    row = rows[0]
    entries = getattr(row, "entries", None) or []
    if entries:
        raise P1ExecutorError("execution mutex queue unexpectedly contains tasks")
    tags = [str(item) for item in (getattr(row, "tags", None) or [])]
    owner_tags = [item for item in tags if item.startswith("owner:")]
    if len(owner_tags) != 1 or _execution_tag(execution_key) not in tags:
        raise P1ExecutorError("execution mutex ownership tags drifted")
    return {
        "name": _lease_queue_name(execution_key),
        "status": "held",
        "queue_id": _clearml_id(getattr(row, "id", None), "mutex queue"),
        "owner_tag": owner_tags[0],
    }


def _legacy_execution_tag() -> str:
    return LEGACY_EXECUTION_KEY_TAG_PREFIX + LEGACY_EXECUTION_KEY


def _legacy_controller_name() -> str:
    return f"ResilientV2X P1 scoped 4-train 4-eval {LEGACY_EXECUTION_KEY} controller"


def _legacy_task_name(task_key: str) -> str:
    if task_key not in LEGACY_TASK_KEYS:
        raise P1ExecutorError("legacy task key is outside the sealed 5090 scope")
    return f"ResilientV2X P1 multiseed {LEGACY_EXECUTION_KEY} {task_key}"


def _query_exact_tagged_tasks(task_class: object, tag: str) -> list[object]:
    query = getattr(task_class, "query_tasks", None)
    if not callable(query):
        raise P1ExecutorError("ClearML cannot query legacy execution-tagged tasks")
    rows = query(
        tags=["__$all", tag],
        additional_return_fields=[
            "parent",
            "name",
            "status",
            "project",
            "tags",
            "system_tags",
        ],
        task_filter={"search_hidden": True},
    )
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise P1ExecutorError("legacy execution-tag query returned an invalid result")
    task_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise P1ExecutorError("legacy execution-tag query omitted requested fields")
        tags = row.get("tags")
        if (
            not isinstance(tags, Sequence)
            or isinstance(tags, (str, bytes))
            or tag not in tags
        ):
            raise P1ExecutorError("legacy execution-tag query binding drifted")
        task_ids.append(_clearml_id(row.get("id"), "legacy execution-tagged task"))
    if len(task_ids) != len(set(task_ids)):
        raise P1ExecutorError("legacy execution-tag query returned duplicate IDs")
    tasks = [task_class.get_task(task_id=task_id) for task_id in task_ids]
    if any(tag not in _task_tags(task) for task in tasks):
        raise P1ExecutorError("legacy execution tag did not round-trip")
    return tasks


def _read_legacy_mutex_snapshot(api_client: object) -> dict[str, object]:
    service = getattr(api_client, "queues", None)
    getter = getattr(service, "get_all", None)
    if not callable(getter):
        raise P1ExecutorError("ClearML cannot query the legacy execution mutex")
    rows = getter(
        name=f"^{re.escape(LEGACY_MUTEX_NAME)}$",
        search_hidden=True,
        max_task_entries=1,
    )
    if isinstance(rows, (str, bytes, Mapping)):
        raise P1ExecutorError("legacy mutex query returned an invalid result")
    try:
        exact = [
            row
            for row in rows
            if str(getattr(row, "name", "") or "") == LEGACY_MUTEX_NAME
        ]
    except TypeError:
        raise P1ExecutorError("legacy mutex query is not iterable") from None
    if exact:
        raise P1ExecutorError(
            "legacy 5090 execution mutex still exists; retirement is incomplete"
        )
    return {
        "name": LEGACY_MUTEX_NAME,
        "status": "available",
        "queue_id": None,
    }


def _read_legacy_queue_snapshot(api_client: object) -> dict[str, object]:
    service = getattr(api_client, "queues", None)
    getter = getattr(service, "get_by_id", None)
    counter = getattr(service, "get_num_entries", None)
    if not callable(getter) or not callable(counter):
        raise P1ExecutorError("ClearML cannot verify the legacy 5090 queue")
    queue = getter(queue=LEGACY_QUEUE_ID, max_task_entries=1)
    entries = getattr(queue, "entries", None)
    count = getattr(counter(queue=LEGACY_QUEUE_ID), "num", None)
    if (
        str(getattr(queue, "id", "") or "") != LEGACY_QUEUE_ID
        or str(getattr(queue, "name", "") or "") != LEGACY_QUEUE_NAME
        or not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes, Mapping))
        or bool(entries)
        or type(count) is not int
        or count != 0
    ):
        raise P1ExecutorError(
            "legacy 5090 live queue entries do not prove the child is unqueued"
        )
    return {
        "queue_id": LEGACY_QUEUE_ID,
        "queue_name": LEGACY_QUEUE_NAME,
        "entry_count": 0,
        "entry_task_ids": [],
    }


def _read_legacy_mutex_journal_snapshot() -> dict[str, object]:
    path = _validate_cache_directory() / LEGACY_MUTEX_JOURNAL_NAME
    if path.exists() or path.is_symlink():
        raise P1ExecutorError(
            "legacy 5090 local mutex journal still exists; retirement is incomplete"
        )
    return {
        "name": LEGACY_MUTEX_JOURNAL_NAME,
        "status": "absent",
    }


def _output_model_count(task: object, context: str) -> int:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise P1ExecutorError(f"{context} cannot expose its model inventory")
    models = getter()
    if not isinstance(models, Mapping):
        raise P1ExecutorError(f"{context} model inventory is invalid")
    output = models.get("output") or []
    if isinstance(output, (str, bytes, Mapping)):
        raise P1ExecutorError(f"{context} output-model inventory is invalid")
    try:
        return len(list(output))
    except TypeError:
        raise P1ExecutorError(
            f"{context} output-model inventory is not iterable"
        ) from None


def _validate_legacy_retirement_manifest(
    value: Mapping[str, object],
    *,
    intent_seal_sha256: str,
    plan: Mapping[str, object],
    pinset: Mapping[str, object],
    execution_key: str,
) -> dict[str, object]:
    manifest = _mapping(value, "legacy 5090 retirement manifest")
    if manifest.get("seal_sha256") != _seal(manifest):
        raise P1ExecutorError("legacy 5090 retirement manifest seal mismatch")
    expected = {
        "schema_version": 1,
        "document_type": LEGACY_RETIREMENT_DOCUMENT_TYPE,
        "phase": "retired",
        "status": "retired",
        "reason": "homogeneous_a100_migration",
        "execution_key": LEGACY_EXECUTION_KEY,
        "legacy_executor_sha256": LEGACY_EXECUTOR_SHA256,
        "controller_task_id": LEGACY_CONTROLLER_ID,
        "child_task_id": LEGACY_CHILD_ID,
        "child_task_key": LEGACY_CHILD_TASK_KEY,
        "legacy_queue_id": LEGACY_QUEUE_ID,
        "legacy_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
        "legacy_mutex_name": LEGACY_MUTEX_NAME,
        "retirement_intent_seal_sha256": intent_seal_sha256,
        "intent_target_a100_executor_sha256": INTENT_TARGET_A100_EXECUTOR_SHA256,
        "final_target_a100_executor_sha256": _executor_sha256(),
        "target_a100_executor_amendment_reason": A100_EXECUTOR_AMENDMENT_REASON,
        "target_a100_plan_seal_sha256": plan["seal_sha256"],
        "target_a100_pinset_seal_sha256": pinset["seal_sha256"],
        "target_a100_execution_key": execution_key,
    }
    for field, expected_value in expected.items():
        _require_equal(
            manifest.get(field),
            expected_value,
            f"legacy 5090 retirement manifest {field}",
        )
    initial = _mapping(
        manifest.get("observed_initial_state"),
        "legacy retirement initial state",
    )
    required_initial = {
        "controller_status": "created",
        "child_status": "queued",
        "legacy_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
        "original_controller_artifacts": [
            "p1_multiseed_execution_journal",
            "p1_multiseed_pinset",
            "p1_multiseed_plan",
        ],
    }
    _require_equal(initial, required_initial, "legacy retirement initial state")
    proofs = _mapping(manifest.get("retirement_proofs"), "legacy retirement proof set")
    expected_proofs = {
        "controller_barrier_tag": LEGACY_RETIREMENT_TAG,
        "legacy_quarantine_tag": LEGACY_ORPHAN_TAG,
        "child_dequeued_to_created_state": True,
        "queue_residency_authority": "clearml_queue_entries",
        "child_historical_execution_queue_id": LEGACY_QUEUE_ID,
        "child_absent_from_legacy_queue_entries": True,
        "legacy_queue_entry_task_ids": [],
        "legacy_queue_entry_count": 0,
        "child_never_claimed_by_worker": True,
        "child_artifact_count": 0,
        "child_output_model_count": 0,
        "other_seven_stable_children_absent": True,
        "no_queued_or_running_scoped_child": True,
        "final_child_status": "created",
        "final_controller_status": "created",
        "final_tasks_archived": False,
        "final_tasks_quarantined": True,
    }
    _require_equal(proofs, expected_proofs, "legacy retirement proof set")
    return manifest


def _validate_legacy_retirement_intent(
    value: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    pinset: Mapping[str, object],
    execution_key: str,
) -> dict[str, object]:
    intent = _mapping(value, "legacy 5090 retirement intent")
    if intent.get("seal_sha256") != _seal(intent):
        raise P1ExecutorError("legacy 5090 retirement intent seal mismatch")
    expected = {
        "schema_version": 1,
        "document_type": LEGACY_RETIREMENT_DOCUMENT_TYPE,
        "phase": "retiring",
        "status": "retiring",
        "reason": "homogeneous_a100_migration",
        "target_a100_plan_seal_sha256": plan["seal_sha256"],
        "target_a100_pinset_seal_sha256": pinset["seal_sha256"],
        "target_a100_execution_key": execution_key,
    }
    for field, expected_value in expected.items():
        _require_equal(
            intent.get(field), expected_value, f"legacy retirement intent {field}"
        )
    legacy = _mapping(intent.get("legacy"), "legacy retirement intent binding")
    _require_equal(
        legacy,
        {
            "execution_key": LEGACY_EXECUTION_KEY,
            "executor_sha256": LEGACY_EXECUTOR_SHA256,
            "controller_task_id": LEGACY_CONTROLLER_ID,
            "child_task_id": LEGACY_CHILD_ID,
            "child_task_key": LEGACY_CHILD_TASK_KEY,
            "queue_id": LEGACY_QUEUE_ID,
            "observed_mutex_queue_id": LEGACY_MUTEX_QUEUE_ID,
            "mutex_name": LEGACY_MUTEX_NAME,
            "original_controller_artifacts": [
                "p1_multiseed_execution_journal",
                "p1_multiseed_pinset",
                "p1_multiseed_plan",
            ],
        },
        "legacy retirement intent binding",
    )
    _require_equal(
        intent.get("intent"),
        {
            "install_controller_quarantine_barrier": True,
            "dequeue_only_if_never_claimed": True,
            "quarantine_created_legacy_scope": True,
            "never_stop_or_archive_tasks": True,
        },
        "legacy retirement intent actions",
    )
    return intent


def _read_legacy_retirement_snapshot(
    task_class: object,
    *,
    plan: Mapping[str, object],
    pinset: Mapping[str, object],
    execution_key: str,
    api_client: object | None = None,
) -> dict[str, object]:
    controller_matches = _query_named_tasks(task_class, name=_legacy_controller_name())
    if len(controller_matches) != 1:
        raise P1ExecutorError(
            "legacy retirement requires exactly one hidden stable controller"
        )
    controller = controller_matches[0]
    controller_id = _clearml_id(getattr(controller, "id", None), "legacy controller")
    if controller_id != LEGACY_CONTROLLER_ID:
        raise P1ExecutorError("legacy controller ID drifted")

    child_by_key = {
        task_key: _query_named_tasks(task_class, name=_legacy_task_name(task_key))
        for task_key in LEGACY_TASK_KEYS
    }
    if len(child_by_key[LEGACY_CHILD_TASK_KEY]) != 1 or any(
        tasks
        for task_key, tasks in child_by_key.items()
        if task_key != LEGACY_CHILD_TASK_KEY
    ):
        raise P1ExecutorError("legacy stable child inventory is not exactly retired")
    child = child_by_key[LEGACY_CHILD_TASK_KEY][0]
    child_id = _clearml_id(getattr(child, "id", None), "legacy child")
    if child_id != LEGACY_CHILD_ID:
        raise P1ExecutorError("legacy child ID drifted")
    direct_ids = {
        _clearml_id(getattr(task, "id", None), "legacy controller descendant")
        for task in _query_direct_children(task_class, LEGACY_CONTROLLER_ID)
    }
    if direct_ids != {LEGACY_CHILD_ID}:
        raise P1ExecutorError("legacy controller descendant inventory drifted")
    tagged_ids = {
        _clearml_id(getattr(task, "id", None), "legacy execution-tagged task")
        for task in _query_exact_tagged_tasks(task_class, _legacy_execution_tag())
    }
    if tagged_ids != {LEGACY_CONTROLLER_ID, LEGACY_CHILD_ID}:
        raise P1ExecutorError("legacy global execution-tag inventory drifted")

    controller_tags = _task_tags(controller)
    child_tags = _task_tags(child)
    required_controller_tags = {
        _legacy_execution_tag(),
        LEGACY_CONTROLLER_TAG,
        LEGACY_ORPHAN_TAG,
        LEGACY_RETIREMENT_TAG,
    }
    required_child_tags = {
        _legacy_execution_tag(),
        f"p1-task-key:{LEGACY_CHILD_TASK_KEY}",
        LEGACY_ORPHAN_TAG,
        LEGACY_RETIREMENT_TAG,
    }
    if (
        _task_project(controller) != PROJECT_ID
        or _task_parent(controller) != SELECTOR_TASK_ID
        or str(getattr(controller, "name", "") or "") != _legacy_controller_name()
        or _task_status(controller) != "created"
        or _task_archived(controller)
        or _task_queue(controller)
        or not required_controller_tags.issubset(controller_tags)
    ):
        raise P1ExecutorError("legacy retired controller binding drifted")
    if (
        _task_project(child) != PROJECT_ID
        or _task_parent(child) != LEGACY_CONTROLLER_ID
        or str(getattr(child, "name", "") or "")
        != _legacy_task_name(LEGACY_CHILD_TASK_KEY)
        or _task_status(child) != "created"
        or _task_archived(child)
        or _task_queue(child) != LEGACY_QUEUE_ID
        or _worker_id(child)
        or not required_child_tags.issubset(child_tags)
        or _artifact_inventory(child)
        or _output_model_count(child, "legacy retired child") != 0
    ):
        raise P1ExecutorError("legacy retired child binding drifted")
    if {
        item
        for item in controller_tags
        if item.startswith(LEGACY_EXECUTION_KEY_TAG_PREFIX)
    } != {_legacy_execution_tag()} or {
        item for item in child_tags if item.startswith(LEGACY_EXECUTION_KEY_TAG_PREFIX)
    } != {_legacy_execution_tag()}:
        raise P1ExecutorError("legacy retirement execution tags drifted")
    if {item for item in child_tags if item.startswith("p1-task-key:")} != {
        f"p1-task-key:{LEGACY_CHILD_TASK_KEY}"
    }:
        raise P1ExecutorError("legacy retirement child-key tag drifted")

    artifacts = _artifact_inventory(controller)
    if set(artifacts) != LEGACY_CONTROLLER_ARTIFACTS:
        raise P1ExecutorError("legacy retired controller artifacts drifted")
    intent = _validate_legacy_retirement_intent(
        _artifact_mapping(
            artifacts[LEGACY_RETIREMENT_INTENT_ARTIFACT],
            "legacy 5090 retirement intent",
        ),
        plan=plan,
        pinset=pinset,
        execution_key=execution_key,
    )
    manifest = _validate_legacy_retirement_manifest(
        _artifact_mapping(
            artifacts[LEGACY_RETIREMENT_ARTIFACT],
            "legacy 5090 retirement manifest",
        ),
        intent_seal_sha256=str(intent["seal_sha256"]),
        plan=plan,
        pinset=pinset,
        execution_key=execution_key,
    )
    client = api_client or _api_client()
    legacy_queue = _read_legacy_queue_snapshot(client)
    mutex = _read_legacy_mutex_snapshot(client)
    journal = _read_legacy_mutex_journal_snapshot()
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_a100_legacy_retirement_gate",
        "status": "retired",
        "readonly": True,
        "remote_mutation_count": 0,
        "legacy_execution_key": LEGACY_EXECUTION_KEY,
        "target_a100_plan_seal_sha256": plan["seal_sha256"],
        "target_a100_pinset_seal_sha256": pinset["seal_sha256"],
        "target_a100_execution_key": execution_key,
        "controller_task_id": LEGACY_CONTROLLER_ID,
        "child_task_id": LEGACY_CHILD_ID,
        "retirement_manifest_seal_sha256": manifest["seal_sha256"],
        "intent_target_a100_executor_sha256": manifest[
            "intent_target_a100_executor_sha256"
        ],
        "final_target_a100_executor_sha256": manifest[
            "final_target_a100_executor_sha256"
        ],
        "target_a100_executor_amendment_reason": manifest[
            "target_a100_executor_amendment_reason"
        ],
        "legacy_queue": legacy_queue,
        "global_mutex": mutex,
        "local_mutex_journal": journal,
        "exact_inventory": {
            "controller_count": 1,
            "child_count": 1,
            "other_stable_child_count": 0,
            "execution_tagged_task_ids": [LEGACY_CONTROLLER_ID, LEGACY_CHILD_ID],
            "no_queued_or_running_child": True,
            "controller_status": "created",
            "child_status": "created",
            "tasks_archived": False,
            "tasks_quarantined": True,
            "queue_residency_authority": "clearml_queue_entries",
            "child_historical_execution_queue_id": LEGACY_QUEUE_ID,
            "child_absent_from_legacy_queue_entries": True,
            "legacy_queue_entry_task_ids": [],
            "legacy_queue_entry_count": 0,
        },
    }
    receipt["seal_sha256"] = _seal(receipt)
    return receipt


def _validate_legacy_retirement_receipt(
    value: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    pinset: Mapping[str, object],
    execution_key: str,
) -> dict[str, object]:
    receipt = _mapping(value, "legacy 5090 retirement gate")
    if receipt.get("seal_sha256") != _seal(receipt):
        raise P1ExecutorError("legacy 5090 retirement gate seal mismatch")
    expected = {
        "schema_version": 1,
        "document_type": "resilient_v2x_p1_a100_legacy_retirement_gate",
        "status": "retired",
        "readonly": True,
        "remote_mutation_count": 0,
        "legacy_execution_key": LEGACY_EXECUTION_KEY,
        "target_a100_plan_seal_sha256": plan["seal_sha256"],
        "target_a100_pinset_seal_sha256": pinset["seal_sha256"],
        "target_a100_execution_key": execution_key,
        "controller_task_id": LEGACY_CONTROLLER_ID,
        "child_task_id": LEGACY_CHILD_ID,
        "intent_target_a100_executor_sha256": INTENT_TARGET_A100_EXECUTOR_SHA256,
        "final_target_a100_executor_sha256": _executor_sha256(),
        "target_a100_executor_amendment_reason": A100_EXECUTOR_AMENDMENT_REASON,
    }
    for field, expected_value in expected.items():
        _require_equal(
            receipt.get(field), expected_value, f"legacy retirement gate {field}"
        )
    _sha256(
        receipt.get("retirement_manifest_seal_sha256"),
        "legacy retirement manifest seal",
    )
    _require_equal(
        receipt.get("legacy_queue"),
        {
            "queue_id": LEGACY_QUEUE_ID,
            "queue_name": LEGACY_QUEUE_NAME,
            "entry_count": 0,
            "entry_task_ids": [],
        },
        "legacy retirement live queue",
    )
    _require_equal(
        receipt.get("global_mutex"),
        {"name": LEGACY_MUTEX_NAME, "status": "available", "queue_id": None},
        "legacy retirement global mutex",
    )
    _require_equal(
        receipt.get("local_mutex_journal"),
        {"name": LEGACY_MUTEX_JOURNAL_NAME, "status": "absent"},
        "legacy retirement local mutex journal",
    )
    _require_equal(
        receipt.get("exact_inventory"),
        {
            "controller_count": 1,
            "child_count": 1,
            "other_stable_child_count": 0,
            "execution_tagged_task_ids": [LEGACY_CONTROLLER_ID, LEGACY_CHILD_ID],
            "no_queued_or_running_child": True,
            "controller_status": "created",
            "child_status": "created",
            "tasks_archived": False,
            "tasks_quarantined": True,
            "queue_residency_authority": "clearml_queue_entries",
            "child_historical_execution_queue_id": LEGACY_QUEUE_ID,
            "child_absent_from_legacy_queue_entries": True,
            "legacy_queue_entry_task_ids": [],
            "legacy_queue_entry_count": 0,
        },
        "legacy retirement exact inventory",
    )
    return receipt


def _task_tags(task: object) -> set[str]:
    getter = getattr(task, "get_tags", None)
    value = getter() if callable(getter) else getattr(task, "tags", [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise P1ExecutorError("ClearML task tags are not readable")
    return {str(item) for item in value}


def _task_archived(task: object) -> bool:
    getter = getattr(task, "get_archived", None)
    return bool(getter()) if callable(getter) else False


def _is_quarantined(task: object) -> bool:
    if ORPHAN_TAG not in _task_tags(task):
        return False
    if _task_queue(task):
        raise P1ExecutorError("a quarantined orphan unexpectedly entered a queue")
    return True


def _query_named_tasks(task_class: object, *, name: str) -> list[object]:
    query = getattr(task_class, "query_tasks", None)
    if not callable(query):
        raise P1ExecutorError("ClearML cannot query hidden tasks for idempotency")
    rows = query(
        task_name=f"^{re.escape(name)}$",
        additional_return_fields=["name", "project", "status", "tags", "parent"],
        task_filter={
            "search_hidden": True,
            "order_by": ["-created"],
        },
    )
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise P1ExecutorError("ClearML named-task query returned an invalid result")
    task_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise P1ExecutorError("ClearML named-task query omitted requested fields")
        if row.get("name") != name:
            continue
        task_ids.append(_clearml_id(row.get("id"), "queried task"))
    exact = [task_class.get_task(task_id=task_id) for task_id in task_ids]
    ids = [_clearml_id(getattr(task, "id", None), "queried task") for task in exact]
    if len(ids) != len(set(ids)):
        raise P1ExecutorError("ClearML named-task query returned duplicate IDs")
    if any(str(getattr(task, "name", "") or "") != name for task in exact):
        raise P1ExecutorError("ClearML named-task name did not round-trip")
    return exact


def _query_direct_children(task_class: object, controller_id: str) -> list[object]:
    controller_id = _clearml_id(controller_id, "controller")
    query = getattr(task_class, "query_tasks", None)
    if not callable(query):
        raise P1ExecutorError("ClearML cannot query controller descendants")
    rows = query(
        additional_return_fields=[
            "parent",
            "name",
            "status",
            "project",
            "type",
            "tags",
            "system_tags",
        ],
        task_filter={
            "parent": controller_id,
            "search_hidden": True,
        },
    )
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise P1ExecutorError("ClearML descendant query returned an invalid result")
    task_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise P1ExecutorError("ClearML descendant query omitted requested fields")
        if row.get("parent") != controller_id:
            raise P1ExecutorError("ClearML descendant query parent binding drifted")
        task_ids.append(_clearml_id(row.get("id"), "controller descendant"))
    if len(task_ids) != len(set(task_ids)):
        raise P1ExecutorError("ClearML descendant query returned duplicate IDs")
    exact = [task_class.get_task(task_id=task_id) for task_id in task_ids]
    for task in exact:
        if _task_parent(task) != controller_id:
            raise P1ExecutorError("controller descendant parent did not round-trip")
    return exact


def _query_execution_tagged_tasks(
    task_class: object, execution_key: str
) -> list[object]:
    expected_tag = _execution_tag(execution_key)
    query = getattr(task_class, "query_tasks", None)
    if not callable(query):
        raise P1ExecutorError("ClearML cannot query execution-tagged tasks")
    rows = query(
        tags=["__$all", expected_tag],
        additional_return_fields=[
            "parent",
            "name",
            "status",
            "project",
            "tags",
            "system_tags",
        ],
        task_filter={"search_hidden": True},
    )
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise P1ExecutorError("ClearML execution-tag query returned an invalid result")
    task_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise P1ExecutorError(
                "ClearML execution-tag query omitted requested fields"
            )
        tags = row.get("tags")
        if (
            not isinstance(tags, Sequence)
            or isinstance(tags, (str, bytes))
            or expected_tag not in tags
        ):
            raise P1ExecutorError("ClearML execution-tag query tag binding drifted")
        task_ids.append(_clearml_id(row.get("id"), "execution-tagged task"))
    if len(task_ids) != len(set(task_ids)):
        raise P1ExecutorError("ClearML execution-tag query returned duplicate IDs")
    exact = [task_class.get_task(task_id=task_id) for task_id in task_ids]
    for task in exact:
        if expected_tag not in _task_tags(task):
            raise P1ExecutorError("execution tag did not round-trip")
    return exact


def _active_named_tasks(task_class: object, *, name: str) -> list[object]:
    return [
        task
        for task in _query_named_tasks(task_class, name=name)
        if not _is_quarantined(task)
    ]


def _validate_quarantined_preflight_controller(
    task_class: object,
    task: object,
    *,
    plan: Mapping[str, object],
    pinset: Mapping[str, object],
    execution_key: str,
) -> set[str]:
    task_id = _clearml_id(getattr(task, "id", None), "quarantined controller")
    tags = _task_tags(task)
    expected_execution_tag = _execution_tag(execution_key)
    execution_tags = {
        item for item in tags if item.startswith(EXECUTION_KEY_TAG_PREFIX)
    }
    task_key_tags = {item for item in tags if item.startswith("p1-task-key:")}
    if (
        not _is_quarantined(task)
        or _task_project(task) != PROJECT_ID
        or str(getattr(task, "name", "") or "") != _controller_name(execution_key)
        or _task_parent(task) not in {"", SELECTOR_TASK_ID}
        or _task_status(task) not in {"created", "failed"}
        or not execution_tags.issubset({expected_execution_tag})
        or task_key_tags
    ):
        raise P1ExecutorError(
            "preflight quarantined controller identity binding drifted"
        )
    expected_parameters = _controller_parameters(
        plan=plan,
        pinset=pinset,
        execution_key=execution_key,
    )
    observed_parameters = _parameters(task)
    if not set(observed_parameters).issubset(expected_parameters) or any(
        not _normalize_parameter(observed_parameters.get(key), expected_parameters[key])
        for key in observed_parameters
    ):
        raise P1ExecutorError("preflight quarantined controller parameters drifted")
    artifacts = _artifact_inventory(task)
    initialization = {PLAN_ARTIFACT: plan, PINSET_ARTIFACT: pinset}
    if not set(artifacts).issubset(initialization):
        raise P1ExecutorError("preflight quarantined controller has durable artifacts")
    for name, expected in initialization.items():
        artifact = artifacts.get(name)
        if artifact is not None and _artifact_mapping(artifact, name) != expected:
            raise P1ExecutorError(f"preflight quarantined controller {name} drifted")

    expected_children: dict[str, str] = {}
    for pair in plan["pairs"]:
        pair = _mapping(pair, "quarantined controller pair")
        for phase in ("training", "evaluation"):
            record = _mapping(pair[phase], f"quarantined controller {phase}")
            task_key = str(record["task_key"])
            expected_children[_task_name(execution_key, task_key)] = task_key
    descendant_ids: set[str] = set()
    for child in _query_direct_children(task_class, task_id):
        child_name = str(getattr(child, "name", "") or "")
        task_key = expected_children.get(child_name)
        child_tags = _task_tags(child)
        child_execution_tags = {
            item for item in child_tags if item.startswith(EXECUTION_KEY_TAG_PREFIX)
        }
        child_task_key_tags = {
            item for item in child_tags if item.startswith("p1-task-key:")
        }
        if (
            task_key is None
            or not _is_quarantined(child)
            or _task_project(child) != PROJECT_ID
            or _task_status(child) not in {"created", "failed"}
            or not child_execution_tags.issubset({expected_execution_tag})
            or not child_task_key_tags.issubset({f"p1-task-key:{task_key}"})
        ):
            raise P1ExecutorError(
                "preflight quarantined controller descendant inventory drifted"
            )
        descendant_ids.add(
            _clearml_id(
                getattr(child, "id", None),
                "quarantined controller descendant",
            )
        )
    return descendant_ids


def preflight(
    *,
    task_class: object | None = None,
    queue_reader: Callable[[object], Mapping[str, object]] | None = None,
    lease_reader: Callable[[str], Mapping[str, object]] | None = None,
    legacy_retirement_reader: Callable[
        [object, Mapping[str, object], Mapping[str, object], str],
        Mapping[str, object],
    ]
    | None = None,
) -> dict[str, object]:
    """Validate every remote pin and return a sealed, read-only receipt."""

    plan = build_plan()
    pinset = build_pinset()
    execution_key = _execution_key(plan, pinset)
    Task = task_class or _load_clearml()
    selector = Task.get_task(task_id=SELECTOR_TASK_ID)
    source_d_task = Task.get_task(task_id=SOURCE_D_TASK_ID)
    teacher_gate = Task.get_task(task_id=TEACHER_GATE_TASK_ID)
    teacher = Task.get_task(task_id=TEACHER_TASK_ID)
    native_build = Task.get_task(task_id=NATIVE_BUILD_TASK_ID)
    _validate_selector(selector)
    source_d = _validate_source_d(source_d_task)
    _validate_teacher_gate(teacher_gate)
    _validate_teacher(teacher)
    native_build_receipt = _validate_native_build_task(native_build)
    _validate_anchor_tasks(Task)
    queue = _validate_a100_queue_receipt(
        (queue_reader or _read_queue_snapshot)(Task),
        context="preflight queue receipt",
    )
    lease = _mapping(
        (lease_reader or _read_lease_snapshot)(execution_key),
        "preflight mutex receipt",
    )
    if lease.get("name") != _lease_queue_name(execution_key) or lease.get(
        "status"
    ) not in {"available", "held"}:
        raise P1ExecutorError("preflight mutex receipt drifted")
    if legacy_retirement_reader is None:
        legacy_retirement = _read_legacy_retirement_snapshot(
            Task,
            plan=plan,
            pinset=pinset,
            execution_key=execution_key,
        )
    else:
        legacy_retirement = legacy_retirement_reader(Task, plan, pinset, execution_key)
    legacy_retirement = _validate_legacy_retirement_receipt(
        legacy_retirement,
        plan=plan,
        pinset=pinset,
        execution_key=execution_key,
    )

    controller_name = _controller_name(execution_key)
    controller_matches = _query_named_tasks(Task, name=controller_name)
    controllers: list[object] = []
    quarantined_controllers: list[object] = []
    known_controller_ids: set[str] = set()
    quarantined_controller_descendant_ids: set[str] = set()
    expected_tagged_ids: set[str] = set()
    expected_execution_tag = _execution_tag(execution_key)
    for task in controller_matches:
        task_id = _clearml_id(getattr(task, "id", None), "stable-name controller")
        known_controller_ids.add(task_id)
        tags = _task_tags(task)
        if expected_execution_tag in tags:
            expected_tagged_ids.add(task_id)
        if not _is_quarantined(task):
            controllers.append(task)
            continue
        quarantined_controller_descendant_ids.update(
            _validate_quarantined_preflight_controller(
                Task,
                task,
                plan=plan,
                pinset=pinset,
                execution_key=execution_key,
            )
        )
        quarantined_controllers.append(task)
    if len(controllers) > 1:
        raise P1ExecutorError("preflight found duplicate active execution controllers")
    controller_id = None
    child_inventory: list[dict[str, object]] = []
    known_child_ids: set[str] = set()
    if controllers:
        controller = controllers[0]
        _validate_preflight_controller(
            Task,
            controller,
            plan=plan,
            pinset=pinset,
            execution_key=execution_key,
        )
        controller_id = _clearml_id(getattr(controller, "id", None), "controller")
    for pair in plan["pairs"]:
        pair = _mapping(pair, "preflight pair")
        for phase in ("training", "evaluation"):
            record = _mapping(pair[phase], f"preflight {phase}")
            task_key = str(record["task_key"])
            matches = _query_named_tasks(
                Task,
                name=_task_name(execution_key, task_key),
            )
            quarantined = [task for task in matches if _is_quarantined(task)]
            active = [task for task in matches if not _is_quarantined(task)]
            known_child_ids.update(
                _clearml_id(getattr(task, "id", None), "stable-name child")
                for task in matches
            )
            if len(active) > 1:
                raise P1ExecutorError(
                    f"preflight found duplicate child {record['task_key']}"
                )
            expected_execution_tag = _execution_tag(execution_key)
            expected_task_key_tag = f"p1-task-key:{task_key}"
            for task in matches:
                tags = _task_tags(task)
                status = _task_status(task)
                if expected_execution_tag in tags:
                    expected_tagged_ids.add(
                        _clearml_id(getattr(task, "id", None), "execution-tagged child")
                    )
                execution_tags = {
                    item for item in tags if item.startswith(EXECUTION_KEY_TAG_PREFIX)
                }
                task_key_tags = {
                    item for item in tags if item.startswith("p1-task-key:")
                }
                is_quarantined = ORPHAN_TAG in tags
                staged_created = False
                if not is_quarantined and status == "created" and not _task_queue(task):
                    models_getter = getattr(task, "get_models", None)
                    models = models_getter() if callable(models_getter) else None
                    staged_created = (
                        not _artifact_inventory(task)
                        and isinstance(models, Mapping)
                        and not models.get("output")
                    )
                tag_subsets_are_safe = execution_tags.issubset(
                    {expected_execution_tag}
                ) and task_key_tags.issubset({expected_task_key_tag})
                if (
                    _task_project(task) != PROJECT_ID
                    or not tag_subsets_are_safe
                    or (
                        not is_quarantined
                        and not staged_created
                        and (
                            execution_tags != {expected_execution_tag}
                            or task_key_tags != {expected_task_key_tag}
                        )
                    )
                    or (is_quarantined and status not in {"created", "failed"})
                    or (
                        not is_quarantined
                        and controller_id is not None
                        and _task_parent(task) != controller_id
                    )
                ):
                    raise P1ExecutorError(
                        f"preflight child {task_key} identity binding drifted"
                    )
                if is_quarantined:
                    _clearml_id(
                        _task_parent(task),
                        f"preflight quarantined child {task_key} parent",
                    )
            if active:
                if controller_id is None:
                    raise P1ExecutorError(
                        f"preflight found child {task_key} without a controller"
                    )
                if _task_parent(active[0]) != controller_id:
                    raise P1ExecutorError(
                        f"preflight child {task_key} parent binding drifted"
                    )
            child_inventory.append(
                {
                    "task_key": record["task_key"],
                    "task_id": (
                        _clearml_id(getattr(active[0], "id", None), "child task")
                        if active
                        else None
                    ),
                    "status": _task_status(active[0]) if active else "absent",
                    "quarantined_task_ids": [
                        _clearml_id(getattr(task, "id", None), "quarantined child task")
                        for task in quarantined
                    ],
                }
            )
    if not quarantined_controller_descendant_ids.issubset(known_child_ids):
        raise P1ExecutorError(
            "quarantined controller descendants differ from stable-name inventory"
        )
    tagged_tasks = _query_execution_tagged_tasks(Task, execution_key)
    observed_tagged_ids = {
        _clearml_id(getattr(task, "id", None), "execution-tagged task")
        for task in tagged_tasks
    }
    if observed_tagged_ids != expected_tagged_ids:
        raise P1ExecutorError(
            "execution-tag query differs from stable-name task inventory"
        )
    known_scoped_ids = known_controller_ids | known_child_ids
    for task_id in observed_tagged_ids:
        if task_id not in known_scoped_ids:
            raise P1ExecutorError("execution tag is attached to an unknown scoped task")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "document_type": PREFLIGHT_RECEIPT_TYPE,
        "status": "passed",
        "readonly": True,
        "remote_mutation_count": 0,
        "plan_seal_sha256": plan["seal_sha256"],
        "pinset_seal_sha256": pinset["seal_sha256"],
        "execution_key": execution_key,
        "source_d_script_bytes": len(source_d.encode("utf-8")),
        "validated_task_ids": {
            "selector": SELECTOR_TASK_ID,
            "source_d": SOURCE_D_TASK_ID,
            "teacher_gate": TEACHER_GATE_TASK_ID,
            "teacher": TEACHER_TASK_ID,
            "native_build": NATIVE_BUILD_TASK_ID,
            "anchor_training": [ANCHORS[item]["training_task_id"] for item in SUBJECTS],
            "anchor_evaluation": [
                ANCHORS[item]["evaluation_task_id"] for item in SUBJECTS
            ],
        },
        "queue": queue,
        "native_build_runtime": native_build_receipt,
        "global_mutex": lease,
        "legacy_5090_retirement": legacy_retirement,
        "existing_execution": {
            "controller_name": controller_name,
            "controller_id": controller_id,
            "controller_status": _task_status(controllers[0])
            if controllers
            else "absent",
            "quarantined_controller_ids": [
                _clearml_id(getattr(task, "id", None), "quarantined controller task")
                for task in quarantined_controllers
            ],
            "children": child_inventory,
        },
        "checks": {
            "selector": True,
            "source_d": True,
            "teacher_gate": True,
            "teacher": True,
            "native_build_runtime": True,
            "seed1_anchors": True,
            "queue": True,
            "plan_seal": True,
            "pinset_seal": True,
            "server_side_duplicates": True,
            "global_mutex": True,
            "legacy_5090_retired": True,
            "overlapping_gpu8_idle": True,
        },
    }
    receipt["seal_sha256"] = _seal(receipt)
    return receipt


def _upload_artifact(task: object, name: str, value: Mapping[str, object]) -> None:
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader):
        raise P1ExecutorError("controller cannot upload artifacts")
    result = uploader(name=name, artifact_object=dict(value), wait_on_upload=True)
    if result is False:
        raise P1ExecutorError(f"controller rejected artifact {name}")


def _mark_failed(task: object) -> None:
    marker = getattr(task, "mark_failed", None)
    if callable(marker):
        marker(force=True)


def _complete_controller(task_class: object, controller: object) -> object:
    controller_id = _clearml_id(getattr(controller, "id", None), "controller")
    marker = getattr(controller, "mark_completed", None)
    if not callable(marker):
        raise P1ExecutorError("controller cannot be marked completed")
    if marker(ignore_errors=False, force=True) is False:
        raise P1ExecutorError("controller rejected completed status")
    controller = _fresh(task_class, controller_id)
    if _task_status(controller) != "completed":
        raise P1ExecutorError("controller completed status did not round-trip")
    return controller


def _controller_parameters(
    *, plan: Mapping[str, object], pinset: Mapping[str, object], execution_key: str
) -> dict[str, object]:
    return {
        "Args/execution_key": execution_key,
        "Args/plan_seal_sha256": plan["seal_sha256"],
        "Args/pinset_seal_sha256": pinset["seal_sha256"],
        "Args/source_d_task_id": SOURCE_D_TASK_ID,
        "Args/teacher_quality_gate_task_id": TEACHER_GATE_TASK_ID,
        "Args/teacher_task_id": TEACHER_TASK_ID,
        "Args/queue_id": QUEUE_ID,
        "Args/max_parallel_tasks": 1,
    }


def _validate_controller(
    task: object,
    *,
    plan: Mapping[str, object],
    pinset: Mapping[str, object],
    execution_key: str,
) -> None:
    task_id = _clearml_id(getattr(task, "id", None), "controller")
    if (
        _task_project(task) != PROJECT_ID
        or _task_parent(task) != SELECTOR_TASK_ID
        or str(getattr(task, "name", "") or "") != _controller_name(execution_key)
        or _task_queue(task)
    ):
        raise P1ExecutorError(f"controller {task_id} identity binding drifted")
    if _task_status(task) not in ACTIVE_STATUSES | {"completed"}:
        raise P1ExecutorError(f"controller {task_id} status is not recoverable")
    tags = _task_tags(task)
    execution_tags = {
        item for item in tags if item.startswith(EXECUTION_KEY_TAG_PREFIX)
    }
    if execution_tags != {_execution_tag(execution_key)} or CONTROLLER_TAG not in tags:
        raise P1ExecutorError(f"controller {task_id} lease tags drifted")
    _require_parameters(
        task,
        _controller_parameters(plan=plan, pinset=pinset, execution_key=execution_key),
        f"controller {task_id}",
    )


def _validate_preflight_controller(
    task_class: object,
    controller: object,
    *,
    plan: Mapping[str, object],
    pinset: Mapping[str, object],
    execution_key: str,
) -> None:
    controller_id = _clearml_id(getattr(controller, "id", None), "controller")
    artifacts = _artifact_inventory(controller)
    direct_children = _query_direct_children(task_class, controller_id)
    named_children: list[object] = []
    for pair in plan["pairs"]:
        pair = _mapping(pair, "preflight controller pair")
        for phase in ("training", "evaluation"):
            record = _mapping(pair[phase], f"preflight controller {phase}")
            named_children.extend(
                _query_named_tasks(
                    task_class,
                    name=_task_name(execution_key, str(record["task_key"])),
                )
            )
    direct_ids = {
        _clearml_id(getattr(task, "id", None), "controller descendant")
        for task in direct_children
    }
    named_ids = {
        _clearml_id(getattr(task, "id", None), "stable-name child")
        for task in named_children
        if _task_parent(task) == controller_id
    }
    if direct_ids != named_ids:
        raise P1ExecutorError(
            "controller descendant inventory differs from stable child inventory"
        )
    durable = (
        JOURNAL_ARTIFACT in artifacts
        or MANIFEST_ARTIFACT in artifacts
        or bool(direct_children)
    )
    if _task_status(controller) != "created" or durable:
        _validate_controller(
            controller,
            plan=plan,
            pinset=pinset,
            execution_key=execution_key,
        )
        return
    if (
        _task_project(controller) != PROJECT_ID
        or str(getattr(controller, "name", "") or "") != _controller_name(execution_key)
        or _task_parent(controller) not in {"", SELECTOR_TASK_ID}
        or _task_queue(controller)
    ):
        raise P1ExecutorError("staged controller identity binding drifted")
    tags = _task_tags(controller)
    expected_execution_tag = _execution_tag(execution_key)
    execution_tags = {
        item for item in tags if item.startswith(EXECUTION_KEY_TAG_PREFIX)
    }
    if not execution_tags.issubset({expected_execution_tag}):
        raise P1ExecutorError("staged controller tags conflict with the execution")
    expected_parameters = _controller_parameters(
        plan=plan,
        pinset=pinset,
        execution_key=execution_key,
    )
    observed_parameters = _parameters(controller)
    if not set(observed_parameters).issubset(expected_parameters) or any(
        not _normalize_parameter(observed_parameters.get(key), expected_parameters[key])
        for key in observed_parameters
    ):
        raise P1ExecutorError(
            "staged controller parameters conflict with the execution"
        )
    initialization = {PLAN_ARTIFACT: plan, PINSET_ARTIFACT: pinset}
    if not set(artifacts).issubset(initialization):
        raise P1ExecutorError("staged controller artifact inventory drifted")
    for name, expected in initialization.items():
        artifact = artifacts.get(name)
        if artifact is not None and _artifact_mapping(artifact, name) != expected:
            raise P1ExecutorError(f"staged controller {name} drifted")


def _quarantine_new_controller_if_pristine(
    task_class: object,
    controller: object,
    *,
    plan: Mapping[str, object],
    execution_key: str,
    reason: str,
) -> None:
    controller_id = _clearml_id(getattr(controller, "id", None), "controller")
    controller = _fresh(task_class, controller_id)
    if (
        _task_status(controller) != "created"
        or _task_queue(controller)
        or _task_project(controller) != PROJECT_ID
        or str(getattr(controller, "name", "") or "") != _controller_name(execution_key)
    ):
        raise P1ExecutorError("new controller is not a pristine quarantine candidate")
    artifacts = _artifact_inventory(controller)
    initialization_artifacts = {PLAN_ARTIFACT, PINSET_ARTIFACT}
    if not set(artifacts).issubset(initialization_artifacts):
        raise P1ExecutorError(
            "controller has durable execution state and must be preserved"
        )
    if _query_direct_children(task_class, controller_id):
        raise P1ExecutorError("controller has descendants and must be preserved")
    for pair in plan["pairs"]:
        pair = _mapping(pair, "controller quarantine pair")
        for phase in ("training", "evaluation"):
            record = _mapping(pair[phase], f"controller quarantine {phase}")
            if _query_named_tasks(
                task_class,
                name=_task_name(execution_key, str(record["task_key"])),
            ):
                raise P1ExecutorError(
                    "controller has scoped children and must be preserved"
                )
    _quarantine_created(task_class, controller, reason=reason)


def _find_or_create_controller(
    task_class: object,
    *,
    plan: Mapping[str, object],
    pinset: Mapping[str, object],
    execution_key: str,
) -> tuple[object, bool]:
    name = _controller_name(execution_key)
    controller = _select_unique_created_candidate(
        task_class, name=name, context="P1 execution controller"
    )
    created = False
    if controller is None:
        creator = getattr(task_class, "create", None)
        if not callable(creator):
            raise P1ExecutorError("ClearML cannot create a recoverable controller")
        task_types = getattr(task_class, "TaskTypes", None)
        controller_type = getattr(task_types, "controller", "controller")
        controller = creator(
            project_name=PROJECT_NAME,
            task_name=name,
            task_type=controller_type,
            repo="",
            script="",
            detect_repository=False,
        )
        if controller is None:
            raise P1ExecutorError("ClearML controller creation returned no task")
        created = True
        controller_id = _clearml_id(getattr(controller, "id", None), "controller")
        setter = getattr(controller, "set_parent", None)
        if not callable(setter) or setter(parent=SELECTOR_TASK_ID) is False:
            _quarantine_new_controller_if_pristine(
                task_class,
                controller,
                plan=plan,
                execution_key=execution_key,
                reason="controller rejected selector parent",
            )
            raise P1ExecutorError("controller rejected the selector parent")
        _add_tags(controller, [_execution_tag(execution_key), CONTROLLER_TAG])
        _set_parameters(
            controller,
            _controller_parameters(
                plan=plan, pinset=pinset, execution_key=execution_key
            ),
        )
        controller.output_uri = FILES_SERVER_URI
        controller = _fresh(task_class, controller_id)
        _upload_artifact(controller, PLAN_ARTIFACT, plan)
        _upload_artifact(controller, PINSET_ARTIFACT, pinset)
        controller = _select_unique_created_candidate(
            task_class, name=name, context="P1 execution controller"
        )
        if controller is None:
            raise P1ExecutorError("controller disappeared during lease election")
    controller_id = _clearml_id(getattr(controller, "id", None), "controller")
    if _task_status(controller) == "created":
        try:
            parent = _task_parent(controller)
            if parent == "":
                setter = getattr(controller, "set_parent", None)
                if not callable(setter) or setter(parent=SELECTOR_TASK_ID) is False:
                    raise P1ExecutorError(
                        "controller rejected selector parent recovery"
                    )
                controller = _fresh(task_class, controller_id)
            elif parent != SELECTOR_TASK_ID:
                raise P1ExecutorError("controller parent is outside recoverable state")
            required_tags = {_execution_tag(execution_key), CONTROLLER_TAG}
            tags = _task_tags(controller)
            conflicting = {
                item
                for item in tags
                if item.startswith(EXECUTION_KEY_TAG_PREFIX)
                and item not in required_tags
            }
            if conflicting:
                raise P1ExecutorError("controller has a conflicting execution-key tag")
            if not required_tags.issubset(tags):
                _add_tags(controller, sorted(required_tags - tags))
                controller = _fresh(task_class, controller_id)
            expected_parameters = _controller_parameters(
                plan=plan, pinset=pinset, execution_key=execution_key
            )
            observed_parameters = _parameters(controller)
            if set(observed_parameters) != set(expected_parameters) or any(
                not _normalize_parameter(observed_parameters.get(key), item)
                for key, item in expected_parameters.items()
            ):
                recoverable = not observed_parameters or (
                    set(observed_parameters).issubset(expected_parameters)
                    and all(
                        _normalize_parameter(
                            observed_parameters.get(key), expected_parameters[key]
                        )
                        for key in observed_parameters
                    )
                )
                if not recoverable:
                    raise P1ExecutorError(
                        "controller parameters are outside recovery stages"
                    )
                _set_parameters(controller, expected_parameters)
                controller = _fresh(task_class, controller_id)
            controller.output_uri = FILES_SERVER_URI
        except Exception as error:
            if created:
                latest = _fresh(task_class, controller_id)
                try:
                    _quarantine_new_controller_if_pristine(
                        task_class,
                        latest,
                        plan=plan,
                        execution_key=execution_key,
                        reason=str(error),
                    )
                except Exception as quarantine_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "new controller could not be proven safe to quarantine: "
                            f"{quarantine_error}"
                        )
            raise
        artifacts = _artifact_inventory(controller)
        allowed = {
            PLAN_ARTIFACT,
            PINSET_ARTIFACT,
            JOURNAL_ARTIFACT,
            MANIFEST_ARTIFACT,
        }
        if not set(artifacts).issubset(allowed):
            raise P1ExecutorError(
                "controller artifact inventory contains unknown entries"
            )
        for name_key, expected_value in (
            (PLAN_ARTIFACT, plan),
            (PINSET_ARTIFACT, pinset),
        ):
            artifact = artifacts.get(name_key)
            if artifact is None:
                _upload_artifact(controller, name_key, expected_value)
            elif _artifact_mapping(artifact, name_key) != expected_value:
                raise P1ExecutorError(f"controller {name_key} drifted")
        controller = _fresh(task_class, controller_id)
    _validate_controller(
        controller, plan=plan, pinset=pinset, execution_key=execution_key
    )
    return controller, created


def _new_journal(
    *, plan: Mapping[str, object], execution_key: str, controller_id: str
) -> dict[str, object]:
    tasks: list[dict[str, object]] = []
    for pair in plan["pairs"]:
        pair = _mapping(pair, "journal pair")
        for phase in ("training", "evaluation"):
            record = _mapping(pair[phase], f"journal {phase}")
            tasks.append(
                {
                    "task_key": record["task_key"],
                    "kind": record["kind"],
                    "subject": record["subject"],
                    "seed_index": record["seed_index"],
                    "training_seed": record["training_seed"],
                    "task_id": None,
                    "state": "absent",
                    "server_status": "absent",
                    "result": None,
                }
            )
    journal: dict[str, object] = {
        "schema_version": 1,
        "document_type": JOURNAL_DOCUMENT_TYPE,
        "execution_key": execution_key,
        "plan_seal_sha256": plan["seal_sha256"],
        "controller_task_id": controller_id,
        "revision": 0,
        "status": "running",
        "tasks": tasks,
    }
    journal["seal_sha256"] = _seal(journal)
    return journal


def _validate_journal(
    value: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    execution_key: str,
    controller_id: str,
) -> dict[str, object]:
    journal = _mapping(value, "P1 execution journal")
    _exact_keys(
        journal,
        {
            "controller_task_id",
            "document_type",
            "execution_key",
            "plan_seal_sha256",
            "revision",
            "schema_version",
            "seal_sha256",
            "status",
            "tasks",
        },
        "P1 execution journal",
    )
    if journal.get("seal_sha256") != _seal(journal):
        raise P1ExecutorError("P1 execution journal seal mismatch")
    expected = {
        "schema_version": 1,
        "document_type": JOURNAL_DOCUMENT_TYPE,
        "execution_key": execution_key,
        "plan_seal_sha256": plan["seal_sha256"],
        "controller_task_id": controller_id,
    }
    for field, item in expected.items():
        _require_equal(journal.get(field), item, f"P1 journal {field}")
    if type(journal.get("revision")) is not int or journal["revision"] < 0:
        raise P1ExecutorError("P1 journal revision is invalid")
    if journal.get("status") not in {
        "running",
        "paused_timeout",
        "paused_error",
        "completed",
    }:
        raise P1ExecutorError("P1 journal status is invalid")
    observed = journal.get("tasks")
    if not isinstance(observed, list) or len(observed) != 12:
        raise P1ExecutorError("P1 journal must contain exactly twelve task slots")
    expected_rows = _new_journal(
        plan=plan, execution_key=execution_key, controller_id=controller_id
    )["tasks"]
    expected_by_key = {row["task_key"]: row for row in expected_rows}
    seen: set[str] = set()
    for row in observed:
        row = _mapping(row, "P1 journal task")
        _exact_keys(
            row,
            {
                "kind",
                "result",
                "seed_index",
                "server_status",
                "state",
                "subject",
                "task_id",
                "task_key",
                "training_seed",
            },
            "P1 journal task",
        )
        key = str(row.get("task_key") or "")
        if key in seen or key not in expected_by_key:
            raise P1ExecutorError("P1 journal task key inventory drifted")
        seen.add(key)
        template = expected_by_key[key]
        for field in ("kind", "seed_index", "subject", "training_seed"):
            _require_equal(row.get(field), template[field], f"P1 journal {key} {field}")
        if row.get("task_id") is not None:
            _clearml_id(row["task_id"], f"P1 journal {key} task")
        if row.get("state") not in {
            "absent",
            "discovered",
            "prepared",
            "enqueued",
            "active",
            "completed",
            "quarantined",
        }:
            raise P1ExecutorError(f"P1 journal {key} state is invalid")
    if seen != set(expected_by_key):
        raise P1ExecutorError("P1 journal task inventory is incomplete")
    return journal


def _load_journal(
    controller: object,
    *,
    plan: Mapping[str, object],
    execution_key: str,
    controller_id: str,
) -> dict[str, object]:
    artifact = _artifact_inventory(controller).get(JOURNAL_ARTIFACT)
    if artifact is None:
        return _new_journal(
            plan=plan, execution_key=execution_key, controller_id=controller_id
        )
    return _validate_journal(
        _artifact_mapping(artifact, "P1 execution journal"),
        plan=plan,
        execution_key=execution_key,
        controller_id=controller_id,
    )


def _persist_journal(controller: object, journal: dict[str, object]) -> None:
    if type(journal.get("revision")) is not int:
        raise P1ExecutorError("P1 journal revision is invalid before persistence")
    journal["revision"] += 1
    journal["seal_sha256"] = _seal(journal)
    _upload_artifact(controller, JOURNAL_ARTIFACT, journal)


def _journal_row(journal: Mapping[str, object], task_key: str) -> dict[str, object]:
    rows = journal.get("tasks")
    if not isinstance(rows, list):
        raise P1ExecutorError("P1 journal task list is invalid")
    matches = [
        row for row in rows if isinstance(row, dict) and row.get("task_key") == task_key
    ]
    if len(matches) != 1:
        raise P1ExecutorError(f"P1 journal slot {task_key} is missing or ambiguous")
    return matches[0]


def _record_journal_task(
    controller: object,
    journal: dict[str, object],
    *,
    task_key: str,
    task_id: str,
    state: str,
    server_status: str,
    result: Mapping[str, object] | None = None,
) -> None:
    candidate = copy.deepcopy(journal)
    row = _journal_row(candidate, task_key)
    existing_id = row.get("task_id")
    if existing_id not in (None, task_id) and row.get("state") != "quarantined":
        raise P1ExecutorError(f"P1 journal {task_key} task identity changed")
    prior_result = row.get("result")
    if existing_id not in (None, task_id) and row.get("state") == "quarantined":
        history = []
        if isinstance(prior_result, Mapping):
            history = list(prior_result.get("quarantined_task_ids", []))
        if existing_id not in history:
            history.append(existing_id)
        prior_result = {"quarantined_task_ids": history}
    row.update(
        {
            "task_id": _clearml_id(task_id, f"P1 journal {task_key}"),
            "state": state,
            "server_status": server_status,
            "result": dict(result) if result is not None else prior_result,
        }
    )
    candidate["status"] = "running"
    _persist_journal(controller, candidate)
    journal.clear()
    journal.update(candidate)


def _reconcile_quarantined_journal_tasks(
    task_class: object,
    controller: object,
    journal: dict[str, object],
    *,
    controller_id: str,
    execution_key: str,
) -> None:
    rows = journal.get("tasks")
    if not isinstance(rows, list):
        raise P1ExecutorError("P1 journal task list is invalid during reconciliation")
    for value in list(rows):
        row = _mapping(value, "P1 reconciliation journal row")
        if row.get("state") not in {"discovered", "prepared"}:
            continue
        task_key = str(row.get("task_key") or "")
        task_id = _clearml_id(row.get("task_id"), f"P1 reconciliation {task_key} task")
        task = _fresh(task_class, task_id)
        tags = _task_tags(task)
        if ORPHAN_TAG not in tags:
            continue
        expected_execution_tag = _execution_tag(execution_key)
        expected_task_key_tag = f"p1-task-key:{task_key}"
        execution_tags = {
            item for item in tags if item.startswith(EXECUTION_KEY_TAG_PREFIX)
        }
        task_key_tags = {item for item in tags if item.startswith("p1-task-key:")}
        if (
            _clearml_id(getattr(task, "id", None), "reconciled orphan") != task_id
            or str(getattr(task, "name", "") or "")
            != _task_name(execution_key, task_key)
            or _task_parent(task) != controller_id
            or _task_project(task) != PROJECT_ID
            or not execution_tags.issubset({expected_execution_tag})
            or not task_key_tags.issubset({expected_task_key_tag})
            or _task_queue(task)
            or _task_status(task) not in {"created", "failed"}
        ):
            raise P1ExecutorError(
                f"P1 journal {task_key} quarantined task binding drifted"
            )
        _record_journal_task(
            controller,
            journal,
            task_key=task_key,
            task_id=task_id,
            state="quarantined",
            server_status=_task_status(task),
            result={"quarantined_task_ids": [task_id]},
        )


@contextmanager
def _execution_lock() -> Iterator[None]:
    cache = _validate_cache_directory()
    # Deliberately reuse the legacy 5090 lock namespace.  Retirement and A100
    # execution therefore cannot overlap on the controller host.
    lock_path = cache / ".p1-scoped-multiseed-executor.lock"
    flags = os.O_CREAT | os.O_RDWR | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise P1ExecutorError("P1 executor lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise P1ExecutorError(
                "another local P1 executor owns the execution lock"
            ) from error
        yield
    finally:
        os.close(descriptor)


def _mutex_receipt_path() -> Path:
    return _validate_cache_directory() / ".p1-a100-scoped-multiseed-mutex.json"


def _validate_mutex_receipt(
    value: Mapping[str, object], *, execution_key: str
) -> dict[str, object]:
    receipt = _mapping(value, "local mutex journal")
    _exact_keys(
        receipt,
        {
            "document_type",
            "execution_key",
            "mutex_name",
            "owner_tag",
            "queue_id",
            "schema_version",
            "seal_sha256",
        },
        "local mutex journal",
    )
    if receipt.get("seal_sha256") != _seal(receipt):
        raise P1ExecutorError("local mutex journal seal mismatch")
    expected = {
        "schema_version": 1,
        "document_type": LOCAL_MUTEX_DOCUMENT_TYPE,
        "execution_key": execution_key,
        "mutex_name": _lease_queue_name(execution_key),
    }
    for field, item in expected.items():
        _require_equal(receipt.get(field), item, f"local mutex journal {field}")
    _clearml_id(receipt.get("queue_id"), "local mutex journal queue")
    if re.fullmatch(r"owner:[0-9a-f]{32}", str(receipt.get("owner_tag") or "")) is None:
        raise P1ExecutorError("local mutex journal owner tag is invalid")
    return receipt


def _write_mutex_receipt(receipt: Mapping[str, object]) -> None:
    path = _mutex_receipt_path()
    if path.exists() or path.is_symlink():
        raise P1ExecutorError("local mutex journal already exists")
    payload = (_canonical_json(receipt) + "\n").encode("utf-8")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(temporary, flags, 0o600)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise P1ExecutorError("local mutex journal write was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _remove_mutex_receipt() -> None:
    path = _mutex_receipt_path()
    if path.is_symlink():
        raise P1ExecutorError("local mutex journal became a symlink")
    path.unlink()
    directory = os.open(path.parent, os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _verify_mutex_holder(
    service: object,
    *,
    queue_id: str,
    name: str,
    owner_tag: str,
    execution_key: str,
) -> object:
    getter = getattr(service, "get_by_id", None)
    counter = getattr(service, "get_num_entries", None)
    if not callable(getter) or not callable(counter):
        raise P1ExecutorError("ClearML cannot verify the execution mutex")
    current = getter(queue=queue_id, max_task_entries=1)
    tags = {str(item) for item in (getattr(current, "tags", None) or [])}
    count = getattr(counter(queue=queue_id), "num", None)
    if (
        str(getattr(current, "id", "") or "") != queue_id
        or str(getattr(current, "name", "") or "") != name
        or owner_tag not in tags
        or _execution_tag(execution_key) not in tags
        or (getattr(current, "entries", None) or [])
        or count != 0
    ):
        raise P1ExecutorError("execution mutex ownership or emptiness drifted")
    return current


def _recover_owned_mutex_from_local_journal(
    execution_key: str, *, api_client: object
) -> dict[str, str] | None:
    path = _mutex_receipt_path()
    if not path.exists() and not path.is_symlink():
        return None
    receipt = _validate_mutex_receipt(
        _read_json_path(path, "local mutex journal"), execution_key=execution_key
    )
    service = getattr(api_client, "queues", None)
    queue_id = str(receipt["queue_id"])
    name = str(receipt["mutex_name"])
    owner_tag = str(receipt["owner_tag"])
    _verify_mutex_holder(
        service,
        queue_id=queue_id,
        name=name,
        owner_tag=owner_tag,
        execution_key=execution_key,
    )
    return {"queue_id": queue_id, "name": name, "owner_tag": owner_tag}


@contextmanager
def _server_execution_mutex(
    execution_key: str,
    *,
    api_client: object | None = None,
    owner_token: str | None = None,
) -> Iterator[dict[str, str]]:
    """Acquire the ClearML company-unique queue mutex for one process lifetime."""

    client = api_client or _api_client()
    service = getattr(client, "queues", None)
    recovered = _recover_owned_mutex_from_local_journal(
        execution_key, api_client=client
    )
    if recovered is not None:
        lock_id = recovered["queue_id"]
        name = recovered["name"]
        owner_tag = recovered["owner_tag"]
        if owner_token is not None and owner_tag != f"owner:{owner_token}":
            raise P1ExecutorError(
                "execution mutex owner token conflicts with the sealed local journal"
            )
    else:
        creator = getattr(service, "create", None)
        if not callable(creator):
            raise P1ExecutorError("ClearML cannot create the execution mutex")
        name = _lease_queue_name(execution_key)
        token = owner_token or secrets.token_hex(16)
        if re.fullmatch(r"[0-9a-f]{32}", token) is None:
            raise P1ExecutorError("execution mutex owner token is invalid")
        owner_tag = f"owner:{token}"
        tags = [MUTEX_TAG, _execution_tag(execution_key), owner_tag]
        try:
            holder = creator(name=name, tags=tags)
        except Exception as error:
            rows = _lease_rows(client, execution_key)
            if len(rows) == 1:
                raise P1ExecutorError(
                    "global P1 execution mutex is already held; inspect with --preflight"
                ) from error
            raise P1ExecutorError(
                "ClearML mutex creation failed without one auditable holder"
            ) from error
        lock_id = _clearml_id(getattr(holder, "id", None), "mutex queue")
        _verify_mutex_holder(
            service,
            queue_id=lock_id,
            name=name,
            owner_tag=owner_tag,
            execution_key=execution_key,
        )
        local_receipt: dict[str, object] = {
            "schema_version": 1,
            "document_type": LOCAL_MUTEX_DOCUMENT_TYPE,
            "execution_key": execution_key,
            "mutex_name": name,
            "queue_id": lock_id,
            "owner_tag": owner_tag,
        }
        local_receipt["seal_sha256"] = _seal(local_receipt)
        _write_mutex_receipt(local_receipt)
    try:
        yield {"queue_id": lock_id, "name": name, "owner_tag": owner_tag}
    finally:
        deleter = getattr(service, "delete", None)
        if not callable(deleter):
            raise P1ExecutorError("ClearML cannot safely release the execution mutex")
        _verify_mutex_holder(
            service,
            queue_id=lock_id,
            name=name,
            owner_tag=owner_tag,
            execution_key=execution_key,
        )
        response = deleter(queue=lock_id, force=False)
        if getattr(response, "deleted", None) != 1:
            raise P1ExecutorError("ClearML did not confirm execution mutex deletion")
        _remove_mutex_receipt()


def _require_server_unique_child(
    task_class: object,
    *,
    task: object,
    controller_id: str,
    execution_key: str,
    task_key: str,
) -> None:
    task_id = _clearml_id(getattr(task, "id", None), "scoped child")
    matches = _active_named_tasks(task_class, name=_task_name(execution_key, task_key))
    if (
        len(matches) != 1
        or _clearml_id(getattr(matches[0], "id", None), "discovered scoped child")
        != task_id
        or _task_parent(matches[0]) != controller_id
    ):
        raise P1ExecutorError(f"scoped child {task_key} is not server-side unique")


def _validate_execution_manifest(
    value: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    execution_key: str,
    controller_id: str,
) -> dict[str, object]:
    manifest = _mapping(value, "P1 execution manifest")
    _exact_keys(
        manifest,
        {
            "controller_task_id",
            "document_type",
            "execution_key",
            "plan_seal_sha256",
            "results",
            "runtime_contract",
            "schema_version",
            "seal_sha256",
            "status",
        },
        "P1 execution manifest",
    )
    if manifest.get("seal_sha256") != _seal(manifest):
        raise P1ExecutorError("P1 execution manifest seal mismatch")
    expected = {
        "schema_version": 1,
        "document_type": MANIFEST_DOCUMENT_TYPE,
        "execution_key": execution_key,
        "plan_seal_sha256": plan["seal_sha256"],
        "controller_task_id": controller_id,
        "status": "completed",
        "runtime_contract": _a100_runtime_contract(),
    }
    for field, item in expected.items():
        _require_equal(manifest.get(field), item, f"P1 manifest {field}")
    results = manifest.get("results")
    if not isinstance(results, list) or len(results) != 6:
        raise P1ExecutorError("P1 execution manifest must contain six results")
    expected_pairs = [
        _mapping(pair["training"], "P1 manifest expected pair")
        for pair in plan["pairs"]
    ]
    for result, record in zip(results, expected_pairs, strict=True):
        result = _mapping(result, "P1 manifest result")
        _exact_keys(
            result,
            {
                "checkpoint_sha256",
                "evaluation",
                "model_id",
                "seed_index",
                "subject",
                "training_runtime",
                "training_seed",
                "training_task_id",
            },
            "P1 manifest result",
        )
        for field in ("subject", "seed_index", "training_seed"):
            _require_equal(result.get(field), record[field], f"P1 manifest {field}")
        training_task_id = _clearml_id(
            result.get("training_task_id"), "P1 manifest training task"
        )
        _clearml_id(result.get("model_id"), "P1 manifest model")
        _sha256(result.get("checkpoint_sha256"), "P1 manifest checkpoint")
        _validate_recorded_runtime_receipt(
            _mapping(result.get("training_runtime"), "P1 manifest training runtime"),
            task_id=training_task_id,
        )
        evaluation = _mapping(result.get("evaluation"), "P1 manifest evaluation")
        _exact_keys(
            evaluation,
            {
                "metrics_artifact_sha256",
                "prediction_evidence_artifact_sha256",
                "runtime",
                "task_id",
                "worker",
            },
            "P1 manifest evaluation",
        )
        evaluation_task_id = _clearml_id(
            evaluation.get("task_id"), "P1 manifest evaluation task"
        )
        _sha256(
            evaluation.get("metrics_artifact_sha256"),
            "P1 manifest evaluation metrics",
        )
        _sha256(
            evaluation.get("prediction_evidence_artifact_sha256"),
            "P1 manifest evaluation evidence",
        )
        if evaluation.get("worker") not in ALLOWED_WORKER_IDS:
            raise P1ExecutorError("P1 manifest evaluation worker drifted")
        _validate_recorded_runtime_receipt(
            _mapping(evaluation.get("runtime"), "P1 manifest evaluation runtime"),
            task_id=evaluation_task_id,
        )
    return manifest


def execute_plan(
    *,
    authorization_token: str,
    poll_seconds: float,
    timeout_hours: float,
    task_class: object | None = None,
    queue_reader: Callable[[object], Mapping[str, object]] | None = None,
    api_client: object | None = None,
    legacy_retirement_reader: Callable[
        [object, Mapping[str, object], Mapping[str, object], str],
        Mapping[str, object],
    ]
    | None = None,
) -> dict[str, object]:
    """Execute or resume the fixed matrix under local and server mutexes."""

    if authorization_token != EXECUTION_TOKEN:
        raise P1ExecutorError("remote execution token mismatch")
    if not 1.0 <= poll_seconds <= 60.0 or not math.isfinite(poll_seconds):
        raise P1ExecutorError("poll-seconds must be finite and within [1, 60]")
    if not 1.0 <= timeout_hours <= 72.0 or not math.isfinite(timeout_hours):
        raise P1ExecutorError("timeout-hours must be finite and within [1, 72]")
    plan = build_plan()
    pinset = build_pinset()
    execution_key = _execution_key(plan, pinset)
    Task = task_class or _load_clearml()
    client = api_client or _api_client()

    def lease_reader(key: str) -> Mapping[str, object]:
        return _read_lease_snapshot(key, api_client=client)

    def retirement_reader(
        task_type: object,
        current_plan: Mapping[str, object],
        current_pinset: Mapping[str, object],
        current_key: str,
    ) -> Mapping[str, object]:
        if legacy_retirement_reader is not None:
            return legacy_retirement_reader(
                task_type, current_plan, current_pinset, current_key
            )
        return _read_legacy_retirement_snapshot(
            task_type,
            plan=current_plan,
            pinset=current_pinset,
            execution_key=current_key,
            api_client=client,
        )

    def assert_gpu8_idle(context: str) -> None:
        _validate_a100_queue_receipt(
            (queue_reader or _read_queue_snapshot)(Task),
            context=context,
        )

    with _execution_lock():
        preflight(
            task_class=Task,
            queue_reader=queue_reader,
            lease_reader=lease_reader,
            legacy_retirement_reader=retirement_reader,
        )
        with _server_execution_mutex(execution_key, api_client=client):
            # Revalidate inside the company-global mutex before the first task mutation.
            preflight(
                task_class=Task,
                queue_reader=queue_reader,
                lease_reader=lease_reader,
                legacy_retirement_reader=retirement_reader,
            )
            source_d_task = Task.get_task(task_id=SOURCE_D_TASK_ID)
            teacher = Task.get_task(task_id=TEACHER_TASK_ID)
            source_d = _validate_source_d(source_d_task)
            _validate_teacher(teacher)
            controller, _ = _find_or_create_controller(
                Task,
                plan=plan,
                pinset=pinset,
                execution_key=execution_key,
            )
            controller_id = _clearml_id(getattr(controller, "id", None), "controller")
            if _task_status(controller) == "completed":
                artifact = _artifact_inventory(controller).get(MANIFEST_ARTIFACT)
                if artifact is None:
                    raise P1ExecutorError(
                        "completed controller has no execution manifest"
                    )
                return _validate_execution_manifest(
                    _artifact_mapping(artifact, "P1 execution manifest"),
                    plan=plan,
                    execution_key=execution_key,
                    controller_id=controller_id,
                )

            journal = _load_journal(
                controller,
                plan=plan,
                execution_key=execution_key,
                controller_id=controller_id,
            )
            _reconcile_quarantined_journal_tasks(
                Task,
                controller,
                journal,
                controller_id=controller_id,
                execution_key=execution_key,
            )
            journal["status"] = "running"
            _persist_journal(controller, journal)
            manifest: dict[str, object] = {
                "schema_version": 1,
                "document_type": MANIFEST_DOCUMENT_TYPE,
                "execution_key": execution_key,
                "plan_seal_sha256": plan["seal_sha256"],
                "controller_task_id": controller_id,
                "status": "completed",
                "runtime_contract": _a100_runtime_contract(),
                "results": [],
            }
            try:
                for pair in plan["pairs"]:
                    pair = _mapping(pair, "P1 pair")
                    training_record = _mapping(pair["training"], "P1 training record")
                    evaluation_record = _mapping(
                        pair["evaluation"], "P1 evaluation record"
                    )
                    subject = str(training_record["subject"])
                    seed = int(training_record["training_seed"])
                    training_key = str(training_record["task_key"])

                    def record_training(task_id: str, origin: str) -> None:
                        state = (
                            "quarantined" if origin == "quarantined" else "discovered"
                        )
                        result = (
                            {"quarantined_task_ids": [task_id]}
                            if state == "quarantined"
                            else None
                        )
                        _record_journal_task(
                            controller,
                            journal,
                            task_key=training_key,
                            task_id=task_id,
                            state=state,
                            server_status=(
                                "failed" if state == "quarantined" else "created"
                            ),
                            result=result,
                        )

                    training = _create_task(
                        Task,
                        teacher_task=teacher,
                        controller_id=controller_id,
                        task_key=training_key,
                        source_d=source_d,
                        parameters=_training_parameters(subject, seed),
                        execution_key=execution_key,
                        on_discovered=record_training,
                    )
                    training_id = _clearml_id(
                        getattr(training, "id", None), "training task"
                    )
                    _require_server_unique_child(
                        Task,
                        task=training,
                        controller_id=controller_id,
                        execution_key=execution_key,
                        task_key=training_key,
                    )
                    _record_journal_task(
                        controller,
                        journal,
                        task_key=training_key,
                        task_id=training_id,
                        state="prepared",
                        server_status=_task_status(training),
                    )
                    assert_gpu8_idle(f"{training_key} pre-enqueue queue receipt")
                    _enqueue_once(
                        Task, training, authorization_token=authorization_token
                    )
                    training = _fresh(Task, training_id)
                    _record_journal_task(
                        controller,
                        journal,
                        task_key=training_key,
                        task_id=training_id,
                        state=(
                            "completed"
                            if _task_status(training) == "completed"
                            else "active"
                        ),
                        server_status=_task_status(training),
                    )
                    training = _wait_for_task(
                        Task,
                        training_id,
                        timeout_hours=timeout_hours,
                        poll_seconds=poll_seconds,
                        on_poll=lambda _task: assert_gpu8_idle(
                            f"{training_key} active queue receipt"
                        ),
                    )
                    assert_gpu8_idle(f"{training_key} completion queue receipt")
                    model = _validate_training_result(
                        training,
                        subject=subject,
                        seed=seed,
                        controller_id=controller_id,
                        source_d=source_d,
                    )
                    _record_journal_task(
                        controller,
                        journal,
                        task_key=training_key,
                        task_id=training_id,
                        state="completed",
                        server_status="completed",
                        result=model,
                    )

                    evaluation_key = str(evaluation_record["task_key"])

                    def record_evaluation(task_id: str, origin: str) -> None:
                        state = (
                            "quarantined" if origin == "quarantined" else "discovered"
                        )
                        result = (
                            {"quarantined_task_ids": [task_id]}
                            if state == "quarantined"
                            else None
                        )
                        _record_journal_task(
                            controller,
                            journal,
                            task_key=evaluation_key,
                            task_id=task_id,
                            state=state,
                            server_status=(
                                "failed" if state == "quarantined" else "created"
                            ),
                            result=result,
                        )

                    evaluation_parameters = _evaluation_parameters(
                        subject,
                        seed,
                        training_task_id=training_id,
                        model_id=str(model["model_id"]),
                        checkpoint_sha256=str(model["checkpoint_sha256"]),
                    )
                    evaluation = _create_task(
                        Task,
                        teacher_task=teacher,
                        controller_id=controller_id,
                        task_key=evaluation_key,
                        source_d=source_d,
                        parameters=evaluation_parameters,
                        execution_key=execution_key,
                        on_discovered=record_evaluation,
                    )
                    evaluation_id = _clearml_id(
                        getattr(evaluation, "id", None), "evaluation task"
                    )
                    _require_server_unique_child(
                        Task,
                        task=evaluation,
                        controller_id=controller_id,
                        execution_key=execution_key,
                        task_key=evaluation_key,
                    )
                    _record_journal_task(
                        controller,
                        journal,
                        task_key=evaluation_key,
                        task_id=evaluation_id,
                        state="prepared",
                        server_status=_task_status(evaluation),
                    )
                    assert_gpu8_idle(f"{evaluation_key} pre-enqueue queue receipt")
                    _enqueue_once(
                        Task, evaluation, authorization_token=authorization_token
                    )
                    evaluation = _fresh(Task, evaluation_id)
                    _record_journal_task(
                        controller,
                        journal,
                        task_key=evaluation_key,
                        task_id=evaluation_id,
                        state=(
                            "completed"
                            if _task_status(evaluation) == "completed"
                            else "active"
                        ),
                        server_status=_task_status(evaluation),
                    )
                    evaluation = _wait_for_task(
                        Task,
                        evaluation_id,
                        timeout_hours=timeout_hours,
                        poll_seconds=poll_seconds,
                        on_poll=lambda _task: assert_gpu8_idle(
                            f"{evaluation_key} active queue receipt"
                        ),
                    )
                    assert_gpu8_idle(f"{evaluation_key} completion queue receipt")
                    evaluation_receipt = _validate_evaluation_result(
                        evaluation,
                        subject=subject,
                        seed=seed,
                        controller_id=controller_id,
                        training_task_id=training_id,
                        model=model,
                        source_d=source_d,
                    )
                    _record_journal_task(
                        controller,
                        journal,
                        task_key=evaluation_key,
                        task_id=evaluation_id,
                        state="completed",
                        server_status="completed",
                        result=evaluation_receipt,
                    )
                    manifest["results"].append(
                        {
                            "subject": subject,
                            "seed_index": training_record["seed_index"],
                            "training_seed": seed,
                            "training_task_id": training_id,
                            "model_id": model["model_id"],
                            "checkpoint_sha256": model["checkpoint_sha256"],
                            "training_runtime": model["runtime"],
                            "evaluation": evaluation_receipt,
                        }
                    )
                if len(manifest["results"]) != 6:
                    raise P1ExecutorError("executor did not complete exactly six pairs")
                manifest["seal_sha256"] = _seal(manifest)
                _validate_execution_manifest(
                    manifest,
                    plan=plan,
                    execution_key=execution_key,
                    controller_id=controller_id,
                )
                _upload_artifact(controller, MANIFEST_ARTIFACT, manifest)
                journal["status"] = "completed"
                _persist_journal(controller, journal)
                controller = _complete_controller(Task, controller)
                return manifest
            except P1RecoverableTimeout:
                journal["status"] = "paused_timeout"
                _persist_journal(controller, journal)
                raise
            except Exception:
                journal["status"] = "paused_error"
                _persist_journal(controller, journal)
                raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="create and enqueue the fixed remote task matrix",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="connect read-only and print a sealed machine-readable receipt",
    )
    parser.add_argument(
        "--enqueue-token",
        default="",
        help="exact acknowledgement token required with --execute",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight:
        if args.enqueue_token:
            raise P1ExecutorError("--enqueue-token is invalid with --preflight")
        receipt = preflight()
        print(
            json.dumps(
                receipt,
                ensure_ascii=True,
                sort_keys=True,
                indent=2 if args.pretty else None,
                separators=None if args.pretty else (",", ":"),
            )
        )
        return 0
    if not args.execute:
        if args.enqueue_token:
            raise P1ExecutorError("--enqueue-token is invalid without --execute")
        plan = build_plan()
        print(
            json.dumps(
                plan,
                ensure_ascii=True,
                sort_keys=True,
                indent=2 if args.pretty else None,
                separators=None if args.pretty else (",", ":"),
            )
        )
        return 0
    if args.enqueue_token != EXECUTION_TOKEN:
        raise P1ExecutorError(
            "--execute requires the exact --enqueue-token acknowledgement"
        )
    manifest = execute_plan(
        authorization_token=args.enqueue_token,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    print(_canonical_json(manifest))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except P1RecoverableTimeout as error:
        print(f"P1-EXECUTOR-RECOVERABLE-TIMEOUT: {error}", file=sys.stderr)
        raise SystemExit(3) from None
    except P1ExecutorError as error:
        print(f"P1-EXECUTOR-ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
