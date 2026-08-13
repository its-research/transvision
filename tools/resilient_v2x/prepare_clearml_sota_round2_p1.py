#!/usr/bin/env python3
"""Prepare the sealed round2 P1 reliability-gated residual source.

P1 is derived strictly from the immutable sealed P0 archive.  Its source delta
replaces exactly three implementation files and adds exactly one configuration;
the independently prepared P2 bbox2.5 configuration is explicitly forbidden.

The default execution contract launches an independently identified P1 50-epoch
formal candidate.  A second, separately identified diagnostic candidate applies
the P1 inference transform to P0's canonical epoch-50 final OutputModel without
training.  Clean-path equality supports only that diagnostic compatibility; it
does not waive formal P1 training.  This helper performs no remote action unless
an exact remote-write token is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared


ROOT = Path(__file__).resolve().parents[2]
P0_OUTPUT_DIR = ROOT / "artifacts/resilient_v2x/sota_round2_p0"
P0_SOURCE_DATASET_ID = "ca2ef9dd8a984df6b05693fb02e89f34"
P0_TEMPLATE_TASK_ID = "13f87c0fd45d4621b3c93c0ff88702d8"
P0_TRAINING_TASK_ID = "8883c51ced4f4951a45edbaefe6342d4"
P0_PACKAGE_MANIFEST_SEAL_SHA256 = (
    "e5b70d39704515d03506470468fd4f33a96761a03e7b3213f1f687cf734cfbe8"
)
P0_STATIC_PLAN_SEAL_SHA256 = (
    "ebd8421be6fcfde395a93e6ffa1af6f568462b1da7b8e8e564ea40b837a2da15"
)
P0_DEPLOYMENT_PLAN_SEAL_SHA256 = (
    "fa0e7a05ade60d0c682944a502ea09eecc382024427e555ad1927fddc83b0b94"
)

OUTPUT_DIR = ROOT / "artifacts/resilient_v2x/sota_round2_p1"
PACKAGE_MANIFEST_NAME = "round2-p1-source-package.json"
DEPLOYMENT_PLAN_NAME = "round2-p1-deployment-plan.json"

P1_EXPERIMENT = "reliability_gated_residual"
P1_CONFIG = "configs/resilient_v2x/improvements/reliability_gated_residual.py"
P1_IDENTITY = "dair_improvement_reliability_gated_residual"
P1_TRAINED_CANDIDATE = "p1_trained_50e_formal"
P1_TRAINED_CANDIDATE_IDENTITY = f"{P1_IDENTITY}_trained_50e"
P1_ZERO_SHOT_CANDIDATE = "p1_zero_shot_p0_epoch50_final"
P1_ZERO_SHOT_CANDIDATE_IDENTITY = f"{P1_IDENTITY}_zero_shot_p0_epoch50_final"
P2_EXPERIMENT = "support_residual_no_reliability_linear_bbox25"
P2_CONFIG = (
    "configs/resilient_v2x/improvements/"
    "support_residual_no_reliability_linear_bbox25.py"
)
P1_MODIFIED_SOURCE_PATHS = (
    "transvision/models/detectors/resilient_v2x.py",
    "transvision/models/resilient_v2x/fusion.py",
    "transvision/models/resilient_v2x/routing.py",
)
P1_TARGET_FILE_SHA256 = {
    "transvision/models/detectors/resilient_v2x.py": (
        "f9a857129a5b5c8fa31e15d70b5d2295012adf484d66488cab5ba9ab512f5f6f"
    ),
    "transvision/models/resilient_v2x/fusion.py": (
        "dc368dfccc9bf14ace570ae114bca3e09af31c3e2932ab059ee5ecc44c61e8cb"
    ),
    "transvision/models/resilient_v2x/routing.py": (
        "66f2b196919c6a29ea6e78598016d68395af6f69ac0e5cc7842e0459ac7910a6"
    ),
    P1_CONFIG: ("76f7164326c7de8f6c5e9148887fb02772c141e981ea42e579ff9bea010c0ef1"),
}

SOURCE_TRANSITION_ARTIFACT = "sota_round2_p1_source_transition"
LAUNCH_RECEIPT_ARTIFACT = "sota_round2_p1_launch_receipt"
ENQUEUE_ACKNOWLEDGEMENT_ARTIFACT = "sota_round2_p1_enqueue_acknowledgement"
SOURCE_DATASET_PREFIX = "ResilientV2X sealed SOTA round2 P1 reliability-gated source"
TEMPLATE_PREFIX = "ResilientV2X round2 P1 reliability-gated bootstrap template"
TASK_PREFIX = "ResilientV2X round2 P1 reliability-gated"

WORKER_QUEUE = shared.WORKER_QUEUE
WORKER_QUEUE_ID = shared.WORKER_QUEUE_ID
GPU_COUNT = shared.GPU_COUNT
BATCH_SIZE_PER_GPU = shared.BATCH_SIZE_PER_GPU
GLOBAL_BATCH_SIZE = shared.GLOBAL_BATCH_SIZE
MAX_EPOCHS = shared.MAX_EPOCHS
VAL_INTERVAL = shared.VAL_INTERVAL
TRAINING_SEED = shared.TRAINING_SEED
PRECISION = shared.PRECISION
OVERLAY_PROTOCOL_SEED = TRAINING_SEED
FORMAL_PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
FORMAL_SAMPLE_COUNT = 1337
FORMAL_DELAYS_MS = (0, 100, 200, 300)
FORMAL_FAULT_CONDITIONS = ("Full", "L-Fail", "C-Fail")
FORMAL_CONDITION_COUNT = len(FORMAL_DELAYS_MS) * len(FORMAL_FAULT_CONDITIONS)
FORMAL_CHECKPOINT_POLICY = "epoch_50_final_only"
FORMAL_CHECKPOINT_FILENAME = "epoch_50.pth"
P1_TASK_PARAMETER_BINDINGS = {
    "Args/candidate_variant": P1_TRAINED_CANDIDATE,
    "Args/candidate_identity": P1_TRAINED_CANDIDATE_IDENTITY,
    "Args/evidence_class": "formal_p1_trained_candidate",
}
P1_CANDIDATE_PROVENANCE = {
    key.removeprefix("Args/"): value
    for key, value in P1_TASK_PARAMETER_BINDINGS.items()
}
P1_RUNTIME_PROFILE = "a100_sm80"
P1_RUNTIME_HARDWARE = "A100"
P1_RUNTIME_COMPUTE_CAPABILITY = "sm80"
P1_RUNTIME_IMPLEMENTATION_COMPATIBILITY_LAYER = {
    "name": "sealed_native_bundle_python_source_compatibility",
    "runner_implementation_profile": "rtx5090",
    "semantics": (
        "implementation compatibility only; result hardware provenance is A100/sm80"
    ),
}

UPLOAD_TOKEN = "UPLOAD_EXACT_ROUND2_P1_SOURCE"
TEMPLATE_TOKEN = "CREATE_EXACT_ROUND2_P1_TEMPLATE"
LAUNCH_TOKEN = "LAUNCH_EXACT_ROUND2_P1_A100"

TARGET_A100_SMOKE_EVENT = "resilient_v2x_p1_target_a100_cuda_smoke_pass"
TARGET_A100_CUDA_SMOKE = r"""
import copy
import hashlib
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
import mmdet3d.models  # noqa: F401
from mmdet3d.registry import MODELS

from transvision import register_all_modules
from transvision.models.resilient_v2x import DynamicExpertRouter

P1_CONFIG_PATH = (
    "configs/resilient_v2x/improvements/"
    "reliability_gated_residual.py"
)
EXPECTED_FILES = {
    P1_CONFIG_PATH: "76f7164326c7de8f6c5e9148887fb02772c141e981ea42e579ff9bea010c0ef1",
    "transvision/models/detectors/resilient_v2x.py": "f9a857129a5b5c8fa31e15d70b5d2295012adf484d66488cab5ba9ab512f5f6f",
    "transvision/models/resilient_v2x/fusion.py": "dc368dfccc9bf14ace570ae114bca3e09af31c3e2932ab059ee5ecc44c61e8cb",
    "transvision/models/resilient_v2x/routing.py": "66f2b196919c6a29ea6e78598016d68395af6f69ac0e5cc7842e0459ac7910a6",
}
for relative_path, expected_sha256 in sorted(EXPECTED_FILES.items()):
    path = Path(relative_path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"P1 smoke source member is invalid: {relative_path}")
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"P1 smoke source SHA-256 drifted for {relative_path}: "
            f"{observed_sha256}"
        )

if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise RuntimeError(
        f"P1 target smoke requires exactly 4 CUDA GPUs; "
        f"got {torch.cuda.device_count()}"
    )
capabilities = [
    tuple(torch.cuda.get_device_capability(index))
    for index in range(torch.cuda.device_count())
]
if capabilities != [(8, 0)] * 4:
    raise RuntimeError(f"P1 target smoke requires four A100-class sm80 GPUs: {capabilities}")

register_all_modules()
config = Config.fromfile(P1_CONFIG_PATH)
init_default_scope(config.get("default_scope", "mmdet3d"))
if config.experiment.name != "dair_improvement_reliability_gated_residual":
    raise RuntimeError("P1 resolved experiment identity drifted")
for key, expected in {
    "ptf_mode": "linear",
    "use_reliability": False,
    "support_residual_weight": 0.5,
    "support_residual_reliability_gate": True,
}.items():
    if config.model.get(key) != expected:
        raise RuntimeError(
            f"P1 resolved model field {key} drifted: {config.model.get(key)!r}"
        )

student_config = copy.deepcopy(config.model)
student_config["teacher"] = None
student_config["teacher_checkpoint"] = None
student_config["distillation"] = None
student_config["camera_encoder"]["image_backbone"]["init_cfg"] = None
full_model = MODELS.build(student_config).cuda().eval()
fusion = full_model.resilient_fusion
if fusion.support_residual_reliability_gate is not True:
    raise RuntimeError("P1 detector-to-fusion gate propagation failed")
if fusion.router.support_residual_reliability_gate is not True:
    raise RuntimeError("P1 fusion-to-router gate propagation failed")
full_model_parameters = sum(
    parameter.numel() for parameter in full_model.parameters()
)
if full_model_parameters <= 0:
    raise RuntimeError("P1 full detector build has no parameters")
del full_model
torch.cuda.empty_cache()

torch.manual_seed(20250218)
torch.cuda.manual_seed_all(20250218)
reference = DynamicExpertRouter(
    channels=256,
    hidden_channels=256,
    support_residual_weight=0.0,
).cuda().eval()
candidate = DynamicExpertRouter(
    channels=256,
    hidden_channels=256,
    support_residual_weight=0.5,
    support_residual_reliability_gate=True,
).cuda().eval()
candidate.load_state_dict(reference.state_dict(), strict=True)

lidar = torch.randn(1, 256, 4, 4, device="cuda", requires_grad=True)
camera = torch.randn(1, 256, 4, 4, device="cuda", requires_grad=True)
support = torch.ones(1, 4, dtype=torch.bool, device="cuda")
observed = torch.tensor(
    [[True, False, True, False]], dtype=torch.bool, device="cuda"
)
propagated = torch.tensor(
    [[False, True, False, True]], dtype=torch.bool, device="cuda"
)
common = {
    "lidar_feature": lidar,
    "camera_feature": camera,
    "lidar_branch_support": support[:, :2],
    "camera_branch_support": support[:, 2:],
    "branch_reliability": torch.tensor(
        [[0.8, 0.4, 0.2, 0.0]], device="cuda"
    ),
    "branch_observed": observed,
    "branch_propagated": propagated,
    "branch_age_intervals": torch.tensor(
        [[0.0, 1.0, 0.0, 2.0]], device="cuda"
    ),
    "rsu_delay_intervals": torch.tensor([[1.0]], device="cuda"),
    "routing_mode": "uniform",
    "use_reliability": False,
    "use_delay_metadata": True,
}
with torch.no_grad():
    reference_output = reference(**common)
candidate_output = candidate(**common)
raw_weighted_mean = (0.6 * lidar + 0.1 * camera) / 0.7
expected = reference_output.fused + 0.175 * (
    raw_weighted_mean - reference_output.fused
)
torch.testing.assert_close(candidate_output.fused, expected, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(
    candidate_output.descriptor[:, 768:770],
    torch.ones(1, 2, device="cuda"),
    rtol=0.0,
    atol=0.0,
)
loss = candidate_output.fused.float().square().mean()
loss.backward()
for name, value in (("lidar", lidar.grad), ("camera", camera.grad)):
    if value is None or not torch.isfinite(value).all().item():
        raise RuntimeError(f"P1 target smoke {name} gradient is invalid")

p0_clean = DynamicExpertRouter(
    channels=256,
    hidden_channels=256,
    support_residual_weight=0.5,
    support_residual_reliability_gate=False,
).cuda().eval()
p0_clean.load_state_dict(candidate.state_dict(), strict=True)
clean = dict(common)
clean["branch_reliability"] = torch.ones(1, 4, device="cuda")
with torch.no_grad():
    p0_clean_output = p0_clean(**clean)
    p1_clean_output = candidate(**clean)
if not torch.equal(p0_clean_output.fused, p1_clean_output.fused):
    raise RuntimeError("P1 clean-path output is not bitwise P0-equivalent")

torch.cuda.synchronize()
print(
    json.dumps(
        {
            "event": "resilient_v2x_p1_target_a100_cuda_smoke_pass",
            "capabilities": capabilities,
            "config": (
                "configs/resilient_v2x/improvements/"
                "reliability_gated_residual.py"
            ),
            "full_model_parameters": full_model_parameters,
            "raw_reliability_gate": 0.35,
            "effective_residual_coefficient": 0.175,
            "clean_path_bitwise_p0_equivalent": True,
            "router_forward_backward": True,
        },
        sort_keys=True,
    )
)
"""
TARGET_A100_CUDA_SMOKE_SHA256 = hashlib.sha256(
    TARGET_A100_CUDA_SMOKE.encode("utf-8")
).hexdigest()

_P1_SPEC_ANCHOR = shared._P0_SPEC_ANCHOR
_P1_SPEC_PATCH = """    ExperimentSpec(
        "reliability_gated_residual",
        "sota_candidate",
        "configs/resilient_v2x/improvements/reliability_gated_residual.py",
        True,
    ),
    ExperimentSpec(
        "linear_no_distillation",
"""
_P1_NESTED_ANCHOR = """        "support_residual_no_reliability_linear",
        "resilient_v2x",
"""
_P1_NESTED_PATCH = """        "support_residual_no_reliability_linear",
        "reliability_gated_residual",
        "resilient_v2x",
"""
_CANDIDATE_PARSER_ANCHOR = """    parser.add_argument(
        "--teacher-task-id",
"""
_CANDIDATE_PARSER_PATCH = """    parser.add_argument("--candidate-variant")
    parser.add_argument("--candidate-identity")
    parser.add_argument("--evidence-class")
    parser.add_argument(
        "--teacher-task-id",
"""
_CANDIDATE_VALIDATE_ANCHOR = """    experiment_name = getattr(args, "experiment_from_task", None)
    if experiment_name is None:
"""
_CANDIDATE_VALIDATE_PATCH = """    experiment_name = getattr(args, "experiment_from_task", None)
    candidate_values = (
        args.candidate_variant,
        args.candidate_identity,
        args.evidence_class,
    )
    if experiment_name == "reliability_gated_residual":
        _p1_candidate_provenance(args, EXPERIMENT_BY_NAME[experiment_name])
    elif any(value is not None for value in candidate_values):
        raise ValueError("candidate provenance is only valid for P1")
    if experiment_name is None:
"""
_RUNTIME_HELPER_ANCHOR = """def _upload_experiment_checkpoint(
    *,
    task: object,
    output_model_class: object,
    spec: ExperimentSpec,
    checkpoint: Path,
) -> dict[str, object]:
"""
_RUNTIME_HELPER_PATCH = (
    f"P1_EXPECTED_CANDIDATE_PROVENANCE = {P1_CANDIDATE_PROVENANCE!r}\n"
    f"P1_RUNTIME_PROFILE = {P1_RUNTIME_PROFILE!r}\n"
    f"P1_RUNTIME_HARDWARE = {P1_RUNTIME_HARDWARE!r}\n"
    f"P1_RUNTIME_COMPUTE_CAPABILITY = {P1_RUNTIME_COMPUTE_CAPABILITY!r}\n"
    "P1_RUNTIME_IMPLEMENTATION_COMPATIBILITY_LAYER = "
    f"{P1_RUNTIME_IMPLEMENTATION_COMPATIBILITY_LAYER!r}\n"
    "P1_FORBIDDEN_RESULT_TAGS = frozenset({'RTX5090', 'sm120'})\n\n\n"
    "def _p1_candidate_provenance(\n"
    "    args: argparse.Namespace, spec: ExperimentSpec\n"
    ") -> dict[str, str] | None:\n"
    "    observed = {\n"
    "        'candidate_variant': args.candidate_variant,\n"
    "        'candidate_identity': args.candidate_identity,\n"
    "        'evidence_class': args.evidence_class,\n"
    "    }\n"
    "    if spec.name != 'reliability_gated_residual':\n"
    "        if any(value is not None for value in observed.values()):\n"
    "            raise ValueError('candidate provenance is only valid for P1')\n"
    "        return None\n"
    "    if observed != P1_EXPECTED_CANDIDATE_PROVENANCE:\n"
    "        raise ValueError(\n"
    "            f'P1 candidate provenance mismatch: {observed!r}'\n"
    "        )\n"
    "    return dict(observed)\n\n\n"
    "def _p1_runtime_tags(\n"
    "    spec: ExperimentSpec, candidate_provenance: Mapping[str, str] | None\n"
    ") -> list[str]:\n"
    "    tags = [\n"
    "        'A100',\n"
    "        'sm80',\n"
    "        'runtime_profile:a100_sm80',\n"
    "        (\n"
    "            'runtime_implementation_compatibility_layer:'\n"
    "            'sealed_native_bundle_python_source_compatibility'\n"
    "        ),\n"
    "    ]\n"
    "    if spec.name == 'reliability_gated_residual':\n"
    "        if candidate_provenance != P1_EXPECTED_CANDIDATE_PROVENANCE:\n"
    "            raise ValueError('P1 candidate provenance is not exact')\n"
    "        tags.extend(\n"
    "            f'{key}={candidate_provenance[key]}'\n"
    "            for key in (\n"
    "                'candidate_variant',\n"
    "                'candidate_identity',\n"
    "                'evidence_class',\n"
    "            )\n"
    "        )\n"
    "    elif candidate_provenance is not None:\n"
    "        raise ValueError('non-P1 output cannot carry P1 provenance')\n"
    "    return tags\n\n\n"
    "def _p1_require_result_tags(\n"
    "    tags: Sequence[str],\n"
    "    spec: ExperimentSpec,\n"
    "    candidate_provenance: Mapping[str, str] | None,\n"
    "    *,\n"
    "    context: str,\n"
    ") -> None:\n"
    "    observed = set(tags)\n"
    "    forbidden = observed & P1_FORBIDDEN_RESULT_TAGS\n"
    "    if forbidden:\n"
    "        raise RuntimeError(\n"
    "            f'{context} contains forbidden result tags: {sorted(forbidden)!r}'\n"
    "        )\n"
    "    required = set(_p1_runtime_tags(spec, candidate_provenance))\n"
    "    if not required <= observed:\n"
    "        raise RuntimeError(\n"
    "            f'{context} is missing required P1/A100 result tags'\n"
    "        )\n\n\n"
    "def _p1_checkpoint_contract_fields(\n"
    "    spec: ExperimentSpec, candidate_provenance: Mapping[str, str] | None\n"
    ") -> dict[str, object]:\n"
    "    required_result_tags = _p1_runtime_tags(spec, candidate_provenance)\n"
    "    fields: dict[str, object] = {\n"
    "        'runtime_profile': P1_RUNTIME_PROFILE,\n"
    "        'runtime_hardware': P1_RUNTIME_HARDWARE,\n"
    "        'cuda_compute_capability': P1_RUNTIME_COMPUTE_CAPABILITY,\n"
    "        'runtime_implementation_compatibility_layer': dict(\n"
    "            P1_RUNTIME_IMPLEMENTATION_COMPATIBILITY_LAYER\n"
    "        ),\n"
    "        'required_result_tags': list(required_result_tags),\n"
    "    }\n"
    "    if candidate_provenance is not None:\n"
    "        fields.update(candidate_provenance)\n"
    "    return fields\n\n\n" + _RUNTIME_HELPER_ANCHOR
)
_FINAL_SIGNATURE_ANCHOR = _RUNTIME_HELPER_ANCHOR
_FINAL_SIGNATURE_PATCH = """def _upload_experiment_checkpoint(
    *,
    task: object,
    output_model_class: object,
    spec: ExperimentSpec,
    checkpoint: Path,
    candidate_provenance: Mapping[str, str] | None,
) -> dict[str, object]:
"""
_FINAL_TAGS_ANCHOR = """        tags=[
            "ResilientV2X",
            spec.kind,
            spec.name,
            "DDP",
            "4GPU",
            "RTX5090",
            "sm120",
            "FP32",
        ],
    )
    uri = model.update_weights(
"""
_FINAL_TAGS_PATCH = """        tags=[
            "ResilientV2X",
            spec.kind,
            spec.name,
            "DDP",
            "4GPU",
            "FP32",
            *_p1_runtime_tags(spec, candidate_provenance),
        ],
    )
    _p1_require_result_tags(
        model.tags,
        spec,
        candidate_provenance,
        context="final checkpoint OutputModel",
    )
    uri = model.update_weights(
"""
_FINAL_CONTRACT_ANCHOR = """    return {
        "model_id": str(getattr(model, "id", "") or ""),
        "name": f"ResilientV2X {spec.name} final checkpoint",
        "url": uploaded_uri,
        "filename": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }
"""
_FINAL_CONTRACT_PATCH = """    return {
        **_p1_checkpoint_contract_fields(spec, candidate_provenance),
        "model_id": str(getattr(model, "id", "") or ""),
        "name": f"ResilientV2X {spec.name} final checkpoint",
        "url": uploaded_uri,
        "filename": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }
"""
_BEST_SIGNATURE_ANCHOR = """def _upload_experiment_best_checkpoint(
    *,
    task: object,
    output_model_class: object,
    spec: ExperimentSpec,
    checkpoint: Path,
    epoch: int,
) -> dict[str, object]:
"""
_BEST_SIGNATURE_PATCH = """def _upload_experiment_best_checkpoint(
    *,
    task: object,
    output_model_class: object,
    spec: ExperimentSpec,
    checkpoint: Path,
    epoch: int,
    candidate_provenance: Mapping[str, str] | None,
) -> dict[str, object]:
"""
_BEST_TAGS_ANCHOR = """        tags=[
            "ResilientV2X",
            spec.kind,
            spec.name,
            "clean-val-best",
            "diagnostic",
            "not-final-claim",
            "DDP",
            "4GPU",
            "FP32",
        ],
    )
    uri = model.update_weights(
"""
_BEST_TAGS_PATCH = """        tags=[
            "ResilientV2X",
            spec.kind,
            spec.name,
            "clean-val-best",
            "diagnostic",
            "not-final-claim",
            "DDP",
            "4GPU",
            "FP32",
            *_p1_runtime_tags(spec, candidate_provenance),
        ],
    )
    _p1_require_result_tags(
        model.tags,
        spec,
        candidate_provenance,
        context="clean-val best OutputModel",
    )
    uri = model.update_weights(
"""
_BEST_CONTRACT_ANCHOR = """    return {
        "model_id": str(getattr(model, "id", "") or ""),
        "name": name,
        "url": _require_files_server_url(
            uri,
            context="clean-val best checkpoint OutputModel URL",
        ),
        "filename": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "epoch": epoch,
        "selection_protocol": "DAIR-CLEAN-PAIR1789-v1",
        "selection_metric": "resilient_v2x/car_bev_ap_r40_0.70",
        "claim_role": "diagnostic checkpoint candidate; final remains canonical",
    }
"""
_BEST_CONTRACT_PATCH = """    return {
        **_p1_checkpoint_contract_fields(spec, candidate_provenance),
        "model_id": str(getattr(model, "id", "") or ""),
        "name": name,
        "url": _require_files_server_url(
            uri,
            context="clean-val best checkpoint OutputModel URL",
        ),
        "filename": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "epoch": epoch,
        "selection_protocol": "DAIR-CLEAN-PAIR1789-v1",
        "selection_metric": "resilient_v2x/car_bev_ap_r40_0.70",
        "claim_role": "diagnostic checkpoint candidate; final remains canonical",
    }
"""
_RUN_CONTRACT_SIGNATURE_ANCHOR = """def _experiment_run_contract(
    args: argparse.Namespace,
    *,
    task_id: str,
    spec: ExperimentSpec,
    dataset_root: Path,
    teacher_contract: Mapping[str, object] | None,
    predecessor_task_id: str,
    config_path: Path,
    training_command: Sequence[str],
    baseline_plan: Mapping[str, object] | None,
) -> dict[str, object]:
"""
_RUN_CONTRACT_SIGNATURE_PATCH = """def _experiment_run_contract(
    args: argparse.Namespace,
    *,
    task_id: str,
    spec: ExperimentSpec,
    dataset_root: Path,
    teacher_contract: Mapping[str, object] | None,
    predecessor_task_id: str,
    config_path: Path,
    training_command: Sequence[str],
    baseline_plan: Mapping[str, object] | None,
    candidate_provenance: Mapping[str, str] | None,
) -> dict[str, object]:
"""
_RUN_CONTRACT_FIELDS_ANCHOR = """        "experiment": spec.name,
        "experiment_kind": spec.kind,
        "config": {
"""
_RUN_CONTRACT_FIELDS_PATCH = """        "experiment": spec.name,
        "experiment_kind": spec.kind,
        **_p1_checkpoint_contract_fields(spec, candidate_provenance),
        "config": {
"""
_RUN_CONTRACT_LEGACY_RUNTIME_ANCHOR = """        "precision": "FP32",
        "runtime_profile": "rtx5090",
        "val_interval": RTX5090_VAL_INTERVAL,
"""
_RUN_CONTRACT_LEGACY_RUNTIME_PATCH = """        "precision": "FP32",
        "val_interval": RTX5090_VAL_INTERVAL,
"""
_EXECUTE_PROVENANCE_ANCHOR = """    spec = EXPERIMENT_BY_NAME[args.experiment_from_task]
    task = _current_or_init_experiment_task(task_class, spec)
"""
_EXECUTE_PROVENANCE_PATCH = """    spec = EXPERIMENT_BY_NAME[args.experiment_from_task]
    candidate_provenance = _p1_candidate_provenance(args, spec)
    task = _current_or_init_experiment_task(task_class, spec)
"""
_TASK_TAGS_ANCHOR = """    existing_tags = list(task.get_tags())
    task.set_tags(
        list(
            dict.fromkeys(
                existing_tags
                + [
                    "ResilientV2X-suite",
                    spec.kind,
                    spec.name,
                    "4gpu",
                    "RTX5090",
                    "sm120",
                    "FP32",
                    "DDP",
                ]
            )
        )
    )

"""
_TASK_TAGS_PATCH = """    existing_tags = [
        tag for tag in task.get_tags() if tag not in P1_FORBIDDEN_RESULT_TAGS
    ]
    required_runtime_tags = _p1_runtime_tags(spec, candidate_provenance)
    updated_tags = list(
        dict.fromkeys(
            existing_tags
            + [
                "ResilientV2X-suite",
                spec.kind,
                spec.name,
                "4gpu",
                "FP32",
                "DDP",
                *required_runtime_tags,
            ]
        )
    )
    if task.set_tags(updated_tags) is False:
        raise RuntimeError("P1 task rejected exact A100 result tags")
    _p1_require_result_tags(
        task.get_tags(),
        spec,
        candidate_provenance,
        context="P1 training task",
    )

"""
_RUN_CONTRACT_CALL_ANCHOR = """        training_command=training_command,
        baseline_plan=baseline_plan,
    )
    if not task.upload_artifact(
        "run_contract",
"""
_RUN_CONTRACT_CALL_PATCH = """        training_command=training_command,
        baseline_plan=baseline_plan,
        candidate_provenance=candidate_provenance,
    )
    if not task.upload_artifact(
        "run_contract",
"""
_FINAL_CALL_ANCHOR = """        output_model_class=output_model_class,
        spec=spec,
        checkpoint=checkpoint,
    )
    if not task.upload_artifact(
        "final_checkpoint_contract",
"""
_FINAL_CALL_PATCH = """        output_model_class=output_model_class,
        spec=spec,
        checkpoint=checkpoint,
        candidate_provenance=candidate_provenance,
    )
    if not task.upload_artifact(
        "final_checkpoint_contract",
"""
_BEST_CALL_ANCHOR = """        spec=spec,
        checkpoint=best_checkpoint,
        epoch=best_epoch,
    )
    if not task.upload_artifact(
        "best_checkpoint_contract",
"""
_BEST_CALL_PATCH = """        spec=spec,
        checkpoint=best_checkpoint,
        epoch=best_epoch,
        candidate_provenance=candidate_provenance,
    )
    if not task.upload_artifact(
        "best_checkpoint_contract",
"""
_SMOKE_COMPILE_ANCHOR = """def _compile_embedded_smoke_scripts() -> None:
    compile(RUNTIME_NATIVE_SMOKE, "<rtx5090-native-smoke>", "exec")
    compile(MODEL_SMOKE, "<rtx5090-model-smoke>", "exec")
"""
_SMOKE_COMPILE_PATCH = (
    f"P1_TARGET_A100_CUDA_SMOKE = {TARGET_A100_CUDA_SMOKE!r}\n\n\n"
    "def _compile_embedded_smoke_scripts() -> None:\n"
    '    compile(RUNTIME_NATIVE_SMOKE, "<rtx5090-native-smoke>", "exec")\n'
    '    compile(MODEL_SMOKE, "<rtx5090-model-smoke>", "exec")\n'
    "    compile(\n"
    "        P1_TARGET_A100_CUDA_SMOKE,\n"
    '        "<resilient-v2x-p1-target-a100-smoke>",\n'
    '        "exec",\n'
    "    )\n"
)
_SMOKE_CALL_ANCHOR = """    _run_smoke(
        venv_python,
        MODEL_SMOKE,
        expected_event="rtx5090_teacher_model_smoke_pass",
        source_root=source_root,
        env=runtime_env,
    )

    runner = (source_root / "tools/resilient_v2x/clearml_train.py").resolve(strict=True)
"""
_SMOKE_CALL_PATCH = """    _run_smoke(
        venv_python,
        MODEL_SMOKE,
        expected_event="rtx5090_teacher_model_smoke_pass",
        source_root=source_root,
        env=runtime_env,
    )
    _run_smoke(
        venv_python,
        P1_TARGET_A100_CUDA_SMOKE,
        expected_event="resilient_v2x_p1_target_a100_cuda_smoke_pass",
        source_root=source_root,
        env=runtime_env,
    )

    runner = (source_root / "tools/resilient_v2x/clearml_train.py").resolve(strict=True)
"""
BOOTSTRAP_EXTRA_PATCHES = (
    (_CANDIDATE_PARSER_ANCHOR, _CANDIDATE_PARSER_PATCH),
    (_CANDIDATE_VALIDATE_ANCHOR, _CANDIDATE_VALIDATE_PATCH),
    (_RUNTIME_HELPER_ANCHOR, _RUNTIME_HELPER_PATCH),
    (_FINAL_SIGNATURE_ANCHOR, _FINAL_SIGNATURE_PATCH),
    (_FINAL_TAGS_ANCHOR, _FINAL_TAGS_PATCH),
    (_FINAL_CONTRACT_ANCHOR, _FINAL_CONTRACT_PATCH),
    (_BEST_SIGNATURE_ANCHOR, _BEST_SIGNATURE_PATCH),
    (_BEST_TAGS_ANCHOR, _BEST_TAGS_PATCH),
    (_BEST_CONTRACT_ANCHOR, _BEST_CONTRACT_PATCH),
    (_RUN_CONTRACT_SIGNATURE_ANCHOR, _RUN_CONTRACT_SIGNATURE_PATCH),
    (_RUN_CONTRACT_FIELDS_ANCHOR, _RUN_CONTRACT_FIELDS_PATCH),
    (_RUN_CONTRACT_LEGACY_RUNTIME_ANCHOR, _RUN_CONTRACT_LEGACY_RUNTIME_PATCH),
    (_EXECUTE_PROVENANCE_ANCHOR, _EXECUTE_PROVENANCE_PATCH),
    (_TASK_TAGS_ANCHOR, _TASK_TAGS_PATCH),
    (_RUN_CONTRACT_CALL_ANCHOR, _RUN_CONTRACT_CALL_PATCH),
    (_FINAL_CALL_ANCHOR, _FINAL_CALL_PATCH),
    (_BEST_CALL_ANCHOR, _BEST_CALL_PATCH),
    (_SMOKE_COMPILE_ANCHOR, _SMOKE_COMPILE_PATCH),
    (_SMOKE_CALL_ANCHOR, _SMOKE_CALL_PATCH),
)


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


BOOTSTRAP_PATCH_CONTRACT_SHA256 = _canonical_sha256(
    [
        {
            "anchor_sha256": hashlib.sha256(anchor.encode("utf-8")).hexdigest(),
            "replacement_sha256": hashlib.sha256(
                replacement.encode("utf-8")
            ).hexdigest(),
        }
        for anchor, replacement in BOOTSTRAP_EXTRA_PATCHES
    ]
)
TARGET_A100_SMOKE_CONTRACT = shared._sealed(
    {
        "schema_version": 1,
        "contract_type": "resilient_v2x_p1_target_a100_cuda_smoke_v1",
        "required_before_training_or_inference": True,
        "worker_queue": WORKER_QUEUE,
        "worker_queue_id": WORKER_QUEUE_ID,
        "visible_gpu_count": 4,
        "required_capabilities": [[8, 0], [8, 0], [8, 0], [8, 0]],
        "event": TARGET_A100_SMOKE_EVENT,
        "script_sha256": TARGET_A100_CUDA_SMOKE_SHA256,
        "bootstrap_patch_contract_sha256": BOOTSTRAP_PATCH_CONTRACT_SHA256,
        "source_file_sha256": dict(sorted(P1_TARGET_FILE_SHA256.items())),
        "checks": [
            "resolved P1 config identity and exact behavior fields",
            "full student detector CUDA build without external initialization",
            "detector-to-fusion-to-router reliability-gate propagation",
            "raw-reliability-gated router CUDA forward and backward",
            "effective residual coefficient 0.175 for the diagnostic vector",
            (
                "clean path is bitwise P0-equivalent only as a zero-shot "
                "diagnostic compatibility check"
            ),
        ],
    }
)
EXECUTION_CONTRACT = shared._sealed(
    {
        "schema_version": 3,
        "contract_type": "resilient_v2x_p1_dual_candidate_execution_v3",
        "default_candidate": P1_TRAINED_CANDIDATE,
        "training_launch_enabled": True,
        "task_parameter_bindings": dict(P1_TASK_PARAMETER_BINDINGS),
        "runtime_execution": {
            "runtime_profile": P1_RUNTIME_PROFILE,
            "runtime_hardware": P1_RUNTIME_HARDWARE,
            "cuda_compute_capability": P1_RUNTIME_COMPUTE_CAPABILITY,
            "runtime_implementation_compatibility_layer": dict(
                P1_RUNTIME_IMPLEMENTATION_COMPATIBILITY_LAYER
            ),
            "forbidden_result_tags": ["RTX5090", "sm120"],
        },
        "candidate_provenance_propagation": {
            "required_targets": [
                "task_parameters",
                "task_tags",
                "run_contract",
                "final_checkpoint_contract",
                "final_output_model_tags",
            ],
            "fail_closed": True,
        },
        "candidate_identity_rule": (
            "trained and zero-shot candidates have distinct identities and "
            "must never be merged in provenance, metrics, or conclusions"
        ),
        "candidate_variants": {
            P1_TRAINED_CANDIDATE: {
                "candidate_variant": P1_TRAINED_CANDIDATE,
                "candidate_identity": P1_TRAINED_CANDIDATE_IDENTITY,
                "evidence_class": "formal_p1_trained_candidate",
                "mode": "train_then_formal_inference",
                "config": P1_CONFIG,
                "training": {
                    "max_epochs": MAX_EPOCHS,
                    "val_interval": VAL_INTERVAL,
                    "gpu_count": GPU_COUNT,
                    "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
                    "global_batch_size": GLOBAL_BATCH_SIZE,
                    "precision": PRECISION,
                    "training_seed": TRAINING_SEED,
                    "training_overlay": {
                        "protocol_seed": OVERLAY_PROTOCOL_SEED,
                        "delays_ms": list(FORMAL_DELAYS_MS),
                        "fault_conditions": list(FORMAL_FAULT_CONDITIONS),
                        "condition_count": FORMAL_CONDITION_COUNT,
                        "uniformly_applied": True,
                    },
                },
                "output_checkpoint": {
                    "checkpoint_policy": FORMAL_CHECKPOINT_POLICY,
                    "checkpoint_filename": FORMAL_CHECKPOINT_FILENAME,
                    "output_model_role": "canonical_epoch_50_final_output_model",
                    "model_id": None,
                },
                "formal_evaluation": {
                    "protocol_id": FORMAL_PROTOCOL_ID,
                    "sample_count": FORMAL_SAMPLE_COUNT,
                    "overlay_seed": OVERLAY_PROTOCOL_SEED,
                    "delays_ms": list(FORMAL_DELAYS_MS),
                    "fault_conditions": list(FORMAL_FAULT_CONDITIONS),
                    "condition_count": FORMAL_CONDITION_COUNT,
                },
            },
            P1_ZERO_SHOT_CANDIDATE: {
                "candidate_identity": P1_ZERO_SHOT_CANDIDATE_IDENTITY,
                "evidence_class": "zero_shot_diagnostic_only",
                "mode": "inference_only",
                "inference_transform_config": P1_CONFIG,
                "source_checkpoint": {
                    "source_candidate": shared.P0_EXPERIMENT,
                    "source_candidate_identity": shared.P0_IDENTITY,
                    "source_task_id": P0_TRAINING_TASK_ID,
                    "required_source_task_status": "completed",
                    "checkpoint_policy": FORMAL_CHECKPOINT_POLICY,
                    "checkpoint_filename": FORMAL_CHECKPOINT_FILENAME,
                    "output_model_role": "canonical_epoch_50_final_output_model",
                    "model_id": None,
                    "model_resolution": (
                        "resolve and pin the completed P0 canonical epoch-50 "
                        "final OutputModel ID, bytes, and SHA-256"
                    ),
                    "forbidden_checkpoint_roles": [
                        "clean_validation_best",
                        "clean_validation_best_for_teacher_handoff",
                    ],
                },
                "compatibility_scope": (
                    "clean-path bitwise equality permits a zero-shot diagnostic "
                    "only and does not substitute for P1 training"
                ),
                "formal_evaluation": {
                    "protocol_id": FORMAL_PROTOCOL_ID,
                    "sample_count": FORMAL_SAMPLE_COUNT,
                    "overlay_seed": OVERLAY_PROTOCOL_SEED,
                    "delays_ms": list(FORMAL_DELAYS_MS),
                    "fault_conditions": list(FORMAL_FAULT_CONDITIONS),
                    "condition_count": FORMAL_CONDITION_COUNT,
                },
            },
        },
        "target_cuda_smoke": dict(TARGET_A100_SMOKE_CONTRACT),
    }
)


def _verified_p0_parent() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Verify all local P0 seals before accepting it as P1's immutable parent."""

    with shared.use_deployment_profile(shared.P0_PROFILE):
        manifest = shared._verify_package(P0_OUTPUT_DIR)
        deployment, static = shared._verify_deployment_plan(
            P0_OUTPUT_DIR,
            manifest,
        )
    if manifest.get("seal_sha256") != P0_PACKAGE_MANIFEST_SEAL_SHA256:
        raise RuntimeError("immutable P0 package manifest seal drifted")
    if static.get("seal_sha256") != P0_STATIC_PLAN_SEAL_SHA256:
        raise RuntimeError("immutable P0 static-plan seal drifted")
    if deployment.get("seal_sha256") != P0_DEPLOYMENT_PLAN_SEAL_SHA256:
        raise RuntimeError("immutable P0 deployment-plan seal drifted")
    return manifest, deployment, static


def _build_profile() -> shared.Round2DeploymentProfile:
    p0_manifest, p0_deployment, p0_static = _verified_p0_parent()
    p0_package = p0_manifest["target_source_package"]
    p0_candidate = p0_manifest["candidate"]
    assert isinstance(p0_package, dict)
    assert isinstance(p0_candidate, dict)
    provenance = {
        "derivation": "P1 exact three-core modification plus one config from P0",
        "parent_candidate": dict(p0_candidate),
        "parent_source_dataset_id": P0_SOURCE_DATASET_ID,
        "parent_source_package_manifest_seal_sha256": p0_manifest["seal_sha256"],
        "parent_source_tree_sha256": p0_package["tree_sha256"],
        "parent_static_plan_seal_sha256": p0_static["seal_sha256"],
        "parent_deployment_plan_seal_sha256": p0_deployment["seal_sha256"],
        "parent_template_task_id": P0_TEMPLATE_TASK_ID,
        "parent_training_task_id": P0_TRAINING_TASK_ID,
        "modified_source_paths": list(P1_MODIFIED_SOURCE_PATHS),
        "added_source_paths": [P1_CONFIG],
        "explicitly_excluded_paths": [P2_CONFIG],
    }
    return shared.Round2DeploymentProfile(
        label="round2 P1",
        output_dir=OUTPUT_DIR,
        package_manifest_name=PACKAGE_MANIFEST_NAME,
        deployment_plan_name=DEPLOYMENT_PLAN_NAME,
        package_type="resilient_v2x_sota_round2_p1_source",
        experiment=P1_EXPERIMENT,
        config=P1_CONFIG,
        identity=P1_IDENTITY,
        base_source_dir=P0_OUTPUT_DIR,
        base_source_inventory=P0_OUTPUT_DIR / "source-inventory.json",
        base_source_archive=P0_OUTPUT_DIR / str(p0_package["archive_name"]),
        base_source_dataset_id=P0_SOURCE_DATASET_ID,
        base_source_package=dict(p0_package),
        provenance=provenance,
        early_gate_policy=None,
        source_transition_artifact=SOURCE_TRANSITION_ARTIFACT,
        launch_receipt_artifact=LAUNCH_RECEIPT_ARTIFACT,
        source_dataset_prefix=SOURCE_DATASET_PREFIX,
        source_dataset_version_suffix="round2-p1-v1",
        source_dataset_tags=(
            "ResilientV2X",
            "source",
            "sota-candidates",
            "round2-p1",
            "reliability-gated-residual",
            "derived-from-p0",
            "sealed",
        ),
        template_prefix=TEMPLATE_PREFIX,
        task_prefix=TASK_PREFIX,
        worker_queue=WORKER_QUEUE,
        worker_queue_id=WORKER_QUEUE_ID,
        gpu_count=GPU_COUNT,
        batch_size_per_gpu=BATCH_SIZE_PER_GPU,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        val_interval=VAL_INTERVAL,
        training_seed=TRAINING_SEED,
        precision=PRECISION,
        upload_token=UPLOAD_TOKEN,
        template_token=TEMPLATE_TOKEN,
        launch_token=LAUNCH_TOKEN,
        spec_anchor=_P1_SPEC_ANCHOR,
        spec_patch=_P1_SPEC_PATCH,
        nested_anchor=_P1_NESTED_ANCHOR,
        nested_patch=_P1_NESTED_PATCH,
        bootstrap_parent=shared.P0_PROFILE,
        transition_type=(
            "exact_three_modified_one_additive_sota_round2_p1_from_p0_revision"
        ),
        plan_type="resilient_v2x_sota_round2_p1_dual_candidate_v3",
        receipt_type="resilient_v2x_sota_round2_p1_launch_v2",
        template_status_message="sealed round2 P1 bootstrap template",
        transition_description=(
            "P1 derived from sealed P0; exactly three core files modified and "
            "one P1 config added; P2 config excluded"
        ),
        invariants=(
            (
                "source transition starts from sealed P0 and modifies exactly "
                "detector, fusion, and routing while adding only the P1 config"
            ),
            "the independent P2 bbox2.5 config is absent from source and bootstrap",
            (
                "P0 Dataset, package, template, task, launcher, checkpoints, "
                "and artifacts are read-only and unchanged"
            ),
            (
                "the default candidate is an independently identified P1 "
                "50-epoch FP32 training run with validation every 10 epochs"
            ),
            (
                "all formal training and evaluation overlay bindings use seed "
                "20250218 and exactly 12 delay-by-fault conditions; 1337 is "
                "the formal evaluation sample count, never a seed"
            ),
            (
                "the separately identified zero-shot candidate may use only "
                "P0's canonical epoch-50 final OutputModel; clean-best "
                "checkpoints are forbidden"
            ),
            (
                "clean-path equality supports zero-shot diagnostic compatibility "
                "only and never waives P1 training; target A100 smoke is required"
            ),
            (
                "trained-candidate identity is fail-closed across parameters, "
                "task/model tags, run contract, and final-checkpoint contract"
            ),
            (
                "queue name resolves to the exact sealed A100 queue ID before "
                "enqueue; post-enqueue state has a sealed server readback"
            ),
        ),
        duplicate_guard_description=(
            "exact task name plus predecessor parent before clone, followed by "
            "forced launch-receipt readback and a unique exact-name+parent "
            "current-clone ID guard immediately before enqueue"
        ),
        duplicate_guard_stages=(
            "pre_clone_requires_zero_exact_name_plus_parent_matches",
            (
                "post_receipt_readback_pre_enqueue_requires_one_exact_name_plus_"
                "parent_match_with_current_clone_id"
            ),
        ),
        relative_script=Path(__file__).relative_to(ROOT),
        cli_description=__doc__ or "",
        modified_source_paths=P1_MODIFIED_SOURCE_PATHS,
        forbidden_source_paths=(P2_CONFIG,),
        bootstrap_extra_patches=BOOTSTRAP_EXTRA_PATCHES,
        bootstrap_required_markers=(
            "P1_TARGET_A100_CUDA_SMOKE",
            TARGET_A100_SMOKE_EVENT,
            P1_CONFIG,
            P1_TRAINED_CANDIDATE,
            P1_TRAINED_CANDIDATE_IDENTITY,
            "formal_p1_trained_candidate",
            "runtime_profile:a100_sm80",
            "runtime_hardware",
            "runtime_implementation_compatibility_layer",
        ),
        bootstrap_forbidden_markers=(P2_EXPERIMENT, P2_CONFIG),
        execution_contract=EXECUTION_CONTRACT,
        training_launch_enabled=True,
        task_parameter_bindings=P1_TASK_PARAMETER_BINDINGS,
        strict_enqueue_acknowledgement=True,
        enqueue_acknowledgement_artifact=ENQUEUE_ACKNOWLEDGEMENT_ARTIFACT,
    )


P1_PROFILE = _build_profile()


def _show_execution_contract(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Show the sealed P1 dual-candidate execution contract."
    )
    parser.add_argument("command", choices=("show-execution-contract",))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(list(argv))
    output_dir = args.output_dir.resolve(strict=False)
    with shared.use_deployment_profile(P1_PROFILE):
        manifest = shared._verify_package(output_dir)
        deployment, static = shared._verify_deployment_plan(output_dir, manifest)
    contract = static.get("execution_contract")
    if contract != dict(EXECUTION_CONTRACT):
        raise RuntimeError("P1 persisted execution contract drifted")
    print(
        json.dumps(
            {
                "deployment_plan_seal_sha256": deployment["seal_sha256"],
                "static_plan_seal_sha256": static["seal_sha256"],
                "source_tree_sha256": manifest["target_source_package"]["tree_sha256"],
                "execution_contract": contract,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "show-execution-contract":
        return _show_execution_contract(arguments)
    with shared.use_deployment_profile(P1_PROFILE):
        return shared.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
