#!/usr/bin/env python3
"""Select a formal ResilientV2X candidate from independently verified results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath

try:
    from tools.resilient_v2x import sota_gate
except ModuleNotFoundError:  # pragma: no cover - standalone ClearML execution
    import sota_gate  # type: ignore[no-redef]

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
AUDIT_ARTIFACT = "formal_1337_comparability_audit"
FORMAL_EVALUATION_PLAN_ARTIFACT = "formal_1337_evaluation_plan"
LEADERBOARD_ARTIFACT = "formal_1337_leaderboard"
METRICS_ARTIFACT = "controlled_baseline_metrics"
SELECTION_ARTIFACT = "formal_candidate_selection"
FORMAL_INPUTS_ARTIFACT = "final_selector_formal_inputs"
FORMAL_INPUTS_DOCUMENT_TYPE = "resilient_v2x_final_selector_formal_inputs"
RUN_CONTRACT_ARTIFACT = "run_contract"
INITIALIZATION_AUDIT_ARTIFACT = "common_teacher_initialization_audit"
FINAL_CHECKPOINT_CONTRACT_ARTIFACT = "final_checkpoint_contract"
EVALUATION_PLAN_ARTIFACT = "evaluation_plan"
PREDICTION_EVIDENCE_ARTIFACT = "controlled_baseline_evidence"

DEFAULT_AUDIT_TASK_ID = "e19bab92884248f4ac07167e7eb66170"
DEFAULT_LEADERBOARD_TASK_ID = "f502bdd329ad4ef4b4b6cf5c5f52aba0"
# Active formal-chain deployment pins, grouped for one auditable identity check.
TRAINING_CONTROLLER_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
TRAINING_PROVENANCE_TASK_ID = "7274a1a5344a44c18aa7bcc6d1cb2e95"
WATCHER_TASK_ID = "7734387ddfb74b11ba6d84f3fea0bb97"
WATCHER_ENTRY_POINT = "clearml_1337_dependency_watcher.py"
LEADERBOARD_ENTRY_POINT = "clearml_1337_leaderboard.py"
AUDIT_ENTRY_POINT = "clearml_formal_comparability_audit.py"
WATCHER_SCRIPT_SHA256 = (
    "80c57bc341b39caa0aa7d41f4e64b947eaa8831ab592784daaf11aebf4b179ff"
)
LEADERBOARD_SCRIPT_SHA256 = (
    "1da0cf5dd4435c6ae85d5a474b5a67ec8435f50e9878a3d8f0c1d2872356524b"
)
AUDIT_SCRIPT_SHA256 = "4746ae18bb67757c974d96e851f19cb01872f460aaf3b334609a0de988aaabaa"
TRAINING_ENTRY_POINT = "clearml_5090_bootstrap.py"
LEGACY_TRAINING_SCRIPT_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
CANONICAL_TRAINING_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
EVALUATION_SCRIPT_SHA256 = CANONICAL_TRAINING_SCRIPT_SHA256
LEGACY_PARENT_TASK_ID = "6525107e60ae4104a2800731d74ecd4e"
NO_DISTILLATION_PARENT_TASK_ID = "d4b83d9b68704050aeb24a2e34540d8a"
RECOVERY_PARENT_TASK_ID = "bfcf18a3fd484776adcc5efd48a2c95e"
LEGACY_SCRIPT_SUBJECTS = ("support_residual", "no_distillation")
MODEL_BYTES_VERIFIER = "tools/resilient_v2x/collect_clearml_formal_models.py"

PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
SEALED_SUCCESSOR_PIN_ARGS = (
    ("evaluation_plan_amendment_producer_task_id", "task_id"),
    ("evaluation_plan_amendment_receipt_seal_sha256", "sha256"),
    ("evaluation_plan_amendment_revised_plan_seal_sha256", "sha256"),
    ("evaluation_plan_amendment_evidence_seal_sha256", "sha256"),
    ("evaluation_plan_amendment_worker_evidence_seal_sha256", "sha256"),
    ("evaluation_plan_amendment_task_ids_sha256", "sha256"),
    ("exact_eval_runtime_recovery_receipt_seal_sha256", "sha256"),
    ("exact_eval_runtime_recovery_attempt_seal_sha256", "sha256"),
    ("exact_eval_runtime_ffnet_source_sha256", "sha256"),
    ("exact_eval_runtime_ffnet_parameters_sha256", "sha256"),
    ("exact_eval_runtime_candidate_source_sha256", "sha256"),
    ("exact_eval_runtime_candidate_parameters_sha256", "sha256"),
)
SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID = "f8c36e508c7d453dadc766207a5b25b2"
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
SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = (
    "29de9700cac66f9998be643e85a8ec646c04ec17fddbb6207bc1438e9e73941b"
)
SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = (
    "c3170f4a88b080f9cc7267053f354640dd260687cb2c35a6f7ffed73d69f4154"
)
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
TRAINING_SEED = 20_250_218
TEACHER_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
TEACHER_MODEL_ID = "d962f6bae8474260b54e170a7a5f0418"
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
SAMPLE_COUNT = 1_337
GROUND_TRUTH_COUNT = 11_330
UNSUPPORTED_SAMPLE_COUNT = 0
DELAYS_MS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
SUPPORTED_EVALUATION_QUEUES = frozenset({"GPU4-A100", "GPU4-V100", "GPU4-5090"})
CHECKPOINT_POLICY = "epoch_50_final_only"
MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
OVERLAY_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
SAMPLE_IDS_SHA256 = "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"

WAITING_STATUSES = frozenset({"created", "queued", "in_progress"})
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
SOURCE_REVISION_BY_SUBJECT = {
    subject: ("new" if subject in NEW_SOURCE_SUBJECTS else "old")
    for subject in SUBJECT_ORDER
}
SOURCE_BY_REVISION = {
    "old": {
        "tree_sha256": SOURCE_TREE_SHA256,
        "dataset_id": SOURCE_DATASET_ID,
        "archive_name": SOURCE_ARCHIVE_NAME,
        "archive_bytes": SOURCE_ARCHIVE_BYTES,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
    },
    "new": {
        "tree_sha256": NEW_SOURCE_TREE_SHA256,
        "dataset_id": NEW_SOURCE_DATASET_ID,
        "archive_name": NEW_SOURCE_ARCHIVE_NAME,
        "archive_bytes": NEW_SOURCE_ARCHIVE_BYTES,
        "archive_sha256": NEW_SOURCE_ARCHIVE_SHA256,
    },
}
SOURCE_REVISION_CERTIFICATE_BY_TREE = {
    SOURCE_TREE_SHA256: {
        "dataset_id": SOURCE_DATASET_ID,
        "tree_sha256": SOURCE_TREE_SHA256,
        "file_count": SOURCE_FILE_COUNT,
        "source_bytes": SOURCE_BYTES,
        "archive": {
            "name": SOURCE_ARCHIVE_NAME,
            "size_bytes": SOURCE_ARCHIVE_BYTES,
            "sha256": SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": SOURCE_INVENTORY_BYTES,
            "sha256": SOURCE_INVENTORY_SHA256,
        },
    },
    NEW_SOURCE_TREE_SHA256: {
        "dataset_id": NEW_SOURCE_DATASET_ID,
        "tree_sha256": NEW_SOURCE_TREE_SHA256,
        "file_count": NEW_SOURCE_FILE_COUNT,
        "source_bytes": NEW_SOURCE_BYTES,
        "archive": {
            "name": NEW_SOURCE_ARCHIVE_NAME,
            "size_bytes": NEW_SOURCE_ARCHIVE_BYTES,
            "sha256": NEW_SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": NEW_SOURCE_INVENTORY_BYTES,
            "sha256": NEW_SOURCE_INVENTORY_SHA256,
        },
    },
}
CANONICAL_LEADERBOARD_BASELINE_SUBJECTS = (
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
# The broad 14-method group remains a descriptive canonical-leaderboard
# taxonomy.  Only the paper's five controlled adaptations define the SOTA gate.
BASELINE_SUBJECTS = sota_gate.BASELINE_SUBJECTS
CANDIDATE_ORDER = ("resilient_v2x",)
PRIMARY_SUBJECT = "resilient_v2x"
FORMAL_INPUT_SUBJECTS = (*BASELINE_SUBJECTS, PRIMARY_SUBJECT)
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
SUBJECT_KIND = {
    subject: (
        "baseline"
        if subject in CANONICAL_LEADERBOARD_BASELINE_SUBJECTS
        else "ablation"
        if subject in ABLATION_SUBJECTS
        else "improvement"
        if subject in IMPROVEMENT_SUBJECTS
        else "primary_method"
    )
    for subject in SUBJECT_ORDER
}
AP_METRIC_KEYS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)
LEADERSHIP_METRIC = sota_gate.LEADERSHIP_METRIC
COUNT_METRICS = {
    "resilient_v2x/sample_count": SAMPLE_COUNT,
    "resilient_v2x/car_ground_truth_count": GROUND_TRUTH_COUNT,
    "resilient_v2x/unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
}
RANKING_KEY = sota_gate.RANKING_KEY

if not all(
    (
        PROTOCOL_ID == sota_gate.PROTOCOL_ID,
        SAMPLE_COUNT == sota_gate.SAMPLE_COUNT,
        GROUND_TRUTH_COUNT == sota_gate.GROUND_TRUTH_COUNT,
        UNSUPPORTED_SAMPLE_COUNT == sota_gate.UNSUPPORTED_SAMPLE_COUNT,
        MANIFEST_CONTENT_SHA256 == sota_gate.MANIFEST_CONTENT_SHA256,
        OVERLAY_INDEX_CONTENT_SHA256 == sota_gate.OVERLAY_INDEX_CONTENT_SHA256,
        SAMPLE_IDS_SHA256 == sota_gate.SAMPLE_IDS_SHA256,
        DELAYS_MS == sota_gate.DELAYS_MS,
        CONDITIONS == sota_gate.CONDITIONS,
    )
):
    raise RuntimeError("formal selector and SOTA gate protocol fingerprints diverged")


def _expected_training_script_sha256(subject: str) -> str:
    if subject not in SUBJECT_ORDER:
        raise ValueError(f"unknown formal subject {subject!r}")
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
    if subject not in SUBJECT_ORDER:
        raise ValueError(f"unknown formal subject {subject!r}")
    if subject in NEW_SOURCE_SUBJECTS:
        return TRAINING_CONTROLLER_TASK_ID
    return SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID


def _expected_source(subject: str) -> Mapping[str, object]:
    if subject not in SUBJECT_ORDER:
        raise ValueError(f"unknown formal subject {subject!r}")
    return SOURCE_BY_REVISION[SOURCE_REVISION_BY_SUBJECT[subject]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-task-id", default=DEFAULT_AUDIT_TASK_ID)
    parser.add_argument("--leaderboard-task-id", default=DEFAULT_LEADERBOARD_TASK_ID)
    parser.add_argument(
        "--formal-inputs-artifact",
        default=FORMAL_INPUTS_ARTIFACT,
        choices=(FORMAL_INPUTS_ARTIFACT,),
        help=(
            "Write the sealed schema-v2 final-selector inputs alongside the "
            "legacy formal selection."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    for name, _kind in SEALED_SUCCESSOR_PIN_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", default="")
    return parser


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _runtime_source() -> str:
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError("cannot read the running selector source") from error
    if not source:
        raise RuntimeError("running selector source is empty")
    return source


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _clearml_id(value: object, context: str) -> str:
    if type(value) is not str or not _is_lower_hex(value, 32):
        raise ValueError(f"{context} must be a lowercase 32-hex ClearML ID")
    return value


def _sha256(value: object, context: str) -> str:
    if type(value) is not str or not _is_lower_hex(value, 64):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _sealed_successor_pins(args: argparse.Namespace) -> dict[str, str]:
    raw = {
        name: str(getattr(args, name, "") or "")
        for name, _kind in SEALED_SUCCESSOR_PIN_ARGS
    }
    if not any(raw.values()):
        return {}
    if not all(raw.values()):
        raise ValueError("all amended plan and runtime recovery pins are required")
    result: dict[str, str] = {}
    for name, kind in SEALED_SUCCESSOR_PIN_ARGS:
        value = (
            _clearml_id(raw[name], name)
            if kind == "task_id"
            else _sha256(raw[name], name)
        )
        result[f"Args/{name}"] = value
    return result


def _exact_json_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        if set(observed) != set(expected):
            return False
        return all(
            _exact_json_equal(observed[key], expected_value)
            for key, expected_value in expected.items()
        )
    if type(expected) is list:
        if len(observed) != len(expected):
            return False
        return all(
            _exact_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = _sha256(value.get("seal_sha256"), f"{context} seal")
    if _sealed(value)["seal_sha256"] != observed:
        raise ValueError(f"{context} seal SHA-256 mismatch")
    return observed


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{context} keys mismatch; missing={missing!r}, extra={extra!r}"
        )


def _server_artifacts(task: object, *, context: str) -> dict[str, object]:
    """Materialize artifacts only from the installed raw backend snapshot."""

    data = getattr(task, "data", None)
    execution = getattr(data, "execution", None)
    raw_artifacts = getattr(execution, "artifacts", None)
    if raw_artifacts is None:
        raw_artifacts = ()
    if type(raw_artifacts) not in {list, tuple}:
        raise RuntimeError(f"{context} raw server artifacts are invalid")
    if ClearMLArtifact is None:
        raise RuntimeError(f"{context} ClearML artifact reader is unavailable")
    result: dict[str, object] = {}
    for raw in raw_artifacts:
        name = getattr(raw, "key", None)
        if type(name) is not str or not name or name in result:
            raise RuntimeError(f"{context} raw server artifact inventory is invalid")
        try:
            result[name] = ClearMLArtifact(raw)
        except Exception as error:
            raise RuntimeError(
                f"{context} raw server artifact {name!r} cannot be materialized"
            ) from error
    return result


def _artifact_names(task: object, *, context: str) -> tuple[str, ...]:
    return tuple(sorted(_server_artifacts(task, context=context)))


def _artifact_mapping(task: object, name: str, *, context: str) -> Mapping[str, object]:
    artifacts = _server_artifacts(task, context=context)
    if name not in artifacts:
        raise RuntimeError(f"{context} lacks artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise RuntimeError(f"{context} artifact {name!r} cannot be read")
    try:
        value = getter(force_download=True)
    except Exception as error:
        raise RuntimeError(
            f"{context} artifact {name!r} cannot be freshly downloaded"
        ) from error
    if isinstance(value, Mapping):
        try:
            document = json.loads(_canonical_json(value))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"{context} artifact {name!r} is not canonical JSON"
            ) from error
        if type(document) is not dict:
            raise RuntimeError(f"{context} artifact {name!r} is not a JSON object")
        return document
    if not isinstance(value, (str, Path)):
        raise RuntimeError(f"{context} artifact {name!r} is not a JSON object")
    path = Path(value)
    try:
        if not path.is_file():
            raise RuntimeError(f"{context} artifact {name!r} is not a regular file")
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{context} artifact {name!r} cannot be read as JSON"
        ) from error
    if type(document) is not dict:
        raise RuntimeError(f"{context} artifact {name!r} is not a JSON object")
    return document


def _artifact_metadata_records(
    task: object, *, context: str
) -> dict[str, dict[str, object]]:
    """Read byte identities from the same authoritative server snapshot."""

    execution = getattr(getattr(task, "data", None), "execution", None)
    raw_artifacts = getattr(execution, "artifacts", None)
    if type(raw_artifacts) not in {list, tuple}:
        raise RuntimeError(f"{context} raw artifact metadata is invalid")
    records: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(raw_artifacts):
        converter = getattr(raw, "to_dict", None)
        value = converter() if callable(converter) else raw

        def field(name: str) -> object:
            if isinstance(value, Mapping):
                return value.get(name)
            return getattr(value, name, None)

        name = field("key")
        if type(name) is not str or not name or name in records:
            raise RuntimeError(
                f"{context} raw artifact record {index} identity is invalid"
            )
        artifact_hash = _sha256(field("hash"), f"{context} artifact {name!r} byte hash")
        content_size = field("content_size")
        if type(content_size) is not int or content_size <= 0:
            raise RuntimeError(f"{context} artifact {name!r} content size is invalid")
        records[name] = {
            "hash": artifact_hash,
            "content_size": content_size,
        }
    if tuple(sorted(records)) != _artifact_names(task, context=context):
        raise RuntimeError(f"{context} artifact metadata inventory drifted")
    return records


def _safe_config_path(value: object, *, context: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{context} config path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or tuple(path.parts)[:1] != ("configs",)
        or "." in path.parts
        or ".." in path.parts
    ):
        raise ValueError(f"{context} config path is outside configs/")
    return value


def _parameters(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot expose parameters")
    try:
        value = getter(cast=False)
    except TypeError:
        value = getter()
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} parameters are invalid")
    return dict(value)


def _parameter_matches(value: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return str(value).casefold() == str(expected).casefold()
    return str(value) == str(expected)


def _require_parameters(
    observed: Mapping[str, object], expected: Mapping[str, object], *, context: str
) -> None:
    for key, expected_value in expected.items():
        if key not in observed or not _parameter_matches(observed[key], expected_value):
            raise RuntimeError(f"{context} parameter {key!r} mismatch")


def _task_parent(task: object) -> str:
    value = getattr(getattr(task, "data", None), "parent", None)
    if value is None:
        return ""
    if type(value) is not str:
        raise RuntimeError("task parent must be a string or None")
    return value


def _script_sha256(task: object, *, entry_point: str, context: str) -> str:
    script = getattr(getattr(task, "data", None), "script", None)
    if script is None:
        raise RuntimeError(f"{context} has no script metadata")
    if str(getattr(script, "repository", "") or "") != "":
        raise RuntimeError(f"{context} is not a standalone script")
    if str(getattr(script, "working_dir", "") or "") != ".":
        raise RuntimeError(f"{context} working directory drifted")
    if str(getattr(script, "entry_point", "") or "") != entry_point:
        raise RuntimeError(f"{context} entry point drifted")
    diff = str(getattr(script, "diff", "") or "")
    if not diff:
        raise RuntimeError(f"{context} standalone source is empty")
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _validate_producer(
    task: object,
    *,
    task_id: str,
    parent_id: str,
    entry_point: str,
    script_sha256: str,
    parameters: Mapping[str, object],
    context: str,
) -> None:
    if _clearml_id(getattr(task, "id", ""), context) != task_id:
        raise RuntimeError(f"{context} identity mismatch")
    if _task_parent(task) != parent_id:
        raise RuntimeError(f"{context} parent mismatch")
    observed_sha = _script_sha256(task, entry_point=entry_point, context=context)
    if observed_sha != script_sha256:
        raise RuntimeError(f"{context} script SHA-256 mismatch")
    observed_parameters = _parameters(task, context=context)
    if set(observed_parameters) != set(parameters):
        raise RuntimeError(f"{context} exact parameter inventory mismatch")
    _require_parameters(observed_parameters, parameters, context=context)


def _reload(task: object, *, context: str) -> None:
    """Install one uncached backend snapshot without using Task.reload()."""

    expected_task_id = _clearml_id(getattr(task, "id", ""), f"{context} local task")
    if bool(getattr(task, "_offline_mode", False)):
        raise RuntimeError(f"{context} cannot use offline reload")
    reloader = getattr(task, "_reload", None)
    if not callable(reloader):
        raise RuntimeError(f"{context} cannot be server-reloaded")
    has_skip_flag = hasattr(task, "_reload_skip_flag")
    previous_skip_flag = getattr(task, "_reload_skip_flag", None)
    try:
        if has_skip_flag:
            setattr(task, "_reload_skip_flag", False)
        snapshot = reloader()
    except Exception as error:
        raise RuntimeError(f"{context} server reload failed") from error
    finally:
        if has_skip_flag:
            setattr(task, "_reload_skip_flag", previous_skip_flag)
    if snapshot is None or isinstance(
        snapshot, (bool, int, float, str, bytes, bytearray)
    ):
        raise RuntimeError(f"{context} server reload returned no snapshot")
    snapshot_task_id = _clearml_id(
        getattr(snapshot, "id", ""), f"{context} server snapshot"
    )
    if snapshot_task_id != expected_task_id:
        raise RuntimeError(f"{context} server snapshot identity mismatch")
    try:
        setattr(task, "_data", snapshot)
    except Exception as error:
        raise RuntimeError(f"{context} server snapshot cannot be installed") from error
    if (
        getattr(task, "_data", None) is not snapshot
        or getattr(task, "data", None) is not snapshot
    ):
        raise RuntimeError(f"{context} server snapshot was not installed exactly")


def _status(task: object, *, context: str) -> str:
    value = getattr(task, "status", None)
    if value is None:
        getter = getattr(task, "get_status", None)
        value = getter() if callable(getter) else None
    value = getattr(value, "value", value)
    if type(value) is not str or not value:
        raise RuntimeError(f"{context} status is unavailable")
    return value.rsplit(".", 1)[-1].lower()


def _validate_output_snapshot(
    task: object,
    *,
    task_id: str,
    parent_id: str,
    script_sha256: str,
    script_source: str,
    parameters_sha256: str,
    context: str,
) -> None:
    if _clearml_id(getattr(task, "id", ""), context) != task_id:
        raise RuntimeError(f"{context} identity drifted")
    if _task_parent(task) != parent_id:
        raise RuntimeError(f"{context} parent drifted")
    if _status(task, context=context) != "in_progress":
        raise RuntimeError(f"{context} is not writable in_progress")
    observed_script_sha256 = _script_sha256(
        task,
        entry_point="clearml_formal_candidate_selector.py",
        context=context,
    )
    if observed_script_sha256 != script_sha256:
        raise RuntimeError(f"{context} script drifted")
    observed_source = getattr(
        getattr(getattr(task, "data", None), "script", None), "diff", None
    )
    if type(observed_source) is not str or observed_source != script_source:
        raise RuntimeError(f"{context} script bytes drifted")
    if _content_sha256(_parameters(task, context=context)) != parameters_sha256:
        raise RuntimeError(f"{context} parameters drifted")


def _wait_for_completed(
    dependencies: Sequence[tuple[str, object]],
    *,
    deadline: float,
    poll_seconds: float,
    monotonic_clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    while True:
        pending = False
        for context, task in dependencies:
            _reload(task, context=context)
            status = _status(task, context=context)
            if status == "completed":
                continue
            if status in FAILED_STATUSES:
                raise RuntimeError(f"{context} ended as {status!r}")
            if status not in WAITING_STATUSES:
                raise RuntimeError(f"{context} has unexpected status {status!r}")
            pending = True
        if not pending:
            return
        if monotonic_clock() >= deadline:
            raise TimeoutError("timed out waiting for formal candidate dependencies")
        sleeper(poll_seconds)


def _require_ap(value: object, *, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise ValueError(f"{context} must be finite and within [0, 100]")
    return result


def _metric_summary(
    runs: Mapping[tuple[int, str], Mapping[str, float]], metric_key: str
) -> dict[str, object]:
    ordered = [
        (delay, condition, runs[(delay, condition)][metric_key])
        for delay in DELAYS_MS
        for condition in CONDITIONS
    ]
    clean = runs[(0, "Full")][metric_key]
    full_300 = runs[(300, "Full")][metric_key]
    worst_delay, worst_condition, worst_value = min(ordered, key=lambda item: item[2])
    return {
        "clean_0ms": clean,
        "mean_12": math.fsum(item[2] for item in ordered) / len(ordered),
        "worst_12": {
            "value": worst_value,
            "delay_ms": worst_delay,
            "condition": worst_condition,
        },
        "full_mean": math.fsum(runs[(delay, "Full")][metric_key] for delay in DELAYS_MS)
        / len(DELAYS_MS),
        "l_fail_mean": math.fsum(
            runs[(delay, "L-Fail")][metric_key] for delay in DELAYS_MS
        )
        / len(DELAYS_MS),
        "c_fail_mean": math.fsum(
            runs[(delay, "C-Fail")][metric_key] for delay in DELAYS_MS
        )
        / len(DELAYS_MS),
        "full_300": full_300,
        "pdr": None if clean == 0.0 else 100.0 * (clean - full_300) / clean,
    }


def _parse_metrics(
    value: Mapping[str, object], *, subject: str, checkpoint_sha256: str
) -> dict[tuple[int, str], dict[str, float]]:
    _require_exact_keys(
        value,
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
        context=f"{subject} metrics",
    )
    if value.get("complete") is not True:
        raise ValueError(f"{subject} metrics complete mismatch")
    expected = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": len(DELAYS_MS) * len(CONDITIONS),
        "baseline": subject,
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
    }
    for key, expected_value in expected.items():
        if not _exact_json_equal(value.get(key), expected_value):
            raise ValueError(f"{subject} metrics {key} mismatch")
    if type(value.get("checkpoint")) is not str or not value["checkpoint"]:
        raise ValueError(f"{subject} metrics checkpoint path is invalid")
    raw_runs = value.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 12:
        raise ValueError(f"{subject} metrics run count mismatch")
    result: dict[tuple[int, str], dict[str, float]] = {}
    expected_pairs = (
        (delay, condition) for delay in DELAYS_MS for condition in CONDITIONS
    )
    for index, (run, pair) in enumerate(zip(raw_runs, expected_pairs, strict=True), 1):
        if not isinstance(run, Mapping):
            raise ValueError(f"{subject} metrics run {index} is not an object")
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
            context=f"{subject} metrics run {index}",
        )
        delay, condition = pair
        condition_id = f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        if not all(
            (
                _exact_json_equal(run.get("delay_ms"), delay),
                _exact_json_equal(run.get("condition"), condition),
                _exact_json_equal(run.get("condition_id"), condition_id),
            )
        ):
            raise ValueError(f"{subject} metrics run {index} order mismatch")
        expected_counts = {
            "sample_count": SAMPLE_COUNT,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        }
        for key, expected_value in expected_counts.items():
            if not _exact_json_equal(run.get(key), expected_value):
                raise ValueError(f"{subject} metrics run {index} {key} mismatch")
        _sha256(run.get("prediction_sha256"), f"{subject} prediction")
        _sha256(run.get("prediction_content_sha256"), f"{subject} prediction content")
        if type(run.get("predictions")) is not str or not run["predictions"]:
            raise ValueError(f"{subject} metrics run {index} predictions path invalid")
        metrics = run.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{subject} metrics run {index} has no metrics object")
        for count_key, expected_count in COUNT_METRICS.items():
            raw_count = metrics.get(count_key)
            if (
                type(raw_count) not in {int, float}
                or not math.isfinite(float(raw_count))
                or not float(raw_count).is_integer()
                or int(raw_count) != expected_count
            ):
                raise ValueError(f"{subject} metrics run {index} {count_key} mismatch")
        result[pair] = {
            key: _require_ap(
                metrics.get(key), context=f"{subject} run {index} metric {key}"
            )
            for key in AP_METRIC_KEYS
        }
    return result


def _validate_source_provenance_binding(
    value: Mapping[str, object], *, context: str
) -> dict[str, object]:
    expected_keys = {
        "training_provenance_task_id",
        "training_provenance_seal_sha256",
        "source_revision_equivalence",
        "source_revision_equivalence_seal_sha256",
        "source_revision_subject_map",
        "source_revision_subject_map_seal_sha256",
        "evaluation_source_revision_tree_sha256",
        "evaluation_source_revision",
    }
    _require_exact_keys(value, expected_keys, context=context)
    provenance_task_id = _clearml_id(
        value.get("training_provenance_task_id"),
        f"{context} training provenance task",
    )
    if provenance_task_id != TRAINING_PROVENANCE_TASK_ID:
        raise ValueError(f"{context} training_provenance_task_id mismatch")
    provenance_seal = _sha256(
        value.get("training_provenance_seal_sha256"),
        f"{context} training provenance seal",
    )
    equivalence = value.get("source_revision_equivalence")
    if not isinstance(equivalence, Mapping):
        raise ValueError(f"{context} source_revision_equivalence is invalid")
    equivalence_seal = _require_seal(
        equivalence, context=f"{context} source revision equivalence"
    )
    root_equivalence_seal = _sha256(
        value.get("source_revision_equivalence_seal_sha256"),
        f"{context} source revision equivalence seal",
    )
    if (
        equivalence_seal != root_equivalence_seal
        or equivalence_seal != SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
    ):
        raise ValueError(f"{context} source revision equivalence mismatch")
    subject_map = value.get("source_revision_subject_map")
    if not isinstance(subject_map, Mapping):
        raise ValueError(f"{context} source_revision_subject_map is invalid")
    subject_map_seal = _require_seal(
        subject_map, context=f"{context} source revision subject map"
    )
    root_subject_map_seal = _sha256(
        value.get("source_revision_subject_map_seal_sha256"),
        f"{context} source revision subject map seal",
    )
    if (
        subject_map_seal != root_subject_map_seal
        or subject_map_seal != SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
    ):
        raise ValueError(f"{context} source revision subject map mismatch")
    evaluation_tree = _sha256(
        value.get("evaluation_source_revision_tree_sha256"),
        f"{context} evaluation source revision tree",
    )
    if evaluation_tree != SOURCE_TREE_SHA256:
        raise ValueError(f"{context} evaluation source revision tree mismatch")
    evaluation_revision = value.get("evaluation_source_revision")
    if not _exact_json_equal(
        evaluation_revision, SOURCE_REVISION_CERTIFICATE_BY_TREE[SOURCE_TREE_SHA256]
    ):
        raise ValueError(f"{context} evaluation source revision mismatch")
    return {
        "training_provenance_task_id": provenance_task_id,
        "training_provenance_seal_sha256": provenance_seal,
        "source_revision_equivalence": dict(equivalence),
        "source_revision_equivalence_seal_sha256": equivalence_seal,
        "source_revision_subject_map": dict(subject_map),
        "source_revision_subject_map_seal_sha256": subject_map_seal,
        "evaluation_source_revision_tree_sha256": evaluation_tree,
        "evaluation_source_revision": dict(evaluation_revision),
    }


def _validate_leaderboard(
    payload: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], str, str, str, dict[str, object], bool]:
    leaderboard = dict(payload)
    seal = _require_seal(leaderboard, context="formal leaderboard")
    seed_fields = {"training_seed", "training_overlay_protocol_seed"}
    present_seed_fields = set(leaderboard) & seed_fields
    if present_seed_fields not in (set(), seed_fields):
        raise ValueError("formal leaderboard seed metadata is partial")
    seeded = present_seed_fields == seed_fields
    expected_keys = {
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
    } | (seed_fields if seeded else set())
    _require_exact_keys(leaderboard, expected_keys, context="formal leaderboard")
    expected = {
        "schema_version": 3,
        "leaderboard_type": "resilient_v2x_formal_1337_leaderboard",
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "ground_truth_count": GROUND_TRUTH_COUNT,
        "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count_per_subject": 12,
        "training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
        "watcher_task_id": WATCHER_TASK_ID,
        "subject_order": list(SUBJECT_ORDER),
        "subject_count": len(SUBJECT_ORDER),
        "baseline_subjects": list(CANONICAL_LEADERBOARD_BASELINE_SUBJECTS),
        "baseline_count": len(CANONICAL_LEADERBOARD_BASELINE_SUBJECTS),
        "metric_keys": list(AP_METRIC_KEYS),
    }
    for key, expected_value in expected.items():
        if not _exact_json_equal(leaderboard.get(key), expected_value):
            raise ValueError(f"formal leaderboard {key} mismatch")
    if seeded and (
        not _exact_json_equal(leaderboard.get("training_seed"), TRAINING_SEED)
        or not _exact_json_equal(
            leaderboard.get("training_overlay_protocol_seed"), TRAINING_SEED
        )
    ):
        raise ValueError("formal leaderboard seed mismatch")
    source_binding = _validate_source_provenance_binding(
        {
            key: leaderboard[key]
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
    manifest_seal = _sha256(
        leaderboard.get("training_manifest_seal_sha256"), "training manifest seal"
    )
    plan_seal = _sha256(
        leaderboard.get("evaluation_plan_seal_sha256"), "evaluation plan seal"
    )
    if not isinstance(leaderboard.get("leadership"), Mapping):
        raise ValueError("formal leaderboard leadership is invalid")
    raw_results = leaderboard.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(SUBJECT_ORDER):
        raise ValueError("formal leaderboard result count mismatch")
    results: dict[str, dict[str, object]] = {}
    task_ids: set[str] = set()
    model_ids: set[str] = set()
    evaluation_ids: set[str] = set()
    for index, (raw, subject) in enumerate(
        zip(raw_results, SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(raw, Mapping):
            raise ValueError(f"formal leaderboard result {index} is invalid")
        _require_exact_keys(
            raw,
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
        if not all(
            (
                _exact_json_equal(raw.get("index"), index),
                _exact_json_equal(raw.get("subject"), subject),
                _exact_json_equal(raw.get("kind"), SUBJECT_KIND[subject]),
            )
        ):
            raise ValueError(f"formal leaderboard result {index} identity mismatch")
        training_task_id = _clearml_id(
            raw.get("training_task_id"), f"{subject} training"
        )
        model_id = _clearml_id(raw.get("training_model_id"), f"{subject} model")
        evaluation_task_id = _clearml_id(
            raw.get("evaluation_task_id"), f"{subject} evaluation"
        )
        if (
            training_task_id in task_ids
            or model_id in model_ids
            or evaluation_task_id in evaluation_ids
        ):
            raise ValueError("formal leaderboard task/model IDs must be unique")
        task_ids.add(training_task_id)
        model_ids.add(model_id)
        evaluation_ids.add(evaluation_task_id)
        checkpoint_sha = _sha256(
            raw.get("training_checkpoint_sha256"), f"{subject} checkpoint"
        )
        source_tree = _sha256(
            raw.get("source_revision_tree_sha256"), f"{subject} source revision tree"
        )
        expected_source_tree = str(_expected_source(subject)["tree_sha256"])
        if source_tree != expected_source_tree:
            raise ValueError(
                f"formal leaderboard {subject} source_revision_tree_sha256 mismatch"
            )
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(AP_METRIC_KEYS):
            raise ValueError(f"formal leaderboard {subject} metric summaries mismatch")
        results[subject] = {
            "index": index,
            "subject": subject,
            "kind": SUBJECT_KIND[subject],
            "training_task_id": training_task_id,
            "training_model_id": model_id,
            "training_checkpoint_sha256": checkpoint_sha,
            "source_revision_tree_sha256": source_tree,
            "evaluation_task_id": evaluation_task_id,
            "metrics": metrics,
        }
    return results, seal, manifest_seal, plan_seal, source_binding, seeded


def _validate_audit(
    payload: Mapping[str, object],
    *,
    audit_task_id: str,
    leaderboard_task_id: str,
    leaderboard_seal: str,
    leaderboard_manifest_seal: str,
    leaderboard_plan_seal: str,
    leaderboard_results: Mapping[str, Mapping[str, object]],
    leaderboard_source_binding: Mapping[str, object],
    leaderboard_seeded: bool,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    audit = dict(payload)
    _require_seal(audit, context="formal comparability audit")
    if audit.get("passed") is not True:
        raise ValueError("formal comparability audit passed mismatch")
    expected = {
        "schema_version": 3,
        "document_type": "resilient_v2x_formal_1337_comparability_audit",
        "passed": True,
        "audit_scope": "protocol_and_clearml_metadata_comparability",
        "audit_task_id": audit_task_id,
        "training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
        "training_provenance_task_id": TRAINING_PROVENANCE_TASK_ID,
        "watcher_task_id": WATCHER_TASK_ID,
        "leaderboard_task_id": leaderboard_task_id,
        "protocol_id": PROTOCOL_ID,
        "evaluation_source_revision_tree_sha256": leaderboard_source_binding[
            "evaluation_source_revision_tree_sha256"
        ],
        "evaluation_source_revision": leaderboard_source_binding[
            "evaluation_source_revision"
        ],
        "source_revision_equivalence": leaderboard_source_binding[
            "source_revision_equivalence"
        ],
        "source_revision_equivalence_seal_sha256": leaderboard_source_binding[
            "source_revision_equivalence_seal_sha256"
        ],
        "source_revision_subject_map": leaderboard_source_binding[
            "source_revision_subject_map"
        ],
        "source_revision_subject_map_seal_sha256": leaderboard_source_binding[
            "source_revision_subject_map_seal_sha256"
        ],
        "source_revision_counts": {
            SOURCE_TREE_SHA256: len(SUBJECT_ORDER) - len(NEW_SOURCE_SUBJECTS),
            NEW_SOURCE_TREE_SHA256: len(NEW_SOURCE_SUBJECTS),
        },
        "training_dataset_id": TRAINING_DATASET_ID,
        "training_seed": TRAINING_SEED,
        "training_seed_evidence": "all_26_live_run_contracts",
        "legacy_unseeded_training_manifest": not leaderboard_seeded,
        "legacy_source_c_seed_schema": (
            None if leaderboard_seeded else "explicit_run_contract_seed_only"
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
        "checkpoint_policy": CHECKPOINT_POLICY,
        "sample_count": SAMPLE_COUNT,
        "ground_truth_count": GROUND_TRUTH_COUNT,
        "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count_per_subject": 12,
        "subject_order": list(SUBJECT_ORDER),
        "subject_count": len(SUBJECT_ORDER),
        "total_evaluation_run_count": len(SUBJECT_ORDER) * 12,
        "metric_keys": list(AP_METRIC_KEYS),
        "final_model_verification_level": "clearml_metadata_contract_only",
        "checkpoint_bytes_verifier": MODEL_BYTES_VERIFIER,
        "checkpoint_bytes_verification_dependency": (
            "separate_collect_clearml_formal_models_byte_audit"
        ),
        "training_manifest_seal_sha256": leaderboard_manifest_seal,
        "evaluation_plan_seal_sha256": leaderboard_plan_seal,
        "training_provenance_seal_sha256": leaderboard_source_binding[
            "training_provenance_seal_sha256"
        ],
        "leaderboard_seal_sha256": leaderboard_seal,
    }
    _require_exact_keys(
        audit,
        set(expected)
        | {
            "checkpoint_bytes_sha256_recomputed",
            "training_progress_seal_sha256",
            "training_summary_seal_sha256",
            "training_tasks",
            "evaluation_tasks",
            "seal_sha256",
        },
        context="formal comparability audit",
    )
    for key, expected_value in expected.items():
        if not _exact_json_equal(audit.get(key), expected_value):
            raise ValueError(f"formal comparability audit {key} mismatch")
    if audit.get("legacy_unseeded_training_manifest") is not (not leaderboard_seeded):
        raise ValueError(
            "formal comparability audit legacy_unseeded_training_manifest mismatch"
        )
    if audit.get("checkpoint_bytes_sha256_recomputed") is not False:
        raise ValueError(
            "formal comparability audit checkpoint byte verification mismatch"
        )
    _sha256(
        audit.get("training_progress_seal_sha256"),
        "formal comparability audit training_progress_seal_sha256",
    )
    _sha256(
        audit.get("training_summary_seal_sha256"),
        "formal comparability audit training_summary_seal_sha256",
    )
    training_records = audit.get("training_tasks")
    evaluation_records = audit.get("evaluation_tasks")
    if not isinstance(training_records, list) or len(training_records) != len(
        SUBJECT_ORDER
    ):
        raise ValueError("formal comparability audit training task count mismatch")
    if not isinstance(evaluation_records, list) or len(evaluation_records) != len(
        SUBJECT_ORDER
    ):
        raise ValueError("formal comparability audit evaluation task count mismatch")
    result: dict[str, dict[str, object]] = {}
    for index, (training, evaluation, subject) in enumerate(
        zip(training_records, evaluation_records, SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(training, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError(f"formal comparability audit record {index} is invalid")
        leaderboard = leaderboard_results[subject]
        source = _expected_source(subject)
        expected_training = {
            "index": index,
            "subject": subject,
            "training_task_id": leaderboard["training_task_id"],
            "model_id": leaderboard["training_model_id"],
            "checkpoint_sha256": leaderboard["training_checkpoint_sha256"],
            "training_seed": TRAINING_SEED,
            "source_revision_tree_sha256": source["tree_sha256"],
            "source_dataset_id": source["dataset_id"],
            "source_archive_name": source["archive_name"],
            "source_archive_bytes": source["archive_bytes"],
            "source_archive_sha256": source["archive_sha256"],
            "training_dataset_id": TRAINING_DATASET_ID,
            "parent_controller_task_id": _expected_training_parent_task_id(subject),
            "script_sha256": _expected_training_script_sha256(subject),
            "run_contract_seed_fields": ["seed"],
            "gpus": 4,
            "precision": "FP32",
            "max_epochs": 50,
            "val_interval": 10,
            "final_model_verification_level": "clearml_metadata_contract_only",
            "checkpoint_bytes_sha256_recomputed": False,
            "checkpoint_bytes_verifier": MODEL_BYTES_VERIFIER,
        }
        training_sha_fields = {
            "run_contract_sha256",
            "final_checkpoint_contract_content_sha256",
            "common_teacher_initialization_audit_sha256",
            "raw_authority_sha256",
        }
        _require_exact_keys(
            training,
            set(expected_training)
            | training_sha_fields
            | {"execution_queue_id", "last_worker"},
            context=f"formal audit training {subject}",
        )
        for key, expected_value in expected_training.items():
            if not _exact_json_equal(training.get(key), expected_value):
                raise ValueError(f"formal audit training {subject} {key} mismatch")
        if training.get("checkpoint_bytes_sha256_recomputed") is not False:
            raise ValueError(
                f"formal audit training {subject} checkpoint byte verification mismatch"
            )
        for key in training_sha_fields:
            _sha256(training.get(key), f"formal audit training {subject} {key}")
        _clearml_id(
            training.get("execution_queue_id"),
            f"formal audit training {subject} execution queue",
        )
        if type(training.get("last_worker")) is not str or not training.get(
            "last_worker"
        ):
            raise ValueError(f"formal audit training {subject} last_worker is invalid")
        expected_evaluation = {
            "index": index,
            "subject": subject,
            "evaluation_task_id": leaderboard["evaluation_task_id"],
            "training_task_id": leaderboard["training_task_id"],
            "model_id": leaderboard["training_model_id"],
            "checkpoint_sha256": leaderboard["training_checkpoint_sha256"],
            "source_revision_tree_sha256": SOURCE_TREE_SHA256,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_archive_name": SOURCE_ARCHIVE_NAME,
            "source_archive_bytes": SOURCE_ARCHIVE_BYTES,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "script_sha256": EVALUATION_SCRIPT_SHA256,
            "run_count": 12,
            "sample_count_per_run": SAMPLE_COUNT,
            "ground_truth_count_per_run": GROUND_TRUTH_COUNT,
            "unsupported_sample_count_per_run": UNSUPPORTED_SAMPLE_COUNT,
            "metric_keys": list(AP_METRIC_KEYS),
        }
        _require_exact_keys(
            evaluation,
            set(expected_evaluation)
            | {
                "planned_queue",
                "execution_queue_id",
                "execution_queue_name",
                "last_worker",
                "metrics_sha256",
            },
            context=f"formal audit evaluation {subject}",
        )
        for key, expected_value in expected_evaluation.items():
            if not _exact_json_equal(evaluation.get(key), expected_value):
                raise ValueError(f"formal audit evaluation {subject} {key} mismatch")
        _clearml_id(
            evaluation.get("execution_queue_id"),
            f"formal audit evaluation {subject} execution queue",
        )
        for key in ("planned_queue", "execution_queue_name", "last_worker"):
            if type(evaluation.get(key)) is not str or not evaluation.get(key):
                raise ValueError(f"formal audit evaluation {subject} {key} is invalid")
        if evaluation.get(
            "planned_queue"
        ) not in SUPPORTED_EVALUATION_QUEUES or evaluation.get(
            "execution_queue_name"
        ) != evaluation.get("planned_queue"):
            raise ValueError(
                f"formal audit evaluation {subject} execution queue mismatch"
            )
        result[subject] = {
            "evaluation_task_id": leaderboard["evaluation_task_id"],
            "training_task_id": leaderboard["training_task_id"],
            "model_id": leaderboard["training_model_id"],
            "checkpoint_sha256": leaderboard["training_checkpoint_sha256"],
            "source_revision_tree_sha256": training["source_revision_tree_sha256"],
            "source_dataset_id": training["source_dataset_id"],
            "source_archive_name": training["source_archive_name"],
            "source_archive_bytes": training["source_archive_bytes"],
            "source_archive_sha256": training["source_archive_sha256"],
            "training_script_sha256": training["script_sha256"],
            "run_contract_content_sha256": training["run_contract_sha256"],
            "initialization_audit_content_sha256": training[
                "common_teacher_initialization_audit_sha256"
            ],
            "final_checkpoint_contract_content_sha256": training[
                "final_checkpoint_contract_content_sha256"
            ],
            "script_sha256": EVALUATION_SCRIPT_SHA256,
            "planned_queue": evaluation["planned_queue"],
            "metrics_sha256": _sha256(
                evaluation.get("metrics_sha256"), f"{subject} audited metrics"
            ),
        }
    chain = {
        "training_provenance_task_id": TRAINING_PROVENANCE_TASK_ID,
        "training_progress_seal_sha256": str(audit["training_progress_seal_sha256"]),
        "training_provenance_seal_sha256": str(
            audit["training_provenance_seal_sha256"]
        ),
        "source_revision_equivalence": leaderboard_source_binding[
            "source_revision_equivalence"
        ],
        "source_revision_equivalence_seal_sha256": leaderboard_source_binding[
            "source_revision_equivalence_seal_sha256"
        ],
        "source_revision_subject_map": leaderboard_source_binding[
            "source_revision_subject_map"
        ],
        "source_revision_subject_map_seal_sha256": leaderboard_source_binding[
            "source_revision_subject_map_seal_sha256"
        ],
        "evaluation_source_revision_tree_sha256": leaderboard_source_binding[
            "evaluation_source_revision_tree_sha256"
        ],
        "evaluation_source_revision": leaderboard_source_binding[
            "evaluation_source_revision"
        ],
        "training_script_equivalence": expected["training_script_equivalence"],
        "evaluation_script_sha256": EVALUATION_SCRIPT_SHA256,
    }
    return result, chain


def _validate_live_evaluation_provenance(
    task: object,
    *,
    subject: str,
    leaderboard: Mapping[str, object],
) -> None:
    if _task_parent(task) != TRAINING_CONTROLLER_TASK_ID:
        raise RuntimeError(f"{subject} evaluation task parent mismatch")
    observed_script_sha = _script_sha256(
        task, entry_point=TRAINING_ENTRY_POINT, context=f"{subject} evaluation"
    )
    if observed_script_sha != EVALUATION_SCRIPT_SHA256:
        raise RuntimeError(f"{subject} evaluation task script SHA-256 mismatch")
    _require_parameters(
        _parameters(task, context=f"{subject} evaluation"),
        {
            "Args/stage": "baseline_validate",
            "Args/controlled_baseline": subject,
            "Args/controlled_baseline_task_id": leaderboard["training_task_id"],
            "Args/controlled_baseline_model_id": leaderboard["training_model_id"],
            "Args/controlled_baseline_checkpoint_sha256": leaderboard[
                "training_checkpoint_sha256"
            ],
            "Args/predecessor_task_id": leaderboard["training_task_id"],
            "Args/source_dataset_id": SOURCE_DATASET_ID,
            "Args/source_archive_name": SOURCE_ARCHIVE_NAME,
            "Args/source_archive_bytes": SOURCE_ARCHIVE_BYTES,
            "Args/source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "Args/training_dataset_id": TRAINING_DATASET_ID,
            "Args/gpus": 4,
            "Args/max_epochs": 50,
            "Args/amp": False,
        },
        context=f"{subject} evaluation",
    )
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError(f"{subject} evaluation cannot expose input models")
    models = getter()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"{subject} evaluation model mapping is invalid")
    inputs = models.get("input")
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise RuntimeError(f"{subject} evaluation input models are invalid")
    input_ids = [str(getattr(model, "id", "") or "") for model in inputs]
    if input_ids != [leaderboard["training_model_id"]]:
        raise RuntimeError(f"{subject} evaluation input model mismatch")


def _load_actual_runs(
    *,
    task_class: object,
    leaderboard_results: Mapping[str, Mapping[str, object]],
    audit_records: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[tuple[int, str], dict[str, float]]]:
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise RuntimeError("ClearML Task class cannot resolve evaluation tasks")
    result: dict[str, dict[tuple[int, str], dict[str, float]]] = {}
    for subject in SUBJECT_ORDER:
        leaderboard = leaderboard_results[subject]
        evaluation_task_id = str(leaderboard["evaluation_task_id"])
        task = getter(task_id=evaluation_task_id)
        if _clearml_id(getattr(task, "id", ""), f"{subject} evaluation") != (
            evaluation_task_id
        ):
            raise RuntimeError(f"{subject} evaluation task identity mismatch")
        _reload(task, context=f"{subject} evaluation")
        if _status(task, context=f"{subject} evaluation") != "completed":
            raise RuntimeError(f"{subject} evaluation task is not completed")
        _validate_live_evaluation_provenance(
            task, subject=subject, leaderboard=leaderboard
        )
        metrics = _artifact_mapping(
            task, METRICS_ARTIFACT, context=f"{subject} evaluation"
        )
        if _content_sha256(metrics) != audit_records[subject]["metrics_sha256"]:
            raise RuntimeError(f"{subject} evaluation metrics SHA-256 mismatch")
        runs = _parse_metrics(
            metrics,
            subject=subject,
            checkpoint_sha256=str(leaderboard["training_checkpoint_sha256"]),
        )
        expected_summaries = {
            metric_key: _metric_summary(runs, metric_key)
            for metric_key in AP_METRIC_KEYS
        }
        if not _exact_json_equal(leaderboard["metrics"], expected_summaries):
            raise ValueError(f"formal leaderboard {subject} metric summary drifted")
        result[subject] = runs
    return result


def _validate_watcher_plan(
    payload: Mapping[str, object],
    *,
    leaderboard_plan_seal: str,
    leaderboard_results: Mapping[str, Mapping[str, object]],
    audit_records: Mapping[str, Mapping[str, object]],
    source_binding: Mapping[str, object],
) -> str:
    plan = dict(payload)
    seal = _require_seal(plan, context="formal evaluation plan")
    _require_exact_keys(
        plan,
        {
            "schema_version",
            "plan_type",
            "training_controller_task_id",
            "training_provenance_task_id",
            "training_provenance_seal_sha256",
            "source_revision_equivalence",
            "source_revision_equivalence_seal_sha256",
            "source_revision_subject_map",
            "source_revision_subject_map_seal_sha256",
            "evaluation_source_revision_tree_sha256",
            "evaluation_source_revision",
            "evaluation_template_task_id",
            "protocol_id",
            "sample_count",
            "delays_ms",
            "conditions",
            "run_count",
            "subject_order",
            "entries",
            "seal_sha256",
        },
        context="formal evaluation plan",
    )
    expected = {
        "schema_version": 2,
        "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
        "training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
        "training_provenance_task_id": TRAINING_PROVENANCE_TASK_ID,
        "training_provenance_seal_sha256": source_binding[
            "training_provenance_seal_sha256"
        ],
        "source_revision_equivalence": source_binding["source_revision_equivalence"],
        "source_revision_equivalence_seal_sha256": source_binding[
            "source_revision_equivalence_seal_sha256"
        ],
        "source_revision_subject_map": source_binding["source_revision_subject_map"],
        "source_revision_subject_map_seal_sha256": source_binding[
            "source_revision_subject_map_seal_sha256"
        ],
        "evaluation_source_revision_tree_sha256": source_binding[
            "evaluation_source_revision_tree_sha256"
        ],
        "evaluation_source_revision": source_binding["evaluation_source_revision"],
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count": 12,
        "subject_order": list(SUBJECT_ORDER),
    }
    for key, expected_value in expected.items():
        if not _exact_json_equal(plan.get(key), expected_value):
            raise ValueError(f"formal evaluation plan {key} mismatch")
    _clearml_id(plan.get("evaluation_template_task_id"), "evaluation template")
    rows = plan.get("entries")
    if not isinstance(rows, list) or len(rows) != len(SUBJECT_ORDER):
        raise ValueError("formal evaluation plan entry count mismatch")
    for index, (raw, subject) in enumerate(
        zip(rows, SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(raw, Mapping):
            raise ValueError(f"formal evaluation plan entry {index} is invalid")
        _require_exact_keys(
            raw,
            {"subject", "evaluation_task_id", "queue"},
            context=f"formal evaluation plan entry {subject}",
        )
        expected_row = {
            "subject": subject,
            "evaluation_task_id": leaderboard_results[subject]["evaluation_task_id"],
            "queue": audit_records[subject]["planned_queue"],
        }
        if not _exact_json_equal(dict(raw), expected_row):
            raise ValueError(f"formal evaluation plan entry {subject} mismatch")
    if seal != leaderboard_plan_seal:
        raise ValueError("formal evaluation plan seal differs from leaderboard")
    return seal


def _validate_watcher_task(
    task: object,
    *,
    source_binding: Mapping[str, object],
) -> None:
    if _clearml_id(getattr(task, "id", ""), "watcher") != WATCHER_TASK_ID:
        raise RuntimeError("watcher identity mismatch")
    if _task_parent(task) != TRAINING_PROVENANCE_TASK_ID:
        raise RuntimeError("watcher parent mismatch")
    if (
        _script_sha256(task, entry_point=WATCHER_ENTRY_POINT, context="watcher")
        != WATCHER_SCRIPT_SHA256
    ):
        raise RuntimeError("watcher script SHA-256 mismatch")
    parameters = _parameters(task, context="watcher")
    _require_parameters(
        parameters,
        {
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "Args/training_provenance_task_id": TRAINING_PROVENANCE_TASK_ID,
            "Args/expected_eval_script_sha256": EVALUATION_SCRIPT_SHA256,
            "Args/expected_training_dataset_id": TRAINING_DATASET_ID,
            "Args/expected_source_dataset_id": SOURCE_DATASET_ID,
            "Args/expected_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
        },
        context="watcher",
    )
    if source_binding["training_provenance_task_id"] != TRAINING_PROVENANCE_TASK_ID:
        raise RuntimeError("watcher provenance binding mismatch")


def _validated_training_identity(
    task: object,
    *,
    subject: str,
    leaderboard: Mapping[str, object],
    audit: Mapping[str, object],
) -> dict[str, object]:
    context = f"{subject} training"
    task_id = str(leaderboard["training_task_id"])
    if _clearml_id(getattr(task, "id", ""), context) != task_id:
        raise RuntimeError(f"{context} task identity mismatch")
    _reload(task, context=context)
    if _status(task, context=context) != "completed":
        raise RuntimeError(f"{context} task is not completed")
    if _task_parent(task) != _expected_training_parent_task_id(subject):
        raise RuntimeError(f"{context} parent mismatch")
    script_sha = _script_sha256(task, entry_point=TRAINING_ENTRY_POINT, context=context)
    if script_sha != audit["training_script_sha256"]:
        raise RuntimeError(f"{context} script SHA-256 mismatch")
    expected_source = _expected_source(subject)
    parameters = _parameters(task, context=context)
    _require_parameters(
        parameters,
        {
            "Args/stage": "all",
            "Args/experiment_from_task": subject,
            "Args/source_dataset_id": expected_source["dataset_id"],
            "Args/source_archive_name": expected_source["archive_name"],
            "Args/source_archive_bytes": expected_source["archive_bytes"],
            "Args/source_archive_sha256": expected_source["archive_sha256"],
            "Args/training_dataset_id": TRAINING_DATASET_ID,
            "Args/gpus": 4,
            "Args/max_epochs": 50,
            "Args/amp": False,
        },
        context=context,
    )
    names = set(_artifact_names(task, context=context))
    required_names = {
        RUN_CONTRACT_ARTIFACT,
        INITIALIZATION_AUDIT_ARTIFACT,
        FINAL_CHECKPOINT_CONTRACT_ARTIFACT,
    }
    if not required_names <= names:
        raise RuntimeError(f"{context} required artifact inventory drifted")
    records = _artifact_metadata_records(task, context=context)
    contract = _artifact_mapping(task, RUN_CONTRACT_ARTIFACT, context=context)
    if _content_sha256(contract) != audit["run_contract_content_sha256"]:
        raise RuntimeError(f"{context} run contract content drifted")
    expected_contract = {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": subject,
        "source_dataset_id": expected_source["dataset_id"],
        "training_dataset_id": TRAINING_DATASET_ID,
        "gpus": 4,
        "global_batch_size": 8,
        "max_epochs": 50,
        "val_interval": 10,
        "precision": "FP32",
        "amp": False,
        "condition_evaluation": False,
    }
    for key, expected_value in expected_contract.items():
        if not _exact_json_equal(contract.get(key), expected_value):
            raise RuntimeError(f"{context} run contract {key} drifted")
    seed_fields = [key for key in ("seed", "training_seed") if key in contract]
    if len(seed_fields) != 1 or contract.get(seed_fields[0]) != TRAINING_SEED:
        raise RuntimeError(f"{context} run contract seed drifted")
    source = contract.get("source_archive")
    if not isinstance(source, Mapping) or not _exact_json_equal(
        dict(source),
        {
            "name": expected_source["archive_name"],
            "size_bytes": expected_source["archive_bytes"],
            "sha256": expected_source["archive_sha256"],
        },
    ):
        raise RuntimeError(f"{context} source archive drifted")
    config = contract.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context} config identity is missing")
    config_path = _safe_config_path(config.get("declared"), context=context)
    config_sha = _sha256(config.get("config_sha256"), f"{context} config")
    resolved_config_sha = config.get("resolved_config_sha256")
    if (
        resolved_config_sha is not None
        and _sha256(resolved_config_sha, f"{context} resolved config") != config_sha
    ):
        raise RuntimeError(f"{context} resolved config SHA-256 drifted")
    teacher = contract.get("teacher")
    if not isinstance(teacher, Mapping) or any(
        teacher.get(key) != expected_value
        for key, expected_value in {
            "task_id": TEACHER_TASK_ID,
            "model_id": TEACHER_MODEL_ID,
            "sha256": TEACHER_CHECKPOINT_SHA256,
        }.items()
    ):
        raise RuntimeError(f"{context} unified teacher drifted")
    initialization = _artifact_mapping(
        task, INITIALIZATION_AUDIT_ARTIFACT, context=context
    )
    if _content_sha256(initialization) != audit["initialization_audit_content_sha256"]:
        raise RuntimeError(f"{context} initialization audit content drifted")
    final = _artifact_mapping(task, FINAL_CHECKPOINT_CONTRACT_ARTIFACT, context=context)
    if _content_sha256(final) != audit["final_checkpoint_contract_content_sha256"]:
        raise RuntimeError(f"{context} final checkpoint contract content drifted")
    checkpoint_size = final.get("size_bytes")
    if type(checkpoint_size) is not int or checkpoint_size <= 0:
        raise RuntimeError(f"{context} checkpoint size is invalid")
    expected_final = {
        "model_id": leaderboard["training_model_id"],
        "filename": "epoch_50.pth",
        "sha256": leaderboard["training_checkpoint_sha256"],
    }
    for key, expected_value in expected_final.items():
        if final.get(key) != expected_value:
            raise RuntimeError(f"{context} final checkpoint {key} drifted")
    model_name = final.get("name")
    model_url = final.get("url")
    if type(model_name) is not str or not model_name:
        raise RuntimeError(f"{context} final model name is invalid")
    if type(model_url) is not str or not model_url:
        raise RuntimeError(f"{context} final model URL is invalid")
    return {
        "label": subject,
        "config_path": config_path,
        "config_sha256": config_sha,
        "source_dataset_id": expected_source["dataset_id"],
        "source_revision_sha256": expected_source["tree_sha256"],
        "source_archive_name": expected_source["archive_name"],
        "source_archive_bytes": expected_source["archive_bytes"],
        "source_archive_sha256": expected_source["archive_sha256"],
        "training_dataset_id": TRAINING_DATASET_ID,
        "teacher_task_id": TEACHER_TASK_ID,
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "training_task_id": task_id,
        "model_id": leaderboard["training_model_id"],
        "model_name": model_name,
        "checkpoint_filename": "epoch_50.pth",
        "checkpoint_sha256": leaderboard["training_checkpoint_sha256"],
        "checkpoint_size_bytes": checkpoint_size,
        "training_script_sha256": script_sha,
        "run_contract_artifact_sha256": records[RUN_CONTRACT_ARTIFACT]["hash"],
        "initialization_audit_artifact_sha256": records[INITIALIZATION_AUDIT_ARTIFACT][
            "hash"
        ],
        "final_checkpoint_contract_artifact_sha256": records[
            FINAL_CHECKPOINT_CONTRACT_ARTIFACT
        ]["hash"],
    }


def _validated_evaluation_identity(
    task: object,
    *,
    subject: str,
    leaderboard: Mapping[str, object],
    audit: Mapping[str, object],
    training_identity: Mapping[str, object],
    expected_runs: Mapping[tuple[int, str], Mapping[str, float]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    context = f"{subject} evaluation"
    evaluation_task_id = str(leaderboard["evaluation_task_id"])
    if _clearml_id(getattr(task, "id", ""), context) != evaluation_task_id:
        raise RuntimeError(f"{context} task identity mismatch")
    _reload(task, context=context)
    if _status(task, context=context) != "completed":
        raise RuntimeError(f"{context} task is not completed")
    _validate_live_evaluation_provenance(task, subject=subject, leaderboard=leaderboard)
    expected_artifacts = {
        RUN_CONTRACT_ARTIFACT,
        EVALUATION_PLAN_ARTIFACT,
        METRICS_ARTIFACT,
        PREDICTION_EVIDENCE_ARTIFACT,
    }
    if set(_artifact_names(task, context=context)) != expected_artifacts:
        raise RuntimeError(f"{context} artifact inventory drifted")
    records = _artifact_metadata_records(task, context=context)
    contract = _artifact_mapping(task, RUN_CONTRACT_ARTIFACT, context=context)
    contract_expected = {
        "schema_version": 1,
        "mode": "baseline_validate",
        "task_id": evaluation_task_id,
        "baseline": subject,
        "baseline_task_id": training_identity["training_task_id"],
        "predecessor_task_id": training_identity["training_task_id"],
        "training_dataset_id": TRAINING_DATASET_ID,
        "protocol_id": PROTOCOL_ID,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_run_count": 12,
        "expected_manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "expected_overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "expected_sample_ids_sha256": SAMPLE_IDS_SHA256,
    }
    for key, expected_value in contract_expected.items():
        if contract.get(key) != expected_value:
            raise RuntimeError(f"{context} run contract {key} drifted")
    checkpoint = contract.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or any(
        checkpoint.get(key) != expected_value
        for key, expected_value in {
            "task_id": training_identity["training_task_id"],
            "model_id": training_identity["model_id"],
            "sha256": training_identity["checkpoint_sha256"],
            "size_bytes": training_identity["checkpoint_size_bytes"],
        }.items()
    ):
        raise RuntimeError(f"{context} checkpoint binding drifted")
    plan = _artifact_mapping(task, EVALUATION_PLAN_ARTIFACT, context=context)
    plan_expected = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_evaluation",
        "protocol_id": PROTOCOL_ID,
        "baseline": subject,
        "checkpoint_sha256": training_identity["checkpoint_sha256"],
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
    }
    for key, expected_value in plan_expected.items():
        if not _exact_json_equal(plan.get(key), expected_value):
            raise RuntimeError(f"{context} evaluation plan {key} drifted")
    plan_runs = plan.get("runs")
    expected_conditions = [
        (delay, condition) for delay in DELAYS_MS for condition in CONDITIONS
    ]
    if not isinstance(plan_runs, list) or len(plan_runs) != 12:
        raise RuntimeError(f"{context} evaluation plan runs drifted")
    for index, (raw, (delay, condition)) in enumerate(
        zip(plan_runs, expected_conditions, strict=True)
    ):
        condition_id = f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        if not isinstance(raw, Mapping) or any(
            raw.get(key) != expected_value
            for key, expected_value in {
                "condition_id": condition_id,
                "delay_ms": delay,
                "condition": condition,
                "agent_scope": "E+R",
                "duration_ticks": 1,
            }.items()
        ):
            raise RuntimeError(f"{context} evaluation plan run {index} drifted")
    metrics = _artifact_mapping(task, METRICS_ARTIFACT, context=context)
    if _content_sha256(metrics) != audit["metrics_sha256"]:
        raise RuntimeError(f"{context} metrics content drifted")
    parsed_runs = _parse_metrics(
        metrics,
        subject=subject,
        checkpoint_sha256=str(training_identity["checkpoint_sha256"]),
    )
    if not _exact_json_equal(parsed_runs, expected_runs):
        raise RuntimeError(f"{context} raw metrics changed after gate evaluation")
    rows = [
        {
            "condition_id": (
                f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
            ),
            "delay_ms": delay,
            "condition": condition,
            "agent_scope": "E+R",
            "metrics": {
                key: parsed_runs[(delay, condition)][key] for key in AP_METRIC_KEYS
            },
        }
        for delay, condition in expected_conditions
    ]
    identity = {
        **dict(training_identity),
        "evaluation_task_id": evaluation_task_id,
        "evaluation_script_sha256": audit["script_sha256"],
        "evaluation_run_contract_artifact_sha256": records[RUN_CONTRACT_ARTIFACT][
            "hash"
        ],
        "evaluation_plan_artifact_sha256": records[EVALUATION_PLAN_ARTIFACT]["hash"],
        "metrics_artifact_sha256": records[METRICS_ARTIFACT]["hash"],
        "prediction_evidence_artifact_sha256": records[PREDICTION_EVIDENCE_ARTIFACT][
            "hash"
        ],
        "prediction_evidence_archive_sha256": records[PREDICTION_EVIDENCE_ARTIFACT][
            "hash"
        ],
    }
    return identity, rows


def build_formal_inputs(
    *,
    producer_task_id: str,
    producer_script_sha256: str,
    watcher_plan: Mapping[str, object],
    watcher_plan_seal: str,
    watcher_artifact_sha256: str,
    leaderboard_task_id: str,
    leaderboard: Mapping[str, object],
    leaderboard_seal: str,
    leaderboard_artifact_sha256: str,
    audit_task_id: str,
    audit: Mapping[str, object],
    audit_seal: str,
    audit_artifact_sha256: str,
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(entries) != len(FORMAL_INPUT_SUBJECTS):
        raise ValueError("formal selector input entry count mismatch")
    return _sealed(
        {
            "schema_version": 2,
            "document_type": FORMAL_INPUTS_DOCUMENT_TYPE,
            "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
            "subject_order": list(FORMAL_INPUT_SUBJECTS),
            "training_seed": TRAINING_SEED,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "authority_bindings": {
                "producer_task_id": producer_task_id,
                "producer_script_sha256": producer_script_sha256,
                "producer_artifact_name": FORMAL_INPUTS_ARTIFACT,
                "training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
                "training_provenance_task_id": TRAINING_PROVENANCE_TASK_ID,
                "watcher_task_id": WATCHER_TASK_ID,
                "watcher_artifact_name": FORMAL_EVALUATION_PLAN_ARTIFACT,
                "watcher_artifact_sha256": watcher_artifact_sha256,
                "watcher_plan_seal_sha256": watcher_plan_seal,
                "watcher_plan_content_sha256": _content_sha256(watcher_plan),
                "leaderboard_task_id": leaderboard_task_id,
                "leaderboard_artifact_name": LEADERBOARD_ARTIFACT,
                "leaderboard_artifact_sha256": leaderboard_artifact_sha256,
                "leaderboard_seal_sha256": leaderboard_seal,
                "leaderboard_content_sha256": _content_sha256(leaderboard),
                "audit_task_id": audit_task_id,
                "audit_artifact_name": AUDIT_ARTIFACT,
                "audit_artifact_sha256": audit_artifact_sha256,
                "audit_seal_sha256": audit_seal,
                "audit_content_sha256": _content_sha256(audit),
            },
            "entries": [dict(item) for item in entries],
        }
    )


def _collect_formal_inputs(
    *,
    task_class: object,
    watcher_task: object,
    leaderboard_task: object,
    audit_task: object,
    producer_task_id: str,
    producer_script_sha256: str,
    leaderboard_task_id: str,
    leaderboard: Mapping[str, object],
    leaderboard_seal: str,
    leaderboard_plan_seal: str,
    leaderboard_results: Mapping[str, Mapping[str, object]],
    source_binding: Mapping[str, object],
    audit_task_id: str,
    audit: Mapping[str, object],
    audit_seal: str,
    audit_records: Mapping[str, Mapping[str, object]],
    runs_by_subject: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
) -> dict[str, object]:
    _reload(watcher_task, context="watcher")
    if _status(watcher_task, context="watcher") != "completed":
        raise RuntimeError("watcher is not completed")
    _validate_watcher_task(watcher_task, source_binding=source_binding)
    if _artifact_names(watcher_task, context="watcher") != (
        FORMAL_EVALUATION_PLAN_ARTIFACT,
    ):
        raise RuntimeError("watcher artifact inventory drifted")
    watcher_plan = _artifact_mapping(
        watcher_task,
        FORMAL_EVALUATION_PLAN_ARTIFACT,
        context="watcher",
    )
    watcher_plan_seal = _validate_watcher_plan(
        watcher_plan,
        leaderboard_plan_seal=leaderboard_plan_seal,
        leaderboard_results=leaderboard_results,
        audit_records=audit_records,
        source_binding=source_binding,
    )
    watcher_records = _artifact_metadata_records(watcher_task, context="watcher")
    leaderboard_records = _artifact_metadata_records(
        leaderboard_task, context="leaderboard"
    )
    audit_artifact_records = _artifact_metadata_records(
        audit_task, context="comparability audit"
    )
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise RuntimeError("ClearML Task class cannot resolve formal input tasks")
    entries: list[dict[str, object]] = []
    for subject in FORMAL_INPUT_SUBJECTS:
        leaderboard_record = leaderboard_results[subject]
        audit_record = audit_records[subject]
        training_task = getter(task_id=str(leaderboard_record["training_task_id"]))
        training_identity = _validated_training_identity(
            training_task,
            subject=subject,
            leaderboard=leaderboard_record,
            audit=audit_record,
        )
        evaluation_task = getter(task_id=str(leaderboard_record["evaluation_task_id"]))
        identity, rows = _validated_evaluation_identity(
            evaluation_task,
            subject=subject,
            leaderboard=leaderboard_record,
            audit=audit_record,
            training_identity=training_identity,
            expected_runs=runs_by_subject[subject],
        )
        entries.append(
            {
                "subject": subject,
                "identity_binding": identity,
                "runs": rows,
            }
        )
    return build_formal_inputs(
        producer_task_id=producer_task_id,
        producer_script_sha256=producer_script_sha256,
        watcher_plan=watcher_plan,
        watcher_plan_seal=watcher_plan_seal,
        watcher_artifact_sha256=watcher_records[FORMAL_EVALUATION_PLAN_ARTIFACT][
            "hash"
        ],
        leaderboard_task_id=leaderboard_task_id,
        leaderboard=leaderboard,
        leaderboard_seal=leaderboard_seal,
        leaderboard_artifact_sha256=leaderboard_records[LEADERBOARD_ARTIFACT]["hash"],
        audit_task_id=audit_task_id,
        audit=audit,
        audit_seal=audit_seal,
        audit_artifact_sha256=audit_artifact_records[AUDIT_ARTIFACT]["hash"],
        entries=entries,
    )


def _final_snapshot_recheck(
    *,
    task_class: object,
    audit_task: object,
    leaderboard_task: object,
    audit_task_id: str,
    leaderboard_task_id: str,
    initial_audit_content_sha256: str,
    initial_leaderboard_content_sha256: str,
    leaderboard_results: Mapping[str, Mapping[str, object]],
    audit_records: Mapping[str, Mapping[str, object]],
) -> None:
    _reload(leaderboard_task, context="leaderboard")
    if _status(leaderboard_task, context="leaderboard") != "completed":
        raise RuntimeError("leaderboard changed status before selector publication")
    _reload(audit_task, context="comparability audit")
    if _status(audit_task, context="comparability audit") != "completed":
        raise RuntimeError(
            "comparability audit changed status before selector publication"
        )
    if _artifact_names(leaderboard_task, context="leaderboard") != (
        LEADERBOARD_ARTIFACT,
    ):
        raise RuntimeError("leaderboard artifact inventory drifted")
    if _artifact_names(audit_task, context="comparability audit") != (AUDIT_ARTIFACT,):
        raise RuntimeError("comparability audit artifact inventory drifted")
    _validate_producer(
        leaderboard_task,
        task_id=leaderboard_task_id,
        parent_id=WATCHER_TASK_ID,
        entry_point=LEADERBOARD_ENTRY_POINT,
        script_sha256=LEADERBOARD_SCRIPT_SHA256,
        parameters={
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "Args/watcher_task_id": WATCHER_TASK_ID,
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
        },
        context="leaderboard",
    )
    _validate_producer(
        audit_task,
        task_id=audit_task_id,
        parent_id=leaderboard_task_id,
        entry_point=AUDIT_ENTRY_POINT,
        script_sha256=AUDIT_SCRIPT_SHA256,
        parameters={
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "Args/training_provenance_task_id": TRAINING_PROVENANCE_TASK_ID,
            "Args/watcher_task_id": WATCHER_TASK_ID,
            "Args/leaderboard_task_id": leaderboard_task_id,
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
        },
        context="comparability audit",
    )
    current_leaderboard = _artifact_mapping(
        leaderboard_task, LEADERBOARD_ARTIFACT, context="leaderboard"
    )
    _require_seal(current_leaderboard, context="formal leaderboard")
    if _content_sha256(current_leaderboard) != initial_leaderboard_content_sha256:
        raise RuntimeError("leaderboard artifact changed before selector publication")
    current_audit = _artifact_mapping(
        audit_task, AUDIT_ARTIFACT, context="comparability audit"
    )
    _require_seal(current_audit, context="formal comparability audit")
    if _content_sha256(current_audit) != initial_audit_content_sha256:
        raise RuntimeError(
            "comparability audit artifact changed before selector publication"
        )
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise RuntimeError("ClearML Task class cannot resolve evaluation tasks")
    for subject in SUBJECT_ORDER:
        evaluation_task_id = str(leaderboard_results[subject]["evaluation_task_id"])
        task = getter(task_id=evaluation_task_id)
        if _clearml_id(getattr(task, "id", ""), f"{subject} evaluation") != (
            evaluation_task_id
        ):
            raise RuntimeError(f"{subject} evaluation task identity changed")
        _reload(task, context=f"{subject} evaluation")
        if _status(task, context=f"{subject} evaluation") != "completed":
            raise RuntimeError(f"{subject} evaluation task changed status")
        _validate_live_evaluation_provenance(
            task, subject=subject, leaderboard=leaderboard_results[subject]
        )
        metrics = _artifact_mapping(
            task, METRICS_ARTIFACT, context=f"{subject} evaluation"
        )
        if _content_sha256(metrics) != audit_records[subject]["metrics_sha256"]:
            raise RuntimeError(f"{subject} evaluation metrics changed")


def _candidate_result(
    subject: str,
    runs_by_subject: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
) -> dict[str, object]:
    return sota_gate.evaluate_candidate(
        subject,
        runs_by_subject[subject],
        {baseline: runs_by_subject[baseline] for baseline in BASELINE_SUBJECTS},
    )


def _margin_sort_key(
    result: Mapping[str, object],
) -> tuple[int, float, float, float, int]:
    return sota_gate.ranking_sort_key(result)


def build_selection(
    *,
    audit_task_id: str,
    leaderboard_task_id: str,
    audit_seal: str,
    leaderboard_seal: str,
    audit_chain: Mapping[str, object],
    runs_by_subject: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
) -> dict[str, object]:
    candidate_results = [
        _candidate_result(subject, runs_by_subject) for subject in CANDIDATE_ORDER
    ]
    performance_ranking = sorted(candidate_results, key=_margin_sort_key)
    passing = [item for item in performance_ranking if item["gate_passed"] is True]
    selected = str(passing[0]["subject"]) if passing else None
    status = "selected" if selected is not None else "architecture_revision_required"
    performance_rank = {
        str(item["subject"]): rank
        for rank, item in enumerate(performance_ranking, start=1)
    }
    ordered_results = []
    for item in candidate_results:
        enriched = dict(item)
        subject = str(item["subject"])
        enriched["performance_rank"] = performance_rank[subject]
        enriched["rank"] = performance_rank[subject]
        ordered_results.append(enriched)
    return _sealed(
        {
            "schema_version": 4,
            "document_type": "resilient_v2x_formal_candidate_selection",
            "status": status,
            "selection_claim": (
                "single_seed_1337_bev70_sota_gate_winner" if selected else None
            ),
            "selection_stage": "single_seed_1337_sota_selection",
            "selection_is_final": selected is not None,
            "claim_scope": "metrics_and_clearml_metadata_only",
            "live_snapshot_recheck": {
                "audit_and_leaderboard": (
                    "status_parent_script_parameters_and_artifact"
                ),
                "evaluation_tasks": (
                    "status_parent_script_parameters_input_model_and_metrics"
                ),
                "training_tasks": "sealed_comparability_audit_snapshot_only",
            },
            "model_checkpoint_byte_verification_complete": False,
            "requires_checkpoint_byte_audit": True,
            "architecture_revision_required": selected is None,
            "recommended_action": (
                "select_single_seed_winner" if selected else "revise_architecture"
            ),
            "audit_task_id": audit_task_id,
            "leaderboard_task_id": leaderboard_task_id,
            "training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "training_provenance_task_id": audit_chain["training_provenance_task_id"],
            "watcher_task_id": WATCHER_TASK_ID,
            "training_progress_seal_sha256": audit_chain[
                "training_progress_seal_sha256"
            ],
            "training_provenance_seal_sha256": audit_chain[
                "training_provenance_seal_sha256"
            ],
            "source_revision_equivalence": audit_chain["source_revision_equivalence"],
            "source_revision_equivalence_seal_sha256": audit_chain[
                "source_revision_equivalence_seal_sha256"
            ],
            "source_revision_subject_map": audit_chain["source_revision_subject_map"],
            "source_revision_subject_map_seal_sha256": audit_chain[
                "source_revision_subject_map_seal_sha256"
            ],
            "evaluation_source_revision_tree_sha256": audit_chain[
                "evaluation_source_revision_tree_sha256"
            ],
            "evaluation_source_revision": audit_chain["evaluation_source_revision"],
            "training_script_equivalence": audit_chain["training_script_equivalence"],
            "evaluation_script_sha256": audit_chain["evaluation_script_sha256"],
            "audit_seal_sha256": audit_seal,
            "leaderboard_seal_sha256": leaderboard_seal,
            "protocol_id": PROTOCOL_ID,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(DELAYS_MS),
            "conditions": list(CONDITIONS),
            "run_count_per_subject": 12,
            "selection_metric": LEADERSHIP_METRIC,
            "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
            "baseline_subjects": list(BASELINE_SUBJECTS),
            "baseline_count": len(BASELINE_SUBJECTS),
            "candidate_subjects": list(CANDIDATE_ORDER),
            "candidate_count": len(CANDIDATE_ORDER),
            "gate": {
                "metric": LEADERSHIP_METRIC,
                "full_0ms": "candidate >= best controlled baseline - 0.5",
                "mean_12": "candidate > best controlled baseline",
                "worst_12": "candidate > best controlled baseline",
                "per_condition_lead_required": False,
            },
            "ranking_key": list(RANKING_KEY),
            "ranking_direction": (
                "gate_passed_first_then_descending_margins_then_ascending_fixed_order"
            ),
            "performance_ranked_candidates": [
                str(item["subject"]) for item in performance_ranking
            ],
            "ranked_candidates_semantics": "performance_rank",
            "ranked_candidates": [str(item["subject"]) for item in performance_ranking],
            "selected_candidate": selected,
            "selected_candidate_gate_passed": selected is not None,
            "candidate_results": ordered_results,
        }
    )


def _publish(
    task: object,
    payload: Mapping[str, object],
    *,
    formal_inputs: Mapping[str, object] | None = None,
    task_id: str,
    parent_id: str,
    script_sha256: str,
    script_source: str,
    parameters_sha256: str,
    validate_bindings: Callable[[], None],
) -> None:
    try:
        expected_selection = json.loads(_canonical_json(payload))
        expected_inputs = (
            json.loads(_canonical_json(formal_inputs))
            if formal_inputs is not None
            else None
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("selector outputs are not canonical JSON") from error
    if not isinstance(expected_selection, Mapping):
        raise RuntimeError("formal candidate selection is not a JSON object")
    if expected_inputs is not None and not isinstance(expected_inputs, Mapping):
        raise RuntimeError("formal selector inputs are not a JSON object")
    expected_outputs: dict[str, Mapping[str, object]] = {
        SELECTION_ARTIFACT: expected_selection,
    }
    upload_order = [SELECTION_ARTIFACT]
    if expected_inputs is not None:
        expected_outputs[FORMAL_INPUTS_ARTIFACT] = expected_inputs
        upload_order.insert(0, FORMAL_INPUTS_ARTIFACT)
    expected_inventory = tuple(sorted(expected_outputs))

    def validate_observed(name: str, *, publication: str) -> None:
        observed = _artifact_mapping(task, name, context="selector task")
        expected = expected_outputs[name]
        if observed != expected or _content_sha256(observed) != _content_sha256(
            expected
        ):
            if name == SELECTION_ARTIFACT:
                message = (
                    "existing formal candidate selection artifact drifted"
                    if publication == "existing"
                    else "published formal candidate selection drifted"
                )
                raise RuntimeError(message)
            raise RuntimeError(f"{publication} formal selector inputs artifact drifted")

    validate_bindings()
    _reload(task, context="selector task")
    _validate_output_snapshot(
        task,
        task_id=task_id,
        parent_id=parent_id,
        script_sha256=script_sha256,
        script_source=script_source,
        parameters_sha256=parameters_sha256,
        context="selector task",
    )
    names = _artifact_names(task, context="selector task")
    allowed_inventories = {(), expected_inventory}
    if expected_inputs is not None:
        allowed_inventories.add((FORMAL_INPUTS_ARTIFACT,))
    if names not in allowed_inventories:
        raise RuntimeError("selector task artifact inventory drifted")
    for name in names:
        validate_observed(name, publication="existing")
    if names == expected_inventory:
        validate_bindings()
        validate_bindings()
        _reload(task, context="selector task final readback")
        _validate_output_snapshot(
            task,
            task_id=task_id,
            parent_id=parent_id,
            script_sha256=script_sha256,
            script_source=script_source,
            parameters_sha256=parameters_sha256,
            context="selector task final readback",
        )
        if _artifact_names(task, context="selector task") != expected_inventory:
            raise RuntimeError("selector task artifact inventory drifted")
        for name in expected_inventory:
            validate_observed(name, publication="existing")
        return
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader):
        raise RuntimeError("selector task cannot publish artifacts")
    existing_names = set(names)
    for name in upload_order:
        if name in existing_names:
            continue
        validate_bindings()
        if (
            uploader(
                name,
                artifact_object=dict(expected_outputs[name]),
                wait_on_upload=True,
            )
            is not True
        ):
            raise RuntimeError(f"failed to publish selector artifact {name!r}")
    flusher = getattr(task, "flush", None)
    if not callable(flusher):
        raise RuntimeError("selector task cannot flush formal candidate selection")
    flushed = flusher(wait_for_uploads=True)
    if flushed is not None and flushed is not True:
        raise RuntimeError("failed to flush formal candidate selection")
    validate_bindings()
    _reload(task, context="selector task publication readback")
    _validate_output_snapshot(
        task,
        task_id=task_id,
        parent_id=parent_id,
        script_sha256=script_sha256,
        script_source=script_source,
        parameters_sha256=parameters_sha256,
        context="selector task publication readback",
    )
    if _artifact_names(task, context="selector task") != expected_inventory:
        raise RuntimeError("selector artifacts are absent after publication")
    for name in expected_inventory:
        validate_observed(name, publication="published")
    validate_bindings()
    validate_bindings()
    _reload(task, context="selector task final readback")
    _validate_output_snapshot(
        task,
        task_id=task_id,
        parent_id=parent_id,
        script_sha256=script_sha256,
        script_source=script_source,
        parameters_sha256=parameters_sha256,
        context="selector task final readback",
    )
    if _artifact_names(task, context="selector task") != expected_inventory:
        raise RuntimeError("selector task artifact inventory drifted")
    for name in expected_inventory:
        validate_observed(name, publication="published")


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    sealed_successor_pins = _sealed_successor_pins(args)
    formal_inputs_artifact = getattr(args, "formal_inputs_artifact", None)
    if formal_inputs_artifact not in {None, FORMAL_INPUTS_ARTIFACT}:
        raise ValueError("formal selector inputs artifact name is not canonical")
    emit_formal_inputs = formal_inputs_artifact == FORMAL_INPUTS_ARTIFACT
    if (
        type(args.poll_seconds) not in {int, float}
        or not math.isfinite(float(args.poll_seconds))
        or args.poll_seconds <= 0
        or type(args.timeout_hours) not in {int, float}
        or not math.isfinite(float(args.timeout_hours))
        or args.timeout_hours <= 0
    ):
        raise ValueError("poll interval and timeout must be finite and positive")
    audit_task_id = _clearml_id(args.audit_task_id, "comparability audit")
    leaderboard_task_id = _clearml_id(args.leaderboard_task_id, "leaderboard")
    if audit_task_id == leaderboard_task_id:
        raise ValueError("audit and leaderboard task IDs must differ")
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise RuntimeError("ClearML Task class cannot resolve dependencies")
    audit_task = getter(task_id=audit_task_id)
    leaderboard_task = getter(task_id=leaderboard_task_id)
    watcher_task = getter(task_id=WATCHER_TASK_ID) if emit_formal_inputs else None
    if (
        _clearml_id(getattr(audit_task, "id", ""), "comparability audit")
        != audit_task_id
    ):
        raise RuntimeError("comparability audit identity mismatch")
    if _clearml_id(getattr(leaderboard_task, "id", ""), "leaderboard") != (
        leaderboard_task_id
    ):
        raise RuntimeError("leaderboard identity mismatch")
    if (
        watcher_task is not None
        and _clearml_id(getattr(watcher_task, "id", ""), "watcher") != WATCHER_TASK_ID
    ):
        raise RuntimeError("watcher identity mismatch")
    if output_task is None:
        current = getattr(task_class, "current_task", None)
        output_task = current() if callable(current) else None
    if output_task is None:
        raise RuntimeError("formal candidate selector requires a current ClearML task")
    output_task_id = _clearml_id(getattr(output_task, "id", ""), "selector task")
    if output_task_id in {
        TRAINING_CONTROLLER_TASK_ID,
        TRAINING_PROVENANCE_TASK_ID,
        WATCHER_TASK_ID,
        audit_task_id,
        leaderboard_task_id,
    }:
        raise RuntimeError("formal candidate selector must be a separate task")
    _reload(output_task, context="selector task")
    output_script_source = _runtime_source()
    output_script_sha256 = hashlib.sha256(
        output_script_source.encode("utf-8")
    ).hexdigest()
    output_parameters_sha256 = _content_sha256(
        _parameters(output_task, context="selector task")
    )
    _validate_output_snapshot(
        output_task,
        task_id=output_task_id,
        parent_id=audit_task_id,
        script_sha256=output_script_sha256,
        script_source=output_script_source,
        parameters_sha256=output_parameters_sha256,
        context="selector task",
    )
    deadline = monotonic_clock() + float(args.timeout_hours) * 3600.0
    _wait_for_completed(
        (("comparability audit", audit_task), ("leaderboard", leaderboard_task)),
        deadline=deadline,
        poll_seconds=float(args.poll_seconds),
        monotonic_clock=monotonic_clock,
        sleeper=sleeper,
    )
    if _artifact_names(leaderboard_task, context="leaderboard") != (
        LEADERBOARD_ARTIFACT,
    ):
        raise RuntimeError("leaderboard artifact inventory drifted")
    if _artifact_names(audit_task, context="comparability audit") != (AUDIT_ARTIFACT,):
        raise RuntimeError("comparability audit artifact inventory drifted")
    leaderboard = _artifact_mapping(
        leaderboard_task, LEADERBOARD_ARTIFACT, context="leaderboard"
    )
    leaderboard_content_sha256 = _content_sha256(leaderboard)
    _validate_producer(
        leaderboard_task,
        task_id=leaderboard_task_id,
        parent_id=WATCHER_TASK_ID,
        entry_point=LEADERBOARD_ENTRY_POINT,
        script_sha256=LEADERBOARD_SCRIPT_SHA256,
        parameters={
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "Args/watcher_task_id": WATCHER_TASK_ID,
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
            **sealed_successor_pins,
        },
        context="leaderboard",
    )
    _validate_producer(
        audit_task,
        task_id=audit_task_id,
        parent_id=leaderboard_task_id,
        entry_point=AUDIT_ENTRY_POINT,
        script_sha256=AUDIT_SCRIPT_SHA256,
        parameters={
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "Args/training_provenance_task_id": TRAINING_PROVENANCE_TASK_ID,
            "Args/watcher_task_id": WATCHER_TASK_ID,
            "Args/leaderboard_task_id": leaderboard_task_id,
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
            **sealed_successor_pins,
        },
        context="comparability audit",
    )
    (
        leaderboard_results,
        leaderboard_seal,
        leaderboard_manifest_seal,
        leaderboard_plan_seal,
        leaderboard_source_binding,
        leaderboard_seeded,
    ) = _validate_leaderboard(leaderboard)
    if sealed_successor_pins and leaderboard_plan_seal != sealed_successor_pins[
        "Args/evaluation_plan_amendment_revised_plan_seal_sha256"
    ]:
        raise RuntimeError("amended evaluation plan seal pin mismatch")
    audit = _artifact_mapping(audit_task, AUDIT_ARTIFACT, context="comparability audit")
    audit_content_sha256 = _content_sha256(audit)
    audit_seal = _require_seal(audit, context="formal comparability audit")
    audit_records, audit_chain = _validate_audit(
        audit,
        audit_task_id=audit_task_id,
        leaderboard_task_id=leaderboard_task_id,
        leaderboard_seal=leaderboard_seal,
        leaderboard_manifest_seal=leaderboard_manifest_seal,
        leaderboard_plan_seal=leaderboard_plan_seal,
        leaderboard_results=leaderboard_results,
        leaderboard_source_binding=leaderboard_source_binding,
        leaderboard_seeded=leaderboard_seeded,
    )
    occupied_task_ids = {
        TRAINING_CONTROLLER_TASK_ID,
        TRAINING_PROVENANCE_TASK_ID,
        WATCHER_TASK_ID,
        audit_task_id,
        leaderboard_task_id,
    }
    occupied_task_ids.update(
        str(item["training_task_id"]) for item in leaderboard_results.values()
    )
    occupied_task_ids.update(
        str(item["evaluation_task_id"]) for item in leaderboard_results.values()
    )
    if output_task_id in occupied_task_ids:
        raise RuntimeError("formal candidate selector task aliases a formal dependency")
    runs_by_subject = _load_actual_runs(
        task_class=task_class,
        leaderboard_results=leaderboard_results,
        audit_records=audit_records,
    )
    formal_inputs = (
        _collect_formal_inputs(
            task_class=task_class,
            watcher_task=watcher_task,
            leaderboard_task=leaderboard_task,
            audit_task=audit_task,
            producer_task_id=output_task_id,
            producer_script_sha256=output_script_sha256,
            leaderboard_task_id=leaderboard_task_id,
            leaderboard=leaderboard,
            leaderboard_seal=leaderboard_seal,
            leaderboard_plan_seal=leaderboard_plan_seal,
            leaderboard_results=leaderboard_results,
            source_binding=leaderboard_source_binding,
            audit_task_id=audit_task_id,
            audit=audit,
            audit_seal=audit_seal,
            audit_records=audit_records,
            runs_by_subject=runs_by_subject,
        )
        if watcher_task is not None
        else None
    )
    payload = build_selection(
        audit_task_id=audit_task_id,
        leaderboard_task_id=leaderboard_task_id,
        audit_seal=audit_seal,
        leaderboard_seal=leaderboard_seal,
        audit_chain=audit_chain,
        runs_by_subject=runs_by_subject,
    )

    def validate_bindings() -> None:
        _final_snapshot_recheck(
            task_class=task_class,
            audit_task=audit_task,
            leaderboard_task=leaderboard_task,
            audit_task_id=audit_task_id,
            leaderboard_task_id=leaderboard_task_id,
            initial_audit_content_sha256=audit_content_sha256,
            initial_leaderboard_content_sha256=leaderboard_content_sha256,
            leaderboard_results=leaderboard_results,
            audit_records=audit_records,
        )
        if watcher_task is not None:
            current_formal_inputs = _collect_formal_inputs(
                task_class=task_class,
                watcher_task=watcher_task,
                leaderboard_task=leaderboard_task,
                audit_task=audit_task,
                producer_task_id=output_task_id,
                producer_script_sha256=output_script_sha256,
                leaderboard_task_id=leaderboard_task_id,
                leaderboard=leaderboard,
                leaderboard_seal=leaderboard_seal,
                leaderboard_plan_seal=leaderboard_plan_seal,
                leaderboard_results=leaderboard_results,
                source_binding=leaderboard_source_binding,
                audit_task_id=audit_task_id,
                audit=audit,
                audit_seal=audit_seal,
                audit_records=audit_records,
                runs_by_subject=runs_by_subject,
            )
            if not _exact_json_equal(current_formal_inputs, formal_inputs):
                raise RuntimeError("formal selector inputs changed before publication")

    validate_bindings()
    tagger = getattr(output_task, "set_tags", None)
    if callable(tagger):
        tagger(
            [
                "ResilientV2X-suite",
                "formal-candidate-selection",
                *(("formal-final-selector-inputs",) if emit_formal_inputs else ()),
                PROTOCOL_ID,
                "cpu-controller",
            ]
        )
    _publish(
        output_task,
        payload,
        formal_inputs=formal_inputs,
        task_id=output_task_id,
        parent_id=audit_task_id,
        script_sha256=output_script_sha256,
        script_source=output_script_source,
        parameters_sha256=output_parameters_sha256,
        validate_bindings=validate_bindings,
    )
    return payload


def main() -> int:
    args = _parser().parse_args()
    task = Task.init(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X post-formal candidate selector",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
    )
    run(args, output_task=task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AUDIT_ARTIFACT",
    "BASELINE_SUBJECTS",
    "CANDIDATE_ORDER",
    "DEFAULT_AUDIT_TASK_ID",
    "DEFAULT_LEADERBOARD_TASK_ID",
    "FORMAL_INPUTS_ARTIFACT",
    "FORMAL_INPUTS_DOCUMENT_TYPE",
    "FORMAL_INPUT_SUBJECTS",
    "SELECTION_ARTIFACT",
    "build_formal_inputs",
    "build_selection",
    "run",
)
