#!/usr/bin/env python3
"""Export sealed paper evidence for the controlled DAIR-CAUSAL-1337 grid.

The command is read-only by default.  It reads the completed formal W/L/A/S
chain, the six selected evaluation tasks, and locally byte-verified final
checkpoints.  A local JSON file is created only with ``--write`` and the exact
write token.  This module never mutates ClearML state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import requests

try:
    from allegroai import Task
except ImportError:
    try:
        from clearml import Task
    except ImportError:  # pragma: no cover - pure unit-test environments
        Task = None  # type: ignore[assignment]

try:
    from clearml.backend_api.session import Session
except ImportError:  # pragma: no cover - pure unit-test environments
    Session = None  # type: ignore[assignment]


PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
TRAINING_SEED = 20_250_218
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
SAMPLE_COUNT = 1_337
GROUND_TRUTH_COUNT = 11_330
UNSUPPORTED_SAMPLE_COUNT = 0
SAMPLE_IDS_SHA256 = "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
OVERLAY_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
DELAYS_MS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
RUN_COUNT = 12
AGENT_SCOPE = "E+R"
CHECKPOINT_POLICY = "epoch_50_final_only"

TEACHER_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
TEACHER_MODEL_ID = "d962f6bae8474260b54e170a7a5f0418"
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)

BASELINE_SUBJECTS = ("ffnet", "coformernet", "v2x_vit", "cobevt", "bevfusion")
FORMAL_LEADERBOARD_BASELINE_SUBJECTS = (
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
DISPLAY_NAMES = {
    "ffnet": "FFNet (controlled adaptation)",
    "coformernet": "CoFormerNet (controlled adaptation)",
    "v2x_vit": "V2X-ViT (controlled adaptation)",
    "cobevt": "CoBEVT (controlled adaptation)",
    "bevfusion": "BEVFusion (controlled adaptation)",
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

AP_METRIC_KEYS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)
COUNT_METRICS = {
    "resilient_v2x/sample_count": SAMPLE_COUNT,
    "resilient_v2x/car_ground_truth_count": GROUND_TRUTH_COUNT,
    "resilient_v2x/unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
}
LEADERSHIP_METRIC = "resilient_v2x/car_bev_ap_r40_0.70"
FULL_0MS_MAX_DEFICIT = 0.5
P1_ZERO_SHOT_SUBJECT = (
    "dair_improvement_reliability_gated_residual_zero_shot_p0_epoch50_final"
)
FINAL_SELECTION_DOCUMENT_TYPE = "resilient_v2x_final_single_seed_selection"
SELECTED_METHOD_IDENTITY_DOCUMENT_TYPE = (
    "resilient_v2x_sealed_selected_method_identity"
)
SELECTED_METHOD_IDENTITY_ARTIFACT = "selected_method_identity"

FORMAL_EVIDENCE_TYPE = "resilient_v2x_paper_controlled_1337_evidence"
FORMAL_EVIDENCE_ARTIFACT = "paper_controlled_1337_evidence"
WRITE_TOKEN = "WRITE_VERIFIED_PAPER_CONTROLLED_1337_EVIDENCE"
DEFAULT_MODEL_ROOT = Path("artifacts/trained_models/completed-live")
DEFAULT_OUTPUT = Path(
    "artifacts/resilient_v2x/paper-controlled-1337/formal-evidence.json"
)
FILES_SERVER_HOST = "10.100.34.118"
FILES_SERVER_PORT = 8081
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 2 * 1024 * 1024 * 1024

_CLEARML_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUBJECT = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


class PaperEvidenceExportError(RuntimeError):
    """Raised when any paper evidence binding is incomplete or inconsistent."""


@dataclass(frozen=True)
class ChainEvidence:
    watcher_task_id: str
    leaderboard_task_id: str
    audit_task_id: str
    selector_task_id: str
    plan: dict[str, object]
    leaderboard: dict[str, object]
    audit: dict[str, object]
    selector: dict[str, object]
    training_manifest: dict[str, object]
    selected_method_identity: dict[str, object] | None = None


@dataclass(frozen=True)
class CollectedSubject:
    subject: str
    display_name: str
    identity: dict[str, object]
    runs: list[dict[str, object]]


_IDENTITY_KEYS = {
    "model_name",
    "modality",
    "backbone",
    "training_task_id",
    "training_model_id",
    "evaluation_task_id",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "checkpoint_bytes_verified",
    "source_revision_tree_sha256",
    "source_dataset_id",
    "source_archive_sha256",
    "config_path",
    "config_sha256",
    "training_script_sha256",
    "teacher_task_id",
    "teacher_model_id",
    "teacher_checkpoint_sha256",
    "training_dataset_id",
    "metrics_artifact_sha256",
    "prediction_evidence_artifact_sha256",
    "prediction_evidence_archive_sha256",
}
_RUN_KEYS = {
    "condition_id",
    "delay_ms",
    "condition",
    "agent_scope",
    "sample_count",
    "ground_truth_count",
    "unsupported_sample_count",
    "sample_ids_sha256",
    "prediction_sha256",
    "prediction_content_sha256",
    "metrics",
}


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
        raise PaperEvidenceExportError(
            f"value is outside canonical JSON: {error}"
        ) from None


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _require_seal(value: Mapping[str, object], context: str) -> str:
    observed = _sha256(value.get("seal_sha256"), f"{context} seal")
    if _sealed(value)["seal_sha256"] != observed:
        raise PaperEvidenceExportError(f"{context} seal mismatch")
    return observed


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if _SHA256.fullmatch(result) is None:
        raise PaperEvidenceExportError(f"{context} must be a lowercase SHA-256")
    return result


def _task_id(value: object, context: str) -> str:
    result = str(value or "")
    if _CLEARML_ID.fullmatch(result) is None:
        raise PaperEvidenceExportError(f"{context} must be a lowercase ClearML ID")
    return result


def _subject(value: object, context: str) -> str:
    result = str(value or "")
    if _SUBJECT.fullmatch(result) is None:
        raise PaperEvidenceExportError(f"{context} is not a safe subject")
    return result


def _safe_relative(value: object, context: str) -> str:
    result = str(value or "")
    path = PurePosixPath(result)
    if not result or path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise PaperEvidenceExportError(f"{context} is not a safe relative path")
    return result


def _training_config_path(config: Mapping[str, object], subject: str) -> str:
    declared = config.get("declared")
    if isinstance(declared, str) and declared:
        return _safe_relative(declared, f"{subject} config path")
    resolved = str(config.get("declared_resolved") or "")
    trusted_root = "/workspace/resilient-v2x-5090-runtime/"
    if not resolved.startswith(trusted_root):
        raise PaperEvidenceExportError(f"{subject} config path is unavailable")
    return _safe_relative(
        resolved.removeprefix(trusted_root), f"{subject} config path"
    )


def _is_final_checkpoint_path(value: object, subject: str) -> bool:
    name = PurePosixPath(str(value or "")).name
    expected = f"{subject}_epoch_50.pth"
    return name == expected or re.fullmatch(
        rf"[0-9a-f]{{32}}\.{re.escape(expected)}", name
    ) is not None


def _positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise PaperEvidenceExportError(f"{context} must be a positive integer")
    return value


def _finite_ap(value: object, context: str) -> float:
    if type(value) not in {int, float}:
        raise PaperEvidenceExportError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise PaperEvidenceExportError(f"{context} must be finite within [0, 100]")
    return result


def _strict_json(raw: bytes, context: str) -> dict[str, object]:
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise PaperEvidenceExportError(f"{context} size is outside the safe limit")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise PaperEvidenceExportError(
                    f"{context} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperEvidenceExportError(f"{context} is not UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise PaperEvidenceExportError(f"{context} must be a JSON object")
    _canonical_json(value)
    return dict(value)


def _read_json(path: Path, context: str) -> dict[str, object]:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise PaperEvidenceExportError(f"{context} must be a regular file")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise PaperEvidenceExportError(f"{context} must be a single-link regular file")
    raw = path.read_bytes()
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise PaperEvidenceExportError(f"{context} changed while being read")
    return _strict_json(raw, context)


def _status(task: object, context: str) -> str:
    value = getattr(task, "status", None)
    if value is None:
        value = getattr(getattr(task, "data", None), "status", None)
    result = str(getattr(value, "value", value) or "").casefold()
    if result != "completed":
        raise PaperEvidenceExportError(f"{context} is not completed")
    return result


def _parent(task: object) -> str:
    value = getattr(task, "parent", None)
    if value is None:
        value = getattr(getattr(task, "data", None), "parent", None)
    return str(value or "")


def _reload(task: object) -> None:
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()


def _artifact_records(task: object, context: str) -> dict[str, dict[str, object]]:
    values = getattr(
        getattr(getattr(task, "data", None), "execution", None), "artifacts", None
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise PaperEvidenceExportError(
            f"{context} cannot expose authoritative artifact metadata"
        )
    result: dict[str, dict[str, object]] = {}
    for raw in values:
        converter = getattr(raw, "to_dict", None)
        raw = converter() if callable(converter) else raw
        if not isinstance(raw, Mapping):
            raise PaperEvidenceExportError(f"{context} has malformed artifact metadata")
        key = str(raw.get("key") or "")
        if not key or key in result:
            raise PaperEvidenceExportError(
                f"{context} has duplicate or unnamed artifact metadata"
            )
        record = dict(raw)
        _sha256(record.get("hash"), f"{context} artifact {key}")
        _positive_int(record.get("content_size"), f"{context} artifact {key} size")
        uri = urlsplit(str(record.get("uri") or ""))
        if (
            uri.scheme not in {"http", "https"}
            or uri.hostname != FILES_SERVER_HOST
            or uri.port != FILES_SERVER_PORT
            or uri.username is not None
            or uri.password is not None
        ):
            raise PaperEvidenceExportError(
                f"{context} artifact {key} has an untrusted URI"
            )
        result[key] = record
    return result


def _artifact_payload(
    task: object, name: str, *, context: str
) -> tuple[dict[str, object], dict[str, object]]:
    records = _artifact_records(task, context)
    if name not in records:
        raise PaperEvidenceExportError(f"{context} lacks artifact {name!r}")
    record = records[name]
    artifacts = getattr(task, "artifacts", None)
    artifact = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    getter = getattr(artifact, "get", None)
    if callable(getter):
        try:
            value = getter()
        except (OSError, ValueError):
            # Some ClearML deployments expose authoritative JSON previews through
            # the API while their files-server URL rejects direct SDK downloads.
            # The preview remains covered by the artifact metadata checks below.
            value = None
        if isinstance(value, Mapping):
            return dict(value), record
    preview = record.get("type_data")
    if isinstance(preview, Mapping) and type(preview.get("preview")) is str:
        try:
            value = json.loads(preview["preview"])
        except json.JSONDecodeError:
            value = None
        if isinstance(value, Mapping):
            return dict(value), record
    value = _authenticated_json_artifact(record, context=f"{context} artifact {name}")
    if isinstance(value, Mapping):
        return dict(value), record
    raise PaperEvidenceExportError(f"{context} artifact {name!r} is unreadable")


def _authenticated_json_artifact(
    record: Mapping[str, object], *, context: str
) -> object:
    """Read a protected ClearML files-server JSON artifact and verify its bytes."""
    if Session is None:
        return None
    try:
        token = str(Session().token or "")
        if not token:
            return None
        response = requests.get(
            str(record["uri"]),
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
    except (KeyError, requests.RequestException, RuntimeError, ValueError):
        return None
    payload = response.content
    if len(payload) != record["content_size"]:
        raise PaperEvidenceExportError(f"{context} content size drifted")
    if hashlib.sha256(payload).hexdigest() != record["hash"]:
        raise PaperEvidenceExportError(f"{context} content hash drifted")
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperEvidenceExportError(f"{context} is not valid JSON") from error


def _artifact_file(
    task: object, name: str, *, context: str
) -> tuple[Path, dict[str, object], bool]:
    records = _artifact_records(task, context)
    if name not in records:
        raise PaperEvidenceExportError(f"{context} lacks artifact {name!r}")
    record = records[name]
    artifacts = getattr(task, "artifacts", None)
    artifact = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    getter = getattr(artifact, "get_local_copy", None)
    raw_path: object = None
    if callable(getter):
        try:
            raw_path = getter(
                extract_archive=False, raise_on_error=True, force_download=True
            )
        except TypeError:
            try:
                raw_path = getter()
            except (OSError, ValueError):
                raw_path = None
        except (OSError, ValueError):
            raw_path = None
    path = Path(str(raw_path or ""))
    temporary = False
    if path.is_symlink() or not path.is_file():
        downloaded = _authenticated_artifact_file(
            record, context=f"{context} artifact {name}"
        )
        if downloaded is None:
            raise PaperEvidenceExportError(
                f"{context} artifact {name!r} is not downloadable"
            )
        path = downloaded
        temporary = True
    if (
        path.stat().st_size != record["content_size"]
        or _sha256_path(path) != record["hash"]
    ):
        raise PaperEvidenceExportError(
            f"{context} artifact {name!r} byte identity drifted"
        )
    return path, record, temporary


def _authenticated_artifact_file(
    record: Mapping[str, object], *, context: str
) -> Path | None:
    if Session is None:
        return None
    path: Path | None = None
    try:
        token = str(Session().token or "")
        if not token:
            return None
        response = requests.get(
            str(record["uri"]),
            headers={"Authorization": f"Bearer {token}"},
            timeout=(30, 120),
            stream=True,
        )
        response.raise_for_status()
        descriptor, raw_path = tempfile.mkstemp(prefix="paper-evidence-", suffix=".bin")
        path = Path(raw_path)
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if size != record["content_size"]:
            raise PaperEvidenceExportError(f"{context} content size drifted")
        if digest.hexdigest() != record["hash"]:
            raise PaperEvidenceExportError(f"{context} content hash drifted")
        return path
    except PaperEvidenceExportError:
        if path is not None:
            path.unlink(missing_ok=True)
        raise
    except (KeyError, requests.RequestException, RuntimeError, ValueError):
        if path is not None:
            path.unlink(missing_ok=True)
        return None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_script_sha256(task: object, context: str) -> str:
    raw = getattr(getattr(task, "data", None), "script", None)
    converter = getattr(raw, "to_dict", None)
    raw = converter() if callable(converter) else raw
    if not isinstance(raw, Mapping):
        raise PaperEvidenceExportError(f"{context} cannot expose script metadata")
    if str(raw.get("repository") or "") or str(raw.get("working_dir") or "") != ".":
        raise PaperEvidenceExportError(f"{context} is not a sealed standalone script")
    if str(raw.get("entry_point") or "") != "clearml_5090_bootstrap.py":
        raise PaperEvidenceExportError(f"{context} entry point drifted")
    diff = str(raw.get("diff") or "")
    if not diff:
        raise PaperEvidenceExportError(f"{context} script source is empty")
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _task_parameters(task: object, context: str) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise PaperEvidenceExportError(f"{context} cannot expose parameters")
    value = getter()
    if not isinstance(value, Mapping):
        raise PaperEvidenceExportError(f"{context} parameters are invalid")
    return {str(key): item for key, item in value.items()}


def _condition_id(delay_ms: int, condition: str) -> str:
    return f"delay_{delay_ms:03d}_{condition.lower().replace('-', '_')}"


def _numbers_equal(left: object, right: object) -> bool:
    if type(left) in {int, float} and type(right) in {int, float}:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _numbers_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numbers_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _metric_summary(
    runs: Sequence[Mapping[str, object]], metric: str
) -> dict[str, object]:
    values = {
        (int(run["delay_ms"]), str(run["condition"])): _finite_ap(
            run["metrics"][metric], f"summary {metric}"
        )
        for run in runs
    }
    ordered = [
        (delay, condition, values[(delay, condition)])
        for delay in DELAYS_MS
        for condition in CONDITIONS
    ]
    clean = values[(0, "Full")]
    full_300 = values[(300, "Full")]
    worst_delay, worst_condition, worst = min(ordered, key=lambda item: item[2])
    return {
        "clean_0ms": clean,
        "mean_12": math.fsum(item[2] for item in ordered) / len(ordered),
        "worst_12": {
            "value": worst,
            "delay_ms": worst_delay,
            "condition": worst_condition,
        },
        "full_mean": math.fsum(values[(delay, "Full")] for delay in DELAYS_MS) / 4,
        "l_fail_mean": math.fsum(values[(delay, "L-Fail")] for delay in DELAYS_MS) / 4,
        "c_fail_mean": math.fsum(values[(delay, "C-Fail")] for delay in DELAYS_MS) / 4,
        "full_300": full_300,
        "pdr": None if clean == 0.0 else 100.0 * (clean - full_300) / clean,
    }


def _validate_collected_subject(item: CollectedSubject) -> None:
    identity = item.identity
    if set(identity) != _IDENTITY_KEYS:
        raise PaperEvidenceExportError(
            f"{item.subject} paper identity inventory drifted"
        )
    for field in ("model_name", "modality", "backbone"):
        if not str(identity[field] or "").strip():
            raise PaperEvidenceExportError(f"{item.subject} {field} is empty")
    for field in ("training_task_id", "training_model_id", "evaluation_task_id"):
        _task_id(identity[field], f"{item.subject} {field}")
    for field in (
        "checkpoint_sha256",
        "source_revision_tree_sha256",
        "source_archive_sha256",
        "config_sha256",
        "training_script_sha256",
        "teacher_checkpoint_sha256",
        "metrics_artifact_sha256",
        "prediction_evidence_artifact_sha256",
        "prediction_evidence_archive_sha256",
    ):
        _sha256(identity[field], f"{item.subject} {field}")
    for field in (
        "source_dataset_id",
        "teacher_task_id",
        "teacher_model_id",
        "training_dataset_id",
    ):
        _task_id(identity[field], f"{item.subject} {field}")
    if identity["checkpoint_bytes_verified"] is not True:
        raise PaperEvidenceExportError(
            f"{item.subject} checkpoint bytes are not verified"
        )
    _positive_int(identity["checkpoint_size_bytes"], f"{item.subject} checkpoint")
    _safe_relative(identity["config_path"], f"{item.subject} config path")
    if {
        "task_id": identity["teacher_task_id"],
        "model_id": identity["teacher_model_id"],
        "checkpoint_sha256": identity["teacher_checkpoint_sha256"],
    } != {
        "task_id": TEACHER_TASK_ID,
        "model_id": TEACHER_MODEL_ID,
        "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
    }:
        raise PaperEvidenceExportError(f"{item.subject} teacher identity drifted")
    if identity["training_dataset_id"] != TRAINING_DATASET_ID:
        raise PaperEvidenceExportError(f"{item.subject} training dataset drifted")
    if len(item.runs) != RUN_COUNT:
        raise PaperEvidenceExportError(f"{item.subject} does not contain 12 runs")
    for index, (run, (delay, condition)) in enumerate(
        zip(
            item.runs,
            ((delay, condition) for delay in DELAYS_MS for condition in CONDITIONS),
            strict=True,
        )
    ):
        if set(run) != _RUN_KEYS:
            raise PaperEvidenceExportError(
                f"{item.subject} run {index} inventory drifted"
            )
        expected = {
            "condition_id": _condition_id(delay, condition),
            "delay_ms": delay,
            "condition": condition,
            "agent_scope": AGENT_SCOPE,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
        }
        for field, value in expected.items():
            if run[field] != value:
                raise PaperEvidenceExportError(
                    f"{item.subject} run {index} {field} drifted"
                )
        _sha256(run["prediction_sha256"], f"{item.subject} run {index} prediction")
        _sha256(
            run["prediction_content_sha256"],
            f"{item.subject} run {index} prediction content",
        )
        metrics = run["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != set(AP_METRIC_KEYS):
            raise PaperEvidenceExportError(
                f"{item.subject} run {index} metric inventory drifted"
            )
        for key, value in metrics.items():
            _finite_ap(value, f"{item.subject} run {index} {key}")


def _normalize_runs(
    metrics: Mapping[str, object], *, subject: str, checkpoint_sha256: str
) -> list[dict[str, object]]:
    expected_top = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": RUN_COUNT,
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
    for key, expected in expected_top.items():
        if metrics.get(key) != expected:
            raise PaperEvidenceExportError(f"{subject} metrics {key} drifted")
    if not _is_final_checkpoint_path(metrics.get("checkpoint"), subject):
        raise PaperEvidenceExportError(f"{subject} metrics are not final-only")
    rows = metrics.get("runs")
    if not isinstance(rows, list) or len(rows) != RUN_COUNT:
        raise PaperEvidenceExportError(f"{subject} metrics must contain 12 runs")
    result: list[dict[str, object]] = []
    for index, (run, (delay, condition)) in enumerate(
        zip(rows, ((d, c) for d in DELAYS_MS for c in CONDITIONS), strict=True)
    ):
        if not isinstance(run, Mapping):
            raise PaperEvidenceExportError(f"{subject} metrics run {index} is invalid")
        expected = {
            "condition_id": _condition_id(delay, condition),
            "delay_ms": delay,
            "condition": condition,
            "sample_count": SAMPLE_COUNT,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        }
        for key, value in expected.items():
            if run.get(key) != value:
                raise PaperEvidenceExportError(
                    f"{subject} metrics run {index} {key} drifted"
                )
        raw_metrics = run.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise PaperEvidenceExportError(f"{subject} run {index} lacks metrics")
        for key, count in COUNT_METRICS.items():
            raw = raw_metrics.get(key)
            if (
                type(raw) not in {int, float}
                or not math.isfinite(float(raw))
                or not float(raw).is_integer()
                or int(raw) != count
            ):
                raise PaperEvidenceExportError(
                    f"{subject} run {index} count metric {key} drifted"
                )
        result.append(
            {
                **expected,
                "agent_scope": AGENT_SCOPE,
                "prediction_sha256": _sha256(
                    run.get("prediction_sha256"),
                    f"{subject} run {index} prediction",
                ),
                "prediction_content_sha256": _sha256(
                    run.get("prediction_content_sha256"),
                    f"{subject} run {index} prediction content",
                ),
                "metrics": {
                    key: _finite_ap(
                        raw_metrics.get(key), f"{subject} run {index} {key}"
                    )
                    for key in AP_METRIC_KEYS
                },
            }
        )
    return result


def _validate_prediction_document(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_content_sha256: str,
    context: str,
) -> None:
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PaperEvidenceExportError(f"{context} byte SHA-256 drifted")
    value = _strict_json(raw, context)
    content_sha = _sha256(value.get("content_sha256"), f"{context} content")
    payload = dict(value)
    payload.pop("content_sha256", None)
    if (
        content_sha != expected_content_sha256
        or _content_sha256(payload) != content_sha
    ):
        raise PaperEvidenceExportError(f"{context} content SHA-256 drifted")
    samples = value.get("samples")
    if (
        value.get("sample_count") != SAMPLE_COUNT
        or not isinstance(samples, list)
        or len(samples) != SAMPLE_COUNT
    ):
        raise PaperEvidenceExportError(f"{context} sample count drifted")
    sample_ids: list[str] = []
    ground_truth_count = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or type(sample.get("sample_id")) is not str:
            raise PaperEvidenceExportError(f"{context} sample {index} is invalid")
        sample_ids.append(str(sample["sample_id"]))
        boxes = sample.get("ground_truth_boxes_lidar_bottom_center")
        labels = sample.get("ground_truth_labels")
        if (
            not isinstance(boxes, list)
            or not isinstance(labels, list)
            or len(boxes) != len(labels)
        ):
            raise PaperEvidenceExportError(
                f"{context} sample {index} GT arrays drifted"
            )
        ground_truth_count += len(boxes)
    if _content_sha256(sample_ids) != SAMPLE_IDS_SHA256:
        raise PaperEvidenceExportError(f"{context} sample-ID hash drifted")
    if ground_truth_count != GROUND_TRUTH_COUNT:
        raise PaperEvidenceExportError(f"{context} ground-truth count drifted")


def _is_allowed_prediction_archive_extra(name: str) -> bool:
    match = re.fullmatch(
        r"(delay_(?:000|100|200|300)_(?:full|l_fail|c_fail))/"
        r"(20[0-9]{6}_[0-9]{6})/(.+)",
        name,
    )
    if match is None:
        return False
    timestamp = match.group(2)
    return match.group(3) in {
        f"{timestamp}.json",
        f"{timestamp}.log",
        "vis_data/config.py",
    }


def _validate_prediction_archive(
    path: Path,
    *,
    subject: str,
    checkpoint_sha256: str,
    plan: Mapping[str, object],
    metrics: Mapping[str, object],
    runs: Sequence[Mapping[str, object]],
) -> str:
    archive_sha = _sha256_path(path)
    try:
        archive = zipfile.ZipFile(path, "r", allowZip64=False)
    except (OSError, zipfile.BadZipFile) as error:
        raise PaperEvidenceExportError(
            f"{subject} evidence is not a valid ZIP"
        ) from error
    with archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        expected_names = {"evaluation_plan.json", "metrics.json"}
        for delay in DELAYS_MS:
            for condition in CONDITIONS:
                condition_id = _condition_id(delay, condition)
                expected_names.update(
                    {
                        f"{condition_id}/resolved_config.py",
                        f"{condition_id}/predictions.json",
                        f"{condition_id}/checkpoint.sha256",
                    }
                )
        observed_names = set(names)
        extras = observed_names - expected_names
        legacy_runner_inventory = bool(extras)
        if (
            len(names) != len(observed_names)
            or not expected_names.issubset(observed_names)
            or any(not _is_allowed_prediction_archive_extra(name) for name in extras)
        ):
            raise PaperEvidenceExportError(f"{subject} evidence ZIP inventory drifted")
        total = 0
        for member in members:
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or member.is_dir():
                raise PaperEvidenceExportError(
                    f"{subject} evidence ZIP has unsafe member"
                )
            if member.file_size < 0 or member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise PaperEvidenceExportError(
                    f"{subject} evidence ZIP member is too large"
                )
            total += member.file_size
        if total > MAX_ARCHIVE_TOTAL_BYTES:
            raise PaperEvidenceExportError(f"{subject} evidence ZIP expands too large")
        archived_plan = _strict_json(
            archive.read("evaluation_plan.json"), f"{subject} archived plan"
        )
        archived_metrics = _strict_json(
            archive.read("metrics.json"), f"{subject} archived metrics"
        )
        if not _numbers_equal(archived_plan, plan) or not _numbers_equal(
            archived_metrics, metrics
        ):
            raise PaperEvidenceExportError(
                f"{subject} evidence ZIP documents differ from task artifacts"
            )
        plan_runs = plan.get("runs")
        if not isinstance(plan_runs, list) or len(plan_runs) != RUN_COUNT:
            raise PaperEvidenceExportError(f"{subject} archived plan run count drifted")
        for index, (run, plan_run) in enumerate(zip(runs, plan_runs, strict=True)):
            if not isinstance(plan_run, Mapping):
                raise PaperEvidenceExportError(f"{subject} plan run {index} is invalid")
            condition_id = str(run["condition_id"])
            config_raw = archive.read(f"{condition_id}/resolved_config.py")
            config_sha = hashlib.sha256(config_raw).hexdigest()
            planned_config_sha = str(plan_run.get("resolved_config_sha256") or "")
            # Legacy Runner archives include timestamped MMEngine diagnostics and
            # a post-run config dump; the sealed plan retains the pre-run config
            # hash. Exact 38-member archives must still match byte-for-byte.
            if (
                _SHA256.fullmatch(planned_config_sha) is None
                or (
                    not legacy_runner_inventory
                    and planned_config_sha != config_sha
                )
            ):
                raise PaperEvidenceExportError(
                    f"{subject} plan run {index} config SHA-256 drifted"
                )
            checkpoint_raw = archive.read(f"{condition_id}/checkpoint.sha256")
            if checkpoint_raw != f"{checkpoint_sha256}\n".encode("ascii"):
                raise PaperEvidenceExportError(
                    f"{subject} run {index} checkpoint binding drifted"
                )
            prediction_raw = archive.read(f"{condition_id}/predictions.json")
            _validate_prediction_document(
                prediction_raw,
                expected_sha256=str(run["prediction_sha256"]),
                expected_content_sha256=str(run["prediction_content_sha256"]),
                context=f"{subject}/{condition_id}/predictions.json",
            )
    return archive_sha


def _validate_plan(
    plan: Mapping[str, object], *, watcher_task_id: str
) -> tuple[str, dict[str, Mapping[str, object]]]:
    seal = _require_seal(plan, "formal evaluation plan")
    expected = {
        "schema_version": 2,
        "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count": RUN_COUNT,
        "subject_order": list(FORMAL_SUBJECT_ORDER),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise PaperEvidenceExportError(f"formal evaluation plan {key} drifted")
    _task_id(plan.get("training_controller_task_id"), "training controller")
    _task_id(plan.get("training_provenance_task_id"), "training provenance")
    rows = plan.get("entries")
    if not isinstance(rows, list) or len(rows) != len(FORMAL_SUBJECT_ORDER):
        raise PaperEvidenceExportError("formal evaluation plan inventory drifted")
    result: dict[str, Mapping[str, object]] = {}
    evaluation_ids: set[str] = set()
    for index, (raw, subject) in enumerate(
        zip(rows, FORMAL_SUBJECT_ORDER, strict=True)
    ):
        if not isinstance(raw, Mapping) or set(raw) != {
            "subject",
            "evaluation_task_id",
            "queue",
        }:
            raise PaperEvidenceExportError(
                f"formal evaluation plan entry {index} drifted"
            )
        if raw.get("subject") != subject:
            raise PaperEvidenceExportError(
                f"formal evaluation plan entry {index} subject drifted"
            )
        evaluation_id = _task_id(raw.get("evaluation_task_id"), f"{subject} evaluation")
        if evaluation_id in evaluation_ids:
            raise PaperEvidenceExportError("formal evaluation IDs are not unique")
        evaluation_ids.add(evaluation_id)
        if raw.get("queue") not in {"GPU4-A100", "GPU4-V100", "GPU4-5090"}:
            raise PaperEvidenceExportError(f"{subject} evaluation queue drifted")
        result[subject] = raw
    del watcher_task_id  # task identity is bound by L/A/S below.
    return seal, result


def _validate_training_manifest(
    value: Mapping[str, object], *, plan: Mapping[str, object]
) -> tuple[str, dict[str, Mapping[str, object]]]:
    seal = _require_seal(value, "formal training manifest")
    expected = {
        "schema_version": 1,
        "manifest_type": "resilient_v2x_formal_1337_training_inputs",
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count": RUN_COUNT,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "training_seed": TRAINING_SEED,
        "training_overlay_protocol_seed": TRAINING_SEED,
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "subject_count": len(FORMAL_SUBJECT_ORDER),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise PaperEvidenceExportError(f"formal training manifest {key} drifted")
    rows = value.get("entries")
    if not isinstance(rows, list) or len(rows) != len(FORMAL_SUBJECT_ORDER):
        raise PaperEvidenceExportError("formal training manifest inventory drifted")
    result: dict[str, Mapping[str, object]] = {}
    task_ids: set[str] = set()
    model_ids: set[str] = set()
    for index, (raw, subject) in enumerate(
        zip(rows, FORMAL_SUBJECT_ORDER, strict=True), start=1
    ):
        if (
            not isinstance(raw, Mapping)
            or raw.get("index") != index
            or raw.get("subject") != subject
        ):
            raise PaperEvidenceExportError(
                f"formal training entry {index} identity drifted"
            )
        task_id = _task_id(raw.get("training_task_id"), f"{subject} training")
        model_id = _task_id(raw.get("model_id"), f"{subject} model")
        if task_id in task_ids or model_id in model_ids:
            raise PaperEvidenceExportError(
                "formal training task/model IDs are not unique"
            )
        task_ids.add(task_id)
        model_ids.add(model_id)
        _sha256(raw.get("checkpoint_sha256"), f"{subject} checkpoint")
        _positive_int(raw.get("checkpoint_size_bytes"), f"{subject} checkpoint size")
        if raw.get("checkpoint_filename") != "epoch_50.pth":
            raise PaperEvidenceExportError(f"{subject} checkpoint policy drifted")
        if (
            raw.get("training_seed") != TRAINING_SEED
            or raw.get("training_overlay_protocol_seed") != TRAINING_SEED
        ):
            raise PaperEvidenceExportError(f"{subject} training seed drifted")
        result[subject] = raw
    _task_id(
        plan.get("training_controller_task_id"),
        "formal training manifest controller binding",
    )
    return seal, result


def _validate_selected_method_identity(
    *,
    identity: Mapping[str, object] | None,
    selector: Mapping[str, object],
    selector_seal: str,
    selected: str,
    winner: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    if identity is None:
        raise PaperEvidenceExportError(
            "final selector lacks the sealed selected-method identity"
        )
    expected_keys = {
        "schema_version",
        "document_type",
        "status",
        "selected_subject",
        "selected_candidate_label",
        "selected_fixed_order_index",
        "selection_artifact",
        "selection_seal_sha256",
        "training_seed",
        "protocol_evidence_fingerprint",
        "checkpoint_policy",
        "method_binding",
        "evidence_bindings",
        "gate_artifact",
        "gate_result",
        "gate_result_content_sha256",
        "seal_sha256",
    }
    if set(identity) != expected_keys:
        raise PaperEvidenceExportError(
            "selected-method identity field inventory drifted"
        )
    identity_seal = _require_seal(identity, "selected-method identity")
    if any(
        identity.get(key) != expected
        for key, expected in {
            "schema_version": 2,
            "document_type": SELECTED_METHOD_IDENTITY_DOCUMENT_TYPE,
            "status": "sealed",
            "selected_subject": selected,
            "selected_candidate_label": winner.get("candidate_label"),
            "selected_fixed_order_index": winner.get("fixed_order_index"),
            "selection_artifact": "final_single_seed_selection",
            "selection_seal_sha256": selector_seal,
            "training_seed": TRAINING_SEED,
            "protocol_evidence_fingerprint": selector.get(
                "protocol_evidence_fingerprint"
            ),
            "checkpoint_policy": CHECKPOINT_POLICY,
        }.items()
    ):
        raise PaperEvidenceExportError(
            "selected-method identity selector binding drifted"
        )
    binding = identity.get("method_binding")
    if not isinstance(binding, Mapping):
        raise PaperEvidenceExportError(
            "selected-method identity lacks the method binding"
        )
    for key in (
        "source_dataset_id",
        "training_dataset_id",
        "teacher_task_id",
        "teacher_model_id",
        "training_task_id",
        "evaluation_task_id",
        "model_id",
    ):
        _task_id(binding.get(key), f"selected method {key}")
    for key in (
        "config_sha256",
        "teacher_checkpoint_sha256",
        "checkpoint_sha256",
        "run_contract_artifact_sha256",
        "initialization_audit_artifact_sha256",
        "final_checkpoint_contract_artifact_sha256",
        "evaluation_plan_artifact_sha256",
        "metrics_artifact_sha256",
        "prediction_evidence_artifact_sha256",
        "prediction_evidence_archive_sha256",
    ):
        _sha256(binding.get(key), f"selected method {key}")
    source_fields = (
        "source_revision_sha256",
        "source_archive_sha256",
    )
    present_source_fields = [
        key for key in source_fields if binding.get(key) is not None
    ]
    if not present_source_fields:
        raise PaperEvidenceExportError(
            "selected-method identity lacks a source revision/archive hash"
        )
    for key in present_source_fields:
        _sha256(binding.get(key), f"selected method {key}")
    config_path = _safe_relative(binding.get("config_path"), "selected config")
    if not config_path.startswith("configs/"):
        raise PaperEvidenceExportError(
            "selected-method identity config is outside configs/"
        )
    if binding.get("checkpoint_filename") != "epoch_50.pth":
        raise PaperEvidenceExportError(
            "selected-method identity is not the final checkpoint"
        )
    _positive_int(binding.get("checkpoint_size_bytes"), "selected checkpoint size")
    for key, expected in {
        "training_dataset_id": TRAINING_DATASET_ID,
        "teacher_task_id": TEACHER_TASK_ID,
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
    }.items():
        if binding.get(key) != expected:
            raise PaperEvidenceExportError(
                f"selected-method identity fixed {key} drifted"
            )
    gate_result = identity.get("gate_result")
    if not _numbers_equal(gate_result, winner):
        raise PaperEvidenceExportError(
            "selected-method identity gate result is unbound"
        )
    if identity.get("gate_result_content_sha256") != _content_sha256(gate_result):
        raise PaperEvidenceExportError(
            "selected-method identity gate result hash drifted"
        )
    gate_artifact = identity.get("gate_artifact")
    if (
        not isinstance(gate_artifact, Mapping)
        or set(gate_artifact)
        != {
            "artifact_name",
            "selection_seal_sha256",
            "upstream_candidate_gate_artifact_sha256",
        }
        or any(
        gate_artifact.get(key) != expected
        for key, expected in {
            "artifact_name": "final_single_seed_selection",
            "selection_seal_sha256": selector_seal,
        }.items()
        )
    ):
        raise PaperEvidenceExportError(
            "selected-method identity gate artifact is unbound"
        )
    upstream_gate_sha = gate_artifact.get(
        "upstream_candidate_gate_artifact_sha256"
    )
    if upstream_gate_sha is not None:
        _sha256(upstream_gate_sha, "selected upstream candidate gate")
    if upstream_gate_sha != binding.get("source_gate_artifact_sha256"):
        raise PaperEvidenceExportError(
            "selected-method identity upstream gate artifact drifted"
        )
    evidence_bindings = identity.get("evidence_bindings")
    selector_bindings = selector.get("input_bindings")
    expected_evidence_binding_keys = {
        "formal_inputs_seal_sha256",
        "candidate_manifest_task_id",
        "candidate_manifest_seal_sha256",
        "sequential_evidence_seal_sha256",
        "sequential_gate_receipt_seal_sha256",
    }
    if (
        not isinstance(evidence_bindings, Mapping)
        or set(evidence_bindings) != expected_evidence_binding_keys
        or not isinstance(selector_bindings, Mapping)
    ):
        raise PaperEvidenceExportError(
            "selected-method identity evidence bindings are unavailable"
        )
    for key in (
        "formal_inputs_seal_sha256",
        "candidate_manifest_task_id",
        "candidate_manifest_seal_sha256",
        "sequential_evidence_seal_sha256",
    ):
        if evidence_bindings.get(key) != selector_bindings.get(key):
            raise PaperEvidenceExportError(
                f"selected-method identity {key} is unbound"
            )
    _sha256(
        evidence_bindings.get("formal_inputs_seal_sha256"),
        "selected formal inputs seal",
    )
    _task_id(
        evidence_bindings.get("candidate_manifest_task_id"),
        "selected candidate manifest task",
    )
    _sha256(
        evidence_bindings.get("candidate_manifest_seal_sha256"),
        "selected candidate manifest seal",
    )
    sequential_evidence_sha = evidence_bindings.get(
        "sequential_evidence_seal_sha256"
    )
    sequential_receipt_sha = evidence_bindings.get(
        "sequential_gate_receipt_seal_sha256"
    )
    if sequential_evidence_sha is not None:
        _sha256(sequential_evidence_sha, "selected sequential evidence seal")
    if sequential_receipt_sha is not None:
        _sha256(sequential_receipt_sha, "selected sequential gate receipt")
    if sequential_receipt_sha != binding.get(
        "sequential_gate_receipt_seal_sha256"
    ):
        raise PaperEvidenceExportError(
            "selected-method identity sequential gate receipt drifted"
        )
    return identity_seal, dict(binding)


def _validate_chain(
    chain: ChainEvidence,
) -> tuple[
    str,
    str,
    str,
    str,
    dict[str, Mapping[str, object]],
    dict[str, Mapping[str, object]],
    str | None,
    dict[str, object] | None,
]:
    plan_seal, plan_entries = _validate_plan(
        chain.plan, watcher_task_id=chain.watcher_task_id
    )
    manifest_seal, training_entries = _validate_training_manifest(
        chain.training_manifest, plan=chain.plan
    )
    leaderboard = chain.leaderboard
    leaderboard_seal = _require_seal(leaderboard, "formal leaderboard")
    leaderboard_expected = {
        "schema_version": 3,
        "leaderboard_type": "resilient_v2x_formal_1337_leaderboard",
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "ground_truth_count": GROUND_TRUTH_COUNT,
        "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count_per_subject": RUN_COUNT,
        "training_seed": TRAINING_SEED,
        "training_overlay_protocol_seed": TRAINING_SEED,
        "training_controller_task_id": chain.plan["training_controller_task_id"],
        "watcher_task_id": chain.watcher_task_id,
        "training_manifest_seal_sha256": manifest_seal,
        "evaluation_plan_seal_sha256": plan_seal,
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "subject_count": len(FORMAL_SUBJECT_ORDER),
        "baseline_subjects": list(FORMAL_LEADERBOARD_BASELINE_SUBJECTS),
        "baseline_count": len(FORMAL_LEADERBOARD_BASELINE_SUBJECTS),
        "metric_keys": list(AP_METRIC_KEYS),
    }
    for key, expected in leaderboard_expected.items():
        if leaderboard.get(key) != expected:
            raise PaperEvidenceExportError(f"formal leaderboard {key} drifted")
    raw_results = leaderboard.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(
        FORMAL_SUBJECT_ORDER
    ):
        raise PaperEvidenceExportError("formal leaderboard result inventory drifted")
    leaderboard_results: dict[str, Mapping[str, object]] = {}
    for index, (row, subject) in enumerate(
        zip(raw_results, FORMAL_SUBJECT_ORDER, strict=True), start=1
    ):
        if (
            not isinstance(row, Mapping)
            or row.get("index") != index
            or row.get("subject") != subject
        ):
            raise PaperEvidenceExportError(f"formal leaderboard row {index} drifted")
        training = training_entries[subject]
        plan_entry = plan_entries[subject]
        bindings = {
            "training_task_id": training["training_task_id"],
            "training_model_id": training["model_id"],
            "training_checkpoint_sha256": training["checkpoint_sha256"],
            "evaluation_task_id": plan_entry["evaluation_task_id"],
        }
        for key, expected in bindings.items():
            if row.get(key) != expected:
                raise PaperEvidenceExportError(
                    f"formal leaderboard {subject} {key} is unbound"
                )
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(AP_METRIC_KEYS):
            raise PaperEvidenceExportError(
                f"formal leaderboard {subject} metrics drifted"
            )
        leaderboard_results[subject] = row

    audit = chain.audit
    audit_seal = _require_seal(audit, "formal comparability audit")
    audit_expected = {
        "schema_version": 3,
        "document_type": "resilient_v2x_formal_1337_comparability_audit",
        "passed": True,
        "audit_task_id": chain.audit_task_id,
        "training_controller_task_id": chain.plan["training_controller_task_id"],
        "training_provenance_task_id": chain.plan["training_provenance_task_id"],
        "watcher_task_id": chain.watcher_task_id,
        "leaderboard_task_id": chain.leaderboard_task_id,
        "protocol_id": PROTOCOL_ID,
        "training_dataset_id": TRAINING_DATASET_ID,
        "training_seed": TRAINING_SEED,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "sample_count": SAMPLE_COUNT,
        "ground_truth_count": GROUND_TRUTH_COUNT,
        "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count_per_subject": RUN_COUNT,
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "subject_count": len(FORMAL_SUBJECT_ORDER),
        "total_evaluation_run_count": len(FORMAL_SUBJECT_ORDER) * RUN_COUNT,
        "metric_keys": list(AP_METRIC_KEYS),
        "training_manifest_seal_sha256": manifest_seal,
        "evaluation_plan_seal_sha256": plan_seal,
        "leaderboard_seal_sha256": leaderboard_seal,
    }
    for key, expected in audit_expected.items():
        if audit.get(key) != expected:
            raise PaperEvidenceExportError(f"formal audit {key} drifted")
    training_records = audit.get("training_tasks")
    evaluation_records = audit.get("evaluation_tasks")
    if not isinstance(training_records, list) or not isinstance(
        evaluation_records, list
    ):
        raise PaperEvidenceExportError("formal audit task records are unavailable")
    if len(training_records) != len(FORMAL_SUBJECT_ORDER) or len(
        evaluation_records
    ) != len(FORMAL_SUBJECT_ORDER):
        raise PaperEvidenceExportError("formal audit task inventory drifted")

    selector = chain.selector
    selector_seal = _require_seal(selector, "formal selector")
    if selector.get("document_type") not in {
        "resilient_v2x_sota_candidate_selection",
        "resilient_v2x_formal_candidate_selection",
        FINAL_SELECTION_DOCUMENT_TYPE,
    }:
        raise PaperEvidenceExportError("formal selector document type drifted")
    if selector.get("selection_is_final") is not True:
        raise PaperEvidenceExportError("formal selector is not final")
    if selector.get("document_type") == FINAL_SELECTION_DOCUMENT_TYPE and any(
        selector.get(key) != expected
        for key, expected in {
            "schema_version": 2,
            "status": "selected",
            "selection_claim": "controlled_leader_selected",
            "claim_scope": (
                "single_seed_same_protocol_DAIR_controlled_leadership"
            ),
            "training_seed": TRAINING_SEED,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "selected_method_identity_artifact": (
                SELECTED_METHOD_IDENTITY_ARTIFACT
            ),
        }.items()
    ):
        raise PaperEvidenceExportError("final selector v2 contract drifted")
    selected = _subject(selector.get("selected_candidate"), "selected candidate")
    if selected in BASELINE_SUBJECTS or selected == P1_ZERO_SHOT_SUBJECT:
        raise PaperEvidenceExportError("formal selector winner is not final-trained")
    fingerprint = selector.get("protocol_evidence_fingerprint")
    fixed = {
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "ground_truth_count": GROUND_TRUTH_COUNT,
        "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "baseline_subjects": list(BASELINE_SUBJECTS),
        "baseline_count": len(BASELINE_SUBJECTS),
    }
    for key, expected in fixed.items():
        observed = selector.get(key)
        if observed is None and isinstance(fingerprint, Mapping):
            observed = fingerprint.get(key)
        if observed != expected:
            raise PaperEvidenceExportError(f"formal selector {key} drifted")
    rows = selector.get("candidate_results")
    if not isinstance(rows, list):
        raise PaperEvidenceExportError(
            "formal selector candidate results are unavailable"
        )
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("subject") == selected
    ]
    if (
        len(matches) != 1
        or matches[0].get("gate_passed") is not True
        or matches[0].get("eligible_for_final_paper_selection") is not True
    ):
        raise PaperEvidenceExportError(
            "formal selector winner is not uniquely gate-eligible"
        )
    for key, expected in {
        "audit_task_id": chain.audit_task_id,
        "leaderboard_task_id": chain.leaderboard_task_id,
        "audit_seal_sha256": audit_seal,
        "leaderboard_seal_sha256": leaderboard_seal,
        "watcher_task_id": chain.watcher_task_id,
    }.items():
        if key in selector and selector.get(key) != expected:
            raise PaperEvidenceExportError(f"formal selector {key} is unbound")
    identity_seal: str | None = None
    selected_binding: dict[str, object] | None = None
    if selector.get("document_type") == FINAL_SELECTION_DOCUMENT_TYPE:
        identity_seal, selected_binding = _validate_selected_method_identity(
            identity=chain.selected_method_identity,
            selector=selector,
            selector_seal=selector_seal,
            selected=selected,
            winner=matches[0],
        )
    elif chain.selected_method_identity is not None:
        raise PaperEvidenceExportError(
            "legacy selector unexpectedly carries a v2 selected-method identity"
        )
    return (
        plan_seal,
        leaderboard_seal,
        audit_seal,
        selector_seal,
        training_entries,
        leaderboard_results,
        identity_seal,
        selected_binding,
    )


def _model_record(
    model_root: Path, subject: str, training_task_id: str
) -> dict[str, object]:
    directory = model_root / subject
    manifest = _read_json(directory / "manifest.json", f"{subject} model manifest")
    if (
        manifest.get("subject") != subject
        or manifest.get("source_task_id") != training_task_id
    ):
        raise PaperEvidenceExportError(f"{subject} model manifest identity drifted")
    schema = manifest.get("schema_version")
    if schema == 2:
        detached = dict(manifest)
        observed_content = _sha256(
            detached.pop("content_sha256", None), f"{subject} model manifest content"
        )
        if _content_sha256(detached) != observed_content:
            raise PaperEvidenceExportError(f"{subject} model manifest content drifted")
        if (
            manifest.get("document_type")
            != "resilient_v2x_completed_candidate_model_archive"
            or manifest.get("status") != "verified"
            or manifest.get("checkpoint_policy") != "epoch_50_final_only_for_paper"
        ):
            raise PaperEvidenceExportError(
                f"{subject} candidate model manifest policy drifted"
            )
    elif schema != 1:
        raise PaperEvidenceExportError(
            f"{subject} model manifest schema is unsupported"
        )
    models = manifest.get("models")
    if not isinstance(models, list):
        raise PaperEvidenceExportError(f"{subject} model manifest models are invalid")
    candidates = [
        row
        for row in models
        if isinstance(row, Mapping) and row.get("role") == "canonical_final"
    ]
    if len(candidates) != 1:
        raise PaperEvidenceExportError(f"{subject} lacks one canonical final model")
    model = dict(candidates[0])
    if schema == 2 and (
        model.get("paper_eligible") is not True or model.get("epoch") != 50
    ):
        raise PaperEvidenceExportError(
            f"{subject} canonical model is not paper-eligible epoch 50"
        )
    model_id = _task_id(model.get("model_id"), f"{subject} model")
    sha = _sha256(model.get("sha256"), f"{subject} checkpoint")
    size = _positive_int(model.get("size_bytes"), f"{subject} checkpoint size")
    filename = str(model.get("filename") or "")
    if Path(filename).name != filename or not filename.endswith("_epoch_50.pth"):
        raise PaperEvidenceExportError(
            f"{subject} canonical checkpoint filename drifted"
        )
    path = directory / filename
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != size
        or _sha256_path(path) != sha
    ):
        raise PaperEvidenceExportError(f"{subject} canonical checkpoint bytes drifted")
    return {
        "manifest": manifest,
        "model_id": model_id,
        "model_name": str(
            model.get("model_name") or f"ResilientV2X {subject} final checkpoint"
        ),
        "sha256": sha,
        "size_bytes": size,
    }


def _training_identity(
    task: object,
    *,
    subject: str,
    training_entry: Mapping[str, object] | None,
    audit: Mapping[str, object],
    model_root: Path,
) -> dict[str, object]:
    context = f"{subject} training task"
    task_id = _task_id(getattr(task, "id", ""), context)
    _reload(task)
    _status(task, context)
    run_contract, _ = _artifact_payload(task, "run_contract", context=context)
    if (
        run_contract.get("task_id") != task_id
        or run_contract.get("experiment") != subject
    ):
        raise PaperEvidenceExportError(
            f"{subject} training run contract identity drifted"
        )
    expected = {
        "training_dataset_id": TRAINING_DATASET_ID,
        "gpus": 4,
        "global_batch_size": 8,
        "max_epochs": 50,
        "val_interval": 10,
        "precision": "FP32",
        "amp": False,
        "condition_evaluation": False,
    }
    for key, value in expected.items():
        if run_contract.get(key) != value:
            raise PaperEvidenceExportError(
                f"{subject} training run contract {key} drifted"
            )
    seed = run_contract.get("training_seed", run_contract.get("seed"))
    if seed != TRAINING_SEED:
        raise PaperEvidenceExportError(f"{subject} training seed drifted")
    config = run_contract.get("config")
    source = run_contract.get("source_archive")
    teacher = run_contract.get("teacher")
    if (
        not isinstance(config, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(teacher, Mapping)
    ):
        raise PaperEvidenceExportError(f"{subject} training identity is incomplete")
    config_path = _training_config_path(config, subject)
    config_sha = _sha256(config.get("config_sha256"), f"{subject} config")
    source_dataset_id = _task_id(
        run_contract.get("source_dataset_id"), f"{subject} source dataset"
    )
    source_archive_sha = _sha256(source.get("sha256"), f"{subject} source archive")
    if {
        "task_id": teacher.get("task_id"),
        "model_id": teacher.get("model_id"),
        "checkpoint_sha256": teacher.get("sha256"),
    } != {
        "task_id": TEACHER_TASK_ID,
        "model_id": TEACHER_MODEL_ID,
        "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
    }:
        raise PaperEvidenceExportError(f"{subject} unified teacher drifted")
    model = _model_record(model_root, subject, task_id)
    manifest = model["manifest"]
    source_tree: object = None
    if isinstance(manifest, Mapping) and manifest.get("schema_version") == 2:
        manifest_source = manifest.get("source")
        manifest_run = manifest.get("run_contract")
        manifest_teacher = manifest.get("teacher")
        if (
            not isinstance(manifest_source, Mapping)
            or not isinstance(manifest_run, Mapping)
            or not isinstance(manifest_teacher, Mapping)
        ):
            raise PaperEvidenceExportError(
                f"{subject} candidate archive bindings are incomplete"
            )
        if (
            manifest_source.get("dataset_id") != source_dataset_id
            or manifest_source.get("archive_sha256") != source_archive_sha
            or manifest_run.get("config_path") != config_path
            or manifest_run.get("config_sha256") != config_sha
            or manifest_run.get("training_dataset_id") != TRAINING_DATASET_ID
            or manifest_teacher.get("task_id") != TEACHER_TASK_ID
            or manifest_teacher.get("model_id") != TEACHER_MODEL_ID
            or manifest_teacher.get("checkpoint_sha256") != TEACHER_CHECKPOINT_SHA256
        ):
            raise PaperEvidenceExportError(
                f"{subject} candidate archive bindings drifted"
            )
        source_tree = manifest_source.get("tree_sha256")
    else:
        records = audit.get("training_tasks")
        if not isinstance(records, list):
            raise PaperEvidenceExportError(
                "formal audit training records are unavailable"
            )
        matches = [
            row
            for row in records
            if isinstance(row, Mapping) and row.get("subject") == subject
        ]
        if len(matches) != 1:
            raise PaperEvidenceExportError(
                f"{subject} formal audit training record is unavailable"
            )
        record = matches[0]
        if (
            record.get("training_task_id") != task_id
            or record.get("source_dataset_id") != source_dataset_id
            or record.get("source_archive_sha256") != source_archive_sha
            or record.get("checkpoint_sha256") != model["sha256"]
            or record.get("checkpoint_size_bytes")
            not in {None, model["size_bytes"]}
        ):
            raise PaperEvidenceExportError(
                f"{subject} formal audit training binding drifted"
            )
        source_tree = record.get("source_revision_tree_sha256")
        if record.get("script_sha256") != _task_script_sha256(task, context):
            raise PaperEvidenceExportError(
                f"{subject} formal audit script binding drifted"
            )
    source_tree_sha = _sha256(source_tree, f"{subject} source tree")
    script_sha = _task_script_sha256(task, context)
    if training_entry is not None:
        if (
            training_entry.get("training_task_id") != task_id
            or training_entry.get("model_id") != model["model_id"]
            or training_entry.get("checkpoint_sha256") != model["sha256"]
            or training_entry.get("checkpoint_size_bytes") != model["size_bytes"]
        ):
            raise PaperEvidenceExportError(
                f"{subject} training manifest/model bytes drifted"
            )
    return {
        "model_name": model["model_name"],
        "modality": "LiDAR+Camera",
        "backbone": "PointPillars+ResNet-50/LSS",
        "training_task_id": task_id,
        "training_model_id": model["model_id"],
        "checkpoint_sha256": model["sha256"],
        "checkpoint_size_bytes": model["size_bytes"],
        "checkpoint_bytes_verified": True,
        "source_revision_tree_sha256": source_tree_sha,
        "source_dataset_id": source_dataset_id,
        "source_archive_sha256": source_archive_sha,
        "config_path": config_path,
        "config_sha256": config_sha,
        "training_script_sha256": script_sha,
        "teacher_task_id": TEACHER_TASK_ID,
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "training_dataset_id": TRAINING_DATASET_ID,
    }


def _collect_evaluation(
    task: object,
    *,
    subject: str,
    training_identity: Mapping[str, object],
    expected_parent: str | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    context = f"{subject} evaluation task"
    task_id = _task_id(getattr(task, "id", ""), context)
    _reload(task)
    _status(task, context)
    if expected_parent is not None and _parent(task) != expected_parent:
        raise PaperEvidenceExportError(f"{subject} evaluation parent drifted")
    records = _artifact_records(task, context)
    expected_artifacts = {
        "run_contract",
        "evaluation_plan",
        "controlled_baseline_metrics",
        "controlled_baseline_evidence",
    }
    if set(records) != expected_artifacts:
        raise PaperEvidenceExportError(
            f"{subject} evaluation artifact inventory drifted"
        )
    run_contract, _ = _artifact_payload(task, "run_contract", context=context)
    expected = {
        "schema_version": 1,
        "mode": "baseline_validate",
        "task_id": task_id,
        "baseline": subject,
        "baseline_task_id": training_identity["training_task_id"],
        "predecessor_task_id": training_identity["training_task_id"],
        "training_dataset_id": TRAINING_DATASET_ID,
        "protocol_id": PROTOCOL_ID,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_run_count": RUN_COUNT,
        "expected_manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "expected_overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "expected_sample_ids_sha256": SAMPLE_IDS_SHA256,
    }
    for key, value in expected.items():
        if run_contract.get(key) != value:
            raise PaperEvidenceExportError(
                f"{subject} evaluation run contract {key} drifted"
            )
    checkpoint = run_contract.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("task_id") != training_identity["training_task_id"]
        or checkpoint.get("model_id") != training_identity["training_model_id"]
        or checkpoint.get("sha256") != training_identity["checkpoint_sha256"]
        or checkpoint.get("size_bytes") != training_identity["checkpoint_size_bytes"]
    ):
        raise PaperEvidenceExportError(
            f"{subject} evaluation checkpoint binding drifted"
        )
    plan, _ = _artifact_payload(task, "evaluation_plan", context=context)
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
    for key, value in plan_expected.items():
        if plan.get(key) != value:
            raise PaperEvidenceExportError(f"{subject} evaluation plan {key} drifted")
    metrics, metrics_record = _artifact_payload(
        task, "controlled_baseline_metrics", context=context
    )
    runs = _normalize_runs(
        metrics,
        subject=subject,
        checkpoint_sha256=str(training_identity["checkpoint_sha256"]),
    )
    evidence_path, evidence_record, evidence_is_temporary = _artifact_file(
        task, "controlled_baseline_evidence", context=context
    )
    try:
        archive_sha = _validate_prediction_archive(
            evidence_path,
            subject=subject,
            checkpoint_sha256=str(training_identity["checkpoint_sha256"]),
            plan=plan,
            metrics=metrics,
            runs=runs,
        )
    finally:
        if evidence_is_temporary:
            evidence_path.unlink(missing_ok=True)
    if archive_sha != evidence_record["hash"]:
        raise PaperEvidenceExportError(f"{subject} evidence artifact SHA-256 drifted")
    identity = {
        **dict(training_identity),
        "evaluation_task_id": task_id,
        "metrics_artifact_sha256": metrics_record["hash"],
        "prediction_evidence_artifact_sha256": evidence_record["hash"],
        "prediction_evidence_archive_sha256": archive_sha,
    }
    return identity, runs


def _gate_result(
    entries: Sequence[CollectedSubject], selected: str
) -> dict[str, object]:
    by_subject = {item.subject: item for item in entries}
    candidate = {
        (int(run["delay_ms"]), str(run["condition"])): float(
            run["metrics"][LEADERSHIP_METRIC]
        )
        for run in by_subject[selected].runs
    }
    baselines = {
        baseline: {
            (int(run["delay_ms"]), str(run["condition"])): float(
                run["metrics"][LEADERSHIP_METRIC]
            )
            for run in by_subject[baseline].runs
        }
        for baseline in BASELINE_SUBJECTS
    }
    comparisons: list[dict[str, object]] = []
    for delay in DELAYS_MS:
        for condition in CONDITIONS:
            values = {
                baseline: baselines[baseline][(delay, condition)]
                for baseline in BASELINE_SUBJECTS
            }
            best = max(values.values())
            best_subjects = [
                baseline for baseline in BASELINE_SUBJECTS if values[baseline] == best
            ]
            candidate_value = candidate[(delay, condition)]
            comparisons.append(
                {
                    "condition_id": _condition_id(delay, condition),
                    "delay_ms": delay,
                    "condition": condition,
                    "candidate_value": candidate_value,
                    "best_baseline_value": best,
                    "best_baseline_subject": best_subjects[0],
                    "best_baseline_subjects": best_subjects,
                    "margin": candidate_value - best,
                    "strictly_leads": candidate_value > best,
                }
            )

    def aggregate(values: Mapping[tuple[int, str], float], dimension: str) -> float:
        ordered = [
            values[(delay, condition)]
            for delay in DELAYS_MS
            for condition in CONDITIONS
        ]
        if dimension == "full_0ms":
            return values[(0, "Full")]
        if dimension == "mean_12":
            return math.fsum(ordered) / len(ordered)
        return min(ordered)

    aggregate_rows: dict[str, dict[str, object]] = {}
    for dimension in ("full_0ms", "mean_12", "worst_12"):
        values = {
            baseline: aggregate(baselines[baseline], dimension)
            for baseline in BASELINE_SUBJECTS
        }
        best = max(values.values())
        best_subjects = [
            baseline for baseline in BASELINE_SUBJECTS if values[baseline] == best
        ]
        candidate_value = aggregate(candidate, dimension)
        margin = candidate_value - best
        full = dimension == "full_0ms"
        aggregate_rows[dimension] = {
            "candidate_value": candidate_value,
            "best_baseline_value": best,
            "best_baseline_subject": best_subjects[0],
            "best_baseline_subjects": best_subjects,
            "margin": margin,
            "strictly_leads": margin > 0.0,
            "gate_comparison": "greater_than_or_equal_to_best_minus_0.5"
            if full
            else "strictly_greater_than_best",
            "passes_gate": margin >= -FULL_0MS_MAX_DEFICIT if full else margin > 0.0,
        }
    return {
        "condition_comparisons": comparisons,
        "aggregate_comparisons": aggregate_rows,
        "gate_passed": all(bool(row["passes_gate"]) for row in aggregate_rows.values()),
        "ranking_values": {
            "worst_12_margin": aggregate_rows["worst_12"]["margin"],
            "mean_12_margin": aggregate_rows["mean_12"]["margin"],
            "full_0ms_margin": aggregate_rows["full_0ms"]["margin"],
        },
    }


def build_paper_evidence(
    *,
    chain: ChainEvidence,
    subjects: Sequence[CollectedSubject],
    producer_task_id: str,
    producer_script_sha256: str,
) -> dict[str, object]:
    (
        _,
        leaderboard_seal,
        _,
        selector_seal,
        _,
        leaderboard_results,
        selected_identity_seal,
        selected_binding,
    ) = _validate_chain(chain)
    selected = _subject(chain.selector.get("selected_candidate"), "selected candidate")
    expected_subjects = (*BASELINE_SUBJECTS, selected)
    if tuple(item.subject for item in subjects) != expected_subjects:
        raise PaperEvidenceExportError(
            "paper evidence must contain five baselines then the winner"
        )
    if len({item.subject for item in subjects}) != 6:
        raise PaperEvidenceExportError("paper evidence subjects must be unique")
    identities: set[tuple[str, str, str]] = set()
    for item in subjects:
        _validate_collected_subject(item)
        identity_key = (
            str(item.identity["training_task_id"]),
            str(item.identity["training_model_id"]),
            str(item.identity["evaluation_task_id"]),
        )
        if identity_key in identities:
            raise PaperEvidenceExportError(
                "paper evidence task/model identities must be unique"
            )
        identities.add(identity_key)
        expected_summaries = {
            metric: _metric_summary(item.runs, metric) for metric in AP_METRIC_KEYS
        }
        leaderboard_row = leaderboard_results.get(item.subject)
        if item.subject in BASELINE_SUBJECTS:
            if leaderboard_row is None or not _numbers_equal(
                leaderboard_row.get("metrics"), expected_summaries
            ):
                raise PaperEvidenceExportError(
                    f"formal leaderboard {item.subject} does not bind all 12x4 values"
                )
            for field, identity_field in {
                "training_task_id": "training_task_id",
                "training_model_id": "training_model_id",
                "training_checkpoint_sha256": "checkpoint_sha256",
                "source_revision_tree_sha256": "source_revision_tree_sha256",
                "evaluation_task_id": "evaluation_task_id",
            }.items():
                if leaderboard_row.get(field) != item.identity.get(identity_field):
                    raise PaperEvidenceExportError(
                        f"formal leaderboard {item.subject} {field} is unbound"
                    )
    selected_item = subjects[-1]
    if selected_binding is not None:
        identity_field_map = {
            "training_task_id": "training_task_id",
            "model_id": "training_model_id",
            "evaluation_task_id": "evaluation_task_id",
            "checkpoint_sha256": "checkpoint_sha256",
            "checkpoint_size_bytes": "checkpoint_size_bytes",
            "source_dataset_id": "source_dataset_id",
            "config_path": "config_path",
            "config_sha256": "config_sha256",
            "teacher_task_id": "teacher_task_id",
            "teacher_model_id": "teacher_model_id",
            "teacher_checkpoint_sha256": "teacher_checkpoint_sha256",
            "training_dataset_id": "training_dataset_id",
            "metrics_artifact_sha256": "metrics_artifact_sha256",
            "prediction_evidence_artifact_sha256": (
                "prediction_evidence_artifact_sha256"
            ),
            "prediction_evidence_archive_sha256": (
                "prediction_evidence_archive_sha256"
            ),
        }
        for binding_key, identity_key in identity_field_map.items():
            if selected_binding.get(binding_key) != selected_item.identity.get(
                identity_key
            ):
                raise PaperEvidenceExportError(
                    f"selected-method identity {binding_key} differs from "
                    "independently collected evidence"
                )
        for binding_key, identity_key in {
            "source_revision_sha256": "source_revision_tree_sha256",
            "source_archive_sha256": "source_archive_sha256",
        }.items():
            if binding_key in selected_binding and selected_binding.get(
                binding_key
            ) != selected_item.identity.get(identity_key):
                raise PaperEvidenceExportError(
                    f"selected-method identity {binding_key} differs from "
                    "independently collected evidence"
                )
    gate = _gate_result(subjects, selected)
    if gate["gate_passed"] is not True:
        raise PaperEvidenceExportError(
            "recomputed selected method does not pass the SOTA gate"
        )
    rows = chain.selector.get("candidate_results")
    if not isinstance(rows, list):
        raise PaperEvidenceExportError(
            "formal selector candidate results are unavailable"
        )
    winner = next(
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("subject") == selected
    )
    for field in ("condition_comparisons", "aggregate_comparisons"):
        if not _numbers_equal(winner.get(field), gate[field]):
            raise PaperEvidenceExportError(f"formal selector winner {field} is unbound")
    ranking = winner.get("ranking_values")
    if not isinstance(ranking, Mapping) or any(
        not _numbers_equal(ranking.get(key), value)
        for key, value in gate["ranking_values"].items()
    ):
        raise PaperEvidenceExportError("formal selector winner ranking is unbound")
    entries = []
    for index, item in enumerate(subjects):
        entries.append(
            {
                "subject": item.subject,
                "role": "baseline" if index < 5 else "winner",
                "display_name": item.display_name,
                "identity": dict(item.identity),
                "runs": [dict(run) for run in item.runs],
            }
        )
    document = _sealed(
        {
            "schema_version": 2 if selected_identity_seal is not None else 1,
            "document_type": FORMAL_EVIDENCE_TYPE,
            "protocol_id": PROTOCOL_ID,
            "training_seed": TRAINING_SEED,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
            "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
            "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
            "delays_ms": list(DELAYS_MS),
            "conditions": list(CONDITIONS),
            "run_count_per_subject": RUN_COUNT,
            "agent_scope": AGENT_SCOPE,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "baseline_subjects": list(BASELINE_SUBJECTS),
            "selected_candidate": selected,
            "leaderboard_seal_sha256": leaderboard_seal,
            "selector_seal_sha256": selector_seal,
            **(
                {
                    "selected_method_identity_seal_sha256": (
                        selected_identity_seal
                    )
                }
                if selected_identity_seal is not None
                else {}
            ),
            "unified_teacher": {
                "task_id": TEACHER_TASK_ID,
                "model_id": TEACHER_MODEL_ID,
                "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            },
            "training_dataset_id": TRAINING_DATASET_ID,
            "producer": {
                "task_id": _task_id(producer_task_id, "producer task"),
                "script_sha256": _sha256(producer_script_sha256, "producer script"),
                "artifact_name": FORMAL_EVIDENCE_ARTIFACT,
            },
            "entries": entries,
        }
    )
    if len(entries) * RUN_COUNT * len(AP_METRIC_KEYS) != 288:
        raise PaperEvidenceExportError(
            "paper evidence does not contain exactly 288 cells"
        )
    return document


def _resolve_task(task_class: object, task_id: str, context: str) -> object:
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise PaperEvidenceExportError("ClearML Task class cannot resolve tasks")
    task = getter(task_id=task_id)
    if _task_id(getattr(task, "id", ""), context) != task_id:
        raise PaperEvidenceExportError(f"{context} identity drifted")
    _reload(task)
    _status(task, context)
    return task


def collect_from_clearml(
    *,
    task_class: object,
    watcher_task_id: str,
    leaderboard_task_id: str,
    audit_task_id: str,
    selector_task_id: str,
    selector_artifact: str,
    selected_method_identity_artifact: str,
    producer_task_id: str,
    model_root: Path,
    winner_training_task_id: str | None = None,
    winner_evaluation_task_id: str | None = None,
) -> dict[str, object]:
    watcher_task_id = _task_id(watcher_task_id, "watcher task")
    leaderboard_task_id = _task_id(leaderboard_task_id, "leaderboard task")
    audit_task_id = _task_id(audit_task_id, "audit task")
    selector_task_id = _task_id(selector_task_id, "selector task")
    watcher = _resolve_task(task_class, watcher_task_id, "watcher task")
    leaderboard_task = _resolve_task(
        task_class, leaderboard_task_id, "leaderboard task"
    )
    audit_task = _resolve_task(task_class, audit_task_id, "audit task")
    selector_task = _resolve_task(task_class, selector_task_id, "selector task")
    if (
        _parent(leaderboard_task) != watcher_task_id
        or _parent(audit_task) != leaderboard_task_id
        or _parent(selector_task) != audit_task_id
    ):
        raise PaperEvidenceExportError("formal W/L/A/S parent chain drifted")
    plan, _ = _artifact_payload(
        watcher, "formal_1337_evaluation_plan", context="watcher task"
    )
    leaderboard, _ = _artifact_payload(
        leaderboard_task, "formal_1337_leaderboard", context="leaderboard task"
    )
    audit, _ = _artifact_payload(
        audit_task, "formal_1337_comparability_audit", context="audit task"
    )
    selector, _ = _artifact_payload(
        selector_task, selector_artifact, context="selector task"
    )
    selected_method_identity: dict[str, object] | None = None
    if selector.get("document_type") == FINAL_SELECTION_DOCUMENT_TYPE:
        selected_method_identity, _ = _artifact_payload(
            selector_task,
            selected_method_identity_artifact,
            context="selector task",
        )
    controller_id = _task_id(
        plan.get("training_controller_task_id"), "training controller"
    )
    controller = _resolve_task(task_class, controller_id, "training controller")
    training_manifest, _ = _artifact_payload(
        controller, "formal_1337_training_manifest", context="training controller"
    )
    chain = ChainEvidence(
        watcher_task_id,
        leaderboard_task_id,
        audit_task_id,
        selector_task_id,
        plan,
        leaderboard,
        audit,
        selector,
        training_manifest,
        selected_method_identity,
    )
    (
        _,
        _,
        _,
        _,
        training_entries,
        leaderboard_results,
        _,
        selected_binding,
    ) = _validate_chain(chain)
    selected = _subject(selector.get("selected_candidate"), "selected candidate")
    collected: list[CollectedSubject] = []
    for subject in (*BASELINE_SUBJECTS, selected):
        if subject in training_entries:
            training_id = str(training_entries[subject]["training_task_id"])
            evaluation_id = str(leaderboard_results[subject]["evaluation_task_id"])
            expected_parent = controller_id
            training_entry = training_entries[subject]
        else:
            if subject != selected or selected_binding is None:
                raise PaperEvidenceExportError(
                    "external selected method requires a sealed method identity"
                )
            bound_training_id = _task_id(
                selected_binding.get("training_task_id"), "identity winner training"
            )
            bound_evaluation_id = _task_id(
                selected_binding.get("evaluation_task_id"),
                "identity winner evaluation",
            )
            if winner_training_task_id is not None and _task_id(
                winner_training_task_id, "winner training"
            ) != bound_training_id:
                raise PaperEvidenceExportError(
                    "explicit winner training task differs from selected identity"
                )
            if winner_evaluation_task_id is not None and _task_id(
                winner_evaluation_task_id, "winner evaluation"
            ) != bound_evaluation_id:
                raise PaperEvidenceExportError(
                    "explicit winner evaluation task differs from selected identity"
                )
            training_id = bound_training_id
            evaluation_id = bound_evaluation_id
            expected_parent = training_id
            training_entry = None
        training_task = _resolve_task(
            task_class, training_id, f"{subject} training task"
        )
        training_identity = _training_identity(
            training_task,
            subject=subject,
            training_entry=training_entry,
            audit=audit,
            model_root=model_root,
        )
        evaluation_task = _resolve_task(
            task_class, evaluation_id, f"{subject} evaluation task"
        )
        identity, runs = _collect_evaluation(
            evaluation_task,
            subject=subject,
            training_identity=training_identity,
            expected_parent=expected_parent,
        )
        collected.append(
            CollectedSubject(
                subject,
                DISPLAY_NAMES.get(subject, f"Selected controlled method ({subject})"),
                identity,
                runs,
            )
        )
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return build_paper_evidence(
        chain=chain,
        subjects=collected,
        producer_task_id=producer_task_id,
        producer_script_sha256=script_sha,
    )


def _atomic_create(path: Path, value: Mapping[str, object]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PaperEvidenceExportError("output already exists; refusing overwrite")
    data = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watcher-task-id", required=True)
    parser.add_argument("--leaderboard-task-id", required=True)
    parser.add_argument("--audit-task-id", required=True)
    parser.add_argument("--selector-task-id", required=True)
    parser.add_argument("--selector-artifact", default="final_single_seed_selection")
    parser.add_argument(
        "--selected-method-identity-artifact",
        default=SELECTED_METHOD_IDENTITY_ARTIFACT,
    )
    parser.add_argument("--producer-task-id", required=True)
    parser.add_argument("--winner-training-task-id")
    parser.add_argument("--winner-evaluation-task-id")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--write-token", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if Task is None:
        raise PaperEvidenceExportError("ClearML is unavailable")
    evidence = collect_from_clearml(
        task_class=Task,
        watcher_task_id=args.watcher_task_id,
        leaderboard_task_id=args.leaderboard_task_id,
        audit_task_id=args.audit_task_id,
        selector_task_id=args.selector_task_id,
        selector_artifact=args.selector_artifact,
        selected_method_identity_artifact=(
            args.selected_method_identity_artifact
        ),
        producer_task_id=args.producer_task_id,
        model_root=args.model_root,
        winner_training_task_id=args.winner_training_task_id,
        winner_evaluation_task_id=args.winner_evaluation_task_id,
    )
    if args.write:
        if args.write_token != WRITE_TOKEN:
            raise PaperEvidenceExportError(
                f"--write requires --write-token {WRITE_TOKEN}"
            )
        _atomic_create(args.output, evidence)
    report = {
        "status": "written" if args.write else "ready",
        "document_type": FORMAL_EVIDENCE_TYPE,
        "protocol_id": PROTOCOL_ID,
        "training_seed": TRAINING_SEED,
        "selected_candidate": evidence["selected_candidate"],
        "subject_count": len(evidence["entries"]),
        "run_count": len(evidence["entries"]) * RUN_COUNT,
        "controlled_cell_count": len(evidence["entries"])
        * RUN_COUNT
        * len(AP_METRIC_KEYS),
        "seal_sha256": evidence["seal_sha256"],
        "output": str(args.output) if args.write else None,
        "clearml_write_performed": False,
        "local_write_performed": bool(args.write),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
