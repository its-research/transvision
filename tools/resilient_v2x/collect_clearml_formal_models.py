#!/usr/bin/env python3
"""Collect the provenance-bound ResilientV2X formal models from ClearML."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import monotonic
from urllib.parse import quote, unquote, unquote_to_bytes, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests


FORMAL_MANIFEST_ARTIFACT = "formal_1337_training_manifest"
TRAINING_SUMMARY_ARTIFACT = "post_main_training_summary"
QUALITY_GATE_ARTIFACT = "teacher_quality_gate"
RUN_CONTRACT_ARTIFACT = "run_contract"
FINAL_CHECKPOINT_ARTIFACT = "final_checkpoint_contract"
COMMON_TEACHER_AUDIT_ARTIFACT = "common_teacher_initialization_audit"
PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
CHECKPOINT_POLICY = "epoch_50_final_only"
RELEASE_SEMANTICS = "formal_manifest_after_full_training_suite_completion"
QUALITY_GATE_DOCUMENT_TYPE = "resilient_v2x_teacher_quality_gate"
COLLECTION_DOCUMENT_TYPE = "resilient_v2x_local_model_registry"
COMPLETION_DOCUMENT_TYPE = "resilient_v2x_local_model_collection_complete"
EXPECTED_SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
EXPECTED_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
EXPECTED_SOURCE_ARCHIVE_BYTES = 1_222_481
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6"
)
EXPECTED_SOURCE_INVENTORY_BYTES = 101_195
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "bed72cd86438f2ba932edda14052e3cd3d9589ec201d09d18a315e18a2f7cff2"
)
EXPECTED_SOURCE_FILE_COUNT = 631
EXPECTED_SOURCE_BYTES = 8_926_106
EXPECTED_NEW_SOURCE_DATASET_ID = "351feedbbe81481fa31f1e9ae11a3f4e"
EXPECTED_NEW_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-ad511d88b731.tar.zst"
EXPECTED_NEW_SOURCE_ARCHIVE_BYTES = 1_222_492
EXPECTED_NEW_SOURCE_ARCHIVE_SHA256 = (
    "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
)
EXPECTED_NEW_SOURCE_TREE_SHA256 = (
    "ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"
)
EXPECTED_NEW_SOURCE_INVENTORY_BYTES = 101_195
EXPECTED_NEW_SOURCE_INVENTORY_SHA256 = (
    "39b1a42af65ad5df935945bd0a4eeac6e7f6e1cfdd1cc608f3dd8a708e9c5ca0"
)
EXPECTED_NEW_SOURCE_FILE_COUNT = 631
EXPECTED_NEW_SOURCE_BYTES = 8_926_102
EXPECTED_SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = (
    "29de9700cac66f9998be643e85a8ec646c04ec17fddbb6207bc1438e9e73941b"
)
EXPECTED_SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = (
    "c3170f4a88b080f9cc7267053f354640dd260687cb2c35a6f7ffed73d69f4154"
)
EXPECTED_TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
EXPECTED_TRAINING_SEED = 20250218
LEGACY_FILES_HOST = "10.100.34.118"
EXPECTED_FILES_HOST = "10.100.35.118"
EXPECTED_FILES_PORT = 8081
EXPECTED_DELAYS_MS = (0, 100, 200, 300)
EXPECTED_CONDITIONS = ("Full", "L-Fail", "C-Fail")
FAILED_STATUSES = frozenset({"failed", "stopped", "closed"})
CLEARML_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SAFE_FILENAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
MALFORMED_PERCENT_ESCAPE_PATTERN = re.compile(r"%(?![0-9A-Fa-f]{2})")
FORBIDDEN_ENCODED_PATH_OCTET_PATTERN = re.compile(
    r"%(?:00|25|2[fF]|5[cC])"
)
PERCENT_ESCAPE_PATTERN = re.compile(r"%[0-9A-Fa-f]{2}")
HTTP_PATH_SAFE_CHARACTERS = "/:@-._~!$&'()*+,;="

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
SOURCE_TREE_BY_SUBJECT = {
    subject: EXPECTED_SOURCE_BY_REVISION[SOURCE_REVISION_BY_SUBJECT[subject]][
        "tree_sha256"
    ]
    for subject in SUBJECT_ORDER
}
SOURCE_REVISION_CERTIFICATE_BY_TREE = {
    EXPECTED_SOURCE_TREE_SHA256: {
        "dataset_id": EXPECTED_SOURCE_DATASET_ID,
        "tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
        "file_count": EXPECTED_SOURCE_FILE_COUNT,
        "source_bytes": EXPECTED_SOURCE_BYTES,
        "archive": {
            "name": EXPECTED_SOURCE_ARCHIVE_NAME,
            "size_bytes": EXPECTED_SOURCE_ARCHIVE_BYTES,
            "sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": EXPECTED_SOURCE_INVENTORY_BYTES,
            "sha256": EXPECTED_SOURCE_INVENTORY_SHA256,
        },
    },
    EXPECTED_NEW_SOURCE_TREE_SHA256: {
        "dataset_id": EXPECTED_NEW_SOURCE_DATASET_ID,
        "tree_sha256": EXPECTED_NEW_SOURCE_TREE_SHA256,
        "file_count": EXPECTED_NEW_SOURCE_FILE_COUNT,
        "source_bytes": EXPECTED_NEW_SOURCE_BYTES,
        "archive": {
            "name": EXPECTED_NEW_SOURCE_ARCHIVE_NAME,
            "size_bytes": EXPECTED_NEW_SOURCE_ARCHIVE_BYTES,
            "sha256": EXPECTED_NEW_SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": EXPECTED_NEW_SOURCE_INVENTORY_BYTES,
            "sha256": EXPECTED_NEW_SOURCE_INVENTORY_SHA256,
        },
    },
}
SOURCE_REVISION_COUNTS = {
    EXPECTED_SOURCE_TREE_SHA256: 21,
    EXPECTED_NEW_SOURCE_TREE_SHA256: 5,
}


def _mixed_source_registry_fields() -> dict[str, object]:
    return {
        "source_revisions": json.loads(
            _canonical_json(SOURCE_REVISION_CERTIFICATE_BY_TREE)
        ),
        "source_revision_by_subject": dict(SOURCE_TREE_BY_SUBJECT),
        "source_revision_equivalence_seal_sha256": (
            EXPECTED_SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        ),
        "source_revision_subject_map_seal_sha256": (
            EXPECTED_SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        ),
        "source_revision_counts": dict(SOURCE_REVISION_COUNTS),
    }
IMPROVEMENT_SUBJECTS = frozenset(
    {
        "support_residual",
        "linear_no_distillation",
        "no_distillation_peak_lr_3e4",
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
BASELINE_SUBJECTS = (
    frozenset(SUBJECT_ORDER)
    - IMPROVEMENT_SUBJECTS
    - (ABLATION_SUBJECTS | {"resilient_v2x"})
)
SUBJECT_KIND = {
    subject: (
        "primary_method"
        if subject == "resilient_v2x"
        else "improvement"
        if subject in IMPROVEMENT_SUBJECTS
        else "ablation"
        if subject in ABLATION_SUBJECTS
        else "baseline"
    )
    for subject in SUBJECT_ORDER
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_ID = "formal-v5-mixed-5c984ad4-ad511d88"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / f"artifacts/trained_models/{COLLECTION_ID}"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "docs/resilient_v2x/model-registry.json"
HTTP_DOWNLOAD_TIMEOUT_SECONDS = 300.0
MIN_FREE_SPACE_MARGIN_BYTES = 1024**3
FREE_SPACE_MARGIN_RATIO = 0.05
_HTTP_SESSION = requests.Session()
_HTTP_SESSION.trust_env = False
_HTTP_GET = _HTTP_SESSION.get


def _canonical_json(value: object, *, ensure_ascii: bool = True) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clearml_id(value: object, context: str) -> str:
    result = str(value or "")
    if CLEARML_ID_PATTERN.fullmatch(result) is None:
        raise RuntimeError(f"{context} must be a lowercase 32-hex ClearML ID")
    return result


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if SHA256_PATTERN.fullmatch(result) is None:
        raise RuntimeError(f"{context} must be a lowercase SHA-256")
    return result


def _positive_integer(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{context} must be a positive integer")
    return value


def _normalized_status(task: object) -> str:
    status = getattr(task, "status", "")
    value = getattr(status, "value", status)
    return str(value or "").strip().lower()


def _reload(task: object) -> None:
    method = getattr(task, "reload", None)
    if callable(method):
        method()


def _wait_for_completed(
    task: object,
    *,
    context: str,
    deadline: float,
    poll_seconds: float,
    monotonic_clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    while True:
        _reload(task)
        status = _normalized_status(task)
        if status == "completed":
            return
        if status in FAILED_STATUSES:
            raise RuntimeError(f"{context} ended with status {status!r}")
        if status not in {"created", "queued", "in_progress"}:
            raise RuntimeError(f"{context} has unsupported status {status!r}")
        if monotonic_clock() >= deadline:
            raise TimeoutError(f"timed out waiting for {context}")
        sleeper(poll_seconds)


def _artifact_metadata(
    task: object, name: str
) -> tuple[object, str, int, str]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"task is missing required artifact {name!r}")
    artifact = artifacts[name]
    url = str(getattr(artifact, "url", "") or "")
    if not url:
        return artifact, "", 0, ""
    try:
        parsed = urlsplit(url)
        decoded_path = unquote_to_bytes(parsed.path).decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(f"artifact {name!r} URL is invalid") from error
    filename = decoded_path.rsplit("/", 1)[-1]
    target = _fileserver_url(
        url,
        expected_filename=filename,
        context=f"artifact {name!r} URL",
    )
    try:
        expected_size = int(getattr(artifact, "size", 0) or 0)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"artifact {name!r} size metadata is invalid") from error
    expected_sha256 = str(getattr(artifact, "hash", "") or "")
    if expected_size <= 0 or SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise RuntimeError(f"artifact {name!r} identity metadata is invalid")
    return artifact, target, expected_size, expected_sha256


def _artifact_payload(
    task: object,
    name: str,
    *,
    auth_header_provider: Callable[[], Mapping[str, str]] | None = None,
    auth_refresher: Callable[[], None] | None = None,
) -> dict[str, object]:
    artifact, url, expected_size, expected_sha256 = _artifact_metadata(task, name)
    if (auth_header_provider is None) != (auth_refresher is None):
        raise RuntimeError(
            "artifact authentication provider and refresher must be supplied together"
        )
    # Unit-test and in-process adapters may expose an already materialized mapping
    # without a durable fileserver URL. Real ClearML artifacts always enter the
    # authenticated branch below because _artifact_metadata requires their sealed
    # URL, size, and SHA-256 metadata.
    if not url:
        getter = getattr(artifact, "get", None)
        if not callable(getter):
            raise RuntimeError(f"artifact {name!r} cannot be downloaded")
        value = getter()
        if not isinstance(value, Mapping):
            raise RuntimeError(f"artifact {name!r} is not a JSON object")
        return dict(value)
    if auth_header_provider is None or auth_refresher is None:
        raise RuntimeError(
            f"artifact {name!r} URL requires authenticated fileserver access"
        )
    response = _authenticated_model_response(
        url=url,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    try:
        normalized_final_url = _fileserver_url(
            str(getattr(response, "url", "") or ""),
            expected_filename=Path(unquote(urlsplit(url).path)).name,
            context=f"artifact {name!r} response URL",
        )
        if normalized_final_url != url:
            raise RuntimeError(f"artifact {name!r} redirected unexpectedly")
        headers = getattr(response, "headers", {})
        content_length = headers.get("Content-Length") if headers else None
        try:
            observed_content_length = (
                int(content_length) if content_length is not None else None
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"artifact {name!r} Content-Length is invalid"
            ) from error
        if (
            observed_content_length is not None
            and observed_content_length != expected_size
        ):
            raise RuntimeError(f"artifact {name!r} Content-Length mismatch")
        chunks: list[bytes] = []
        observed_size = 0
        for block in response.iter_content(chunk_size=1024 * 1024):
            if not block:
                continue
            observed_size += len(block)
            if observed_size > expected_size:
                raise RuntimeError(f"artifact {name!r} exceeded sealed size")
            chunks.append(block)
        raw = b"".join(chunks)
    finally:
        closer = getattr(response, "close", None)
        if callable(closer):
            closer()
    if len(raw) != expected_size:
        raise RuntimeError(f"artifact {name!r} size mismatch")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f"artifact {name!r} SHA-256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"artifact {name!r} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"artifact {name!r} is not a JSON object")
    return dict(payload)


def _verify_seal(payload: Mapping[str, object], *, context: str) -> None:
    observed = _sha256(payload.get("seal_sha256"), f"{context} seal")
    detached = dict(payload)
    detached.pop("seal_sha256", None)
    expected = hashlib.sha256(
        _canonical_json(detached, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if observed != expected:
        raise RuntimeError(f"{context} seal_sha256 mismatch")


def _verify_content_hash(payload: Mapping[str, object], *, context: str) -> None:
    observed = _sha256(payload.get("content_sha256"), f"{context} content hash")
    detached = dict(payload)
    detached.pop("content_sha256", None)
    expected = hashlib.sha256(
        _canonical_json(detached, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if observed != expected:
        raise RuntimeError(f"{context} content_sha256 mismatch")


def _fileserver_url(value: object, *, expected_filename: str, context: str) -> str:
    url = str(value or "")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise RuntimeError(
            f"{context} is not the expected durable fileserver URL"
        ) from error
    raw_path = parsed.path
    invalid_encoding = (
        MALFORMED_PERCENT_ESCAPE_PATTERN.search(raw_path) is not None
        or FORBIDDEN_ENCODED_PATH_OCTET_PATTERN.search(raw_path) is not None
    )
    try:
        decoded_path = unquote_to_bytes(raw_path).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RuntimeError(
            f"{context} is not the expected durable fileserver URL"
        ) from error
    double_encoded = PERCENT_ESCAPE_PATTERN.search(decoded_path) is not None
    unsafe_decoded_path = (
        "\x00" in decoded_path
        or "\\" in decoded_path
        or any(part in {".", ".."} for part in decoded_path.split("/"))
    )
    filename = decoded_path.rsplit("/", 1)[-1]
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {LEGACY_FILES_HOST, EXPECTED_FILES_HOST}
        or port != EXPECTED_FILES_PORT
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or not raw_path.startswith("/")
        or invalid_encoding
        or double_encoded
        or unsafe_decoded_path
        or filename != expected_filename
        or SAFE_FILENAME_PATTERN.fullmatch(filename) is None
    ):
        raise RuntimeError(f"{context} is not the expected durable fileserver URL")
    request_path = quote(
        decoded_path,
        safe=HTTP_PATH_SAFE_CHARACTERS,
        encoding="utf-8",
        errors="strict",
    )
    return urlunsplit(
        (
            parsed.scheme,
            f"{EXPECTED_FILES_HOST}:{EXPECTED_FILES_PORT}",
            request_path,
            "",
            "",
        )
    )


def _task_parameters(task: object) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError("ClearML task cannot expose parameters")
    value = getter()
    if not isinstance(value, Mapping):
        raise RuntimeError("ClearML task parameters are invalid")
    return dict(value)


def _resolve_download_authentication(
    task_class: object,
    *,
    auth_header_provider: Callable[[], Mapping[str, str]] | None,
    auth_refresher: Callable[[], None] | None,
) -> tuple[Callable[[], Mapping[str, str]], Callable[[], None]]:
    if (auth_header_provider is None) != (auth_refresher is None):
        raise RuntimeError(
            "download authentication provider and refresher must be supplied together"
        )
    if auth_header_provider is not None and auth_refresher is not None:
        return auth_header_provider, auth_refresher
    session_getter = getattr(task_class, "_get_default_session", None)
    if not callable(session_getter):
        raise RuntimeError("ClearML Task class cannot expose its authenticated session")
    session = session_getter()
    add_auth_headers = getattr(session, "add_auth_headers", None)
    refresh_token = getattr(session, "refresh_token", None)
    if not callable(add_auth_headers) or not callable(refresh_token):
        raise RuntimeError("ClearML session cannot authenticate model downloads")

    def provide_headers() -> Mapping[str, str]:
        headers: dict[str, str] = {}
        value = add_auth_headers(headers)
        if isinstance(value, Mapping):
            return dict(value)
        return headers

    def refresh() -> None:
        refresh_token()

    return provide_headers, refresh


def _authorization_header(
    provider: Callable[[], Mapping[str, str]],
) -> dict[str, str]:
    try:
        headers = provider()
    except Exception:
        raise RuntimeError("failed to obtain ClearML download authorization") from None
    if not isinstance(headers, Mapping):
        raise RuntimeError("ClearML download authorization headers are invalid")
    authorization = headers.get("Authorization")
    if (
        not isinstance(authorization, str)
        or not authorization.startswith("Bearer ")
        or len(authorization) <= len("Bearer ")
        or "\r" in authorization
        or "\n" in authorization
    ):
        raise RuntimeError("ClearML download authorization header is invalid")
    return {"Authorization": authorization}


def _validate_controller_parameters(
    controller: object,
    *,
    controller_task_id: str,
    quality_gate_task_id: str,
) -> dict[str, str]:
    if _clearml_id(getattr(controller, "id", ""), "controller task") != (
        controller_task_id
    ):
        raise RuntimeError("resolved controller task identity mismatch")
    parameters = _task_parameters(controller)
    if str(parameters.get("Args/gate_task_id") or "") != quality_gate_task_id:
        raise RuntimeError("controller parameter drifted: Args/gate_task_id")
    optional_data_identity = {
        "Args/source_dataset_id": EXPECTED_NEW_SOURCE_DATASET_ID,
        "Args/source_archive_name": EXPECTED_NEW_SOURCE_ARCHIVE_NAME,
        "Args/source_archive_bytes": str(EXPECTED_NEW_SOURCE_ARCHIVE_BYTES),
        "Args/source_archive_sha256": EXPECTED_NEW_SOURCE_ARCHIVE_SHA256,
        "Args/training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
    }
    present = optional_data_identity.keys() & parameters.keys()
    if present and present != optional_data_identity.keys():
        raise RuntimeError("controller has partial optional data identity parameters")
    for key, value in optional_data_identity.items():
        if key not in parameters:
            continue
        if str(parameters.get(key) or "") != value:
            raise RuntimeError(f"controller parameter drifted: {key}")
    return {
        "Args/gate_task_id": quality_gate_task_id,
        **({} if not present else optional_data_identity),
    }


def _manifest_seed_mode(
    manifest: Mapping[str, object], entries: Sequence[Mapping[str, object]]
) -> tuple[str, int]:
    fields = {"training_seed", "training_overlay_protocol_seed"}
    top_present = fields.intersection(manifest)
    entry_presence = [fields.intersection(entry) for entry in entries]
    if not top_present and all(not item for item in entry_presence):
        return "legacy_source_c_run_contract", EXPECTED_TRAINING_SEED
    if top_present != fields or any(item != fields for item in entry_presence):
        raise RuntimeError("formal training manifest has partial seed metadata")
    training_seed = manifest.get("training_seed")
    overlay_seed = manifest.get("training_overlay_protocol_seed")
    if type(training_seed) is not int or training_seed < 0:
        raise RuntimeError("formal training manifest has invalid training_seed")
    if overlay_seed != EXPECTED_TRAINING_SEED:
        raise RuntimeError("formal training overlay protocol seed mismatch")
    for entry in entries:
        if entry.get("training_seed") != training_seed:
            raise RuntimeError("formal methods do not share one training_seed")
        if entry.get("training_overlay_protocol_seed") != overlay_seed:
            raise RuntimeError("formal method overlay protocol seed mismatch")
    return "explicit_manifest", training_seed


def validate_formal_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    manifest = dict(payload)
    _verify_seal(manifest, context="formal training manifest")
    expected_scalars = {
        "schema_version": 1,
        "manifest_type": "resilient_v2x_formal_1337_training_inputs",
        "protocol_id": PROTOCOL_ID,
        "sample_count": 1337,
        "delays_ms": list(EXPECTED_DELAYS_MS),
        "conditions": list(EXPECTED_CONDITIONS),
        "run_count": 12,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "evaluation_release_semantics": RELEASE_SEMANTICS,
        "subject_order": list(SUBJECT_ORDER),
        "subject_count": len(SUBJECT_ORDER),
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"formal training manifest drifted: {key}")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != len(SUBJECT_ORDER):
        raise RuntimeError("formal training manifest entry count mismatch")
    entries: list[dict[str, object]] = []
    task_ids: set[str] = set()
    model_ids: set[str] = set()
    for index, (raw, subject) in enumerate(
        zip(raw_entries, SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(raw, Mapping):
            raise RuntimeError("formal training manifest entry is not an object")
        entry = dict(raw)
        if entry.get("index") != index or entry.get("subject") != subject:
            raise RuntimeError("formal training manifest entry order mismatch")
        if entry.get("kind") != SUBJECT_KIND[subject]:
            raise RuntimeError(f"formal training kind mismatch for {subject}")
        task_id = _clearml_id(entry.get("training_task_id"), f"{subject} training task")
        _clearml_id(
            entry.get("training_predecessor_task_id"),
            f"{subject} training predecessor",
        )
        model_id = _clearml_id(entry.get("model_id"), f"{subject} model")
        if task_id in task_ids or model_id in model_ids:
            raise RuntimeError("formal task/model IDs are not unique")
        task_ids.add(task_id)
        model_ids.add(model_id)
        expected_model_name = f"ResilientV2X {subject} final checkpoint"
        if entry.get("model_name") != expected_model_name:
            raise RuntimeError(f"formal model name mismatch for {subject}")
        expected_remote_filename = f"{subject}_epoch_50.pth"
        _fileserver_url(
            entry.get("model_url"),
            expected_filename=expected_remote_filename,
            context=f"{subject} model URL",
        )
        if entry.get("checkpoint_filename") != "epoch_50.pth":
            raise RuntimeError(f"formal checkpoint filename mismatch for {subject}")
        _sha256(entry.get("checkpoint_sha256"), f"{subject} checkpoint")
        _positive_integer(
            entry.get("checkpoint_size_bytes"), f"{subject} checkpoint size"
        )
        if entry.get("common_teacher_initialization_audit_artifact") != (
            COMMON_TEACHER_AUDIT_ARTIFACT
        ):
            raise RuntimeError(f"formal initialization audit mismatch for {subject}")
        _sha256(
            entry.get("common_teacher_initialization_audit_sha256"),
            f"{subject} initialization audit",
        )
        entries.append(entry)
    seed_evidence, training_seed = _manifest_seed_mode(manifest, entries)
    return {
        "manifest": manifest,
        "entries": entries,
        "seed_evidence": seed_evidence,
        "training_seed": training_seed,
    }


def _validate_summary(
    payload: Mapping[str, object],
    *,
    controller_task_id: str,
    quality_gate_task_id: str,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    summary = dict(payload)
    _verify_seal(summary, context="training summary")
    expected = {
        "schema_version": 1,
        "summary_type": "resilient_v2x_post_main_sequential_training",
        "status": "completed",
        "controller_task_id": controller_task_id,
        "gate_task_id": quality_gate_task_id,
        "experiment_order": list(SUBJECT_ORDER),
        "task_count": len(SUBJECT_ORDER),
        "formal_1337_manifest_artifact": FORMAL_MANIFEST_ARTIFACT,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"training summary drifted: {key}")
    if summary.get("formal_1337_evaluation_manifest") != manifest:
        raise RuntimeError("standalone and nested formal manifests differ")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != len(SUBJECT_ORDER):
        raise RuntimeError("training summary result count mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(results):
        raise RuntimeError("training summary cannot join formal entries")
    for index, (result, entry, subject) in enumerate(
        zip(results, entries, SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(result, Mapping) or not isinstance(entry, Mapping):
            raise RuntimeError("training summary result join is invalid")
        expected_result = {
            "index": index,
            "experiment": subject,
            "task_id": entry.get("training_task_id"),
            "predecessor_task_id": entry.get("training_predecessor_task_id"),
            "model_id": entry.get("model_id"),
            "model_name": entry.get("model_name"),
            "model_url": entry.get("model_url"),
            "checkpoint_sha256": entry.get("checkpoint_sha256"),
            "checkpoint_size_bytes": entry.get("checkpoint_size_bytes"),
        }
        for key, value in expected_result.items():
            if result.get(key) != value:
                raise RuntimeError(
                    f"training summary result drifted for {subject}: {key}"
                )
        seed_fields = {"training_seed", "training_overlay_protocol_seed"}
        for key in seed_fields.intersection(entry):
            if result.get(key) != entry.get(key):
                raise RuntimeError(
                    f"training summary result drifted for {subject}: {key}"
                )
    return summary


def _validate_quality_gate(
    payload: Mapping[str, object], *, quality_gate_task_id: str
) -> dict[str, object]:
    gate = dict(payload)
    _verify_content_hash(gate, context="teacher quality gate")
    if gate.get("schema_version") != 1:
        raise RuntimeError("teacher quality gate schema mismatch")
    if gate.get("document_type") != QUALITY_GATE_DOCUMENT_TYPE:
        raise RuntimeError("teacher quality gate document type mismatch")
    if gate.get("quality_gate_task_id") != quality_gate_task_id:
        raise RuntimeError("teacher quality gate task ID mismatch")
    if gate.get("passed") is not True:
        raise RuntimeError("teacher quality gate did not pass")
    if gate.get("validation_count") != 5:
        raise RuntimeError("teacher quality gate validation count mismatch")
    teacher = gate.get("teacher")
    if not isinstance(teacher, Mapping):
        raise RuntimeError("teacher quality gate has no teacher reference")
    reference = dict(teacher)
    teacher_task_id = _clearml_id(reference.get("task_id"), "teacher task")
    if gate.get("teacher_task_id") != teacher_task_id:
        raise RuntimeError("teacher task identity drifted in quality gate")
    _clearml_id(reference.get("model_id"), "teacher model")
    if reference.get("model_name") != "ResilientV2X clean teacher":
        raise RuntimeError("teacher model name mismatch")
    filename = str(reference.get("checkpoint_filename") or "")
    if SAFE_FILENAME_PATTERN.fullmatch(filename) is None:
        raise RuntimeError("teacher checkpoint filename is unsafe")
    _fileserver_url(
        reference.get("model_url"),
        expected_filename=filename,
        context="teacher model URL",
    )
    _positive_integer(reference.get("checkpoint_size_bytes"), "teacher model size")
    _sha256(reference.get("checkpoint_sha256"), "teacher checkpoint")
    selected_epoch = _positive_integer(
        reference.get("selected_epoch"), "teacher selected epoch"
    )
    if selected_epoch not in {10, 20, 30, 40, 50}:
        raise RuntimeError("teacher selected epoch is not a validation epoch")
    return {"gate": gate, "teacher": reference}


def _require_unique_model(task: object, reference: Mapping[str, object]) -> object:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError("ClearML task cannot expose models")
    models = getter()
    if not isinstance(models, Mapping):
        raise RuntimeError("ClearML task model collection is invalid")
    outputs = list(models.get("output") or [])
    candidates = [
        model
        for model in outputs
        if str(getattr(model, "name", "") or "") == str(reference["model_name"])
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "model-owning task must expose exactly one named final OutputModel"
        )
    model = candidates[0]
    expected = {
        "id": str(reference["model_id"]),
        "task": str(reference["task_id"]),
        "name": str(reference["model_name"]),
        "url": str(reference["model_url"]),
    }
    for attribute, value in expected.items():
        if str(getattr(model, attribute, "") or "") != value:
            raise RuntimeError(f"OutputModel identity drifted: {attribute}")
    return model


def _validate_training_task(
    task: object,
    *,
    entry: Mapping[str, object],
    training_seed: int,
    auth_header_provider: Callable[[], Mapping[str, str]] | None = None,
    auth_refresher: Callable[[], None] | None = None,
) -> dict[str, object]:
    subject = str(entry["subject"])
    task_id = str(entry["training_task_id"])
    if _clearml_id(getattr(task, "id", ""), f"{subject} resolved task") != task_id:
        raise RuntimeError(f"resolved training task mismatch for {subject}")
    if _normalized_status(task) != "completed":
        raise RuntimeError(f"training task {subject} is not completed")
    source_revision = SOURCE_REVISION_BY_SUBJECT[subject]
    expected_source = EXPECTED_SOURCE_BY_REVISION[source_revision]
    parameters = _task_parameters(task)
    expected_source_parameters = {
        "Args/source_dataset_id": expected_source["dataset_id"],
        "Args/source_archive_name": expected_source["archive_name"],
        "Args/source_archive_bytes": expected_source["archive_size_bytes"],
        "Args/source_archive_sha256": expected_source["archive_sha256"],
    }
    for key, expected in expected_source_parameters.items():
        if str(parameters.get(key) or "") != str(expected):
            raise RuntimeError(f"{subject} task parameter drifted: {key}")
    contract = _artifact_payload(
        task,
        RUN_CONTRACT_ARTIFACT,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    expected_contract = {
        "task_id": task_id,
        "experiment": subject,
        "source_dataset_id": expected_source["dataset_id"],
        "training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
        "gpus": 4,
        "ddp_processes": 4,
        "max_epochs": 50,
        "amp": False,
        "precision": "FP32",
        "val_interval": 10,
        "condition_evaluation": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"{subject} run contract drifted: {key}")
    source_archive = contract.get("source_archive")
    if not isinstance(source_archive, Mapping):
        raise RuntimeError(f"{subject} run contract has no source archive identity")
    if source_archive != {
        "name": expected_source["archive_name"],
        "size_bytes": expected_source["archive_size_bytes"],
        "sha256": expected_source["archive_sha256"],
    }:
        raise RuntimeError(f"{subject} run contract source archive identity mismatch")
    if "training_seed" in contract:
        observed_seed = contract.get("training_seed")
        seed_field = "training_seed"
    else:
        observed_seed = contract.get("seed")
        seed_field = "seed"
    if observed_seed != training_seed:
        raise RuntimeError(f"{subject} run contract {seed_field} mismatch")
    checkpoint = _artifact_payload(
        task,
        FINAL_CHECKPOINT_ARTIFACT,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    expected_checkpoint = {
        "model_id": entry["model_id"],
        "name": entry["model_name"],
        "url": entry["model_url"],
        "filename": "epoch_50.pth",
        "size_bytes": entry["checkpoint_size_bytes"],
        "sha256": entry["checkpoint_sha256"],
    }
    for key, expected in expected_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"{subject} final checkpoint contract drifted: {key}")
    initialization_audit = _artifact_payload(
        task,
        COMMON_TEACHER_AUDIT_ARTIFACT,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    initialization_audit_sha256 = hashlib.sha256(
        _canonical_json(initialization_audit, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if initialization_audit_sha256 != entry[
        "common_teacher_initialization_audit_sha256"
    ]:
        raise RuntimeError(f"{subject} initialization audit SHA-256 mismatch")
    _require_unique_model(
        task,
        {
            "task_id": task_id,
            "model_id": entry["model_id"],
            "model_name": entry["model_name"],
            "model_url": entry["model_url"],
        },
    )
    return {
        "source_revision_tree_sha256": expected_source["tree_sha256"],
        "source_dataset_id": expected_source["dataset_id"],
        "source_archive_name": expected_source["archive_name"],
        "source_archive_bytes": expected_source["archive_size_bytes"],
        "source_archive_sha256": expected_source["archive_sha256"],
        "parameters": parameters,
        "run_contract": contract,
        "final_checkpoint_contract": checkpoint,
        "common_teacher_initialization_audit": initialization_audit,
    }


def _safe_output_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists() and expanded.is_symlink():
        raise RuntimeError("model output directory must not be a symlink")
    expanded.mkdir(parents=True, exist_ok=True)
    root = expanded.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("model output path is not a directory")
    return root


@contextmanager
def _collection_lock(root: Path):
    lock_path = root / ".collect.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError("cannot open the formal model collection lock") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("formal model collection lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("formal model collection is already locked") from error
        yield
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _available_bytes(path: Path) -> int:
    filesystem = os.statvfs(path)
    return int(filesystem.f_bavail) * int(filesystem.f_frsize)


def _require_collection_space(path: Path, expected_model_bytes: int) -> None:
    margin = max(
        MIN_FREE_SPACE_MARGIN_BYTES,
        int(expected_model_bytes * FREE_SPACE_MARGIN_RATIO),
    )
    required = expected_model_bytes + margin
    available = _available_bytes(path)
    if available < required:
        raise RuntimeError(
            "insufficient free space for formal models: "
            f"required={required}, available={available}, "
            f"model_bytes={expected_model_bytes}, safety_margin={margin}"
        )


def _safe_destination(root: Path, *parts: str) -> Path:
    if not parts or any(
        not part
        or part in {".", ".."}
        or "/" in part
        or "\\" in part
        or SAFE_FILENAME_PATTERN.fullmatch(part) is None
        for part in parts
    ):
        raise RuntimeError("unsafe model destination component")
    parent = root
    for component in parts[:-1]:
        candidate = parent / component
        if candidate.exists() and candidate.is_symlink():
            raise RuntimeError("model destination parent must not be a symlink")
        candidate.mkdir(exist_ok=True)
        if candidate.is_symlink():
            raise RuntimeError("model destination parent must not be a symlink")
        parent = candidate
    resolved_parent = parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as error:
        raise RuntimeError("model destination escapes output directory") from error
    destination = resolved_parent / parts[-1]
    if destination.is_symlink():
        raise RuntimeError("model destination must not be a symlink")
    return destination


def _scrub_authorization_from_request_owner(owner: object) -> None:
    candidates = (owner, getattr(owner, "response", None))
    for candidate in candidates:
        request = getattr(candidate, "request", None)
        headers = getattr(request, "headers", None)
        if not isinstance(headers, MutableMapping):
            continue
        for key in tuple(headers):
            if str(key).casefold() != "authorization":
                continue
            try:
                del headers[key]
            except Exception:
                try:
                    headers[key] = "<redacted>"
                except Exception:
                    pass


def _authenticated_model_response(
    *,
    url: str,
    auth_header_provider: Callable[[], Mapping[str, str]],
    auth_refresher: Callable[[], None],
) -> object:
    for attempt in range(2):
        headers = _authorization_header(auth_header_provider)
        headers["Accept-Encoding"] = "identity"
        try:
            response = _HTTP_GET(
                url,
                headers=headers,
                stream=True,
                allow_redirects=False,
                timeout=HTTP_DOWNLOAD_TIMEOUT_SECONDS,
            )
        except Exception as error:
            _scrub_authorization_from_request_owner(error)
            raise RuntimeError("authenticated model request failed") from None
        finally:
            headers.clear()
        _scrub_authorization_from_request_owner(response)
        status = getattr(response, "status_code", None)
        if status == 200:
            return response
        closer = getattr(response, "close", None)
        if callable(closer):
            closer()
        if status == 401 and attempt == 0:
            try:
                auth_refresher()
            except Exception:
                raise RuntimeError(
                    "failed to refresh ClearML download authorization"
                ) from None
            continue
        if status == 401:
            raise RuntimeError(
                "authenticated model request remained unauthorized after one refresh"
            )
        raise RuntimeError(f"authenticated model request returned HTTP {status}")
    raise RuntimeError("authenticated model request retry invariant failed")


def _download_verified(
    *,
    model_url: str,
    destination: Path,
    expected_size_bytes: int,
    expected_sha256: str,
    auth_header_provider: Callable[[], Mapping[str, str]],
    auth_refresher: Callable[[], None],
) -> tuple[Path, bool]:
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError("existing model destination is not a regular file")
        if (
            destination.stat().st_size != expected_size_bytes
            or _sha256_path(destination) != expected_sha256
        ):
            raise RuntimeError("existing model destination failed identity checks")
        return destination.resolve(strict=True), True
    url = _fileserver_url(
        model_url,
        expected_filename=destination.name,
        context="streaming model URL",
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        observed_size = 0
        response = _authenticated_model_response(
            url=url,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        )
        try:
            final_url = str(getattr(response, "url", "") or "")
            normalized_final_url = _fileserver_url(
                final_url,
                expected_filename=destination.name,
                context="streaming response URL",
            )
            if normalized_final_url != url:
                raise RuntimeError("streaming model URL redirected unexpectedly")
            headers = getattr(response, "headers", {})
            content_length = headers.get("Content-Length") if headers else None
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        "streaming model response has invalid Content-Length"
                    ) from error
                if declared_size != expected_size_bytes:
                    raise RuntimeError(
                        "streaming model Content-Length does not match contract"
                    )
            iterator = getattr(response, "iter_content", None)
            if not callable(iterator):
                raise RuntimeError("streaming model response cannot expose content")
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                for block in iterator(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    if not isinstance(block, bytes):
                        raise RuntimeError("streaming model response returned non-bytes")
                    observed_size += len(block)
                    if observed_size > expected_size_bytes:
                        raise RuntimeError("streaming model exceeds contracted size")
                    digest.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            closer = getattr(response, "close", None)
            if callable(closer):
                closer()
        if observed_size != expected_size_bytes:
            raise RuntimeError("streaming model size mismatch")
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("streaming model SHA-256 mismatch")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination.resolve(strict=True), False


def _atomic_write_text(destination: Path, content: str) -> None:
    if destination.is_symlink():
        raise RuntimeError("metadata destination must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise RuntimeError("metadata destination parent must not be a symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _registry_document(payload: Mapping[str, object]) -> dict[str, object]:
    document = {
        "schema_version": 2,
        "document_type": COLLECTION_DOCUMENT_TYPE,
        **dict(payload),
    }
    document.pop("content_sha256", None)
    document["content_sha256"] = hashlib.sha256(
        _canonical_json(document, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return document


def _completion_document(
    *,
    revision: str,
    registry_content_sha256: str,
    model_count: int,
) -> dict[str, object]:
    document = {
        "schema_version": 2,
        "document_type": COMPLETION_DOCUMENT_TYPE,
        "collection_id": COLLECTION_ID,
        "status": "complete",
        **_mixed_source_registry_fields(),
        "revision": _sha256(revision, "formal model revision"),
        "registry_content_sha256": _sha256(
            registry_content_sha256, "model registry content hash"
        ),
        "model_count": model_count,
    }
    document["content_sha256"] = hashlib.sha256(
        _canonical_json(document, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return document


def _read_json_object(path: Path, *, context: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} is not a JSON object")
    return dict(value)


def _revision_relative_path(revision: str, *parts: str) -> str:
    return (Path("revisions") / revision / Path(*parts)).as_posix()


def _registry_checksum_content(registry: Mapping[str, object]) -> str:
    teacher = registry.get("teacher")
    methods = registry.get("methods")
    if not isinstance(teacher, Mapping) or not isinstance(methods, list):
        raise RuntimeError("model registry cannot produce checksums")
    records = [teacher, *methods]
    prefix = Path("revisions") / str(registry.get("revision") or "")
    lines: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("model registry contains an invalid model record")
        relative = Path(str(record.get("relative_path") or ""))
        try:
            inside_revision = relative.relative_to(prefix)
        except ValueError as error:
            raise RuntimeError("model registry path is outside its revision") from error
        lines.append(
            f"{_sha256(record.get('sha256'), 'registered model')}  "
            f"{inside_revision.as_posix()}"
        )
    return "\n".join(lines) + "\n"


def _validate_registry_against_sources(
    payload: Mapping[str, object],
    *,
    root: Path,
    revision: str,
    controller_task_id: str,
    quality_gate_task_id: str,
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    quality_gate: Mapping[str, object],
    teacher_reference: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    registry = dict(payload)
    _verify_content_hash(registry, context="local model registry")
    expected_top = {
        "schema_version": 2,
        "document_type": COLLECTION_DOCUMENT_TYPE,
        "collection_id": COLLECTION_ID,
        "status": "complete",
        "controller_task_id": controller_task_id,
        "quality_gate_task_id": quality_gate_task_id,
        "protocol_id": PROTOCOL_ID,
        **_mixed_source_registry_fields(),
        "training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "output_directory": str(root),
        "revision": revision,
        "revision_directory": (Path("revisions") / revision).as_posix(),
        "formal_manifest_seal_sha256": manifest.get("seal_sha256"),
        "training_summary_seal_sha256": summary.get("seal_sha256"),
        "quality_gate_content_sha256": quality_gate.get("content_sha256"),
        "teacher_model_count": 1,
        "epoch50_method_model_count": len(SUBJECT_ORDER),
        "model_count": 1 + len(SUBJECT_ORDER),
        "method_count": len(SUBJECT_ORDER),
    }
    for key, expected in expected_top.items():
        if registry.get(key) != expected:
            raise RuntimeError(f"local model registry drifted: {key}")
    teacher = registry.get("teacher")
    if not isinstance(teacher, Mapping):
        raise RuntimeError("local model registry has no teacher record")
    teacher_filename = Path(
        unquote(urlsplit(str(teacher_reference["model_url"])).path)
    ).name
    teacher_expected = {
        "task_id": teacher_reference["task_id"],
        "model_id": teacher_reference["model_id"],
        "model_name": teacher_reference["model_name"],
        "selected_epoch": teacher_reference["selected_epoch"],
        "checkpoint_filename": teacher_filename,
        "relative_path": _revision_relative_path(
            revision,
            "_teacher",
            str(teacher_reference["task_id"]),
            teacher_filename,
        ),
        "remote_url": teacher_reference["model_url"],
        "size_bytes": teacher_reference["checkpoint_size_bytes"],
        "sha256": teacher_reference["checkpoint_sha256"],
    }
    for key, expected in teacher_expected.items():
        if teacher.get(key) != expected:
            raise RuntimeError(f"local teacher record drifted: {key}")
    methods = registry.get("methods")
    if not isinstance(methods, list) or len(methods) != len(entries):
        raise RuntimeError("local model registry method count mismatch")
    for entry, record in zip(entries, methods, strict=True):
        if not isinstance(record, Mapping):
            raise RuntimeError("local method record is invalid")
        subject = str(entry["subject"])
        task_id = str(entry["training_task_id"])
        remote_filename = Path(
            unquote(urlsplit(str(entry["model_url"])).path)
        ).name
        expected_record = {
            "index": entry["index"],
            "subject": subject,
            "kind": entry["kind"],
            "training_task_id": task_id,
            "model_id": entry["model_id"],
            "model_name": entry["model_name"],
            "source_revision_tree_sha256": SOURCE_TREE_BY_SUBJECT[subject],
            "source_dataset_id": EXPECTED_SOURCE_BY_REVISION[
                SOURCE_REVISION_BY_SUBJECT[subject]
            ]["dataset_id"],
            "source_archive_name": EXPECTED_SOURCE_BY_REVISION[
                SOURCE_REVISION_BY_SUBJECT[subject]
            ]["archive_name"],
            "source_archive_bytes": EXPECTED_SOURCE_BY_REVISION[
                SOURCE_REVISION_BY_SUBJECT[subject]
            ]["archive_size_bytes"],
            "source_archive_sha256": EXPECTED_SOURCE_BY_REVISION[
                SOURCE_REVISION_BY_SUBJECT[subject]
            ]["archive_sha256"],
            "checkpoint_filename": remote_filename,
            "relative_path": _revision_relative_path(
                revision,
                "methods",
                SOURCE_TREE_BY_SUBJECT[subject],
                subject,
                task_id,
                remote_filename,
            ),
            "remote_url": entry["model_url"],
            "size_bytes": entry["checkpoint_size_bytes"],
            "sha256": entry["checkpoint_sha256"],
            "checkpoint_policy": CHECKPOINT_POLICY,
        }
        for key, expected in expected_record.items():
            if record.get(key) != expected:
                raise RuntimeError(f"local {subject} record drifted: {key}")
    expected_total = int(teacher_reference["checkpoint_size_bytes"]) + sum(
        int(entry["checkpoint_size_bytes"]) for entry in entries
    )
    if registry.get("total_size_bytes") != expected_total:
        raise RuntimeError("local model registry total size drifted")
    return registry


def _validate_revision_files(
    *,
    root: Path,
    revision_dir: Path,
    registry: Mapping[str, object],
    require_metadata: bool,
    check_entire_root: bool,
) -> None:
    revision = str(registry["revision"])
    prefix = Path("revisions") / revision
    teacher = registry.get("teacher")
    methods = registry.get("methods")
    if not isinstance(teacher, Mapping) or not isinstance(methods, list):
        raise RuntimeError("local model registry has invalid records")
    records: list[Mapping[str, object]] = [teacher]
    for item in methods:
        if not isinstance(item, Mapping):
            raise RuntimeError("local model registry has invalid method records")
        records.append(item)
    expected_inside: set[Path] = set()
    expected_root: set[Path] = set()
    for record in records:
        relative = Path(str(record["relative_path"]))
        try:
            inside = relative.relative_to(prefix)
        except ValueError as error:
            raise RuntimeError("registered model path escapes revision") from error
        if inside.is_absolute() or ".." in inside.parts:
            raise RuntimeError("registered model path is unsafe")
        expected_inside.add(inside)
        expected_root.add(relative)
        path = revision_dir / inside
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("registered model is missing or not a regular file")
        if (
            path.stat().st_size != int(record["size_bytes"])
            or _sha256_path(path) != str(record["sha256"])
        ):
            raise RuntimeError("registered model failed final identity verification")
    observed_inside = {
        path.relative_to(revision_dir)
        for path in revision_dir.rglob("*.pth")
        if path.is_file() or path.is_symlink()
    }
    if observed_inside != expected_inside:
        raise RuntimeError(
            "formal model revision contains missing or unexpected .pth files"
        )
    if any(path.is_symlink() for path in revision_dir.rglob("*.pth")):
        raise RuntimeError("formal model revision contains a symlinked checkpoint")
    if check_entire_root:
        observed_root = {
            path.relative_to(root)
            for path in root.rglob("*.pth")
            if path.is_file() or path.is_symlink()
        }
        if observed_root != expected_root:
            raise RuntimeError("formal model output contains unexpected .pth files")
    if not require_metadata:
        return
    stored_registry = _read_json_object(
        revision_dir / "manifest.json", context="revision manifest"
    )
    if stored_registry != registry:
        raise RuntimeError("revision manifest differs from the validated registry")
    checksum_path = revision_dir / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise RuntimeError("revision checksum inventory is missing")
    if checksum_path.read_text(encoding="utf-8") != _registry_checksum_content(
        registry
    ):
        raise RuntimeError("revision checksum inventory drifted")
    completion = _read_json_object(
        revision_dir / "COMPLETE", context="revision completion marker"
    )
    _verify_content_hash(completion, context="revision completion marker")
    expected_completion = _completion_document(
        revision=revision,
        registry_content_sha256=str(registry["content_sha256"]),
        model_count=int(registry["model_count"]),
    )
    if completion != expected_completion:
        raise RuntimeError("revision completion marker drifted")


def _revalidate_remote_snapshot(
    *,
    controller: object,
    quality_gate_task: object,
    teacher_task: object,
    controller_task_id: str,
    quality_gate_task_id: str,
    controller_parameters: Mapping[str, object],
    quality_gate_parameters: Mapping[str, object],
    teacher_parameters: Mapping[str, object],
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    quality_gate: Mapping[str, object],
    teacher_reference: Mapping[str, object],
    training_seed: int,
    training_sources: Sequence[
        tuple[object, Mapping[str, object], Mapping[str, object]]
    ],
    auth_header_provider: Callable[[], Mapping[str, str]],
    auth_refresher: Callable[[], None],
) -> None:
    for task, context in (
        (controller, "formal training controller"),
        (quality_gate_task, "teacher quality gate"),
        (teacher_task, "teacher task"),
    ):
        _reload(task)
        if _normalized_status(task) != "completed":
            raise RuntimeError(f"{context} drifted from completed status")
    if _task_parameters(controller) != dict(controller_parameters):
        raise RuntimeError("formal training controller parameters drifted")
    _validate_controller_parameters(
        controller,
        controller_task_id=controller_task_id,
        quality_gate_task_id=quality_gate_task_id,
    )
    if _task_parameters(quality_gate_task) != dict(quality_gate_parameters):
        raise RuntimeError("teacher quality gate parameters drifted")
    if _task_parameters(teacher_task) != dict(teacher_parameters):
        raise RuntimeError("teacher task parameters drifted")
    if _artifact_payload(
        controller,
        FORMAL_MANIFEST_ARTIFACT,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    ) != manifest:
        raise RuntimeError("formal training manifest changed during collection")
    if _artifact_payload(
        controller,
        TRAINING_SUMMARY_ARTIFACT,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    ) != summary:
        raise RuntimeError("training summary changed during collection")
    if _artifact_payload(
        quality_gate_task,
        QUALITY_GATE_ARTIFACT,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    ) != quality_gate:
        raise RuntimeError("teacher quality gate changed during collection")
    _require_unique_model(teacher_task, teacher_reference)
    for task, entry, initial_snapshot in training_sources:
        _reload(task)
        current = _validate_training_task(
            task,
            entry=entry,
            training_seed=training_seed,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        )
        for key in (
            "parameters",
            "run_contract",
            "final_checkpoint_contract",
            "common_teacher_initialization_audit",
        ):
            if current[key] != initial_snapshot[key]:
                raise RuntimeError(
                    f"{entry['subject']} remote identity changed during collection: {key}"
                )


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


def _safe_registry_destination(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise RuntimeError("model registry destination must not be a symlink")
    expanded.parent.mkdir(parents=True, exist_ok=True)
    if expanded.parent.is_symlink():
        raise RuntimeError("model registry parent must not be a symlink")
    parent = expanded.parent.resolve(strict=True)
    destination = parent / expanded.name
    if destination.is_symlink():
        raise RuntimeError("model registry destination must not be a symlink")
    if destination.exists() and not destination.is_file():
        raise RuntimeError("model registry destination must be a regular file")
    return destination


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
    except ValueError:
        pass
    else:
        return True
    try:
        second.relative_to(first)
    except ValueError:
        return False
    return True


def _isolated_registry_destination(*, root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    lexical_destination = Path(os.path.abspath(os.fspath(expanded)))
    lexical_staging_root = root.parent / f".{root.name}.staging"
    if lexical_staging_root.is_symlink():
        raise RuntimeError("model staging directory must not be a symlink")
    staging_root = (
        lexical_staging_root.resolve(strict=True)
        if lexical_staging_root.exists()
        else lexical_staging_root
    )
    precreation_destination = expanded.resolve(strict=False)
    for destination in (lexical_destination, precreation_destination):
        for reserved, context in (
            (root, "model output"),
            (staging_root, "model staging"),
        ):
            if _paths_overlap(destination, reserved):
                raise RuntimeError(
                    f"model registry destination must be isolated from {context} path"
                )
    destination = _safe_registry_destination(expanded)
    for reserved, context in (
        (root, "model output"),
        (staging_root, "model staging"),
    ):
        if _paths_overlap(destination, reserved):
            raise RuntimeError(
                f"model registry destination must be isolated from {context} path"
            )
    return destination


def _root_completion_exists(root: Path) -> bool:
    marker = root / "COMPLETE"
    return marker.exists() or marker.is_symlink()


def _root_completion_matches_registry(
    *, root: Path, registry: Mapping[str, object]
) -> bool:
    marker = root / "COMPLETE"
    if not _root_completion_exists(root):
        return False
    try:
        observed = _read_json_object(marker, context="root completion marker")
    except RuntimeError:
        return False
    expected = _completion_document(
        revision=str(registry["revision"]),
        registry_content_sha256=str(registry["content_sha256"]),
        model_count=int(registry["model_count"]),
    )
    return observed == expected


def _quarantine_root_completion(root: Path) -> Path | None:
    marker = root / "COMPLETE"
    if not _root_completion_exists(root):
        return None
    for attempt in range(100):
        name = f".COMPLETE.stale.{os.getpid()}.{time.time_ns()}.{attempt}"
        quarantine = root / name
        if quarantine.exists() or quarantine.is_symlink():
            continue
        try:
            os.replace(marker, quarantine)
        except OSError as error:
            raise RuntimeError(
                "cannot quarantine stale root completion marker"
            ) from error
        _fsync_directory(root)
        return quarantine
    raise RuntimeError("cannot allocate stale root completion quarantine path")


def _publish_completion(
    *,
    root: Path,
    registry_path: Path,
    registry: Mapping[str, object],
) -> None:
    completion = _completion_document(
        revision=str(registry["revision"]),
        registry_content_sha256=str(registry["content_sha256"]),
        model_count=int(registry["model_count"]),
    )
    registry_text = (
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    destination = _isolated_registry_destination(root=root, path=registry_path)
    _atomic_write_text(destination, registry_text)
    _atomic_write_text(
        _safe_destination(root, "COMPLETE"),
        json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _collect_formal_models_unstaged(
    *,
    task_class: object,
    controller_task_id: str,
    quality_gate_task_id: str,
    output_dir: Path,
    registry_path: Path,
    wait: bool = True,
    poll_seconds: float = 60.0,
    timeout_hours: float = 720.0,
    monotonic_clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    raise RuntimeError("the unstaged formal model collector is disabled")
    controller_task_id = _clearml_id(controller_task_id, "controller task")
    quality_gate_task_id = _clearml_id(quality_gate_task_id, "quality gate task")
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise RuntimeError("ClearML Task class cannot resolve tasks")
    controller = getter(task_id=controller_task_id)
    quality_gate = getter(task_id=quality_gate_task_id)
    deadline = monotonic_clock() + timeout_hours * 3600.0
    if wait:
        _wait_for_completed(
            quality_gate,
            context="teacher quality gate",
            deadline=deadline,
            poll_seconds=poll_seconds,
            monotonic_clock=monotonic_clock,
            sleeper=sleeper,
        )
        _wait_for_completed(
            controller,
            context="formal training controller",
            deadline=deadline,
            poll_seconds=poll_seconds,
            monotonic_clock=monotonic_clock,
            sleeper=sleeper,
        )
    elif (
        _normalized_status(quality_gate) != "completed"
        or _normalized_status(controller) != "completed"
    ):
        raise RuntimeError("quality gate and controller must both be completed")
    _validate_controller_parameters(
        controller,
        controller_task_id=controller_task_id,
        quality_gate_task_id=quality_gate_task_id,
    )
    manifest_payload = _artifact_payload(controller, FORMAL_MANIFEST_ARTIFACT)
    validated = validate_formal_manifest(manifest_payload)
    manifest = validated["manifest"]
    entries = validated["entries"]
    if not isinstance(manifest, Mapping) or not isinstance(entries, list):
        raise RuntimeError("validated formal manifest has invalid types")
    summary = _validate_summary(
        _artifact_payload(controller, TRAINING_SUMMARY_ARTIFACT),
        controller_task_id=controller_task_id,
        quality_gate_task_id=quality_gate_task_id,
        manifest=manifest,
    )
    quality = _validate_quality_gate(
        _artifact_payload(quality_gate, QUALITY_GATE_ARTIFACT),
        quality_gate_task_id=quality_gate_task_id,
    )
    teacher_reference = quality["teacher"]
    if not isinstance(teacher_reference, Mapping):
        raise RuntimeError("validated teacher reference is invalid")
    summary_teacher = summary.get("teacher")
    if not isinstance(summary_teacher, Mapping):
        raise RuntimeError("training summary has no teacher reference")
    for key in ("task_id", "model_id", "checkpoint_sha256"):
        if summary_teacher.get(key) != teacher_reference.get(key):
            raise RuntimeError(f"training summary teacher drifted: {key}")

    root = _safe_output_root(output_dir)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    teacher_task_id = str(teacher_reference["task_id"])
    teacher_task = getter(task_id=teacher_task_id)
    _reload(teacher_task)
    if _normalized_status(teacher_task) != "completed":
        raise RuntimeError("teacher task is not completed")
    _require_unique_model(teacher_task, teacher_reference)
    teacher_filename = Path(
        unquote(urlsplit(str(teacher_reference["model_url"])).path)
    ).name
    teacher_destination = _safe_destination(
        root, "_teacher", teacher_task_id, teacher_filename
    )
    teacher_path, teacher_reused = _download_verified(
        model_url=str(teacher_reference["model_url"]),
        destination=teacher_destination,
        expected_size_bytes=int(teacher_reference["checkpoint_size_bytes"]),
        expected_sha256=str(teacher_reference["checkpoint_sha256"]),
    )
    teacher_record = {
        "task_id": teacher_task_id,
        "model_id": teacher_reference["model_id"],
        "model_name": teacher_reference["model_name"],
        "selected_epoch": teacher_reference["selected_epoch"],
        "selection_protocol": quality["gate"].get("selection_protocol"),
        "checkpoint_filename": teacher_filename,
        "relative_path": teacher_path.relative_to(root).as_posix(),
        "remote_url": teacher_reference["model_url"],
        "size_bytes": teacher_reference["checkpoint_size_bytes"],
        "sha256": teacher_reference["checkpoint_sha256"],
        "reused_existing_file": teacher_reused,
        "verified_at": now,
    }

    method_records: list[dict[str, object]] = []
    for entry in entries:
        subject = str(entry["subject"])
        task_id = str(entry["training_task_id"])
        task = getter(task_id=task_id)
        _reload(task)
        task_snapshot = _validate_training_task(
            task,
            entry=entry,
            training_seed=int(validated["training_seed"]),
        )
        remote_filename = Path(unquote(urlsplit(str(entry["model_url"])).path)).name
        destination = _safe_destination(
            root,
            "methods",
            str(task_snapshot["source_revision_tree_sha256"]),
            subject,
            task_id,
            remote_filename,
        )
        local_path, reused = _download_verified(
            model_url=str(entry["model_url"]),
            destination=destination,
            expected_size_bytes=int(entry["checkpoint_size_bytes"]),
            expected_sha256=str(entry["checkpoint_sha256"]),
        )
        method_records.append(
            {
                "index": entry["index"],
                "subject": subject,
                "kind": entry["kind"],
                "training_task_id": task_id,
                "model_id": entry["model_id"],
                "model_name": entry["model_name"],
                "source_revision_tree_sha256": task_snapshot[
                    "source_revision_tree_sha256"
                ],
                "source_dataset_id": task_snapshot["source_dataset_id"],
                "source_archive_name": task_snapshot["source_archive_name"],
                "source_archive_bytes": task_snapshot["source_archive_bytes"],
                "source_archive_sha256": task_snapshot["source_archive_sha256"],
                "checkpoint_filename": remote_filename,
                "relative_path": local_path.relative_to(root).as_posix(),
                "remote_url": entry["model_url"],
                "size_bytes": entry["checkpoint_size_bytes"],
                "sha256": entry["checkpoint_sha256"],
                "checkpoint_policy": CHECKPOINT_POLICY,
                "reused_existing_file": reused,
                "verified_at": now,
            }
        )

    registry = _registry_document(
        {
            "collection_id": COLLECTION_ID,
            "status": "complete",
            "controller_task_id": controller_task_id,
            "quality_gate_task_id": quality_gate_task_id,
            "protocol_id": PROTOCOL_ID,
            **_mixed_source_registry_fields(),
            "training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
            "training_seed": validated["training_seed"],
            "training_seed_evidence": validated["seed_evidence"],
            "checkpoint_policy": CHECKPOINT_POLICY,
            "output_directory": str(root),
            "teacher": teacher_record,
            "teacher_model_count": 1,
            "epoch50_method_model_count": len(method_records),
            "model_count": 1 + len(method_records),
            "total_size_bytes": int(teacher_record["size_bytes"])
            + sum(int(item["size_bytes"]) for item in method_records),
            "method_count": len(method_records),
            "methods": method_records,
            "downloaded_at": now,
        }
    )
    manifest_path = _safe_destination(root, "manifest.json")
    _atomic_write_text(
        manifest_path,
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    checksum_lines = [
        f"{teacher_record['sha256']}  {teacher_record['relative_path']}"
    ] + [f"{item['sha256']}  {item['relative_path']}" for item in method_records]
    checksum_path = _safe_destination(root, "SHA256SUMS")
    _atomic_write_text(checksum_path, "\n".join(checksum_lines) + "\n")
    _atomic_write_text(
        registry_path.expanduser().resolve(),
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return registry


def _capture_collection_sources(
    *,
    getter: Callable[..., object],
    controller: object,
    quality_gate_task: object,
    controller_task_id: str,
    quality_gate_task_id: str,
    auth_header_provider: Callable[[], Mapping[str, str]],
    auth_refresher: Callable[[], None],
) -> dict[str, object]:
    _validate_controller_parameters(
        controller,
        controller_task_id=controller_task_id,
        quality_gate_task_id=quality_gate_task_id,
    )
    controller_parameters = _task_parameters(controller)
    manifest_payload = _artifact_payload(
        controller,
        FORMAL_MANIFEST_ARTIFACT,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    validated = validate_formal_manifest(manifest_payload)
    manifest = validated["manifest"]
    entries = validated["entries"]
    if not isinstance(manifest, Mapping) or not isinstance(entries, list):
        raise RuntimeError("validated formal manifest has invalid types")
    summary = _validate_summary(
        _artifact_payload(
            controller,
            TRAINING_SUMMARY_ARTIFACT,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        ),
        controller_task_id=controller_task_id,
        quality_gate_task_id=quality_gate_task_id,
        manifest=manifest,
    )
    quality = _validate_quality_gate(
        _artifact_payload(
            quality_gate_task,
            QUALITY_GATE_ARTIFACT,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        ),
        quality_gate_task_id=quality_gate_task_id,
    )
    quality_gate = quality["gate"]
    teacher_reference = quality["teacher"]
    if not isinstance(quality_gate, Mapping) or not isinstance(
        teacher_reference, Mapping
    ):
        raise RuntimeError("validated teacher quality evidence is invalid")
    summary_teacher = summary.get("teacher")
    if not isinstance(summary_teacher, Mapping):
        raise RuntimeError("training summary has no teacher reference")
    for key in ("task_id", "model_id", "checkpoint_sha256"):
        if summary_teacher.get(key) != teacher_reference.get(key):
            raise RuntimeError(f"training summary teacher drifted: {key}")
    teacher_task = getter(task_id=str(teacher_reference["task_id"]))
    _reload(teacher_task)
    if _normalized_status(teacher_task) != "completed":
        raise RuntimeError("teacher task is not completed")
    _require_unique_model(teacher_task, teacher_reference)
    training_sources: list[
        tuple[object, Mapping[str, object], Mapping[str, object]]
    ] = []
    training_seed = int(validated["training_seed"])
    for entry in entries:
        task = getter(task_id=str(entry["training_task_id"]))
        _reload(task)
        snapshot = _validate_training_task(
            task,
            entry=entry,
            training_seed=training_seed,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        )
        training_sources.append((task, entry, snapshot))
    return {
        "controller_parameters": controller_parameters,
        "quality_gate_parameters": _task_parameters(quality_gate_task),
        "teacher_parameters": _task_parameters(teacher_task),
        "manifest": manifest,
        "entries": entries,
        "summary": summary,
        "quality_gate": quality_gate,
        "teacher_reference": teacher_reference,
        "teacher_task": teacher_task,
        "training_seed": training_seed,
        "training_seed_evidence": validated["seed_evidence"],
        "training_sources": training_sources,
    }


def _build_staged_registry(
    *,
    root: Path,
    stage: Path,
    revision: str,
    controller_task_id: str,
    quality_gate_task_id: str,
    sources: Mapping[str, object],
    auth_header_provider: Callable[[], Mapping[str, str]],
    auth_refresher: Callable[[], None],
) -> dict[str, object]:
    teacher_reference = sources["teacher_reference"]
    entries = sources["entries"]
    training_sources = sources["training_sources"]
    if not isinstance(teacher_reference, Mapping) or not isinstance(entries, list):
        raise RuntimeError("captured collection sources are invalid")
    if not isinstance(training_sources, list):
        raise RuntimeError("captured training sources are invalid")
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    teacher_task_id = str(teacher_reference["task_id"])
    teacher_filename = Path(
        unquote(urlsplit(str(teacher_reference["model_url"])).path)
    ).name
    teacher_destination = _safe_destination(
        stage, "_teacher", teacher_task_id, teacher_filename
    )
    _download_verified(
        model_url=str(teacher_reference["model_url"]),
        destination=teacher_destination,
        expected_size_bytes=int(teacher_reference["checkpoint_size_bytes"]),
        expected_sha256=str(teacher_reference["checkpoint_sha256"]),
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    teacher_record = {
        "task_id": teacher_task_id,
        "model_id": teacher_reference["model_id"],
        "model_name": teacher_reference["model_name"],
        "selected_epoch": teacher_reference["selected_epoch"],
        "selection_protocol": sources["quality_gate"].get("selection_protocol"),
        "checkpoint_filename": teacher_filename,
        "relative_path": _revision_relative_path(
            revision, "_teacher", teacher_task_id, teacher_filename
        ),
        "remote_url": teacher_reference["model_url"],
        "size_bytes": teacher_reference["checkpoint_size_bytes"],
        "sha256": teacher_reference["checkpoint_sha256"],
        "reused_existing_file": False,
        "verified_at": now,
    }
    method_records: list[dict[str, object]] = []
    for item in training_sources:
        if not isinstance(item, tuple) or len(item) != 3:
            raise RuntimeError("captured training source is invalid")
        _, entry, snapshot = item
        if not isinstance(entry, Mapping) or not isinstance(snapshot, Mapping):
            raise RuntimeError("captured training source is invalid")
        subject = str(entry["subject"])
        task_id = str(entry["training_task_id"])
        remote_filename = Path(
            unquote(urlsplit(str(entry["model_url"])).path)
        ).name
        destination = _safe_destination(
            stage,
            "methods",
            str(snapshot["source_revision_tree_sha256"]),
            subject,
            task_id,
            remote_filename,
        )
        _download_verified(
            model_url=str(entry["model_url"]),
            destination=destination,
            expected_size_bytes=int(entry["checkpoint_size_bytes"]),
            expected_sha256=str(entry["checkpoint_sha256"]),
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        )
        method_records.append(
            {
                "index": entry["index"],
                "subject": subject,
                "kind": entry["kind"],
                "training_task_id": task_id,
                "model_id": entry["model_id"],
                "model_name": entry["model_name"],
                "source_revision_tree_sha256": snapshot[
                    "source_revision_tree_sha256"
                ],
                "source_dataset_id": snapshot["source_dataset_id"],
                "source_archive_name": snapshot["source_archive_name"],
                "source_archive_bytes": snapshot["source_archive_bytes"],
                "source_archive_sha256": snapshot["source_archive_sha256"],
                "checkpoint_filename": remote_filename,
                "relative_path": _revision_relative_path(
                    revision,
                    "methods",
                    str(snapshot["source_revision_tree_sha256"]),
                    subject,
                    task_id,
                    remote_filename,
                ),
                "remote_url": entry["model_url"],
                "size_bytes": entry["checkpoint_size_bytes"],
                "sha256": entry["checkpoint_sha256"],
                "checkpoint_policy": CHECKPOINT_POLICY,
                "reused_existing_file": False,
                "verified_at": now,
            }
        )
    manifest = sources["manifest"]
    summary = sources["summary"]
    quality_gate = sources["quality_gate"]
    if not all(
        isinstance(value, Mapping) for value in (manifest, summary, quality_gate)
    ):
        raise RuntimeError("captured sealed collection evidence is invalid")
    return _registry_document(
        {
            "collection_id": COLLECTION_ID,
            "status": "complete",
            "controller_task_id": controller_task_id,
            "quality_gate_task_id": quality_gate_task_id,
            "protocol_id": PROTOCOL_ID,
            **_mixed_source_registry_fields(),
            "training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
            "training_seed": sources["training_seed"],
            "training_seed_evidence": sources["training_seed_evidence"],
            "checkpoint_policy": CHECKPOINT_POLICY,
            "output_directory": str(root),
            "revision": revision,
            "revision_directory": (Path("revisions") / revision).as_posix(),
            "formal_manifest_seal_sha256": manifest["seal_sha256"],
            "training_summary_seal_sha256": summary["seal_sha256"],
            "quality_gate_content_sha256": quality_gate["content_sha256"],
            "teacher": teacher_record,
            "teacher_model_count": 1,
            "epoch50_method_model_count": len(method_records),
            "model_count": 1 + len(method_records),
            "total_size_bytes": int(teacher_record["size_bytes"])
            + sum(int(item["size_bytes"]) for item in method_records),
            "method_count": len(method_records),
            "methods": method_records,
            "downloaded_at": now,
        }
    )


def _revalidate_captured_sources(
    *,
    controller: object,
    quality_gate_task: object,
    controller_task_id: str,
    quality_gate_task_id: str,
    sources: Mapping[str, object],
    auth_header_provider: Callable[[], Mapping[str, str]],
    auth_refresher: Callable[[], None],
) -> None:
    training_sources = sources["training_sources"]
    mappings = (
        sources["controller_parameters"],
        sources["quality_gate_parameters"],
        sources["teacher_parameters"],
        sources["manifest"],
        sources["summary"],
        sources["quality_gate"],
        sources["teacher_reference"],
    )
    if not isinstance(training_sources, list) or not all(
        isinstance(value, Mapping) for value in mappings
    ):
        raise RuntimeError("captured remote source snapshot is invalid")
    _revalidate_remote_snapshot(
        controller=controller,
        quality_gate_task=quality_gate_task,
        teacher_task=sources["teacher_task"],
        controller_task_id=controller_task_id,
        quality_gate_task_id=quality_gate_task_id,
        controller_parameters=mappings[0],
        quality_gate_parameters=mappings[1],
        teacher_parameters=mappings[2],
        manifest=mappings[3],
        summary=mappings[4],
        quality_gate=mappings[5],
        teacher_reference=mappings[6],
        training_seed=int(sources["training_seed"]),
        training_sources=training_sources,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )


def _collect_locked(
    *,
    root: Path,
    registry_path: Path,
    controller: object,
    quality_gate_task: object,
    controller_task_id: str,
    quality_gate_task_id: str,
    sources: Mapping[str, object],
    auth_header_provider: Callable[[], Mapping[str, str]],
    auth_refresher: Callable[[], None],
) -> dict[str, object]:
    manifest = sources["manifest"]
    summary = sources["summary"]
    quality_gate = sources["quality_gate"]
    teacher_reference = sources["teacher_reference"]
    entries = sources["entries"]
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(summary, Mapping)
        or not isinstance(quality_gate, Mapping)
        or not isinstance(teacher_reference, Mapping)
        or not isinstance(entries, list)
    ):
        raise RuntimeError("captured collection evidence is invalid")
    revision = _sha256(manifest.get("seal_sha256"), "formal manifest seal")
    revisions_root = _safe_output_root(root / "revisions")
    revision_dir = revisions_root / revision
    for child in revisions_root.iterdir():
        if child.name != revision:
            raise RuntimeError("formal model output contains an unexpected revision")
    if revision_dir.exists():
        if revision_dir.is_symlink() or not revision_dir.is_dir():
            _quarantine_root_completion(root)
            raise RuntimeError("formal model revision is not a regular directory")
        try:
            registry = _validate_registry_against_sources(
                _read_json_object(
                    revision_dir / "manifest.json", context="revision manifest"
                ),
                root=root,
                revision=revision,
                controller_task_id=controller_task_id,
                quality_gate_task_id=quality_gate_task_id,
                manifest=manifest,
                summary=summary,
                quality_gate=quality_gate,
                teacher_reference=teacher_reference,
                entries=entries,
            )
            _validate_revision_files(
                root=root,
                revision_dir=revision_dir,
                registry=registry,
                require_metadata=True,
                check_entire_root=True,
            )
        except Exception:
            _quarantine_root_completion(root)
            raise
        if _root_completion_exists(root) and not _root_completion_matches_registry(
            root=root, registry=registry
        ):
            _quarantine_root_completion(root)
        _revalidate_captured_sources(
            controller=controller,
            quality_gate_task=quality_gate_task,
            controller_task_id=controller_task_id,
            quality_gate_task_id=quality_gate_task_id,
            sources=sources,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        )
        _validate_revision_files(
            root=root,
            revision_dir=revision_dir,
            registry=registry,
            require_metadata=True,
            check_entire_root=True,
        )
        _publish_completion(root=root, registry_path=registry_path, registry=registry)
        return registry
    _quarantine_root_completion(root)
    if any(path.is_file() or path.is_symlink() for path in root.rglob("*.pth")):
        raise RuntimeError("formal model output contains unexpected .pth files")
    expected_model_bytes = int(teacher_reference["checkpoint_size_bytes"]) + sum(
        int(entry["checkpoint_size_bytes"]) for entry in entries
    )
    staging_parent = _safe_output_root(root.parent / f".{root.name}.staging")
    if staging_parent.stat().st_dev != revisions_root.stat().st_dev:
        raise RuntimeError(
            "model staging and revision directories must share one filesystem"
        )
    _require_collection_space(staging_parent, expected_model_bytes)
    stage: Path | None = Path(
        tempfile.mkdtemp(prefix=f"{revision}.", dir=staging_parent)
    ).resolve(strict=True)
    try:
        registry = _build_staged_registry(
            root=root,
            stage=stage,
            revision=revision,
            controller_task_id=controller_task_id,
            quality_gate_task_id=quality_gate_task_id,
            sources=sources,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        )
        registry = _validate_registry_against_sources(
            registry,
            root=root,
            revision=revision,
            controller_task_id=controller_task_id,
            quality_gate_task_id=quality_gate_task_id,
            manifest=manifest,
            summary=summary,
            quality_gate=quality_gate,
            teacher_reference=teacher_reference,
            entries=entries,
        )
        _validate_revision_files(
            root=root,
            revision_dir=stage,
            registry=registry,
            require_metadata=False,
            check_entire_root=False,
        )
        _atomic_write_text(
            stage / "manifest.json",
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        _atomic_write_text(stage / "SHA256SUMS", _registry_checksum_content(registry))
        completion = _completion_document(
            revision=revision,
            registry_content_sha256=str(registry["content_sha256"]),
            model_count=int(registry["model_count"]),
        )
        _atomic_write_text(
            stage / "COMPLETE",
            json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        _fsync_tree(stage)
        _validate_revision_files(
            root=root,
            revision_dir=stage,
            registry=registry,
            require_metadata=True,
            check_entire_root=False,
        )
        _revalidate_captured_sources(
            controller=controller,
            quality_gate_task=quality_gate_task,
            controller_task_id=controller_task_id,
            quality_gate_task_id=quality_gate_task_id,
            sources=sources,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        )
        _validate_revision_files(
            root=root,
            revision_dir=stage,
            registry=registry,
            require_metadata=True,
            check_entire_root=False,
        )
        if revision_dir.exists():
            raise RuntimeError("formal model revision appeared while collecting")
        os.replace(stage, revision_dir)
        stage = None
        _fsync_directory(revisions_root)
        _validate_revision_files(
            root=root,
            revision_dir=revision_dir,
            registry=registry,
            require_metadata=True,
            check_entire_root=True,
        )
        _publish_completion(root=root, registry_path=registry_path, registry=registry)
        return registry
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def collect_formal_models(
    *,
    task_class: object,
    controller_task_id: str,
    quality_gate_task_id: str,
    output_dir: Path,
    registry_path: Path,
    wait: bool = True,
    poll_seconds: float = 60.0,
    timeout_hours: float = 720.0,
    monotonic_clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    auth_header_provider: Callable[[], Mapping[str, str]] | None = None,
    auth_refresher: Callable[[], None] | None = None,
) -> dict[str, object]:
    controller_task_id = _clearml_id(controller_task_id, "controller task")
    quality_gate_task_id = _clearml_id(quality_gate_task_id, "quality gate task")
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise RuntimeError("ClearML Task class cannot resolve tasks")
    auth_header_provider, auth_refresher = _resolve_download_authentication(
        task_class,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    controller = getter(task_id=controller_task_id)
    quality_gate_task = getter(task_id=quality_gate_task_id)
    deadline = monotonic_clock() + timeout_hours * 3600.0
    if wait:
        _wait_for_completed(
            quality_gate_task,
            context="teacher quality gate",
            deadline=deadline,
            poll_seconds=poll_seconds,
            monotonic_clock=monotonic_clock,
            sleeper=sleeper,
        )
        _wait_for_completed(
            controller,
            context="formal training controller",
            deadline=deadline,
            poll_seconds=poll_seconds,
            monotonic_clock=monotonic_clock,
            sleeper=sleeper,
        )
    elif (
        _normalized_status(quality_gate_task) != "completed"
        or _normalized_status(controller) != "completed"
    ):
        raise RuntimeError("quality gate and controller must both be completed")
    sources = _capture_collection_sources(
        getter=getter,
        controller=controller,
        quality_gate_task=quality_gate_task,
        controller_task_id=controller_task_id,
        quality_gate_task_id=quality_gate_task_id,
        auth_header_provider=auth_header_provider,
        auth_refresher=auth_refresher,
    )
    root = _safe_output_root(output_dir)
    registry_path = _isolated_registry_destination(root=root, path=registry_path)
    with _collection_lock(root):
        return _collect_locked(
            root=root,
            registry_path=registry_path,
            controller=controller,
            quality_gate_task=quality_gate_task,
            controller_task_id=controller_task_id,
            quality_gate_task_id=quality_gate_task_id,
            sources=sources,
            auth_header_provider=auth_header_provider,
            auth_refresher=auth_refresher,
        )


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-controller-task-id", required=True)
    parser.add_argument("--teacher-quality-gate-task-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--poll-seconds", type=_positive_float, default=60.0)
    parser.add_argument("--timeout-hours", type=_positive_float, default=720.0)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="fail immediately unless the quality gate and controller are completed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from clearml import Task

    collect_formal_models(
        task_class=Task,
        controller_task_id=args.training_controller_task_id,
        quality_gate_task_id=args.teacher_quality_gate_task_id,
        output_dir=args.output_dir,
        registry_path=args.registry_path,
        wait=not args.no_wait,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REGISTRY_PATH",
    "SUBJECT_ORDER",
    "collect_formal_models",
    "main",
    "validate_formal_manifest",
)
