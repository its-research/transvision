#!/usr/bin/env python3
"""Persistent history-only adapter subprocess controller.

The controller separates evaluation histories from future targets at the
process protocol boundary.  It is a causal data-isolation mechanism: it does
not sandbox malicious Python code from the host operating system.  An adapter
can still open host files or create child processes unless a deployment adds
container, mount, user, syscall, and network restrictions.

The worker receives no dataset root.  Inference messages contain exactly a
sample identity, a history object, and the SHA-256 digest of that history.
Future targets are sent only after every expected prediction has been received
in order and its digest has been recomputed by this controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "RTPV2X-HISTORY-ADAPTER-v1"
ISOLATION_CONTRACT_ID = "RTPV2X-HISTORY-ONLY-SUBPROCESS-v1"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_STRING_BYTES = 2 * 1024 * 1024
MAX_JSON_INTEGER = 2**63 - 1
STDERR_CAPTURE_LIMIT = 256 * 1024

REQUEST_FIELDS = frozenset({"protocol", "seq", "op", "payload", "payload_sha256"})
RESPONSE_FIELDS = frozenset(
    {"protocol", "seq", "op", "ok", "payload", "payload_sha256"}
)

WORKER_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUDA_HOME",
        "NVIDIA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
    }
)
MANAGED_WORKER_ENV = frozenset(
    {
        "TMPDIR",
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
        "__CF_USER_TEXT_ENCODING",
    }
)

_SENSITIVE_ENV_NAME = re.compile(
    r"(?:CLEARML|TRAINS|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION|"
    r"BEARER|API(?:_|-)?KEY|ACCESS(?:_|-)?KEY|PRIVATE(?:_|-)?KEY)",
    re.IGNORECASE,
)
_SENSITIVE_KEY = re.compile(
    r"^(?:clearml|trains|token|secret|password|passwd|credential|authorization|"
    r"bearer|api_?key|access_?key|private_?key)(?:_|$)",
    re.IGNORECASE,
)
_SENSITIVE_TEXT = re.compile(
    r"(?i)\b(clearml(?:_[a-z0-9_-]+)?|trains(?:_[a-z0-9_-]+)?|api[_-]?key|"
    r"access[_-]?key|private[_-]?key|token|secret|password|passwd|authorization|"
    r"bearer)\b(?:\s*[:=]\s*|\s+)([^\s,;]+)"
)
_FACTORY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

_DATA_ROOT_KEYS = frozenset(
    {
        "dataset_root",
        "data_root",
        "dataset_path",
        "data_mount",
        "dataset_mount",
        "mount_root",
    }
)
_FUTURE_KEYS = frozenset(
    {
        "answer",
        "future",
        "futures",
        "gt",
        "ground_truth",
        "groundtruth",
        "label",
        "targets",
        "labels",
        "supervision",
        "truth",
        "future_positions",
        "future_trajectory",
        "target_positions",
        "target_trajectory",
        "target_timestamps",
    }
)


class HistoryAdapterError(RuntimeError):
    """Base class for controller and worker contract failures."""


class ProtocolValidationError(HistoryAdapterError):
    """A JSON value or protocol envelope violates the frozen contract."""


class ProtocolStateError(HistoryAdapterError):
    """An operation was attempted outside its permitted phase."""


class ProtocolPollutionError(HistoryAdapterError):
    """Worker stdout contained anything other than the expected response."""


class RemoteAdapterError(HistoryAdapterError):
    """The worker reported a sanitized adapter or protocol failure."""


class WorkerTimeoutError(HistoryAdapterError):
    """The worker did not return one complete protocol response in time."""


class WorkerExitError(HistoryAdapterError):
    """The worker exited unexpectedly or with a non-zero status."""


class SessionState(str, Enum):
    NEW = "new"
    LOADED = "loaded"
    ENVIRONMENT_REPORTED = "environment_reported"
    TRAINED = "trained"
    INFERENCING = "inferencing"
    INFERENCE_COMPLETE = "inference_complete"
    EVALUATED = "evaluated"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CLOSED = "closed"
    FAILED = "failed"
    ABORTED = "aborted"


def _duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProtocolValidationError(f"non-finite JSON constant is forbidden: {value}")


def validate_finite_json(
    value: Any, label: str = "JSON value", *, _depth: int = 0
) -> Any:
    """Validate a bounded, finite, ordinary JSON value without coercion."""

    if _depth > MAX_JSON_DEPTH:
        raise ProtocolValidationError(f"{label} exceeds the maximum JSON nesting depth")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_JSON_INTEGER:
            raise ProtocolValidationError(f"{label} contains an out-of-range integer")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolValidationError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES:
            raise ProtocolValidationError(f"{label} contains an oversized string")
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_json(item, f"{label}[{index}]", _depth=_depth + 1)
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolValidationError(
                    f"{label} contains a non-string object key"
                )
            validate_finite_json(key, f"{label} key", _depth=_depth + 1)
            validate_finite_json(item, f"{label}.{key}", _depth=_depth + 1)
        return value
    raise ProtocolValidationError(
        f"{label} contains a non-JSON value: {type(value).__name__}"
    )


def strict_json_loads(data: bytes | str, label: str = "protocol message") -> Any:
    if isinstance(data, bytes):
        if len(data) > MAX_MESSAGE_BYTES:
            raise ProtocolValidationError(f"{label} exceeds the message-size limit")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolValidationError(f"{label} is not UTF-8") from exc
    else:
        text = data
        if len(text.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ProtocolValidationError(f"{label} exceeds the message-size limit")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_object,
            parse_constant=_reject_constant,
        )
    except ProtocolValidationError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ProtocolValidationError(f"{label} is not strict JSON") from exc
    return validate_finite_json(value, label)


def canonical_json_bytes(value: Any) -> bytes:
    validate_finite_json(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProtocolValidationError(
            "value cannot be encoded as canonical JSON"
        ) from exc
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ProtocolValidationError("canonical JSON exceeds the message-size limit")
    return encoded


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def canonical_json_line_sha256(value: Any) -> str:
    """Hash one value using the TFD canonical-JSON-line convention."""

    return sha256_bytes(canonical_json_bytes(value) + b"\n")


def history_input_sha256(value: Any) -> str:
    return canonical_json_line_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ProtocolValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ProtocolValidationError(f"{label} must be a bounded non-empty string")
    return value


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{label} must be a JSON object")
    return value


def require_exact_fields(
    value: Mapping[str, Any], fields: set[str] | frozenset[str], label: str
) -> None:
    if set(value) != set(fields):
        raise ProtocolValidationError(
            f"{label} fields must be exactly {sorted(fields)}"
        )


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def reject_sensitive_keys(value: Any, label: str = "payload") -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_sensitive_keys(item, f"{label}[{index}]")
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        normalized = _normalized_key(key)
        if _SENSITIVE_KEY.match(normalized):
            raise ProtocolValidationError(f"{label} contains a sensitive field name")
        reject_sensitive_keys(item, f"{label}.{key}")


def reject_dataset_roots(value: Any, label: str = "adapter context") -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_dataset_roots(item, f"{label}[{index}]")
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        normalized = _normalized_key(key)
        if normalized in _DATA_ROOT_KEYS:
            raise ProtocolValidationError(
                f"{label} may not contain a dataset-root field"
            )
        reject_dataset_roots(item, f"{label}.{key}")


def reject_host_paths(value: Any, label: str = "adapter context") -> None:
    """Reject host filesystem paths from data-plane JSON values."""

    if isinstance(value, str):
        if value.startswith(("/", "~/")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ProtocolValidationError(
                f"{label} may not contain a host filesystem path"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_host_paths(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            reject_host_paths(item, f"{label}.{key}")


def reject_future_fields(value: Any, label: str = "inference sample") -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_future_fields(item, f"{label}[{index}]")
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        normalized = _normalized_key(key)
        if (
            normalized in _FUTURE_KEYS
            or normalized.startswith("future_")
            or normalized.endswith("_future")
            or normalized.startswith("ground_truth_")
            or (normalized.startswith("target_") and normalized != "target_id")
            or normalized.endswith("_target")
        ):
            raise ProtocolValidationError(
                f"{label} contains a forbidden future/target field"
            )
        reject_future_fields(item, f"{label}.{key}")


def validate_history_sample(
    value: Any, label: str = "inference sample"
) -> dict[str, Any]:
    sample = require_object(validate_finite_json(value, label), label)
    reject_future_fields(sample, label)
    reject_sensitive_keys(sample, label)
    reject_host_paths(sample, label)
    require_exact_fields(
        sample,
        {"scene_id", "target_id", "history", "input_sha256"},
        label,
    )
    require_identifier(sample["scene_id"], f"{label}.scene_id")
    require_identifier(sample["target_id"], f"{label}.target_id")
    history = require_object(sample["history"], f"{label}.history")
    if not history:
        raise ProtocolValidationError(f"{label}.history must be non-empty")
    expected = require_digest(sample["input_sha256"], f"{label}.input_sha256")
    if history_input_sha256(history) != expected:
        raise ProtocolValidationError(
            f"{label}.input_sha256 does not match history bytes"
        )
    return sample


def validate_training_sample(
    value: Any, label: str = "training sample"
) -> dict[str, Any]:
    sample = require_object(validate_finite_json(value, label), label)
    require_exact_fields(
        sample,
        {"scene_id", "target_id", "history", "input_sha256", "ground_truth"},
        label,
    )
    require_identifier(sample["scene_id"], f"{label}.scene_id")
    require_identifier(sample["target_id"], f"{label}.target_id")
    history = require_object(sample["history"], f"{label}.history")
    ground_truth = require_object(sample["ground_truth"], f"{label}.ground_truth")
    if not history or not ground_truth:
        raise ProtocolValidationError(
            f"{label} history and ground_truth must be non-empty"
        )
    reject_sensitive_keys(sample, label)
    expected = require_digest(sample["input_sha256"], f"{label}.input_sha256")
    if history_input_sha256(history) != expected:
        raise ProtocolValidationError(
            f"{label}.input_sha256 does not match history bytes"
        )
    return sample


def validate_evaluation_target(
    value: Any, label: str = "evaluation target"
) -> dict[str, Any]:
    target = require_object(validate_finite_json(value, label), label)
    require_exact_fields(
        target,
        {"scene_id", "target_id", "ground_truth", "input_sha256"},
        label,
    )
    require_identifier(target["scene_id"], f"{label}.scene_id")
    require_identifier(target["target_id"], f"{label}.target_id")
    require_object(target["ground_truth"], f"{label}.ground_truth")
    require_digest(target["input_sha256"], f"{label}.input_sha256")
    reject_sensitive_keys(target, label)
    return target


def build_worker_environment(
    source: Mapping[str, str], temporary_directory: Path
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return the allowlisted worker environment and stripped secret values."""

    environment: dict[str, str] = {}
    sensitive_values: list[str] = []
    for name, value in source.items():
        if _SENSITIVE_ENV_NAME.search(name):
            if value:
                sensitive_values.append(value)
            continue
        # Ambient PYTHONPATH is intentionally not forwarded.  The worker runs
        # under -I and inserts only its sealed source directory plus the
        # adapter's package-sealed directory.
        if name == "PYTHONPATH":
            continue
        if name in WORKER_ENV_ALLOWLIST and isinstance(value, str):
            environment[name] = value
    environment.update(
        {
            "TMPDIR": str(temporary_directory),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment, tuple(sorted(set(sensitive_values), key=len, reverse=True))


def default_isolation_contract_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "clearml"
        / "protocols"
        / "history-only-subprocess-v1.json"
    )


def validate_isolation_contract(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    if path.is_symlink():
        raise ProtocolValidationError("isolation contract may not be a symlink")
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ProtocolValidationError("isolation contract must be a regular file")
    observed = sha256_file(resolved)
    if expected_sha256 is not None and observed != require_digest(
        expected_sha256, "expected isolation contract SHA-256"
    ):
        raise ProtocolValidationError("isolation contract SHA-256 mismatch")
    try:
        document = strict_json_loads(resolved.read_bytes(), "isolation contract")
    except OSError as exc:
        raise ProtocolValidationError("cannot read isolation contract") from exc
    contract = require_object(document, "isolation contract")
    required = {
        "schema_version",
        "contract_id",
        "status",
        "scientific_claim_allowed",
        "os_sandbox",
        "guarantee",
        "not_guaranteed",
        "implementation",
        "transport",
        "state_machine",
        "payload_boundaries",
        "environment",
        "failure_policy",
    }
    require_exact_fields(contract, required, "isolation contract")
    if contract["schema_version"] != 1:
        raise ProtocolValidationError("isolation contract schema_version must be 1")
    if contract["contract_id"] != ISOLATION_CONTRACT_ID:
        raise ProtocolValidationError("isolation contract identity mismatch")
    if contract["status"] != "implemented_causal_data_isolation":
        raise ProtocolValidationError("isolation contract status is not implemented")
    if contract["scientific_claim_allowed"] is not False:
        raise ProtocolValidationError(
            "isolation contract cannot allow scientific claims"
        )
    if contract["os_sandbox"] is not False:
        raise ProtocolValidationError("isolation contract cannot claim an OS sandbox")
    boundaries = require_object(contract["payload_boundaries"], "payload_boundaries")
    if boundaries.get("inference_fields") != [
        "scene_id",
        "target_id",
        "history",
        "input_sha256",
    ]:
        raise ProtocolValidationError(
            "isolation contract inference fields have drifted"
        )
    if boundaries.get("dataset_root_transmitted") is not False:
        raise ProtocolValidationError(
            "isolation contract must forbid dataset-root transmission"
        )
    environment = require_object(contract["environment"], "isolation environment")
    if environment.get("allowlist") != sorted(WORKER_ENV_ALLOWLIST):
        raise ProtocolValidationError("isolation environment allowlist has drifted")
    return contract


@dataclass(frozen=True)
class TrainingRecord:
    backward_report: dict[str, Any]
    optimizer_report: dict[str, Any]
    response_payload_sha256: str


@dataclass(frozen=True)
class PredictionRecord:
    index: int
    scene_id: str
    target_id: str
    input_sha256: str
    prediction: dict[str, Any]
    prediction_sha256: str
    response_payload_sha256: str

    def confirmation(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "scene_id": self.scene_id,
            "target_id": self.target_id,
            "input_sha256": self.input_sha256,
            "prediction_sha256": self.prediction_sha256,
        }


@dataclass(frozen=True)
class EvaluationRecord:
    metrics: dict[str, Any]
    prediction_set_sha256: str
    target_set_sha256: str
    response_payload_sha256: str


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    size_bytes: int
    sha256: str
    response_payload_sha256: str


class HistoryOnlyAdapterProcess:
    """Own one persistent, fail-closed history-only adapter worker."""

    def __init__(
        self,
        *,
        adapter_path: Path,
        factory: str,
        context: Mapping[str, Any],
        expected_inference_count: int,
        timeout_seconds: float = 30.0,
        python_executable: Path | None = None,
        worker_path: Path | None = None,
        isolation_contract_path: Path | None = None,
        expected_isolation_contract_sha256: str | None = None,
        source_environment: Mapping[str, str] | None = None,
    ) -> None:
        if (
            isinstance(expected_inference_count, bool)
            or not isinstance(expected_inference_count, int)
            or expected_inference_count <= 0
        ):
            raise ProtocolValidationError(
                "expected_inference_count must be a positive integer"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ProtocolValidationError("timeout_seconds must be finite and positive")
        if not isinstance(factory, str) or _FACTORY_NAME.fullmatch(factory) is None:
            raise ProtocolValidationError("adapter factory must be a Python identifier")

        resolved_adapter = Path(adapter_path).resolve()
        if (
            not resolved_adapter.is_file()
            or Path(adapter_path).is_symlink()
            or resolved_adapter.is_symlink()
        ):
            raise ProtocolValidationError(
                "adapter_path must be a non-symlink regular file"
            )
        resolved_worker = (
            Path(worker_path).resolve()
            if worker_path is not None
            else Path(__file__).with_name("history_adapter_worker.py").resolve()
        )
        if (
            not resolved_worker.is_file()
            or (worker_path is not None and Path(worker_path).is_symlink())
            or resolved_worker.is_symlink()
        ):
            raise ProtocolValidationError(
                "worker_path must be a non-symlink regular file"
            )
        resolved_python = Path(python_executable or sys.executable).resolve()
        if not resolved_python.is_file():
            raise ProtocolValidationError("python_executable must resolve to a file")

        detached_context = strict_json_loads(
            canonical_json_bytes(dict(context)), "adapter context"
        )
        reject_dataset_roots(detached_context)
        reject_sensitive_keys(detached_context, "adapter context")
        reject_host_paths(detached_context, "adapter context")
        contract_path = Path(
            isolation_contract_path or default_isolation_contract_path()
        )
        self._isolation_contract = validate_isolation_contract(
            contract_path,
            expected_isolation_contract_sha256,
        )

        self.adapter_path = resolved_adapter
        self.adapter_sha256 = sha256_file(resolved_adapter)
        self.factory = factory
        self.context = detached_context
        self.context_sha256 = sha256_json(detached_context)
        self.expected_inference_count = expected_inference_count
        self.timeout_seconds = float(timeout_seconds)
        self.python_executable = resolved_python
        self.worker_path = resolved_worker
        self.worker_sha256 = sha256_file(resolved_worker)
        self.parent_source_sha256 = sha256_file(Path(__file__).resolve())
        self.python_executable_sha256 = sha256_file(resolved_python)
        self.isolation_contract_path = contract_path.resolve()
        self.isolation_contract_sha256 = sha256_file(self.isolation_contract_path)
        self._source_environment = dict(
            os.environ if source_environment is None else source_environment
        )

        self._state = SessionState.NEW
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._working_directory: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._selector: selectors.BaseSelector | None = None
        self._stdout_buffer = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lock = threading.Lock()
        self._stderr_buffer = bytearray()
        self._stderr_total = 0
        self._stderr_digest = hashlib.sha256()
        self._sensitive_values: tuple[str, ...] = ()
        self._sequence = 0
        self._predictions: list[PredictionRecord] = []
        self._transcript_records: list[dict[str, Any]] = []
        self._transcript_chain = bytes(32)

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def transcript_sha256(self) -> str:
        return self._transcript_chain.hex()

    @property
    def transcript_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._transcript_records)

    @property
    def predictions(self) -> tuple[PredictionRecord, ...]:
        return tuple(self._predictions)

    @property
    def stderr_summary(self) -> dict[str, Any]:
        with self._stderr_lock:
            return {
                "byte_count": self._stderr_total,
                "sha256": self._stderr_digest.copy().hexdigest(),
                "truncated": self._stderr_total > len(self._stderr_buffer),
            }

    @property
    def redacted_stderr(self) -> str:
        with self._stderr_lock:
            text = bytes(self._stderr_buffer).decode("utf-8", errors="replace")
        for secret in self._sensitive_values:
            if len(secret) >= 4:
                text = text.replace(secret, "<redacted>")
        return _SENSITIVE_TEXT.sub(lambda match: f"{match.group(1)}=<redacted>", text)

    def _ensure_state(self, *allowed: SessionState) -> None:
        if self._state not in allowed:
            names = ", ".join(state.value for state in allowed)
            raise ProtocolStateError(
                f"operation requires state [{names}], observed {self._state.value}"
            )

    def start(self) -> "HistoryOnlyAdapterProcess":
        self._ensure_state(SessionState.NEW)
        self._temporary = tempfile.TemporaryDirectory(prefix="rtpv2x-history-adapter-")
        self._working_directory = Path(self._temporary.name).resolve()
        self._working_directory.chmod(0o700)
        environment, secrets = build_worker_environment(
            self._source_environment, self._working_directory
        )
        self._sensitive_values = secrets
        try:
            bootstrap = (
                "import runpy,sys;"
                "sys.path.insert(0,sys.argv[1]);"
                "runpy.run_path(sys.argv[2],run_name='__main__')"
            )
            self._process = subprocess.Popen(
                [
                    str(self.python_executable),
                    "-I",
                    "-B",
                    "-u",
                    "-c",
                    bootstrap,
                    str(self.worker_path.parent),
                    str(self.worker_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._working_directory,
                env=environment,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            self._cleanup_process()
            self._state = SessionState.FAILED
            raise WorkerExitError("cannot start history-only adapter worker") from exc
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="rtpv2x-history-adapter-stderr",
            daemon=True,
        )
        self._stderr_thread.start()
        load_payload = {
            "adapter_path": str(self.adapter_path),
            "adapter_sha256": self.adapter_sha256,
            "factory": self.factory,
            "context": self.context,
            "expected_inference_count": self.expected_inference_count,
            "isolation_contract_id": ISOLATION_CONTRACT_ID,
            "isolation_contract_sha256": self.isolation_contract_sha256,
            "parent_source_sha256": self.parent_source_sha256,
            "worker_source_sha256": self.worker_sha256,
        }
        response, response_hash = self._request("load", load_payload)
        require_exact_fields(
            response,
            {
                "loaded",
                "adapter_sha256",
                "context_sha256",
                "expected_inference_count",
                "isolation_contract_sha256",
                "parent_source_sha256",
                "worker_source_sha256",
            },
            "load response",
        )
        expected = {
            "loaded": True,
            "adapter_sha256": self.adapter_sha256,
            "context_sha256": self.context_sha256,
            "expected_inference_count": self.expected_inference_count,
            "isolation_contract_sha256": self.isolation_contract_sha256,
            "parent_source_sha256": self.parent_source_sha256,
            "worker_source_sha256": self.worker_sha256,
        }
        mismatched_bindings = sorted(
            key for key, value in expected.items() if response.get(key) != value
        )
        if response_hash != sha256_json(response):
            mismatched_bindings.append("response_payload_sha256")
        if mismatched_bindings:
            self._fail(
                ProtocolValidationError(
                    "load response binding mismatch: " + ", ".join(mismatched_bindings)
                )
            )
        self._state = SessionState.LOADED
        return self

    def runtime_environment(self) -> dict[str, Any]:
        self._ensure_state(SessionState.LOADED)
        response, _ = self._request("runtime_environment", {})
        try:
            require_exact_fields(response, {"adapter", "worker"}, "runtime response")
            adapter = require_object(response["adapter"], "adapter runtime environment")
            worker = require_object(response["worker"], "worker runtime environment")
            require_exact_fields(
                worker,
                {
                    "python_version",
                    "python_implementation",
                    "python_cache_tag",
                    "python_executable_basename",
                    "python_executable_sha256",
                    "platform",
                    "environment_keys",
                    "adapter_sha256",
                    "parent_source_sha256",
                    "worker_source_sha256",
                    "isolation_contract_sha256",
                    "context_sha256",
                    "temporary_cwd",
                },
                "worker runtime environment",
            )
            keys = worker["environment_keys"]
            if not isinstance(keys, list) or keys != sorted(set(keys)):
                raise ProtocolValidationError(
                    "worker environment keys are not sorted and unique"
                )
            if not set(keys).issubset(WORKER_ENV_ALLOWLIST | MANAGED_WORKER_ENV):
                raise ProtocolValidationError(
                    "worker reports a non-allowlisted environment key"
                )
            if any(_SENSITIVE_ENV_NAME.search(str(key)) for key in keys):
                raise ProtocolValidationError(
                    "worker reports a sensitive environment key"
                )
            bindings = {
                "python_executable_basename": self.python_executable.name,
                "python_executable_sha256": self.python_executable_sha256,
                "adapter_sha256": self.adapter_sha256,
                "parent_source_sha256": self.parent_source_sha256,
                "worker_source_sha256": self.worker_sha256,
                "isolation_contract_sha256": self.isolation_contract_sha256,
                "context_sha256": self.context_sha256,
                "temporary_cwd": True,
            }
            if any(worker.get(key) != value for key, value in bindings.items()):
                raise ProtocolValidationError("worker runtime binding mismatch")
            reject_sensitive_keys(adapter, "adapter runtime environment")
        except HistoryAdapterError as exc:
            self._fail(exc)
        self._state = SessionState.ENVIRONMENT_REPORTED
        return response

    def train(self, sample: Mapping[str, Any]) -> TrainingRecord:
        self._ensure_state(SessionState.ENVIRONMENT_REPORTED)
        try:
            normalized = validate_training_sample(dict(sample))
        except HistoryAdapterError as exc:
            self._fail(exc)
        response, response_hash = self._request("train", {"sample": normalized})
        try:
            require_exact_fields(
                response, {"backward_report", "optimizer_report"}, "train response"
            )
            backward = require_object(response["backward_report"], "backward report")
            optimizer = require_object(response["optimizer_report"], "optimizer report")
        except HistoryAdapterError as exc:
            self._fail(exc)
        self._state = SessionState.TRAINED
        return TrainingRecord(backward, optimizer, response_hash)

    def infer(self, sample: Mapping[str, Any]) -> PredictionRecord:
        self._ensure_state(SessionState.TRAINED, SessionState.INFERENCING)
        if len(self._predictions) >= self.expected_inference_count:
            raise ProtocolStateError(
                "all expected inference samples have already completed"
            )
        try:
            normalized = validate_history_sample(dict(sample))
        except HistoryAdapterError as exc:
            self._fail(exc)
        expected_index = len(self._predictions)
        response, response_hash = self._request(
            "infer", {"index": expected_index, "sample": normalized}
        )
        try:
            require_exact_fields(
                response,
                {
                    "index",
                    "scene_id",
                    "target_id",
                    "input_sha256",
                    "prediction",
                    "prediction_sha256",
                },
                "infer response",
            )
            if response["index"] != expected_index:
                raise ProtocolValidationError(
                    "inference response index is out of order"
                )
            if (
                response["scene_id"] != normalized["scene_id"]
                or response["target_id"] != normalized["target_id"]
            ):
                raise ProtocolValidationError("inference response identity mismatch")
            if response["input_sha256"] != normalized["input_sha256"]:
                raise ProtocolValidationError(
                    "inference response input digest mismatch"
                )
            prediction = require_object(response["prediction"], "prediction")
            observed_prediction_hash = canonical_json_line_sha256(prediction)
            if (
                require_digest(response["prediction_sha256"], "prediction_sha256")
                != observed_prediction_hash
            ):
                raise ProtocolValidationError("prediction SHA-256 mismatch")
            reject_sensitive_keys(prediction, "prediction")
        except HistoryAdapterError as exc:
            self._fail(exc)
        record = PredictionRecord(
            index=expected_index,
            scene_id=normalized["scene_id"],
            target_id=normalized["target_id"],
            input_sha256=normalized["input_sha256"],
            prediction=prediction,
            prediction_sha256=observed_prediction_hash,
            response_payload_sha256=response_hash,
        )
        self._predictions.append(record)
        self._state = (
            SessionState.INFERENCE_COMPLETE
            if len(self._predictions) == self.expected_inference_count
            else SessionState.INFERENCING
        )
        return record

    def evaluate(self, targets: Sequence[Mapping[str, Any]]) -> EvaluationRecord:
        self._ensure_state(SessionState.INFERENCE_COMPLETE)
        if len(self._predictions) != self.expected_inference_count:
            raise ProtocolStateError("not all predictions have been confirmed")
        try:
            for record in self._predictions:
                if (
                    canonical_json_line_sha256(record.prediction)
                    != record.prediction_sha256
                ):
                    raise ProtocolValidationError(
                        "a confirmed prediction was mutated before evaluation"
                    )
            normalized_targets = [
                validate_evaluation_target(dict(target), f"evaluation target {index}")
                for index, target in enumerate(targets)
            ]
            if len(normalized_targets) != self.expected_inference_count:
                raise ProtocolValidationError(
                    "evaluation target count does not match confirmed predictions"
                )
            for target, prediction in zip(normalized_targets, self._predictions):
                if (
                    target["scene_id"] != prediction.scene_id
                    or target["target_id"] != prediction.target_id
                    or target["input_sha256"] != prediction.input_sha256
                ):
                    raise ProtocolValidationError(
                        "evaluation target order or identity does not match predictions"
                    )
            confirmations = [record.confirmation() for record in self._predictions]
        except HistoryAdapterError as exc:
            self._fail(exc)
        response, response_hash = self._request(
            "evaluate",
            {
                "prediction_confirmations": confirmations,
                "targets": normalized_targets,
            },
        )
        try:
            require_exact_fields(
                response,
                {"metrics", "prediction_set_sha256", "target_set_sha256"},
                "evaluate response",
            )
            metrics = require_object(response["metrics"], "evaluation metrics")
            prediction_set_hash = require_digest(
                response["prediction_set_sha256"], "prediction_set_sha256"
            )
            target_set_hash = require_digest(
                response["target_set_sha256"], "target_set_sha256"
            )
            expected_prediction_set_hash = sha256_json(
                [record.prediction for record in self._predictions]
            )
            if prediction_set_hash != expected_prediction_set_hash:
                raise ProtocolValidationError(
                    "evaluation prediction-set digest mismatch"
                )
            if target_set_hash != sha256_json(normalized_targets):
                raise ProtocolValidationError("evaluation target-set digest mismatch")
        except HistoryAdapterError as exc:
            self._fail(exc)
        self._state = SessionState.EVALUATED
        return EvaluationRecord(
            metrics=metrics,
            prediction_set_sha256=prediction_set_hash,
            target_set_sha256=target_set_hash,
            response_payload_sha256=response_hash,
        )

    def save_checkpoint(self, destination: Path) -> CheckpointRecord:
        self._ensure_state(SessionState.EVALUATED)
        target = Path(destination).resolve()
        if target.exists():
            raise ProtocolValidationError(
                "checkpoint destination must not already exist"
            )
        if not target.parent.is_dir():
            raise ProtocolValidationError("checkpoint destination parent must exist")
        if self._working_directory is None:
            raise ProtocolStateError("worker temporary directory is unavailable")
        try:
            target.relative_to(self._working_directory)
        except ValueError:
            pass
        else:
            raise ProtocolValidationError(
                "checkpoint destination must be outside worker tempdir"
            )
        response, response_hash = self._request(
            "save_checkpoint", {"relative_path": "checkpoint.bin"}
        )
        try:
            require_exact_fields(
                response,
                {"relative_path", "size_bytes", "sha256"},
                "checkpoint response",
            )
            if response["relative_path"] != "checkpoint.bin":
                raise ProtocolValidationError(
                    "worker changed the checkpoint relative path"
                )
            size = response["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ProtocolValidationError(
                    "checkpoint size must be a positive integer"
                )
            digest = require_digest(response["sha256"], "checkpoint SHA-256")
            source = self._working_directory / "checkpoint.bin"
            if not source.is_file() or source.is_symlink():
                raise ProtocolValidationError("worker checkpoint is not a regular file")
            if source.stat().st_size != size or sha256_file(source) != digest:
                raise ProtocolValidationError(
                    "worker checkpoint bytes do not match its response"
                )
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with (
                    source.open("rb") as input_stream,
                    os.fdopen(descriptor, "wb") as output_stream,
                ):
                    shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
            except BaseException:
                try:
                    target.unlink()
                except OSError:
                    pass
                raise
            if target.stat().st_size != size or sha256_file(target) != digest:
                target.unlink(missing_ok=True)
                raise ProtocolValidationError(
                    "copied checkpoint failed parent-side verification"
                )
        except (HistoryAdapterError, OSError) as exc:
            error = (
                exc
                if isinstance(exc, HistoryAdapterError)
                else ProtocolValidationError("cannot materialize verified checkpoint")
            )
            self._fail(error)
        self._state = SessionState.CHECKPOINT_SAVED
        return CheckpointRecord(target, size, digest, response_hash)

    def close(self) -> None:
        self._ensure_state(SessionState.CHECKPOINT_SAVED)
        response, _ = self._request("close", {})
        try:
            require_exact_fields(response, {"closed", "final_state"}, "close response")
            if response != {"closed": True, "final_state": "checkpoint_saved"}:
                raise ProtocolValidationError("worker close response state mismatch")
            assert self._process is not None
            try:
                exit_code = self._process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise WorkerTimeoutError("worker did not exit after close") from exc
            if exit_code != 0:
                raise WorkerExitError(
                    f"worker exited non-zero after close: {exit_code}"
                )
            self._assert_stdout_clean_after_exit()
        except HistoryAdapterError as exc:
            self._fail(exc)
        self._state = SessionState.CLOSED
        self._cleanup_process()

    def abort(self) -> None:
        if self._state in {SessionState.CLOSED, SessionState.ABORTED}:
            return
        self._terminate_process()
        self._state = SessionState.ABORTED
        self._cleanup_process()

    def __enter__(self) -> "HistoryOnlyAdapterProcess":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None and self._state == SessionState.CHECKPOINT_SAVED:
            self.close()
        elif self._state not in {SessionState.CLOSED, SessionState.ABORTED}:
            self.abort()

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                chunk = process.stderr.read(65536)
                if not chunk:
                    break
                with self._stderr_lock:
                    self._stderr_total += len(chunk)
                    self._stderr_digest.update(chunk)
                    remaining = STDERR_CAPTURE_LIMIT - len(self._stderr_buffer)
                    if remaining > 0:
                        self._stderr_buffer.extend(chunk[:remaining])
        except (OSError, ValueError):
            return

    def _read_response_line(self, deadline: float) -> bytes:
        if self._process is None or self._selector is None:
            raise WorkerExitError("worker process is unavailable")
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[: newline + 1]
                if not line:
                    raise ProtocolPollutionError(
                        "worker stdout contained an empty protocol line"
                    )
                return line
            if len(self._stdout_buffer) > MAX_MESSAGE_BYTES:
                raise ProtocolPollutionError("worker stdout protocol line is oversized")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeoutError("worker response timed out")
            events = self._selector.select(remaining)
            if not events:
                raise WorkerTimeoutError("worker response timed out")
            assert self._process.stdout is not None
            try:
                chunk = os.read(self._process.stdout.fileno(), 65536)
            except OSError as exc:
                raise WorkerExitError("cannot read worker stdout") from exc
            if not chunk:
                exit_code = self._process.poll()
                if exit_code is None:
                    raise WorkerExitError("worker stdout closed unexpectedly")
                raise WorkerExitError(f"worker exited before response: {exit_code}")
            self._stdout_buffer.extend(chunk)

    def _write_request(self, encoded: bytes) -> None:
        if self._process is None or self._process.stdin is None:
            raise WorkerExitError("worker stdin is unavailable")
        if self._process.poll() is not None:
            raise WorkerExitError(
                f"worker exited before request: {self._process.returncode}"
            )
        try:
            self._process.stdin.write(encoded)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            exit_code = self._process.poll()
            raise WorkerExitError(
                f"cannot write worker request; exit={exit_code}"
            ) from exc

    def _assert_stdout_clean_after_exit(self) -> None:
        """Reject protocol bytes emitted after the final close response."""

        if self._stdout_buffer:
            raise ProtocolPollutionError(
                "worker stdout contained trailing protocol pollution"
            )
        if (
            self._process is None
            or self._process.stdout is None
            or self._selector is None
        ):
            raise WorkerExitError("worker stdout is unavailable after close")
        while self._selector.select(0):
            try:
                chunk = os.read(self._process.stdout.fileno(), 65536)
            except OSError as exc:
                raise WorkerExitError("cannot verify worker stdout EOF") from exc
            if not chunk:
                return
            raise ProtocolPollutionError(
                "worker stdout contained trailing protocol pollution"
            )

    def _request(
        self, op: str, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str]:
        try:
            normalized_payload = require_object(
                strict_json_loads(canonical_json_bytes(dict(payload)), f"{op} payload"),
                f"{op} payload",
            )
            reject_sensitive_keys(normalized_payload, f"{op} payload")
            self._sequence += 1
            request_hash = sha256_json(normalized_payload)
            request = {
                "protocol": PROTOCOL_ID,
                "seq": self._sequence,
                "op": op,
                "payload": normalized_payload,
                "payload_sha256": request_hash,
            }
            encoded = canonical_json_bytes(request) + b"\n"
            if len(encoded) > MAX_MESSAGE_BYTES:
                raise ProtocolValidationError("request exceeds the message-size limit")
            self._write_request(encoded)
            line = self._read_response_line(time.monotonic() + self.timeout_seconds)
            try:
                parsed = strict_json_loads(line, "worker stdout response")
            except ProtocolValidationError as exc:
                raise ProtocolPollutionError(
                    "worker stdout contained non-protocol output"
                ) from exc
            response = require_object(parsed, "worker response")
            require_exact_fields(response, RESPONSE_FIELDS, "worker response")
            if (
                response["protocol"] != PROTOCOL_ID
                or response["seq"] != self._sequence
                or response["op"] != op
                or not isinstance(response["ok"], bool)
            ):
                raise ProtocolPollutionError(
                    "worker response identity or sequence mismatch"
                )
            response_payload = require_object(
                response["payload"], "worker response payload"
            )
            reject_sensitive_keys(response_payload, "worker response payload")
            response_hash = require_digest(
                response["payload_sha256"], "worker response payload_sha256"
            )
            if sha256_json(response_payload) != response_hash:
                raise ProtocolPollutionError("worker response payload SHA-256 mismatch")
            transcript = {
                "seq": self._sequence,
                "op": op,
                "request_payload_sha256": request_hash,
                "response_payload_sha256": response_hash,
                "ok": response["ok"],
            }
            self._transcript_chain = hashlib.sha256(
                self._transcript_chain + canonical_json_bytes(transcript)
            ).digest()
            self._transcript_records.append(transcript)
            if response["ok"] is not True:
                require_exact_fields(
                    response_payload,
                    {"error_code", "error_type", "message"},
                    "worker error payload",
                )
                error_type = require_identifier(
                    response_payload["error_type"], "worker error_type"
                )
                error_code = require_identifier(
                    response_payload["error_code"], "worker error_code"
                )
                raise RemoteAdapterError(
                    f"worker rejected {op}: code={error_code}, type={error_type}"
                )
            return response_payload, response_hash
        except HistoryAdapterError as exc:
            self._fail(exc)
        raise AssertionError("unreachable")

    def _terminate_process(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _cleanup_process(self) -> None:
        selector = self._selector
        if selector is not None:
            try:
                selector.close()
            except OSError:
                pass
            self._selector = None
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        thread = self._stderr_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._stderr_thread = None
        temporary = self._temporary
        if temporary is not None:
            temporary.cleanup()
        self._temporary = None
        self._working_directory = None

    def _fail(self, error: HistoryAdapterError) -> None:
        self._terminate_process()
        self._state = SessionState.FAILED
        self._cleanup_process()
        raise error


__all__ = [
    "CheckpointRecord",
    "EvaluationRecord",
    "HistoryAdapterError",
    "HistoryOnlyAdapterProcess",
    "ISOLATION_CONTRACT_ID",
    "PredictionRecord",
    "PROTOCOL_ID",
    "ProtocolPollutionError",
    "ProtocolStateError",
    "ProtocolValidationError",
    "RemoteAdapterError",
    "SessionState",
    "TrainingRecord",
    "WorkerExitError",
    "WorkerTimeoutError",
    "build_worker_environment",
    "canonical_json_line_sha256",
    "canonical_json_bytes",
    "default_isolation_contract_path",
    "history_input_sha256",
    "reject_future_fields",
    "reject_host_paths",
    "sha256_file",
    "sha256_json",
    "strict_json_loads",
    "validate_finite_json",
    "validate_history_sample",
    "validate_isolation_contract",
]
