#!/usr/bin/env python3
"""Download and verify the fixed ClearML evidence bundle for the synthetic canary.

The module deliberately imports ClearML only inside the runtime adapter.  Its
content and file validators are therefore testable without credentials, network
access, or a ClearML installation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import ipaddress
import io
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import SplitResult, unquote_to_bytes, urlsplit, urlunsplit


PROTOCOL_ID = "SYNTH-CAUSAL-CANARY-v1"
CLEARML_PROJECT = "Thesis/RTP-V2X"
CLEARML_TASK_PREFIX = "rtpv2x__synth-causal-canary-v1__"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TASK_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
TASK_NAME_RE = re.compile(r"rtpv2x__[a-z0-9][a-z0-9_-]*(?:__[a-z0-9][a-z0-9_-]*)+\Z")
PRIVATE_IPV4_RE = re.compile(
    r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.(?:\d{1,3}\.)\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})\b"
)
SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:^|[_-])(?:access[_-]?key|api[_-]?key|authorization|cookie|"
    r"credential|password|passwd|private[_-]?key|secret|session[_-]?key|token)"
    r"(?:$|[_-])"
)
SENSITIVE_TEXT_RE = re.compile(
    r"(?i)(?:\b(?:bearer|basic)\s+[a-z0-9._~+/=-]+|"
    r"(?:access[_-]?key|api[_-]?key|authorization|credential|password|passwd|"
    r"private[_-]?key|secret|token)\s*[:=])"
)
WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)")
POSIX_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s=\"'(:])/(?!/)[^\s,;\"')\]]+")
INTERNAL_HOST_RE = re.compile(
    r"(?i)\b(?:localhost|[a-z0-9-]+\.(?:internal|local|lan))\b"
)
IP_TOKEN_SPLIT_RE = re.compile(r"[\s,;()\[\]{}<>=\"']+")

ARTIFACT_SPECS: dict[str, tuple[str, str, int]] = {
    "metrics": ("metrics.json", ".json", 1024 * 1024),
    "run_manifest": ("run_manifest.json", ".json", 1024 * 1024),
    "events": ("events.jsonl", ".jsonl", 64 * 1024 * 1024),
}
METRIC_METHOD_KEYS = {
    "position_rmse",
    "prediction_ade",
    "prediction_fde",
    "identity_switches",
    "future_messages_consumed",
    "late_messages_consumed",
    "messages_available",
}
EVENT_KEYS = {
    "source",
    "truth_id",
    "event_time",
    "arrival_time",
    "position",
    "velocity",
    "reliability",
}


class EvidenceValidationError(RuntimeError):
    """A fixed-code, non-sensitive validation failure."""


def _fail(code: str) -> None:
    raise EvidenceValidationError(code)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _require_keys(value: Any, expected: set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_sensitive(value: Any) -> None:
    """Reject credentials, URLs, private IPs, and absolute paths recursively."""

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or SENSITIVE_KEY_RE.search(key):
                _fail("sensitive_key")
            assert_no_sensitive(key)
            assert_no_sensitive(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            assert_no_sensitive(item)
        return
    if not isinstance(value, str):
        return
    for token in IP_TOKEN_SPLIT_RE.split(value):
        candidate = token.strip(".!?")
        if "%" in candidate:
            candidate = candidate.split("%", 1)[0]
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        _fail("sensitive_text")
    if (
        "://" in value
        or PRIVATE_IPV4_RE.search(value)
        or SENSITIVE_TEXT_RE.search(value)
        or POSIX_ABSOLUTE_PATH_RE.search(value)
        or WINDOWS_ABSOLUTE_PATH_RE.search(value)
        or INTERNAL_HOST_RE.search(value)
    ):
        _fail("sensitive_text")


def _reject_json_constant(_: str) -> None:
    _fail("non_finite_json_number")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def parse_json_text(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except EvidenceValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        _fail("invalid_json")


def load_json_file(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError):
        _fail("json_read_failed")
    value = parse_json_text(text)
    if not isinstance(value, dict):
        _fail("json_root_not_object")
    assert_no_sensitive(value)
    return value


def validate_metrics(
    value: Mapping[str, Any], expected_protocol: str, require_a100: bool
) -> int:
    root = _require_keys(
        value,
        {
            "protocol_id",
            "scientific_claim_allowed",
            "diagnostic_only",
            "naive",
            "reliability",
            "delta",
            "deterministic_replay",
            "gpu",
        },
        "metrics_schema",
    )
    if root["protocol_id"] != expected_protocol:
        _fail("metrics_protocol")
    if root["scientific_claim_allowed"] is not False:
        _fail("metrics_scientific_claim")
    if root["diagnostic_only"] is not True:
        _fail("metrics_diagnostic_flag")
    if root["deterministic_replay"] is not True:
        _fail("metrics_replay_flag")

    methods: dict[str, Mapping[str, Any]] = {}
    for method in ("naive", "reliability"):
        row = _require_keys(root[method], METRIC_METHOD_KEYS, "metrics_method_schema")
        for key in ("position_rmse", "prediction_ade", "prediction_fde"):
            if not _is_number(row[key]) or float(row[key]) < 0.0:
                _fail("metrics_numeric")
        for key in (
            "identity_switches",
            "future_messages_consumed",
            "late_messages_consumed",
            "messages_available",
        ):
            if not _is_int(row[key]) or row[key] < 0:
                _fail("metrics_count")
        if row["future_messages_consumed"] != 0:
            _fail("metrics_future_message")
        if row["messages_available"] <= 0:
            _fail("metrics_empty_events")
        methods[method] = row

    if (
        methods["naive"]["messages_available"]
        != methods["reliability"]["messages_available"]
    ):
        _fail("metrics_event_count_disagreement")

    delta = _require_keys(
        root["delta"],
        {"position_rmse", "prediction_ade", "identity_switches"},
        "metrics_delta_schema",
    )
    for key in ("position_rmse", "prediction_ade"):
        if not _is_number(delta[key]):
            _fail("metrics_delta_numeric")
        expected = float(methods["reliability"][key]) - float(methods["naive"][key])
        if not math.isclose(float(delta[key]), expected, rel_tol=1e-12, abs_tol=1e-12):
            _fail("metrics_delta_mismatch")
    if not _is_int(delta["identity_switches"]):
        _fail("metrics_delta_count")
    expected_switch_delta = (
        methods["reliability"]["identity_switches"]
        - methods["naive"]["identity_switches"]
    )
    if delta["identity_switches"] != expected_switch_delta:
        _fail("metrics_delta_mismatch")

    gpu = root["gpu"]
    if not isinstance(gpu, dict) or type(gpu.get("available")) is not bool:
        _fail("metrics_gpu_schema")
    if gpu["available"]:
        if set(gpu) != {"available", "devices"}:
            _fail("metrics_gpu_schema")
        devices = gpu["devices"]
        if (
            not isinstance(devices, list)
            or not devices
            or not all(isinstance(row, str) and row for row in devices)
        ):
            _fail("metrics_gpu_devices")
    else:
        if set(gpu) != {"available", "error"} or not isinstance(gpu["error"], str):
            _fail("metrics_gpu_schema")
        devices = []
    if require_a100 and (
        not gpu["available"] or not any("A100" in row for row in devices)
    ):
        _fail("metrics_a100_required")

    return int(methods["naive"]["messages_available"])


def _validate_vector(value: Any) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(_is_number(item) for item in value)
    ):
        _fail("events_vector")


def validate_events(path: Path) -> int:
    count = 0
    try:
        with path.open("rb") as stream:
            for raw_line in stream:
                if len(raw_line) > 64 * 1024:
                    _fail("events_line_too_large")
                if not raw_line.strip():
                    _fail("events_blank_line")
                try:
                    text = raw_line.decode("utf-8")
                except UnicodeError:
                    _fail("events_encoding")
                row = parse_json_text(text)
                row = _require_keys(row, EVENT_KEYS, "events_schema")
                assert_no_sensitive(row)
                if row["source"] not in {"ego", "rsu"}:
                    _fail("events_source")
                if not _is_int(row["truth_id"]) or row["truth_id"] < 0:
                    _fail("events_truth_id")
                if not _is_int(row["event_time"]) or row["event_time"] < 0:
                    _fail("events_event_time")
                if (
                    not _is_int(row["arrival_time"])
                    or row["arrival_time"] < row["event_time"]
                ):
                    _fail("events_causality")
                _validate_vector(row["position"])
                _validate_vector(row["velocity"])
                if (
                    not _is_number(row["reliability"])
                    or not 0.0 <= float(row["reliability"]) <= 1.0
                ):
                    _fail("events_reliability")
                count += 1
    except EvidenceValidationError:
        raise
    except OSError:
        _fail("events_read_failed")
    if count <= 0:
        _fail("events_empty")
    return count


def validate_manifest(
    value: Mapping[str, Any],
    *,
    expected_task_id: str,
    expected_protocol: str,
    expected_script_sha256: str,
    expected_diff_sha256: str,
    expected_config_sha256: str,
    require_a100: bool,
    actual_metrics_sha256: str,
    actual_events_sha256: str,
) -> None:
    root = _require_keys(
        value,
        {
            "schema_version",
            "protocol_id",
            "scientific_claim_allowed",
            "diagnostic_only",
            "clearml_task_id",
            "clearml_task_name",
            "clearml_project",
            "clearml_required",
            "config",
            "config_sha256",
            "script_sha256",
            "standalone_diff_sha256",
            "metrics_sha256",
            "events_sha256",
            "artifact_commit_order",
            "python",
            "platform",
            "pid",
        },
        "manifest_schema",
    )
    if root["schema_version"] != 1:
        _fail("manifest_schema_version")
    if root["protocol_id"] != expected_protocol:
        _fail("manifest_protocol")
    if root["scientific_claim_allowed"] is not False:
        _fail("manifest_scientific_claim")
    if root["diagnostic_only"] is not True:
        _fail("manifest_diagnostic_flag")
    if root["clearml_task_id"] != expected_task_id:
        _fail("manifest_task_id")
    task_name = root["clearml_task_name"]
    if (
        not isinstance(task_name, str)
        or not TASK_NAME_RE.fullmatch(task_name)
        or not task_name.startswith(CLEARML_TASK_PREFIX)
    ):
        _fail("manifest_task_name")
    if root["clearml_project"] != CLEARML_PROJECT:
        _fail("manifest_project")
    if root["clearml_required"] is not True:
        _fail("manifest_clearml_required")
    if root["artifact_commit_order"] != ["metrics", "events", "run_manifest"]:
        _fail("manifest_artifact_order")
    for key in (
        "config_sha256",
        "script_sha256",
        "standalone_diff_sha256",
        "metrics_sha256",
        "events_sha256",
    ):
        if not isinstance(root[key], str) or not SHA256_RE.fullmatch(root[key]):
            _fail("manifest_sha_format")
    if root["script_sha256"] != expected_script_sha256:
        _fail("manifest_script_sha")
    if root["standalone_diff_sha256"] != expected_diff_sha256:
        _fail("manifest_diff_sha")
    if root["config_sha256"] != expected_config_sha256:
        _fail("manifest_config_sha")
    if root["metrics_sha256"] != actual_metrics_sha256:
        _fail("manifest_metrics_sha")
    if root["events_sha256"] != actual_events_sha256:
        _fail("manifest_events_sha")

    config = _require_keys(
        root["config"],
        {
            "protocol_id",
            "seed",
            "steps",
            "packet_loss",
            "max_latency",
            "require_gpu",
            "require_a100",
            "clearml_mode",
        },
        "manifest_config_schema",
    )
    if config["protocol_id"] != expected_protocol:
        _fail("manifest_config_protocol")
    if not _is_int(config["seed"]):
        _fail("manifest_config_seed")
    if not _is_int(config["steps"]) or config["steps"] < 24:
        _fail("manifest_config_steps")
    if (
        not _is_number(config["packet_loss"])
        or not 0.0 <= float(config["packet_loss"]) < 0.8
    ):
        _fail("manifest_config_packet_loss")
    if not _is_int(config["max_latency"]) or config["max_latency"] < 0:
        _fail("manifest_config_latency")
    if type(config["require_gpu"]) is not bool:
        _fail("manifest_config_require_gpu")
    if type(config["require_a100"]) is not bool:
        _fail("manifest_config_require_a100")
    if config["require_a100"] is not require_a100:
        _fail("manifest_config_a100_mismatch")
    if require_a100 and config["require_gpu"] is not True:
        _fail("manifest_config_gpu_mismatch")
    if config["clearml_mode"] != "required":
        _fail("manifest_config_clearml_mode")
    if canonical_sha256(config) != root["config_sha256"]:
        _fail("manifest_config_canonical_sha")
    if not isinstance(root["python"], str) or not root["python"]:
        _fail("manifest_python")
    if not isinstance(root["platform"], str) or not root["platform"]:
        _fail("manifest_platform")
    if not _is_int(root["pid"]) or root["pid"] <= 0:
        _fail("manifest_pid")


def validate_regular_source(
    path: Path, expected_suffix: str, max_bytes: int
) -> os.stat_result:
    if path.suffix != expected_suffix:
        _fail("artifact_extension")
    try:
        source_stat = path.lstat()
    except OSError:
        _fail("artifact_missing")
    if stat.S_ISLNK(source_stat.st_mode):
        _fail("artifact_symlink")
    if not stat.S_ISREG(source_stat.st_mode):
        _fail("artifact_not_regular")
    if source_stat.st_size <= 0 or source_stat.st_size > max_bytes:
        _fail("artifact_size")
    return source_stat


def copy_regular_artifact(
    source: Path, destination: Path, expected_suffix: str, max_bytes: int
) -> None:
    source_stat = validate_regular_source(source, expected_suffix, max_bytes)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError:
        _fail("artifact_open_failed")
    try:
        opened_stat = os.fstat(source_fd)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_dev != source_stat.st_dev
            or opened_stat.st_ino != source_stat.st_ino
            or opened_stat.st_size != source_stat.st_size
        ):
            _fail("artifact_changed")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            copied = 0
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > max_bytes:
                    _fail("artifact_size")
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:
                        _fail("artifact_write_failed")
                    view = view[written:]
            if copied != opened_stat.st_size:
                _fail("artifact_changed")
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
    except EvidenceValidationError:
        raise
    except OSError:
        _fail("artifact_copy_failed")
    finally:
        os.close(source_fd)


@contextlib.contextmanager
def _quiet_third_party() -> Any:
    previous_disable = logging.root.manager.disable
    saved_stdout_fd: int | None = None
    saved_stderr_fd: int | None = None
    null_fd: int | None = None
    logging.disable(logging.CRITICAL)
    try:
        null_fd = os.open(
            os.devnull,
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        )
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            yield
    finally:
        if saved_stdout_fd is not None:
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)
        if saved_stderr_fd is not None:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
        if null_fd is not None:
            os.close(null_fd)
        logging.disable(previous_disable)


def _quiet_call(
    code: str, function: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    try:
        with _quiet_third_party():
            return function(*args, **kwargs)
    except Exception:
        _fail(code)


def _quiet_attempt(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Return ``None`` instead of exposing a third-party download failure."""

    try:
        with _quiet_third_party():
            return function(*args, **kwargs)
    except Exception:
        return None


def _parse_download_url(value: Any, *, trusted_base: bool) -> SplitResult:
    code = "trusted_files_url" if trusted_base else "artifact_url"
    if not isinstance(value, str) or not value:
        _fail(code)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        _fail(code)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
        or port is not None
        and not 1 <= port <= 65535
    ):
        _fail(code)
    if trusted_base and parsed.path not in {"", "/"}:
        _fail(code)
    if not trusted_base and (not parsed.path.startswith("/") or parsed.path == "/"):
        _fail(code)

    path = parsed.path
    if "\\" in path or any(ord(character) < 0x20 for character in path):
        _fail(code)
    for segment in path.split("/"):
        if not segment:
            continue
        if re.search(r"%(?![0-9A-Fa-f]{2})", segment):
            _fail(code)
        decoded = unquote_to_bytes(segment)
        if decoded in {b".", b".."} or b"/" in decoded or b"\\" in decoded:
            _fail(code)
        if any(byte < 0x20 or byte == 0x7F for byte in decoded):
            _fail(code)
    return parsed


def trusted_artifact_download_url(artifact_url: Any, trusted_files_base: Any) -> str:
    """Move only a validated artifact path onto the configured files origin."""

    artifact = _parse_download_url(artifact_url, trusted_base=False)
    trusted = _parse_download_url(trusted_files_base, trusted_base=True)
    return urlunsplit((trusted.scheme.lower(), trusted.netloc, artifact.path, "", ""))


def download_from_trusted_files_server(
    *,
    artifact_url: Any,
    trusted_files_base: Any,
    destination: Path,
    max_bytes: int,
    request_get: Callable[..., Any],
    auth_headers_provider: Callable[[], Mapping[str, str]],
) -> None:
    """Stream one artifact from the configured ClearML files origin."""

    download_url = trusted_artifact_download_url(artifact_url, trusted_files_base)
    trusted = _parse_download_url(trusted_files_base, trusted_base=True)
    requested = _parse_download_url(download_url, trusted_base=False)
    if (requested.scheme.lower(), requested.hostname, requested.port) != (
        trusted.scheme.lower(),
        trusted.hostname,
        trusted.port,
    ):
        _fail("trusted_files_origin")

    try:
        headers = auth_headers_provider()
    except Exception:
        _fail("trusted_files_auth")
    if not isinstance(headers, Mapping) or set(headers) != {"Authorization"}:
        _fail("trusted_files_auth")
    authorization = headers.get("Authorization")
    if (
        not isinstance(authorization, str)
        or not authorization
        or "\r" in authorization
        or "\n" in authorization
    ):
        _fail("trusted_files_auth")

    response: Any = None
    destination_fd: int | None = None
    try:
        response = request_get(
            download_url,
            headers={"Authorization": authorization},
            stream=True,
            allow_redirects=False,
            timeout=(5.0, 60.0),
        )
        if getattr(response, "status_code", None) != 200:
            _fail("trusted_files_status")
        response_headers = getattr(response, "headers", None)
        if not isinstance(response_headers, Mapping):
            _fail("trusted_files_headers")
        content_length = response_headers.get("Content-Length")
        if content_length is not None:
            if (
                not isinstance(content_length, str)
                or not content_length.isascii()
                or not content_length.isdecimal()
            ):
                _fail("trusted_files_content_length")
            declared_bytes = int(content_length)
            if declared_bytes <= 0 or declared_bytes > max_bytes:
                _fail("trusted_files_size")

        destination_fd = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        received = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not isinstance(chunk, bytes):
                _fail("trusted_files_chunk")
            if not chunk:
                continue
            received += len(chunk)
            if received > max_bytes:
                _fail("trusted_files_size")
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    _fail("trusted_files_write")
                view = view[written:]
        if received <= 0:
            _fail("trusted_files_size")
        if content_length is not None and received != int(content_length):
            _fail("trusted_files_content_length_mismatch")
        os.fsync(destination_fd)
    except EvidenceValidationError:
        raise
    except Exception:
        _fail("trusted_files_download_failed")
    finally:
        if destination_fd is not None:
            try:
                os.close(destination_fd)
            except OSError:
                pass
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if sys.exc_info()[0] is not None:
            destination.unlink(missing_ok=True)


def load_trusted_download_adapter() -> (
    tuple[str, Callable[..., Any], Callable[[], Mapping[str, str]]]
):
    """Load configured ClearML files origin, requests, and session auth lazily."""

    def _load() -> tuple[str, Callable[..., Any], Callable[[], Mapping[str, str]]]:
        import requests  # type: ignore
        from clearml.backend_api.session import Session  # type: ignore
        from clearml.backend_interface.base import InterfaceBase  # type: ignore

        session = InterfaceBase._get_default_session()

        def _auth_headers() -> Mapping[str, str]:
            return session.add_auth_headers({})

        return Session.get_files_server_host(), requests.get, _auth_headers

    return _quiet_call("trusted_files_adapter_failed", _load)


def load_clearml_task(task_id: str) -> Any:
    def _load() -> Any:
        from clearml import Task  # type: ignore

        return Task.get_task(task_id=task_id)

    return _quiet_call("clearml_task_lookup_failed", _load)


def snapshot_task(
    task: Any, expected_task_id: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if getattr(task, "id", None) != expected_task_id:
        _fail("task_id_mismatch")
    raw_status = getattr(task, "status", None)
    status = getattr(raw_status, "value", raw_status)
    if str(status).lower() != "published":
        _fail("task_not_published")
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACT_SPECS):
        _fail("task_artifact_set")

    artifact_snapshot: dict[str, Any] = {}
    for name, (_, _, max_bytes) in ARTIFACT_SPECS.items():
        artifact = artifacts[name]
        advertised_size = getattr(artifact, "size", None)
        advertised_hash = getattr(artifact, "hash", None)
        if advertised_size is not None and (
            not _is_int(advertised_size) or not 0 < advertised_size <= max_bytes
        ):
            _fail("artifact_advertised_size")
        if not isinstance(advertised_hash, str) or not SHA256_RE.fullmatch(
            advertised_hash
        ):
            _fail("artifact_advertised_hash")
        artifact_snapshot[name] = {
            "size": advertised_size,
            "sha256": advertised_hash,
        }
    return artifacts, {
        "task_id": expected_task_id,
        "status": "published",
        "artifacts": artifact_snapshot,
    }


def validate_task(task: Any, expected_task_id: str) -> Mapping[str, Any]:
    artifacts, _ = snapshot_task(task, expected_task_id)
    return artifacts


def stage_task_artifacts(
    task: Any,
    stage_directory: Path,
    expected_task_id: str,
    expected_snapshot: Mapping[str, Any] | None = None,
    *,
    trusted_files_base: str | None = None,
    request_get: Callable[..., Any] | None = None,
    auth_headers_provider: Callable[[], Mapping[str, str]] | None = None,
    fallback_loader: Callable[
        [], tuple[str, Callable[..., Any], Callable[[], Mapping[str, str]]]
    ] = load_trusted_download_adapter,
) -> dict[str, Path]:
    artifacts, observed_snapshot = snapshot_task(task, expected_task_id)
    if expected_snapshot is not None and observed_snapshot != expected_snapshot:
        _fail("task_snapshot_changed")
    staged: dict[str, Path] = {}
    for name, (filename, suffix, max_bytes) in ARTIFACT_SPECS.items():
        artifact = artifacts[name]
        metadata = observed_snapshot["artifacts"][name]
        advertised_size = metadata["size"]
        advertised_hash = metadata["sha256"]
        local_copy = _quiet_attempt(
            artifact.get_local_copy,
            extract_archive=False,
            raise_on_error=True,
            force_download=True,
        )
        destination = stage_directory / filename
        if isinstance(local_copy, (str, os.PathLike)):
            copy_regular_artifact(Path(local_copy), destination, suffix, max_bytes)
        elif local_copy is None:
            if (
                trusted_files_base is None
                and request_get is None
                and auth_headers_provider is None
            ):
                adapter = _quiet_call("trusted_files_adapter_failed", fallback_loader)
                if not isinstance(adapter, tuple) or len(adapter) != 3:
                    _fail("trusted_files_adapter_invalid")
                trusted_files_base, request_get, auth_headers_provider = adapter
            if (
                trusted_files_base is None
                or request_get is None
                or auth_headers_provider is None
            ):
                _fail("trusted_files_adapter_incomplete")
            artifact_url = _quiet_call(
                "artifact_url_unavailable", lambda: getattr(artifact, "url", None)
            )
            _quiet_call(
                "trusted_files_download_failed",
                download_from_trusted_files_server,
                artifact_url=artifact_url,
                trusted_files_base=trusted_files_base,
                destination=destination,
                max_bytes=max_bytes,
                request_get=request_get,
                auth_headers_provider=auth_headers_provider,
            )
        else:
            _fail("artifact_download_path")
        if (
            advertised_size is not None
            and destination.stat().st_size != advertised_size
        ):
            _fail("artifact_size_mismatch")
        if file_sha256(destination) != advertised_hash:
            _fail("artifact_hash_mismatch")
        staged[name] = destination
    return staged


def validate_staged_bundle(
    staged: Mapping[str, Path],
    *,
    expected_task_id: str,
    expected_protocol: str,
    expected_script_sha256: str,
    expected_diff_sha256: str,
    expected_config_sha256: str,
    require_a100: bool,
) -> dict[str, Any]:
    if set(staged) != set(ARTIFACT_SPECS):
        _fail("staged_artifact_set")
    for name, path in staged.items():
        _, suffix, max_bytes = ARTIFACT_SPECS[name]
        validate_regular_source(path, suffix, max_bytes)

    metrics_path = staged["metrics"]
    manifest_path = staged["run_manifest"]
    events_path = staged["events"]
    metrics_sha = file_sha256(metrics_path)
    manifest_sha = file_sha256(manifest_path)
    events_sha = file_sha256(events_path)
    metrics = load_json_file(metrics_path)
    manifest = load_json_file(manifest_path)
    expected_event_count = validate_metrics(metrics, expected_protocol, require_a100)
    actual_event_count = validate_events(events_path)
    if actual_event_count != expected_event_count:
        _fail("events_count_mismatch")
    validate_manifest(
        manifest,
        expected_task_id=expected_task_id,
        expected_protocol=expected_protocol,
        expected_script_sha256=expected_script_sha256,
        expected_diff_sha256=expected_diff_sha256,
        expected_config_sha256=expected_config_sha256,
        require_a100=require_a100,
        actual_metrics_sha256=metrics_sha,
        actual_events_sha256=events_sha,
    )

    artifact_receipt: dict[str, Any] = {}
    for name, path in staged.items():
        artifact_receipt[name] = {
            "filename": ARTIFACT_SPECS[name][0],
            "bytes": path.stat().st_size,
            "sha256": {
                "metrics": metrics_sha,
                "run_manifest": manifest_sha,
                "events": events_sha,
            }[name],
        }
    bundle_sha = canonical_sha256(
        {
            "task_id": expected_task_id,
            "protocol_id": expected_protocol,
            "artifacts": artifact_receipt,
        }
    )
    receipt = {
        "schema_version": 1,
        "task_id": expected_task_id,
        "validated": True,
        "protocol_id": expected_protocol,
        "scientific_claim_allowed": False,
        "require_a100": require_a100,
        "script_sha256": expected_script_sha256,
        "standalone_diff_sha256": expected_diff_sha256,
        "config_sha256": expected_config_sha256,
        "event_count": actual_event_count,
        "artifacts": artifact_receipt,
        "bundle_sha256": bundle_sha,
    }
    assert_no_sensitive(receipt)
    return receipt


def write_json_atomic(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except (FileExistsError, FileNotFoundError, OSError):
            _fail("receipt_no_clobber")
        temporary.unlink()
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        _fail("receipt_write_failed")
    return hashlib.sha256(payload).hexdigest()


def publish_staged_directory(stage_directory: Path, output_directory: Path) -> None:
    """Publish validated files without overwriting and place the receipt last."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd: int | None = None
    stage_fd: int | None = None
    output_fd: int | None = None
    output_created = False
    filenames = [
        ARTIFACT_SPECS[name][0] for name in ("metrics", "events", "run_manifest")
    ] + ["receipt.json"]
    try:
        try:
            if stage_directory.parent != output_directory.parent:
                _fail("output_parent_mismatch")
            parent_fd = os.open(output_directory.parent, directory_flags)
            stage_fd = os.open(stage_directory.name, directory_flags, dir_fd=parent_fd)
            os.mkdir(output_directory.name, 0o700, dir_fd=parent_fd)
            output_created = True
            output_fd = os.open(
                output_directory.name,
                directory_flags,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            _fail("output_exists")
        except OSError:
            _fail("output_create_failed")

        assert parent_fd is not None and stage_fd is not None and output_fd is not None
        for filename in filenames:
            os.link(
                filename,
                filename,
                src_dir_fd=stage_fd,
                dst_dir_fd=output_fd,
                follow_symlinks=False,
            )
            os.unlink(filename, dir_fd=stage_fd)
        os.fsync(output_fd)
        stage_stat = os.fstat(stage_fd)
        named_stage_stat = os.stat(
            stage_directory.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (stage_stat.st_dev, stage_stat.st_ino) != (
            named_stage_stat.st_dev,
            named_stage_stat.st_ino,
        ):
            _fail("stage_directory_changed")
        os.rmdir(stage_directory.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except EvidenceValidationError:
        if output_fd is not None:
            for filename in filenames:
                try:
                    os.unlink(filename, dir_fd=output_fd)
                except OSError:
                    pass
        if output_created and parent_fd is not None:
            try:
                os.rmdir(output_directory.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    except Exception:
        if output_fd is not None:
            for filename in filenames:
                try:
                    os.unlink(filename, dir_fd=output_fd)
                except OSError:
                    pass
        if output_created and parent_fd is not None:
            try:
                os.rmdir(output_directory.name, dir_fd=parent_fd)
            except OSError:
                pass
        _fail("output_publish_failed")
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if stage_fd is not None:
            os.close(stage_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def _validate_cli_hash(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("expected 64 lowercase hexadecimal characters")
    return value


def _validate_task_id(value: str) -> str:
    if not TASK_ID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a 32-character lowercase task id")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=_validate_task_id)
    parser.add_argument(
        "--expected-protocol", default=PROTOCOL_ID, choices=[PROTOCOL_ID]
    )
    parser.add_argument(
        "--expected-script-sha256", required=True, type=_validate_cli_hash
    )
    parser.add_argument(
        "--expected-diff-sha256", required=True, type=_validate_cli_hash
    )
    parser.add_argument(
        "--expected-config-sha256", required=True, type=_validate_cli_hash
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-a100", action="store_true")
    return parser.parse_args()


def run(
    args: argparse.Namespace, task_loader: Callable[[str], Any] = load_clearml_task
) -> dict[str, Any]:
    requested_output = Path(args.output).expanduser().absolute()
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    output_directory = (
        requested_output.parent.resolve(strict=True) / requested_output.name
    )
    if os.path.lexists(output_directory):
        _fail("output_exists")
    stage_directory = Path(
        tempfile.mkdtemp(prefix=".canary-evidence.", dir=output_directory.parent)
    )
    try:
        task = task_loader(args.task_id)
        _, initial_snapshot = snapshot_task(task, args.task_id)
        staged = stage_task_artifacts(
            task,
            stage_directory,
            args.task_id,
            expected_snapshot=initial_snapshot,
        )
        receipt = validate_staged_bundle(
            staged,
            expected_task_id=args.task_id,
            expected_protocol=args.expected_protocol,
            expected_script_sha256=args.expected_script_sha256,
            expected_diff_sha256=args.expected_diff_sha256,
            expected_config_sha256=args.expected_config_sha256,
            require_a100=args.require_a100,
        )
        refreshed_task = task_loader(args.task_id)
        _, refreshed_snapshot = snapshot_task(refreshed_task, args.task_id)
        if refreshed_snapshot != initial_snapshot:
            _fail("task_snapshot_changed")
        receipt["task_status"] = "published"
        assert_no_sensitive(receipt)
        receipt_sha = write_json_atomic(stage_directory / "receipt.json", receipt)
        if os.path.lexists(output_directory):
            _fail("output_exists")
        publish_staged_directory(stage_directory, output_directory)
        return {
            "task_id": args.task_id,
            "validated": True,
            "receipt_sha256": receipt_sha,
        }
    except Exception:
        if stage_directory.exists():
            shutil.rmtree(stage_directory, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except EvidenceValidationError:
        print("download_canary_evidence_failed", file=sys.stderr)
        return 2
    except Exception:
        print("download_canary_evidence_failed", file=sys.stderr)
        return 3
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
