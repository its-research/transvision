#!/usr/bin/env python3
"""Publish a sealed, byte-reproducible Source-C to Source-D evidence chain."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

try:
    from allegroai import Task
except ImportError:
    try:
        from clearml import Task
    except ImportError:
        Task = None  # type: ignore[assignment]


SOURCE_C_TASK_ID = "95e72da24d464ab08d117dedabd6652e"
SOURCE_C_PARENT_TASK_ID = "6525107e60ae4104a2800731d74ecd4e"
DEFAULT_PROJECT = "ResilientV2X/Training"
FILES_SERVER_URI = "http://10.100.34.118:8081"
SOURCE_C_ENTRY_POINT = "clearml_5090_bootstrap.py"
PRODUCER_ENTRY_POINT = "clearml_formal_source_d_evidence.py"
SOURCE_C_SHA256 = "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
SOURCE_D_SHA256 = "e7a9ab0fb05339223cf2c18c52eb72652c311733bf96d1058aa7a769096cf8c3"
EQUIVALENCE_ARTIFACT_SHA256 = (
    "1156fe53f2fe924f91c1c6b50b6b21090d98cd2d74840d6d6f5a358316420433"
)
BUILDER_SOURCE_SHA256 = (
    "904fafd08d710b03b62bc57140121f2a546ada8fa2fb8763e6db8a44b9f8f7e7"
)
TRANSFORMATION_ID = "source-c-to-source-d-explicit-seed-evidence-v2"
TRAINING_OVERLAY_PROTOCOL_SEED = 20_250_218
DECLARED_REPLACEMENT_COUNT = 22
UNCHANGED_SEGMENT_COUNT = 23
PORTABLE_RUNNER_LOAD_MARKER = "_validate_rtx5090_runtime_contract_multi_gpu"
EXPECTED_PORTABLE_RUNNER_LOAD_MARKER_COUNT = 2
LEGACY_RUNNER_LOAD_TARGET_ANCHOR = """    if args.predecessor_task_id == task_id:
        raise RuntimeError("an experiment task cannot be its own predecessor")

    runner = _load_source_training_runner(source_root)
    dataset_root, env = _prepare_experiment_environment(
"""
STAGING_ANCHOR_NAMES = (
    "seal_evidence_security_imports",
    "declare_controlled_evidence_contract",
    "declare_controlled_evidence_receipts",
    "stage_and_verify_controlled_evidence",
    "stage_after_controlled_runner",
    "upload_only_sealed_controlled_evidence",
)
STAGING_REQUIRED_FRAGMENTS = (
    ("class ControlledEvidenceMemberReceipt(NamedTuple):", 1),
    ("class ControlledEvidenceStage(NamedTuple):", 1),
    ("CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024", 1),
    ("CONTROLLED_EVIDENCE_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024", 1),
    ("CONTROLLED_EVIDENCE_MAX_COMPRESSION_RATIO = 128.0", 1),
    ("CONTROLLED_EVIDENCE_REQUIRED_FILENAMES = (", 1),
    ("def _stage_controlled_baseline_evidence(", 1),
    ("def _verify_controlled_evidence_stage(", 1),
    ("def _read_uploaded_artifact_bytes(", 1),
    ("def _snapshot_uploaded_evidence_zip(", 1),
    ("def _preflight_uploaded_evidence_zip(", 1),
    ("def _verify_uploaded_evidence_zip(", 1),
    ("def _verify_uploaded_controlled_evidence(", 1),
    ("def _reload_controlled_evidence_task(", 1),
    ("def _upload_controlled_baseline_artifacts(", 1),
    ("evidence_stage = _stage_controlled_baseline_evidence(", 1),
    ("if len(ordered) != 38 or sum(item.size_bytes for item in ordered) > (", 1),
    ("len(stage.members) != 38\n", 1),
    ("len(names) != 38\n", 1),
    ("or entries_on_disk != 38\n", 1),
    ("or entry_count != 38\n", 1),
    ('archive.read(4) == b"PK\\x06\\x07"', 1),
    (
        "central_directory_size > CONTROLLED_EVIDENCE_MAX_CENTRAL_DIRECTORY_BYTES",
        1,
    ),
    ('run["resolved_config_sha256"] = copied["resolved_config.py"].sha256', 1),
    ('resealed_plan["content_sha256"] = _producer_content_sha256(resealed_plan)', 1),
    ('("controlled_baseline_evidence", str(evidence_stage.root))', 1),
    ("extract_archive=False", 2),
    ("force_download=True", 2),
    ('original_name = getattr(member, "orig_filename", None)', 1),
    ("with tempfile.TemporaryFile(", 1),
    ('reloader = getattr(task, "_reload", None)', 1),
    ('setattr(task, "_data", snapshot)', 1),
    ("failed to flush controlled baseline artifact uploads", 1),
    ("st_nlink != 1", 5),
    ("_stat_identity(", 39),
    ("changed while being staged", 3),
    ("changed during readback", 1),
    ("staging root changed during verification", 1),
    ("cannot disable automatic archive extraction", 1),
)
STAGING_FORBIDDEN_FRAGMENTS = (
    '("controlled_baseline_evidence", str(work_dir))',
    "metrics_getter()",
    'reloader = getattr(task, "reload", None)',
    "os.fdopen(os.dup(descriptor)",
)

SOURCE_C_SNAPSHOT_ARTIFACT = "formal_source_c_snapshot"
SOURCE_D_SCRIPT_ARTIFACT = "formal_source_d_script"
EQUIVALENCE_ARTIFACT = "formal_source_d_equivalence"
RECEIPT_ARTIFACT = "formal_source_d_evidence_receipt"
PUBLICATION_ORDER = (
    SOURCE_C_SNAPSHOT_ARTIFACT,
    SOURCE_D_SCRIPT_ARTIFACT,
    EQUIVALENCE_ARTIFACT,
    RECEIPT_ARTIFACT,
)

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

# ``generate_standalone_source`` replaces this one exact anchor. The resulting
# ClearML script therefore carries the byte-pinned builder and never imports it
# from a repository checkout on a no-repo worker.
_EMBEDDED_BUILDER_SOURCE_B64 = ""  # __SOURCE_D_BUILDER_BYTES_V2__


class SourceDEvidenceError(RuntimeError):
    """Raised when any provenance or publication invariant fails closed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    parser.add_argument(
        "--emit-standalone",
        type=Path,
        help="write a new standalone wrapper with the pinned builder embedded",
    )
    return parser


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _content_sha256(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = str(value.get("seal_sha256") or "")
    if not _is_lower_hex(observed, 64):
        raise SourceDEvidenceError(f"{context} seal is not a lowercase SHA-256")
    if _sealed(value)["seal_sha256"] != observed:
        raise SourceDEvidenceError(f"{context} seal SHA-256 mismatch")
    return observed


def _require_equivalence_artifact_hash(
    value: Mapping[str, object], *, context: str
) -> str:
    observed = str(value.get("artifact_sha256") or "")
    if not _is_lower_hex(observed, 64):
        raise SourceDEvidenceError(
            f"{context} artifact SHA-256 is not lowercase hexadecimal"
        )
    unhashed = dict(value)
    unhashed.pop("artifact_sha256", None)
    if _content_sha256(unhashed) != observed:
        raise SourceDEvidenceError(f"{context} artifact SHA-256 mismatch")
    return observed


def _frozen_mapping(
    value: Mapping[str, object], *, context: str
) -> tuple[dict[str, object], str]:
    try:
        canonical = _canonical_json(value)
        frozen = json.loads(canonical)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SourceDEvidenceError(f"{context} is not canonical JSON") from error
    if not isinstance(frozen, dict):
        raise SourceDEvidenceError(f"{context} is not a JSON object")
    return frozen, _sha256_text(canonical)


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _clearml_id(value: object, *, context: str) -> str:
    if type(value) is not str or not _is_lower_hex(value, 32):
        raise SourceDEvidenceError(f"{context} must be a lowercase 32-hex ClearML ID")
    return value


def _task_parent(task: object) -> str:
    value = getattr(task, "parent", None)
    if value is None or (type(value) is str and value == ""):
        value = getattr(getattr(task, "data", None), "parent", None)
    if value is None:
        return ""
    if type(value) is not str:
        raise SourceDEvidenceError(
            "task parent must be a string, empty string, or None"
        )
    return value


def _reload(task: object, *, context: str) -> None:
    """Perform one uncached backend read and surface every transport failure.

    ClearML's public ``Task.reload()`` intentionally swallows backend exceptions
    and returns ``None`` on both success and failure. Evidence publication must
    therefore call the underlying backend read with its cache-skip flag disabled,
    then install the returned server snapshot explicitly.
    """

    expected_task_id = _clearml_id(
        getattr(task, "id", ""), context=f"{context} local task"
    )
    if bool(getattr(task, "_offline_mode", False)):
        raise SourceDEvidenceError(f"{context} cannot use offline reload")
    reloader = getattr(task, "_reload", None)
    if not callable(reloader):
        raise SourceDEvidenceError(f"{context} cannot be server-reloaded")
    has_skip_flag = hasattr(task, "_reload_skip_flag")
    previous_skip_flag = getattr(task, "_reload_skip_flag", None)
    try:
        if has_skip_flag:
            setattr(task, "_reload_skip_flag", False)
        result = reloader()
    except Exception as error:
        raise SourceDEvidenceError(f"{context} server reload failed") from error
    finally:
        if has_skip_flag:
            setattr(task, "_reload_skip_flag", previous_skip_flag)
    if result is None or isinstance(result, (bool, int, float, str, bytes, bytearray)):
        raise SourceDEvidenceError(f"{context} server reload returned no snapshot")
    snapshot_task_id = _clearml_id(
        getattr(result, "id", ""), context=f"{context} server snapshot"
    )
    if snapshot_task_id != expected_task_id:
        raise SourceDEvidenceError(f"{context} server snapshot identity mismatch")
    try:
        setattr(task, "_data", result)
    except Exception as error:
        raise SourceDEvidenceError(
            f"{context} server snapshot cannot be installed"
        ) from error


def _status(task: object) -> str:
    return str(getattr(task, "status", "") or "").lower()


def _wait_for_source_c(
    task: object,
    *,
    deadline: float,
    poll_seconds: float,
    monotonic_clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    while True:
        _reload(task, context="Source-C teacher")
        status = _status(task)
        if status == "completed":
            return
        if status in FAILED_STATUSES:
            raise SourceDEvidenceError(f"Source-C teacher ended as {status!r}")
        if status not in WAITING_STATUSES:
            raise SourceDEvidenceError(
                f"Source-C teacher has unexpected status {status!r}"
            )
        if monotonic_clock() >= deadline:
            raise TimeoutError("timed out waiting for completed Source-C teacher")
        sleeper(poll_seconds)


def _source_c_script(task: object) -> tuple[str, dict[str, object]]:
    script = getattr(getattr(task, "data", None), "script", None)
    if script is None:
        raise SourceDEvidenceError("Source-C teacher has no script metadata")
    repository = getattr(script, "repository", None)
    working_dir = getattr(script, "working_dir", None)
    entry_point = getattr(script, "entry_point", None)
    source = getattr(script, "diff", None)
    if repository != "":
        raise SourceDEvidenceError("Source-C teacher is not a standalone task")
    if working_dir != ".":
        raise SourceDEvidenceError("Source-C teacher working directory drifted")
    if entry_point != SOURCE_C_ENTRY_POINT:
        raise SourceDEvidenceError("Source-C teacher entry point drifted")
    if type(source) is not str or not source:
        raise SourceDEvidenceError("Source-C teacher script diff is not raw text")
    observed_sha256 = _sha256_text(source)
    if observed_sha256 != SOURCE_C_SHA256:
        raise SourceDEvidenceError("Source-C teacher script SHA-256 mismatch")
    return source, {
        "repository": repository,
        "working_dir": working_dir,
        "entry_point": entry_point,
        "sha256": observed_sha256,
        "size_bytes": len(source.encode("utf-8")),
        "line_count": len(source.splitlines()),
    }


def _runtime_producer_source() -> str:
    try:
        source = Path(__file__).read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise SourceDEvidenceError(
            "cannot read the running Source-D evidence producer"
        ) from error
    if not source:
        raise SourceDEvidenceError("running Source-D evidence producer is empty")
    return source


def _producer_script(task: object) -> tuple[str, dict[str, object]]:
    script = getattr(getattr(task, "data", None), "script", None)
    if script is None:
        raise SourceDEvidenceError("output task has no producer script metadata")
    repository = getattr(script, "repository", None)
    working_dir = getattr(script, "working_dir", None)
    entry_point = getattr(script, "entry_point", None)
    source = getattr(script, "diff", None)
    if repository != "":
        raise SourceDEvidenceError("output producer is not a standalone task")
    if working_dir != ".":
        raise SourceDEvidenceError("output producer working directory drifted")
    if entry_point != PRODUCER_ENTRY_POINT:
        raise SourceDEvidenceError("output producer entry point drifted")
    if type(source) is not str or not source:
        raise SourceDEvidenceError("output producer script diff is not raw text")
    if source != _runtime_producer_source():
        raise SourceDEvidenceError(
            "output producer script bytes differ from the running wrapper"
        )
    return source, {
        "repository": repository,
        "working_dir": working_dir,
        "entry_point": entry_point,
        "sha256": _sha256_text(source),
        "size_bytes": len(source.encode("utf-8")),
        "line_count": len(source.splitlines()),
    }


def _validate_publication_bindings(
    *,
    source_task: object,
    output_task: object,
    output_task_id: str,
    expected_source_c: str,
    expected_source_script_metadata: Mapping[str, object],
    expected_producer_source: str,
    expected_producer_script_metadata: Mapping[str, object],
    phase: str,
) -> None:
    """Re-read and bind both provenance endpoints around every write."""

    _reload(source_task, context=f"Source-C teacher {phase} snapshot")
    if _status(source_task) != "completed":
        raise SourceDEvidenceError(f"Source-C teacher {phase} status drifted")
    if (
        _clearml_id(
            getattr(source_task, "id", ""),
            context=f"Source-C teacher {phase} snapshot",
        )
        != SOURCE_C_TASK_ID
    ):
        raise SourceDEvidenceError(f"Source-C teacher {phase} identity drifted")
    if _task_parent(source_task) != SOURCE_C_PARENT_TASK_ID:
        raise SourceDEvidenceError(f"Source-C teacher {phase} parent drifted")
    source_c, source_script_metadata = _source_c_script(source_task)
    if (
        source_c != expected_source_c
        or source_script_metadata != expected_source_script_metadata
    ):
        raise SourceDEvidenceError(f"Source-C teacher {phase} snapshot drifted")

    _reload(output_task, context=f"Source-D evidence output task {phase} snapshot")
    if (
        _clearml_id(getattr(output_task, "id", ""), context="output task")
        != output_task_id
    ):
        raise SourceDEvidenceError(
            f"Source-D evidence output identity drifted during {phase}"
        )
    if _task_parent(output_task) != SOURCE_C_TASK_ID:
        raise SourceDEvidenceError(
            f"Source-D evidence output parent drifted during {phase}"
        )
    producer_source, producer_script_metadata = _producer_script(output_task)
    if (
        producer_source != expected_producer_source
        or producer_script_metadata != expected_producer_script_metadata
    ):
        raise SourceDEvidenceError(
            f"output producer script snapshot drifted during {phase}"
        )


def _builder_source_bytes(builder_path: Path | None = None) -> bytes:
    if _EMBEDDED_BUILDER_SOURCE_B64:
        try:
            source = base64.b64decode(
                _EMBEDDED_BUILDER_SOURCE_B64.encode("ascii"), validate=True
            )
        except (ValueError, UnicodeError) as error:
            raise SourceDEvidenceError(
                "embedded Source-D builder is not valid base64"
            ) from error
    else:
        path = builder_path or Path(__file__).with_name("formal_source_d_seed.py")
        try:
            source = path.read_bytes()
        except OSError as error:
            raise SourceDEvidenceError("cannot read the Source-D builder") from error
    if _sha256_bytes(source) != BUILDER_SOURCE_SHA256:
        raise SourceDEvidenceError("Source-D builder byte SHA-256 mismatch")
    try:
        source.decode("utf-8")
    except UnicodeError as error:
        raise SourceDEvidenceError("Source-D builder is not UTF-8") from error
    return source


def _load_builder_module(builder_path: Path | None = None) -> ModuleType:
    source = _builder_source_bytes(builder_path)
    module_name = "_resilient_v2x_formal_source_d_seed_pinned"
    module = ModuleType(module_name)
    module.__file__ = "<pinned-formal-source-d-seed.py>"
    sys.modules[module_name] = module
    try:
        exec(compile(source.decode("utf-8"), module.__file__, "exec"), module.__dict__)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise SourceDEvidenceError(
            "cannot execute the pinned Source-D builder"
        ) from error
    for name in ("build_source_d", "verify_source_d"):
        if not callable(getattr(module, name, None)):
            raise SourceDEvidenceError(f"pinned Source-D builder lacks {name}")
    return module


def generate_standalone_source(
    *,
    builder_path: Path | None = None,
    wrapper_path: Path | None = None,
) -> str:
    """Return this wrapper with the exact pinned builder bytes embedded."""

    builder_source = _builder_source_bytes(builder_path)
    path = wrapper_path or Path(__file__)
    try:
        wrapper_source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SourceDEvidenceError(
            "cannot read the Source-D evidence wrapper"
        ) from error
    assignment = " ".join(("_EMBEDDED_BUILDER_SOURCE_B64", "=", '""'))
    marker_comment = " ".join(("#", "__SOURCE_D_BUILDER_BYTES_V2__"))
    anchor = assignment + "  " + marker_comment
    if wrapper_source.count(anchor) != 1:
        raise SourceDEvidenceError("standalone builder anchor count mismatch")
    encoded = base64.b64encode(builder_source).decode("ascii")
    embedded_assignment = " ".join(("_EMBEDDED_BUILDER_SOURCE_B64", "=", repr(encoded)))
    replacement = embedded_assignment + "  " + marker_comment
    standalone = wrapper_source.replace(anchor, replacement, 1)
    try:
        compile(standalone, "<clearml-formal-source-d-evidence>", "exec")
    except SyntaxError as error:
        raise SourceDEvidenceError(
            "generated Source-D evidence standalone script does not compile"
        ) from error
    if encoded not in standalone:
        raise SourceDEvidenceError("generated standalone source lost builder bytes")
    return standalone


def _validate_build(
    builder: ModuleType,
    source_c: str,
) -> tuple[object, dict[str, object]]:
    try:
        build = builder.build_source_d(source_c)
        verified = builder.verify_source_d(
            source_c, build.source_d_text, build.artifact
        )
    except Exception as error:
        raise SourceDEvidenceError(
            "Source-D build or replay verification failed"
        ) from error
    if verified != build:
        raise SourceDEvidenceError("Source-D verification result drifted")
    source_d_text = getattr(build, "source_d_text", None)
    source_d_sha256 = getattr(build, "source_d_sha256", None)
    build_artifact = getattr(build, "artifact", None)
    if type(source_d_text) is not str:
        raise SourceDEvidenceError("Source-D script is not raw text")
    if source_d_sha256 != SOURCE_D_SHA256:
        raise SourceDEvidenceError("Source-D script SHA-256 mismatch")
    if _sha256_text(source_d_text) != SOURCE_D_SHA256:
        raise SourceDEvidenceError("Source-D script content drifted")
    marker_count = source_d_text.count(PORTABLE_RUNNER_LOAD_MARKER)
    if marker_count != EXPECTED_PORTABLE_RUNNER_LOAD_MARKER_COUNT:
        raise SourceDEvidenceError("Source-D portable runner-load marker count drifted")
    if source_d_text.count(LEGACY_RUNNER_LOAD_TARGET_ANCHOR) != 0:
        raise SourceDEvidenceError(
            "Source-D retains the legacy runner-load target anchor"
        )
    builder_staging_anchors = getattr(builder, "_STAGING_ANCHOR_NAMES", None)
    if (
        not isinstance(builder_staging_anchors, tuple)
        or builder_staging_anchors != STAGING_ANCHOR_NAMES
    ):
        raise SourceDEvidenceError("Source-D builder staging anchors drifted")
    for fragment, expected_count in STAGING_REQUIRED_FRAGMENTS:
        if source_d_text.count(fragment) != expected_count:
            raise SourceDEvidenceError(
                f"Source-D staging invariant count drifted for {fragment!r}"
            )
    for fragment in STAGING_FORBIDDEN_FRAGMENTS:
        if fragment in source_d_text:
            raise SourceDEvidenceError(
                f"Source-D retains unsafe evidence upload {fragment!r}"
            )
    if not isinstance(build_artifact, Mapping):
        raise SourceDEvidenceError("Source-D equivalence artifact is invalid")
    equivalence_artifact = dict(build_artifact)
    observed_equivalence_hash = _require_equivalence_artifact_hash(
        equivalence_artifact, context="Source-D equivalence build"
    )
    if observed_equivalence_hash != EQUIVALENCE_ARTIFACT_SHA256:
        raise SourceDEvidenceError("Source-D equivalence seal mismatch")
    if equivalence_artifact.get("transformation_id") != TRANSFORMATION_ID:
        raise SourceDEvidenceError("Source-D transformation identity mismatch")
    diff = equivalence_artifact.get("diff")
    equivalence = equivalence_artifact.get("equivalence")
    seed_contract = equivalence_artifact.get("seed_contract")
    if not isinstance(diff, list) or len(diff) != DECLARED_REPLACEMENT_COUNT:
        raise SourceDEvidenceError("Source-D replacement count mismatch")
    if any(
        not isinstance(item, Mapping)
        or type(item.get("expected_count")) is not int
        or item.get("expected_count") != 1
        or type(item.get("observed_count")) is not int
        or item.get("observed_count") != 1
        for item in diff
    ):
        raise SourceDEvidenceError("Source-D replacement evidence drifted")
    diff_names = [item.get("name") for item in diff if isinstance(item, Mapping)]
    if any(diff_names.count(name) != 1 for name in STAGING_ANCHOR_NAMES):
        raise SourceDEvidenceError("Source-D staging replacement evidence drifted")
    if (
        not isinstance(equivalence, Mapping)
        or equivalence.get("only_declared_anchor_replacements") is not True
        or type(equivalence.get("declared_replacement_count")) is not int
        or equivalence.get("declared_replacement_count") != DECLARED_REPLACEMENT_COUNT
        or type(equivalence.get("unchanged_segment_count")) is not int
        or equivalence.get("unchanged_segment_count") != UNCHANGED_SEGMENT_COUNT
        or equivalence.get("source_c_replay_sha256") != SOURCE_C_SHA256
        or equivalence.get("source_d_replay_sha256") != SOURCE_D_SHA256
        or equivalence.get("source_d_compiles") is not True
    ):
        raise SourceDEvidenceError("Source-D equivalence evidence drifted")
    if (
        not isinstance(seed_contract, Mapping)
        or type(seed_contract.get("training_overlay_protocol_seed")) is not int
        or seed_contract.get("training_overlay_protocol_seed")
        != TRAINING_OVERLAY_PROTOCOL_SEED
        or type(seed_contract.get("default_training_seed")) is not int
        or seed_contract.get("default_training_seed") != TRAINING_OVERLAY_PROTOCOL_SEED
    ):
        raise SourceDEvidenceError("Source-D overlay seed contract drifted")
    return build, equivalence_artifact


def _artifact_mapping(task: object, name: str) -> Mapping[str, object]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise SourceDEvidenceError(f"output task lacks artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise SourceDEvidenceError(f"artifact {name!r} cannot be read")
    value = getter()
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, (str, Path)):
        raise SourceDEvidenceError(f"artifact {name!r} is not a JSON object")
    path = Path(value)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceDEvidenceError(
            f"artifact {name!r} cannot be read as JSON"
        ) from error
    if not isinstance(document, Mapping):
        raise SourceDEvidenceError(f"artifact {name!r} is not a JSON object")
    return document


def _has_artifact(task: object, name: str) -> bool:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        raise SourceDEvidenceError("output task artifacts are unavailable")
    return name in artifacts


def _artifact_self_hash(
    name: str,
    value: Mapping[str, object],
    *,
    artifact_hashes: Mapping[str, object] | None = None,
) -> str:
    if name in {SOURCE_C_SNAPSHOT_ARTIFACT, SOURCE_D_SCRIPT_ARTIFACT}:
        return _require_seal(value, context=name)
    if name == EQUIVALENCE_ARTIFACT:
        return _require_equivalence_artifact_hash(value, context=name)
    if name == RECEIPT_ARTIFACT:
        seal = _require_seal(value, context=name)
        if artifact_hashes is None or _canonical_json(
            value.get("artifact_hashes")
        ) != _canonical_json(dict(artifact_hashes)):
            raise SourceDEvidenceError(
                "Source-D evidence receipt artifact hashes mismatch"
            )
        return seal
    raise SourceDEvidenceError(f"unknown publication artifact {name!r}")


def _read_verified_artifact(
    task: object,
    name: str,
    *,
    expected: Mapping[str, object],
    expected_canonical_sha256: str,
    artifact_hashes: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], str]:
    current, current_canonical_sha256 = _frozen_mapping(
        _artifact_mapping(task, name), context=f"artifact {name!r} readback"
    )
    current_self_hash = _artifact_self_hash(
        name, current, artifact_hashes=artifact_hashes
    )
    # Canonical JSON distinguishes values that Python equality aliases, such as
    # integer 1 and boolean true. Mapping equality is forbidden for evidence.
    if current_canonical_sha256 != expected_canonical_sha256:
        raise SourceDEvidenceError(f"artifact {name!r} canonical bytes drifted")
    expected_frozen, expected_sha256 = _frozen_mapping(
        expected, context=f"expected artifact {name!r}"
    )
    if expected_sha256 != expected_canonical_sha256:
        raise SourceDEvidenceError(f"expected artifact {name!r} mutated")
    if _canonical_json(current) != _canonical_json(expected_frozen):
        raise SourceDEvidenceError(f"artifact {name!r} canonical bytes drifted")
    return current, current_self_hash


def _publish_one(
    task: object,
    name: str,
    *,
    expected: Mapping[str, object],
    expected_canonical_sha256: str,
    uploader: object,
    flusher: object,
    validate_bindings: Callable[[], None],
    artifact_hashes: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], str]:
    validate_bindings()
    if _has_artifact(task, name):
        return _read_verified_artifact(
            task,
            name,
            expected=expected,
            expected_canonical_sha256=expected_canonical_sha256,
            artifact_hashes=artifact_hashes,
        )
    upload_payload, upload_sha256 = _frozen_mapping(
        expected, context=f"upload artifact {name!r}"
    )
    if upload_sha256 != expected_canonical_sha256:
        raise SourceDEvidenceError(f"expected artifact {name!r} mutated")
    uploaded = uploader(
        name,
        artifact_object=upload_payload,
        wait_on_upload=True,
    )
    if uploaded is not True:
        raise SourceDEvidenceError(f"failed to upload artifact {name!r}")
    flushed = flusher(wait_for_uploads=True)
    if flushed is not None and flushed is not True:
        raise SourceDEvidenceError(f"failed to flush artifact {name!r}")
    validate_bindings()
    if not _has_artifact(task, name):
        raise SourceDEvidenceError(f"artifact {name!r} is absent after upload")
    return _read_verified_artifact(
        task,
        name,
        expected=expected,
        expected_canonical_sha256=expected_canonical_sha256,
        artifact_hashes=artifact_hashes,
    )


def _publish_sequence(
    task: object,
    payloads: Mapping[str, Mapping[str, object]],
    *,
    validate_bindings: Callable[[], None],
) -> dict[str, object]:
    if set(payloads) != set(PUBLICATION_ORDER):
        raise SourceDEvidenceError("Source-D evidence payload names mismatch")
    frozen: dict[str, dict[str, object]] = {}
    frozen_sha256: dict[str, str] = {}
    for name in PUBLICATION_ORDER:
        value = payloads[name]
        if not isinstance(value, Mapping):
            raise SourceDEvidenceError(f"expected artifact {name!r} is invalid")
        frozen[name], frozen_sha256[name] = _frozen_mapping(
            value, context=f"expected artifact {name!r}"
        )

    expected_artifact_hashes = {
        name: _artifact_self_hash(name, frozen[name]) for name in PUBLICATION_ORDER[:-1]
    }
    _artifact_self_hash(
        RECEIPT_ARTIFACT,
        frozen[RECEIPT_ARTIFACT],
        artifact_hashes=expected_artifact_hashes,
    )

    validate_bindings()
    missing_seen = False
    for name in PUBLICATION_ORDER:
        present = _has_artifact(task, name)
        if missing_seen and present:
            raise SourceDEvidenceError(
                "existing Source-D evidence artifacts violate publication order"
            )
        missing_seen = missing_seen or not present

    uploader = getattr(task, "upload_artifact", None)
    flusher = getattr(task, "flush", None)
    if not callable(uploader) or not callable(flusher):
        raise SourceDEvidenceError("output task cannot atomically publish artifacts")

    for name in PUBLICATION_ORDER[:-1]:
        _publish_one(
            task,
            name,
            expected=frozen[name],
            expected_canonical_sha256=frozen_sha256[name],
            uploader=uploader,
            flusher=flusher,
            validate_bindings=validate_bindings,
        )

    # Derive the receipt cross-links from freshly read, independently verified
    # server artifacts. Never trust the prebuilt expected hash fields alone.
    validate_bindings()
    actual_artifact_hashes: dict[str, str] = {}
    for name in PUBLICATION_ORDER[:-1]:
        _, actual_artifact_hashes[name] = _read_verified_artifact(
            task,
            name,
            expected=frozen[name],
            expected_canonical_sha256=frozen_sha256[name],
        )
    receipt_base, _ = _frozen_mapping(
        frozen[RECEIPT_ARTIFACT], context="receipt derivation"
    )
    receipt_base["artifact_hashes"] = dict(actual_artifact_hashes)
    derived_receipt = _sealed(receipt_base)
    derived_receipt, derived_receipt_sha256 = _frozen_mapping(
        derived_receipt, context="derived Source-D evidence receipt"
    )
    if derived_receipt_sha256 != frozen_sha256[RECEIPT_ARTIFACT]:
        raise SourceDEvidenceError(
            "prebuilt receipt differs from verified artifact readbacks"
        )

    _publish_one(
        task,
        RECEIPT_ARTIFACT,
        expected=derived_receipt,
        expected_canonical_sha256=derived_receipt_sha256,
        uploader=uploader,
        flusher=flusher,
        validate_bindings=validate_bindings,
        artifact_hashes=actual_artifact_hashes,
    )

    validate_bindings()
    final_artifact_hashes: dict[str, str] = {}
    for name in PUBLICATION_ORDER[:-1]:
        _, final_artifact_hashes[name] = _read_verified_artifact(
            task,
            name,
            expected=frozen[name],
            expected_canonical_sha256=frozen_sha256[name],
        )
    final_receipt, _ = _read_verified_artifact(
        task,
        RECEIPT_ARTIFACT,
        expected=derived_receipt,
        expected_canonical_sha256=derived_receipt_sha256,
        artifact_hashes=final_artifact_hashes,
    )
    return final_receipt


def _build_payloads(
    *,
    source_task: object,
    output_task: object,
    source_c: str,
    script_metadata: Mapping[str, object],
    producer_script_metadata: Mapping[str, object],
    source_d_build: object,
    equivalence_artifact: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    source_task_id = _clearml_id(
        getattr(source_task, "id", ""), context="Source-C teacher"
    )
    output_task_id = _clearml_id(getattr(output_task, "id", ""), context="output task")
    source_c_snapshot = _sealed(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_c_snapshot",
            "complete": True,
            "source_c_task_id": source_task_id,
            "source_c_task_status": "completed",
            "script": dict(script_metadata),
            "script_diff": source_c,
        }
    )
    source_d_script = _sealed(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_d_script",
            "complete": True,
            "transformation_id": TRANSFORMATION_ID,
            "source_c_sha256": SOURCE_C_SHA256,
            "source_d_sha256": SOURCE_D_SHA256,
            "size_bytes": len(source_d_build.source_d_text.encode("utf-8")),
            "line_count": len(source_d_build.source_d_text.splitlines()),
            "script": source_d_build.source_d_text,
        }
    )
    equivalence = dict(equivalence_artifact)
    artifact_hashes = {
        SOURCE_C_SNAPSHOT_ARTIFACT: source_c_snapshot["seal_sha256"],
        SOURCE_D_SCRIPT_ARTIFACT: source_d_script["seal_sha256"],
        EQUIVALENCE_ARTIFACT: EQUIVALENCE_ARTIFACT_SHA256,
    }
    receipt = _sealed(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_d_evidence_receipt",
            "complete": True,
            "publication_order": list(PUBLICATION_ORDER),
            "provenance": {
                "source_c_task_id": source_task_id,
                "source_c_task_status": "completed",
                "source_c_task_parent": _task_parent(source_task),
                "source_c_entry_point": SOURCE_C_ENTRY_POINT,
                "source_c_sha256": SOURCE_C_SHA256,
                "output_task_id": output_task_id,
                "output_parent_task_id": SOURCE_C_TASK_ID,
                "builder_source_sha256": BUILDER_SOURCE_SHA256,
                "transformation_id": TRANSFORMATION_ID,
                "producer_entry_point": PRODUCER_ENTRY_POINT,
                "producer_script_sha256": producer_script_metadata["sha256"],
            },
            "transformation": {
                "source_d_sha256": SOURCE_D_SHA256,
                "equivalence_artifact_sha256": EQUIVALENCE_ARTIFACT_SHA256,
                "declared_replacement_count": DECLARED_REPLACEMENT_COUNT,
                "unchanged_segment_count": UNCHANGED_SEGMENT_COUNT,
                "only_declared_anchor_replacements": True,
                "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
                "training_seed_cli": "--training-seed",
                "portable_runner_load_marker": PORTABLE_RUNNER_LOAD_MARKER,
                "portable_runner_load_marker_count": (
                    EXPECTED_PORTABLE_RUNNER_LOAD_MARKER_COUNT
                ),
                "legacy_runner_load_target_anchor_count": 0,
            },
            "artifact_hashes": artifact_hashes,
        }
    )
    return {
        SOURCE_C_SNAPSHOT_ARTIFACT: source_c_snapshot,
        SOURCE_D_SCRIPT_ARTIFACT: source_d_script,
        EQUIVALENCE_ARTIFACT: equivalence,
        RECEIPT_ARTIFACT: receipt,
    }


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    builder_path: Path | None = None,
) -> dict[str, object]:
    if (
        type(args.poll_seconds) not in {int, float}
        or not math.isfinite(float(args.poll_seconds))
        or args.poll_seconds <= 0
        or type(args.timeout_hours) not in {int, float}
        or not math.isfinite(float(args.timeout_hours))
        or args.timeout_hours <= 0
    ):
        raise ValueError("poll interval and timeout must be finite and positive")
    if task_class is None:
        raise SourceDEvidenceError("ClearML is unavailable")
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise SourceDEvidenceError("ClearML Task class cannot resolve Source-C")
    source_task = getter(task_id=SOURCE_C_TASK_ID)
    if (
        _clearml_id(getattr(source_task, "id", ""), context="Source-C teacher")
        != SOURCE_C_TASK_ID
    ):
        raise SourceDEvidenceError("Source-C teacher identity mismatch")
    source_task_parent = _task_parent(source_task)
    if source_task_parent != SOURCE_C_PARENT_TASK_ID:
        raise SourceDEvidenceError("Source-C teacher parent mismatch")
    if output_task is None:
        current = getattr(task_class, "current_task", None)
        output_task = current() if callable(current) else None
    if output_task is None:
        raise SourceDEvidenceError("Source-D evidence requires a current output task")
    output_task_id = _clearml_id(getattr(output_task, "id", ""), context="output task")
    if output_task_id == SOURCE_C_TASK_ID or output_task is source_task:
        raise SourceDEvidenceError("Source-D evidence output aliases Source-C teacher")
    if _task_parent(output_task) != SOURCE_C_TASK_ID:
        raise SourceDEvidenceError("Source-D evidence output parent mismatch")
    producer_source, producer_script_metadata = _producer_script(output_task)

    deadline = monotonic_clock() + float(args.timeout_hours) * 3600.0
    _wait_for_source_c(
        source_task,
        deadline=deadline,
        poll_seconds=float(args.poll_seconds),
        monotonic_clock=monotonic_clock,
        sleeper=sleeper,
    )
    source_c, script_metadata = _source_c_script(source_task)
    builder = _load_builder_module(builder_path)
    source_d_build, equivalence_artifact = _validate_build(builder, source_c)

    # Check the completed source and output-parent bindings after the potentially
    # long build and then around every persistent publication below.
    def validate_bindings(*, phase: str = "publication") -> None:
        _validate_publication_bindings(
            source_task=source_task,
            output_task=output_task,
            output_task_id=output_task_id,
            expected_source_c=source_c,
            expected_source_script_metadata=script_metadata,
            expected_producer_source=producer_source,
            expected_producer_script_metadata=producer_script_metadata,
            phase=phase,
        )

    validate_bindings(phase="final")
    payloads = _build_payloads(
        source_task=source_task,
        output_task=output_task,
        source_c=source_c,
        script_metadata=script_metadata,
        producer_script_metadata=producer_script_metadata,
        source_d_build=source_d_build,
        equivalence_artifact=equivalence_artifact,
    )
    return _publish_sequence(
        output_task,
        payloads,
        validate_bindings=validate_bindings,
    )


def _write_new(path: Path, content: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.emit_standalone is not None:
        standalone = generate_standalone_source()
        _write_new(args.emit_standalone, standalone)
        print(
            json.dumps(
                {
                    "standalone_sha256": _sha256_text(standalone),
                    "builder_source_sha256": BUILDER_SOURCE_SHA256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if Task is None:
        raise SourceDEvidenceError("ClearML is unavailable")
    task = Task.init(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X formal Source-D transformation evidence",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
    )
    receipt = run(args, task_class=Task, output_task=task)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
